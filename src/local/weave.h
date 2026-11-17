#pragma once
// The weave (KYTY_WEAVE, docs/WEAVE.md): queue 0's draws translated on worker threads and woven back, in order, into
// the GPU thread's stream.
//
// The reader (Kyty.WeaveRead) copies each queue-0 graphics submission as GuestGpu::Enqueue queues it, walks the copy
// ahead of the GPU thread with the real register handlers and cuts it into parcels of up to KYTY_WEAVE_CALLS table
// calls, each ending early at a packet only the GPU thread can run. Workers (Kyty.Weave1 on) make a parcel's draws with
// the table path's worker instantiation, recording their Vulkan calls apart, and leave a draw they cannot make as a
// hole. At a parcel's first draw the GPU thread opens it (its start registers and everything it read checked), then at
// each strand's first draw does the strand's needs, tests it from its lists, does its actions and splices a part of it
// into its own stream; holes and the draws after a failing test it makes itself, today's way, between the spliced
// parts.
//
// What workers read of the GPU thread's structures is safe by safepoints: the GPU thread holds every worker outside a
// draw (Hold) before a structural write of what they read, notes each pair, set and draw state it frees (NoteFreed),
// and keeps a retired buffer or image slot until no worker that might have seen it is still in that draw (HoldsSlot). A
// worker never waits for the GPU thread inside a draw.
//
// Off (the default) none of this runs: each check on a path the GPU thread takes is one relaxed load of a plain global.
// The weave runs only while KYTY_SPECULATE is 0 (one parallel path at a time).

#include <atomic>
#include <cstdint>
#include <span>
#include <vector>

extern "C" {
extern volatile std::atomic_uint32_t kyty_local_weave_mode;    // KYTY_WEAVE: 0 off, 1 on
extern volatile std::atomic_uint32_t kyty_local_weave_workers; // KYTY_WEAVE_WORKERS: 0 auto, 1-8
extern volatile std::atomic_uint32_t kyty_local_weave_calls;   // KYTY_WEAVE_CALLS: 16-1024, default 128
}

namespace Weave {

struct Registers;
struct RegisterLog;
struct Parcel;

inline constexpr uint32_t MaxWorkers     = 8;
inline constexpr uint32_t StretchDraws   = 64; // draw packets a stretch (its needs grouped; a strand breaks there too)
inline constexpr uint32_t ShortestParcel = 16; // the fewest draws a parcel is published with
inline constexpr uint32_t WaitBudgetUs   = 20; // the GPU thread's wait at an open, by the worker's pace
// Each worker's upload ring: its size, its records, its alignment; the slab a parcel takes of it, and what an empty
// slab keeps (so the ring's tail never meets its head while a slab is live).
inline constexpr uint64_t RingBytes = uint64_t {16} << 20u, SlabBytes = uint64_t {1} << 20u, RingAlign = 256;
inline constexpr uint32_t RingRecords   = 1024;
inline constexpr uint64_t SlabKeepBytes = 256;
inline constexpr uint32_t ParcelSlots = 512, FeedEntries = 1024, FreedEntries = 65536, CopyRecords = 16384;

// --- Start-up and run time ---

// InitializePerformanceSwitches, after mode 4's gates: the stand-downs (each one line; the run stays in mode 0, and
// kyty_local_weave_mode is 0 after one) and the worker count.
void ApplyStartGates();
// What an unset KYTY_RENDER_CORE stands for (KeepRenderCoreAlone's fallback; the start line names it): "auto" while
// the weave stays up after its gates (the GPU thread's core kept for it), else null.
[[nodiscard]] const char* RenderCoreFallback() noexcept;
// The resolved worker count (0 when stood down).
[[nodiscard]] uint32_t Workers() noexcept;
// The switch on and not stood down (the start-up result: a stand-down sets the switch to 0).
[[nodiscard]] inline bool Enabled() noexcept {
	return kyty_local_weave_mode.load(std::memory_order_relaxed) != 0;
}
// Workers may be inside a draw: the holds, the free-log notes and the slot holds act. Set by Start and Resume, cleared
// by Pause and StopAll (weave.cpp).
extern std::atomic<bool> g_active;

[[nodiscard]] inline bool Active() noexcept {
	return g_active.load(std::memory_order_relaxed);
}
// The GPU thread, between queue 0's submissions: Pause with KYTY_SPECULATE not 0, at the top of ThreadRun's loop right
// after a live change set it (every parcel dropped, the reader stops cutting, every worker outside a draw and idle,
// every slot and slab released, Active() false); Resume at a queue-0 graphics submission's first slice with it 0 again
// (Active(), the reader out of step until the GPU thread posts its registers). Both no-ops before Start.
void Pause();
void Resume();
// The GPU thread, before Scout::Stop (the reader and the workers read its write clock): the reader and the workers
// stopped and joined, Active() false, every held slot let go.
void StopAll();

enum class Thread : uint8_t { None, Reader, Worker };
// The calling thread's part in the weave (weave.cpp's threads set theirs). Read only after Enabled() or Active().
extern thread_local Thread t_thread;

[[nodiscard]] inline bool OffGpuThread() noexcept {
	return Active() && t_thread != Thread::None;
}

// --- Safepoints ---

// Hold: the GPU thread, around a structural write of what workers read (pairs, variants, sets, draw states and their
// maps): it waits until no worker is inside a draw and keeps them out until it ends; a line once a wait passes 1 s,
// naming the worker. A no-op unless Active(). While Active() it does EXIT_IF(t_thread != Thread::None): a hold taken on
// a weave thread would wait for itself, and a worker never reaches a structural write. Holds nest.
class Hold {
public:
	Hold() noexcept {
		if (Active()) [[unlikely]]
			Open();
	}
	~Hold() {
		if (m_open) [[unlikely]]
			Close();
	}
	Hold(const Hold&)            = delete;
	Hold& operator=(const Hold&) = delete;

private:
	void Open() noexcept;
	void Close() noexcept;

	bool m_open = false;
};
// DrawScope: worker `worker`, around one draw's table path. Its begin waits while the GPU thread holds; nothing inside
// it waits for the GPU thread.
class DrawScope {
public:
	explicit DrawScope(uint32_t worker) noexcept;
	~DrawScope();
	DrawScope(const DrawScope&)            = delete;
	DrawScope& operator=(const DrawScope&) = delete;

private:
	uint32_t m_worker = 0;
};

// The free log: the GPU thread notes a pair, set or draw state before freeing it (a no-op unless Active()).
void NoteFreed(const void* object) noexcept;
// Entries written so far: a worker reads it as its parcel begins.
[[nodiscard]] uint64_t FreedClock() noexcept;
// The log read on from entry `scanned` by the GPU thread (which alone writes it), each entry once: an object found in
// `sorted` (ascending) appended to `hits`, `scanned` moved to the log's end. 0 none found, 1 one was, 2 the log turned
// over past `scanned` (not every free since is in it).
[[nodiscard]] uint32_t FreedHits(uint64_t& scanned, std::span<const void* const> sorted,
                                 std::vector<const void*>& hits);

// Deferred erase (EraseRetired's predicate in both caches): true keeps the slot until every worker that was inside a
// draw when it was first asked about here has left that draw. For an image `generation` is its id's (a buffer's ask
// passes none). Called only while Active().
enum class Slot : uint8_t { Buffer, Image };
[[nodiscard]] bool HoldsSlot(Slot kind, uint32_t index, uint32_t generation = 0) noexcept;
// An image retiring, by its id (index and generation): asked about by HoldsSlot, its erase not yet known done. A
// retired buffer is out of every lookup a worker makes (by address), but a retired image keeps its id alive until its
// erase, and a set or a draw state still names it: a worker inside a draw asks here before it reads an image by such
// an id, and leaves a retiring one to the GPU thread. ImagesRetiring: whether any is (one load).
[[nodiscard]] bool ImagesRetiring() noexcept;
[[nodiscard]] bool ImageRetiring(uint32_t index, uint32_t generation) noexcept;

// Why a worker made a draw a hole (counted by group, never per reason).
enum class Miss : uint8_t { Packet, Pair, Shaders, Variant, State, Targets, Tables, Set, Buffers, Ring, Memory, Other };

// --- Threads, the feed and the parcels (weave.cpp) ---

// What weave.cpp's threads run, given once at Start: the reader's loop (returns once Stopping()), one parcel on worker
// `worker` (the parcel Taken; its state is Done after), and the release of what a parcel holds besides its slot (its
// slab, its apart buffers, its commands' copy), run once on whichever thread frees the slot.
struct Bodies {
	void (*read)()                                = nullptr;
	void (*make)(uint32_t worker, Parcel& parcel) = nullptr;
	void (*release)(Parcel& parcel)               = nullptr;
};
// The GPU thread, at queue 0's first draw packet with the weave enabled: the reader and Workers() workers started, the
// start line printed, Active(); once (later calls return the first one's answer). False: none run (stood down, the
// scout not running, KYTY_SPECULATE not 0, or a thread could not start).
[[nodiscard]] bool Start(const Bodies& bodies);
// StopAll has begun: every loop returns.
[[nodiscard]] bool Stopping() noexcept;

// A queue-0 graphics submission for the reader, as GuestGpu::Enqueue feeds the scout.
struct Feed {
	uint64_t        sequence = 0;       // the scout's sequence number
	const uint32_t* commands = nullptr; // the guest's commands
	uint32_t        dwords   = 0;
	bool            constant = false; // it has constant engine commands
	bool            reset    = false; // the command processor is reset before it (its own or a flip preparation's)
};
// GuestGpu::Enqueue (the queue mutex held): the feed queued and the reader woken; false, nothing queued, when the ring
// of FeedEntries is full (the reader then finds a gap in the sequence numbers).
bool PushFeed(const Feed& feed) noexcept;
// The reader: the oldest feed not taken (false: none), and its take.
[[nodiscard]] bool PeekFeed(Feed& feed) noexcept;
void               TakeFeed() noexcept;
// The reader's wake count: it moves with each feed, each WakeReader and StopAll. The reader reads it before it looks
// for work; WaitReader(seen) returns once it has moved from `seen` (at once if it has), spinning first while
// SpinGate::Open(false), then blocking. WakeReader: any thread (the GPU thread posting its registers, sending the
// reader out of step).
[[nodiscard]] uint32_t ReaderWakes() noexcept;
void                   WaitReader(uint32_t seen) noexcept;
void                   WakeReader() noexcept;

// The reader: the slot of the next parcel (serial Published()), Filling, its fields the reader's to fill; it waits
// while that slot is not Free (a worker or the GPU thread still holds the parcel there). Null once Stopping() or while
// not Active().
[[nodiscard]] Parcel* BeginParcel() noexcept;
// The reader: the parcel cut whole, Cut and counted in Published(), a worker woken; or one begun and not published (too
// short, or the read stopped), Free again with its serial unused.
void PublishParcel(Parcel& parcel) noexcept;
void AbandonParcel(Parcel& parcel) noexcept;
// Parcels published so far: serials 0 to Published() - 1, each cut once.
[[nodiscard]] uint64_t Published() noexcept;
// The slot of parcel `serial` (its serial % ParcelSlots); the slot's `serial` names the parcel it holds now.
[[nodiscard]] Parcel& ParcelAt(uint64_t serial) noexcept;
// The GPU thread, done with a parcel (opened and closed, or gone past): dropped; a Cut one no worker took becomes
// Skipped; a Skipped or Done one is released now (Bodies::release, then Free), a Taken one by its worker as it stops.
void PassParcel(Parcel& parcel) noexcept;

// --- The table path's two instantiations (render.h, table-xpr.inc) ---

// The GPU thread's, today's code; and a worker's, whose writes to shared state are skipped, kept private, or noted for
// the GPU thread's turn.
struct Ordered {
	static constexpr bool worker = false;
};
struct Worker {
	static constexpr bool worker = true;
};

// --- Linear copies ---

// The GPU thread's last linear copy (BufferCache::CopyBuffer writes it while Active(): path 1 the CPU copy, 2 the GPU
// path; cleared before each top-level DISPATCH_DIRECT the weave's loop checks, read after it), compared with the
// reader's record of that copy.
struct CopyDone {
	uint64_t dst = 0, src = 0, bytes = 0;
	uint8_t  path = 0;
};
inline thread_local CopyDone t_copy_done;

} // namespace Weave

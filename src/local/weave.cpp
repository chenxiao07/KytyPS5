// The weave's switches, start-up gates, threads, parcels, safepoints and free log (weave.h).
//
// Threads. The reader (Kyty.WeaveRead) and the workers (Kyty.Weave1 on) are plain threads the GPU thread starts at
// queue 0's first draw packet (Start) and joins before the scout stops (StopAll). What they run is the command
// processor's and the executor's (Bodies); this file only hands work between them. None is pinned: on Windows they
// take the process's default CPU set, which leaves out the GPU thread's core while KYTY_RENDER_CORE keeps it; on
// Linux each leaves the GPU thread's CPUs it was born on (AvoidCpuList).
//
// Handing off. Submissions reach the reader through a ring of FeedEntries (PushFeed, under GuestGpu's queue mutex);
// parcels reach the workers through a ring of ParcelSlots slots, numbered by serial. A worker takes serials in
// turn (one ticket each), so each parcel has one worker that may take it. The reader and the workers spin for work
// only while the spin gate is open (spin-gate.h), and block otherwise. The gate stops only the spinning: a worker
// takes its parcel shut or open, since a parcel left is made by the GPU thread, on the frame's path.
//
// A slot: Free, then the reader's (Filling) until published (Cut); a worker takes it (Taken) and finishes it (Done);
// one the GPU thread passes before its worker took it becomes Skipped. The GPU thread passes each parcel
// once (PassParcel); whoever comes last, it or the worker, releases the slot (Releasing, then Free).
//
// Safepoints. Each worker has a gate it marks busy around one draw (DrawScope). A Hold asks every worker to stay
// out of its next draw and waits until none is inside one, so the GPU thread may then change what workers read
// (a pair rebuilt, a map grown). A worker never waits for the GPU thread inside a draw; the GPU thread's wait is
// one draw's at most, and a line tells once it has lasted a second (a freeze would sit there).

#include "weave.h"

#include "common/assert.h"
#include "graphics/guest_gpu/scout.h"
#include "live-counters.h"
#include "local-platform.h"
#include "spin-gate.h"
#include "weave-parcel.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

extern "C" {
volatile std::atomic_uint32_t kyty_local_weave_mode {0};
volatile std::atomic_uint32_t kyty_local_weave_workers {0};
volatile std::atomic_uint32_t kyty_local_weave_calls {128};

// (Read only: the weave stands down while mode 4 runs, and needs the scout.)
extern volatile std::atomic_uint32_t kyty_local_speculate_mode; // KYTY_SPECULATE
extern volatile std::atomic_uint32_t kyty_local_scout_mode;     // KYTY_SCOUT (graphics/guest_gpu/scout.cpp)
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
extern volatile std::atomic_uint32_t kyty_local_vulkan_recording_mode; // KYTY_VULKAN_RECORDING
extern volatile std::atomic_uint32_t kyty_local_table_xpr_mode;        // KYTY_TABLE_XPR
extern volatile std::atomic_uint32_t kyty_local_native_xpr_mode;       // KYTY_NATIVE_XPR
#endif
}

namespace Weave {

std::atomic<bool>   g_active {false};
thread_local Thread t_thread = Thread::None;

namespace {

// The spins before a reader or worker with nothing to do blocks (while the spin gate is open).
constexpr uint32_t IdleSpins = 2048;

[[nodiscard]] int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// A short pause in a spin (x64: PAUSE; elsewhere a yield).
void SpinPause() {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

// A wait's report: a line once it has lasted a second, and again at each doubling, so a wait that never ends (a
// freeze) says so where it is. The clock is read once every 4096 spins, never while a wait is short.
class Stall {
public:
	explicit Stall(const char* what) noexcept: m_what(what) {}
	template <typename Line>
	void Spin(Line&& line) {
		SpinPause();
		if ((++m_spins & 4095u) != 0) return;
		const auto now = NowNs();
		if (m_start == 0) {
			m_start = now;
			return;
		}
		if (const auto waited_ms = static_cast<uint64_t>((now - m_start) / 1'000'000); waited_ms >= m_next_ms) {
			m_next_ms *= 2;
			line(static_cast<uint32_t>(waited_ms), m_what);
			std::fflush(stdout);
		}
	}

private:
	const char* m_what    = nullptr;
	uint32_t    m_spins   = 0;
	int64_t     m_start   = 0;
	uint64_t    m_next_ms = 1000;
};

// A worker's gate: inside a draw (DrawScope), and the slot clock as that draw began (HoldsSlot).
struct alignas(64) Gate {
	std::atomic<uint32_t> busy {0};
	std::atomic<uint64_t> seen {0};
};

std::array<Gate, MaxWorkers> g_gates;
// Holds open (each thread's outermost Hold counts one): a worker stays out of its next draw while this isn't 0.
std::atomic<uint32_t> g_requested {0};
// Moves at each slot first asked about (HoldsSlot): a draw that began before a slot's stamp may hold that slot.
std::atomic<uint64_t> g_slot_clock {1};
// The image slots retiring (ImageRetiring), each an entry naming the image's id (RetiringId; 0: free), taken lowest
// first below `high`. The id, not the index: a slot erased goes back to the slot vector's free list, whose next insert
// takes it, so a new image may hold the index while the entry of the one before is still here. An entry is written
// before its slot's stamp, so a draw that read the clock after the stamp finds it (seq_cst, as the gates); it goes at
// the next image ask after its slot was let go, by when that slot is erased (the texture cache's EraseRetired runs one
// at a time, and erases a slot let go before it asks about the next).
constexpr uint32_t                                 RetiringEntries = 64;
std::array<std::atomic<uint64_t>, RetiringEntries> g_retiring {};
std::atomic<uint32_t>                              g_retiring_used {0}, g_retiring_high {0};
// The free log's entries written (NoteFreed; FreedClock, FreedHits read it). Resume moves it past a whole log.
std::atomic<uint64_t> g_freed_seq {0};
thread_local uint32_t t_holds       = 0;     // (the calling thread's Holds open, nested)
uint32_t              g_workers     = 0;     // (ApplyStartGates' count)
int                   g_answer      = -1;    // (the GPU thread's: Start's answer, -1 not decided)
bool                  g_told_paused = false; // (Pause's line, once a run)

// What runs once Start made it: never freed (a late PushFeed from a guest thread may still read it after StopAll).
struct Runtime {
	Bodies   bodies;
	uint32_t workers = 0;

	std::unique_ptr<Parcel[]>                   parcels; // ParcelSlots
	std::unique_ptr<std::atomic<bool>[]>        passed;  // each slot's: the GPU thread is done with its parcel
	std::unique_ptr<std::atomic<const void*>[]> freed;   // the free log, FreedEntries (entry i at i % FreedEntries)

	// The feed: written under GuestGpu's queue mutex (tail), read by the reader (head).
	std::array<Feed, FeedEntries> feeds;
	alignas(64) std::atomic<uint64_t> feed_tail {0};
	alignas(64) std::atomic<uint64_t> feed_head {0};

	// The reader's wake: `wakes` moves with each feed, WakeReader, pause, resume and stop; the reader says when it
	// blocks on it (`reader_sleeping`), so a waker makes the system call only then. `parked`: the pause it has stopped
	// for (WaitReader, while `epoch` is odd).
	alignas(64) std::atomic<uint32_t> wakes {0};
	std::atomic<uint32_t> reader_sleeping {0};
	std::atomic<uint32_t> parked {0};

	// The parcels: published so far (the reader's), the next serial a worker takes, and the workers' wake (`signal`
	// moves with each parcel published and the stop; `sleepers` say they block on it).
	alignas(64) std::atomic<uint64_t> published {0};
	alignas(64) std::atomic<uint64_t> take {0};
	std::atomic<uint32_t> signal {0}, sleepers {0};
	// Slots released (the reader blocks on it while the slot it needs is not Free; `release_waiter` says so).
	std::atomic<uint32_t> releases {0}, release_waiter {0};

	// Odd while paused (Pause, Resume: the GPU thread's).
	std::atomic<uint32_t> epoch {0};
	std::atomic<bool>     stopping {false};
	bool                  stopped = false; // (the GPU thread's)

	std::thread                         reader;
	std::array<std::thread, MaxWorkers> threads;

	// The slots kept retired (HoldsSlot): kind << 32 | index, its stamp, and an image's retiring entry (NoEntry: a
	// buffer). The entries of the image slots let go at the last ask.
	// EraseRetired may run off the GPU thread (a cache's deferred operation), so under a lock.
	static constexpr uint32_t NoEntry = UINT32_MAX;
	struct Held {
		uint64_t key = 0, stamp = 0;
		uint32_t entry = NoEntry;
	};
	std::mutex            holds_mutex;
	std::vector<Held>     holds;
	std::vector<uint32_t> released;
};

std::atomic<Runtime*> g_runtime {nullptr};

[[nodiscard]] Runtime* Started() noexcept {
	return g_runtime.load(std::memory_order_acquire);
}

[[nodiscard]] bool Paused(const Runtime& r) noexcept {
	return (r.epoch.load(std::memory_order_seq_cst) & 1u) != 0;
}

[[nodiscard]] uint32_t SlotIndex(const Runtime& r, const Parcel& parcel) noexcept {
	return static_cast<uint32_t>(&parcel - r.parcels.get());
}

// Every thread of the weave woken from its block (a pause, the stop).
void WakeAll(Runtime& r) {
	r.wakes.fetch_add(1, std::memory_order_seq_cst);
	r.wakes.notify_all();
	r.signal.fetch_add(1, std::memory_order_seq_cst);
	r.signal.notify_all();
	r.releases.fetch_add(1, std::memory_order_seq_cst);
	r.releases.notify_all();
}

// Slot `index` released by the thread that moved it to Releasing: what its parcel holds, then the slot Free, and a
// reader that waits for it told.
void ReleaseSlot(Runtime& r, uint32_t index) {
	auto& parcel = r.parcels[index];
	r.bodies.release(parcel);
	parcel.state.store(Parcel::Free, std::memory_order_seq_cst);
	r.releases.fetch_add(1, std::memory_order_seq_cst);
	if (r.release_waiter.load(std::memory_order_seq_cst) != 0) r.releases.notify_all();
}

// The GPU thread done with slot `index`'s parcel: its worker stops at its next draw; a parcel no worker took is
// Skipped and released now, as is a Done one; a Taken one its worker releases as it stops (it reads `passed` after
// its Done, and this reads the state after `passed`: one of the two sees the other's store). A Free or Filling slot
// holds nothing published and is left alone.
void PassSlot(Runtime& r, uint32_t index) {
	auto&    parcel = r.parcels[index];
	uint32_t state  = parcel.state.load(std::memory_order_seq_cst);
	if (state == Parcel::Free || state == Parcel::Filling) return;
	r.passed[index].store(true, std::memory_order_seq_cst);
	parcel.dropped.store(true, std::memory_order_seq_cst);
	state = parcel.state.load(std::memory_order_seq_cst);
	if (state == Parcel::Cut && parcel.state.compare_exchange_strong(state, Parcel::Skipped, std::memory_order_acq_rel))
		state = Parcel::Skipped;
	if ((state == Parcel::Skipped || state == Parcel::Done) &&
	    parcel.state.compare_exchange_strong(state, Parcel::Releasing, std::memory_order_acq_rel))
		ReleaseSlot(r, index);
}

// An image's id as a retiring entry holds it: generation << 32 | index, plus 1 (never 0, a free entry's).
[[nodiscard]] constexpr uint64_t RetiringId(uint32_t index, uint32_t generation) noexcept {
	return ((uint64_t {generation} << 32u) | index) + 1;
}

// An image slot's retiring entry taken (HoldsSlot, before its stamp), and given back once its slot is erased;
// Runtime::NoEntry while every entry is taken.
[[nodiscard]] uint32_t Retire(uint64_t id) noexcept {
	for (uint32_t entry = 0; entry < RetiringEntries; ++entry)
		if (g_retiring[entry].load(std::memory_order_relaxed) == 0) {
			g_retiring_used.fetch_add(1, std::memory_order_seq_cst);
			g_retiring[entry].store(id, std::memory_order_seq_cst);
			if (entry >= g_retiring_high.load(std::memory_order_relaxed))
				g_retiring_high.store(entry + 1, std::memory_order_seq_cst);
			return entry;
		}
	return Runtime::NoEntry;
}
void Unretire(uint32_t entry) noexcept {
	g_retiring[entry].store(0, std::memory_order_release);
	auto high = g_retiring_high.load(std::memory_order_relaxed);
	while (high != 0 && g_retiring[high - 1].load(std::memory_order_relaxed) == 0)
		--high;
	g_retiring_high.store(high, std::memory_order_release);
	g_retiring_used.fetch_sub(1, std::memory_order_release);
}

// The slots kept retired let go: no worker is inside a draw (paused, stopped).
void ForgetSlotHolds(Runtime& r) {
	std::lock_guard lock(r.holds_mutex);
	for (const auto& held: r.holds)
		if (held.entry != Runtime::NoEntry) Unretire(held.entry);
	for (const auto entry: r.released)
		Unretire(entry);
	r.holds.clear();
	r.released.clear();
}

// A thread the GPU thread made is on its CPUs (Linux: a new thread takes its creator's, one CPU while
// KYTY_RENDER_CORE keeps a core, else KYTY_RENDER_CPUS): moved to the process's other CPUs, as mode 4's workers and
// the helpers are (AvoidCpuList). Windows gives a new thread the process's CPUs, and its default CPU set leaves the
// kept core out: nothing to do.
void LeaveRenderCpus() {
#if defined(__linux__)
	LocalPlatform::AvoidCpuList(std::getenv("KYTY_RENDER_CPUS"));
#endif
}

// A worker, holding serial `ticket`: until that parcel is published (false: the stop).
[[nodiscard]] bool WaitPublished(Runtime& r, uint64_t ticket) {
	for (uint32_t spin = 0; r.published.load(std::memory_order_acquire) <= ticket;) {
		if (r.stopping.load(std::memory_order_acquire)) return false;
		if (spin < IdleSpins && SpinGate::Open(false)) {
			++spin;
			SpinPause();
			continue;
		}
		r.sleepers.fetch_add(1, std::memory_order_seq_cst);
		const auto signal = r.signal.load(std::memory_order_seq_cst);
		if (r.published.load(std::memory_order_seq_cst) <= ticket && !r.stopping.load(std::memory_order_seq_cst))
			r.signal.wait(signal, std::memory_order_seq_cst);
		r.sleepers.fetch_sub(1, std::memory_order_seq_cst);
		spin = 0;
	}
	return true;
}

void ReaderMain(Runtime& r) {
	LocalPlatform::SetThreadName("Kyty.WeaveRead");
	t_thread = Thread::Reader;
	LeaveRenderCpus();
	r.bodies.read();
}

// Worker `worker` (Kyty.Weave<worker + 1>): the parcels in serial order, one ticket at a time.
void WorkerMain(Runtime& r, uint32_t worker) {
	char name[16];
	std::snprintf(name, sizeof(name), "Kyty.Weave%u", worker + 1);
	LocalPlatform::SetThreadName(name);
	t_thread = Thread::Worker;
	LeaveRenderCpus();
	for (;;) {
		const uint64_t ticket = r.take.fetch_add(1, std::memory_order_relaxed);
		if (!WaitPublished(r, ticket)) return;
		const auto index  = static_cast<uint32_t>(ticket % ParcelSlots);
		auto&      parcel = r.parcels[index];
		// (Passed and its slot made over to a later parcel while this worker was away: that one is its own worker's.)
		if (parcel.state.load(std::memory_order_acquire) != Parcel::Cut || parcel.serial != ticket) continue;
		// (Before the take, so a GPU thread that sees it Taken sees them: its wait reads the worker's pace.)
		parcel.progress.store(0, std::memory_order_relaxed);
		parcel.taken_ns.store(NowNs(), std::memory_order_relaxed);
		uint32_t expected = Parcel::Cut;
		if (!parcel.state.compare_exchange_strong(expected, Parcel::Taken, std::memory_order_acq_rel)) continue;
		// Made while the spin gate is shut too: the gate stops the spinning (WaitPublished blocks), and a parcel left
		// would be the GPU thread's to make, on the frame's path. Counted, so a window shows the gate shut.
		if (!SpinGate::Open(false)) LiveCounters::Add(LiveCounters::WeaveParcelsGateSkipped);
		r.bodies.make(worker, parcel);
		parcel.state.store(Parcel::Done, std::memory_order_seq_cst);
		if (r.passed[index].load(std::memory_order_seq_cst)) {
			expected = Parcel::Done;
			if (parcel.state.compare_exchange_strong(expected, Parcel::Releasing, std::memory_order_acq_rel))
				ReleaseSlot(r, index);
		}
	}
}

// The GPU thread's core as the start line names it, from the switches KeepRenderCoreAlone read (the CPU it kept is
// in the Performance switches line).
[[nodiscard]] std::string CoreText() {
	if (const char* core = std::getenv("KYTY_RENDER_CORE"); core != nullptr && *core != '\0')
		return std::string("KYTY_RENDER_CORE=") + core;
	if (const char* cpus = std::getenv("KYTY_RENDER_CPUS"); cpus != nullptr && *cpus != '\0')
		return std::string("placed by KYTY_RENDER_CPUS=") + cpus;
	if (const char* fallback = RenderCoreFallback(); fallback != nullptr)
		return std::string("KYTY_RENDER_CORE=") + fallback;
	return "not kept";
}

// The weave's threads stopped and joined (a worker stops at its next draw: every parcel is dropped), every slot that
// holds anything released, every retired slot let go, and Active() false. The GPU thread's, once.
void Shut(Runtime& r) {
	r.stopped = true;
	r.stopping.store(true, std::memory_order_seq_cst);
	for (uint32_t index = 0; index < ParcelSlots; ++index)
		r.parcels[index].dropped.store(true, std::memory_order_seq_cst);
	WakeAll(r);
	if (r.reader.joinable()) r.reader.join();
	for (auto& thread: r.threads)
		if (thread.joinable()) thread.join();
	// (No thread of the weave runs now.)
	for (uint32_t index = 0; index < ParcelSlots; ++index)
		if (r.parcels[index].state.load(std::memory_order_acquire) != Parcel::Free) ReleaseSlot(r, index);
	ForgetSlotHolds(r);
	g_active.store(false, std::memory_order_seq_cst);
}

void StandDown(const char* line) {
	kyty_local_weave_mode.store(0, std::memory_order_relaxed);
	g_workers = 0;
	std::printf("%s This run stays in mode 0.\n", line);
	std::fflush(stdout);
}

} // namespace

// --- Start-up ---

void ApplyStartGates() {
	if (kyty_local_weave_mode.load(std::memory_order_relaxed) == 0) return;
#if !defined(KYTY_LOCAL_VULKAN_RECORDING)
	// (Non-x86-64 builds, and the test targets that compile the emulator's sources without it.)
	StandDown("Weave: needs the Vulkan recording (this build has none).");
#else
	// One parallel path at a time: mode 4 wins, as the weave may not write its switch.
	if (kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0) {
		StandDown("Weave: KYTY_WEAVE=1 stands down: KYTY_SPECULATE=4 is on (one parallel path at a time).");
		return;
	}
	// The workers: KYTY_WEAVE_WORKERS, or as many as the performance cores spare after the GPU thread's core, the
	// recording and upload workers, the reader and the game's threads: their logical CPUs less 8, at most 8 (8 on a
	// 16-thread CPU, 4 on a 6-core with SMT). Under 2 the weave stands down; a count set lifts the check. Where the
	// classes are not known every core counts.
	uint32_t workers = kyty_local_weave_workers.load(std::memory_order_relaxed);
	if (workers == 0) {
		uint32_t cores = 0, performance = 0, logical = 0;
		LocalPlatform::CoreCounts(cores, performance, logical);
		const uint32_t cpus = performance != 0 ? performance
		                      : logical != 0   ? logical
		                                       : std::thread::hardware_concurrency();
		workers             = cpus > 8 ? std::min(MaxWorkers, cpus - 8) : 0;
		if (workers < 2) {
			kyty_local_weave_mode.store(0, std::memory_order_relaxed);
			std::printf(
			    "Weave: KYTY_WEAVE=1 stands down: %u logical CPUs on the performance cores, under the 10 it needs "
			    "(KYTY_WEAVE_WORKERS lifts the check). This run stays in mode 0.\n",
			    cpus);
			std::fflush(stdout);
			return;
		}
	}
	// The workers' draws are the scout's write ring's to check, and the table path's to make.
	if (kyty_local_scout_mode.load(std::memory_order_relaxed) == 0) {
		StandDown("Weave: needs the scout (KYTY_SCOUT=0).");
		return;
	}
	if (kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) == 0 ||
	    kyty_local_table_xpr_mode.load(std::memory_order_relaxed) == 0 ||
	    kyty_local_native_xpr_mode.load(std::memory_order_relaxed) == 0) {
		StandDown("Weave: needs KYTY_VULKAN_RECORDING, KYTY_TABLE_XPR and KYTY_NATIVE_XPR.");
		return;
	}
	g_workers = std::min(workers, MaxWorkers);
#endif
}

// While the weave runs, an unset KYTY_RENDER_CORE means auto: the GPU thread on a core of its own and every other
// thread of the process kept off it (KeepRenderCoreAlone; KYTY_RENDER_CPUS set, the thread is placed and nothing is
// chosen). Without the weave nothing changes.
const char* RenderCoreFallback() noexcept {
	return Enabled() ? "auto" : nullptr;
}

uint32_t Workers() noexcept {
	return Enabled() ? g_workers : 0;
}

bool Start(const Bodies& bodies) {
	if (g_answer >= 0) return g_answer != 0;
	if (!Enabled()) return false;
	// (Not decided while mode 4 runs: a run started in mode 0 may come back to it.)
	if (kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0) return false;
	const auto decide = [](bool started) {
		g_answer = started ? 1 : 0;
		if (!started) kyty_local_weave_mode.store(0, std::memory_order_relaxed);
		return started;
	};
	if (!Libs::Graphics::Scout::Running()) {
		std::printf("Weave: needs the scout (not running at queue 0's first draw); the weave stays off.\n");
		std::fflush(stdout);
		return decide(false);
	}
	if (bodies.read == nullptr || bodies.make == nullptr || bodies.release == nullptr || g_workers == 0)
		return decide(false);
	auto* r    = new Runtime;
	r->bodies  = bodies;
	r->workers = g_workers;
	r->parcels = std::make_unique<Parcel[]>(ParcelSlots);
	r->passed  = std::make_unique<std::atomic<bool>[]>(ParcelSlots);
	r->freed   = std::make_unique<std::atomic<const void*>[]>(FreedEntries);
	// (Each slot's result, its vectors kept allocated from parcel to parcel.)
	for (uint32_t i = 0; i < ParcelSlots; ++i)
		r->parcels[i].result = std::make_shared<Result>();
	g_runtime.store(r, std::memory_order_release);
	g_active.store(true, std::memory_order_seq_cst);
	try {
		r->reader = std::thread([r] { ReaderMain(*r); });
		for (uint32_t worker = 0; worker < r->workers; ++worker)
			r->threads[worker] = std::thread([r, worker] { WorkerMain(*r, worker); });
	} catch (const std::system_error& error) {
		Shut(*r);
		std::printf("Weave: its threads could not start (%s); the weave stays off.\n", error.what());
		std::fflush(stdout);
		return decide(false);
	}
	std::printf("Weave: on (KYTY_WEAVE=1): %u workers, %u calls a parcel, the GPU thread's core %s\n", r->workers,
	            kyty_local_weave_calls.load(std::memory_order_relaxed), CoreText().c_str());
	std::fflush(stdout);
	return decide(true);
}

bool Stopping() noexcept {
	const auto* r = Started();
	return r == nullptr || r->stopping.load(std::memory_order_relaxed);
}

// --- Pause, resume, stop ---

void Pause() {
	auto* r = Started();
	if (r == nullptr || r->stopped || Paused(*r)) return;
	if (!g_told_paused) {
		g_told_paused = true;
		std::printf("Weave: paused while KYTY_SPECULATE=4.\n");
		std::fflush(stdout);
	}
	// From here the reader cuts nothing (BeginParcel, PublishParcel) and is given no feed (PeekFeed), so it reaches
	// WaitReader and stays there; each parcel is passed, so a worker stops at its next draw and releases it.
	const uint32_t epoch = r->epoch.fetch_add(1, std::memory_order_seq_cst) + 1;
	WakeAll(*r);
	Stall stall("pausing");
	for (;;) {
		const bool parked = r->parked.load(std::memory_order_seq_cst) == epoch;
		bool       busy   = false;
		// (Scanned after `parked` was read: a parcel the reader published before it parked is seen here.)
		for (uint32_t index = 0; index < ParcelSlots; ++index) {
			PassSlot(*r, index);
			const auto state = r->parcels[index].state.load(std::memory_order_acquire);
			busy |= state == Parcel::Cut || state == Parcel::Taken || state == Parcel::Releasing;
		}
		if (parked && !busy) break;
		stall.Spin([&](uint32_t waited_ms, const char* what) {
			std::printf("Weave: the GPU thread has waited %u ms %s (the reader %s)\n", waited_ms, what,
			            parked ? "stopped" : "still reading");
		});
	}
	// The submissions fed before the pause go unread: the reader finds a gap after Resume and asks for the GPU
	// thread's registers (no submission is fed while paused, PushFeed).
	r->feed_head.store(r->feed_tail.load(std::memory_order_acquire), std::memory_order_release);
	ForgetSlotHolds(*r);
	g_active.store(false, std::memory_order_seq_cst);
}

void Resume() {
	auto* r = Started();
	if (r == nullptr || r->stopped || !Paused(*r)) return;
	// The frees made while paused went unlogged: every clock read before reads as the log turned over (FreedHits 2).
	g_freed_seq.store(g_freed_seq.load(std::memory_order_relaxed) + FreedEntries + 1, std::memory_order_release);
	g_active.store(true, std::memory_order_seq_cst);
	r->epoch.fetch_add(1, std::memory_order_seq_cst);
	WakeReader();
}

void StopAll() {
	if (auto* r = Started(); r != nullptr && !r->stopped) Shut(*r);
}

// --- Safepoints ---

void Hold::Open() noexcept {
	EXIT_IF(t_thread != Thread::None);
	m_open = true;
	if (t_holds++ != 0) return;
	g_requested.fetch_add(1, std::memory_order_seq_cst);
	// (Each worker either is inside a draw, which it ends without waiting for this thread, or sees the request
	// before its next one and stays out: both sides store, then load the other's, seq_cst.)
	for (uint32_t worker = 0; worker < MaxWorkers; ++worker) {
		Stall stall("at a safepoint");
		while (g_gates[worker].busy.load(std::memory_order_seq_cst) != 0)
			stall.Spin([&](uint32_t waited_ms, const char* what) {
				std::printf("Weave: the GPU thread has waited %u ms %s (worker %u in a draw)\n", waited_ms, what,
				            worker + 1);
			});
	}
	LiveCounters::Add(LiveCounters::WeaveSafepoints);
}

void Hold::Close() noexcept {
	if (--t_holds == 0) g_requested.fetch_sub(1, std::memory_order_seq_cst);
}

DrawScope::DrawScope(uint32_t worker) noexcept: m_worker(worker) {
	auto& gate = g_gates[m_worker];
	for (;;) {
		// (Busy first, then the clock: a slot first asked about after this draw's clock read can't be one it finds,
		// HoldsSlot.)
		gate.busy.store(1, std::memory_order_seq_cst);
		gate.seen.store(g_slot_clock.load(std::memory_order_seq_cst), std::memory_order_seq_cst);
		if (g_requested.load(std::memory_order_seq_cst) == 0) return;
		// A hold is open: out of the draw until it ends (a structural write, short), so the holder never waits for a
		// draw begun after it asked.
		gate.busy.store(0, std::memory_order_seq_cst);
		for (uint32_t spin = 0; g_requested.load(std::memory_order_acquire) != 0; ++spin) {
			if (spin < 4096)
				SpinPause();
			else
				std::this_thread::yield();
		}
	}
}

DrawScope::~DrawScope() {
	g_gates[m_worker].busy.store(0, std::memory_order_release);
}

// --- The free log ---

void NoteFreed(const void* object) noexcept {
	if (!Active()) return;
	auto* r = Started();
	if (r == nullptr) return;
	// (Only the GPU thread writes the log.)
	const auto seq = g_freed_seq.load(std::memory_order_relaxed);
	r->freed[seq % FreedEntries].store(object, std::memory_order_relaxed);
	g_freed_seq.store(seq + 1, std::memory_order_release);
}

uint64_t FreedClock() noexcept {
	return g_freed_seq.load(std::memory_order_acquire);
}

uint32_t FreedHits(uint64_t& scanned, std::span<const void* const> sorted, std::vector<const void*>& hits) {
	const auto* r   = Started();
	const auto  seq = g_freed_seq.load(std::memory_order_acquire);
	if (r == nullptr || scanned >= seq) return 0;
	if (seq - scanned > FreedEntries) {
		scanned = seq;
		return 2;
	}
	uint32_t found = 0;
	for (; scanned < seq; ++scanned)
		if (const void* object = r->freed[scanned % FreedEntries].load(std::memory_order_relaxed);
		    object != nullptr && std::binary_search(sorted.begin(), sorted.end(), object, std::less<>())) {
			hits.push_back(object);
			found = 1;
		}
	return found;
}

// --- Deferred erase ---

bool HoldsSlot(Slot kind, uint32_t index, uint32_t generation) noexcept {
	auto* r = Started();
	if (r == nullptr) return false;
	const uint64_t  key = (static_cast<uint64_t>(kind) << 32u) | index;
	std::lock_guard lock(r->holds_mutex);
	// (The image slots let go at the last image ask are erased by now, its caller's next: their entries go. A buffer
	// ask may come from the other cache, whose erase is no sign of it.)
	if (kind == Slot::Image) {
		for (const auto entry: r->released)
			Unretire(entry);
		r->released.clear();
	}
	auto       held  = std::find_if(r->holds.begin(), r->holds.end(), [key](const auto& h) { return h.key == key; });
	const bool known = held != r->holds.end();
	// The stamp of its first ask: a buffer slot was out of every lookup by then (retired), so a draw that read the
	// clock after it can't have found it; an image slot is in the retiring entries from before it, so such a draw
	// doesn't read it (ImageRetiring). Kept while a worker is inside a draw that began before. An image slot that finds
	// every entry taken is kept as it is, with no stamp, until an ask finds one free: a draw may read a slot not
	// erased, and every entry is then a slot held, let go once the draws in progress end.
	const uint32_t entry = known                 ? held->entry
	                       : kind == Slot::Image ? Retire(RetiringId(index, generation))
	                                             : Runtime::NoEntry;
	if (kind == Slot::Image && entry == Runtime::NoEntry) return true;
	const uint64_t stamp = known ? held->stamp : g_slot_clock.fetch_add(1, std::memory_order_seq_cst) + 1;
	bool           busy  = false;
	for (const auto& gate: g_gates)
		if (gate.busy.load(std::memory_order_seq_cst) != 0 && gate.seen.load(std::memory_order_seq_cst) < stamp) {
			busy = true;
			break;
		}
	if (!busy) {
		if (known) r->holds.erase(held);
		if (entry != Runtime::NoEntry) r->released.push_back(entry);
		return false;
	}
	if (!known) r->holds.push_back({.key = key, .stamp = stamp, .entry = entry});
	return true;
}

bool ImagesRetiring() noexcept {
	return g_retiring_used.load(std::memory_order_acquire) != 0;
}

bool ImageRetiring(uint32_t index, uint32_t generation) noexcept {
	if (g_retiring_used.load(std::memory_order_acquire) == 0) return false;
	const uint64_t id = RetiringId(index, generation);
	for (uint32_t entry = 0, high = g_retiring_high.load(std::memory_order_acquire); entry < high; ++entry)
		if (g_retiring[entry].load(std::memory_order_acquire) == id) return true;
	return false;
}

// --- The feed ---

bool PushFeed(const Feed& feed) noexcept {
	auto* r = Started();
	if (r == nullptr || r->stopping.load(std::memory_order_relaxed) || Paused(*r)) return false;
	// (One producer at a time: the queue mutex.)
	const auto tail = r->feed_tail.load(std::memory_order_relaxed);
	if (tail - r->feed_head.load(std::memory_order_acquire) >= FeedEntries) return false;
	r->feeds[tail % FeedEntries] = feed;
	r->feed_tail.store(tail + 1, std::memory_order_release);
	WakeReader();
	return true;
}

bool PeekFeed(Feed& feed) noexcept {
	auto* r = Started();
	if (r == nullptr || Paused(*r)) return false;
	const auto head = r->feed_head.load(std::memory_order_acquire);
	if (head == r->feed_tail.load(std::memory_order_acquire)) return false;
	feed = r->feeds[head % FeedEntries];
	return true;
}

void TakeFeed() noexcept {
	auto* r = Started();
	if (r == nullptr) return;
	const auto head = r->feed_head.load(std::memory_order_acquire);
	if (head != r->feed_tail.load(std::memory_order_acquire)) r->feed_head.store(head + 1, std::memory_order_release);
}

uint32_t ReaderWakes() noexcept {
	const auto* r = Started();
	return r != nullptr ? r->wakes.load(std::memory_order_seq_cst) : 0;
}

void WaitReader(uint32_t seen) noexcept {
	auto* r = Started();
	if (r == nullptr) return;
	for (uint32_t spin = 0;;) {
		if (r->stopping.load(std::memory_order_acquire)) return;
		// Paused: stopped here until Resume, and Pause told (`parked` names the pause seen, so an earlier one's word
		// never counts for a later one).
		const uint32_t epoch  = r->epoch.load(std::memory_order_seq_cst);
		const bool     paused = (epoch & 1u) != 0;
		if (paused)
			r->parked.store(epoch, std::memory_order_seq_cst);
		else if (r->wakes.load(std::memory_order_acquire) != seen)
			return;
		if (!paused && spin < IdleSpins && SpinGate::Open(false)) {
			++spin;
			SpinPause();
			continue;
		}
		// (Said before the last look, so a waker that moves `wakes` after it sees the reader blocked and wakes it.)
		const auto now = r->wakes.load(std::memory_order_seq_cst);
		r->reader_sleeping.store(1, std::memory_order_seq_cst);
		if (r->wakes.load(std::memory_order_seq_cst) == now && r->epoch.load(std::memory_order_seq_cst) == epoch &&
		    !r->stopping.load(std::memory_order_seq_cst) && (paused || now == seen))
			r->wakes.wait(now, std::memory_order_seq_cst);
		r->reader_sleeping.store(0, std::memory_order_relaxed);
		spin = 0;
	}
}

void WakeReader() noexcept {
	auto* r = Started();
	if (r == nullptr) return;
	r->wakes.fetch_add(1, std::memory_order_seq_cst);
	if (r->reader_sleeping.load(std::memory_order_seq_cst) != 0) r->wakes.notify_one();
}

// --- The parcels ---

Parcel* BeginParcel() noexcept {
	auto* r = Started();
	if (r == nullptr) return nullptr;
	const uint64_t serial = r->published.load(std::memory_order_relaxed);
	const auto     index  = static_cast<uint32_t>(serial % ParcelSlots);
	auto&          parcel = r->parcels[index];
	// (The ring holds about nine frames' parcels: its next slot is nearly always Free. Otherwise the GPU thread or a
	// worker releases it.)
	for (uint32_t spin = 0;;) {
		if (r->stopping.load(std::memory_order_acquire) || Paused(*r) || !Active()) return nullptr;
		if (parcel.state.load(std::memory_order_acquire) == Parcel::Free) break;
		if (spin < IdleSpins && SpinGate::Open(false)) {
			++spin;
			SpinPause();
			continue;
		}
		const auto seen = r->releases.load(std::memory_order_seq_cst);
		r->release_waiter.store(1, std::memory_order_seq_cst);
		if (parcel.state.load(std::memory_order_seq_cst) != Parcel::Free && !Paused(*r) &&
		    !r->stopping.load(std::memory_order_seq_cst))
			r->releases.wait(seen, std::memory_order_seq_cst);
		r->release_waiter.store(0, std::memory_order_relaxed);
		spin = 0;
	}
	r->passed[index].store(false, std::memory_order_relaxed);
	parcel.serial = serial;
	parcel.dropped.store(false, std::memory_order_relaxed);
	parcel.progress.store(0, std::memory_order_relaxed);
	parcel.taken_ns.store(0, std::memory_order_relaxed);
	parcel.calls = parcel.draws = 0;
	parcel.cuts.clear();
	parcel.state.store(Parcel::Filling, std::memory_order_release);
	return &parcel;
}

void PublishParcel(Parcel& parcel) noexcept {
	auto* r = Started();
	if (r == nullptr) return;
	// (Paused or stopping: the parcel goes unpublished, as one too short, and its serial is the next one's.)
	if (Paused(*r) || r->stopping.load(std::memory_order_seq_cst)) {
		AbandonParcel(parcel);
		return;
	}
	LiveCounters::Add(LiveCounters::WeaveParcels);
	parcel.state.store(Parcel::Cut, std::memory_order_release);
	r->published.store(parcel.serial + 1, std::memory_order_release);
	r->signal.fetch_add(1, std::memory_order_seq_cst);
	// (Every sleeper: each waits for its own serial.)
	if (r->sleepers.load(std::memory_order_seq_cst) != 0) r->signal.notify_all();
}

void AbandonParcel(Parcel& parcel) noexcept {
	auto* r = Started();
	if (r == nullptr) return;
	r->bodies.release(parcel);
	parcel.state.store(Parcel::Free, std::memory_order_release);
}

uint64_t Published() noexcept {
	const auto* r = Started();
	return r != nullptr ? r->published.load(std::memory_order_acquire) : 0;
}

Parcel& ParcelAt(uint64_t serial) noexcept {
	auto* r = Started();
	if (r == nullptr) {
		std::fprintf(stderr, "Weave: ParcelAt before Start\n");
		std::abort();
	}
	return r->parcels[serial % ParcelSlots];
}

void PassParcel(Parcel& parcel) noexcept {
	auto* r = Started();
	if (r == nullptr) return;
	PassSlot(*r, SlotIndex(*r, parcel));
}

} // namespace Weave

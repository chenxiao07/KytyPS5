#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace vk::detail { class DispatchLoaderDynamic; }
namespace Libs::Graphics { class MasterSemaphore; }
struct VkCommandBuffer_T;

// Vulkan command recording on a worker thread (KYTY_VULKAN_RECORDING). The
// renderer remains the sole resource/cache owner; a producer scope transfers
// only fully copied Vulkan call arguments.
namespace LocalVulkanRecording {
void Install();
// The Vulkan entry points without recording: for a queue the recording worker does not own.
const vk::detail::DispatchLoaderDynamic& DirectDispatch();
// Replaces the vkCmdBindPipeline every bind ends in (recorded, packet or direct: the device-loss breadcrumbs of
// device-fault.h) and returns the one replaced. After Install, before any command is recorded.
void* ReplaceBindPipeline(void* entry);
// Complete this producer's CPU recording before handing its Vulkan objects to
// another thread. A receiver's drain cannot flush the sender's thread-local queue.
// This does not wait for GPU execution.
void Drain();
struct Segment {
    const void* data = nullptr;
    size_t size = 0;
};
using ReplayPacket = void (*)(std::span<const Segment>, const vk::detail::DispatchLoaderDynamic&);
// Draws and descriptor encoding are recorded as whole packets.
bool PacketsEnabled();
// Segments are copied transactionally. An optional immutable owner is retained
// until replay finishes; callers never lend stack or mutable-cache pointers.
bool EnqueuePacket(ReplayPacket replay, std::span<const Segment> segments,
                   std::shared_ptr<const void> owner = {}, bool fold_barrier = false);
// One segment, no owner, and (when true is returned) `copy`: where its copy lives in the stream's chunk. A passive
// producer's chunk stays where it is, read by nobody, until the render thread splices its segment (then its replay reads
// it) or discards it; the splicing thread may rewrite the copy's bytes before the splice (a speculation's relocatable
// buffer references, Spec::RelocCell), never after.
bool EnqueuePacketCopy(ReplayPacket replay, const Segment& segment, void** copy);
// Preserve order when a packet cannot fit. Does not retain any argument.
void ReplayInline(ReplayPacket replay, std::span<const Segment> segments);
// A packet's replay that records commands the recording does not see (a library's own: DLSS through Streamline)
// ends the reuse of bound pipelines and dynamic state the draws' replays keep, as a direct call does.
void InvalidateReplayState();
// Command-buffer begin/end and queue submission, queued in order with the recorded
// commands instead of draining the recording queue first (KYTY_DEFERRED_SUBMIT).
// A submit packet is published at once: other threads may wait on its tick.
bool DeferredSubmitEnabled();
// A queued submit on a separate queue forms a boundary in graphics submit replay.
bool EnqueueDeferred(ReplayPacket replay, std::span<const Segment> segments, bool publish,
                     bool fold_barrier = false);
namespace Detail {
// Each generation has three states: off, pending activation, and active.
inline constexpr uint64_t SubmitFoldPending = 1;
inline constexpr uint64_t SubmitFoldActive = 2;
inline constinit std::atomic<uint64_t> g_submit_fold_state {0};
}
inline bool SubmitFoldEnabled() {
    return (Detail::g_submit_fold_state.load(std::memory_order_relaxed) & Detail::SubmitFoldActive) != 0;
}
// Zero while off. The producer runs activate under the graphics queue lock before assigning a tick.
template<typename Activate>
uint64_t SubmitFoldEpoch(Activate activate) {
    const auto state = Detail::g_submit_fold_state.load(std::memory_order_relaxed);
    if ((state & Detail::SubmitFoldPending) != 0) [[unlikely]] return activate(state);
    return (state & Detail::SubmitFoldActive) != 0 ? state : 0;
}
// The caller holds the graphics queue lock shared by the direct submit paths.
uint64_t ActivateSubmitFold(uint64_t pending);
bool IsCurrentSubmitFold(uint64_t epoch);
bool SetSubmitFold(int64_t value);
void EndSubmitFold(); // a direct readback submit on this producer, without publishing or draining
// Bracket a direct graphics-queue submit under the caller's existing queue lock.
class DirectGraphicsSubmit {
public:
    explicit DirectGraphicsSubmit(bool active = SubmitFoldEnabled()) : m_active(active) {
        if (m_active) Begin();
    }
    ~DirectGraphicsSubmit() {
        if (m_active) End();
    }
    DirectGraphicsSubmit(const DirectGraphicsSubmit&) = delete;
    DirectGraphicsSubmit& operator=(const DirectGraphicsSubmit&) = delete;

private:
    static void Begin();
    static void End();
    const bool m_active;
};
// Graphics submissions use the existing cap, leaving the two slots reserved by the scheduler.
inline constexpr uint32_t MaxFoldCommands = 22;
struct FoldCompletion {
    void* context;
    // After a successful driver call, release replayed chunks before the next pending request (if any).
    void (*complete)(void*, size_t next_request);
};
struct SubmitReplay {
    ReplayPacket single;
    void (*prepare)(const Segment&, const vk::detail::DispatchLoaderDynamic&);
    void (*fold)(std::span<const Segment>, const vk::detail::DispatchLoaderDynamic&,
                 const Libs::Graphics::MasterSemaphore&, uint64_t epoch, FoldCompletion);
    bool (*has_wait)(const Segment&, const Libs::Graphics::MasterSemaphore&);
    bool (*can_append)(const Segment&, const Segment&);
};
bool EnqueueSubmit(const SubmitReplay& replay, const Segment& segment, uint32_t commands,
                   const Libs::Graphics::MasterSemaphore& timeline, uint64_t epoch, bool ends_fold);
// Deferred queue submissions: counted when queued and when the worker has handed them to
// the driver. WaitDeferredSubmits blocks until every submission queued so far reached the
// driver (not the GPU). A present must not depend on a signal still in the worker's queue:
// the Windows driver then blocks inside vkQueuePresentKHR, holding the queue lock the
// worker needs for that very submission.
void NoteDeferredSubmitQueued();
void NoteDeferredSubmitDone(size_t requests = 1);
void WaitDeferredSubmits();
// Deferred submissions queued and handed to the driver so far (diagnostics).
uint64_t DeferredSubmitsQueued();
uint64_t DeferredSubmitsDone();
uint64_t StateEpoch();
// Commands the calling thread recorded that do GPU work or synchronize (draws, dispatches,
// copies, clears, rendering scopes, barriers): unchanged between two barriers = no work in between.
uint64_t WorkCalls();
// Work calls and packets (draws, descriptors) the calling thread recorded: unchanged = nothing recorded.
uint64_t RecordedWork();
// Commands that change memory or images (draws, dispatches, copies, clears) and packets the calling thread
// recorded: unchanged = nothing but barriers and rendering scopes recorded.
uint64_t RecordedWrites();
class ProducerScope {
public:
    ProducerScope();
    ~ProducerScope();
    ProducerScope(const ProducerScope&) = delete;
    ProducerScope& operator=(const ProducerScope&) = delete;
};

// Worker packets spliced at commit. A passive producer (a speculation worker's) records exactly as a producer
// does, into chunks from its own pool, with no replay thread: nothing it records reaches the driver until the
// render thread splices the chunks into its own stream, where its replay thread executes them in order between
// the render thread's packets. Its commands name the placeholder command buffer, which a spliced chunk's replay
// resolves to the splice's target (the render thread's open command buffer).
struct Chunk;
// A passive producer's closed segment: its chunks in recording order, not replayed. Destroying (or Discard) runs
// each packet's destructor without executing it and returns the chunks to their pool.
class ChunkList {
public:
    ChunkList() = default;
    ChunkList(ChunkList&& other) noexcept;
    ChunkList& operator=(ChunkList&& other) noexcept;
    ~ChunkList();
    ChunkList(const ChunkList&) = delete;
    ChunkList& operator=(const ChunkList&) = delete;
    [[nodiscard]] bool empty() const { return chunks.empty(); }
    std::vector<Chunk*> chunks;
    uint64_t commands = 0;
    // A command of the segment could not be recorded (too large, an unknown extension chain, a direct call on the
    // placeholder): the segment is incomplete and must be discarded, never spliced.
    bool failed = false;
};
VkCommandBuffer_T* PlaceholderCommandBuffer();
// The placeholder resolved to the target of the spliced chunk being replayed; any other handle is itself.
VkCommandBuffer_T* Resolve(VkCommandBuffer_T* command);
// The calling thread records into its passive producer (inside a segment).
bool Passive();
// The calling thread has a passive producer (PassiveProducerScope).
bool HasPassiveProducer();
// The passive producer's segment: Begin discards anything recorded since the last End and arms the producer (the
// thread's calls are recorded from then on); End disarms it, closes the open chunk and hands the segment's chunks over.
// Outside a segment the thread's calls go to the driver as without a producer.
void BeginSegment();
ChunkList EndSegment();
void Discard(ChunkList& list);
// Render thread (a producer's thread): the list's chunks appended to its stream after everything it recorded so
// far, each replayed with the placeholder resolved to `target`. A failed list is discarded instead (false).
bool Splice(ChunkList& list, VkCommandBuffer_T* target);
// Chunks spliced and passive chunks allocated so far (diagnostics).
uint64_t SplicedChunks();
uint64_t PassiveChunksAllocated();
class PassiveProducerScope {
public:
    PassiveProducerScope();
    ~PassiveProducerScope();
    PassiveProducerScope(const PassiveProducerScope&) = delete;
    PassiveProducerScope& operator=(const PassiveProducerScope&) = delete;
};

// A weave worker's calls recorded apart (weave.h). The worker has no producer: between BeginApart and EndApart every
// call it records goes, numbered, into a buffer of its own, from a pool of this file's chunks, and none reaches the
// driver. It records through ApartMarker() as its command buffer (the address of an object of this file's: no driver
// handle equals it); a direct call on the marker fails the buffer. The GPU thread, a producer, splices parts of the
// buffer into its own stream, in order: SpliceApartPart enqueues calls [begin, end) as one packet, published at once,
// whose replay on the recording worker writes `target` over each call's marker and executes it, carrying on from where
// the last part of the same buffer stopped; the calls before `begin` no part took are destroyed unexecuted, and with
// `last` the calls after `end` too. A buffer no part takes destroys its calls unexecuted. Nothing here changes how any
// other call is recorded or replayed.
class Apart;
VkCommandBuffer_T* ApartMarker();
void BeginApart();
// The open buffer closed and handed over: with `keep`, once every call's first captured word is proven the marker;
// null when not kept or not so proven (the buffer destroyed).
std::shared_ptr<Apart> EndApart(bool keep);
// GPU thread: false, nothing enqueued, for a range the buffer doesn't hold or one before the last part's end.
bool SpliceApartPart(const std::shared_ptr<Apart>& buffer, VkCommandBuffer_T* target, uint64_t begin, uint64_t end,
                     bool last);
// The calls the calling thread recorded into its open buffer so far (a made draw's calls are [before, after)).
uint64_t ApartCalls();
}

#include "vulkan-recording.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "live-census.h"
#include "live-counters.h"
#include "live-trace-gpu.h"
#include "local-platform.h"
#include "mode-toggle.h"
#include "spin-gate.h"
#include "time-census.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

extern "C" {
// 1: record Vulkan commands for the worker thread, draws and descriptor
// encoding as whole packets (KYTY_VULKAN_RECORDING).
volatile std::atomic<uint32_t> kyty_local_vulkan_recording_mode {1};
// 1: queue command-buffer begin/end/submit instead of draining (DeferredSubmitEnabled).
volatile std::atomic<uint32_t> kyty_local_deferred_submit_mode {1};
// Pauses the idle worker spins before it blocks (a blocked worker costs the render thread a
// wake-up system call per publish on Windows; Linux the same unless KYTY_LINUX_FAST_LOCKS=0).
#if defined(_WIN32)
volatile std::atomic<uint32_t> kyty_local_recording_spin {4000};
#elif defined(__linux__)
volatile std::atomic<uint32_t> kyty_local_recording_spin {[] {
    const char* value = std::getenv("KYTY_LINUX_FAST_LOCKS");
    return value == nullptr || value[0] != '0' ? 4000u : 0u;
}()};
#else
volatile std::atomic<uint32_t> kyty_local_recording_spin {0};
#endif
}

extern "C" {
extern volatile std::atomic_uint32_t kyty_local_speculate_mode; // KYTY_SPECULATE (graphics/guest_gpu/speculation.cpp)
}

namespace LocalVulkanRecording {
namespace {
constexpr size_t ChunkBytes = 64 * 1024;
// Deep enough that a burst of recorded calls never blocks the producer in WaitForRoom.
constexpr size_t ChunkCount = 64;
constexpr uint32_t End = std::numeric_limits<uint32_t>::max();
vk::detail::DispatchLoaderDynamic original;

// The offset and kind share the old 32-bit payload field, with no mask in the legacy loop.
enum class PacketKind : uint16_t { Record, Barrier, Submit };
static_assert(ChunkBytes - 1 <= std::numeric_limits<uint16_t>::max());
struct Header {
    void (*execute)(void*) = nullptr;
    void (*destroy)(void*) = nullptr;
    uint16_t payload = 0;
    PacketKind kind = PacketKind::Record;
    uint32_t next = End;
};

void InvalidateRawState();

struct FoldSubmitInfo {
    const Libs::Graphics::MasterSemaphore* timeline;
    uint64_t epoch;
    uint64_t boundary;
    bool ends_fold;
    uint64_t graphics_boundary = 0;
};

struct SubmitPacket {
    // Keep the same capture footprint as EnqueueDeferred. Only fold packets copy the second segment.
    std::array<Segment, 8> copied {};
    size_t commands;
    const SubmitReplay* replay;

    const FoldSubmitInfo& FoldInfo() const {
        return *static_cast<const FoldSubmitInfo*>(copied[1].data);
    }

    void operator()() {
        InvalidateRawState();
        replay->single({copied.data(), 1}, original);
    }
};

std::mutex g_submit_fold_switch_mutex;

// Serialize only the publication and the edges of a direct submit. Packets published during a direct
// submit end their own fold, so neither edge can be crossed. Never hold this lock in a driver call.
std::mutex g_graphics_publication_mutex;
uint64_t g_graphics_boundary = 0;
size_t g_direct_graphics_submits = 0;
} // namespace
struct ChunkPool;
struct Chunk {
    alignas(64) std::array<std::byte, ChunkBytes> data;
    size_t used = 0;
    uint32_t first = End, last = End, count = 0;
    // A passive producer's chunk: returned to this pool once replayed (spliced) or discarded. Null: a stream's own.
    ChunkPool* pool = nullptr;
    // The command buffer the placeholder resolves to while this (spliced) chunk is replayed.
    VkCommandBuffer target = VK_NULL_HANDLE;
};
namespace {
std::atomic<uint64_t> g_spliced_chunks {0}, g_passive_chunks {0};
}
// A passive producer's chunks. Spliced chunks come back from the replay thread, so the free list is locked (a few
// hundred chunks a frame). Pools are never freed: a producer's pool goes back to the idle list for the next one,
// and its chunks still in flight find it there.
struct ChunkPool {
    std::mutex mutex;
    std::vector<Chunk*> free;
    Chunk* Get() {
        {
            const std::lock_guard lock(mutex);
            if (!free.empty()) {
                auto* chunk = free.back();
                free.pop_back();
                return chunk;
            }
        }
        g_passive_chunks.fetch_add(1, std::memory_order_relaxed);
        auto* chunk = new Chunk;
        chunk->pool = this;
        return chunk;
    }
    void Put(Chunk* chunk) {
        chunk->used = 0;
        chunk->count = 0;
        chunk->first = chunk->last = End;
        chunk->target = VK_NULL_HANDLE;
        const std::lock_guard lock(mutex);
        free.push_back(chunk);
    }
};
namespace {
std::mutex g_pools_mutex;
std::vector<ChunkPool*> g_idle_pools;
ChunkPool* AcquirePool() {
    const std::lock_guard lock(g_pools_mutex);
    if (g_idle_pools.empty()) return new ChunkPool;
    auto* pool = g_idle_pools.back();
    g_idle_pools.pop_back();
    return pool;
}
void ReleasePool(ChunkPool* pool) {
    const std::lock_guard lock(g_pools_mutex);
    g_idle_pools.push_back(pool);
}
// A discarded chunk: every packet's captures destroyed, none executed, the chunk back to its pool.
void DiscardChunk(Chunk* chunk) {
    for (auto offset = chunk->first; offset != End;) {
        const auto* header = reinterpret_cast<const Header*>(chunk->data.data() + offset);
        const auto next = header->next;
        if (header->destroy) header->destroy(chunk->data.data() + header->payload);
        offset = next;
    }
    chunk->pool->Put(chunk);
}
// The placeholder: an address no driver hands out as a command buffer.
char g_placeholder_object;
VkCommandBuffer Placeholder() { return reinterpret_cast<VkCommandBuffer>(&g_placeholder_object); }
// The replay thread's spliced chunk target (set per chunk).
thread_local VkCommandBuffer t_splice_target = VK_NULL_HANDLE;
inline VkCommandBuffer ResolveCommand(VkCommandBuffer command) {
    return command == Placeholder() ? t_splice_target : command;
}
// This thread records into its passive producer: between BeginSegment and EndSegment (outside a segment its calls go
// to the driver as without one).
thread_local bool t_passive = false;

// A transaction owns one command plus its parameter graph in a single chunk.
// Failure never publishes any of its partially copied data.
class Writer {
public:
    explicit Writer(Chunk& chunk) : data(chunk.data.data()), used(chunk.used) {}
    bool valid = true, overflow = false;
    std::byte* data;
    size_t used;
    uint32_t command = End;

    void* Allocate(size_t bytes, size_t alignment) {
        if (!valid) return nullptr;
        const size_t offset = (used + alignment - 1) & ~(alignment - 1);
        if (bytes > ChunkBytes || offset > ChunkBytes - bytes) {
            valid = false;
            overflow = true;
            return nullptr;
        }
        used = offset + bytes;
        return data + offset;
    }
    const void* Copy(const void* source, size_t bytes) {
        if (!source || !bytes) return nullptr;
        auto* target = Allocate(bytes, alignof(std::max_align_t));
        if (target) std::memcpy(target, source, bytes);
        return target;
    }
    template<typename T>
    const T* Copy(const T* source, size_t count) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (!source || !count) return nullptr;
        if (count > ChunkBytes / sizeof(T)) { valid = false; overflow = true; return nullptr; }
        auto* target = static_cast<T*>(Allocate(count * sizeof(T), alignof(T)));
        if (!target) return nullptr;
        std::memcpy(target, source, count * sizeof(T));
        for (size_t i = 0; i < count && valid; ++i) {
            auto& value = target[i];
            if constexpr (requires { value.pNext; }) {
                // Unknown extension chains retain the exact synchronous call.
                if (value.pNext) { valid = false; break; }
            }
            if constexpr (std::is_same_v<T, VkDependencyInfo>) {
                value.pMemoryBarriers = Copy(value.pMemoryBarriers, value.memoryBarrierCount);
                value.pBufferMemoryBarriers = Copy(value.pBufferMemoryBarriers, value.bufferMemoryBarrierCount);
                value.pImageMemoryBarriers = Copy(value.pImageMemoryBarriers, value.imageMemoryBarrierCount);
            } else if constexpr (std::is_same_v<T, VkRenderingInfo>) {
                value.pColorAttachments = Copy(value.pColorAttachments, value.colorAttachmentCount);
                value.pDepthAttachment = Copy(value.pDepthAttachment, 1);
                value.pStencilAttachment = Copy(value.pStencilAttachment, 1);
            } else if constexpr (std::is_same_v<T, VkWriteDescriptorSet>) {
                // Ignored fields need not point to readable memory. Copy only
                // the payload selected by descriptorType, never all pointers.
                value.pImageInfo = nullptr;
                value.pBufferInfo = nullptr;
                value.pTexelBufferView = nullptr;
                switch (value.descriptorType) {
                    case VK_DESCRIPTOR_TYPE_SAMPLER:
                    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                    case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                        value.pImageInfo = Copy(source[i].pImageInfo, value.descriptorCount);
                        break;
                    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                        value.pBufferInfo = Copy(source[i].pBufferInfo, value.descriptorCount);
                        break;
                    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                        value.pTexelBufferView = Copy(source[i].pTexelBufferView, value.descriptorCount);
                        break;
                    default: valid = false; break;
                }
            }
        }
        return target;
    }
    template<typename Function>
    void Command(Function function, PacketKind kind = PacketKind::Record) {
        static_assert(std::is_nothrow_move_constructible_v<Function>);
        static_assert(std::is_nothrow_destructible_v<Function>);
        auto* header = static_cast<Header*>(Allocate(sizeof(Header), alignof(Header)));
        void* payload = Allocate(sizeof(Function), alignof(Function));
        if (!valid) return;
        new (payload) Function(std::move(function));
        new (header) Header {
            .execute = [](void* pointer) { (*static_cast<Function*>(pointer))(); },
            .destroy = nullptr,
            .payload = static_cast<uint16_t>(static_cast<std::byte*>(payload) - data),
            .kind = kind,
        };
        if constexpr (!std::is_trivially_destructible_v<Function>)
            header->destroy = [](void* pointer) { static_cast<Function*>(pointer)->~Function(); };
        command = static_cast<uint32_t>(reinterpret_cast<std::byte*>(header) - data);
    }
};

// A producer waiting for the worker (a full stream, a drain): the slow-frame lines' record_wait (render thread) and
// the live counter record_wait_us.
struct RecordWait {
    LiveCensus::WaitScope                       scope {LiveCensus::WaitRecord};
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    RecordWait() = default;
    RecordWait(const RecordWait&) = delete;
    RecordWait& operator=(const RecordWait&) = delete;
    ~RecordWait() {
        LiveCounters::Add(LiveCounters::RecordWaitUs,
                          static_cast<uint64_t>((std::chrono::steady_clock::now() - start).count()) / 1000u);
    }
};

class Stream;
thread_local Stream* executing = nullptr;
class Stream {
public:
    struct PassiveTag {};
    Stream() : own(std::make_unique<std::array<Chunk, ChunkCount>>()) {
        for (size_t i = 0; i < ChunkCount; ++i) slots[i] = &(*own)[i];
        worker = std::thread([this] { Run(); });
    }
    // A passive producer: no replay thread; its segments leave as chunk lists (EndSegment).
    explicit Stream(PassiveTag) : passive(true), pool(AcquirePool()) {}
    ~Stream() {
        if (passive) {
            {
                ChunkList dropped = EndSegment();
            }
            if (open) pool->Put(open);
            ReleasePool(pool);
            return;
        }
        Drain();
        stopping.store(true, std::memory_order_release);
        // atomic::wait requires a changed value, not just notify, to terminate.
        published.fetch_add(1, std::memory_order_release);
        published.notify_one();
        worker.join();
    }
    alignas(64) std::atomic<uint64_t> state_epoch {0};
    alignas(64) bool was_recording = false;
    const bool passive = false;
    void BeforeDirect() {
        if (passive) {
            state_epoch.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Drain();
        const bool active = kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0;
        if (active || was_recording) state_epoch.fetch_add(1, std::memory_order_relaxed);
        was_recording = active;
    }
    // (By reference: the wrappers pass a named capture, which a by-value parameter would copy on the GPU thread's
    // hottest path.)
    template<bool Submit = false, typename Capture>
    bool Enqueue(Capture&& capture) {
        if constexpr (!Submit) {
            if (passive) [[unlikely]] return EnqueuePassive(capture);
        }
        for (int attempt = 0; attempt < 2; ++attempt) {
            WaitForRoom();
            auto& chunk = *slots[sequence % ChunkCount];
            Writer writer(chunk);
            FoldSubmitInfo* submit = nullptr;
            if constexpr (Submit) submit = capture(writer);
            else capture(writer);
            if (writer.valid && writer.command != End) {
                if (chunk.last != End)
                    reinterpret_cast<Header*>(chunk.data.data() + chunk.last)->next = writer.command;
                else chunk.first = writer.command;
                chunk.last = writer.command;
                chunk.used = writer.used;
                ++chunk.count;
                if constexpr (Submit) Publish<true>(submit);
                else if (chunk.count >= 128 || chunk.used >= ChunkBytes * 3 / 4) Publish();
                return true;
            }
            if (!writer.overflow || chunk.count == 0) break;
            Publish();
        }
        return false;
    }
    void Flush() {
        if (!passive) Publish();
    }
    void EndSubmitFold() { ++submit_fold_boundary; }
    uint64_t SubmitFoldBoundary() const { return submit_fold_boundary; }
    // Packets recorded or published and not yet replayed (a direct call would wait for them).
    [[nodiscard]] bool Pending() const {
        if (passive) return false;
        // (completed first: until the replay thread is done with every slot, one may still be its.)
        return completed.load(std::memory_order_acquire) != sequence || slots[sequence % ChunkCount]->count != 0;
    }
    // A passive producer's segment.
    void BeginSegment() {
        ChunkList dropped = EndSegment();
    }
    ChunkList EndSegment() {
        ClosePassive();
        ChunkList list;
        list.chunks.swap(closed);
        list.commands = segment_commands;
        list.failed = segment_failed;
        segment_commands = 0;
        segment_failed = false;
        return list;
    }
    void FailSegment() { segment_failed = true; }
    // Render thread: the foreign chunks published after everything recorded so far, one slot each. The replay
    // thread puts the stream's own chunk back in the slot when it is done with a foreign one.
    void Splice(ChunkList& list, VkCommandBuffer target) {
        Publish();
        for (auto* chunk : list.chunks) {
            WaitForRoom();
            chunk->target = target;
            slots[sequence % ChunkCount] = chunk;
            published.store(++sequence, std::memory_order_seq_cst);
            if (sleeping.load(std::memory_order_seq_cst)) published.notify_one();
        }
        g_spliced_chunks.fetch_add(list.chunks.size(), std::memory_order_relaxed);
        list.chunks.clear();
        list.commands = 0;
    }
    void Drain() {
        if (passive) return;
        Publish();
        if (completed.load(std::memory_order_acquire) == sequence) return;
        const RecordWait waiting;
#if defined(__linux__)
        static const bool fast_locks = [] {
            const char* value = std::getenv("KYTY_LINUX_FAST_LOCKS");
            return value == nullptr || value[0] != '0';
        }();
        // Linux: a short spin first, the same as the worker's (KYTY_LINUX_FAST_LOCKS=0: waits at once).
        if (fast_locks) {
            for (uint32_t spin = 0; spin < 4000; ++spin) {
                if (completed.load(std::memory_order_acquire) == sequence) return;
                _mm_pause();
            }
        }
#endif
        for (auto done = completed.load(std::memory_order_acquire); done != sequence;
             done = completed.load(std::memory_order_acquire)) completed.wait(done, std::memory_order_acquire);
    }
private:
    // Producer and consumer only touch a slot while they own it. Release /
    // acquire counters also publish resets before the producer reuses a slot.
    // A slot holds the stream's own chunk of that index, or a spliced (foreign) chunk until replayed.
    std::unique_ptr<std::array<Chunk, ChunkCount>> own;
    std::array<Chunk*, ChunkCount> slots {};
    uint64_t sequence = 0;
    uint64_t submit_fold_boundary = 0; // producer only
    alignas(64) std::atomic<uint64_t> published {0};
    alignas(64) std::atomic<uint64_t> completed {0};
    alignas(64) std::atomic<bool> sleeping {false};
    std::atomic<bool> stopping {false};
    std::thread worker;
    // A passive producer's state: the open chunk, the segment's closed chunks.
    ChunkPool* pool = nullptr;
    Chunk* open = nullptr;
    std::vector<Chunk*> closed;
    uint64_t segment_commands = 0;
    bool segment_failed = false;

    void ClosePassive() {
        if (open && open->count != 0) {
            closed.push_back(open);
            open = nullptr;
        }
    }
    // Never a direct fallback (that would reorder): a command that does not fit closes the chunk and retries; one
    // refused again (larger than a chunk, an extension chain) fails the segment.
    template<typename Capture>
    bool EnqueuePassive(Capture& capture) {
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (!open) open = pool->Get();
            auto& chunk = *open;
            Writer writer(chunk);
            capture(writer);
            if (writer.valid && writer.command != End) {
                if (chunk.last != End)
                    reinterpret_cast<Header*>(chunk.data.data() + chunk.last)->next = writer.command;
                else chunk.first = writer.command;
                chunk.last = writer.command;
                chunk.used = writer.used;
                ++chunk.count;
                ++segment_commands;
                if (chunk.count >= 128 || chunk.used >= ChunkBytes * 3 / 4) ClosePassive();
                return true;
            }
            if (!writer.overflow || chunk.count == 0) break;
            ClosePassive();
        }
        segment_failed = true;
        return false;
    }

    void WaitForRoom() {
        auto done = completed.load(std::memory_order_acquire);
        if (sequence - done < ChunkCount) return;
        const RecordWait waiting;
        for (; sequence - done >= ChunkCount; done = completed.load(std::memory_order_acquire))
            completed.wait(done, std::memory_order_acquire);
    }
    template<bool Submit = false>
    void Publish(FoldSubmitInfo* submit = nullptr) {
        WaitForRoom();
        auto& chunk = *slots[sequence % ChunkCount];
        if (chunk.count == 0) return;
        if constexpr (Submit) {
            const std::lock_guard lock(g_graphics_publication_mutex);
            submit->graphics_boundary = g_graphics_boundary;
            submit->ends_fold |= g_direct_graphics_submits != 0;
            published.store(++sequence, std::memory_order_seq_cst);
        } else {
            published.store(++sequence, std::memory_order_seq_cst);
        }
        // Wake the worker only when it said it blocks (both sides seq_cst): a wake-up is a
        // system call on Windows, and the worker is usually busy or still spinning.
        if (sleeping.load(std::memory_order_seq_cst)) published.notify_one();
    }
    // The replay thread must not land on an efficiency core: every GPU wait includes its
    // latency.  KYTY_RECORDING_CPUS (e.g. "1,2,3,6,7") restricts it; re-applied now and
    // then because the launcher assigns one mask to every game thread at startup.
    static void KeepAffinity() {
        static const char* const list = std::getenv("KYTY_RECORDING_CPUS");
        if (list) LocalPlatform::PinThreadToCpuList(list);
#if defined(_WIN32) && defined(KYTY_LOCAL_VULKAN_RECORDING)
        if (LocalPlatform::g_irq_avoid_ever_armed.load(std::memory_order_acquire)) {
            static thread_local bool was_mode4 = false;
            // Record follows the mode at its own existing pin points. The ideal processor is a scheduler hint.
            // A 4 -> 0 -> 4 between pins leaves Record in the placement of the mode it ends in.
            // Folded batches can delay a placement update until Record's next actual pin. No end state is wrong.
            const bool active = kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4;
            if (was_mode4 == active) return;
            if (active) {
                LocalPlatform::ApplyRecordingIrqAvoid(LocalPlatform::g_recording_irq_placement);
                was_mode4 = true;
            } else if (LocalPlatform::RestoreIrqIdeal()) {
                was_mode4 = false;
            }
        }
#endif
    }
    void CompleteChunk(uint64_t& current) {
        const size_t index = current % ChunkCount;
        auto& chunk = *slots[index];
        if (chunk.pool) {
            slots[index] = &(*own)[index];
            chunk.pool->Put(&chunk);
        } else {
            chunk.used = 0;
            chunk.count = 0;
            chunk.first = chunk.last = End;
        }
        completed.store(++current, std::memory_order_release);
        completed.notify_one();
    }
    // Look ahead through the published recording packets. A submit is always the last packet in its chunk.
    bool ReplayFold(uint64_t& current, const SubmitPacket& first) {
        const auto& info = first.FoldInfo();
        const uint64_t ready = published.load(std::memory_order_acquire);
        // Each submit publishes its chunk, so a second submit needs at least one more published chunk.
        if (ready == current + 1 || first.commands >= MaxFoldCommands || info.ends_fold) return false;

        if (first.replay->has_wait(first.copied[0], *info.timeline)) return false;

        std::array<Segment, MaxFoldCommands> requests;
        std::array<uint64_t, MaxFoldCommands + 1> request_chunks;
        requests[0] = first.copied[0];
        request_chunks[0] = current;
        size_t count = 1;
        size_t commands = first.commands;
        for (uint64_t scan = current + 1; scan < ready; ++scan) {
            const auto& chunk = *slots[scan % ChunkCount];
            for (auto pos = chunk.first; pos != End;) {
                const auto* header = reinterpret_cast<const Header*>(chunk.data.data() + pos);
                if (header->kind == PacketKind::Barrier) goto planned;
                if (header->kind == PacketKind::Submit) {
                    const auto& candidate = *reinterpret_cast<const SubmitPacket*>(chunk.data.data() + header->payload);
                    const auto& next_info = candidate.FoldInfo();
                    if (candidate.replay != first.replay || next_info.epoch != info.epoch ||
                        next_info.timeline != info.timeline || next_info.boundary != info.boundary ||
                        next_info.graphics_boundary != info.graphics_boundary ||
                        commands + candidate.commands > MaxFoldCommands ||
                        !first.replay->can_append(requests[count - 1], candidate.copied[0])) goto planned;
                    request_chunks[count] = scan;
                    requests[count++] = candidate.copied[0];
                    commands += candidate.commands;
                    if (commands == MaxFoldCommands || next_info.ends_fold ||
                        first.replay->has_wait(candidate.copied[0], *info.timeline)) goto planned;
                }
                pos = header->next;
            }
        }
    planned:
        if (count == 1) return false;

        request_chunks[count] = request_chunks[count - 1] + 1;
        size_t request = 0;
        for (uint64_t replay = current; replay < request_chunks[count]; ++replay) {
            auto& chunk = *slots[replay % ChunkCount];
            if (chunk.pool) t_splice_target = chunk.target;
            // ReplayChunk already executed the packets before the first submit.
            for (auto pos = replay == current ? chunk.last : chunk.first; pos != End;) {
                const auto* header = reinterpret_cast<const Header*>(chunk.data.data() + pos);
                auto* payload = chunk.data.data() + header->payload;
                if (header->kind == PacketKind::Submit) {
                    InvalidateRawState();
                    first.replay->prepare(requests[request++], original);
                } else {
                    header->execute(payload);
                }
                if (header->destroy) header->destroy(payload);
                pos = header->next;
            }
        }
        struct Completion {
            Stream& stream;
            uint64_t& current;
            const uint64_t* request_chunks;
        } completion {*this, current, request_chunks.data()};
        first.replay->fold({requests.data(), count}, original, *info.timeline, info.epoch,
                           {&completion, [](void* context, size_t next_request) {
                               auto& done = *static_cast<Completion*>(context);
                               // The recording-only chunks before the next submit have already replayed.
                               const auto end = done.request_chunks[next_request];
                               while (done.current < end) done.stream.CompleteChunk(done.current);
                           }});
        return true;
    }
    template<bool Fold>
    void ReplayChunk(Chunk* chunk, uint64_t& current) {
        if (chunk->pool) t_splice_target = chunk->target;
        for (auto offset = chunk->first; offset != End;) {
            const auto* header = reinterpret_cast<const Header*>(chunk->data.data() + offset);
            auto* payload = chunk->data.data() + header->payload;
            if constexpr (Fold) {
                if (header->kind == PacketKind::Submit) {
                    const auto& submit = *reinterpret_cast<const SubmitPacket*>(payload);
                    if (IsCurrentSubmitFold(submit.FoldInfo().epoch) && ReplayFold(current, submit)) return;
                }
            }
            header->execute(payload);
            if (header->destroy) header->destroy(payload);
            offset = header->next;
        }
        CompleteChunk(current);
    }
    void Run() {
        executing = this;
        LocalPlatform::SetThreadName("Kyty.Record");
        KeepAffinity();
        constexpr uint64_t RepinInterval = 4096;
        uint64_t next_pin = RepinInterval;
        uint64_t current = 0;
        for (;;) {
            if (current >= next_pin) {
                KeepAffinity();
                next_pin = current + RepinInterval;
            }
            for (auto ready = published.load(std::memory_order_acquire); current == ready;
                 ready = published.load(std::memory_order_acquire)) {
                const uint32_t spins = SpinGate::Open(true) ? kyty_local_recording_spin.load(std::memory_order_relaxed) : 0;
                for (uint32_t spin = 0; spin < spins && ready == current; ++spin) {
                    _mm_pause();
                    ready = published.load(std::memory_order_acquire);
                }
                if (ready != current) break;
                LiveCounters::Add(LiveCounters::RecordSleeps);
                sleeping.store(true, std::memory_order_seq_cst);
                if (published.load(std::memory_order_seq_cst) == current)
                    published.wait(current, std::memory_order_seq_cst);
                sleeping.store(false, std::memory_order_relaxed);
            }
            if (stopping.load(std::memory_order_acquire)) return;
            // Tagged submits sample or activate the epoch before publishing. After acquiring
            // publication above, the gate sees that activation or a later state.
            auto* chunk = slots[current % ChunkCount];
            if (SubmitFoldEnabled()) {
                // Fold-specific submits always close their chunk.
                const auto* last = reinterpret_cast<const Header*>(chunk->data.data() + chunk->last);
                if (last->kind == PacketKind::Submit) {
                    ReplayChunk<true>(chunk, current);
                    continue;
                }
            }
            ReplayChunk<false>(chunk, current);
        }
    }
};

thread_local std::unique_ptr<Stream> producer;
thread_local std::unique_ptr<Stream> passive_producer; // (PassiveProducerScope)
Stream* RecordingStream() {
    if (producer && kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0) {
        producer->was_recording = true;
        return producer.get();
    }
    // (A passive producer records whatever the mode: a direct call would reach the driver unordered.)
    if (t_passive) [[unlikely]] return passive_producer.get();
    return nullptr;
}
// A direct call on a passive producer's thread that names the placeholder (a command it could not record): it never
// reaches the driver, and the segment fails.
[[gnu::noinline]] void FailPassiveSegment() {
    if (t_passive) passive_producer->FailSegment();
}
void InvalidateRawState() {
    if (executing) executing->state_epoch.fetch_add(1, std::memory_order_relaxed);
}
[[gnu::noinline]] void BeforeDirect() {
    if (producer) {
        // Diagnostic (live timecensus, source DirectDrain): direct calls made while packets are pending.
        if (producer->Pending()) KYTY_TIME_CENSUS(DirectDrain, 0);
        producer->BeforeDirect();
    } else if (t_passive) {
        passive_producer->BeforeDirect();
    }
}

// Work calls recorded by this thread (read by the render thread's barrier dedupe, which orders
// only its own command stream): a plain increment, not a locked add on every command.
thread_local uint64_t g_writing_calls = 0; // (the work calls that change memory or images)
thread_local uint64_t g_sync_calls    = 0; // (and the others: barriers, render pass begins, timestamps)
thread_local uint64_t g_packets       = 0;
#include "weave-recording.inc"
#include "local-vulkan-recording.inc"
} // namespace

ChunkList::ChunkList(ChunkList&& other) noexcept
    : chunks(std::move(other.chunks)), commands(other.commands), failed(other.failed) {
    other.chunks.clear();
    other.commands = 0;
    other.failed = false;
}
ChunkList& ChunkList::operator=(ChunkList&& other) noexcept {
    if (this != &other) {
        Discard(*this);
        chunks = std::move(other.chunks);
        commands = other.commands;
        failed = other.failed;
        other.chunks.clear();
        other.commands = 0;
        other.failed = false;
    }
    return *this;
}
ChunkList::~ChunkList() { Discard(*this); }
void Discard(ChunkList& list) {
    for (auto* chunk : list.chunks) DiscardChunk(chunk);
    list.chunks.clear();
    list.commands = 0;
    list.failed = false;
}
VkCommandBuffer_T* PlaceholderCommandBuffer() { return Placeholder(); }
VkCommandBuffer_T* Resolve(VkCommandBuffer_T* command) { return ResolveCommand(command); }
bool Passive() { return t_passive; }
bool HasPassiveProducer() { return passive_producer != nullptr; }
void BeginSegment() {
    if (!passive_producer) return;
    passive_producer->BeginSegment();
    t_passive = true;
}
ChunkList EndSegment() {
    if (!passive_producer) return {};
    t_passive = false;
    return passive_producer->EndSegment();
}
bool Splice(ChunkList& list, VkCommandBuffer_T* target) {
    if (list.failed || t_passive || !producer) {
        Discard(list);
        return false;
    }
    if (list.chunks.empty()) return true;
    producer->Splice(list, target);
    return true;
}
uint64_t SplicedChunks() { return g_spliced_chunks.load(std::memory_order_relaxed); }
uint64_t PassiveChunksAllocated() { return g_passive_chunks.load(std::memory_order_relaxed); }
PassiveProducerScope::PassiveProducerScope() {
    if (producer || passive_producer) std::abort();
    passive_producer = std::make_unique<Stream>(Stream::PassiveTag {});
}
PassiveProducerScope::~PassiveProducerScope() {
    t_passive = false;
    passive_producer.reset();
}

void Install() {
    // Register the switch before live control starts.
    const auto& setting = LiveSwitch::Find("submit_fold", ModeToggle::Default("submit_fold"));
    if (setting.load(std::memory_order_relaxed) == 1)
        Detail::g_submit_fold_state.store(Detail::SubmitFoldPending, std::memory_order_relaxed);
    InstallDispatch();
}
const vk::detail::DispatchLoaderDynamic& DirectDispatch() { return original; }
void* ReplaceBindPipeline(void* entry) {
    auto* replaced = reinterpret_cast<void*>(original.vkCmdBindPipeline);
    original.vkCmdBindPipeline = reinterpret_cast<PFN_vkCmdBindPipeline>(entry);
    return replaced;
}
uint64_t WorkCalls() { return g_writing_calls + g_sync_calls; }
uint64_t RecordedWork() { return g_writing_calls + g_sync_calls + g_packets; }
uint64_t RecordedWrites() { return g_writing_calls + g_packets; }
void Drain() { if (producer) producer->Drain(); }
bool PacketsEnabled() {
    return (producer && kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0) || t_passive;
}
uint64_t StateEpoch() {
    return executing ? executing->state_epoch.load(std::memory_order_relaxed) : UINT64_MAX;
}
bool EnqueuePacket(ReplayPacket replay, std::span<const Segment> segments,
                   std::shared_ptr<const void> owner, bool fold_barrier) {
    if (!PacketsEnabled() || !replay || segments.size() > 8) return false;
    for (const auto& segment : segments)
        if (segment.size && !segment.data) return false;
    ++g_packets;
    auto* stream = RecordingStream();
    return stream->Enqueue([&](Writer& writer) {
        std::array<Segment, 8> copied {};
        for (size_t i = 0; i < segments.size(); ++i)
            copied[i] = {writer.Copy(segments[i].data, segments[i].size), segments[i].size};
        const auto count = segments.size();
        writer.Command([owner, copied, count, replay] { replay({copied.data(), count}, original); },
                       fold_barrier ? PacketKind::Barrier : PacketKind::Record);
    });
}
bool EnqueuePacketCopy(ReplayPacket replay, const Segment& segment, void** copy) {
    if (!PacketsEnabled() || !replay || (segment.size && !segment.data)) return false;
    ++g_packets;
    auto* stream = RecordingStream();
    return stream->Enqueue([&](Writer& writer) {
        // (Each attempt's: a retry after an overflow copies again into the next chunk, and the last attempt's stands.
        // The chunk's own bytes: mutable memory.)
        const Segment copied {writer.Copy(segment.data, segment.size), segment.size};
        *copy = const_cast<void*>(copied.data);
        writer.Command([copied, replay] { replay({&copied, 1}, original); });
    });
}
bool DeferredSubmitEnabled() {
    // (Never on a passive producer: a worker does not submit.)
    return producer && !t_passive && kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0 &&
        kyty_local_deferred_submit_mode.load(std::memory_order_relaxed) != 0;
}
uint64_t ActivateSubmitFold(uint64_t pending) {
    // Direct submits sample the gate under the same queue lock. An untracked submit must finish before
    // activation, and every direct submit after activation will use the publication boundary.
    const auto active = pending + 1;
    if (Detail::g_submit_fold_state.compare_exchange_strong(pending, active, std::memory_order_seq_cst)) return active;
    return 0; // A live switch can cancel this activation while it waits for the queue.
}
bool IsCurrentSubmitFold(uint64_t epoch) {
    return Detail::g_submit_fold_state.load(std::memory_order_seq_cst) == epoch;
}
bool SetSubmitFold(int64_t value) {
    const std::lock_guard lock(g_submit_fold_switch_mutex);
    if (!LiveSwitch::Set("submit_fold", value)) return false;
    auto& state = Detail::g_submit_fold_state;
    const auto current = state.load(std::memory_order_relaxed);
    constexpr auto mask = Detail::SubmitFoldPending | Detail::SubmitFoldActive;
    if ((current & mask) == 0 && value == 1)
        state.store(current + Detail::SubmitFoldPending, std::memory_order_seq_cst);
    else if ((current & mask) != 0 && value != 1)
        state.store((current | mask) + 1, std::memory_order_seq_cst);
    return true;
}
void DirectGraphicsSubmit::Begin() {
    const std::lock_guard lock(g_graphics_publication_mutex);
    ++g_graphics_boundary;
    ++g_direct_graphics_submits;
}
void DirectGraphicsSubmit::End() {
    const std::lock_guard lock(g_graphics_publication_mutex);
    --g_direct_graphics_submits;
    ++g_graphics_boundary;
}
void EndSubmitFold() {
    if (producer && SubmitFoldEnabled()) producer->EndSubmitFold();
}
bool EnqueueDeferred(ReplayPacket replay, std::span<const Segment> segments, bool publish, bool fold_barrier) {
    if (!DeferredSubmitEnabled() || !replay || segments.size() > 8) return false;
    auto* stream = RecordingStream();
    if (!stream) return false;
    const bool queued = stream->Enqueue([&](Writer& writer) {
        std::array<Segment, 8> copied {};
        for (size_t i = 0; i < segments.size(); ++i)
            copied[i] = {writer.Copy(segments[i].data, segments[i].size), segments[i].size};
        const auto count = segments.size();
        // A new command buffer may reuse a handle: drop raw-state reuse as a direct call would.
        writer.Command([copied, count, replay] { InvalidateRawState(); replay({copied.data(), count}, original); },
                       fold_barrier ? PacketKind::Barrier : PacketKind::Record);
    });
    if (queued && publish) stream->Flush();
    return queued;
}
bool EnqueueSubmit(const SubmitReplay& replay, const Segment& segment, uint32_t commands,
                   const Libs::Graphics::MasterSemaphore& timeline, uint64_t epoch, bool ends_fold) {
    if (!DeferredSubmitEnabled()) return false;
    auto* stream = RecordingStream();
    if (!stream) return false;
    return stream->Enqueue<true>([&](Writer& writer) {
        const FoldSubmitInfo info {&timeline, epoch, stream->SubmitFoldBoundary(), ends_fold};
        SubmitPacket packet {.commands = commands, .replay = &replay};
        packet.copied[0] = {writer.Copy(segment.data, segment.size), segment.size};
        auto* copied = const_cast<FoldSubmitInfo*>(writer.Copy(&info, 1));
        packet.copied[1] = {copied, sizeof(info)};
        writer.Command(packet, PacketKind::Submit);
        return copied;
    });
}
namespace {
std::atomic<uint64_t> g_deferred_queued {0};
std::atomic<uint64_t> g_deferred_done {0};
} // namespace
void NoteDeferredSubmitQueued() { g_deferred_queued.fetch_add(1, std::memory_order_release); }
uint64_t DeferredSubmitsQueued() { return g_deferred_queued.load(std::memory_order_acquire); }
uint64_t DeferredSubmitsDone() { return g_deferred_done.load(std::memory_order_acquire); }
void NoteDeferredSubmitDone(size_t requests) {
    g_deferred_done.fetch_add(requests, std::memory_order_release);
    g_deferred_done.notify_all();
}
void WaitDeferredSubmits() {
    const auto target = g_deferred_queued.load(std::memory_order_acquire);
    for (auto done = g_deferred_done.load(std::memory_order_acquire); done < target;
         done = g_deferred_done.load(std::memory_order_acquire)) g_deferred_done.wait(done, std::memory_order_acquire);
}
void ReplayInline(ReplayPacket replay, std::span<const Segment> segments) {
    // A passive producer's packet that could not be recorded: executing it now would reach the driver out of
    // order (and on the placeholder); the segment fails instead.
    if (t_passive) [[unlikely]] {
        FailPassiveSegment();
        return;
    }
    if (producer) producer->BeforeDirect();
    replay(segments, original);
}
void InvalidateReplayState() { InvalidateRawState(); }
ProducerScope::ProducerScope() {
    if (producer) std::abort();
    producer = std::make_unique<Stream>();
}
ProducerScope::~ProducerScope() { producer.reset(); }
#include "weave-recording-api.inc"
} // namespace LocalVulkanRecording

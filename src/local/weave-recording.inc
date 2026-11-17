// A weave worker's calls recorded apart (vulkan-recording.h: BeginApart, EndApart, SpliceApartPart). Included into
// vulkan-recording.cpp inside its anonymous namespace, before the generated wrappers, which call ApartStream, Marker
// and FailApart. A worker has no producer: each call it records goes, numbered, into a buffer of its own (Apart) in
// chunks of this file's, and none of it reaches the driver until the GPU thread splices a part of the buffer into its
// own stream, where the recording worker replays it in order between the GPU thread's own packets. Self-contained:
// vulkan_recording_tests builds this file alone.

// The apart marker: the command buffer a worker records with, the address of an object of this file's (no driver hands
// it out). A part's replay writes the real command buffer over it in each call's closure.
char g_apart_marker;
inline VkCommandBuffer Marker() { return reinterpret_cast<VkCommandBuffer>(&g_apart_marker); }

// The buffers' chunks: a free list of their own, taken by the workers while they record and given back by whoever ends
// a buffer, mostly the recording worker after a last part. It keeps 512 at most (32 MiB). A chunk is made without
// zeroing its 64 KiB: Writer writes before anything reads.
constexpr size_t ApartPoolChunks = 512;
std::mutex g_apart_pool_mutex;
std::vector<Chunk*> g_apart_pool;
Chunk* TakeApartChunk() {
    {
        const std::lock_guard lock(g_apart_pool_mutex);
        if (!g_apart_pool.empty()) {
            auto* chunk = g_apart_pool.back();
            g_apart_pool.pop_back();
            return chunk;
        }
    }
    return new Chunk;
}
void GiveApartChunk(Chunk* chunk) {
    chunk->used = 0;
    chunk->count = 0;
    chunk->first = chunk->last = End;
    {
        const std::lock_guard lock(g_apart_pool_mutex);
        if (g_apart_pool.size() < ApartPoolChunks) {
            g_apart_pool.push_back(chunk);
            return;
        }
    }
    delete chunk;
}
} // namespace

// The header's class, defined at its namespace's scope (as Chunk is above) for the wrappers below to record into. Each
// stage of a buffer's life is one thread's, and the hand-over between them orders it: the worker records and proves it
// (Enqueue, Proven) until EndApart hands it over; then the GPU thread takes its parts (Take), and the recording worker
// replays them (Replay) in the order they were taken, which is the stream's. The last owner to let go (a packet's
// replay, the GPU thread, the worker) destroys what no part took.
class Apart {
public:
    Apart() = default;
    Apart(const Apart&) = delete;
    Apart& operator=(const Apart&) = delete;
    ~Apart() { Release(); }

    // A call appended, numbered `calls`. One that does not fit closes the chunk and retries in a new one; one that
    // cannot be recorded (larger than a chunk, an unknown extension chain) fails the buffer. Never a direct call: that
    // would reach the driver on the marker, and out of order.
    template<typename Capture>
    void Enqueue(Capture& capture) {
        if (failed) return;
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (chunks.empty() || attempt != 0) chunks.push_back(TakeApartChunk());
            auto& chunk = *chunks.back();
            Writer writer(chunk);
            capture(writer);
            if (writer.valid && writer.command != End) {
                if (chunk.last != End)
                    reinterpret_cast<Header*>(chunk.data.data() + chunk.last)->next = writer.command;
                else chunk.first = writer.command;
                chunk.last = writer.command;
                chunk.used = writer.used;
                ++chunk.count;
                ++calls;
                return;
            }
            if (!writer.overflow || chunk.count == 0) break;
        }
        failed = true;
    }
    void Fail() { failed = true; }
    [[nodiscard]] uint64_t Calls() const { return calls; }
    // Every call's closure starts with the marker, so what a replay overwrites is the call's command buffer:
    // Writer::Command places the closure at the header's payload, and a wrapper's closure captures commandBuffer first
    // (its first use). A call made through any other handle, or a closure laid out otherwise, fails the proof.
    [[nodiscard]] bool Proven() const {
        if (failed) return false;
        const VkCommandBuffer marker = Marker();
        for (const auto* chunk : chunks) {
            for (auto offset = chunk->first; offset != End;) {
                const auto* header = reinterpret_cast<const Header*>(chunk->data.data() + offset);
                VkCommandBuffer first = VK_NULL_HANDLE;
                if (header->payload > ChunkBytes - sizeof(first)) return false;
                std::memcpy(&first, chunk->data.data() + header->payload, sizeof(first));
                if (first != marker) return false;
                offset = header->next;
            }
        }
        return true;
    }
    // The GPU thread: calls [begin, end) taken as the next part, which starts at or after the last part's end; after a
    // `last` part nothing more is taken.
    [[nodiscard]] bool Take(uint64_t begin, uint64_t end, bool last) {
        if (begin < taken || begin >= end || end > calls) return false;
        taken = last ? calls : end;
        return true;
    }
    // The recording worker, a part's replay. It starts from the call the last part stopped at: the calls before `begin`
    // are destroyed unexecuted, [begin, end) are executed with `target` written over the marker, and with `last` the
    // rest are destroyed and the chunks given back. A wrong starting call would run wrong calls.
    void Replay(VkCommandBuffer target, uint64_t begin, uint64_t end, bool last) {
        Walk(last ? calls : end, target, begin, end);
        if (last) Release();
    }

private:
    std::vector<Chunk*> chunks;
    uint64_t calls = 0;
    bool failed = false;
    uint64_t taken = 0; // (the GPU thread's: the end of the parts taken)
    // (The recording worker's: the next call to replay or destroy, numbered `walked`, in chunks[walk_chunk] at
    // walk_offset, End for that chunk's first call.)
    uint64_t walked = 0;
    size_t walk_chunk = 0;
    uint32_t walk_offset = End;

    // Calls from the next one up to `stop`, each executed if in [begin, end), and each destroyed.
    void Walk(uint64_t stop, VkCommandBuffer target, uint64_t begin, uint64_t end) {
        while (walked < stop && walk_chunk < chunks.size()) {
            auto& chunk = *chunks[walk_chunk];
            const auto offset = walk_offset != End ? walk_offset : chunk.first;
            if (offset == End) { // (a chunk taken for a call that fit in none: the buffer failed)
                ++walk_chunk;
                continue;
            }
            const auto* header = reinterpret_cast<const Header*>(chunk.data.data() + offset);
            auto* payload = chunk.data.data() + header->payload;
            if (walked >= begin && walked < end) {
                std::memcpy(payload, &target, sizeof(target));
                header->execute(payload);
            }
            if (header->destroy) header->destroy(payload);
            ++walked;
            walk_offset = header->next;
            if (walk_offset == End) ++walk_chunk;
        }
    }
    // What no part took destroyed unexecuted, and the chunks given back.
    void Release() {
        Walk(calls, VK_NULL_HANDLE, 0, 0);
        for (auto* chunk : chunks) GiveApartChunk(chunk);
        chunks.clear();
    }
};

namespace {
// Threads with a buffer open (BeginApart to EndApart). While there are none, ApartStream is one relaxed load, and a
// thread with no stream of its own (the upload worker, or any thread while KYTY_VULKAN_RECORDING=0) never reads its
// thread-local. A worker always sees its own increment.
std::atomic<uint32_t> g_apart_threads {0};
// The calling thread's open buffer, which every call recorded apart tests (a plain pointer), and its owner, which also
// gives the count back if the thread ends with a buffer open.
thread_local Apart* t_apart = nullptr;
struct ApartOwner {
    std::shared_ptr<Apart> buffer;
    ApartOwner() = default;
    ApartOwner(const ApartOwner&) = delete;
    ApartOwner& operator=(const ApartOwner&) = delete;
    ~ApartOwner() {
        if (buffer) g_apart_threads.fetch_sub(1, std::memory_order_relaxed);
    }
};
thread_local ApartOwner t_apart_owner;
// The open buffer of a thread whose call has no stream to go to (RecordingStream() null), else null.
inline Apart* ApartStream() {
    if (g_apart_threads.load(std::memory_order_relaxed) == 0) [[likely]] return nullptr;
    return t_apart;
}
// A direct call on the marker (a command with no recorded form): it never reaches the driver, and the calling thread's
// open buffer, if it has one, fails.
[[gnu::noinline]] void FailApart() {
    if (t_apart) t_apart->Fail();
}
// A spliced part: SpliceApartPart's packet, whose owner keeps the buffer alive until its replay ends.
struct ApartPart {
    Apart* buffer = nullptr;
    VkCommandBuffer target = VK_NULL_HANDLE;
    uint64_t begin = 0, end = 0;
    bool last = false;
};
void ReplayApartPart(std::span<const Segment> segments, const vk::detail::DispatchLoaderDynamic&) {
    const auto& part = *static_cast<const ApartPart*>(segments[0].data);
    part.buffer->Replay(part.target, part.begin, part.end, part.last);
}

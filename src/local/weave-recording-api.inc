// The apart recording's public API (vulkan-recording.h), included into vulkan-recording.cpp after its anonymous
// namespace closes, so that these are the header's LocalVulkanRecording:: functions.
VkCommandBuffer_T* ApartMarker() { return Marker(); }
void BeginApart() {
    // (A buffer still open is destroyed, its calls unexecuted: the thread starts over.)
    if (t_apart == nullptr) g_apart_threads.fetch_add(1, std::memory_order_relaxed);
    t_apart = nullptr;
    t_apart_owner.buffer = std::make_shared<Apart>();
    t_apart = t_apart_owner.buffer.get();
}
std::shared_ptr<Apart> EndApart(bool keep) {
    if (t_apart == nullptr) return nullptr;
    t_apart = nullptr;
    auto buffer = std::move(t_apart_owner.buffer);
    g_apart_threads.fetch_sub(1, std::memory_order_relaxed);
    if (!keep || !buffer->Proven()) return nullptr; // (destroyed here: nothing of it executed)
    return buffer;
}
bool SpliceApartPart(const std::shared_ptr<Apart>& buffer, VkCommandBuffer_T* target, uint64_t begin, uint64_t end,
                     bool last) {
    if (!buffer || target == VK_NULL_HANDLE || !buffer->Take(begin, end, last)) return false;
    // (Another thread recorded them: to this thread's stream they are writing work recorded now, which its barrier
    // dedupe must see.)
    g_writing_calls += end - begin;
    const ApartPart part {buffer.get(), target, begin, end, last};
    const Segment segment {&part, sizeof(part)};
    if (LocalVulkanRecording::EnqueuePacket(ReplayApartPart, {&segment, 1}, buffer)) {
        // (Published now: a part is hundreds of calls in one entry, which would otherwise wait for 127 more.)
        if (producer) producer->Flush();
    } else {
        LocalVulkanRecording::ReplayInline(ReplayApartPart, {&segment, 1});
    }
    return true;
}
uint64_t ApartCalls() { return t_apart ? t_apart->Calls() : 0; }

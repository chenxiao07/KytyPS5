#pragma once
// batch_map_syncs hoists a KernelBatchMap2 call's GPU-side unmaps to
// the call's start and skips each entry's own; a buffer or image the GPU thread registers over a hoisted range in that
// window would outlive the range's OS unmap, and a write guard armed there (KYTY_WRITE_GUARD) would never be dropped, its
// lift then restoring a stale protection over what maps there next. While a hoist is active, every registration and every
// new guard over a hoisted range marks it touched (BufferCache::ChangeRegister, TextureCache::RegisterImage,
// WriteGuard::Arm); the entry's own unmap then runs (UnmapGpuRange), which drops the range's guards.

#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace HoistWatch {
inline std::atomic<int>                           g_active {0};
inline std::mutex                                 g_mutex;
inline std::vector<std::pair<uint64_t, uint64_t>> g_ranges;
inline std::vector<uint8_t>                       g_touched;

inline void Begin(const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	std::lock_guard lock(g_mutex);
	g_ranges = ranges;
	g_touched.assign(ranges.size(), 0);
	g_active.store(1, std::memory_order_release);
}
inline void End() {
	g_active.store(0, std::memory_order_release);
	std::lock_guard lock(g_mutex);
	g_ranges.clear();
	g_touched.clear();
}
// A registration over [vaddr, vaddr + size): true when it overlaps an active hoisted range (marked touched).
inline bool NoteRegistration(uint64_t vaddr, uint64_t size) {
	if (g_active.load(std::memory_order_acquire) == 0 || size == 0) return false;
	std::lock_guard lock(g_mutex);
	bool hit = false;
	for (size_t i = 0; i < g_ranges.size(); ++i)
		if (vaddr < g_ranges[i].first + g_ranges[i].second && g_ranges[i].first < vaddr + size) g_touched[i] = 1, hit = true;
	return hit;
}
inline bool Touched(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(g_mutex);
	for (size_t i = 0; i < g_ranges.size(); ++i)
		if (g_touched[i] != 0 && vaddr < g_ranges[i].first + g_ranges[i].second && g_ranges[i].first < vaddr + size) return true;
	return false;
}
} // namespace HoistWatch

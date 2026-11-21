#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPURESOURCEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPURESOURCEMANAGER_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include <array>
#include <deque>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;
class GuestGpu;

class GpuResourceManager {
public:
	GpuResourceManager(GraphicContext& graphics, CommandScheduler& scheduler);
	~GpuResourceManager();
	KYTY_CLASS_NO_COPY(GpuResourceManager);

	[[nodiscard]] BufferCache&  GetBufferCache() { return m_buffer_cache; }
	[[nodiscard]] TextureCache& GetTextureCache() { return m_texture_cache; }
	[[nodiscard]] bool          HasReadWatchers(uint64_t vaddr, uint64_t size) const noexcept {
        return m_page_manager.HasReadWatchers(vaddr, size);
	}
	[[nodiscard]] const std::atomic<uint64_t>* ReadGranules() const noexcept { return m_page_manager.ReadGranules(); }
	void                        SetGpu(GuestGpu* gpu) noexcept { m_gpu = gpu; }
	// After a guest protection change: tracked pages get the tracker's protection back.
	void ReapplyProtection(uint64_t vaddr, uint64_t size) { m_page_manager.ReapplyProtection(vaddr, size); }
	// The pages of a range the trackers protect (PageManager::WatchedSpan).
	[[nodiscard]] bool WatchedSpan(uint64_t vaddr, uint64_t size, uint64_t* first, uint64_t* end) const noexcept {
		return m_page_manager.WatchedSpan(vaddr, size, first, end);
	}
	void SyncProtection(uint64_t vaddr, uint64_t size) { m_page_manager.SyncProtection(vaddr, size); }
	// KYTY_READ_GUARD: the page's trackers' protection after its read guard's fault was served (PageManager).
	void SettleProtection(uint64_t vaddr) { m_page_manager.SettleProtection(vaddr); }
	// (PageManager::WatchedRuns: for a holder of the guest address-space lock.)
	void WatchedRuns(uint64_t vaddr, uint64_t size, std::vector<PageManager::WatchedRun>* runs) const {
		m_page_manager.WatchedRuns(vaddr, size, runs);
	}

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	// Before a host write of the whole range: GPU-written bytes it replaces are dropped (BufferCache::InvalidateMemoryForWrite).
	[[nodiscard]] bool InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size);
	// Before a host write of the range: the images over it only (their bytes are rewritten, as InvalidateMemory does).
	[[nodiscard]] bool InvalidateImages(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	// The mapped parts of [vaddr, vaddr + size), added to `out`.
	void               MappedRanges(uint64_t vaddr, uint64_t size, RangeSet& out) const;
	void               MapMemory(uint64_t vaddr, uint64_t size);
	// `releasing`: the caller unmaps or decommits the range right after (munmap, free,
	// memory pool decommit), not a range about to receive a mapping.
	void               UnmapMemory(uint64_t vaddr, uint64_t size, bool releasing = false);
	// batch_map_syncs: a guest batch's ranges in one GPU-thread command each way (KernelBatchMap2).
	void MapMemoryBatch(const std::vector<std::pair<uint64_t, uint64_t>>& ranges);
	void UnmapMemoryBatch(const std::vector<std::pair<uint64_t, uint64_t>>& ranges, bool releasing = false);
	// The game freed the range and keeps it mapped (LibKernel::Memory::NoteGameFree): queued for the GPU thread, whose
	// commands run before any later submission (and between the packets of the one under way), without waiting. The
	// buffers it holds go as for an unmap (BufferCache::DropFreedRange). KYTY_ENGINE_FREE_BUFFERS=0: nothing; =2: their
	// GPU-written bytes dropped instead of read back first.
	void               FreedByGame(uint64_t vaddr, uint64_t size);
	// GPU thread, while it has nothing else to do (GuestGpu::ThreadRun): a freed range whose copies completed written back
	// and let go (BufferCache::ServiceFreedReadback), one a call. FreedReadbackReady: one is; FreedReadbackSubmitted: one will
	// be with no new work (its copies were submitted).
	[[nodiscard]] bool FreedReadbackReady() { return m_buffer_cache.FreedReadbackReady(); }
	[[nodiscard]] bool FreedReadbackSubmitted() const { return m_buffer_cache.FreedReadbackSubmitted(); }
	void               ServiceFreedReadback();
	// upload_at_copy: from the streamer's thread, right after its copy into [vaddr, vaddr + size): the GPU thread uploads
	// the streamed levels there ahead of their first view (TextureCache::UploadLevelAhead). Never waits.
	void               PostLevelUploadAhead(uint64_t vaddr, uint64_t size);
	void               PrepareBda();
	void BeginAliasWrite() noexcept;
	void EndAliasWrite() noexcept;
	[[nodiscard]] uint64_t PreparationAliasEpoch() const noexcept;
	// Written only under m_mapped_ranges_mutex; readers compare it for change only,
	// so an acquire load needs no lock.
	[[nodiscard]] uint64_t MappingEpoch() const noexcept {
		return m_mapping_epoch.load(std::memory_order_acquire);
	}
	// Whether memory in [vaddr, vaddr + size) was unmapped after mapping epoch `epoch` (by 16 KiB
	// granule). GPU thread, as UnmapMemory.
	[[nodiscard]] bool UnmappedSince(uint64_t epoch, uint64_t vaddr, uint64_t size) const noexcept;
	// Changed's stamps (inside Changed, mode 4: the list UnmappedSince's per-item tests read there):
	// UnmappedSince's answer from the merged unmap list only, 1 or 0; -1 when the list cannot answer (no granule walk).
	[[nodiscard]] int UnmappedSinceListOnly(uint64_t epoch, uint64_t vaddr, uint64_t size) const noexcept {
		if (size == 0 || m_unmap_latest <= epoch) return 0;
		return UnmappedSinceByList(epoch, vaddr, size);
	}
	// The weave's (GPU thread): any of `sorted` ([begin, end) ranges, ascending by begin, disjoint) unmapped, or mapped
	// within a 16 KiB granule of it, after mapping epoch `epoch`, in one call (each unmap leaf looked up once, each map
	// since matched by a search). Maps past the map log's reach count as moved.
	[[nodiscard]] bool MappingMovedSince(uint64_t                                       epoch,
	                                     std::span<const std::pair<uint64_t, uint64_t>> sorted) const noexcept;
	// Its second half: any of `sorted` mapped, or a map within a 16 KiB granule of it, after `epoch`, by the log of the
	// latest maps (past its reach: true).
	[[nodiscard]] bool MappedSinceLogged(uint64_t                                       epoch,
	                                     std::span<const std::pair<uint64_t, uint64_t>> sorted) const noexcept;
	// UnmappedSince's question from the log of the latest unmaps, by overlap at 16 KiB granules: 0 none, 1 one did, 2
	// the log doesn't reach back to `epoch` (UnmappedSince answers then).
	[[nodiscard]] int UnmappedSinceLogged(uint64_t epoch, uint64_t vaddr, uint64_t size) const noexcept;

	bool PrepareBdaReadRanges(std::span<const GuestRange> ranges);
	// (KYTY_SPECULATE=4: EnsureBdaReadable for the ranges not known held at the registration epoch that stands.)
	void EnsureBdaReadableMemo(std::span<const GuestRange> ranges);
	// After each completed submission: its shaders' fault records and the image downloads.
	void               EndSubmission();
	// A flip the GPU makes (an end-of-pipe flip, all of Demon's Souls' frames): the collectors run at the end of its
	// submission (EndSubmission), as AdvanceFrame runs them for a CPU flip. Since collection moved to the flips
	// (81cd311) they never ran in the game, and a GPU under 32 GB filled up. (The texture cache's frame, which
	// ResolveOverlap's stale-image test counts, still advances with CPU flips only: in this game it never did.)
	void               NoteGpuFlip() { m_gpu_flip_pending = true; }
	// Once per CPU flip: the frame count and the garbage collection (its ages count frames).
	void               AdvanceFrame();

private:
	// The GPU thread's side of MapMemory / UnmapMemory (and of their batch forms).
	void MapOnGpuThread(uint64_t vaddr, uint64_t size);
	void UnmapOnGpuThread(uint64_t vaddr, uint64_t size, bool releasing);
	void                      RunCollectors();
	[[nodiscard]] bool        TryInvalidateCpuWriteWindow(uint64_t fault);
	// One window of the fault's mapped range: [window_begin, window_begin + window_size) within it.
	[[nodiscard]] bool        TryInvalidateCpuWriteWindow(uint64_t fault, uint64_t window_begin, uint64_t window_size);
	void                      SynchronizeDirtyBdaRegions(GuestRange range);
	void RefreshBdaRanges();
	// Fresh memory (src/local/fresh-memory.h): the never-written granules of a mapping the CPU and the GPU may use,
	// write-watched until their first write (MapMemory), unwatched when unmapped (UnmapMemory: `mark` counts them as
	// written, as when the mapping stays without the watch), a first write's (HandleFault: true when it was one) and a
	// host overwrite's (InvalidateMemoryForWrite) granules written. Each watched part is one granule's bytes in one
	// mapping, by its guest address; the parts of a granule (all its mappings) go together.
	void FreshWatch(uint64_t vaddr, uint64_t size);
	void FreshUnwatch(uint64_t vaddr, uint64_t size, bool mark);
	bool FreshFirstWrite(uint64_t vaddr);
	void FreshOverwrite(uint64_t vaddr, uint64_t size);
	void FreshUnwatchGranule(uint64_t granule); // (m_fresh_mutex held)
	struct FreshPart {
		uint64_t end     = 0;
		uint64_t granule = 0;
	};
	std::mutex                                          m_fresh_mutex;
	std::map<uint64_t, FreshPart>                       m_fresh_parts;    // by guest address
	std::unordered_map<uint64_t, std::vector<uint64_t>> m_fresh_granules; // the addresses of the parts of a granule
	uint64_t                                            m_fresh_report_flip = 0;
	PageManager               m_page_manager;
	CommandScheduler&         m_scheduler;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	std::atomic<uint64_t> m_mapping_epoch {1};
	// The mapping epoch of each 16 KiB granule's latest unmap, in 1 GiB leaves made at its first unmap
	// (written by UnmapMemory's GPU-thread part). The latest 64 unmaps before: streaming textures in
	// unmaps that many 64 KiB pool layers within frames, so a proof a few frames old counted as stale.
	static constexpr uint64_t UnmapGranuleBits = 14, UnmapLeafBits = 16;
	std::unordered_map<uint64_t, std::unique_ptr<uint64_t[]>> m_unmap_epochs;
	uint64_t m_unmap_latest = 0; // the latest unmap's mapping epoch
	// spec_unmap_list (under KYTY_SPECULATE=4 only, the mode read at each use): the unmaps
	// in epoch order, rounded out to granules (GPU
	// thread); `m_unmap_log_dropped` the newest epoch dropped from its front. The commit's Changed asks for one `since` (the
	// result's mapping epoch) over every range it read, ~40,000 a frame in an area transition, each a hash lookup and a
	// granule walk; from the merged ranges unmapped after `since` (built once a (since, latest unmap)) each is a span test
	// and a binary search, whatever the range's size. Only inside Changed (t_unmap_list_site): other callers ask with their
	// own epochs, which would rebuild the list between the commit's checks.
	struct UnmapLogEntry {
		uint64_t epoch, begin, end;
	};
	std::deque<UnmapLogEntry>                          m_unmap_log;
	uint64_t                                           m_unmap_log_dropped = 0;
	mutable std::vector<std::pair<uint64_t, uint64_t>> m_unmap_list;
	mutable uint64_t                                   m_unmap_list_since = UINT64_MAX, m_unmap_list_latest = 0;
	mutable bool                                       m_unmap_list_ok = false;
	// 1: unmapped (by the list), 0: not, -1: the list cannot answer.
	[[nodiscard]] int UnmappedSinceByList(uint64_t epoch, uint64_t vaddr, uint64_t size) const noexcept;
	// The spans whose part of RefreshBdaRanges' list (registered buffers' mapped parts) may have changed since it
	// last ran: mapping changes over a registered buffer and buffers (un)registered. Only their tracker regions are
	// collected again: the whole list (0.4 ms over ~2200 buffers) was, several times a frame while the game streams
	// in a new area. `m_bda_rebuild`: everything (at first, or after more changes than are kept). GPU thread.
	void                    NoteBdaSpan(uint64_t vaddr, uint64_t size);
	std::vector<GuestRange> m_bda_spans;
	bool                    m_bda_rebuild = true;
	std::vector<BufferCache::SyncRegionRequest> m_bda_region_requests, m_fresh_bda_region_requests;
	GuestGpu*                 m_gpu = nullptr;
	std::atomic<uint64_t> m_preparation_alias_epoch {uint64_t{1} << 32};
	bool                      m_fault_process_pending = false;
	bool                      m_bda_used              = false;
	bool                      m_gpu_flip_pending      = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPURESOURCEMANAGER_H_

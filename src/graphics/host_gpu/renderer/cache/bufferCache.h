#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/cache/tickClock.h"
#include "speculation-state.h"

#include <vk_mem_alloc.h>

#include <array>
#include <atomic>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <mutex>
#include <vector>

namespace ReadbackQueue { class Queue; }

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;
class GpuResourceManager;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = uint64_t {1} << (40 - CACHING_PAGEBITS);
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache, GpuResourceManager* resources = nullptr);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	// Before a host write of all of [vaddr, vaddr + size) (any thread): as InvalidateMemory, but the GPU-written bytes the
	// write replaces are dropped, not read back (only those of its first and last page outside it are).
	void                   InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size);
	// Live "sync" (any thread, racy by design: for a stalled GPU thread).
	void                   PrintSyncState();
	// Live "buffers <path>": the cached buffers as a tab-separated table (on the GPU thread).
	void                   WriteBufferReport(const char* path);
	// The staging ring's size (KYTY_STAGING_MB).
	[[nodiscard]] uint64_t StagingSize() const { return m_staging_buffer.Size(); }
	[[nodiscard]] bool TryInvalidateCpuWriteWindow(uint64_t fault, uint64_t begin, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] Buffer* GetBufferIfLive(BufferId id) {
		auto* buffer = m_slot_buffers.try_get(id);
		return buffer != nullptr && !buffer->is_deleted ? buffer : nullptr;
	}
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	// FindBuffer's speculative-translation form, out of line (bufferCache.cpp).
	[[gnu::noinline]] [[nodiscard]] BufferId FindBufferSpec(uint64_t vaddr, uint64_t size);
	// FindBuffer's registered buffer over the range, none registered: the null id (a speculative translation's
	// question whether ObtainBuffer gave the range's buffer).
	// (Out of line, bufferCache.cpp: its page-table entry read once, LoadOwner; a worker calls it, table-xpr.inc's held
	// index / indirect args lookup.)
	[[nodiscard]] BufferId RegisteredBuffer(uint64_t vaddr, uint64_t size);
	// The live buffer (registered, not deleted) that holds the range, and its id: FindBuffer's answer without making a
	// buffer, read only; {nullptr, BufferId {}} (which tests false) when none does. Its page-table entry is read once
	// (LoadOwner), so the id names the buffer returned. A weave worker asks too, while the GPU thread alone registers
	// (weave.h).
	[[nodiscard]] std::pair<const Buffer*, BufferId> FindLiveBuffer(uint64_t vaddr, uint64_t size) const;
	// A pending guest readback holds pages of the range: what ObtainBuffer finishes first for a lookup that is no plain
	// read (DrainGuestReadback). Read only. On a weave thread (Weave::OffGpuThread) it reads the copy the GPU thread
	// publishes at each change of the readback slots (PublishGuestReadbacks); on the GPU thread, the slots.
	[[nodiscard]] bool GuestReadbackOver(uint64_t vaddr, uint64_t size) const;
	// The weave's turn (GPU thread): a read range of a live buffer made current as ObtainBuffer makes it for a read:
	// the buffer touched, its CPU-dirty pages uploaded (EnsureBufferContents' body). Called only after the strand's
	// registration test passed.
	void SynchronizeAtTurn(uint64_t vaddr, uint64_t size);
	// Whether a buffer registered or unregistered after registration epoch `epoch` spans any of [vaddr, vaddr + size):
	// 0 none, 1 one does, 2 the log of the last 1,024 changes doesn't reach back. GPU thread, as every change is.
	[[nodiscard]] uint32_t RegisteredOverSince(uint64_t epoch, uint64_t vaddr, uint64_t size) const;
	void                   EnsureBufferContents(uint64_t vaddr, uint64_t size);
	// As EnsureBufferContents, for a texel buffer: an image GPU work wrote over the range is copied in (SynchronizeBuffer).
	void                   EnsureTexelContents(uint64_t vaddr, uint64_t size);
	// The start of each range shaders read through pointers held by a buffer with the guest's bytes (see the definition).
	void                   EnsureBdaReadable(std::span<const GuestRange> ranges);
	// Every page EnsureBdaReadable looks at for `range` (its head) held by a buffer now.
	[[nodiscard]] bool     BdaHeadHeld(const GuestRange& range) const;
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	// ObtainBuffer's speculative-translation form, out of line (bufferCache.cpp).
	[[gnu::noinline]] [[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferSpec(Spec::State& spec, uint64_t vaddr,
	                                                                              uint64_t size, bool is_written,
	                                                                              bool is_texel_buffer);
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	// CPU-written snapshots consumed as shader storage buffers. Each selected
	// ring keeps its own GPU retirement watches across runtime mode changes.
	[[nodiscard]] StreamBuffer& GetShaderUploadBuffer() noexcept;
	// The per-draw blocks of table draws (src/local/table-xpr.inc), read by their device addresses.
	[[nodiscard]] StreamBuffer& GetTableUploadBuffer() noexcept;
	[[nodiscard]] const Buffer* GetLodStatsBuffer() const noexcept { return &m_lod_stats_buffer; }
	void ReportLodStats(void* dst, uint32_t size, bool reset);
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	// The BDA page table descriptor writes take (the newest: GrowBdaTable). A set made with it records
	// BdaTableGeneration() read before it, and is made again once that moved.
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return m_bda_table.load(std::memory_order_acquire); }
	[[nodiscard]] static uint64_t BdaTableGeneration() noexcept {
		return s_bda_table_generation.load(std::memory_order_acquire);
	}
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	// `refillable`: set when the bytes went to the staging ring by the upload worker (RefillImageStaging may copy them
	// again while the upload is unsubmitted).
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size,
	                                                               bool* refillable = nullptr);
	// KYTY_ASYNC_REPROTECT: the write protection of the pages an image starts watching again (TextureCache's
	// DeferredImageProtects sets the page manager's deferred write-protect sink to this list around TrackImage), applied
	// by the upload worker ahead of an upload copy that reads those pages whole (the whole image: PushImageStagingCopies;
	// parts of it: StageImagePieces), else before any other read of them (ApplyImageProtects, and what is left when the
	// texture cache's upload returns): a CPU write before the protection is in the copy, one after it faults.
	[[nodiscard]] std::vector<PageManager::DeferredRange>* ImageProtectSink(PageManager& manager) noexcept {
		m_image_protect_manager = &manager;
		return &m_image_protects;
	}
	void ApplyImageProtects();
	// The guest bytes of [vaddr, vaddr + size) copied again by the upload worker to a staging copy ObtainBufferForImage
	// made (`ring` at `offset`): whose GPU copies are recorded but not submitted, so they upload the new bytes. False,
	// with nothing done, where ObtainBufferForImage would not stage them that way now.
	[[nodiscard]] bool RefillImageStaging(const Buffer& ring, uint64_t offset, uint64_t vaddr, uint64_t size);
	// Guest ranges staged one after another in one staging allocation of `total` bytes, each
	// at its `offset`. Returns nullptr (nothing staged) when a range holds GPU-written pages: the
	// caller then uploads the whole image instead.
	struct StagingPiece {
		uint64_t vaddr  = 0;
		uint64_t size   = 0;
		uint64_t offset = 0;
	};
	// (`why`: what refused, when nothing is staged - "gpu" or "map".)
	[[nodiscard]] std::pair<Buffer*, uint64_t> StageImagePieces(const std::vector<StagingPiece>& pieces,
	                                                            uint64_t total, const char** why = nullptr);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// CopyBuffer's copy on the CPU: only the destination pages whose bytes differ are written.
	void CopyGuestMemory(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size);
	// A commit's copy bytes written by a guarded write (speculation.cpp, GuardedCopy): the regions' epochs move, as
	// CopyGuestMemory's writes move them.
	void NoteGuardedCopyWrite(uint64_t vaddr, uint64_t size);
	// KYTY_SPEC_LABEL_KNOWN_WRITER (live spec_label_known_writer; on under KYTY_SPECULATE=4, 0 otherwise: mode-toggle.h;
	// set, it acts in either mode; the GPU thread only): the GPU thread's own write of guest bytes into watched memory
	// made through the backing view, its pages marked CPU-dirty with their write protection kept (CopyGuestMemory's
	// form), in place of the write fault that released the watch (and the upload that armed it again). True: written,
	// the caller writes nothing. False: not written, the caller writes as before (images over the range may have been
	// invalidated and its pages marked CPU-dirty: both exact before a write). `source_in_guest`: `data` may be guest
	// memory (a packet's bytes): read into a local copy first, so a fault reading it is served before the backing write's
	// locks are taken (GuardedWrite's order).
	[[nodiscard]] bool WriteGuestKnownWriter(uint64_t vaddr, const void* data, uint64_t size, bool source_in_guest = false);
	// Called after an identified small GPU copy; it never submits or publishes guest data.
	void ScheduleCopyFeedback(uint64_t vaddr, uint64_t size);
	// KYTY_EAGER_FEEDBACK (live eager_feedback, default off: fps flat in PR #300): GPU thread, the copy-feedback
	// snapshots whose tick has completed are written back to guest memory now (TryGuestReadFromFeedback, as a fault's
	// read would), before a guest read faults on them. `may_wait`: between guest packets, where a synchronous download
	// may run (the verify's).
	void WriteBackCompletedFeedback(bool may_wait);
	// GPU thread: retire a pending guest read before touching overlapping backing.
	// With no range, drain before mapping changes, unknown BDA access, or shutdown.
	// `gpu_write`: the range is about to be written by the GPU again; a read whose
	// copy has not started may be detached instead of awaited (KYTY_READBACK_DETACH).
	void DrainGuestReadback(uint64_t address = 0, uint64_t size = UINT64_MAX,
	                        bool gpu_read_only = false, bool gpu_write = false);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	// After a speculative translation's work was submitted (its commit obtained the ranges it writes as written
	// before it): the writes are that work's, which readbacks of the range wait for.
	void NoteSpeculativeWrite(uint64_t vaddr, uint64_t size) {
		InvalidateCopyFeedback(vaddr, size);
		NoteGpuWrite(vaddr, size);
	}
	// The tracker's answer whatever a speculative translation will upload (what a cache may keep).
	[[nodiscard]] bool IsRegionCpuDirty(uint64_t vaddr, uint64_t size) {
		return m_memory_tracker.IsRegionCpuModified(vaddr, size);
	}
	// The same, read only: no region manager made (an untracked region answers CPU-dirty, as a made one would).
	[[nodiscard]] bool IsRegionCpuDirtyNoCreate(uint64_t vaddr, uint64_t size) const {
		return m_memory_tracker.IsRegionCpuModifiedNoCreate(vaddr, size);
	}
	// The commit (on the GPU thread): a relocated reference's CPU-dirty range uploaded into its live buffer `id` as the
	// in-order draw's lookup of it (ObtainBuffer, read, not texel) does: SynchronizeBuffer, into the open command buffer
	// or its upload prologue, barriers its own. Refused with nothing done when the range holds GPU-written bytes (tracker
	// or exact ranges), a guest readback is pending over it, the texture offload must restore or wait for its bytes, or
	// the caller is not the GPU thread outside a speculative translation; StillDirty: the sync ran and pages stay
	// CPU-dirty (unmapped ones).
	enum class RelocUpload : uint8_t { Uploaded, Refused, StillDirty };
	[[nodiscard]] RelocUpload UploadRelocated(BufferId id, uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               ProcessFaultBuffer();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	struct SyncRegionRequest {
		uint64_t address = 0, size = 0, cpu_epoch = 0, registration_epoch = 0;
	};
	void SynchronizeRegionRequest(SyncRegionRequest& request);
	[[nodiscard]] uint64_t RegistrationEpoch() const { return m_registration_epoch.load(std::memory_order_acquire); }
	// Moves whenever a buffer is unregistered (deleted, or joined into a new one): Vulkan buffers a binding resolved
	// before it is unchanged are still the owners of their bytes. A join copies the old buffer's bytes into the new
	// one in the open command buffer, so a binding to the old one GPU work records after it reads or writes bytes
	// nothing reads again (the old buffer lives until the tick). The render thread's.
	[[nodiscard]] uint64_t UnregisterEpoch() const { return m_unregister_epoch; }
	// A buffer registered or unregistered in the tracker regions of the range since `epoch`. (Mode 4, quiet inserts
	// (QuietResolution): a buffer unregistered, or registered by a change that is not a quiet insert - the regions'
	// resolution epochs - and true for an epoch before the floor; see ChangeRegister for why that answers every caller.)
	[[nodiscard]] bool RegisteredSince(uint64_t vaddr, uint64_t size, uint64_t epoch) const {
		if (const auto* resolution = QuietResolution(); resolution != nullptr) [[unlikely]] {
			if (epoch < m_exact_floor.load(std::memory_order_acquire)) return true;
			for (auto region = vaddr / TRACKER_REGION_SIZE, last = (vaddr + size - 1) / TRACKER_REGION_SIZE; region <= last; ++region)
				if (ResolutionLoad(resolution, region) > epoch) return true;
			return false;
		}
		for (auto region = vaddr / TRACKER_REGION_SIZE, last = (vaddr + size - 1) / TRACKER_REGION_SIZE; region <= last; ++region)
			if (RegionRegistrationEpoch(region * TRACKER_REGION_SIZE) > epoch) return true;
		return false;
	}
	// spec_exact_reg_stop: as RegisteredSince, over the range's 16 KiB granules: a buffer registered or
	// unregistered over the range itself since `epoch` (KYTY_SPECULATE=4 marks them; the render thread's, as the
	// commit that asks). False also when nothing was marked (another mode). True for an epoch before the granules'
	// floor (MarkExactEpochsFromNow): a change before it may have gone unmarked (made outside mode 4).
	[[nodiscard]] bool RegisteredSinceExact(uint64_t vaddr, uint64_t size, uint64_t epoch) const {
		if (size == 0) return false;
		if (epoch < m_exact_floor.load(std::memory_order_acquire)) return true;
		// (Each leaf looked up once for its granules.)
		constexpr uint64_t leaf_mask = (uint64_t {1} << RegLeafBits) - 1;
		for (uint64_t g = vaddr >> RegGranuleBits, last = (vaddr + size - 1) >> RegGranuleBits; g <= last;) {
			const auto stop = std::min(last, g | leaf_mask);
			if (const auto it = m_reg_granule_epochs.find(g >> RegLeafBits); it != m_reg_granule_epochs.end()) {
				const auto* epochs = it->second.get();
				for (auto i = g & leaf_mask; i <= (stop & leaf_mask); ++i)
					if (epochs[i] > epoch) return true;
			}
			g = stop + 1;
		}
		return false;
	}
	// spec_regional_pending (KYTY_SPECULATE=4 only: the granule epochs): true when no buffer was registered or unregistered
	// over the range since `epoch`, proven at a bounded cost: its 4 MiB tracker regions' epochs (at most 64 of them, the
	// range inside the tracker's address space), then, when one of them moved, its 16 KiB granules' (at most 256). False
	// (not proven) otherwise. No lock; the render thread's, as RegisteredSinceExact.
	[[nodiscard]] bool RegistrationUnchangedSince(uint64_t vaddr, uint64_t size, uint64_t epoch) const {
		constexpr uint64_t MaxRegions = 64, MaxGranules = 256;
		if (size == 0 || vaddr + size < vaddr || vaddr + size > TRACKER_ADDRESS_SIZE) return false;
		const auto last = vaddr + size - 1;
		if (last / TRACKER_REGION_SIZE - vaddr / TRACKER_REGION_SIZE + 1 > MaxRegions) return false;
		if (!RegisteredSince(vaddr, size, epoch)) return true;
		return (last >> RegGranuleBits) - (vaddr >> RegGranuleBits) + 1 <= MaxGranules &&
		       !RegisteredSinceExact(vaddr, size, epoch);
	}
	// The granule epochs' floor: every change from the registration epoch now on is marked in them (RegisteredSinceExact
	// answers changed for an epoch before it). Called as mode 4's machinery starts (graphicsRun.cpp start_speculation: at
	// a KYTY_SPECULATE=4 boot, and on a live `mode 4` after the mode reads 4), on the thread that registers: before any
	// speculation exists to commit, but not before every thread that reads the mode (a weave worker runs on until the
	// pause after it, and so asks nothing the floor guards: RegionResolutionEpoch). A live switch from 4 to 0 and back
	// leaves the changes made in mode 0 unmarked.
	// (Quiet inserts: the regions' resolution epochs made here, at mode 4's first start, after the floor that guards their
	// readers - a mode-0 boot never makes them.)
	void MarkExactEpochsFromNow() {
		m_exact_floor.store(m_registration_epoch.load(std::memory_order_relaxed), std::memory_order_release);
		if (m_region_resolution_storage == nullptr) {
			m_region_resolution_storage = std::make_unique<std::atomic<uint64_t>[]>(TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE);
			m_region_resolution.store(m_region_resolution_storage.get(), std::memory_order_release);
		}
	}
	// The registration epoch of the last change of a buffer covering the tracker region of `vaddr`: a region
	// request's proof only depends on the buffers of its own region.
	[[nodiscard]] uint64_t RegionRegistrationEpoch(uint64_t vaddr) const {
		const auto region = vaddr / TRACKER_REGION_SIZE;
		return region < TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE ? m_region_registrations[region].load(std::memory_order_acquire)
		                                                           : 0;
	}
	// Quiet inserts (under KYTY_SPECULATE=4 only): a buffer registration over pages no buffer holds moves no resolution
	// epoch (ChangeRegister). The regions' resolution epochs while they act: null outside mode 4, and before mode 4's
	// machinery first started (MarkExactEpochsFromNow makes them). One relaxed load of the mode, then (mode 4) one of the
	// array, read at each change and each proof.
	[[nodiscard]] const std::atomic<uint64_t>* QuietResolution() const {
		if (::kyty_local_speculate_mode.load(std::memory_order_relaxed) != 4) return nullptr;
		return m_region_resolution.load(std::memory_order_acquire);
	}
	// The epoch a resolution of a range in the tracker region of `vaddr` is proven against (table-xpr.inc's held V#
	// entries: "no buffer covering its tracker regions moved since"): RegionRegistrationEpoch, or in mode 4
	// (QuietResolution) the region's last change that is not a quiet insert, never below the granules' floor (a change
	// before it, made outside mode 4, is not marked there: an entry proven before the floor is not current). Never a weave
	// worker's question: it can read the mode as 4 before the floor moved (MarkExactEpochsFromNow), and so asks
	// RegionRegistrationEpoch, which every change moves, quiet or not.
	[[nodiscard]] uint64_t RegionResolutionEpoch(uint64_t vaddr) const {
		const auto* resolution = QuietResolution();
		if (resolution == nullptr) return RegionRegistrationEpoch(vaddr);
		return std::max(ResolutionLoad(resolution, vaddr / TRACKER_REGION_SIZE), m_exact_floor.load(std::memory_order_acquire));
	}
	// Proofs that an earlier resolution still holds. Pure queries; TouchLiveBuffer
	// only refreshes the LRU slot exactly as ObtainBuffer's TouchBuffer would.
	// Zero means "cannot prove anything" (see MemoryTracker::CpuModificationEpoch).
	// A range spanning several 4 MiB tracker regions returns the sum of their
	// epochs: region managers are never destroyed and each epoch only grows, so
	// an unchanged sum proves that no region saw a new CPU-dirty mark.
	[[nodiscard]] bool DeferredProtectionPending(uint64_t vaddr) const {
		return m_memory_tracker.DeferredProtectionPending(vaddr);
	}
	// KYTY_SPEC_BDA_CLEAN_SKIP: MemoryTracker::CleanRangeEpoch (GpuResourceManager::SynchronizeDirtyBdaRegions).
	[[nodiscard]] uint64_t CleanRangeEpoch(uint64_t vaddr, uint64_t size) const {
		return m_memory_tracker.CleanRangeEpoch(vaddr, size);
	}
	[[nodiscard]] uint64_t CpuModificationEpoch(uint64_t vaddr, uint64_t size) const {
		if (!GuestRange {vaddr, size}.Valid()) return 0;
		constexpr uint64_t MaxRegions = 64;
		const uint64_t     first      = vaddr / TRACKER_REGION_SIZE;
		const uint64_t     last       = (vaddr + size - 1) / TRACKER_REGION_SIZE;
		if (last - first >= MaxRegions) return 0;
		uint64_t sum = 0;
		for (uint64_t region = first; region <= last; ++region) {
			const uint64_t begin = std::max(vaddr, region * TRACKER_REGION_SIZE);
			const uint64_t end   = std::min(vaddr + size, (region + 1) * TRACKER_REGION_SIZE);
			const uint64_t epoch = m_memory_tracker.CpuModificationEpoch(begin, end - begin);
			if (epoch == 0) return 0;
			sum += epoch;
		}
		return sum;
	}
	// A speculative translation's use of the buffer, when it is committed.
	void TouchSpeculated(BufferId id) {
		if (auto* buffer = m_slot_buffers.try_get(id)) TouchBuffer(*buffer);
	}
	// True when `id` names a live (registered, not deleted) game buffer that
	// covers the range and whose Vulkan handle is `handle`; refreshes its LRU slot.
	[[nodiscard]] bool TouchLiveBuffer(BufferId id, vk::Buffer handle, uint64_t vaddr,
	                                   uint64_t size) {
		auto* buffer = m_slot_buffers.try_get(id);
		if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(vaddr, size) ||
		    buffer->Handle() != handle) {
			return false;
		}
		TouchBuffer(*buffer);
		return true;
	}
	// The buffers no binding used for a while go (RunGarbageCollector, once a flip).
	void RetireStaleBuffers(uint64_t tick);
	// The buffers [vaddr, vaddr + size) holds whole go, after the guest unmapped it (GpuResourceManager::UnmapMemory); their
	// bytes.
	uint64_t DeleteUnmappedBuffers(uint64_t vaddr, uint64_t size);
	// The game's free blocks (its page allocator's, pages kept mapped) over [begin, end) added to `blocks` as (address, size):
	// false when not known (Loader::DemonsSoulsPageFree::FreeBlocksIn).
	using FreeBlocksQuery = bool (*)(uint64_t begin, uint64_t end, std::vector<std::pair<uint64_t, uint64_t>>* blocks);
	// What DropFreedRange's free block lists did: lists read, bytes cut off only thanks to them.
	struct FreeBlockSplits {
		uint64_t lists = 0, cut = 0;
	};
	// Then a buffer the range holds in part, mapped in no more than half of it now, is cut to its mapped parts; the bytes cut
	// off. With `free_blocks` (a freed range, DropFreedRange), the game's free blocks over it count as unmapped too.
	// (`read_back_over`: false, a buffer over other free blocks with GPU-written bytes stays as it is, nothing read back.)
	uint64_t SplitUnmappedBuffers(uint64_t vaddr, uint64_t size, FreeBlocksQuery free_blocks = nullptr,
	                              FreeBlockSplits* splits = nullptr, bool read_back_over = true);
	// The game freed [vaddr, vaddr + size) and keeps it mapped (GpuResourceManager::FreedByGame): its GPU-written bytes read
	// back (`read_back`) or dropped, then its buffers go as an unmapped range's do (the range, and the game's other free
	// blocks a buffer over it spans, count as unmapped for SplitUnmappedBuffers); the bytes of buffers let go. Nothing when
	// no buffer is registered there. A read-back range may be left pending instead (BeginFreedReadback): 0 then, its
	// buffers let go later (ServiceFreedReadback).
	uint64_t DropFreedRange(uint64_t vaddr, uint64_t size, bool read_back, FreeBlocksQuery free_blocks,
	                        FreeBlockSplits* splits);
	// GPU thread. The oldest freed range left pending by DropFreedRange whose copies completed: true when there is one.
	[[nodiscard]] bool FreedReadbackReady();
	// A pending freed range whose copies were submitted (FreedReadbackReady will be true without new work).
	[[nodiscard]] bool   FreedReadbackSubmitted() const;
	[[nodiscard]] size_t FreedReadbackPendingCount() const { return m_freed_readbacks.size(); }
	// GPU thread. That range's bytes written back where the GPU still owns their pages and wrote nothing since (and, with
	// KYTY_FREED_READBACK_BYTE_GATE, only those still GPU-written themselves: not ones given to the CPU since), then its
	// buffers let go as DropFreedRange's are (none where GPU-written bytes stayed in it: in use again); `let_go` adds the
	// bytes of buffers let go. False when no pending range was ready.
	bool ServiceFreedReadback(FreeBlocksQuery free_blocks, FreeBlockSplits* splits, uint64_t* let_go);
	// The pending freed ranges' counts since the last call (the 30 s report): ranges left pending, their GPU-written bytes,
	// ranges read back at the free instead (no room in the ring), ranges serviced, ranges whose buffers stayed (GPU-written
	// bytes left), bytes not written back (written by the GPU since, or given back to the CPU by a fault's readback), bytes
	// written back, and bytes on pages the GPU owns again that the byte gate did not write back (CPU-owned since the free,
	// KYTY_FREED_READBACK_BYTE_GATE; not in stale_bytes) or written back already by an older pending range over them.
	struct FreedReadbackCounts {
		uint64_t pending = 0, pending_bytes = 0, at_free = 0, serviced = 0, kept = 0, stale_bytes = 0;
		uint64_t written_bytes = 0, refused_bytes = 0, older_bytes = 0;
	};
	[[nodiscard]] FreedReadbackCounts TakeFreedReadbackCounts() { return std::exchange(m_freed_counts, {}); }
	// The mapped parts of registered buffers in [begin, end), adjacent ones merged.
	// The mapped parts of guest [vaddr, vaddr + size) (GpuResourceManager::MappedRanges), added to `out`.
	void MappedRanges(uint64_t vaddr, uint64_t size, RangeSet& out) const;
	void CollectMappedRegisteredRanges(const RangeSet& mapped, uint64_t begin, uint64_t end,
	                                   std::vector<RangeSet::Range>& ranges) const;
	// The spans of the buffers registered or unregistered since the last call, appended to `spans`; false when
	// more changed than it keeps (then anything may have).
	[[nodiscard]] bool TakeRegistrationSpans(std::vector<GuestRange>& spans);

	void               RunGarbageCollector(bool collect = true);
	// The video memory the game buffers in use that got system memory need back, with headroom (0: none): the texture
	// collector makes this room too (TextureCache::RunGarbageCollector). Updated every 60 flips.
	[[nodiscard]] uint64_t SpilledRoom() const noexcept { return m_spilled_room; }

private:
	friend struct BufferCacheTestAccess;

	// A game buffer in use made in system memory (Buffer::IsInSystemMemory: video memory over budget then) goes back to
	// video memory once there is room for it, one a call (RunGarbageCollector, every 60 flips). True when one moved.
	bool MoveSpilledBufferBack();
	// The game buffers' pool compacted (GraphicContext::game_buffer_pool): a pass of moves at a time, every flip while
	// one runs (RunGarbageCollector).
	void Defragment(uint64_t tick);
	void EndDefragmentation();
	struct DefragMove {
		BufferId source, destination;
	};
	VmaDefragmentationContext      m_defrag = nullptr;
	VmaDefragmentationPassMoveInfo m_defrag_pass {};
	std::vector<DefragMove>        m_defrag_moves; // (a pass's, by its moves' order; none: ignored)
	bool                           m_defrag_pass_open = false;
	std::future<VkResult>          m_defrag_ending;     // (the pass ending on a worker thread)
	std::vector<VmaAllocation>     m_defrag_free_later; // (moved allocations of new buffers destroyed meanwhile)
	uint64_t                       m_defrag_next_check = 0;
	uint64_t                       m_defrag_free_after = 0; // the pool's free bytes after the last compaction
	uint32_t                       m_defrag_passes     = 0;
	uint32_t                       m_defrag_ignored    = 0; // (moves of allocations no buffer holds: Defragment)
	double                         m_defrag_ms = 0, m_defrag_max_ms = 0; // the GPU thread's time in its passes

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	struct DownloadCopy;
	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 40, 16>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	static constexpr uint64_t               DOWNLOAD_ALIGNMENT = 64;
	[[nodiscard]] static constexpr uint64_t AlignDownload(uint64_t size) noexcept {
		return (size + DOWNLOAD_ALIGNMENT - 1) & ~(DOWNLOAD_ALIGNMENT - 1);
	}
	[[nodiscard]] static std::pair<uint64_t, uint64_t> DownloadEnvelope(const DownloadCopy& copy);
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	// The BDA page tables' entries [first, first + count): written (`entries`) or cleared (null), in every table that
	// holds them (m_bda_tables). GrowBdaTable: a table with `entries` entries at least replaces the newest.
	void WriteBdaEntries(uint64_t first, uint64_t count, const vk::DeviceAddress* entries);
	void GrowBdaTable(uint64_t entries);
	// GPU thread: InvalidateMemoryForWrite's work where GPU-written bytes are in the range.
	void DiscardGpuWrites(uint64_t vaddr, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);

public:
	// spec_buffer_ahead (under KYTY_SPECULATE=4 only; a V# range left pending is posted too): a speculation's thread
	// posts a range no buffer covers (PostAhead: once, at most 512 waiting); the
	// GPU thread registers the posted ranges nothing overlaps (ServeAhead, at each commit).
	void PostAhead(uint64_t vaddr, uint64_t size);
	// The table path's call site of a speculation's lookup (0 other): spec_ring_index's index ranges.
	static inline thread_local int t_ahead_site = 0;
	struct AheadSite {
		int old;
		explicit AheadSite(int site): old(t_ahead_site) { t_ahead_site = site; }
		~AheadSite() { t_ahead_site = old; }
	};
	void ServeAhead();

private:
	std::mutex                                 m_ahead_mutex;
	std::vector<std::pair<uint64_t, uint64_t>> m_ahead;
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	// Deleted buffers a speculation's packet may still read (Spec::PacketsPassed), destroyed once it passed.
	std::vector<std::pair<BufferId, Spec::PacketMarks>> m_retired;
	void                                                EraseRetired();
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	void UploadDirtyRanges(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size, const Buffer** ring = nullptr);
	// The open upload prologue's copies by device address (VK_NV_copy_memory_indirect): one command as it closes.
	std::vector<VkCopyMemoryIndirectCommandNV> m_prologue_copies;
	void                                       FlushPrologueCopies(vk::CommandBuffer command);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void DownloadBufferMemory(std::span<const DownloadCopy> copies);
	void WriteBackGpuOwned(uint64_t address, const uint8_t* data, uint64_t size, const char* source);
	void ReadMemoryOnGpu(uint64_t vaddr, uint64_t size, bool is_write);
	void FinishWriteReadback(uint64_t vaddr, uint64_t size);
	struct GuestReadback;
	std::shared_ptr<GuestReadback> BeginGuestReadback(uint64_t address, uint64_t size,
	                                                  bool* completed = nullptr);
	void CopyGuestReadback(const std::shared_ptr<GuestReadback>& request);
	void FinishGuestReadback(size_t slot, bool detach = false);
	static constexpr size_t GuestReadbackSlots = 32;
	std::array<std::shared_ptr<GuestReadback>, GuestReadbackSlots> m_guest_readbacks {};
	std::array<std::unique_ptr<Buffer>, GuestReadbackSlots> m_guest_downloads {};
	uint32_t m_active_guest_readbacks = 0;
	// While the weave is enabled: the pending readbacks published for a weave thread (bufferCache.cpp's copy, kept
	// outside the cache), and a weave thread's GuestReadbackOver read from it.
	void               PublishGuestReadbacks();
	[[nodiscard]] bool PublishedReadbackOver(uint64_t vaddr, uint64_t size) const;
	struct CopyFeedback;
	void InvalidateCopyFeedback(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool TryReadCopyFeedback(Buffer& buffer, uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool TryGuestReadFromFeedback(uint64_t vaddr, uint64_t size, bool is_write);
	std::unique_ptr<CopyFeedback> m_copy_feedback;
	// KYTY_EAGER_FEEDBACK: the snapshots scheduled, in tick order.
	struct EagerFeedback {
		size_t   slot;
		uint64_t address, size, tick;
	};
	std::deque<EagerFeedback> m_eager_feedback;
	uint64_t                  m_eager_verify_count = 0;
	[[nodiscard]] bool        VerifyEagerFeedback(const EagerFeedback& entry);
	// KYTY_FEEDBACK_BATCH: the open buffer's copy-feedback snapshots, recorded together as it closes (FlushFeedbackBatch).
	struct FeedbackCopy {
		vk::Buffer source;
		uint64_t   source_offset, destination_offset, size;
	};
	std::vector<FeedbackCopy> m_feedback_batch;
	void                      FlushFeedbackBatch(CommandBuffer& command);
	bool TryReportLodStatsOnGpu(uint64_t address, bool reset);
	vk::Pipeline m_lod_pack_pipeline = nullptr;
	vk::PipelineLayout m_lod_pack_layout = nullptr;
	vk::DescriptorSetLayout m_lod_pack_descriptors = nullptr;

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_lod_stats_buffer;
	// The BDA page tables: the newest last (m_bda_table), the ones it replaced kept and written alike - sets made before
	// it grew may still read them, never a freed buffer or a stale entry (GrowBdaTable).
	std::vector<std::unique_ptr<Buffer>>              m_bda_tables;
	std::atomic<Buffer*>                              m_bda_table {nullptr};
	bool                                              m_bda_zeroed = false; // the first table (WriteBdaEntries)
	static inline std::atomic<uint64_t>               s_bda_table_generation {0};
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	// (Read by a speculative translation's thread: src/graphics/guest_gpu/speculation.h.)
	std::atomic<uint64_t> m_registration_epoch {1};
	uint64_t              m_unregister_epoch {1};
	// Per tracker region, see RegionRegistrationEpoch.
	std::unique_ptr<std::atomic<uint64_t>[]> m_region_registrations =
	    std::make_unique<std::atomic<uint64_t>[]>(TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE);
	// Quiet inserts (ChangeRegister): the value of the last registration change, every change's (GPU thread only). Equal
	// to the published epoch whenever no quiet insert came after the last other change (in mode 0 from its first change
	// on: none there is quiet).
	uint64_t m_change_counter {1};
	// Quiet inserts: per tracker region, the value of the last change over it that is not a quiet insert, written in mode
	// 4 only, from the granules' floor on (QuietResolution, RegionResolutionEpoch). Made at mode 4's first start
	// (MarkExactEpochsFromNow), never freed before the cache: a mode-0 boot has none.
	std::unique_ptr<std::atomic<uint64_t>[]> m_region_resolution_storage;
	std::atomic<std::atomic<uint64_t>*>      m_region_resolution {nullptr};
	[[nodiscard]] static uint64_t ResolutionLoad(const std::atomic<uint64_t>* resolution, uint64_t region) {
		return region < TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE ? resolution[region].load(std::memory_order_acquire) : 0;
	}
	// spec_exact_reg_stop: per 16 KiB granule, the epoch of the last registration change over it (RegisteredSinceExact).
	static constexpr uint64_t                                 RegGranuleBits = 14, RegLeafBits = 16;
	std::unordered_map<uint64_t, std::unique_ptr<uint64_t[]>> m_reg_granule_epochs;
	std::atomic<uint64_t>                                     m_exact_floor {0}; // (MarkExactEpochsFromNow)
	std::vector<GuestRange> m_registration_spans; // see TakeRegistrationSpans
	bool                    m_registration_spans_lost = true;
	struct SyncBuffer {
		uint64_t start;
		uint64_t end;
		Buffer* buffer;
		BufferId id; // with its generation: a later buffer in the same slot is another buffer
	};
	// GPU-thread-only derived index. SlotVector preserves addresses until erase;
	// every registration change invalidates this view before it can be reused.
	std::vector<SyncBuffer>                            m_sync_buffers;
	bool                                              m_sync_buffers_valid = false;
	struct SyncStamp { uint64_t begin = 0, end = 0, epoch = 0; };
	std::vector<SyncStamp> m_sync_stamps;
	std::vector<SyncBuffer> m_old_sync_buffers; // rebuild scratch
	std::vector<SyncStamp>  m_old_sync_stamps;
	// The registration changes since the index was current, applied to it in order (ApplySyncChanges) instead of a
	// rebuild from every registered buffer (a walk of m_buffers' nodes and slots: ~5 a frame at 1-1, 2% of the GPU
	// thread). Lost: too many, or the index was never built; the next use rebuilds it.
	struct SyncChange {
		BufferId id;
		uint64_t address = 0, size = 0;
		bool     insert  = false;
	};
	std::vector<SyncChange> m_sync_changes;
	bool                    m_sync_changes_lost = true;
	[[nodiscard]] bool      ApplySyncChanges();
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	std::shared_mutex                                 m_gpu_modified_mutex; // (its changes: HasGpuDirtyBytes)
	// Moves whenever m_gpu_modified_ranges loses bytes (readbacks, copy feedback, downloads, discards: CPU writes and
	// unmaps go through those first). GPU thread.
	uint64_t m_gpu_coverage_epoch = 1;
	// The ranges ObtainBuffer made GPU-written lately (GPU thread): one written again while no buffer was registered
	// or unregistered and no GPU-written byte was given back since is the same buffer's, its pages still GPU-owned
	// and its bytes still in m_gpu_modified_ranges (what SynchronizeBuffer and the Add would find).
	struct WriteMemo {
		uint64_t vaddr = 0, size = 0, registration = 0, coverage = 0;
		BufferId id;
		bool     texel = false;
	};
	std::array<WriteMemo, 4096> m_write_memo {};
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	// ObtainBufferForImage (KYTY_ASYNC_UPLOAD=2): the backing pieces of an image upload.
	std::vector<std::pair<const uint8_t*, uint64_t>>  m_backing_pieces;
	// The upload worker's copies of guest [vaddr, vaddr + size) to `staging`; false when it reads the mapped parts
	// only (a partly unmapped range: the rest is zeros).
	bool PushImageStagingCopies(uint8_t* staging, uint64_t vaddr, uint64_t size);
	// ImageProtectSink's protections: those of pages a copy of [vaddr, vaddr + size) the upload worker makes next reads
	// whole go ahead of it, the others are applied here. `whole_image`: the range is the image's (bytes of its first and
	// last pages outside it are no other watcher's, or their pages would have been watched already).
	std::vector<PageManager::DeferredRange> m_image_protects;
	PageManager*                            m_image_protect_manager = nullptr;
	void QueueImageProtects(std::span<const StagingPiece> copied, bool whole_image);
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_host_shader_upload;
	StreamBuffer                                      m_table_upload;
	// Buffer uploads stage here while it has room (UploadCopies): device-local host-visible memory, whose copies read
	// video memory instead of system memory across the bus.
	StreamBuffer                                      m_staging_device;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	GpuResourceManager*                               m_resources = nullptr;

public:
	// The guest range is mapped for the GPU (not unmapped since): deferred work checks this
	// before it creates buffers over ranges recorded earlier.
	[[nodiscard]] bool IsGpuMapped(uint64_t vaddr, uint64_t size) const noexcept;

private:
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	TickClock m_tick_clock;                // RetireStaleBuffers counts idleness in time
	bool      m_retire_pressure = false; // the video memory in use at the critical level
	// The registered game buffers in system memory and the room those in use need (MoveSpilledBufferBack).
	std::vector<BufferId> m_spilled;
	uint64_t              m_spilled_room = 0;

	// KYTY_READBACK_QUEUE: GPU writes that may still be in flight, oldest first. An entry's
	// tick is set at the next point between commands (every write noted by then is recorded
	// at or before the current tick: m_gpu_write_stamps); entries the GPU completed are dropped
	// from the front.
	struct GpuWrite {
		uint64_t begin, end;
		bool     big = false; // spans more granules than GpuWriteSpan: in m_gpu_big_writes, not m_gpu_write_last
	};
	// The ticks of the stamped writes, oldest first: (end, tick) dates the writes numbered from the previous entry's end
	// (the head's number for the first) up to `end`. (A tick in every entry: each was written once and read once more,
	// a pass over thousands of cold entries a frame.)
	std::deque<std::pair<uint64_t, uint64_t>> m_gpu_write_stamps;
	[[nodiscard]] uint64_t                    GpuWriteTick(size_t index) const;
	// Per 64 KiB granule (hashed into the counters), 1 + the number (m_gpu_writes_base + index) of the last small write
	// there; a write over more than GpuWriteSpan granules is in m_gpu_big_writes instead. The in-flight writes are the
	// log's tail: a range whose granules' last write completed overlaps no in-flight small write, and the walk for one
	// that may starts at that last write, not at the log's end (thousands of entries a frame at 1-1). (Counting the
	// writes per granule, up at each write and down at its completion, cost a random read-modify-write per granule.)
	static constexpr uint32_t   GpuWriteGranuleBits = 16, GpuWriteCounters = 1u << 16u, GpuWriteSpan = 64;
	std::unique_ptr<uint64_t[]> m_gpu_write_last = std::make_unique<uint64_t[]>(GpuWriteCounters);
	// The numbers of the big writes, oldest first (the completed ones dropped at queries): a query looks for an
	// overlapping one in them (one in flight, a 4-8 MiB copy every frame, made every query walk the log from its end).
	std::deque<uint64_t>        m_gpu_big_writes;
	uint64_t                    m_gpu_writes_base = 0; // the number of m_gpu_writes[0]
	[[nodiscard]] static uint32_t GpuWriteCounter(uint64_t granule) {
		return static_cast<uint32_t>((granule * 0x9e3779b97f4a7c15ull) >> (64u - 16u));
	}
	void ResetGpuWrites();
	void                  NoteGpuWrite(uint64_t vaddr, uint64_t size);
	[[nodiscard]] uint64_t InflightWriteTick(uint64_t begin, uint64_t end, uint64_t completed,
	                                         bool between_commands);
	// The graphics tick a transfer-queue copy of `ranges` ({begin, end}) waits for; false when
	// the copy must drain the graphics queue as before.
	[[nodiscard]] bool ReadbackQueueReady(std::span<const std::pair<uint64_t, uint64_t>> ranges,
	                                      bool between_commands, uint64_t& wait_tick);
	std::vector<GpuWrite> m_gpu_writes;
	size_t                m_gpu_writes_head    = 0;
	size_t                m_gpu_writes_stamped = 0;
	bool                  m_gpu_writes_on      = false;
	// Writes before this tick were not noted (UINT64_MAX: set at the next stamp).
	uint64_t              m_gpu_writes_from    = UINT64_MAX;
	// Last copy-engine value, and last graphics tick, that wrote each download slot: a detached
	// request's copy may still be pending on either queue when its slot is reused.
	std::array<uint64_t, GuestReadbackSlots> m_download_queue_values {};
	std::array<uint64_t, GuestReadbackSlots> m_download_ticks {};

	// Freed ranges whose GPU-written bytes are on their way to guest memory (DropFreedRange, BeginFreedReadback), oldest
	// first, and the ring in host memory their copies write (m_freed_ring_head/tail: byte counts allocated and given back,
	// in order). m_freed_lo/hi: the pages they span, NoteGpuWrite's test (an empty span when none is pending).
	struct FreedReadback;
	std::deque<std::unique_ptr<FreedReadback>> m_freed_readbacks;
	std::unique_ptr<Buffer>                    m_freed_ring;
	uint64_t                                   m_freed_ring_head = 0, m_freed_ring_tail = 0;
	uint64_t                                   m_freed_lo = UINT64_MAX, m_freed_hi = 0;
	FreedReadbackCounts                        m_freed_counts;
	[[nodiscard]] bool                         BeginFreedReadback(uint64_t vaddr, uint64_t size);
	void                                       NoteFreedWrite(uint64_t vaddr, uint64_t size);
	void                                       UpdateFreedBounds();

	// Last member: destroyed (queue idle) before the buffers its copies use.
	std::unique_ptr<ReadbackQueue::Queue> m_readback_queue;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

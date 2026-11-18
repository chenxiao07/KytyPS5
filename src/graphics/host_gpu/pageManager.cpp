#include "graphics/host_gpu/pageManager.h"
#include "live-counters.h"
#include "live-switch.h"
#include "mode-toggle.h"
#include "slow-log.h"

#include "graphics/host_gpu/regionDefinitions.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <sched.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {
namespace {
thread_local std::vector<PageManager::DeferredRange>* g_deferred_write_protect = nullptr;
thread_local uint64_t g_unmapping_begin = 0;
thread_local uint64_t g_unmapping_end   = 0;
// PageManager::DeferReadProtection: the GPU thread's pending no-access ranges, disjoint and not adjacent (a range
// next to one extends it).
thread_local bool g_defer_read_protect = false;
struct PendingRange {
	uint64_t begin, end;
};
thread_local std::vector<PendingRange> g_read_protects;
thread_local uint64_t                  g_observation_epoch = 0; // (PageManager::ObservationEpoch)

// KYTY_SPEC_PROTECT_ASYNC (live spec_protect_async; on under KYTY_SPECULATE=4, off otherwise, mode-toggle.h; acts under
// KYTY_SPECULATE=4 only): (1) the GPU thread's deferred no-access ranges pending at a submission handed to the upload worker
// (PageManager::FlushDeferredProtectionAtSubmit) instead of applied there, the GPU thread's next other flush applying
// those the worker has not begun; (2) the write watch an upload of a buffer GPU work writes arms (read-only) queued to the
// upload worker ahead of the upload's copies, as a read-only buffer's is (KYTY_ASYNC_REPROTECT), instead of applied at
// once (PageManager::WrittenArmAsync, ReapplyWrittenProtection).
// (The mode first: a mode-0 caller reads one global, not the switch. A mode-4 boot or a live flip to mode 4 makes the
// switch with mode-toggle.h's table before any test here could, so a live `set` finds it.)
bool ProtectAsyncOn() noexcept {
	if (kyty_local_speculate_mode.load(std::memory_order_relaxed) != 4) return false;
	static auto& on = LiveSwitch::Find("spec_protect_async", ModeToggle::Default("spec_protect_async"));
	return on.load(std::memory_order_relaxed) != 0;
}
// The pending no-access ranges' per-page marks (Region::na_pending; both halves of the switch read them). Kept on the GPU
// thread from a moment none is pending once the switch is on, until a moment none is pending, handed or read by a queued
// arm job once it is off (a mark is set and cleared under its page's region lock; a pending range then lies in one
// tracker region). Not while batch_busy_once is on: the flush then holds a region lock across its system call, which a
// batch thread's busy section must not fault into.
thread_local bool g_track_pending = false;
// Written buffers' arm jobs queued and not yet run (PageManager::NoteWrittenArmQueued, ReapplyWrittenProtection): the
// marks stay kept while any is (Impl::UntrackIfUnwanted).
std::atomic<uint64_t> g_written_arms_queued {0};
bool TrackWanted() noexcept {
	static auto& busy_once = LiveSwitch::Find("batch_busy_once", 0);
	return ProtectAsyncOn() && busy_once.load(std::memory_order_relaxed) <= 0;
}

constexpr uint64_t PAGE_SIZE    = TRACKER_PAGE_SIZE;
constexpr uint64_t REGION_SIZE  = TRACKER_REGION_SIZE;
constexpr uint64_t ADDRESS_SIZE = TRACKER_ADDRESS_SIZE;
constexpr uint64_t REGION_COUNT = ADDRESS_SIZE / REGION_SIZE;

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
// The tracker reuses Win32 memory-protection tags as internal page-state values.
// Mirror their canonical numeric values so the shared state-machine logic is identical.
constexpr uint32_t PAGE_NOACCESS  = 0x01;
constexpr uint32_t PAGE_READONLY  = 0x02;
constexpr uint32_t PAGE_READWRITE = 0x04;
#endif
constexpr uint64_t REGION_PAGES = REGION_SIZE / PAGE_SIZE;
// HasReadWatchers' filter: a bit per 64 KiB granule.
constexpr uint64_t READ_GRANULE_SIZE  = 64 * 1024;
constexpr uint64_t READ_GRANULE_PAGES = READ_GRANULE_SIZE / PAGE_SIZE;
// The lowest 64 KiB of the address space: never guest memory (UpdateRegionWatchers).
constexpr size_t NULL_PAGES = 64 * 1024 / PAGE_SIZE;

constexpr uint32_t NO_ACCESS_PROTECTION  = PAGE_NOACCESS;
constexpr uint32_t READ_ONLY_PROTECTION  = PAGE_READONLY;
constexpr uint32_t READ_WRITE_PROTECTION = PAGE_READWRITE;

// The raw return addresses of a fatal end, on stderr (resolve them with the build's map: the end itself names no
// caller; 10-18 a 1-1 tour ended on "invalid range vaddr=0, size=0x400000" with nothing else).
void WriteBacktrace() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	void*      frames[32];
	const auto count = RtlCaptureStackBackTrace(0, 32, frames, nullptr);
	std::fprintf(stderr, "--- Backtrace --- (exe base %p)\n", static_cast<void*>(GetModuleHandleW(nullptr)));
	for (USHORT i = 0; i < count; ++i) std::fprintf(stderr, "  %p\n", frames[i]);
	std::fflush(stderr);
#endif
}

[[noreturn]] void FailFast(const char* reason = nullptr) noexcept {
	std::fputs("PageManager fail-fast: ", stderr);
	std::fputs(reason != nullptr ? reason : "invalid page state", stderr);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	WriteBacktrace();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(EXCEPTION_NONCONTINUABLE_EXCEPTION));
#endif
	std::_Exit(322);
}

[[noreturn]] void Fatal(const char* format, ...) {
	std::fputs("PageManager fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	WriteBacktrace();
	std::_Exit(322);
}

Common::VirtualMemory::Mode ToMemoryMode(uint32_t protection) {
	switch (protection) {
		case NO_ACCESS_PROTECTION: return Common::VirtualMemory::Mode::NoAccess;
		case READ_ONLY_PROTECTION: return Common::VirtualMemory::Mode::Read;
		case READ_WRITE_PROTECTION: return Common::VirtualMemory::Mode::ReadWrite;
		default: Fatal("unmappable protection 0x%08" PRIx32, protection);
	}
}

class SpinGuard final {
public:
	explicit SpinGuard(std::atomic_flag& lock): m_lock(lock) {
		// Owners call VirtualProtect inside: past a short spin a waiter yields its CPU instead
		// of competing with a descheduled owner (see TrackingSpinLock).
		uint32_t spins = 0;
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			do {
				if (++spins < 512) {
					_mm_pause();
				} else {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
					SwitchToThread();
#else
					sched_yield();
#endif
				}
			} while (m_lock.test(std::memory_order_relaxed));
		}
	}
	~SpinGuard() { m_lock.clear(std::memory_order_release); }
	KYTY_CLASS_NO_COPY(SpinGuard);

private:
	std::atomic_flag& m_lock;
};

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

uint64_t PageStart(uint64_t vaddr) {
	return vaddr & ~(PAGE_SIZE - 1);
}

uint64_t PageEnd(uint64_t vaddr, uint64_t size) {
	ValidateRange(vaddr, size);
	return PageStart(vaddr + size - 1) + PAGE_SIZE;
}

} // namespace

struct PageManager::Impl {
	struct PageState {
		uint8_t write_watchers  : 7 = 0;
		uint8_t access_watchers : 1 = 0;

		[[nodiscard]] uint32_t Perms() const noexcept {
			if (access_watchers != 0) {
				return NO_ACCESS_PROTECTION;
			}
			if (write_watchers != 0) {
				return READ_ONLY_PROTECTION;
			}
			return READ_WRITE_PROTECTION;
		}

		template <int delta, bool is_read>
		uint32_t AddDelta(uint64_t address) {
			static_assert(delta >= -1 && delta <= 1);
			if constexpr (is_read) {
				if constexpr (delta == 1) {
					if (access_watchers != 0) {
						Fatal("read-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++access_watchers;
				} else if constexpr (delta == -1) {
					if (access_watchers == 0) {
						Fatal("read-watcher underflow at 0x%016" PRIx64, address);
					}
					return --access_watchers;
				} else {
					return access_watchers;
				}
			} else {
				if constexpr (delta == 1) {
					if (write_watchers == 0x7f) {
						Fatal("write-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++write_watchers;
				} else if constexpr (delta == -1) {
					if (write_watchers == 0) {
						Fatal("write-watcher underflow at 0x%016" PRIx64, address);
					}
					return --write_watchers;
				} else {
					return write_watchers;
				}
			}
		}
	};
	static_assert(sizeof(PageState) == 1);

	struct Region {
		std::atomic_flag                    lock = ATOMIC_FLAG_INIT;
		std::array<PageState, REGION_PAGES> pages;
		// Nonzero while a page has read watchers (HasReadWatchers).
		std::array<std::atomic<uint8_t>, REGION_PAGES> read_hint {};
		// The texture offload's holds of a page (src/local/texture-offload.h: its guest bytes dropped from RAM): no access
		// while any lasts, whatever its watchers. A count of its own: as a read watcher it took the one read bit a page
		// has, and the GPU's tracking of a GPU write over such a texture's memory (a pool reused while a level loads)
		// ended the emulator with "read-watcher overflow" (issue #91, an RTX 3050 with 16 GB of RAM). (Atomic: WatchedRuns
		// reads without the lock.)
		std::array<std::atomic<uint8_t>, REGION_PAGES> offload {};
		// KYTY_SPEC_PROTECT_ASYNC (g_track_pending): the pages in a no-access range the GPU thread holds pending
		// (g_read_protects): set as it joins, cleared as it is applied or handed to the upload worker; under the lock.
		// Made by the region's first mark (NaMark), under the lock, so a region never marked holds none (null: none set).
		std::unique_ptr<uint8_t[]> na_pending;
	};

	// The region's mark of `page` (under its lock).
	static bool NaPending(const Region& region, size_t page) noexcept {
		return region.na_pending != nullptr && region.na_pending[page] != 0;
	}
	// The marks of pages [first, last) of the region set or cleared (under its lock); made at the first set.
	static void NaMark(Region& region, size_t first, size_t last, uint8_t on) noexcept {
		if (region.na_pending == nullptr) {
			if (on == 0) return;
			region.na_pending = std::make_unique<uint8_t[]>(REGION_PAGES); // (zeroed)
		}
		for (auto page = first; page < last; ++page) region.na_pending[page] = on;
	}

	// A page's protection: its watchers', no access while the offload holds it.
	static uint32_t PermsOf(const Region& region, size_t page) noexcept {
		return region.offload[page].load(std::memory_order_relaxed) != 0 ? NO_ACCESS_PROTECTION : region.pages[page].Perms();
	}

	Impl() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		if (info.dwPageSize != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32,
			      static_cast<uint32_t>(info.dwPageSize));
		}
#elif defined(__APPLE__)
		// Under Rosetta the host page size is 4 KB, matching TRACKER_PAGE_SIZE.
		if (static_cast<uint64_t>(getpagesize()) != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32, static_cast<uint32_t>(getpagesize()));
		}
#else
		const auto host_page_size = ::sysconf(_SC_PAGESIZE);
		if (host_page_size < 0 || static_cast<uint64_t>(host_page_size) != PAGE_SIZE) {
			Fatal("unsupported host page size %ld", static_cast<long>(host_page_size));
		}
#endif
		regions = std::make_unique<std::atomic<Region*>[]>(REGION_COUNT);
		for (uint64_t i = 0; i < REGION_COUNT; i++) {
			regions[i].store(nullptr, std::memory_order_relaxed);
		}
	}

	~Impl() {
		SettleHanded(); // (KYTY_SPEC_PROTECT_ASYNC: no handed job reads this owner after it)
		for (const auto& region: region_storage) {
			SpinGuard lock(region->lock);
			for (size_t page = 0; page < REGION_PAGES; page++) {
				if (PermsOf(*region, page) != READ_WRITE_PROTECTION) {
					FailFast("PageManager destroyed with live page state");
				}
			}
		}
	}

	Region* FindRegion(uint64_t vaddr) const noexcept {
		return vaddr < ADDRESS_SIZE ? regions[vaddr / REGION_SIZE].load(std::memory_order_acquire)
		                            : nullptr;
	}

	Region* GetOrCreateRegion(uint64_t vaddr) {
		const auto index = vaddr / REGION_SIZE;
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		std::lock_guard lock(region_mutex);
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		auto  region = std::make_unique<Region>();
		auto* ptr    = region.get();
		region_storage.push_back(std::move(region));
		regions[index].store(ptr, std::memory_order_release);
		return ptr;
	}

	static void ProtectHost(uint64_t vaddr, uint64_t size, uint32_t protection) noexcept {
		if (!Libs::LibKernel::Memory::ProtectGuestHostMemory(vaddr, size,
		                                                     ToMemoryMode(protection))) {
			Fatal("address-space protection failed at 0x%016" PRIx64 ", new=0x%08" PRIx32, vaddr,
			      protection);
		}
	}

	// The deferred read protections' no access, by minor mode where the platform can (VirtualMemory::ZapNoAccess:
	// Linux, KYTY_UFFD_MINOR; live switch uffd_minor, 0: mprotect for the protections made from then on).
	struct ZapScope {
		ZapScope() noexcept {
			static auto& zap = LiveSwitch::Find("uffd_minor", 1);
			Common::VirtualMemory::ZapNoAccess(zap.load(std::memory_order_relaxed) != 0);
		}
		~ZapScope() { Common::VirtualMemory::ZapNoAccess(false); }
	};

	static void FlushReadProtect() noexcept {
		if (g_read_protects.empty()) return;
		ZapScope zap;
		for (const auto& range: g_read_protects) ApplyPending(range, nullptr);
		g_read_protects.clear();
	}

	// KYTY_SPEC_PROTECT_ASYNC: the page manager whose pages this thread marks pending (g_track_pending).
	static inline thread_local Impl* pending_owner = nullptr;

	// The pages of [begin, end) (in one region) marked or unmarked as held pending (Region::na_pending), under its lock.
	static void MarkPending(Region& region, uint64_t begin, uint64_t end, uint8_t on) noexcept {
		const auto base = begin / REGION_SIZE * REGION_SIZE;
		const auto last = std::min<uint64_t>(REGION_PAGES, (end - base + PAGE_SIZE - 1) / PAGE_SIZE);
		NaMark(region, static_cast<size_t>((begin - base) / PAGE_SIZE), static_cast<size_t>(last), on);
	}

	// A pending range applied (no access). With the marks kept (g_track_pending), its pages unmarked and the call made
	// under its region's lock, so a reader of the marks (a written buffer's arm job) sees either the mark and the old
	// protection, or neither; `held`: the region whose lock the caller holds (a pending range lies in one region then),
	// else taken here. (batch_busy_once turned on live while the marks are kept, TrackWanted false: the flush's call is
	// made outside the lock, the marks cleared under it first; a reader of the marks then sees the mark and applies what
	// it has before the call, or none and the state's no access: either way the call's no access is last.)
	static void ApplyPending(const PendingRange& range, Region* held) noexcept {
		Region* region = g_track_pending && pending_owner != nullptr ? pending_owner->FindRegion(range.begin) : nullptr;
		if (region == nullptr) return ProtectHost(range.begin, range.end - range.begin, NO_ACCESS_PROTECTION);
		if (region != held && !TrackWanted()) {
			{
				SpinGuard lock(region->lock);
				MarkPending(*region, range.begin, range.end, 0);
			}
			return ProtectHost(range.begin, range.end - range.begin, NO_ACCESS_PROTECTION);
		}
		std::optional<SpinGuard> lock;
		if (region != held) lock.emplace(region->lock);
		MarkPending(*region, range.begin, range.end, 0);
		ProtectHost(range.begin, range.end - range.begin, NO_ACCESS_PROTECTION);
	}

	// KYTY_SPEC_PROTECT_ASYNC: a pending range handed to the upload worker at a submission (HandOff). Applied once, by
	// whichever thread takes it first: the worker's job, or the GPU thread at its next flush that is not a submission's
	// (TakeOverHanded). The job ends only once it is applied (by either), so the submission's upload sequence, which its
	// work waits for, comes after the no access.
	struct HandedRange {
		std::atomic<uint32_t> state {0}; // 0 queued, 1 being applied, 2 applied
		std::atomic<uint32_t> refs {2};  // the job's and the GPU thread's list's
		uint64_t              begin = 0, end = 0;
		Impl*                 owner = nullptr;
		HandedRange*          prev = nullptr, *next = nullptr; // (HandedLive's list, under its mutex)
	};
	static inline thread_local std::vector<HandedRange*> handed_ranges; // the GPU thread's, not yet seen applied
	// Every handed range still referenced, any thread's (its owner's teardown settles those of its own: ~Impl): linked
	// at its hand-off, unlinked as its last reference goes, under the mutex. Never freed (an owner or a job at static
	// destruction still finds it); `count` lets a teardown with none skip the mutex.
	struct HandedLive {
		std::mutex            mutex;
		HandedRange*          head = nullptr;
		std::atomic<uint64_t> count {0};
	};
	static HandedLive& Handed() noexcept {
		static auto* live = new HandedLive;
		return *live;
	}

	static void YieldCpu(uint32_t& spins) noexcept {
		if (++spins < 512) {
			_mm_pause();
		} else {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			SwitchToThread();
#else
			sched_yield();
#endif
		}
	}

	// A handed range applied: no access to each of its pages whose state is still a read watcher's (access_watchers) and
	// that no newer pending range holds (na_pending: its own flush applies it), the offload's held pages among them (a
	// hold over a read-watched page changes no protection itself); any other page keeps what the changes made to it since
	// the hand-off gave it (each under the region's lock, as this). One call a run, through the GPU thread's own protect
	// entry (ProtectGuestHostMemory: the address space's serial wait for maps and unmaps unchanged). Not zapped (ZapScope,
	// Linux uffd minor): a plain no access, as with uffd_minor 0, on whichever thread applies it. Fatal where the call
	// fails, as the GPU thread's ProtectHost: a page unmapped since is no longer watched, so only a due page is called for.
	void ApplyHanded(uint64_t begin, uint64_t end) noexcept {
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, (chunk_begin / REGION_SIZE + 1) * REGION_SIZE);
			const auto region_base = chunk_begin / REGION_SIZE * REGION_SIZE;
			if (auto* region = FindRegion(chunk_begin); region != nullptr) {
				SpinGuard  lock(region->lock);
				const auto due = [&](size_t page) {
					return region->pages[page].access_watchers != 0 && !NaPending(*region, page);
				};
				const auto last = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
				for (auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE); first < last;) {
					const bool apply = due(first);
					auto       next  = first + 1;
					while (next < last && due(next) == apply) next++;
					// (A page still due is mapped and read-watched: its failure is the GPU thread's ProtectHost's, fatal.)
					if (apply && !Libs::LibKernel::Memory::ProtectGuestHostMemory(region_base + first * PAGE_SIZE,
					                                                              (next - first) * PAGE_SIZE,
					                                                              ToMemoryMode(NO_ACCESS_PROTECTION))) {
						LiveCounters::Add(LiveCounters::Rp1Failed);
						Fatal("address-space protection failed at 0x%016" PRIx64 " (a handed range's), new=0x%08" PRIx32,
						      region_base + first * PAGE_SIZE, NO_ACCESS_PROTECTION);
					}
					first = next;
				}
			}
			chunk_begin = chunk_end;
		}
	}

	// Whoever finds a handed range queued applies it (true); one that finds it being applied waits until it is.
	static bool ApplyOrAwait(HandedRange& range) noexcept {
		uint32_t expected = 0;
		if (range.state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
			range.owner->ApplyHanded(range.begin, range.end);
			range.state.store(2, std::memory_order_release);
			return true;
		}
		if (expected == 1) {
			uint32_t spins = 0;
			while (range.state.load(std::memory_order_acquire) != 2) YieldCpu(spins);
		}
		return false;
	}
	static void ReleaseHanded(HandedRange* range) noexcept {
		if (range->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
		{
			auto&           live = Handed();
			std::lock_guard lock(live.mutex);
			(range->prev != nullptr ? range->prev->next : live.head) = range->next;
			if (range->next != nullptr) range->next->prev = range->prev;
			live.count.fetch_sub(1, std::memory_order_relaxed);
		}
		delete range;
	}
	// At its owner's teardown: its handed ranges still queued settled (state 2: the worker's job, which the upload worker
	// may run after this, then does nothing and never reads the owner); one being applied waited for. Their no access is
	// not applied: the pages' state goes with the owner.
	void SettleHanded() noexcept {
		auto& live = Handed();
		if (live.count.load(std::memory_order_acquire) == 0) return;
		std::lock_guard lock(live.mutex);
		for (auto* range = live.head; range != nullptr; range = range->next) {
			if (range->owner != this) continue;
			uint32_t expected = 0;
			if (range->state.compare_exchange_strong(expected, 2, std::memory_order_acq_rel)) continue;
			uint32_t spins = 0;
			while (range->state.load(std::memory_order_acquire) != 2) YieldCpu(spins);
		}
	}
	// The upload worker's job (or whoever runs it: Help, a drain).
	static void RunHanded(void* context, uint64_t, uint64_t) noexcept {
		auto* range = static_cast<HandedRange*>(context);
		(void)ApplyOrAwait(*range);
		ReleaseHanded(range);
	}

	// At a submission (GPU thread): the pending ranges handed to the upload worker, each a job pushed (`push`) ahead of the
	// submission's upload sequence, their marks cleared first. False (nothing done): the switch off, nothing pending, the
	// marks not kept, or no worker (`push` null); nor while batch_busy_once is on (TrackWanted false: the worker's job
	// applies under the region's lock).
	static bool HandOff(PageManager::HandPush push) noexcept {
		if (g_read_protects.empty() || push == nullptr || !g_track_pending || pending_owner == nullptr || !TrackWanted())
			return false;
		auto& handed = handed_ranges;
		for (size_t i = 0; i < handed.size();) { // (those applied already dropped)
			if (handed[i]->state.load(std::memory_order_acquire) == 2) {
				ReleaseHanded(handed[i]);
				handed[i] = handed.back();
				handed.pop_back();
			} else {
				++i;
			}
		}
		for (const auto& range: g_read_protects) {
			if (auto* region = pending_owner->FindRegion(range.begin); region != nullptr) {
				SpinGuard lock(region->lock);
				MarkPending(*region, range.begin, range.end, 0);
			}
			auto* item  = new HandedRange;
			item->begin = range.begin;
			item->end   = range.end;
			item->owner = pending_owner;
			{
				auto&           live = Handed();
				std::lock_guard lock(live.mutex);
				item->next = live.head;
				if (live.head != nullptr) live.head->prev = item;
				live.head = item;
				live.count.fetch_add(1, std::memory_order_relaxed);
			}
			handed.push_back(item);
			push(&RunHanded, item, range.begin, range.end - range.begin);
		}
		LiveCounters::Add(LiveCounters::Rp1Handed, g_read_protects.size());
		g_read_protects.clear();
		return true;
	}

	// At the GPU thread's every other full flush: the handed ranges not yet applied are applied here (or waited for, if
	// the worker is applying one), before anything after the flush runs.
	static void TakeOverHanded() noexcept {
		auto& handed = handed_ranges;
		if (handed.empty()) return;
		for (auto* range: handed) {
			(void)ApplyOrAwait(*range);
			ReleaseHanded(range);
		}
		handed.clear();
	}
	// At the texture offload's partial flush (FlushDeferredProtection(vaddr, size)): the handed ranges over [begin, end)
	// applied (or waited for) too, so its hold's no access is in force at once there as well (a hold over a read-watched
	// page makes no call of its own). They stay listed until a full flush or a hand-off drops them.
	static void TakeOverHandedOverlapping(uint64_t begin, uint64_t end) noexcept {
		for (auto* range: handed_ranges)
			if (range->begin < end && begin < range->end) (void)ApplyOrAwait(*range);
	}
	// The marks no longer kept (g_track_pending off) once the switch no longer wants them, at a moment none is pending,
	// none handed over and no written buffer's arm job queued (every mark is 0 then, and no job reads them later).
	static void UntrackIfUnwanted() noexcept {
		if (g_track_pending && g_read_protects.empty() && handed_ranges.empty() &&
		    g_written_arms_queued.load(std::memory_order_acquire) == 0 && !TrackWanted())
			g_track_pending = false;
	}

	// The pending no-access ranges that overlap [begin, end), applied first (a protection change of those pages
	// comes after them; pages of the others keep theirs pending). Not zapped for the texture offload's hold (`zap`
	// false): it then frees the pages, and a hole's access would not fault. `held`: ApplyPending's.
	static void FlushReadProtectOverlapping(uint64_t begin, uint64_t end, bool zap = true, Region* held = nullptr) noexcept {
		auto& ranges = g_read_protects;
		for (size_t i = 0; i < ranges.size();) {
			if (ranges[i].begin < end && begin < ranges[i].end) {
				std::optional<ZapScope> zapping;
				if (zap) zapping.emplace();
				ApplyPending(ranges[i], held);
				ranges[i] = ranges.back();
				ranges.pop_back();
			} else {
				++i;
			}
		}
	}

	// A no-access range joins the pending ones: merged with those it overlaps or touches (one call for all).
	static void AddReadProtect(uint64_t begin, uint64_t end) noexcept {
		auto& ranges = g_read_protects;
		for (size_t i = 0; i < ranges.size();) {
			// (With the marks kept, g_track_pending, only within one tracker region: ApplyPending takes its lock.)
			if (ranges[i].begin <= end && begin <= ranges[i].end &&
			    (!g_track_pending || ranges[i].begin / REGION_SIZE == begin / REGION_SIZE)) {
				begin     = std::min(begin, ranges[i].begin);
				end       = std::max(end, ranges[i].end);
				ranges[i] = ranges.back();
				ranges.pop_back();
			} else {
				++i;
			}
		}
		ranges.push_back({begin, end});
	}

	// `reapply`: a protection the GPU thread re-applies to the pages' state (SyncProtection, ReapplyProtection, a
	// reapplying watcher update), not a watcher edge's.
	void Protect(uint64_t vaddr, uint64_t size, uint32_t protection, bool reapply = false) noexcept {
		if (g_defer_read_protect) {
			// (Every caller of Protect holds the lock of the one region [vaddr, vaddr + size) lies in.)
			if (protection == NO_ACCESS_PROTECTION) {
				// KYTY_SPEC_PROTECT_ASYNC: the pages marked pending, from a moment none is pending once wanted
				// (TrackWanted), until a moment none is pending, handed or read by a queued arm job once no longer wanted
				// (UntrackIfUnwanted).
				if (g_read_protects.empty()) {
					if (!g_track_pending && TrackWanted()) g_track_pending = true;
					else UntrackIfUnwanted();
				}
				// (A re-applied no access stays pending, merged as any, but its pages unmarked: its pages may be no access
				// on the host already, due; a mark made between a hand-off and its job would have the job skip such a page
				// still read-watched (ApplyHanded: a newer pending range applies it), leaving it without its no access until
				// the next flush, and a written buffer's arm job would lower it to its write watchers' read only. Unmarked,
				// both give it its state's no access: early at worst, never late.)
				if (g_track_pending) {
					pending_owner = this;
					if (!reapply)
						if (auto* region = FindRegion(vaddr); region != nullptr) MarkPending(*region, vaddr, vaddr + size, 1);
				}
				AddReadProtect(vaddr, vaddr + size);
				return;
			}
			// (Another protection of the pages of a pending range comes after it; the other ranges stay pending:
			// protections of disjoint pages do not depend on each other's order.)
			FlushReadProtectOverlapping(vaddr, vaddr + size, true, g_track_pending ? FindRegion(vaddr) : nullptr);
		}
		const auto end = vaddr + size;
		if (protection == READ_WRITE_PROTECTION && g_unmapping_end > g_unmapping_begin &&
		    vaddr < g_unmapping_end && end > g_unmapping_begin) {
			// Releases inside the range being unmapped are left to the unmap (SetUnmappingRange).
			if (vaddr < g_unmapping_begin) ProtectHost(vaddr, g_unmapping_begin - vaddr, protection);
			if (end > g_unmapping_end) ProtectHost(g_unmapping_end, end - g_unmapping_end, protection);
			return;
		}
		ProtectHost(vaddr, size, protection);
	}

	template <bool track, bool is_read, bool masked, bool offload = false, bool reapply = false>
	void UpdateRegionWatchers(Region& region, uint64_t base_addr, size_t first, size_t last,
	                          const RegionBits* mask = nullptr) {
		// The lowest 64 KiB hold no guest memory (no mapping is ever made there): a range a stale descriptor put there
		// is not watched, both ways alike. Watched, the deferred write protection of its region's first pages started
		// at address 0, an invalid range, and the upload worker's protection job ended the emulator ("invalid range
		// vaddr=0x0, size=0x400000", issue #70 and a 1-1 tour, both with DLSS super resolution on).
		if (base_addr == 0 && first < NULL_PAGES) {
			static std::atomic<bool> reported {false};
			if (!reported.exchange(true, std::memory_order_relaxed)) {
				std::printf("Page watchers: pages below 64 KiB (%s 0x%llx-0x%llx) left unwatched: no guest memory there\n",
				            is_read ? "read" : "write", static_cast<unsigned long long>(first * PAGE_SIZE),
				            static_cast<unsigned long long>(last * PAGE_SIZE));
				std::fflush(stdout);
			}
			first = NULL_PAGES;
			if (first >= last) return;
		}
		SpinGuard lock(region.lock);
		auto      perms                 = PermsOf(region, first);
		uint64_t  range_begin           = 0;
		uint64_t  range_bytes           = 0;
		uint64_t  potential_range_bytes = 0;
		[[maybe_unused]] bool hints_to_clear = false; // (read watchers and offload holds)
		// KYTY_SPEC_PROTECT_ASYNC: the pages whose watcher edge here takes them to no access, kept only on the thread that
		// defers its read protections while the marks are kept or the switch is on (the mode first: mode 0 tests one
		// global). (The switch turned on live during this call: the run's pages stay unmarked, as a re-applied no access's:
		// early at worst, never late.)
		std::optional<std::bitset<REGION_PAGES>> na_edges;
		if (g_defer_read_protect && (g_track_pending || ProtectAsyncOn())) na_edges.emplace();

		const auto release_pending = [&] {
			if (range_bytes != 0) {
				if (track && !is_read && !offload && g_deferred_write_protect != nullptr) {
					g_deferred_write_protect->push_back({base_addr + range_begin * PAGE_SIZE, range_bytes});
				} else if (!reapply && perms == NO_ACCESS_PROTECTION) {
					// KYTY_SPEC_PROTECT_ASYNC: deferred unmarked, then only the pages of an edge to no access in it marked
					// (the run's other pages, between edges, already had it: due, a mark would have a handed job skip them
					// and a written buffer's arm job lower them to read only); one call a run still.
					Protect(base_addr + range_begin * PAGE_SIZE, range_bytes, perms, /*reapply=*/true);
					if (na_edges && g_track_pending)
						for (auto page = range_begin; page < range_begin + range_bytes / PAGE_SIZE; ++page)
							if (na_edges->test(page)) NaMark(region, page, page + 1, 1);
				} else {
					Protect(base_addr + range_begin * PAGE_SIZE, range_bytes, perms, reapply);
				}
				range_bytes           = 0;
				potential_range_bytes = 0;
			}
		};

		for (size_t page_index = first; page_index < last; page_index++) {
			auto&      page    = region.pages[page_index];
			const auto address = base_addr + page_index * PAGE_SIZE;
			const bool update  = !masked || mask->Get(page_index);

			const auto old_perms = PermsOf(region, page_index);
			uint32_t   new_count = 0;
			if constexpr (offload) {
				auto& holds = region.offload[page_index];
				new_count   = holds.load(std::memory_order_relaxed);
				if (update) {
					if (new_count == (track ? 255u : 0u)) Fatal("offload-hold %s at 0x%016" PRIx64, track ? "overflow" : "underflow", address);
					new_count += track ? 1 : -1;
					holds.store(static_cast<uint8_t>(new_count), std::memory_order_relaxed);
				}
			} else {
				new_count = update ? page.AddDelta<track ? 1 : -1, is_read>(address) : page.AddDelta<0, is_read>(address);
			}
			const auto new_perms = PermsOf(region, page_index);
			if constexpr (is_read || offload) {
				// (A page the offload holds is read-watched to HasReadWatchers: a read of it faults.) A hint is set here,
				// before the protection that takes the page's reads away, and cleared only after the one that gives them
				// back (below): cleared here, a speculation's thread that asked in between was told the page was free
				// and faulted reading it (SPEC_THREAD_GPU_FAULT; found by Bios2000 in PR #342's Linux runs).
				if (new_perms == NO_ACCESS_PROTECTION) {
					MarkReadGranule(address); // (before the page's hint: the bits stay a superset)
					region.read_hint[page_index].store(1, std::memory_order_release);
				} else if (region.read_hint[page_index].load(std::memory_order_relaxed) != 0) {
					hints_to_clear = true;
				}
			}

			if (new_perms != perms) [[unlikely]] {
				release_pending();
				perms = new_perms;
			} else if (range_bytes != 0) {
				potential_range_bytes += PAGE_SIZE;
			}

			if (!update) {
				continue;
			}

			const bool watcher_edge = (track && new_count == 1) || (!track && new_count == 0);
			if (!reapply && na_edges && watcher_edge && old_perms != new_perms && new_perms == NO_ACCESS_PROTECTION)
				na_edges->set(page_index);
			if (reapply || (watcher_edge && old_perms != new_perms)) {
				if (range_bytes == 0) {
					range_begin           = page_index;
					potential_range_bytes = PAGE_SIZE;
				}
				range_bytes = potential_range_bytes;
			}
		}

		release_pending();
		if constexpr (is_read || offload) {
			// (Every protection of the range made: the hints of the pages whose reads are back cleared now.)
			if (hints_to_clear)
				for (size_t page_index = first; page_index < last; page_index++)
					if (region.read_hint[page_index].load(std::memory_order_relaxed) != 0 &&
					    PermsOf(region, page_index) != NO_ACCESS_PROTECTION)
						region.read_hint[page_index].store(0, std::memory_order_release);
			SyncReadGranules(region, base_addr, first, last);
		}
	}

	// A bit per 64 KiB granule: some page there may have read watchers. Set before a page's read_hint, cleared after
	// the granule's last one (both under the region's lock). HasReadWatchers asks it first: a hint byte per page sat on
	// its own cold line, the region pointer on another, for every page a table read touched (1% of the GPU thread at
	// 1-1); a line of these bits covers 32 MiB.
	std::unique_ptr<std::atomic<uint64_t>[]> read_granules =
	    std::make_unique<std::atomic<uint64_t>[]>(ADDRESS_SIZE / READ_GRANULE_SIZE / 64);

	void MarkReadGranule(uint64_t address) noexcept {
		const auto granule = address / READ_GRANULE_SIZE;
		read_granules[granule / 64].fetch_or(uint64_t {1} << (granule % 64), std::memory_order_release);
	}

	void SyncReadGranules(const Region& region, uint64_t base_addr, size_t first, size_t last) noexcept {
		if (first >= last) return;
		for (size_t g = first / READ_GRANULE_PAGES; g <= (last - 1) / READ_GRANULE_PAGES; ++g) {
			bool any = false;
			for (size_t p = g * READ_GRANULE_PAGES; p < (g + 1) * READ_GRANULE_PAGES; ++p)
				any |= region.read_hint[p].load(std::memory_order_relaxed) != 0;
			const auto granule = base_addr / READ_GRANULE_SIZE + g;
			const auto bit     = uint64_t {1} << (granule % 64);
			if (any)
				read_granules[granule / 64].fetch_or(bit, std::memory_order_release);
			else if ((read_granules[granule / 64].load(std::memory_order_relaxed) & bit) != 0)
				read_granules[granule / 64].fetch_and(~bit, std::memory_order_release);
		}
	}

	[[nodiscard]] bool MayHaveReadWatchers(uint64_t vaddr, uint64_t last_byte) const noexcept {
		if (last_byte >= ADDRESS_SIZE) return true;
		for (auto g = vaddr / READ_GRANULE_SIZE; g <= last_byte / READ_GRANULE_SIZE; ++g)
			if (((read_granules[g / 64].load(std::memory_order_acquire) >> (g % 64)) & 1u) != 0) return true;
		return false;
	}

	void SyncProtection(uint64_t vaddr, uint64_t size) {
		const auto begin = PageStart(vaddr);
		const auto end   = PageEnd(vaddr, size);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, (chunk_begin / REGION_SIZE + 1) * REGION_SIZE);
			const auto region_base = chunk_begin / REGION_SIZE * REGION_SIZE;
			if (auto* region = FindRegion(chunk_begin); region != nullptr) {
				SpinGuard lock(region->lock);
				const auto last = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
				for (auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE); first < last;) {
					const auto perms = PermsOf(*region, first);
					auto       next  = first + 1;
					while (next < last && PermsOf(*region, next) == perms) next++;
					Protect(region_base + first * PAGE_SIZE, (next - first) * PAGE_SIZE, perms, /*reapply=*/true);
					first = next;
				}
			}
			chunk_begin = chunk_end;
		}
	}

	void WatchedRuns(uint64_t vaddr, uint64_t size, std::vector<PageManager::WatchedRun>* runs) const {
		const auto begin = PageStart(vaddr);
		const auto end   = PageEnd(vaddr, size);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, (chunk_begin / REGION_SIZE + 1) * REGION_SIZE);
			const auto region_base = chunk_begin / REGION_SIZE * REGION_SIZE;
			// (No region lock: a page's state is one byte, and a change made meanwhile protects after the caller.)
			if (const auto* region = FindRegion(chunk_begin); region != nullptr) {
				const auto last = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
				for (auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE); first < last;) {
					const auto perms = PermsOf(*region, first);
					auto       next  = first + 1;
					while (next < last && PermsOf(*region, next) == perms) next++;
					if (perms != READ_WRITE_PROTECTION) {
						const auto address = region_base + first * PAGE_SIZE;
						const auto mode    = ToMemoryMode(perms);
						if (!runs->empty() && runs->back().address + runs->back().size == address && runs->back().mode == mode) {
							runs->back().size += (next - first) * PAGE_SIZE;
						} else {
							runs->push_back({address, (next - first) * PAGE_SIZE, mode});
						}
					}
					first = next;
				}
			}
			chunk_begin = chunk_end;
		}
	}

	void ReapplyProtection(uint64_t vaddr, uint64_t size) {
		const auto begin = PageStart(vaddr);
		const auto end   = PageEnd(vaddr, size);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, (chunk_begin / REGION_SIZE + 1) * REGION_SIZE);
			const auto region_base = chunk_begin / REGION_SIZE * REGION_SIZE;
			if (auto* region = FindRegion(chunk_begin); region != nullptr) {
				SpinGuard lock(region->lock);
				const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
				for (auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
				     first < last;) {
					const auto perms = PermsOf(*region, first);
					auto       next  = first + 1;
					while (next < last && PermsOf(*region, next) == perms) next++;
					if (perms != READ_WRITE_PROTECTION) {
						Protect(region_base + first * PAGE_SIZE, (next - first) * PAGE_SIZE, perms, /*reapply=*/true);
					}
					first = next;
				}
			}
			chunk_begin = chunk_end;
		}
	}

	// KYTY_SPEC_PROTECT_ASYNC: ReapplyProtection for a written buffer's upload, by the pages' marks (Region::na_pending),
	// read under the region's lock with the state: a page whose no access the GPU thread still holds pending gets what it
	// has until that flush, its write watchers' (read-only; the offload's hold: no access), not the no access early; any
	// other page its state's protection, as ReapplyProtection (no access when its range was applied or handed over since:
	// it is due). Through ProtectHost: on the GPU thread too (a drain's help) nothing is deferred or flushed early.
	void ReapplyWritten(uint64_t vaddr, uint64_t size) {
		const auto begin = PageStart(vaddr);
		const auto end   = PageEnd(vaddr, size);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, (chunk_begin / REGION_SIZE + 1) * REGION_SIZE);
			const auto region_base = chunk_begin / REGION_SIZE * REGION_SIZE;
			if (auto* region = FindRegion(chunk_begin); region != nullptr) {
				SpinGuard  lock(region->lock);
				const auto perms_of = [&](size_t page) -> uint32_t {
					if (!NaPending(*region, page)) return PermsOf(*region, page);
					if (region->offload[page].load(std::memory_order_relaxed) != 0) return NO_ACCESS_PROTECTION;
					return region->pages[page].write_watchers != 0 ? READ_ONLY_PROTECTION : READ_WRITE_PROTECTION;
				};
				const auto last = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
				for (auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE); first < last;) {
					const auto perms = perms_of(first);
					auto       next  = first + 1;
					while (next < last && perms_of(next) == perms) next++;
					if (perms != READ_WRITE_PROTECTION) ProtectHost(region_base + first * PAGE_SIZE, (next - first) * PAGE_SIZE, perms);
					first = next;
				}
			}
			chunk_begin = chunk_end;
		}
	}

	template <bool track, bool is_read, bool offload = false, bool reapply = false>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
		const auto begin = PageStart(vaddr);
		const auto end   = PageEnd(vaddr, size);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, (chunk_begin / REGION_SIZE + 1) * REGION_SIZE);
			const auto region_base = chunk_begin / REGION_SIZE * REGION_SIZE;
			auto*      region = track ? GetOrCreateRegion(chunk_begin) : FindRegion(chunk_begin);
			if (region == nullptr) {
				Fatal("untracking unknown page 0x%016" PRIx64, chunk_begin);
			}
			const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			UpdateRegionWatchers<track, is_read, false, offload, reapply>(*region, region_base,
			                                                              first, last);
			chunk_begin = chunk_end;
		}
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	std::mutex                              region_mutex;
};

static_assert(std::atomic<void*>::is_always_lock_free);

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {}

PageManager::~PageManager() = default;

uint64_t PageManager::GetPageSize() const {
	return PAGE_SIZE;
}

const std::atomic<uint64_t>* PageManager::ReadGranules() const noexcept {
	static_assert(READ_GRANULE_SIZE == uint64_t {1} << READ_GRANULE_BITS);
	return m_impl->read_granules.get();
}

bool PageManager::HasReadWatchers(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) return false;
	if (!m_impl->MayHaveReadWatchers(vaddr, vaddr + size - 1)) return false;
	const auto end = PageStart(vaddr + size - 1);
	for (auto page = PageStart(vaddr);; page += PAGE_SIZE) {
		if (const auto* region = m_impl->FindRegion(page);
		    region != nullptr &&
		    region->read_hint[(page % REGION_SIZE) / PAGE_SIZE].load(std::memory_order_acquire))
			return true;
		if (page == end) return false;
	}
}

void PageManager::SetDeferredWriteProtectSink(std::vector<DeferredRange>* sink) noexcept {
	g_deferred_write_protect = sink;
}

void PageManager::SyncProtection(uint64_t vaddr, uint64_t size) {
	m_impl->SyncProtection(vaddr, size);
}

void PageManager::SetUnmappingRange(uint64_t begin, uint64_t end) noexcept {
	g_unmapping_begin = end == 0 ? 0 : begin;
	g_unmapping_end   = end;
}

bool PageManager::RestoreIfUnwatched(uint64_t vaddr) noexcept {
	// A region that does not exist yet never held a watcher: the protection is not ours.
	auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	SpinGuard lock(region->lock);
	if (Impl::PermsOf(*region, (vaddr % REGION_SIZE) / PAGE_SIZE) != READ_WRITE_PROTECTION) {
		return false;
	}
	return Libs::LibKernel::Memory::RestoreGuestWritable(PageStart(vaddr), PAGE_SIZE);
}

void PageManager::SettleProtection(uint64_t vaddr) noexcept {
	auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		(void)Libs::LibKernel::Memory::RestoreGuestProtection(PageStart(vaddr), PAGE_SIZE);
		return;
	}
	SpinGuard  lock(region->lock);
	const auto perms = Impl::PermsOf(*region, (vaddr % REGION_SIZE) / PAGE_SIZE);
	if (perms == READ_WRITE_PROTECTION) (void)Libs::LibKernel::Memory::RestoreGuestProtection(PageStart(vaddr), PAGE_SIZE);
	else Impl::ProtectHost(PageStart(vaddr), PAGE_SIZE, perms);
}

bool PageManager::WatchedSpan(uint64_t vaddr, uint64_t size, uint64_t* first, uint64_t* end) const noexcept {
	bool           any  = false;
	const uint64_t stop = UINT64_MAX - vaddr < size ? UINT64_MAX : vaddr + size;
	for (uint64_t at = PageStart(vaddr); at < stop && at < ADDRESS_SIZE;) {
		const uint64_t until = std::min(stop, (at / REGION_SIZE + 1) * REGION_SIZE);
		// (A region that does not exist yet never held a watcher.)
		if (auto* region = m_impl->FindRegion(at); region != nullptr) {
			SpinGuard lock(region->lock);
			for (uint64_t page = at; page < until; page += PAGE_SIZE) {
				if (Impl::PermsOf(*region, (page % REGION_SIZE) / PAGE_SIZE) == READ_WRITE_PROTECTION) continue;
				if (!any) *first = page;
				any  = true;
				*end = page + PAGE_SIZE;
			}
		}
		at = until;
	}
	return any;
}

void PageManager::ReapplyProtection(uint64_t vaddr, uint64_t size) {
	m_impl->ReapplyProtection(vaddr, size);
}

void PageManager::ReapplyWrittenProtection(uint64_t vaddr, uint64_t size, uint32_t jobs) {
	m_impl->ReapplyWritten(vaddr, size);
	g_written_arms_queued.fetch_sub(jobs, std::memory_order_acq_rel); // (after: the marks are read until here)
}

void PageManager::NoteWrittenArmQueued() noexcept {
	g_written_arms_queued.fetch_add(1, std::memory_order_acq_rel);
}

bool PageManager::WrittenArmAsync() noexcept {
	return g_defer_read_protect && g_track_pending && TrackWanted(); // (TrackWanted: the switch, not under batch_busy_once)
}

void PageManager::WatchedRuns(uint64_t vaddr, uint64_t size, std::vector<WatchedRun>* runs) const {
	m_impl->WatchedRuns(vaddr, size, runs);
}

template <bool track>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	SlowLog::Scope slow([&](double ms) {
		std::printf("SLOW UpdatePageWatchers %.1f ms track=%d addr=0x%llx size=0x%llx\n", ms, track ? 1 : 0,
		            static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size));
	});
	m_impl->UpdatePageWatchers<track, false>(vaddr, size);
}

template void PageManager::UpdatePageWatchers<true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false>(uint64_t, uint64_t);

template <bool track, bool reapply>
void PageManager::UpdateReadWatchers(uint64_t vaddr, uint64_t size) {
	m_impl->UpdatePageWatchers<track, false, true, reapply>(vaddr, size);
}

template void PageManager::UpdateReadWatchers<true>(uint64_t, uint64_t);
template void PageManager::UpdateReadWatchers<false>(uint64_t, uint64_t);
template void PageManager::UpdateReadWatchers<false, true>(uint64_t, uint64_t);

void PageManager::DeferReadProtection(bool on) noexcept {
	if (!on) {
		Impl::TakeOverHanded(); // (KYTY_SPEC_PROTECT_ASYNC: those handed at a submission first; none without it)
		Impl::FlushReadProtect();
		++g_observation_epoch;
	}
	g_defer_read_protect = on;
}

void PageManager::FlushDeferredProtection() noexcept {
	Impl::TakeOverHanded(); // (KYTY_SPEC_PROTECT_ASYNC: those handed at a submission first; none without it)
	Impl::FlushReadProtect();
	++g_observation_epoch;
}

bool PageManager::FlushDeferredProtectionAtSubmit(HandPush push) noexcept {
	const bool handed = Impl::HandOff(push);
	if (!handed) Impl::FlushReadProtect();
	++g_observation_epoch;
	return handed;
}

// (No new observation epoch: the guest learns nothing new from protections applied early.)
void PageManager::FlushDeferredProtection(uint64_t vaddr, uint64_t size) noexcept {
	Impl::TakeOverHandedOverlapping(vaddr, vaddr + size); // (KYTY_SPEC_PROTECT_ASYNC: none without it)
	Impl::FlushReadProtectOverlapping(vaddr, vaddr + size, false);
}

uint64_t PageManager::ObservationEpoch() noexcept {
	return g_observation_epoch;
}

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	if (base_addr % REGION_SIZE != 0 || base_addr >= ADDRESS_SIZE ||
	    REGION_SIZE > ADDRESS_SIZE - base_addr) {
		Fatal("invalid tracking region base 0x%016" PRIx64, base_addr);
	}

	const auto start_range = mask.FirstRange();
	const auto end_range   = mask.LastRange();
	if (start_range.first == REGION_PAGES) {
		FailFast("empty region watcher mask");
	}
	const auto first = start_range.first;
	const auto last  = end_range.second;
	if (start_range.second == end_range.second) {
		m_impl->UpdatePageWatchers<track, is_read>(base_addr + first * PAGE_SIZE,
		                                           (last - first) * PAGE_SIZE);
		return;
	}

	auto* region = track ? m_impl->GetOrCreateRegion(base_addr) : m_impl->FindRegion(base_addr);
	if (region == nullptr) {
		Fatal("untracking unknown region 0x%016" PRIx64, base_addr);
	}
	m_impl->UpdateRegionWatchers<track, is_read, true>(*region, base_addr, first, last, &mask);
}

template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);

} // namespace Libs::Graphics

// --- The table path's worker instantiation (weave.h) ----------------------------------------------------------------
//
// Included by table-xpr.inc after its structures, before the functions the two policies compile.
//
// A weave worker runs the GPU thread's executor's table functions (TableTry, TableContinue, TableDraw and what they
// call) under Weave::Worker, on its own view of the GPU thread's command buffer, recording apart. It reads the
// executor's catalog (the pairs, their programs, variants and image sets, the draw states) unlocked: the GPU thread
// holds every worker outside a draw (Weave::Hold) before it changes one of them structurally, and notes every pair, set
// and draw state it frees (Weave::NoteFreed), so a worker drops what it kept of them once the free log moved. Where the
// GPU thread's code writes what another thread reads, a worker keeps its own answer here (its lookups, resolved
// entries, ring copies, emission, last draw, proofs), leaves the write to the GPU thread's turn (a stamp, a proof or a
// need in the parcel's result, weave-parcel.h), or declines the draw (a hole: the GPU thread makes it, today's way).
//
// Plain words of the catalog the GPU thread rewrites between draws (a pair's shader generation, a set's proof stamps, a
// draw state's target epochs) are read here as hints: each is the GPU thread's proof at some draw, and a worker that
// reads an older one proves again itself. Image barrier states are read under the image's state lock; the turn emits
// the targets and transits the sets again when the images epoch moved since the parcel was made.

// The table path's census and counters under a policy: a worker's draws are counted where they are spliced.
struct TableNoCensus {
	template <typename... Args>
	explicit TableNoCensus(Args&&... /*args*/) noexcept {}
};
template <typename P>
using TableCensus = std::conditional_t<P::worker, TableNoCensus, LiveCensus::Scope>;
template <typename P>
static void TableAdd(LiveCounters::Id id, uint64_t n = 1) {
	if constexpr (!P::worker) LiveCounters::Add(id, n);
}

// A Vulkan handle's bits (a record's `uint64_t`; non-dispatchable handles are pointers on 64-bit builds).
template <typename Handle>
static uint64_t WeaveHandleBits(Handle handle) noexcept {
	if constexpr (std::is_pointer_v<Handle>)
		return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
	else
		return static_cast<uint64_t>(handle);
}

// A range none of whose pages the guest wrote since they were uploaded, with no write protection of them still queued
// (a page marked clean can change unseen until it lands): TableBlocks' proof of a held range, and a worker's of a
// buffer it takes as ObtainBuffer would find it.
template <typename Cache>
static bool TableRangeClean(Cache& buffer_cache, uint64_t address, uint64_t size) {
	for (auto region = address / TRACKER_REGION_SIZE; region <= (address + size - 1) / TRACKER_REGION_SIZE; ++region)
		if (buffer_cache.DeferredProtectionPending(region * TRACKER_REGION_SIZE)) return false;
	return !buffer_cache.IsRegionCpuModified(address, size);
}

struct RenderExecutor::WeaveScratch {
	// Its own lookups (the pair found last, the pair cache), the draw's words, slots and blocks, resolved entries, ring
	// copies (in its slabs), written ranges, held indices and arguments, and its last draw. Of the executor's table a
	// worker reads only the pairs.
	TableXpr table;
	// What its strand's buffer has bound, as NativeXprCache::emit is the GPU thread's, and the draw state it found
	// last.
	NativeXprCache::Emit      emit;
	const NativeXprDrawState* last_state     = nullptr;
	uint64_t                  last_state_key = 0;
	// The draw-state key of the draw being made (NativeXprStateKey of the worker's registers; a clean draw keeps it).
	uint64_t state_key       = 0;
	bool     state_key_valid = false;
	// The free log's clock and the worker's forget count (RenderExecutor::WeaveForgetCount) as last seen.
	uint64_t freed_seen = UINT64_MAX, forget_seen = UINT64_MAX;
	// Its proofs, direct-mapped by object: a set proven as TableValidateSet would find it (in that frame, at that
	// resolution epoch); a draw state's targets proven current (the resolution epoch); found in place (the images
	// epoch); a set's images in place under a draw state (its capture, the images epoch).
	struct SetNote {
		const void* set   = nullptr;
		uint64_t    frame = 0, epoch = 0;
	};
	struct TransitNote {
		const void* set     = nullptr;
		const void* state   = nullptr;
		uint64_t    capture = 0, epoch = 0;
	};
	std::array<SetNote, 1024>                         set_notes {};
	std::array<std::pair<const void*, uint64_t>, 256> targets_notes {}, placed_notes {};
	std::array<TransitNote, 1024>                     transit_notes {};
	// The set a pair's last draw here took, direct-mapped by pair: tried first, as the GPU thread tries
	// TablePair::last_set (which a worker doesn't read: the GPU thread rewrites it at every draw).
	std::array<std::pair<const TablePair*, TableImageSet*>, 256> set_hints {};
	// The last range noted at each of 64 page slots of the parcel's pages: a range inside one adds nothing to the union
	// the parcel's end sorts and merges. (Cleared at the parcel's first note: its pages start empty.)
	std::array<std::pair<uint64_t, uint64_t>, 64> pages_seen {};

	// The linear copies listed before the worker's parcel and not yet run as it began (the reader's records): a range
	// one writes is read from the copy's source instead (RenderExecutor::WeaveCopiesBegin, WeaveCopy, WeaveCopiesEnd
	// list them). Sorted by destination; `copies_covered` is their destinations' union.
	struct Copy {
		uint64_t dst = 0, end = 0, src = 0, index = 0, max_end = 0; // (max_end: the largest end up to this entry)
		bool     decoded = true;
	};
	std::vector<Copy>                          copies;
	std::vector<std::pair<uint64_t, uint64_t>> copies_covered;
	const std::atomic<uint64_t>*               copies_run = nullptr; // the records the GPU thread has run
	[[nodiscard]] bool                         CopiesMeet(uint64_t begin, uint64_t end) const;
	template <typename Read>
	[[nodiscard]] bool CopiesResolve(uint64_t run, uint64_t begin, uint64_t end, uint64_t limit, uint8_t* out,
	                                 Read& read);
	// [address, address + bytes) into `out` through the listed copies, when one is listed over it (false: none is, the
	// caller reads it as ever); `ok` false when a read failed or met a copy the reader could not decode.
	template <typename Read>
	[[nodiscard]] bool CopiesThrough(uint64_t address, uint64_t bytes, void* out, bool& ok, Read&& read);

	// What the strand's buffer bound is not assumed bound any more: its emission, its last draw and state, and the key.
	void ForgetDraws() noexcept {
		emit               = {};
		last_state         = nullptr;
		last_state_key     = 0;
		state_key_valid    = false;
		table.last_pair    = nullptr;
		table.last_variant = 0;
		table.last_state   = nullptr;
		table.last_set     = nullptr;
		table.images_epoch = 0;
	}
	// Pairs, sets or draw states left the executor's catalog: no pointer to one is kept.
	void ForgetObjects() noexcept {
		ForgetDraws();
		set_notes        = {};
		targets_notes    = {};
		placed_notes     = {};
		transit_notes    = {};
		set_hints        = {};
		table.pair_cache = {};
		table.found_key  = {};
		table.found_pair = nullptr;
		table.gathered   = nullptr;
		table.memo_set   = nullptr;
		table.store      = nullptr;
	}
	// The parcel's notes as a draw began: a continuation that failed leaves none of its own before the draw is tried in
	// full (its pages and copies stay: what it read).
	struct Mark {
		size_t held_uses = 0, held_words = 0, held_buffers = 0, ring_uses = 0, touched = 0, placed = 0, proofs = 0,
		       needs = 0;
	};
	[[nodiscard]] static Mark Take(const Weave::Result& result) noexcept {
		return {result.held_uses.size(), result.held_words.size(), result.held_buffers.size(), result.ring_uses.size(),
		        result.touched.size(),   result.placed.size(),     result.proofs.size(),       result.needs.size()};
	}
	static void Restore(Weave::Result& result, const Mark& mark) {
		result.held_uses.resize(mark.held_uses);
		result.held_words.resize(mark.held_words);
		result.held_buffers.resize(mark.held_buffers);
		result.ring_uses.resize(mark.ring_uses);
		result.touched.resize(mark.touched);
		result.placed.resize(mark.placed);
		result.proofs.resize(mark.proofs);
		result.needs.resize(mark.needs);
		auto& made           = result.made.back();
		made.blocks          = {};
		made.pipeline_handle = 0;
		made.cpu_indices     = false;
	}
	// A range of guest memory the worker read (the parcel's pages, which the open tests against the scout's write
	// ring).
	void NoteRange(Weave::Result& result, uint64_t address, uint64_t bytes);
	// A need: the GPU thread's turn does it at the stretch's first draw (a set's validation, a draw state's targets, a
	// set's transit under a draw state), once a stretch.
	static void NoteNeed(Weave::NeedKind kind, const NativeXprDrawState* state, const TableImageSet* set,
	                     uint64_t capture);
	// The guest's bytes for a ring copy, through the listed copies, the ranges noted (a failed read is a hole).
	[[nodiscard]] bool ReadBacking(uint64_t address, void* out, uint64_t size);

	// Whether an image the set or the draw state names by its id is retiring (Weave::ImageRetiring): its erase may come
	// while the worker reads it, so each question below that reads images leaves it to the GPU thread then.
	[[nodiscard]] static bool Retiring(const TableImageSet* set, const NativeXprDrawState* state);
	// NativeXprTargetsCurrent for a worker: its targets current at the resolution epoch of now, by the GPU thread's own
	// proof, one this worker made, or one made now, read only (no shortcut: the registration log the GPU thread appends
	// to is not read), noted for the turn, which writes it to the draw state.
	[[nodiscard]] bool TargetsCurrent(RenderExecutor& executor, const NativeXprDrawState& draw_state);
	// TableValidateSet's proof for a worker (TableSetProven), kept for the set's frame and resolution epoch.
	[[nodiscard]] bool SetProven(const RenderExecutor& executor, const TableImageSet& set);
	// The targets and the set's images in place (NativeXprTargetsPlaced, TableTransitPlaced), kept at the images epoch;
	// else a need each. A draw state whose targets were found in place owes their images' last use (Result::placed).
	void TargetsPlaced(const RenderExecutor& executor, const NativeXprDrawState& draw_state, const TableImageSet& set,
	                   uint64_t images_epoch);
	void TransitPlaced(const RenderExecutor& executor, const TableImageSet& set, const NativeXprDrawState& draw_state,
	                   uint64_t images_epoch);

	// A worker's reads of guest memory for its table evaluation (NativeXprReader's on the GPU thread). A page with no
	// read watchers is copied directly, through a guarded copy: a page can lose its read access between the test and
	// the copy, and that is a failed read, never the guest-memory handler's readback (which would wait for the GPU
	// thread). A page a pending guest write guards (KYTY_WRITE_GUARD) comes from its backing view, as the GPU thread
	// reads it. A watched page comes from the backing where the GPU thread's own proof holds for it (no GPU-dirty byte,
	// no GPU-modified image: Memory::TryReadGpuCleanBackingToHost); otherwise the read fails (the GPU thread's reader
	// may read back from the GPU, a worker may not). A range a listed copy writes is read from its source. Each range
	// read is noted in the parcel's pages.
	class Reader {
	public:
		explicit Reader(GpuResourceManager& resources)
		    : m_resources(resources), m_granules(resources.ReadGranules()), m_scratch(WeaveHere()) {}
		void               Read(uint64_t address, uint32_t count, uint32_t* out);
		[[nodiscard]] bool Failed() const noexcept { return m_failed; }

	private:
		bool ReadBytes(uint64_t address, uint64_t bytes, uint8_t* out);

		GpuResourceManager&          m_resources;
		const std::atomic<uint64_t>* m_granules;
		WeaveScratch&                m_scratch;
		uint64_t                     m_page   = UINT64_MAX;
		bool                         m_direct = false;
		bool                         m_failed = false;
	};

	// The parcel's slab with the table upload ring's interface (TableBlocks' `ring`): bytes for a block or a ring copy
	// from RenderExecutor::WeaveSlabTake, an offset being the slab's device address itself (`BufferDeviceAddress()` 0).
	// Map commits what it gives.
	struct Slab {
		[[nodiscard]] static vk::DeviceAddress      BufferDeviceAddress() noexcept { return 0; }
		[[nodiscard]] std::pair<uint8_t*, uint64_t> Map(uint64_t size, uint64_t alignment = 0, bool /*wait*/ = false) {
			const auto [at, device] = WeaveSlabTake(size, std::max<uint64_t>(alignment, 1));
			if (at == nullptr) return WeaveMiss(Weave::Miss::Ring), std::pair<uint8_t*, uint64_t> {nullptr, 0};
			return {at, device};
		}
		void                                  Commit() noexcept {}
		[[nodiscard]] std::optional<uint64_t> Copy(const void* source, uint64_t size, uint64_t alignment = 0) {
			const auto [at, offset] = Map(size, alignment);
			if (at == nullptr) return std::nullopt;
			std::memcpy(at, source, size);
			return offset;
		}
		[[nodiscard]] std::optional<uint64_t> TryCopy(const void* source, uint64_t size, uint64_t alignment = 0) {
			return Copy(source, size, alignment);
		}
	};
	Slab slab;
};

template <typename P>
RenderExecutor::TableXpr& RenderExecutor::TableOf() {
	if constexpr (P::worker)
		return WeaveHere().table;
	else
		return OwnTable();
}

template <typename P>
auto& RenderExecutor::TableEmit() {
	if constexpr (P::worker)
		return WeaveHere().emit;
	else
		return m_native_xpr->emit;
}

template <typename P>
bool RenderExecutor::TableTargetsCurrent(const NativeXprDrawState& draw_state) {
	if constexpr (P::worker)
		return WeaveHere().TargetsCurrent(*this, draw_state);
	else
		return NativeXprTargetsCurrent(draw_state);
}

RenderExecutor::WeaveScratch& RenderExecutor::WeaveHere() {
	thread_local const auto scratch = std::make_unique<WeaveScratch>();
	return *scratch;
}

void RenderExecutor::WeaveScratch::NoteRange(Weave::Result& result, uint64_t address, uint64_t bytes) {
	auto& ranges = result.pages;
	if (ranges.empty()) pages_seen.fill({0, 0});
	if (bytes == 0) return;
	if (const auto& known = pages_seen[(address >> 12u) & 63u];
	    known.first <= address && address + bytes <= known.second)
		return;
	// (Ranges mostly come in address order: one that starts inside the last one extends it.)
	if (!ranges.empty() && address >= ranges.back().first && address <= ranges.back().second) {
		ranges.back().second                           = std::max(ranges.back().second, address + bytes);
		pages_seen[(ranges.back().first >> 12u) & 63u] = ranges.back();
		return;
	}
	ranges.emplace_back(address, address + bytes);
	pages_seen[(address >> 12u) & 63u] = ranges.back();
}

void RenderExecutor::WeaveScratch::NoteNeed(Weave::NeedKind kind, const NativeXprDrawState* state,
                                            const TableImageSet* set, uint64_t capture) {
	auto&      result = *WeaveParcel().result;
	const auto from   = result.stretches.empty() ? 0u : result.stretches.back().needs.begin;
	for (auto i = from; i < result.needs.size(); ++i)
		if (const auto& need = result.needs[i]; need.kind == kind && need.state == state && need.set == set) return;
	result.needs.push_back({.kind       = kind,
	                        .state      = state,
	                        .set        = set,
	                        .capture    = capture,
	                        .first_made = static_cast<uint32_t>(result.made.size() - 1)});
}

bool RenderExecutor::WeaveScratch::CopiesMeet(uint64_t begin, uint64_t end) const {
	if (copies_covered.empty() || end <= copies_covered.front().first || begin >= copies_covered.back().second)
		return false;
	const auto after = std::upper_bound(copies_covered.begin(), copies_covered.end(), end - 1,
	                                    [](uint64_t value, const auto& range) { return value < range.first; });
	return after != copies_covered.begin() && std::prev(after)->second > begin;
}

// [begin, end) through the copies below `limit` the GPU thread has not run (`run` and on), the latest first: what such
// a copy writes is read from its source at the same offset (split at its edges; through chains of copies), the rest by
// `read`. False: a read failed, or the range is under a copy the reader could not decode.
template <typename Read>
bool RenderExecutor::WeaveScratch::CopiesResolve(uint64_t run, uint64_t begin, uint64_t end, uint64_t limit,
                                                 uint8_t* out, Read& read) {
	const Copy* best  = nullptr;
	auto        after = std::upper_bound(copies.begin(), copies.end(), end - 1,
	                                     [](uint64_t value, const Copy& copy) { return value < copy.dst; });
	while (after != copies.begin()) {
		--after;
		if (after->max_end <= begin) break;
		if (after->end > begin && after->index < limit && after->index >= run &&
		    (best == nullptr || after->index > best->index))
			best = &*after;
	}
	if (best == nullptr) return read(begin, end - begin, out);
	if (!best->decoded) return false;
	auto& used = WeaveParcel().result->copies_used;
	if (used.empty() || used.back() != best->index) used.push_back(best->index);
	const auto from = std::max(begin, best->dst), to = std::min(end, best->end);
	const auto here = best->index, source = best->src + (from - best->dst);
	if (begin < from && !CopiesResolve(run, begin, from, here, out, read)) return false;
	if (!CopiesResolve(run, source, source + (to - from), here, out + (from - begin), read)) return false;
	return to >= end || CopiesResolve(run, to, end, here, out + (to - begin), read);
}

template <typename Read>
bool RenderExecutor::WeaveScratch::CopiesThrough(uint64_t address, uint64_t bytes, void* out, bool& ok, Read&& read) {
	if (bytes == 0 || copies_run == nullptr || !CopiesMeet(address, address + bytes)) return false;
	const auto run = copies_run->load(std::memory_order_acquire);
	ok             = CopiesResolve(run, address, address + bytes, UINT64_MAX, static_cast<uint8_t*>(out), read);
	return true;
}

void RenderExecutor::WeaveCopiesBegin(const std::atomic<uint64_t>* run) {
	auto& scratch = WeaveHere();
	scratch.copies.clear();
	scratch.copies_covered.clear();
	scratch.copies_run = run;
}

void RenderExecutor::WeaveCopy(uint64_t index, uint64_t dst, uint64_t src, uint64_t bytes, bool decoded) {
	if (bytes != 0)
		WeaveHere().copies.push_back({.dst = dst, .end = dst + bytes, .src = src, .index = index, .decoded = decoded});
}

void RenderExecutor::WeaveCopiesEnd() {
	auto& scratch = WeaveHere();
	auto& copies  = scratch.copies;
	std::ranges::sort(copies, {}, &WeaveScratch::Copy::dst);
	uint64_t most = 0;
	for (auto& copy: copies) {
		most          = std::max(most, copy.end);
		copy.max_end  = most;
		auto& covered = scratch.copies_covered;
		if (!covered.empty() && copy.dst <= covered.back().second)
			covered.back().second = std::max(covered.back().second, copy.end);
		else
			covered.emplace_back(copy.dst, copy.end);
	}
	if (copies.empty()) scratch.copies_run = nullptr;
}

bool RenderExecutor::WeaveScratch::ReadBacking(uint64_t address, void* out, uint64_t size) {
	auto&      result = *WeaveParcel().result;
	const auto piece  = [&](uint64_t at, uint64_t bytes, uint8_t* to) {
        NoteRange(result, at, bytes);
        return Libs::LibKernel::Memory::TryReadBackingToHost(at, to, bytes);
	};
	if (bool ok = true; CopiesThrough(address, size, out, ok, piece)) return ok;
	return piece(address, size, static_cast<uint8_t*>(out));
}

void RenderExecutor::WeaveScratch::Reader::Read(uint64_t address, uint32_t count, uint32_t* out) {
	const uint64_t bytes  = uint64_t {count} * 4u;
	auto&          result = *WeaveParcel().result;
	m_scratch.NoteRange(result, address, bytes);
	bool       ok    = true;
	const auto piece = [&](uint64_t at, uint64_t size, uint8_t* to) {
		if (at != address) m_scratch.NoteRange(result, at, size);
		return ReadBytes(at, size, to);
	};
	if (m_scratch.CopiesThrough(address, bytes, out, ok, piece)) {
		if (!ok) {
			std::memset(out, 0, bytes);
			m_failed = true;
		}
		return;
	}
	if (!ReadBytes(address, bytes, reinterpret_cast<uint8_t*>(out))) m_failed = true;
}

bool RenderExecutor::WeaveScratch::Reader::ReadBytes(uint64_t address, uint64_t bytes, uint8_t* out) {
	const uint64_t first = address >> 12u, last = (address + bytes - 1u) >> 12u;
	bool           direct = false;
	if (first == last) {
		if (first != m_page) {
			m_page             = first;
			const auto granule = first >> (PageManager::READ_GRANULE_BITS - 12u);
			m_direct           = (address < TRACKER_ADDRESS_SIZE &&
                        ((m_granules[granule / 64u].load(std::memory_order_acquire) >> (granule % 64u)) & 1u) == 0) ||
			           !m_resources.HasReadWatchers(first << 12u, 4096);
		}
		direct = m_direct;
	} else {
		direct = !m_resources.HasReadWatchers(first << 12u, (last - first + 1u) << 12u);
	}
	if (direct) {
		if (WriteGuard::g_count.load(std::memory_order_relaxed) != 0) {
			if (const auto* backing = WriteGuard::GuardedBacking(address, bytes)) {
				std::memcpy(out, backing, bytes);
				return true;
			}
		}
#if !defined(_WIN32)
		// Linux: Common::GuardedCopy is a pread of /proc/self/mem there, a system call per read under the mmap
		// lock, and it reads a page a read watcher protected after the test above (stale bytes, where Windows' copy
		// stops at the fault). So as mode 4's workers read (Spec::WorkerCopy): such a page fails the read
		// (Spec::WorkerReadable, asked, never probed), and the rest comes from the backing view (live
		// spec_backing_reads 1, the default; 0: the guarded copy).
		if (Spec::WorkerReadable(reinterpret_cast<const void*>(address), bytes) != bytes) {
			std::memset(out, 0, bytes);
			return false;
		}
		static auto& backing_reads = LiveSwitch::Find("spec_backing_reads", 1);
		if (backing_reads.load(std::memory_order_relaxed) != 0 &&
		    Libs::LibKernel::Memory::TryReadBackingToHost(address, out, bytes))
			return true;
#endif
		if (Common::GuardedCopy(out, reinterpret_cast<const void*>(address), bytes) == 0) return true;
	} else if (Libs::LibKernel::Memory::TryReadGpuCleanBackingToHost(address, out, bytes)) {
		return true;
	}
	std::memset(out, 0, bytes);
	return false;
}

bool RenderExecutor::WeaveScratch::Retiring(const TableImageSet* set, const NativeXprDrawState* state) {
	if (!Weave::ImagesRetiring()) return false;
	if (set != nullptr)
		for (const auto& dep: set->images)
			if (Weave::ImageRetiring(dep.id.index, dep.id.generation)) return true;
	if (state != nullptr)
		for (const auto& target: state->targets)
			if (Weave::ImageRetiring(target.id.index, target.id.generation)) return true;
	return false;
}

bool RenderExecutor::WeaveScratch::TargetsCurrent(RenderExecutor& executor, const NativeXprDrawState& draw_state) {
	if (NativeXprHudlessTarget(draw_state)) return WeaveMiss(Weave::Miss::Targets), false;
	const auto epoch = executor.m_context.GetTextureCache().ResolutionEpoch();
	if (std::atomic_ref<uint64_t>(draw_state.targets_checked).load(std::memory_order_relaxed) == epoch) return true;
	auto& note = targets_notes[(reinterpret_cast<uintptr_t>(&draw_state) * 0x9e3779b97f4a7c15ull) >> 56u];
	if (note.first == &draw_state && note.second == epoch) return true;
	NativeXprTargetsProof proof;
	if (Retiring(nullptr, &draw_state) || !executor.NativeXprTargetsProve(draw_state, proof, false))
		return WeaveMiss(Weave::Miss::Targets), false;
	note = {&draw_state, proof.checked};
	WeaveParcel().result->proofs.push_back({.state = &draw_state,
	                                        .proof = {.epoch      = proof.epoch,
	                                                  .erased     = proof.erased,
	                                                  .checked    = proof.checked,
	                                                  .span_begin = proof.span_begin,
	                                                  .span_end   = proof.span_end}});
	return true;
}

bool RenderExecutor::WeaveScratch::SetProven(const RenderExecutor& executor, const TableImageSet& set) {
	const auto frame      = executor.m_context.FrameNumber();
	const auto resolution = executor.m_context.GetTextureCache().ResolutionEpoch();
	auto&      note       = set_notes[(reinterpret_cast<uintptr_t>(&set) * 0x9e3779b97f4a7c15ull) >> 54u];
	if (note.set == &set && note.frame == frame && note.epoch == resolution) return true;
	if (Retiring(&set, nullptr) || !executor.TableSetProven(set)) return false;
	note = {&set, frame, resolution};
	return true;
}

void RenderExecutor::WeaveScratch::TargetsPlaced(const RenderExecutor& executor, const NativeXprDrawState& draw_state,
                                                 const TableImageSet& set, uint64_t images_epoch) {
	auto& note = placed_notes[(reinterpret_cast<uintptr_t>(&draw_state) * 0x9e3779b97f4a7c15ull) >> 56u];
	if (note.first != &draw_state || note.second != images_epoch) {
		if (Retiring(nullptr, &draw_state) || !executor.NativeXprTargetsPlaced(draw_state))
			return NoteNeed(Weave::NeedKind::Targets, &draw_state, &set, draw_state.capture_id);
		note = {&draw_state, images_epoch};
	}
	WeaveParcel().result->placed.push_back(&draw_state);
}

void RenderExecutor::WeaveScratch::TransitPlaced(const RenderExecutor& executor, const TableImageSet& set,
                                                 const NativeXprDrawState& draw_state, uint64_t images_epoch) {
	auto& note = transit_notes[((reinterpret_cast<uintptr_t>(&set) ^ (reinterpret_cast<uintptr_t>(&draw_state) << 1u)) *
	                            0x9e3779b97f4a7c15ull) >>
	                           54u];
	if (note.set == &set && note.state == &draw_state && note.capture == draw_state.capture_id &&
	    note.epoch == images_epoch)
		return;
	if (Retiring(&set, nullptr) || !executor.TableTransitPlaced(set, &draw_state))
		return NoteNeed(Weave::NeedKind::Transit, &draw_state, &set, draw_state.capture_id);
	note = {&set, &draw_state, draw_state.capture_id, images_epoch};
}

// A worker's table draw, as the command processor's entries make the GPU thread's (TableContinue for a clean draw, else
// NativeXprTry's checks, the draw-state key and TableTry), under the worker's policy. Called by WeaveDraw
// (weave-worker.inc), which holds the worker's DrawScope and has emplaced the draw's record (result->made.back()).
RenderExecutor::TableResult RenderExecutor::WeaveTableDraw(CommandBuffer& view, std::span<const uint64_t> commands,
                                                           uint64_t index_base, uint64_t index_bytes,
                                                           vk::IndexType index_type, const NativeXprDirectDraw* direct,
                                                           bool clean) {
	auto& scratch = WeaveHere();
	// (What it kept of the catalog is dropped once anything left it, and what its buffer bound once the worker began a
	// parcel, a stretch or a strand, or made a hole. Read inside the draw's DrawScope: nothing is freed until it ends.)
	if (const auto freed = Weave::FreedClock(); freed != scratch.freed_seen) {
		scratch.ForgetObjects();
		scratch.freed_seen = freed;
	}
	if (const auto forget = WeaveForgetCount(); forget != scratch.forget_seen) {
		scratch.ForgetDraws();
		scratch.forget_seen = forget;
	}
	if (!clean) scratch.state_key_valid = false;
	// As NativeXprTry's: its table path on, a valid buffer and a draw form the table path takes; not a draw of vertices
	// whose arguments the GPU reads (`indirect`, mode 4's form).
	const bool indexed = direct == nullptr || direct->indexed;
	if (kyty_local_table_xpr_mode.load(std::memory_order_relaxed) == 0 || !m_table_xpr || !m_native_xpr ||
	    view.IsInvalid() || (direct != nullptr && direct->indirect) ||
	    (direct == nullptr ? commands.empty() : !commands.empty()) || commands.size() > 64 ||
	    (indexed && (index_base == 0 || index_bytes == 0)))
		return WeaveMiss(Weave::Miss::Packet), TableResult::Native;
	if (clean) {
		auto&      result = *WeaveParcel().result;
		const auto mark   = WeaveScratch::Take(result);
		if (TableContinue<Weave::Worker>(view, commands, index_base, index_bytes, index_type, direct))
			return TableResult::Drawn;
		WeaveScratch::Restore(result, mark);
	}
	if (!clean || !scratch.state_key_valid) {
		// (The worker's registers are hashed in full: the draw state observer may not have seen this thread's packets.)
		DrawStateObserver::g_shadow.key_changes |= 1u | 4u;
		scratch.state_key       = NativeXprStateKey(view);
		scratch.state_key_valid = true;
	}
	return TableTry<Weave::Worker>(view, commands, index_base, index_bytes, index_type, direct);
}

// The worker's instantiation, compiled whether or not anything calls it yet (and what it calls with it).
template RenderExecutor::TableResult RenderExecutor::TableTry<Weave::Worker>(CommandBuffer&, std::span<const uint64_t>,
                                                                             uint64_t, uint64_t, vk::IndexType,
                                                                             const NativeXprDirectDraw*);
template bool RenderExecutor::TableContinue<Weave::Worker>(CommandBuffer&, std::span<const uint64_t>, uint64_t,
                                                           uint64_t, vk::IndexType, const NativeXprDirectDraw*);

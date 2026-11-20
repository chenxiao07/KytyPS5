// --- The weave on a worker (weave.h, docs/WEAVE.md): a parcel's draws, holes, strands and lists -------------------
//
// Included by renderDraw.cpp after weave-draw.inc (local Vulkan recording builds only). The command processor's walk of
// a parcel (weave-reader.inc) brings each of its draws here, on the worker's thread, with the worker's own registers:
// WeaveDraw makes it with the table path's worker instantiation (WeaveTableDraw, weave-draw.inc) into the worker's view
// of the GPU thread's command buffer, which records through the apart marker into a buffer of the worker's own
// (vulkan-recording.h), and the draw's notes go into the parcel's result. Here: a parcel's begin and end; its stretches
// (from a cut of the reader's or a render pass's first draw: its needs are grouped by them); its strands (the calls
// between two holes or cuts, one apart buffer each, spliced by the GPU thread in parts); its holes; the slab of the
// worker's ring its blocks and copies go into; and at its end what the GPU thread's turn reads (weave-turn.inc): each
// strand's lists, the objects its draws used, its pages, the register log's links.
//
// A worker writes only its own state and its parcel's result. It reads the GPU thread's structures only inside a
// Weave::DrawScope, which keeps it out while the GPU thread changes one under a Weave::Hold, and the turn reads no
// pair, set or draw state of a parcel before the free log shows it alive (Weave::FreedHits).

// A worker's upload ring: RingBytes of host-coherent memory with a device address, whose slabs its parcels' blocks and
// copies go into. The worker takes a slab at a parcel's begin (Take) and keeps what it used at its end (Keep). The slab
// is released once (Release): by the GPU thread with the tick of the command buffer the parcel's calls were spliced
// into, or with none when nothing of it was spliced. The GPU thread frees released slabs in order once their tick is
// done (Reclaim). An empty slab keeps SlabKeepBytes: kept at 0 it left the head where it was, and with the slabs before
// it freed, the tail met the head while it was live and Take read the ring as full until it was freed too.
struct WeaveRing {
	enum : uint32_t { Free, Writing, Done, Claimed, Released };
	struct Record {
		std::atomic<uint32_t> state {Free};
		uint64_t              id = 0, begin = 0, end = 0, tick = 0;
	};
	std::unique_ptr<StreamBuffer>          buffer;
	uint8_t*                               mapped  = nullptr; // (null: no ring)
	vk::DeviceAddress                      address = 0;
	std::array<Record, Weave::RingRecords> records;
	uint64_t                               head = 0, made = 0;      // the worker's
	std::atomic<uint64_t>                  tail {0}, reclaimed {0}; // the GPU thread's (Reclaim)

	// The worker: a slab of SlabBytes, its record and where it begins (-1: no room, or every record live).
	int64_t Take(uint64_t& begin) {
		const auto live = made - reclaimed.load(std::memory_order_acquire);
		if (live >= Weave::RingRecords) return -1;
		const auto t  = tail.load(std::memory_order_acquire);
		uint64_t   at = 0;
		if (live == 0 || head > t) {
			// (Free: [head, RingBytes) and [0, tail).)
			if (Weave::RingBytes - head >= Weave::SlabBytes)
				at = head;
			else if (live == 0 || Weave::SlabBytes <= t)
				at = 0;
			else
				return -1;
		} else if (head + Weave::SlabBytes <= t) {
			at = head; // (free: [head, tail))
		} else {
			return -1;
		}
		auto& record = records[made % Weave::RingRecords];
		record.id    = made;
		record.begin = record.end = at;
		record.tick               = 0;
		record.state.store(Writing, std::memory_order_relaxed);
		begin = at;
		return static_cast<int64_t>(made++);
	}
	// The worker, at the parcel's end: the slab ends after what it used (Take placed it where a whole slab fits).
	void Keep(int64_t id, uint64_t used) {
		auto&          record = records[static_cast<uint64_t>(id) % Weave::RingRecords];
		const uint64_t kept   = std::max(used, Weave::SlabKeepBytes);
		record.end            = record.begin + (kept + Weave::RingAlign - 1) / Weave::RingAlign * Weave::RingAlign;
		head                  = record.end;
		record.state.store(Done, std::memory_order_release);
	}
	// Any thread, once the slab is kept: released with `tick` (0: none). False: it was released already (its record is
	// another slab's by now, which the claim puts back).
	bool Release(int64_t id, uint64_t tick) {
		auto&    record   = records[static_cast<uint64_t>(id) % Weave::RingRecords];
		uint32_t expected = Done;
		if (!record.state.compare_exchange_strong(expected, Claimed, std::memory_order_acq_rel)) return false;
		if (record.id != static_cast<uint64_t>(id)) {
			record.state.store(Done, std::memory_order_release);
			return false;
		}
		record.tick = tick;
		record.state.store(Released, std::memory_order_release);
		return true;
	}
	// The GPU thread: the released slabs, in order, whose tick is done. (A record's other fields are read only once its
	// state says the worker and the release wrote them.)
	void Reclaim(CommandScheduler& scheduler) {
		for (auto at = reclaimed.load(std::memory_order_relaxed);; ++at) {
			auto& record = records[at % Weave::RingRecords];
			if (record.state.load(std::memory_order_acquire) != Released || record.id != at) return;
			if (record.tick != 0 && !scheduler.IsFree(record.tick)) return;
			tail.store(record.end, std::memory_order_release);
			record.state.store(Free, std::memory_order_relaxed);
			reclaimed.store(at + 1, std::memory_order_release);
		}
	}
};

// A worker's own state: its view of the GPU thread's command buffer (aimed at the apart marker), its ring, and the
// parcel it is making. Only its thread touches it while a parcel is made, but for the ring's GPU-thread side.
struct WeaveWorker {
	uint32_t                       index = 0;
	std::unique_ptr<CommandBuffer> view;
	WeaveRing                      ring;
	// The parcel being made (null between parcels), its result, and its slot's notes beside it (WeaveTurn::Begun).
	Weave::Parcel* parcel = nullptr;
	Weave::Result* result = nullptr;
	void*          begun  = nullptr;
	// Its slab: mapped bytes, their device address, the bytes taken (null: no slab).
	uint8_t*          slab_mapped = nullptr;
	vk::DeviceAddress slab_device = 0;
	uint64_t          slab_used   = 0;
	// The draw being made: its first packet, why the table path declined it, whether its pass change began its
	// pass inside the strand.
	const uint32_t* draw_packet = nullptr;
	Weave::Miss     miss        = Weave::Miss::Other;
	bool            pass_opened = false;
	// Moves whenever nothing the view's buffer has bound may be assumed bound (WeaveForgetCount).
	uint64_t forget = 0;
	// An apart buffer is open; the next made draw begins a strand (its buffer holds no made draw yet); a hole since
	// the last strand began; the last made draw began one.
	bool recording = false, pending = true, hole = false, started = false;
	// The last draw's render pass (a change of it begins a stretch).
	bool        pass_known = false;
	RenderState pass;
	// Counted once at the parcel's end.
	uint32_t holes = 0, holes_buffers = 0, holes_ring = 0;
};

// The worker of the calling thread (null on every other thread): set at each parcel's begin.
static thread_local WeaveWorker* t_weave_worker = nullptr;

// The depth clear the normal path's target acquisition would make at a draw of a draw state now (AcquireRenderTargets):
// the rendering to begin the draw's pass with (its state's, the depth attachment's clear flags and values as that
// function sets them), the depth target's metadata and slice whose pending clear it drops, and whether the turn can't
// make it so (`refused`).
struct RenderExecutor::WeaveClear {
	RenderState rendering;
	uint64_t    metadata = 0;
	uint32_t    slice    = 0;
	bool        refused  = false;
};

// The weave's executor state: the workers', and the GPU thread's turn over the open parcel (weave-turn.inc). One,
// made by WeaveStart for the executor every weave call is made on (the GPU thread's: WeaveCp::executor,
// weave-reader.inc), and deleted with that executor (~RenderExecutor, at shutdown), or when the start failed
// (WeaveForget).
struct WeaveTurn {
	const RenderExecutor*                      executor = nullptr;
	std::array<WeaveWorker, Weave::MaxWorkers> workers;
	uint32_t                                   worker_count = 0;
	// What a parcel slot's worker notes beside its result (by the parcel's serial % ParcelSlots; written by the worker
	// that makes the parcel, read by the GPU thread once it is Done): the epochs as it began that the result's clocks
	// leave out (the mapping's, the images', the CPU-dirty images'), its slab's record, and by strand whether it began
	// after a hole (the narrow walk test's).
	struct Begun {
		uint64_t             mapping = 0, images = 0, cpu_dirty = 0;
		int64_t              slab = -1;
		std::vector<uint8_t> after_hole;
	};
	std::array<Begun, Weave::ParcelSlots> begun;
	// The narrow walk test: whether the GPU thread's processor's registers are the worker's before a strand's made draw
	// `index` (`first` its first; CommandProcessor::WeaveSameOf).
	const void* processor        = nullptr;
	bool (*same)(const void* processor, const Weave::Registers& start, const Weave::RegisterLog& log, uint32_t first,
	             uint32_t index) = nullptr;
	// The open parcel's turn, strand by strand.
	struct StrandTurn {
		bool     tested = false, seen = false, late = false, spliced = false, counted = false;
		uint32_t prefix_end = 0; // its made draws before this may be spliced (absolute)
		uint32_t resume_at  = 0; // the next part's first made draw, when the last part ended before a pass (0: none)
	};
	// The turn's actions for the strand last tested, by made draw (ascending): a held use copied again (its index in
	// Result::held_uses), an index range uploaded, a set's CPU-written sampled images refreshed.
	struct Recopy {
		uint32_t made = 0, use = 0;
	};
	struct Upload {
		uint32_t made    = 0;
		uint64_t address = 0, size = 0;
	};
	struct Refresh {
		uint32_t    made  = 0;
		const void* set   = nullptr;
		const void* state = nullptr;
	};
	// A target need refused only for a depth target's pending clear: the strand's first draw's (deferred to its splice,
	// which makes the clear with its pass), and later draws' of the same strand on that clear (done once it is made).
	struct Deferred {
		uint32_t    made  = UINT32_MAX;
		const void* state = nullptr;
		const void* set   = nullptr;
	};
	struct Open {
		Weave::Parcel*          parcel = nullptr; // (null: none open)
		Weave::Result*          result = nullptr;
		const Begun*            begun  = nullptr;
		std::vector<StrandTurn> strands;
		std::vector<uint32_t>   refused; // by stretch: the first made draw a refused need cuts from (UINT32_MAX: none)
		uint32_t                cursor = 0, stretch_cursor = 0;
		// What the needs put in place at needs_packet: the layouts there don't do it again.
		const uint32_t*                                  needs_packet = nullptr;
		std::vector<const void*>                         needs_states;
		std::vector<std::pair<const void*, const void*>> needs_sets;
		Deferred                                         clear;
		uint64_t                                         clear_metadata = 0;
		uint32_t                                         clear_slice    = 0;
		std::vector<Deferred>                            clear_later;
		std::vector<Recopy>                              recopies;
		std::vector<Upload>                              uploads;
		std::vector<Refresh>                             refreshes;
		// The part just spliced, for the skip.
		bool     skip       = false;
		uint32_t skip_begin = 0, skip_end = 0;
		// Counted at the close.
		uint64_t own = 0, spliced = 0, strands_spliced = 0, strands_cut = 0;
		// The free log over the parcel's objects (Result::objects), read from the parcel's begin clock and each entry
		// once: how far, the objects freed since the parcel began, and whether it turned over (none provably alive).
		uint64_t                 freed_scanned = 0;
		std::vector<const void*> freed_hits;
		bool                     freed_lost = false;
		// The entries since the last read; whether `object` (one of the parcel's) may have been freed since it began.
		void FreedRead() {
			if (!freed_lost && Weave::FreedHits(freed_scanned, result->objects, freed_hits) == 2) freed_lost = true;
		}
		[[nodiscard]] bool Freed(const void* object) const {
			return freed_lost || (object != nullptr && std::ranges::find(freed_hits, object) != freed_hits.end());
		}
	} open;
	// The texture cache's image start log, as the open's test last read it (a copy of every (epoch, address) with
	// from < epoch <= seen).
	struct Starts {
		std::vector<std::pair<uint64_t, uint64_t>> recent;
		std::vector<uint64_t>                      addresses;
		uint64_t                                   from = 0, seen = 0;
	} starts;
};

// (Null until WeaveStart. The weave's threads read it, a worker at each parcel's begin and any thread at a slot's
// release: WeaveStart wrote it before they started, and it is deleted only once StopAll has joined them, or once a
// start that failed has joined them, WeaveForget.)
static WeaveTurn* g_weave_turn = nullptr;

namespace {

// An exact set of the entries a list holds, by their index in it: open addressing on a hash of an entry's key, emptied
// by a new stamp. Past half its slots it takes no more (`full`), and the caller searches the rest of the list itself.
struct WeaveSeen {
	static constexpr uint32_t  Size = 4096;
	std::array<uint32_t, Size> at {}, stamp {};
	uint32_t                   now = 0, count = 0;

	void Clear() {
		if (++now == 0) {
			stamp.fill(0);
			now = 1;
		}
		count = 0;
	}
	// True: an entry `same` holds is in the set. Else `index` goes in under `hash`, unless the set is full.
	template <class Same>
	bool Seen(uint64_t hash, uint32_t index, Same&& same, bool& full) {
		full = false;
		for (auto slot = static_cast<uint32_t>(hash) & (Size - 1);; slot = (slot + 1) & (Size - 1)) {
			if (stamp[slot] != now) {
				if (count >= Size / 2) return full = true, false;
				stamp[slot] = now;
				at[slot]    = index;
				++count;
				return false;
			}
			if (same(at[slot])) return true;
		}
	}
};

uint64_t WeaveHash(std::initializer_list<uint64_t> values) {
	uint64_t h = 0x9e3779b97f4a7c15ull;
	for (const auto v: values) {
		h ^= v + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
		h ^= h >> 31u;
		h *= 0xbf58476d1ce4e5b9ull;
	}
	return h ^ (h >> 29u);
}

uint64_t WeavePointer(const void* object) {
	return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(object));
}

// `entry` added to `list` (from `begin`, the strand's part of it) unless an equal one is there.
template <typename T, typename Same>
void WeaveAddOnce(std::vector<T>& list, size_t begin, const T& entry, uint64_t hash, WeaveSeen& seen, Same&& same) {
	bool full = false;
	if (seen.Seen(hash, static_cast<uint32_t>(list.size()), [&](uint32_t k) { return same(list[k], entry); }, full))
		return;
	if (full)
		for (auto k = begin; k < list.size(); ++k)
			if (same(list[k], entry)) return;
	list.push_back(entry);
}

// At a parcel's end, on its worker: what the turn reads of the made draws, made once here rather than by the GPU
// thread. The pages read, sorted by their start and merged (the open's mapping and write-ring tests take them in one
// call each). The pairs, sets and draw states the draws used, sorted and unique (the free log's test). The buffers they
// took, each once (touched at the open). Each strand's lists: its distinct items (pair, draw state, set, and the
// pipeline of a draw that bound one), each with the first of its draws that holds it; its distinct ranges (held V#s,
// held index and argument ranges, slab copies), with their first draw too; its least epochs; its stamps (pairs, sets
// and draw states once each, with the last draw that holds them, whose variant and proof the stamp takes; pipelines in
// the order of their first bind; draw states whose targets were found in place, in the order of their last use). The
// register log's links: each block's next write of the same block, so the skip applies only the last.
void WeaveLists(Weave::Result& r) {
	std::sort(r.pages.begin(), r.pages.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
	size_t kept = 0;
	for (size_t i = 0; i < r.pages.size(); ++i) {
		if (kept != 0 && r.pages[i].first <= r.pages[kept - 1].second)
			r.pages[kept - 1].second = std::max(r.pages[kept - 1].second, r.pages[i].second);
		else
			r.pages[kept++] = r.pages[i];
	}
	r.pages.resize(kept);
	r.objects.clear();
	for (const auto& made: r.made)
		for (const void* object: {made.pair, made.set, made.state})
			if (object != nullptr) r.objects.push_back(object);
	std::sort(r.objects.begin(), r.objects.end());
	r.objects.erase(std::unique(r.objects.begin(), r.objects.end()), r.objects.end());
	const auto touch_key = [](const Weave::Touch& touch) {
		return std::tuple {touch.handle, touch.buffer, touch.address, touch.size};
	};
	r.touched_unique = r.touched;
	std::sort(r.touched_unique.begin(), r.touched_unique.end(),
	          [&](const auto& x, const auto& y) { return touch_key(x) < touch_key(y); });
	r.touched_unique.erase(std::unique(r.touched_unique.begin(), r.touched_unique.end(),
	                                   [&](const auto& x, const auto& y) { return touch_key(x) == touch_key(y); }),
	                       r.touched_unique.end());

	thread_local WeaveSeen seen;
	auto&                  lists = r.lists;

	const auto same_item = [](const Weave::ListItem& x, const Weave::ListItem& y) {
		return x.pair == y.pair && x.state == y.state && x.set == y.set && x.pipeline == y.pipeline;
	};
	const auto same_range = [](const Weave::ListRange& x, const Weave::ListRange& y) {
		return x.kind == y.kind && x.begin == y.begin && x.end == y.end && x.cpu_epoch == y.cpu_epoch;
	};
	const auto same_object = [](const void* x, const void* y) { return x == y; };

	const auto range_of = [](const auto& list, size_t begin) {
		return Weave::Range {static_cast<uint32_t>(begin), static_cast<uint32_t>(list.size())};
	};
	for (auto& strand: r.strands) {
		const uint32_t first = strand.first_made, last = strand.first_made + strand.count;
		strand.mapping = strand.registration = strand.resolution = UINT64_MAX;
		// Items and ranges, each with the first draw holding it: the draws come in order.
		const auto items_begin = lists.items.size();
		seen.Clear();
		for (auto i = first; i < last; ++i) {
			const auto& made    = r.made[i];
			strand.mapping      = std::min(strand.mapping, made.mapping);
			strand.registration = std::min(strand.registration, made.registration);
			strand.resolution   = std::min(strand.resolution, made.resolution);
			const Weave::ListItem item {.pair     = made.pair,
			                            .state    = made.state,
			                            .set      = made.set,
			                            .pipeline = made.pipeline_handle != 0 ? made.pipeline : nullptr,
			                            .first    = i - first};
			WeaveAddOnce(lists.items, items_begin, item,
			             WeaveHash({WeavePointer(item.pair), WeavePointer(item.state), WeavePointer(item.set),
			                        WeavePointer(item.pipeline)}),
			             seen, same_item);
		}
		strand.items            = range_of(lists.items, items_begin);
		const auto ranges_begin = lists.ranges.size();
		seen.Clear();
		const auto add_range = [&](const Weave::ListRange& range) {
			WeaveAddOnce(lists.ranges, ranges_begin, range,
			             WeaveHash({static_cast<uint64_t>(range.kind), range.begin, range.end, range.cpu_epoch}), seen,
			             same_range);
		};
		for (auto i = first; i < last; ++i) {
			const auto& made = r.made[i];
			for (auto k = made.held_uses.begin; k < made.held_uses.end; ++k) {
				const auto& use = r.held_uses[k];
				add_range({.kind      = Weave::RangeKind::Held,
				           .first     = i - first,
				           .begin     = use.address,
				           .end       = use.address + use.size,
				           .cpu_epoch = use.cpu_epoch});
			}
			for (auto k = made.held_buffers.begin; k < made.held_buffers.end; ++k)
				if (const auto& use = r.held_buffers[k]; use.end > use.begin)
					add_range({.kind      = Weave::RangeKind::HeldBuffer,
					           .first     = i - first,
					           .begin     = use.begin,
					           .end       = use.end,
					           .cpu_epoch = use.cpu_epoch});
			for (auto k = made.ring_uses.begin; k < made.ring_uses.end; ++k) {
				const auto& use = r.ring_uses[k];
				add_range({.kind  = Weave::RangeKind::Ring,
				           .first = i - first,
				           .begin = use.address,
				           .end   = use.address + use.size});
			}
		}
		strand.ranges = range_of(lists.ranges, ranges_begin);
		// Stamps: the draws from the last, so each object's is its last draw's.
		const auto stamps_begin = lists.stamps.size();
		seen.Clear();
		for (auto i = last; i-- > first;) {
			const auto& made = r.made[i];

			const auto add = [&](const Weave::ListItem& stamp, const void* object) {
				WeaveAddOnce(lists.stamps, stamps_begin, stamp, WeaveHash({WeavePointer(object)}), seen, same_item);
			};
			if (made.pair != nullptr) add({.pair = made.pair, .first = i - first}, made.pair);
			if (made.set != nullptr) add({.set = made.set, .first = i - first}, made.set);
			if (made.state != nullptr) add({.state = made.state, .first = i - first}, made.state);
		}
		strand.stamps              = range_of(lists.stamps, stamps_begin);
		const auto pipelines_begin = lists.pipelines.size();
		seen.Clear();
		for (auto i = first; i < last; ++i)
			if (const auto& made = r.made[i]; made.pipeline_handle != 0 && made.pipeline != nullptr)
				WeaveAddOnce(lists.pipelines, pipelines_begin, made.pipeline, WeaveHash({WeavePointer(made.pipeline)}),
				             seen, same_object);
		strand.pipelines = range_of(lists.pipelines, pipelines_begin);
		// (Buffers are touched once a parcel, at its open: no strand lists them.)
		strand.touches          = range_of(lists.touches, lists.touches.size());
		const auto placed_begin = lists.placed.size();
		seen.Clear();
		for (auto i = last; i-- > first;) {
			const auto& made = r.made[i];
			for (auto k = made.placed.end; k-- > made.placed.begin;)
				WeaveAddOnce(lists.placed, placed_begin, r.placed[k], WeaveHash({WeavePointer(r.placed[k])}), seen,
				             same_object);
		}
		std::reverse(lists.placed.begin() + static_cast<std::ptrdiff_t>(placed_begin), lists.placed.end());
		strand.placed = range_of(lists.placed, placed_begin);
	}
	// The register log, when it holds every made draw: each block's next write of the same block id.
	auto& log = r.log;
	if (log.on && log.draws.size() != r.made.size()) log.on = false;
	log.block_next.clear();
	if (log.on) {
		thread_local std::vector<uint32_t> next_of;
		log.block_next.assign(log.block_ids.size(), UINT32_MAX);
		for (size_t k = log.block_ids.size(); k-- > 0;) {
			const auto id = log.block_ids[k];
			if (id >= next_of.size()) next_of.resize(size_t {id} + 1, UINT32_MAX);
			log.block_next[k] = next_of[id];
			next_of[id]       = static_cast<uint32_t>(k);
		}
		for (const auto id: log.block_ids)
			next_of[id] = UINT32_MAX;
	}
}

} // namespace

bool RenderExecutor::WeaveParcelBegin(uint32_t worker, Weave::Parcel& parcel, HW::Context& registers,
                                      HW::UserConfig& user_config, HW::Shader& shaders) {
	if (!g_weave_turn || worker >= g_weave_turn->worker_count || !g_weave_turn->workers[worker].view) return false;
	auto& t        = *g_weave_turn;
	auto& w        = t.workers[worker];
	t_weave_worker = &w;
	if (!parcel.result) parcel.result = std::make_shared<Weave::Result>();
	auto& r = *parcel.result;
	// (The apart buffers the slot's last parcel left unspliced are destroyed here, on the worker, their calls with
	// them. The vectors keep their memory from parcel to parcel.)
	for (auto& strand: r.strands)
		strand.buffer = {};
	r.made.clear();
	r.held_uses.clear();
	r.held_buffers.clear();
	r.ring_uses.clear();
	r.held_words.clear();
	r.touched.clear();
	r.touched_unique.clear();
	r.placed.clear();
	r.proofs.clear();
	r.needs.clear();
	r.stretches.clear();
	r.strands.clear();
	r.lists.items.clear();
	r.lists.ranges.clear();
	r.lists.stamps.clear();
	r.lists.pipelines.clear();
	r.lists.touches.clear();
	r.lists.placed.clear();
	r.log.draws.clear();
	r.log.block_ids.clear();
	r.log.block_bytes.clear();
	r.log.block_next.clear();
	r.log.counts.clear();
	r.log.on = false;
	r.pages.clear();
	r.objects.clear();
	r.copies_used.clear();
	r.spoiled = r.lost = false;
	// The clocks, before the parcel's first read: a host write after it is a later write clock, a moved epoch a later
	// value, an object freed after it a later entry of the free log. The pipelines evicted, before any handle is read.
	auto& textures      = m_context.GetTextureCache();
	auto& begun         = t.begun[parcel.serial % Weave::ParcelSlots];
	r.write_clock       = Scout::WriteClock();
	r.freed_clock       = Weave::FreedClock();
	r.shader_generation = ShaderMapGeneration();
	r.image_start_epoch = textures.StartEpoch();
	r.bda_generation    = BufferCache::BdaTableGeneration();
	r.evictions         = PipelineEvictionCount();
	begun.mapping       = m_context.GetGpuResources().MappingEpoch();
	begun.images        = textures.ImagesEpoch();
	begun.cpu_dirty     = Image::CpuDirtyEpoch();
	begun.after_hole.clear();
	// Its slab (none: every draw that needs a block is a hole, and the parcel is never opened).
	begun.slab    = -1;
	r.slab        = {};
	w.slab_mapped = nullptr;
	w.slab_device = 0;
	w.slab_used   = 0;
	if (w.ring.mapped != nullptr) {
		uint64_t at = 0;
		if (const auto id = w.ring.Take(at); id >= 0) {
			begun.slab    = id;
			r.slab.ring   = static_cast<int32_t>(worker);
			r.slab.offset = at;
			w.slab_mapped = w.ring.mapped + at;
			w.slab_device = w.ring.address + at;
		} else {
			LiveCounters::Add(LiveCounters::WeaveSlabRefused);
		}
	}
	w.parcel = &parcel;
	w.result = &r;
	w.begun  = &begun;
	// Its view draws with the worker's registers, in no render pass yet: the turn begins a strand's first.
	w.view->Bind(registers, user_config, shaders);
	w.view->m_rendering    = false;
	w.view->m_render_state = {};
	w.pending              = true;
	w.hole = w.started = w.pass_known = w.pass_opened = false;
	w.holes = w.holes_buffers = w.holes_ring = 0;
	++w.forget;
	LocalVulkanRecording::BeginApart();
	w.recording = true;
	return true;
}

// The worker's open strand closed (its buffer kept with it when it holds a made draw) and a new apart buffer begun for
// the next, which starts with nothing bound and in no render pass.
void RenderExecutor::WeaveBreak() {
	auto& w = *t_weave_worker;
	++w.forget;
	w.view->m_rendering    = false;
	w.view->m_render_state = {};
	// (No made draw in it and nothing recorded: the buffer serves the next strand as it is.)
	if (w.recording && w.pending && LocalVulkanRecording::ApartCalls() == 0) return;
	if (w.recording) {
		auto buffer = LocalVulkanRecording::EndApart(!w.pending);
		if (!w.pending) w.result->strands.back().buffer = std::move(buffer);
	}
	LocalVulkanRecording::BeginApart();
	w.recording = true;
	w.pending   = true;
}

// A stretch from draw packet `packet` on: the needs noted from here are its (the last stretch's end where this begins).
void RenderExecutor::WeaveNewStretch(const uint32_t* packet) {
	auto&      w     = *t_weave_worker;
	auto&      r     = *w.result;
	const auto needs = static_cast<uint32_t>(r.needs.size());
	if (!r.stretches.empty()) r.stretches.back().needs.end = needs;
	r.stretches.push_back({.first_packet = packet, .needs = {needs, needs}});
	++w.forget;
}

void RenderExecutor::WeaveStretch(const uint32_t* packet) {
	WeaveNewStretch(packet);
	WeaveBreak();
}

void RenderExecutor::WeaveHole(Weave::Miss miss) {
	auto& w = *t_weave_worker;
	++w.holes;
	if (miss == Weave::Miss::Buffers) ++w.holes_buffers;
	if (miss == Weave::Miss::Ring) ++w.holes_ring;
	w.hole    = true;
	w.started = false;
	WeaveBreak();
}

bool RenderExecutor::WeaveDraw(const uint32_t* packet, const uint32_t* packet_end, bool clean,
                               std::span<const uint64_t> commands, uint64_t index_base, uint64_t index_bytes,
                               vk::IndexType index_type, const NativeXprDirectDraw* direct) {
	auto& w   = *t_weave_worker;
	auto& r   = *w.result;
	w.started = false;
	if (r.stretches.empty()) WeaveNewStretch(packet);
	// (As NativeXprTry's callers: a draw of GPU-read arguments names them, a direct draw names none, a run is at most
	// 64 draws, an indexed draw has indices.)
	const bool indexed = direct == nullptr || direct->indexed;
	if (kyty_local_table_xpr_mode.load(std::memory_order_relaxed) == 0 || w.view->IsInvalid() ||
	    (direct == nullptr || direct->indirect ? commands.empty() : !commands.empty()) || commands.size() > 64 ||
	    (indexed && (index_base == 0 || index_bytes == 0))) {
		WeaveHole(Weave::Miss::Packet);
		return false;
	}
	// (No hold of the GPU thread's from here to the draw's end: what it reads of the shared caches holds still.)
	const Weave::DrawScope scope(w.index);
	// The epochs as the draw begins, before any of its reads: what it read is proven from them.
	const uint64_t mapping      = m_context.GetGpuResources().MappingEpoch();
	const uint64_t registration = m_context.GetBufferCache().RegistrationEpoch();
	const uint64_t resolution   = m_context.GetTextureCache().ResolutionEpoch();
	const auto     size_of      = [](const auto& vector) { return static_cast<uint32_t>(vector.size()); };
	const uint32_t held_uses = size_of(r.held_uses), held_buffers = size_of(r.held_buffers),
	               ring_uses = size_of(r.ring_uses), held_words = size_of(r.held_words), touched = size_of(r.touched),
	               placed = size_of(r.placed), proofs = size_of(r.proofs), needs = size_of(r.needs);
	w.draw_packet = packet;
	w.miss        = Weave::Miss::Other;
	w.pass_opened = false;
	{
		auto& made          = r.made.emplace_back();
		made.packet         = packet;
		made.packet_end     = packet_end;
		made.commands_begin = static_cast<uint32_t>(LocalVulkanRecording::ApartCalls());
		made.mapping        = mapping;
		made.registration   = registration;
		made.resolution     = resolution;
	}
	if (WeaveTableDraw(*w.view, commands, index_base, index_bytes, index_type, direct, clean) != TableResult::Drawn) {
		// (Its notes go: no turn reads them. What it read stays in the pages: a later draw may take what it cached.)
		r.made.pop_back();
		r.held_uses.resize(held_uses);
		r.held_buffers.resize(held_buffers);
		r.ring_uses.resize(ring_uses);
		r.held_words.resize(held_words);
		r.touched.resize(touched);
		r.placed.resize(placed);
		r.proofs.resize(proofs);
		r.needs.resize(needs);
		WeaveHole(w.miss);
		return false;
	}
	const auto index  = static_cast<uint32_t>(r.made.size() - 1);
	auto&      made   = r.made.back();
	made.commands_end = static_cast<uint32_t>(LocalVulkanRecording::ApartCalls());
	made.pass_begin   = w.pass_opened;
	made.held_uses    = {held_uses, size_of(r.held_uses)};
	made.held_buffers = {held_buffers, size_of(r.held_buffers)};
	made.ring_uses    = {ring_uses, size_of(r.ring_uses)};
	made.held_words   = {held_words, size_of(r.held_words)};
	made.touched      = {touched, size_of(r.touched)};
	made.placed       = {placed, size_of(r.placed)};
	made.proofs       = {proofs, size_of(r.proofs)};
	if (w.pending) {
		auto& strand      = r.strands.emplace_back();
		strand.first_made = index;
		strand.stretch    = static_cast<uint32_t>(r.stretches.size() - 1);
		strand.log_first  = index;
		static_cast<WeaveTurn::Begun*>(w.begun)->after_hole.push_back(w.hole ? 1 : 0);
		w.pending = w.hole = false;
		w.started          = true;
	}
	made.strand = static_cast<uint32_t>(r.strands.size() - 1);
	++r.strands.back().count;
	return true;
}

Weave::Registers* RenderExecutor::WeaveStrandStart() noexcept {
	auto& w = *t_weave_worker;
	return w.started ? &w.result->strands.back().start : nullptr;
}

void RenderExecutor::WeaveSpoil() noexcept {
	t_weave_worker->result->spoiled = true;
}

uint32_t RenderExecutor::WeaveMadeCount() const noexcept {
	return static_cast<uint32_t>(t_weave_worker->result->made.size());
}

void RenderExecutor::WeaveParcelEnd() {
	auto& w = *t_weave_worker;
	auto& r = *w.result;
	if (!r.stretches.empty()) r.stretches.back().needs.end = static_cast<uint32_t>(r.needs.size());
	if (w.recording) {
		auto buffer = LocalVulkanRecording::EndApart(!w.pending);
		if (!w.pending) r.strands.back().buffer = std::move(buffer);
		w.recording = false;
	}
	if (auto& begun = *static_cast<WeaveTurn::Begun*>(w.begun); begun.slab >= 0) {
		w.ring.Keep(begun.slab, w.slab_used);
		r.slab.used = w.slab_used;
	}
	WeaveLists(r);
	if (w.holes != 0) LiveCounters::Add(LiveCounters::WeaveHoles, w.holes);
	if (w.holes_buffers != 0) LiveCounters::Add(LiveCounters::WeaveHolesBuffers, w.holes_buffers);
	if (w.holes_ring != 0) LiveCounters::Add(LiveCounters::WeaveHolesRing, w.holes_ring);
	w.parcel = nullptr;
	w.result = nullptr;
	w.begun  = nullptr;
}

// --- What the table path's worker instantiation asks of its worker ---

Weave::Parcel& RenderExecutor::WeaveParcel() noexcept {
	return *t_weave_worker->parcel;
}

void RenderExecutor::WeaveMiss(Weave::Miss miss) noexcept {
	t_weave_worker->miss = miss;
}

std::pair<uint8_t*, vk::DeviceAddress> RenderExecutor::WeaveSlabTake(uint64_t size, uint64_t alignment) noexcept {
	auto& w = *t_weave_worker;
	if (w.slab_mapped == nullptr || size == 0) return {nullptr, 0};
	const uint64_t align = std::max<uint64_t>(alignment, 1);
	const uint64_t at    = (w.slab_used + align - 1) / align * align;
	if (at > Weave::SlabBytes || size > Weave::SlabBytes - at) return {nullptr, 0};
	w.slab_used = at + size;
	return {w.slab_mapped + at, w.slab_device + at};
}

uint64_t RenderExecutor::WeaveParcelSerial() noexcept {
	return t_weave_worker->parcel->serial;
}

uint64_t RenderExecutor::WeaveForgetCount() noexcept {
	return t_weave_worker->forget;
}

// First in the worker's TableDraw, before its evaluation, so that what the draw notes goes with its pass. A pass other
// than the last draw's begins a stretch at this draw: its needs are the GPU thread's to do before that pass. Inside an
// open strand the strand goes on into the new pass: its buffer ends the one and begins the other, so that a run of
// passes between two holes is spliced at once (the turn ends a part before a later pass it must prepare, and the next
// part begins there, after these calls). A strand's first pass is noted only: the turn begins it. (Inside a strand a
// stretch begins exactly where its buffer changes pass, as BeginRendering compares passes: the turn finds a later
// pass's needs only there.)
void RenderExecutor::WeavePass(CommandBuffer& view, const RenderState& rendering) {
	auto&      w     = *t_weave_worker;
	const bool other = w.pass_known && !(w.pass == rendering);
	w.pass           = rendering;
	w.pass_known     = true;
	if (w.pending) {
		if (other) WeaveNewStretch(w.draw_packet);
		view.AdoptRendering(rendering);
		return;
	}
	if (view.RenderingWith(rendering)) return;
	WeaveNewStretch(w.draw_packet);
	view.PassChange(rendering);
	view.AdoptRendering(rendering);
	w.pass_opened                        = true;
	w.result->made.back().commands_begin = static_cast<uint32_t>(LocalVulkanRecording::ApartCalls());
	++w.forget;
}

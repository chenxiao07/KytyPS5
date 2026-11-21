// --- The weave's turn (weave.h, docs/WEAVE.md): what the GPU thread does with a worker's parcel -------------------
//
// Included by renderDraw.cpp after weave-worker.inc (local Vulkan recording builds only). The GPU thread walks queue
// 0's stream in order as today; at a parcel's first draw packet the command processor (weave-order.inc) opens it here
// (WeaveOpen: nothing the parcel read moved since its worker began), and at each of its draw packets asks for a turn
// (WeaveTurnAt). At a strand's first draw the turn does the needs of every stretch begun by then with today's
// functions, tests the strand once from its lists (the prefix before its first failing draw may be spliced), does its
// actions (constants the guest wrote copied again, CPU-written indices uploaded, CPU-written sampled images refreshed,
// the depth clear its first pass makes, the layouts), begins the pass and splices a part of the strand's apart buffer
// into the GPU thread's own stream, stamps what the part used and forgets what the GPU thread had bound. The command
// processor then skips the part's packets (WeaveSkip: by the register log). Every other draw packet of the parcel the
// GPU thread makes itself, today's way, into the same stream between the spliced parts.
//
// Everything here runs on the GPU thread: the shared structures it changes are its own to change, and the parcel's
// result is read only after its worker is done with it (its slot Done, read with acquire).

// The pipeline handle a worker's bind recorded, as Weave::Made keeps it.
static uint64_t WeavePipelineBits(vk::Pipeline handle) {
	return std::bit_cast<uint64_t>(static_cast<VkPipeline>(handle));
}

// The buffer a worker's lookup owes a last use to, as Weave::Touch keeps it (BufferCache::TouchLiveBuffer's terms).
static BufferId WeaveBufferId(uint64_t bits) {
	return std::bit_cast<BufferId>(bits);
}
static vk::Buffer WeaveBufferHandle(uint64_t bits) {
	return vk::Buffer(std::bit_cast<VkBuffer>(bits));
}

bool RenderExecutor::WeaveStart(uint32_t workers, const void* processor,
                                bool (*same)(const void* processor, const Weave::Registers& start,
                                             const Weave::RegisterLog& log, uint32_t first, uint32_t index)) {
	workers = std::min(workers, Weave::MaxWorkers);
	if (workers == 0 || same == nullptr) return false;
	(void)OwnTable();
	(void)NativeXprState();
	auto& scheduler = m_context.GetCommandScheduler();
	auto& buffer    = scheduler.Current();
	if (buffer.IsInvalid()) return false;
	if (!g_weave_turn) {
		g_weave_turn           = new WeaveTurn();
		g_weave_turn->executor = this;
	}
	auto& t     = *g_weave_turn;
	t.processor = processor;
	t.same      = same;
	for (uint32_t index = 0; index < workers; ++index) {
		auto& w = t.workers[index];
		w.index = index;
		// Its view of the command buffer, with registers of its own bound at each parcel; it records through the apart
		// marker only, into buffers of its own, and compares its handle only with its own emission's.
		if (!w.view)
			w.view.reset(new CommandBuffer(buffer, buffer.GetRegisters(), buffer.GetUserConfig(), buffer.GetShaders()));
		w.view->Aim(LocalVulkanRecording::ApartMarker());
		// Its ring (made here: making a buffer is a direct call), used only when the host writes it coherently.
		if (w.ring.mapped != nullptr) continue;
		w.ring.buffer =
		    std::make_unique<StreamBuffer>(m_context.GetGraphics(), scheduler, MemoryUsage::Upload, Weave::RingBytes,
		                                   AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress);
		const auto mapped = w.ring.buffer->Mapped();
		if (mapped.size() < Weave::RingBytes || !w.ring.buffer->IsCoherent() ||
		    w.ring.buffer->BufferDeviceAddress() == 0) {
			std::printf("Weave: worker %u's upload ring is not host-coherent; the weave stays off.\n", index + 1);
			std::fflush(stdout);
			w.ring.buffer.reset();
			return false;
		}
		w.ring.mapped  = mapped.data();
		w.ring.address = w.ring.buffer->BufferDeviceAddress();
	}
	t.worker_count = workers;
	return true;
}

// The weave's executor state goes with the executor WeaveStart made it for, as RenderContext is destroyed at shutdown:
// the GPU thread is joined by then, and it joined the weave's threads (StopAll).
RenderExecutor::~RenderExecutor() {
	if (g_weave_turn != nullptr && g_weave_turn->executor == this) [[unlikely]] {
		delete g_weave_turn;
		g_weave_turn = nullptr;
	}
}

// Whether an image began since start epoch `since` inside a range one of the parcel's draws bound (its slab copies and
// held V#s). The GPU thread's own draw makes a formatted V# an image starts at from the image
// (BufferCache::SynchronizeBufferFromImage), which the worker's didn't. The texture cache's log is read once for each
// move of its epoch and kept; a parcel from before what it holds is refused as moved.
bool RenderExecutor::WeaveStartsHit(const Weave::Parcel& parcel) {
	auto&       s        = g_weave_turn->starts;
	const auto& r        = *parcel.result;
	auto&       textures = m_context.GetTextureCache();
	const auto  since    = r.image_start_epoch;
	if (const auto epoch = textures.StartEpoch(); s.seen != epoch) {
		const auto before = s.recent.size();
		if (s.seen != 0 && textures.NewImageStartsSince(s.seen, s.recent)) {
			if (s.recent.size() > before) s.seen = std::max(s.seen, s.recent.back().first);
		} else {
			s.recent.clear();
			s.from = s.seen = epoch;
		}
		if (s.recent.size() > 4096) {
			const auto drop = s.recent.size() - 2048;
			s.from          = s.recent[drop - 1].first;
			s.recent.erase(s.recent.begin(), s.recent.begin() + static_cast<std::ptrdiff_t>(drop));
		}
	}
	if (since < s.from) return true;
	s.addresses.clear();
	for (auto it = std::ranges::upper_bound(s.recent, since, {}, &std::pair<uint64_t, uint64_t>::first);
	     it != s.recent.end(); ++it)
		s.addresses.push_back(it->second);
	if (s.addresses.empty()) return false;
	std::ranges::sort(s.addresses);
	const auto inside = [&](uint64_t address, uint64_t size) {
		const auto at = std::ranges::lower_bound(s.addresses, address);
		return at != s.addresses.end() && *at - address < std::max<uint64_t>(size, 1);
	};
	for (const auto& use: r.ring_uses)
		if (inside(use.address, use.size)) return true;
	for (const auto& use: r.held_uses)
		if (inside(use.address, use.size)) return true;
	return false;
}

bool RenderExecutor::WeaveOpen(Weave::Parcel& parcel) {
	if (!g_weave_turn || !parcel.result) return false;
	auto&       t     = *g_weave_turn;
	auto&       r     = *parcel.result;
	const auto& begun = t.begun[parcel.serial % Weave::ParcelSlots];

	const auto refuse = [](LiveCounters::Id id) {
		LiveCounters::Add(id);
		return false;
	};
	// Nothing of it to open: its walk spoiled, no draw made, no slab, or a pipeline evicted since it began (a handle
	// its binds recorded may be gone; one evicted later is the strand test's).
	if (r.spoiled || r.lost || r.made.empty() || r.slab.ring < 0 || begun.slab < 0 ||
	    PipelineEvictionCount() != r.evictions)
		return false;
	auto& resources = m_context.GetGpuResources();
	auto& textures  = m_context.GetTextureCache();
	// Each test asks only where its epoch moved since the worker began, and only over what the parcel read: the mapping
	// over its pages; shader registrations at its draws' programs; image starts inside the ranges it bound; the
	// device-address table; the free log over its objects (turned over since: no object of it can be proven alive);
	// the scout's write ring over its pages (a host write, a write-back or a GPU-written mark since).
	if (resources.MappingEpoch() != begun.mapping && resources.MappingMovedSince(begun.mapping, r.pages))
		return refuse(LiveCounters::WeaveParcelsRefusedMapping);
	if (ShaderMapGeneration() != r.shader_generation) {
		thread_local std::vector<uint64_t> programs;
		programs.clear();
		for (const auto& made: r.made)
			for (const auto address: {made.vs, made.ps})
				if (address != 0) programs.push_back(address);
		std::ranges::sort(programs);
		programs.erase(std::unique(programs.begin(), programs.end()), programs.end());
		if (ShaderRegisteredAtSince(r.shader_generation, programs))
			return refuse(LiveCounters::WeaveParcelsRefusedShaders);
	}
	if (textures.StartEpoch() != r.image_start_epoch && WeaveStartsHit(parcel))
		return refuse(LiveCounters::WeaveParcelsRefusedImageStart);
	if (BufferCache::BdaTableGeneration() != r.bda_generation) return refuse(LiveCounters::WeaveParcelsRefusedBda);
	// (The free log read from the parcel's begin clock here, on in each later test: each entry once.)
	auto& o         = t.open;
	o.freed_scanned = r.freed_clock;
	o.freed_hits.clear();
	o.freed_lost = Weave::FreedHits(o.freed_scanned, r.objects, o.freed_hits) == 2;
	if (o.freed_lost) return refuse(LiveCounters::WeaveParcelsRefusedFreed);
	if (Scout::RangesWrittenSince(r.write_clock, r.pages) != 0) return refuse(LiveCounters::WeaveParcelsRefusedRing);
	o.parcel          = &parcel;
	o.result          = &r;
	o.begun           = &begun;
	o.cursor          = 0;
	o.stretch_cursor  = 0;
	o.needs_packet    = nullptr;
	o.clear           = {};
	o.skip            = false;
	o.own             = 0;
	o.spliced         = 0;
	o.strands_spliced = 0;
	o.strands_cut     = 0;
	o.strands.assign(r.strands.size(), {});
	o.refused.assign(r.stretches.size(), UINT32_MAX);
	o.clear_later.clear();
	o.recopies.clear();
	o.uploads.clear();
	o.refreshes.clear();
	// The buffers its draws took, each touched once here (their last use, which the worker's lookups owe).
	auto& buffers = m_context.GetBufferCache();
	for (const auto& touch: r.touched_unique)
		(void)buffers.TouchLiveBuffer(WeaveBufferId(touch.buffer), WeaveBufferHandle(touch.handle), touch.address,
		                              touch.size);
	LiveCounters::Add(LiveCounters::WeaveParcelsOpened);
	return true;
}

// The needs of each stretch of the open parcel begun at or before `packet` (draw packets ascend), done with today's
// functions in TableDraw's order: the sets' validations, the draw states' targets, the sets' transits. A need whose set
// or draw state the free log shows freed since the parcel began, or whose state was captured again, is refused, and so
// is one that fails; a refusal cuts the stretch's draws from the need's first. A target need refused only for a depth
// target's pending clear, at a strand's first draw, is deferred to that strand's splice, which makes the clear with its
// pass as the normal path makes it at the draw; the same strand's later target needs on that clear are done after it.
void RenderExecutor::WeaveNeedsAt(const uint32_t* packet) {
	auto& o = g_weave_turn->open;
	auto& r = *o.result;
	if (o.needs_packet != packet) {
		o.needs_packet = packet;
		o.needs_states.clear();
		o.needs_sets.clear();
	}
	auto&                          scheduler = m_context.GetCommandScheduler();
	auto&                          textures  = m_context.GetTextureCache();
	const bool                     raw_on    = kyty_local_draw_read_after_write.load(std::memory_order_relaxed) != 0;
	std::optional<RenderLockGuard> lock;
	while (o.stretch_cursor < r.stretches.size() && r.stretches[o.stretch_cursor].first_packet <= packet) {
		const auto  index   = o.stretch_cursor++;
		const auto& stretch = r.stretches[index];
		if (stretch.needs.empty()) continue;
		if (!lock) {
			scheduler.PopPendingOperations(); // (as before the GPU thread's own draw, NativeXprTry)
			lock.emplace(m_context.GetMutex());
		}
		auto&      buffer    = scheduler.Current();
		const bool invalid   = buffer.IsInvalid();
		const auto vk_buffer = invalid ? vk::CommandBuffer {} : buffer.Handle();
		o.FreedRead();

		const auto refuse = [&](const Weave::Need& need) {
			o.refused[index] = std::min(o.refused[index], need.first_made);
		};
		// (The depth target of `state` with a clear pending: its metadata and slice.)
		const auto pending_depth = [&](const NativeXprDrawState& state, uint64_t& metadata, uint32_t& slice) {
			for (const auto& target: state.targets)
				if (target.metadata != 0 && target.requested.metadata.kind == ImageMetadataKind::Htile &&
				    textures.IsMetaCleared(target.metadata, target.slice)) {
					metadata = target.metadata;
					slice    = target.slice;
					return true;
				}
			return false;
		};
		for (const auto kind: {Weave::NeedKind::Validate, Weave::NeedKind::Targets, Weave::NeedKind::Transit})
			for (auto i = stretch.needs.begin; i < stretch.needs.end; ++i) {
				const auto& need = r.needs[i];
				if (need.kind != kind) continue;
				if (invalid || need.first_made >= r.made.size()) {
					refuse(need);
					continue;
				}
				if (o.Freed(need.set) || o.Freed(need.state)) {
					refuse(need);
					continue;
				}
				const auto* state = static_cast<const NativeXprDrawState*>(need.state);
				auto*       set   = const_cast<TableImageSet*>(static_cast<const TableImageSet*>(need.set));
				if (state != nullptr && state->capture_id != need.capture) {
					refuse(need);
					continue;
				}
				bool done = false;
				switch (kind) {
					case Weave::NeedKind::Validate: done = set != nullptr && TableValidateSet(*set); break;
					case Weave::NeedKind::Targets:
						done = state != nullptr && NativeXprEmitTargets(*state, vk_buffer, set, &buffer);
						break;
					case Weave::NeedKind::Transit:
						// (TableDraw transits a set the draw reads and writes at every draw,
						// KYTY_DRAW_READ_AFTER_WRITE.)
						done = set != nullptr && !(set->read_after_write && raw_on) &&
						       TableTransit(*set, vk_buffer, state);
						break;
				}
				if (done) {
					if (kind == Weave::NeedKind::Targets) o.needs_states.push_back(state);
					if (kind == Weave::NeedKind::Transit) o.needs_sets.emplace_back(set, state);
					continue;
				}
				uint64_t metadata = 0;
				uint32_t slice    = 0;
				if (kind == Weave::NeedKind::Targets && state != nullptr && pending_depth(*state, metadata, slice)) {
					const auto strand = r.made[need.first_made].strand;
					if (r.strands[strand].first_made == need.first_made) {
						o.clear          = {.made = need.first_made, .state = need.state, .set = need.set};
						o.clear_metadata = metadata;
						o.clear_slice    = slice;
						o.clear_later.clear();
						continue;
					}
					if (o.clear.made != UINT32_MAX && o.clear_metadata == metadata && o.clear_slice == slice &&
					    need.first_made > o.clear.made && r.made[o.clear.made].strand == strand) {
						o.clear_later.push_back({.made = need.first_made, .state = need.state, .set = need.set});
						continue;
					}
				}
				refuse(need);
			}
	}
}

// What the normal path's target acquisition would clear at a draw of `state` now (AcquireRenderTargets, its rule for
// the depth attachment's load: htile_load_decision): false when no depth target (HTile) has a clear pending in its
// slice since the state's last emission. Else `clear`: the state's rendering with the depth attachment's clear flags
// and values as that function sets them, and whether the turn can't make it so (`refused`: a colour target's clear
// pending too; the registers' own depth or stencil clear, which that function makes another way; a stencil clear under
// a layout that keeps stencil read-only, where it chooses a writable one). Read only.
bool RenderExecutor::WeaveClearPlan(const NativeXprDrawState& state, WeaveClear& clear) {
	auto& textures = m_context.GetTextureCache();
	if (state.meta_epoch == textures.MetaEpoch()) return false;
	const NativeXprDrawState::Target* depth  = nullptr;
	bool                              colour = false, known = false;
	uint32_t                          fill = 0;
	for (const auto& target: state.targets) {
		if (target.metadata == 0) continue;
		if (target.requested.metadata.kind == ImageMetadataKind::Htile && depth == nullptr) {
			if (textures.IsMetaCleared(target.metadata, target.slice, &fill, &known)) depth = &target;
		} else if (textures.IsMetaCleared(target.metadata, target.slice, nullptr, nullptr,
		                                  target.requested.metadata.kind == ImageMetadataKind::Dcc)) {
			colour = true;
		}
	}
	if (depth == nullptr) return false;
	clear                 = {};
	clear.rendering       = state.rendering;
	clear.metadata        = depth->metadata;
	clear.slice           = depth->slice;
	const auto& registers = m_context.GetCommandScheduler().Current().GetRegisters();
	const auto& control   = registers.GetRenderControl();
	if (colour || control.depth_clear_enable || control.stencil_clear_enable) return clear.refused = true, true;
	const auto decision   = htile_load_decision(true, fill, known, depth->requested.metadata.stencil_compressed);
	auto&      attachment = clear.rendering.depth_stencil_attachment;
	attachment.clear_value[0] =
	    std::bit_cast<uint32_t>(decision.depth_uniform ? decision.uniform_value : registers.GetDepthClearValue());
	attachment.clear_value[1] = registers.GetStencilClearValue();
	attachment.depth_clear    = decision.depth_meta_clear;
	attachment.stencil_clear  = decision.stencil_meta_clear;
	if (attachment.stencil_clear && attachment.has_stencil &&
	    (attachment.image_layout == vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal ||
	     attachment.image_layout == vk::ImageLayout::eDepthStencilReadOnlyOptimal ||
	     attachment.image_layout == vk::ImageLayout::eStencilReadOnlyOptimal))
		clear.refused = true;
	return true;
}

// Whether the GPU thread's own draw of made draw `made` would read its held V# `use` (an index into Result::held_uses),
// which the guest wrote, from a new copy in the table upload ring now (TableBlocks' terms: 16 KiB or less, CPU-dirty
// and not GPU-written, aligned, and for a formatted V# no image starting at it). Read only.
bool RenderExecutor::WeaveRecopyable(uint32_t made, uint32_t use) {
	const auto& r     = *g_weave_turn->open.result;
	const auto& draw  = r.made[made];
	const auto& held  = r.held_uses[use];
	auto&       cache = m_context.GetBufferCache();
	if (held.size == 0 || held.size > BufferCache::CACHING_PAGESIZE || held.address % 4 != 0) return false;
	if (!cache.IsRegionCpuDirty(held.address, held.size) || cache.IsRegionGpuModified(held.address, held.size))
		return false;
	const auto* pair = static_cast<const TablePair*>(draw.pair);
	if (pair == nullptr || held.stage >= 2 || pair->programs[held.stage] == nullptr || draw.blocks[held.stage] == 0)
		return false;
	const auto& program = *pair->programs[held.stage];
	if (held.buffer >= program.info.buffers.size()) return false;
	return !program.info.buffers[held.buffer].formatted ||
	       !m_context.GetTextureCache().HasImageStartingAt(held.address);
}

// Held V# `use` of made draw `made` copied again as the GPU thread's own draw would copy it (TableRingCopy,
// TableBlocks' own copy), and the worker's block in its slab pointed at the copy: its address and size words, the
// others left. The block's five words there must first be those the worker noted (Result::held_words); else, or with no
// copy, false. The push words the worker recorded hold the block's address, which doesn't move.
bool RenderExecutor::WeaveRecopy(uint32_t made, uint32_t use) {
	auto&       t    = *g_weave_turn;
	const auto& r    = *t.open.result;
	const auto& draw = r.made[made];
	const auto& held = r.held_uses[use];
	const auto* pair = static_cast<const TablePair*>(draw.pair);
	if (r.slab.ring < 0 || static_cast<uint32_t>(r.slab.ring) >= t.worker_count || pair == nullptr || held.stage >= 2 ||
	    pair->programs[held.stage] == nullptr)
		return false;
	const auto& ring   = t.workers[static_cast<uint32_t>(r.slab.ring)].ring;
	const auto  device = draw.blocks[held.stage];
	if (ring.mapped == nullptr || device < ring.address || size_t {held.word} + 5 > r.held_words.size()) return false;
	const auto&    plan = pair->programs[held.stage]->table_plan;
	const uint64_t at   = device - ring.address + uint64_t {plan.BufferDword(held.buffer)} * sizeof(uint32_t);
	if (at % sizeof(uint32_t) != 0 || at + 5 * sizeof(uint32_t) > Weave::RingBytes) return false;
	auto* const words = reinterpret_cast<uint32_t*>(ring.mapped + at);
	if (!std::equal(words, words + 5, r.held_words.begin() + held.word)) return false;
	const auto copy = TableRingCopy(held.address, held.size);
	if (copy == 0) return false;
	words[0] = static_cast<uint32_t>(copy);
	words[1] = static_cast<uint32_t>(copy >> 32u);
	words[2] = static_cast<uint32_t>(held.size / 4);
	return true;
}

// At a strand's first visit (WeaveTurnAt): the strand tested once from its lists, now. Its made draws before the first
// that fails are its prefix, which may be spliced; from there on its draws are the GPU thread's own. In order: the
// refused need of the stretch it begins in; a first draw that binds no pipeline of its own (the strand's buffer starts
// with nothing bound); the free log over its items (before anything of theirs is read); the pipeline handles its binds
// recorded; then, only where the epoch moved since its draws began, buffer registrations over the ranges it took from
// buffers, mappings over its ranges and its images, and image resolutions; the images the CPU wrote since the parcel
// began (a set's sampled ones refreshed at the turn where they can be, as FindTexture would; a target's cut); and the
// held ranges the guest wrote, asked only from the first draw that holds one of them, each copied again or uploaded
// where the GPU thread's own draw would, else the prefix ends at the draw. Each failure cuts at the first draw that
// holds what failed (an item's or a range's `first`). Last, the depth clear its first draw's pass would make. The
// actions listed are the turn's for the draws before the prefix's end.
void RenderExecutor::WeaveTest(uint32_t s) {
	auto&          o      = g_weave_turn->open;
	auto&          r      = *o.result;
	const auto&    strand = r.strands[s];
	auto&          st     = o.strands[s];
	const uint32_t first = strand.first_made, last = strand.first_made + strand.count;
	uint32_t       end = last;
	st.tested          = true;
	o.recopies.clear();
	o.uploads.clear();
	o.refreshes.clear();
	const auto cut   = [&](uint32_t at) { end = std::min(end, at); };
	const auto held  = [&](uint32_t relative) { return first + relative < end; };
	const auto items = std::span(r.lists.items).subspan(strand.items.begin, strand.items.size());
	const auto lists = std::span(r.lists.ranges).subspan(strand.ranges.begin, strand.ranges.size());
	if (strand.stretch < o.refused.size()) cut(o.refused[strand.stretch]);
	if (!strand.buffer || r.made[first].pipeline_handle == 0) cut(first);
	// The free log, before anything of an item is read: its entries since the turn last read it, over the parcel's
	// objects; an item with an object freed since the parcel began cuts at its first draw.
	if (end > first) {
		o.FreedRead();
		if (o.freed_lost)
			cut(first);
		else if (!o.freed_hits.empty())
			for (const auto& item: items)
				for (const void* object: {item.pair, item.set, item.state})
					if (o.Freed(object)) cut(first + item.first);
	}
	// The pipeline handles its binds recorded (a pipeline evicted and made again since has another).
	for (const auto& item: items)
		if (item.pipeline != nullptr && held(item.first) &&
		    WeavePipelineBits(static_cast<const PipelineCache::Pipeline*>(item.pipeline)->pipeline) !=
		        r.made[first + item.first].pipeline_handle)
			cut(first + item.first);
	// KYTY_FRAMEGEN: while the frame has no camera, its first G-buffer draw goes the GPU thread's normal way, whose
	// hook reads the camera (TableTry's rule; the strands are tested in order, and no flip is inside a parcel).
	if (FrameGen::Enabled() && FrameGen::WantsCamera())
		for (auto i = first; i < end; ++i)
			if (r.made[i].gbuffer) {
				cut(i);
				break;
			}
	auto& cache     = m_context.GetBufferCache();
	auto& resources = m_context.GetGpuResources();
	auto& textures  = m_context.GetTextureCache();
	// (A set's images, a draw state's targets, each with its id, serial and memory.)
	const auto any_image = [&](const Weave::ListItem& item, auto&& test) {
		if (const auto* set = static_cast<const TableImageSet*>(item.set); set != nullptr)
			for (const auto& dep: set->images)
				if (test(dep.id, dep.serial, dep.desc.info.data.address, dep.desc.info.data.size)) return true;
		if (const auto* state = static_cast<const NativeXprDrawState*>(item.state); state != nullptr)
			for (const auto& target: state->targets)
				if (test(target.id, target.serial, target.requested.data.address,
				         std::max<uint64_t>(target.requested.data.size, 1)))
					return true;
		return false;
	};
	// (The gates are the strand's least epochs; past one, each range and item is asked from the epoch of the first
	// draw that holds it, as that draw read it.)
	if (cache.RegistrationEpoch() != strand.registration)
		for (const auto& range: lists) {
			if (range.kind == Weave::RangeKind::Ring || range.end <= range.begin || !held(range.first)) continue;
			const auto since = r.made[first + range.first].registration;
			for (auto region = range.begin / TRACKER_REGION_SIZE; region <= (range.end - 1) / TRACKER_REGION_SIZE;
			     ++region)
				if (cache.RegionRegistrationEpoch(region * TRACKER_REGION_SIZE) > since) {
					if (cache.RegisteredOverSince(since, range.begin, range.end - range.begin) != 0)
						cut(first + range.first);
					break;
				}
		}
	if (const auto mapping = resources.MappingEpoch(); mapping != strand.mapping) {
		// (An unmap by the log of the latest unmaps, the granule walk only past its reach; a map by the map log.)
		const auto moved = [&](uint64_t since, uint64_t address, uint64_t size) {
			if (size == 0 || since == mapping) return false;
			const int unmapped = resources.UnmappedSinceLogged(since, address, size);
			if (unmapped == 1 || (unmapped == 2 && resources.UnmappedSince(since, address, size))) return true;
			const std::pair<uint64_t, uint64_t> one {address, address + size};
			return resources.MappedSinceLogged(since, std::span(&one, 1));
		};
		for (const auto& range: lists)
			if (held(range.first) && moved(r.made[first + range.first].mapping, range.begin, range.end - range.begin))
				cut(first + range.first);
		for (const auto& item: items) {
			if (!held(item.first)) continue;
			const auto since = r.made[first + item.first].mapping;
			if (any_image(item, [&](const auto&, uint64_t, uint64_t address, uint64_t size) {
				    return moved(since, address, size);
			    }))
				cut(first + item.first);
		}
	}
	if (textures.ResolutionEpoch() != strand.resolution)
		for (const auto& item: items) {
			if (!held(item.first)) continue;
			const auto since = r.made[first + item.first].resolution;
			if (any_image(item, [&](const auto& id, uint64_t serial, uint64_t address, uint64_t size) {
				    const auto* image = textures.m_slot_images.try_get(id);
				    return image == nullptr || image->serial != serial ||
				           textures.RegistrationsSince(address, size, since);
			    }))
				cut(first + item.first);
		}
	if (Image::CpuDirtyEpoch() != o.begun->cpu_dirty)
		for (const auto& item: items) {
			if (!held(item.first)) continue;
			const auto at = first + item.first;
			if (const auto* state = static_cast<const NativeXprDrawState*>(item.state); state != nullptr)
				for (const auto& target: state->targets)
					if (const auto* image = textures.TryGetImage(target.id);
					    image != nullptr && image->serial == target.serial && image->IsCpuDirty())
						cut(at);
			const auto* set = static_cast<const TableImageSet*>(item.set);
			if (set == nullptr) continue;
			bool written = false, fit = true;
			for (const auto& dep: set->images) {
				if (dep.target) continue;
				const auto* image = textures.TryGetImage(dep.id);
				if (image == nullptr || !(image->IsMaybeCpuDirty() || image->IsDefinitelyCpuDirty())) continue;
				written = true;
				// (Refreshed as the normal path's FindTexture would, unless it can't be, or an earlier draw of the
				// strand renders into it.)
				fit = fit && image->registered && !image->depth_id && !image->binding.needs_rebind &&
				      image->serial == dep.serial;
				for (auto earlier = first; fit && earlier < at; ++earlier)
					if (const auto* state = static_cast<const NativeXprDrawState*>(r.made[earlier].state);
					    state != nullptr)
						for (const auto& target: state->targets)
							if (target.id == dep.id) fit = false;
			}
			if (!written) continue;
			if (!fit) {
				cut(at);
				continue;
			}
			if (std::ranges::none_of(o.refreshes, [&](const auto& refresh) { return refresh.set == item.set; }))
				o.refreshes.push_back({.made = at, .set = item.set, .state = item.state});
		}
	// The held ranges the guest wrote since: noted here, then asked draw by draw from the first that holds one.
	const auto range_dirty = [&](uint64_t address, uint64_t size) {
		for (auto region = address / TRACKER_REGION_SIZE; region <= (address + size - 1) / TRACKER_REGION_SIZE;
		     ++region)
			if (cache.DeferredProtectionPending(region * TRACKER_REGION_SIZE)) return true;
		return cache.IsRegionCpuModified(address, size);
	};
	thread_local std::vector<Weave::ListRange> failing;
	failing.clear();
	uint32_t failing_first = UINT32_MAX;
	for (const auto& range: lists) {
		if (range.kind == Weave::RangeKind::Ring || !held(range.first) || range.cpu_epoch == 0 ||
		    range.end <= range.begin)
			continue;
		const auto size = range.end - range.begin;
		if (cache.CpuModificationEpoch(range.begin, size) == range.cpu_epoch || !range_dirty(range.begin, size))
			continue;
		failing.push_back(range);
		failing_first = std::min(failing_first, first + range.first);
	}
	const auto is_failing = [&](Weave::RangeKind kind, uint64_t begin, uint64_t end_at, uint64_t epoch) {
		return std::ranges::any_of(failing, [&](const auto& range) {
			return range.kind == kind && range.begin == begin && range.end == end_at && range.cpu_epoch == epoch;
		});
	};
	for (auto i = failing.empty() ? end : failing_first; i < end; ++i) {
		const auto& draw = r.made[i];
		for (auto k = draw.held_uses.begin; k < draw.held_uses.end && i < end; ++k) {
			const auto& use = r.held_uses[k];
			if (!is_failing(Weave::RangeKind::Held, use.address, use.address + use.size, use.cpu_epoch)) continue;
			if (WeaveRecopyable(i, k))
				o.recopies.push_back({.made = i, .use = k});
			else
				cut(i);
		}
		for (auto k = draw.held_buffers.begin; k < draw.held_buffers.end && i < end; ++k) {
			const auto& use = r.held_buffers[k];
			if (use.end <= use.begin || !is_failing(Weave::RangeKind::HeldBuffer, use.begin, use.end, use.cpu_epoch))
				continue;
			// (The draw's index range inside the held range: uploaded at the turn, as the own draw's ObtainBuffer would
			// make it current, into the buffer it bound.)
			if (!use.args && draw.indexed && draw.index_bytes != 0 && draw.index_base >= use.begin &&
			    draw.index_base + draw.index_bytes <= use.end)
				o.uploads.push_back({.made = i, .address = draw.index_base, .size = draw.index_bytes});
			else
				cut(i);
		}
	}
	// The CPU-written indices the worker took as they are, uploaded first.
	for (auto i = first; i < end; ++i)
		if (const auto& draw = r.made[i];
		    draw.cpu_indices && draw.index_bytes != 0 &&
		    std::ranges::none_of(o.uploads, [&](const auto& upload) { return upload.made == i; }))
			o.uploads.push_back({.made = i, .address = draw.index_base, .size = draw.index_bytes});
	std::ranges::sort(o.uploads, {}, &WeaveTurn::Upload::made);
	std::ranges::sort(o.refreshes, {}, &WeaveTurn::Refresh::made);
	// The depth clear its first draw's pass would make, where the turn can't make it.
	if (end > first && r.made[first].state != nullptr) {
		WeaveClear clear;
		if (WeaveClearPlan(*static_cast<const NativeXprDrawState*>(r.made[first].state), clear) && clear.refused)
			cut(first);
	}
	st.prefix_end = end;
	if (end < last && !st.counted) {
		st.counted = true;
		++o.strands_cut;
	}
}

// The stamps the strand's draws [begin, end) owe, as the GPU thread's own draws would have written them: each pair's
// last use, variant hint and the worker's proof of its shaders (written only while the shader map is still at the
// proved generation); each set's and draw state's last use; each pipeline's (BindPipelineHandle's LRU); the last use of
// the targets found in place; the targets' proofs, each only when newer than the draw state's. A whole strand takes
// them from its lists, each object once; a part, draw by draw. Read after the strand's tests passed in this turn: every
// object is alive.
void RenderExecutor::WeaveStamps(uint32_t s, uint32_t begin, uint32_t end) {
	const auto& r          = *g_weave_turn->open.result;
	const auto& strand     = r.strands[s];
	auto&       textures   = m_context.GetTextureCache();
	const auto  frame      = m_context.FrameNumber();
	const auto  generation = ShaderMapGeneration();

	const auto pair_stamp = [&](const Weave::Made& made) {
		auto* pair         = const_cast<TablePair*>(static_cast<const TablePair*>(made.pair));
		pair->used_frame   = frame;
		pair->variant_hint = made.variant_index;
		if (made.pair_proved != 0 && pair->generation != made.pair_proved && generation == made.pair_proved)
			pair->generation = made.pair_proved;
	};
	const auto set_stamp = [&](const void* set) {
		const_cast<TableImageSet*>(static_cast<const TableImageSet*>(set))
		    ->used_frame.store(frame, std::memory_order_relaxed);
	};
	const auto placed_stamp = [&](const void* state) {
		for (const auto& target: static_cast<const NativeXprDrawState*>(state)->targets)
			if (const auto* image = textures.m_slot_images.try_get(target.id);
			    image != nullptr && image->serial == target.serial)
				(void)textures.GetImage(target.id);
	};
	if (begin == strand.first_made && end == strand.first_made + strand.count) {
		for (auto k = strand.stamps.begin; k < strand.stamps.end; ++k) {
			const auto& stamp = r.lists.stamps[k];
			if (stamp.pair != nullptr) pair_stamp(r.made[strand.first_made + stamp.first]);
			if (stamp.set != nullptr) set_stamp(stamp.set);
			if (stamp.state != nullptr) static_cast<const NativeXprDrawState*>(stamp.state)->Use(frame);
		}
		for (auto k = strand.pipelines.begin; k < strand.pipelines.end; ++k)
			(void)BindPipelineHandle(*static_cast<const PipelineCache::Pipeline*>(r.lists.pipelines[k]));
		for (auto k = strand.placed.begin; k < strand.placed.end; ++k)
			placed_stamp(r.lists.placed[k]);
	} else {
		for (auto i = begin; i < end; ++i) {
			const auto& made = r.made[i];
			if (made.pair != nullptr) pair_stamp(made);
			if (made.set != nullptr) set_stamp(made.set);
			if (made.state != nullptr) static_cast<const NativeXprDrawState*>(made.state)->Use(frame);
			if (made.pipeline_handle != 0 && made.pipeline != nullptr)
				(void)BindPipelineHandle(*static_cast<const PipelineCache::Pipeline*>(made.pipeline));
		}
		for (auto k = r.made[begin].placed.begin; k < r.made[end - 1].placed.end; ++k)
			placed_stamp(r.placed[k]);
	}
	for (auto k = r.made[begin].proofs.begin; k < r.made[end - 1].proofs.end; ++k) {
		const auto& proven = r.proofs[k];
		const auto& state  = *static_cast<const NativeXprDrawState*>(proven.state);
		if (state.targets_checked >= proven.proof.checked) continue;
		state.targets_span_begin = proven.proof.span_begin;
		state.targets_span_end   = proven.proof.span_end;
		state.targets_epoch      = proven.proof.epoch;
		state.targets_erased     = proven.proof.erased;
		state.targets_checked    = proven.proof.checked;
	}
}

bool RenderExecutor::WeaveTurnAt(const uint32_t* packet, const uint32_t*& until) {
	until = nullptr;
	if (!g_weave_turn || g_weave_turn->open.parcel == nullptr) return false;
	auto& t = *g_weave_turn;
	auto& o = t.open;
	auto& r = *o.result;
	o.skip  = false;
	WeaveNeedsAt(packet);
	const auto own = [&] {
		++o.own;
		return false;
	};
	while (o.cursor < r.made.size() && r.made[o.cursor].packet < packet)
		++o.cursor;
	if (o.cursor >= r.made.size() || r.made[o.cursor].packet != packet) return own();
	const uint32_t index  = o.cursor;
	const uint32_t s      = r.made[index].strand;
	auto&          strand = r.strands[s];
	auto&          st     = o.strands[s];
	if (!st.tested) WeaveTest(s);
	// A strand is spliced from its first draw; a part of it from a later pass where the last part ended; and a strand
	// first seen past its first draw (the GPU thread's own run of draws took its start) from a draw that binds its own
	// pipeline. Any other draw of it is the GPU thread's.
	const bool resume = st.resume_at != 0 && index == st.resume_at;
	const bool late =
	    index != strand.first_made && !resume && !st.seen && !st.late && r.made[index].pipeline_handle != 0;
	if (index != strand.first_made && !resume && !late) {
		if (!st.seen) st.late = true;
		return own();
	}
	if (resume)
		st.resume_at = 0;
	else
		st.seen = true;
	if (st.spliced || index >= st.prefix_end) return own();
	const uint32_t begin = index;
	uint32_t       end   = st.prefix_end;
	// (A resumed pass's stretch, whose needs were done at this packet: a need of it refused ends the part there.)
	if (resume && o.stretch_cursor != 0 && o.refused[o.stretch_cursor - 1] < end)
		end = std::max(begin, o.refused[o.stretch_cursor - 1]);
	if (end <= begin) return own();
	// The narrow walk test: a strand that follows a hole, or a part spliced late, is spliced only where the GPU
	// thread's registers are the worker's there, whole (a hole's handler, or the GPU thread's own draws, may have left
	// others): the strand's start image, or for a late part that image brought to its draw by the register log; else
	// the strand is cut here.
	const bool after_hole = begin == strand.first_made && s < o.begun->after_hole.size() && o.begun->after_hole[s] != 0;
	if ((late || after_hole) && !t.same(t.processor, strand.start, r.log, strand.first_made, begin)) {
		st.prefix_end = begin;
		if (!st.counted) {
			st.counted = true;
			++o.strands_cut;
		}
		return own();
	}
	if (!strand.buffer || r.made[end - 1].packet_end == nullptr || r.made[begin].state == nullptr) return own();
	auto&    scheduler = m_context.GetCommandScheduler();
	auto&    textures  = m_context.GetTextureCache();
	uint32_t resume_at = 0, passes = 1;
	bool     spliced = false;
	// (The depth clear deferred to this strand's splice, and the later target needs on it: taken now.)
	const auto                                    clear_first = std::exchange(o.clear, {});
	thread_local std::vector<WeaveTurn::Deferred> clear_later;
	clear_later.swap(o.clear_later);
	o.clear_later.clear();
	scheduler.PopPendingOperations(); // (as before the GPU thread's own draw, NativeXprTry)
	{
		RenderLockGuard lock(m_context.GetMutex());
		auto&           buffer = scheduler.Current();
		if (buffer.IsInvalid()) return own();
		const vk::CommandBuffer handle = buffer.Handle();
		// The part's held V#s the guest wrote, copied again as its draws would copy them, before anything else here:
		// one that can't be ends the part before its draw.
		for (const auto& recopy: o.recopies) {
			if (recopy.made < begin) continue;
			if (recopy.made >= end) break;
			if (!WeaveRecopy(recopy.made, recopy.use)) {
				end = recopy.made;
				break;
			}
		}
		// Its CPU-written index ranges, uploaded now into the buffers its draws bound, as ObtainBuffer would make them
		// current; one still CPU-modified after (not uploaded, or the guest's write since) ends the part before its
		// draw.
		auto& buffers = m_context.GetBufferCache();
		for (const auto& upload: o.uploads) {
			if (upload.made < begin) continue;
			if (upload.made >= end) break;
			buffers.SynchronizeAtTurn(upload.address, upload.size);
			if (buffers.IsRegionCpuModified(upload.address, upload.size)) {
				end = upload.made;
				break;
			}
		}
		// Its sets' sampled images the CPU wrote, refreshed with the normal path's FindTexture as TableValidateSet
		// refreshes one, then the set transited as TableDraw would before its draw; a set still not placed ends the
		// part.
		for (const auto& refresh: o.refreshes) {
			if (refresh.made < begin) continue;
			if (refresh.made >= end) break;
			const auto& set = *static_cast<const TableImageSet*>(refresh.set);
			bool        fit = true;
			for (const auto& dep: set.images) {
				if (dep.target) continue;
				auto* image = textures.m_slot_images.try_get(dep.id);
				if (image == nullptr || image->serial != dep.serial) {
					fit = false;
					break;
				}
				if (!TableRefreshPending(*image, dep.desc)) continue;
				if (!image->registered || image->depth_id || image->binding.needs_rebind) {
					fit = false;
					break;
				}
				(void)textures.FindTexture(dep.id, dep.desc);
			}
			if (!fit || scheduler.Current().RawHandle() != handle ||
			    !TableTransit(set, handle, static_cast<const NativeXprDrawState*>(refresh.state))) {
				end = refresh.made;
				break;
			}
		}
		// What is in place for this packet already: what its needs placed moments ago.
		thread_local std::vector<const void*>                         states;
		thread_local std::vector<std::pair<const void*, const void*>> sets;
		states.clear();
		sets.clear();
		if (o.needs_packet == packet) {
			states.insert(states.end(), o.needs_states.begin(), o.needs_states.end());
			sets.insert(sets.end(), o.needs_sets.begin(), o.needs_sets.end());
		}
		// The strand's first pass's depth clear, as the normal path's own draw here would make it: its targets put in
		// their layouts with the pending clear dropped first (NativeXprEmitTargets then finds none), its set transited,
		// and the pass begun clearing. A refusal on the way puts the clear back pending and leaves the strand to the
		// GPU thread. Then the same strand's later target needs on that clear, which now pass (one refused ends the
		// part before its draw). A deferred need whose clear is gone by now: its targets emitted as the need would.
		WeaveClear clear;
		bool       clearing = false;
		if (end > begin && begin == strand.first_made && !resume) {
			const auto& state = *static_cast<const NativeXprDrawState*>(r.made[begin].state);
			const auto* set   = static_cast<const TableImageSet*>(r.made[begin].set);
			if (WeaveClearPlan(state, clear)) {
				if (clear.refused || !textures.TouchMeta(clear.metadata, clear.slice, false)) {
					end = begin;
				} else if (!NativeXprEmitTargets(state, handle, set, &buffer) ||
				           (set != nullptr && !TableTransit(*set, handle, &state))) {
					(void)textures.TouchMeta(clear.metadata, clear.slice, true);
					end = begin;
				} else {
					clearing = true;
					states.push_back(&state);
					if (set != nullptr) sets.emplace_back(set, &state);
				}
			} else if (clear_first.made == begin &&
			           !NativeXprEmitTargets(*static_cast<const NativeXprDrawState*>(clear_first.state), handle,
			                                 static_cast<const TableImageSet*>(clear_first.set), &buffer)) {
				end = begin;
			}
			for (const auto& later: clear_later) {
				if (later.made < begin || later.made >= end) continue;
				if (NativeXprEmitTargets(*static_cast<const NativeXprDrawState*>(later.state), handle,
				                         static_cast<const TableImageSet*>(later.set), &buffer))
					states.push_back(later.state);
				else
					end = later.made;
			}
		}
		// The layouts the part's draws were made for. With the images epoch moved since the parcel began, each draw
		// state's targets and each set's images under its state are put in place by today's functions, in the part's
		// order, as TableDraw does before its draw; a refusal ends the part there. The part goes on into a later pass
		// of the strand only when the GPU thread has nothing to do before it: its stretch noted no needs and, with the
		// epoch moved, its targets and images are in place now (NativeXprTargetsPlaced, TableTransitPlaced). Else the
		// part ends before that pass, and the next part begins there.
		const bool moved      = textures.ImagesEpoch() != o.begun->images;
		bool       later      = false;
		uint32_t   pass_first = begin, stretch_at = o.stretch_cursor;
		for (auto i = begin; i < end; ++i) {
			const auto& made = r.made[i];
			if (i > begin && made.pass_begin) {
				later      = true;
				pass_first = i;
				while (stretch_at < r.stretches.size() && r.stretches[stretch_at].first_packet < made.packet)
					++stretch_at;
				if (stretch_at < r.stretches.size() && r.stretches[stretch_at].first_packet == made.packet &&
				    !r.stretches[stretch_at].needs.empty()) {
					end = resume_at = i;
					break;
				}
			}
			if (!moved) continue;
			const auto* state = static_cast<const NativeXprDrawState*>(made.state);
			const auto* set   = static_cast<const TableImageSet*>(made.set);
			if (state != nullptr && std::ranges::find(states, made.state) == states.end()) {
				states.push_back(made.state);
				if (later && !NativeXprTargetsPlaced(*state)) {
					end = resume_at = pass_first;
					break;
				}
				if (!NativeXprEmitTargets(*state, handle, set, &buffer)) {
					end = i;
					break;
				}
			}
			if (set != nullptr && std::ranges::find(sets, std::pair {made.set, made.state}) == sets.end()) {
				sets.emplace_back(made.set, made.state);
				if (later && !TableTransitPlaced(*set, state)) {
					end = resume_at = pass_first;
					break;
				}
				if (!TableTransit(*set, handle, state)) {
					end = i;
					break;
				}
			}
		}
		if (end <= begin) resume_at = 0;
		// (A deferred need past the part: the strand's later draws are the GPU thread's, which emits their targets.)
		if (resume_at != 0 &&
		    std::ranges::any_of(clear_later, [&](const auto& deferred) { return deferred.made >= end; }))
			resume_at = 0;
		if (end > begin) {
			// The part's first pass, begun by the GPU thread in its own stream; then the part's calls, which end and
			// begin its later passes themselves, written over the marker with this command buffer.
			scheduler.BeginRendering(clearing ? clear.rendering
			                                  : static_cast<const NativeXprDrawState*>(r.made[begin].state)->rendering);
			for (auto i = begin + 1; i < end; ++i)
				passes += r.made[i].pass_begin ? 1u : 0u;
			const uint64_t from = begin == strand.first_made ? 0 : r.made[begin].commands_begin;
			const bool     last = resume_at == 0;
			spliced = LocalVulkanRecording::SpliceApartPart(strand.buffer, static_cast<VkCommandBuffer>(handle), from,
			                                                r.made[end - 1].commands_end, last);
			if (spliced) {
				// (The replay holds the buffer until it has run: the strand's hold goes with its last part.)
				if (last) strand.buffer = {};
				// (The pass the part's calls left open is its last draw's: the command buffer's to end.)
				if (passes > 1)
					buffer.AdoptRendering(static_cast<const NativeXprDrawState*>(r.made[end - 1].state)->rendering);
				WeaveStamps(s, begin, end);
			}
		}
	}
	if (end <= begin || !spliced) return own();
	// What the GPU thread had bound is not what its command buffer holds after the part.
	auto& table           = OwnTable();
	NativeXprState().emit = {};
	table.last_pair       = nullptr;
	table.last_variant    = 0;
	table.last_state      = nullptr;
	table.last_set        = nullptr;
	table.images_epoch    = 0;
	NativeXprForgetState();
	if (!resume) ++o.strands_spliced;
	st.spliced   = resume_at == 0;
	st.resume_at = resume_at;
	// The part's packets, for the command processor to skip by the register log (or pass one by one without it).
	const auto& log = r.log;
	o.skip          = log.on && log.draws.size() == r.made.size();
	o.skip_begin    = begin;
	o.skip_end      = end;
	if (o.skip)
		for (auto i = begin; i < end; ++i)
			o.spliced += log.draws[i].packets;
	else
		o.spliced += end - begin;
	until    = r.made[end - 1].packet_end;
	o.cursor = end;
	return true;
}

bool RenderExecutor::WeaveSkip(uint32_t& begin, uint32_t& end) noexcept {
	if (!g_weave_turn || !g_weave_turn->open.skip) return false;
	auto& o = g_weave_turn->open;
	o.skip  = false;
	begin   = o.skip_begin;
	end     = o.skip_end;
	return true;
}

void RenderExecutor::WeaveClose() noexcept {
	if (!g_weave_turn || g_weave_turn->open.parcel == nullptr) return;
	auto& o = g_weave_turn->open;
	if (o.own != 0) LiveCounters::Add(LiveCounters::WeaveDrawsOwn, o.own);
	if (o.spliced != 0) LiveCounters::Add(LiveCounters::WeaveDrawsSpliced, o.spliced);
	if (o.strands_spliced != 0) LiveCounters::Add(LiveCounters::WeaveStrandsSpliced, o.strands_spliced);
	if (o.strands_cut != 0) LiveCounters::Add(LiveCounters::WeaveStrandsCut, o.strands_cut);
	o.parcel = nullptr;
	o.result = nullptr;
	o.begun  = nullptr;
	o.skip   = false;
}

void RenderExecutor::WeaveRelease(Weave::Parcel& parcel, bool used) {
	if (!g_weave_turn || !parcel.result) return;
	auto&      t     = *g_weave_turn;
	auto&      begun = t.begun[parcel.serial % Weave::ParcelSlots];
	const auto ring  = parcel.result->slab.ring;
	// (Released once: the GPU thread's release of a parcel it opened comes before its slot's, which then finds none.)
	if (begun.slab < 0 || ring < 0 || static_cast<uint32_t>(ring) >= t.worker_count) return;
	(void)t.workers[static_cast<uint32_t>(ring)].ring.Release(begun.slab,
	                                                          used ? m_context.GetCommandScheduler().CurrentTick() : 0);
	begun.slab = -1;
}

void RenderExecutor::WeaveReclaim() {
	if (!g_weave_turn) return;
	auto& scheduler = m_context.GetCommandScheduler();
	for (uint32_t i = 0; i < g_weave_turn->worker_count; ++i)
		if (auto& ring = g_weave_turn->workers[i].ring; ring.mapped != nullptr) ring.Reclaim(scheduler);
}

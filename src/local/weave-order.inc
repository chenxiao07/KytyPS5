// The weave on the GPU thread: CommandProcessor members at queue 0's top-level draw packets (weave.h). Included by
// graphicsRun.cpp under KYTY_LOCAL_VULKAN_RECORDING, after weave-reader.inc and before ProcessPm4.
//
// ProcessPm4's loop calls WeaveAt at each such packet while the weave runs on queue 0's processor
// (ProcessPm4Loop<true>, before the loop's own Scout::NoteDraw). With no parcel open, it passes the parcels it is past,
// posts its registers for the reader when the reader asked, and finds the parcel that starts at this packet. It opens a
// parcel whose worker is done (or, by its pace, done within Weave::WaitBudgetUs) when the parcel's start registers are
// its own and the executor's tests pass (RenderExecutor::WeaveOpen). Inside the open parcel each draw packet is the
// executor's turn (WeaveTurnAt): a part of a worker's strand spliced there, whose packets the GPU thread then skips,
// applying the register log's last write of each block (WeaveSkip); or the GPU thread's own draw, made today's way by
// the loop. The parcel closes at the first draw packet outside it, or when a handler's draw in it changed the
// registers.

namespace {

// The GPU thread's side of the order: the first published parcel it has not passed; after it sent the reader out of
// step, the reader's generation then, at or below which every parcel is passed unused; whether a parcel is open, and
// the open parcel (its serial, submission, buffer and packets); the packet a spliced part's draws are passed one by one
// up to (the register log off); a handler's draw in it that may have changed the registers, one being made (watched:
// WeaveYieldBegin), and the registers before that draw; a queue-0 DISPATCH_DIRECT's place and the write clock before
// it, the records' cursor, and the matched copies' write-clock windows an open passes over (the linear-copy records);
// the weave started, or stood down at its start; the submission queue 0's processor is in (the scout's numbering,
// WeaveSubmission), and that processor while the weave runs on it (null: none; ProcessPm4 tests it, one load). All of
// it the GPU thread's: there is one weave, on queue 0's one processor.
struct WeaveOrder {
	uint64_t                                   cursor = 0, distrust = 0, sequence = 0;
	bool                                       distrusting = false, open = false;
	uint64_t                                   open_serial = 0, open_sequence = 0;
	const uint32_t*                            open_base  = nullptr;
	const uint32_t*                            open_first = nullptr;
	const uint32_t*                            open_end   = nullptr;
	const uint32_t*                            skip_until = nullptr;
	bool                                       changed = false, yield = false;
	Weave::Registers                           before;
	const uint32_t*                            copy_base  = nullptr;
	uint32_t                                   copy_at    = 0;
	uint64_t                                   copy_clock = 0, copies_cursor = 0;
	std::vector<std::pair<uint64_t, uint64_t>> copies_passed;
	bool                                       started = false, failed = false;
	const CommandProcessor*                    turns = nullptr;
};
WeaveOrder g_weave_order;

[[nodiscard]] int64_t WeaveNowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// The reader's registers no longer trusted (a parcel's start registers not the GPU thread's, or a handler's draw in an
// open parcel changed them): the parcels it cut before its next sync, from the same registers, are passed unused, and
// it goes out of step.
void WeaveOutOfStep() {
	auto& order       = g_weave_order;
	order.distrust    = g_weave->generation.load(std::memory_order_acquire);
	order.distrusting = true;
	g_weave->force_out.fetch_add(1, std::memory_order_release);
	Weave::WakeReader();
}

// A parcel a worker is still on (Taken): waited for only when its pace so far (the time since it took the parcel over
// the dwords it has reached) says it is done within Weave::WaitBudgetUs, and at most that long. Its state then.
[[nodiscard]] uint32_t WeaveWait(const Weave::Parcel& parcel) {
	constexpr int64_t budget_ns = int64_t {Weave::WaitBudgetUs} * 1000;
	const auto        now       = WeaveNowNs();
	const auto        words     = std::max<uint64_t>(static_cast<uint64_t>(parcel.end - parcel.first), 1u);
	const auto        done      = std::min(parcel.progress.load(std::memory_order_relaxed), words);
	const auto        elapsed   = std::max<int64_t>(now - parcel.taken_ns.load(std::memory_order_relaxed), 0);
	if (done == 0 || static_cast<double>(elapsed) * static_cast<double>(words - done) / static_cast<double>(done) >
	                     static_cast<double>(budget_ns))
		return Weave::Parcel::Taken;
	uint32_t state = Weave::Parcel::Taken;
	for (const auto deadline = now + budget_ns;
	     (state = parcel.state.load(std::memory_order_acquire)) == Weave::Parcel::Taken && WeaveNowNs() < deadline;)
		_mm_pause();
	return state;
}

} // namespace

// The weave's register image of this processor, taken, applied and compared whole (the HW structs' defaulted
// operator==, never a byte compare: their padding is not theirs). Left out, as no packet of a parcel writes them: the
// CE/DE counters and the constant RAM (the constant engine's packets and a submission's slices), the flip and the
// submit id (set per submission), the synthetic occlusion counter (EVENT_WRITE's), the draw-run skip (the GPU thread's
// skip clears it).
void CommandProcessor::WeaveSnapshot(Weave::Registers& out) const {
	out.ctx            = m_ctx;
	out.saved_ctx      = m_saved_ctx;
	out.pushed         = m_context_state_pushed;
	out.ucfg           = m_ucfg;
	out.sh             = m_sh_ctx;
	out.marker         = m_user_data_marker;
	out.index_type     = m_index_type_and_size;
	out.index_size     = m_index_buffer_size;
	out.index_base     = m_index_base_addr;
	out.draw_args      = m_draw_indirect_args_base_addr;
	out.dispatch_args  = m_dispatch_indirect_args_base_addr;
	out.num_instances  = m_num_instances;
	out.predicate_skip = m_predicate_skip;
}

void CommandProcessor::WeaveRestore(const Weave::Registers& in) {
	m_ctx                              = in.ctx;
	m_saved_ctx                        = in.saved_ctx;
	m_context_state_pushed             = in.pushed;
	m_ucfg                             = in.ucfg;
	m_sh_ctx                           = in.sh;
	m_user_data_marker                 = in.marker;
	m_index_type_and_size              = in.index_type;
	m_index_buffer_size                = in.index_size;
	m_index_base_addr                  = in.index_base;
	m_draw_indirect_args_base_addr     = in.draw_args;
	m_dispatch_indirect_args_base_addr = in.dispatch_args;
	m_num_instances                    = in.num_instances;
	m_predicate_skip                   = in.predicate_skip;
}

// WeaveSame's compare of a large part: == answered by its bytes first, by its members only where the bytes differ (a
// member, a padding byte, +0 against -0). Equal bytes are equal members in HW::UserConfig and HW::Shader: every member
// there, through its nested structs, is an integer, a bool, an enum or an array of them, and every operator== in
// hardwareContext.h is defaulted. In HW::Context too, but for its floats, where a NaN is unequal to itself: a context of
// equal bytes is equal when none of them is a NaN (WeaveNoNan, which a float member added to the context joins).
template <typename Part>
[[nodiscard]] static bool WeaveSamePart(const Part& a, const Part& b) {
	return std::memcmp(&a, &b, sizeof(Part)) == 0 || a == b;
}

// None of the context's 145 floats is a NaN: the line width, the depth clear value and bounds, the blend colour, the
// polygon offset, the guard bands and the 16 viewports' scales, offsets and depth ranges.
[[nodiscard]] static bool WeaveNoNan(const HW::Context& ctx) {
	uint32_t nan = 0; // (a NaN's bits, the sign left out, are above infinity's)
	const auto note = [&nan](float value) {
		nan |= static_cast<uint32_t>((std::bit_cast<uint32_t>(value) & 0x7fffffffu) > 0x7f800000u);
	};
	note(ctx.GetLineWidth());
	note(ctx.GetDepthClearValue());
	note(ctx.GetDepthBoundsMin());
	note(ctx.GetDepthBoundsMax());
	const auto& blend = ctx.GetBlendColor();
	for (const float value: {blend.red, blend.green, blend.blue, blend.alpha}) note(value);
	const auto& poly = ctx.GetPolyOffset();
	for (const float value: {poly.clamp, poly.front_scale, poly.front_offset, poly.back_scale, poly.back_offset}) note(value);
	const auto& screen = ctx.GetScreenViewport();
	for (const float value: {screen.guard_band_horz_clip, screen.guard_band_vert_clip, screen.guard_band_horz_discard,
	                         screen.guard_band_vert_discard})
		note(value);
	for (const auto& v: screen.viewports)
		for (const float value: {v.zmin, v.zmax, v.xscale, v.xoffset, v.yscale, v.yoffset, v.zscale, v.zoffset}) note(value);
	return nan == 0;
}

[[nodiscard]] static bool WeaveSameContext(const HW::Context& a, const HW::Context& b) {
	return std::memcmp(&a, &b, sizeof(HW::Context)) == 0 ? WeaveNoNan(a) : a == b;
}

bool CommandProcessor::WeaveSame(const Weave::Registers& in) const {
	return m_index_type_and_size == in.index_type && m_index_buffer_size == in.index_size &&
	       m_index_base_addr == in.index_base && m_draw_indirect_args_base_addr == in.draw_args &&
	       m_dispatch_indirect_args_base_addr == in.dispatch_args && m_num_instances == in.num_instances &&
	       m_user_data_marker == in.marker && m_predicate_skip == in.predicate_skip &&
	       m_context_state_pushed == in.pushed && WeaveSamePart(m_sh_ctx, in.sh) && WeaveSamePart(m_ucfg, in.ucfg) &&
	       WeaveSameContext(m_ctx, in.ctx) && WeaveSameContext(m_saved_ctx, in.saved_ctx);
}

// The log's blocks [begin, end) written into a register image's parts, but a block written again before `end`
// (block_next). True: the scalars' block was among them (in `scalars`).
static bool WeaveLogPut(const Weave::RegisterLog& log, uint32_t begin, uint32_t end, HW::Context& ctx,
                        HW::UserConfig& ucfg, HW::Shader& sh, WeaveScalars& scalars) {
	constexpr uint32_t block = Weave::LogBlockBytes;
	// (A block into one object of the image, `size` bytes: what of it lies inside the object.)
	const auto put = [&](void* live, uint32_t offset, uint32_t size, const uint8_t* bytes) {
		std::memcpy(static_cast<uint8_t*>(live) + offset, bytes, std::min(block, size - offset));
	};
	bool scalars_set = false;
	for (auto k = begin; k < end; ++k) {
		if (log.block_next[k] < end) continue;
		const uint32_t at    = log.block_ids[k] * block;
		const auto*    bytes = log.block_bytes.data() + size_t {k} * block;
		if (at < WeaveLogUcfg) {
			put(&ctx, at, sizeof(ctx), bytes);
		} else if (at < WeaveLogSh) {
			put(&ucfg, at - WeaveLogUcfg, sizeof(ucfg), bytes);
		} else if (at < WeaveLogScalars) {
			put(&sh, at - WeaveLogSh, sizeof(sh), bytes);
		} else {
			std::memcpy(&scalars, bytes, sizeof(scalars));
			scalars_set = true;
		}
	}
	return scalars_set;
}

// The narrow walk test, which the executor's turn asks before it splices a strand that follows a hole, or a part of one
// reached past its first draw (RenderExecutor::WeaveTurnAt): the GPU thread's registers now against the worker's
// before made draw `index`, whole. A block a draw reads but no packet of the part writes may differ too, so the test is
// not the blocks the log writes. At the strand's first draw (`index` == `first`) the worker's are the strand's start
// image; past it, that image with the register log's blocks of the strand's draws up to `index`, which hold the
// registers at each one's end (its own draw packets set only the start-instance SGPR, which a draw packet sets anew:
// it is taken as the GPU thread's). With no log, a late part is not spliced.
bool CommandProcessor::WeaveSameOf(const void* processor, const Weave::Registers& start, const Weave::RegisterLog& log,
                                   uint32_t first, uint32_t index) {
	const auto& cp = *static_cast<const CommandProcessor*>(processor);
	if (index == first) return cp.WeaveSame(start);
	if (!log.on || index < first || index >= log.draws.size()) return false;
	thread_local Weave::Registers worker;
	worker = start;
	WeaveScalars scalars {.marker        = static_cast<uint32_t>(worker.marker),
	                      .index_type    = worker.index_type,
	                      .index_size    = worker.index_size,
	                      .num_instances = worker.num_instances,
	                      .index_base    = worker.index_base,
	                      .draw_args     = worker.draw_args,
	                      .dispatch_args = worker.dispatch_args};
	if (WeaveLogPut(log, first == 0 ? 0u : log.draws[first - 1].blocks_end, log.draws[index].blocks_end, worker.ctx,
	                worker.ucfg, worker.sh, scalars)) {
		worker.marker        = static_cast<HW::UserSgprType>(scalars.marker);
		worker.index_type    = scalars.index_type;
		worker.index_size    = scalars.index_size;
		worker.num_instances = scalars.num_instances;
		worker.index_base    = scalars.index_base;
		worker.draw_args     = scalars.draw_args;
		worker.dispatch_args = scalars.dispatch_args;
	}
	worker.sh.SetStartInstanceUserSgpr(cp.m_sh_ctx.GetVs().start_instance_user_sgpr);
	return cp.WeaveSame(worker);
}

namespace {

// The linear-copy records WeaveCopiesStart made, freed (weave-copies.inc).
void WeaveCopiesForget();

} // namespace

// The GPU thread, at queue 0's first top-level draw packet woven: the reader's and the workers' processors, the
// executor's state for them, the linear-copy records, and the threads. Once; false: the weave stood down (each reason
// says so in one line, Weave::Start), and the run goes on in mode 0 as if it had never been asked (`undo`).
bool CommandProcessor::WeaveStart() {
	auto& order = g_weave_order;
	if (order.started || order.failed) return order.started;
	order.failed        = true; // (until it has started: never tried again)
	const auto workers  = Weave::Workers();
	auto&      executor = m_renderer.GetRenderExecutor();
	// A start that failed frees what was made for it. No weave thread runs (none started, or Weave::Start joined them)
	// and no parcel was opened, so it goes in the reverse order: the linear-copy records, the reader's and the
	// workers' processors, the executor's views and rings (Weave::Start's runtime stays, as it was made to). Then the
	// switch is 0 and the GPU thread's core is given back if only the weave kept it (Weave::StartFailed).
	const auto undo = [&executor] {
		WeaveCopiesForget();
		delete std::exchange(g_weave, nullptr);
		executor.WeaveForget();
		Weave::StartFailed();
	};
	CheckBuffer();
	if (workers == 0 || !executor.WeaveStart(workers, this, &CommandProcessor::WeaveSameOf)) {
		undo(); // (stood down: Enabled() false from here on)
		return false;
	}
	auto* weave     = new WeaveCp;
	weave->executor = &executor;
	weave->reader   = std::make_unique<CommandProcessor>(m_renderer, m_interrupt_event_id);
	for (uint32_t i = 0; i < workers && i < Weave::MaxWorkers; ++i)
		weave->workers[i] = std::make_unique<CommandProcessor>(m_renderer, m_interrupt_event_id);
	weave->gpu_sequence.store(g_weave_order.sequence, std::memory_order_relaxed);
	g_weave = weave;
	WeaveCopiesStart(); // (before the reader and the workers start)
	if (!Weave::Start({.read    = &CommandProcessor::WeaveRead,
	                   .make    = &CommandProcessor::WeaveMake,
	                   .release = &CommandProcessor::WeaveRelease})) {
		undo();
		return false;
	}
	order.started = true;
	order.failed  = false;
	return true;
}

void CommandProcessor::WeaveSubmission(uint64_t sequence) {
	// (Queue 0's other submissions, a flip preparation's: not the reader's, their draws made today's way.)
	if (sequence == 0) {
		g_weave_order.turns = nullptr;
		return;
	}
	g_weave_order.sequence = sequence;
	// One parallel path at a time: while KYTY_SPECULATE is 4 the weave stays paused (the GPU thread paused it at the
	// live change, before any other packet or guest command: WeaveModeChanged; here a no-op then), and it resumes at
	// the first submission back in mode 0.
	if (kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0) {
		if (g_weave_order.started && Weave::Active()) WeavePause();
		g_weave_order.turns = nullptr;
		return;
	}
	if (g_weave_order.failed) {
		g_weave_order.turns = nullptr;
		return;
	}
	if (g_weave_order.started) {
		g_weave->gpu_sequence.store(sequence, std::memory_order_release);
		if (!Weave::Active()) {
			// (Back from mode 4: the registers the reader went on with are the stream's no longer.)
			Weave::Resume();
			WeaveOutOfStep();
		}
	}
	g_weave_order.turns = this;
}

// ThreadRun, at the top of its loop, right after a live change set KYTY_SPECULATE to 4 (no queue-0 submission is
// started then: the change waits until none is). Mode 4's threads, started by the change, only read until the GPU
// thread runs; the weave pauses here first, so the GPU thread runs no packet or guest command while a worker may be
// inside a draw.
void CommandProcessor::WeaveModeChanged() {
	if (g_weave_order.started && Weave::Active()) WeavePause();
}

// The open parcel closed with the tick of now, then every parcel dropped and released, the workers outside every draw
// and idle (Weave::Pause); the ones published are passed with it. Only between queue 0's submissions, before the
// next one's first packet (WeaveSubmission, WeaveModeChanged).
void CommandProcessor::WeavePause() {
	WeaveClose(false);
	Weave::Pause();
	g_weave_order.cursor = Weave::Published();
}

bool CommandProcessor::WeaveAt(Pm4Execution& execution) {
	auto& order = g_weave_order;
	if (!order.started && !WeaveStart()) {
		order.turns = nullptr;
		return false;
	}
	// (No longer active, stopping: the packet is made today's way.)
	if (!Weave::Active()) return false;
	const auto& cursor = execution.m_buffer_stack.back();
	const auto* base   = cursor.Position();
	const auto* packet = base + cursor.offset_dw;
	if (order.open) {
		if (order.changed) {
			WeaveClose(true); // (a handler's draw in it changed the registers: the reader's after it may not hold)
		} else if (base == order.open_base && g_weave_order.sequence == order.open_sequence &&
		           packet >= order.open_first && packet < order.open_end) {
			return WeaveInside(execution, packet);
		} else {
			WeaveClose(false);
		}
	}
	const auto serial = WeaveFind(cursor);
	if (serial < 0) return false;
	auto& parcel = Weave::ParcelAt(static_cast<uint64_t>(serial));
	++order.cursor;
	// (Not under a frame capture, a debug dump or the XPR capture, which see every draw on the normal path; not while
	// predicated packets are skipped one by one; not with the native path off. A capture can begin while the game
	// runs.)
	if (GraphicsRunDebugDumpEnabled() || FrameCapture::Active() || XprCapture::Enabled() ||
	    ShouldSkipPredicatedPackets() || kyty_local_native_xpr_mode.load(std::memory_order_relaxed) == 0) {
		LiveCounters::Add(LiveCounters::WeaveParcelsRefusedCapture);
		Weave::PassParcel(parcel);
		return false;
	}
	auto state = parcel.state.load(std::memory_order_acquire);
	if (state == Weave::Parcel::Taken) state = WeaveWait(parcel);
	if (state != Weave::Parcel::Done) {
		// (Not done, its worker on it or not yet at it: dropped, its worker stops at its next draw, and the GPU thread
		// makes it today's way.)
		if (state == Weave::Parcel::Taken || state == Weave::Parcel::Cut)
			LiveCounters::Add(LiveCounters::WeaveParcelsRefusedLate);
		Weave::PassParcel(parcel);
		return false;
	}
	// (Its worker started from the reader's registers at its first draw: they must be the GPU thread's here. A
	// difference sends the reader out of step, so the parcels after this one, cut from the same registers, are not each
	// found wrong in turn.)
	if (!WeaveSame(parcel.start)) {
		LiveCounters::Add(LiveCounters::WeaveParcelsRefusedRegisters);
		WeaveOutOfStep();
		Weave::PassParcel(parcel);
		return false;
	}
	// (What the linear-copy records say of it: every copy it read through ran as recorded; the windows of the matched
	// copies listed before it, which the write ring's test passes over.)
	const bool copies_matched = WeaveCopiesOf(parcel, order.copies_passed);
	CheckBuffer();
	if (!g_weave->executor->WeaveOpen(parcel, copies_matched, order.copies_passed)) {
		Weave::PassParcel(parcel);
		return false;
	}
	order.open          = true;
	order.open_serial   = static_cast<uint64_t>(serial);
	order.open_sequence = g_weave_order.sequence;
	order.open_base     = base;
	order.open_first    = parcel.first;
	order.open_end      = parcel.end;
	order.skip_until    = nullptr;
	order.changed       = false;
	return WeaveInside(execution, packet);
}

// At a draw packet of queue 0's top level outside an open parcel (the cursor on it): the slabs whose tick is done
// reclaimed, the linear-copy records and the parcels the GPU thread is past passed; the serial of the parcel that
// starts here (-1: none). With none here, the GPU thread's registers posted when the reader asked for them at or before
// this packet.
int64_t CommandProcessor::WeaveFind(const Pm4Execution::BufferCursor& cursor) {
	auto&       order  = g_weave_order;
	const auto* base   = cursor.Position();
	const auto* packet = base + cursor.offset_dw;
	g_weave->executor->WeaveReclaim();
	(void)WeaveCopiesPass(base, cursor.offset_dw, false);
	const auto published = Weave::Published();
	for (; order.cursor < published; ++order.cursor) {
		auto& parcel = Weave::ParcelAt(order.cursor);
		if (parcel.serial != order.cursor) break; // (never: parcels are published and passed in order)
		const bool distrusted = order.distrusting && parcel.generation <= order.distrust;
		const bool behind =
		    parcel.sequence < g_weave_order.sequence ||
		    (parcel.sequence == g_weave_order.sequence && (parcel.guest != base || parcel.first < packet));
		if (!behind && !distrusted) break;
		Weave::PassParcel(parcel);
	}
	if (order.cursor < published) {
		const auto& parcel = Weave::ParcelAt(order.cursor);
		if (parcel.serial == order.cursor && parcel.sequence == g_weave_order.sequence && parcel.guest == base &&
		    parcel.first == packet)
			return static_cast<int64_t>(order.cursor);
	}
	auto& sync = g_weave->sync;
	if (sync.state.load(std::memory_order_acquire) == 1 &&
	    (g_weave_order.sequence > sync.wanted_sequence ||
	     (g_weave_order.sequence == sync.wanted_sequence && cursor.offset_dw >= sync.wanted_offset))) {
		sync.sequence = g_weave_order.sequence;
		sync.guest    = base;
		sync.dwords   = static_cast<uint32_t>(cursor.commands.size());
		sync.offset   = cursor.offset_dw;
		WeaveSnapshot(sync.registers);
		sync.state.store(2, std::memory_order_release);
		Weave::WakeReader();
	}
	return -1;
}

// A draw packet of the open parcel (`packet`, the guest's): passed when a spliced part's draws are passed one by one up
// to `skip_until`; else the executor's turn there: a part spliced (true; its packets skipped from the register log, or
// passed one by one without it) or the GPU thread's own draw (false; the loop makes it today's way).
bool CommandProcessor::WeaveInside(Pm4Execution& execution, const uint32_t* packet) {
	auto& order = g_weave_order;
	if (packet < order.skip_until) {
		WeavePassDraw(execution);
		return true;
	}
	CheckBuffer(); // (as before the GPU thread's own draw: the scheduler active)
	auto&           executor = *g_weave->executor;
	const uint32_t* until    = nullptr;
	if (!executor.WeaveTurnAt(packet, until)) return false;
	uint32_t    begin = 0, end = 0;
	const auto& parcel = Weave::ParcelAt(order.open_serial);
	if (executor.WeaveSkip(begin, end) && begin < end && end <= parcel.result->log.draws.size()) {
		WeaveSkip(execution, parcel, begin, end, until);
		return true;
	}
	// (No register log: the part's packets go through the loop, its draw packets passed here one by one.)
	order.skip_until = until;
	WeavePassDraw(execution);
	return true;
}

// A spliced part's made draws [begin, end) skipped to `until` (the packet after the last one's draw packets): the
// registers set from the log's blocks, only the last write of each block in the part, which give what the handlers
// would; the start-instance SGPR as the part's last draw packet sets it (a block it doesn't change keeps the GPU
// thread's value from before the part); and what the loop counts and calls for the packets advanced: the counts by
// opcode (each made draw one, as its entry is one turn of the loop), the scout's draw numbering (its last packet noted,
// as the loop notes a run's first and skips the rest), the observer's record of the registers forgotten (the next draw
// is not clean), the table draws and the batch's draws, once for the part (a batch end inside it submits at its end
// what the splice put in the command buffer already). Guest commands are served at the next packet, as after a faster
// walk.
void CommandProcessor::WeaveSkip(Pm4Execution& execution, const Weave::Parcel& parcel, uint32_t begin, uint32_t end,
                                 const uint32_t* until) {
	auto&       cursor = execution.m_buffer_stack.back();
	const auto& log    = parcel.result->log;
	WeaveLogApply(parcel, begin == 0 ? 0u : log.draws[begin - 1].blocks_end, log.draws[end - 1].blocks_end);
	uint32_t packets = 0, draws = 0;
	for (auto i = begin; i < end; ++i) {
		const auto& draw = log.draws[i];
		if (i > begin)
			for (auto k = log.draws[i - 1].counts_end; k < draw.counts_end; ++k)
				for (uint32_t n = 0; n < log.counts[k].count; ++n)
					LiveCounters::AddPm4(log.counts[k].opcode);
		LiveCounters::AddPm4(draw.opcode);
		packets += draw.packets;
		draws += draw.draws;
	}
	const auto& last   = log.draws[end - 1];
	const auto* packet = cursor.commands.data() + (last.last - cursor.Position()); // (the view the loop reads)
	if (Scout::Running()) {
		if (packets > 1) Scout::SkipDraws(packets - 1);
		Scout::NoteDraw(packet);
	}
	m_sh_ctx.SetStartInstanceUserSgpr(WeaveStartInstanceSgpr(last.opcode, packet));
	DrawStateObserver::Invalidate();
	LiveCounters::Add(LiveCounters::TableDraws, draws);
	m_draw_run_skip           = 0;
	cursor.offset_dw          = static_cast<uint32_t>(until - cursor.Position());
	execution.m_made_progress = true;
	GetScheduler().CompleteDraws(draws);
}

// The log's blocks [begin, end) written into the registers, but a block the part writes again later (block_next).
void CommandProcessor::WeaveLogApply(const Weave::Parcel& parcel, uint32_t begin, uint32_t end) {
	WeaveScalars scalars;
	if (!WeaveLogPut(parcel.result->log, begin, end, m_ctx, m_ucfg, m_sh_ctx, scalars)) return;
	m_user_data_marker                 = static_cast<HW::UserSgprType>(scalars.marker);
	m_index_type_and_size              = scalars.index_type;
	m_index_buffer_size                = scalars.index_size;
	m_num_instances                    = scalars.num_instances;
	m_index_base_addr                  = scalars.index_base;
	m_draw_indirect_args_base_addr     = scalars.draw_args;
	m_dispatch_indirect_args_base_addr = scalars.dispatch_args;
}

// A draw packet of a spliced part passed as the loop passes a draw the table path took (no register log: the part's
// other packets go through the loop): the scout's numbering, the start-instance SGPR, the observer (its record then
// forgotten: the next draw is not clean), the counts, the table draws and the batch's draws.
void CommandProcessor::WeavePassDraw(Pm4Execution& execution) {
	auto&          cursor = execution.m_buffer_stack.back();
	const auto*    packet = cursor.commands.data() + cursor.offset_dw;
	const uint32_t header = packet[0];
	const uint32_t opcode = (header >> 8u) & 0xffu;
	const uint32_t length =
	    std::min(KYTY_PM4_LEN(header), static_cast<uint32_t>(cursor.commands.size()) - cursor.offset_dw);
	if (Scout::Running()) Scout::NoteDraw(packet);
	m_sh_ctx.SetStartInstanceUserSgpr(WeaveStartInstanceSgpr(opcode, packet));
	if (kyty_local_native_xpr_mode.load(std::memory_order_relaxed) != 0)
		DrawStateObserver::ObservePacket(opcode, packet, length, false);
	DrawStateObserver::Invalidate();
	LiveCounters::AddPm4(opcode);
	m_draw_run_skip      = 0;
	const uint32_t draws = header == 0xc0083800u ? packet[5] : 1u;
	LiveCounters::Add(LiveCounters::TableDraws, draws);
	cursor.offset_dw += length;
	execution.m_made_progress = true;
	GetScheduler().CompleteDraws(draws);
}

// The open parcel left: the turn's state closed (its counts added), its slab released with the tick of now (after its
// last splice), the parcel passed. `distrust`: a handler's draw in it changed the registers, and the reader goes out of
// step.
void CommandProcessor::WeaveClose(bool distrust) {
	auto& order = g_weave_order;
	if (!order.open) return;
	auto& parcel   = Weave::ParcelAt(order.open_serial);
	auto& executor = *g_weave->executor;
	executor.WeaveClose();
	executor.WeaveRelease(parcel, true);
	Weave::PassParcel(parcel);
	order.open       = false;
	order.yield      = false;
	order.skip_until = nullptr;
	order.changed    = false;
	if (distrust) WeaveOutOfStep();
}

// A draw of the open parcel a handler makes (the table path did not take it on the GPU thread: after the native
// attempts): the registers before it, compared once the packet is consumed (WeaveYieldEnd). A handler may change
// registers a worker's walk doesn't (an indirect draw's instance count, the interface's scopes); then the parcel
// closes at its next draw packet and the reader goes out of step. A packet that never comes back to the loop (a
// suspension) closes it too.
void CommandProcessor::WeaveYieldBegin(const Pm4Execution& execution) {
	auto&       order  = g_weave_order;
	const auto& cursor = execution.m_buffer_stack.back();
	const auto* packet = cursor.Position() + cursor.offset_dw;
	if (execution.m_buffer_stack.size() != 1 || cursor.Position() != order.open_base ||
	    g_weave_order.sequence != order.open_sequence || packet < order.open_first || packet >= order.open_end)
		return;
	WeaveSnapshot(order.before);
	order.changed = true;
	order.yield   = true;
}

void CommandProcessor::WeaveYieldEnd() {
	g_weave_order.yield = false;
	if (WeaveSame(g_weave_order.before)) g_weave_order.changed = false;
}

// The weave's linear-copy records: CommandProcessor members of the reader, the workers and the GPU thread (weave.h).
// Included by graphicsRun.cpp under KYTY_LOCAL_VULKAN_RECORDING, after weave-order.inc.
//
// Demon's Souls copies its constants with a compute shader (the linear copy shader), which the GPU thread runs on the
// CPU (DemonsSouls::TryLinearCopy, BufferCache::CopyBuffer). A parcel's draws read what the copies dispatched before it
// write, and those run only when the GPU thread reaches them. So the reader records each queue-0 top-level dispatch of
// the shader it steps over, decoded as the GPU thread's fast path decodes it, and a worker reads a range that a copy
// listed before its parcel and not yet run will write from that copy's source instead (RenderExecutor::WeaveCopy). The
// GPU thread tests each record at its dispatch against what CopyBuffer did there (Weave::t_copy_done) and publishes how
// many it is past. At a parcel's open it refuses the parcel if a copy it read through didn't run as recorded, and the
// scout's write ring passes over the entries of the copies that did.

namespace {

// A record, made in order (the reader's): a dispatch of the linear copy shader, decoded or undecoded (then with its
// destination's bound); its submission (the scout's numbering), its commands and its packet's dword there.
struct WeaveCopyRecord {
	enum Kind : uint8_t { Decoded, Undecoded };
	uint64_t        index    = UINT64_MAX;
	uint64_t        sequence = 0;
	const uint32_t* guest    = nullptr;
	uint32_t        offset   = 0;
	Kind            kind     = Decoded;
	uint64_t        src = 0, dst = 0, bytes = 0;
};
// What became of a record (the GPU thread's): its CPU copy ran with its numbers (matched), it ran otherwise
// (mismatched), or the GPU thread went past it unrun (unmet); the scout's write clock around its dispatch.
struct WeaveCopyRun {
	enum Result : uint8_t { Pending, Matched, Mismatched, Unmet };
	uint64_t index       = UINT64_MAX;
	Result   result      = Pending;
	uint64_t clock_begin = 0, clock_end = 0;
};
// The records (`written` made) and their runs (final below `run`), record `index` at `index % Weave::CopyRecords`.
// Made by the GPU thread before the reader and the workers start, never freed while the game runs (unless the weave
// could not start: WeaveCopiesForget).
struct WeaveCopies {
	std::array<WeaveCopyRecord, Weave::CopyRecords> records;
	std::array<WeaveCopyRun, Weave::CopyRecords>    runs;
	std::atomic<uint64_t>                           written {0}, run {0};
};
WeaveCopies* g_weave_copies = nullptr;

void WeaveCopiesForget() {
	delete std::exchange(g_weave_copies, nullptr);
}

// The records a worker may still read once listed: the reader waits for the GPU thread to be this far past a record
// before it writes over it.
constexpr uint64_t WeaveCopyMargin = 1024;

} // namespace

// The GPU thread, as the weave starts: the records made.
void CommandProcessor::WeaveCopiesStart() {
	if (g_weave_copies == nullptr) g_weave_copies = new WeaveCopies;
}

// The records made so far (a parcel's copies_end: those before it).
uint64_t CommandProcessor::WeaveCopiesMade() {
	return g_weave_copies != nullptr ? g_weave_copies->written.load(std::memory_order_relaxed) : 0;
}

// The reader, at a queue-0 top-level DISPATCH_DIRECT (`packet`, at dword `at` of `guest`'s commands): a dispatch of the
// linear copy shader recorded, decoded (DemonsSouls::LinearCopyOf), or undecoded with its destination's bound (256
// bytes a thread group). Any other dispatch is a GPU write the scout's write ring marks, and is not recorded. False: no
// record is free (the GPU thread is a whole ring behind), and the read stops here rather than wait.
bool CommandProcessor::WeaveNoteCopy(uint64_t sequence, const uint32_t* guest, uint32_t at, const uint32_t* packet) {
	auto* const copies = g_weave_copies;
	if (copies == nullptr) return true;
	const auto& cs         = m_sh_ctx.GetCs();
	const auto& regs       = cs.cs_regs;
	uint64_t    generation = 0;
	const auto  shader     = m_renderer.GetComputeRenderExecutor().LinearCopyShader(generation);
	if (shader == 0 || regs.data_addr != shader) return true;
	const uint32_t  x = packet[1], y = packet[2], z = packet[3], mode = packet[4];
	const auto*     data = cs.cs_user_sgpr.value;
	WeaveCopyRecord record {.sequence = sequence, .guest = guest, .offset = at, .kind = WeaveCopyRecord::Undecoded};
	record.dst   = data[4] | uint64_t {data[5] & 0xffffu} << 32u;
	record.bytes = uint64_t {x} * 256u;
	// (A shader found at another generation of the shader map may be another program now.)
	if (ShaderMapGeneration() == generation && regs.user_sgpr <= std::size(cs.cs_user_sgpr.value)) {
		const DemonsSouls::LinearCopyDispatch dispatch {
		    .user_data          = std::span<const uint32_t> {data, regs.user_sgpr},
		    .threads            = {regs.num_thread_x, regs.num_thread_y, regs.num_thread_z},
		    .group_id           = {regs.tgid_x_en, regs.tgid_y_en, regs.tgid_z_en},
		    .thread_ids         = regs.tidig_comp_cnt + 1u,
		    .workgroup_register = regs.user_sgpr,
		    .tg_size            = regs.tg_size_en,
		    .thread_dimensions  = (mode & (1u << 5u)) != 0};
		if (const auto decoded = DemonsSouls::LinearCopyOf(dispatch, x, y, z, mode); decoded) {
			record.kind  = WeaveCopyRecord::Decoded;
			record.src   = decoded->src;
			record.dst   = decoded->dst;
			record.bytes = decoded->bytes;
		}
	}
	if (record.dst == 0) return true;
	// (A free record: the GPU thread far enough past the one it replaces.)
	const auto index = copies->written.load(std::memory_order_relaxed);
	if (index - copies->run.load(std::memory_order_acquire) >= Weave::CopyRecords - WeaveCopyMargin) return false;
	record.index                                = index;
	copies->records[index % Weave::CopyRecords] = record;
	copies->written.store(index + 1, std::memory_order_release);
	return true;
}

// A worker, before its parcel's first read: the copies listed before the parcel and not yet run, for its reads to go
// through (RenderExecutor::WeaveCopiesBegin, WeaveCopy, WeaveCopiesEnd). Read before the executor takes the parcel's
// clocks (RenderExecutor::WeaveParcelBegin), so a copy run after the list was made writes under a later write clock.
void CommandProcessor::WeaveCopiesList(const Weave::Parcel& parcel) {
	auto* const copies = g_weave_copies;
	if (copies == nullptr) return;
	auto& executor = *g_weave->executor;
	executor.WeaveCopiesBegin(&copies->run);
	const auto run = copies->run.load(std::memory_order_acquire);
	const auto end = std::max(parcel.copies_end, run);
	// (Records below `run` are past; the reader writes over a record only once the GPU thread is WeaveCopyMargin
	// records past it, which a list this short outruns.)
	if (end - run <= Weave::CopyRecords - WeaveCopyMargin)
		for (auto index = run; index < end; ++index) {
			const auto& record = copies->records[index % Weave::CopyRecords];
			if (record.index != index) continue;
			executor.WeaveCopy(index, record.dst, record.src, record.bytes, record.kind == WeaveCopyRecord::Decoded);
		}
	executor.WeaveCopiesEnd();
}

// The GPU thread at dword `at` of queue 0's top-level buffer `base` (the guest's): the records behind it passed, unrun.
// `here` (at a DISPATCH_DIRECT): true when the next record is this dispatch's, left for WeaveCopyAfter.
bool CommandProcessor::WeaveCopiesPass(const uint32_t* base, uint32_t at, bool here) {
	auto* const copies = g_weave_copies;
	if (copies == nullptr) return false;
	auto&      cursor  = g_weave_order.copies_cursor;
	const auto written = copies->written.load(std::memory_order_acquire);
	const auto start   = cursor;
	bool       found   = false;
	for (; cursor < written; ++cursor) {
		const auto& record = copies->records[cursor % Weave::CopyRecords];
		if (record.index != cursor) break; // (never: made in order)
		if (here && record.sequence == g_weave_order.sequence && record.guest == base && record.offset == at) {
			found = true;
			break;
		}
		const bool behind = record.sequence < g_weave_order.sequence ||
		                    (record.sequence == g_weave_order.sequence &&
		                     (record.guest != base || record.offset < at || (here && record.offset == at)));
		if (!behind) break;
		copies->runs[cursor % Weave::CopyRecords] = {.index = cursor, .result = WeaveCopyRun::Unmet};
	}
	if (cursor != start) copies->run.store(cursor, std::memory_order_release);
	return found;
}

// Before a queue-0 top-level DISPATCH_DIRECT's handler (the cursor on it): what BufferCache::CopyBuffer does in it
// recorded from nothing, and the write clock before it. False: no records are made.
bool CommandProcessor::WeaveCopyBefore(const Pm4Execution& execution) {
	if (g_weave_copies == nullptr) return false;
	auto&       order  = g_weave_order;
	const auto& cursor = execution.m_buffer_stack.back();
	order.copy_base    = cursor.Position();
	order.copy_at      = cursor.offset_dw;
	order.copy_clock   = Scout::WriteClock();
	Weave::t_copy_done = {};
	return true;
}

// After it: what CopyBuffer did there against the reader's record of it (a CPU copy of the same numbers: matched; none,
// the GPU path or other numbers: mismatched), with the write clock around it; the record passed.
void CommandProcessor::WeaveCopyAfter() {
	auto&      order = g_weave_order;
	const auto done  = Weave::t_copy_done;
	if (!WeaveCopiesPass(order.copy_base, order.copy_at, true)) return;
	auto&       copies  = *g_weave_copies;
	auto&       cursor  = order.copies_cursor;
	const auto& record  = copies.records[cursor % Weave::CopyRecords];
	const bool  matched = record.kind == WeaveCopyRecord::Decoded && done.path == 1 && done.dst == record.dst &&
	                     done.src == record.src && done.bytes == record.bytes;
	copies.runs[cursor % Weave::CopyRecords] = {.index  = cursor,
	                                            .result = matched ? WeaveCopyRun::Matched : WeaveCopyRun::Mismatched,
	                                            .clock_begin = order.copy_clock,
	                                            .clock_end   = Scout::WriteClock()};
	++cursor;
	copies.run.store(cursor, std::memory_order_release);
}

// At a parcel's open: whether every linear copy it read through (Result::copies_used) ran as the reader recorded it;
// and in `passed`, ascending, the write-clock windows (begin, end] of the matched copies listed before it that ran
// after its worker began: the scout's write ring holds their writes over the parcel's pages, which it read through the
// copies.
bool CommandProcessor::WeaveCopiesOf(const Weave::Parcel& parcel, std::vector<std::pair<uint64_t, uint64_t>>& passed) {
	passed.clear();
	auto* const copies = g_weave_copies;
	if (copies == nullptr) return true;
	const auto& result = *parcel.result;
	// (A record's run, while the ring still holds it.)
	const auto run_of = [&](uint64_t index) -> const WeaveCopyRun* {
		const auto& run = copies->runs[index % Weave::CopyRecords];
		return run.index == index ? &run : nullptr;
	};
	bool matched = true;
	for (const auto index: result.copies_used)
		if (const auto* run = run_of(index); run == nullptr || run->result != WeaveCopyRun::Matched) matched = false;
	// (The GPU thread runs the records in order, so their clocks ascend: from the last listed back to the first that
	// ran before the worker's write clock. Every listed record is final here: the GPU thread passed them before this
	// packet.)
	for (auto index = parcel.copies_end; index-- > 0;) {
		const auto* run = run_of(index);
		if (run == nullptr) break; // (older than the ring holds)
		if (run->result != WeaveCopyRun::Matched && run->result != WeaveCopyRun::Mismatched) continue; // (unrun)
		if (run->clock_end <= result.write_clock) break;
		if (run->result == WeaveCopyRun::Matched && run->clock_end > run->clock_begin)
			passed.emplace_back(run->clock_begin, run->clock_end);
	}
	std::reverse(passed.begin(), passed.end());
	return matched;
}

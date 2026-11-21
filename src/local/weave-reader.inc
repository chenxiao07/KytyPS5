// The weave's reader and a worker's walk of a parcel: CommandProcessor members off the GPU thread (weave.h). Included
// by graphicsRun.cpp under KYTY_LOCAL_VULKAN_RECORDING, before ProcessPm4.
//
// The reader (Kyty.WeaveRead) is fed each queue-0 graphics submission as GuestGpu::Enqueue feeds the scout. It copies
// the commands and walks the copy ahead of the GPU thread on a processor of its own, through the real register
// handlers, drawing nothing. This is the one second walk of a command buffer besides ProcessPm4's loop: it runs ahead
// of the GPU thread on a copy, so it can neither wait for the GPU nor draw. Each draw packet the table path's entries
// take is a table call of a parcel (a row of DRAW_INDEX_INDIRECT that differ only in the argument offset is one call,
// as TryNativeXprDraws takes it). A parcel ends at KYTY_WEAVE_CALLS calls, at the submission's end, and before any
// packet that is neither a register packet nor such a draw (an ender: only the GPU thread runs it). The reader reads on
// past an ender whose effect on the registers it knows (a wait, a label, a dispatch, a draw a handler makes, a register
// reset) and stops at one whose effect it doesn't know (a nested buffer, a rewind, a conditional, a predicate, the
// constant engine, an unknown packet, a register table on a page a read watcher guards). Stopped, at a gap in the feed,
// or told so by the GPU thread, it is out of step: it asks for the GPU thread's registers, which the GPU thread posts
// at its next draw packet no parcel covers (weave-order.inc), and reads on from there.
//
// A worker (Kyty.Weave1 on) walks a parcel on a processor of its own, from the registers the reader noted at its first
// draw, through the same handlers, and gives each draw to the executor's worker draw (weave-worker.inc) with the
// arguments the native entries decode (NativeXprDrawsArgs, NativeMultiDrawArgs, NativeDirectDrawArgs). It reads the
// reader's copy and names each packet by its guest address. The registers at each made draw go into the parcel's
// register log, from which the GPU thread skips a spliced part's packets.

namespace {

// The draw forms the table path's entries take (ProcessPm4: TryNativeXprDraws, TryNativeMultiDraw, TryNativeDirectDraw,
// TryNativeAutoDraw).
[[nodiscard]] constexpr bool WeaveDrawPacket(uint32_t opcode) {
	return opcode == Pm4::IT_DRAW_INDEX_INDIRECT || opcode == Pm4::IT_DRAW_INDEX_INDIRECT_MULTI ||
	       opcode == Pm4::IT_DRAW_INDEX_2 || opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 || opcode == Pm4::IT_DRAW_INDEX_AUTO;
}

// The start-instance user SGPR a draw packet of those forms sets before its shaders are translated, as ProcessPm4 sets
// it.
[[nodiscard]] int32_t WeaveStartInstanceSgpr(uint32_t opcode, const uint32_t* packet) {
	return opcode == Pm4::IT_DRAW_INDEX_INDIRECT || opcode == Pm4::IT_DRAW_INDEX_INDIRECT_MULTI
	           ? GsUserSgprAt(packet[3] & 0xffffu)
	           : -1;
}

enum class WeavePacket : uint8_t { Ender, State, Draw };

// What a packet is to a parcel (`packet` has at least two dwords): a draw the table path's entries take, a packet that
// changes nothing but the processor's registers, or an ender.
[[nodiscard]] WeavePacket WeavePacketKind(const uint32_t* packet, uint32_t length) {
	const uint32_t header = packet[0] & ~1u;
	switch ((header >> 8u) & 0xffu) {
		case Pm4::IT_DRAW_INDEX_INDIRECT: return header == 0xc0032500u ? WeavePacket::Draw : WeavePacket::Ender;
		case Pm4::IT_DRAW_INDEX_AUTO: return header == 0xc0012d00u ? WeavePacket::Draw : WeavePacket::Ender;
		case Pm4::IT_DRAW_INDEX_2: return header == 0xc0042700u ? WeavePacket::Draw : WeavePacket::Ender;
		case Pm4::IT_DRAW_INDEX_OFFSET_2: return header == 0xc0033500u ? WeavePacket::Draw : WeavePacket::Ender;
		// (A count read from guest memory is the handler's.)
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
			return header == 0xc0083800u && length >= 10 && ((packet[4] >> 30u) & 1u) == 0 ? WeavePacket::Draw
			                                                                               : WeavePacket::Ender;
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX:
		case Pm4::IT_SET_SH_REG_INDIRECT:
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
		case Pm4::IT_SET_BASE:
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_NUM_INSTANCES: return WeavePacket::State;
		case Pm4::IT_NOP:
			switch (KYTY_PM4_R(header)) {
				case Pm4::R_ZERO:
					// A marker (CpOpMarker) sets nothing or the user-data marker; a flip's marker is an ender.
					if ((packet[1] & 0xffff0000u) == 0x68750000u) {
						const uint32_t id = packet[1] & 0xfffu;
						return id == 0x0u || id == 0x4u || id == 0xdu ? WeavePacket::State : WeavePacket::Ender;
					}
					return WeavePacket::State;
				case Pm4::R_PUSH_MARKER:
				case Pm4::R_POP_MARKER: return WeavePacket::State;
				default: return WeavePacket::Ender;
			}
		default: return WeavePacket::Ender;
	}
}

// A NOP of a parcel whose handler changes something of the processor's: a marker (CpOpMarker). Its other NOPs change
// nothing and are not run.
[[nodiscard]] bool WeaveRunsNop(const uint32_t* packet) {
	return KYTY_PM4_R(packet[0]) == Pm4::R_ZERO && (packet[1] & 0xffff0000u) == 0x68750000u;
}

// A SET_*_REG_INDIRECT's register table: its address, and its bytes in `bytes` (0: not such a packet, or no entries).
[[nodiscard]] uint64_t WeaveTable(uint32_t opcode, const uint32_t* packet, uint32_t length, uint64_t& bytes) {
	bytes = 0;
	if ((opcode != Pm4::IT_SET_SH_REG_INDIRECT && opcode != Pm4::IT_SET_CONTEXT_REG_INDIRECT &&
	     opcode != Pm4::IT_SET_UCONFIG_REG_INDIRECT) ||
	    length != 5)
		return 0;
	bytes = uint64_t {packet[4] & 0x3fffu} * 8u;
	return (uint64_t {packet[1]} & 0xfffffffcu) | uint64_t {packet[2]} << 32u;
}

// The register log's image (Weave::RegisterLog): the context, the user configuration and the shader registers, each
// from a 64-byte boundary, then one block of the scalars WeaveSnapshot takes with them. Not the pushed context or the
// predicate: no packet of a parcel changes them (a context push or pop and a SET_PREDICATION end a parcel). The image's
// padding bytes are zero.
struct WeaveScalars {
	uint32_t marker = 0, index_type = 0, index_size = 0, num_instances = 0;
	uint64_t index_base = 0, draw_args = 0, dispatch_args = 0, zero = 0;
};
static_assert(sizeof(WeaveScalars) <= Weave::LogBlockBytes && std::is_trivially_copyable_v<WeaveScalars>);
static_assert(std::is_trivially_copyable_v<HW::Context> && std::is_trivially_copyable_v<HW::UserConfig> &&
              std::is_trivially_copyable_v<HW::Shader>);
[[nodiscard]] constexpr uint32_t WeaveLogPad(size_t size) {
	return static_cast<uint32_t>((size + Weave::LogBlockBytes - 1) & ~size_t {Weave::LogBlockBytes - 1});
}
constexpr uint32_t WeaveLogUcfg    = WeaveLogPad(sizeof(HW::Context));
constexpr uint32_t WeaveLogSh      = WeaveLogUcfg + WeaveLogPad(sizeof(HW::UserConfig));
constexpr uint32_t WeaveLogScalars = WeaveLogSh + WeaveLogPad(sizeof(HW::Shader));
constexpr uint32_t WeaveLogBytes   = WeaveLogScalars + Weave::LogBlockBytes;

// The reader's copies of submissions kept for reuse.
constexpr size_t WeavePoolCopies = 64;

// The command processor's side of the weave: made by the GPU thread as the weave starts, never freed while the game
// runs (its threads read it until StopAll has joined them).
struct WeaveCp {
	RenderExecutor*                                                  executor = nullptr;
	std::unique_ptr<CommandProcessor>                                reader;
	std::array<std::unique_ptr<CommandProcessor>, Weave::MaxWorkers> workers;
	// The submission the GPU thread is in (the scout's numbering).
	std::atomic<uint64_t> gpu_sequence {0};
	// The reader's syncs taken (a parcel's generation: those before it was cut), and the GPU thread's requests that it
	// go out of step.
	std::atomic<uint64_t> generation {0}, force_out {0};
	// The GPU thread's registers for the reader: 0 none asked, 1 asked (to be posted at a draw packet at or past where
	// the reader lost step), 2 posted. The reader writes its fields before 1, the GPU thread its own before 2.
	struct Sync {
		std::atomic<uint32_t> state {0};
		uint64_t              wanted_sequence = 0;
		uint32_t              wanted_offset   = 0;
		uint64_t              sequence        = 0;
		const uint32_t*       guest           = nullptr;
		uint32_t              dwords = 0, offset = 0;
		Weave::Registers      registers;
	} sync;
	// The reader's copies of submissions, each reused once no parcel or worker holds it.
	std::vector<std::shared_ptr<std::vector<uint32_t>>> pool;
};
WeaveCp* g_weave = nullptr;

// The reader's copy of a submission's commands (null: a page of them could not be read): one of its pool that no parcel
// or worker holds any more (the smallest that fits, else the smallest free one, grown), so the GPU thread's and the
// workers' releases free nothing; a new one joins the pool while it has room.
std::shared_ptr<const std::vector<uint32_t>> WeaveCopy(const uint32_t* commands, uint32_t dwords) {
	auto&                                  pool = g_weave->pool;
	std::shared_ptr<std::vector<uint32_t>> copy;
	size_t                                 fit = pool.size(), spare = pool.size();
	for (size_t i = 0; i < pool.size(); ++i) {
		if (pool[i].use_count() != 1) continue; // (a parcel or a worker holds it)
		const auto capacity = pool[i]->capacity();
		if (capacity >= dwords && (fit == pool.size() || capacity < pool[fit]->capacity())) fit = i;
		if (spare == pool.size() || capacity < pool[spare]->capacity()) spare = i;
	}
	// (Seen free: after its last holder's release, before the reader writes into it.)
	std::atomic_thread_fence(std::memory_order_acquire);
	if (fit != pool.size()) {
		copy = pool[fit];
	} else if (spare != pool.size()) {
		copy = pool[spare];
	} else {
		copy = std::make_shared<std::vector<uint32_t>>();
		if (pool.size() < WeavePoolCopies) pool.push_back(copy);
	}
	copy->resize(dwords);
	if (dwords != 0 && Common::GuardedCopy(copy->data(), commands, uint64_t {dwords} * 4u) != 0) return nullptr;
	return copy;
}

} // namespace

void CommandProcessor::WeaveFeed(uint64_t sequence, std::span<const uint32_t> commands, bool constant, bool reset) {
	// (A full feed drops it: the reader then finds a gap in the sequence numbers and goes out of step.)
	(void)Weave::PushFeed({.sequence = sequence,
	                       .commands = commands.data(),
	                       .dwords   = static_cast<uint32_t>(commands.size()),
	                       .constant = constant,
	                       .reset    = reset});
}

// Kyty.WeaveRead: the reader's loop (Weave::Bodies::read).
void CommandProcessor::WeaveRead() {
	auto&    weave   = *g_weave;
	auto&    reader  = *weave.reader;
	auto&    sync    = weave.sync;
	bool     in_step = false;
	uint64_t last    = 0; // the last submission read (to its end, or from a sync)
	uint64_t forced  = weave.force_out.load(std::memory_order_acquire);
	// Where it lost step: the GPU thread's registers are posted at or past it.
	uint64_t   want_sequence = 0;
	uint32_t   want_offset = 0, stopped_at = 0;
	const auto lose = [&](uint64_t sequence, uint32_t offset) {
		in_step       = false;
		want_sequence = sequence;
		want_offset   = offset;
	};
	while (!Weave::Stopping()) {
		// (Read before it looks for work: a feed, a post or a request after this moves it, and the wait returns.)
		const auto wakes = Weave::ReaderWakes();
		if (const auto force = weave.force_out.load(std::memory_order_acquire); force != forced) {
			forced = force;
			lose(weave.gpu_sequence.load(std::memory_order_acquire), 0);
		}
		Weave::Feed feed;
		if (!in_step) {
			// Out of step: the GPU thread's registers once it has posted them, and the read on from there. Meanwhile
			// the submissions it is past are dropped, and the others wait in the feed for the read after the sync.
			const auto state = sync.state.load(std::memory_order_acquire);
			if (state == 2) {
				reader.WeaveRestore(sync.registers);
				const auto  sequence = sync.sequence;
				const auto* guest    = sync.guest;
				const auto  dwords = sync.dwords, offset = sync.offset;
				weave.generation.fetch_add(1, std::memory_order_release); // (the parcels cut from here on: after it)
				sync.state.store(0, std::memory_order_release);
				last = std::max(last, sequence);
				// (The GPU thread is in this submission: its commands stay the game's until the GPU thread is past it.
				// Asked again once the copy is made, its reads ordered first: past it by then, the copy may hold the
				// game's next commands, and it is dropped.)
				if (weave.gpu_sequence.load(std::memory_order_acquire) == sequence) {
					const auto copy = WeaveCopy(guest, dwords);
					std::atomic_thread_fence(std::memory_order_acquire);
					if (copy != nullptr && weave.gpu_sequence.load(std::memory_order_acquire) == sequence) {
						in_step = reader.WeaveReadFrom(sequence, guest, copy, offset, stopped_at);
						if (!in_step) lose(sequence, stopped_at);
					}
				}
				continue;
			}
			if (state == 0) {
				sync.wanted_sequence = want_sequence;
				sync.wanted_offset   = want_offset;
				sync.state.store(1, std::memory_order_release);
			}
			if (Weave::PeekFeed(feed) && feed.sequence < weave.gpu_sequence.load(std::memory_order_acquire)) {
				Weave::TakeFeed();
				continue;
			}
			Weave::WaitReader(wakes);
			continue;
		}
		if (!Weave::PeekFeed(feed)) {
			Weave::WaitReader(wakes);
			continue;
		}
		Weave::TakeFeed();
		if (feed.sequence <= last) continue; // (read already, from a sync)
		if (feed.sequence != last + 1) {
			lose(feed.sequence, 0); // (a feed the full ring dropped)
			continue;
		}
		// (The GPU thread resets its processor before the submission, GuestGpu::Process: a reset of its own or a flip
		// preparation's. The reader does the same and reads on.)
		if (feed.reset) reader.Reset();
		// (A submission the GPU thread is past: its commands may be the game's again, and its registers are lost.)
		if (weave.gpu_sequence.load(std::memory_order_acquire) > feed.sequence) {
			lose(feed.sequence + 1, 0);
			continue;
		}
		const auto copy = WeaveCopy(feed.commands, feed.dwords);
		if (copy == nullptr) {
			lose(feed.sequence, 0);
			continue;
		}
		// (Asked again once the copy is made, its reads ordered first: the GPU thread past it by then, the copy may
		// hold the game's next commands.)
		std::atomic_thread_fence(std::memory_order_acquire);
		if (weave.gpu_sequence.load(std::memory_order_acquire) > feed.sequence) {
			lose(feed.sequence + 1, 0);
			continue;
		}
		last    = feed.sequence;
		in_step = reader.WeaveReadFrom(feed.sequence, feed.commands, copy, 0, stopped_at);
		if (!in_step) lose(feed.sequence, stopped_at);
	}
}

// The reader's processor: `copy` (the bytes of `guest`'s commands) read from dword `from`, its parcels published. True:
// read to its end, the registers there the stream's; false: stopped at dword `stopped_at`.
bool CommandProcessor::WeaveReadFrom(uint64_t sequence, const uint32_t* guest,
                                     const std::shared_ptr<const std::vector<uint32_t>>& copy, uint32_t from,
                                     uint32_t& stopped_at) {
	const auto* const base      = copy->data();
	const auto        total     = static_cast<uint32_t>(copy->size());
	const uint32_t    max_calls = kyty_local_weave_calls.load(std::memory_order_relaxed);
	Weave::Parcel*    parcel    = nullptr;
	// A row of DRAW_INDEX_INDIRECT the native entry takes as one call: its packets so far, the dword after its last,
	// and the words its packets share.
	uint32_t                row = 0, row_next = 0;
	std::array<uint32_t, 3> row_words {};
	// (A parcel opened at a draw packet: the slot of the next one, its fields the reader's to fill.)
	const auto open = [&](uint32_t at) {
		parcel = Weave::BeginParcel();
		if (parcel == nullptr) return false; // (paused, or stopping)
		parcel->generation = g_weave->generation.load(std::memory_order_relaxed);
		parcel->sequence   = sequence;
		parcel->guest      = guest;
		parcel->first      = guest + at;
		parcel->end        = guest + at;
		parcel->commands   = copy;
		parcel->calls = parcel->draws = 0;
		parcel->cuts.clear();
		parcel->progress.store(0, std::memory_order_relaxed);
		parcel->dropped.store(false, std::memory_order_relaxed);
		return true;
	};
	// (A parcel of fewer than Weave::ShortestParcel draws is left to the GPU thread, its slot the next one's.)
	const auto close = [&] {
		if (parcel == nullptr) return;
		auto& cut = *std::exchange(parcel, nullptr);
		if (cut.draws < Weave::ShortestParcel)
			Weave::AbandonParcel(cut);
		else
			Weave::PublishParcel(cut);
	};
	const auto run = [&](uint32_t at, uint32_t header, uint32_t length) {
		return g_cp_op_func[(header >> 8u) & 0xffu](*this, header & ~1u, base + at + 1, total - at, total) + 1 ==
		       length;
	};
	bool     stop = false;
	uint32_t at   = from;
	while (at < total) {
		if (Weave::Stopping()) {
			stop = true;
			break;
		}
		const uint32_t header = base[at];
		if (header == 0x80000000u) {
			++at;
			continue;
		}
		const uint32_t length = KYTY_PM4_LEN(header);
		if (total - at < 2 || (header >> 30u) != 3u || length > total - at) {
			stop = true; // (malformed)
			break;
		}
		const uint32_t opcode = (header >> 8u) & 0xffu;
		const auto*    packet = base + at;
		if ((header & 1u) != 0 && m_predicate_skip) {
			stop = true; // (the GPU thread skips it)
			break;
		}
		const auto kind = WeavePacketKind(packet, length);
		if (kind == WeavePacket::Draw) {
			const bool indirect  = header == 0xc0032500u;
			const bool continues = indirect && row != 0 && row < NativeDrawArgs::MaxDraws && at == row_next &&
			                       packet[2] == row_words[0] && packet[3] == row_words[1] && packet[4] == row_words[2];
			if (parcel != nullptr && !continues && parcel->calls >= max_calls) close();
			if (parcel == nullptr && !open(at)) {
				stop = true;
				break;
			}
			// (At its first draw, the registers its worker starts from, before the draw sets its start-instance SGPR:
			// as the GPU thread has them where it opens the parcel. Its stretches' starts after the first, by draw
			// packets.)
			if (parcel->draws == 0)
				WeaveSnapshot(parcel->start);
			else if (parcel->draws % Weave::StretchDraws == 0)
				parcel->cuts.push_back(at);
			if (continues) {
				++row;
			} else {
				++parcel->calls;
				row = indirect ? 1u : 0u;
				if (indirect) row_words = {packet[2], packet[3], packet[4]};
			}
			row_next = at + length;
			m_sh_ctx.SetStartInstanceUserSgpr(WeaveStartInstanceSgpr(opcode, packet));
			++parcel->draws;
			at += length;
			parcel->end = guest + at;
			continue;
		}
		row = 0;
		if (kind == WeavePacket::State) {
			// (A register table on a page a read watcher guards, or none: the GPU thread brings the GPU's bytes back
			// before it reads it, SyncCommandMemory, which nothing off the GPU thread does. The read stops there, as
			// the scout's.)
			uint64_t   bytes = 0;
			const auto table = WeaveTable(opcode, packet, length, bytes);
			if (bytes != 0 && (table == 0 || GetGpuResources().HasReadWatchers(table, bytes))) {
				stop = true;
				break;
			}
			if ((opcode != Pm4::IT_NOP || WeaveRunsNop(packet)) && !run(at, header, length)) {
				stop = true;
				break;
			}
			at += length;
			continue;
		}
		// An ender: the parcel ends before it, and only the GPU thread runs it.
		close();
		switch (opcode) {
			case Pm4::IT_REWIND:
			case Pm4::IT_COND_EXEC:
			case Pm4::IT_INDIRECT_BUFFER:
			case Pm4::IT_SET_PREDICATION:
			case Pm4::IT_WRITE_CONST_RAM:
			case Pm4::IT_DUMP_CONST_RAM:
			case Pm4::IT_INCREMENT_CE_COUNTER:
			case Pm4::IT_INCREMENT_DE_COUNTER:
			case Pm4::IT_WAIT_ON_CE_COUNTER:
			case Pm4::IT_WAIT_ON_DE_COUNTER_DIFF: stop = true; break;
			case Pm4::IT_CLEAR_STATE: stop = !run(at, header, length); break;
			case Pm4::IT_DISPATCH_DIRECT:
				// (Its handler sets the compute wave size, DispatchDirect.)
				if (length == 5)
					m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(packet[4]));
				else
					stop = true;
				break;
			case Pm4::IT_DISPATCH_INDIRECT:
				if (header == 0xc0011600u)
					m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(packet[2]));
				else if (header == 0xc0021600u)
					m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(packet[3]));
				else
					stop = true;
				break;
			case Pm4::IT_NOP:
				if (KYTY_PM4_R(header) == Pm4::R_CONTEXT_STATE)
					stop = !run(at, header, length);
				else if (KYTY_PM4_R(header) == Pm4::R_DISPATCH_RESET)
					Reset(); // (every register to its default, as CpOpDispatchReset)
				break;
			default:
				if (WeaveDrawPacket(opcode) || Scout::IsDraw(opcode) || opcode == Pm4::IT_DISPATCH_DRAW_PREAMBLE) {
					// (A draw a handler makes: the start-instance SGPR set as ProcessPm4 sets it, the other registers
					// kept. A handler that changes them shows at the next parcel's start, which the GPU thread
					// compares.)
					const bool indirect = opcode == Pm4::IT_DRAW_INDIRECT || opcode == Pm4::IT_DRAW_INDEX_INDIRECT ||
					                      opcode == Pm4::IT_DRAW_INDIRECT_MULTI ||
					                      opcode == Pm4::IT_DRAW_INDEX_INDIRECT_MULTI;
					if (indirect && length < 4)
						stop = true;
					else
						m_sh_ctx.SetStartInstanceUserSgpr(indirect ? GsUserSgprAt(packet[3] & 0xffffu) : -1);
				} else if (g_cp_op_func[opcode] == nullptr) {
					stop = true;
				}
				break;
		}
		if (stop) break;
		at += length;
	}
	close();
	stopped_at = at;
	return !stop;
}

// A worker (Weave::Bodies::make): parcel `parcel` walked on worker `worker`'s processor.
void CommandProcessor::WeaveMake(uint32_t worker, Weave::Parcel& parcel) {
	g_weave->workers[worker]->WeaveWalk(worker, parcel);
}

// What a parcel holds besides its slot, let go (Weave::Bodies::release, on whichever thread frees the slot): its slab
// with no tick (none once the GPU thread released it after its splices) and its commands' copy (the reader's pool holds
// that: nothing is freed here).
void CommandProcessor::WeaveRelease(Weave::Parcel& parcel) {
	g_weave->executor->WeaveRelease(parcel, false);
	parcel.commands.reset();
}

// On worker `worker`'s processor: parcel `parcel`'s packets from its first draw to its end, from the registers the
// reader noted there, read from the reader's copy and known to the executor by their guest addresses. The walk stops at
// its next draw once the GPU thread has dropped the parcel (the GPU thread makes the rest), and spoils the parcel at a
// packet it cannot take (a register table it cannot read in a guarded copy, a handler that takes other dwords).
void CommandProcessor::WeaveWalk(uint32_t worker, Weave::Parcel& parcel) {
	auto&             executor = *g_weave->executor;
	const auto* const copy     = parcel.commands->data();
	const auto        total    = static_cast<uint32_t>(parcel.commands->size());
	const auto        identity = [&](const uint32_t* read) { return parcel.guest + (read - copy); };
	const auto        first    = static_cast<uint32_t>(parcel.first - parcel.guest);
	const auto        end      = static_cast<uint32_t>(parcel.end - parcel.guest);
	WeaveRestore(parcel.start);
	// (Its registers were replaced: nothing the observer holds is known, and its first draw is not clean.)
	DrawStateObserver::Invalidate();
	if (!executor.WeaveParcelBegin(worker, parcel, m_ctx, m_ucfg, m_sh_ctx)) {
		if (parcel.result) parcel.result->spoiled = true; // (nothing of it made: never opened)
		return;
	}
	// The register log, by made draw: the image's 64-byte blocks that changed since the last made draw's end (`at_last`
	// holds the image there), the other packets before it by opcode, its own draw packets (the executor makes each
	// block's next write at the parcel's end). None while the write guard is armed (KYTY_WRITE_GUARD=1): the GPU thread
	// reads the tables there through views of its own, and passes a spliced part's packets one by one.
	auto& log = parcel.result->log;
	log.on    = kyty_local_write_guard_mode.load(std::memory_order_relaxed) != 1;
	thread_local std::vector<uint8_t> at_last;
	if (log.on) {
		at_last.resize(WeaveLogBytes);
		WeaveLogTake(at_last.data());
	}
	size_t   cut         = 0;
	bool     stretch     = true; // (the next draw opens a stretch)
	uint32_t counts_from = 0;    // (the next made draw's counts begin)
	for (uint32_t at = first; at < end;) {
		// (A stretch's edge: needs are grouped from there on, and a new strand begins; its first draw is not clean.)
		while (cut < parcel.cuts.size() && at >= parcel.cuts[cut]) {
			++cut;
			stretch = true;
			DrawStateObserver::Invalidate();
		}
		const uint32_t header = copy[at];
		if (header == 0x80000000u) {
			++at;
			continue;
		}
		const uint32_t length = KYTY_PM4_LEN(header);
		const uint32_t opcode = (header >> 8u) & 0xffu;
		const auto*    packet = copy + at;
		if (WeaveDrawPacket(opcode)) {
			// (As ProcessPm4, before the draw's entry.)
			const int32_t start_sgpr = m_sh_ctx.GetVs().start_instance_user_sgpr;
			m_sh_ctx.SetStartInstanceUserSgpr(WeaveStartInstanceSgpr(opcode, packet));
			DrawStateObserver::ObservePacket(opcode, packet, length, false);
			// (The dwords reached, for the GPU thread's wait at the open.)
			parcel.progress.store(at - first, std::memory_order_relaxed);
			if (parcel.dropped.load(std::memory_order_relaxed)) {
				executor.WeaveSpoil();
				break;
			}
			if (stretch) {
				executor.WeaveStretch(identity(packet));
				stretch = false;
			}
			uint32_t       skip  = length;
			const uint32_t taken = WeaveDrawAt({packet, end - at}, identity(packet), skip);
			if (taken == 0) {
				// (A hole: the GPU thread makes it, as its entry takes it; the next draw starts as a strand's first.)
				DrawStateObserver::Invalidate();
				at += skip;
				continue;
			}
			// (A strand begun at this draw: its start image, the registers as they were before the draw packet set its
			// start-instance SGPR, as the GPU thread has them where it reaches the packet and compares them.)
			if (auto* start = executor.WeaveStrandStart(); start != nullptr) {
				WeaveSnapshot(*start);
				start->sh.SetStartInstanceUserSgpr(start_sgpr);
			}
			if (log.on) {
				WeaveLogDelta(at_last.data(), parcel);
				const bool row  = header == 0xc0032500u;
				auto&      draw = log.draws.emplace_back();
				draw.blocks_end = static_cast<uint32_t>(log.block_ids.size());
				draw.counts_end = static_cast<uint32_t>(log.counts.size());
				draw.packets    = row ? taken / 5u : 1u;
				draw.draws      = header == 0xc0083800u ? packet[5] : draw.packets;
				draw.opcode     = opcode;
				draw.last       = identity(packet + (row ? taken - 5u : 0u));
				counts_from     = draw.counts_end;
				if (log.draws.size() != executor.WeaveMadeCount()) log.on = false; // (one entry a made draw, or no log)
			}
			at += taken;
			continue;
		}
		if (log.on) {
			auto k = counts_from;
			while (k < log.counts.size() && log.counts[k].opcode != opcode)
				++k;
			if (k == log.counts.size()) log.counts.push_back({.opcode = opcode});
			++log.counts[k].count;
		}
		// (A register table read once in a guarded copy for the observer; the handler reads it in its own, GuardTable.
		// One on a page a read watcher guards, or none, or one that faults: its bytes may be the GPU's, which the GPU
		// thread brings back first, SyncCommandMemory. One with an entry past its bank: GuardTable applies none of it
		// off the GPU thread, where the GPU thread applies the rest. The parcel is spoiled there.)
		uint64_t   bytes = 0;
		const auto table = WeaveTable(opcode, packet, length, bytes);
		if (bytes != 0) {
			thread_local std::vector<uint32_t> words;
			words.resize(bytes / 4u);
			const void*    backing = table != 0 ? WriteGuard::GuardedBacking(table, bytes) : nullptr;
			const auto     pairs   = static_cast<uint32_t>(bytes / 8u);
			const uint32_t bank    = opcode == Pm4::IT_SET_CONTEXT_REG_INDIRECT ? 0u
			                         : opcode == Pm4::IT_SET_SH_REG_INDIRECT    ? 1u
			                                                                    : 2u;
			if (table == 0 || GetGpuResources().HasReadWatchers(table, bytes) ||
			    Common::GuardedCopy(words.data(), backing != nullptr ? backing : reinterpret_cast<const void*>(table),
			                        bytes) != 0 ||
			    RegisterTableFirstPastBank(words.data(), pairs, bank) != pairs) {
				executor.WeaveSpoil();
				break;
			}
			const std::span<const uint32_t> read {words};
			DrawStateObserver::ObservePacket(opcode, packet, length, false, &read);
		} else {
			DrawStateObserver::ObservePacket(opcode, packet, length, false);
		}
		if ((opcode != Pm4::IT_NOP || WeaveRunsNop(packet)) &&
		    g_cp_op_func[opcode](*this, header & ~1u, packet + 1, total - at, total) + 1 != length) {
			executor.WeaveSpoil();
			break;
		}
		// (Its page read-protected since the copy above: GuardTable's own copy may have faulted, or on Linux been
		// refused, and applied nothing.)
		if (bytes != 0 && GetGpuResources().HasReadWatchers(table, bytes)) {
			executor.WeaveSpoil();
			break;
		}
		at += length;
	}
	executor.WeaveParcelEnd();
}

// A draw packet of a parcel on a worker's processor (`identity`: its guest address), given to the worker's table draw
// with the arguments the native entries decode. A draw those leave to a handler, DRAW_INDEX_AUTO (its entry changes the
// registers around the draw: the interface's scopes) and every draw into the interface's scaled target
// (HudAnchor::Scaled reads shader hashes from a table of the GPU thread's) are holes. The dwords the draw took, or 0:
// a hole, `skip` the dwords the GPU thread's entry takes of it.
uint32_t CommandProcessor::WeaveDrawAt(std::span<const uint32_t> packets, const uint32_t* identity, uint32_t& skip) {
	auto&          executor = *g_weave->executor;
	const uint32_t header   = packets[0];
	NativeDrawArgs args;
	const bool     decoded = header == 0xc0032500u                            ? NativeXprDrawsArgs(packets, args)
	                         : header == 0xc0083800u                          ? NativeMultiDrawArgs(packets, args)
	                         : header == 0xc0042700u || header == 0xc0033500u ? NativeDirectDrawArgs(packets, args)
	                                                                          : false;
	if (decoded) skip = args.dwords;
	if (!decoded || (HudAnchor::OutputScale() > 0.0f && HudAnchor::OutputTarget(m_ctx))) {
		executor.WeaveHole(Weave::Miss::Packet);
		return 0;
	}
	const auto commands =
	    args.is_direct ? std::span<const uint64_t> {} : std::span<const uint64_t> {args.commands.data(), args.count};
	return executor.WeaveDraw(identity, identity + args.dwords, DrawStateObserver::g_shadow.last_clean, commands,
	                          args.index_base, args.index_bytes, args.index_type,
	                          args.is_direct ? &args.direct : nullptr)
	           ? args.dwords
	           : 0u;
}

// The register log's image of the registers now, whole (padding zero).
void CommandProcessor::WeaveLogTake(uint8_t* image) const {
	const WeaveScalars scalars {.marker        = static_cast<uint32_t>(m_user_data_marker),
	                            .index_type    = m_index_type_and_size,
	                            .index_size    = m_index_buffer_size,
	                            .num_instances = m_num_instances,
	                            .index_base    = m_index_base_addr,
	                            .draw_args     = m_draw_indirect_args_base_addr,
	                            .dispatch_args = m_dispatch_indirect_args_base_addr};
	std::memset(image, 0, WeaveLogBytes);
	std::memcpy(image, &m_ctx, sizeof(m_ctx));
	std::memcpy(image + WeaveLogUcfg, &m_ucfg, sizeof(m_ucfg));
	std::memcpy(image + WeaveLogSh, &m_sh_ctx, sizeof(m_sh_ctx));
	std::memcpy(image + WeaveLogScalars, &scalars, sizeof(scalars));
}

// The blocks of the registers now that differ from `at_last` (the image at the last made draw's end), appended to the
// parcel's log with their values, `at_last` brought up to date.
void CommandProcessor::WeaveLogDelta(uint8_t* at_last, Weave::Parcel& parcel) const {
	constexpr uint32_t block = Weave::LogBlockBytes;
	auto&              log   = parcel.result->log;
	// (One object of the image, `size` bytes at `at`, block by block.)
	const auto part = [&](const void* live, uint32_t at, uint32_t size) {
		const auto* from = static_cast<const uint8_t*>(live);
		for (uint32_t offset = 0; offset < size; offset += block) {
			auto*      to = at_last + at + offset;
			const auto n  = std::min(block, size - offset);
			if (std::memcmp(to, from + offset, n) == 0) continue;
			std::memcpy(to, from + offset, n);
			log.block_ids.push_back((at + offset) / block);
			log.block_bytes.insert(log.block_bytes.end(), to, to + block);
		}
	};
	const WeaveScalars scalars {.marker        = static_cast<uint32_t>(m_user_data_marker),
	                            .index_type    = m_index_type_and_size,
	                            .index_size    = m_index_buffer_size,
	                            .num_instances = m_num_instances,
	                            .index_base    = m_index_base_addr,
	                            .draw_args     = m_draw_indirect_args_base_addr,
	                            .dispatch_args = m_dispatch_indirect_args_base_addr};
	part(&m_ctx, 0, sizeof(m_ctx));
	part(&m_ucfg, WeaveLogUcfg, sizeof(m_ucfg));
	part(&m_sh_ctx, WeaveLogSh, sizeof(m_sh_ctx));
	part(&scalars, WeaveLogScalars, sizeof(scalars));
}

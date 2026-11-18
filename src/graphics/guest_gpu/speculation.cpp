#include "graphics/guest_gpu/speculation.h"

#include "common/assert.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/bdaDirtyRegions.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h" // (spec_depth_layers2: htile_load_decision)
#include "kernel/memory.h" // (GuardedCopy: IsUniqueGuestBackingRangeAcrossMappings)
#include "async-upload.h"
#include "draw-state-observer.h"
#include "frame-gen.h"
#include "local-platform.h"
#include "mode-toggle.h"
#include "vulkan-recording.h"
#include "write-guard.h"
#include <xxhash.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

extern "C" {
volatile std::atomic_uint32_t kyty_local_speculate_mode {0};
volatile std::atomic_uint32_t kyty_local_speculate_threads {4};
// Translations going on and results not finished (Translate to Finish). While there are any, a speculation's executor
// may hold what it read of the owners' catalogs (pairs, draw states, image sets) and a result's commands their
// pipelines: the catalog sweeps (NativeXprSweepStates, TableSweepPairs) and EvictIdlePipelines wait for none.
volatile std::atomic_uint32_t kyty_local_speculations_live {0};
// EvictIdlePipelines is destroying pipeline handles: no translation begins until it is done.
volatile std::atomic_uint32_t kyty_local_pipelines_evicting {0};
}

namespace Libs::Graphics {

bool CommitGuardedWrite(void* dst, const void* src, uint32_t size); // (graphicsRun.cpp: KYTY_WRITE_GUARD_SPEC)
bool CommitGuardedCopyWrite(void* dst, const void* src, uint32_t size); // (graphicsRun.cpp: GuardedCopy)
// (after_guard_batch)
size_t CommitGuardedWriteRun(const uint64_t* addresses, const uint64_t* values, const uint32_t* sizes, size_t count);

// KYTY_SPECULATE_FLUSH_EVERY (live speculate_flush_every N; 8 by default under KYTY_SPECULATE=4, else 0; 0 and 1:
// every commit): a commit's held segment buffers go to the GPU at its end only every Nth commit; the rest wait for the
// next flush (a label's, a guest submission's end, a readback's, a chunk wait's). The GPU thread flushes them before
// it waits for a speculation's thread (which may wait for their tick), and when a commit blocks at a wait hole.
static uint32_t FlushEvery() {
	static auto& every = LiveSwitch::Find("speculate_flush_every", ModeToggle::Default("speculate_flush_every"));
	return static_cast<uint32_t>(std::max<int64_t>(every.load(std::memory_order_relaxed), 0));
}

uint64_t Speculation::NowNs() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// The packets a refusal makes a hole of: draws, dispatches, end-of-pipe events and waits (what they change besides
// their work is checked when the commit translates them: Speculation::RunHole; a wait not satisfied then blocks it).
static bool IsHole(uint32_t header) {
	switch ((header >> 8u) & 0xffu) {
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_EVENT_WRITE_EOP:
		case Pm4::IT_EVENT_WRITE_EOS:
		case Pm4::IT_RELEASE_MEM:
		case Pm4::IT_WAIT_REG_MEM:
		case Pm4::IT_WAIT_REG_MEM_64: return true;
		// (A wait for a display buffer's flip: the commit waits there before the rendering after it. A super resolution
		// dispatch runs its upscaler there.)
		case Pm4::IT_NOP:
			return KYTY_PM4_R(header) == Pm4::R_RELEASE_MEM || KYTY_PM4_R(header) == Pm4::R_WAIT_FLIP_DONE ||
			       KYTY_PM4_R(header) == Pm4::R_FLIP || KYTY_PM4_R(header) == Pm4::R_SUPER_RESOLUTION;
		case Pm4::IT_GET_LOD_STATS: return true;
		default: return false;
	}
}

// A chunk's translation stopped where the next chunk begins (Result::stop).
static constexpr char ChunkEnd[] = "chunk end";
// A result that ends before a packet whose command words lie on a page a read watcher guards (ProcessPm4Spec).
static constexpr char CommandWordsWatched[] = "command words watched";

// "Work wrote what it read", by the write's kind and what logged it (null: not found).
static const char* WroteWhatItRead(const Spec::EffectsLog::Entry* entry) {
	static constexpr std::array<std::array<const char*, 3>, 2> names {
	    {{"commit: GPU work wrote what it read (normal)", "commit: GPU work wrote what it read (hole)",
	      "commit: GPU work wrote what it read (commit)"},
	     {"commit: a processor write wrote what it read (normal)", "commit: a processor write wrote what it read (hole)",
	      "commit: a processor write wrote what it read (commit)"}}};
	if (entry == nullptr) return "commit: work wrote what it read";
	return names[entry->kind == Spec::EffectsLog::Kind::GpuWrite ? 0 : 1][static_cast<size_t>(entry->source)];
}

Speculation::Translator::Translator(RenderContext& renderer, uint32_t number, int interrupt_id)
    : index(number), draw(std::make_unique<RenderExecutor>(renderer)), compute(std::make_unique<RenderExecutor>(renderer)),
      executors {draw.get(), compute.get()}, cp(std::make_unique<CommandProcessor>(renderer, interrupt_id)),
      table_upload(std::make_unique<StreamBuffer>(renderer.GetGraphics(), renderer.GetCommandScheduler(), MemoryUsage::Upload,
                                                  uint64_t {64} << 20u,
                                                  AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress)),
      shader_upload(std::make_unique<StreamBuffer>(renderer.GetGraphics(), renderer.GetCommandScheduler(),
                                                   MemoryUsage::Upload, uint64_t {16} << 20u)) {
	draw->ReadCatalogOf(renderer.DefaultRenderExecutor(), 1 + number);
	compute->ReadCatalogOf(renderer.DefaultComputeRenderExecutor(), 1 + number);
}

Speculation::Speculation(RenderContext& renderer, int interrupt_id, uint32_t base, uint32_t threads, bool pass_thread)
    : m_renderer(renderer), m_chain(std::make_unique<CommandProcessor>(renderer, interrupt_id)),
      m_state(std::make_unique<CommandProcessor>(renderer, interrupt_id)), m_interrupt_id(interrupt_id) {
	m_state->SetStateOnly(true);
	if (threads == 0)
		threads = kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4
		              ? kyty_local_speculate_threads.load(std::memory_order_relaxed)
		              : 1u;
	// (Packet slots and executor roles: Spec::MaxPacketSlots, RenderExecutor::Roles less the GPU thread's.)
	threads = std::clamp<uint32_t>(threads, 1u, static_cast<uint32_t>(Spec::MaxPacketSlots) - std::min<uint32_t>(base, Spec::MaxPacketSlots - 1));
	for (uint32_t i = 0; i < threads; ++i) m_translators.push_back(std::make_unique<Translator>(renderer, base + i, interrupt_id));
	// KYTY_SPEC_PASS_THREAD: the state pass's own translator, last (the caller left it a packet slot).
	if (pass_thread && base + threads < Spec::MaxPacketSlots) {
		m_translators.push_back(std::make_unique<Translator>(renderer, base + threads, interrupt_id));
		m_translators.back()->pass_only = true;
		m_pass_thread                   = true;
	}
}

void Speculation::Stop() {
	{
		std::lock_guard lock(m_mutex);
		m_quit = true;
	}
	m_ahead_wake.notify_all();
	for (auto& translator: m_translators)
		if (translator->thread.joinable()) translator->thread.join();
	if (m_blocked) Finish(std::move(m_blocked), 0);
	for (;;) {
		std::unique_ptr<Result> result;
		{
			std::lock_guard lock(m_mutex);
			if (m_ahead.empty()) break;
			result = std::move(m_ahead.front());
			m_ahead.pop_front();
		}
		Finish(std::move(result), 0);
	}
}

Speculation::~Speculation() {
	{
		std::lock_guard lock(m_mutex);
		m_quit = true;
	}
	m_ahead_wake.notify_all();
	for (auto& translator: m_translators)
		if (translator->thread.joinable()) translator->thread.join();
	auto& scheduler = m_renderer.GetCommandScheduler();
	if (m_blocked) scheduler.DestroyRecorder(std::move(m_blocked->recorder));
	for (auto& result: m_ahead) scheduler.DestroyRecorder(std::move(result->recorder));
	for (auto& translator: m_translators)
		for (auto& result: translator->free) scheduler.DestroyRecorder(std::move(result->recorder));
}

Speculation::Epochs Speculation::CurrentEpochs() const {
	auto& resources = m_renderer.GetGpuResources();
	auto& textures  = m_renderer.GetTextureCache();
	return {m_renderer.GetBufferCache().RegistrationEpoch(), textures.ResolutionEpoch(), resources.MappingEpoch()};
}

Speculation::Outcome Speculation::TranslateAndCommit(CommandProcessor& cp, uint64_t serial,
                                                     std::span<const uint32_t> commands, Pm4Execution& execution) {
	auto result    = Translate(*m_translators[0], cp, commands, execution);
	result->serial = serial;
	return Commit(std::move(result), cp, execution);
}

Speculation::Outcome Speculation::Resume(CommandProcessor& cp, Pm4Execution& execution) {
	return Continue(std::move(m_blocked), cp, execution);
}

void Speculation::Push(const Job& job) {
	{
		std::lock_guard lock(m_mutex);
		if (!m_translators[0]->thread.joinable())
			for (auto& translator: m_translators)
				translator->thread = std::thread([this, worker = translator.get()] { AheadRun(*worker); });
		m_jobs.push_back({job});
		m_jobs.back().pushed_ns = NowNs();
	}
	m_ahead_wake.notify_all();
}

void Speculation::Done(uint64_t serial, const CommandProcessor& cp) {
	std::lock_guard lock(m_mutex);
	while (!m_jobs.empty() && m_jobs.front().job.serial <= serial) {
		// (Its records' writes are made: what read through them reads memory.)
		for (auto& chunk: m_jobs.front().chunks)
			if (auto record = std::move(chunk.record)) {
				Unlink(record.get(), nullptr);
				Retire(std::move(record));
			}
		m_jobs.pop_front();
	}
	m_done_serial = serial;
	if (!m_done_state) m_done_state = std::make_unique<CommandProcessor>(m_renderer, m_interrupt_id);
	m_done_state->CopyStateFrom(cp, false);
	m_ahead_wake.notify_all();
}

void Speculation::ResyncStatePass() {
	if (m_state_busy || m_done_serial == 0) return;
	// (It cannot follow, and the GPU thread did that submission; or it is behind the GPU thread.)
	if (m_broken ? m_done_serial >= m_broken_at : m_state_serial < m_done_serial) {
		m_state->CopyStateFrom(*m_done_state, false);
		m_state_serial = m_done_serial;
		m_broken       = false;
	}
}

bool Speculation::StatePass(const Job& job, uint64_t which) {
	if (job.reset) m_state->Reset();
	if (job.kind != Job::Kind::Graphics) return job.kind == Job::Kind::Flip; // (a flip preparation leaves the state as it is)
	m_state->ResetDeCe();
	m_state->SetFlip({});
	m_state->SetChunking(ChunkWork);
	Pm4Execution execution;
	for (;;) {
		// A chunk from here: where it starts, and its record, which reads through the one before while that is not done.
		Chunk chunk;
		chunk.start = std::make_unique<CommandProcessor>(m_renderer, m_interrupt_id);
		chunk.start->CopyStateFrom(*m_state, false);
		chunk.at = execution;
		{
			std::lock_guard lock(m_mutex);
			chunk.record = TakeRecord();
			if (const auto* entry = EntryOf(job.serial); entry != nullptr && !entry->chunks.empty())
				chunk.record->previous = entry->chunks.back().record.get();
			else if (const auto* before = EntryOf(job.serial - 1); before != nullptr && !before->chunks.empty())
				chunk.record->previous = before->chunks.back().record.get();
			m_pass_record = chunk.record.get();
		}
		chunk.record->Inherit(); // (a copy's source the records before it write is not read: Spec::State::Resolve)
		Spec::t_record       = chunk.record.get();
		const auto processed = m_state->Process(execution, job.commands);
		Spec::t_record       = nullptr;
		(void)m_state->TakeWatchedStop(); // (a stop before a packet on a page without access: the pass ends, not followed)
		const bool boundary  = processed != Pm4ProcessResult::Complete && m_state->TakeBoundary();
		chunk.end            = execution;
		chunk.last           = !boundary;
		std::lock_guard lock(m_mutex);
		m_pass_record = nullptr;
		// (Not followed: what is not passed is translated the normal way.)
		auto*      entry    = EntryOf(job.serial);
		const bool followed = processed == Pm4ProcessResult::Complete || boundary;
		if (entry != nullptr && which == m_pass && followed) {
			entry->pass = which;
			entry->chunks.push_back(std::move(chunk));
			m_ahead_wake.notify_all();
		} else {
			Unlink(chunk.record.get(), nullptr);
			Retire(std::move(chunk.record));
		}
		if (!boundary || entry == nullptr || which != m_pass) {
			m_state->SetChunking(0);
			return followed && entry != nullptr;
		}
	}
}

bool Speculation::ChunkInOrder() {
	static auto& on = LiveSwitch::Find("spec_chunk_in_order", [] {
		const char* value = std::getenv("KYTY_SPEC_CHUNK_IN_ORDER");
		return value != nullptr ? std::atoll(value) : ModeToggle::Default("spec_chunk_in_order");
	}());
	return on.load(std::memory_order_relaxed) != 0;
}

const Spec::State* Speculation::RecordOf(uint64_t serial, uint32_t chunk) {
	const auto* entry = EntryOf(serial);
	return entry != nullptr && chunk < entry->chunks.size() ? entry->chunks[chunk].record.get() : nullptr;
}

Speculation::Entry* Speculation::EntryOf(uint64_t serial) {
	for (auto& entry: m_jobs)
		if (entry.job.serial == serial) return &entry;
	return nullptr;
}

void Speculation::Unlink(const Spec::State* from, const Spec::State* to) {
	const auto relink = [&](Spec::State& state) {
		if (state.previous.load(std::memory_order_relaxed) == from) state.previous = to;
	};
	for (auto& ahead: m_ahead) relink(ahead->state);
	for (auto& translator: m_translators)
		if (translator->current != nullptr) relink(translator->current->state);
	for (auto& entry: m_jobs)
		for (auto& chunk: entry.chunks)
			if (chunk.record) relink(*chunk.record);
	if (m_pass_record != nullptr) relink(*m_pass_record);
}

uint64_t Speculation::Horizon() const {
	uint64_t horizon = UINT64_MAX;
	for (const auto& translator: m_translators)
		if (translator->since != 0) horizon = std::min(horizon, translator->since);
	return horizon;
}

std::unique_ptr<Spec::State> Speculation::TakeRecord() {
	const auto horizon = Horizon();
	for (auto it = m_retired_records.begin(); it != m_retired_records.end(); ++it)
		if (it->first < horizon) {
			auto record = std::move(it->second);
			m_retired_records.erase(it);
			record->Clear();
			return record;
		}
	return std::make_unique<Spec::State>();
}

void Speculation::Retire(std::unique_ptr<Spec::State> record) {
	m_retired_records.emplace_back(++m_clock, std::move(record));
}

// The workers' CPUs: KYTY_SPECULATE_CPUS=<list> when set; otherwise the upper half of the performance cores
// (LocalPlatform::SpeculationCpuList: CPUs 8-15 on an 8 P-core + 16 E-core part), never an E-core, the GPU, record and
// upload threads still free over the whole render list; where there is no such list (Linux), off the render threads'
// CPUs (KYTY_RENDER_CPUS). On the E-cores a worker translates a chunk slower than the GPU thread can wait for it: in
// the 8 Nov placement trial the arena gained 2.8 fps in-process with the workers on CPUs 8-15, Latria and 1-1 were
// level, and the translation lost to cancels fell by a third to a half everywhere. Live: speculate_cpu_mask
// (KYTY_SPECULATE_CPU_MASK), a CPU mask written in decimal (LiveSwitch reads every switch in base 10, the variable and
// `set` alike: CPUs 8-15 are 65280; "0xff00" reads as 0, this default), taken by each worker at its next wake. (It was
// named speculate_cpus, so LiveSwitch read KYTY_SPECULATE_CPUS, the list, as a number: "8,10,12,14" became mask 8, CPU
// 3, and every worker was pinned there.) More workers than the list's CPUs share them (SMT).
static const std::pair<std::string, const char*>& SpeculationCpus() {
	static const auto cpus = []() -> std::pair<std::string, const char*> {
		if (const char* list = std::getenv("KYTY_SPECULATE_CPUS"); list != nullptr && *list != 0) return {list, "KYTY_SPECULATE_CPUS"};
		return {LocalPlatform::SpeculationCpuList(), "the upper half of the performance cores"};
	}();
	return cpus;
}

static void PinSpeculationWorker(int64_t mask) {
	if (mask != 0) {
		std::string list;
		for (uint32_t cpu = 0; cpu < 64; ++cpu)
			if ((static_cast<uint64_t>(mask) >> cpu) & 1u) list += (list.empty() ? "" : ",") + std::to_string(cpu);
		LocalPlatform::PinThreadToCpuList(list.c_str());
	} else if (const auto& cpus = SpeculationCpus(); !cpus.first.empty()) {
		LocalPlatform::PinThreadToCpuList(cpus.first.c_str());
	} else {
		// (Off the GPU thread's CPUs: it commits what they translate, and waits for it.)
		LocalPlatform::AvoidCpuList(std::getenv("KYTY_RENDER_CPUS"));
	}
}

// KYTY_SPEC_PASS_CPUS: the state pass thread's CPUs. Unset: one CPU of the lowest efficiency class the process may run
// on, outside the workers' CPUs where there is one (LocalPlatform::EfficiencyCpu: an E-core, 16 on an 8 P-core + 16
// E-core part). On Linux, where efficiency classes are not read: a CPU on a physical core with none of the render
// threads' (KYTY_RENDER_CPUS) or the workers' CPUs (LocalPlatform::SpareCpu; 8, core 0, with KYTY_RENDER_CPUS=1-7 on an
// 8-core part numbered 0-7, 8-15 siblings), else a sibling of one; with none left the thread is not pinned. Where every
// core is alike on Windows it is not pinned.
static const std::string& PassCpus() {
	static const std::string cpus = [] {
		const char* list = std::getenv("KYTY_SPEC_PASS_CPUS");
		if (list != nullptr && *list != 0) return std::string(list);
		const std::string& workers = SpeculationCpus().first;
		if (std::string efficiency = LocalPlatform::EfficiencyCpu(workers.c_str()); !efficiency.empty()) return efficiency;
		std::string avoid  = workers;
		const char* render = std::getenv("KYTY_RENDER_CPUS");
		if (render != nullptr && *render != 0) avoid += (avoid.empty() ? "" : ",") + std::string(render);
		return LocalPlatform::SpareCpu(avoid.c_str());
	}();
	return cpus;
}

// KYTY_SPEC_GPU_AFFINITY (live spec_gpu_affinity; on under KYTY_SPECULATE=4, nothing done in mode 0): the GPU thread kept
// off the physical cores the speculation workers run on, still free over the rest of its own: its mask as set today
// (KYTY_RENDER_CPUS, or the process's) less every logical CPU of a core that holds a CPU of the workers' set (the live
// speculate_cpu_mask, else SpeculationCpus; SMT siblings with it: LocalPlatform::CoresOf). On an 8 P-core + 16 E-core
// part (GPU thread on CPUs 2-15, workers 8-15) about half the boots ran with the GPU thread's and the workers' work
// 20-45 % slower; with the GPU thread on 2-7, 3 of 3 were fast. Not done (said at each change) where the workers have no
// CPU list, or where what is left would hold fewer than two physical performance cores (SMT siblings one core, E-cores
// none: LocalPlatform::PerformanceCoresIn): a small CPU, or one whose workers take most of it, keeps today's mask
// (3 P-cores, workers on 3-5: 0-1 and the E-cores left, one performance core). Back to 0, or out of mode 4: the mask it
// had. Taken at the next flip after a change (the switch's, speculate_cpu_mask's); the GPU thread calls it at each
// guest flip in mode 4 (CommandProcessor::Flip) and once on a live `mode 0` (GuestGpu::ThreadRun), to restore.
void Speculation::GpuThreadAffinityAtFlip() {
	static uint64_t original = 0; // (the GPU thread's mask before it was changed; 0: not changed)
	static int64_t  seen     = 0;
	const auto      restore  = [](const char* why) {
		if (original == 0) return;
		const bool done = LocalPlatform::SetCurrentThreadAffinity(original);
		std::printf("Speculation: GPU thread mask 0x%llx %s (spec_gpu_affinity: %s)\n", static_cast<unsigned long long>(original),
		            done ? "restored" : "NOT restored", why);
		if (done) original = 0;
	};
	if (kyty_local_speculate_mode.load(std::memory_order_relaxed) != 4) {
		// (Out of mode 4: the mask back, and applied again on the return to 4.)
		if (original != 0) {
			restore("mode not 4");
			original = 0; // (tried once: a refused restore is said, not retried)
			std::fflush(stdout);
		}
		seen = 0;
		return;
	}
	static auto&   on        = LiveSwitch::Find("spec_gpu_affinity", ModeToggle::Default("spec_gpu_affinity"));
	static auto&   live_cpus = LiveSwitch::Find("speculate_cpu_mask", 0);
	static int64_t seen_cpus = 0;
	const int64_t  value     = on.load(std::memory_order_relaxed);
	const int64_t  cpus      = live_cpus.load(std::memory_order_relaxed);
	if (value == seen && (value == 0 || cpus == seen_cpus)) return;
	seen      = value;
	seen_cpus = cpus;
	if (value == 0) {
		restore("off");
		std::fflush(stdout);
		return;
	}
	const uint64_t workers = cpus != 0 ? static_cast<uint64_t>(cpus) : LocalPlatform::CpuListMask(SpeculationCpus().first.c_str());
	const uint64_t current = original != 0 ? original : LocalPlatform::CurrentThreadAffinity();
	const uint64_t wanted  = current & ~LocalPlatform::CoresOf(workers); // (whole cores: the workers' SMT siblings too)
	const uint32_t cores   = LocalPlatform::PerformanceCoresIn(wanted);
	const char*    refused = workers == 0        ? "the workers have no CPU list"
	                         : current == 0      ? "the GPU thread's mask is not known"
	                         : wanted == current ? "the GPU thread's CPUs and the workers' are apart already"
	                         : cores < 2         ? "fewer than 2 performance cores would be left to the GPU thread"
	                                             : nullptr;
	if (refused == nullptr && !LocalPlatform::SetCurrentThreadAffinity(wanted)) refused = "the OS refused the mask";
	if (refused != nullptr) {
		std::printf("Speculation: GPU thread affinity not changed: %s (mask 0x%llx, workers 0x%llx, %u performance cores "
		            "left; spec_gpu_affinity)\n",
		            refused, static_cast<unsigned long long>(current), static_cast<unsigned long long>(workers), cores);
		restore(refused);
		std::fflush(stdout);
		return;
	}
	if (original == 0) original = current;
	std::printf("Speculation: GPU thread mask 0x%llx -> 0x%llx (%u performance cores), off the cores of the workers' 0x%llx "
	            "(spec_gpu_affinity)\n",
	            static_cast<unsigned long long>(current), static_cast<unsigned long long>(wanted), cores,
	            static_cast<unsigned long long>(workers));
	std::fflush(stdout);
}

void Speculation::AheadRun(Translator& translator) {
	LocalPlatform::SetThreadName(translator.pass_only ? "Kyty.SpecPass" : "Kyty.Speculation");
	// (Worker packets, live spec_worker_packets: its segments recorded as packets, spliced at commit. Armed only
	// inside a segment recorded so: otherwise its calls go to the driver as before.)
	const LocalVulkanRecording::PassiveProducerScope passive_producer;
	static auto& live_cpus    = LiveSwitch::Find("speculate_cpu_mask", 0);
	int64_t      applied_cpus = live_cpus.load(std::memory_order_relaxed);
	if (translator.pass_only) {
		// (Normal priority; never re-pinned live. The CPUs said are those the pin took, or none.)
		const auto& cpus   = PassCpus();
		const bool  pinned = !cpus.empty() && LocalPlatform::TryPinThreadToCpuList(cpus.c_str());
		std::printf("Speculation: the state pass on its own thread, %s%s (KYTY_SPEC_PASS_THREAD; the workers take only "
		            "chunks)\n",
		            pinned ? "CPUs " : "not pinned", pinned ? LocalPlatform::CpuListResolved(cpus.c_str()).c_str() : "");
	} else {
		PinSpeculationWorker(applied_cpus);
	}
	if (translator.index == 0) {
		const auto& cpus   = SpeculationCpus();
		const char* render = std::getenv("KYTY_RENDER_CPUS");
		std::printf("Speculation: %u workers on CPUs %s (%s); render threads on CPUs %s\n",
		            static_cast<unsigned>(m_translators.size() - (m_pass_thread ? 1u : 0u)),
		            cpus.first.empty() ? "off the render threads' CPUs" : LocalPlatform::CpuListResolved(cpus.first.c_str()).c_str(),
		            cpus.first.empty() ? "no list" : cpus.second,
		            render != nullptr && *render != 0 ? LocalPlatform::CpuListResolved(render).c_str() : "any");
	}
	// (Its Vulkan calls go to the driver as they are made, no recording stream: what they name is alive then.)
	RenderContext::SetThreadExecutors(&translator.executors);
	GuestGpu::SetSpeculationThread(true);
	Spec::t_packet_slot = translator.index;
	std::unique_lock lock(m_mutex);
	for (;;) {
		Entry*   pass       = nullptr;
		Entry*   work       = nullptr;
		uint32_t work_chunk = 0;
		m_ahead_wake.wait(lock, [&] {
			if (m_quit) return true;
			ResyncStatePass();
			// (KYTY_SPEC_PASS_THREAD: only its thread takes the pass, and it takes nothing else.)
			const bool takes_pass = translator.pass_only || !m_pass_thread;
			pass                  = takes_pass && !m_state_busy && !m_broken ? EntryOf(m_state_serial + 1) : nullptr;
			work                  = nullptr;
			if (translator.pass_only) return pass != nullptr;
			if (pass == nullptr)
				for (auto& entry: m_jobs) {
					if (entry.pass != m_pass || entry.job.serial <= m_done_serial) continue;
					for (uint32_t c = 0; c < entry.chunks.size() && work == nullptr; ++c)
						if (!entry.chunks[c].taken && !Taken(entry.job.serial, c)) {
							work       = &entry;
							work_chunk = c;
						}
					if (work != nullptr) break;
				}
			return pass != nullptr || work != nullptr;
		});
		if (m_quit) return;
		if (const auto mask = live_cpus.load(std::memory_order_relaxed); !translator.pass_only && mask != applied_cpus) {
			applied_cpus = mask;
			PinSpeculationWorker(mask);
		}
		if (pass != nullptr) {
			// The state pass of the next submission (one thread at a time; the state it starts from goes to it, and its
			// record, which reads through the one before while that is not done).
			m_state_busy     = true;
			translator.since = ++m_clock;
			const auto job   = pass->job;
			const auto which = m_pass;
			// (Chunks of an earlier pass over it go: what read through their records reads memory.)
			for (auto& chunk: pass->chunks)
				if (auto record = std::move(chunk.record)) {
					Unlink(record.get(), nullptr);
					Retire(std::move(record));
				}
			pass->chunks.clear();
			lock.unlock();
			const bool followed = StatePass(job, which);
			lock.lock();
			translator.since = 0;
			m_state_busy     = false;
			if (!followed) {
				m_broken    = true;
				m_broken_at = job.serial;
			}
			if (which == m_pass) m_state_serial = job.serial;
			m_ahead_wake.notify_all();
			continue;
		}
		// A translation, from the state the pass gave it (with the pending writes of the one before when it is done).
		auto& chunk                  = work->chunks[work_chunk];
		chunk.taken                  = true;
		translator.translating       = work->job.serial;
		translator.translating_chunk = work_chunk;
		translator.since             = ++m_clock; // (before `previous` can be retired: it is not reused meanwhile)
		translator.began_ns          = NowNs();
		translator.cancel.store(false, std::memory_order_relaxed);
		const auto         job       = work->job;
		const auto         which     = work->pass;
		auto               start     = std::move(chunk.start);
		const Pm4Execution at        = chunk.at;
		const auto         stop      = chunk.last ? std::optional<Pm4Execution> {} : std::optional {chunk.end};
		// The chunk before it: its translation not committed yet, else the pass's record of it.
		const auto* before_entry = work_chunk > 0 ? work : EntryOf(job.serial - 1);
		const auto  before_serial = work_chunk > 0 ? job.serial : job.serial - 1;
		const auto  before_chunk  = work_chunk > 0 ? work_chunk - 1
		                            : before_entry != nullptr && !before_entry->chunks.empty()
		                                ? static_cast<uint32_t>(before_entry->chunks.size() - 1)
		                                : UINT32_MAX;
		const Result* previous = nullptr;
		for (const auto& ahead: m_ahead)
			if (ahead->serial == before_serial && ahead->chunk == before_chunk && ahead->pass == which) previous = ahead.get();
		if (work_chunk == 0) {
			m_start_delay_ns += NowNs() - work->pushed_ns;
			++m_started;
		}
		const Spec::State* record = RecordOf(before_serial, before_chunk);
		const bool         blind  = previous == nullptr && record == nullptr && job.serial > m_done_serial + 1;
		lock.unlock();
		const auto began  = NowNs();
		auto       result = Translate(translator, *start, job.commands, at, previous, record, stop ? &*stop : nullptr);
		result->serial = job.serial;
		result->chunk  = work_chunk;
		result->pass   = which;
		result->blind  = blind;
		Spec::LeavePackets();
		lock.lock();
		{
			const auto took = NowNs() - began;
			const auto ms   = took / 1000000.0;
			++m_durations[ms < 0.5 ? 0 : ms < 1 ? 1 : ms < 2 ? 2 : ms < 4 ? 3 : ms < 8 ? 4 : 5];
			m_longest_ns = std::max(m_longest_ns, took);
		}
		translator.translating = 0;
		m_ahead.push_back(std::move(result));
		m_ahead_wake.notify_all();
	}
}

Speculation::Outcome Speculation::CommitReady(CommandProcessor& cp, const Job& job, uint32_t chunk, Pm4Execution& execution,
                                              std::optional<Pm4Execution>& end) {
	std::unique_ptr<Result> result;
	{
		std::unique_lock lock(m_mutex);
		// Where the chunk ends (the rest of it the normal way, up to there, when there is none for it or its commit
		// stops).
		const auto chunk_end = [&] {
			end.reset();
			if (const auto* entry = EntryOf(job.serial);
			    entry != nullptr && entry->pass == m_pass && chunk < entry->chunks.size() && !entry->chunks[chunk].last)
				end = entry->chunks[chunk].end;
		};
		// The normal way: the speculation's threads do not begin it, nor, its end unknown, any chunk of it after it.
		const auto take = [&] {
			chunk_end();
			// (KYTY_SPEC_CHUNK_IN_ORDER: a chunk the pass has not cut is translated here up to where the pass will end it;
			// the chunks after it stay the speculation's threads'.)
			const auto taken = end || ChunkInOrder() ? chunk : UINT32_MAX;
			if (job.serial > m_taken || (job.serial == m_taken && taken > m_taken_chunk)) {
				m_taken       = job.serial;
				m_taken_chunk = taken;
			}
		};
		for (;;) {
			// (Those of chunks before it, taken the normal way since, or of a state pass that went wrong.)
			for (size_t i = 0; i < m_ahead.size();) {
				if ((m_ahead[i]->serial > job.serial || (m_ahead[i]->serial == job.serial && m_ahead[i]->chunk >= chunk)) &&
				    m_ahead[i]->pass == m_pass) {
					++i;
					continue;
				}
				auto dropped = std::move(m_ahead[i]);
				m_ahead.erase(m_ahead.begin() + static_cast<ptrdiff_t>(i));
				lock.unlock();
				Finish(std::move(dropped), 0);
				lock.lock();
				i = 0;
			}
			const auto found = std::ranges::find_if(m_ahead, [&](const auto& ahead) {
				return ahead->serial == job.serial && ahead->chunk == chunk;
			});
			if (found != m_ahead.end()) {
				result = std::move(*found);
				m_ahead.erase(found);
				break;
			}
			// Being translated: its result (the normal translation would write what tells the guest the command
			// buffer is done, which it may then reuse under the speculation's thread). Not begun: this thread translates
			// it the normal way, the speculation's threads do not begin it.
			const auto translating = [&] {
				return std::ranges::any_of(m_translators, [&](const auto& translator) {
					return translator->translating == job.serial && translator->translating_chunk == chunk;
				});
			};
			if (translating()) {
				// Mode 4, the graphics instance: before the cancel, another queue's compute front that is runnable and ready
				// (the async speculation's front has its chunk translated, or an unspeculated one: in-order work this thread
				// does anyway) runs first: Yield, the submission requeued at this chunk, whose worker goes on meanwhile; on
				// the return the chunk is committed if ready, else cancelled as before. One check, so one yield, a chunk.
				// Lock order: this speculation's m_mutex is released for the check (GuestGpu's queue mutex, then the other
				// instance's own m_mutex in FrontReady, one at a time), then retaken and the translation looked at again.
				// Exact: between queues the guest sees order only through labels and WAIT_REG_MEM (each queue's waits are
				// evaluated when it runs) and the frame epoch (CanProcessSubmission); on the console the queues run
				// concurrently, so any interleaving at packet boundaries that keeps these is a legal schedule, as the
				// Blocked requeue already makes. A yielded-to queue's write over what a later graphics chunk read is caught
				// by that chunk's commit checks (the effects log): a stop, never a wrong frame.
				if (m_interrupt_id == 0 && kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4 &&
				    !(m_ready_checked_serial == job.serial && m_ready_checked_chunk == chunk)) {
					m_ready_checked_serial = job.serial, m_ready_checked_chunk = chunk;
					lock.unlock();
					const int target = GuestGpu::ReadyFrontElsewhere();
					lock.lock();
					if (target != 0) {
						if (!translating()) continue; // (done or stopped meanwhile: looked at again)
						end.reset();
						return Outcome::Yield;
					}
				}
				// KYTY_SPECULATE_CANCEL_US (live speculate_cancel_us, 0: never): a translation begun less than that long ago
				// has more left than this thread takes for the whole chunk (a speculation's thread is several times slower
				// per draw): it stops at its next packet, what it did is committed, the rest is translated here.
				static auto& cancel_us = LiveSwitch::Find("speculate_cancel_us", 1000000); // (default 1 s: always)
				if (const auto us = cancel_us.load(std::memory_order_relaxed); us > 0)
					for (auto& translator: m_translators)
						if (translator->translating == job.serial && translator->translating_chunk == chunk &&
						    NowNs() - translator->began_ns < static_cast<uint64_t>(us) * 1000u &&
						    !translator->cancel.exchange(true, std::memory_order_relaxed))
							++m_cancelled;
				++m_waits_for_ahead;
				const auto waited = NowNs();
				// (KYTY_SPECULATE_FLUSH_EVERY: the commits held go before waiting on a speculation's thread.)
				if (FlushEvery() > 1) {
					lock.unlock();
					m_renderer.GetCommandScheduler().FlushRecorded();
					lock.lock();
				}
				// (KYTY_SERVE_IN_WAITS 4: the guest commands queued for this thread run while it waits.)
				if ((GuestGpu::ServeInWaits() & 4u) != 0) {
					while (translating()) {
						if (GuestGpu::CommandsPending()) {
							lock.unlock();
							GuestGpu::ServePendingCommands();
							lock.lock();
							continue;
						}
						m_ahead_wake.wait_for(lock, std::chrono::microseconds(50));
					}
				} else {
					m_ahead_wake.wait(lock, [&] { return !translating(); });
				}
				m_wait_ns += NowNs() - waited;
				continue;
			}
			take();
			++m_ahead_misses;
			return Outcome::None;
		}
		if (!cp.SameState(*result->start_state) || !CommandProcessor::SamePosition(execution, result->start_at)) {
			// From another state or place: the state pass went wrong (what began from it goes); it goes on from the state
			// after this submission (Done), which is translated on the normal way.
			++m_stop_reasons["ahead: another start state"];
			++m_pass;
			m_broken    = true;
			m_broken_at = job.serial;
			m_taken     = job.serial;
			m_taken_chunk = UINT32_MAX;
			end.reset();
			lock.unlock();
			Finish(std::move(result), 0);
			return Outcome::None;
		}
		chunk_end();
	}
	return Commit(std::move(result), cp, execution);
}

bool Speculation::FrontReady(uint64_t serial, uint32_t chunk) {
	std::lock_guard lock(m_mutex);
	return std::ranges::any_of(m_ahead, [&](const auto& ahead) {
		return ahead->serial == serial && ahead->chunk == chunk && ahead->pass == m_pass;
	});
}

Speculation::Outcome Speculation::CommitAhead(CommandProcessor& cp, const Job& job, std::span<const Job> queued,
                                              Pm4Execution& execution) {
	auto& translator = *m_translators[0];
	// Speculations of command buffers taken the normal way since, or begun from another state.
	const auto drop_ahead = [&] {
		while (!m_ahead.empty()) {
			auto dropped = std::move(m_ahead.front());
			m_ahead.pop_front();
			Finish(std::move(dropped), 0);
		}
	};
	while (!m_ahead.empty() && m_ahead.front()->serial < job.serial) {
		auto dropped = std::move(m_ahead.front());
		m_ahead.pop_front();
		Finish(std::move(dropped), 0);
	}
	std::unique_ptr<Result> result;
	if (!m_ahead.empty() && m_ahead.front()->serial == job.serial) {
		result = std::move(m_ahead.front());
		m_ahead.pop_front();
		if (!cp.SameState(*result->start_state)) {
			++m_stop_reasons["ahead: another start state"];
			Finish(std::move(result), 0);
			drop_ahead();
		}
	}
	if (!result) {
		result         = Translate(translator, cp, job.commands, execution);
		result->serial = job.serial;
	}
	// The ones queued after the last translated, from the state it leaves: while each takes the one before's place
	// and none stopped before its end.
	const Result* last = m_ahead.empty() ? result.get() : m_ahead.back().get();
	auto&         from = m_chain;
	from->CopyStateFrom(*last->end_state, false);
	uint64_t serial = last->serial;
	for (const auto& next: queued) {
		if (next.serial <= serial) continue;
		if (next.serial != serial + 1 || next.kind == Job::Kind::Other || last->stop != nullptr || last->failed ||
		    m_ahead.size() >= MaxAhead)
			break;
		serial = next.serial;
		if (next.reset) from->Reset();
		if (next.kind == Job::Kind::Flip) continue;
		from->ResetDeCe();
		from->SetFlip({});
		auto ahead    = Translate(translator, *from, next.commands, Pm4Execution {}, last);
		ahead->serial = next.serial;
		m_ahead.push_back(std::move(ahead));
		last = m_ahead.back().get();
		from->CopyStateFrom(*last->end_state, false);
	}
	return Commit(std::move(result), cp, execution);
}

std::unique_ptr<Speculation::Result> Speculation::Translate(Translator& translator, const CommandProcessor& cp,
                                                            std::span<const uint32_t> commands,
                                                            const Pm4Execution& execution, const Result* previous,
                                                            const Spec::State* record, const Pm4Execution* stop) {
	auto&                                      scheduler = m_renderer.GetCommandScheduler();
	std::unique_ptr<Result>                    result;
	std::vector<std::pair<uint64_t, uint64_t>> resolved;
	// KYTY_SPEC_RESULT_POOL_CAP (live spec_result_pool_cap; 0, the default: off): a translator keeps at most that many
	// finished results for reuse. Without it `free` only grows: a translation finds none to reuse while a translation
	// going on began before they were retired (a long one, a burst: a level load), makes one, and keeps it (two
	// processor copies and its hole states, ~55 KB each). The rest is freed a few at a time as translations begin, on
	// the thread that translates with them (that made them, their recorder's pool its own), only one no translation
	// going on reaches (retired before the horizon, as reuse) whose command buffers the GPU is done with.
	// KYTY_SPEC_HOLE_STATES_CAP (live spec_hole_states_cap; 0, the default: off): a result reused keeps at most that many
	// hole states (more than its holes: kept for reuse; a translation with more makes them again).
	static auto&            pool_cap        = LiveSwitch::Find("spec_result_pool_cap", 0);
	static auto&            hole_states_cap = LiveSwitch::Find("spec_hole_states_cap", 0);
	static constexpr size_t FreedATranslation = 4;
	std::vector<std::unique_ptr<Result>> surplus;
	{
		std::lock_guard lock(m_mutex);
		if (translator.since == 0) translator.since = ++m_clock;
		// (One retired before every translation going on began: none of them reaches it through `previous` any more.)
		const auto horizon = Horizon();
		if (const auto reusable = std::ranges::find_if(translator.free, [&](const auto& kept) { return kept->retired < horizon; });
		    reusable != translator.free.end()) {
			result = std::move(*reusable);
			translator.free.erase(reusable);
		}
		if (const auto cap = pool_cap.load(std::memory_order_relaxed); cap > 0) {
			const auto& master = scheduler.GetMasterSemaphore();
			for (auto it = translator.free.begin(); it != translator.free.end() &&
			                                        translator.free.size() > static_cast<uint64_t>(cap) &&
			                                        surplus.size() < FreedATranslation;) {
				const auto& kept = **it;
				// (Its recorder's pool: destroyed on the kind of thread that recorded with it last. Not the translator's
				// current result: Unlink relinks it, the log's DropBefore reads it.)
				const bool done = kept.retired < horizon && &kept != translator.current &&
				                  // (A worker's recorder only, freed on a worker: a GPU-thread recorder's discarded buffers read
				                  // tick 0 while their reset still waits on the Record stream, modes 1-3.)
				                  kept.recorder->own_thread && !GuestGpu::IsGpuThread() &&
				                  std::ranges::all_of(kept.recorder->buffers, [&](const auto& buffer) {
					                  // (UINT64_MAX: being recorded or recorded, not yet submitted: not done; DestroyRecorder
					                  // would not wait for it.)
					                  return buffer.second != UINT64_MAX && (buffer.second == 0 || master.IsFree(buffer.second));
				                  });
				if (!done) {
					++it;
					continue;
				}
				surplus.push_back(std::move(*it));
				it = translator.free.erase(it);
			}
		}
		resolved.swap(translator.resolved);
	}
	for (auto& freed: surplus) FreeResult(std::move(freed));
	// (What the commits released of its rings, released on the thread that uses them.)
	for (const auto& [pending, tick]: resolved) ResolvePending(translator, pending, tick);
	if (const auto cap = hole_states_cap.load(std::memory_order_relaxed);
	    result && cap > 0 && result->hole_states.size() > static_cast<uint64_t>(cap))
		result->hole_states.resize(static_cast<size_t>(cap));
	if (!result) {
		result              = std::make_unique<Result>();
		result->owner       = &translator;
		result->recorder    = scheduler.CreateRecorder();
		result->start_state = std::make_unique<CommandProcessor>(m_renderer, m_interrupt_id);
		result->end_state   = std::make_unique<CommandProcessor>(m_renderer, m_interrupt_id);
	}
	auto& r = *result;
	// (A speculation's thread records and ends its buffers; the GPU thread's go through its recording stream.)
	r.recorder->own_thread   = !GuestGpu::IsGpuThread();
	r.recorder->pending_tick = m_pending.fetch_add(1, std::memory_order_relaxed) + 1;
	// KYTY_SPEC_WORKER_PACKETS (live spec_worker_packets; on under KYTY_SPECULATE=4): a speculation thread records its
	// segments as packets against a placeholder, spliced into the GPU thread's recording stream at commit (no command
	// buffers of its own, no submissions of them). The GPU thread's own translations record as before.
	static auto& worker_packets = LiveSwitch::Find("spec_worker_packets", ModeToggle::Default("spec_worker_packets"));
	r.recorder->packets = r.recorder->own_thread && worker_packets.load(std::memory_order_relaxed) != 0 &&
	                      LocalVulkanRecording::HasPassiveProducer();
	auto& processor          = *translator.cp;
	processor.CopyStateFrom(cp, false); // (no packet it takes uses the constant RAM)
	r.start_state->CopyStateFrom(cp, false);
	r.serial = 0;
	r.pass   = 0;
	r.blind  = false;
	r.state.Clear();
	// (Relocatable buffer references: the mode once a translation, so a live change cannot split a segment's notes.)
	r.state.reloc_on      = kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4;
	r.state.table_upload  = translator.table_upload.get();
	r.state.shader_upload = translator.shader_upload.get();
	r.state.cancel        = &translator.cancel;
	{
		// (Linked under the lock: one committed or dropped since it was chosen is not; Finish unlinks the others.)
		std::lock_guard lock(m_mutex);
		r.retired = 0;
		kyty_local_speculations_live.fetch_add(1, std::memory_order_seq_cst); // (Finish takes it back)
		if (previous != nullptr && previous->retired == 0)
			r.state.previous = &previous->state;
		else if (record != nullptr && std::ranges::any_of(m_jobs, [&](const Entry& entry) {
			         return std::ranges::any_of(entry.chunks, [&](const Chunk& chunk) { return chunk.record.get() == record; });
		         }))
			r.state.previous = record;
		else
			r.state.previous = nullptr;
		// (Where its effects begin, as it becomes the translation going on: the log keeps what it may need.)
		r.effects_from     = m_log.End();
		r.state.log_end    = &m_log.end;
		translator.current = &r;
	}
	r.state.Inherit();
	r.holes.clear();
	r.epochs = CurrentEpochs();
	r.segment_ends.clear();
	r.stop           = nullptr;
	r.stop_execution = {};
	r.failed         = false;
	r.submitted      = 0;
	r.last_tick      = 0;
	r.next           = 0;
	r.blocked        = SIZE_MAX;
	r.start_ns       = NowNs();
	r.start_at       = execution;
	Pm4Execution spec = execution;
	// (A chunk's: up to where the next begins.)
	Pm4Execution stop_at;
	if (stop != nullptr) stop_at = *stop;
	processor.SetStopAt(stop != nullptr ? &stop_at : nullptr);
	// (What it translates is not done: nothing it does is an effect.)
	auto* const effects = std::exchange(Spec::t_effects, nullptr);
	auto* const effects_async = std::exchange(Spec::t_effects_async, nullptr);
	// (Counted above: EvictIdlePipelines either saw it and evicts none, or set its flag first and this waits for it
	// before any pipeline is bound.)
	while (kyty_local_pipelines_evicting.load(std::memory_order_seq_cst) != 0) std::this_thread::yield();
	RenderContext::SetThreadExecutors(&translator.executors);
	// (What it kept of the catalogs from its last translation, dropped when a sweep or reset happened since.)
	translator.draw->ForgetStaleCatalog();
	translator.compute->ForgetStaleCatalog();
	// (This thread's draw state shadow and state key as nothing known: a state pass on the same thread (no pass thread,
	// KYTY_SPEC_PASS_THREAD=0 or no slot for it; the async queue always) writes this chunk's own register values into
	// the shadow, and a write of the same value in the translation would then look unchanged: its next draw taken as
	// clean, with the previous draw's key.)
	DrawStateObserver::Invalidate();
	translator.draw->NativeXprForgetState();
	{
	Spec::t_state = &r.state;
	CommandScheduler::SetThreadRecorder(r.recorder.get());
	try {
		processor.BufferInit();
		for (;;) {
			const auto processed = processor.Process(spec, commands);
			if (r.state.refused == nullptr) {
				// (Its chunk's end, or a constant engine wait.)
				// (CommandWordsWatched: its next packet's words on a page a read watcher guards, ProcessPm4Spec.)
				if (processed != Pm4ProcessResult::Complete)
					r.stop = processor.TakeStopReached() ? ChunkEnd : processor.TakeWatchedStop() ? CommandWordsWatched : "suspended";
				break;
			}
			// (A packet that does not fit its buffer, refused "malformed packet" before its Mark: a command buffer the game
			// rewrote after the GPU thread passed it. Not sized (SuspendedPacket would end the process) and nothing taken:
			// the checkpoint is the packet before's, so a Rollback would drop that packet's notes; the GPU thread redoes it.)
			if (!CommandProcessor::SuspendedPacketFits(spec)) {
				r.failed = true;
				r.stop   = r.state.refused;
				break;
			}
			uint32_t   dwords  = 0;
			const auto packet  = CommandProcessor::SuspendedPacket(spec, dwords);
			const bool partial = LocalVulkanRecording::RecordedWrites() != r.state.checkpoint_work;
			r.state.Rollback();
			if (partial) { // (its recorded work cannot be taken back)
				static const auto names = [] {
					std::array<std::string, 256> all;
					for (uint32_t i = 0; i < 256; ++i) all[i] = "partly recorded packet " + std::to_string(i);
					return all;
				}();
				static std::atomic<uint32_t> reports {0};
				if (reports.fetch_add(1, std::memory_order_relaxed) < 16)
					std::printf("Speculation: packet 0x%08x refused (%s) after recording work\n", *packet, r.state.refused);
				r.failed = true;
				r.stop   = names[(*packet >> 8u) & 0xffu].c_str();
				break;
			}
			if (!IsHole(*packet) || r.state.stop) {
				r.stop = r.state.refused;
				break;
			}
			// A hole: the segment ends before it, the next begins after it, from the processor state there.
			if (r.hole_states.size() == r.holes.size()) r.hole_states.push_back(std::make_unique<CommandProcessor>(m_renderer, m_interrupt_id));
			r.hole_states[r.holes.size()]->CopyStateFrom(processor, false);
			auto& hole  = r.holes.emplace_back();
			hole.packet = packet;
			hole.dwords = dwords;
			hole.header = *packet;
			hole.reason = r.state.refused;
			hole.at     = spec; // (spec_stop_as_hole)
			CommandProcessor::SkipSuspendedPacket(spec);
			hole.after = spec;
			scheduler.EndRecording(*r.recorder);
			r.state.CloseSegment(r.recorder->recorded.size());
			r.segment_ends.push_back(NowNs());
			r.state.refused = nullptr;
			// (The work after it does not continue the draw state of the work before it.)
			DrawStateObserver::Invalidate();
			translator.draw->NativeXprForgetState();
			processor.BufferInit();
		}
		scheduler.EndRecording(*r.recorder);
	} catch (const CommandScheduler::RecorderRefusal& refusal) {
		r.failed = true;
		r.stop   = refusal.what;
		// (Worker packets: the open segment ends here, on its thread, so the producer stops recording this thread's
		// calls; Finish drops it.)
		if (r.recorder->packets) scheduler.EndRecording(*r.recorder);
	}
	CommandScheduler::SetThreadRecorder(nullptr);
	Spec::t_state = nullptr;
	processor.SetStopAt(nullptr);
	(void)processor.TakeStopReached();
	(void)processor.TakeWatchedStop();
	// (The draw state the translation goes on from is not this thread's shadow any more.)
	DrawStateObserver::Invalidate();
	translator.draw->NativeXprForgetState();
	}
	if (GuestGpu::IsGpuThread()) RenderContext::SetThreadExecutors(nullptr);
	Spec::t_effects = effects;
	Spec::t_effects_async = effects_async;
	DrawStateObserver::Invalidate();
	r.end_ns     = NowNs();
	r.progressed = !r.holes.empty() || spec.MadeProgress();
	if (r.progressed && !r.failed) {
		r.state.CloseSegment(r.recorder->recorded.size());
		r.segment_ends.push_back(r.end_ns);
		r.stop_execution = std::move(spec);
	}
	r.end_state->CopyStateFrom(processor, false);
	{
		std::lock_guard lock(m_mutex);
		translator.since = 0;
	}
	return result;
}

Speculation::Outcome Speculation::Commit(std::unique_ptr<Result> result, CommandProcessor& cp, Pm4Execution& execution) {
	auto& r = *result;
	// Stopped at its first packet (a wait it does not find satisfied yet): nothing to commit.
	if (!r.progressed) {
		Finish(std::move(result), 0);
		return Outcome::None;
	}
	if (r.stop != nullptr && r.stop != ChunkEnd) ++m_stop_reasons[r.stop];
	m_effects.Clear(r.effects_from);
	return Continue(std::move(result), cp, execution);
}

Speculation::Outcome Speculation::Continue(std::unique_ptr<Result> result, CommandProcessor& cp, Pm4Execution& execution) {
	auto&      r        = *result;
	const auto start    = NowNs();
	const auto progress = r.failed ? Progress::Stopped : CommitSegments(r, cp);
	// (Its segments' buffers held for one submission go now: a speculation thread may wait for their tick.
	// KYTY_SPECULATE_FLUSH_EVERY: every Nth commit, and one that blocks at a wait hole.)
	if (FlushEvery() <= 1 || progress == Progress::Blocked || ++m_held_commits >= FlushEvery()) {
		m_held_commits = 0;
		m_renderer.GetCommandScheduler().FlushRecorded();
	}
	m_commit_ns += NowNs() - start;
	if (progress == Progress::Blocked) {
		m_blocked = std::move(result);
		return Outcome::Blocked;
	}
	const auto committed = r.next;
	m_blind += r.blind ? 1 : 0;
	if (progress == Progress::Stopped) {
		++m_stopped;
		m_blind_stopped += r.blind ? 1 : 0;
	}
	const auto kept      = committed == 0 ? r.start_ns : r.segment_ends[committed - 1];
	m_kept_ns += kept - r.start_ns;
	m_lost_ns += r.end_ns - kept;
	m_segments += committed;
	auto outcome = Outcome::None;
	if (progress == Progress::Complete) {
		EXIT_IF(committed != r.state.closed.size());
		// Committed whole: the processor state at its end (or at the packet it stopped at).
		cp.CopyStateFrom(*r.end_state, false);
		// (The registers are the worker's end state now, not what this thread's draw state shadow and state key saw
		// before the chunk: without this, its next draw could be taken as clean, or keep the old context hash, and bind
		// the draw state of the last draw it translated itself. As after a hole or spliced worker packets.)
		DrawStateObserver::Invalidate();
		m_renderer.GetRenderExecutor().NativeXprForgetState();
		outcome = r.stop == nullptr ? Outcome::Whole : r.stop == ChunkEnd ? Outcome::Chunk : Outcome::Partly;
		if (r.stop != nullptr) execution = std::move(r.stop_execution);
	} else if (r.resume_at) {
		// Stopped where `cp` ran a segment as a hole to (SegmentAsHole). A run that missed its hole and ended the
		// submission (its stack empty) made it whole: an empty execution handed back would process it again from its start.
		const bool ended = !r.resume_at->Started();
		execution        = std::move(*r.resume_at);
		outcome          = ended ? Outcome::Whole : Outcome::Partly;
	} else if (committed != 0) {
		// Stopped after the hole that followed the last segment committed (`cp` translated it).
		execution = std::move(r.holes[committed - 1].after);
		outcome   = Outcome::Partly;
	}
	r.resume_at.reset();
	++(outcome == Outcome::Whole || outcome == Outcome::Chunk ? m_whole : outcome == Outcome::Partly ? m_partly : m_none);
	Finish(std::move(result), committed);
	uint64_t trim_from = 0; // (KYTY_SPEC_STAMP_TRIM: the bound, taken under the lock; the leaves freed after it)
	bool     trim      = false;
	{
		// What no result began before: of those translated ahead (in the order they were done, not begun) and those
		// being translated (from when their translator took them: its current result's until then).
		std::lock_guard lock(m_mutex);
		auto            oldest = m_log.End();
		for (const auto& ahead: m_ahead) oldest = std::min(oldest, ahead->effects_from);
		for (const auto& translator: m_translators)
			if (translator->translating != 0 && translator->current != nullptr)
				oldest = std::min(oldest, translator->current->effects_from);
		m_log.DropBefore(oldest);
		// KYTY_SPEC_STAMP_TRIM (live spec_stamp_trim; 0, the default: off): every 1024 commits, the stamps' leaves
		// (128 KiB and 8.5 KiB, one a 4 MiB granule written, kept for good without it) of granules nothing was written
		// over since what no result began before (nor the one waiting at a wait hole) freed: every question is asked
		// from a result's beginning or later. (This thread made them: the thread that commits alone stamps.) The bound is
		// taken here; the leaves are freed after the lock (up to ~270 MiB the first time, which workers would otherwise
		// wait out on m_mutex): only this thread stamps and reads them (workers read the log's end alone).
		static auto& stamp_trim = LiveSwitch::Find("spec_stamp_trim", 0);
		if (stamp_trim.load(std::memory_order_relaxed) != 0 && ++m_stamp_trim_commits >= 1024) {
			m_stamp_trim_commits = 0;
			trim                 = true;
			trim_from            = m_blocked ? std::min(oldest, m_blocked->effects_from) : oldest;
		}
	}
	if (trim) {
		m_log.stamps.FreeLeavesBefore(trim_from);
		m_log.host_stamps.FreeLeavesBefore(trim_from);
	}
	Report();
	return outcome;
}

const char* Speculation::Changed(const Result& result, size_t index, bool skip_registered) const {
	extern thread_local bool t_unmap_list_site; // (spec_unmap_list: its checks share one `since`)
	struct ListSite {
		ListSite() { t_unmap_list_site = true; }
		~ListSite() { t_unmap_list_site = false; }
	} list_site;
	auto&       buffers  = m_renderer.GetBufferCache();
	auto&       textures = m_renderer.GetTextureCache();
	const auto& state    = result.state;
	const auto& epochs   = result.epochs;
	const auto& segment  = state.closed[index];
	const auto  from     = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
	// (Memory mapped since changes nothing it used; memory unmapped under what it read or uses does.)
	auto&      resources = m_renderer.GetGpuResources();
	const bool mapping   = resources.MappingEpoch() != epochs.mapping;
	// (Nothing registered, released or unmapped since it began, mostly: no range or image to look at.)
	if (mapping || buffers.RegistrationEpoch() != epochs.buffers)
		for (size_t i = from.buffer_ranges; i < segment.end.buffer_ranges; ++i) {
			const auto& range = state.buffer_ranges[i];
			if (!skip_registered && buffers.RegisteredSince(range.begin, range.end - range.begin, epochs.buffers)) {
				// The 4 MiB region changed over the range itself, or only beside it (live spec_exact_reg_stop; -1, the default,
				// and 1: on under KYTY_SPECULATE=4; 0 off: only a change over the range stops). Only mode 4 keeps the granule
				// epochs RegisteredSinceExact reads, so outside it the switch is ignored whatever its value (a forced 1 there
				// would let a registration over the read range pass): every region change stops, as without the switch.
				static auto&       exact_stop = LiveSwitch::Find("spec_exact_reg_stop", -1);
				static std::atomic exact_said {false};
				const auto         exact_v    = exact_stop.load(std::memory_order_relaxed);
				const bool         mode4      = kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4;
				if (exact_v > 0 && !mode4 && !exact_said.exchange(true, std::memory_order_relaxed))
					std::printf("Speculation: spec_exact_reg_stop=1 ignored outside KYTY_SPECULATE=4"
					            " (no granule epochs there)\n");
				const bool exact_on = exact_v != 0 && mode4;
				if (!exact_on || buffers.RegisteredSinceExact(range.begin, range.end - range.begin, epochs.buffers))
					return "commit: buffers registered";
			}
			if (mapping && resources.UnmappedSince(epochs.mapping, range.begin, range.end - range.begin))
				return "commit: memory unmapped";
		}
	if (mapping) {
		// (spec_pending_buffers / spec_reloc_refs: the table V# entries, noted as no buffer range, unmapped as any.)
		for (size_t i = from.table_ranges; i < segment.end.table_ranges; ++i) {
			const auto& range = state.table_ranges[i];
			if (resources.UnmappedSince(epochs.mapping, range.begin, range.end - range.begin)) return "commit: memory unmapped";
		}
		for (size_t i = from.clean_reads; i < segment.end.clean_reads; ++i) {
			const auto& read = state.clean_reads[i];
			if (resources.UnmappedSince(epochs.mapping, read.begin, read.end - read.begin)) return "commit: memory unmapped";
		}
		for (size_t i = from.late_reads; i < segment.end.late_reads; ++i) {
			const auto& read = state.late_reads[i];
			if (resources.UnmappedSince(epochs.mapping, read.begin, read.end - read.begin)) return "commit: memory unmapped";
		}
		for (const auto& use: segment.images)
			if (resources.UnmappedSince(epochs.mapping, use.image->info.data.address, use.image->info.data.size)) {
				// redo_level_precise: not over the bytes of the levels the segment sampled, when it only sampled it.
				if (Spec::RedoLevelPreciseOn() && SampledLevelsMapped(use, epochs.mapping)) continue;
				return "commit: memory unmapped";
			}
	}
	// spec_set_cpu_dirty_stop (Spec::SetCpuDirtyStopOn): every image the segment uses is still the object it found
	// (serial) and registered, whatever the epoch says. UnregisterImage moves the resolution epoch before it clears
	// `registered`: a speculation that took its epochs and made an entry between the two (TableEarlyEntry read it still
	// registered) has an epoch that includes the unregister, so the check below is skipped, and Before would restore the
	// retired image (a barrier on a VkImage EraseRetired may destroy before it runs). Plain data on the GPU thread, one
	// pass over the segment's images. (The null image is never registered: exempt.)
	if (Spec::SetCpuDirtyStopOn())
		for (const auto& use: segment.images) {
			const auto* image = use.image;
			if (image->serial == use.serial &&
			    (image->registered || (image->info.data.address == 0 && image->info.data.size == 0)))
				continue;
			LiveCounters::Add(LiveCounters::SpecImageUnregisteredStops);
			return "commit: an image it uses was unregistered";
		}
	if (textures.ResolutionEpoch() == epochs.images) return nullptr;
	// (Live spec_exact_image_stop: the registration log's exact ranges; -1, the default: on under KYTY_SPECULATE=4; 1 on,
	// 0 off.)
	static auto& exact_image = LiveSwitch::Find("spec_exact_image_stop", -1);
	const auto   exact_v     = exact_image.load(std::memory_order_relaxed);
	const bool   exact_on    = exact_v > 0 || (exact_v < 0 && kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4);
	const auto   resolved    = [&](const auto& image, uint64_t serial) {
		return exact_on ? textures.StillResolvedExact(image, serial, epochs.images)
		                : textures.StillResolved(image, serial, epochs.images);
	};
	for (const auto& use: segment.images)
		if (!resolved(*use.image, use.serial))
			return "commit: images registered";
	for (size_t i = from.touched_images; i < segment.end.touched_images; ++i)
		if (!resolved(*state.touched_images[i].first, state.touched_images[i].second))
			return "commit: images registered";
	return nullptr;
}

// redo_level_precise: whether a used image's unmap since the speculation began left the levels the segment sampled mapped,
// so that what its work read of the image is what the in-order path would read. Only when every access the segment made
// to it was a sampled view with a base level above 0 (ImageUse::sample_other clear: Image::GetBarriers records every
// transition with an entry, and a speculation transits every image it uses each time), on a single-layer, non-volume
// colour image with no metadata, its object the one the segment used (serial). The bytes are ViewLevelBytes': the
// sampled levels' spans, grown over every level that shares a byte with them (the mip tail).
bool Speculation::SampledLevelsMapped(const Spec::ImageUse& use, uint64_t mapping) const {
	const auto* image = use.image;
	if (use.sample_other || use.sample_lo == 0 || use.sample_lo >= use.sample_hi || image->serial != use.serial) return false;
	const auto& info   = image->info;
	const auto  levels = info.resources.levels;
	if (info.resources.layers != 1 || info.IsVolume() || info.IsDepth() || info.HasMetadata() ||
	    levels > info.mip_layout.size() || use.sample_hi > levels)
		return false;
	uint64_t first = UINT64_MAX, last = 0;
	for (uint32_t level = use.sample_lo; level < use.sample_hi; ++level) {
		first = std::min(first, info.mip_layout[level].offset);
		last  = std::max(last, info.mip_layout[level].offset + info.mip_layout[level].size);
	}
	for (bool grew = true; grew;) {
		grew = false;
		for (uint32_t level = 0; level < levels; ++level) {
			const auto& mip = info.mip_layout[level];
			if (mip.size == 0 || mip.offset >= last || mip.offset + mip.size <= first) continue;
			if (mip.offset < first || mip.offset + mip.size > last) {
				first = std::min(first, mip.offset);
				last  = std::max(last, mip.offset + mip.size);
				grew  = true;
			}
		}
	}
	last = std::min(last, info.data.size);
	if (first >= last) return false;
	return !m_renderer.GetGpuResources().UnmappedSince(mapping, info.data.address + first, last - first);
}

static bool ReadGuestNow(void* out, uint64_t address, uint64_t bytes); // (below)

// Whether a clean read of segment `index` that a write was logged over since the speculation began still gives the
// translation's bytes. Only when: no GPU write is logged over it (a GPU write is logged as its work is recorded, before
// it reaches memory: memory cannot tell); none of its bytes was read without keeping them (unvalued_reads: command
// words, the buffer cache's, the kernel's, the table path's readers, a copy's source, reads over MaxBytes); every byte of
// it is in a kept read of the segment; and memory now (as the translation read it: the write guard's backing view where
// a guest write is pending; on Linux a read-watched page refused, the rest through the backing view; else a guarded
// copy, a fault refusing) holds every kept read's bytes. Then the translation, which depended only on those bytes, is
// the one the GPU thread would make now: the write wrote what was there, or changed it back. Late reads keep their
// rule.
bool Speculation::ReadValuesHold(const Result& result, size_t index, const Spec::Range& read) const {
	const auto& state = result.state;
	if (!state.read_values_on) return false;
	const auto  from = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
	const auto& to   = state.closed[index].end;
	const auto  size = read.end - read.begin;
	if (m_log.FirstGpuWriteOver(result.effects_from, read.begin, size) != nullptr) return false;
	// (The segment's kept reads sorted by address and its unvalued reads sorted and merged, once, at its first read
	// checked; the thread that commits alone.)
	thread_local struct {
		uint64_t                            id = 0, index = 0, count = 0, unvalued_count = 0;
		std::vector<const Spec::ReadValue*> sorted;
		std::vector<Spec::Range>            unvalued;
	} kept;
	if (kept.id != state.id || kept.index != index || kept.count != to.read_values - from.read_values ||
	    kept.unvalued_count != to.unvalued_reads - from.unvalued_reads) {
		kept.id             = state.id;
		kept.index          = index;
		kept.count          = to.read_values - from.read_values;
		kept.unvalued_count = to.unvalued_reads - from.unvalued_reads;
		kept.sorted.clear();
		for (size_t i = from.read_values; i < to.read_values; ++i) kept.sorted.push_back(&state.read_values[i]);
		std::sort(kept.sorted.begin(), kept.sorted.end(),
		          [](const Spec::ReadValue* a, const Spec::ReadValue* b) { return a->address < b->address; });
		kept.unvalued.assign(state.unvalued_reads.begin() + static_cast<ptrdiff_t>(from.unvalued_reads),
		                     state.unvalued_reads.begin() + static_cast<ptrdiff_t>(to.unvalued_reads));
		std::sort(kept.unvalued.begin(), kept.unvalued.end(),
		          [](const Spec::Range& a, const Spec::Range& b) { return a.begin < b.begin; });
		size_t merged = 0;
		for (const auto& range: kept.unvalued) {
			if (merged != 0 && range.begin <= kept.unvalued[merged - 1].end)
				kept.unvalued[merged - 1].end = std::max(kept.unvalued[merged - 1].end, range.end);
			else
				kept.unvalued[merged++] = range;
		}
		kept.unvalued.resize(merged);
	}
	// (No unvalued read over it: the first merged range ending past its start begins past its end.)
	const auto ends_after = [](uint64_t at, const Spec::Range& range) { return at < range.end; };
	if (const auto unvalued = std::upper_bound(kept.unvalued.begin(), kept.unvalued.end(), read.begin, ends_after);
	    unvalued != kept.unvalued.end() && unvalued->begin < read.end)
		return false;
	// (Every byte of the read in a kept read: the kept reads over it, by address, without a gap; those begin at most
	// MaxBytes before it.)
	thread_local std::vector<const Spec::ReadValue*> over;
	over.clear();
	const uint64_t lowest = read.begin > Spec::ReadValue::MaxBytes ? read.begin - Spec::ReadValue::MaxBytes : 0;
	for (auto it = std::lower_bound(kept.sorted.begin(), kept.sorted.end(), lowest,
	                                [](const Spec::ReadValue* value, uint64_t at) { return value->address < at; });
	     it != kept.sorted.end() && (*it)->address < read.end; ++it)
		if (read.begin < (*it)->address + (*it)->size) over.push_back(*it);
	uint64_t covered = read.begin;
	for (const auto* value: over)
		if (value->address <= covered) covered = std::max(covered, value->address + value->size);
	if (covered < read.end) return false;
	for (const auto* value: over) {
		std::array<uint8_t, Spec::ReadValue::MaxBytes> now;
		if (!ReadGuestNow(now.data(), value->address, value->size)) return false;
		if (std::memcmp(now.data(), value->bytes.data(), value->size) != 0) return false;
	}
	return true;
}

// KYTY_SPEC_COMMIT_VERIFY: whether this one is the Nth (the thread that commits).
static bool CommitVerifyDue() {
	static uint64_t count = 0;
	const auto      every = Spec::CommitVerifyEvery();
	return every > 0 && ++count % static_cast<uint64_t>(every) == 0;
}

// Relocatable buffer references (KYTY_SPECULATE=4): see speculation.h. A range "moved" as Changed() finds it (the
// region's registrations since the speculation began, and with spec_exact_reg_stop its granules'); covered when the
// relocatable references' ranges of the segment cover it whole; each such reference resolves when one live registered
// game buffer covers its range.
bool Speculation::RelocEligible(const Result& result, size_t index, const char* why) const {
	const auto& state = result.state;
	if (!state.reloc_on || why == nullptr || (std::strcmp(why, "commit: buffers registered") != 0 &&
	                                         std::strcmp(why, "commit: preparation registered buffers") != 0))
		return false;
	const auto  from = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
	const auto& to   = state.closed[index].end;
	if (to.reloc_refs == from.reloc_refs) return false;
	auto&        buffers    = m_renderer.GetBufferCache();
	static auto& exact_stop = LiveSwitch::Find("spec_exact_reg_stop", -1);
	const bool   exact_on   = exact_stop.load(std::memory_order_relaxed) != 0; // (mode 4: reloc_on)
	const auto   moved      = [&](const Spec::Range& range) { // (Changed()'s own predicate)
		const auto size = range.end - range.begin;
		return buffers.RegisteredSince(range.begin, size, result.epochs.buffers) &&
		       (!exact_on || buffers.RegisteredSinceExact(range.begin, size, result.epochs.buffers));
	};
	// (The other references: allowed only when no registration moved over any of their ranges.)
	for (size_t i = from.reloc_unrel; i < to.reloc_unrel; ++i)
		if (moved(state.reloc_unrel[i])) return false;
	thread_local std::vector<std::pair<uint64_t, uint64_t>> covered;
	covered.clear();
	for (size_t i = from.reloc_refs; i < to.reloc_refs; ++i) {
		const auto& ref  = state.reloc_refs[i];
		const auto* live = buffers.GetBufferIfLive(buffers.RegisteredBuffer(ref.begin, ref.end - ref.begin));
		if (live == nullptr || !live->IsInBounds(ref.begin, ref.end - ref.begin)) return false;
		covered.emplace_back(ref.begin, ref.end);
	}
	std::sort(covered.begin(), covered.end());
	const auto whole = [&](uint64_t begin, uint64_t end) {
		auto at = begin;
		for (const auto& [b, e]: covered) {
			if (b > at) break;
			at = std::max(at, e);
			if (at >= end) return true;
		}
		return at >= end;
	};
	for (size_t i = from.buffer_ranges; i < to.buffer_ranges; ++i) {
		const auto& range = state.buffer_ranges[i];
		if (moved(range) && !whole(range.begin, range.end)) return false;
	}
	return true;
}

// A segment's relocatable references let go of their cells once its chunks are spliced or discarded (a passive
// producer's chunk goes back to its pool after its replay or its discard, and is reused: RelocPatch's null check then
// stops the segment rather than write through one).
static void RelocCellsRelease(Spec::State& state, size_t index) {
	const auto from = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
	const auto to   = state.closed[index].end;
	for (size_t i = from.reloc_refs; i < to.reloc_refs && i < state.reloc_refs.size(); ++i) state.reloc_refs[i].cell = nullptr;
}

const char* Speculation::RelocPatch(Result& result, size_t index) {
	auto&       state   = result.state;
	auto&       buffers = m_renderer.GetBufferCache();
	const auto  from    = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
	const auto& to      = state.closed[index].end;
	for (size_t i = from.reloc_refs; i < to.reloc_refs; ++i) {
		const auto& ref  = state.reloc_refs[i];
		const auto  id   = buffers.RegisteredBuffer(ref.begin, ref.end - ref.begin);
		const auto* live = buffers.GetBufferIfLive(id);
		if (live == nullptr || !live->IsInBounds(ref.begin, ref.end - ref.begin) || ref.cell == nullptr)
			return "commit: a relocated buffer reference unresolved";
		// The live buffer holds the old one's bytes only when they were copied (a join, a defragmentation or split
		// move): a collector's retire copies none and leaves the pages CPU-dirty, and a buffer made later over the range
		// uploads only its request. Before made current what the segment noted and the holes' writes, so CPU-dirty
		// memory here is bytes the live buffer may not hold. (Read only: no region manager made, an untracked region
		// answers dirty.) Its dirty pages are uploaded into the live buffer here, as the in-order draw's own lookup of
		// the range would sync them before binding it (BufferCache::UploadRelocated): into the open command buffer
		// (after the joins' copies, which mark the buffer so no upload prologue runs ahead of them) or the prologue
		// ahead of it, both before the segment's work, which SubmitRecorded splices after. Refused: stopped as before.
		if (buffers.IsRegionCpuDirtyNoCreate(ref.begin, ref.end - ref.begin)) {
			const Spec::EffectSourceScope source(Spec::EffectSource::Commit);
			const RenderLockGuard         lock(m_renderer.GetMutex());
			if (buffers.UploadRelocated(id, ref.begin, ref.end - ref.begin) != BufferCache::RelocUpload::Uploaded)
				return "commit: buffers registered";
		}
	}
	for (size_t i = from.reloc_refs; i < to.reloc_refs; ++i) {
		const auto& ref  = state.reloc_refs[i];
		const auto  id   = buffers.RegisteredBuffer(ref.begin, ref.end - ref.begin);
		const auto* live = buffers.GetBufferIfLive(id);
		ref.cell->buffer = reinterpret_cast<uint64_t>(static_cast<VkBuffer>(live->Handle())); // NOLINT
		ref.cell->offset = live->Offset(ref.begin);
		buffers.TouchSpeculated(id);
	}
	return nullptr;
}

static bool SegmentWritesRange(const Spec::State& state, size_t index, const Spec::Counts& from, const Spec::Counts& to,
                               uint64_t address, uint64_t size); // (below)

Speculation::Progress Speculation::CommitSegments(Result& result, CommandProcessor& cp) {
	m_renderer.GetBufferCache().ServeAhead(); // (spec_buffer_ahead: the ranges speculations found no buffer over)
	auto&       scheduler = m_renderer.GetCommandScheduler();
	const auto& state     = result.state;
	const auto  segments  = state.closed.size();
	EXIT_IF(result.holes.size() + 1 != segments);
	// (Blocked at a wait hole: from there.)
	// (After its segment ran as a hole, `cp` is already in the state the hole runs from: not the worker's copied.)
	if (const auto hole = std::exchange(result.blocked, SIZE_MAX); hole != SIZE_MAX) {
		const bool as_hole = std::exchange(result.blocked_as_hole, false);
		const auto run     = RunHole(result, cp, hole, !as_hole);
		if (run == HoleRun::Blocked) return result.blocked = hole, result.blocked_as_hole = as_hole, Progress::Blocked;
		if (run == HoleRun::Changed) return Progress::Stopped;
	}
	while (result.next < segments) {
		const auto index = result.next;
		m_effects.Take(m_log);
		// The checks after Changed(): the stop's reason, or null when they all hold. (Under KYTY_SPECULATE=4 a stop here
		// runs the segment as a hole, as Changed()'s do: nothing of its preparation ran, so the run is the in-order
		// path's from the true state.)
		const auto post_checks = [&]() -> const char* {
			// The images it uses as it found them: none written through a buffer since (a hole's storage-buffer write, fill or
			// copy over its memory, or an earlier segment's, replayed in its Before). The normal path's binding uploads such an
			// image from those bytes before the work; this work was recorded without the upload and would read it as it was
			// (the game's probe input, 0x27ce00000 at the tester's spot, written per face by a refused dispatch: the dark).
			// Under KYTY_SPECULATE=4 such an image is refreshed in the segment's Before instead, as the normal path's binding
			// refreshes it (TextureCache::RefreshAtCommit -> RefreshImage: the buffer-written bytes uploaded, or the fill's
			// clear), after the texel syncs, invalidations and pending targets, before the restore loop. Exact:
			// - the work read the image's identity (the VkImage its descriptors name; RefreshImage keeps it) and its contents
			//   at execution; the writes that made it buffer-modified were made before the segment (holes, earlier segments'
			//   Befores, in-order work), the ones the in-order binding sees. Nothing between Before and the segment's uses
			//   writes that memory again: refused (the stop as before) when the segment's own work writes over the image (its
			//   invalidated ranges, the GPU writes replayed in its Before; its written ranges; its journal's writes). Every use
			//   of the image in the segment is at or after its first binding, where in order the refresh is made;
			// - the same image: serial and registration (here, and RefreshAtCommit again under the cache's lock, by its
			//   slot); not CPU-dirty or maybe-dirty, not stencil-modified (their stops stay), uncompressed, single-sampled
			//   (RefreshImage uploads neither);
			// - the layout: the restore loop puts it back in ImageUse::found, the state the recorded barriers start from,
			//   from whatever the refresh left. A use found Undefined (or at any subresource) is refused: that barrier would
			//   discard the upload. (An image a set of the segment completed in this Before: the restore loop's no-barrier
			//   keep requires it still in its completion's layout and access, which the refresh's upload moves: the segment
			//   stops there.)
			// - a refresh that fails in Before (not that image any more, or still buffer-modified) fails the segment as a
			//   target not acquired does (target_failed: stopped before the splice); what it uploaded is the in-order path's
			//   own upload of those bytes.
			bool image_refresh = false;
			for (const auto& use: state.closed[index].images)
				if (!use.found_buffer_modified && use.image->IsBufferModified()) {
					if (kyty_local_speculate_mode.load(std::memory_order_relaxed) != 4)
						return "commit: GPU work wrote an image it uses";
					const auto  from_c = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
					const auto& to_c   = state.closed[index].end;
					const auto& image  = *use.image;
					const auto  begin = image.info.data.address, end = image.info.data.End();
					const auto  over  = [&](const std::vector<Spec::Range>& list, size_t a, size_t b) {
						for (size_t k = a; k < b; ++k)
							if (list[k].begin < end && begin < list[k].end) return true;
						return false;
					};
					bool journal_over = false;
					for (size_t k = from_c.writes; k < to_c.writes && !journal_over; ++k)
						journal_over = state.writes[k].address < end && begin < state.writes[k].address + state.writes[k].size;
					const auto undefined       = [](const VulkanImageState& s) { return s.layout == vk::ImageLayout::eUndefined; };
					const bool found_undefined = use.found_subresources.empty() ? undefined(use.found)
					                                                            : std::ranges::any_of(use.found_subresources, undefined);
					if (image.serial != use.serial || !image.registered || image.IsCpuDirty() || image.IsStencilModified() ||
					    image.info.metadata.compression != VideoOutCompression::Uncompressed || image.info.samples > 1 ||
					    found_undefined || over(state.invalidated, from_c.invalidated, to_c.invalidated) ||
					    over(state.written, from_c.written, to_c.written) || journal_over)
						return "commit: GPU work wrote an image it uses";
					image_refresh = true;
				}
			if (image_refresh)
				for (auto& use: result.state.closed[index].images)
					use.refresh = !use.found_buffer_modified && use.image->IsBufferModified();
			// spec_copy_image_refuse (Spec::CopyImageRefuseOn): nor its stencil plane (MarkStencilModified: a buffer write over
			// it), which the normal path's refresh uploads (RefreshImage) and this work was recorded without.
			if (Spec::CopyImageRefuseOn())
				for (const auto& use: state.closed[index].images)
					if (!use.found_stencil_modified && use.image->IsStencilModified()) {
						LiveCounters::Add(LiveCounters::SpecStencilModifiedStops);
						return "commit: GPU work wrote a stencil it uses";
					}
			// spec_set_cpu_dirty_stop (Spec::SetCpuDirtyStopOn): nor written on the CPU since (a hole's host write over its
			// pages, or an earlier segment's guest memory writes). The work's set reuse checked it only at translation
			// (TableTransit's and TableValidateSet's refresh tests, which make its entry first); the normal path's next
			// acquisition uploads (a storage image: guest bytes over what the GPU wrote). CPU-dirty now and either clean when
			// found or written since (its validity moved: every CPU write's does). A use that only sampled views above level 0
			// (redo_level_precise's sample-only use: a texture streaming in from its small levels) passes while its sampled
			// levels' bytes are not dirty: RefreshPending of those levels (DirtyForView), the normal path's own test for such a
			// view. Maybe-dirty only: when its refresh would upload, the stored hash of its guest edges no longer theirs
			// (RefreshImage's own test, read only).
			if (Spec::SetCpuDirtyStopOn())
				for (const auto& use: state.closed[index].images) {
					const auto* image = use.image;
					if (!image->IsCpuDirty() ||
					    (use.found_cpu_dirty && image->validity.load(std::memory_order_acquire) == use.found_validity))
						continue;
					if (!use.sample_other && use.sample_lo != 0 && use.sample_lo < use.sample_hi && image->serial == use.serial) {
						TextureCache::ImageDesc sampled;
						sampled.type                  = TextureCache::BindingType::Texture;
						sampled.view_info.base_level  = use.sample_lo;
						sampled.view_info.level_count = use.sample_hi - use.sample_lo;
						if (!TextureCache::RefreshPending(*image, sampled)) continue;
					}
					if (image->IsDefinitelyCpuDirty()) {
						LiveCounters::Add(LiveCounters::SpecCpuDirtyStopsDefinite);
						return "commit: the CPU wrote an image it uses";
					}
					if (image->MaybeCpuHashMoved()) {
						LiveCounters::Add(LiveCounters::SpecCpuDirtyStopsMaybe);
						return "commit: the CPU wrote an image it uses (maybe-dirty)";
					}
				}
			for (const auto& left: state.left_clears)
				if (left.segment == index && (left.image->serial != left.serial ||
				                              left.image->validity.load(std::memory_order_relaxed) != left.validity))
					return "commit: a left clear's image moved";
			for (const auto& fact: state.meta_facts)
				if (!fact.cleared && fact.first <= index && index <= fact.last &&
				    m_renderer.GetTextureCache().IsMetaCleared(fact.address, fact.slice))
					return "commit: a surface cleared";
			{
				const auto  from    = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
				const auto& to      = state.closed[index].end;
				const auto& effects = m_effects;
				// The guest memory it read on the CPU as no GPU work wrote it (and as it was): not written since.
				// (Stamped: a page written since is a candidate, the log's entries since over the read decide.)
				const bool any_since = m_log.End() > result.effects_from;
				if (effects.stamped) {
					// (KYTY_SPEC_WALK_FROM_STAMP: the walk tries the entry the read's latest stamp names first;
					// KYTY_SPEC_COMMIT_VERIFY: every Nth such walk, the walk from the speculation's start as well.)
					if (any_since) {
						const bool from_stamp = Spec::WalkFromStampOn();
						for (size_t i = from.clean_reads; i < to.clean_reads; ++i) {
							const auto& read = state.clean_reads[i];
							const auto  size = read.end - read.begin;
							if (!m_log.stamps.MaybeSince(read.begin, size, result.effects_from)) continue;
							const Spec::EffectsLog::Entry* write = nullptr;
							if (from_stamp) {
								bool direct = false;
								write       = m_log.WriteOverFromStamp(result.effects_from, read.begin, size, direct);
								if (direct) LiveCounters::Add(LiveCounters::SpecWalkDirect);
								if (CommitVerifyDue()) {
									const bool first = m_log.FirstWriteOver(result.effects_from, read.begin, size) != nullptr;
									LiveCounters::Add(LiveCounters::SpecCommitVerified);
									if (first != (write != nullptr)) LiveCounters::Add(LiveCounters::SpecCommitMismatch);
								}
							} else {
								write = m_log.FirstWriteOver(result.effects_from, read.begin, size);
							}
							if (write != nullptr && !ReadValuesHold(result, index, read))
								return WroteWhatItRead(write);
						}
					}
					// (KYTY_SPEC_LATE_READS: a late read, by the processor's writes logged after it was read, and GPU work's since
					// the speculation began: a GPU write is logged as its work is recorded, before it reaches memory, and its
					// page's read protection may be deferred, PageManager::DeferReadProtection.)
					if (any_since)
						for (size_t i = from.late_reads; i < to.late_reads; ++i) {
							const auto& read  = state.late_reads[i];
							const auto  bytes = read.end - read.begin;
							if (!m_log.stamps.MaybeSince(read.begin, bytes, result.effects_from)) continue;
							if (const auto* gpu = m_log.FirstGpuWriteOver(result.effects_from, read.begin, bytes))
								return WroteWhatItRead(gpu);
							const auto since = std::max(result.effects_from, read.since);
							if (m_log.End() <= since || !m_log.stamps.MaybeSince(read.begin, bytes, since)) continue;
							if (const auto* write = m_log.FirstWriteOver(since, read.begin, bytes))
								return WroteWhatItRead(write);
						}
				} else if (!effects.gpu_writes.Empty() || !effects.host_writes.Empty()) {
					for (size_t i = from.clean_reads; i < to.clean_reads; ++i) {
						const auto& read = state.clean_reads[i];
						if (effects.gpu_writes.Intersects(read.begin, read.end - read.begin) ||
						    (effects.host_writes.Intersects(read.begin, read.end - read.begin) &&
						     !ReadValuesHold(result, index, read)))
							return WroteWhatItRead(m_log.FirstWriteOver(result.effects_from, read.begin, read.end - read.begin));
					}
					// (Late reads as clean ones here: every write since it began.)
					for (size_t i = from.late_reads; i < to.late_reads; ++i) {
						const auto& read = state.late_reads[i];
						if (effects.gpu_writes.Intersects(read.begin, read.end - read.begin) ||
						    effects.host_writes.Intersects(read.begin, read.end - read.begin))
							return WroteWhatItRead(m_log.FirstWriteOver(result.effects_from, read.begin, read.end - read.begin));
					}
				}
				// The descriptor sets it binds: none retired since (one may be reused under it).
				for (const auto retired: effects.retired_sets)
					if (std::find(state.sets.begin() + static_cast<ptrdiff_t>(from.sets),
					              state.sets.begin() + static_cast<ptrdiff_t>(to.sets), retired) !=
					    state.sets.begin() + static_cast<ptrdiff_t>(to.sets))
						return "commit: a hole retired a set";
				// spec_set_cpu_dirty_stop off: the guards that keep a set with pending slots an earlier segment proposed from
				// being bound again (TableResolveSet, TableTransit, TableValidateSet) are off, so this segment may bind a set an
				// earlier segment's commit completed (its Before, before this segment's writes). Each completed slot's image
				// (its data, metadata range, stencil plane) is tested against this segment's writes, the whole segment
				// (SegmentWritesRange: its Before applies them after these checks, its replay and After after that): one written
				// stops it, as one no longer the image completed (erased, or its serial moved: its memory unknown). Only the sets
				// it binds (State::sets; ProposedSet::completed, recorded with the switch off).
				if (!Spec::SetCpuDirtyStopOn() && to.sets > from.sets) {
					auto&      textures    = m_renderer.GetTextureCache();
					const auto bound_begin = state.sets.begin() + static_cast<ptrdiff_t>(from.sets);
					const auto bound_end   = state.sets.begin() + static_cast<ptrdiff_t>(to.sets);
					const auto written     = [&](uint64_t address, uint64_t size) {
						return SegmentWritesRange(state, index, from, to, address, size);
					};
					for (const auto& proposed: state.proposed_sets) {
						if (proposed.segment >= index || proposed.completed.empty() ||
						    std::find(bound_begin, bound_end, proposed.completed.front().set) == bound_end)
							continue;
						for (const auto& slot: proposed.completed) {
							if (!textures.LiveImageId(*slot.image) || slot.image->serial != slot.serial)
								return "commit: a completed pending image moved";
							const auto& info = slot.image->info;
							if (written(info.data.address, info.data.size) ||
							    (info.HasMetadata() && written(info.metadata.range.address, info.metadata.range.size)) ||
							    (info.HasStencil() && written(info.stencil.address, info.stencil.size)))
								return "commit: a completed pending image's memory written by its segment";
						}
					}
				}
				// The sources and destinations of the copies it makes on the CPU at its commit: no GPU work wrote them since.
				// (A GPU write logged since it began over a range: the copies' test.)
				const auto copy_written = [&](uint64_t begin, uint64_t size) {
					if (effects.stamped)
						return any_since && m_log.stamps.MaybeSince(begin, size, result.effects_from) &&
						       m_log.FirstGpuWriteOver(result.effects_from, begin, size) != nullptr;
					return !effects.gpu_writes.Empty() && effects.gpu_writes.Intersects(begin, size);
				};
				bool copies_written = false;
				for (size_t i = from.cpu_copies; i < to.cpu_copies && !copies_written; ++i)
					copies_written = copy_written(state.cpu_copies[i].begin, state.cpu_copies[i].end - state.cpu_copies[i].begin);
				if (copies_written) {
					// Under KYTY_SPECULATE=4 each such copy is made in After as the in-order path makes it now
					// (BufferCache::CopyBuffer on the GPU thread, in the journal's place: WriteLabels), instead of the stop.
					// Exact:
					// - the translation read none of the copy's bytes: it decided the CPU path from the tracker. At the commit,
					//   on the GPU thread in queue order, CopyBuffer re-decides as in order: a range GPU-modified (or an image
					//   over the source) takes the GPU copy, which reads the bytes at execution, after the write. One whose
					//   ranges the tracker no longer holds GPU-modified is made as the processor copy, as before;
					// - its place: recorded after the segment's work (WriteLabels flushes it first), in the journal's order.
					//   Refused (the stop as before) when the segment's own work reads or writes the source or destination
					//   (its current, written, buffer, table or texel ranges: that work would then run before the copy
					//   instead of after it), or the destination is image memory (HasTrackedDataOverlap);
					// - in the journal's order, a later copy over a GPU copy's source or destination is made as the GPU copy
					//   too (in order CopyBuffer would see the range GPU-modified), and a processor write (a label) over one
					//   stops the segment (it would land before the GPU copy's execution, not after it);
					// - later segments of the chunk: the GPU path's own marking (InvalidateMemoryFromGPU, ObtainBuffer
					//   written: the tracker's GPU-modified range, the copy feedback invalidated, Spec::NoteGpuWrite into the
					//   effects log) makes a later CPU-side read of the destination stop on "GPU work wrote what it read";
					//   the destination's guest memory stays as in order (stale until the copy feedback).
					if (kyty_local_speculate_mode.load(std::memory_order_relaxed) != 4)
						return "commit: GPU work wrote what it copies";
					const auto own = [&](uint64_t begin, uint64_t end) {
						const auto over = [&](const std::vector<Spec::Range>& list, size_t a, size_t b) {
							for (size_t k = a; k < b; ++k)
								if (list[k].begin < end && begin < list[k].end) return true;
							return false;
						};
						return over(state.current, from.current, to.current) || over(state.written, from.written, to.written) ||
						       over(state.buffer_ranges, from.buffer_ranges, to.buffer_ranges) ||
						       over(state.table_ranges, from.table_ranges, to.table_ranges) ||
						       over(state.texel_syncs, from.texel_syncs, to.texel_syncs);
					};
					// (A journaled copy GPU work wrote: its source or destination written since. Every cpu_copies range is a
					// journaled copy's source or destination, noted with it.)
					const auto noted = [&](const Spec::MemoryWrite& write) {
						return write.copy && (copy_written(write.address, write.size) || copy_written(write.value, write.size));
					};
					auto&                                      textures = m_renderer.GetTextureCache();
					std::vector<std::pair<uint64_t, uint64_t>> touched; // (the GPU copies' sources and destinations, [begin, end))
					std::vector<size_t>                        marked;
					const auto over_touched = [&](uint64_t begin, uint64_t size) {
						for (const auto& [b, e]: touched)
							if (b < begin + size && begin < e) return true;
						return false;
					};
					for (size_t i = from.writes; i < to.writes; ++i) {
						const auto& write   = state.writes[i];
						const bool  follows = over_touched(write.address, write.size) || (write.copy && over_touched(write.value, write.size));
						if (!write.copy) {
							if (follows) return "commit: GPU work wrote what it copies";
							continue;
						}
						if (!follows && !noted(write)) continue;
						if (own(write.address, write.address + write.size) || own(write.value, write.value + write.size) ||
						    textures.HasTrackedDataOverlap(write.address, write.size))
							return "commit: GPU work wrote what it copies";
						marked.push_back(i);
						touched.emplace_back(write.address, write.address + write.size);
						touched.emplace_back(write.value, write.value + write.size);
					}
					for (const auto i: marked) result.state.writes[i].gpu = true;
				}
				for (size_t i = from.chain_reads; i < to.chain_reads; ++i) {
					const auto& read = state.chain_reads[i];
					// (KYTY_WRITE_GUARD_SPEC: a page a pending guest write guards, through the backing view: wg2's census had this
					// read fault ~1 a frame.)
					const void* word = reinterpret_cast<const void*>(read.address);
					if (WriteGuard::g_count.load(std::memory_order_relaxed) != 0)
						if (const auto* backing = WriteGuard::GuardedBacking(read.address, 4u)) word = backing;
					if (*static_cast<const uint32_t*>(word) != read.value)
						return "commit: a command buffer before wrote otherwise";
				}
				// The words it read from its own speculation's journal that an earlier segment's write gave, as the chain
				// reads (KYTY_SPECULATE=4: JournalReadsMoved).
				if (state.journal_reads_on && to.journal_reads > from.journal_reads)
					if (const char* moved = JournalReadsMoved(result, index))
						return moved;
				for (size_t i = from.waits; i < to.waits; ++i) {
					const auto& wait  = state.waits[i];
					uint64_t    value = 0;
					const void* from  = reinterpret_cast<const void*>(wait.address);
					if (WriteGuard::g_count.load(std::memory_order_relaxed) != 0)
						if (const auto* backing = WriteGuard::GuardedBacking(wait.address, wait.bytes)) from = backing;
					std::memcpy(&value, from, wait.bytes);
					if (!TestWaitRegMemValue(value, wait.reference, wait.mask, wait.function))
						return "commit: a wait not satisfied";
				}
			}
			// A completion uploads its images' bytes, metadata and stencil plane in Before, before the segment's own writes
			// over them execute (the replay, After), and leaves them clean: in order a write before the bind is uploaded
			// after it, and one after the bind marks the image again. So a set its Before would complete whose pending
			// slots' memory the segment writes (the whole segment: SegmentWritesRange) stops it here, before anything of
			// its Before ran: run as a hole under KYTY_SPECULATE=4, as the other stops here. (Under the render lock, as
			// Before reads the sets and images.)
			if (std::ranges::any_of(state.proposed_sets, [&](const auto& proposed) {
				    return proposed.segment == index && proposed.complete && proposed.pending_written;
			    })) {
				const RenderLockGuard lock(m_renderer.GetMutex());
				const auto            from           = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
				const auto&           to             = state.closed[index].end;
				const auto            segment_writes = [&](uint64_t address, uint64_t size) {
					return SegmentWritesRange(state, index, from, to, address, size);
				};
				for (const auto& proposed: state.proposed_sets)
					if (proposed.segment == index && proposed.complete && proposed.pending_written &&
					    proposed.pending_written(segment_writes))
						return "commit: a pending image's memory written by its segment";
			}
			return nullptr;
		};
		// What the work assumed: nothing registered or released since; the surface metadata it found not cleared not
		// cleared since; the guest memory it read as no GPU work had written it not written since (the effects
		// logged since it began); the words it read from a speculation before it in memory now; the waits it found
		// satisfied satisfied.
		// (A segment stopped only for registrations over ranges its relocatable references cover passes; they are
		// patched after its Before.)
		bool        reloc       = false;
		const char* changed_why = Changed(result, index);
		if (changed_why != nullptr && RelocEligible(result, index, changed_why)) {
			changed_why = Changed(result, index, true);
			reloc       = changed_why == nullptr;
		}
		const char* why  = changed_why;
		const bool  post = why == nullptr && (why = post_checks()) != nullptr; // (a stop after Changed())
		if (why != nullptr) {
			// (spec_stop_as_hole: the segment run the normal way as a hole, its recorded work dropped; the hole after it as
			// after any segment. Under KYTY_SPECULATE=4 (journal_reads_on, as the state was made) the stops after Changed()
			// too, before Before: nothing of the segment's preparation ran, so the run is the in-order path's from the
			// true state, as for Changed()'s kinds; the segments after it are still checked against the effects log, which
			// holds the run's effects, and their journal words. A stop then costs its own packets, not the rest of the
			// chunk.)
			if (post && !state.journal_reads_on) return ++m_stop_reasons[why], Progress::Stopped;
			const auto as_hole = SegmentAsHole(result, cp, index);
			if (as_hole == AsHole::NotTaken) return ++m_stop_reasons[why], Progress::Stopped;
			result.next = index + 1;
			// (The GPU thread ran it: on the normal way from where it stopped, on its own state, when its run did not reach
			// the hole (the packets changed under the walk), or reached it in another state than the work after it assumed
			// (the worker's at the hole, which nothing checked: what Changed found moved may have moved a register).)
			if (as_hole == AsHole::NotReached)
				return ++m_stop_reasons["commit: a segment run as a hole did not reach its hole"], Progress::Stopped;
			if (!cp.SameState(*result.hole_states[index]))
				return ++m_stop_reasons["commit: a segment run as a hole ended in another state"], Progress::Stopped;
			// (Its hole's header as translated: RunHole reads the packet's words as they are now but runs it with the
			// translation's length, which the header holds. A write over what the segment read, or the run itself, may
			// have written the hole's words (a Changed() stop checked none of them). Read as the chain reads are,
			// through the backing view where a guest write is pending. A stop leaves the GPU thread at the hole's
			// packet, Result::resume_at.)
			{
				const auto& hole = result.holes[index];
				const void* word = hole.packet;
				if (WriteGuard::g_count.load(std::memory_order_relaxed) != 0)
					if (const auto* backing = WriteGuard::GuardedBacking(reinterpret_cast<uint64_t>(hole.packet), 4u))
						word = backing;
				if (*static_cast<const uint32_t*>(word) != hole.header)
					return ++m_stop_reasons["commit: a segment run as a hole: its hole packet moved"], Progress::Stopped;
			}
			result.resume_at.reset();
			// (Its Before did not run: the worker sets it proposed are the next segment's to check and complete before that
			// one's replay, as they would have been this one's. A later segment may bind any of them, through the reuse of
			// its own or the role's last / previous / memo set, so all move; one only the dropped replay bound may then
			// stop that segment's commit for nothing: the price of the rule.)
			for (auto& proposed: result.state.proposed_sets)
				if (proposed.segment == index) {
					proposed.segment = index + 1;
					if (proposed.on_hole) proposed.on_hole();
				}
			const auto run = RunHole(result, cp, index, false);
			if (run == HoleRun::Blocked) return result.blocked = index, result.blocked_as_hole = true, Progress::Blocked;
			if (run == HoleRun::Changed) return Progress::Stopped;
			continue;
		}
		const auto prepared = CurrentEpochs();
		Before(result, index);
		// (Its preparation registers what the work did not find registered (BDA ranges): a buffer merged away under
		// the recorded work leaves it reading a destroyed one. The work after it is translated in order.)
		if (CurrentEpochs() != prepared)
			if (const char* prepared_why = Changed(result, index)) {
				// (Before's own registrations over its relocatable references: the same rule.)
				if (!RelocEligible(result, index, prepared_why) || Changed(result, index, true) != nullptr)
					return ++m_stop_reasons["commit: preparation registered buffers"], Progress::Stopped;
				reloc = true;
			}
		// Its relocatable references written as they resolve now (after Before's registrations; the joins' copies are in
		// the open buffer, which SubmitRecorded submits before the segment's work).
		if (const char* why = reloc ? RelocPatch(result, index) : nullptr) return ++m_stop_reasons[why], Progress::Stopped;
		if (std::exchange(result.state.target_failed, false)) // (spec_commit_targets)
			return ++m_stop_reasons["commit: a pending target not acquired"], Progress::Stopped;
		if (std::exchange(result.state.images_failed, false)) // (spec_worker_images)
			return ++m_stop_reasons["commit: a pending image not completed"], Progress::Stopped;
		// spec_pending_buffers / spec_reloc_refs: the table V# entries, after Before's registrations. A pending one, one
		// Before had to obtain, and (spec_reloc_refs) one with a buffer registered or unregistered over its range since
		// its proof are looked up as they are now and written into the block; the others are left as they are (no
		// lookup). Every entry that needs it, or the segment stops before it is spliced: a block never reaches the GPU
		// unpatched.
		if (const auto pfrom = index == 0 ? size_t {0} : state.closed[index - 1].end.pending_buffers,
		    pto = state.closed[index].end.pending_buffers;
		    pto != pfrom) {
			auto&                pbuffers = m_renderer.GetBufferCache();
			size_t               patched = 0, required = 0;
			const auto           registration_now = pbuffers.RegistrationEpoch();
			const auto           now = [&](const Spec::PendingBuffer& entry) -> uint64_t {
                const auto* owner = pbuffers.GetBufferIfLive(pbuffers.RegisteredBuffer(entry.address, entry.size));
                return owner != nullptr && owner->HasDeviceAddress() ? owner->BufferDeviceAddress() + owner->Offset(entry.address)
				                                                                : 0;
			};
			for (size_t i = pfrom; i < pto; ++i) {
				const auto& entry = state.pending_buffers[i];
				const bool  moved = !entry.pending && entry.obtained && entry.since < registration_now &&
				                   pbuffers.RegisteredSince(entry.address, entry.size, entry.since) &&
				                   pbuffers.RegisteredSinceExact(entry.address, entry.size, entry.since);
				if (!entry.pending && entry.obtained && !moved) continue;
				++required;
				const auto device = entry.obtained ? now(entry) : 0;
				if (device == 0 || entry.ring == nullptr) continue;
				const auto mapped = entry.ring->Mapped();
				if (entry.offset + 3 * sizeof(uint32_t) > mapped.size()) continue;
				const uint32_t words[3] = {static_cast<uint32_t>(device), static_cast<uint32_t>(device >> 32u),
				                           static_cast<uint32_t>(std::min<uint64_t>(entry.size / 4, UINT32_MAX))};
				std::memcpy(mapped.data() + entry.offset, words, sizeof(words));
				entry.ring->Flush(entry.offset, sizeof(words));
				pbuffers.TouchSpeculated(pbuffers.RegisteredBuffer(entry.address, entry.size));
				++patched;
			}
			if (patched != required) return ++m_stop_reasons["commit: a pending buffer unpatched"], Progress::Stopped;
		}
		// KYTY_SPEC_DRAW_INDIRECT_VERIFY: its sampled DRAW_INDIRECT table draws, here where their arguments are current.
		for (const auto& entry: state.draw_verifies)
			if (entry.segment == index) VerifyDrawIndirect(m_renderer, entry);
		const auto& segment = state.closed[index];
		if (const auto tick = scheduler.SubmitRecorded(*result.recorder, segment.recorded - result.submitted); tick != 0)
			result.last_tick = tick;
		result.submitted = segment.recorded;
		RelocCellsRelease(result.state, index); // (its chunks spliced)
		// (Worker packets: spliced into the open buffer, which goes on: the draw state this thread knew of it is not
		// the buffer's any more. The recording worker's replay state is: the same buffer, replayed on the same thread.)
		if (result.recorder->packets) {
			DrawStateObserver::Invalidate();
			m_renderer.GetRenderExecutor().NativeXprForgetState();
		}
		After(result, index);
		if ((GuestGpu::ServeInWaits() & 2u) != 0) GuestGpu::ServePendingCommands(); // (KYTY_SERVE_IN_WAITS 2)
		m_renderer.GetBufferCache().WriteBackCompletedFeedback(false); // (KYTY_EAGER_FEEDBACK)
		result.next = index + 1;
		if (index < result.holes.size()) {
			const auto run = RunHole(result, cp, index);
			if (run == HoleRun::Blocked) return result.blocked = index, Progress::Blocked;
			if (run == HoleRun::Changed) return Progress::Stopped;
		}
	}
	return Progress::Complete;
}

// (Before: an image of the segment's entries as Before's start found it (`at_start`, taken before anything there can
// release one), and whether an image was erased since (`erased`: TextureCache::EraseEpoch moved).)
struct ImageAtBeforeStart {
	ImageId id;                 // (TextureCache::LiveImageId then: {} when it was erased already)
	bool    registered = false;
};

// (Before: an image of the segment's entries gone: erased (before Before or in it: its slot's generation moved), its
// slot another image's (the serial), or unregistered in this Before (ReclaimImageMemory takes registered images only:
// a victim a speculation's packet still holds is erased by a later EraseRetired, whatever the GPU still has recorded).
// Not one already unregistered when Before began and still alive (a parked image, one retired but not yet erased): its
// VkImage lives, and it gets the barrier as before. The liveness is read first: the fields of an erased object are not.)
static bool ImageGoneInBefore(TextureCache& textures, const Image& image, uint64_t serial,
                              const ImageAtBeforeStart& at_start, bool erased) {
	if (!at_start.id || (erased && textures.TryGetImage(at_start.id) != &image)) return true;
	return image.serial != serial || (at_start.registered && !image.registered);
}

// Whether segment `index`'s work (its counts [from, to)) writes [address, address + size), by what its commit's Before
// applies after a pending slot's completion: the ranges its GPU work writes (written: the buffers its shaders write;
// invalidated: the image-invalidating GPU writes), its guest memory writes (the journal, made in After: labels,
// WRITE_DATA words, copy destinations), the images its own work writes (Segment::images, ImageUse::writes: a colour or
// depth / stencil target, a storage image, a transfer, a clear; by the image's data, metadata range or stencil plane) and
// the targets its Before acquires after the completion (State::pending_targets: an upload or a clear).
static bool SegmentWritesRange(const Spec::State& state, size_t index, const Spec::Counts& from, const Spec::Counts& to,
                               uint64_t address, uint64_t size) {
	if (size == 0) return false;
	const auto meets = [&](uint64_t begin, uint64_t bytes) { return bytes != 0 && begin < address + size && address < begin + bytes; };
	const auto over  = [&](const std::vector<Spec::Range>& list, size_t a, size_t b) {
		for (size_t k = a; k < std::min(b, list.size()); ++k)
			if (list[k].begin < address + size && address < list[k].end) return true;
		return false;
	};
	if (over(state.written, from.written, to.written) || over(state.invalidated, from.invalidated, to.invalidated)) return true;
	for (size_t k = from.writes; k < std::min(to.writes, state.writes.size()); ++k)
		if (meets(state.writes[k].address, state.writes[k].size)) return true;
	const auto image_over = [&](const ImageInfo& info) {
		return meets(info.data.address, info.data.size) ||
		       (info.HasMetadata() && meets(info.metadata.range.address, info.metadata.range.size)) ||
		       (info.HasStencil() && meets(info.stencil.address, info.stencil.size));
	};
	for (const auto& use: state.closed[index].images)
		if (use.writes != 0 && use.image != nullptr && use.image->serial == use.serial && image_over(use.image->info))
			return true;
	for (size_t k = from.pending_targets; k < std::min(to.pending_targets, state.pending_targets.size()); ++k) {
		const auto& target = state.pending_targets[k];
		if (target.image != nullptr && target.image->serial == target.serial && image_over(target.image->info)) return true;
		if (target.depth_desc && image_over(static_cast<const TextureCache::ImageDesc*>(target.depth_desc.get())->info))
			return true;
	}
	return false;
}

void Speculation::Before(Result& result, size_t index) {
	const Spec::EffectSourceScope source(Spec::EffectSource::Commit);
	auto&       resources = m_renderer.GetGpuResources();
	auto&       buffers   = m_renderer.GetBufferCache();
	auto&       textures  = m_renderer.GetTextureCache();
	auto&       scheduler = m_renderer.GetCommandScheduler();
	auto&       state     = result.state;
	const auto& segment   = state.closed[index];
	const auto  from      = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
	const auto& to        = segment.end;
	BdaDirtyRegions::Publish();
	// In the scheduler's command buffer (submitted before the segment's): the ranges it writes obtained written
	// (their CPU-dirty pages uploaded, then GPU-owned), those it reads made current, its BDA reads prepared, and each
	// image where the work found it.
	RenderLockGuard lock(m_renderer.GetMutex());
	// Its images as they are before anything here can erase or unregister one (the last loop's ImageGoneInBefore).
	const auto                      erase_epoch = textures.EraseEpoch();
	std::vector<ImageAtBeforeStart> images_at_start;
	images_at_start.reserve(segment.images.size());
	for (const auto& use: segment.images) {
		const auto id = textures.LiveImageId(*use.image);
		images_at_start.push_back({id, id && use.image->registered});
	}
	// spec_pending_buffers: the V# ranges it found no buffer for, obtained in order first (made or joined, uploaded), so
	// the check after Before (preparation registered buffers) sees what that moved under the segment's other ranges.
	const auto registration_now = buffers.RegistrationEpoch(); // (an entry proven at it or after: nothing moved under it)
	for (size_t i = from.pending_buffers; i < to.pending_buffers; ++i) {
		// (As the owner's table path: the range's game buffer, made or joined, then made current, then obtained, which
		// then gives that buffer, not a copy in the shader ring (no device address: an exit in a long run). Anything but
		// the range's live game buffer is left unobtained: the patch step then stops the segment.)
		auto& entry   = state.pending_buffers[i];
		// (spec_reloc_refs: a found buffer's entry still over a live game buffer needs nothing here: a join copied its
		// bytes; the patch step decides whether its address moved. None any more: obtained as a pending one. Over
		// CPU-dirty memory (a collector's retire copies no bytes; read only, no region manager made) obtained too: its
		// bytes uploaded to the live buffer.)
		if (!entry.pending && (entry.since >= registration_now ||
		                       (buffers.GetBufferIfLive(buffers.RegisteredBuffer(entry.address, entry.size)) != nullptr &&
		                        !buffers.IsRegionCpuDirtyNoCreate(entry.address, entry.size)))) {
			entry.obtained = true;
			continue;
		}
		(void)buffers.FindBuffer(entry.address, entry.size);
		if (!entry.written) buffers.EnsureBufferContents(entry.address, entry.size);
		const auto [owner, offset] = buffers.ObtainBuffer(entry.address, entry.size, entry.written, entry.texel);
		(void)offset;
		const auto* game = buffers.GetBufferIfLive(buffers.RegisteredBuffer(entry.address, entry.size));
		entry.obtained   = owner != nullptr && owner == game && game->HasDeviceAddress();
	}
	for (size_t i = from.written; i < to.written; ++i) {
		const auto& range = state.written[i];
		(void)buffers.ObtainBuffer(range.begin, range.end - range.begin, true);
	}
	for (size_t i = from.current; i < to.current; ++i) {
		const auto& range = state.current[i];
		buffers.EnsureBufferContents(range.begin, range.end - range.begin);
	}
	// (What the holes wrote on the CPU since, over ranges it reads on the GPU.)
	if (m_effects.stamped) {
		if (m_log.End() > result.effects_from)
			for (size_t i = from.buffer_ranges; i < to.buffer_ranges; ++i) {
				const auto& range = state.buffer_ranges[i];
				m_log.ForEachHostWriteOver(result.effects_from, range.begin, range.end - range.begin,
				                           [&](uint64_t address, uint64_t size) { buffers.EnsureBufferContents(address, size); });
			}
	} else if (!m_effects.host_writes.Empty())
		for (size_t i = from.buffer_ranges; i < to.buffer_ranges; ++i) {
			const auto& range = state.buffer_ranges[i];
			m_effects.host_writes.ForEachIntersection(range.begin, range.end - range.begin, [&](RangeSet::Range written) {
				buffers.EnsureBufferContents(written.address, written.size);
			});
		}
	// (spec_reloc_refs: the table V# entries, noted as no buffer range: the same for theirs.)
	for (size_t i = from.table_ranges; i < to.table_ranges; ++i) {
		const auto& entry = state.table_ranges[i];
		if (m_effects.stamped) {
			if (m_log.End() > result.effects_from)
				m_log.ForEachHostWriteOver(result.effects_from, entry.begin, entry.end - entry.begin,
				                           [&](uint64_t address, uint64_t size) { buffers.EnsureBufferContents(address, size); });
		} else if (!m_effects.host_writes.Empty()) {
			m_effects.host_writes.ForEachIntersection(entry.begin, entry.end - entry.begin, [&](RangeSet::Range written) {
				buffers.EnsureBufferContents(written.address, written.size);
			});
		}
	}
	// The texel buffers it reads over images GPU work wrote (KYTY_SPECULATE_TEXEL_SYNC), synchronized from them as the
	// normal path does at the draw: after the uploads above (the image's bytes win, as there), before the work (what it
	// writes over them lands after the copy, as in order).
	for (size_t i = from.texel_syncs; i < to.texel_syncs; ++i) {
		const auto& range = state.texel_syncs[i];
		buffers.EnsureTexelContents(range.begin, range.end - range.begin);
	}
	if (segment.bda_all) {
		resources.PrepareBda();
	} else {
		for (size_t first = from.bda; first < to.bda; first += 128) {
			const std::span ranges(state.bda.data() + first, std::min<size_t>(128, to.bda - first));
			if (!resources.PrepareBdaReadRanges(ranges)) resources.PrepareBda();
		}
	}
	for (size_t i = from.invalidated; i < to.invalidated; ++i) {
		const auto& range = state.invalidated[i];
		textures.InvalidateMemoryFromGPU(range.begin, range.end - range.begin, "speculation");
	}
	// spec_worker_images: the sets its worker built with pending slots, completed by the owner now (the set is bound by
	// the replay after this; their images are not among the segment's, which RestoreState below restores).
	// spec_proposed_check (under KYTY_SPECULATE=4, the only mode with worker sets):
	// the sets its worker proposed in this segment, or handed on by one re-run as a hole (bound by the replay after this)
	// still current: the BDA page table's
	// generation theirs, their images the ones they were built with. One not stops the segment.
	if (kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4)
		for (const auto& proposed: state.proposed_sets)
			if (proposed.segment == index && proposed.check) {
				if (!proposed.check()) state.images_failed = true;
			}
	// (A set whose pending slots' memory the segment writes stopped it before this: the commit's checks before Before.)
	for (auto& proposed: state.proposed_sets)
		if (proposed.segment == index && proposed.complete) {
			if (!proposed.complete()) state.images_failed = true;
			proposed.complete = nullptr;
		}
	// spec_commit_targets: the targets it left to its commit, acquired as the normal path does at the draw, before their
	// states are restored to those its barriers start from (the acquisition's upload and clear stay: RestoreState
	// transitions from them). A target found Undefined or per subresource, or not acquired, fails the segment (the
	// caller stops it before the splice).
	std::vector<std::pair<const Image*, uint32_t>> layers2_drawn; // (spec_depth_layers2: the (image, layer)s acted on)
	for (size_t i = from.pending_targets; i < to.pending_targets; ++i) {
		const auto& pending = state.pending_targets[i];
		const auto  use     = std::ranges::find(segment.images, pending.image, &Spec::ImageUse::image);
		if (pending.depth_layers2) {
			// spec_depth_layers2: this draw's acquisition and load, as the normal path makes them at the draw, in record
			// order, ahead of the segment's replay: FindDepthTarget (the layers' upload, the metadata, the HTile entry), the
			// clean check, IsMetaCleared, the load rule (htile_load_decision, AcquireRenderTargets' own), the clear (an empty
			// rendering of the draw's view over its area, depth load op CLEAR: the draw's load op exactly), TouchMeta. Each
			// act is one the normal path makes for this draw; a stop before an entry's acts leaves the earlier entries' acts,
			// which the segment's re-run on the normal path repeats identically (FindDepthTarget finds nothing to do; a
			// consumed clear loads the bytes the commit cleared). Stopped (nothing more done, from the first failure on): not
			// on the GPU thread; an earlier pending target failed; the image used otherwise in the segment (its state left in
			// a layout that is not a depth attachment's); a second clear of a layer an earlier entry drew (it would come
			// after that draw's replay); an open rendering of the view that already clears it (BeginRendering would continue
			// it: the normal path would not clear again); the image found Undefined (the acquisition's upload / clear would be
			// discarded by RestoreState's barrier back to Undefined, and the replayed draw would load garbage).
			const auto stop = [&] { state.target_failed = true; };
			if (state.target_failed || !GuestGpu::IsGpuThread() || use == segment.images.end() ||
			    Spec::UsedOtherwise(use->found, use->found_subresources, use->left, use->left_subresources) ||
			    (use->found_subresources.empty() ? use->found.layout == vk::ImageLayout::eUndefined
			                                     : std::ranges::any_of(use->found_subresources, [](const auto& s) {
				                                       return s.layout == vk::ImageLayout::eUndefined;
			                                       }))) {
				stop();
				continue;
			}
			const auto& desc = *static_cast<const TextureCache::ImageDesc*>(pending.depth_desc.get());
			TextureCache::DepthDrawState drawn {};
			if (!textures.AcquireDepthDrawAtCommit(pending.id, pending.serial, desc, &drawn)) {
				stop();
				continue;
			}
			const auto key      = std::make_pair(static_cast<const Image*>(pending.image), desc.view_info.base_layer);
			const bool repeated = std::ranges::find(layers2_drawn, key) != layers2_drawn.end();
			if (!repeated) layers2_drawn.push_back(key);
			const auto decision = htile_load_decision(drawn.meta_cleared, drawn.fill, drawn.fill_known,
			                                          desc.info.metadata.stencil_compressed);
			if (decision.depth_meta_clear) {
				auto&       command = scheduler.Current();
				const auto* open    = command.OpenRendering();
				if (repeated || (open != nullptr && open->depth_stencil_attachment.image_view == drawn.view &&
				                 open->depth_stencil_attachment.depth_clear)) {
					stop();
					continue;
				}
				const auto& view   = desc.view_info;
				const auto  layout = vk::ImageLayout::eDepthAttachmentOptimal;
				command.EndRendering();
				pending.image->Transit(
				    layout, vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
				    ImageSubresourceRange {view.base_level, view.level_count, view.base_layer, view.layer_count},
				    command.Handle());
				RenderState clear {};
				clear.width                 = pending.layers2_width;
				clear.height                = pending.layers2_height;
				clear.num_layers            = pending.layers2_layers;
				clear.num_color_attachments = 0;
				auto& depth                 = clear.depth_stencil_attachment;
				depth.image_view            = drawn.view;
				depth.image_layout          = layout;
				depth.clear_value[0]        = std::bit_cast<uint32_t>(decision.depth_uniform ? decision.uniform_value
				                                                                             : pending.layers2_db_depth_clear);
				depth.has_depth             = true;
				depth.depth_clear           = true;
				command.BeginRendering(clear);
				command.EndRendering();
			}
			if (drawn.meta_cleared && !textures.TouchMeta(desc.info.metadata.range.address, desc.view_info.base_layer, false)) {
				stop(); // (cannot happen after FindDepthTarget made the entry HTile; slices >= 32 are never cleared)
				continue;
			}
			continue;
		}
		const bool  ok      = use != segment.images.end() && use->found.layout != vk::ImageLayout::eUndefined &&
		                use->found_subresources.empty() &&
		                textures.AcquireTargetAtCommit(pending.id, pending.serial, pending.base_level, pending.level_count,
		                                               pending.base_layer, pending.layer_count, pending.clear);
		if (!ok) state.target_failed = true;
	}
	// (Mode 4: the images GPU work wrote through a buffer since the translation found them, refreshed as the normal path's
	// binding refreshes them (the commit's checks marked them: ImageUse::refresh), before the restore loop puts them back
	// in the states the segment's barriers start from. One not refreshed fails the segment.)
	for (const auto& use: segment.images)
		if (use.refresh && !textures.RefreshAtCommit(*use.image, use.serial)) state.target_failed = true;
	// The images back in the states the segment's barriers start from. This Before may have released one: a creation it
	// made failed for memory (ReclaimImageMemory unregisters its victims, among them maybe the segment's own, and erases
	// them after a wait), and any EraseRetired here (that one, a deferred RetireImage a Finish ran) erases retired images
	// whose packets passed, among them maybe one the segment found already unregistered. One erased, its slot reused, or
	// unregistered here gets no barrier (it would name a destroyed VkImage, or another image) and the segment stops before
	// the splice (target_failed). One unregistered before Before and still alive (a parked image) gets it, as before.
	// (With spec_set_cpu_dirty_stop on, Changed stops the segment first at an image already unregistered when it ran.)
	// (Was an EXIT_IF on the serial alone; then registered-or-null-image too, which stopped ~0.9 segments a frame on the
	// 1-3 walk in mode 4 for images merely unregistered before Before: 9,701 "a pending target not acquired" stops.)
	// A use found Undefined (whole, or a subresource) where the image's state now is defined: a hole before this segment
	// in order (the overlay never sees one), or this Before's own completion or acquisition, defined it after the
	// segment's thread looked. The restore would record a barrier back to Undefined (invalid usage, the contents
	// discarded) and After would leave the image Undefined, so the next in-order Transit zeroes it. So:
	// - a use the segment recorded no barrier on (no Image::GetBarriers call: sample_lo still UINT32_MAX, sample_other
	//   clear; the entry a refused draw's TableEarlyEntry left), found wholly Undefined: no restore, and After leaves its
	//   state (keep_state). Exact: none of the segment's commands names the image's layout (each transition is a
	//   GetBarriers call, which marks the entry), and in order the segment does nothing to its state (the refused draw is a
	//   hole, run in order after the commit), so the in-order state is the one it has now.
	// - any other: the segment stops before the splice (target_failed), its restore not made. Its recorded barriers start
	//   from Undefined (discard; or Image::Transit's zero clear, which in order the defined image never gets): no state
	//   the restore could reach gives the in-order contents. A stop is exact: the image keeps its state and contents, and
	//   the in-order path runs the segment's packets again.
	// And a use whose image a set of this segment's completed above (ProposedSet::completed, segment == index:
	// TableCompletePendingImages left it in the slot's layout Lc, the replay binds it so). RestoreState(found) would move
	// it back over the completion's layout, under the binding. In order the set's build transits it at its draw, as the
	// completion did:
	// - a use with no barrier recorded (a refused draw's TableEarlyEntry: none of the segment's commands names its layout,
	//   and the refused draw is a hole run after the commit): no restore, and After leaves its state (keep_state), as the
	//   Undefined form above; exact, the state the completion left is the in-order one. Only while the image is still
	//   whole in the layout of every completion of it (two in different layouts: never kept) with the completion's access
	//   (shader read): anything of this Before after the completion that moved it (a pending target's acquisition of the
	//   same image, the refresh above: its upload ends in General) stops the segment instead;
	// - any other: the replay binds the completed set's descriptor (written in Lc) wherever the segment's own barriers
	//   have the image then (its draw made no transition: its TableTransit ran before the completion added it), where in
	//   order the set's build puts it in Lc. Restored (no stop) only when the image is in Lc at every point of the replay
	//   whatever the order: every completion of it in this segment in the shader-read-only layout (a sampled colour slot,
	//   not a depth one's), `found` in that layout (whole, or at every found subresource; RestoreState makes it `found`
	//   whatever the state now) and every access of the segment a shader read in it (not_sampled clear); else the segment
	//   stops before the splice (target_failed), its restore not made.
	struct CompletedNow {
		const Image*    image  = nullptr;
		uint64_t        serial = 0;
		vk::ImageLayout layout = vk::ImageLayout::eUndefined;
	};
	std::vector<CompletedNow> completed_now;
	for (const auto& proposed: state.proposed_sets)
		if (proposed.segment == index)
			for (const auto& slot: proposed.completed) completed_now.push_back({slot.image, slot.serial, slot.layout});
	// Two completions of one image (the same object and serial) in different layouts: one barrier state cannot hold both
	// (two sets of this segment completing it, a colour slot's ShaderReadOnly and a depth slot's DepthStencilReadOnly:
	// TableCompletePendingImages transits it once for each, and the replay binds both descriptors with it in the last
	// one's layout). Among this segment's completions, and those of an earlier segment's set this segment binds (its
	// handle among the segment's sets: spec_set_cpu_dirty_stop off only). Whether or not the segment's images hold it (with
	// no use of it the loop below never meets it): the segment stops before the splice (target_failed).
	if (!completed_now.empty() && !state.target_failed && !state.images_failed) {
		const auto bound_begin = state.sets.begin() + static_cast<ptrdiff_t>(from.sets);
		const auto bound_end   = state.sets.begin() + static_cast<ptrdiff_t>(to.sets);
		for (const auto& c: completed_now) {
			const auto other_layout = [&](const auto& s) { return s.image == c.image && s.serial == c.serial && s.layout != c.layout; };
			bool       conflict     = std::ranges::any_of(completed_now, other_layout);
			for (const auto& proposed: state.proposed_sets) {
				if (conflict) break;
				if (proposed.segment >= index || proposed.completed.empty() ||
				    std::find(bound_begin, bound_end, proposed.completed.front().set) == bound_end)
					continue;
				conflict = std::ranges::any_of(proposed.completed, other_layout);
			}
			if (conflict) {
				state.target_failed = true;
				break;
			}
		}
	}
	const bool erased = textures.EraseEpoch() != erase_epoch;
	for (size_t i = 0; i < segment.images.size(); ++i) {
		auto& use      = state.closed[index].images[i];
		use.keep_state = false;
		if (ImageGoneInBefore(textures, *use.image, use.serial, images_at_start[i], erased)) {
			state.target_failed = true;
			LiveCounters::Add(LiveCounters::SpecImageGoneInBeforeN);
			continue;
		}
		const auto undefined   = [](const VulkanImageState& s) { return s.layout == vk::ImageLayout::eUndefined; };
		const bool whole_undef = use.found_subresources.empty() && undefined(use.found);
		if (whole_undef || std::ranges::any_of(use.found_subresources, undefined)) {
			if (use.sample_lo == UINT32_MAX && !use.sample_other && whole_undef) {
				use.keep_state = true;
				continue;
			}
			bool now_defined = false; // (at an index found Undefined: the backing's state there now)
			{
				const Image::StateLock lock(*use.image);
				const auto&            now   = use.image->backing.subresource_states;
				const uint32_t         count = use.image->info.resources.levels * use.image->info.resources.layers;
				if (use.found_subresources.empty() && now.empty()) {
					now_defined = !undefined(use.image->backing.state);
				} else {
					// (Either side whole: its one state at every index.)
					for (uint32_t k = 0; k < count && !now_defined; ++k) {
						const auto& found_k = k < use.found_subresources.size() ? use.found_subresources[k] : use.found;
						const auto& now_k   = k < now.size() ? now[k] : use.image->backing.state;
						now_defined         = undefined(found_k) && !undefined(now_k);
					}
				}
			}
			if (now_defined) {
				state.target_failed = true;
				continue;
			}
		}
		const auto this_image = [&](const CompletedNow& c) { return c.image == use.image && c.serial == use.serial; };
		if (!completed_now.empty() && std::ranges::any_of(completed_now, this_image)) {
			if (use.sample_lo == UINT32_MAX && !use.sample_other) {
				bool kept = false; // (still whole in the layout of every completion of it, with the completion's access)
				{
					const Image::StateLock lock(*use.image);
					const auto&            now = use.image->backing.state;
					kept = use.image->backing.subresource_states.empty() &&
					       now.access_mask == vk::AccessFlags2 {vk::AccessFlagBits2::eShaderRead} &&
					       std::ranges::all_of(completed_now, [&](const CompletedNow& c) { return !this_image(c) || c.layout == now.layout; });
				}
				if (kept) use.keep_state = true;
				else state.target_failed = true;
				continue;
			}
			constexpr auto sro    = vk::ImageLayout::eShaderReadOnlyOptimal;
			const bool     lc_sro = std::ranges::all_of(completed_now, [&](const CompletedNow& c) { return !this_image(c) || c.layout == sro; });
			const bool     found_sro = use.found_subresources.empty()
			                               ? use.found.layout == sro
			                               : std::ranges::all_of(use.found_subresources, [&](const auto& s) { return s.layout == sro; });
			if (!lc_sro || use.not_sampled || !found_sro) {
				state.target_failed = true;
				continue;
			}
		}
		use.image->RestoreState(use.found, use.found_subresources, scheduler.Current().Handle());
	}
}

// KYTY_SPECULATE=4 with KYTY_WRITE_GUARD_SPEC: a commit's processor copy (a linear copy made on the CPU, between memory no
// GPU work wrote) made as the commit's labels are instead of behind the whole upload queue. The copy's destination is a
// release the game reads: the full drain made every upload job pushed before it read guest memory before the
// destination's new bytes could be seen (issue #196). Here each run of destination pages whose bytes differ
// (BufferCache::CopyGuestMemory's rule: the others are not written) is a guarded write (CommitGuardedCopyWrite:
// DrainRange over the run, so a pending job reading it takes the old bytes; its pages armed; lift jobs pushed behind the
// pending ones; the bytes written through the backing view): the game's view of them changes only after every earlier
// job ran, as with the drain. The source is waited for by nothing (the jobs never write guest bytes; a lift is pushed
// over the guest range [0, 1) on purpose, so no DrainRange meets it) and is read through its backing view: a source
// page still guarded by an earlier label of this commit shows the label's bytes there (program order) and no read
// faults into WriteGuard::OnFault's wait. The destination is compared through its backing view too (the same bytes; a
// page an earlier run or label guarded is not read through the guest mapping). Spec::HostWriteParts as
// CopyGuestMemory's writes: only what changes. The regions' epochs move as CopyGuestMemory's (NoteGuardedCopyWrite).
// Not taken (false; the caller drains and copies from `done` on, as before):
// - larger than 64 guard pages (GuardedWrite's fast form; the touch and the arm a page each: the 4 MiB culling copy keeps
//   CopyGuestMemory's helpers and its fault-free protection-keeping writes);
// - source and destination overlapping (each run is read as it stands, not as a copy made at once);
// - a read watcher on either (an offload's pages: their bytes are not in the backing view yet, ReadFaults);
// - GPU-modified memory on either (its bytes are not in the backing view until read back);
// - no single backing range for either (TryGetBackingPointer);
// - no upload job pending (the drain then waits for nothing and CopyGuestMemory needs no guard);
// - the destination's memory mapped at another guest address too (IsUniqueGuestBackingRangeAcrossMappings false): the
//   guard holds only this mapping's pages, and the game polling the release through the other one would see the new
//   bytes, written through the backing view, before the earlier jobs ran;
// - a guarded write refused (GuardedWrite's false: a page guarded with a protection not writable, no backing, the queue
//   empty after the run's DrainRange, ...): its run and those after it copied after the drain, which waits for the lifts
//   of the runs written before it too, so the guest mapping shows their bytes then.
// A trade: a written run's write-protected pages take the write fault (TouchForWrite) where CopyGuestMemory kept their
// protection and wrote through the backing view.
static bool GuardedCopy(BufferCache& buffers, uint64_t dst, uint64_t src, uint64_t size, uint64_t& done) {
	constexpr uint64_t MaxPages = 64;
	done                        = 0;
	const uint64_t first        = dst & ~(WriteGuard::Page - 1);
	if (size == 0 || (dst + size - first + WriteGuard::Page - 1) / WriteGuard::Page > MaxPages) return false;
	if (dst < src + size && src < dst + size) return false;
	if (Libs::LibKernel::Memory::ReadFaults(dst, size) || Libs::LibKernel::Memory::ReadFaults(src, size)) return false;
	if (buffers.IsRegionGpuModified(dst, size) || buffers.IsRegionGpuModified(src, size)) return false;
	const uint8_t* to   = Libs::LibKernel::Memory::TryGetBackingPointer(dst, size);
	const uint8_t* from = Libs::LibKernel::Memory::TryGetBackingPointer(src, size);
	if (to == nullptr || from == nullptr) return false;
	if (auto& worker = AsyncUpload::Get(); worker.Pushed() <= worker.Done()) return false;
	// (After the idle test: the address space's shared lock and a coverage lookup a mapping, only when the guard would
	// be used.)
	if (!Libs::LibKernel::Memory::IsUniqueGuestBackingRangeAcrossMappings(dst, size)) return false;
	const auto page_bytes = [&](uint64_t at) { return std::min(WriteGuard::Page - (dst + at) % WriteGuard::Page, size - at); };
	const auto differs    = [&](uint64_t at) { return std::memcmp(to + at, from + at, page_bytes(at)) != 0; };
	for (uint64_t at = 0; at < size;) {
		while (at < size && !differs(at)) at += page_bytes(at);
		done = at;
		if (at == size) break;
		uint64_t end = at;
		while (end < size && differs(end)) end += page_bytes(end);
		Spec::HostWriteParts parts;
		parts.Find(dst + at, from + at, end - at, to + at);
		if (!CommitGuardedCopyWrite(reinterpret_cast<void*>(dst + at), from + at, static_cast<uint32_t>(end - at)))
			return false;
		parts.Note();
		buffers.NoteGuardedCopyWrite(dst + at, end - at);
		at = done = end;
	}
	return true;
}

void Speculation::After(Result& result, size_t index) {
	const Spec::EffectSourceScope source(Spec::EffectSource::Commit);
	auto&       buffers  = m_renderer.GetBufferCache();
	auto&       textures = m_renderer.GetTextureCache();
	auto&       state    = result.state;
	const auto& segment  = state.closed[index];
	const auto  from     = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
	const auto& to       = segment.end;
	// spec_worker_sets: the image sets it built up to this segment, into the catalog (the owner's, under its write). Each
	// proposal SettleProposedSets would settle, in its order: its in-order part now, the rest under its owner's catalog write
	// had with a try-lock, else queued behind the ones waiting (RenderExecutor::SettleOrQueue); one that retires a set never
	// waits past its commit (those waiting and it run now, under a write that waits); then the ones waiting while the tries
	// succeed.
	if (!state.proposed_sets.empty()) {
		for (auto& proposed: state.proposed_sets) {
			if (proposed.settled || proposed.segment > index) continue;
			proposed.settled = true;
			if (proposed.settle_now) proposed.settle_now();
			const bool must_now = proposed.settle_must_now && proposed.settle_must_now();
			RenderExecutor::SettleOrQueue(proposed.catalog_owner, [settle = proposed.settle] { settle(true); });
			if (must_now && RenderExecutor::s_settle_pending_n.load(std::memory_order_relaxed) != 0)
				RenderExecutor::SettleDrain(true);
		}
		RenderExecutor::SettleDrain(false);
	}
	// What its thread could not note for the presenter (Spec::State::scenes): its indexed draws and its TAA dispatch,
	// noted here, where the GPU thread would have run them, before the flip after them takes them (FrameGen::TakeScene).
	// (A segment not committed is not here: its packets are translated again the normal way, which notes them.)
	if (to.indexed_draws != from.indexed_draws) FrameGen::NoteIndexedDraws(to.indexed_draws - from.indexed_draws);
	if (to.scenes != from.scenes) FrameGen::NoteScene();
	// The table pairs its draws and dispatches used (Spec::State::used_pairs), stamped as the GPU thread's lookup of them
	// would have: the sweep keeps a pair used within PairKeepFrames. (Alive: the sweep waits for no speculation live.)
	if (to.used_pairs != from.used_pairs) {
		const auto frame = m_renderer.FrameNumber();
		for (size_t i = from.used_pairs; i < to.used_pairs; ++i) *state.used_pairs[i] = frame;
	}
	// Its writes are the latest GPU work over those ranges; the images as it left them; the caches' LRU; the command
	// processor's guest memory writes.
	for (size_t i = from.written; i < to.written; ++i) {
		const auto& range = state.written[i];
		buffers.NoteSpeculativeWrite(range.begin, range.end - range.begin);
	}
	for (auto& use: state.closed[index].images) {
		const Image::StateLock lock(*use.image);
		// (A use Before did not restore keeps the state it has, its in-order state: see Before's restore loop.)
		if (use.keep_state) {
			MoveImageStateEpoch(use.image);
			continue;
		}
		use.image->backing.state              = use.left;
		use.image->backing.subresource_states = std::move(use.left_subresources);
		use.image->transit_group              = use.left_group;
		MoveImageStateEpoch(use.image); // (KYTY_TABLE_SERIALS: the image's serial moves, and the epoch)
	}
	// KYTY_SPEC_TOUCH_BATCH (live spec_touch_batch, default on): one lock for the segment's images (~0.4 ms a frame of
	// LRU touches at 1-1 were taken image by image); 0: one by one, as before.
	static auto& touch_batch = LiveSwitch::Find("spec_touch_batch", 1);
	if (touch_batch.load(std::memory_order_relaxed) == 0) {
		for (size_t i = from.touched_images; i < to.touched_images; ++i)
			textures.TouchSpeculated(*state.touched_images[i].first, state.touched_images[i].second);
	} else if (to.touched_images > from.touched_images) {
		textures.TouchSpeculated(std::span(state.touched_images).subspan(from.touched_images, to.touched_images - from.touched_images));
	}
	for (size_t i = from.touched_buffers; i < to.touched_buffers; ++i) buffers.TouchSpeculated(state.touched_buffers[i]);
	// The work before the processor's guest memory writes is on the GPU before them, as in order (BufferFlush): the
	// segments' buffers held for one submission go first. (Labels written while that work was held made ~1.5 frames
	// in 1,000 lose whole passes at the stone arch: speculate_batch_submits.) The buffer copies recorded before a label
	// are made first, as WriteGuestMemory does: issue #196.
	// (KYTY_WRITE_GUARD_SPEC: a label lands at once, guarded until the copies pushed before it are made, as on the normal
	// path; a copy, or a write the guard does not take, drains first as before.)
	const bool guard   = WriteGuard::InSpeculation();
	bool       drained = false;
	const auto drain   = [&] {
		if (drained) return;
		drained = true;
		AsyncUpload::Drain();
	};
	if (from.writes != to.writes) {
		m_renderer.GetCommandScheduler().FlushRecorded();
		if (!guard) drain();
	}
	// after_guard_batch (under KYTY_SPECULATE=4 only): a run of labels on one guard page, guarded once for the page
	// (graphicsRun.cpp).
	const int64_t batch = ::kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4 ? 1 : 0;
	for (size_t i = from.writes; i < to.writes; ++i) {
		const auto& write = state.writes[i];
		if (batch != 0 && guard && !write.copy) {
			const uint64_t page = write.address >> 12u;
			size_t         end  = i;
			while (end < to.writes && !state.writes[end].copy && (state.writes[end].address >> 12u) == page &&
			       ((state.writes[end].address + state.writes[end].size - 1) >> 12u) == page)
				++end;
			if (end - i >= 2) {
				thread_local std::vector<uint64_t> addresses, values;
				thread_local std::vector<uint32_t> sizes;
				addresses.clear(), values.clear(), sizes.clear();
				for (size_t k = i; k < end; ++k) {
					addresses.push_back(state.writes[k].address);
					values.push_back(state.writes[k].value);
					sizes.push_back(state.writes[k].size);
				}
				const size_t done = CommitGuardedWriteRun(addresses.data(), values.data(), sizes.data(), end - i);
				for (size_t k = i; k < end; ++k) {
					const auto& w = state.writes[k];
					if (k >= i + done && !CommitGuardedWrite(reinterpret_cast<void*>(w.address), &w.value, w.size)) {
						drain();
						std::memcpy(reinterpret_cast<void*>(w.address), &w.value, w.size);
					}
					Spec::NoteHostWrite(w.address, w.size);
					Spec::NoteLabel(w.address);
				}
				i = end - 1;
				continue;
			}
		}
		// (Mode 4: a copy GPU work wrote since the translation, made here as the in-order path makes it now, where
		// BufferCache::CopyBuffer takes its GPU path: either range GPU-modified in the tracker, or an image over the source.
		// Else the processor copy below, as before: CopyBuffer's processor path is that copy. Recorded after the segment's
		// work, flushed above.)
		if (write.copy && write.gpu &&
		    (buffers.IsRegionGpuModified(write.address, write.size) || buffers.IsRegionGpuModified(write.value, write.size) ||
		     textures.FindImageFromRange(write.value, write.size))) {
			buffers.CopyBuffer(write.address, write.value, write.size, false, false);
			continue;
		}
		if (write.copy) {
			// (Under KYTY_SPECULATE=4, the guard on and no full drain yet: the copy made as guarded writes of its
			// differing destination pages, GuardedCopy above; else, or for what it did not write, the drain and
			// CopyGuestMemory as before.)
			uint64_t done = 0;
			if (batch != 0 && guard && !drained && GuardedCopy(buffers, write.address, write.value, write.size, done)) continue;
			drain();
			if (done < write.size) buffers.CopyGuestMemory(write.address + done, write.value + done, write.size - done);
		} else {
			if (!guard || !CommitGuardedWrite(reinterpret_cast<void*>(write.address), &write.value, write.size)) {
				drain();
				std::memcpy(reinterpret_cast<void*>(write.address), &write.value, write.size);
			}
			// (A speculation translated after it that read memory there before: Spec::EffectsLog.)
			Spec::NoteHostWrite(write.address, write.size);
			Spec::NoteLabel(write.address);
		}
	}
	if (from.feedbacks != to.feedbacks) {
		RenderLockGuard lock(m_renderer.GetMutex());
		for (size_t i = from.feedbacks; i < to.feedbacks; ++i)
			buffers.ScheduleCopyFeedback(state.feedbacks[i].begin, state.feedbacks[i].end - state.feedbacks[i].begin);
	}
}

Speculation::HoleRun Speculation::RunHole(Result& result, CommandProcessor& cp, size_t index, bool copy_state) {
	const Spec::EffectSourceScope source(Spec::EffectSource::Hole);
	const auto& hole  = result.holes[index];
	const auto& state = *result.hole_states[index];
	if (copy_state) cp.CopyStateFrom(state, false);
	// (The draw state it starts from is not the shadow's.)
	auto& executor = m_renderer.GetRenderExecutor();
	DrawStateObserver::Invalidate();
	executor.NativeXprForgetState();
	Pm4Execution execution;
	const auto   start     = NowNs();
	const auto   processed = cp.Process(execution, {hole.packet, hole.dwords});
	m_hole_ns += NowNs() - start;
	DrawStateObserver::Invalidate();
	executor.NativeXprForgetState();
	// (A wait not satisfied yet: the commit waits there. Draws, dispatches and end-of-pipe events do not wait.)
	if (processed != Pm4ProcessResult::Complete) {
		EXIT_IF(!CommandProcessor::AtWait(execution));
		++m_waits_blocked;
		return HoleRun::Blocked;
	}
	++m_holes_run;
	// The work after it assumed it changed no processor state but its registers' (they are the same). (What it wrote
	// and retired: Spec::Effects, checked before each segment.)
	if (!cp.SameDrawState(state)) return ++m_stop_reasons["commit: a hole changed the draw state"], HoleRun::Changed;
	return HoleRun::Done;
}

Speculation::AsHole Speculation::SegmentAsHole(Result& result, CommandProcessor& cp, size_t index) {
	// (Under KYTY_SPECULATE=4 only; otherwise the stop stands.)
	if (kyty_local_speculate_mode.load(std::memory_order_relaxed) != 4 || index >= result.holes.size())
		return AsHole::NotTaken;
	const auto& state = result.state;
	const auto  from  = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
	const auto& to    = state.closed[index].end;
	// (A wait inside it: the GPU thread could block there, before the hole it must stop at. Also one its own pending
	// writes or a previous speculation's gave whole, which pushed no WaitCheck: State::covered_waits.)
	if (to.waits != from.waits || to.covered_waits != from.covered_waits) return AsHole::NotTaken;
	// From the segment's start (the chunk's, or past the hole before it: `cp` is in that state, the commit checked the
	// chunk's start state and ran the hole) up to its hole's packet.
	Pm4Execution execution = index == 0 ? result.start_at : result.holes[index - 1].after;
	if (!execution.Started() || !result.holes[index].at.Started()) return AsHole::NotTaken;
	// Its recorded work dropped, never submitted: the GPU thread records it again.
	m_renderer.GetCommandScheduler().DiscardFirst(*result.recorder, state.closed[index].recorded - result.submitted);
	result.submitted = state.closed[index].recorded;
	RelocCellsRelease(result.state, index); // (its chunks discarded)
	const Spec::EffectSourceScope source(Spec::EffectSource::Hole);
	auto&                         executor = m_renderer.GetRenderExecutor();
	DrawStateObserver::Invalidate();
	executor.NativeXprForgetState();
	const auto stop = result.holes[index].at;
	cp.SetStopAt(&stop);
	const auto start     = NowNs();
	const auto processed = cp.Process(execution, {});
	m_hole_ns += NowNs() - start;
	cp.SetStopAt(nullptr);
	const bool reached = cp.TakeStopReached();
	DrawStateObserver::Invalidate();
	executor.NativeXprForgetState();
	result.resume_at = std::move(execution); // (at the hole's packet, or where the run ended)
	if (!reached) {
		// (The packets changed under the walk: the commit stops, the GPU thread goes on from where its run ended.)
		const auto& hole = result.holes[index];
		std::printf("Speculation: hole %zu of %zu not reached (processed %d): %s, %u dwords, header 0x%08x\n", index,
		            result.holes.size(), static_cast<int>(processed), hole.reason != nullptr ? hole.reason : "(no reason)",
		            hole.dwords, hole.packet != nullptr ? *hole.packet : 0u);
		return AsHole::NotReached;
	}
	return AsHole::Reached;
}

// Guest memory now as a speculation's thread reads it (Spec::State::CopyGuest): through the write guard's backing view
// where a guest write is pending; on Linux, a page a read watcher guards refused (GPU-written bytes not downloaded yet,
// which a pread of /proc/self/mem would read stale) and the rest through the backing view (spec_backing_reads 1, the
// default); else a guarded copy (a page without access refuses). False: not read.
static bool ReadGuestNow(void* out, uint64_t address, uint64_t bytes) {
	if (const auto* backing = WriteGuard::GuardedBacking(address, bytes)) {
		std::memcpy(out, backing, bytes);
		return true;
	}
#if !defined(_WIN32)
	static auto& backing_reads = LiveSwitch::Find("spec_backing_reads", 1);
	if (backing_reads.load(std::memory_order_relaxed) != 0) {
		if (Libs::LibKernel::Memory::ReadFaults(address, bytes)) return false;
		if (Libs::LibKernel::Memory::TryReadBackingToHost(address, out, bytes)) return true;
	}
#endif
	return Common::GuardedCopy(out, reinterpret_cast<const void*>(address), bytes) == 0;
}

// KYTY_SPECULATE=4: the words segment `index` read from the speculation's own journal that a write of an earlier
// segment gave (Spec::State::journal_reads with `entry` before the segment's own writes), checked at its commit as the
// chain reads are: such a word notes nothing else the commit checks (a wait over it pushes no WaitCheck). It runs among
// the checks before the segment's Before. Memory then is what the in-order path reads at the segment's start: the
// earlier segments' writes landed at their commits (After), the holes before it ran (RunHole), a segment run as a hole
// (SegmentAsHole) ran its packets in order, and a previous speculation's commits are done; nothing of this segment's
// has landed, and a word its own writes gave is not checked (in order it reads them). So each must be in memory as the
// reader took it (Spec::State::journal_words), with no GPU write logged over it since the speculation began (logged as
// recorded, before it reaches memory: memory cannot tell). Null: all hold.
const char* Speculation::JournalReadsMoved(const Result& result, size_t index) const {
	const auto&                            state = result.state;
	const auto                             from  = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
	const auto&                            to    = state.closed[index].end;
	thread_local std::array<uint8_t, 4096> now;
	for (size_t i = from.journal_reads; i < to.journal_reads; ++i) {
		const auto& read = state.journal_reads[i];
		if (read.entry >= from.writes) continue; // (its own writes gave it)
		if (m_log.stamps.MaybeSince(read.address, read.bytes, result.effects_from) &&
		    m_log.FirstGpuWriteOver(result.effects_from, read.address, read.bytes) != nullptr)
			return "commit: a journal word read, a GPU write over it";
		const auto* taken = reinterpret_cast<const uint8_t*>(state.journal_words.data() + read.words);
		for (uint64_t done = 0; done < read.bytes;) { // (by page)
			const auto address = read.address + done;
			const auto n       = std::min<uint64_t>(read.bytes - done, now.size() - (address & (now.size() - 1)));
			if (!ReadGuestNow(now.data(), address, n)) return "commit: a journal word read, not readable now";
			if (std::memcmp(now.data(), taken + done, n) != 0) return "commit: a journal word read otherwise";
			done += n;
		}
	}
	return nullptr;
}

void Speculation::Finish(std::unique_ptr<Result> result, size_t first) {
	auto& scheduler = m_renderer.GetCommandScheduler();
	auto& r         = *result;
	r.state.ReleaseConsumePending(); // (its holes' clears are consumed now, or never: spec_predict_dcc=3)
	// (spec_worker_sets: its proposed sets no commit settled, settled here on the GPU thread while it still counts as live,
	// not at its Result's reuse on a worker: the catalog's pair sweeps wait for no speculation live, so the pairs those
	// sets name are alive, and the aside release's queue and the owner's state are the GPU thread's.)
	if (!r.state.proposed_sets.empty()) r.state.SettleProposedSets(0, true);
	for (const auto& hole: r.holes) ++m_hole_reasons[hole.reason];
	if (first < r.state.closed.size() || !r.recorder->recorded.empty() || r.recorder->open != SIZE_MAX)
		scheduler.DiscardRecorded(*r.recorder);
	// (Its rings' data is read by the command buffers it submitted, done with the last; none submitted: free now. Under
	// speculate_batch_submits or a splice the last is the GPU thread's open or held buffer's tick, not yet submitted:
	// CommandScheduler::Wait refuses such a tick on a speculation's thread, so no worker waits for it.)
	const auto tick    = r.last_tick;
	const auto pending = r.recorder->pending_tick;
	auto&      owner   = *r.owner;
	if (!r.recorder->own_thread) ResolvePending(owner, pending, tick);
	std::lock_guard lock(m_mutex);
	// (A speculation thread's: released when it begins its next translation.)
	if (r.recorder->own_thread) owner.resolved.emplace_back(pending, tick);
	// (Its pending writes are made or dropped: what was translated after it reads through its command buffer's record
	// while that is not done, else memory.)
	Unlink(&r.state, RecordOf(r.serial, r.chunk));
	// (Its commands are submitted or discarded: none of them binds a pipeline later.)
	kyty_local_speculations_live.fetch_sub(1, std::memory_order_seq_cst);
	r.retired = ++m_clock;
	owner.free.push_back(std::move(result));
}

void Speculation::FreeResult(std::unique_ptr<Result> result) {
	result->state.Clear(); // (as a reuse would: what it held of the caches let go)
	m_renderer.GetCommandScheduler().DestroyRecorder(std::move(result->recorder));
}

void Speculation::ResolvePending(Translator& translator, uint64_t pending, uint64_t tick) {
	translator.table_upload->ResolvePendingTicks(pending, tick);
	translator.shader_upload->ResolvePendingTicks(pending, tick);
	translator.draw->ResolvePendingTicks(pending, tick);
	translator.compute->ResolvePendingTicks(pending, tick);
}

void Speculation::Report() {
	if ((m_whole + m_partly + m_none) % 2000 != 0) return;
	const auto print = [](const char* title, std::unordered_map<const char*, uint64_t>& counts) {
		std::vector<std::pair<const char*, uint64_t>> sorted(counts.begin(), counts.end());
		std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second > b.second; });
		std::printf("; %s:", title);
		for (const auto& [reason, count]: sorted) std::printf(" %s %llu", reason, static_cast<unsigned long long>(count));
		counts.clear();
	};
	const auto frame  = m_renderer.FrameNumber();
	const auto frames = static_cast<double>(std::max<uint64_t>(frame - m_report_frame, 1));
	m_report_frame    = frame;
	const auto ms     = [&](uint64_t ns) { return static_cast<double>(ns) / 1e6 / frames; };
	std::printf("Speculation: %llu whole, %llu partly, %llu not committed; %llu segments, %llu holes (%llu waits blocked); "
	            "ahead: %llu being translated, %llu not yet; blind %llu (stopped %llu of %llu); per frame: queue %.2f ms, "
	            "translation kept %.2f ms, lost %.2f ms, commit %.2f ms (holes %.2f ms), waits %.2f ms; start delay %.2f ms; "
	            "translations %llu/%llu/%llu/%llu/%llu/%llu (<0.5/1/2/4/8 ms/more, longest %.1f ms)",
	            static_cast<unsigned long long>(m_whole), static_cast<unsigned long long>(m_partly),
	            static_cast<unsigned long long>(m_none), static_cast<unsigned long long>(m_segments),
	            static_cast<unsigned long long>(m_holes_run), static_cast<unsigned long long>(m_waits_blocked),
	            static_cast<unsigned long long>(m_waits_for_ahead), static_cast<unsigned long long>(m_ahead_misses),
	            static_cast<unsigned long long>(m_blind), static_cast<unsigned long long>(m_blind_stopped),
	            static_cast<unsigned long long>(m_stopped),
	            ms(m_queue_ns), ms(m_kept_ns), ms(m_lost_ns), ms(m_commit_ns - m_hole_ns), ms(m_hole_ns), ms(m_wait_ns),
	            m_started != 0 ? static_cast<double>(m_start_delay_ns) / 1e6 / static_cast<double>(m_started) : 0.0,
	            static_cast<unsigned long long>(m_durations[0]), static_cast<unsigned long long>(m_durations[1]),
	            static_cast<unsigned long long>(m_durations[2]), static_cast<unsigned long long>(m_durations[3]),
	            static_cast<unsigned long long>(m_durations[4]), static_cast<unsigned long long>(m_durations[5]),
	            static_cast<double>(m_longest_ns) / 1e6);
	m_waits_for_ahead = m_ahead_misses = 0;
	m_blind = m_stopped = m_blind_stopped = 0;
	m_kept_ns = m_lost_ns = m_commit_ns = m_hole_ns = m_queue_ns = 0;
	{
		std::lock_guard lock(m_mutex);
		m_wait_ns = m_start_delay_ns = m_started = m_longest_ns = 0;
		m_durations.fill(0);
	}
	print("stops", m_stop_reasons);
	print("holes", m_hole_reasons);
	{
		std::lock_guard lock(m_mutex);
		std::printf("; stopped for the GPU thread: %llu", static_cast<unsigned long long>(std::exchange(m_cancelled, 0)));
	}
	std::printf("\n");
	std::fflush(stdout);
	m_whole = m_partly = m_none = m_segments = m_holes_run = m_waits_blocked = 0;
}

} // namespace Libs::Graphics

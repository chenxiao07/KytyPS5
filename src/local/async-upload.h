#pragma once
// KYTY_ASYNC_UPLOAD: buffer upload copies (guest memory -> host-visible staging memory) run on
// one worker, in the order the render thread recorded them; a queue submission first waits for
// the copies recorded before it. The render thread only records the copy commands.
// KYTY_ASYNC_UPLOAD=2: the staging copies of image uploads (ObtainBufferForImage) as well.
//
// The worker reads the guest range through the guest's mapping where it can (CopyFrom: a page without access
// ends a guarded copy) and through the backing view past that: a guest write or protection change never takes it
// into the guest's fault handler, and unmapping a range waits for the jobs on it before the range can be reused.

#include "common/guardedCopy.h"
#include "live-census.h"
#include "live-counters.h"
#include "local-platform.h"
#include "mode-toggle.h"
#include "parallel-pages.h"
#include "spin-gate.h"
#include "write-guard.h"

#include <algorithm>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <x86intrin.h>

extern "C" {
extern volatile std::atomic<uint32_t> kyty_local_async_upload_mode;
extern volatile std::atomic_uint32_t  kyty_local_speculate_threads; // (KYTY_SPECULATE_THREADS: PipeCopyThreads)
// KYTY_UPLOAD_LANES (live 1..3, default 1; PR #287): 2 = a run's write protections split by
// guest region (4 MiB, the trackers' and the page manager's region: their per-region locks) over the worker and a second thread
// (Kyty.Upload2), which sleeps on a futex between runs and never spins; the run's copies still start only after both lanes are done,
// and one region's protections stay in queue order in its one lane. 3 = 2 with the order checks (upload_lane_mismatch must stay 0).
// (At 1-1 the waited batches' protections were all single-region with no call among them, the lanes 45-57 / 43-55 %.)
extern volatile std::atomic<uint32_t> kyty_local_upload_lanes;
// KYTY_UPLOAD_HELP (live kyty_local_upload_help): whether the render thread's Drain runs the jobs the upload worker is not
// at itself (Worker::Help): 1 always (the Steam Deck build's default), 0 never, 2 (the default elsewhere) while the
// process's CPUs are busy, the spin gate shut (fewer than three of them idle, spin-gate.h). On the Deck's four cores the
// worker, asleep or waiting for a CPU, left the render thread waiting 7.2-7.7 ms a frame, 1.1-1.3 with it helping (stand
// 30.5 -> 32.8 fps). The 22-CPU desktop held to six of them: the wait 1.8-1.9 -> 0.6 ms, 63.7-66.3 -> 66.6-68.5 fps at
// 1-1, the same lows; to four: 4.1-4.4 -> 0.9-1.0 ms, +4% (the jobs at normal priority: Help). With CPUs to spare the
// worker is at its jobs at once, and the render thread taking them gained nothing and made the 1% lows worse (10-27).
inline volatile std::atomic<uint32_t> kyty_local_upload_help {[] {
	const char* value = std::getenv("KYTY_UPLOAD_HELP");
	if (value != nullptr && (value[0] == '0' || value[0] == '1' || value[0] == '2') && value[1] == '\0')
		return static_cast<uint32_t>(value[0] - '0');
#if defined(KYTY_STEAM_DECK)
	return 1u;
#else
	return 2u;
#endif
}()};
}

namespace AsyncUpload {

// Whether mode 4 made the state pass its own thread (graphicsRun.cpp start_speculation; -1: not made yet, then the same
// test is repeated with SpecPacketSlots, Spec::MaxPacketSlots: static_assert there). Read by Worker::PipeCopyThreads.
inline std::atomic<int32_t>      g_spec_pass_made {-1};
inline constexpr uint32_t        SpecPacketSlots = 12;

// True on the worker's thread (a fault there cannot wait for the render thread, which may wait for the worker).
inline thread_local bool t_worker = false;

// KYTY_SERVE_IN_WAITS bit 1 (GuestGpu::ServeInWaits), set by the GPU thread. Asked only by a Wait that was told to serve
// (the drains before a guest memory write of the command processor and at the frame boundary; a submit's or an
// unmap's drain never serves). With false: whether that Wait spins to the copies' end, running the guest commands
// queued for the GPU thread meanwhile (a game thread's readback fault waits for its command there); with true: run
// them, true when some ran.
inline bool (*g_serve_hook)(bool serve) = nullptr;

class Worker {
public:
	static constexpr uint64_t Capacity = 1u << 16;
	// What a call job is, set where it is pushed (KYTY_SPEC_UPLOAD_PIPE: a lift's or a Read's pages are known; any other
	// call holds the pipe's walk until it has run).
	enum class JobKind : uint8_t { Other, Lift, Read };
	// KYTY_UPLOAD_LANES: the guest region a lane is chosen by: the trackers' region managers' and the page manager's region,
	// whose locks protections take (bufferCache.cpp asserts it is TRACKER_REGION_SIZE).
	static constexpr uint64_t LaneRegion = uint64_t {4} << 20u;

	Worker(): m_jobs(std::make_unique<Job[]>(Capacity)), m_thread([this] { Run(); }) {}
	~Worker() {
		m_stop.store(true, std::memory_order_seq_cst);
		m_head.fetch_add(0, std::memory_order_seq_cst);
		m_head.notify_all();
		m_thread.join();
		// (The wake below reads as a post: no jobs in it. The worker is gone and the second thread idle.)
		m_lane_jobs[1].clear();
		m_lane_posted.fetch_add(1, std::memory_order_seq_cst);
		m_lane_posted.notify_all();
		if (m_lane_thread.joinable()) m_lane_thread.join();
	}
	Worker(const Worker&)            = delete;
	Worker& operator=(const Worker&) = delete;

	// Render thread only: copies the backing of guest [guest, guest + size). Returns the sequence number
	// that covers this copy.
	uint64_t Push(uint8_t* destination, const uint8_t* source, uint64_t size, uint64_t guest) {
		return Add({destination, {source}, size, nullptr, nullptr, guest, size, false});
	}
	// Render thread only: runs call(context, a, b) on the worker in queue order; it works on guest
	// [guest, guest + guest_size). `kind`: a lift (`a` its page) or a Read (`a`, `b` its staging slice), else Other.
	uint64_t PushCall(void (*call)(void*, uint64_t, uint64_t), void* context, uint64_t a, uint64_t b, uint64_t guest,
	                  uint64_t guest_size, JobKind kind = JobKind::Other) {
		return Add({reinterpret_cast<uint8_t*>(a), {nullptr}, b, call, context, guest, guest_size, false, kind});
	}
	// Render thread only: as PushCall, for a write protection of guest pages the copies pushed after it read
	// (KYTY_ASYNC_REPROTECT): it may run before copies pushed ahead of it too (Run).
	// `merge` (KYTY_SPEC_UPLOAD_PIPE; nullptr: never merged): merge(context, a, b, count) does what `count` of these calls
	// over adjacent ranges joined into [a, a + b) would.
	using MergeCall = void (*)(void* context, uint64_t a, uint64_t b, uint64_t count);
	uint64_t PushProtection(void (*call)(void*, uint64_t, uint64_t), void* context, uint64_t a, uint64_t b,
	                        uint64_t guest, uint64_t guest_size, MergeCall merge = nullptr) {
		Job job {reinterpret_cast<uint8_t*>(a), {.merge = merge}, b, call, context, guest, guest_size, true};
		return Add(job);
	}
	// Render thread only: wakes the worker after a batch of pushes.
	void Kick() {
		if (m_sleeping.load(std::memory_order_seq_cst)) m_head.notify_one();
	}
	// Render thread only: the sequence number of the last pushed copy.
	[[nodiscard]] uint64_t Pushed() const { return m_head.load(std::memory_order_relaxed); }
	// Render thread only: the sequence number of the last finished job (KYTY_WRITE_GUARD: whether any is pending).
	[[nodiscard]] uint64_t Done() const { return m_done.load(std::memory_order_acquire); }
	// Render thread only: the sequence number of the last pending job on guest [vaddr, vaddr + size), or 0.
	[[nodiscard]] uint64_t PendingOn(uint64_t vaddr, uint64_t size) const {
		const uint64_t head = m_head.load(std::memory_order_relaxed);
		uint64_t       last = 0;
		for (uint64_t sequence = m_done.load(std::memory_order_acquire); sequence < head; ++sequence) {
			const Job& job = m_jobs[sequence % Capacity];
			if (job.guest < vaddr + size && vaddr < job.guest + job.guest_size) last = sequence + 1;
		}
		return last;
	}

	// Any thread: returns once every copy up to `sequence` is in staging memory. `serve`: GPU thread only, run the
	// guest commands queued meanwhile (g_serve_hook).
	void Wait(uint64_t sequence, bool serve = false) {
		if (m_done.load(std::memory_order_acquire) >= sequence) return;
		const auto start = std::chrono::steady_clock::now();
		const auto note  = [&] {
			const auto ns = (std::chrono::steady_clock::now() - start).count();
			LiveCensus::g_upload_wait_ns.fetch_add(ns, std::memory_order_relaxed);
			LiveCounters::Add(LiveCounters::UploadWaitUs, static_cast<uint64_t>(ns) / 1000u);
		};
		if (serve && g_serve_hook != nullptr && g_serve_hook(false)) {
			Kick();
			while (m_done.load(std::memory_order_acquire) < sequence)
				if (!g_serve_hook(true)) _mm_pause();
			return note();
		}
		for (int spin = 0; spin < 4000; ++spin) {
			if (m_done.load(std::memory_order_acquire) >= sequence) return note();
			_mm_pause();
		}
		m_waiters.fetch_add(1, std::memory_order_seq_cst);
		for (uint64_t done; (done = m_done.load(std::memory_order_seq_cst)) < sequence;)
			m_done.wait(done, std::memory_order_seq_cst);
		m_waiters.fetch_sub(1, std::memory_order_relaxed);
		note();
	}

	// Render thread (Drain): as Wait, but the jobs up to `sequence` that the worker is not at (asleep, or not yet given a
	// CPU) are run here, in queue order: on the Steam Deck's four cores the render thread waited ~7 ms a frame for jobs
	// the worker ran in ~3.6 ms. Only write protections and copies: a call of another kind (a read through the guest's
	// mapping, which may fault into a readback) is left to the worker, and waited for.
	void Help(uint64_t sequence) {
		if (m_done.load(std::memory_order_acquire) >= sequence) return;
		const auto start = std::chrono::steady_clock::now();
		// The jobs at the worker's priority, not the render thread's above the game's threads: held to four CPUs, the game's
		// threads starved behind them, frames of 100-500 ms with the render thread idle (1-1, 1% lows 25-45 -> 7-13 fps;
		// at normal priority the lows as without the help, +4% fps). (The Steam Deck build keeps the render thread's core
		// for it alone, KYTY_RENDER_CORE: nothing of the game's runs there.)
		std::optional<LocalPlatform::NormalPriority> normal;
		for (int spin = 0; m_done.load(std::memory_order_acquire) < sequence;) {
			if (m_process.try_lock()) {
#if !defined(KYTY_STEAM_DECK)
				if (!normal) normal.emplace();
#endif
				const uint64_t done = m_done.load(std::memory_order_relaxed);
				const uint64_t head = m_head.load(std::memory_order_acquire);
				const uint64_t ran  = done < head ? Process(done, head, true, PipeOn()) : done;
				m_process.unlock();
				if (ran == done) break; // (the next job is the worker's)
				continue;
			}
			if (++spin >= 4000) break; // (the worker is at them)
			_mm_pause();
		}
		normal.reset();
		const auto helped = (std::chrono::steady_clock::now() - start).count();
		if (m_done.load(std::memory_order_acquire) < sequence) {
			Kick();
			Wait(sequence);
		}
		LiveCounters::Add(LiveCounters::UploadHelpUs, static_cast<uint64_t>(helped) / 1000u);
	}

private:
	// (64 bytes: `merge` shares the copy's `source`, `kind` sits in the padding after `protection`.)
	struct Job {
		uint8_t* destination;
		union {
			const uint8_t* source; // a copy's
			MergeCall      merge;  // a protection's (PushProtection)
		};
		uint64_t size;
		void (*call)(void*, uint64_t, uint64_t); // with destination as `a`, size as `b`
		void*    context;
		uint64_t guest, guest_size;
		bool     protection;              // a call PushProtection queued
		JobKind  kind = JobKind::Other;   // a call's (PushCall)
	};
	static_assert(sizeof(Job) == 64);

	uint64_t Add(const Job& job) {
		const uint64_t head = m_head.load(std::memory_order_relaxed);
		while (head - m_done.load(std::memory_order_acquire) >= Capacity) {
			Kick();
			_mm_pause();
		}
		m_jobs[head % Capacity] = job;
		m_head.store(head + 1, std::memory_order_seq_cst);
		return head + 1;
	}

	// The worker's threads above the game's (KYTY_UPLOAD_PRIORITY=0: at their normal priority, as before 11-14). Held to the
	// performance cores (KYTY_RECORDING_CPUS) at the same priority as the game's threads, which fill those cores while a
	// world loads, the worker lost its CPU for whole time slices while it held the guest address space's lock to protect
	// pages: the game's next map waited for it, and every thread for that map, the render thread in its label drains (one
	// boot, the soak's world changes, the priority switched live: 22 stalls of 3-30 ms in 6 loads at normal priority, none
	// above it; 1-1 standing at 120 Hz 72.5 fps either way, 1% low 65.1 -> 65.9). Not pinned instead: no stalls either, but
	// 1% low -3 fps (it ran on efficiency cores). (Called unconditionally: Run keeps its trained profile.)
	static void RaisePriority() {
		static const bool raise = [] {
			const char* value = std::getenv("KYTY_UPLOAD_PRIORITY");
			return value == nullptr || value[0] != '0';
		}();
		if (raise) LocalPlatform::MakeAboveGameThread();
	}

	void Run() {
		LocalPlatform::SetThreadName("Kyty.Upload");
		t_worker = true;
		WriteGuard::t_no_wait = true; // (KYTY_WRITE_GUARD: the lifts are this thread's jobs)
		// Created by the render thread: do not inherit its dedicated CPU.
		LocalPlatform::PinThreadToCpuList(std::getenv("KYTY_RECORDING_CPUS"));
		RaisePriority();
		// Pause iterations the worker polls for new copies before it sleeps (while its CPUs idle: spin-gate.h).
		constexpr uint32_t UploadSpins = 20000;
		for (;;) {
			uint64_t       done  = m_done.load(std::memory_order_acquire);
			uint64_t       head  = m_head.load(std::memory_order_acquire);
			const uint32_t spins = head == done && SpinGate::Open(false) ? UploadSpins : 0;
			for (uint32_t spin = 0; head == done && spin < spins; ++spin) {
				_mm_pause();
				head = m_head.load(std::memory_order_acquire);
				done = m_done.load(std::memory_order_acquire);
			}
			if (head == done) {
				LiveCounters::Add(LiveCounters::UploadSleeps);
				m_sleeping.store(true, std::memory_order_seq_cst);
				head = m_head.load(std::memory_order_seq_cst);
				if (head == done) {
					if (m_stop.load(std::memory_order_seq_cst)) return;
					m_head.wait(done, std::memory_order_seq_cst);
				}
				m_sleeping.store(false, std::memory_order_relaxed);
				continue;
			}
			// (The render thread may be running jobs itself: Help.)
			std::lock_guard process(m_process);
			done = m_done.load(std::memory_order_relaxed);
			head = m_head.load(std::memory_order_acquire);
			if (done < head) {
				if (PipeOn()) PipeSession();
				else Process(done, head, false, false);
			}
		}
	}

	// KYTY_SPEC_UPLOAD_PIPE (live spec_upload_pipe; on under KYTY_SPECULATE=4, 0 otherwise: mode-toggle.h; set, it acts in
	// either mode), read at each batch: the worker's batches as pipe sessions (PipeSession) in place of Process's runs, and
	// the queued protections a batch runs merged where adjacent (RunProtectionMerged). Off: one relaxed load a batch. With
	// no copy threads (PipeCopyThreads 0) it stays off as if the switch were 0.
	static bool PipeOn() {
		static auto& on = LiveSwitch::Find("spec_upload_pipe", ModeToggle::Default("spec_upload_pipe"));
		return on.load(std::memory_order_relaxed) != 0 && PipeCopyThreads() != 0;
	}

	// KYTY_SPEC_UPLOAD_PIPE_THREADS=<n> (0-16, digits only): the pipe's copy threads; 0: none, the walker only, which is
	// the pipe off (Process's runs, no merged protections: KYTY_SPEC_UPLOAD_PIPE=0); anything else is not a count, said,
	// and the rule decides. Unset: the physical cores left after the threads that are busy, at most 7 (every pipe measure
	// ran with 7): the cores the process may run on, every efficiency class (LocalPlatform::CoresIn), less 3 (the GPU,
	// record and upload threads), and under KYTY_SPECULATE=4 less the workers' (KYTY_SPECULATE_THREADS, at most the
	// cores of their CPU list: speculate_cpu_mask, else KYTY_SPECULATE_CPUS, else LocalPlatform::SpeculationCpuList; a
	// core each where there is no list) and 1 for the state pass thread where it has one (g_spec_pass_made, or
	// start_speculation's test). The async queue's 2 speculation threads are not counted. 8 P-cores (16 CPUs) + 8
	// E-cores, workers on CPUs 8-15: 16 - 4 - 3 - 1 = 8, so 7. 8 cores alike (16 CPUs; the workers on 8-15), or that
	// part held to CPUs 0-15: 8 - 4 - 3 - 1 = 0, the walker only (an 8-core 7800X3D measured +5 % with the pipe off).
	// Cores not known: 7. Worked out once, at the first batch the pipe switch is on for, and said with its sum (the count
	// fixed for the run: the threads are made once).
	static constexpr uint32_t PipeThreadsMax = 7;
	static uint32_t PipeCopyThreads() {
		static const uint32_t threads = [] {
			char        why[256] = {};
			uint32_t    count    = PipeThreadsMax;
			const char* value    = std::getenv("KYTY_SPEC_UPLOAD_PIPE_THREADS");
			const bool  set      = value != nullptr && *value != '\0';
			const bool  digits   = set && std::strlen(value) <= 9 && std::strspn(value, "0123456789") == std::strlen(value);
			if (digits) {
				count = static_cast<uint32_t>(std::min<unsigned long>(std::strtoul(value, nullptr, 10), 16ul));
				std::snprintf(why, sizeof(why), "KYTY_SPEC_UPLOAD_PIPE_THREADS=%s", value);
			} else {
				const bool        spec      = ::kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4;
				static auto&      live_cpus = LiveSwitch::Find("speculate_cpu_mask", 0);
				const int64_t     live      = live_cpus.load(std::memory_order_relaxed);
				const char*       named     = std::getenv("KYTY_SPECULATE_CPUS");
				const std::string list = named != nullptr && *named != '\0' ? std::string(named) : LocalPlatform::SpeculationCpuList();
				const uint64_t    mask = live != 0 ? static_cast<uint64_t>(live) : LocalPlatform::CpuListMask(list.c_str());
				const uint32_t    cores        = LocalPlatform::CoresIn(~uint64_t {0});
				const uint32_t    workers      = ::kyty_local_speculate_threads.load(std::memory_order_relaxed);
				const uint32_t    list_cores   = mask != 0 ? LocalPlatform::CoresIn(mask) : 0;
				const uint32_t    worker_cores = !spec ? 0u : list_cores != 0 ? std::min(workers, list_cores) : workers;
				const char*       pass_env     = std::getenv("KYTY_SPEC_PASS_THREAD");
				const int32_t     made         = g_spec_pass_made.load(std::memory_order_relaxed);
				const bool        pass_test    = (pass_env == nullptr || std::atoi(pass_env) > 0) && workers + 1 < SpecPacketSlots;
				const uint32_t    pass         = spec && (made >= 0 ? made > 0 : pass_test) ? 1u : 0u;
				const int64_t     left = int64_t {cores} - int64_t {worker_cores} - 3 - int64_t {pass};
				char              rejected[96] = {};
				if (set)
					std::snprintf(rejected, sizeof(rejected), "KYTY_SPEC_UPLOAD_PIPE_THREADS='%.40s' not a count: the rule; ", value);
				if (cores == 0) {
					std::snprintf(why, sizeof(why), "%sthe cores not known: %u", rejected, PipeThreadsMax);
				} else if (!spec) {
					count = static_cast<uint32_t>(std::clamp<int64_t>(left, 0, PipeThreadsMax));
					std::snprintf(why, sizeof(why),
					              "%s%u cores - 3 GPU, record, upload = %lld, at most %u (no speculation threads: not mode 4); "
					              "KYTY_SPEC_UPLOAD_PIPE_THREADS sets it",
					              rejected, cores, static_cast<long long>(left), PipeThreadsMax);
				} else {
					count = static_cast<uint32_t>(std::clamp<int64_t>(left, 0, PipeThreadsMax));
					std::snprintf(why, sizeof(why),
					              "%s%u cores - %u of the %u workers' - 3 GPU, record, upload - %u state pass = %lld, at most %u; "
					              "KYTY_SPEC_UPLOAD_PIPE_THREADS sets it",
					              rejected, cores, worker_cores, workers, pass, static_cast<long long>(left), PipeThreadsMax);
				}
			}
			if (count == 0) std::printf("Upload pipe: off, the walker only: no copy threads (%s)\n", why);
			else std::printf("Upload pipe: %u copy threads (%s)\n", count, why);
			std::fflush(stdout);
			return count;
		}();
		return threads;
	}

	// KYTY_SPEC_UPLOAD_PIPE: one session (the worker, holding m_process: Help waits for the prefix meanwhile). One walker
	// (this thread) walks [m_done, head) in queue order, rereading head: it runs the protections there in order, keeps each
	// call (a lift, a Read) for its turn (m_done reaching it), and posts each copy to the copy threads (Kyty.UploadPipe),
	// which run it in 256 KiB chunks as it comes, not in fork-joined runs; it runs chunks itself whenever it has nothing
	// due. Every job is flagged done in its slot by whoever finishes it (a copy: its last chunk's thread); m_done is the
	// prefix of those flags, moved by every finisher (AdvancePrefix). The session ends caught up: every walked job done,
	// the copy threads idle. Exactness:
	// - m_done stays a prefix; every reader of it (submissions, Drain, the unmap's DrainRange, slot reuse, the idle tests,
	//   the write guard's cycle detector) is unchanged.
	// - No job runs twice: the walker runs each protection and call, and posts each copy once; a protection merged into an
	//   earlier one's call is flagged done by it and passed.
	// - A copy follows its pages' earlier protections: the walker runs every protection in order and posts a copy only
	//   once it has passed every job before it; a page's protections keep their queue order (one walker, no lanes).
	// - Staging order: a copy (or Read) whose staging bytes overlap an earlier one not done waits for it.
	// - Calls: a lift or a Read runs on the walker at its turn. Copies after it may run before it (they were pushed after the
	//   guarded write: a guarded copy stops at the armed page and reads the backing, the new bytes, as after the lift). A
	//   protection after it runs before it only over other 4 KiB pages; over its pages it waits. Any other call holds the
	//   walk until it has run.
	// - Read jobs run on the walker only (t_worker). Help runs Process as before, between sessions.
	// - The copy threads are t_worker (a texture-offload fault is a miss, as on the worker); a guarded copy's fault is
	//   resumed before the guest handler on any thread.
	void PipeSession() {
		EnsurePipe();
		const auto start = std::chrono::steady_clock::now();
		uint64_t pos = m_done.load(std::memory_order_acquire), stall_ns = 0, call_ns = 0;
		m_pipe_calls.clear();
		m_pipe_barriers = 0;
		m_pipe_staging.clear();
		for (;;) {
			call_ns += RunDueCalls();
			const uint64_t head = m_head.load(std::memory_order_acquire);
			if (pos < head) {
				if (WalkOne(pos, call_ns)) {
					++pos;
					// (The copy threads asleep, the spin gate shut or the feed short of a wake: the walker's own share now.)
					if (FeedQueued() != 0 && m_copy_busy.load(std::memory_order_relaxed) == 0 && !WakeCopiers(true))
						(void)RunOneChunk();
					continue;
				}
			} else if (m_pipe_calls.empty() && m_done.load(std::memory_order_seq_cst) >= pos && FeedQueued() == 0 &&
			           m_copy_running.load(std::memory_order_seq_cst) == 0) {
				// (No copy thread holds a chunk, or is still moving the prefix after one: none touches m_done after this.)
				break;
			}
			if (RunOneChunk()) continue;
			const auto waiting = std::chrono::steady_clock::now();
			PipeWait(pos);
			stall_ns += static_cast<uint64_t>((std::chrono::steady_clock::now() - waiting).count());
		}
		m_pipe_staging.clear();
		// (The rows Process keeps: the walker's busy time, its calls' and protections' time.)
		const auto wall = static_cast<uint64_t>((std::chrono::steady_clock::now() - start).count());
		LiveCounters::Add(LiveCounters::UploadBusyUs, (wall - std::min(wall, stall_ns)) / 1000u);
		LiveCounters::Add(LiveCounters::UploadCallUs, call_ns / 1000u);
	}

	// KYTY_SPEC_UPLOAD_PIPE: the job at `pos`, the walker's step. True: done, posted, run, or kept for its turn; false: held
	// (a staging overlap, or a kept call over its pages).
	bool WalkOne(uint64_t pos, uint64_t& call_ns) {
		if (SlotDone(pos)) { // (a protection merged into an earlier one's call: its flag moves the prefix)
			if (m_done.load(std::memory_order_acquire) == pos) AdvancePrefix();
			return true;
		}
		if (m_pipe_barriers != 0) return false;
		const Job& job = m_jobs[pos % Capacity];
		if (job.call == nullptr) {
			const uint64_t size = job.size, begin = reinterpret_cast<uint64_t>(job.destination), end = begin + size;
			if (size == 0) {
				MarkDone(pos);
				AdvancePrefix();
				return true;
			}
			if (StagingBusy(begin, end)) return false;
			PostCopy(pos, job);
			if (!SpinGate::g_open.load(std::memory_order_relaxed)) {
				// (CPUs short: the copy threads stay asleep; the walker copies, as Process did.)
				while (RunOneChunk()) {}
			} else {
				(void)WakeCopiers(false);
			}
			// (Its staging bytes busy until its flag. Read from locals: once done, its slot may be refilled.)
			m_pipe_staging.insert_or_assign(begin, std::pair<uint64_t, uint64_t> {end, pos});
			return true;
		}
		if (job.protection) {
			const auto pages = Pages(job.guest, job.guest_size);
			if (CallBlocks(pages.first, pages.second)) return false;
			(void)WakeCopiers(true); // (copies run under the protection's system call)
			const auto protecting = std::chrono::steady_clock::now();
			RunProtectionMerged(pos, std::min(m_head.load(std::memory_order_acquire), pos + MergeWindow), true);
			call_ns += static_cast<uint64_t>((std::chrono::steady_clock::now() - protecting).count());
			AdvancePrefix();
			return true;
		}
		// A call, kept for its turn.
		PipeCall kept {pos, 0, 0, false};
		if (job.kind == JobKind::Lift) {
			const auto page = Pages(reinterpret_cast<uint64_t>(job.destination), 1); // (its page: `a`)
			kept.begin = page.first, kept.end = page.second;
		} else if (job.kind == JobKind::Read) {
			const uint64_t begin = reinterpret_cast<uint64_t>(job.destination), end = begin + job.size;
			if (job.size != 0 && StagingBusy(begin, end)) return false;
			const auto pages = Pages(job.guest, job.guest_size);
			kept.begin = pages.first, kept.end = pages.second;
			if (job.size != 0) m_pipe_staging.insert_or_assign(begin, std::pair<uint64_t, uint64_t> {end, pos});
		} else {
			kept.barrier = true; // (any other call: the walk waits until it has run)
			++m_pipe_barriers;
		}
		m_pipe_calls.push_back(kept);
		return true;
	}

	// KYTY_SPEC_UPLOAD_PIPE: the kept calls whose turn came (m_done at them), in order, on the walker. Returns their time, ns.
	uint64_t RunDueCalls() {
		uint64_t ns = 0;
		while (!m_pipe_calls.empty()) {
			const PipeCall call = m_pipe_calls.front();
			if (m_done.load(std::memory_order_acquire) < call.sequence) break;
			const auto called = std::chrono::steady_clock::now();
			CallJob(m_jobs[call.sequence % Capacity]);
			ns += static_cast<uint64_t>((std::chrono::steady_clock::now() - called).count());
			MarkDone(call.sequence);
			AdvancePrefix();
			if (call.barrier) --m_pipe_barriers;
			m_pipe_calls.pop_front();
		}
		return ns;
	}

	// KYTY_SPEC_UPLOAD_PIPE: whether a kept call is over the 4 KiB pages [begin, end) (any other call: over every page).
	[[nodiscard]] bool CallBlocks(uint64_t begin, uint64_t end) const {
		for (const auto& call: m_pipe_calls)
			if (call.barrier || (call.begin < end && begin < call.end)) return true;
		return false;
	}

	// KYTY_SPEC_UPLOAD_PIPE: whether the staging bytes [begin, end) overlap a posted copy (or kept Read) not done. Done ones
	// met are dropped. (The set's ranges are disjoint: one overlapping a new range is done before the new one goes in.)
	bool StagingBusy(uint64_t begin, uint64_t end) {
		auto it = m_pipe_staging.upper_bound(begin);
		if (it != m_pipe_staging.begin()) --it;
		while (it != m_pipe_staging.end() && it->first < end) {
			if (it->second.first > begin) {
				if (!SlotDone(it->second.second)) return true;
				it = m_pipe_staging.erase(it);
				continue;
			}
			++it;
		}
		// (Kept small: the done ones dropped once it grows.)
		if (m_pipe_staging.size() > 8192)
			for (auto at = m_pipe_staging.begin(); at != m_pipe_staging.end();)
				at = SlotDone(at->second.second) ? m_pipe_staging.erase(at) : std::next(at);
		return false;
	}

	// KYTY_SPEC_UPLOAD_PIPE: the walker can go no further now and has no chunk to run: the copy threads woken if they can
	// take some, then a wait for the prefix's front job when it is a posted copy, its flag then moving the prefix.
	void PipeWait(uint64_t pos) {
		(void)WakeCopiers(true);
		AdvancePrefix();
		const uint64_t front = m_done.load(std::memory_order_acquire);
		// (Caught up and a copy thread finishing; or the front a kept call whose turn just came, or just done.)
		if (front >= pos || SlotWord(front).load(std::memory_order_acquire) != (SlotTag(front) | 1u)) {
			_mm_pause();
			return;
		}
		AwaitDone(front);
		AdvancePrefix();
	}

	// KYTY_SPEC_UPLOAD_PIPE: m_done moved over the slots flagged done after it (any finisher, after its flag; monotonic).
	void AdvancePrefix() {
		uint64_t at = m_done.load(std::memory_order_seq_cst);
		for (;;) {
			uint64_t end = at;
			while (TaggedDone(end)) ++end; // (a slot not pushed yet holds an earlier sequence's tag: never done)
			if (end == at) return;
			if (m_done.compare_exchange_weak(at, end, std::memory_order_seq_cst, std::memory_order_seq_cst)) {
				if (m_waiters.load(std::memory_order_seq_cst) != 0) m_done.notify_all();
				at = end;
			}
		}
	}

	// KYTY_SPEC_UPLOAD_PIPE: a slot's word, ((sequence + 1) << 2) | state (1 posted, a copy whose chunks are out; 2 done); a
	// word of another sequence (an earlier use of the slot, or 0) is neither. Tagged, a late flag of a slot's earlier job
	// never reads as the new job's. Written in a session only (the walker, and the copy threads for its copies), made by
	// the first session before its copy threads start.
	[[nodiscard]] static constexpr uint64_t SlotTag(uint64_t sequence) { return (sequence + 1) << 2u; }
	[[nodiscard]] std::atomic<uint64_t>& SlotWord(uint64_t sequence) const { return m_slot_words[sequence % Capacity]; }
	// (The walker or a copy's last chunk; wakes the walker asleep on the word: AwaitDone. Before the prefix can pass the
	// slot, so before the slot can be refilled.)
	void MarkDone(uint64_t sequence) {
		SlotWord(sequence).store(SlotTag(sequence) | 2u, std::memory_order_seq_cst);
		if (m_slot_waiters.load(std::memory_order_seq_cst) != 0) SlotWord(sequence).notify_all();
	}
	[[nodiscard]] bool TaggedDone(uint64_t sequence) const {
		return SlotWord(sequence).load(std::memory_order_acquire) == (SlotTag(sequence) | 2u);
	}
	[[nodiscard]] bool SlotDone(uint64_t sequence) const {
		return m_done.load(std::memory_order_acquire) > sequence || TaggedDone(sequence);
	}
	// (The walker.) A posted copy: a short spin, then asleep on the slot's word until its last chunk flags it done (m_done
	// passing it ends the wait too).
	void AwaitDone(uint64_t sequence) {
		for (int spin = 0; spin < 4000; ++spin) {
			if (SlotDone(sequence)) return;
			_mm_pause();
		}
		auto&          word   = SlotWord(sequence);
		const uint64_t posted = SlotTag(sequence) | 1u;
		m_slot_waiters.fetch_add(1, std::memory_order_seq_cst);
		while (word.load(std::memory_order_seq_cst) == posted && m_done.load(std::memory_order_acquire) <= sequence)
			word.wait(posted, std::memory_order_seq_cst);
		m_slot_waiters.fetch_sub(1, std::memory_order_relaxed);
	}

	// KYTY_SPEC_UPLOAD_PIPE: the copy threads, the feed of copy chunks and each posted copy's chunks left (made once).
	static constexpr uint64_t PipeChunk     = uint64_t {256} << 10u;
	static constexpr uint64_t PipeWakeBytes = uint64_t {128} << 10u; // (less queued: no wake but the walker's force)
	static constexpr uint64_t FeedCapacity  = uint64_t {1} << 12u;
	struct FeedSlot {
		std::atomic<uint64_t> sequence {0}, offset {0}, size {0};
	};
	struct FeedItem {
		uint64_t sequence = 0, offset = 0, size = 0;
	};
	void EnsurePipe() {
		if (m_pipe_started) return;
		m_pipe_started = true;
		m_slot_words   = std::make_unique<std::atomic<uint64_t>[]>(Capacity);
		m_chunks_left  = std::make_unique<std::atomic<uint32_t>[]>(Capacity);
		m_feed         = std::make_unique<FeedSlot[]>(FeedCapacity);
		const uint32_t threads = PipeCopyThreads();
		for (uint32_t i = 0; i < threads; ++i) std::thread([this] { PipeCopyRun(); }).detach(); // (never stopped)
		std::printf("Upload pipe: on (KYTY_SPEC_UPLOAD_PIPE: one walker runs protections and calls in order, %u copy threads "
		            "take the copies as they come)\n",
		            threads);
		std::fflush(stdout);
	}
	[[nodiscard]] uint64_t FeedQueued() const {
		const uint64_t posted = m_feed_posted.load(std::memory_order_acquire), next = m_feed_next.load(std::memory_order_acquire);
		return posted > next ? posted - next : 0;
	}
	// (The walker only.) The copy's slot flagged posted and its chunks into the feed; a full feed: chunks run here until
	// there is room.
	void PostCopy(uint64_t sequence, const Job& job) {
		const uint64_t size = job.size, chunks = (size + PipeChunk - 1) / PipeChunk;
		m_chunks_left[sequence % Capacity].store(static_cast<uint32_t>(chunks), std::memory_order_relaxed);
		SlotWord(sequence).store(SlotTag(sequence) | 1u, std::memory_order_relaxed); // (published by the feed's store)
		uint64_t posted = m_feed_posted.load(std::memory_order_relaxed);
		for (uint64_t chunk = 0; chunk < chunks; ++chunk) {
			while (posted - m_feed_next.load(std::memory_order_acquire) >= FeedCapacity)
				if (!RunOneChunk()) _mm_pause();
			const uint64_t offset = chunk * PipeChunk, bytes = std::min(PipeChunk, size - offset);
			auto&          slot   = m_feed[posted % FeedCapacity];
			slot.sequence.store(sequence, std::memory_order_relaxed);
			slot.offset.store(offset, std::memory_order_relaxed);
			slot.size.store(bytes, std::memory_order_relaxed);
			m_feed_bytes.fetch_add(bytes, std::memory_order_relaxed);
			m_feed_posted.store(++posted, std::memory_order_seq_cst);
		}
	}
	// (Any feed taker.) The next chunk, or false when none is queued. Its fields are read before the take: a slot is
	// written again only once the take index has passed it, so a take that succeeds read this chunk's.
	bool TakeFeed(FeedItem& item) {
		uint64_t next = m_feed_next.load(std::memory_order_acquire);
		for (;;) {
			if (next >= m_feed_posted.load(std::memory_order_acquire)) return false;
			const auto& slot = m_feed[next % FeedCapacity];
			item             = {slot.sequence.load(std::memory_order_relaxed), slot.offset.load(std::memory_order_relaxed),
			                    slot.size.load(std::memory_order_relaxed)};
			if (m_feed_next.compare_exchange_weak(next, next + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
				m_feed_bytes.fetch_sub(item.size, std::memory_order_relaxed);
				return true;
			}
		}
	}
	// A chunk copied (through CopyFrom, as every copy); the copy's last chunk flags it done and moves the prefix.
	void RunChunk(const FeedItem& item) {
		const Job& job = m_jobs[item.sequence % Capacity];
		CopyFrom(job.destination + item.offset, job.source + item.offset, job.guest + item.offset, item.size);
		if (m_chunks_left[item.sequence % Capacity].fetch_sub(1, std::memory_order_acq_rel) == 1) {
			MarkDone(item.sequence);
			AdvancePrefix();
		}
	}
	// (The walker.)
	bool RunOneChunk() {
		FeedItem item;
		if (!TakeFeed(item)) return false;
		RunChunk(item);
		return true;
	}
	// (The walker.) Copy threads woken for the queued chunks: as many as the queued chunks when a wake's worth of bytes is
	// queued; `force` (before a protection, or a stall): one, when none is at the feed. Never with the spin gate shut.
	// True when it woke one.
	bool WakeCopiers(bool force) {
		if (!SpinGate::g_open.load(std::memory_order_relaxed)) return false;
		const uint32_t sleepers = m_feed_sleepers.load(std::memory_order_seq_cst);
		const uint64_t queued   = FeedQueued();
		if (sleepers == 0 || queued == 0) return false;
		uint64_t want = m_feed_bytes.load(std::memory_order_relaxed) >= PipeWakeBytes ? queued : 0;
		if (want == 0 && force && m_copy_busy.load(std::memory_order_relaxed) == 0) want = 1;
		want = std::min<uint64_t>(want, sleepers);
		for (uint64_t i = 0; i < want; ++i) m_feed_posted.notify_one();
		return want != 0;
	}
	// A copy thread: the feed's chunks while there are any, a short spin (the spin gate open), then asleep until a wake.
	void PipeCopyRun() {
		LocalPlatform::SetThreadName("Kyty.UploadPipe");
		LocalPlatform::AvoidCpuList(std::getenv("KYTY_RENDER_CPUS"));
		t_worker = true; // (a texture-offload fault is a miss, as on the worker: the render thread may wait for these)
		for (;;) {
			m_copy_busy.fetch_add(1, std::memory_order_seq_cst);
			for (uint32_t spin = 0;;) {
				if (FeedQueued() != 0) {
					// (Counted from before the take to after the prefix moved: the session's end waits for none.)
					m_copy_running.fetch_add(1, std::memory_order_seq_cst);
					FeedItem   item;
					const bool took = TakeFeed(item);
					if (took) RunChunk(item);
					m_copy_running.fetch_sub(1, std::memory_order_seq_cst);
					if (took) {
						spin = 0;
						continue;
					}
				}
				if (++spin > 500 || !SpinGate::g_open.load(std::memory_order_relaxed)) break;
				_mm_pause();
			}
			m_copy_busy.fetch_sub(1, std::memory_order_seq_cst);
			const uint64_t seen = m_feed_posted.load(std::memory_order_seq_cst);
			m_feed_sleepers.fetch_add(1, std::memory_order_seq_cst);
			if (m_feed_next.load(std::memory_order_seq_cst) >= seen) m_feed_posted.wait(seen, std::memory_order_seq_cst);
			m_feed_sleepers.fetch_sub(1, std::memory_order_seq_cst);
		}
	}

	// Jobs [done, head) in queue order (the caller holds m_process), m_done published as they finish; `helper`: stops at a
	// call other than a write protection; `merge` (KYTY_SPEC_UPLOAD_PIPE): adjacent queued protections merged. Returns the
	// jobs' new end.
	uint64_t Process(uint64_t done, uint64_t head, bool helper, bool merge) {
		const auto start = std::chrono::steady_clock::now();
		const auto since = [](std::chrono::steady_clock::time_point from) {
			return static_cast<uint64_t>((std::chrono::steady_clock::now() - from).count()) / 1000u;
		};
		const auto     call  = [](const Job& job) { CallJob(job); };
		// (KYTY_UPLOAD_LANES, read once a batch. Only the worker splits: a helping render thread would wait on the second
		// thread, the stall Help exists to avoid, and the second thread takes its creator's priority and CPUs.)
		const uint32_t lanes = helper ? 1u : kyty_local_upload_lanes.load(std::memory_order_relaxed);
		m_progress           = !helper && ProgressOn(); // (KYTY_UPLOAD_PROGRESS: the worker's batches)
		m_merge              = merge;
		while (done < head) {
			if (const Job& job = m_jobs[done % Capacity]; job.call != nullptr && !job.protection) {
				if (helper) break;
				const auto called = std::chrono::steady_clock::now();
				call(job);
				LiveCounters::Add(LiveCounters::UploadCallUs, since(called));
				++done;
				if (m_progress) Publish(done);
				continue;
			}
			// The jobs up to the next other call (which keeps its place): their write protections first, in their
			// order, then their copies, which are independent of each other. A copy follows the protection of its
			// pages (pushed before it) either way: one pushed after the copy only re-arms its pages sooner, and the
			// copy that is its own still comes after it. (Each upload pushes its protections then its copies: in
			// queue order the copies came in runs of one upload's few, a few KiB to tens of KiB each.)
			uint64_t end = done, bytes = 0;
			for (; end < head; ++end) {
				const Job& next = m_jobs[end % Capacity];
				if (next.call == nullptr) bytes += next.size;
				else if (!next.protection) break;
			}
			const auto protecting = std::chrono::steady_clock::now();
			Protections(done, end, lanes);
			LiveCounters::Add(LiveCounters::UploadCallUs, since(protecting));
			// (KYTY_UPLOAD_PROGRESS: the run's protections all done, published up to its first copy; the copies from there:
			// a slot below a published value may be refilled by Add at once, so none is read again.)
			uint64_t copies_from = done;
			if (m_progress) {
				while (copies_from < end && m_jobs[copies_from % Capacity].call != nullptr) ++copies_from;
				Publish(copies_from);
			}
			CopyRun(copies_from, end, bytes);
			done = end;
		}
		if (!helper) LiveCounters::Add(LiveCounters::UploadBusyUs, since(start));
		m_done.store(done, std::memory_order_seq_cst);
		if (m_waiters.load(std::memory_order_seq_cst) != 0) m_done.notify_all();
		return done;
	}

	static void CallJob(const Job& job) { job.call(job.context, reinterpret_cast<uint64_t>(job.destination), job.size); }

	// KYTY_UPLOAD_PROGRESS (live upload_progress; -1, the default: on under KYTY_SPECULATE=4; 1 on, 0 off): m_done
	// published as the batch goes, in its order (each call, a run's protections, each copy or the done prefix of a
	// shared run's chunks), not only at its end: a DrainRange waits for its range's jobs, not for the rest of the batch
	// they fell in (1.3-3.5 ms in a spike frame). Every job below a published value is done.
	static bool ProgressOn() {
		static auto& on    = LiveSwitch::Find("upload_progress", -1);
		const auto   value = on.load(std::memory_order_relaxed);
		return value >= 0 ? value != 0 : kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4;
	}
	// (Monotonic: a later value only; any thread inside Process, the shared run's helpers included.)
	void Publish(uint64_t done) {
		uint64_t at = m_done.load(std::memory_order_relaxed);
		while (at < done)
			if (m_done.compare_exchange_weak(at, done, std::memory_order_seq_cst, std::memory_order_relaxed)) {
				if (m_waiters.load(std::memory_order_seq_cst) != 0) m_done.notify_all();
				return;
			}
	}

	// KYTY_UPLOAD_LANES: the jobs each lane needs at least for a run to be split: the second thread's wake-up (~3-4 us)
	// plus the waker's FUTEX_WAKE (~2-3 us) against ~3 us a protection; lane 1 runs late by the wake-up, so it needs more.
	static constexpr size_t LaneMinJobs0 = 2;
	static constexpr size_t LaneMinJobs1 = 4;

	// The write protections among jobs [first, last), in queue order (lanes 1), or split by region over two lanes (2, 3 =
	// with the order checks) when no job spans regions and both lanes get LaneMinJobs. (The caller holds m_process.)
	void Protections(uint64_t first, uint64_t last, uint32_t lanes) {
		if (lanes >= 2) {
			m_lane_jobs[0].clear();
			m_lane_jobs[1].clear();
			bool serial = false;
			for (uint64_t i = first; i < last && !serial; ++i) {
				const Job& job = m_jobs[i % Capacity];
				if (job.call == nullptr) continue;
				const uint64_t region = job.guest / LaneRegion;
				if (region != (job.guest + std::max<uint64_t>(job.guest_size, 1) - 1) / LaneRegion) serial = true;
				else m_lane_jobs[region & 1u].push_back(i);
			}
			// (Only while the process's CPUs keep idle ones, the spin gate open: else the second thread waits for a CPU.)
			if (!serial && m_lane_jobs[0].size() >= LaneMinJobs0 && m_lane_jobs[1].size() >= LaneMinJobs1 &&
			    SpinGate::g_open.load(std::memory_order_relaxed)) {
				RunLanes(lanes == 3, first, last);
				return;
			}
		}
		if (!m_merge) {
			for (uint64_t i = first; i < last; ++i)
				if (const Job& next = m_jobs[i % Capacity]; next.call != nullptr) CallJob(next);
			return;
		}
		// (KYTY_SPEC_UPLOAD_PIPE: a member merged into an earlier call is passed.)
		m_merged_ahead.clear();
		for (uint64_t i = first; i < last; ++i) {
			if (m_jobs[i % Capacity].call == nullptr) continue;
			if (std::binary_search(m_merged_ahead.begin(), m_merged_ahead.end(), i)) continue;
			RunProtectionMerged(i, last, false);
		}
	}

	// KYTY_SPEC_UPLOAD_PIPE: the protection at `first` (a serial run's, or the pipe walker's: `walking`), with the queued
	// protections after it in [first + 1, last) (at most MergeWindow jobs on, MergeMembers in all) that continue its range,
	// run as one call. A member must have the same call, manager (context) and merged form (the same purpose and the same
	// Honour path: the call reads each page's state when it runs), lie address-adjacent to the range so far, and no job
	// between `first` and it may be over its 4 KiB pages: a copy reading them, another protection, a lift (its page), a
	// Read; any other call ends the search. So a page's protections keep their order, and no member's pages see a job
	// between run on the other side of it. (A guarded write's arm comes with its lift: a job between.) On the walker a
	// member must not be under a kept call either, and the lead and its members are flagged done; in a serial run the
	// members are passed (m_merged_ahead).
	static constexpr uint64_t MergeWindow = 32, MergeMembers = 16;
	void RunProtectionMerged(uint64_t first, uint64_t last, bool walking) {
		const Job& lead = m_jobs[first % Capacity];
		using Range     = std::pair<uint64_t, uint64_t>;
		thread_local std::vector<uint64_t> members;
		thread_local std::vector<Range>    touched;
		members.clear(), touched.clear();
		uint64_t   low = reinterpret_cast<uint64_t>(lead.destination), high = low + lead.size;
		const auto over = [&](const Range& pages) {
			for (const auto& other: touched)
				if (pages.first < other.second && other.first < pages.second) return true;
			return false;
		};
		for (uint64_t j = first + 1; lead.merge != nullptr && j < last && j < first + MergeWindow && members.size() + 1 < MergeMembers;
		     ++j) {
			if (walking ? SlotDone(j) : std::binary_search(m_merged_ahead.begin(), m_merged_ahead.end(), j)) continue;
			const Job& job = m_jobs[j % Capacity];
			if (job.call != nullptr && !job.protection) {
				if (job.kind == JobKind::Lift) touched.push_back(Pages(reinterpret_cast<uint64_t>(job.destination), 1));
				else if (job.kind == JobKind::Read) touched.push_back(Pages(job.guest, job.guest_size));
				else break;
				continue;
			}
			const auto pages = Pages(job.guest, job.guest_size);
			if (job.protection && job.call == lead.call && job.context == lead.context && job.merge == lead.merge) {
				const uint64_t at = reinterpret_cast<uint64_t>(job.destination), size = job.size;
				if ((at == high || at + size == low) && !over(pages) && !(walking && CallBlocks(pages.first, pages.second))) {
					members.push_back(j);
					low  = std::min(low, at);
					high = std::max(high, at + size);
					continue;
				}
			}
			touched.push_back(pages);
		}
		if (members.empty()) CallJob(lead);
		else lead.merge(lead.context, low, high - low, members.size() + 1);
		if (walking) {
			for (const uint64_t j: members) MarkDone(j);
			MarkDone(first); // (last: its slot may be refilled once the prefix passes it)
		} else {
			for (const uint64_t j: members) m_merged_ahead.insert(std::lower_bound(m_merged_ahead.begin(), m_merged_ahead.end(), j), j);
		}
	}

	// A guest range's 4 KiB pages [begin, end) (an empty one its first page).
	[[nodiscard]] static std::pair<uint64_t, uint64_t> Pages(uint64_t at, uint64_t size) {
		constexpr uint64_t Page = 4096;
		return {at & ~(Page - 1), (at + std::max<uint64_t>(size, 1) + Page - 1) & ~(Page - 1)};
	}

	// Lane 1's jobs to the second thread, lane 0's here, then waits for the second (a short spin: the lanes are about even;
	// then a futex), so the run's copies start after both. (The caller holds m_process: one run in lanes at a time.)
	void RunLanes(bool verify, uint64_t first, uint64_t last) {
		if (!m_lane_thread.joinable()) {
			m_lane_thread = std::thread([this] { LaneRun(); });
			std::printf("Upload lanes: on (KYTY_UPLOAD_LANES=%u: a second upload thread runs the odd 4 MiB regions' write protections; "
			            "it sleeps between runs)\n",
			            verify ? 3u : 2u);
			std::fflush(stdout);
		}
		m_lane_verify.store(verify, std::memory_order_relaxed);
		const uint32_t generation = m_lane_posted.load(std::memory_order_relaxed) + 1;
		m_lane_posted.store(generation, std::memory_order_seq_cst);
		m_lane_posted.notify_one();
		for (const uint64_t i: m_lane_jobs[0]) CallJob(m_jobs[i % Capacity]);
		// (The join, after this lane's own jobs: its wait is the split's cost, upload_lane_join_us.)
		const auto joining = std::chrono::steady_clock::now();
		for (int spin = 0; spin < 2000 && m_lane_finished.load(std::memory_order_acquire) != generation; ++spin) _mm_pause();
		if (m_lane_finished.load(std::memory_order_seq_cst) != generation) {
			LiveCounters::Add(LiveCounters::UploadLaneFutexJoins);
			m_lane_main_waiting.store(true, std::memory_order_seq_cst);
			for (uint32_t finished; (finished = m_lane_finished.load(std::memory_order_seq_cst)) != generation;)
				m_lane_finished.wait(finished, std::memory_order_seq_cst);
			m_lane_main_waiting.store(false, std::memory_order_relaxed);
		}
		LiveCounters::Add(LiveCounters::UploadLaneJoinUs,
		                  static_cast<uint64_t>((std::chrono::steady_clock::now() - joining).count()) / 1000u);
		LiveCounters::Add(LiveCounters::UploadLaneRuns);
		LiveCounters::Add(LiveCounters::UploadLaneJobs, m_lane_jobs[1].size());
		if (verify) LiveCounters::Add(LiveCounters::UploadLaneMismatch, CheckLanes(first, last));
	}

	// The second lane's thread: each posted run's lane-1 jobs in their order, then sleeps again (never spins: a spinning
	// thread in the process makes every protection's TLB flush reach one more CPU, ~0.27 us each).
	void LaneRun() {
		LocalPlatform::SetThreadName("Kyty.Upload2");
		t_worker = true;
		LocalPlatform::PinThreadToCpuList(std::getenv("KYTY_RECORDING_CPUS"));
		RaisePriority();
		uint32_t seen = 0;
		for (;;) {
			uint32_t posted = 0;
			while ((posted = m_lane_posted.load(std::memory_order_acquire)) == seen) {
				if (m_stop.load(std::memory_order_seq_cst)) return;
				m_lane_posted.wait(seen, std::memory_order_acquire);
			}
			seen = posted;
			// (A post runs and finishes even when stopping: its poster waits for it; the destructor's wake carries no jobs
			// of its own: the worker has stopped by then.)
			const bool verify = m_lane_verify.load(std::memory_order_relaxed);
			if (verify) m_lane_ran.clear();
			for (const uint64_t i: m_lane_jobs[1]) {
				if (verify) m_lane_ran.push_back(i);
				CallJob(m_jobs[i % Capacity]);
			}
			m_lane_finished.store(seen, std::memory_order_seq_cst);
			if (m_lane_main_waiting.load(std::memory_order_seq_cst)) m_lane_finished.notify_one();
			if (m_stop.load(std::memory_order_seq_cst)) return;
		}
	}

	// KYTY_UPLOAD_LANES=3, after both lanes are done: the order rules a split must keep, the violations counted. Each lane holds
	// only its regions (one region never in both: its protections keep their queue order), in queue order; the second thread
	// ran exactly lane 1's jobs, in that order.
	[[nodiscard]] uint64_t CheckLanes(uint64_t first, uint64_t last) const {
		uint64_t mismatches = 0;
		// (Every protection job of the run in one of the lanes, each single-region.)
		size_t protections = 0;
		for (uint64_t i = first; i < last; ++i) {
			const Job& job = m_jobs[i % Capacity];
			if (job.call == nullptr) continue;
			protections++;
			if (job.guest / LaneRegion != (job.guest + std::max<uint64_t>(job.guest_size, 1) - 1) / LaneRegion) mismatches++;
		}
		if (protections != m_lane_jobs[0].size() + m_lane_jobs[1].size()) mismatches++;
		for (uint32_t lane = 0; lane < 2; ++lane) {
			for (size_t k = 0; k < m_lane_jobs[lane].size(); ++k) {
				const Job& job = m_jobs[m_lane_jobs[lane][k] % Capacity];
				if (((job.guest / LaneRegion) & 1u) != lane) mismatches++;
				if (k != 0 && m_lane_jobs[lane][k] <= m_lane_jobs[lane][k - 1]) mismatches++;
			}
		}
		for (const uint64_t a: m_lane_jobs[0])
			for (const uint64_t b: m_lane_jobs[1])
				if (m_jobs[a % Capacity].guest / LaneRegion == m_jobs[b % Capacity].guest / LaneRegion) mismatches++;
		if (m_lane_ran != m_lane_jobs[1]) mismatches++;
		return mismatches;
	}

	// The copies among jobs [first, last) (calls skipped), `bytes` in all. Copies of 1 MiB or more together are
	// shared with helper threads of their own (not the render thread's page comparisons' pool, which would wait
	// behind it), as one job over their bytes: an image the game streams in comes as many copies (its pieces, its
	// small neighbours), and a frame where it brings a new area (~1 GB of textures, every submission waiting for their
	// copies) stalled ~300 ms. (Copied one by one, a walk's slow frames at Boletarian Palace 1-3 waited 25-28 ms for
	// them; as runs, 14 ms.)
	void CopyRun(uint64_t first, uint64_t last, uint64_t bytes) {
		constexpr uint64_t Chunk = uint64_t {256} << 10u;
		if (bytes == 0) return;
		if (bytes < 4 * Chunk) {
			for (; first < last; ++first)
				if (const Job& job = m_jobs[first % Capacity]; job.call == nullptr) {
					CopyFrom(job.destination, job.source, job.guest, job.size);
					if (m_progress) Publish(first + 1); // (KYTY_UPLOAD_PROGRESS)
				}
			return;
		}
		// The run's copies, each at its offset in the run's bytes.
		m_run.clear();
		uint64_t offset = 0;
		for (; first < last; ++first) {
			const Job& job = m_jobs[first % Capacity];
			if (job.call != nullptr) continue;
			m_run.push_back({offset, job.destination, job.source, job.guest, first});
			offset += job.size;
		}
		m_run.push_back({offset, nullptr, nullptr, 0, last});
		// (KYTY_UPLOAD_PROGRESS: a flag a chunk, the done prefix in chunks.)
		const uint64_t chunks = (bytes + Chunk - 1) / Chunk;
		if (m_progress) {
			if (m_chunk_flags_size < chunks) {
				m_chunk_flags      = std::make_unique<std::atomic<uint8_t>[]>(chunks);
				m_chunk_flags_size = chunks;
			}
			for (uint64_t c = 0; c < chunks; ++c) m_chunk_flags[c].store(0, std::memory_order_relaxed);
			m_chunk_prefix.store(0, std::memory_order_relaxed);
		}
		m_run_chunks = chunks;
		LiveCounters::Add(LiveCounters::UploadSharedRuns);
		static auto* const pool = new ParallelPages::Pool(7, "Kyty.UploadCopy"); // (never destroyed)
		pool->For(bytes, Chunk, [](void* at, uint64_t from, uint64_t to) {
			auto&       self  = *static_cast<Worker*>(at);
			const auto& run   = self.m_run;
			const auto  chunk = from / Chunk;
			// The copy holding byte `from`, then the ones after it up to `to`.
			auto it = std::prev(std::upper_bound(run.begin(), std::prev(run.end()), from,
			                                     [](uint64_t at_byte, const RunCopy& copy) { return at_byte < copy.offset; }));
			for (; from < to; ++it) {
				const auto end  = std::min(to, std::next(it)->offset);
				const auto skip = from - it->offset;
				CopyFrom(it->destination + skip, it->source + skip, it->guest + skip, end - from);
				from = end;
			}
			if (self.m_progress) self.ChunkDone(chunk);
		}, this);
	}

	// The guest's bytes read through its own mapping where it can (Windows): a page of the backing view faults on its
	// first read (~1.2 us: 2.3-3.8 GB/s on one thread, 6-11 GB/s on eight), where most uploaded pages were just
	// written through the guest mapping, which has them (the waits of those 1-3 slow frames: 18 -> 4 ms). A page that
	// mapping cannot read (GPU-written, read-protected) ends that part (GuardedCopy); the rest comes from the backing
	// view, the same bytes (unmapping a range waits for its jobs: the guest mapping is the backing's).
	static void CopyFrom(uint8_t* destination, const uint8_t* backing, uint64_t guest, uint64_t size) {
#if defined(_WIN32)
		if (const auto left = Common::GuardedCopy(destination, reinterpret_cast<const void*>(guest), size); left != 0) {
			LiveCounters::Add(LiveCounters::UploadGuardStops);
			const auto done = size - left;
			std::memcpy(destination + done, backing + done, left);
		}
#else
		std::memcpy(destination, backing, size);
#endif
	}

	struct RunCopy {
		uint64_t       offset;
		uint8_t*       destination;
		const uint8_t* source;
		uint64_t       guest;
		uint64_t       job; // (its queue index; the end entry: the run's end)
	};
	// KYTY_UPLOAD_PROGRESS: shared-run chunk `chunk` done (on the thread that copied it). The done prefix extended by
	// whoever finds the next chunk flagged; its first unfinished copy published (the copies before it, and the run's
	// protections, done).
	void ChunkDone(uint64_t chunk) {
		constexpr uint64_t Chunk = uint64_t {256} << 10u;
		// (seq_cst: two finishers each flagging and then scanning see each other's flag, no stalled prefix.)
		m_chunk_flags[chunk].store(1, std::memory_order_seq_cst);
		uint64_t prefix = m_chunk_prefix.load(std::memory_order_seq_cst);
		while (prefix < m_run_chunks && m_chunk_flags[prefix].load(std::memory_order_seq_cst) != 0)
			if (m_chunk_prefix.compare_exchange_weak(prefix, prefix + 1, std::memory_order_acq_rel, std::memory_order_acquire))
				++prefix;
		const uint64_t done_bytes = std::min(prefix * Chunk, m_run.back().offset);
		// (Entry k's offset is copy k-1's end. The first entry k >= 1 whose offset is past done_bytes: copy k-1 is the
		// first not wholly done, the ones before it are. None: every copy done, the end entry's job, the run's end.)
		const auto it = std::upper_bound(std::next(m_run.begin()), m_run.end(), done_bytes,
		                                 [](uint64_t at_byte, const RunCopy& copy) { return at_byte < copy.offset; });
		Publish(std::prev(it)->job);
	}
	bool                                     m_progress = false; // (the batch's KYTY_UPLOAD_PROGRESS, under m_process)
	// KYTY_SPEC_UPLOAD_PIPE (under m_process): whether the batch merges protections, and a serial run's members merged
	// into an earlier call, sorted. The walker's kept calls, how many of them are barriers, the staging bytes of posted
	// copies and kept Reads (begin -> end, sequence), and whether the copy threads exist; the slots' words, each posted
	// copy's chunks left and the feed (made once); the feed's post and take indexes, its queued bytes, its sleepers, the
	// copy threads awake and those taking or running a chunk; the walker asleep on a slot's word.
	struct PipeCall {
		uint64_t sequence, begin, end; // (begin, end: its 4 KiB pages)
		bool     barrier;
	};
	bool                                              m_merge = false;
	std::vector<uint64_t>                             m_merged_ahead;
	std::deque<PipeCall>                              m_pipe_calls;
	uint32_t                                          m_pipe_barriers = 0;
	std::map<uint64_t, std::pair<uint64_t, uint64_t>> m_pipe_staging;
	bool                                              m_pipe_started = false;
	std::unique_ptr<std::atomic<uint64_t>[]>          m_slot_words;
	std::unique_ptr<std::atomic<uint32_t>[]>          m_chunks_left;
	std::unique_ptr<FeedSlot[]>                       m_feed;
	alignas(64) std::atomic<uint64_t>                 m_feed_posted {0};
	alignas(64) std::atomic<uint64_t>                 m_feed_next {0};
	alignas(64) std::atomic<uint64_t>                 m_feed_bytes {0};
	std::atomic<uint32_t>                             m_feed_sleepers {0};
	std::atomic<uint32_t>                             m_copy_busy {0};    // (awake: at the feed or spinning)
	std::atomic<uint32_t>                             m_copy_running {0}; // (holding a chunk, or about to take one)
	std::atomic<uint32_t>                             m_slot_waiters {0};
	std::unique_ptr<std::atomic<uint8_t>[]>  m_chunk_flags;
	uint64_t                                 m_chunk_flags_size = 0, m_run_chunks = 0;
	alignas(64) std::atomic<uint64_t>        m_chunk_prefix {0};
	std::vector<RunCopy>               m_run; // (the worker's)
	// KYTY_UPLOAD_LANES: the run's jobs per lane (written under m_process before a post, read by the second thread until it
	// finishes that post), the second thread, its post / finish generations, and whether the poster sleeps on the finish.
	std::vector<uint64_t>              m_lane_jobs[2];
	std::vector<uint64_t>              m_lane_ran; // (=3: the second thread's)
	std::thread                        m_lane_thread;
	alignas(64) std::atomic<uint32_t>  m_lane_posted {0};
	alignas(64) std::atomic<uint32_t>  m_lane_finished {0};
	std::atomic<bool>                  m_lane_main_waiting {false};
	std::atomic<bool>                  m_lane_verify {false};
	std::unique_ptr<Job[]>             m_jobs;
	alignas(64) std::atomic<uint64_t>  m_head {0};
	alignas(64) std::atomic<uint64_t>  m_done {0};
	alignas(64) std::atomic<bool>      m_sleeping {false};
	std::atomic<uint32_t>              m_waiters {0};
	std::atomic<bool>                  m_stop {false};
	std::mutex                         m_process; // held by whichever thread runs jobs (the worker, or Help)
	std::thread                        m_thread;
};

[[nodiscard]] inline bool Enabled() {
	return kyty_local_async_upload_mode.load(std::memory_order_relaxed) != 0;
}

// Set by the render thread before its first push; waits stay armed after the switch goes off.
inline std::atomic<bool> g_used {false};

inline Worker& Get() {
	// Never destroyed: threads that still record or submit may outlive static destruction.
	static Worker* const worker = new Worker();
	g_used.store(true, std::memory_order_relaxed);
	return *worker;
}

// Render thread: the sequence a submission recorded now must wait for (0: nothing pending).
[[nodiscard]] inline uint64_t SubmitSequence() {
	return g_used.load(std::memory_order_relaxed) ? Get().Pushed() : 0;
}

// Any thread, with a sequence taken on the render thread.
inline void Wait(uint64_t sequence) {
	if (sequence != 0) Get().Wait(sequence);
}

// Render thread: every recorded copy reaches staging memory before this returns; the jobs the worker is not at made
// here where kyty_local_upload_help says (above). `serve`: the guest commands queued for the GPU thread run meanwhile
// (KYTY_SERVE_IN_WAITS); only from a drain that no command and no submit is inside.
inline void Drain(bool serve = false) {
	if (const auto sequence = SubmitSequence(); sequence != 0) {
		const auto help = kyty_local_upload_help.load(std::memory_order_relaxed);
		if (help == 1 || (help == 2 && !SpinGate::g_open.load(std::memory_order_relaxed)))
			Get().Help(sequence);
		else
			Get().Wait(sequence, serve);
	}
}

// Render thread: every recorded job on guest [vaddr, vaddr + size) is done before this returns.
inline void DrainRange(uint64_t vaddr, uint64_t size) {
	if (g_used.load(std::memory_order_relaxed)) Wait(Get().PendingOn(vaddr, size));
}

} // namespace AsyncUpload

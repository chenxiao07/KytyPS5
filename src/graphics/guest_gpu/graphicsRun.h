#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <optional>
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>

namespace Libs::Graphics {

class RenderContext;
class Speculation;

class GuestGpu final {
public:
	explicit GuestGpu(RenderContext& renderer);
	~GuestGpu();
	KYTY_CLASS_NO_COPY(GuestGpu);

	void               Shutdown();
	[[nodiscard]] bool IsStopping();
	void               SendCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);
	// SendCommand, or false once the GPU thread takes no more (shutting down): for callers on guest threads that may still
	// run then.
	[[nodiscard]] bool TrySendCommand(Common::UniqueFunction<void>&& command);

	// Submitted command memory is borrowed and must remain valid until GPU execution completes.
	void              Submit(std::span<const uint32_t> draw_commands,
	                         std::span<const uint32_t> constant_commands);
	void              SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void              SubmitFlipPreparation(uint64_t request_id);
	void              Done();
	[[nodiscard]] int GetFrameNum() const;

	[[nodiscard]] static bool IsGpuThread() noexcept;
	// (upload_at_copy: a queued command that takes the render lock as the draw path does.)
	[[nodiscard]] RenderContext& Renderer() noexcept { return m_renderer; }
	// KYTY_SERVE_IN_WAITS (live serve_in_waits, default 1): where the GPU thread runs the guest commands queued for it (a
	// game thread's readback fault among them) besides before each packet: 1 while it waits for the upload worker (only
	// from the drains told to serve, never inside a command), 2 after each segment a speculation commits, 4 while it
	// waits for a speculation's chunk and between chunks.
	[[nodiscard]] static uint32_t ServeInWaits() noexcept;
	// GPU thread: runs the guest commands queued for it, if any (ProcessCommands); not from inside one of them.
	static void               ServePendingCommands();
	[[nodiscard]] static bool CommandsPending() noexcept;
	// The GPU thread, or the speculation's thread while the GPU thread waits for its translation (KYTY_SPECULATE=2):
	// what reads the GPU thread's caches to translate (not what may fault: only the GPU thread reads back).
	[[nodiscard]] static bool IsTranslatingThread() noexcept;
	static void               SetSpeculationThread(bool speculation) noexcept;

	// Offline tools (xpr_replay_tests) run the renderer on their own thread.
	static void SetOfflineGpuThread(bool gpu_thread) noexcept;

	// Tripwire (spec_thread_gpu_faults): the fault this thread's exception handler serves (KytyExceptionHandler around
	// HandleGpuFault; null outside one), returning the one it replaces. A SendCommandSync made under it from a host thread
	// other than the GPU thread (a speculation's; not a guest thread) is a fault that waits for the GPU thread: counted
	// and logged, then served as before. The invariant says it never happens: it must read 0. It is the only
	// unconditional rule: the GPU thread serves queued commands in some of its waits only: a flip's (WAIT_FLIP_DONE,
	// FlipQueue::Wait: live serve_in_flip_waits, default 1, 2face264), the upload worker's from the drains told to serve
	// (KYTY_SERVE_IN_WAITS & 1), after a committed segment (& 2) and a speculation's chunk's, outside a command (& 4)
	// (serve_in_waits: 7, all three, under KYTY_SPECULATE=4; 1 otherwise); not in its other waits (joining the workers
	// in Speculation::Stop / ~Speculation, any plain lock: PipelineCache m_mutex, CatalogWrite, the texture cache m_lock,
	// the descriptor pools' alloc_mutex).
	struct FaultSite {
		uint64_t vaddr = 0, rip = 0, rsp = 0;
	};
	static const FaultSite* NoteFault(const FaultSite* site) noexcept;

private:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;
	static constexpr uint32_t ComputeQueue1        = 1;
	static constexpr uint32_t ComputeQueue49       = 49;

	enum class SubmissionType { Graphics, Compute, FlipPreparation, FrameBoundary };

	struct Submission {
		SubmissionType            type     = SubmissionType::Graphics;
		uint32_t                  queue_id = 0;
		std::span<const uint32_t> commands;
		std::span<const uint32_t> constant_commands;
		Pm4Execution              command_execution;
		Pm4Execution              constant_execution;
		bool                      reset_processor   = false;
		bool                      started           = false;
		bool                      command_complete  = false;
		bool                      constant_complete = false;
		bool                      blocked           = false;
		// KYTY_SPECULATE=4: requeued by a yield (Speculation::Outcome::Yield), not blocked: passed over once while
		// another front is runnable (GuestGpu::m_yields counts the queued ones).
		bool                      yielded           = false;
		uint64_t                  flip_request_id   = 0;
		uint64_t                  frame_epoch       = 0;
		uint64_t                  scout_sequence    = 0; // queue 0's graphics submissions, numbered (scout.h)
		uint64_t                  serial            = 0; // its place in its queue (from 1)
		// KYTY_SPECULATE=4: the chunk committed next; whether the one it is in is translated the normal way, up to where
		// the next begins (none: to the end).
		uint32_t                    spec_chunk  = 0;
		bool                        spec_normal = false;
		std::optional<Pm4Execution> spec_end;
	};

	void              Enqueue(Submission submission);
	static void       PushGraphics(Speculation& speculation, const Submission& submission);
	static void       PushCompute(Speculation& speculation, const Submission& submission);
	[[nodiscard]] Speculation* ComputeLane(uint32_t queue_id) const noexcept;
	void              WakeForComputeReclaim();
	void              ReclaimCompute(bool idle = false);
	[[nodiscard]] bool ReclaimComputeBeforeIdle();
	void              WaitForIdle();
	bool              CanProcessSubmission(const Submission& submission) const;

public:
	// On the GPU thread, with no lane lock held. A runnable front elsewhere: 1 with a translated chunk, 2 in order.
	// A waitless compute lane can also yield to queue 0.
	[[nodiscard]] static int ReadyFrontElsewhere(uint32_t except = 0) noexcept;

private:
	// KYTY_SPECULATE=4, on the GPU thread holding m_queue_mutex (released for the reads, taken again): the blocked fronts
	// whose stored WAIT_REG_MEM (m_failed_waits) passes now unblocked; how many. `at` (front_unblock_retest,
	// graphicsRun.cpp): Selector, as always (a word not read: kept blocked); at a completion or the selector's idle
	// timeout, in place of the blanket unblock, a word not read unblocks its front as the blanket did, and a front whose
	// word was read and fails is kept blocked.
	enum class RetestAt { Selector, Completion, IdleTimeout };
	uint32_t          RetestBlockedFronts(RetestAt at = RetestAt::Selector);
	// (front_unblock_retest, holding m_queue_mutex: at a completion / the idle timeout) every blocked front unblocked as
	// before except those holding a stored wait, which RetestBlockedFronts(at) re-reads until the mode-4 bound.
	void              UnblockFrontsRetested(RetestAt at, bool bound_retest);
	void              ProcessCommands();
	bool              Process(Submission& submission);
	static void       ThreadRun(void* data);
	CommandProcessor& GetProcessor(uint32_t queue_id);

	RenderContext&                                 m_renderer;
	const std::atomic<int64_t>&                     m_commit_walks_switch;
	Common::Mutex                                  m_submission_mutex;
	Common::Mutex                                  m_queue_mutex;
	std::mutex                                     m_shutdown_mutex;
	Common::CondVar                                m_work_available;
	Common::CondVar                                m_idle;
	std::array<std::deque<Submission>, QueueCount> m_queues;
	std::deque<Common::UniqueFunction<void>>       m_commands;
	std::atomic_uint32_t                           m_pending_commands {0};
	uint32_t                                       m_next_queue        = 0;
	std::array<uint64_t, QueueCount>               m_serials {};
	std::atomic<uint64_t>                           m_copy_enqueue_serial {0};
	uint32_t                                       m_submission_count  = 0;
	// Producer epoch is owned by m_submission_mutex; consumed epoch by m_queue_mutex.
	uint64_t                                       m_submitted_frame   = 0;
	uint64_t                                       m_consumed_frame    = 0;
	bool                                           m_processing        = false;
	bool                                           m_graphics_done     = true;
	bool                                           m_accepting         = true;
	bool                                           m_stopping          = false;
	bool                                           m_shutdown_complete = false;
	// Under m_queue_mutex: the last graphics submission's number, and a reset a flip preparation carried (the next one's).
	uint64_t                                       m_scout_sequence    = 0;
	bool                                           m_scout_reset       = false;
	// Under m_queue_mutex, KYTY_SPECULATE=4 only (0 / empty otherwise): the queued submissions a yield marked; and for a
	// compute front requeued blocked at a WAIT_REG_MEM, that wait with the front's serial (size 0: none), and how many.
	// (front_unblock_retest on, any mode: the fronts it covers, graphicsRun.cpp.)
	struct FailedWait {
		uint64_t serial = 0, address = 0, reference = 0, mask = 0;
		uint32_t function = 0, size = 0;
		uint32_t suppressed_unblocks = 0;
	};
	uint32_t                                       m_yields            = 0;
	std::array<FailedWait, QueueCount>             m_failed_waits {};
	uint32_t                                       m_failed_waits_n    = 0;

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	// KYTY_SPECULATE (speculation.h): made at the first graphics command buffer it translates.
	std::unique_ptr<Speculation>                                     m_speculation;
	std::atomic<Speculation*>                                        m_ahead {nullptr}; // KYTY_SPECULATE=4's
	std::array<bool, QueueCount>                                     m_spec_start_failed {}; // GPU thread only
	// KYTY_SPEC_COMPUTE=1 retains round 4's single configurable async queue.
	std::unique_ptr<Speculation>                                     m_speculation_async;
	std::atomic<Speculation*>                                        m_ahead_async {nullptr};
	uint32_t                                                         m_async_queue = 0;
	// Each fixed compute lane has its own results, state pass and workers.
	std::unique_ptr<Speculation>                                     m_speculation_q49;
	std::unique_ptr<Speculation>                                     m_speculation_q1;
	std::atomic<Speculation*>                                        m_ahead_q49 {nullptr};
	std::atomic<Speculation*>                                        m_ahead_q1 {nullptr};
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;

	uint64_t        m_submit_id = 0;
	std::atomic_int m_done_num  = 0;
	std::jthread    m_thread;

	friend class CommandProcessor;
	friend class Speculation;
};
} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */

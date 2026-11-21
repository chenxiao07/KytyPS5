#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H

#include "common/assert.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "weave.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics {

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func);
// The first entry of an indirect register table (`pairs` register-value pairs; `bank`: 0 context, 1 SH, 2 uconfig) that
// names no register of its bank (not the sentinel, a context table's fake registers or the bank's NOP), or `pairs` when
// every entry does. Read only: what GuardTable checks off the GPU thread before it applies a table (pm4Handlers.cpp).
[[nodiscard]] uint32_t RegisterTableFirstPastBank(const uint32_t* words, uint32_t pairs, uint32_t bank);

// Off the GPU thread (ProcessPm4Spec: a speculation's or a state pass's thread), a SET_*_REG_INDIRECT packet's register
// table read once, in a guarded copy before the packet is entered: the observer (DrawStateObserver::ObservePacket) reads
// this copy, never the table, and so does a state pass's handler (GuardTable; a speculation's copies after its check and
// compares: `shared` false). Keyed by the packet and the table it names (`packet` null: none; GuardTable takes it).
struct IndirectTableCopy {
	const uint32_t*       packet = nullptr;
	uint64_t              table  = 0;
	uint32_t              pairs  = 0;
	bool                  shared = false;
	std::vector<uint32_t> words;
};
inline thread_local IndirectTableCopy t_indirect_table_copy;

enum class Pm4ProcessResult { Complete, Blocked };

enum class ContextStateOperation : uint32_t {
	Clear     = 0,
	Push      = 1,
	Pop       = 2,
	PushClear = 3,
};

class Pm4Execution {
public:
	[[nodiscard]] bool MadeProgress() const noexcept { return m_made_progress; }
	// A position within a command buffer (not an empty execution that Process begins at its commands).
	[[nodiscard]] bool Started() const noexcept { return !m_buffer_stack.empty(); }
	// Each buffer's commands through `rebase`, which gives another view of the same bytes back (KYTY_WRITE_GUARD). The
	// guest view's address is kept: a position is the buffer's guest address and offset, whichever view reads it
	// (SamePosition; 4 Nov wg6: compared by the view's pointer, a rebased in-order run missed its chunk's end).
	template <typename F>
	void RebaseCommands(F&& rebase) {
		for (auto& cursor: m_buffer_stack) {
			if (cursor.guest == nullptr) cursor.guest = cursor.commands.data();
			cursor.commands = rebase(cursor.commands);
		}
	}

private:
	friend class CommandProcessor;

	struct BufferCursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset_dw           = 0;
		uint32_t                  deferred_advance_dw = 0;
		const uint32_t*           guest               = nullptr; // the guest view's commands once rebased, else null
		[[nodiscard]] const uint32_t* Position() const noexcept { return guest != nullptr ? guest : commands.data(); }
	};

	std::vector<BufferCursor> m_buffer_stack;
	bool                      m_suspended     = false;
	bool                      m_made_progress = false;
};

class CommandProcessor {
public:
	struct FlipInfo {
		int     handle    = 0;
		int     index     = 0;
		int     flip_mode = 0;
		int64_t flip_arg  = 0;
	};

	CommandProcessor(RenderContext& renderer, int interrupt_event_id)
	    : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id) {}
	~CommandProcessor() = default;

	KYTY_CLASS_NO_COPY(CommandProcessor);

	void Reset();
	void ApplyContextStateOperation(ContextStateOperation operation);
	// The registers and the rest of the processor's state (with the constant RAM: `const_ram`), not the queue it
	// serves: a speculative translation starts from the processor's state and, committed, leaves its own
	// (speculation.cpp).
	void CopyStateFrom(const CommandProcessor& other, bool const_ram);
	// The state draws and dispatches may change besides the registers they read is the same.
	[[nodiscard]] bool SameDrawState(const CommandProcessor& other) const;
	// The same registers and draw state (all CopyStateFrom copies but the flip, submit id and constant engine).
	[[nodiscard]] bool SameState(const CommandProcessor& other) const;
	// A state pass (KYTY_SPECULATE=4): Process runs only the packets that change this processor's state (registers,
	// index and indirect bases, instances, counters, predication, context pushes), translating and executing nothing:
	// the state a command buffer leaves, before it is translated.
	void SetStateOnly(bool state_only) noexcept { m_state_only = state_only; }
	// Chunks of a command buffer (KYTY_SPECULATE=4): a state pass suspends before every `every`-th draw or dispatch that
	// does not continue a run of indirect draws (TakeBoundary: it did); Process suspends before the packet at `at` (the
	// next chunk's start; null: none; TakeStopReached: it did). Positions are compared by the packets they are at.
	void SetChunking(uint32_t every) noexcept {
		m_chunk_every       = every;
		m_chunk_work        = 0;
		m_chunk_last_opcode = 0;
	}
	[[nodiscard]] bool        TakeBoundary() noexcept { return std::exchange(m_at_boundary, false); }
	void                      SetStopAt(const Pm4Execution* at) noexcept { m_stop_at = at; }
	[[nodiscard]] bool        TakeStopReached() noexcept { return std::exchange(m_stop_reached, false); }
	// The execution suspended before a packet on a page a read watcher guards (ProcessPm4Spec, off the GPU thread).
	[[nodiscard]] bool        TakeWatchedStop() noexcept { return std::exchange(m_watched_stop, false); }
	[[nodiscard]] static bool SamePosition(const Pm4Execution& a, const Pm4Execution& b);
	// (DIAG, SegmentAsHole's exit: the packet at a position, whether the packet 5 dwords before it is an indexed indirect
	// draw of the same words 2-4 (a run of indirect draws passed it) and how far past a 4 KiB boundary it is.)
	static void PrintPacketAt(const Pm4Execution& at, const char* what);
	// The packet a suspended execution stopped at (its length in `dwords`), and past it.
	[[nodiscard]] static const uint32_t* SuspendedPacket(const Pm4Execution& execution, uint32_t& dwords);
	// SuspendedPacket would size it (the packet is in its buffer: not past a rewritten buffer's end).
	[[nodiscard]] static bool            SuspendedPacketFits(const Pm4Execution& execution);
	// The execution stopped at a WAIT_REG_MEM.
	[[nodiscard]] static bool AtWait(const Pm4Execution& execution);
	static void                          SkipSuspendedPacket(Pm4Execution& execution);

	void            BufferInit();
	void            BufferFlush();
	// A flush nothing waits on (a label without an interrupt, a slice's end): deferred while the GPU is busy.
	void            BufferFlushDeferrable();
	void            BufferFlushAndWait();
	void            BufferWait();
	HW::Context&    GetCtx() { return m_ctx; }
	HW::UserConfig& GetUcfg() { return m_ucfg; }
	HW::Shader&     GetShCtx() { return m_sh_ctx; }

	void SetIndexType(uint32_t index_type_and_size);
	void SetIndexBaseAddress(uint64_t index_base_addr);
	void SetIndexBufferSize(uint32_t index_buffer_size);
	void SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr);
	void SetDispatchIndirectArgsBaseAddress(uint64_t dispatch_indirect_args_base_addr);
	void DispatchIndirectAddress(uint64_t args_address, uint32_t mode);
	void SetNumInstances(uint32_t num_instances);
	bool DrawIndex(DrawIndexArgs args);
	// An indirect draw whose arguments the GPU wrote, drawn from them on the GPU.
	[[nodiscard]] bool TryGpuIndirectDraw(uint64_t args, uint32_t count, uint32_t stride, bool indexed);
	void DrawIndexOffset(uint32_t index_offset, uint32_t index_count);
	bool DrawIndexAuto(DrawAutoArgs args);
	// Native XPR draws (kyty_local_native_xpr_mode): packets consumed, 0 = normal path.
	[[nodiscard]] uint32_t TryNativeXprDraws(std::span<const uint32_t> packets, bool clean);
	// A DRAW_INDEX_2 / DRAW_INDEX_OFFSET_2 from a native record: the packet's dwords, or 0.
	[[nodiscard]] uint32_t TryNativeDirectDraw(std::span<const uint32_t> packet);
	// A DRAW_INDEX_AUTO of a triangle list or strip as a table draw: the packet's dwords, or 0.
	[[nodiscard]] uint32_t TryNativeAutoDraw(std::span<const uint32_t> packet);
	// KYTY_SPEC_DRAW_INDIRECT (live spec_draw_indirect): a DRAW_INDIRECT (non-indexed) as a table draw, its arguments
	// read on the GPU: the packet's dwords, or 0.
	[[nodiscard]] uint32_t TryNativeIndirectAutoDraw(std::span<const uint32_t> packet);
	// A DRAW_INDEX_INDIRECT_MULTI of a count it holds, as that many DRAW_INDEX_INDIRECT: the packet's dwords, or 0.
	[[nodiscard]] uint32_t TryNativeMultiDraw(std::span<const uint32_t> packet);
	uint32_t TryDrawIndirectRun(std::span<const uint32_t> packets);
	void DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed);
	void DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
	                       const volatile uint32_t* count_addr, uint32_t stride_in_bytes,
	                       uint32_t draw_initiator, bool indexed);
	void WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint32_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint64_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void Flip();
	void Flip(void* dst_gpu_addr, uint32_t value);
	void FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action, void* dst_gpu_addr,
	                       uint32_t value);
	void PrepareCpuFlip(uint64_t request_id);
	void SynchronizeGpu();
	void EmitGlobalBarrier();
	void BreakComputeChain();
	void TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id);
	void DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y, uint32_t thread_group_z,
	                    uint32_t mode, uint64_t indirect_args = 0);
	void DispatchIndirect(uint32_t data_offset, uint32_t mode);
	void WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index);
	// An R_SUPER_RESOLUTION packet (graphics/guest_gpu/superResolution.h): the upscaler in place of Sony's.
	void SuperResolution(const uint32_t* packet, uint32_t dwords);
	void TriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address = 0);

	void SetUserDataMarker(HW::UserSgprType type) { m_user_data_marker = type; }
	[[nodiscard]] HW::UserSgprType GetUserDataMarker() const { return m_user_data_marker; }

	void ResetDeCe();
	void SetCeComplete(bool complete) { m_ce_complete = complete; }
	void WaitCe();
	void WaitDeDiff(uint32_t diff);
	void WaitForRewind(bool valid);
	void IncrementDe();
	void IncrementCe();

	void WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num);
	void DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num);

	template <typename T>
	void WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll, uint32_t wait_op);
	void WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num, uint32_t write_control);
	void WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes);
	void ReportLodStats(void* dst, uint32_t size, bool reset) { GetGpuResources().GetBufferCache().ReportLodStats(dst, size, reset); }
	void DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
	             uint64_t dst_address_or_offset, uint8_t src_sel, uint8_t src_cache_policy,
	             uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
	             uint8_t wait_for_previous, uint8_t write_confirm, uint8_t block_engine);
	void SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
	                    const volatile void* address, uint32_t count_in_dwords);
	[[nodiscard]] bool ShouldSkipPredicatedPackets() const { return m_predicate_skip; }

	Pm4ProcessResult Process(Pm4Execution& execution, std::span<const uint32_t> commands);
	void             ProcessIndirectBuffer(std::span<const uint32_t> commands);

	void SetFlip(const FlipInfo& flip) { m_flip = flip; }

	[[nodiscard]] uint64_t GetSubmitId() const { return m_submit_id; }
	void                   SetSubmitId(uint64_t submit_id) { m_submit_id = submit_id; }
	[[nodiscard]] bool     IsAsyncComputeQueue() const { return m_interrupt_event_id >= 0x20; }

	// The weave (weave.h): GuestGpu::Enqueue's feed of a queue-0 graphics submission to the reader (the queue mutex
	// held, as the scout is fed), and the GPU thread's at such a submission's first slice, on queue 0's processor
	// (whose draw packets are then woven). WeaveModeChanged: the GPU thread between submissions, on queue 0's
	// processor, once a live change has set KYTY_SPECULATE to 4: the weave paused there (a no-op unless it runs).
	static void WeaveFeed(uint64_t sequence, std::span<const uint32_t> commands, bool constant, bool reset);
	void        WeaveSubmission(uint64_t sequence);
	void        WeaveModeChanged();

private:
	uint32_t            m_draw_run_skip = 0;
	bool                m_state_only    = false;
	uint32_t            m_chunk_every = 0, m_chunk_work = 0, m_chunk_last_opcode = 0;
	bool                m_at_boundary = false, m_stop_reached = false, m_watched_stop = false;
	const Pm4Execution* m_stop_at = nullptr;
	template <typename T>
	void WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest, uint32_t eop_event_type,
	                      uint32_t cache_action, uint32_t event_index, uint32_t event_write_source,
	                      void* dst_gpu_addr, T value, uint32_t interrupt_selector,
	                      uint32_t interrupt_context_id);
	void ProcessPm4(Pm4Execution& execution, size_t stop_depth);
	void ProcessPm4Spec(Pm4Execution& execution, size_t stop_depth);
	// ProcessPm4's loop, one body: <false> as it is, <true> with the weave's hooks at queue 0's draw packets (taken
	// only while the weave runs on this processor; a build without KYTY_LOCAL_VULKAN_RECORDING instantiates only
	// <false>).
	template <bool kWeave>
	void ProcessPm4Loop(Pm4Execution& execution, size_t stop_depth);
	// The weave's register image of this processor (Weave::Registers): taken, applied, compared whole.
	void               WeaveSnapshot(Weave::Registers& out) const;
	void               WeaveRestore(const Weave::Registers& in);
	[[nodiscard]] bool WeaveSame(const Weave::Registers& in) const;
	// A table draw's arguments as the native entries decode them from their packets and the index and indirect
	// registers, read only: TryNativeXprDraws (a run of DRAW_INDEX_INDIRECT that differ only in the argument offset),
	// TryNativeMultiDraw (a DRAW_INDEX_INDIRECT_MULTI of a count the packet holds) and TryNativeDirectDraw
	// (DRAW_INDEX_2, DRAW_INDEX_OFFSET_2) decode through them, and so does a weave worker's walk. False: the entry
	// leaves the packet to its handler. What reads the GPU thread's own state (a frame capture, the interface's scaled
	// draws) stays in the entries.
	struct NativeDrawArgs {
		static constexpr uint32_t           MaxDraws = 64;
		std::array<uint64_t, MaxDraws>      commands; // the indirect forms' argument addresses (the first `count`)
		uint32_t                            count      = 0;
		uint64_t                            index_base = 0, index_bytes = 0;
		vk::IndexType                       index_type = vk::IndexType::eUint16;
		bool                                is_direct  = false; // `direct` holds the draw (no `commands`)
		RenderExecutor::NativeXprDirectDraw direct {};
		uint32_t                            dwords = 0; // the packets' dwords the draw takes
	};
	[[nodiscard]] bool NativeXprDrawsArgs(std::span<const uint32_t> packets, NativeDrawArgs& args) const;
	[[nodiscard]] bool NativeMultiDrawArgs(std::span<const uint32_t> packet, NativeDrawArgs& args) const;
	[[nodiscard]] bool NativeDirectDrawArgs(std::span<const uint32_t> packet, NativeDrawArgs& args) const;
	// Weave: the reader and the GPU thread's order
	// The reader (weave-reader.inc): its loop (Weave::Bodies::read) and its read of a submission's copy from dword
	// `from` (true: to its end, in step with the stream there; false: stopped at `stopped_at`). A worker's walk of a
	// parcel (Weave::Bodies::make, on worker `worker`'s processor) and its draw entry (the dwords the draw took; 0, a
	// hole: `skip` the dwords the GPU thread's entry takes); a parcel's release (Weave::Bodies::release). The register
	// log's image: taken whole, its blocks changed since `at_last` appended to the parcel's log, the log's blocks
	// [begin, end) applied.
	static void            WeaveRead();
	[[nodiscard]] bool     WeaveReadFrom(uint64_t sequence, const uint32_t* guest,
	                                     const std::shared_ptr<const std::vector<uint32_t>>& copy, uint32_t from,
	                                     uint32_t& stopped_at);
	static void            WeaveMake(uint32_t worker, Weave::Parcel& parcel);
	void                   WeaveWalk(uint32_t worker, Weave::Parcel& parcel);
	[[nodiscard]] uint32_t WeaveDrawAt(std::span<const uint32_t> packets, const uint32_t* identity, uint32_t& skip);
	static void            WeaveRelease(Weave::Parcel& parcel);
	void                   WeaveLogTake(uint8_t* image) const;
	void                   WeaveLogDelta(uint8_t* at_last, Weave::Parcel& parcel) const;
	void                   WeaveLogApply(const Weave::Parcel& parcel, uint32_t begin, uint32_t end);
	// The GPU thread's order (weave-order.inc): the weave started (once; false: stood down); at a top-level draw packet
	// (true: passed, a part of a parcel spliced there); the parcel starting at the cursor's packet (its serial, -1
	// none); a draw packet of the open parcel; a spliced part's made draws [begin, end) skipped to `until` from the
	// register log, or one draw packet passed; the open parcel closed (`distrust`: the reader goes out of step); the
	// weave paused; a handler's draw in the open parcel watched; the narrow walk test for the executor's turn.
	[[nodiscard]] bool    WeaveStart();
	[[nodiscard]] bool    WeaveAt(Pm4Execution& execution);
	[[nodiscard]] int64_t WeaveFind(const Pm4Execution::BufferCursor& cursor);
	[[nodiscard]] bool    WeaveInside(Pm4Execution& execution, const uint32_t* packet);
	void                  WeaveSkip(Pm4Execution& execution, const Weave::Parcel& parcel, uint32_t begin, uint32_t end,
	                                const uint32_t* until);
	void                  WeavePassDraw(Pm4Execution& execution);
	void                  WeaveClose(bool distrust);
	void                  WeavePause();
	void                  WeaveYieldBegin(const Pm4Execution& execution);
	void                  WeaveYieldEnd();
	static bool WeaveSameOf(const void* processor, const Weave::Registers& start, const Weave::RegisterLog& log,
	                        uint32_t first, uint32_t index);
	// Weave: end of the reader and the GPU thread's order
	void SuspendPm4();
	CommandScheduler&   GetScheduler() const { return m_renderer.GetCommandScheduler(); }
	CommandBuffer&      CurrentBuffer() { return GetScheduler().Current(); }
	void                CheckBuffer() const { GetScheduler().CheckActive(); }
	GpuResourceManager& GetGpuResources() const { return m_renderer.GetGpuResources(); }

	RenderContext&   m_renderer;
	HW::Context      m_ctx;
	HW::Context      m_saved_ctx;
	bool             m_context_state_pushed = false;
	HW::UserConfig   m_ucfg;
	HW::Shader       m_sh_ctx;
	HW::UserSgprType m_user_data_marker                 = HW::UserSgprType::Unknown;
	uint32_t         m_index_type_and_size              = 0;
	uint32_t         m_index_buffer_size                = 0;
	uint64_t         m_index_base_addr                  = 0;
	uint64_t         m_draw_indirect_args_base_addr     = 0;
	uint64_t         m_dispatch_indirect_args_base_addr = 0;
	// Persistent draw state: indirect draws update it for subsequent draws.
	uint32_t m_num_instances = 1;

	uint32_t m_de_count    = 0;
	uint32_t m_ce_count    = 0;
	bool     m_ce_complete = false;

	uint32_t m_const_ram[0x3000] = {0};
	// The dwords [0, m_const_ram_used) may be nonzero: a reset clears only them (all 48 KiB at every DispatchReset).
	uint32_t m_const_ram_used = 0;

	FlipInfo  m_flip;
	const int m_interrupt_event_id;
	uint64_t  m_submit_id                   = 0;
	uint64_t  m_synthetic_occlusion_counter = 0;
	bool      m_predicate_skip              = false;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H

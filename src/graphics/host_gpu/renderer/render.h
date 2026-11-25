#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/guest_gpu/superResolution.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include "live-switch.h"
#include "weave.h"

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <atomic>
#include <shared_mutex>
#include <unordered_set>
#include <vector>

extern "C" {
extern volatile std::atomic_uint32_t kyty_local_speculate_mode; // KYTY_SPECULATE (graphics/guest_gpu/speculation.cpp)
}

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW
namespace Spec {
struct State;
} // namespace Spec

struct GraphicContext;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawRenderState;
class RenderContext;
class CommandScheduler;
class TextureCache;
struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	EopWrite,
	EopInterrupt,
	EopWriteBack,
	EopFlip,
	EopWriteBackFlip,
	EopOnlyFlip,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// Arguments the GPU reads (gpu_args_count DrawIndexedIndirectArgs at gpu_args, gpu_args_stride
	// apart), not read on the CPU: index_count and index_addr then cover the whole index buffer.
	uint64_t gpu_args        = 0;
	uint32_t gpu_args_count  = 0;
	uint32_t gpu_args_stride = 0;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// Arguments the GPU reads (DrawIndirectArgs), as in DrawIndexArgs.
	uint64_t gpu_args        = 0;
	uint32_t gpu_args_count  = 0;
	uint32_t gpu_args_stride = 0;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	// (Mode 4, a speculation's thread, around one BeginRendering: RenderingHookScope) The begin's call recorded by the hook
	// instead when it returns true (a rendering naming a view its commit makes: table-xpr.inc's TableVirtualRendering).
	using RenderingHook = bool (*)(void* data, vk::CommandBuffer command, const vk::RenderingInfo& rendering);
	static inline thread_local RenderingHook t_rendering_hook      = nullptr;
	static inline thread_local void*         t_rendering_hook_data = nullptr;
	struct RenderingHookScope {
		RenderingHookScope(RenderingHook hook, void* data) noexcept { t_rendering_hook = hook, t_rendering_hook_data = data; }
		~RenderingHookScope() { t_rendering_hook = nullptr, t_rendering_hook_data = nullptr; }
		RenderingHookScope(const RenderingHookScope&)            = delete;
		RenderingHookScope& operator=(const RenderingHookScope&) = delete;
	};
	void EndRendering() const;
	// The render pass `state` again after an interruption (a copy of an attachment): its attachments loaded, not
	// cleared, and `state` kept as the current one, so that the next BeginRendering with it continues the pass.
	void ResumeRendering(const RenderState& state) const;
	// The render pass in progress is `state`'s, as BeginRendering compares them.
	[[nodiscard]] bool RenderingWith(const RenderState& state) const;
	// A render pass is in progress.
	[[nodiscard]] bool Rendering() const noexcept { return m_rendering; }
	// The render pass in progress (none: nullptr), as BeginRendering began it (spec_depth_layers2's commit).
	[[nodiscard]] const RenderState* OpenRendering() const noexcept { return m_rendering ? &m_render_state : nullptr; }
	// Raw internal graphics operations must invalidate the values recorded by
	// the ordinary draw path. Compute and transfer commands do not alter them.
	void InvalidateGraphicsState() const noexcept;
	// Advances with every InvalidateGraphicsState: a path that binds graphics
	// state itself can tell whether anything else bound state since.
	[[nodiscard]] uint64_t GraphicsGeneration() const noexcept { return m_graphics_generation; }

	[[nodiscard]] vk::CommandBuffer Handle() const;
	// Only the dispatch path can bypass a pending dependency, after preparing
	// resources. Every other host command drains it through Handle().
	[[nodiscard]] vk::CommandBuffer ChainHandle() const;
	[[nodiscard]] bool ComputeChainPending() const noexcept { return m_compute_access_pending; }
	void ContinueComputeChain() const;
	[[nodiscard]] vk::CommandBuffer HandleForFullBarrier() const;
	// Guest buffer copies in a row (Buffer::CopyInRun): one barrier before the run and one after it, recorded before
	// the next other command (Handle, ChainHandle; a full barrier covers it). A copy that reads what an earlier copy of
	// the run writes, or writes what one reads or writes, starts a new run.
	[[nodiscard]] vk::CommandBuffer CopyRunHandle(vk::Buffer source, uint64_t source_offset, vk::Buffer destination,
	                                              uint64_t destination_offset, uint64_t size) const;
	// A run is open: everything recorded since its barrier is its copies (any other command flushes it first).
	[[nodiscard]] bool InCopyRun() const noexcept { return !m_copy_run.empty(); }
	// A copy of what the open run wrote (a copy-feedback snapshot, BufferCache::ScheduleCopyFeedback): recorded when the
	// run ends, after its copies, behind one transfer barrier, then made available to the host. A copy of the run that
	// writes its source range starts a new run, which records it first.
	void CopyAfterRun(vk::Buffer source, uint64_t source_offset, vk::Buffer destination, uint64_t destination_offset,
	                  uint64_t size) const;
	// Local diagnostic (GPU marks): the handle without draining a pending dependency.
	[[nodiscard]] vk::CommandBuffer RawHandle() const noexcept { return m_buffer; }
	// The recorded work count after this buffer's last global barrier (CommandProcessor::EmitGlobalBarrier).
	[[nodiscard]] uint64_t& BarrierWork() const noexcept { return m_barrier_work; }

	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept { return *m_shaders; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	// A weave worker's view of `real` (context.cpp): its graphics generation and handle, with the worker's own
	// registers. Only the table path's worker instantiation is given it; it records through the apart marker (Aim), and
	// its calls go into the worker's apart buffer.
	CommandBuffer(const CommandBuffer& real, HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void Aim(VkCommandBuffer_T* marker) noexcept { m_buffer = vk::CommandBuffer(marker); }
	// A worker's view at a render pass change inside its strand (context.cpp): the pass ended and `next` begun in its
	// buffer, as EndRendering and BeginRendering make them.
	void PassChange(const RenderState& next);
	// The weave's turn: the pass the spliced part of a worker's strand left open is `state`'s (nothing recorded).
	void AdoptRendering(const RenderState& state) {
		m_render_state = state;
		m_rendering    = true;
	}
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	// (`placeholder`: a speculation thread's segment recorded as worker packets: the state reset, no Vulkan begin.)
	void Begin(bool placeholder = false);
	void End() const;
	void FlushCopyRun() const;
	void FlushRunTail() const;

	struct CopyRunRange {
		VkBuffer buffer;
		uint64_t begin, end;
		bool     written;
	};
	struct RunTailCopy {
		VkBuffer source, destination;
		uint64_t source_offset, destination_offset, size;
	};

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	mutable bool        m_compute_access_pending = false;
	mutable std::vector<CopyRunRange> m_copy_run; // a run's ranges (not empty: its after barrier is owed)
	mutable std::vector<RunTailCopy>  m_run_tail; // CopyAfterRun's copies, owed with the run's end
	mutable uint64_t    m_graphics_generation    = 0;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_timestamp_slot  = UINT32_MAX; // live trace GPU timestamps
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable uint64_t    m_barrier_work = UINT64_MAX;
	mutable bool        m_rendering   = false;
	mutable uint32_t    m_handle_requests = 0; // Handle and the like since Begin (none: nothing recorded)
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
	// The weave's: a worker's view and its pass changes; the turn reads m_rendering and m_render_state.
	friend class RenderExecutor;
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	// (The weave's executor state, when WeaveStart made it for this executor, goes with it: weave-turn.inc.)
	~RenderExecutor();
#endif
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode,
	                    uint64_t indirect_args = 0);
	// A state pass's direct dispatch (Spec::t_record): the copy the linear copy shader makes (journaled, as a speculation's
	// CopyBuffer journals it; one on GPU-written memory: its destination written) or the surface a compute clear clears
	// (predicted), as the catalog knows them.
	void PredictDispatch(const HW::Shader& sh_ctx, uint32_t thread_group_x, uint32_t thread_group_y,
	                     uint32_t thread_group_z, uint32_t mode, Spec::State& record);
	// An R_SUPER_RESOLUTION packet's upscale (graphics/guest_gpu/superResolution.h; renderSuperResolution.cpp).
	void RunSuperResolution(CommandBuffer& buffer, const SuperResolution::Packet& packet);

	[[nodiscard]] PreparedBindings PrepareBindings(const ShaderStageRuntime& runtime);
	void PrepareBindingsInto(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(PreparedBindings& bindings);
	void PrepareBdaBindings(const PreparedBindings& first, const PreparedBindings* second = nullptr);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	[[nodiscard]] bool             AcquireImages(PreparedBindings& bindings);
	[[nodiscard]] bool             AcquirableImage(const TextureBinding& binding);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings, bool compute_chain = false);

private:
	[[nodiscard]] bool ReadOnlyDrawBufferRangeSafe(GuestRange range, const DrawRenderState& state,
	                                               bool sampled_overlaps);
	struct TextureResolveCache;
	std::shared_ptr<TextureResolveCache> m_texture_resolve_cache;

	bool TryDrawIndexRun(uint64_t submit_id, CommandBuffer& buffer, std::span<const DrawIndexArgs> draws,
	                     std::span<const uint64_t> argument_addresses);
	// False only for GPU-read arguments this draw cannot use (a mesh draw): the caller reads
	// them on the CPU and draws again.
	bool DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args); // as DrawIndex

	struct GraphicsBindings {
		PreparedBindings                vertex;
		std::optional<PreparedBindings> pixel;
	};

	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);
	[[nodiscard]] TextureBinding ResolveTextureUncached(
	    const ShaderRecompiler::IR::ImageResource& resource,
	    const ShaderRecompiler::IR::DescriptorValue& value);
	[[nodiscard]] GraphicsBindings PrepareGraphicsBindings(const ShaderStageRuntime& vertex,
	                                                       const ShaderStageRuntime& pixel,
	                                                       bool                      pixel_active);
	void PrepareGraphicsBindingsInto(const ShaderStageRuntime& vertex,
	                                  const ShaderStageRuntime& pixel, bool pixel_active,
	                                  GraphicsBindings& bindings);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          bool log_setup_phases, DrawRenderState& state);
	bool ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable, bool log_pipeline_phase,
	                         bool set_bind_debug, bool set_auto_debug);
	void CommitGraphicsState(CommandBuffer& buffer, const ShaderVertexInputInfo& input,
	                         const RenderColorInfo* colors, uint32_t color_count,
	                         const RenderDepthInfo& depth, vk::Pipeline pipeline,
	                         vk::ImageAspectFlags feedback_aspects);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               const std::optional<PreparedBindings>& pixel = std::nullopt);
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      TrackImageBinding(ImageId id);
	void                      ResetBindings();
	// (`cleared`: the metadata address it cleared.)
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer& buffer, uint32_t group_x,
	                                                     uint32_t group_y, uint32_t group_z,
	                                                     uint32_t mode, uint64_t& cleared);
	[[nodiscard]] bool TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
	                                              CommandBuffer& command, uint32_t group_x,
	                                              uint32_t group_y, uint32_t group_z, uint32_t mode);

	// XPR draw capture for the offline replay tests (xpr-capture.h).
	void CaptureXprDraw(CommandBuffer& buffer, const DrawRenderState& state, const DrawIndexArgs& args);
	void CaptureXprTargets(const DrawRenderState& state, const GraphicsBindings& bindings);

	// Native XPR draws (src/local/native-xpr.inc).
	struct NativeXprDrawState;
	struct NativeXprRecord;
	struct NativeXprCache;
	struct NativeXprDrawArgs {
		uint64_t      index_address = 0, index_bytes = 0;
		vk::IndexType index_type    = vk::IndexType::eUint16;
		uint64_t      args_address  = 0; // GPU-side DrawIndexedIndirectCommand
		bool          keep_clears   = false;
	};
	std::shared_ptr<NativeXprCache> m_native_xpr;
	bool                            m_native_xpr_store = false; // next prepared draw stores
	struct NativeXprVerifyRequest {
		const NativeXprRecord*    record = nullptr;
		const NativeXprDrawState* state  = nullptr;
	} m_native_xpr_verify; // mode 2: the next prepared draw is compared with this record
	void NativeXprVerify(CommandBuffer& buffer, const DrawRenderState& state, const RenderState& rendering,
	                     std::span<PreparedBindings* const> stages, const DrawEmitInfo& emit);
	NativeXprCache& NativeXprState();
	void NativeXprRequestStore();
	void NativeXprKey(CommandBuffer& buffer, std::vector<uint32_t>& key) const;
	[[nodiscard]] uint64_t NativeXprStateKey(CommandBuffer& buffer) const;
	[[nodiscard]] vk::DescriptorSet NativeXprAllocateSet(vk::DescriptorSetLayout layout,
	                                                     vk::DescriptorPool& pool);
	void NativeXprRetire(std::unique_ptr<NativeXprRecord> record, bool keep_for_learning);
	void NativeXprRetireSet(vk::DescriptorPool pool, vk::DescriptorSet set);
	[[nodiscard]] const char* NativeXprBindResources(NativeXprRecord& record,
	                                                 std::span<PreparedBindings* const> stages,
	                                                 vk::DescriptorSetLayout layout, bool runtime);
	[[nodiscard]] bool NativeXprRebind(NativeXprRecord& record,
	                                   std::array<ShaderRecompiler::IR::ResourceSnapshot, 2>& fresh);
	[[nodiscard]] bool NativeXprTexelRangesClear(const NativeXprRecord& record);
	[[nodiscard]] bool NativeXprSwitchVariant(NativeXprRecord& record, uint64_t signature);
	void               NativeXprDropVariants(NativeXprRecord& record);
	[[nodiscard]] bool NativeXprRebindSlots(NativeXprRecord& record,
	                                        std::span<const std::pair<uint32_t, uint32_t>> images,
	                                        std::span<const std::pair<uint32_t, uint32_t>> buffers);
	void NativeXprCollect();
	bool NativeXprSweepStates(NativeXprCache& cache, uint64_t frame);
	// A draw state's dynamic state, only what differs from what the emissions recorded in the command buffer.
	static void NativeXprEmitDynamic(NativeXprCache& cache, vk::CommandBuffer vk_buffer, const NativeXprDrawState& draw_state,
	                                 bool same_command);
	void NativeXprLearn(const NativeXprRecord& stale, NativeXprRecord& fresh);
	void NativeXprStore(CommandBuffer& buffer, const DrawRenderState& state,
	                    vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                    const RenderState& rendering, std::span<PreparedBindings* const> stages,
	                    const DrawEmitInfo& emit);
	[[nodiscard]] bool NativeXprValidate(NativeXprRecord& record, uint64_t frame);
	[[nodiscard]] bool NativeXprReevaluate(NativeXprRecord& record);
	[[nodiscard]] bool NativeXprValidateResources(NativeXprRecord& record, uint64_t frame, bool mapping_moved);
	void NativeXprRetrace(NativeXprRecord& record);
	[[nodiscard]] bool NativeXprGather(NativeXprRecord& record);
	enum class NativeXprPatch { Done, Evaluate, Invalid };
	[[nodiscard]] NativeXprPatch NativeXprPatchDescriptors(NativeXprRecord& record);
	struct NativeXprDirect; // one direct draw (public NativeXprDirectDraw) with its record's offsets
	[[nodiscard]] bool NativeXprEmit(CommandBuffer& buffer, NativeXprRecord& record,
	                                 const NativeXprDrawState& draw_state,
	                                 const PipelineCache::Pipeline& pipeline,
	                                 std::span<const uint64_t> commands, uint64_t index_base,
	                                 uint64_t index_bytes, vk::IndexType index_type, bool keep_clears,
	                                 const NativeXprDirect* direct = nullptr);
	[[nodiscard]] bool NativeXprDraw(CommandBuffer& buffer, const NativeXprDrawArgs& args);

public:
	// (Mode 4; table-xpr.inc) What a worker made for a draw until its commit publishes it (public: the kept holders' list is
	// a file-level static there).
	struct TableEntryHolder;
	// Runtime entry points for the command processor (kyty_local_native_xpr_mode).
	// A direct indexed draw (DRAW_INDEX_2 / DRAW_INDEX_OFFSET_2): its indices, count and instances; or a direct draw
	// of vertices (DRAW_INDEX_AUTO, not `indexed`: the count is of vertices; table draws only). `indirect` (not
	// `indexed`; KYTY_SPEC_DRAW_INDIRECT): a draw of vertices whose arguments the GPU reads, one VkDrawIndirectCommand at
	// `commands[0]` (DRAW_INDIRECT; table draws only; the counts here unused).
	struct NativeXprDirectDraw {
		uint64_t index_address  = 0;
		uint32_t index_count    = 0;
		uint32_t instance_count = 1;
		bool     indexed        = true;
		bool     indirect       = false;
	};
	// KYTY_SPEC_DRAW_INDIRECT_VERIFY: the arguments the last `indirect` table draw read (TableDraw): its buffer (identity
	// only), offset, stride and count; `copy`: a host-visible ring's copy of the bytes, `copied` them.
	struct TableArgsEmit {
		const void*             buffer = nullptr;
		uint64_t                offset = 0;
		uint32_t                stride = 0, count = 0;
		bool                    copy   = false;
		std::array<uint32_t, 4> copied {};
	};
	TableArgsEmit m_table_args_emit;
	// `direct`: that draw instead of the DRAW_INDEX_INDIRECT run `commands` (an `indirect` one: with its arguments).
	[[nodiscard]] bool NativeXprTry(CommandBuffer& buffer, bool clean, std::span<const uint64_t> commands,
	                                uint64_t index_base, uint64_t index_bytes, vk::IndexType index_type,
	                                const NativeXprDirectDraw* direct = nullptr);
	// After a refused NativeXprTry: the normal path stores. Returns the log the
	// caller's ShaderReadObserver fills with the SRT evaluation's reads.
	[[nodiscard]] std::vector<std::pair<uint64_t, uint64_t>>* NativeXprReadLog();
	void NativeXprEndPacket();
	// Graphics registers may have changed without the observer seeing it.
	void NativeXprForgetState();
	// A clean draw (only user data and index state changed) after a table draw whose bindings still stand:
	// drawn the same way without the native path's lookups (KYTY_TABLE_XPR=2).
	[[nodiscard]] bool TableContinue(CommandBuffer& buffer, std::span<const uint64_t> commands, uint64_t index_base,
	                                 uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct);
	// Table dispatches (KYTY_TABLE_DISPATCH): a dispatch drawn the table way, and the normal path's dispatches.
	[[nodiscard]] bool TableDispatch(CommandBuffer& buffer, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
	                                 uint64_t indirect_args, bool thread_dimensions);
	// Why the last TableDispatch that declined did (the hole's reason when a speculation's dispatch has no other way).
	const char* m_dispatch_declined = nullptr;
	// (`special`: the special path that took the dispatch, as a speculation's hole reason; null: none.)
	void TableDispatchSeen(const HW::ComputeShaderInfo& cs, const char* special, bool thread_dimensions);
	// What a speculative translation retired with `pending` (descriptor sets; CommandScheduler::PendingTick and up)
	// gets `tick`.
	void ResolvePendingTicks(uint64_t pending, uint64_t tick);
	// A speculation's executor draws with what `owner` (the graphics queue's) learned: its table pairs and dispatches,
	// draw states, linear copy shader and compute clears (RenderExecutor::Catalog); `role` its own proofs' (1..Roles-1:
	// one per speculation thread). The owner sweeps its catalog only while no translation is going on and no result is
	// left unfinished (kyty_local_speculations_live).
	void ReadCatalogOf(RenderExecutor& owner, uint32_t role) noexcept {
		m_catalog = &owner;
		m_role    = role;
	}
	// At the start of a translation: what this executor kept of the catalog from its last one, dropped when the owner
	// changed it since (TableXpr::epoch; table-xpr.inc).
	void ForgetStaleCatalog();
	static constexpr uint32_t Roles = 13; // (the GPU thread, Spec::MaxPacketSlots speculation threads)
	// (Mode 4, the GPU thread) A segment's commit settles each proposal's catalog part under its owner's catalog write
	// taken with a try-lock (SettleOrQueue; `owner`: the proposal's catalog owner, Spec::State::ProposedSet::catalog_owner,
	// null: run at once, its own writes taken); held by a worker's read, or a part already waiting, the part waits in a list
	// of the GPU thread's, in order, run at the next commit's try that succeeds (SettleDrain(false)) and under a write that
	// waits (SettleDrain(true)) at each guest flip, before a sweep of the catalog (TableSweepPairs), before a live mode 0,
	// and where a settle must not wait past its commit. table-xpr.inc.
	static void SettleOrQueue(const void* owner, std::function<void()> work);
	static void SettleDrain(bool blocking);
	// (The renderer's teardown, the GPU thread stopped, the device still there: the parts still waiting dropped, so none
	// outlives the device in the static list.)
	static void SettleClear();
	static inline std::atomic<uint32_t> s_settle_pending_n {0}; // (the parts waiting)

private:
	// Table draws (src/local/table-xpr.inc).
	struct TableXpr;
	struct TablePair;
	void TableMemoRetired(); // KYTY_TABLE_MEMO: a set left service (memo_epoch moves; table-xpr.inc)
	struct TableFlatStage;
	struct TableVariant;
	void TablePack(TablePair& pair, const std::array<TableFlatStage, 2>& flat);
	struct TableImageSet;
	// Deferred: the pair's table variant is on its way (TableTry): the normal path draws, storing no native record.
	enum class TableResult { Drawn, Native, Store, Deferred };
	std::shared_ptr<TableXpr> m_table_xpr;
	// The executor whose catalog this one reads (a speculation's: the graphics queue's), null: its own. It changes none
	// of it but its own proofs in it (Role: TablePair::generation, TableImageSet::validated_frame...); what it holds of
	// it across draws is dropped when the owner changed it (TableXpr::epoch).
	RenderExecutor* m_catalog = nullptr;
	uint32_t        m_role    = 0;
	[[nodiscard]] RenderExecutor& Catalog() noexcept { return m_catalog != nullptr ? *m_catalog : *this; }
	// The catalog's owner changes its pairs and dispatches, image sets, draw states and compute clears under its lock
	// exclusively (CatalogWrite); another executor reads them under it shared (CatalogRead, on its own thread).
	mutable std::shared_mutex m_catalog_mutex;
	// (The GPU thread: the catalog whose write a settle part runs under, held for it by SettleOrQueue / SettleDrain; null:
	// none.)
	static inline thread_local const std::shared_mutex* t_settle_held = nullptr;
	// catalog_writer_first (under KYTY_SPECULATE=4, the only mode with readers): the owner raises this while it waits for
	// the lock exclusively, and a worker waits for it to fall before its table try takes the lock shared. std::shared_mutex gives a waiting writer no priority: in an
	// area transition the workers' table tries, taking it shared in turns, held the owner's writes 123-193 us each.
	mutable std::atomic<int32_t> m_catalog_writer_waiting {0};
	[[nodiscard]] static bool CatalogWriterFirst() {
		return ::kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4;
	}
	// (Outside KYTY_SPECULATE=4 no other executor reads the catalog: no lock. The mode changes only between submissions.)
	[[nodiscard]] std::unique_lock<std::shared_mutex> CatalogWrite() const {
		if (::kyty_local_speculate_mode.load(std::memory_order_relaxed) != 4) return std::unique_lock<std::shared_mutex>();
		// (A settle part run with this catalog's write held for it, SettleOrQueue / SettleDrain: no second take.)
		if (t_settle_held == &m_catalog_mutex) [[unlikely]]
			return std::unique_lock<std::shared_mutex>();
		std::unique_lock lock(m_catalog_mutex, std::try_to_lock);
		if (!lock.owns_lock()) {
			const bool first = CatalogWriterFirst();
			if (first) m_catalog_writer_waiting.fetch_add(1, std::memory_order_acq_rel);
			lock.lock();
			if (first) m_catalog_writer_waiting.fetch_sub(1, std::memory_order_acq_rel);
		}
		return lock;
	}
	[[nodiscard]] std::shared_lock<std::shared_mutex> CatalogRead() { return std::shared_lock(Catalog().m_catalog_mutex); }
	[[nodiscard]] uint32_t        Role() const noexcept { return m_role; }
	// The catalog's table (null: nothing learned yet), and this executor's own table state.
	[[nodiscard]] TableXpr*       CatalogTable() noexcept { return Catalog().m_table_xpr.get(); }
	TableXpr&                     OwnTable();
	void                          DropStaleCatalog();
	void                          TableForgetStates();
	void                          TableSweepPairs();
	// The linear copy shader the normal path last copied with (DemonsSouls::TryLinearCopy), the shader map generation
	// before its program was prepared, and its dispatches since (every 64th takes the normal path again).
	std::atomic<uint64_t> m_linear_copy_shader {0}, m_linear_copy_generation {0};
	uint32_t              m_linear_copy_uses = 0;
	// (spec_linear_copy_identity) the learned shader's identity beside the generation: the shader map's code size and
	// user-data pointer at its address and its code hash (ShaderHashAt), stored before the address and generation.
	std::atomic<uint64_t> m_linear_copy_size {0}, m_linear_copy_user_data {0}, m_linear_copy_hash {0};
	// Is the shader at `data_addr` the learned linear copy shader: today's rule (the generation it was learned under), or
	// on a speculation's thread / a state pass under KYTY_SPECULATE=4, its identity proven again.
	[[nodiscard]] bool LinearCopyKnown(uint64_t data_addr, bool other_thread);
	// The metadata surfaces compute dispatches cleared (TryConsumeComputeMetaClear), by their shader and user data: a
	// speculation's hole of such a dispatch clears it (Spec::State::PredictClear). (Not the DCC fills
	// TryConsumeComputeImageClear tracks: the normal path consumes those only where it can clear the image, 10-06.)
	std::unordered_map<uint64_t, uint64_t> m_meta_clears;
	// KYTY_SPEC_PREDICT_DCC (live spec_predict_dcc): the DCC surfaces image-clear dispatches filled with a clear code
	// (TryConsumeComputeImageClear's TrackDccFill), by their shader and user data: a speculation's hole of such a
	// dispatch clears them too (Spec::State::PredictClear), as m_meta_clears' do. Without it the draws after the hole
	// were translated as on an uncleared surface and the commit stopped there ("commit: a surface cleared"). (The
	// surfaces it filled lately, newest first: one dispatch and user data runs more than once a frame over descriptors
	// the game rewrites between the runs; all are predicted, a stale one costs a hole.)
	std::unordered_map<uint64_t, std::array<uint64_t, 4>> m_dcc_fills;
	uint64_t m_last_dcc_fill     = 0;     // (the dispatch being made: the surface its DCC fill cleared, 0 none)
	bool     m_last_dcc_fill_set = false; // (it tracked a DCC fill)
	TableResult TableTry(CommandBuffer& buffer, std::span<const uint64_t> commands, uint64_t index_base,
	                     uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct);
	TableResult TableDraw(CommandBuffer& buffer, TablePair& pair, TableVariant& variant,
	                      const NativeXprDrawState& draw_state, std::span<const uint64_t> commands,
	                      uint64_t index_base, uint64_t index_bytes, vk::IndexType index_type,
	                      const NativeXprDirectDraw* direct);
	[[gnu::noinline]] TableImageSet* TableDrawStateSet(CommandBuffer& buffer, TablePair& pair, TableVariant& variant,
	                                                   const NativeXprDrawState& draw_state, TableImageSet* set,
	                                                   Spec::State* spec_state);
	// The table path's functions a weave worker runs, under a policy (weave.h): Weave::Ordered is the GPU thread's,
	// today's code, which the forms without one call; Weave::Worker a weave worker's (table-xpr.inc, weave-draw.inc).
	template <typename P>
	TableResult TableTry(CommandBuffer& buffer, std::span<const uint64_t> commands, uint64_t index_base,
	                     uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct);
	template <typename P>
	bool TableContinue(CommandBuffer& buffer, std::span<const uint64_t> commands, uint64_t index_base,
	                   uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct);
	template <typename P>
	TableResult TableDraw(CommandBuffer& buffer, TablePair& pair, TableVariant& variant,
	                      const NativeXprDrawState& draw_state, std::span<const uint64_t> commands,
	                      uint64_t index_base, uint64_t index_bytes, vk::IndexType index_type,
	                      const NativeXprDirectDraw* direct);
	template <typename P>
	[[nodiscard]] bool TableEvaluate(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data);
	template <typename P>
	void TableGather(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data, bool versioned);
	template <typename P, bool Versioned>
	void TableGather(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data);
	template <typename P>
	[[nodiscard]] TableImageSet* TableSet(CommandBuffer& buffer, TablePair& pair, const TableVariant* variant,
	                                      const NativeXprDrawState* draw_state);
	template <typename P>
	[[nodiscard]] bool TableBlocks(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data,
	                               std::array<vk::DeviceAddress, 2>& blocks, bool* writes, bool versioned);
	template <typename P, bool Versioned>
	[[nodiscard]] bool TableBlocks(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data,
	                               std::array<vk::DeviceAddress, 2>& blocks, bool* writes,
	                               TableXpr& table);
	// A copy in the table upload ring of a small range the guest wrote, as TableBlocks makes one (0: none;
	// table-xpr.inc).
	template <typename P>
	[[nodiscard]] vk::DeviceAddress TableRingCopy(uint64_t address, uint64_t size, bool shareable,
	                                              uint64_t& observation, uint64_t& cpu, bool& copied);
	[[nodiscard]] vk::DeviceAddress TableRingCopy(uint64_t address, uint64_t size);
	// spec_worker_images: a worker's set's pending slots completed by the GPU thread at its commit (Speculation::Before).
	[[nodiscard]] bool TableCompletePendingImages(TableImageSet& set);
	[[nodiscard]] static uint64_t TablePredictPendingConflict(const TableImageSet& set, TextureCache& textures) noexcept;
	// (Mode 4; table-xpr.inc) A speculation makes the table entry a draw lacks (a new pair, a new draw state from a
	// sibling's capture, a variant), draws with it, and its commit publishes it (nullopt: not made, refused as before).
	[[nodiscard]] std::optional<TableResult> TableWorkerEntry(CommandBuffer& buffer, TablePair* existing, uint64_t pair_hash,
	                                                          std::span<const uint64_t> commands, uint64_t index_base,
	                                                          uint64_t index_bytes, vk::IndexType index_type,
	                                                          const NativeXprDirectDraw* direct);
	[[nodiscard]] bool TableEntryCheck(TableEntryHolder& holder);
	void               TableEntrySettle(const std::shared_ptr<TableEntryHolder>& holder, bool commit);
	void               TableEntrySettleInOrder(TableEntryHolder& holder);
	void               TableRetireEntryOrphans();
	// (Mode 4; table-xpr.inc) A worker's own capture of a draw state no capture of whose context exists (its targets
	// resolved read only); the commit's resolution of an entry parked on its table pipeline's compile; the commit's making
	// of a captured state's colour targets no image existed for, and the hook that records a rendering's begin naming them.
	[[nodiscard]] std::unique_ptr<NativeXprDrawState> WorkerCaptureTargets(CommandBuffer& buffer, vk::PrimitiveTopology topology,
	                                                                       bool restart, uint64_t ctx_hash);
	[[nodiscard]] bool TableEntryResolveParked(TableEntryHolder& holder);
	[[nodiscard]] bool TableEntryResolveVirtual(TableEntryHolder& holder, const void* spec_state);
	static bool        TableVirtualRendering(void* data, vk::CommandBuffer command, const vk::RenderingInfo& rendering);
	// (Mode 4; table-xpr.inc) A worker's draw drawn with an entry not in the catalog yet (its translation's own new pair's,
	// the catalog's pair made since; or one whose segment committed, its catalog part waiting), the commit's add of such
	// an entry, and the check at the commit of a draw drawn with one.
	[[nodiscard]] std::optional<TableResult> TableEntryOverlay(CommandBuffer& buffer, const TablePair* existing,
	                                                           std::span<const uint64_t> commands, uint64_t index_base,
	                                                           uint64_t index_bytes, vk::IndexType index_type,
	                                                           const NativeXprDirectDraw* direct);
	void               TableEntryOverlayAdd(const std::shared_ptr<TableEntryHolder>& holder);
	[[nodiscard]] bool TableEntryOverlayCheck(TableEntryHolder& holder);
	// (Mode 4) Whether a captured state's adopted targets are still what the in-order FindImage would reuse.
	[[nodiscard]] bool TableAdoptedCurrent(const NativeXprDrawState& draw_state);
	[[nodiscard]] TableImageSet* TableResolveSet(CommandBuffer& buffer, TablePair& pair, const TableVariant* variant,
	                                             const NativeXprDrawState* draw_state, std::span<const uint32_t> words);
	[[nodiscard]] bool           TableValidateSet(TableImageSet& set);
	// TableValidateSet's answer read only, when it would write nothing but the set's stamps: proven this frame under
	// these registrations, or every image held as it was proven and its full check not due (a weave worker's proof).
	[[nodiscard]] bool           TableSetProven(const TableImageSet& set) const;
	void                         TableRuns(TablePair& pair);
	void                         TableReset(TablePair& pair);
	[[nodiscard]] bool           TableEvaluate(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data);
	[[nodiscard]] TableImageSet* TableSet(CommandBuffer& buffer, TablePair& pair, const TableVariant* variant,
	                                      const NativeXprDrawState* draw_state);
	void                         TablePrepareBda(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data,
	                                            bool flat);
	[[nodiscard]] bool           TableTransit(const TableImageSet& set, vk::CommandBuffer vk_buffer,
	                                          const NativeXprDrawState* draw_state);
	// TableTransit would record nothing: the set's images are those the draw state's targets allow, none awaits an
	// upload or a clear, none is written by the shaders, and each is in its layout and access (read only; a weave
	// worker's).
	[[nodiscard]] bool TableTransitPlaced(const TableImageSet& set, const NativeXprDrawState* draw_state) const;
	void TableStore(CommandBuffer& buffer, const DrawRenderState& state, vk::PrimitiveTopology topology,
	                bool primitive_restart_enable, const RenderState& rendering, bool storable);
	void NativeXprCaptureState(CommandBuffer& buffer, const DrawRenderState& state, const RenderState& rendering,
	                           uint64_t state_key, vk::PrimitiveTopology topology, bool primitive_restart_enable);
	[[nodiscard]] bool NativeXprTargetsCurrent(const NativeXprDrawState& draw_state);
	// NativeXprTargetsCurrent's proof, read only: the targets current at the resolution epoch of now, and what it then
	// writes to the draw state. `shortcut`: the registration log may be read (the GPU thread's; never a weave
	// worker's).
	struct NativeXprTargetsProof {
		uint64_t epoch = 0, erased = 0, checked = 0, span_begin = 0, span_end = 0;
	};
	[[nodiscard]] bool NativeXprTargetsProve(const NativeXprDrawState& draw_state, NativeXprTargetsProof& proof,
	                                         bool shortcut);
	// (`command`: the draw's command buffer, its registers for spec_depth_layers2; none: no target is left to the commit.)
	[[nodiscard]] bool NativeXprEmitTargets(const NativeXprDrawState& draw_state, vk::CommandBuffer vk_buffer,
	                                        const TableImageSet* reading = nullptr, const CommandBuffer* command = nullptr);
	// NativeXprEmitTargets would refuse nothing and record nothing: every target live, GPU-owned, with no fast clear
	// pending since its last emission, its whole image in the target's layout (read only; a weave worker's).
	[[nodiscard]] bool NativeXprTargetsPlaced(const NativeXprDrawState& draw_state) const;
	// The shader access a table image set adds to target `id` (KYTY_TABLE_TARGET_READ; none: empty).
	[[nodiscard]] static vk::AccessFlags2 TableTargetRead(const TableImageSet& set, ImageId id);
	// A weave worker's own state for the table path's worker instantiation (weave-draw.inc): its own lookups, entries,
	// ring copies, emission, last draw and proofs. One a thread, made at its first use.
	struct WeaveScratch;
	[[nodiscard]] static WeaveScratch& WeaveHere();
	// The table a policy keeps its lookups, entries and last draw in: the executor's own, or the worker's scratch's;
	// what its last emission left bound (NativeXprCache::Emit); and its NativeXprTargetsCurrent.
	template <typename P>
	[[nodiscard]] TableXpr& TableOf();
	template <typename P>
	[[nodiscard]] auto& TableEmit();
	template <typename P>
	[[nodiscard]] bool TableTargetsCurrent(const NativeXprDrawState& draw_state);
	// The worker's thread, before its parcel's first draw: the linear copies listed before the parcel and not yet run
	// as it began (`run`: the records the GPU thread has run), one WeaveCopy each, which its reads go through.
	static void WeaveCopiesBegin(const std::atomic<uint64_t>* run);
	static void WeaveCopy(uint64_t index, uint64_t dst, uint64_t src, uint64_t bytes, bool decoded);
	static void WeaveCopiesEnd();
	// The weave (weave.h). A worker's table draw (weave-draw.inc), from the command processor's walk of a parcel on
	// `view`, the worker's view of the GPU thread's command buffer: the table path's worker instantiation,
	// TableContinue<Weave::Worker> when `clean`, else TableTry<Weave::Worker>.
	TableResult WeaveTableDraw(CommandBuffer& view, std::span<const uint64_t> commands, uint64_t index_base,
	                           uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct,
	                           bool clean);
	// The linear copy shader's address (0: none found yet) and the shader map generation it was found at, for the
	// weave's reader (its records of the game's linear copies).
	[[nodiscard]] uint64_t LinearCopyShader(uint64_t& generation) const {
		generation = m_linear_copy_generation.load(std::memory_order_acquire);
		return m_linear_copy_shader.load(std::memory_order_acquire);
	}
	// Weave: the worker and the turn
	// A worker's side (weave-worker.inc), from the command processor's walk of a parcel on worker `worker`'s thread, in
	// this order. WeaveParcelBegin, once the processor holds the parcel's start registers: the worker's view bound to
	// them, the parcel's result cleared (the apart buffers its slot's last parcel left freed here, on the worker), the
	// clocks taken before any read, a slab of the worker's ring, its recording apart begun; false: nothing of it can be
	// made, and it is never opened. WeaveStretch at its first draw packet and at each cut: its needs are grouped from
	// there, and the strand breaks. WeaveDraw at each draw packet, with the arguments the native entries decode: the
	// table draw under a DrawScope, made (a Made record, its calls in the strand's apart buffer) or a hole. WeaveHole:
	// a draw packet the walk declines itself. WeaveStrandStart after a made draw: the start image of the strand it
	// began, for the walk to fill as the registers were before the draw packet (null: it went on in the strand before).
	// WeaveSpoil: a packet the walk could not take. WeaveParcelEnd, always: the last strand closed, the slab kept, the
	// pages merged, the objects, the strands' lists and the register log's block_next made.
	[[nodiscard]] bool              WeaveParcelBegin(uint32_t worker, Weave::Parcel& parcel, HW::Context& registers,
	                                                 HW::UserConfig& user_config, HW::Shader& shaders);
	void                            WeaveStretch(const uint32_t* packet);
	[[nodiscard]] bool              WeaveDraw(const uint32_t* packet, const uint32_t* packet_end, bool clean,
	                                          std::span<const uint64_t> commands, uint64_t index_base, uint64_t index_bytes,
	                                          vk::IndexType index_type, const NativeXprDirectDraw* direct);
	void                            WeaveHole(Weave::Miss miss);
	[[nodiscard]] Weave::Registers* WeaveStrandStart() noexcept;
	void                            WeaveSpoil() noexcept;
	[[nodiscard]] uint32_t          WeaveMadeCount() const noexcept; // (the register log takes one entry each)
	void                            WeaveParcelEnd();
	// What the table path's worker instantiation asks of its worker (weave-worker.inc). The parcel being made: its
	// result takes the draw's notes, and result->made.back() is the draw's own record (WeaveDraw made it). Why the draw
	// is declined. Bytes of the parcel's slab for a block or a copy ({null, 0}: no room). The parcel's serial, its ring
	// copies' sharing terms. A count that moves whenever nothing the worker's buffer bound may be assumed bound (the
	// worker's emission and last draw are then forgotten). WeavePass, first in the worker's TableDraw: the draw's pass
	// noted, and at a change a new stretch there, the pass ended and the next begun in the strand's buffer.
	[[nodiscard]] static Weave::Parcel&                         WeaveParcel() noexcept;
	static void                                                 WeaveMiss(Weave::Miss miss) noexcept;
	[[nodiscard]] static std::pair<uint8_t*, vk::DeviceAddress> WeaveSlabTake(uint64_t size,
	                                                                          uint64_t alignment) noexcept;
	[[nodiscard]] static uint64_t                               WeaveParcelSerial() noexcept;
	[[nodiscard]] static uint64_t                               WeaveForgetCount() noexcept;
	void WeavePass(CommandBuffer& view, const RenderState& rendering);
	// The GPU thread's side (weave-turn.inc). WeaveStart, before the threads start: each worker's view of the command
	// buffer aimed at the apart marker, and its ring (direct calls); `same` compares the processor's registers with the
	// worker's before a strand's draw (its start image, and the register log past its first draw). WeaveOpen at a
	// parcel's first draw packet, its worker done and its start registers the GPU thread's: nothing it read moved since
	// (each refusal counted); true: open. What the command processor's records of the game's linear copies say of the
	// parcel: every copy it read through ran as recorded (`copies_matched`), and the write-clock windows of those
	// copies, (begin, end] and sorted, which the write ring's test passes over
	// (`copies_passed`). WeaveTurnAt at each draw packet of the open parcel: true when a part of a strand was spliced
	// there (`until` the packet after it); false: the GPU thread makes the draw. WeaveSkip after a splice: the part's
	// made draws, when its packets may be skipped by the parcel's register log. WeaveClose: the open parcel closed.
	// WeaveRelease: a parcel's slab released once (`used`: by the GPU thread, with the tick of now). WeaveReclaim: the
	// slabs whose tick is done freed.
	[[nodiscard]] bool WeaveStart(uint32_t workers, const void* processor,
	                              bool (*same)(const void* processor, const Weave::Registers& start,
	                                           const Weave::RegisterLog& log, uint32_t first, uint32_t index));
	// A start that failed (no weave thread runs, no parcel was opened): what WeaveStart made, the views and rings,
	// freed.
	void WeaveForget();
	[[nodiscard]] bool WeaveOpen(Weave::Parcel& parcel, bool copies_matched,
	                             std::span<const std::pair<uint64_t, uint64_t>> copies_passed);
	[[nodiscard]] bool WeaveTurnAt(const uint32_t* packet, const uint32_t*& until);
	[[nodiscard]] bool WeaveSkip(uint32_t& begin, uint32_t& end) noexcept;
	void               WeaveClose() noexcept;
	void               WeaveRelease(Weave::Parcel& parcel, bool used);
	void               WeaveReclaim();
	// Their parts: a worker's strand break and stretch; the turn's needs, strand test, depth clear plan, recopies,
	// stamps and image-start test.
	struct WeaveClear;
	static void        WeaveBreak();
	static void        WeaveNewStretch(const uint32_t* packet);
	void               WeaveNeedsAt(const uint32_t* packet);
	void               WeaveTest(uint32_t strand);
	[[nodiscard]] bool WeaveClearPlan(const NativeXprDrawState& state, WeaveClear& clear);
	[[nodiscard]] bool WeaveRecopyable(uint32_t made, uint32_t use);
	[[nodiscard]] bool WeaveRecopy(uint32_t made, uint32_t use);
	void               WeaveStamps(uint32_t strand, uint32_t begin, uint32_t end);
	[[nodiscard]] bool WeaveStartsHit(const Weave::Parcel& parcel);
	// Weave: end of the worker and the turn

private:
	RenderContext&                        m_context;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorBufferInfo> m_descriptor_buffers;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	std::vector<uint32_t>                 m_image_occurrences;
	std::unordered_set<uint64_t> m_unrepresentable_textures;
	std::unordered_set<uint64_t> m_depth_tiled_reports;

	friend class CommandProcessor;
	friend class Speculation;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */

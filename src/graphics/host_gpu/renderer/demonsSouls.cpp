#include "graphics/host_gpu/renderer/demonsSouls.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

namespace Libs::Graphics::DemonsSouls {
bool IsSupportedGame() {
	// Called by the dispatch path after game metadata has been loaded.
	static const bool supported = [] {
		std::string title;
		return Loader::SystemContentParamSfoGetString("TITLE_ID", &title) && IsSupportedTitle(title);
	}();
	return supported;
}

// The engine's cs_memset32 shader (coredata/enginesupport/shaders/_ps5/cs_memset32) as the builds seen have it:
// 01.007.000's, and 01.005.000's with the period in s12 instead of s8 (the same translated program).
constexpr std::array<uint64_t, 2> LinearCopyShaders {0xeb7456322124ecc7ULL, 0x81bb2b1b9c751ecaULL};

// Verified shader semantics: dst[i] = src[i % period], i < count. Only the
// linear subset is replaced. Other formats, workgroup shapes, overlaps and
// GPU-owned control words stay on the shader path; CopyBuffer owns coherence.
bool TryLinearCopy(const ShaderComputeInputInfo& input, BufferCache& cache, uint32_t x, uint32_t y,
                   uint32_t z, uint32_t mode) {
	const auto& program = *input.stage.program;
	if (!IsSupportedGame() || std::find(LinearCopyShaders.begin(), LinearCopyShaders.end(), program.shader_hash) ==
	                              LinearCopyShaders.end() ||
	    program.user_data_base != 0)
		return false;
	const LinearCopyDispatch dispatch {.user_data          = input.stage.resources.user_data,
	                                   .threads            = {input.threads_num[0], input.threads_num[1],
	                                                          input.threads_num[2]},
	                                   .group_id           = {input.group_id[0], input.group_id[1], input.group_id[2]},
	                                   .thread_ids         = static_cast<uint32_t>(input.thread_ids_num),
	                                   .workgroup_register = static_cast<uint32_t>(input.workgroup_register),
	                                   .tg_size            = input.tg_size_en,
	                                   .thread_dimensions  = input.dispatch_thread_dimensions};
	return TryLinearCopy(dispatch, cache, x, y, z, mode);
}

bool TryLinearCopy(const LinearCopyDispatch& dispatch, BufferCache& cache, uint32_t x, uint32_t y, uint32_t z,
                   uint32_t mode) {
	const auto copy = LinearCopyOf(dispatch, x, y, z, mode);
	if (!copy) return false;
	cache.CopyBuffer(copy->dst, copy->src, copy->bytes, false, false);
	cache.ScheduleCopyFeedback(copy->dst, copy->bytes);
	return true;
}

std::optional<LinearCopy> LinearCopyOf(const LinearCopyDispatch& dispatch, uint32_t x, uint32_t y, uint32_t z, uint32_t mode) {
	const auto& data = dispatch.user_data;
	if (data.size() != 12 || dispatch.thread_dimensions || mode != 0x41 || dispatch.threads[0] != 64 ||
	    dispatch.threads[1] != 1 || dispatch.threads[2] != 1 || !dispatch.group_id[0] || dispatch.group_id[1] ||
	    dispatch.group_id[2] || dispatch.thread_ids != 1 || dispatch.workgroup_register != 12 || dispatch.tg_size ||
	    y != 1 || z != 1 || x == 0)
		return std::nullopt;
	ShaderBufferResource source, destination, parameters;
	std::memcpy(source.fields, data.data(), 16);
	std::memcpy(destination.fields, data.data() + 4, 16);
	std::memcpy(parameters.fields, data.data() + 8, 16);
	for (const auto* d: {&source, &destination}) {
		// Exact control bits observed for idxen format_x UInt32. Reject every
		// unanalysed format/swizzle/OOB/cache/addressing variation.
		if (d->fields[3] != 0x00014004 || (d->fields[1] & 0xffff0000u) != 0x00040000u ||
		    d->Base48() == 0 || (d->Base48() & 3) != 0)
			return std::nullopt;
	}
	if (parameters.fields[3] != 0x0004dfac || (parameters.fields[1] & 0xffff0000u) != 0x00100000u ||
	    parameters.NumRecords() != 1 || (parameters.Base48() & 3) != 0)
		return std::nullopt;
	std::array<uint32_t, 2> controls {};
	if (!Libs::LibKernel::Memory::TryReadGpuCleanBackingToHost(parameters.Base48(), controls.data(),
	                                                           sizeof(controls)))
		return std::nullopt;
	const uint64_t count = controls[0], period = controls[1], bytes = count * 4;
	if (count == 0 || period < count || count > source.NumRecords() ||
	    count > destination.NumRecords() || x != (count + 63) / 64 || bytes > 16 * 1024 * 1024)
		return std::nullopt;
	const auto src = source.Base48(), dst = destination.Base48();
	if ((src < dst + bytes && dst < src + bytes) ||
	    (parameters.Base48() < dst + bytes && dst < parameters.Base48() + 16))
		return std::nullopt;
	if (!LibKernel::Memory::IsUniqueGuestBackingRange(src, bytes) ||
	    !LibKernel::Memory::IsUniqueGuestBackingRange(dst, bytes) ||
	    !LibKernel::Memory::IsUniqueGuestBackingRange(parameters.Base48(), 16))
		return std::nullopt;
	return LinearCopy {src, dst, bytes};
}
} // namespace Libs::Graphics::DemonsSouls

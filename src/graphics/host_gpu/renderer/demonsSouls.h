#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace Libs::Graphics {
class BufferCache;
struct ShaderComputeInputInfo;
namespace DemonsSouls {
// Demon's Souls in the regions seen (PPSA01341 01.007.000 and 01.005.000, PPSA01340 01.005.000), any version:
// each code patch finds its code by the signature of the builds seen or checks the bytes of an audited build,
// the periodic copy checks its shader hashes, and the explicit compute boundaries are the engine's.
constexpr bool IsSupportedTitle(std::string_view title) {
	return title == "PPSA01340" || title == "PPSA01341";
}
bool IsSupportedGame();
// The registers of a dispatch the linear copy shader reads: its user data, workgroup and thread-group inputs.
struct LinearCopyDispatch {
	std::span<const uint32_t> user_data;
	std::array<uint32_t, 3>   threads {};
	std::array<bool, 3>       group_id {};
	uint32_t                  thread_ids = 0, workgroup_register = 0;
	bool                      tg_size = false, thread_dimensions = false;
};
// What a dispatch of the linear copy shader copies (its checks passed), as TryLinearCopy copies it.
struct LinearCopy {
	uint64_t src = 0, dst = 0, bytes = 0;
};
std::optional<LinearCopy> LinearCopyOf(const LinearCopyDispatch& dispatch, uint32_t x, uint32_t y, uint32_t z, uint32_t mode);
// A dispatch of the linear copy shader (a program of its hash), copied on the CPU.
bool TryLinearCopy(const ShaderComputeInputInfo& input, BufferCache& cache, uint32_t x, uint32_t y,
                   uint32_t z, uint32_t mode);
// The same from the registers: for a shader the program-level checks passed for (its code unchanged since).
bool TryLinearCopy(const LinearCopyDispatch& dispatch, BufferCache& cache, uint32_t x, uint32_t y, uint32_t z,
                   uint32_t mode);
} // namespace DemonsSouls
} // namespace Libs::Graphics

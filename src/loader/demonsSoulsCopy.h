#pragma once
#include "common/abi.h"
#include "common/hostException.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace Loader {
struct Program;
namespace DemonsSoulsCopy {
void Install(Program* program);
void Clear();
using PrepareWrite = bool (*)(uint64_t, uint64_t);
// The game's move of more than 32 bytes that does not overlap its source is made again after an exception taken in it
// (HostException::ThreadExceptions: the CRT's memmove holds its first and last 32 bytes in ymm registers until it ends, and
// their upper halves came back zero once in ~120,000 such moves, 11-15: the register tables the game copies into its
// command buffers got 0=0 pairs, a G-buffer target lost its format and ~100 pipelines compiled at once on the render
// thread, 0.2-15 s). A move over its own source cannot be made again: as before. (32 bytes or fewer: xmm registers only.)
inline void* MoveBytes(void* destination, const void* source, size_t size) {
	const auto to = reinterpret_cast<uintptr_t>(destination), from = reinterpret_cast<uintptr_t>(source);
	if (size <= 32 || (to < from + size && from < to + size)) return std::memmove(destination, source, size);
	Common::HostException::RepeatAfterException([&] { std::memmove(destination, source, size); });
	return destination;
}
inline void* Move(void* destination, const void* source, size_t size, PrepareWrite prepare) {
	if (size == 0 || destination == source) return destination;
	// Preserve the normal mapping/fault path after preparing a complete large write.
	if (size >= 256 * 1024 && prepare) (void)prepare(reinterpret_cast<uint64_t>(destination), size);
	return MoveBytes(destination, source, size);
}
inline std::array<uint8_t, 13> Tailcall(uint64_t function) {
	std::array<uint8_t, 13> code {0x5d, 0x48, 0xb8}; // pop rbp; mov rax, function; jmp rax
	std::memcpy(code.data() + 3, &function, sizeof(function));
	code[11] = 0xff;
	code[12] = 0xe0;
	return code;
}
} // namespace DemonsSoulsCopy
} // namespace Loader

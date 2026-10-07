#include "loader/demonsSoulsCopy.h"

#include "common/logging/log.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/runtimeLinker.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <xxhash.h>

namespace Loader::DemonsSoulsCopy {
namespace {
uint64_t                site = 0;
std::array<uint8_t, 13> original {}, patch {};
void* KYTY_SYSV_ABI     CoherentMove(void* destination, const void* source, size_t size) {
    return Move(destination, source, size, &Libs::LibKernel::Memory::TryPrepareHostWrite);
}
// libc.prx's memmove/bcopy/memcpy body, shared by the three entries, which jump to it after the guest's
// argument/stack checks with one saved RBP. Hashed whole, not just the prologue, in each build audited; nothing
// else branches into the replaced bytes (the body's size jump table neither). Other builds, relocations or
// platform loader patches fail closed.
struct Body {
	const char* build;
	uint64_t    offset, size, hash;
};
constexpr std::array<Body, 2> Bodies {{
    {"01.007.000", 0x3c36, 2048, 0xe654325b8be848a9ull},
    {"01.005.000", 0x3d06, 1705, 0xc4be692ee02e9f77ull}, // an older body (another small-size table), the same entries
}};
} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (site || !program || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "libc.prx")
		return;
	const auto body = std::find_if(Bodies.begin(), Bodies.end(), [&](const Body& b) {
		const auto address = program->base_vaddr + b.offset;
		return program->mapped_size >= b.offset + b.size && program->base_vaddr <= UINT64_MAX - b.offset - b.size &&
		       Libs::Graphics::HostMemoryRangeIsReadable(address, b.size) &&
		       XXH3_64bits(reinterpret_cast<const void*>(address), b.size) == b.hash;
	});
	if (body == Bodies.end()) {
		std::printf("Demon's Souls copy: libc.prx of an unknown build; retaining its memmove\n");
		return;
	}
	const auto address = program->base_vaddr + body->offset;
	std::memcpy(original.data(), reinterpret_cast<const void*>(address), original.size());
	patch = Tailcall(reinterpret_cast<uint64_t>(&CoherentMove));
	std::memcpy(reinterpret_cast<void*>(address), patch.data(), patch.size());
	if (!Common::VirtualMemory::FlushInstructionCache(address, patch.size())) {
		std::memcpy(reinterpret_cast<void*>(address), original.data(), original.size());
		Common::VirtualMemory::FlushInstructionCache(address, original.size());
		return;
	}
	site = address;
	std::printf("Demon's Souls copy: installed verified coherent memmove (libc.prx of %s)\n", body->build);
#endif
}

void Clear() {
	if (!site) return;
	if (std::memcmp(reinterpret_cast<const void*>(site), patch.data(), patch.size()) == 0) {
		std::memcpy(reinterpret_cast<void*>(site), original.data(), original.size());
		Common::VirtualMemory::FlushInstructionCache(site, original.size());
	}
	site = 0;
}
} // namespace Loader::DemonsSoulsCopy

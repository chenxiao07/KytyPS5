#include "loader/demonsSoulsIdle.h"

#include "common/logging/log.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"

#include <cstdio>
#include <cstring>

namespace Loader::DemonsSoulsIdle {
namespace {
constexpr uint64_t     PageSize = 0x4000, CaveOffset = 0x8000000;
uint64_t               cave = 0, site = 0;
std::array<uint8_t, 5> installed_call {}, original_call {};

void KYTY_SYSV_ABI WaitForWork() {
	Common::Thread::SleepMicroWithoutSpinning(50);
}

bool Readable(uint64_t address, uint64_t size) {
	// Executable modules are private runtime allocations, not GPU backing aliases.
	return Libs::Graphics::HostMemoryRangeIsReadable(address, size);
}

// The idle loop's call of the poll (one such call site in the game's code) and the poll it calls.
bool FindSites(const Program& program, uint64_t* call, uint64_t* poll) {
	const auto site = GuestCode::FindUnique(program, GuestCode::Pattern(IdleCall));
	if (!site) return false;
	*call = *site + IdleCallAt;
	int32_t displacement = 0;
	std::memcpy(&displacement, reinterpret_cast<const void*>(*call + 1), sizeof(displacement));
	*poll = *call + 5 + static_cast<int64_t>(displacement);
	const GuestCode::Pattern prologue(PollPrologue);
	const uint64_t           base = program.base_vaddr, end = base + program.mapped_size;
	return *poll >= base && *poll <= end - prologue.Size() && Readable(*poll, prologue.Size()) &&
	       prologue.Matches(reinterpret_cast<const uint8_t*>(*poll));
}
} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (cave != 0 || program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin" || program->mapped_size > CaveOffset ||
	    program->base_vaddr > UINT64_MAX - CaveOffset - PageSize)
		return;
	uint64_t call = 0, poll = 0;
	if (!FindSites(*program, &call, &poll)) {
		std::printf("Demon's Souls idle wait: no idle loop of a known build found; retaining guest code\n");
		return;
	}
	const auto requested = program->base_vaddr + CaveOffset;
	const auto allocated = Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    requested, PageSize, Common::VirtualMemory::Mode::ExecuteReadWrite,
	    "demons_souls_idle_wait", true);
	if (allocated != requested) {
		if (allocated) Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	Xbyak::ClearError();
	Xbyak::CodeGenerator code(PageSize, reinterpret_cast<void*>(allocated));
	EmitThunk(code, reinterpret_cast<const void*>(poll),
	          reinterpret_cast<const void*>(&WaitForWork));
	code.ready();
	if (Xbyak::GetError() ||
	    !Common::VirtualMemory::FlushInstructionCache(allocated, code.getSize()) ||
	    !Libs::LibKernel::Memory::ProtectGuestMemory(
	        allocated, PageSize, Common::VirtualMemory::Mode::ExecuteRead, nullptr)) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	installed_call[0]       = 0xe8;
	const auto displacement = static_cast<int32_t>(allocated - (call + 5));
	std::memcpy(installed_call.data() + 1, &displacement, sizeof(displacement));
	std::memcpy(original_call.data(), reinterpret_cast<const void*>(call), original_call.size());
	// Installation occurs before module initializers or guest worker threads run.
	// An external patch with different bytes is deliberately left untouched.
	// SetProgramMemoryProtection keeps executable code writable for loader patches.
	std::memcpy(reinterpret_cast<void*>(call), installed_call.data(), installed_call.size());
	if (!Common::VirtualMemory::FlushInstructionCache(call, installed_call.size())) {
		std::memcpy(reinterpret_cast<void*>(call), original_call.data(), original_call.size());
		Common::VirtualMemory::FlushInstructionCache(call, installed_call.size());
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	cave = allocated;
	site = call;
	std::printf("Demon's Souls idle wait: installed 50 us backoff (call at eboot+0x%llx)\n",
	            static_cast<unsigned long long>(call - program->base_vaddr));
#endif
}

void Clear() {
	if (!cave) return;
	std::array<uint8_t, 5> bytes {};
	std::memcpy(bytes.data(), reinterpret_cast<const void*>(site), bytes.size());
	if (bytes == installed_call) {
		std::memcpy(reinterpret_cast<void*>(site), original_call.data(), bytes.size());
		Common::VirtualMemory::FlushInstructionCache(site, bytes.size());
	}
	Libs::LibKernel::Memory::FreeGuestMemory(cave, PageSize);
	cave = site = 0;
}
} // namespace Loader::DemonsSoulsIdle

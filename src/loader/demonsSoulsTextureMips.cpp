// Demon's Souls' texture streamer (BPE TextureManager; 01.007.000 eboot+0xac5b70 updates a texture's record each frame)
// streams a texture's levels in parts, the smallest first: a 2048x2048 texture as levels 3.. (the tail), then 2, 1 and 0,
// a part each (the texture's part table at + 0x70, each part's first level at its + 0). From the level the frame wants
// (one more once that one is in) the update picks the first level to hold, r15, then lets textureForceWantTopMip
// override it. With KYTY_TEXTURE_REDUCE the GPU holds a large block-compressed texture without its 1 or 2 largest levels
// (descriptors.cpp): their parts were streamed into the game's memory and never read. Here the first level to hold stays
// at or past the levels the GPU leaves out, for the textures it leaves them out of: one layer (+ 0x26c), more levels than
// that (+ 0x274), at least 1024 texels on a side (+ 0x264, + 0x268), and a T# (+ 0xbc) of a 2D image, of the texture's
// height, in a block-compressed format. The GPU holds the same levels as before (from the T#'s first level). Textures
// the game keeps whole (TextureManager::CacheHighMips, + 0x6c > 0) do not come here.
#include "loader/demonsSoulsTextureMips.h"

#include "common/abi.h"
#include "common/virtualMemory.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "live-switch.h"

#ifndef XBYAK_NO_EXCEPTION
#define XBYAK_NO_EXCEPTION
#endif
#include <xbyak/xbyak.h>

extern "C" {
extern volatile std::atomic_uint32_t kyty_local_speculate_mode; // KYTY_SPECULATE (graphics/guest_gpu/speculation.cpp)
}

namespace Loader::DemonsSoulsTextureMips {
namespace {
constexpr uint64_t PageSize = 0x4000, CaveOffset = 0x8000000 + 8 * PageSize;

// 01.007.000 eboot+0xac5c8a (rdx: the texture): test r15d, r15d; setg r8b; test ecx, ecx; mov [r14 + 0x18], ecx; setne
// r9b; and r9b, sil; and r9b, r8b; movzx esi, r9b; mov r9, [rdx + 0x78]; sub r15d, esi (the level more); mov rsi,
// [rip + d] (the override's value pointer); mov esi, [rsi]; test esi, esi; cmovns r15d, esi.
constexpr const char* Pick = "45 85 ff 41 0f 9f c0 85 c9 41 89 4e 18 41 0f 95 c1 41 20 f1 45 20 c1 41 0f b6 f1 4c 8b 4a 78 "
                             "41 29 f7 48 8b 35 ?? ?? ?? ?? 8b 36 85 f6 44 0f 49 fe";
constexpr uint64_t SiteOffset = 31; // sub r15d, esi (3 bytes) and mov rsi, [rip + d] (7) move to the stub
constexpr uint32_t SiteSize   = 10;
constexpr uint32_t Formats    = 512; // the T#'s format field: 9 bits

// upload_at_copy (env KYTY_UPLOAD_AT_COPY; -1, the default: on when the run starts with KYTY_SPECULATE=4, else off). The
// streamer's apply (0x901330) copies a part's decoded bytes into the texture's memory (memcpy at 0x901459) a median
// 14.5 ms before it lowers the T#'s base level; our first view of that level comes ~12.5 ms after that. The apply is
// hooked after the memcpy, where the resident part count is stored (mov [r14 + 0xb8], ebx): the memcpy has returned its
// destination in rax ([record+0x98] + [part+8]) and its size is still in r15 ([part+4]; callee-saved). The bytes just
// written are posted to the GPU thread, which uploads the streamed levels there ahead of their first view
// (TextureCache::UploadLevelAhead).
constexpr const char* ApplyStore = "4c 8b 25 ?? ?? ?? ?? 41 89 9e b8 00 00 00 80 7d c0 01"; // the store at + 7, 7 bytes
constexpr uint64_t    ApplyCave  = 0x8000000 + 16 * PageSize;

void KYTY_SYSV_ABI UploadAtCopy(uint64_t destination, uint64_t size) {
	static auto& on = LiveSwitch::Find("upload_at_copy", -1);
	// (Mode 4 asked at each call: a mode 4 that stands down for memory or CPUs, or a live switch to mode 0, posts nothing.)
	if (on.load(std::memory_order_relaxed) == 0 || ::kyty_local_speculate_mode.load(std::memory_order_relaxed) != 4) return;
	size &= 0xffffffffu;
	if (size != 0) Libs::LibKernel::Memory::PostLevelUploadAhead(destination, size);
}

// fn(a0, a1) with every register, the flags and the guest's red zone kept.
void EmitHostCall(Xbyak::CodeGenerator& c, const void* fn, const Xbyak::Reg64& a0, const Xbyak::Reg64& a1) {
	using namespace Xbyak::util;
	c.lea(rsp, ptr[rsp - 128]);
	c.pushfq();
	for (const auto& reg: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11}) c.push(reg);
	c.mov(r11, rsp);
	c.and_(rsp, -16);
	c.sub(rsp, 528);
	c.mov(ptr[rsp + 512], r11);
	constexpr uint8_t save_fp[] {0x48, 0x0f, 0xae, 0x04, 0x24}; // FXSAVE64 [rsp]
	c.db(save_fp, sizeof(save_fp));
	c.mov(rax, a0);
	c.mov(rsi, a1);
	c.mov(rdi, rax);
	c.mov(rax, reinterpret_cast<uint64_t>(fn));
	c.call(rax);
	c.fxrstor64(ptr[rsp]);
	c.mov(rsp, ptr[rsp + 512]);
	for (const auto& reg: {r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax}) c.pop(reg);
	c.popfq();
	c.lea(rsp, ptr[rsp + 128]);
}

void InstallUploadAtCopy(Program* program) {
	// The performance switches are read on the GPU thread as it starts, which nothing orders before this: KYTY_SPECULATE
	// is read here. A run started in mode 0 has no hook.
	static auto&  upload    = LiveSwitch::Find("upload_at_copy", -1);
	const int64_t value     = upload.load(std::memory_order_relaxed);
	const char*   speculate = std::getenv("KYTY_SPECULATE");
	const bool    on        = value < 0 ? speculate != nullptr && std::atoi(speculate) == 4 : value != 0;
	if (!on) return;
	const auto apply = GuestCode::FindUnique(*program, GuestCode::Pattern(ApplyStore));
	if (!apply || program->base_vaddr > UINT64_MAX - ApplyCave - PageSize) {
		std::printf("Demon's Souls texture streamer: its apply store was not found; upload_at_copy not installed\n");
		return;
	}
	const auto requested = program->base_vaddr + ApplyCave;
	const auto allocated = Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    requested, PageSize, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_upload_at_copy", true);
	if (allocated != requested) {
		if (allocated) Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	const uint64_t site = *apply + 7;
	using namespace Xbyak::util;
	Xbyak::ClearError();
	Xbyak::CodeGenerator c(PageSize, reinterpret_cast<void*>(allocated));
	EmitHostCall(c, reinterpret_cast<const void*>(&UploadAtCopy), rax, r15);
	c.mov(dword[r14 + 0xb8], ebx); // (the replaced store)
	c.jmp(reinterpret_cast<const void*>(site + 7));
	c.ready();
	if (Xbyak::GetError() || !Common::VirtualMemory::FlushInstructionCache(allocated, c.getSize()) ||
	    !Libs::LibKernel::Memory::ProtectGuestMemory(allocated, PageSize, Common::VirtualMemory::Mode::ExecuteRead, nullptr)) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	// (Before guest threads run, as Install's own patch; nothing jumps into the 7-byte store.)
	uint8_t code[7];
	std::memset(code, 0xcc, sizeof(code));
	code[0]         = 0xe9;
	const auto jump = static_cast<int32_t>(allocated - (site + 5));
	std::memcpy(code + 1, &jump, sizeof(jump));
	std::memcpy(reinterpret_cast<void*>(site), code, sizeof(code));
	Common::VirtualMemory::FlushInstructionCache(site, sizeof(code));
	std::printf("Demon's Souls texture streamer: upload_at_copy %lld (KYTY_UPLOAD_AT_COPY; -1: on in mode 4) at its apply "
	            "(eboot+0x%llx)\n",
	            static_cast<long long>(value), static_cast<unsigned long long>(site - program->base_vaddr));
}
} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin" ||
	    program->mapped_size > CaveOffset || program->base_vaddr > UINT64_MAX - CaveOffset - PageSize)
		return;
	// Wherever the GPU leaves those levels out (GameTextureReduceLevels: 12 GiB of video memory or less), whatever the RAM:
	// streamed whole, the game's threads decode and keep levels no draw sees. From v20261108 to here PCs with more than
	// 16 GiB for the emulator streamed whole, because left out they made the buffers BDA shaders read grow with the worlds
	// the game loads (11-07: 2 -> 3-3.4 GiB in 30-40 loads, an 8 GB GPU's budget passed for good); since cb1d7c93 (the
	// game's page-allocator frees drop the buffers over a freed block) they stay at 1.5-1.9 GiB left out too, 59-60 fps on
	// an 8 GB GPU. A 6 GB GPU with 24 GiB of RAM (issue #85: an RTX 2060 slower and hitching since that release; 11-12,
	// simulated, an hour of world changes each): streamed whole 163 slow frames, the worst 149 ms, 33.2 fps; left out 31,
	// 60 ms, 35.6. KYTY_TEXTURE_REDUCE_GAME=0: streamed whole on any PC; =1: left out on any PC.
	InstallUploadAtCopy(program); // (its own page, independent of the level stub below)
	const char*    game   = std::getenv("KYTY_TEXTURE_REDUCE_GAME");
	const uint32_t levels = Libs::Graphics::GameTextureReduceLevels();
	if (levels == 0) return;
	if (game != nullptr && *game != '\0' && game[0] == '0') return;
	const auto pick = GuestCode::FindUnique(*program, GuestCode::Pattern(Pick));
	if (!pick) {
		std::printf("Demon's Souls texture streaming: the streamer is not of a known build; textures streamed whole\n");
		return;
	}
	const auto site = *pick + SiteOffset;
	int32_t    displacement = 0;
	std::memcpy(&displacement, reinterpret_cast<const void*>(site + 6), sizeof(displacement));
	const uint64_t override_pointer = site + SiteSize + static_cast<int64_t>(displacement);
	const auto     requested        = program->base_vaddr + CaveOffset;
	const auto     allocated        = Libs::LibKernel::Memory::AllocateRuntimeMemory(
        requested, PageSize, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_texture_mips", true);
	if (allocated != requested) {
		if (allocated) Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	// The block-compressed formats (the GPU's ImageInfo::IsBlock), a bit each, at the page's end.
	auto* formats = reinterpret_cast<uint8_t*>(allocated + PageSize - Formats / 8);
	std::memset(formats, 0, Formats / 8);
	for (uint32_t format = 0; format < Formats; format++) {
		if (Libs::Graphics::Prospero::BlockCompressedBytesPerBlock(static_cast<Libs::Graphics::Prospero::BufferFormat>(format)) != 0)
			formats[format / 8] |= static_cast<uint8_t>(1u << (format % 8));
	}
	using namespace Xbyak::util;
	Xbyak::ClearError();
	Xbyak::CodeGenerator c(PageSize - Formats / 8, reinterpret_cast<void*>(allocated));
	Xbyak::Label        keep;
	// The level more, then the first level held past the levels left out. (esi and r8 are free: the code after the site
	// loads both before a use.)
	c.sub(r15d, esi);
	c.cmp(r15d, levels);
	c.jge(keep);
	c.cmp(dword[rdx + 0x26c], 1);
	c.jne(keep);
	c.cmp(dword[rdx + 0x274], levels);
	c.jbe(keep);
	c.mov(esi, dword[rdx + 0x264]);
	c.mov(r8d, dword[rdx + 0x268]);
	c.cmp(esi, r8d);
	c.cmovb(esi, r8d);
	c.cmp(esi, 1024);
	c.jb(keep);
	c.mov(esi, dword[rdx + 0xc4]); // T# dword 2: the height less 1 at bits 14..27
	c.shr(esi, 14);
	c.and_(esi, 0x3fff);
	c.inc(esi);
	c.cmp(esi, r8d);
	c.jne(keep);
	c.mov(esi, dword[rdx + 0xc8]); // dword 3: the type at bits 28..31 (9: 2D)
	c.shr(esi, 28);
	c.cmp(esi, 9);
	c.jne(keep);
	c.mov(esi, dword[rdx + 0xc0]); // dword 1: the format at bits 20..28
	c.shr(esi, 20);
	c.and_(esi, Formats - 1);
	c.mov(r8, reinterpret_cast<uint64_t>(formats));
	c.bt(dword[r8], esi);
	c.jnc(keep);
	c.mov(r15d, levels);
	// The moved load of the override's value pointer (its address now absolute) and back.
	c.L(keep);
	c.mov(rsi, override_pointer);
	c.mov(rsi, qword[rsi]);
	c.jmp(reinterpret_cast<const void*>(site + SiteSize));
	c.ready();
	if (Xbyak::GetError() || !Common::VirtualMemory::FlushInstructionCache(allocated, c.getSize()) ||
	    !Libs::LibKernel::Memory::ProtectGuestMemory(allocated, PageSize, Common::VirtualMemory::Mode::ExecuteRead, nullptr)) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	// Installation occurs before module initializers or guest threads run; program memory is writable to the host.
	// (The bytes after the jump are never reached: nothing jumps into the moved instructions.)
	uint8_t code[SiteSize];
	std::memset(code, 0xcc, sizeof(code));
	code[0]          = 0xe9;
	const auto jump  = static_cast<int32_t>(allocated - (site + 5));
	std::memcpy(code + 1, &jump, sizeof(jump));
	std::memcpy(reinterpret_cast<void*>(site), code, sizeof(code));
	Common::VirtualMemory::FlushInstructionCache(site, sizeof(code));
	std::printf("Demon's Souls texture streaming: large block-compressed textures streamed without their %u largest "
	            "level(s), as the GPU holds them (eboot+0x%llx; KYTY_TEXTURE_REDUCE_GAME=0 streams them whole)\n",
	            levels, static_cast<unsigned long long>(site - program->base_vaddr));
#endif
}

} // namespace Loader::DemonsSoulsTextureMips

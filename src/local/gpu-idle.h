#pragma once
// Whether the GPU thread is in one of its idle waits (GuestGpu::ThreadRun), for the address space's work on parked
// memory off the game's thread (kernel/memoryAddressSpace.inc), which then runs without holding the address space busy
// under the GPU thread's protection changes.

#include <atomic>

namespace GpuIdle {
inline std::atomic<bool> g_idle {false};
} // namespace GpuIdle

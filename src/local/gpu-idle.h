#pragma once
// release_trickle: whether the GPU thread is in one of its idle waits (GuestGpu::ThreadRun), for
// Kyty.Release (kernel/memoryAddressSpace.inc), which then releases a parked view without holding the address space
// busy under the GPU thread's protection changes.

#include <atomic>

namespace GpuIdle {
inline std::atomic<bool> g_idle {false};
} // namespace GpuIdle

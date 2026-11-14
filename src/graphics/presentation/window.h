#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_

#include "common/abi.h"
#include "common/common.h"

namespace Libs::Graphics {

class Presenter;

[[nodiscard]] Presenter& WindowInit(uint32_t width, uint32_t height);
void                     WindowRun();
void                     WindowShutdown();
// The size in pixels the window shows the game at (in full screen the display's), as WindowInit found it; 0 x 0 before.
void WindowOutputSize(uint32_t* width, uint32_t* height);
// KYTY_PREFETCH_WAIT=1: the calling thread (the guest's main thread, before any guest code) waits for the shader prefetch
// (PipelineCache::HoldForShaderPrefetch), its progress on the start-up screen while the window's loop runs. Off: returns.
void WindowHoldForShaderPrefetch();

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_ */

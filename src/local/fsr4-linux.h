#ifndef EMULATOR_SRC_LOCAL_FSR4_LINUX_H_
#define EMULATOR_SRC_LOCAL_FSR4_LINUX_H_
// AMD FSR 4 (FSR Upscaling 4.1.2) on Linux, from AMD's own Windows DLLs (FSR SDK 2.4: amd_fidelityfx_loader_vk.dll,
// amd_fidelityfx_upscaler_vk.dll, and amdvkffx64.dll from the Adrenalin driver), run in this native process by
// ffx-vk-linux (github.com/DuIslingr/ffx-vk-linux, MIT: a small PE loader, no Wine). The DLLs are the user's: they are
// not in this repository and not distributed with it (AMD's licence is binary-only).
//
// KYTY_FSR4=1 (with KYTY_UPSCALER=fsr): FsrNative::Upscale tries FSR 4 first and falls back to FSR 3 built in
// (fsr-native.h) when it cannot run: no libffx_vk_linux.so or DLLs, an unsupported GPU (AMD lists RX 6000 and newer for
// FSR 4.1.2), or a failed context. Where it looks: KYTY_FSR4_DIR, else next to the emulator, else its "fsr4" folder; the
// DLLs are looked for in the same folder (FFXVK_DRIVER_DLL names amdvkffx64.dll elsewhere). get-fsr4-linux.sh builds
// the library and makes that folder.
//
// Built only where ffx-vk-linux's headers were found at configure time (CMakeLists.txt: KYTY_FFX_VK_LINUX); elsewhere
// every function answers false and the device is made as before.
#include "local/frame-gen.h"

#include <vector>

namespace Libs::Graphics {
struct GraphicContext;
namespace Fsr4Linux {

// KYTY_FSR4=1 in a build with the ffx-vk-linux headers.
[[nodiscard]] bool Requested();
// Device creation (vulkanWindow.cpp): the device extensions FSR 4's shaders and AMD's Vulkan backend need that the GPU
// offers, when FSR 4 is requested. (The features they bring are enabled by the caller, next to the device's creation.)
void AddDeviceExtensions(const std::vector<vk::ExtensionProperties>& available, std::vector<const char*>& device_extensions);
// GPU thread or the recording worker: records FSR 4's upscale of `inputs` into `command`, as FsrNative::Upscale takes
// them; false when it could not (the caller uses FSR 3 built in). The images are left in their layouts.
[[nodiscard]] bool Upscale(GraphicContext& graphics, vk::CommandBuffer command, const FrameGen::UpscaleInputs& inputs);

} // namespace Fsr4Linux
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_LOCAL_FSR4_LINUX_H_

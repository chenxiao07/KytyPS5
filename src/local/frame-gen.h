#ifndef EMULATOR_SRC_LOCAL_FRAME_GEN_H_
#define EMULATOR_SRC_LOCAL_FRAME_GEN_H_
// DLSS Frame Generation through NVIDIA Streamline (KYTY_FRAMEGEN=<generated frames, 1..5>), for
// Demon's Souls PPSA01341 01.007.000 and 01.005.000. The inputs come from the game's own passes, found by frame
// capture (docs/RE-DEMONS-SOULS.md): the G-buffer vertex shaders' constants hold the camera
// (world->clip matrix, the previous frame's, and the TAA jitter), the TAA dispatch reads the depth
// and motion-vector images. The renderer thread gathers them per guest frame; at the guest flip
// they are copied next to the presenter's frame; the present thread hands them to Streamline and
// presents through it (sl.interposer.dll is the process' Vulkan loader, see Initialize()).
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/image.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace Libs::Graphics {
class CommandBuffer;
class TextureCache;
namespace FrameGen {

// The TAA compute shader, and the G-buffer vertex shaders sharing the constant layout below (flattened SRT
// words), of 01.007.000 and of 01.005.000: the same shaders recompiled (KYTY_SHADER_DUMP: the same resources,
// the vertex shaders the same translated program; the TAA images are what is read from its dispatch).
constexpr std::array<uint64_t, 2> TaaShaders {0xa44e580f4f0cfb2bULL, 0x934a6e1197b99148ULL};
constexpr std::array<uint64_t, 6> GbufferVertexShaders {0xf4d2e78cd1769fc7ULL, 0x08e16923875aa8e9ULL, 0x0ffcbf08be26e74eULL,
                                                        0x1d69b5d27198446aULL, 0x5931e753b757f3bcULL, 0xa97f610824942162ULL};
inline bool IsTaaShader(uint64_t hash) {
	return std::find(TaaShaders.begin(), TaaShaders.end(), hash) != TaaShaders.end();
}
inline bool IsGbufferVertexShader(uint64_t hash) {
	return std::find(GbufferVertexShaders.begin(), GbufferVertexShaders.end(), hash) != GbufferVertexShaders.end();
}
constexpr uint32_t SrtViewProj     = 29; // 16 floats, row-major, clip = M * world
constexpr uint32_t SrtPrevViewProj = 45; // 12 floats: the previous frame's rows x, y and w
constexpr uint32_t SrtJitter       = 57; // 2 floats: clip-space offset (NDC units)

struct Camera {
	std::array<float, 16> view_proj {};
	std::array<float, 16> prev_view_proj {};
	std::array<float, 2>  jitter_ndc {};
	bool                  valid = false;
};

// One presented frame's inputs, owned by the presenter's frame.
struct FrameInputs {
	VulkanImage   depth, motion;
	vk::ImageView depth_view = nullptr, motion_view = nullptr;
	vk::Extent2D  render_extent {};
	Camera        camera;
	uint64_t      guest_frame = 0;
	bool          motion_attachment = false; // the motion copy is a color attachment too
	bool          valid       = false;
};

// Process start, before the window: loads Streamline when KYTY_FRAMEGEN or KYTY_UPSCALER=dlss is set (it then
// becomes SDL's Vulkan loader, and so the proxy of instance, device and swapchain calls).
void Initialize();
// After Initialize: Streamline is loaded (device creation then adds what it relies on but does
// not add itself, VK_KHR_present_id for its VK_NV_low_latency2).
[[nodiscard]] bool Requested();
// Before the device is created: why Streamline says DLSS-G cannot run with this GPU, driver and Windows (e.g.
// hardware-accelerated GPU scheduling turned off), logged; empty when it can, or when frame generation is off. A refused
// device's error repeats it.
[[nodiscard]] std::string SupportProblem(vk::PhysicalDevice physical_device);
// A refused device's advice when this process runs without bottom-up ASLR (issue #45: forcing it on for the emulator in
// Windows' Exploit protection let the driver make the device); empty otherwise.
[[nodiscard]] std::string RefusedDeviceAdvice();
// After the device exists: checks that DLSS-G is available on it.
void OnDevice(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device);
// Right after the dispatcher is loaded for the device (before the recording layer copies it):
// the command-buffer calls Streamline intercepts only for its command-list state tracking
// (disabled here) go straight to the driver.
void BypassCommandHooks(vk::Device device);
[[nodiscard]] bool Enabled();
[[nodiscard]] bool OpenDlssgDetected();

// Present thread, around vkQueuePresentKHR of the image showing `inputs` (null: a frame
// without inputs, e.g. a menu). `backbuffer` is the swapchain extent, `region` the part the
// game image covers.
void BeforePresent(const FrameInputs* inputs, vk::Extent2D backbuffer, vk::Rect2D region,
                   vk::Format format, uint32_t image_count);
void AfterPresent();
// Present thread: the swapchain is made again (DLSS-G off until frames with inputs turn it on again).
void SwapchainChanging();
// Any thread: a frame with inputs was left unshown (the presenter kept a newer one): the next one resets DLSS-G's history.
void NoteDroppedFrame();
// Present thread: the frames DLSS-G showed per presented frame (1 without generation), as last
// queried (about once a second), for the KYTY_FPS_HUD panel.
[[nodiscard]] uint32_t PresentedFrames();
// KYTY_FPS_HUD: the panel's rectangle, normalized to the game image. Its motion vectors are
// zeroed so generated frames keep the panel in place.
void SetStaticRect(float x0, float y0, float x1, float y1);

// Renderer thread (Kyty.Gpu).
// The guest frame ran the TAA pass: the game's 3D view (play, or a cutscene in the engine), not a movie, a menu or a
// loading screen; with or without frame generation, by the normal or the table way. The guest flip takes it
// (Presenter::PrepareFrame): above 60 Hz the other frames keep 60 (videoOut.cpp).
void NoteScene();
[[nodiscard]] bool TakeScene();
// The current guest frame ran the TAA pass so far, or the last one flipped did (HudAnchor: the game queues a frame's
// interface before the flip of the one before).
[[nodiscard]] bool SceneNoted();
// The guest flips taken so far (TakeScene).
[[nodiscard]] uint64_t GuestFlips();
// The current guest frame has no camera yet: its next G-buffer draw goes the normal way (the table way's draws have
// no constants on the CPU), whose hook reads it.
[[nodiscard]] bool WantsCamera();
void OnDraw(uint64_t vs_hash, std::span<const uint32_t> flattened_srt);
void OnDispatch(uint64_t cs_hash, ImageId depth, uint64_t depth_address, ImageId motion,
                uint64_t motion_address);
// The game's super resolution dispatch in place of its TAA (renderSuperResolution.cpp): the same depth and motion.
void OnUpscale(ImageId depth, uint64_t depth_address, ImageId motion, uint64_t motion_address);
// At the guest flip, in the frame's command buffer: copy the gathered images into `inputs`.
void PrepareInputs(GraphicContext& graphics, TextureCache& cache, CommandBuffer& command,
                   FrameInputs& inputs);
void DestroyInputs(GraphicContext& graphics, FrameInputs& inputs);

// DLSS super resolution (KYTY_UPSCALER=dlss) in place of Sony's PS5 Pro upscaler the game dispatches
// (graphics/guest_gpu/superResolution.h, renderer/renderSuperResolution.cpp).
struct UpscaleImage {
	vk::Image           image  = nullptr;
	vk::ImageView       view   = nullptr;
	vk::ImageLayout     layout = vk::ImageLayout::eUndefined; // as the command buffer leaves it for the upscaler
	vk::Format          format = vk::Format::eUndefined;
	vk::Extent2D        extent {};
	vk::ImageUsageFlags usage {};
};
struct UpscaleInputs {
	UpscaleImage color, depth, motion, output; // color, depth and motion at the render size
	float        jitter_x = 0.0f, jitter_y = 0.0f;       // the projection's offset, in render pixels
	float        motion_scale_x = 0.0f, motion_scale_y = 0.0f; // motion vector * scale = render pixels to the previous
	bool         reset = false;                           // no history (a cut, a level load)
	std::array<float, 16> view_to_clip {};                // row vectors (p' = p * M)
	float        near_plane = 0.1f, far_plane = 3000.0f;
};
// Streamline offers DLSS super resolution on this device.
[[nodiscard]] bool UpscalerReady();
// Records DLSS into `command` (outside a render pass; the images in their `layout`s): false when it could not.
[[nodiscard]] bool Upscale(vk::CommandBuffer command, const UpscaleInputs& inputs);

} // namespace FrameGen
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_LOCAL_FRAME_GEN_H_

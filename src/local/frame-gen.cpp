#include "frame-gen.h"

#include "common/logging/log.h"
#include "local-platform.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/render.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>

#ifdef KYTY_FRAMEGEN_STREAMLINE
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
// NVIDIA Streamline SDK headers (_Build/deps/streamline/sdk/include, see CMakeLists.txt).
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss.h>
#include <sl_dlss_g.h>
#include <sl_helpers.h>
#include <sl_pcl.h>
#include <sl_reflex.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#endif

extern "C" {
// DLSS-G frames' Reflex sleep (slReflexSleep, every present): 0 skips it (KYTY_FRAMEGEN_SLEEP=0).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_framegen_sleep {1};
}

namespace Libs::Graphics::FrameGen {

namespace {

struct Sources {
	ImageId  depth, motion;
	uint64_t depth_address = 0, motion_address = 0;
	bool     valid = false;
};

int      g_generate    = 0;     // frames to generate per rendered frame (0 = off)
bool     g_ready       = false; // Streamline loaded and the feature available
bool     g_upscale     = false; // KYTY_UPSCALER=dlss: DLSS super resolution asked for (Streamline loaded for it)
bool     g_upscale_ready = false; // ... and available on the device
Camera   g_camera;              // Kyty.Gpu: the current guest frame
Sources  g_sources;             // Kyty.Gpu: the current guest frame
uint64_t g_guest_frame = 0;     // Kyty.Gpu
std::atomic<bool> g_scene {false}; // the current guest frame ran the TAA pass
std::atomic<bool> g_last_scene {false}; // ... the last one the guest flipped
std::atomic<uint64_t> g_flips_taken {0}; // TakeScene calls (guest flips)

float AsFloat(uint32_t word) {
	float value = 0.0f;
	std::memcpy(&value, &word, sizeof(value));
	return value;
}

bool IsDepthFormat(vk::Format format) {
	switch (format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD32Sfloat:
		case vk::Format::eD32SfloatS8Uint:
		case vk::Format::eX8D24UnormPack32: return true;
		default: return false;
	}
}

void Transition(vk::CommandBuffer command, VulkanImage& image, vk::ImageAspectFlags aspect,
                vk::ImageLayout layout, vk::AccessFlags2 access, vk::PipelineStageFlags2 stage) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = image.state.pl_stage;
	barrier.srcAccessMask       = image.state.access_mask;
	barrier.dstStageMask        = stage;
	barrier.dstAccessMask       = access;
	barrier.oldLayout           = image.state.layout;
	barrier.newLayout           = layout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image.image;
	barrier.subresourceRange    = {aspect, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
}

// (Re)create `image` as a sampled copy target like `source`; with `attachment`, a color
// attachment too where the format allows it (*attachment tells).
void Configure(GraphicContext& graphics, VulkanImage& image, vk::ImageView& view,
               const VulkanImage& source, bool* attachment = nullptr) {
	if (image.image != nullptr && image.format == source.format &&
	    image.extent.width == source.extent.width && image.extent.height == source.extent.height) {
		return;
	}
	if (view != nullptr) {
		graphics.device.destroyImageView(view, nullptr);
		view = nullptr;
	}
	if (image.image != nullptr) {
		graphics.DeleteImage(image);
	}
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {source.extent.width, source.extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = source.format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	if (attachment != nullptr) {
		*attachment = !IsDepthFormat(source.format) &&
		              static_cast<bool>(graphics.physical_device.getFormatProperties(source.format).optimalTilingFeatures &
		                                vk::FormatFeatureFlagBits::eColorAttachment);
		if (*attachment) {
			create.usage |= vk::ImageUsageFlagBits::eColorAttachment;
		}
	}
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, image)) {
		EXIT("frame generation: cannot allocate a %ux%u input image\n", source.extent.width,
		     source.extent.height);
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = image.image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = source.format;
	view_info.subresourceRange = {IsDepthFormat(source.format) ? vk::ImageAspectFlagBits::eDepth
	                                                           : vk::ImageAspectFlagBits::eColor,
	                              0, 1, 0, 1};
	RequireVulkanSuccess(graphics.device.createImageView(&view_info, nullptr, &view),
	                     "frame generation input view");
}

void CopyInto(vk::CommandBuffer command, Image& source, VulkanImage& target) {
	const auto aspect = IsDepthFormat(source.backing.format) ? vk::ImageAspectFlagBits::eDepth
	                                                         : vk::ImageAspectFlagBits::eColor;
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transition(command, target, aspect, vk::ImageLayout::eTransferDstOptimal,
	           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
	vk::ImageCopy copy {};
	copy.srcSubresource = {aspect, 0, 0, 1};
	copy.dstSubresource = {aspect, 0, 0, 1};
	copy.extent         = target.extent;
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, target.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy);
	Transition(command, target, aspect, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
}

// KYTY_FPS_HUD: frames shown per presented frame, and the panel's normalized rectangle (16 bits
// per coordinate: x0, y0, x1, y1; 0 = none).
std::atomic<uint32_t> g_presented_frames {1};
std::atomic<uint64_t> g_static_rect {0};

// No motion under the KYTY_FPS_HUD panel: DLSS-G would move it with the scene behind it.
void ZeroStaticRect(vk::CommandBuffer command, VulkanImage& motion, vk::ImageView view) {
	const auto packed = g_static_rect.load(std::memory_order_relaxed);
	if (packed == 0) {
		return;
	}
	const auto coordinate = [&](uint32_t index, uint32_t size) {
		return static_cast<int32_t>(((packed >> (16u * index)) & 0xffffu) * size / 0xffffu);
	};
	const int32_t x0 = coordinate(0, motion.extent.width), y0 = coordinate(1, motion.extent.height);
	const int32_t x1 = coordinate(2, motion.extent.width), y1 = coordinate(3, motion.extent.height);
	if (x1 <= x0 || y1 <= y0) {
		return;
	}
	Transition(command, motion, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal,
	           vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
	vk::RenderingAttachmentInfo color {};
	color.imageView   = view;
	color.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	color.loadOp      = vk::AttachmentLoadOp::eLoad;
	color.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = vk::Extent2D {motion.extent.width, motion.extent.height};
	rendering.layerCount           = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &color;
	command.beginRendering(rendering);
	vk::ClearAttachment clear {};
	clear.aspectMask       = vk::ImageAspectFlagBits::eColor;
	clear.colorAttachment  = 0;
	clear.clearValue.color = vk::ClearColorValue {std::array<float, 4> {0.0f, 0.0f, 0.0f, 0.0f}};
	vk::ClearRect rect {};
	rect.rect       = vk::Rect2D {{x0, y0}, {static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0)}};
	rect.layerCount = 1;
	command.clearAttachments(1, &clear, 1, &rect);
	command.endRendering();
	Transition(command, motion, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
}

#ifdef KYTY_FRAMEGEN_STREAMLINE
// Streamline entry points (sl.interposer.dll, loaded at run time) and feature functions.
struct Api {
	PFun_slInit*                 init                = nullptr;
	PFun_slIsFeatureSupported*   is_feature_supported = nullptr;
	PFun_slIsFeatureLoaded*      is_feature_loaded   = nullptr;
	PFun_slGetFeatureFunction*   get_feature_function = nullptr;
	PFun_slGetNewFrameToken*     get_new_frame_token = nullptr;
	PFun_slSetConstants*         set_constants       = nullptr;
	PFun_slSetTagForFrame*       set_tag_for_frame   = nullptr;
	PFun_slDLSSGSetOptions*      dlssg_set_options   = nullptr;
	PFun_slDLSSGGetState*        dlssg_get_state     = nullptr;
	PFun_slReflexSetOptions*     reflex_set_options  = nullptr;
	PFun_slPCLSetMarker*         pcl_set_marker      = nullptr;
	PFun_slReflexSleep*          reflex_sleep        = nullptr;
	PFun_slEvaluateFeature*      evaluate_feature    = nullptr;
	PFun_slDLSSSetOptions*       dlss_set_options    = nullptr;
	PFun_slDLSSGetOptimalSettings* dlss_get_optimal_settings = nullptr;
};
Api              g_api;
sl::FrameToken*  g_token = nullptr; // present thread: the frame being presented
uint32_t         g_frame_index = 0;
bool             g_options_set = false;
vk::Extent2D     g_options_backbuffer {};
vk::Extent2D     g_options_render {};
vk::Format       g_options_format = vk::Format::eUndefined; // SDR or HDR10 swapchain
uint64_t         g_presents = 0;
// DLSS-G on (generating) or off with its resources kept (eRetainResourcesWhenOff): off after TurnOffAfter presents in a
// row without inputs (menus, movies, loading), on after TurnOnAfter with them, so a fade or a cut does not flap it. On
// with null tags it was asked to generate frames it had no inputs for (Streamline "Failed to find global tag").
constexpr uint32_t TurnOffAfter = 30, TurnOnAfter = 10;
bool               g_mode_on    = false;
uint32_t           g_run_with   = 0, g_run_without = 0; // presents in a row with inputs, without them
// A frame with inputs the presenter left unshown (NoteDroppedFrame): the next one's motion spans two game frames.
std::atomic<bool>  g_discontinuity {false};

// Matrices in Streamline's convention: row-major, row vectors (p' = p * M).
using Mat4 = sl::float4x4;

Mat4 Transposed(const std::array<float, 16>& m) { // our column-vector matrix -> SL
	Mat4 r;
	for (uint32_t i = 0; i < 4; ++i) {
		r[i] = sl::float4(m[i], m[4 + i], m[8 + i], m[12 + i]);
	}
	return r;
}

float At(const sl::float4& v, uint32_t i) {
	return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

Mat4 Multiply(const Mat4& a, const Mat4& b) {
	Mat4 r;
	for (uint32_t i = 0; i < 4; ++i) {
		float row[4] {};
		for (uint32_t j = 0; j < 4; ++j) {
			for (uint32_t k = 0; k < 4; ++k) row[j] += At(a[i], k) * At(b[k], j);
		}
		r[i] = sl::float4(row[0], row[1], row[2], row[3]);
	}
	return r;
}

bool Inverted(const Mat4& m, Mat4& out) {
	double a[4][8] {};
	for (uint32_t i = 0; i < 4; ++i) {
		for (uint32_t j = 0; j < 4; ++j) a[i][j] = At(m[i], j);
		a[i][4 + i] = 1.0;
	}
	for (uint32_t c = 0; c < 4; ++c) {
		uint32_t pivot = c;
		for (uint32_t r = c + 1; r < 4; ++r) {
			if (std::fabs(a[r][c]) > std::fabs(a[pivot][c])) pivot = r;
		}
		if (std::fabs(a[pivot][c]) < 1e-12) return false;
		for (uint32_t j = 0; j < 8; ++j) std::swap(a[c][j], a[pivot][j]);
		const double scale = 1.0 / a[c][c];
		for (uint32_t j = 0; j < 8; ++j) a[c][j] *= scale;
		for (uint32_t r = 0; r < 4; ++r) {
			if (r == c) continue;
			const double f = a[r][c];
			for (uint32_t j = 0; j < 8; ++j) a[r][j] -= f * a[c][j];
		}
	}
	for (uint32_t i = 0; i < 4; ++i) {
		out[i] = sl::float4(static_cast<float>(a[i][4]), static_cast<float>(a[i][5]),
		                    static_cast<float>(a[i][6]), static_cast<float>(a[i][7]));
	}
	return true;
}

// The game's world->clip matrix is P * V with a perspective P whose w row is (0, 0, 1, 0) (w =
// view depth) and whose z row is constant (reversed-Z, infinite far). Split it into camera
// axes, position and projection terms.
struct CameraFrame {
	sl::float3 right, up, forward, position;
	float      sx = 1.0f, sy = 1.0f, cx = 0.0f, cy = 0.0f, z_near = 0.1f;
	bool       valid = false;
};

CameraFrame Decompose(const std::array<float, 16>& m) {
	CameraFrame c;
	const auto row = [&](uint32_t i) { return std::array<double, 4> {m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]}; };
	const auto r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
	const double flen = std::sqrt(r3[0] * r3[0] + r3[1] * r3[1] + r3[2] * r3[2]);
	if (flen < 1e-6) return c;
	const double f[3] {r3[0] / flen, r3[1] / flen, r3[2] / flen};
	const double tf = r3[3] / flen;
	const auto split = [&](const std::array<double, 4>& r, double& s, double& off, double axis[3], double& t) {
		off = (r[0] * f[0] + r[1] * f[1] + r[2] * f[2]) / flen;
		double v[3] {r[0] / flen - off * f[0], r[1] / flen - off * f[1], r[2] / flen - off * f[2]};
		s = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		if (s < 1e-9) return false;
		for (int i = 0; i < 3; ++i) axis[i] = v[i] / s;
		t = (r[3] / flen - off * tf) / s;
		return true;
	};
	double sx = 0, cx = 0, sy = 0, cy = 0, x[3] {}, y[3] {}, tx = 0, ty = 0;
	if (!split(r0, sx, cx, x, tx) || !split(r1, sy, cy, y, ty)) return c;
	// Up = forward x right (Streamline's convention); a flipped clip y shows up as a negative sy.
	const double up[3] {f[1] * x[2] - f[2] * x[1], f[2] * x[0] - f[0] * x[2], f[0] * x[1] - f[1] * x[0]};
	if (up[0] * y[0] + up[1] * y[1] + up[2] * y[2] < 0) {
		sy = -sy;
		ty = -ty;
	}
	c.right    = sl::float3(float(x[0]), float(x[1]), float(x[2]));
	c.up       = sl::float3(float(up[0]), float(up[1]), float(up[2]));
	c.forward  = sl::float3(float(f[0]), float(f[1]), float(f[2]));
	// View translation (tx, ty, tf) = -axes . position.
	const double pos[3] {-(tx * x[0] + ty * up[0] + tf * f[0]), -(tx * x[1] + ty * up[1] + tf * f[1]),
	                     -(tx * x[2] + ty * up[2] + tf * f[2])};
	c.position = sl::float3(float(pos[0]), float(pos[1]), float(pos[2]));
	c.sx       = float(sx);
	c.sy       = float(sy);
	c.cx       = float(cx);
	c.cy       = float(cy);
	c.z_near   = float(r2[3] / flen);
	c.valid    = true;
	return c;
}

void LogMessage(sl::LogType type, const char* message) {
	LOGF("Streamline[%d]: %s", static_cast<int>(type), message);
	// KYTY_FRAMEGEN_VERBOSE: every message on stdout (the run log), not only errors.
	static const bool verbose = std::getenv("KYTY_FRAMEGEN_VERBOSE") != nullptr;
	if (type == sl::LogType::eError || verbose) {
		std::printf("Streamline %s: %s", type == sl::LogType::eError ? "error" : "log", message);
		std::fflush(stdout);
	}
}

template <typename F>
bool FeatureFunction(sl::Feature feature, const char* name, F*& function) {
	void* pointer = nullptr;
	if (g_api.get_feature_function(feature, name, pointer) != sl::Result::eOk || pointer == nullptr) {
		LOGF("frame generation: %s is unavailable\n", name);
		return false;
	}
	function = reinterpret_cast<F*>(pointer);
	return true;
}
#endif

} // namespace

bool Enabled() {
	return g_ready;
}

bool OpenDlssgDetected() {
#if defined(_WIN32)
	static const bool detected = [] {
		HMODULE ver = GetModuleHandleW(L"version.dll");
		wchar_t exe[MAX_PATH] {};
		GetModuleFileNameW(nullptr, exe, MAX_PATH);
		std::error_code ec;
		const auto exe_dir = std::filesystem::path(exe).parent_path();
		return (ver != nullptr && std::filesystem::exists(exe_dir / "version.dll", ec)) ||
		       std::filesystem::exists(exe_dir / "opendlssg.ini", ec);
	}();
	return detected;
#else
	return false;
#endif
}

bool Requested() {
	return g_generate > 0;
}

std::string SupportProblem(vk::PhysicalDevice physical_device) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (g_upscale) {
		sl::AdapterInfo adapter {};
		adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
		const auto result        = g_api.is_feature_supported(sl::kFeatureDLSS, adapter);
		std::printf("DLSS super resolution: %s\n", result == sl::Result::eOk
		                                               ? "Streamline offers it with this GPU and driver"
		                                               : (std::string("not offered here (") + sl::getResultAsStr(result) +
		                                                  "; an NVIDIA RTX GPU and a recent driver)")
		                                                     .c_str());
		std::fflush(stdout);
	}
#endif
	if (g_generate == 0) {
		return {};
	}
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// (Streamline answers before the device too: its programming guide asks while the adapters are enumerated. Issue
	// #45: a device refused with no word why; the driver's own refusal says nothing either.)
	sl::AdapterInfo adapter {};
	adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
	const auto  result       = g_api.is_feature_supported(sl::kFeatureDLSS_G, adapter);
	std::string problem;
	switch (result) {
		case sl::Result::eOk: break;
		case sl::Result::eErrorOSDisabledHWS:
			problem = "Windows' hardware-accelerated GPU scheduling is off (Settings > System > Display > Graphics > "
			          "Advanced graphics settings; then restart the PC)";
			break;
		case sl::Result::eErrorOSOutOfDate: problem = "this Windows version is too old for DLSS frame generation"; break;
		case sl::Result::eErrorDriverOutOfDate: problem = "the NVIDIA driver is too old for DLSS frame generation"; break;
		case sl::Result::eErrorNoSupportedAdapterFound:
		case sl::Result::eErrorAdapterNotSupported:
			problem = "this graphics card has no DLSS frame generation (NVIDIA RTX 40 or newer)";
			break;
		default: problem = std::string("Streamline does not offer DLSS frame generation here (") + sl::getResultAsStr(result) + ")"; break;
	}
	if (problem.empty()) {
		LOGF("frame generation: Streamline offers DLSS-G with this GPU, driver and Windows\n");
		std::printf("Frame generation: Streamline offers DLSS-G with this GPU, driver and Windows\n");
	} else {
		LOGF("frame generation: %s\n", problem.c_str());
		std::printf("Frame generation: %s\n", problem.c_str());
	}
	// (Where the driver's reservations go, for a refused device's log: RefusedDeviceAdvice.)
	PROCESS_MITIGATION_ASLR_POLICY aslr {};
	if (GetProcessMitigationPolicy(GetCurrentProcess(), ProcessASLRPolicy, &aslr, sizeof(aslr))) {
		std::printf("Frame generation: bottom-up ASLR %s for the emulator\n", aslr.EnableBottomUpRandomization != 0 ? "on" : "off");
	}
	std::fflush(stdout);
	return problem;
#else
	(void)physical_device;
	return {};
#endif
}

std::string RefusedDeviceAdvice() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// (The emulator is built with fixed addresses, so without bottom-up ASLR its host allocations pack upward from the
	// guest's ceiling; an RTX 5080's driver refused the device until Exploit protection forced it on for the emulator.)
	PROCESS_MITIGATION_ASLR_POLICY aslr {};
	if (GetProcessMitigationPolicy(GetCurrentProcess(), ProcessASLRPolicy, &aslr, sizeof(aslr)) &&
	    aslr.EnableBottomUpRandomization == 0) {
		return " If it keeps failing, in Windows Security: App & browser control > Exploit protection > Program settings > "
		       "add kyty_emulator.exe > Randomize memory allocations (Bottom-up ASLR): Override system settings, On "
		       "(issue #45).";
	}
#endif
	return {};
}

bool WantsCamera() {
	return g_ready && !g_camera.valid;
}

void BypassCommandHooks(vk::Device device) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// KYTY_FRAMEGEN_HOOKS=1 keeps Streamline's wrappers (for comparison).
	if ((g_generate == 0 && !g_upscale) || std::getenv("KYTY_FRAMEGEN_HOOKS") != nullptr) {
		return;
	}
	// The real loader: Streamline itself forwards to it.
	HMODULE loader = GetModuleHandleW(L"vulkan-1.dll");
	auto*   get_device_proc = loader != nullptr ? reinterpret_cast<PFN_vkGetDeviceProcAddr>(
	                                                reinterpret_cast<void*>(GetProcAddress(loader, "vkGetDeviceProcAddr")))
	                                          : nullptr;
	if (get_device_proc == nullptr) {
		return;
	}
	auto&      dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto native     = static_cast<VkDevice>(device);
	uint32_t   replaced   = 0;
	const auto direct     = [&](auto& function, const char* name) {
        if (auto* pointer = get_device_proc(native, name); pointer != nullptr) {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(pointer);
            ++replaced;
        }
	};
	direct(dispatcher.vkBeginCommandBuffer, "vkBeginCommandBuffer");
	direct(dispatcher.vkCmdBindPipeline, "vkCmdBindPipeline");
	direct(dispatcher.vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
	direct(dispatcher.vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
	std::printf("Frame generation: %u command-buffer calls bypass Streamline\n", replaced);
#else
	(void)device;
#endif
}

void Initialize() {
	const char* value = std::getenv("KYTY_FRAMEGEN");
	g_generate        = value != nullptr ? std::clamp(std::atoi(value), 0, 5) : 0;
	const char* upscaler = std::getenv("KYTY_UPSCALER");
	g_upscale            = upscaler != nullptr && std::string_view(upscaler) == "dlss";
	if (g_generate == 0 && !g_upscale) {
		return;
	}
	// (Every way out without Streamline turns both off.)
	struct Off {
		bool done = false;
		~Off() {
			if (!done) {
				g_generate = 0;
				g_upscale  = false;
			}
		}
	} off;
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// NVIDIA's Streamline runtime, which no release ships (the user's own copy: the Streamline SDK's bin\x64):
	// KYTY_STREAMLINE_DIR (the launcher's "Streamline..." folder, run-windows -StreamlineDir), else a "streamline"
	// folder next to the emulator, else the SDK unpacked beside the build (_Build\deps\streamline\sdk). A folder
	// may be the SDK's own (its bin\x64 is taken).
	std::vector<std::filesystem::path> candidates;
	wchar_t                            exe[MAX_PATH] {};
	GetModuleFileNameW(nullptr, exe, MAX_PATH);
	const auto exe_dir = std::filesystem::path(exe).parent_path();
	// (Wide: a folder's name in any language.)
	if (const wchar_t* env = _wgetenv(L"KYTY_STREAMLINE_DIR"); env != nullptr && *env != L'\0') {
		candidates.emplace_back(env);
	} else {
		candidates.push_back(exe_dir / "streamline");
		candidates.push_back(exe_dir.parent_path() / "deps" / "streamline" / "sdk");
	}
	std::filesystem::path dir;
	for (const auto& candidate: candidates) {
		for (const auto& folder: {candidate, candidate / "bin" / "x64"}) {
			std::error_code error;
			if (dir.empty() && std::filesystem::exists(folder / "sl.interposer.dll", error)) dir = folder;
		}
	}
	if (dir.empty()) {
		LOGF("frame generation: no sl.interposer.dll (KYTY_STREAMLINE_DIR or a streamline folder next to the emulator)\n");
		std::printf("Frame generation: NVIDIA's Streamline runtime was not found (choose its folder in the launcher)\n");
		return;
	}
	const auto interposer = dir / "sl.interposer.dll";
	// Only NVIDIA's own (Streamline then checks its plugins' signatures itself).
	if (!LocalPlatform::IsNvidiaSigned(interposer.wstring())) {
		LOGF("frame generation: %s is not signed by NVIDIA: not loaded\n", interposer.string().c_str());
		std::printf("Frame generation: %s is not signed by NVIDIA, not loaded\n", interposer.string().c_str());
		return;
	}
	if (OpenDlssgDetected()) {
		LOGF("frame generation: OpenDLSSG Enabler detected (version.dll/opendlssg.ini)\n");
		std::printf("Frame generation: OpenDLSSG Enabler detected (version.dll/opendlssg.ini)\n");
	}
	if (std::getenv("KYTY_FRAMEGEN_VERBOSE") != nullptr) {
		// Graphics modules already loaded before Streamline (it must come first).
		for (const char* name: {"vulkan-1.dll", "nvoglv64.dll", "opengl32.dll", "dxgi.dll", "d3d11.dll", "d3d12.dll",
		                        "d3d9.dll", "nvapi64.dll", "nvcuda.dll", "nvofapi64.dll", "sl.interposer.dll"}) {
			if (GetModuleHandleA(name) != nullptr) std::printf("Frame generation: %s already loaded\n", name);
		}
	}
	HMODULE    module     = LoadLibraryW(interposer.c_str());
	if (module == nullptr) {
		LOGF("frame generation: cannot load %s\n", interposer.string().c_str());
		std::printf("Frame generation: cannot load %s\n", interposer.string().c_str());
		return;
	}
	const auto load = [&](auto& function, const char* name) {
		function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
		    reinterpret_cast<void*>(GetProcAddress(module, name)));
		return function != nullptr;
	};
	if (!load(g_api.init, "slInit") || !load(g_api.is_feature_supported, "slIsFeatureSupported") ||
	    !load(g_api.is_feature_loaded, "slIsFeatureLoaded") ||
	    !load(g_api.get_feature_function, "slGetFeatureFunction") ||
	    !load(g_api.get_new_frame_token, "slGetNewFrameToken") ||
	    !load(g_api.set_constants, "slSetConstants") ||
	    !load(g_api.set_tag_for_frame, "slSetTagForFrame") ||
	    !load(g_api.evaluate_feature, "slEvaluateFeature")) {
		LOGF("frame generation: sl.interposer.dll lacks an entry point\n");
		return;
	}
	static const std::wstring plugin_dir = dir.wstring();
	static const std::wstring log_dir    = (dir.parent_path() / "logs").wstring();
	static const wchar_t*     plugin_paths[] {plugin_dir.c_str()};
	// KYTY_FRAMEGEN_FEATURES=dlssg,reflex,pcl: the plugins to load (a diagnostic subset).
	static std::vector<sl::Feature> features;
	if (g_generate > 0) features = {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
	if (const char* list = std::getenv("KYTY_FRAMEGEN_FEATURES"); list != nullptr && g_generate > 0) {
		const std::string text = list;
		features.clear();
		if (text.find("dlssg") != std::string::npos) features.push_back(sl::kFeatureDLSS_G);
		if (text.find("reflex") != std::string::npos) features.push_back(sl::kFeatureReflex);
		if (text.find("pcl") != std::string::npos) features.push_back(sl::kFeaturePCL);
	}
	if (g_upscale) features.push_back(sl::kFeatureDLSS);
	sl::Preferences           preferences {};
	preferences.showConsole        = false;
	preferences.logLevel           = std::getenv("KYTY_FRAMEGEN_VERBOSE") ? sl::LogLevel::eVerbose
	                                                                      : sl::LogLevel::eDefault;
	preferences.pathsToPlugins     = plugin_paths;
	preferences.numPathsToPlugins  = 1;
	preferences.pathToLogsAndData  = nullptr;
	preferences.logMessageCallback = LogMessage;
	preferences.flags              = sl::PreferenceFlags::eDisableCLStateTracking |
	                    sl::PreferenceFlags::eUseFrameBasedResourceTagging;
	preferences.featuresToLoad    = features.data();
	preferences.numFeaturesToLoad = static_cast<uint32_t>(features.size());
	preferences.engine            = sl::EngineType::eCustom;
	preferences.engineVersion     = "1.0";
	preferences.projectId         = "6a3f1d5e-4b7c-4f2a-9d8e-1c2b3a4d5e6f";
	preferences.renderAPI         = sl::RenderAPI::eVulkan;
	if (const auto result = g_api.init(preferences, sl::kSDKVersion); result != sl::Result::eOk) {
		LOGF("frame generation: slInit failed (%d)\n", static_cast<int>(result));
		std::printf("Frame generation: slInit failed (%d)\n", static_cast<int>(result));
		return;
	}
	// SDL loads the interposer as the Vulkan loader: every instance/device/swapchain call then
	// goes through Streamline, which adds what DLSS-G needs.
	SetEnvironmentVariableW(L"SDL_VULKAN_LIBRARY", interposer.c_str());
	_putenv_s("SDL_VULKAN_LIBRARY", interposer.string().c_str());
	LOGF("frame generation: Streamline loaded from %s, %d generated frame(s), DLSS super resolution %s\n",
	     dir.string().c_str(), g_generate, g_upscale ? "asked for" : "off");
	off.done = true;
#else
	LOGF("frame generation: this build has no Streamline support\n");
	std::printf("DLSS: this build has no Streamline support\n");
#endif
}

void OnDevice(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device) {
	(void)instance;
	(void)device;
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (g_upscale) {
		sl::AdapterInfo adapter {};
		adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
		bool loaded              = false;
		if (g_api.is_feature_supported(sl::kFeatureDLSS, adapter) == sl::Result::eOk &&
		    g_api.is_feature_loaded(sl::kFeatureDLSS, loaded) == sl::Result::eOk && loaded &&
		    FeatureFunction(sl::kFeatureDLSS, "slDLSSSetOptions", g_api.dlss_set_options) &&
		    FeatureFunction(sl::kFeatureDLSS, "slDLSSGetOptimalSettings", g_api.dlss_get_optimal_settings)) {
			g_upscale_ready = true;
			std::printf("DLSS super resolution: ready\n");
		} else {
			std::printf("DLSS super resolution: not available on this device (the game's image is upscaled without it)\n");
		}
		std::fflush(stdout);
	}
#endif
	if (g_generate == 0) {
		return;
	}
#ifdef KYTY_FRAMEGEN_STREAMLINE
	sl::AdapterInfo adapter {};
	adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
	bool loaded              = false;
	if (g_api.is_feature_supported(sl::kFeatureDLSS_G, adapter) != sl::Result::eOk ||
	    g_api.is_feature_loaded(sl::kFeatureDLSS_G, loaded) != sl::Result::eOk || !loaded) {
		LOGF("frame generation: DLSS-G is not supported on this device\n");
		std::printf("Frame generation: DLSS-G is not supported on this device\n");
		return;
	}
	if (!FeatureFunction(sl::kFeatureDLSS_G, "slDLSSGSetOptions", g_api.dlssg_set_options) ||
	    !FeatureFunction(sl::kFeatureDLSS_G, "slDLSSGGetState", g_api.dlssg_get_state) ||
	    !FeatureFunction(sl::kFeatureReflex, "slReflexSetOptions", g_api.reflex_set_options) ||
	    !FeatureFunction(sl::kFeaturePCL, "slPCLSetMarker", g_api.pcl_set_marker)) {
		return;
	}
	// (Required with DLSS-G, whatever the Reflex mode: Streamline's DLSS-G and Reflex guides.)
	(void)FeatureFunction(sl::kFeatureReflex, "slReflexSleep", g_api.reflex_sleep);
	sl::ReflexOptions reflex {};
	reflex.mode = sl::ReflexMode::eLowLatency;
	if (const auto result = g_api.reflex_set_options(reflex); result != sl::Result::eOk) {
		LOGF("frame generation: slReflexSetOptions failed (%d)\n", static_cast<int>(result));
	}
	g_ready = true;
	LOGF("frame generation: DLSS-G ready\n");
	std::printf("Frame generation: DLSS-G ready (%d generated frame(s) per frame)\n", g_generate);
	std::fflush(stdout);
#endif
}

void BeforePresent(const FrameInputs* inputs, vk::Extent2D backbuffer, vk::Rect2D region,
                   vk::Format format, uint32_t image_count) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_ready) {
		return;
	}
	const sl::ViewportHandle viewport {0};
	const bool               usable = inputs != nullptr && inputs->valid;
	g_run_with    = usable ? g_run_with + 1 : 0;
	g_run_without = usable ? 0 : g_run_without + 1;
	const bool mode_on = g_mode_on ? g_run_without < TurnOffAfter : g_run_with >= TurnOnAfter;
	// DLSS-G options: on while frames carry inputs; sizes are hints for its allocations.
	if (!g_options_set || g_options_format != format || mode_on != g_mode_on ||
	    (usable && (g_options_backbuffer != backbuffer || g_options_render != inputs->render_extent))) {
		sl::DLSSGOptions options {};
		options.mode                = mode_on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
		options.flags               = sl::DLSSGFlags::eRetainResourcesWhenOff;
		options.numFramesToGenerate = static_cast<uint32_t>(g_generate);
		options.numBackBuffers      = image_count;
		options.colorWidth          = backbuffer.width;
		options.colorHeight         = backbuffer.height;
		options.colorBufferFormat   = static_cast<uint32_t>(format);
		if (usable) {
			options.mvecDepthWidth    = inputs->render_extent.width;
			options.mvecDepthHeight   = inputs->render_extent.height;
			options.mvecBufferFormat  = static_cast<uint32_t>(inputs->motion.format);
			options.depthBufferFormat = static_cast<uint32_t>(inputs->depth.format);
			g_options_render          = inputs->render_extent;
		}
		if (const auto result = g_api.dlssg_set_options(viewport, options); result != sl::Result::eOk) {
			LOGF("frame generation: slDLSSGSetOptions failed (%d)\n", static_cast<int>(result));
		}
		g_options_set        = true;
		g_options_backbuffer = backbuffer;
		g_options_format     = format;
		if (mode_on != g_mode_on) {
			LOGF("frame generation: DLSS-G %s\n", mode_on ? "on" : "off (no inputs)");
			g_mode_on = mode_on;
		}
	}

	++g_frame_index;
	if (g_api.get_new_frame_token(g_token, &g_frame_index) != sl::Result::eOk || g_token == nullptr) {
		g_token = nullptr;
		return;
	}
	auto& token = *g_token;
	// The frame's Reflex sleep, then its markers (all at its present: the game's frame was made before).
	if (g_api.reflex_sleep != nullptr && kyty_local_framegen_sleep.load(std::memory_order_relaxed) != 0)
		g_api.reflex_sleep(token);
	g_api.pcl_set_marker(sl::PCLMarker::eSimulationStart, token);
	g_api.pcl_set_marker(sl::PCLMarker::eSimulationEnd, token);
	g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitStart, token);

	sl::Constants constants {};
	const auto    camera = usable ? Decompose(inputs->camera.view_proj) : CameraFrame {};
	Mat4          identity;
	for (uint32_t i = 0; i < 4; ++i) {
		identity[i] = sl::float4(i == 0, i == 1, i == 2, i == 3);
	}
	constants.cameraViewToClip = identity;
	constants.clipToCameraView = identity;
	constants.clipToLensClip   = identity;
	constants.clipToPrevClip   = identity;
	constants.prevClipToClip   = identity;
	constants.jitterOffset     = sl::float2(0.0f, 0.0f);
	constants.mvecScale        = sl::float2(1.0f, 1.0f);
	constants.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
	constants.cameraPos        = sl::float3(0, 0, 0);
	constants.cameraUp         = sl::float3(0, 1, 0);
	constants.cameraRight      = sl::float3(1, 0, 0);
	constants.cameraFwd        = sl::float3(0, 0, 1);
	constants.cameraNear       = 0.1f;
	constants.cameraFar        = 10000.0f;
	constants.cameraFOV        = 1.0f;
	constants.cameraAspectRatio = 16.0f / 9.0f;
	constants.motionVectorsInvalidValue = sl::INVALID_FLOAT;
	constants.depthInverted       = sl::Boolean::eTrue;
	constants.cameraMotionIncluded = sl::Boolean::eTrue;
	constants.motionVectors3D     = sl::Boolean::eFalse;
	// (A frame left unshown before this one: its motion and matrices are to the game's previous frame, not to the one
	// presented last.)
	const bool discontinuity      = g_discontinuity.exchange(false, std::memory_order_relaxed);
	constants.reset               = usable && !discontinuity ? sl::Boolean::eFalse : sl::Boolean::eTrue;
	constants.orthographicProjection = sl::Boolean::eFalse;
	constants.motionVectorsDilated   = sl::Boolean::eFalse;
	constants.motionVectorsJittered  = sl::Boolean::eFalse;
	if (usable && camera.valid) {
		// Projection in view space: x' = sx*x + cx*z, y' = sy*y + cy*z, z' = near, w' = z.
		Mat4 projection;
		projection[0] = sl::float4(camera.sx, 0, 0, 0);
		projection[1] = sl::float4(0, camera.sy, 0, 0);
		projection[2] = sl::float4(camera.cx, camera.cy, 0, 1);
		projection[3] = sl::float4(0, 0, camera.z_near, 0);
		constants.cameraViewToClip = projection;
		Inverted(projection, constants.clipToCameraView);
		const auto current  = Transposed(inputs->camera.view_proj);
		const auto previous = Transposed(inputs->camera.prev_view_proj);
		Mat4       clip_to_world;
		if (Inverted(current, clip_to_world)) {
			constants.clipToPrevClip = Multiply(clip_to_world, previous);
			Inverted(constants.clipToPrevClip, constants.prevClipToClip);
		}
		const float width  = static_cast<float>(inputs->render_extent.width);
		const float height = static_cast<float>(inputs->render_extent.height);
		constants.jitterOffset = sl::float2(inputs->camera.jitter_ndc[0] * width * 0.5f,
		                                    -inputs->camera.jitter_ndc[1] * height * 0.5f);
		static const bool pixels_mv = [] {
			const char* mv = std::getenv("KYTY_FRAMEGEN_MV");
			return mv != nullptr && std::string(mv) == "px";
		}();
		// The game's vectors point along the motion, in UV units: its TAA reads the history at
		// uv - mv. DLSS wants them towards the previous position (uv + mv = previous uv), so they
		// are negated; with them as they were, generated frames moved the scene the wrong way
		// when the camera turned. KYTY_FRAMEGEN_MV_SIGN=1 keeps the game's sign (a comparison).
		static const float mv_sign = [] {
			const char* sign = std::getenv("KYTY_FRAMEGEN_MV_SIGN");
			return sign != nullptr && std::atof(sign) > 0 ? 1.0f : -1.0f;
		}();
		constants.mvecScale = pixels_mv ? sl::float2(mv_sign / width, mv_sign / height) : sl::float2(mv_sign, mv_sign);
		constants.cameraPos   = camera.position;
		constants.cameraUp    = camera.up;
		constants.cameraRight = camera.right;
		constants.cameraFwd   = camera.forward;
		constants.cameraNear  = camera.z_near;
		constants.cameraFar   = 100000.0f;
		constants.cameraFOV   = 2.0f * std::atan(1.0f / std::fabs(camera.sy));
		constants.cameraAspectRatio = std::fabs(camera.sy / camera.sx);
	}
	if (const auto result = g_api.set_constants(constants, token, viewport); result != sl::Result::eOk &&
	                                                                          g_presents < 8) {
		LOGF("frame generation: slSetConstants failed (%d)\n", static_cast<int>(result));
	}

	if (usable) {
		sl::Resource depth(sl::ResourceType::eTex2d, static_cast<VkImage>(inputs->depth.image), nullptr,
		                   static_cast<VkImageView>(inputs->depth_view),
		                   static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
		depth.width        = inputs->depth.extent.width;
		depth.height       = inputs->depth.extent.height;
		depth.nativeFormat = static_cast<uint32_t>(inputs->depth.format);
		depth.mipLevels    = 1;
		depth.arrayLayers  = 1;
		depth.flags        = 0;
		depth.usage        = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(inputs->depth.usage));
		sl::Resource motion(sl::ResourceType::eTex2d, static_cast<VkImage>(inputs->motion.image), nullptr,
		                    static_cast<VkImageView>(inputs->motion_view),
		                    static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
		motion.width        = inputs->motion.extent.width;
		motion.height       = inputs->motion.extent.height;
		motion.nativeFormat = static_cast<uint32_t>(inputs->motion.format);
		motion.mipLevels    = 1;
		motion.arrayLayers  = 1;
		motion.flags        = 0;
		motion.usage = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(inputs->motion.usage));
		const sl::Extent full {0, 0, inputs->render_extent.width, inputs->render_extent.height};
		const sl::Extent game {static_cast<uint32_t>(region.offset.y), static_cast<uint32_t>(region.offset.x),
		                       region.extent.width, region.extent.height};
		// The backbuffer is tagged only when the game image covers part of it (KYTY_PRESENT_ASPECT).
		const bool      partial = region.offset.x != 0 || region.offset.y != 0 ||
		                     region.extent.width != backbuffer.width || region.extent.height != backbuffer.height;
		sl::ResourceTag tags[] {
		    sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &full),
		    sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent,
		                    &full),
		    sl::ResourceTag(nullptr, sl::kBufferTypeBackbuffer, sl::ResourceLifecycle::eValidUntilPresent,
		                    &game),
		};
		if (const auto result = g_api.set_tag_for_frame(token, viewport, tags, partial ? 3u : 2u, nullptr);
		    result != sl::Result::eOk && g_presents < 8) {
			LOGF("frame generation: slSetTagForFrame failed (%d)\n", static_cast<int>(result));
		}
	} else {
		// Menus, movies, loading: no valid depth and motion (DLSS-G then presents the frame as is).
		sl::ResourceTag tags[] {
		    sl::ResourceTag(nullptr, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent),
		    sl::ResourceTag(nullptr, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent),
		};
		g_api.set_tag_for_frame(token, viewport, tags, static_cast<uint32_t>(std::size(tags)), nullptr);
	}
	g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitEnd, token);
	g_api.pcl_set_marker(sl::PCLMarker::ePresentStart, token);
#else
	(void)inputs;
	(void)backbuffer;
	(void)region;
	(void)format;
	(void)image_count;
#endif
}

void AfterPresent() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_ready || g_token == nullptr) {
		return;
	}
	g_api.pcl_set_marker(sl::PCLMarker::ePresentEnd, *g_token);
	const auto presents = ++g_presents;
	if (presents % 30 != 1) {
		return;
	}
	sl::DLSSGState state {};
	if (g_api.dlssg_get_state(sl::ViewportHandle {0}, state, nullptr) != sl::Result::eOk) {
		return;
	}
	g_presented_frames.store(std::max(state.numFramesActuallyPresented, 1u), std::memory_order_relaxed);
	// More generated frames asked for than the GPU makes (the launcher's 3x and up): as many as it makes.
	if (state.numFramesToGenerateMax != 0 && static_cast<uint32_t>(g_generate) > state.numFramesToGenerateMax) {
		LOGF("frame generation: %d generated frames asked for, the GPU makes up to %u\n", g_generate,
		     state.numFramesToGenerateMax);
		std::printf("Frame generation: %d generated frame(s) asked for, the GPU makes up to %u\n", g_generate,
		            state.numFramesToGenerateMax);
		g_generate    = static_cast<int>(state.numFramesToGenerateMax);
		g_options_set = false;
	}
	if (presents % 600 == 1) {
		LOGF("frame generation: status=0x%x presented=%u max_generated=%u vram=%" PRIu64 " MiB\n",
		     static_cast<uint32_t>(state.status), state.numFramesActuallyPresented,
		     state.numFramesToGenerateMax, state.estimatedVRAMUsageInBytes >> 20u);
		std::printf("Frame generation: status=0x%x presented=%u max_generated=%u\n",
		            static_cast<uint32_t>(state.status), state.numFramesActuallyPresented,
		            state.numFramesToGenerateMax);
		std::fflush(stdout);
	}
#endif
}

void SwapchainChanging() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// Off before the swapchain is made again (DLSS-G guide: before a resize or a full screen change).
	if (!g_ready || !g_mode_on) return;
	sl::DLSSGOptions options {};
	options.mode                = sl::DLSSGMode::eOff;
	options.flags               = sl::DLSSGFlags::eRetainResourcesWhenOff;
	options.numFramesToGenerate = static_cast<uint32_t>(g_generate);
	if (const auto result = g_api.dlssg_set_options(sl::ViewportHandle {0}, options); result != sl::Result::eOk)
		LOGF("frame generation: slDLSSGSetOptions (off) failed (%d)\n", static_cast<int>(result));
	g_mode_on     = false;
	g_run_with    = 0;
	g_options_set = false; // (the next present sends its sizes again)
#endif
}

void NoteDroppedFrame() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	g_discontinuity.store(true, std::memory_order_relaxed);
#endif
	static std::atomic<uint32_t> dropped {0};
	if (const auto count = dropped.fetch_add(1, std::memory_order_relaxed) + 1; count <= 8 || count % 600 == 0)
		LOGF("frame generation: %u frame(s) with inputs left unshown (the next resets DLSS-G's history)\n", count);
}

uint32_t PresentedFrames() {
	return g_presented_frames.load(std::memory_order_relaxed);
}

void SetStaticRect(float x0, float y0, float x1, float y1) {
	const auto pack = [](float value, uint32_t index) {
		return static_cast<uint64_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 65535.0f)) << (16u * index);
	};
	g_static_rect.store(pack(x0, 0) | pack(y0, 1) | pack(x1, 2) | pack(y1, 3), std::memory_order_relaxed);
}

void OnDraw(uint64_t vs_hash, std::span<const uint32_t> srt) {
	if (g_camera.valid || srt.size() < SrtJitter + 2 || !IsGbufferVertexShader(vs_hash)) {
		return;
	}
	for (uint32_t i = 0; i < 16; ++i) {
		g_camera.view_proj[i] = AsFloat(srt[SrtViewProj + i]);
	}
	// The previous matrix has rows x, y, w; its z row is taken from the current one.
	for (uint32_t i = 0; i < 4; ++i) {
		g_camera.prev_view_proj[i]      = AsFloat(srt[SrtPrevViewProj + i]);
		g_camera.prev_view_proj[4 + i]  = AsFloat(srt[SrtPrevViewProj + 4 + i]);
		g_camera.prev_view_proj[8 + i]  = g_camera.view_proj[8 + i];
		g_camera.prev_view_proj[12 + i] = AsFloat(srt[SrtPrevViewProj + 8 + i]);
	}
	g_camera.jitter_ndc = {AsFloat(srt[SrtJitter]), AsFloat(srt[SrtJitter + 1])};
	g_camera.valid      = true;
}

void NoteScene() {
	g_scene.store(true, std::memory_order_relaxed);
}

bool TakeScene() {
	const bool scene = g_scene.exchange(false, std::memory_order_relaxed);
	g_last_scene.store(scene, std::memory_order_relaxed);
	g_flips_taken.fetch_add(1, std::memory_order_relaxed);
	return scene;
}

uint64_t GuestFlips() {
	return g_flips_taken.load(std::memory_order_relaxed);
}

bool SceneNoted() {
	return g_scene.load(std::memory_order_relaxed) || g_last_scene.load(std::memory_order_relaxed);
}

void OnDispatch(uint64_t cs_hash, ImageId depth, uint64_t depth_address, ImageId motion,
                uint64_t motion_address) {
	if (!IsTaaShader(cs_hash)) {
		return;
	}
	g_sources = {depth, motion, depth_address, motion_address, true};
}

void OnUpscale(ImageId depth, uint64_t depth_address, ImageId motion, uint64_t motion_address) {
	g_sources = {depth, motion, depth_address, motion_address, true};
}

void PrepareInputs(GraphicContext& graphics, TextureCache& cache, CommandBuffer& command,
                   FrameInputs& inputs) {
	inputs.valid = false;
	const bool ready = g_sources.valid && g_camera.valid;
	const auto sources = g_sources;
	const auto camera  = g_camera;
	g_sources.valid    = false;
	g_camera.valid     = false;
	++g_guest_frame;
	if (!ready) {
		return;
	}
	auto& depth  = cache.GetImage(sources.depth);
	auto& motion = cache.GetImage(sources.motion);
	if (depth.info.data.address != sources.depth_address ||
	    motion.info.data.address != sources.motion_address ||
	    depth.backing.extent.width != motion.backing.extent.width ||
	    depth.backing.extent.height != motion.backing.extent.height) {
		return; // the images were replaced after the TAA pass
	}
	Configure(graphics, inputs.depth, inputs.depth_view, depth.backing);
	Configure(graphics, inputs.motion, inputs.motion_view, motion.backing, &inputs.motion_attachment);
	command.EndRendering();
	auto vk_command = command.Handle();
	CopyInto(vk_command, depth, inputs.depth);
	CopyInto(vk_command, motion, inputs.motion);
	if (inputs.motion_attachment) {
		ZeroStaticRect(vk_command, inputs.motion, inputs.motion_view);
	}
	inputs.render_extent = {motion.backing.extent.width, motion.backing.extent.height};
	inputs.camera        = camera;
	inputs.guest_frame   = g_guest_frame;
	inputs.valid         = true;
}

bool UpscalerReady() {
	return g_upscale_ready;
}

bool Upscale(vk::CommandBuffer command, const UpscaleInputs& inputs) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_upscale_ready) return false;
	// Its own viewport: DLSS-G's (0) takes its constants and tags at the present.
	const sl::ViewportHandle viewport {1};
	static vk::Extent2D options_render {}, options_output {};
	if (options_render != inputs.color.extent || options_output != inputs.output.extent) {
		// The mode by the input's share of the output's height (DLSS takes any input size; the mode picks its model and
		// limits): PS5 Pro's 1512 of 2160 and 1440 are quality, 1080 performance, the output's own DLAA.
		const double share = static_cast<double>(inputs.color.extent.height) / inputs.output.extent.height;
		sl::DLSSOptions options {};
		options.mode = share >= 0.95   ? sl::DLSSMode::eDLAA
		               : share >= 0.62 ? sl::DLSSMode::eMaxQuality
		               : share >= 0.55 ? sl::DLSSMode::eBalanced
		               : share >= 0.45 ? sl::DLSSMode::eMaxPerformance
		                               : sl::DLSSMode::eUltraPerformance;
		options.outputWidth     = inputs.output.extent.width;
		options.outputHeight    = inputs.output.extent.height;
		options.colorBuffersHDR = sl::Boolean::eTrue; // (the game's linear HDR scene, before its tone mapping)
		options.useAutoExposure = sl::Boolean::eTrue;
		// KYTY_DLSS_PRESET=<letter> (a test): that DLSS model for every mode instead of the mode's default.
		if (const char* preset = std::getenv("KYTY_DLSS_PRESET"); preset != nullptr && *preset >= 'A' && *preset <= 'O') {
			const auto value = static_cast<sl::DLSSPreset>(*preset - 'A' + 1);
			options.dlaaPreset = options.qualityPreset = options.balancedPreset = options.performancePreset =
			    options.ultraPerformancePreset = options.ultraQualityPreset = value;
		}
		if (const auto result = g_api.dlss_set_options(viewport, options); result != sl::Result::eOk) {
			std::printf("DLSS super resolution: slDLSSSetOptions failed (%s)\n", sl::getResultAsStr(result));
			std::fflush(stdout);
			return false;
		}
		sl::DLSSOptimalSettings settings {};
		(void)g_api.dlss_get_optimal_settings(options, settings);
		std::printf("DLSS super resolution: %ux%u -> %ux%u, mode %u (its range %ux%u-%ux%u)\n", inputs.color.extent.width,
		            inputs.color.extent.height, inputs.output.extent.width, inputs.output.extent.height,
		            static_cast<uint32_t>(options.mode), settings.renderWidthMin, settings.renderHeightMin,
		            settings.renderWidthMax, settings.renderHeightMax);
		std::fflush(stdout);
		options_render = inputs.color.extent;
		options_output = inputs.output.extent;
	}
	static uint32_t frame = 0;
	++frame;
	sl::FrameToken* token = nullptr;
	if (g_api.get_new_frame_token(token, &frame) != sl::Result::eOk || token == nullptr) return false;

	sl::Constants constants {};
	Mat4          identity;
	for (uint32_t i = 0; i < 4; ++i) {
		identity[i] = sl::float4(i == 0, i == 1, i == 2, i == 3);
	}
	Mat4 view_to_clip;
	for (uint32_t i = 0; i < 4; ++i) {
		view_to_clip[i] = sl::float4(inputs.view_to_clip[i * 4], inputs.view_to_clip[i * 4 + 1], inputs.view_to_clip[i * 4 + 2],
		                             inputs.view_to_clip[i * 4 + 3]);
	}
	constants.cameraViewToClip = view_to_clip;
	if (!Inverted(view_to_clip, constants.clipToCameraView)) constants.clipToCameraView = identity;
	constants.clipToLensClip      = identity;
	constants.clipToPrevClip      = identity;
	constants.prevClipToClip      = identity;
	constants.jitterOffset        = sl::float2(inputs.jitter_x, inputs.jitter_y);
	// (Streamline's motion vectors are in render-size UV units: its scale makes them so from the game's pixels.)
	constants.mvecScale = sl::float2(inputs.motion_scale_x / static_cast<float>(inputs.color.extent.width),
	                                 inputs.motion_scale_y / static_cast<float>(inputs.color.extent.height));
	constants.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
	constants.cameraPos           = sl::float3(0, 0, 0);
	constants.cameraUp            = sl::float3(0, 1, 0);
	constants.cameraRight         = sl::float3(1, 0, 0);
	constants.cameraFwd           = sl::float3(0, 0, 1);
	constants.cameraNear          = inputs.near_plane;
	constants.cameraFar           = inputs.far_plane;
	constants.cameraFOV           = inputs.view_to_clip[5] != 0.0f ? 2.0f * std::atan(1.0f / std::fabs(inputs.view_to_clip[5])) : 1.0f;
	constants.cameraAspectRatio   = static_cast<float>(inputs.color.extent.width) / static_cast<float>(inputs.color.extent.height);
	constants.motionVectorsInvalidValue = sl::INVALID_FLOAT;
	constants.depthInverted             = sl::Boolean::eTrue;
	constants.cameraMotionIncluded      = sl::Boolean::eTrue;
	constants.motionVectors3D           = sl::Boolean::eFalse;
	constants.reset                     = inputs.reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
	constants.orthographicProjection    = sl::Boolean::eFalse;
	constants.motionVectorsDilated      = sl::Boolean::eFalse;
	constants.motionVectorsJittered     = sl::Boolean::eFalse;
	if (const auto result = g_api.set_constants(constants, *token, viewport); result != sl::Result::eOk) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 8)
			std::printf("DLSS super resolution: slSetConstants failed (%s)\n", sl::getResultAsStr(result));
		return false;
	}
	const auto resource = [](const UpscaleImage& image) {
		sl::Resource r(sl::ResourceType::eTex2d, static_cast<VkImage>(image.image), nullptr, static_cast<VkImageView>(image.view),
		               static_cast<uint32_t>(static_cast<VkImageLayout>(image.layout)));
		r.width        = image.extent.width;
		r.height       = image.extent.height;
		r.nativeFormat = static_cast<uint32_t>(static_cast<VkFormat>(image.format));
		r.mipLevels    = 1;
		r.arrayLayers  = 1;
		r.flags        = 0;
		r.usage        = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image.usage));
		return r;
	};
	auto             color  = resource(inputs.color);
	auto             depth  = resource(inputs.depth);
	auto             motion = resource(inputs.motion);
	auto             output = resource(inputs.output);
	// (KYTY_SR_LOG: the first two evaluations' images and constants.)
	static const bool describe = std::getenv("KYTY_SR_LOG") != nullptr;
	if (static std::atomic<uint32_t> described {0}; describe && described.fetch_add(1, std::memory_order_relaxed) < 2) {
		const std::pair<const char*, const UpscaleImage*> images[] {
		    {"color", &inputs.color}, {"depth", &inputs.depth}, {"motion", &inputs.motion}, {"output", &inputs.output}};
		for (const auto& [name, image]: images) {
			std::printf("DLSS super resolution: %s image %p view %p layout %d format %d %ux%u usage 0x%x\n", name,
			            static_cast<void*>(static_cast<VkImage>(image->image)),
			            static_cast<void*>(static_cast<VkImageView>(image->view)), static_cast<int>(image->layout),
			            static_cast<int>(image->format), image->extent.width, image->extent.height,
			            static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image->usage)));
		}
		std::printf("DLSS super resolution: jitter %g %g, motion scale %g %g, reset %d, near %g far %g\n", inputs.jitter_x,
		            inputs.jitter_y, inputs.motion_scale_x, inputs.motion_scale_y, inputs.reset ? 1 : 0, inputs.near_plane,
		            inputs.far_plane);
		std::fflush(stdout);
	}
	const sl::Extent render {0, 0, inputs.color.extent.width, inputs.color.extent.height};
	const sl::Extent full {0, 0, inputs.output.extent.width, inputs.output.extent.height};
	const sl::ResourceTag tags[] {
	    sl::ResourceTag(&color, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &render),
	    sl::ResourceTag(&output, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &full),
	    sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilEvaluate, &render),
	    sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilEvaluate, &render),
	};
	auto* command_buffer = reinterpret_cast<sl::CommandBuffer*>(static_cast<VkCommandBuffer>(command));
	if (const auto result = g_api.set_tag_for_frame(*token, viewport, tags, static_cast<uint32_t>(std::size(tags)), command_buffer);
	    result != sl::Result::eOk) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 8)
			std::printf("DLSS super resolution: slSetTagForFrame failed (%s)\n", sl::getResultAsStr(result));
		return false;
	}
	const sl::BaseStructure* evaluate_inputs[] {&viewport};
	if (const auto result = g_api.evaluate_feature(sl::kFeatureDLSS, *token, evaluate_inputs, 1, command_buffer);
	    result != sl::Result::eOk) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 8)
			std::printf("DLSS super resolution: slEvaluateFeature failed (%s)\n", sl::getResultAsStr(result));
		return false;
	}
	return true;
#else
	(void)command;
	(void)inputs;
	return false;
#endif
}

void DestroyInputs(GraphicContext& graphics, FrameInputs& inputs) {
	for (auto* view: {&inputs.depth_view, &inputs.motion_view}) {
		if (*view != nullptr) {
			graphics.device.destroyImageView(*view, nullptr);
			*view = nullptr;
		}
	}
	for (auto* image: {&inputs.depth, &inputs.motion}) {
		if (image->image != nullptr) {
			graphics.DeleteImage(*image);
		}
	}
	inputs.valid = false;
}

} // namespace Libs::Graphics::FrameGen

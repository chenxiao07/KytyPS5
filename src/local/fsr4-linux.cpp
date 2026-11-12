#include "local/fsr4-linux.h"

#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>

// ffx-vk-linux's headers (and AMD's API headers it vendors) are found by CMake; without them this file is empty.
#if defined(__linux__) && defined(KYTY_FSR4_LINUX)
#define KYTY_FSR4_AVAILABLE 1
#else
#define KYTY_FSR4_AVAILABLE 0
#endif

#if KYTY_FSR4_AVAILABLE
#include <dlfcn.h>
#include <unistd.h>

#include <cfloat>
#include <cmath>
#include <vector>

// (AMD's headers use MSVC pragmas: clang skips them.)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#include "ffx_vk_linux.h"
#pragma clang diagnostic pop
#endif

namespace Libs::Graphics::Fsr4Linux {

bool Requested() {
#if KYTY_FSR4_AVAILABLE
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_FSR4");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	}();
	return requested;
#else
	// KYTY_FSR4 set in a build without ffx-vk-linux's headers (the release packages): nothing here can run FSR 4, and the
	// launcher offers it to every Radeon owner who has put the library and the DLLs beside the emulator. Say so once.
#if defined(__linux__)
	static const bool reported = [] {
		const char* value = std::getenv("KYTY_FSR4");
		if (value != nullptr && value[0] != '\0' && value[0] != '0') {
			std::printf("FSR 4: this build was made without ffx-vk-linux (build from source after get-fsr4-linux.sh); "
			            "FSR 3 is used\n");
			std::fflush(stdout);
		}
		return true;
	}();
	(void)reported;
#endif
	return false;
#endif
}

#if KYTY_FSR4_AVAILABLE
namespace {

void Report(const char* format, auto... args) {
	std::printf(format, args...);
	std::fflush(stdout);
}

std::filesystem::path ExecutableFolder() {
	char        path[4096] {};
	const auto  length = readlink("/proc/self/exe", path, sizeof(path) - 1);
	std::error_code error;
	if (length <= 0) return std::filesystem::current_path(error);
	return std::filesystem::path(std::string(path, static_cast<size_t>(length))).parent_path();
}

// The folders FSR 4's pieces are looked for in (KYTY_FSR4_DIR alone when it is set).
std::vector<std::filesystem::path> Folders() {
	if (const char* env = std::getenv("KYTY_FSR4_DIR"); env != nullptr && *env != 0) return {std::filesystem::path(env)};
	const auto exe = ExecutableFolder();
	return {exe, exe / "fsr4"};
}

using CreateFn  = ffxReturnCode_t (*)(ffxContext*, ffxCreateContextDescHeader*, const ffxAllocationCallbacks*);
using DestroyFn = ffxReturnCode_t (*)(ffxContext*, const ffxAllocationCallbacks*);
using QueryFn   = ffxReturnCode_t (*)(ffxContext*, ffxQueryDescHeader*);
using DispatchFn = ffxReturnCode_t (*)(ffxContext*, const ffxDispatchDescHeader*);
using LoadFn    = int (*)(const char*);
using ErrorFn   = const char* (*)();

// libffx_vk_linux.so, loaded once (the library keeps AMD's DLLs loaded for the process).
struct Library {
	CreateFn   create   = nullptr;
	DestroyFn  destroy  = nullptr;
	QueryFn    query    = nullptr;
	DispatchFn dispatch = nullptr;
	bool       ready    = false;
};

void Message(uint32_t type, const wchar_t* text) {
	char buffer[1024] {};
	std::wcstombs(buffer, text, sizeof(buffer) - 1);
	if (type == FFX_API_MESSAGE_TYPE_ERROR || std::getenv("KYTY_FSR_DEBUG") != nullptr)
		Report("FSR 4 %s: %s\n", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", buffer);
}

const Library& TheLibrary() {
	static const Library library = [] {
		Library result;
		void*   module = nullptr;
		std::filesystem::path folder;
		for (const auto& candidate: Folders()) {
			std::error_code error;
			const auto      path = candidate / "libffx_vk_linux.so";
			if (!std::filesystem::exists(path, error)) continue;
			module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
			if (module == nullptr) {
				Report("FSR 4: cannot load %s (%s)\n", path.c_str(), dlerror());
				return result;
			}
			folder = candidate;
			break;
		}
		if (module == nullptr) {
			Report("FSR 4: libffx_vk_linux.so not found (KYTY_FSR4_DIR, next to the emulator, its fsr4 folder: run "
			       "get-fsr4-linux.sh); FSR 3 built in\n");
			return result;
		}
		const auto load = reinterpret_cast<LoadFn>(dlsym(module, "ffxvkLoad"));
		const auto error = reinterpret_cast<ErrorFn>(dlsym(module, "ffxvkError"));
		result.create    = reinterpret_cast<CreateFn>(dlsym(module, "ffxCreateContext"));
		result.destroy   = reinterpret_cast<DestroyFn>(dlsym(module, "ffxDestroyContext"));
		result.query     = reinterpret_cast<QueryFn>(dlsym(module, "ffxQuery"));
		result.dispatch  = reinterpret_cast<DispatchFn>(dlsym(module, "ffxDispatch"));
		if (load == nullptr || result.create == nullptr || result.destroy == nullptr || result.query == nullptr ||
		    result.dispatch == nullptr) {
			Report("FSR 4: libffx_vk_linux.so lacks an entry point (build it again with get-fsr4-linux.sh); FSR 3 built in\n");
			return result;
		}
		// AMD's DLLs: in the library's folder, else the other candidates'.
		std::filesystem::path dlls = folder;
		for (const auto& candidate: Folders()) {
			std::error_code ec;
			if (std::filesystem::exists(candidate / "amd_fidelityfx_loader_vk.dll", ec) ||
			    std::filesystem::exists(candidate / "amd_fidelityfx_upscaler_vk.dll", ec)) {
				dlls = candidate;
				break;
			}
		}
		if (load(dlls.c_str()) != 0) {
			Report("FSR 4: AMD's DLLs could not be loaded from %s: %s (amd_fidelityfx_loader_vk.dll and "
			       "amd_fidelityfx_upscaler_vk.dll from the FSR SDK 2.4's Kits/FidelityFX/signedbin, and amdvkffx64.dll "
			       "from AMD's Windows driver); FSR 3 built in\n",
			       dlls.c_str(), error != nullptr ? error() : "?");
			return result;
		}
		result.ready = true;
		Report("FSR 4: AMD's DLLs loaded from %s (ffx-vk-linux)\n", dlls.c_str());
		return result;
	}();
	return library;
}

// An upscaler input or output as AMD's runtime takes it: the image's own format (it makes its views: a depth image's
// with the depth aspect; the game's depth is a D32S8 image viewed as R32), the state its layout is in. (As
// frame-gen.cpp's FsrResource on Windows.)
FfxApiResource ToResource(const FrameGen::UpscaleImage& image, bool written) {
	VkImageCreateInfo info {};
	info.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	info.imageType   = VK_IMAGE_TYPE_2D;
	info.format      = static_cast<VkFormat>(image.image_format != vk::Format::eUndefined ? image.image_format : image.format);
	info.extent      = {image.extent.width, image.extent.height, 1};
	info.mipLevels   = 1;
	info.arrayLayers = 1;
	info.samples     = VK_SAMPLE_COUNT_1_BIT;
	info.usage       = static_cast<VkImageUsageFlags>(image.usage);
	const uint32_t state = written ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS
	                       : image.layout == vk::ImageLayout::eGeneral ? FFX_API_RESOURCE_STATE_COMMON
	                                                                   : FFX_API_RESOURCE_STATE_COMPUTE_READ;
	return ffxApiGetResourceVK(static_cast<VkImage>(image.image),
	                           ffxApiGetImageResourceDescriptionVK(static_cast<VkImage>(image.image), info,
	                                                               written ? FFX_API_RESOURCE_USAGE_UAV
	                                                                       : FFX_API_RESOURCE_USAGE_READ_ONLY),
	                           state);
}

struct Upscaler {
	std::mutex   mutex;
	ffxContext   context = nullptr;
	vk::Extent2D render {}, display {};
	bool         failed = false, reset = true;
	std::chrono::steady_clock::time_point last {};

	void Destroy(GraphicContext& graphics) {
		if (context == nullptr) return;
		DeviceQueuesGuard queues(graphics);
		(void)graphics.queue.waitIdle();
		(void)TheLibrary().destroy(&context, nullptr);
		context = nullptr;
	}

	bool Fail(const char* what, ffxReturnCode_t code) {
		failed = true;
		Report("Super resolution: FSR 4 %s failed (%u); FSR 3 built in from now on\n", what, static_cast<uint32_t>(code));
		return false;
	}

	// The id of the FSR 4 provider the SDK lists for this GPU (0: none: the DLLs do not take this GPU, or amdvkffx64.dll
	// is missing and only the SDK's own FSR 3.1 providers are there).
	static uint64_t FindFsr4(GraphicContext& graphics) {
		VkDeviceContext device {};
		device.vkDevice         = static_cast<VkDevice>(graphics.device);
		device.vkPhysicalDevice = static_cast<VkPhysicalDevice>(graphics.physical_device);
		device.vkDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
		uint64_t              count = 0;
		ffxQueryDescGetVersions versions {};
		versions.header.type  = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
		versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
		versions.device       = &device;
		versions.outputCount  = &count;
		if (TheLibrary().query(nullptr, &versions.header) != FFX_API_RETURN_OK || count == 0 || count > 256) return 0;
		std::vector<uint64_t>    ids(count);
		std::vector<const char*> names(count);
		versions.versionIds   = ids.data();
		versions.versionNames = names.data();
		if (TheLibrary().query(nullptr, &versions.header) != FFX_API_RETURN_OK) return 0;
		uint64_t fsr4 = 0;
		for (uint64_t i = 0; i < std::min<uint64_t>(count, ids.size()); ++i) {
			Report("FSR 4: provider %016llx %s\n", static_cast<unsigned long long>(ids[i]), names[i] != nullptr ? names[i] : "?");
			if (names[i] != nullptr && names[i][0] == '4' && names[i][1] == '.' && fsr4 == 0) fsr4 = ids[i];
		}
		return fsr4;
	}

	bool Create(GraphicContext& graphics, vk::Extent2D render_size, vk::Extent2D display_size) {
		render  = render_size;
		display = display_size;
		if (!TheLibrary().ready) {
			failed = true;
			return false;
		}
		uint64_t fsr4 = 0;
		{
			DeviceQueuesGuard queues(graphics);
			fsr4 = FindFsr4(graphics);
		}
		if (fsr4 == 0) {
			failed = true;
			Report("Super resolution: FSR 4 is not offered for this GPU (is amdvkffx64.dll next to the other DLLs, and "
			       "is the GPU an RX 6000 or newer?); FSR 3 built in\n");
			return false;
		}
		ffxCreateContextDescUpscaleVersion version {};
		version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
		version.version     = FFX_UPSCALER_VERSION;
		// Pin FSR 4: the SDK would otherwise be free to pick its FSR 3.1 provider.
		ffxOverrideVersion override_version {};
		override_version.header.type  = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
		override_version.header.pNext = &version.header;
		override_version.versionId    = fsr4;
		ffxCreateBackendVKDesc backend {};
		backend.header.type                    = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
		backend.header.pNext                   = &override_version.header;
		backend.deviceContext.vkDevice         = static_cast<VkDevice>(graphics.device);
		backend.deviceContext.vkPhysicalDevice = static_cast<VkPhysicalDevice>(graphics.physical_device);
		backend.deviceContext.vkDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
		ffxCreateContextDescUpscale create {};
		create.header.type  = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
		create.header.pNext = &backend.header;
		// The game's linear HDR scene before its tone mapping; reversed-Z with an infinite far plane (as FSR 3's).
		create.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
		               FFX_UPSCALE_ENABLE_DEPTH_INFINITE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
		if (std::getenv("KYTY_FSR_DEBUG") != nullptr) create.flags |= FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
		create.maxRenderSize  = {render.width, render.height};
		create.maxUpscaleSize = {display.width, display.height};
		create.fpMessage      = Message;
		{
			DeviceQueuesGuard queues(graphics);
			if (const auto code = TheLibrary().create(&context, &create.header, nullptr); code != FFX_API_RETURN_OK) {
				context = nullptr;
				return Fail("context", code);
			}
		}
		reset = true;
		Report("Super resolution: AMD FSR 4 %ux%u -> %ux%u (AMD's DLLs through ffx-vk-linux)\n", render.width, render.height,
		       display.width, display.height);
		return true;
	}

	bool Record(GraphicContext& graphics, vk::CommandBuffer command, const FrameGen::UpscaleInputs& inputs) {
		std::lock_guard lock(mutex);
		if (failed) return false;
		// (The runtime makes its own views of mip level 0 and layer 0.)
		for (const auto* image: {&inputs.color, &inputs.depth, &inputs.motion, &inputs.output}) {
			if (image->base_level != 0 || image->base_layer != 0) return false;
		}
		if (context != nullptr && (render != inputs.color.extent || display != inputs.output.extent)) Destroy(graphics);
		if (context == nullptr && !Create(graphics, inputs.color.extent, inputs.output.extent)) return false;
		static const float sharpness = [] {
			const char* value = std::getenv("KYTY_FSR_SHARPNESS");
			return value != nullptr ? std::clamp(std::strtof(value, nullptr), 0.0f, 1.0f) : 0.6f;
		}();
		ffxDispatchDescUpscale dispatch {};
		dispatch.header.type   = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
		dispatch.commandList   = static_cast<VkCommandBuffer>(command);
		dispatch.color         = ToResource(inputs.color, false);
		dispatch.depth         = ToResource(inputs.depth, false);
		dispatch.motionVectors = ToResource(inputs.motion, false);
		dispatch.output        = ToResource(inputs.output, true);
		// (The game's reactive mask: the pixels taken from this frame rather than the history, issue #232.)
		if (inputs.reactive.image) dispatch.reactive = ToResource(inputs.reactive, false);
		// (Render pixels, as DLSS takes them; the game's vectors times their scale: render pixels to the previous position.)
		dispatch.jitterOffset      = {inputs.jitter_x, inputs.jitter_y};
		dispatch.motionVectorScale = {inputs.motion_scale_x, inputs.motion_scale_y};
		dispatch.renderSize        = {render.width, render.height};
		dispatch.upscaleSize       = {display.width, display.height};
		dispatch.enableSharpening  = sharpness > 0.0f;
		dispatch.sharpness         = sharpness;
		const auto now             = std::chrono::steady_clock::now();
		const float ms             = last.time_since_epoch().count() == 0
		                                 ? 16.7f
		                                 : std::chrono::duration<float, std::milli>(now - last).count();
		last                       = now;
		dispatch.frameTimeDelta    = ms > 0.0f && ms < 200.0f ? ms : 16.7f;
		dispatch.preExposure       = 1.0f;
		dispatch.reset             = reset || inputs.reset;
		dispatch.cameraNear        = inputs.near_plane > 0.0f ? inputs.near_plane : 0.1f;
		dispatch.cameraFar         = FLT_MAX;
		// (Row vectors: the projection's y scale is cot(fov / 2).)
		const float y_scale              = std::fabs(inputs.view_to_clip[5]);
		dispatch.cameraFovAngleVertical  = y_scale > 0.0f ? 2.0f * std::atan(1.0f / y_scale) : 1.0f;
		dispatch.viewSpaceToMetersFactor = 1.0f;
		if (static uint32_t reports = 0; reports < 2) {
			++reports;
			Report("Super resolution: FSR 4 inputs: color %s, depth %s, motion %s, output %s %ux%u; jitter %g %g, motion "
			       "scale %g %g, near %g, vertical field of view %g, frame time %.1f ms, reset %d\n",
			       vk::to_string(inputs.color.format).c_str(), vk::to_string(inputs.depth.format).c_str(),
			       vk::to_string(inputs.motion.format).c_str(), vk::to_string(inputs.output.format).c_str(),
			       display.width, display.height, dispatch.jitterOffset.x, dispatch.jitterOffset.y,
			       dispatch.motionVectorScale.x, dispatch.motionVectorScale.y, dispatch.cameraNear,
			       dispatch.cameraFovAngleVertical, dispatch.frameTimeDelta, dispatch.reset ? 1 : 0);
		}
		if (const auto code = TheLibrary().dispatch(&context, &dispatch.header); code != FFX_API_RETURN_OK)
			return Fail("upscaling", code);
		reset = false;
		return true;
	}
};

Upscaler& TheUpscaler() {
	static auto* upscaler = new Upscaler; // (kept for the process: its context lives as long as the device)
	return *upscaler;
}

} // namespace

void AddDeviceExtensions(const std::vector<vk::ExtensionProperties>& available, std::vector<const char*>& device_extensions) {
	if (!Requested()) return;
	const auto has = [](const auto& list, const char* name) {
		return std::any_of(list.begin(), list.end(), [name](const auto& entry) {
			if constexpr (std::is_same_v<std::decay_t<decltype(entry)>, vk::ExtensionProperties>)
				return std::strcmp(entry.extensionName, name) == 0;
			else
				return std::strcmp(entry, name) == 0;
		});
	};
	// FSR 4's shaders: compute shader derivatives; AMD's coherent memory where advertised (the probe in ffx-vk-linux
	// enables both); and the extensions FidelityFX's Vulkan backend calls the functions of when the GPU offers them
	// (the same set as FSR 3 frame generation's, vulkanWindow.cpp).
	for (const char* extension:
	     {"VK_KHR_compute_shader_derivatives", "VK_AMD_device_coherent_memory", "VK_KHR_get_memory_requirements2",
	      "VK_KHR_dedicated_allocation", "VK_AMD_buffer_marker", "VK_KHR_synchronization2"}) {
		if (has(available, extension) && !has(device_extensions, extension)) device_extensions.push_back(extension);
	}
}

bool Upscale(GraphicContext& graphics, vk::CommandBuffer command, const FrameGen::UpscaleInputs& inputs) {
	return Requested() && TheUpscaler().Record(graphics, command, inputs);
}

#else

void AddDeviceExtensions(const std::vector<vk::ExtensionProperties>& /*available*/, std::vector<const char*>& /*device_extensions*/) {}

bool Upscale(GraphicContext& /*graphics*/, vk::CommandBuffer /*command*/, const FrameGen::UpscaleInputs& /*inputs*/) {
	return false;
}

#endif

} // namespace Libs::Graphics::Fsr4Linux

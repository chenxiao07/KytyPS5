#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEPTHRENDERTARGET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEPTHRENDERTARGET_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

// Without a stencil plane, Hi-Stencil fields are inactive. An active plane is compatible when
// Hi-Stencil is disabled or HTile backing is present.
inline constexpr bool depth_htile_stencil_acceleration_compatible(bool has_stencil, bool has_htile,
                                                                  bool htile_stencil_disabled) {
	return !has_stencil || htile_stencil_disabled || has_htile;
}

inline constexpr bool htile_fill_clears_depth(uint32_t fill) {
	return (fill & 0xfu) == 0u;
}

inline constexpr bool htile_fill_clears_stencil(uint32_t fill) {
	return ((fill >> 8u) & 0x3u) == 0u;
}

inline constexpr bool htile_fill_depth_uniform(uint32_t fill, bool stencil_compressed) {
	if ((fill & 0xfu) == 0xfu) {
		return false;
	}
	if (stencil_compressed) {
		return ((fill >> 12u) & 0x3fu) == 0u;
	}
	return ((fill >> 4u) & 0x3fffu) == ((fill >> 18u) & 0x3fffu);
}

inline constexpr float htile_fill_depth_value(uint32_t fill, bool stencil_compressed) {
	const uint32_t z = stencil_compressed ? fill >> 18u : (fill >> 4u) & 0x3fffu;
	return static_cast<float>(z) / 16383.0f;
}

// A depth target's load from its HTile clear state (IsMetaCleared's answer and fill), as AcquireRenderTargets decides it.
// One rule for both callers: the normal path, and the spec_depth_layers2 commit that decides it for a speculation's draw.
struct HtileLoadDecision {
	bool  depth_meta_clear   = false; // the depth loads CLEAR for the metadata's clear
	bool  stencil_meta_clear = false; // the stencil does
	bool  depth_uniform      = false; // the clear value is `uniform_value`, not DB_DEPTH_CLEAR's
	float uniform_value      = 0.0f;
};
inline constexpr HtileLoadDecision htile_load_decision(bool meta_cleared, uint32_t fill, bool fill_known,
                                                       bool stencil_compressed) {
	HtileLoadDecision d {};
	d.depth_uniform = meta_cleared && fill_known && !htile_fill_clears_depth(fill) &&
	                  htile_fill_depth_uniform(fill, stencil_compressed);
	d.depth_meta_clear   = meta_cleared && (!fill_known || htile_fill_clears_depth(fill) || d.depth_uniform);
	d.stencil_meta_clear = meta_cleared && fill_known && stencil_compressed && htile_fill_clears_stencil(fill);
	if (d.depth_uniform) d.uniform_value = htile_fill_depth_value(fill, stencil_compressed);
	return d;
}

struct RenderDepthInfo {
	// Discovery keeps guest image information but can remap the view into a larger cache image.
	TextureCache::ImageDesc     desc;
	bool                        depth_clear_enable       = false;
	bool                        depth_load_clear_enable  = false;
	float                       depth_clear_value        = 0.0f;
	bool                        depth_test_enable        = false;
	// Effective draw writes; discovery applies test, target-write and clear controls.
	bool                        depth_write_enable       = false;
	vk::CompareOp               depth_compare_op         = vk::CompareOp::eNever;
	bool                        depth_bounds_test_enable = false;
	float                       depth_min_bounds         = 0.0f;
	float                       depth_max_bounds         = 0.0f;
	bool                        stencil_clear_enable     = false;
	bool                        stencil_meta_clear_enable = false;
	uint8_t                     stencil_clear_value      = 0;
	bool                        stencil_test_enable      = false;
	PipelineStencilStaticState  stencil_static_front;
	PipelineStencilStaticState  stencil_static_back;
	PipelineStencilDynamicState stencil_dynamic_front;
	PipelineStencilDynamicState stencil_dynamic_back;
	ImageId                     image_id;

	[[nodiscard]] vk::ImageAspectFlags AttachmentWriteAspects() const;
};

inline vk::ImageLayout depth_attachment_layout(const RenderDepthInfo& depth) {
	const auto available     = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
	const auto writes        = depth.AttachmentWriteAspects();
	const bool has_depth     = static_cast<bool>(available & vk::ImageAspectFlagBits::eDepth);
	const bool has_stencil   = static_cast<bool>(available & vk::ImageAspectFlagBits::eStencil);
	// LoadOp clears are separate from guest depth writes.
	const bool depth_write   = depth.depth_write_enable;
	const bool stencil_write = static_cast<bool>(writes & vk::ImageAspectFlagBits::eStencil);
	if (!has_stencil) {
		return depth_write ? vk::ImageLayout::eDepthAttachmentOptimal
		                   : vk::ImageLayout::eDepthReadOnlyOptimal;
	}
	if (!has_depth) {
		return stencil_write ? vk::ImageLayout::eStencilAttachmentOptimal
		                     : vk::ImageLayout::eStencilReadOnlyOptimal;
	}
	if (depth_write && stencil_write) {
		return vk::ImageLayout::eDepthStencilAttachmentOptimal;
	}
	if (depth_write) {
		return vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal;
	}
	if (stencil_write) {
		return vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal;
	}
	return vk::ImageLayout::eDepthStencilReadOnlyOptimal;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEPTHRENDERTARGET_H_

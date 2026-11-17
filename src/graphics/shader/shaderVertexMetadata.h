#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADERVERTEXMETADATA_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADERVERTEXMETADATA_H_

#include "graphics/shader/shader.h"

#include <array>
#include <string>

namespace Libs::Graphics {

struct ShaderVertexMetadata {
	int                                                        vertex_buffer_reg = -1;
	int                                                        vertex_attrib_reg = -1;
	std::array<ShaderSemantic, ShaderVertexInputInfo::RES_MAX> input_semantics {};
	uint32_t                                                   input_semantics_count = 0;
};

// Copies the small AGC metadata subset used by the vertex path after validating every guest range.
// `read` (null: memcpy) copies the guest bytes; false fails the read ("AGC metadata unread").
bool ShaderReadVertexMetadata(const ShaderMappedData& data, uint32_t max_user_sgprs,
                              ShaderVertexMetadata& metadata, std::string* error,
                              bool (*read)(void* out, const void* guest, size_t bytes) = nullptr);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADERVERTEXMETADATA_H_ */

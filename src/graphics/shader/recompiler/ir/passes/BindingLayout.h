#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// `table_dead_user_data` (CompileOptions::table_dead_user_data, KYTY_TABLE_DEAD_USER_DATA): a table mode program whose
// user data do not fit the push constants from `push_data_start_dword` tries once more without the registers only
// handles the table module emits nothing for read (TableLiveUserData); a program that fits is laid out as without it.
void AllocateBindings(Program& program, uint32_t push_data_start_dword = 0, bool enable_lod_stats = false,
                      bool table_dead_user_data = false);

// A handle (GetBufferResource, GetAddressResource) whose every use is a load only the CPU-side plan reads
// (MemoryInfo::planning_only: LoadAddressU32, ReadConstBuffer), which the SPIR-V emitter emits nothing for, so no word
// of the handle is read on the GPU.
[[nodiscard]] bool PlanningOnlyHandle(const Program& program, const Inst& handle);

// Table mode (Program::table_mode) for a program it can serve: only reads, of buffers whose V#s and of SRT
// words whose addresses the flattened SRT plan derives from user data. Exits (recoverable) otherwise.
void EnterTableMode(Program& program);

const DescriptorBinding* FindBinding(const BindingLayout& layout, DescriptorBindingKind kind);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_ */

#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <optional>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

[[noreturn]] void BindingFail(const char* message) {
	EXIT("shader binding layout failed: %s", message);
	std::abort();
}

std::vector<uint32_t> CollectUserData(const Program& program) {
	std::array<bool, NumScalarRegs> registers {};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::GetUserData || !inst.HasUses()) {
				continue;
			}
			if (inst.Arg(0).GetType() != Type::ScalarReg) {
				BindingFail("typed shader contains an invalid user-data register");
			}
			const auto index = RegIndex(inst.Arg(0).ScalarRegister());
			if (index >= NumScalarRegs) {
				BindingFail("typed shader contains an invalid user-data register");
			}
			registers[index] = true;
		}
	}
	std::vector<uint32_t> result;
	for (uint32_t index = 0; index < registers.size(); index++) {
		if (registers[index]) {
			result.push_back(index);
		}
	}
	return result;
}

void AddBinding(BindingLayout& layout, DescriptorBindingKind kind,
                std::vector<uint32_t> resources = {}) {
	layout.descriptors.push_back({kind, std::move(resources)});
}

bool UsesGds(const Program& program) {
	bool uses_gds = false;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (SharedAccessOf(inst.GetOpcode()) == SharedAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= program.memory_info.size()) {
				BindingFail("typed shader contains invalid shared-memory metadata");
			}
			const auto kind = program.memory_info[index].kind;
			if (kind != ResourceKind::Lds && kind != ResourceKind::Gds) {
				BindingFail("typed shader contains invalid shared-memory metadata");
			}
			uses_gds |= kind == ResourceKind::Gds;
		}
	}
	return uses_gds;
}

// A value table mode evaluates at the program's entry: an immediate, user data or an SRT slot below `slots`.
std::optional<TablePlan::Operand> TableOperand(Value value, size_t slots) {
	using Kind = TablePlan::Operand::Kind;
	value      = value.Resolve();
	if (value.IsImmediate()) {
		if (value.GetType() != Type::U32) return std::nullopt;
		return TablePlan::Operand {Kind::Immediate, value.U32()};
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return std::nullopt;
	if (inst->GetOpcode() == ValueOpcode::GetUserData && inst->Arg(0).GetType() == Type::ScalarReg)
		return TablePlan::Operand {Kind::UserData, static_cast<uint32_t>(RegIndex(inst->Arg(0).ScalarRegister()))};
	if (inst->GetOpcode() != ValueOpcode::ReadConst) return std::nullopt;
	const auto slot = inst->Arg(1).Resolve();
	if (!slot.IsImmediate() || slot.GetType() != Type::U32 || slot.U32() >= slots) return std::nullopt;
	return TablePlan::Operand {Kind::Slot, slot.U32()};
}

} // namespace

void EnterTableMode(Program& program) {
	const auto refuse = [&](const char* reason) {
		EXIT("table mode refused: hash=0x%016" PRIx64 " %s\n", program.shader_hash, reason);
	};
	if (!program.srt_plan_complete || !program.resource_tracking_complete || !program.dynamic_reads.empty())
		refuse("no complete SRT plan");
	if (UsesGds(program)) refuse("GDS");
	TablePlan plan;
	plan.global_memory = program.info.uses_dma; // (table mode reads its block by device address in any case)
	// The flattened SRT's slots in order (a slot's address only depends on lower slots and user data).
	for (size_t slot = 0; slot < program.srt_reads.size(); ++slot) {
		const auto* load = program.srt_reads[slot].value.Resolve().TryInstruction();
		if (load == nullptr || load->GetOpcode() != ValueOpcode::LoadAddressU32 || load->NumArgs() != 4 ||
		    load->Flags<MemoryFlags>().index >= program.memory_info.size())
			refuse("an SRT slot that is no address load");
		const auto& memory = program.memory_info[load->Flags<MemoryFlags>().index];
		const auto  active = load->Arg(3).Resolve();
		const auto* handle = load->Arg(0).Resolve().TryInstruction();
		if (memory.kind != ResourceKind::ScalarAddress || memory.address_is_full || !active.IsImmediate() ||
		    !active.U1() || handle == nullptr ||
		    handle->GetOpcode() != ValueOpcode::GetAddressResource || handle->NumArgs() != 2)
			refuse("an SRT slot that is no scalar address load");
		const auto low = TableOperand(handle->Arg(0), slot), high = TableOperand(handle->Arg(1), slot);
		const auto offset = TableOperand(load->Arg(1), slot);
		if (!low || !high || !offset) refuse("an SRT slot address table mode cannot evaluate");
		plan.slots.push_back({*low, *high, *offset,
		                      static_cast<int32_t>(static_cast<uint32_t>(static_cast<int32_t>(memory.offset)) & ~3u)});
	}
	const auto words_of = [&](uint32_t source_index, uint32_t count, auto& words) {
		if (source_index >= program.descriptor_sources.size()) refuse("a descriptor without a source");
		const auto& source = program.descriptor_sources[source_index];
		if (source.dword_count != count || source.indirect_image) refuse("a descriptor table mode cannot evaluate");
		for (uint32_t i = 0; i < count; ++i) {
			const auto word = TableOperand(source.dwords[i], program.srt_reads.size());
			if (!word) refuse("a descriptor word table mode cannot evaluate");
			words[i] = *word;
		}
	};
	// (Written and atomic buffers too: through their device address, as the renderer resolved them.)
	for (const auto& buffer: program.info.buffers) {
		if (buffer.image_alias != BufferResource::NoImageAlias || (buffer.packed_stride & ((1u << 14u) | (1u << 20u))) != 0)
			refuse("an aliased, swizzled or ADD_TID buffer");
		words_of(buffer.source, 4, plan.buffers.emplace_back());
	}
	for (const auto& image: program.info.images) {
		if (image.mip_mode != ImageMipMode::None || image.indirect_root != ImageResource::NoIndirectImage ||
		    image.indirect_search_iterations != 0)
			refuse("an indirect image or one of dynamic mip levels");
		// Pixel and compute shaders may write storage images (TableResolveSet binds them as CommitBindings does: the
		// deferred decals, the async culling chain's dispatches). The chain's translation is on the frame's critical
		// path since its uploads stage in video memory: 1-1 same process 51.81 -> 53.21 fps (10-07; 10-05, when the
		// chain waited on those uploads, 51.1 vs 51.0).
		if (image.atomic || ((image.written || image.resource_class == ImageResourceClass::Storage) &&
		                     program.stage != ShaderType::Pixel && program.stage != ShaderType::Compute))
			refuse("an image the shader writes");
		auto& descriptor = plan.images.emplace_back();
		descriptor.count = program.descriptor_sources.at(image.source).dword_count;
		if (descriptor.count != 4 && descriptor.count != 8) refuse("an image descriptor of another size");
		words_of(image.source, descriptor.count, descriptor.words);
	}
	for (const auto& sampler: program.info.samplers) {
		auto& descriptor = plan.samplers.emplace_back();
		descriptor.count = 4;
		words_of(sampler.source, 4, descriptor.words);
	}
	// The slots the shader reads (from its block): the V#s' words, branch conditions and any read with a use other
	// than the address of planning-only loads (which only told the CPU-side plan where the slots are).
	using Kind     = TablePlan::Operand::Kind;
	plan.gpu.assign(plan.slots.size(), 0);
	plan.cpu.assign(plan.slots.size(), 0);
	const auto mark = [](std::vector<uint8_t>& set, const TablePlan::Operand& operand) {
		if (operand.kind == Kind::Slot) set[operand.value] = 1;
	};
	const auto slot_of = [&](Value value) -> std::optional<uint32_t> {
		const auto* inst = value.Resolve().TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::ReadConst) return std::nullopt;
		return inst->Arg(1).Resolve().U32();
	};
	const auto planning = [&](const Inst& inst) {
		const auto index = inst.Flags<MemoryFlags>().index;
		return (inst.GetOpcode() == ValueOpcode::LoadAddressU32 || inst.GetOpcode() == ValueOpcode::ReadConstBuffer) &&
		       index < program.memory_info.size() && program.memory_info[index].planning_only;
	};
	for (const auto& words: plan.buffers)
		for (const auto& word: words) mark(plan.cpu, word);
	for (const auto* descriptors: {&plan.images, &plan.samplers})
		for (const auto& descriptor: *descriptors)
			for (uint32_t i = 0; i < descriptor.count; ++i) mark(plan.cpu, descriptor.words[i]);
	for (const auto& info: program.block_info)
		for (const auto value: {info.condition, info.indirect_target})
			if (const auto slot = value.IsEmpty() ? std::nullopt : slot_of(value); slot && *slot < plan.gpu.size())
				plan.gpu[*slot] = 1;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			const auto slot = slot_of(Value(&inst));
			if (!slot || *slot >= plan.gpu.size()) continue;
			for (const auto& use: inst.Uses()) {
				const auto& user = *use.user;
				if (user.GetOpcode() != ValueOpcode::GetAddressResource ||
				    !std::ranges::all_of(user.Uses(), [&](const Use& address) { return planning(*address.user); }))
					plan.gpu[*slot] = 1;
			}
		}
	}
	// The renderer evaluates what the shader reads and the image and sampler words, with the slots their addresses
	// are read through.
	for (size_t slot = 0; slot < plan.slots.size(); ++slot) plan.cpu[slot] |= plan.gpu[slot];
	for (size_t slot = plan.slots.size(); slot-- > 0;) {
		if (plan.cpu[slot] == 0) continue;
		const auto& read = plan.slots[slot];
		for (const auto* operand: {&read.low, &read.high, &read.offset}) mark(plan.cpu, *operand);
	}
	program.table_plan    = std::move(plan);
	program.table_mode    = true;
	program.info.uses_dma = true;
}

void AllocateBindings(Program& program, uint32_t push_data_start_dword, bool enable_lod_stats) {
	if (!program.shader_info_complete || program.binding_layout_complete) {
		EXIT("shader binding layout failed: %s", !program.shader_info_complete
		                                             ? "shader info is not ready"
		                                             : "binding layout already allocated");
	}
	BindingLayout next;
	next.user_data_registers = CollectUserData(program);
	next.memory_offset_dword = static_cast<uint32_t>(next.user_data_registers.size());
	// Table mode reads its slots and buffers through its block: no offsets or buffer words in the shader data, the
	// block's device address instead.
	if (program.table_mode) {
		next.table_block_dwords = 2;
	} else {
		next.memory_offset_count = static_cast<uint32_t>(program.info.buffers.size());
		if (PortableShaders()) next.buffer_word_count = static_cast<uint32_t>(program.info.buffers.size());
	}
	if (enable_lod_stats && program.stage == ShaderType::Pixel && !program.info.images.empty()) {
		next.lod_stats_count = static_cast<uint32_t>(program.info.images.size());
		AddBinding(next, DescriptorBindingKind::LodStats);
	}
	next.push_data_start_dword =
	    PushData::StartFor(push_data_start_dword, next.ShaderDataDwords());

	if (!program.info.buffers.empty() && !program.table_mode) {
		std::vector<uint32_t> resources(program.info.buffers.size());
		for (uint32_t i = 0; i < resources.size(); i++) {
			resources[i] = i;
		}
		AddBinding(next, DescriptorBindingKind::Buffers, std::move(resources));
	}

	std::array<std::vector<uint32_t>, ImageBindingCount> image_groups;
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto kind = DescriptorBindingForImage(program.info.images[i]);
		if (!kind.has_value()) {
			EXIT("shader binding layout failed: image %u has an invalid binding class", i);
		}
		const auto group = ImageBindingIndex(*kind);
		if (group >= image_groups.size()) {
			EXIT("shader binding layout failed: image %u has an unmapped binding class", i);
		}
		auto&      resources = image_groups[group];
		const auto dynamic   = program.info.images[i].mip_mode == ImageMipMode::DynamicStorage;
		const auto count     = dynamic ? program.info.images[i].mip_count : 1u;
		if (count == 0u || (!dynamic && program.info.images[i].mip_count != 1u)) {
			EXIT("shader binding layout failed: image %u has invalid specialized mip count %u", i,
			     program.info.images[i].mip_count);
		}
		resources.insert(resources.end(), count, i);
	}
	for (uint32_t i = 0; i < image_groups.size(); i++) {
		if (!image_groups[i].empty()) {
			AddBinding(next, static_cast<DescriptorBindingKind>(FirstImageBinding + i),
			           std::move(image_groups[i]));
		}
	}

	if (!program.info.samplers.empty()) {
		std::vector<uint32_t> resources(program.info.samplers.size());
		for (uint32_t i = 0; i < resources.size(); i++) {
			resources[i] = i;
		}
		AddBinding(next, DescriptorBindingKind::Samplers, std::move(resources));
	}
	if (UsesGds(program)) {
		AddBinding(next, DescriptorBindingKind::Gds);
	}
	if (program.info.uses_dma) {
		AddBinding(next, DescriptorBindingKind::BdaPagetable);
		AddBinding(next, DescriptorBindingKind::FaultBuffer);
	}
	const bool uses_flattened_runtime =
	    !program.srt_reads.empty() ||
	    std::ranges::any_of(program.info.images, [](const ImageResource& image) {
		    return image.indirect_search_iterations != 0u;
	    });
	if (uses_flattened_runtime && !program.table_mode) {
		AddBinding(next, DescriptorBindingKind::FlattenedSrt);
	}

	if (next.ShaderDataDwords() != 0 && !next.UsesPushData()) {
		if (program.table_mode) BindingFail("table mode user data do not fit the push constants");
		AddBinding(next, DescriptorBindingKind::ShaderData);
	}

	program.bindings                = std::move(next);
	program.binding_layout_complete = true;
}

const DescriptorBinding* FindBinding(const BindingLayout& layout, DescriptorBindingKind kind) {
	for (const auto& binding: layout.descriptors) {
		if (binding.kind == kind) {
			return &binding;
		}
	}
	return nullptr;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR

// --- KYTY_LANE_LOCAL_SPILL=2: R1's bitwise output compare (lane-spill-verify.h) ---------------------------------------
//
// Included by table-xpr.inc (namespace Libs::Graphics). TableDispatch of an R1 program whose fallback pipeline is made
// (TablePair::spill_pipeline) records, for its first LaneSpillVerify::Limit() dispatches:
//   barrier; each written image's subresources -> section 0 (saved: the targets before the dispatch); barrier
//   the FALLBACK pipeline's dispatch (same set, push constants and group counts); barrier
//   the targets -> section 1 (the fallback's output); barrier; section 0 -> the targets (restored); barrier
//   the R1 pipeline's dispatch (the one mode 1 records: its output stays); barrier
//   the targets -> section 2 (R1's output); barrier to the host and to every later command
// and, once that submission's work completed (CommandScheduler::DeferOperation: no stall), compares sections 1 and 2
// word by word on the CPU. With the targets restored in between, R1 sees the same inputs as the fallback even where it
// reads what it writes, and a word R1 fails to write shows (it keeps the saved value, not the fallback's).
// The sections are host-visible buffers in a pool of Slots slots, each kept (grown, never shrunk) for the session; a
// dispatch that finds every slot waiting for its compare runs R1 alone (busy_skips).
// Covered: written storage images of color formats whose texel is a power of two, one sample, in the general layout.
// Not covered (the program is skipped, with a line once): a program that writes buffers or flat memory, a fallback
// whose table form or pipeline layout differs from R1's, a set made for another specialization.

namespace LaneSpillCompare {

constexpr size_t   Slots          = 4;
constexpr uint64_t SectionAlign   = 256;
constexpr uint32_t MismatchReport = 8; // lines per program naming a mismatching word

struct Copy {
	vk::Image                        image = nullptr;
	vk::Format                       format = vk::Format::eUndefined;
	uint64_t                         offset = 0, bytes = 0; // in a section
	std::vector<vk::BufferImageCopy> regions;               // offsets in a section
	uint32_t                         level0 = 0, levels = 0, layer0 = 0, layers = 0;
};

struct Slot {
	VkBuffer      buffer   = VK_NULL_HANDLE;
	VmaAllocation memory   = nullptr;
	uint8_t*      mapped   = nullptr;
	uint64_t      capacity = 0;
	bool          busy     = false;
};

struct Totals {
	uint32_t started = 0, compared = 0, mismatch_lines = 0;
	uint64_t words = 0, mismatches = 0, fallback_wrote = 0, r1_wrote = 0, busy_skips = 0;
	bool     reads_output = false, skip_logged = false;
	std::chrono::steady_clock::time_point printed {};
};

struct State {
	std::mutex                           mutex;
	std::array<Slot, Slots>              slots;
	std::unordered_map<uint64_t, Totals> totals;
	VmaAllocator                         allocator = nullptr;
};

inline State& Get() {
	static State state;
	return state;
}

// One compared dispatch being recorded (Prepare to Finish).
struct Pending {
	uint64_t          hash    = 0;
	size_t            slot    = Slots;
	uint64_t          section = 0;
	bool              reads_output = false;
	std::vector<Copy> copies;
	vk::Buffer        buffer = nullptr;
};

inline void Skip(Totals& totals, uint64_t hash, const char* why) {
	if (totals.skip_logged) return;
	totals.skip_logged = true;
	std::printf("LaneLocal verify %016llx: skipped, %s\n", static_cast<unsigned long long>(hash), why);
	std::fflush(stdout);
}

inline void Barrier(vk::CommandBuffer command, vk::PipelineStageFlags source, vk::AccessFlags source_access,
                    vk::PipelineStageFlags destination, vk::AccessFlags destination_access) {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = source_access;
	barrier.dstAccessMask = destination_access;
	command.pipelineBarrier(source, destination, vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr);
}

// The regions of `copy` moved to section `index`.
inline std::vector<vk::BufferImageCopy> Regions(const Copy& copy, uint64_t section, uint32_t index) {
	auto regions = copy.regions;
	for (auto& region: regions) region.bufferOffset += section * index;
	return regions;
}

// Whether this dispatch is compared; on true, `pending` holds its copies and a slot it took. `set`: the dispatch's
// TableImageSet; `program` / `fallback`: R1's and the fallback's compiled info; `layout` / `fallback_layout`: their
// pipelines' layouts; `storage_writes`: TableBlocks found the program writing buffers.
template <typename ImageSet>
bool Prepare(GraphicContext& graphics, TextureCache& textures, const ImageSet& set,
             const ShaderRecompiler::IR::CompiledShaderInfo& program,
             const ShaderRecompiler::IR::CompiledShaderInfo& fallback, vk::PipelineLayout layout,
             vk::PipelineLayout fallback_layout, bool storage_writes, Pending& pending) {
	auto&           s    = Get();
	const auto      hash = program.shader_hash;
	std::lock_guard lock(s.mutex);
	auto&           totals = s.totals[hash];
	if (totals.started >= LaneSpillVerify::Limit()) return false;
	if (set.program != nullptr || set.pipeline != nullptr)
		return false; // (a set made for another specialization: its own program, no fallback for it)
	if (!TableInterchangeable(program, fallback)) return Skip(totals, hash, "the fallback's table form differs"), false;
	if (layout != fallback_layout) return Skip(totals, hash, "the fallback's pipeline layout differs"), false;
	const bool writes_buffers = std::ranges::any_of(program.info.buffers, [](const auto& item) {
		return item.written || item.atomic;
	});
	if (writes_buffers || storage_writes) return Skip(totals, hash, "writes buffers (not covered)"), false;
	if (program.table_plan.global_memory) return Skip(totals, hash, "reads or writes flat memory (not covered)"), false;
	pending              = {};
	pending.hash         = hash;
	pending.reads_output = std::ranges::any_of(program.info.images, [](const auto& item) {
		return (item.written || item.atomic) && item.read;
	});
	uint64_t offset = 0;
	for (const auto& dep: set.images) {
		if (!dep.storage) continue;
		if (dep.whole) pending.reads_output = true;
		for (const auto& other: set.images)
			if (&other != &dep && !other.storage && other.id == dep.id) pending.reads_output = true;
		auto* image = textures.TryGetImage(dep.id);
		if (image == nullptr || image->backing.image == nullptr) return Skip(totals, hash, "a written image without a Vulkan image"), false;
		const auto& backing = image->backing;
		const auto  extent  = vk::blockExtent(backing.format);
		const auto  texel   = static_cast<uint64_t>(vk::blockSize(backing.format));
		if (image->info.IsDepth() || extent[0] != 1 || extent[1] != 1 || extent[2] != 1 || texel == 0 ||
		    !std::has_single_bit(texel) || backing.samples != 1)
			return Skip(totals, hash, "a written image of a format it does not copy (depth, blocks, 3-byte texels, MSAA)"), false;
		if (dep.layout != vk::ImageLayout::eGeneral)
			return Skip(totals, hash, "a written image not in the general layout"), false;
		Copy copy;
		copy.image          = backing.image;
		copy.format         = backing.format;
		const bool volume   = backing.image_type == vk::ImageType::e3D;
		const auto level0   = dep.range.base_level;
		const auto levels   = std::min(dep.range.level_count, backing.mip_levels > level0 ? backing.mip_levels - level0 : 0u);
		const auto layer0   = volume ? 0u : dep.range.base_layer;
		const auto layers   = volume ? 1u
		                             : std::min(dep.range.layer_count, backing.layers > layer0 ? backing.layers - layer0 : 0u);
		// (The same subresources twice: one copy. TableResolveSet marks every binding of an image that is bound both as
		// a texture and as storage `storage` (force_general: the image's shader_write), so such an image came twice, and
		// its two restores were two unordered transfer writes of one image: Swap's SYNC-HAZARD-WRITE-AFTER-WRITE.)
		if (std::ranges::any_of(pending.copies, [&](const Copy& other) {
			    return other.image == copy.image && other.level0 == level0 && other.levels == levels &&
			           other.layer0 == layer0 && other.layers == layers;
		    }))
			continue;
		copy.level0 = level0;
		copy.levels = levels;
		copy.layer0 = layer0;
		copy.layers = layers;
		const uint64_t align = std::max<uint64_t>(texel, 4);
		offset               = (offset + SectionAlign - 1) & ~(SectionAlign - 1);
		copy.offset          = offset;
		for (uint32_t level = level0; level < level0 + levels && layers != 0; ++level) {
			vk::BufferImageCopy region {};
			region.bufferOffset     = offset;
			region.imageSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, level, layer0, layers};
			region.imageExtent      = vk::Extent3D {std::max(backing.extent.width >> level, 1u),
			                                        std::max(backing.extent.height >> level, 1u),
			                                        volume ? std::max(backing.extent.depth >> level, 1u) : 1u};
			copy.regions.push_back(region);
			offset += uint64_t {region.imageExtent.width} * region.imageExtent.height * region.imageExtent.depth * layers * texel;
			offset = (offset + align - 1) & ~(align - 1);
		}
		copy.bytes = offset - copy.offset;
		if (copy.bytes != 0) pending.copies.push_back(std::move(copy));
	}
	if (pending.copies.empty()) return Skip(totals, hash, "writes no image it can compare"), false;
	totals.reads_output = totals.reads_output || pending.reads_output;
	const uint64_t section = (offset + SectionAlign - 1) & ~(SectionAlign - 1);
	if (section * 3 > LaneSpillVerify::SlotBytes())
		return Skip(totals, hash, "its written images exceed KYTY_LANE_LOCAL_SPILL_VERIFY_MB"), false;
	// A free slot, grown as needed (free: no recorded work uses its buffer any more).
	size_t index = Slots;
	for (size_t i = 0; i < Slots && index == Slots; ++i)
		if (!s.slots[i].busy) index = i;
	if (index == Slots) return ++totals.busy_skips, false;
	auto& slot = s.slots[index];
	if (slot.capacity < section * 3) {
		if (slot.buffer != VK_NULL_HANDLE) vmaDestroyBuffer(s.allocator, slot.buffer, slot.memory);
		slot = {};
		VkBufferCreateInfo create {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		create.size  = section * 3;
		create.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo allocate {};
		allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		allocate.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo mapped {};
		if (vmaCreateBuffer(graphics.allocator, &create, &allocate, &slot.buffer, &slot.memory, &mapped) != VK_SUCCESS ||
		    mapped.pMappedData == nullptr) {
			if (slot.buffer != VK_NULL_HANDLE) vmaDestroyBuffer(graphics.allocator, slot.buffer, slot.memory);
			slot = {};
			return Skip(totals, hash, "no host-visible memory for its copies"), false;
		}
		s.allocator   = graphics.allocator;
		slot.mapped   = static_cast<uint8_t*>(mapped.pMappedData);
		slot.capacity = section * 3;
	}
	slot.busy       = true;
	pending.slot    = index;
	pending.section = section;
	pending.buffer  = vk::Buffer {slot.buffer};
	totals.started++;
	return true;
}

// Before the fallback's dispatch: the targets saved (section 0).
inline void Save(vk::CommandBuffer command, const Pending& pending) {
	Barrier(command, vk::PipelineStageFlagBits::eAllCommands, vk::AccessFlagBits::eMemoryWrite,
	        vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite);
	for (const auto& copy: pending.copies) {
		const auto regions = Regions(copy, pending.section, 0);
		command.copyImageToBuffer(copy.image, vk::ImageLayout::eGeneral, pending.buffer,
		                          static_cast<uint32_t>(regions.size()), regions.data());
	}
	Barrier(command, vk::PipelineStageFlagBits::eTransfer,
	        vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
	        vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

// Between the two dispatches: the fallback's output (section 1), then the targets as they were (section 0).
inline void Swap(vk::CommandBuffer command, const Pending& pending) {
	Barrier(command, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite,
	        vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite);
	for (const auto& copy: pending.copies) {
		const auto regions = Regions(copy, pending.section, 1);
		command.copyImageToBuffer(copy.image, vk::ImageLayout::eGeneral, pending.buffer,
		                          static_cast<uint32_t>(regions.size()), regions.data());
	}
	// (The image read above before it is written below; section 0 written by Save before it is read below.)
	Barrier(command, vk::PipelineStageFlagBits::eTransfer,
	        vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite, vk::PipelineStageFlagBits::eTransfer,
	        vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite);
	for (size_t c = 0; c < pending.copies.size(); ++c) {
		const auto& copy = pending.copies[c];
		// (An image restored again, other subresources that may overlap the last: its writes ordered.)
		if (std::any_of(pending.copies.begin(), pending.copies.begin() + static_cast<std::ptrdiff_t>(c),
		                [&](const Copy& other) { return other.image == copy.image; }))
			Barrier(command, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite,
			        vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		const auto regions = Regions(copy, pending.section, 0);
		command.copyBufferToImage(pending.buffer, copy.image, vk::ImageLayout::eGeneral,
		                          static_cast<uint32_t>(regions.size()), regions.data());
	}
	Barrier(command, vk::PipelineStageFlagBits::eTransfer,
	        vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
	        vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
}

// The compare of a finished dispatch's sections (its submission's work completed).
inline void Compare(const Pending& pending) {
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	auto&           slot   = s.slots[pending.slot];
	auto&           totals = s.totals[pending.hash];
	if (slot.mapped != nullptr) {
		(void)vmaInvalidateAllocation(s.allocator, slot.memory, 0, VK_WHOLE_SIZE);
		uint64_t mismatches = 0;
		for (size_t c = 0; c < pending.copies.size(); ++c) {
			const auto& copy  = pending.copies[c];
			const auto  words = copy.bytes / 4u;
			const auto* saved = reinterpret_cast<const uint32_t*>(slot.mapped + copy.offset);
			const auto* fall  = reinterpret_cast<const uint32_t*>(slot.mapped + pending.section + copy.offset);
			const auto* r1    = reinterpret_cast<const uint32_t*>(slot.mapped + 2 * pending.section + copy.offset);
			for (uint64_t w = 0; w < words; ++w) {
				totals.fallback_wrote += fall[w] != saved[w] ? 1u : 0u;
				totals.r1_wrote += r1[w] != saved[w] ? 1u : 0u;
				if (fall[w] == r1[w]) continue;
				++mismatches;
				if (totals.mismatch_lines < MismatchReport) {
					totals.mismatch_lines++;
					std::printf("LaneLocal verify %016llx: mismatch at dispatch %u, image %zu (%s) byte %llu: "
					            "fallback %08x r1 %08x (saved %08x)\n",
					            static_cast<unsigned long long>(pending.hash), totals.compared, c,
					            vk::to_string(copy.format).c_str(), static_cast<unsigned long long>(w * 4u), fall[w], r1[w],
					            saved[w]);
				}
			}
			totals.words += words;
		}
		totals.mismatches += mismatches;
		totals.compared++;
		const auto now = std::chrono::steady_clock::now();
		if (totals.compared == 1 || totals.compared == LaneSpillVerify::Limit() || mismatches != 0 ||
		    now - totals.printed >= std::chrono::seconds(1)) {
			totals.printed = now;
			std::printf("LaneLocal verify %016llx: dispatches=%u compared_words=%llu mismatches=%llu fallback_wrote=%llu "
			            "r1_wrote=%llu busy_skips=%llu reads_output=%d%s\n",
			            static_cast<unsigned long long>(pending.hash), totals.compared,
			            static_cast<unsigned long long>(totals.words), static_cast<unsigned long long>(totals.mismatches),
			            static_cast<unsigned long long>(totals.fallback_wrote),
			            static_cast<unsigned long long>(totals.r1_wrote),
			            static_cast<unsigned long long>(totals.busy_skips), totals.reads_output ? 1 : 0,
			            totals.compared == LaneSpillVerify::Limit() ? " (done)" : "");
			std::fflush(stdout);
		}
	}
	slot.busy = false;
}

// After R1's dispatch: its output (section 2), made visible to the host and ordered before every later command; the
// compare once the submission's work completed.
inline void Finish(vk::CommandBuffer command, CommandScheduler& scheduler, Pending&& pending) {
	Barrier(command, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite,
	        vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
	for (const auto& copy: pending.copies) {
		const auto regions = Regions(copy, pending.section, 2);
		command.copyImageToBuffer(copy.image, vk::ImageLayout::eGeneral, pending.buffer,
		                          static_cast<uint32_t>(regions.size()), regions.data());
	}
	Barrier(command, vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
	        vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
	        vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eAllCommands,
	        vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
	scheduler.DeferOperation([pending = std::move(pending)] { Compare(pending); });
}

} // namespace LaneSpillCompare

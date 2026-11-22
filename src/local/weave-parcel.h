#pragma once
// The weave's data (weave.h): what the reader cuts (a Parcel), what a worker makes of it (a Result) and what the GPU
// thread's turn reads back. The executor's objects (pairs, variants, sets, draw states, pipelines) are its private
// types, so they are held here as `const void*` and cast back inside RenderExecutor (weave-worker.inc, weave-turn.inc).
// A packet is named by its guest address (Pm4Execution's Position(): the guest's view of its buffer, whichever view is
// read).

#include "graphics/guest_gpu/hardwareContext.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace LocalVulkanRecording {
class Apart;
}

namespace Weave {

// The command processor's registers and the state a draw reads besides them: what a worker starts a parcel or a strand
// from (CommandProcessor::WeaveSnapshot, WeaveRestore), compared whole with the GPU thread's (WeaveSame: the HW
// structs' defaulted operator==, never a byte compare). Left out, as no packet a worker takes writes them: the CE/DE
// counters and constant RAM, the flip and submit id, the draw-run skip (the GPU thread's skip clears it).
struct Registers {
	Libs::Graphics::HW::Context      ctx;
	Libs::Graphics::HW::Context      saved_ctx;
	bool                             pushed = false;
	Libs::Graphics::HW::UserConfig   ucfg;
	Libs::Graphics::HW::Shader       sh;
	Libs::Graphics::HW::UserSgprType marker         = Libs::Graphics::HW::UserSgprType::Unknown;
	uint32_t                         index_type     = 0; // index type and size
	uint32_t                         index_size     = 0; // index buffer size
	uint64_t                         index_base     = 0;
	uint64_t                         draw_args      = 0; // draw indirect arguments' base
	uint64_t                         dispatch_args  = 0; // dispatch indirect arguments' base
	uint32_t                         num_instances  = 1;
	bool                             predicate_skip = false;

	bool operator==(const Registers&) const = default;
};

// A range of indices into one of a Result's vectors: [begin, end).
struct Range {
	uint32_t               begin = 0, end = 0;
	[[nodiscard]] uint32_t size() const noexcept { return end - begin; }
	[[nodiscard]] bool     empty() const noexcept { return end == begin; }
};

// A V# a worker took from a held entry as it is: its stage and buffer (in the stage's plan), where its five words are
// in Result::held_words, its range and that range's CPU modification epoch when taken.
struct HeldUse {
	uint8_t  stage   = 0;
	uint32_t buffer  = 0;
	uint32_t word    = 0;
	uint64_t address = 0, size = 0, cpu_epoch = 0;
};
// An index (or indirect argument, `args`) range a worker took from a live buffer, and its CPU epoch then; an epoch that
// can't match (~epoch) marks indices the CPU wrote, which the turn uploads first.
struct HeldBuffer {
	bool     args  = false;
	uint64_t begin = 0, end = 0, cpu_epoch = 0;
};
// A guest range a worker copied into its slab.
struct RingUse {
	uint64_t address = 0, size = 0;
};
// A buffer whose last use a worker's lookup owes (BufferCache::TouchLiveBuffer at the turn): its id (a BufferId's bits)
// and Vulkan handle (a VkBuffer's), and the range.
struct Touch {
	uint64_t buffer = 0, handle = 0;
	uint64_t address = 0, size = 0;
};
// A draw state's targets proven current read only, and the proof NativeXprTargetsCurrent would have written to it.
struct TargetsProof {
	uint64_t epoch = 0, erased = 0, checked = 0, span_begin = 0, span_end = 0;
};
struct Proof {
	const void*  state = nullptr;
	TargetsProof proof;
};

// A draw a worker made: one compact record, all the turn reads of it.
struct Made {
	const uint32_t* packet         = nullptr; // its first draw packet
	const uint32_t* packet_end     = nullptr; // the packet after the draw packets it took
	uint32_t        strand         = 0;
	uint32_t        commands_begin = 0, commands_end = 0; // its calls in its strand's apart buffer: [begin, end)
	bool        pass_begin = false; // its strand's buffer ended the last pass and began this draw's before its calls
	const void* pair       = nullptr;
	const void* set        = nullptr;
	const void* state      = nullptr;
	uint64_t    vs = 0, ps = 0; // its programs' addresses
	uint32_t    variant_index   = 0;
	const void* pipeline        = nullptr;
	uint64_t    pipeline_handle = 0; // the VkPipeline its bind recorded (0: it bound none)
	uint64_t    pair_proved     = 0; // the shader map generation the worker proved its pair's shaders at (0 none)
	uint64_t    mapping = 0, registration = 0, resolution = 0; // the epochs as its draw began
	std::array<uint64_t, 2> blocks {}; // each stage's block in the slab (device address; 0 none)
	Range                   held_uses, held_buffers, ring_uses, held_words, touched, placed, proofs;
	bool                    indexed    = false;
	uint64_t                index_base = 0, index_bytes = 0;
	bool                    cpu_indices = false; // its indices are a CPU-written range the turn uploads first
	bool                    gbuffer     = false; // KYTY_FRAMEGEN: its vertex shader is a G-buffer one
};

// What a worker left to the GPU thread instead of writing it: a set's validation, a draw state's targets, a set's
// transit under a draw state. Done at its stretch's first draw with today's functions, in TableDraw's order.
enum class NeedKind : uint8_t { Validate, Targets, Transit };
struct Need {
	NeedKind    kind       = NeedKind::Validate;
	const void* state      = nullptr;
	const void* set        = nullptr;
	uint64_t    capture    = 0; // the draw state's capture when noted
	uint32_t    first_made = 0; // the first made draw that needed it
};
// StretchDraws draw packets (or fewer, to a hole or the parcel's end): its first packet and its needs.
struct Stretch {
	const uint32_t* first_packet = nullptr;
	Range           needs;
};

// A strand's lists: what its draws hold, each once, tested at its first draw instead of draw by draw; and the stamps a
// spliced strand owes, each object once (an LRU touched in this order ends as the per-draw stamps leave it).
struct ListItem {
	const void* pair     = nullptr;
	const void* variant  = nullptr;
	const void* state    = nullptr;
	const void* set      = nullptr;
	const void* pipeline = nullptr;
	uint32_t    first    = 0; // the made draw (in the strand) that first holds it
};
enum class RangeKind : uint8_t { Held, HeldBuffer, Ring };
struct ListRange {
	RangeKind kind  = RangeKind::Held;
	uint32_t  first = 0; // the made draw (in the strand) that first holds it
	uint64_t  begin = 0, end = 0, cpu_epoch = 0;
};
struct Lists {
	std::vector<ListItem>    items;     // distinct (pair, variant, state, set, pipeline)
	std::vector<ListRange>   ranges;    // distinct ranges read, by kind
	std::vector<ListItem>    stamps;    // pairs, sets and draw states once each (the last draw's variant)
	std::vector<const void*> pipelines; // in first-bind order
	std::vector<uint32_t>    touches;   // Result::touched indices, in last-use order
	std::vector<const void*> placed;    // draw states whose targets were found in place, in last-use order
};

// The calls between two holes (or a stretch's edge): one apart buffer, spliced in parts.
struct Strand {
	uint32_t                                     first_made = 0, count = 0;
	uint32_t                                     stretch = 0;
	std::shared_ptr<LocalVulkanRecording::Apart> buffer; // null: discarded (a call didn't hold the marker)
	Range                                        items, ranges, stamps, pipelines, touches, placed; // in Result::lists
	uint64_t mapping = 0, registration = 0, resolution = 0; // its draws' least epochs
	uint32_t log_first = 0;                                 // its first made draw's entry in Result::log.draws
	// The registers at its first draw: at a strand that follows a hole, or is first seen past its first draw, the GPU
	// thread's are compared with them whole, and the strand is cut on a difference.
	Registers start;
};

// The register log: for each made draw, the 64-byte blocks of the register image (WeaveSnapshot's members, each
// from a 64-byte boundary) that changed since the previous made draw, with their values at its end; its draw packets
// and the draws they make; and the other packets before it by opcode (LiveCounters::AddPm4 at the skip). The GPU thread
// skips a spliced part's packets by applying the last write of each block (block_next). Made only while
// kyty_local_write_guard_mode != 1; without it (`on` false) the GPU thread passes the spliced packets one by one.
inline constexpr uint32_t LogBlockBytes = 64;
struct RegisterLog {
	struct Draw {
		uint32_t        blocks_end = 0, counts_end = 0; // its blocks and counts: from the previous draw's ends
		uint32_t        packets = 0, draws = 0;         // its draw packets, and the draws they make
		uint32_t        opcode = 0;                     // its draw packets' opcode
		const uint32_t* last   = nullptr;               // its last draw packet (the guest's)
	};
	struct Count {
		uint32_t opcode = 0, count = 0;
	};
	std::vector<Draw>     draws;
	std::vector<uint32_t> block_ids;
	std::vector<uint8_t>  block_bytes; // LogBlockBytes a block
	std::vector<uint32_t> block_next;  // each block's next write of the same id, UINT32_MAX none
	std::vector<Count>    counts;
	bool                  on = false;
};

// What a worker made of a parcel (its slot's; the vectors kept allocated from parcel to parcel).
struct Result {
	std::vector<Made>                          made;
	std::vector<HeldUse>                       held_uses;
	std::vector<HeldBuffer>                    held_buffers;
	std::vector<RingUse>                       ring_uses;
	std::vector<uint32_t>                      held_words; // five a held use
	std::vector<Touch>                         touched;
	std::vector<Touch>                         touched_unique; // each buffer once: touched at the open
	std::vector<const void*>                   placed;
	std::vector<Proof>                         proofs;
	std::vector<Need>                          needs;
	std::vector<Stretch>                       stretches;
	std::vector<Strand>                        strands;
	Lists                                      lists;
	RegisterLog                                log;
	std::vector<std::pair<uint64_t, uint64_t>> pages;   // the byte ranges read, [begin, end), sorted and merged
	std::vector<const void*>                   objects; // the made draws' pairs, sets and draw states, sorted, unique
	std::vector<uint64_t>                      copies_used; // the linear copies read through (their records)
	// The clocks as the worker began: the scout's write clock, the free log's, the shader map generation, the texture
	// cache's image start epoch, the BDA table generation, the pipelines evicted.
	uint64_t write_clock = 0, freed_clock = 0, shader_generation = 0, image_start_epoch = 0, bda_generation = 0;
	uint64_t evictions = 0;
	bool     spoiled   = false; // stopped inside a packet, or a packet it took failed: never opened
	bool     lost      = false; // calls lost: never opened
	// The slab its blocks and copies went into: its worker's ring (-1 none), where, and the bytes used.
	struct Slab {
		int32_t  ring   = -1;
		uint64_t offset = 0, used = 0;
	} slab;
};

// A parcel's slot. Free, then the reader's (Filling) until published (Cut); a worker takes it (Taken) and finishes it
// (Done), or the GPU thread passes it first (Skipped); the GPU thread passes each parcel once, and whoever comes
// last, it or the worker, releases it (Releasing, then Free).
struct Parcel {
	enum State : uint32_t { Free, Filling, Cut, Taken, Done, Skipped, Releasing };
	std::atomic<uint32_t> state {Free};
	uint64_t              serial     = 0;       // its number among the parcels published
	uint64_t              generation = 0;       // the reader's syncs taken before it was cut
	uint64_t              sequence   = 0;       // its submission's (the scout's numbering)
	uint64_t              copies_end = 0;       // the linear-copy records made before it
	const uint32_t*       guest      = nullptr; // its submission's commands as the guest has them
	const uint32_t*       first = nullptr; // its first draw packet, and the packet after its last (guest addresses)
	const uint32_t*       end   = nullptr;
	std::shared_ptr<const std::vector<uint32_t>> commands; // the reader's copy of the submission (pooled)
	Registers                                    start;    // the registers at its first draw
	uint32_t                                     calls = 0, draws = 0;
	std::vector<uint32_t>   cuts;            // its stretches' starts after the first, as dword offsets into `commands`
	std::atomic<uint64_t>   progress {0};    // the dwords its worker has reached
	std::atomic<bool>       dropped {false}; // the GPU thread makes the rest; a worker stops at its next draw
	std::atomic<int64_t>    taken_ns {0};    // when a worker took it (steady clock)
	std::shared_ptr<Result> result;
};

} // namespace Weave

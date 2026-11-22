// --- Table draws (GPU-driven XPR, docs/GPU-DRIVEN-XPR.md) ------------------------------
//
// Included by renderDraw.cpp after native-xpr.inc (local Vulkan recording builds only).
//
// KYTY_TABLE_XPR=1: the XPR draws of a shader pair whose programs allow table mode
// (ShaderRecompiler::IR::EnterTableMode) need no record. For each draw the renderer evaluates the programs' table
// plans (IR::TablePlan): the slots their shaders read and the T#/S# words, from guest memory as it is when the draw
// is translated (as the normal path's SRT evaluation, which the game's command buffers patch between draws); the
// T#/S# words pick the pair's image set - a descriptor set with the images and the samplers, proven current once
// a frame - and each V# is resolved to its buffer's device address (cached while its buffer and region stay the
// same). The slots and buffer addresses go into a block in the table upload ring whose device address is pushed
// with the user data: the shaders read it, no flattened SRT binding, no buffer descriptors.
//
// The normal path, asked to store (TableStore), compiles the pair's table mode programs and the pipeline of each
// draw state, and captures the draw state (NativeXprCaptureState).
//
// KYTY_TABLE_XPR=2 also continues a run (TableContinue): a draw the observer calls clean (no register but user
// data and index state changed) right after a table draw whose emission is the last one in the command buffer
// has the same pair and draw state, with its pipeline and dynamic state still bound.

// Roles 1.. (the speculation threads': RenderExecutor::Role) of a pair's or a set's per-role values, in a block of its
// own allocated at a speculation thread's first use (KYTY_SPECULATE=4); role 0 (the GPU thread) keeps its value inline,
// so in mode 0 the pair and the set are the base's fields and size, and nothing is allocated. Moved with its owner.
template <typename T>
struct TableRoleSide {
	std::atomic<T*> block {nullptr};
	TableRoleSide() = default;
	TableRoleSide(TableRoleSide&& other) noexcept: block(other.block.exchange(nullptr)) {}
	TableRoleSide& operator=(TableRoleSide&& other) noexcept {
		delete block.exchange(other.block.exchange(nullptr));
		return *this;
	}
	TableRoleSide(const TableRoleSide&)            = delete;
	TableRoleSide& operator=(const TableRoleSide&) = delete;
	~TableRoleSide() { delete block.load(); }
	[[nodiscard]] T* Peek() const noexcept { return block.load(std::memory_order_acquire); }
	T& Get() {
		return Get([](T&) {});
	}
	// (`init` fills a block before it is published.)
	template <typename Init>
	T& Get(Init&& init) {
		T* at = block.load(std::memory_order_acquire);
		if (at != nullptr) return *at;
		auto* made = new T;
		init(*made);
		if (block.compare_exchange_strong(at, made, std::memory_order_acq_rel)) return *made;
		delete made;
		return *at;
	}
};

// A draw state's pipeline.
struct RenderExecutor::TableVariant {
	uint64_t                       state          = 0;
	const PipelineCache::Pipeline* pipeline       = nullptr; // null while a worker compiles it
	bool                           fetch_embedded = false;   // the vertex program fetches (a direct draw's offsets)
	bool                           refused        = false;   // another specialization than the pair's programs
	// The pixel program's export mapping under this state (GetGraphicsPrograms).
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> mapping {};
};

// Words a draw writes whole after resizing (TableEvaluate's T#/S# words): resize leaves the new ones uninitialized. (With
// std::allocator it zeroed them first, a memset call per draw: TableEvaluate 7.1 -> 6.8% of the GPU thread at Latria.)
template <typename T>
struct TableUninitialized: std::allocator<T> {
	using std::allocator<T>::allocator;
	template <typename U>
	struct rebind {
		using other = TableUninitialized<U>;
	};
	template <typename U>
	void construct(U* at) noexcept {
		::new (static_cast<void*>(at)) U;
	}
	template <typename U, typename... Args>
	void construct(U* at, Args&&... args) {
		::new (static_cast<void*>(at)) U(std::forward<Args>(args)...);
	}
};
using TableWords = std::vector<uint32_t, TableUninitialized<uint32_t>>;

// The descriptor set of one choice of images and samplers (the T#/S# words of every stage).
struct RenderExecutor::TableImageSet {
	struct Image {
		ImageId                 id;
		uint64_t                serial = 0;
		vk::ImageLayout         layout = vk::ImageLayout::eUndefined;
		ImageSubresourceRange   range;
		TextureCache::ImageDesc desc;
		vk::AccessFlags2        access = vk::AccessFlagBits2::eShaderRead;
		bool                    target  = false; // a target of the draw state the set was made under (its access)
		// A target the shaders sample (KYTY_TABLE_TARGET_READ): the shader access added to the target's (in `access`).
		vk::AccessFlags2        target_read {};
		bool                    storage = false; // the shader writes it: GPU-modified at every use (TableTransit)
		bool                    whole   = false; // bound both as a texture and as storage: the whole image's state
		// The main executor's proof of its use (a target, or a texture without DCC found current): the image's validity
		// then, which the proof holds with while no registration stamped the view's pages since (TableValidateSet).
		bool                    proven   = false;
		uint64_t                validity = 0;
	};
	TableWords            words; // without the LOD statistics fields (NativeXprLodStatsMask)
	// spec_worker_images: its slots holding a null image until the commit completes them (Speculation::Before).
	struct PendingImage {
		uint32_t                binding = 0, element = 0;
		vk::DescriptorType      type    = vk::DescriptorType::eSampledImage;
		TextureCache::ImageDesc desc;
		vk::ImageLayout         layout  = vk::ImageLayout::eShaderReadOnlyOptimal;
		// Its view has metadata: the commit re-tests FindTexture's metadata step before it (MetadataBindNoop; the segment
		// stops when the step would change anything: an image found without the view's metadata would have it adopted in
		// Before) and notes the use as the owner's build does (NoteMetaUse).
		bool                    metadata = false;
	};
	std::vector<PendingImage> pending_images;
	// A speculation's proposal (spec_worker_sets): the Spec::State::id of the translation that built it (0: the owner's),
	// set before the set is reachable and never changed; `published` once the commit puts it in the catalog (under the
	// catalog's write). Until then only that translation takes it from a role's last / previous set (TableSet): its
	// pending slots are filled only if that translation commits the segment that proposed it (a worker's next translation
	// bound a set its first one then forgot unfilled: null image slots).
	uint64_t              proposer = 0;
	size_t                proposed_segment = SIZE_MAX; // (the proposer's open segment then: spec_set_cpu_dirty_stop)
	std::atomic<bool>     published {false};
	vk::DescriptorSet     set  = nullptr;
	vk::DescriptorPool    pool = nullptr;
	std::vector<Image>    images;
	// BufferCache::BdaTableGeneration when the set got the BDA page table: made again once the table grew.
	uint64_t              bda_generation = 0;
	// Proven current in that frame under those registrations, by role (RenderExecutor::Role: [0] also the frame it was
	// made or refused in); used in that frame (by any executor).
	uint64_t validated_frame  = UINT64_MAX; // (role 0's; roles 1..: roles)
	uint64_t resolution_epoch = 0;
	struct Roles1 {
		std::array<uint64_t, Roles - 1> validated_frame, resolution_epoch {};
		Roles1() { validated_frame.fill(UINT64_MAX); }
	};
	TableRoleSide<Roles1> roles;
	// The worker roles' epochs start from role 0's proof (RoleZeroEpoch). At 0 every registration over a view's pages
	// counted, the image's own included, so a set sampling a depth view never validated in a worker (IsSampledDepthCurrent)
	// and every draw with it was a hole: 'table: set validation', ~24-42 a frame settled at three of the owner's spots.
	Roles1& Side() {
		return roles.Get([this](Roles1& side) { side.resolution_epoch.fill(RoleZeroEpoch()); });
	}
	[[nodiscard]] uint64_t RoleZeroEpoch() { return std::atomic_ref<uint64_t>(resolution_epoch).load(std::memory_order_relaxed); }
	uint64_t& ValidatedFrame(uint32_t role) { return role == 0 ? validated_frame : Side().validated_frame[role - 1]; }
	uint64_t& ResolutionEpoch(uint32_t role) { return role == 0 ? resolution_epoch : Side().resolution_epoch[role - 1]; }
	std::atomic<uint64_t>       used_frame {0};
	// The texture cache's flip count (TextureCache::GcTick) and frame (CurrentFrame) when the main executor last checked
	// every image in full, which touches their uses (the LRU, the last use): again within TouchTicks of either (an image
	// unused for 32 frames may go to an overlap, for 600 flips to the collector).
	uint64_t                  touched_tick  = 0;
	uint64_t                  touched_frame = 0;
	static constexpr uint64_t TouchTicks    = 8;
	// The memory its images' views span (from the lowest to the highest byte): registrations elsewhere reach none of
	// its proofs (TableValidateSet). 0, 0: not known yet.
	uint64_t                  span_begin = 0, span_end = 0;
	// TextureCache::ImagesEpoch after the main executor last transited its images (TableTransit) for a draw under
	// transit_state (null: a dispatch): they stay in place while it does not move (UINT64_MAX: not yet).
	uint64_t                  transit_epoch = UINT64_MAX;
	const NativeXprDrawState* transit_state = nullptr;
	// The sum of its images' serials then (KYTY_TABLE_SERIALS; UINT64_MAX: none), and transit_state's capture then.
	uint64_t                  transit_serials  = UINT64_MAX;
	uint64_t                  transit_capture  = 0;
	// The images' ids alone, and whether one is a target of the state it was made under: what the serial sums and
	// TableSetApart read (an Image holds an ImageDesc, lines of it).
	std::vector<ImageId>      image_ids;
	bool                      holds_target = false;
	// An image bound both as a texture and as storage that the shader writes (the deferred decals' G-buffer): with
	// KYTY_DRAW_READ_AFTER_WRITE=1 a draw with the set transits it every time (TableTransit), never in place, ordered
	// after the last draw's write (Image::GetBarriers, issue #315).
	bool                      read_after_write = false;
	// The words select another specialization, or an image no set can hold: the draws take the native path
	// (TableResolveSet tries again after TableImageSet::RetryFrames).
	bool                  refused = false;
	static constexpr uint64_t RetryFrames = 300;
	// KYTY_TABLE_LEARN_PORTABLE=2: a retry that found its program still translating waits for that translation (asked
	// again at every draw until then, it went pending hundreds of times a retry).
	std::shared_ptr<const std::atomic<bool>> retry_wait;
	// A dispatch's words that select another specialization of the pair's program whose table form is the same
	// (TableInterchangeable: its images of other dimensions or formats): that program and its pipeline, which the set
	// is made for (null: the pair's).
	const ShaderRecompiler::IR::CompiledShaderInfo* program  = nullptr;
	const PipelineCache::Pipeline*                  pipeline = nullptr;
	// (Mode 4 only, after the fields every mode reads, whose offsets they leave as they were.)
	// The targets of every draw state its translation bound it under while it had pending slots (TableDraw; that
	// translation's thread only, until its commit reads them: TableCompletePendingImages).
	std::vector<ImageId>      pending_targets;
	// A worker built it with a storage image resolved read only (its commit's check notes the GPU write the owner's
	// FindTexture would have, once each such image is still as the worker found it).
	bool                      worker_storage = false;
};

// Two specializations of a program the table way alike but for their images' and samplers' bindings: the same slots,
// buffers (and their decoded formats), descriptor words, page table reads and shader data. The words TableEvaluate
// reads and the blocks TableBlocks makes for one serve the other.
static bool TableInterchangeable(const ShaderRecompiler::IR::CompiledShaderInfo& a,
                                 const ShaderRecompiler::IR::CompiledShaderInfo& b) {
	const auto& x = a.bindings;
	const auto& y = b.bindings;
	return a.table_mode && b.table_mode && a.stage == b.stage && a.shader_hash == b.shader_hash &&
	       a.user_data_base == b.user_data_base && a.table_plan == b.table_plan && a.info.buffers == b.info.buffers &&
	       a.info.images.size() == b.info.images.size() && a.info.samplers.size() == b.info.samplers.size() &&
	       a.bda_read_plan == b.bda_read_plan && x.push_data_start_dword == y.push_data_start_dword &&
	       x.memory_offset_dword == y.memory_offset_dword && x.memory_offset_count == y.memory_offset_count &&
	       x.buffer_word_count == y.buffer_word_count && x.lod_stats_count == y.lod_stats_count &&
	       x.table_block_dwords == y.table_block_dwords && x.user_data_registers == y.user_data_registers;
}

// A stage's flat form as TableRuns makes it (TablePack packs it into the pair): every operand an index into one of
// three arrays (the stage's user data, its block's slots, the stage's constants).
struct RenderExecutor::TableFlatStage {
	struct Operand {
		uint32_t index = 0;
		uint32_t base  = 2; // 0: user data, 1: slots, 2: constants
	};
	struct Run {
		std::array<Operand, 3> address; // low, high, offset
		int32_t                immediate = 0;
		uint32_t               first = 0, count = 0;
		Operand                records; // a bounded run's (TablePlan::Slot::bounded)
		bool                   bounded = false;
	};
	std::vector<Run>      runs;         // the runs in level order
	std::vector<uint32_t> level_ends;   // the run after each level
	std::vector<Operand>  image_words;  // every image's words, then every sampler's
	std::vector<uint32_t> image_counts;
	std::vector<Operand>  buffer_words; // every buffer's four V# words (TableBlocks)
	std::vector<uint32_t> constants;
	uint32_t              user_words = 0; // the user data it reads: below this
};

struct RenderExecutor::TablePair {
	// Evaluated slots at consecutive addresses (the same base, immediates 4 apart): one read, at the address of the
	// first. Ordered by level: a run whose address is made of slots is read after the runs of those slots.
	struct Run {
		ShaderRecompiler::IR::TablePlan::Slot read;
		uint32_t                              first = 0, count = 0, level = 0;
	};
	uint64_t                                                    vs = 0, ps = 0; // (a dispatch's: its shader, 0)
	// The GS user SGPR the CP writes an indirect draw's start instance into (START_INST_LOC, -1: none): the vertex
	// program is translated for it (ShaderVertexInputInfo::start_instance_sgpr).
	int32_t                                                     start_instance = -1;
	bool                                                        refused  = false;
	bool push_fit = false; // KYTY_SPEC_PAIR_PUSH_FIT: its programs the retry's (the reduced pixel form): spec_pair_push_fit_draws_n
	const char* special = nullptr; // a dispatch's: the normal path's special path that takes it (refused)
	std::array<const ShaderRecompiler::IR::CompiledShaderInfo*, 2> programs {}; // vertex, pixel (null: inactive)
	std::array<std::vector<Run>, 2>                             runs;
	// TableEvaluate's form of a stage's plan (TableRuns, TablePack): every operand an index into one of three arrays (the
	// stage's user data, its block's slots, the stage's constants), so a draw's evaluation branches on no operand kind.
	// Both stages' arrays in one block, each operand 16 bits (its array in the top two, its index in the low 14), each
	// run 16 bytes: a pair a draw switches to is cold, and its arrays were a dozen allocations of 8-byte operands, as
	// many cold lines on as many pages (TablePrefetchArrays). Same process, 10 rounds: Stonefang standing render 8.94 ->
	// 8.75 ms, Latria 13.47 -> 13.31.
	struct PackedRun {
		std::array<uint16_t, 3>   address {}; // low, high, offset
		uint16_t                  records = NotBounded;
		int32_t                   immediate = 0;
		uint16_t                  first = 0, count = 0;
		static constexpr uint16_t NotBounded = 0xffff;
	};
	struct PackedStage {
		std::span<const PackedRun> runs;
		std::span<const uint16_t>  level_ends, image_words, buffer_words;
		std::span<const uint8_t>   image_counts;
		std::span<const uint32_t>  constants;
		uint32_t                   user_words = 0;
		bool                       valid      = false; // (else an index or a count past its packed width)
	};
	std::array<PackedStage, 2>      packed;
	std::unique_ptr<std::byte[]>    packed_block;
	const std::byte*                packed_at    = nullptr; // (the block's first 64-byte line)
	uint32_t                        packed_bytes = 0;
	std::vector<TableVariant>                                   variants;
	uint32_t                                                    variant_hint = 0; // the one TableTry found last
	uint64_t                                                    store_frame = 0, store_state = 0; // the last request
	std::unordered_map<uint64_t, std::unique_ptr<TableImageSet>, HashedKey> sets; // by T#/S# words
	TableImageSet*                                              last_set = nullptr; // (role 0's; roles 1..: roles)
	// The set before the last, by role (TableSet asks it before the map: a pass's draws of one pair alternate between
	// two materials' sets). Cleared with last_set.
	TableImageSet*                                              previous_set = nullptr;
	// ShaderMapGeneration under which the shaders at vs and ps were proven those the programs were compiled
	// from (the game registers shaders often; code replaced at an address resets the pair), by role.
	uint64_t                                                    generation = UINT64_MAX;
	struct Roles1 {
		std::array<TableImageSet*, Roles - 1> last_set {}, previous_set {};
		std::array<uint64_t, Roles - 1>       generation;
		Roles1() { generation.fill(UINT64_MAX); }
	};
	TableRoleSide<Roles1> roles;
	TableImageSet*& LastSet(uint32_t role) { return role == 0 ? last_set : roles.Get().last_set[role - 1]; }
	TableImageSet*& PreviousSet(uint32_t role) { return role == 0 ? previous_set : roles.Get().previous_set[role - 1]; }
	uint64_t&       Generation(uint32_t role) { return role == 0 ? generation : roles.Get().generation[role - 1]; }
	// Every role's last and previous pointer to `set` dropped (a worker set that goes unpublished: spec_worker_images).
	void ForgetSet(const TableImageSet* set) noexcept {
		if (last_set == set) last_set = nullptr;
		if (previous_set == set) previous_set = nullptr;
		if (auto* side = roles.Peek()) {
			for (auto& at: side->last_set)
				if (at == set) at = nullptr;
			for (auto& at: side->previous_set)
				if (at == set) at = nullptr;
		}
	}
	// Every role's last and previous set dropped (a set replaced or retired).
	void ClearSets() noexcept {
		last_set = previous_set = nullptr;
		if (auto* side = roles.Peek()) {
			side->last_set.fill(nullptr);
			side->previous_set.fill(nullptr);
		}
	}
	// A dispatch's (TableDispatch): its program's input and pipeline, and the normal path's dispatches seen.
	std::unique_ptr<ShaderComputeInputInfo> compute;
	ShaderProgram                           compute_program;
	const PipelineCache::Pipeline*          pipeline  = nullptr;
	uint32_t                                sightings = 0;
	// Per stage, the buffers whose V#s were seen in another format than the programs decode (bits by buffer): the
	// programs are compiled again with those buffers' formats decoded at run time (`respecialize`, TableReset).
	std::array<uint32_t, 2> portable {};
	bool                    respecialize = false;
	// A translation or a pipeline compile the pair's next store needs (TableStore, TableDispatch): until it is done,
	// its draws keep the native path without asking for a store (a request a frame each kept the store budget).
	std::shared_ptr<const std::atomic<bool>> waiting;
	// A dispatch's: the pair of the table dispatch that came after this one's the last time (the main executor's):
	// TablePrefetchNext loads its lines while this one is made.
	TablePair* next = nullptr;
	// The frame the main executor last looked it up (TableSweepPairs lets one go that none used for a while).
	uint64_t used_frame = 0;
	// The version of the view the scout evaluates its draws from (Scout::View, TableScoutPublish; 0: none).
	uint64_t scout_version = 0;
};

struct RenderExecutor::TableXpr {
	using Pair    = TablePair;
	using Variant = TableVariant;
	std::unordered_map<uint64_t, Pair, HashedKey> pairs;      // by shader addresses
	std::unordered_map<uint64_t, Pair, HashedKey> dispatches; // by the compute stage's registers (TableDispatchKey)
	// Pairs no draw or dispatch used for PairKeepFrames leave (TableSweepPairs: every PairSweepFrames, past PairsKept):
	// they are keyed by the shaders' addresses, and a world change loaded its shaders elsewhere (~1000 new pairs, ~8 KB
	// each with their runs). Pairs do not move: a pointer stays valid until then, when every one held is dropped.
	static constexpr uint64_t PairSweepFrames = 600, PairKeepFrames = 3600;
	static constexpr size_t   PairsKept       = 2048;
	uint64_t                  pairs_swept     = 0;
	// The pairs found last, direct-mapped by key.
	static constexpr size_t PairCacheEntries = 4096; // (kyty_local_table_mru bit 1: else its first 256 only)
	std::array<std::pair<uint64_t, Pair*>, PairCacheEntries> pair_cache {};
	// The pair TableTry found last, by its key (consecutive draws mostly keep their shaders: no hash of the key, ~0.6% of
	// the render thread at 1-1 standing, nor lookup then).
	std::array<uint64_t, 3> found_key {};
	Pair*                   found_pair = nullptr;
	Pair*                   previous_dispatch = nullptr; // the pair of the last table dispatch (TablePrefetchNext)
	// The same for TableDispatch's dispatches (the main executor's own: its catalog), by TableDispatchWords.
	std::array<uint64_t, 4> found_dispatch_key {};
	Pair*                   found_dispatch = nullptr;
	uint64_t collected  = 0;                           // the frame unused image sets were last dropped
	// Moves when a pair is reset or a set leaves (sets are replaced): an executor reading this catalog drops its last
	// pair, state and set (RenderExecutor::m_catalog) when it moved since it saw it (seen_epoch, its own table's).
	uint64_t epoch = 0, seen_epoch = 0;
	// The next prepared draw of the pair under this state stores (TableStore); the frame's store requests so far.
	Pair*    store       = nullptr;
	uint64_t store_state = 0, store_frame = 0;
	uint32_t store_requests = 0;
	// The draw's T#/S# words (the set's key) and pixel stage LOD statistics words; its runs' addresses.
	TableWords                           words;
	std::vector<uint32_t>                lod_stats;
	std::vector<uint64_t>                addresses;
	// 16 KiB guest pages seen mapped (direct-mapped by page), while the mapping epoch is the same.
	std::array<uint64_t, 256> mapped {};
	uint64_t                  mapping_epoch = UINT64_MAX;
	// The last table draw (TableContinue): its pair, the index of its variant and its draw state.
	Pair*                     last_pair    = nullptr;
	size_t                    last_variant = 0;
	const NativeXprDrawState* last_state   = nullptr;
	const TableImageSet*      last_set     = nullptr; // the set the last emission bound
	// KYTY_TABLE_MEMO: by pair and state key (direct-mapped), what a table draw of them drew with last: the variant's
	// index, the draw state and the image set. Entries hold while memo_epoch does not move: every set retirement
	// (NativeXprRetireSet), state sweep (TableForgetStates), pair sweep and reset moves it.
	struct Memo {
		const Pair*               pair    = nullptr;
		uint64_t                  key     = 0;
		uint64_t                  epoch   = 0;
		const NativeXprDrawState* state   = nullptr;
		TableImageSet*            set     = nullptr;
		uint32_t                  variant = 0;
	};
	static constexpr size_t     MemoEntries = 4096;
	std::array<Memo, MemoEntries> memo {};
	uint64_t                    memo_epoch  = 1;
	TableImageSet*              memo_set    = nullptr; // TableSet's first try for the draw being made
	// KYTY_TABLE_MEMO_VERIFY: memo hits counted (every Nth checked), and whether TableSet checks the memo's set.
	uint64_t                    memo_verify_count = 0;
	bool                        memo_verify       = false;
	[[nodiscard]] Memo& MemoAt(const Pair* pair, uint64_t key) {
		return memo[((reinterpret_cast<uintptr_t>(pair) * 0x9e3779b97f4a7c15ull) ^ (key * 0xc2b2ae3d27d4eb4full)) >> 52u];
	}
	uint64_t                  images_epoch = 0;       // TextureCache::ImagesEpoch after it
	// The game buffer the last arguments came from, and the indices' by their address (each mesh has its own): used
	// again while no registration changed and the guest wrote none of its region (ObtainBuffer then returns the same
	// buffer; it streams CPU-dirty ranges).
	struct HeldBuffer {
		uint64_t begin = 0, end = 0, cpu_epoch = 0, registration = 0;
		Buffer*  buffer = nullptr;
	};
	HeldBuffer                   args;
	std::array<HeldBuffer, 8192> indices;
	// A V#'s buffer as the shaders read it (direct-mapped by the V#'s words, one cache line each): its device address
	// and size in dwords (0: none; clamped to the mappings), while no buffer covering its tracker regions was registered
	// or unregistered since `registration` (the registration epoch it was last proven at: a region stamped later moved
	// a buffer) and the guest wrote none of its range (the tracker regions' epoch, else the range's dirty pages). One
	// whose proof also needs the image starts and the mappings (formatted: an image may start at it; clamped: a mapping
	// may extend it) holds while `special` is the table's special generation (0: no such need). (Buffer registrations
	// anywhere used to drop them all: 3800 resolves a frame walking at 1-1, 1000 standing.)
	// One holding a copy in the table upload ring of a range the guest wrote (RingCopy, made or shared for the main
	// executor) has `registration` RingCopyMark | the observation epoch it is shared in and `cpu_epoch` the range's
	// tracker regions' epoch then: the use takes it as RingCopy's terms allow, without resolving the V# again.
	struct alignas(64) ResolvedBuffer {
		std::array<uint32_t, 4> words {};
		uint32_t                dwords = 0, special = 0;
		vk::DeviceAddress       device = 0;
		uint64_t                address = 0, size = 0, cpu_epoch = 0, registration = 0;
	};
	static constexpr uint64_t RingCopyMark = uint64_t {1} << 63u;
	static constexpr size_t           ResolvedEntries = Scout::ResolvedEntries; // (the scout's tickets index it)
	std::unique_ptr<ResolvedBuffer[]> resolved = std::make_unique<ResolvedBuffer[]>(ResolvedEntries);
	// A copy in the table upload ring of a small range the guest wrote (TableBlocks reads such a range from a copy,
	// none held: the guest writes its pages unseen), direct-mapped by the range: the main executor's uses share it
	// while no point the guest could learn that GPU work ran at passed (PageManager::ObservationEpoch: until then the
	// guest writes nothing that work reads), in the same submission (`tick`: the ring keeps the copy for it), with
	// the range's tracker regions' epoch unchanged (`cpu_epoch`: it moves when pages become CPU-dirty, when the
	// emulator writes dirty pages, MemoryTracker::NoteHostWrite, and when GPU-written pages are given back) and no
	// page of it GPU-written at the use. A frame's constants are read by many draws: in the Latria spin ~2950 copies,
	// 11.8 MB a frame, before; ~1140 and 4.6 MB with the shared ones.
	struct RingCopy {
		uint64_t          address = 0, size = 0, observation = 0, tick = 0, cpu_epoch = 0;
		vk::DeviceAddress device = 0;
	};
	static constexpr size_t           RingCopies = 4096;
	std::unique_ptr<RingCopy[]>       ring_copies = std::make_unique<RingCopy[]>(RingCopies);
	// (Mode 4) a speculation's copies, shared on the same terms within one translation (Spec::State::id), segment and
	// wait count (State::waits and covered_waits: every wait it passes moves it) while no Rollback took notes back since
	// (State::rollbacks). TableBlocks.
	struct SpecRingCopy {
		RingCopy copy;
		uint64_t id = 0, rollbacks = 0, waits = 0;
		size_t   segment = 0;
	};
	uint32_t                          special_generation = 1; // moves with the image starts and the mappings
	// Addresses no image starts at, each with the TextureCache::StartEpoch it was proven at (only an address gaining
	// its first image moves that epoch): TableBlocks asks the texture cache (a lock and a map lookup) again only then.
	static constexpr size_t                           NoImageStartEntries = 1024;
	std::array<std::pair<uint64_t, uint64_t>, NoImageStartEntries> no_image_start {};
	uint64_t                          resolved_start = UINT64_MAX, resolved_mapping = UINT64_MAX;
	// Each stage's V#s and their entries (TableGather), gathered for `gathered` (the pair of the draw or dispatch being
	// made, null once TableBlocks took them).
	std::array<std::vector<std::pair<std::array<uint32_t, 4>, ResolvedBuffer*>>, 2> entries;
	const TablePair*                                                                gathered = nullptr;
	// A written V#'s range clamped to the mappings, and (`image_checked`) that no image starts at it, while `special`
	// is the special generation (mappings and image starts as then): obtained as written at every use, its proofs
	// (a memory lock and a texture cache lock and lookup) once (direct-mapped by the words).
	struct WrittenRange {
		std::array<uint32_t, 4> words {};
		uint32_t                special = 0;
		bool                    image_checked = false;
		uint64_t                address = 0, size = 0;
	};
	static constexpr size_t                WrittenEntries = 1024;
	std::array<WrittenRange, WrittenEntries> written;
	// The draw's blocks (per stage): its slots (IR::TablePlan::BlockDwords), then its buffers.
	std::array<std::vector<uint32_t>, 2> blocks;
	// An indirect image's candidates for the draw or dispatch being made (per stage), in the order its set binds them
	// (TableEvaluate: the set's key; TableResolveSet), and the enumeration's scratch.
	std::array<std::vector<ShaderRecompiler::IR::DescriptorValue>, 2> indirect;
	ShaderRecompiler::IR::AddressProbeCandidates                      probe;
	// (Mode 4 only, after the fields every mode reads per draw, whose offsets they leave as they were.)
	// A speculation's last table draw here, by its translation (Spec::State::id), segment, chain head, draw state's
	// capture (a state re-captured in place keeps its address), the images epoch read before its checks, and its overlay
	// and open-write generations after it (id 0: none to match). TableDraw's spec_in_place.
	struct SpecPlace {
		uint64_t           id = 0, images = 0, overlay = 0, open = 0, capture = 0;
		size_t             segment = 0;
		const Spec::State* chain   = nullptr;
	} spec_place;
	std::unique_ptr<SpecRingCopy[]> spec_ring_copies; // (SpecRingCopy above; made at a speculation's first copy)
	// (KYTY_SPEC_PAIR_DENSE_INDIRECT: per stage, each indirect image's count of them in `indirect`, in plan order: a
	// program with several dense roots has their candidates one after another there. Written for a stage with an indirect
	// image only (TableEvaluate, TableScoutApply), the one TableResolveSet reads it for.)
	std::array<std::vector<uint32_t>, 2> indirect_counts;
};
// KYTY_TABLE_MEMO: a descriptor set left service (NativeXprRetireSet): a memo entry may hold it.
void RenderExecutor::TableMemoRetired() {
	if (m_table_xpr) ++m_table_xpr->memo_epoch;
}
// KYTY_TABLE_MEMO_VERIFY: a memo hit checked; `differ`: 1 the variant, 2 the draw state, 4 the image set the full path
// found were not the memo's.
static void TableMemoChecked(uint32_t differ, uint64_t vs, uint64_t ps, uint64_t key) {
	if (differ == 0) return;
	LiveCounters::Add(LiveCounters::TableMemoMismatch);
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 8)
		std::printf("TABLE_MEMO_MISMATCH vs=0x%llx ps=0x%llx key=0x%llx variant=%d state=%d set=%d\n",
		            static_cast<unsigned long long>(vs), static_cast<unsigned long long>(ps),
		            static_cast<unsigned long long>(key), (differ & 1u) != 0 ? 1 : 0, (differ & 2u) != 0 ? 1 : 0,
		            (differ & 4u) != 0 ? 1 : 0);
}

#include "weave-draw.inc"

// A draw pair's key as TableTry hashes it (the scout's view directory too).
template <typename Pair>
static uint64_t TableScoutHash(const Pair& pair) {
	const std::array<uint64_t, 3> key {pair.vs, pair.ps, static_cast<uint32_t>(pair.start_instance)};
	return XXH3_64bits(key.data(), sizeof(key));
}

// The pair's programs and what was made for them leave (its shaders changed, or a V#'s format did:
// TablePair::portable); the next draw or dispatch compiles them again.
void RenderExecutor::TableReset(TablePair& pair) {
	if (pair.scout_version != 0) Scout::Retire(TableScoutHash(pair), pair.scout_version);
	auto& table = *m_table_xpr;
	EXIT_IF(Role() != 0);
	++table.memo_epoch;
	// (Its callers hold the weave's workers outside a draw. The pair is rebuilt in place: what a worker kept of it
	// goes.)
	if (Weave::Active()) Weave::NoteFreed(&pair);
	for (auto& [words, set]: pair.sets) {
		if (Weave::Active()) Weave::NoteFreed(set.get());
		NativeXprRetireSet(set->pool, set->set);
	}
	if (table.store == &pair) table.store = nullptr;
	if (table.last_pair == &pair) table.last_pair = nullptr;
	table.last_set = nullptr;
	++table.epoch;
	// (The key's start instance SGPR too: without it every later draw of the pair failed TableTry's key check.)
	pair = {.vs = pair.vs, .ps = pair.ps, .start_instance = pair.start_instance, .generation = pair.generation,
	        .roles = std::move(pair.roles), .sightings = pair.sightings, .portable = pair.portable};
	pair.ClearSets();
}

// Draw states left (NativeXprSweepStates): the last table draw's and the image sets' transits under one are forgotten,
// and the executors reading this catalog drop what they hold of it (epoch).
void RenderExecutor::TableForgetStates() {
	auto* table = m_table_xpr.get();
	if (table == nullptr) return;
	++table->memo_epoch;
	table->last_pair  = nullptr;
	table->last_state = nullptr;
	table->last_set   = nullptr;
	++table->epoch;
	for (auto* pairs: {&table->pairs, &table->dispatches})
		for (auto& [key, pair]: *pairs)
			for (auto& [words, set]: pair.sets)
				if (set->transit_state != nullptr) {
					set->transit_state   = nullptr;
					set->transit_epoch   = UINT64_MAX;
					set->transit_serials = UINT64_MAX;
				}
}

// (Mode 4, TableWorkerEntry below) This draw's topology and primitive restart, as the normal path derives them
// (GetDrawTopology; ResolvePrimitiveRestart less its scan, which only stops the emulator when a custom reset value is in the
// indices): the draw kind and the index size are not in the state key or the context hash. False: not derivable here
// (refused).
static bool WorkerEntryDrawTopology(const CommandBuffer& buffer, bool indexed, vk::IndexType index_type,
                                    vk::PrimitiveTopology& topology, bool& restart) {
	restart = false;
	if (!GetDrawTopology(buffer.GetUserConfig(), !indexed, topology)) return false;
	if (!indexed) return true; // (DrawAuto: no restart)
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	if ((control & ~0x3u) != 0) return false;
	if ((control & 0x1u) == 0) return true;
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return true;
	}
	uint32_t element_size = 0;
	if (index_type == vk::IndexType::eUint16) element_size = 2;
	else if (index_type == vk::IndexType::eUint32) element_size = 4;
	else return false;
	const auto index_mask  = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) return true;
	restart = (reset_index & index_mask) == index_mask;
	return true;
}

// (Mode 4, TableWorkerEntry below) What a worker made for a draw, held until its commit publishes it.
struct RenderExecutor::TableEntryHolder {
	std::remove_cvref_t<decltype(TableXpr::pairs)> node; // a new pair: its one node (moved into the catalog's map at publication)
	TablePair*                                     pair = nullptr; // drawn with: the node's, or the catalog's
	TableVariant                                   variant;        // the variant made here (own_variant)
	TableVariant*                                  drawn_variant = nullptr;
	bool                                           own_variant   = false;
	std::unique_ptr<NativeXprDrawState>            state;          // derived from a sibling's capture (null: the key's own)
	const NativeXprDrawState*                      drawn_state = nullptr;
	uint64_t                                       capture     = 0; // drawn_state's capture_id
	std::array<const ShaderRecompiler::IR::CompiledShaderInfo*, 2> programs {};
	uint64_t                                       pair_hash = 0, key = 0;
	bool                                           new_pair  = false;
	std::atomic<bool>                              poisoned {false}; // a set of its draws sampled a target: never published
	// (The records of the unrecorded permutations it rests on and the normal pipeline made for it, put in at its commit
	// (PipelineCache::CommitAdopted); dropped if never committed.)
	std::shared_ptr<PipelineCache::AdoptedCommit> adopted;
	// (An entry made with records: a later draw of this translation drawn with it is so only when its own normal lookups
	// read clean and are in its commit (TableWorkerEntry's reuse, PipelineCache::AdoptedCovers); the targets its normal
	// pipeline key is built from: the base state's inputs.)
	bool                                                      reuse_checked = false;
	std::shared_ptr<const NativeXprDrawState::PipelineInputs> inputs;
	bool                                                      meta_noted = false; // (TableEntrySettleInOrder done)
	// (Its table pipeline was compiling: the compile's job (null: not parked); the stand-in its draws were recorded with
	// (the pipeline's layouts, no handle: the variant's pipeline until the commit); the cells of its draws' pipeline binds
	// (each in its packet's copy in the passive producer's chunk, with the segment recording it); resolved: the compiled
	// pipeline written in every cell and put in the variant (TableEntryResolveParked, the commit's check).)
	std::shared_ptr<PipelineCache::ParkedPipeline> parked;
	std::unique_ptr<PipelineCache::Pipeline>       parked_stand_in;
	std::vector<std::pair<size_t, uint64_t*>>      parked_cells;
	bool                                           parked_resolved = false;
	// (Its state a worker's capture whose commit makes colour targets: its segment ran as a hole (its draws went with it;
	// the hole made the targets in order): nothing made at a later commit, never published.)
	bool                                           virtual_hole = false;
	// (TableEntryOverlay) The shader map generation its programs are proven as of (the code hashed again when it moved); its
	// catalog part begun (TableEntrySettle: out of the overlay from then); put in the overlay (the GPU thread's); its segment
	// re-run as a hole (on_hole: never in the overlay, and a draw drawn with it through the overlay fails its check).
	uint64_t                                       generation = 0;
	std::atomic<bool>                              overlay_settled {false};
	bool                                           overlay_added = false;
	std::atomic<bool>                              ran_as_hole {false};
};
// (Mode 4) Holders not published whole, kept until no speculation translates or has a result unfinished (TableSweepPairs): a
// draw recorded with them, and the proposals settled after theirs (their sets, into a new pair's node), may still refer to
// them.
static std::mutex                                                     g_entry_orphans_mutex;
static std::vector<std::shared_ptr<RenderExecutor::TableEntryHolder>> g_entry_orphans;
static std::atomic<uint32_t>                                          g_entry_orphans_n {0};
// (Mode 4, TableEntryOverlay) The entries whose segment committed and whose catalog part waits (the settle queue's): the
// GPU thread's adds (a try of the mutex: none when a worker holds it), the workers' lookups.
static std::mutex                                                     g_entry_overlay_mutex;
static std::vector<std::shared_ptr<RenderExecutor::TableEntryHolder>> g_entry_overlay;
static std::atomic<uint32_t>                                          g_entry_overlay_n {0};

// The pairs and dispatches no draw or dispatch looked up for PairKeepFrames leave, their image sets retired. Every
// pointer to a pair goes too: the lookups' (pair_cache, found_*), the last draw's and dispatch's, the store request,
// the gather and the prefetch chain (next). Only the catalog's owner, while no speculation translates or has a result
// unfinished (kyty_local_speculations_live; its next translation drops what it held, ForgetStaleCatalog), and only at
// the start of a table draw or dispatch (no pair is held then).
void RenderExecutor::TableSweepPairs() {
	auto* table = m_table_xpr.get();
	if (table == nullptr || Role() != 0) return;
	const auto frame = m_context.FrameNumber();
	if (frame - table->pairs_swept < TableXpr::PairSweepFrames) return;
	// (Put off while a speculation translates or has a result unfinished: tried at every call until done. Checked again
	// under the lock: a translation counted after it reads the catalog after the sweep, and sees its epoch.)
	if (kyty_local_speculations_live.load(std::memory_order_seq_cst) != 0) return;
	// (Mode 4's only: the commits' catalog parts still waiting run first, they may name pairs the sweep frees; then the
	// entries' orphans.)
	if (s_settle_pending_n.load(std::memory_order_relaxed) != 0) SettleDrain(true);
	if (g_entry_orphans_n.load(std::memory_order_relaxed) != 0) TableRetireEntryOrphans();
	table->pairs_swept = frame;
	if (table->pairs.size() + table->dispatches.size() <= TableXpr::PairsKept) return;
	const auto write = CatalogWrite();
	if (kyty_local_speculations_live.load(std::memory_order_seq_cst) != 0) {
		table->pairs_swept = frame - TableXpr::PairSweepFrames;
		return;
	}
	const Weave::Hold hold; // (the pairs a weave worker finds pairs in)
	const size_t before = table->pairs.size() + table->dispatches.size();
	for (auto* pairs: {&table->pairs, &table->dispatches})
		std::erase_if(*pairs, [&](auto& item) {
			auto& pair = item.second;
			// (A translation or compile it waits for keeps it.)
			if (pair.used_frame + TableXpr::PairKeepFrames >= frame || pair.waiting) return false;
			// (KYTY_SCOUT: its view retired too, as TableReset does: else it stays in the directory, its slot lost to others.)
			if (pair.scout_version != 0) Scout::Retire(TableScoutHash(pair), pair.scout_version);
			for (auto& [words, set]: pair.sets) {
				if (Weave::Active()) Weave::NoteFreed(set.get());
				NativeXprRetireSet(set->pool, set->set);
			}
			if (Weave::Active()) Weave::NoteFreed(&pair);
			return true;
		});
	const size_t after = table->pairs.size() + table->dispatches.size();
	if (after == before) return;
	++table->memo_epoch;
	table->pair_cache         = {};
	table->found_key          = {};
	table->found_pair         = nullptr;
	table->found_dispatch_key = {};
	table->found_dispatch     = nullptr;
	table->previous_dispatch  = nullptr;
	table->store              = nullptr;
	table->gathered           = nullptr;
	table->last_pair          = nullptr;
	table->last_state         = nullptr;
	table->last_set           = nullptr;
	++table->epoch;
	for (auto* pairs: {&table->pairs, &table->dispatches})
		for (auto& [key, pair]: *pairs) pair.next = nullptr;
	if (SlowLog::HitchThreshold() > 0.0)
		std::printf("[tsc %llu] table pairs %zu -> %zu\n", static_cast<unsigned long long>(__rdtsc()), before, after);
}

RenderExecutor::TableXpr& RenderExecutor::OwnTable() {
	if (!m_table_xpr) {
		const Weave::Hold hold; // (a weave worker reads the pointer)
		m_table_xpr = std::make_shared<TableXpr>();
	}
	return *m_table_xpr;
}

// What this executor held of another's catalog (its last pair, state and set), dropped when the owner reset a pair or
// let a set go since.
void RenderExecutor::DropStaleCatalog() {
	auto* catalog = CatalogTable();
	auto& table   = OwnTable();
	if (Role() == 0 || catalog == nullptr || catalog->epoch == table.seen_epoch) return;
	table.seen_epoch = catalog->epoch;
	table.last_pair  = nullptr;
	table.last_state = nullptr;
	table.last_set   = nullptr;
	// (Its memo holds the owner's sets and states too: the owner's retirements move only the owner's memo_epoch.)
	++table.memo_epoch;
	table.memo_set = nullptr;
}

// A speculation's executor, before its translation reads the catalog (Speculation::Translate): the catalog's sweeps
// (TableSweepPairs, NativeXprSweepStates) run only between translations, so what it kept from the last one of the
// owner's pairs, draw states and sets is dropped here when the owner's epoch moved since: its table's lookups, last
// draw and memo, and its native emission's last state.
void RenderExecutor::ForgetStaleCatalog() {
	if (Role() == 0) return;
	const auto read    = CatalogRead();
	auto*      catalog = CatalogTable();
	if (catalog == nullptr || m_table_xpr == nullptr || catalog->epoch == m_table_xpr->seen_epoch) return;
	auto& table              = *m_table_xpr;
	table.seen_epoch         = catalog->epoch;
	table.pair_cache         = {};
	table.found_key          = {};
	table.found_pair         = nullptr;
	table.found_dispatch_key = {};
	table.found_dispatch     = nullptr;
	table.previous_dispatch  = nullptr;
	table.store              = nullptr;
	table.gathered           = nullptr;
	table.last_pair          = nullptr;
	table.last_state         = nullptr;
	table.last_set           = nullptr;
	table.memo_set           = nullptr;
	++table.memo_epoch;
	if (m_native_xpr) {
		m_native_xpr->emit           = {};
		m_native_xpr->last_state     = nullptr;
		m_native_xpr->last_state_key = 0;
	}
}

// A table draw's or dispatch's pair: each draw's pair is another, cold, object and its evaluation
// walks a dozen arrays one after the other (runs, operands, V# words, constants, the last image set's words). Their
// lines are asked for once the pair is known (its own lines first: the vectors' headers), so the misses overlap the
// checks before the evaluation instead of following each other (1-1 spin, per call: table draw path 1.78 -> 1.68 us,
// table dispatch path 2.96 -> 2.86 us).
template <typename Pair>
static void TablePrefetchPair(const Pair& pair) {
	for (size_t line = 0; line < sizeof(Pair); line += 64) __builtin_prefetch(reinterpret_cast<const char*>(&pair) + line);
}
template <typename Pair, typename Set>
static void TablePrefetchArrays(const Pair& pair, const Set* set) {
	const auto lines = [](const auto& items) {
		const auto* at    = reinterpret_cast<const char*>(items.data());
		const auto  bytes = std::min<size_t>(items.size() * sizeof(*items.data()), 256);
		for (size_t offset = 0; offset < bytes; offset += 64) __builtin_prefetch(at + offset);
	};
	for (uint32_t offset = 0; offset < std::min<uint32_t>(pair.packed_bytes, 2048); offset += 64)
		__builtin_prefetch(pair.packed_at + offset);
	for (uint32_t st = 0; st < 2 && pair.programs[st] != nullptr; ++st) {
		lines(pair.programs[st]->table_plan.buffers); // (TableBlocks' plan and resource flags of each V#)
		lines(pair.programs[st]->info.buffers);
	}
	if (set != nullptr) {
		__builtin_prefetch(set);
		lines(set->words);
	}
}

// A frame's dispatches come in much the same order frame after frame (the culling chain's ~1000 of a few hundred
// programs): the pair that followed this one's last dispatch is noted, and the pairs of the next two dispatches expected
// load while this one is made, in two steps (each line a load reads is in by then: the next pair's arrays from its own
// lines, which the dispatch before asked for, and the pair after it, whose lines the next one's arrays need). Their
// lines were cold at each use: TablePrefetchPair asks for them only once the pair is found, ~0.1 us before the first
// read. Census per call, 1-1 standing and Latria (2 boots x 2 rounds each): async-queue table dispatches 1.76 -> 1.70
// and 2.06 -> 2.00 us (their evaluation -9%), graphics-queue ones -4%. (For draws, whose successors vary more, the
// same cost +3% a draw.)
template <typename Pair>
static void TablePrefetchNext(Pair*& previous, Pair& pair) {
	if (previous != nullptr && previous != &pair) previous->next = &pair;
	previous = &pair;
	auto* const next = pair.next;
	if (next == nullptr || next == &pair) return;
	using Set = std::remove_pointer_t<std::remove_cvref_t<decltype(next->last_set)>>;
	TablePrefetchArrays(*next, static_cast<const Set*>(nullptr));
	__builtin_prefetch(next->last_set);
	if (auto* const after = next->next; after != nullptr && after != &pair && after != next) TablePrefetchPair(*after);
}

// spec_worker_sets: the catalog read lock the speculation's table draw or dispatch holds (TableTry, TableDispatch,
// TableContinue). The set build keeps it across the program lookup (`unlocked` there only tests it), so the lookup's guest
// reads are made without a fault: a fault there would wait for the GPU thread while it waits for the catalog's write.
static thread_local std::shared_lock<std::shared_mutex>* t_catalog_read = nullptr;
// (Mode 4, TableWorkerEntry) The variant it made and draws with now, not in its pair's vector (TableResolveSet's re-find
// after an unlock does not apply to it); the poisoned flag of the entry drawn with (TableDraw sets it).
static thread_local const void*        t_entry_variant = nullptr;
static thread_local std::atomic<bool>* t_entry_poison  = nullptr;
// (Mode 4) The entry drawn with now when it is parked on its table pipeline's compile (TableDraw binds its stand-in through a
// cell).
static thread_local RenderExecutor::TableEntryHolder* t_entry_parked = nullptr;
// (Mode 4) A parked entry's pipeline bind, recorded as a packet whose pipeline handle is a cell the commit writes
// (TableEntryResolveParked, the GPU thread, before the segment is spliced; nothing reads a passive producer's chunk before
// its splice, and a discard runs nothing of it), as the relocatable buffer references' cells are (TableReloc). Its replay
// ends the draws' replays' reuse of the pipeline they bound last (LocalDrawRecording keeps it): a bind they do not see.
#ifdef KYTY_LOCAL_VULKAN_RECORDING
namespace TableParked {
struct Packet {
	VkCommandBuffer command  = VK_NULL_HANDLE;
	uint64_t        pipeline = 0; // (the cell: a VkPipeline, written by the commit)
};
static_assert(std::is_standard_layout_v<Packet> && std::is_trivially_copyable_v<Packet>);
static void Replay(std::span<const LocalVulkanRecording::Segment> segments, const vk::detail::DispatchLoaderDynamic& dispatch) {
	Packet packet;
	std::memcpy(&packet, segments[0].data, sizeof(packet));
	LocalVulkanRecording::InvalidateReplayState();
	// (Never spliced unwritten: the commit's check stops the segment first. A null one binds nothing rather than fault.)
	if (packet.pipeline == 0) return;
	dispatch.vkCmdBindPipeline(LocalVulkanRecording::Resolve(packet.command), VK_PIPELINE_BIND_POINT_GRAPHICS,
	                           reinterpret_cast<VkPipeline>(packet.pipeline)); // NOLINT (a 64-bit non-dispatchable handle)
}
} // namespace TableParked
#endif
// Whether this thread records packets whose cells its commit writes (a passive producer's: a speculation worker's segment).
static bool TableCellsRecordable() {
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	return LocalVulkanRecording::Passive();
#else
	return false;
#endif
}
// The parked entry's bind recorded with a cell, noted with the segment recording it (false: not recorded, the draw refused).
static bool TableParkedBind(vk::CommandBuffer vk_buffer, RenderExecutor::TableEntryHolder& h) {
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	if (!LocalVulkanRecording::Passive()) return false;
	const TableParked::Packet           packet {static_cast<VkCommandBuffer>(vk_buffer), 0};
	const LocalVulkanRecording::Segment segment {&packet, sizeof(packet)};
	void*                               copy = nullptr;
	if (!LocalVulkanRecording::EnqueuePacketCopy(TableParked::Replay, segment, &copy) || copy == nullptr) return false;
	const auto* spec = Spec::Current();
	h.parked_cells.emplace_back(spec != nullptr ? spec->closed.size() : 0,
	                            reinterpret_cast<uint64_t*>(static_cast<std::byte*>(copy) + offsetof(TableParked::Packet, pipeline)));
	return true;
#else
	(void)vk_buffer, (void)h;
	return false;
#endif
}
// (Mode 4, the GPU thread) TableDraw refused a draw under a state a worker derived from a sibling's capture whose set holds
// a target; its caller (TableTry) then takes the store path for the draw (request_store), so the in-order store captures
// the state and replaces the derived one.
static thread_local bool t_store_derived = false;
// (Mode 4) A speculation's draw that lacks its table entry makes it itself (TableWorkerEntry).
[[nodiscard]] static bool WorkerEntryOn() {
	return kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4;
}
// catalog_writer_first: before a worker's table try takes the catalog shared, it waits while the owner waits to write
// (pause, then yield), unless this thread holds the catalog shared already.
static void CatalogGate(const std::atomic<int32_t>& writer_waiting, bool on) {
	if (!on || writer_waiting.load(std::memory_order_acquire) == 0) return;
	if (t_catalog_read != nullptr && t_catalog_read->owns_lock()) return;
	for (uint32_t spins = 0; writer_waiting.load(std::memory_order_acquire) != 0; ++spins) {
		if (spins < 64) _mm_pause();
		else std::this_thread::yield();
	}
}
// (t_catalog_read is read only by a speculation's executor (CatalogGate behind Role() != 0, the set build's `unlocked`
// behind a speculation), which runs only while the mode is not 0: mode 0 tests the global and touches no thread-local.)
struct CatalogReadScope {
	std::shared_lock<std::shared_mutex>* old = nullptr;
	const bool                           on;
	explicit CatalogReadScope(std::shared_lock<std::shared_mutex>* lock): on(Spec::On()) {
		if (on) old = t_catalog_read, t_catalog_read = lock;
	}
	~CatalogReadScope() {
		if (on) t_catalog_read = old;
	}
	CatalogReadScope(const CatalogReadScope&)            = delete;
	CatalogReadScope& operator=(const CatalogReadScope&) = delete;
};

RenderExecutor::TableResult RenderExecutor::TableTry(CommandBuffer& buffer, std::span<const uint64_t> commands,
                                                     uint64_t index_base, uint64_t index_bytes,
                                                     vk::IndexType index_type, const NativeXprDirectDraw* direct) {
	return TableTry<Weave::Ordered>(buffer, commands, index_base, index_bytes, index_type, direct);
}

// (Under Weave::Worker: the executor's pairs are read, the worker's own table holds its lookups and last draw, and the
// draw-state key and the state found last are the worker's. What the GPU thread's draw would write, a worker skips,
// keeps or declines: weave-draw.inc.)
template <typename P>
RenderExecutor::TableResult RenderExecutor::TableTry(CommandBuffer& buffer, std::span<const uint64_t> commands,
                                                     uint64_t index_base, uint64_t index_bytes,
                                                     vk::IndexType index_type, const NativeXprDirectDraw* direct) {
	auto& table  = TableOf<P>();
	auto& common = P::worker ? OwnTable() : table; // (the executor's: its pairs)
	// (Another executor's catalog is read under its lock.)
	std::shared_lock<std::shared_mutex> catalog_read;
	if (Role() != 0) CatalogGate(Catalog().m_catalog_writer_waiting, CatalogWriterFirst()), catalog_read = CatalogRead();
	const CatalogReadScope catalog_read_scope(&catalog_read); // (spec_worker_sets)
	DropStaleCatalog();
	if constexpr (!P::worker) TableSweepPairs();
	TableCensus<P> census(LiveCensus::DrawPhase, 0x7ab1e, 0); // census: the table path's phases (b)
	auto*       catalog = CatalogTable();
	auto&       cache   = *m_native_xpr;
	// (The draw-state key and the state found last: the GPU thread's in the native cache, a worker's in its scratch.)
	auto& keys = [&]() -> auto& {
		if constexpr (P::worker)
			return WeaveHere();
		else
			return cache;
	}();
	if constexpr (P::worker) WeaveMiss(Weave::Miss::Pair);
	const auto& sh      = buffer.GetShaders();
	const auto& vs      = sh.GetVs();
	// The vertex program is the ES stage's (ShaderGetInputInfoVS), without tessellation.
	if (vs.es_regs.data_addr == 0 || vs.ls_regs.data_addr != 0 || vs.hs_regs.data_addr != 0)
		return Spec::Refuse("table: stages"), TableResult::Native;
	const auto frame      = m_context.FrameNumber();
	const auto vs_address = vs.es_regs.data_addr, ps_address = sh.GetPs().ps_regs.data_addr;
	const std::array<uint64_t, 3> pair_key {vs_address, ps_address, static_cast<uint32_t>(vs.start_instance_user_sgpr)};
	auto* own   = Role() != 0 ? nullptr : &common.pairs;
	auto* known = own != nullptr && table.found_key == pair_key ? table.found_pair : nullptr;
	if (known == nullptr) {
		const auto pair_hash = XXH3_64bits(pair_key.data(), sizeof(pair_key));
		// (Another executor's catalog: what it learned, none of it made here.)
		const auto found = catalog == nullptr || Role() == 0 ? decltype(catalog->pairs.end()) {} : catalog->pairs.find(pair_hash);
		if (Role() != 0 && (catalog == nullptr || found == catalog->pairs.end())) {
			// (Mode 4: the worker makes the pair itself, TableWorkerEntry.)
			if (catalog != nullptr && WorkerEntryOn())
				if (const auto drawn = TableWorkerEntry(buffer, nullptr, pair_hash, commands, index_base, index_bytes, index_type, direct))
					return *drawn;
			return Spec::Refuse("table store"), TableResult::Native;
		}
		// (A frame's draws use ~600 pairs at 1-1: 256 entries missed often, each a cold map bucket and node.)
		const auto entries = (kyty_local_table_mru.load(std::memory_order_relaxed) & 1u) != 0 ? TableXpr::PairCacheEntries : 256;
		auto&      cached  = table.pair_cache[pair_hash & (entries - 1)];
		known        = own != nullptr && cached.first == pair_hash ? cached.second : nullptr;
		if (known == nullptr) {
			// (A weave worker reads the pairs unlocked: the GPU thread holds it outside a draw to insert or erase one.)
			auto it = own != nullptr ? own->find(pair_hash) : found;
			if (own != nullptr && it == own->end()) {
				if constexpr (P::worker) return TableResult::Native; // (a new pair: the GPU thread's to make)
				const auto write    = CatalogWrite();
				const Weave::Hold hold;
				it                  = own->try_emplace(pair_hash).first;
				auto& fresh         = it->second;
				fresh.vs            = vs_address;
				fresh.ps            = ps_address;
				fresh.start_instance = vs.start_instance_user_sgpr;
			}
			known = &it->second;
			if (own != nullptr) cached = {pair_hash, known};
		}
		if (own != nullptr) {
			table.found_key  = pair_key;
			table.found_pair = known;
		}
	}
	auto& pair = *known;
	TablePrefetchPair(pair);
	// (A speculation's thread notes the pair in its segment, whose commit stamps it (Speculation::After): only the GPU
	// thread writes it, and a pair only speculations use is not swept as unused. A refused draw's note goes with it.)
	// (A weave worker's draw leaves the pair's last use to the turn's stamp.)
	if constexpr (!P::worker) {
		if (Role() == 0) pair.used_frame = frame;
		else if (auto* spec = Spec::Current(); spec != nullptr) spec->NoteUsedPair(&pair.used_frame);
	}
	if (pair.vs != vs_address || pair.ps != ps_address || pair.start_instance != vs.start_instance_user_sgpr)
		return Spec::Refuse("table: pair key"), TableResult::Native;
	if constexpr (P::worker) {
		// Proving the shaders again and a reset are the GPU thread's. A worker proves them read only, the code read
		// from the backing, never through a page that can fault (ShaderHashAtNoFault), and takes the draw when both are
		// still the pair's programs, noting the generation it proved them at for the turn, which stamps the pair with
		// it while the shader map is still there. A pair to compile again is a hole.
		WeaveMiss(Weave::Miss::Shaders);
		if (const auto generation = ShaderMapGeneration();
		    pair.programs[0] != nullptr &&
		    generation != std::atomic_ref<uint64_t>(pair.generation).load(std::memory_order_relaxed)) {
			uint64_t vs_hash = 0, ps_hash = 0;
			if (!ShaderHashAtNoFault(vs_address, vs_hash) || vs_hash != pair.programs[0]->shader_hash ||
			    (pair.programs[1] != nullptr &&
			     (!ShaderHashAtNoFault(ps_address, ps_hash) || ps_hash != pair.programs[1]->shader_hash)))
				return TableResult::Native;
			WeaveParcel().result->made.back().pair_proved = generation;
		}
		if (pair.respecialize) return TableResult::Native;
	} else {
		if (const auto generation = ShaderMapGeneration(); generation != pair.Generation(Role())) {
			if (pair.programs[0] != nullptr &&
			    (ShaderHashNoFaultScope {Role() != 0 && !GuestGpu::IsGpuThread()}, ShaderHashAt(vs_address) != pair.programs[0]->shader_hash ||
			     (pair.programs[1] != nullptr && ShaderHashAt(ps_address) != pair.programs[1]->shader_hash))) {
				if (Role() != 0) return Spec::Refuse("table: shaders changed"), TableResult::Native;
				const auto write = CatalogWrite();
				const Weave::Hold hold;
				pair.portable    = {};
				TableReset(pair);
			}
			pair.Generation(Role()) = generation;
		}
		if (pair.respecialize) {
			if (Role() != 0) return Spec::Refuse("table: shaders changed"), TableResult::Native;
			const auto write = CatalogWrite();
			const Weave::Hold hold;
			TableReset(pair);
		}
	}
	if (pair.refused) return Spec::Refuse(Spec::TablePairRefusedText), TableResult::Native;
	// KYTY_FRAMEGEN: the frame's first G-buffer draw goes the normal way (or a native record's), whose hook reads the
	// camera from its constants. A speculation's thread cannot see whether its frame has the camera yet: it leaves the
	// first G-buffer draw of its speculation to the GPU thread (a hole, the normal way there while the frame has no
	// camera) and draws the others the table way. The frame's first G-buffer draw is the first of the speculation that
	// holds it, so the camera comes from the same draw as without speculation. (Leaving them all made ~30 holes a frame,
	// 4-9 ms of the commit's, issue #378.) A weave worker's draw is marked instead, and the turn leaves the frame's first
	// to the GPU thread (Weave::Made::gbuffer).
	if (FrameGen::Enabled() && pair.programs[0] != nullptr &&
	    FrameGen::IsGbufferVertexShader(pair.programs[0]->shader_hash) &&
	    (!P::worker && (Role() == 0 ? FrameGen::WantsCamera() : Spec::LeaveCameraDraw())))
		return Spec::Refuse("table: frame generation camera"), TableResult::Native;
	// A store at most once a frame per pair and draw state, within the frame's budget (kyty_local_table_store_budget):
	// while its programs and pipeline compile, the pair's draws take the normal path (Deferred). (Once a frame per pair
	// kept its other states' draws on the native path, storing native records, until a later frame.) Those draws
	// stored native records too, which the table variant replaced a frame or two later: 100-230 a slow frame walking
	// into new parts of Boletarian Palace 1-3 (each traced and sorted, ~3 ms of the frame).
	const auto request_store = [&] {
		if constexpr (P::worker) return TableResult::Native; // (a store is the GPU thread's: its own draw asks for it)
		if (Spec::Current() != nullptr) return Spec::Refuse("table store"), TableResult::Native;
		if (pair.waiting) {
			if (!pair.waiting->load(std::memory_order_acquire))
				return LiveCounters::Add(LiveCounters::TableStoreDeferred), TableResult::Deferred; // (programs compiling)
			pair.waiting.reset();
		}
		if (table.store_frame != frame) {
			table.store_frame    = frame;
			table.store_requests = 0;
		}
		if (pair.store_frame == frame && pair.store_state == cache.state_key)
			return LiveCounters::Add(LiveCounters::TableStoreDeferred), TableResult::Deferred;
		// (Past the budget, the others draw from native records meanwhile.)
		if (const auto budget = kyty_local_table_store_budget.load(std::memory_order_relaxed);
		    budget != 0 && table.store_requests >= budget)
			return LiveCounters::Add(LiveCounters::TableStoreDeferred), TableResult::Native;
		++table.store_requests;
		pair.store_frame  = frame;
		pair.store_state  = cache.state_key;
		table.store       = &pair;
		table.store_state = cache.state_key;
		return TableResult::Store;
	};
	// KYTY_TABLE_MEMO: what this pair and state key drew with last (the main executor's), its variant still that key's.
	const bool memo = !P::worker && Role() == 0 && TableMemoOn();
	if (memo) {
		auto& entry = table.MemoAt(&pair, cache.state_key);
		if (entry.pair == &pair && entry.key == cache.state_key && entry.epoch == table.memo_epoch &&
		    entry.variant < pair.variants.size() && pair.programs[0] != nullptr) {
			auto& memo_variant = pair.variants[entry.variant];
			// KYTY_TABLE_MEMO_VERIFY: every Nth hit, what the full path below finds (the variant by its hint or the
			// find, the draw state by the cache or the states' find; read only) against the entry's. A difference
			// drops the entry and the draw takes the full path.
			bool verify = false, differs = false;
			if (const auto every = TableMemoVerifyEvery();
			    every > 0 && memo_variant.state == cache.state_key && memo_variant.pipeline != nullptr &&
			    !memo_variant.refused && ++table.memo_verify_count % static_cast<uint64_t>(every) == 0) {
				verify           = true;
				const bool hinted = (kyty_local_table_mru.load(std::memory_order_relaxed) & 2u) != 0 &&
				                    pair.variant_hint < pair.variants.size() &&
				                    pair.variants[pair.variant_hint].state == cache.state_key;
				const auto full_variant =
				    hinted ? pair.variant_hint
				           : static_cast<uint32_t>(std::ranges::find(pair.variants, cache.state_key, &TableXpr::Variant::state) -
				                                   pair.variants.begin());
				const NativeXprDrawState* full_state = nullptr;
				if (cache.last_state != nullptr && cache.last_state_key == cache.state_key) {
					full_state = cache.last_state;
				} else {
					const auto& states   = Catalog().NativeXprState().states;
					const auto  state_it = states.find(cache.state_key);
					if (state_it != states.end()) full_state = state_it->second.get();
				}
				LiveCounters::Add(LiveCounters::TableMemoVerified);
				const uint32_t differ = (full_variant != entry.variant ? 1u : 0u) | (full_state != entry.state ? 2u : 0u);
				TableMemoChecked(differ, pair.vs, pair.ps, cache.state_key);
				if (differ != 0) {
					differs     = true;
					entry.epoch = 0;
				}
			}
			if (!differs && memo_variant.state == cache.state_key && memo_variant.pipeline != nullptr &&
			    !memo_variant.refused) {
				const auto& memo_state = *entry.state;
				pair.variant_hint      = entry.variant;
				TablePrefetchArrays(pair, pair.LastSet(Role()));
				if (!NativeXprTargetsCurrent(memo_state))
					return LiveCounters::Add(LiveCounters::TableStoreTargets), request_store();
				LiveCounters::Add(LiveCounters::TableMemoHits);
				table.memo_set     = entry.set;
				table.memo_verify  = verify;
				const auto result  = TableDraw<P>(buffer, pair, memo_variant, memo_state, commands, index_base, index_bytes,
				                                  index_type, direct);
				table.memo_set     = nullptr;
				table.memo_verify  = false;
				// (Mode 4: a derived state's draw refused for a target in its set: the store path, as below.)
				if (memo_state.derived && std::exchange(t_store_derived, false)) return request_store();
				return result;
			}
		}
	}
	// (The variant found last first: a pair's variants are a cold line each, its draws mostly keep their state. A weave
	// worker finds it: the hint is the GPU thread's, rewritten at every draw.)
	if constexpr (P::worker) WeaveMiss(Weave::Miss::Variant);
	const bool hint    = !P::worker && (kyty_local_table_mru.load(std::memory_order_relaxed) & 2u) != 0 &&
	                  pair.variant_hint < pair.variants.size() && pair.variants[pair.variant_hint].state == keys.state_key;
	const auto variant = hint ? pair.variants.begin() + pair.variant_hint
	                          : std::ranges::find(pair.variants, keys.state_key, &TableXpr::Variant::state);
	if constexpr (!P::worker)
		if (variant != pair.variants.end()) pair.variant_hint = static_cast<uint32_t>(variant - pair.variants.begin());
	if (variant != pair.variants.end() && variant->refused) return Spec::Refuse("table: variant refused"), TableResult::Native;
	if (pair.programs[0] == nullptr || variant == pair.variants.end() || variant->pipeline == nullptr) {
		// (Mode 4: the worker makes the variant, and the draw state if the key has none, TableWorkerEntry.)
		if (Role() != 0 && pair.programs[0] != nullptr && WorkerEntryOn())
			if (const auto drawn = TableWorkerEntry(buffer, &pair, 0, commands, index_base, index_bytes, index_type, direct))
				return *drawn;
		// (Mode 4: the pair's programs not made yet (its store compiling): an entry a worker made for the key, not in the
		// catalog yet, TableEntryOverlay.)
		if (Role() != 0 && pair.programs[0] == nullptr && WorkerEntryOn())
			if (const auto drawn = TableEntryOverlay(buffer, &pair, commands, index_base, index_bytes, index_type, direct))
				return *drawn;
		return TableAdd<P>(LiveCounters::TableStoreVariant), request_store();
	}
	// The draw state (TableStore captures it).
	if constexpr (P::worker) WeaveMiss(Weave::Miss::State);
	if (keys.last_state == nullptr || keys.last_state_key != keys.state_key) {
		// (A weave worker reads the draw states unlocked: the GPU thread holds it outside a draw to insert or erase
		// one.)
		const auto& states   = Catalog().NativeXprState().states;
		const auto  state_it = states.find(keys.state_key);
		if (state_it == states.end()) {
			// (Mode 4: the worker derives the draw state from a sibling's, TableWorkerEntry.)
			if (Role() != 0 && WorkerEntryOn())
				if (const auto drawn = TableWorkerEntry(buffer, &pair, 0, commands, index_base, index_bytes, index_type, direct))
					return *drawn;
			return request_store();
		}
		keys.last_state_key = keys.state_key;
		keys.last_state     = state_it->second.get();
	}
	const auto& draw_state = *keys.last_state;
	TablePrefetchArrays(pair, pair.LastSet(Role()));
	if (!TableTargetsCurrent<P>(draw_state)) return TableAdd<P>(LiveCounters::TableStoreTargets), request_store();
	const auto result = TableDraw<P>(buffer, pair, *variant, draw_state, commands, index_base, index_bytes, index_type, direct);
	// (Mode 4, the GPU thread: a state a worker derived from a sibling's capture whose draw TableDraw refused for a target in
	// its set: the store path for it, as for a draw whose state has no capture or whose targets are not current; TableStore's
	// capture then replaces the derived state.)
	if (draw_state.derived && std::exchange(t_store_derived, false)) return request_store();
	if (memo && result == TableResult::Drawn)
		table.MemoAt(&pair, cache.state_key) = {&pair, cache.state_key, table.memo_epoch, &draw_state,
		                                        const_cast<TableImageSet*>(table.last_set),
		                                        static_cast<uint32_t>(variant - pair.variants.begin())};
	return result;
}

bool RenderExecutor::TableContinue(CommandBuffer& buffer, std::span<const uint64_t> commands, uint64_t index_base,
                                   uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct) {
	return TableContinue<Weave::Ordered>(buffer, commands, index_base, index_bytes, index_type, direct);
}

// (Under Weave::Worker the last draw is the worker's, in its strand's buffer: false when it has none to continue.)
template <typename P>
bool RenderExecutor::TableContinue(CommandBuffer& buffer, std::span<const uint64_t> commands, uint64_t index_base,
                                   uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct) {
	if (kyty_local_table_xpr_mode.load(std::memory_order_relaxed) < 2 || !m_table_xpr || !m_native_xpr) return false;
	std::shared_lock<std::shared_mutex> catalog_read;
	if (Role() != 0) CatalogGate(Catalog().m_catalog_writer_waiting, CatalogWriterFirst()), catalog_read = CatalogRead();
	const CatalogReadScope catalog_read_scope(&catalog_read); // (spec_worker_sets)
	DropStaleCatalog();
	auto& table = TableOf<P>();
	if (table.last_pair == nullptr || buffer.IsInvalid() || index_base == 0 || index_bytes == 0 ||
	    commands.size() > 64 || table.last_pair->Generation(Role()) != ShaderMapGeneration() ||
	    table.last_pair->start_instance != buffer.GetShaders().GetVs().start_instance_user_sgpr)
		return false;
	auto&       variant = table.last_pair->variants[table.last_variant];
	const auto& emit    = TableEmit<P>();
	if (emit.command != buffer.Handle() || emit.generation != buffer.GraphicsGeneration() ||
	    emit.state != table.last_state || emit.pipeline != variant.pipeline)
		return false;
	// (The GPU thread's: a speculation's thread does not take it, nor a weave worker, which records apart.)
	std::optional<RenderLockGuard> lock;
	if constexpr (!P::worker)
		if (Role() == 0) lock.emplace(m_context.GetMutex());
	if (!TableTargetsCurrent<P>(*table.last_state)) return false;
	const auto drawn = TableDraw<P>(buffer, *table.last_pair, variant, *table.last_state, commands, index_base, index_bytes,
	                                index_type, direct);
	// (Mode 4: a derived state's refusal mark cleared: the packet goes to NativeXprTry again, whose TableTry asks the store.)
	if (table.last_state->derived) t_store_derived = false;
	if (drawn != TableResult::Drawn) return false;
	TableAdd<P>(LiveCounters::TableContinued, std::max<size_t>(commands.size(), 1));
	return true;
}

// The image set of the T#/S# words (`words`) of a draw (under `variant`) or a dispatch (`variant` null): its
// resources as the normal path evaluates them, which must select the pair's table mode programs (the same
// specializations), and its images and samplers as the normal path resolves and acquires them (PrepareBindingsInto,
// RebindImages), in a descriptor set with the page table and the fault buffer. Refused for another specialization,
// or when an image cannot be held in a set (a target the shader cannot read, or writes as storage).
// KYTY_TABLE_LEARN_PORTABLE (live table_learn_portable; mode 4: 2, else 0): 1, a set refused for another
// specialization that only a buffer's V# format makes (one pair's draws with an 8 and an 8_8_8_8 vertex stream) learns
// that buffer as portable instead; 2, also a retry still pending waits for its translation.
static int64_t TableLearnPortable() {
	static auto& value = LiveSwitch::Find("table_learn_portable", 0);
	return Spec::On() ? value.load(std::memory_order_relaxed) : 0;
}

// spec_worker_sets and spec_worker_images (under KYTY_SPECULATE=4 only, the mode whose descriptor pools a worker may
// share): a speculation's thread builds an image set
// itself (TableResolveSet) when every image is resident, current and already acquired for sampling (the texture cache
// read only: TextureCache::ReadOnlyScope); the set is its own (Spec::State::proposed_sets) until a segment from the one
// that made it on is committed, when the owner publishes it into the catalog (or keeps it aside: never freed, a recorded
// draw may hold it). Anything else is refused as before ('table image set').
[[nodiscard]] static bool WorkerImagesOn() {
	return kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4;
}

static int64_t WorkerSets() {
	if (!Spec::On()) return 0;
	return kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4 ? 1 : 0;
}
static thread_local bool t_table_revalidating = false; // (spec_worker_revalidate: TableResolveSet's caller)
static void TableKeepAside(std::shared_ptr<void> holder) {
	static std::mutex                          mutex;
	static std::vector<std::shared_ptr<void>> kept;
	std::scoped_lock                           lock(mutex);
	kept.push_back(std::move(holder));
}
// spec_aside_release (under KYTY_SPECULATE=4; otherwise kept for good as above): a set kept
// aside is released instead. When it goes aside, the roles' pointers to it are dropped under the catalog's write (as an
// unpublished pending set's are: no later lookup finds it). At a commit two frames after it went aside (every speculation
// that could still bind it is done), the GPU thread retires its descriptor set as the owner retires one
// (NativeXprRetireSet: in flight until the
// current tick; a speculation that bound it and commits later stops, "a hole retired a set"), and the set itself is
// destroyed once the translations in flight then have passed (Spec::PacketsPassed).
[[nodiscard]] static bool TableAsideReleaseOn() {
	return kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4;
}
struct TableAsideQueue {
	struct Queued {
		uint64_t              frame = 0; // (the frame it went aside in)
		std::shared_ptr<void> holder;
		std::function<void()> retire; // (its descriptor set: on the GPU thread)
	};
	std::mutex                                                       mutex;
	std::vector<Queued>                                              queued;
	std::vector<std::pair<Spec::PacketMarks, std::shared_ptr<void>>> retiring;
};
static TableAsideQueue& TableAside() {
	static TableAsideQueue aside;
	return aside;
}
// (The GPU thread, at a commit.)
static void TableAsideDrain(uint64_t frame) {
	auto&                                aside = TableAside();
	std::vector<TableAsideQueue::Queued> due;
	{
		std::scoped_lock lock(aside.mutex);
		std::erase_if(aside.retiring, [](const auto& item) { return Spec::PacketsPassed(item.first); });
		std::erase_if(aside.queued, [&](auto& item) {
			if (item.frame + 2 > frame) return false;
			due.push_back(std::move(item));
			return true;
		});
	}
	if (due.empty()) return;
	for (auto& item: due) item.retire();
	const auto       marks = Spec::PacketNow();
	std::scoped_lock lock(aside.mutex);
	for (auto& item: due) aside.retiring.emplace_back(marks, std::move(item.holder));
}

// spec_worker_images: the GPU thread, at the commit of the segment that built `set` (Speculation::Before, before the replay
// binds it): each pending slot's image as its own TableResolveSet would find it now (FindImage: made if needed;
// FindTexture: uploaded, its view), transited to the sampled layout in the open buffer, its view written into the slot,
// and the image added to the set's (its later draws validate and transit it as any). All of them, or false (the
// segment stops; the set is never published).
bool RenderExecutor::TableCompletePendingImages(TableImageSet& set) {
	auto&                                textures = m_context.GetTextureCache();
	std::vector<vk::DescriptorImageInfo> infos(set.pending_images.size());
	std::vector<vk::WriteDescriptorSet>  writes;
	size_t                               done = 0;
	for (size_t i = 0; i < set.pending_images.size(); ++i) {
		auto&      pending = set.pending_images[i];
		auto       desc    = pending.desc;
		const auto id      = textures.FindImage(desc, false, true); // (may fail: no memory for it, a refusal)
		if (!id) continue;
		// An image a draw that bound the set renders to: the slot would hold it read-only as a texture (the dep below, not
		// a target) while that draw's attachment holds it, where the owner's build lays it out as the target
		// (TableResolveSet). The segment stops: in order, the owner's build. (The rest of the set's slots are left: it is
		// retired unpublished.)
		if (std::ranges::find(set.pending_targets, id) != set.pending_targets.end()) break;
		// (Its view with metadata: FindTexture's metadata step still a no-op here, the in-order point before the work,
		// else the slot fails and the segment stops.)
		if (pending.metadata && !textures.MetadataBindNoop(id, desc)) continue;
		const auto view = textures.FindTexture(id, desc);
		if (!view) continue;
		auto& image = textures.GetImage(id);
		image.usage.texture = true;
		const ImageSubresourceRange range {desc.view_info.base_level, desc.view_info.level_count, desc.view_info.base_layer,
		                                   desc.view_info.layer_count};
		// (The open buffer read at each slot: a slot's FindImage / FindTexture that reclaims memory (a failed creation:
		// ReclaimImageMemory waits for the current tick) submitted the one open before and began the next.)
		image.Transit(pending.layout, vk::AccessFlagBits2::eShaderRead, range,
		              m_context.GetCommandScheduler().Current().Handle());
		infos[i] = vk::DescriptorImageInfo {nullptr, view, pending.layout};
		vk::WriteDescriptorSet write;
		write.dstSet          = set.set;
		write.dstBinding      = pending.binding;
		write.dstArrayElement = pending.element;
		write.descriptorCount = 1;
		write.descriptorType  = pending.type;
		write.pImageInfo      = &infos[i];
		writes.push_back(write);
		auto& dep  = set.images.emplace_back();
		dep.id     = id;
		dep.serial = image.serial;
		dep.layout = pending.layout;
		dep.range  = range;
		dep.desc   = desc;
		set.image_ids.push_back(id);
		// (Its view's metadata noted as the owner's build notes a dep's, KYTY_TABLE_SERIALS.)
		if (pending.metadata) textures.NoteMetaUse(desc.info.metadata.range.address, id.index);
		++done;
	}
	if (!writes.empty())
		m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr
#ifdef KYTY_LOCAL_VULKAN_RECORDING
		                                                    , LocalVulkanRecording::DirectDispatch()
#endif
		);
	bool ok = done == set.pending_images.size();
	set.span_begin = set.span_end = 0; // (its span grew: found again at its next proof)
	if (ok) set.pending_images.clear(), std::vector<ImageId>().swap(set.pending_targets);
	return ok;
}

RenderExecutor::TableImageSet* RenderExecutor::TableResolveSet(CommandBuffer& buffer, TablePair& pair,
                                                               const TableVariant*       variant,
                                                               const NativeXprDrawState* draw_state,
                                                               std::span<const uint32_t> words) {
	auto* const spec       = Spec::Current();
	// (A speculation's set only: its proposals are found by the words' hash. Mode 0 hashes nothing here.)
	const auto  words_hash = spec != nullptr ? XXH3_64bits(words.data(), words.size() * sizeof(uint32_t)) : uint64_t {0};
	std::optional<TextureCache::ReadOnlyScope> read_only;
	const auto worker_refuse = [](const char* why) -> TableImageSet* {
		return Spec::Refuse(why), nullptr;
	};
	if (spec != nullptr) {
		if (WorkerSets() == 0) return Spec::Refuse("table image set"), nullptr;
		// (One it built already in this translation. spec_set_cpu_dirty_stop: not one with pending slots proposed in an
		// earlier segment, which TableTransit refuses here: TableSet resolves its words afresh, into a set of this segment.)
		for (const auto& proposed: spec->proposed_sets)
			if (const auto* built = static_cast<TableImageSet*>(proposed.set);
			    proposed.pair == &pair && proposed.hash == words_hash && std::ranges::equal(built->words, words) &&
			    !(!built->pending_images.empty() && built->proposed_segment != spec->closed.size() &&
			      Spec::SetCpuDirtyStopOn()))
				return static_cast<TableImageSet*>(proposed.set);
		if (Catalog().m_native_xpr == nullptr || CatalogTable() == nullptr) return worker_refuse("worker set: no catalog");
		read_only.emplace();
	}
	auto       set      = std::make_unique<TableImageSet>();
	auto&      textures = m_context.GetTextureCache();
	// (Before the lookups: a registration they make over a depth image's pages is one IsSampledDepthCurrent sees.)
	const auto epoch    = textures.ResolutionEpoch();
	auto&      buffers  = m_context.GetBufferCache();
	const auto stages   = pair.programs[1] != nullptr ? 2u : 1u;
	const auto& sh      = buffer.GetShaders();
	const auto& ctx     = buffer.GetRegisters();
	set->words.assign(words.begin(), words.end());
	set->validated_frame = m_context.FrameNumber();
	const auto publish = [&]() -> TableImageSet* {
		if (spec != nullptr) {
			// A worker's: its own until committed (Speculation::After), then the catalog's, under the owner's write,
			// when the catalog did not change since (its slot free, the pair's sets not cleared); else kept aside.
			set->ValidatedFrame(Role())  = set->validated_frame;
			set->ResolutionEpoch(Role()) = set->resolution_epoch;
			set->validated_frame         = UINT64_MAX; // (the owner proves it in full at its first use)
			set->proposer                = spec->id;   // (before it is reachable: TableSet's foreign-proposal check)
			set->proposed_segment        = spec->closed.size();
			auto* const raw      = set.get();
			auto* const owner    = &Catalog();
			auto* const target   = &pair;
			const auto  since    = CatalogTable()->epoch;
			const bool  replaces = t_table_revalidating; // (spec_worker_revalidate: built for a set found no longer valid)
			auto        holder   = std::make_shared<std::unique_ptr<TableImageSet>>(std::move(set));
			spec->proposed_sets.push_back({&pair, words_hash, raw, spec->closed.size(), false,
			                               [owner, target, words_hash, since, holder, replaces](bool commit) {
				                               if (commit && TableAsideReleaseOn()) TableAsideDrain(owner->m_context.FrameNumber());
				                               // (spec_worker_images: never completed, never published or kept; the roles' pointers to it
				                               // dropped first, under the write lock their reads exclude; its descriptor set retired as the
				                               // owner retires one. No other translation binds it (an unpublished proposal is foreign to
				                               // them, TableSet), and its own segment's commit stopped before any replay that binds it ran.)
				                               if (*holder && !(*holder)->pending_images.empty()) {
					                               const auto write = owner->CatalogWrite();
					                               target->ForgetSet(holder->get());
					                               owner->NativeXprRetireSet((*holder)->pool, (*holder)->set);
					                               return;
				                               }
				                               if (commit && *holder) {
					                               const auto write = owner->CatalogWrite();
					                               auto*      table = owner->m_table_xpr.get();
					                               if (table != nullptr && table->epoch == since) {
						                               auto& slot = target->sets[words_hash];
						                               if (!slot) {
							                               (*holder)->published.store(true, std::memory_order_release);
							                               slot = std::move(*holder);
							                               return;
						                               }
						                               // (spec_worker_revalidate: the slot's set replaced as the owner's own
						                               // resolution replaces one.)
						                               if (replaces && slot.get() != holder->get()) {
							                               owner->NativeXprRetireSet(slot->pool, slot->set);
							                               target->ClearSets();
							                               ++table->epoch;
							                               (*holder)->published.store(true, std::memory_order_release);
							                               slot = std::move(*holder);
							                               return;
						                               }
					                               }
				                               }
				                               if (*holder && TableAsideReleaseOn()) { // (spec_aside_release)
					                               {
						                               const auto write = owner->CatalogWrite();
						                               target->ForgetSet(holder->get());
					                               }
					                               auto&            aside = TableAside();
					                               std::scoped_lock lock(aside.mutex);
					                               aside.queued.push_back({owner->m_context.FrameNumber(), holder, [owner, holder] {
						                                                       owner->NativeXprRetireSet((*holder)->pool, (*holder)->set);
					                                                       }});
					                               return;
				                               }
				                               TableKeepAside(holder);
			                               }});
			// (Its commit's settle is all catalog part, under the owner's write; one that retires a set (a revalidated set
			// replacing its slot's, the unpublished one of a set with pending images, read as the settle reads it) runs at
			// its commit.)
			spec->proposed_sets.back().catalog_owner   = owner;
			spec->proposed_sets.back().settle_must_now = [holder, replaces] {
				return replaces || (*holder && !(*holder)->pending_images.empty());
			};
			if (!raw->pending_images.empty()) {
				// (The images it completes, each with its serial and the slot's layout, and the set's handle: for its segment's
				// Before, and spec_set_cpu_dirty_stop off for the commits of later segments that bind it, CommitSegments'
				// reuse test.)
				spec->proposed_sets.back().complete = [owner, raw, spec, index = spec->proposed_sets.size() - 1] {
					const auto first = raw->images.size();
					const bool ok    = owner->TableCompletePendingImages(*raw);
					auto&      textures  = owner->m_context.GetTextureCache();
					auto&      completed = spec->proposed_sets[index].completed;
					const auto handle    = reinterpret_cast<uint64_t>(static_cast<VkDescriptorSet>(raw->set));
					const bool snapshot  = Spec::CompletedSnapshotOn();
					for (size_t i = first; i < raw->images.size(); ++i)
						if (const auto* image = textures.TryGetImage(raw->images[i].id)) {
							completed.push_back({image, raw->images[i].serial, handle, raw->images[i].layout});
							// (spec_completed_snapshot: its state as this completion left it, every slot of the set done.)
							if (snapshot) {
								const Image::StateLock lock(*image);
								completed.back().snapshot     = true;
								completed.back().state        = image->backing.state;
								completed.back().subresources = image->backing.subresource_states;
							}
						}
					return ok;
				};
				// (The memory its completion uploads from, for the commit's test before it against the committing segment's
				// writes (Speculation::Before): each pending slot's own desc range, metadata and stencil plane, and the images
				// over the slot's bytes as FindImage would meet them. Read only, the GPU thread.)
				spec->proposed_sets.back().pending_over = [raw](uint64_t address, uint64_t size) {
					for (const auto& pending: raw->pending_images) {
						const auto& data = pending.desc.info.data;
						if (data.address < address + size && address < data.address + data.size) return true;
					}
					return false;
				};
				spec->proposed_sets.back().pending_written =
				    [owner, raw](const std::function<bool(uint64_t, uint64_t)>& written) {
					    auto& textures = owner->m_context.GetTextureCache();
					    for (const auto& pending: raw->pending_images) {
						    const auto& info = pending.desc.info;
						    if (written(info.data.address, info.data.size) ||
						        (info.HasMetadata() && written(info.metadata.range.address, info.metadata.range.size)) ||
						        (info.HasStencil() && written(info.stencil.address, info.stencil.size)) ||
						        textures.ImagesOverWritten(info.data.address, info.data.size, written))
							    return true;
					    }
					    return false;
				    };
			}
			// (spec_proposed_check: what TableValidateSet's first steps ask, at the commit: the BDA page table's generation
			// the set's, and each image the one it was built with.)
			spec->proposed_sets.back().check = [owner, raw] {
				if (raw->bda_generation != BufferCache::BdaTableGeneration()) return false;
				auto& textures = owner->m_context.GetTextureCache();
				for (const auto& dep: raw->images) {
					const auto* image = textures.m_slot_images.try_get(dep.id);
					if (image == nullptr || image->serial != dep.serial) return false;
				}
				// (Mode 4: each storage image still one the owner's FindTexture would change nothing on here, before the
				// replay that binds the set (else the segment stops: the normal path refreshes or marks it); then the GPU
				// write that FindTexture notes (CommitGpuWrite), as the owner's build of the set would have. A colour target
				// bound as storage too is a target dep with the shader write in target_read (KYTY_TABLE_TARGET_READ): one as
				// well.)
				if (raw->worker_storage) {
					const auto written = [](const TableImageSet::Image& dep) {
						return dep.storage || (dep.target && (dep.target_read & vk::AccessFlagBits2::eShaderWrite));
					};
					for (const auto& dep: raw->images)
						if (written(dep) && TextureCache::StorageChanges(textures.m_slot_images[dep.id]) != nullptr) return false;
					for (const auto& dep: raw->images)
						if (written(dep)) {
							const auto& data = textures.m_slot_images[dep.id].info.data;
							Spec::NoteGpuWrite(data.address, data.size);
						}
				}
				return true;
			};
			return raw;
		}
		const auto write = CatalogWrite();
		const Weave::Hold hold; // (the pair's sets a weave worker finds a set in)
		auto&      slot  = pair.sets[XXH3_64bits(words.data(), words.size() * sizeof(uint32_t))];
		if (slot) {
			if (Weave::Active()) Weave::NoteFreed(slot.get());
			NativeXprRetireSet(slot->pool, slot->set);
			pair.ClearSets();
			++m_table_xpr->epoch;
		}
		slot = std::move(set);
		return slot.get();
	};
	const auto refuse = [&](const char* reason) -> TableImageSet* {
		if (spec != nullptr) return worker_refuse("worker set: refused set");
		static std::atomic<uint32_t> reports[2] {}; // (dispatches, draws)
		if (reports[pair.ps != 0 ? 1 : 0].fetch_add(1, std::memory_order_relaxed) < 32)
			std::printf("Table: VS 0x%llx PS 0x%llx image set refused: %s\n", static_cast<unsigned long long>(pair.vs),
			            static_cast<unsigned long long>(pair.ps), reason);
		set->refused = true;
		return publish();
	};
	// (Mode 4: a speculation's program lookup runs with the catalog's read lock dropped, below: taken back here, through
	// the writer-first gate, as the worker entry's lookups and capture take it.)
	const auto relock = [this] {
		CatalogGate(Catalog().m_catalog_writer_waiting, CatalogWriterFirst());
		t_catalog_read->lock();
	};
	ShaderVertexInputInfo  vertex_info;
	ShaderPixelInputInfo   pixel_info;
	ShaderComputeInputInfo compute_info;
	std::array<const ShaderStageRuntime*, 2> runtimes {&vertex_info.stage, &pixel_info.stage};
	if (variant != nullptr) {
		ResetNativeVertexInput(vertex_info);
		ResetNativeStageInput(pixel_info);
		// RULE (from a freeze in a long run): a speculation thread never holds a lock the GPU thread can take (the
		// catalog read, the pipeline cache m_mutex, the texture cache m_lock, the descriptor pools alloc_mutex) across anything
		// that can fault into tracked guest memory or send a sync to the GPU thread: a fault there waits for the GPU thread,
		// which waits for the lock. Under these locks a worker reads guest memory only non-faulting with a refusal (SRT words:
		// t_no_fault_reads here; SRT spans: TryReadGpuShaderSpan guarded copy); shader code is GPU-written too (a level load's
		// DMA: a freeze in Get's table job copy), copied before m_mutex (CopyShaderCode); anything else before the lock is taken. New worker paths (pair proposals) are audited against this list.
		PipelineCache::t_no_fault_reads  = spec != nullptr;
		PipelineCache::t_no_fault_failed = false;
		// (A speculation's lookup runs under the catalog's read lock (TableTry / TableContinue / TableDispatch hold it across
		// it: nothing releases it here), which the GPU thread's CatalogWrite waits for: its guest reads (PrepareProgramInto's
		// hash, AGC metadata and vertex tables, Get's code and SRT words) are made without a fault, the set refused when
		// one cannot be (PipelineCache: "worker set: code unread", "GPU-written shader code"; t_no_fault_failed). The
		// epoch and variant checks after it hold for the lock; they stay for a lookup made without it: mode 4 drops it
		// around the lookup, and proves the rest of what the draw holds as well, below.)
		// (Mode 4: the lock dropped around the lookup, so the GPU thread's catalog writes (TableStore in holes) do not wait
		// for it. EXACT: the lookup reads nothing of the catalog (the variant's mapping and the pair's portable masks are
		// the copies held below, the rest is the draw's registers, guest memory and the pipeline cache under its own lock).
		// Once the lock is back, everything the draw holds of the catalog and reads after it is proven as it was: the epoch
		// (TableReset, a set replaced or retired; the sweeps of pairs and states run only between translations), the
		// variant found again at the same address (the vector did not grow) with its pipeline, mapping and fetch_embedded
		// (TableStore's in-place update), the pair's portable masks (TableLearnPortable's in-place update), the draw state's
		// capture (NativeXprCaptureState replaces a state in place: capture_id moves). The pair's programs are compared
		// after, as before. Anything moved: the set refused ("worker set: catalog changed", the draw a hole), never built
		// over a mix; the draw's references may be stale then, so no retry in the hold.)
		// (Copies held by a speculation only. Mode 0 reads them in place.)
		struct Held {
			std::remove_cvref_t<decltype(variant->mapping)> mapping;
			std::remove_cvref_t<decltype(variant->state)>   state;
			std::remove_cvref_t<decltype(variant->pipeline)> pipeline; // (an in-place update meanwhile, below)
			std::remove_cvref_t<decltype(pair.portable)>    portable;
		};
		std::optional<Held> held;
		if (spec != nullptr) held.emplace(Held {variant->mapping, variant->state, variant->pipeline, pair.portable});
		const auto* mapping  = held ? &held->mapping : &variant->mapping;
		const auto* portable = held ? &held->portable : &pair.portable;
		const auto  since    = spec != nullptr && CatalogTable() != nullptr ? CatalogTable()->epoch : 0;
		const bool  unlocked = spec != nullptr && t_catalog_read != nullptr && t_catalog_read->owns_lock();
		const bool     outside      = unlocked && kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4;
		const bool     fetch_held   = outside && variant->fetch_embedded;
		const uint64_t capture_held = outside && draw_state != nullptr ? draw_state->capture_id : 0;
		if (outside) t_catalog_read->unlock();
		(void)m_context.GetPipelineCache().GetGraphicsPrograms(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx,
		                                                       buffer.GetUserConfig(), *mapping, stages == 2,
		                                                       vertex_info, pixel_info, true, *portable);
		if (outside) relock();
		PipelineCache::t_no_fault_reads = false;
		if (spec != nullptr && PipelineCache::t_no_fault_failed) return worker_refuse("worker set: program read");
		if (unlocked) {
			if (CatalogTable() == nullptr || CatalogTable()->epoch != since) return worker_refuse("worker set: catalog changed");
			// (The owner's TableStore adds a variant under its write without moving the epoch: a pair's variants vector that
			// grew meanwhile moved the variant TableDraw holds by reference, and reads after this set (its pipeline, its
			// index as the role's last variant). Refused then, as when the variant left.)
			const auto again = std::ranges::find(pair.variants, held->state, &TableVariant::state);
			if (variant != t_entry_variant && (again == pair.variants.end() || &*again != variant)) // (a worker entry's own)
				return worker_refuse("worker set: catalog changed");
			// (TableStore's targets path updates a variant in place, without moving the epoch or
			// the vector: its pipeline swapped, or null while TryCreateGraphicsPipeline is pending, or the variant refused.
			// The set's layout and TableDraw's bind read that pipeline: refused unless it is the one held before the unlock.)
			if (variant != t_entry_variant && (again->pipeline != held->pipeline || again->pipeline == nullptr || again->refused))
				return worker_refuse("worker set: catalog changed");
			// (Mode 4, the lock dropped for the lookup: the rest of what the draw read before the drop and reads after it, as
			// it was: the variant's other in-place fields (TableStore), the pair's portable masks (the lookup's input), the
			// draw state's capture (NativeXprCaptureState replaces a state in place, at its address).)
			if (outside && variant != t_entry_variant && (again->mapping != held->mapping || again->fetch_embedded != fetch_held))
				return worker_refuse("worker set: catalog changed");
			if (outside && pair.portable != held->portable) return worker_refuse("worker set: catalog changed");
			if (outside && draw_state != nullptr && draw_state->capture_id != capture_held)
				return worker_refuse("worker set: catalog changed");
		}
		if (m_context.GetPipelineCache().TablePending())
			return spec != nullptr ? worker_refuse("worker set: program pending") : nullptr;
		if (vertex_info.stage.program != pair.programs[0] || (stages == 2 && pixel_info.stage.program != pair.programs[1])) {
			// A format of a buffer's V# alone: the buffers learned as portable, as TableBlocks learns them when it sees a
			// format change (it never runs for a draw whose set is refused), and the pair compiled again (respecialize,
			// TableReset) instead of the set refused for RetryFrames, its draws native. (Demon's Souls at the view of
			// 9 Nov: 11 pairs at spawn; ~38 refused draws a frame gone, s1-10 fps +2.4, settled +2.2.)
			if (TableLearnPortable() != 0 && spec == nullptr) {
				bool                      other = false;
				const ShaderStageRuntime* got[2] {&vertex_info.stage, &pixel_info.stage};
				std::array<uint32_t, 2>   masks {};
				for (uint32_t st = 0; st < stages && !other; ++st) {
					if (got[st]->program == pair.programs[st]) continue;
					const auto mask = m_context.GetPipelineCache().TableFormatOnlyBuffers(pair.programs[st], got[st]->program);
					if (mask == UINT32_MAX || mask == 0) {
						other = true;
						break;
					}
					const auto& buffers = pair.programs[st]->info.buffers;
					for (uint32_t i = 0; i < 32; ++i)
						if (((mask >> i) & 1u) != 0 && (i >= buffers.size() || !ShaderRecompiler::IR::RuntimeBufferFormat(buffers[i])))
							other = true;
					masks[st] = mask;
				}
				if (!other && ((pair.portable[0] | masks[0]) != pair.portable[0] || (pair.portable[1] | masks[1]) != pair.portable[1])) {
					const auto write = CatalogWrite();
					const Weave::Hold hold;
					pair.portable[0] |= masks[0];
					pair.portable[1] |= masks[1];
					pair.respecialize = true;
					return nullptr; // (no set this draw: the next one compiles the pair again)
				}
			}
			// (KYTY_SPEC_PAIR_DENSE_INDIRECT: counted for a pair with a dense root: its candidates' shapes another program's.)
			if (!pair.programs[0]->table_plan.indirect.empty() ||
			    (stages == 2 && !pair.programs[1]->table_plan.indirect.empty()))
				LiveCounters::Add(LiveCounters::SpecPairDenseShapeRefused);
			return refuse(vertex_info.stage.program != pair.programs[0] ? "another specialization (VS)"
			                                                            : "another specialization (PS)");
		}
	} else {
		ResetNativeStageInput(compute_info);
		compute_info.dispatch_thread_dimensions = pair.compute->dispatch_thread_dimensions;
		auto&      pipelines = m_context.GetPipelineCache();
		PipelineCache::t_no_fault_reads  = spec != nullptr;
		PipelineCache::t_no_fault_failed = false;
		// (Held by a speculation only, as above.)
		std::optional<std::remove_cvref_t<decltype(pair.portable[0])>> held_portable;
		if (spec != nullptr) held_portable.emplace(pair.portable[0]);
		const auto* portable0 = held_portable ? &*held_portable : &pair.portable[0];
		const auto since     = spec != nullptr && CatalogTable() != nullptr ? CatalogTable()->epoch : 0;
		const bool unlocked  = spec != nullptr && t_catalog_read != nullptr && t_catalog_read->owns_lock();
		// (Mode 4: the lock dropped around the lookup, as above. A dispatch's pair changes in place only its portable
		// masks; its compute input, programs and pipeline with a TableReset, which moves the epoch.)
		const bool outside   = unlocked && kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4;
		if (outside) t_catalog_read->unlock();
		const auto handle    = pipelines.GetComputeProgram(sh.GetCs(), ctx.GetShaderRegisters(), compute_info, nullptr, true,
		                                                   *portable0);
		if (outside) relock();
		PipelineCache::t_no_fault_reads = false;
		if (spec != nullptr && PipelineCache::t_no_fault_failed) return worker_refuse("worker set: program read");
		if (unlocked && (CatalogTable() == nullptr || CatalogTable()->epoch != since))
			return worker_refuse("worker set: catalog changed");
		if (outside && pair.portable[0] != *held_portable) return worker_refuse("worker set: catalog changed");
		if (pipelines.TablePending()) return spec != nullptr ? worker_refuse("worker set: program pending") : nullptr;
		if (compute_info.stage.program != pair.programs[0]) {
			if (spec != nullptr) return worker_refuse("worker set: other specialization");
			// Another specialization the table way alike (a particle simulation's texture 2D in one emitter and 3D in
			// another, at Latria 90 programs): the set is made for it, with its pipeline.
			const auto* other = compute_info.stage.program;
			if (!handle || other == nullptr || !TableInterchangeable(*pair.programs[0], *other))
				return refuse("another specialization");
			set->pipeline = pipelines.TryCreateComputePipeline(compute_info, handle);
			if (set->pipeline == nullptr) {
				pair.waiting = pipelines.PipelineWaiting(); // (asked again once it is done)
				return nullptr;
			}
			set->program = other;
		}
		runtimes[0] = &compute_info.stage;
	}
	std::array<PreparedBindings, 2>                prepared;
	TextureCache::t_pending_images_ok = spec != nullptr && WorkerImagesOn(); // (spec_worker_images)
	for (uint32_t st = 0; st < stages; ++st) PrepareBindingsInto(*runtimes[st], prepared[st]);
	for (uint32_t st = 0; st < stages; ++st) RebindImages(prepared[st]);
	TextureCache::t_pending_images_ok = false;
	if (spec != nullptr && TextureCache::ReadOnlyFailed() != nullptr) {
		ResetBindings();
		return worker_refuse(TextureCache::ReadOnlyFailed());
	}
	// (Mode 4, a worker's set with a storage image: TextureCache::FindImageReadOnly.) A storage image the set also binds
	// otherwise (either stage) is one the owner's build marks force_general and shader_write (BindImage), from this set's
	// bindings alone (the owner's build binds both stages' images after ResetBindings, and resets them after; a target's
	// binding marks no force_general; a null image none): `twice`, laid out below as the owner's build lays it out
	// (General, whole, read and written; a target's attachment layout). A sampled binding left pending holds the null image
	// here: of the storage image's bytes, it would be bound twice and its commit would make it read-only while the storage
	// dep makes it General: refused.
	bool                 worker_storage = false;
	std::vector<ImageId> twice;
	if (spec != nullptr)
		for (uint32_t st = 0; st < stages; ++st)
			for (const auto& binding: prepared[st].images) {
				if (binding.desc.type != TextureCache::BindingType::Storage) continue;
				worker_storage = true;
				for (uint32_t other = 0; other < stages; ++other)
					for (const auto& sampled: prepared[other].images) {
						if (sampled.pending && sampled.pending_desc != nullptr) {
							const auto& data  = sampled.pending_desc->info.data;
							const auto& bytes = textures.GetImage(binding.image_id).info.data;
							if (!bytes.Empty() && data.address < bytes.End() && bytes.address < data.End()) {
								ResetBindings();
								return worker_refuse("worker storage: bound twice, pending");
							}
						}
						if (sampled.image_id == binding.image_id && sampled.desc.type != TextureCache::BindingType::Storage &&
						    !textures.GetImage(binding.image_id).info.data.Empty() &&
						    std::ranges::find(twice, binding.image_id) == twice.end())
							twice.push_back(binding.image_id);
					}
			}
	// An address probe's indirect image (a dispatch's): its root's elements in the order of the candidates the set's
	// key holds (TableEvaluate's), each found among the dispatch's own (the normal path's materialization: the same
	// enumeration, in the order the records name them); past them the null images it was padded with.
	if (variant == nullptr) {
		const auto& program = set->program != nullptr ? *set->program : *pair.programs[0];
		for (const auto& entry: program.table_plan.indirect) {
			const auto& elements = program.info.images.at(entry.root).indirect_resources;
			const auto& own      = compute_info.stage.resources.images;
			const auto& wanted   = OwnTable().indirect[0];
			auto&       images   = prepared[0].images;
			if (wanted.size() > elements.size()) return refuse("indirect candidates differ");
			const auto original = images;
			for (size_t j = 0; j < wanted.size(); ++j) {
				const auto found = std::ranges::find_if(elements, [&](uint32_t resource) {
					return resource < own.size() && own[resource] == wanted[j];
				});
				if (found == elements.end() || elements[j] >= images.size() || *found >= original.size())
					return refuse("indirect candidates differ");
				images[elements[j]] = original[*found];
			}
		}
	} else {
		// KYTY_SPEC_PAIR_DENSE_INDIRECT: a vertex or pixel program's dense indirect image (a draw's), as the dispatch's
		// above, per stage: its root's elements in the order of the candidates the set's key holds (TableEvaluate's: key
		// 0's first, the rest by their words), each found among the stage's own (the normal path's materialization,
		// MaterializeDenseIndirectImage: each distinct T# in the order of the first key that names it), past them the
		// null images it was padded with. Required: the block's mapping (TableEvaluate) names element positions in the
		// key's order; bound in the specialization's order instead, a key whose T# sorts elsewhere would sample another
		// key's texture. (No stage has an indirect image without the switch: a pair's programs, never a module, decide.)
		for (uint32_t st = 0; st < stages; ++st) {
			const auto& program = *pair.programs[st];
			if (program.table_plan.indirect.empty()) continue;
			LiveCounters::Add(LiveCounters::SpecPairDenseSetMisses);
			const auto differ = [&] {
				LiveCounters::Add(LiveCounters::SpecPairDenseSetDiffer);
				ResetBindings();
				return refuse("indirect candidates differ");
			};
			// (Each root's candidates, in plan order, one after another in TableEvaluate's `indirect`: its count of them.)
			const auto& all    = OwnTable().indirect[st];
			const auto& counts = OwnTable().indirect_counts[st];
			if (counts.size() != program.table_plan.indirect.size()) return differ();
			size_t first = 0, t = 0;
			for (const auto& entry: program.table_plan.indirect) {
				const auto& elements = program.info.images.at(entry.root).indirect_resources;
				const auto& own      = runtimes[st]->resources.images;
				if (first + counts[t] > all.size()) return differ();
				const auto wanted = std::span<const ShaderRecompiler::IR::DescriptorValue>(all).subspan(first, counts[t]);
				first += counts[t++];
				auto& images = prepared[st].images;
				if (wanted.size() > elements.size()) return differ();
				const auto original = images;
				for (size_t j = 0; j < wanted.size(); ++j) {
					const auto found = std::ranges::find_if(elements, [&](uint32_t resource) {
						return resource < own.size() && own[resource] == wanted[j];
					});
					if (found == elements.end() || elements[j] >= images.size() || *found >= original.size())
						return differ();
					images[elements[j]] = original[*found];
				}
			}
		}
	}
	// Each image's layout and access as the normal path's CommitBindings gives them: a null image (a texture not
	// streamed in yet) as it is bound (not held); a target of the draw state in its attachment layout and access
	// (NativeXprEmitTargets moves it there); one bound both as a texture and as storage (force_general: the deferred
	// decals' G-buffer) wholly in General; storage in General; else read-only.
	std::array<std::vector<TableImageSet::Image>, 2> deps;
	bool                                             ok = true;
	for (uint32_t st = 0; st < stages; ++st) deps[st].reserve(prepared[st].images.size()); // (no regrowth: new sets come in bursts)
	for (uint32_t st = 0; st < stages && ok; ++st)
		for (auto& binding: prepared[st].images) {
			auto&       image   = textures.GetImage(binding.image_id);
			// (spec_worker_sets: the set's own bindings' state, which a speculation's read-only build does not mark (BindImage):
			// the image's fields are the GPU thread's draw. An image the set binds both ways, `twice`, is what BindImage marks
			// force_general and shader_write; its other images' shader_write changes nothing below, a storage binding's own
			// test before it.)
			const bool  twice_here    = spec != nullptr && std::ranges::find(twice, binding.image_id) != twice.end();
			const bool  bound_general = spec == nullptr ? image.binding.force_general : twice_here;
			const bool  bound_written = spec == nullptr ? image.binding.shader_write : twice_here;
			const bool  storage = binding.desc.type == TextureCache::BindingType::Storage;
			const auto  found   = draw_state == nullptr ? decltype(draw_state->targets.end()) {}
			                                                : std::ranges::find(draw_state->targets, binding.image_id,
			                                                                    &NativeXprDrawState::Target::id);
			const auto* target  = draw_state == nullptr || found == draw_state->targets.end() ? nullptr : &*found;
			const auto& view    = binding.desc.view_info;
			auto&       dep     = deps[st].emplace_back();
			dep                 = {binding.image_id, textures.m_slot_images[binding.image_id].serial,
			                       vk::ImageLayout::eUndefined, {view.base_level, view.level_count, view.base_layer,
			                                                     view.layer_count}, binding.desc};
			if (image.info.data.Empty() && !storage && image.backing.state.layout == vk::ImageLayout::eGeneral) {
				binding.layout = vk::ImageLayout::eGeneral;
				continue; // (dep.layout undefined: not held)
			}
			// (A sampled aspect the attachment writes is not shader-readable: the target's access then lacks it. A
			// linear image the shader writes is enrolled for readback at every acquisition: TrackImageDownload.)
			// KYTY_TABLE_TARGET_READ: a target sampled in a layout that lets the shader read what it samples (the checks of
			// the normal path's CommitBindings) is held with the shader access added, as that path transits it.
			const bool target_read = [&] {
				if (target == nullptr || (target->access & vk::AccessFlagBits2::eShaderRead) ||
				    kyty_local_table_target_read.load(std::memory_order_relaxed) == 0)
					return false;
				const auto layout = target->layout;
				// (A colour target bound as storage too, in General: the deferred decals' G-buffer, as CommitBindings.)
				if (storage) return !image.info.IsDepth() && layout == vk::ImageLayout::eGeneral;
				if (layout == vk::ImageLayout::eGeneral) return true;
				const bool pixel_feedback = layout == vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT && stages == 2 && st == 1;
				if (!image.info.IsDepth()) return pixel_feedback;
				const auto aspect     = binding.desc.view_info.aspect;
				const bool depth_read = pixel_feedback || layout == vk::ImageLayout::eDepthReadOnlyOptimal ||
				                        layout == vk::ImageLayout::eDepthStencilReadOnlyOptimal ||
				                        layout == vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal;
				const bool stencil_read = layout == vk::ImageLayout::eStencilReadOnlyOptimal ||
				                          layout == vk::ImageLayout::eDepthStencilReadOnlyOptimal ||
				                          layout == vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal;
				return (!(aspect & vk::ImageAspectFlagBits::eDepth) || depth_read) &&
				       (!(aspect & vk::ImageAspectFlagBits::eStencil) || stencil_read);
			}();
			if (image.info.data.Empty() ||
			    (target != nullptr && !target_read && (storage || !(target->access & vk::AccessFlagBits2::eShaderRead))) ||
			    ((storage || (bound_general && bound_written)) && !image.info.IsTiled())) {
				ok = false;
				break;
			}
			if (target != nullptr) {
				binding.layout  = target->layout;
				dep.target_read = target_read ? vk::AccessFlagBits2::eShaderRead |
				                                    (storage || bound_written ? vk::AccessFlagBits2::eShaderWrite
				                                                                           : vk::AccessFlags2 {})
				                              : vk::AccessFlags2 {};
				dep.access      = target->access | dep.target_read;
				dep.target      = true;
			} else if (bound_general && !image.info.IsDepth()) {
				binding.layout = vk::ImageLayout::eGeneral;
				dep.access     = vk::AccessFlagBits2::eShaderRead |
				                 (bound_written ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlags2 {});
				dep.storage    = bound_written;
				dep.whole      = true;
			} else if (storage) {
				binding.layout = vk::ImageLayout::eGeneral;
				dep.access     = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
				dep.storage    = true;
			} else {
				binding.layout = image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                      : vk::ImageLayout::eShaderReadOnlyOptimal;
			}
			dep.layout = binding.layout;
		}
	ResetBindings();
	if (!ok) return refuse("an image no set can hold");
	// The set: per stage its bindings (images, samplers, LOD statistics, page table, fault buffer).
	struct Pending {
		vk::WriteDescriptorSet write;
		size_t                 start = 0;
		bool                   image = false;
	};
	std::vector<Pending>                  pending;
	std::vector<vk::DescriptorBufferInfo> buffer_infos;
	std::vector<vk::DescriptorImageInfo>  image_infos;
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	const auto fixed = [](const Buffer* value) -> vk::DescriptorBufferInfo { return {value->Handle(), 0, value->Size()}; };
	set->bda_generation = BufferCache::BdaTableGeneration(); // (read before the table: BufferCache::GrowBdaTable)
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = set->program != nullptr ? *set->program : *pair.programs[st];
		std::vector<uint32_t> occurrences(prepared[st].images.size(), 0);
		for (const auto& binding: program.bindings.descriptors) {
			Pending item;
			item.write.dstBinding      = ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind);
			item.write.descriptorType  = NativeDescriptorType(binding.kind);
			item.write.descriptorCount = NativeDescriptorCount(binding);
			item.image = ShaderRecompiler::IR::ImageBindingResourceClass(binding.kind) !=
			                 ShaderRecompiler::IR::ImageResourceClass::None ||
			             binding.kind == Kind::Samplers;
			item.start = item.image ? image_infos.size() : buffer_infos.size();
			if (binding.kind == Kind::Samplers) {
				for (const auto resource: binding.resources)
					image_infos.emplace_back(prepared[st].samplers.at(resource), nullptr, vk::ImageLayout::eUndefined);
			} else if (item.image) {
				uint32_t element = 0;
				for (const auto resource: binding.resources) {
					const auto& bound = prepared[st].images.at(resource);
					// (spec_worker_images: the slot the commit completes; the null image's view meanwhile.)
					if (bound.pending && occurrences.at(resource) == 0) {
						const auto& desc = *bound.pending_desc;
						set->pending_images.push_back({item.write.dstBinding, element, item.write.descriptorType, desc,
						                               desc.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
						                                                   : vk::ImageLayout::eShaderReadOnlyOptimal,
						                               desc.info.HasMetadata()});
					}
					image_infos.push_back(MakeImageInfo(bound, occurrences.at(resource)++));
					++element;
				}
			} else if (binding.kind == Kind::LodStats) {
				buffer_infos.push_back(fixed(buffers.GetLodStatsBuffer()));
			} else if (binding.kind == Kind::BdaPagetable) {
				buffer_infos.push_back(fixed(buffers.GetBdaPageTableBuffer()));
			} else if (binding.kind == Kind::FaultBuffer) {
				buffer_infos.push_back(fixed(buffers.GetFaultBuffer()));
			} else {
				return refuse("a binding table mode does not have");
			}
			pending.push_back(item);
		}
		if (st == 0) set->images.reserve(deps[0].size() + deps[1].size());
		for (const auto& dep: deps[st])
			if (dep.layout != vk::ImageLayout::eUndefined) {
				set->images.push_back(dep);
				set->image_ids.push_back(dep.id);
				set->holds_target |= dep.target;
				set->read_after_write |= dep.whole && dep.storage;
				// (Its view's metadata, which may not be the image's own: KYTY_TABLE_SERIALS.)
				if (dep.desc.info.metadata.kind != ImageMetadataKind::None) {
					if (spec != nullptr) return worker_refuse("worker set: metadata view");
					m_context.GetTextureCache().NoteMetaUse(dep.desc.info.metadata.range.address, dep.id.index);
				}
			}
	}
	set->set = (spec != nullptr ? Catalog() : *this).NativeXprAllocateSet((variant != nullptr      ? variant->pipeline
	                                 : set->pipeline != nullptr ? set->pipeline
	                                                            : pair.pipeline)
	                                    ->descriptor_set_layout,
	                                set->pool);
	std::vector<vk::WriteDescriptorSet> writes;
	for (auto& item: pending) {
		item.write.dstSet = set->set;
		if (item.image) item.write.pImageInfo = image_infos.data() + item.start;
		else item.write.pBufferInfo = buffer_infos.data() + item.start;
		writes.push_back(item.write);
	}
	// Written now, not queued in order with the recorded commands (the recording stream's descriptor writes): a
	// speculation's thread binds the set once it is published, which was before the stream wrote it. No recorded
	// command names the set: a new one, or one retired at a tick completed since.
	if (!writes.empty())
		m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr
#ifdef KYTY_LOCAL_VULKAN_RECORDING
		                                                    , LocalVulkanRecording::DirectDispatch()
#endif
		);
	set->resolution_epoch = epoch;
	set->worker_storage   = worker_storage;
	LiveCounters::Add(LiveCounters::TableEvaluations);
	return publish();
}

// What FindTexture uploads before an image is used with `desc`: guest bytes the CPU wrote, a GPU buffer write over them,
// or one over its stencil plane (of a sampled view, only into its levels: TextureCache::RefreshPending).
static bool TableRefreshPending(const Image& image, const TextureCache::ImageDesc& desc) {
	return TextureCache::RefreshPending(image, desc);
}

// spec_set_cpu_dirty_stop: a speculation's entry for a set's image (Image::SpeculativeEntry) made before its refresh
// test, so that what the entry notes as found is what the test saw. Only for a registered image. Every entry joins the
// segment's images, which its commit restores (Before: RestoreState records a barrier on the image's VkImage) and leaves
// (After: its barrier state), and Changed's "images registered" proves only that none of them was unregistered SINCE the
// speculation began (skipped while the resolution epoch did not move). A catalog set can still hold an image unregistered
// before that (an area travel's unmap: DeleteImage -> RetireImage, erased once the packets in flight then passed,
// TextureCache::EraseRetired; its VkImage destroyed with it), and TableValidateSet's proof refuses it only after this
// entry: the entry was kept, its barrier recorded in Before, and the image could be erased before that command buffer
// ran. The normal path never makes one: a set's unregistered image fails TableValidateSet's proof first.
static void TableEarlyEntry(Image& image) {
	if (::kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0 && Spec::SetCpuDirtyStopOn() && image.registered)
		(void)image.SpeculativeEntry();
}

// A set's images are the same objects, registered and current (as NativeXprValidateResources proves a record's):
// once a frame, and again when registrations moved. The main executor checks a use it proved again only when its
// image's validity moved or a registration stamped the view's pages since (1-1: ~1500 sets a frame, ~0.5 ms, nearly
// none changed; the registrations, every frame, one depth surface's memory taken as a color target and back), and
// every use in full once in TouchTicks flips or frames, which touches it (KYTY_TABLE_SET_PROOFS: 0 checks all of
// them in full, 2 does and counts the uses a proof held for that did not pass).
// (A use is held by its proof while its image's validity is the one proven and, when registrations moved, none stamped
// the view's pages since the proof: `registered`, the texture cache's answer for the view, asked only then.)
template <typename Use, typename Held, typename Registered>
[[nodiscard]] static bool TableUseHeld(const Use& dep, const Held& image, bool moved, Registered&& registered) {
	return dep.proven && image.validity == dep.validity && (!moved || !registered());
}
bool RenderExecutor::TableValidateSet(TableImageSet& set) {
	auto&      textures   = m_context.GetTextureCache();
	const auto frame      = m_context.FrameNumber();
	const auto resolution = textures.ResolutionEpoch();
	if (set.bda_generation != BufferCache::BdaTableGeneration()) return false; // (the page table grew since)
	// (A worker's proof is never older than role 0's latest: the registrations since role 0's are what can have moved
	// the views; its own stays where its last pass left it otherwise, and a registration after it failed it for good.)
	if (Role() != 0) {
		auto& mine = set.ResolutionEpoch(Role());
		mine        = std::max(mine, set.RoleZeroEpoch());
	}
	if (set.ValidatedFrame(Role()) == frame && set.ResolutionEpoch(Role()) == resolution) return true;
	LiveCensus::Scope census(LiveCensus::DrawPhase, 0x7ab1e, 7);
	const auto mode  = kyty_local_table_set_proofs.load(std::memory_order_relaxed);
	const bool own   = Role() == 0 && Spec::Current() == nullptr && mode != 0;
	const bool full  = !own || textures.GcTick() >= set.touched_tick + TableImageSet::TouchTicks ||
	                  textures.CurrentFrame() >= set.touched_frame + TableImageSet::TouchTicks ||
	                  m_context.GetCommandScheduler().Current().IsInvalid();
	// Registrations since: their pages say which proofs they reach (none, when the log of the latest has none over the
	// views' span: the game moves registrations every frame somewhere).
	bool moved = set.ResolutionEpoch(Role()) != resolution;
	if (moved && own) {
		if (set.span_end == 0) {
			set.span_begin = UINT64_MAX;
			for (const auto& dep: set.images) {
				set.span_begin = std::min(set.span_begin, dep.desc.info.data.address);
				set.span_end   = std::max(set.span_end, dep.desc.info.data.End());
			}
		}
		bool known = false;
		moved      = set.span_end <= set.span_begin ||
		        textures.RegistrationsOverlapSince(set.span_begin, set.span_end - set.span_begin, set.resolution_epoch, known) ||
		        !known;
	}
	if (!full)
		for (const auto& dep: set.images) textures.m_slot_images.prefetch(dep.id);
	for (auto& dep: set.images) {
		auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr || image->serial != dep.serial) return false;
		const bool held = !full && TableUseHeld(dep, *image, moved, [&] {
			const auto& data = dep.desc.info.data;
			return textures.RegistrationsSince(data.address, data.size, set.resolution_epoch);
		});
		if (held && mode != 2) continue;
		const auto refuse = [&] {
			if (held) {
				LiveCounters::Add(LiveCounters::TableProofMismatches);
				static std::atomic<uint32_t> reports {0};
				if (reports.fetch_add(1, std::memory_order_relaxed) < 16)
					std::printf("Table: proven image use failed: addr=0x%llx size=0x%llx serial=%llu validity=%llu\n",
					            static_cast<unsigned long long>(dep.desc.info.data.address),
					            static_cast<unsigned long long>(dep.desc.info.data.size),
					            static_cast<unsigned long long>(dep.serial), static_cast<unsigned long long>(dep.validity));
			}
			return false;
		};
		// The guest rewrote it (a texture streaming in), or a GPU buffer write its bytes or its stencil plane: upload it
		// as the normal path's FindTexture does. (A speculation's entry for it first, as TableTransit's, for a registered
		// image only: TableEarlyEntry.)
		TableEarlyEntry(*image);
		if (TableRefreshPending(*image, dep.desc)) {
			if (held) (void)refuse();
			if (Spec::Current() != nullptr) return Spec::Refuse("texture upload"), false;
			if (!image->registered || image->depth_id || image->binding.needs_rebind) return false;
			(void)textures.FindTexture(dep.id, dep.desc);
			image = textures.m_slot_images.try_get(dep.id);
			if (image == nullptr || image->serial != dep.serial || TableRefreshPending(*image, dep.desc)) return false;
		}
		if (dep.target) {
			if (!image->registered || image->binding.needs_rebind) return refuse();
			(void)textures.GetImage(dep.id); // LRU
			if (own) {
				dep.proven   = true;
				dep.validity = image->validity;
			}
			continue;
		}
		// (And its LRU, as FindTexture's. A DCC image written as storage, a downsampling chain's levels: TableTransit
		// checks its pending clears at every use, as a sampled one's.) A depth image sampled as a texture (the scene's
		// depth and stencil read by lighting and post passes: at 1-1 their sets were made again every other frame) is
		// found by other rules than a texture's: the proof is that nothing was registered over its view since the set
		// was last proven (or made).
		bool sampled = textures.IsSampledImageCurrent(dep.id, dep.desc, true);
		bool current = sampled || textures.IsSampledDepthCurrent(dep.id, dep.desc, set.ResolutionEpoch(Role()));
		const bool texture = dep.desc.type == TextureCache::BindingType::Texture;
		// (Mode 4: a view of an image ResolveOverlap picked (a worker's set, FindImageReadOnly) fails IsSampledImageCurrent's
		// own-backing proof; role 0's own validation proves it by the lookup below, any other by IsOverlapImageCurrent: the
		// candidates unchanged since this role's last proof, the pick the same.)
		if (!current && !own && (texture || dep.desc.type == TextureCache::BindingType::Storage) &&
		    kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4 &&
		    textures.IsOverlapImageCurrent(dep.id, dep.desc, set.ResolutionEpoch(Role())))
			current = true;
		if (!current && own && (texture || dep.desc.type == TextureCache::BindingType::Storage) &&
		    !m_context.GetCommandScheduler().Current().IsInvalid()) {
			// A surface the game uses as depth and as colour in turn (0x28a010000 at 1-1 and Latria, twice a frame): the
			// form a use leaves is parked (TextureCache::ParkImage), registrations over it move every frame, and the sets
			// using it failed here at every use (made again, 30-70 us each, the dispatch on the normal path meanwhile). The
			// normal path's lookup takes the set's image back (TakeParkedImage: the same id and serial, its views kept, the
			// other form's bytes copied in): the set then binds what that lookup and its acquisition bind. (The lookup is
			// the proof for a depth-tiled view too, which IsSampledImageCurrent declines: the surface's storage use.)
			// (A view the set's lookup moved into a larger image, its first layer in that image (on AMD GPUs a texture in the
			// layers a texture pool's views hold), is no request of its own: looked up again it ended the emulator (issue
			// #102, an RX 9070 XT). Its image is the one the set holds, and the checks below are its proof.)
			auto       desc  = dep.desc;
			const bool taken = !TextureCache::ViewFitsImage(desc) ||
			                   (textures.FindImage(desc) == dep.id && desc.view_info == dep.desc.view_info);
			image            = textures.m_slot_images.try_get(dep.id); // (the lookup may have moved the slots)
			if (taken && image != nullptr && image->serial == dep.serial && image->registered && !image->depth_id &&
			    !image->binding.needs_rebind) {
				if (texture && (dep.desc.info.IsDepth() || image->info.IsDepth())) {
					current = textures.IsSampledDepthCurrent(dep.id, dep.desc, textures.ResolutionEpoch());
				} else {
					(void)textures.FindTexture(dep.id, dep.desc);
					sampled = textures.IsSampledImageCurrent(dep.id, dep.desc, true);
					current = sampled || (!dep.desc.info.HasMetadata() && !dep.desc.info.HasStencil());
				}
			}
		}
		if (!current) return refuse();
		// (A DCC image's surface and a depth image's view are proven otherwise: checked at every validation.)
		if (own) {
			dep.proven   = sampled && dep.desc.info.metadata.kind != ImageMetadataKind::Dcc;
			dep.validity = image->validity;
		}
	}
	// (The epoch before the checks: a registration they made, FindTexture's, stamps pages a later check looks at.)
	set.ValidatedFrame(Role()) = frame;
	if (Role() == 0)
		std::atomic_ref<uint64_t>(set.resolution_epoch).store(resolution, std::memory_order_relaxed); // (workers read it)
	else
		set.ResolutionEpoch(Role()) = resolution;
	if (own && full) {
		set.touched_tick  = textures.GcTick();
		set.touched_frame = textures.CurrentFrame();
	}
	return true;
}

// TableValidateSet's first answer, or the one it reaches having changed nothing but the stamps it writes after the
// checks: every use held by its proof (KYTY_TABLE_SET_PROOFS=1, the full check not due). Images are registered,
// unregistered and their slots erased on the GPU thread, a slot only when no weave worker may be in a draw that saw it
// (Weave::HoldsSlot; a worker asks nothing of a retiring one, Weave::ImageRetiring); an image's validity is atomic. The
// set's proof stamps are the GPU thread's, read as hints. The registration log is not read here (the GPU thread appends
// to it and erases from it): moved registrations ask each view's pages.
bool RenderExecutor::TableSetProven(const TableImageSet& set) const {
	auto&      textures   = m_context.GetTextureCache();
	const auto frame      = m_context.FrameNumber();
	const auto resolution = textures.ResolutionEpoch();
	if (set.bda_generation != BufferCache::BdaTableGeneration()) return false;
	auto&      stamps    = const_cast<TableImageSet&>(set);
	const auto validated = std::atomic_ref<uint64_t>(stamps.validated_frame).load(std::memory_order_relaxed);
	const auto proven    = std::atomic_ref<uint64_t>(stamps.resolution_epoch).load(std::memory_order_relaxed);
	if (validated == frame && proven == resolution) return true;
	if (kyty_local_table_set_proofs.load(std::memory_order_relaxed) != 1 ||
	    textures.GcTick() >= set.touched_tick + TableImageSet::TouchTicks ||
	    textures.CurrentFrame() >= set.touched_frame + TableImageSet::TouchTicks)
		return false;
	const bool moved = proven != resolution;
	for (const auto& dep: set.images) {
		const auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr || image->serial != dep.serial ||
		    !TableUseHeld(dep, *image, moved, [&] {
			    return textures.RegistrationsSince(dep.desc.info.data.address, dep.desc.info.data.size, proven);
		    }))
			return false;
	}
	return true;
}

// An operand of the stage's code address (a V# of constants the code embeds), and its word: the low or high dword of
// the pair's code address (the stage's registers named it) plus the operand's offset.
static bool TableCodeOperand(const ShaderRecompiler::IR::TablePlan::Operand& operand) {
	using Kind = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	return operand.kind == Kind::ShaderBaseLow || operand.kind == Kind::ShaderBaseHigh;
}
static uint32_t TableCodeWord(const ShaderRecompiler::IR::TablePlan::Operand& operand, uint64_t code) {
	const auto address = code + operand.value;
	return operand.kind == ShaderRecompiler::IR::TablePlan::Operand::Kind::ShaderBaseLow ? static_cast<uint32_t>(address)
	                                                                                    : static_cast<uint32_t>(address >> 32u);
}

// KYTY_SPEC_PAIR_DENSE_INDIRECT (live spec_pair_dense_indirect, default 0; KYTY_SPECULATE=4 only; the pipeline cache
// sets CompileOptions::table_dense_indirect from it): a vertex or pixel table program's dense indirect image (T#s at heap +
// table_offset + key * 32, the key below key_bound: IR::EnterTableMode) enumerated as resource materialization does
// (IR::EnumerateDense: MaterializeDenseIndirectImage, the normal path's function, over the heap address the block's slots
// give), with its counts. False: the normal path takes the draw. The enumeration's guest reads are the materialization
// readers' (MaterializationReaders, as the address probe's): 1 KiB clean span reads of the table, word by word where a
// span is refused, all through TryReadGpuCleanBackingToHost, which refuses GPU-dirty or GPU-modified bytes and, on a
// speculation's thread, its own written or journaled ranges (the enumeration then fails: the normal path) and notes the
// rest as the speculation's clean reads (SpeculationReadsClean / CleanReadKept): the commit re-checks them as it does
// TableEvaluate's slot reads (NativeXprReader::Read -> ReadJournaled, noted clean): a hole or processor write over them
// stops the commit before the reading segment (an unvalued read: any write; ReadJournaled's are valued). Its range sync
// (MakeRangeReadable) refuses the speculation over GPU-written bytes (SyncGpuCleanBackingSpec); on the GPU thread it reads
// them back, as the normal path's materialization does at that draw.
[[gnu::noinline]] static bool TableEnumerateDense(const ShaderRecompiler::IR::TablePlan::IndirectImage& entry,
                                                  const ShaderRecompiler::IR::ImageResource& root, uint64_t heap,
                                                  const ShaderRecompiler::IR::SrtRuntime&    runtime,
                                                  ShaderRecompiler::IR::AddressProbeCandidates& found) {
	using Plan = ShaderRecompiler::IR::TablePlan;
	// (KYTY_TABLE_INDIRECT=0 keeps indirect images off the table path: dispatches are declined for one, these draws too.)
	if (kyty_local_table_indirect_mode.load(std::memory_order_relaxed) == 0)
		return LiveCounters::Add(LiveCounters::SpecPairDenseRefusedIndirectOff), false;
	if (!ShaderRecompiler::IR::EnumerateDense(entry.probe, root, heap, runtime, found))
		return LiveCounters::Add(LiveCounters::SpecPairDenseUnreadable), false;
	const auto count = static_cast<uint32_t>(found.descriptors.size());
	uint32_t   valid = 0;
	for (const auto& descriptor: found.descriptors)
		if (std::ranges::any_of(descriptor.dwords, [](uint32_t word) { return word != 0; })) ++valid;
	if (valid > 15) LiveCounters::Add(LiveCounters::SpecPairDenseWide);
	if (found.dropped_shapes != 0) LiveCounters::Add(LiveCounters::SpecPairDenseShapeDrops);
	if (count > Plan::IndirectCapacity) return LiveCounters::Add(LiveCounters::SpecPairDenseOverCapacity), false;
	if (found.keys.size() > Plan::IndirectMaxKeys || found.candidates.size() != found.keys.size())
		return LiveCounters::Add(LiveCounters::SpecPairDenseRefusedKeys), false;
	if (count == 0) return LiveCounters::Add(LiveCounters::SpecPairDenseRefusedOther), false;
	LiveCounters::Add(LiveCounters::SpecPairDenseTaken);
	return true;
}
// KYTY_SPEC_PAIR_DENSE_INDIRECT: a dense candidate's words without the LOD statistics fields (NativeXprLodStatsMask), the
// form its order and the set's key take.
static std::array<uint32_t, 8> TableDenseLodMasked(const ShaderRecompiler::IR::DescriptorValue& candidate) {
	std::array<uint32_t, 8> masked;
	for (uint32_t w = 0; w < 8; ++w) masked[w] = candidate.dwords[w] & ~NativeXprLodStatsMask[w];
	return masked;
}
// KYTY_SPEC_PAIR_DENSE_INDIRECT: a pixel program's indirect elements' LOD statistics words. TableEvaluate pushes 0 for
// them (their plan descriptors have no words); element j (the root's indirect_resources[j]) takes candidates[j]'s, the T#
// the set binds there (TableResolveSet's order), as RebindBuffers gives a normal path element its own T#'s. The stage's
// words are the last `pushed` of `lod_stats`, by image; elements past the candidates (null images) keep 0. (`candidates`:
// this root's, of TableEvaluate's `out`.)
[[gnu::noinline]] static void TableDenseLodStats(const ShaderRecompiler::IR::CompiledShaderInfo&            program,
                                                 const ShaderRecompiler::IR::TablePlan::IndirectImage&       entry,
                                                 std::span<const ShaderRecompiler::IR::DescriptorValue>      candidates,
                                                 std::vector<uint32_t>&                                      lod_stats) {
	const auto pushed = std::min<size_t>(program.bindings.lod_stats_count, program.table_plan.images.size());
	if (lod_stats.size() < pushed || entry.root >= program.info.images.size()) return;
	const auto  base     = lod_stats.size() - pushed;
	const auto& elements = program.info.images[entry.root].indirect_resources;
	for (size_t j = 0; j < candidates.size() && j < elements.size(); ++j) {
		if (elements[j] >= pushed) continue;
		ShaderTextureResource desc;
		std::memcpy(&desc, candidates[j].dwords.data(), sizeof(desc));
		lod_stats[base + elements[j]] = desc.MipStatsCntEn() ? (0x80000000u | desc.MipStatsCntId() |
		                                                        (uint32_t(desc.BaseLevel()) << 8u) |
		                                                        (uint32_t(desc.MinLodWarn5()) << 12u))
		                                                     : 0u;
	}
}
// KYTY_SPEC_PAIR_DENSE_INDIRECT: a pixel program whose LOD statistics words are in its block (lod_stats_in_block, a
// program with a dense root): the stage's words (the last `words` of `lod_stats`, by image) copied to
// TablePlan::LodStatsBlockDword, where the shader reads them (TableDraw pushes none for it).
[[gnu::noinline]] static void TableDenseLodToBlock(const ShaderRecompiler::IR::TablePlan& plan,
                                                   const std::vector<uint32_t>& lod_stats, std::vector<uint32_t>& block) {
	const auto words = std::min<size_t>(plan.lod_stats_words, lod_stats.size());
	const auto at    = plan.LodStatsBlockDword();
	if (block.size() < at + plan.lod_stats_words) return;
	std::fill_n(block.begin() + at, plan.lod_stats_words, 0u);
	std::copy_n(lod_stats.end() - static_cast<ptrdiff_t>(words), words, block.begin() + at);
}

// The slots of the pair's stages, read into their blocks where the plans address them from the user data and lower
// slots (a pointer read from memory may hold anything: what is read through it must be mapped), and the T#/S# words
// they make (TableXpr::words, the set's key) and the pixel stage's LOD statistics words. False: something unmapped.
bool RenderExecutor::TableEvaluate(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	return TableEvaluate<Weave::Ordered>(pair, user_data);
}

// (Under Weave::Worker: into the worker's own table, read through its reader, which fails on a page it may not read: a
// hole. An indirect image's candidates are enumerated by the materialization's readers: the GPU thread's.)
template <typename P>
bool RenderExecutor::TableEvaluate(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	auto&      table     = TableOf<P>();
	const auto stages    = pair.programs[1] != nullptr ? 2u : 1u;
	if constexpr (P::worker) WeaveMiss(Weave::Miss::Tables);
	auto&      resources = m_context.GetGpuResources();
	if (const auto epoch = resources.MappingEpoch(); epoch != table.mapping_epoch) {
		table.mapped.fill(UINT64_MAX);
		table.mapping_epoch = epoch;
	}
	using Kind  = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto& words = table.words;
	words.clear();
	table.lod_stats.clear();
	std::conditional_t<P::worker, WeaveScratch::Reader, NativeXprReader> reader(resources);
	// A run's address, its pages proven mapped and its lines loading.
	// (A bounded run, read through a V#: within its range, NUM_RECORDS bytes or NUM_RECORDS strides, as the normal
	// path's evaluation, which fails past it: so does this one.)
	const auto locate = [&](uint64_t low, uint64_t high, uint32_t offset, int32_t immediate, uint32_t count,
	                        uint64_t& out, bool bounded = false, uint32_t records = 0) {
		if (bounded) {
			const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
			const auto size   = stride == 0 ? uint64_t {records} : uint64_t {stride} * records;
			if (immediate < 0 || static_cast<uint64_t>(immediate) + count * 4ull > size) return false;
		}
		const auto address = ((low | high << 32u) & 0x0000fffffffffffcull) + (offset & ~3u) +
		                     static_cast<uint64_t>(int64_t {immediate});
		for (auto page = address >> 14u; page <= (address + count * 4ull - 1) >> 14u; ++page) {
			auto& seen = table.mapped[page & 255u];
			if (seen == page) continue;
			if (!Libs::LibKernel::Memory::IsFullyMapped(page << 14u, 16384)) return false;
			seen = page;
		}
		for (auto line = address & ~63ull; line < address + count * 4ull; line += 64)
			__builtin_prefetch(reinterpret_cast<const void*>(line));
		out = address;
		return true;
	};
	// The LOD statistics fields of the T#s (the game reassigns the counters every frame) reach only the shader data
	// (as RebindBuffers computes it), not the set.
	const auto lod_stats = [&](const std::array<uint32_t, 8>& image) {
		ShaderTextureResource desc;
		std::memcpy(&desc, image.data(), sizeof(desc));
		table.lod_stats.push_back(desc.MipStatsCntEn() ? (0x80000000u | desc.MipStatsCntId() |
		                                                  (uint32_t(desc.BaseLevel()) << 8u) |
		                                                  (uint32_t(desc.MinLodWarn5()) << 12u))
		                                               : 0u);
	};
	// A stage from its plan's operands: for user data the flat form cannot index (fewer words than it reads, or more
	// than 64: one beyond them reads 0 here, the flat form's constant 0 stands for those of 64 and up).
	const auto by_operands = [&](uint32_t st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		auto&       slots   = table.blocks[st];
		const auto  value   = [&](const ShaderRecompiler::IR::TablePlan::Operand& operand) -> uint32_t {
			if (operand.kind == Kind::Immediate) return operand.value;
			if (operand.kind == Kind::Slot) return slots[operand.value];
			if (TableCodeOperand(operand)) return TableCodeWord(operand, st == 0 ? pair.vs : pair.ps);
			const auto index = operand.value - program.user_data_base;
			return index < user_data[st].size() ? user_data[st][index] : 0;
		};
		const auto& runs = pair.runs[st];
		for (size_t begin = 0, end = 0; begin < runs.size(); begin = end) {
			for (end = begin; end < runs.size() && runs[end].level == runs[begin].level; ++end) {
				const auto& run = runs[end];
				if (!locate(value(run.read.low), value(run.read.high), value(run.read.offset), run.read.immediate, run.count,
				            table.addresses[end], run.read.bounded, run.read.bounded ? value(run.read.records) : 0))
					return false;
			}
			for (size_t i = begin; i < end; ++i) reader.Read(table.addresses[i], runs[i].count, slots.data() + runs[i].first);
			if constexpr (P::worker)
				if (reader.Failed()) return WeaveMiss(Weave::Miss::Memory), false;
		}
		for (size_t i = 0; i < plan.images.size(); ++i) {
			const auto& descriptor = plan.images[i];
			std::array<uint32_t, 8> image {};
			for (uint32_t w = 0; w < descriptor.count; ++w) {
				image[w] = value(descriptor.words[w]);
				words.push_back(image[w] & ~NativeXprLodStatsMask[w]);
			}
			if (i < program.bindings.lod_stats_count) lod_stats(image);
		}
		for (const auto& descriptor: plan.samplers)
			for (uint32_t w = 0; w < descriptor.count; ++w) words.push_back(value(descriptor.words[w]));
		return true;
	};
	// An address probe's indirect image (IR::TablePlan::IndirectImage): its candidates as resource materialization
	// enumerates them (the same code and readers), in an order of their own (key 0's first, the rest by their words:
	// one set whatever order the records name them in), their count and words in the set's key, the key mapping (by
	// key) in the block. False: the normal path takes it (a table not readable, more candidates or keys than room).
	const auto indirect = [&](uint32_t st) {
		const auto& program = *pair.programs[st];
		auto&       out     = table.indirect[st];
		out.clear();
		if (program.table_plan.indirect.empty()) return true;
		table.indirect_counts[st].clear(); // (read for a stage with an indirect image only)
		thread_local const auto runtime = [] {
			ShaderRecompiler::IR::SrtRuntime value;
			MaterializationReaders(value);
			return value;
		}();
		const auto& slots   = table.blocks[st];
		const auto  operand = [&](const ShaderRecompiler::IR::TablePlan::Operand& value) -> uint32_t {
			if (value.kind == Kind::Immediate) return value.value;
			if (value.kind == Kind::Slot) return slots[value.value];
			if (TableCodeOperand(value)) return TableCodeWord(value, st == 0 ? pair.vs : pair.ps);
			const auto index = value.value - program.user_data_base;
			return index < user_data[st].size() ? user_data[st][index] : 0;
		};
		const auto address = [&](const ShaderRecompiler::IR::TablePlan::Operand& low,
		                         const ShaderRecompiler::IR::TablePlan::Operand& high) {
			return (uint64_t {operand(low)} | uint64_t {operand(high)} << 32u) & 0x0000ffffffffffffull;
		};
		using Plan = ShaderRecompiler::IR::TablePlan;
		for (const auto& entry: program.table_plan.indirect) {
			auto& found = table.probe;
			// (KYTY_SPEC_PAIR_DENSE_INDIRECT: a vertex or pixel program's dense table, key_bound set (EnterTableMode takes
			// no other form with one): TableEnumerateDense. The rest is the address probe's: the order, the mapping by
			// key, the words in the set's key.)
			if (entry.probe.key_bound != 0
			        ? !TableEnumerateDense(entry, program.info.images[entry.root], address(entry.heap_low, entry.heap_high),
			                               runtime, found)
			        : !ShaderRecompiler::IR::EnumerateAddressProbe(entry.probe, program.info.images[entry.root],
			                                                       address(entry.records_low, entry.records_high),
			                                                       address(entry.heap_low, entry.heap_high), runtime, found))
				return false;
			const auto count = static_cast<uint32_t>(found.descriptors.size());
			if (count == 0 || count > Plan::IndirectCapacity || found.keys.size() > Plan::IndirectMaxKeys ||
			    found.candidates.size() != found.keys.size())
				return false;
			std::array<uint32_t, Plan::IndirectCapacity> order {}, position {};
			for (uint32_t j = 0; j < count; ++j) order[j] = j;
			// (KYTY_SPEC_PAIR_DENSE_INDIRECT: a dense table's candidates ordered, and keyed below, by their words without the
			// LOD statistics fields, as plain images are keyed: the game reassigns the counters every frame, which never
			// change an image's resolution or view (NativeXprLodStatsMask), and a key of the full words would make a new
			// set every frame. Candidates equal without them resolve to the same image: either order binds the same.
			// The set's own reorder (TableResolveSet) and the LOD words (TableDenseLodStats) take the full words of `out`.)
			const bool dense = entry.probe.key_bound != 0;
			if (dense)
				std::sort(order.begin() + 1, order.begin() + count, [&](uint32_t a, uint32_t b) {
					return TableDenseLodMasked(found.descriptors[a]) < TableDenseLodMasked(found.descriptors[b]);
				});
			else
				std::sort(order.begin() + 1, order.begin() + count,
				          [&](uint32_t a, uint32_t b) { return found.descriptors[a].dwords < found.descriptors[b].dwords; });
			for (uint32_t j = 0; j < count; ++j) position[order[j]] = j;
			const auto first = out.size(); // (this root's candidates: out[first, first + count); several dense roots)
			for (uint32_t j = 0; j < count; ++j) out.push_back(found.descriptors[order[j]]);
			table.indirect_counts[st].push_back(count);
			const auto own = std::span<const ShaderRecompiler::IR::DescriptorValue>(out).subspan(first, count);
			std::array<uint32_t, Plan::IndirectMaxKeys> by_key {};
			const auto keys = static_cast<uint32_t>(found.keys.size());
			for (uint32_t k = 0; k < keys; ++k) by_key[k] = k;
			std::sort(by_key.begin(), by_key.begin() + keys,
			          [&](uint32_t a, uint32_t b) { return found.keys[a] < found.keys[b]; });
			auto* mapping = table.blocks[st].data() + entry.mapping;
			mapping[0]    = keys;
			for (uint32_t k = 0; k < keys; ++k) {
				mapping[1 + 2 * k] = found.keys[by_key[k]];
				mapping[2 + 2 * k] = position[found.candidates[by_key[k]]];
			}
			// (KYTY_SPEC_PAIR_DENSE_INDIRECT: a pixel program's elements have LOD statistics words (a compute program's
			// none): element j, which the set binds out[j] at, takes out[j]'s, as the normal path's element takes its
			// own T#'s (RebindBuffers); the stage's words are the last ones pushed above, by image; the padding's null: 0.)
			if (program.bindings.lod_stats_count != 0) TableDenseLodStats(program, entry, own, table.lod_stats);
			words.push_back(count);
			if (dense)
				for (const auto& candidate: own) {
					const auto masked = TableDenseLodMasked(candidate);
					words.insert(words.end(), masked.begin(), masked.end());
				}
			else
				for (const auto& candidate: own) words.insert(words.end(), candidate.dwords.begin(), candidate.dwords.end());
		}
		// (KYTY_SPEC_PAIR_DENSE_INDIRECT: a dense pixel program's LOD statistics words, complete now, into its block.)
		if (program.table_plan.lod_stats_words != 0) TableDenseLodToBlock(program.table_plan, table.lod_stats, table.blocks[st]);
		return true;
	};
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& form    = pair.packed[st];
		if constexpr (P::worker)
			if (!program.table_plan.indirect.empty()) return false;
		// (Slots no stage reads keep what an earlier draw left: the block carries them unread.)
		auto& slots = table.blocks[st];
		slots.resize(program.table_plan.BlockDwords());
		if (table.addresses.size() < pair.runs[st].size()) table.addresses.resize(pair.runs[st].size());
		if (!form.valid || form.runs.size() != pair.runs[st].size() || user_data[st].size() < form.user_words ||
		    user_data[st].size() > 64) {
			if (!by_operands(st) || !indirect(st)) return false;
			continue;
		}
		// The flat form (TablePair::packed): every operand one load from one of three arrays. A level's addresses are
		// known together: their lines load at once (the guest's tables are cold).
		const std::array<const uint32_t*, 3> bases {user_data[st].data(), slots.data(), form.constants.data()};
		const auto get = [&](uint16_t operand) { return bases[operand >> 14u][operand & 0x3fffu]; };
		uint32_t   begin = 0;
		for (const auto end: form.level_ends) {
			for (auto i = begin; i < end; ++i) {
				const auto& run     = form.runs[i];
				const bool  bounded = run.records != TablePair::PackedRun::NotBounded;
				if (!locate(get(run.address[0]), get(run.address[1]), get(run.address[2]), run.immediate, run.count,
				            table.addresses[i], bounded, bounded ? get(run.records) : 0))
					return false;
			}
			for (auto i = begin; i < end; ++i) reader.Read(table.addresses[i], form.runs[i].count, slots.data() + form.runs[i].first);
			if constexpr (P::worker)
				if (reader.Failed()) return WeaveMiss(Weave::Miss::Memory), false;
			begin = end;
		}
		const auto first = words.size();
		words.resize(first + form.image_words.size());
		auto*  out = words.data() + first;
		size_t at  = 0;
		for (size_t i = 0; i < form.image_counts.size(); ++i) {
			std::array<uint32_t, 8> image {};
			const uint32_t          count = form.image_counts[i];
			for (uint32_t w = 0; w < count; ++w) {
				image[w]    = get(form.image_words[at + w]);
				out[at + w] = image[w] & ~NativeXprLodStatsMask[w];
			}
			at += count;
			if (i < program.bindings.lod_stats_count) lod_stats(image);
		}
		for (; at < form.image_words.size(); ++at) out[at] = get(form.image_words[at]);
		if (!indirect(st)) return false;
	}
	return true;
}

// The image set of the words TableEvaluate made: the last one when they are the same, else the pair's set for them
// (`variant` null: a dispatch's), proven current. Null: the normal path takes the draw or dispatch.
RenderExecutor::TableImageSet* RenderExecutor::TableSet(CommandBuffer& buffer, TablePair& pair,
                                                        const TableVariant*       variant,
                                                        const NativeXprDrawState* draw_state) {
	return TableSet<Weave::Ordered>(buffer, pair, variant, draw_state);
}

template <typename P>
RenderExecutor::TableImageSet* RenderExecutor::TableSet(CommandBuffer& buffer, TablePair& pair,
                                                        const TableVariant*       variant,
                                                        const NativeXprDrawState* draw_state) {
	if constexpr (P::worker) {
		// A set is made, tried again, retired and swept by the GPU thread: a weave worker takes one the pair has for
		// the words, or declines the draw. The set its last draw of the pair took first; else the pair's sets, read
		// unlocked: the GPU thread holds it outside a draw to insert or erase one.
		WeaveMiss(Weave::Miss::Set);
		auto&       scratch = WeaveHere();
		const auto& words   = scratch.table.words;
		auto&       hint    = scratch.set_hints[(reinterpret_cast<uintptr_t>(&pair) * 0x9e3779b97f4a7c15ull) >> 56u];
		auto*       set     = hint.first == &pair ? hint.second : nullptr;
		if (set == nullptr || set->words != words) {
			const auto found = pair.sets.find(XXH3_64bits(words.data(), words.size() * sizeof(uint32_t)));
			set = found != pair.sets.end() && found->second->words == words ? found->second.get() : nullptr;
			if (set == nullptr) return nullptr;
			hint = {&pair, set};
		}
		// (Refused, or made for a page table that grew since, which TableValidateSet retires it for: a hole.) Proven as
		// TableValidateSet would find it with nothing to write but its stamps, or its validation is a need of the
		// turn's. Its last use and the pair's last sets are the turn's stamps.
		if (set->refused || set->bda_generation != BufferCache::BdaTableGeneration()) return nullptr;
		if (!scratch.SetProven(*this, *set)) scratch.NoteNeed(Weave::NeedKind::Validate, nullptr, set, 0);
		return set;
	}
	auto&       table = TableOf<P>();
	const auto& words = table.words;
	const auto  frame = m_context.FrameNumber();
	auto*       set   = pair.LastSet(Role());
	// (A set another translation proposed and its commit has not published: not taken from these caches, which only the
	// role's own translations fill: its pending slots are filled only if that translation commits its segment. Looked up
	// again below: the catalog holds only published ones.)
	const auto* const spec    = Spec::Current();
	const auto        foreign = [spec](const TableImageSet* s) {
		if (s == nullptr || s->proposer == 0 || (spec != nullptr && s->proposer == spec->id) ||
		    s->published.load(std::memory_order_acquire))
			return false;
		return true;
	};
	if (set == nullptr || set->words != words || foreign(set)) {
		auto* const last     = set;
		auto* const previous = pair.PreviousSet(Role());
		auto* const memo     = std::exchange(table.memo_set, nullptr); // (KYTY_TABLE_MEMO: the set the pair and key drew with)
		const bool  verify   = std::exchange(table.memo_verify, false);
		if (memo != nullptr && memo != last && !memo->refused && memo->words == words && !foreign(memo)) {
			set = memo;
			// (KYTY_TABLE_MEMO_VERIFY: the set the words' lookup finds, read only; another one is taken as the full
			// path would take it.)
			if (verify) {
				const auto found = pair.sets.find(XXH3_64bits(words.data(), words.size() * sizeof(uint32_t)));
				auto* const full = found != pair.sets.end() && found->second->words == words ? found->second.get() : nullptr;
				if (full != memo) {
					TableMemoChecked(4u, pair.vs, pair.ps, 0);
					set = full;
					if (set == nullptr || (set->refused && frame >= set->validated_frame + TableImageSet::RetryFrames))
						set = TableResolveSet(buffer, pair, variant, draw_state, words);
				}
			}
		// (The set before the last next, without the words' hash or the map's cold bucket and node.)
		} else if ((kyty_local_table_mru.load(std::memory_order_relaxed) & 4u) != 0 && previous != nullptr && previous != last &&
		    !previous->refused && previous->words == words && !foreign(previous)) {
			set = previous;
		} else {
			const auto found = pair.sets.find(XXH3_64bits(words.data(), words.size() * sizeof(uint32_t)));
			set              = found != pair.sets.end() && found->second->words == words ? found->second.get() : nullptr;
			// (KYTY_TABLE_LEARN_PORTABLE=2: a refused set whose retry found its program translating is due once that
			// translation is done.)
			const auto retry_due = [&](const TableImageSet& s) {
				if (s.retry_wait != nullptr && TableLearnPortable() == 2) return s.retry_wait->load(std::memory_order_acquire);
				return frame >= s.validated_frame + TableImageSet::RetryFrames;
			};
			if (set == nullptr || (set->refused && retry_due(*set))) {
				auto* const retried = set;
				set                 = TableResolveSet(buffer, pair, variant, draw_state, words);
				// (A retry that came back pending: the refused set is still the slot's.) (The GPU thread's only: workers
				// would write the shared_ptr under the shared lock while others read it. A speculation's TableResolveSet keeps
				// the catalog's read across its program lookup, its guest reads made without a fault; were it ever released
				// there, the owner could destroy `retried` meanwhile (TableReset, the set collection, a publish). Mode 4
				// does: `retried` is read below by the GPU thread's own call only.)
				if (Spec::Current() == nullptr && retried != nullptr && set == nullptr && TableLearnPortable() == 2 &&
				    m_context.GetPipelineCache().TablePending()) {
					// (Under the catalog's write lock: workers read retry_wait under the shared one, and this assignment can drop
					// the last reference to the old job's flag. The old one is released after the unlock.)
					auto       waiting = m_context.GetPipelineCache().TableWaiting();
					const auto write   = CatalogWrite();
					const Weave::Hold hold;
					retried->retry_wait.swap(waiting);
				}
			}
		}
		// (TableResolveSet may have cleared both: a set it replaced.)
		if (last != nullptr && pair.LastSet(Role()) == last) pair.PreviousSet(Role()) = last;
		pair.LastSet(Role()) = set;
		if (set == nullptr) return nullptr; // (a program still translating)
	}
	// spec_set_cpu_dirty_stop: not its own set with pending slots proposed in an earlier segment (the role's last,
	// previous or memo set). Those slots are completed in that segment's Before and are no image entry of this one, so
	// TableTransit refuses it; kept as the role's last set, every later draw with these words was refused too, each a
	// hole, for the rest of the translation. Resolved afresh instead, as a set first proposed in this segment (its slots
	// completed in this segment's Before; TableResolveSet passes over the earlier one) and the role's last set from here.
	if (spec != nullptr && set->proposer == spec->id && !set->pending_images.empty() &&
	    set->proposed_segment != spec->closed.size() && Spec::SetCpuDirtyStopOn()) {
		LiveCounters::Add(LiveCounters::SpecPendingReuseFreshN);
		set                  = TableResolveSet(buffer, pair, variant, draw_state, words);
		pair.LastSet(Role()) = set;
		if (set == nullptr) return nullptr; // (refused, or a program still translating)
	}
	set->used_frame.store(frame, std::memory_order_relaxed);
	if (set->refused) return LiveCounters::Add(LiveCounters::TableRefusedSets), Spec::Refuse("table: refused set"), nullptr;
	if (!TableValidateSet(*set)) {
		// spec_worker_revalidate (under KYTY_SPECULATE=4 only): a speculation resolves the set afresh itself, as it does a
		// set it finds none for (a proposed set: published at its commit, replacing the slot's, or kept aside; the
		// worker-set checks at the commit), instead of the hole. The set found is proven as one the lookup above found
		// (TableValidateSet); the set that just failed (one this translation proposed, found again by its words) is refused.
		if (Spec::Current() != nullptr && Role() != 0 && kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4) {
			t_table_revalidating = true;
			auto* const fresh    = TableResolveSet(buffer, pair, variant, draw_state, words);
			t_table_revalidating = false;
			if (fresh == nullptr || fresh == set) return Spec::Refuse("table: set validation"), nullptr;
			pair.LastSet(Role()) = fresh;
			if (fresh->refused)
				return LiveCounters::Add(LiveCounters::TableRefusedSets), Spec::Refuse("table: refused set"), nullptr;
			if (!TableValidateSet(*fresh)) return Spec::Refuse("table: set validation"), nullptr;
			fresh->used_frame.store(frame, std::memory_order_relaxed);
			set = fresh;
		} else if (Spec::Current() != nullptr) {
			return Spec::Refuse("table: set validation"), nullptr;
		} else { // (the owner's path, as before)
			// Resolved again at the next draw that uses it.
			const auto write = CatalogWrite();
			const Weave::Hold hold; // (the set erased: a weave worker finds sets in the pair's)
			if (table.last_set == set) table.last_set = nullptr;
			pair.ClearSets();
			++table.epoch;
			const auto found = pair.sets.find(XXH3_64bits(set->words.data(), set->words.size() * sizeof(uint32_t)));
			if (Weave::Active()) Weave::NoteFreed(set);
			NativeXprRetireSet(set->pool, set->set);
			if (found != pair.sets.end() && found->second.get() == set) pair.sets.erase(found);
			return nullptr;
		}
	}
	// Unused image sets leave once a second (the catalog's owner drops them).
	if (Role() == 0 && frame >= table.collected + 60) {
		const auto write = CatalogWrite();
		const Weave::Hold hold; // (sets erased)
		table.collected  = frame;
		for (auto* pairs: {&table.pairs, &table.dispatches})
			for (auto& [key, held]: *pairs)
				std::erase_if(held.sets, [&](auto& item) {
					if (item.second->used_frame.load(std::memory_order_relaxed) + 60 >= frame) return false;
					held.ClearSets();
					if (table.last_set == item.second.get()) table.last_set = nullptr;
					++table.epoch;
					if (Weave::Active()) Weave::NoteFreed(item.second.get());
					NativeXprRetireSet(item.second->pool, item.second->set);
					return true;
				});
	}
	return set;
}

// The pair's blocks (TableEvaluate read their slots): each stage's buffers' device addresses and sizes in dwords
// after its slots, in the table upload ring. A read-only V# is resolved once (TableXpr::resolved); a written one
// is obtained as written at every use, as the normal path's NativeStorageBuffer (the buffer cache tracks the GPU's
// writes). False: a buffer the shaders cannot take.
// The V#s' words and their resolved entries (TableXpr::entries), as soon as TableEvaluate made the slots they are read
// from: the entries' lines (a 4 MiB direct-mapped table: TLB misses too) load while the image set is found (TableSet).
void RenderExecutor::TableGather(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	TableGather<Weave::Ordered>(pair, user_data);
}

// (Under Weave::Worker: the worker's own entries.)
template <typename P>
void RenderExecutor::TableGather(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	using Kind        = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto&      table  = TableOf<P>();
	const auto stages = pair.programs[1] != nullptr ? 2u : 1u;
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		const auto& block   = table.blocks[st];
		auto&       entries = table.entries[st];
		entries.resize(plan.buffers.size());
		const auto& form = pair.packed[st];
		const bool  flat = form.valid && form.buffer_words.size() == 4 * plan.buffers.size() &&
		                  user_data[st].size() >= form.user_words && user_data[st].size() <= 64;
		const std::array<const uint32_t*, 3> bases {user_data[st].data(), block.data(), form.constants.data()};
		const bool memo = kyty_local_table_gather_memo.load(std::memory_order_relaxed) != 0; // (KYTY_TABLE_GATHER_MEMO)
		uint32_t   memo_hits = 0;
		for (size_t i = 0; i < plan.buffers.size(); ++i) {
			auto& [words4, entry] = entries[i];
			std::array<uint32_t, 4> fresh;
			if (flat) {
				for (uint32_t w = 0; w < 4; ++w) {
					const auto operand = form.buffer_words[4 * i + w];
					fresh[w]           = bases[operand >> 14u][operand & 0x3fffu];
				}
			} else {
				for (uint32_t w = 0; w < 4; ++w) {
					const auto& operand = plan.buffers[i][w];
					fresh[w] = operand.kind == Kind::Immediate ? operand.value
					           : operand.kind == Kind::Slot    ? block[operand.value]
					           : TableCodeOperand(operand)     ? TableCodeWord(operand, st == 0 ? pair.vs : pair.ps)
					           : operand.value - program.user_data_base < user_data[st].size()
					               ? user_data[st][operand.value - program.user_data_base]
					               : 0;
				}
			}
			// (The same words as gathered last at this index hash to the same entry: kept, no hash; the prefetch stays, the
			// line is what TableBlocks reads.)
			if (memo && entry != nullptr && words4 == fresh) {
				++memo_hits;
			} else {
				words4 = fresh;
				entry  = &table.resolved[XXH3_64bits(words4.data(), sizeof(words4)) & (TableXpr::ResolvedEntries - 1)];
			}
			__builtin_prefetch(entry);
		}
		if (memo_hits != 0) TableAdd<P>(LiveCounters::TableGatherMemoHits, memo_hits);
	}
	table.gathered = &pair;
}

// As ObtainBuffer: a small range the guest wrote, [address, address + size), read from a copy in the ring (nothing
// held: the main executor's later uses share it while it holds, TableXpr::RingCopy) instead of uploaded into its
// buffer: the copy of the range shared under the same terms, else a new one, kept for them. 0: no room, or the bytes
// unreadable. `shareable`: the copy may be shared and kept (TableBlocks says when). `observation` and `cpu`: the terms
// a held entry keeps it under (`cpu` 0: none, it is not shared). `copied`: a new copy was read (not a shared one).
// TableBlocks' and the weave turn's recopy of a range a worker held (the form without a policy: the GPU thread's, the
// main executor's, shareable). Under Weave::Worker: in the worker's slab, read through the copies listed before its
// parcel, shared within the parcel only.
vk::DeviceAddress RenderExecutor::TableRingCopy(uint64_t address, uint64_t size) {
	uint64_t observation = 0, cpu = 0;
	bool     copied      = false;
	return TableRingCopy<Weave::Ordered>(address, size, true, observation, cpu, copied);
}

template <typename P>
vk::DeviceAddress RenderExecutor::TableRingCopy(uint64_t address, uint64_t size, bool shareable, uint64_t& observation,
                                                uint64_t& cpu, bool& copied) {
	auto&       table        = TableOf<P>();
	auto&       buffer_cache = m_context.GetBufferCache();
	auto* const shared =
	    shareable ? &table.ring_copies[((address ^ (size << 40u)) * 0x9e3779b97f4a7c15ull) >> 52u] : nullptr;
	observation              = [&] {
		if constexpr (P::worker)
			return WeaveParcelSerial() | uint64_t {1} << 62u;
		else
			return PageManager::ObservationEpoch();
	}();
	const auto tick = P::worker ? WeaveParcelSerial() : m_context.GetCommandScheduler().CurrentTick();
	cpu             = buffer_cache.CpuModificationEpoch(address, size); // (before the copy)
	vk::DeviceAddress device = 0;
	if (shared != nullptr && cpu != 0 && shared->address == address && shared->size == size &&
	    shared->observation == observation && shared->tick == tick && shared->cpu_epoch == cpu) {
		device = shared->device;
		TableAdd<P>(LiveCounters::TableRingShared);
		TableAdd<P>(LiveCounters::TableRingSharedBytes, size);
	} else {
		auto& ring = [&]() -> auto& {
			if constexpr (P::worker)
				return WeaveHere().slab;
			else
				return buffer_cache.GetTableUploadBuffer();
		}();
		const auto read = [&](void* out) {
			if constexpr (P::worker)
				return WeaveHere().ReadBacking(address, out, size);
			else
				return Libs::LibKernel::Memory::TryReadBackingToHost(address, out, size);
		};
		if (auto [mapped, offset] = ring.Map(size, 16, false); mapped != nullptr && read(mapped)) {
			ring.Commit();
			copied = true;
			device = ring.BufferDeviceAddress() + offset;
			TableAdd<P>(LiveCounters::TableRingCopies);
			TableAdd<P>(LiveCounters::TableRingBytes, size);
			if (shared != nullptr && cpu != 0)
				*shared = {.address     = address,
				           .size        = size,
				           .observation = observation,
				           .tick        = tick,
				           .cpu_epoch   = cpu,
				           .device      = device};
		}
	}
	if (shared == nullptr) cpu = 0;
	return device;
}

bool RenderExecutor::TableBlocks(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data,
                                 std::array<vk::DeviceAddress, 2>& blocks, bool* writes) {
	return TableBlocks<Weave::Ordered>(pair, user_data, blocks, writes);
}

// Under Weave::Worker: the worker's own entries, ring copies and special generation; its blocks and ring copies in the
// parcel's slab, shared within the parcel only (WeaveParcelSerial). A V# the GPU thread would resolve through the
// buffer cache (ObtainBuffer: a lookup, an upload, a buffer made or registered) is taken from a live buffer that holds
// its range clean, as ObtainBuffer would return it with nothing to upload, the buffer's last use a stamp; anything
// else, a buffer the shaders write, or a format the pair isn't compiled for (a respecialization), is a hole. Every V#
// taken from a game buffer is noted with its range's CPU epoch (a held use, its five block words kept for the turn's
// recopy), every ring copy with its range.
template <typename P>
bool RenderExecutor::TableBlocks(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data,
                                 std::array<vk::DeviceAddress, 2>& blocks, bool* writes) {
	using Kind         = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto&      table        = TableOf<P>();
	auto&      buffer_cache = m_context.GetBufferCache();
	auto&      textures     = m_context.GetTextureCache();
	auto&      upload       = buffer_cache.GetTableUploadBuffer();
	// (Where the blocks and the ring copies go: the table upload ring, or a worker's slab.)
	auto& ring = [&]() -> auto& {
		if constexpr (P::worker)
			return WeaveHere().slab;
		else
			return upload;
	}();
	// (The ring copies' sharing terms: the GPU thread's observation epoch and tick; a worker's parcel.)
	const auto observation_now = [&] {
		if constexpr (P::worker)
			return WeaveParcelSerial() | uint64_t {1} << 62u;
		else
			return PageManager::ObservationEpoch();
	};
	const auto stages       = pair.programs[1] != nullptr ? 2u : 1u;
	if constexpr (P::worker) WeaveMiss(Weave::Miss::Buffers);
	const auto start_epoch  = textures.StartEpoch(); // (before any question below: a later image start moves it)
	const auto image_starts = [&](uint64_t address) {
		auto& known = table.no_image_start[(address * 0x9e3779b97f4a7c15ull) >> 54u];
		if (known.first == address && known.second == start_epoch) return false;
		if (textures.HasImageStartingAt(address)) return true;
		known = {address, start_epoch};
		return false;
	};
	if (table.resolved_start != start_epoch || table.resolved_mapping != table.mapping_epoch) {
		table.resolved_start   = start_epoch;
		table.resolved_mapping = table.mapping_epoch;
		++table.special_generation;
	}
	// A tracker region's epoch, read once here (the buffers mostly share one: the line is another core's).
	std::array<std::pair<uint64_t, uint64_t>, 4> region_epochs;
	region_epochs.fill({UINT64_MAX, 0});
	const auto cpu_epoch = [&](uint64_t address, uint64_t size) {
		const auto region = address / TRACKER_REGION_SIZE;
		if (region != (address + size - 1) / TRACKER_REGION_SIZE) return buffer_cache.CpuModificationEpoch(address, size);
		auto& known = region_epochs[region & 3u];
		if (known.first != region) known = {region, buffer_cache.CpuModificationEpoch(address, size)};
		return known.second;
	};
	// A range none of whose pages the guest wrote since they were uploaded (with no write protection still queued:
	// a page marked clean can change unseen until it lands).
	const auto range_clean = [&](uint64_t address, uint64_t size) {
		return TableRangeClean(buffer_cache, address, size);
	};
	// A null buffer's device address: valid memory (the shaders' loads of a buffer are not branched around its bounds:
	// out of bounds they load its first dword).
	const vk::DeviceAddress null_device = upload.BufferDeviceAddress();
	// A V#'s buffer word (IR::BufferWord, as IR::BufferDescriptorWord makes it) and stride. (Each format's class
	// once: the V#s of every draw's buffers asked for it.)
	static const auto format_classes = [] {
		std::array<uint32_t, 128> classes {};
		for (uint32_t format = 0; format < classes.size(); ++format)
			classes[format] = ShaderRecompiler::IR::BufferFormatClass(static_cast<Prospero::BufferFormat>(format));
		return classes;
	}();
	const auto word_of = [](const std::array<uint32_t, 4>& words4) -> std::pair<uint32_t, uint32_t> {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		if (descriptor.Type() != 0) return {0, 0};
		using Word = ShaderRecompiler::IR::BufferWord;
		return {(descriptor.Stride() <= Word::MaxStride ? descriptor.Stride() : 0u) |
		            format_classes[descriptor.RawFormat()] << Word::LayoutShift |
		            descriptor.DstSelXYZW() << Word::DstSelShift,
		        descriptor.Stride()};
	};
	// The bytes a V# names (0: none).
	const auto requested_of = [](const std::array<uint32_t, 4>& words4) -> uint64_t {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		if (descriptor.Type() != 0 || descriptor.Base48() == 0) return 0;
		return descriptor.Stride() != 0 ? uint64_t {descriptor.Stride()} * descriptor.NumRecords()
		                                : uint64_t {descriptor.NumRecords()};
	};
	// As RenderExecutor::FindBuffers: a V# that is no buffer, or of nothing mapped, is a null buffer (size 0).
	const auto range_of = [](const std::array<uint32_t, 4>& words4) -> std::pair<uint64_t, uint64_t> {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		const auto requested = descriptor.Stride() != 0 ? uint64_t {descriptor.Stride()} * descriptor.NumRecords()
		                                                : uint64_t {descriptor.NumRecords()};
		const auto address   = descriptor.Base48();
		return {address, descriptor.Type() != 0 || address == 0 || requested == 0
		                     ? 0
		                     : Libs::LibKernel::Memory::ClampRangeSize(address, requested)};
	};
	// A V# the program was specialized for (a format and dst_sel it decodes, no swizzle or ADD_TID; strides are the
	// block's): another one takes the normal path, which chooses the program for it.
	const auto specialized = [](const ShaderRecompiler::IR::BufferResource& resource,
	                            const std::array<uint32_t, 4>&             words4) {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		if (descriptor.Type() != 0) return true; // no buffer: read as a null one
		return !descriptor.SwizzleEnabled() && !descriptor.AddTid() &&
		       (!resource.formatted || resource.descriptor_format == Prospero::BufferFormat::kInvalid ||
		        (descriptor.Format() == resource.descriptor_format && descriptor.DstSelXYZW() == resource.descriptor_swizzle));
	};
	if (table.gathered != &pair) TableGather<P>(pair, user_data);
	table.gathered = nullptr; // (taken: a later draw or dispatch gathers its own)
	// (A weave worker's notes: a V# taken from a game buffer, its five block words placed once the stage's block is
	// made; a ring copy; a buffer whose last use the turn stamps.)
	const auto note_held = [&](uint32_t st, size_t i, uint64_t address, uint64_t size, uint64_t epoch) {
		if constexpr (P::worker)
			WeaveParcel().result->held_uses.push_back({.stage     = static_cast<uint8_t>(st),
			                                           .buffer    = static_cast<uint32_t>(i),
			                                           .word      = UINT32_MAX,
			                                           .address   = address,
			                                           .size      = size,
			                                           .cpu_epoch = epoch});
	};
	const auto note_ring = [&](uint64_t address, uint64_t size) {
		if constexpr (P::worker) WeaveParcel().result->ring_uses.push_back({.address = address, .size = size});
	};
	// spec_pending_buffers: the stage's V#s left to the commit (their dword in the block, range, written, texel).
	// (spec_reloc_refs: the V#s of found game buffers too, with the epoch their address was proven at.)
	struct PendingHere {
		uint32_t dword;
		uint64_t address, size;
		bool     written, texel;
		bool     pending = true;
		uint64_t since   = 0;
	};
	std::vector<PendingHere> pending_here;
	// (Outside a speculation, mode 0 always, none of the bookkeeping below is read: the flags and the lookup site are
	// read only by ObtainBufferSpec, a speculation's lookup. It is skipped there.)
	const bool               spec_here  = Spec::Current() != nullptr;
	const bool               pending_ok = spec_here && Spec::PendingBuffersOn();
	const bool               reloc      = pending_ok && Spec::RelocRefsOn();
	// (spec_reloc_refs: the V#'s own lookups note no buffer range, the entry stands for it; restored on every return.)
	struct NoRange {
		bool old;
		explicit NoRange(bool on): old(Spec::t_no_range) { Spec::t_no_range = on; }
		~NoRange() { Spec::t_no_range = old; }
		NoRange(const NoRange&)            = delete;
		NoRange& operator=(const NoRange&) = delete;
	};
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		auto&       block   = table.blocks[st];
		auto&       entries = table.entries[st];
		pending_here.clear();
		// The buffers it writes first: a range it also reads through another V# is GPU-written by then, as the
		// normal path's second lookup (RebindBuffers) sees it (not a copy of the guest's bytes in the ring).
		for (size_t n = 0; n < 2 * plan.buffers.size(); ++n) {
			const auto  i         = n < plan.buffers.size() ? n : n - plan.buffers.size();
			const auto& words4    = entries[i].first;
			const auto& resource  = program.info.buffers[i];
			const bool  formatted = resource.formatted;
			if ((resource.written || resource.atomic) != (n < plan.buffers.size())) continue;
			if (!specialized(resource, words4)) {
				if constexpr (P::worker) return false; // (the pair compiled again: the GPU thread's)
				if (Role() != 0) return Spec::Refuse("table: another format"), false;
				if (ShaderRecompiler::IR::RuntimeBufferFormat(resource) && i < 32 && ((pair.portable[st] >> i) & 1u) == 0) {
					const auto write = CatalogWrite();
					const Weave::Hold hold;
					pair.portable[st] |= 1u << i;
					pair.respecialize = true;
				}
				return false;
			}
			if (resource.written || resource.atomic) {
				// (Obtained as written at every use, its pages marked: the GPU thread's.)
				if constexpr (P::worker) return false;
				// The shader writes it: marked GPU-written now, whatever an earlier use proved. (A write hazard as
				// HasShaderBufferWrites sees one: a V# of a nonempty range.)
				if (writes != nullptr && resource.written && (uint64_t {words4[0]} | (words4[1] & 0xffffu)) != 0 &&
				    words4[2] != 0)
					*writes = true;
				auto& proof = table.written[XXH3_64bits(words4.data(), sizeof(words4)) & (TableXpr::WrittenEntries - 1)];
				if (proof.special != table.special_generation || proof.words != words4) {
					const auto [clamped_address, clamped_size] = range_of(words4);
					proof = {.words = words4, .special = table.special_generation, .address = clamped_address, .size = clamped_size};
				}
				const auto address = proof.address, size = proof.size;
				vk::DeviceAddress device = null_device;
				if (size != 0) {
					if (address % 4 != 0) return false;
					if (formatted && !proof.image_checked) {
						if (image_starts(address)) return false;
						proof.image_checked = true;
					}
					std::optional<BufferCache::AheadSite> ahead_site; // (the lookup's site: spec_ring_index)
					std::optional<NoRange>                no_range;
					uint64_t                              written_since = 0; // (spec_reloc_refs: before its lookup)
					if (spec_here) {
						ahead_site.emplace(2);
						written_since = buffer_cache.RegistrationEpoch();
						no_range.emplace(reloc);
						Spec::t_pending_ok  = pending_ok;
						Spec::t_pending_hit = false;
					}
					const auto [owner, offset] = buffer_cache.ObtainBuffer(address, size, true, formatted);
					if (spec_here) Spec::t_pending_ok = false;
					region_epochs.fill({UINT64_MAX, 0}); // (its uploads may have moved them)
					if (spec_here && owner == nullptr && Spec::t_pending_hit) {
						// (spec_pending_buffers: the null buffer, size 0, until the commit writes the range's.)
						if (formatted) textures.InvalidateMemoryFromGPU(address, size, "storage-buffer");
						pending_here.push_back({plan.BufferDword(i), address, size, true, formatted});
						const auto [word, stride]      = word_of(words4);
						block[plan.BufferDword(i)]     = static_cast<uint32_t>(null_device);
						block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(null_device >> 32u);
						block[plan.BufferDword(i) + 2] = 0;
						block[plan.BufferDword(i) + 3] = word;
						block[plan.BufferDword(i) + 4] = stride;
						continue;
					}
					if (owner == nullptr || buffer_cache.GetBufferIfLive(buffer_cache.FindBuffer(address, size)) != owner)
						return false;
					if (formatted) textures.InvalidateMemoryFromGPU(address, size, "storage-buffer");
					device = owner->BufferDeviceAddress() + offset;
					// (spec_reloc_refs: proven at the epoch read before its lookup.)
					if (reloc)
						pending_here.push_back({plan.BufferDword(i), address, size, true, formatted, false, written_since});
				}
				const auto [word, stride]      = word_of(words4);
				block[plan.BufferDword(i)]     = static_cast<uint32_t>(device);
				block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(device >> 32u);
				block[plan.BufferDword(i) + 2] = static_cast<uint32_t>(std::min<uint64_t>(size / 4, UINT32_MAX));
				block[plan.BufferDword(i) + 3] = word;
				block[plan.BufferDword(i) + 4] = stride;
				continue;
			}
			auto& held    = *entries[i].second;
			bool  current = held.words == words4 && (held.special == 0 || held.special == table.special_generation);
			if (current && (held.registration & TableXpr::RingCopyMark) != 0) {
				// Its ring copy (RingCopy's terms: the same observation epoch, the regions' epoch unchanged, no page
				// GPU-written).
				if (Role() == 0 && Spec::Current() == nullptr &&
				    held.registration == (TableXpr::RingCopyMark | observation_now()) &&
				    cpu_epoch(held.address, held.size) == held.cpu_epoch &&
				    !buffer_cache.IsRegionGpuModified(held.address, held.size)) {
					TableAdd<P>(LiveCounters::TableRingShared);
					TableAdd<P>(LiveCounters::TableRingSharedBytes, held.size);
					note_ring(held.address, held.size);
					const auto [word, stride]      = word_of(words4);
					block[plan.BufferDword(i)]     = static_cast<uint32_t>(held.device);
					block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(held.device >> 32u);
					block[plan.BufferDword(i) + 2] = held.dwords;
					block[plan.BufferDword(i) + 3] = word;
					block[plan.BufferDword(i) + 4] = stride;
					continue;
				}
				current = false;
			}
			// (A speculation's guest memory writes wait for its commit: such a range is resolved, ObtainBuffer decides.)
			if (const auto* spec = Spec::Current();
			    current && held.size != 0 && spec != nullptr && spec->JournalIntersects(held.address, held.size))
				current = false;
			if (const auto registration = buffer_cache.RegistrationEpoch();
			    current && held.size != 0 && held.registration != registration) {
				// Buffers were registered or unregistered since: it holds while none covered its tracker regions. (The
				// epoch read before the regions': a speculative translation's thread runs beside the render thread's
				// registrations. The render thread's own entries cannot meet one midway: it registers itself. A
				// worker's entries outlive its translation, in its executor's table, so an epoch read after a
				// registration began proved them for later translations too.)
				for (auto region = held.address / TRACKER_REGION_SIZE, last = (held.address + held.size - 1) / TRACKER_REGION_SIZE;
				     current && region <= last; ++region)
					// (Quiet inserts: a quiet insert moves no owner, so not the resolution epoch: BufferCache::ChangeRegister. A
					// weave worker keeps the epoch every change moves: on a live 0 -> 4 it can read the mode as 4 before the
					// floor moves and before the weave pauses, and an unregister made in mode 0 moved that epoch alone.)
					current = (P::worker ? buffer_cache.RegionRegistrationEpoch(region * TRACKER_REGION_SIZE)
					                     : buffer_cache.RegionResolutionEpoch(region * TRACKER_REGION_SIZE)) <= held.registration;
				if (current) held.registration = registration;
			}
			if (current && held.size != 0) {
				// Another page of its tracker regions was written: the range itself may still be clean.
				if (const auto epoch = cpu_epoch(held.address, held.size); epoch == 0 || epoch != held.cpu_epoch) {
					TableCensus<P> reprove(LiveCensus::DrawPhase, 0x7ab1e, 6);
					current = range_clean(held.address, held.size);
					// (Inside a speculation range_clean's BufferCache::IsRegionCpuModified answers clean for a range in the
					// translation's current_set, its pages CPU-dirty still: noted current by a packet a Rollback took back
					// since (Rollback cuts `current`, keeps current_set), by a translation dropped since, or by an earlier
					// segment of this one, whose notes the commit applies only if it keeps that segment (one it runs in
					// order as a hole instead applies none: Speculation::SegmentAsHole, no Before). The tracker's own bits
					// decide: dirty, no epoch is kept (re-proven at each use while dirty: a later translation, whose
					// current_set lacks the range, then resolves it afresh and notes it there) and it is noted current in
					// the open segment, as the fresh lookup's EnsureBufferContents notes it, so the segment's own commit
					// uploads it before the work, whatever sealed segments noted (NoteCurrent drops a repeat against the
					// open segment's notes only). Before, nothing uploaded it and the work read the buffer's stale bytes.
					// Any speculation mode.)
					if (current) {
						// (Behind the plain global, as BufferCache::IsRegionCpuModified's: mode 0, no thread-local read.)
						auto* spec = ::kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0 ? Spec::Current() : nullptr;
						const bool dirty = spec != nullptr && buffer_cache.IsRegionCpuDirty(held.address, held.size);
						held.cpu_epoch   = dirty ? 0 : epoch;
						if (dirty) spec->NoteCurrent(held.address, held.size);
					}
				}
			}
			if (!current) {
				TableCensus<P> resolve(LiveCensus::DrawPhase, 0x7ab1e, 5);
				// (Kept only once resolved: a refusal leaves nothing a later use could take.)
				held = {};
				TableXpr::ResolvedBuffer fresh {.words = words4, .device = null_device};
				const auto [address, size] = range_of(words4);
				if (formatted || size < requested_of(words4)) fresh.special = table.special_generation;
				if (size != 0) {
					// The shader reads whole dwords from the device address; a buffer an image starts at is made from
					// the image by ObtainBuffer only (SynchronizeBufferFromImage).
					if (address % 4 != 0 || (formatted && image_starts(address))) return false;
					// (A speculation's: the guest's writes, not those its commit makes current; with none of its own
					// guest memory writes pending over the range, read as no GPU work wrote it.)
					auto* const spec        = Spec::Current();
					const bool  cpu_written = buffer_cache.IsRegionCpuDirty(address, size) &&
					                         !buffer_cache.IsRegionGpuModified(address, size);
					if (cpu_written && size <= BufferCache::CACHING_PAGESIZE &&
					    (spec == nullptr || !spec->JournalIntersects(address, size))) {
						// As ObtainBuffer: a small range the guest wrote is read from a copy in the ring (nothing held: the
						// main executor's later uses share it while it holds, TableXpr::RingCopy), not uploaded into its
						// buffer.
						// (A weave worker's copy: in its parcel's slab, shared within the parcel only, TableRingCopy<P>. The
						// GPU thread's and a speculation's: the else branch, the copy's terms and notes in their order.)
						if constexpr (P::worker) {
							uint64_t   observation = 0, cpu = 0;
							bool       copied      = false;
							const auto device =
							    TableRingCopy<P>(address, size, spec == nullptr && Role() == 0, observation, cpu, copied);
							if (spec != nullptr && copied) spec->NoteCleanRead(address, size);
							// (The V#'s later uses take it from its held entry.)
							if (device != 0 && cpu != 0)
								held = {.words        = words4,
								        .dwords       = static_cast<uint32_t>(size / 4),
								        .special      = fresh.special,
								        .device       = device,
								        .address      = address,
								        .size         = size,
								        .cpu_epoch    = cpu,
								        .registration = TableXpr::RingCopyMark | observation};
							if (device != 0) {
								note_ring(address, size);
								const auto [word, stride]      = word_of(words4);
								block[plan.BufferDword(i)]     = static_cast<uint32_t>(device);
								block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(device >> 32u);
								block[plan.BufferDword(i) + 2] = static_cast<uint32_t>(size / 4);
								block[plan.BufferDword(i) + 3] = word;
								block[plan.BufferDword(i) + 4] = stride;
								continue;
							}
						} else {
							auto* const shared = spec == nullptr && Role() == 0
							                         ? &table.ring_copies[((address ^ (size << 40u)) * 0x9e3779b97f4a7c15ull) >> 52u]
							                         : nullptr;
							const auto observation = PageManager::ObservationEpoch();
							const auto tick        = m_context.GetCommandScheduler().CurrentTick();
							const auto cpu         = buffer_cache.CpuModificationEpoch(address, size); // (before the copy)
							vk::DeviceAddress device = 0;
							// (Mode 4: a speculation's copy used again within its translation, on the GPU thread's RingCopy
							// terms (the same range, observation epoch, tick and tracker regions' CPU epoch, none 0; the checks
							// above, CPU-dirty, not GPU-modified, no journaled write over it, made as before), in the same
							// segment and wait count, with no Rollback since. What keeps the bytes is what keeps the GPU
							// thread's share: the observation epoch, this thread's, which ProcessPm4Spec steps before every
							// packet that is not a draw, dispatch or register packet (FlushDeferredProtection), so a share
							// never crosses a label, write, wait or DMA. The use's note is the copy's (NoteCleanRead of the
							// same range: the commit's checks see the same lists); the shader reads the copy's bytes at its
							// address, in the same ring, fenced for the same commit.)
							TableXpr::SpecRingCopy* spec_shared = nullptr;
							if (spec != nullptr && cpu != 0 && ::kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4) {
								if (!table.spec_ring_copies)
									table.spec_ring_copies = std::make_unique<TableXpr::SpecRingCopy[]>(TableXpr::RingCopies);
								spec_shared = &table.spec_ring_copies[((address ^ (size << 40u)) * 0x9e3779b97f4a7c15ull) >> 52u];
							}
							const auto spec_waits = [&] { return uint64_t {spec->waits.size()} + spec->covered_waits; };
							const auto spec_own   = [&] {
								const auto& own = spec_shared->copy;
								return own.address == address && own.size == size && own.observation == observation &&
								       own.tick == tick && own.cpu_epoch == cpu && own.device != 0 && spec_shared->id == spec->id &&
								       spec_shared->segment == spec->closed.size() && spec_shared->rollbacks == spec->rollbacks &&
								       spec_shared->waits == spec_waits();
							};
							if (shared != nullptr && cpu != 0 && shared->address == address && shared->size == size &&
							    shared->observation == observation && shared->tick == tick && shared->cpu_epoch == cpu) {
								device = shared->device;
								LiveCounters::Add(LiveCounters::TableRingShared);
								LiveCounters::Add(LiveCounters::TableRingSharedBytes, size);
							} else if (spec_shared != nullptr && spec_own()) {
								device = spec_shared->copy.device;
								spec->NoteCleanRead(address, size);
							} else if (auto [mapped, offset] = upload.Map(size, 16, false);
							           mapped != nullptr && Libs::LibKernel::Memory::TryReadBackingToHost(address, mapped, size)) {
								upload.Commit();
								if (spec != nullptr) spec->NoteCleanRead(address, size);
								device = upload.BufferDeviceAddress() + offset;
								LiveCounters::Add(LiveCounters::TableRingCopies);
								LiveCounters::Add(LiveCounters::TableRingBytes, size);
								if (spec_shared != nullptr) // (after its note)
									*spec_shared = {.copy      = {.address     = address,
									                              .size        = size,
									                              .observation = observation,
									                              .tick        = tick,
									                              .cpu_epoch   = cpu,
									                              .device      = device},
									                .id        = spec->id,
									                .rollbacks = spec->rollbacks,
									                .waits     = spec_waits(),
									                .segment   = spec->closed.size()};
								if (shared != nullptr && cpu != 0)
									*shared = {.address     = address,
									           .size        = size,
									           .observation = observation,
									           .tick        = tick,
									           .cpu_epoch   = cpu,
									           .device      = device};
							}
							// (The V#'s later uses take it from its held entry.)
							if (device != 0 && shared != nullptr && cpu != 0)
								held = {.words        = words4,
								        .dwords       = static_cast<uint32_t>(size / 4),
								        .special      = fresh.special,
								        .device       = device,
								        .address      = address,
								        .size         = size,
								        .cpu_epoch    = cpu,
								        .registration = TableXpr::RingCopyMark | observation};
							if (device != 0) {
								const auto [word, stride]      = word_of(words4);
								block[plan.BufferDword(i)]     = static_cast<uint32_t>(device);
								block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(device >> 32u);
								block[plan.BufferDword(i) + 2] = static_cast<uint32_t>(size / 4);
								block[plan.BufferDword(i) + 3] = word;
								block[plan.BufferDword(i) + 4] = stride;
								continue;
							}
						}
					}
					if constexpr (P::worker) {
						// (A small range the guest wrote whose copy found no room in the slab or could not be read: a
						// hole.) A live buffer that holds the range, none of it CPU-modified, is what ObtainBuffer
						// returns with nothing to upload. For a formatted V# too: no image starts at it (asked above),
						// so the buffer is not made from an image, unless a guest readback is pending over the range,
						// which ObtainBuffer finishes first. Kept in the worker's own entry (under the special
						// generation where the image starts or the mappings bear on it); the buffer's last use is the
						// turn's stamp. Anything else is the GPU thread's.
						if (cpu_written && size <= BufferCache::CACHING_PAGESIZE) return false;
						if (formatted && buffer_cache.GuestReadbackOver(address, size)) return false;
					// (The epochs before the lookup and before the clean check.)
						const auto registration_before = buffer_cache.RegistrationEpoch();
						const auto epoch               = buffer_cache.CpuModificationEpoch(address, size);
						const auto [found, id]         = buffer_cache.FindLiveBuffer(address, size);
						if (found == nullptr || !range_clean(address, size)) return false;
						fresh.device       = found->BufferDeviceAddress() + found->Offset(address);
						fresh.dwords       = static_cast<uint32_t>(std::min<uint64_t>(size / 4, UINT32_MAX));
						fresh.address      = address;
						fresh.size         = size;
						fresh.cpu_epoch    = epoch;
						fresh.registration = registration_before;
						held               = fresh;
						WeaveParcel().result->touched.push_back(
						    {.buffer  = std::bit_cast<uint64_t>(id),
						     .handle  = WeaveHandleBits(static_cast<VkBuffer>(found->Handle())),
						     .address = address,
						     .size    = size});
						note_held(st, i, address, size, epoch);
						const auto [word, stride]      = word_of(words4);
						block[plan.BufferDword(i)]     = static_cast<uint32_t>(held.device);
						block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(held.device >> 32u);
						block[plan.BufferDword(i) + 2] = held.dwords;
						block[plan.BufferDword(i) + 3] = word;
						block[plan.BufferDword(i) + 4] = stride;
						continue;
					}
					if (cpu_written) buffer_cache.EnsureBufferContents(address, size);
					const auto registration_before = buffer_cache.RegistrationEpoch(); // (a speculation's: before its lookup)
					std::optional<BufferCache::AheadSite> ahead_site; // (the lookup's site)
					std::optional<NoRange>                no_range;
					if (spec_here) {
						ahead_site.emplace(1);
						no_range.emplace(reloc);
						Spec::t_pending_ok  = pending_ok;
						Spec::t_pending_hit = false;
					}
					const auto [owner, offset] = buffer_cache.ObtainBuffer(address, size, false, formatted);
					if (spec_here) Spec::t_pending_ok = false;
					region_epochs.fill({UINT64_MAX, 0});
					if (spec_here && owner == nullptr && Spec::t_pending_hit) {
						// (spec_pending_buffers: the null buffer, size 0, until the commit writes the range's; nothing
						// held, nothing noted: the commit obtains the range itself.)
						pending_here.push_back({plan.BufferDword(i), address, size, false, formatted});
						const auto [word, stride]      = word_of(words4);
						block[plan.BufferDword(i)]     = static_cast<uint32_t>(null_device);
						block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(null_device >> 32u);
						block[plan.BufferDword(i) + 2] = 0;
						block[plan.BufferDword(i) + 3] = word;
						block[plan.BufferDword(i) + 4] = stride;
						continue;
					}
					// (The epoch before the dirty check: a write after it moves the epoch.)
					fresh.cpu_epoch = buffer_cache.CpuModificationEpoch(address, size);
					if (owner == nullptr || buffer_cache.GetBufferIfLive(buffer_cache.FindBuffer(address, size)) != owner ||
					    buffer_cache.IsRegionCpuModified(address, size))
						return false;
					if (Spec::Current() != nullptr && buffer_cache.IsRegionCpuDirty(address, size)) fresh.cpu_epoch = 0;
					fresh.device       = owner->BufferDeviceAddress() + offset;
					fresh.dwords       = static_cast<uint32_t>(std::min<uint64_t>(size / 4, UINT32_MAX));
					fresh.address      = address;
					fresh.size         = size;
					// (The owner's: after ObtainBuffer's own registrations. A speculation's: before its lookup, which the
					// render thread's registrations may pass.)
					fresh.registration = Spec::Current() != nullptr ? registration_before : buffer_cache.RegistrationEpoch();
				}
				held = fresh;
			}
			// (A speculation's: the buffer of the range its device address is in.)
			// (spec_reloc_refs: an entry instead, proven at its held proof's epoch: the commit looks again only when a
			// buffer was registered or unregistered over its range since that, and writes the address it finds then.)
			if (auto* spec = Spec::Current(); spec != nullptr && held.size != 0) {
				if (reloc)
					pending_here.push_back({plan.BufferDword(i), held.address, held.size, false,
					                        program.info.buffers[i].formatted, false, held.registration});
				else
					spec->NoteBufferRange(held.address, held.size), spec->NoteUnrel(held.address, held.size);
			}
			if constexpr (P::worker)
				if (held.size != 0) note_held(st, i, held.address, held.size, held.cpu_epoch); // (its entry, as it is)
			const auto [word, stride]      = word_of(words4);
			block[plan.BufferDword(i)]     = static_cast<uint32_t>(held.device);
			block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(held.device >> 32u);
			block[plan.BufferDword(i) + 2] = held.dwords;
			block[plan.BufferDword(i) + 3] = word;
			block[plan.BufferDword(i) + 4] = stride;
		}
		if (plan.BlockDwords() != 0) { // (a stage with neither slots nor buffers reads no block)
			// (A speculation's thread never waits for its ring: it holds the catalog shared; the hole runs the draw.)
			const auto copied = Spec::Current() == nullptr
			                        ? std::optional<uint64_t> {ring.Copy(block.data(), block.size() * sizeof(uint32_t), 16)}
			                        : ring.TryCopy(block.data(), block.size() * sizeof(uint32_t), 16);
			if (!copied) return Spec::Refuse("table: upload ring busy"), false;
			blocks[st] = ring.BufferDeviceAddress() + *copied;
			if constexpr (P::worker) {
				// (Its held uses' five words each, as the block has them, for the turn's recopy of one the guest
				// wrote.)
				auto& result = *WeaveParcel().result;
				for (auto use = result.held_uses.rbegin();
				     use != result.held_uses.rend() && use->word == UINT32_MAX && use->stage == st; ++use) {
					use->word         = static_cast<uint32_t>(result.held_words.size());
					const auto* words = block.data() + plan.BufferDword(use->buffer);
					result.held_words.insert(result.held_words.end(), words, words + 5);
				}
				result.made.back().blocks[st] = blocks[st];
			}
			// (spec_pending_buffers: where the commit writes each one's buffer.)
			if (!pending_here.empty()) {
				auto& pending = Spec::Current()->pending_buffers;
				for (const auto& here: pending_here) {
					pending.push_back({.address = here.address,
					                   .size    = here.size,
					                   .ring    = &upload,
					                   .offset  = *copied + uint64_t {here.dword} * 4u,
					                   .written = here.written,
					                   .texel   = here.texel,
					                   .pending = here.pending,
					                   .since   = here.since});
					Spec::Current()->NoteTableRange(here.address, here.size);
				}
			}
		}
	}
	return true;
}

// The page table ranges the pair's flat and global loads read made current, as the normal path's PrepareBdaBindings
// does at every use (the read plans' words from the user data and the blocks' slots); an unbounded plan, all of them.
// (Without it, particle simulations at Stonefang read stale pages: readbacks the game crashed on, and indirect
// arguments that lost the device.)
void RenderExecutor::TablePrepareBda(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	const auto global = [&](uint32_t st) { return pair.programs[st] != nullptr && pair.programs[st]->table_plan.global_memory; };
	if (!global(0) && !global(1)) return;
	auto&                                               table = OwnTable();
	thread_local ShaderRecompiler::IR::ResourceSnapshot snapshot;
	thread_local std::vector<GuestRange>                ranges, stage_ranges;
	ranges.clear();
	bool bounded = true;
	for (uint32_t st = 0; st < 2 && bounded; ++st) {
		if (!global(st)) continue;
		const auto& program = *pair.programs[st];
		snapshot.user_data.assign(user_data[st].begin(), user_data[st].end());
		snapshot.flattened_srt.assign(table.blocks[st].begin(), table.blocks[st].begin() + program.table_plan.slots.size());
		bounded = ShaderRecompiler::IR::EvaluateBdaReadPlan(program.bda_read_plan, snapshot, stage_ranges);
		ranges.insert(ranges.end(), stage_ranges.begin(), stage_ranges.end());
	}
	auto& resources = m_context.GetGpuResources();
	if (!bounded || !resources.PrepareBdaReadRanges(ranges)) resources.PrepareBda();
}

vk::AccessFlags2 RenderExecutor::TableTargetRead(const TableImageSet& set, ImageId id) {
	vk::AccessFlags2 read {};
	for (const auto& dep: set.images)
		if (dep.target && dep.id == id) read |= dep.target_read;
	return read;
}

// KYTY_TABLE_SERIALS=2: a check the serial sums would have skipped that changed something (the first 32 printed).
static void TableSerialsUnsafe(const char* what) {
	static std::atomic<uint32_t> count {0};
	LiveCounters::Add(LiveCounters::SerialsUnsafe);
	if (const auto n = count.fetch_add(1, std::memory_order_relaxed); n < 32)
		std::printf("TABLE-SERIALS UNSAFE: %s (%u)\n", what, n + 1);
}

// The image serial sums (ImageSerials) of a draw state's targets and of a set's images, with the unattributed serial.
template<typename State> // (RenderExecutor::NativeXprDrawState)
static uint64_t TableTargetSerials(const State& draw_state) {
	uint64_t sum = ImageSerials::Unattributed();
	for (const auto id: draw_state.target_ids) sum += ImageSerials::Read(id.index);
	return sum;
}
template<typename Set> // (RenderExecutor::TableImageSet)
static uint64_t TableSetSerials(const Set& set) {
	uint64_t sum = ImageSerials::Unattributed();
	for (const auto id: set.image_ids) sum += ImageSerials::Read(id.index);
	return sum;
}
// A set transited under another draw state is in place under this one when it holds no target of either: TableTransit
// asks the state only which of the set's images it targets.
template<typename Set, typename State>
static bool TableSetApart(const Set& set, const State& draw_state) {
	if (set.holds_target) return false;
	for (const auto id: set.image_ids)
		if (std::ranges::find(draw_state.target_ids, id) != draw_state.target_ids.end()) return false;
	return true;
}
// (Mode 4) Whether TableTransit's target tests refuse the set under a draw state: an image the set holds as a target the
// draw does not target, the reverse, or another attachment layout or access. (Those tests only: TableTransit may refuse
// for another cause too.) A set's deps are laid out for the draw state it was resolved under (TableResolveSet: a target
// in its attachment layout and access, any other image as a texture or storage), and its descriptors carry those
// layouts; the pair's sets are keyed by the words alone.
template<typename Set, typename State> // (RenderExecutor::TableImageSet, RenderExecutor::NativeXprDrawState)
static bool TableTargetsDiffer(const Set& set, const State& draw_state) {
	for (const auto& dep: set.images) {
		const auto target    = std::ranges::find(draw_state.targets, dep.id, &State::Target::id);
		const bool is_target = target != draw_state.targets.end();
		if (dep.target != is_target ||
		    (is_target && (target->layout != dep.layout || (target->access | dep.target_read) != dep.access)))
			return true;
	}
	return false;
}
// (Mode 4) The draw state's targets (each one's image, attachment layout and access: what TableResolveSet lays a target
// dep out from), the words' tag for the set resolved under them. All of them, not only those among the set's image_ids:
// a set's pending slots are not there until its commit completes them.
inline constexpr uint32_t TableTargetsTag = 0x7a4e5170u;
template<typename State>
static uint64_t TableTargetsKey(const State& draw_state) {
	uint64_t key = TableTargetsTag;
	for (const auto& target: draw_state.targets) {
		const std::array<uint64_t, 3> part {uint64_t {target.id.index} | uint64_t {target.id.generation} << 32u,
		                                    static_cast<uint64_t>(target.layout),
		                                    static_cast<uint64_t>(static_cast<VkAccessFlags2>(target.access))};
		key = XXH3_64bits_withSeed(part.data(), sizeof(part), key);
	}
	return key;
}

// A target of the draw: NativeXprEmitTargets put it in its attachment layout, which the set must have been made for
// (the access the set holds it with: the target's, with its shaders' sampling access added); an image the set holds as
// a texture must not be one. `is_target`: `dep` is one of the draw's targets.
template <typename Dep, typename State>
[[nodiscard]] static bool TableTargetFits(const Dep& dep, const State* draw_state, bool& is_target) {
	const auto target = draw_state == nullptr ? decltype(draw_state->targets.end()) {}
	                                          : std::ranges::find(draw_state->targets, dep.id, &State::Target::id);
	is_target = draw_state != nullptr && target != draw_state->targets.end();
	return is_target == dep.target &&
	       (!is_target || (target->layout == dep.layout && (target->access | dep.target_read) == dep.access));
}

// A set's images in their layouts (other work may have moved them since the set was last bound), a DCC image's
// pending fast clear applied first, as the normal path's FindTexture does (a speculation's refuses: its packet is a
// hole), unless the draw's targets are not those the set was made for (an image held as a texture became a target,
// or the reverse).
bool RenderExecutor::TableTransit(const TableImageSet& set, vk::CommandBuffer vk_buffer,
                                  const NativeXprDrawState* draw_state) {
	auto& textures = m_context.GetTextureCache();
	// spec_copy_image_refuse: a speculation's set reused without a lookup (a catalog set, the role's last) holding an image
	// whose bytes its open segment writes, as the read-only lookups refuse one (TextureCache): looked at only when the
	// open segment writes anything at all.
	const auto* const open_writer = ::kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0 &&
	                                        Spec::CopyImageRefuseOn()
	                                    ? Spec::Current()
	                                    : nullptr;
	const bool open_writes = open_writer != nullptr && open_writer->OpenSegmentWritesAny();
	const auto reused_written = [] {
		LiveCounters::Add(LiveCounters::SpecImageOpenWritesReusedN);
		return Spec::Refuse("table: written in its segment"), false;
	};
	// Its own set's pending slots too (spec_worker_images: completed in its commit's Before from the bytes before these
	// writes), which set.images does not hold. (Only the translation that proposed a set binds it before it is published,
	// and its commit completes the slots after the translation: its pending_images is this thread's to read.)
	if (const auto* spec = Spec::On() ? Spec::Current() : nullptr; spec != nullptr && set.proposer == spec->id &&
	                                                                 !set.pending_images.empty()) {
		if (open_writes)
			for (const auto& pending: set.pending_images)
				if (open_writer->OpenSegmentWritesImage(pending.desc.info)) return reused_written();
		// spec_set_cpu_dirty_stop: nor bound in a later segment than the one that proposed it. Its slots' images are
		// completed in that one's Before and are no image entry of the later segment, whose commit's CPU-dirty and
		// buffer-write stops never see them (a hole between the two writing them): refused. (TableSet resolves such a
		// set's words afresh before it gets here: this is the guard.)
		if (set.proposed_segment != spec->closed.size() && Spec::SetCpuDirtyStopOn()) {
			LiveCounters::Add(LiveCounters::SpecPendingReuseRefusedN);
			return Spec::Refuse("table: pending set of an earlier segment"), false;
		}
	}
	for (const auto& dep: set.images) {
		auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr) return false;
		// A speculation's set (a catalog set reused without a lookup) naming an image released since (an area travel's
		// unmap: DeleteImage -> RetireImage; erased once the packets of its release passed, which a packet that began
		// after it does not hold back): refused before anything reads it further. One still registered here is held
		// by this packet (Spec::PacketNow at a later release sees it). A view past the image's levels or layers too
		// (Image::GetBarriers would walk states it does not have). Both on a speculation's thread only: the GPU
		// thread's sets are proven against registrations before they get here (TableValidateSet).
		// (One line each of the first 16, then every 256th: the address the set's view names.)
		const auto say = [&](std::atomic<uint64_t>& count, const char* what) {
			if (const auto n = count.fetch_add(1, std::memory_order_relaxed) + 1; n <= 16 || n % 256 == 0) {
				std::printf("Table: %s (addr=0x%llx serial=%llu): %llu times\n", what,
				            static_cast<unsigned long long>(dep.desc.info.data.address),
				            static_cast<unsigned long long>(dep.serial), static_cast<unsigned long long>(n));
				std::fflush(stdout);
			}
		};
		if (Spec::Current() != nullptr && (!image->registered || image->serial != dep.serial)) {
			static std::atomic<uint64_t> released {0};
			say(released, "a speculation's set named an image released since, refused");
			return Spec::Refuse("table: image released"), false;
		}
		if (Spec::Current() != nullptr && !dep.whole &&
		    (dep.range.base_level + dep.range.level_count > image->info.resources.levels ||
		     (!image->info.IsVolume() && // (a volume's slice views: GetBarriers takes its one layer)
		      dep.range.base_layer + dep.range.layer_count > image->info.resources.layers))) {
			static std::atomic<uint64_t> past {0};
			say(past, "a speculation's set viewed past its image's levels or layers, refused");
			return Spec::Refuse("table: view past its image"), false;
		}
		bool is_target = false;
		if (!TableTargetFits(dep, draw_state, is_target)) return false;
		if (is_target) continue;
		if (open_writes && (open_writer->OpenSegmentWritesImage(image->info) || open_writer->OpenSegmentWritesImage(dep.desc.info)))
			return reused_written();
		// An image the shader writes is GPU-modified, CPU-clean and current with buffer writes (as the normal path's
		// FindTexture leaves a storage image: it acquires it again otherwise). One written since the set was proven this
		// frame (by the CPU, a GPU buffer write, or into its stencil plane) is uploaded by the normal path's FindTexture.
		// (A speculation's entry for it first: the state it notes as found is the one the test saw, or a write after,
		// which its commit then stops at: spec_set_cpu_dirty_stop. A registered image only: TableEarlyEntry.)
		TableEarlyEntry(*image);
		if ((dep.storage && !image->SafeToDownload()) || TableRefreshPending(*image, dep.desc)) return false;
		if (dep.desc.info.metadata.kind == ImageMetadataKind::Dcc) {
			if (image->info.IsVolume()) return false;
			const auto address = dep.desc.info.metadata.range.address;
			for (uint32_t layer = dep.range.base_layer; layer < dep.range.base_layer + dep.range.layer_count; ++layer) {
				if (layer >= 32) return false;
				if (!textures.IsMetaCleared(address, layer)) continue;
				// The normal path's FindTexture clears its layers first. A speculation's packet is a hole that does:
				// the work after it finds them cleared (speculation-state.h).
				if (auto* spec = Spec::Current()) {
					// Not where the normal path's FindTexture leaves every clear of the image pending, whatever its metadata
					// holds (TextureCache::DccClearLeftPending: 94% of these holes at 5-1, an image of more than one level):
					// it binds the image uncleared, as here, and so does the commit while the image's validity has not moved
					// (Spec::State::left_clears). As the normal path's, no other layer of it is asked.
					if (uint64_t serial = 0, validity = 0; textures.DccClearLeftPending(dep.id, dep.desc, serial, validity)) {
						spec->left_clears.push_back({image, serial, validity, spec->closed.size()});
						break;
					}
					// (KYTY_SPEC_PREDICT_DCC: not where the normal path's FindTexture was seen to leave the clear pending,
					// TextureCache::DccClearSticks: the work after the hole finds it cleared still, and a "not cleared" fact
					// would fail at its commit. Its later uses read the cache, and are holes too.)
					if (!textures.DccClearSticks(address))
						for (uint32_t at = layer; at < std::min(dep.range.base_layer + dep.range.layer_count, 32u); ++at)
							if (textures.IsMetaCleared(address, at)) spec->ConsumeMeta(address, at);
					return Spec::Refuse("table: cleared texture"), false;
				}
				// The main executor's: FindTexture, as the normal path's use (it clears what it can, into this command
				// buffer, before the transitions below; what it leaves pending, the normal path binds as well).
				if (Role() != 0 || !image->registered || image->depth_id || image->binding.needs_rebind) return false;
				(void)textures.FindTexture(dep.id, dep.desc);
				// (DccClearSticks; KYTY_SPECULATE unset or 0 keeps nothing, as the texture cache says.)
				if (Spec::On()) textures.NoteDccClear(address, textures.IsMetaCleared(address, layer));
				image = textures.m_slot_images.try_get(dep.id);
				if (image == nullptr || image->serial != dep.serial ||
				    m_context.GetCommandScheduler().Current().RawHandle() != vk_buffer)
					return false;
				break;
			}
		}
		// A depth image with an HTile clear pending: the normal path's FindTexture clears it first
		// (KYTY_SAMPLE_PENDING_HTILE_CLEAR); the draw or dispatch is left to it.
		if (image->info.IsDepth() && TextureCache::SampleHtileClearOn()) {
			uint64_t meta    = 0;
			bool     consume = false;
			if (textures.SampleHtileClearPending(*image, dep.desc, &meta, &consume)) {
				// (A speculation's: a hole that consumes the clear where the normal path does.)
				if (auto* spec = Spec::Current()) {
					if (consume) spec->ConsumeMeta(meta, 0);
					return Spec::Refuse("table: depth texture with an HTile clear pending"), false;
				}
				return false;
			}
		}
	}
	// As CommitBindings': the transitions all precede the one draw or dispatch. A draw's image bound both as a texture
	// and as storage that the shader writes: ordered after the last draw's write with KYTY_DRAW_READ_AFTER_WRITE=1
	// (a dispatch: by the game's barriers).
	const TransitGroup transit_group;
	for (const auto& dep: set.images)
		if (!dep.target)
			textures.m_slot_images[dep.id].Transit(dep.layout, dep.access, dep.whole ? std::nullopt : std::optional {dep.range},
			                                       vk_buffer, draw_state != nullptr && dep.whole && dep.storage);
		else if (dep.target_read) // (as CommitBindings transits a target the shaders sample: the whole image)
			textures.m_slot_images[dep.id].Transit(dep.layout, dep.access, std::nullopt, vk_buffer);
	return true;
}

// (Read only, off the GPU thread, which transits images as it records: each image's barrier state is read under its
// state lock, and its CPU-written state by its flags alone, game threads changing its dirty ranges. A DCC image's
// pending clears, a depth image with HTile metadata (a pending HTile clear, KYTY_SAMPLE_PENDING_HTILE_CLEAR), a target
// the shaders sample, an image the shaders write and one in states of its own are left to TableTransit.)
bool RenderExecutor::TableTransitPlaced(const TableImageSet& set, const NativeXprDrawState* draw_state) const {
	auto&          textures = m_context.GetTextureCache();
	constexpr auto writes   = vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderWrite |
	                        vk::AccessFlagBits2::eMemoryWrite;
	for (const auto& dep: set.images) {
		const auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr || image->serial != dep.serial) return false;
		bool is_target = false;
		if (!TableTargetFits(dep, draw_state, is_target)) return false;
		if (is_target) {
			if (dep.target_read) return false;
			continue;
		}
		if (dep.storage || dep.desc.info.metadata.kind == ImageMetadataKind::Dcc || image->IsBufferModified() ||
		    image->IsStencilModified() || image->IsMaybeCpuDirty() || image->IsDefinitelyCpuDirty())
			return false;
		if (image->info.IsDepth() && (dep.desc.info.metadata.kind == ImageMetadataKind::Htile ||
		                              image->info.metadata.kind == ImageMetadataKind::Htile))
			return false;
		const Image::StateLock lock(*image);
		const auto&            state = image->backing.state;
		if (!image->backing.subresource_states.empty() || state.layout != dep.layout ||
		    state.access_mask != dep.access || static_cast<bool>(dep.access & writes))
			return false;
	}
	return true;
}

// KYTY_SCOUT (scout.h): the draw pair's view for the scout, when its stages take TableEvaluate's and TableGather's flat
// paths with no indirect image; none otherwise.
template <typename Pair>
static void TableScoutPublish(Pair& pair) {
	if (!Scout::Running() || pair.programs[0] == nullptr) return;
	auto view    = std::make_unique<Scout::View>();
	view->key    = {pair.vs, pair.ps, static_cast<uint32_t>(pair.start_instance)};
	view->stages = pair.programs[1] != nullptr ? 2u : 1u;
	for (uint32_t st = 0; st < view->stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		const auto& form    = pair.packed[st];
		if (!form.valid || form.runs.size() != pair.runs[st].size() || form.user_words > 32 || !plan.indirect.empty() ||
		    form.buffer_words.size() != 4 * plan.buffers.size())
			return;
		auto& stage = view->stage[st];
		for (const auto& run: form.runs) {
			stage.runs.push_back({run.address, run.records, run.immediate, run.first, run.count});
			stage.slot_words += run.count;
		}
		if (stage.slot_words > 0xffff) return;
		// Block order: the runs by first slot, side by side ones one segment.
		std::vector<uint16_t> order(stage.runs.size());
		for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<uint16_t>(i);
		std::ranges::sort(order, {}, [&](uint16_t i) { return stage.runs[i].first; });
		uint16_t at = 0;
		for (const auto i: order) {
			auto& run = stage.runs[i];
			run.at    = at;
			at        = static_cast<uint16_t>(at + run.count);
			if (!stage.segments.empty() && stage.segments.back().first + stage.segments.back().count == run.first)
				stage.segments.back().count = static_cast<uint16_t>(stage.segments.back().count + run.count);
			else
				stage.segments.push_back({run.first, run.count});
		}
		stage.level_ends.assign(form.level_ends.begin(), form.level_ends.end());
		stage.image_words.assign(form.image_words.begin(), form.image_words.end());
		stage.buffer_words.assign(form.buffer_words.begin(), form.buffer_words.end());
		stage.image_counts.assign(form.image_counts.begin(), form.image_counts.end());
		stage.constants.assign(form.constants.begin(), form.constants.end());
		stage.user_words      = form.user_words;
		stage.block_dwords    = static_cast<uint32_t>(plan.BlockDwords());
		stage.lod_stats_count = static_cast<uint32_t>(program.bindings.lod_stats_count);
		stage.buffers         = static_cast<uint32_t>(plan.buffers.size());
	}
	view->version = pair.scout_version = Scout::NextVersion();
	Scout::Publish(view.release(), TableScoutHash(pair));
}

// The scout's ticket for the draw TableDraw makes (Scout::Current), when it holds: made for this packet from this
// pair's view, with the user data the render thread has, under the mapping epoch of now, and no page of its tables with
// read watchers now (GPU work recorded after the scout read them writes it). Null otherwise, counted by reason.
template <typename Pair>
static const Scout::Ticket* TableScoutTicket(const Pair& pair, std::array<std::span<const uint32_t>, 2> user_data,
                                             uint32_t stages, GpuResourceManager& resources) {
	LiveCounters::Add(LiveCounters::ScoutChecks);
	if (pair.scout_version == 0) return LiveCounters::Add(LiveCounters::ScoutLostNoView), nullptr;
	const auto* ticket = Scout::Current();
	if (ticket == nullptr) {
		LiveCounters::Add(LiveCounters::ScoutLostMissing);
		static_assert(LiveCounters::ScoutMissFenced - LiveCounters::ScoutMissUnwalked == static_cast<uint32_t>(Scout::Reason::Fenced)); // (the miss counters follow Reason)
		LiveCounters::Add(static_cast<LiveCounters::Id>(LiveCounters::ScoutMissUnwalked + static_cast<uint32_t>(Scout::CurrentReason())));
		return nullptr;
	}
	if (ticket->packet != Scout::CurrentPacket()) return LiveCounters::Add(LiveCounters::ScoutLostIdentity), nullptr;
	if (ticket->version != pair.scout_version || ticket->stages != stages)
		return LiveCounters::Add(LiveCounters::ScoutLostPair), nullptr;
	for (uint32_t st = 0; st < stages; ++st)
		if (ticket->user_words[st] > user_data[st].size() ||
		    std::memcmp(ticket->user_data[st].data(), user_data[st].data(), ticket->user_words[st] * sizeof(uint32_t)) != 0)
			return LiveCounters::Add(LiveCounters::ScoutLostRegisters), nullptr;
	if (ticket->mapping_epoch != resources.MappingEpoch()) return LiveCounters::Add(LiveCounters::ScoutLostEpoch), nullptr;
	if (bool write_back = false; Scout::WrittenSince(*ticket, write_back)) {
		if (write_back) LiveCounters::Add(LiveCounters::ScoutLostWriteBack);
		return LiveCounters::Add(LiveCounters::ScoutLostWritten), nullptr;
	}
	// Each page once (in order), its granule's bit once a granule.
	const auto* pages    = Scout::Pages(*ticket);
	const auto* granules = resources.ReadGranules();
	uint64_t    clear    = UINT64_MAX; // a granule whose bit is clear
	for (uint32_t i = 0; i < ticket->pages; ++i) {
		if ((pages[2 * i + 1] & Scout::Ticket::PageBacking) != 0) continue; // (its ranges are proven below)
		const auto page    = Scout::PageAt(pages, i);
		const auto granule = page >> (PageManager::READ_GRANULE_BITS - 12u);
		if (granule == clear) continue;
		if ((page << 12u) < TRACKER_ADDRESS_SIZE &&
		    ((granules[granule / 64u].load(std::memory_order_acquire) >> (granule % 64u)) & 1u) == 0) {
			clear = granule;
			continue;
		}
		if (resources.HasReadWatchers(page << 12u, 4096)) return LiveCounters::Add(LiveCounters::ScoutLostWatched), nullptr;
	}
	// Each range the scout read from the backing, as this thread's own read takes it (clean bytes
	// from the backing; otherwise the GPU's, read back) and the same as the ticket's.
	const auto* ranges = pages + 2u * ticket->pages;
	for (uint32_t i = 0; i < ticket->backing; ++i) {
		const auto               address = ranges[3 * i] | uint64_t {ranges[3 * i + 1]} << 32u;
		const auto*              held    = ticket->data + (ranges[3 * i + 2] & 0xffffu);
		const uint32_t           count   = ranges[3 * i + 2] >> 16u;
		std::array<uint32_t, 64> fresh;
		for (uint32_t done = 0; done < count; done += static_cast<uint32_t>(fresh.size())) {
			const auto chunk = std::min<uint32_t>(count - done, static_cast<uint32_t>(fresh.size()));
			if (!Libs::LibKernel::Memory::TryReadGpuCleanBackingToHost(address + 4ull * done, fresh.data(), chunk * 4ull))
				return LiveCounters::Add(LiveCounters::ScoutLostDirty), nullptr;
			if (std::memcmp(fresh.data(), held + done, chunk * 4ull) != 0)
				return LiveCounters::Add(LiveCounters::ScoutLostChanged), nullptr;
		}
	}
	LiveCounters::Add(LiveCounters::ScoutUsed);
	if (ticket->backing != 0) LiveCounters::Add(LiveCounters::ScoutUsedBacking);
	return ticket;
}

// Words copied without a library call: a ticket's segments are a few words each, and a memcpy call a run cost more
// than its copy.
static inline void TableCopyWords(uint32_t* to, const uint32_t* from, uint32_t count) {
	for (; count >= 8; count -= 8, to += 8, from += 8) __builtin_memcpy_inline(to, from, 32);
	if ((count & 4u) != 0) {
		__builtin_memcpy_inline(to, from, 16);
		to += 4;
		from += 4;
	}
	if ((count & 2u) != 0) {
		__builtin_memcpy_inline(to, from, 8);
		to += 2;
		from += 2;
	}
	if ((count & 1u) != 0) *to = *from;
}

// A ticket's slot words of one stage (`in`) into its block `slots`, by segment (`segments`, `count` of them).
static void TableScoutSlots(uint32_t* slots, const uint32_t* in, const uint32_t* segments, uint32_t count) {
	for (uint32_t i = 0; i < count; ++i) {
		const uint32_t words = segments[i] >> 16u;
		TableCopyWords(slots + (segments[i] & 0xffffu), in, words);
		in += words;
	}
}

// A ticket's evaluation where TableEvaluate and TableGather leave theirs: each stage's runs' slots in its block (the
// other slots as they were), its T#/S# and LOD statistics words, and its V#s with their entries.
template <typename Table, typename Pair>
static void TableScoutApply(Table& table, const Pair& pair, const Scout::Ticket& ticket, uint32_t stages) {
	const uint32_t* in       = ticket.data;
	const uint32_t* segments = ticket.data + ticket.slot_words[0] + ticket.slot_words[1];
	for (uint32_t st = 0; st < stages; ++st) {
		auto& slots = table.blocks[st];
		slots.resize(pair.programs[st]->table_plan.BlockDwords());
		TableScoutSlots(slots.data(), in, segments, ticket.segments[st]);
		in += ticket.slot_words[st];
		segments += ticket.segments[st];
		table.indirect[st].clear();
		if (!pair.programs[st]->table_plan.indirect.empty()) table.indirect_counts[st].clear(); // (read for such a stage only)
	}
	in = segments;
	table.words.assign(in, in + ticket.image_words);
	in += ticket.image_words;
	table.lod_stats.assign(in, in + ticket.lod_stats);
	in += ticket.lod_stats;
	for (uint32_t st = 0; st < stages; ++st) {
		auto& entries = table.entries[st];
		entries.resize(ticket.buffers[st]);
		for (auto& [words4, entry]: entries) {
			std::copy_n(in, 4, words4.begin());
			entry = &table.resolved[in[4]];
			__builtin_prefetch(entry);
			in += 5;
		}
	}
	table.gathered = &pair;
}

// KYTY_SCOUT=2: the ticket against the render thread's own evaluation (`evaluated`: TableEvaluate's result, then
// TableGather's), field by field; a difference is counted and the first ones reported, the first slot that differs with
// both values.
template <typename Table, typename Pair>
static void TableScoutVerify(const Table& table, const Pair& pair, const Scout::Ticket& ticket, uint32_t stages,
                             bool evaluated) {
	// slots, T#/S# words, LOD statistics words, V#s and entries, and the apply (its slots right, the block it copies
	// them into wrong)
	std::array<bool, 5> differ {};
	char                detail[160] = "";
	if (evaluated) {
		const uint32_t* in       = ticket.data;
		const uint32_t* segments = ticket.data + ticket.slot_words[0] + ticket.slot_words[1];
		thread_local std::vector<uint32_t> scratch;
		for (uint32_t st = 0; st < stages; ++st) {
			const auto& runs   = pair.packed[st].runs;
			const bool  before = differ[0];
			for (size_t r = 0; r < runs.size(); ++r) {
				const auto& run  = runs[r];
				const auto* mine = table.blocks[st].data() + run.first;
				// The run's words, in block order: its segment's, then its own.
				uint32_t at = 0;
				for (uint32_t k = 0; k < ticket.segments[st]; ++k) {
					const uint32_t first = segments[k] & 0xffffu, count = segments[k] >> 16u;
					if (run.first >= first && run.first < first + count) {
						at += run.first - first;
						break;
					}
					at += count;
				}
				for (uint32_t w = 0; w < run.count && !differ[0]; ++w) {
					if (mine[w] == in[at + w]) continue;
					differ[0] = true;
					std::snprintf(detail, sizeof(detail), " (stage %u run %zu/%zu slot %u: ticket 0x%08x, render 0x%08x)", st, r,
					              runs.size(), run.first + w, in[at + w], mine[w]);
				}
			}
			// The ticket applied as TableScoutApply does, into a copy of the render thread's block.
			scratch.assign(table.blocks[st].begin(), table.blocks[st].end());
			TableScoutSlots(scratch.data(), in, segments, ticket.segments[st]);
			if (differ[0] == before && scratch != table.blocks[st]) {
				differ[4] = true;
				std::snprintf(detail, sizeof(detail), " (stage %u: the apply's block differs)", st);
			}
			LiveCounters::Add(LiveCounters::ScoutApplyRuns, runs.size());
			LiveCounters::Add(LiveCounters::ScoutApplySegments, ticket.segments[st]);
			segments += ticket.segments[st];
			in += ticket.slot_words[st];
		}
		in = segments;
		differ[1] = table.words.size() != ticket.image_words || !std::equal(table.words.begin(), table.words.end(), in);
		in += ticket.image_words;
		differ[2] = table.lod_stats.size() != ticket.lod_stats || !std::equal(table.lod_stats.begin(), table.lod_stats.end(), in);
		in += ticket.lod_stats;
		for (uint32_t st = 0; st < stages; ++st) {
			if (table.entries[st].size() != ticket.buffers[st]) {
				differ[3] = true;
				break;
			}
			for (const auto& [words4, entry]: table.entries[st]) {
				differ[3] = differ[3] || !std::equal(words4.begin(), words4.end(), in) || entry != &table.resolved[in[4]];
				in += 5;
			}
		}
		if (differ == std::array<bool, 5> {}) return;
	}
	LiveCounters::Add(LiveCounters::ScoutMismatches);
	if (!evaluated) LiveCounters::Add(LiveCounters::ScoutMismatchRefused);
	if (differ[0]) LiveCounters::Add(LiveCounters::ScoutMismatchSlots);
	if (differ[1]) LiveCounters::Add(LiveCounters::ScoutMismatchWords);
	if (differ[2]) LiveCounters::Add(LiveCounters::ScoutMismatchLod);
	if (differ[3]) LiveCounters::Add(LiveCounters::ScoutMismatchEntries);
	if (differ[4]) LiveCounters::Add(LiveCounters::ScoutMismatchApply);
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
		std::printf("Scout: mismatch VS 0x%llx PS 0x%llx submission %llu draw %u:%s%s%s%s%s%s%s\n",
		            static_cast<unsigned long long>(pair.vs), static_cast<unsigned long long>(pair.ps),
		            static_cast<unsigned long long>(ticket.sequence), ticket.ordinal, evaluated ? "" : " refused",
		            differ[0] ? " slots" : "", differ[1] ? " words" : "", differ[2] ? " lod" : "", differ[3] ? " entries" : "",
		            differ[4] ? " apply" : "", detail);
}

// A speculation worker's table index-buffer bind and indirect draws under KYTY_SPECULATE=4 (Spec::State::reloc_on)
// recorded as packets that carry the buffer handle and offset in a cell of their own (the packet's copy in the passive
// producer's chunk), which the commit may rewrite (Speculation::RelocPatch: a registration that joined or replaced the
// buffer since, the range's live buffer written in before the segment is spliced; nothing reads a passive producer's
// chunk before its splice, and a discard reads none of it). Only over a game buffer; else (a ring copy, no worker
// packets) the call is recorded as before.
#ifdef KYTY_LOCAL_VULKAN_RECORDING
namespace TableReloc {
struct Packet {
	VkCommandBuffer command = VK_NULL_HANDLE;
	Spec::RelocCell cell;                 // (the copy's: RelocRef::cell points here)
	uint32_t        op = 0, a = 0, b = 0; // 0 bind index (a: type); 1 draw indirect, 2 draw indexed indirect (a: count, b: stride)
};
static_assert(std::is_standard_layout_v<Packet> && std::is_trivially_copyable_v<Packet>);
static void Replay(std::span<const LocalVulkanRecording::Segment> segments, const vk::detail::DispatchLoaderDynamic& dispatch) {
	Packet packet;
	std::memcpy(&packet, segments[0].data, sizeof(packet));
	const auto command = LocalVulkanRecording::Resolve(packet.command);
	const auto handle  = reinterpret_cast<VkBuffer>(packet.cell.buffer); // NOLINT (a 64-bit non-dispatchable handle)
	if (packet.op == 0) dispatch.vkCmdBindIndexBuffer(command, handle, packet.cell.offset, static_cast<VkIndexType>(packet.a));
	else if (packet.op == 1) dispatch.vkCmdDrawIndirect(command, handle, packet.cell.offset, packet.a, packet.b);
	else dispatch.vkCmdDrawIndexedIndirect(command, handle, packet.cell.offset, packet.a, packet.b);
}
} // namespace TableReloc
#endif
// The call recorded as such a packet (true), with its reference noted in the speculation's state; false: record it plainly.
// (A plain one over a game buffer, not a ring's copy (no guest address), is one more reference the commit cannot
// relocate: noted as such, so the segment stops as before when a registration moved over its range.)
// (No check that `used` is still the range's registered buffer: a registration over the range after the speculation began
// moves it past the segment's base, which the commit's Changed() finds, whether it came after the draw's lookups or during
// them (a worker's draw is kept over an epoch that moved during its lookups: TableDraw): the cell is then rewritten with
// the live buffer (RelocEligible / RelocPatch) or the segment stops.)
// The caller calls it only with relocation on (a speculation's state with reloc_on).
static bool TableRelocRecord(vk::CommandBuffer vk_buffer, Spec::State& spec, uint8_t kind,
                             const std::pair<Buffer*, uint64_t>& used, uint64_t address, uint64_t size, uint32_t op,
                             uint32_t a, uint32_t b) {
	if (used.first == nullptr || size == 0 || used.first->CpuAddress() == 0) return false; // (a ring's copy: as before)
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	if (LocalVulkanRecording::Passive()) {
		TableReloc::Packet packet {static_cast<VkCommandBuffer>(vk_buffer), {}, op, a, b};
		packet.cell.buffer = reinterpret_cast<uint64_t>(static_cast<VkBuffer>(used.first->Handle())); // NOLINT
		packet.cell.offset = used.second;
		// (The cell is the packet's copy in the chunk: it stays there, unread, until the segment is spliced (replayed
		// after) or discarded. The commit writes it on the GPU thread before the splice, the worker never after.)
		const LocalVulkanRecording::Segment segment {&packet, sizeof(packet)};
		void*                               copy = nullptr;
		if (LocalVulkanRecording::EnqueuePacketCopy(TableReloc::Replay, segment, &copy) && copy != nullptr) {
			auto* cell = reinterpret_cast<Spec::RelocCell*>(static_cast<std::byte*>(copy) + offsetof(TableReloc::Packet, cell));
			spec.reloc_refs.push_back({address, address + size, kind, cell});
			return true;
		}
	}
#else
	(void)vk_buffer, (void)kind, (void)op, (void)a, (void)b;
#endif
	spec.NoteUnrel(address, size);
	return false;
}

// (Mode 4) A worker's table dispatch whose lookups saw the registration epoch move (the GPU thread's registrations,
// anywhere: a speculation's lookups register nothing) is kept when no registration since reached the tracker regions of a
// buffer range its open segment used (a superset of this dispatch's: its notes merge into them), and refused as before
// otherwise. The commit's Changed() asks the same of those ranges since the speculation began. (A worker's table draw is
// kept without this test, its commit deciding: TableDraw.)
static bool TableWorkerKeepsMoved(const Spec::State& spec, const BufferCache& cache, uint64_t epoch) {
	if (kyty_local_speculate_mode.load(std::memory_order_relaxed) != 4) return false;
	for (size_t i = spec.Sealed().buffer_ranges; i < spec.buffer_ranges.size(); ++i) {
		const auto& range = spec.buffer_ranges[i];
		if (cache.RegisteredSince(range.begin, range.end - range.begin, epoch)) return false;
	}
	return true;
}

// (Mode 4, TableDraw's set: out of line, so the other modes' TableDraw carries none of it.) A set resolved under another
// draw state, whose deps TableTransit's target tests refuse for this one's targets, is not the draw's: the words tagged with
// the draw state's targets (TableTargetsKey, three words more than any words TableEvaluate makes, so no untagged lookup
// compares equal) find, or resolve, the set made for them, in a slot of its own (TableSet: the same lookups, proofs and
// publication, the tagged words the set's key); the untagged words back right after. The set found is tested again by
// TableTransit as any. The memo's set is not offered to the second call: a first call that took the role's last set may
// have run the collection, which can have destroyed it.
RenderExecutor::TableImageSet* RenderExecutor::TableDrawStateSet(CommandBuffer& buffer, TablePair& pair, TableVariant& variant,
                                                                 const NativeXprDrawState& draw_state, TableImageSet* set,
                                                                 Spec::State* spec_state) {
	auto& table = OwnTable();
	if (TableTargetsDiffer(*set, draw_state)) {
		const auto key  = TableTargetsKey(draw_state);
		const auto size = table.words.size();
		table.memo_set    = nullptr;
		table.memo_verify = false;
		table.words.push_back(TableTargetsTag);
		table.words.push_back(static_cast<uint32_t>(key));
		table.words.push_back(static_cast<uint32_t>(key >> 32u));
		set = TableSet(buffer, pair, &variant, &draw_state);
		// The tagged slot's set made for these targets and still not laid out for them (a worker's, published before its
		// commit's completion refused such a slot, below): the owner retires it, as its own path retires a set found
		// invalid, and resolves the slot's set afresh under this draw state. A speculation leaves it to the owner (its draw
		// a hole). (A worker's only: the owner's own build is laid out for the draw state it ran under, and one that
		// differed would be retired at every draw.)
		if (set != nullptr && spec_state == nullptr && Role() == 0 && set->proposer != 0 && TableTargetsDiffer(*set, draw_state)) {
			bool retired = false;
			{
				const auto write = CatalogWrite();
				const auto found = pair.sets.find(XXH3_64bits(set->words.data(), set->words.size() * sizeof(uint32_t)));
				if (found != pair.sets.end() && found->second.get() == set) {
					if (table.last_set == set) table.last_set = nullptr;
					pair.ClearSets();
					++table.epoch;
					NativeXprRetireSet(set->pool, set->set);
					pair.sets.erase(found);
					retired = true;
				}
			}
			if (retired) set = TableSet(buffer, pair, &variant, &draw_state);
		}
		table.words.resize(size);
	}
	// A set this translation proposed with pending slots notes the draw's targets, which its commit's completion must not
	// hold as textures (TableCompletePendingImages).
	if (spec_state != nullptr && set != nullptr && !set->pending_images.empty() && set->proposer == spec_state->id)
		for (const auto& target: draw_state.targets)
			if (std::ranges::find(set->pending_targets, target.id) == set->pending_targets.end())
				set->pending_targets.push_back(target.id);
	return set;
}

// (Mode 4, TableDraw's spec_in_place: out of line, so the other modes' TableDraw carries none of it.) The speculation's
// key of its executor's last table draw (TableXpr::spec_place) is this draw's: the same translation, segment, chain head,
// images epoch, overlay and open-write generations and capture.
template <typename Table>
[[gnu::noinline]] static bool TableSpecInPlace(const Table& table, const Spec::State& spec, uint64_t images_epoch,
                                               uint64_t capture_id) {
	return table.spec_place.id == spec.id && table.spec_place.segment == spec.closed.size() &&
	       table.spec_place.chain == spec.previous.load(std::memory_order_acquire) && table.spec_place.images == images_epoch &&
	       table.spec_place.overlay == spec.overlay_gen && table.spec_place.open == spec.open_gen &&
	       table.spec_place.capture == capture_id;
}
// The speculation's facts an emission can note (a meta answer, a left clear, a consumed clear): their count.
[[gnu::noinline]] static size_t TableSpecFacts(const Spec::State& spec) {
	return spec.meta_facts.size() + spec.left_clears.size() + spec.consume_pending.size();
}
// The draw's key for spec_in_place, its own transits and writes taken (`keyable`, and no fact noted since
// `facts_before`), else none.
template <typename Table>
[[gnu::noinline]] static void TableSpecPlaceNote(Table& table, const Spec::State& spec, uint64_t images_epoch, uint64_t capture_id,
                                                 size_t facts_before, bool keyable) {
	if (TableSpecFacts(spec) == facts_before && keyable)
		table.spec_place = {.id      = spec.id,
		                    .images  = images_epoch,
		                    .overlay = spec.overlay_gen,
		                    .open    = spec.open_gen,
		                    .capture = capture_id,
		                    .segment = spec.closed.size(),
		                    .chain   = spec.previous.load(std::memory_order_acquire)};
	else
		table.spec_place.id = 0;
}

RenderExecutor::TableResult RenderExecutor::TableDraw(CommandBuffer& buffer, TablePair& pair, TableVariant& variant,
                                                      const NativeXprDrawState& draw_state,
                                                      std::span<const uint64_t> commands, uint64_t index_base,
                                                      uint64_t index_bytes, vk::IndexType index_type,
                                                      const NativeXprDirectDraw* direct) {
	return TableDraw<Weave::Ordered>(buffer, pair, variant, draw_state, commands, index_base, index_bytes, index_type,
	                                 direct);
}

// Under Weave::Worker the draw is recorded into the worker's strand (its view of the GPU thread's command buffer,
// through the apart marker) with what its strand has bound. The worker begins no render pass (the turn begins the
// part's; WeavePass ends one and begins the next inside a strand), takes the indices and arguments from live buffers
// that hold them clean (or CPU-written indices, for the turn to upload), binds the pipeline's handle as it is, and
// leaves to the turn the targets' emission and the set's transit (found in place, or a need each), the BDA page table
// (a hole), and every stamp (the draw state's use, the pipeline's, the buffers').
// (TableDraw's table buffer lookups under relocation: their references recorded relocatable. At file scope: a local
// class of the generic lambda inside the TableDraw template would get no inline destructor from clang.)
struct RelocScope {
	uint8_t old;
	explicit RelocScope(uint8_t kind): old(std::exchange(Spec::t_reloc_scope, kind)) {}
	~RelocScope() { Spec::t_reloc_scope = old; }
};

template <typename P>
RenderExecutor::TableResult RenderExecutor::TableDraw(CommandBuffer& buffer, TablePair& pair, TableVariant& variant,
                                                      const NativeXprDrawState& draw_state,
                                                      std::span<const uint64_t> commands, uint64_t index_base,
                                                      uint64_t index_bytes, vk::IndexType index_type,
                                                      const NativeXprDirectDraw* direct) {
	if constexpr (P::worker) WeavePass(buffer, draw_state.rendering);
	auto&       table  = TableOf<P>();
	// (A speculative translation's state, read once behind a plain global: KYTY_SPECULATE unset, no thread-local
	// reads in the draw. The mode kept for the set's target test below.)
	const auto         speculate_mode = ::kyty_local_speculate_mode.load(std::memory_order_relaxed);
	Spec::State* const spec_state     = speculate_mode != 0 ? Spec::Current() : nullptr;
	// (A speculation's relocatable index buffer and draw arguments, TableRelocRecord: nothing of it runs without one.)
	const bool reloc = spec_state != nullptr && spec_state->reloc_on;
	const auto& sh     = buffer.GetShaders();
	const auto  stages = pair.programs[1] != nullptr ? 2u : 1u;
	// (KYTY_SPEC_PAIR_PUSH_FIT's row, mode 4 only: the mode in a register, the pair's flag not read in the other modes.)
	if (speculate_mode == 4 && pair.push_fit) [[unlikely]] TableAdd<P>(LiveCounters::SpecPairPushFitDraws);
	const std::array<std::span<const uint32_t>, 2> user_data {std::span<const uint32_t> {sh.GetVs().gs_user_sgpr.value},
	                                                          std::span<const uint32_t> {sh.GetPs().ps_user_sgpr.value}};
	// (The index buffer's held entry, a cold line of a 320 KiB table, loads while the tables are evaluated.)
	if (index_base != 0) {
		const auto* held = &table.indices[(index_base * 0x9e3779b97f4a7c15ull) >> 51u];
		__builtin_prefetch(held);
		__builtin_prefetch(reinterpret_cast<const char*>(held + 1) - 1);
	}
	std::optional<TableCensus<P>> phase;
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 1);
	// KYTY_SCOUT: the scout's evaluation when its ticket holds (2: compared with the render thread's own); not in a
	// speculation (the scout follows the render thread's submissions), nor on a weave worker (its tickets are the GPU
	// thread's packets').
	const auto* scout = !P::worker && Scout::Running() && spec_state == nullptr ? TableScoutTicket(pair, user_data, stages, m_context.GetGpuResources()) : nullptr;
	if (scout != nullptr && Scout::Mode() == 1) {
		TableScoutApply(table, pair, *scout, stages);
	} else {
		const bool evaluated = TableEvaluate<P>(pair, user_data);
		if (evaluated) TableGather<P>(pair, user_data);
		if (scout != nullptr) TableScoutVerify(table, pair, *scout, stages, evaluated);
		if (!evaluated)
			return TableAdd<P>(LiveCounters::TableDrawNative), Spec::Refuse("table: evaluation"), TableResult::Native;
	}
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 2);
	auto* set = TableSet<P>(buffer, pair, &variant, &draw_state);
	// (Mode 4: a set resolved under another draw state, TableDrawStateSet, out of line.)
	// (Not a weave worker's: it writes the GPU thread's table and catalog. A worker's set made for other targets
	// fails TableTransitPlaced: a Transit need, which the turn's TableTransit refuses.)
	if constexpr (!P::worker)
		if (set != nullptr && speculate_mode == 4) [[unlikely]]
			set = TableDrawStateSet(buffer, pair, variant, draw_state, set, spec_state);
	if (spec_state != nullptr && set != nullptr)
		spec_state->NoteSet(reinterpret_cast<uint64_t>(static_cast<VkDescriptorSet>(set->set)));
	if (set == nullptr) return TableAdd<P>(LiveCounters::TableNoSet), Spec::Refuse("table: no set"), TableResult::Native;
	// (Mode 4: a state a worker derived from a sibling's capture has that capture's attachment layouts, made for shaders that
	// sampled no target: a set that holds a target leaves the table path, and an entry not yet published with it. The GPU
	// thread's in-order draw: its caller takes the store path, t_store_derived.)
	if (draw_state.derived && !TableSetApart(*set, draw_state)) {
		if (t_entry_poison != nullptr) t_entry_poison->store(true, std::memory_order_relaxed);
		if (spec_state == nullptr && Role() == 0 && Spec::Current() == nullptr) t_store_derived = true;
		return Spec::Refuse("table: derived state, a target sampled"), TableResult::Native;
	}

	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 3);
	// Emission, as NativeXprEmit's: indices and arguments are read by the GPU; lookups may restart the
	// scheduler, so the command buffer handle is taken after them.
	auto& buffer_cache = m_context.GetBufferCache();
	// (KYTY_SPEC_DRAW_INDIRECT: a draw of vertices of arguments the GPU reads, VkDrawIndirectCommand; obtained as a run's.)
	const bool     vertex_arguments = direct != nullptr && direct->indirect;
	const uint64_t stride = vertex_arguments ? sizeof(vk::DrawIndirectCommand) : sizeof(vk::DrawIndexedIndirectCommand);
	// A tracker region's epoch, read once in the draw (its buffers mostly share one: the line is another core's).
	std::array<std::pair<uint64_t, uint64_t>, 4> region_epochs;
	region_epochs.fill({UINT64_MAX, 0});
	const auto cpu_epoch = [&](uint64_t address, uint64_t size) {
		const auto region = address / TRACKER_REGION_SIZE;
		if (region != (address + size - 1) / TRACKER_REGION_SIZE) return buffer_cache.CpuModificationEpoch(address, size);
		auto& known = region_epochs[region & 3u];
		if (known.first != region) known = {region, buffer_cache.CpuModificationEpoch(address, size)};
		return known.second;
	};
	// (A weave worker's note of an index or argument range it took, tested by its CPU epoch at the turn.)
	const auto note_buffer = [&](bool args, uint64_t begin, uint64_t end, uint64_t epoch) {
		if constexpr (P::worker)
			WeaveParcel().result->held_buffers.push_back(
			    {.args = args, .begin = begin, .end = end, .cpu_epoch = epoch});
	};
	const auto obtain = [&](TableXpr::HeldBuffer& held, uint64_t address, uint64_t size,
	                        auto&& prepare) -> std::pair<Buffer*, uint64_t> {
		const auto* spec = spec_state; // (its guest memory writes wait for its commit: ObtainBuffer decides)
		if (held.buffer != nullptr && address >= held.begin && address + size <= held.end &&
		    held.registration == buffer_cache.RegistrationEpoch() && held.cpu_epoch != 0 &&
		    cpu_epoch(held.begin, held.end - held.begin) == held.cpu_epoch &&
		    (spec == nullptr || !spec->JournalIntersects(address, size))) {
			if (spec != nullptr) spec_state->NoteBufferRange(address, size);
			note_buffer(&held == &table.args, held.begin, held.end, held.cpu_epoch);
			return {held.buffer, held.buffer->Offset(address)};
		}
		if constexpr (P::worker) {
			// A lookup in the buffer cache, an upload, the GPU thread's held entry are the GPU thread's. A live buffer
			// that holds the range, none of it CPU-modified, is what ObtainBuffer returns with nothing to upload (no
			// small copy into the ring, nothing to synchronize): kept in the worker's own entry, the buffer's last use
			// the turn's stamp. One that holds CPU-written indices is taken too, noted with an epoch no range has
			// (~epoch): the turn uploads them (BufferCache::SynchronizeAtTurn) before the splice. Anything else is a
			// hole.
		// (The epochs before the lookup and before the clean check.)
			const bool args                = &held == &table.args;
			const auto registration_before = buffer_cache.RegistrationEpoch();
			const auto epoch               = buffer_cache.CpuModificationEpoch(address, size);
			const auto [found, id]         = buffer_cache.FindLiveBuffer(address, size);
			if (found == nullptr || epoch == 0) return {};
			const bool clean = TableRangeClean(buffer_cache, address, size);
			if (!clean && args) return {};
			auto* const live = const_cast<Buffer*>(found); // (bound by its handle and offset)
			if (clean) held = {address, address + size, epoch, registration_before, live};
			auto& result = *WeaveParcel().result;
			result.touched.push_back({.buffer  = std::bit_cast<uint64_t>(id),
			                          .handle  = WeaveHandleBits(static_cast<VkBuffer>(live->Handle())),
			                          .address = address,
			                          .size    = size});
			note_buffer(args, address, address + size, clean ? epoch : ~epoch);
			if (!clean) result.made.back().cpu_indices = true;
			return {live, live->Offset(address)};
		}
		held = {};
		TableCensus<P> miss(LiveCensus::DrawPhase, 0x7ab1e, &held == &table.args ? 10 : 11);
		prepare();
		const auto registration_before = buffer_cache.RegistrationEpoch(); // (a speculation's: before its lookup)
		std::optional<BufferCache::AheadSite> ahead_site; // (the lookup's site: read by a speculation's lookup only)
		if (spec_state != nullptr) ahead_site.emplace(&held == &table.args ? 4 : 3);
		// (Relocation on: these lookups' references are recorded relocatable, not noted as other references: RelocScope.)
		const auto lookup = [&] { return buffer_cache.ObtainBuffer(address, size, false); };
		const auto result = reloc ? (RelocScope(&held == &table.args ? Spec::RelocArgs : Spec::RelocIndex), lookup()) : lookup();
		region_epochs.fill({UINT64_MAX, 0}); // (its uploads may have moved them)
		const auto owner  = spec_state != nullptr ? buffer_cache.RegisteredBuffer(address, size)
		                                               : buffer_cache.FindBuffer(address, size);
		if (result.first != nullptr && buffer_cache.GetBufferIfLive(owner) == result.first) {
			// The buffer within the tracker region of the address (one region's epoch proves it) when none of it is
			// CPU-dirty, else the range. (The epoch before the dirty check: a write after it moves the epoch.)
			const auto region = address & ~(TRACKER_REGION_SIZE - 1);
			auto       begin  = std::max(result.first->CpuAddress(), region);
			auto       end    = std::min(result.first->CpuAddress() + result.first->Size(), region + TRACKER_REGION_SIZE);
			if (address + size <= end) {
				const auto epoch = buffer_cache.CpuModificationEpoch(begin, end - begin);
				if (buffer_cache.IsRegionCpuModified(begin, end - begin)) {
					begin = address;
					end   = address + size;
				}
				if (epoch != 0 && !buffer_cache.IsRegionCpuDirty(begin, end - begin))
					held = {begin, end, epoch, spec_state != nullptr ? registration_before : buffer_cache.RegistrationEpoch(),
					        result.first};
			}
		}
		return result;
	};
	std::array<vk::DeviceAddress, 2> blocks {};
	bool                             writes = false;
	// (Before the blocks, which it does not depend on (the slots TableEvaluate read): the entries TableGather asked
	// for meanwhile arrive.) (The page table ranges a stage's global loads read are made current at every use: the GPU
	// thread's, a weave worker's draw with one a hole.)
	if constexpr (P::worker) {
		for (uint32_t st = 0; st < stages; ++st)
			if (pair.programs[st]->table_plan.global_memory) return WeaveMiss(Weave::Miss::Memory), TableResult::Native;
	} else {
		TablePrepareBda(pair, user_data);
	}
	// The blocks take each buffer's device address as it resolves: a later lookup that registers a buffer (a merge
	// replaces an earlier one) leaves an earlier address stale, where the normal path binds after all its lookups
	// (RebindBuffers). Such a draw takes the normal path; the next use finds the buffers registered.
	const auto registration = buffer_cache.RegistrationEpoch();
	if (!TableBlocks<P>(pair, user_data, blocks, &writes))
		return TableAdd<P>(LiveCounters::TableDrawNative), Spec::Refuse("table: blocks"), TableResult::Native;
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 9);
	std::array<std::pair<Buffer*, uint64_t>, 64> arguments;
	std::array<uint32_t, 64>                     group_sizes;
	uint32_t                                     groups = 0;
	for (size_t first = 0; first < commands.size();) {
		size_t count = 1;
		while (first + count < commands.size() && commands[first + count] == commands[first] + count * stride) ++count;
		const auto [owner, offset] = obtain(table.args, commands[first], count * stride, [] {});
		if (owner == nullptr) return TableResult::Native;
		arguments[groups]   = {owner, offset};
		group_sizes[groups] = static_cast<uint32_t>(count);
		++groups;
		first += count;
	}
	auto&      emit       = TableEmit<P>();
	const bool indexed    = direct == nullptr || direct->indexed; // (a draw of vertices binds no indices)
	const bool same_index = !indexed || (emit.command == buffer.Handle() && emit.generation == buffer.GraphicsGeneration() &&
	                                     emit.index_base == index_base && emit.index_bytes == index_bytes &&
	                                     emit.index_type == index_type);
	std::pair<Buffer*, uint64_t> indices {};
	if (!same_index) {
		indices = obtain(table.indices[(index_base * 0x9e3779b97f4a7c15ull) >> 51u], index_base, index_bytes, [&] {
			if (index_bytes <= BufferCache::CACHING_PAGESIZE && buffer_cache.IsRegionCpuModified(index_base, index_bytes) &&
			    !buffer_cache.IsRegionGpuModified(index_base, index_bytes))
				buffer_cache.EnsureBufferContents(index_base, index_bytes);
		});
		if (indices.first == nullptr) return TableResult::Native;
	}
	// The registration epoch moved during the lookups (the GPU thread's registrations, anywhere: a speculation's lookups
	// register nothing). The GPU thread's draw, and any draw outside mode 4: Native as before. A worker's draw (mode 4) is
	// kept and its segment's commit decides, as it already does for every reference the draw records, since the
	// speculation began (at or before this draw's epoch) and with the finer predicate (regions, then granules):
	// - the index buffer and draw arguments are relocatable cells (reloc_refs, with their buffer ranges: RelocEligible /
	//   RelocPatch re-point them to the live buffer; a plain bind over a game buffer is a reloc_unrel range, below);
	// - the table V#s are spec_reloc_refs' entries with their `since` (looked up again when registered over since);
	// - any other game buffer reference is a buffer range and a reloc_unrel range (Changed() stops; RelocEligible refuses a
	//   moved one): a segment stopped there runs as a hole, as any other.
	// Before, it was refused unless no registration since reached its open segment's buffer ranges (TableWorkerKeepsMoved,
	// still the dispatch's test), and TableTry's caller ran it as a "native record" hole.
	if (buffer_cache.RegistrationEpoch() != registration &&
	    (spec_state == nullptr || speculate_mode != 4))
		return TableAdd<P>(LiveCounters::TableRegistered), TableResult::Native;
	// (A speculation's lookups and preparation refuse without failing: nothing is recorded for a refused draw.)
	if (spec_state != nullptr && spec_state->refused != nullptr) return TableResult::Native;
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 4);
	if (buffer.IsInvalid()) return Spec::Refuse("table: buffer"), TableResult::Native;
	const auto vk_buffer    = buffer.Handle();
	const bool same_command = emit.command == vk_buffer && emit.generation == buffer.GraphicsGeneration();
	// The targets and the set's images as the last table draw in this command buffer left them (the same state and
	// set), no image's state, registration, metadata or bytes changed since: in place, nothing to check or transit.
	// (A weave worker's last draw is in its strand, whose targets and set the turn places with the strand's first: in
	// place too. Its draw state's and set's stamps are the GPU thread's: never `own`.)
	auto&      texture_cache = m_context.GetTextureCache();
	const auto images_epoch  = texture_cache.ImagesEpoch();
	const bool own           = !P::worker && spec_state == nullptr;
	// (A set with an image the draw reads and writes is transited for every draw: KYTY_DRAW_READ_AFTER_WRITE.)
	const bool read_after_write =
	    set->read_after_write && kyty_local_draw_read_after_write.load(std::memory_order_relaxed) != 0;
	if constexpr (P::worker)
		if (read_after_write) return WeaveMiss(Weave::Miss::Set), TableResult::Native; // (a barrier at every draw)
	const bool in_place      = same_command && emit.state == &draw_state && table.last_set == set &&
	                      table.images_epoch == images_epoch && (own || P::worker) && !read_after_write;
	// (Mode 4) A speculation's draw in place too (spec_in_place: NativeXprEmitTargets and TableTransit skipped) when its
	// executor's last table draw (TableXpr::spec_place, taken at that draw's end) drew the same draw state (and capture)
	// and set into the same command buffer, as the GPU thread's in_place, in the same translation, segment and chain
	// before it, and since then:
	// - TextureCache::ImagesEpoch has not moved from its value before that draw's checks (image states, registration,
	//   metadata, CPU-written bytes: what the GPU thread's own in_place trusts for the cache's side);
	// - no overlay entry was taken (State::overlay_gen: Image::SpeculativeEntry, through which every transition and every
	//   read of the speculation's image states goes, its first touch in the segment included);
	// - the open segment noted no write (State::open_gen: NoteWritten, Journal; OpenSegmentWrites* read only those).
	// Then the two calls would read what they read for that draw and decide the same: no refusal (it passed), no
	// transition (the overlay holds the layouts it left), and their notes are already the segment's: the entries' found
	// states and used marks (per segment), the sampled levels (min / max of the same ranges), pending targets (in the
	// segment already). A draw whose emission noted a meta answer, a left clear or a consumed clear leaves no key (those
	// answers came from the cache, whose bits some changes move without moving ImagesEpoch), nor does a worker-captured
	// draw state (its depth access follows its target's uniformity, which a partial transit splits).
	// (TableSpecInPlace, TableSpecFacts, TableSpecPlaceNote: out of line.)
	// (Never a weave worker's: its `own` is false with no speculation state.)
	const bool spec_emit     = !P::worker && !own && speculate_mode == 4;
	const bool spec_in_place = spec_emit && same_command && emit.state == &draw_state && table.last_set == set &&
	                           !read_after_write && TableSpecInPlace(table, *spec_state, images_epoch, draw_state.capture_id);
	const size_t spec_facts_before = spec_emit ? TableSpecFacts(*spec_state) : 0;
	// KYTY_TABLE_SERIALS: the checks again only when a serial sum moved (1), or as before with the sums counted (2).
	const auto serials = own ? TableSerialsMode() : 0u;
	if (!in_place && !spec_in_place) {
		// The same for the targets apart, as this draw state's last emission left them, and the set's images, as its
		// last transit under this state did (the passes alternate materials: another set or state in between).
		const bool epoch_targets = !own || draw_state.emitted_epoch != images_epoch;
		const auto targets_sum   = serials != 0 ? TableTargetSerials(draw_state) : 0;
		const bool sum_targets   = !own || draw_state.emitted_serials != targets_sum;
		// (A weave worker's targets and set are found in place, read only, or are needs of the turn's: the layouts
		// follow the GPU thread's recording order.)
		const bool due_targets   = !P::worker && (serials == 1 ? sum_targets : epoch_targets);
		if (serials != 0 && epoch_targets != sum_targets)
			LiveCounters::Add(sum_targets ? LiveCounters::SerialsExtra : LiveCounters::SerialsSkipped);
		if constexpr (P::worker) WeaveHere().TargetsPlaced(*this, draw_state, *set, images_epoch);
		if (due_targets && !NativeXprEmitTargets(draw_state, vk_buffer, set, &buffer)) {
			if (serials == 2 && !sum_targets) TableSerialsUnsafe("targets refused");
			return LiveCounters::Add(LiveCounters::TableDrawNative), Spec::Refuse("table: targets"), TableResult::Native;
		}
		if (serials == 2 && due_targets && !sum_targets && TableTargetSerials(draw_state) != targets_sum)
			TableSerialsUnsafe("targets moved");
		// The set's images in their layouts (a DCC image's pending fast clear applied first).
		const bool epoch_transit = !own || set->transit_epoch != images_epoch || set->transit_state != &draw_state;
		const auto set_sum       = serials != 0 ? TableSetSerials(*set) : 0;
		const bool sum_transit   = !own || set->transit_serials != set_sum ||
		                         ((set->transit_state != &draw_state || set->transit_capture != draw_state.capture_id) &&
		                          !TableSetApart(*set, draw_state));
		const bool due_transit   = !P::worker && ((serials == 1 ? sum_transit : epoch_transit) || read_after_write);
		if (serials != 0 && epoch_transit != sum_transit)
			LiveCounters::Add(sum_transit ? LiveCounters::SerialsExtra : LiveCounters::SerialsSkipped);
		if constexpr (P::worker) WeaveHere().TransitPlaced(*this, *set, draw_state, images_epoch);
		if (due_transit && !TableTransit(*set, vk_buffer, &draw_state)) {
			if (serials == 2 && !sum_transit && !read_after_write) TableSerialsUnsafe("transit refused");
			return LiveCounters::Add(LiveCounters::TableDrawNative), Spec::Refuse("table: transit"), TableResult::Native;
		}
		if (serials == 2 && due_transit && !sum_transit && !read_after_write && TableSetSerials(*set) != set_sum)
			TableSerialsUnsafe("transit moved");
		// (Neither moves what the other placed: a target is no image the set holds as a texture.)
		if (own) {
			draw_state.emitted_epoch = set->transit_epoch = texture_cache.ImagesEpoch();
			set->transit_state       = &draw_state;
			set->transit_capture     = draw_state.capture_id;
			if (serials != 0) {
				draw_state.emitted_serials = due_targets || due_transit ? TableTargetSerials(draw_state) : targets_sum;
				set->transit_serials       = due_targets || due_transit ? TableSetSerials(*set) : set_sum;
			}
		}
	}
	// (A weave worker's pass: WeavePass, first in its draw.)
	if constexpr (!P::worker) {
		if (spec_state != nullptr && draw_state.virtual_targets != nullptr) {
			// (Mode 4: its rendering names colour targets its commit makes: the begin recorded with view cells.)
			const CommandBuffer::RenderingHookScope hook(&TableVirtualRendering, draw_state.virtual_targets.get());
			m_context.GetCommandScheduler().BeginRendering(draw_state.rendering);
		} else {
			m_context.GetCommandScheduler().BeginRendering(draw_state.rendering);
		}
	}
	if (same_command && emit.state != &draw_state && emit.state != nullptr &&
	    emit.state->dynamic_hash == draw_state.dynamic_hash &&
	    std::memcmp(&emit.state->dynamic, &draw_state.dynamic, sizeof(draw_state.dynamic)) == 0)
		emit.state = &draw_state;
	if (!same_command || emit.state != &draw_state) NativeXprEmitDynamicInto(emit, vk_buffer, draw_state, same_command);
	const auto& pipeline = *variant.pipeline;
	if constexpr (P::worker) {
		// (A weave worker reads the handle once and binds it as it is, noted: the pipeline's last use,
		// BindPipelineHandle's, is a stamp. A handle that left while idle (null) is made again at its bind, the GPU
		// thread's: the draw is a hole, and no bind of a null handle is ever recorded.)
		const vk::Pipeline handle = pipeline.pipeline;
		if (!handle) return WeaveMiss(Weave::Miss::Variant), TableResult::Native;
		if (!same_command || emit.pipeline != &pipeline) {
			WeaveParcel().result->made.back().pipeline_handle = WeaveHandleBits(static_cast<VkPipeline>(handle));
			vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
		}
	} else if (!same_command || emit.pipeline != &pipeline) {
		// (Mode 4: a parked entry's stand-in (no handle): bound through a cell its commit writes.)
		if (spec_state != nullptr && t_entry_parked != nullptr && &pipeline == t_entry_parked->parked_stand_in.get()) {
			if (!TableParkedBind(vk_buffer, *t_entry_parked)) return Spec::Refuse("parked pipeline: bind not recorded"), TableResult::Native;
		} else {
			vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, BindPipelineHandle(pipeline));
		}
	}
	if (!same_command || emit.pipeline != &pipeline || table.last_set != set)
		vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.pipeline_layout, 0, 1, &set->set, 0,
		                             nullptr);
	// Each stage's user data and the pixel stage's LOD statistics words, where its push data starts.
	ShaderRecompiler::IR::PushData push_data;
	uint32_t                       push_dwords = 0; // the stages' shader data end there: nothing after it is pushed
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program  = *pair.programs[st];
		const auto& bindings = program.bindings;
		if (!bindings.UsesPushData()) continue;
		push_dwords = std::max(push_dwords, bindings.push_data_start_dword + bindings.ShaderDataDwords());
		for (size_t i = 0; i < bindings.user_data_registers.size(); ++i) {
			const auto index = bindings.user_data_registers[i] - program.user_data_base;
			push_data.dwords[bindings.push_data_start_dword + i] = index < user_data[st].size() ? user_data[st][index] : 0;
		}
		// (KYTY_SPEC_PAIR_DENSE_INDIRECT: a program with its LOD statistics words in its block has none here: TableEvaluate
		// put them there.)
		if (program.stage == ShaderType::Pixel && !bindings.lod_stats_in_block)
			for (size_t i = 0; i < table.lod_stats.size(); ++i)
				push_data.dwords[bindings.push_data_start_dword + bindings.LodStatsDword() + i] = table.lod_stats[i];
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword()]     = static_cast<uint32_t>(blocks[st]);
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword() + 1] = static_cast<uint32_t>(blocks[st] >> 32u);
	}
	if (push_dwords != 0)
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
		                        push_dwords * sizeof(uint32_t), push_data.dwords.data());
	if (!same_index && !(reloc && TableRelocRecord(vk_buffer, *spec_state, Spec::RelocIndex, indices, index_base, index_bytes,
	                                               0, static_cast<uint32_t>(static_cast<VkIndexType>(index_type)), 0)))
		vk_buffer.bindIndexBuffer(indices.first->Handle(), indices.second, index_type);
	if (direct != nullptr && !vertex_arguments) {
		// This draw's offsets, as the normal path's ResolveVertexOffset/ResolveInstanceOffset give them: the index
		// offset register, else (a vertex program that fetches) the user data its fetch adds.
		const auto& vertex = *pair.programs[0];
		const auto  word   = [&](int32_t sgpr) {
			const auto index = int64_t {sgpr} - vertex.user_data_base;
			return index >= 0 && index < static_cast<int64_t>(user_data[0].size()) ? user_data[0][index] : 0u;
		};
		const uint32_t index_offset   = buffer.GetUserConfig().GetIndexOffset();
		const uint32_t vertex_offset  = variant.fetch_embedded && index_offset == 0 ? word(vertex.info.vertex_offset_sgpr)
		                                                                            : index_offset;
		const uint32_t first_instance = variant.fetch_embedded ? word(vertex.info.instance_offset_sgpr) : 0u;
		if (indexed)
			vk_buffer.drawIndexed(direct->index_count, direct->instance_count, 0, static_cast<int32_t>(vertex_offset),
			                      first_instance);
		else
			vk_buffer.draw(direct->index_count, direct->instance_count, vertex_offset, first_instance);
	}
	// (A draw of vertices of GPU-read arguments: their first vertex and instance from them, as the normal path's
	// DrawAuto with gpu_args gives them (DrawOffsetSource::IndirectArgs), not from the index offset or user data.)
	for (uint32_t group = 0, group_first = 0; group < groups; group_first += group_sizes[group++]) {
		// (Relocation: the group's guest address, its first command, the groups' sizes summed before it.)
		if (reloc && TableRelocRecord(vk_buffer, *spec_state, Spec::RelocArgs, arguments[group], commands[group_first],
		                              uint64_t {group_sizes[group]} * stride, vertex_arguments ? 1u : 2u, group_sizes[group],
		                              static_cast<uint32_t>(stride)))
			continue;
		if (vertex_arguments)
			vk_buffer.drawIndirect(arguments[group].first->Handle(), arguments[group].second, group_sizes[group],
			                       static_cast<uint32_t>(stride));
		else
			vk_buffer.drawIndexedIndirect(arguments[group].first->Handle(), arguments[group].second, group_sizes[group],
			                              static_cast<uint32_t>(stride));
	}
	// As the normal path's: shaders that write buffers end the rendering, and a barrier orders their writes.
	if (writes) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader);
	}
	emit.MovedTo(vk_buffer, buffer.GraphicsGeneration());
	emit.state       = &draw_state;
	emit.pipeline    = &pipeline;
	if constexpr (!P::worker) draw_state.Use(m_context.FrameNumber());
	if (indexed) {
		emit.index_base  = index_base;
		emit.index_bytes = index_bytes;
		emit.index_type  = index_type;
	}
	table.last_pair    = &pair;
	table.last_variant = static_cast<size_t>(&variant - pair.variants.data());
	table.last_state   = &draw_state;
	table.last_set     = set;
	table.images_epoch = texture_cache.ImagesEpoch();
	// (Mode 4: this draw's key for spec_in_place, its own transits and writes taken; none after an emission that noted a
	// cache answer, or for a worker-captured state.)
	if (spec_emit) [[unlikely]]
		TableSpecPlaceNote(table, *spec_state, images_epoch, draw_state.capture_id, spec_facts_before,
		                   !draw_state.worker_captured);
	if constexpr (P::worker) {
		// (What the turn stamps, tests and lists of the draw; the rest of its record is the worker's walk's.)
		auto& made         = WeaveParcel().result->made.back();
		made.pair          = &pair;
		made.set           = set;
		made.state         = &draw_state;
		made.vs            = pair.vs;
		made.ps            = pair.ps;
		made.variant_index = static_cast<uint32_t>(&variant - pair.variants.data());
		made.pipeline      = &pipeline;
		made.indexed       = indexed;
		made.index_base    = indexed ? index_base : 0;
		made.index_bytes   = indexed ? index_bytes : 0;
		made.gbuffer       = FrameGen::Enabled() && pair.programs[0] != nullptr &&
		               FrameGen::IsGbufferVertexShader(pair.programs[0]->shader_hash);
	}
	if (!P::worker && vertex_arguments && groups != 0) { // (KYTY_SPEC_DRAW_INDIRECT_VERIFY: what it read)
		auto&       noted  = m_table_args_emit;
		const auto* source = arguments[0].first;
		const auto  mapped = source->Mapped();
		noted              = {source, arguments[0].second, static_cast<uint32_t>(stride), group_sizes[0]};
		// (A ring's copy: no game buffer (those are at their guest address), host-visible.)
		noted.copy = source->CpuAddress() == 0 && mapped.size() >= arguments[0].second + sizeof(noted.copied);
		if (noted.copy) std::memcpy(noted.copied.data(), mapped.data() + arguments[0].second, sizeof(noted.copied));
	}
	TableAdd<P>(LiveCounters::TableDraws, std::max<size_t>(commands.size(), 1));
	return TableResult::Drawn;
}

// The pair's runs (TablePair::Run): the slots its stages evaluate, in reads of consecutive slots, by level.
void RenderExecutor::TableRuns(TablePair& pair) {
	using Kind  = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto& table = *m_table_xpr;
	std::array<TableFlatStage, 2> flat_forms; // (packed into the pair: TablePack)
	for (uint32_t st = 0; st < 2; ++st) {
		if (pair.programs[st] == nullptr) continue;
		const auto&           plan = pair.programs[st]->table_plan;
		std::vector<uint32_t> levels(plan.slots.size(), 0);
		const auto            same = [](const ShaderRecompiler::IR::TablePlan::Operand& a,
		                     const ShaderRecompiler::IR::TablePlan::Operand& b) {
			return a.kind == b.kind && a.value == b.value;
		};
		pair.runs[st].clear();
		for (uint32_t first = 0; first < plan.slots.size();) {
			if (plan.cpu[first] == 0) {
				++first;
				continue;
			}
			const auto& slot  = plan.slots[first];
			uint32_t    count = 1;
			while (first + count < plan.slots.size() && plan.cpu[first + count] != 0 &&
			       same(plan.slots[first + count].low, slot.low) && same(plan.slots[first + count].high, slot.high) &&
			       same(plan.slots[first + count].offset, slot.offset) &&
			       plan.slots[first + count].bounded == slot.bounded && same(plan.slots[first + count].records, slot.records) &&
			       plan.slots[first + count].immediate == slot.immediate + static_cast<int32_t>(4 * count))
				++count;
			uint32_t level = 0;
			for (const auto* operand: {&slot.low, &slot.high, &slot.offset, &slot.records})
				if (operand->kind == Kind::Slot) level = std::max(level, levels[operand->value] + 1);
			std::fill_n(levels.begin() + first, count, level);
			pair.runs[st].push_back({slot, first, count, level});
			first += count;
		}
		std::ranges::stable_sort(pair.runs[st], {}, &TablePair::Run::level);
		table.addresses.resize(std::max(table.addresses.size(), pair.runs[st].size()));
		// The flat form: a user data register by its index (TableEvaluate takes it where the stage's user data hold it,
		// value() reads one beyond them as 0: one of 64 and up is the constant 0), a slot by its index, an immediate
		// from the constants (the first is 0).
		const auto& program = *pair.programs[st];
		auto&       flat    = flat_forms[st];
		flat.constants.push_back(0);
		const auto flat_of = [&](const ShaderRecompiler::IR::TablePlan::Operand& operand) -> TableFlatStage::Operand {
			if (operand.kind == Kind::Slot) return {operand.value, 1};
			if (TableCodeOperand(operand)) {
				flat.constants.push_back(TableCodeWord(operand, st == 0 ? pair.vs : pair.ps));
				return {static_cast<uint32_t>(flat.constants.size() - 1), 2};
			}
			if (operand.kind == Kind::UserData) {
				const auto index = operand.value - program.user_data_base;
				if (index >= 64) return {0, 2};
				flat.user_words = std::max(flat.user_words, index + 1);
				return {index, 0};
			}
			flat.constants.push_back(operand.value);
			return {static_cast<uint32_t>(flat.constants.size() - 1), 2};
		};
		const auto& runs = pair.runs[st];
		for (size_t i = 0; i < runs.size(); ++i) {
			const auto& run = runs[i];
			flat.runs.push_back({{flat_of(run.read.low), flat_of(run.read.high), flat_of(run.read.offset)},
			                     run.read.immediate, run.first, run.count, flat_of(run.read.records), run.read.bounded});
			if (i + 1 == runs.size() || runs[i + 1].level != run.level) flat.level_ends.push_back(static_cast<uint32_t>(i + 1));
		}
		for (const auto& descriptor: plan.images) {
			flat.image_counts.push_back(descriptor.count);
			for (uint32_t w = 0; w < descriptor.count; ++w) flat.image_words.push_back(flat_of(descriptor.words[w]));
		}
		for (const auto& descriptor: plan.samplers)
			for (uint32_t w = 0; w < descriptor.count; ++w) flat.image_words.push_back(flat_of(descriptor.words[w]));
		for (const auto& words: plan.buffers)
			for (const auto& word: words) flat.buffer_words.push_back(flat_of(word));
	}
	TablePack(pair, flat_forms);
}

// The pair's flat forms packed into one block (TablePair::packed): a stage whose operand indices, counts or levels do not
// fit their packed widths is left invalid (evaluated by its plan's operands).
void RenderExecutor::TablePack(TablePair& pair, const std::array<TableFlatStage, 2>& flat_forms) {
	using Pair = TablePair;
	pair.packed       = {};
	pair.packed_block = nullptr;
	pair.packed_at    = nullptr;
	pair.packed_bytes = 0;
	const auto operand = [](const TableFlatStage::Operand& from, bool& fits) -> uint16_t {
		if (from.base > 2 || from.index >= 0x4000u) fits = false;
		return static_cast<uint16_t>(from.base << 14u | (from.index & 0x3fffu));
	};
	const auto align = [](size_t at, size_t to) { return (at + to - 1) / to * to; };
	// (Layout per stage: runs, constants, level ends, image words, buffer words, image counts.)
	std::array<std::array<size_t, 6>, 2> offsets {};
	size_t                               bytes = 0;
	std::array<bool, 2>                  fits {};
	for (uint32_t st = 0; st < 2; ++st) {
		if (pair.programs[st] == nullptr) continue;
		const auto& flat = flat_forms[st];
		fits[st]         = flat.runs.size() < 0x10000 && flat.image_words.size() < 0x10000;
		for (const auto end: flat.level_ends) fits[st] = fits[st] && end < 0x10000;
		for (const auto count: flat.image_counts) fits[st] = fits[st] && count < 0x100;
		for (const auto& run: flat.runs) fits[st] = fits[st] && run.first < 0x10000 && run.count < 0x10000;
		if (!fits[st]) continue;
		bytes          = align(bytes, 64);
		offsets[st][0] = bytes;
		bytes += flat.runs.size() * sizeof(Pair::PackedRun);
		offsets[st][1] = bytes;
		bytes += flat.constants.size() * sizeof(uint32_t);
		offsets[st][2] = bytes;
		bytes += flat.level_ends.size() * sizeof(uint16_t);
		offsets[st][3] = bytes;
		bytes += flat.image_words.size() * sizeof(uint16_t);
		offsets[st][4] = bytes;
		bytes += flat.buffer_words.size() * sizeof(uint16_t);
		offsets[st][5] = bytes;
		bytes += flat.image_counts.size();
	}
	if (bytes == 0) return;
	pair.packed_block = std::make_unique<std::byte[]>(bytes + 64);
	auto* base        = reinterpret_cast<std::byte*>(align(reinterpret_cast<uintptr_t>(pair.packed_block.get()), 64));
	pair.packed_at    = base;
	pair.packed_bytes = static_cast<uint32_t>(bytes);
	for (uint32_t st = 0; st < 2; ++st) {
		if (!fits[st]) continue;
		const auto& flat    = flat_forms[st];
		auto&       packed  = pair.packed[st];
		bool        operand_fits = true;
		auto*       runs    = reinterpret_cast<Pair::PackedRun*>(base + offsets[st][0]);
		for (size_t i = 0; i < flat.runs.size(); ++i) {
			const auto& run = flat.runs[i];
			runs[i]         = {.address   = {operand(run.address[0], operand_fits), operand(run.address[1], operand_fits),
			                                 operand(run.address[2], operand_fits)},
			                   .records   = run.bounded ? operand(run.records, operand_fits) : Pair::PackedRun::NotBounded,
			                   .immediate = run.immediate,
			                   .first     = static_cast<uint16_t>(run.first),
			                   .count     = static_cast<uint16_t>(run.count)};
			if (run.bounded && runs[i].records == Pair::PackedRun::NotBounded) operand_fits = false;
		}
		auto* constants = reinterpret_cast<uint32_t*>(base + offsets[st][1]);
		std::copy(flat.constants.begin(), flat.constants.end(), constants);
		auto* level_ends = reinterpret_cast<uint16_t*>(base + offsets[st][2]);
		for (size_t i = 0; i < flat.level_ends.size(); ++i) level_ends[i] = static_cast<uint16_t>(flat.level_ends[i]);
		auto* image_words = reinterpret_cast<uint16_t*>(base + offsets[st][3]);
		for (size_t i = 0; i < flat.image_words.size(); ++i) image_words[i] = operand(flat.image_words[i], operand_fits);
		auto* buffer_words = reinterpret_cast<uint16_t*>(base + offsets[st][4]);
		for (size_t i = 0; i < flat.buffer_words.size(); ++i) buffer_words[i] = operand(flat.buffer_words[i], operand_fits);
		auto* image_counts = reinterpret_cast<uint8_t*>(base + offsets[st][5]);
		for (size_t i = 0; i < flat.image_counts.size(); ++i) image_counts[i] = static_cast<uint8_t>(flat.image_counts[i]);
		packed = {.runs         = {runs, flat.runs.size()},
		          .level_ends   = {level_ends, flat.level_ends.size()},
		          .image_words  = {image_words, flat.image_words.size()},
		          .buffer_words = {buffer_words, flat.buffer_words.size()},
		          .image_counts = {image_counts, flat.image_counts.size()},
		          .constants    = {constants, flat.constants.size()},
		          .user_words   = flat.user_words,
		          .valid        = operand_fits};
	}
}

// After a prepared draw (`storable`: a draw without vertex buffers) of a pair the table path asked to
// store: the pair's table mode programs (with this draw's specializations), the pipeline for this draw state,
// compiled by a worker, and the draw state.
void RenderExecutor::TableStore(CommandBuffer& buffer, const DrawRenderState& state, vk::PrimitiveTopology topology,
                                bool primitive_restart_enable, const RenderState& rendering, bool storable) {
	auto&       table = *m_table_xpr;
	auto* const pair  = std::exchange(table.store, nullptr);
	if (!storable || pair == nullptr) return;
	const auto& sh = buffer.GetShaders();
	// (A request whose packet never reached a prepared draw must not store for another state.)
	if (sh.GetVs().es_regs.data_addr != pair->vs || sh.GetPs().ps_regs.data_addr != pair->ps ||
	    NativeXprStateKey(buffer) != table.store_state)
		return;
	auto&           pipelines   = m_context.GetPipelineCache();
	const auto&     ctx         = buffer.GetRegisters();
	DrawRenderState table_state = state;
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> mapping {};
	for (uint32_t i = 0; i < state.color_count; i++)
		mapping[state.color_info[i].target_slot] = state.color_info[i].export_mapping;
	// (spec_format_ahead, mode 4 only: the pair's birth lookup takes its recorded pair's mask: PipelineCache::t_fa_birth.)
	const bool fa_birth = pair->programs[0] == nullptr && FormatAheadOn();
	if (fa_birth) [[unlikely]] PipelineCache::t_fa_birth = true;
	table_state.programs = pipelines.GetGraphicsPrograms(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx,
	                                                     buffer.GetUserConfig(), mapping, state.ps_active,
	                                                     table_state.vs_input_info, table_state.ps_input_info, true,
	                                                     pair->portable);
	if (pipelines.TablePending()) {
		pair->waiting = pipelines.TableWaiting(); // (the pair stores again once it is done)
		return;
	}
	const auto* vertex = table_state.vs_input_info.stage.program;
	const auto* pixel  = state.ps_active ? table_state.ps_input_info.stage.program : nullptr;
	const auto  refuse = [&](const char* reason) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
			std::printf("Table: VS 0x%llx PS 0x%llx refused: %s\n", static_cast<unsigned long long>(pair->vs),
			            static_cast<unsigned long long>(pair->ps), reason);
		const auto write = CatalogWrite();
		const Weave::Hold hold;
		pair->refused    = true;
	};
	if (!table_state.programs.vertex || vertex == nullptr || !vertex->table_mode)
		return refuse("no table mode vertex program");
	if (vertex->stage != ShaderType::Vertex) return refuse("not a vertex program");
	if (state.ps_active && (!table_state.programs.pixel || pixel == nullptr || !pixel->table_mode))
		return refuse("no table mode pixel program");
	const std::array<const ShaderRecompiler::IR::CompiledShaderInfo*, 2> programs {vertex, pixel};
	if (pair->programs[0] == nullptr) {
		{
			const auto write = CatalogWrite();
			const Weave::Hold hold;
			pair->programs   = programs;
			pair->push_fit   = Spec::On() && PipelineCache::t_push_fit_taken; // (the lookup above: mode 4 sets it)
			// (spec_format_ahead: the buffers its recorded pair's mask made portable, kept as learned bits: every later
			// lookup of the pair passes them in, so gets the form it was born on.)
			if (fa_birth) [[unlikely]] pair->portable[0] |= PipelineCache::t_fa_pair_bits;
			TableRuns(*pair);
			TableScoutPublish(*pair);
		}
		const auto count = [](const ShaderRecompiler::IR::CompiledShaderInfo* program, bool cpu) -> size_t {
			if (program == nullptr) return 0;
			const auto& plan = program->table_plan;
			return static_cast<size_t>(std::ranges::count(cpu ? plan.cpu : plan.gpu, uint8_t {1}));
		};
		std::printf("Table: VS 0x%llx (0x%016llx) PS 0x%llx (0x%016llx): reads %zu+%zu slots (evaluates %zu+%zu), "
		            "buffers %zu+%zu, images %zu, samplers %zu\n",
		            static_cast<unsigned long long>(pair->vs), static_cast<unsigned long long>(vertex->shader_hash),
		            static_cast<unsigned long long>(pair->ps),
		            static_cast<unsigned long long>(pixel != nullptr ? pixel->shader_hash : 0), count(vertex, false),
		            count(pixel, false), count(vertex, true), count(pixel, true), vertex->table_plan.buffers.size(),
		            pixel != nullptr ? pixel->table_plan.buffers.size() : 0,
		            vertex->table_plan.images.size() + (pixel != nullptr ? pixel->table_plan.images.size() : 0),
		            vertex->table_plan.samplers.size() + (pixel != nullptr ? pixel->table_plan.samplers.size() : 0));
	} else if (pair->programs != programs) {
		// (KYTY_TABLE_LEARN_PORTABLE: a state whose programs differ only in runtime-format buffers' V# format, as the
		// resolver's refusals did (Demon's Souls' view save: 4 states, 173 draws a frame as holes in mode 4), learns those
		// buffers as portable and the pair compiles again, instead of the state refused for good.)
		if (TableLearnPortable() != 0) {
			std::array<uint32_t, 2> masks {};
			bool                    other = false;
			for (uint32_t st = 0; st < 2 && !other; ++st) {
				if (pair->programs[st] == programs[st]) continue;
				const auto mask = pipelines.TableFormatOnlyBuffers(pair->programs[st], programs[st]);
				if (mask == UINT32_MAX || mask == 0) {
					other = true;
					break;
				}
				const auto& buffers = pair->programs[st]->info.buffers;
				for (uint32_t i = 0; i < 32; ++i)
					if (((mask >> i) & 1u) != 0 && (i >= buffers.size() || !ShaderRecompiler::IR::RuntimeBufferFormat(buffers[i])))
						other = true;
				masks[st] = mask;
			}
			if (!other && ((pair->portable[0] | masks[0]) != pair->portable[0] || (pair->portable[1] | masks[1]) != pair->portable[1])) {
				const auto write = CatalogWrite();
				const Weave::Hold hold;
				pair->portable[0] |= masks[0];
				pair->portable[1] |= masks[1];
				pair->respecialize = true;
				return;
			}
		}
		// Another specialization (the pixel program's export of another draw state's targets): this state's draws
		// keep the native path.
		const auto write = CatalogWrite();
		const Weave::Hold hold;
		const auto found = std::ranges::find(pair->variants, table.store_state, &TableXpr::Variant::state);
		if (found == pair->variants.end()) pair->variants.push_back({.state = table.store_state, .refused = true});
		return;
	}
	const auto* pipeline = pipelines.TryCreateGraphicsPipeline(
	    std::span {state.color_info, state.color_count}, state.depth_info, table_state.vs_input_info, buffer,
	    state.ps_active ? &table_state.ps_input_info : nullptr, topology, primitive_restart_enable,
	    table_state.programs.vertex, table_state.programs.pixel, true);
	if (pipeline == nullptr) pair->waiting = pipelines.PipelineWaiting();
	{
		const auto write   = CatalogWrite();
		const Weave::Hold hold;
		auto       variant = std::ranges::find(pair->variants, table.store_state, &TableXpr::Variant::state);
		if (variant == pair->variants.end()) variant = pair->variants.insert(variant, {.state = table.store_state});
		variant->pipeline       = pipeline;
		variant->mapping        = mapping;
		variant->fetch_embedded = table_state.vs_input_info.fetch_embedded;
	}
	NativeXprCaptureState(buffer, state, rendering, table.store_state, topology, primitive_restart_enable);
}

// --- Table dispatches ---------------------------------------------------------------------------------------------
//
// KYTY_TABLE_DISPATCH=1: a compute shader the normal path dispatched twice (its resource tables evaluated in full),
// whose program allows table mode, is dispatched as table draws are drawn: its slots in a block, its read-only V#s
// resolved once, its written ones obtained as written at every dispatch (the buffer cache tracks the GPU's writes),
// its images and samplers in a set; the barriers and the compute chain as the normal path's DispatchDirect. Shaders
// a special path of the normal path took (linear copies, clears), indirect thread-dimension dispatches, frame
// capture and frame generation stay on the normal path.

// The compute stage's registers that select the program (ShaderGetStaticInputInfoCS reads them), and whether the
// dispatch counts threads (another program: ShaderComputeInputInfo::dispatch_thread_dimensions).
static std::array<uint64_t, 4> TableDispatchWords(const HW::ComputeShaderInfo& cs, bool thread_dimensions) {
	const auto& r = cs.cs_regs;
	return {r.data_addr, uint64_t {r.num_thread_x} | uint64_t {r.num_thread_y} << 32u,
	        uint64_t {r.num_thread_z} | uint64_t {r.lds_size} << 32u,
	        uint64_t {r.wave_size} | uint64_t {r.user_sgpr} << 8u | uint64_t {r.tidig_comp_cnt} << 16u |
	            uint64_t {r.tgid_x_en} << 24u | uint64_t {r.tgid_y_en} << 25u | uint64_t {r.tgid_z_en} << 26u |
	            uint64_t {r.tg_size_en} << 27u | uint64_t {r.scratch_en} << 28u | uint64_t {thread_dimensions} << 29u};
}
static uint64_t TableDispatchKey(const std::array<uint64_t, 4>& words) {
	return XXH3_64bits(words.data(), sizeof(words));
}
static uint64_t TableDispatchKey(const HW::ComputeShaderInfo& cs, bool thread_dimensions) {
	return TableDispatchKey(TableDispatchWords(cs, thread_dimensions));
}

// The normal path dispatched (`special`: one of its special paths took the dispatch).
void RenderExecutor::TableDispatchSeen(const HW::ComputeShaderInfo& cs, const char* special, bool thread_dimensions) {
	if (kyty_local_table_dispatch_mode.load(std::memory_order_relaxed) == 0 || Role() != 0) return;
	auto&      dispatches = OwnTable().dispatches;
	const auto key        = TableDispatchKey(cs, thread_dimensions);
	auto       it         = dispatches.find(key);
	if (it == dispatches.end() || (special != nullptr && !it->second.refused)) {
		const auto write = CatalogWrite();
		it               = dispatches.try_emplace(key).first;
		it->second.vs    = cs.cs_regs.data_addr;
		if (special != nullptr) {
			it->second.refused = true;
			it->second.special = special;
		}
	}
	++it->second.sightings;
	it->second.used_frame = m_context.FrameNumber();
}

bool RenderExecutor::TableDispatch(CommandBuffer& buffer, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
                                   uint64_t indirect_args, bool thread_dimensions) {
	auto* catalog = CatalogTable();
	m_dispatch_declined = "dispatch: nothing learned";
	if (catalog == nullptr || FrameCapture::Active() || XprCapture::Enabled()) return false;
	std::shared_lock<std::shared_mutex> catalog_read;
	if (Role() != 0) CatalogGate(Catalog().m_catalog_writer_waiting, CatalogWriterFirst()), catalog_read = CatalogRead();
	const CatalogReadScope catalog_read_scope(&catalog_read); // (spec_worker_sets)
	DropStaleCatalog();
	TableSweepPairs();
	const auto& sh    = buffer.GetShaders();
	const auto& cs    = sh.GetCs();
	const auto  words = TableDispatchWords(cs, thread_dimensions);
	// (The pair found last again without its hash or lookup: runs of dispatches keep their program. A pointer stays
	// valid until TableSweepPairs, which drops this one; the main executor's own catalog only.)
	const bool own   = Role() == 0 && catalog == m_table_xpr.get();
	auto*      known = own && catalog->found_dispatch_key == words ? catalog->found_dispatch : nullptr;
	if (known == nullptr) {
		const auto found = catalog->dispatches.find(TableDispatchKey(words));
		// (Another executor's: a pair with its programs and pipeline had its sightings.)
		if (found == catalog->dispatches.end()) return m_dispatch_declined = "dispatch: not seen", false;
		known = &found->second;
		if (own) {
			catalog->found_dispatch_key = words;
			catalog->found_dispatch     = known;
		}
	}
	if (known->refused)
		return m_dispatch_declined = known->special != nullptr ? known->special : "dispatch: refused", false;
	if ((Role() == 0 && known->sightings < 2) || known->vs != cs.cs_regs.data_addr) return false;
	auto& pair = *known;
	// (The game's 3D view; a dispatch declined below goes the normal way, which notes it too. A speculation's thread,
	// translating ahead of the flips, notes it in its segment, which its commit notes in order (Speculation::After): it
	// was never noted, and the frame of its commit, as a menu's, waited a second vblank at 120 Hz. A refused dispatch's
	// note goes with its packet's (Rollback): the hole that runs it notes it.)
	if (pair.programs[0] != nullptr && FrameGen::IsTaaShader(pair.programs[0]->shader_hash)) {
		if (Role() == 0) FrameGen::NoteScene();
		else if (auto* spec = Spec::Current(); spec != nullptr) ++spec->scenes;
	}
	// KYTY_FRAMEGEN: the TAA dispatch goes the normal way, whose hook reads its depth and motion-vector images (all
	// dispatches did: +3 ms a frame at 1-1).
	if (FrameGen::Enabled() && pair.programs[0] != nullptr && FrameGen::IsTaaShader(pair.programs[0]->shader_hash))
		return m_dispatch_declined = "dispatch: frame generation inputs", false;
	TablePrefetchPair(pair);
	if (Role() == 0) pair.used_frame = m_context.FrameNumber();
	else if (auto* spec = Spec::Current(); spec != nullptr) spec->NoteUsedPair(&pair.used_frame); // (as TableTry's)
	if (Role() == 0) TablePrefetchNext(OwnTable().previous_dispatch, pair);
	LiveCensus::Scope census(LiveCensus::DrawPhase, 0x7ab1f, 0); // census: the dispatches' phases (b)
	// The dispatch executor's sets retire here too (NativeXprTry collects only the draw executor's): without it they
	// were never reused or freed, a new pool every 1024.
	NativeXprCollect();
	// (Another executor's pair: its programs and pipeline as its owner made them, or the normal path's dispatch.)
	if (Role() != 0 && (pair.programs[0] == nullptr || pair.pipeline == nullptr || pair.respecialize))
		return Spec::Refuse("table program"), false;
	if (const auto generation = ShaderMapGeneration(); generation != pair.Generation(Role())) {
		const ShaderHashNoFaultScope no_fault {Role() != 0 && !GuestGpu::IsGpuThread()}; // (a worker under the catalog: no read that can fault)
		if (pair.programs[0] != nullptr && ShaderHashAt(pair.vs) != pair.programs[0]->shader_hash) {
			if (Role() != 0) return Spec::Refuse("table: shaders changed"), false;
			const auto write = CatalogWrite();
			pair.portable    = {};
			TableReset(pair);
		}
		pair.Generation(Role()) = generation;
	}
	if (pair.respecialize) {
		const auto write = CatalogWrite();
		TableReset(pair);
	}
	if (pair.waiting && Role() == 0) {
		if (!pair.waiting->load(std::memory_order_acquire)) return false;
		pair.waiting.reset();
	}
	TablePrefetchArrays(pair, pair.LastSet(Role()));
	auto& pipelines = m_context.GetPipelineCache();
	if (Spec::Current() != nullptr && (pair.programs[0] == nullptr || pair.pipeline == nullptr))
		return Spec::Refuse("table program"), false;
	if (pair.programs[0] == nullptr) {
		// The table mode program of the shader the normal path prepared.
		pair.compute = std::make_unique<ShaderComputeInputInfo>();
		ResetNativeStageInput(*pair.compute);
		pair.compute->dispatch_thread_dimensions = thread_dimensions;
		pair.compute_program = pipelines.GetComputeProgram(cs, buffer.GetRegisters().GetShaderRegisters(), *pair.compute,
		                                                   nullptr, true, pair.portable[0]);
		const auto* program = pair.compute->stage.program;
		if (pipelines.TablePending()) {
			pair.waiting = pipelines.TableWaiting(); // (asked again once it is done)
			return false;
		}
		if (!pair.compute_program || program == nullptr || !program->table_mode) {
			static std::atomic<uint32_t> reports {0};
			if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
				std::printf("Table: CS 0x%llx refused: no table mode program\n", static_cast<unsigned long long>(pair.vs));
			const auto write = CatalogWrite();
			pair.refused     = true;
			return false;
		}
		{
			const auto write = CatalogWrite();
			pair.programs[0] = program;
			TableRuns(pair);
		}
		std::printf("Table: CS 0x%llx (0x%016llx): reads %zu slots (evaluates %zu), buffers %zu (written %zu), "
		            "images %zu, samplers %zu\n",
		            static_cast<unsigned long long>(pair.vs), static_cast<unsigned long long>(program->shader_hash),
		            static_cast<size_t>(std::ranges::count(program->table_plan.gpu, uint8_t {1})),
		            static_cast<size_t>(std::ranges::count(program->table_plan.cpu, uint8_t {1})),
		            program->table_plan.buffers.size(),
		            static_cast<size_t>(std::ranges::count_if(program->info.buffers, [](const auto& item) {
			            return item.written || item.atomic;
		            })),
		            program->table_plan.images.size(), program->table_plan.samplers.size());
	}
	if (pair.pipeline == nullptr) {
		const auto* pipeline = pipelines.TryCreateComputePipeline(*pair.compute, pair.compute_program);
		{
			const auto write = CatalogWrite();
			pair.pipeline    = pipeline;
		}
		if (pair.pipeline == nullptr) {
			pair.waiting = pipelines.PipelineWaiting();
			return false;
		}
	}
	const std::array<std::span<const uint32_t>, 2> user_data {std::span<const uint32_t> {cs.cs_user_sgpr.value},
	                                                          std::span<const uint32_t> {}};
	std::optional<LiveCensus::Scope> phase;
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 1);
	if (kyty_local_table_indirect_mode.load(std::memory_order_relaxed) == 0 &&
	    (!pair.programs[0]->table_plan.indirect.empty() || pair.programs[0]->table_plan.dynamic_reads))
		return m_dispatch_declined = "table: indirect off", false;
	if (!TableEvaluate(pair, user_data)) return m_dispatch_declined = "table: tables unmapped", false;
	TableGather(pair, user_data);
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 2);
	auto* set = TableSet(buffer, pair, nullptr, nullptr);
	if (set == nullptr) return false;
	if (auto* spec = Spec::Current()) spec->NoteSet(reinterpret_cast<uint64_t>(static_cast<VkDescriptorSet>(set->set)));
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 3);
	std::array<vk::DeviceAddress, 2> blocks {};
	bool                             storage_writes  = false;
	auto&                            cache           = m_context.GetBufferCache();
	vk::Buffer                       indirect_buffer = nullptr;
	uint64_t                         indirect_offset = 0;
	// (Mode 4) A speculation's dispatch whose lookups saw the buffer registration epoch move under them takes back its
	// notes since its BDA preparation (State::RollbackTo) and takes its preparation, blocks and arguments once more at
	// once, from the epoch then; a second move refuses as before. A first take that recorded work (RecordedWrites moved)
	// is refused as before, never taken again, so no recorded work outlives its notes. What the first take did besides
	// its notes (ring copies no command reads, posted ranges) is what a refused packet's Rollback leaves too.
	auto* const retry_spec =
	    kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4 ? Spec::Current() : nullptr;
	std::optional<Spec::Counts> retry_mark;
	bool                        retry_bda  = false;
	uint64_t                    retry_work = 0;
	if (retry_spec != nullptr) {
		retry_mark.emplace(retry_spec->Sizes());
		retry_bda  = retry_spec->bda_all;
		retry_work = LocalVulkanRecording::RecordedWrites();
	}
	for (uint32_t attempt = 0;; ++attempt) {
		TablePrepareBda(pair, user_data); // (as TableDraw's: before the blocks)
		const auto registration = cache.RegistrationEpoch(); // (as TableDraw's)
		if (!TableBlocks(pair, user_data, blocks, &storage_writes)) return m_dispatch_declined = "table: blocks", false;
		if (indirect_args != 0) {
			// As DispatchDirect's: the arguments' owner, made current.
			cache.EnsureBufferContents(indirect_args, 3u * sizeof(uint32_t));
			const auto id = cache.FindBuffer(indirect_args, 3u * sizeof(uint32_t));
			if (Spec::Current() != nullptr && Spec::Current()->refused != nullptr) return false;
			auto& owner     = cache.GetBuffer(id);
			indirect_buffer = owner.Handle();
			indirect_offset = owner.Offset(indirect_args);
		}
		if (cache.RegistrationEpoch() == registration) break;
		if (retry_spec != nullptr && retry_spec->refused == nullptr) {
			if (attempt == 0 && LocalVulkanRecording::RecordedWrites() == retry_work) {
				// (The blocks and arguments again, from the epoch now: what the commit checks is noted again by them.)
				retry_spec->RollbackTo(*retry_mark, retry_bda);
				blocks          = {};
				storage_writes  = false;
				indirect_buffer = nullptr;
				indirect_offset = 0;
				continue;
			}
			// Keep the dispatch if registrations since its epoch missed the open segment's buffers.
			if (TableWorkerKeepsMoved(*retry_spec, cache, registration)) {
				LiveCounters::Add(LiveCounters::SpecTablePatchRegisteredKept);
				break;
			}
			LiveCounters::Add(LiveCounters::SpecTablePatchRegisteredRefused);
		}
		return LiveCounters::Add(LiveCounters::TableRegistered), m_dispatch_declined = "table: buffers registered meanwhile", false;
	}
	// (A speculation's lookups and preparation refuse without failing: nothing is recorded for a refused dispatch.)
	if (const auto* spec = Spec::Current(); spec != nullptr && spec->refused != nullptr) return false;
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 4);
	// Emission, as DispatchDirect's.
	buffer.EndRendering();
	const bool chain     = DemonsSouls::ComputeChains();
	const auto vk_buffer = chain ? buffer.ChainHandle() : buffer.Handle();
	// (As a draw's: the set's images as its last dispatch's transit left them, while no image changed since.)
	auto&      dispatch_textures = m_context.GetTextureCache();
	const bool own_transit       = Role() == 0 && Spec::Current() == nullptr;
	if ((!own_transit || set->transit_epoch != dispatch_textures.ImagesEpoch() || set->transit_state != nullptr) &&
	    !TableTransit(*set, vk_buffer, nullptr))
		return m_dispatch_declined = "table: transit", false;
	if (own_transit) {
		set->transit_epoch = dispatch_textures.ImagesEpoch();
		set->transit_state = nullptr;
	}
	// (A set made for another specialization: its program's pipeline and shader data.)
	const auto& pipeline = set->pipeline != nullptr ? *set->pipeline : *pair.pipeline;
	const auto& program  = set->program != nullptr ? *set->program : *pair.programs[0];
	const auto& bindings = program.bindings;
	vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.pipeline_layout, 0, 1, &set->set, 0, nullptr);
	if (bindings.UsesPushData()) {
		ShaderRecompiler::IR::PushData push_data;
		for (size_t i = 0; i < bindings.user_data_registers.size(); ++i) {
			const auto index = bindings.user_data_registers[i] - program.user_data_base;
			push_data.dwords[bindings.push_data_start_dword + i] = index < user_data[0].size() ? user_data[0][index] : 0;
		}
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword()]     = static_cast<uint32_t>(blocks[0]);
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword() + 1] = static_cast<uint32_t>(blocks[0] >> 32u);
		vk_buffer.pushConstants(pipeline.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
		                        (bindings.push_data_start_dword + bindings.ShaderDataDwords()) * sizeof(uint32_t),
		                        push_data.dwords.data());
	}
	const bool continues_chain = chain && buffer.ComputeChainPending();
	if (storage_writes && !continues_chain) ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, BindPipelineHandle(pipeline));
	if (indirect_args != 0) {
		vk::BufferMemoryBarrier args_barrier {};
		args_barrier.srcAccessMask       = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite |
		                                   vk::AccessFlagBits::eMemoryWrite;
		args_barrier.dstAccessMask       = vk::AccessFlagBits::eIndirectCommandRead;
		args_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		args_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		args_barrier.buffer              = indirect_buffer;
		args_barrier.offset              = indirect_offset;
		args_barrier.size                = 3u * sizeof(uint32_t);
		if (!continues_chain)
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eDrawIndirect,
			                          vk::DependencyFlags {}, 0, nullptr, 1, &args_barrier, 0, nullptr);
		vk_buffer.dispatchIndirect(indirect_buffer, indirect_offset);
		DeviceFault::NoteDispatch(m_context.GetCommandScheduler().CurrentTick(), program.shader_hash, 0, 0, 0, indirect_args);
	} else if (thread_dimensions) {
		// Thread counts: the groups that cover them, as the normal path's.
		const auto& r      = cs.cs_regs;
		const auto  groups = [](uint32_t threads, uint32_t size) { return (threads + std::max(size, 1u) - 1u) / std::max(size, 1u); };
		vk_buffer.dispatch(groups(groups_x, r.num_thread_x), groups(groups_y, r.num_thread_y), groups(groups_z, r.num_thread_z));
		DeviceFault::NoteDispatch(m_context.GetCommandScheduler().CurrentTick(), program.shader_hash,
		                          groups(groups_x, r.num_thread_x), groups(groups_y, r.num_thread_y),
		                          groups(groups_z, r.num_thread_z), 0);
	} else {
		vk_buffer.dispatch(groups_x, groups_y, groups_z);
		DeviceFault::NoteDispatch(m_context.GetCommandScheduler().CurrentTick(), program.shader_hash, groups_x, groups_y,
		                          groups_z, 0);
	}
	if (chain) buffer.ContinueComputeChain();
	else ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	m_context.GetCommandScheduler().CompleteDispatch();
	// (A later table draw binds its pipeline and set again: the graphics state is untouched.)
	LiveCounters::Add(LiveCounters::TableDispatches);
	return true;
}

// --- Mode 4: a worker's own capture of a draw state ---------------------------------------------------------------------
//
// A speculation's thread, TableWorkerEntry's state choice for a draw whose context no capture exists of (no sibling): the
// state from the worker's own build of the draw's targets, as PrepareDrawRenderState resolves them (slice offset 0: the
// table path's draws have none) and AcquireRenderTargets makes the rendering, but read only (TextureCache::ReadOnlyScope with
// t_read_only_targets: FindImage takes FindTargetReadOnly, the resolvers bind nothing): each target an existing,
// registered, current image already used as that kind of target, with an existing attachment view (FindTargetViewReadOnly);
// a colour target whose in-order FindImage reuses an existing image, that image (its pick noted: `adopted`, proven again at
// the commit); a colour target with no image on its pages, no metadata or DCC metadata, left to the commit to make
// (`virtual_targets`); else null. Layouts attachment-optimal (a colour target's ColorAttachmentOptimal, the depth's
// depth_attachment_layout), the depth's access as the normal path's acquisition gives it (DepthAttachmentAccess, from the
// image's state as this speculation sees it at the draw: its overlay entry, the state at its first touch, which the commit
// restores before the replay; no aspect sampled, a draw under this state whose shaders sample a target leaving the table
// path: TableDraw), no clears (a draw that clears is refused, as a speculation refuses a DB clear), the acquisition left to
// the commit as for any state (spec_commit_targets). Marked `derived`: the table path's only (the native record refuses it;
// a set holding one of its targets leaves the table path, TableDraw, and the GPU thread's in-order draw then stores the
// real state). The dynamic state as NativeXprCaptureState takes it (CaptureZeroed), its indexed viewports set by the caller
// from the normal vertex program. The refusals decided by the registers alone come first (the MSAA resolve, the DB
// clears; the depth target's own clear flags are DB_RENDER_CONTROL's too). `ctx_hash`: the caller's WorkerEntryContextHash.
// The texture cache's lock is taken under the catalog's read lock (the order the workers' image sets keep); no guest
// memory is read and nothing records.
std::unique_ptr<RenderExecutor::NativeXprDrawState> RenderExecutor::WorkerCaptureTargets(CommandBuffer& buffer,
                                                                                         vk::PrimitiveTopology topology,
                                                                                         bool restart, uint64_t ctx_hash) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode == 3) return nullptr; // (an MSAA resolve: the normal path's own copy)
	if (hw.GetRenderControl().depth_clear_enable || hw.GetRenderControl().stencil_clear_enable) return nullptr;
	auto* const spec = Spec::Current();
	if (spec == nullptr) return nullptr;
	// (The attempt's lookup flags, reset on every way out.)
	struct Flags {
		Flags() {
			TextureCache::t_virtual_targets_ok = TableCellsRecordable();
			TextureCache::t_adopt_targets_ok   = true;
			if (!TextureCache::t_adopt_picks.empty()) TextureCache::t_adopt_picks.clear();
		}
		~Flags() {
			TextureCache::t_read_only_targets  = false;
			TextureCache::t_virtual_targets_ok = false;
			TextureCache::t_virtual_hit        = false;
			TextureCache::t_adopt_targets_ok   = false;
			if (!TextureCache::t_adopt_picks.empty()) TextureCache::t_adopt_picks.clear();
		}
		Flags(const Flags&)            = delete;
		Flags& operator=(const Flags&) = delete;
	} flags;
	uint32_t virtual_slots = 0; // (the colour attachments found with no image, a bit each)
	auto     state_storage = std::make_unique<DrawRenderState>();
	auto&    state         = *state_storage;
	{
		const TextureCache::ReadOnlyScope read_only;
		TextureCache::t_read_only_targets = true;
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			if (slot == 0 || (render_target_mask_slot(hw.GetRenderTargetMask(), slot) != 0 && hw.GetRenderTarget(slot).base.addr != 0)) {
				TextureCache::t_virtual_hit = false;
				ResolveRenderColorTarget(buffer, state.color_info[state.color_count], 0, slot);
				if (state.color_info[state.color_count].image_id) {
					state.color_count++;
				} else if (TextureCache::t_virtual_hit) { // (kept: the commit makes its image)
					virtual_slots |= 1u << state.color_count;
					state.color_info[state.color_count].image_id = NativeXprDrawState::VirtualId();
					state.color_count++;
				}
			}
		}
		TextureCache::t_virtual_hit = false;
		ResolveRenderDepthTarget(buffer, state.depth_info);
		TextureCache::t_read_only_targets = false;
		if (TextureCache::ReadOnlyFailed() != nullptr) return nullptr;
		if (TextureCache::t_virtual_hit) return nullptr; // (a depth target is never left to the commit)
	}
	state.ps_active = DrawHasActivePixelShader(buffer);
	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) return nullptr;
	if (state.depth_info.depth_clear_enable || state.depth_info.stencil_clear_enable) return nullptr;
	// The colour targets with no image, made by the commit ahead of the segment's replay (TableEntryResolveVirtual). Declined
	// (refused, as with no image) when that is not where the in-order draw makes it: a pending image of the open segment
	// (spec_worker_images: completed by the commit after this target's making, in order before it); its memory written by
	// the open segment (the making's upload would read it before those writes); another such target over that memory in this
	// translation or this draw (each commit makes its own: in order the second finds the first); a DCC one's metadata with a
	// clear or consume fact of the open segment's own (from after this segment's packet: not in the records the commit's
	// FindRenderTarget reads).
	std::shared_ptr<NativeXprDrawState::Virtual> virt;
	if (virtual_slots != 0) {
		for (const auto& proposed: spec->proposed_sets)
			if (proposed.segment == spec->closed.size() && proposed.complete) return nullptr;
		auto& ranges = TextureCache::t_virtual_ranges;
		ranges.Hold(spec->id);
		virt                       = std::make_shared<NativeXprDrawState::Virtual>();
		virt->segment              = spec->closed.size();
		const auto sealed          = spec->Sealed();
		virt->from_current         = sealed.current;
		virt->from_written         = sealed.written;
		virt->from_invalidated     = sealed.invalidated;
		virt->from_texel_syncs     = sealed.texel_syncs;
		virt->from_buffer_ranges   = sealed.buffer_ranges;
		virt->from_cpu_copies      = sealed.cpu_copies;
		virt->from_pending_buffers = sealed.pending_buffers;
		virt->from_table_ranges    = sealed.table_ranges;
		for (uint32_t i = 0; i < state.color_count; ++i) {
			if (((virtual_slots >> i) & 1u) == 0) continue;
			const auto& info = state.color_info[i].desc.info;
			if (spec->OpenSegmentWritesAny() && spec->OpenSegmentWritesImage(info)) return nullptr;
			if (ranges.Overlaps(info.data.address, info.data.size)) return nullptr;
			for (const auto& other: virt->targets)
				if (other.desc.info.data.address < info.data.address + info.data.size &&
				    info.data.address < other.desc.info.data.address + other.desc.info.data.size)
					return nullptr;
			if (info.metadata.kind == ImageMetadataKind::Dcc)
				for (const auto& fact: spec->meta_facts)
					if (fact.address == info.metadata.range.address && fact.first > spec->closed.size()) return nullptr;
			virt->targets.push_back({i, 0, state.color_info[i].desc});
		}
	}
	uint32_t    virtual_index = 0; // (the next of virt->targets, in attachment order)
	auto&       textures      = m_context.GetTextureCache();
	const auto  metadata_of   = [](const TextureCache::ImageDesc& desc) {
		return desc.info.metadata.kind == ImageMetadataKind::None ? uint64_t {0} : desc.info.metadata.range.address;
	};
	auto        made = std::make_unique<NativeXprDrawState>();
	RenderState rendering {};
	rendering.width                 = std::numeric_limits<uint32_t>::max();
	rendering.height                = std::numeric_limits<uint32_t>::max();
	rendering.num_layers            = std::numeric_limits<uint32_t>::max();
	rendering.num_color_attachments = state.color_count;
	uint32_t samples                = 0;
	for (uint32_t i = 0; i < state.color_count; ++i) {
		const auto&              target = state.color_info[i];
		TextureCache::TargetView found;
		if (((virtual_slots >> i) & 1u) != 0) { // (its view the commit's; a placeholder till then)
			virt->targets[virtual_index].target = static_cast<uint32_t>(made->targets.size());
			found.view                          = virt->Placeholder(virtual_index++);
		} else if (!textures.FindTargetViewReadOnly(target.image_id, target.desc, found)) {
			return nullptr;
		}
		if (samples == 0) samples = target.desc.info.samples;
		else if (samples != target.desc.info.samples) return nullptr;
		const auto& view   = target.desc.view_info;
		const auto  layout = vk::ImageLayout::eColorAttachmentOptimal;
		const auto  access = vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		const auto  extent = target.Extent();
		rendering.width                             = std::min(rendering.width, extent.width);
		rendering.height                            = std::min(rendering.height, extent.height);
		rendering.num_layers                        = std::min(rendering.num_layers, view.layer_count);
		rendering.color_attachments[i].image_view   = found.view;
		rendering.color_attachments[i].image_layout = layout;
		made->targets.push_back({target.image_id, layout, access,
		                         {view.base_level, view.level_count, view.base_layer, view.layer_count},
		                         metadata_of(target.desc), view.base_layer, found.serial, target.desc.info});
	}
	if (const auto& depth = state.depth_info; depth.image_id) {
		TextureCache::TargetView found;
		if (!textures.FindTargetViewReadOnly(depth.image_id, depth.desc, found)) return nullptr;
		if (samples == 0) samples = depth.desc.info.samples;
		else if (samples != depth.desc.info.samples) return nullptr;
		const auto& view   = depth.desc.view_info;
		const auto  layout = depth_attachment_layout(depth);
		// (The access from this speculation's entry for the image: none to read, refused.)
		auto* image = textures.m_slot_images.try_get(depth.image_id);
		auto* entry = image != nullptr && image->serial == found.serial ? image->SpeculativeEntry() : nullptr;
		if (entry == nullptr) return nullptr;
		auto writable = depth.AttachmentWriteAspects() & vk::ImageAspectFlagBits::eStencil;
		if (depth.depth_write_enable) writable |= vk::ImageAspectFlagBits::eDepth;
		const auto access  = DepthAttachmentAccess(layout, entry->state, entry->subresources.empty(), vk::ImageAspectFlags {}, writable);
		const auto aspects = ImageViewOps::DepthAspectMask(view.format);
		rendering.width           = std::min(rendering.width, depth.desc.info.extent.width);
		rendering.height          = std::min(rendering.height, depth.desc.info.extent.height);
		rendering.num_layers      = std::min(rendering.num_layers, view.layer_count);
		auto& attachment          = rendering.depth_stencil_attachment;
		attachment.image_view     = found.view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		made->targets.push_back({depth.image_id, layout, access,
		                         {view.base_level, view.level_count, view.base_layer, view.layer_count},
		                         metadata_of(depth.desc), view.base_layer, found.serial, depth.desc.info});
	}
	if (state.color_count == 0 && !state.depth_info.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		rendering.width    = limits.maxFramebufferWidth;
		rendering.height   = limits.maxFramebufferHeight;
	} else if (samples == 0 || vulkan_sample_count(samples) == vk::SampleCountFlagBits {}) {
		return nullptr;
	}
	if (rendering.num_layers == std::numeric_limits<uint32_t>::max()) rendering.num_layers = 1;
	if (rendering.width == 0 || rendering.height == 0 || rendering.num_layers == 0 ||
	    rendering.width == std::numeric_limits<uint32_t>::max() || rendering.height == std::numeric_limits<uint32_t>::max())
		return nullptr;
	made->first_rendering = rendering;
	made->rendering       = rendering;
	made->meta_epoch      = 0;
	made->capture_id      = g_xpr_captures.fetch_add(1, std::memory_order_relaxed) + 1;
	for (const auto& target: made->targets)
		if (target.id != NativeXprDrawState::VirtualId()) made->target_ids.push_back(target.id); // (no image: no serial to sum)
	made->dynamic = LocalDrawRecording::CaptureZeroed(buffer, state.vs_input_info, state.color_info, state.color_count,
	                                                  state.depth_info, m_context.GetGraphics().attachment_feedback_loop_enabled,
	                                                  vk::ImageAspectFlags {});
	if (Spec::DepthLayers2On()) made->depth_info = std::make_shared<const RenderDepthInfo>(state.depth_info);
	auto inputs = std::make_shared<NativeXprDrawState::PipelineInputs>();
	for (uint32_t i = 0; i < state.color_count; ++i) inputs->color_info[i] = state.color_info[i];
	inputs->color_count              = state.color_count;
	inputs->depth_info               = state.depth_info;
	inputs->ps_active                = state.ps_active;
	inputs->topology                 = topology;
	inputs->primitive_restart_enable = restart;
	made->pipeline_inputs            = std::move(inputs);
	made->ctx_hash                   = ctx_hash;
	made->derived                    = true;
	made->worker_captured            = true;
	made->virtual_targets            = std::move(virt); // (its memory noted once an entry is made on it)
	made->adopted                    = TextureCache::t_adopt_picks;
	return made;
}

// (Mode 4) A worker's refused captures in its open segment (this thread's: its translation's id and segment), by (state
// key, context hash), with the texture cache's CaptureRefusalEpoch read before each attempt. The key and the hash cover the
// shaders, the context registers and the user configuration WorkerCaptureTargets reads; the rest of a refusal is the
// texture cache's (an image registered, given a view or first used as a target moves the epoch: every entry is dropped and
// each refusal is tried again), and the speculation's own segment (its writes: the segment's end drops all). A refused
// capture is not drawn either way: a kept refusal only refuses again what an attempt would have refused, until one of
// those moves (an image's needs_rebind or depth alias alone does not: the segment's end then).
struct CaptureRefusals {
	struct Entry {
		uint64_t key = 0, ctx_hash = 0;
	};
	static constexpr size_t MaxEntries = 256; // (all dropped when full)
	uint64_t                spec_id = 0, epoch = 0;
	size_t                  segment = SIZE_MAX;
	std::vector<Entry>      entries;
	// The entries of this segment at `now`, else none (and kept from here for this segment and epoch).
	void Hold(const Spec::State& spec, uint64_t now) {
		if (spec.id == spec_id && spec.closed.size() == segment && epoch == now) return;
		entries.clear();
		spec_id = spec.id;
		segment = spec.closed.size();
		epoch   = now;
	}
	[[nodiscard]] bool Holds(uint64_t key, uint64_t ctx_hash) const {
		for (const auto& entry: entries)
			if (entry.key == key && entry.ctx_hash == ctx_hash) return true;
		return false;
	}
	void Note(uint64_t key, uint64_t ctx_hash) {
		if (entries.size() >= MaxEntries) entries.clear();
		entries.push_back({key, ctx_hash});
	}
};
static thread_local CaptureRefusals t_capture_refusals;

// --- Mode 4: the worker that meets the draw makes the table entry -----------------------------------------------------
//
// Under KYTY_SPECULATE=4 (WorkerEntryOn), a speculation's draw whose pair is not in the catalog, or whose pair lacks the
// draw-state key's variant or the key's captured state, was refused ("table store"): the GPU thread ran it as a hole, the
// normal draw, and TableStore made the entry. Here the worker makes what TableStore would, from the same inputs, or nothing:
//  - THE GATE (the recording): the draw's normal programs are permutations in memory (GetGraphicsPrograms with t_find_only:
//    none translated or recorded) and its normal pipeline is built in memory (FindGraphicsPipeline); so the normal draw this
//    replaces would record nothing. Or the commit records for it what the normal draw would (below: adopted permutations,
//    a private normal pipeline). Its vertex shader binds no vertex buffers and no mesh stage (TableStore's storable gate).
//  - The draw state: the key's own capture if there is one; else DERIVED from a sibling's, the state captured last under the
//    same context and user configuration hash (other shaders): the targets, layouts and rendering as captured (the context
//    resolves them alike; the sibling's targets current, NativeXprTargetsCurrent), its dynamic state but for the vertex
//    program's indexed viewports (LocalDrawRecording::Capture's one shader input: from the normal vertex program), its
//    hash and resolution recomputed, a capture id of its own. Refused when the sibling's capture holds a feedback-loop or
//    general layout or a feedback aspect (a pixel shader that sampled a target), and a set of a draw under a derived state
//    that holds a target leaves the table path (TableDraw; the holder is then never published; the GPU thread's own draw
//    under it takes the store path, which captures the state again): no feedback guess.
//  - ps_active as the normal path computes it (DrawHasActivePixelShader), equal to the capture's.
//  - The programs: GetGraphicsPrograms in table mode with the capture's export mapping and ps_active (a pair's own portable
//    masks); table programs and ready now (else refused: TableStore's own refusals stay the GPU thread's); for a pair in the
//    catalog, its programs (else another specialization: refused).
//  - The variant: the pair's for the key when stored (its pipeline), else TryCreateGraphicsPipelineWorker as TableStore's
//    TryCreateGraphicsPipeline (ready now; else its compile posted and this draw refused), its mapping and fetch_embedded as
//    TableStore sets them.
//  - A new pair: vs, ps, start instance, the programs, TableRuns; its variant in it; proven by this role (generation).
// A normal program not in memory whose module the prefetch holds done is adopted by the worker (t_adopt_only), under the
// pipeline cache's lock as the GPU thread's normal draw adopts it, but not recorded there: the entry's commit records it
// from the lookup's own inputs (ProgramCache::NoteMet, AdoptedCommit) and, when the normal pipeline of the draw is not in
// memory, puts in the one the worker made privately (CreatePrivateGraphicsPipeline: a Fast first build only, after the
// table gates passed), with its recipe, as the in-order draw's create would have. Nothing is translated: a program the
// prefetch does not hold done stays a refusal.
// The lookups run with the catalog's read lock dropped (the RULE at TableResolveSet), relocked; then the catalog's epoch,
// the pair (absent; or its programs and variant), the base state (its capture) are those held before, else refused. Drawn as
// any worker table draw; proposed (Spec::State::proposed_sets, `pair` null) before the draw, checked at its segment's Before
// (TableEntryCheck: the state's targets current, the pair's programs), published at its commit (TableEntrySettle, under the
// owner's write): a new pair by node insert while the catalog has none for the key, a variant into the pair (as TableStore
// stores one), a derived state into the states while none is there for the key. What is not published whole is kept until no
// speculation is live (g_entry_orphans; TableRetireEntryOrphans retires a kept node's sets).
std::optional<RenderExecutor::TableResult> RenderExecutor::TableWorkerEntry(CommandBuffer& buffer, TablePair* existing,
                                                                           uint64_t pair_hash, std::span<const uint64_t> commands,
                                                                           uint64_t index_base, uint64_t index_bytes,
                                                                           vk::IndexType index_type,
                                                                           const NativeXprDirectDraw* direct) {
	auto* const spec = Spec::Current();
	if (spec == nullptr || Role() == 0 || GuestGpu::IsGpuThread() || CatalogTable() == nullptr ||
	    Catalog().m_native_xpr == nullptr || m_native_xpr == nullptr || FrameGen::Enabled())
		return std::nullopt;
	const auto&    sh             = buffer.GetShaders();
	const auto&    ctx            = buffer.GetRegisters();
	const auto     vs_address     = sh.GetVs().es_regs.data_addr, ps_address = sh.GetPs().ps_regs.data_addr;
	const auto     start_instance = sh.GetVs().start_instance_user_sgpr;
	const auto     key            = m_native_xpr->state_key;
	const uint64_t tag            = existing != nullptr ? static_cast<uint64_t>(reinterpret_cast<uintptr_t>(existing)) : pair_hash;
	const uint64_t mix            = (tag * 0x9e3779b97f4a7c15ull) ^ key;
	const auto     draw           = [&](TableEntryHolder& h) -> std::optional<TableResult> {
		t_entry_variant = h.own_variant ? h.drawn_variant : nullptr;
		t_entry_poison  = &h.poisoned;
		// (A parked entry's: TableDraw binds its stand-in through a cell. Cleared on every way out, a refusal thrown through
		// TableDraw too.)
		struct ParkedScope {
			explicit ParkedScope(TableEntryHolder* entry) noexcept { t_entry_parked = entry; }
			~ParkedScope() { t_entry_parked = nullptr; }
			ParkedScope(const ParkedScope&)            = delete;
			ParkedScope& operator=(const ParkedScope&) = delete;
		} parked_scope(h.parked != nullptr ? &h : nullptr);
		const auto result = TableDraw(buffer, *h.pair, *h.drawn_variant, *h.drawn_state, commands, index_base, index_bytes,
		                              index_type, direct);
		t_entry_variant   = nullptr;
		t_entry_poison    = nullptr;
		OwnTable().last_pair = nullptr; // (TableContinue reads the role's last variant out of the pair's vector)
		return result;
	};
	const auto& states = Catalog().NativeXprState().states;
	// (The recording, per draw: a draw drawn with an entry made with records is served early as the entry's own draw is, so its
	// own normal lookups (find-only, as the entry's: this draw's registers, the entry's export mapping and targets, this draw's
	// topology and restart) must meet only what that entry's commit records and puts in: the unrecorded permutations among its
	// records, the normal pipeline in memory or its own. Else (another specialization or key: what this draw's in-order normal
	// draw would record is in no commit) it is not drawn with that entry; it makes its own below, whose commit holds its
	// records and pipeline (or is refused, and the hole records them). Run with the catalog's read lock dropped, as the entry's
	// lookups; false in `changed`: the catalog as it was (else refused, nothing held across the relock).)
	const auto reuse_covered = [&](const TableEntryHolder& h, bool& changed) {
		changed            = false;
		const auto& inputs = *h.inputs;
		const bool  ps_on  = DrawHasActivePixelShader(buffer);
		if (ps_on != inputs.ps_active) return false;
		vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList;
		bool                  restart  = false;
		if (!WorkerEntryDrawTopology(buffer, direct != nullptr ? direct->indexed : index_bytes != 0, index_type, topology, restart))
			return false;
		const auto since    = CatalogTable()->epoch;
		auto&      cache    = m_context.GetPipelineCache();
		const bool unlocked = t_catalog_read != nullptr && t_catalog_read->owns_lock();
		if (unlocked) t_catalog_read->unlock();
		struct Flags {
			bool reads = false, find_only = false;
			~Flags() {
				if (reads) PipelineCache::t_no_fault_reads = false, PipelineCache::t_strict_reads = false;
				if (find_only) PipelineCache::t_find_only = false, PipelineCache::g_find_only_users.fetch_sub(1, std::memory_order_relaxed);
			}
		} flags;
		PipelineCache::t_no_fault_reads  = true;
		PipelineCache::t_strict_reads    = true; // (a specialization word not read clean fails the lookup: another entry)
		flags.reads                      = true;
		PipelineCache::t_no_fault_failed = false;
		ShaderVertexInputInfo vertex;
		ShaderPixelInputInfo  pixel;
		ResetNativeVertexInput(vertex);
		ResetNativeStageInput(pixel);
		PipelineCache::g_find_only_users.fetch_add(1, std::memory_order_relaxed);
		PipelineCache::t_find_only   = true;
		flags.find_only              = true;
		PipelineCache::t_adopt_ids_n = 0;
		const auto normal = cache.GetGraphicsPrograms(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx, buffer.GetUserConfig(),
		                                              h.variant.mapping, ps_on, vertex, pixel);
		PipelineCache::t_find_only = false;
		PipelineCache::g_find_only_users.fetch_sub(1, std::memory_order_relaxed);
		flags.find_only = false;
		const bool own  = !PipelineCache::t_no_fault_failed && static_cast<bool>(normal.vertex) && vertex.stage.program != nullptr &&
		                 (!ps_on || (static_cast<bool>(normal.pixel) && pixel.stage.program != nullptr)) &&
		                 cache.AdoptedCovers(h.adopted.get(), {inputs.color_info, inputs.color_count}, inputs.depth_info, vertex, buffer,
		                                     ps_on ? &pixel : nullptr, topology, restart, normal.vertex, normal.pixel);
		PipelineCache::t_adopt_ids_n    = 0;
		PipelineCache::t_no_fault_reads = false;
		PipelineCache::t_strict_reads   = false;
		flags.reads                     = false;
		if (unlocked) {
			CatalogGate(Catalog().m_catalog_writer_waiting, CatalogWriterFirst());
			t_catalog_read->lock();
		}
		changed = spec->refused != nullptr || CatalogTable() == nullptr || CatalogTable()->epoch != since;
		return own;
	};
	// (One this translation made already for this pair and key: drawn with again.)
	for (const auto& p: spec->proposed_sets) {
		if (p.pair != nullptr || p.hash != mix) continue;
		auto& h = *static_cast<TableEntryHolder*>(p.set);
		if (h.key != key || h.new_pair != (existing == nullptr) ||
		    (existing != nullptr ? h.pair != existing : h.pair_hash != pair_hash))
			continue;
		// (An entry whose commit makes colour targets is drawn with in its own segment only: a later one's draws would start
		// from the layout a hole between leaves; an entry of its own then, whose capture is refused over that memory.)
		if (h.state != nullptr && h.state->virtual_targets != nullptr && h.state->virtual_targets->segment != spec->closed.size())
			continue;
		// (An entry made with records: this draw's own normal lookups in its commit, above; else the next, or an entry of its
		// own.)
		if (h.reuse_checked) {
			bool       changed = false;
			const bool own     = reuse_covered(h, changed);
			if (changed) return std::nullopt;
			if (!own) continue;
		}
		if (h.poisoned.load(std::memory_order_relaxed)) return std::nullopt;
		if (h.pair->vs != vs_address || h.pair->ps != ps_address || h.pair->start_instance != start_instance) return std::nullopt;
		if (existing != nullptr && (existing->programs != h.programs || existing->refused || existing->respecialize))
			return std::nullopt;
		if (h.state == nullptr) {
			const auto again = states.find(key);
			if (again == states.end() || again->second.get() != h.drawn_state || h.drawn_state->capture_id != h.capture)
				return std::nullopt;
		}
		// (The pair's stored variant: found again, its vector may have grown since: TableStore pushes under its write.)
		if (!h.new_pair && !h.own_variant) {
			const auto v = std::ranges::find(existing->variants, key, &TableVariant::state);
			if (v == existing->variants.end() || v->refused || v->pipeline != h.variant.pipeline) return std::nullopt;
			h.drawn_variant = &*v;
		}
		if (!NativeXprTargetsCurrent(*h.drawn_state)) return std::nullopt;
		return draw(h);
	}
	// (An entry a worker made that is not in the catalog yet, before one of its own: TableEntryOverlay.)
	if (const auto drawn = TableEntryOverlay(buffer, existing, commands, index_base, index_bytes, index_type, direct)) return *drawn;
	std::array<const ShaderRecompiler::IR::CompiledShaderInfo*, 2> programs_held {};
	bool push_fit_held = false; // (KYTY_SPEC_PAIR_PUSH_FIT: a new pair's programs the retry's: TablePair::push_fit)
	uint32_t fa_pair_bits_held = 0; // (spec_format_ahead: a new pair's recorded-pair bits: PipelineCache::t_fa_pair_bits)
	std::array<uint32_t, 2>                                        portable {};
	TableVariant*                                                  stored_variant = nullptr;
	if (existing != nullptr) {
		if (existing->programs[0] == nullptr || existing->refused || existing->respecialize) return std::nullopt;
		programs_held = existing->programs;
		portable      = existing->portable;
		if (const auto v = std::ranges::find(existing->variants, key, &TableVariant::state); v != existing->variants.end()) {
			if (v->refused) return std::nullopt;
			if (v->pipeline != nullptr) stored_variant = &*v;
		}
	}
	// The draw state: the key's own capture, else a sibling's to derive it from, else the worker's own capture (no capture of
	// this context exists: WorkerCaptureTargets).
	std::unique_ptr<NativeXprDrawState> captured;
	const NativeXprDrawState*           base     = nullptr;
	uint64_t                            base_key = key;
	bool                                derive   = false;
	if (const auto own = states.find(key); own != states.end()) {
		base = own->second.get();
	} else {
		const auto  ctx_hash = WorkerEntryContextHash(buffer);
		const auto& index    = Catalog().NativeXprState().sibling_index;
		const auto  named    = index.find(ctx_hash);
		const auto  sibling  = named != index.end() ? states.find(named->second) : states.end();
		if (sibling == states.end() || sibling->second->ctx_hash != ctx_hash) {
			vk::PrimitiveTopology capture_topology = vk::PrimitiveTopology::eTriangleList;
			bool                  capture_restart  = false;
			if (!WorkerEntryDrawTopology(buffer, direct != nullptr ? direct->indexed : index_bytes != 0, index_type, capture_topology,
			                             capture_restart))
				return std::nullopt;
			// (A refusal of this key and context kept in this segment while the texture cache's targets did not move:
			// refused again without the attempt. The epoch read before the attempt: one that moves during it drops what it
			// keeps.)
			auto& refusals = t_capture_refusals;
			refusals.Hold(*spec, m_context.GetTextureCache().CaptureRefusalEpoch());
			if (refusals.Holds(key, ctx_hash)) return std::nullopt;
			captured = WorkerCaptureTargets(buffer, capture_topology, capture_restart, ctx_hash);
			if (captured == nullptr) {
				refusals.Note(key, ctx_hash);
				return std::nullopt;
			}
			base = captured.get();
		} else {
			base     = sibling->second.get();
			base_key = named->second;
		}
		derive = true;
	}
	const auto inputs = base->pipeline_inputs; // (shared: alive after the unlock)
	if (!inputs) return std::nullopt;
	if (derive) {
		constexpr auto feedback = vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT;
		constexpr auto general  = vk::ImageLayout::eGeneral;
		bool           loop     = base->dynamic.feedback_aspects != vk::ImageAspectFlags {};
		for (const auto& target: base->targets) loop |= target.layout == feedback || target.layout == general;
		for (const auto* rendering: {&base->rendering, &base->first_rendering}) {
			for (const auto& attachment: rendering->color_attachments)
				loop |= attachment.image_layout == feedback || attachment.image_layout == general;
			loop |= rendering->depth_stencil_attachment.image_layout == feedback ||
			        rendering->depth_stencil_attachment.image_layout == general;
		}
		if (loop) return std::nullopt;
	}
	if (!NativeXprTargetsCurrent(*base)) return std::nullopt;
	const bool ps_active = DrawHasActivePixelShader(buffer);
	if (ps_active != inputs->ps_active) return std::nullopt;
	if (existing != nullptr && (programs_held[1] != nullptr) != ps_active) return std::nullopt;
	// (This draw's topology and restart, as TableStore takes them: the capture's are its own draw's.)
	vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList;
	bool                  restart  = false;
	if (!WorkerEntryDrawTopology(buffer, direct != nullptr ? direct->indexed : index_bytes != 0, index_type, topology, restart))
		return std::nullopt;
	const auto                          base_capture = base->capture_id;
	std::unique_ptr<NativeXprDrawState> derived;
	// (Copied under the lock: a capture again rewrites it in place. The worker's own capture is its own already.)
	if (derive) derived = captured != nullptr ? std::move(captured) : std::make_unique<NativeXprDrawState>(*base);
	const bool worker_captured = derive && base == derived.get();
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> mapping {};
	for (uint32_t i = 0; i < inputs->color_count; ++i) mapping[inputs->color_info[i].target_slot] = inputs->color_info[i].export_mapping;
	const auto since      = CatalogTable()->epoch;
	const auto generation = ShaderMapGeneration(); // (before the lookups: a new pair is proven as of then)
	auto&      pipelines  = m_context.GetPipelineCache();
	// (The lookups with the catalog's read lock dropped; guest reads without a fault.)
	// (The stored variant's pipeline, read under the lock: TableStore's push_back may move the vector while it is dropped.)
	const PipelineCache::Pipeline* stored_pipeline = stored_variant != nullptr ? stored_variant->pipeline : nullptr;
	const bool unlocked = t_catalog_read != nullptr && t_catalog_read->owns_lock();
	if (unlocked) t_catalog_read->unlock();
	// (The lookups' thread flags, cleared on the way out of a RecoverableExit through them too; each only if set here and not
	// cleared on the normal way. The catalog's relock is not covered.)
	struct LookupFlags {
		bool reads = false, find_only = false, adopt_only = false, records = false;
		~LookupFlags() {
			if (reads) PipelineCache::t_no_fault_reads = false;
			if (adopt_only) PipelineCache::t_adopt_only = false;
			if (records) PipelineCache::t_adopt_records = false;
			if (find_only) PipelineCache::t_find_only = false, PipelineCache::g_find_only_users.fetch_sub(1, std::memory_order_relaxed);
		}
	} flags;
	PipelineCache::t_no_fault_reads  = true;
	flags.reads                      = true;
	PipelineCache::t_no_fault_failed = false;
	ShaderVertexInputInfo normal_vertex;
	ShaderPixelInputInfo  normal_pixel;
	ResetNativeVertexInput(normal_vertex);
	ResetNativeStageInput(normal_pixel);
	PipelineCache::g_find_only_users.fetch_add(1, std::memory_order_relaxed);
	PipelineCache::t_find_only   = true;
	flags.find_only              = true;
	PipelineCache::t_adopt_ids_n = 0; // (the unrecorded permutations this lookup finds, adopted earlier by a worker's lookup)
	// (These lookups make the records of the unrecorded permutations they meet, from their own inputs, for the entry's commit.)
	PipelineCache::t_adopt_records = true;
	flags.records                  = true;
	auto normal = pipelines.GetGraphicsPrograms(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx, buffer.GetUserConfig(),
	                                            mapping, ps_active, normal_vertex, normal_pixel);
	uint32_t unrecorded_met     = PipelineCache::t_adopt_ids_n;
	bool     normal_read_failed = PipelineCache::t_no_fault_failed;
	bool gate_programs = !normal_read_failed && static_cast<bool>(normal.vertex) && normal_vertex.stage.program != nullptr &&
	                     (!ps_active || (static_cast<bool>(normal.pixel) && normal_pixel.stage.program != nullptr));
	// (A program not in memory whose module the prefetch holds done is adopted here, as the GPU thread's normal draw would
	// adopt it: under the pipeline cache's lock (GetGraphicsPrograms takes it; the catalog's is dropped above), the permutation
	// published but not recorded here: the entry's commit records it with these lookups' own inputs, else the first in-order
	// lookup that finds it. Nothing is translated: a program the prefetch does not hold done stays a refusal, the GPU thread's
	// draw makes it.)
	if (!gate_programs && !normal_read_failed) {
		ResetNativeVertexInput(normal_vertex);
		ResetNativeStageInput(normal_pixel);
		PipelineCache::t_adopt_only      = true;
		flags.adopt_only                 = true;
		PipelineCache::t_adopt_adopted   = 0;
		PipelineCache::t_adopt_ids_n     = 0;
		PipelineCache::t_no_fault_failed = false;
		normal = pipelines.GetGraphicsPrograms(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx, buffer.GetUserConfig(), mapping,
		                                       ps_active, normal_vertex, normal_pixel);
		PipelineCache::t_adopt_only = false;
		flags.adopt_only            = false;
		unrecorded_met              = PipelineCache::t_adopt_ids_n;
		normal_read_failed          = PipelineCache::t_no_fault_failed;
		gate_programs = !normal_read_failed && static_cast<bool>(normal.vertex) && normal_vertex.stage.program != nullptr &&
		                (!ps_active || (static_cast<bool>(normal.pixel) && normal_pixel.stage.program != nullptr));
	}
	PipelineCache::t_find_only = false;
	PipelineCache::g_find_only_users.fetch_sub(1, std::memory_order_relaxed);
	flags.find_only                = false;
	PipelineCache::t_adopt_records = false;
	flags.records                  = false;
	const bool storable      = gate_programs && normal_vertex.buffers_num == 0 && normal_vertex.mesh.threads_num[0] == 0;
	bool       gate_pipeline = storable && pipelines.FindGraphicsPipeline({inputs->color_info, inputs->color_count}, inputs->depth_info,
	                                                                       normal_vertex, buffer, ps_active ? &normal_pixel : nullptr,
	                                                                       topology, restart, normal.vertex, normal.pixel) != nullptr;
	// (The records of the unrecorded permutations met; the normal pipeline, when missing, is made privately below, once the
	// table gates pass. Same inputs as the lookup above: this draw's normal programs and input infos, the capture's targets,
	// this draw's topology and restart.)
	std::shared_ptr<PipelineCache::AdoptedCommit> adopted_commit;
	if (storable && unrecorded_met != 0) adopted_commit = pipelines.TakeAdoptedRecords();
	// (Only where the private build may be made at all (a Fast first build: CreatePrivateGraphicsPipeline).)
	const bool want_private = adopted_commit != nullptr && !gate_pipeline && pipelines.PrivateBuildsAllowed();
	// (An entry on an unrecorded permutation only with its commit's records in hand.)
	const bool unrecorded_held = unrecorded_met != 0 && adopted_commit == nullptr;
	ShaderVertexInputInfo vertex_info;
	ShaderPixelInputInfo  pixel_info;
	ResetNativeVertexInput(vertex_info);
	ResetNativeStageInput(pixel_info);
	bool                                     read_failed = false, pending = false, table_mode = false, other = false;
	const PipelineCache::Pipeline*           pipeline    = stored_pipeline;
	std::shared_ptr<const std::atomic<bool>> compiling;
	// (The compile of its table pipeline met, parked on (null: not), and the stand-in the entry's draws are recorded with.)
	std::shared_ptr<PipelineCache::ParkedPipeline> parked;
	std::unique_ptr<PipelineCache::Pipeline>       parked_stand_in;
	// (An entry on an unrecorded permutation without its records is refused below: nothing made for it. One with its records
	// goes on to the table gates before its private normal pipeline is made.)
	if ((gate_pipeline || want_private) && !unrecorded_held) {
		PipelineCache::t_no_fault_failed = false;
		const bool fa_birth = existing == nullptr && FormatAheadOn(); // (spec_format_ahead: a new pair's birth lookup)
		if (fa_birth) [[unlikely]] PipelineCache::t_fa_birth = true;
		const auto programs = pipelines.GetGraphicsPrograms(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx, buffer.GetUserConfig(),
		                                                    mapping, ps_active, vertex_info, pixel_info, true, portable);
		read_failed        = PipelineCache::t_no_fault_failed;
		pending            = pipelines.TablePending();
		const auto* vertex = vertex_info.stage.program;
		const auto* pixel  = ps_active ? pixel_info.stage.program : nullptr;
		table_mode = !read_failed && !pending && static_cast<bool>(programs.vertex) && vertex != nullptr && vertex->table_mode &&
		             vertex->stage == ShaderType::Vertex && vertex_info.buffers_num == 0 &&
		             (!ps_active || (static_cast<bool>(programs.pixel) && pixel != nullptr && pixel->table_mode));
		if (table_mode && existing != nullptr && (vertex != programs_held[0] || pixel != programs_held[1])) table_mode = false, other = true;
		if (table_mode && pipeline == nullptr)
			pipeline = pipelines.TryCreateGraphicsPipelineWorker({inputs->color_info, inputs->color_count}, inputs->depth_info, vertex_info,
			                                                     buffer, ps_active ? &pixel_info : nullptr, topology, restart,
			                                                     programs.vertex, programs.pixel, true, compiling);
		// (Its table pipeline still compiling: the entry made anyway, parked on that compile (the job held for the commit:
		// PipelineCache::ParkCompiling, which takes the cache's m_mutex, the catalog's read lock dropped here), its draws
		// recorded with a stand-in that has the pipeline's layouts and binding it through a cell (TableParkedBind); the
		// commit's check binds the compiled one (TableEntryResolveParked) or stops the segment.)
		if (table_mode && pipeline == nullptr && compiling != nullptr && TableCellsRecordable()) {
			parked_stand_in = std::make_unique<PipelineCache::Pipeline>();
			parked          = pipelines.ParkCompiling(compiling, *parked_stand_in);
			if (parked == nullptr) parked_stand_in.reset();
		}
		if (existing == nullptr) programs_held = {vertex, pixel}, push_fit_held = PipelineCache::t_push_fit_taken;
		if (fa_birth) [[unlikely]] fa_pair_bits_held = PipelineCache::t_fa_pair_bits;
		// (The normal pipeline made privately only now, the table gates passed: a refused entry makes none. Not made because
		// the cache has the key by now (made meanwhile): found, the records alone committed. A parked entry's too.)
		if (want_private && table_mode && (pipeline != nullptr || parked != nullptr)) {
			gate_pipeline = pipelines.CreatePrivateGraphicsPipeline(*adopted_commit, {inputs->color_info, inputs->color_count},
			                                                        inputs->depth_info, normal_vertex, buffer,
			                                                        ps_active ? &normal_pixel : nullptr, topology, restart,
			                                                        normal.vertex, normal.pixel) ||
			                pipelines.FindGraphicsPipeline({inputs->color_info, inputs->color_count}, inputs->depth_info, normal_vertex,
			                                               buffer, ps_active ? &normal_pixel : nullptr, topology, restart,
			                                               normal.vertex, normal.pixel) != nullptr;
		}
	}
	PipelineCache::t_no_fault_reads = false;
	flags.reads                     = false;
	if (unlocked) {
		CatalogGate(Catalog().m_catalog_writer_waiting, CatalogWriterFirst());
		t_catalog_read->lock();
	}
	// (A Spec::Refuse during the lookups, e.g. SyncGpuCleanBackingSpec's "GPU-written guest memory" (its read then goes on):
	// the packet is refused, and what the lookups read may not be the in-order memory: no entry.)
	if (spec->refused != nullptr) return std::nullopt;
	if (pipeline == nullptr && parked != nullptr) pipeline = parked_stand_in.get(); // (its commit binds the compiled one)
	if (!gate_programs || !storable || !gate_pipeline || unrecorded_held || read_failed || pending || other || !table_mode ||
	    pipeline == nullptr)
		return std::nullopt;
	// (After the relock: the catalog, the pair and the base state as they were.)
	if (CatalogTable() == nullptr || CatalogTable()->epoch != since) return std::nullopt;
	if (existing == nullptr) {
		if (CatalogTable()->pairs.contains(pair_hash)) return std::nullopt;
	} else {
		if (existing->programs != programs_held || existing->refused || existing->respecialize) return std::nullopt;
		const auto v = std::ranges::find(existing->variants, key, &TableVariant::state);
		if (stored_variant != nullptr ? (v == existing->variants.end() || &*v != stored_variant || v->pipeline != pipeline || v->refused)
		                              : (v != existing->variants.end() && (v->refused || v->pipeline != nullptr)))
			return std::nullopt;
	}
	if (const auto again = states.find(base_key);
	    !worker_captured && (again == states.end() || again->second.get() != base || base->capture_id != base_capture ||
	                         base->pipeline_inputs != inputs))
		return std::nullopt;
	if (derive && states.contains(key)) return std::nullopt;
	if (!NativeXprTargetsCurrent(*base)) return std::nullopt;
	// The entry.
	auto  holder             = std::make_shared<TableEntryHolder>();
	auto& h                  = *holder;
	h.key                    = key;
	// (Its programs proven as of: a new pair's lookups, else the pair's proof by this role, TableTry's: TableEntryOverlay.)
	h.generation             = existing != nullptr ? existing->Generation(Role()) : generation;
	h.reuse_checked          = adopted_commit != nullptr; // (its later draws' own lookups checked, above)
	h.inputs                 = inputs;
	h.pair_hash              = pair_hash;
	h.new_pair               = existing == nullptr;
	h.programs               = programs_held;
	h.adopted                = std::move(adopted_commit);
	h.variant.state          = key;
	h.variant.pipeline       = pipeline;
	h.variant.mapping        = mapping;
	h.variant.fetch_embedded = vertex_info.fetch_embedded;
	if (parked != nullptr && pipeline == parked_stand_in.get()) {
		h.parked          = std::move(parked);
		h.parked_stand_in = std::move(parked_stand_in);
	}
	if (derive) {
		auto& made           = *derived;
		made.capture_id      = g_xpr_captures.fetch_add(1, std::memory_order_relaxed) + 1;
		made.meta_epoch      = 0;
		made.targets_epoch   = 0;
		// (A state whose commit makes some of its targets is never marked checked: NativeXprTargetsCurrent's own branch.)
		made.targets_checked = made.virtual_targets != nullptr ? UINT64_MAX : 0;
		made.targets_erased = 0, made.targets_span_begin = 0, made.targets_span_end = 0;
		made.emitted_epoch   = UINT64_MAX;
		made.emitted_serials = UINT64_MAX;
		made.used_frame      = 0;
		made.derived         = true;
		// (Its first rendering without the sibling's clears: those were the sibling draw's own; the rendering has none.)
		made.first_rendering           = made.rendering;
		made.dynamic.indexed_viewports = UsesIndexedViewports(normal_vertex);
		made.dynamic_hash              = XXH3_64bits(&made.dynamic, sizeof(made.dynamic));
		made.resolved = ResolveGraphicsDynamicParams(made.dynamic.buffer, made.dynamic.indexed_viewports, made.dynamic.colors,
		                                             made.dynamic.color_count, made.dynamic.depth);
		h.state       = std::move(derived);
		h.drawn_state = h.state.get();
		// (A capture whose commit makes colour targets: their memory noted for the rest of the translation (in order their
		// images are there from this draw on: TextureCache::t_virtual_ranges); a DCC one's clear, consumed by the commit's
		// acquisition as NativeXprEmitTargets leaves a colour DCC one, asked (a not-cleared answer noted, checked at the
		// commit) and consumed by the speculation, every layer of the view as PrepareDccClear does: its later lookups see the
		// surface as after the draw. The consume is a fact from the next segment on: later lookups of this open segment would
		// still read the cache's "cleared", which none makes: every lookup over the target's memory is refused, and the
		// metadata's memory is checked by the making.)
		if (const auto& virt = h.state->virtual_targets; virt != nullptr) {
			auto& ranges = TextureCache::t_virtual_ranges;
			ranges.Hold(spec->id);
			for (const auto& target: virt->targets) ranges.Note(target.desc.info.data.address, target.desc.info.data.size);
			for (const auto& target: virt->targets) {
				const auto& meta = target.desc.info.metadata;
				const auto& view = target.desc.view_info;
				if (meta.kind != ImageMetadataKind::Dcc) continue;
				for (uint32_t at = view.base_layer; at < std::min(view.base_layer + view.layer_count, 32u); ++at)
					if (m_context.GetTextureCache().IsMetaCleared(meta.range.address, at, nullptr, nullptr, true))
						spec->ConsumeMeta(meta.range.address, at);
			}
		}
	} else {
		h.drawn_state = base;
	}
	h.capture = h.drawn_state->capture_id;
	if (existing == nullptr) {
		auto& pair          = h.node.try_emplace(pair_hash).first->second;
		pair.vs             = vs_address;
		pair.ps             = ps_address;
		pair.start_instance = start_instance;
		pair.programs       = programs_held;
		pair.push_fit       = push_fit_held;
		if (fa_pair_bits_held != 0) [[unlikely]] // (spec_format_ahead: kept as learned bits, as TableStore keeps them)
			pair.portable[0] |= fa_pair_bits_held;
		TableRuns(pair);
		pair.variants.push_back(h.variant);
		pair.Generation(Role()) = generation; // (its programs are those of the code at its addresses as of the lookups)
		h.pair          = &pair;
		h.drawn_variant = &pair.variants.back();
		h.own_variant   = false; // (in the node pair's vector: TableResolveSet finds it there)
	} else {
		h.pair = existing;
		if (stored_variant != nullptr) {
			h.drawn_variant = stored_variant;
		} else {
			h.drawn_variant = &h.variant;
			h.own_variant   = true;
		}
	}
	auto* const owner = &Catalog();
	auto* const raw   = holder.get();
	spec->proposed_sets.push_back({nullptr, mix, raw, spec->closed.size(), false,
	                               [owner, holder](bool commit) { owner->TableEntrySettle(holder, commit); }});
	spec->proposed_sets.back().check = [owner, raw] { return owner->TableEntryCheck(*raw); };
	// (Its segment re-run as a hole drew the draw in order, maybe through the table without a normal lookup: what its commit
	// would do for the draw is not done at a later segment's (records, private pipeline dropped here).)
	if (raw->adopted != nullptr) spec->proposed_sets.back().on_hole = [raw] { raw->adopted.reset(); };
	// (A capture whose commit makes colour targets: made after every check of the segment passed (Speculation::Before's
	// resolve pass); the memory they cover for the completions' write test (resolve_over); its segment run as a hole: none
	// made later (only that segment draws with it), the state never published.)
	if (raw->state != nullptr && raw->state->virtual_targets != nullptr) {
		auto& proposed        = spec->proposed_sets.back();
		proposed.resolve      = [owner, raw, spec] { return owner->TableEntryResolveVirtual(*raw, spec); };
		proposed.resolve_over = [raw](uint64_t address, uint64_t size) {
			const auto* state = raw->state.get();
			if (state == nullptr || state->virtual_targets == nullptr || raw->virtual_hole) return false;
			const auto meets = [&](uint64_t begin, uint64_t bytes) { return bytes != 0 && begin < address + size && address < begin + bytes; };
			for (const auto& made: state->virtual_targets->targets) {
				const auto& info = made.desc.info;
				if (meets(info.data.address, info.data.size) ||
				    (info.HasMetadata() && meets(info.metadata.range.address, info.metadata.range.size)) ||
				    (info.HasStencil() && meets(info.stencil.address, info.stencil.size)))
					return true;
			}
			return false;
		};
		spec->any_resolve = true;
		proposed.on_hole  = [raw, before = std::move(proposed.on_hole)] {
			if (before) before();
			raw->virtual_hole = true;
		};
	}
	// (Its segment re-run as a hole: never in the overlay, and a draw drawn with it through the overlay fails its check (the
	// views a capture made may go with its translation).)
	{
		auto& proposed   = spec->proposed_sets.back();
		proposed.on_hole = [raw, before = std::move(proposed.on_hole)] {
			if (before) before();
			raw->ran_as_hole.store(true, std::memory_order_release);
		};
	}
	// (Its commit's settle: the in-order part at once, the catalog part when the owner's write is had; committed, into the
	// overlay while that part waits: TableEntryOverlayAdd.)
	spec->proposed_sets.back().settle_now = [owner, raw, overlay = std::weak_ptr<TableEntryHolder>(holder)] {
		owner->TableEntrySettleInOrder(*raw);
		if (const auto held = overlay.lock()) owner->TableEntryOverlayAdd(held);
	};
	spec->proposed_sets.back().catalog_owner = owner;
	return draw(h);
}

// (Mode 4) At the entry's segment's Before, the GPU thread (the catalog's owner): the state drawn with still current, the
// pair's programs those it was made with. False: the segment is not committed as recorded (Speculation: images_failed).
bool RenderExecutor::TableEntryCheck(TableEntryHolder& h) {
	// (A capture whose commit makes colour targets, its segment run as a hole: no later segment draws with it (never
	// published); a test failing here would stop a segment for nothing. Parked: its own resolution still runs below.)
	if (h.virtual_hole && h.parked == nullptr) return true;
	if (m_table_xpr == nullptr || m_native_xpr == nullptr || h.drawn_state == nullptr) return false;
	if (!h.new_pair && (h.pair->programs != h.programs || h.pair->refused || h.pair->respecialize)) return false;
	if (h.state == nullptr) {
		const auto& states = NativeXprState().states;
		const auto  again  = states.find(h.key);
		if (again == states.end() || again->second.get() != h.drawn_state || h.drawn_state->capture_id != h.capture) return false;
	}
	if (!NativeXprTargetsCurrent(*h.drawn_state) || !TableAdoptedCurrent(*h.drawn_state)) return false;
	// (Parked on its table pipeline's compile: the compiled pipeline bound in its cells, after the tests above.)
	return h.parked == nullptr || h.parked_resolved || TableEntryResolveParked(h);
}

// (Mode 4) Each colour target a worker's capture took as the in-order FindImage's reuse of an existing image, still that
// reuse (TextureCache::AdoptedTargetCurrent, the GPU thread).
bool RenderExecutor::TableAdoptedCurrent(const NativeXprDrawState& draw_state) {
	for (const auto& pick: draw_state.adopted) {
		const auto at = std::ranges::find_if(draw_state.targets, [&pick](const NativeXprDrawState::Target& t) {
			return t.id == pick.id && t.requested.data.address == pick.address;
		});
		if (at == draw_state.targets.end() || !m_context.GetTextureCache().AdoptedTargetCurrent(at->requested, pick)) return false;
	}
	return true;
}

// (Mode 4) A worker entry whose table pipeline was still compiling, parked on the compile (TableWorkerEntry): the commit's
// check (the GPU thread, the segment's Before) takes, once the compile is done, the pipeline the in-order lookup of its key
// gives (PipelineCache::ResolveParked), writes its handle in every cell of the segments to be spliced (a segment run as a
// hole took its cells with it: never written) and puts it in the variant the settle publishes. EXACT:
// - A pipeline is a pure function of its key: the key the job was posted under is the one this draw's own lookup built
//   (TryCreateGraphicsPipelineWorker named that job), and the pipeline bound is the one the cache holds for it (the job's,
//   moved in as the next lookup of the key would move it). Its layouts are the stand-in's: the same bindings give the same
//   shared layouts (checked again here), so the descriptor set (allocated on the stand-in's set layout), the binds of the
//   set and the push constants (on its pipeline layout) are those the compiled pipeline takes.
// - Everything else recorded for the draw is what is recorded with the pipeline ready: only the handle is late, and
//   nothing reads a passive producer's chunk before its splice. The packet's replay ends the draws' replays' reuse of the
//   pipeline they bound last (InvalidateReplayState), as a recorded bind does.
// - Not resolved (the compile not done at the commit, or the key dropped): the check fails and the segment stops before its
//   splice, as any failed check; the entry is never published with the stand-in (TableEntrySettle). The GPU thread never
//   waits for the compile: the in-order path at such a draw does not wait either (its table store finds the pipeline
//   compiling and draws on the normal path). Nothing is done once a check before this one failed.
bool RenderExecutor::TableEntryResolveParked(TableEntryHolder& h) {
	if (h.parked == nullptr || h.parked_stand_in == nullptr) return false;
	if (Spec::t_checks_failed || !PipelineCache::ParkedDone(*h.parked)) return false;
	auto* const compiled = m_context.GetPipelineCache().ResolveParked(*h.parked);
	if (compiled == nullptr || compiled->pipeline_layout != h.parked_stand_in->pipeline_layout ||
	    compiled->descriptor_set_layout != h.parked_stand_in->descriptor_set_layout ||
	    compiled->native_bindings != h.parked_stand_in->native_bindings)
		return false;
	const auto handle = reinterpret_cast<uint64_t>(static_cast<VkPipeline>(BindPipelineHandle(*compiled))); // NOLINT
	if (handle == 0) return false;
	// (Only the cells of this segment on: those of segments before it are in chunks their holes discarded.)
	for (const auto& [segment, cell]: h.parked_cells)
		if (segment >= Spec::t_check_segment) *cell = handle;
	// (The variant the settle publishes, and the one the draws were drawn with: a new pair's copy in its node.)
	if (h.drawn_variant != nullptr && h.drawn_variant->pipeline == h.parked_stand_in.get()) h.drawn_variant->pipeline = compiled;
	h.variant.pipeline = compiled;
	h.parked_resolved  = true;
	return true;
}

// (Mode 4) A worker's capture that took a colour target no image exists for yet (nothing on any of its pages; no metadata or
// DCC metadata): the worker records the draw with a placeholder view (NativeXprDrawState::Virtual), its rendering's begin
// as a packet whose view is a cell (TableVirtualRendering); the target gets no transition from the worker
// (NativeXprEmitTargets leaves it). The commit (TableEntryResolveVirtual, the GPU thread, the segment's Before, once every
// check of the segment passed: Speculation::Before's resolve pass) makes and acquires it as the in-order draw does
// (FindImage, then FindRenderTarget, whose PrepareDccClear applies a DCC target's pending clear:
// AcquireVirtualTargetAtCommit), moves it to the attachment layout as AcquireRenderTargets does (recorded ahead of the
// splice), writes its view in every cell and puts the image in the state its settle publishes. EXACT:
// - What the draw lacks is the image's handles alone: the pipeline takes the target's format and samples only (from the
//   desc the worker resolved, the one the in-order FindImage is asked with, checked unmoved by it at the commit), the
//   rendering's area and layers come from that desc, and no other command names the image.
// - Its making sits ahead of the segment's replay instead of at the draw. Nothing of the segment before the draw meets it:
//   the worker found nothing on its pages; the segment does not write its memory or its metadata's (declined: the
//   making's upload would read it first); no pending image of the segment is completed after it at the commit (declined);
//   FindImage makes it fresh, with no overlap to resolve and no alias to free (tested again at the commit: nothing on its
//   pages, or the one exact image made since by the GPU thread's own work, which the in-order lookup takes); a reclaim of
//   memory during the making (which could free an image the segment's earlier draws would have touched first) stops the
//   segment. What the commits after see (the image, its first use as a target, its GPU write, the attachment layout, a DCC
//   clear consumed) is what they see after the in-order draw; the worker consumed that clear in its own facts.
// - Nothing of the translation after the draw is translated against the missing image: a read-only lookup over its memory
//   is refused, an image GPU-written and tracked there is answered, a range lookup refuses its packet (SpecVirtualOverlap);
//   a GPU write, buffer or texel read, pending V# or table range, or journaled CPU copy over its memory or its metadata's
//   from the capture's segment on stops the segment here; an open-segment write over it refuses a later draw of the
//   entry; a completion of the segment whose memory it meets stops the segment before Before (resolve_over).
// - A target drawn into again by a later segment after a hole would need the hole's layout: the entry is drawn with in its
//   own segment only, and its memory stays held for the rest of the translation.
// - Not made (any test fails at the commit): the segment stops before its splice, as any failed check; a state not made is
//   never published. The steps of Before after the making that can still stop the segment leave the image made, which the
//   in-order re-run then finds (the same image and bytes).
bool RenderExecutor::TableEntryResolveVirtual(TableEntryHolder& h, const void* spec_state) {
	auto* const state = h.state.get();
	if (state != nullptr && state->virtual_targets == nullptr) return true; // (made already: a Before run again)
	if (h.virtual_hole) return true; // (its segment ran as a hole: nothing to make for a later one; never published)
	// (No draw recorded with it (its first draw refused after the entry was proposed): nothing made, never published.)
	if (state != nullptr && state->virtual_targets->cells.empty()) return true;
	if (state == nullptr || spec_state == nullptr || state->pipeline_inputs == nullptr || !GuestGpu::IsGpuThread()) return false;
	auto& v = *state->virtual_targets;
	// (What the speculation's work did from the capture's segment on over a target's memory, which the in-order path does
	// with the image there (a GPU write invalidating it, a buffer or texel read synchronized from it, a pending V# or table
	// range obtained over it, a CPU copy journaled over it or reading from it), while Before made those ranges ready with no
	// image there: refused. Conservative: every later segment's too, and the capture segment's before the draw.)
	const auto& spec  = *static_cast<const Spec::State*>(spec_state);
	const auto  meets = [&](uint64_t address, uint64_t size) {
		const auto in = [&](const std::vector<Spec::Range>& list, size_t from) {
			for (size_t i = from; i < list.size(); ++i)
				if (list[i].begin < address + size && address < list[i].end) return true;
			return false;
		};
		for (size_t i = v.from_pending_buffers; i < spec.pending_buffers.size(); ++i)
			if (const auto& entry = spec.pending_buffers[i]; entry.address < address + size && address < entry.address + entry.size)
				return true;
		return in(spec.current, v.from_current) || in(spec.written, v.from_written) || in(spec.invalidated, v.from_invalidated) ||
		       in(spec.texel_syncs, v.from_texel_syncs) || in(spec.buffer_ranges, v.from_buffer_ranges) ||
		       in(spec.cpu_copies, v.from_cpu_copies) || in(spec.table_ranges, v.from_table_ranges) ||
		       spec.journal_set.Intersects(address, size) || spec.open_journal_set.Intersects(address, size);
	};
	for (const auto& made: v.targets) {
		if (meets(made.desc.info.data.address, made.desc.info.data.size)) return false;
		if (const auto& meta = made.desc.info.metadata; meta.kind != ImageMetadataKind::None && meets(meta.range.address, meta.range.size))
			return false;
	}
	auto&                      textures = m_context.GetTextureCache();
	auto                       inputs   = std::make_shared<NativeXprDrawState::PipelineInputs>(*state->pipeline_inputs);
	std::vector<vk::ImageView> views(v.targets.size());
	for (size_t j = 0; j < v.targets.size(); ++j) {
		const auto&   made = v.targets[j];
		auto          desc = made.desc;
		vk::ImageView view;
		uint64_t      serial = 0;
		const auto    id     = textures.AcquireVirtualTargetAtCommit(desc, view, serial);
		if (!id) return false;
		auto& image = textures.GetImage(id);
		if (image.backing.samples != desc.info.samples) return false;
		// (AcquireRenderTargets' transition of a colour target the draw does not sample (a fresh image is bound as no
		// texture of it): attachment-optimal, read and write, over the view's subresources, ahead of the splice.)
		const auto& vi = desc.view_info;
		image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
		              vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
		              ImageSubresourceRange {vi.base_level, vi.level_count, vi.base_layer, vi.layer_count},
		              m_context.GetCommandScheduler().Current().Handle());
		auto& target     = state->targets[made.target];
		target.id        = id;
		target.serial    = serial;
		target.requested = desc.info;
		state->rendering.color_attachments[made.attachment].image_view       = view;
		state->first_rendering.color_attachments[made.attachment].image_view = view;
		inputs->color_info[made.attachment].image_id = id;
		inputs->color_info[made.attachment].desc     = desc;
		views[j]                                     = view;
	}
	for (const auto& [j, cell]: v.cells) *cell = reinterpret_cast<uint64_t>(static_cast<VkImageView>(views[j])); // NOLINT
	state->target_ids.clear();
	for (const auto& target: state->targets) state->target_ids.push_back(target.id);
	state->pipeline_inputs = inputs;
	if (h.inputs != nullptr) h.inputs = inputs;
	state->targets_checked = 0;
	state->virtual_targets.reset(); // (made: a state as any capture's from here on)
	return true;
}

#ifdef KYTY_LOCAL_VULKAN_RECORDING
namespace TableVirtual {
// A rendering's begin as CommandBuffer::BeginRendering makes it, its views cells (a no-image target's: written by the commit).
struct Attachment {
	uint64_t view   = 0;
	int32_t  layout = 0, load = 0, store = 0;
	uint32_t clear[4] {};
};
struct Packet {
	VkCommandBuffer command = VK_NULL_HANDLE;
	int32_t         x = 0, y = 0;
	uint32_t        width = 0, height = 0, layers = 0, colors = 0, has_depth = 0, has_stencil = 0;
	Attachment      color[RENDER_COLOR_ATTACHMENTS_MAX] {};
	Attachment      depth {}, stencil {};
};
static_assert(std::is_standard_layout_v<Packet> && std::is_trivially_copyable_v<Packet>);
static Attachment Pack(const vk::RenderingAttachmentInfo& in) {
	Attachment out;
	out.view   = reinterpret_cast<uint64_t>(static_cast<VkImageView>(in.imageView)); // NOLINT
	out.layout = static_cast<int32_t>(in.imageLayout);
	out.load   = static_cast<int32_t>(in.loadOp);
	out.store  = static_cast<int32_t>(in.storeOp);
	static_assert(sizeof(out.clear) == sizeof(VkClearValue));
	std::memcpy(out.clear, &in.clearValue, sizeof(out.clear));
	return out;
}
static VkRenderingAttachmentInfo Unpack(const Attachment& in) {
	VkRenderingAttachmentInfo out {};
	out.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	out.imageView   = reinterpret_cast<VkImageView>(in.view); // NOLINT
	out.imageLayout = static_cast<VkImageLayout>(in.layout);
	out.resolveMode = VK_RESOLVE_MODE_NONE;
	out.loadOp      = static_cast<VkAttachmentLoadOp>(in.load);
	out.storeOp     = static_cast<VkAttachmentStoreOp>(in.store);
	std::memcpy(&out.clearValue, in.clear, sizeof(out.clearValue));
	return out;
}
static void Replay(std::span<const LocalVulkanRecording::Segment> segments, const vk::detail::DispatchLoaderDynamic& dispatch) {
	Packet packet;
	std::memcpy(&packet, segments[0].data, sizeof(packet));
	std::array<VkRenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < packet.colors; ++i) colors[i] = Unpack(packet.color[i]);
	const auto      depth = Unpack(packet.depth), stencil = Unpack(packet.stencil);
	VkRenderingInfo info {};
	info.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
	info.renderArea.offset    = {packet.x, packet.y};
	info.renderArea.extent    = {packet.width, packet.height};
	info.layerCount           = packet.layers;
	info.colorAttachmentCount = packet.colors;
	info.pColorAttachments    = colors.data();
	info.pDepthAttachment     = packet.has_depth != 0 ? &depth : nullptr;
	info.pStencilAttachment   = packet.has_stencil != 0 ? &stencil : nullptr;
	const auto begin = dispatch.vkCmdBeginRendering != nullptr ? dispatch.vkCmdBeginRendering : dispatch.vkCmdBeginRenderingKHR;
	begin(LocalVulkanRecording::Resolve(packet.command), &info);
}
} // namespace TableVirtual
#endif

// (CommandBuffer::RenderingHook, set around a no-image capture's BeginRendering, TableDraw.) True: recorded here, as a
// packet whose views are cells its commit writes. A begin that names a placeholder is never recorded raw: the translation
// is refused instead.
bool RenderExecutor::TableVirtualRendering(void* data, vk::CommandBuffer command, const vk::RenderingInfo& rendering) {
	auto&                                             v = *static_cast<NativeXprDrawState::Virtual*>(data);
	std::array<int32_t, RENDER_COLOR_ATTACHMENTS_MAX> which {};
	which.fill(-1);
	bool any = false;
	for (uint32_t i = 0; i < std::min<uint32_t>(rendering.colorAttachmentCount, RENDER_COLOR_ATTACHMENTS_MAX); ++i)
		for (uint32_t j = 0; j < v.targets.size(); ++j)
			if (rendering.pColorAttachments[i].imageView == v.Placeholder(j)) which[i] = static_cast<int32_t>(j), any = true;
	if (!any) return false;
	using Refusal = CommandScheduler::RecorderRefusal;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	if (!LocalVulkanRecording::Passive()) throw Refusal {"no-image target: not a passive recording"};
	if (rendering.pNext != nullptr || rendering.flags != vk::RenderingFlags {} || rendering.viewMask != 0 ||
	    rendering.colorAttachmentCount > RENDER_COLOR_ATTACHMENTS_MAX)
		throw Refusal {"no-image target: a rendering the packet does not hold"};
	TableVirtual::Packet packet {};
	packet.command = static_cast<VkCommandBuffer>(command);
	packet.x       = rendering.renderArea.offset.x;
	packet.y       = rendering.renderArea.offset.y;
	packet.width   = rendering.renderArea.extent.width;
	packet.height  = rendering.renderArea.extent.height;
	packet.layers  = rendering.layerCount;
	packet.colors  = rendering.colorAttachmentCount;
	const auto plain = [](const vk::RenderingAttachmentInfo& a) {
		return a.pNext == nullptr && a.resolveMode == vk::ResolveModeFlagBits::eNone;
	};
	for (uint32_t i = 0; i < packet.colors; ++i) {
		if (!plain(rendering.pColorAttachments[i])) throw Refusal {"no-image target: a rendering the packet does not hold"};
		packet.color[i] = TableVirtual::Pack(rendering.pColorAttachments[i]);
	}
	if (rendering.pDepthAttachment != nullptr) {
		if (!plain(*rendering.pDepthAttachment)) throw Refusal {"no-image target: a rendering the packet does not hold"};
		packet.has_depth = 1, packet.depth = TableVirtual::Pack(*rendering.pDepthAttachment);
	}
	if (rendering.pStencilAttachment != nullptr) {
		if (!plain(*rendering.pStencilAttachment)) throw Refusal {"no-image target: a rendering the packet does not hold"};
		packet.has_stencil = 1, packet.stencil = TableVirtual::Pack(*rendering.pStencilAttachment);
	}
	const LocalVulkanRecording::Segment segment {&packet, sizeof(packet)};
	void*                               copy = nullptr;
	if (!LocalVulkanRecording::EnqueuePacketCopy(TableVirtual::Replay, segment, &copy) || copy == nullptr)
		throw Refusal {"no-image target: rendering not recorded"};
	auto* stored = static_cast<TableVirtual::Packet*>(copy);
	for (uint32_t i = 0; i < packet.colors; ++i)
		if (which[i] >= 0) v.cells.emplace_back(static_cast<uint32_t>(which[i]), &stored->color[i].view);
	return true;
#else
	(void)command;
	throw Refusal {"no-image target: no recording"};
#endif
}

// (Mode 4) A worker's draw whose pair and draw state key have no entry in the catalog (TableWorkerEntry, before it makes
// one), or whose catalog pair has no programs yet (a store compiling: TableTry), is drawn with an entry a worker made that
// is not in the catalog yet, where there is one (nullopt: none, or refused: the path as before):
//  - its own translation's entry for a new pair, though the catalog has a pair for the key since (the GPU thread's): the
//    reuse in TableWorkerEntry looks for an entry of that pair, not this one;
//  - an entry of an earlier translation whose segment committed (Speculation::After, its in-order part: settle_now, where
//    TableEntryOverlayAdd puts it in the overlay) while its catalog part waits (the settle queue); out of the overlay once
//    that part begins (TableEntrySettle: overlay_settled).
// Exact: an entry is a pure function of the pair's key (vs, ps, start instance), the draw state key and the programs the code
// at those addresses translates to: the table programs and pipeline its lookups found, the variant made of them, the key's
// base draw state (the catalog's capture) or the state derived for it. A draw drawn with it is what the in-order path draws
// for that key once the entry, or the GPU thread's store of the same key, is in the catalog. What the catalog path tests is
// tested here, under the catalog's read lock TableTry holds across the lookup and the draw (an entry's catalog part takes
// the write: it cannot run meanwhile): the code at the addresses still its programs' (hashed again when the shader map's
// generation moved since the entry's proof), its pair not refused, respecialized or given other programs, the catalog's
// own pair or variant for the key not refused, its variant (or the pair's stored one, with the same pipeline), its base
// state still the catalog's same capture, its state's targets current, the entry not poisoned and not run as a hole. Only
// a committed entry is in the overlay: a proposing segment cancelled or never committed never puts it there, and one run as
// a hole is never added; a poisoned, parked, no-image-target or recorded-permutation entry is not added. The draw's own
// commit checks the entry again (TableEntryOverlayCheck, at the segment's Before, the GPU thread: the same conditions, a
// poisoning since included): a check that fails stops the segment and the draw is made in order as before. Published or kept
// aside, the objects drawn with stay where they are (a new pair's node moved whole into the catalog's map, the derived
// state's unique_ptr moved, the variant found again by key), and none is freed while a speculation is live
// (TableSweepPairs, TableRetireEntryOrphans); the holder is held by the draw's proposal (a settle that does nothing). An
// earlier translation's committed entry is drawn with a poison flag of its own: a draw of this uncommitted translation
// does not change what that commit publishes (the draw is refused either way). The GPU thread waits for no worker here: its
// add only tries the overlay's lock.
std::optional<RenderExecutor::TableResult> RenderExecutor::TableEntryOverlay(CommandBuffer& buffer, const TablePair* existing,
                                                                            std::span<const uint64_t> commands, uint64_t index_base,
                                                                            uint64_t index_bytes, vk::IndexType index_type,
                                                                            const NativeXprDirectDraw* direct) {
	auto* const spec = Spec::Current();
	if (spec == nullptr || spec->refused != nullptr || Role() == 0 || GuestGpu::IsGpuThread() || CatalogTable() == nullptr ||
	    Catalog().m_native_xpr == nullptr || m_native_xpr == nullptr || t_catalog_read == nullptr || !t_catalog_read->owns_lock())
		return std::nullopt;
	const auto& sh             = buffer.GetShaders();
	const auto  vs_address     = sh.GetVs().es_regs.data_addr, ps_address = sh.GetPs().ps_regs.data_addr;
	const auto  start_instance = sh.GetVs().start_instance_user_sgpr;
	const auto  key            = m_native_xpr->state_key;
	const auto& states         = Catalog().NativeXprState().states;
	// (What the catalog path tests, read under the catalog's read lock; the variant drawn with. False: refused.)
	const auto usable = [&](TableEntryHolder& h, TableVariant*& variant) {
		if (h.poisoned.load(std::memory_order_relaxed) || h.reuse_checked || h.ran_as_hole.load(std::memory_order_acquire))
			return false;
		// (The catalog's own pair for the key, refused or its variant for the key refused: the in-order path refuses.)
		if (existing != nullptr) {
			if (existing->refused) return false;
			const auto refused = std::ranges::find(existing->variants, key, &TableVariant::state);
			if (refused != existing->variants.end() && refused->refused) return false;
		}
		if (h.parked != nullptr || h.pair == nullptr || h.drawn_state == nullptr || h.programs[0] == nullptr || h.key != key)
			return false;
		auto& pair = *h.pair;
		if (pair.vs != vs_address || pair.ps != ps_address || pair.start_instance != start_instance) return false;
		if (pair.programs != h.programs || pair.refused || pair.respecialize) return false;
		if (ShaderMapGeneration() != h.generation &&
		    (ShaderHashNoFaultScope {true}, ShaderHashAt(vs_address) != h.programs[0]->shader_hash ||
		                                        (h.programs[1] != nullptr && ShaderHashAt(ps_address) != h.programs[1]->shader_hash)))
			return false;
		if (h.state == nullptr) {
			const auto again = states.find(key);
			if (again == states.end() || again->second.get() != h.drawn_state || h.drawn_state->capture_id != h.capture)
				return false;
		}
		const auto v = std::ranges::find(pair.variants, key, &TableVariant::state);
		if (h.own_variant) {
			if (v != pair.variants.end() && v->refused) return false;
			variant = &h.variant;
		} else {
			if (v == pair.variants.end() || v->refused || v->pipeline != h.variant.pipeline) return false;
			variant = &*v;
		}
		return NativeXprTargetsCurrent(*h.drawn_state);
	};
	const auto draw = [&](TableEntryHolder& h, TableVariant& variant, std::atomic<bool>* poison) {
		struct EntryScope { // (cleared on every way out, a refusal thrown through TableDraw too)
			EntryScope(const void* v, std::atomic<bool>* p) noexcept { t_entry_variant = v, t_entry_poison = p; }
			~EntryScope() { t_entry_variant = nullptr, t_entry_poison = nullptr; }
			EntryScope(const EntryScope&)            = delete;
			EntryScope& operator=(const EntryScope&) = delete;
		} scope(h.own_variant ? &variant : nullptr, poison);
		const auto result = TableDraw(buffer, *h.pair, variant, *h.drawn_state, commands, index_base, index_bytes, index_type, direct);
		OwnTable().last_pair = nullptr; // (TableContinue reads the role's last variant out of the pair's vector)
		return result;
	};
	// (The draw's commit checks the entry (TableEntryOverlayCheck), once a segment; the proposal holds an earlier
	// translation's holder while this translation is (its own's is held by its own proposal), its settle nothing.)
	const auto note_check = [&](TableEntryHolder& h, std::shared_ptr<TableEntryHolder> held, TableResult result) {
		if (result == TableResult::Native || Spec::Current() != spec || spec->refused != nullptr) return;
		auto* const raw     = &h;
		const auto  segment = spec->closed.size();
		for (const auto& p: spec->proposed_sets)
			if (p.pair == nullptr && p.hash == 0 && p.set == raw && p.segment == segment) return;
		auto* const owner = &Catalog();
		spec->proposed_sets.push_back({nullptr, 0, raw, segment, false, [held = std::move(held)](bool) { (void)held; }});
		spec->proposed_sets.back().check = [owner, raw] { return owner->TableEntryOverlayCheck(*raw); };
	};
	// (Its own translation's entry for a new pair, the catalog's pair for the key made since: proposed with the mix
	// TableWorkerEntry gives a new pair's.)
	if (existing != nullptr) {
		const std::array<uint64_t, 3> pair_key {vs_address, ps_address, static_cast<uint32_t>(start_instance)};
		const auto                    content = XXH3_64bits(pair_key.data(), sizeof(pair_key));
		const uint64_t                mix     = (content * 0x9e3779b97f4a7c15ull) ^ key;
		for (const auto& p: spec->proposed_sets) {
			if (p.pair != nullptr || p.hash != mix) continue;
			auto& h = *static_cast<TableEntryHolder*>(p.set);
			if (!h.new_pair || h.key != key || h.pair_hash != content) continue;
			// (A capture whose commit makes colour targets: as the reuse, only in the segment that made it.)
			if (h.state != nullptr && h.state->virtual_targets != nullptr && h.state->virtual_targets->segment != spec->closed.size())
				continue;
			TableVariant* variant = nullptr;
			if (!usable(h, variant)) return std::nullopt;
			const auto result = draw(h, *variant, &h.poisoned);
			note_check(h, nullptr, result);
			return result;
		}
	}
	// (An earlier translation's committed entry whose catalog part waits.)
	if (g_entry_overlay_n.load(std::memory_order_relaxed) == 0) return std::nullopt;
	std::shared_ptr<TableEntryHolder> later;
	{
		const std::lock_guard lock(g_entry_overlay_mutex);
		for (const auto& entry: g_entry_overlay) {
			const auto& h = *entry;
			if (h.key != key || h.overlay_settled.load(std::memory_order_acquire)) continue;
			if (h.pair->vs != vs_address || h.pair->ps != ps_address || h.pair->start_instance != start_instance) continue;
			later = entry;
			break;
		}
	}
	if (later == nullptr) return std::nullopt;
	TableVariant* variant = nullptr;
	if (!usable(*later, variant)) return std::nullopt;
	std::atomic<bool> poison_aside {false};
	const auto        result = draw(*later, *variant, &poison_aside);
	note_check(*later, later, result);
	return result;
}

// (Mode 4) At an entry's segment's commit, its in-order part done (settle_now, the GPU thread): into the overlay, unless its
// catalog part began already; the settled ones trimmed (released after the lock).
void RenderExecutor::TableEntryOverlayAdd(const std::shared_ptr<TableEntryHolder>& holder) {
	auto& h = *holder;
	if (h.overlay_added || h.overlay_settled.load(std::memory_order_relaxed)) return;
	if (h.poisoned.load(std::memory_order_relaxed) || h.parked != nullptr || h.reuse_checked || h.virtual_hole ||
	    h.ran_as_hole.load(std::memory_order_relaxed) || h.drawn_state == nullptr ||
	    (h.state != nullptr && h.state->virtual_targets != nullptr))
		return;
	std::vector<std::shared_ptr<TableEntryHolder>> released;
	{
		std::unique_lock<std::mutex> lock(g_entry_overlay_mutex, std::try_to_lock);
		if (!lock.owns_lock()) return;
		for (auto it = g_entry_overlay.begin(); it != g_entry_overlay.end();) {
			if ((*it)->overlay_settled.load(std::memory_order_relaxed) || g_entry_overlay.size() >= 256) {
				released.push_back(std::move(*it));
				it = g_entry_overlay.erase(it);
			} else {
				++it;
			}
		}
		g_entry_overlay.push_back(holder);
		g_entry_overlay_n.store(static_cast<uint32_t>(g_entry_overlay.size()), std::memory_order_relaxed);
	}
	h.overlay_added = true;
}

// (Mode 4) At the segment's Before (the GPU thread, the catalog's owner) of a draw drawn with an entry of the overlay: what
// TableEntryOverlay tested, again (a poisoning or a run as a hole since included). False: the segment is not committed as
// recorded.
bool RenderExecutor::TableEntryOverlayCheck(TableEntryHolder& h) {
	if (m_table_xpr == nullptr || m_native_xpr == nullptr || h.pair == nullptr || h.drawn_state == nullptr || h.programs[0] == nullptr)
		return false;
	if (h.poisoned.load(std::memory_order_relaxed) || h.ran_as_hole.load(std::memory_order_acquire)) return false;
	const auto& pair = *h.pair;
	if (pair.programs != h.programs || pair.refused || pair.respecialize) return false;
	// (The catalog's pair for the key, refused or its variant for the key refused: the in-order path refuses. A new pair's
	// entry: the catalog's pair for its hash, when another; else its own pair, above and below.)
	const TablePair* catalog_pair = h.pair;
	if (h.new_pair)
		if (const auto found = m_table_xpr->pairs.find(h.pair_hash); found != m_table_xpr->pairs.end()) catalog_pair = &found->second;
	if (catalog_pair->refused) return false;
	if (const auto v = std::ranges::find(catalog_pair->variants, h.key, &TableVariant::state);
	    v != catalog_pair->variants.end() && v->refused)
		return false;
	if (ShaderMapGeneration() != h.generation &&
	    (ShaderHashAt(pair.vs) != h.programs[0]->shader_hash ||
	     (h.programs[1] != nullptr && ShaderHashAt(pair.ps) != h.programs[1]->shader_hash)))
		return false;
	if (h.state == nullptr) {
		const auto& states = NativeXprState().states;
		const auto  again  = states.find(h.key);
		if (again == states.end() || again->second.get() != h.drawn_state || h.drawn_state->capture_id != h.capture) return false;
	}
	if (!h.own_variant) {
		const auto v = std::ranges::find(pair.variants, h.key, &TableVariant::state);
		if (v == pair.variants.end() || v->refused || v->pipeline != h.variant.pipeline) return false;
	}
	return NativeXprTargetsCurrent(*h.drawn_state) && TableAdoptedCurrent(*h.drawn_state);
}

// (Mode 4) The commit's settle never waits for the workers' reads of the catalog: a proposal's catalog part runs under its
// owner's write had with a try-lock; else it waits in g_settle_pending (the GPU thread's alone, in order: a part waits
// while one before it does). The list runs at the next commit's settle while its tries succeed (SettleDrain(false)), and
// under a write that waits (SettleDrain(true): the writer-first gate raised, as any write) at each guest flip
// (CommandProcessor::Flip), before a sweep of the catalog (TableSweepPairs, which may free the pairs a waiting part
// names), before a live mode 0, and after a settle that must not wait past its commit. A part runs with its owner's write
// held for it (t_settle_held): its own CatalogWrite calls take nothing more. What a waiting part holds (the holder, the
// set) its closure holds. Late publishing is safe by the checks each part makes under its write: a new pair only while
// the catalog has none for its key, a variant only into a pair whose programs are the entry's and that is not refused or
// respecialized, a state only while the key has none, an image set only while the catalog's epoch is the one it was built
// under and its slot empty; else kept, as before.
struct SettlePendingPart {
	const RenderExecutor* owner = nullptr;
	std::function<void()> work;
};
static std::vector<SettlePendingPart> g_settle_pending; // (the GPU thread's; a short list, taken from its front)

void RenderExecutor::SettleOrQueue(const void* owner_opaque, std::function<void()> work) {
	const auto* owner = static_cast<const RenderExecutor*>(owner_opaque);
	if (owner == nullptr) return work();
	if (g_settle_pending.empty()) {
		std::unique_lock<std::shared_mutex> lock(owner->m_catalog_mutex, std::try_to_lock);
		if (lock.owns_lock()) {
			t_settle_held = &owner->m_catalog_mutex;
			struct Release {
				~Release() { t_settle_held = nullptr; }
			} release;
			work();
			return;
		}
	}
	g_settle_pending.push_back({owner, std::move(work)});
	s_settle_pending_n.store(static_cast<uint32_t>(g_settle_pending.size()), std::memory_order_relaxed);
}

void RenderExecutor::SettleDrain(bool blocking) {
	while (!g_settle_pending.empty()) {
		const auto* owner = g_settle_pending.front().owner;
		std::unique_lock<std::shared_mutex> lock;
		if (blocking) {
			lock = owner->CatalogWrite(); // (outside mode 4 an empty lock: no reader then)
		} else {
			lock = std::unique_lock<std::shared_mutex>(owner->m_catalog_mutex, std::try_to_lock);
			if (!lock.owns_lock()) return;
		}
		t_settle_held = &owner->m_catalog_mutex;
		struct Release {
			~Release() { t_settle_held = nullptr; }
		} release;
		// (Every waiting part of this owner at the front, in order, under the one write.)
		while (!g_settle_pending.empty() && g_settle_pending.front().owner == owner) {
			auto part = std::move(g_settle_pending.front());
			g_settle_pending.erase(g_settle_pending.begin());
			s_settle_pending_n.store(static_cast<uint32_t>(g_settle_pending.size()), std::memory_order_relaxed);
			part.work();
		}
	}
}

void RenderExecutor::SettleClear() {
	// (The parts dropped below never settle their entries: none stays in the overlay.)
	if (g_entry_overlay_n.load(std::memory_order_relaxed) != 0) {
		std::vector<std::shared_ptr<TableEntryHolder>> released;
		{
			const std::lock_guard lock(g_entry_overlay_mutex);
			released.swap(g_entry_overlay);
			g_entry_overlay_n.store(0, std::memory_order_relaxed);
		}
	}
	if (g_settle_pending.empty()) return;
	g_settle_pending.clear();
	s_settle_pending_n.store(0, std::memory_order_relaxed);
}

static void TableEntryOrphan(const std::shared_ptr<RenderExecutor::TableEntryHolder>& holder) {
	const std::lock_guard lock(g_entry_orphans_mutex);
	g_entry_orphans.push_back(holder);
	g_entry_orphans_n.store(static_cast<uint32_t>(g_entry_orphans.size()), std::memory_order_relaxed);
}

// (Mode 4) What an entry's commit does in order, before its catalog part (Speculation::After runs it at the commit, the
// catalog part when the owner's write is had: SettleOrQueue): what the in-order draw would have done for this draw, which
// the commit spares it, in stream order: the records, then the normal pipeline and its recipe (published whole or not:
// the draw is committed; the cache's m_mutex only, no other lock held); then, unless poisoned, the metadata notes of a
// derived state's targets (as NativeXprCaptureState notes its own). Once each.
void RenderExecutor::TableEntrySettleInOrder(TableEntryHolder& h) {
	if (h.adopted != nullptr) {
		m_context.GetPipelineCache().CommitAdopted(*h.adopted);
		h.adopted.reset();
	}
	if (h.meta_noted || h.poisoned.load(std::memory_order_relaxed)) return;
	h.meta_noted = true;
	if (h.state != nullptr) {
		auto& textures = m_context.GetTextureCache();
		for (const auto& target: h.state->targets) textures.NoteMetaUse(target.metadata, target.id.index);
	}
}

// (Mode 4) At the entry's segment's commit (`commit`; false: never committed), the GPU thread, under the owner's write:
// published as TableStore stores, while the catalog has nothing there; what is not published whole is kept
// (TableEntryOrphan).
void RenderExecutor::TableEntrySettle(const std::shared_ptr<TableEntryHolder>& holder, bool commit) {
	auto& h = *holder;
	// (Out of the overlay from here: TableEntryOverlay; the objects a worker drew with stay where they are.)
	h.overlay_settled.store(true, std::memory_order_release);
	// (Never committed: its records and private pipeline go (destroyed here, on the GPU thread).)
	if (!commit) return h.adopted.reset(), TableEntryOrphan(holder);
	TableEntrySettleInOrder(h); // (done at the commit already when the rest waited: SettleOrQueue)
	if (h.poisoned.load(std::memory_order_relaxed)) return TableEntryOrphan(holder);
	// (Never published with a parked entry's stand-in (only once its commit's check bound the compiled pipeline), nor with a
	// state whose colour targets its commit did not make.)
	if (h.parked != nullptr && !h.parked_resolved) return TableEntryOrphan(holder);
	if (h.state != nullptr && h.state->virtual_targets != nullptr) return TableEntryOrphan(holder);
	bool whole = true;
	{
		const auto write = CatalogWrite();
		auto*      table = m_table_xpr.get();
		auto&      cache = NativeXprState();
		const auto frame = m_context.FrameNumber();
		if (table == nullptr) {
			whole = false;
		} else if (h.new_pair) {
			if (!table->pairs.contains(h.pair_hash)) {
				auto node                = h.node.extract(h.pair_hash);
				node.mapped().used_frame = frame;
				auto inserted            = table->pairs.insert(std::move(node));
				TableScoutPublish(inserted.position->second);
			} else {
				whole = false;
			}
		} else if (h.own_variant) {
			auto& pair = *h.pair;
			if (pair.programs != h.programs || pair.refused || pair.respecialize) {
				whole = false;
			} else {
				auto stored = std::ranges::find(pair.variants, h.key, &TableVariant::state);
				if (stored == pair.variants.end()) {
					pair.variants.push_back(h.variant);
				} else if (!stored->refused && stored->pipeline == nullptr) {
					stored->pipeline       = h.variant.pipeline;
					stored->mapping        = h.variant.mapping;
					stored->fetch_embedded = h.variant.fetch_embedded;
				} else if (stored->refused || stored->pipeline != h.variant.pipeline) {
					whole = false;
				}
			}
		}
		if (h.state != nullptr) {
			if (whole && !cache.states.contains(h.key)) {
				auto& slot = cache.states[h.key];
				slot       = std::move(h.state);
				slot->Use(frame);
			} else {
				whole = false;
			}
		}
	}
	if (!whole) TableEntryOrphan(holder);
}

// (Mode 4) The kept holders go once no speculation translates or has a result unfinished (TableSweepPairs, the catalog's
// owner): a kept node's sets retired as a swept pair's; the executors' and the sets' pointers into a kept state dropped as
// when states leave (TableForgetStates).
void RenderExecutor::TableRetireEntryOrphans() {
	if (Role() != 0 || m_table_xpr == nullptr) return;
	std::vector<std::shared_ptr<TableEntryHolder>> list;
	{
		const std::lock_guard lock(g_entry_orphans_mutex);
		list.swap(g_entry_orphans);
		g_entry_orphans_n.store(0, std::memory_order_relaxed);
	}
	if (list.empty()) return;
	const auto write = CatalogWrite();
	if (kyty_local_speculations_live.load(std::memory_order_seq_cst) != 0) {
		const std::lock_guard lock(g_entry_orphans_mutex);
		g_entry_orphans.insert(g_entry_orphans.end(), list.begin(), list.end());
		g_entry_orphans_n.store(static_cast<uint32_t>(g_entry_orphans.size()), std::memory_order_relaxed);
		return;
	}
	for (auto& holder: list)
		for (auto& [hash, pair]: holder->node)
			for (auto& [words, set]: pair.sets) NativeXprRetireSet(set->pool, set->set);
	list.clear();
	auto& cache          = NativeXprState();
	cache.emit           = {};
	cache.last_state     = nullptr;
	cache.last_state_key = 0;
	TableForgetStates();
}

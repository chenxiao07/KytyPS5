#pragma once
// KYTY_LANE_LOCAL_SPILL=2: the verify mode of R1 (lane-private LDS spill in the lane-local proof).
//
// A compute program the lane-local proof accepted only because of its lane-private LDS spill slots
// (IR::Program::lane_private_lds) is translated twice in table mode (PipelineCache::ProgramCache::Get's table job): the
// R1 module, which the dispatch uses as in mode 1, and the fallback module, exactly as KYTY_LANE_LOCAL_SPILL=0
// translates it (CompileOptions::lane_private_lds_off: two lanes per invocation). For the first
// KYTY_LANE_LOCAL_SPILL_VERIFY_N (default 64) table dispatches of each such program the renderer runs both into the same
// targets and compares their written words bitwise (lane-spill-verify.inc, TableDispatch), printing
//   LaneLocal verify <hash>: dispatches=N compared_words=M mismatches=K ...
// KYTY_LANE_LOCAL_SPILL_VERIFY_HASH=<hex> restricts the second translation and the compare to one program.
// Modes 0 and 1 never reach any of it.

#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace LaneSpillVerify {

// The mode is 2 (the translator reads the same variable: Frontend::LaneLocalSpillMode).
inline bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_LANE_LOCAL_SPILL");
		return value != nullptr && std::string_view(value) == "2";
	}();
	return on;
}

// The program `hash` gets the fallback translation and the compare (KYTY_LANE_LOCAL_SPILL_VERIFY_HASH: that one only).
//   if (LaneSpillVerify::Wanted(options.shader_hash)) { /* translate the fallback too */ }
inline bool Wanted(uint64_t hash) {
	static const uint64_t only = [] {
		const char* value = std::getenv("KYTY_LANE_LOCAL_SPILL_VERIFY_HASH");
		if (value == nullptr || *value == '\0') return uint64_t {0};
		return static_cast<uint64_t>(std::strtoull(value, nullptr, 16));
	}();
	return Enabled() && (only == 0 || only == hash);
}

// Compared dispatches per program (KYTY_LANE_LOCAL_SPILL_VERIFY_N, default 64); later ones run R1 alone, as mode 1.
inline uint32_t Limit() {
	static const uint32_t n = [] {
		const char* value = std::getenv("KYTY_LANE_LOCAL_SPILL_VERIFY_N");
		return value != nullptr && *value != '\0' ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 64u;
	}();
	return n;
}

// Bytes one compared dispatch may copy per image set (KYTY_LANE_LOCAL_SPILL_VERIFY_MB, default 256): its written
// subresources, three times (saved, the fallback's, R1's), in host-visible memory.
inline uint64_t SlotBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_LANE_LOCAL_SPILL_VERIFY_MB");
		const uint64_t mb = value != nullptr && *value != '\0' ? std::strtoull(value, nullptr, 10) : 256u;
		return mb << 20u;
	}();
	return bytes;
}

} // namespace LaneSpillVerify

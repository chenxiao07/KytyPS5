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
// KYTY_LANE_LOCAL_SPILL_POISON=1 starts each run from a poison word instead of the restored targets (Poison()).
// Modes 0 and 1 never reach any of it.

#include <cstdint>
#include <cstdio>
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

// The word every compared 32-bit word of the targets holds before each run in poison mode (a NaN with a payload as a
// float, an unlikely count as an integer; per word, whatever the format, so 8- and 16-bit texels see its bytes).
constexpr uint32_t PoisonWord = 0x7fbadbadu;

// KYTY_LANE_LOCAL_SPILL_POISON=1 (read once; unset, empty or any other value: 0, the targets restored between the runs):
// the compared subresources are filled with PoisonWord before the fallback's run and again before R1's, so a word a run
// left equal to PoisonWord is a word it did not write ("wrote the same" and "wrote nothing" told apart). A program that
// reads its output reads the poison in both runs (fair to the compare), and R1's output, which stays, is then wrong for
// that frame: diagnostic only. Prints its engage line once, on the first call with mode 2 on.
//   if (LaneSpillVerify::Poison()) { /* fill the targets with PoisonWord before each run */ }
inline bool Poison() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_LANE_LOCAL_SPILL_POISON");
		const bool  set   = Enabled() && value != nullptr && std::string_view(value) == "1";
		if (set) {
			std::printf("LaneLocal verify: poison mode (KYTY_LANE_LOCAL_SPILL_POISON=1): each run starts from %08x in every "
			            "compared word; a program that reads its output reads the poison, so its frame output is wrong "
			            "(diagnostic only)\n",
			            PoisonWord);
			std::fflush(stdout);
		}
		return set;
	}();
	return on;
}

} // namespace LaneSpillVerify

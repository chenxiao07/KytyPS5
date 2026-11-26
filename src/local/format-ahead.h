#ifndef EMULATOR_SRC_LOCAL_FORMAT_AHEAD_H_
#define EMULATOR_SRC_LOCAL_FORMAT_AHEAD_H_
// spec_format_ahead (live KYTY_SPEC_FORMAT_AHEAD; mode-toggle row {1, 0}: on under KYTY_SPECULATE=4; it acts under
// KYTY_SPECULATE=4 only, FormatAheadOn): a table pair whose recorded draws met a vertex buffer in more than one format is
// born with that buffer decoding its V#'s format at run time (IR::PortableFormats), so a V# in another recorded format no
// longer refuses on a worker ("table: another format") nor makes the GPU thread learn the buffer and reset the pair at
// its next draw (TableReset).
// - The mask is the recorded PAIR's, from two sources (ProgramCache::FoldRecordedPairs):
//   - this PC's warmup recording as loaded at boot: the buffers the vertex records of that (VS, PS) pair's recipes differ
//     in the format or dst_sel of (the fold the static precompile's portable forms are made of, SeedVarying);
//   - the hints the release ships for this game version (shader-hints.h: recorded play without the game's code), OR'd in:
//     the same, between values with a format only (HintVarying: their canonical records and null V#s show no change).
// - Made once a run. A mode-4 boot with hints: at the warmup's load on the main thread, published by the GPU thread as it
//   applies mode 4 (PipelineCache::ApplyFormatAhead), with a recording or none. No hints: as mode 4 is applied with the
//   switch on (a boot, a live `mode 4`), on the GPU thread while the warmup still holds the recording, else on a thread
//   of its own from the recording's file; that thread also where a live `mode 4` has the hints still to read. Published
//   with release / acquire. Turned on later: no masks until mode 4 is applied again, every pair born as before.
// - It is taken by the pair's birth lookup only (PipelineCache::t_fa_birth: TableStore's first programs, a worker's new
//   pair) and kept in the pair's own `portable` as learned bits, so its later lookups and its learns see it as learned,
//   and a reset keeps it. A pair whose recorded draws never varied is born specialised, and so is a pair in neither
//   source (as before); a buffer neither foresaw is learned as before.
// - The mode-4 ahead pool makes the forms of the recording's pairs from the same masks (AheadPair::fa_mask), and a pair
//   whose form another seed's job makes gets its pipelines after that job (TableJob::after_more).
// Pixel stages and compute are unchanged. In mode 0 nothing of it runs: the masks are not computed, no lookup is a birth
// lookup, the ahead pool is mode 4's.
#include "mode-toggle.h"

#include <atomic>
#include <cstdint>

inline std::atomic<int64_t>& FormatAheadSwitch() {
	static auto& on = LiveSwitch::Find("spec_format_ahead", ModeToggle::Default("spec_format_ahead"));
	return on;
}
// (The mode first: in mode 0 one relaxed load.)
[[nodiscard]] inline bool FormatAheadOn() {
	return ::kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4 &&
	       FormatAheadSwitch().load(std::memory_order_relaxed) != 0;
}

#endif /* EMULATOR_SRC_LOCAL_FORMAT_AHEAD_H_ */

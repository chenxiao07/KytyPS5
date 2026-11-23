#ifndef KYTY_LAUNCHER_LINUX_WEAVE_H
#define KYTY_LAUNCHER_LINUX_WEAVE_H

#include "mode4.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

// The Experimental page's weave (KYTY_WEAVE=1), launcher-windows' Weave.cs: off by default, and offered only where the
// emulator would not stand it down. The gate is the emulator's own, applied as it applies it at its start
// (src/local/weave.cpp ApplyStartGates), to the variables as the game will see them. It sits beside parallel translation
// (mode4.h): the emulator runs one of the two at a time, mode 4 first, so the page turns the other off when one is turned
// on, and Play sets the weave only when it sets no mode 4. With the switch off the launcher sets nothing, and the
// emulator runs as without it.
namespace Weave {

// The memory the weave takes beyond mode 0, in MB, shown next to the switch (docs/PORTABLE-README.md's KYTY_WEAVE=1 row
// says the same): the weave's private memory over mode 0's, same settings, at the same tour stops (12 stops, the six
// worlds twice: 548 MB).
inline constexpr const char* ExtraMemoryMb = "550";
// The performance cores' logical CPUs it needs without KYTY_WEAVE_WORKERS: 8 for the GPU thread's core, the recording
// and upload workers, the reader and the game's threads, then 2 workers at least.
inline constexpr uint32_t CpusNeeded = 10;

enum class Gate { Available, Mode4, FewCores, Needs };

// What the gate found: whether it is offered, the logical CPUs it counted (FewCores), and the switch the extra options
// turn off (Needs).
struct Verdict {
	Gate        gate = Gate::Available;
	uint32_t    cpus = 0;
	std::string needs;
};

// A switch's value as the emulator reads it (whole digits, strtoul), or none: not set, or not a number (the emulator
// does not start then).
template <typename Variable>
[[nodiscard]] std::optional<unsigned long> Value(const Variable& variable, const char* name) {
	const std::string text = variable(name);
	if (text.empty()) return std::nullopt;
	char*      end   = nullptr;
	const auto value = std::strtoul(text.c_str(), &end, 10);
	if (*end != '\0') return std::nullopt;
	return value;
}

// The gate as the emulator applies it at its start. `variable(name)`: the variable's value as the game will see it ("":
// not set); mode4: mode 4's gate on the same variables; performance_cpus and logical_cpus: what the launcher counted
// (0: not known).
// - Mode 4: KYTY_SPECULATE=4 where mode 4 would run (its own gate first, as there) stands the weave down. The page's own
//   mode 4 switch is not this: the two switches turn each other off, and Play sets one.
// - CPU: KYTY_WEAVE_WORKERS 1 to 8 lifts it; else the performance cores' logical CPUs (every logical CPU where the
//   classes are not known), less 8, are the workers, and under 2 it stands down. (CoreCounts: no KYTY_SIMULATE_PCORES.)
// - KYTY_SCOUT, KYTY_VULKAN_RECORDING, KYTY_TABLE_XPR or KYTY_NATIVE_XPR at 0: stood down.
template <typename Variable>
[[nodiscard]] Verdict Check(const Mode4::Verdict& mode4, uint32_t performance_cpus, uint32_t logical_cpus,
                            const Variable& variable) {
	Verdict verdict;
	if (Value(variable, "KYTY_SPECULATE") == 4ul && mode4.gate == Mode4::Gate::Available) {
		verdict.gate = Gate::Mode4;
		return verdict;
	}
	if (const auto workers = Value(variable, "KYTY_WEAVE_WORKERS"); !workers || *workers == 0) {
		verdict.cpus = performance_cpus != 0 ? performance_cpus : logical_cpus;
		if (verdict.cpus < CpusNeeded) {
			verdict.gate = Gate::FewCores;
			return verdict;
		}
	}
	for (const char* name: {"KYTY_SCOUT", "KYTY_VULKAN_RECORDING", "KYTY_TABLE_XPR", "KYTY_NATIVE_XPR"})
		if (Value(variable, name) == 0ul) {
			verdict.gate  = Gate::Needs;
			verdict.needs = name;
			return verdict;
		}
	return verdict;
}

// KYTY_WEAVE=1 is set: only when it is chosen and offered, and mode 4 is not set.
[[nodiscard]] inline bool On(bool chosen, const Verdict& verdict, bool mode4_on) {
	return chosen && verdict.gate == Gate::Available && !mode4_on;
}

} // namespace Weave

#endif

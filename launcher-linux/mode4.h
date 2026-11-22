#ifndef KYTY_LAUNCHER_LINUX_MODE4_H
#define KYTY_LAUNCHER_LINUX_MODE4_H

#include <cstdint>
#include <cstdlib>
#include <string>

// The Experimental page's parallel translation (KYTY_SPECULATE=4, mode 4), launcher-windows' Mode4.cs: off by default,
// and offered only where the emulator would not stand it down. The gate is the emulator's own, applied as it applies it
// at its start (src/local/speculation-ram-gate.h, src/local/performance-switches.h), to the variables as the game will see
// them. With the switch off the launcher sets nothing, and the emulator runs as without it.
namespace Mode4 {

// The memory mode 4 takes beyond mode 0, in GB, shown next to the switch (docs/PORTABLE-README.md's KYTY_SPECULATE=4 row
// says the same): mode 4's private memory over mode 0's, same settings, at the same tour stops, with the speculation
// states kept without the constant RAM (2.19 GB).
inline constexpr const char* ExtraMemoryGb = "2.2";
// KYTY_SPECULATE_RAM_GB's default (speculation-ram-gate.h), and KYTY_SPECULATE_THREADS' (speculation.cpp).
inline constexpr uint32_t DefaultRamGib = 28, DefaultWorkers = 4;

enum class Gate { Available, LowRam, FewCores };

// What the gate found: whether it is offered, the memory it needs (GiB), and the logical CPUs on the performance cores it
// needs and found (the reasons' numbers).
struct Verdict {
	Gate     gate        = Gate::Available;
	uint32_t ram_gib     = DefaultRamGib;
	uint32_t cpus_needed = 3 + DefaultWorkers + 2;
	uint32_t cpus        = 0;
};

// The gate as the emulator applies it at its start. `variable(name)`: the variable's value as the game will see it ("":
// not set); ram_bytes: the memory the launcher measured (0: not read, as the emulator takes a failed query). The
// emulator's Linux build counts no performance cores (0: not checked) unless KYTY_SIMULATE_PCORES gives a count.
// - Memory (SpeculationRamGate): KYTY_SIMULATE_RAM_MB (MiB) stands for it; KYTY_SPECULATE_RAM_GB is the threshold in GiB
//   when it is whole digits up to 4096 (else 28), 0 lifting it; under it (0 memory too), stood down.
// - CPU: KYTY_SPECULATE_CPUS set lifts it; else 3 + KYTY_SPECULATE_THREADS (1 to 12, else 4) + 2 logical CPUs on the
//   performance cores; a count of 0 is not checked.
template <typename Variable>
[[nodiscard]] Verdict Check(uint64_t ram_bytes, const Variable& variable) {
	Verdict           verdict;
	const std::string simulated_ram = variable("KYTY_SIMULATE_RAM_MB");
	const uint64_t    ram = simulated_ram.empty() ? ram_bytes : std::strtoull(simulated_ram.c_str(), nullptr, 10) << 20u;
	if (const std::string threshold = variable("KYTY_SPECULATE_RAM_GB"); !threshold.empty()) {
		char*      end    = nullptr;
		const auto parsed = std::strtoul(threshold.c_str(), &end, 10);
		verdict.ram_gib   = *end == '\0' && parsed <= 4096 ? static_cast<uint32_t>(parsed) : DefaultRamGib;
	}
	if (verdict.ram_gib != 0 && ram < (uint64_t {verdict.ram_gib} << 30u)) {
		verdict.gate = Gate::LowRam;
		return verdict;
	}
	uint32_t workers = DefaultWorkers;
	if (const std::string text = variable("KYTY_SPECULATE_THREADS"); !text.empty()) {
		char*      end   = nullptr;
		const auto value = std::strtoul(text.c_str(), &end, 10);
		if (*end == '\0' && value >= 1 && value <= 12) workers = static_cast<uint32_t>(value); // (else it does not start)
	}
	verdict.cpus_needed = 3 + workers + 2;
	if (const std::string simulated = variable("KYTY_SIMULATE_PCORES"); !simulated.empty())
		verdict.cpus = static_cast<uint32_t>(std::strtoul(simulated.c_str(), nullptr, 10));
	if (variable("KYTY_SPECULATE_CPUS").empty() && verdict.cpus != 0 && verdict.cpus < verdict.cpus_needed)
		verdict.gate = Gate::FewCores;
	return verdict;
}

// KYTY_SPECULATE=4 is set: only when it is chosen and offered.
[[nodiscard]] inline bool On(bool chosen, const Verdict& verdict) {
	return chosen && verdict.gate == Gate::Available;
}

} // namespace Mode4

#endif

#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace CpuShape {

struct Core {
	uint64_t cpus;
	uint32_t efficiency;
};

inline constexpr uint32_t DefaultWorkers = 4;
inline constexpr uint32_t BaselinePacketSlots = 12;
inline constexpr uint32_t PacketSlots = BaselinePacketSlots + 1;
inline constexpr uint32_t RenderThreads = 3;
inline constexpr uint32_t CopyThreadsMax = 7;

constexpr bool HasPass(uint32_t workers, bool enabled) {
	return enabled && workers + 1 < BaselinePacketSlots;
}

constexpr bool StandsDown(uint32_t performance_logical, uint32_t workers) {
	return performance_logical != 0 && performance_logical < RenderThreads + workers + 2;
}

constexpr uint32_t FixedGraphicsWorkers(uint32_t requested, bool pass) {
	return std::min(requested, BaselinePacketSlots - (pass ? 1u : 0u) - 2u);
}

struct SlotRange {
	uint32_t base, workers;
};

struct Lanes {
	SlotRange q1, q49;
};

constexpr Lanes FixedLanes(uint32_t base, bool pass, uint32_t requested) {
	// Keep q1's packet slots reserved even if its workers fail to start.
	const uint32_t q1 = std::min(requested, BaselinePacketSlots - base - 1u);
	// Only eight q0 workers plus the pass and two requested per lane give q49 the 13th slot.
	const uint32_t slots = pass && base == 9 && requested == 2 ? PacketSlots : BaselinePacketSlots;
	const uint32_t q49 = std::min(requested, slots - base - q1);
	return {{base, q1}, {base + q1, q49}};
}

constexpr SlotRange LegacyAsyncSlots(uint32_t base, uint32_t requested) {
	return {base, std::min(requested, BaselinePacketSlots - std::min(base, BaselinePacketSlots))};
}

constexpr bool HasSixPCorePeer(uint32_t cores) {
	// Six P-cores and at least one group of four E-cores.
	return cores >= 10 && (cores - 6) % 4 == 0;
}

inline bool AllCoresHaveSmt2(const std::vector<Core>& cores, uint64_t affinity) {
	uint64_t two_threads = 0;
	uint32_t efficiency = 0;
	for (const auto& core: cores) {
		const auto cpus = core.cpus & affinity;
		if (cpus != 0) {
			if (two_threads == 0) {
				efficiency = core.efficiency;
			}
			if (std::popcount(cpus) != 2 || core.efficiency != efficiency) {
				return false;
			}
			two_threads |= cpus;
		}
	}
	return affinity != 0 && two_threads == affinity;
}

constexpr uint32_t CopyBudget(uint32_t cores, uint32_t workers, uint32_t list_cores, bool pass,
                              bool match_hybrid = false) {
	if (cores == 0) {
		return CopyThreadsMax;
	}
	auto worker_cores = list_cores == 0 ? workers : std::min(workers, list_cores);
	if (match_hybrid && HasSixPCorePeer(cores)) {
		// The default six-P-core worker list covers three physical cores. Give its alike peer the same copy budget.
		worker_cores = std::min(worker_cores, 3u);
	}
	const auto reserved = RenderThreads + worker_cores + (pass ? 1u : 0u);
	return cores > reserved ? std::min(cores - reserved, CopyThreadsMax) : 0u;
}

struct Plan {
	uint64_t workers_cpus = 0, gpu_cpus = 0, record_cpus = 0, copy_cpus = 0, pass_cpu = 0;
	uint32_t cores = 0, performance_cores = 0, performance_logical = 0;
	uint32_t workers = DefaultWorkers, copies = 0;
	bool pass = false, stands_down = false, derived_workers = false;
	const char* reason = "default";
};

// Use the existing affinity lists for sizing. A default worker list allows a count increase when the budget fits.
inline Plan Choose(const std::vector<Core>& cores, uint64_t affinity, uint32_t requested_workers, bool named_workers,
                   uint64_t worker_cpus, bool worker_override, uint64_t render, uint64_t record, uint64_t render_core,
                   bool pass_enabled, bool derive, std::optional<uint32_t> copy_override = {},
                   bool gpu_apart = true) {
	Plan plan;
	const auto effective = [affinity](uint64_t mask) { return (mask & affinity) != 0 ? mask & affinity : affinity; };
	plan.gpu_cpus = effective(render_core != 0 ? render_core : render);
	plan.record_cpus = effective(record);
	plan.copy_cpus = effective(~(render_core != 0 ? render_core : render));
	uint32_t fastest = 0;
	for (const auto& core: cores) {
		if ((core.cpus & affinity) != 0) {
			fastest = std::max(fastest, core.efficiency);
		}
	}
	plan.workers = requested_workers != 0 ? requested_workers : DefaultWorkers;
	plan.workers_cpus = worker_cpus & affinity;
	uint32_t list_cores = 0, single = 0, dual = 0, lowest = fastest;
	uint32_t gpu_cores = 0, record_cores = 0, render_cores = 0;
	uint32_t copy_only = 0, shared_single = 0, shared_dual = 0;
	uint32_t gpu_without_copy = 0, record_without_copy = 0, render_without_copy = 0;
	uint64_t worker_cores = 0, slowest = 0, copy_only_cpus = 0;
	for (const auto& core: cores) {
		const auto cpus = core.cpus & affinity;
		if (cpus == 0) {
			continue;
		}
		++plan.cores;
		if (core.efficiency < lowest) {
			lowest = core.efficiency;
			slowest = 0;
		}
		if (core.efficiency == lowest) {
			slowest |= cpus;
		}
		const auto eligible = std::popcount(cpus & plan.workers_cpus);
		const bool copier = (cpus & plan.copy_cpus) != 0;
		if (copier && eligible == 0) {
			++copy_only;
			copy_only_cpus |= cpus;
		}
		if (eligible != 0) {
			++list_cores;
			worker_cores |= cpus;
			single += eligible == 1;
			dual += eligible >= 2;
			shared_single += copier && eligible == 1;
			shared_dual += copier && eligible >= 2;
		}
		if (core.efficiency == fastest) {
			++plan.performance_cores;
			plan.performance_logical += std::popcount(cpus);
			if (eligible == 0) {
				const bool gpu = (cpus & plan.gpu_cpus) != 0;
				const bool record = (cpus & plan.record_cpus) != 0;
				gpu_cores += gpu;
				record_cores += record;
				render_cores += gpu || record;
				if (!copier) {
					gpu_without_copy += gpu;
					record_without_copy += record;
					render_without_copy += gpu || record;
				}
			}
		}
	}
	plan.stands_down = !named_workers && StandsDown(plan.performance_logical, plan.workers);
	if (plan.stands_down) {
		plan.reason = "mode 4 stands down";
		plan.workers = 0;
		plan.workers_cpus = 0;
		return plan;
	}
	if (gpu_apart && gpu_cores >= 2) {
		plan.gpu_cpus &= ~worker_cores; // Default spec_gpu_affinity, once the GPU reaches a flip.
	}
	plan.pass = HasPass(plan.workers, pass_enabled);
	if (plan.pass && lowest != fastest) {
		const auto outside = slowest & ~plan.workers_cpus;
		const auto eligible = outside != 0 ? outside : slowest;
		plan.pass_cpu = eligible & (uint64_t {0} - eligible);
	}
	const bool automatic = requested_workers == 0 && !named_workers && !worker_override;
	plan.copies = copy_override.value_or(CopyBudget(plan.cores, plan.workers, list_cores, plan.pass,
	                                               derive && automatic && AllCoresHaveSmt2(cores, affinity)));
	if (!derive || !automatic || plan.cores == 0) {
		return plan;
	}
	if (plan.copies == 0) {
		plan.reason = "no copy-thread budget";
		return plan;
	}
	if (gpu_cores == 0 || record_cores < 2 || render_cores < RenderThreads ||
	    (plan.gpu_cpus & worker_cores) != 0) {
		plan.reason = "render masks cannot reserve three cores outside q0";
		return plan;
	}
	const auto reserved = RenderThreads + (plan.pass ? 1u : 0u) + plan.copies;
	const auto spare = plan.cores > reserved ? plan.cores - reserved : 0u;
	// Place copiers outside q0 wherever possible, leaving room there for GPU, Record, upload and a pinned pass.
	const auto render_on_copy = std::max({1u - std::min(1u, gpu_without_copy),
	                                     2u - std::min(2u, record_without_copy),
	                                     RenderThreads - std::min(RenderThreads, render_without_copy)});
	const auto outside_reserved = render_on_copy + ((plan.pass_cpu & copy_only_cpus) != 0 ? 1u : 0u);
	const auto outside_copiers = copy_only - std::min(copy_only, outside_reserved);
	const auto shared_copiers = plan.copies - std::min(plan.copies, outside_copiers);
	const auto used_single = std::min(shared_copiers, shared_single);
	const auto used_dual = std::min(shared_copiers - used_single, shared_dual);
	const auto spare_dual = std::min(spare, dual - used_dual);
	const auto limit = std::min(8u, 2 * spare_dual + std::min(spare - spare_dual, single - used_single));
	for (uint32_t workers = limit; workers > DefaultWorkers; --workers) {
		if (!StandsDown(plan.performance_logical, workers)) {
			plan.workers = workers;
			plan.derived_workers = true;
			plan.reason = "spare cores and worker mask";
			return plan;
		}
	}
	plan.reason = "spare cores or worker mask cannot raise q0";
	return plan;
}

std::string CpuList(uint64_t mask);
Plan BootPlan(uint32_t requested_workers, bool pass_enabled, bool pipe_enabled);
uint32_t PipeCopyBudget(uint32_t workers, uint64_t mask, bool pass, bool automatic);
std::optional<uint32_t> PipeOverride();

} // namespace CpuShape

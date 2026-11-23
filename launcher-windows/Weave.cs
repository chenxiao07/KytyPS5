// SPDX-License-Identifier: GPL-2.0-or-later
using System;

namespace KytyLauncher {

// The Experimental page's weave (KYTY_WEAVE=1): off by default, and offered only where the emulator would not stand it
// down. The gate is the emulator's own, applied as it applies it at its start (src/local/weave.cpp ApplyStartGates), to
// the variables as the game will see them. It sits beside parallel translation (Mode4.cs): the emulator runs one of the
// two at a time, mode 4 first, so the page turns the other off when one is turned on, and Play sends the weave only
// when it sends no mode 4. With the switch off the launcher sends nothing, and the emulator runs as without it.
static class Weave {
	// The memory the weave takes beyond mode 0, in MB, shown next to the switch (docs/PORTABLE-README.md's KYTY_WEAVE=1
	// row says the same): the weave's private memory over mode 0's, same settings, at the same tour stops (12 stops,
// the six worlds twice: 548 MB).
	public const string ExtraMemoryMb = "550";
	// The performance cores' logical CPUs it needs without KYTY_WEAVE_WORKERS: 8 for the GPU thread's core, the recording
	// and upload workers, the reader and the game's threads, then 2 workers at least.
	public const uint CpusNeeded = 10;

	public enum Gate { Available, Mode4, FewCores, Needs }

	// What the gate found: whether it is offered, the logical CPUs it counted (FewCores), and the switch the Command-line
	// options turn off (Needs).
	public sealed class Verdict {
		public Gate Gate;
		public uint Cpus;
		public string Needs;
	}

	// The switches it needs on (ApplyStartGates' scout and table path checks), in the order it checks them.
	static readonly string[] Needed = { "KYTY_SCOUT", "KYTY_VULKAN_RECORDING", "KYTY_TABLE_XPR", "KYTY_NATIVE_XPR" };

	// A switch's value as the emulator reads it (whole digits, strtoul), or null: not set, or not a number (the emulator
	// does not start then).
	static ulong? Value(Func<string, string> variable, string name) {
		var text = variable(name);
		if (string.IsNullOrEmpty(text)) return null;
		bool whole;
		var value = Mode4.Parse(text, uint.MaxValue, out whole);
		return whole ? value : (ulong?)null;
	}

	// The gate as the emulator applies it at its start. `variable`: each variable's value as the game will see it (null or
	// empty: not set); mode4: mode 4's gate on the same variables; performanceCpus and logicalCpus: what the launcher
	// counted (0: not known).
	// - Mode 4: KYTY_SPECULATE=4 where mode 4 would run (its own gate first, as there) stands the weave down. The page's own
	//   mode 4 switch is not this: the two switches turn each other off, and Play sends one.
	// - CPU: KYTY_WEAVE_WORKERS 1 to 8 lifts it; else the performance cores' logical CPUs (every logical CPU where the
	//   classes are not known), less 8, are the workers, and under 2 it stands down. (CoreCounts: no KYTY_SIMULATE_PCORES.)
	// - KYTY_SCOUT, KYTY_VULKAN_RECORDING, KYTY_TABLE_XPR or KYTY_NATIVE_XPR at 0: stood down.
	public static Verdict Check(Mode4.Verdict mode4, uint performanceCpus, uint logicalCpus, Func<string, string> variable) {
		var verdict = new Verdict();
		if (Value(variable, "KYTY_SPECULATE") == 4 && mode4.Gate == Mode4.Gate.Available) {
			verdict.Gate = Gate.Mode4;
			return verdict;
		}
		var workers = Value(variable, "KYTY_WEAVE_WORKERS");
		if (workers == null || workers == 0) {
			verdict.Cpus = performanceCpus != 0 ? performanceCpus : logicalCpus;
			if (verdict.Cpus < CpusNeeded) {
				verdict.Gate = Gate.FewCores;
				return verdict;
			}
		}
		foreach (var name in Needed)
			if (Value(variable, name) == 0) {
				verdict.Gate  = Gate.Needs;
				verdict.Needs = name;
				return verdict;
			}
		return verdict;
	}

	// The text under the switch: what it does, or why it is not offered here.
	public static string Description(Verdict verdict) {
		switch (verdict.Gate) {
			case Gate.Mode4: return Strings.T("weave.mode4");
			case Gate.FewCores: return Strings.F("weave.fewCores", CpusNeeded, verdict.Cpus);
			case Gate.Needs: return Strings.F("weave.needs", verdict.Needs);
			default: return Strings.F("weave.desc", ExtraMemoryMb);
		}
	}

	// Play's -Set pair for it, or null: only when it is on and offered, and mode 4 is not sent.
	public static string Set(bool on, Verdict verdict, bool mode4On) {
		return on && verdict.Gate == Gate.Available && !mode4On ? "KYTY_WEAVE=1" : null;
	}
}

}

// SPDX-License-Identifier: GPL-2.0-or-later
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace KytyLauncher {

// The Experimental page's parallel translation (KYTY_SPECULATE=4, mode 4): off by default, and offered only where the
// emulator would not stand it down. The gate is the emulator's own, applied as it applies it at its start
// (src/local/speculation-ram-gate.h, src/local/performance-switches.h), to the variables as the game will see them. With
// the switch off the launcher sends nothing, and the emulator runs as without it.
static class Mode4 {
	// The memory mode 4 takes beyond mode 0, in GB, shown next to the switch (docs/PORTABLE-README.md's KYTY_SPECULATE=4
	// row says the same): mode 4's private memory over mode 0's, same settings, at the same tour stops, with the
	// speculation states kept without the constant RAM (2.19 GB).
	public const string ExtraMemoryGb = "2.2";
	// KYTY_SPECULATE_RAM_GB's default (speculation-ram-gate.h), and KYTY_SPECULATE_THREADS' (speculation.cpp).
	public const uint DefaultRamGib = 28, DefaultWorkers = 4;

	public enum Gate { Available, LowRam, FewCores }

	// A variable's value from -Set options as run-windows.ps1 applies them: each piece split at its first "=", an empty
	// value or a piece without "=" giving "0" (for a variable neither the config nor the script sets, as the gate's are),
	// the last piece naming it winning. Null: none names it.
	public static string FromSetOptions(IEnumerable<string> options, string name) {
		string found = null;
		foreach (var option in options)
			foreach (var piece in Regex.Split(option, ",(?=[A-Za-z_][A-Za-z0-9_]*=)")) {
				var at  = piece.IndexOf('=');
				var key = at < 0 ? piece : piece.Substring(0, at);
				if (!string.Equals(key, name, StringComparison.OrdinalIgnoreCase)) continue;
				var value = at < 0 ? "" : piece.Substring(at + 1);
				found = value.Length != 0 ? value : "0";
			}
		return found;
	}

	// What the gate found: whether it is offered, the memory it needs (GiB), and the logical CPUs on the performance
	// cores it needs and found (the reasons' numbers).
	public sealed class Verdict {
		public Gate Gate;
		public uint RamGib = DefaultRamGib, CpusNeeded = 3 + DefaultWorkers + 2, Cpus;
	}

	// The C library's reading of `text` as strtoull (`max` ulong.MaxValue) or Windows' strtoul (`max` uint.MaxValue)
	// reads it: leading white space, a sign, the digits that follow (none: 0), `max` past the range, a minus negated as
	// unsigned. `whole`: digits and nothing after them.
	public static ulong Parse(string text, ulong max, out bool whole) {
		int at = 0;
		while (at < text.Length && (text[at] == ' ' || (text[at] >= '\t' && text[at] <= '\r'))) ++at;
		bool minus = false;
		if (at < text.Length && (text[at] == '+' || text[at] == '-')) minus = text[at++] == '-';
		int   first = at;
		ulong value = 0;
		bool  over  = false;
		for (; at < text.Length && text[at] >= '0' && text[at] <= '9'; ++at) {
			var digit = (ulong)(text[at] - '0');
			if (over || value > (max - digit) / 10) over = true;
			else value = value * 10 + digit;
		}
		whole = at > first && at == text.Length;
		if (at == first) return 0;
		if (over) return max;
		return minus ? unchecked(0UL - value) & max : value;
	}

	// The gate as the emulator applies it at its start. `variable`: each variable's value as the game will see it (null
	// or empty: not set); ramBytes and performanceCpus: what the launcher measured (0: not read, as the emulator takes a
	// failed query).
	// - Memory (SpeculationRamGate): KYTY_SIMULATE_RAM_MB (MiB, strtoull) stands for it; KYTY_SPECULATE_RAM_GB is the
	//   threshold in GiB when it is whole digits up to 4096 (else 28), 0 lifting it; under it (0 memory too), stood down.
	// - CPU: KYTY_SPECULATE_CPUS set lifts it; else 3 + KYTY_SPECULATE_THREADS (1 to 12, else 4) + 2 logical CPUs on the
	//   performance cores, KYTY_SIMULATE_PCORES (strtoul) standing for the count; a count of 0 (not known) is not checked.
	public static Verdict Check(ulong ramBytes, uint performanceCpus, Func<string, string> variable) {
		var  verdict = new Verdict();
		bool whole;
		var  simulatedRam = variable("KYTY_SIMULATE_RAM_MB");
		var  ram          = string.IsNullOrEmpty(simulatedRam) ? ramBytes : Parse(simulatedRam, ulong.MaxValue, out whole) << 20;
		var  threshold    = variable("KYTY_SPECULATE_RAM_GB");
		if (!string.IsNullOrEmpty(threshold)) {
			var gib = Parse(threshold, uint.MaxValue, out whole);
			verdict.RamGib = whole && gib <= 4096 ? (uint)gib : DefaultRamGib;
		}
		if (verdict.RamGib != 0 && ram < ((ulong)verdict.RamGib << 30)) {
			verdict.Gate = Gate.LowRam;
			return verdict;
		}
		var workers     = DefaultWorkers;
		var workersText = variable("KYTY_SPECULATE_THREADS");
		if (!string.IsNullOrEmpty(workersText)) {
			var value = Parse(workersText, uint.MaxValue, out whole);
			if (whole && value >= 1 && value <= 12) workers = (uint)value; // (another value: the emulator does not start)
		}
		verdict.CpusNeeded = 3 + workers + 2;
		var simulatedCpus  = variable("KYTY_SIMULATE_PCORES");
		verdict.Cpus       = string.IsNullOrEmpty(simulatedCpus) ? performanceCpus
		                                                         : (uint)Parse(simulatedCpus, uint.MaxValue, out whole);
		var named          = variable("KYTY_SPECULATE_CPUS");
		verdict.Gate       = string.IsNullOrEmpty(named) && verdict.Cpus != 0 && verdict.Cpus < verdict.CpusNeeded
		                         ? Gate.FewCores : Gate.Available;
		return verdict;
	}

	// The text under the switch: what it does, or why it is not offered here.
	public static string Description(Verdict verdict) {
		switch (verdict.Gate) {
			case Gate.LowRam: return Strings.F("mode4.lowRam", verdict.RamGib);
			case Gate.FewCores: return Strings.F("mode4.fewCores", verdict.CpusNeeded, verdict.Cpus);
			default: return Strings.F("mode4.desc", ExtraMemoryGb);
		}
	}

	// Play's -Set pair for it, or null: only when it is on and offered.
	public static string Set(bool on, Verdict verdict) {
		return on && verdict.Gate == Gate.Available ? "KYTY_SPECULATE=4" : null;
	}
}

}

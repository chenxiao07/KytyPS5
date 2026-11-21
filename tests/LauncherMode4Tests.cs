// SPDX-License-Identifier: GPL-2.0-or-later
// The Windows launcher's parallel translation switch (launcher-windows/Mode4.cs): off by default, kept in
// launcher-settings.json, offered only where the emulator would not stand mode 4 down (its own gate, with its tuning and
// simulation variables), KYTY_SPECULATE=4 in Play's command only when on and offered (a Command-line options' -Set pair
// winning), its memory figure stated, its text in English and Chinese. The repository's folder is the argument.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace KytyLauncher {
static class LauncherMode4Tests {
	const ulong GiB = 1UL << 30;
	static int checks;
	static void Check(bool value, string message) {
		++checks;
		if (!value) throw new Exception(message);
	}
	static Mode4.Verdict Gate(ulong ram, uint cpus, params string[] pairs) {
		var variables = new Dictionary<string, string>();
		for (int i = 0; i + 1 < pairs.Length; i += 2) variables[pairs[i]] = pairs[i + 1];
		return Mode4.Check(ram, cpus, name => variables.ContainsKey(name) ? variables[name] : null);
	}
	// Play's -Set pairs (Commands.PlayCommand) on a PC with this memory and these performance cores.
	static List<string> PlaySets(string root, bool on, string extra, ulong ram, int cpus) {
		var settings = new Settings(Path.Combine(root, "play-" + Guid.NewGuid().ToString("N") + ".json"));
		settings.Set("mode4", on);
		settings.Set("extra_args", extra);
		var command = new Commands(new LauncherContext(root, settings, ram, cpus)).PlayCommand(null, true);
		var parts   = command.Split(' ');
		var at      = Array.IndexOf(parts, "-Set");
		return at < 0 || at + 1 >= parts.Length ? new List<string>() : parts[at + 1].Split(',').ToList();
	}
	static int Main(string[] args) {
		var root = Path.Combine(Path.GetTempPath(), "kyty-mode4-test-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(root);
		try {
			foreach (var name in new[] { "KYTY_SIMULATE_RAM_MB", "KYTY_SIMULATE_PCORES", "KYTY_SPECULATE_RAM_GB",
			                             "KYTY_SPECULATE_THREADS", "KYTY_SPECULATE_CPUS" })
				Environment.SetEnvironmentVariable(name, null);

			// Off by default, in a new settings file and in one written before the switch existed.
			var path = Path.Combine(root, "launcher-settings.json");
			Check(!new Settings(path).Bool("mode4"), "mode4 on by default");
			File.WriteAllText(path, "{\"fps30\": true}");
			Check(!new Settings(path).Bool("mode4"), "mode4 on in a file without it");
			// Kept: on, saved, read back; off again.
			var settings = new Settings(path);
			settings.Set("mode4", true);
			settings.Save();
			Check(new Settings(path).Bool("mode4"), "mode4 not kept on");
			Check(new Settings(path).Bool("fps30"), "another key lost");
			settings.Set("mode4", false);
			settings.Save();
			Check(!new Settings(path).Bool("mode4"), "mode4 not kept off");

			// The memory figure is stated (not a placeholder), and the README's KYTY_SPECULATE=4 row says the same.
			Check(Mode4.ExtraMemoryGb != "?" && Mode4.ExtraMemoryGb.Length != 0, "the memory figure is still a placeholder");
			if (args.Length > 0)
				Check(File.ReadAllText(Path.Combine(args[0], "docs", "PORTABLE-README.md"))
				          .Contains("it uses about " + Mode4.ExtraMemoryGb + " GB more"),
				      "PORTABLE-README.md's KYTY_SPECULATE=4 row has another figure");

			// The emulator's gate: memory first (28 GiB; 0, a failed read, stands down too), then the performance cores
			// (3 + 4 workers + 2; 0, not known, not checked).
			Check(Gate(28 * GiB, 9).Gate == Mode4.Gate.Available, "28 GiB, 9 CPUs");
			Check(Gate(28 * GiB - 1, 16).Gate == Mode4.Gate.LowRam, "under 28 GiB");
			Check(Gate(0, 16).Gate == Mode4.Gate.LowRam, "memory not read");
			Check(Gate(16 * GiB, 4).Gate == Mode4.Gate.LowRam, "both short: memory first");
			Check(Gate(32 * GiB, 8).Gate == Mode4.Gate.FewCores, "8 CPUs");
			Check(Gate(32 * GiB, 0).Gate == Mode4.Gate.Available, "CPUs not known");
			// Its tuning variables, read as the emulator reads them.
			Check(Gate(16 * GiB, 16, "KYTY_SPECULATE_RAM_GB", "0").Gate == Mode4.Gate.Available, "RAM_GB 0 lifts the gate");
			Check(Gate(16 * GiB, 16, "KYTY_SPECULATE_RAM_GB", "12").Gate == Mode4.Gate.Available, "RAM_GB 12");
			Check(Gate(16 * GiB, 16, "KYTY_SPECULATE_RAM_GB", "12x").RamGib == 28, "RAM_GB not whole digits: 28");
			Check(Gate(16 * GiB, 16, "KYTY_SPECULATE_RAM_GB", "5000").RamGib == 28, "RAM_GB over 4096: 28");
			Check(Gate(32 * GiB, 7, "KYTY_SPECULATE_THREADS", "2").Gate == Mode4.Gate.Available, "2 workers need 7");
			Check(Gate(32 * GiB, 9, "KYTY_SPECULATE_THREADS", "8").CpusNeeded == 13, "8 workers need 13");
			Check(Gate(32 * GiB, 9, "KYTY_SPECULATE_THREADS", "13").CpusNeeded == 9, "13 workers: not taken");
			Check(Gate(32 * GiB, 4, "KYTY_SPECULATE_CPUS", "8-15").Gate == Mode4.Gate.Available, "CPUS lifts the CPU check");
			// -Set pieces as run-windows.ps1 applies them: an empty value, or no "=", is 0; the last one wins.
			Check(Mode4.FromSetOptions(new[] { "A=1,KYTY_SIMULATE_RAM_MB=" }, "KYTY_SIMULATE_RAM_MB") == "0", "an empty value is 0");
			Check(Mode4.FromSetOptions(new[] { "KYTY_SPECULATE_CPUS" }, "KYTY_SPECULATE_CPUS") == "0", "a piece without = is 0");
			Check(Mode4.FromSetOptions(new[] { "KYTY_SPECULATE_RAM_GB=8", "KYTY_SPECULATE_RAM_GB=0" }, "KYTY_SPECULATE_RAM_GB") == "0",
			      "the last piece wins");
			Check(Mode4.FromSetOptions(new[] { "A=1" }, "KYTY_SPECULATE_RAM_GB") == null, "a variable not named");
			// The simulation variables (strtoull / strtoul on their leading digits).
			Check(Gate(64 * GiB, 16, "KYTY_SIMULATE_RAM_MB", "16384").Gate == Mode4.Gate.LowRam, "simulated memory");
			Check(Gate(64 * GiB, 16, "KYTY_SIMULATE_RAM_MB", "40000MB").Gate == Mode4.Gate.Available, "leading digits");
			Check(Gate(64 * GiB, 16, "KYTY_SIMULATE_RAM_MB", "none").Gate == Mode4.Gate.LowRam, "no digits: 0");
			Check(Gate(64 * GiB, 16, "KYTY_SIMULATE_PCORES", "6").Gate == Mode4.Gate.FewCores, "simulated CPUs");
			Check(Gate(64 * GiB, 4, "KYTY_SIMULATE_PCORES", "0").Gate == Mode4.Gate.Available, "simulated 0: not checked");

			// Play: KYTY_SPECULATE=4 alone added when on and offered; nothing when off or not offered; a -Set pair of the
			// Command-line options wins, and its tuning variables are the gate's.
			var off = PlaySets(root, false, "", 32 * GiB, 16);
			var on  = PlaySets(root, true, "", 32 * GiB, 16);
			Check(on.Except(off).SequenceEqual(new[] { "KYTY_SPECULATE=4" }) && off.Except(on).Count() == 0, "on adds only KYTY_SPECULATE=4");
			Check(!off.Any(s => s.StartsWith("KYTY_SPECULATE=")), "off adds it");
			Check(!PlaySets(root, true, "", 16 * GiB, 16).Any(s => s.StartsWith("KYTY_SPECULATE=")), "too little memory adds it");
			Check(!PlaySets(root, true, "", 32 * GiB, 6).Any(s => s.StartsWith("KYTY_SPECULATE=")), "too few CPUs add it");
			var own = PlaySets(root, true, "-Set KYTY_SPECULATE=0", 32 * GiB, 16);
			Check(own.Contains("KYTY_SPECULATE=0") && !own.Contains("KYTY_SPECULATE=4"), "the user's own -Set lost");
			Check(PlaySets(root, true, "-Set KYTY_SPECULATE_RAM_GB=0", 16 * GiB, 16).Contains("KYTY_SPECULATE=4"),
			      "the options' RAM_GB=0 not taken by the gate");
			Check(!PlaySets(root, true, "-Set KYTY_SIMULATE_PCORES=6", 32 * GiB, 16).Contains("KYTY_SPECULATE=4"),
			      "the options' KYTY_SIMULATE_PCORES not taken by the gate");
			Check(!PlaySets(root, true, "-Set KYTY_SIMULATE_RAM_MB=", 32 * GiB, 16).Contains("KYTY_SPECULATE=4"),
			      "the options' empty KYTY_SIMULATE_RAM_MB (0, as run-windows.ps1 makes it) not taken by the gate");
			// The launcher's environment (the game inherits it) is the gate's too, and an options' -Set pair beats it.
			Environment.SetEnvironmentVariable("KYTY_SIMULATE_RAM_MB", "16384");
			Check(!PlaySets(root, true, "", 32 * GiB, 16).Contains("KYTY_SPECULATE=4"), "the environment not taken by the gate");
			Check(PlaySets(root, true, "-Set KYTY_SIMULATE_RAM_MB=65536", 32 * GiB, 16).Contains("KYTY_SPECULATE=4"),
			      "the options' pair does not beat the environment");
			Environment.SetEnvironmentVariable("KYTY_SIMULATE_RAM_MB", null);

			// The text: what it does with the memory figure, or why it is not offered, in English and Chinese, the Chinese
			// its own (T falls back to English for a key Chinese lacks).
			var keys    = new[] { "mode4", "mode4.desc", "mode4.lowRam", "mode4.fewCores", "summary.mode4", "experimental.translation" };
			Strings.Language = UiLanguage.English;
			var english = keys.ToDictionary(k => k, k => Strings.T(k));
			foreach (var key in keys) Check(english[key] != key, "English: " + key + " missing");
			Strings.Language = UiLanguage.Chinese;
			foreach (var key in keys)
				Check(Strings.T(key) != english[key] && Strings.T(key) != key, "Chinese: " + key + " is not its own");
			foreach (var language in new[] { UiLanguage.English, UiLanguage.Chinese }) {
				Strings.Language = language;
				var desc = Mode4.Description(Gate(32 * GiB, 16));
				Check(desc.Contains(Mode4.ExtraMemoryGb + " GB") && desc.Contains("KYTY_SPECULATE=4"), language + ": desc");
				Check(Mode4.Description(Gate(16 * GiB, 16)).Contains("28 GB"), language + ": lowRam");
				var cores = Mode4.Description(Gate(32 * GiB, 6));
				Check(cores.Contains("9") && cores.Contains("6"), language + ": fewCores");
			}
			Strings.Language = UiLanguage.English;
			Console.WriteLine("Launcher mode 4 tests passed (" + checks + " checks).");
			return 0;
		} catch (Exception error) {
			Console.WriteLine("FAIL: " + error.Message);
			return 1;
		} finally {
			try {
				Directory.Delete(root, true);
			} catch (Exception) {
			}
		}
	}
}
}

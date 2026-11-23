// SPDX-License-Identifier: GPL-2.0-or-later
// The Windows launcher's weave switch (launcher-windows/Weave.cs): off by default, kept in launcher-settings.json, offered
// only where the emulator would not stand the weave down (its own gate: mode 4, the performance cores, the switches it
// needs), KYTY_WEAVE=1 in Play's command only when on, offered and no mode 4 is sent (a Command-line options' -Set pair
// winning), its memory figure stated, its text in English and Chinese. The repository's folder is the argument.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace KytyLauncher {
static class LauncherWeaveTests {
	const ulong GiB = 1UL << 30;
	static int checks;
	static void Check(bool value, string message) {
		++checks;
		if (!value) throw new Exception(message);
	}
	static Func<string, string> Variables(params string[] pairs) {
		var variables = new Dictionary<string, string>();
		for (int i = 0; i + 1 < pairs.Length; i += 2) variables[pairs[i]] = pairs[i + 1];
		return name => variables.ContainsKey(name) ? variables[name] : null;
	}
	// The weave's gate with mode 4's on the same variables, on a PC with this memory and these logical CPUs.
	static Weave.Verdict Gate(ulong ram, uint performance, uint logical, params string[] pairs) {
		var variable = Variables(pairs);
		return Weave.Check(Mode4.Check(ram, performance, variable), performance, logical, variable);
	}
	// Play's -Set pairs (Commands.PlayCommand) on a PC with this memory and these performance cores.
	static List<string> PlaySets(string root, bool mode4, bool weave, string extra, ulong ram, int cpus) {
		var settings = new Settings(Path.Combine(root, "play-" + Guid.NewGuid().ToString("N") + ".json"));
		settings.Set("mode4", mode4);
		settings.Set("weave", weave);
		settings.Set("extra_args", extra);
		var command = new Commands(new LauncherContext(root, settings, ram, cpus)).PlayCommand(null, true);
		var parts   = command.Split(' ');
		var at      = Array.IndexOf(parts, "-Set");
		return at < 0 || at + 1 >= parts.Length ? new List<string>() : parts[at + 1].Split(',').ToList();
	}
	static int Main(string[] args) {
		var root = Path.Combine(Path.GetTempPath(), "kyty-weave-test-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(root);
		try {
			foreach (var name in new[] { "KYTY_SIMULATE_RAM_MB", "KYTY_SIMULATE_PCORES", "KYTY_SPECULATE_RAM_GB", "KYTY_SPECULATE_THREADS",
			                             "KYTY_SPECULATE_CPUS", "KYTY_SPECULATE", "KYTY_WEAVE", "KYTY_WEAVE_WORKERS", "KYTY_SCOUT",
			                             "KYTY_VULKAN_RECORDING", "KYTY_TABLE_XPR", "KYTY_NATIVE_XPR" })
				Environment.SetEnvironmentVariable(name, null);

			// Off by default, in a new settings file and in one written before the switch existed; kept on and off.
			var path = Path.Combine(root, "launcher-settings.json");
			Check(!new Settings(path).Bool("weave"), "weave on by default");
			File.WriteAllText(path, "{\"mode4\": true}");
			Check(!new Settings(path).Bool("weave"), "weave on in a file without it");
			var settings = new Settings(path);
			settings.Set("weave", true);
			settings.Save();
			Check(new Settings(path).Bool("weave"), "weave not kept on");
			Check(new Settings(path).Bool("mode4"), "another key lost");
			settings.Set("weave", false);
			settings.Save();
			Check(!new Settings(path).Bool("weave"), "weave not kept off");

			// The memory figure is stated (not a placeholder), and the README's KYTY_WEAVE=1 row says the same.
			Check(Weave.ExtraMemoryMb != "?" && Weave.ExtraMemoryMb.Length != 0, "the memory figure is still a placeholder");
			if (args.Length > 0)
				Check(File.ReadAllText(Path.Combine(args[0], "docs", "PORTABLE-README.md"))
				          .Contains("it uses about " + Weave.ExtraMemoryMb + " MB more"),
				      "PORTABLE-README.md's KYTY_WEAVE=1 row has another figure");

			// The performance cores: 10 logical CPUs (8 kept, 2 workers); every logical CPU where the classes are not known;
			// KYTY_WEAVE_WORKERS 1 to 8 lifting the check, 0 not; KYTY_SIMULATE_PCORES not read (CoreCounts).
			Check(Gate(16 * GiB, 10, 32).Gate == Weave.Gate.Available, "10 CPUs");
			var nine = Gate(16 * GiB, 9, 32);
			Check(nine.Gate == Weave.Gate.FewCores && nine.Cpus == 9, "9 CPUs");
			Check(Gate(16 * GiB, 0, 12).Gate == Weave.Gate.Available, "classes not known: 12 logical");
			Check(Gate(16 * GiB, 0, 8).Gate == Weave.Gate.FewCores, "classes not known: 8 logical");
			Check(Gate(16 * GiB, 6, 12, "KYTY_WEAVE_WORKERS", "4").Gate == Weave.Gate.Available, "WEAVE_WORKERS lifts it");
			Check(Gate(16 * GiB, 6, 12, "KYTY_WEAVE_WORKERS", "0").Gate == Weave.Gate.FewCores, "WEAVE_WORKERS 0 counts");
			Check(Gate(16 * GiB, 6, 12, "KYTY_SIMULATE_PCORES", "16").Gate == Weave.Gate.FewCores, "SIMULATE_PCORES not read");
			// No memory gate: 8 GiB is enough.
			Check(Gate(8 * GiB, 16, 32).Gate == Weave.Gate.Available, "a memory gate");
			// Mode 4 in the variables: it stands the weave down where mode 4 would run, not where mode 4 stands down itself.
			Check(Gate(32 * GiB, 16, 32, "KYTY_SPECULATE", "4").Gate == Weave.Gate.Mode4, "KYTY_SPECULATE=4");
			Check(Gate(16 * GiB, 16, 32, "KYTY_SPECULATE", "4").Gate == Weave.Gate.Available, "mode 4 stood down for memory");
			Check(Gate(32 * GiB, 16, 32, "KYTY_SPECULATE", "0").Gate == Weave.Gate.Available, "KYTY_SPECULATE=0");
			// The switches it needs on, the first found named.
			var scout = Gate(16 * GiB, 16, 32, "KYTY_SCOUT", "0", "KYTY_TABLE_XPR", "0");
			Check(scout.Gate == Weave.Gate.Needs && scout.Needs == "KYTY_SCOUT", "KYTY_SCOUT=0");
			Check(Gate(16 * GiB, 16, 32, "KYTY_NATIVE_XPR", "0").Needs == "KYTY_NATIVE_XPR", "KYTY_NATIVE_XPR=0");
			Check(Gate(16 * GiB, 16, 32, "KYTY_SCOUT", "2").Gate == Weave.Gate.Available, "KYTY_SCOUT=2");
			Check(Gate(16 * GiB, 9, 32, "KYTY_SCOUT", "0").Gate == Weave.Gate.FewCores, "the CPU check before the scout's");

			// Play: KYTY_WEAVE=1 alone added when on and offered; nothing when off or not offered; mode 4 sent instead when
			// both are on and mode 4 is offered (the emulator's order), the weave where mode 4 is not; the user's own pair wins.
			var off = PlaySets(root, false, false, "", 16 * GiB, 16);
			var on  = PlaySets(root, false, true, "", 16 * GiB, 16);
			Check(on.Except(off).SequenceEqual(new[] { "KYTY_WEAVE=1" }) && off.Except(on).Count() == 0, "on adds only KYTY_WEAVE=1");
			Check(!off.Any(s => s.StartsWith("KYTY_WEAVE=")), "off adds it");
			Check(!PlaySets(root, false, true, "", 16 * GiB, 9).Any(s => s.StartsWith("KYTY_WEAVE=")), "too few CPUs add it");
			var both = PlaySets(root, true, true, "", 32 * GiB, 16);
			Check(both.Contains("KYTY_SPECULATE=4") && !both.Any(s => s.StartsWith("KYTY_WEAVE=")), "both on: mode 4 alone");
			var low = PlaySets(root, true, true, "", 16 * GiB, 16);
			Check(low.Contains("KYTY_WEAVE=1") && !low.Any(s => s.StartsWith("KYTY_SPECULATE=")), "mode 4 not offered: the weave");
			var own = PlaySets(root, false, true, "-Set KYTY_WEAVE=0", 16 * GiB, 16);
			Check(own.Contains("KYTY_WEAVE=0") && !own.Contains("KYTY_WEAVE=1"), "the user's own -Set lost");
			Check(!PlaySets(root, false, true, "-Set KYTY_SPECULATE=4", 32 * GiB, 16).Contains("KYTY_WEAVE=1"),
			      "the options' KYTY_SPECULATE=4 not taken by the gate");
			Check(PlaySets(root, false, true, "-Set KYTY_WEAVE_WORKERS=2", 16 * GiB, 6).Contains("KYTY_WEAVE=1"),
			      "the options' KYTY_WEAVE_WORKERS not taken by the gate");
			Check(!PlaySets(root, false, true, "-Set KYTY_SCOUT=", 16 * GiB, 16).Contains("KYTY_WEAVE=1"),
			      "the options' empty KYTY_SCOUT (0, as run-windows.ps1 makes it) not taken by the gate");
			// The launcher's environment (the game inherits it) is the gate's too, and an options' -Set pair beats it.
			Environment.SetEnvironmentVariable("KYTY_SCOUT", "0");
			Check(!PlaySets(root, false, true, "", 16 * GiB, 16).Contains("KYTY_WEAVE=1"), "the environment not taken by the gate");
			Check(PlaySets(root, false, true, "-Set KYTY_SCOUT=1", 16 * GiB, 16).Contains("KYTY_WEAVE=1"),
			      "the options' pair does not beat the environment");
			Environment.SetEnvironmentVariable("KYTY_SCOUT", null);

			// The text: what it does with the memory figure, or why it is not offered, in English and Chinese, the Chinese
			// its own (T falls back to English for a key Chinese lacks).
			var keys    = new[] { "weave", "weave.desc", "weave.mode4", "weave.fewCores", "weave.needs", "summary.weave" };
			Strings.Language = UiLanguage.English;
			var english = keys.ToDictionary(k => k, k => Strings.T(k));
			foreach (var key in keys) Check(english[key] != key, "English: " + key + " missing");
			Strings.Language = UiLanguage.Chinese;
			foreach (var key in keys)
				Check(Strings.T(key) != english[key] && Strings.T(key) != key, "Chinese: " + key + " is not its own");
			foreach (var language in new[] { UiLanguage.English, UiLanguage.Chinese }) {
				Strings.Language = language;
				var desc = Weave.Description(Gate(16 * GiB, 16, 32));
				Check(desc.Contains(Weave.ExtraMemoryMb + " MB") && desc.Contains("KYTY_WEAVE=1"), language + ": desc");
				Check(Weave.Description(Gate(32 * GiB, 16, 32, "KYTY_SPECULATE", "4")).Contains("KYTY_SPECULATE=4"), language + ": mode4");
				var cores = Weave.Description(Gate(16 * GiB, 6, 32));
				Check(cores.Contains("10") && cores.Contains("6"), language + ": fewCores");
				Check(Weave.Description(Gate(16 * GiB, 16, 32, "KYTY_SCOUT", "0")).Contains("KYTY_SCOUT"), language + ": needs");
			}
			Strings.Language = UiLanguage.English;
			Console.WriteLine("Launcher weave tests passed (" + checks + " checks).");
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

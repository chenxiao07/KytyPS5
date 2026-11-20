// SPDX-License-Identifier: GPL-2.0-or-later
// Uses the user's own originals as temporary fixtures; none are redistributed.
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace KytyLauncher {
sealed class GameInfo { public string Path, TitleId, Version; }
static class KeyboardPromptsTests {
	static void Check(bool value, string message) { if (!value) throw new Exception(message); }
	static string Reject(Action action) {
		try { action(); } catch (IOException error) {
			Check(error.Message.Contains("turn off") && error.Message.Contains("press Play") &&
				error.Message.Contains(".kyty-keyboard-prompts-v1"), "Missing recovery instructions");
			return error.Message;
		}
		throw new Exception("Expected the operation to be rejected.");
	}
	static string Sha256(byte[] data) {
		using (var sha = System.Security.Cryptography.SHA256.Create())
			return BitConverter.ToString(sha.ComputeHash(data)).Replace("-", "").ToLowerInvariant();
	}
	[DllImport("kernel32.dll", SetLastError = true)]
	static extern bool LockFileEx(Microsoft.Win32.SafeHandles.SafeFileHandle file, uint flags, uint reserved, uint lengthLow, uint lengthHigh, ref System.Threading.NativeOverlapped overlapped);
	[DllImport("kernel32.dll", SetLastError = true)]
	static extern bool UnlockFileEx(Microsoft.Win32.SafeHandles.SafeFileHandle file, uint reserved, uint lengthLow, uint lengthHigh, ref System.Threading.NativeOverlapped overlapped);
	// The embedded pixel offsets of the page marks, as the launcher reads them: 4-byte little-endian offsets.
	static byte[] MarksMap() {
		using (var input = typeof(PageMarks).Assembly.GetManifestResourceStream("KytyLauncher.KeyboardPrompts.page-marks-offsets.bin"))
		using (var output = new MemoryStream()) { input.CopyTo(output); return output.ToArray(); }
	}
	static int Main(string[] args) {
		var root = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "kyty-prompts-test-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(root);
		try {
			var folder = System.IO.Path.Combine(root, "ui", "textures", "buttonicons", "_ps5");
			Directory.CreateDirectory(folder);
			var game = new GameInfo { Path = root, TitleId = "PPSA01341", Version = "01.007.000" };
			KeyboardPrompts.Apply(game, false);
			Check(!Directory.Exists(System.IO.Path.Combine(root, ".kyty-keyboard-prompts-v1")), "Default off wrote a backup");
			foreach (var path in Directory.GetFiles(args[0], "*.ctxr")) File.Copy(path, System.IO.Path.Combine(folder, System.IO.Path.GetFileName(path)));
			// Optional: page-marks-original.bin (Build-PageMarksMap.ps1 -OriginalOut) becomes a sheet of the real size with only
			// the two mark regions filled in.
			string atlas = null;
			byte[] marks = null;
			var marksFixture = System.IO.Path.Combine(args[0], "page-marks-original.bin");
			if (File.Exists(marksFixture)) {
				marks = File.ReadAllBytes(marksFixture);
				atlas = System.IO.Path.Combine(root, "ui", "textures", "_ps5", "_ps5", "global_pages.ctxr");
				Directory.CreateDirectory(System.IO.Path.GetDirectoryName(atlas));
				using (var sheet = new FileStream(atlas, FileMode.CreateNew)) {
					sheet.SetLength(335544397);
					// A header of 0xA5 before the mapped pixels (the pixel data starts at byte 65), so a stray write there shows.
					sheet.Write(Enumerable.Repeat((byte)0xA5, 65).ToArray(), 0, 65);
				}
				PageMarks.WriteRegions(atlas, marks);
			}
			var originals = Directory.GetFiles(folder).ToDictionary(p => p, File.ReadAllBytes);
			game.Version = "01.005.000";
			Reject(() => KeyboardPrompts.Apply(game, true));
			game.Version = "01.007.000";
			KeyboardPrompts.Apply(game, true);
			var installed = originals.Keys.ToDictionary(p => p, File.ReadAllBytes);
			Check(installed.Count(p => !p.Value.SequenceEqual(originals[p.Key])) == 22, "Expected exactly 22 changed icons");
			KeyboardPrompts.Apply(game, true);
			foreach (var p in installed) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Second enable changed an icon");
			KeyboardPrompts.Apply(game, false);
			foreach (var p in originals) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Restore did not recover exact original bytes");
			// Simulate an interrupted install: both original and replacement states are recoverable.
			var first = installed.First(p => !p.Value.SequenceEqual(originals[p.Key]));
			File.WriteAllBytes(first.Key, first.Value);
			KeyboardPrompts.Apply(game, false);
			Check(File.ReadAllBytes(first.Key).SequenceEqual(originals[first.Key]), "Partial install did not restore");
			var changed = (byte[])originals[first.Key].Clone(); changed[changed.Length - 1] ^= 1;
			File.WriteAllBytes(first.Key, changed);
			Reject(() => KeyboardPrompts.Apply(game, true));
			Check(File.ReadAllBytes(first.Key).SequenceEqual(changed), "Existing mod was overwritten");
			File.WriteAllBytes(first.Key, originals[first.Key]);
			var backup = System.IO.Path.Combine(root, ".kyty-keyboard-prompts-v1", System.IO.Path.GetFileName(first.Key));
			File.WriteAllBytes(backup, changed);
			Reject(() => KeyboardPrompts.Apply(game, true));
			foreach (var p in originals) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Rejected operation changed a game file");
			File.WriteAllBytes(backup, originals[first.Key]);
			// Force replacement to fail after the first icon; the batch must roll back.
			var second = installed.Where(p => !p.Value.SequenceEqual(originals[p.Key])).OrderBy(p => p.Key).Skip(1).First();
			using (var held = new FileStream(second.Key, FileMode.Open, FileAccess.Read, FileShare.Read)) {
				Reject(() => KeyboardPrompts.Apply(game, true));
			}
			foreach (var p in originals) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Failed batch left a modified icon");
            // Custom key labels are generated and recognized across profiles and interrupted installs.
            var layout = new KeyboardLayout();
            Check(layout.Assign("Cross", "R") == "Square", "Duplicate input should swap actions");
            Check(layout.Bindings["Square"] == "E", "Swap lost the previous input");
            layout.Assign("LeftStickUp", "I"); layout.Assign("R3", "Mouse:X1");
            layout.Sensitivity = "2.0";
            var settingsPath = System.IO.Path.Combine(root, "launcher-settings.json");
            var settings = new Settings(settingsPath); layout.Save(settings); settings.Save(true);
            var loaded = KeyboardLayout.Read(new Settings(settingsPath));
            Check(loaded.Bindings["Cross"] == "R" && loaded.Sensitivity == "2.0", "Layout did not survive reload");
            try { loaded.Assign("Cross", "F7"); throw new Exception("Reserved key accepted"); } catch (IOException) { }
            // A combo refusal, a reserved key and a duplicate binding read in the launcher's language.
            var refusal = new KeyboardLayout(); refusal.Assign("Cross", "Left Alt");
            var duplicate = new KeyboardLayout(); duplicate.Bindings["Circle"] = "E";
            var language = Strings.Language;
            try {
                Strings.Language = UiLanguage.Chinese;
                try { refusal.Assign("R2", "Left Alt+Mouse:Left"); throw new Exception("Held single input accepted"); }
                catch (IOException error) { Check(error.Message == "Left Alt 已单独绑定到 Cross，不能再作为组合键中按住的输入。", "Chinese combo refusal text"); }
                try { refusal.Assign("Cross", "F7"); throw new Exception("Reserved key accepted"); }
                catch (IOException error) { Check(error.Message == "不支持的按键。F7、F8、F11、Pause、Alt+Enter 和 Alt+F4 为模拟器保留。", "Chinese reserved key text"); }
                try { duplicate.Validate(); throw new Exception("Duplicate binding accepted"); }
                catch (IOException error) { Check(error.Message == "键位重复：E。请在键盘控制中为每个动作分配各自的输入。", "Chinese duplicate binding text"); }
                Strings.Language = UiLanguage.English;
                try { refusal.Assign("R2", "Left Alt+Mouse:Left"); throw new Exception("Held single input accepted"); }
                catch (IOException error) { Check(error.Message == "Left Alt is bound to Cross on its own, so it cannot be held for a combo.", "English combo refusal text"); }
                try { refusal.Assign("Cross", "F7"); throw new Exception("Reserved key accepted"); }
                catch (IOException error) { Check(error.Message == "Unsupported key. F7, F8, F11, Pause, Alt+Enter and Alt+F4 are reserved for the emulator.", "English reserved key text"); }
                try { duplicate.Validate(); throw new Exception("Duplicate binding accepted"); }
                catch (IOException error) { Check(error.Message == "Duplicate binding: E. Edit keyboard controls to give each action its own input.", "English duplicate binding text"); }
            } finally { Strings.Language = language; }
            KeyboardPrompts.Apply(game, true, loaded);
            var custom = originals.Keys.ToDictionary(p => p, File.ReadAllBytes);
            foreach(var pair in custom) {
                Check(pair.Value.Take(65).SequenceEqual(installed[pair.Key].Take(65)) && pair.Value.Skip(131137).SequenceEqual(installed[pair.Key].Skip(131137)), "Custom texture changed its metadata");
            }
            Check(!custom[System.IO.Path.Combine(folder,"ps4_button_cross.ctxr")].SequenceEqual(installed[System.IO.Path.Combine(folder,"ps4_button_cross.ctxr")]), "Custom label was not generated");
            KeyboardPrompts.Apply(game, true, loaded);
            foreach(var p in custom) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Custom reapply changed artwork");
            // A saved generated variant is accepted; an unrelated edit still is not.
            var modified = (byte[])custom[first.Key].Clone(); modified[80] ^= 1;
            File.WriteAllBytes(first.Key, modified); Reject(() => KeyboardPrompts.Apply(game, false));
            File.WriteAllBytes(first.Key, custom[first.Key]);
            KeyboardPrompts.Apply(game, true);
            // Mixed generated/default states after interruption must also restore exactly.
            File.WriteAllBytes(System.IO.Path.Combine(folder,"ps4_button_cross.ctxr"), custom[System.IO.Path.Combine(folder,"ps4_button_cross.ctxr")]);
            KeyboardPrompts.Apply(game, false);
            foreach(var p in originals) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Custom profile restore lost original bytes");
            using (var held = new FileStream(second.Key, FileMode.Open, FileAccess.Read, FileShare.Read)) {
                Reject(() => KeyboardPrompts.Apply(game, true, loaded));
            }
            foreach(var p in originals) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Failed custom batch left an icon changed");
            using(var held = new FileStream(settingsPath, FileMode.Open, FileAccess.Read, FileShare.Read)) {
                try { settings.Save(true); throw new Exception("Locked settings write was not reported"); } catch(IOException) { }
            }
            KeyboardPrompts.Apply(game, true);
            File.Delete(backup);
			Reject(() => KeyboardPrompts.Apply(game, false));
			foreach (var p in installed) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "Missing backup caused a partial restore");
			if (atlas != null) {
				// The two page-switch marks in the UI sheet. Each case leaves original spots, original icons and no
				// page-marks files unless the next one says it starts from something else.
				File.WriteAllBytes(backup, originals[first.Key]);
				KeyboardPrompts.Apply(game, false);
				Check(PageMarks.ReadRegions(atlas).SequenceEqual(marks), "Page marks were not original before their checks");
				var backupFolder = System.IO.Path.Combine(root, ".kyty-keyboard-prompts-v1");
				var marksBackup = System.IO.Path.Combine(backupFolder, "page-marks.bin");
				var marksPending = System.IO.Path.Combine(backupFolder, "page-marks.pending");
				var marksRecord = System.IO.Path.Combine(backupFolder, "page-marks.generated");
				File.Delete(marksBackup); File.Delete(marksRecord);
				var standard = new KeyboardLayout();
				var drawn = PageMarks.Render(standard.IconLabel("ps4_button_square.ctxr"), standard.IconLabel("ps4_button_triangle.ctxr"));
				var stamp = new DateTime(2000, 1, 1, 0, 0, 0, DateTimeKind.Utc);
				var map = MarksMap();
				// Apply.
				KeyboardPrompts.Apply(game, true);
				Check(!drawn.SequenceEqual(marks) && PageMarks.ReadRegions(atlas).SequenceEqual(drawn), "Enable did not draw the default page marks");
				Check(File.ReadAllBytes(marksBackup).SequenceEqual(marks), "page-marks.bin does not hold the original bytes");
				Check(File.ReadAllLines(marksRecord).SequenceEqual(new[] { Sha256(drawn) }), "page-marks.generated does not list exactly the written marks");
				Check(!File.Exists(marksPending), "Enable left page-marks.pending");
				// Idempotence. Starts from: own default marks, icons replaced.
				File.SetLastWriteTimeUtc(atlas, stamp);
				KeyboardPrompts.Apply(game, true);
				Check(File.GetLastWriteTimeUtc(atlas) == stamp, "Second enable rewrote the page marks");
				// Restore. Starts from: own default marks, icons replaced.
				KeyboardPrompts.Apply(game, false);
				Check(PageMarks.ReadRegions(atlas).SequenceEqual(marks) && new FileInfo(atlas).Length == 335544397, "Disable did not restore the page marks");
				File.Delete(marksBackup); File.Delete(marksRecord);
				KeyboardPrompts.Apply(game, false);
				Check(!File.Exists(marksBackup), "A disable with original page marks created page-marks.bin");
				// Foreign sheet: one byte of a spot changed.
				var edited = (byte[])marks.Clone(); edited[100] ^= 1;
				PageMarks.WriteRegions(atlas, edited);
				File.SetLastWriteTimeUtc(atlas, stamp);
				KeyboardPrompts.Apply(game, true);
				foreach (var p in installed) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "A foreign page mark stopped the icons");
				Check(File.GetLastWriteTimeUtc(atlas) == stamp && PageMarks.ReadRegions(atlas).SequenceEqual(edited) && !File.Exists(marksBackup) &&
					!File.Exists(marksPending) && !File.Exists(marksRecord), "Enable wrote a foreign sheet or created page mark files for it");
				KeyboardPrompts.Apply(game, false);
				foreach (var p in originals) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "A foreign page mark stopped the icon restore");
				PageMarks.WriteRegions(atlas, marks);
				// Wrong size: the sheet one byte longer.
				using (var sheet = new FileStream(atlas, FileMode.Open)) sheet.SetLength(335544398);
				File.SetLastWriteTimeUtc(atlas, stamp);
				KeyboardPrompts.Apply(game, true);
				foreach (var p in installed) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "A wrong-size sheet stopped the icons");
				Check(File.GetLastWriteTimeUtc(atlas) == stamp && new FileInfo(atlas).Length == 335544398, "Enable touched a wrong-size sheet");
				using (var sheet = new FileStream(atlas, FileMode.Open)) sheet.SetLength(335544397);
				KeyboardPrompts.Apply(game, false);
				// Missing backup: own marks with page-marks.bin deleted.
				KeyboardPrompts.Apply(game, true);
				File.Delete(marksBackup);
				Check(Reject(() => KeyboardPrompts.Apply(game, false)).Contains("global_pages.ctxr"), "The missing page marks backup refusal does not name the sheet to restore");
				Check(PageMarks.ReadRegions(atlas).SequenceEqual(drawn), "Missing page marks backup changed the sheet");
				File.WriteAllBytes(marksBackup, marks);
				KeyboardPrompts.Apply(game, false);
				File.Delete(marksBackup); File.Delete(marksRecord);
				// Custom Square label.
				var squareOnly = new KeyboardLayout(); squareOnly.Assign("Square", "J");
				KeyboardPrompts.Apply(game, true, squareOnly);
				var relabeled = PageMarks.ReadRegions(atlas);
				int half = PageMarks.Side * PageMarks.Side * 4;
				Check(!relabeled.Take(half).SequenceEqual(drawn.Take(half)) && relabeled.Skip(half).SequenceEqual(drawn.Skip(half)), "Custom Square key did not change only the square mark");
				KeyboardPrompts.Apply(game, false);
				File.Delete(marksBackup); File.Delete(marksRecord);
				// Failed write after the sheet is open: a shared byte-range lock on the last mapped pixel makes the last
				// write fail, and the rollback's too, so page-marks.pending stays for the next Play.
				var overlapped = new System.Threading.NativeOverlapped { OffsetLow = BitConverter.ToInt32(map, map.Length - 4) };
				using (var held = new FileStream(atlas, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete)) {
					Check(LockFileEx(held.SafeFileHandle, 1, 0, 4, 0, ref overlapped), "Could not lock the last mapped pixel of the sheet");
					Reject(() => KeyboardPrompts.Apply(game, true));
					Check(File.Exists(marksPending), "A write that failed after the sheet was open left no page-marks.pending");
					Check(PageMarks.ReadRegions(atlas).SequenceEqual(marks), "The rollback did not restore the page marks");
					foreach (var p in originals) Check(File.ReadAllBytes(p.Key).SequenceEqual(p.Value), "A failed page marks write left an icon changed");
					Check(UnlockFileEx(held.SafeFileHandle, 0, 4, 0, ref overlapped), "Could not unlock the last mapped pixel of the sheet");
				}
				KeyboardPrompts.Apply(game, true);
				Check(PageMarks.ReadRegions(atlas).SequenceEqual(drawn) && !File.Exists(marksPending), "Enable after a failed write did not relabel and remove page-marks.pending");
				KeyboardPrompts.Apply(game, false);
				Check(PageMarks.ReadRegions(atlas).SequenceEqual(marks), "Disable after a failed write did not restore the page marks");
				File.Delete(marksBackup); File.Delete(marksRecord);
				// Whole-sheet scan, last. Every byte outside the mapped pixels still holds the fixture: 0xA5 in the 65-byte
				// header, zero everywhere else. Each mapped offset covers 4 bytes.
				var positions = Enumerable.Range(0, map.Length).Select(i => BitConverter.ToInt32(map, i - i % 4) + i % 4).OrderBy(x => x).ToArray();
				long at = 0, stray = -1;
				int next = 0;
				var chunk = new byte[1 << 20];
				using (var sheet = new FileStream(atlas, FileMode.Open, FileAccess.Read, FileShare.Read))
					for (int read; (read = sheet.Read(chunk, 0, chunk.Length)) > 0; )
						for (int i = 0; i < read; i++, at++)
							if (next < positions.Length && positions[next] == at) next++;
							else if (chunk[i] != (at < 65 ? 0xA5 : 0) && stray < 0) stray = at;
				Check(at == 335544397 && next == positions.Length && stray < 0, "A byte outside the mapped pixels changed (first at " + stray + ")");
			}
			Console.WriteLine("PASS: default off, version guard, apply, idempotence, exact restore, partial recovery, mod/corrupt/missing backup protection, locked-file rollback, custom profile swaps/persistence/artwork/restore, protected settings writes, combo refusal language, older message language" +
				(atlas != null ? ", page marks apply/idempotence/restore/foreign sheet/wrong size/missing backup/custom Square/failed write after open/whole-sheet scan" : "; page marks skipped (no page-marks-original.bin fixture)"));
			return 0;
		} finally {
			var expectedParent = System.IO.Path.GetFullPath(System.IO.Path.GetTempPath()).TrimEnd(System.IO.Path.DirectorySeparatorChar);
			Check(System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(root)) == expectedParent &&
				System.IO.Path.GetFileName(root).StartsWith("kyty-prompts-test-"), "Unexpected cleanup path");
			Directory.Delete(root, true);
		}
	}
}
}

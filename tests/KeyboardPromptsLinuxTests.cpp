// SPDX-License-Identifier: GPL-2.0-or-later
// The Linux launcher's keyboard prompts (launcher-linux/keyboard_prompts.cpp), the counterpart of
// KeyboardPromptsTests.cs. No game files are needed: the "originals" are made from the pack's own
// replacement textures (one byte changed), and the pack's hashes are replaced for the test.
//
//   g++ -std=c++20 -iquote launcher-linux -I3rdparty/stb -DKYTY_PROMPT_ASSETS_DIR='"<repo>/launcher-windows/KeyboardPrompts"'
//       tests/KeyboardPromptsLinuxTests.cpp launcher-linux/keyboard_prompts.cpp launcher-linux/keyboard_layout.cpp
//       launcher-linux/page_marks.cpp launcher-linux/prompt_texture.cpp launcher-linux/prompt_assets.cpp
//       launcher-linux/bytes.cpp -o prompts_test
//
// The page marks of the UI sheet are tried on a sparse file of the sheet's real size (about 320 MB of holes) whose two
// mapped spots hold made-up "original" bytes, as the icons' originals are made up.
#include "keyboard_layout.h"
#include "keyboard_prompts.h"
#include "page_marks.h"
#include "prompt_texture.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Bytes::Buffer;
using namespace KeyboardPrompts;

static int g_checks = 0;
static void Check(bool value, const std::string& message) {
	++g_checks;
	if (!value) {
		std::printf("FAIL: %s\n", message.c_str());
		std::exit(1);
	}
}

static Buffer Read(const fs::path& path) {
	std::ifstream file(path, std::ios::binary);
	return Buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static void Write(const fs::path& path, const Buffer& data) {
	std::ofstream file(path, std::ios::binary | std::ios::trunc);
	file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

static void Reject(const std::function<void()>& action, const std::string& what) {
	try {
		action();
	} catch (const PromptError& error) {
		const std::string text = error.what();
		Check(text.find("turn off") != std::string::npos && text.find("press Play") != std::string::npos &&
		          text.find(".kyty-keyboard-prompts-v1") != std::string::npos,
		      "missing recovery instructions: " + what);
		return;
	}
	Check(false, "expected a rejection: " + what);
}

using Snapshot = std::map<std::string, Buffer>;
static Snapshot Take(const fs::path& folder) {
	Snapshot all;
	for (const auto& entry: fs::directory_iterator(folder))
		all[entry.path().filename().string()] = Read(entry.path());
	return all;
}

static std::vector<std::string> Lines(const fs::path& path) {
	std::vector<std::string> lines;
	std::ifstream            file(path);
	for (std::string line; std::getline(file, line);) lines.push_back(line);
	return lines;
}

// The text of the PromptError an action ends with.
static std::string RejectText(const std::function<void()>& action, const std::string& what) {
	try {
		action();
	} catch (const PromptError& error) {
		return error.what();
	}
	Check(false, "expected a rejection: " + what);
	return {};
}

int main() {
	const fs::path root = fs::temp_directory_path() / ("kyty-prompts-linux-test-" + std::to_string(::getpid()));
	fs::remove_all(root);
	fs::create_directories(root);
	try {
		// The pack's real icons, and test originals derived from them.
		const auto        pack = PackIcons();
		Check(pack.size() == 22, "pack has 22 icons");
		std::vector<Icon> icons;
		Snapshot          originals;
		for (const auto& icon: pack) {
			Buffer original = icon.replacement;
			original[original.size() - 1] ^= 0x5a;
			originals[icon.name] = original;
			icons.push_back({icon.name, Bytes::Sha256Hex(original), icon.replacement_hash, icon.replacement});
		}
		SetIconsForTesting(icons);

		const fs::path game_root = root / "game";
		const fs::path folder    = game_root / "ui" / "textures" / "buttonicons" / "_ps5";
		fs::create_directories(folder);
		Game game {game_root.string(), "PPSA01341", "01.007.000"};

		// Off by default: no backup, nothing written.
		Apply(game, false);
		Check(!fs::exists(game_root / ".kyty-keyboard-prompts-v1"), "default off wrote a backup");
		for (const auto& [name, data]: originals) Write(folder / name, data);

		// The version guard.
		game.version = "01.005.000";
		Reject([&] { Apply(game, true); }, "wrong version");
		game.version = "01.007.000";
		Game other = game;
		other.title_id = "PPSA00000";
		Reject([&] { Apply(other, true); }, "wrong title");
		Check(Take(folder) == originals, "a rejected apply changed a file");

		// Apply, idempotence, exact restore.
		Apply(game, true);
		const auto installed = Take(folder);
		int        changed   = 0;
		for (const auto& [name, data]: installed) changed += data != originals[name] ? 1 : 0;
		Check(changed == 22, "exactly 22 changed icons");
		for (const auto& icon: pack) Check(installed.at(icon.name) == icon.replacement, "default artwork is the pack's");
		Apply(game, true);
		Check(Take(folder) == installed, "second enable changed an icon");
		Apply(game, false);
		Check(Take(folder) == originals, "restore did not recover the exact original bytes");
		Check(fs::exists(game_root / ".kyty-keyboard-prompts-v1" / "ps4_l1.ctxr"), "the backup stays when off");
		Apply(game, false);
		Check(Take(folder) == originals, "second disable changed an icon");

		// An interrupted install: one replaced icon, then restore.
		const std::string first = "ps4_button_cross.ctxr", second = "ps4_button_circle.ctxr";
		Write(folder / first, installed.at(first));
		Apply(game, false);
		Check(Read(folder / first) == originals[first], "a partial install did not restore");
		// ... and complete.
		Write(folder / first, installed.at(first));
		Apply(game, true);
		Check(Take(folder) == installed, "a partial install did not complete");
		Apply(game, false);

		// An unrelated edit (another mod) is never overwritten.
		Buffer modded = originals[first];
		modded[modded.size() - 1] ^= 1;
		Write(folder / first, modded);
		Reject([&] { Apply(game, true); }, "modified icon");
		Check(Read(folder / first) == modded, "an existing mod was overwritten");
		Write(folder / first, originals[first]);

		// A damaged backup stops the batch before any write.
		const fs::path backup = game_root / ".kyty-keyboard-prompts-v1";
		Write(backup / first, modded);
		Reject([&] { Apply(game, true); }, "damaged backup");
		Check(Take(folder) == originals, "a rejected batch changed a game file");
		Write(backup / first, originals[first]);

		// A write that fails after the first icon: the batch rolls back.
		int writes = 0;
		SetWriteHookForTesting([&](const std::string& path) {
			if (path.find("/_ps5/") != std::string::npos && ++writes == 2) throw PromptError("simulated write failure");
		});
		Reject([&] { Apply(game, true); }, "failed write");
		SetWriteHookForTesting({});
		Check(Take(folder) == originals, "a failed batch left a modified icon");
		// A rollback that fails too: reported, the others still go back.
		writes = 0;
		SetWriteHookForTesting([&](const std::string& path) {
			if (path.find("/_ps5/") == std::string::npos) return;
			++writes;
			if (writes == 3 || writes == 4) throw PromptError("simulated write failure");
		});
		try {
			Apply(game, true);
			Check(false, "failed write with a failed rollback was accepted");
		} catch (const PromptError& error) {
			Check(std::string(error.what()).find("also failed") != std::string::npos, "the failed rollback is reported");
		}
		SetWriteHookForTesting({});
		// (One icon is left changed, as reported; restoring from the backup recovers it.)
		Apply(game, false);
		Check(Take(folder) == originals, "recovery after a failed rollback");

		// Custom layouts.
		KeyboardLayout layout;
		Check(layout.Assign("Cross", "R") == "Square", "a duplicate input swaps the actions");
		Check(layout.Binding("Square") == "E", "the swap kept the previous input");
		layout.Assign("LeftStickUp", "I");
		layout.Assign("R3", "Mouse:X1");
		layout.sensitivity = "2.0";
		try {
			layout.Assign("Cross", "F7");
			Check(false, "a reserved key was accepted");
		} catch (const PromptError&) {}
		Apply(game, true, &layout);
		const auto custom = Take(folder);
		for (const auto& [name, data]: custom) {
			Check(Buffer(data.begin(), data.begin() + 65) == Buffer(installed.at(name).begin(), installed.at(name).begin() + 65) &&
			          Buffer(data.begin() + 131137, data.end()) == Buffer(installed.at(name).begin() + 131137, installed.at(name).end()),
			      "a custom texture changed its metadata: " + name);
		}
		Check(custom.at(first) != installed.at(first), "the custom label was generated");
		Check(custom.at("ps4_l1.ctxr") == installed.at("ps4_l1.ctxr"), "an unchanged label keeps the pack's artwork");
		Check(custom.at("ps4_stick_r.ctxr") == installed.at("ps4_stick_r.ctxr"), "the mouse icon stays");
		Check(custom.at("ps4_stick_l.ctxr") != installed.at("ps4_stick_l.ctxr"), "the stick shows its new keys");
		Apply(game, true, &layout);
		Check(Take(folder) == custom, "reapplying a custom layout changed artwork");
		// Another profile (the defaults) over it, and back.
		Apply(game, true);
		Check(Take(folder) == installed, "switching to the default profile");
		Apply(game, true, &layout);
		Check(Take(folder) == custom, "switching to the custom profile");
		// An unrelated edit of a generated icon is still refused.
		Buffer edited = custom.at(first);
		edited[80] ^= 1;
		Write(folder / first, edited);
		Reject([&] { Apply(game, false); }, "edited generated icon");
		Write(folder / first, custom.at(first));
		// A mix of generated and default states after an interruption restores exactly.
		Apply(game, true);
		Write(folder / first, custom.at(first));
		Apply(game, false);
		Check(Take(folder) == originals, "a custom profile restore lost original bytes");
		// A failed custom batch rolls back.
		writes = 0;
		SetWriteHookForTesting([&](const std::string& path) {
			if (path.find("/_ps5/") != std::string::npos && ++writes == 2) throw PromptError("simulated write failure");
		});
		Reject([&] { Apply(game, true, &layout); }, "failed custom batch");
		SetWriteHookForTesting({});
		Check(Take(folder) == originals, "a failed custom batch left an icon changed");
		// Generated variants are readable gzip files (the Windows launcher reads them too).
		std::string why;
		bool        found = false;
		for (const auto& entry: fs::recursive_directory_iterator(backup / "generated")) {
			if (entry.path().extension() != ".gz") continue;
			found = true;
			const auto bytes = Bytes::Gunzip(Read(entry.path()), &why);
			Check(why.empty() && bytes.size() == 131149 && Bytes::Sha256Hex(bytes) == entry.path().stem().string(), "generated record is valid");
		}
		Check(found, "generated variants were recorded");

		// A missing backup stops a restore before anything is written.
		Apply(game, true);
		fs::remove(backup / first);
		Reject([&] { Apply(game, false); }, "missing backup");
		Check(Take(folder) == installed, "a missing backup caused a partial restore");
		Write(backup / first, originals[first]);
		Apply(game, false);

		// Linux specifics: a link inside the game folder is refused; the game folder reached through
		// a link is fine; names that differ only in case are found.
		fs::rename(folder / first, root / "moved.ctxr");
		fs::create_symlink(root / "moved.ctxr", folder / first);
		Reject([&] { Apply(game, true); }, "a linked icon");
		fs::remove(folder / first);
		fs::rename(root / "moved.ctxr", folder / first);
		const fs::path link = root / "game-link";
		fs::create_directory_symlink(game_root, link);
		Game linked = game;
		linked.path = link.string();
		Apply(linked, true);
		Check(Take(folder) == installed, "applied through a linked game folder");
		Apply(linked, false);
		Check(Take(folder) == originals, "restored through a linked game folder");
		fs::rename(game_root / "ui", game_root / "UI");
		Apply(game, true);
		Check(Take(game_root / "UI" / "textures" / "buttonicons" / "_ps5") == installed, "a folder of another case");
		Apply(game, false);
		fs::rename(game_root / "UI", game_root / "ui");

		// Two launchers at once: the second one waits for nobody.
		const int fd = ::open((backup / "lock").c_str(), O_RDWR);
		Check(fd >= 0 && ::flock(fd, LOCK_EX | LOCK_NB) == 0, "test lock");
		Reject([&] { Apply(game, true); }, "locked by another launcher");
		::close(fd);
		Apply(game, true);
		Apply(game, false);

		// The resampler the page marks (66x66) are drawn with: a power of two gives what Scaled gives; other sizes average
		// the part of the drawing a pixel covers, edges by share, colours by alpha.
		{
			const auto cap = PromptTexture::Draw("A");
			for (const int width: {1, 2, 64, 128, 512})
				Check(PromptTexture::Resampled(cap, width) == PromptTexture::Scaled(cap, width), "Resampled equals Scaled at " + std::to_string(width));
			Buffer half(512 * 512 * 4, 0);
			for (int y = 0; y < 512; ++y)
				for (int x = 0; x < 250; ++x) {
					uint8_t* p = &half[(static_cast<size_t>(y) * 512 + static_cast<size_t>(x)) * 4];
					p[0] = 255, p[3] = 255; // opaque red left of column 250, nothing right of it
				}
			const auto small = PromptTexture::Resampled(half, 66);
			auto       at    = [&](int x, int y) { return &small[(static_cast<size_t>(y) * 66 + static_cast<size_t>(x)) * 4]; };
			Check(small.size() == 66 * 66 * 4, "Resampled size");
			Check(at(0, 0)[0] == 255 && at(0, 0)[1] == 0 && at(0, 0)[3] == 255, "an opaque pixel stays opaque");
			Check(at(65, 30)[0] == 0 && at(65, 30)[3] == 0, "an empty pixel stays empty");
			Check(at(32, 10)[3] > 0 && at(32, 10)[3] < 255 && at(32, 10)[0] == 255 && at(32, 10)[1] == 0, "an edge pixel is partly covered and keeps its colour");
			try {
				PromptTexture::Resampled(half, 513);
				Check(false, "a size above 512 was accepted");
			} catch (const PromptError&) {}
		}

		// The two page-switch marks in the UI sheet (KeyboardPromptsTests.cs' page marks section). The sheet is a sparse file
		// of the real size with a 0xA5 header before the mapped pixels, so that a stray write there shows.
		{
			const auto&   map = PageMarks::Offsets();
			Buffer        marks(map.size() * 4);
			uint32_t      seed = 12345;
			for (auto& byte: marks) {
				seed = seed * 1664525u + 1013904223u;
				byte = static_cast<uint8_t>(seed >> 24u);
			}
			PageMarks::SetOriginalForTesting(marks);
			const uint64_t size  = PageMarks::ExpectedSize();
			const fs::path atlas = game_root / PageMarks::RelativePath();
			fs::create_directories(atlas.parent_path());
			{
				std::ofstream sheet(atlas, std::ios::binary);
				sheet << std::string(65, '\xA5');
			}
			fs::resize_file(atlas, size);
			PageMarks::WriteRegions(atlas, marks);
			Check(PageMarks::ReadRegions(atlas) == marks, "the fixture's page marks");

			const fs::path marks_backup = backup / "page-marks.bin", pending = backup / "page-marks.pending",
			               record = backup / "page-marks.generated";
			auto           clear = [&] {
                fs::remove(marks_backup);
                fs::remove(record);
			};
			// A sheet's age is set back a year so that a rewrite shows.
			auto age = [&] {
				fs::last_write_time(atlas, fs::file_time_type::clock::now() - std::chrono::hours(24 * 365));
				return fs::last_write_time(atlas);
			};
			const KeyboardLayout standard;
			const auto           drawn = PageMarks::Render(standard.IconLabel("ps4_button_square.ctxr"), standard.IconLabel("ps4_button_triangle.ctxr"));
			Check(drawn.size() == marks.size() && drawn != marks, "the default page marks are drawn");
			Check(!std::equal(drawn.begin(), drawn.begin() + 66 * 66 * 4, drawn.begin() + 66 * 66 * 4), "the square and triangle marks differ");
			Check(std::any_of(drawn.begin(), drawn.end(), [](uint8_t byte) { return byte != 0; }), "the marks are not blank");

			// Apply. The icons and the marks change together.
			Apply(game, true);
			Check(Take(folder) == installed, "enable changed the icons");
			Check(PageMarks::ReadRegions(atlas) == drawn, "enable did not draw the default page marks");
			Check(Read(marks_backup) == marks, "page-marks.bin does not hold the original bytes");
			Check(Lines(record) == std::vector<std::string> {Bytes::Sha256Hex(drawn)}, "page-marks.generated does not list exactly the written marks");
			Check(!fs::exists(pending), "enable left page-marks.pending");
			Check(fs::file_size(atlas) == size, "enable changed the sheet's size");
			// Idempotence. Starts from: own default marks, icons replaced.
			auto stamp = age();
			Apply(game, true);
			Check(fs::last_write_time(atlas) == stamp && PageMarks::ReadRegions(atlas) == drawn, "a second enable rewrote the page marks");
			// Restore. Starts from: own default marks, icons replaced.
			Apply(game, false);
			Check(PageMarks::ReadRegions(atlas) == marks && fs::file_size(atlas) == size, "disable did not restore the page marks");
			Check(Take(folder) == originals, "disable did not restore the icons");
			clear();
			Apply(game, false);
			Check(!fs::exists(marks_backup), "a disable with original page marks created page-marks.bin");

			// A foreign sheet: one byte of a spot changed (a mod, another game version). The icons go ahead without it.
			Buffer edited = marks;
			edited[100] ^= 1;
			PageMarks::WriteRegions(atlas, edited);
			stamp = age();
			Apply(game, true);
			Check(Take(folder) == installed, "a foreign page mark stopped the icons");
			Check(fs::last_write_time(atlas) == stamp && PageMarks::ReadRegions(atlas) == edited && !fs::exists(marks_backup) &&
			          !fs::exists(pending) && !fs::exists(record),
			      "enable wrote a foreign sheet or created page mark files for it");
			Apply(game, false);
			Check(Take(folder) == originals, "a foreign page mark stopped the icon restore");
			PageMarks::WriteRegions(atlas, marks);

			// Another size: the sheet one byte longer.
			fs::resize_file(atlas, size + 1);
			stamp = age();
			Apply(game, true);
			Check(Take(folder) == installed, "a wrong-size sheet stopped the icons");
			Check(fs::last_write_time(atlas) == stamp && fs::file_size(atlas) == size + 1 && !fs::exists(marks_backup), "enable touched a wrong-size sheet");
			Apply(game, false);
			fs::resize_file(atlas, size);
			clear();

			// No sheet at all.
			fs::rename(atlas, root / "sheet-away");
			Apply(game, true);
			Check(Take(folder) == installed && !fs::exists(marks_backup) && !fs::exists(pending), "a missing sheet stopped the icons or left page mark files");
			Apply(game, false);
			fs::rename(root / "sheet-away", atlas);
			clear();

			// A missing backup: own marks with page-marks.bin deleted. Stops before anything is written.
			Apply(game, true);
			fs::remove(marks_backup);
			const auto missing = RejectText([&] { Apply(game, false); }, "a missing page marks backup");
			Check(missing.find("global_pages.ctxr") != std::string::npos && missing.find("page-marks.bin") != std::string::npos,
			      "the missing page marks backup refusal does not name the sheet to restore");
			Check(PageMarks::ReadRegions(atlas) == drawn && Take(folder) == installed, "a missing page marks backup changed the game");
			Write(marks_backup, marks);
			Apply(game, false);
			Check(PageMarks::ReadRegions(atlas) == marks && Take(folder) == originals, "restore with the backup put back");
			// A damaged one is refused too.
			Apply(game, true);
			Buffer damaged = marks;
			damaged[0] ^= 1;
			Write(marks_backup, damaged);
			Check(RejectText([&] { Apply(game, false); }, "a damaged page marks backup").find("Invalid original prompt backup") != std::string::npos,
			      "the damaged page marks backup is named");
			Check(PageMarks::ReadRegions(atlas) == drawn, "a damaged page marks backup changed the sheet");
			Write(marks_backup, marks);
			Apply(game, false);
			clear();

			// A custom Square label changes only the square mark.
			KeyboardLayout square_only;
			square_only.Assign("Square", "J");
			Apply(game, true, &square_only);
			const auto   relabeled = PageMarks::ReadRegions(atlas);
			const size_t half      = 66 * 66 * 4;
			Check(!std::equal(relabeled.begin(), relabeled.begin() + half, drawn.begin()) &&
			          std::equal(relabeled.begin() + half, relabeled.end(), drawn.begin() + half),
			      "a custom Square key did not change only the square mark");
			Apply(game, true);
			Check(PageMarks::ReadRegions(atlas) == drawn, "switching back to the default profile did not redraw the marks");
			Apply(game, false);
			clear();

			// Failed write after the sheet is open: the last mapped pixel cannot be written, nor can the rollback write it,
			// so page-marks.pending stays for the next Play.
			PageMarks::SetWriteHookForTesting([&](size_t pixel) {
				if (pixel == map.size() - 1) throw PromptError("simulated sheet write failure");
			});
			const auto failed = RejectText([&] { Apply(game, true); }, "a failed sheet write");
			PageMarks::SetWriteHookForTesting({});
			Check(failed.find("also failed") != std::string::npos && failed.find("global_pages.ctxr") != std::string::npos, "the failed sheet rollback is reported");
			Check(fs::exists(pending), "a write that failed after the sheet was open left no page-marks.pending");
			Check(PageMarks::ReadRegions(atlas) == marks, "the rollback did not restore the page marks");
			Check(Take(folder) == originals, "a failed page marks write left an icon changed");
			Apply(game, true);
			Check(PageMarks::ReadRegions(atlas) == drawn && !fs::exists(pending), "enable after a failed write did not relabel and remove page-marks.pending");
			Apply(game, false);
			Check(PageMarks::ReadRegions(atlas) == marks, "disable after a failed write did not restore the page marks");
			clear();
			// A write that fails but whose rollback works: nothing is left behind.
			size_t calls = 0;
			PageMarks::SetWriteHookForTesting([&](size_t pixel) {
				if (pixel == 10 && ++calls == 1) throw PromptError("simulated sheet write failure");
			});
			const auto undone = RejectText([&] { Apply(game, true); }, "a failed sheet write with a rollback");
			PageMarks::SetWriteHookForTesting({});
			Check(undone.find("also failed") == std::string::npos, "a rollback that worked was reported as failed");
			Check(!fs::exists(pending) && PageMarks::ReadRegions(atlas) == marks && Take(folder) == originals, "a rolled-back sheet write left something behind");
			clear();

			// No font to draw the marks with. Over an untouched sheet the icons still go ahead; over the launcher's own
			// marks the batch stops, as it does for the other things its own work needs.
			PageMarks::SetRenderHookForTesting([] { throw PromptError("No bold sans-serif TrueType font was found"); });
			stamp = age();
			Apply(game, true);
			Check(Take(folder) == installed, "a missing font stopped the icons");
			Check(fs::last_write_time(atlas) == stamp && PageMarks::ReadRegions(atlas) == marks && !fs::exists(marks_backup) && !fs::exists(pending) &&
			          !fs::exists(record),
			      "enable touched the sheet or created page mark files without a font");
			Apply(game, false);
			PageMarks::SetRenderHookForTesting({});
			Apply(game, true);
			PageMarks::SetRenderHookForTesting([] { throw PromptError("No bold sans-serif TrueType font was found"); });
			Check(RejectText([&] { Apply(game, true); }, "own marks without a font").find("font") != std::string::npos, "the missing font is named");
			Check(PageMarks::ReadRegions(atlas) == drawn && Take(folder) == installed, "a missing font changed the game");
			Apply(game, false);
			PageMarks::SetRenderHookForTesting({});
			Check(PageMarks::ReadRegions(atlas) == marks, "disable without a font did not restore the marks");
			clear();

			// Links. A linked sheet, a sheet below a linked folder and a hard-linked sheet are left alone; the icons go
			// ahead and the other name of the file keeps its bytes.
			fs::rename(atlas, root / "sheet-target");
			fs::create_symlink(root / "sheet-target", atlas);
			Apply(game, true);
			Check(Take(folder) == installed, "a linked sheet stopped the icons");
			Check(PageMarks::ReadRegions(root / "sheet-target") == marks && !fs::exists(marks_backup) && !fs::exists(pending) && !fs::exists(record),
			      "enable wrote through a linked sheet");
			Apply(game, false);
			fs::remove(atlas);
			fs::rename(root / "sheet-target", atlas);
			const fs::path sheet_folder = game_root / "ui" / "textures" / "_ps5";
			fs::rename(sheet_folder, root / "folder-target");
			fs::create_directory_symlink(root / "folder-target", sheet_folder);
			Apply(game, true);
			Check(Take(folder) == installed, "a linked folder stopped the icons");
			Check(PageMarks::ReadRegions(root / "folder-target" / "_ps5" / "global_pages.ctxr") == marks && !fs::exists(marks_backup),
			      "enable wrote through a linked folder");
			Apply(game, false);
			fs::remove(sheet_folder);
			fs::rename(root / "folder-target", sheet_folder);
			fs::create_hard_link(atlas, root / "sheet-twin");
			Apply(game, true);
			Check(Take(folder) == installed, "a hard-linked sheet stopped the icons");
			Check(PageMarks::ReadRegions(atlas) == marks && PageMarks::ReadRegions(root / "sheet-twin") == marks && !fs::exists(marks_backup) &&
			          !fs::exists(pending),
			      "enable wrote a hard-linked sheet");
			Apply(game, false);
			fs::remove(root / "sheet-twin");
			clear();
			// The launcher's own marks on a sheet that has since been linked: the sheet cannot be looked at, so the marks
			// are left as they are, and the icons are restored.
			Apply(game, true);
			fs::rename(atlas, root / "sheet-target");
			fs::create_symlink(root / "sheet-target", atlas);
			Apply(game, false);
			Check(Take(folder) == originals && PageMarks::ReadRegions(root / "sheet-target") == drawn, "a linked sheet with own marks stopped the icon restore or was written");
			fs::remove(atlas);
			fs::rename(root / "sheet-target", atlas);
			Apply(game, true);
			Apply(game, false);
			Check(PageMarks::ReadRegions(atlas) == marks, "restore after the sheet was linked and unlinked");
			clear();
			// A sheet that cannot be opened for writing (a user who may write it is not root, who may write anything).
			if (::geteuid() != 0) {
				fs::permissions(atlas, fs::perms::owner_write | fs::perms::group_write | fs::perms::others_write, fs::perm_options::remove);
				Apply(game, true);
				Check(Take(folder) == installed && PageMarks::ReadRegions(atlas) == marks && !fs::exists(marks_backup), "a read-only sheet stopped the icons or was written");
				Apply(game, false);
				fs::permissions(atlas, fs::perms::owner_write, fs::perm_options::add);
				clear();
			}

			// Folders of another case (a dump made on Windows).
			fs::rename(game_root / "ui", game_root / "UI");
			Apply(game, true);
			const fs::path upper = game_root / "UI" / "textures" / "_ps5" / "_ps5" / "global_pages.ctxr";
			Check(PageMarks::ReadRegions(upper) == drawn, "a sheet in folders of another case was not relabeled");
			Apply(game, false);
			Check(PageMarks::ReadRegions(upper) == marks, "a sheet in folders of another case was not restored");
			fs::rename(game_root / "UI", game_root / "ui");
			clear();

			// The Windows launcher's record: marks it wrote are the launcher's own here too.
			{
				std::ofstream lines(record);
				lines << Bytes::Sha256Hex(drawn) << "\r\n";
			}
			Write(marks_backup, marks);
			PageMarks::WriteRegions(atlas, drawn);
			Apply(game, false);
			Check(PageMarks::ReadRegions(atlas) == marks && Take(folder) == originals, "marks recorded by the other launcher were not restored");
			clear();

			// Whole-sheet scan, last. Every byte outside the mapped pixels still holds the fixture: 0xA5 in the 65-byte header,
			// zero everywhere else.
			std::vector<uint64_t> positions;
			for (const int32_t at: map)
				for (int k = 0; k < 4; ++k) positions.push_back(static_cast<uint64_t>(at) + static_cast<uint64_t>(k));
			std::sort(positions.begin(), positions.end());
			// (By runs between the mapped bytes, so that a build without optimisation does not take minutes over 320 MB.)
			uint64_t          at = 0, stray = UINT64_MAX;
			size_t            next = 0;
			std::ifstream     sheet(atlas, std::ios::binary);
			std::vector<char> chunk(1 << 20), zeros(1 << 20, 0);
			while (sheet.read(chunk.data(), static_cast<std::streamsize>(chunk.size())) || sheet.gcount() > 0) {
				const auto read = static_cast<size_t>(sheet.gcount());
				for (size_t pos = 0; pos < read;) {
					const uint64_t here = at + pos;
					if (next < positions.size() && positions[next] == here) {
						++next;
						++pos;
						continue;
					}
					size_t run = read - pos;
					if (next < positions.size() && positions[next] - here < run) run = static_cast<size_t>(positions[next] - here);
					for (size_t done = 0; done < run && stray == UINT64_MAX;) {
						const uint64_t from   = here + done;
						const size_t   length = from < 65 ? static_cast<size_t>(std::min<uint64_t>(run - done, 65 - from)) : std::min(run - done, zeros.size());
						const char*    bytes  = chunk.data() + pos + done;
						const bool     same   = from < 65 ? std::all_of(bytes, bytes + length, [](char c) { return static_cast<uint8_t>(c) == 0xA5; })
						                                  : std::memcmp(bytes, zeros.data(), length) == 0;
						if (!same) stray = from;
						done += length;
					}
					pos += run;
				}
				at += read;
			}
			Check(at == size && next == positions.size() && stray == UINT64_MAX, "a byte outside the mapped pixels changed (first at " + std::to_string(stray) + ")");
			PageMarks::SetOriginalForTesting(Buffer());
			fs::remove(atlas);
		}

		// A game that is not extracted (a ZArchive) is refused to enable, and ignored when off.
		Game zar {(root / "game.zar").string(), "", ""};
		Write(root / "game.zar", {1, 2, 3});
		Reject([&] { Apply(zar, true); }, "a ZArchive");
		Apply(zar, false);
		Game none {"", "", ""};
		Apply(none, false);
		Reject([&] { Apply(none, true); }, "no game");

		// Layout rules.
		KeyboardLayout rules;
		Check(rules.Pairs().size() == 21, "21 pairs without a sensitivity");
		rules.sensitivity = "0.05";
		try {
			rules.Validate();
			Check(false, "a sensitivity below 0.1 was accepted");
		} catch (const PromptError&) {}
		for (const char* good: {"0.1", "5", "2.", ".5", "1.25"}) {
			rules.sensitivity = good;
			rules.Validate();
		}
		for (const char* bad: {"5.1", "-1", "1e2", "1,5", "abc", ".", " 1"}) {
			rules.sensitivity = bad;
			try {
				rules.Validate();
				Check(false, std::string("sensitivity accepted: ") + bad);
			} catch (const PromptError&) {}
		}
		for (const char* good: {";", "'", "\\", "Keypad +", "F13", "F24", "Mouse:WheelUp", "Left Alt"})
			Check(KeyboardLayout::Supported(good), std::string("supported: ") + good);
		for (const char* bad: {"F7", "F8", "F11", "F25", "F0", "F05", "Pause", "a", "Keypad 55", "Mouse:Foo", ""})
			Check(!KeyboardLayout::Supported(bad), std::string("not supported: ") + bad);
		KeyboardLayout twice;
		twice.SetBinding("Circle", "E");
		try {
			twice.Validate();
			Check(false, "a duplicate input was accepted");
		} catch (const PromptError& error) {
			Check(std::string(error.what()).find("Duplicate binding: E") == 0, "the duplicate is named");
		}
		Check(KeyboardLayout::Label("Left Shift") == "SHIFT" && KeyboardLayout::Label("Keypad 5") == "NUM 5" &&
		          KeyboardLayout::Label("Mouse:WheelDown") == "WHEEL\nDOWN" && KeyboardLayout::Label("q") == "Q",
		      "labels");
		KeyboardLayout defaults;
		Check(defaults.IconLabel("ps4_stick_l.ctxr") == "W A\nS D" && defaults.IconLabel("ps4_dpad_ud") == "\xE2\x86\x91 \xE2\x86\x93", "icon labels");
		const auto pairs = defaults.Pairs();
		Check(pairs[0] == "Cross=E" && pairs[7] == "R1=Mouse:Left" && pairs.back() == "LeftStickRight=D", "pairs");

		std::printf("PASS (%d checks): default off, version guard, apply, idempotence, exact restore, partial recovery, mod/corrupt/missing backup protection, write-failure rollback, custom profile swaps/artwork/restore, links, case, locking, resampler, page marks apply/idempotence/restore/foreign sheet/wrong size/missing sheet/missing or damaged backup/custom Square/failed write after open/no font/links/read-only/case/whole-sheet scan, layout rules\n", g_checks);
	} catch (const std::exception& error) {
		std::printf("EXCEPTION: %s\n", error.what());
		fs::remove_all(root);
		return 1;
	}
	fs::remove_all(root);
	return 0;
}

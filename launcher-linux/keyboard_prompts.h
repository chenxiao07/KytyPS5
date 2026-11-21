#ifndef KYTY_LAUNCHER_LINUX_KEYBOARD_PROMPTS_H
#define KYTY_LAUNCHER_LINUX_KEYBOARD_PROMPTS_H

#include "bytes.h"
#include "keyboard_layout.h"

#include <functional>
#include <string>
#include <vector>

// The keyboard and mouse prompts (launcher-windows' KeyboardPrompts.cs): 22 button icons of an
// extracted Demon's Souls (PPSA01341 01.007.000) game folder, ui/textures/buttonicons/_ps5, swapped
// for key caps on Play and put back when the option is off. Only artwork: never launch.json,
// saves or shader caches. Every original is verified by its SHA-256 and kept in
// .kyty-keyboard-prompts-v1 inside the game folder (the same folder and files as the Windows
// launcher's, so a game folder used from both keeps working); a modified, missing or damaged
// icon stops the whole batch before anything is written.
//
// The square and triangle page-switch marks of the UI sheet (ui/textures/_ps5/_ps5/global_pages.ctxr,
// see page_marks.h) are relabelled in place in the same batch, and put back the same way. A sheet
// the launcher has not written to never stops a Play the icons alone would allow (it is absent,
// linked, of another size, unreadable, holds other bytes in the two spots, is hard-linked or not
// writable, or there is no font to draw the marks with): it is left alone. Marks the launcher did
// write, or a write it did not finish (page-marks.pending), need the saved original bytes
// (page-marks.bin) and a sheet it can write, or the batch stops.
namespace KeyboardPrompts {

struct Game {
	std::string path;     // the game folder
	std::string title_id; // from its param.json
	std::string version;
};

// Puts the prompts in (enabled; with the layout's key caps) or the originals back. Throws
// PromptError: what is wrong, and how to recover a partly applied batch.
void Apply(const Game& game, bool enabled, const KeyboardLayout* layout = nullptr);

// One icon of the pack: its file name, the SHA-256 of the game's original and of the pack's
// replacement (a manifest.txt line), and the replacement.
struct Icon {
	std::string        name;
	std::string        original_hash;
	std::string        replacement_hash;
	Bytes::Buffer      replacement;
};

// For tests only: the icons used instead of the pack's (a test has no original game files to
// match the manifest's hashes), and a function called before each file is written (it may throw
// PromptError to fail that write); both reset by an empty list / empty function.
void SetIconsForTesting(std::vector<Icon> icons);
void SetWriteHookForTesting(std::function<void(const std::string& path)> hook);

// The pack's icons as manifest.txt lists them.
std::vector<Icon> PackIcons();

} // namespace KeyboardPrompts

#endif

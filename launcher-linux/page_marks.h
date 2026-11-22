#ifndef KYTY_LAUNCHER_LINUX_PAGE_MARKS_H
#define KYTY_LAUNCHER_LINUX_PAGE_MARKS_H

#include "bytes.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

// The square and triangle page-switch marks at the bottom of the inventory and status panels
// (launcher-windows' PageMarks.cs): two 66x66 R8G8B8A8 sprites inside the UI sheet named in
// page-marks.txt, rewritten in place at the file offsets listed in page-marks-offsets.bin (square
// region then triangle region, row-major). Both files are the Windows launcher's own, linked in
// with the rest of the pack (prompt_assets.cpp). The files kept beside the icons' originals in
// .kyty-keyboard-prompts-v1 are the Windows launcher's too: page-marks.bin (the original 34,848
// bytes), page-marks.generated (the SHA-256 of each version written, one per line) and
// page-marks.pending (empty: an in-place write that may not have finished).
namespace PageMarks {

constexpr int Side = 66;

inline constexpr const char* BackupName  = "page-marks.bin";
inline constexpr const char* PendingName = "page-marks.pending";
inline constexpr const char* RecordName  = "page-marks.generated";

// The sheet's path inside the game folder ("ui/textures/_ps5/_ps5/global_pages.ctxr"), its size,
// and the SHA-256 of the two original regions. Throw PromptError when the pack's files are damaged.
const std::string& RelativePath();
uint64_t           ExpectedSize();
const std::string& OriginalHash();

// The file offset of each byte of the two regions (2 * 66 * 66 * 4 entries, square first).
const std::vector<int32_t>& Offsets();

// True when an untouched sheet can be written in place: it opens for writing and has no second
// name on the disk (a hard link), since an in-place write would change that other file too.
bool Startable(const std::filesystem::path& sheet);

// True when the sheet, or a folder between it and the game folder `root`, is a symbolic link.
bool Linked(const std::filesystem::path& root, const std::filesystem::path& sheet);

// The two marks for a pair of labels: each drawn like a 128-pixel icon and resampled to 66x66;
// R,G,B,A, square then triangle. Throws PromptError when no font can draw a label.
Bytes::Buffer Render(const std::string& square_label, const std::string& triangle_label);

// Only a read of the two regions (8712 pixels). Throws PromptError when the sheet cannot be read or
// has another size.
Bytes::Buffer ReadRegions(const std::filesystem::path& sheet);

// One pass over the open sheet, flushed to disk. "opened" runs once the sheet is open for writing
// and before its first byte changes. A failed batch restores the previous bytes the same way.
// Throws PromptError.
void WriteRegions(const std::filesystem::path& sheet, const Bytes::Buffer& bytes,
                  const std::function<void()>& opened = {});

// For tests only: the original regions to expect instead of the pack's (a test has no original
// sheet to match the pack's hash; empty resets), a function called before each pixel is written
// (it may throw PromptError to fail that write), and one called before the marks are drawn (it may
// throw PromptError, as a PC without a usable font does); the last two reset by an empty function.
void SetOriginalForTesting(const Bytes::Buffer& original);
void SetWriteHookForTesting(std::function<void(size_t pixel)> hook);
void SetRenderHookForTesting(std::function<void()> hook);

} // namespace PageMarks

#endif

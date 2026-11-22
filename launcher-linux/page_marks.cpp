#include "page_marks.h"

#include "keyboard_layout.h"
#include "prompt_assets.h"
#include "prompt_texture.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <set>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

namespace fs = std::filesystem;

namespace PageMarks {

namespace {

using Bytes::Buffer;

constexpr size_t Regions        = 2;
constexpr size_t FirstPixelByte = 65; // the header; every pixel is 4 bytes

std::string                      g_original_hash; // (a test's; empty: the pack's)
std::function<void(size_t)>      g_write_hook;
std::function<void()>            g_render_hook;

struct Pack {
	std::vector<int32_t> offsets;
	std::string          path, hash;
	uint64_t             size = 0;
};

Buffer Resource(const char* name) {
	const auto* asset = PromptAssets::Find(name);
	if (asset == nullptr) throw PromptError(std::string("Missing prompt texture resource: ") + name);
	std::string error;
	auto        bytes = Bytes::Gunzip(asset->data, asset->size, &error);
	if (!error.empty()) throw PromptError(std::string("Damaged prompt texture resource: ") + name);
	return bytes;
}

bool IsHex(const std::string& text) {
	return std::all_of(text.begin(), text.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

// The relative path is checked before any path call: lowercase letters, digits, '_', '.' and '/' only, no empty
// part, no "..".
bool SafeRelative(const std::string& text) {
	if (text.empty() || text.front() == '/' || text.back() == '/') return false;
	if (!std::all_of(text.begin(), text.end(), [](char c) {
		    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '/';
	    }))
		return false;
	size_t start = 0;
	while (start <= text.size()) {
		size_t end = text.find('/', start);
		if (end == std::string::npos) end = text.size();
		const auto part = text.substr(start, end - start);
		if (part.empty() || part == "..") return false;
		start = end + 1;
	}
	return true;
}

// One line: relative sheet path, expected file size, SHA-256 of the original region bytes.
Pack LoadPack() {
	const auto  raw = Resource("page-marks.txt.gz");
	std::string line(raw.begin(), raw.end());
	while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
	std::vector<std::string> fields;
	size_t                   from = 0;
	for (size_t at; (at = line.find(' ', from)) != std::string::npos; from = at + 1)
		fields.push_back(line.substr(from, at - from));
	fields.push_back(line.substr(from));
	if (fields.size() != 3 || !SafeRelative(fields[0]) || fields[1].empty() || fields[1].size() > 18 ||
	    !std::all_of(fields[1].begin(), fields[1].end(), [](char c) { return c >= '0' && c <= '9'; }) || fields[2].size() != 64 ||
	    !IsHex(fields[2]))
		throw PromptError("Invalid page marks resource");
	Pack pack;
	pack.path = fields[0];
	pack.size = std::strtoull(fields[1].c_str(), nullptr, 10);
	pack.hash = fields[2];

	const auto data = Resource("page-marks-offsets.bin.gz");
	if (data.size() != Regions * Side * Side * 4) throw PromptError("Invalid page marks layout");
	pack.offsets.resize(Regions * Side * Side);
	std::memcpy(pack.offsets.data(), data.data(), data.size()); // (little-endian, as on Windows)
	const std::set<int32_t> seen(pack.offsets.begin(), pack.offsets.end());
	if (seen.size() != pack.offsets.size()) throw PromptError("Invalid page marks pixel offset");
	for (const int32_t at: pack.offsets)
		if (at < static_cast<int32_t>(FirstPixelByte) || (at - static_cast<int32_t>(FirstPixelByte)) % 4 != 0 ||
		    static_cast<uint64_t>(at) + 4 > pack.size)
			throw PromptError("Invalid page marks pixel offset");
	return pack;
}

// Loaded once, on first use, so a damaged resource is reported as a PromptError like other prompt failures.
const Pack& Data() {
	static const Pack pack = LoadPack();
	return pack;
}

struct Fd {
	int fd = -1;
	~Fd() {
		if (fd >= 0) ::close(fd);
	}
};

std::string Errno(int code) {
	return std::strerror(code);
}

PromptError UnexpectedSize() {
	return PromptError("Unrecognized or modified prompt: global_pages.ctxr has an unexpected size.");
}

// The sheet opened as `flags` says, never through a link, and of the expected size.
int OpenSheet(const fs::path& path, int flags, const char* what) {
	const int fd = ::open(path.c_str(), flags | O_CLOEXEC | O_NOFOLLOW);
	if (fd < 0) throw PromptError(std::string("Cannot ") + what + " " + path.string() + ": " + Errno(errno));
	struct stat info {};
	if (::fstat(fd, &info) != 0) {
		const int code = errno;
		::close(fd);
		throw PromptError(std::string("Cannot ") + what + " " + path.string() + ": " + Errno(code));
	}
	if (!S_ISREG(info.st_mode) || static_cast<uint64_t>(info.st_size) != Data().size) {
		::close(fd);
		throw UnexpectedSize();
	}
	return fd;
}

} // namespace

const std::string& RelativePath() {
	return Data().path;
}

uint64_t ExpectedSize() {
	return Data().size;
}

const std::string& OriginalHash() {
	return g_original_hash.empty() ? Data().hash : g_original_hash;
}

const std::vector<int32_t>& Offsets() {
	return Data().offsets;
}

bool Startable(const fs::path& sheet) {
	const int fd = ::open(sheet.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
	if (fd < 0) return false;
	Fd         guard {fd};
	struct stat info {};
	return ::fstat(fd, &info) == 0 && S_ISREG(info.st_mode) && info.st_nlink == 1;
}

bool Linked(const fs::path& root, const fs::path& sheet) {
	const auto relative = sheet.lexically_relative(root);
	if (relative.empty() || *relative.begin() == "..") return true; // (not inside the game folder: not to be touched)
	std::error_code error;
	fs::path        current = root;
	for (const auto& part: relative) {
		if (part == ".") continue;
		current /= part;
		const auto status = fs::symlink_status(current, error);
		if (error) break; // (does not exist)
		if (fs::is_symlink(status)) return true;
	}
	return false;
}

Buffer Render(const std::string& square_label, const std::string& triangle_label) {
	if (g_render_hook) g_render_hook();
	Buffer result;
	result.reserve(Regions * Side * Side * 4);
	for (const auto* label: {&square_label, &triangle_label}) {
		const auto mark = PromptTexture::Resampled(PromptTexture::Draw(*label), Side);
		result.insert(result.end(), mark.begin(), mark.end());
	}
	return result;
}

Buffer ReadRegions(const fs::path& sheet) {
	const auto& offsets = Data().offsets;
	Fd          file {OpenSheet(sheet, O_RDONLY, "read")};
	Buffer      result(offsets.size() * 4);
	for (size_t i = 0; i < offsets.size(); ++i) {
		size_t at = 0;
		while (at < 4) {
			const auto n = ::pread(file.fd, result.data() + i * 4 + at, 4 - at, static_cast<off_t>(offsets[i]) + static_cast<off_t>(at));
			if (n < 0) {
				if (errno == EINTR) continue;
				throw PromptError("Cannot read " + sheet.string() + ": " + Errno(errno));
			}
			if (n == 0) throw UnexpectedSize();
			at += static_cast<size_t>(n);
		}
	}
	return result;
}

void WriteRegions(const fs::path& sheet, const Buffer& bytes, const std::function<void()>& opened) {
	const auto& offsets = Data().offsets;
	if (bytes.size() != offsets.size() * 4) throw PromptError("Invalid page marks data");
	Fd file {OpenSheet(sheet, O_RDWR, "write")};
	if (opened) opened();
	for (size_t i = 0; i < offsets.size(); ++i) {
		if (g_write_hook) g_write_hook(i);
		size_t at = 0;
		while (at < 4) {
			const auto n = ::pwrite(file.fd, bytes.data() + i * 4 + at, 4 - at, static_cast<off_t>(offsets[i]) + static_cast<off_t>(at));
			if (n < 0) {
				if (errno == EINTR) continue;
				throw PromptError("Cannot write " + sheet.string() + ": " + Errno(errno));
			}
			at += static_cast<size_t>(n);
		}
	}
	if (::fsync(file.fd) != 0) throw PromptError("Cannot flush " + sheet.string() + ": " + Errno(errno));
	const int fd = file.fd;
	file.fd      = -1;
	if (::close(fd) != 0) throw PromptError("Cannot write " + sheet.string() + ": " + Errno(errno));
}

void SetOriginalForTesting(const Buffer& original) {
	g_original_hash = original.empty() ? std::string() : Bytes::Sha256Hex(original);
}

void SetWriteHookForTesting(std::function<void(size_t pixel)> hook) {
	g_write_hook = std::move(hook);
}

void SetRenderHookForTesting(std::function<void()> hook) {
	g_render_hook = std::move(hook);
}

} // namespace PageMarks

#ifndef EMULATOR_SRC_LOCAL_LIVE_SWITCH_H_
#define EMULATOR_SRC_LOCAL_LIVE_SWITCH_H_
// Named integer switches a running emulator changes without a restart: the KYTY_LIVE_FILE command `set <name> <value>`
// (live-control.h), for same-process A/B tests and screenshots of each variant. A switch is made by its first Find,
// with the value of the environment variable KYTY_<NAME> (upper case) when set, else `fallback`; the code that reads
// it keeps the reference and loads it where it decides.
#include <array>
#include <cstdio>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace LiveSwitch {

struct Entry {
	char                 name[48] {};
	std::atomic<int64_t> value {0};
};

// (256: 64 overflowed at 81-87 names, every later name sharing one atomic.)
inline std::array<Entry, 256>& Entries() {
	static std::array<Entry, 256> entries;
	return entries;
}
inline std::mutex& Mutex() {
	static std::mutex mutex;
	return mutex;
}

// The switch `name` (made with the environment's or `fallback`'s value the first time).
inline std::atomic<int64_t>& Find(const char* name, int64_t fallback) {
	std::lock_guard lock(Mutex());
	for (auto& entry: Entries()) {
		if (entry.name[0] == '\0') {
			std::strncpy(entry.name, name, sizeof(entry.name) - 1);
			std::string variable = "KYTY_";
			for (const char* c = name; *c != '\0'; ++c) variable += static_cast<char>(std::toupper(static_cast<unsigned char>(*c)));
			const char* env = std::getenv(variable.c_str());
			entry.value.store(env != nullptr && *env != '\0' ? std::strtoll(env, nullptr, 10) : fallback,
			                  std::memory_order_relaxed);
			return entry.value;
		}
		if (std::strcmp(entry.name, name) == 0) return entry.value;
	}
	// (Full: said once a name, loudly; the name gets an atomic of its own that no `set` reaches, found again by name.)
	static std::vector<std::pair<std::string, std::unique_ptr<std::atomic<int64_t>>>> overflow;
	for (auto& [overflow_name, value]: overflow)
		if (overflow_name == name) return *value;
	std::fprintf(stderr, "LiveSwitch OVERFLOW: %s (table full)\n", name);
	std::printf("LiveSwitch OVERFLOW: %s (table full)\n", name);
	return *overflow.emplace_back(name, std::make_unique<std::atomic<int64_t>>(fallback)).second;
}

// `set`: false when no switch has that name (yet).
inline bool Set(const char* name, int64_t value) {
	std::lock_guard lock(Mutex());
	for (auto& entry: Entries()) {
		if (entry.name[0] == '\0') return false;
		if (std::strcmp(entry.name, name) == 0) {
			entry.value.store(value, std::memory_order_relaxed);
			return true;
		}
	}
	return false;
}

} // namespace LiveSwitch

#endif // EMULATOR_SRC_LOCAL_LIVE_SWITCH_H_

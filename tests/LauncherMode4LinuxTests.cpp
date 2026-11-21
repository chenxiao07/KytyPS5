// SPDX-License-Identifier: GPL-2.0-or-later
// The Linux launcher's parallel translation switch (launcher-linux/mode4.h), the counterpart of LauncherMode4Tests.cs:
// off by default, kept in launcher-settings.json, offered only where the emulator would not stand mode 4 down (its own
// gate, with its tuning and simulation variables), KYTY_SPECULATE=4 in Play's variables only when on and offered (an
// extra option's pair winning), its memory figure stated, its text in English and Chinese.
//
//   cmake --build <build> --target kyty_launcher_mode4_tests && ctest -R launcher_mode4
#include "commands.h"
#include "game.h"
#include "mode4.h"
#include "platform.h"
#include "settings.h"
#include "strings.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

static int g_checks = 0;
static void Check(bool value, const std::string& message) {
	++g_checks;
	if (!value) {
		std::printf("FAIL: %s\n", message.c_str());
		std::exit(1);
	}
}

static constexpr uint64_t GiB = uint64_t {1} << 30u;

// The gate with these variables (the rest not set).
static Mode4::Verdict Gate(uint64_t ram, const std::map<std::string, std::string>& variables = {}) {
	return Mode4::Check(ram, [&](const char* name) {
		const auto it = variables.find(name);
		return it != variables.end() ? it->second : std::string();
	});
}

// Play's variables (Commands::Play) on a PC with this memory.
static std::vector<std::pair<std::string, std::string>> PlaySet(const fs::path& root, bool on, const std::string& extra,
                                                                uint64_t ram) {
	static int      plays = 0;
	LauncherContext context {LauncherContext::Bare {}};
	context.root     = root;
	context.settings = std::make_unique<Settings>(root / ("play-" + std::to_string(++plays) + ".json"));
	context.settings->read_only = true;
	context.settings->Set("mode4", on);
	context.settings->Set("extra_args", extra);
	context.ram = ram;
	return Commands(context).Play(GameInfo {}, true).set;
}

static const std::string* Value(const std::vector<std::pair<std::string, std::string>>& set, const std::string& key) {
	for (const auto& [name, value]: set)
		if (name == key) return &value;
	return nullptr;
}

int main() {
	const fs::path root = fs::temp_directory_path() / ("kyty-mode4-test-" + std::to_string(getpid()));
	fs::remove_all(root);
	fs::create_directories(root);
	for (const char* name: {"KYTY_SIMULATE_RAM_MB", "KYTY_SIMULATE_PCORES", "KYTY_SPECULATE_RAM_GB", "KYTY_SPECULATE_THREADS",
	                        "KYTY_SPECULATE_CPUS", "KYTY_SPECULATE"})
		unsetenv(name);
	const auto path = root / "launcher-settings.json";

	// Off by default, in a new settings file and in one written before the switch existed.
	Check(!Settings(path).Bool("mode4"), "mode4 on by default");
	std::ofstream(path) << "{\"fps30\": true}";
	Check(!Settings(path).Bool("mode4"), "mode4 on in a file without it");
	// Kept: on, saved, read back; off again.
	{
		Settings settings(path);
		settings.Set("mode4", true);
		Check(settings.Save(), "save");
		Check(Settings(path).Bool("mode4"), "mode4 not kept on");
		Check(Settings(path).Bool("fps30"), "another key lost");
		settings.Set("mode4", false);
		Check(settings.Save(), "save");
		Check(!Settings(path).Bool("mode4"), "mode4 not kept off");
	}

	// The memory figure is stated, not a placeholder.
	Check(std::strcmp(Mode4::ExtraMemoryGb, "?") != 0 && *Mode4::ExtraMemoryGb != '\0', "the memory figure is still a placeholder");

	// The emulator's gate: memory first (28 GiB; 0, a failed read, stands down too); the performance cores only when a
	// count is given (the Linux emulator counts none: KYTY_SIMULATE_PCORES), 3 + 4 workers + 2.
	Check(Gate(28 * GiB).gate == Mode4::Gate::Available, "28 GiB");
	Check(Gate(28 * GiB - 1).gate == Mode4::Gate::LowRam, "under 28 GiB");
	Check(Gate(0).gate == Mode4::Gate::LowRam, "memory not read");
	Check(Gate(32 * GiB).gate == Mode4::Gate::Available, "no CPU count: not checked");
	Check(Gate(32 * GiB, {{"KYTY_SIMULATE_PCORES", "8"}}).gate == Mode4::Gate::FewCores, "8 CPUs");
	Check(Gate(32 * GiB, {{"KYTY_SIMULATE_PCORES", "9"}}).gate == Mode4::Gate::Available, "9 CPUs");
	Check(Gate(32 * GiB, {{"KYTY_SIMULATE_PCORES", "0"}}).gate == Mode4::Gate::Available, "simulated 0: not checked");
	// Its tuning variables, read as the emulator reads them.
	Check(Gate(16 * GiB, {{"KYTY_SPECULATE_RAM_GB", "0"}}).gate == Mode4::Gate::Available, "RAM_GB 0 lifts the gate");
	Check(Gate(16 * GiB, {{"KYTY_SPECULATE_RAM_GB", "12"}}).gate == Mode4::Gate::Available, "RAM_GB 12");
	Check(Gate(16 * GiB, {{"KYTY_SPECULATE_RAM_GB", "12x"}}).ram_gib == 28, "RAM_GB not whole digits: 28");
	Check(Gate(16 * GiB, {{"KYTY_SPECULATE_RAM_GB", "5000"}}).ram_gib == 28, "RAM_GB over 4096: 28");
	Check(Gate(32 * GiB, {{"KYTY_SIMULATE_PCORES", "7"}, {"KYTY_SPECULATE_THREADS", "2"}}).gate == Mode4::Gate::Available,
	      "2 workers need 7");
	Check(Gate(32 * GiB, {{"KYTY_SPECULATE_THREADS", "8"}}).cpus_needed == 13, "8 workers need 13");
	Check(Gate(32 * GiB, {{"KYTY_SPECULATE_THREADS", "13"}}).cpus_needed == 9, "13 workers: not taken");
	Check(Gate(32 * GiB, {{"KYTY_SIMULATE_PCORES", "4"}, {"KYTY_SPECULATE_CPUS", "8-15"}}).gate == Mode4::Gate::Available,
	      "CPUS lifts the CPU check");
	Check(Gate(64 * GiB, {{"KYTY_SIMULATE_RAM_MB", "16384"}}).gate == Mode4::Gate::LowRam, "simulated memory");
	Check(Gate(64 * GiB, {{"KYTY_SIMULATE_RAM_MB", "40000MB"}}).gate == Mode4::Gate::Available, "leading digits");
	Check(Gate(64 * GiB, {{"KYTY_SIMULATE_RAM_MB", "none"}}).gate == Mode4::Gate::LowRam, "no digits: 0");

	// Play: KYTY_SPECULATE=4 alone added when on and offered; nothing when off or not offered; an extra option's pair
	// wins, and its tuning variables are the gate's.
	const auto off = PlaySet(root, false, "", 32 * GiB);
	const auto on  = PlaySet(root, true, "", 32 * GiB);
	Check(Value(off, "KYTY_SPECULATE") == nullptr, "off sets KYTY_SPECULATE");
	Check(Value(on, "KYTY_SPECULATE") != nullptr && *Value(on, "KYTY_SPECULATE") == "4", "on does not set KYTY_SPECULATE=4");
	Check(on.size() == off.size() + 1, "on sets more than KYTY_SPECULATE");
	for (const auto& [key, value]: off)
		Check(Value(on, key) != nullptr && *Value(on, key) == value, "on changes " + key);
	Check(Value(PlaySet(root, true, "", 16 * GiB), "KYTY_SPECULATE") == nullptr, "too little memory sets it");
	Check(Value(PlaySet(root, true, "KYTY_SIMULATE_PCORES=6", 32 * GiB), "KYTY_SPECULATE") == nullptr,
	      "too few CPUs (the options' KYTY_SIMULATE_PCORES) set it");
	const auto own = PlaySet(root, true, "KYTY_SPECULATE=0", 32 * GiB);
	Check(Value(own, "KYTY_SPECULATE") != nullptr && *Value(own, "KYTY_SPECULATE") == "0", "the user's own pair lost");
	const auto lifted = PlaySet(root, true, "-Set KYTY_SPECULATE_RAM_GB=0", 16 * GiB);
	Check(Value(lifted, "KYTY_SPECULATE") != nullptr && *Value(lifted, "KYTY_SPECULATE") == "4",
	      "the options' RAM_GB=0 not taken by the gate");
	Check(Value(PlaySet(root, true, "KYTY_SIMULATE_RAM_MB=", 32 * GiB), "KYTY_SPECULATE") == nullptr,
	      "the options' empty KYTY_SIMULATE_RAM_MB (0, as Play makes it) not taken by the gate");
	// The launcher's environment (the game inherits it) is the gate's too, and an extra option's pair beats it.
	setenv("KYTY_SIMULATE_RAM_MB", "16384", 1);
	Check(Value(PlaySet(root, true, "", 32 * GiB), "KYTY_SPECULATE") == nullptr, "the environment not taken by the gate");
	Check(Value(PlaySet(root, true, "KYTY_SIMULATE_RAM_MB=65536", 32 * GiB), "KYTY_SPECULATE") != nullptr,
	      "the extra option's pair does not beat the environment");
	unsetenv("KYTY_SIMULATE_RAM_MB");

	// The text: what it does with the memory figure, or why it is not offered, in English and Chinese, the Chinese its
	// own (T falls back to English for a key Chinese lacks).
	const char* keys[] = {"mode4", "mode4.desc", "mode4.lowRam", "mode4.fewCores", "summary.mode4", "experimental.translation"};
	std::map<std::string, std::string> english;
	Strings::Current = Strings::Language::English;
	for (const char* key: keys) {
		english[key] = Strings::T(key);
		Check(english[key] != key, std::string("English: ") + key + " missing");
	}
	Strings::Current = Strings::Language::Chinese;
	for (const char* key: keys)
		Check(english[key] != Strings::T(key), std::string("Chinese: ") + key + " is not its own");
	for (const auto language: {Strings::Language::English, Strings::Language::Chinese}) {
		Strings::Current     = language;
		const std::string at = language == Strings::Language::English ? "English: " : "Chinese: ";
		const auto desc      = Strings::F("mode4.desc", {Mode4::ExtraMemoryGb});
		Check(desc.find(std::string(Mode4::ExtraMemoryGb) + " GB") != std::string::npos &&
		          desc.find("KYTY_SPECULATE=4") != std::string::npos,
		      at + "desc");
		Check(Strings::F("mode4.lowRam", {"28"}).find("28 GB") != std::string::npos, at + "lowRam");
		const auto cores = Strings::F("mode4.fewCores", {"9", "6"});
		Check(cores.find('9') != std::string::npos && cores.find('6') != std::string::npos, at + "fewCores");
	}
	Strings::Current = Strings::Language::English;

	fs::remove_all(root);
	std::printf("Launcher mode 4 tests passed (%d checks).\n", g_checks);
	return 0;
}

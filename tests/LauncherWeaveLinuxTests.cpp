// SPDX-License-Identifier: GPL-2.0-or-later
// The Linux launcher's weave switch (launcher-linux/weave.h), the counterpart of LauncherWeaveTests.cs: off by default,
// kept in launcher-settings.json, offered only where the emulator would not stand the weave down (its own gate: mode 4,
// the performance cores, the switches it needs), KYTY_WEAVE=1 in Play's variables only when on, offered and no mode 4 is
// set (an extra option's pair winning), its memory figure stated, its text in English and Chinese.
//
//   cmake --build <build> --target kyty_launcher_weave_tests && ctest -R launcher_weave
#include "commands.h"
#include "game.h"
#include "mode4.h"
#include "platform.h"
#include "settings.h"
#include "strings.h"
#include "weave.h"

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

// The weave's gate with mode 4's on the same variables (the rest not set), on a PC with this memory and these CPUs.
static Weave::Verdict Gate(uint64_t ram, uint32_t performance, uint32_t logical,
                           const std::map<std::string, std::string>& variables = {}) {
	const auto variable = [&](const char* name) {
		const auto it = variables.find(name);
		return it != variables.end() ? it->second : std::string();
	};
	return Weave::Check(Mode4::Check(ram, variable), performance, logical, variable);
}

// Play's variables (Commands::Play) on a PC with this memory and these performance cores.
static std::vector<std::pair<std::string, std::string>> PlaySet(const fs::path& root, bool mode4, bool weave,
                                                                const std::string& extra, uint64_t ram, uint32_t cpus) {
	static int      plays = 0;
	LauncherContext context {LauncherContext::Bare {}};
	context.root     = root;
	context.settings = std::make_unique<Settings>(root / ("play-" + std::to_string(++plays) + ".json"));
	context.settings->read_only = true;
	context.settings->Set("mode4", mode4);
	context.settings->Set("weave", weave);
	context.settings->Set("extra_args", extra);
	context.ram              = ram;
	context.performance_cpus = cpus;
	return Commands(context).Play(GameInfo {}, true).set;
}

static const std::string* Value(const std::vector<std::pair<std::string, std::string>>& set, const std::string& key) {
	for (const auto& [name, value]: set)
		if (name == key) return &value;
	return nullptr;
}

static bool Is(const std::string* value, const char* text) {
	return value != nullptr && *value == text;
}

int main() {
	const fs::path root = fs::temp_directory_path() / ("kyty-weave-test-" + std::to_string(getpid()));
	fs::remove_all(root);
	fs::create_directories(root);
	for (const char* name: {"KYTY_SIMULATE_RAM_MB", "KYTY_SIMULATE_PCORES", "KYTY_SPECULATE_RAM_GB", "KYTY_SPECULATE_THREADS",
	                        "KYTY_SPECULATE_CPUS", "KYTY_SPECULATE", "KYTY_WEAVE", "KYTY_WEAVE_WORKERS", "KYTY_SCOUT",
	                        "KYTY_VULKAN_RECORDING", "KYTY_TABLE_XPR", "KYTY_NATIVE_XPR"})
		unsetenv(name);
	const auto path = root / "launcher-settings.json";

	// Off by default, in a new settings file and in one written before the switch existed; kept on and off.
	Check(!Settings(path).Bool("weave"), "weave on by default");
	std::ofstream(path) << "{\"mode4\": true}";
	Check(!Settings(path).Bool("weave"), "weave on in a file without it");
	{
		Settings settings(path);
		settings.Set("weave", true);
		Check(settings.Save(), "save");
		Check(Settings(path).Bool("weave"), "weave not kept on");
		Check(Settings(path).Bool("mode4"), "another key lost");
		settings.Set("weave", false);
		Check(settings.Save(), "save");
		Check(!Settings(path).Bool("weave"), "weave not kept off");
	}

	// The memory figure is stated, not a placeholder.
	Check(std::strcmp(Weave::ExtraMemoryMb, "?") != 0 && *Weave::ExtraMemoryMb != '\0', "the memory figure is still a placeholder");

	// The performance cores: 10 logical CPUs (8 kept, 2 workers); every logical CPU where the classes are not known;
	// KYTY_WEAVE_WORKERS 1 to 8 lifting the check, 0 not; KYTY_SIMULATE_PCORES not read (CoreCounts).
	Check(Gate(16 * GiB, 10, 32).gate == Weave::Gate::Available, "10 CPUs");
	const auto nine = Gate(16 * GiB, 9, 32);
	Check(nine.gate == Weave::Gate::FewCores && nine.cpus == 9, "9 CPUs");
	Check(Gate(16 * GiB, 0, 12).gate == Weave::Gate::Available, "classes not known: 12 logical");
	Check(Gate(16 * GiB, 0, 8).gate == Weave::Gate::FewCores, "classes not known: 8 logical");
	Check(Gate(16 * GiB, 6, 12, {{"KYTY_WEAVE_WORKERS", "4"}}).gate == Weave::Gate::Available, "WEAVE_WORKERS lifts it");
	Check(Gate(16 * GiB, 6, 12, {{"KYTY_WEAVE_WORKERS", "0"}}).gate == Weave::Gate::FewCores, "WEAVE_WORKERS 0 counts");
	Check(Gate(16 * GiB, 6, 12, {{"KYTY_SIMULATE_PCORES", "16"}}).gate == Weave::Gate::FewCores, "SIMULATE_PCORES not read");
	// No memory gate: 8 GiB is enough.
	Check(Gate(8 * GiB, 16, 32).gate == Weave::Gate::Available, "a memory gate");
	// Mode 4 in the variables: it stands the weave down where mode 4 would run, not where mode 4 stands down itself.
	Check(Gate(32 * GiB, 16, 32, {{"KYTY_SPECULATE", "4"}}).gate == Weave::Gate::Mode4, "KYTY_SPECULATE=4");
	Check(Gate(16 * GiB, 16, 32, {{"KYTY_SPECULATE", "4"}}).gate == Weave::Gate::Available, "mode 4 stood down for memory");
	Check(Gate(32 * GiB, 16, 32, {{"KYTY_SPECULATE", "0"}}).gate == Weave::Gate::Available, "KYTY_SPECULATE=0");
	// The switches it needs on, the first found named.
	const auto scout = Gate(16 * GiB, 16, 32, {{"KYTY_SCOUT", "0"}, {"KYTY_TABLE_XPR", "0"}});
	Check(scout.gate == Weave::Gate::Needs && scout.needs == "KYTY_SCOUT", "KYTY_SCOUT=0");
	Check(Gate(16 * GiB, 16, 32, {{"KYTY_NATIVE_XPR", "0"}}).needs == "KYTY_NATIVE_XPR", "KYTY_NATIVE_XPR=0");
	Check(Gate(16 * GiB, 16, 32, {{"KYTY_SCOUT", "2"}}).gate == Weave::Gate::Available, "KYTY_SCOUT=2");
	Check(Gate(16 * GiB, 9, 32, {{"KYTY_SCOUT", "0"}}).gate == Weave::Gate::FewCores, "the CPU check before the scout's");

	// Play: KYTY_WEAVE=1 alone added when on and offered; nothing when off or not offered; mode 4 set instead when both
	// are on and mode 4 is offered (the emulator's order), the weave where mode 4 is not; the user's own pair wins.
	const auto off = PlaySet(root, false, false, "", 16 * GiB, 16);
	const auto on  = PlaySet(root, false, true, "", 16 * GiB, 16);
	Check(Value(off, "KYTY_WEAVE") == nullptr, "off sets KYTY_WEAVE");
	Check(Is(Value(on, "KYTY_WEAVE"), "1"), "on does not set KYTY_WEAVE=1");
	Check(on.size() == off.size() + 1, "on sets more than KYTY_WEAVE");
	for (const auto& [key, value]: off)
		Check(Value(on, key) != nullptr && *Value(on, key) == value, "on changes " + key);
	Check(Value(PlaySet(root, false, true, "", 16 * GiB, 9), "KYTY_WEAVE") == nullptr, "too few CPUs set it");
	const auto both = PlaySet(root, true, true, "", 32 * GiB, 16);
	Check(Is(Value(both, "KYTY_SPECULATE"), "4") && Value(both, "KYTY_WEAVE") == nullptr, "both on: mode 4 alone");
	const auto low = PlaySet(root, true, true, "", 16 * GiB, 16);
	Check(Is(Value(low, "KYTY_WEAVE"), "1") && Value(low, "KYTY_SPECULATE") == nullptr, "mode 4 not offered: the weave");
	Check(Is(Value(PlaySet(root, false, true, "KYTY_WEAVE=0", 16 * GiB, 16), "KYTY_WEAVE"), "0"), "the user's own pair lost");
	Check(!Is(Value(PlaySet(root, false, true, "KYTY_SPECULATE=4", 32 * GiB, 16), "KYTY_WEAVE"), "1"),
	      "the options' KYTY_SPECULATE=4 not taken by the gate");
	Check(Is(Value(PlaySet(root, false, true, "KYTY_WEAVE_WORKERS=2", 16 * GiB, 6), "KYTY_WEAVE"), "1"),
	      "the options' KYTY_WEAVE_WORKERS not taken by the gate");
	Check(Value(PlaySet(root, false, true, "KYTY_SCOUT=", 16 * GiB, 16), "KYTY_WEAVE") == nullptr,
	      "the options' empty KYTY_SCOUT (0, as Play makes it) not taken by the gate");
	// The launcher's environment (the game inherits it) is the gate's too, and an extra option's pair beats it.
	setenv("KYTY_SCOUT", "0", 1);
	Check(Value(PlaySet(root, false, true, "", 16 * GiB, 16), "KYTY_WEAVE") == nullptr, "the environment not taken by the gate");
	Check(Is(Value(PlaySet(root, false, true, "KYTY_SCOUT=1", 16 * GiB, 16), "KYTY_WEAVE"), "1"),
	      "the extra option's pair does not beat the environment");
	unsetenv("KYTY_SCOUT");

	// The text: what it does with the memory figure, or why it is not offered, in English and Chinese, the Chinese its
	// own (T falls back to English for a key Chinese lacks).
	const char* keys[] = {"weave", "weave.desc", "weave.mode4", "weave.fewCores", "weave.needs", "summary.weave"};
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
		const auto desc      = Strings::F("weave.desc", {Weave::ExtraMemoryMb});
		Check(desc.find(std::string(Weave::ExtraMemoryMb) + " MB") != std::string::npos &&
		          desc.find("KYTY_WEAVE=1") != std::string::npos,
		      at + "desc");
		Check(std::string(Strings::T("weave.mode4")).find("KYTY_SPECULATE=4") != std::string::npos, at + "mode4");
		const auto cores = Strings::F("weave.fewCores", {"10", "6"});
		Check(cores.find("10") != std::string::npos && cores.find('6') != std::string::npos, at + "fewCores");
		Check(Strings::F("weave.needs", {"KYTY_SCOUT"}).find("KYTY_SCOUT") != std::string::npos, at + "needs");
	}
	Strings::Current = Strings::Language::English;

	fs::remove_all(root);
	std::printf("Launcher weave tests passed (%d checks).\n", g_checks);
	return 0;
}

// Pad input as the game sees it (src/libs/controller.cpp): the state history scePadRead gives.
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/presentation/window/hostInput.h"
#include "libs/controller.h"
#include "libs/padData.h"

#include "SDL_keycode.h"
#include "SDL_mouse.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Controller;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "PadInputTests: failed: %s\n", text);
    std::abort();
  }
}

// The unread states, oldest first (Demon's Souls reads 64 a tick).
int Read(PadData *data) { return PadRead(1, data, 64); }

// A tap, then far more stick events before the next read than the history holds: the read still has the press and
// the release, and its newest state the sticks' last values (KYTY_PAD_STATE_COALESCE).
void CheckTapAcrossStickBurst() {
  static PadData data[64];
  (void)Read(data);
  SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_CROSS, true);
  for (int i = 0; i < 200; i++) {
    SetAxis(HOST_INPUT_CONTROLLER_ID, Axis::LeftX, 100 + i % 50);
  }
  SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_CROSS, false);
  for (int i = 0; i < 200; i++) {
    SetAxis(HOST_INPUT_CONTROLLER_ID, Axis::LeftX, 150 - i % 50);
  }
  const int count = Read(data);
  bool pressed = false;
  bool released_after = false;
  for (int i = 0; i < count; i++) {
    if ((data[i].buttons & PAD_BUTTON_CROSS) != 0) {
      pressed = true;
    } else if (pressed) {
      released_after = true;
    }
  }
  Check(pressed && released_after, "a tap between two reads keeps its press and its release");
  Check(count > 0 && data[count - 1].left_stick_x == 150 - 199 % 50,
        "the newest state has the sticks' last values");
}

// The buttons held now.
uint32_t Buttons() {
  static PadData data[64];
  const int count = Read(data);
  return count > 0 ? data[count - 1].buttons : 0;
}

// The keymap's combos (src/graphics/presentation/window/hostInput.cpp), with the pairs main gives: a second input
// pressed while the first is held presses the combo's control, and its release lets go of what its press started.
void CheckKeymapCombos() {
  using namespace Libs::Graphics;
  const auto key = [](SDL_Keycode code, bool down) { HostInputKey(static_cast<int>(code), down); };
  Check(Buttons() == 0, "no button before the keys");

  HostInputMouseButton(SDL_BUTTON_LEFT, true);
  Check(Buttons() == PAD_BUTTON_CROSS, "the second input alone: its own control (Cross=Mouse:Left)");
  HostInputMouseButton(SDL_BUTTON_LEFT, false);
  Check(Buttons() == 0, "and its release");

  key(SDLK_RSHIFT, true);
  Check(Buttons() == PAD_BUTTON_CIRCLE, "the first input keeps its own control (Circle=Left Shift; Right Shift is it)");
  HostInputMouseButton(SDL_BUTTON_LEFT, true);
  Check(Buttons() == (PAD_BUTTON_CIRCLE | PAD_BUTTON_SQUARE),
        "the second pressed while the first is held: the combo's control (Square=Left Shift+Mouse:Left), not its own");
  key(SDLK_RSHIFT, false);
  Check(Buttons() == PAD_BUTTON_SQUARE, "the first released: the combo's control stays down");
  HostInputMouseButton(SDL_BUTTON_LEFT, false);
  Check(Buttons() == 0, "the second released: what its press started goes up");

  key(SDLK_KP_PLUS, true);
  Check(Buttons() == PAD_BUTTON_TRIANGLE, "a key named with a '+' is one key (Triangle=Keypad +)");
  key(SDLK_e, true);
  Check(Buttons() == (PAD_BUTTON_TRIANGLE | PAD_BUTTON_L1), "and the first of a combo (L1=Keypad ++E)");
  key(SDLK_e, false);
  key(SDLK_KP_PLUS, false);
  key(SDLK_e, true);
  Check(Buttons() == 0, "the combo's second alone: no control of its own");
  key(SDLK_e, false);

  key(SDLK_a, true);
  key(SDLK_b, true);
  Check(Buttons() == PAD_BUTTON_OPTIONS, "two combos of one pair: the later (R1=A+B, then Options=A+B)");
  key(SDLK_b, false);
  key(SDLK_a, false);

  key(SDLK_F8, true);
  key(SDLK_d, true);
  Check(Buttons() == 0, "a combo with one of the emulator's own keys is skipped (Up=F8+D)");
  key(SDLK_d, false);
  key(SDLK_F8, false);

  key(SDLK_RALT, true);
  key(SDLK_RETURN, true);
  key(SDLK_KP_ENTER, true);
  Check(Buttons() == 0, "Alt+Enter, either Enter, is skipped: the emulator's full screen (L3=Left Alt+Return, R3=Right "
                        "Alt+Keypad Enter)");
  key(SDLK_KP_ENTER, false);
  key(SDLK_RETURN, false);
  key(SDLK_RALT, false);

  key(SDLK_LALT, true);
  key(SDLK_F4, true);
  Check(Buttons() == 0, "Alt+F4 is skipped: SDL turns it into closing the window (L2=Left Alt+F4)");
  key(SDLK_F4, false);
  key(SDLK_LALT, false);

  key(SDLK_LSHIFT, true);
  HostInputReleaseHeld(true);
  Check(Buttons() == 0, "a system dialog takes the keyboard: the control a held input pressed goes up (Circle)");
  HostInputMouseButton(SDL_BUTTON_LEFT, true);
  Check(Buttons() == PAD_BUTTON_CROSS,
        "Shift's release went to the dialog: a left click after it is Cross, not the combo's Square");
  HostInputMouseButton(SDL_BUTTON_LEFT, false);

  key(SDLK_LSHIFT, true);
  HostInputMouseButton(SDL_BUTTON_LEFT, true);
  HostInputReleaseHeld(false);
  Check(Buttons() == PAD_BUTTON_CIRCLE,
        "a system dialog takes only the mouse: the click's control goes up (Square), the held key's stays (Circle)");
  HostInputMouseButton(SDL_BUTTON_LEFT, true);
  Check(Buttons() == (PAD_BUTTON_CIRCLE | PAD_BUTTON_SQUARE),
        "the click's release went to the dialog, Shift's did not: a left click after it is the combo's Square");
  HostInputMouseButton(SDL_BUTTON_LEFT, false);
  key(SDLK_LSHIFT, false);

  Check(HostInputMapsKey(SDLK_LSHIFT) && HostInputMapsKey(SDLK_a) && HostInputMapsKey(SDLK_b) &&
            !HostInputMapsKey(SDLK_SPACE),
        "a key in a combo, either side, is the keymap's, and Space, in none, is not");
  Check(Buttons() == 0, "no button after the keys");
}

// A control takes one key and one mouse input, a combo of its second's kind; a later line replaces an earlier one for
// the same control and kind. The wheel: a notch presses its control at once (released 80 ms later, when the window's
// wait comes round), a combo's with the first held, and never as a combo's first.
void CheckKeymapKindsAndWheel() {
  using namespace Libs::Graphics;
  const auto key = [](SDL_Keycode code, bool down) { HostInputKey(static_cast<int>(code), down); };
  Check(Buttons() == 0, "no button before the kinds");

  key(SDLK_z, true);
  HostInputMouseButton(SDL_BUTTON_RIGHT, true);
  key(SDLK_z, false);
  Check(Buttons() == PAD_BUTTON_DOWN,
        "a control a key and a mouse button hold stays down when the key goes up (Down=Z, Down=Mouse:Right)");
  HostInputMouseButton(SDL_BUTTON_RIGHT, false);
  Check(Buttons() == 0, "and goes up when the mouse button does too");
  HostInputMouseButton(SDL_BUTTON_RIGHT, true);
  key(SDLK_z, true);
  HostInputMouseButton(SDL_BUTTON_RIGHT, false);
  Check(Buttons() == PAD_BUTTON_DOWN, "the mouse button up first: the key still holds it");
  key(SDLK_z, false);
  Check(Buttons() == 0, "and the key's release lets it go");

  key(SDLK_f, true);
  Check(Buttons() == PAD_BUTTON_R2, "a control's key (R2=F)");
  key(SDLK_f, false);
  key(SDLK_LCTRL, true);
  HostInputMouseButton(SDL_BUTTON_MIDDLE, true);
  Check(Buttons() == PAD_BUTTON_R2, "and a combo ending in a mouse button on it too (R2=Left Ctrl+Mouse:Middle)");
  HostInputMouseButton(SDL_BUTTON_MIDDLE, false);
  key(SDLK_LCTRL, false);
  Check(Buttons() == 0, "the combo released");

  key(SDLK_g, true);
  Check(Buttons() == 0 && !HostInputMapsKey(SDLK_g),
        "a later key for the same control replaces the earlier (Left=G, then Left=H)");
  key(SDLK_g, false);
  key(SDLK_h, true);
  Check(Buttons() == PAD_BUTTON_LEFT, "the later key presses it");
  key(SDLK_h, false);

  Check(!HostInputMapsKey(SDLK_j), "the wheel as a combo's first is skipped (Right=Mouse:WheelUp+J)");

  HostInputMouseWheel(3);
  Check(Buttons() == PAD_BUTTON_UP, "a wheel notch presses its control at once (Up=Mouse:WheelUp)");
  HostInputReleaseHeld(false);
  Check(Buttons() == 0, "a system dialog: the notch's press in flight goes up");
  HostInputMouseWheel(1);
  Check(Buttons() == PAD_BUTTON_UP, "and a notch after it is a new press at once");
  HostInputReleaseHeld(false);
  Check(Buttons() == 0, "a system dialog again");

  key(SDLK_LSHIFT, true);
  HostInputMouseWheel(1);
  Check(Buttons() == (PAD_BUTTON_CIRCLE | PAD_BUTTON_R3),
        "with Left Shift held, the notch presses the combo's control (R3=Left Shift+Mouse:WheelUp), not Up");
  HostInputReleaseHeld(false);
  Check(Buttons() == PAD_BUTTON_CIRCLE, "a system dialog taking only the mouse: the pulse goes, the held key stays");
  key(SDLK_LSHIFT, false);
  Check(Buttons() == 0, "no button after the wheel");
}

} // namespace

int main() {
  Common::InitializeThreads();
  Common::Subsystems subsystems;
  subsystems.Initialize<Config::Lifecycle>();
  Config::ConfigOptions options;
  options.printf_direction = Config::OutputDirection::Silent;
  options.keymap = {"Cross=Mouse:Left", "Square=Left Shift+Mouse:Left", "Circle=Left Shift", "Triangle=Keypad +",
                    "L1=Keypad ++E", "R1=A+B", "Options=A+B", "Up=F8+D", "L3=Left Alt+Return", "L2=Left Alt+F4",
                    "R3=Right Alt+Keypad Enter", "Down=Z", "Down=Mouse:Right", "R2=F", "R2=Left Ctrl+Mouse:Middle",
                    "Left=G", "Left=H", "Right=Mouse:WheelUp+J", "Up=Mouse:WheelUp", "R3=Left Shift+Mouse:WheelUp"};
  Config::Load(options);
  subsystems.Initialize<Log::Lifecycle>();

  Libs::Controller::Initialize();
  Libs::Graphics::HostInputInit();
  CheckTapAcrossStickBurst();
  CheckKeymapCombos();
  CheckKeymapKindsAndWheel();
  Libs::Controller::Shutdown();
  subsystems.Destroy();

  std::printf("PadInputTests: all cases passed\n");
  return 0;
}

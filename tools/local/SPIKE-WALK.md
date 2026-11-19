# The 1-3 spike walk: a reproducible area-crossing benchmark (Demon's Souls 01.007.000, Windows)

An 18 s run in Boletarian Palace 1-3 from the archstone, through the arch and the corridor, out onto the bridge. Around 9-10 s in, the run crosses into the next area. The game streams it in, and the emulator retires and registers resources and translates new pipelines. That crossing is the lag the player sees when moving between areas: one long frame and a tail of slow ones. The walk measures it frame by frame.

- `tools/local/windows/spike-walk.ps1` boots and walks. One boot can do one or more walks.
- `tools/local/spike-walk-read.py` reads the rows.

Both build on `bench-run.ps1` and `bench-windows.py`. The bench save is swapped in and restored, verified by hash, as for the other benchmarks.

## What the walk is, and is not

- **Spawn-side warm.** The walk starts from the 1-3 archstone after a level load, so everything around the start is loaded and warm. Only the far side of the crossing is new.
- **The warp is the test's.** `place` puts the player at the walk's start facing down the corridor, so every walk runs the same line. Nobody plays it that way: it is there to make the run exact.
- **The crossing is real play.** Load the area, retire what the last area left, meet the new one. That happens after any level load, archstone travel or death in normal play.
- `-TravelBack` gives the cold form: travel to 1-1, 70 s there, then back to 1-3. By the time the walk starts, the 1-3 area's buffers and images have aged out.

## The recipe

All commands go through the live control file (`bench-windows.py live`). The driver sends them; they are listed here so a run can be checked or done by hand.

| step | command | why |
|---|---|---|
| 1 | `gamecmd +cp11_loadLevel 1 3`, `sleep 20` | archstone travel to 1-3: the walk always starts from a fresh load |
| 2 | `place 17.25 47.999 455.5 -0.0635`, `sleep 0.3` | the player at the start, body yaw −0.0635 rad, set by command (never by the stick) |
| 3 | `where` | the position and yaw read back (`LIVE_WHERE`): check x ≈ 17.25, z ≈ 455.5, yaw ≈ −0.0635 |
| 4 | `pad 128 128 128 128 4 0.15`, `sleep 1.5` | the camera reset behind the player (R3; nothing to lock on). The run then heads 0.027-0.028 rad |
| 5 | `gamekill 300`, `sleep 1` | the enemies near the start killed, so nothing interrupts the run (once, before the run, never during it) |
| 6 | `pad 128 0 128 128 0 18` | the left stick held forward for 18 s. Nothing else is sent until the run ends |
| 7 | `measure 9 c00`, 28 × `measure 0.25`, `measure 2 c99`, then `-Tail` × 2 × `measure 0.5` | frame windows over the run and the stand after it |
| 8 | `where`, `flipsx 8192 <file>` | the end position (on the bridge, z ≈ 390), and every frame's time and buffer registrations |

The run's positions every 2 s (x / z): 17.28/449.3, 17.03/441.9, 16.78/434.6, 16.53/427.2, 16.29/420.0, 16.03/412.6, 15.78/405.2, 15.54/398.0. It ends at 15.3-15.5/390.0-390.6 on the bridge, within 0.1 m walk to walk.

The two crossings:
- **First crossing:** ~9-10 s into the run, at the arch mouth (x ≈ 15, z ≈ 424). This is the detector's spike and the crossing window.
- **Second window:** the longest frame from 11 frames after the first crossing to the run's end, taken out on the bridge. The "Boletarian Palace - The Lord's Path" banner comes at ~18 s.

**The save.** Any save with the 1-3 archstone (The Lord's Path) unlocked: step 1 travels there from anywhere. The save's game must also have the Tower Knight (the 1-2 boss) beaten in play (a Linux tester's report, BIOS2000 on #342, 16 Nov). The reader checks each walk's end position (its last `where`: z ≈ 390.3 on the bridge) and warns when it is off. Pass it with `-Baseline <dir>` (a folder with `PPSA01341` and `.kyty-capacity`, as bench-windows.py's `KYTY_BENCH_BASELINE`). Your own save stays moved aside and comes back verified at the end.

**Settings the driver uses:**
- `KYTY_FPS_HUD=2`: bench-run finds the game by the HUD.
- `--vblank-frequency 120` (bench-run's `-Vblank 120`).
- Mode 4: `KYTY_SPECULATE=4` and `KYTY_SCOUT=0`. Mode 0: neither.
- Optional `-Key <name>`: a fixed shader-cache key, so the cache stays warm across builds.
- Measuring starts only after "Shader prefetch: all" is in the log and the HUD is up.
- Extra switches go in with `-Set`.

## Run and read

```
powershell -File tools\local\windows\spike-walk.ps1 -Label base1 -Exe <exe> -Mode 4 -Key bench
powershell -File tools\local\windows\spike-walk.ps1 -Label test1 -Exe <exe> -Mode 4 -Key bench
python tools\local\spike-walk-read.py base1 base2 base3 base4
```

Alternate the two sides (A B B A …), one boot each, in one sitting.

**The rows**, per walk:
- **calm:** the mean frame of the 60 before the crossing.
- **crossing ms:** the frame with the most buffer registrations within 10 frames of the first frame over 50 ms from 8.5 s in.
- **window:** the 21 frames around the crossing: worst, time over calm, frames over 25 / 33 ms.
- **tail:** the 10 frames after the crossing: peak, time over calm, frames over 25 / 33 ms.
- **2nd window:** as the window, around the longest frame later in the run.
- **write faults a frame:** the mean over the walk's measure windows.

Then, after those rows (per walk and in the means), the rows past the crossing window. A walk too short for a row prints "-".
- **late burst worst ms:** the worst frame from +30 to +60 frames after the crossing frame.
- **late burst over-calm ms:** the time over calm from that worst frame to 10 frames after it (11 frames).
- **plateau ms:** the mean frame time over frames +60 to +120 after the crossing.
- **walk mean / p95 / p99 ms, walk >25:** over the whole run, from the pad's start (`LIVE_PAD flip=`) to the release 18 s later in the dump: the mean frame time, the 95th and 99th percentile frame (nearest rank), and the frames over 25 ms.
- **stand ms:** the mean frame time over the `-Tail` standing windows after the run: their `LIVE_MEASURE` replies' time over their frames. "-" for a boot run with `-Tail 0`.

The most telling single rows are "window over-calm ms", the crossing frame and "tail over-calm ms".

## Noise, and how many boots

Over 16 boots of one build:

| row | boot-to-boot spread |
|---|---|
| window over calm | cv 12 % (sd ~25 ms on ~210) |
| crossing frame | sd 4.5 ms |
| tail over calm | cv 15 % |

A 10 % change on the window row needs ~12 boots a side to show at 2 standard errors. Smaller sets give a direction, not a result.

**Storm boots.** Some boots (31 of 233 walks over one day, 13 %) run with a partial-dirty write-fault storm: over 100 write faults a frame instead of ~10-20. The reader flags them STORM (40-100: mild) and leaves STORM walks out of the means, because their window rows measure the storm, not the crossing. Count them per side: the storm rate is a row of its own.

## Emulator support it needs

All in `src/local/live-control.h` and `src/loader/demonsSoulsWarp.*`. These are live commands only: nothing runs unless a command asks.
- `place X Y Z YAW` and `where`: the player's position by the game's warp request, and read back.
- `flipsx <count> <path>`: the flips dump with each frame's buffer registrations.
- `flip=` on `LIVE_PAD` and `last=` on `LIVE_FLIPS` / `LIVE_FLIPSX`: these place the run on the dump.
- `gamecmd`, `pad`, `gamekill`, `measure` and `LIVE_COUNTERS` (`write_faults`, `buffer_registrations`) were there already.

The cost in every mode is two relaxed loads and one store a flip (the buffer registrations counter kept per flip for `flipsx`).

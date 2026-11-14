# Methodology: measure, decide, stay safe

## Harness (build once, early; it pays for itself in days)

1. **Baseline save swap**: move the player's save aside, install a fixed test save before each run, restore with a
   per-file hash check; refuse to restore while an emulator runs. A killed script skips its `finally`: restore by hand.
   Scripts exit when the swap fails (prevents overlapping runs). Never chain a run after a build with `;`.
2. **Scripted start + level check**: skip logos, detect "in game" from pixels, timeout = bug. Confirm the level from
   the start-up screenshot (a tutorial was benchmarked as "1-1" for a long time). Script input as one key plan from
   one process (separate key processes drifted ~1 s and routes ended at different walls).
3. **Live control** (the emulator polls a command file): `measure <s>` (flips/s, render ms, busy %, 1% low, p99,
   counters), `poke32`/`set <symbol>` (flip a `kyty_local_*` switch in the running process), `census <s>` (scoped
   timers per kind/shader/phase: calls and ms per frame), `trace`/`tracem` (event timeline; `tracem` adds GPU marks
   but serializes the GPU ~2x), `prof`/external sampler, `gamecmd` (the engine's console), heap/VMA/image/buffer dumps.
4. **Tours**: in-game world switching through the engine's own level loader (2-4 s per load) for multi-world soaks
   (worlds x cycles: fps, render ms, private/working set, VMA, image/buffer MiB, mapping counts per stop); god mode +
   teleport route tours over real enemy positions as the fps-and-crash benchmark (reproduced user crashes in minutes).
5. **Symbols**: fixed image base + linker map; keep exe + map + PDB per profiled build, under their original names.

## A/B protocol

- **Same process, live switch, alternating** (A B A B, 4-6 rounds of 5-8 s at a fixed spot); accept when every B
  window beats every A window. Within-run noise +-0.1-0.3 fps; render ms of one 10 s window +-1 ms.
- **Prove the switch moved** before reading the numbers: the log's `LIVE_SET <name>=<v> ok=1` for a live switch, a
  counter only one side touches, or a screenshot. Until 11-14 the bench tool's `set` silently dropped every live-switch
  name (it only poked map globals), and two "flat" A/Bs (barrier scopes, an unsafe drain ceiling) had measured the same
  setting twice. A ceiling that moves nothing, an unsafe skip included, is a reason to check the toggle first.
- **Cross-boot is noisy**: one view 55.7 vs 60 fps across boots; whole sweeps differ ~7% per world. Needed for things
  that cannot be poked (layouts, build changes): >=2 boots per side, alternate exes (`-Exe` from a copied dir).
- **Effects under 3%**: compare census per call (us per draw/dispatch/phase), not fps. Under ~0.5%: profile share.
- **Benchmark under a 120 Hz virtual vblank** (60-cap fps hides differences; keep the game's speed fixed there).
  Know which fps you read (guest flips vs presents).
- **Spots**: a fixed standing view + a camera spin from the spawn point. Forward walks are nondeterministic (round 2
  faces elsewhere: 9.2K vs 2.3-6.2K draws; deaths; walls) -> one walk per boot, or teleport tours.
- **Test short** (1 boot, 2-4 rounds) for a change; broad (soak, tours, screenshots) only before keeping risky work.
- Aggregate as total frames / total time and keep every window. Never benchmark while compiling; a hot CPU package
  throttled ~7%.

## Proof before code

- **Causal delay probe**: add a known delay at the site, measure frame-time slope. Re-measure after every big win:
  slopes moved (graphics translation 1.1 -> 0.6; async 0 -> 0.8; graphics dispatches 0.2; job-thread faults 0.06).
  "Busy %" of a thread is not criticality.
- **Ceiling first**: unsafe skip, measure. Re-run ceilings when the bottleneck moves (barrier skip +0.3% one week,
  +11.5% two weeks later). A ceiling is invalid if its precondition did not hold (an epoch that moved every frame,
  broken depth letting culling pass +40% draws, a "+19%" that was missing geometry).
- **Log what follows the event before deferring work**: a temporary log of each partial unmap and of the maps after it
  showed the game maps over <1% of freed ranges within 1 s (96%: after 10 s or never), so doing the work lazily pays.
- **Count the opportunity**: one temporary counter before building (bytes a trimming could save, draws a fast path
  could take). Several ideas died in minutes this way.
- **Simulate from traces before an architecture change**, but sims ignore commit/prep costs: a "-5.7 ms" parallel
  translation sim became a 15% slowdown.
- **Verify modes** for anything replacing correctness work: run old and new, count mismatches over tours (0 required).
  Same-process screenshot A/B for image changes; flicker counters for GPU-side evaluation; pixel-identical offline
  replay of captured draws; the regression suite against a list of known failures.

## Profiling traps

- Symbolize only with the exact build's files; a rebuild silently re-attributes (a codec appeared inside a draw store).
- Exports-only DLLs: the sample takes the nearest export (vcruntime `_NLG_Return2` = memmove, `memchr` = memcmp).
- Garbled inline frames: check the whole chain; line tools without inline info misattribute.
- Cache-miss skid: samples land on the instruction after the miss (a "hot hash" was cold data; a cheaper hash = 0).
- Memory-level parallelism: misses overlap; removing sampled lookups often leaves per-call cost unchanged. Expect it
  whenever per-item cost is latency (cold tables, guest pages, TLB).
- In-process samplers distort what they measure (thread suspension made 0.5-1.8 s walk frames, let game threads take
  the render CPUs); judge hitches unprofiled, sample with a bounded external unwinder. Worst case: a thread that holds
  locks across system calls (the upload worker: the address space's lock shared, a region's spin lock) suspended at
  4 kHz stalls every writer and then every reader - 0-3 slow frames per world change unprofiled became 12-39, with
  frames of 1.5-35 s. Compare the hitch counts of profiled and unprofiled rounds before reading a hitch profile.
- Diagnostics leak cost: TEMP counters slowed a day of A/Bs; stdout writes blocked ~6 ms (async stdout pipe);
  census scopes inflate small phases ~10x. Off = one relaxed load; keep them in an uncommitted diag set.
- Worker "busy" is mostly spin; a worker is a wall only if the critical thread waits on it.

## Locating the bottleneck

- **Frame chain anatomy** from a trace: flip -> previous GPU tail -> async culling -> guest readbacks -> main submit
  (p2) + main-batch translation (p4). Removing work in one phase can just move idle into the other.
- The bound moves: translation -> GPU latency chain -> render thread -> the game's own CPU waits -> memory latency.
  Re-anatomize after each big win before choosing the next target.
- Know the game's own CPU ceiling (its threads alone, GPU work skipped: ~100 fps here) before blaming the game.
- Split profiles by guest queue and by phase; join a refusal census (shader + reason) with census cost to find what
  still takes slow paths.
- Hitches: per-frame "SLOW" lines with counters (render CPU, upload/record waits, uploads, stores, registrations),
  a render-idle watchdog dumping all guest thread stacks (found synchronous audio reads), lock wait/hold events
  before assuming CPU work, protect traces with callers (found IPI costs), slow-call lines naming the resource
  (big-image churn - untrack, full dirty, unmap delete - explained most streaming hitches).
- Shader stutter: compare the modules compiled at run time with the precompiled set (coverage by stage), not stall
  counts; the driver's own disk cache warms "cold" runs.

## Stop, park, record

- Stop when: the ceiling is small, census per call does not move, or 2-4 same-process rounds are noise. Keep the
  diff as a dropped patch and write the numbers and the reason down (several ideas were built twice for ~1%).
- Park when the cause stays unknown but a guard prevents the crash; record open items instead of chasing them.
- Delete default-off experiments that lost; a pile of switches hides the real cause and costs every A/B.

## Safety rules for automated runs

- One emulator at a time; scripts kill only the process they started; never rebuild while it runs. A crashed process
  can linger holding the exe: rename the exe to rebuild. Don't fight the user for focus: mark the route untrusted.
- Never run experiments on the player's save; verify the restore every time.
- Cold-start tests: fresh full-length driver-cache key, empty `__GL_SHADER_DISK_CACHE_PATH` dir, empty warmup file;
  killed runs never save driver caches (periodic saves fix that).
- Low-VRAM simulation: a ballast allocation is faithful for placement but adds its own stalls; a fake limit flag is
  not faithful (fallback memory lands in VRAM anyway). Confirm on a real small GPU.

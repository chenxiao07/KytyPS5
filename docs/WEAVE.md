# The weave: queue 0's draws translated on worker threads, woven back in order (KYTY_WEAVE=1)

With `KYTY_WEAVE=1` the GPU thread's main work, turning queue 0's table draws into Vulkan calls, is shared with worker
threads. Workers make runs of draws ahead of the GPU thread; the GPU thread keeps the stream's order, checks what each
run relied on, and splices the runs that pass into its own command buffer between the draws it makes itself. Off (the
default, `KYTY_WEAVE` unset or 0) the emulator runs upstream's path: each new check on a path the GPU thread takes is
one relaxed load of a plain global (section 9). The weave and mode 4 (`KYTY_SPECULATE=4`,
docs/PARALLEL-TRANSLATION.md) take turns: one parallel path at a time (section 5). It changes nothing the shader
recompiler emits.

## 1. How it works

The parts, in the order a draw meets them (`src/local/weave.h` has the same overview; the code is in
`src/local/weave*.inc`, `weave.cpp` and the table path's `table-xpr.inc`):

1. **The reader** (thread `Kyty.WeaveRead`). `GuestGpu::Enqueue` hands it each queue-0 graphics submission as it feeds
   the scout. It copies the commands, walks the copy ahead of the GPU thread with the real register handlers and cuts
   the stream into **parcels** of up to `KYTY_WEAVE_CALLS` table calls. A parcel ends early at a packet only the GPU
   thread can run. At each parcel's first draw the reader keeps the registers. It reads on past frame resets, past
   draws a handler makes and past waits. When it loses step (a submission it never got, a packet it cannot follow), the
   GPU thread posts its own registers at its next draw and the reader starts again from them. This is a second walk of
   the packets, on a copy, ahead of the GPU thread's own.
2. **The workers** (threads `Kyty.Weave1` to `Kyty.Weave8`). A worker takes a parcel and makes its draws with the table
   path's own functions, compiled a second time under a worker policy (`Weave::Worker`; the GPU thread's instantiation,
   `Weave::Ordered`, is today's code). A worker reads the GPU thread's caches and writes only its own state. Where the
   GPU thread's code would change something shared, a worker keeps a private answer, notes a **need** (an upload, a
   layout, a stamp) for the GPU thread, or gives the draw up: a **hole**. It reads a pipeline's handle once, at the
   bind; a handle that reads null (the pipeline was evicted while idle) makes the draw a hole, so no bind of a null
   handle is recorded. It records its Vulkan calls **apart**, into a buffer of its own, through a marker command buffer
   that no driver hands out, and writes its blocks and small copies into a 1 MiB **slab** of its own 16 MiB upload
   ring. The calls between two holes form a **strand**; strands also
   break every 64 draw packets (a **stretch**, by which needs are grouped). At the parcel's end the worker condenses
   its notes into lists for each strand and a register log.
3. **The GPU thread's turn.** The GPU thread walks the stream in order as today. At a parcel's first draw it **opens**
   the parcel: the worker is done (or will be within 20 µs by its pace), its start registers equal the GPU thread's
   (`operator==` on the `HW::` register structs), no pipeline was evicted since its worker began, and nothing it read
   has moved since then: the mapping, shader registrations at its programs, an image start inside a range it bound,
   the device-address table, the linear copies it read through, and the scout's write ring over its pages. Then at
   each strand's first draw:
   1. it does the stretch's needs with today's functions;
   2. it tests the strand once from its lists: the free log, the pipeline handles its binds recorded, and the
      registration, mapping and resolution gates, where each range or item is asked about from the epoch of the first
      draw that holds it; with frame generation on, the frame's first G-buffer draw while the frame has no camera, which
      the GPU thread makes itself so its hook reads the camera. When a test fails it finds the first draw that failed
      and keeps the strand's prefix before it; the rest of the strand is made by the GPU thread;
   3. at a strand that follows a hole, or one spliced late (the GPU thread's own draws took its start, so it is spliced
      from a later draw that binds its own pipeline), it compares its registers whole with the worker's there (the
      strand's start image, brought to a late part's draw by the register log) and cuts the strand on a difference;
   4. it does the strand's **actions**: constants the guest wrote copied again, CPU-written index ranges uploaded,
      CPU-written sampled images refreshed, the depth clear the strand's first pass makes, the layouts;
   5. it begins the render pass and **splices** a **part** of the strand into its own stream, stamps what the part used
      and forgets what it had bound;
   6. it skips the part's packets: the last write of each register block from the register log, the opcode counts and
      `CompleteDraws`, the scout's draw numbering, and the draw-state observer invalidated. While the write guard is
      armed (`KYTY_WRITE_GUARD=1`) no register log is made and each spliced packet is passed as the loop passes a table
      draw.
4. **Holes.** A hole, the rest of a cut strand and any draw a worker never reached go through the packet's handler and
   today's table path, into the same command buffer, between the spliced parts. A handler's draw inside an open parcel
   is watched: if it changes the registers, the parcel closes.
5. **The replay.** `Kyty.Record`, the recording worker, replays each spliced part as soon as it is published, writing
   the real command buffer over each call's marker, and carries on from where the last part of the same strand stopped.

**Why the workers' reads are safe.** Every unlocked read a worker makes of a shared container carries a comment
naming what makes it safe. The GPU thread changes what workers read (pairs, variants, sets, draw states and their maps)
only inside a `Weave::Hold`, a **safepoint**: it waits until no worker is inside a draw and keeps them out until it
ends. A worker never waits for the GPU thread inside a draw without limit, so the GPU thread's wait is one draw at
most: it spins briefly, then sleeps until the worker leaves the draw, which wakes it (the GPU thread runs above the
normal priority, and with the core placement off a worker may share its CPU). The reader and a worker wait for the
texture cache's lock at most 5 µs (`AskWait`): still held then, the question
gets the answer that declines (for a worker a hole or a need; for the reader, which asks the texture cache as it reads
a linear copy's source, the copy left undecoded). The GPU thread notes each pair, set and draw state it frees or rebuilds in place in a **free log**; an
object freed since a parcel began cuts its strand at the object's first draw. A retired buffer or image slot is
erased (`EraseRetired`) only once no worker that may have found it is still inside that draw. A retired image keeps
its id alive until then, and a set or a draw state still names it, so each retired image slot `EraseRetired` asks
about is published as retiring, by its full id (index and generation), before the slot's stamp; a worker's draw begun
after that leaves the image to the GPU thread. While all 64 retiring entries are taken, a slot asked about is kept,
with no stamp, until an ask finds an entry free. A pipeline evicted while idle is counted after its handle is
cleared, with release order, and a parcel whose count moved is not opened.

## 2. Running it

`run.cmd -Set KYTY_WEAVE=1` (or `-Set KYTY_WEAVE=1` in the launcher's Advanced, "Command-line options"). Check the
log before trusting a number:
- the `Performance switches:` line names `KYTY_WEAVE=1` and, unless placement is off, `KYTY_RENDER_CORE=<cpu>` (the
  first CPU of the GPU thread's core: 2 on an 8-core CPU with SMT);
- at the first draw: `Weave: on (KYTY_WEAVE=1): 8 workers, 128 calls a parcel, the GPU thread's core
  KYTY_RENDER_CORE=auto`;
- no `Weave:` stand-down line (section 6); the `Performance switches:` line names `KYTY_WEAVE=1` even then.

The counters (section 8) are read with the live command `measure <s>`. They are counted once a parcel or a strand,
never per draw.

## 3. Switches

All are read once at start-up; none is a live switch. The three `KYTY_WEAVE*` rows are switch rows of
`src/local/performance-switches.h`: an out-of-range value ends start-up, as for every row there. `KYTY_RENDER_CORE` is
read by `KeepRenderCoreAlone` (`src/local/local-platform.cpp`): a value it cannot use (a number outside 1-63, other
text, or `auto` with no core to keep) keeps no core and does not end start-up. The two CPU lists are read as upstream
reads them.

| switch | values, default | what it does |
|---|---|---|
| `KYTY_WEAVE` | 0 or 1, 0 | The weave on (1) or off |
| `KYTY_WEAVE_WORKERS` | 0 (auto) or 1-8, 0 | The workers. Auto: the logical CPUs of the performance cores less 8, at most 8 (8 on 16 logical CPUs, 4 on 12). Auto under 2 stands the weave down; a number set here lifts that check. Where the core classes are not known every core counts |
| `KYTY_WEAVE_CALLS` | 16-1024, 128 | The table calls in one parcel |
| `KYTY_RENDER_CORE` | `<cpu>` (1-63), `auto`, `off` or 0 | The GPU thread's core kept for it alone (section 4). Unset: `auto` while the weave runs, else not kept (the Steam Deck build keeps its 2) |
| `KYTY_RENDER_CPUS` | a CPU list | Upstream's, unchanged. When it is set, an unset `KYTY_RENDER_CORE` stays unset under the weave (the GPU thread was placed by hand) |
| `KYTY_RECORDING_CPUS` | a CPU list | Upstream's, unchanged. With the core kept, it must leave that core out: a thread's own mask wins over the process's default CPU set |

The weave also needs, at their defaults: `KYTY_SCOUT=1` (its write ring), `KYTY_VULKAN_RECORDING`,
`KYTY_TABLE_XPR` and `KYTY_NATIVE_XPR`.

## 4. Placement

While the weave runs, the GPU thread gets one physical core to itself, as `KYTY_RENDER_CORE` already did on Linux:
- **Which core** (`auto`): the lowest-numbered core of the performance class that isn't CPU 0's, which takes most
  interrupts on Windows. CPUs 2 and 3 on an 8-core CPU with SMT; a P-core on a hybrid Intel part.
- **Windows** (`src/local/local-platform.cpp`, `KeepRenderCoreAlone`): the topology from
  `GetLogicalProcessorInformationEx`. The GPU thread is pinned to both CPUs of its core, and the process's default CPU
  set becomes every other CPU of the process. Every thread without a CPU set or mask of its own takes it: the game's,
  the driver's, the recording and upload workers, the reader and the workers, those made later too. Helpers that call
  `AvoidCpuList` keep off the core as well.
- **Linux**: upstream's body (the GPU thread on its CPU, `Kyty.CoreKeeper` moving other threads off the core every
  250 ms), with `auto` read from the sysfs topology and the performance class from `/sys/devices/cpu_core/cpus` where
  there is one. Each weave thread leaves the GPU thread's CPUs it was born on (`AvoidCpuList`).
- **No worker or the reader is pinned.**
- `KYTY_RENDER_CORE=off` turns placement off under the weave. `KYTY_RENDER_CORE=auto` asks for it in any mode.

**A changed meaning on Windows.** Before this change `KeepRenderCoreAlone` did nothing on Windows, so
`KYTY_RENDER_CORE=<cpu>` had no effect there. Now it keeps that CPU's core in any mode, as on Linux.

The comment at `KeepRenderCoreAlone`'s call in `performance-switches.h` records that a dedicated core made the whole
process slower on Windows. Placement is therefore the weave's default only, never mode 0's. If the
weave stands down after start-up (its checks at the first draw, section 6), the core is given back, as if the weave
had never been asked: the default CPU set and the threads kept off the core get it again. A core `KYTY_RENDER_CORE`
names stays.

## 5. With mode 4

The weave runs only while `KYTY_SPECULATE` is 0. It reads mode 4's switch and nothing else of mode 4.
- `KYTY_SPECULATE=4` in effect after mode 4's own start-up gates: the weave stands down for the whole run. No weave
  thread starts, and a later live `mode 0` doesn't bring it up. When mode 4 stands itself down at start-up (its RAM
  or CPU gate sets `KYTY_SPECULATE` to 0), the weave runs as in a run started in mode 0.
- A run started in mode 0 and switched live to `mode 4`: the change is applied at the top of the GPU thread's loop,
  between queue 0's submissions, and mode 4's threads start there, at the change. In the same pass of the loop, before
  the GPU thread runs any other packet or guest command, it pauses the weave (`Weave: paused while
  KYTY_SPECULATE=4.`, once): every parcel is dropped, the reader stops cutting, every worker is held outside a draw and
  idle, and every slot and slab is released. While the pause waits for the workers, mode 4's threads only read and the
  GPU thread writes nothing. The change sets the table path's switches for mode 4 before the pause; a worker inside a
  draw may read one as it flips, and its parcel is dropped with the rest, nothing of it spliced. Back in `mode 0`, the
  weave resumes at queue 0's next graphics submission and the reader starts out of step.
- The scout keeps running under the weave, which reads its write ring.

## 6. Stand-downs

Each prints one line and the run stays in mode 0.

| when | line |
|---|---|
| `KYTY_SPECULATE=4` in effect after its start-up gates | `Weave: KYTY_WEAVE=1 stands down: KYTY_SPECULATE=4 is on (one parallel path at a time). This run stays in mode 0.` |
| auto workers under 2 (fewer than 10 logical CPUs on the performance cores) | `Weave: KYTY_WEAVE=1 stands down: <n> logical CPUs on the performance cores, under the 10 it needs (KYTY_WEAVE_WORKERS lifts the check). This run stays in mode 0.` |
| `KYTY_SCOUT=0` (the Steam Deck build's default) | `Weave: needs the scout (KYTY_SCOUT=0). This run stays in mode 0.` |
| the recording or the table path off | `Weave: needs KYTY_VULKAN_RECORDING, KYTY_TABLE_XPR and KYTY_NATIVE_XPR. This run stays in mode 0.` |
| a build without the Vulkan recording (non-x86-64 builds) | `Weave: needs the Vulkan recording (this build has none). This run stays in mode 0.` |
| the scout not running at queue 0's first draw | `Weave: needs the scout (not running at queue 0's first draw); the weave stays off.` |
| its threads could not start | `Weave: its threads could not start (<error>); the weave stays off.` |
| a worker's upload ring not host-coherent | `Weave: worker <n>'s upload ring is not host-coherent; the weave stays off.` |

The last three also free what the start made (the workers' upload rings and views, the reader's and the workers'
processors, the linear-copy records) and give back the GPU thread's core if only the weave kept it (section 4), so
the run goes on in mode 0 as if the weave had never been asked. The Steam Deck build stands down by both its scout
default and its four cores. There is no memory gate (section 7).

Two more lines are reports, not stand-downs: `Weave: the GPU thread has waited <ms> ms at a safepoint (worker <n> in
a draw)`, once a wait passes 1 s and again at each doubling (a freeze would sit there), and the same while pausing.

## 7. Memory

From the sizes in the code:
- the workers' upload rings: 16 MiB each in host-visible upload memory, 128 MiB with 8 workers;
- the apart recording buffers: blocks of 64 KiB, of which up to 512 (32 MiB) are kept free for reuse;
- 512 parcel slots, each with the vectors of its result, kept from parcel to parcel; each parcel holds a copy of its
  submission's commands until it is released;
- the free log (65,536 entries, 512 KiB), the feed ring (1,024 entries) and the linear-copy records (16,384).

About 200 MB in all with 8 workers. Mode 0 allocates none of it. Mode 0 does carry a few fixed static logs beside the
caches, which the weave reads (a 24 KiB registration log, 2 KiB of published readbacks, two 6 KiB map logs and a 64 KiB
shader log), written only while the weave is enabled; the caches' objects don't grow.

## 8. Counters

Ids `Weave*` in `src/local/live-counters.h`; names as `measure` prints them:

| counter | counts |
|---|---|
| `weave_parcels` | parcels the reader cut and published |
| `weave_parcels_opened` | parcels the GPU thread opened |
| `weave_parcels_refused_registers` | ... refused at the open: the start registers not the GPU thread's |
| `weave_parcels_refused_mapping` | ... the mapping moved over its pages |
| `weave_parcels_refused_shaders` | ... a shader registered at one of its programs |
| `weave_parcels_refused_image_start` | ... an image began inside a range it bound |
| `weave_parcels_refused_bda` | ... the device-address table moved |
| `weave_parcels_refused_copies` | ... a linear copy it read through didn't run as the reader recorded it |
| `weave_parcels_refused_ring` | ... the scout's write ring holds a write over its pages |
| `weave_parcels_refused_freed` | ... the free log turned over since it began (not every free since is known); an object freed since cuts its strand instead |
| `weave_parcels_refused_late` | ... its worker not done within the wait |
| `weave_parcels_refused_capture` | ... at the draw packet: a capture or dump, predicated skipping, the table path off |
| `weave_parcels_gate_skipped` | parcels a worker took while the spin gate was shut (made as when it is open) |
| `weave_strands_spliced` | strands spliced (whole or a prefix) |
| `weave_strands_cut` | strands whose test failed at a draw (the rest made by the GPU thread) |
| `weave_draws_spliced` | draw packets spliced |
| `weave_draws_own` | draw packets the GPU thread made itself inside open parcels |
| `weave_holes` | draws a worker left as holes |
| `weave_holes_buffers` | ... for a buffer it could not take as it is |
| `weave_holes_ring` | ... for its slab or a copy into it |
| `weave_slab_refused` | parcels whose worker found no slab in its ring |
| `weave_safepoints` | holds the GPU thread took while workers ran |

**The spin gate.** The reader and the workers spin for work only while the spin gate is open (`spin-gate.h`,
`KYTY_SPIN_GATE`), and block otherwise. The gate stops only the spinning: a worker takes and makes its parcel while
the gate is shut too (`weave_parcels_gate_skipped` counts those), since a parcel left is made by the GPU thread, on
the frame's path.

## 9. Mode 0

With `KYTY_WEAVE` unset, mode 0 runs upstream's code. Each hook on a path the GPU thread takes is one relaxed load of a
plain global (`Weave::Enabled()` or `Weave::Active()`), read before any thread-local, or one load of a plain global of
the weave's (`ProcessPm4`'s test for the processor the weave runs on). The GPU thread's loop pays one such load a pass
(the pause at a live `mode 4`). `ProcessPm4`'s loop is one body, `ProcessPm4Loop<bool>`, whose `false` instantiation
is the loop as it was. The table path's functions are templates on a policy whose GPU-thread instantiation is today's
code; where a worker needed part of an upstream function, that part was split out read only and the function calls
it. No upstream struct that mode 0 uses grows: the weave's state is in its own statics. In builds with the Vulkan
recording the render executor gains a destructor, which tests one such global. `EvictIdlePipelines`' two adds to its
eviction count use release order (on x86-64 the same locked add as relaxed). Nothing is collected in mode 0. The
weave's code is one more set of functions under the shipped PGO profile, and the templated and split functions lose
their counts until the profile is retrained.

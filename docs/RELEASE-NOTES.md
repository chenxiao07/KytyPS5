## New in this build

Your precompiled shaders stay valid. On AMD graphics cards under Windows, the decal shaders made with the pixel interlock
are compiled once more without it the first time they appear (or with your next precompile).

- **AMD on Windows: characters' faces are right again** (#355, credit @supercc4e): with v20261118 and a precompile made
  with it, a Radeon under Windows showed the player character's face as the game's missing-texture checker (green,
  with magenta dots and red lines). AMD's Windows driver does have the pixel interlock that v20261118 began to use for
  the decals (that version's notes said it hadn't), and with it the face came out wrong. The emulator leaves the
  interlock off on AMD's Windows driver now, as before v20261118. Without it, the decal holes of #315/#341 can show
  again there; `-Set KYTY_DRAW_READ_AFTER_WRITE=1` remains the workaround for those, and `-Set KYTY_PIXEL_INTERLOCK=1`
  turns the interlock on anyway. If you used `-Set KYTY_PIXEL_INTERLOCK=0` for the faces, you can drop it.
- **Less memory, most of all in long sessions:** the shader recording (the shaders and pipelines the warmup replays,
  ~100 MB once the whole game has been played) no longer stays in memory after start-up: each save copies its earlier
  part from the file on disk, checked against the file's checksum. Shader resource analyses are kept in large blocks
  instead of one heap allocation per value (millions of them); pipelines kept for re-creation store only the vertex
  inputs they use (~1.6 KB instead of ~11 KB each); the ~224,000 file paths the game looks up are stored once, and the
  .zar reader's tables once for all its read lanes. Over an hour of travel through every world (RTX 5090), the
  emulator used about 0.5 GB less memory on average than v20261118 with the 16 GB PC settings (simulated), and about
  0.6 GB less with plenty of RAM. The frame rate is the same.
- **Keyboard and mouse: one control on two inputs at once** (PR #360, credit @shanedirksen): `--keymap
  R2=Left Alt+Mouse:Left` presses R2 when you click with Left Alt held, while a plain click keeps its own control. In
  the Windows launcher's keyboard editor, hold one input and press another to bind both; when a combo cannot be bound,
  the editor says why in the launcher's language (PR #365). Single-input bindings work as before.
- **Parallel translation (`KYTY_SPECULATE=4`, experimental), a follow-up by @CarpetHead** (PR #349): CPU lists such as
  `KYTY_RENDER_CPUS=0,8-15` now read their ranges (a range was cut short before); on Linux the state pass gets a core
  of its own; long sessions no longer meet single frames of 80–160 ms as the label set grows (also found by
  @Rufus7787, PR #356); and the commit checks more of what a worker read. The default mode is unchanged.
- **A frozen picture now leaves a line in the log** (#357): if the picture stops while the game goes on (the music
  plays), the log says after 5 s which step of handing frames to Windows is waiting, so a report can show where it
  hangs.
- **Thanks** to @supercc4e for the face report and tests, @CarpetHead and @shanedirksen for their PRs, @Rufus7787 for
  tracking down the label hitch, and @Bios2000 for testing #349 on Linux.

## New in v20261118

Your precompiled shaders stay valid. On graphics cards with the pixel interlock, precompiling once more brings the
decal fix below (only those shaders are compiled again).

- **Overlapping decals: the flickering square holes are fixed** (#341, #315): the candle wax in the Nexus and the marks
  at the foot of 1-3's walls showed holes of 2x2 pixels, different every frame. The game's decals read the surface data
  under them and write it back, one decal after another. The PS5 keeps them in order for each pixel, but a PC graphics
  card ran several at once, so a decal could overwrite the one before it. On cards with the pixel interlock (NVIDIA,
  Intel, and AMD on Linux with RADV) these shaders now run in pixel order, as on the PS5. No measurable cost (RTX 5090
  at the 120 Hz cap, without -> with: 1-1 66.8 -> 68.9 fps, Latria 73.9 -> 74.0, the 1-1 arena 113.1 -> 113.7; RX 9070 XT
  on Linux 60.0 -> 59.6).
  - **If you have already precompiled:** press Precompile on the launcher's Shaders page once more. Only the decal
    pipelines are compiled again; until then the game runs exactly as before. A new install gets the fix with its
    first precompile. `-Set KYTY_PIXEL_INTERLOCK=1` turns it on without a precompile; each decal shader then compiles
    the first time it appears, with a short stutter.
  - **AMD on Windows:** the driver has no pixel interlock. `-Set KYTY_DRAW_READ_AFTER_WRITE=1` is still the workaround
    (each decal waits for the one before; it costs frame rate where there are hundreds of decals).
  - `-Set KYTY_PIXEL_INTERLOCK=0` turns the fix off.
- **HDR: highlights fitted to your display** (PR #350, credit @Mustard-Tiger): at the game's default brightness its HDR
  highlights reach 2,500–3,000 nits, and a monitor clips everything above its own peak, so their detail was lost. With
  HDR on, the brightest part is now compressed into the display's peak (the BT.2390 curve, colours kept); darker and
  middle tones stay as they were. Windows supplies the display's peak (from the monitor, or from the Windows HDR
  Calibration app once you have run it). `-HdrMaxNits` sets the peak by hand; `-HdrSourceNits 0` brings back the old
  clipping. SDR is unchanged.
- **Less memory:** the emulator no longer keeps data it never reads again: the shader prefetch's pipeline recipes, a
  second copy of the warmup recording, an 8 MiB read window on every game thread that once read a large file from a
  .zar, the system font files (now read in place), a second copy of the ~224,000 file paths the game looks up, and the
  shader translator's work buffers once it is idle. At the start of 1-1 with the 16 GB PC settings (simulated, RTX
  5090), the emulator's heap is about 0.4 GB smaller and its working set about 0.45 GB smaller than in v20261117. The
  frame rate is the same.
- **A rare ordering gap closed** (credit @CarpetHead, who found it while reviewing the code for #349): a copy the game
  asks the GPU for (made on the CPU here) could change bytes that earlier work still had to upload, so that work used
  the new bytes instead of the old. Such work now finishes before the copy. No visible case is known; no cost measured.
- **Linux: GPU fault reports name the memory at the faulting address** (PR #353, credit @Bios2000): when the GPU faults
  (the game closes with "device lost"), the log now lists what was bound last at the faulting address, from the
  driver's own reports (RADV; `-Set KYTY_FAULT_BINDINGS=1` with other drivers). This is to track down #314.
- **Thanks** to @Bios2000 for the RADV tests and the fault report, @Mustard-Tiger for the HDR work, @CarpetHead for
  spotting the copy gap, and @zezoman and @robbely for the decal reports.

## 本版本的变化

已预编译的着色器依然有效。Windows 上的 AMD 显卡：用像素互锁生成的贴花着色器会在第一次出现时去掉互锁重新编译一次（或在下次预编译时编译）。

- **Windows 上的 AMD 显卡：角色的脸恢复正常**（#355，感谢 @supercc4e）：用 v20261118 并重新预编译后，Radeon 在 Windows 上会把玩家角色的脸显示成游戏的“缺失纹理”格子（绿底、紫色方点、红色斜线）。AMD 的 Windows 驱动其实支持 v20261118 开始用于贴花的像素互锁（上一版说明写的是不支持），开启后脸部就会出错。现在模拟器在 AMD 的 Windows 驱动上不再开启像素互锁，与 v20261118 之前相同。关闭后，#315/#341 里贴花重叠处的小孔可能再次出现；仍可用 `-Set KYTY_DRAW_READ_AFTER_WRITE=1` 规避，`-Set KYTY_PIXEL_INTERLOCK=1` 可强制开启互锁。之前为修脸加了 `-Set KYTY_PIXEL_INTERLOCK=0` 的可以去掉。
- **内存占用更少，长时间游玩尤其明显：** 着色器录制（预热时重放的着色器和管线列表，通关后约 100 MB）启动后不再常驻内存，每次保存时从磁盘上的文件拷贝已有部分，并用文件的校验和核对。着色器的资源分析改为整块存放，不再每个值单独分配一次（总数达数百万）；为重建而保留的管线只存它用到的顶点输入（每条约 1.6 KB，原为约 11 KB）；游戏查找的约 22.4 万个文件路径只存一份，.zar 读取器的各个读取通道共用一份索引表。在所有世界之间来回一小时（RTX 5090），与 v20261118 相比：模拟 16 GB 内存的设置下平均少用约 0.5 GB 内存，内存充足时平均少约 0.6 GB。帧数不变。
- **键盘鼠标：一个按键可绑定为两个输入的组合**（PR #360，感谢 @shanedirksen）：`--keymap R2=Left Alt+Mouse:Left` 表示按住左 Alt 再点鼠标左键时按下 R2，单独点左键仍是它原来的按键。Windows 启动器的键位编辑器里，按住一个输入再按另一个即可绑定组合；无法绑定时，编辑器会用启动器的语言说明原因（PR #365）。单个输入的绑定与以前相同。
- **并行翻译（`KYTY_SPECULATE=4`，实验功能），@CarpetHead 的后续修复**（PR #349）：`KYTY_RENDER_CPUS=0,8-15` 这类 CPU 列表现在能正确读取范围（以前范围会被截断）；Linux 上状态遍历线程有了自己的核心；长时间游玩时不再随着标签集合增大出现 80–160 ms 的单帧卡顿（@Rufus7787 在 PR #356 中也发现了这个问题）；提交时会核对更多工作线程读过的数据。默认模式不变。
- **画面冻结时日志会留下记录**（#357）：如果画面停住而游戏仍在运行（音乐还在播），5 秒后日志会写明把画面交给 Windows 的哪一步在等待，方便报告问题时定位。
- **感谢** @supercc4e 的报告和测试，@CarpetHead 和 @shanedirksen 的 PR，@Rufus7787 找到标签卡顿，以及 @Bios2000 在 Linux 上测试 #349。

## v20261118 的变化

已有的预编译着色器继续有效。在支持像素互锁（pixel interlock）的显卡上，再预编译一次即可获得下面的贴花修复（只会重新编译
这部分着色器）。

- **重叠贴花的闪烁方块修复了**（#341、#315）：Nexus 的蜡烛蜡油、1-3 墙脚的痕迹上会出现 2x2 像素的小洞，每帧位置都不同。
  游戏的贴花会读取下方的表面数据再写回去，一个接一个地画。PS5 对每个像素按顺序执行，而 PC 显卡会同时执行好几个，于是
  后一个贴花可能覆盖前一个的结果。在支持像素互锁的显卡上（NVIDIA、Intel，以及 Linux 上使用 RADV 的 AMD），这些着色器
  现在按像素顺序执行，和 PS5 一样。没有可测的性能开销（RTX 5090，120 Hz 上限，关闭 -> 开启：1-1 66.8 -> 68.9 fps，
  Latria 73.9 -> 74.0，1-1 竞技场 113.1 -> 113.7；Linux 上 RX 9070 XT 60.0 -> 59.6）。
  - **已经预编译过的：** 在启动器的"着色器"页再点一次"预编译"。只会重新编译贴花相关的管线；在此之前游戏和以前完全一样。
    新安装的在第一次预编译时就会带上这个修复。`-Set KYTY_PIXEL_INTERLOCK=1` 不预编译也能开启，之后每个贴花着色器第一次
    出现时会编译一次，有短暂卡顿。
  - **Windows 上的 AMD 显卡：** 驱动没有像素互锁，仍可用 `-Set KYTY_DRAW_READ_AFTER_WRITE=1` 规避（每个贴花等前一个画完，
    贴花很多的地方会掉帧）。
  - `-Set KYTY_PIXEL_INTERLOCK=0` 关闭这个修复。
- **HDR：高光按显示器亮度压缩**（PR #350，感谢 @Mustard-Tiger）：游戏默认亮度下 HDR 高光可达 2500–3000 尼特，而显示器会
  把超过自身峰值的部分全部截掉，高光细节因此丢失。开启 HDR 时，现在会把最亮的部分压缩到显示器的峰值以内（BT.2390 曲线，
  保持色彩），暗部和中间调不变。显示器的峰值由 Windows 提供（来自显示器本身，或运行过 Windows HDR 校准应用后的数值）。
  `-HdrMaxNits` 可手动指定峰值，`-HdrSourceNits 0` 恢复旧的截断方式。SDR 不受影响。
- **内存占用更少：** 模拟器不再保留之后用不到的数据：着色器预取的管线配方、预热录制的第二份拷贝、每个从 .zar 读过大文件
  的游戏线程各自保留的 8 MiB 读取窗口、系统字体文件（现在直接映射读取）、游戏查找的约 22.4 万个文件路径的第二份拷贝，
  以及着色器翻译器空闲后的工作缓冲。在 1-1 起点按 16 GB 电脑的设置测试（模拟，RTX 5090）：和 v20261117 相比，模拟器的
  堆内存少约 0.4 GB，工作集少约 0.45 GB。帧率不变。
- **修复一个罕见的顺序问题**（感谢 @CarpetHead 在 #349 中审阅代码时发现）：游戏交给 GPU 的复制（这里在 CPU 上完成）可能
  改写之前的工作还没上传的数据，那份工作就会用到新数据而不是旧数据。现在这些工作会在复制之前完成。目前没有已知的可见
  问题，也没有测到开销。
- **Linux：GPU 故障报告会写出故障地址上的内存**（PR #353，感谢 @Bios2000）：GPU 出错（游戏以 "device lost" 结束）时，
  日志现在会根据驱动自己的报告，列出故障地址上最后绑定的内存（RADV；其他驱动用 `-Set KYTY_FAULT_BINDINGS=1`）。用于排查
  #314。
- **感谢** @Bios2000 的 RADV 测试和故障报告功能，@Mustard-Tiger 的 HDR 改进，@CarpetHead 发现的复制问题，以及 @zezoman
  和 @robbely 的贴花报告。

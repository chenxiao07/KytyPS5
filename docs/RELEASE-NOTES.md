## New in this build

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

## New in v20261117

Your precompiled shaders stay valid: the shader translation is the same as in v20261116.

- **Malwarebytes no longer flags the launcher** (#346, PR #348, credit @DevList1454): its machine-learning check
  reported every `KytyLauncher.exe` since v20261105 as "MachineLearning/Anomalous". It reacted to compressed image data
  inside the exe (the keyboard prompt textures), which a program unpacks and writes like a dropper would. They are
  stored uncompressed now; the launcher works the same, and the exe is 3.3 MB instead of 0.35 MB (the download barely
  changes).
- **Parallel translation (`KYTY_SPECULATE=4`, experimental)**, a round by @CarpetHead (PR #342):
  - crossing into a new area stutters much less: in a fixed 18 s run across an area boundary in 1-3 (6 runs each, RTX
    5090), the ten frames after the crossing spent 139 -> 77 ms above the calm frame time, and the crossing frame took
    66 -> 55 ms. The commit of a translation now finishes more of its own work instead of stopping, its checks are
    exact, and the game's memory calls at a crossing no longer wait on the GPU thread.
  - with "Up to 120 fps" on, 13–29% of the frames at 5-1 were held to two refreshes (a frame a worker translated was not
    marked as the 3D view); now none are.
  - about a quarter of the boots had a storm of write faults (one page faulting every frame, about 3 fps lost per 100
    faults a frame); it no longer happens in this mode.
  - the write guard's wait for a written label no longer takes its lock over and over, which could hold up the upload
    worker that releases the label (seen as frames of 40–280 ms waiting for uploads in some test runs; they did not
    come back in the later runs with or without the change, so this one is unmeasured).
  - Linux (PR #345, credit @Bios2000): after a load step the game could crawl at about 5 fps or freeze. A page the write
    guard had protected was left without access and without an owner; this was also the cause of #289.
  - the default mode runs as before: everything above is off there.
- **Linux: NVMe power saving** (#343, credit @Bipboop-beep): if the game stalls for up to seconds at a time while it
  loads textures, with the SSD busy but reading little, the drive's power saving (APST) may be waking it too slowly. The
  package README now names the kernel parameter that turns it off (`nvme_core.default_ps_max_latency_us=0`).
- **Performance profile retrained** on the new code, so the default mode keeps its speed (RTX 5090, 120 Hz cap): the
  1-1 start 68.0 fps against 66.1 before this round, the 1-1 arena 114.6 against 112.9 with the old profile, Latria the
  same.
- **Thanks** to @CarpetHead for the parallel translation round, @Bios2000 for the Linux runs that found the write guard
  bug, @DevList1454 for tracking down the Malwarebytes flag, @Bipboop-beep for the NVMe measurements, and @Osbrin for
  the report.

## 本版本的变化

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

## v20261117 的变化

已有的预编译着色器继续有效：着色器翻译结果和 v20261116 相同。

- **Malwarebytes 不再误报启动器**（#346，PR #348，感谢 @DevList1454）：从 v20261105 起，Malwarebytes 的机器学习检测
  把每个版本的 `KytyLauncher.exe` 都报为 "MachineLearning/Anomalous"。触发它的是 exe 里压缩存放的图片数据（键位提示
  贴图）：程序把一段压缩数据解开再写到磁盘，看起来像是释放器。现在这些贴图不再压缩存放；启动器功能不变，exe 从 0.35 MB
  变成 3.3 MB（下载包大小几乎不变）。
- **并行翻译（`KYTY_SPECULATE=4`，实验功能）**，@CarpetHead 的一轮改进（PR #342）：
  - 进入新区域时的卡顿明显减少：在 1-3 跨越区域边界的固定 18 秒路线上（每边 6 次，RTX 5090），跨越后 10 帧比平稳帧多出的
    总时间 139 -> 77 ms，跨越那一帧 66 -> 55 ms。翻译的提交现在会把更多工作自己做完而不是中途停下，检查改为精确判断，
    跨越时游戏的内存调用也不再等待 GPU 线程。
  - 开启"最高 120 帧"时，5-1 有 13–29% 的帧被压到两个刷新周期（工作线程翻译的那一帧没有被标记为 3D 画面）；现在没有了。
  - 大约四分之一的启动会出现写入异常风暴（同一页每帧都触发异常，每帧每 100 次异常约损失 3 fps）；这个模式下不再出现。
  - 写保护在等待已写入的标签时不再反复加锁，以前这可能拖住负责解除标签保护的上传线程（部分测试运行中出现过 40–280 ms
    等待上传的帧；之后的运行中无论有没有这项改动都没有再出现，所以这一项没有测量数字）。
  - Linux（PR #345，感谢 @Bios2000）：读图的某一步之后，游戏可能只有约 5 fps 或直接卡住：写保护保护过的某一页被留成了
    不可访问且没有任何组件负责恢复；这也是 #289 的原因。
  - 默认模式不受影响：以上改动在默认模式下都不生效。
- **Linux：NVMe 节能**（#343，感谢 @Bipboop-beep）：如果读贴图时游戏一次卡顿最长达数秒，而硬盘很忙却几乎没读多少数据，
  可能是硬盘的节能功能（APST）唤醒太慢。Linux 包的 README 里现在写了关闭它的内核参数
  （`nvme_core.default_ps_max_latency_us=0`）。
- **重新训练了性能配置文件（PGO）**，适配新代码，默认模式保持原来的速度（RTX 5090，120 Hz 上限）：1-1 起点 68.0 fps，
  这一轮改动之前是 66.1；1-1 竞技场 114.6，旧配置文件下是 112.9；Latria 不变。
- **感谢** @CarpetHead 的并行翻译改进，@Bios2000 在 Linux 上的测试找出了写保护的问题，@DevList1454 查明了 Malwarebytes
  误报的原因，@Bipboop-beep 提供了 NVMe 的测量数据，以及 @Osbrin 的报告。

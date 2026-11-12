## New in this build

Your precompiled shaders stay valid: the shader translation is the same as in v20261110, except on AMD Radeon RX
400/500 and Vega cards (the first item below). Precompile again after updating on such a card.

- **RX 400/500 / Vega: corrupted cutscenes and movies** (#305, credit @FKF-GR): these cards run compute shaders only on
  64-lane subgroups, and the emulator emulated the game's 64-lane waves on 32-lane ones there, so two waves shared one
  subgroup. Compute programs now run natively at 64 lanes on such cards. Precompile again after updating.
- **Cards with 12 GB of video memory or less: fewer hitches** (#85, credit @brokeneggg): the game streamed its large
  textures whole even where the graphics card leaves their largest levels out, so the game's threads decoded and kept
  levels nothing drew. They are now streamed without those levels whatever the RAM. A 6 GB card with 24 GB of RAM
  (simulated, an hour of world changes): slow frames 163 -> 31, the worst frame 149 -> 60 ms, RAM growth +384 -> +21 MiB
  an hour. `-Set KYTY_TEXTURE_REDUCE_GAME=0` streams them whole as before.
- **Version 1.005: the freeze at King Allant's fog gate** (#327, credit @Lukeyrich for the log and the save): a guard
  the emulator adds to the game's resource loader on 1.005 (for #185/#204) could loop forever on a broken list entry.
  It now rebuilds that list and the loader carries on (the log says "loader guard: list ... rebuilt").
- **Xbox controllers: View opens the Toolbelt again** (#328, credit @kms810530-arch): the game opens it only when the
  touch pad click comes with a finger on the pad, as a DualSense reports it; View now sends both.
- **Game code runs with the PS5's floating-point settings** (port of upstream KytyPS5 f399d481 by Nawid3333): guest
  threads and fibers use the console's MXCSR (flush-to-zero, denormals-are-zero) instead of Windows' defaults. No
  speed change measured on the test PC. `-Set KYTY_GUEST_FP_ENV=0` restores the old behaviour.
- **Parallel translation (`KYTY_SPECULATE=4`, experimental): five fixes** (PRs #321 and #329, credit @CarpetHead): two
  races that could hand the GPU a stale buffer address, a freeze when a worker waited for GPU work only the render thread
  submits, an abort at area loads, and an exit on a rewritten command buffer.
- **Faster at 1-1**: the build's profile-guided optimization was trained again on this code: 1-1 at 120 fps on an RTX
  5090, 69.2 -> 73.6 fps (the render thread 13.8 -> 13.3 ms a frame); Latria and the arena unchanged. Protection changes
  of guest memory also find their mapping faster, and the render thread no longer reads its switches from the cache
  lines the scout thread keeps writing.
- **Linux: FSR 4 on Radeon cards** (PR #330, credit @Bipboop-beep): an opt-in for builds from source. `get-fsr4-linux.sh`
  builds ffx-vk-linux, which runs AMD's Windows FSR DLLs (you supply them, see the README), and the Linux launcher then
  offers "FSR 4". The release packages are built without it: there `KYTY_FSR4=1` logs why and FSR 3 is used.
- **Steam Deck guide**: how to update an existing install (#240).
- **Thanks** to @CarpetHead, @Bipboop-beep and @DevList1454 for the pull requests, and to @FKF-GR, @brokeneggg,
  @Lukeyrich and @kms810530-arch for the logs, saves and tests.

## New in v20261110

Your precompiled shaders stay valid: the shader translation is the same as in v20261109, except on AMD's older driver
for Radeon RX Vega and Polaris cards (the first item below).

- **RX Vega / Polaris with AMD's older driver: the black title screen** (#305, credit @FKF-GR): that driver ("AMD
  proprietary shader compiler" in the log's GPU line) refused every pipeline of 173 of the game's pixel shaders, the
  ones that read the GPU's barycentric coordinates without per-vertex values, the picture's post-processing among them.
  On that driver these shaders now get the same coordinates through AMD's own extension
  (VK_AMD_shader_explicit_vertex_parameter), which it lists. Other graphics cards translate them as before. Precompile
  again after updating on such a card. We could not run AMD's older driver here (the shaders pass validation and AMD's
  own compiler for Vega takes them), so please tell us in #305 whether the title shows.
- **New, experimental: parallel translation** (PR #293, credit @CarpetHead): with `-Set KYTY_SPECULATE=4` (run.cmd, or
  the launcher's Advanced > Extra options, "Command-line options"), worker threads translate the game's GPU work ahead
  of the render thread, which then only checks and commits it. Where the render thread limits the frame rate it is
  faster: +6% to +16% over the worlds in the author's test at 120 fps (Core i9-12900K, RTX 5090). It needs 28 GB of RAM
  or more (about 3.6 GB more than without it) and at least 9 logical CPUs on the performance cores; on other PCs it
  stays off and the log says why. Off by default; without it the emulator runs as before. Tested on Windows only.
- **A little faster**: the build's profile-guided optimization was trained again on this code. 1-1 at 120 fps on an
  RTX 5090: 70.3 -> 72.5 fps (the render thread 13.9 -> 13.4 ms a frame); Latria and the arena unchanged.
- **Thanks** to @CarpetHead for the pull request and the many rounds of measurements, and to @FKF-GR for the logs.

## 本版本的变化

已有的预编译着色器继续有效：着色器翻译结果和 v20261110 相同，只有 AMD Radeon RX 400/500 和 Vega 显卡例外（见第一条）。
这类显卡更新后请重新预编译。

- **RX 400/500 / Vega：过场动画和视频花屏**（#305，感谢 @FKF-GR）：这类显卡的计算着色器只能用 64 通道的子组运行，
  而模拟器在上面把游戏的 64 通道波次模拟在 32 通道的子组上，两个波次挤在同一个子组里。现在这类显卡上的计算程序直接以
  64 通道原生运行。更新后请重新预编译。
- **显存 12 GB 及以下的显卡：卡顿更少**（#85，感谢 @brokeneggg）：显卡本来就不保留大纹理最大的那一两级，游戏却仍然整张
  流送这些纹理，游戏线程解码并保存了没有任何绘制会用到的层级。现在不管内存多大，都不再流送这些层级。6 GB 显卡 + 24 GB
  内存（模拟，切换世界一小时）：慢帧 163 -> 31，最慢一帧 149 -> 60 ms，内存增长每小时 +384 -> +21 MiB。
  `-Set KYTY_TEXTURE_REDUCE_GAME=0` 恢复整张流送。
- **1.005 版：King Allant 雾门前卡死**（#327，感谢 @Lukeyrich 提供日志和存档）：模拟器为 1.005 的资源加载器加的保护
  （为 #185/#204 而加）遇到损坏的链表项时会无限循环。现在它会重建这个链表，加载器继续工作（日志会写
  “loader guard: list ... rebuilt”）。
- **Xbox 手柄：View 键重新能打开工具栏（Toolbelt）**（#328，感谢 @kms810530-arch）：游戏只在触摸板按下时同时有手指触点
  才打开它，就像 DualSense 报告的那样；现在 View 键两样都发。
- **游戏代码使用 PS5 的浮点设置**（移植自上游 KytyPS5 的 f399d481，作者 Nawid3333）：guest 线程和 fiber 使用 PS5 的 MXCSR
  （清零非规格化数），不再用 Windows 的默认值。测试电脑上速度没有变化。`-Set KYTY_GUEST_FP_ENV=0` 恢复旧行为。
- **并行翻译（`KYTY_SPECULATE=4`，实验功能）：五项修复**（PR #321、#329，感谢 @CarpetHead）：两处可能让 GPU 拿到过期
  缓冲区地址的竞争、工作线程等待只有渲染线程才会提交的 GPU 工作而卡死、切换区域时的中止，以及命令缓冲区被改写时的退出。
- **1-1 更快**：重新训练了这个版本的 PGO（按运行情况优化）。RTX 5090 在 1-1、120 fps 下：69.2 -> 73.6 fps（渲染线程每帧
  13.8 -> 13.3 ms）；Latria 和竞技场不变。另外 guest 内存的保护调用也能更快找到对应的映射；渲染线程读的开关也不再和
  scout 线程不停写入的变量挤在同一条缓存行上。
- **Linux：Radeon 显卡可用 FSR 4**（PR #330，感谢 @Bipboop-beep）：从源码编译时可选。`get-fsr4-linux.sh` 会编译
  ffx-vk-linux，用它运行 AMD 的 Windows 版 FSR DLL（需要自己准备，见 README），之后 Linux 启动器会提供“FSR 4”。
  发布包不含这项：在发布包里设 `KYTY_FSR4=1` 会在日志里说明原因并使用 FSR 3。
- **Steam Deck 指南**：补充了如何更新已有安装（#240）。
- **感谢** @CarpetHead、@Bipboop-beep 和 @DevList1454 的 PR，以及 @FKF-GR、@brokeneggg、@Lukeyrich、@kms810530-arch 提供的日志、存档和测试。

## v20261110 的变化

已有的预编译着色器继续有效：着色器的翻译与 v20261109 相同，只有 Radeon RX Vega、Polaris 显卡使用 AMD 旧驱动时例外
（见第一条）。

- **RX Vega / Polaris 配 AMD 旧驱动：标题画面黑屏**（#305，感谢 @FKF-GR）：这个驱动（日志 GPU 一行写着“AMD
  proprietary shader compiler”）会拒绝游戏中 173 个像素着色器的全部管线，也就是读取 GPU 重心坐标、但不读逐顶点数据的
  那些，画面的后处理也在其中。现在在这个驱动上，这些着色器改用 AMD 自己的扩展（VK_AMD_shader_explicit_vertex_parameter，
  驱动有列出）取得同样的坐标，其他显卡的翻译不变。这类显卡更新后请重新预编译。我们这里没法运行 AMD 的旧驱动（这些着色器
  通过了校验，AMD 自己的 Vega 编译器也能编译），所以请在 #305 告诉我们标题画面是否正常显示。
- **新增实验功能：并行翻译**（PR #293，感谢 @CarpetHead）：加上 `-Set KYTY_SPECULATE=4`（run.cmd，或启动器的高级 >
  附加参数“命令行参数”）后，工作线程会提前翻译游戏的 GPU 工作，渲染线程只需检查并提交。在渲染线程限制帧率的地方会更快：
  作者在 120 fps 下逐个世界测试，提升 6% 到 16%（Core i9-12900K、RTX 5090）。需要 28 GB 以上内存（比不开多用约 3.6 GB），
  以及性能核上至少 9 个逻辑处理器；不满足的电脑上它不会开启，日志会说明原因。默认关闭，不开时模拟器和以前一样。目前只在 Windows 上测试过。
- **稍快一些**：重新训练了这个版本的 PGO（按运行情况优化）。RTX 5090 在 1-1、120 fps 下：70.3 -> 72.5 fps（渲染线程
  每帧 13.9 -> 13.4 ms）；Latria 和竞技场不变。
- **感谢** @CarpetHead 的 PR 和多轮测量，以及 @FKF-GR 提供的日志。

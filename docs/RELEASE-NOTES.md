## New in this build

Your precompiled shaders stay valid: the shader translation is the same as in v20261113.

- **European 01.005.000: the freeze at King Allant's fog fixed for real** (#327, credit @Lukeyrich): the game's
  resource loader keeps its pending loads in ten lists, and v20261113's repair of a broken list looked at five of them,
  so the freeze at the fog could still happen. All ten are repaired now; a repair that fails says why in the log.
- **Xbox controllers: View now really opens the Toolbelt** (#332, credit @kms810530-arch): v20261113 made View click the
  touch pad with a finger at its exact middle, and Demon's Souls opens the menu of the half the finger is on (the
  Toolbelt on the left by default), so the middle opened nothing. View now touches the left half, as the keyboard's
  Backspace does. Tested with a virtual Xbox Series controller: View opens the Toolbelt, B closes it.
- **Linux launcher: GameMode switch** (PR #331, credit @Bipboop-beep): on the Advanced page, "GameMode" starts the game
  through `gamemoderun` when GameMode is installed (`GAMEMODE=1` for `run_ds.sh`), off by default. MangoHud is now off by
  default too (`MANGO=1` or its switch turns it on); a settings file that already has it on keeps it.
- **Linux with an AMD card (RADV): faster render thread** (PR #333, credit @Bios2000): RADV reads the GPU's progress
  with a system call, which the emulator did ~2,200 times a frame on its render thread at 1-1. A thread now watches the
  GPU's progress there by default (as on the Steam Deck already): 97.6 -> 110.8 fps at 1-1 on an RX 9070 XT.
  `KYTY_TIMELINE_WATCHER=0` brings the old behaviour back. Windows and NVIDIA are unchanged.
- **Thanks** to @Bipboop-beep and @Bios2000 for the pull requests and to @kms810530-arch and @Lukeyrich for the logs and the tests.

## New in v20261113

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

## 本版本的变化

已有的预编译着色器继续有效：着色器翻译结果和 v20261113 相同。

- **欧版 01.005.000：King Allant 雾门的卡死这次真正修好了**（#327，感谢 @Lukeyrich）：游戏的资源加载器把待加载的
  资源放在 10 条链表里，而 v20261113 的修复只检查了其中 5 条，所以过雾门时仍可能卡死。现在 10 条都会修复；
  万一修复失败，日志会写明原因。
- **Xbox 手柄：View 键现在真的能打开工具栏（Toolbelt）了**（#332，感谢 @kms810530-arch）：v20261113 让 View 键在触摸板
  正中间按下，而恶魔之魂按手指所在的半边打开对应菜单（默认左半边是工具栏），正中间不属于任何一边，所以什么都没打开。
  现在 View 键按在左半边，和键盘的 Backspace 一样。用虚拟的 Xbox Series 手柄测试过：View 打开工具栏，B 关闭。
- **Linux 启动器：GameMode 开关**（PR #331，感谢 @Bipboop-beep）：「高级」页新增「GameMode」，装了 GameMode 时通过
  `gamemoderun` 启动游戏（`run_ds.sh` 用 `GAMEMODE=1`），默认关闭。MangoHud 现在也默认关闭（`MANGO=1` 或它的开关打开）；
  设置文件里已经打开的保持打开。
- **Linux + AMD 显卡（RADV）：渲染线程更快**（PR #333，感谢 @Bios2000）：RADV 读取 GPU 进度要走一次系统调用，而模拟器在
  1-1 每帧要在渲染线程上读约 2,200 次。现在 RADV 下默认由一个专门线程监视 GPU 进度（Steam Deck 上早已如此）：RX 9070 XT
  在 1-1 从 97.6 fps 提高到 110.8 fps。`KYTY_TIMELINE_WATCHER=0` 恢复旧行为。Windows 和 NVIDIA 不受影响。
- **感谢** @Bipboop-beep 和 @Bios2000 的 PR，以及 @kms810530-arch 和 @Lukeyrich 提供的日志和测试。

## v20261113 的变化

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

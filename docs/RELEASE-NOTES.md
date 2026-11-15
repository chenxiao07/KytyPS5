## New in this build

Your precompiled shaders stay valid: the shader translation is the same as in v20261114.

- **Long sessions: the 0.2–15 s freezes fixed**: about once every one to two and a half hours of play, the game could
  stop for up to 15 s while about a hundred pipelines compiled at once (the pass that draws the world's surfaces had
  lost one of its six targets). The cause was in Windows: when one of the game's memory copies took a fault, Windows
  sometimes resumed it with the upper halves of its AVX registers cleared, and the copy wrote 32 zero bytes into the
  game's render state. A copy that took a fault is now made again. Two hours of world changes after the fix: no freeze.
- **Fewer stutters after world changes, faster loads**: the game's threads hand their memory maps, unmaps and
  readbacks to the emulator's GPU thread and wait for it, and that thread did not take them while it waited for a
  frame to be shown (at 60 fps, every frame it finished early). Now it does. Same session, switched on and off at each
  world change (RTX 5090, 60 fps): the game's threads waited 16.2 -> 0.9 ms a frame in play, the 1% low went
  53.0 -> 55.4 fps (25.6 -> 37.2 fps in the seconds when the game frees a world's textures), frames of 45 ms or more
  14 -> 6, and the world appeared 6.2 -> 5.0 s after a load. `-Set KYTY_SERVE_IN_FLIP_WAITS=0` brings the old
  behaviour back.
- **No more whole-emulator pauses while a world loads** (Windows): in the seconds after a world change, every thread of
  the emulator stopped together for 3–44 ms, several times a load. The upload worker, kept on the performance cores at
  the game's priority, lost its CPU to the game's threads while it held a lock that every memory map of the game needs.
  It now runs just above them: 22 such pauses in 6 loads before, none after; standing in 1-1 the frame rate is the same
  and the 1% low 65.1 -> 65.9 fps. `-Set KYTY_UPLOAD_PRIORITY=0` brings the old behaviour back.
- **Precompile: says what it is doing** (#334, credit @shark20619): it printed nothing for minutes. It now prints each
  step (the shader list, the plans, the shards, the merge) with the time since it began, and its progress line reads
  "shards: 12 of 1096 done (1%), 8 running, 1076 waiting". Windows and Linux.
- **Parallel translation (`KYTY_SPECULATE=4`, experimental)**:
  - no more render-thread stalls of 1–6 s: a lock the render thread shares with the translation workers only spun, and
    a worker that lost its CPU while holding it kept the render thread spinning until Windows boosted the worker, about
    four seconds later (about 20 frames of 0.3–5.7 s in 43 minutes on the test PC). A thread waiting for it now sleeps
    after a short spin: none in 36 minutes. The default mode measured the same.
  - the scout stands down while it is on (PR #339, credit @Bios2000): the workers translate the draws it prepares.
    Linux, 1-1: +3.7% fps on 16 CPUs, +8.0% on 12.
  - Linux (PRs #336, #337, #338, credit @Bios2000): the workers read game memory directly instead of one system call a
    read; four bytes of the draw state left uninitialized made some runs refuse nearly every draw in this mode; threads
    started by the render thread now leave its CPUs; long thread names are set again (15 characters at most).
- **Thanks** to @Bios2000 for the pull requests and to @shark20619 for the report and the tests.

## New in v20261114

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

## 本版本的变化

已有的预编译着色器继续有效：着色器翻译结果和 v20261114 相同。

- **长时间游玩：0.2–15 秒的卡死修复了**：大约每玩 1 到 2.5 小时，游戏可能会停住最多 15 秒，同时一次编译约一百个管线
  （绘制场景表面的那一遍渲染丢了 6 个渲染目标中的一个）。原因在 Windows：游戏的某次内存复制遇到缺页等异常时，Windows
  有时恢复执行后会把 AVX 寄存器的高半部分清零，于是这次复制往游戏的渲染状态里写进了 32 个字节的 0。现在遇到异常的复制
  会重新做一遍。修复后连续两小时切换世界：没有再卡死。
- **切换世界后的卡顿更少、读图更快**：游戏线程把内存映射/取消映射、回读交给模拟器的 GPU 线程处理并等待其完成，而 GPU
  线程在等待画面显示时（60 fps 下每一帧提前做完时都会这样）不处理这些请求；现在会处理。同一次运行中每次切换世界时交替
  开关对比（RTX 5090，60 fps）：游玩时游戏线程每帧等待 16.2 -> 0.9 ms，1% low 53.0 -> 55.4 fps（游戏释放一整个世界的
  纹理那几秒 25.6 -> 37.2 fps），45 ms 以上的帧 14 -> 6，读图后出现世界画面的时间 6.2 -> 5.0 秒。
  `-Set KYTY_SERVE_IN_FLIP_WAITS=0` 恢复旧行为。
- **读图时不再整个模拟器一起卡住**（Windows）：切换世界后的几秒里，模拟器的所有线程会一起停 3–44 ms，每次读图好几次。
  原因是上传线程固定在性能核上、优先级和游戏线程一样，拿着一把游戏每次映射内存都要用的锁时被游戏线程挤下 CPU。现在它的
  优先级略高于游戏线程：以前 6 次读图有 22 次这样的停顿，现在没有；1-1 站立时帧率不变，1% low 65.1 -> 65.9 fps。
  `-Set KYTY_UPLOAD_PRIORITY=0` 恢复旧行为。
- **预编译：会显示正在做什么**（#334，感谢 @shark20619）：以前好几分钟没有任何输出。现在每一步（列出着色器、规划、
  分片编译、合并）都会打印出来并带上已用时间，进度行改为 "shards: 12 of 1096 done (1%), 8 running, 1076 waiting"。
  Windows 和 Linux 都有。
- **并行翻译（`KYTY_SPECULATE=4`，实验功能）**：
  - 渲染线程不再卡 1–6 秒：渲染线程和翻译工作线程共用的一把锁以前只会自旋，持锁的工作线程被挤下 CPU 时，渲染线程会
    一直自旋，直到约 4 秒后 Windows 提升那个工作线程的优先级（测试电脑上 43 分钟内约 20 帧 0.3–5.7 秒）。现在等锁的线程
    短暂自旋后就睡眠：36 分钟内一次也没有。默认模式测下来不变。
  - 开启时 scout 线程会停下（PR #339，感谢 @Bios2000）：它准备的绘制本来就由工作线程翻译。Linux 1-1：16 个 CPU +3.7%，
    12 个 CPU +8.0%。
  - Linux（PR #336、#337、#338，感谢 @Bios2000）：工作线程直接读取游戏内存，不再每次读取都走一次系统调用；绘制状态里有
    4 个字节未初始化，导致某些运行中这个模式几乎拒绝所有绘制；渲染线程启动的线程现在会离开渲染线程的 CPU；较长的线程名
    也能设置了（最多 15 个字符）。
- **感谢** @Bios2000 的 PR，以及 @shark20619 的报告和测试。

## v20261114 的变化

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

## New in this build

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

## New in v20261116

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

## 本版本的变化

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

## v20261116 的变化

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

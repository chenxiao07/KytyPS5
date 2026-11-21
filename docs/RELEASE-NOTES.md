## New in this build

Your precompiled shaders stay valid.

- **Fewer hitches when the game frees memory** (#368): when the game gives back a block of memory the graphics card
  had written, the emulator copied those bytes back at once and waited for the graphics card first. In a player's log
  that took up to 11 ms in one go, about 0.1 s every 30 s, most of it at area loads. The copy now runs without the wait,
  and the bytes go back into game memory a moment later, while the emulator's render thread has nothing else to do;
  anything that touches that memory in between finds it as before. On 10 minutes of travel through every world, the
  render thread's time at frees went from 0.79 s to 0.16 s. `-Set KYTY_FREED_READBACK_ASYNC=0` brings back the old way.
- **No dark slabs in the distance after Continue** (PR #373, credit @CarpetHead): after quitting to the title and
  Continue, the far field showed dark hard-edged slabs over water and ground. The game builds its occlusion culling
  from a depth surface it fast-clears every frame; the emulator applied that clear only when the surface was next drawn
  into, so a reload culled with the previous load's occluders. The clear is now applied before the surface is read, as
  on the PS5. `-Set KYTY_SAMPLE_PENDING_HTILE_CLEAR=0` turns it off.
- **Linux, Mesa 26.3: no crash right after the world loads** (PR #370, credit @Bios2000): RADV in Mesa 26.3 crashed at
  the first depth-writing draw, in every mode. The emulator now turns on the driver feature RADV needs for it (on RADV
  only; other drivers are unaffected).
- **Keyboard prompts: the inventory's page-switch marks too** (PR #372, credit @shanedirksen, Windows launcher): with
  Keyboard and mouse prompts on, the square and triangle marks at the bottom of the inventory and status panels show
  your keys. The launcher writes them into the game's UI sheet in place and puts the original bytes back when you turn
  the option off; a sheet it doesn't recognise is left alone.
- **Parallel translation (`KYTY_SPECULATE=4`, experimental), round 3 by @CarpetHead** (PR #364): area crossings are
  shorter (the 1-3 crossing frame ~63 -> 46-51 ms on the author's PC) with fewer long frames, through more of the table
  work done by the workers and a new upload pipe, whose copy threads take only the cores the CPU has left (none on an
  8-core CPU); a worker that met an image replaced during an area load no longer ends the game (1 of 17 boots on
  Linux). The default mode is unchanged.
- **Thanks** to @Ludicrance for the screenshots and logs, @CarpetHead, @Bios2000 and @shanedirksen for their PRs, and
  @Bios2000 for the Linux tests of #364.

## New in v20261120

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
  the Windows launcher's keyboard editor, hold one input and press another to bind both; the editor's messages (a
  combo it cannot bind, a duplicate or unsupported key) now come in the launcher's language (PRs #365, #367).
  Single-input bindings work as before.
- **Parallel translation (`KYTY_SPECULATE=4`, experimental), a follow-up by @CarpetHead** (PR #349): CPU lists such as
  `KYTY_RENDER_CPUS=0,8-15` now read their ranges (a range was cut short before); on Linux the state pass gets a core
  of its own; long sessions no longer meet single frames of 80–160 ms as the label set grows (also found by
  @Rufus7787, PR #356); and the commit checks more of what a worker read. The default mode is unchanged.
- **A frozen picture now leaves a line in the log** (#357): if the picture stops while the game goes on (the music
  plays), the log says after 5 s which step of handing frames to Windows is waiting, so a report can show where it
  hangs.
- **Thanks** to @supercc4e for the face report and tests, @CarpetHead and @shanedirksen for their PRs, @Rufus7787 for
  tracking down the label hitch, and @Bios2000 for testing #349 on Linux.

## 本版本的变化

已预编译的着色器依然有效。

- **游戏释放内存时的卡顿更少了**（#368）：游戏归还一块显卡写过的内存时，模拟器以前会立刻把这些数据拷回来，而且要先等显卡做完。一位玩家的日志里单次最长 11 毫秒，每 30 秒累计约 0.1 秒，大多发生在读图时。现在拷贝不再等待，数据稍后在模拟器的渲染线程空闲时写回游戏内存；在此之前访问这块内存，看到的数据和以前一样。在 6 个世界之间往返 10 分钟，渲染线程花在释放上的时间从 0.79 秒降到 0.16 秒。`-Set KYTY_FREED_READBACK_ASYNC=0` 恢复旧的做法。
- **“继续游戏”后远景不再出现黑色块**（PR #373，感谢 @CarpetHead）：退回标题再“继续游戏”后，远处的水面和地面会出现边缘生硬的黑色块。游戏用一张每帧快速清除的深度图来做遮挡剔除，模拟器以前要等下次往这张图上画东西时才执行清除，于是重新读档后会拿上一次读档时的遮挡物来剔除。现在会在读取这张图之前先执行清除，和 PS5 一致。`-Set KYTY_SAMPLE_PENDING_HTILE_CLEAR=0` 可关闭。
- **Linux 上 Mesa 26.3：进入世界后不再崩溃**（PR #370，感谢 @Bios2000）：Mesa 26.3 的 RADV 在第一次写深度的绘制时崩溃，所有模式都会。模拟器现在开启 RADV 这里需要的驱动功能（只在 RADV 上开启，其他驱动不受影响）。
- **键盘提示：物品栏的翻页标记也换成按键了**（PR #372，感谢 @shanedirksen，Windows 启动器）：开启“键盘鼠标提示”后，物品栏和状态栏底部的方块、三角翻页标记也会显示你的按键。启动器直接改写游戏 UI 贴图里的这两处，关闭选项时还原原始数据；遇到认不出的贴图文件则不动它。
- **并行翻译（`KYTY_SPECULATE=4`，实验功能），@CarpetHead 的第三轮改进**（PR #364）：区域切换时的长帧更短更少（作者电脑上 1-3 切换帧约 63 -> 46-51 毫秒），因为更多的表格工作交给了工作线程，并新增了上传管线，其拷贝线程只使用 CPU 剩余的核心（8 核 CPU 上为 0）；工作线程在读图时遇到被替换的图像不会再导致游戏退出（Linux 上 17 次启动中出现 1 次）。默认模式不变。
- **感谢** @Ludicrance 的截图和日志，@CarpetHead、@Bios2000 和 @shanedirksen 的 PR，以及 @Bios2000 在 Linux 上测试 #364。

## v20261120 的变化

已预编译的着色器依然有效。Windows 上的 AMD 显卡：用像素互锁生成的贴花着色器会在第一次出现时去掉互锁重新编译一次（或在下次预编译时编译）。

- **Windows 上的 AMD 显卡：角色的脸恢复正常**（#355，感谢 @supercc4e）：用 v20261118 并重新预编译后，Radeon 在 Windows 上会把玩家角色的脸显示成游戏的“缺失纹理”格子（绿底、紫色方点、红色斜线）。AMD 的 Windows 驱动其实支持 v20261118 开始用于贴花的像素互锁（上一版说明写的是不支持），开启后脸部就会出错。现在模拟器在 AMD 的 Windows 驱动上不再开启像素互锁，与 v20261118 之前相同。关闭后，#315/#341 里贴花重叠处的小孔可能再次出现；仍可用 `-Set KYTY_DRAW_READ_AFTER_WRITE=1` 规避，`-Set KYTY_PIXEL_INTERLOCK=1` 可强制开启互锁。之前为修脸加了 `-Set KYTY_PIXEL_INTERLOCK=0` 的可以去掉。
- **内存占用更少，长时间游玩尤其明显：** 着色器录制（预热时重放的着色器和管线列表，通关后约 100 MB）启动后不再常驻内存，每次保存时从磁盘上的文件拷贝已有部分，并用文件的校验和核对。着色器的资源分析改为整块存放，不再每个值单独分配一次（总数达数百万）；为重建而保留的管线只存它用到的顶点输入（每条约 1.6 KB，原为约 11 KB）；游戏查找的约 22.4 万个文件路径只存一份，.zar 读取器的各个读取通道共用一份索引表。在所有世界之间来回一小时（RTX 5090），与 v20261118 相比：模拟 16 GB 内存的设置下平均少用约 0.5 GB 内存，内存充足时平均少约 0.6 GB。帧数不变。
- **键盘鼠标：一个按键可绑定为两个输入的组合**（PR #360，感谢 @shanedirksen）：`--keymap R2=Left Alt+Mouse:Left` 表示按住左 Alt 再点鼠标左键时按下 R2，单独点左键仍是它原来的按键。Windows 启动器的键位编辑器里，按住一个输入再按另一个即可绑定组合；编辑器的提示（无法绑定的组合、重复或不支持的按键）现在都使用启动器的语言（PR #365、#367）。单个输入的绑定与以前相同。
- **并行翻译（`KYTY_SPECULATE=4`，实验功能），@CarpetHead 的后续修复**（PR #349）：`KYTY_RENDER_CPUS=0,8-15` 这类 CPU 列表现在能正确读取范围（以前范围会被截断）；Linux 上状态遍历线程有了自己的核心；长时间游玩时不再随着标签集合增大出现 80–160 ms 的单帧卡顿（@Rufus7787 在 PR #356 中也发现了这个问题）；提交时会核对更多工作线程读过的数据。默认模式不变。
- **画面冻结时日志会留下记录**（#357）：如果画面停住而游戏仍在运行（音乐还在播），5 秒后日志会写明把画面交给 Windows 的哪一步在等待，方便报告问题时定位。
- **感谢** @supercc4e 的报告和测试，@CarpetHead 和 @shanedirksen 的 PR，@Rufus7787 找到标签卡顿，以及 @Bios2000 在 Linux 上测试 #349。

## New in this build

Your precompiled shaders stay valid.

- **Stream water no longer goes spiky after Continue** (PR #374, credit @CarpetHead): after quitting to the title and
  Continue, the water over a stream bed in the tutorial sometimes showed a jagged, spiked sheet (about 1 reload in 4).
  An old texture of the previous load stayed at the address of the water simulation's state, and the emulator handed
  it to the simulation in the wrong format, so the simulation started from that texture's bytes. Such an image is now
  replaced by a fresh one. `-Set KYTY_BLOCK_CLASS_OVERLAP_REPLACE=0` turns it off.

## New in v20261121

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

## 本版本的变化

已预编译的着色器依然有效。

- **“继续游戏”后溪水不再变成尖刺状**（PR #374，感谢 @CarpetHead）：退回标题再“继续游戏”后，教学关溪床上的水面有时会变成参差的尖刺面（约每 4 次重新读档出现 1 次）。上一次读档留下的旧纹理还登记在水面模拟状态的地址上，模拟器把它用错误的格式交给了水面模拟，模拟于是从这张纹理的数据开始。现在遇到这种情况会换成一张新的图像。`-Set KYTY_BLOCK_CLASS_OVERLAP_REPLACE=0` 可关闭。

## v20261121 的变化

已预编译的着色器依然有效。

- **游戏释放内存时的卡顿更少了**（#368）：游戏归还一块显卡写过的内存时，模拟器以前会立刻把这些数据拷回来，而且要先等显卡做完。一位玩家的日志里单次最长 11 毫秒，每 30 秒累计约 0.1 秒，大多发生在读图时。现在拷贝不再等待，数据稍后在模拟器的渲染线程空闲时写回游戏内存；在此之前访问这块内存，看到的数据和以前一样。在 6 个世界之间往返 10 分钟，渲染线程花在释放上的时间从 0.79 秒降到 0.16 秒。`-Set KYTY_FREED_READBACK_ASYNC=0` 恢复旧的做法。
- **“继续游戏”后远景不再出现黑色块**（PR #373，感谢 @CarpetHead）：退回标题再“继续游戏”后，远处的水面和地面会出现边缘生硬的黑色块。游戏用一张每帧快速清除的深度图来做遮挡剔除，模拟器以前要等下次往这张图上画东西时才执行清除，于是重新读档后会拿上一次读档时的遮挡物来剔除。现在会在读取这张图之前先执行清除，和 PS5 一致。`-Set KYTY_SAMPLE_PENDING_HTILE_CLEAR=0` 可关闭。
- **Linux 上 Mesa 26.3：进入世界后不再崩溃**（PR #370，感谢 @Bios2000）：Mesa 26.3 的 RADV 在第一次写深度的绘制时崩溃，所有模式都会。模拟器现在开启 RADV 这里需要的驱动功能（只在 RADV 上开启，其他驱动不受影响）。
- **键盘提示：物品栏的翻页标记也换成按键了**（PR #372，感谢 @shanedirksen，Windows 启动器）：开启“键盘鼠标提示”后，物品栏和状态栏底部的方块、三角翻页标记也会显示你的按键。启动器直接改写游戏 UI 贴图里的这两处，关闭选项时还原原始数据；遇到认不出的贴图文件则不动它。
- **并行翻译（`KYTY_SPECULATE=4`，实验功能），@CarpetHead 的第三轮改进**（PR #364）：区域切换时的长帧更短更少（作者电脑上 1-3 切换帧约 63 -> 46-51 毫秒），因为更多的表格工作交给了工作线程，并新增了上传管线，其拷贝线程只使用 CPU 剩余的核心（8 核 CPU 上为 0）；工作线程在读图时遇到被替换的图像不会再导致游戏退出（Linux 上 17 次启动中出现 1 次）。默认模式不变。
- **感谢** @Ludicrance 的截图和日志，@CarpetHead、@Bios2000 和 @shanedirksen 的 PR，以及 @Bios2000 在 Linux 上测试 #364。

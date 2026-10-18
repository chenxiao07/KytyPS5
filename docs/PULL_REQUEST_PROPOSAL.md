# Pull Request Proposal: Enable Experimental RTX 30 Series Frame Generation via OpenDLSSG Enabler Detection

## Summary
This PR enables compatibility for NVIDIA GeForce RTX 30-series (Ampere) graphics cards to utilize NVIDIA Streamline DLSS Frame Generation when a third-party architecture spoofer / kernel retargeter (such as [OpenDLSSG Enabler](https://github.com/rodjunger/opendlssgenabler) via `version.dll` proxy) is present.

This change is **strictly opt-in and non-invasive**:
- **Zero Proprietary Blobs:** Does not bundle or distribute any third-party or NVIDIA binaries.
- **Graceful Detection:** Automatically detects if `version.dll` or `opendlssg.ini` is present next to `kyty_emulator.exe`.
- **Informative Diagnostics:** If Streamline reports an unsupported adapter on an RTX 30 card, the console advises the user on driver status and ensuring Hardware-Accelerated GPU Scheduling (HAGS) is enabled.
- **UI & Automation:** The launcher updates its frame generation note dynamically (`DLSS; RTX 40+ native, RTX 30 with OpenDLSSG Enabler`) and passes `-Rtx30FrameGen` (`KYTY_EXPERIMENTAL_RTX30_FG=1`).

---

## Benchmark & Proof
Tested on **Demon's Souls PS5 (PPSA01341 01.007.000)** at native 1440p (`2560x1440` resolution, `1440` render height):
- **Hardware:** AMD Ryzen 7 5800X3D, NVIDIA GeForce RTX 3070 (8 GB), 32 GB RAM, 240Hz monitor.
- **Performance:** Native 30 FPS doubled to **60 FPS rock-solid** with `2x` Frame Generation.
- **Latency:** Total latency **33.0 ms** via NVIDIA Reflex.
- **VRAM:** 7.1 GB emulator / 7.7 GB total (fits cleanly inside 8GB VRAM envelope).

---

## Files Changed
1. `src/local/frame-gen.h`:
   - Declared `OpenDlssgDetected()` and `ExperimentalRtx30Enabled()`.
2. `src/local/frame-gen.cpp`:
   - Implemented `OpenDlssgDetected()` and `ExperimentalRtx30Enabled()`.
   - Updated `SupportProblem()`, `Initialize()`, and `OnDevice()` with diagnostic logging for RTX 30 / OpenDLSSG.
3. `run-windows.ps1`:
   - Added `-Rtx30FrameGen` switch (`KYTY_EXPERIMENTAL_RTX30_FG=1`).
4. `launcher.ps1`:
   - Updated `Update-FrameGenNote` to reflect RTX 30 compatibility when proxy is detected.

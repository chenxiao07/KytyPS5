#!/bin/bash
# Production switch set for Demon's Souls (from run-windows.json) + best binary.
cd /mnt/AI/PS5-chenxiao07 || exit 1
export KYTY_SHADER_WARMUP=1
export KYTY_GPU_BUFFER_PAGES=1
export KYTY_RECORDING_CPUS=auto
export KYTY_SRT_NATIVE=1
export KYTY_SRT_PREDICATES=1
export KYTY_PREPARATION_SCRATCH=1
export KYTY_PREPARATION_LOOKUP=1
export KYTY_PREPARATION_TRIM=1
export KYTY_SPECIALIZATION_GUARD=1
export KYTY_PIPELINE_INDEX=1
export KYTY_BINDING_SCRATCH=1
export KYTY_DRAW_RUN_RANGES=1
export KYTY_BUFFER_RESIDENCY=1
export KYTY_COPY_FEEDBACK=1
export KYTY_ASYNC_LOD_STATS=1
export KYTY_BACKING_READ=1
export KYTY_STREAM_UPLOAD=1
export KYTY_FRAME_PIPELINE=1
export KYTY_IMAGE_BARRIER_DEDUPE=1
export KYTY_IMAGE_POOL=1
export KYTY_PENDING_DRAIN=1
export KYTY_DISPATCH_BATCH=128
export KYTY_VULKAN_RECORDING=1
export KYTY_DEFERRED_SUBMIT=1
export KYTY_NATIVE_XPR=1
export KYTY_WRITE_WINDOW_HANDOFF=1
export KYTY_READBACK_DETACH=1
export KYTY_ASYNC_WRITE_READBACK=1
export KYTY_READBACK_SLOTS=1
export KYTY_ASYNC_UPLOAD=2
export KYTY_BDA_DIRTY_REGIONS=1
export KYTY_GLOBAL_BARRIER_DEDUPE=1
export KYTY_RANGE_SET_FAST=1
export KYTY_IMAGE_GRANULES=2
export KYTY_NATIVE_XPR_PREDICT=1
export KYTY_NATIVE_IMAGE_PROOF=1
export KYTY_TEXTURE_RESOLVE_PAGES=1
export KYTY_ASYNC_REPROTECT=1
export KYTY_READBACK_NARROW=1
export KYTY_SHADER_WARMUP_THREADS=16
export KYTY_READBACK_QUEUE=1
export KYTY_TABLE_XPR=2
export KYTY_TABLE_DISPATCH=1
# The game renders 2560x1440 internally and downsamples into the 1080p window. Rendering at the
# output resolution instead cuts GPU pixel work ~1.8x with no visible change (the window shows 1080p
# either way). 900 = 1600x900 -> 1080p for more fps; 1440 restores the original behavior.
export KYTY_RENDER_HEIGHT=${KYTY_RENDER_HEIGHT:-1080}
# Live VRAM-budget GC: texture/buffer collection thresholds follow the current device-memory budget
# (prints KYTYGCBUDGET lines) instead of fixed startup values.
export KYTY_GC_LIVE_BUDGET=1
# GPU render thread pinned off CPU 0 (interrupt-heavy); physical cores 1-7.
export KYTY_RENDER_CPUS=${KYTY_RENDER_CPUS:-"1,2,3,4,5,6,7"}
BIN="${KYTY_BIN:-./_Build/linux-nt/kyty_emulator}"
# NOTE: default is the current (linux-nt) build, NOT the stale Oct 7 _Build/linux one.
# Hands-off stutter capture (KYTY_LIVE_FILE): a fresh frame-interval (flips) file every ~12 s.
# The game writes the last 3000 presented-frame intervals (us) per file, so any hitch is recorded
# without sending any input. Analyse with tools/local/stutter-report.py (or manually).
# System monitor for the stutter diagnosis: 1 Hz memory / swap / PSI-memory.
(
  while :; do
    pid=$(pgrep -x kyty_emulator | head -1)
    rss=0; [ -n "$pid" ] && rss=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null)
    read -r avail swt swf < <(awk '/MemAvailable/{a=$2}/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print a,t,f}' /proc/meminfo)
    psi=$(awk -F'avg10=' '/^some/{split($2,x," ");print x[1]}' /proc/pressure/memory 2>/dev/null)
    printf '%s rss_mb=%d avail_mb=%d swap_used_mb=%d psi_mem=%s\n' "$(date +%H:%M:%S)" \
      "$((rss/1024))" "$((avail/1024))" "$(( (swt-swf)/1024 ))" "$psi"
    sleep 1
  done
) > /mnt/AI/tmp/opencode/monitor9.log 2>&1 &
if [ -n "$KYTY_LIVE_FILE" ]; then
  mkdir -p /mnt/AI/tmp/opencode/flips
  printf 'id 0\n' > "$KYTY_LIVE_FILE"
  (
    sleep 6; n=0
    while :; do
      n=$((n+1))
      printf 'id %d\nflips 3000 /mnt/AI/tmp/opencode/flips/%05d_%s.txt\n' "$n" "$n" "$(date +%H%M%S)" \
        > "$KYTY_LIVE_FILE.tmp" && mv "$KYTY_LIVE_FILE.tmp" "$KYTY_LIVE_FILE"
      sleep 12
    done
  ) &
fi
# Mailbox = vblank-aligned, no tearing, low latency (smooth). Override with PRESENT_MODE=Fifo|Immediate.
# MANGO=1 (default): MangoHud FPS/CPU/GPU overlay; MANGO=0 disables it.
RUN=("$BIN")
if [ "${MANGO:-1}" = 1 ] && command -v mangohud >/dev/null 2>&1; then
  RUN=(mangohud "$BIN")
  export MANGOHUD_CONFIG=${MANGOHUD_CONFIG:-fps,frame_timing,vram,ram,gpu_temp,cpu_temp,gpu_power,cpu_power}
fi
exec "${RUN[@]}" --game "/mnt/4tb/REtro/ps5/Demon's Souls.zar" \
  --screen-width 1920 --screen-height 1080 --console-language 1 \
  --shader-validation false --shader-log-direction Silent --printf-direction Silent \
  --vblank-frequency 60 --present-mode "${PRESENT_MODE:-Mailbox}" "$@"
#!/usr/bin/env python3
"""spike-walk-read.py [--dir DIR] LABEL... : the crossing-window rows of spike-walk.ps1 boots (tools/local/SPIKE-WALK.md).

Per walk, from its flipsx dump (each frame's time and buffer registrations) and the live replies (LIVE_PAD flip=, LIVE_FLIPSX
last=, LIVE_COUNTERS):
  calm             mean frame time of the 60 frames before the crossing's minus one (ms)
  crossing ms      the crossing frame: the most buffer registrations within 10 frames of the detector's spike (the first
                   frame over 50 ms from 8.5 s after the run's start; else the longest frame 8.5-11.5 s in)
  window           the 21 frames around the crossing: worst frame, sum of time over calm, frames over 25 / 33 ms
  tail             the 10 frames after the crossing: peak, sum over calm, frames over 25 / 33 ms
  2nd window       the same around the longest frame from 11 frames after the crossing to the run's end (18 s)
  write faults     the mean write_faults a frame over the walk's measure windows: over 100 a STORM boot, 40-100 MILD
Then, beside those (printed after them in both tables; "-" when the walk is too short for the row):
  late burst       the worst frame +30..+60 frames after the crossing frame, and the sum of time over calm from that
                   frame to +10 frames after it (11 frames)
  plateau ms       the mean frame time over frames +60..+120 after the crossing
  walk             the run (the pad's start to the release, 18 s): mean frame time, p95, p99 (nearest rank), frames over 25 ms
  stand ms         the mean frame time over the -Tail standing windows (their LIVE_MEASURE replies: time over frames);
                   "-" for a boot run with -Tail 0
Boots flagged STORM are listed but left out of the means (their window rows measure the storm, not the crossing).
Each walk's end position (its last LIVE_WHERE before the flipsx reply) is checked against the bridge's (z 390.3, SPIKE-WALK.md):
a walk that ends elsewhere gets a WARNING line (a stuck run, a wrong start); no reply, or ok=0,
"where failed".
Labels are spike-walk.ps1's -Label; each walk of a boot is LABEL-wN. Means and standard errors over the labels given.
"""
import math
import re
import statistics as S
import sys
from pathlib import Path

END_Z, END_Z_TOLERANCE = 390.3, 1.5  # the run's end on the bridge: z 390.0-390.6, within 0.1 m walk to walk
# The rows beside the crossing ones: kept out of the first table's key list and printed after it, "-" when None.
NEW_ROWS = ['late burst worst ms', 'late burst over-calm ms', 'plateau ms', 'walk mean ms', 'walk p95 ms', 'walk p99 ms',
            'walk >25', 'stand ms']


def nearest_rank(v, p):
    s = sorted(v)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def new_rows(r, ft, x, pad, release, calm, live, walk):
    """The late burst, plateau, whole-walk and stand rows (None: the walk is too short for the row)."""
    r.update({k: None for k in NEW_ROWS})
    if x + 60 < release:
        j = max(range(x + 30, x + 61), key=lambda k: ft[k])
        r['late burst worst ms'] = ft[j]
        if j + 10 < release:
            r['late burst over-calm ms'] = sum(max(0.0, v - calm) for v in ft[j:j + 11])
    if x + 120 < release:
        r['plateau ms'] = S.mean(ft[x + 60:x + 121])
    if release < len(ft):  # the dump holds the whole 18 s run
        run = ft[pad:release]
        r['walk mean ms'] = S.mean(run)
        r['walk p95 ms'] = nearest_rank(run, 95); r['walk p99 ms'] = nearest_rank(run, 99)
        r['walk >25'] = sum(v > 25 for v in run)
    # The standing windows (measure 0.5 <walk>tNN after the run): each reply's time is frames / fps (seconds is rounded
    # to 0.01 s); a window with no frame counts its seconds.
    stand = [re.search(r' seconds=([0-9.]+) frames=(\d+) fps=([0-9.]+)', l) for l in live
             if l.startswith('LIVE_MEASURE') and re.search(rf' label={walk}t\d\d ', l)]
    frames = sum(int(m.group(2)) for m in stand if m)
    if frames:
        secs = sum(int(m.group(2)) / float(m.group(3)) if int(m.group(2)) else float(m.group(1)) for m in stand if m)
        r['stand ms'] = 1000 * secs / frames


def walk_rows(d, label, walk):
    live = (d / f'{label}.live.txt').read_text(errors='replace').splitlines()
    fx = [l.split() for l in (d / f'{label}-{walk}.flipsx.txt').read_text().splitlines()]
    head, frames = fx[0], [x for x in fx[1:] if len(x) == 2]
    ft = [int(x[0]) / 1000 for x in frames]
    regs = [int(x[1]) for x in frames]
    # Where this walk's lines are: from its first measure window to its flipsx reply.
    tag = f'{walk}c00'
    start = next(i for i, l in enumerate(live) if f'label={tag}' in l or f' {tag}' in l)
    end = next(i for i in range(start, len(live)) if live[i].startswith('LIVE_FLIPSX'))
    wheres = [l for l in live[start:end] if l.startswith('LIVE_WHERE')]
    z = re.search(r' z=(-?[0-9.]+)', wheres[-1]) if wheres else None
    if not wheres:
        print(f'WARNING {label} {walk}: where failed (no LIVE_WHERE reply before the flipsx one)')
    elif ' ok=1 ' not in wheres[-1] or z is None:
        print(f'WARNING {label} {walk}: where failed at the run\'s end (no end position read)')
    elif abs(float(z.group(1)) - END_Z) > END_Z_TOLERANCE:
        print(f'WARNING {label} {walk}: the run ended at z={float(z.group(1)):.1f}, not on the bridge (z {END_Z}); '
              f'check the run (stuck, or a wrong start: SPIKE-WALK.md)')
    last = int(re.search(r'last=(\d+)', live[end]).group(1))
    pad_line = max(i for i in range(start) if live[i].startswith('LIVE_PAD'))
    pad_flip = int(re.search(r'flip=(\d+)', live[pad_line]).group(1))
    pad = pad_flip - (last - len(ft))  # the run's first frame in the dump
    if not 0 <= pad < len(ft):
        raise ValueError(f'{label} {walk}: the run is not in the dump')
    t, release = 0.0, len(ft)
    for i in range(pad, len(ft)):
        t += ft[i]
        if t >= 18000: release = i; break
    t, first = 0.0, None
    for i in range(pad, release):
        if t >= 8500 and ft[i] > 50: first = i; break
        t += ft[i]
    if first is None:
        t, best = 0.0, None
        for i in range(pad, release):
            if 8500 <= t <= 11500 and (best is None or ft[i] > ft[best]): best = i
            t += ft[i]
        first = best
    x = max(range(max(first - 10, 0), min(first + 11, len(ft))), key=lambda k: regs[k])
    calm = S.mean(ft[x - 61:x - 1])
    r = {'calm ms': calm, 'crossing ms': ft[x], 'crossing at (vs detector)': x - first, 'detector spike ms': ft[first]}

    def win(j, tag):
        w = ft[j - 10:j + 11]
        r[f'{tag} worst ms'] = max(w); r[f'{tag} over-calm ms'] = sum(max(0.0, v - calm) for v in w)
        r[f'{tag} >25'] = sum(v > 25 for v in w); r[f'{tag} >33'] = sum(v > 33 for v in w)

    win(x, 'window')
    tl = ft[x + 1:x + 11]
    r['tail peak ms'] = max(tl); r['tail over-calm ms'] = sum(max(0.0, v - calm) for v in tl)
    r['tail >25'] = sum(v > 25 for v in tl); r['tail >33'] = sum(v > 33 for v in tl)
    if release > x + 21:
        win(max(range(x + 11, release - 10), key=lambda k: ft[k]), '2nd window')
    wf = [float(m.group(1)) for l in live[start:end] if l.startswith('LIVE_COUNTERS')
          for m in [re.search(r' write_faults=([0-9.]+)', l)] if m]
    r['write faults a frame'] = S.mean(wf) if wf else float('nan')
    new_rows(r, ft, x, pad, release, calm, live[start:end], walk)
    return r


def main(argv):
    d = Path('_Build/spike-walk')
    if len(argv) > 2 and argv[1] == '--dir':
        d, argv = Path(argv[2]), argv[:1] + argv[3:]
    res = {}
    for label in argv[1:]:
        for f in sorted(d.glob(f'{label}-w*.flipsx.txt')):
            walk = f.name[len(label) + 1:].split('.')[0]
            try:
                res[f'{label}-{walk}'] = walk_rows(d, label, walk)
            except (ValueError, StopIteration, IndexError) as e:
                print(f'{label} {walk}: not read ({e})')
    if not res:
        sys.exit(__doc__)
    keys = [k for k in next(iter(res.values())) if k not in NEW_ROWS]
    storm = {k for k, r in res.items() if r['write faults a frame'] > 100}
    mild = {k for k, r in res.items() if 40 <= r['write faults a frame'] <= 100}
    print(f'{"row":26}' + ''.join(f'{k[-16:]:>17}' for k in res))
    for key in keys:
        print(f'{key:26}' + ''.join(f'{r.get(key, math.nan):17.1f}' for r in res.values()))
    for key in NEW_ROWS:
        print(f'{key:26}' + ''.join(f'{"-":>17}' if r[key] is None else f'{r[key]:17.1f}' for r in res.values()))
    print('flag' + ' ' * 22 + ''.join(f'{"STORM" if k in storm else "mild" if k in mild else "calm":>17}' for k in res))
    use = [r for k, r in res.items() if k not in storm]
    print(f'\nmean +/- se over {len(use)} walks (STORM walks left out: {len(storm)})')
    for key in keys:
        v = [r[key] for r in use if key in r and not math.isnan(r[key])]
        if v:
            se = S.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else math.nan
            print(f'{key:26}{S.mean(v):10.2f} +/- {se:.2f}')
    for key in NEW_ROWS:
        v = [r[key] for r in use if r[key] is not None]
        se = S.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else math.nan
        print(f'{key:26}' + (f'{S.mean(v):10.2f} +/- {se:.2f}' + (f' ({len(v)} walks)' if len(v) < len(use) else '')
                             if v else f'{"-":>10}'))


if __name__ == '__main__':
    main(sys.argv)

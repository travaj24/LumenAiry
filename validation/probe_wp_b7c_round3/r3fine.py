"""The FINE ladders -- the bar's two margins measured at the step the
reading's own excursion lives on.

The round-2 verification's section 2.3 found that near a fold onset the
reading is not smooth in ``z``: on its fast singlet it read
1.0002 -> 1.1810 -> 0.9989 over 20 nm of defocus, and the FIELD moved with
it.  A band ladder at 20 um steps cannot see that, so the two numbers the
bar's margins ARE -- the largest reading it returns and the smallest it
refuses -- would otherwise be a property of the step, not of the optic.

This probe reads a coarse band scan, finds on each optic the pair of
neighbouring planes that straddles the bar (or, if none does, the two planes
closest to it from each side), and re-scans between them at 30-60x the step.
Nothing is hard-coded per optic.

Usage:
    python r3fine.py <tag> <pool> <bar> -- <band scan json> ...
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import r3join

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, '..', '..'))


def brackets(rows, bar, n=11):
    """Per fixture, the z interval that straddles the bar."""
    by = {}
    for r in rows:
        if r.get('pixel_continuity') is None or 'error' in r:
            continue
        by.setdefault(r['fixture'], []).append(r)
    out = {}
    for nm, rs in by.items():
        rs.sort(key=lambda r: r['z_um'])
        pairs = []
        for a, b in zip(rs, rs[1:]):
            ca, cb = a['pixel_continuity'], b['pixel_continuity']
            if (ca <= bar) != (cb <= bar):
                pairs.append((a['z_um'], b['z_um']))
        if not pairs:
            # no crossing on this optic: refine around the plane whose
            # reading is CLOSEST to the bar from below, which is the one that
            # sets this optic's contribution to the margin above
            rs2 = sorted(rs, key=lambda r: abs(r['pixel_continuity'] - bar))
            z = rs2[0]['z_um']
            i = [r['z_um'] for r in rs].index(z)
            lo = rs[max(0, i - 1)]['z_um']
            hi = rs[min(len(rs) - 1, i + 1)]['z_um']
            if hi > lo:
                pairs = [(lo, hi)]
        zs = []
        for lo, hi in pairs[:3]:
            step = (hi - lo) / float(n - 1)
            zs += [lo + k * step for k in range(n)]
        if zs:
            out[nm] = sorted(set(round(z, 6) for z in zs))
    return out


def main():
    tag, pool, bar = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    paths = [a for a in sys.argv[4:] if a != '--']
    rows = list(r3join.collect(paths).values())
    br = brackets(rows, bar)
    print(json.dumps({k: len(v) for k, v in br.items()}, indent=1))
    print('total', sum(len(v) for v in br.values()), flush=True)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=TREE)
    queue = sorted(br.items())
    running = []
    t0 = time.time()
    while queue or running:
        while queue and len(running) < pool:
            nm, zs = queue.pop(0)
            out = os.path.join(HERE, f'fine_{nm}_{tag}_win.json')
            log = os.path.join(HERE, f'log_fine_{nm}_{tag}_win.txt')
            f = open(log, 'w', encoding='cp1252')
            cmd = [sys.executable, os.path.join(HERE, 'r3scan.py'), nm,
                   ','.join('%.6f' % z for z in zs), out]
            p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=f, stderr=f)
            running.append((nm, p, f, time.time()))
            print('start', nm, len(zs), flush=True)
        time.sleep(3.0)
        for rec in list(running):
            nm, p, f, ts = rec
            if p.poll() is not None:
                f.close()
                running.remove(rec)
                print('done', nm, p.returncode, '%.1fs' % (time.time() - ts),
                      flush=True)
    print('ALL DONE %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()

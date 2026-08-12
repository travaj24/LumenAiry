# DISCRIMINATOR: at 32 orders the like-for-like `crop` leg exceeds the
# campaign's 4e-05 energy bar against the shipped per-order path.  Is that a
# defect in the staged pipeline, or is it the crop window truncating the
# NEIGHBOURING beams of a filled fan?
#
# THE HYPOTHESIS, and it is the probe's own.  PROBE_SUM_AT_APERTURE S6.4:
# "crop cuts a 4.738 mm window out of the SUM, truncating the neighbouring
# beams mid-aperture, and a hard truncation of a neighbour diffracts into this
# frame.  full truncates nothing and reports the genuine tail."  It measured
# a 35x gap between the two legs on THREE orders, which were far apart.  On the
# full 32-order fan every frame has up to eight neighbours at a 480 um pitch
# and the 4.74 mm crop always cuts several of them mid-aperture, so the same
# mechanism should be far larger -- and it should VANISH on the `full` leg.
#
# THE TEST.  One summed field (the pipeline's own 32-beam checkpoint, read
# back), one common carrier, the SAME frames, two legs:
#
#   crop -- the pipeline's own resolved plan, read from its leg_crop.json
#   full -- one leg on the whole 10.07 mm summed aperture, truncating nothing
#
# each scored against that order's shipped per-order tile (the chain's own
# exact readout, checkpointed by the pipeline's chains stage).  If the excess
# is the crop truncation, `full` collapses it; if it is the aggregation, both
# legs carry it.
#
# This is a PROBE, not a pipeline stage: it reads checkpoints and calls the
# library directly, so it writes no keyed artifact and cannot disturb the run
# it is measuring.
#
#   python pipeline_legmode_32_121.py [--workdir DIR] [--frames k,k,...]
#
# cp1252-safe ASCII only.
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
for _p in (_ROOT, os.path.join(_ROOT, 'validation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

warnings.filterwarnings('ignore', message='.*prescription aperture.*')
warnings.filterwarnings('ignore', message='.*residual transverse.*')
warnings.filterwarnings('ignore', message='.*under-sampled.*')

from lumenairy.propagators.carrier import (  # noqa: E402
    carrier_referenced_exact_focus_readout)
from lumenairy.propagators.carrier_field import (  # noqa: E402
    load_carrier_field_zarr)
from pipeline.metrics import compare, spot  # noqa: E402

LAM = 1.31e-6
TRAILING = 7.7058e-3
DXO = 0.2e-6
TILE = 1024
RAMB = float('inf')


def _payload(path):
    with open(path, 'r', encoding='cp1252') as fh:
        return json.load(fh)['payload']


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', default=os.path.join(
        _ROOT, 'validation', 'pipeline', '_work', 'd121_32order'))
    ap.add_argument('--frames', default=None,
                    help='comma-separated beam keys; default = the 4 worst '
                         'crop-leg energy deltas plus the on-axis frame')
    ap.add_argument('--json', default=os.path.join(
        _HERE, '_pipe_legmode_32.json'))
    a = ap.parse_args(argv)

    agg = [d for d in os.listdir(os.path.join(a.workdir, 'aggregate'))]
    tag = agg[0]
    plan = _payload(os.path.join(a.workdir, 'aggregate', tag, 'leg_crop.json'))
    ro = None
    for d in os.listdir(os.path.join(a.workdir, 'readout')):
        if d.startswith(tag) and d.endswith('__crop'):
            ro = _payload(os.path.join(a.workdir, 'readout', d,
                                       'metrics.json'))
    if ro is None:
        raise SystemExit("no crop readout metrics under %s" % a.workdir)

    if a.frames:
        frames = [k.strip() for k in a.frames.split(',')]
    else:
        worst = sorted(ro['frames'],
                       key=lambda k: -abs(float(ro['rows'][k]['power_ratio'])
                                          - 1.0))[:4]
        frames = worst + ([f for f in ('p0_p0',) if f not in worst])

    print('=' * 78)
    print('LEG-MODE DISCRIMINATOR -- 32-order fan, crop vs full')
    print('  workdir %s' % a.workdir)
    print('  tag     %s' % tag)
    print('=' * 78)
    t0 = time.perf_counter()
    fsum = load_carrier_field_zarr(os.path.join(a.workdir, 'aggregate', tag,
                                                'summed.zarr'))
    full = fsum.full_field()
    dx_c = fsum.grid.dx
    n_c = fsum.grid.shape[-1]
    R_c = fsum.carrier.R
    print('  summed field %d^2 at dx %.4f um, R_c %.6f mm  [%.1f s to load]'
          % (n_c, dx_c * 1e6, R_c * 1e3, time.perf_counter() - t0))
    print()
    print('  %-9s %-6s %12s %13s %11s %11s %8s'
          % ('frame', 'leg', 'window power', 'P/P_shipped-1', 'relL2',
             'EE3 %', 'leg/s'))
    rows = {}
    for k in frames:
        p = plan['frames'][k]
        ref = np.load(os.path.join(a.workdir, 'chains', f'{k}_ref.npy'))
        sref = spot(ref, DXO, (3e-6, 6e-6, 12e-6))
        rows[k] = {'shipped': sref}
        for mode in ('crop', 'full'):
            if mode == 'crop':
                kw = dict(N_fine=int(p['N_fine']),
                          window_factor=float(p['window_factor']),
                          centre=tuple(p['centre']))
            else:
                kw = dict(N_fine=int(n_c), window_factor=1e6,
                          centre=(0.0, 0.0))
            t1 = time.perf_counter()
            F = carrier_referenced_exact_focus_readout(
                full, R_c, TRAILING, LAM, dx_c, dx_out=DXO, N_out=TILE,
                tilt=(0.0, 0.0), centre_out=tuple(p['centre_out']),
                ram_budget=RAMB, on_ram_cap='error', on_replica='warn',
                on_readout_window='warn', on_n_fine_cap='error', **kw)
            dt = time.perf_counter() - t1
            s = spot(F, DXO, (3e-6, 6e-6, 12e-6))
            c = compare(ref, F)
            rows[k][mode] = {'spot': s, 'cmp': c,
                             'power_ratio': s['power'] / sref['power'],
                             't_leg': dt}
            print('  %-9s %-6s %12.7e %+13.3e %11.4e %11.4f %8.1f'
                  % (k, mode, s['power'], s['power'] / sref['power'] - 1.0,
                     c['rel_l2'], s['ee3'] * 100.0, dt))
            del F
        del ref

    print()
    wc = max(abs(rows[k]['crop']['power_ratio'] - 1.0) for k in frames)
    wf = max(abs(rows[k]['full']['power_ratio'] - 1.0) for k in frames)
    print('  worst |P/P_shipped - 1|:  crop %.3e   full %.3e   ratio %.1fx'
          % (wc, wf, wc / wf if wf else float('inf')))
    print('  campaign energy bar 4e-05: crop %s, full %s'
          % ('FAIL' if wc > 4e-5 else 'ok', 'FAIL' if wf > 4e-5 else 'ok'))
    ec = max(abs(rows[k]['crop']['spot']['ee3']
                 - rows[k]['shipped']['ee3']) * 100 for k in frames)
    ef = max(abs(rows[k]['full']['spot']['ee3']
                 - rows[k]['shipped']['ee3']) * 100 for k in frames)
    print('  worst |dEE3| (points):    crop %.4f   full %.4f   (bar 0.1)'
          % (ec, ef))
    with open(a.json, 'w', encoding='cp1252') as fh:
        json.dump({'tag': tag, 'frames': frames, 'rows': rows,
                   'worst_crop': wc, 'worst_full': wf}, fh, indent=1,
                  default=float)
    print('  wrote %s' % a.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

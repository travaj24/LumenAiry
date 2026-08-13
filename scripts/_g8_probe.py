"""G8 PROBE RE-ARCHITECTURE -- the measurement driver.

Three things, on niche C6's own fixture, on BOTH ``newton_fit`` backends:

  1. the probe trace is the LATTICE trace.  Hand ``probe_trace`` the launch
     lattice itself and it must reproduce ``x_out_grid`` / ``y_out_grid`` /
     ``opl_grid``, or the guard's ground truth is a different congruence from
     the one on trial;
  2. the probe POINTS are bit-identical on the two bases (they are chosen from
     the launch lattice and the census mask, both basis-free);
  3. the new G8 record, arm by arm, with the verdict.

Scratch driver for FIX_G8_PROBE_2026_08_12; not a test.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)
sys.path.insert(1, os.path.join(_ROOT, 'tests', 'unit'))

import warnings

import numpy as np
import test_niche_c6_stationary_phase_launch as C6  # noqa: E402

from lumenairy.elements import _lens_imap as IM  # noqa: E402


def capture(basis, **over):
    cap = {}
    orig = IM.build_inverse_map

    def wrapper(xs_in, XO, YO, OP, *a, **kw):
        cap['xs'] = np.asarray(xs_in).copy()
        cap['XO'] = np.asarray(XO).copy()
        cap['YO'] = np.asarray(YO).copy()
        cap['OP'] = np.asarray(OP).copy()
        cap['inv'] = kw.get('parity_invert')
        cap['probe'] = kw.get('probe_trace')
        cap['weights'] = kw.get('weights')
        out = orig(xs_in, XO, YO, OP, *a, **kw)
        cap['model'] = out
        return out

    IM.build_inverse_map = wrapper
    try:
        E, _X, _Y = C6._field()
        rec = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            C6._run(E, True, newton_fit=basis, _imap_out=rec, **over)
    finally:
        IM.build_inverse_map = orig
    cap['rec'] = rec
    return cap


def _census(cap):
    XO, YO, OP, W = cap['XO'], cap['YO'], cap['OP'], cap['weights']
    good = np.isfinite(XO) & np.isfinite(YO) & np.isfinite(OP)
    DJ = IM._detj_landing_stencil(cap['xs'], XO, YO)
    good &= np.isfinite(DJ)
    sig = good
    if W is not None:
        s = good & (W >= 0.5 * float(W[good].max()))
        if int(s.sum()) >= 16:
            sig = s
    return sig


def report(tag, cap):
    r = cap['rec']
    v = 'ENGAGE' if r.get('engaged') else 'REFUSE ' + str(r.get('refused'))
    print('  %-11s %-8s n_probe %s/%s  n_parity %s'
          % (tag, v, r.get('n_probe_traced'), r.get('n_probe_requested'),
             r.get('n_parity')))
    print('               a_pos %.4e  b_pos %.4e  ratio %.4f'
          % (r.get('parity_map_pos_m', float('nan')),
             r.get('parity_incumbent_pos_m', float('nan')),
             r.get('parity_ratio_pos', float('nan'))))
    print('               a_opl %.4e  b_opl %.4e  ratio %.4f'
          % (r.get('parity_map_opl_waves', float('nan')),
             r.get('parity_incumbent_opl_waves', float('nan')),
             r.get('parity_ratio_opl', float('nan'))))
    print('               a_rms %.4e  b_rms %.4e  ratio %.4f'
          % (r.get('parity_map_opl_rms_waves', float('nan')),
             r.get('parity_incumbent_opl_rms_waves', float('nan')),
             r.get('parity_ratio_opl_rms', float('nan'))))
    if r.get('refused'):
        print('               %s' % r.get('detail'))


if __name__ == '__main__':
    IM.TRACED_INVERSE_MAP = True
    IM._IMAP_CACHE_SIZE = 0
    over = {}
    for a in sys.argv[1:]:
        k, _, v = a.partition('=')
        over[k] = None if v == 'None' else float(v) if '.' in v else int(v)
    caps = {}
    for basis in ('polynomial', 'spline'):
        caps[basis] = capture(basis, **over)

    print('1. THE PROBE TRACE IS THE LATTICE TRACE')
    for basis, c in caps.items():
        xs = c['xs']
        X, Y = np.meshgrid(xs, xs, indexing='ij')
        px, py = X.ravel(), Y.ravel()
        qx, qy, qo = c['probe'](px, py)
        m = np.isfinite(c['XO'].ravel())
        d = max(float(np.nanmax(np.abs(qx[m] - c['XO'].ravel()[m]))),
                float(np.nanmax(np.abs(qy[m] - c['YO'].ravel()[m]))))
        do = float(np.nanmax(np.abs(qo[m] - c['OP'].ravel()[m])))
        print('  %-11s %d nodes   pos %.3e m   OPL %.3e m   exact=%s'
              % (basis, int(m.sum()), d, do,
                 bool(np.array_equal(qx[m], c['XO'].ravel()[m])
                      and np.array_equal(qo[m], c['OP'].ravel()[m]))))

    print('2. THE PROBE POINTS ARE THE SAME ON BOTH BASES')
    pts = {}
    for basis, c in caps.items():
        pts[basis] = IM._probe_entrance_points(c['xs'], _census(c))
    same = (np.array_equal(pts['polynomial'][0], pts['spline'][0])
            and np.array_equal(pts['polynomial'][1], pts['spline'][1]))
    print('  %d probe points, bit-identical: %s'
          % (pts['polynomial'][0].size, same))

    print('3. G8, ARM BY ARM')
    for basis, c in caps.items():
        report(basis, c)

"""ARM B AT ITS OWN KNOTS vs BETWEEN THEM -- with the C6 launch ENGAGED.

Truth comes from the element itself at HALF the ray subsample: the launch
lattices nest exactly (``linspace(-R, R, 2n-1)[2k] == linspace(-R, R, n)[k]``),
so the finer trace supplies the element's OWN landings at the midpoints of the
coarse lattice.  The nesting is VERIFIED here (shared nodes must agree to the
trace's own floor) before any conclusion is drawn from it -- if the two runs
disagree at shared nodes the measurement is void and says so.

Then the coarse call's incumbent inversion (G8's arm B, captured verbatim) is
run at BOTH sets of landings:

  * at the coarse NODES        -- the points G8 actually probes;
  * at the MIDPOINTS           -- points arm B has not interpolated.
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


def capture(basis, rs):
    """Run the element and capture what it hands the model, plus arm B."""
    cap = {}
    orig = IM.build_inverse_map

    def wrapper(xs_in, XO, YO, OP, *a, **kw):
        cap['xs'] = np.asarray(xs_in).copy()
        cap['XO'] = np.asarray(XO).copy()
        cap['YO'] = np.asarray(YO).copy()
        cap['inv'] = kw.get('parity_invert')
        return orig(xs_in, XO, YO, OP, *a, **kw)

    IM.build_inverse_map = wrapper
    try:
        E, _X, _Y = C6._field()
        rec = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            C6._run(E, True, newton_fit=basis, ray_subsample=rs,
                    _imap_out=rec)
    finally:
        IM.build_inverse_map = orig
    cap['rec'] = rec
    return cap


def err(inv, xq, yq, xt, yt):
    m = np.isfinite(xq) & np.isfinite(yq)
    rx, ry, _o = inv(np.asarray(xq)[m], np.asarray(yq)[m])
    return float(np.nanmax(np.maximum(np.abs(np.ravel(rx) - np.asarray(xt)[m]),
                                      np.abs(np.ravel(ry) - np.asarray(yt)[m]))))


if __name__ == '__main__':
    IM.TRACED_INVERSE_MAP = True
    IM._IMAP_CACHE_SIZE = 0
    fine = capture('polynomial', 2)
    xs2, XO2, YO2 = fine['xs'], fine['XO'], fine['YO']
    print('%-11s %-13s %-13s %-13s %-13s %-7s'
          % ('basis', 'nest resid', 'AT NODES', 'AT MIDPTS', 'G8 b_pos',
             'mid/node'))
    for basis in ('polynomial', 'spline'):
        c = capture(basis, 4)
        xs4, XO4, YO4, inv = c['xs'], c['XO'], c['YO'], c['inv']
        assert xs2.size == 2 * xs4.size - 1, (xs2.size, xs4.size)
        nest = float(np.nanmax(np.abs(xs2[::2] - xs4)))
        # the element's own landings at shared nodes, both runs
        sh = np.isfinite(XO2[::2, ::2]) & np.isfinite(XO4)
        dnest = max(float(np.nanmax(np.abs((XO2[::2, ::2] - XO4)[sh]))),
                    float(np.nanmax(np.abs((YO2[::2, ::2] - YO4)[sh]))))
        X4, Y4 = np.meshgrid(xs4, xs4, indexing='ij')
        e_node = err(inv, XO4[sh], YO4[sh], X4[sh], Y4[sh])
        # MIDPOINTS: odd indices of the fine lattice in BOTH axes
        xm = xs2[1::2]
        XM, YM = np.meshgrid(xm, xm, indexing='ij')
        XOm, YOm = XO2[1::2, 1::2], YO2[1::2, 1::2]
        sm = np.isfinite(XOm) & np.isfinite(YOm)
        # score on the same disc the model is fitted on
        r2 = XM ** 2 + YM ** 2
        rmax = float(np.nanmax((X4 ** 2 + Y4 ** 2)[sh]))
        sm = sm & (r2 <= rmax)
        e_mid = err(inv, XOm[sm], YOm[sm], XM[sm], YM[sm])
        print('%-11s %-13.3e %-13.4e %-13.4e %-13.4e %-7.1f'
              % (basis, dnest, e_node, e_mid,
                 c['rec'].get('parity_incumbent_pos_m', float('nan')),
                 e_mid / max(e_node, 1e-300)))
        print('              lattice nesting %.1e m; %d node / %d midpoint '
              'probes' % (nest, int(sh.sum()), int(sm.sum())))

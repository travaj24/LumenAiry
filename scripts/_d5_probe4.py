"""IS G8's ARM B A MEASUREMENT, OR AN INTERPOLATION IDENTITY?

G8 probes both arms at HELD-OUT LAUNCH NODES.  ``RectBivariateSpline`` with
``s=0`` interpolates: it reproduces every launch node exactly, held out or
not, so at those points arm B's error is the Newton loop's leftover residual
rather than the incumbent's production accuracy.  The polynomial basis is a
global least-squares fit and has real error at its own nodes.

This measures arm B's error at points it has NOT interpolated -- the
MIDPOINTS of the launch lattice, with exact-ray truth from the same tracer the
element uses -- and compares that with what G8 records at the nodes.

Run with the C6 stationary-phase launch OFF so the element launches along
``grad(W)``, which the oracle reproduces in closed form; then the entrance
coordinates recovered by the incumbent can be compared with the exact ones
without importing the residual-fit error.
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

_CAP: dict = {}


def _capture(orig):
    def wrapper(xs_in, XO, YO, OP, *a, **kw):
        _CAP['xs_in'] = np.asarray(xs_in).copy()
        _CAP['XO'] = np.asarray(XO).copy()
        _CAP['YO'] = np.asarray(YO).copy()
        _CAP['inv'] = kw.get('parity_invert')
        return orig(xs_in, XO, YO, OP, *a, **kw)
    return wrapper


def run(basis):
    _CAP.clear()
    orig = IM.build_inverse_map
    IM.build_inverse_map = _capture(orig)
    try:
        E, _X, _Y = C6._field()
        out = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            C6._run(E, False, newton_fit=basis, _imap_out=out)
    finally:
        IM.build_inverse_map = orig
    return out


if __name__ == '__main__':
    IM.TRACED_INVERSE_MAP = True
    IM._IMAP_CACHE_SIZE = 0
    surfs = C6._surfaces(C6._free_leg())
    print('%-11s %-14s %-14s %-14s %-8s'
          % ('basis', 'node err (m)', 'midpt err (m)', 'G8 b_pos (m)',
             'ratio'))
    for basis in ('polynomial', 'spline'):
        rec = run(basis)
        xs = _CAP['xs_in']
        inv = _CAP['inv']
        # -- sanity: the oracle reproduces the element's own landings --------
        i = np.arange(3, xs.size - 3, 7)
        XN, YN = np.meshgrid(xs[i], xs[i], indexing='ij')
        xo_n, yo_n = C6._trace_total(XN, YN, surfs, 0.0)[:2]
        dl = max(float(np.nanmax(np.abs(xo_n - _CAP['XO'][np.ix_(i, i)].ravel()))),
                 float(np.nanmax(np.abs(yo_n - _CAP['YO'][np.ix_(i, i)].ravel()))))
        # -- arm B at NODES (what G8 probes) ---------------------------------
        rx, ry, _o = inv(xo_n, yo_n)
        e_node = float(np.nanmax(np.maximum(np.abs(np.ravel(rx) - XN.ravel()),
                                            np.abs(np.ravel(ry) - YN.ravel()))))
        # -- arm B at MIDPOINTS (points it has not interpolated) -------------
        xm = 0.5 * (xs[i] + xs[i + 1])
        XM, YM = np.meshgrid(xm, xm, indexing='ij')
        xo_m, yo_m = C6._trace_total(XM, YM, surfs, 0.0)[:2]
        rx, ry, _o = inv(xo_m, yo_m)
        e_mid = float(np.nanmax(np.maximum(np.abs(np.ravel(rx) - XM.ravel()),
                                           np.abs(np.ravel(ry) - YM.ravel()))))
        print('%-11s %-14.4e %-14.4e %-14.4e %-8.1f'
              % (basis, e_node, e_mid, rec.get('parity_incumbent_pos_m',
                                               float('nan')),
                 e_mid / max(e_node, 1e-300)))
        print('            (oracle-vs-element landing agreement %.3e m)' % dl)

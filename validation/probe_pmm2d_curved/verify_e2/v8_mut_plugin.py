"""pytest plugin: apply ONE curved-mortar mutant (env ``V8_MUT``) at import,
so a test file can be run against it (Phase E2 verifier, item 8):

  cd /c/tmp/lum_vcurved_e2 && V8_MUT=h_no_measure \
    PYTHONPATH="C:/tmp/lum_vcurved_e2;C:/tmp/lum_vcurved_e2/validation/probe_pmm2d_curved/verify_e2" \
    python -m pytest tests/unit/test_pmm2d_staggered_curved_e2.py -p v8_mut_plugin ...

Mutants: h_no_measure, h_no_offdiag, e_no_factor, sqrt_off, tang_missed,
maps_ignored, no_ride, ride_below, no_qmatch.
"""
import os

import numpy as np

from lumenairy.elements.pmm import _curvemortar as CMOR, stack2d_pure as SP, twod_staggered as TS

MUT = os.environ.get("V8_MUT", "")
_orig_init = CMOR.StagCrossOpsMapped.__init__
_orig_add = CMOR._add_pair
_orig_outer = CMOR._outer_rule


def _scaled_x(ga, gb, n, fn):
    CMOR._add_pair = fn
    try:
        return CMOR.curved_cross_mass(ga, gb, n)
    finally:
        CMOR._add_pair = _orig_add


def _h_no_measure(self, ga, gb, tol=None):
    _orig_init(self, ga, gb, tol)

    def add(X, ga_, gb_, A, B, ac, bc, ua, va, ub_, vb_, w, side, sa, sb):
        g1 = A.geom(ac[0], ac[1], ua, va)
        g2 = B.geom(bc[0], bc[1], ub_, vb_)
        with np.errstate(all="ignore"):
            s = np.nan_to_num((g1[2] * g1[5] - g1[3] * g1[4])
                              / (g2[2] * g2[5] - g2[3] * g2[4]))
        _orig_add(X, ga_, gb_, A, B, ac, bc, ua, va, ub_, vb_, w * s, side,
                  sa, sb)
    self.H = CMOR.cross_h_from_x(_scaled_x(ga, gb, self.n, add), ga.qq,
                                 gb.qq)


def _h_no_offdiag(self, ga, gb, tol=None):
    _orig_init(self, ga, gb, tol)
    qa, qb = ga.qq, gb.qq
    H = np.zeros_like(self.H)
    H[:qa, :qb] = self.EH[qb:, qa:].conj().T
    H[qa:, qb:] = self.EH[:qb, :qa].conj().T
    self.H = H


def _e_no_factor(self, ga, gb, tol=None):
    _orig_init(self, ga, gb, tol)

    def add(X, ga_, gb_, A, B, ac, bc, ua, va, ub_, vb_, w, side, sa, sb):
        g2 = B.geom(bc[0], bc[1], ub_, vb_)

        class Fake:
            def geom(s, sx, sy, U, V):
                g = A.geom(sx, sy, U, V)
                return (g[0], g[1]) + tuple(g2[2:])
        if side == "a":
            g1 = A.geom(ac[0], ac[1], ua, va)
            with np.errstate(all="ignore"):
                w = np.nan_to_num(w * (g1[2] * g1[5] - g1[3] * g1[4])
                                  / (g2[2] * g2[5] - g2[3] * g2[4]))
        _orig_add(X, ga_, gb_, Fake(), B, ac, bc, ua, va, ub_, vb_, w, side,
                  sa, sb)
    self.EH = _scaled_x(ga, gb, self.n, add)
    self.H = CMOR.cross_h_from_x(self.EH, ga.qq, gb.qq)


class _MapsIgnored:
    dense = False

    def __init__(self, ga, gb, tol=None):
        o = TS.StagCrossOps(ga, gb)
        self.C1, self.C2, self.C1H, self.C2H = o.C1, o.C2, o.C1H, o.C2H


if MUT == "h_no_measure":
    CMOR.StagCrossOpsMapped.__init__ = _h_no_measure
elif MUT == "h_no_offdiag":
    CMOR.StagCrossOpsMapped.__init__ = _h_no_offdiag
elif MUT == "e_no_factor":
    CMOR.StagCrossOpsMapped.__init__ = _e_no_factor
elif MUT == "sqrt_off":
    CMOR._outer_rule = lambda o0, o1, a, b, n: _orig_outer(o0, o1, False,
                                                           False, n)
elif MUT == "tang_missed":
    CMOR._tangencies = lambda *a, **k: []
elif MUT == "maps_ignored":
    CMOR.StagCrossOpsMapped = _MapsIgnored
elif MUT == "no_ride":
    SP.PMM2DStackPure._e2_no_ride = True
elif MUT == "ride_below":
    _orig_geo = SP.PMM2DStackPure._perlayer_geometry

    def _geo_below(self, Ms):
        geo = _orig_geo(self, Ms)
        Ls = self._layers

        def own(i):
            return (geo[i][0] is Ls[i]["wx"]
                    and geo[i][3] is Ls[i].get("cmap"))
        out = list(geo)
        n = len(Ls)
        for i in range(n):
            if not own(i):
                # a rider: take the nearest non-rider BELOW instead
                for k in range(i + 1, n):
                    if own(k):
                        out[i] = geo[k]
                        break
        return out
    SP.PMM2DStackPure._perlayer_geometry = _geo_below
elif MUT == "no_qmatch":
    _orig_counts = SP.PMM2DStackPure._perlayer_modal_counts

    def _counts(self, *a, **k):
        saved = []
        for L in self._layers:
            saved.append(L.get("pl_keywords"))
            if L.get("own") is not None:
                L["pl_keywords"] = True
        try:
            return _orig_counts(self, *a, **k)
        finally:
            for L, s in zip(self._layers, saved):
                L["pl_keywords"] = s
    SP.PMM2DStackPure._perlayer_modal_counts = _counts
elif MUT:
    raise RuntimeError(f"unknown V8_MUT {MUT!r}")

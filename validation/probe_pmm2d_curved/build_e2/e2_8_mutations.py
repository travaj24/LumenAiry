"""E2-8: the mutation matrix of the curved mortar, each arm measured on the
E2-4 device (circle over a crossing sinusoidal wall, lossless) at rung M
(arg 1) against the shipped build: the change of R / T, the closure.

  inv_tol_1e3   -- the map-inversion Newton tolerance loosened 1e3x
  one_map       -- the cross-mass computed on ONE layer's map (both bases
                   evaluated through the upper layer's map: T = I)
  nocut         -- the plain tensor rule per cell, ignoring the other
                   layer's walls (the plan's "composite map" route), n = 64
  hswap_off     -- the V1/V2 swap of the H-row masses off (the shipped G3
                   fail-before)
  fast_forced   -- the merge's CROSSING check disabled, so the stack-wide
                   merge (the fast path) is attempted on the crossing pair
The -R flux-Gram arm is in e2_4_overlap.py (mode absorb)."""
import sys
import time

import numpy as np
from _common import D1, D2, EPS_P, EPS_W, R_CIRC, dump, shapes_stack, solve

from lumenairy.elements.pmm import _core, _curvemortar as CMM, shapes2d as S2
from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall

M = int(sys.argv[1])
lay = [(D1, [Circle(0.6, 0.6, R_CIRC, EPS_P)], 1.0),
       (D2, [SinusoidalWall("x", 0.6, 0.12, eps=EPS_W)], 1.0)]


def run():
    st = shapes_stack(lay, M)
    o, R, T, J = solve(st)
    return R, T, st


R0, T0, _ = run()
out = {"M": M, "closure_shipped": np.abs(R0.sum(1) + T0.sum(1) - 1.0)}


def arm(name, patch, unpatch):
    t0 = time.perf_counter()
    patch()
    try:
        R, T, st = run()
        res = dict(d_RT=float(max(np.abs(R - R0).max(), np.abs(T - T0).max())),
                   closure=np.abs(R.sum(1) + T.sum(1) - 1.0))
    except Exception as ex:  # a mutation may be refused outright
        res = dict(raised=f"{type(ex).__name__}: {str(ex)[:200]}")
    finally:
        unpatch()
    res["wall"] = time.perf_counter() - t0
    out[name] = res
    print(name, res)


orig_tol = CMM._CURVE_MORTAR_INV_TOL
arm("inv_tol_1e3",
    lambda: setattr(CMM, "_CURVE_MORTAR_INV_TOL", orig_tol * 1e3),
    lambda: setattr(CMM, "_CURVE_MORTAR_INV_TOL", orig_tol))

orig_ccm = CMM.curved_cross_mass
orig_ops = CMM.StagCrossOpsMapped


def maps_ignored(ga, gb, tol=None):
    """the cross-mass with BOTH maps ignored: the shipped separable
    StagCrossOps on the two (u, v) wall grids, as if u_a = u_b = x"""
    from lumenairy.elements.pmm import twod_staggered as TSm
    return TSm.StagCrossOps(ga, gb)


arm("maps_ignored", lambda: setattr(CMM, "StagCrossOpsMapped", maps_ignored),
    lambda: setattr(CMM, "StagCrossOpsMapped", orig_ops))


orig_ad = CMM.curved_cross_mass_adaptive


def nocut(ga, gb, tol=None, cap=None):
    return orig_ccm(ga, gb, 64, cut=False), 64, 0.0


arm("nocut", lambda: setattr(CMM, "curved_cross_mass_adaptive", nocut),
    lambda: setattr(CMM, "curved_cross_mass_adaptive", orig_ad))
arm("hswap_off", lambda: setattr(_core, "PMM2D_MORTAR_H_SWAP", False),
    lambda: setattr(_core, "PMM2D_MORTAR_H_SWAP", True))
orig_cross = S2._crossing
arm("fast_forced", lambda: setattr(S2, "_crossing", lambda *a, **k: False),
    lambda: setattr(S2, "_crossing", orig_cross))
S2._crossing = lambda *a, **k: False
st_ff = shapes_stack(lay, M)
S2._crossing = orig_cross
out["fast_forced"]["merge_refusal"] = st_ff._merge_refusal
out["fast_forced"]["fast_ok"] = st_ff._perlayer_fast_ok()
print("fast_forced refusal:", st_ff._merge_refusal)
# maps_ignored on the NON-overlapping pair, where the merged map is the
# reference the E2-4 "both ways" gate reads
lay_n = [(D1, [Circle(0.6, 0.6, R_CIRC, EPS_P)], 1.0),
         (D2, [SinusoidalWall("x", 0.12, 0.05, eps=EPS_W)], 1.0)]
stm = shapes_stack(lay_n, M)
_o, Rm, Tm, _J = solve(stm)
res = {}
for nm, ops in (("shipped", orig_ops), ("maps_ignored", maps_ignored)):
    CMM.StagCrossOpsMapped = ops
    try:
        stp = shapes_stack(lay_n, M)
        stp._e2_per_layer_maps = True
        _o, Rp, Tp, _J = solve(stp)
    finally:
        CMM.StagCrossOpsMapped = orig_ops
    res[nm] = float(max(np.abs(Rp - Rm).max(), np.abs(Tp - Tm).max()))
out["nonoverlap_perlayer_vs_merged"] = res
print("non-overlapping per-layer vs merged", res)
dump(f"e2_8_mutations_M{M}.json", out)

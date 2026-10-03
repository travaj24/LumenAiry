"""V8 -- the verifier's mutation matrix of the curved mortar (Phase E2,
item 8), by in-process monkeypatching only.

Device (arg 1): 'builder' = the build's E2-4 pair (eps-4 disk r 0.36 at the
cell centre, t 0.3, over the wall x = 0.6 + 0.12 sin(2 pi y / 1.2), eps
2.25, t 0.25; P 1.2, lambda 1, n_sub 1.45); 'own' = the verifier's crossing
pair (disk r 0.33 at (0.55, 0.62), eps 3.2, t 0.27, over the wall
x = 0.66 + 0.1 sin(2 pi y / 1.2 + 0.7), eps 1.9, t 0.31; lambda 0.93).
Rung M (arg 2).

Arms (R / T moved against the shipped answer, closure, the main interface's
cross operators against the shipped ones):
  h_no_measure  -- the H row without the measure change det(T) (the H-type
                   quantity pulled back with T^-T but integrated in the
                   WRONG layer's du dv): CrossH = cross_h_from_x(X / det T)
  h_no_offdiag  -- the H row's off-diagonal cofactor blocks dropped:
                   CrossH = blkdiag(X22^H, X11^H)
  e_no_factor   -- the E row's pull-back factor dropped (T -> I, measure
                   kept): X with the weight I in b's du dv
  sqrt_off      -- the square-root substitution at tangencies disabled
                   (_outer_rule ignores the tangency flags)
  tang_missed   -- tangencies not searched (_tangencies -> [])
  maps_ignored  -- the separable cross-mass on the two (u, v) grids
  newton_tol    -- the inversion step tolerance x 1e3
  hswap_off     -- _core.PMM2D_MORTAR_H_SWAP = False
  fast_forced   -- the merge's crossing test disabled (the fast path is
                   attempted on the crossing pair)
"""
import sys
import time
import warnings

import numpy as np
from _ve import closure, dump, rt_diff, solve

from lumenairy.elements.pmm import (
    PMM2DStackPure,
    _core,
    _curvemortar as CMOR,
    shapes2d as S2,
    twod_staggered as TS,
)
from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall

dev = sys.argv[1]
M = int(sys.argv[2])
P = 1.2
if dev == "builder":
    WL = 1.0
    L1 = (0.3, [Circle(0.6, 0.6, 0.36, 4.0)])
    L2 = (0.25, [SinusoidalWall("x", 0.6, 0.12, eps=2.25)])
else:
    WL = 0.93
    L1 = (0.27, [Circle(0.55, 0.62, 0.33, 3.2)])
    L2 = (0.31, [SinusoidalWall("x", 0.66, 0.1, phase=0.7, eps=1.9)])


def stack():
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=3, layer_grids="per-layer")
    st.add_layer(L1[0], shapes=L1[1], background_eps=1.0)
    st.add_layer(L2[0], shapes=L2[1], background_eps=1.0)
    return st


REC = []
orig_init = CMOR.StagCrossOpsMapped.__init__
orig_add = CMOR._add_pair


def rec_init(self, ga, gb, tol=None):
    orig_init(self, ga, gb, tol)
    REC.append((self.n, self.change, self.EH.copy(), self.H.copy()))


def run(name):
    REC.clear()
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            st = stack()
            res = solve(st, wl=WL)
            err = None
        except Exception as ex:          # noqa: BLE001 -- recorded
            res, err, st = None, f"{type(ex).__name__}: {ex}", None
    return dict(res=res, err=err, wall=time.perf_counter() - t0,
                warn=[str(w.message)[:200] for w in wl], ops=list(REC),
                st=st)


CMOR.StagCrossOpsMapped.__init__ = rec_init
out = {"device": dev, "M": M}
base = run("shipped")
assert base["err"] is None, base["err"]
assert not base["st"]._perlayer_fast_ok()
out["shipped"] = dict(closure=closure(base["res"]), wall=base["wall"],
                      n=[o[0] for o in base["ops"]],
                      change=[o[1] for o in base["ops"]])
print("shipped closure", out["shipped"]["closure"], "n",
      out["shipped"]["n"])
X0, H0 = base["ops"][0][2], base["ops"][0][3]


def report(name, r):
    d = dict(wall=r["wall"], warnings=r["warn"], error=r["err"])
    if r["res"] is not None:
        d.update(rt_moved=rt_diff(r["res"], base["res"]),
                 closure=closure(r["res"]))
        if r["ops"]:
            Xm, Hm = r["ops"][0][2], r["ops"][0][3]
            if Xm.shape == X0.shape:
                d.update(dX=float(np.max(np.abs(Xm - X0))
                                  / np.max(np.abs(X0))),
                         dH=float(np.max(np.abs(Hm - H0))
                                  / np.max(np.abs(H0))),
                         n=[o[0] for o in r["ops"]],
                         change=[o[1] for o in r["ops"]])
    out[name] = d
    print(name, {k: v for k, v in d.items() if k != "warnings"},
          len(r["warn"]), "warnings")


# ---- H row without the measure change ---------------------------------
def init_h_no_measure(self, ga, gb, tol=None):
    rec_init(self, ga, gb, tol)

    def add_scaled(X, ga_, gb_, A, B, ac, bc, ua, va, ub_, vb_, w, side,
                   sa, sb):
        ga1 = A.geom(ac[0], ac[1], ua, va)
        gb1 = B.geom(bc[0], bc[1], ub_, vb_)
        da = ga1[2] * ga1[5] - ga1[3] * ga1[4]
        db = gb1[2] * gb1[5] - gb1[3] * gb1[4]
        with np.errstate(all="ignore"):
            s = np.nan_to_num(da / db)          # 1 / det T
        orig_add(X, ga_, gb_, A, B, ac, bc, ua, va, ub_, vb_, w * s, side,
                 sa, sb)
    CMOR._add_pair = add_scaled
    try:
        Xs = CMOR.curved_cross_mass(ga, gb, self.n)
    finally:
        CMOR._add_pair = orig_add
    self.H = CMOR.cross_h_from_x(Xs, ga.qq, gb.qq)
    REC[-1] = (self.n, self.change, self.EH.copy(), self.H.copy())


CMOR.StagCrossOpsMapped.__init__ = init_h_no_measure
report("h_no_measure", run("h_no_measure"))


def init_h_no_offdiag(self, ga, gb, tol=None):
    rec_init(self, ga, gb, tol)
    qa, qb = ga.qq, gb.qq
    X = self.EH
    H = np.zeros_like(self.H)
    H[:qa, :qb] = X[qb:, qa:].conj().T
    H[qa:, qb:] = X[:qb, :qa].conj().T
    self.H = H
    REC[-1] = (self.n, self.change, self.EH.copy(), self.H.copy())


CMOR.StagCrossOpsMapped.__init__ = init_h_no_offdiag
report("h_no_offdiag", run("h_no_offdiag"))


def init_e_no_factor(self, ga, gb, tol=None):
    rec_init(self, ga, gb, tol)

    def add_noT(X, ga_, gb_, A, B, ac, bc, ua, va, ub_, vb_, w, side, sa,
                sb):
        # replace the geometric weight by the identity in b's du dv: do it
        # by making A's Jacobian equal B's (T = I) for this call
        gb1 = B.geom(bc[0], bc[1], ub_, vb_)

        class _Fake:
            def __init__(s, real):
                s.real = real

            def geom(s, sx, sy, U, V):
                g = s.real.geom(sx, sy, U, V)
                return (g[0], g[1]) + tuple(gb1[2:])
        if side == "a":
            ga1 = A.geom(ac[0], ac[1], ua, va)
            da = ga1[2] * ga1[5] - ga1[3] * ga1[4]
            db = gb1[2] * gb1[5] - gb1[3] * gb1[4]
            with np.errstate(all="ignore"):
                w = np.nan_to_num(w * da / db)  # du_a dv_a -> du_b dv_b
        orig_add(X, ga_, gb_, _Fake(A), B, ac, bc, ua, va, ub_, vb_, w,
                 side, sa, sb)
    CMOR._add_pair = add_noT
    try:
        self.EH = CMOR.curved_cross_mass(ga, gb, self.n)
    finally:
        CMOR._add_pair = orig_add
    self.H = CMOR.cross_h_from_x(self.EH, ga.qq, gb.qq)
    REC[-1] = (self.n, self.change, self.EH.copy(), self.H.copy())


CMOR.StagCrossOpsMapped.__init__ = init_e_no_factor
report("e_no_factor", run("e_no_factor"))
CMOR.StagCrossOpsMapped.__init__ = rec_init

# ---- the square-root substitution off / tangencies missed -------------
orig_outer = CMOR._outer_rule
NSQ = [0]


def outer_count(o0, o1, sq0, sq1, n):
    NSQ[0] += int(bool(sq0)) + int(bool(sq1))
    return orig_outer(o0, o1, sq0, sq1, n)


CMOR._outer_rule = outer_count
_ = run("count")
out["tangency_ends_in_shipped_run"] = NSQ[0]
print("tangency-flagged interval ends in the shipped run:", NSQ[0])
CMOR._outer_rule = lambda o0, o1, sq0, sq1, n: orig_outer(o0, o1, False,
                                                          False, n)
report("sqrt_off", run("sqrt_off"))
CMOR._outer_rule = orig_outer
orig_tang = CMOR._tangencies
CMOR._tangencies = lambda *a, **k: []
report("tang_missed", run("tang_missed"))
CMOR._tangencies = orig_tang


# ---- the rest of the builder's matrix, re-measured --------------------
class MapsIgnored:
    dense = False

    def __init__(self, ga, gb, tol=None):
        o = TS.StagCrossOps(ga, gb)
        self.C1, self.C2 = o.C1, o.C2
        self.C1H, self.C2H = o.C1H, o.C2H


orig_cls = CMOR.StagCrossOpsMapped
CMOR.StagCrossOpsMapped = MapsIgnored
report("maps_ignored", run("maps_ignored"))
CMOR.StagCrossOpsMapped = orig_cls
tol0 = CMOR._CURVE_MORTAR_INV_TOL
CMOR._CURVE_MORTAR_INV_TOL = tol0 * 1e3
report("newton_tol", run("newton_tol"))
CMOR._CURVE_MORTAR_INV_TOL = tol0
_core.PMM2D_MORTAR_H_SWAP = False
report("hswap_off", run("hswap_off"))
_core.PMM2D_MORTAR_H_SWAP = True
orig_cross = S2._crossing
S2._crossing = lambda *a, **k: False
r = run("fast_forced")
S2._crossing = orig_cross
report("fast_forced", r)
if r["st"] is not None:
    out["fast_forced"].update(
        merge_refusal=r["st"]._merge_refusal,
        fast_ok=bool(r["st"]._perlayer_fast_ok()))
    print("fast_forced refusal:", r["st"]._merge_refusal)
CMOR.StagCrossOpsMapped.__init__ = orig_init
for k in list(out):
    if isinstance(out[k], dict):
        out[k].pop("st", None)
dump(f"v8_mutations_{dev}_M{M}", out)

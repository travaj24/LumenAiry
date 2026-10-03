"""E2 verifier item 9 -- the Phase C verifier fold-in, re-measured.

V-D1 (``_VERTEX_SNAP = 1e-13`` in ``shapes2d._merge``): rectangles-only
layouts must merge to the IDENTITY map.
  (a) >= 200 random two-layer rectangle layouts per period (3 periods incl.
      SI units), edges on a coarse lattice but computed through ``cx -+ w/2``
      (so coinciding walls differ by round-off): fraction that miss the
      identity (``_merge(...)[4]`` False) and fraction refused.
  (b) the snap BOUNDARY: a layer-2 rectangle whose left wall is the layer-1
      wall + d * P, d in 0 .. 1e-3 -- identity / mapped / refused per d,
      plus (when mapped) the solve-level consequence: a tensor rectangle
      refused? R / T of the near-identity mapped solve vs the exact
      (d = 0) unmapped solve at oblique incidence.
  (c) snap safety: a SinusoidalWall of tiny amplitude (its vertices within
      the snap of the grid) -- does the snap break the map (edge curve
      endpoint != vertex)?
  (d) the proposed tolerance fix (``_VERTEX_SNAP = _WALL_SNAP``) applied
      in-process: (b) again.
V-D2 (rotated Ellipse corners at the outward-normal 45-degree points):
  (e) aspect 1.01 .. 5 x angle -44.9 .. 44.9 deg: compile (FOLD?), the
      independent material oracle (_wrong_map of the verify-C test file),
      min det J on a 24 x 24 interior sample of every cell; 45 deg and
      beyond: what happens.
  (f) solve-level: the +30 deg ellipse vs its mirror image (-30 deg) at
      normal incidence (orders (m, n) -> (m, -n), Jones -> D J D with
      D = diag(1, -1)).
Args: [skip_solve]  Output v9_vd1_vd2_<tree>_<build>.json.
"""
import importlib.util
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ve import HERE, TREE, dump, solve  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    Ellipse,
    PMM2DStackPure,
    Rect,
    SinusoidalWall,
    compile_shapes,
    shapes2d as SH,  # noqa: E402
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

SKIP_SOLVE = len(sys.argv) > 1 and sys.argv[1] == "skip_solve"
warnings.simplefilter("ignore")
OUT = {"tree": TREE, "vertex_snap": getattr(SH, "_VERTEX_SNAP", None),
       "wall_snap": SH._WALL_SNAP, "claim_tol": SH._CLAIM_TOL}

# the verify-C test file's independent oracle (POST tree path; it only uses
# the public map API, which the PRE tree has too)
_tp = os.path.join(HERE, "..", "..", "..", "tests", "unit",
                   "test_verify_pmm2d_curved_c.py")
_spec = importlib.util.spec_from_file_location("_v9_vc", _tp)
VC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(VC)


# ---- (a) random rectangles-only layouts -----------------------------------
def rand_layout(rng, P):
    lat = np.arange(1, 20) / 20.0                  # walls on k / 20, 1..19

    def rect(xlo, xhi, ylo, yhi, e):
        cx, w = 0.5 * (xlo + xhi) * P, (xhi - xlo) * P
        cy, h = 0.5 * (ylo + yhi) * P, (yhi - ylo) * P
        return Rect(cx, cy, w, h, e)

    def pair():
        a, b = sorted(rng.choice(lat, 2, replace=False))
        return float(a), float(b)

    l1 = []
    # layer 1: one or two rectangles side by side (non-overlapping in x)
    xs = sorted(rng.choice(lat, 4, replace=False))
    l1.append(rect(xs[0], xs[1], *pair(), 4.0))
    if rng.random() < 0.6:
        l1.append(rect(xs[2], xs[3], *pair(), 2.25))
    l2 = [rect(*pair(), *pair(), 3.0)]
    return [("layer 1", l1, 1.0), ("layer 2", l2, 1.5)]


rnd = {}
for P in (1.2, 1.0, 1.2e-6):
    rng = np.random.default_rng(20261003)
    n_id = n_map = n_ref = 0
    for _ in range(220):
        lay = rand_layout(rng, P)
        try:
            ident = bool(SH._merge(P, P, lay)[4])
        except ValueError:
            n_ref += 1
            continue
        n_id += ident
        n_map += not ident
    rnd[repr(P)] = dict(n=220, identity=n_id, missed_identity=n_map,
                        refused=n_ref,
                        miss_fraction=n_map / max(1, n_id + n_map))
OUT["random_rects"] = rnd
print("random", rnd)


# ---- (b) the snap boundary --------------------------------------------------
P = 1.2
DS = (0.0, 5e-14, 0.9e-13, 1.1e-13, 2e-13, 5e-13, 0.9e-12, 1.1e-12, 1e-9,
      1e-6, 0.99e-3, 1.01e-3, 2e-3)


def bnd_layers(d, eps2=3.0):
    x1 = 0.3 * P
    r1 = Rect(x1 + 0.15 * P, 0.5 * P, 0.3 * P, 0.4 * P, 4.0)   # x 0.3 .. 0.6
    x2 = x1 + d * P
    r2 = Rect(0.5 * (x2 + 0.8 * P), 0.45 * P, 0.8 * P - x2, 0.5 * P, eps2)
    return [("layer 1", [r1], 1.0), ("layer 2", [r2], 1.0)]


def boundary_scan():
    res = {}
    for d in DS:
        try:
            U, V, cmap, cells, ident = SH._merge(P, P, bnd_layers(d))[:5]
            res[repr(d)] = dict(outcome="identity" if ident else "MAPPED",
                                nx=int(U.size - 1))
        except ValueError as e:
            res[repr(d)] = dict(outcome="refused", msg=str(e)[:120])
    return res


OUT["boundary"] = boundary_scan()
print("boundary", {k: v["outcome"] for k, v in OUT["boundary"].items()})


def stack_for(d, eps2=3.0, M=4):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2)
    for lab, shapes, bg in bnd_layers(d, eps2):
        st.add_layer(0.25, shapes=shapes, background_eps=bg)
    return st


if not SKIP_SOLVE:
    LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.4)
    sol = {}
    for d in (0.0, 5e-14, 5e-13):
        rec = {}
        st = stack_for(d)
        rec["cmap_is_none"] = st.cmap is None
        t0 = time.perf_counter()
        r = solve(st, theta=0.3, phi=0.2)
        rec["wall"] = time.perf_counter() - t0
        sol[d] = r
        try:
            stt = stack_for(d, eps2=LC)
            solve(stt, theta=0.3, phi=0.2)
            rec["tensor"] = "solved"
        except Exception as e:  # noqa: BLE001
            rec["tensor"] = f"{type(e).__name__}: {str(e)[:160]}"
        OUT.setdefault("boundary_solve", {})[repr(d)] = rec
    for d in (5e-14, 5e-13):
        OUT["boundary_solve"][repr(d)]["dRT_vs_d0"] = float(max(
            np.abs(sol[d][1] - sol[0.0][1]).max(),
            np.abs(sol[d][2] - sol[0.0][2]).max()))
        OUT["boundary_solve"][repr(d)]["bytes_equal_d0"] = bool(
            sol[d][1].tobytes() == sol[0.0][1].tobytes()
            and sol[d][2].tobytes() == sol[0.0][2].tobytes())
    print("boundary_solve", OUT["boundary_solve"])

def min_detJ(cm, n=24):
    s = 0.5 + 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)   # interior
    U0, V0 = cm.u_bounds, cm.v_bounds
    m = np.inf
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            U = U0[i] + s * (U0[i + 1] - U0[i])
            V = V0[j] + s * (V0[j + 1] - V0[j])
            _X, _Y, xu, xv, yu, yv = cm.geom(i, j, U, V)
            m = min(m, float(np.min(xu * yv - xv * yu)))
    return m



# ---- (c) snap safety: a sinusoid of tiny amplitude --------------------------
tiny = {}
for amp in (3e-14, 5e-13, 1e-9):
    sw = SinusoidalWall("x", 0.6 * P, amp * P, eps=2.25)
    try:
        cell, xw, yw, cm = compile_shapes(P, P, [sw], 1.0)
        rec = dict(cmap=None if cm is None else type(cm).__name__)
        if cm is not None:
            rec["wrong"] = VC._wrong_map(cm, [cell], [[sw]], [1.0])
            rec["min_detJ"] = min_detJ(cm)
            if not SKIP_SOLVE:
                st = PMM2DStackPure(P, P, n_superstrate=1.0,
                                    n_substrate=1.45, n_modes=4, n_orders=1)
                st.add_layer(0.3, shapes=[sw], background_eps=1.0)
                r = solve(st)
                st0 = PMM2DStackPure(P, P, n_superstrate=1.0,
                                     n_substrate=1.45, n_modes=4, n_orders=1)
                st0.add_layer(0.3, shapes=[SinusoidalWall(
                    "x", 0.6 * P, 0.0, eps=2.25)], background_eps=1.0)
                r0 = solve(st0)
                rec["dRT_vs_flat"] = float(max(np.abs(r[1] - r0[1]).max(),
                                               np.abs(r[2] - r0[2]).max()))
                rec["closure"] = float(np.abs(r[1].sum(1) + r[2].sum(1)
                                              - 1).max())
        tiny[repr(amp)] = rec
    except Exception as e:  # noqa: BLE001
        tiny[repr(amp)] = dict(err=f"{type(e).__name__}: {str(e)[:160]}")
OUT["tiny_sinusoid"] = tiny
print("tiny", tiny)

# ---- (d) proposed fix: _VERTEX_SNAP = _WALL_SNAP ----------------------------
if hasattr(SH, "_VERTEX_SNAP"):
    keep = SH._VERTEX_SNAP
    SH._VERTEX_SNAP = SH._WALL_SNAP
    OUT["boundary_snap_eq_wall"] = boundary_scan()
    rnd2 = {}
    for Pp in (1.2, 1.0, 1.2e-6):
        rng = np.random.default_rng(20261003)
        n_map = 0
        for _ in range(220):
            try:
                n_map += not bool(SH._merge(Pp, Pp, rand_layout(rng, Pp))[4])
            except ValueError:
                pass
        rnd2[repr(Pp)] = n_map
    OUT["random_rects_snap_eq_wall_missed"] = rnd2
    SH._VERTEX_SNAP = keep
    print("snap=wall", {k: v["outcome"] for k, v in
                        OUT["boundary_snap_eq_wall"].items()})


# ---- (e) rotated Ellipse sweep ---------------------------------------------
ell = {}
nbad = 0
for asp in (1.01, 1.05, 1.5, 2.0, 3.0, 4.0, 5.0):
    a = 0.4
    b = a / asp
    for ang in (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 44.0, 44.9, -10.0, -30.0,
                -44.9):
        e = Ellipse(0.6, 0.6, a, b, 4.0, angle=np.deg2rad(ang))
        key = f"{asp}_{ang}"
        try:
            cell, _x, _y, cm = compile_shapes(P, P, [e], 1.0)
            wr = VC._wrong_map(cm, [cell], [[e]], [1.0])
            md = min_detJ(cm)
            ok = wr == (0, 0) and md > 0
            ell[key] = dict(outcome="EXACT" if ok else "WRONG", wrong=wr,
                            min_detJ=md)
        except ValueError as ex:
            ok = False
            ell[key] = dict(outcome="FOLD" if "FOLDS" in str(ex)
                            else "REFUSED", msg=str(ex)[:100])
        nbad += not ok
OUT["ellipse"] = ell
OUT["ellipse_bad"] = nbad
OUT["ellipse_total"] = len(ell)
print("ellipse bad", nbad, "of", len(ell))
edge = {}
for ang in (44.99, 45.0, 50.0, -45.0):
    try:
        e = Ellipse(0.6, 0.6, 0.4, 0.2, 4.0, angle=np.deg2rad(ang))
        cell, _x, _y, cm = compile_shapes(P, P, [e], 1.0)
        edge[repr(ang)] = dict(outcome="laid out",
                               wrong=VC._wrong_map(cm, [cell], [[e]], [1.0]),
                               min_detJ=min_detJ(cm))
    except Exception as ex:  # noqa: BLE001
        edge[repr(ang)] = dict(outcome=f"{type(ex).__name__}",
                               msg=str(ex)[:200])
OUT["ellipse_edge"] = edge
print("edge", edge)

# ---- (f) solve-level mirror check -------------------------------------------
if not SKIP_SOLVE:
    def ell_stack(ang, M=4):
        st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                            n_modes=M, n_orders=1)
        st.add_layer(0.3, shapes=[Ellipse(0.6, 0.6, 0.4, 0.2, 4.0,
                                          angle=np.deg2rad(ang))],
                     background_eps=1.0)
        return st
    mir = {}
    for M in (4, 5):
        try:
            o, Rp, Tp, Jp = solve(ell_stack(30.0, M))
        except ValueError as ex:
            mir[M] = dict(err=str(ex)[:120])
            continue
        o2, Rm, Tm, Jm = solve(ell_stack(-30.0, M))
        idx = {(int(a), int(b)): k for k, (a, b) in enumerate(o2)}
        perm = [idx[(int(a), -int(b))] for a, b in o]
        # the mirror y -> -y swaps nothing between the two lab inputs'
        # efficiencies: R[pol] of input x / y map onto themselves
        dR = float(np.abs(Rp - Rm[:, perm]).max())
        dT = float(np.abs(Tp - Tm[:, perm]).max())
        D = np.diag([1.0, -1.0])
        dJ = float(np.abs(Jp - D @ Jm @ D).max())
        clo = float(np.abs(Rp.sum(1) + Tp.sum(1) - 1).max())
        mir[M] = dict(dR=dR, dT=dT, dJones=dJ, closure=clo,
                      jones_shape=list(np.shape(Jp)))
    OUT["mirror"] = mir
    print("mirror", mir)

dump(f"v9_vd1_vd2_{TREE}", OUT)

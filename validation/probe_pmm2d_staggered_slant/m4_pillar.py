"""M4 -- a genuinely 2-D SLANTED PILLAR, three ways.

Geometry: ``px = py = 1.2 lam``, pillar ``[0.3, 0.6] x [0.3, 0.6]`` (a quarter-
period square), ``eps`` 4 / 1, ``depth = 0.8 lam``, ``n_sub = 1.5``.  The pillar
translates by ``t * depth = (0.6, 0)`` or ``(0.6, 0.6)`` -- i.e. HALF a period
over the layer, ``t_x = 0.75`` (36.9 deg).  Every staircase wall then lands on
the uniform ``Nx = Ny = 8`` grid (spacing 0.15), which the staggered basis
requires (``Basis1D`` segments are uniform).

Arms:
  (a) the PROTOTYPE metric layer (ONE solve, Nx = 4 and Nx = 8 -- also a grid /
      position-invariance check);
  (b) the shipped HYBRID slant metric ``PMM2DStackHybrid.add_layer(slant=...)``
      -- an INDEPENDENT formulation (Fourier basis, in-plane tensor fold,
      convection on the 4N generator) with its own Fourier truncation floor;
  (c) a z-STAIRCASE built from the PURE solver itself, on the union grid,
      converging in the slice count ``n`` (two samplings: midpoint n = 1, 2 and
      leading-edge n = 1, 2, 4 -- the only ones whose walls land on the uniform
      grid at this slant).

The hybrid's slant sign convention is pinned two-sided here (both ``+t`` and
``-t`` are run against the prototype).  The hybrid's normal-incidence
even-parity fold is DISABLED (``symmetry=False``) -- the load-bearing finding of
BUILD_PMM2D_SLANT_METRIC_2026_08_16 S3.4.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import assert_worktree, solve_slant_stack  # noqa: E402
from lumenairy.elements.pmm import PMM2DStackHybrid  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PX = PY = 1.2
WL, DEPTH = 1.0, 0.8
EPS_P, EPS_B = 4.0, 1.0
NSUP, NSUB = 1.0, 1.5
TX = 0.75                      # t*depth = 0.6 = PX/2
ORD_CMP = [(-1, 0), (0, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1)]
res = {"lumenairy": assert_worktree()}
t00 = time.time()


def cell_at(N, shift_x, shift_y):
    """Scalar (N, N) cell: the pillar [0.3, 0.6]^2 translated by (sx, sy),
    wrapped periodically, on the uniform N-segment grid."""
    h = PX / N
    c = np.full((N, N), EPS_B, dtype=complex)
    for i in range(N):
        xa = i * h
        for j in range(N):
            ya = j * h
            # segment centre, un-shifted back into the frame
            xc = (xa + 0.5 * h - shift_x) % PX
            yc = (ya + 0.5 * h - shift_y) % PY
            if 0.3 <= xc < 0.6 and 0.3 <= yc < 0.6:
                c[i, j] = EPS_P
    return c


def per_order(o, R, T):
    idx = {(int(m), int(n)): i for i, (m, n) in enumerate(o)}
    return {k: (float(R[0, idx[k]]), float(R[1, idx[k]]),
                float(T[0, idx[k]]), float(T[1, idx[k]]))
            for k in ORD_CMP if k in idx}


def dd(a, b):
    return max(max(abs(x - y) for x, y in zip(a[k], b[k]))
               for k in a if k in b)


def proto(N, M, slant, nsl=1, sampling="metric"):
    """nsl = 1 & sampling='metric' -> ONE slanted layer; otherwise a staircase
    of nsl VERTICAL layers on the same union grid."""
    if sampling == "metric":
        layers = [{"thickness": DEPTH, "cell": cell_at(N, 0.0, 0.0),
                   "slant": slant}]
    else:
        h = DEPTH / nsl
        layers = []
        for i in range(nsl):
            f = (i + 0.5) / nsl if sampling == "mid" else i / nsl
            layers.append({"thickness": h,
                           "cell": cell_at(N, slant[0] * DEPTH * f,
                                           slant[1] * DEPTH * f),
                           "slant": (0.0, 0.0)})
    tt = time.time()
    o, R, T, _Jr, _Jt, _i = solve_slant_stack(
        PX, PY, layers, NSUP, NSUB, WL, M=M, n_orders=3, theta=TH, phi=PH)
    return per_order(o, R, T), time.time() - tt, float(
        np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))


def hybrid(slant, n_orders, degree=9):
    st = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                          degree=degree, n_orders=n_orders, symmetry=False,
                          formulation="li")
    st.add_layer(DEPTH, eps_cell=cell_at(48, 0.0, 0.0), slant=slant)
    st.set_source(WL, theta=TH, phi=PH)
    tt = time.time()
    out = st.solve()
    o, R, T = out[0], out[1], out[2]
    return per_order(o, np.atleast_2d(R), np.atleast_2d(T)), time.time() - tt


for TH, PH, mount in ((0.0, 0.0, "normal"),
                      (np.deg2rad(20), np.deg2rad(35), "conical20_35")):
    for sname, slant in (("x", (TX, 0.0)), ("diag", (TX, TX))):
        key = f"{mount}/{sname}"
        print(f"\n=== M4 {key} ===")
        rows = {}
        # (a) prototype metric layer, grid + M ladder
        met = {}
        for N, M in ((4, 5), (4, 6), (8, 4), (8, 5)):
            v, t, clo = proto(N, M, slant)
            met[f"N{N}_M{M}"] = v
            rows[f"metric_N{N}_M{M}"] = {"t_s": t, "closure": clo}
            print(f"  metric N={N} M={M}  {t:5.1f}s  |R+T-1| {clo:.2e}")
        ref = met["N8_M5"]
        rows["metric_selfmove_N4M5_vs_N8M5"] = dd(met["N4_M5"], ref)
        rows["metric_selfmove_N4M6_vs_N8M5"] = dd(met["N4_M6"], ref)
        rows["metric_selfmove_N8M4_vs_N8M5"] = dd(met["N8_M4"], ref)
        print(f"  metric self-move: N4M5 {rows['metric_selfmove_N4M5_vs_N8M5']:.2e}"
              f" N4M6 {rows['metric_selfmove_N4M6_vs_N8M5']:.2e}"
              f" N8M4 {rows['metric_selfmove_N8M4_vs_N8M5']:.2e}")
        # (b) hybrid, both slant signs, order ladder
        hyb = {}
        for sgn in (+1.0, -1.0):
            for no in (5, 7, 9):
                try:
                    v, t = hybrid((sgn * slant[0], sgn * slant[1]), no)
                except Exception as exc:                    # noqa: BLE001
                    rows[f"hybrid_s{sgn:+.0f}_n{no}"] = f"RAISED {exc}"
                    print(f"  hybrid t*{sgn:+.0f} n={no}: RAISED {exc}")
                    continue
                hyb[(sgn, no)] = v
                rows[f"hybrid_s{sgn:+.0f}_n{no}"] = {
                    "vs_metric": dd(v, ref), "t_s": t}
                print(f"  hybrid t*{sgn:+.0f} n_orders={no}  vs metric "
                      f"{dd(v, ref):.3e}  ({t:.1f}s)")
        for sgn in (+1.0, -1.0):
            if (sgn, 7) in hyb and (sgn, 9) in hyb:
                rows[f"hybrid_s{sgn:+.0f}_owndrift_7to9"] = dd(hyb[(sgn, 7)],
                                                               hyb[(sgn, 9)])
        # (c) pure staircase on the union grid
        for samp, ns in (("mid", (1, 2)), ("lead", (1, 2, 4))):
            for n in ns:
                v, t, clo = proto(8, 4, slant, nsl=n, sampling=samp)
                rows[f"stair_{samp}_n{n}"] = {"vs_metric": dd(v, ref),
                                              "t_s": t, "closure": clo}
                print(f"  staircase {samp} n={n}  vs metric {dd(v, ref):.3e}"
                      f"  ({t:.1f}s, |R+T-1| {clo:.2e})")
        res[key] = rows

res["wall_s"] = time.time() - t00
with open(os.path.join(OUT, "m4_pillar.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print(f"\nWROTE results/m4_pillar.json  ({res['wall_s']:.1f} s)")

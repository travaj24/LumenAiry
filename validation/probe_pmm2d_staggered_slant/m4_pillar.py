"""M4 -- a genuinely 2-D SLANTED PILLAR, three ways.

Geometry: ``px = py = 1.2 lam``, pillar ``[0.3, 0.6] x [0.3, 0.6]`` (a quarter-
period square), ``eps`` 4 / 1, ``depth = 0.8 lam``, ``n_sub = 1.5``, ``wl = 1``.
The pillar translates by ``t * depth = 0.6`` = HALF a period over the layer
(``t = 0.75``, i.e. 36.9 deg).  Every staircase wall then lands on the uniform
``Nx = Ny = 8`` grid (spacing 0.15) -- which the pure staggered solver REQUIRES,
because ``Basis1D``'s segments are a ``linspace``.  That constraint is itself
part of the finding (S4.6): a z-staircase in this engine cannot keep each
slice's own grid the way the Fourier hybrid can.

Arms:
  (a) the PROTOTYPE metric layer -- ONE solve, on TWO different grids
      (``Nx = 4`` and ``Nx = 8``) and two modal counts, which is also a
      grid / position-invariance check;
  (b) the shipped HYBRID slant metric ``PMM2DStackHybrid.add_layer(slant=...)``
      -- an INDEPENDENT formulation (Fourier basis, in-plane tensor fold, a
      convection on the 4N generator) with its own Fourier truncation floor.
      Its slant sign is pinned two-sided here.  Its normal-incidence
      even-parity fold is DISABLED (``symmetry=False``) -- the load-bearing
      finding of BUILD_PMM2D_SLANT_METRIC_2026_08_16 S3.4;
  (c) a z-STAIRCASE built from the PURE solver itself on the union grid,
      converging in the slice count ``n``.  Two samplings, because only these
      put every wall on the uniform grid at this slant: MIDPOINT n = 1, 2 and
      LEADING-edge n = 1, 2, 4.

Reference for every difference is the prototype metric layer at ``Nx = 8,
M = 4`` (its own self-move against the other three metric rungs is reported, so
the reader can see what that reference is worth).
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
TX = 0.75                       # t * depth = 0.6 = PX / 2
ORD_CMP = [(-1, 0), (0, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1)]
res = {"lumenairy": assert_worktree()}
t00 = time.time()


def cell_at(N, sx, sy):
    """Scalar (N, N) cell: the pillar [0.3, 0.6]^2 translated by (sx, sy),
    wrapped periodically, on the uniform N-segment grid."""
    h = PX / N
    c = np.full((N, N), EPS_B, dtype=complex)
    for i in range(N):
        for j in range(N):
            xc = (i * h + 0.5 * h - sx) % PX
            yc = (j * h + 0.5 * h - sy) % PY
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


def proto(N, M, slant, th, ph, nsl=1, sampling="metric"):
    if sampling == "metric":
        layers = [{"thickness": DEPTH, "cell": cell_at(N, 0.0, 0.0),
                   "slant": slant}]
    else:
        h = DEPTH / nsl
        layers = [{"thickness": h,
                   "cell": cell_at(N, slant[0] * DEPTH * f,
                                   slant[1] * DEPTH * f),
                   "slant": (0.0, 0.0)}
                  for f in [((i + 0.5) / nsl if sampling == "mid" else i / nsl)
                            for i in range(nsl)]]
    tt = time.time()
    o, R, T, _Jr, _Jt, _i = solve_slant_stack(
        PX, PY, layers, NSUP, NSUB, WL, M=M, n_orders=3, theta=th, phi=ph)
    return (per_order(o, R, T), time.time() - tt,
            float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))))


def hybrid(slant, th, ph, n_orders, degree=9):
    st = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                          degree=degree, n_orders=n_orders, symmetry=False,
                          formulation="li")
    st.add_layer(DEPTH, eps_cell=cell_at(48, 0.0, 0.0), slant=slant)
    st.set_source(WL, theta=th, phi=ph)
    tt = time.time()
    out = st.solve()
    return (per_order(out[0], np.atleast_2d(out[1]), np.atleast_2d(out[2])),
            time.time() - tt)


BLOCKS = (("normal", 0.0, 0.0, "diag", (TX, TX)),
          ("conical20_35", np.deg2rad(20), np.deg2rad(35), "x", (TX, 0.0)))

for mount, TH, PH, sname, slant in BLOCKS:
    key = f"{mount}/{sname}"
    print(f"\n=== M4 {key} ===", flush=True)
    rows = {}
    met = {}
    for N, M in ((4, 5), (4, 6), (8, 3), (8, 4)):
        v, t, clo = proto(N, M, slant, TH, PH)
        met[f"N{N}_M{M}"] = v
        rows[f"metric_N{N}_M{M}"] = {"t_s": t, "closure": clo,
                                     "dim": int(4 * (N * (M - 1)) ** 2)}
        print(f"  metric N={N} M={M} dim={rows[f'metric_N{N}_M{M}']['dim']:5d} "
              f"{t:6.1f}s  |R+T-1| {clo:.2e}", flush=True)
    ref = met["N8_M4"]
    for k in ("N4_M5", "N4_M6", "N8_M3"):
        rows[f"metric_selfmove_{k}"] = dd(met[k], ref)
    print("  metric self-move vs N8_M4: "
          + " ".join(f"{k} {rows[f'metric_selfmove_{k}']:.2e}"
                     for k in ("N4_M5", "N4_M6", "N8_M3")), flush=True)

    hyb = {}
    for sgn in (+1.0, -1.0):
        for no in (5, 7):
            try:
                v, t = hybrid((sgn * slant[0], sgn * slant[1]), TH, PH, no)
            except Exception as exc:                        # noqa: BLE001
                rows[f"hybrid_s{sgn:+.0f}_n{no}"] = f"RAISED {exc}"
                print(f"  hybrid t*{sgn:+.0f} n={no}: RAISED {exc}", flush=True)
                continue
            hyb[(sgn, no)] = v
            rows[f"hybrid_s{sgn:+.0f}_n{no}"] = {"vs_metric": dd(v, ref),
                                                 "t_s": t}
            print(f"  hybrid t*{sgn:+.0f} n_orders={no}  vs metric "
                  f"{dd(v, ref):.3e}  ({t:.1f}s)", flush=True)
    for sgn in (+1.0, -1.0):
        if (sgn, 5) in hyb and (sgn, 7) in hyb:
            rows[f"hybrid_s{sgn:+.0f}_owndrift_5to7"] = dd(hyb[(sgn, 5)],
                                                           hyb[(sgn, 7)])

    for samp, ns in (("mid", (1, 2)), ("lead", (1, 2, 4))):
        for n in ns:
            v, t, clo = proto(8, 3, slant, TH, PH, nsl=n, sampling=samp)
            rows[f"stair_M3_{samp}_n{n}"] = {"vs_metric": dd(v, ref),
                                             "t_s": t, "closure": clo}
            print(f"  staircase M=3 {samp} n={n}  vs metric {dd(v, ref):.3e}"
                  f"  ({t:.1f}s, |R+T-1| {clo:.2e})", flush=True)
    res[key] = rows

# one deeper staircase ladder (M = 4) as the "not an M=3 artifact" control
mount, TH, PH, sname, slant = BLOCKS[0]
print(f"\n=== M4 control: staircase at M = 4, {mount}/{sname} ===", flush=True)
ctrl = {}
v4, t4, c4 = proto(8, 4, slant, TH, PH)
ctrl["metric_N8_M4"] = {"t_s": t4, "closure": c4}
for n in (1, 2, 4):
    v, t, clo = proto(8, 4, slant, TH, PH, nsl=n, sampling="lead")
    ctrl[f"stair_M4_lead_n{n}"] = {"vs_metric": dd(v, v4), "t_s": t,
                                   "closure": clo}
    print(f"  staircase M=4 lead n={n}  vs metric {dd(v, v4):.3e}  "
          f"({t:.1f}s, |R+T-1| {clo:.2e})", flush=True)
res["control_M4_staircase"] = ctrl

res["wall_s"] = time.time() - t00
with open(os.path.join(OUT, "m4_pillar.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print(f"\nWROTE results/m4_pillar.json  ({res['wall_s']:.1f} s)")

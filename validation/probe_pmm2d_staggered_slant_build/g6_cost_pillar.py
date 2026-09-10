"""COST + the genuinely 2-D SLANTED PILLAR.

COST -- the slanted region solve against the VERTICAL OUT-OF-PLANE one it has
to use anyway (a sheared cell is an out-of-plane cell in the frame), and against
the in-plane ``2 q^2`` pencil.

PILLAR -- a quarter-period square pillar solved three ways:
  (a) the shipped metric layer, on four ``(Nx, M)`` settings (self-move);
  (b) the INDEPENDENT ``PMM2DStackHybrid`` slant metric (Fourier basis, tensor
      fold, lab-Cartesian convection -- nothing in common but the physics),
      both SIGNS and an ``n_orders`` ladder;
  (c) a z-STAIRCASE built from the PURE solver itself, marching WITH and
      AGAINST the slant.
"""
import time
import warnings

import numpy as np
from _lib import write

from lumenairy.elements.pmm import PMM2DStackHybrid, PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    Granet2DTransverseE,
    _region_modes,
    _region_modes_oop,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor

res = {}
t00 = time.time()

# ------------------------------------------------------------------- COST
print("COST  per-region assembly + eig, (2,2) grid, conical Bloch shift")
K0 = 2.0 * np.pi
TIL = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
SCA = np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex)
OOPC = np.zeros((2, 2, 3, 3), dtype=complex)
OOPC[:, :] = np.eye(3)
OOPC[0, 0] = TIL
T369 = float(np.tan(np.deg2rad(36.9)))
A0 = (0.25 * K0, 0.18 * K0)


def timeit(fn, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


cost = []
for M in (4, 5, 6, 7):
    def inplane():
        s = Granet2DTransverseE(1.2, 1.2, 2, 2, M, SCA, alpha0x=A0[0],
                                alpha0y=A0[1], k0=K0)
        _region_modes(s)

    def oop():
        s = Granet2DTransverseE(1.2, 1.2, 2, 2, M, OOPC, alpha0x=A0[0],
                                alpha0y=A0[1], k0=K0)
        _region_modes_oop(s)

    def slanted():
        s = Granet2DTransverseE(1.2, 1.2, 2, 2, M, SCA, alpha0x=A0[0],
                                alpha0y=A0[1], k0=K0, slant=(T369, 0.0))
        _region_modes_oop(s)

    ti, to, ts = timeit(inplane), timeit(oop), timeit(slanted)
    q2 = (2 * (M - 1)) ** 2
    cost.append({"M": M, "q2": q2, "inplane_s": ti, "oop_s": to,
                 "slant_s": ts, "slant_over_oop": ts / to,
                 "slant_over_inplane": ts / ti})
    print("  M=%d q^2=%3d  in-plane 2q^2 %7.1f ms | vertical OOP 4q^2 %7.1f ms"
          " | SLANTED 4q^2 %7.1f ms  -> slant/OOP %.2fx  slant/in-plane %.2fx"
          % (M, q2, ti * 1e3, to * 1e3, ts * 1e3, ts / to, ts / ti))
res["COST_region"] = cost
res["COST_slant_over_oop_range"] = [min(r["slant_over_oop"] for r in cost),
                                    max(r["slant_over_oop"] for r in cost)]

print("")
print("COST  end-to-end single-layer solve (scalar pillar, conical 20/35)")
e2e = []
for M in (5, 6, 7):
    def run(sl):
        st = PMM2DStackPure(1.2, 1.2, n_superstrate=1.0, n_substrate=1.5,
                            n_modes=M, n_orders=3)
        st.add_layer(0.8, eps_cell=SCA, slant=sl)
        st.set_source(1.0, theta=np.deg2rad(20.0), phi=np.deg2rad(35.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.solve()

    tv = timeit(lambda: run(None), 2)
    ts = timeit(lambda: run((T369, 0.0)), 2)
    e2e.append({"M": M, "vertical_s": tv, "slanted_s": ts, "ratio": ts / tv})
    print("  M=%d  vertical %.3f s | slanted %.3f s  -> %.2fx" % (M, tv, ts,
                                                                  ts / tv))
res["COST_e2e"] = e2e
res["COST_e2e_ratio_range"] = [min(r["ratio"] for r in e2e),
                               max(r["ratio"] for r in e2e)]

# ------------------------------------------------------------------ PILLAR
print("")
print("PILLAR  quarter-period square, t = (0.75, 0), depth 0.8, eps 4/1")
PXP = PYP = 1.2
WLP = 1.0
DEPP = 0.8
TP = 0.75
NSUBP = 1.5
KEYS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]


def pillar_cell(nx, x0, x1, y0, y1, hi=4.0, lo=1.0):
    """Square pillar [x0, x1) x [y0, y1) in period units on an nx x nx grid."""
    c = np.full((nx, nx), lo, dtype=complex)
    h = 1.0 / nx
    for i in range(nx):
        for j in range(nx):
            xc, yc = (i + 0.5) * h, (j + 0.5) * h
            if x0 - 1e-9 <= xc < x1 - 1e-9 and y0 - 1e-9 <= yc < y1 - 1e-9:
                c[i, j] = hi
    return c


def pure_solve(cells, slants, depths, M, n_orders, theta, phi):
    st = PMM2DStackPure(PXP, PYP, n_superstrate=1.0, n_substrate=NSUBP,
                        n_modes=M, n_orders=n_orders)
    for c, s, d in zip(cells, slants, depths):
        st.add_layer(d, eps_cell=c, slant=s)
    st.set_source(WLP, theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve(jones=True)


def vec(o, R, T):
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(o)}
    return np.concatenate([np.array([R[:, idx[k]] for k in KEYS]).ravel(),
                           np.array([T[:, idx[k]] for k in KEYS]).ravel()])


MOUNTS = (("normal", 0.0, 0.0),
          ("conical", np.deg2rad(20.0), np.deg2rad(35.0)))
print("  (a) the shipped metric layer -- grid / degree SELF-MOVE")
selfmove = []
ref = {}
for mn, th, ph in MOUNTS:
    c8 = pillar_cell(8, 0.25, 0.5, 0.25, 0.5)
    o, R, T, J = pure_solve([c8], [(TP, 0.0)], [DEPP], 4, 3, th, ph)
    ref[mn] = vec(o, R, T)
    for nx, M in ((4, 5), (4, 6), (8, 3), (12, 3)):
        c = pillar_cell(nx, 0.25, 0.5, 0.25, 0.5)
        o2, R2, T2, J2 = pure_solve([c], [(TP, 0.0)], [DEPP], M, 3, th, ph)
        d = float(np.max(np.abs(vec(o2, R2, T2) - ref[mn])))
        selfmove.append({"mount": mn, "Nx": nx, "M": M, "vs_ref": d})
        print("     %-8s Nx=%2d M=%d  %.2e" % (mn, nx, M, d))
res["PILLAR_selfmove"] = selfmove
res["PILLAR_selfmove_worst"] = max(r["vs_ref"] for r in selfmove)

print("  (b) the INDEPENDENT hybrid slant metric (both signs, n_orders ladder)")
hyb = []
for mn, th, ph in MOUNTS:
    for sgn in (+1.0, -1.0):
        prev = None
        for no in (5, 7, 9):
            hs = PMM2DStackHybrid(PXP, PYP, n_superstrate=1.0,
                                  n_substrate=NSUBP, n_orders=no)
            hs.add_layer(DEPP, eps_cell=pillar_cell(8, 0.25, 0.5, 0.25, 0.5),
                         slant=(sgn * TP, 0.0))
            hs.set_source(WLP, theta=th, phi=ph)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oh, Rh, Th, Jh = hs.solve()
            v = vec(oh, Rh, Th)
            step = None if prev is None else float(np.max(np.abs(v - prev)))
            prev = v
            d = float(np.max(np.abs(v - ref[mn])))
            hyb.append({"mount": mn, "sign": "+t" if sgn > 0 else "-t",
                        "n_orders": no, "vs_pure": d, "own_step": step})
            print("     %-8s %s n_orders=%2d  vs pure %.3e   own step %s"
                  % (mn, "+t" if sgn > 0 else "-t", no, d,
                     "-" if step is None else "%.3e" % step))
res["PILLAR_hybrid"] = hyb

print("  (c) a z-STAIRCASE from the PURE solver (both marching directions)")
stair = []
for mn, th, ph in MOUNTS:
    for direc in (+1.0, -1.0):
        for n in (1, 2, 4):
            cells, slants, depths = [], [], []
            step = TP * DEPP / n           # lateral walk per slice, period 1.2
            for k in range(n):
                sh = direc * (step * k) / PXP
                cells.append(pillar_cell(8, 0.25 + sh, 0.5 + sh, 0.25, 0.5))
                slants.append(None)
                depths.append(DEPP / n)
            o2, R2, T2, J2 = pure_solve(cells, slants, depths, 3, 3, th, ph)
            # reference on the SAME grid/degree so the discretization is
            # common-mode
            o1, R1, T1, J1 = pure_solve(
                [pillar_cell(8, 0.25, 0.5, 0.25, 0.5)], [(TP, 0.0)], [DEPP],
                3, 3, th, ph)
            d = float(np.max(np.abs(vec(o2, R2, T2) - vec(o1, R1, T1))))
            stair.append({"mount": mn,
                          "direction": "with" if direc > 0 else "against",
                          "n": n, "vs_metric": d})
            print("     %-8s march %-7s n=%d  %.3e"
                  % (mn, "with" if direc > 0 else "against", n, d))
res["PILLAR_staircase"] = stair

res["wall_s"] = time.time() - t00
write("g6_cost_pillar", res)

"""M6 -- z-STAIRCASE taper, the use case: each slice drawn on the grid its own
pillar width needs, against the union-grid staircase (one fine lattice).

Granet's lattice is UNIFORM, so a slice whose pillar width is ``1/N`` is
representable on ``N`` segments and on every MULTIPLE of ``N`` -- and the
common lattice of a set of slices is the LCM of their ``N``.  That LCM is what
the union-grid stack must pay for EVERY slice.

(A) ACCURACY: widths 1/2, 1/3, 1/6 -> per-layer N = 2, 3, 6; LCM = 6, so the
    union reference is computable.  Modal ladder.
(B) COST: widths 1/2, 1/3, 1/4 -> per-layer N = 2, 3, 4; LCM = 12.  Measured at
    the minimum modal counts, plus the arithmetic at a production M."""
import json
import sys
import time
import warnings
import numpy as np
from mortar2d import guard, MortarStack2D
print("lumenairy:", guard(), flush=True)
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

warnings.simplefilter("ignore")
PX = PY = 1.2e-6
WL = 0.85e-6
EPS_P, EPS_H = 9.0, 2.25
TH, PH = 0.18, 0.0
NORD = 2
TS = [0.12e-6, 0.12e-6, 0.12e-6]


def pillar(N, k):
    c = np.full((N, N), EPS_H + 0j)
    c[:k, :k] = EPS_P
    return c


def run_union(Ns, ks, M):
    s = PMM2DStackPure(PX, PY, n_modes=M, n_orders=NORD)
    for N, k, t in zip(Ns, ks, TS):
        s.add_layer(t, eps_cell=pillar(N, k))
    s.set_source(WL, theta=TH, phi=PH)
    t0 = time.perf_counter()
    out = s.solve()
    return out, time.perf_counter() - t0


def run_mortar(Ns, ks, M):
    s = MortarStack2D(PX, PY, n_modes=M, n_orders=NORD)
    for N, k, t in zip(Ns, ks, TS):
        s.add_layer(t, eps_cell=pillar(N, k))
    s.set_source(WL, theta=TH, phi=PH)
    t0 = time.perf_counter()
    out = s.solve()
    return out, time.perf_counter() - t0, s


rows = []
print("(A) widths 1/2, 1/3, 1/6 -- per-layer N = 2,3,6 vs union N = 6")
for M in [int(x) for x in (sys.argv[1:] or [4, 5, 6])]:
    (o0, R0, T0, J0), t0 = run_union([6, 6, 6], [3, 2, 1], M)
    (o1, R1, T1, J1), t1, s = run_mortar([2, 3, 6], [1, 1, 1], M)
    assert np.array_equal(o0, o1)
    sc = max(float(np.max(R0)), float(np.max(T0)))
    d = float(np.max(np.abs(np.concatenate([R0.ravel(), T0.ravel()])
                            - np.concatenate([R1.ravel(), T1.ravel()])))) / sc
    dJ = float(np.max(np.abs(J0 - J1))) / float(np.max(np.abs(J0)))
    c0 = max(abs(R0[p].sum() + T0[p].sum() - 1.0) for p in (0, 1))
    c1 = max(abs(R1[p].sum() + T1[p].sum() - 1.0) for p in (0, 1))
    rows.append(dict(part="A", M=M, d=d, dJ=dJ, clo_union=float(c0),
                     clo_mortar=float(c1), t_union=t0, t_mortar=t1))
    print(f"  M={M}  d(R,T) {d:9.2e}  dJ {dJ:9.2e} | closure union {c0:8.1e} "
          f"mortar {c1:8.1e} | t_union {t0:7.1f}s t_mortar {t1:7.1f}s "
          f"({t0/t1:5.2f}x)", flush=True)

print("\n(B) widths 1/2, 1/3, 1/4 -- per-layer N = 2,3,4 vs union N = 12")
for M in (3, 4):
    (o1, R1, T1, J1), t1, s = run_mortar([2, 3, 4], [1, 1, 1], M)
    try:
        (o0, R0, T0, J0), t0 = run_union([12, 12, 12], [6, 4, 3], M)
        sc = max(float(np.max(R0)), float(np.max(T0)))
        d = float(np.max(np.abs(np.concatenate([R0.ravel(), T0.ravel()])
                                - np.concatenate([R1.ravel(), T1.ravel()])))) / sc
        c0 = float(max(abs(R0[p].sum() + T0[p].sum() - 1.0) for p in (0, 1)))
    except Exception as exc:                     # noqa: BLE001
        t0, d, c0 = float("nan"), float("nan"), float("nan")
        print("  union N=12 failed:", exc)
    c1 = max(abs(R1[p].sum() + T1[p].sum() - 1.0) for p in (0, 1))
    rows.append(dict(part="B", M=M, d=d, clo_union=c0, clo_mortar=float(c1),
                     t_union=t0, t_mortar=t1))
    print(f"  M={M}  d(R,T) {d:9.2e} | closure union {c0:8.1e} mortar "
          f"{c1:8.1e} | t_union {t0:7.1f}s t_mortar {t1:7.1f}s "
          f"({t0/t1:5.2f}x)", flush=True)

# eig-dimension arithmetic for the production point
print("\n(B') eig dimension / memory arithmetic (2*(N(M-1))^2 pencil, 16n^2 B)")
for M in (4, 6, 8):
    per = [2 * (N * (M - 1)) ** 2 for N in (2, 3, 4)]
    uni = 2 * (12 * (M - 1)) ** 2
    print(f"  M={M}: per-layer dims {per} (sum n^3 {sum(n**3 for n in per):.3e})"
          f" | union N=12 dim {uni} x3 slices (sum n^3 "
          f"{3*uni**3:.3e}, {16*uni**2/2**30:.3f} GB/matrix) -> "
          f"eig-work ratio {3*uni**3/sum(n**3 for n in per):.1f}x")

json.dump(rows, open("validation/probe_pmm2d_staggered_mortar/m6_staircase.json",
                     "w"), indent=1)

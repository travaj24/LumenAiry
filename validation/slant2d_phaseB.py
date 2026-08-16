"""PHASE A, second half -- a GENUINELY 2-D slanted structure.

A slanted rectangular pillar (crossed/lamellar cell) translating along a
DIAGONAL slant vector (t_x, t_y), solved as ONE metric layer and as an N-slice
staircase, at NORMAL, OBLIQUE and CONICAL incidence.

Why n_orders can be modest here: the metric layer and the staircase share the
SAME Rayleigh truncation, so the hybrid's Fourier floor is COMMON MODE and
cancels in their difference.  This comparison therefore isolates the staircase
discretization error (M5 S3.2's ladder logic) and is NOT floor-limited, unlike
the 2-D-vs-1-D comparison in gate A4.

The 1-D oracle does not exist for this geometry (it is not y-uniform) and the
shipped 1-D slant path refuses conical incidence outright, so the staircase is
the only available reference -- which is exactly why the DIRECTION of
convergence, not an absolute number, is the claim.
"""
import time
import warnings

import numpy as np

import slant2d_proto as SP
from lumenairy.elements.pmm import pmm_jones_2d
from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid

warnings.simplefilter("ignore")

P = 700e-9
WL = 633e-9
D = 300e-9
ER, EG = 4.0 + 0j, 1.0 + 0j
NSUB, NSUP = 1.5, 1.0
NPIX = 480
DEG = 11

# pillar bounds in pixels (fixed cross-section; only its POSITION walks)
PX0, PX1 = int(0.25 * NPIX), int(0.60 * NPIX)
PY0, PY1 = int(0.30 * NPIX), int(0.65 * NPIX)

WALK_X_PX = 96          # 140 nm over d -> t_x = 0.4667 (25.0 deg)
WALK_Y_PX = 48          #  70 nm over d -> t_y = 0.2333 (13.1 deg)
TX = WALK_X_PX / NPIX * P / D
TY = WALK_Y_PX / NPIX * P / D


def pillar_cell(sx=0, sy=0):
    """Rectangular pillar rolled by INTEGER pixel offsets (exact walls)."""
    cell = np.zeros((NPIX, NPIX, 3, 3), dtype=np.complex128)
    line = np.full((NPIX, NPIX), EG, dtype=np.complex128)
    line[PX0:PX1, PY0:PY1] = ER
    line = np.roll(np.roll(line, int(sx), axis=0), int(sy), axis=1)
    for i in range(3):
        cell[:, :, i, i] = line
    return cell


def _n_all(o, R, T):
    o = np.asarray(o)
    idx = np.lexsort((o[:, 1], o[:, 0]))
    return np.asarray(R)[:, idx], np.asarray(T)[:, idx]


def metric_layer(n_ord, theta, phi_inc):
    SP.set_slant(TX, TY, -1j)
    try:
        t0 = time.time()
        o, R, T, J = pmm_jones_2d(P, P, pillar_cell(), NSUB, NSUP, D, WL,
                                  theta=theta, phi=phi_inc, degree=DEG,
                                  n_orders=n_ord, formulation="laurent")
        dt = time.time() - t0
    finally:
        SP.set_slant(0.0, 0.0)
    return _n_all(o, R, T), dt


def staircase(ns, n_ord, theta, phi_inc):
    t0 = time.time()
    st = PMM2DStackHybrid(P, P, n_substrate=NSUB, n_superstrate=NSUP,
                          degree=DEG, n_orders=n_ord, formulation="laurent")
    for k in range(ns):
        # frame anchored at the layer TOP (add_sheared_grating's convention);
        # stack is superstrate-first so slice k is the k-th from the top
        f = (k + 0.5) / ns
        st.add_layer(D / ns, eps_tensor_cell=pillar_cell(
            int(round(WALK_X_PX * f)), int(round(WALK_Y_PX * f))))
    st.set_source(WL, theta=theta, phi=phi_inc)
    o, R, T, J = st.solve()
    return _n_all(o, R, T), time.time() - t0


def dmax(a, b):
    (Ra, Ta), (Rb, Tb) = a, b
    return max(float(np.max(np.abs(Ra - Rb))), float(np.max(np.abs(Ta - Tb))))


def closure(x):
    R, T = x
    return float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0)))


CASES = (("normal ", 0.0, 0.0),
         ("oblique", 0.20, 0.0),
         ("conical", 0.20, 0.60))

LADDER = (1, 2, 4, 8, 12, 24)     # all divide both walks under the midpoint rule


def gate_B1(n_ord=7):
    print(f"\n=== B1  genuinely 2-D slanted pillar, metric vs staircase "
          f"(n_orders={n_ord}) ===")
    print(f"    slant vector t = ({TX:.4f}, {TY:.4f}) = "
          f"({np.degrees(np.arctan(TX)):.1f}, {np.degrees(np.arctan(TY)):.1f}) deg")
    print(f"    pillar {(PX1-PX0)/NPIX:.2f} x {(PY1-PY0)/NPIX:.2f} of the cell, "
          f"eps {ER.real}/{EG.real}, d={D*1e9:.0f}nm, P={P*1e9:.0f}nm\n")
    results = {}
    for name, th, ph in CASES:
        met, tm = metric_layer(n_ord, th, ph)
        print(f"  --- {name} (theta={th}, phi={ph}) ---")
        print(f"      metric layer: {tm:6.2f}s   |R+T-1| = {closure(met):.2e}")
        prev = None
        rows = []
        for ns in LADDER:
            st, ts = staircase(ns, n_ord, th, ph)
            d = dmax(st, met)
            rat = f"  ratio {prev/d:5.2f}" if prev else "           "
            print(f"      ns={ns:>2}: |stair-metric| = {d:.3e}{rat}"
                  f"   {ts:6.2f}s   |R+T-1| = {closure(st):.2e}")
            rows.append((ns, d, ts))
            prev = d
        results[name] = (tm, closure(met), rows)
    return results


def gate_B2(results):
    """Accuracy-per-cost: who wins at MATCHED wall-clock, and at typical practice.

    The staircase ladder converges toward the metric layer, so the metric
    layer's own residual against the (unavailable) exact answer is bounded
    ABOVE by its distance to the most-converged staircase rung.  That distance
    is what the staircase must beat to be worth its cost.
    """
    print("\n=== B2  accuracy per unit cost ===")
    for name, (tm, _clos, rows) in results.items():
        best_ns, best_d, _ = rows[-1]
        print(f"\n  --- {name} ---")
        print(f"      metric layer cost {tm:.2f}s; its bound vs the ns={best_ns} "
              f"rung is {best_d:.2e}")
        # matched wall clock: the largest ns whose cost <= the metric layer's
        matched = [r for r in rows if r[2] <= tm]
        if matched:
            ns_m, d_m, t_m = matched[-1]
            print(f"      MATCHED COST  ({t_m:.2f}s <= {tm:.2f}s): staircase "
                  f"ns={ns_m} is {d_m:.2e} from the metric answer")
        else:
            print(f"      MATCHED COST: even ns=1 ({rows[0][2]:.2f}s) costs more "
                  f"than the metric layer ({tm:.2f}s)")
        for ns_t in (8, 12):
            hit = [r for r in rows if r[0] == ns_t]
            if hit:
                ns_x, d_x, t_x = hit[0]
                print(f"      TYPICAL ns={ns_x}: {d_x:.2e} at {t_x:.2f}s "
                      f"= {t_x/tm:.2f}x the metric layer's cost")


if __name__ == "__main__":
    res = gate_B1(7)
    gate_B2(res)

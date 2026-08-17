"""THE ARBITER -- which of the two 2-D routes is converging?

Gate A5 measured |staircase(ns) - metric| falling 2.3e-1 -> 1.0e-2 over
ns = 1..8 and then TURNING AROUND (1.5e-2 at ns=16, 1.8e-2 at ns=32).  Two
readings are consistent with that:

  (i)  the metric layer is wrong at ~1e-2 and the staircase merely passes near
       it on its way to the true answer;
  (ii) the staircase degrades at high ns -- exactly M5 S4's measured pathology
       (the PMM tapered stack scatters non-monotonically across (degree,
       n_slice) with energy conserved in 25 of 26 wrong cells).

On a Y-UNIFORM slanted grating BOTH routes have an exact, independent oracle:
the shipped 1-D slant solver.  Whichever route's distance to that oracle keeps
falling is the one that is converging.  This is the arbiter A5 lacked, and it
is an INDEPENDENT oracle, not an energy check -- the lossless trap is the named
hazard for this campaign.

RUN WITH WARNINGS ENABLED.  The staircase's high-ns cells are DETECTED by the
library's energy guard ("near-singular interface mode-match ... reduce
n_slices"); suppressing warnings here is what produced the initial, wrong
reading that the divergence was silent.  See S13 of the build doc.
"""
import time

import numpy as np
import slant2d_proto as SP

from lumenairy.elements.pmm import pmm_jones_1d_slanted, pmm_jones_2d
from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid

P = 700e-9
WL = 633e-9
D = 300e-9
DUTY = 0.5
ER, EG = 4.0 + 0j, 1.0 + 0j
NSUB, NSUP = 1.5, 1.0
NPIX = 480
DEG = 11
THETA = 0.20

WALK_PX = 96
TX = WALK_PX / NPIX * P / D
PHI_SLANT = float(np.arctan(TX))


def _n0(o, R, T):
    o = np.asarray(o)
    keep = o[:, 1] == 0
    idx = np.argsort(o[keep][:, 0])
    return (o[keep][:, 0][idx], np.asarray(R)[:, keep][:, idx],
            np.asarray(T)[:, keep][:, idx])


def oracle1d():
    o, R, T, J = pmm_jones_1d_slanted(
        P, ER * np.eye(3), EG * np.eye(3), NSUB, NSUP, D, DUTY, WL,
        PHI_SLANT, angle=THETA, degree=20, far_field_orders=21,
        factorization="convection")
    return np.asarray(o), np.asarray(R), np.asarray(T)


def cmp_o(m2, R2, T2, o1, R1, T1):
    common = np.intersect1d(m2, o1)
    i2, i1 = np.searchsorted(m2, common), np.searchsorted(o1, common)
    d = 0.0
    for r in (0, 1):
        d = max(d, float(np.max(np.abs(R2[r][i2] - R1[r][i1]))))
        d = max(d, float(np.max(np.abs(T2[r][i2] - T1[r][i1]))))
    return d


def metric(n_ord):
    """The single exact slanted layer -- now the SHIPPED native path."""
    t0 = time.time()
    o, R, T, J = pmm_jones_2d(P, P, SP.binary_cell(NPIX, DUTY, ER, EG),
                              NSUB, NSUP, D, WL, theta=THETA, phi=0.0,
                              degree=DEG, n_orders=n_ord,
                              formulation="laurent", slant=(TX, 0.0))
    return _n0(o, R, T), time.time() - t0


def stair(ns, n_ord):
    t0 = time.time()
    st = PMM2DStackHybrid(P, P, n_substrate=NSUB, n_superstrate=NSUP,
                          degree=DEG, n_orders=n_ord, formulation="laurent")
    for k in range(ns):
        sh = int(round(WALK_PX * (k + 0.5) / ns))
        st.add_layer(D / ns,
                     eps_tensor_cell=SP.binary_cell(NPIX, DUTY, ER, EG, sh))
    st.set_source(WL, theta=THETA, phi=0.0)
    o, R, T, J = st.solve()
    return _n0(o, R, T), time.time() - t0


def clos(x):
    _m, R, T = x
    return float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0)))


if __name__ == "__main__":
    o1, R1, T1 = oracle1d()
    print(f"ARBITER: y-uniform slanted grating, slant "
          f"{np.degrees(PHI_SLANT):.2f} deg, theta={THETA}")
    print("  oracle = shipped pmm_jones_1d_slanted (degree 20), "
          "independent of both 2-D routes\n")
    for n_ord in (7, 11, 15):
        met, tm = metric(n_ord)
        dm = cmp_o(*met, o1, R1, T1)
        print(f"  --- n_orders={n_ord} ---")
        print(f"      METRIC layer      : vs 1-D oracle = {dm:.3e}   "
              f"{tm:6.2f}s   |R+T-1| = {clos(met):.2e}")
        for ns in (1, 2, 4, 8, 16, 32):
            if WALK_PX % ns:
                continue
            stx, ts = stair(ns, n_ord)
            ds = cmp_o(*stx, o1, R1, T1)
            dsm = max(float(np.max(np.abs(stx[1] - met[1]))),
                      float(np.max(np.abs(stx[2] - met[2]))))
            print(f"      staircase ns={ns:>2}   : vs 1-D oracle = {ds:.3e}   "
                  f"{ts:6.2f}s   |R+T-1| = {clos(stx):.2e}   "
                  f"(vs metric {dsm:.2e})")
        print()

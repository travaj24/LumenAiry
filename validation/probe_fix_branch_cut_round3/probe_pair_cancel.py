"""ROUND 3 mechanism: at NEAR-NORMAL incidence the hybrid's propagating modes
come in nearly-degenerate +/-m PAIRS.  ``eig`` splits such a pair CONJUGATELY
(lam^2 = -s +/- i eta), so the principal roots carry real parts +a and +a with
IMAGINARY parts of OPPOSITE sign -- one of the pair is flipped, one is not.

  ``-r``      -> the pair's real parts are  +a, -a   : the backward error is
                 ANTISYMMETRIC across the pair and CANCELS in any symmetric
                 functional (sum R, sum T).
  ``conj(r)`` -> the pair's real parts are  +a, +a   : RECTIFIED.  The error no
                 longer cancels, it accumulates -- and ``a`` jitters with theta
                 because it IS the eigensolver's backward error.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from lumenairy.elements.rcwa import _core as C

ROWS = []
_ORIG = C._sqrt_decay


def _spy(x, xp=None, band=C._CUT_BAND_REL):
    r = np.sqrt(np.asarray(x).astype(C._C))
    if r.size > 4:
        scale = max(float(np.max(np.abs(r))), 1.0)
        flip = (np.abs(r.real) <= band * scale) & (r.imag < 0)
        if flip.any():
            ROWS.append((r.copy(), flip.copy(), scale))
    return _ORIG(x, xp, band)


C._sqrt_decay = _spy
import lumenairy.elements.pmm.twod as T
T._sqrt_decay = _spy
from lumenairy.elements.pmm import pmm_efficiency_2d

P, WL, DEP = 0.6e-6, 0.55e-6, 0.25e-6
XB = (0.2 * P, 0.6 * P)
pmm_efficiency_2d(P, P, 6.0 + 0j, 1.0, XB, XB, 1.5, 1.0, DEP, WL,
                  theta=1e-6, degree=5, n_orders=2, polarization="te")

print(f"arrays with >=1 on-cut flip: {len(ROWS)}")
for r, flip, scale in ROWS[:4]:
    idx = np.where(flip)[0]
    print(f"\n  array N={r.size}  scale={scale:.4f}  flipped={len(idx)}")
    for i in idx:
        # nearest partner by |Im| (the +/-m twin)
        d = np.abs(np.abs(r.imag) - abs(r.imag[i]))
        d[i] = np.inf
        j = int(np.argmin(d))
        conj_sum = r.real[i] + r.real[j]                 # conj rule: both +a
        neg_sum = -r.real[i] + r.real[j] if not flip[j] \
            else -r.real[i] - r.real[j]                  # -r rule
        print(f"    m={i:3d} r={r[i]: .6e}   partner m={j:3d} "
              f"r={r[j]: .6e}  |dIm|={d[j]:.3e}")
        print(f"        pair sum Re(lam):  conj rule {conj_sum: .6e}   "
              f"-r rule {neg_sum: .6e}")

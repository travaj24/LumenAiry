"""H9 -- the exact quantities the hardened tests assert."""
import os
import sys
import warnings

import numpy as np

import lumenairy

_ROOT = os.path.abspath(sys.argv[1]).replace("\\", "/").lower()
assert os.path.abspath(lumenairy.__file__).replace("\\", "/").lower().startswith(_ROOT)

from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402
from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments  # noqa: E402

PX, WL, DEPTH = 0.7e-6, 1.0e-6, 0.5e-6
LADDER = range(11, 42, 2)


def _rot(phi, no, ne):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


def _stripe(er, eg, duty=0.5, Sx=64, Sy=8):
    xm = (np.arange(Sx) + 0.5) / Sx < duty
    c = np.zeros((Sx, Sy, 3, 3), complex)
    for ix in range(Sx):
        c[ix, :] = er if xm[ix] else eg
    return c


ER = _rot(np.deg2rad(35.0), 1.5, 2.3)


def worst_best(g):
    eg = np.diag([g] * 3).astype(complex)
    v = []
    for n in LADDER:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R1, T1, _J = rcwa_jones_1d_segments(
                PX, [(0.5, ER), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0,
                n_orders=n)
        v.append(abs(float(np.sum(R1) + np.sum(T1) - 2.0)))
    return max(v), min(v), sum(1 for x in v if x < 1e-9), len(v)


print(f"numpy {np.__version__} OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")
for g, tag in ((2.10, "clean 2.10"), (2.25, "coincident 2.25")):
    w, b, ns, tot = worst_best(g)
    print(f"  {tag:<18} worst {w:.3e}  best {b:.3e}  sound {ns}/{tot}")

eg = np.diag([2.10] * 3).astype(complex)
cell = _stripe(ER, eg)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _o, R1, T1, J1 = rcwa_jones_1d_segments(
        PX, [(0.5, ER), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=81)
    _o, Rf, Tf, Jf = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                   n_orders_x=11, n_orders_y=1,
                                   formulation="fff_nv", symmetry=False)
    _o, Rl, Tl, Jl = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                   n_orders_x=11, n_orders_y=1,
                                   formulation="laurent", symmetry=False)
ef = abs(float(np.sum(Rf) - np.sum(R1)))
el = abs(float(np.sum(Rl) - np.sum(R1)))
jf = float(np.max(np.abs(Jf - J1)))
jl = float(np.max(np.abs(Jl - J1)))
print(f"  reference n=81: closure {float(np.sum(R1)+np.sum(T1)-2.0):.3e}  "
      f"sumR {float(np.sum(R1)):.12f}")
print(f"  fff_nv closure {abs(float(np.sum(Rf)+np.sum(Tf)-2.0)):.3e}")
print(f"  ef {ef:.4e}  el {el:.4e}  ef/el {ef/el:.4f}")
print(f"  jf {jf:.4e}  jl {jl:.4e}  jf/jl {jf/jl:.4f}")

# the same two ratios on the COINCIDENT fixture, same reference order
egd = np.diag([2.25] * 3).astype(complex)
celld = _stripe(ER, egd)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _o, R1d, T1d, J1d = rcwa_jones_1d_segments(
        PX, [(0.5, ER), (0.5, egd)], 1.5, 1.0, DEPTH, WL, theta=0.0,
        n_orders=81)
    _o, Rfd, Tfd, Jfd = rcwa_jones_2d(PX, PX, celld, 1.5, 1.0, DEPTH, WL,
                                      n_orders_x=11, n_orders_y=1,
                                      formulation="fff_nv", symmetry=False)
    _o, Rld, Tld, Jld = rcwa_jones_2d(PX, PX, celld, 1.5, 1.0, DEPTH, WL,
                                      n_orders_x=11, n_orders_y=1,
                                      formulation="laurent", symmetry=False)
efd = abs(float(np.sum(Rfd) - np.sum(R1d)))
eld = abs(float(np.sum(Rld) - np.sum(R1d)))
jfd = float(np.max(np.abs(Jfd - J1d)))
jld = float(np.max(np.abs(Jld - J1d)))
print(f"  COINCIDENT @ n_ref=81: ref closure "
      f"{float(np.sum(R1d)+np.sum(T1d)-2.0):.3e}  fff_nv closure "
      f"{abs(float(np.sum(Rfd)+np.sum(Tfd)-2.0)):.3e}")
print(f"  COINCIDENT ef/el {efd/eld:.4f}   jf/jl {jfd/jld:.4f}")

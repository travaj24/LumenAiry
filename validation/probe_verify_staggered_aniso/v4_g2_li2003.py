"""V4 -- G2 reconciliation against the ORIGINAL published oracle, Li 2003.

Granet 2023 Sec. 4.B says his Tables 2/3 "come from an in-house code and
correspond perfectly with those reported by Li [1]" -- i.e.
    L. Li, J. Opt. A 5, 345 (2003), "Fourier modal method for crossed
    anisotropic gratings with arbitrary permittivity and permeability
    tensors", Example 1 (p. 352) and Table 1 (p. 353).

Li's Example 1 states the geometry WITHOUT the two ambiguities that defeated
the build's G2 attempt:

    zeta = Theta = Phi = 0,  d1 = 2.4 lam0,  d2 = 1.4 lam0,  h = lam0,
    w1/d1 = w2/d2 = 0.5,
    n^(+1) = 1.0,   n^(-1) = 1.0 + i5.0        <-- a refractive INDEX
    eps_a = 2.25(xx + yy) + i0.5(xy - yx) + 2.0 zz
    eps_b = 2.25(xx + yy) - i0.5(xy - yx) + 2.0 zz
    theta = phi = 0, incident polarization in the Oxz plane (E along x)

and Fig. 3 of Li ("Convergence of the REFLECTED (0,0) order efficiency for
the grating in example 1", converging to 0.2980) pins Table 1 as the
REFLECTED orders.  Granet's running text says "transmitted"; Li's own figure
caption says reflected, and the listed order set (columns m = 0, +1, +2; rows
n = -1, 0, +1) is EXACTLY the propagating set of the VACUUM superstrate at
d1 = 2.4 lam, d2 = 1.4 lam -- which the lossy substrate's set is not.

Two consequences the build's sweep could not have hit:
  * 1 - i5 is Granet's mis-transcription of Li's INDEX 1 + i5 (conjugated into
    Granet's exp(+iwt)); the substrate permittivity is n^2 = -24 + 10i.
  * the oracle is the REFLECTED set into VACUUM, so the "efficiency
    definition into a lossy substrate" question -- the build's whole
    definition sweep A/B/C/D/E -- does not arise at all.

Li's Table 1 (first row of each cell = the grating as defined; column = the
x order m, row = the y order n):

    (0, 0)  0.2980     (+1, 0)  0.1195     (+2, 0)  0.0222
    (0,-1)  0.0619     (+1,-1)  0.0269     (+1,+1)  0.0137

with space-reversal symmetry (m, n) -> (-m, -n).  Granet's Table 2 quotes the
same four numbers he calls (1,1) 0.0268, (-1,1) 0.0139, (0,-1) 0.0620,
(0,0) 0.2979 -- i.e. Li's (+1,-1), (+1,+1), (0,-1), (0,0).
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

_ROOT = os.path.abspath("C:/tmp/lum_aniso")
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__

from lumenairy.elements.pmm import pmm_jones_2d  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)

LAM = 1.0e-6
DX, DY, H = 2.4 * LAM, 1.4 * LAM, 1.0 * LAM
N_SUP = 1.0
N_SUB = 1.0 + 5.0j                       # Li's n^(-1); eps = -24 + 10i

EPS_B = np.array([[2.25, -0.5j, 0.0], [0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                 dtype=complex)          # the PILLAR (Li fig. 3 inset)
EPS_A = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                 dtype=complex)          # the surround

#: Li 2003 table 1, reflected orders, first row of each cell (m, n).
LI = {(0, 0): 0.2980, (1, 0): 0.1195, (2, 0): 0.0222,
      (0, -1): 0.0619, (1, -1): 0.0269, (1, 1): 0.0137}
LI_SWAPPED = dict(LI)                     # eps_a <-> eps_b (Li's second row)
LI_SWAPPED[(1, -1)], LI_SWAPPED[(1, 1)] = 0.0137, 0.0269


def cell(pillar, host, n=2):
    """(n, n, 3, 3) cell with the pillar filling the first half of each axis
    (w1/d1 = w2/d2 = 0.5).  Position is immaterial (the staggered basis is
    position-invariant); this is the n = 2 realization of a 0.5/0.5 fill."""
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = host
    c[:n // 2, :n // 2] = pillar
    return c


def run(pillar, host, M, n_sub=N_SUB, dx=DX, dy=DY, nseg=2):
    o, R, T, _J = pmm_jones_2d_staggered(dx, dy, cell(pillar, host, nseg),
                                         n_sub, N_SUP, H, LAM, degree=M,
                                         n_orders=4)
    i = {(int(a), int(b)): k for k, (a, b) in enumerate(np.asarray(o))}
    return o, R, T, i


def score(R, i, table):
    """max |measured - Li| over the six published reflected orders, and the
    full per-order record (both signs of each pair)."""
    rec, dev = {}, 0.0
    for (m, n), ref in table.items():
        got = float(R[0][i[(m, n)]])
        mir = float(R[0][i[(-m, -n)]])
        rec[f"({m},{n})"] = dict(ref=ref, got=got, mirror=mir,
                                 d=abs(got - ref))
        dev = max(dev, abs(got - ref))
    return dev, rec


OUT = {}

# ---- the primary reading, convergence in M ------------------------------
for M in (5, 6, 7, 8):
    o, R, T, i = run(EPS_B, EPS_A, M)
    dev, rec = score(R, i, LI)
    OUT[f"primary_M{M}"] = dict(maxdev=dev, orders=rec,
                                sumR=float(R[0].sum()),
                                sumT=float(T[0].sum()),
                                sumRT=float(R[0].sum() + T[0].sum()))

# ---- the pillar/host swap = Li's second row ------------------------------
o, R, T, i = run(EPS_A, EPS_B, 7)
dev, rec = score(R, i, LI_SWAPPED)
OUT["swapped_M7_vs_LI_second_row"] = dict(maxdev=dev, orders=rec)
dev2, _ = score(R, i, LI)
OUT["swapped_M7_vs_LI_first_row"] = dict(maxdev=dev2)

# ---- position invariance: a (4,4) centred pillar -------------------------
c4 = np.empty((4, 4, 3, 3), dtype=complex)
c4[:] = EPS_A
c4[1:3, 1:3] = EPS_B
o, R, T, _J = pmm_jones_2d_staggered(DX, DY, c4, N_SUB, N_SUP, H, LAM,
                                     degree=6, n_orders=4)
i = {(int(a), int(b)): k for k, (a, b) in enumerate(np.asarray(o))}
dev, rec = score(R, i, LI)
OUT["centred_44_M6"] = dict(maxdev=dev, orders=rec)

# ---- INDEPENDENT ENGINE: the hybrid on the identical reading --------------
import warnings  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    oh, Rh, Th, _Jh = pmm_jones_2d(DX, DY, cell(EPS_B, EPS_A), N_SUB, N_SUP,
                                   H, LAM, degree=11, n_orders=13)
ih = {(int(a), int(b)): k for k, (a, b) in enumerate(np.asarray(oh))}
devh, rech = score(Rh, ih, LI)
OUT["hybrid_d11_n13"] = dict(maxdev=devh, orders=rech)

# ---- controls: the readings the build tried, for the record ---------------
o, R, T, i = run(EPS_B, EPS_A, 7, n_sub=np.sqrt(1.0 + 5.0j))
OUT["control_eps_sub_1plus5i_M7"] = dict(maxdev=score(R, i, LI)[0])
o, R, T, i = run(EPS_B, EPS_A, 7, n_sub=1.0)
OUT["control_air_substrate_M7"] = dict(maxdev=score(R, i, LI)[0],
                                       sumRT=float(R[0].sum() + T[0].sum()))
o, R, T, i = run(EPS_B, EPS_A, 7, dx=DY, dy=DX)
OUT["control_axes_swapped_M7"] = dict(
    maxdev=max(abs(float(R[0][i[(n, m)]]) - v) for (m, n), v in LI.items()))

fn = "C:/tmp/lum_aniso/validation/probe_verify_staggered_aniso/out_v4_g2.json"
json.dump(OUT, open(fn, "w"), indent=1, default=float)
for k, v in OUT.items():
    print(f"{k:34s} maxdev = {v.get('maxdev', float('nan')):.4e}  "
          f"{'' if 'sumRT' not in v else 'sumR+T=%.6f' % v['sumRT']}")
print()
print(json.dumps(OUT["primary_M7"], indent=1))
print("written", fn)

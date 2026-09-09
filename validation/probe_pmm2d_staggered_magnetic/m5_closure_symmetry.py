"""Probe 5 (G5, G6) -- lossless closure with a Hermitian permeability, the
lossy-mu deficit, the tripwire's magnetic predicate, and the x<->y transpose
symmetry with the mu blocks swapped.

G5.  A HERMITIAN mu absorbs nothing (the Poynting dissipation term carries the
anti-Hermitian parts of BOTH eps and mu), so ``sum R + sum T = 1`` is exact for
a Hermitian (eps, mu) pair between lossless half-spaces; an anti-Hermitian mu
part must close BELOW 1 by a visible margin.  Both arms are measured here, at
two modal counts, so the bar can be derived and the claim is two-sided.

G6.  Transposing the cell about x = y -- swap the grid axes, the periods, AND
the tensor components (e11<->e22, e12<->e21, and the SAME on mu) -- maps order
(m, n) to (n, m) exactly.  A mu whose off-diagonal is ANTI-symmetric
(m12 = -m21) is the discriminator: interchanging m12 and m21 in one arm must
break the symmetry.
"""
import warnings

import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_mag"), \
    lumenairy.__file__

from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

P, WL, DEP = 0.70e-6, 0.55e-6, 0.28e-6
LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
MU_LC = uniaxial_tensor(1.05, 1.30, np.pi / 2, phi=-0.40)
#: HERMITIAN gyrotropic permeability (m12 = -m21 = +0.4i): lossless
MU_GYRO = np.array([[1.6, 0.4j, 0.0], [-0.4j, 1.6, 0.0], [0.0, 0.0, 1.2]],
                   dtype=complex)
#: LOSSY: Im(m11) > 0 in the PUBLIC gauge -- an anti-Hermitian part
MU_LOSSY = np.array([[1.6 + 0.25j, 0.0, 0.0], [0.0, 1.6, 0.0],
                     [0.0, 0.0, 1.2]], dtype=complex)
EYE = np.eye(3, dtype=complex)
S3 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
S2 = np.array([[0, 1], [1, 0]], dtype=float)


def cell(host, incl, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = host
    c[0, 0] = incl
    return c


def transpose_cell(A):
    B = np.empty_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i, j] = S3 @ A[j, i] @ S3
    return B


def closure(ec, mc, M, theta=0.0, phi=0.0):
    o, R, T, J = pmm_jones_2d_staggered(P, P, ec, 1.0, 1.0, DEP, WL,
                                        mu_cell=mc, degree=M, n_orders=3,
                                        theta=theta, phi=phi)
    return [float(R[r].sum() + T[r].sum()) for r in (0, 1)]


def transpose_residual(ea, ma, eb, mb, M=6):
    kw = dict(n_substrate=1.5, n_superstrate=1.0, depth=DEP, wavelength=WL,
              degree=M, n_orders=3)
    oA, RA, TA, JA = pmm_jones_2d_staggered(P, 1.3 * P, ea, mu_cell=ma, **kw)
    oB, RB, TB, JB = pmm_jones_2d_staggered(1.3 * P, P, eb, mu_cell=mb, **kw)
    ia = {(int(a), int(b)): i for i, (a, b) in enumerate(np.asarray(oA))}
    ib = {(int(a), int(b)): i for i, (a, b) in enumerate(np.asarray(oB))}
    dev = max(max(abs(RB[r][ib[(m, n)]] - RA[1 - r][ia[(n, m)]]),
                  abs(TB[r][ib[(m, n)]] - TA[1 - r][ia[(n, m)]]))
              for (m, n) in ib if (n, m) in ia for r in (0, 1))
    return float(dev), float(np.max(np.abs(JB - S2 @ JA @ S2)))


if __name__ == "__main__":
    print("=== G5a  LOSSLESS closure |sum R + sum T - 1| (Hermitian eps, mu)")
    for tag, ec, mc in (
            ("uniform LC eps, gyrotropic mu",
             np.broadcast_to(LC, (2, 2, 3, 3)).copy(),
             np.broadcast_to(MU_GYRO, (2, 2, 3, 3)).copy()),
            ("patterned LC/iso eps, gyro/1.2 mu",
             cell(LC, 4.0 * EYE), cell(MU_GYRO, 1.2 * EYE)),
            ("patterned eps, LC-like mu", cell(LC, 4.0 * EYE),
             cell(MU_LC, EYE)),
            ("vacuum eps, patterned mu", cell(EYE, EYE),
             cell(MU_GYRO, 1.2 * EYE))):
        for M in (6, 8):
            for th, ph in ((0.0, 0.0), (0.25, 0.6)):
                c = closure(ec, mc, M, th, ph)
                print(f"  {tag:<36s} M={M} th={th} ph={ph}  "
                      f"dev={max(abs(v - 1.0) for v in c):.3e}")
    print()
    print("=== G5b  LOSSY mu must close BELOW 1 by a margin")
    for M in (6, 8):
        c = closure(cell(LC, 4.0 * EYE), cell(MU_LOSSY, 1.2 * EYE), M)
        print(f"  Im(m11) = +0.25  M={M}  sum R+T = {c[0]:.6f}, {c[1]:.6f}")
    print()
    print("=== G5c  the tripwire: Hermitian mu = LOSSLESS (must NOT fire), "
          "under-resolved must fire")
    for M, tag in ((8, "resolved"), (3, "under-resolved M=3")):
        st = PMM2DStackPure(P, P, n_modes=M, n_orders=3)
        st.add_layer(DEP, eps_cell=cell(LC, 4.0 * EYE),
                     mu_cell=cell(MU_GYRO, 1.2 * EYE))
        st.set_source(WL, theta=0.25, phi=0.6)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            o, R, T, J = st.solve()
        fired = [x for x in w if "energy closure" in str(x.message)]
        print(f"  Hermitian mu, {tag:<20s}: warnings={len(fired)}  "
              f"sum R+T = {R[0].sum() + T[0].sum():.6f}")
    st = PMM2DStackPure(P, P, n_modes=8, n_orders=3)
    st.add_layer(DEP, eps_cell=cell(LC, 4.0 * EYE),
                 mu_cell=cell(MU_LOSSY, 1.2 * EYE))
    st.set_source(WL, theta=0.25, phi=0.6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        st.solve()
    print(f"  LOSSY mu (no unity claim exists): warnings="
          f"{len([x for x in w if 'energy closure' in str(x.message)])}")
    print()
    print("=== G6  x<->y transpose with the mu blocks swapped")
    ea, ma = cell(LC, 4.0 * EYE), cell(MU_GYRO, 1.2 * EYE)
    dev, dJ = transpose_residual(ea, ma, transpose_cell(ea),
                                 transpose_cell(ma))
    print(f"  correct placement            dev={dev:.3e}  dJones={dJ:.3e}")
    maw = ma.copy()
    maw[..., 0, 1], maw[..., 1, 0] = ma[..., 1, 0].copy(), ma[..., 0, 1].copy()
    devb, dJb = transpose_residual(ea, maw, transpose_cell(ea),
                                   transpose_cell(ma))
    print(f"  m12/m21 SWAPPED in one arm   dev={devb:.3e}  dJones={dJb:.3e}")
    eaw = ea.copy()
    eaw[..., 0, 1], eaw[..., 1, 0] = ea[..., 1, 0].copy(), ea[..., 0, 1].copy()
    devc, dJc = transpose_residual(eaw, ma, transpose_cell(ea),
                                   transpose_cell(ma))
    print(f"  (eps e12/e21 swapped control) dev={devc:.3e}  dJones={dJc:.3e}")

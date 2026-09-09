"""Probe 10 -- G6 closure, G7 discrete symmetries, G8 stack, G9 absorption."""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

WL = 0.55e-6
P = 0.70e-6
DEP = 0.28e-6
LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
LCm = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=-0.55)
GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                dtype=complex)
ISO = 4.0 * np.eye(3, dtype=complex)
S3 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
S2 = np.array([[0, 1], [1, 0]], dtype=float)
M2 = np.diag([1.0, -1.0])


def cell(host, pill, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = host
    c[0, 0] = pill
    return c


def idx(o):
    return {(int(a), int(b)): i for i, (a, b) in enumerate(np.asarray(o))}


def g6():
    print("=== G6 closure two-sided ===")
    for name, c in (("LC host + iso pillar", cell(LC, ISO)),
                    ("gyrotropic host + iso pillar", cell(GYRO, ISO)),
                    ("LC host + LOSSY pillar",
                     cell(LC, ISO + 0.8j * np.eye(3)))):
        for M in (6, 8):
            o, R, T, _J = pmm_jones_2d_staggered(P, P, c, 1.5, 1.0, DEP, WL,
                                                 degree=M, n_orders=4)
            tot = R.sum(axis=1) + T.sum(axis=1)
            print(f"  {name:30s} M={M}  R+T = {tot[0]:.12f} {tot[1]:.12f}"
                  f"  |dev| = {np.max(np.abs(tot - 1)):.3e}")


def g7():
    print("=== G7 discrete symmetries ===")
    A = cell(LC, ISO)
    B = np.empty_like(A)
    for i in range(2):
        for j in range(2):
            B[i, j] = S3 @ A[j, i] @ S3
    kw = dict(n_substrate=1.5, n_superstrate=1.0, depth=DEP, wavelength=WL,
              degree=7, n_orders=4)
    oA, RA, TA, JA = pmm_jones_2d_staggered(P, 1.3 * P, A, **kw)
    oB, RB, TB, JB = pmm_jones_2d_staggered(1.3 * P, P, B, **kw)
    ia, ib = idx(oA), idx(oB)
    dev = max(abs(RB[r][ib[(m, n)]] - RA[1 - r][ia[(n, m)]])
              for (m, n) in ib if (n, m) in ia for r in (0, 1))
    devT = max(abs(TB[r][ib[(m, n)]] - TA[1 - r][ia[(n, m)]])
               for (m, n) in ib if (n, m) in ia for r in (0, 1))
    print(f"  (a) x<->y transpose: dR={dev:.3e} dT={devT:.3e} "
          f"dJones={np.max(np.abs(JB - S2 @ JA @ S2)):.3e}")
    # (b) mirror y -> -y with director phi -> -phi
    C = cell(LC, ISO)
    D = np.empty_like(C)
    Mm = np.diag([1.0, -1.0, 1.0])
    for i in range(2):
        for j in range(2):
            D[i, j] = Mm @ C[i, 1 - j] @ Mm
    oC, RC, TC, JC = pmm_jones_2d_staggered(P, P, C, **kw)
    oD, RD, TD, JD = pmm_jones_2d_staggered(P, P, D, **kw)
    ic, idd = idx(oC), idx(oD)
    dev = max(abs(RD[r][idd[(m, n)]] - RC[r][ic[(m, -n)]])
              for (m, n) in idd if (m, -n) in ic for r in (0, 1))
    print(f"  (b) mirror y: dR={dev:.3e} "
          f"dJones={np.max(np.abs(JD - M2 @ JC @ M2)):.3e}")
    # a WRONG placement (swap e12/e21 in the cell) must BREAK (a)
    Aw = A.copy()
    Aw[..., 0, 1], Aw[..., 1, 0] = A[..., 1, 0].copy(), A[..., 0, 1].copy()
    oW, RW, TW, JW = pmm_jones_2d_staggered(P, 1.3 * P, Aw, **kw)
    iw = idx(oW)
    devw = max(abs(RB[r][ib[(m, n)]] - RW[1 - r][iw[(n, m)]])
               for (m, n) in ib if (n, m) in iw for r in (0, 1))
    print(f"      control (e12<->e21 swapped in one arm): dR={devw:.3e}")
    # the LC cell has SYMMETRIC e12 = e21, so the swap is a no-op there;
    # repeat with the GYROTROPIC (antisymmetric) tensor, where it is not.
    Ag = cell(GYRO, ISO)
    Bg = np.empty_like(Ag)
    for i in range(2):
        for j in range(2):
            Bg[i, j] = S3 @ Ag[j, i] @ S3
    Agw = Ag.copy()
    Agw[..., 0, 1], Agw[..., 1, 0] = (Ag[..., 1, 0].copy(),
                                      Ag[..., 0, 1].copy())
    og, Rg, Tg, Jg = pmm_jones_2d_staggered(1.3 * P, P, Bg, **kw)
    ogw, Rgw, Tgw, Jgw = pmm_jones_2d_staggered(P, 1.3 * P, Agw, **kw)
    oga, Rga, Tga, Jga = pmm_jones_2d_staggered(P, 1.3 * P, Ag, **kw)
    ig, iga, igw = idx(og), idx(oga), idx(ogw)
    d_ok = max(abs(Rg[r][ig[(m, n)]] - Rga[1 - r][iga[(n, m)]])
               for (m, n) in ig if (n, m) in iga for r in (0, 1))
    d_bad = max(abs(Rg[r][ig[(m, n)]] - Rgw[1 - r][igw[(n, m)]])
                for (m, n) in ig if (n, m) in igw for r in (0, 1))
    print(f"  (c) gyrotropic transpose: correct dR={d_ok:.3e}   "
          f"e12<->e21-swapped dR={d_bad:.3e}")


def g8():
    print("=== G8 stack ===")
    # (a) A(t) + A(t) == A(2t)
    c = cell(LC, ISO)
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.5, n_modes=7,
                        n_orders=4)
    st.add_layer(DEP, eps_cell=c).add_layer(DEP, eps_cell=c)
    st.set_source(WL, theta=0.15, phi=0.4)
    o2, R2, T2, J2 = st.solve()
    o1, R1, T1, J1 = pmm_jones_2d_staggered(P, P, c, 1.5, 1.0, 2 * DEP, WL,
                                            degree=7, n_orders=4, theta=0.15,
                                            phi=0.4)
    print(f"  (a) split-vs-single: dR={np.max(np.abs(R2 - R1)):.3e} "
          f"dT={np.max(np.abs(T2 - T1)):.3e} "
          f"dJ={np.max(np.abs(J2 - J1)):.3e}")
    # (b) all-uniform anisotropic multilayer vs Berreman, oblique conical
    layers = [(LC, 0.21e-6), (GYRO, 0.13e-6), (LCm, 0.17e-6)]
    for M in (5, 7):
        stk = PMM2DStackPure(0.4e-6, 0.4e-6, n_superstrate=1.0,
                             n_substrate=1.5, n_modes=M, n_orders=2)
        for t33, th in layers:
            stk.add_layer(th, eps=t33)
        stk.set_source(1.0e-6, theta=25 * np.pi / 180, phi=40 * np.pi / 180)
        o, R, T, J = stk.solve()
        Rb, Tb, jr, _ = berreman_jones_1d(
            [(t33, th) for t33, th in layers], 1.5, 1.0, 1.0e-6,
            angle=25 * np.pi / 180, phi=40 * np.pi / 180)
        print(f"  (b) uniform tensor multilayer M={M}: "
              f"dR={np.max(np.abs(R.sum(axis=1) - Rb)):.3e} "
              f"dT={np.max(np.abs(T.sum(axis=1) - Tb)):.3e} "
              f"dJ={np.max(np.abs(J - jr)):.3e}")


def g9():
    print("=== G9 absorption budget ===")
    lossy = cell(LC, ISO + 0.8j * np.eye(3))
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.5, n_modes=7,
                        n_orders=4)
    st.add_layer(0.12e-6, eps_cell=cell(LC, ISO))
    st.add_layer(DEP, eps_cell=lossy)
    st.add_layer(0.09e-6, eps=GYRO)
    st.set_source(WL, theta=0.12, phi=0.3)
    o, R, T, J = st.solve(retain_internal=True)
    A = st.layer_absorption()
    for col in (0, 1):
        lhs = float(A[:, col].sum())
        rhs = float(1.0 - R[col].sum() - T[col].sum())
        print(f"  pol {col}: sum A = {lhs:.10f}  1-R-T = {rhs:.10f}  "
              f"dev = {abs(lhs - rhs):.3e}   per-layer A = "
              f"{np.round(A[:, col], 6)}")


g6()
g7()
g8()
g9()

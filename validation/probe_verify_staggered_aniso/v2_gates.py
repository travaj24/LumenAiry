"""V2 -- independent RE-MEASUREMENT of gates G1, G3..G9 on FRESH fixtures.

Nothing here re-uses the build's geometries, tensors, angles or modal counts:
the point is to reproduce the ORDERS OF MAGNITUDE of the build doc's tables
from a different corner of the parameter space, so that a bar tuned to one
fixture shows up as a durability defect.

    python v2_gates.py <gate> [<gate> ...]      # gates: g1 g3 g4 g5 g6 g7 g8 g9
"""
import json
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

_ROOT = os.path.abspath("C:/tmp/lum_aniso")
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm import (  # noqa: E402
    PMM2DStackPure,
    pmm_jones_1d,
    pmm_jones_2d,
)
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import rcwa_jones_1d, rcwa_jones_2d  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

# ---------------------------------------------------------------- fixtures
LC = uniaxial_tensor(1.45, 1.75, np.pi / 2, phi=0.90)          # e12 = e21 real
LCM = uniaxial_tensor(1.45, 1.75, np.pi / 2, phi=-0.90)
GY = np.array([[2.60, 0.35j, 0.0], [-0.35j, 2.60, 0.0],
               [0.0, 0.0, 2.40]], dtype=complex)               # e12 = -e21
ISO = 5.0 * np.eye(3, dtype=complex)


def cell(host, pillar, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = host
    c[0, 0] = pillar
    return c


def uni(t, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = t
    return c


def promote(m):
    t = np.zeros(np.shape(m) + (3, 3), dtype=complex)
    for i in range(3):
        t[..., i, i] = m
    return t


def idx(o):
    return {(int(a), int(b)): i for i, (a, b) in enumerate(np.asarray(o))}


def o0(o, A):
    i = int(np.where((np.asarray(o)[:, 0] == 0)
                     & (np.asarray(o)[:, 1] == 0))[0][0])
    return np.array([float(A[0][i]), float(A[1][i])])


RES = {}


# ============================================================== G1
def g1():
    out = {}
    fixtures = (
        ("cellA_22_M7", np.array([[3.24, 1.0], [2.10, 7.29]], complex), 7,
         dict(alpha0x=-0.44, alpha0y=0.23, k0=2 * np.pi / 0.71)),
        ("cellB_33_M5", np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0],
                                  [8.0, 9.0, 1.5]], complex), 5,
         dict(alpha0x=0.05, alpha0y=-0.61, k0=2 * np.pi / 0.48)),
    )
    for tag, c, M, kw in fixtures:
        n = c.shape[0]
        a = Granet2DTransverseE(0.93, 0.93, n, n, M, c, **kw)
        b = Granet2DTransverseE(0.93, 0.93, n, n, M, promote(c), **kw)
        rec = {}
        for nm in ("Lmat", "Rmat", "Stt", "Schur"):
            rec[nm + "_maxdiff"] = float(
                np.max(np.abs(getattr(a, nm) - getattr(b, nm))))
            rec[nm + "_bitequal"] = bool(
                np.array_equal(getattr(a, nm), getattr(b, nm)))
        rec["Et0_bitequal"] = bool(np.array_equal(a.Et_blocks[0],
                                                  b.Et_blocks[0]))
        rec["Et1_bitequal"] = bool(np.array_equal(a.Et_blocks[1],
                                                  b.Et_blocks[1]))
        rec["scalar_Et_offdiag_is_None"] = a.Et_offdiag is None
        rec["tensor_offdiag_exact_zero"] = bool(
            all(np.array_equal(x, np.zeros_like(x)) for x in b.Et_offdiag))
        rec["dead_attrs_absent"] = all(
            not hasattr(a, d) and not hasattr(b, d)
            for d in ("Curl", "Kzt", "Ktz", "G3", "Meps33"))
        out[tag] = rec
    c = np.array([[3.24, 1.0], [2.10, 7.29]], complex)
    g = dict(period_x=0.95e-6, period_y=0.95e-6, depth=0.37e-6,
             wavelength=0.71e-6, n_substrate=1.45, n_superstrate=1.0)
    o1, R1, T1 = pmm_efficiency_2d_staggered(eps_cell=c, degree=7, n_orders=4,
                                             polarization="tm", **g)
    o2, R2, T2, J2 = pmm_jones_2d_staggered(eps_cell=promote(c), degree=7,
                                            n_orders=4, **g)
    o3, R3, T3, J3 = pmm_jones_2d_staggered(eps_cell=c, degree=7, n_orders=4,
                                            **g)
    out["public_tm_vs_tensor_eyeI"] = {
        "maxdR": float(np.max(np.abs(R1 - R2[0]))),
        "maxdT": float(np.max(np.abs(T1 - T2[0]))),
        "bitequal_R": bool(np.array_equal(R1, R2[0])),
        "bitequal_T": bool(np.array_equal(T1, T2[0])),
        "scalarmap_vs_tensor_bitequal": bool(np.array_equal(R3, R2)
                                             and np.array_equal(J3, J2)),
        "closure": [float(R2[r].sum() + T2[r].sum()) for r in (0, 1)],
    }
    return out


# ============================================================== G3
G3G = dict(period=0.33e-6, wl=0.90e-6, dep=0.62e-6, nsub=1.6, nsup=1.0)


def g3_res(t33, nseg, M, theta, phi):
    _o, R, T, J = pmm_jones_2d_staggered(
        G3G["period"], G3G["period"], uni(t33, nseg), G3G["nsub"], G3G["nsup"],
        G3G["dep"], G3G["wl"], degree=M, n_orders=2, theta=theta, phi=phi)
    Rb, Tb, jr, _jt = berreman_jones_1d([(t33, G3G["dep"])], G3G["nsub"],
                                        G3G["nsup"], G3G["wl"], angle=theta,
                                        phi=phi)
    dR = float(np.max(np.abs(R.sum(axis=1) - Rb)))
    dT = float(np.max(np.abs(T.sum(axis=1) - Tb)))
    dJ = float(np.max(np.abs(J - jr)))
    return dict(R=dR, T=dT, J=dJ, worst=max(dR, dT, dJ))


def g3():
    out = {}
    for name, t in (("lc", LC), ("gyro", GY)):
        for nseg in (2, 3):
            for th, ph in ((0.0, 0.0), (18 * np.pi / 180, 0.0),
                           (18 * np.pi / 180, 65 * np.pi / 180)):
                for M in (5, 7):
                    k = (f"{name}_n{nseg}_th{int(round(np.degrees(th)))}"
                         f"_ph{int(round(np.degrees(ph)))}_M{M}")
                    out[k] = g3_res(t, nseg, M, th, ph)
    return out


# ============================================================== G4
G4P, G4WL, G4DEP = 1.10e-6, 0.63e-6, 0.42e-6
G4R, G4G = LC, 1.90 * np.eye(3, dtype=complex)
G4TH = 0.35


def g4_stripe():
    c = np.empty((2, 2, 3, 3), dtype=complex)
    c[:] = G4G
    c[0, :] = G4R
    return c


def g4():
    o1, R1, T1, J1 = pmm_jones_1d(G4P, G4R, G4G, 1.45, 1.0, G4DEP, 0.5, G4WL,
                                  angle=G4TH, degree=16, stabilize=False)
    o1r, R1r, T1r, J1r = rcwa_jones_1d(G4P, G4R, G4G, 1.45, 1.0, G4DEP, 0.5,
                                       G4WL, angle=G4TH, n_orders=81)
    one = {int(m): i for i, m in enumerate(np.asarray(o1).ravel())}
    oner = {int(m): i for i, m in enumerate(np.asarray(o1r).ravel())}
    common01 = [m for m in one if m in oner]
    spread = max(max(abs(R1[r][one[m]] - R1r[r][oner[m]]),
                     abs(T1[r][one[m]] - T1r[r][oner[m]]))
                 for m in common01 for r in (0, 1))
    out = {"oracle_spread_RT": float(spread),
           "oracle_spread_J": float(np.max(np.abs(J1 - J1r)))}
    for M in (5, 6, 7, 8):
        o, R, T, J = pmm_jones_2d_staggered(G4P, G4P, g4_stripe(), 1.45, 1.0,
                                            G4DEP, G4WL, degree=M, n_orders=3,
                                            theta=G4TH, phi=0.0)
        sel = np.asarray(o)[:, 1] == 0
        two = {int(m): i for i, m in zip(np.arange(len(o))[sel],
                                         np.asarray(o)[sel, 0])}
        common = [m for m in one if m in two]
        dRT = max(max(abs(R[r][two[m]] - R1[r][one[m]]),
                      abs(T[r][two[m]] - T1[r][one[m]]))
                  for m in common for r in (0, 1))
        forb = float(max(np.max(np.abs(R[:, ~sel])),
                         np.max(np.abs(T[:, ~sel]))))
        out[f"M{M}"] = dict(
            dRT=float(dRT), dJ=float(np.max(np.abs(J - J1))),
            y_forbidden=forb,
            closure=float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1))))
    return out


# ============================================================== G5
G5P, G5WL, G5DEP, G5NS = 0.62e-6, 0.50e-6, 0.31e-6, 1.45


def g5_cell(up=1, reverse=False):
    n = 2 * up
    c = np.empty((n, n, 3, 3), dtype=complex)
    host, pill = (ISO, LC) if reverse else (LC, ISO)
    c[:] = host
    c[:up, :up] = pill
    return c


def g5():
    out = {}
    for rev in (False, True):
        tag = "lc_pillar" if rev else "iso_pillar"
        c = g5_cell(reverse=rev)
        o, R, T, J = pmm_jones_2d_staggered(G5P, G5P, c, G5NS, 1.0, G5DEP,
                                            G5WL, degree=7, n_orders=5)
        b0, b1, bJ = o0(o, R), o0(o, T), J
        rec = {"staggered_R00": b0.tolist(), "staggered_T00": b1.tolist(),
               "staggered_closure": float(
                   np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1)))}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for dg, nor in ((9, 9), (11, 13)):
                oh, Rh, Th, Jh = pmm_jones_2d(G5P, G5P, c, G5NS, 1.0, G5DEP,
                                              G5WL, degree=dg, n_orders=nor)
                rec[f"hybrid_d{dg}_n{nor}"] = dict(
                    dR=float(np.max(np.abs(o0(oh, Rh) - b0))),
                    dT=float(np.max(np.abs(o0(oh, Th) - b1))),
                    dJ=float(np.max(np.abs(Jh - bJ))),
                    closure=float(np.max(np.abs(Rh.sum(axis=1)
                                                + Th.sum(axis=1) - 1))))
        for nor in (9, 13):
            orc, Rr, Tr, Jr = rcwa_jones_2d(G5P, G5P, g5_cell(32, rev), G5NS,
                                            1.0, G5DEP, G5WL,
                                            n_orders_x=nor, n_orders_y=nor)
            rec[f"rcwa_n{nor}"] = dict(
                dR=float(np.max(np.abs(o0(orc, Rr) - b0))),
                dT=float(np.max(np.abs(o0(orc, Tr) - b1))),
                dJ=float(np.max(np.abs(Jr - bJ))),
                closure=float(np.max(np.abs(Rr.sum(axis=1)
                                            + Tr.sum(axis=1) - 1))))
        out[tag] = rec
    c = g5_cell()
    st, hy = {}, {}
    for nor in (4, 8):
        o, R, T, _J = pmm_jones_2d_staggered(G5P, G5P, c, G5NS, 1.0, G5DEP,
                                             G5WL, degree=7, n_orders=nor)
        st[nor] = np.concatenate([o0(o, R), o0(o, T)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oh, Rh, Th, _ = pmm_jones_2d(G5P, G5P, c, G5NS, 1.0, G5DEP, G5WL,
                                         degree=9, n_orders=nor)
        hy[nor] = np.concatenate([o0(oh, Rh), o0(oh, Th)])
    out["nofloor"] = dict(staggered=float(np.max(np.abs(st[4] - st[8]))),
                          hybrid=float(np.max(np.abs(hy[4] - hy[8]))))
    return out


# ============================================================== G6
def g6():
    kw = dict(period_x=G5P, period_y=G5P, n_substrate=G5NS, n_superstrate=1.0,
              depth=G5DEP, wavelength=G5WL, n_orders=4)
    out = {}
    for name, host in (("lc", LC), ("gyro", GY)):
        for M in (6, 8):
            _o, R, T, _J = pmm_jones_2d_staggered(eps_cell=cell(host, ISO),
                                                  degree=M, **kw)
            tot = R.sum(axis=1) + T.sum(axis=1)
            out[f"{name}_M{M}"] = dict(tot=tot.tolist(),
                                       dev=float(np.max(np.abs(tot - 1.0))))
    for M in (6, 8):
        lossy = cell(LC, ISO + 0.9j * np.eye(3))
        _o, R, T, _J = pmm_jones_2d_staggered(eps_cell=lossy, degree=M, **kw)
        tot = R.sum(axis=1) + T.sum(axis=1)
        out[f"lossy_M{M}"] = dict(tot=tot.tolist(),
                                  deficit=float(np.min(1.0 - tot)))
    return out


# ============================================================== G7
S3 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], float)
S2 = np.array([[0, 1], [1, 0]], float)
M2 = np.diag([1.0, -1.0])
G7KW = dict(n_substrate=G5NS, n_superstrate=1.0, depth=G5DEP,
            wavelength=G5WL, degree=6, n_orders=3)


def tcell(A):
    B = np.empty_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i, j] = S3 @ A[j, i] @ S3
    return B


def tres(A, B):
    oA, RA, TA, JA = pmm_jones_2d_staggered(G5P, 1.45 * G5P, A, **G7KW)
    oB, RB, TB, JB = pmm_jones_2d_staggered(1.45 * G5P, G5P, B, **G7KW)
    ia, ib = idx(oA), idx(oB)
    dev = max(max(abs(RB[r][ib[(m, n)]] - RA[1 - r][ia[(n, m)]]),
                  abs(TB[r][ib[(m, n)]] - TA[1 - r][ia[(n, m)]]))
              for (m, n) in ib if (n, m) in ia for r in (0, 1))
    return float(dev), float(np.max(np.abs(JB - S2 @ JA @ S2)))


def g7():
    out = {}
    for name, host in (("lc", LC), ("gyro", GY)):
        A = cell(host, ISO)
        d, dJ = tres(A, tcell(A))
        out[f"transpose_{name}"] = dict(perorder=d, jones=dJ)
        Aw = A.copy()
        Aw[..., 0, 1], Aw[..., 1, 0] = A[..., 1, 0].copy(), A[..., 0, 1].copy()
        d2, dJ2 = tres(Aw, tcell(A))
        out[f"swapped_control_{name}"] = dict(perorder=d2, jones=dJ2)
    C = cell(LC, ISO)
    Mm = np.diag([1.0, -1.0, 1.0])
    D = np.empty_like(C)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            D[i, j] = Mm @ C[i, C.shape[1] - 1 - j] @ Mm
    oC, RC, _TC, JC = pmm_jones_2d_staggered(G5P, G5P, C, **G7KW)
    oD, RD, _TD, JD = pmm_jones_2d_staggered(G5P, G5P, D, **G7KW)
    ic, ide = idx(oC), idx(oD)
    dev = max(abs(RD[r][ide[(m, n)]] - RC[r][ic[(m, -n)]])
              for (m, n) in ide if (m, -n) in ic for r in (0, 1))
    out["ymirror_lc"] = dict(perorder=float(dev),
                             jones=float(np.max(np.abs(JD - M2 @ JC @ M2))))
    return out


# ============================================================== G8
def g8():
    out = {}
    c = cell(LC, ISO)
    st = PMM2DStackPure(G5P, G5P, n_superstrate=1.0, n_substrate=G5NS,
                        n_modes=6, n_orders=3)
    st.add_layer(G5DEP, eps_cell=c).add_layer(G5DEP, eps_cell=c)
    st.set_source(G5WL, theta=0.27, phi=0.83)
    _o2, R2, T2, J2 = st.solve()
    _o1, R1, T1, J1 = pmm_jones_2d_staggered(G5P, G5P, c, G5NS, 1.0,
                                             2 * G5DEP, G5WL, degree=6,
                                             n_orders=3, theta=0.27, phi=0.83)
    out["split_vs_single"] = dict(dR=float(np.max(np.abs(R2 - R1))),
                                  dT=float(np.max(np.abs(T2 - T1))),
                                  dJ=float(np.max(np.abs(J2 - J1))))
    layers = [(LC, 0.19e-6), (GY, 0.11e-6), (LCM, 0.23e-6)]
    for M in (5, 7):
        stk = PMM2DStackPure(0.31e-6, 0.31e-6, n_superstrate=1.0,
                             n_substrate=1.6, n_modes=M, n_orders=2)
        for t33, th in layers:
            stk.add_layer(th, eps=t33)
        stk.set_source(0.90e-6, theta=18 * np.pi / 180, phi=65 * np.pi / 180)
        _o, R, T, J = stk.solve()
        Rb, Tb, jr, _jt = berreman_jones_1d(layers, 1.6, 1.0, 0.90e-6,
                                            angle=18 * np.pi / 180,
                                            phi=65 * np.pi / 180)
        out[f"berreman_multilayer_M{M}"] = float(max(
            np.max(np.abs(R.sum(axis=1) - Rb)),
            np.max(np.abs(T.sum(axis=1) - Tb)),
            np.max(np.abs(J - jr))))
    return out


# ============================================================== G9
def g9():
    out = {}
    for M in (6, 7, 8):
        st = PMM2DStackPure(G5P, G5P, n_superstrate=1.0, n_substrate=G5NS,
                            n_modes=M, n_orders=3)
        st.add_layer(0.10e-6, eps_cell=cell(LC, ISO))
        st.add_layer(G5DEP, eps_cell=cell(LC, ISO + 0.9j * np.eye(3)))
        st.add_layer(0.08e-6, eps=GY)
        st.set_source(G5WL, theta=0.19, phi=0.44)
        t0 = time.time()
        _o, R, T, _J = st.solve(retain_internal=True)
        A = st.layer_absorption()
        dev = [abs(float(A[:, c].sum()) - float(1.0 - R[c].sum() - T[c].sum()))
               for c in (0, 1)]
        out[f"M{M}"] = dict(dev=dev, A_lossy=A[1].tolist(),
                            A_lossless_max=float(np.max(np.abs(A[[0, 2]]))),
                            wall=round(time.time() - t0, 2),
                            shape=list(A.shape))
    return out


GATES = dict(g1=g1, g3=g3, g4=g4, g5=g5, g6=g6, g7=g7, g8=g8, g9=g9)

if __name__ == "__main__":
    want = sys.argv[1:] or list(GATES)
    for g in want:
        t0 = time.time()
        RES[g] = GATES[g]()
        RES[g]["_wall_s"] = round(time.time() - t0, 2)
        print("=== %s  (%s s) ===" % (g, RES[g]["_wall_s"]))
        print(json.dumps(RES[g], indent=1, default=float))
    fn = ("C:/tmp/lum_aniso/validation/probe_verify_staggered_aniso/out_v2_"
          + "_".join(want) + ".json")
    json.dump(RES, open(fn, "w"), indent=1, default=float)
    print("written", fn)

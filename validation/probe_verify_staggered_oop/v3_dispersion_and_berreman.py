"""V3 -- the assembled out-of-plane generator's DISPERSION against an exact
quartic written here from scratch, plus the Berreman gates and the fail-before
controls, all on fixtures the build did not use.

  A  the exact quartic ``det[eps + kk^T - |k|^2 I] = 0`` in ``q = kz/k0`` is
     built term by term from polynomial arithmetic (no library call), its four
     roots compared to the generator's eigenvalue set at ``+k_t`` and ``-k_t``,
     and its SUM-OF-ROOTS discriminator measured for an in-plane control, a
     symmetric out-of-plane tensor and a NON-RECIPROCAL one.  The rot-sign
     arm is walked too.
  B  a uniform out-of-plane slab vs ``berreman_jones_1d`` on an M ladder
     (two-sided: the residual must FALL, and the M=3 end must be visibly worse).
  C  the T9 fail-before controls -- drop / negate / TRANSPOSE the out-of-plane
     block -- on a symmetric AND on a NON-RECIPROCAL (``e13 != e31``, Hermitian)
     tensor, at normal / oblique / conical.  The transposition trap is the one
     no energy or eigenvalue check can see.
  D  the dispatch floor: a ``1e-16`` stray in an ``xz`` slot must leave the
     result BYTE-identical to the clean cell; a ``1e-3`` stray must not.

Usage:  PYTHONPATH=<root> python v3_dispersion_and_berreman.py <root> <out.json>
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

import lumenairy  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {ROOT}")

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import uniaxial_tensor  # noqa: E402

OUT = sys.argv[2]

PX = 0.91e-6
WL = 0.67e-6
DEP = 0.33e-6
NSUB = 1.55
NSUP = 1.0


# ------------------------------------------------- the exact quartic ------
def quartic_coeffs(eps, a, b):
    """Coefficients (highest power first) of ``det[eps + kk^T - |k|^2 I]`` as a
    polynomial in ``q = kz/k0``, for transverse ``(a, b) = (kx, ky)/k0``.

    Written out entry by entry with numpy polynomial arithmetic -- no library
    dispersion routine is consulted.  ``M = eps + k k^T - |k|^2 I`` follows
    from ``k x (k x E) + k0^2 eps E = 0``."""
    e = np.asarray(eps, dtype=complex)
    M = [[None] * 3 for _ in range(3)]
    M[0][0] = np.array([-1.0, 0.0, e[0, 0] - b * b], dtype=complex)
    M[0][1] = np.array([e[0, 1] + a * b], dtype=complex)
    M[0][2] = np.array([a, e[0, 2]], dtype=complex)
    M[1][0] = np.array([e[1, 0] + a * b], dtype=complex)
    M[1][1] = np.array([-1.0, 0.0, e[1, 1] - a * a], dtype=complex)
    M[1][2] = np.array([b, e[1, 2]], dtype=complex)
    M[2][0] = np.array([a, e[2, 0]], dtype=complex)
    M[2][1] = np.array([b, e[2, 1]], dtype=complex)
    M[2][2] = np.array([e[2, 2] - a * a - b * b], dtype=complex)
    mul, add, sub = np.polymul, np.polyadd, np.polysub
    c0 = mul(M[0][0], sub(mul(M[1][1], M[2][2]), mul(M[1][2], M[2][1])))
    c1 = mul(M[0][1], sub(mul(M[1][0], M[2][2]), mul(M[1][2], M[2][0])))
    c2 = mul(M[0][2], sub(mul(M[1][0], M[2][1]), mul(M[1][1], M[2][0])))
    return add(sub(c0, c1), c2)


def quartic_roots(eps, a, b):
    c = quartic_coeffs(eps, a, b)
    c = np.trim_zeros(np.asarray(c, dtype=complex), "f")
    return np.roots(c), c


def gen_eigs(cell, px, py, M, a0x, a0y, k0):
    s = Granet2DTransverseE(px, py, cell.shape[0], cell.shape[1], M, cell,
                            alpha0x=a0x, alpha0y=a0y, k0=k0)
    assert s.offplane, "fixture must dispatch to the out-of-plane generator"
    Lc = np.linalg.cholesky(s.Bgen)
    Ah = sla.solve_triangular(Lc, s.Agen, lower=True)
    Ah = sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
    return np.linalg.eigvals(Ah)


def cell_of(t, n=2):
    c = np.zeros((n, n, 3, 3), dtype=complex)
    c[:] = t
    return c


T_SYM = uniaxial_tensor(1.44, 1.76, 0.72, phi=0.44)
T_IN = uniaxial_tensor(1.44, 1.76, np.pi / 2, phi=0.44)
T_NR = T_SYM.copy()
T_NR[0, 2] = T_SYM[0, 2] + 0.23j          # e13 != e31, but eps stays HERMITIAN
T_NR[2, 0] = np.conj(T_NR[0, 2])
assert np.allclose(T_NR, T_NR.conj().T)
assert not np.allclose(T_NR, T_NR.T)

out = {"root": ROOT, "lumenairy": lumenairy.__file__, "A": [], "B": [],
       "C": [], "D": {}}


def save():
    json.dump(out, open(OUT, "w"), indent=1)


# --------------------------------------------------------------- A --------
k0 = 2.0 * np.pi / WL
for tname, tens in (("in_plane_control", T_IN), ("symmetric_oop", T_SYM),
                    ("non_reciprocal_oop", T_NR)):
    for mtag, th, ph in (("normal", 0.0, 0.0),
                         ("conical25_40", np.deg2rad(25.0), np.deg2rad(40.0))):
        a = float(np.real(NSUP) * np.sin(th) * np.cos(ph))
        b = float(np.real(NSUP) * np.sin(th) * np.sin(ph))
        rts, coeffs = quartic_roots(tens, a, b)
        rts_m, _ = quartic_roots(tens, -a, -b)
        rts_T, cT = quartic_roots(np.asarray(tens).T, a, b)
        sumroots = complex(-coeffs[1] / coeffs[0]) if len(coeffs) == 5 else None
        cell = cell_of(tens, 2)
        rec = {"tensor": tname, "mount": mtag, "a": a, "b": b,
               "sum_of_roots": [float(np.real(sumroots)),
                                float(np.imag(sumroots))],
               "roots_transpose_gap": float(np.max(np.abs(
                   np.sort_complex(rts) - np.sort_complex(rts_T)))),
               "arms": []}
        if tname == "in_plane_control":
            # forced dispatch so the OOP generator runs on a zero-cross cell
            cell = cell.copy()
            cell[..., 0, 2] = cell[..., 1, 2] = 0.0
            cell[..., 2, 0] = cell[..., 2, 1] = 0.0
            orig = TS._tile_is_offplane
            TS._tile_is_offplane = lambda _t: True
            try:
                for rot in (-1.0, 1.0):
                    TS._OOP_ROT_SIGN = rot
                    ev = gen_eigs(cell, PX, PX, 8, a * k0, b * k0, k0)
                    rec["arms"].append({
                        "rot": rot,
                        "d_plus_kt": float(np.max([np.min(np.abs(ev - r))
                                                   for r in rts])),
                        "d_minus_kt": float(np.max([np.min(np.abs(ev - r))
                                                    for r in rts_m]))})
            finally:
                TS._tile_is_offplane = orig
                TS._OOP_ROT_SIGN = -1.0
        else:
            for rot in (-1.0, 1.0):
                TS._OOP_ROT_SIGN = rot
                ev = gen_eigs(cell, PX, PX, 8, a * k0, b * k0, k0)
                rec["arms"].append({
                    "rot": rot,
                    "d_plus_kt": float(np.max([np.min(np.abs(ev - r))
                                               for r in rts])),
                    "d_minus_kt": float(np.max([np.min(np.abs(ev - r))
                                                for r in rts_m]))})
            TS._OOP_ROT_SIGN = -1.0
        out["A"].append(rec)
        print(f"[A {tname} {mtag}] sum_of_roots={sumroots:.6g}  "
              f"transpose_gap={rec['roots_transpose_gap']:.2e}", flush=True)
        for arm in rec["arms"]:
            print(f"    rot={arm['rot']:+.0f}  d(+k_t)={arm['d_plus_kt']:.3e}"
                  f"  d(-k_t)={arm['d_minus_kt']:.3e}", flush=True)
        save()


# --------------------------------------------------------------- B --------
def stag_uniform(tens, th, ph, M, cell=None):
    c = cell_of(tens, 2) if cell is None else cell
    o, R, T, J = pmm_jones_2d_staggered(PX, PX, c, NSUB, NSUP, DEP, WL,
                                        degree=M, n_orders=3, theta=th, phi=ph)
    oo = np.asarray(o)
    k = int(np.where((oo[:, 0] == 0) & (oo[:, 1] == 0))[0][0])
    return R[:, k], T[:, k], np.asarray(J), R, T, oo, k


for mtag, th, ph in (("normal", 0.0, 0.0),
                     ("oblique25", np.deg2rad(25.0), 0.0),
                     ("conical25_40", np.deg2rad(25.0), np.deg2rad(40.0))):
    for tname, tens in (("symmetric_oop", T_SYM),
                        ("non_reciprocal_oop", T_NR)):
        Rb, Tb, Jrb, _ = berreman_jones_1d([(tens, DEP)], NSUB, NSUP, WL,
                                           theta=th, phi=ph)
        row = {"tensor": tname, "mount": mtag, "ladder": []}
        for M in (3, 4, 5, 6, 7, 8):
            R0, T0, J, Rall, Tall, oo, k = stag_uniform(tens, th, ph, M)
            row["ladder"].append({
                "M": M, "dR": float(np.max(np.abs(R0 - Rb))),
                "dT": float(np.max(np.abs(T0 - Tb))),
                "dJones": float(np.max(np.abs(J - Jrb))),
                "leak": float(max(np.max(np.abs(np.delete(Rall, k, axis=1))),
                                  np.max(np.abs(np.delete(Tall, k, axis=1)))))})
        out["B"].append(row)
        s = "  ".join(f"M{d['M']}:{d['dJones']:.2e}" for d in row["ladder"])
        print(f"[B {tname} {mtag}] dJones ladder  {s}", flush=True)
        print("    dR ladder  " + "  ".join(
            f"M{d['M']}:{d['dR']:.2e}" for d in row["ladder"]), flush=True)
        print("    leak       " + "  ".join(
            f"M{d['M']}:{d['leak']:.1e}" for d in row["ladder"]), flush=True)
        save()

# --------------------------------------------------------------- C --------
CTRL = {
    "reference": lambda t: t,
    "drop_oop": lambda t: np.array([[t[0, 0], t[0, 1], 0], [t[1, 0], t[1, 1], 0],
                                    [0, 0, t[2, 2]]], dtype=complex),
    "negate_oop": lambda t: np.array(
        [[t[0, 0], t[0, 1], -t[0, 2]], [t[1, 0], t[1, 1], -t[1, 2]],
         [-t[2, 0], -t[2, 1], t[2, 2]]], dtype=complex),
    "transpose_oop": lambda t: np.array(
        [[t[0, 0], t[0, 1], t[2, 0]], [t[1, 0], t[1, 1], t[2, 1]],
         [t[0, 2], t[1, 2], t[2, 2]]], dtype=complex),
}
for mtag, th, ph in (("normal", 0.0, 0.0),
                     ("oblique25", np.deg2rad(25.0), 0.0),
                     ("conical25_40", np.deg2rad(25.0), np.deg2rad(40.0))):
    for tname, tens in (("symmetric_oop", T_SYM),
                        ("non_reciprocal_oop", T_NR)):
        Rb, Tb, Jrb, _ = berreman_jones_1d([(tens, DEP)], NSUB, NSUP, WL,
                                           theta=th, phi=ph)
        row = {"tensor": tname, "mount": mtag, "controls": []}
        for cname, f in CTRL.items():
            R0, T0, J, _Ra, _Ta, _oo, _k = stag_uniform(f(tens), th, ph, 8)
            row["controls"].append({
                "control": cname, "dR": float(np.max(np.abs(R0 - Rb))),
                "dT": float(np.max(np.abs(T0 - Tb))),
                "dJones": float(np.max(np.abs(J - Jrb)))})
            print(f"[C {tname} {mtag}] {cname:15s} dR="
                  f"{row['controls'][-1]['dR']:.3e} dJ="
                  f"{row['controls'][-1]['dJones']:.3e}", flush=True)
        out["C"].append(row)
        save()

# --------------------------------------------------------------- D --------
clean = cell_of(T_IN, 2)
clean[..., 0, 2] = clean[..., 1, 2] = 0.0
clean[..., 2, 0] = clean[..., 2, 1] = 0.0
clean[0, 0] = 2.31 * np.eye(3, dtype=complex)
base = pmm_jones_2d_staggered(PX, PX, clean, NSUB, NSUP, DEP, WL, degree=6,
                              n_orders=3, theta=np.deg2rad(20.0),
                              phi=np.deg2rad(35.0))
scale = float(np.max(np.abs(clean)))
res = {}
for stray in (1e-16, 1e-14, 1e-12, 1e-11, 1e-3):
    c = clean.copy()
    c[0, 0, 0, 2] = stray * scale
    disp = bool(TS._tile_is_offplane(c))
    a = pmm_jones_2d_staggered(PX, PX, c, NSUB, NSUP, DEP, WL, degree=6,
                               n_orders=3, theta=np.deg2rad(20.0),
                               phi=np.deg2rad(35.0))
    res[f"{stray:g}"] = {
        "dispatch_offplane": disp,
        "R_bit_identical": bool(np.array_equal(a[1], base[1])),
        "T_bit_identical": bool(np.array_equal(a[2], base[2])),
        "J_bit_identical": bool(np.array_equal(np.asarray(a[3]),
                                               np.asarray(base[3]))),
        "dR": float(np.max(np.abs(a[1] - base[1]))),
        "dJones": float(np.max(np.abs(np.asarray(a[3])
                                      - np.asarray(base[3]))))}
    print(f"[D] stray={stray:g}*scale offplane={disp} "
          f"bit_identical={res[f'{stray:g}']['R_bit_identical']} "
          f"dR={res[f'{stray:g}']['dR']:.3e}", flush=True)
out["D"] = res
save()
print("DONE")

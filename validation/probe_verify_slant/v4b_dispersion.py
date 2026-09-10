"""V4b -- THE SHEARED-FRAME DISPERSION, against the EXACT quartic roots, with
three ablations.

For a UNIFORM cell the pencil's spectrum is knowable in closed form.  A plane
wave ``exp(i(alpha . r_t + kz z))`` in a medium ``eps`` satisfies
``det(k k^T - (k.k) I + eps) = 0`` -- a QUARTIC in ``kz`` (all in units of
``k0``), whose coefficients this probe builds by exact polynomial arithmetic on
the 3x3 cofactor expansion (no fitting, no sampling).  In the sheared frame
``x = u + t w`` the same wave reads
``exp(i(alpha . (u, v) + (kz + t . alpha) w))``, so::

    q_frame(m)  =  kz_root(eps_lab; alpha_m)  +  t_int . alpha_m

with FOUR roots per transverse harmonic.  The gate is: every exact value has a
computed eigenvalue on top of it.

FOUR ARMS on the gauge, because the assembly runs in the ``_OOP_ROT_SIGN``
rotated frame and a wrong half is exactly the failure the build reports:
``a+/a-`` = harmonics ``+alpha`` / ``-alpha``, ``s+/s-`` = shift
``+t.alpha`` / ``-t.alpha``.

THREE ABLATIONS, each driven through the SHIPPED ``_assemble_oop`` by setting
the attributes it reads and re-assembling (no monkeypatching of any function):

  * SIX BLOCKS REMOVED  -- ``solver._slant_rot = (0, 0)`` (congruence kept);
  * CONGRUENCE REMOVED  -- ``solver.eps_cell`` put back to the rotated LAB
    tensor (six blocks kept);
  * WRONG GAUGE         -- the congruence taken with the UN-rotated ``t`` while
    the six blocks use the rotated one (the ``eps^{lm}(R eps R, -t) =
    R eps^{lm}(eps, t) R`` consistency broken in one place).
"""
import numpy as np
import scipy.linalg as sla
from _lib import arm, dump  # noqa: I001

from lumenairy.elements.pmm.twod_staggered import (
    _OOP_ROT_SIGN,
    Granet2DTransverseE,
    _slant_congruence,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor

WL = 1.0
K0 = 2.0 * np.pi / WL
PX = PY = 0.85
NXG = 2
import os as _os

CORE = int(_os.environ.get("VSLANT_CORE", "2"))
#: |m|, |n| <= CORE are the compared harmonics.  CORE = 0 is the
#: FUNDAMENTAL only -- the statement the build makes (its 1.4e-14);
#: CORE = 1 / 2 add the higher harmonics, which a (2,2) x M=8 basis
#: resolves progressively less well, so the gap grows by construction.

UNI = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
NONREC = np.array([[2.4, 0.15, 0.22],
                   [0.15, 2.1, -0.09],
                   [-0.31, 0.05, 2.3]], dtype=complex)   # e13 != e31
ISO = np.diag([2.25, 2.25, 2.25]).astype(complex)
TENSORS = {"uniaxial_tilt35": UNI, "nonreciprocal": NONREC, "isotropic": ISO}

T20 = float(np.tan(np.deg2rad(20.0)))
T35 = float(np.tan(np.deg2rad(35.0)))
SLANTS = {"none": (0.0, 0.0), "x20": (T20, 0.0),
          "diag35": (T35 / np.sqrt(2), T35 / np.sqrt(2))}
MOUNTS = {"normal": (0.0, 0.0), "conical": (0.30, 0.21)}   # normalized ax0/ay0

_pm = np.polynomial.polynomial


def _pmul(a, b):
    return _pm.polymul(a, b)


def _padd(*ps):
    r = np.array([0.0 + 0j])
    for p in ps:
        r = _pm.polyadd(r, p)
    return r


def quartic_roots(eps, ax, ay):
    """The four EXACT ``kz`` (in units of k0) of ``det(k k^T - (k.k) I + eps)``
    by cofactor expansion in exact polynomial arithmetic."""
    e = np.asarray(eps, dtype=complex)
    z = np.array([0.0 + 0j])
    q1 = np.array([0.0 + 0j, 1.0 + 0j])          # q
    q2 = np.array([0.0 + 0j, 0.0 + 0j, 1.0 + 0j])  # q^2
    M11 = _padd(np.array([-(ay ** 2) + e[0, 0] + 0j]), -q2)
    M22 = _padd(np.array([-(ax ** 2) + e[1, 1] + 0j]), -q2)
    M33 = np.array([-(ax ** 2) - (ay ** 2) + e[2, 2] + 0j])
    M12 = np.array([ax * ay + e[0, 1] + 0j])
    M21 = np.array([ax * ay + e[1, 0] + 0j])
    M13 = _padd(np.array([e[0, 2] + 0j]), ax * q1)
    M31 = _padd(np.array([e[2, 0] + 0j]), ax * q1)
    M23 = _padd(np.array([e[1, 2] + 0j]), ay * q1)
    M32 = _padd(np.array([e[2, 1] + 0j]), ay * q1)
    det = _padd(_pmul(M11, _padd(_pmul(M22, M33), -_pmul(M23, M32))),
                -_pmul(M12, _padd(_pmul(M21, M33), -_pmul(M23, M31))),
                _pmul(M13, _padd(_pmul(M21, M32), -_pmul(M22, M31))))
    det = np.trim_zeros(det, "b")
    if det.size < 2:
        return np.array([], dtype=complex)
    _ = z
    return np.roots(det[::-1])


def exact_set(eps, ax0, ay0, t, sgn_alpha, sgn_shift):
    vals = []
    for m in range(-CORE, CORE + 1):
        for n in range(-CORE, CORE + 1):
            ax = sgn_alpha * (ax0 + m * WL / PX)
            ay = sgn_alpha * (ay0 + n * WL / PY)
            for r in quartic_roots(eps, ax, ay):
                vals.append(r + sgn_shift * (t[0] * ax + t[1] * ay))
    return np.asarray(vals)


def pencil_eigs(A, B):
    return sla.eig(A, B, right=False)


def build(eps, sl, ax0, ay0, M):
    cell = np.zeros((NXG, NXG, 3, 3), dtype=complex)
    cell[:, :] = np.asarray(eps, dtype=complex)
    return Granet2DTransverseE(PX, PY, NXG, NXG, M, cell,
                               alpha0x=ax0 * K0, alpha0y=ay0 * K0, k0=K0,
                               slant=(sl if (sl[0] or sl[1]) else None))


def worst_gap(qv, ex):
    return float(np.max([np.min(np.abs(qv - e)) for e in ex]))


def main():
    M = 8
    out = {"config": dict(px=PX, M=M, grid=NXG, core=CORE), "arms": {},
           "ablations": {}, "sum_of_roots": {}, "ladder": {}}
    for tname, eps in TENSORS.items():
        for sname, sl in SLANTS.items():
            for mname, (ax0, ay0) in MOUNTS.items():
                sol = build(eps, sl, ax0, ay0, M)
                if not sol.offplane:
                    continue
                qv = pencil_eigs(sol.Agen, sol.Bgen)
                tint = (-sl[0], -sl[1])          # x = u + t w, t = -slant
                row = {}
                for sa, salab in ((+1, "a+"), (-1, "a-")):
                    for ss, sslab in ((+1, "s+"), (-1, "s-")):
                        ex = exact_set(eps, ax0, ay0, tint, sa, ss)
                        row[f"{salab}{sslab}"] = worst_gap(qv, ex)
                key = f"{tname}/{sname}/{mname}"
                out["arms"][key] = row
                print(f"[arm] {key}: " +
                      "  ".join(f"{k} {v:.3e}" for k, v in row.items()))

    # ---- ablations, on the conical mount (where the shear couples)
    ax0, ay0 = MOUNTS["conical"]
    for tname, eps in TENSORS.items():
        for sname, sl in SLANTS.items():
            if sname == "none":
                continue
            tint = (-sl[0], -sl[1])
            ex = exact_set(eps, ax0, ay0, tint, +1, +1)
            base = build(eps, sl, ax0, ay0, M)
            full = worst_gap(pencil_eigs(base.Agen, base.Bgen), ex)

            s1 = build(eps, sl, ax0, ay0, M)
            s1._slant_rot = (0.0, 0.0)
            s1._assemble_oop()
            no_blocks = worst_gap(pencil_eigs(s1.Agen, s1.Bgen), ex)

            s2 = build(eps, sl, ax0, ay0, M)
            e_rot = np.array(s2.eps_lab, dtype=complex, copy=True)
            e_rot[..., 0, 2] *= _OOP_ROT_SIGN
            e_rot[..., 1, 2] *= _OOP_ROT_SIGN
            e_rot[..., 2, 0] *= _OOP_ROT_SIGN
            e_rot[..., 2, 1] *= _OOP_ROT_SIGN
            s2.eps_cell = e_rot                  # congruence REMOVED
            s2._assemble_oop()
            no_cong = worst_gap(pencil_eigs(s2.Agen, s2.Bgen), ex)

            s3 = build(eps, sl, ax0, ay0, M)
            s3.eps_cell = _slant_congruence(e_rot, tint[0], tint[1])
            s3._assemble_oop()                   # WRONG gauge on the eps half
            wrong_gauge = worst_gap(pencil_eigs(s3.Agen, s3.Bgen), ex)

            out["ablations"][f"{tname}/{sname}"] = dict(
                full=full, six_blocks_removed=no_blocks,
                congruence_removed=no_cong, wrong_gauge=wrong_gauge)
            print(f"[abl] {tname}/{sname}: full {full:.3e}  no-blocks "
                  f"{no_blocks:.3e}  no-congruence {no_cong:.3e}  "
                  f"wrong-gauge {wrong_gauge:.3e}")

    # ---- the SUM-OF-ROOTS discriminator on the fundamental harmonic
    for sname, sl in SLANTS.items():
        tint = (-sl[0], -sl[1])
        ax, ay = MOUNTS["conical"]
        r = quartic_roots(UNI, ax, ay)
        want = float(np.real(np.sum(r)) + 4.0 * (tint[0] * ax + tint[1] * ay))
        sol = build(UNI, sl, ax, ay, M)
        qv = pencil_eigs(sol.Agen, sol.Bgen)
        near = sorted(qv, key=lambda q: min(abs(q - (rr + tint[0] * ax +
                                                    tint[1] * ay))
                                            for rr in r))[:4]
        got = float(np.real(np.sum(near)))
        out["sum_of_roots"][sname] = dict(generator=got, exact=want)
        print(f"[sum] {sname}: generator {got:.6f}  exact {want:.6f}")

    # ---- the M-ladder
    ax0, ay0 = MOUNTS["conical"]
    sl = SLANTS["diag35"]
    tint = (-sl[0], -sl[1])
    ex = exact_set(UNI, ax0, ay0, tint, +1, +1)
    for Mi in (4, 5, 6, 7, 8):
        sol = build(UNI, sl, ax0, ay0, Mi)
        out["ladder"][Mi] = worst_gap(pencil_eigs(sol.Agen, sol.Bgen), ex)
        print(f"[lad] M={Mi}: {out['ladder'][Mi]:.3e}")

    dump("v4b_dispersion", out)
    print("arm", arm())


if __name__ == "__main__":
    main()

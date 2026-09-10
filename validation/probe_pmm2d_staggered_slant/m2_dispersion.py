"""M2 -- UNIFORM-SLAB DISPERSION IN THE SHEARED FRAME (the sign/factor gate).

For a UNIFORM cell every physical mode is a plane wave on a Bloch harmonic.  In
the frame ``u = x - t_x w``, ``v = y - t_y w``, ``w = z`` a lab plane wave
``exp(i alpha.x_t + i k_z z)`` reads ``exp(i alpha.u_t + i (k_z + t.alpha) w)``,
so the generator's eigenvalues must be

    q(m, n) = kz_root(eps_lab; u, v) + t_x u + t_y v          (all in k0 units)

with ``kz_root`` the four EXACT roots of ``det(k k^T - |k|^2 I + eps) = 0``
(``probe_common.exact_kz_roots``, exact polynomial arithmetic).  The shear
translates each harmonic's four roots RIGIDLY by ``t.alpha``: it cannot be
mimicked by any tensor error, so this is the gate that catches a sign or factor
slip in either the covariant congruence or the six slant blocks.

Two-sided arms, all four combinations of
  * the transverse gauge ``alpha -> +alpha`` vs ``-alpha`` (the shipped
    ``_OOP_ROT_SIGN`` question, re-asked with a slant present), and
  * the shift ``+t.alpha`` vs ``-t.alpha``,
plus the CONTROL of dropping the six slant blocks (tensor congruence only) and
of dropping the congruence (slant blocks only).  Only one arm may pass.

Discriminator recorded per row: ``sum`` of the fundamental's four roots, which
is ``4 t.alpha`` above its vertical value -- a quantity no in-plane tensor can
produce.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "probe_pmm2d_staggered_oop"))

from slant_lib import (  # noqa: E402
    SlantSolver, assert_worktree, cov_tensor, tensor_uniaxial,
)
import probe_common as pc  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
res = {"lumenairy": assert_worktree(), "rows": []}
t00 = time.time()

WL = 1.0
PX = PY = 0.9
K0 = 2.0 * np.pi / WL


def gen_spectrum(cell33, slant, kx0, ky0, M, *, blocks=True, congruence=True):
    """Eigenvalues q of the prototype generator for a UNIFORM tensor cell."""
    c = np.zeros((2, 2, 3, 3), dtype=complex)
    c[:, :] = cell33
    sol = SlantSolver.__new__(SlantSolver)
    # explicit construction so the two CONTROL ablations can be driven
    from lumenairy.elements.pmm.twod_staggered import Basis1D, _OOP_ROT_SIGN
    sol.k0 = K0
    sol.alpha0x, sol.alpha0y = kx0 * K0, ky0 * K0
    sol.bx = Basis1D(PX, 2, M, np.exp(-1j * kx0 * K0 * PX))
    sol.by = Basis1D(PY, 2, M, np.exp(-1j * ky0 * K0 * PY))
    e = np.array(c, dtype=complex, copy=True)
    for a, b in ((0, 2), (1, 2), (2, 0), (2, 1)):
        e[..., a, b] *= _OOP_ROT_SIGN
    txr, tyr = _OOP_ROT_SIGN * slant[0], _OOP_ROT_SIGN * slant[1]
    sol.eps_cell = cov_tensor(e, txr, tyr) if congruence else e
    sol.tx, sol.ty = (txr, tyr) if blocks else (0.0, 0.0)
    sol.eps_lab, sol.slant_lab = c, slant
    sol.q = sol.bx.dim
    sol.offplane = True
    sol._assemble_slant()
    import scipy.linalg as sla
    Lc = np.linalg.cholesky(sol.Bgen)
    Ah = sla.solve_triangular(Lc, sol.Agen, lower=True)
    Ah = sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
    return np.linalg.eigvals(Ah)


def match(qv, roots):
    """max over the 4 exact roots of the distance to the nearest eigenvalue."""
    return float(max(np.min(np.abs(qv - r)) for r in roots))


TENS = {
    "uniax_tilt35_azim25": tensor_uniaxial(1.5, 1.7, np.deg2rad(35),
                                           np.deg2rad(25)),
    "isotropic_2.25": 2.25 * np.eye(3, dtype=complex),
    "nonreciprocal": np.array([[2.25, 0, 0.35j], [0, 2.4, 0],
                               [-0.35j, 0, 2.6]], dtype=complex),
}
MOUNTS = {"normal": (0.0, 0.0), "conical25_40": (np.deg2rad(25),
                                                 np.deg2rad(40))}
SLANTS = {"none": (0.0, 0.0), "x20": (np.tan(np.deg2rad(20)), 0.0),
          "diag35": (np.tan(np.deg2rad(35)), 0.7 * np.tan(np.deg2rad(35)))}

for tname, t33 in TENS.items():
    for mname, (th, ph) in MOUNTS.items():
        kx0 = np.sin(th) * np.cos(ph)
        ky0 = np.sin(th) * np.sin(ph)
        for sname, sl in SLANTS.items():
            M = 8
            qv = gen_spectrum(t33, sl, kx0, ky0, M)
            row = {"tensor": tname, "mount": mname, "slant": sname}
            for gs, gname in ((+1.0, "a+"), (-1.0, "a-")):
                u, v = gs * kx0, gs * ky0
                base = np.array(pc.exact_kz_roots(t33, u, v))
                for ss, sgn in ((+1.0, "s+"), (-1.0, "s-")):
                    shift = ss * (sl[0] * u + sl[1] * v)
                    row[f"{gname}{sgn}"] = match(qv, base + shift)
            # controls (only meaningful when a slant is present)
            if sl != (0.0, 0.0):
                qb = gen_spectrum(t33, sl, kx0, ky0, M, blocks=False)
                qc = gen_spectrum(t33, sl, kx0, ky0, M, congruence=False)
                base = np.array(pc.exact_kz_roots(t33, kx0, ky0))
                sh = sl[0] * kx0 + sl[1] * ky0
                row["ctrl_no_blocks"] = match(qb, base + sh)
                row["ctrl_no_congruence"] = match(qc, base + sh)
            # the sum-of-roots discriminator on the fundamental
            base = np.array(pc.exact_kz_roots(t33, kx0, ky0))
            sh = sl[0] * kx0 + sl[1] * ky0
            near = [qv[np.argmin(np.abs(qv - (r + sh)))] for r in base]
            row["sum_gen"] = complex(np.sum(near)).real
            row["sum_exact"] = complex(np.sum(base) + 4 * sh).real
            res["rows"].append(row)
            print(f"M2 {tname:20s} {mname:12s} {sname:6s} "
                  f"a+s+ {row['a+s+']:.2e} a+s- {row['a+s-']:.2e} "
                  f"a-s+ {row['a-s+']:.2e} a-s- {row['a-s-']:.2e} "
                  f"| sum {row['sum_gen']:+.6f} vs {row['sum_exact']:+.6f}"
                  + (f" | ctrl noblk {row['ctrl_no_blocks']:.2e} "
                     f"nocov {row['ctrl_no_congruence']:.2e}"
                     if "ctrl_no_blocks" in row else ""))

# M-ladder on the worst slanted row (the basis is POLYNOMIAL, so the match is a
# convergent statement, not an identity)
lad = []
t33 = TENS["uniax_tilt35_azim25"]
kx0 = np.sin(np.deg2rad(25)) * np.cos(np.deg2rad(40))
ky0 = np.sin(np.deg2rad(25)) * np.sin(np.deg2rad(40))
sl = SLANTS["diag35"]
base = np.array(pc.exact_kz_roots(t33, kx0, ky0)) + (sl[0] * kx0 + sl[1] * ky0)
for M in (4, 5, 6, 7, 8, 9):
    qv = gen_spectrum(t33, sl, kx0, ky0, M)
    lad.append({"M": M, "resid": match(qv, base)})
    print(f"M2 ladder M={M}  {lad[-1]['resid']:.3e}")
res["ladder"] = lad
res["wall_s"] = time.time() - t00
with open(os.path.join(OUT, "m2_dispersion.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"\nWROTE results/m2_dispersion.json  ({res['wall_s']:.1f} s)")

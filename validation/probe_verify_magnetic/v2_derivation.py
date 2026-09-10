"""V2 -- the MAGNETIC operator algebra, re-derived and re-assembled here.

Granet, JOSA A 40, 652 (2023), Eqs. 13-14 (paper convention ``exp(+i w t)``,
``U = U(x1,x2) exp(-i gamma x3)``), eliminating H:

    H_t = (i / (w mu0)) [chi_t] ( i gamma C E_t + [d2; -d1] E3 )
    H3  = (i chi33 / (w mu0)) [-d2  d1] E_t

into Eq. 14a and multiplying by ``-i w mu0``:

    -gamma^2 C[chi_t]C E_t = k^2 [eps_t] E_t - i gamma C[chi_t][d2;-d1] E3
                             + [d2;-d1] chi33 [d2  -d1] E_t

so, with ``C = [[0,1],[-1,0]]`` and ``[chi_t] = [[chi11,chi12],[chi21,chi22]]``,

    C[chi_t]      = [[ chi21,  chi22], [-chi11, -chi12]]
    R  = C[chi_t]C        = [[-chi22,  chi21], [ chi12, -chi11]]      (Eq. 24)
    K_tz = C[chi_t][d2;-d1] = [ -chi22 d1 + chi21 d2 ;
                                -chi11 d2 + chi12 d1 ]                (Eq. 21)
    S_tt = [d2;-d1] chi33 [d2  -d1]  ->  <v,S_tt E> = -<curl v, chi33 curl E>

    Eq. 25:  gamma (w mu0) C H_t = (k^2 [eps_t] + S_tt) E_t  -- NO chi_t.

This probe re-assembles ``R``, ``S_tt``, ``K_tz``, ``M_eps33``, ``K_zt`` and
hence ``L`` for a UNIFORM cell from the module's 1-D primitives ONLY (mass /
stiff / mixed / c_ref), using the placements above, and compares with the
shipped retained operators.  ``S_tt`` in particular is built the APPENDIX Eq. 42
way (stiffness + derivative-on-left matrices) -- a route the shipped code never
takes (it goes through ``Gw^-1 Gw_chi Gw^-1``), so agreement also confirms the
de Rham exactness the shipped form assumes.

Arms 2 and 3: the R-vs-Gram TRAP -- collapse the two roles and measure.
"""
import numpy as np

import lumenairy
from lumenairy.elements.pmm import stack2d_pure as _sp
from lumenairy.elements.pmm import twod_staggered as _ts
from lumenairy.elements.pmm.twod_staggered import Granet2DTransverseE

assert lumenairy.__file__.startswith("C:\\tmp\\lum_vmag"), lumenairy.__file__

_K0 = 2.0 * np.pi / 0.55e-6


def _prims(b, k0):
    """The 1-D globals this probe needs, all from :class:`Basis1D`."""
    return {
        "Mbb": b.mass(b.B, b.B),
        "Mtt": b.mass(b.Btilde, b.Btilde),
        "Mbt": b.mass(b.B, b.Btilde),          # <B | Btilde>
        "Mtb": b.mass(b.Btilde, b.B),          # <Btilde | B>
        "Sxtt": b.stiff(b.Btilde, b.Btilde) / k0 ** 2,   # <dBtilde|dBtilde>
        "dR_bt": b.mixed(b.B, b.Btilde) / k0,            # <B | d Btilde>
        "dR_tt": b.mixed(b.Btilde, b.Btilde) / k0,       # <Btilde | d Btilde>
        # derivative on the LEFT (test): INT (d L)* R, c_ref, scale 1
        "dL_tb": b._global_matrix(b.c_ref, b.Btilde, b.B) / k0,
        "dL_tt": b._global_matrix(b.c_ref, b.Btilde, b.Btilde) / k0,
    }


def independent_assembly(px, py, nx, ny, m, eps, mu, a0x, a0y, k0=_K0):
    """R, S_tt, K_tz, Schur, L for a UNIFORM isotropic ``eps`` and a UNIFORM
    block-form ``mu``, assembled from the derivation above."""
    bx = _ts.Basis1D(px, nx, m, np.exp(-1j * a0x * px))
    by = _ts.Basis1D(py, ny, m, np.exp(-1j * a0y * py))
    x, y = _prims(bx, k0), _prims(by, k0)
    kron = np.kron

    mt = np.asarray(mu, dtype=complex)[:2, :2]
    chi = np.linalg.inv(mt)
    c11, c12, c21, c22 = chi[0, 0], chi[0, 1], chi[1, 0], chi[1, 1]
    c33 = 1.0 / complex(mu[2, 2])

    # --- R = C[chi_t]C = [[-chi22, chi21], [chi12, -chi11]] -----------------
    g1 = kron(y["Mtt"], x["Mbb"])           # <V1|V1>
    g2 = kron(y["Mbb"], x["Mtt"])           # <V2|V2>
    m12 = kron(y["Mtb"], x["Mbt"])          # <V1|V2>
    m21 = kron(y["Mbt"], x["Mtb"])          # <V2|V1>
    r = np.block([[-c22 * g1, c21 * m12], [c12 * m21, -c11 * g2]])

    # --- S_tt : -<curl v, chi33 curl E>, curl_z = d1 E2 - d2 E1 -------------
    s11 = -c33 * kron(y["Sxtt"], x["Mbb"])
    s12 = c33 * kron(y["dL_tb"], x["dR_bt"])
    s21 = c33 * kron(y["dR_bt"], x["dL_tb"])
    s22 = -c33 * kron(y["Mbb"], x["Sxtt"])
    stt = np.block([[s11, s12], [s21, s22]])

    # --- K_tz = C[chi_t][d2;-d1] -------------------------------------------
    d1_v1 = kron(y["Mtt"], x["dR_bt"])      # <V1 | d1 V3>
    d2_v1 = kron(y["dR_tt"], x["Mbt"])      # <V1 | d2 V3>
    d2_v2 = kron(y["dR_bt"], x["Mtt"])      # <V2 | d2 V3>
    d1_v2 = kron(y["Mbt"], x["dR_tt"])      # <V2 | d1 V3>
    ktz = np.concatenate([-c22 * d1_v1 + c21 * d2_v1,
                          -c11 * d2_v2 + c12 * d1_v2], axis=0)

    # --- Meps33 (V3 mass) and K_zt = [d1 eps, d2 eps], isotropic eps --------
    meps33 = eps * kron(y["Mtt"], x["Mtt"])
    kzt = np.concatenate([-eps * kron(y["Mtt"], x["dL_tb"]),
                          -eps * kron(y["dL_tb"], x["Mtt"])], axis=1)
    schur = ktz @ np.linalg.solve(meps33, kzt)

    et = np.block([[eps * g1, np.zeros_like(g1)],
                   [np.zeros_like(g2), eps * g2]])
    return {"R": r, "Stt": stt, "Ktz": ktz, "Schur": schur,
            "L": et + stt - schur, "Ggram": (g1, g2)}


def _rel(a, b):
    d = float(np.max(np.abs(a - b)))
    s = max(float(np.max(np.abs(b))), 1e-300)
    return d / s


def _mu(m11=1.30, m22=1.75, m12=0.22, m21=-0.15, m33=1.45):
    return np.array([[m11, m12, 0.0], [m21, m22, 0.0], [0.0, 0.0, m33]],
                    dtype=complex)


def arm_operators():
    px = py = 0.90e-6
    nx = ny = 2
    m = 5
    a0x, a0y = 0.31 * _K0, 0.17 * _K0
    eps = 4.0 + 0.0j
    print("--- V2A: shipped magnetic operators vs the independent assembly ---")
    for tag, mu in (("mu = I (identity, forced)", _mu(1, 1, 0, 0, 1)),
                    ("mu = diag(1.3, 1.75, 1.45)", _mu(m12=0.0, m21=0.0)),
                    ("mu general (m12 != m21)", _mu()),
                    ("mu gyrotropic (m12 = -m21 = 0.4i)",
                     _mu(1.20, 1.20, 0.4j, -0.4j, 1.10)),
                    ("mu lossy (Im m11 = 0.25)",
                     _mu(1.30 + 0.25j, 1.75, 0.22, -0.15, 1.45))):
        cell = np.full((nx, ny), eps)
        mcell = np.zeros((nx, ny, 3, 3), dtype=complex)
        mcell[...] = mu
        sol = Granet2DTransverseE(px, py, nx, ny, m, cell, alpha0x=a0x,
                                  alpha0y=a0y, k0=_K0, mu_cell=mcell)
        ref = independent_assembly(px, py, nx, ny, m, eps, mu, a0x, a0y)
        gg = sol.Ggram_blocks
        print(f"  {tag:36s} R {_rel(sol.Rmat, ref['R']):.3e}  "
              f"Stt {_rel(sol.Stt, ref['Stt']):.3e}  "
              f"Schur {_rel(sol.Schur, ref['Schur']):.3e}  "
              f"L {_rel(sol.Lmat, ref['L']):.3e}  "
              f"Ggram {max(_rel(gg[0], ref['Ggram'][0]), _rel(gg[1], ref['Ggram'][1])):.3e}")


def arm_placement_knockouts():
    """The SAME comparison with each placement deliberately wrong: does the
    metric actually discriminate the choice, or would any placement pass?"""
    px = py = 0.90e-6
    nx = ny = 2
    m = 5
    a0x, a0y = 0.31 * _K0, 0.17 * _K0
    eps = 4.0 + 0.0j
    mu = _mu()
    cell = np.full((nx, ny), eps)
    mcell = np.zeros((nx, ny, 3, 3), dtype=complex)
    mcell[...] = mu
    sol = Granet2DTransverseE(px, py, nx, ny, m, cell, alpha0x=a0x,
                              alpha0y=a0y, k0=_K0, mu_cell=mcell)
    print("--- V2B: is the comparison SENSITIVE to the placement? ---")
    variants = {
        "as derived (R = C[chi]C, Ktz = C[chi][d2;-d1])": mu,
        "chi11 <-> chi22 on the R/Ktz diagonal": None,
        "chi12 <-> chi21 (mixed blocks swapped)": None,
        "no C rotation (R = -[chi_t])": None,
    }
    ref = independent_assembly(px, py, nx, ny, m, eps, mu, a0x, a0y)
    print(f"  {'as derived':52s} R {_rel(sol.Rmat, ref['R']):.3e}  "
          f"Schur {_rel(sol.Schur, ref['Schur']):.3e}")
    chi = np.linalg.inv(np.asarray(mu)[:2, :2])
    for name, swap in (("chi11 <-> chi22 (diagonal)", "diag"),
                       ("chi12 <-> chi21 (mixed)", "mixed"),
                       ("no C rotation: R = -[chi_t]", "noC")):
        c = chi.copy()
        if swap == "diag":
            c[0, 0], c[1, 1] = chi[1, 1], chi[0, 0]
        elif swap == "mixed":
            c[0, 1], c[1, 0] = chi[1, 0], chi[0, 1]
        else:
            c = np.array([[chi[1, 1], -chi[1, 0]], [-chi[0, 1], chi[0, 0]]])
        mu_alt = np.eye(3, dtype=complex)
        mu_alt[:2, :2] = np.linalg.inv(c)
        mu_alt[2, 2] = mu[2, 2]
        alt = independent_assembly(px, py, nx, ny, m, eps, mu_alt, a0x, a0y)
        print(f"  {name:52s} R {_rel(sol.Rmat, alt['R']):.3e}  "
              f"Schur {_rel(sol.Schur, alt['Schur']):.3e}")
    del variants


# --------------------------------------------------------------- the R/Gram trap
class _CollapsedGram(Granet2DTransverseE):
    """The TRAP arm: -R used for BOTH the pencil and the Eq.-25 H recovery."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.Ggram_blocks = None


def arm_gram_trap():
    print("--- V2C: the R-vs-Gram separation, against the analytic oracle ---")
    from v3_airy import airy_residual  # noqa: I001, PLC0415
    base = dict(eps=4.0, mu=2.0, thickness=0.30e-6, wl=0.55e-6, theta=0.35,
                m=8)
    print(f"  SHIPPED (plain Gram)            {airy_residual(**base):.4e}")
    orig = _sp.Granet2DTransverseE
    _sp.Granet2DTransverseE = _CollapsedGram
    try:
        print(f"  KNOCKOUT (-R used as the Gram) {airy_residual(**base):.4e}")
    finally:
        _sp.Granet2DTransverseE = orig
    print(f"  SHIPPED again (restored)        {airy_residual(**base):.4e}")


def main():
    arm_operators()
    print()
    arm_placement_knockouts()
    print()
    arm_gram_trap()


if __name__ == "__main__":
    main()

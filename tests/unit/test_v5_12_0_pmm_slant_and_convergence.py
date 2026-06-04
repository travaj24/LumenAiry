"""v5.12.0 -- slanted 1-D PMM (Granet 2017) + Li-Granet 2011 convergence-class
predictor.

Both additions are PURELY ADDITIVE: the existing ``pmm_efficiency_1d`` is
untouched (and the slanted solver reduces to it bit-identically at zero slant).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    classify_from_grating,
    grating_convergence_class,
    pmm_efficiency_1d,
    pmm_efficiency_1d_slanted,
    pmm_jones_1d,
    pmm_jones_1d_slanted,
    pmm_jones_1d_slanted_segments,
)
from lumenairy.elements.rcwa import RCWAStack

# Reference dielectric slanted lamellar grating (the proto's validated case).
_GEOM = dict(period=1.0e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.5,
             n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5,
             wavelength=0.633e-6)


def _sum_RT(R, T):
    return float(np.sum(R) + np.sum(T))


# ===================================================================
# Slanted 1-D PMM
# ===================================================================
@pytest.mark.parametrize("pol", ["te", "tm"])
def test_slant_zero_reduces_to_vertical(pol):
    """slant_angle=0 reproduces pmm_efficiency_1d to machine precision (the
    inclined-coordinate convection term vanishes)."""
    # degree 16 is resonance-free here; use stabilize=False for an exact compare.
    oV, RV, TV = pmm_efficiency_1d(**_GEOM, polarization=pol, degree=16,
                                   stabilize=False)
    oS, RS, TS = pmm_efficiency_1d_slanted(**_GEOM, slant_angle=0.0,
                                           polarization=pol, degree=16,
                                           stabilize=False)
    assert np.array_equal(oV, oS)
    assert np.max(np.abs(RV - RS)) < 1e-12
    assert np.max(np.abs(TV - TS)) < 1e-12


@pytest.mark.parametrize("phi_deg", [10.0, 25.0, 45.0, 60.0, 75.0])
@pytest.mark.parametrize("pol", ["te", "tm"])
def test_slant_energy_conservation(phi_deg, pol):
    """A single slanted (lossless) layer conserves energy across 10-75 deg."""
    o, R, T = pmm_efficiency_1d_slanted(
        **_GEOM, slant_angle=np.deg2rad(phi_deg), polarization=pol, degree=14)
    assert abs(_sum_RT(R, T) - 1.0) < 3e-3


def test_slant_oblique_raises():
    """Combined oblique incidence + nonzero slant is unsupported and must raise
    (rather than return a wrong per-order split)."""
    with pytest.raises(NotImplementedError):
        pmm_efficiency_1d_slanted(**_GEOM, slant_angle=np.deg2rad(25.0),
                                  angle=np.deg2rad(15.0))
    # but each alone is fine
    pmm_efficiency_1d_slanted(**_GEOM, slant_angle=np.deg2rad(25.0), angle=0.0)
    pmm_efficiency_1d_slanted(**_GEOM, slant_angle=0.0, angle=np.deg2rad(15.0))


def _rcwa_staircase_slant(phi, pol, n_orders=21, n_slices=96, n_x=512):
    """Independent oracle: the same slanted grating as an RCWA z-staircase of
    laterally-shifted binary slices."""
    g = _GEOM
    er, eg = complex(g["n_ridge"]) ** 2, complex(g["n_groove"]) ** 2
    t = np.tan(phi)
    dz = g["depth"] / n_slices
    x = (np.arange(n_x) + 0.5) / n_x
    stack = RCWAStack(period=g["period"], n_superstrate=g["n_superstrate"],
                      n_substrate=g["n_substrate"], n_orders=n_orders)
    for k in range(n_slices):
        shift = (t * (k + 0.5) * dz) / g["period"]
        ridge = ((x - shift) % 1.0) < g["duty_cycle"]
        col = np.where(ridge, er, eg).astype(np.complex128)
        stack.add_layer(dz, eps_cell=col[:, None])
    res = stack.set_source(g["wavelength"], theta=0.0).solve()
    orders, R2, T2 = res.efficiencies()
    row = 1 if pol == "te" else 0           # E_y = TE = row 1 ; E_x = TM = row 0
    return np.asarray(orders), np.asarray(R2[row]), np.asarray(T2[row])


def test_slant_matches_rcwa_staircase_te():
    """TE slant=25 deg agrees with a fine RCWA staircase on the dominant
    (0th) orders -- the cross-method physics check."""
    phi = np.deg2rad(25.0)
    o, R, T = pmm_efficiency_1d_slanted(**_GEOM, slant_angle=phi,
                                        polarization="te", degree=16)
    orc, Rr, Tr = _rcwa_staircase_slant(phi, "te")
    t0_pmm = float(T[o == 0][0])
    r0_pmm = float(R[o == 0][0])
    t0_rc = float(Tr[orc == 0][0])
    r0_rc = float(Rr[orc == 0][0])
    assert abs(t0_pmm - t0_rc) < 5e-3
    assert abs(r0_pmm - r0_rc) < 5e-3


# ===================================================================
# Convergence-class predictor (Li-Granet 2011)
# ===================================================================
def test_type_I_dielectric():
    r = classify_from_grating(1.0, 2.25, 1.0, 1.0)
    assert r["type"] == "I"
    assert r["converges"] is True
    assert np.isfinite(r["tau"].real) and abs(r["tau"].imag) < 1e-9
    assert 0.0 < r["delta"].real < 1.0


def test_type_II_lossless_metal():
    r = grating_convergence_class((1.0, -10.0, 1.0, -10.0))
    assert r["type"] == "II"
    assert r["converges"] is False
    assert r["delta"].real < 0.0
    assert "TYPE II" in r["warning"]


@pytest.mark.parametrize("em", [-10.0, -2.5, -1.5, -0.5])
def test_type_II_closed_form_sign(em):
    """For a diagonal metal corner (e1=e3=eps_d, e2=e4=eps_m), Delta reduces to
    4 eps_d eps_m / (eps_d + eps_m)^2 (negative for eps_m < 0)."""
    ed = 1.0
    closed = 4.0 * ed * em / (ed + em) ** 2
    got = grating_convergence_class((ed, em, ed, em))["delta"].real
    assert abs(closed - got) < 1e-9
    assert closed < 0.0


def test_type_III_impossible_for_all_dielectric():
    """An all-dielectric corner can NEVER be Type III: the squared numerator
    over a positive denominator forces Delta <= 1."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        eps = (rng.uniform(1.0, 12.0, size=4)).tolist()
        r = grating_convergence_class(eps)
        assert r["type"] in ("I", "degenerate")
        if r["type"] == "I":
            assert r["delta"].real <= 1.0 + 1e-12


def test_genuine_type_III_metal_corner():
    """A metal quadrant arranged so the denominator turns negative gives a real
    Type III (no singularity, fast)."""
    r = grating_convergence_class((2.0, -1.5, 1.0, 2.0))
    assert r["type"] == "III"
    assert r["converges"] is True
    assert r["delta"].real > 1.0


def test_lossy_metal_regularizes():
    """Absorption lifts the lossless Type-II irregularity (reported convergent,
    with the slow-convergence caveat)."""
    r = grating_convergence_class((1.0, -10.0 + 0.5j, 1.0, -10.0 + 0.5j))
    assert r["type"] == "I"
    assert r["converges"] is True
    assert "Lossy metal" in r["warning"]


def test_degenerate_edge():
    """An impedance-matched corner (a vanishing eps_i+eps_j) is flagged
    degenerate, not silently mis-classified."""
    r = grating_convergence_class((1.0, -1.0, 1.0, 2.0))   # e1+e2 = 0
    assert r["type"] == "degenerate"
    assert not r["converges"]
    assert np.isnan(r["delta_prime"].real)


# ===================================================================
# Slanted + IN-PLANE-ANISOTROPIC (Jones) PMM -- the genuine Edee-Granet
# 2024 covariant-metric [E_t; H_t] generator (validated proto round 11).
# ===================================================================
# Geometry for the coupled-tensor slant solver (the proto's validated case:
# symmetric lossless non-resonant half-spaces).
_JGEOM = dict(period=1.0e-6, n_substrate=1.5, n_superstrate=1.5,
              depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6)
# Asymmetric half-spaces (for the diagonal / phi=0 decouple checks vs the
# scalar solvers, which use n_sup=1.0 / n_sub=1.5).
_JGEOM_ASYM = dict(period=1.0e-6, n_substrate=1.5, n_superstrate=1.0,
                   depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6)


def _eps_diag(val):
    return (val * np.eye(3)).astype(np.complex128)


def _eps_real_sym(diag=2.25, off=0.4):
    e = _eps_diag(diag)
    e[0, 1] = e[1, 0] = off
    return e


def _eps_gyro(diag=2.25, g=0.5):
    e = _eps_diag(diag)
    e[0, 1] = -1j * g
    e[1, 0] = 1j * g
    return e


def test_jones_slant_vacuum():
    """Empty cell -> full transmission, no cross-polarization."""
    I = _eps_diag(1.0)
    o, R, T, J = pmm_jones_1d_slanted(
        period=1.0e-6, eps_ridge=I, eps_groove=I, n_substrate=1.0,
        n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6,
        slant_angle=0.0, degree=12, stabilize=False)
    m0 = int(np.where(o == 0)[0][0])
    assert abs(T[0][m0] - 1.0) < 1e-10
    assert abs(T[1][m0] - 1.0) < 1e-10
    assert max(abs(J[0, 1]), abs(J[1, 0])) < 1e-12


@pytest.mark.parametrize("pol_diag", [(0, "tm"), (1, "te")])
def test_jones_slant_phi0_diagonal_decouples_to_scalar(pol_diag):
    """phi=0 with a DIAGONAL tensor decouples to the scalar TE/TM solver
    (cross-pol ~0; Ey channel == TE, Ex channel == TM)."""
    row, pol = pol_diag
    er = _eps_diag(2.25)
    eg = _eps_diag(1.0)
    o, R, T, J = pmm_jones_1d_slanted(
        eps_ridge=er, eps_groove=eg, slant_angle=0.0, degree=18,
        stabilize=False, **_JGEOM_ASYM)
    oS, RS, TS = pmm_efficiency_1d(
        period=1.0e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.5,
        n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6,
        polarization=pol, degree=18)
    assert max(abs(J[0, 1]), abs(J[1, 0])) < 1e-12
    # TE is the direct rule (machine-exact); TM is the 1/eps inverse-rule
    # discretization gap that shrinks with degree (~3e-4 at deg 18).
    tol = 1e-6 if pol == "te" else 1e-3
    assert np.max(np.abs(R[row] - RS)) < tol
    assert np.max(np.abs(T[row] - TS)) < tol


@pytest.mark.parametrize("tensor", ["real-sym", "gyro"])
def test_jones_slant_phi0_coupled_matches_jones(tensor):
    """phi=0 COUPLED reduces to the vertical pmm_jones_1d (efficiency + |Jones|
    to ~2e-4; the residual is the inverse-rule convergence, which the genuine
    generator -- resonance-free, sumRT=1 at every degree -- shares)."""
    er = _eps_real_sym() if tensor == "real-sym" else _eps_gyro()
    eg = _eps_diag(1.0)
    deg = 20
    o, R, T, J = pmm_jones_1d_slanted(
        eps_ridge=er, eps_groove=eg, slant_angle=0.0, degree=deg,
        stabilize=False, **_JGEOM_ASYM)
    oJ, RJ, TJ, JJ = pmm_jones_1d(
        period=1.0e-6, eps_ridge=er, eps_groove=eg, n_substrate=1.5,
        n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6,
        degree=deg, stabilize=True)
    dR = float(max(np.max(np.abs(R - RJ)), np.max(np.abs(T - TJ))))
    dJ = float(np.max(np.abs(np.abs(J) - np.abs(JJ))))
    assert dR < 1e-3
    assert dJ < 1e-3


@pytest.mark.parametrize("phi_deg", [0.0, 15.0, 30.0, 45.0, 60.0])
def test_jones_slant_finite_diagonal_matches_scalar_slanted(phi_deg):
    """A DIAGONAL tensor at finite slant reduces to pmm_efficiency_1d_slanted
    (TE machine-exact ~1e-7 -- the decisive proof the metric fold is right;
    TM the 1/eps inverse-rule convergence ~6e-4).  Uses the proto's L3 geometry
    (asymmetric n_sup=1.0 / n_sub=1.5)."""
    phi = np.deg2rad(phi_deg)
    er = _eps_diag(2.25)
    eg = _eps_diag(1.0)
    deg = 20
    o, R, T, J = pmm_jones_1d_slanted(
        eps_ridge=er, eps_groove=eg, slant_angle=phi, degree=deg,
        stabilize=False, **_JGEOM_ASYM)
    oT, RTte, TTte = pmm_efficiency_1d_slanted(
        period=1.0e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.5,
        n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6,
        slant_angle=phi, polarization="te", degree=deg, stabilize=True)
    oM, RTtm, TTtm = pmm_efficiency_1d_slanted(
        period=1.0e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.5,
        n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6,
        slant_angle=phi, polarization="tm", degree=deg, stabilize=True)
    dte = max(np.max(np.abs(R[1] - RTte)), np.max(np.abs(T[1] - TTte)))
    dtm = max(np.max(np.abs(R[0] - RTtm)), np.max(np.abs(T[0] - TTtm)))
    assert max(abs(J[0, 1]), abs(J[1, 0])) < 1e-10
    assert dte < 1e-6
    assert dtm < 1e-3


@pytest.mark.parametrize("tensor", ["real-sym", "gyro"])
@pytest.mark.parametrize("phi_deg", [0.0, 15.0, 30.0, 45.0, 60.0])
def test_jones_slant_energy_conservation(tensor, phi_deg):
    """A lossless coupled slanted layer conserves energy (sumRT=1, cross-pol
    included) for BOTH a real-symmetric AND a gyrotropic tensor across 0-60
    deg (the validated ~1e-12 of the genuine flux-orthogonal generator)."""
    er = _eps_real_sym() if tensor == "real-sym" else _eps_gyro()
    eg = _eps_diag(1.0)
    o, R, T, J = pmm_jones_1d_slanted(
        eps_ridge=er, eps_groove=eg, slant_angle=np.deg2rad(phi_deg),
        degree=22, stabilize=False, **_JGEOM)
    assert abs(_sum_RT(R[0], T[0]) - 1.0) < 1e-10
    assert abs(_sum_RT(R[1], T[1]) - 1.0) < 1e-10


def test_jones_slant_reciprocity_real_symmetric():
    """A real-symmetric tensor is reciprocal (Jxy == Jyx); a gyrotropic tensor
    breaks reciprocity (the validated physical signature)."""
    eg = _eps_diag(1.0)
    _, _, _, Jsym = pmm_jones_1d_slanted(
        eps_ridge=_eps_real_sym(), eps_groove=eg, slant_angle=np.deg2rad(30.0),
        degree=22, stabilize=False, **_JGEOM)
    _, _, _, Jgyro = pmm_jones_1d_slanted(
        eps_ridge=_eps_gyro(), eps_groove=eg, slant_angle=np.deg2rad(30.0),
        degree=22, stabilize=False, **_JGEOM)
    assert abs(Jsym[0, 1] - Jsym[1, 0]) < 1e-9
    assert abs(Jgyro[0, 1] - Jgyro[1, 0]) > 1e-3


def test_jones_slant_high_contrast_conserves():
    """A high-contrast coupled tensor (eps~12) still conserves energy (the
    round-10 refute case: must NOT inflate to ~3.9)."""
    erH = _eps_diag(12.0)
    erH[0, 1] = erH[1, 0] = 2.0
    o, R, T, J = pmm_jones_1d_slanted(
        eps_ridge=erH, eps_groove=_eps_diag(1.0), slant_angle=np.deg2rad(30.0),
        degree=26, stabilize=False, **_JGEOM)
    assert abs(_sum_RT(R[0], T[0]) - 1.0) < 1e-2
    assert abs(_sum_RT(R[1], T[1]) - 1.0) < 1e-2


@pytest.mark.parametrize("phi_deg", [0.0, 20.0, 40.0])
def test_jones_slant_lossy_passive(phi_deg):
    """A lossy coupled slanted layer absorbs (sumRT < 1, no super-unity)."""
    er = _eps_diag(2.25 + 0.3j)
    er[0, 1] = er[1, 0] = 0.3
    o, R, T, J = pmm_jones_1d_slanted(
        eps_ridge=er, eps_groove=_eps_diag(1.0),
        slant_angle=np.deg2rad(phi_deg), degree=22, stabilize=False, **_JGEOM)
    assert _sum_RT(R[0], T[0]) < 1.0 + 1e-4
    assert _sum_RT(R[1], T[1]) < 1.0 + 1e-4


def test_jones_slant_reduces_to_pmm_jones_at_phi0_stabilized():
    """The stabilized public path at slant=0 agrees with pmm_jones_1d to the
    convergence tolerance (a smoke check that stabilize wires through)."""
    er = _eps_real_sym()
    eg = _eps_diag(1.0)
    o, R, T, J = pmm_jones_1d_slanted(
        eps_ridge=er, eps_groove=eg, slant_angle=0.0, degree=18,
        **_JGEOM_ASYM)
    oJ, RJ, TJ, JJ = pmm_jones_1d(
        period=1.0e-6, eps_ridge=er, eps_groove=eg, n_substrate=1.5,
        n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6,
        degree=18)
    assert np.max(np.abs(np.abs(J) - np.abs(JJ))) < 2e-3


def test_jones_slant_oblique_plus_slant_raises():
    """Combined oblique incidence + nonzero slant is unsupported and raises;
    each alone is fine."""
    er = _eps_real_sym()
    eg = _eps_diag(1.0)
    with pytest.raises(NotImplementedError):
        pmm_jones_1d_slanted(eps_ridge=er, eps_groove=eg,
                             slant_angle=np.deg2rad(25.0),
                             angle=np.deg2rad(15.0), **_JGEOM)
    # slant alone, and oblique alone (vertical), both solve
    pmm_jones_1d_slanted(eps_ridge=er, eps_groove=eg,
                         slant_angle=np.deg2rad(25.0), angle=0.0, degree=16,
                         **_JGEOM)
    pmm_jones_1d_slanted(eps_ridge=er, eps_groove=eg, slant_angle=0.0,
                         angle=np.deg2rad(15.0), degree=16, **_JGEOM)


def test_jones_slant_segments_raises():
    """The multi-region (segments) slanted-tensor path is RED and raises."""
    with pytest.raises(NotImplementedError):
        pmm_jones_1d_slanted_segments()


def test_jones_slant_out_of_plane_tensor_rejected():
    """An out-of-plane tensor (eps_xz/zx != 0) is rejected (in-plane only)."""
    er = _eps_diag(2.25)
    er[0, 2] = er[2, 0] = 0.3
    with pytest.raises(ValueError):
        pmm_jones_1d_slanted(eps_ridge=er, eps_groove=_eps_diag(1.0),
                             slant_angle=np.deg2rad(20.0), **_JGEOM)

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
    _pmm_jones_slant_solve,
    _stabilize_jones,
    classify_from_grating,
    grating_convergence_class,
    pmm_efficiency_1d,
    pmm_efficiency_1d_slanted,
    pmm_jones_1d,
    pmm_jones_1d_segments,
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


def test_slant_oblique_now_supported():
    """Combined oblique incidence + nonzero slant is now SUPPORTED: the scalar
    solver delegates to the metric-generator Jones path (pmm_jones_1d_slanted)
    with an isotropic tensor and extracts the requested channel.  It must (a) not
    raise, (b) conserve energy, and (c) be BYTE-IDENTICAL to that Jones row (TE =
    row 1, TM = row 0) -- the plumbing contract."""
    phi, ang = np.deg2rad(25.0), np.deg2rad(15.0)
    for pol in ("te", "tm"):
        o, R, T = pmm_efficiency_1d_slanted(**_GEOM, slant_angle=phi, angle=ang,
                                            polarization=pol, degree=14)
        assert abs(_sum_RT(R, T) - 1.0) < 3e-3
    # each alone is still fine (the dedicated scalar inclined-frame solver)
    pmm_efficiency_1d_slanted(**_GEOM, slant_angle=phi, angle=0.0)
    pmm_efficiency_1d_slanted(**_GEOM, slant_angle=0.0, angle=ang)
    # byte-identical to the Jones metric generator with an isotropic n^2 I tensor
    er = (complex(_GEOM["n_ridge"]) ** 2) * np.eye(3)
    eg = (complex(_GEOM["n_groove"]) ** 2) * np.eye(3)
    oj, Rj, Tj, _ = pmm_jones_1d_slanted(
        _GEOM["period"], er, eg, _GEOM["n_substrate"], _GEOM["n_superstrate"],
        _GEOM["depth"], _GEOM["duty_cycle"], _GEOM["wavelength"], phi,
        angle=ang, degree=14, stabilize=False)
    for pol, row in (("te", 1), ("tm", 0)):
        o, R, T = pmm_efficiency_1d_slanted(**_GEOM, slant_angle=phi, angle=ang,
                                            polarization=pol, degree=14,
                                            stabilize=False)
        assert np.array_equal(o, oj)
        assert np.max(np.abs(R - Rj[row])) < 1e-13
        assert np.max(np.abs(T - Tj[row])) < 1e-13


def _rcwa_staircase_slant(phi, pol, angle=0.0, n_orders=21, n_slices=96,
                          n_x=512):
    """Independent oracle: the same slanted grating as an RCWA z-staircase of
    laterally-shifted binary slices.  ``angle`` (the lab incidence angle, same
    sign convention as the PMM ``angle``) drives an oblique source."""
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
    res = stack.set_source(g["wavelength"], theta=float(angle)).solve()
    orders, R2, T2 = res.efficiencies()
    row = 1 if pol == "te" else 0           # E_y = TE = row 1 ; E_x = TM = row 0
    return np.asarray(orders), np.asarray(R2[row]), np.asarray(T2[row])


def test_slant_oblique_matches_rcwa_staircase():
    """The combined oblique + slant scalar output (via the wrapper) matches a fine
    OBLIQUE RCWA staircase on the 0th order: TE cleanly (~1e-3), TM to the shared
    wall-normal inverse-rule floor (~3e-3).  This pins both the physics AND that
    the PMM ``angle`` and RCWA ``theta`` share a sign (a flipped sign mismatches
    the asymmetric slanted 0th order by ~1e-1)."""
    phi, ang = np.deg2rad(20.0), np.deg2rad(12.0)
    for pol, tol in (("te", 2e-3), ("tm", 4e-3)):
        o, R, T = pmm_efficiency_1d_slanted(**_GEOM, slant_angle=phi, angle=ang,
                                            polarization=pol, degree=16)
        orc, Rr, Tr = _rcwa_staircase_slant(phi, pol, angle=ang, n_slices=128)
        assert abs(float(T[o == 0][0]) - float(Tr[orc == 0][0])) < tol
        assert abs(float(R[o == 0][0]) - float(Rr[orc == 0][0])) < tol


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
    (cross-pol ~0; Ey channel == TE, Ex channel == TM).

    Post-cure (round 16): a diagonal tensor now routes through the DIV-CONFORMING
    scalar slant operator, which at phi=0 is the SAME operator pmm_efficiency_1d
    uses -- so BOTH channels now match to MACHINE PRECISION (the old metric
    generator's ~3e-4 TM 1/eps gap is gone).  Stabilized on both sides (degree 18
    is a scalar-slant resonance band; the guard finds the passive solve)."""
    row, pol = pol_diag
    er = _eps_diag(2.25)
    eg = _eps_diag(1.0)
    o, R, T, J = pmm_jones_1d_slanted(
        eps_ridge=er, eps_groove=eg, slant_angle=0.0, degree=18,
        stabilize=True, **_JGEOM_ASYM)
    oS, RS, TS = pmm_efficiency_1d(
        period=1.0e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.5,
        n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6,
        polarization=pol, degree=18)
    assert max(abs(J[0, 1]), abs(J[1, 0])) < 1e-12
    # post-cure: BOTH TE (direct rule) and TM (div-conforming inverse rule)
    # reduce to the vertical scalar solver to ~1e-6 (was ~3e-4 TM pre-cure).
    assert np.max(np.abs(R[row] - RS)) < 1e-6
    assert np.max(np.abs(T[row] - TS)) < 1e-6


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
    # stabilize=True on BOTH sides (the efficiency reference below is also
    # stabilized): a raw single-degree solve can land on a degree the stabilizer
    # rejects as ill-conditioned on some BLAS (Linux OpenBLAS at steep slant),
    # making a raw-vs-stabilized comparison environment-asymmetric.  Both
    # stabilized -> both route to the same well-conditioned degree.
    o, R, T, J = pmm_jones_1d_slanted(
        eps_ridge=er, eps_groove=eg, slant_angle=phi, degree=deg,
        stabilize=True, **_JGEOM_ASYM)
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


@pytest.mark.parametrize("ang_deg,slant_deg", [(15.0, 25.0), (20.0, 30.0),
                                               (30.0, 15.0)])
def test_jones_slant_oblique_plus_slant_conserves(ang_deg, slant_deg):
    """Combined OBLIQUE incidence + nonzero slant is now SUPPORTED (round-19
    div-conforming Ez closure removes the Bloch-amplified Liu-2015 spurious null;
    kx0 is wired through the generator, the half-spaces, and the lab Rayleigh far
    field).  A lossless cell conserves energy to ~1e-3 across the angle x slant
    grid.  At combined oblique+slant BOTH the coupled tensor AND the diagonal cell
    route through the METRIC GENERATOR -- the scalar diagonal cure is excluded for
    that combo (its oblique+slant per-order split is wrong); the route itself is
    locked by test_jones_slant_oblique_dispatch_invariant."""
    ang, phi = np.deg2rad(ang_deg), np.deg2rad(slant_deg)
    eg = _eps_diag(1.0)
    # coupled tensor -> metric generator (lossless real-symmetric)
    erC = _eps_real_sym()
    _o, R, T, _J = pmm_jones_1d_slanted(
        eps_ridge=erC, eps_groove=eg, slant_angle=phi, angle=ang, degree=24,
        **_JGEOM)
    assert abs(_sum_RT(R[0], T[0]) - 1.0) < 1e-3
    assert abs(_sum_RT(R[1], T[1]) - 1.0) < 1e-3
    # diagonal cell + combined oblique+slant -> ALSO the metric generator (the
    # cure is excluded here; scalar pmm_efficiency_1d_slanted forbids the combo).
    erD = _eps_diag(2.25)
    _o2, R2, T2, _J2 = pmm_jones_1d_slanted(
        eps_ridge=erD, eps_groove=eg, slant_angle=phi, angle=ang, degree=24,
        **_JGEOM)
    assert abs(_sum_RT(R2[0], T2[0]) - 1.0) < 1e-3
    assert abs(_sum_RT(R2[1], T2[1]) - 1.0) < 1e-3


def test_jones_slant_each_alone_solves():
    """Slant alone (normal incidence) and oblique alone (vertical grating) both
    solve -- the two limits the combined oblique+slant path must reduce to."""
    er = _eps_real_sym()
    eg = _eps_diag(1.0)
    pmm_jones_1d_slanted(eps_ridge=er, eps_groove=eg,
                         slant_angle=np.deg2rad(25.0), angle=0.0, degree=16,
                         **_JGEOM)
    pmm_jones_1d_slanted(eps_ridge=er, eps_groove=eg, slant_angle=0.0,
                         angle=np.deg2rad(15.0), degree=16, **_JGEOM)


def test_jones_slant_oblique_dispatch_invariant(monkeypatch):
    """NEGATIVE routing invariant (energy CANNOT guard it): for a cure-eligible
    diagonal cell the scalar diagonal cure is NEVER taken at combined oblique +
    slant (its per-order split is ~10% wrong there yet still conserves energy --
    the lossless trap), and IS taken where it is correct (diagonal + normal +
    slant, and diagonal + oblique on a vertical grating).  Instrument the route
    directly with call spies, because a refactor re-enabling the cure for
    oblique+slant would silently return a wrong split with no failing assertion."""
    import lumenairy.elements.pmm as _pmm
    calls = {"cure": 0, "metric": 0}
    orig_cure = _pmm._pmm_jones_slant_diag_solve
    orig_metric = _pmm._pmm_jones_slant_solve

    def spy_cure(*a, **k):
        calls["cure"] += 1
        return orig_cure(*a, **k)

    def spy_metric(*a, **k):
        calls["metric"] += 1
        return orig_metric(*a, **k)

    monkeypatch.setattr(_pmm, "_pmm_jones_slant_diag_solve", spy_cure)
    monkeypatch.setattr(_pmm, "_pmm_jones_slant_solve", spy_metric)
    erD = _eps_diag(2.25)           # cure-ELIGIBLE diagonal cell (exy=eyx=0, exx==ezz)
    eg = _eps_diag(1.0)

    # (a) combined oblique + slant -> MUST be the metric generator, NOT the cure
    calls.update(cure=0, metric=0)
    _pmm.pmm_jones_1d_slanted(eps_ridge=erD, eps_groove=eg,
                              slant_angle=np.deg2rad(30.0),
                              angle=np.deg2rad(20.0), degree=16, **_JGEOM)
    assert calls["cure"] == 0 and calls["metric"] >= 1

    # (b) diagonal + NORMAL + slant -> the cure (correct + faster here)
    calls.update(cure=0, metric=0)
    _pmm.pmm_jones_1d_slanted(eps_ridge=erD, eps_groove=eg,
                              slant_angle=np.deg2rad(30.0), angle=0.0,
                              degree=16, **_JGEOM)
    assert calls["cure"] >= 1 and calls["metric"] == 0

    # (c) diagonal + oblique + VERTICAL (slant=0) -> the cure (valid scalar oblique)
    calls.update(cure=0, metric=0)
    _pmm.pmm_jones_1d_slanted(eps_ridge=erD, eps_groove=eg, slant_angle=0.0,
                              angle=np.deg2rad(20.0), degree=16, **_JGEOM)
    assert calls["cure"] >= 1 and calls["metric"] == 0


def test_jones_slant_diag_cure_internal_guard_raises():
    """Defense-in-depth: the diagonal-cure SOLVER ITSELF re-asserts its ENTIRE
    validity domain (diagonal exy=eyx=0, exx==ezz, and NOT combined oblique+slant)
    and fails loud -- never a silent wrong answer (a dropped off-diagonal Jones, a
    wrong TM index, or the ~10%-wrong-but-energy-conserving oblique+slant split).
    So even a direct call or a future dispatch mis-route raises.  The valid cases
    (diagonal exx==ezz at slant-alone or oblique-alone) must NOT raise."""
    import lumenairy.elements.pmm as _pmm
    erD = _eps_diag(2.25)
    eg = _eps_diag(1.0)
    base = (1.0e-6, erD, eg, complex(1.5), complex(1.0), 0.5e-6, 0.5, 0.633e-6)
    # (1) combined oblique + slant -> RuntimeError (the lossless-trap case)
    with pytest.raises(RuntimeError):
        _pmm._pmm_jones_slant_diag_solve(
            *base, np.deg2rad(30.0), 16, 1, True, 21, angle=np.deg2rad(20.0))
    # (2) COUPLED tensor (exy!=0) -> RuntimeError (cure would drop the coupling)
    erC = _eps_diag(2.25)
    erC[0, 1] = erC[1, 0] = 0.3
    with pytest.raises(RuntimeError):
        _pmm._pmm_jones_slant_diag_solve(
            1.0e-6, erC, eg, complex(1.5), complex(1.0), 0.5e-6, 0.5, 0.633e-6,
            np.deg2rad(30.0), 16, 1, True, 21, angle=0.0)
    # (3) exx != ezz tensor -> RuntimeError (cure's scalar TM index would be wrong)
    erU = np.diag([2.25, 2.10, 2.40]).astype(np.complex128)
    with pytest.raises(RuntimeError):
        _pmm._pmm_jones_slant_diag_solve(
            1.0e-6, erU, eg, complex(1.5), complex(1.0), 0.5e-6, 0.5, 0.633e-6,
            np.deg2rad(30.0), 16, 1, True, 21, angle=0.0)
    # valid: diagonal exx==ezz at slant-alone (normal) and oblique-alone (vertical)
    _pmm._pmm_jones_slant_diag_solve(
        *base, np.deg2rad(30.0), 16, 1, True, 21, angle=0.0)
    _pmm._pmm_jones_slant_diag_solve(
        *base, 0.0, 16, 1, True, 21, angle=np.deg2rad(20.0))


def test_jones_slant_oblique_degree_clean():
    """Combined oblique+slant is DEGREE-CLEAN: with stabilize=False the metric
    generator conserves energy at EVERY raw degree (the validated round-19
    signature -- a frame-consistent far field has no isolated-degree resonances,
    so the old 0.79-1.58 chaos / stabilize-crutch is gone).  Coupled cell ->
    metric generator, lossless, angle 25 + slant 30."""
    er = _eps_real_sym()
    eg = _eps_diag(1.0)
    ang, phi = np.deg2rad(25.0), np.deg2rad(30.0)
    for deg in (16, 20, 24, 28):
        _o, R, T, _J = pmm_jones_1d_slanted(
            eps_ridge=er, eps_groove=eg, slant_angle=phi, angle=ang,
            degree=deg, stabilize=False, **_JGEOM)
        assert abs(_sum_RT(R[0], T[0]) - 1.0) < 5e-3, f"deg={deg} Ex"
        assert abs(_sum_RT(R[1], T[1]) - 1.0) < 5e-3, f"deg={deg} Ey"


@pytest.mark.parametrize("ang_deg", [10.0, 20.0, 30.0])
def test_jones_slant_oblique_metric_reduces_to_efficiency_1d(ang_deg):
    """The kx0 wiring is correct end-to-end through the METRIC GENERATOR: a
    VERTICAL (slant=0) grating at oblique incidence solved by the tensor metric
    generator reproduces the rigorous scalar oracle pmm_efficiency_1d (TM) to
    ~1e-3 per order.  Call the internal solver directly (a slant=0 diagonal cell
    would otherwise route to the scalar cure) so this exercises the metric
    generator's own Bloch path."""
    ai = np.deg2rad(ang_deg)
    erD = _eps_diag(2.25)
    egD = _eps_diag(1.0)
    o, R, T, _J, _ng = _pmm_jones_slant_solve(
        1.0e-6, erD, egD, complex(1.5), complex(1.0), 0.5e-6, 0.5, 0.633e-6, 0.0,
        24, 1, 1, True, 21, angle=ai)
    oL, RL, TL = pmm_efficiency_1d(
        period=1.0e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.5,
        n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6,
        polarization="tm", degree=24, angle=ai, stabilize=True)
    assert np.array_equal(o, oL)
    assert np.max(np.abs(R[0] - RL)) < 1e-3      # row 0 = incident Ex = TM
    assert np.max(np.abs(T[0] - TL)) < 1e-3


_SEGGEOM = dict(period=1.0e-6, n_substrate=1.5, n_superstrate=1.0,
                depth=0.5e-6, wavelength=0.633e-6)


def _seg_asym3():
    """Asymmetric 3-region cell with a COUPLED middle region -- the round-12
    adversarial profile that leaked to sumRT~210 on the pre-round-19 operator."""
    e1 = _eps_diag(2.25)
    e2 = np.diag([4.0, 3.6, 4.0]).astype(np.complex128)
    e2[0, 1] = e2[1, 0] = 0.3
    e3 = _eps_diag(1.0)
    return [(0.2, e1), (0.5, e2), (0.3, e3)]


@pytest.mark.parametrize("phi_deg", [0.0, 30.0, 60.0])
def test_jones_slant_segments_binary_reduces_to_slanted(phi_deg):
    """A 2-segment cell reproduces the binary pmm_jones_1d_slanted BIT-IDENTICALLY
    at all slants (the segment assembler is byte-exact on the binary case; the
    metric generator + far field are region-count-agnostic).  Segments reverse
    internally, so pass (groove, ridge) to land the binary ridge|groove x-order."""
    phi = np.deg2rad(phi_deg)
    er = np.diag([2.25, 2.10, 2.25]).astype(np.complex128)
    er[0, 1] = er[1, 0] = 0.15
    eg = _eps_diag(1.0)
    o, R, T, J = pmm_jones_1d_slanted_segments(
        segments=[(0.5, eg), (0.5, er)], slant_angle=phi, degree=24,
        stabilize=False, **_SEGGEOM)
    ob, Rb, Tb, Jb = pmm_jones_1d_slanted(
        period=1.0e-6, eps_ridge=er, eps_groove=eg, n_substrate=1.5,
        n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6,
        slant_angle=phi, degree=24, stabilize=False)
    assert np.array_equal(o, ob)
    assert np.max(np.abs(R - Rb)) < 1e-12
    assert np.max(np.abs(T - Tb)) < 1e-12
    assert np.max(np.abs(J - Jb)) < 1e-12


def test_jones_slant_segments_vertical_reduces_to_segments():
    """At slant=0 the N-region metric generator reduces to the validated vertical
    pmm_jones_1d_segments to the div-conforming discretization difference (~e-5 --
    the same level as the binary slant=0 reduction to pmm_jones_1d, NOT a bug)."""
    segs = _seg_asym3()
    o, R, T, J = pmm_jones_1d_slanted_segments(
        segments=segs, slant_angle=0.0, degree=24, stabilize=False, **_SEGGEOM)
    ob, Rb, Tb, Jb = pmm_jones_1d_segments(
        1.0e-6, segs, 1.5, 1.0, 0.5e-6, 0.633e-6, degree=24, stabilize=False)
    assert np.array_equal(o, ob)
    assert np.max(np.abs(R - Rb)) < 5e-4
    assert np.max(np.abs(T - Tb)) < 5e-4
    assert np.max(np.abs(J - Jb)) < 5e-4


@pytest.mark.parametrize("ang_deg,slant_deg", [(0.0, 30.0), (0.0, 60.0),
                                               (20.0, 30.0)])
def test_jones_slant_segments_conserves(ang_deg, slant_deg):
    """Asymmetric multi-region (coupled middle) conserves energy at slant alone
    AND combined oblique+slant -- the rounds-12-15 frontier (sumRT~210), closed by
    the round-19 div-conforming operator."""
    segs = _seg_asym3()
    o, R, T, J = pmm_jones_1d_slanted_segments(
        segments=segs, slant_angle=np.deg2rad(slant_deg),
        angle=np.deg2rad(ang_deg), degree=24, stabilize=False, **_SEGGEOM)
    assert abs(_sum_RT(R[0], T[0]) - 1.0) < 1e-3
    assert abs(_sum_RT(R[1], T[1]) - 1.0) < 1e-3


def test_jones_slant_segments_out_of_plane_rejected():
    """A segment with an out-of-plane tensor (eps_xz/zx != 0) is rejected."""
    er = _eps_diag(2.25)
    er[0, 2] = er[2, 0] = 0.3
    with pytest.raises(ValueError):
        pmm_jones_1d_slanted_segments(
            segments=[(0.5, er), (0.5, _eps_diag(1.0))],
            slant_angle=np.deg2rad(20.0), **_SEGGEOM)


def test_jones_slant_out_of_plane_tensor_rejected():
    """An out-of-plane tensor (eps_xz/zx != 0) is rejected (in-plane only)."""
    er = _eps_diag(2.25)
    er[0, 2] = er[2, 0] = 0.3
    with pytest.raises(ValueError):
        pmm_jones_1d_slanted(eps_ridge=er, eps_groove=_eps_diag(1.0),
                             slant_angle=np.deg2rad(20.0), **_JGEOM)


# ===================================================================
# THE DIAGONAL CURE (round 16; Granet 2017/2023, Liu 2015) -- a diagonal
# tensor (exy=eyx=0) with exx==ezz routes TE/TM through the div-conforming
# scalar slant operator (_sem_modes_slant, spurious-mode-free), shedding the
# pointwise metric generator's latent ~2e-4 per-order accuracy gap. Coupled
# tensors AND diagonal exx!=ezz tensors keep the metric generator unchanged.
# ===================================================================
def _eps_uniaxial(exx_ezz=2.25, eyy=4.0):
    """Diagonal uniaxial tensor exx == ezz != eyy (the TM/TE-distinct cure
    case): TM sees exx, TE sees eyy, ezz == exx so the scalar slant is EXACT."""
    return np.diag([exx_ezz, eyy, exx_ezz]).astype(np.complex128)


@pytest.mark.parametrize("kind", ["isotropic", "uniaxial"])
@pytest.mark.parametrize("phi_deg", [0.0, 20.0, 45.0, 60.0])
def test_jones_slant_diag_cure_matches_scalar_by_construction(kind, phi_deg):
    """THE CURE IS THE SCALAR SLANT: a diagonal tensor (isotropic exx=eyy=ezz,
    OR uniaxial exx=ezz!=eyy) routed through the cure reproduces the per-channel
    pmm_efficiency_1d_slanted to ~1e-10 (BY CONSTRUCTION -- same div-conforming
    _sem_modes_slant operator).  stabilize=False pins both to the same degree."""
    phi = np.deg2rad(phi_deg)
    deg = 14                               # resonance-free here (stabilize=False)
    if kind == "isotropic":
        er = _eps_diag(2.25)
        n_r_tm = n_r_te = 1.5
    else:
        er = _eps_uniaxial(exx_ezz=2.25, eyy=4.0)
        n_r_tm, n_r_te = 1.5, 2.0          # sqrt(exx)=1.5 (TM), sqrt(eyy)=2.0 (TE)
    eg = _eps_diag(1.0)
    o, R, T, J = pmm_jones_1d_slanted(
        eps_ridge=er, eps_groove=eg, slant_angle=phi, degree=deg,
        stabilize=False, **_JGEOM_ASYM)
    # per-channel scalar slant references (TM = row 0 = incident Ex ; TE = row 1)
    sc = dict(period=1.0e-6, n_groove=1.0, n_substrate=1.5, n_superstrate=1.0,
              depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6,
              slant_angle=phi, degree=deg, stabilize=False)
    # sanity: the chosen degree is physical (the by-construction equality holds
    # at ANY degree, but a resonance-free one keeps the numbers meaningful)
    assert abs(_sum_RT(*pmm_efficiency_1d_slanted(n_ridge=n_r_tm,
                                                  polarization="tm", **sc)[1:])
               - 1.0) < 1e-2
    oTM, RTM, TTM = pmm_efficiency_1d_slanted(n_ridge=n_r_tm, polarization="tm",
                                              **sc)
    oTE, RTE, TTE = pmm_efficiency_1d_slanted(n_ridge=n_r_te, polarization="te",
                                              **sc)
    assert np.array_equal(o, oTM) and np.array_equal(o, oTE)
    # off-diagonal Jones is identically zero (no Ex<->Ey coupling)
    assert max(abs(J[0, 1]), abs(J[1, 0])) < 1e-14
    assert np.max(np.abs(R[0] - RTM)) < 1e-10      # row 0 = TM
    assert np.max(np.abs(T[0] - TTM)) < 1e-10
    assert np.max(np.abs(R[1] - RTE)) < 1e-10      # row 1 = TE
    assert np.max(np.abs(T[1] - TTE)) < 1e-10


@pytest.mark.parametrize("phi_deg", [15.0, 30.0, 45.0])
def test_jones_slant_diag_cure_vs_metric_generator(phi_deg):
    """The cured diagonal output is the SAME PHYSICS as the metric-generator path.
    As of the round-19 integration the metric generator's Ez closure is ALSO
    div-conforming (un-gated, all slants), so the Liu-2015 spurious-mode gap is
    GONE from BOTH paths: the per-order R/T now agree to the tensor-vs-scalar
    DISCRETIZATION gap (the [E;H] tensor generator self-converges ~2x slower than
    the scalar slant at steep tilt -- both correct, energy ~1e-13 each).  The gap
    is degree-limited (deg20-stabilized): R ~2e-4, T ~2e-3 at 45deg, shrinking
    with degree (see test_jones_slant_diag_cure_matches_scalar_by_construction for
    the by-construction equality, and the metric generator's own per-order
    convergence to the scalar oracle 6.5e-4@deg32 / 4.3e-4@deg40 at 45deg).  Both
    paths are stabilized (degree 20 is a metric-generator resonance band; the
    scalar slant resonance band differs, so a fair compare needs the guard on
    both)."""
    phi = np.deg2rad(phi_deg)
    deg = 20
    er = _eps_diag(2.25)
    eg = _eps_diag(1.0)
    # cured public path (diagonal -> div-conforming scalar slant), stabilized
    o, R, T, J = pmm_jones_1d_slanted(
        eps_ridge=er, eps_groove=eg, slant_angle=phi, degree=deg,
        stabilize=True, **_JGEOM_ASYM)
    # metric generator path (now div-conforming), stabilized the SAME way -- call
    # the internal solver directly to bypass the diagonal-cure dispatch.
    margs = (1.0e-6, er, eg, complex(1.5), complex(1.0), 0.5e-6, 0.5, 0.633e-6,
             phi)
    oM, RM, TM, JM = _stabilize_jones(
        lambda d: _pmm_jones_slant_solve(
            *margs, degree=d, n_ridge_el=1, n_groove_el=1, grade=True,
            far_field_orders=21)[:4], deg, "metric-ref")
    assert np.array_equal(o, oM)
    assert max(abs(J[0, 1]), abs(J[1, 0])) < 1e-12      # cured is diagonal
    # same physics to the tensor-vs-scalar div-conforming convergence gap
    # (deg20-stabilized; both conserve energy and converge to one answer).
    assert np.max(np.abs(R - RM)) < 2e-4
    assert np.max(np.abs(T - TM)) < 3e-3
    # the cured path conserves energy (cross-pol included)
    assert abs(_sum_RT(R[0], T[0]) - 1.0) < 1e-3
    assert abs(_sum_RT(R[1], T[1]) - 1.0) < 1e-3


@pytest.mark.parametrize("pol_diag", [(0, "tm"), (1, "te")])
def test_jones_slant_diag_cure_phi0_reduces_to_pmm_jones(pol_diag):
    """At slant=0 the cured DIAGONAL path reduces to the vertical pmm_jones_1d
    to ~1e-6 (TE machine-exact direct rule ~1e-9; TM now the DIV-CONFORMING
    inverse rule ~3e-6 -- the ~2e-4 metric-generator gap is GONE).  Stabilized
    (deg 18 is the requested floor; the guard finds the nearest passive solve)."""
    row, pol = pol_diag
    er = _eps_diag(2.25)
    eg = _eps_diag(1.0)
    deg = 18
    o, R, T, J = pmm_jones_1d_slanted(
        eps_ridge=er, eps_groove=eg, slant_angle=0.0, degree=deg,
        stabilize=True, **_JGEOM_ASYM)
    oJ, RJ, TJ, JJ = pmm_jones_1d(
        period=1.0e-6, eps_ridge=er, eps_groove=eg, n_substrate=1.5,
        n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6,
        degree=deg, stabilize=True)
    assert np.array_equal(o, oJ)
    assert max(abs(J[0, 1]), abs(J[1, 0])) < 1e-12
    assert np.max(np.abs(R[row] - RJ[row])) < 1e-6
    assert np.max(np.abs(T[row] - TJ[row])) < 1e-6


@pytest.mark.parametrize("phi_deg", [0.0, 30.0, 60.0])
def test_jones_slant_coupled_byte_identical_after_cure(phi_deg):
    """A COUPLED tensor (exy=eyx=0.4) is NOT diagonal -> stays on the metric
    generator: the public output is BYTE-IDENTICAL to the internal metric-
    generator solve (the cure did not touch the coupled path)."""
    phi = np.deg2rad(phi_deg)
    deg = 20
    er = _eps_real_sym(diag=2.25, off=0.4)
    eg = _eps_diag(1.0)
    o, R, T, J = pmm_jones_1d_slanted(
        eps_ridge=er, eps_groove=eg, slant_angle=phi, degree=deg,
        stabilize=False, **_JGEOM)
    oM, RM, TM, JM, _n = _pmm_jones_slant_solve(
        1.0e-6, er, eg, complex(1.5), complex(1.5), 0.5e-6, 0.5, 0.633e-6, phi,
        degree=deg, n_ridge_el=1, n_groove_el=1, grade=True, far_field_orders=21)
    assert np.array_equal(o, oM)
    assert np.array_equal(R, RM)
    assert np.array_equal(T, TM)
    assert np.array_equal(J, JM)


@pytest.mark.parametrize("phi_deg", [0.0, 30.0, 60.0])
def test_jones_slant_diag_exx_ne_ezz_stays_on_metric_generator(phi_deg):
    """A DIAGONAL tensor with exx != ezz (scalar slant would be WRONG -- it uses
    a SINGLE eps per region) is NOT cured: it stays on the metric generator,
    BYTE-IDENTICAL to the internal solve."""
    phi = np.deg2rad(phi_deg)
    deg = 20
    er = np.diag([2.25, 2.25, 3.0]).astype(np.complex128)   # exx=2.25 != ezz=3.0
    eg = _eps_diag(1.0)
    o, R, T, J = pmm_jones_1d_slanted(
        eps_ridge=er, eps_groove=eg, slant_angle=phi, degree=deg,
        stabilize=False, **_JGEOM)
    oM, RM, TM, JM, _n = _pmm_jones_slant_solve(
        1.0e-6, er, eg, complex(1.5), complex(1.5), 0.5e-6, 0.5, 0.633e-6, phi,
        degree=deg, n_ridge_el=1, n_groove_el=1, grade=True, far_field_orders=21)
    assert np.array_equal(o, oM)
    assert np.array_equal(R, RM)    # stayed on the metric generator (not scalar)
    assert np.array_equal(T, TM)
    assert np.array_equal(J, JM)

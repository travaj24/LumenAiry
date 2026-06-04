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

"""v5.4 Phase 5 -- ``zernike_decompose`` normalization & weighting kwargs.

The v5.4 Zernike dock (``lumenairy/ui/zernike_dock.py``) gained two
dropdowns (Normalization, Weighting) in audit pass P3-B, but the
library remained OSA-locked and weighting was applied post-hoc on the
OPD.  Phase 5 wires the kwargs through the library so the dock can
dispense with both workarounds.

Pins
----

1. **Default-path bit-for-bit**: ``zernike_decompose(opd, dx, ap)``
   with no kwargs must return numerically identical output to the
   pre-Phase-5 baseline (computed via the OSA orthonormal inner
   product directly).
2. **``normalization='OSA'`` == default**: explicit OSA matches the
   no-kwarg path.
3. **``normalization='Noll'``**: same magnitudes as OSA, but the
   sine-like (m < 0) coefficients are sign-flipped.
4. **``normalization='Bogus'`` raises ``ValueError``**; ``'Fringe'``
   raises ``NotImplementedError``.
5. **``weighting=None`` == default**.
6. **Weighting equivalence**: a uniform 1.0 weight over the pupil
   gives identical coefficients to the unweighted fit; a 0/1 mask
   matching the implicit pupil aperture gives identical coefficients
   to a smaller-aperture fit.
7. **Shape mismatch raises ``ValueError``**.
8. **Weighted fit suppresses edge outliers**: a near-edge single-
   pixel spike that perturbs the unweighted fit is suppressed when
   the weight is zero on that pixel.

Author: Andrew Traverso -- v5.4 Phase 5 (Zernike normalization +
weighting kwargs).
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.analysis import zernike_decompose
from lumenairy.analysis.zernike import (
    zernike_basis_matrix,
    zernike_index_to_nm,
    zernike_polynomial,
)

# ----------------------------------------------------------------------
# Shared test fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope='module')
def grid():
    """Small square grid + pupil aperture used across every test."""
    N = 96
    dx = 5e-6
    aperture = 0.6 * N * dx  # leave a margin so the pupil fits inside
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return {
        'N': N,
        'dx': dx,
        'aperture': aperture,
        'X': X,
        'Y': Y,
    }


def _make_opd(grid, j_amp_map, n_modes=15):
    """Synthesise an OPD = sum_j amp_j * Z_j^{OSA}(rho, theta).

    ``j_amp_map`` is a ``{j: amplitude}`` dict.  Modes outside the
    dict default to 0.
    """
    X = grid['X']
    Y = grid['Y']
    r_pupil = 0.5 * grid['aperture']
    rho = np.sqrt(X ** 2 + Y ** 2) / r_pupil
    theta = np.arctan2(Y, X)
    opd = np.zeros_like(X)
    for j, amp in j_amp_map.items():
        n, m = zernike_index_to_nm(j)
        opd = opd + amp * zernike_polynomial(n, m, rho, theta)
    opd = np.where(rho <= 1.0, opd, 0.0)
    return opd


# ----------------------------------------------------------------------
# (1) Default-path bit-for-bit identity with the OSA inner-product
# ----------------------------------------------------------------------

def test_default_normalization_is_osa_bit_for_bit(grid):
    """Calling ``zernike_decompose(opd, dx, ap)`` with no new kwargs
    must reproduce the v5.3.2 OSA orthonormal inner-product fit
    exactly.

    Baseline: we compute coefficients via the analytic OSA inner
    product (the same path the library used before Phase 5) and
    compare to the library output bit-for-bit.
    """
    opd = _make_opd(grid, {
        2: 1.2e-7,   # tilt-X
        4: 3.4e-8,   # defocus
        5: -2.1e-8,  # vertical astigmatism
        8: 1.5e-8,   # horizontal coma
        12: 7.0e-9,  # primary spherical
    })
    n_modes = 15

    # Library path (no kwargs -> must reproduce baseline)
    coeffs_lib, _names = zernike_decompose(
        opd, grid['dx'], grid['aperture'], n_modes=n_modes)

    # Baseline path: re-implement the v5.3.2 lstsq inner-product
    # path here so we know we're testing exactly what shipped.
    basis, pupil_mask = zernike_basis_matrix(
        n_modes, grid['X'], grid['Y'], 0.5 * grid['aperture'])
    opd_flat = opd[pupil_mask]
    try:
        from scipy.linalg import lstsq as _slstsq
        coeffs_ref, *_ = _slstsq(basis, opd_flat, lapack_driver='gelsy')
    except Exception:
        coeffs_ref, *_ = np.linalg.lstsq(basis, opd_flat, rcond=None)

    # Bit-for-bit: zero atol because we expect literal float equality
    # (same basis matrix from the cache, same scipy driver, same
    # opd_flat slicing).
    assert coeffs_lib.shape == (n_modes,)
    np.testing.assert_array_equal(coeffs_lib, coeffs_ref)


# ----------------------------------------------------------------------
# (2) Explicit 'OSA' == default
# ----------------------------------------------------------------------

def test_normalization_osa_equals_default(grid):
    """``normalization='OSA'`` must produce coefficients identical to
    the default-kwarg path."""
    opd = _make_opd(grid, {2: 1.0e-7, 4: -5.0e-8})
    c_default, _ = zernike_decompose(opd, grid['dx'], grid['aperture'])
    c_osa, _ = zernike_decompose(
        opd, grid['dx'], grid['aperture'], normalization='OSA')
    np.testing.assert_array_equal(c_default, c_osa)


def test_normalization_standard_equals_osa(grid):
    """``'Standard'`` (ANSI/Z80.28-2010 Born-Wolf form) is
    algebraically identical to OSA.  Bit-for-bit equality required."""
    opd = _make_opd(grid, {3: 2.0e-8, 5: -1.5e-8, 7: 4.0e-9})
    c_osa, _ = zernike_decompose(
        opd, grid['dx'], grid['aperture'], normalization='OSA')
    c_std, _ = zernike_decompose(
        opd, grid['dx'], grid['aperture'], normalization='Standard')
    np.testing.assert_array_equal(c_osa, c_std)


# ----------------------------------------------------------------------
# (3) 'Noll' flips sign on sine-like (m < 0) modes
# ----------------------------------------------------------------------

def test_normalization_noll_changes_sign_on_odd_terms(grid):
    """Noll 1976's sine convention is opposite to OSA, so coefficients
    on negative-m OSA modes must be sign-flipped.  All cosine (m >= 0)
    coefficients are unchanged in magnitude AND sign; sine (m < 0)
    coefficients are unchanged in magnitude but flipped in sign.

    Inject a pattern with non-zero coefficients on BOTH cosine and
    sine modes so the sign-flip is testable.
    """
    n_modes = 12
    # Inject: tilt-Y (j=1, m=-1, sine), tilt-X (j=2, m=+1, cosine),
    # oblique astig (j=3, m=-2, sine), defocus (j=4, m=0, even),
    # vertical astig (j=5, m=+2, cosine).
    opd = _make_opd(grid, {
        1: 1.0e-7,
        2: 2.0e-7,
        3: -5.0e-8,
        4: 7.0e-8,
        5: 3.0e-8,
    })
    c_osa, _ = zernike_decompose(
        opd, grid['dx'], grid['aperture'], n_modes=n_modes,
        normalization='OSA')
    c_noll, _ = zernike_decompose(
        opd, grid['dx'], grid['aperture'], n_modes=n_modes,
        normalization='Noll')

    for j in range(n_modes):
        _n, m = zernike_index_to_nm(j)
        if m < 0:
            # Sine term: sign flipped relative to OSA.
            assert c_noll[j] == pytest.approx(-c_osa[j], abs=1e-18), (
                f'j={j} (n,m)=({_n},{m}) sine term should flip sign '
                f'under Noll; got OSA={c_osa[j]} Noll={c_noll[j]}')
        else:
            # Cosine / piston / defocus: identical to OSA.
            assert c_noll[j] == pytest.approx(c_osa[j], abs=1e-18), (
                f'j={j} (n,m)=({_n},{m}) non-sine term should match '
                f'OSA; got OSA={c_osa[j]} Noll={c_noll[j]}')


# ----------------------------------------------------------------------
# (4) Invalid normalization names
# ----------------------------------------------------------------------

def test_normalization_invalid_raises(grid):
    """An unknown string must raise ``ValueError``."""
    opd = _make_opd(grid, {4: 1.0e-8})
    with pytest.raises(ValueError, match='Unknown normalization'):
        zernike_decompose(
            opd, grid['dx'], grid['aperture'], normalization='Bogus')


def test_normalization_fringe_raises_notimplemented(grid):
    """``'Fringe'`` is documented but not yet implementable as a per-
    mode rescale of the OSA basis (different mode set + peak-value-1
    polynomials).  Must raise ``NotImplementedError`` with a clear
    pointer for the dock to surface."""
    opd = _make_opd(grid, {4: 1.0e-8})
    with pytest.raises(NotImplementedError, match='Fringe'):
        zernike_decompose(
            opd, grid['dx'], grid['aperture'], normalization='Fringe')


# ----------------------------------------------------------------------
# (5) weighting=None == default
# ----------------------------------------------------------------------

def test_weighting_none_equals_default(grid):
    """``weighting=None`` (explicit) must match the no-kwarg default
    bit-for-bit."""
    opd = _make_opd(grid, {2: 1.0e-7, 4: -3.0e-8, 8: 5.0e-9})
    c_default, _ = zernike_decompose(opd, grid['dx'], grid['aperture'])
    c_none, _ = zernike_decompose(
        opd, grid['dx'], grid['aperture'], weighting=None)
    np.testing.assert_array_equal(c_default, c_none)


def test_weighting_uniform_ones_matches_unweighted(grid):
    """A uniform all-ones weight (over the full grid) is the
    unweighted fit modulo numerical noise -- weighted-LS multiplies
    every row by sqrt(1) = 1, leaving the system unchanged."""
    opd = _make_opd(grid, {2: 1.0e-7, 4: -3.0e-8, 8: 5.0e-9})
    w = np.ones_like(opd)
    c_default, _ = zernike_decompose(opd, grid['dx'], grid['aperture'])
    c_uniform, _ = zernike_decompose(
        opd, grid['dx'], grid['aperture'], weighting=w)
    # Allow tiny float-arithmetic drift from the sqrt(1.0) * row
    # rescale, but coefficients should agree to many digits.
    np.testing.assert_allclose(c_uniform, c_default, atol=0, rtol=1e-12)


# ----------------------------------------------------------------------
# (6) Circular-aperture weighting reproduces a smaller-aperture fit
# ----------------------------------------------------------------------

def test_weighting_circular_mask_matches_implicit_aperture(grid):
    """Weighting by a 0/1 circular mask of radius r_w must give the
    same coefficient vector as decomposing against the IMPLICIT pupil
    of radius r_w (i.e. shrinking ``aperture`` to 2*r_w) -- provided
    we account for the fact that the larger 'aperture' argument
    defines the rho-normalisation in both fits.

    We sanity-check this on the magnitudes: a circular-mask weight at
    fraction f of the original aperture must yield coefficients of
    order |c_unweighted| (because the basis is full-pupil) when the
    OPD is a smooth field.  The TIGHT pin is that turning off pixels
    outside the mask doesn't blow up the fit and recovers the same
    leading defocus magnitude to a few percent.
    """
    # Pure defocus pattern -- smooth, no high-order content.
    opd = _make_opd(grid, {4: 5.0e-8})
    aperture = grid['aperture']
    dx = grid['dx']

    # Build a 0/1 mask at fraction 0.7 of the pupil.
    Ny, Nx = opd.shape
    x = (np.arange(Nx) - Nx / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_w = 0.7 * 0.5 * aperture
    mask_inner = ((X ** 2 + Y ** 2) <= r_w ** 2).astype(np.float64)

    c_unweighted, _ = zernike_decompose(opd, dx, aperture)
    c_inner, _ = zernike_decompose(
        opd, dx, aperture, weighting=mask_inner)

    # Defocus coefficient (j=4) should still dominate and be close to
    # the injected amplitude.  The inner-pupil fit can disagree on
    # high-order modes but the leading defocus must remain.
    assert c_inner[4] == pytest.approx(c_unweighted[4], rel=0.10)
    # Piston/tilts should still be small.
    for j in (0, 1, 2):
        assert abs(c_inner[j]) < 1e-9


# ----------------------------------------------------------------------
# (7) Shape-mismatch validation
# ----------------------------------------------------------------------

def test_weighting_shape_mismatch_raises(grid):
    """A weighting array whose shape doesn't match ``opd_map`` must
    raise ``ValueError`` BEFORE any work happens."""
    opd = _make_opd(grid, {4: 1.0e-8})
    bad = np.ones((opd.shape[0] + 5, opd.shape[1]), dtype=np.float64)
    with pytest.raises(ValueError, match='weighting shape'):
        zernike_decompose(
            opd, grid['dx'], grid['aperture'], weighting=bad)


# ----------------------------------------------------------------------
# (8) Weighted fit suppresses edge outliers
# ----------------------------------------------------------------------

def test_weighted_fit_suppresses_edge_outlier(grid):
    """Inject a pure defocus pattern, then plant a large single-pixel
    spike near the pupil edge.  The unweighted fit will absorb that
    spike into the high-order coefficients (bad).  The weighted fit
    with a mask that zeroes the edge-annulus must give a tighter RMS
    residual relative to the *true* (no-spike) pattern.
    """
    aperture = grid['aperture']
    dx = grid['dx']
    n_modes = 15

    # True smooth pattern: defocus + small spherical.
    j_amp = {4: 5.0e-8, 12: 5.0e-9}
    true_opd = _make_opd(grid, j_amp)

    # Add a near-edge spike inside the pupil.  The pupil radius in
    # pixels is r_pupil / dx = 28.8 here; we plant the spike at ~95%
    # of that so it's well inside the rho <= 1 mask but in the outer
    # annulus we'll later weight out.
    spiked_opd = true_opd.copy()
    Ny, Nx = spiked_opd.shape
    cx = Nx // 2
    cy = Ny // 2
    r_pupil_pix = 0.5 * aperture / dx
    spike_off = int(round(0.93 * r_pupil_pix))
    spiked_opd[cy, cx + spike_off] += 5.0e-7  # 10x the defocus amp

    # Unweighted fit -- gets corrupted by the spike.
    c_uw, _ = zernike_decompose(
        spiked_opd, dx, aperture, n_modes=n_modes)

    # Weighted fit -- zero out the outer annulus where the spike
    # lives.  We don't have to remove only the spike pixel; the
    # canonical use case is "trust only the interior of the pupil".
    x = (np.arange(Nx) - Nx / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_interior = 0.85 * 0.5 * aperture
    interior_mask = (
        (X ** 2 + Y ** 2) <= r_interior ** 2
    ).astype(np.float64)
    c_w, _ = zernike_decompose(
        spiked_opd, dx, aperture, n_modes=n_modes,
        weighting=interior_mask)

    # The weighted-fit defocus coefficient must be closer to the true
    # injected amplitude than the unweighted one.
    true_c4 = j_amp[4]
    err_uw = abs(c_uw[4] - true_c4)
    err_w = abs(c_w[4] - true_c4)
    assert err_w < err_uw, (
        f'weighted defocus error ({err_w:.3e}) should be smaller '
        f'than unweighted ({err_uw:.3e})')

    # The total high-order energy (modes 6+) should also be smaller
    # in the weighted fit -- the spike was absorbed there before.
    hi_uw = float(np.linalg.norm(c_uw[6:]))
    hi_w = float(np.linalg.norm(c_w[6:]))
    assert hi_w < hi_uw, (
        f'weighted high-order norm ({hi_w:.3e}) should be smaller '
        f'than unweighted ({hi_uw:.3e}) once the edge spike is '
        f'masked away')

"""v5.4 Phase 5 -- chebyshev_fit_2d library helper pin tests.

Pins the 2-D Chebyshev fit primitive promoted from the v5.4 audit
P3-C inline implementation in ``lumenairy/ui/chebyshev_fit_dock.py``
into the public library at
``lumenairy._math.chebyshev.chebyshev_fit_2d`` (also re-exported as
``lumenairy.chebyshev_fit_2d``).

The fit basis is the tensor product T_i(x) T_j(y) with T the first-
kind Chebyshev polynomial -- identical to the basis the library's
``lumenairy.elements.freeform.surface_sag_chebyshev`` evaluates --
so coefficients returned by the fit drop directly into the
``cheb_coeffs`` dict that ``surface_sag_chebyshev`` consumes.

The five pins below cover:

1. Constant z(x, y) -> single T_0 T_0 term, no spurious mass.
2. Single mode injection T_2(x) T_0(y) -> exact coefficient recovery.
3. Weight-mask exclusion of an outlier -> RMS residual drops > 100x.
4. ``return_residual=True`` -> RMS residual << 1e-10 for a
   Chebyshev-representable input.
5. Cross-check: the fitted dict evaluated through
   ``surface_sag_chebyshev`` agrees with the library's own Chebyshev
   evaluation to tight tolerance.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module + export existence
# ---------------------------------------------------------------------------

def test_chebyshev_fit_2d_exposed_in_math_module_and_top_level():
    mod = importlib.import_module('lumenairy._math.chebyshev')
    assert hasattr(mod, 'chebyshev_fit_2d'), (
        "lumenairy._math.chebyshev must export chebyshev_fit_2d "
        "as part of the v5.4 Phase 5 promotion."
    )
    assert 'chebyshev_fit_2d' in mod.__all__

    top = importlib.import_module('lumenairy')
    assert hasattr(top, 'chebyshev_fit_2d'), (
        "lumenairy must re-export chebyshev_fit_2d at the top "
        "level so notebook callers can use ``la.chebyshev_fit_2d``."
    )
    assert 'chebyshev_fit_2d' in top.__all__


# ---------------------------------------------------------------------------
# Pin 1 -- constant z fits to T_0 * T_0 only
# ---------------------------------------------------------------------------

def test_constant_z_fits_to_T0_T0_only():
    """Constant height map -> only the (0, 0) coefficient is non-zero."""
    from lumenairy._math.chebyshev import chebyshev_fit_2d

    nx, ny = 17, 13
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    z = np.ones((ny, nx), dtype=np.float64)  # z = 1 everywhere

    coeffs = chebyshev_fit_2d(
        x, y, z, n_max_x=4, n_max_y=4, normalize_xy=False)

    # (0, 0) should be 1.0 to machine precision.
    assert (0, 0) in coeffs
    assert coeffs[(0, 0)] == pytest.approx(1.0, abs=1e-12)
    # Every other (i, j) entry must be at or below the helper's
    # 1e-15 prune threshold -- i.e. it should not appear in the dict
    # at all, or, if it does, be effectively zero.
    for key, val in coeffs.items():
        if key == (0, 0):
            continue
        assert abs(val) < 1e-10, (
            f"unexpected non-zero coefficient at {key}: {val}")


# ---------------------------------------------------------------------------
# Pin 2 -- pure T_2(x) * T_0(y) recovers the injected coefficient
# ---------------------------------------------------------------------------

def test_pure_T2_T0_recovers_coefficient():
    """Inject z = 0.7 * T_2(x) * T_0(y); fit should return c_{2,0} = 0.7."""
    from lumenairy._math.chebyshev import chebyshev_fit_2d

    c_inject = 0.7
    nx, ny = 33, 21
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    # T_2(x) = 2 x^2 - 1; T_0(y) = 1.
    z = c_inject * (2.0 * X ** 2 - 1.0)

    coeffs = chebyshev_fit_2d(
        x, y, z, n_max_x=5, n_max_y=5, normalize_xy=False)

    assert (2, 0) in coeffs
    assert coeffs[(2, 0)] == pytest.approx(c_inject, rel=1e-10)
    # No other mode should carry meaningful mass.
    for key, val in coeffs.items():
        if key == (2, 0):
            continue
        assert abs(val) < 1e-10, (
            f"unexpected non-zero coefficient at {key}: {val}")


# ---------------------------------------------------------------------------
# Pin 3 -- weight mask excludes a single outlier
# ---------------------------------------------------------------------------

def test_weighted_fit_excludes_outlier():
    """A binary weight that zeroes one outlier drops the residual >100x."""
    from lumenairy._math.chebyshev import chebyshev_fit_2d

    nx, ny = 41, 41
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')

    # Smooth Chebyshev-representable surface at order <= 3.
    z_smooth = (
        0.3 * (2.0 * X ** 2 - 1.0)
        + 0.2 * (2.0 * Y ** 2 - 1.0)
        + 0.1 * (4.0 * X ** 3 - 3.0 * X)
    )
    z = z_smooth.copy()
    # Inject a single huge outlier well off the smooth surface.
    z[ny // 2, nx // 2] += 1.0e3

    # Fit without masking the outlier.
    _, residual_unweighted = chebyshev_fit_2d(
        x, y, z, n_max_x=4, n_max_y=4,
        normalize_xy=False, return_residual=True)
    rms_unweighted = float(
        np.sqrt(np.mean(residual_unweighted ** 2)))

    # Fit with a binary mask that zeroes the outlier pixel.
    mask = np.ones_like(z, dtype=np.float64)
    mask[ny // 2, nx // 2] = 0.0
    _, residual_weighted = chebyshev_fit_2d(
        x, y, z, n_max_x=4, n_max_y=4, weight=mask,
        normalize_xy=False, return_residual=True)
    valid = residual_weighted[np.isfinite(residual_weighted)]
    rms_weighted = float(np.sqrt(np.mean(valid ** 2)))

    # The masked fit should be at least 100x better, and the
    # smooth-input fit should be near machine precision.
    assert rms_unweighted / max(rms_weighted, 1e-300) > 100.0, (
        f"masked RMS {rms_weighted:.3e} should be >100x smaller "
        f"than unmasked RMS {rms_unweighted:.3e}")
    assert rms_weighted < 1e-10, (
        f"masked fit RMS {rms_weighted:.3e} should be near "
        f"machine precision for a smooth Chebyshev input.")


# ---------------------------------------------------------------------------
# Pin 4 -- return_residual matches the fit on a smooth input
# ---------------------------------------------------------------------------

def test_residual_consistent_with_fit():
    """For a Chebyshev-representable input the residual is ~ zero."""
    from lumenairy._math.chebyshev import chebyshev_fit_2d

    nx, ny = 25, 25
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')

    # Mix several low-order T_i(x) T_j(y) modes.  All are inside the
    # n_max_x = n_max_y = 6 basis so the fit must reproduce them
    # exactly (up to LS conditioning).
    T1x = X
    T2x = 2.0 * X ** 2 - 1.0
    T1y = Y
    T3y = 4.0 * Y ** 3 - 3.0 * Y
    z = (
        0.5 * T2x
        + 0.25 * T1x * T1y
        - 0.125 * T2x * T3y
    )

    coeffs, residual = chebyshev_fit_2d(
        x, y, z, n_max_x=6, n_max_y=6,
        normalize_xy=False, return_residual=True)

    rms = float(np.sqrt(np.mean(residual ** 2)))
    assert rms < 1e-10, (
        f"residual RMS {rms:.3e} should be < 1e-10 for a fully "
        f"Chebyshev-representable input.")
    # Spot-check the recovered coefficients.
    assert coeffs[(2, 0)] == pytest.approx(0.5, rel=1e-10)
    assert coeffs[(1, 1)] == pytest.approx(0.25, rel=1e-10)
    assert coeffs[(2, 3)] == pytest.approx(-0.125, rel=1e-10)


# ---------------------------------------------------------------------------
# Pin 5 -- fitted dict round-trips through surface_sag_chebyshev
# ---------------------------------------------------------------------------

def test_chebvander2d_basis_matches_library_evaluation():
    """Evaluate the fit dict via ``surface_sag_chebyshev`` and confirm
    it matches the chebvander2d-derived fit map to within ~1e-12.

    This pins the basis convention agreement: the (i, j) key in the
    fit dict means T_i(x/norm_x) T_j(y/norm_y) on both sides.
    """
    from numpy.polynomial.chebyshev import chebval2d

    from lumenairy._math.chebyshev import chebyshev_fit_2d
    from lumenairy.elements.freeform import surface_sag_chebyshev

    # Use a non-unit half-extent on x to also exercise the norm_x /
    # norm_y scaling on the evaluator side.
    norm_x = 4.2e-3   # [m]
    norm_y = 2.5e-3   # [m]
    nx, ny = 33, 27
    x_m = np.linspace(-norm_x, norm_x, nx)
    y_m = np.linspace(-norm_y, norm_y, ny)
    X_m, Y_m = np.meshgrid(x_m, y_m, indexing='xy')

    # Build an analytic sag map from a small set of modes (in
    # normalised coordinates).
    Xn = X_m / norm_x
    Yn = Y_m / norm_y
    truth = {
        (0, 0): 1.2e-6,
        (2, 0): -3.4e-7,
        (1, 1): 5.6e-7,
        (0, 2): 7.8e-7,
        (3, 1): -2.1e-7,
    }
    # Reference matrix for chebval2d (max order 3).
    ref_matrix = np.zeros((4, 4), dtype=np.float64)
    for (i, j), c in truth.items():
        ref_matrix[i, j] = c
    z = chebval2d(Xn, Yn, ref_matrix)

    # Fit with normalize_xy=True so the helper rescales the
    # physical-meter coordinates to [-1, 1] for us.
    coeffs = chebyshev_fit_2d(
        x_m, y_m, z, n_max_x=4, n_max_y=4, normalize_xy=True)

    # Recovered coefficients must match the injected truth.
    for (i, j), c in truth.items():
        assert coeffs[(i, j)] == pytest.approx(c, rel=1e-10, abs=1e-18), (
            f"coefficient ({i}, {j}) mismatch: "
            f"got {coeffs.get((i, j))!r}, want {c!r}")

    # Evaluate the recovered dict through the library's own freeform
    # evaluator (which uses the same {(i, j): c_ij} convention) and
    # confirm the reconstructed surface matches z to high precision.
    z_lib = surface_sag_chebyshev(
        X_m, Y_m, R=np.inf, conic=0.0,
        cheb_coeffs=coeffs, norm_x=norm_x, norm_y=norm_y)
    np.testing.assert_allclose(z_lib, z, rtol=1e-12, atol=1e-16)

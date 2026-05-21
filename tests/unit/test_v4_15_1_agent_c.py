"""Unit tests for v4.15.1 Agent C analysis-module patch.

Covers the v4.15.0 audit's analytical-value pin meta-finding and
four P1 bug fixes in ``lumenairy.analysis.core``:

* **C.1 (P1-NEW-A)** -- ``sparrow_resolution`` rewritten to the
  canonical two-source dip-vanishing condition via brentq root-
  finding on the cubic-spline second derivative of the PSF radial
  profile.  Replaces the v4.15.0 ``np.gradient``-twice + zero-cross
  approach that returned 15% low on clean Airy at f/#=4.
* **C.2 (P1-F1-3)** -- ``strehl_vector`` default-reference branch
  removed.  An explicit ``reference=`` is now required (breaking
  change).
* **C.3 (P1-F1-4)** -- ``rayleigh_resolution`` returns NaN +
  ``RuntimeWarning`` on Gaussian-like PSFs without a true first-ring
  minimum, instead of a spurious value at the float-underflow plateau.
* **C.4 (P1-F1-5)** -- ``astigmatism_mag_angle`` docstring range
  corrected from (-pi/4, pi/4] to the actual (-pi/2, pi/2].
* **C.5** -- resolution functions accept ``dy`` and thread ``_xp_of``
  for backend dispatch.

Author: Andrew Traverso -- v4.15.1 / Agent C
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.analysis.core import (
    astigmatism_mag_angle,
    fwhm_resolution,
    rayleigh_resolution,
    sparrow_resolution,
    strehl_vector,
)

# ============================================================================
# Helpers
# ============================================================================

def _airy_psf_analytic(N: int, dx: float, wavelength: float,
                      f_number: float) -> np.ndarray:
    """Analytic 2-D Airy PSF on an N x N grid for analytical pinning.

    Avoids the discrete sampling / oversampling quirks of
    :func:`compute_psf` and gives a noise-free reference for the
    closed-form Sparrow / FWHM constants.
    """
    from scipy.special import j1
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X ** 2 + Y ** 2)
    arg = np.pi * r / (wavelength * f_number)
    psf = np.where(
        arg < 1e-10, 1.0,
        (2.0 * j1(arg) / np.maximum(arg, 1e-30)) ** 2)
    return psf


def _gaussian_psf(N: int, dx: float, w0: float) -> np.ndarray:
    """Symmetric Gaussian PSF with 1/e^2 radius w0.  Has NO true
    first-ring zero in the radial profile."""
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-2.0 * (X ** 2 + Y ** 2) / w0 ** 2)


def _make_diffraction_limited_reference(N: int, dx: float,
                                        D: float, wavelength: float,
                                        focal_length: float):
    """Return (Ex_ref, Ey_ref) where Ex_ref is the amplitude of the
    aperture's diffraction-limited focal spot."""
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    aperture = ((X ** 2 + Y ** 2) <= (D / 2) ** 2).astype(np.complex128)
    psf, _ = la.compute_psf(
        aperture, wavelength, focal_length, dx, oversample=2)
    Ex_ref = np.sqrt(psf).astype(np.complex128)
    Ey_ref = np.zeros_like(Ex_ref)
    return Ex_ref, Ey_ref


# ============================================================================
# C.1 -- sparrow_resolution canonical Airy pin
# ============================================================================

class TestSparrowResolutionFix:
    """v4.15.1 (P1-NEW-A): canonical Sparrow algorithm with analytical
    pins on a clean Airy PSF."""

    def test_sparrow_resolution_airy_analytical(self):
        """Sparrow on the analytic Airy at lambda=600 nm, f/#=4 must
        match the canonical ``0.947 * lambda * f/# = 2.273 um`` to
        within 1%.  v4.15.0 returned 1.93 um (15% low) on the same
        config; v4.15.1 returns 2.27 um (~0.02% error -- well within
        the v4.15.2 tightened 1% tolerance).

        v4.15.2 (AUDIT_V4_15_1 P3 closure): tolerance tightened from
        5% to 1% to match the achievable accuracy and align with the
        in-test docstring claim of "<1% error".  Empirical
        measurement on a 256-pixel grid at 0.1-um pitch shows the
        cubic-spline-second-derivative + brentq path recovers the
        canonical constant to 0.0173% -- well below the new 1% pin.
        """
        wavelength = 600e-9
        f_number = 4.0
        # Properly-sampled grid: dx_psf well below first-zero radius.
        psf = _airy_psf_analytic(N=256, dx=0.1e-6,
                                  wavelength=wavelength,
                                  f_number=f_number)
        d_sparrow = sparrow_resolution(psf, 0.1e-6, axis='radial')
        d_expected = 0.947 * wavelength * f_number
        rel = abs(d_sparrow - d_expected) / d_expected
        assert rel < 0.01, (
            f"sparrow_resolution on analytic Airy: got "
            f"{d_sparrow*1e6:.4f} um, expected {d_expected*1e6:.4f} um "
            f"(0.947 * lam * f/#); rel error {rel*100:.2f}% > 1% "
            f"(v4.15.2 tolerance).")

    def test_sparrow_resolution_below_rayleigh(self):
        """Sanity-check property: Sparrow < Rayleigh on the same Airy
        PSF.  Carryover from v4.15.0 but kept as a sibling pin on top
        of the analytical-value pin (test ``test_sparrow_resolution_
        airy_analytical``).  Sparrow ~0.947 vs Rayleigh ~1.22."""
        wavelength = 600e-9
        psf = _airy_psf_analytic(N=256, dx=0.1e-6,
                                  wavelength=wavelength, f_number=4.0)
        d_sparrow = sparrow_resolution(psf, 0.1e-6, axis='radial')
        d_rayleigh = rayleigh_resolution(
            psf, 0.1e-6, wavelength, axis='radial')
        assert d_sparrow < d_rayleigh, (
            f"Sparrow {d_sparrow*1e6:.4f} um must be smaller than "
            f"Rayleigh {d_rayleigh*1e6:.4f} um for clean Airy.")

    def test_sparrow_resolution_x_and_radial_agree_on_symmetric_psf(
            self):
        """For a rotationally-symmetric Airy PSF, axis='x' and
        axis='radial' must agree to within 5%."""
        wavelength = 600e-9
        psf = _airy_psf_analytic(N=256, dx=0.1e-6,
                                  wavelength=wavelength, f_number=4.0)
        d_x = sparrow_resolution(psf, 0.1e-6, axis='x')
        d_r = sparrow_resolution(psf, 0.1e-6, axis='radial')
        rel = abs(d_x - d_r) / max(d_x, d_r)
        assert rel < 0.05, (
            f"sparrow_resolution axis='x' vs 'radial' on symmetric "
            f"Airy disagree by {rel*100:.2f}%: x={d_x*1e6:.4f} um, "
            f"radial={d_r*1e6:.4f} um.")


# ============================================================================
# C.2 -- strehl_vector explicit reference required
# ============================================================================

class TestStrehlVectorBreakingChange:
    """v4.15.1 (P1-F1-3): default-reference branch removed."""

    def test_strehl_vector_explicit_reference_required(self):
        """Omitting ``reference`` must raise (ValueError, or the
        TypeError that Python raises for the missing keyword-only
        argument)."""
        N = 16
        Ex = np.ones((N, N), dtype=np.complex128)
        Ey = np.zeros_like(Ex)
        with pytest.raises((ValueError, TypeError),
                            match="reference"):
            strehl_vector(Ex, Ey)

    def test_strehl_vector_with_diffraction_limited_reference(self):
        """A focused PSF compared against its aberration-free
        diffraction-limited reference gives Strehl <= 1.0 (i.e.
        no Strehl > 1 silent bug)."""
        N = 32
        dx = 50e-6
        D = 1e-3
        wavelength = 600e-9
        focal_length = 50e-3
        Ex_ref, Ey_ref = _make_diffraction_limited_reference(
            N, dx, D, wavelength, focal_length)
        s = strehl_vector(
            Ex_ref.copy(), Ey_ref.copy(),
            reference=(Ex_ref, Ey_ref))
        assert abs(s - 1.0) < 1e-12, (
            f"strehl_vector(unaberrated, ref=unaberrated) = {s!r}; "
            f"expected 1.0.")

    def test_strehl_vector_aberrated_under_unity(self):
        """A defocused field must have Strehl strictly less than the
        unaberrated reference's 1.0.  Previously the default-reference
        path returned Strehl > 1 silently."""
        N_pupil = 64
        dx = 25e-6
        D = 1e-3
        wavelength = 600e-9
        focal_length = 30e-3
        x = (np.arange(N_pupil) - N_pupil / 2) * dx
        X, Y = np.meshgrid(x, x)
        r2 = X ** 2 + Y ** 2
        aperture = (r2 <= (D / 2) ** 2).astype(np.complex128)
        psf_ref, _ = la.compute_psf(
            aperture, wavelength, focal_length, dx, oversample=2)
        Ex_ref = np.sqrt(psf_ref).astype(np.complex128)
        Ey_ref = np.zeros_like(Ex_ref)
        # Defocus pupil
        r_norm2 = r2 / ((D / 2) ** 2)
        phase = 0.25 * 2 * np.pi * (2.0 * r_norm2 - 1.0)
        pupil_aber = aperture * np.exp(1j * phase)
        psf_aber, _ = la.compute_psf(
            pupil_aber, wavelength, focal_length, dx, oversample=2)
        Ex_aber = np.sqrt(psf_aber).astype(np.complex128)
        Ey_aber = np.zeros_like(Ex_aber)
        s = strehl_vector(Ex_aber, Ey_aber,
                          reference=(Ex_ref, Ey_ref))
        assert 0.0 <= s < 1.0, (
            f"strehl_vector(defocus=0.25 waves) = {s!r}; "
            f"expected 0 <= S < 1.")


# ============================================================================
# C.3 -- rayleigh_resolution Gaussian-PSF false-positive fix
# ============================================================================

class TestRayleighGaussianFalsePositiveFix:
    """v4.15.1 (P1-F1-4): Gaussian PSFs without a true first-ring zero
    return NaN + RuntimeWarning."""

    def test_rayleigh_resolution_gaussian_returns_nan_with_warning(
            self):
        """A narrow Gaussian (which underflows to a zero floor in the
        wings) used to return a spurious positive value where the
        underflow plateau looked like a local minimum.  Post-fix:
        NaN return + RuntimeWarning emit."""
        # Narrow enough that the wings underflow to zero in float64
        psf = _gaussian_psf(N=256, dx=0.1e-6, w0=0.5e-6)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = rayleigh_resolution(psf, 0.1e-6, 600e-9, axis='radial')
        assert np.isnan(val), (
            f"rayleigh_resolution(Gaussian no-zero) = {val!r}; "
            f"expected NaN.")
        warn_msgs = [str(ww.message).lower() for ww in w
                      if issubclass(ww.category, RuntimeWarning)]
        assert any('true' in m or 'gaussian' in m or 'fwhm' in m
                    for m in warn_msgs), (
            f"rayleigh_resolution on Gaussian should emit "
            f"RuntimeWarning suggesting fwhm/sparrow_resolution; "
            f"got warnings: {warn_msgs}.")

    def test_rayleigh_resolution_airy_still_works(self):
        """The fix must not regress the Airy case: a real first-zero
        is still detected and returns near 1.22*lambda*f/#."""
        wavelength = 600e-9
        psf = _airy_psf_analytic(N=256, dx=0.1e-6,
                                  wavelength=wavelength, f_number=4.0)
        val = rayleigh_resolution(psf, 0.1e-6, wavelength, axis='radial')
        expected = 1.22 * wavelength * 4.0
        rel = abs(val - expected) / expected
        assert rel < 0.05, (
            f"rayleigh_resolution on Airy regression: got "
            f"{val*1e6:.4f} um, expected {expected*1e6:.4f} um.")


# ============================================================================
# C.4 -- astigmatism_mag_angle docstring range fix
# ============================================================================

class TestAstigmatismMagAngleRangeFix:
    """v4.15.1 (P1-F1-5): docstring claimed (-pi/4, pi/4]; actual
    range is (-pi/2, pi/2]."""

    def test_astigmatism_mag_angle_range(self):
        """Sample (c3, c5) over a full circle and verify that the
        observed theta range is (-pi/2, pi/2]."""
        thetas = []
        rng = np.random.default_rng(42)
        for _ in range(500):
            c = np.zeros(6, dtype=np.float64)
            c[3] = rng.uniform(-1.0, 1.0)
            c[5] = rng.uniform(-1.0, 1.0)
            _mag, theta = astigmatism_mag_angle(c)
            thetas.append(theta)
        thetas = np.asarray(thetas, dtype=np.float64)
        # All in (-pi/2, pi/2] strictly: lower bound > -pi/2 exclusive
        # (since atan2(0, -1) = pi -> theta = pi/2, never -pi/2).
        assert thetas.min() > -np.pi / 2 - 1e-12, (
            f"observed theta minimum {thetas.min()!r} below "
            f"(-pi/2, pi/2] range.")
        assert thetas.max() <= np.pi / 2 + 1e-12, (
            f"observed theta maximum {thetas.max()!r} above "
            f"(-pi/2, pi/2] range.")
        # Edge case: pure c5=-1, c3=0 -> atan2(0, -1) = pi -> 0.5*pi = pi/2
        c = np.zeros(6)
        c[5] = -1.0
        _mag, theta_edge = astigmatism_mag_angle(c)
        assert abs(theta_edge - np.pi / 2) < 1e-12, (
            f"(c3=0, c5=-1) edge case: theta = {theta_edge!r}, "
            f"expected pi/2.")

    def test_astigmatism_mag_angle_docstring_range_matches(self):
        """The docstring must state the correct theta range
        (-pi/2, pi/2] -- not the v4.15.0 erroneous (-pi/4, pi/4].
        This is a regression pin: any future docstring drift that
        re-introduces the wrong range fails this test."""
        doc = inspect.getdoc(astigmatism_mag_angle) or ''
        doc_lower = doc.lower()
        # Should mention (-pi/2, pi/2] (in TeX) or `pi/2`.
        assert ('pi/2' in doc_lower or
                r'\pi/2' in doc or
                'pi}/2' in doc_lower), (
            "astigmatism_mag_angle docstring should mention the "
            "actual theta range of (-pi/2, pi/2].")
        # Must NOT claim the old wrong range (-pi/4, pi/4]:
        assert not ('(-pi/4, pi/4]' in doc_lower or
                     r'(-\pi/4, \pi/4]' in doc or
                     r'(-\pi/4,\, \pi/4]' in doc), (
            "astigmatism_mag_angle docstring still claims the "
            "v4.15.0 wrong range (-pi/4, pi/4]; v4.15.1 corrects "
            "this to (-pi/2, pi/2].")


# ============================================================================
# C.5 -- resolution functions thread dy and _xp_of
# ============================================================================

class TestResolutionDyAndXpDispatch:
    """v4.15.1 P3 follow-ups: ``dy`` kwarg + ``_xp_of`` backend
    dispatch on all three resolution functions."""

    def test_resolution_functions_thread_dy(self):
        """``dy=None`` (default) and ``dy=dx`` must give identical
        results on a square grid; ``dy != dx`` (anamorphic) must
        produce a different radial profile."""
        wavelength = 600e-9
        psf = _airy_psf_analytic(N=256, dx=0.1e-6,
                                  wavelength=wavelength, f_number=4.0)

        # Square grid: dy=None matches dy=dx exactly
        r_no_dy = rayleigh_resolution(
            psf, 0.1e-6, wavelength, axis='radial')
        r_dy_eq = rayleigh_resolution(
            psf, 0.1e-6, wavelength, axis='radial', dy=0.1e-6)
        s_no_dy = sparrow_resolution(
            psf, 0.1e-6, axis='radial')
        s_dy_eq = sparrow_resolution(
            psf, 0.1e-6, axis='radial', dy=0.1e-6)
        f_no_dy = fwhm_resolution(psf, 0.1e-6, axis='radial')
        f_dy_eq = fwhm_resolution(psf, 0.1e-6, axis='radial', dy=0.1e-6)

        assert r_no_dy == r_dy_eq, (
            f"rayleigh_resolution: dy=None vs dy=dx differ: "
            f"{r_no_dy!r} vs {r_dy_eq!r}.")
        assert s_no_dy == s_dy_eq, (
            f"sparrow_resolution: dy=None vs dy=dx differ: "
            f"{s_no_dy!r} vs {s_dy_eq!r}.")
        assert f_no_dy == f_dy_eq, (
            f"fwhm_resolution: dy=None vs dy=dx differ: "
            f"{f_no_dy!r} vs {f_dy_eq!r}.")

        # Anamorphic: dy = 2 * dx changes the radial metric
        # measurably (a Sparrow value on the anamorphic radial
        # profile is NOT the same as on the isotropic one).
        s_anamorphic = sparrow_resolution(
            psf, 0.1e-6, axis='radial', dy=0.2e-6)
        # We don't pin the numerical value (no analytical reference
        # exists for anamorphic Sparrow on an isotropic Airy PSF);
        # we only require that the dispatch actually consumes dy.
        assert s_anamorphic != s_no_dy, (
            f"sparrow_resolution: dy=2*dx anamorphic returned the "
            f"same value {s_anamorphic!r} as the isotropic case "
            f"{s_no_dy!r}; dy threading appears broken.")

    def test_resolution_functions_xp_dispatch(self):
        """The functions must call ``_xp_of`` for backend dispatch.
        On numpy-only test environments this is a smoke test
        (consuming a numpy array through the dispatch path).  If a
        CuPy / JAX backend is available we additionally verify the
        function tolerates a non-numpy array.
        """
        wavelength = 600e-9
        psf = _airy_psf_analytic(N=128, dx=0.2e-6,
                                  wavelength=wavelength, f_number=4.0)

        # Smoke test: numpy backend completes without raising
        s_np = sparrow_resolution(psf, 0.2e-6, axis='radial')
        r_np = rayleigh_resolution(psf, 0.2e-6, wavelength,
                                    axis='radial')
        f_np = fwhm_resolution(psf, 0.2e-6, axis='radial')
        assert np.isfinite(s_np)
        assert np.isfinite(r_np)
        assert np.isfinite(f_np)

        # If CuPy is importable, run the same PSF through the cupy
        # path and verify the dispatch works without error.  If
        # CuPy is not available (the default), this is a no-op skip.
        try:
            import cupy  # noqa: F401
        except Exception:
            pytest.skip("CuPy not available; numpy-only smoke test.")
        else:  # pragma: no cover -- exercised only on CuPy environments
            import cupy as cp
            psf_cp = cp.asarray(psf)
            s_cp = sparrow_resolution(psf_cp, 0.2e-6, axis='radial')
            r_cp = rayleigh_resolution(psf_cp, 0.2e-6, wavelength,
                                        axis='radial')
            f_cp = fwhm_resolution(psf_cp, 0.2e-6, axis='radial')
            assert abs(s_cp - s_np) / s_np < 1e-6
            assert abs(r_cp - r_np) / r_np < 1e-6
            assert abs(f_cp - f_np) / f_np < 1e-6


# ============================================================================
# Audit recommendation 9c -- drop Richards-Wolf claim from strehl_vector docstring
# ============================================================================

class TestStrehlVectorDocstringSoftening:
    """v4.15.1 audit recommendation 9c: ``strehl_vector`` docstring no
    longer asserts a verified Richards-Wolf high-NA Ez behaviour."""

    def test_strehl_vector_no_richards_wolf_claim_in_docstring(self):
        """The Richards-Wolf / high-NA claim was never verified in
        the v4.15.0 release.  v4.15.1 softens the docstring to
        'scalar Strehl-like metric over vector field components'."""
        doc = inspect.getdoc(strehl_vector) or ''
        doc_lower = doc.lower()
        assert 'richards-wolf' not in doc_lower and \
                'richards wolf' not in doc_lower, (
            f"strehl_vector docstring contains unverified "
            f"Richards-Wolf claim; v4.15.1 drops this language.\n"
            f"Docstring: {doc[:600]!r}")
        # 'high-NA' alone is OK as a use-case mention, but the
        # 'high-NA imaging where the longitudinal component is non-
        # negligible' claim from v4.15.0 is removed.
        assert 'as needed for high-na' not in doc_lower, (
            f"strehl_vector docstring still asserts 'as needed for "
            f"high-NA / Richards-Wolf imaging'; v4.15.1 softens "
            f"this.\nDocstring: {doc[:600]!r}")

"""Unit tests for v4.15.0 Agent C analysis-module public API expansion.

Covers four ROADMAP v4.16 user-facing API gaps closed in v4.15.0:

* **C.1** -- ``ee_polychromatic`` convenience wrapper around
  :func:`polychromatic_psf` + :func:`encircled_energy_radius`.
* **C.2** -- ``strehl_vector`` (vector Strehl with optional Ez) and
  ``coupling_efficiency_vector`` (vector overlap integral).
* **C.3** -- ``rayleigh_resolution``, ``sparrow_resolution``,
  ``fwhm_resolution`` two-point-separability metrics.
* **C.4** -- ``astigmatism_mag_angle`` Mahajan-convention helper for
  primary astigmatism magnitude + orientation from an OSA-indexed
  Zernike coefficient array.

v4.15.1 -- per audit ``AUDIT_V4_15_0_2026_05_18.md`` meta-finding
("property-only test pins miss algorithm-correctness regressions")
the three resolution-metric property pins (sparrow < rayleigh,
fwhm < rayleigh) are upgraded to analytical-value pins against the
canonical Airy constants (0.947 / 1.029 / 1.22 * lambda * f/#) at
+/- 5%.  ``strehl_vector`` no longer supports a default plane-wave
reference (audit P1-F1-3); the previously-tautological "unity for
plane wave" pin is replaced with an explicit-reference round-trip
pin + a breaking-change pin that the default path now raises.  See
``test_v4_15_1_agent_c.py`` for the v4.15.1 patch tests proper.

Convention pins (documented in :mod:`lumenairy.analysis.core`):

* ``ee_polychromatic`` accepts ``N`` / ``output_grid`` and ``dx`` /
  ``output_dx`` aliases.  Passing both forms raises ``ValueError``.
* ``strehl_vector`` reduces to the scalar peak-ratio Strehl when
  ``Ey == Ez == 0``.  ``reference=`` is required (v4.15.1 breaking
  change; the v4.15.0 default plane-wave reference returned
  Strehl > 1 for any focused PSF).
* Resolution metrics return distances in metres; ``rayleigh`` is the
  first-zero radius, ``sparrow`` is the dual-source separation,
  ``fwhm`` is the FWHM diameter (twice the half-max radius).
* ``astigmatism_mag_angle`` uses OSA / ANSI indexing as
  :func:`zernike_decompose`: ``coeffs[3] = c3 = oblique``,
  ``coeffs[5] = c5 = vertical``; ``theta = 0.5 * atan2(c3, c5)``.

Author: Andrew Traverso -- v4.15.0 / Agent C (v4.15.1 patch updates)
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.analysis.core import (
    astigmatism_mag_angle,
    coupling_efficiency_vector,
    ee_polychromatic,
    fwhm_resolution,
    rayleigh_resolution,
    sparrow_resolution,
    strehl_vector,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_singlet_for_ee():
    """Modest singlet used as the prescription input for ee_polychromatic
    tests.  Numbers are chosen so the resulting focal grid fits in 32x32
    pupil samples (fast)."""
    return la.make_singlet(
        R1=50e-3, R2=float('inf'), d=2e-3,
        glass='N-BK7', aperture=10e-3,
    )


def _make_airy_psf(
    *,
    N_pupil: int = 128,
    dx_pupil: float = 50e-6,
    D_aperture: float = 4e-3,
    wavelength: float = 600e-9,
    focal_length: float = 80e-3,
    oversample: int = 8,
):
    """Synthesize a clean diffraction-limited Airy PSF via
    :func:`compute_psf` on a circular aperture.  Returns
    ``(psf, dx_psf, f_number)``."""
    x = (np.arange(N_pupil) - N_pupil / 2) * dx_pupil
    X, Y = np.meshgrid(x, x)
    aperture = ((X ** 2 + Y ** 2) <= (D_aperture / 2) ** 2).astype(
        np.complex128)
    psf, dx_psf = la.compute_psf(
        aperture, wavelength, focal_length, dx_pupil,
        oversample=oversample, normalize='power')
    f_number = focal_length / D_aperture
    return psf, dx_psf, f_number


# ============================================================================
# C.1 -- ee_polychromatic
# ============================================================================

class TestEePolychromatic:
    """Convenience wrapper for polychromatic encircled-energy curves."""

    def test_c1_identical_wavelengths_matches_polychromatic_psf_path(self):
        """When all entries in ``wavelengths`` are identical and the
        weights are equal, ``ee_polychromatic`` must produce the same
        curve as running ``polychromatic_psf`` followed by
        ``encircled_energy_curve`` -- the wrapper performs no
        additional smoothing or rebinning.
        """
        rx = _make_singlet_for_ee()
        wls = [1.30e-6, 1.30e-6, 1.30e-6]
        wts = [1.0, 1.0, 1.0]
        radii = np.array([3e-6, 6e-6, 12e-6, 25e-6, 50e-6])
        N, dx = 32, 5e-6
        r_out, ee_out = ee_polychromatic(
            rx, wls, wts, radii, output_grid=N, output_dx=dx)

        psf_ref, dx_ref, _ = la.polychromatic_psf(
            rx, wls, [1.0 / 3, 1.0 / 3, 1.0 / 3], N, dx,
            normalize='power')
        psf_amp = np.sqrt(np.asarray(psf_ref)).astype(np.complex128)
        r_ref, ee_ref = la.encircled_energy_curve(
            psf_amp, dx_ref, radii=radii)
        assert np.allclose(r_out, r_ref, atol=1e-18), (
            f"ee_polychromatic radii diverge from reference: "
            f"{r_out!r} vs {r_ref!r}.")
        assert np.allclose(ee_out, ee_ref, atol=1e-10), (
            f"ee_polychromatic encircled-energy diverges from "
            f"polychromatic_psf + encircled_energy_curve path: "
            f"max |diff| = {float(np.max(np.abs(ee_out - ee_ref)))!r}.")
        # Monotone non-decreasing -- a basic invariant of encircled
        # energy.
        assert np.all(np.diff(ee_out) >= -1e-12), (
            f"ee_polychromatic curve is not monotonically "
            f"non-decreasing: diffs = {np.diff(ee_out)!r}.")

    def test_c1_weights_normalisation_is_scale_invariant(self):
        """Multiplying the weights by an arbitrary positive constant
        must not change the ee curve (the wrapper re-normalises
        internally)."""
        rx = _make_singlet_for_ee()
        wls = [1.30e-6, 1.55e-6]
        radii = np.array([5e-6, 10e-6, 20e-6, 50e-6])
        N, dx = 32, 5e-6
        r1, ee1 = ee_polychromatic(
            rx, wls, [1.0, 1.0], radii, output_grid=N, output_dx=dx)
        r2, ee2 = ee_polychromatic(
            rx, wls, [100.0, 100.0], radii, output_grid=N, output_dx=dx)
        r3, ee3 = ee_polychromatic(
            rx, wls, [0.5, 0.5], radii, output_grid=N, output_dx=dx)
        assert np.allclose(ee1, ee2, atol=1e-10) and \
            np.allclose(ee1, ee3, atol=1e-10), (
            f"ee_polychromatic ee curve depends on weight scale: "
            f"ee1={ee1!r}, ee2={ee2!r}, ee3={ee3!r}.")
        assert np.allclose(r1, r2) and np.allclose(r1, r3), (
            "ee_polychromatic radii axis depends on weight scale.")

    def test_c1_mismatched_wavelengths_weights_raises(self):
        """``wavelengths`` and ``weights`` must have the same length."""
        rx = _make_singlet_for_ee()
        radii = [5e-6, 10e-6]
        N, dx = 32, 5e-6
        with pytest.raises(ValueError, match='same length'):
            ee_polychromatic(
                rx, [1.30e-6, 1.55e-6, 1.65e-6], [1.0, 1.0], radii,
                output_grid=N, output_dx=dx)

    def test_c1_radii_non_increasing_raises(self):
        """``radii`` must be strictly increasing.  Decreasing or
        equal-adjacent values raise ``ValueError``."""
        rx = _make_singlet_for_ee()
        N, dx = 32, 5e-6
        with pytest.raises(ValueError, match='strictly increasing'):
            ee_polychromatic(
                rx, [1.30e-6, 1.55e-6], [1.0, 1.0],
                [10e-6, 5e-6, 20e-6],
                output_grid=N, output_dx=dx)
        # Equal-adjacent also fails (strictly increasing means strict).
        with pytest.raises(ValueError, match='strictly increasing'):
            ee_polychromatic(
                rx, [1.30e-6, 1.55e-6], [1.0, 1.0],
                [5e-6, 5e-6, 10e-6],
                output_grid=N, output_dx=dx)

    def test_c1_weights_sum_zero_raises(self):
        """If ``sum(weights) <= 0`` the wrapper raises before doing any
        propagation work."""
        rx = _make_singlet_for_ee()
        N, dx = 32, 5e-6
        with pytest.raises(ValueError, match='strictly positive'):
            ee_polychromatic(
                rx, [1.30e-6, 1.55e-6], [0.0, 0.0],
                [5e-6, 10e-6, 20e-6],
                output_grid=N, output_dx=dx)
        with pytest.raises(ValueError, match='strictly positive'):
            ee_polychromatic(
                rx, [1.30e-6, 1.55e-6], [-1.0, 1.0],
                [5e-6, 10e-6, 20e-6],
                output_grid=N, output_dx=dx)

    def test_c1_radii_non_positive_raises(self):
        """``radii`` must be strictly positive."""
        rx = _make_singlet_for_ee()
        N, dx = 32, 5e-6
        with pytest.raises(ValueError, match='strictly positive'):
            ee_polychromatic(
                rx, [1.30e-6, 1.55e-6], [1.0, 1.0],
                [0.0, 5e-6, 10e-6],
                output_grid=N, output_dx=dx)
        with pytest.raises(ValueError, match='strictly positive'):
            ee_polychromatic(
                rx, [1.30e-6, 1.55e-6], [1.0, 1.0],
                [-1e-6, 5e-6, 10e-6],
                output_grid=N, output_dx=dx)


# ============================================================================
# C.2 -- strehl_vector + coupling_efficiency_vector
# ============================================================================

class TestStrehlVector:
    """Vector (polarisation-aware) Strehl ratio.

    NOTE (v4.15.1): the v4.15.0 default-reference branch (uniform
    plane wave of matching total power) was removed because it
    produced Strehl > 1 on any focused PSF.  Tests below now require
    explicit `reference=(Ex_ref, Ey_ref[, Ez_ref])`.  The original
    `plane_wave_vs_default_reference_is_unity` pin was tautological
    (uniform field vs uniform reference of equal power is
    identically 1.0 by construction) and has been replaced with an
    analytical-value pin on a diffraction-limited reference plus an
    explicit `ValueError` pin for the removed default branch.
    """

    def test_c2_strehl_vector_unaberrated_against_diffraction_ref_is_unity(
            self):
        """A diffraction-limited focused field compared against itself
        (same field as reference) must yield Strehl = 1.0 to roundoff.
        Analytical pin (audit V4.15.0 meta-finding): replaces the
        prior tautological uniform-vs-uniform pin with a meaningful
        diffraction-limited-reference assertion."""
        N_pupil = 32
        dx = 50e-6
        D = 1e-3
        x = (np.arange(N_pupil) - N_pupil / 2) * dx
        X, Y = np.meshgrid(x, x)
        aperture = ((X ** 2 + Y ** 2) <= (D / 2) ** 2).astype(np.complex128)
        psf, _ = la.compute_psf(aperture, 600e-9, 50e-3, dx, oversample=2)
        Ex_ref = np.sqrt(psf).astype(np.complex128)
        Ey_ref = np.zeros_like(Ex_ref)
        s = strehl_vector(
            Ex_ref.copy(), Ey_ref.copy(),
            reference=(Ex_ref, Ey_ref))
        assert abs(s - 1.0) < 1e-12, (
            f"strehl_vector(field, reference=field) -> {s!r}; "
            f"expected 1.0.")

    def test_c2_strehl_vector_default_reference_path_now_raises(self):
        """v4.15.1 breaking change: omitting ``reference`` raises
        ValueError (previously returned a tautological 1.0 for uniform
        fields and produced Strehl > 1 for focused PSFs)."""
        N = 16
        Ex = np.ones((N, N), dtype=np.complex128)
        Ey = np.zeros_like(Ex)
        with pytest.raises((ValueError, TypeError),
                            match="reference"):
            strehl_vector(Ex, Ey)

    def test_c2_strehl_vector_monotone_decreasing_under_defocus(self):
        """Adding more defocus to the same field should *not* increase
        the Strehl ratio.  Synthesize a pupil-plane field, propagate to
        focus, and compare Strehl at successively larger defocus
        magnitudes against the unaberrated reference."""
        # Pupil-plane aperture
        N = 64
        dx = 25e-6
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        r2 = X ** 2 + Y ** 2
        D = 1e-3
        aperture = (r2 <= (D / 2) ** 2).astype(np.complex128)
        wavelength = 633e-9
        f = 30e-3
        # Reference PSF (no aberration)
        psf_ref, _ = la.compute_psf(
            aperture, wavelength, f, dx, oversample=2)
        Ex_ref = np.sqrt(psf_ref).astype(np.complex128)
        Ey_ref = np.zeros_like(Ex_ref)

        # Series of defocus amplitudes in waves.
        strehls = []
        for n_waves in [0.0, 0.05, 0.10, 0.20, 0.40]:
            # Defocus phase: alpha * (2 r^2 / R^2 - 1) for Zernike
            # defocus.  We use a simple radial parabola in units of
            # waves of WFE for monotonicity testing.
            r_norm2 = r2 / ((D / 2) ** 2)
            phase = n_waves * 2 * np.pi * (2.0 * r_norm2 - 1.0)
            pupil_aber = aperture * np.exp(1j * phase)
            psf_aber, _ = la.compute_psf(
                pupil_aber, wavelength, f, dx, oversample=2)
            Ex_aber = np.sqrt(psf_aber).astype(np.complex128)
            Ey_aber = np.zeros_like(Ex_aber)
            s = strehl_vector(
                Ex_aber, Ey_aber,
                reference=(Ex_ref, Ey_ref))
            strehls.append(s)
        # Monotone non-increasing (no defocus -> highest; more defocus
        # -> equal or lower).
        diffs = np.diff(strehls)
        assert np.all(diffs <= 1e-9), (
            f"strehl_vector is not monotone non-increasing under "
            f"defocus: strehls = {strehls!r}, diffs = {diffs!r}.")
        assert abs(strehls[0] - 1.0) < 1e-9, (
            f"unaberrated strehl_vector deviates from 1.0: "
            f"{strehls[0]!r}.")

    def test_c2_ez_none_equals_ez_zeros(self):
        """Omitting ``Ez`` must be numerically equivalent to passing a
        zero-array Ez of matching shape."""
        rng = np.random.default_rng(2026)
        Ex = (rng.standard_normal((8, 8)) +
              1j * rng.standard_normal((8, 8)))
        Ey = (rng.standard_normal((8, 8)) +
              1j * rng.standard_normal((8, 8)))
        # Use a fixed reference so both branches see the same denom.
        Ex_ref = np.ones_like(Ex)
        Ey_ref = np.zeros_like(Ex)
        s_no_ez = strehl_vector(
            Ex, Ey, None, reference=(Ex_ref, Ey_ref))
        s_zero_ez = strehl_vector(
            Ex, Ey, np.zeros_like(Ex),
            reference=(Ex_ref, Ey_ref, np.zeros_like(Ex_ref)))
        assert abs(s_no_ez - s_zero_ez) < 1e-12, (
            f"strehl_vector(Ez=None) and strehl_vector(Ez=zeros) "
            f"disagree: {s_no_ez!r} vs {s_zero_ez!r}.")

    def test_c2_strehl_vector_1d_field_rejected(self):
        """1-D inputs must raise ``ValueError`` -- we explicitly do
        not accept ravelled vector fields.  v4.15.1: reference is
        also required, so we pass a 2-D reference and let the 1-D
        input trigger the shape-validation error."""
        Ex_ref = np.ones((16, 16), dtype=complex)
        Ey_ref = np.zeros((16, 16), dtype=complex)
        with pytest.raises(ValueError, match='must be 2-D'):
            strehl_vector(np.ones(16, dtype=complex),
                          np.zeros(16, dtype=complex),
                          reference=(Ex_ref, Ey_ref))


class TestCouplingEfficiencyVector:
    """Vector mode-overlap coupling."""

    def test_c2_coupling_vector_mode_equals_field_is_unity(self):
        """When the field equals the target mode, coupling = 1.0."""
        N, dx = 16, 1e-6
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        w0 = 4e-6
        gauss = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
        # Linear-x polarisation field.
        Ex = gauss
        Ey = np.zeros_like(gauss)
        eta = coupling_efficiency_vector(
            Ex, Ey,
            mode_Ex=gauss.copy(), mode_Ey=np.zeros_like(gauss),
            dx=dx,
        )
        assert abs(eta - 1.0) < 1e-12, (
            f"coupling_efficiency_vector(mode=field) -> {eta!r}; "
            f"expected 1.0.")

    def test_c2_coupling_vector_orthogonal_mode_is_zero(self):
        """Orthogonal modes give coupling = 0.  Use a polarisation-
        orthogonality argument: a pure linear-x field has zero overlap
        with a pure linear-y mode (regardless of spatial shape), so
        the cross-polarisation overlap saturates the orthogonality
        condition without depending on grid-symmetry quirks."""
        N, dx = 16, 1e-6
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        w0 = 4e-6
        gauss = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
        # Field: x-polarised Gaussian.
        Ex_field = gauss
        Ey_field = np.zeros_like(gauss)
        # Mode: y-polarised Gaussian (orthogonal polarisation).  The
        # spatial profile is identical, but the cross-polarisation
        # overlap is identically zero.
        mode_Ex = np.zeros_like(gauss)
        mode_Ey = gauss
        eta = coupling_efficiency_vector(
            Ex_field, Ey_field,
            mode_Ex=mode_Ex, mode_Ey=mode_Ey,
            dx=dx,
        )
        assert eta < 1e-12, (
            f"coupling_efficiency_vector(orthogonal polarisation) "
            f"-> {eta!r}; expected ~ 0.")

    def test_c2_coupling_vector_with_ez_normalisation(self):
        """Coupling with non-trivial Ez (high-NA case) should match the
        analytic formula and saturate at 1.0 when field == mode."""
        N, dx = 8, 1e-6
        rng = np.random.default_rng(0)
        Ex = (rng.standard_normal((N, N)) +
              1j * rng.standard_normal((N, N)))
        Ey = (rng.standard_normal((N, N)) +
              1j * rng.standard_normal((N, N)))
        Ez = (rng.standard_normal((N, N)) +
              1j * rng.standard_normal((N, N)))
        eta = coupling_efficiency_vector(
            Ex, Ey, Ez,
            mode_Ex=Ex.copy(), mode_Ey=Ey.copy(), mode_Ez=Ez.copy(),
            dx=dx,
        )
        assert abs(eta - 1.0) < 1e-12, (
            f"coupling_efficiency_vector(self, with Ez) -> {eta!r}; "
            f"expected 1.0.")

    def test_c2_coupling_vector_1d_field_rejected(self):
        """1-D inputs must raise ``ValueError``."""
        with pytest.raises(ValueError, match='must be 2-D'):
            coupling_efficiency_vector(
                np.ones(16, dtype=complex),
                np.zeros(16, dtype=complex),
                mode_Ex=np.ones(16, dtype=complex),
                mode_Ey=np.zeros(16, dtype=complex),
                dx=1e-6,
            )

    def test_c2_coupling_vector_shape_mismatch_rejected(self):
        """Mode shape must match field shape."""
        with pytest.raises(ValueError, match='shape'):
            coupling_efficiency_vector(
                np.ones((8, 8), dtype=complex),
                np.zeros((8, 8), dtype=complex),
                mode_Ex=np.ones((16, 16), dtype=complex),
                mode_Ey=np.zeros((16, 16), dtype=complex),
                dx=1e-6,
            )


# ============================================================================
# C.3 -- Optical resolution metrics (Rayleigh / Sparrow / FWHM)
# ============================================================================

class TestResolutionMetrics:
    """Two-point-separability resolution criteria from a 2-D PSF."""

    def test_c3_rayleigh_on_airy_within_5_percent(self):
        """Rayleigh resolution of a clean Airy PSF must match
        ``1.22 lambda f/#`` to within 5%."""
        wavelength = 600e-9
        psf, dx_psf, f_number = _make_airy_psf(wavelength=wavelength)
        d_rayleigh = rayleigh_resolution(
            psf, dx_psf, wavelength, axis='radial')
        d_expected = 1.22 * wavelength * f_number
        rel = abs(d_rayleigh - d_expected) / d_expected
        assert rel < 0.05, (
            f"rayleigh_resolution on Airy: got {d_rayleigh*1e6:.4f} um, "
            f"expected {d_expected*1e6:.4f} um "
            f"(rel error {rel*100:.2f}% > 5%).")

    def test_c3_sparrow_resolution_on_airy_analytical_value(self):
        """Sparrow resolution on a clean Airy PSF must match the
        analytical ``0.947 * lambda * f/#`` to within 5%.

        v4.15.1 (audit V4.15.0 meta-finding): replaces the prior
        property-only ``sparrow < rayleigh`` pin with an analytical-
        value pin against the canonical Sparrow constant
        (the dip-vanishing separation for a circular-aperture Airy
        pattern).  The old property-only pin masked the 15%
        algorithmic error that the audit caught."""
        wavelength = 600e-9
        psf, dx_psf, f_number = _make_airy_psf(wavelength=wavelength)
        d_sparrow = sparrow_resolution(psf, dx_psf, axis='radial')
        d_expected = 0.947 * wavelength * f_number
        rel = abs(d_sparrow - d_expected) / d_expected
        assert rel < 0.05, (
            f"sparrow_resolution on Airy: got {d_sparrow*1e6:.4f} um, "
            f"expected {d_expected*1e6:.4f} um (0.947 * lam * f/#); "
            f"rel error {rel*100:.2f}% > 5%.")
        # Property pin kept as a sanity check on top of the
        # analytical pin: Sparrow < Rayleigh on a clean Airy.
        d_rayleigh = rayleigh_resolution(
            psf, dx_psf, wavelength, axis='radial')
        assert d_sparrow < d_rayleigh, (
            f"Sparrow {d_sparrow*1e6:.4f} um should be smaller than "
            f"Rayleigh {d_rayleigh*1e6:.4f} um for an Airy.")

    def test_c3_fwhm_resolution_on_airy_analytical_value(self):
        """FWHM diameter on a clean Airy PSF must match the
        analytical ``1.029 * lambda * f/#`` to within 5%.

        v4.15.1 (audit V4.15.0 meta-finding): replaces the prior
        property-only ``fwhm < rayleigh`` pin with an analytical-
        value pin against the canonical Airy FWHM constant."""
        wavelength = 600e-9
        psf, dx_psf, f_number = _make_airy_psf(wavelength=wavelength)
        d_fwhm = fwhm_resolution(psf, dx_psf, axis='radial')
        d_expected = 1.029 * wavelength * f_number
        rel = abs(d_fwhm - d_expected) / d_expected
        assert rel < 0.05, (
            f"fwhm_resolution on Airy: got {d_fwhm*1e6:.4f} um, "
            f"expected {d_expected*1e6:.4f} um (1.029 * lam * f/#); "
            f"rel error {rel*100:.2f}% > 5%.")
        # Property pin kept as a sanity check: FWHM < Rayleigh on a
        # clean Airy.
        d_rayleigh = rayleigh_resolution(
            psf, dx_psf, wavelength, axis='radial')
        assert d_fwhm < d_rayleigh, (
            f"FWHM {d_fwhm*1e6:.4f} um should be less than Rayleigh "
            f"{d_rayleigh*1e6:.4f} um for an Airy.")

    def test_c3_axis_x_vs_radial_both_finite(self):
        """Both ``axis='x'`` and ``axis='radial'`` produce finite,
        positive metrics for a symmetric Airy PSF; they need not agree
        bit-for-bit but should be in the same ballpark."""
        wavelength = 600e-9
        psf, dx_psf, _ = _make_airy_psf(wavelength=wavelength)
        for fn_label, fn in [
                ('rayleigh', lambda axis: rayleigh_resolution(
                    psf, dx_psf, wavelength, axis=axis)),
                ('sparrow', lambda axis: sparrow_resolution(
                    psf, dx_psf, axis=axis)),
                ('fwhm', lambda axis: fwhm_resolution(
                    psf, dx_psf, axis=axis))]:
            d_x = fn('x')
            d_rad = fn('radial')
            assert np.isfinite(d_x) and d_x > 0, (
                f"{fn_label}_resolution(axis='x') -> {d_x!r}; "
                f"expected positive finite.")
            assert np.isfinite(d_rad) and d_rad > 0, (
                f"{fn_label}_resolution(axis='radial') -> {d_rad!r}; "
                f"expected positive finite.")
            # Order of magnitude agreement; the symmetry of the PSF
            # keeps these within a factor of 2.
            assert 0.3 < d_x / d_rad < 3.0, (
                f"{fn_label}_resolution axis='x' vs 'radial' disagree "
                f"by more than a factor of 3: x={d_x!r}, "
                f"radial={d_rad!r}.")

    def test_c3_non_2d_rejected(self):
        """Non-2-D inputs raise ``ValueError`` on all three resolution
        functions."""
        psf_1d = np.ones(64)
        psf_3d = np.ones((8, 8, 8))
        for fn_label, fn in [
                ('rayleigh', lambda p: rayleigh_resolution(p, 1e-6, 600e-9)),
                ('sparrow', lambda p: sparrow_resolution(p, 1e-6)),
                ('fwhm', lambda p: fwhm_resolution(p, 1e-6))]:
            with pytest.raises(ValueError, match='2-D'):
                fn(psf_1d)
            with pytest.raises(ValueError, match='2-D'):
                fn(psf_3d)

    def test_c3_flat_field_returns_nan_sentinel(self):
        """A flat / zero-intensity PSF yields NaN sentinels rather than
        crashing or returning misleading positives."""
        flat = np.zeros((16, 16))
        assert np.isnan(rayleigh_resolution(flat, 1e-6, 600e-9)), (
            "rayleigh_resolution on zero PSF should return NaN.")
        assert np.isnan(sparrow_resolution(flat, 1e-6)), (
            "sparrow_resolution on zero PSF should return NaN.")
        assert np.isnan(fwhm_resolution(flat, 1e-6)), (
            "fwhm_resolution on zero PSF should return NaN.")


# ============================================================================
# C.4 -- astigmatism_mag_angle
# ============================================================================

class TestAstigmatismMagAngle:
    """Mahajan-convention magnitude+angle of primary astigmatism from
    an OSA-indexed Zernike coefficient array."""

    def test_c4_zero_coeffs_zero_magnitude(self):
        """All-zero coefficients -> magnitude 0, angle 0."""
        c = np.zeros(6, dtype=np.float64)
        mag, theta = astigmatism_mag_angle(c)
        assert mag == 0.0, f"zero coeffs -> magnitude {mag!r}; expected 0."
        assert theta == 0.0, f"zero coeffs -> theta {theta!r}; expected 0."

    def test_c4_pure_c5_gives_theta_zero(self):
        """Pure vertical astigmatism (c5 = +1, c3 = 0) -> theta = 0."""
        c = np.zeros(6, dtype=np.float64)
        c[5] = 1.0
        mag, theta = astigmatism_mag_angle(c)
        assert abs(mag - 1.0) < 1e-15, (
            f"pure c5=1 -> magnitude {mag!r}; expected 1.0.")
        assert abs(theta) < 1e-15, (
            f"pure c5=1 -> theta {theta!r}; expected 0.")

    def test_c4_pure_c3_gives_theta_pi_over_4(self):
        """Pure oblique astigmatism (c3 = +1, c5 = 0) -> theta = pi/4."""
        c = np.zeros(6, dtype=np.float64)
        c[3] = 1.0
        mag, theta = astigmatism_mag_angle(c)
        assert abs(mag - 1.0) < 1e-15, (
            f"pure c3=1 -> magnitude {mag!r}; expected 1.0.")
        assert abs(theta - np.pi / 4) < 1e-15, (
            f"pure c3=1 -> theta {theta!r}; expected pi/4.")

    def test_c4_short_coeffs_raises(self):
        """Need at least 6 coefficients to index c5."""
        with pytest.raises(ValueError, match='at least 6'):
            astigmatism_mag_angle(np.zeros(5))
        with pytest.raises(ValueError, match='at least 6'):
            astigmatism_mag_angle([1.0, 2.0, 3.0])

    def test_c4_recovers_zernike_decompose_round_trip(self):
        """Synthesize an OPD map containing only c3 and c5
        astigmatism, decompose, and verify the recovered magnitude
        and angle match Mahajan's formula on the input coefficients
        directly (within fit roundoff)."""
        N, dx = 64, 1e-4
        aperture = 5e-3
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        r_pupil = 0.5 * aperture
        pupil_mask = (X ** 2 + Y ** 2) <= r_pupil ** 2
        c3_true = 5e-7
        c5_true = -3e-7
        # Synthesize the OPD via zernike_reconstruct so it lives on the
        # same OSA basis as the decomposition.
        coeffs_in = np.zeros(6)
        coeffs_in[3] = c3_true
        coeffs_in[5] = c5_true
        opd_map = la.zernike_reconstruct(
            coeffs_in, dx, (N, N), aperture)
        coeffs, _names = la.zernike_decompose(
            opd_map, dx, aperture, n_modes=15)
        mag_recovered, theta_recovered = astigmatism_mag_angle(coeffs)
        mag_expected = np.sqrt(c3_true ** 2 + c5_true ** 2)
        theta_expected = 0.5 * np.arctan2(c3_true, c5_true)
        assert abs(mag_recovered - mag_expected) / mag_expected < 0.05, (
            f"recovered |astig| {mag_recovered!r} != input "
            f"|astig| {mag_expected!r} within 5%.")
        assert abs(theta_recovered - theta_expected) < 1e-3, (
            f"recovered theta {theta_recovered!r} != "
            f"input theta {theta_expected!r}.")

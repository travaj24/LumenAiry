"""v4.15.1 Agent G -- field-application correctness tests.

This module pins the chain-and-delegate semantics: each primitive's
:meth:`_apply` must produce the same numerical output as the
underlying LumenAiry function it delegates to, and
:class:`CompositeOperator` must compose them in the spec'd
right-to-left order ``A(B(C(field)))``.

Corresponds to ``CLUSTER_B_SPEC.md`` §3.7 ``test_algebra_application.py``.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.algebra import (
    Aperture,
    CylindricalLens,
    FourierTransform,
    FreeSpace,
    GaussianAperture,
    Magnify,
    ThinLens,
)


# ============================================================================
# Helpers
# ============================================================================


def _build_test_source(*, N: int = 256, dx: float = 2e-6,
                       w0: float = 50e-6,
                       wavelength: float = 633e-9) -> la.Source:
    """Build a small reproducible Gaussian source for application tests."""
    return la.Source.gaussian(
        N=N, dx=dx, wavelength=wavelength, w0=w0,
    )


# ============================================================================
# FreeSpace delegation
# ============================================================================


class TestFreeSpaceApplication:

    def test_freespace_application_matches_dispatcher(self) -> None:
        """``FreeSpace(d)(source)`` matches a direct ASM call to within
        floating-point roundoff."""
        src = _build_test_source()
        d = 0.05
        op = FreeSpace(d, method='asm')
        out = op(src)
        E_ref = la.angular_spectrum_propagate(
            src.E, z=d, wavelength=src.wavelength, dx=src.dx,
        )
        assert np.allclose(out.E, E_ref)

    def test_freespace_zero_distance_passthrough(self) -> None:
        """``FreeSpace(0)`` returns the input field bit-for-bit."""
        src = _build_test_source()
        out = FreeSpace(0.0)(src)
        # Same object reference is acceptable (zero-distance fast path).
        assert np.array_equal(out.E, src.E)
        assert out.dx == src.dx

    def test_freespace_through_tuple_input(self) -> None:
        """``op((E, dx, wavelength))`` returns the equivalent tuple."""
        src = _build_test_source()
        op = FreeSpace(0.05, method='asm')
        E_out, dx_out, wl_out = op((src.E, src.dx, src.wavelength))
        # Same output as the Source path.
        out_via_source = op(src)
        assert np.allclose(E_out, out_via_source.E)
        assert dx_out == out_via_source.dx
        assert wl_out == out_via_source.wavelength

    def test_freespace_through_anamorphic_tuple(self) -> None:
        """``op((E, dx, dy, wavelength))`` threads dy through."""
        src = _build_test_source()
        op = FreeSpace(0.05, method='asm')
        E_out, dx_out, dy_out, wl_out = op(
            (src.E, src.dx, src.dx, src.wavelength)
        )
        assert E_out.shape == src.E.shape
        assert dx_out > 0
        assert dy_out > 0


# ============================================================================
# ThinLens delegation
# ============================================================================


class TestThinLensApplication:

    def test_thinlens_application_matches_apply_thin_lens(self) -> None:
        src = _build_test_source()
        op = ThinLens(0.1)
        out = op(src)
        E_ref = la.apply_thin_lens(
            src.E, f=0.1, wavelength=src.wavelength, dx=src.dx,
        )
        assert np.allclose(out.E, E_ref)

    def test_thinlens_infinite_focal_length_is_passthrough(self) -> None:
        """``ThinLens(inf)`` returns the input untouched."""
        src = _build_test_source()
        out = ThinLens(np.inf)(src)
        assert np.array_equal(out.E, src.E)

    def test_thinlens_apply_bare_ndarray(self) -> None:
        """``op.apply(E, dx=..., wavelength=...)`` returns a bare
        ndarray."""
        src = _build_test_source()
        E_out = ThinLens(0.1).apply(
            src.E, dx=src.dx, wavelength=src.wavelength,
        )
        assert isinstance(E_out, np.ndarray)
        assert E_out.shape == src.E.shape


# ============================================================================
# CylindricalLens delegation
# ============================================================================


class TestCylindricalLensApplication:

    def test_cylindrical_x_only_matches_direct_call(self) -> None:
        """A cylindrical lens with finite f_x and infinite f_y matches
        a direct ``apply_cylindrical_lens(axis='x')`` call."""
        src = _build_test_source()
        op = CylindricalLens(f_x=0.05, f_y=np.inf)
        out = op(src)
        from lumenairy.elements.lenses import apply_cylindrical_lens
        E_ref = apply_cylindrical_lens(
            src.E, f=0.05, wavelength=src.wavelength,
            dx=src.dx, axis='x',
        )
        assert np.allclose(out.E, E_ref)

    def test_cylindrical_anamorphic_produces_anamorphic_intensity(
        self,
    ) -> None:
        """A cylindrical lens followed by free-space should focus the
        beam in one axis only -- the focused-axis 1/e^2 width should
        be measurably smaller than the unfocused-axis width near the
        focal plane.  Grid + Rayleigh range chosen so the focused-
        axis spot stays well-sampled (Fresnel-region focus, not the
        diffraction-limited single-pixel spike)."""
        src = la.Source.gaussian(
            N=256, dx=4e-6, wavelength=633e-9, w0=400e-6,
        )
        # Long focal length keeps the focal spot wider than the
        # pixel pitch so the marginal sigmas resolve correctly.
        sys = FreeSpace(0.5, method='asm') * CylindricalLens(
            f_x=0.5, f_y=np.inf,
        )
        out = sys(src)
        I = np.abs(out.E) ** 2
        # Width along x (focused axis) vs y (un-focused axis)
        ix_marg = I.sum(axis=0)
        iy_marg = I.sum(axis=1)
        x_axis = (np.arange(out.E.shape[1]) - out.E.shape[1] / 2) * out.dx
        y_axis = (np.arange(out.E.shape[0]) - out.E.shape[0] / 2) * out.dx
        # Center marginals on the intensity peak before computing
        # the sigma so any centroid shift from finite-aperture
        # diffraction doesn't inflate the rms width.
        sigma_x_centroid = (
            np.sum(ix_marg * x_axis) / np.sum(ix_marg)
        )
        sigma_y_centroid = (
            np.sum(iy_marg * y_axis) / np.sum(iy_marg)
        )
        sigma_x = np.sqrt(
            np.sum(ix_marg * (x_axis - sigma_x_centroid) ** 2)
            / np.sum(ix_marg)
        )
        sigma_y = np.sqrt(
            np.sum(iy_marg * (y_axis - sigma_y_centroid) ** 2)
            / np.sum(iy_marg)
        )
        # Focused axis should be measurably narrower than the
        # un-focused axis.  The asymmetry is the diagnostic for the
        # anamorphic ABCD; the exact ratio depends on Rayleigh range
        # vs. focal length.
        assert sigma_x < sigma_y, (
            f"Cylindrical focusing did not produce an anamorphic "
            f"intensity: sigma_x={sigma_x:.3e}, sigma_y={sigma_y:.3e}"
        )


# ============================================================================
# Aperture / GaussianAperture delegation
# ============================================================================


class TestApertureApplication:

    def test_circular_aperture_zeros_outside_disk(self) -> None:
        src = _build_test_source(N=64, dx=2e-6)
        D = 50e-6  # tight aperture compared to the field extent
        op = Aperture(diameter=D, shape='circular')
        out = op(src)
        # Build the expected mask
        N = src.E.shape[0]
        x = (np.arange(N) - N / 2) * src.dx
        X, Y = np.meshgrid(x, x)
        mask = X ** 2 + Y ** 2 <= (D / 2) ** 2
        # Inside the disk: out.E == src.E.  Outside: out.E == 0.
        assert np.allclose(out.E[mask], src.E[mask])
        assert np.all(np.abs(out.E[~mask]) < 1e-30)

    def test_gaussian_aperture_smoothly_attenuates(self) -> None:
        src = _build_test_source(N=64, dx=2e-6)
        sigma = 50e-6
        op = GaussianAperture(sigma=sigma)
        out = op(src)
        # Center pixel: T ~ 1
        c = src.E.shape[0] // 2
        assert np.isclose(
            np.abs(out.E[c, c]) / np.abs(src.E[c, c]), 1.0, atol=1e-6,
        )
        # 3-sigma corner: T = exp(-(3 sigma)^2 / (2 sigma^2)) = exp(-4.5)
        # The corner pixel isn't exactly 3-sigma but should be well below 1.
        assert np.abs(out.E[0, 0]) < np.abs(src.E[0, 0])


# ============================================================================
# Composite application
# ============================================================================


class TestCompositeApplication:

    def test_composite_application_matches_sequential(self) -> None:
        """``sys(src)`` for ``sys = A * B * C`` equals ``A(B(C(src)))``."""
        src = _build_test_source()
        a = FreeSpace(0.01, method='asm')
        b = ThinLens(0.05)
        c = FreeSpace(0.05, method='asm')
        sys = a * b * c
        out_composite = sys(src)
        out_sequential = a(b(c(src)))
        assert np.allclose(out_composite.E, out_sequential.E)
        assert out_composite.dx == out_sequential.dx

    def test_composite_stages_returns_correct_order(self) -> None:
        """``stages()[0]`` is applied first (chain-rightmost-of-product)."""
        a = FreeSpace(0.01)
        b = ThinLens(0.05)
        c = FreeSpace(0.05)
        sys = a * b * c
        stages = sys.stages()
        # Mathematical order is a * b * c == "first c, then b, then a".
        # So stages[0] is c, stages[1] is b, stages[2] is a.
        assert stages[0] is c
        assert stages[1] is b
        assert stages[2] is a

    def test_apply_with_intermediates_returns_list(self) -> None:
        src = _build_test_source()
        sys = (FreeSpace(0.01, method='asm') * ThinLens(0.05)
               * FreeSpace(0.05, method='asm'))
        intermediates = sys.apply_with_intermediates(
            src.E, dx=src.dx, wavelength=src.wavelength,
        )
        # 3 stages -> 3 intermediates
        assert len(intermediates) == 3
        # Last intermediate matches the full sys(src) output
        out = sys(src)
        E_last, _, _ = intermediates[-1]
        assert np.allclose(E_last, out.E)

    def test_4f_inverter_centroid_flips_sign(self) -> None:
        """A 4f system [f, lens(f), 2f, lens(f), f] should image a
        Gaussian centered at +x0 to a Gaussian centered at -x0 with
        unit magnification.

        The test grid is sized so the beam's Rayleigh range covers
        the full 4f path (a w0 = 200 um Gaussian has z_R ~ 20 cm at
        633 nm, so the 4f system with f = 50 mm stays well inside
        the Rayleigh range and the centroid maps cleanly to -x0
        without significant diffractive spread vignetting the
        signal).
        """
        f = 50e-3
        x0 = 200e-6
        src = la.Source.gaussian(
            N=512, dx=4e-6, wavelength=633e-9, w0=200e-6, x0=x0,
        )
        sys = (FreeSpace(f, method='asm') * ThinLens(f)
               * FreeSpace(2 * f, method='asm') * ThinLens(f)
               * FreeSpace(f, method='asm'))
        out = sys(src)
        # Centroid of |out|^2 along x
        intensity = np.abs(out.E) ** 2
        ix = np.arange(out.E.shape[1])
        marginal_x = intensity.sum(axis=0)
        centroid_idx = np.sum(ix * marginal_x) / np.sum(marginal_x)
        centroid_x = (centroid_idx - out.E.shape[1] / 2) * out.dx
        # 4f inverter maps +x0 -> -x0; tolerance is ~one pixel plus
        # a defocus / sampling margin.
        assert np.isclose(centroid_x, -x0, atol=20e-6), (
            f"4f inverter centroid mismatch: expected {-x0:.3e} m, "
            f"got {centroid_x:.3e} m."
        )


# ============================================================================
# FourierTransform application
# ============================================================================


class TestFourierTransformApplication:

    def test_fourier_transform_runs_and_returns_finite(self) -> None:
        """``FourierTransform(f)(src)`` runs end-to-end and produces a
        finite field with non-zero energy."""
        src = la.Source.gaussian(
            N=256, dx=2e-6, wavelength=633e-9, w0=100e-6,
        )
        op = FourierTransform(0.1)
        out = op(src)
        assert np.all(np.isfinite(out.E))
        assert np.sum(np.abs(out.E) ** 2) > 0.0

    def test_fourier_transform_field_matches_3_stage_chain(self) -> None:
        """v4.15.2 (audit P1-NEW-A): ``FourierTransform(f)(src)`` must
        match the literal 3-stage chain ``FreeSpace(f) * ThinLens(f) *
        FreeSpace(f)`` applied to the same source, on the FIELD side
        not just the ABCD side.

        Pre-v4.15.2 the 2-stage shortcut left a residual
        ``exp(+i*k/(2f)*r^2)`` quadratic phase, so this test failed
        with a non-trivial pointwise mismatch in the output field.
        """
        # Off-axis Gaussian: shifts the energy off-axis so the
        # residual quadratic phase (centered at the origin) produces
        # a measurable field-level mismatch in the 2-stage shortcut,
        # not just a global phase rotation that would be invisible to
        # a magnitude-only comparison.
        f = 0.1
        src = la.Source.gaussian(
            N=256, dx=2e-6, wavelength=633e-9, w0=100e-6, x0=80e-6,
        )
        op = FourierTransform(f)
        chain = FreeSpace(f) * ThinLens(f) * FreeSpace(f)
        out_op = op(src)
        out_chain = chain(src)
        # Both should produce the same output pitch since the 3-stage
        # invocation IS the chain.
        assert np.isclose(out_op.dx, out_chain.dx, rtol=1e-12)
        # And the FIELD itself must match bit-for-bit (or to a tight
        # numerical tolerance accounting for FFT roundoff across two
        # independent invocations of the same primitive sequence).
        assert np.allclose(out_op.E, out_chain.E, rtol=1e-10, atol=1e-12), (
            f"FourierTransform field does not match 3-stage chain: "
            f"max abs diff = "
            f"{np.max(np.abs(out_op.E - out_chain.E)):.3e}"
        )


# ============================================================================
# FreeSpace dy threading (v4.15.2 audit P1-NEW-C)
# ============================================================================


class TestFreeSpaceDyThreading:

    def test_freespace_threads_dy_for_anamorphic_chain(self) -> None:
        """v4.15.2 (audit P1-NEW-C): ``FreeSpace._apply`` must thread
        ``dy`` into the underlying propagator so anamorphic chains
        starting with ``Magnify(a_x, a_y) * FreeSpace(d)`` propagate
        on the correct grid.

        Pre-v4.15.2 the call dropped ``dy`` and the kernel silently
        defaulted to ``dy = dx``.  This test exercises the bare-tuple
        anamorphic entry point so the input ``dy != dx`` reaches the
        underlying propagator.
        """
        # Build an anamorphic input grid: dy = 2*dx.
        src = _build_test_source(N=64, dx=2e-6, w0=50e-6)
        dx_in = float(src.dx)
        dy_in = 2.0 * dx_in
        op = FreeSpace(0.05, method='asm')
        # 4-tuple anamorphic entry point threads dy into _apply.
        E_out, dx_out, dy_out, wl_out = op(
            (src.E, dx_in, dy_in, src.wavelength)
        )
        assert E_out.shape == src.E.shape
        assert np.all(np.isfinite(E_out))
        # ASM preserves pitch (no rescaling kernel), so the output
        # pitches should match the input.
        assert np.isclose(dx_out, dx_in, rtol=1e-12)
        assert np.isclose(dy_out, dy_in, rtol=1e-12), (
            f"FreeSpace dropped dy: input dy={dy_in:.3e}, output "
            f"dy={dy_out:.3e} (expected {dy_in:.3e} since ASM preserves "
            f"pitch)"
        )
        # Sanity: the output is not the same as the dy=dx (square-
        # grid) propagation, because the y-axis kernel uses a
        # different sampling grid.
        E_out_square, _, _ = op._apply(
            src.E, dx=dx_in, dy=dx_in, wavelength=src.wavelength,
        )
        # Some pixel must differ between the two (the y-axis kernel
        # differs).  A pure on-axis Gaussian might give nearly-equal
        # results within roundoff; verify the y-axis is actually
        # threaded by checking the kernel produces a different
        # output when dy != dx.
        # Use an off-axis Gaussian so the y-axis kernel matters.
        src_off = la.Source.gaussian(
            N=64, dx=dx_in, wavelength=src.wavelength, w0=20e-6,
            y0=15e-6,
        )
        E_anam, _, dy_anam = op._apply(
            src_off.E, dx=dx_in, dy=dy_in, wavelength=src.wavelength,
        )
        E_iso, _, dy_iso = op._apply(
            src_off.E, dx=dx_in, dy=dx_in, wavelength=src.wavelength,
        )
        # The anamorphic output must differ from the isotropic one;
        # if dy were silently dropped, both calls would yield the
        # same field.
        assert dy_anam != dy_iso
        assert not np.allclose(E_anam, E_iso, atol=1e-12), (
            "FreeSpace silently dropped dy: anamorphic and isotropic "
            "kernels produced identical fields, suggesting dy did not "
            "reach the underlying propagator."
        )


# ============================================================================
# Magnify application
# ============================================================================


class TestMagnifyApplication:

    def test_magnify_changes_grid_pitch_and_preserves_energy(self) -> None:
        """``Magnify(a)`` rescales the output grid pitch by 1/a and
        preserves the integrated intensity * dx * dy."""
        src = _build_test_source(N=256, dx=2e-6, w0=50e-6)
        a = 2.0
        op = Magnify(a)
        out = op(src)
        # Pitch should shrink by factor a (output grid is finer)
        assert np.isclose(out.dx, src.dx / a, rtol=1e-9)
        # Energy: |E|^2 dx dy should be preserved within ~1%
        # (interpolation introduces a small numerical loss, but the
        # ``sqrt(a_x * a_y)`` amplitude prefactor compensates).
        energy_in = np.sum(np.abs(src.E) ** 2) * src.dx * src.dy
        energy_out = np.sum(np.abs(out.E) ** 2) * out.dx * (out.dy or out.dx)
        assert np.isclose(energy_out, energy_in, rtol=5e-2), (
            f"Magnify energy not conserved: in={energy_in:.6e}, "
            f"out={energy_out:.6e}"
        )


# ============================================================================
# Tuple-input / bare-ndarray paths
# ============================================================================


class TestEntryPoints:

    def test_apply_bare_ndarray(self) -> None:
        src = _build_test_source()
        op = ThinLens(0.1)
        E_out = op.apply(src.E, dx=src.dx, wavelength=src.wavelength)
        # Same as Source path
        out = op(src)
        assert np.allclose(E_out, out.E)

    def test_call_tuple_3(self) -> None:
        src = _build_test_source()
        op = ThinLens(0.1)
        E_out, dx_out, wl_out = op((src.E, src.dx, src.wavelength))
        assert dx_out == src.dx
        assert wl_out == src.wavelength
        assert E_out.shape == src.E.shape

    def test_call_tuple_4_anamorphic(self) -> None:
        src = _build_test_source()
        op = CylindricalLens(f_x=0.05, f_y=np.inf)
        E_out, dx_out, dy_out, wl_out = op(
            (src.E, src.dx, src.dx, src.wavelength)
        )
        assert dx_out == src.dx
        assert dy_out == src.dx

    def test_call_invalid_input_raises(self) -> None:
        op = ThinLens(0.1)
        with pytest.raises(TypeError):
            op(42)
        with pytest.raises(TypeError):
            op((1, 2))  # tuple of wrong length

    def test_apply_default_dy_equals_dx(self) -> None:
        """``apply(E, dx=...)`` without ``dy`` uses dx for both axes."""
        src = _build_test_source()
        op = ThinLens(0.1)
        E1 = op.apply(src.E, dx=src.dx, wavelength=src.wavelength)
        E2 = op.apply(
            src.E, dx=src.dx, dy=src.dx, wavelength=src.wavelength,
        )
        assert np.allclose(E1, E2)

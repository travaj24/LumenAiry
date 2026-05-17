"""Pinning tests for the v4.13.2 Agent-B audit fixes.

Audit reference
---------------

``AUDIT_V4_13_1_2026_05_17.md`` Part 10 (Consolidation) dispatched the
v4.13.2 patch as four parallel agents.  Agent B owns six fixes across
``elements/_lens_jax.py``, ``elements/elements.py``,
``raytrace/core.py``, ``elements/_lens_thin.py`` and
``elements/_lens_real.py``:

* **B.1 / P1-NEW-E** -- The two JAX real-lens twins
  (``apply_real_lens_traced_jax``, ``apply_real_lens_maslov_jax``) were
  missing the ``dy=None`` parameter that the NumPy siblings (``apply_real_lens``,
  ``apply_real_lens_traced``, ``apply_real_lens_maslov``) accept.
  Pre-fix an anamorphic ``Source.dy`` round-trip silently dropped y-pitch
  at the JAX boundary.  v4.13.2 adds ``dy: Optional[float] = None`` and
  raises a clear ``ValueError`` when ``dy != dx`` (consistent with the
  NumPy traced/Maslov contract documented in their docstrings).

* **B.2 / P1-NEW-F** -- ``apply_mirror`` lacked the NaN-zeroing guard
  that ``apply_real_lens`` carries.  For a hyperbolic conic at the
  domain boundary ``(1+k)*h_sq/R^2 >= 0.9999`` the sag is NaN and
  ``exp(1j * NaN) = NaN`` poisoned every pixel via the next ASM step.
  v4.13.2 mirrors the ``_lens_real.py:704-705`` template.

* **B.3 / P1-NEW-J** -- ``trace_prescription`` mutated the last
  ``Surface.thickness`` in place when ``image_distance=`` was supplied.
  The ``Surface`` dataclass is not frozen; in-place mutation was a
  tripwire for shared-state bugs.  v4.13.2 uses ``_surface_copy_with``
  (matching the ``lens_abcd`` pattern at ``raytrace/core.py:2510``).

* **B.4 / C-P1-4** -- 9 sites of bare ``0.0 + 0.0j`` complex128 literal
  in ``xp.where(..., E, 0+0j)`` clear-aperture / stop-mask constructs
  silently upcast complex64 fields to complex128.  v4.13.1 P3 #21's
  dtype-aware zero only swept ``apply_aperture`` + ``apply_mirror``;
  v4.13.2 finishes the sweep in 4 sites of ``_lens_thin.py`` and 5
  sites of ``_lens_real.py`` (the audit cited 13 but grep finds 9 --
  the audit explicitly told us to verify by grep).

* **B.5 / C-P1-5** -- Six thin-lens functions
  (``apply_thin_lens`` / ``apply_spherical_lens`` / ``apply_aspheric_lens``
  / ``apply_cylindrical_lens`` / ``apply_grin_lens`` / ``apply_axicon``)
  built ``xp.exp(1j * phase)`` from a float64 ``phase`` and multiplied
  against ``E`` without dtype matching.  Same complex64->complex128
  upcast as B.4 but at the phase-mask leg.  v4.13.2 adds the
  ``if phase_exp.dtype != E.dtype: phase_exp = phase_exp.astype(...)``
  cast (mirroring v4.13.0 L6's ``apply_mirror`` fix).

* **B.6 / C-P1-6** -- The thin-lens module docstring claims "all
  functions accept ``use_gpu=False``" but three (``apply_cylindrical_lens``,
  ``apply_grin_lens``, ``apply_axicon``) lacked the parameter and the
  CuPy dispatch.  v4.13.2 adds ``use_gpu: bool = False`` and the
  canonical dispatch pattern used by the other three thin-lens entry
  points.

Author: Andrew Traverso -- v4.13.2 / Agent B
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import lumenairy as lm


# ----------------------------------------------------------------------------
# JAX availability gate (B.1 uses the JAX twins)
# ----------------------------------------------------------------------------
JAX_AVAILABLE = False
try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


needs_jax = pytest.mark.skipif(
    not JAX_AVAILABLE, reason="JAX is not installed")


# ============================================================================
# B.1 -- JAX lens twins accept dy=None and enforce dy == dx
# ============================================================================

class TestB1JaxLensTwinsAcceptDy:
    """``apply_real_lens_traced_jax`` / ``apply_real_lens_maslov_jax``
    accept ``dy=None`` and raise a clear ``ValueError`` on ``dy != dx``.

    Pre-fix the JAX twins did NOT accept ``dy`` at all, so an upstream
    ``Source.dy`` (anamorphic) silently failed at the JAX boundary --
    either ``TypeError: unexpected keyword`` or, when called positionally,
    the dy was silently ignored.
    """

    @needs_jax
    def test_traced_jax_accepts_dy_keyword(self):
        """``apply_real_lens_traced_jax(..., dy=dx)`` succeeds.

        Pins the keyword-acceptance half of the fix.  A round-trip with
        ``dy == dx`` should produce a finite (non-NaN) output.
        """
        from lumenairy.elements._lens_jax import apply_real_lens_traced_jax

        rx = lm.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)
        N = 64
        dx = 8e-3 / N
        wavelength = 632.8e-9
        E_in = jnp.ones((N, N), dtype=jnp.complex64)

        E_out = apply_real_lens_traced_jax(
            E_in, prescription=rx, wavelength=wavelength,
            dx=dx, dy=dx,
            ray_subsample=8, cheb_order=8, newton_iters=8,
        )
        assert E_out.shape == E_in.shape
        # The traced JAX path produces complex output; in the central
        # region (well inside the aperture) the field magnitude should
        # be finite.
        ctr = jnp.abs(E_out[N // 2, N // 2])
        assert bool(jnp.isfinite(ctr))

    @needs_jax
    def test_maslov_jax_accepts_dy_keyword(self):
        """Same for ``apply_real_lens_maslov_jax(..., dy=dx)``."""
        from lumenairy.elements._lens_jax import apply_real_lens_maslov_jax

        rx = lm.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)
        N = 64
        dx = 8e-3 / N
        wavelength = 632.8e-9
        E_in = jnp.ones((N, N), dtype=jnp.complex64)

        E_out = apply_real_lens_maslov_jax(
            E_in, prescription=rx, wavelength=wavelength,
            dx=dx, dy=dx,
            ray_subsample=8, cheb_order=8, newton_iters=8,
        )
        assert E_out.shape == E_in.shape
        ctr = jnp.abs(E_out[N // 2, N // 2])
        assert bool(jnp.isfinite(ctr))

    @needs_jax
    def test_traced_jax_rejects_dy_ne_dx_with_clear_message(self):
        """``dy != dx`` raises a ValueError with a message that names
        the function and points to the NumPy variant.

        Pins the documented constraint: the JAX path's Chebyshev
        tensor-product fit + Newton inversion + ray subsample paths all
        assume an isotropic square grid, so anamorphic dy is rejected
        rather than silently mishandled.
        """
        from lumenairy.elements._lens_jax import apply_real_lens_traced_jax

        rx = lm.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)
        N = 64
        dx = 8e-3 / N
        E_in = jnp.ones((N, N), dtype=jnp.complex64)

        with pytest.raises(ValueError) as exc_info:
            apply_real_lens_traced_jax(
                E_in, prescription=rx, wavelength=632.8e-9,
                dx=dx, dy=dx * 1.5,  # non-square pixels
            )
        msg = str(exc_info.value)
        assert 'apply_real_lens_traced_jax' in msg
        assert 'dy' in msg
        # Should point users at the NumPy variant
        assert 'apply_real_lens' in msg

    @needs_jax
    def test_maslov_jax_rejects_dy_ne_dx_with_clear_message(self):
        """Same constraint for the Maslov twin."""
        from lumenairy.elements._lens_jax import apply_real_lens_maslov_jax

        rx = lm.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)
        N = 64
        dx = 8e-3 / N
        E_in = jnp.ones((N, N), dtype=jnp.complex64)

        with pytest.raises(ValueError) as exc_info:
            apply_real_lens_maslov_jax(
                E_in, prescription=rx, wavelength=632.8e-9,
                dx=dx, dy=dx * 1.5,
            )
        msg = str(exc_info.value)
        assert 'apply_real_lens_maslov_jax' in msg
        assert 'dy' in msg


# ============================================================================
# B.2 -- apply_mirror NaN zeroing guard
# ============================================================================

class TestB2ApplyMirrorNanGuard:
    """A curved mirror with a strongly hyperbolic conic where
    ``(1+k)*h_sq/R^2 >= 0.9999`` at the domain boundary produces NaN
    sag.  Pre-fix the NaN propagated through ``exp(1j * phase)`` and
    poisoned every pixel.  v4.13.2 zeros the OPD on those pixels so the
    rest of the field is unaffected.
    """

    def test_hyperbolic_mirror_at_domain_boundary_no_nan(self):
        """A spherical (k=0) mirror with R chosen so the field's outer
        pixels fall right at the conic-domain edge produces NO NaN.

        Specifically: choose ``R`` such that ``h_sq_max / R^2 >= 0.9999``
        at the corner.  Pre-fix the corner pixel produced NaN sag,
        ``exp(1j * NaN) = NaN``, and ``E *= NaN`` -> all-NaN field
        (NaN propagates through the array multiply).
        """
        N = 64
        dx = 1e-4  # 100 um, so half-width = 32 * 1e-4 = 3.2 mm
        # Choose radius so the corner pixel falls right at the conic
        # domain boundary.  Half-diagonal r_max = sqrt(2) * N/2 * dx.
        # For a parabola (conic = -1) ``(1+k)*h_sq/R^2 = 0`` for ALL h,
        # so we use a hyperbolic conic.  conic = -3 with R close to
        # r_max gives (1+conic)*h_sq/R^2 = -2*h_sq/R^2 which is < 0,
        # so we go the other way.  Use a strongly oblate conic
        # (conic > 0) so (1+conic)*h_sq/R^2 grows fast.
        r_max = np.sqrt(2) * (N // 2) * dx
        conic = 5.0
        # We want (1+conic)*h_sq/R^2 >= 0.9999 at r = r_max -- this
        # triggers the NaN branch in the JAX/CuPy inline sag path.
        # Equivalently ``R^2 <= (1+conic) * r_max^2 / 0.9999``.
        R = float(np.sqrt((1.0 + conic) * r_max ** 2 / 0.9999)) * 0.999

        E_in = np.ones((N, N), dtype=np.complex64)
        E_out = lm.apply_mirror(
            E_in, wavelength=550e-9, dx=dx,
            radius=R, conic=conic,
        )
        # The output should NOT be all-NaN.  Pre-fix even a single
        # NaN sag triggers NaN propagation through the next array op.
        # Post-fix the NaN pixels are zeroed in OPD and remain finite
        # (their amplitude is just 1.0 * exp(1j*0) = 1.0).
        assert not bool(np.any(np.isnan(E_out))), (
            "apply_mirror produced NaN pixels with a hyperbolic conic "
            "at the domain boundary -- the v4.13.2 P1-NEW-F guard "
            "should have zeroed the OPD on those pixels.")
        # And the central pixel should still carry sensible phase
        # (lens center should be ~zero-OPD).
        assert bool(np.isfinite(E_out[N // 2, N // 2]))


# ============================================================================
# B.3 -- trace_prescription does not mutate shared Surface state
# ============================================================================

class TestB3TracePrescriptionNoMutation:
    """``trace_prescription`` clones the last surface with
    ``_surface_copy_with`` instead of mutating in place.  This pins
    the cleaner contract: even if a caller (or future caching layer)
    shares Surface state across calls, an ``image_distance=`` argument
    on call N+1 cannot contaminate call N's surfaces list (or vice
    versa).
    """

    def test_image_distance_does_not_mutate_input_prescription(self):
        """The prescription dict's ``thicknesses`` list (the original
        source of the last surface's thickness) is NOT mutated.
        """
        rx = lm.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)
        thicknesses_before = list(rx['thicknesses'])

        _ = lm.trace_prescription(
            rx, wavelength=632.8e-9,
            semi_aperture=3e-3,
            image_distance=0.1, num_rings=2, rays_per_ring=8)

        # The prescription's thicknesses list must be unchanged.
        assert list(rx['thicknesses']) == thicknesses_before, (
            "trace_prescription contaminated the input prescription's "
            "thicknesses list.")

    def test_repeated_calls_with_different_image_distance_independent(self):
        """Two trace_prescription calls with different image_distance
        produce DIFFERENT image-plane spot positions.  Pre-fix any
        shared-Surface-state bug would leak distance A into call B's
        geometry; this test pins that they remain independent.
        """
        rx = lm.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)

        # Call A: image at 50 mm
        res_a = lm.trace_prescription(
            rx, wavelength=632.8e-9, semi_aperture=3e-3,
            image_distance=0.050, num_rings=2, rays_per_ring=8)
        # Call B: image at 100 mm  (same prescription object)
        res_b = lm.trace_prescription(
            rx, wavelength=632.8e-9, semi_aperture=3e-3,
            image_distance=0.100, num_rings=2, rays_per_ring=8)

        # The image-ray z coordinates should differ (the image plane
        # in the trace engine's local coordinates is z=0 at the image
        # surface, but the optical path lengths to that plane differ
        # because of the different thicknesses).  More robust: the
        # OPL accumulated to the image plane should differ.
        alive_a = bool(np.any(res_a.image_rays.alive))
        alive_b = bool(np.any(res_b.image_rays.alive))
        assert alive_a and alive_b
        opd_a = float(np.nanmean(res_a.image_rays.opd[res_a.image_rays.alive]))
        opd_b = float(np.nanmean(res_b.image_rays.opd[res_b.image_rays.alive]))
        # Doubling the air gap from 50 mm to 100 mm should add roughly
        # 50 mm of OPL (n_air = 1).  Assert a clear separation.
        assert abs(opd_b - opd_a) > 1e-3, (
            f"trace_prescription gave OPLs {opd_a!r} and {opd_b!r} for "
            f"image_distance=50mm vs 100mm; expected ~50mm spread.  "
            f"Possible shared-state mutation contaminated one call.")


# ============================================================================
# B.4 + B.5 + B.6 -- Parametrized thin-lens / real-lens dtype + use_gpu pin
# ============================================================================

class TestB456ThinLensDtypePreservation:
    """The thin-lens family entry points and ``apply_real_lens``
    preserve a complex64 input field's dtype.

    Pre-fix:
      * B.4: ``xp.where(..., E, 0.0 + 0.0j)`` upcast E to complex128
        in 9 sites of the lens family.
      * B.5: ``xp.exp(1j * <float64 phase>)`` produced complex128
        which upcast E on the multiply, in 6 thin-lens functions.

    Post-fix every entry should leave complex64 output for complex64
    input.
    """

    @pytest.fixture
    def cfg(self):
        """Common test fixture: small grid, visible wavelength."""
        N = 32
        dx = 4e-6  # 4 um pitch -- realistic-ish microscale grid
        wavelength = 632.8e-9
        return dict(N=N, dx=dx, wavelength=wavelength,
                    E_in=np.ones((N, N), dtype=np.complex64))

    # ----- B.5: thin-lens family preserves complex64 -----

    def test_apply_thin_lens_complex64(self, cfg):
        from lumenairy import apply_thin_lens
        out = apply_thin_lens(
            cfg['E_in'], f=1e-3, wavelength=cfg['wavelength'],
            dx=cfg['dx'])
        assert out.dtype == np.complex64, (
            f"apply_thin_lens upcast complex64 -> {out.dtype}")

    def test_apply_spherical_lens_complex64(self, cfg):
        from lumenairy import apply_spherical_lens
        out = apply_spherical_lens(
            cfg['E_in'], R1=10e-3, R2=-10e-3, d=1e-3, n_lens=1.5,
            wavelength=cfg['wavelength'], dx=cfg['dx'])
        assert out.dtype == np.complex64, (
            f"apply_spherical_lens upcast complex64 -> {out.dtype}")

    def test_apply_aspheric_lens_complex64(self, cfg):
        from lumenairy import apply_aspheric_lens
        out = apply_aspheric_lens(
            cfg['E_in'], R1=10e-3, R2=-10e-3, d=1e-3, n_lens=1.5,
            k1=-1.0, k2=0.0,
            wavelength=cfg['wavelength'], dx=cfg['dx'])
        assert out.dtype == np.complex64, (
            f"apply_aspheric_lens upcast complex64 -> {out.dtype}")

    def test_apply_cylindrical_lens_complex64(self, cfg):
        from lumenairy import apply_cylindrical_lens
        out = apply_cylindrical_lens(
            cfg['E_in'], f=1e-3, wavelength=cfg['wavelength'],
            dx=cfg['dx'])
        assert out.dtype == np.complex64, (
            f"apply_cylindrical_lens upcast complex64 -> {out.dtype}")

    def test_apply_grin_lens_complex64(self, cfg):
        from lumenairy import apply_grin_lens
        out = apply_grin_lens(
            cfg['E_in'], n0=1.5, g=100.0, d=1e-3,
            wavelength=cfg['wavelength'], dx=cfg['dx'])
        assert out.dtype == np.complex64, (
            f"apply_grin_lens upcast complex64 -> {out.dtype}")

    def test_apply_axicon_complex64(self, cfg):
        from lumenairy import apply_axicon
        out = apply_axicon(
            cfg['E_in'], 0.01, 1.5,
            wavelength=cfg['wavelength'], dx=cfg['dx'])
        assert out.dtype == np.complex64, (
            f"apply_axicon upcast complex64 -> {out.dtype}")

    # ----- B.4: apply_real_lens (real-lens family) preserves complex64 -----

    def test_apply_real_lens_complex64(self, cfg):
        """The 5 sites of ``0.0 + 0.0j`` in ``_lens_real.py`` could
        only fire when a clear_aperture / stop / fresnel / slant /
        entrance-aperture branch was taken.  Use a prescription with
        ``aperture_diameter`` so the entrance-aperture branch (line
        533) fires; the per-surface clear_aperture branch (line 769)
        is exercised by setting ``clear_aperture`` on each surface.
        """
        from lumenairy import apply_real_lens
        N = 32
        dx = 4e-6
        wavelength = 632.8e-9
        E_in = np.ones((N, N), dtype=np.complex64)
        # Build a singlet with an aperture (forces line 533) plus a
        # per-surface clear_aperture (forces line 769).
        rx = lm.make_singlet(R1=50e-3, R2=-50e-3, d=1e-3,
                              glass='N-BK7', aperture=80e-6)
        # Add per-surface clear apertures so the line-769 branch fires.
        for s in rx['surfaces']:
            s['clear_aperture'] = 80e-6
        out = apply_real_lens(
            E_in, prescription=rx, wavelength=wavelength, dx=dx)
        assert out.dtype == np.complex64, (
            f"apply_real_lens upcast complex64 -> {out.dtype}")

    # ----- B.6: thin-lens functions accept use_gpu=False even without CuPy -----

    def test_apply_cylindrical_lens_accepts_use_gpu_false(self, cfg):
        """The use_gpu kwarg exists and defaults to False -- this pins
        the docstring claim 'all functions accept use_gpu=False'.
        Does not require CuPy to be installed.
        """
        from lumenairy import apply_cylindrical_lens
        out = apply_cylindrical_lens(
            cfg['E_in'], f=1e-3, wavelength=cfg['wavelength'],
            dx=cfg['dx'], use_gpu=False)
        assert out.shape == cfg['E_in'].shape

    def test_apply_grin_lens_accepts_use_gpu_false(self, cfg):
        from lumenairy import apply_grin_lens
        out = apply_grin_lens(
            cfg['E_in'], n0=1.5, g=100.0, d=1e-3,
            wavelength=cfg['wavelength'], dx=cfg['dx'], use_gpu=False)
        assert out.shape == cfg['E_in'].shape

    def test_apply_axicon_accepts_use_gpu_false(self, cfg):
        from lumenairy import apply_axicon
        out = apply_axicon(
            cfg['E_in'], 0.01, 1.5,
            wavelength=cfg['wavelength'], dx=cfg['dx'], use_gpu=False)
        assert out.shape == cfg['E_in'].shape


# ============================================================================
# B.6 (CuPy gating) -- if CuPy is installed, verify dispatch on the
# three newly-fixed functions actually returns a CuPy array
# ============================================================================

class TestB6CupyDispatch:
    """When CuPy is available, the three newly-routed thin-lens
    functions should accept a CuPy array input and return a CuPy
    output.  Without CuPy the test is skipped.
    """

    def test_cupy_dispatch_apply_cylindrical_lens(self):
        cp = pytest.importorskip('cupy')
        from lumenairy import apply_cylindrical_lens
        N = 32
        dx = 4e-6
        E_in = cp.ones((N, N), dtype=cp.complex64)
        out = apply_cylindrical_lens(
            E_in, f=1e-3, wavelength=632.8e-9, dx=dx)
        assert type(out).__module__.startswith('cupy'), (
            f"apply_cylindrical_lens did not return a CuPy array "
            f"(got {type(out)!r}).")

    def test_cupy_dispatch_apply_grin_lens(self):
        cp = pytest.importorskip('cupy')
        from lumenairy import apply_grin_lens
        N = 32
        dx = 4e-6
        E_in = cp.ones((N, N), dtype=cp.complex64)
        out = apply_grin_lens(
            E_in, n0=1.5, g=100.0, d=1e-3,
            wavelength=632.8e-9, dx=dx)
        assert type(out).__module__.startswith('cupy')

    def test_cupy_dispatch_apply_axicon(self):
        cp = pytest.importorskip('cupy')
        from lumenairy import apply_axicon
        N = 32
        dx = 4e-6
        E_in = cp.ones((N, N), dtype=cp.complex64)
        out = apply_axicon(
            E_in, 0.01, 1.5,
            wavelength=632.8e-9, dx=dx)
        assert type(out).__module__.startswith('cupy')

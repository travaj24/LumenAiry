"""v5.2 ROADMAP physics-correctness closures (AUDIT_V4_13_1 deferred).

One test class per fix.  Pins:

* ``TestV5_2_P1_G_DOESignPreservation`` -- ``apply_doe_phase_traced``
  preserves the sign of the longitudinal direction cosine like the
  inline DOE kick in :func:`lumenairy.raytrace.trace`.

* ``TestV5_2_P1_1_MultiPrescriptionScaleFloor`` --
  :class:`MultiPrescriptionParameterization` exposes ``scale_floor``
  with per-variable defaults (radii / thicknesses 1e-6, conics /
  aspherics 1e-3) so a near-zero parameter does not collapse the
  optimiser's FD step.

* ``TestV5_2_P1_A_OutputShapeRename`` -- the GBD / HFPI / HF
  sub-propagators accept the new ``output_shape`` kwarg and emit a
  ``DeprecationWarning`` on the legacy ``output_grid`` spelling.

* ``TestV5_2_P1_C_MaslovSubdomainGridLoss`` --
  :func:`prescription_subdomain` accepts an output grid that differs
  from the input grid when ``method='maslov'`` (v5.2.3 substantive
  closure -- the maslov branch resamples the kernel-native (input-grid)
  output onto ``out_surface``).  v5.2.0 raised ``ValueError`` here
  (option a); v5.2.3 ships option b (substantive resampling).  The
  narrow retained-raise covers non-square out_surface, which the
  maslov kernel's downstream resampler cannot handle.

* ``TestV5_2_P1_F_PartitionOfUnitySourcePlaneWarning`` --
  :func:`propagate_subaperture_asymptotic` emits a ``UserWarning`` on
  non-unit-magnification systems, documenting the limitation that the
  partition-of-unity windows are centred on SOURCE-plane positions.
  Also pins the new ``image_centres`` / ``image_half_widths`` kwargs
  on :func:`combine_patch_fields`.

All comments / strings cp1252-safe.  Citation: ``v5.2 (AUDIT_V4_13_1
P1-<X> closure)``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

# ===========================================================================
# Fix 1: apply_doe_phase_traced sign preservation (AUDIT_V4_13_1 P1-G)
# ===========================================================================

class TestV5_2_P1_G_DOESignPreservation:
    """``apply_doe_phase_traced`` must preserve sign(N) on the
    longitudinal direction cosine.  Pre-v5.2 the function always
    returned ``N_new = +sqrt(1 - L^2 - M^2)``; the inline DOE kick in
    :func:`trace` (line ~193) preserves sign via
    ``r.N = np.where(r.N < 0, -_N_new, _N_new)``.  The
    ``apply_doe_phase_traced`` parallel was missed."""

    def test_negative_first_order_keeps_negative_N_for_backward_ray(self):
        """A ray with N < 0 (backward-propagating) should still have
        N < 0 after a small first-order DOE kick."""
        from lumenairy.raytrace import RayBundle, apply_doe_phase_traced

        wavelength = 1.31e-6
        period = 30e-6
        # Single ray; direction = (L, M, N) = (0, 0, -1) (pure -z).
        rays = RayBundle(
            x=np.array([0.0]),
            y=np.array([0.0]),
            z=np.array([0.0]),
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([-1.0]),
            wavelength=wavelength,
            alive=np.array([True]),
            opd=np.array([0.0]),
        )
        out = apply_doe_phase_traced(
            rays, order_x=-1, order_y=0,
            period_x=period, wavelength=wavelength,
        )
        assert out.N.shape == (1,)
        # Expected: L_new = -lambda/period, M_new = 0,
        # |N_new| = sqrt(1 - (lambda/period)^2), sign = negative
        # (matches the input).
        L_exp = -1.0 * wavelength / period
        N_mag = float(np.sqrt(1.0 - L_exp ** 2))
        assert out.L[0] == pytest.approx(L_exp, rel=1e-12)
        assert out.N[0] < 0, (
            "apply_doe_phase_traced must preserve sign(N) for backward-"
            "propagating rays; v5.2 (AUDIT_V4_13_1 P1-G closure).")
        assert abs(out.N[0]) == pytest.approx(N_mag, rel=1e-12)

    def test_positive_first_order_keeps_positive_N_for_forward_ray(self):
        """Forward-propagating rays (N > 0) keep N > 0 -- the
        legacy path was already correct here; this pins the
        invariant under the v5.2 sign restoration."""
        from lumenairy.raytrace import RayBundle, apply_doe_phase_traced

        wavelength = 1.31e-6
        period = 30e-6
        rays = RayBundle(
            x=np.array([0.0]),
            y=np.array([0.0]),
            z=np.array([0.0]),
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([1.0]),
            wavelength=wavelength,
            alive=np.array([True]),
            opd=np.array([0.0]),
        )
        out = apply_doe_phase_traced(
            rays, order_x=1, order_y=0,
            period_x=period, wavelength=wavelength,
        )
        assert out.N[0] > 0

    def test_inline_kick_and_apply_match_sign_convention(self):
        """The inline DOE kick in :func:`trace` and
        :func:`apply_doe_phase_traced` should produce the same
        sign(N) under the same input.  v5.2 closure aligns the
        two."""
        from lumenairy.raytrace import RayBundle, apply_doe_phase_traced

        wavelength = 1.31e-6
        period = 30e-6
        m = -1
        # Backward ray
        rays = RayBundle(
            x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]),
            L=np.array([0.0]), M=np.array([0.0]), N=np.array([-1.0]),
            wavelength=wavelength,
            alive=np.array([True]), opd=np.array([0.0]),
        )
        out = apply_doe_phase_traced(
            rays, order_x=m, order_y=0,
            period_x=period, wavelength=wavelength,
        )

        # Reproduce the inline-kick computation from trace.py:
        # _N_new = sqrt(1 - L^2 - M^2); r.N = np.where(r.N < 0, -_N_new,
        # _N_new).
        L_in = 0.0
        M_in = 0.0
        N_in = -1.0
        dL = m * wavelength / period
        dM = 0.0
        L_new = L_in + dL
        M_new = M_in + dM
        sumsq = L_new ** 2 + M_new ** 2
        _N_new = float(np.sqrt(max(0.0, 1.0 - sumsq)))
        N_expected = -_N_new if N_in < 0 else _N_new

        assert out.N[0] == pytest.approx(N_expected, rel=1e-12)


# ===========================================================================
# Fix 2: MultiPrescriptionParameterization.scale_floor
#        (AUDIT_V4_13_1 P1-1)
# ===========================================================================

class TestV5_2_P1_1_MultiPrescriptionScaleFloor:
    """``MultiPrescriptionParameterization`` exposes ``scale_floor``
    with path-aware defaults so a near-zero radius / thickness /
    conic / aspheric coefficient does not collapse ``x_scale[i]`` to
    zero in the FD-gradient step."""

    def _make_planar_template(self):
        """Two-surface planar prescription with a near-zero radius
        slot we can target as a free variable."""
        return {
            'surfaces': [
                {'radius': 1e-9, 'glass_before': 'air',
                 'glass_after': 'N-BK7'},
                {'radius': -50e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air'},
            ],
            'thicknesses': [5e-3, 100e-3],
        }

    def test_default_scale_floor_picks_radius_floor_for_near_zero_radius(self):
        """A free var on a 1e-9 m radius should land on the 1 um
        radius floor (1e-6 m), not collapse to 0."""
        from lumenairy.optimize.parameterizations import (
            MultiPrescriptionParameterization,
        )

        pres0 = self._make_planar_template()
        pres1 = self._make_planar_template()
        param = MultiPrescriptionParameterization(
            templates=[pres0, pres1],
            free_vars=[
                (0, 'surfaces', 0, 'radius'),  # near-zero radius (1e-9)
            ],
        )
        assert hasattr(param, 'scale_floor')
        assert param.scale_floor is not None
        sf = np.asarray(param.scale_floor)
        assert sf.shape == (1,)
        # 1e-6 m floor for radius slots.
        assert sf[0] == pytest.approx(1e-6, rel=1e-12)
        # Critically: the floor must be > the parameter magnitude
        # (1e-9 < 1e-6) so the FD step uses the floor, not 0.
        assert sf[0] > 1e-9

    def test_conic_default_floor_is_1e_minus_3(self):
        """Conic free-vars get the dimensionless 1e-3 floor."""
        from lumenairy.optimize.parameterizations import (
            MultiPrescriptionParameterization,
        )

        pres = {
            'surfaces': [
                {'radius': 50e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0},
                {'radius': -50e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air'},
            ],
            'thicknesses': [5e-3, 100e-3],
        }
        param = MultiPrescriptionParameterization(
            templates=[pres],
            free_vars=[(0, 'surfaces', 0, 'conic')],
        )
        sf = np.asarray(param.scale_floor)
        assert sf[0] == pytest.approx(1e-3, rel=1e-12)

    def test_thickness_default_floor_is_1_micron(self):
        from lumenairy.optimize.parameterizations import (
            MultiPrescriptionParameterization,
        )

        pres = self._make_planar_template()
        param = MultiPrescriptionParameterization(
            templates=[pres],
            free_vars=[(0, 'thicknesses', 0)],
        )
        sf = np.asarray(param.scale_floor)
        assert sf[0] == pytest.approx(1e-6, rel=1e-12)

    def test_user_supplied_array_overrides_default(self):
        from lumenairy.optimize.parameterizations import (
            MultiPrescriptionParameterization,
        )

        pres = self._make_planar_template()
        param = MultiPrescriptionParameterization(
            templates=[pres],
            free_vars=[
                (0, 'surfaces', 0, 'radius'),
                (0, 'thicknesses', 0),
            ],
            scale_floor=[2e-5, 3e-7],
        )
        sf = np.asarray(param.scale_floor)
        assert sf[0] == pytest.approx(2e-5)
        assert sf[1] == pytest.approx(3e-7)

    def test_user_scalar_broadcasts(self):
        from lumenairy.optimize.parameterizations import (
            MultiPrescriptionParameterization,
        )

        pres = self._make_planar_template()
        param = MultiPrescriptionParameterization(
            templates=[pres],
            free_vars=[
                (0, 'surfaces', 0, 'radius'),
                (0, 'thicknesses', 0),
            ],
            scale_floor=5e-5,
        )
        sf = np.asarray(param.scale_floor)
        assert np.all(sf == pytest.approx(5e-5))
        assert sf.shape == (2,)

    def test_user_array_wrong_length_raises(self):
        from lumenairy.optimize.parameterizations import (
            MultiPrescriptionParameterization,
        )

        pres = self._make_planar_template()
        with pytest.raises(ValueError):
            MultiPrescriptionParameterization(
                templates=[pres],
                free_vars=[(0, 'surfaces', 0, 'radius')],
                scale_floor=[1e-6, 1e-6, 1e-6],  # length 3, n=1
            )

    def test_driver_picks_up_scale_floor_via_getattr(self):
        """The driver's _fd_grad_for() does ``getattr(parameterization,
        'scale_floor', None)``; v5.2 makes that return the per-slot
        array instead of ``None``.  This pins the integration: a near-
        zero radius slot does NOT collapse x_scale[i]."""
        from lumenairy.optimize.parameterizations import (
            DesignParameterization,
            MultiPrescriptionParameterization,
        )

        pres = self._make_planar_template()
        single = DesignParameterization(
            template=pres,
            free_vars=[('surfaces', 0, 'radius')],
        )
        multi = MultiPrescriptionParameterization(
            templates=[pres],
            free_vars=[(0, 'surfaces', 0, 'radius')],
        )
        # Both expose scale_floor; both deliver the radius floor.
        for p in (single, multi):
            sf = getattr(p, 'scale_floor', None)
            assert sf is not None
            assert float(np.asarray(sf)[0]) == pytest.approx(1e-6)


# ===========================================================================
# Fix 3: output_grid -> output_shape rename
#        (AUDIT_V4_13_1 Part 2 P1-A)
# ===========================================================================

class TestV5_2_P1_A_OutputShapeRename:
    """The GBD / HFPI / HF prescription-aware sub-propagators rename
    ``output_grid`` -> ``output_shape``.  Legacy ``output_grid`` calls
    emit a ``DeprecationWarning`` directing users to ``output_shape``
    (for the (Ny, Nx) shape-only meaning) or
    ``propagate(output_grid=(N, dx_out))`` (for the dispatcher's grid
    spec)."""

    def test_propagate_gbd_freespace_accepts_output_shape(self):
        from lumenairy.propagators.gbd import propagate_gbd_freespace
        E = np.ones((32, 32), dtype=np.complex128)
        # No warning should fire on the new spelling.
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            out = propagate_gbd_freespace(
                E, dx=10e-6, z=1e-3, wavelength=633e-9,
                output_shape=(16, 16),
            )
        assert out.shape == (16, 16)

    def test_propagate_gbd_freespace_legacy_output_grid_warns(self):
        from lumenairy.propagators.gbd import propagate_gbd_freespace
        E = np.ones((32, 32), dtype=np.complex128)
        with pytest.warns(DeprecationWarning, match='output_shape'):
            out = propagate_gbd_freespace(
                E, dx=10e-6, z=1e-3, wavelength=633e-9,
                output_grid=(16, 16),
            )
        assert out.shape == (16, 16)

    def test_propagate_gbd_freespace_both_kwargs_raises(self):
        from lumenairy.propagators.gbd import propagate_gbd_freespace
        E = np.ones((32, 32), dtype=np.complex128)
        with pytest.raises(ValueError, match='both'):
            propagate_gbd_freespace(
                E, dx=10e-6, z=1e-3, wavelength=633e-9,
                output_shape=(16, 16), output_grid=(16, 16),
            )

    def test_propagate_hfpi_freespace_aperture_accepts_output_shape(self):
        from lumenairy.propagators.hfpi import (
            propagate_hfpi_freespace_aperture,
        )
        E = np.ones((16, 16), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            out = propagate_hfpi_freespace_aperture(
                E, dx=20e-6,
                z_to_aperture=1e-3,
                aperture_radius=200e-6,
                z_aperture_to_output=1e-3,
                wavelength=633e-9,
                n_paths=8,
                rng=0,
                output_shape=(8, 8),
            )
        assert out.shape == (8, 8)

    def test_propagate_hfpi_freespace_aperture_legacy_warns(self):
        from lumenairy.propagators.hfpi import (
            propagate_hfpi_freespace_aperture,
        )
        E = np.ones((16, 16), dtype=np.complex128)
        with pytest.warns(DeprecationWarning, match='output_shape'):
            out = propagate_hfpi_freespace_aperture(
                E, dx=20e-6,
                z_to_aperture=1e-3,
                aperture_radius=200e-6,
                z_aperture_to_output=1e-3,
                wavelength=633e-9,
                n_paths=8,
                rng=0,
                output_grid=(8, 8),
            )
        assert out.shape == (8, 8)


# ===========================================================================
# Fix 4: MHS prescription_subdomain maslov-method grid-loss
#        (AUDIT_V4_13_1 Part 2 P1-C)
# ===========================================================================

class TestV5_2_P1_C_MaslovSubdomainGridLoss:
    """``prescription_subdomain(method='maslov')`` accepts a different
    output Huygens surface and resamples kernel-native (input-grid)
    output onto it (v5.2.3 substantive closure).  v5.2.0 raised here
    (option a -- block); v5.2.3 ships option b (substantive resample).

    The narrow retained-raise covers non-square out_surface; the maslov
    kernel itself requires square pixels.  See the standalone
    ``test_v5_2_3_mhs_maslov_resampling.py`` for the full physics-pin
    coverage of the new resampling behaviour."""

    def _make_surface(self, Ny, Nx, dx):
        from lumenairy.propagators.mhs import HuygensSurface
        return HuygensSurface(z=0.0, dx=dx, Ny=Ny, Nx=Nx)

    def test_maslov_with_different_output_grid_does_not_raise(self):
        """v5.2.3 (AUDIT_V4_13_1 P1-C substantive closure): pre-v5.2.3
        this raised ``ValueError`` (option a -- block).  v5.2.3 now
        resamples the kernel-native input-grid output onto the declared
        ``out_surface`` instead."""
        from lumenairy.propagators.mhs import prescription_subdomain
        in_s = self._make_surface(64, 64, 10e-6)
        out_s = self._make_surface(128, 128, 5e-6)
        pres = {
            'surfaces': [
                {'radius': 50e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7'},
                {'radius': -50e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air'},
            ],
            'thicknesses': [5e-3, 100e-3],
        }
        # Construction must succeed; the resample fires at run-time.
        sub = prescription_subdomain(
            in_s, out_s, pres,
            wavelength=633e-9, method='maslov',
        )
        assert sub is not None
        assert sub.in_surface.Nx == 64
        assert sub.out_surface.Nx == 128

    def test_maslov_same_grid_does_not_raise(self):
        """Same-grid maslov call -- the resampler is skipped and the
        kernel-native output is returned directly."""
        from lumenairy.propagators.mhs import prescription_subdomain
        in_s = self._make_surface(64, 64, 10e-6)
        out_s = self._make_surface(64, 64, 10e-6)
        pres = {
            'surfaces': [
                {'radius': 50e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7'},
                {'radius': -50e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air'},
            ],
            'thicknesses': [5e-3, 100e-3],
        }
        # Should not raise.
        sub = prescription_subdomain(
            in_s, out_s, pres,
            wavelength=633e-9, method='maslov',
        )
        assert sub is not None

    def test_non_maslov_with_different_grid_does_not_raise(self):
        """Non-maslov methods that natively support output-grid
        resampling (gbd / hfpi / hf) keep their native code path
        (unchanged by v5.2.3)."""
        from lumenairy.propagators.mhs import prescription_subdomain
        in_s = self._make_surface(64, 64, 10e-6)
        out_s = self._make_surface(128, 128, 5e-6)
        pres = {
            'surfaces': [
                {'radius': 50e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7'},
                {'radius': -50e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air'},
            ],
            'thicknesses': [5e-3, 100e-3],
        }
        # Construction-only test; we don't run the pipeline.
        sub = prescription_subdomain(
            in_s, out_s, pres,
            wavelength=633e-9, method='gbd',
        )
        assert sub is not None

    def test_maslov_non_square_out_surface_raises(self):
        """Narrow retained-raise corner case (v5.2.3): the maslov kernel
        requires square pixels and its downstream resampler is built
        on a square ``N_out`` -- a non-square output grid is rejected
        at construction time."""
        from lumenairy.propagators.mhs import prescription_subdomain
        in_s = self._make_surface(64, 64, 10e-6)
        out_s = self._make_surface(128, 64, 5e-6)  # Ny != Nx
        pres = {
            'surfaces': [
                {'radius': 50e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7'},
                {'radius': -50e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air'},
            ],
            'thicknesses': [5e-3, 100e-3],
        }
        with pytest.raises(ValueError, match='square'):
            prescription_subdomain(
                in_s, out_s, pres,
                wavelength=633e-9, method='maslov',
            )


# ===========================================================================
# Fix 5: partition-of-unity convention (AUDIT_V4_13_1 Part 2 P1-F)
# ===========================================================================

class TestV5_2_P1_F_PartitionOfUnitySourcePlaneWarning:
    """v5.2.0 P1-F pinning: the new ``image_centres`` /
    ``image_half_widths`` kwargs on :func:`combine_patch_fields` give
    callers an escape hatch from the legacy source-plane partition-of-
    unity behaviour.

    .. versionchanged:: 5.2.3
        v5.2.3 closed P1-F substantively by computing the image-plane
        centres from the system ABCD internally inside
        :func:`propagate_subaperture_asymptotic`, so the original
        v5.2.0 ``UserWarning`` is silenced for the typical-call case
        (2x telephoto, 0.5x condenser, etc.).  The warning is now
        retained only as a fallback for the corner case where the
        paraxial ABCD probe itself fails.  See
        ``test_v5_2_3_subaperture_image_plane.py`` for the substantive
        closure pins; this class continues to pin the
        :func:`combine_patch_fields` kwarg shape contract (opt-in
        kwargs take precedence over the auto-computed values, and
        the default ``None`` path remains bit-for-bit legacy)."""

    def test_combine_patch_fields_accepts_image_centres(self):
        """The new ``image_centres`` kwarg overrides the legacy
        source-plane patch_grid centres."""
        from lumenairy.propagators.subaperture import (
            combine_patch_fields,
            patches_for_box,
        )
        pg = patches_for_box(
            box_size=(200e-6, 200e-6),
            patch_size=(100e-6, 100e-6),
            overlap=0.25,
        )
        n_patch = len(pg)
        # Dummy unit-amplitude patch fields on a small output grid.
        Ny, Nx = 16, 16
        ox = (np.arange(Nx) - Nx / 2) * 5e-6
        oy = (np.arange(Ny) - Ny / 2) * 5e-6
        patch_fields = [np.ones((Ny, Nx), dtype=np.complex128)
                        for _ in range(n_patch)]
        # 2x magnification: image centres are twice the source ones.
        img_centres = np.asarray(pg.centres) * 2.0
        img_half_widths = np.asarray(pg.half_widths) * 2.0
        out = combine_patch_fields(
            patch_fields, pg,
            output_grid_x=ox, output_grid_y=oy,
            image_centres=img_centres,
            image_half_widths=img_half_widths,
        )
        assert out.shape == (Ny, Nx)
        # Sanity: result is finite.
        assert np.all(np.isfinite(out.real))
        assert np.all(np.isfinite(out.imag))

    def test_combine_patch_fields_image_centres_shape_mismatch_raises(self):
        from lumenairy.propagators.subaperture import (
            combine_patch_fields,
            patches_for_box,
        )
        pg = patches_for_box(
            box_size=(200e-6, 200e-6),
            patch_size=(100e-6, 100e-6),
            overlap=0.25,
        )
        n_patch = len(pg)
        Ny, Nx = 8, 8
        ox = (np.arange(Nx) - Nx / 2) * 5e-6
        oy = (np.arange(Ny) - Ny / 2) * 5e-6
        patch_fields = [np.ones((Ny, Nx), dtype=np.complex128)
                        for _ in range(n_patch)]
        with pytest.raises(ValueError, match='image_centres'):
            combine_patch_fields(
                patch_fields, pg,
                output_grid_x=ox, output_grid_y=oy,
                image_centres=np.zeros((n_patch + 1, 2)),
            )

    def test_combine_patch_fields_default_path_is_legacy_bit_for_bit(self):
        """When ``image_centres`` is None, the function must return
        the same result as the v5.1 / pre-v5.2 implementation (which
        always used ``patch_grid.centres``)."""
        from lumenairy.propagators.subaperture import (
            combine_patch_fields,
            patch_window,
            patches_for_box,
        )
        pg = patches_for_box(
            box_size=(200e-6, 200e-6),
            patch_size=(100e-6, 100e-6),
            overlap=0.25,
        )
        n_patch = len(pg)
        Ny, Nx = 16, 16
        ox = (np.arange(Nx) - Nx / 2) * 5e-6
        oy = (np.arange(Ny) - Ny / 2) * 5e-6
        rng = np.random.default_rng(seed=42)
        patch_fields = [
            (rng.standard_normal((Ny, Nx))
             + 1j * rng.standard_normal((Ny, Nx))).astype(np.complex128)
            for _ in range(n_patch)
        ]

        out_new = combine_patch_fields(
            patch_fields, pg,
            output_grid_x=ox, output_grid_y=oy,
            edge_smoothness=0.1,
        )

        # Reproduce the legacy summation by hand using patch_window
        # directly.
        X, Y = np.meshgrid(ox, oy, indexing='xy')
        acc = np.zeros((Ny, Nx), dtype=np.complex128)
        w_tot = np.zeros((Ny, Nx), dtype=np.float64)
        for i, F in enumerate(patch_fields):
            centre = (float(pg.centres[i, 0]), float(pg.centres[i, 1]))
            hw = (float(pg.half_widths[i, 0]),
                  float(pg.half_widths[i, 1]))
            w = patch_window(X, Y, centre, hw, edge_smoothness=0.1)
            acc += F * w
            w_tot += w
        w_tot = np.where(w_tot > 1e-12, w_tot, 1.0)
        expected = acc / w_tot

        np.testing.assert_allclose(out_new, expected, rtol=1e-12,
                                   atol=1e-15)

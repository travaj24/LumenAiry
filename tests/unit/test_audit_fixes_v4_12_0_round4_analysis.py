"""Pinning tests for the v4.12.0 round-4 audit Tier-2 analysis-domain
fixes (``AUDIT_ROUND4_2026_05_16.md`` items B2-3, B2-4, B2-5, B1-11,
B2-1, B2-2).

Each test pins a finding from the round-4 audit:

* B2-3 -- ``image_plane_wfe`` reference-sphere radius now carries the
  ``1/N_chief`` factor.  For off-axis chief with ``N_chief < 1`` the
  sphere genuinely passes through the chief image landing; pre-4.12.0
  the radius was the AXIAL distance, so for off-axis fields the
  sphere missed the chief by ``img_d_m * (1 - 1/N_chief)`` and the
  quadratic shape error was absorbed as phantom defocus by
  ``best_rms``.

* B2-4 -- ``distortion_grid`` raises ``ValueError`` up front for
  ``sin(tx)^2 + sin(ty)^2 >= 1`` field directions (e.g.
  ``tx = ty = 45 deg``) instead of swallowing the trace failure in
  ``except: pass`` and returning an all-NaN grid.

* B2-5 -- ``apply_real_lens_traced`` raises ``ValueError`` with a
  mirror-specific message when ``prescription['surfaces']`` carries
  ``is_mirror=True`` (or ``glass_after='MIRROR'``).  The shared
  ``_check_no_silent_fold_drop`` only sees ``prescription['elements']``
  so a hand-built prescription would otherwise sneak past.

* B1-11 -- ``makedammann2d(seed=N)`` does not mutate the user's global
  ``np.random`` state.  Pre-4.12.0 it called ``np.random.seed(seed)``;
  4.12.0 uses ``np.random.default_rng(seed)`` and threads the local
  generator through the function body.

* B2-1 / B2-2 -- ``ghost.py`` module-level and ``ghost_analysis``
  docstrings call out (a) reported intensity is an upper bound
  ignoring transmission losses ``Prod (1 - R_k)``, (b)
  ``focus_z_estimate`` is a heuristic harmonic-mean sort key, not a
  physical focal distance.

These pins guard against regression of the v4.12.0 fixes.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import lumenairy as la


# ============================================================================
# B2-3 -- image_plane_wfe reference-sphere radius carries 1/N_chief
# ============================================================================

class TestImagePlaneWfeSphereRadiusOffAxis:
    """``eval_image_plane_wfe`` for an off-axis chief: the reference
    sphere actually passes through the chief image.

    Pre-4.12.0 the sphere radius was the AXIAL distance ``img_d_m``;
    for an off-axis chief with z-direction cosine ``N_chief < 1`` the
    chief image is at axial-distance ``img_d_m`` from the last
    surface but ``img_d_m / N_chief`` of RAY-ARC length away.  The
    sphere centred on the chief image with radius equal to axial
    distance misses the chief tangent point by ``img_d_m * (1/N -
    1)``; the quadratic shape error was absorbed as phantom defocus
    by ``best_rms``, biasing the reported chief residual.
    """

    def _off_axis_singlet(self):
        """Stop-at-front singlet with object_distance set so that a
        Hx ~= 1 field gives a substantially off-axis chief in image
        space."""
        # Plano-convex BK7 with EFL ~ 100 mm; object at finite
        # conjugate so the chief in image space has notable angle.
        p = la.make_singlet(R1=51.5e-3, R2=float('inf'), d=4e-3,
                            glass='N-BK7', aperture=12e-3)
        p['object_distance'] = 500e-3
        # Field of 20 mm half-height -> chief angle ~ 4 deg (sin ~ 0.07)
        return p

    def test_sphere_radius_includes_inv_N_chief_for_off_axis(self):
        """Pin the resulting ``r_sphere_m`` is ``img_d_m / N_chief``
        within numerical roundoff (vertex tangent)."""
        p = self._off_axis_singlet()
        wfe = la.eval_image_plane_wfe(
            p, 587.56e-9, field=(1.0, 0.0), n_pupil=21,
            field_max_m=20e-3, image_plane='paraxial',
            sphere_tangent='vertex',
        )
        # Recover the chief direction cosine N_chief from the public
        # accessors.  We don't have direct API for the chief ray's
        # direction so we re-trace + extract.  The pin is that
        # r_sphere_m == img_d_m / N_chief (vertex tangent).
        from lumenairy.raytrace import (
            surfaces_from_prescription, trace, _make_bundle,
            first_order_data,
        )
        surfs = surfaces_from_prescription(p)
        # Match the eval_image_plane_wfe chief-aim geometry: aim the
        # chief at the EP centre with H = (1, 0).
        fod = first_order_data(surfs, 587.56e-9)
        obj_d = float(p['object_distance'])
        ep_z = float(getattr(fod, 'ep_z', 0.0))
        ep_r = float(getattr(fod, 'ep_radius', 6e-3))
        if not np.isfinite(ep_r) or ep_r <= 0:
            ep_r = 6e-3
        Hx, Hy = 1.0, 0.0
        field_max_m = 20e-3
        src_x = Hx * field_max_m
        src_y = Hy * field_max_m
        # Single chief ray (px=py=0) aimed at EP centre.
        aim_x = 0.0 - src_x
        aim_y = 0.0 - src_y
        aim_z = obj_d + ep_z
        norm = float(np.sqrt(aim_x**2 + aim_y**2 + aim_z**2))
        L = aim_x / norm
        M = aim_y / norm
        bundle = _make_bundle(
            x=np.array([src_x]), y=np.array([src_y]),
            L=np.array([L]), M=np.array([M]),
            wavelength=587.56e-9,
        )
        bundle.z = np.array([-obj_d])
        res = trace(bundle, surfs, 587.56e-9, output_filter='last')
        f = res.image_rays
        assert bool(f.alive[0]), "Chief ray didn't survive the trace"
        N_chief = float(f.N[0])
        assert abs(N_chief) > 0.05, (
            f"Test setup needs noticeably off-axis N_chief; got "
            f"{N_chief:.4f}")
        # The fix: r_sphere_m == img_d_m / N_chief for vertex.
        expected_R = wfe.img_d_m / N_chief
        rel_err = abs(wfe.r_sphere_m - expected_R) / max(abs(expected_R), 1e-12)
        assert rel_err < 1e-9, (
            f"r_sphere_m = {wfe.r_sphere_m:.9e} should equal "
            f"img_d_m / N_chief = {expected_R:.9e}; rel_err = "
            f"{rel_err:.2e} (B2-3 sphere-radius 1/N_chief fix).")

    def test_sphere_passes_through_chief_image_for_off_axis(self):
        """Geometric pin: the sphere centred on the chief image with
        the computed radius truly passes through the chief tangent
        point on the last surface, within < lambda/100 of the chief
        ray's path length.

        Pre-4.12.0 this distance was off by ``img_d_m * (1 - 1/N)``,
        which for a 4 deg field on a 100 mm singlet is ~250 um =
        ~430 wavelengths -- far above the lambda/100 threshold.
        """
        p = self._off_axis_singlet()
        wfe = la.eval_image_plane_wfe(
            p, 587.56e-9, field=(1.0, 0.0), n_pupil=21,
            field_max_m=20e-3, image_plane='paraxial',
            sphere_tangent='vertex',
        )
        # The sphere is centred at (cx, cy, cz) where (cx, cy) is the
        # chief image landing and cz = z_chief + img_d_m.  Its radius
        # is wfe.r_sphere_m.  For the sphere to pass through the
        # chief tangent point (the chief's intersection with the last
        # surface), the distance from sphere centre to that point
        # must equal r_sphere_m.
        #
        # We don't have the chief tangent location in the returned
        # dataclass.  Reconstruct via a single-ray trace at px=py=0
        # under the same aim geometry, then compute the residual.
        from lumenairy.raytrace import (
            surfaces_from_prescription, trace, _make_bundle,
            first_order_data,
        )
        surfs = surfaces_from_prescription(p)
        fod = first_order_data(surfs, 587.56e-9)
        obj_d = float(p['object_distance'])
        ep_z = float(getattr(fod, 'ep_z', 0.0))
        ep_r = float(getattr(fod, 'ep_radius', 6e-3))
        if not np.isfinite(ep_r) or ep_r <= 0:
            ep_r = 6e-3
        src_x, src_y = 20e-3, 0.0
        aim_x = -src_x; aim_y = -src_y
        aim_z = obj_d + ep_z
        norm = float(np.sqrt(aim_x**2 + aim_y**2 + aim_z**2))
        L = aim_x / norm
        M = aim_y / norm
        bundle = _make_bundle(
            x=np.array([src_x]), y=np.array([src_y]),
            L=np.array([L]), M=np.array([M]),
            wavelength=587.56e-9,
        )
        bundle.z = np.array([-obj_d])
        res = trace(bundle, surfs, 587.56e-9, output_filter='last')
        f = res.image_rays
        assert bool(f.alive[0])
        s2x = float(f.x[0]); s2y = float(f.y[0]); s2z = float(f.z[0])
        Ld = float(f.L[0]); Md = float(f.M[0]); Nd = float(f.N[0])
        # Chief image landing.
        t_adv = wfe.img_d_m / Nd
        cx = s2x + Ld * t_adv
        cy = s2y + Md * t_adv
        cz = s2z + wfe.img_d_m
        # Distance from sphere centre (cx, cy, cz) to the chief
        # tangent point on the last surface (s2x, s2y, s2z).
        dist = float(np.sqrt(
            (cx - s2x)**2 + (cy - s2y)**2 + (cz - s2z)**2))
        # Residual: how far the sphere misses the chief tangent.
        residual = abs(dist - wfe.r_sphere_m)
        # Pin: residual < lambda/100 (in metres).
        threshold = 587.56e-9 / 100.0
        assert residual < threshold, (
            f"Sphere doesn't pass through chief tangent: residual = "
            f"{residual*1e9:.3f} nm > lambda/100 = "
            f"{threshold*1e9:.3f} nm; r_sphere_m = "
            f"{wfe.r_sphere_m:.6e}, true dist = {dist:.6e}, "
            f"N_chief = {Nd:.4f} (B2-3 fix).")

    def test_on_axis_unchanged(self):
        """Sanity: on-axis chief (N_chief = 1) gives r_sphere_m ==
        img_d_m (vertex tangent), unchanged by the off-axis fix."""
        p = la.make_singlet(R1=51.5e-3, R2=float('inf'), d=4e-3,
                            glass='N-BK7', aperture=12e-3)
        p['object_distance'] = 500e-3
        wfe = la.eval_image_plane_wfe(
            p, 587.56e-9, field=(0.0, 0.0), n_pupil=15,
            image_plane='paraxial', sphere_tangent='vertex',
        )
        # On-axis: r_sphere_m == img_d_m.
        rel_err = (abs(wfe.r_sphere_m - wfe.img_d_m)
                   / max(abs(wfe.img_d_m), 1e-12))
        assert rel_err < 1e-9, (
            f"On-axis chief should have r_sphere_m == img_d_m; got "
            f"r_sphere_m={wfe.r_sphere_m:.6e}, "
            f"img_d_m={wfe.img_d_m:.6e}")


# ============================================================================
# B2-4 -- distortion_grid rejects unphysical (L^2 + M^2 >= 1) fields
# ============================================================================

class TestDistortionGridGuard:
    """``distortion_grid`` raises ``ValueError`` up front for field
    directions where ``sin(tx)^2 + sin(ty)^2 >= 1`` (N <= 0) instead
    of swallowing the trace failure in ``except: pass`` and producing
    an all-NaN grid (round-3 C-AB-2, still open).
    """

    def _singlet(self):
        return la.make_singlet(R1=50e-3, R2=float('inf'), d=4e-3,
                               glass='N-BK7', aperture=12e-3)

    def test_45_deg_corner_raises(self):
        """A grid spanning [-45, +45] deg with at least one even-grid
        point gives tx = ty = 45 deg in the corner, where
        sin^2 + sin^2 = 1 and N = 0.  Must raise."""
        p = self._singlet()
        with pytest.raises(ValueError, match=r'sin\(tx\)|N'):
            la.distortion_grid(p, 587.56e-9, max_field_deg=45.0,
                               n_grid=3)

    def test_50_deg_corners_raises(self):
        """50-deg corners have sin^2 + sin^2 ~ 1.17 > 1, so N would
        be imaginary; must raise."""
        p = self._singlet()
        with pytest.raises(ValueError, match=r'sin\(tx\)|N'):
            la.distortion_grid(p, 587.56e-9, max_field_deg=50.0,
                               n_grid=5)

    def test_30_deg_grid_runs(self):
        """A modest 30-deg grid has sin^2 + sin^2 ~ 0.5 < 1 at every
        corner; must run normally (no regression)."""
        p = self._singlet()
        result = la.distortion_grid(p, 587.56e-9, max_field_deg=30.0,
                                    n_grid=3)
        # Should return a DistortionGrid with the on-axis cell finite.
        center = 1  # for n_grid=3, centre = (1, 1)
        assert np.isfinite(result.actual_x[center, center])
        assert np.isfinite(result.actual_y[center, center])

    def test_error_lists_bad_pairs(self):
        """The error message must include the offending
        (theta_x_deg, theta_y_deg) pair so the user can see exactly
        which cell tripped the guard."""
        p = self._singlet()
        try:
            la.distortion_grid(p, 587.56e-9, max_field_deg=45.0,
                               n_grid=3)
            pytest.fail("distortion_grid did not raise for 45 deg corner")
        except ValueError as e:
            msg = str(e)
            assert '45' in msg or '-45' in msg, (
                f"Error message missing offending angle: {msg!r}")


# ============================================================================
# B2-5 -- apply_real_lens_traced explicit mirror guard
# ============================================================================

class TestApplyRealLensTracedMirrorGuard:
    """``apply_real_lens_traced`` raises ``ValueError`` with a
    mirror-specific message when ``prescription['surfaces']`` has
    ``is_mirror=True`` on any surface.

    Pre-4.12.0 this only fired if ``prescription['elements']`` had an
    ``element_type == 'mirror'`` entry; a hand-built prescription
    with ``surfaces[i]['is_mirror'] = True`` would slip past and the
    ray-traced OPL leg would treat the mirror as a refractor.
    """

    def _cassegrain_like_prescription(self):
        """Two-mirror Cassegrain-like prescription, with mirrors
        marked via ``is_mirror=True`` on each surface (the manual /
        hand-built case)."""
        return {
            'surfaces': [
                {'radius': -1.0, 'conic': -1.0,
                 'glass_before': 'air', 'glass_after': 'air',
                 'is_mirror': True},
                {'radius': -0.3, 'conic': -1.0,
                 'glass_before': 'air', 'glass_after': 'air',
                 'is_mirror': True},
            ],
            'thicknesses': [0.4],
            'aperture_diameter': 0.5,
            'object_distance': 1.0,
        }

    def test_raises_on_is_mirror_in_surfaces(self):
        """Hand-built prescription with ``is_mirror=True`` raises."""
        rx = self._cassegrain_like_prescription()
        E = np.ones((64, 64), dtype=complex)
        with pytest.raises(ValueError, match=r'mirror'):
            la.apply_real_lens_traced(
                E, prescription=rx, wavelength=587.56e-9, dx=1e-2)

    def test_raises_on_glass_after_MIRROR_marker(self):
        """The same guard fires for the alternate
        ``glass_after='MIRROR'`` marker convention used by the .zmx
        loader."""
        rx = {
            'surfaces': [
                {'radius': -1.0, 'conic': -1.0,
                 'glass_before': 'air', 'glass_after': 'MIRROR'},
                {'radius': float('inf'), 'conic': 0.0,
                 'glass_before': 'air', 'glass_after': 'air'},
            ],
            'thicknesses': [0.4],
            'aperture_diameter': 0.5,
            'object_distance': 1.0,
        }
        E = np.ones((64, 64), dtype=complex)
        with pytest.raises(ValueError, match=r'mirror'):
            la.apply_real_lens_traced(
                E, prescription=rx, wavelength=587.56e-9, dx=1e-2)

    def test_error_message_names_apply_real_lens_traced(self):
        """The error must name ``apply_real_lens_traced`` (not
        ``apply_real_lens``) and recommend the ``apply_mirror`` /
        per-segment pattern.

        Pre-4.12.0 the error came from the shared inner check and
        named ``apply_real_lens`` even when the user called
        ``apply_real_lens_traced``.
        """
        rx = self._cassegrain_like_prescription()
        E = np.ones((64, 64), dtype=complex)
        try:
            la.apply_real_lens_traced(
                E, prescription=rx, wavelength=587.56e-9, dx=1e-2)
            pytest.fail("apply_real_lens_traced did not raise on mirror")
        except ValueError as e:
            msg = str(e)
            assert 'apply_real_lens_traced' in msg, (
                f"Error message must name apply_real_lens_traced; "
                f"got: {msg!r}")
            assert 'apply_mirror' in msg, (
                f"Error message must recommend apply_mirror "
                f"per-segment pattern; got: {msg!r}")

    def test_pure_refractive_prescription_unaffected(self):
        """A bare singlet prescription with no mirrors must still
        pass through ``apply_real_lens_traced`` without raising the
        new mirror guard."""
        rx = la.make_singlet(R1=50e-3, R2=float('inf'), d=3e-3,
                             glass='N-BK7', aperture=5e-3)
        E = np.ones((64, 64), dtype=complex)
        # Should run without raising on the mirror guard (may raise
        # on other unrelated conditions like undersampling, which we
        # tolerate).
        try:
            la.apply_real_lens_traced(
                E, prescription=rx, wavelength=1.31e-6, dx=10e-6)
        except ValueError as e:
            msg = str(e)
            assert 'mirror' not in msg, (
                f"Bare singlet should not trip the mirror guard; "
                f"got: {msg!r}")


# ============================================================================
# B1-11 -- makedammann2d uses default_rng, doesn't mutate global state
# ============================================================================

class TestMakedammann2dNoGlobalRngMutation:
    """``makedammann2d(seed=N)`` must NOT mutate the user's global
    ``np.random`` state.  Pre-4.12.0 the function called
    ``np.random.seed(seed)`` which permanently shifted the process-
    wide RNG state -- a high-severity library anti-pattern.
    """

    def test_global_rng_state_unchanged_with_seed(self):
        """Pin: after a ``makedammann2d(seed=42, ...)`` call, the
        legacy global ``np.random`` state matches what it would have
        been if no random work had been done at all."""
        from lumenairy.elements.doe import makedammann2d
        np.random.seed(0)
        state_before = np.random.get_state()
        # Tiny run -- we only care about the RNG side-effects, not
        # the design quality.  itr=2 keeps it fast.
        _ = makedammann2d(
            periodx=20.0, periody=20.0,
            phaselevels=2, phasesteps=1,
            diforders=np.ones((2, 2)),
            itr=2, plot=False, seed=42,
        )
        state_after = np.random.get_state()
        # Compare every element of the state tuple (state[0] is the
        # algorithm name, state[1] is the int array of state, etc.).
        assert state_before[0] == state_after[0]
        assert np.array_equal(state_before[1], state_after[1]), (
            "makedammann2d(seed=42) perturbed the global np.random "
            "state -- it must use a local Generator (B1-11 fix).")
        assert state_before[2:] == state_after[2:]

    def test_global_rng_state_unchanged_without_seed(self):
        """Without a ``seed`` kwarg the function still must not seed
        the global RNG -- it should just consume randomness from its
        own local generator."""
        from lumenairy.elements.doe import makedammann2d
        np.random.seed(0)
        # Consume some random values to put the state at a non-trivial
        # point.
        _ = np.random.rand(5)
        state_before = np.random.get_state()
        _ = makedammann2d(
            periodx=20.0, periody=20.0,
            phaselevels=2, phasesteps=1,
            diforders=np.ones((2, 2)),
            itr=2, plot=False, seed=None,
        )
        state_after = np.random.get_state()
        # The global state must be unchanged: makedammann2d
        # internally uses default_rng(None) for unseeded runs and
        # MUST NOT consume from the legacy global pool.
        assert np.array_equal(state_before[1], state_after[1]), (
            "makedammann2d(seed=None) consumed the global np.random "
            "stream -- it must use a local Generator (B1-11 fix).")

    def test_same_seed_reproducible(self):
        """Two calls with the same ``seed=`` must produce identical
        output (the API guarantee preserved through the fix)."""
        from lumenairy.elements.doe import makedammann2d
        nf_a, ff_a, dx_a = makedammann2d(
            periodx=20.0, periody=20.0,
            phaselevels=2, phasesteps=1,
            diforders=np.ones((2, 2)),
            itr=3, plot=False, seed=123,
        )
        nf_b, ff_b, dx_b = makedammann2d(
            periodx=20.0, periody=20.0,
            phaselevels=2, phasesteps=1,
            diforders=np.ones((2, 2)),
            itr=3, plot=False, seed=123,
        )
        assert np.array_equal(nf_a, nf_b), (
            "Same seed must give identical near-field output.")
        assert np.array_equal(ff_a, ff_b), (
            "Same seed must give identical far-field output.")

    def test_different_seeds_differ(self):
        """Different ``seed=`` values must give different output
        (the seed is actually consumed by the local generator)."""
        from lumenairy.elements.doe import makedammann2d
        nf_a, _, _ = makedammann2d(
            periodx=20.0, periody=20.0,
            phaselevels=2, phasesteps=1,
            diforders=np.ones((2, 2)),
            itr=3, plot=False, seed=7,
        )
        nf_b, _, _ = makedammann2d(
            periodx=20.0, periody=20.0,
            phaselevels=2, phasesteps=1,
            diforders=np.ones((2, 2)),
            itr=3, plot=False, seed=11,
        )
        assert not np.array_equal(nf_a, nf_b), (
            "Different seeds must produce different near-field "
            "output -- seed kwarg not consumed.")


# ============================================================================
# B2-1 / B2-2 -- ghost.py docstring callouts
# ============================================================================

class TestGhostModuleDocstringCallouts:
    """``ghost.py`` carries explicit "upper bound" and "heuristic"
    docstring callouts (B2-1 / B2-2).  No physics changes, just
    documentation.
    """

    def test_module_docstring_calls_out_upper_bound(self):
        """Module docstring must explicitly call ``'intensity'``
        values an UPPER BOUND that omits transmission losses."""
        from lumenairy.analysis import ghost as _g
        doc = (_g.__doc__ or '').lower()
        assert 'upper bound' in doc, (
            "ghost.py module docstring must call intensity an "
            "'upper bound' (B2-1).")
        assert '1 - r' in doc.replace('_', '').replace(' ', '') or \
               '(1-r' in doc.replace(' ', ''), (
            "ghost.py module docstring must show the missing "
            "transmission factor (1 - R_k) (B2-1).")

    def test_module_docstring_calls_out_heuristic(self):
        """Module docstring must explicitly call
        ``focus_z_estimate`` a HEURISTIC, not a physical focal
        distance (B2-2)."""
        from lumenairy.analysis import ghost as _g
        doc = (_g.__doc__ or '').lower()
        assert 'heuristic' in doc, (
            "ghost.py module docstring must call focus_z_estimate a "
            "'heuristic' (B2-2).")

    def test_ghost_analysis_docstring_has_caveats(self):
        """``ghost_analysis`` function docstring must mention both
        the upper-bound and heuristic semantics for the relevant
        return-dict keys."""
        from lumenairy.analysis.ghost import ghost_analysis
        doc = inspect.getdoc(ghost_analysis) or ''
        lower = doc.lower()
        assert 'upper bound' in lower, (
            "ghost_analysis docstring must call 'intensity' an "
            "upper bound (B2-1).")
        assert 'heuristic' in lower, (
            "ghost_analysis docstring must call "
            "'focus_z_estimate' a heuristic (B2-2).")

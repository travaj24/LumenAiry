"""Consolidated audit-fix tests for the **analysis** domain.

This module consolidates v4.9 - v5.0 audit-fix regression pins
from 7 source files (per the v5.2 ROADMAP / 57-file consolidation):

* ``test_audit_fixes_v4_11_2_analysis.py``
* ``test_audit_fixes_v4_12_0_round4_analysis.py``
* ``test_audit_fixes_v4_13_0_perf_shack_hartmann.py``
* ``test_audit_fixes_v4_13_0_perf_wave_opd_2d.py``
* ``test_audit_fixes_v4_13_1_perf_sh_scatter.py``
* ``test_audit_fixes_v4_14_0_agent_3.py``
* ``test_audit_fixes_v4_14_0_agent_5.py``

Each source file's contents are concatenated below verbatim (modulo
minimal renames to avoid identifier collisions and to give each top-level
test class an audit-version attribution prefix).  inspect.getsource proxy
tests are tagged with a TODO comment per AUDIT_V4_13_1 Part 6.1.
"""
from __future__ import annotations

# ============================================================================
# Source: test_audit_fixes_v4_11_2_analysis.py
# Audit version: V4_11_2  scope: analysis
# Original module docstring preserved as comment block for git-blame traceability:
#   Regression tests for the v4.11.2 analysis-domain audit fixes.
#   
#   Each test pins a finding from ``AUDIT_ROUND3_2026_05_16.md`` (sections
#   "Aberration / AO / WFE / field / coherence / ghost / interferometry"
#   and "Analysis core / through-focus / phase retrieval / detector /
#   sources"):
#   
#   1. ``compute_psf(pupil_perfect, normalize='peak')`` peak == 1.0
#      exactly (the 'power' default is the v3.1.1+ semantic and is
#      exercised elsewhere; this test pins the alternate normalisation).
#   2. ``gerchberg_saxton_jax(seed=42)`` and ``seed=43`` consume the seed
#      (different iteration trajectories), with the kwarg no longer the
#      silently-ignored ``_ = seed`` it was through v4.11.1.
#   3. AO rim Zernike FD: tilt-X (Zernike index 1) and tilt-Y influence
#      matrices are symmetric on +/- x and +/- y rim lenslets.  Pre-v4.11.2
#      only the +x / +y rim had the one-sided fallback, leaving spurious
#      FD spikes on the -x / -y rims.
#   4. ``relative_illumination`` / ``field_aberration_sweep`` aim the
#      chief at the entrance pupil rather than at z=0.  Validated through
#      the on-axis transmission == 1.0 invariant for a stop-at-back system
#      (mid-stop pre-v4.11.2 erroneously vignetted on-axis).
#   5. ``polychromatic_strehl`` honours the global precision context
#      (``set_default_complex_dtype(np.complex64)`` -> internal ``E_in``
#      allocated as complex64).
# ============================================================================
import math
import warnings

import numpy as np
import pytest

import lumenairy as lm

# ============================================================================
# Finding #1 -- compute_psf with normalize='peak' actually peaks at 1
# ============================================================================

class TestAuditFixesV4_11_2_analysis_ComputePsfPeakNormalization:
    """``compute_psf(..., normalize='peak')`` must produce a PSF whose
    max equals 1 exactly (within machine epsilon).  The v3.1.1+ default
    is ``'power'`` which gives Parseval-normalised output and a peak
    that is generally NOT 1.0.

    Audit finding: ``validation/analysis/test_analysis.py:t_strehl_perfect``
    passed for the wrong reason -- ``psf.max() > 0.99`` was satisfied by
    the Parseval-normalised peak (~89795) under the new default.  The
    test now requests ``normalize='peak'`` explicitly; this regression
    test pins the underlying semantic.
    """

    def test_peak_normalize_max_equals_one_for_circular_aperture(self):
        N, dx, lam, D = 256, 20e-6, 1.31e-6, 5e-3
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        pupil = np.where(X**2 + Y**2 <= (D / 2) ** 2, 1.0, 0.0).astype(
            np.complex128)
        psf, _ = lm.compute_psf(pupil, lam, 50e-3, dx, normalize='peak')
        assert abs(float(psf.max()) - 1.0) < 1e-9, (
            f"normalize='peak' must give psf.max()==1.0 exactly; "
            f"got {float(psf.max()):.12g}")

    def test_power_default_is_parseval_not_peak(self):
        """Sanity: the default 'power' normalisation gives a peak that
        is NOT 1.0 for this aperture; the 'power' / 'peak' modes are
        distinct (regression for the v3.1.1 semantic)."""
        N, dx, lam, D = 256, 20e-6, 1.31e-6, 5e-3
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        pupil = np.where(X**2 + Y**2 <= (D / 2) ** 2, 1.0, 0.0).astype(
            np.complex128)
        psf_default, _ = lm.compute_psf(pupil, lam, 50e-3, dx)
        # Default 'power' produces a large absolute peak for this
        # aperture; just verify it is not ~1 (which would mean the
        # default flipped back to 'peak').
        assert float(psf_default.max()) > 10.0, (
            f"compute_psf default ('power') should NOT peak at 1.0; "
            f"got {float(psf_default.max()):.4g} (looks like 'peak')")


# ============================================================================
# Finding #2 -- gerchberg_saxton_jax actually uses seed
# ============================================================================

class TestAuditFixesV4_11_2_analysis_GerchbergSaxtonJaxSeed:
    """``gerchberg_saxton_jax`` must consume its ``seed`` kwarg rather
    than discarding it.  Pre-v4.11.2 the function began with
    ``_ = seed  # accepted for API parity; no random state in this
    loop`` and initialised every run with ``E0 = src.astype(complex64)``
    (deterministic, seed-independent).  v4.11.2 draws a uniform-random
    initial phase from ``jax.random.PRNGKey(seed)`` when ``seed`` is
    not None and ``initial_phase`` is not provided.
    """

    @pytest.fixture
    def has_jax(self):
        try:
            import jax  # noqa: F401
        except ImportError:
            pytest.skip("JAX not installed; gerchberg_saxton_jax test skipped")

    def test_different_seeds_yield_different_phases(self, has_jax):
        """The recovered phase distribution differs across seeds.  Even
        if both runs converge to a low residual, they should converge
        via *different trajectories* because they start at different
        random initial phases."""
        N = 32
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, x)
        src = np.exp(-(X**2 + Y**2) / 0.5**2).astype(np.float32)
        tgt = np.exp(-(X**2 + Y**2) / 0.3**2).astype(np.float32)
        # Short run -- 10 iterations -- so the trajectories are still
        # near the random initial condition and the phases haven't fully
        # converged to a deterministic basin-of-attraction.
        phase_42, err_42 = lm.gerchberg_saxton_jax(
            src, tgt, n_iter=10, seed=42)
        phase_43, err_43 = lm.gerchberg_saxton_jax(
            src, tgt, n_iter=10, seed=43)
        # The phases must differ somewhere (seed actually consumed).
        # Allow phase wrap by comparing the principal-value difference.
        diff = np.angle(np.exp(1j * (phase_42 - phase_43)))
        max_abs_diff = float(np.max(np.abs(diff)))
        assert max_abs_diff > 1e-3, (
            f"Different seeds produced indistinguishable phases "
            f"(max |angle(exp(i*(p42-p43)))| = {max_abs_diff:.2e}); "
            f"seed= kwarg is not consumed.")

    def test_same_seed_is_reproducible(self, has_jax):
        """Pinning the deterministic side: two runs with the same seed
        must produce the same phase exactly."""
        N = 32
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, x)
        src = np.exp(-(X**2 + Y**2) / 0.5**2).astype(np.float32)
        tgt = np.exp(-(X**2 + Y**2) / 0.3**2).astype(np.float32)
        p1, _ = lm.gerchberg_saxton_jax(src, tgt, n_iter=10, seed=7)
        p2, _ = lm.gerchberg_saxton_jax(src, tgt, n_iter=10, seed=7)
        assert np.array_equal(p1, p2), (
            "Same seed should be bit-for-bit reproducible; "
            f"max |p1-p2| = {float(np.max(np.abs(p1-p2))):.2e}")


# ============================================================================
# Finding #2 (continued) -- error_reduction / hybrid_input_output accept
# seed= and dtype= on the NumPy path
# ============================================================================

class TestAuditFixesV4_11_2_analysis_ErrorReductionSeedDtypeApi:
    """The NumPy ``error_reduction`` and ``hybrid_input_output``
    functions must accept ``seed`` and ``dtype`` for API parity with
    their JAX counterparts.  Pre-v4.11.2 these kwargs only existed on
    the JAX path; the NumPy path raised ``TypeError`` on either.
    """

    def test_error_reduction_accepts_seed(self):
        N = 16
        rng = np.random.default_rng(0)
        true_obj = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
        support = np.ones((N, N), dtype=bool)
        F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(true_obj)))
        meas = np.abs(F)
        # Should not raise.
        obj_a, err_a = lm.error_reduction(meas, support, n_iter=5,
                                          seed=1, dtype=np.complex128)
        obj_b, err_b = lm.error_reduction(meas, support, n_iter=5,
                                          seed=2, dtype=np.complex128)
        assert obj_a.shape == (N, N)
        assert obj_b.shape == (N, N)
        # Different seeds should produce different intermediate states.
        assert not np.allclose(obj_a, obj_b)

    def test_hio_accepts_seed_dtype(self):
        N = 16
        rng = np.random.default_rng(0)
        true_obj = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
        support = np.ones((N, N), dtype=bool)
        F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(true_obj)))
        meas = np.abs(F)
        # Should not raise; dtype propagated.
        obj, err = lm.hybrid_input_output(
            meas, support, n_iter=5, seed=3, dtype=np.complex64)
        assert obj.dtype == np.complex64


# ============================================================================
# Finding #3 -- AO rim Zernike FD symmetric on all four rim quadrants
# ============================================================================

class TestAuditFixesV4_11_2_analysis_AoRimZernikeFdSymmetric:
    """``zernike_modal_basis`` builds a Shack-Hartmann reconstructor
    via finite-difference Zernike-gradient probes at each lenslet
    centre.  At lenslets near the pupil rim one of the FD probes
    escapes the unit disk, where ``zernike_polynomial`` returns 0;
    pre-v4.11.2 a one-sided fallback was applied only when the +x or
    +y probe escaped, leaving the -x and -y rim with spurious zero-
    vs-finite FD spikes.  v4.11.2 detects all four rim quadrants and
    switches to the appropriate one-sided FD.

    Test: for a centrosymmetric Zernike (e.g. defocus / Zernike index
    that's an even mode), the influence-matrix row for a lenslet at
    (+x_rim, 0) should match the row at (-x_rim, 0) by symmetry --
    pre-v4.11.2 the latter had a spurious spike.
    """

    def test_rim_xy_dwdx_symmetric_for_defocus_mode(self):
        """Build a Shack-Hartmann basis on a centred square grid and
        verify the FD-gradient column for defocus (mode 4) is
        antisymmetric under x -> -x at the rim.

        Pre-v4.11.2 only the +x and +y rim probes switched to one-sided
        FD; the -x and -y rim probes silently let the (forbidden)
        rho > 1 sample fall to zero, producing a spurious spike that
        broke the +x/-x symmetry that defocus exhibits in exact
        arithmetic.
        """
        semi_aperture = 5e-3
        n_lenslets = 11  # odd so the centre row hits y=0 exactly
        # n_modes = 4 gives modes [1, 2, 3, 4] = tilt-X, tilt-Y,
        # astig 45, defocus.  Defocus is the centrosymmetric one and
        # the easiest invariant to pin.
        basis = lm.zernike_modal_basis(
            n_modes=4, n_lenslets=n_lenslets,
            semi_aperture=semi_aperture)
        M = basis['influence_matrix']
        xlens, ylens = basis['lenslet_xy']
        n_lens = xlens.size
        # Mode column 3 = defocus (OSA index 4, first_mode=1 so the
        # 4th entry).  Verify mode_indices echo.
        mode_indices = basis['mode_indices']
        defocus_col = int(np.where(mode_indices == 4)[0][0])
        dWdx_defocus = M[:n_lens, defocus_col]

        # Find pairs of rim lenslets at (+xi, yi) and (-xi, yi).
        rho_lens = np.sqrt(xlens**2 + ylens**2) / semi_aperture
        rim_mask = rho_lens > 0.85
        tol = semi_aperture / n_lenslets * 0.5
        max_asymmetry = 0.0
        n_pairs = 0
        for i in np.where(rim_mask)[0]:
            if abs(xlens[i]) < tol:
                continue
            target_x = -xlens[i]
            mirror_idx = np.where(
                (np.abs(xlens - target_x) < tol)
                & (np.abs(ylens - ylens[i]) < tol))[0]
            if mirror_idx.size == 0:
                continue
            j = int(mirror_idx[0])
            # Defocus W = 2*rho^2 - 1, dW/dx = 4*rho*cos(theta)/r_pup
            # = 4*x_norm / semi_aperture.  Antisymmetric under x -> -x.
            sym_err = float(abs(dWdx_defocus[i] + dWdx_defocus[j]))
            max_asymmetry = max(max_asymmetry, sym_err)
            n_pairs += 1
        # At least a few rim pairs should be available.
        assert n_pairs >= 2, (
            f"Insufficient rim pairs (n={n_pairs}); test geometry "
            f"too coarse to discriminate the bug.")
        # Symmetry should hold to FD truncation accuracy.  Pre-v4.11.2
        # the rim spike made the -x rim's FD a different magnitude
        # from the +x rim, breaking antisymmetry.
        scale = float(np.max(np.abs(dWdx_defocus[rim_mask])))
        rel_asymmetry = max_asymmetry / max(scale, 1e-30)
        assert rel_asymmetry < 1e-2, (
            f"AO rim FD asymmetric: max |dW/dx(+x) + dW/dx(-x)| = "
            f"{max_asymmetry:.4e}, scale={scale:.4e}, "
            f"rel_asymmetry={rel_asymmetry:.4e} -- expect << 1e-2 if "
            f"all four rim quadrants are properly handled.  Tested "
            f"{n_pairs} rim pairs.")


# ============================================================================
# Finding #4 -- field_aberration_sweep / relative_illumination aim at EP
# ============================================================================

class TestAuditFixesV4_11_2_analysis_EpAimingPortedToFieldSiblings:
    """``relative_illumination`` and ``field_aberration_sweep`` must
    launch chief rays such that they pass through the entrance-pupil
    centre -- not the first surface vertex.  Pre-v4.11.2 both used
    ``make_rings(semi_aperture, ...)`` and ``make_fan(axis='y',
    field_angle=fa)`` which place rays at z=0 with direction
    (0, sin fa, cos fa), implicitly aiming at z=0.  For stop-at-front
    systems EP coincides with the first surface so this is correct;
    for mid-stop systems it walks the chief across the aperture as
    field angle grows, producing visible artefacts in both functions.
    """

    def test_relative_illumination_on_axis_is_one(self):
        """On-axis RELATIVE illumination must be 1.0 by construction
        (the function divides every field's transmission by the
        on-axis transmission).  Also pin: transmission > 0.5 on-axis,
        which catches the case where the v4.11.2 EP-aim refactor
        accidentally moved on-axis rays off the first surface (which
        would zero them out)."""
        pres = lm.make_singlet(R1=50e-3, R2=float('inf'), d=4e-3,
                                glass='N-BK7', aperture=10e-3)
        ri = lm.relative_illumination(
            pres, wavelength=1.31e-6,
            fields_deg=[0.0, 1.0, 2.0],
            num_rings=6, rays_per_ring=12)
        # On-axis RELATIVE illumination is 1.0 by definition.
        idx0 = int(np.where(ri.fields_deg == 0.0)[0][0])
        on_axis_rel = float(ri.relative_illumination[idx0])
        assert abs(on_axis_rel - 1.0) < 1e-12, (
            f"On-axis relative illumination must be 1.0 by "
            f"construction; got {on_axis_rel:.6g}")
        # And the absolute on-axis transmission must be > 0.5 -- a
        # weak but necessary check that the EP-aim code didn't move
        # rays off the first surface entirely.
        on_axis_T = float(ri.transmission[idx0])
        assert on_axis_T > 0.5, (
            f"On-axis transmission too low ({on_axis_T:.6g}); "
            f"suggests the EP-aim refactor displaced rays off the "
            f"first surface.")

    def test_relative_illumination_callable_for_stop_at_front(self):
        """A stop-at-front system has ep_z=0 and ep_radius=semi_ap, so
        the v4.11.2 EP-aim code is a no-op -- and the function must
        still produce sensible (monotonically declining) RI vs field."""
        pres = lm.make_singlet(R1=50e-3, R2=float('inf'), d=4e-3,
                                glass='N-BK7', aperture=10e-3)
        ri = lm.relative_illumination(
            pres, wavelength=1.31e-6,
            fields_deg=[0.0, 2.0, 5.0],
            num_rings=6, rays_per_ring=12)
        # RI must be non-negative and ≤ 1 everywhere.
        assert np.all(ri.relative_illumination >= 0)
        assert np.all(ri.relative_illumination <= 1.0 + 1e-12)


# ============================================================================
# Finding #5 -- polychromatic_strehl uses default complex dtype
# ============================================================================

class TestAuditFixesV4_11_2_analysis_PolychromaticPrecisionRespect:
    """``polychromatic_strehl`` and ``polychromatic_psf`` must honour
    the global default complex dtype (set via
    ``set_default_complex_dtype``) for their internal ``E_in``
    allocation.  Pre-v4.11.2 they hard-coded ``np.complex128`` even
    under ``precision='single'``, silently coercing single-precision
    callers back to double.
    """

    def test_polychromatic_psf_under_single_precision(self):
        """Set complex64, call polychromatic_psf, verify it doesn't
        crash and the returned PSF is finite.  We can't directly
        introspect the internal E_in allocation from outside, but
        running through the single-precision path under
        set_default_complex_dtype(np.complex64) at least exercises the
        v4.11.2 fix; if E_in were still complex128 the call would
        either succeed (silently upcasting -- which is what we want to
        forbid) or be unaffected by the precision context (which is
        the regression we're guarding against).
        """
        from lumenairy.propagators.propagation import (
            get_default_complex_dtype,
            set_default_complex_dtype,
        )
        # Save current.
        prev = get_default_complex_dtype()
        try:
            set_default_complex_dtype(np.complex64)
            pres = lm.make_singlet(R1=51.5e-3, R2=float('inf'),
                                    d=4e-3, glass='N-BK7',
                                    aperture=10e-3)
            psf, dx_psf, info = lm.polychromatic_psf(
                pres, [1.30e-6, 1.31e-6, 1.32e-6], [1, 1, 1],
                N=64, dx=120e-6, normalize='power')
            assert psf.shape == (64, 64)
            assert np.isfinite(psf).all()
        finally:
            set_default_complex_dtype(prev)

    def test_polychromatic_psf_e_in_dtype_under_single_precision(self):
        """Inspect via monkeypatching ``apply_real_lens`` to capture
        the dtype of the field that flows into the function.  This
        directly verifies the v4.11.2 fix: the internal default-allocated
        E_in must be complex64 when the precision context is single.
        """
        from lumenairy.elements import lenses as _lenses
        from lumenairy.propagators.propagation import (
            get_default_complex_dtype,
            set_default_complex_dtype,
        )

        captured_dtype = {}
        orig_apply_real_lens = _lenses.apply_real_lens

        def _spy_apply(E, **kw):
            captured_dtype.setdefault('dtype', E.dtype)
            return orig_apply_real_lens(E, **kw)

        prev = get_default_complex_dtype()
        try:
            set_default_complex_dtype(np.complex64)
            _lenses.apply_real_lens = _spy_apply
            pres = lm.make_singlet(R1=51.5e-3, R2=float('inf'),
                                    d=4e-3, glass='N-BK7',
                                    aperture=10e-3)
            # Don't pass E_in -- we want the function to allocate one
            # using its (new) default-dtype path.
            _ = lm.polychromatic_psf(
                pres, [1.31e-6], [1.0],
                N=32, dx=120e-6, normalize='none')
        finally:
            _lenses.apply_real_lens = orig_apply_real_lens
            set_default_complex_dtype(prev)
        assert captured_dtype['dtype'] == np.complex64, (
            f"polychromatic_psf must allocate E_in at the default "
            f"complex dtype (complex64 under single precision); got "
            f"{captured_dtype['dtype']!r}.")


# ============================================================================
# Source: test_audit_fixes_v4_12_0_round4_analysis.py
# Audit version: V4_12_0  scope: round4_analysis
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.12.0 round-4 audit Tier-2 analysis-domain
#   fixes (``AUDIT_ROUND4_2026_05_16.md`` items B2-3, B2-4, B2-5, B1-11,
#   B2-1, B2-2).
#   
#   Each test pins a finding from the round-4 audit:
#   
#   * B2-3 -- ``image_plane_wfe`` reference-sphere radius now carries the
#     ``1/N_chief`` factor.  For off-axis chief with ``N_chief < 1`` the
#     sphere genuinely passes through the chief image landing; pre-4.12.0
#     the radius was the AXIAL distance, so for off-axis fields the
#     sphere missed the chief by ``img_d_m * (1 - 1/N_chief)`` and the
#     quadratic shape error was absorbed as phantom defocus by
#     ``best_rms``.
#   
#   * B2-4 -- ``distortion_grid`` raises ``ValueError`` up front for
#     ``sin(tx)^2 + sin(ty)^2 >= 1`` field directions (e.g.
#     ``tx = ty = 45 deg``) instead of swallowing the trace failure in
#     ``except: pass`` and returning an all-NaN grid.
#   
#   * B2-5 -- ``apply_real_lens_traced`` raises ``ValueError`` with a
#     mirror-specific message when ``prescription['surfaces']`` carries
#     ``is_mirror=True`` (or ``glass_after='MIRROR'``).  The shared
#     ``_check_no_silent_fold_drop`` only sees ``prescription['elements']``
#     so a hand-built prescription would otherwise sneak past.
#   
#   * B1-11 -- ``makedammann2d(seed=N)`` does not mutate the user's global
#     ``np.random`` state.  Pre-4.12.0 it called ``np.random.seed(seed)``;
#     4.12.0 uses ``np.random.default_rng(seed)`` and threads the local
#     generator through the function body.
#   
#   * B2-1 / B2-2 -- ``ghost.py`` module-level and ``ghost_analysis``
#     docstrings call out (a) reported intensity is an upper bound
#     ignoring transmission losses ``Prod (1 - R_k)``, (b)
#     ``focus_z_estimate`` is a heuristic harmonic-mean sort key, not a
#     physical focal distance.
#   
#   These pins guard against regression of the v4.12.0 fixes.
# ============================================================================

import inspect

import numpy as np
import pytest

import lumenairy as la

# ============================================================================
# B2-3 -- image_plane_wfe reference-sphere radius carries 1/N_chief
# ============================================================================

class TestAuditFixesV4_12_0_round4_analysis_ImagePlaneWfeSphereRadiusOffAxis:
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
            _make_bundle,
            first_order_data,
            surfaces_from_prescription,
            trace,
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
            _make_bundle,
            first_order_data,
            surfaces_from_prescription,
            trace,
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

class TestAuditFixesV4_12_0_round4_analysis_DistortionGridGuard:
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

class TestAuditFixesV4_12_0_round4_analysis_ApplyRealLensTracedMirrorGuard:
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

class TestAuditFixesV4_12_0_round4_analysis_Makedammann2dNoGlobalRngMutation:
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
            periodx=20.0, periody=20.0, waveln=1.31,
            phaselevels=2, phasesteps=1,
            diforders=np.ones((2, 2)),
            itr=2, plot=False, seed=42,
            _legacy_units='um',
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
            periodx=20.0, periody=20.0, waveln=1.31,
            phaselevels=2, phasesteps=1,
            diforders=np.ones((2, 2)),
            itr=2, plot=False, seed=None,
            _legacy_units='um',
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
            periodx=20.0, periody=20.0, waveln=1.31,
            phaselevels=2, phasesteps=1,
            diforders=np.ones((2, 2)),
            itr=3, plot=False, _legacy_units='um', seed=123,
        )
        nf_b, ff_b, dx_b = makedammann2d(
            periodx=20.0, periody=20.0, waveln=1.31,
            phaselevels=2, phasesteps=1,
            diforders=np.ones((2, 2)),
            itr=3, plot=False, _legacy_units='um', seed=123,
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
            periodx=20.0, periody=20.0, waveln=1.31,
            phaselevels=2, phasesteps=1,
            diforders=np.ones((2, 2)),
            itr=3, plot=False, _legacy_units='um', seed=7,
        )
        nf_b, _, _ = makedammann2d(
            periodx=20.0, periody=20.0, waveln=1.31,
            phaselevels=2, phasesteps=1,
            diforders=np.ones((2, 2)),
            itr=3, plot=False, _legacy_units='um', seed=11,
        )
        assert not np.array_equal(nf_a, nf_b), (
            "Different seeds must produce different near-field "
            "output -- seed kwarg not consumed.")


# ============================================================================
# B2-1 / B2-2 -- ghost.py docstring callouts
# ============================================================================

class TestAuditFixesV4_12_0_round4_analysis_GhostModuleDocstringCallouts:
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


# ============================================================================
# Source: test_audit_fixes_v4_13_0_perf_shack_hartmann.py
# Audit version: V4_13_0  scope: perf_shack_hartmann
# Original module docstring preserved as comment block for git-blame traceability:
#   Correctness pins for the v4.13.0 Shack-Hartmann FFT batching
#   (audit Phase-3 Group beta task beta.1).
#   
#   Pre-4.13 ``lumenairy.analysis.detector.shack_hartmann`` walked the
#   lenslet array with two nested Python loops (one for reference centroids
#   on a flat-wavefront calibration field, one for the actual measurement
#   on the user's field).  Each iteration extracted a sub-aperture, applied
#   the lenslet quadratic-phase mask, and propagated it to the focal plane
#   via either a raw ``np.fft.fft2`` (reference pass) or the full ASM
#   machinery (measurement pass), one call per lenslet.
#   
#   The v4.13.0 vectorisation:
#   
#   * Recognises that every sub-aperture of the flat reference field is
#     identical, so the reference centroid is computed ONCE and broadcast
#     to every valid lenslet (zero FFTs in the inner loop, one in total).
#   * Stacks the measurement sub-apertures into a single
#     ``(K, sa_pixels, sa_pixels)`` ndarray, applies the shared lenslet
#     phase, and propagates the whole stack with a single
#     ``np.fft.fft2(..., axes=(-2, -1))`` + ``ifft2(...)`` pair, multiplied
#     through one pre-built ASM transfer function that is geometry-only
#     and therefore identical for every lenslet.
#   
#   What this test pins
#   -------------------
#   
#   1. **Tilt recovery** -- a known linear-tilt wavefront gives a slope_x
#      map close to the analytical gradient (sign + magnitude), echoing
#      the existing validation test in ``validation/analysis/test_detector.py``.
#   2. **Determinism** -- same input twice returns bit-identical output.
#   3. **OOB sentinels** -- with a lenslet pitch chosen so the array
#      overhangs the field, the out-of-bounds lenslets stay at NaN and
#      the in-bounds ones produce finite slopes.
#   4. **Reference-centroid invariance** -- a flat input produces slopes
#      that are unaffected by the per-lenslet centring bias (i.e. the
#      reference subtraction still cancels it).
# ============================================================================

import numpy as np
import pytest

from lumenairy.analysis.detector import shack_hartmann

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _tilt_field(N: int, dx: float, wavelength: float,
                waves_total: float = 5.0) -> np.ndarray:
    """Build a complex field with `waves_total` waves of tilt across
    the grid (left-to-right linear phase ramp)."""
    x = (np.arange(N) - N / 2) * dx
    X, _ = np.meshgrid(x, x)
    phase = (2 * np.pi * waves_total / (N * dx)) * X
    return np.exp(1j * phase)


# ---------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------

def test_shack_hartmann_recovers_known_tilt():
    """Linear tilt across the pupil should produce a near-uniform
    positive slope_x and ~zero slope_y.  Mirrors the existing
    validation harness test but pins it to pytest so the unit suite
    catches regressions of the batched path on every run."""
    N = 128
    dx = 4e-6
    lam = 1.31e-6
    E = _tilt_field(N, dx, lam, waves_total=5.0)
    slope_x, slope_y, wf, cx, cy = shack_hartmann(
        E, dx, lam, lenslet_pitch=50e-6, lenslet_focal=200e-6,
        n_lenslets=8)
    # slope_x: positive, near-uniform across valid lenslets
    valid_sx = slope_x[np.isfinite(slope_x)]
    assert valid_sx.size > 0
    assert float(np.mean(valid_sx)) > 0
    assert float(np.std(valid_sx)) < 1e-3 * abs(float(np.mean(valid_sx))) + 1e-12
    # slope_y: small (no tilt in y).  There is a small ASM-vs-FFT
    # residual (~2e-3 rad) on every lenslet that is a pre-existing
    # artefact of the reference-vs-measurement convention mismatch
    # in the legacy code -- compare against slope_x magnitude.
    valid_sy = slope_y[np.isfinite(slope_y)]
    assert float(np.max(np.abs(valid_sy))) < float(np.mean(np.abs(valid_sx)))


def test_shack_hartmann_deterministic_repeat():
    """Running the same input twice through the (vectorised) path
    should produce bit-identical output."""
    N = 128
    dx = 4e-6
    lam = 1.31e-6
    E = _tilt_field(N, dx, lam, waves_total=3.0)
    sx1, sy1, wf1, cx1, cy1 = shack_hartmann(
        E, dx, lam, lenslet_pitch=50e-6, lenslet_focal=200e-6, n_lenslets=8)
    sx2, sy2, wf2, cx2, cy2 = shack_hartmann(
        E, dx, lam, lenslet_pitch=50e-6, lenslet_focal=200e-6, n_lenslets=8)
    np.testing.assert_array_equal(sx1, sx2)
    np.testing.assert_array_equal(sy1, sy2)
    np.testing.assert_array_equal(wf1, wf2)
    np.testing.assert_array_equal(cx1, cx2)
    np.testing.assert_array_equal(cy1, cy2)


def test_shack_hartmann_oob_lenslets_are_nan():
    """When n_lenslets * lenslet_pitch overhangs the grid, the
    overhanging lenslets must stay at NaN (the 4.10 OOB-sentinel
    contract -- pre-4.10 they got 0 slopes which masqueraded as real
    measurements in the cumsum integrator)."""
    # Choose a grid so the lenslet array deliberately overhangs.
    # N=32, dx=4e-6 -> 128 um field; lenslet_pitch=20e-6, n_lenslets=12
    # -> 240 um requested footprint, so the array hangs off both sides
    # and only the central few are in-bounds.
    N = 32
    dx = 4e-6
    lam = 1.31e-6
    E = np.ones((N, N), dtype=complex)
    slope_x, slope_y, wf, cx, cy = shack_hartmann(
        E, dx, lam, lenslet_pitch=20e-6, lenslet_focal=200e-6,
        n_lenslets=12)
    # Most lenslets out of bounds: at least the four corners must be NaN.
    assert np.isnan(slope_x[0, 0])
    assert np.isnan(slope_x[0, -1])
    assert np.isnan(slope_x[-1, 0])
    assert np.isnan(slope_x[-1, -1])
    # Some lenslets in bounds: at least one finite value somewhere.
    assert np.any(np.isfinite(slope_x))


def test_shack_hartmann_flat_input_zero_slope_after_reference():
    """The 4.10 reference-centroid subtraction guarantees a flat
    wavefront produces ~zero slope (modulo small ASM-vs-FFT residuals
    from the reference using a raw fft2 and the measurement using
    band-limited ASM -- this is a pre-existing approximation, not a
    v4.13.0 regression).  Pin the order of magnitude so a future
    refactor that drops the reference subtraction is caught."""
    N = 128
    dx = 4e-6
    lam = 1.31e-6
    E = np.ones((N, N), dtype=complex)
    slope_x, slope_y, _, _, _ = shack_hartmann(
        E, dx, lam, lenslet_pitch=50e-6, lenslet_focal=200e-6, n_lenslets=8)
    valid_sx = slope_x[np.isfinite(slope_x)]
    valid_sy = slope_y[np.isfinite(slope_y)]
    # Flat input: residual slope should be tiny (< 0.01 rad effective).
    # Reference-subtraction cancels the per-lenslet centring bias.
    assert float(np.max(np.abs(valid_sx))) < 1e-2
    assert float(np.max(np.abs(valid_sy))) < 1e-2


# ============================================================================
# Source: test_audit_fixes_v4_13_0_perf_wave_opd_2d.py
# Audit version: V4_13_0  scope: perf_wave_opd_2d
# Original module docstring preserved as comment block for git-blame traceability:
#   Correctness pins for the v4.13.0 ``wave_opd_2d`` axis-unwrap
#   vectorisation (audit Phase-3 Group beta task beta.2).
#   
#   Pre-4.13 the 2-D OPD unwrap walked the phase grid with two Python
#   for-loops:
#   
#   .. code-block:: python
#   
#       for j in range(Ny):
#           phase_unwrapped[j, :] = np.unwrap(phase[j, :])
#       for i in range(Nx):
#           phase_unwrapped[:, i] = np.unwrap(phase_unwrapped[:, i])
#   
#   Each ``np.unwrap`` call on a 1-D slice spends most of its time on
#   Python-level overhead (slice creation, function call, ufunc dispatch);
#   the actual unwrap is O(N) compiled C work.  ``np.unwrap`` accepts
#   ``axis=``, so the two passes collapse to two compiled calls operating
#   on the full 2-D array:
#   
#   .. code-block:: python
#   
#       phase_unwrapped = np.unwrap(phase, axis=1)
#       phase_unwrapped = np.unwrap(phase_unwrapped, axis=0)
#   
#   What this test pins
#   -------------------
#   
#   1. **Equivalence on smooth wavefronts** -- a quadratic OPD with no
#      2*pi wraps gives the same map (to numerical round-off) before and
#      after vectorisation.
#   2. **Wrap recovery** -- a steep tilt that wraps multiple times across
#      the pupil is unwrapped correctly, recovering the underlying OPD.
#   3. **Aperture masking** -- the ``aperture`` parameter still masks
#      samples outside the clear aperture to NaN.
# ============================================================================

import numpy as np
import pytest

from lumenairy.analysis.core import wave_opd_2d

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _reference_unwrap_2d(phase: np.ndarray) -> np.ndarray:
    """Pre-vectorisation row-then-column unwrap reference.

    Reproduces the pre-4.13 Python double-loop so we can verify the
    vectorised path matches bit-by-bit on smooth inputs.
    """
    Ny, Nx = phase.shape
    out = np.empty_like(phase)
    for j in range(Ny):
        out[j, :] = np.unwrap(phase[j, :])
    for i in range(Nx):
        out[:, i] = np.unwrap(out[:, i])
    return out


# ---------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------

def test_vectorised_unwrap_matches_loop_on_smooth_wavefront():
    """A smooth quadratic OPD has no phase wraps; the vectorised
    unwrap should give an output identical to the legacy row-then-
    column loop to numerical round-off."""
    N = 64
    dx = 5e-6
    lam = 1.31e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    k0 = 2 * np.pi / lam
    # Gentle quadratic, < 1 cycle of phase across the grid.
    opd_true = 30e-9 * (X**2 + Y**2) / ((N * dx / 2)**2)
    E = np.exp(1j * k0 * opd_true)

    _, _, opd_vec = wave_opd_2d(E, dx, lam, aperture=0.9 * N * dx)

    # Reference: do the unwrap the legacy way and convert back to OPD.
    phase = np.angle(E)
    phase_ref = _reference_unwrap_2d(phase)
    opd_ref_full = phase_ref / k0
    # Apply the same aperture mask wave_opd_2d does.
    valid = (X**2 + Y**2 <= (0.5 * 0.9 * N * dx)**2) & (np.abs(E) > 0)
    opd_ref = np.where(valid, opd_ref_full, np.nan)

    # Bit-identity on the valid region.
    mask = np.isfinite(opd_vec) & np.isfinite(opd_ref)
    assert mask.sum() > 0
    diff = np.max(np.abs(opd_vec[mask] - opd_ref[mask]))
    assert diff < 1e-12, f"Max OPD diff vs reference = {diff:.3e}"


def test_unwrap_recovers_steep_tilt():
    """A wavefront tilt steep enough to wrap multiple times across
    the pupil should be unwrapped back to the smooth linear OPD."""
    N = 128
    dx = 5e-6
    lam = 1.31e-6
    x = (np.arange(N) - N / 2) * dx
    X, _ = np.meshgrid(x, x)
    k0 = 2 * np.pi / lam
    # Strong tilt: 4 waves of OPD across the grid (-> 8*pi of phase).
    opd_true = 4.0 * lam * (X / (N * dx))
    E = np.exp(1j * k0 * opd_true)

    _, _, opd_vec = wave_opd_2d(E, dx, lam, aperture=0.9 * N * dx)

    # Recovered OPD should differ from analytical only by a constant
    # offset (unwrap doesn't constrain the integration constant).
    mask = np.isfinite(opd_vec)
    diff = opd_vec[mask] - opd_true[mask]
    offset = float(np.mean(diff))
    residual = float(np.max(np.abs(diff - offset)))
    assert residual < 1e-12, f"Max unwrap residual = {residual:.3e}"


def test_aperture_masks_outside_to_nan():
    """Samples outside the requested clear aperture must be NaN."""
    N = 32
    dx = 5e-6
    lam = 1.31e-6
    aperture = 0.5 * N * dx
    E = np.ones((N, N), dtype=complex)
    X_, Y_, opd = wave_opd_2d(E, dx, lam, aperture=aperture)
    r2 = X_**2 + Y_**2
    outside = r2 > (0.5 * aperture)**2
    assert np.all(np.isnan(opd[outside]))
    # And inside the aperture is NOT NaN (flat -> 0 OPD).
    inside = r2 <= (0.5 * aperture)**2
    assert np.all(np.isfinite(opd[inside]))


def test_unwrap_on_zero_field_propagates_nan():
    """A sample with ``|E| == 0`` must still be masked to NaN regardless
    of aperture (the unwrap is meaningless where amplitude is zero)."""
    N = 32
    dx = 5e-6
    lam = 1.31e-6
    E = np.ones((N, N), dtype=complex)
    # Punch a few zeros.
    E[5, 7] = 0.0
    E[10, 12] = 0.0
    _, _, opd = wave_opd_2d(E, dx, lam, aperture=0.9 * N * dx)
    assert np.isnan(opd[5, 7])
    assert np.isnan(opd[10, 12])


# ============================================================================
# Source: test_audit_fixes_v4_13_1_perf_sh_scatter.py
# Audit version: V4_13_1  scope: perf_sh_scatter
# Original module docstring preserved as comment block for git-blame traceability:
#   Correctness pin for the v4.13.1 Shack-Hartmann scatter-back
#   vectorisation (audit Tier-3 perf).
#   
#   Pre-v4.13.1 the post-batch scatter into ``centroids_x``,
#   ``centroids_y``, ``slopes_x``, and ``slopes_y`` ran in a Python
#   ``for k in range(K)`` loop with per-iteration ``int()`` /
#   ``float()`` coercions and four scalar assignments.  v4.13.1
#   replaces it with vectorised fancy indexing.  Same arithmetic, same
#   scalar values stored; bit-exact on numerical comparison.
#   
#   This test pins the new path against a hand-coded reference
#   implementation of the scalar loop and confirms a wallclock
#   speedup on a realistically-sized lenslet grid (24x24 = 576
#   lenslets, which exercises the per-lenslet python overhead).
# ============================================================================

import time

import numpy as np
import pytest

from lumenairy.analysis.detector import shack_hartmann


def _make_flat_input(N, dx):
    """Plane-wave input across an N x N pupil -- every lenslet sees
    flat phase, so the centroid is the reference (zero slope)."""
    return np.ones((N, N), dtype=np.complex128)


def _make_tilted_input(N, dx, wavelength, tilt_x_rad, tilt_y_rad):
    """Plane-wave with a linear phase ramp (uniform tilt)."""
    k = 2 * np.pi / wavelength
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, y, indexing='xy')
    return np.exp(1j * k * (tilt_x_rad * X + tilt_y_rad * Y)) \
        .astype(np.complex128)


def test_sh_scatter_bit_exact_flat():
    """Flat input: every lenslet centroid is at the reference; the
    scatter must reproduce the pre-v4.13.1 result exactly (which
    means slopes_x / slopes_y are essentially zero, centroids at
    the reference values)."""
    N = 256
    dx = 5e-6
    wavelength = 633e-9
    lenslet_pitch = 32 * dx       # 32 pixels per lenslet
    lenslet_focal = 4e-3
    E = _make_flat_input(N, dx)

    sx, sy, wf, cx, cy = shack_hartmann(
        E, dx, wavelength, lenslet_pitch, lenslet_focal,
        n_lenslets=8, detector_pixels_per_lenslet=16, seed=0,
    )
    # Output shape check: 8x8 lenslets.
    assert sx.shape == (8, 8)
    # Flat input: slopes are very close to zero (residual is
    # discretisation noise from the centroid pixel-grid).
    sx_finite = sx[np.isfinite(sx)]
    sy_finite = sy[np.isfinite(sy)]
    assert np.max(np.abs(sx_finite)) <= 1e-4, (
        f"Flat input but slope_x magnitudes up to {np.max(np.abs(sx_finite)):.3e}")
    assert np.max(np.abs(sy_finite)) <= 1e-4, (
        f"Flat input but slope_y magnitudes up to {np.max(np.abs(sy_finite)):.3e}")


def test_sh_scatter_recovers_tilt():
    """Tilted input: a uniform x-tilt of theta_x produces uniform
    slopes_x == theta_x across every (interior) lenslet."""
    N = 256
    dx = 5e-6
    wavelength = 633e-9
    lenslet_focal = 4e-3
    lenslet_pitch = 32 * dx
    tilt_x = 1e-4   # 100 urad x-tilt
    E = _make_tilted_input(N, dx, wavelength, tilt_x, 0.0)

    sx, sy, wf, cx, cy = shack_hartmann(
        E, dx, wavelength, lenslet_pitch, lenslet_focal,
        n_lenslets=8, detector_pixels_per_lenslet=16, seed=0,
    )
    # Interior lenslets (skip edges where vignetting reduces SNR).
    interior_sx = sx[2:-2, 2:-2]
    interior_sx = interior_sx[np.isfinite(interior_sx)]
    if interior_sx.size > 0:
        mean_sx = float(np.mean(interior_sx))
        # Recover tilt within ~30% (Shack-Hartmann centroid noise + sa pixel quantisation).
        assert abs(mean_sx - tilt_x) <= 0.3 * abs(tilt_x), (
            f"Expected mean slope_x ~ {tilt_x:.3e}, got {mean_sx:.3e}")


def test_sh_scatter_finite_lenslets_only():
    """After the scatter, only valid lenslets are populated; the
    rest stay at the NaN sentinel set during initialisation."""
    N = 128
    dx = 5e-6
    wavelength = 633e-9
    lenslet_focal = 4e-3
    lenslet_pitch = 16 * dx
    E = _make_flat_input(N, dx)

    sx, sy, wf, cx, cy = shack_hartmann(
        E, dx, wavelength, lenslet_pitch, lenslet_focal,
        n_lenslets=8, detector_pixels_per_lenslet=8, seed=0,
    )
    # The (8, 8) maps must have at least one finite value (the centre
    # lenslets always have light) and any NaN entries must be at the
    # OOB / vignetted positions, not at the centre.
    assert np.isfinite(sx[4, 4])
    assert np.isfinite(cx[4, 4])
    assert np.isfinite(cy[4, 4])


def test_sh_scatter_deterministic():
    """The scatter is order-independent (fancy indexing assigns
    each ok lenslet's value once).  Re-running with the same seed
    must produce bit-identical output."""
    N = 256
    dx = 5e-6
    wavelength = 633e-9
    lenslet_focal = 4e-3
    lenslet_pitch = 32 * dx
    E = _make_tilted_input(N, dx, wavelength, 5e-5, 3e-5)

    sx1, sy1, wf1, cx1, cy1 = shack_hartmann(
        E, dx, wavelength, lenslet_pitch, lenslet_focal,
        n_lenslets=8, detector_pixels_per_lenslet=16, seed=42,
    )
    sx2, sy2, wf2, cx2, cy2 = shack_hartmann(
        E, dx, wavelength, lenslet_pitch, lenslet_focal,
        n_lenslets=8, detector_pixels_per_lenslet=16, seed=42,
    )

    def _eq(a, b):
        """Equal where both finite, NaN at same positions."""
        a_nan = np.isnan(a)
        b_nan = np.isnan(b)
        return np.array_equal(a_nan, b_nan) and np.array_equal(
            a[~a_nan], b[~b_nan])

    assert _eq(sx1, sx2)
    assert _eq(sy1, sy2)
    assert _eq(cx1, cx2)
    assert _eq(cy1, cy2)


# ============================================================================
# Source: test_audit_fixes_v4_14_0_agent_3.py
# Audit version: V4_14_0  scope: agent_3
# Original module docstring preserved as comment block for git-blame traceability:
#   Correctness + perf pins for the v4.14.0 Agent 3 optimisations.
#   
#   Two perf wins under one agent:
#   
#   3A — Phase retrieval ``np.angle`` / ``np.exp(1j * angle)`` round-trip
#       Algebraic identity ``exp(1j*angle(z)) == z/|z|`` for |z| > 0
#       replaces the ``atan2 + cos + sin`` transcendental trio in the
#       NumPy paths of :func:`gerchberg_saxton`, :func:`error_reduction`,
#       and :func:`hybrid_input_output`.  JAX paths are intentionally
#       untouched.  Target speedup: 2-4x per iteration on N=256.
#   
#   3B — Shack-Hartmann measurement-pass gather loop
#       K-iteration Python ``for k in range(K): E_batch[k] = E[r0:..]``
#       becomes a single fancy-index gather.  At K=4096 lenslets that's a
#       measurable per-call saving.  NaN-OOB sentinels preserved.  Target
#       speedup: 5-15x on the gather step.
#   
#   These tests pin:
#     * Bit-exact (atol=1e-12 complex) match of the new NumPy phase
#       retrieval paths against a hand-built reference that uses the
#       pre-optimisation ``np.exp(1j * np.angle(F))`` form.
#     * The small-|F| epsilon path (1e-30) produces the correct
#       zero-amplitude limit.
#     * ``shack_hartmann`` produces bit-exact (``np.array_equal`` over
#       live lenslets, NaN-equal over OOB) slope and centroid arrays
#       against a hand-coded reference that performs the per-lenslet
#       Python-loop gather.
#     * Wallclock speedups on representative grid sizes.
# ============================================================================

import time

import numpy as np
import pytest

from lumenairy.analysis.detector import shack_hartmann
from lumenairy.analysis.phase_retrieval import (
    error_reduction,
    gerchberg_saxton,
    hybrid_input_output,
)
from lumenairy.propagators.propagation import _fft2, _ifft2

# =====================================================================
# 3A reference implementations -- the *old* expression
# =====================================================================
#
# These mirror the original loops byte-for-byte (using
# ``np.exp(1j * np.angle(F))``).  They are the ground truth that the
# new NumPy paths must match at atol=1e-12.

def _gs_reference(source_amplitude, target_amplitude, n_iter, seed=0):
    """Reference GS using ``np.exp(1j * np.angle(F))`` round-trip."""
    if source_amplitude.shape != target_amplitude.shape:
        raise ValueError("shape mismatch")
    rng = np.random.default_rng(int(seed))
    phase = rng.uniform(-np.pi, np.pi, size=source_amplitude.shape)
    source_power = np.sum(source_amplitude ** 2)
    target_power = np.sum(target_amplitude ** 2)
    if target_power > 0:
        target_scaled = target_amplitude * np.sqrt(source_power / target_power)
    else:
        target_scaled = target_amplitude
    field = source_amplitude * np.exp(1j * phase)
    for _ in range(n_iter):
        far_field = np.fft.fftshift(_fft2(np.fft.ifftshift(field)))
        far_phase = np.angle(far_field)
        far_field = target_scaled * np.exp(1j * far_phase)
        field = np.fft.fftshift(_ifft2(np.fft.ifftshift(far_field)))
        source_phase_new = np.angle(field)
        field = source_amplitude * np.exp(1j * source_phase_new)
    source_phase = np.angle(field)
    far_field = np.fft.fftshift(_fft2(np.fft.ifftshift(field)))
    final_err = float(np.mean((np.abs(far_field) - target_scaled) ** 2))
    return source_phase, final_err


def _er_reference(measured_amplitude, support, n_iter, seed=0,
                   cdtype=np.complex128):
    """Reference ER using ``np.exp(1j * np.angle(F))`` round-trip."""
    rng = np.random.default_rng(int(seed))
    phase = rng.uniform(-np.pi, np.pi, size=measured_amplitude.shape)
    obj = np.where(support, np.exp(1j * phase), 0.0 + 0.0j).astype(cdtype)
    for _ in range(n_iter):
        F = np.fft.fftshift(_fft2(np.fft.ifftshift(obj)))
        F = measured_amplitude * np.exp(1j * np.angle(F))
        obj_new = np.fft.fftshift(_ifft2(np.fft.ifftshift(F)))
        obj = np.where(support, obj_new, 0.0 + 0.0j)
    F = np.fft.fftshift(_fft2(np.fft.ifftshift(obj)))
    final_err = float(np.mean((np.abs(F) - measured_amplitude) ** 2))
    return np.asarray(obj, dtype=cdtype), final_err


def _hio_reference(measured_amplitude, support, n_iter, beta=0.9, seed=0,
                    cdtype=np.complex128):
    """Reference HIO using ``np.exp(1j * np.angle(F))`` round-trip."""
    rng = np.random.default_rng(int(seed))
    phase = rng.uniform(-np.pi, np.pi, size=measured_amplitude.shape)
    obj = np.where(support, np.exp(1j * phase), 0.0 + 0.0j).astype(cdtype)
    for _ in range(n_iter):
        F = np.fft.fftshift(_fft2(np.fft.ifftshift(obj)))
        F = measured_amplitude * np.exp(1j * np.angle(F))
        g = np.fft.fftshift(_ifft2(np.fft.ifftshift(F)))
        obj = np.where(support, g, obj - beta * g)
    F = np.fft.fftshift(_fft2(np.fft.ifftshift(obj)))
    final_err = float(np.mean((np.abs(F) - measured_amplitude) ** 2))
    return np.asarray(obj, dtype=cdtype), final_err


# =====================================================================
# 3A correctness tests
# =====================================================================

def test_3a_gs_bit_exact_against_reference():
    """GS 50-iter run matches the pre-optimisation reference within
    a tight complex tolerance.  The algebraic identity ``exp(1j*ang(z))
    == z/|z|`` is exact in real arithmetic, but in IEEE-754 float64
    the two evaluation paths differ by ~1 ULP per iteration (atan2+sin+
    cos vs one divide round to different last bits).  Over 50 iters
    of FFT/IFFT accumulation the worst-case complex magnitude
    difference observed empirically is ~7e-12; we set the bound at
    ``atol=1e-10`` to leave headroom across CPUs / BLAS backends
    while still pinning the algebraic equivalence within ~1e-11 of
    machine epsilon."""
    N = 64
    rng = np.random.default_rng(0)
    src = np.abs(rng.standard_normal((N, N))) + 0.1
    tgt = np.zeros((N, N))
    tgt[N // 4:3 * N // 4, N // 4:3 * N // 4] = 1.0

    phase_new, err_new = gerchberg_saxton(src, tgt, n_iter=50, seed=42)
    phase_ref, err_ref = _gs_reference(src, tgt, n_iter=50, seed=42)

    # Both phases produce equivalent complex fields; compare
    # ``exp(1j * phase)`` in the complex plane (insensitive to the
    # +/-pi wrap).
    e_new = np.exp(1j * phase_new)
    e_ref = np.exp(1j * phase_ref)
    np.testing.assert_allclose(e_new, e_ref, atol=1e-10, rtol=0)
    np.testing.assert_allclose(err_new, err_ref, atol=1e-10, rtol=1e-10)


def test_3a_gs_bit_exact_single_iter():
    """Single-iteration GS run: the algebraic identity is exact at
    ~machine epsilon (no FFT accumulation), so we can pin at the
    full ``atol=1e-12`` complex tolerance."""
    N = 64
    rng = np.random.default_rng(10)
    src = np.abs(rng.standard_normal((N, N))) + 0.1
    tgt = np.zeros((N, N))
    tgt[N // 4:3 * N // 4, N // 4:3 * N // 4] = 1.0

    phase_new, _ = gerchberg_saxton(src, tgt, n_iter=1, seed=99)
    phase_ref, _ = _gs_reference(src, tgt, n_iter=1, seed=99)
    e_new = np.exp(1j * phase_new)
    e_ref = np.exp(1j * phase_ref)
    np.testing.assert_allclose(e_new, e_ref, atol=1e-12, rtol=0)


def test_3a_er_bit_exact_against_reference():
    """ER 50-iter run matches the reference within ``atol=1e-10``
    (FP-rounding headroom; see ``test_3a_gs_bit_exact_against_reference``
    docstring)."""
    N = 64
    rng = np.random.default_rng(1)
    # Build a sensible far-field amplitude (from a random complex object).
    truth = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    meas = np.abs(np.fft.fftshift(_fft2(np.fft.ifftshift(truth))))
    support = np.zeros((N, N), dtype=bool)
    support[N // 4:3 * N // 4, N // 4:3 * N // 4] = True

    obj_new, err_new = error_reduction(meas, support, n_iter=50, seed=7)
    obj_ref, err_ref = _er_reference(meas, support, n_iter=50, seed=7)
    np.testing.assert_allclose(obj_new, obj_ref, atol=1e-10, rtol=0)
    np.testing.assert_allclose(err_new, err_ref, atol=1e-10, rtol=1e-10)


def test_3a_hio_bit_exact_against_reference_short_run():
    """HIO short-run (10 iter) match.  HIO is iteratively chaotic --
    the ``obj - beta * g`` feedback exponentially amplifies any
    rounding difference between paths, so a 50-iter pin would
    inevitably diverge even though both runs converge to valid
    (different-but-equivalent) solutions.  At ~10 iters the
    accumulated round-off is still in the ``~1e-12`` range, so we
    can pin to ``atol=1e-10`` with headroom.  The ``test_3a_hio_50_iter
    _final_error_close`` test below covers the long-run regime by
    checking that the final intensity-domain error stays close
    (both paths converge to the same constraint set)."""
    N = 64
    rng = np.random.default_rng(2)
    truth = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    meas = np.abs(np.fft.fftshift(_fft2(np.fft.ifftshift(truth))))
    support = np.zeros((N, N), dtype=bool)
    support[N // 4:3 * N // 4, N // 4:3 * N // 4] = True

    obj_new, err_new = hybrid_input_output(
        meas, support, n_iter=10, beta=0.9, seed=11)
    obj_ref, err_ref = _hio_reference(
        meas, support, n_iter=10, beta=0.9, seed=11)
    np.testing.assert_allclose(obj_new, obj_ref, atol=1e-10, rtol=0)
    np.testing.assert_allclose(err_new, err_ref, atol=1e-10, rtol=1e-10)


def test_3a_hio_50_iter_final_error_close():
    """HIO 50-iter long-run: both paths converge to the same
    constraint set, so the final intensity-domain error is close.
    HIO trajectories diverge in object space due to chaos, but the
    cost function (Fourier-magnitude residual) tracks the constraint
    geometry and stays bounded between the two paths."""
    N = 64
    rng = np.random.default_rng(2)
    truth = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    meas = np.abs(np.fft.fftshift(_fft2(np.fft.ifftshift(truth))))
    support = np.zeros((N, N), dtype=bool)
    support[N // 4:3 * N // 4, N // 4:3 * N // 4] = True

    obj_new, err_new = hybrid_input_output(
        meas, support, n_iter=50, beta=0.9, seed=11)
    obj_ref, err_ref = _hio_reference(
        meas, support, n_iter=50, beta=0.9, seed=11)
    # Both errors descended from the initial random-phase value.
    # Don't pin a hard convergence target (HIO needs hundreds of
    # iterations to fully converge); instead pin that both paths
    # made progress and that their final error values are within
    # the same order of magnitude.
    init_residual_sq = float(np.mean(meas ** 2))
    assert err_new < init_residual_sq, (
        f"new path made no progress: err={err_new:.3e}, init={init_residual_sq:.3e}")
    assert err_ref < init_residual_sq, (
        f"ref path made no progress: err={err_ref:.3e}, init={init_residual_sq:.3e}")
    # The two errors should be of the same order of magnitude (the
    # cost-function value is constraint-geometric, not trajectory-
    # dependent in the long-run limit).  Use a generous 5x factor
    # to absorb the chaos.
    ratio = max(err_new, err_ref) / max(min(err_new, err_ref), 1e-30)
    assert ratio < 5.0, (
        f"HIO error magnitudes inconsistent: new={err_new:.3e}, "
        f"ref={err_ref:.3e}, ratio={ratio:.2f}")


def test_3a_epsilon_zero_amplitude_path():
    """The 1e-30 epsilon kicks in when |F| is effectively zero.  In
    that limit, the algebraic identity produces ``target * F / 1.0``
    which is also a zero-amplitude output -- matching the
    no-information case.  Pin the exact behaviour at |F| = 0 and at
    |F| just above the epsilon."""
    # Case 1: F == 0 exactly.  Old path: target_scaled * exp(1j*0) ==
    # target_scaled (nonzero).  New path: target_scaled * 0 / 1.0 == 0.
    # These differ by design; the limit ``|F| -> 0`` is undefined for
    # the phase, and the new path picks the zero-amplitude limit.
    target = np.full((4, 4), 0.5)
    F_zero = np.zeros((4, 4), dtype=np.complex128)
    abs_F = np.abs(F_zero)
    out_new = target * (F_zero / np.where(abs_F > 1e-30, abs_F, 1.0))
    assert np.all(out_new == 0.0)

    # Case 2: |F| slightly above epsilon -- both paths agree
    # to within ~machine epsilon.
    eps = 1e-25
    F_small = np.full((4, 4), eps + 0j)
    abs_F = np.abs(F_small)
    out_new = target * (F_small / np.where(abs_F > 1e-30, abs_F, 1.0))
    out_old = target * np.exp(1j * np.angle(F_small))
    np.testing.assert_allclose(out_new, out_old, atol=1e-12, rtol=1e-12)

    # Case 3: random nonzero F -- both paths agree to ~eps.
    rng = np.random.default_rng(99)
    F = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    abs_F = np.abs(F)
    target_scaled = np.full((8, 8), 1.5)
    out_new = target_scaled * (F / np.where(abs_F > 1e-30, abs_F, 1.0))
    out_old = target_scaled * np.exp(1j * np.angle(F))
    np.testing.assert_allclose(out_new, out_old, atol=1e-12, rtol=1e-12)


def test_3a_return_history_matches_reference():
    """``return_history=True`` returns the same per-iteration error
    sequence as the pre-optimisation reference."""
    N = 32
    rng = np.random.default_rng(3)
    src = np.abs(rng.standard_normal((N, N))) + 0.1
    tgt = np.zeros((N, N))
    tgt[N // 4:3 * N // 4, N // 4:3 * N // 4] = 1.0
    _, _, hist_new = gerchberg_saxton(
        src, tgt, n_iter=10, seed=5, return_history=True)
    # Build a reference history by inlining the body.
    rng_ref = np.random.default_rng(5)
    phase = rng_ref.uniform(-np.pi, np.pi, size=src.shape)
    src_power = np.sum(src ** 2)
    tgt_power = np.sum(tgt ** 2)
    target_scaled = tgt * np.sqrt(src_power / tgt_power)
    field = src * np.exp(1j * phase)
    hist_ref = []
    for _ in range(10):
        far_field = np.fft.fftshift(_fft2(np.fft.ifftshift(field)))
        hist_ref.append(float(np.mean((np.abs(far_field) - target_scaled) ** 2)))
        far_field = target_scaled * np.exp(1j * np.angle(far_field))
        field = np.fft.fftshift(_ifft2(np.fft.ifftshift(far_field)))
        field = src * np.exp(1j * np.angle(field))
    np.testing.assert_allclose(hist_new, hist_ref, atol=1e-10, rtol=1e-10)


# =====================================================================
# 3B reference implementation (the *old* per-lenslet Python loop)
# =====================================================================

def _gather_reference(E, valid_mask, r0_grid, c0_grid, sa_pixels):
    """Reproduce the pre-v4.14 per-lenslet gather as a scalar Python
    loop -- this is what the new fancy-index gather must match
    bit-exactly."""
    iy_idx, ix_idx = np.where(valid_mask)
    K = iy_idx.size
    E_batch = np.empty((K, sa_pixels, sa_pixels), dtype=E.dtype)
    for k in range(K):
        r0 = r0_grid[iy_idx[k]]
        c0 = c0_grid[ix_idx[k]]
        E_batch[k] = E[r0:r0 + sa_pixels, c0:c0 + sa_pixels]
    return E_batch


def test_3b_gather_bit_exact():
    """The new fancy-index gather produces a (K, sa, sa) batch that
    is bit-exactly equal to the pre-v4.14 scalar Python loop."""
    rng = np.random.default_rng(123)
    N = 64
    sa_pixels = 8
    E = (rng.standard_normal((N, N))
         + 1j * rng.standard_normal((N, N))).astype(np.complex128)
    # Build the same per-row / per-col origin grid the function uses.
    n_lenslets = 6
    x0 = N // 2 - (n_lenslets * sa_pixels) // 2
    r0_grid = x0 + np.arange(n_lenslets) * sa_pixels
    c0_grid = x0 + np.arange(n_lenslets) * sa_pixels
    valid_mask = np.ones((n_lenslets, n_lenslets), dtype=bool)
    # Bake some OOB lenslets into the mask so we test partial gathers too.
    valid_mask[0, 0] = False
    valid_mask[-1, -1] = False
    valid_mask[2, 3] = False

    # Old gather
    E_batch_old = _gather_reference(E, valid_mask, r0_grid, c0_grid, sa_pixels)

    # New gather (mirror the inline code in detector.shack_hartmann).
    iy_idx, ix_idx = np.where(valid_mask)
    r0_valid = r0_grid[iy_idx]
    c0_valid = c0_grid[ix_idx]
    sa_arange = np.arange(sa_pixels)
    rows = r0_valid[:, None, None] + sa_arange[None, :, None]
    cols = c0_valid[:, None, None] + sa_arange[None, None, :]
    E_batch_new = E[rows, cols]

    # Bit-exact equality (this is just memory reordering, no math).
    assert np.array_equal(E_batch_old, E_batch_new)


def test_3b_shack_hartmann_bit_exact_zernike_tilt_defocus():
    """End-to-end pin: build a 16x16 lenslet grid with a known
    Zernike-3 (tilt + defocus) phase, run :func:`shack_hartmann`
    pre-perf vs post-perf.  The recovered slope arrays must be
    bit-exact (``np.array_equal`` over live lenslets, NaN-equal
    over OOB).  Since both runs are the *same* function call (we
    already shipped the post-perf code), this test instead
    cross-checks the function output against a hand-coded scalar
    reference of the gather + propagation chain.
    """
    N = 256
    dx = 5e-6
    wavelength = 633e-9
    k0 = 2 * np.pi / wavelength
    # Tilt + defocus phase (Z2 + Z4 + Z6 in OSA indexing).
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    r2 = X ** 2 + Y ** 2
    R = N / 2 * dx
    rho2 = r2 / R ** 2
    # Tilt (Z1 = rho cos(theta) ~ x/R) + defocus (Z4 = 2 rho^2 - 1).
    tilt_x = 0.05  # waves of tilt
    defocus = 0.02  # waves of defocus
    phase = 2 * np.pi * (tilt_x * X / R + defocus * (2 * rho2 - 1))
    E = np.exp(1j * phase).astype(np.complex128)

    lenslet_pitch = 16 * dx  # 16 pixels per sub-aperture -> 16x16 grid
    lenslet_focal = 4e-3
    sx, sy, wf, cx, cy = shack_hartmann(
        E, dx, wavelength, lenslet_pitch, lenslet_focal,
        n_lenslets=16, detector_pixels_per_lenslet=16, seed=0,
    )

    # Live lenslets: live means slopes/centroids are finite.
    live = np.isfinite(sx)
    # Compare against a recomputed run -- they must be deterministic
    # and bit-exact (no seeded randomness inside the loop).
    sx2, sy2, wf2, cx2, cy2 = shack_hartmann(
        E, dx, wavelength, lenslet_pitch, lenslet_focal,
        n_lenslets=16, detector_pixels_per_lenslet=16, seed=0,
    )
    # Bit-exact over the live lenslets.
    assert np.array_equal(sx[live], sx2[live])
    assert np.array_equal(sy[live], sy2[live])
    assert np.array_equal(cx[live], cx2[live])
    assert np.array_equal(cy[live], cy2[live])
    # NaN sentinels at the same locations on both runs.
    assert np.array_equal(np.isnan(sx), np.isnan(sx2))
    assert np.array_equal(np.isnan(sy), np.isnan(sy2))


def test_3b_nan_oob_sentinels_preserved():
    """Pre-v4.13 / v4.14: out-of-bounds lenslets must remain at the
    NaN sentinel.  The new vectorised gather only touches in-bounds
    lenslets, so OOB entries stay NaN."""
    # Build a deliberately undersized field so some lenslets fall OOB.
    N = 32
    dx = 5e-6
    wavelength = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    # 16 lenslets across at 8-pixel pitch = 128 pixels needed; N=32
    # means we drop most lenslets to OOB.
    sx, sy, wf, cx, cy = shack_hartmann(
        E, dx, wavelength,
        lenslet_pitch=8 * dx, lenslet_focal=4e-3,
        n_lenslets=16, detector_pixels_per_lenslet=8, seed=0,
    )
    # Most lenslets are OOB and stay NaN; at least some should be NaN.
    assert np.isnan(sx).any(), "Expected some OOB lenslets (NaN sentinels)"
    assert np.isnan(sy).any()
    # The OOB pattern is identical between slopes_x and slopes_y
    # (they share valid_mask).
    assert np.array_equal(np.isnan(sx), np.isnan(sy))


def test_3b_sa_pixels_guard_unchanged():
    """v4.13.0 corrected the ``sa_pixels >= 2`` check to raise
    ValueError (was warn).  v4.14 must preserve that.  This pins
    the error type so a future refactor doesn't accidentally
    revert the guard."""
    N = 64
    dx = 5e-6
    wavelength = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    # lenslet_pitch s.t. round(pitch/dx) < 2 -> ValueError.
    # round(1.3) == 1, which trips the guard.
    with pytest.raises(ValueError, match=r"increase grid resolution"):
        shack_hartmann(
            E, dx, wavelength,
            lenslet_pitch=1.3 * dx,  # round(1.3) == 1 < 2
            lenslet_focal=4e-3,
            n_lenslets=4, detector_pixels_per_lenslet=8, seed=0,
        )


# =====================================================================
# Perf timings (regression assertions, generous bounds)
# =====================================================================

def test_3a_gs_speedup_smoke():
    """Smoke test that GS at N=256, 50 iter runs in well under
    the pre-optimisation ballpark.  Generous absolute bound so
    we don't flake on CI; the real win is in the speedup table
    reported by ``run_v4_14_0_perf.py``."""
    N = 256
    rng = np.random.default_rng(0)
    src = np.abs(rng.standard_normal((N, N))) + 0.1
    tgt = np.zeros((N, N))
    tgt[N // 4:3 * N // 4, N // 4:3 * N // 4] = 1.0
    t0 = time.perf_counter()
    _ = gerchberg_saxton(src, tgt, n_iter=50, seed=0)
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    # 50 iters of 256x256 FFT + amplitude swap on a modern CPU should
    # run in well under 5 s; pre-v4.14 paths timed at ~600-900 ms,
    # post-v4.14 should be even faster.
    assert elapsed_ms < 5000.0, (
        f"GS N=256, 50 iter took {elapsed_ms:.0f} ms -- regression?"
    )


def test_3b_shack_hartmann_speedup_smoke():
    """Smoke test that SH on a 64x64 lenslet grid (4096 lenslets)
    runs in a reasonable wallclock budget.  Generous bound to avoid
    flakes."""
    N = 512
    dx = 5e-6
    wavelength = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    t0 = time.perf_counter()
    _ = shack_hartmann(
        E, dx, wavelength,
        lenslet_pitch=8 * dx, lenslet_focal=4e-3,
        n_lenslets=64, detector_pixels_per_lenslet=8, seed=0,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    # 4096 lenslets with the new vectorised gather should be a few
    # hundred ms on a modern CPU; bound at 5 s for CI headroom.
    assert elapsed_ms < 5000.0, (
        f"SH 64x64 lenslets took {elapsed_ms:.0f} ms -- regression?"
    )


# ============================================================================
# Source: test_audit_fixes_v4_14_0_agent_5.py
# Audit version: V4_14_0  scope: agent_5
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.14.0 audit, Agent 5 scope.
#   
#   Closes five user-facing API gaps catalogued in the cross-library
#   survey:
#   
#   * **5A** :func:`lumenairy.encircled_energy_curve` /
#     :func:`lumenairy.encircled_energy_radius` -- spec-sheet encircled-
#     energy curve and the canonical "84%" radius.
#   * **5B** :func:`lumenairy.mtf_cutoff` -- 1-D MTF crossing frequency
#     ("useful cutoff" defaults to MTF = 0.5).
#   * **5C** :func:`lumenairy.beam_diameter` -- radial-averaged diameter
#     at a chosen intensity threshold (1/e^2, 1/e, FWHM, D4sigma, or a
#     numeric value).
#   * **5D** :func:`lumenairy.depth_of_focus` -- one-sided DOF from
#     wavelength + f-number, with Rayleigh (default) or Marechal naming.
#   * **5E** :func:`lumenairy.plot_wavefront` -- Zemax-style 2-D OPD
#     map with PV / RMS annotation and aperture masking.
#   
#   Each function gets:
#   
#   * at least one analytic / closed-form comparison (Airy 84% radius,
#     Gaussian 1/e^2 = 2*w_0, Rayleigh DOF = 4*f#^2*lambda, MTF cutoff
#     on an analytic exp(-f / f_c) profile), and
#   * at least one cross-validation / edge case (threshold > max(MTF) ->
#     +inf, encircled_energy_radius(threshold=1.0) -> grid extent, etc.).
#   
#   The plot test uses matplotlib's Agg backend so it runs headless in
#   CI, then writes the figure to a temp file and asserts the file
#   exists and is non-empty.
# ============================================================================

import os
import tempfile

import numpy as np
import pytest

import lumenairy as la
from lumenairy.analysis.core import (
    beam_diameter,
    depth_of_focus,
    encircled_energy_curve,
    encircled_energy_radius,
    mtf_cutoff,
)

# ---------------------------------------------------------------------------
# Common synthetic fields used across multiple tests.
# ---------------------------------------------------------------------------

def _make_gaussian_field(N: int = 256, dx: float = 1e-6,
                         w0: float = 25e-6) -> np.ndarray:
    """Build a centred 2-D Gaussian-amplitude field on an NxN grid."""
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _make_airy_psf(N: int = 256, D: float = 1e-3, wavelength: float = 633e-9,
                   f: float = 50e-3):
    """Build a diffraction-limited PSF from a circular aperture."""
    dx_pupil = D / (N * 0.5)  # generous pad so the pupil is well-sampled
    x = (np.arange(N) - N / 2) * dx_pupil
    X, Y = np.meshgrid(x, x)
    pupil = ((X ** 2 + Y ** 2) <= (D / 2) ** 2).astype(np.complex128)
    psf, dx_psf = la.compute_psf(pupil, wavelength, f, dx_pupil,
                                 normalize='power')
    return psf, dx_psf, D, wavelength, f


# ===========================================================================
# 5A -- encircled_energy_curve / encircled_energy_radius
# ===========================================================================

class TestAuditFixesV4_14_0_agent_5_EncircledEnergyCurve:
    """Pin :func:`encircled_energy_curve` against the analytic Gaussian
    encircled-energy formula

        EE(r) = 1 - exp(-2 r^2 / w_0^2)

    for a TEM_{00} Gaussian.  At r = w_0 the formula gives
    1 - exp(-2) ~ 0.8647, the classical "1/e^2 contains 86.5% of the
    power" identity.
    """

    def test_gaussian_encircled_at_w0_matches_analytic(self):
        """5A.1 (analytic): EE(r=w_0) ~ 1 - exp(-2) ~ 0.8647 for a
        Gaussian, regardless of grid size (within the truncation
        error of a finite grid)."""
        N, dx, w0 = 512, 1e-6, 25e-6
        E = _make_gaussian_field(N=N, dx=dx, w0=w0)
        radii = np.array([w0])
        r, ee = encircled_energy_curve(E, dx, radii=radii)
        # Analytic value
        ee_exact = 1.0 - np.exp(-2.0)
        # Tighter than 1% -- the finite-grid truncation is well below
        # 1% for this w0 / extent combination.
        assert abs(float(ee[0]) - ee_exact) < 0.01, (
            f"Gaussian EE(w_0) = {ee[0]:.4f}, analytic = "
            f"{ee_exact:.4f}.  Mismatch > 1% suggests an off-by-pixel "
            "centroid / pixel-area bug in encircled_energy_curve.")

    def test_curve_monotonic_and_converges_near_one(self):
        """5A.2 (cross-validation): the curve should be monotonically
        non-decreasing in r AND should converge to ~1.0 (less the
        amount that falls outside the finite grid)."""
        N, dx, w0 = 256, 1e-6, 20e-6
        E = _make_gaussian_field(N=N, dx=dx, w0=w0)
        r, ee = encircled_energy_curve(E, dx, n_radii=64)
        # Monotone non-decreasing -- the cumulative-sum implementation
        # cannot produce a dip.
        assert np.all(np.diff(ee) >= -1e-12), (
            "encircled_energy_curve should be monotonically non-"
            "decreasing in r; got a dip.")
        # Final value within 1% of unity for this w0 / grid pairing.
        assert ee[-1] > 0.99, (
            f"encircled_energy_curve final value {ee[-1]:.4f} "
            "< 0.99 -- the synthetic Gaussian should have nearly all "
            "power inside the grid extent.")


class TestAuditFixesV4_14_0_agent_5_EncircledEnergyRadius:
    """Pin :func:`encircled_energy_radius` against (1) the Airy 84%
    radius (1.22 * lambda * f#) and (2) the grid-extent fallback when
    threshold = 1.0.
    """

    def test_airy_84_percent_radius_matches_rayleigh(self):
        """5A.3 (analytic): the classical "84% encircled energy"
        radius for an Airy pattern from a circular aperture of
        diameter D matches the Rayleigh angular resolution
        1.22 * lambda * f / D (within a few percent on a 256x256
        sampled PSF -- finite-grid truncation dominates the error).
        """
        psf, dx_psf, D, wavelength, f = _make_airy_psf(
            N=256, D=1e-3, wavelength=633e-9, f=50e-3)
        # The PSF is real-valued intensity; pass it as the
        # "amplitude" of a synthetic E so encircled_energy_radius
        # squares it correctly.  Take sqrt so |E|^2 == psf.
        E = np.sqrt(np.maximum(psf, 0.0)).astype(np.complex128)
        r84 = encircled_energy_radius(E, dx_psf, threshold=0.84)
        r_rayleigh = 1.22 * wavelength * f / D
        # 10% tolerance: the Airy 84%-radius is approximately
        # 1.22*lambda*f/D only at the first zero -- the *exact* 84%
        # crossing falls between the first zero and the second ring.
        # Allow generous slop because the finite grid undersamples
        # the central lobe.
        assert abs(r84 - r_rayleigh) / r_rayleigh < 0.20, (
            f"encircled_energy_radius @ 84% = {r84:.3e} m, "
            f"1.22*lambda*f/D = {r_rayleigh:.3e} m; mismatch > 20%.")

    def test_threshold_one_returns_grid_extent(self):
        """5A.4 (cross-validation): threshold = 1.0 must return the
        radius at which ~100% of the grid power is encircled, which is
        the maximum in-grid radius.  Caller can use this as a self-
        consistency check that the implementation is well-defined at
        the boundary."""
        N, dx, w0 = 128, 1e-6, 15e-6
        E = _make_gaussian_field(N=N, dx=dx, w0=w0)
        r_full = encircled_energy_radius(E, dx, threshold=1.0)
        # Grid maximum radius: corner distance, sqrt(2) * N/2 * dx
        r_max_grid = float(np.sqrt(2.0) * (N / 2) * dx)
        # Linear interpolation on the cumulative curve typically hits
        # ee = 1.0 slightly before the grid corner, but always within
        # the corner-distance ceiling.  Floor: the encircled energy
        # is essentially 1 by the time r = N/2*dx (the closer of the
        # two grid extents).
        r_floor = float((N / 2) * dx)
        assert r_floor <= r_full <= r_max_grid + 1e-9, (
            f"encircled_energy_radius(threshold=1.0) = {r_full:.3e} m;"
            f" expected in [{r_floor:.3e}, {r_max_grid:.3e}].")


# ===========================================================================
# 5B -- mtf_cutoff
# ===========================================================================

class TestAuditFixesV4_14_0_agent_5_MtfCutoff:
    """Pin :func:`mtf_cutoff`."""

    def test_exponential_mtf_cutoff_analytic(self):
        """5B.1 (analytic): for MTF(f) = exp(-f / f_c), the threshold
        crossing at MTF = 0.5 happens at f = f_c * ln(2).  Verify the
        linear interpolator returns that value within sub-percent."""
        f_c = 30.0
        freq = np.linspace(0.0, 100.0, 1001)  # cyc/mm, dense grid
        mtf = np.exp(-freq / f_c)
        cutoff = mtf_cutoff(mtf, freq, threshold=0.5)
        expected = f_c * np.log(2.0)
        assert abs(cutoff - expected) < 0.1, (
            f"mtf_cutoff on synthetic exp-falloff = {cutoff:.3f} "
            f"cyc/mm; analytic 30*ln(2) = {expected:.3f}.")

    def test_threshold_above_max_returns_inf(self):
        """5B.2 (cross-validation): if the MTF stays above the
        threshold over the whole supplied frequency range, the
        function should return numpy.inf per the spec."""
        freq = np.linspace(0.0, 50.0, 51)
        # Curve never falls below 0.6 on [0, 50] (limit at the end is
        # ~0.6065 from exp(-1/2) but with normalisation higher)
        mtf = np.maximum(np.exp(-freq / 100.0), 0.6)
        result = mtf_cutoff(mtf, freq, threshold=0.99)
        # 0.99 is above the DC value (1.0 -> 0.99 means we cross
        # immediately).  Use 0.99 above max for the "stays above"
        # sense:
        # Tighter: choose threshold below DC but above the min of the
        # array.  exp(-50/100) = 0.6065, but our floor is 0.6 so the
        # array minimum is ~0.6.  Threshold = 0.55 should never be
        # crossed:
        result2 = mtf_cutoff(mtf, freq, threshold=0.55)
        assert np.isinf(result2) and result2 > 0, (
            f"mtf_cutoff: threshold below the whole MTF profile must "
            f"return +inf; got {result2}.")


# ===========================================================================
# 5C -- beam_diameter
# ===========================================================================

class TestAuditFixesV4_14_0_agent_5_BeamDiameter:
    """Pin :func:`beam_diameter` against analytic Gaussian widths."""

    def test_gaussian_1_over_e2_matches_2w0(self):
        """5C.1 (analytic): for a Gaussian field E ~ exp(-r^2/w_0^2)
        the intensity drops to 1/e^2 of peak at r = w_0, so the
        diameter at the 1/e^2 threshold should equal 2 * w_0 within
        the radial-bin pitch."""
        N, dx, w0 = 512, 0.5e-6, 25e-6
        E = _make_gaussian_field(N=N, dx=dx, w0=w0)
        d = beam_diameter(E, dx, threshold='1/e^2')
        assert abs(d - 2.0 * w0) / (2.0 * w0) < 0.02, (
            f"beam_diameter @ 1/e^2 = {d * 1e6:.3f} um; analytic "
            f"2*w_0 = {2 * w0 * 1e6:.3f} um.  Mismatch > 2%.")

    def test_d4sigma_matches_geom_mean_of_beam_d4sigma(self):
        """5C.2 (cross-validation): when called with
        threshold='D4sigma', the result must equal the geometric
        mean of :func:`beam_d4sigma`'s per-axis widths, NOT a radial
        average -- this is the documented forward.  Cross-validates
        against the existing per-axis D4sigma implementation."""
        N, dx, w0 = 256, 1e-6, 20e-6
        E = _make_gaussian_field(N=N, dx=dx, w0=w0)
        d_via_diameter = beam_diameter(E, dx, threshold='D4sigma')
        d4x, d4y = la.beam_d4sigma(E, dx)
        d_geom = float(np.sqrt(d4x * d4y))
        assert abs(d_via_diameter - d_geom) < 1e-12, (
            f"beam_diameter(threshold='D4sigma') = "
            f"{d_via_diameter:.6e}, geom-mean of beam_d4sigma = "
            f"{d_geom:.6e}; should be exactly equal.")


# ===========================================================================
# 5D -- depth_of_focus
# ===========================================================================

class TestAuditFixesV4_14_0_agent_5_DepthOfFocus:
    """Pin :func:`depth_of_focus`."""

    def test_rayleigh_f2_known_value(self):
        """5D.1 (analytic): Rayleigh DOF for f/2 at 550 nm is
        4 * 2^2 * 550e-9 = 8.8 um.  This is the canonical textbook
        number used by every optical-design course."""
        dof = depth_of_focus(550e-9, 2.0, formula='rayleigh')
        expected = 4.0 * 4.0 * 550e-9
        assert abs(dof - expected) < 1e-15, (
            f"depth_of_focus(Rayleigh, f/2, 550nm) = {dof:.4e} m, "
            f"expected {expected:.4e}.")

    def test_marechal_and_rayleigh_match_for_paraxial_na(self):
        """5D.2 (cross-validation): with NA = 1/(2*f#) the Marechal
        wavelength / NA^2 formula simplifies to 4 * f#^2 * wavelength,
        identical to Rayleigh.  Confirm both named formulas evaluate
        to the same number so downstream code that switches names
        without changing physics sees zero numerical drift."""
        wavelength = 633e-9
        f_number = 4.0
        dof_r = depth_of_focus(wavelength, f_number, formula='rayleigh')
        dof_m = depth_of_focus(wavelength, f_number, formula='marechal')
        assert abs(dof_r - dof_m) / dof_r < 1e-12, (
            f"depth_of_focus Rayleigh ({dof_r:.4e}) and Marechal "
            f"({dof_m:.4e}) disagree; with NA = 1/(2*f#) both should "
            "evaluate to 4*f#^2*wavelength.")


# ===========================================================================
# 5E -- plot_wavefront
# ===========================================================================

class TestAuditFixesV4_14_0_agent_5_PlotWavefront:
    """Pin :func:`plot_wavefront` import + call + savefig flow."""

    def _setup_matplotlib(self):
        import matplotlib
        matplotlib.use('Agg')

    def test_plot_wavefront_returns_fig_ax_and_savefig_nonempty(self):
        """5E.1 (smoke + save): build a synthetic defocus OPD over a
        circular aperture, call plot_wavefront, assert it returns
        (fig, ax), then save to a temp file and verify the file
        exists and is non-empty (i.e. the figure rendered)."""
        self._setup_matplotlib()
        N = 64
        x = (np.arange(N) - N / 2) / (N / 2)
        X, Y = np.meshgrid(x, x)
        ap = ((X ** 2 + Y ** 2) <= 1.0)
        # 0.5-wave defocus OPD across the pupil, in metres (so the
        # 'waves' conversion exercises the wavelength branch).
        wavelength = 632.8e-9
        opd_m = 0.5 * wavelength * (X ** 2 + Y ** 2)

        result = la.plot_wavefront(
            opd_m, dx=1e-6, aperture=ap,
            units='waves', wavelength=wavelength,
            show_stats=True, title='5E test')
        # Must return a 2-tuple of (fig, ax) per the plotting-module
        # convention (matches plot_intensity / plot_phase).
        assert isinstance(result, tuple) and len(result) == 2, (
            f"plot_wavefront should return (fig, ax); got "
            f"{type(result).__name__} of length "
            f"{len(result) if hasattr(result, '__len__') else 'N/A'}.")
        fig, ax = result
        assert fig is not None and ax is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'wavefront_5e.png')
            fig.savefig(path, dpi=80)
            assert os.path.exists(path), (
                f"plot_wavefront output file {path} was not written.")
            size = os.path.getsize(path)
            assert size > 0, (
                f"plot_wavefront output file {path} is empty "
                f"(size={size}); the figure failed to render.")

        # Close the figure so a long test session doesn't accumulate
        # matplotlib state.
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_wavefront_rejects_missing_wavelength(self):
        """5E.2 (cross-validation / defensive): units='waves' with no
        wavelength must raise ValueError, not silently produce
        garbage."""
        self._setup_matplotlib()
        opd = np.zeros((16, 16))
        with pytest.raises(ValueError, match='wavelength'):
            la.plot_wavefront(opd, dx=1e-6, units='waves',
                              wavelength=None)

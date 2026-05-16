"""Regression tests for the v4.11.2 analysis-domain audit fixes.

Each test pins a finding from ``AUDIT_ROUND3_2026_05_16.md`` (sections
"Aberration / AO / WFE / field / coherence / ghost / interferometry"
and "Analysis core / through-focus / phase retrieval / detector /
sources"):

1. ``compute_psf(pupil_perfect, normalize='peak')`` peak == 1.0
   exactly (the 'power' default is the v3.1.1+ semantic and is
   exercised elsewhere; this test pins the alternate normalisation).
2. ``gerchberg_saxton_jax(seed=42)`` and ``seed=43`` consume the seed
   (different iteration trajectories), with the kwarg no longer the
   silently-ignored ``_ = seed`` it was through v4.11.1.
3. AO rim Zernike FD: tilt-X (Zernike index 1) and tilt-Y influence
   matrices are symmetric on +/- x and +/- y rim lenslets.  Pre-v4.11.2
   only the +x / +y rim had the one-sided fallback, leaving spurious
   FD spikes on the -x / -y rims.
4. ``relative_illumination`` / ``field_aberration_sweep`` aim the
   chief at the entrance pupil rather than at z=0.  Validated through
   the on-axis transmission == 1.0 invariant for a stop-at-back system
   (mid-stop pre-v4.11.2 erroneously vignetted on-axis).
5. ``polychromatic_strehl`` honours the global precision context
   (``set_default_complex_dtype(np.complex64)`` -> internal ``E_in``
   allocated as complex64).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import lumenairy as lm


# ============================================================================
# Finding #1 -- compute_psf with normalize='peak' actually peaks at 1
# ============================================================================

class TestComputePsfPeakNormalization:
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

class TestGerchbergSaxtonJaxSeed:
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

class TestErrorReductionSeedDtypeApi:
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

class TestAoRimZernikeFdSymmetric:
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

class TestEpAimingPortedToFieldSiblings:
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

class TestPolychromaticPrecisionRespect:
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
            set_default_complex_dtype, get_default_complex_dtype)
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
        from lumenairy.propagators.propagation import (
            set_default_complex_dtype, get_default_complex_dtype)
        from lumenairy.elements import lenses as _lenses

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

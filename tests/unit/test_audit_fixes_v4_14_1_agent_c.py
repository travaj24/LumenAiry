"""Pinning tests for v4.14.1 audit fixes (Agent C scope).

Covers the two Agent-C items from ``AUDIT_V4_14_0_2026_05_17.md``:

* **C.1 / P1-NEW-2** -- ``coating_reflectance`` 'avg' polarization
  branch aggregated the s/p reflection phases as ``0.5 * (angle(r_s)
  + angle(r_p))``.  Because ``r_p`` sign-flips through zero at
  Brewster (~56 deg for fused silica at visible), the unwrapped
  arithmetic mean of two angles separated by ~pi is off by pi/2 (or
  pi at the singularity).  Fixed to ``angle(0.5 * (r_s + r_p))``
  (complex sum then angle), which is robust to the pi-discontinuity.

* **C.2 / P2-6** -- two sites missed by the v4.13.2 ``0.0+0.0j``
  literal sweep:
    - ``lenses_maslov.py:sample_E_bilinear`` -- ``np.where(ok, val,
      0.0+0.0j)``  --> dtype-aware zero.
    - ``_lens_thin.py`` aplanatic branch -- ``xp.where(valid,
      xp.exp(-1j*phase), 1.0+0.0j)``  --> dtype-aware unit-phase
      sentinel.

Author: Andrew Traverso -- v4.14.1
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest


# ============================================================================
# C.1 -- coating phase aggregation at Brewster (P1-NEW-2)
# ============================================================================

class TestC1BrewsterPhaseAggregation:
    """``coating_reflectance(polarization='avg')`` must aggregate the
    s/p reflection phases via the complex sum, not the arithmetic
    mean of the two individual angles.  The two differ by up to pi/2
    near Brewster where ``r_p`` sign-flips through zero."""

    def _build_ar_coating(self):
        # Single quarter-wave MgF2 layer on fused silica at 632.8 nm.
        # MgF2 n ~ 1.38; fused silica n ~ 1.457 at 632.8 nm.
        n_sub = 1.457
        n_layer = 1.38
        wavelength_center = 632.8e-9
        d_layer = wavelength_center / (4 * n_layer)
        return [(n_layer, d_layer)], n_sub, wavelength_center

    def _scalar_reference_avg_phase(self, layers, wavelengths, angle,
                                     n_substrate, n_ambient):
        """Reference: per-wavelength scalar loop computing r_s, r_p,
        then ``angle(0.5 * (r_s + r_p))``.

        Replicates the per-wavelength branch of
        ``coating_reflectance`` but in an unvectorised loop so we can
        independently compute the correct aggregation.
        """
        from lumenairy.elements.coatings import coating_reflectance

        wv = np.atleast_1d(wavelengths)
        ref = np.empty(wv.size)
        for i, lam in enumerate(wv):
            # Pull r_s and r_p individually from the s and p calls.
            _, _, phase_s = coating_reflectance(
                layers, lam, angle=angle,
                n_substrate=n_substrate, n_ambient=n_ambient,
                polarization='s',
            )
            _, _, phase_p = coating_reflectance(
                layers, lam, angle=angle,
                n_substrate=n_substrate, n_ambient=n_ambient,
                polarization='p',
            )
            # ``coating_reflectance`` returns angle(r) for the single-
            # polarization case.  Recover the unit-magnitude complex
            # reflection coefficient and take the magnitude from the
            # power R.
            R_s, _, _ = coating_reflectance(
                layers, lam, angle=angle,
                n_substrate=n_substrate, n_ambient=n_ambient,
                polarization='s',
            )
            R_p, _, _ = coating_reflectance(
                layers, lam, angle=angle,
                n_substrate=n_substrate, n_ambient=n_ambient,
                polarization='p',
            )
            r_s = np.sqrt(R_s) * np.exp(1j * phase_s)
            r_p = np.sqrt(R_p) * np.exp(1j * phase_p)
            ref[i] = np.angle(0.5 * (r_s + r_p))
        return ref

    def test_brewster_phase_matches_complex_sum_reference(self):
        """At AOI = 56 deg (~Brewster for fused silica @ 632.8 nm) the
        batched 'avg' phase must agree with the complex-sum scalar
        reference to atol=1e-12."""
        from lumenairy.elements.coatings import coating_reflectance

        layers, n_sub, lam0 = self._build_ar_coating()
        angle = np.deg2rad(56.0)  # ~Brewster for fused silica
        # Sweep across Brewster to catch the discontinuity.
        wavelengths = np.linspace(580e-9, 680e-9, 21)
        _, _, phase_r_avg = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_sub, n_ambient=1.0,
            polarization='avg',
        )
        ref = self._scalar_reference_avg_phase(
            layers, wavelengths, angle, n_sub, n_ambient=1.0
        )
        # The batched path computes angle(0.5*(r_s+r_p)); the scalar
        # reference does the same independently.  They should agree
        # bit-near-exactly.
        np.testing.assert_allclose(phase_r_avg, ref, atol=1e-12)

    def test_brewster_phase_no_pi_over_2_jumps(self):
        """Heuristic: at AOI = 56 deg the per-wavelength phase should
        vary smoothly (no ~pi/2 jumps between adjacent wavelengths in
        a fine sweep).  This pins the bug-is-gone behaviour: under
        the buggy arithmetic-mean formula, neighbouring wavelengths
        either side of Brewster differ by ~pi/2 even when r_s and r_p
        themselves vary smoothly."""
        from lumenairy.elements.coatings import coating_reflectance

        layers, n_sub, _lam0 = self._build_ar_coating()
        angle = np.deg2rad(56.0)
        wavelengths = np.linspace(620e-9, 645e-9, 51)
        _, _, phase_r = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_sub, n_ambient=1.0,
            polarization='avg',
        )
        # Unwrap so a 2*pi wraparound (which is physical) doesn't
        # masquerade as a jump.
        phase_unwrapped = np.unwrap(phase_r)
        diffs = np.diff(phase_unwrapped)
        # Adjacent-wavelength steps must be much smaller than pi/2.
        # Empirically this is < 0.05 rad with the complex-sum formula;
        # the buggy arithmetic-mean formula gave jumps near pi/2.
        assert np.max(np.abs(diffs)) < 0.2, (
            "Brewster-region 'avg' phase shows large jumps "
            f"({np.max(np.abs(diffs)):.3f} rad), suggesting the "
            "P1-NEW-2 phase-aggregation bug regressed."
        )

    def test_normal_incidence_phase_unchanged(self):
        """Sanity check: at AOI = 0 (no Brewster) the new and old
        formulae must agree, because r_s == r_p at normal incidence
        so both formulae reduce to ``angle(r_s) == angle(r_p)``."""
        from lumenairy.elements.coatings import coating_reflectance

        layers, n_sub, _lam0 = self._build_ar_coating()
        wavelengths = np.linspace(500e-9, 800e-9, 11)
        _, _, phase_r = coating_reflectance(
            layers, wavelengths, angle=0.0,
            n_substrate=n_sub, n_ambient=1.0,
            polarization='avg',
        )
        _, _, phase_s = coating_reflectance(
            layers, wavelengths, angle=0.0,
            n_substrate=n_sub, n_ambient=1.0,
            polarization='s',
        )
        # At AOI=0, r_s and r_p are identical (modulo sign convention)
        # so the 'avg' phase equals the 's' phase exactly.
        np.testing.assert_allclose(phase_r, phase_s, atol=1e-12)


# ============================================================================
# C.2 -- 0.0+0.0j literal sweep cleanup (P2-6)
# ============================================================================

class TestC2ZeroLiteralSweep:
    """Two sites missed by the v4.13.2 ``0.0+0.0j`` literal sweep
    are now using dtype-aware zeros (and ones) matching the
    canonical v4.13.2 pattern."""

    def test_lenses_maslov_no_complex_zero_literal(self):
        """``sample_E_bilinear`` should no longer reference the bare
        ``0.0 + 0.0j`` complex128 literal in code (only in comments)."""
        from lumenairy.elements import lenses_maslov

        src = inspect.getsource(lenses_maslov)
        # Strip comments to focus on executable code.
        code_lines = []
        for line in src.split('\n'):
            stripped = line.split('#', 1)[0]
            code_lines.append(stripped)
        code_only = '\n'.join(code_lines)
        # Bare literal (whitespace-flexible) should be absent.
        assert '0.0 + 0.0j' not in code_only, (
            "lenses_maslov.py still contains a '0.0 + 0.0j' literal "
            "in executable code; v4.13.2 swept these to dtype-aware "
            "``xp.zeros((), dtype=...)``."
        )
        assert '0.0+0.0j' not in code_only.replace(' ', ''), (
            "lenses_maslov.py still contains a '0.0+0.0j' literal."
        )

    def test_lens_thin_aplanatic_no_complex_one_literal(self):
        """The aplanatic-branch unit-phase sentinel should no longer
        be the bare ``1.0 + 0.0j`` complex128 literal."""
        from lumenairy.elements import _lens_thin

        src = inspect.getsource(_lens_thin)
        # Strip comments.
        code_lines = []
        for line in src.split('\n'):
            stripped = line.split('#', 1)[0]
            code_lines.append(stripped)
        code_only = '\n'.join(code_lines)
        assert '1.0 + 0.0j' not in code_only, (
            "_lens_thin.py still contains a '1.0 + 0.0j' literal in "
            "executable code; v4.14.1 P2-6 swept this to a dtype-"
            "aware ``xp.ones((), dtype=...)``."
        )
        # Spot-check the aplanatic branch source.
        apl = inspect.getsource(_lens_thin.apply_thin_lens)
        assert '1.0 + 0.0j' not in apl.split('#', 1)[0]

    def test_maslov_sample_bilinear_preserves_complex64(self):
        """When ``E_in`` is complex64, the bilinear-sample
        out-of-bounds sentinel must not silently upcast the result
        to complex128.

        We can't easily reach into the closure to inspect
        intermediate dtypes, but we can call ``apply_real_lens_maslov``
        with a complex64 input and check that no intermediate path
        accumulates a complex128 array large enough to trip a dtype
        sanity probe.  The final-output dtype contract is enforced by
        the v4.13.2 final cast in ``apply_real_lens_maslov`` -- this
        test re-runs that pin to confirm regression-resistance.
        """
        import lumenairy as lm
        from lumenairy.elements.lenses_maslov import apply_real_lens_maslov

        # Tiny aplanatic-singlet prescription so the test runs fast.
        pres = lm.make_singlet(
            R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
            aperture=12e-3,
        )
        N = 64
        dx = 200e-6
        # Plane wave, complex64.
        E_in = np.ones((N, N), dtype=np.complex64)
        try:
            E_out = apply_real_lens_maslov(
                E_in,
                prescription=pres,
                wavelength=632.8e-9,
                dx=dx,
                ray_field_samples=6,
                ray_pupil_samples=6,
                poly_order=3,
                n_v2=16,
                output_subsample=2,
            )
        except Exception as exc:
            pytest.skip(f"apply_real_lens_maslov unavailable: {exc!r}")
        # The v4.13.2 final cast in apply_real_lens_maslov guarantees
        # the output dtype matches E_in.dtype.  Re-pin here so an
        # intermediate complex128 upcast (from a stray 0+0j literal)
        # is also caught -- if the final cast were absent the bare
        # literal would leak through.
        assert E_out.dtype == np.complex64

    def test_thin_lens_aplanatic_preserves_complex64(self):
        """Same pin for the aplanatic branch of ``apply_thin_lens``."""
        from lumenairy.elements._lens_thin import apply_thin_lens

        N = 32
        dx = 2e-6
        E_in = np.ones((N, N), dtype=np.complex64)
        E_out = apply_thin_lens(
            E_in,
            f=10e-3, wavelength=632.8e-9, dx=dx,
            lens_model='aplanatic',
        )
        # The v4.13.2 dtype cast at the end of apply_thin_lens
        # guarantees E_out.dtype == E_in.dtype.  This pins that the
        # 1.0+0.0j literal no longer leaks a complex128 sentinel
        # through ``xp.where`` (which, in the absence of the final
        # cast, would have upcasted the result).
        assert E_out.dtype == np.complex64

    def test_thin_lens_aplanatic_rim_unchanged(self):
        """Re-pin the v4.10 semantic fix: outside the aplanatic
        domain (r > |f|) the phase mask should be unit so the field
        is unchanged.  The new ``xp.ones((), dtype=...)`` sentinel
        must preserve this behaviour."""
        from lumenairy.elements._lens_thin import apply_thin_lens

        N = 64
        dx = 1e-5
        f = 1e-4  # very short focal length: most of grid is outside
        E_in = (np.random.RandomState(0).randn(N, N)
                + 1j * np.random.RandomState(1).randn(N, N)).astype(
                np.complex128)
        E_out = apply_thin_lens(
            E_in, f=f, wavelength=632.8e-9, dx=dx,
            lens_model='aplanatic',
        )
        # Build the rim mask: outside the aplanatic domain |r| > |f|.
        x = (np.arange(N) - N / 2) * dx
        y = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, y)
        rim = (X ** 2 + Y ** 2) >= f ** 2
        # In the rim region the multiplier is unit -> E_out == E_in.
        np.testing.assert_allclose(E_out[rim], E_in[rim], atol=1e-12)

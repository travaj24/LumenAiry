"""Wave-5 audit fixes -- elements-misc cluster.

Covers AUDIT_V5_17_0_2026_07_01_DEEP findings:

* P2-15 (BEHAVIOR CHANGE): ``apply_waveplate`` slow-axis retardance
  sign flipped from ``exp(-i*phi)`` to ``exp(+i*phi)`` so the
  Jones-element family matches the library's rigorous Berreman/RCWA
  transmission Jones under the public ``exp(-i omega t)`` convention.
  Derivation: with carrier ``exp(-i omega t)`` the forward phasor is
  ``exp(+i k0 n z)``, so the slow axis (larger n, longer delay)
  accumulates POSITIVE relative phase; ``berreman_jones_1d`` on a
  uniaxial quarter-wave slab returns
  ``Jt = diag(e^{i k0 no d}, e^{i k0 ne d})`` (slow-rel-fast +pi/2).
* P2-07: ``makedammann2d`` far-field phase projector no longer relies
  on NumPy >= 2.0 complex ``np.sign`` semantics (NumPy 1.x returned
  ``sign(z.real)``, destroying the IFTA phase).
* P2-08: ``surface_sag_zernike_freeform`` now validates
  ``norm_radius > 0`` like its four P1-NEW-11 siblings (negative ->
  silent odd-term parity flip + defeated pupil mask; zero -> silent
  all-zero departure).
* P2-03: ``apply_real_lens_maslov_jax`` docstring no longer claims
  algorithmic parity with the NumPy ``apply_real_lens_maslov`` and
  documents the even-multiplicity (point-focus Gouy) blind spot of
  its det(J)-sign-flip counter.
"""

import numpy as np
import pytest

from lumenairy.elements import doe as _doe
from lumenairy.elements._lens_jax import apply_real_lens_maslov_jax
from lumenairy.elements.berreman import berreman_jones_1d
from lumenairy.elements.freeform import surface_sag_zernike_freeform
from lumenairy.elements.polarization import (
    apply_half_wave_plate,
    apply_quarter_wave_plate,
    apply_waveplate,
    create_circular_polarized,
    create_linear_polarized,
    stokes_parameters,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_NO, _NE = 1.5, 1.6
_LAM = 1.55e-6
_D_QWP = _LAM / (4.0 * (_NE - _NO))


def _berreman_qwp_jones(fast_axis_deg=0.0):
    """Transmission Jones of a rigorous uniaxial quarter-wave slab with
    the fast (ordinary, ``no``) axis at ``fast_axis_deg``."""
    eps = np.diag([_NO ** 2, _NE ** 2, _NO ** 2]).astype(complex)
    th = np.radians(fast_axis_deg)
    Rz = np.array([[np.cos(th), -np.sin(th), 0.0],
                   [np.sin(th), np.cos(th), 0.0],
                   [0.0, 0.0, 1.0]])
    eps_rot = Rz @ eps @ Rz.T
    _R, _T, _Jr, Jt = berreman_jones_1d(
        [(eps_rot, _D_QWP)], _NO, _NO, _LAM, angle=0.0)
    return Jt


def _s3_of(jf):
    S = stokes_parameters(jf)
    return float(np.mean(S['S3']) / np.mean(S['S0']))


# ==========================================================================
# P2-15 -- apply_waveplate sign matches the Berreman solver Jones
# ==========================================================================

class TestP2_15_WaveplateSignMatchesSolvers:

    def test_slow_axis_relative_phase_is_plus_pi_over_2(self):
        """Both families must give slow-rel-fast phase +pi/2 for a QWP
        (fast axis = x) under the public exp(-i omega t) convention."""
        Jt = _berreman_qwp_jones(0.0)
        rel_berreman = float(np.angle(Jt[1, 1] / Jt[0, 0]))
        assert abs(rel_berreman - np.pi / 2) < 1e-9, (
            f"Berreman QWP slab slow-rel-fast phase = "
            f"{rel_berreman/np.pi:+.6f} pi; expected +0.5 pi.")

        # Element family: pass x-pol and y-pol through the same device.
        scalar = np.ones((4, 4), dtype=complex)
        jf_x = apply_waveplate(
            create_linear_polarized(scalar, 2e-6, angle=0.0),
            np.pi / 2, angle=0.0)
        jf_y = apply_waveplate(
            create_linear_polarized(scalar, 2e-6, angle=np.pi / 2),
            np.pi / 2, angle=0.0)
        rel_element = float(np.angle(jf_y.Ey[0, 0] / jf_x.Ex[0, 0]))
        assert abs(rel_element - np.pi / 2) < 1e-12, (
            f"apply_waveplate slow-rel-fast phase = "
            f"{rel_element/np.pi:+.6f} pi; expected +0.5 pi to match "
            f"berreman_jones_1d (audit P2-15).  The pre-fix "
            f"exp(-i*retardance) gave -0.5 pi.")

    def test_qwp_at_45_handedness_matches_berreman(self):
        """Fast axis +45 deg on x-pol: both families must give the SAME
        S3 sign (-1) for the same physical device."""
        Jt = _berreman_qwp_jones(45.0)
        Eout = Jt @ np.array([1.0, 0.0], dtype=complex)
        s3_berreman = -2.0 * np.imag(Eout[0] * np.conj(Eout[1]))
        s3_berreman /= np.abs(Eout[0]) ** 2 + np.abs(Eout[1]) ** 2

        scalar = np.ones((4, 4), dtype=complex)
        jf = apply_quarter_wave_plate(
            create_linear_polarized(scalar, 2e-6, angle=0.0),
            angle=np.pi / 4)
        s3_element = _s3_of(jf)

        assert s3_berreman < -0.99  # rigorous slab: left-circular
        assert s3_element * s3_berreman > 0, (
            f"Circular handedness flips between element and solver "
            f"families: apply_waveplate S3 = {s3_element:+.3f} vs "
            f"Berreman S3 = {s3_berreman:+.3f} (audit P2-15).")
        assert abs(s3_element - s3_berreman) < 1e-9

    def test_qwp_at_minus_45_reproduces_create_right(self):
        """Under the solver-aligned sign, fast axis -45 deg is the
        recipe for 'right' (S3 = +1)."""
        scalar = np.ones((4, 4), dtype=complex)
        jf = apply_quarter_wave_plate(
            create_linear_polarized(scalar, 2e-6, angle=0.0),
            angle=-np.pi / 4)
        s3 = _s3_of(jf)
        s3_ref = _s3_of(create_circular_polarized(scalar, 2e-6, 'right'))
        assert s3 > 0.99 and s3_ref > 0.99
        assert abs(s3 - s3_ref) < 1e-12

    def test_half_wave_plate_unaffected(self):
        """HWP results are sign-independent (exp(+-i pi) = -1): x-pol
        through HWP at 22.5 deg -> 45 deg linear, S3 ~ 0."""
        scalar = np.ones((4, 4), dtype=complex)
        jf = apply_half_wave_plate(
            create_linear_polarized(scalar, 2e-6, angle=0.0),
            angle=np.pi / 8)
        ratio = abs(jf.Ey[0, 0]) / abs(jf.Ex[0, 0])
        assert abs(ratio - 1.0) < 1e-10
        assert abs(_s3_of(jf)) < 1e-10


# ==========================================================================
# P2-07 -- makedammann2d no longer depends on np.sign complex semantics
# ==========================================================================

class TestP2_07_DammannPhaseProjectorVersionIndependent:

    _KW = dict(periodx=61e-6, periody=61e-6, waveln=1.31e-6,
               itr=30, seed=1234, plot=False)

    @staticmethod
    def _legacy_sign(x, *args, **kwargs):
        """NumPy 1.x complex sign: sign(z.real), falling back to
        sign(z.imag) on the imaginary axis."""
        x = np.asarray(x)
        if np.iscomplexobj(x):
            re = np.where(x.real != 0, np.sign(x.real), np.sign(x.imag))
            return re.astype(x.dtype)
        return np.sign(x, *args, **kwargs)

    def test_output_invariant_under_numpy1_sign_semantics(self):
        """The IFTA must not call np.sign on the complex far field any
        more: swapping in the NumPy 1.x semantics must leave the design
        bit-identical.  Pre-fix this collapsed the design (uniformity
        0.97 -> 0.00 at itr=200)."""
        target = np.ones((4, 4))
        nf_ref, ff_ref, _ = _doe.makedammann2d(diforders=target, **self._KW)

        orig_np = _doe.np
        legacy_sign = self._legacy_sign

        class _ShimNp:
            def __getattr__(self, name):
                if name == 'sign':
                    return legacy_sign
                return getattr(orig_np, name)

        _doe.np = _ShimNp()
        try:
            nf_leg, ff_leg, _ = _doe.makedammann2d(
                diforders=target, **self._KW)
        finally:
            _doe.np = orig_np

        assert np.array_equal(nf_ref, nf_leg), (
            "makedammann2d output changed under NumPy 1.x complex-sign "
            "semantics -- the far-field phase projector still depends "
            "on np.sign(complex) (audit P2-07).")
        assert np.array_equal(ff_ref, ff_leg)

    def test_design_quality_is_sane(self):
        """Discriminating quality gate: with the correct z/|z| phase
        projector an 8x8 Dammann design converges to high uniformity;
        the NumPy-1.x sign degeneration gave uniformity ~0.02."""
        target = np.ones((8, 8))
        kw = dict(self._KW, itr=200)
        nf, _ff, _sz = _doe.makedammann2d(diforders=target, **kw)
        ff = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(nf)))
        inten = np.abs(ff) ** 2 / nf.size
        n = inten.shape[0]
        x0 = (n - 8) // 2
        orders = inten[x0:x0 + 8, x0:x0 + 8]
        uniformity = float(orders.min() / orders.max())
        assert uniformity > 0.5, (
            f"Dammann 8x8 design uniformity = {uniformity:.4f}; "
            f"expected > 0.5 (healthy IFTA).  The P2-07 legacy-sign "
            f"failure mode gave ~0.02.")


# ==========================================================================
# cell_pixels -- grid-native Dammann generation (lossless DOE tiling)
# ==========================================================================

class TestDammannCellPixelsGridNative:
    """``cell_pixels`` pins the unit-cell pixel count so the returned
    cell_pixel_size equals the propagation grid dx exactly and
    :func:`create_periodic_phase_mask` maps one cell pixel to one grid
    pixel -- no nearest-neighbour resample, no power scattered out of the
    design orders by sampling jitter."""

    _KW = dict(waveln=1.31e-6, itr=20, seed=7, plot=False)

    def test_int_sets_both_axes_and_exact_pixel_size(self):
        period = 61e-6
        nf, _ff, sz = _doe.makedammann2d(
            periodx=period, periody=period, diforders=np.ones((4, 4)),
            cell_pixels=40, **self._KW)
        assert nf.shape == (40, 40)
        # cell_pixel_size == period / cell_pixels EXACTLY (drives the 1:1 map)
        assert sz[0] == period / 40
        assert sz[1] == period / 40

    def test_tuple_sets_axes_independently(self):
        nf, _ff, sz = _doe.makedammann2d(
            periodx=61e-6, periody=80e-6, diforders=np.ones((4, 4)),
            cell_pixels=(40, 50), **self._KW)
        assert nf.shape == (40, 50)
        assert sz[0] == 61e-6 / 40
        assert sz[1] == 80e-6 / 50

    def test_grid_native_tiling_is_lossless(self):
        """With cell_pixels = round(period/dx) the tiled mask is an exact
        integer tiling of the cell (each cell pixel reproduced verbatim),
        so no interpolation error / order scatter is introduced."""
        period = 61e-6
        n_per = 40
        dx = period / n_per                      # grid dx
        nf, _ff, sz = _doe.makedammann2d(
            periodx=period, periody=period, diforders=np.ones((4, 4)),
            cell_pixels=n_per, **self._KW)
        assert sz[0] == dx                        # exact -> 1:1 mapping
        phase_cell = np.angle(nf)
        N = 3 * n_per                             # 3 full periods
        mask = _doe.create_periodic_phase_mask(N, dx, phase_cell, sz[0])
        # Reconstruct the exact expected tiling and compare bit-for-bit.
        idx = np.mod(np.arange(N) - N // 2, n_per)
        expected = np.exp(1j * phase_cell[np.ix_(idx, idx)])
        assert np.array_equal(mask, expected), (
            "grid-native tiling is not an exact 1:1 map -- "
            "create_periodic_phase_mask introduced a resample.")
        # Phase-only: unit amplitude everywhere (no absorption).
        assert np.allclose(np.abs(mask), 1.0)

    def test_odd_cell_pixels_rejected(self):
        with pytest.raises(ValueError, match="even integer"):
            _doe.makedammann2d(diforders=np.ones((4, 4)),
                               cell_pixels=41, **self._KW)

    def test_cell_smaller_than_orders_rejected(self):
        with pytest.raises(ValueError, match="smaller than the target"):
            _doe.makedammann2d(diforders=np.ones((8, 8)),
                               cell_pixels=6, **self._KW)


# ==========================================================================
# P2-08 -- surface_sag_zernike_freeform norm_radius validation
# ==========================================================================

class TestP2_08_ZernikeFreeformNormRadiusGuard:

    _X, _Y = np.meshgrid(np.linspace(-0.04, 0.04, 21),
                         np.linspace(-0.04, 0.04, 21))
    _COEFFS = {7: 1e-6}  # vertical coma (odd-n -> parity-sensitive)

    @pytest.mark.parametrize('bad', [-0.05, 0.0, -1e-300,
                                     np.nan, np.inf, -np.inf])
    def test_non_positive_or_non_finite_norm_radius_raises(self, bad):
        with pytest.raises(ValueError, match='norm_radius'):
            surface_sag_zernike_freeform(
                self._X, self._Y, R=np.inf, conic=0.0,
                zernike_coeffs=self._COEFFS, norm_radius=bad)

    def test_positive_norm_radius_still_works(self):
        sag = surface_sag_zernike_freeform(
            self._X, self._Y, R=np.inf, conic=0.0,
            zernike_coeffs=self._COEFFS, norm_radius=0.05)
        assert np.all(np.isfinite(sag))
        assert np.max(np.abs(sag)) > 0.0

    def test_no_coeffs_does_not_bypass_guard(self):
        """The guard must fire at entry even when zernike_coeffs is
        empty, matching the sibling P1-NEW-11 guards' fail-at-callsite
        contract."""
        with pytest.raises(ValueError, match='norm_radius'):
            surface_sag_zernike_freeform(
                self._X, self._Y, R=0.1, conic=0.0,
                zernike_coeffs=None, norm_radius=-0.05)


# ==========================================================================
# P2-03 -- apply_real_lens_maslov_jax docstring honesty
# ==========================================================================

class TestP2_03_MaslovJaxDocstringHonest:

    def test_no_false_parity_claim(self):
        doc = apply_real_lens_maslov_jax.__doc__
        assert doc is not None
        assert 'same definition the NumPy' not in doc, (
            "apply_real_lens_maslov_jax docstring still claims "
            "algorithmic parity with the NumPy apply_real_lens_maslov "
            "(audit P2-03): the NumPy path is a stationary-phase "
            "integral with Hessian-signature caustic phase and has no "
            "radial sign-flip scan.")

    def test_even_multiplicity_limitation_documented(self):
        doc = apply_real_lens_maslov_jax.__doc__
        assert 'even-multiplicity' in doc.lower()
        assert 'point focus' in doc.lower()
        assert 'NOT' in doc

    def test_numpy_maslov_really_has_no_radial_scan(self):
        """Guard the premise: if someone later ports the sign-flip scan
        into the NumPy Maslov (making the parity claim true again),
        this pin should be revisited rather than silently rot."""
        import inspect

        from lumenairy.elements import lenses_maslov
        src = inspect.getsource(lenses_maslov)
        assert 'fori_loop' not in src
        assert 'maslov_count' not in src
        # Its caustic phase is the Hessian signature factor.
        assert '(np.pi / 4.0)' in src or 'pi / 4' in src

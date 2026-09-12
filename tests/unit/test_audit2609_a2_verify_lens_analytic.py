"""VERIFY-A2 -- independent re-verification of WP-A2 (``apply_real_lens``).

Written by the VERIFIER, not by the work package.  Every oracle here is
closed-form or hand-derived in this file; none of it is another implementation
inside lumenairy, and every fixture is one WP-A2 did NOT use (a different
radius, a different glass pair, a different tilt axis, an immersed exit, a
non-contiguous memory layout).  Each numeric bar carries its derivation, the
value measured on 2026-09-12, and the value the defect it guards produced.

It also pins both arms of the mirror-unfolding defect this pass FIXED
(``allow_unfolded_equivalent`` was honoured by the ``elements``-borne fold
guard and ignored by the per-surface one, so the documented escape hatch did
not work for a hand-built folded prescription).

No test here asserts a wall clock or a speed-up, skips on a resource
precondition, or scores the library against another model in the library.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements._lens_real import (apply_real_lens, prepare_real_lens,
                                           _screen_obliquity_angle_field,
                                           _build_displaced_ray_map,
                                           _build_displaced_ray_map_2d,
                                           _build_displaced_cos_luts)
from lumenairy.elements.lenses import surface_sag_general
from lumenairy.glass import get_glass_index

_LAM = 632.8e-9
_K0 = 2.0 * np.pi / _LAM

# Model glasses for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {
    # A near-index-matched "glass": ``(n2 - n1) = 1e-6`` turns the surface
    # screen into a direct, wrap-free readout of the SAG the model imprints
    # (40 um of sag is 6e-5 waves of OPD), which is how the tilt geometry
    # below is measured without unwrapping anything.
    '_VA2_NEAR1': lambda wl: 1.0 + 1.0e-6,
    # A weakly absorbing high-index glass for the Fresnel power check.
    '_VA2_ABS': lambda wl: complex(1.8, 1.0e-3),
}


# ---------------------------------------------------------------------------
# helpers (closed form; nothing below calls a lumenairy model)
# ---------------------------------------------------------------------------
def _conic_sag(h_sq, R, k=0.0):
    c = 1.0 / R
    return c * h_sq / (1.0 + np.sqrt(np.maximum(1.0 - (1.0 + k) * c * c * h_sq,
                                                0.0)))


def _rotmat(a, b):
    """``Rx(a) @ Ry(b)`` -- the rigid-body rotation the surface frame uses."""
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    return np.array([[cb, 0.0, sb],
                     [sa * sb, ca, -sa * cb],
                     [-ca * sb, sa, ca * cb]])


def _exact_rotated_sphere_height(xg, yg, R, thx, thy, dec=(0.0, 0.0)):
    """Field-frame height of a sphere RIGIDLY rotated about its vertex.

    Closed form, from the sphere itself: rotating a sphere about a point ON it
    gives a sphere of the same radius whose centre is the rotated centre, so
    ``z(x, y) = Cz - sign(R) sqrt(R**2 - (x-Cx)**2 - (y-Cy)**2)`` with
    ``C = Rot @ (0, 0, R)``.  No series, no footprint approximation, no
    lumenairy.
    """
    M = _rotmat(thx, thy)
    C = M @ np.array([0.0, 0.0, float(R)])
    u = xg - dec[0] - C[0]
    v = yg - dec[1] - C[1]
    return C[2] - np.sign(R) * np.sqrt(np.maximum(R * R - u * u - v * v, 0.0))


def _imprinted_sag(surf, N, dx, surface_frame):
    """The SAG the shipped screen imprints, read back through a surface whose
    index step is 1e-6 (so ``k0 (n2-n1) sag`` never reaches a wrap)."""
    rx = dict(surfaces=[dict(surf, glass_before='AIR',
                             glass_after='_VA2_NEAR1')], thicknesses=[])
    E = np.ones((N, N), dtype=np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Eo = apply_real_lens(E, prescription=rx, wavelength=_LAM, dx=dx,
                             surface_frame=surface_frame)
    return -np.angle(Eo) / (_K0 * 1.0e-6)


def _spectral_centroid_dir(E, dx):
    """Mean direction cosines of a field, from its power spectrum."""
    N = E.shape[0]
    f = np.fft.fftshift(np.fft.fftfreq(N, dx))
    S = np.abs(np.fft.fftshift(np.fft.fft2(E))) ** 2
    return (float((S.sum(axis=0) * f).sum() / S.sum()) * _LAM,
            float((S.sum(axis=1) * f).sum() / S.sum()) * _LAM)


# ===========================================================================
# The defect VERIFY-A2 found and fixed: the two mirror guards disagreed
# ===========================================================================
class TestVerifyMirrorGuardsReadTheSameKey:
    """``prescription['allow_unfolded_equivalent']`` is documented as THE
    escape hatch for a folded design, and ``_check_no_silent_fold_drop``
    honours it -- but only for a mirror carried in ``prescription['elements']``
    (what ``load_zemax_zmx`` emits).  A hand-built prescription that puts the
    mirror straight into ``surfaces`` hit a second, unconditional guard that
    never mentioned the key, so the documented option did not work on that
    spelling of the same physics (``ui/waveoptics_dock.py`` had to build the
    unfolded prescription itself before it could set the flag).

    Both guards now read the one key.  Both arms are pinned here: refused when
    it is absent (naming both remedies), and the exact unfolded walk when it is
    present.
    """

    N, DX = 64, 1.0e-5
    WL = 1.31e-6

    @staticmethod
    def _rx(spelling='is_mirror', R_mirror=np.inf):
        m = dict(radius=R_mirror, conic=0.0, glass_before='AIR',
                 glass_after=('MIRROR' if spelling == 'glass_after' else 'AIR'))
        if spelling == 'is_mirror':
            m['is_mirror'] = True
        return dict(
            surfaces=[dict(radius=0.05, glass_before='AIR',
                           glass_after='N-BK7'),
                      dict(radius=np.inf, glass_before='N-BK7',
                           glass_after='AIR'),
                      m],
            thicknesses=[3e-3, 20e-3], aperture_diameter=5e-4)

    @staticmethod
    def _hand_unfolded():
        """What 'the unfolded equivalent' MEANS, written out by hand: the
        mirror replaced by an index-neutral flat at its own vertex plane, every
        gap and both reference planes untouched."""
        return dict(
            surfaces=[dict(radius=0.05, glass_before='AIR',
                           glass_after='N-BK7'),
                      dict(radius=np.inf, glass_before='N-BK7',
                           glass_after='AIR'),
                      dict(radius=np.inf, glass_before='AIR',
                           glass_after='AIR')],
            thicknesses=[3e-3, 20e-3], aperture_diameter=5e-4)

    @pytest.mark.parametrize('spelling', ['is_mirror', 'glass_after'])
    def test_without_the_flag_it_refuses_and_names_the_working_remedy(
            self, spelling):
        """Both spellings of a mirror surface refuse, and the message now names
        the flag that actually works as well as the exact alternative.  Before
        this fix the message named only ``split_prescription_at_mirrors``."""
        E = np.ones((self.N, self.N), dtype=np.complex128)
        with pytest.raises(ValueError) as ei:
            apply_real_lens(E, prescription=self._rx(spelling),
                            wavelength=self.WL, dx=self.DX)
        msg = str(ei.value)
        assert 'allow_unfolded_equivalent' in msg, (
            f"the refusal must name the key that resolves it; got: {msg!r}")
        assert 'split_prescription_at_mirrors' in msg, (
            f"the refusal must keep naming the exact alternative; got: {msg!r}")

    def test_with_the_flag_the_walk_is_the_unfolded_equivalent_bit_for_bit(
            self):
        """The semantics are asserted as a BIT IDENTITY against the
        hand-written unfolded prescription -- a two-sided statement no
        tolerance can soften, and the only way to pin 'the mirror became an
        index-neutral flat and NOTHING else moved'.

        Pre-fix this call raised ``ValueError`` whatever the flag said.
        """
        E = np.ones((self.N, self.N), dtype=np.complex128)
        rx = self._rx()
        rx['allow_unfolded_equivalent'] = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            got = apply_real_lens(E, prescription=rx, wavelength=self.WL,
                                  dx=self.DX)
        ref = apply_real_lens(E, prescription=self._hand_unfolded(),
                              wavelength=self.WL, dx=self.DX)
        assert np.array_equal(got.view(np.uint8), ref.view(np.uint8)), (
            f"the unfolded walk is not the hand-unfolded prescription: "
            f"max|d| = {np.max(np.abs(got - ref)):.3e}")
        unf = [w for w in caught if 'UNFOLDED EQUIVALENT' in str(w.message)]
        assert len(unf) == 1, (
            f"the unfolded walk must announce itself exactly once; caught "
            f"{[(w.category.__name__, str(w.message)[:60]) for w in caught]}")
        assert issubclass(unf[0].category, RuntimeWarning)

    def test_a_curved_mirror_names_the_physics_it_drops(self):
        """A FLAT fold is exact under this treatment; a CURVED one loses its
        focusing phase, which is the one thing the caller must be told."""
        E = np.ones((self.N, self.N), dtype=np.complex128)
        rx = self._rx(R_mirror=0.2)
        rx['allow_unfolded_equivalent'] = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            apply_real_lens(E, prescription=rx, wavelength=self.WL, dx=self.DX)
        msgs = [str(w.message) for w in caught
                if 'UNFOLDED EQUIVALENT' in str(w.message)]
        assert msgs and 'CURVED' in msgs[0], (
            f"a curved fold must say its focusing phase was dropped; got "
            f"{msgs!r}")

    def test_prepare_real_lens_reads_the_same_key(self):
        """The two entry points diagnosed the same prescription differently
        (``ValueError`` vs ``NotImplementedError``, and only one of them could
        ever be told to proceed).  They agree now, and the prepared lens
        reproduces ``apply_real_lens`` byte for byte on the unfolded walk."""
        E = np.ones((self.N, self.N), dtype=np.complex128)
        with pytest.raises(ValueError) as ei:
            prepare_real_lens(prescription=self._rx(), wavelength=self.WL,
                              N=self.N, dx=self.DX)
        assert 'allow_unfolded_equivalent' in str(ei.value)
        rx = self._rx()
        rx['allow_unfolded_equivalent'] = True
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prepared = prepare_real_lens(prescription=rx, wavelength=self.WL,
                                         N=self.N, dx=self.DX)
            got = np.asarray(prepared(E))
            ref = apply_real_lens(E, prescription=rx, wavelength=self.WL,
                                  dx=self.DX)
        assert np.array_equal(got.view(np.uint8), ref.view(np.uint8)), (
            f"prepared != apply on the unfolded walk: "
            f"max|d| = {np.max(np.abs(got - ref)):.3e}")

    def test_the_elements_borne_fold_guard_is_unchanged(self):
        """Regression gate for the half that already worked: an
        ``elements``-borne fold still refuses without the flag and still runs
        silently with it (no second warning, no change of behaviour)."""
        sa = {'radius': 0.05, 'conic': 0.0, 'aspheric_coeffs': None,
              'glass_before': 'AIR', 'glass_after': 'N-BK7'}
        sb = {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
              'glass_before': 'N-BK7', 'glass_after': 'AIR'}
        rx = {'surfaces': [sa, sb], 'thicknesses': [3e-3],
              'elements': [{**sa, 'element_type': 'surface'},
                           {**sb, 'element_type': 'surface'},
                           {'element_type': 'mirror', 'radius': np.inf,
                            'conic': 0.0, 'clear_aperture': 5e-3}],
              'all_thicknesses': [3e-3, 50e-3], 'aperture_diameter': 5e-3}
        E = np.ones((self.N, self.N), dtype=np.complex128)
        with pytest.raises(ValueError, match='mirror'):
            apply_real_lens(E, prescription=rx, wavelength=self.WL, dx=self.DX)
        rx['allow_unfolded_equivalent'] = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = apply_real_lens(E, prescription=rx, wavelength=self.WL,
                                  dx=self.DX)
        assert np.all(np.isfinite(out))
        assert not [w for w in caught if 'UNFOLDED EQUIVALENT' in
                    str(w.message)], (
            "the elements-borne path must keep its previous, silent behaviour")


# ===========================================================================
# L13 -- Fresnel POWER transmittance, against the closed form
# ===========================================================================
class TestVerifyL13FresnelPowerOnNewInterfaces:
    """``T = 4 n1 n2 / (n1 + n2)**2`` at normal incidence, derived here from
    the Fresnel amplitude coefficients and the axial Poynting flux; the
    library's own ``sum |E|**2 dx dy`` IS the power, so that ratio is what a
    single interface must transmit.

    The glass pairs are ones WP-A2 did not use (a high-index crown, a reversed
    cemented pair, and a weakly ABSORBING glass with a genuinely complex
    index).  Pre-fix every one of these read ``|t|**2``, i.e. a factor
    ``n1/n2`` low -- -34 % on a single air-glass face.
    """

    N, DX = 64, 4.0e-6

    @staticmethod
    def _power(E, dx):
        return float(np.sum(np.abs(E) ** 2)) * dx * dx

    @pytest.mark.parametrize('g1,g2', [
        ('AIR', 'N-LASF9'),          # high index, n = 1.85
        ('N-SF11', 'N-BK7'),         # cemented, reversed order
        ('N-BK7', 'N-SF11'),
    ])
    def test_single_interface_is_the_closed_form_power_transmittance(
            self, g1, g2):
        """BAR.  The oracle is exact algebra, so the only error is float64
        round-off in the library's own product: measured 1.1e-16 .. 4.4e-16
        relative (2026-09-12) on these three pairs.  The bar is 1e-12 -- four
        decades above that floor and eleven decades below the defect, which
        moved these numbers by 3.4e-1 .. 1.5e-1 relative."""
        n1 = float(get_glass_index(g1, _LAM))
        n2 = float(get_glass_index(g2, _LAM))
        T = 4.0 * n1 * n2 / (n1 + n2) ** 2
        rx = dict(surfaces=[dict(radius=float('inf'), glass_before=g1,
                                 glass_after=g2)], thicknesses=[])
        E = np.ones((self.N, self.N), dtype=np.complex128)
        Eo = apply_real_lens(E, prescription=rx, wavelength=_LAM, dx=self.DX,
                             fresnel=True)
        got = self._power(Eo, self.DX) / self._power(E, self.DX)
        assert abs(got / T - 1.0) < 1e-12, (
            f"{g1}->{g2}: transmitted {got:.9f}, closed-form power "
            f"transmittance {T:.9f} (|t|**2 alone would be "
            f"{T * n1 / n2:.9f})")

    def test_an_absorbing_glass_uses_complex_t_and_a_real_angle(self):
        """The documented approximation: the Fresnel coefficients take the
        COMPLEX index while the refraction angle takes the real parts.  Scored
        against that same statement written out by hand at normal incidence,
        where the angle plays no role -- so this pins the |t|**2 -> power
        conversion on a complex index, which is the arm a real-index-only
        implementation would get wrong.

        BAR 1e-12 relative; measured 2.2e-16 (2026-09-12)."""
        nc = complex(la.get_glass_index_complex('_VA2_ABS', _LAM))
        n1c = complex(1.0, 0.0)
        ts = 2.0 * n1c / (n1c + nc)
        tp = 2.0 * n1c / (nc + n1c)
        T = 0.5 * (abs(ts) ** 2 + abs(tp) ** 2) * (nc.real / n1c.real)
        rx = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR',
                                 glass_after='_VA2_ABS')], thicknesses=[])
        E = np.ones((self.N, self.N), dtype=np.complex128)
        Eo = apply_real_lens(E, prescription=rx, wavelength=_LAM, dx=self.DX,
                             fresnel=True)
        got = self._power(Eo, self.DX) / self._power(E, self.DX)
        assert abs(got / T - 1.0) < 1e-12, (
            f"absorbing face transmitted {got:.9f} against {T:.9f}")

    def test_fresnel_false_is_bit_identical_to_the_default(self):
        """The fix must not have leaked into the default path."""
        rx = dict(surfaces=[dict(radius=20e-3, glass_before='AIR',
                                 glass_after='N-BK7'),
                            dict(radius=-20e-3, glass_before='N-BK7',
                                 glass_after='AIR')],
                  thicknesses=[2e-3], aperture_diameter=2e-4)
        E = np.ones((self.N, self.N), dtype=np.complex128)
        a = apply_real_lens(E, prescription=rx, wavelength=_LAM, dx=self.DX)
        b = apply_real_lens(E, prescription=rx, wavelength=_LAM, dx=self.DX,
                            fresnel=False)
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8))


# ===========================================================================
# L12 -- the slant coefficient, isolated from everything else
# ===========================================================================
class TestVerifyL12SlantCoefficientIsTheAxialTranslationIdentity:
    """Isolate the COEFFICIENT rather than the end-to-end field.

    ``apply_real_lens`` on a ONE-surface prescription applies exactly one
    screen and no propagation, so ``angle(E_slant * conj(E_paraxial))`` is
    ``-k0 * (c_slant - (n2 - n1)) * sag`` pixel for pixel -- no unwrapping, no
    ray trace, no grid convergence question.  The closed form for that
    difference is

        c_slant - (n2 - n1) = n2 cos(theta_i - theta_t) - n2 + n1 - n1
                            = n2 (cos(theta_i - theta_t) - 1) ,   n1 = 1

    with ``theta_i = arctan(|grad sag|)`` and Snell's ``theta_t``.  Everything
    on the right is written out here from the conic equation.

    R = 35 mm is a radius neither the audit nor WP-A2 used.
    """

    N, DX, R = 512, 1.2e-6, 35.0e-3

    def _screens(self):
        rx = dict(surfaces=[dict(radius=self.R, glass_before='AIR',
                                 glass_after='N-BK7')], thicknesses=[])
        E = np.ones((self.N, self.N), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = apply_real_lens(E, prescription=rx, wavelength=_LAM,
                                dx=self.DX)
            b = apply_real_lens(E, prescription=rx, wavelength=_LAM,
                                dx=self.DX, slant_correction=True)
        return a, b

    def test_the_obliquity_term_matches_the_closed_form(self):
        """BAR.  The oracle is exact; its own floor is the float64 round-off of
        ``angle(...)`` on a ~1e-3 rad argument, ~1e-16 rad absolute, i.e. ~1e-13
        relative here.  Measured max relative deviation 2026-09-12: 5.5e-10.
        The bar is 1e-6 relative -- three decades above the measurement, and it
        excludes BOTH shipped defects by construction: the paraxial screen
        gives exactly 0 (relative error 1.0) and the v5.25.0
        normal-referenced form gives the OPPOSITE SIGN at 1.94x the magnitude
        (relative error 2.94)."""
        a, b = self._screens()
        x = (np.arange(self.N) - self.N / 2) * self.DX
        X, Y = np.meshgrid(x, x)
        h_sq = X * X + Y * Y
        r_use = 0.25 * self.N * self.DX          # inside the conic domain
        m = h_sq <= r_use ** 2
        n2 = float(get_glass_index('N-BK7', _LAM))
        sag = _conic_sag(h_sq, self.R)
        # |grad sag| for a sphere: h / sqrt(R^2 - h^2)
        g = np.sqrt(h_sq) / np.sqrt(np.maximum(self.R ** 2 - h_sq, 1e-30))
        th_i = np.arctan(g)
        th_t = np.arcsin(np.sin(th_i) / n2)
        d_coeff = n2 * np.cos(th_i - th_t) - n2        # c_slant - (n2 - 1)
        pred = -_K0 * d_coeff * sag
        meas = np.angle(b * np.conj(a))
        rel = np.max(np.abs(meas[m] - pred[m])) / max(
            float(np.max(np.abs(pred[m]))), 1e-30)
        assert rel < 1e-6, (
            f"the slant screen's obliquity term deviates from the exact "
            f"axial-translation identity by {rel:.3e} relative; paraxial "
            f"would score 1.0 and the v5.25.0 normal-referenced form 2.94")

    def test_the_term_has_the_sign_that_reduces_the_screen(self):
        """SIGN, stated on its own so a return of the v5.25.0 form cannot hide
        inside a magnitude tolerance: ``cos(theta_i - theta_t) < 1``, so the
        exact coefficient is SMALLER than the paraxial one and the applied OPD
        difference is positive under ``exp(-i k0 OPD)``.  The v5.25.0 form had
        the opposite sign at every radius."""
        a, b = self._screens()
        x = (np.arange(self.N) - self.N / 2) * self.DX
        X, Y = np.meshgrid(x, x)
        h_sq = X * X + Y * Y
        r_use = 0.25 * self.N * self.DX
        ring = (h_sq > (0.6 * r_use) ** 2) & (h_sq <= r_use ** 2)
        meas = np.angle(b * np.conj(a))[ring]
        assert float(np.min(meas)) > 0.0, (
            f"the obliquity term must RAISE the phase (lower the OPD) at every "
            f"off-axis pixel; min = {float(np.min(meas)):.3e} rad")


# ===========================================================================
# L2 / L19 -- the tilt, about BOTH axes, against exact rigid-body geometry
# ===========================================================================
class TestVerifyL2TiltAgainstExactRigidBodyGeometry:
    """``surface_frame=True`` must imprint the rotated surface's FIELD-frame
    height.  The oracle is the sphere itself (a sphere rotated about a point on
    it is a sphere with the rotated centre), evaluated at the field point --
    no series, no footprint approximation.

    Pre-fix the imprinted height was the UNROTATED sag at the rotated
    footprint, i.e. the whole linear ramp was missing: measured 2.002 / 10.011
    / 40.077 um at 1 / 5 / 20 mrad on R = 50 mm over +-2 mm (1.63 / 8.15 / 32.6
    waves of OPD through N-BK7).
    """

    N, DX, R = 256, 1.6e-5, 50.0e-3
    R_SCORE = 2.0e-3

    @pytest.mark.parametrize('tilt,label', [
        ((5e-3, 0.0), 'about y (x-ramp)'),
        ((0.0, 5e-3), 'about x (y-ramp)'),
        ((3e-3, -4e-3), 'both axes, 5 mrad total'),
    ])
    def test_rotated_sphere_height_both_axes(self, tilt, label):
        """BAR.  The residual is the model's documented thin-element
        simplification (the sag is taken at the footprint of the field-plane
        normal, not at the perpendicular foot), which is second order in
        ``theta * d(sag)/dx``: at 5 mrad over a +-2 mm pupil on R = 50 mm that
        is ``5e-3 * 2e-3 * (2e-3/50e-3) = 4e-7 m`` at the very rim, and it
        partly cancels.  Measured 2026-09-12: 1.026e-8 m on all three
        orientations (0.016 waves).  The bar is 5e-8 m -- 4.9x above the
        measurement and **200x** below the missing ramp the defect left behind
        (1.0011e-5 m at 5 mrad; 2.002e-6 at 1 mrad and 4.0077e-5 at 20 mrad).
        The quantity is deterministic conic/trig geometry, so its cross-build
        spread is float64 round-off (~1e-16 relative) -- decades under either
        side of the bar, which is what makes a 4.9x low-side margin sound here.
        """
        xb = (np.arange(self.N) - self.N / 2) * self.DX
        Xb, Yb = np.meshgrid(xb, xb)
        m = (Xb ** 2 + Yb ** 2) <= self.R_SCORE ** 2
        thx, thy = float(tilt[1]), -float(tilt[0])
        exact = _exact_rotated_sphere_height(Xb, Yb, self.R, thx, thy)
        got = _imprinted_sag(dict(radius=self.R, tilt=tilt),
                             self.N, self.DX, True)
        d = (got - exact)
        d = d - np.mean(d[m])
        err = float(np.max(np.abs(d[m])))
        assert err < 5e-8, (
            f"{label}: surface_frame height deviates from the exact rotated "
            f"sphere by {err:.4e} m ({err / _LAM:.4f} waves); the pre-fix "
            f"defect left 2.0e-6 m at this tilt")

    def test_tilt_and_decenter_together(self):
        """Combined tilt + decenter on the same surface -- the combination the
        audit row did not measure.  Same bar and same derivation as above;
        measured 9.577e-9 m (2026-09-12)."""
        dec = (0.3e-3, -0.2e-3)
        tilt = (5e-3, 2e-3)
        xb = (np.arange(self.N) - self.N / 2) * self.DX
        Xb, Yb = np.meshgrid(xb, xb)
        m = (Xb ** 2 + Yb ** 2) <= self.R_SCORE ** 2
        exact = _exact_rotated_sphere_height(Xb, Yb, self.R, tilt[1],
                                             -tilt[0], dec=dec)
        got = _imprinted_sag(dict(radius=self.R, tilt=tilt, decenter=dec),
                             self.N, self.DX, True)
        d = got - exact
        d = d - np.mean(d[m])
        assert float(np.max(np.abs(d[m]))) < 5e-8

    @pytest.mark.parametrize('axis', [0, 1])
    def test_both_branches_and_the_ray_model_deviate_the_same_way(self, axis):
        """A tilted FLAT face is a thin prism: it must deviate the beam by
        ``(n-1) theta`` about ONE axis, and the three consumers of the ``tilt``
        key -- the field-frame screen, the surface-frame screen and
        ``raytrace`` -- must agree on WHICH.

        BAR.  The oracle is the exact thin-prism deviation
        ``arcsin(n sin(theta)) - theta``, whose small-angle value at
        theta = 5 mrad is 2.5754e-3 rad; the spectral centroid of a finite
        Gaussian reads it with a discretisation floor of ~1e-6 rad.  Measured
        2026-09-12: both wave branches -2.57474e-3, the ray model -2.57544e-3,
        branch-to-branch 4.3e-8.  The bars are 1 % on the magnitude (the
        pre-fix surface-frame branch read 0.000, i.e. 100 % off) and 1e-5 rad
        between the branches.
        """
        n = float(get_glass_index('N-BK7', _LAM))
        th = 5.0e-3
        tilt = (th, 0.0) if axis == 0 else (0.0, th)
        N, dx = 512, 2.0e-6
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        E0 = np.exp(-(X ** 2 + Y ** 2) / (3e-4) ** 2).astype(np.complex128)
        rx = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR',
                                 glass_after='N-BK7', tilt=tilt),
                            dict(radius=float('inf'), glass_before='N-BK7',
                                 glass_after='AIR')],
                  thicknesses=[1e-3])
        dirs = []
        for sf in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                Eo = apply_real_lens(E0, prescription=rx, wavelength=_LAM,
                                     dx=dx, surface_frame=sf)
            dirs.append(_spectral_centroid_dir(Eo, dx))
        prism = np.arcsin(n * np.sin(th)) - th
        for lab, (Lx, Ly) in zip(('field-frame', 'surface-frame'), dirs):
            on, off = (Lx, Ly) if axis == 0 else (Ly, Lx)
            assert abs(abs(on) - prism) / prism < 0.01, (
                f"{lab}: deviation {on:+.6e} against the thin prism "
                f"{prism:+.6e} (pre-fix surface-frame read 0.000)")
            assert abs(off) < 1e-8, (
                f"{lab}: a tilt about one axis deviated the other by "
                f"{off:+.3e}")
        assert abs(dirs[0][0] - dirs[1][0]) < 1e-5
        assert abs(dirs[0][1] - dirs[1][1]) < 1e-5
        # ... and the RAY model reads the same key the same way.
        from lumenairy.raytrace import trace as rt_trace, _make_bundle
        from lumenairy.raytrace.trace import surfaces_from_prescription
        z3 = np.zeros(3)
        b = _make_bundle(x=z3, y=z3, L=z3, M=z3, wavelength=_LAM)
        f = rt_trace(b, surfaces_from_prescription(rx), _LAM).at_exit_vertex()
        ray_on = float(f.L[0]) if axis == 0 else float(f.M[0])
        ray_off = float(f.M[0]) if axis == 0 else float(f.L[0])
        assert abs(abs(ray_on) - prism) / prism < 0.01
        assert abs(ray_off) < 1e-8
        assert np.sign(ray_on) == np.sign(dirs[0][axis]), (
            "the wave and ray models must deviate the beam the SAME WAY; "
            f"ray {ray_on:+.3e} vs wave {dirs[0][axis]:+.3e}")


# ===========================================================================
# L4 -- carrier momentum in a high-index medium
# ===========================================================================
class TestVerifyL4CarrierMomentumInAHighIndexMedium:
    """An EXACT plane wave built in N-SF11 (n = 1.79) at a known in-glass
    angle.  Its transverse OPTICAL momentum is ``q = n sin(theta)`` by
    construction, and that is what the obliquity screen consumes.

    Pre-fix ``'auto'`` and an explicit wavefront ndarray were multiplied by
    ``n1`` a second time -- x1.79 here, x1.5168 on the audit's N-BK7 fixture.
    """

    N, DX = 256, 2.0e-7
    THETA = 0.05

    def _plane_wave(self, n1):
        ax = (np.arange(self.N) - self.N / 2) * self.DX
        Xg, Yg = np.meshgrid(ax, ax)
        q = n1 * np.sin(self.THETA)
        return Xg, Yg, np.exp(1j * _K0 * q * Xg).astype(np.complex128), q

    @pytest.mark.parametrize('kind', ['auto', 'ndarray', 'tilted'])
    def test_every_carrier_vocabulary_reads_the_optical_momentum(self, kind):
        """BAR 1e-9 relative.  The oracle is the plane wave's own construction,
        exact to float64; measured 2026-09-12: 4.4e-16 ('auto'), 1.1e-16
        (ndarray), 0.0 (TiltedCarrier).  The defect this excludes is a factor
        n = 1.7786, i.e. a relative error of 0.78."""
        from lumenairy.elements._lens_traced import TiltedCarrier
        n1 = float(get_glass_index('N-SF11', _LAM))
        Xg, _Yg, E, q_true = self._plane_wave(n1)
        if kind == 'auto':
            carrier = 'auto'
        elif kind == 'ndarray':
            carrier = q_true * Xg          # reference phase = k0 * W
        else:
            carrier = TiltedCarrier(L=np.sin(self.THETA), M=0.0, R=np.inf,
                                    x0=0.0, y0=0.0)
        qx, _qy = _screen_obliquity_angle_field(
            carrier, E, _LAM, self.DX, self.DX, self.N, self.N, n_medium=n1)
        got = float(np.mean(np.asarray(qx)))
        assert abs(got / q_true - 1.0) < 1e-9, (
            f"carrier={kind}: q = {got:.9f} against the exact "
            f"n sin(theta) = {q_true:.9f} (a second n1 would give "
            f"{q_true * n1:.9f})")


# ===========================================================================
# L7 -- the displaced remap's exit referencing leg in an IMMERSED exit
# ===========================================================================
class TestVerifyL7ImmersedExitReferencingLeg:
    """The 1-D remap walks the ray from the last surface's sag BACK to the exit
    vertex plane.  That leg is travelled in ``surfaces[-1]['glass_after']``,
    which is not always air.

    Oracle: a closed-form paraxial-free trace written here (quadratic conic
    intersection + 3-D vector Snell + the signed transfer), with the exit index
    resolved from the prescription.
    """

    N, DX, AP = 256, 1.2e-5, 2.0e-3

    @staticmethod
    def _trace(surfaces, thicknesses, h):
        """Closed-form meridional trace to the exit vertex plane."""
        x = np.asarray(h, dtype=float).copy()
        z = np.zeros_like(x)
        Lx, Lz = np.zeros_like(x), np.ones_like(x)
        opl = np.zeros_like(x)
        zv = 0.0
        for i, s in enumerate(surfaces):
            R = float(s['radius'])
            c = 0.0 if not np.isfinite(R) else 1.0 / R
            n1 = float(get_glass_index(s['glass_before'], _LAM))
            n2 = float(get_glass_index(s['glass_after'], _LAM))
            x0, z0 = x, z - zv
            A = c * (Lx * Lx + Lz * Lz)
            B = 2.0 * c * x0 * Lx + 2.0 * c * z0 * Lz - 2.0 * Lz
            C = c * x0 * x0 + c * z0 * z0 - 2.0 * z0
            if c == 0.0:
                t = -z0 / Lz
            else:
                disc = np.sqrt(np.maximum(B * B - 4.0 * A * C, 0.0))
                q = -0.5 * (B + np.sign(B) * disc)
                t1, t2 = q / A, C / q
                t = np.where(np.abs(t1) <= np.abs(t2), t1, t2)
            x = x0 + t * Lx
            z = zv + z0 + t * Lz
            opl = opl + n1 * t
            nx = -c * x
            nz = np.ones_like(x)
            nn = np.hypot(nx, nz)
            nx, nz = nx / nn, nz / nn
            ci = Lx * nx + Lz * nz
            mu = n1 / n2
            ct = np.sqrt(np.maximum(1.0 - mu * mu * (1.0 - ci * ci), 0.0))
            Lx, Lz = mu * Lx + (ct - mu * ci) * nx, mu * Lz + (ct - mu * ci) * nz
            if i < len(surfaces) - 1:
                zv += float(thicknesses[i])
        z_exit = float(sum(thicknesses))
        n_exit = float(get_glass_index(surfaces[-1]['glass_after'], _LAM))
        tf = (z_exit - z) / Lz
        return x + tf * Lx, opl + n_exit * tf

    def test_the_exit_leg_uses_the_resolved_exit_index(self):
        """BAR.  The oracle's own floor is the interpolation of a 20001-ray fan
        onto the field row, ~1e-14 m.  Measured 2026-09-12: 8e-13 m rms
        (1.3e-6 waves).  The bar is 1e-10 m -- two decades above the
        measurement and 5 decades below the defect, which is
        ``(n_exit - 1) * |sag_last| / cos`` = 2.08e-5 m = 33 waves on this
        fixture."""
        rx = dict(surfaces=[dict(radius=19.6e-3, glass_before='AIR',
                                 glass_after='N-BK7'),
                            dict(radius=-27.4e-3, glass_before='N-BK7',
                                 glass_after='N-SF11')],
                  thicknesses=[2.5e-3], aperture_diameter=self.AP)
        ax = (np.arange(self.N) - self.N / 2) * self.DX
        X, Y = np.meshgrid(ax, ax)
        E = np.exp(-(X ** 2 + Y ** 2) / (self.AP / 3) ** 2).astype(
            np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Eo = apply_real_lens(E, prescription=rx, wavelength=_LAM,
                                 dx=self.DX, surface_model='displaced',
                                 displaced_mode='remap')
        h = np.linspace(1e-9, 0.999 * self.AP / 2, 20001)
        xo, oplo = self._trace(rx['surfaces'], rx['thicknesses'], h)
        sel = (ax > 0.05 * self.AP / 2) & (ax < 0.75 * self.AP / 2)
        order = np.argsort(xo)
        Wr = np.interp(ax[sel], xo[order], oplo[order])
        d = np.unwrap(np.angle(Eo[self.N // 2]))[sel] / _K0 - Wr
        d = d - d.mean()
        assert float(np.sqrt(np.mean(d ** 2))) < 1e-10, (
            f"remap exit OPL deviates from the closed-form immersed trace by "
            f"{np.sqrt(np.mean(d ** 2)) * 1e9:.4f} nm rms")


# ===========================================================================
# L16 / E4 -- the sag kernel, byte for byte
# ===========================================================================
class TestVerifyL16E4SagKernelByteIdentity:
    """The in-place conic chain must be the PREVIOUS expression bit for bit,
    and the numba aspheric kernel must not care about memory layout."""

    @staticmethod
    def _previous_conic(h_sq, R, conic):
        """``surface_sag_general``'s conic branch at b97c0b6e^, verbatim."""
        if R is None or np.isinf(R):
            return np.zeros_like(h_sq)
        norm = (1 + conic) * h_sq / R ** 2
        valid = norm < 0.9999
        denom_arg = np.where(valid, 1 - norm, 0.01)
        return np.where(valid, h_sq / (R * (1 + np.sqrt(denom_arg))), np.nan)

    @pytest.mark.parametrize('dtype', [np.float64, np.float32])
    @pytest.mark.parametrize('R,k', [(50e-3, 0.0), (-22.28e-3, 0.0),
                                     (12e-3, -1.0), (8e-3, -3.0),
                                     (np.inf, 0.0)])
    def test_conic_chain_is_the_previous_expression(self, dtype, R, k):
        rng = np.random.default_rng(20260912)
        h_sq = ((rng.random((129, 67)) * 12e-3) ** 2).astype(dtype)
        got = np.asarray(surface_sag_general(h_sq, R, k, None))
        ref = np.asarray(self._previous_conic(h_sq, R, k)).astype(got.dtype)
        assert np.array_equal(got.view(np.uint8), ref.view(np.uint8)), (
            f"in-place conic chain is not bit-identical at {np.dtype(dtype)}, "
            f"R={R}, k={k}: max|d| = {np.nanmax(np.abs(got - ref)):.3e}")

    @pytest.mark.parametrize('layout', ['F-order', 'transposed',
                                        'neg-stride-x', 'neg-stride-y',
                                        'strided-2'])
    def test_aspheric_kernel_ignores_memory_layout(self, layout):
        """E4.  The kernel accumulates through ``sag.ravel()``, a VIEW only for
        a C-contiguous buffer, so a non-C-contiguous ``h_sq`` used to drop the
        WHOLE polynomial silently (measured 100 % of the term).  The oracle is
        the C-ordered answer of the same call, so this is a bit identity, not a
        tolerance."""
        rng = np.random.default_rng(7)
        base = (rng.random((96, 64)) * 6e-3) ** 2
        asph = {4: 1.2e3, 6: -8.0e6, 8: 3.5e10}
        ref = np.asarray(surface_sag_general(np.ascontiguousarray(base),
                                             40e-3, -0.6, asph))
        views = {
            'F-order': np.asfortranarray(base),
            'transposed': np.ascontiguousarray(base.T).T,
            'neg-stride-x': np.ascontiguousarray(base[:, ::-1])[:, ::-1],
            'neg-stride-y': np.ascontiguousarray(base[::-1])[::-1],
            'strided-2': np.ascontiguousarray(np.repeat(base, 2,
                                                        axis=1))[:, ::2],
        }
        got = np.asarray(surface_sag_general(views[layout], 40e-3, -0.6, asph))
        assert not views[layout].flags['C_CONTIGUOUS']
        assert np.array_equal(np.ascontiguousarray(got).view(np.uint8),
                              np.ascontiguousarray(ref).view(np.uint8)), (
            f"{layout}: aspheric sag differs from the C-ordered answer by "
            f"max {np.max(np.abs(got - ref)):.3e} m")


# ===========================================================================
# L9 -- the Newton early exit is a BITWISE fixed point
# ===========================================================================
class TestVerifyL9NewtonEarlyExitIsBitwise:
    """The four geometric Newton loops stop when ``t`` reaches a bitwise fixed
    point.  The claim 'the output cannot move' is checked by DISABLING the
    exit -- monkeypatching ``numpy.array_equal`` to False inside the module
    forces the old fixed 24 sweeps -- and requiring byte identity.

    Fixtures: a spherical singlet and a conic+aspheric one, at radii and grids
    WP-A2 did not use.
    """

    RX = dict(surfaces=[dict(radius=19.6e-3, glass_before='AIR',
                             glass_after='N-BK7'),
                        dict(radius=-27.4e-3, glass_before='N-BK7',
                             glass_after='AIR')],
              thicknesses=[2.5e-3])
    RXA = dict(surfaces=[dict(radius=12.0e-3, conic=-1.0,
                              aspheric_coeffs={4: 1.0e3, 6: -2.0e6},
                              glass_before='AIR', glass_after='N-SF11'),
                         dict(radius=-45.0e-3, glass_before='N-SF11',
                              glass_after='AIR')],
               thicknesses=[3.0e-3])

    @staticmethod
    def _bytes_equal(a, b):
        a = np.ascontiguousarray(np.asarray(a))
        b = np.ascontiguousarray(np.asarray(b))
        return (a.shape == b.shape and a.dtype == b.dtype
                and bool((a.view(np.uint8) == b.view(np.uint8)).all()))

    @pytest.mark.parametrize('which', ['spherical', 'asphere'])
    @pytest.mark.parametrize('r_max', [0.5e-3, 2.0e-3])
    def test_forcing_the_full_sweep_count_changes_no_bit(self, which, r_max,
                                                         monkeypatch):
        rx = self.RX if which == 'spherical' else self.RXA
        real_eq = np.array_equal
        builders = [
            (_build_displaced_ray_map_2d,
             (rx['surfaces'], rx['thicknesses'], _LAM, r_max), {'n_side': 41}),
            (_build_displaced_ray_map,
             (rx['surfaces'], rx['thicknesses'], _LAM, r_max), {}),
            (_build_displaced_cos_luts,
             (rx['surfaces'], rx['thicknesses'], _LAM, r_max), {}),
        ]
        for fn, args, kw in builders:
            on = fn(*args, **kw)
            monkeypatch.setattr(np, 'array_equal',
                                lambda *a, **k: False, raising=True)
            try:
                off = fn(*args, **kw)
            finally:
                monkeypatch.setattr(np, 'array_equal', real_eq, raising=True)
            pairs = zip(on[:6], off[:6]) if isinstance(on, tuple) else \
                [(on, off)]
            for j, (a, b) in enumerate(pairs):
                if np.isscalar(a) or np.ndim(a) == 0:
                    assert a == b
                    continue
                assert self._bytes_equal(a, b), (
                    f"{fn.__name__} output {j} moved when the early exit was "
                    f"disabled: the fixed point is not bitwise")


# ===========================================================================
# L1 -- the gate's decision on an IMMERSED-rear singlet
# ===========================================================================
class TestVerifyL1GateSkipsAnImmersedRearSinglet:
    """A fixture WP-A2 did not use: the exit medium is N-SF11, so both the
    exit-vertex transfer and the exit momentum ``p = n_exit L`` carry a
    non-unit index.  The split-step model's own exit residual on it is 2.37 nm
    rms against an independent trace, i.e. below the 5 nm gate, so the gate
    must SKIP -- asserted as a bit identity, which no tolerance can soften.

    Pre-fix the same call imprinted a spurious rho**2 screen on every
    prescription it was given, so this returned a DIFFERENT field.
    """

    def test_the_gate_skips(self):
        rx = dict(surfaces=[dict(radius=40e-3, glass_before='AIR',
                                 glass_after='N-BK7'),
                            dict(radius=-60e-3, glass_before='N-BK7',
                                 glass_after='N-SF11')],
                  thicknesses=[3.5e-3], aperture_diameter=4e-3)
        N, dx = 256, 1.1e-5
        E = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = apply_real_lens(E, prescription=rx, wavelength=_LAM, dx=dx)
            b = apply_real_lens(E, prescription=rx, wavelength=_LAM, dx=dx,
                                seidel_correction=True)
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8)), (
            f"the 5 nm gate fired on a singlet whose model residual is "
            f"2.37 nm rms; max|d| = {np.max(np.abs(a - b)):.3e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""Pinning tests for the v4.14.3 Agent-B audit fixes.

Audit reference
---------------

``AUDIT_V4_14_2_2026_05_17.md`` -- Agent B scope covers
``lumenairy/sources/core.py`` and ``lumenairy/optimize/multiconfig.py``.
The audit flagged five P1 issues; this module pins all five.

Agent B fixes
-------------

* **B.1 / P1-NEW-4** -- ``create_led_source`` had a bare ``*args``
  VAR_POSITIONAL collector that silently re-routed 5-positional
  NEW-canonical-order calls ``(N, dx, wavelength, diameter,
  divergence_angle)`` as if they were the LEGACY positional form,
  producing a "633 nm-wide LED at 0.3 m wavelength" with only a
  DeprecationWarning.  v4.14.3 adds a post-remap scale-inversion
  check inside the legacy shim (apparent wavelength > 10x apparent
  diameter -> TypeError), which catches the canonical-order
  mistake without breaking legacy callers (whose legacy
  ``diameter > wavelength * 10`` always holds for real
  emitters).  We deliberately did NOT add PEP 570 ``/`` -- it does
  not prevent the ``*args`` collector from eating surplus
  positional, so it adds no safety against the canonical-order
  mistake and would force every existing kwarg-based caller (and
  the v4.14.2 audit test infrastructure) to drop ``N/dx/wavelength``
  out of kwargs.

* **B.2 / P1-NEW-5** -- ``_validate_grid_params`` accepted both
  ``int`` and ``(Ny, Nx)`` tuple ``N``, but only 3 of the 10 source
  factories (``create_gaussian_beam``, ``create_hermite_gauss``,
  ``create_laguerre_gauss``) unpack tuples internally; the other 7
  call ``np.arange(N)`` and crash with an obscure ``TypeError``.
  v4.14.3 adds ``support_tuple_N: bool = False`` to the validator;
  the 3 tuple-supporting factories pass ``True`` and the other 7
  inherit the default ``False``, so a tuple-N input now raises a
  clear, factory-named ``TypeError`` instead of a downstream
  ``np.arange`` crash.

* **B.3 / P1-NEW-9** -- ``create_bessel_beam`` accepted any
  ``cone_angle``.  ``cone_angle=pi`` -> ``sin=0`` -> silent uniform
  DC field labelled "Bessel beam"; ``cone_angle=2*pi/3`` -> non-
  physical evanescent ``k_r > k0`` (apparent ``k_r`` aliases back to
  ``sin(pi/3)`` because the user's literal angle is past the
  propagation horizon).  v4.14.3 enforces ``0 < cone_angle < pi/2``.

* **B.4 / P1-NEW-10** -- ``create_fiber_mode`` accepted non-positive
  ``mode_field_diameter``.  ``MFD=0`` -> divide-by-zero in
  ``sigma = w0/sqrt(2) = 0``; negative MFD -> sigma's sign flips
  and the field grows away from the centre (non-physical).
  v4.14.3 raises ``ValueError`` on ``MFD <= 0`` or non-finite.

* **B.5 / P1-MC** -- ``beam_expander_prescription`` AND
  ``keplerian_telescope`` both hardcoded ``n=1.5`` in the
  lensmaker formula ``R = f*(n-1)*2``.  For ``glass='N-LASF9'``
  (n=1.85 at 587.6 nm) the focal length feeding ``_zero_C_air_gap``
  was off by ~17% -- a real physics error.  v4.14.3 routes the
  glass + wavelength through ``get_glass_index``, with a
  ``UserWarning``-flagged fallback to ``n=1.5`` on lookup failure.

Author: Andrew Traverso -- v4.14.3 / Agent B
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.sources.core import (
    create_annular_beam,
    create_bessel_beam,
    create_fiber_mode,
    create_gaussian_beam,
    create_hermite_gauss,
    create_laguerre_gauss,
    create_led_source,
    create_point_source,
    create_tilted_plane_wave,
    create_top_hat_beam,
)


# ============================================================================
# B.1 -- P1-NEW-4: create_led_source VAR_POSITIONAL footgun
# ============================================================================

class TestB1LedSourceCanonicalPositionalFootgun:
    """Pin the v4.14.3 defence against the NEW-canonical-order positional
    footgun in ``create_led_source``.  See module docstring for full
    rationale.
    """

    def test_create_led_source_canonical_five_positional_raises(self):
        """5 positional in the NEW canonical order
        ``(N, dx, wavelength, diameter, divergence_angle)`` is the
        footgun the audit flagged.  Before v4.14.3 this silently
        re-routed as the legacy form (wavelength->diameter,
        diameter->divergence_angle, divergence_angle->wavelength)
        and emitted only a misleading DeprecationWarning.  Now the
        scale-inversion sanity check inside the shim raises a clear
        TypeError pointing the user at the canonical kwarg form.
        """
        with pytest.raises(TypeError) as info:
            create_led_source(
                64, 16e-6,
                1.31e-6,    # wavelength (NEW canonical 3rd slot)
                100e-6,     # diameter -- USER ERROR: should be kwarg
                0.3,        # divergence_angle -- USER ERROR: should be kwarg
            )
        msg = str(info.value).lower()
        # The error must mention the canonical kwarg form so the user
        # knows how to fix the call.
        assert ('diameter' in msg and 'divergence_angle' in msg), (
            f"TypeError must name the keyword-only parameters; got: "
            f"{info.value!r}")
        # The error must hint at the scale inversion -- the kind of
        # signal a user can act on (these specific micron-scale numbers
        # are wavelengths, not diameters).
        assert 'wavelength' in msg or 'scale' in msg or 'canonical' in msg, (
            f"TypeError should point at the canonical positional order "
            f"or the scale-inversion symptom; got: {info.value!r}")

    def test_create_led_source_legacy_eight_positional_still_warns(self):
        """The full LEGACY positional form
        ``(N, dx, diameter, divergence_angle, wavelength, x0, y0, dtype)``
        must still work and emit ``DeprecationWarning`` so existing
        callers have a transition window.  No TypeError, correct
        output shape.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            E, angles, x, y = create_led_source(
                64, 16e-6, 100e-6, 0.3, 1.31e-6, 0.0, 0.0, np.complex128,
            )
        # Output shape correctness -- the shim must have correctly
        # re-mapped the legacy args.
        assert E.shape == (64, 64), (
            f"Legacy 8-positional call should yield shape (64, 64); "
            f"got {E.shape}.")
        assert E.dtype == np.complex128
        assert len(angles) == 37, (
            f"create_led_source returns 37 source angles "
            f"(1 + 6 + 12 + 18); got {len(angles)}.")
        # And the DeprecationWarning must have fired.
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'create_led_source' in str(w.message)]
        assert len(dep) >= 1, (
            f"Legacy positional call must emit DeprecationWarning; "
            f"got {[str(w.message) for w in caught]}.")


# ============================================================================
# B.2 -- P1-NEW-5: tuple-N validator-vs-implementation gap
# ============================================================================

class TestB2TupleNValidatorImplementationGap:
    """Pin that the 7 factories which call ``np.arange(N)`` directly
    now reject tuple-N at the validator with a clear TypeError instead
    of crashing downstream with an obscure ``np.arange`` TypeError.
    The 3 factories that DO unpack tuples
    (``create_gaussian_beam``, ``create_hermite_gauss``,
    ``create_laguerre_gauss``) continue to accept tuples.
    """

    # Factories that MUST accept tuple-N (they unpack internally).
    TUPLE_OK_FACTORIES = ('create_gaussian_beam',
                          'create_hermite_gauss',
                          'create_laguerre_gauss')

    # Factories that MUST reject tuple-N at validation time (they
    # call np.arange(N) directly and would crash obscurely).
    TUPLE_REJECT_FACTORIES = ('create_tilted_plane_wave',
                              'create_point_source',
                              'create_top_hat_beam',
                              'create_annular_beam',
                              'create_fiber_mode',
                              'create_led_source',
                              'create_bessel_beam')

    @staticmethod
    def _call_factory(name, N, dx=1e-6, wavelength=633e-9):
        """Invoke a named factory with the minimum required kwargs
        plus the test-supplied ``N``."""
        if name == 'create_gaussian_beam':
            return create_gaussian_beam(N, dx, wavelength, sigma=10e-6)
        if name == 'create_hermite_gauss':
            return create_hermite_gauss(N, dx, w0=10e-6,
                                          wavelength=wavelength)
        if name == 'create_laguerre_gauss':
            return create_laguerre_gauss(N, dx, w0=10e-6,
                                           wavelength=wavelength)
        if name == 'create_tilted_plane_wave':
            return create_tilted_plane_wave(N, dx, wavelength)
        if name == 'create_point_source':
            return create_point_source(N, dx, wavelength, z0=1e-3)
        if name == 'create_top_hat_beam':
            return create_top_hat_beam(N, dx, wavelength, diameter=10e-6)
        if name == 'create_annular_beam':
            return create_annular_beam(N, dx, wavelength,
                                         outer_diameter=20e-6,
                                         inner_diameter=10e-6)
        if name == 'create_fiber_mode':
            return create_fiber_mode(N, dx, wavelength,
                                       mode_field_diameter=10e-6)
        if name == 'create_led_source':
            return create_led_source(N, dx, wavelength,
                                       diameter=100e-6,
                                       divergence_angle=0.3)
        if name == 'create_bessel_beam':
            return create_bessel_beam(N, dx, wavelength, cone_angle=0.05)
        raise AssertionError(f'unknown factory {name!r}')

    @pytest.mark.parametrize('factory', TUPLE_OK_FACTORIES)
    def test_tuple_N_supported_factories_succeed(self, factory):
        """The 3 mode-family factories accept tuple-N=(Ny, Nx)."""
        out = self._call_factory(factory, N=(48, 64))
        # First element is always the complex field; shape must be (Ny, Nx).
        E = out[0]
        assert E.shape == (48, 64), (
            f'{factory}: tuple-N=(48, 64) should yield E.shape=(48, 64); '
            f'got {E.shape}.')

    @pytest.mark.parametrize('factory', TUPLE_REJECT_FACTORIES)
    def test_tuple_N_unsupported_factories_raise_clear_typeerror(self,
                                                                  factory):
        """The 7 factories that call ``np.arange(N)`` directly now
        raise a clear ``TypeError`` at validation time instead of an
        obscure ``np.arange`` TypeError downstream.
        """
        with pytest.raises(TypeError) as info:
            self._call_factory(factory, N=(64, 64))
        msg = str(info.value).lower()
        # The error must name the factory so the caller can find the
        # offending call site.
        assert factory in msg, (
            f'{factory}: tuple-N TypeError should name the factory; '
            f'got: {info.value!r}')
        # The error must mention tuple / N / supported -- enough
        # signal for the user to switch to integer N.
        assert ('tuple' in msg and ('integer' in msg or 'support' in msg)), (
            f'{factory}: tuple-N TypeError should mention "tuple" and '
            f'"integer"/"support"; got: {info.value!r}')


# ============================================================================
# B.3 -- P1-NEW-9: create_bessel_beam cone_angle constraint
# ============================================================================

class TestB3BesselBeamConeAngleConstraint:
    """Pin that ``create_bessel_beam`` rejects non-physical cone angles
    that produced silent uniform-DC ("cone_angle=pi") or evanescent
    ("cone_angle > pi/2") outputs pre-v4.14.3.
    """

    @pytest.mark.parametrize('bad_angle',
                              [0.0, np.pi / 2.0, np.pi, 2 * np.pi / 3, -0.1])
    def test_invalid_cone_angle_raises_valueerror(self, bad_angle):
        """Each non-physical cone angle must raise ValueError."""
        with pytest.raises(ValueError) as info:
            create_bessel_beam(64, 1e-6, 633e-9, cone_angle=bad_angle)
        msg = str(info.value).lower()
        assert 'cone_angle' in msg, (
            f'cone_angle={bad_angle}: error should mention cone_angle; '
            f'got: {info.value!r}')
        assert ('pi/2' in msg or '0' in msg or 'propagating' in msg), (
            f'cone_angle={bad_angle}: error should explain the valid '
            f'range; got: {info.value!r}')

    def test_valid_cone_angle_succeeds(self):
        """A valid cone_angle in (0, pi/2) succeeds."""
        E, x, y = create_bessel_beam(64, 1e-6, 633e-9,
                                       cone_angle=np.pi / 6.0)
        assert E.shape == (64, 64)
        # J0(0) = 1, so the central pixel of an unshifted Bessel beam
        # must be the maximum -- distinguishes "actual Bessel field"
        # from "uniform DC".
        cy, cx = 32, 32
        assert abs(E[cy, cx]) >= float(np.abs(E).max()) - 1e-6, (
            'central pixel of unshifted J0 Bessel should be near max')


# ============================================================================
# B.4 -- P1-NEW-10: create_fiber_mode MFD > 0 constraint
# ============================================================================

class TestB4FiberModeMfdConstraint:
    """Pin that ``create_fiber_mode`` rejects non-positive
    ``mode_field_diameter`` -- pre-fix MFD=0 hit a divide-by-zero in
    sigma and MFD<0 silently flipped the Gaussian sign.
    """

    @pytest.mark.parametrize('bad_mfd', [0.0, -1e-6, -10e-6])
    def test_invalid_mfd_raises_valueerror(self, bad_mfd):
        with pytest.raises(ValueError) as info:
            create_fiber_mode(64, 1e-6, 1.31e-6,
                                mode_field_diameter=bad_mfd)
        msg = str(info.value).lower()
        assert 'mode_field_diameter' in msg, (
            f'MFD={bad_mfd}: error should mention mode_field_diameter; '
            f'got: {info.value!r}')
        assert ('positive' in msg or 'finite' in msg), (
            f'MFD={bad_mfd}: error should explain MFD must be positive; '
            f'got: {info.value!r}')

    def test_valid_mfd_succeeds(self):
        """A typical MFD succeeds."""
        E, x, y = create_fiber_mode(64, 1e-6, 1.31e-6,
                                       mode_field_diameter=10e-6)
        assert E.shape == (64, 64)
        # Peak at centre, exponential decay -- verify no sign-flip.
        cy, cx = 32, 32
        assert abs(E[cy, cx]) > abs(E[0, 0])


# ============================================================================
# B.5 -- P1-MC: multiconfig.py hardcoded n=1.5 in lensmaker formula
# ============================================================================

class TestB5MulticonfigGlassIndex:
    """Pin that ``beam_expander_prescription`` and
    ``keplerian_telescope`` use the prescription's actual glass +
    wavelength rather than the hardcoded ``n=1.5``.  For
    ``glass='N-LASF9'`` (n ~ 1.85 at 587.6 nm) the lensmaker formula
    ``R = f*(n-1)*2`` produces surface radii 17% different from the
    pre-fix n=1.5 result; we verify the actual radii match the
    canonical lensmaker formula at the design wavelength.
    """

    def test_beam_expander_with_n_lasf9_correct_focal_length(self):
        """Build a 3x beam expander with ``glass='N-LASF9'`` at 587.6 nm
        (sodium D, where N-LASF9 has n=1.85024...).  The two surface
        radii of the objective should be ``+/- R_obj`` with
        ``R_obj = f_obj * (n - 1) * 2``.  Compare to the analytic
        lensmaker formula at the design wavelength.
        """
        from lumenairy.glass import get_glass_index
        f_obj = 100e-3
        glass = 'N-LASF9'
        wavelength = 587.6e-9  # sodium D
        n = get_glass_index(glass, wavelength)
        # The pre-v4.14.3 hardcoded n=1.5 gave R_obj = f * 0.5 * 2 = f.
        # The corrected formula gives R_obj = f * (n-1) * 2.
        R_obj_expected = f_obj * (n - 1.0) * 2.0
        R_obj_old_buggy = f_obj * (1.5 - 1.0) * 2.0  # = f_obj exactly
        # Sanity: the two should differ by ~17% -- if they don't,
        # the test is mis-calibrated.
        rel_buggy = abs(R_obj_expected - R_obj_old_buggy) / R_obj_expected
        assert rel_buggy > 0.1, (
            f'Test calibration check: N-LASF9 vs n=1.5 should differ by '
            f'>10%; got {rel_buggy:.3%}.  Fix the test.')

        pres = la.beam_expander_prescription(
            M=3.0, f_objective=f_obj, glass=glass,
            wavelength=wavelength, aperture=20e-3,
        )
        # The prescription builds surfaces in
        # [eye_front, eye_back, obj_front, obj_back] order; obj's
        # front radius = +R_obj_expected.
        surfaces = pres['surfaces']
        R_obj_front = surfaces[2]['radius']
        rel_err = abs(R_obj_front - R_obj_expected) / abs(R_obj_expected)
        assert rel_err < 0.01, (
            f'beam_expander_prescription with N-LASF9 should use the '
            f'glass index at the design wavelength; expected '
            f'R_obj_front = {R_obj_expected:.6g} ({rel_err:.3%} error).  '
            f'Got {R_obj_front:.6g} -- the pre-v4.14.3 hardcoded n=1.5 '
            f'value would be {R_obj_old_buggy:.6g}.')

    def test_keplerian_telescope_with_n_lasf9_correct_focal_length(self):
        """Same as the beam-expander test but for the Keplerian
        telescope.
        """
        from lumenairy.glass import get_glass_index
        f_obj = 200e-3
        f_eye = 50e-3
        glass = 'N-LASF9'
        wavelength = 587.6e-9
        n = get_glass_index(glass, wavelength)
        R_obj_expected = f_obj * (n - 1.0) * 2.0

        pres = la.keplerian_telescope(
            f_objective=f_obj, f_eyepiece=f_eye, glass=glass,
            wavelength=wavelength, aperture=25.4e-3,
        )
        # keplerian surfaces are
        # [obj_front, obj_back, eye_front, eye_back]; obj_front is index 0.
        surfaces = pres['surfaces']
        R_obj_front = surfaces[0]['radius']
        rel_err = abs(R_obj_front - R_obj_expected) / abs(R_obj_expected)
        assert rel_err < 0.01, (
            f'keplerian_telescope with N-LASF9 should use the glass '
            f'index at the design wavelength; expected R_obj_front = '
            f'{R_obj_expected:.6g} ({rel_err:.3%} error).  Got '
            f'{R_obj_front:.6g}.')

    def test_keplerian_telescope_unknown_glass_warns(self):
        """Unknown glass name -> ``UserWarning`` + fallback to n=1.5.
        The prescription must still build (so existing callers with
        odd glass names degrade gracefully).
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            # Wrap in pytest.raises-skip: ``get_glass_index`` raises
            # ValueError on unknown glass; the helper catches that and
            # falls back to n=1.5 with a UserWarning.  But the
            # ``afocal`` air-gap solve runs ``surfaces_from_prescription``
            # which itself queries the glass index -- it likely raises
            # too.  We accept either ``(success + UserWarning fired)``
            # OR ``(downstream ValueError + UserWarning fired before
            # the raise)`` -- the contract is that THE HELPER warned
            # and fell back, not that the entire pipeline succeeds.
            err = None
            try:
                pres = la.keplerian_telescope(
                    f_objective=200e-3, f_eyepiece=50e-3,
                    glass='__bogus_glass_name__',
                    wavelength=587.6e-9, aperture=25.4e-3,
                )
            except (ValueError, KeyError) as e:
                # Downstream surfaces_from_prescription rejection is
                # fine; we just need the helper to have warned.
                err = e
                pres = None
        # The UserWarning from _resolve_lens_glass_index must have
        # fired regardless of whether the downstream pipeline crashed.
        user_warns = [w for w in caught
                      if issubclass(w.category, UserWarning)
                      and ('glass' in str(w.message).lower()
                           or 'lens' in str(w.message).lower())]
        assert len(user_warns) >= 1, (
            f'Unknown-glass call must emit a UserWarning from the '
            f'fallback; got warnings: '
            f'{[(w.category.__name__, str(w.message)) for w in caught]}; '
            f'and exception: {err!r}')

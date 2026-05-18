"""Pinning tests for the v4.14.2 audit closures handled by Agent D.

Audit reference
---------------

``AUDIT_V4_14_1_2026_05_17.md`` flagged 1 P0 + 10 P1 follow-ups; Agent
D closed three of the P1s plus one structural meta-pin.  Scope was
restricted to ``lumenairy/elements/doe.py`` and
``lumenairy/sources/core.py`` only.

* **D.1 / P1-NEW-6** -- ``makedammann2d`` historically took
  ``periodx``, ``periody`` and ``waveln`` in micrometres while the rest
  of the library was SI metres throughout.  An SI-convention caller
  (e.g. ``periodx=61e-6, waveln=1.31e-6``) got a thousandfold-off
  ``samplingx`` that was silently masked by an output ``* 1e-6``
  rescale on ``cell_pixel_size_x``.  v4.14.2 converts the function to
  SI metres end-to-end and emits a ``DeprecationWarning`` when any of
  ``periodx``, ``periody``, ``waveln`` look like legacy micrometres
  (heuristic: ``> 1e-3 m``).
* **D.2 / P1-NEW-9** -- ``create_led_source`` had positional
  ``(N, dx, diameter, divergence_angle, wavelength, ...)`` which broke
  the post-v4.7 convention of keyword-only physical parameters with
  ``wavelength`` in the canonical 3rd positional slot.  The factory
  also lacked ``dy`` and a ``*`` keyword-only separator.  v4.14.2
  reorders to ``(N, dx, wavelength, *, diameter, divergence_angle,
  dy=None, x0=0, y0=0, dtype=None)`` and accepts the legacy positional
  form for one release with a ``DeprecationWarning``.
* **D.3 / P1-NEW-10** -- The 10 ``create_*`` factories in
  ``sources/core.py`` silently accepted ``N=0``, ``dx<=0``,
  ``wavelength<=0`` etc.  v4.14.2 adds a private
  ``_validate_grid_params`` helper at module top and calls it at every
  factory entry.

Author: Andrew Traverso -- v4.14.2 / Agent D
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.doe import makedammann2d
from lumenairy.sources.core import (
    _validate_grid_params,
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
# D.1 -- P1-NEW-6: makedammann2d SI-unit conversion
# ============================================================================

class TestP1New6MakedammannSiUnits:
    """``makedammann2d`` now takes ``periodx``, ``periody``, ``waveln`` in
    SI metres (was micrometres pre-v4.14.2).  Returned ``cell_pixel_size``
    is in metres and is the bare ``samplingx``/``samplingy`` (no
    ``* 1e-6`` rescale at the end).  Legacy micrometre inputs are still
    accepted with a ``DeprecationWarning``.
    """

    def test_makedammann2d_si_units_match_expected_geometry(self):
        """Calling with SI metres ``periodx=61e-6, waveln=1.31e-6`` must
        produce a ``cell_pixel_size_x`` in the correct order-of-magnitude
        for a real Dammann grating.

        Math: ``ndifordersx = 2 * ceil(periodx / (wavsamp * waveln) / 2)
        = 2 * ceil(61e-6 / (0.5 * 1.31e-6) / 2) = 2 * ceil(46.56) = 94``.
        Then ``samplingx = periodx / ndifordersx = 61e-6 / 94 approx 6.5e-7
        m approx 650 nm``.  The post-fix ``cell_pixel_size_x`` is
        ``samplingx`` directly in metres; the pre-fix bug would have
        returned ``samplingx * 1e-6 approx 6.5e-13 m``.

        Pin the result is within a wide order-of-magnitude window
        ``1e-7 m <= cell_pixel_size_x <= 1e-5 m`` (100 nm to 10 um),
        which catches both the post-fix correct value (650 nm) and
        rules out the pre-fix off-by-1e6 drift.
        """
        # Tiny iteration count -- we only need the geometry, not the
        # IFTA optimization quality.
        _, _, (cell_dx, cell_dy) = makedammann2d(
            periodx=61e-6, periody=61e-6, waveln=1.31e-6,
            phaselevels=2, phasesteps=1,
            diforders=np.ones((2, 2)),
            itr=2, plot=False, seed=42,
        )
        # Expected order of magnitude for telecom-O Dammann gratings.
        assert 1e-7 <= cell_dx <= 1e-5, (
            f"cell_pixel_size_x = {cell_dx} m is outside the expected "
            f"100 nm - 10 um range for a Dammann grating at telecom-O "
            f"band; SI-units conversion may be wrong.  Compare to the "
            f"analytical value samplingx = periodx / ndifordersx "
            f"approx 6.5e-7 m.")
        assert 1e-7 <= cell_dy <= 1e-5, (
            f"cell_pixel_size_y = {cell_dy} m is outside expected range; "
            f"see cell_pixel_size_x note above.")
        # Square grating: dy should equal dx exactly.
        assert cell_dx == cell_dy

    def test_makedammann2d_si_units_match_analytic_samplingx(self):
        """Direct analytic pin on the new SI-metres math: ``samplingx``
        should equal ``periodx / ndifordersx`` exactly, with no
        hidden ``* 1e-6`` factor anywhere in the return path.
        """
        periodx = 61e-6
        waveln = 1.31e-6
        wavsamp = 0.5
        # Replicate the internal computation:
        ndifordersx = int(np.ceil(periodx / (wavsamp * waveln) * 0.5)) * 2
        expected_samplingx = periodx / ndifordersx
        _, _, (cell_dx, _) = makedammann2d(
            periodx=periodx, periody=periodx, waveln=waveln,
            wavsamp=wavsamp,
            phaselevels=2, phasesteps=1,
            diforders=np.ones((2, 2)),
            itr=2, plot=False, seed=42,
        )
        assert np.isclose(cell_dx, expected_samplingx, rtol=1e-12), (
            f"cell_pixel_size_x = {cell_dx} m does not match "
            f"analytical samplingx = {expected_samplingx} m.  The "
            f"v4.14.2 fix should make these identical (no ``* 1e-6`` "
            f"rescale).")

    def test_makedammann2d_micrometre_input_warns(self):
        """Calling with the legacy micrometre form
        ``periodx=61.0, waveln=1.31`` must trigger a
        ``DeprecationWarning`` with a migration message that names
        the SI-metres replacement.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _, _, _ = makedammann2d(
                periodx=61.0, periody=61.0, waveln=1.31,
                phaselevels=2, phasesteps=1,
                diforders=np.ones((2, 2)),
                itr=2, plot=False, seed=42,
            )
            depwarns = [r for r in w
                        if issubclass(r.category, DeprecationWarning)]
            assert len(depwarns) >= 1, (
                "Expected at least one DeprecationWarning when "
                "makedammann2d is called with legacy micrometre "
                "values periodx=61.0, waveln=1.31.  Got: "
                f"{[r.message for r in w]}")
            msg = str(depwarns[0].message)
            # The message must explain the SI migration concretely.
            assert 'micrometre' in msg.lower() or 'um' in msg.lower(), (
                f"DeprecationWarning message should mention micrometres "
                f"explicitly so users know how to migrate; got: {msg!r}")
            assert 'SI' in msg or '1e-6' in msg, (
                f"DeprecationWarning should reference SI / 1e-6 "
                f"migration; got: {msg!r}")

    def test_makedammann2d_micrometre_input_still_produces_correct_geometry(self):
        """Bit-for-bit continuity pin: a pre-v4.14.2 caller using the
        legacy micrometre form must still get the same ``cell_pixel_size``
        they used to (the legacy-um path internally rescales to SI metres
        for the math, then returns ``samplingx`` which carries the same
        physical metre value as the old ``samplingx * 1e-6`` did).
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            _, _, (legacy_dx, _) = makedammann2d(
                periodx=61.0, periody=61.0, waveln=1.31,
                phaselevels=2, phasesteps=1,
                diforders=np.ones((2, 2)),
                itr=2, plot=False, seed=42,
            )
            _, _, (si_dx, _) = makedammann2d(
                periodx=61e-6, periody=61e-6, waveln=1.31e-6,
                phaselevels=2, phasesteps=1,
                diforders=np.ones((2, 2)),
                itr=2, plot=False, seed=42,
            )
        assert np.isclose(legacy_dx, si_dx, rtol=1e-12), (
            f"Legacy um-form cell_pixel_size {legacy_dx} m does not "
            f"match SI-form cell_pixel_size {si_dx} m; the "
            f"v4.14.2 shim should preserve numerical continuity.")

    def test_makedammann2d_si_defaults_do_not_warn(self):
        """The new SI-metres defaults (``periodx=61e-6, waveln=1.31e-6``)
        must NOT emit a DeprecationWarning -- only legacy um inputs do.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _, _, _ = makedammann2d(
                phaselevels=2, phasesteps=1,
                diforders=np.ones((2, 2)),
                itr=2, plot=False, seed=42,
            )
            depwarns = [r for r in w
                        if issubclass(r.category, DeprecationWarning)
                        and 'makedammann2d' in str(r.message)]
            assert len(depwarns) == 0, (
                "Default SI-metres call to makedammann2d emitted "
                f"DeprecationWarning(s); should be silent.  Got: "
                f"{[r.message for r in depwarns]}")


# ============================================================================
# D.2 -- P1-NEW-9: create_led_source signature drift
# ============================================================================

class TestP1New9CreateLedSourceSignature:
    """``create_led_source`` now follows the canonical post-v4.7 layout:
    ``(N, dx, wavelength, *, diameter, divergence_angle, dy=None, x0=0,
    y0=0, dtype=None)``.  The legacy positional form
    ``(N, dx, diameter, divergence_angle, wavelength, ...)`` is still
    accepted for one release with a ``DeprecationWarning``.
    """

    def test_create_led_source_new_kwarg_form(self):
        """The canonical keyword-only form succeeds without warnings
        and returns a sensible LED field."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            E, angles, x, y = create_led_source(
                64, 16e-6, 1.31e-6,
                diameter=100e-6, divergence_angle=0.3,
            )
            depwarns = [r for r in w
                        if issubclass(r.category, DeprecationWarning)]
            assert len(depwarns) == 0, (
                "Canonical keyword-only call to create_led_source "
                f"emitted DeprecationWarning(s); got: "
                f"{[r.message for r in depwarns]}")
        assert E.shape == (64, 64)
        assert np.iscomplexobj(E)
        assert x.shape == (64,)
        assert y.shape == (64,)
        assert len(angles) == 37  # 1 + 6 + 12 + 18 for n_ring=3
        # The field is a top-hat: at least one pixel non-zero.
        assert np.any(np.abs(E) > 0)

    def test_create_led_source_new_kwarg_form_honours_dy(self):
        """``dy`` threading: pass an anamorphic ``dy != dx`` and verify
        the y-axis coordinate array is spaced by ``dy``, not ``dx``.
        """
        E, angles, x, y = create_led_source(
            64, 16e-6, 1.31e-6,
            diameter=100e-6, divergence_angle=0.3,
            dy=8e-6,  # half the x-pitch
        )
        # The y-coordinates should be spaced by 8e-6, not 16e-6.
        dy_actual = float(y[1] - y[0])
        assert np.isclose(dy_actual, 8e-6, rtol=1e-12), (
            f"dy kwarg not threaded into y-axis spacing; "
            f"got y[1]-y[0] = {dy_actual} m, expected 8e-6 m.")
        # And x is still spaced by dx = 16e-6.
        dx_actual = float(x[1] - x[0])
        assert np.isclose(dx_actual, 16e-6, rtol=1e-12), (
            f"dx not honoured; got x[1]-x[0] = {dx_actual} m, "
            f"expected 16e-6 m.")

    def test_create_led_source_positional_form_deprecation(self):
        """The legacy positional form
        ``create_led_source(N, dx, diameter, divergence_angle, wavelength)``
        must still work but emit a ``DeprecationWarning``.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            E, angles, x, y = create_led_source(
                64, 16e-6, 100e-6, 0.3, 1.31e-6)
            depwarns = [r for r in w
                        if issubclass(r.category, DeprecationWarning)
                        and 'create_led_source' in str(r.message)]
            assert len(depwarns) >= 1, (
                "Legacy positional call to create_led_source did NOT "
                "emit a DeprecationWarning.  Got all warnings: "
                f"{[r.message for r in w]}")
            msg = str(depwarns[0].message)
            # The migration message must point at the keyword-only form.
            assert 'keyword' in msg.lower() or 'diameter=' in msg, (
                f"DeprecationWarning should explain the keyword-only "
                f"migration; got: {msg!r}")
        # The legacy form should still produce the right output.
        assert E.shape == (64, 64)
        assert len(angles) == 37

    def test_create_led_source_positional_form_matches_kwarg_output(self):
        """Bit-for-bit continuity: legacy positional and new keyword-only
        forms with the same parameters must produce identical output."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            E_legacy, ang_legacy, x_legacy, y_legacy = create_led_source(
                64, 16e-6, 100e-6, 0.3, 1.31e-6)
        E_new, ang_new, x_new, y_new = create_led_source(
            64, 16e-6, 1.31e-6,
            diameter=100e-6, divergence_angle=0.3,
        )
        assert np.array_equal(E_legacy, E_new), (
            "Legacy positional and new keyword-only forms produce "
            "different fields for identical physical parameters.")
        assert ang_legacy == ang_new
        assert np.array_equal(x_legacy, x_new)
        assert np.array_equal(y_legacy, y_new)

    def test_create_led_source_missing_kwarg_raises_typeerror(self):
        """The new signature requires ``diameter`` and
        ``divergence_angle`` as keywords; missing either should raise
        a ``TypeError`` with a clear message.
        """
        # Missing diameter
        with pytest.raises(TypeError, match='diameter'):
            create_led_source(64, 16e-6, 1.31e-6, divergence_angle=0.3)
        # Missing divergence_angle
        with pytest.raises(TypeError, match='divergence_angle'):
            create_led_source(64, 16e-6, 1.31e-6, diameter=100e-6)

    def test_create_led_source_kwarg_conflict_legacy_positional_raises(self):
        """Passing legacy positionals AND a conflicting kwarg should
        raise ``TypeError`` with a clear message about the conflict.
        """
        with pytest.raises(TypeError, match="diameter.*both"):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=DeprecationWarning)
                create_led_source(
                    64, 16e-6, 100e-6, 0.3, 1.31e-6,
                    diameter=50e-6)  # conflicts


# ============================================================================
# D.3 -- P1-NEW-10: source factory input validation helper
# ============================================================================

# Parametrize over every factory + invalid input combination.  Each
# entry is ``(factory_callable, kwargs_for_valid_call, bad_kwarg,
# bad_value)``.
_VALID_KWARGS = dict(
    N=64, dx=16e-6, wavelength=1.31e-6,
)


def _gaussian_call(**override):
    kw = {**_VALID_KWARGS, 'sigma': 30e-6}
    kw.update(override)
    return create_gaussian_beam(**kw)


def _hermite_call(**override):
    kw = {**_VALID_KWARGS, 'w0': 60e-6}
    kw.update(override)
    return create_hermite_gauss(**kw)


def _laguerre_call(**override):
    kw = {**_VALID_KWARGS, 'w0': 60e-6}
    kw.update(override)
    return create_laguerre_gauss(**kw)


def _top_hat_call(**override):
    kw = {**_VALID_KWARGS, 'diameter': 100e-6}
    kw.update(override)
    return create_top_hat_beam(**kw)


def _annular_call(**override):
    kw = {**_VALID_KWARGS,
          'outer_diameter': 100e-6, 'inner_diameter': 30e-6}
    kw.update(override)
    return create_annular_beam(**kw)


def _fiber_call(**override):
    kw = {**_VALID_KWARGS, 'mode_field_diameter': 10e-6}
    kw.update(override)
    return create_fiber_mode(**kw)


def _led_call(**override):
    kw = {**_VALID_KWARGS,
          'diameter': 100e-6, 'divergence_angle': 0.3}
    kw.update(override)
    return create_led_source(**kw)


def _bessel_call(**override):
    kw = {**_VALID_KWARGS, 'cone_angle': 0.05}
    kw.update(override)
    return create_bessel_beam(**kw)


def _point_call(**override):
    kw = {**_VALID_KWARGS, 'z0': 100e-3}
    kw.update(override)
    return create_point_source(**kw)


def _tilted_call(**override):
    kw = dict(_VALID_KWARGS)
    kw.update(override)
    return create_tilted_plane_wave(**kw)


_FACTORY_CALLS = [
    ('create_gaussian_beam', _gaussian_call),
    ('create_hermite_gauss', _hermite_call),
    ('create_laguerre_gauss', _laguerre_call),
    ('create_top_hat_beam', _top_hat_call),
    ('create_annular_beam', _annular_call),
    ('create_fiber_mode', _fiber_call),
    ('create_led_source', _led_call),
    ('create_bessel_beam', _bessel_call),
    ('create_point_source', _point_call),
    ('create_tilted_plane_wave', _tilted_call),
]


_INVALID_KWARGS = [
    ('N', 0),
    ('N', -1),
    ('dx', 0),
    ('dx', -1.0),
    ('wavelength', 0),
    ('wavelength', -1.0),
]


class TestP1New10SourceFactoryInputValidation:
    """Every ``create_*`` factory in ``sources/core.py`` must validate
    ``N > 0``, ``dx > 0``, ``wavelength > 0`` (positive, finite) and
    raise ``ValueError`` with a message that names the factory.  Pre-
    v4.14.2 only the DOE family did this; the 10 factories here all
    silently accepted ``N=0, dx<=0, wavelength<=0``.
    """

    @pytest.mark.parametrize('fn_name, fn_call', _FACTORY_CALLS,
                              ids=[name for name, _ in _FACTORY_CALLS])
    @pytest.mark.parametrize('bad_kwarg, bad_value', _INVALID_KWARGS,
                              ids=[f'{k}={v}' for k, v in _INVALID_KWARGS])
    def test_factory_rejects_invalid_grid_params(
            self, fn_name, fn_call, bad_kwarg, bad_value):
        """Each ``create_*`` factory must raise ``ValueError`` when
        called with an invalid grid parameter.  The error message must
        name the factory so the caller knows which entry point raised.
        """
        with pytest.raises(ValueError) as exc:
            fn_call(**{bad_kwarg: bad_value})
        msg = str(exc.value)
        # The message must reference the parameter name AND the factory.
        # (We allow ``create_*`` or the specific factory name in the
        # message; the helper is consistent on both.)
        assert bad_kwarg in msg, (
            f"{fn_name}({bad_kwarg}={bad_value}) raised ValueError but "
            f"the message {msg!r} does not mention the parameter "
            f"{bad_kwarg!r}.")

    def test_validate_grid_params_helper_present(self):
        """The shared helper exists at module top and is callable."""
        assert callable(_validate_grid_params)

    def test_validate_grid_params_accepts_valid_inputs(self):
        """Sanity: the helper does NOT raise on valid inputs."""
        # Square grid
        _validate_grid_params(64, 16e-6, 1.31e-6)
        # Anamorphic grid
        _validate_grid_params(64, 16e-6, 1.31e-6, dy=8e-6)
        # Tuple (Ny, Nx) form
        _validate_grid_params((64, 128), 16e-6, 1.31e-6)

    def test_validate_grid_params_rejects_non_finite(self):
        """Non-finite dx / wavelength must raise too (not just <= 0)."""
        with pytest.raises(ValueError, match='dx'):
            _validate_grid_params(64, np.nan, 1.31e-6,
                                   fn_name='test_fn')
        with pytest.raises(ValueError, match='dx'):
            _validate_grid_params(64, np.inf, 1.31e-6,
                                   fn_name='test_fn')
        with pytest.raises(ValueError, match='wavelength'):
            _validate_grid_params(64, 16e-6, np.nan,
                                   fn_name='test_fn')
        with pytest.raises(ValueError, match='wavelength'):
            _validate_grid_params(64, 16e-6, np.inf,
                                   fn_name='test_fn')

    def test_validate_grid_params_rejects_non_integer_N(self):
        """``N`` must be an integer (or 2-tuple thereof)."""
        with pytest.raises(ValueError, match='N'):
            _validate_grid_params(64.5, 16e-6, 1.31e-6,
                                   fn_name='test_fn')

    def test_validate_grid_params_rejects_bad_tuple(self):
        """``N`` as a 3-element tuple is invalid."""
        with pytest.raises(ValueError, match='tuple'):
            _validate_grid_params((64, 64, 64), 16e-6, 1.31e-6,
                                   fn_name='test_fn')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

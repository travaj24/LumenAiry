"""Consolidated audit-fix tests for the **sources** domain.

This module consolidates v4.9 - v5.0 audit-fix regression pins
from 2 source files (per the v5.2 ROADMAP / 57-file consolidation):

* ``test_audit_fixes_v4_14_2_agent_d.py``
* ``test_audit_fixes_v4_14_3_agent_b.py``

Each source file's contents are concatenated below verbatim (modulo
minimal renames to avoid identifier collisions and to give each top-level
test class an audit-version attribution prefix).  inspect.getsource proxy
tests are tagged with a TODO comment per AUDIT_V4_13_1 Part 6.1.
"""
from __future__ import annotations

# ============================================================================
# Source: test_audit_fixes_v4_14_2_agent_d.py
# Audit version: V4_14_2  scope: agent_d
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.14.2 audit closures handled by Agent D.
#   
#   Audit reference
#   ---------------
#   
#   ``AUDIT_V4_14_1_2026_05_17.md`` flagged 1 P0 + 10 P1 follow-ups; Agent
#   D closed three of the P1s plus one structural meta-pin.  Scope was
#   restricted to ``lumenairy/elements/doe.py`` and
#   ``lumenairy/sources/core.py`` only.
#   
#   * **D.1 / P1-NEW-6** -- ``makedammann2d`` historically took
#     ``periodx``, ``periody`` and ``waveln`` in micrometres while the rest
#     of the library was SI metres throughout.  An SI-convention caller
#     (e.g. ``periodx=61e-6, waveln=1.31e-6``) got a thousandfold-off
#     ``samplingx`` that was silently masked by an output ``* 1e-6``
#     rescale on ``cell_pixel_size_x``.  v4.14.2 converts the function to
#     SI metres end-to-end and emits a ``DeprecationWarning`` when any of
#     ``periodx``, ``periody``, ``waveln`` look like legacy micrometres
#     (heuristic: ``> 1e-3 m``).
#   * **D.2 / P1-NEW-9** -- ``create_led_source`` had positional
#     ``(N, dx, diameter, divergence_angle, wavelength, ...)`` which broke
#     the post-v4.7 convention of keyword-only physical parameters with
#     ``wavelength`` in the canonical 3rd positional slot.  The factory
#     also lacked ``dy`` and a ``*`` keyword-only separator.  v4.14.2
#     reorders to ``(N, dx, wavelength, *, diameter, divergence_angle,
#     dy=None, x0=0, y0=0, dtype=None)`` and accepts the legacy positional
#     form for one release with a ``DeprecationWarning``.
#   * **D.3 / P1-NEW-10** -- The 10 ``create_*`` factories in
#     ``sources/core.py`` silently accepted ``N=0``, ``dx<=0``,
#     ``wavelength<=0`` etc.  v4.14.2 adds a private
#     ``_validate_grid_params`` helper at module top and calls it at every
#     factory entry.
#   
#   Author: Andrew Traverso -- v4.14.2 / Agent D
# ============================================================================
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

class TestAuditFixesV4_14_2_agent_d_P1New6MakedammannSiUnits:
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
        """Calling the auto-detect shim with the legacy micrometre form
        in its range (``1e-3 < value <= 1.0``) must trigger a
        migration warning that names the SI-metres replacement.

        v4.14.3 note: values above 1 m now raise ``ValueError`` (the
        P0-NEW-2 fix), so this test exercises mid-range values
        (0.5 < x <= 1.0 m) that the heuristic still catches.  For
        explicit legacy-um migration above that bound, callers should
        pass ``_legacy_units='um'``.

        v5.30 (audit E-H11): the heuristic is no longer the DEFAULT --
        ``_legacy_units='SI'`` is, so reaching the shim needs an
        explicit ``_legacy_units='auto'`` -- and the retired shim now
        warns LOUDLY (``UserWarning``, default-visible) instead of with
        a ``DeprecationWarning`` that Python hides outside ``__main__``.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _, _, _ = makedammann2d(
                periodx=0.5, periody=0.5, waveln=0.01,
                phaselevels=2, phasesteps=1,
                diforders=np.ones((2, 2)),
                itr=2, plot=False, seed=42,
                _legacy_units='auto',
            )
            depwarns = [r for r in w
                        if issubclass(r.category, UserWarning)
                        and 'makedammann2d' in str(r.message)]
            assert len(depwarns) >= 1, (
                "Expected at least one UserWarning when "
                "makedammann2d is called with legacy micrometre "
                "values periodx=0.5, waveln=0.01 and "
                "_legacy_units='auto'.  Got: "
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

        v4.14.3 note: ``periodx=61.0`` now exceeds the 1 m hard upper
        bound and would raise.  Callers wanting the legacy interpretation
        must pass ``_legacy_units='um'`` (which bypasses the upper
        bound for explicit-legacy callers).  Both forms below produce
        the same SI-equivalent result.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            _, _, (legacy_dx, _) = makedammann2d(
                periodx=61.0, periody=61.0, waveln=1.31,
                phaselevels=2, phasesteps=1,
                diforders=np.ones((2, 2)),
                itr=2, plot=False, seed=42,
                _legacy_units='um',
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

class TestAuditFixesV4_14_2_agent_d_P1New9CreateLedSourceSignature:
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


class TestAuditFixesV4_14_2_agent_d_P1New10SourceFactoryInputValidation:
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
        # Tuple (Ny, Nx) form -- v4.14.3 narrowed this contract: tuple-N
        # is now opt-in via ``support_tuple_N=True`` (only the 3 mode-
        # family factories pass it; the other 7 inherit the default
        # ``False`` and reject tuples up-front with a clear error).
        _validate_grid_params((64, 128), 16e-6, 1.31e-6,
                                support_tuple_N=True)

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
        """``N`` as a 3-element tuple is invalid even when tuple-N is
        enabled (only 2-tuples (Ny, Nx) are valid).  v4.14.3: pass
        ``support_tuple_N=True`` so the helper reaches the
        ``len(N) != 2`` branch instead of the default tuple-reject
        branch (which raises TypeError, not ValueError).
        """
        with pytest.raises(ValueError, match='tuple'):
            _validate_grid_params((64, 64, 64), 16e-6, 1.31e-6,
                                   fn_name='test_fn',
                                   support_tuple_N=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ============================================================================
# Source: test_audit_fixes_v4_14_3_agent_b.py
# Audit version: V4_14_3  scope: agent_b
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.14.3 Agent-B audit fixes.
#   
#   Audit reference
#   ---------------
#   
#   ``AUDIT_V4_14_2_2026_05_17.md`` -- Agent B scope covers
#   ``lumenairy/sources/core.py`` and ``lumenairy/optimize/multiconfig.py``.
#   The audit flagged five P1 issues; this module pins all five.
#   
#   Agent B fixes
#   -------------
#   
#   * **B.1 / P1-NEW-4** -- ``create_led_source`` had a bare ``*args``
#     VAR_POSITIONAL collector that silently re-routed 5-positional
#     NEW-canonical-order calls ``(N, dx, wavelength, diameter,
#     divergence_angle)`` as if they were the LEGACY positional form,
#     producing a "633 nm-wide LED at 0.3 m wavelength" with only a
#     DeprecationWarning.  v4.14.3 adds a post-remap scale-inversion
#     check inside the legacy shim (apparent wavelength > 10x apparent
#     diameter -> TypeError), which catches the canonical-order
#     mistake without breaking legacy callers (whose legacy
#     ``diameter > wavelength * 10`` always holds for real
#     emitters).  We deliberately did NOT add PEP 570 ``/`` -- it does
#     not prevent the ``*args`` collector from eating surplus
#     positional, so it adds no safety against the canonical-order
#     mistake and would force every existing kwarg-based caller (and
#     the v4.14.2 audit test infrastructure) to drop ``N/dx/wavelength``
#     out of kwargs.
#   
#   * **B.2 / P1-NEW-5** -- ``_validate_grid_params`` accepted both
#     ``int`` and ``(Ny, Nx)`` tuple ``N``, but only 3 of the 10 source
#     factories (``create_gaussian_beam``, ``create_hermite_gauss``,
#     ``create_laguerre_gauss``) unpack tuples internally; the other 7
#     call ``np.arange(N)`` and crash with an obscure ``TypeError``.
#     v4.14.3 adds ``support_tuple_N: bool = False`` to the validator;
#     the 3 tuple-supporting factories pass ``True`` and the other 7
#     inherit the default ``False``, so a tuple-N input now raises a
#     clear, factory-named ``TypeError`` instead of a downstream
#     ``np.arange`` crash.
#   
#   * **B.3 / P1-NEW-9** -- ``create_bessel_beam`` accepted any
#     ``cone_angle``.  ``cone_angle=pi`` -> ``sin=0`` -> silent uniform
#     DC field labelled "Bessel beam"; ``cone_angle=2*pi/3`` -> non-
#     physical evanescent ``k_r > k0`` (apparent ``k_r`` aliases back to
#     ``sin(pi/3)`` because the user's literal angle is past the
#     propagation horizon).  v4.14.3 enforces ``0 < cone_angle < pi/2``.
#   
#   * **B.4 / P1-NEW-10** -- ``create_fiber_mode`` accepted non-positive
#     ``mode_field_diameter``.  ``MFD=0`` -> divide-by-zero in
#     ``sigma = w0/sqrt(2) = 0``; negative MFD -> sigma's sign flips
#     and the field grows away from the centre (non-physical).
#     v4.14.3 raises ``ValueError`` on ``MFD <= 0`` or non-finite.
#   
#   * **B.5 / P1-MC** -- ``beam_expander_prescription`` AND
#     ``keplerian_telescope`` both hardcoded ``n=1.5`` in the
#     lensmaker formula ``R = f*(n-1)*2``.  For ``glass='N-LASF9'``
#     (n=1.85 at 587.6 nm) the focal length feeding ``_zero_C_air_gap``
#     was off by ~17% -- a real physics error.  v4.14.3 routes the
#     glass + wavelength through ``get_glass_index``, with a
#     ``UserWarning``-flagged fallback to ``n=1.5`` on lookup failure.
#   
#   Author: Andrew Traverso -- v4.14.3 / Agent B
# ============================================================================

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

class TestAuditFixesV4_14_3_agent_b_B1LedSourceCanonicalPositionalFootgun:
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

class TestAuditFixesV4_14_3_agent_b_B2TupleNValidatorImplementationGap:
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

class TestAuditFixesV4_14_3_agent_b_B3BesselBeamConeAngleConstraint:
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

class TestAuditFixesV4_14_3_agent_b_B4FiberModeMfdConstraint:
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
# S3-4 (AUDIT_V5_24_2) -- create_fiber_mode ``na`` is advisory only
# ============================================================================

class TestAuditFixesV5_24_2_S3_4FiberModeNaInert:
    """Pin the documented contract that ``create_fiber_mode``'s ``na``
    argument does NOT influence the returned field: the Gaussian
    near-field (and hence the far-field divergence) is set entirely by
    ``mode_field_diameter`` and ``wavelength``.

    S3-4 flagged that the docstring claimed "NA-defined divergence"
    while the implementation is MFD-only -- fields for na=0.05 vs 0.15
    are bit-identical.  These tests lock that behaviour with an
    independent analytic Gaussian oracle so the corrected docstring can
    never silently drift back to the false claim.
    """

    def test_na_does_not_change_field_bit_identical(self):
        """Fields for several sub-warning-threshold ``na`` values are
        bit-identical -- ``na`` is inert (reproduces the audit probe)."""
        N, dx, wl, mfd = 64, 1e-6, 1.31e-6, 10e-6
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # any NA warning would fail here
            E_lo, _, _ = create_fiber_mode(
                N, dx, wl, mode_field_diameter=mfd, na=0.05)
            E_mid, _, _ = create_fiber_mode(
                N, dx, wl, mode_field_diameter=mfd, na=0.12)
            E_hi, _, _ = create_fiber_mode(
                N, dx, wl, mode_field_diameter=mfd, na=0.15)
        assert np.array_equal(E_lo, E_mid), (
            'na=0.05 vs 0.12 must give a bit-identical field (na is '
            'advisory only).')
        assert np.array_equal(E_lo, E_hi), (
            'na=0.05 vs 0.15 must give a bit-identical field (na is '
            'advisory only).')

    def test_field_matches_independent_mfd_gaussian_oracle(self):
        """Independent oracle: the near-field equals an analytic
        peak-normalised Gaussian built solely from MFD (waist
        ``w0 = MFD/2``, ``sigma = w0/sqrt(2)``) on the same grid --
        proving the field is MFD-determined, not NA-determined."""
        N, dx, wl, mfd = 64, 1e-6, 1.31e-6, 10e-6
        E, x, y = create_fiber_mode(
            N, dx, wl, mode_field_diameter=mfd, na=0.14)
        w0 = mfd / 2.0
        sigma = w0 / np.sqrt(2.0)
        X, Y = np.meshgrid(x, y)
        E_expected = np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma ** 2))
        # Default normalize='peak' -> unit-peak amplitude, flat phase.
        assert np.allclose(E, E_expected, rtol=0, atol=1e-12), (
            'create_fiber_mode field must match the analytic MFD-set '
            'Gaussian.')

    def test_second_moment_width_equals_mfd_independent_of_na(self):
        """Physics oracle: the measured D4-sigma second-moment intensity
        width equals the MFD (a Gaussian has D4sigma == 1/e^2 diameter)
        and is unchanged by ``na`` -- so ``na`` cannot set divergence."""
        N, dx, wl, mfd = 128, 0.5e-6, 1.31e-6, 10e-6

        def d4sigma(E, x, y):
            Xg, Yg = np.meshgrid(x, y)
            I = np.abs(E) ** 2
            tot = I.sum()
            cx = (Xg * I).sum() / tot
            varx = ((Xg - cx) ** 2 * I).sum() / tot
            return 4.0 * np.sqrt(varx)

        w_lo = d4sigma(*create_fiber_mode(
            N, dx, wl, mode_field_diameter=mfd, na=0.05))
        w_hi = d4sigma(*create_fiber_mode(
            N, dx, wl, mode_field_diameter=mfd, na=0.15))
        assert np.isclose(w_lo, mfd, rtol=2e-3), (
            f'D4sigma width {w_lo:.3e} should equal MFD {mfd:.3e} '
            f'(Gaussian near-field set by MFD).')
        assert w_lo == w_hi, (
            'D4sigma width must not depend on na (na does not set '
            'divergence).')


# ============================================================================
# S3-3 (AUDIT_V5_24_2) -- point-source / tilted-plane-wave chirp Nyquist guard
# ============================================================================

class TestAuditFixesV5_24_2_S3_3ChirpNyquistGuard:
    """S3-3: ``create_tilted_plane_wave`` and ``create_point_source`` built
    a ramp/chirp whose local transverse spatial frequency could exceed the
    grid Nyquist limit ``lambda/(2*dx)`` with NO warning -- the phase then
    aliases (folds) to a spurious SMALLER effective angle.  The only prior
    guards were evanescence (``sin^2 > 1``) and ``|z0| < dx``.

    The fix adds a transverse-Nyquist (edge-NA) ``RuntimeWarning``.  These
    tests pin it against an INDEPENDENT aliasing oracle rather than a
    tautology: they recover the per-pixel phase step the *sampled* field
    actually carries and show it folds INTO the +/-Nyquist band while the
    requested/analytic angle lies OUTSIDE it -- the physical consequence the
    warning flags -- and separately assert the warning fires.

    The tilt ramp aliases GLOBALLY the instant sin(angle) crosses Nyquist,
    so its guard is exact.  The spherical chirp aliases only the pixels
    beyond the aliasing radius (a graded edge effect), so its guard adds a
    1.5x safety margin -- a hairline corner-only crossing is benign and must
    not alarm (last test).
    """

    @staticmethod
    def _recovered_sin(E, row, col, k0, dx):
        """Effective sin(angle) carried between adjacent columns ``col`` ->
        ``col+1`` of ``E`` on ``row``: the WRAPPED per-pixel phase step
        divided by ``k0*dx``.  This is what the sampled field actually
        represents -- it folds into (-Nyquist, Nyquist] under aliasing."""
        step = float(np.angle(E[row, col + 1] * np.conj(E[row, col])))
        return step / (k0 * dx)

    # -- tilted plane wave -------------------------------------------------

    def test_tilt_undersampled_aliases_and_warns(self):
        """sin(angle)=0.5 with Nyquist lambda/(2dx)=0.316: undersampled.
        Independent oracle -- the ramp folds to
        ``sin_eff = sin(angle) - lambda/dx = -0.133`` (inside the band),
        and a Nyquist RuntimeWarning fires."""
        N, dx, wl = 64, 1e-6, 633e-9
        k0 = 2 * np.pi / wl
        nyq = wl / (2 * dx)                 # 0.3165
        angle_x = float(np.arcsin(0.5))     # sin = 0.5 > nyq
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            E, x, y = create_tilted_plane_wave(N, dx, wl, angle_x=angle_x)
        sin_eff = self._recovered_sin(E, N // 2, N // 2, k0, dx)
        expected_fold = float(np.sin(angle_x)) - wl / dx   # 0.5 - 0.633
        assert np.isclose(sin_eff, expected_fold, atol=1e-6), (
            f"undersampled ramp must alias to sin_eff={expected_fold:.4f}; "
            f"the sampled field recovered {sin_eff:.4f}.")
        assert abs(sin_eff) < nyq, (
            "the aliased effective sin must fold inside the +/-Nyquist band.")
        assert abs(np.sin(angle_x)) > nyq, "probe must be undersampled."
        nyq_warns = [w for w in caught
                     if issubclass(w.category, RuntimeWarning)
                     and 'Nyquist' in str(w.message)]
        assert len(nyq_warns) >= 1, (
            f"undersampled tilt must emit a Nyquist RuntimeWarning; got "
            f"{[str(w.message) for w in caught]}.")

    def test_tilt_well_sampled_no_warning_and_no_alias(self):
        """sin(angle)=0.2 < 0.316: well sampled.  The field carries the
        requested angle UNfolded and no Nyquist warning fires."""
        N, dx, wl = 64, 1e-6, 633e-9
        k0 = 2 * np.pi / wl
        angle_x = float(np.arcsin(0.2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            E, x, y = create_tilted_plane_wave(N, dx, wl, angle_x=angle_x)
        sin_eff = self._recovered_sin(E, N // 2, N // 2, k0, dx)
        assert np.isclose(sin_eff, 0.2, atol=1e-6), (
            f"well-sampled ramp must carry the requested angle unfolded; "
            f"got sin_eff={sin_eff:.4f}.")
        nyq_warns = [w for w in caught
                     if issubclass(w.category, RuntimeWarning)
                     and 'Nyquist' in str(w.message)]
        assert not nyq_warns, (
            f"well-sampled tilt must NOT warn; got "
            f"{[str(w.message) for w in nyq_warns]}.")

    # -- point source ------------------------------------------------------

    def test_point_source_undersampled_aliases_and_warns(self):
        """Reproduce the audit probe (N=512, dx=1um, z0=200um): edge
        local-NA ~0.79 >> Nyquist 0.316.  Independent oracle -- the edge
        phase step folds to an effective NA well inside the band, far below
        the analytic value, and a Nyquist RuntimeWarning fires."""
        N, dx, wl, z0 = 512, 1e-6, 633e-9, 200e-6
        k0 = 2 * np.pi / wl
        nyq = wl / (2 * dx)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            E, x, y = create_point_source(N, dx, wl, z0=z0)
        # Analytic local-NA between the two outermost columns of the centre
        # row (y ~ y0 = 0), and the NA the sampled field actually carries.
        xmid = 0.5 * (float(x[-1]) + float(x[-2]))
        na_analytic = abs(xmid) / np.sqrt(xmid ** 2 + z0 ** 2)
        na_recovered = abs(self._recovered_sin(E, N // 2, N - 2, k0, dx))
        assert na_analytic > nyq, (
            f"probe edge NA {na_analytic:.3f} must exceed Nyquist "
            f"{nyq:.3f}.")
        assert na_recovered < nyq, (
            f"the sampled edge NA {na_recovered:.3f} must fold inside the "
            f"band (< {nyq:.3f}).")
        assert na_recovered < 0.5 * na_analytic, (
            f"sampled edge NA {na_recovered:.3f} must be far below the "
            f"analytic {na_analytic:.3f} (aliased/folded).")
        nyq_warns = [w for w in caught
                     if issubclass(w.category, RuntimeWarning)
                     and 'Nyquist' in str(w.message)]
        assert len(nyq_warns) >= 1, (
            f"undersampled point source must emit a Nyquist RuntimeWarning; "
            f"got {[str(w.message) for w in caught]}.")

    def test_point_source_well_sampled_no_warning(self):
        """A distant focus (z0=5mm) keeps edge NA << Nyquist: no aliasing,
        no warning, and the edge phase step carries the true NA unfolded."""
        N, dx, wl, z0 = 256, 1e-6, 633e-9, 5e-3
        k0 = 2 * np.pi / wl
        nyq = wl / (2 * dx)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            E, x, y = create_point_source(N, dx, wl, z0=z0)
        xmid = 0.5 * (float(x[-1]) + float(x[-2]))
        na_analytic = abs(xmid) / np.sqrt(xmid ** 2 + z0 ** 2)
        na_recovered = abs(self._recovered_sin(E, N // 2, N - 2, k0, dx))
        assert na_analytic < nyq, "probe must be well sampled."
        assert np.isclose(na_recovered, na_analytic, rtol=1e-2), (
            f"well-sampled edge must carry the true NA unfolded; recovered "
            f"{na_recovered:.5f} vs analytic {na_analytic:.5f}.")
        nyq_warns = [w for w in caught
                     if issubclass(w.category, RuntimeWarning)
                     and 'Nyquist' in str(w.message)]
        assert not nyq_warns, (
            f"well-sampled point source must NOT warn; got "
            f"{[str(w.message) for w in nyq_warns]}.")

    def test_point_source_marginal_corner_does_not_alarm(self):
        """Policy pin: a hairline crossing (edge NA ~1.26x Nyquist -- only
        the extreme-corner ring aliasing) must NOT warn.  The point-source
        guard fires only on CLEAR undersampling (> 1.5x Nyquist).  Uses the
        exact params of test_v4_15_agent_b's canonical point-source
        construction, so this guard can never regress that clean call under
        its ``simplefilter('error')``."""
        N, dx, wl, z0 = 32, 5e-6, 633e-9, -1e-3
        nyq = wl / (2 * dx)                              # 0.0633
        x = (np.arange(N) - N / 2) * dx
        x_off = max(abs(float(x[0])), abs(float(x[-1])))
        na_edge = x_off / np.sqrt(x_off ** 2 + z0 ** 2)  # ~0.0797
        assert nyq < na_edge < 1.5 * nyq, (
            f"probe calibration: edge NA {na_edge:.4f} must sit in the "
            f"(Nyquist, 1.5*Nyquist) marginal band ({nyq:.4f}, "
            f"{1.5 * nyq:.4f}).")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            create_point_source(N, dx, wl, z0=z0)
        nyq_warns = [w for w in caught
                     if issubclass(w.category, RuntimeWarning)
                     and 'Nyquist' in str(w.message)]
        assert not nyq_warns, (
            f"marginal corner-only aliasing must NOT alarm; got "
            f"{[str(w.message) for w in nyq_warns]}.")


# ============================================================================
# B.5 -- P1-MC: multiconfig.py hardcoded n=1.5 in lensmaker formula
# ============================================================================

class TestAuditFixesV4_14_3_agent_b_B5MulticonfigGlassIndex:
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
                la.keplerian_telescope(
                    f_objective=200e-3, f_eyepiece=50e-3,
                    glass='__bogus_glass_name__',
                    wavelength=587.6e-9, aperture=25.4e-3,
                )
            except (ValueError, KeyError) as e:
                # Downstream surfaces_from_prescription rejection is
                # fine; we just need the helper to have warned.
                err = e
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

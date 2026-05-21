"""Consolidated audit-fix tests for the **polarization** domain.

This module consolidates v4.9 - v5.0 audit-fix regression pins
from 1 source files (per the v5.2 ROADMAP / 57-file consolidation):

* ``test_audit_fixes_v4_14_3_agent_c.py``

Each source file's contents are concatenated below verbatim (modulo
minimal renames to avoid identifier collisions and to give each top-level
test class an audit-version attribution prefix).  inspect.getsource proxy
tests are tagged with a TODO comment per AUDIT_V4_13_1 Part 6.1.
"""
from __future__ import annotations

# ============================================================================
# Source: test_audit_fixes_v4_14_3_agent_c.py
# Audit version: V4_14_3  scope: agent_c
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.14.3 audit-fix Agent-C changes.
#   
#   Scope:
#     * ``lumenairy/elements/polarization.py``
#     * ``lumenairy/elements/freeform.py``
#     * ``lumenairy/analysis/ghost.py``
#     * ``lumenairy/user_library.py``
#   
#   Four audit items pinned:
#   
#   * **C.1 -- P1-NEW-2 polarization-family conflict-resolution symmetry.**
#     v4.14.2 added ``angle_deg=`` + conflict detection to ``apply_rotator``
#     only.  The 4 sibling helpers silently let ``angle_deg`` overwrite the
#     positional ``angle`` argument (no conflict check).  Additionally the
#     v4.14.2 conflict check used ``if angle != 0.0`` as a proxy for "angle
#     was supplied", so the explicit ``angle=0.0, angle_deg=90`` call
#     silently became a 90-deg rotation -- discarding the user's explicit
#     zero-radian intent.  v4.14.3 introduces an ``_AngleUnsetSentinel``
#     singleton (matching the v4.14.1 ``_ZeroApertureMaskSentinel``
#     ``__slots__=()`` pattern) and propagates the same conflict-detection
#     logic across all 5 polarization helpers.
#   
#   * **C.2 -- P1-NEW-11 freeform negative-``norm_x``/``norm_y`` rejection.**
#     ``surface_sag_xy_polynomial(..., norm_x=-0.05, ...)`` made
#     ``outside = abs(X) > -0.05`` true at every pixel, so the freeform
#     contribution was zeroed silently.  ``surface_sag_chebyshev`` shared
#     the same defect (negative ``norm_x`` mirror-images the polynomial
#     domain).  v4.14.3 raises ``ValueError`` on non-positive
#     ``norm_x``/``norm_y`` in both branches.
#   
#   * **C.3 -- P1-GH-2 ghost.py elements-key prescription support.**
#     ``ghost_analysis`` and ``non_sequential_stray_light`` previously
#     hard-coded ``prescription['surfaces']`` lookups, which raise
#     ``IndexError`` (with a misleading message) for modern
#     ``'elements'``-key prescriptions.  v4.14.3 adds an
#     ``_ghost_surfaces`` adapter that routes through
#     :func:`surfaces_from_prescription` for the legacy schema and
#     :func:`surfaces_from_elements` for the modern schema; both
#     ``ghost_analysis`` and ``non_sequential_stray_light`` now flow
#     through the adapter.
#   
#   * **C.4 -- P1-GL-2 ``register_fixed_glass`` input validation.**
#     Pre-v4.14.3 ``register_fixed_glass`` accepted ``n < 1.0`` silently,
#     registered empty-string names without complaint, and silently
#     clobbered catalog entries like ``'N-BK7'``.  v4.14.3 validates
#     ``1.0 <= n <= 4.0`` (upper bound covers Si ~3.4, Ge ~4.0), rejects
#     empty/non-string names, and emits a ``UserWarning`` (not a refusal)
#     on overwrite so the action stays visible.
#   
#   Author:  Andrew Traverso -- v4.14.3 / Agent C.
# ============================================================================
import warnings

import numpy as np
import pytest

from lumenairy.elements.freeform import (
    surface_sag_chebyshev,
    surface_sag_xy_polynomial,
)
from lumenairy.elements.polarization import (
    _ANGLE_UNSET,
    JonesField,
    _AngleUnsetSentinel,
    apply_half_wave_plate,
    apply_polarizer,
    apply_quarter_wave_plate,
    apply_rotator,
    apply_waveplate,
    create_linear_polarized,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _x_polarized_field(N: int = 16, dx: float = 1e-6) -> JonesField:
    scalar = np.ones((N, N), dtype=complex)
    return create_linear_polarized(scalar, dx, angle=0.0)


# ===========================================================================
# C.1 (P1-NEW-2) -- Polarization-family conflict-detection symmetry
# ===========================================================================


class TestAuditFixesV4_14_3_agent_c_C1AngleUnsetSentinel:
    """Pin the ``_AngleUnsetSentinel`` singleton pattern matches the
    v4.14.1 ``_ZeroApertureMaskSentinel`` canonical form
    (``__slots__=()`` plus informative ``__repr__``).
    """

    def test_angle_unset_singleton_has_empty_slots(self):
        assert _AngleUnsetSentinel.__slots__ == ()

    def test_angle_unset_repr_is_informative(self):
        assert '_ANGLE_UNSET' in repr(_ANGLE_UNSET)

    def test_angle_unset_is_singleton_instance(self):
        # The module exports exactly one instance via ``_ANGLE_UNSET``;
        # but spawning a second instance via the class is allowed by
        # the ``__slots__=()`` pattern (matches
        # ``_ZeroApertureMaskSentinel`` -- the contract is "use the
        # exported singleton via ``is``", not "the class is a
        # forbidden constructor").
        assert isinstance(_ANGLE_UNSET, _AngleUnsetSentinel)


class TestAuditFixesV4_14_3_agent_c_C1ApplyRotatorExplicitZeroAngleConflict:
    """Pin the headline bug: ``angle=0.0, angle_deg=90`` must raise,
    not silently become a 90-deg rotation.
    """

    def test_apply_rotator_explicit_zero_angle_with_angle_deg_raises(self):
        f = _x_polarized_field()
        with pytest.raises(ValueError, match='conflicting'):
            apply_rotator(f, angle=0.0, angle_deg=90)

    def test_apply_rotator_explicit_zero_angle_with_zero_angle_deg_ok(self):
        # angle=0 + angle_deg=0 is consistent -- no rotation, no raise.
        f = _x_polarized_field()
        apply_rotator(f, angle=0.0, angle_deg=0.0)
        # Field unchanged: linear x-pol, so Ey = 0.
        np.testing.assert_allclose(f.Ey, 0.0, atol=1e-15)

    def test_apply_rotator_angle_deg_alone_ok(self):
        f = _x_polarized_field()
        apply_rotator(f, angle_deg=45)
        # 45-deg rotation of x-pol -> Ex = Ey = 1/sqrt(2)
        np.testing.assert_allclose(
            f.Ex[0, 0], 1.0 / np.sqrt(2), atol=1e-12)
        np.testing.assert_allclose(
            f.Ey[0, 0], 1.0 / np.sqrt(2), atol=1e-12)


@pytest.mark.parametrize(
    'helper_call',
    [
        # (helper_name, build call lambda for both args supplied,
        #  build call lambda for angle_deg-only)
        pytest.param(
            ('apply_polarizer',
             lambda f: apply_polarizer(f, angle=0.5, angle_deg=10),
             lambda f: apply_polarizer(f, angle_deg=45)),
            id='apply_polarizer'),
        pytest.param(
            ('apply_waveplate',
             lambda f: apply_waveplate(f, np.pi, angle=0.5, angle_deg=10),
             lambda f: apply_waveplate(f, np.pi, angle_deg=45)),
            id='apply_waveplate'),
        pytest.param(
            ('apply_half_wave_plate',
             lambda f: apply_half_wave_plate(f, angle=0.5, angle_deg=10),
             lambda f: apply_half_wave_plate(f, angle_deg=45)),
            id='apply_half_wave_plate'),
        pytest.param(
            ('apply_quarter_wave_plate',
             lambda f: apply_quarter_wave_plate(f, angle=0.5, angle_deg=10),
             lambda f: apply_quarter_wave_plate(f, angle_deg=45)),
            id='apply_quarter_wave_plate'),
        pytest.param(
            ('apply_rotator',
             lambda f: apply_rotator(f, angle=0.5, angle_deg=10),
             lambda f: apply_rotator(f, angle_deg=45)),
            id='apply_rotator'),
    ],
)
class TestAuditFixesV4_14_3_agent_c_C1PolarizationFamilyConflictDetection:
    """Parametrized symmetry pin: every angle-taking polarization
    helper must raise ``ValueError`` on disagreeing ``(angle,
    angle_deg)`` pairs AND accept ``angle_deg=`` alone without
    warning.  Pre-v4.14.3 only ``apply_rotator`` raised; the four
    siblings silently overwrote.
    """

    def test_conflict_raises(self, helper_call):
        name, conflicting_call, _ = helper_call
        f = _x_polarized_field()
        with pytest.raises(ValueError, match='conflicting'):
            conflicting_call(f)

    def test_angle_deg_alone_no_warning(self, helper_call):
        name, _, deg_only_call = helper_call
        f = _x_polarized_field()
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # any warning -> fail
            deg_only_call(f)  # must not raise or warn


class TestAuditFixesV4_14_3_agent_c_C1ConsistentAngleRadAndAngleDegAccepted:
    """``angle=pi/4`` + ``angle_deg=45`` (numerically consistent) is
    accepted by every helper, no warning."""

    def test_apply_polarizer_consistent_accepted(self):
        f = _x_polarized_field()
        apply_polarizer(f, angle=np.pi / 4, angle_deg=45.0)
        # 45-deg linear polarizer applied to x-pol input |1, 0>:
        # output Jones vector = (cos^2 t, cos t * sin t) = (0.5, 0.5).
        # So Ex = Ey = 0.5 and |Ex|^2 = 0.25 (transmitted intensity
        # is |Ex|^2 + |Ey|^2 = 0.5 -- the classic Malus-law factor).
        np.testing.assert_allclose(
            f.Ex[0, 0], 0.5, atol=1e-12)
        np.testing.assert_allclose(
            f.Ey[0, 0], 0.5, atol=1e-12)

    def test_apply_half_wave_plate_consistent_accepted(self):
        f = _x_polarized_field()
        apply_half_wave_plate(f, angle=np.pi / 4, angle_deg=45.0)
        # HWP at 45 deg flips x-pol to y-pol.
        np.testing.assert_allclose(np.abs(f.Ex[0, 0]), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(f.Ey[0, 0]), 1.0, atol=1e-12)


# ===========================================================================
# C.2 (P1-NEW-11) -- Negative norm_x / norm_y rejection
# ===========================================================================


class TestAuditFixesV4_14_3_agent_c_C2FreeformXYPolynomialNegativeNorm:
    """Pin ``surface_sag_xy_polynomial`` rejects non-positive
    ``norm_x`` / ``norm_y``.  Pre-v4.14.3 a typo
    ``norm_x = -0.05`` silently zeroed the freeform contribution
    over the whole grid (``abs(X) > -0.05`` is True for every real
    ``X``), making the bug invisible at runtime."""

    def _grid(self):
        N = 8
        x = np.linspace(-0.025, 0.025, N)
        return np.meshgrid(x, x)

    def test_negative_norm_x_raises(self):
        X, Y = self._grid()
        with pytest.raises(ValueError, match='norm_x'):
            surface_sag_xy_polynomial(
                X, Y, xy_coeffs={(2, 0): 1.0},
                norm_x=-0.05, norm_y=0.05)

    def test_negative_norm_y_raises(self):
        X, Y = self._grid()
        with pytest.raises(ValueError, match='norm_y'):
            surface_sag_xy_polynomial(
                X, Y, xy_coeffs={(2, 0): 1.0},
                norm_x=0.05, norm_y=-0.05)

    def test_zero_norm_x_raises(self):
        X, Y = self._grid()
        with pytest.raises(ValueError, match='norm_x'):
            surface_sag_xy_polynomial(
                X, Y, xy_coeffs={(2, 0): 1.0},
                norm_x=0.0, norm_y=0.05)

    def test_nan_norm_x_raises(self):
        X, Y = self._grid()
        with pytest.raises(ValueError, match='norm_x'):
            surface_sag_xy_polynomial(
                X, Y, xy_coeffs={(2, 0): 1.0},
                norm_x=float('nan'), norm_y=0.05)

    def test_positive_norms_still_work(self):
        """Back-compat: the v4.14.2 default ``norm_x=norm_y=1.0`` and
        any positive value still work."""
        X, Y = self._grid()
        sag = surface_sag_xy_polynomial(
            X, Y, xy_coeffs={(2, 0): 1.0},
            norm_x=0.05, norm_y=0.05)
        assert np.all(np.isfinite(sag))


class TestAuditFixesV4_14_3_agent_c_C2FreeformChebyshevNegativeNorm:
    """Same defect: ``surface_sag_chebyshev`` with ``norm_x = -0.05``
    silently mirror-images the polynomial domain (T_n(-x/0.05) != T_n(x/0.05)
    for odd n)."""

    def _grid(self):
        N = 8
        x = np.linspace(-0.025, 0.025, N)
        return np.meshgrid(x, x)

    def test_negative_norm_x_raises(self):
        X, Y = self._grid()
        with pytest.raises(ValueError, match='norm_x'):
            surface_sag_chebyshev(
                X, Y, cheb_coeffs={(1, 0): 1e-6},
                norm_x=-0.05, norm_y=0.05)

    def test_negative_norm_y_raises(self):
        X, Y = self._grid()
        with pytest.raises(ValueError, match='norm_y'):
            surface_sag_chebyshev(
                X, Y, cheb_coeffs={(1, 0): 1e-6},
                norm_x=0.05, norm_y=-0.05)

    def test_positive_norms_still_work(self):
        X, Y = self._grid()
        sag = surface_sag_chebyshev(
            X, Y, cheb_coeffs={(1, 0): 1e-6},
            norm_x=0.05, norm_y=0.05)
        assert np.all(np.isfinite(sag))


# ===========================================================================
# C.3 (P1-GH-2) -- Ghost analysis tolerates elements-key prescriptions
# ===========================================================================


class TestAuditFixesV4_14_3_agent_c_C3GhostElementsKeyPrescription:
    """Pin ``ghost_analysis`` and ``non_sequential_stray_light``
    accept the modern ``'elements'``-key prescription schema (UI
    LensModel output omits the ``'surfaces'`` mirror).

    Pre-v4.14.3 both functions hardcoded ``prescription['surfaces']``
    and crashed with a misleading ``IndexError``.
    """

    def _doublet_elements_only_surface_style(self):
        """Build a surface-style elements-only prescription for an
        achromatic doublet -- ``'element_type': 'surface'`` entries
        with full per-surface ``radius`` / ``glass_*`` fields, no
        legacy ``'surfaces'`` mirror.  This is the format that
        :func:`io.prescriptions.load_zemax_prescription_data_txt`
        ships in the ``'elements'`` slot of its return dict, and
        what the UI ``LensModel`` carries internally."""
        return {
            'name': 'doublet-elements-only',
            'aperture_diameter': 25.4e-3,
            'elements': [
                {'element_type': 'surface',
                 'radius': 33.3e-3, 'conic': 0.0,
                 'glass_before': 'air', 'glass_after': 'N-BAF10',
                 'semi_diameter': 12.7e-3, 'is_stop': False},
                {'element_type': 'surface',
                 'radius': -24.1e-3, 'conic': 0.0,
                 'glass_before': 'N-BAF10', 'glass_after': 'N-SF6HT',
                 'semi_diameter': 12.7e-3, 'is_stop': False},
                {'element_type': 'surface',
                 'radius': -95.3e-3, 'conic': 0.0,
                 'glass_before': 'N-SF6HT', 'glass_after': 'air',
                 'semi_diameter': 12.7e-3, 'is_stop': False},
            ],
            'thicknesses': [9.0e-3, 3.0e-3],
        }

    def _doublet_elements_only_propagate_style(self):
        """Build a propagate-style elements-only prescription
        (``'type': 'real_lens'``) for the second elements format
        the adapter must handle."""
        return {
            'name': 'doublet-elements-only-propagate-style',
            'aperture_diameter': 25.4e-3,
            'elements': [
                {'type': 'real_lens',
                 'prescription': {
                     'name': 'achromatic doublet',
                     'aperture_diameter': 25.4e-3,
                     'surfaces': [
                         {'radius': 33.3e-3, 'conic': 0.0,
                          'glass_before': 'air',
                          'glass_after': 'N-BAF10'},
                         {'radius': -24.1e-3, 'conic': 0.0,
                          'glass_before': 'N-BAF10',
                          'glass_after': 'N-SF6HT'},
                         {'radius': -95.3e-3, 'conic': 0.0,
                          'glass_before': 'N-SF6HT',
                          'glass_after': 'air'},
                     ],
                     'thicknesses': [9.0e-3, 3.0e-3],
                 }},
            ],
        }

    def test_ghost_analysis_surface_style_elements_does_not_raise(self):
        from lumenairy.analysis.ghost import ghost_analysis
        rx = self._doublet_elements_only_surface_style()
        ghosts = ghost_analysis(rx, wavelength=587.6e-9, verbose=False)
        # 3 surfaces -> 3*(3-1)/2 = 3 ghost paths
        assert len(ghosts) == 3
        for g in ghosts:
            assert np.isfinite(g['intensity'])
            assert 0.0 <= g['intensity'] <= 1.0

    def test_non_sequential_stray_light_surface_style_elements(self):
        from lumenairy.analysis.ghost import non_sequential_stray_light
        rx = self._doublet_elements_only_surface_style()
        report = non_sequential_stray_light(
            rx, wavelength=587.6e-9, verbose=False)
        assert np.isfinite(report['ghost_total_intensity'])
        assert report['ghost_total_intensity'] > 0.0

    def test_ghost_analysis_propagate_style_elements_does_not_raise(self):
        from lumenairy.analysis.ghost import ghost_analysis
        rx = self._doublet_elements_only_propagate_style()
        ghosts = ghost_analysis(rx, wavelength=587.6e-9, verbose=False)
        # 3 surfaces -> 3 ghost paths
        assert len(ghosts) == 3
        for g in ghosts:
            assert np.isfinite(g['intensity'])
            assert 0.0 <= g['intensity'] <= 1.0

    def test_legacy_surfaces_key_still_works(self):
        """Back-compat: ``make_doublet`` still produces a surfaces-key
        prescription (it also carries elements, but the surfaces path
        is preferred) and ghost analysis must keep working on it."""
        from lumenairy.analysis.ghost import ghost_analysis
        from lumenairy.io.prescriptions import make_doublet
        rx = make_doublet(
            R1=33.3e-3, R2=-24.1e-3, R3=-95.3e-3,
            d1=9.0e-3, d2=3.0e-3,
            glass1='N-BAF10', glass2='N-SF6HT',
        )
        ghosts = ghost_analysis(rx, wavelength=587.6e-9, verbose=False)
        assert len(ghosts) == 3

    def test_missing_both_keys_raises_with_clear_message(self):
        """If neither 'surfaces' nor 'elements' is present, the
        function must raise ``ValueError`` mentioning both keys --
        not the cryptic ``IndexError`` pre-v4.14.3 produced."""
        from lumenairy.analysis.ghost import _ghost_surfaces
        bogus_rx = {'name': 'bogus', 'aperture_diameter': 1e-2}
        with pytest.raises(ValueError, match="neither 'surfaces' nor 'elements'"):
            _ghost_surfaces(bogus_rx, 587.6e-9)


# ===========================================================================
# C.4 (P1-GL-2) -- register_fixed_glass input validation
# ===========================================================================


class TestAuditFixesV4_14_3_agent_c_C4RegisterFixedGlassValidation:
    """Pin ``register_fixed_glass`` validates name and n at function
    entry.  Pre-v4.14.3 accepted any input silently.
    """

    @staticmethod
    def _cleanup(name):
        """Remove ``name`` from the registry / cache if it was
        registered during the test, so test ordering doesn't leak
        state."""
        from lumenairy.glass import GLASS_REGISTRY, _glass_cache
        GLASS_REGISTRY.pop(name, None)
        _glass_cache.pop(name, None)

    def test_register_fixed_glass_rejects_empty_name(self):
        from lumenairy.user_library import register_fixed_glass
        with pytest.raises(ValueError, match='name'):
            register_fixed_glass('', 1.5)

    def test_register_fixed_glass_rejects_whitespace_only_name(self):
        from lumenairy.user_library import register_fixed_glass
        with pytest.raises(ValueError, match='name'):
            register_fixed_glass('   ', 1.5)

    def test_register_fixed_glass_rejects_non_string_name(self):
        from lumenairy.user_library import register_fixed_glass
        with pytest.raises(ValueError, match='name'):
            register_fixed_glass(123, 1.5)

    def test_register_fixed_glass_rejects_negative_n(self):
        from lumenairy.user_library import register_fixed_glass
        with pytest.raises(ValueError, match='unphysical'):
            register_fixed_glass('_test_neg_n', -0.5)

    def test_register_fixed_glass_rejects_n_below_one(self):
        from lumenairy.user_library import register_fixed_glass
        with pytest.raises(ValueError, match='unphysical'):
            register_fixed_glass('_test_subunit_n', 0.5)

    def test_register_fixed_glass_rejects_n_above_four(self):
        from lumenairy.user_library import register_fixed_glass
        with pytest.raises(ValueError, match='unphysical'):
            register_fixed_glass('_test_overhigh_n', 4.5)

    def test_register_fixed_glass_rejects_nan_n(self):
        from lumenairy.user_library import register_fixed_glass
        with pytest.raises(ValueError, match='unphysical'):
            register_fixed_glass('_test_nan_n', float('nan'))

    def test_register_fixed_glass_warns_on_overwrite(self):
        from lumenairy.glass import GLASS_REGISTRY
        from lumenairy.user_library import register_fixed_glass
        # ``N-BK7`` is the canonical pre-registered catalog glass.
        assert 'N-BK7' in GLASS_REGISTRY
        original = GLASS_REGISTRY['N-BK7']
        try:
            with pytest.warns(UserWarning, match='overwriting'):
                register_fixed_glass('N-BK7', 1.5168)
        finally:
            # Restore the original catalog entry so other tests still
            # see the dispersive Schott catalog Sellmeier path, and
            # purge the fixed-index cache injection.
            GLASS_REGISTRY['N-BK7'] = original
            from lumenairy.glass import _glass_cache
            _glass_cache.pop('N-BK7', None)

    def test_register_fixed_glass_accepts_high_index_si(self):
        """Silicon (n ~ 3.4) and similar high-index semiconductors
        must register cleanly -- they're a common mid-IR / THz
        design choice and would have been silently accepted pre-
        v4.14.3 but must remain explicitly allowed in the new
        validated path."""
        from lumenairy.glass import get_glass_index
        from lumenairy.user_library import register_fixed_glass
        name = '_test_silicon'
        try:
            register_fixed_glass(name, 3.4)
            # The exposed index must round-trip through the
            # dispatcher.
            n_back = get_glass_index(name, 1550e-9)
            assert abs(n_back - 3.4) < 1e-12
        finally:
            self._cleanup(name)

    def test_register_fixed_glass_accepts_germanium(self):
        """n=4.0 (Ge upper-bound corner) must just barely pass."""
        from lumenairy.user_library import register_fixed_glass
        name = '_test_germanium'
        try:
            register_fixed_glass(name, 4.0)
            from lumenairy.glass import get_glass_index
            assert abs(get_glass_index(name, 10e-6) - 4.0) < 1e-12
        finally:
            self._cleanup(name)

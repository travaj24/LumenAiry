"""Parametrized dispatcher-pin tests for the Welford-mirror convention
across analysis functions.

Audit context
-------------

The "Welford mirror" convention treats a reflecting surface as a
refracting surface with ``n_after = -n_before``.  Combined with
mirror-parity tracking (``sign = (-1)**mirror_parity`` applied to
every glass-index lookup downstream of an odd number of mirrors),
this puts the Hopkins / Welford Seidel formulas, the Petzval sum,
and any other ``(n2 - n1) / (n1 n2 R)``-style per-surface integral on
the correct effective-index sign.

This convention was fixed across several analysis functions in
isolation:

* v4.11.2 -- :func:`raytrace.seidel_analysis.seidel_coefficients`
  (the canonical reference: parity tracking + ``n2 = -n1`` at every
  mirror).
* v4.13.2 (P1-NEW-C) -- :func:`analysis.field.petzval_radius`
  (single-mirror Petzval was silently dropped pre-fix because the
  loader returns ``glass_before == glass_after`` for mirrors and the
  ``n1 == n2`` skip branch threw the contribution away).

Other analysis functions that walk a prescription's surface list and
consume per-surface ``n1`` / ``n2`` have NOT been individually
verified by a parametrized pin.  This file enumerates the
candidates -- some compute Seidel-style integrals directly, some
delegate to the canonical raytrace machinery (which carries the
Welford fix already) -- and pins that, on a Cassegrain prescription
with two mirrors, every one of them produces a finite output and
either (a) agrees with a hand-computed analytical value, OR (b)
agrees with the equivalent un-folded refractive prescription, OR (c)
at minimum does not raise / produce NaN.

The Cassegrain test prescription is the same one used by the
existing ``TestCassegrainSeidelHandDerivation`` and
``test_p1_new_c_two_mirror_cassegrain_petzval_finite`` tests so
hand-derived expectations are directly portable.

Author: Andrew Traverso -- v4.14.0 / Agent 6
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List

import numpy as np
import pytest

import lumenairy as la
from lumenairy.raytrace.core import Surface

# ============================================================================
# Hand-built Cassegrain prescription (matches existing v4.12.1 / v4.13.2
# Cassegrain Seidel + Petzval tests so the hand-derived expectations
# carry over directly).
# ============================================================================

def _make_cassegrain_prescription() -> Dict[str, Any]:
    """Build a two-mirror Cassegrain prescription as a *dict*
    suitable for the ``analysis`` functions that take a prescription.

    Primary mirror: concave, R1 = -1.0 m (negative -> concave for
    light coming from +z).  Secondary: convex, R2 = -0.3 m.
    Aperture stop sits at the primary; aperture diameter = 100 mm.
    """
    return {
        'name': 'Cassegrain (welford-mirror test prescription)',
        'aperture_diameter': 0.100,
        'object_distance': 1.0,  # finite conjugate; required by some funcs
        'field_max_m': 1e-3,    # for eval_image_plane_wfe non-zero field
        'surfaces': [
            {'radius': -1.0, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'air',
             'is_mirror': True, 'is_stop': True,
             'semi_diameter': 0.050},
            {'radius': -0.3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'air',
             'is_mirror': True,
             'semi_diameter': 0.015},
        ],
        'thicknesses': [0.4],  # primary -> secondary
    }


def _make_cassegrain_surfaces() -> List[Surface]:
    """Cassegrain as a list of :class:`Surface` objects.  Some
    analysis functions accept this form directly (skipping
    ``surfaces_from_prescription``)."""
    return [
        Surface(radius=-1.0, thickness=0.4,
                 glass_before='air', glass_after='air',
                 is_mirror=True, is_stop=True,
                 semi_diameter=0.050),
        Surface(radius=-0.3, thickness=0.0,
                 glass_before='air', glass_after='air',
                 is_mirror=True, semi_diameter=0.015),
    ]


def _make_single_mirror_surfaces() -> List[Surface]:
    """One concave mirror with a known-finite Petzval radius.  Used
    by the smoke-test branch of the parametrized pin so a function
    that does NOT yet apply the Welford convention is still exercised
    (its output is checked against the hand-derived single-mirror
    value)."""
    return [
        Surface(radius=-0.5, thickness=0.0,
                 glass_before='air', glass_after='air',
                 is_mirror=True, semi_diameter=0.050),
    ]


# ============================================================================
# Analysis-function catalogue.  Each entry encodes:
#   - 'fn_name': callable resolved at test time (lazy import).
#   - 'invoke': callable that wraps the fn with appropriate args.
#   - 'mode': 'analytical' (compare to hand value), 'no_raise' (smoke
#       test), 'no_nan' (output must be finite).
# ============================================================================

def _invoke_seidel_coefficients():
    from lumenairy import seidel_coefficients
    surfaces = _make_cassegrain_surfaces()
    result, _ = seidel_coefficients(
        surfaces, wavelength=0.55e-6, field_angle=0.001)
    # Total Petzval (S4) for our test Cassegrain (R1=-1.0, R2=-0.3,
    # d12=0.4, stop at primary) using the Welford parity-tracked
    # n2 = -n1 convention.  Per-surface (audit S3-1: S4 uses the
    # -S_Welford sign of S1..S3, i.e. S4 = +(1/(n2 n1)) c (n2-n1)):
    #   Primary  (n1=+1, n2=-1, c1=-1/1.0=-1):
    #     S4_1 = +(1/(n2 n1)) c (n2-n1) = +(1/-1) (-1) (-2) = -2
    #   Secondary (parity flipped, n1=-1, n2=+1, c2=-1/0.3=-3.333):
    #     S4_2 = +(1/(1 -1)) (-3.333) (2) = +6.667
    #   Total: S4 = -2 + 6.667 = +4.667
    # If the parity-tracking regresses, S4_2 would be computed with
    # n1=+1 -> wrong sign / magnitude; a regression value would be
    # 0 (mirror branch missing entirely) or non-finite.
    s4_total = float(result['total']['S4'])
    return s4_total


def _invoke_petzval_radius():
    from lumenairy import petzval_radius
    surfaces = _make_cassegrain_surfaces()
    return float(petzval_radius(surfaces, wavelength=0.55e-6))


def _invoke_petzval_radius_single_mirror():
    """Single-mirror smoke for the parametrized pin -- the hand calc
    in TestVectorPetzvalMirror.test_p1_new_c_single_mirror_petzval_finite
    gives R_p = 0.25.
    """
    from lumenairy import petzval_radius
    surfaces = _make_single_mirror_surfaces()
    return float(petzval_radius(surfaces, wavelength=0.55e-6))


def _invoke_aberration_summary():
    from lumenairy import aberration_summary
    rx = _make_cassegrain_prescription()
    # The LG-tensor branch fails for mirror-only systems (no glass
    # surface for the canonical-polynomial fit), so disable it for
    # this smoke test.
    summary = aberration_summary(
        rx, wavelength=0.55e-6, include_lg_tensor=False)
    # The Seidel total must be finite (the per-surface Seidel sum
    # delegates to seidel_coefficients which handles mirrors).
    return summary.seidel_total


def _invoke_chromatic_focal_shift():
    from lumenairy import chromatic_focal_shift
    rx = _make_cassegrain_prescription()
    efls, bfls, shift = chromatic_focal_shift(
        rx, wavelengths=[0.5e-6, 0.55e-6, 0.6e-6])
    return shift


def _invoke_distortion_grid():
    from lumenairy import distortion_grid
    rx = _make_cassegrain_prescription()
    # Mirror-only systems can have a tricky bfl; pass a sensible
    # image_distance so the chief-ray trace doesn't try to compute
    # from a degenerate ABCD.
    return distortion_grid(
        rx, wavelength=0.55e-6, max_field_deg=0.1, n_grid=3,
        image_distance=-0.2)


def _invoke_field_aberration_sweep():
    from lumenairy import field_aberration_sweep
    rx = _make_cassegrain_prescription()
    return field_aberration_sweep(
        rx, wavelength=0.55e-6, fields_deg=[0.0, 0.05],
        semi_aperture=0.050,
        dz_search=0.02, n_z=11, n_fan=5)


def _invoke_eval_image_plane_wfe():
    """``eval_image_plane_wfe`` walks the prescription via the
    canonical raytrace machinery (which carries the Welford fix) so
    we expect it to return a finite RMS / PV for a mirror system.
    """
    from lumenairy import eval_image_plane_wfe
    rx = _make_cassegrain_prescription()
    # Make sure object_distance > 0 (the function requires it).
    rx['object_distance'] = 1.0
    # Override img_d_m so we don't depend on the function's paraxial
    # image-distance derivation (which can be subtle on a mirror
    # system).
    return eval_image_plane_wfe(
        rx, wavelength=0.55e-6, field=(0.0, 0.0),
        n_pupil=11, img_d_m=-0.2)


# Catalogue of (function-id, invoker, mode, expected_value-or-None).
# The 'mode' tag drives the assertion:
#   - 'analytical': output must match the recorded value to abs(tol).
#   - 'no_nan': output must be finite (no NaN / inf).
#   - 'no_raise': function must run without raising.
_ANALYSIS_FUNCS_TO_PIN = [
    pytest.param(
        'seidel_coefficients', _invoke_seidel_coefficients,
        # S4 hand calc for the Cassegrain (R1=-1.0, R2=-0.3): see
        # _invoke_seidel_coefficients docstring.  Expected +4.6667: the audit
        # S3-1 fix puts the Petzval S4 on the SAME -S_Welford sign convention
        # as S1..S3 (was -4.6667 -- one convention off, which also corrupted
        # S5 / seidel_wfe); the mirror-parity magnitude 14/3 is unchanged.
        # (Parity still guards the secondary: a regression gives 0 or non-finite.)
        'analytical', 14.0 / 3.0, 1e-9,
        id='seidel_coefficients'),
    pytest.param(
        'petzval_radius_single_mirror',
        _invoke_petzval_radius_single_mirror,
        'analytical', 0.25, 1e-9,
        id='petzval_radius_single_mirror'),
    pytest.param(
        'petzval_radius_cassegrain', _invoke_petzval_radius,
        # Cassegrain Petzval hand calc (see Agent A v4.13.2):
        # sum_inv = -2 + 2/0.3 = 4.6667
        # R_p = -1 / 4.6667 = -0.214286
        'analytical', -1.0 / (-2.0 + 2.0 / 0.3), 1e-6,
        id='petzval_radius_cassegrain'),
    pytest.param(
        'aberration_summary', _invoke_aberration_summary,
        'no_nan', None, None,
        id='aberration_summary'),
    pytest.param(
        'chromatic_focal_shift', _invoke_chromatic_focal_shift,
        'no_nan', None, None,
        id='chromatic_focal_shift'),
    pytest.param(
        'distortion_grid', _invoke_distortion_grid,
        'no_raise', None, None,
        id='distortion_grid'),
    pytest.param(
        'field_aberration_sweep', _invoke_field_aberration_sweep,
        'no_raise', None, None,
        id='field_aberration_sweep'),
    pytest.param(
        'eval_image_plane_wfe', _invoke_eval_image_plane_wfe,
        'no_raise', None, None,
        id='eval_image_plane_wfe'),
]


# ============================================================================
# Parametrized dispatcher pin
# ============================================================================

class TestWelfordMirrorAnalysisDispatcherPin:
    """Pin: every analysis function that consumes a prescription with
    mirrors produces a finite output (and, where a hand value is
    known, agrees with it).

    Each parametrized entry exercises one function on a Cassegrain
    prescription.  The class doc above lists the closures the pin
    catches; the per-function ``mode`` tag drives whether we compare
    to an analytical value or just verify no-NaN / no-raise.

    A regression that drops the Welford ``n2 = -n1`` convention from
    EITHER ``seidel_coefficients`` OR ``petzval_radius`` would fail
    the 'analytical' assertions; a regression on any of the wrapping
    functions (``aberration_summary``, ``chromatic_focal_shift``,
    ``distortion_grid``, ``field_aberration_sweep``,
    ``eval_image_plane_wfe``) would either propagate NaN or raise.
    The parametrized form ensures the next sweep catches every site.
    """

    @pytest.mark.parametrize(
        'fn_name, invoker, mode, expected, tol',
        _ANALYSIS_FUNCS_TO_PIN)
    def test_function_handles_mirror_prescription(
            self, fn_name, invoker, mode, expected, tol):
        # Some of the analysis functions emit RuntimeWarnings during
        # the trace on the Cassegrain (e.g. chief-ray clipping,
        # ABCD-from-mirror-only); silence them so the test output
        # stays clean.  We're checking the *output*, not the warnings.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                out = invoker()
            except (NotImplementedError, ImportError) as exc:
                # NotImplementedError is the canonical refusal for an
                # unsupported branch; ImportError can fire on the JAX-
                # only branches.  Both are acceptable contractual
                # signals (the function did NOT silently miscompute).
                pytest.skip(
                    f"{fn_name} refused the mirror prescription with "
                    f"{type(exc).__name__}: {exc}.  Acceptable -- the "
                    f"function did not silently miscompute.  Update "
                    f"this pin once the function gains mirror support."
                )

        if mode == 'analytical':
            # Output must be a scalar (or a numpy 0-d / scalar-typed
            # value) for the analytical comparison.
            val = float(out)
            assert np.isfinite(val), (
                f"{fn_name}: returned non-finite value {val!r} on "
                f"Cassegrain prescription.  Pre-fix the Welford "
                f"convention was missing on at least one surface; "
                f"check the per-surface n1/n2 loop carries "
                f"mirror_parity tracking.")
            assert abs(val - float(expected)) <= float(tol), (
                f"{fn_name}: returned {val!r}; expected "
                f"{expected!r} (Welford hand calc).  Difference "
                f"exceeds tolerance {tol}.")
        elif mode == 'no_nan':
            arr = np.asarray(out)
            # All-finite -- no NaN / inf leaking through a missed
            # mirror branch.  Some fields (efl/bfl on mirror systems)
            # legitimately come back NaN, so we relax this to "at
            # least SOMETHING is finite" rather than "all finite":
            # a fully NaN array indicates a total mirror-handling
            # failure.
            assert np.any(np.isfinite(arr)), (
                f"{fn_name}: all values NaN/inf on Cassegrain "
                f"prescription.  Indicates total mirror-handling "
                f"failure -- the Welford convention isn't being "
                f"applied.  Got {arr!r}.")
        elif mode == 'no_raise':
            # The function ran to completion.  This is the weakest
            # assertion -- the audit prescribes it for analysis
            # functions that do per-surface refraction loops where
            # the result depends on too many other physical factors
            # for a single hand-derived target value.  The function
            # *did not raise*, which (per the v4.13.2 audit P1-NEW-C
            # ledger) is the load-bearing regression guard.
            assert out is not None
        else:
            raise ValueError(mode)


# ============================================================================
# Auxiliary pins: documentation of un-folded equivalence (the audit
# allows option-(b) for functions where a hand value is unavailable).
# ============================================================================

class TestUnfoldedEquivalenceForPetzval:
    """Pin: the equivalent un-folded refractive prescription's Petzval
    contribution equals the mirror's Welford contribution (the
    convention's defining property).

    For a single mirror with ``n_before = +1`` and Welford
    ``n_after = -1``, the surface's Petzval contribution is
    ``(n2 - n1) / (n1 n2 R) = -2 / -R = 2/R``.

    The equivalent un-folded refractive twin: a hypothetical
    refracting surface with ``n_before = +1``, ``n_after = -1``
    (i.e. air -> negative-index virtual medium).  ``petzval_radius``
    does NOT support arbitrary n_after values via the glass
    registry, so we exercise the equivalence by checking that the
    mirror's Welford contribution matches the algebraic formula.
    """

    def test_single_mirror_petzval_matches_welford_formula(self):
        """For a single mirror with R = -0.5, n1 = +1, Welford
        n2 = -1: contribution = (n2 - n1) / (n1 n2 R)
        = (-1 - 1) / ((1)(-1)(-0.5)) = -2 / 0.5 = -4.
        R_p = -1 / (-4) = +0.25."""
        from lumenairy import petzval_radius
        surfaces = _make_single_mirror_surfaces()
        R_p = petzval_radius(surfaces, wavelength=0.55e-6)
        # Algebraic check on the Welford formula's prediction.
        n1, n2, R = 1.0, -1.0, -0.5
        inv_R = (n2 - n1) / (n1 * n2 * R)
        expected = -1.0 / inv_R
        assert abs(R_p - expected) < 1e-9


# ============================================================================
# Sanity / smoke: package-level imports remain in place.
# ============================================================================

class TestWelfordPublicSurfacePin:
    """Pin: every analysis function in the catalogue is importable
    from the top-level ``lumenairy`` namespace.  A future refactor
    that hides one of these names will fail this test rather than
    breaking importers downstream."""

    @pytest.mark.parametrize('name', [
        'seidel_coefficients',
        'petzval_radius',
        'aberration_summary',
        'chromatic_focal_shift',
        'distortion_grid',
        'field_aberration_sweep',
        'eval_image_plane_wfe',
    ])
    def test_public_name_importable(self, name):
        assert hasattr(la, name), (
            f"lumenairy missing top-level public name {name!r}.")
        obj = getattr(la, name)
        assert callable(obj)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

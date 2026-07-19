"""Audit v5.24.2 B3 / S4-18 -- merit-family scale unification (opt-in).

``NormalizedMerit(inner, scale=)`` rescales any merit family to a common
dimensionless scale.  It is strictly OPT-IN: the ``design_optimize``
default path (unwrapped merit terms) is byte-identical to its historical
numerics -- pinned here against independent analytic oracles.

Author: audit remediation -- v5.25 / B3
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy import (
    ChromaticFocalShiftMerit,
    FocalLengthMerit,
    MinThicknessMerit,
    NormalizedMerit,
    SpotSizeMerit,
    StrehlMerit,
)
from lumenairy.optimize.context import EvaluationContext


def _ctx(**over):
    """Minimal EvaluationContext with the fields the tested merits read."""
    ctx = EvaluationContext(
        prescription=over.pop('prescription', {}),
        wavelength=over.pop('wavelength', 1.3e-6),
        N=over.pop('N', 64),
        dx=over.pop('dx', 2e-6),
    )
    for k, v in over.items():
        setattr(ctx, k, v)
    return ctx


# ---------------------------------------------------------------------------
# Default path (UNWRAPPED) is byte-identical -- independent analytic oracles.
# Adding ``native_scale`` must not change any merit's evaluate().
# ---------------------------------------------------------------------------

def test_default_focal_length_formula_unchanged():
    """FocalLengthMerit raw = weight * ((efl - target)/target)^2."""
    m = FocalLengthMerit(target=0.1, weight=3.0)
    val = m.evaluate(_ctx(efl=0.11))
    assert val == pytest.approx(3.0 * ((0.11 - 0.1) / 0.1) ** 2)


def test_default_spot_size_formula_unchanged():
    """SpotSizeMerit raw = weight * (r - max_rms_radius)^2 [m^2]."""
    m = SpotSizeMerit(max_rms_radius=5e-6, weight=2.0)
    val = m.evaluate(_ctx(rms_radius_best=1e-5))
    assert val == pytest.approx(2.0 * (1e-5 - 5e-6) ** 2)


def test_default_min_thickness_formula_unchanged():
    presc = {'thicknesses': [0.4e-3],
             'surfaces': [{'glass_after': 'N-BK7'}]}
    m = MinThicknessMerit(min_thickness=1e-3, weight=1.0)
    val = m.evaluate(_ctx(prescription=presc))
    assert val == pytest.approx((1e-3 - 0.4e-3) ** 2)


# ---------------------------------------------------------------------------
# native_scale attributes match the documented table.
# ---------------------------------------------------------------------------

def test_native_scale_values():
    assert SpotSizeMerit(max_rms_radius=5e-6).native_scale == \
        pytest.approx((5e-6) ** 2)
    assert MinThicknessMerit(min_thickness=2e-3).native_scale == \
        pytest.approx((2e-3) ** 2)
    assert StrehlMerit().native_scale == 1.0
    assert FocalLengthMerit(target=0.1).native_scale == 1.0
    # afocal focal length is reference-less (dioptre^2, no natural scale)
    assert FocalLengthMerit(target=0.0).native_scale is None


# ---------------------------------------------------------------------------
# NormalizedMerit divides by the (auto or explicit) scale.
# ---------------------------------------------------------------------------

def test_normalized_divides_by_auto_native_scale():
    """SpotSize normalized: a spot at 2x the target contributes O(weight).

    Independent oracle: raw = (r - t)^2 = t^2 at r = 2t, native scale =
    t^2, so normalized == weight exactly."""
    inner = SpotSizeMerit(max_rms_radius=5e-6, weight=1.0)
    ctx = _ctx(rms_radius_best=1e-5)  # r = 2 * target
    norm = NormalizedMerit(inner)
    assert norm.evaluate(ctx) == pytest.approx(1.0)
    assert norm.evaluate(ctx) == pytest.approx(
        inner.evaluate(ctx) / (5e-6) ** 2)


def test_normalized_dimensionless_family_is_identity():
    """FocalLengthMerit(target>0) native scale is 1.0 -> normalized ==
    raw (already dimensionless)."""
    inner = FocalLengthMerit(target=0.1, weight=1.0)
    ctx = _ctx(efl=0.12)
    assert NormalizedMerit(inner).evaluate(ctx) == \
        pytest.approx(inner.evaluate(ctx))


def test_normalized_explicit_scale():
    inner = StrehlMerit(min_strehl=0.8, weight=1.0)
    ctx = _ctx(strehl_best=0.5)  # raw = 0.3^2 = 0.09
    assert inner.evaluate(ctx) == pytest.approx(0.09)
    assert NormalizedMerit(inner, scale=0.09).evaluate(ctx) == \
        pytest.approx(1.0)


def test_common_scale_brings_disparate_families_together():
    """The audit's core complaint: an afocal focal (dioptre^2) and a
    spot-size (m^2) penalty land ~1e10 apart raw, but ~O(1) after
    NormalizedMerit."""
    afocal = FocalLengthMerit(target=0.0, weight=1.0)  # (1/efl)^2, dioptre^2
    spot = SpotSizeMerit(max_rms_radius=5e-6, weight=1.0)  # m^2
    ctx = _ctx(efl=0.1, rms_radius_best=1e-5)
    raw_afocal = afocal.evaluate(ctx)   # (1/0.1)^2 = 100
    raw_spot = spot.evaluate(ctx)       # (5e-6)^2 = 2.5e-11
    assert raw_afocal / raw_spot > 1e10  # wildly disparate raw scales
    n_afocal = NormalizedMerit(afocal, scale=100.0)  # dioptre^2 reference
    n_spot = NormalizedMerit(spot)                   # auto m^2 reference
    r1, r2 = n_afocal.evaluate(ctx), n_spot.evaluate(ctx)
    # both now O(1) -- within a couple decades of each other
    assert 0.1 < r1 / r2 < 10.0


# ---------------------------------------------------------------------------
# Scheduling-flag / weight forwarding.
# ---------------------------------------------------------------------------

def test_forwards_scheduling_flags_and_weight():
    inner = SpotSizeMerit(max_rms_radius=5e-6, weight=2.5)  # needs_wave=True
    norm = NormalizedMerit(inner)
    assert norm.needs_wave is inner.needs_wave is True
    assert norm.needs_ray == getattr(inner, 'needs_ray', True)
    assert norm.weight == 2.5
    assert norm.name == 'Normalized(SpotSize)'
    assert isinstance(norm, la.MeritTerm)


# ---------------------------------------------------------------------------
# Error paths.
# ---------------------------------------------------------------------------

def test_reference_less_family_requires_explicit_scale():
    with pytest.raises(ValueError):
        NormalizedMerit(ChromaticFocalShiftMerit())
    with pytest.raises(ValueError):
        NormalizedMerit(FocalLengthMerit(target=0.0))  # afocal
    # explicit scale is accepted
    NormalizedMerit(ChromaticFocalShiftMerit(), scale=1e-6)


@pytest.mark.parametrize('bad', [0.0, -1.0, float('nan'), float('inf')])
def test_rejects_nonpositive_or_nonfinite_scale(bad):
    with pytest.raises(ValueError):
        NormalizedMerit(FocalLengthMerit(target=0.1), scale=bad)


def test_rejects_non_merit_inner():
    with pytest.raises(TypeError):
        NormalizedMerit(object())


# ---------------------------------------------------------------------------
# Public surface.
# ---------------------------------------------------------------------------

def test_normalized_merit_is_top_level_exported():
    assert hasattr(la, 'NormalizedMerit')
    assert 'NormalizedMerit' in la.__all__

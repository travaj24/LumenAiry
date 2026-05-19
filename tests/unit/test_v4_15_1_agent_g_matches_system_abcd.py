"""v4.15.1 Agent G -- ``Operator.from_prescription`` ABCD parity tests.

This module pins the cross-check between the operator-algebra
``Operator.from_prescription`` factory and the ground-truth
:func:`lumenairy.raytrace.system_abcd`.  For every representative
prescription (singlet, doublet, telephoto, BK7+SF11 close pair,
flat detector window), the composite ABCD must match ``system_abcd``
to within 1e-12 absolute.

Corresponds to ``CLUSTER_B_SPEC.md`` §3.7
``test_algebra_matches_system_abcd.py``.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.raytrace.core import surfaces_from_prescription, system_abcd


# ============================================================================
# Prescription fixtures
# ============================================================================


def _build_singlet() -> dict:
    return la.make_singlet(
        R1=50e-3, R2=-50e-3, d=5e-3,
        glass='N-BK7', aperture=25.4e-3,
    )


def _build_doublet() -> dict:
    # Thorlabs-style achromatic doublet (AC254-050-C parameters)
    return la.make_doublet(
        R1=33.3e-3, R2=-24.1e-3, R3=-95.3e-3,
        d1=9.0e-3, d2=3.0e-3,
        glass1='N-BAF10', glass2='N-SF6HT',
        aperture=25.4e-3,
    )


def _build_telephoto() -> dict:
    """Two-group telephoto: positive front group, negative rear group,
    big axial gap between."""
    return {
        'name': 'Telephoto',
        'aperture_diameter': 25.4e-3,
        'surfaces': [
            {'radius': 100e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -100e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'},
            {'radius': -50e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-SF11'},
            {'radius': 50e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-SF11', 'glass_after': 'air'},
        ],
        'thicknesses': [3e-3, 60e-3, 3e-3, 0.0],
    }


def _build_close_pair() -> dict:
    """Two thin singlets in close axial succession (Petzval-like)."""
    return {
        'name': 'ClosePair',
        'aperture_diameter': 25.4e-3,
        'surfaces': [
            {'radius': 75e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -150e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'},
            {'radius': 150e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-SF11'},
            {'radius': -75e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-SF11', 'glass_after': 'air'},
        ],
        'thicknesses': [4e-3, 0.5e-3, 3e-3, 0.0],
    }


def _build_flat_window() -> dict:
    """Plane-parallel BK7 window -- no power, just a thickness shift."""
    return {
        'name': 'FlatWindow',
        'aperture_diameter': 25.4e-3,
        'surfaces': [
            {'radius': float('inf'), 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': float('inf'), 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [3e-3, 0.0],
    }


# ============================================================================
# Parity tests
# ============================================================================


PRESCRIPTIONS = [
    ('singlet', _build_singlet),
    ('doublet', _build_doublet),
    ('telephoto', _build_telephoto),
    ('close_pair', _build_close_pair),
    ('flat_window', _build_flat_window),
]

WAVELENGTHS = [
    ('VIS', 633e-9),
    ('NIR', 1.55e-6),
]


@pytest.mark.parametrize('name,builder', PRESCRIPTIONS)
@pytest.mark.parametrize('wl_label,wavelength', WAVELENGTHS)
def test_abcd_matches_system_abcd(
    name: str,
    builder,
    wl_label: str,
    wavelength: float,
) -> None:
    """``Operator.from_prescription(p, lam).abcd`` must equal
    ``system_abcd(surfaces_from_prescription(p), lam)[0]`` to within
    1e-12 absolute.

    This is the spec §3.9 acceptance criterion (at least 3
    representative prescriptions; we run 5 x 2 = 10 here for extra
    coverage)."""
    prescription = builder()
    surfaces = surfaces_from_prescription(prescription)
    M_ref, efl_ref, _, _ = system_abcd(surfaces, wavelength)

    op = la.Operator.from_prescription(prescription, wavelength)

    assert np.allclose(op.abcd, M_ref, atol=1e-12), (
        f"{name} @ {wl_label}: ABCD mismatch.\n"
        f"  reference (system_abcd):\n{M_ref}\n"
        f"  operator algebra:\n{op.abcd}\n"
        f"  max abs diff: {np.max(np.abs(op.abcd - M_ref)):.3e}"
    )

    # EFL parity: when both are finite, they must match.  Both
    # implementations use the same -1/C convention so the only
    # way the values can disagree is via numerical roundoff in the
    # leg-by-leg accumulation -- 1e-9 relative is plenty here.
    if np.isfinite(efl_ref) and np.isfinite(op.efl):
        assert np.isclose(op.efl, efl_ref, rtol=1e-9, atol=1e-12), (
            f"{name} @ {wl_label}: EFL mismatch. "
            f"ref={efl_ref}, op={op.efl}"
        )


@pytest.mark.parametrize('name,builder', PRESCRIPTIONS)
def test_from_prescription_returns_composite(name: str, builder) -> None:
    """The factory must return a :class:`CompositeOperator`, not a
    bare :class:`Operator`, so callers can call :meth:`stages`."""
    from lumenairy.algebra import CompositeOperator
    op = la.Operator.from_prescription(builder(), 633e-9)
    assert isinstance(op, CompositeOperator)
    assert len(op.stages()) >= 1


def test_from_prescription_invalid_wavelength_raises() -> None:
    rx = _build_singlet()
    with pytest.raises(ValueError):
        la.Operator.from_prescription(rx, -1.0)
    with pytest.raises(ValueError):
        la.Operator.from_prescription(rx, float('inf'))


def test_from_prescription_method_kwarg_threaded() -> None:
    """The ``method`` kwarg should reach every FreeSpace primitive
    in the emitted chain."""
    op = la.Operator.from_prescription(
        _build_singlet(), 633e-9, method='asm',
    )
    # Every FreeSpace stage carries the requested method.
    free_space_stages = [
        s for s in op.stages()
        if isinstance(s, la.FreeSpace)
    ]
    assert len(free_space_stages) >= 1
    for fs in free_space_stages:
        assert fs.method == 'asm'


# ============================================================================
# Folded-prescription parity (v4.15.2 audit P1-NEW-B)
# ============================================================================


def _build_folded_singlet() -> dict:
    """Build a folded prescription: refractive singlet followed by a
    flat fold mirror, then a final image plane.

    ``system_abcd`` walks the surface list and treats the flat mirror
    via the ``R_mat = np.eye(2)`` branch without flipping
    ``mirror_parity``.  The operator-algebra ``from_prescription`` must
    match this exactly -- pre-v4.15.2 it flipped parity for flat
    mirrors and produced an off-sign ABCD on this prescription.
    """
    glass = 'N-BK7'
    return {
        'name': 'FoldedSinglet',
        'aperture_diameter': 25.4e-3,
        'surfaces': [
            {'radius': 50e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': glass},
            {'radius': -50e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': glass, 'glass_after': 'air'},
            # Flat fold mirror: R = inf, glass_after = MIRROR marker
            # (replaced by glass_before in surfaces_from_prescription).
            {'radius': float('inf'), 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'MIRROR'},
        ],
        'thicknesses': [5e-3, 30e-3, 0.0],
    }


def _build_folded_telephoto() -> dict:
    """Two-group telephoto with a flat fold mirror between the groups.

    Doubles up the v4.15.2 audit scenario: a real folded design from a
    Cassegrain-ish layout where the first group is positive, then a
    flat fold mirror, then a negative rear group.  The flat mirror
    must NOT flip parity in either the algebra layer or
    ``system_abcd``.
    """
    return {
        'name': 'FoldedTelephoto',
        'aperture_diameter': 25.4e-3,
        'surfaces': [
            {'radius': 100e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -100e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'},
            # Flat fold mirror.
            {'radius': float('inf'), 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'MIRROR'},
            {'radius': -50e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-SF11'},
            {'radius': 50e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-SF11', 'glass_after': 'air'},
        ],
        'thicknesses': [3e-3, 40e-3, 30e-3, 3e-3, 0.0],
    }


def _build_folded_4fold_periscope() -> dict:
    """Four-flat-fold-mirror periscope chain (v4.15.3 audit P3 / Tier-1).

    Four flat fold mirrors in sequence with a refractive singlet
    interleaved between the second and third fold.  This doubles the
    2-fold ``_build_folded_singlet`` case to exercise the
    flat-mirror-parity path **four times** in a single prescription
    -- each flat fold must NOT flip ``mirror_parity`` in either
    layer (algebra or ``system_abcd``).

    Geometry mirrors a real-world periscope: a beam enters, bounces
    off four flat folds, traverses a refractive singlet, and exits.
    Total mirror count: 4; refractive elements: 1 (BK7 singlet).
    """
    glass = 'N-BK7'
    return {
        'name': 'Folded4FoldPeriscope',
        'aperture_diameter': 25.4e-3,
        'surfaces': [
            # Fold mirror 1 (flat).
            {'radius': float('inf'), 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'MIRROR'},
            # Fold mirror 2 (flat).
            {'radius': float('inf'), 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'MIRROR'},
            # Refractive singlet (BK7) -- the only powered element.
            {'radius': 60e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': glass},
            {'radius': -60e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': glass, 'glass_after': 'air'},
            # Fold mirror 3 (flat).
            {'radius': float('inf'), 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'MIRROR'},
            # Fold mirror 4 (flat).
            {'radius': float('inf'), 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'MIRROR'},
        ],
        'thicknesses': [15e-3, 15e-3, 4e-3, 25e-3, 15e-3, 0.0],
    }


def _build_folded_cassegrain_2curved_2flat() -> dict:
    """Cassegrain-style 4-mirror system (v4.15.3 audit P3 / Tier-1).

    Two CURVED mirrors (primary concave + secondary convex) plus two
    FLAT fold mirrors after the cassegrain group.  This is the
    canonical mixed-mirror geometry: ``system_abcd`` and
    ``from_prescription`` must agree even when CURVED mirrors flip
    parity twice (cancelling) and FLAT mirrors don't flip parity at
    all.  The cumulative parity at the output is therefore even (no
    net flip) -- the parity bookkeeping is non-trivial but the
    algebra layer must still match the ground truth.

    Surface count: 4 mirrors.  Refractive elements: 0.
    """
    return {
        'name': 'FoldedCassegrain2Curved2Flat',
        'aperture_diameter': 25.4e-3,
        'surfaces': [
            # Primary mirror: concave R = -200 mm (Welford convention).
            {'radius': -200e-3, 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'MIRROR'},
            # Secondary mirror: convex R = +50 mm (Welford convention
            # for the rear-going leg after the primary flip).
            {'radius': 50e-3, 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'MIRROR'},
            # Flat fold mirror 1 (post-secondary).
            {'radius': float('inf'), 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'MIRROR'},
            # Flat fold mirror 2.
            {'radius': float('inf'), 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'MIRROR'},
        ],
        # Thicknesses (note the negative leg for the rear-going
        # post-primary reflection): primary -> secondary (back),
        # secondary -> fold 1 (forward), fold 1 -> fold 2, fold 2 ->
        # exit.
        'thicknesses': [-180e-3, 200e-3, 20e-3, 0.0],
    }


FOLDED_PRESCRIPTIONS = [
    ('folded_singlet', _build_folded_singlet),
    ('folded_telephoto', _build_folded_telephoto),
    # v4.15.3 (audit P3 / Tier-1): 4-fold mirror parity coverage.
    ('folded_4fold_periscope', _build_folded_4fold_periscope),
    ('folded_cassegrain_2curved_2flat',
     _build_folded_cassegrain_2curved_2flat),
]


@pytest.mark.parametrize('name,builder', FOLDED_PRESCRIPTIONS)
@pytest.mark.parametrize('wl_label,wavelength', WAVELENGTHS)
def test_from_prescription_matches_system_abcd_folded(
    name: str,
    builder,
    wl_label: str,
    wavelength: float,
) -> None:
    """v4.15.2 (audit P1-NEW-B) + v4.15.3 (audit P3 / Tier-1): the
    operator-algebra ``from_prescription`` must match ``system_abcd``
    on FOLDED prescriptions that contain at least one flat fold
    mirror.

    Pre-v4.15.2 ``from_prescription`` unconditionally flipped
    ``mirror_parity`` on every ``is_mirror=True`` surface, while
    ``system_abcd`` only flips parity for CURVED mirrors.  A
    folded design with a flat fold mirror therefore got an off-sign
    ABCD from the algebra layer.

    v4.15.3 (Tier-1): adds 4-fold-mirror coverage (periscope and
    cassegrain-style) per the v4.15.2 audit P3 recommendation.

    Acceptance criterion: composite ABCD must match
    ``system_abcd(surfaces_from_prescription(p), lam)[0]`` to within
    1e-12 absolute.
    """
    prescription = builder()
    surfaces = surfaces_from_prescription(prescription)
    M_ref, efl_ref, _, _ = system_abcd(surfaces, wavelength)

    op = la.Operator.from_prescription(prescription, wavelength)

    assert np.allclose(op.abcd, M_ref, atol=1e-12), (
        f"{name} @ {wl_label}: folded-prescription ABCD mismatch.\n"
        f"  reference (system_abcd):\n{M_ref}\n"
        f"  operator algebra:\n{op.abcd}\n"
        f"  max abs diff: {np.max(np.abs(op.abcd - M_ref)):.3e}"
    )

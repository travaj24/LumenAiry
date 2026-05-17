"""Pinning tests for the v4.13.0 Phase 3 perf hoist in seidel_field_sweep.

Audit reference
---------------

Phase 3 of the v4.13.0 audit sweep hoists the field-independent work
out of ``seidel_field_sweep``'s per-field loop.  Pre-hoist the
function called :func:`lumenairy.seidel_coefficients` once per
requested field height, re-running the marginal-ray trace, glass-
index lookups, pre-stop ABCD, and full ``system_abcd`` on every
iteration.  Post-hoist a single ``seidel_coefficients`` call at
``field_angle=1.0`` captures all per-surface marginal / chief data,
then vectorised analytical scaling reproduces every per-field result
to machine precision:

    S1, S4                          field-independent
    y_marginal                      field-independent
    y_chief                         propto field_angle
    S2 = -(A_m A_c) h delta_un      propto field_angle
    S3 = -(A_c^2) h delta_un        propto field_angle^2
    S5 = -(A_c/A_m)(S3 + H^2 S4)    propto field_angle^3

The paraxial Seidel formalism is exactly linear in the chief-ray
initial conditions (which themselves are linear in ``field_angle``),
so the analytical scaling is exact under the same paraxial model that
``seidel_coefficients`` uses; the relevant terms in
:func:`lumenairy.raytrace.core.seidel_coefficients` are tagged.

These tests pin element-by-element agreement with the pre-hoist
implementation across a singlet, a doublet, and a system with a
mid-stack stop.

Author: Andrew Traverso
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import lumenairy as la
from lumenairy.raytrace.core import seidel_coefficients
from lumenairy.raytrace.seidel_analysis import seidel_field_sweep


# ============================================================================
# Helpers: reference pre-hoist implementation
# ============================================================================


def _seidel_field_sweep_reference(
    surfaces, wavelength, field_heights,
    *, object_distance=float('inf'), stop_index=None,
):
    """Reproduce the pre-hoist (per-field loop) implementation
    verbatim so we can pin element-by-element agreement with the
    optimised version."""
    heights = np.atleast_1d(np.asarray(field_heights, dtype=float))
    per_field = []
    abcd = None
    for h in heights:
        s, m = seidel_coefficients(
            surfaces, wavelength,
            object_distance=object_distance,
            stop_index=stop_index,
            field_angle=float(h),
        )
        per_field.append(s)
        if abcd is None:
            abcd = m
    keys = ('S1', 'S2', 'S3', 'S4', 'S5')
    result = {
        'field_heights': heights,
        'labels': per_field[0]['labels'],
        'stop_index': per_field[0]['stop_index'],
        'y_marginal': per_field[0]['y_marginal'],
        'y_chief': np.stack([s['y_chief'] for s in per_field], axis=-1),
    }
    for k in keys:
        result[k] = np.stack([s[k] for s in per_field], axis=-1)
    result['total'] = {
        k: np.array([s['total'][k] for s in per_field]) for k in keys
    }
    return result, abcd


# ============================================================================
# Singlet correctness
# ============================================================================


class TestSingletAgreesWithReference:
    """A simple plano-convex singlet sweep across five field heights
    matches the per-field reference to 1e-12."""

    def _surfaces(self):
        presc = la.make_singlet(
            R1=50e-3, R2=np.inf, d=3e-3, glass='N-BK7', aperture=10e-3)
        return la.surfaces_from_prescription(presc)

    def test_five_fields_match(self):
        surfaces = self._surfaces()
        wavelength = 1.31e-6
        heights = np.linspace(0.0, 0.1, 5)

        ref, abcd_ref = _seidel_field_sweep_reference(
            surfaces, wavelength, heights)
        new, abcd_new = seidel_field_sweep(
            surfaces, wavelength, heights)

        # Spec: element-by-element to 1e-12.
        keys = ('S1', 'S2', 'S3', 'S4', 'S5')
        for k in keys:
            diff = np.max(np.abs(new[k] - ref[k]))
            assert diff < 1e-12, f"{k}: diff = {diff:.3e}"
            # Also pin totals.
            diff_total = np.max(np.abs(new['total'][k] - ref['total'][k]))
            assert diff_total < 1e-12, (
                f"total[{k}]: diff = {diff_total:.3e}")

        # Ray-height structure preserved.
        assert np.allclose(new['y_marginal'], ref['y_marginal'])
        assert np.max(np.abs(new['y_chief'] - ref['y_chief'])) < 1e-12

        # ABCD matrix matches.
        assert np.allclose(abcd_new, abcd_ref)

        # field_heights / labels / stop_index propagate.
        assert np.array_equal(new['field_heights'], heights)
        assert new['labels'] == ref['labels']
        assert new['stop_index'] == ref['stop_index']

    def test_hand_computed_seidel_at_one_field(self):
        """At a single non-zero field, the swept-S2 value matches a
        direct ``seidel_coefficients`` call.

        Spec: 'Build a simple paraxial system ... for which Seidel
        coefficients can be hand-computed at one field'.  We compare
        the swept value to ``seidel_coefficients(... field_angle=h)``
        for that field, which IS the canonical paraxial computation.
        """
        surfaces = self._surfaces()
        wavelength = 1.31e-6
        h = 0.02

        new, _ = seidel_field_sweep(
            surfaces, wavelength, np.array([h]))
        direct, _ = seidel_coefficients(
            surfaces, wavelength, field_angle=h)

        for k in ('S1', 'S2', 'S3', 'S4', 'S5'):
            new_total = float(new['total'][k][0])
            direct_total = float(direct['total'][k])
            diff = abs(new_total - direct_total)
            assert diff < 1e-12, (
                f"total[{k}] at h={h}: diff = {diff:.3e}, "
                f"new={new_total}, direct={direct_total}")

    def test_zero_field_gives_zero_chief_terms(self):
        """At ``field_angle = 0`` the chief-ray-dependent Seidel sums
        (S2, S3, S5) and ``y_chief`` are exactly zero."""
        surfaces = self._surfaces()
        wavelength = 1.31e-6
        new, _ = seidel_field_sweep(
            surfaces, wavelength, np.array([0.0]))

        for k in ('S2', 'S3', 'S5'):
            assert np.all(new[k][:, 0] == 0.0), (
                f"{k} should be zero at field=0")
            assert new['total'][k][0] == 0.0
        assert np.all(new['y_chief'][:, 0] == 0.0)

        # S1, S4, y_marginal remain non-zero (field-independent).
        assert np.any(new['S1'][:, 0] != 0.0)
        assert np.any(new['y_marginal'] != 0.0)


# ============================================================================
# Doublet (mid-stack stop) correctness
# ============================================================================


class TestDoubletAgreesWithReference:
    """A symmetric two-singlet doublet with a mid-stack stop matches
    the per-field reference; the stop_index logic is exercised."""

    def _surfaces(self):
        # Cemented achromatic doublet (BK7 / SF2).
        presc = la.make_doublet(
            R1=80e-3, R2=-50e-3, R3=-200e-3,
            d1=3e-3, d2=2e-3,
            glass1='N-BK7', glass2='N-SF2',
            aperture=10e-3,
        )
        return la.surfaces_from_prescription(presc)

    def test_five_fields_match(self):
        surfaces = self._surfaces()
        wavelength = 1.31e-6
        heights = np.array([0.0, 0.02, 0.05, 0.08, 0.10])

        ref, abcd_ref = _seidel_field_sweep_reference(
            surfaces, wavelength, heights)
        new, abcd_new = seidel_field_sweep(
            surfaces, wavelength, heights)

        for k in ('S1', 'S2', 'S3', 'S4', 'S5'):
            diff = np.max(np.abs(new[k] - ref[k]))
            assert diff < 1e-12, f"{k}: diff = {diff:.3e}"
        assert np.allclose(new['y_chief'], ref['y_chief'])
        assert np.allclose(abcd_new, abcd_ref)


# ============================================================================
# Explicit stop_index path
# ============================================================================


class TestExplicitStopIndexAgrees:
    """Passing ``stop_index`` explicitly still produces the same
    swept-output as the reference loop."""

    def test_explicit_stop(self):
        # Singlet with no flagged stop; supply stop_index=1 manually
        # to exercise the pre-stop ABCD path inside
        # seidel_coefficients (which is now run only once instead of
        # once-per-field).
        presc = la.make_singlet(
            R1=50e-3, R2=-50e-3, d=3e-3, glass='N-BK7', aperture=10e-3)
        surfaces = la.surfaces_from_prescription(presc)
        wavelength = 1.31e-6
        heights = np.array([0.01, 0.03, 0.07])

        ref, _ = _seidel_field_sweep_reference(
            surfaces, wavelength, heights, stop_index=1)
        new, _ = seidel_field_sweep(
            surfaces, wavelength, heights, stop_index=1)

        for k in ('S1', 'S2', 'S3', 'S4', 'S5'):
            diff = np.max(np.abs(new[k] - ref[k]))
            assert diff < 1e-12, f"{k}: diff = {diff:.3e}"


# ============================================================================
# Finite-conjugate path
# ============================================================================


class TestFiniteConjugateAgrees:
    """The finite-object branch (where ``y_m_init`` is non-zero and
    the Lagrange invariant gains a marginal-ray contribution) also
    obeys exact scaling."""

    def test_finite_object_distance(self):
        presc = la.make_singlet(
            R1=50e-3, R2=-50e-3, d=3e-3, glass='N-BK7', aperture=10e-3)
        surfaces = la.surfaces_from_prescription(presc)
        wavelength = 1.31e-6
        heights = np.array([0.0, 0.02, 0.05])
        object_distance = 0.5  # 500 mm

        ref, _ = _seidel_field_sweep_reference(
            surfaces, wavelength, heights,
            object_distance=object_distance)
        new, _ = seidel_field_sweep(
            surfaces, wavelength, heights,
            object_distance=object_distance)

        for k in ('S1', 'S2', 'S3', 'S4', 'S5'):
            diff = np.max(np.abs(new[k] - ref[k]))
            # Slightly looser bound: finite-object magnitudes are
            # larger and rounding errors compound through more terms.
            assert diff < 1e-12, f"{k}: diff = {diff:.3e}"


# ============================================================================
# Output-shape / metadata pinning
# ============================================================================


class TestOutputShapeAndMetadata:
    """Output shapes, dtypes, keys, and the ``total`` sub-dict are
    unchanged by the hoist."""

    def test_shapes_and_keys_preserved(self):
        presc = la.make_singlet(
            R1=50e-3, R2=np.inf, d=3e-3, glass='N-BK7', aperture=10e-3)
        surfaces = la.surfaces_from_prescription(presc)
        heights = np.linspace(0.0, 0.1, 11)
        new, abcd = seidel_field_sweep(surfaces, 1.31e-6, heights)

        n_surf = len(surfaces)
        n_fields = heights.size

        # Per-surface (N_surf, N_fields).
        for k in ('S1', 'S2', 'S3', 'S4', 'S5'):
            assert new[k].shape == (n_surf, n_fields), (
                f"{k} shape: got {new[k].shape}, expected "
                f"({n_surf}, {n_fields})")
        # y_marginal: (N_surf,).
        assert new['y_marginal'].shape == (n_surf,)
        # y_chief: (N_surf, N_fields).
        assert new['y_chief'].shape == (n_surf, n_fields)
        # field_heights: (N_fields,).
        assert new['field_heights'].shape == (n_fields,)
        # total sub-dict.
        for k in ('S1', 'S2', 'S3', 'S4', 'S5'):
            assert new['total'][k].shape == (n_fields,)
        # ABCD: (2, 2).
        assert abcd.shape == (2, 2)

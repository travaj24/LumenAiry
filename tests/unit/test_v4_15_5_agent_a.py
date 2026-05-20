"""v4.15.5 Agent A regression tests.

Pins the v4.15.5 P1-NEW-2WAY-1 (V6 walker first-positional-name
discovery), P1-NEW-F2-2 (``DeformableMirror.apply`` direct guard),
and P2-NEW-F2-4 (walker class-body descent) audit closures.

Each unguarded analyzer in v4.15.4 had one of two failure modes:

* For a :class:`~lumenairy.PartialCoherenceMCF` input: a Python
  ``TypeError`` from attribute access / arithmetic on the MCF
  object (e.g. ``np.abs(mcf)`` raises ``TypeError: bad operand
  type for abs()``).  After the v4.15.5 guard the same input
  raises ``TypeError`` with the canonical v4.16-roadmap message
  via :func:`lumenairy._validation._check_2d_scalar_field`.
* For a 3-D NumPy ensemble (shape ``(n_realizations, Ny, Nx)``):
  either a silent wrong-axis broadcast (e.g. ``E_in * np.exp(...)``
  returns a 3-D array) or a meaningless reduction (e.g.
  ``np.fft.fft2`` FFTs along the last two axes of a 3-D array).
  After the v4.15.5 guard the same input raises ``ValueError``
  with the canonical iterate-over-ensemble hint.

The post-v4.15.5 contract: every guarded entry point routes both
failure modes through :func:`_check_2d_scalar_field` so the error
message is uniform.  This file pins that contract end-to-end.

Author: Andrew Traverso -- v4.15.5 / Agent A
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest


# ============================================================================
# Shared fixtures.
# ============================================================================

@pytest.fixture(scope='module')
def mcf_2_realisations():
    """A minimal :class:`PartialCoherenceMCF` object for guard tests.

    Built once per module to amortise the
    :func:`create_gaussian_schell_source` cost (10s of ms per call).
    The fields ``N=16``, ``dx=2e-6``, ``wavelength=633e-9``,
    ``w0=10e-6``, ``sigma_g=5e-6`` are the same as the v4.15.3
    dispatcher meta-pin's helper-rejection test, so the same MCF
    is exercised across both pin and regression suites.
    """
    from lumenairy.sources.core import create_gaussian_schell_source
    return create_gaussian_schell_source(
        N=16, dx=2e-6, wavelength=633e-9,
        w0=10e-6, sigma_g=5e-6,
        n_realizations=2, seed=0, return_kind='mcf')


@pytest.fixture(scope='module')
def ensemble_3d():
    """A minimal 3-D ensemble (shape ``(4, 16, 16)``) for guard tests."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((4, 16, 16))
            + 1j * rng.standard_normal((4, 16, 16))).astype(np.complex128)


@pytest.fixture(scope='module')
def field_2d():
    """A baseline-valid 2-D complex field (passes the guard)."""
    return np.ones((16, 16), dtype=np.complex128)


# ============================================================================
# Section A: ``DeformableMirror.apply`` direct guard (P1-NEW-F2-2).
# ============================================================================

def test_deformable_mirror_apply_rejects_mcf(mcf_2_realisations):
    """``DeformableMirror.apply`` must raise ``TypeError`` with the
    canonical v4.16 message when handed a ``PartialCoherenceMCF``.

    v4.15.4 was silently broken: the v4.15.4 walker excluded class
    methods entirely on the documented (but only partly-true)
    assumption that they delegate to module-level guarded scalar
    kernels.  ``DeformableMirror.apply`` is in fact a direct
    ``E_in * np.exp(1j * phi)`` multiply with no delegation, so
    ``dm.apply(mcf)`` failed at the bare arithmetic with an
    unhelpful Python ``TypeError``.  v4.15.5 P1-NEW-F2-2 adds the
    inline guard.
    """
    import lumenairy as la
    dm = la.DeformableMirror(n_actuators=3, pitch=4e-6, dx=2e-6, N=16)
    with pytest.raises(TypeError) as excinfo:
        dm.apply(mcf_2_realisations)
    msg = str(excinfo.value)
    assert 'DeformableMirror.apply' in msg
    assert 'PartialCoherenceMCF' in msg
    # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the rejection
    # message no longer cites the stale "v4.16+" scope marker;
    # it now points at the propagate_ensemble helper.
    assert 'propagate_ensemble' in msg


def test_deformable_mirror_apply_rejects_3d_ensemble(ensemble_3d):
    """``DeformableMirror.apply`` must raise ``ValueError`` with the
    iterate-over-ensemble hint when handed a 3-D NumPy ensemble.

    Pre-v4.15.5 a 3-D ensemble silently broadcast through
    ``E_in * np.exp(1j * phi)`` (the phi is 2-D, so the output is
    3-D) and the caller got a meaninglessly-shaped output instead
    of an error.
    """
    import lumenairy as la
    dm = la.DeformableMirror(n_actuators=3, pitch=4e-6, dx=2e-6, N=16)
    with pytest.raises(ValueError) as excinfo:
        dm.apply(ensemble_3d)
    msg = str(excinfo.value)
    assert 'DeformableMirror.apply' in msg
    assert '3-D' in msg
    assert 'for k in range(ensemble.shape[0])' in msg


def test_deformable_mirror_apply_accepts_2d_field(field_2d):
    """Counter-pin: a valid 2-D complex field must still pass
    through ``DeformableMirror.apply`` without raising.

    Without this counter-pin, an over-strict guard could pass the
    rejection tests above while breaking every legitimate caller.
    """
    import lumenairy as la
    dm = la.DeformableMirror(n_actuators=3, pitch=4e-6, dx=2e-6, N=16)
    out = dm.apply(field_2d)
    assert out.shape == field_2d.shape
    assert out.dtype.kind == 'c'


# ============================================================================
# Section B: ``analysis/core.py`` analyzer guards (parametrized).
# ============================================================================

# Each entry: (name, callable_factory, expected_arg_names_substring_in_msg)
# Some analyzers consume the field as positional 0 ONLY; others (like
# ``coupling_efficiency``) consume two fields.  The callable_factory
# closes over a single ``E`` argument and supplies any additional
# required args (dx, wavelength, mode, etc.) with safe defaults.
_CORE_ANALYZERS = [
    pytest.param(
        'wave_opd_2d',
        lambda E: __import__('lumenairy').wave_opd_2d(
            E, dx=2e-6, wavelength=633e-9),
        id='wave_opd_2d'),
    pytest.param(
        'M2',
        lambda E: __import__('lumenairy').M2(E, dx=2e-6, wavelength=633e-9),
        id='M2'),
    pytest.param(
        'strehl_ratio',
        lambda E: __import__('lumenairy').strehl_ratio(
            E, E, dx=2e-6),
        id='strehl_ratio'),
    pytest.param(
        'beam_d4sigma',
        lambda E: __import__('lumenairy').beam_d4sigma(E, dx=2e-6),
        id='beam_d4sigma'),
    pytest.param(
        'coupling_efficiency',
        lambda E: __import__('lumenairy').coupling_efficiency(
            E, np.ones((16, 16), dtype=np.complex128), dx=2e-6),
        id='coupling_efficiency'),
    pytest.param(
        'compute_psf',
        lambda E: __import__('lumenairy').compute_psf(
            E, wavelength=633e-9, f=1e-2, dx_pupil=2e-6),
        id='compute_psf'),
    pytest.param(
        'compute_otf',
        lambda E: __import__('lumenairy').compute_otf(E),
        id='compute_otf'),
    pytest.param(
        'compute_mtf',
        lambda E: __import__('lumenairy').compute_mtf(E),
        id='compute_mtf'),
    pytest.param(
        'encircled_energy_curve',
        lambda E: __import__('lumenairy').encircled_energy_curve(
            E, dx=2e-6, n_radii=4),
        id='encircled_energy_curve'),
]


@pytest.mark.parametrize('name,fn', _CORE_ANALYZERS)
def test_core_analyzer_rejects_mcf(name, fn, mcf_2_realisations):
    """Parametrized: every guarded ``analysis/core.py`` analyzer must
    reject ``PartialCoherenceMCF`` with the canonical v4.16 message.
    """
    with pytest.raises(TypeError) as excinfo:
        fn(mcf_2_realisations)
    msg = str(excinfo.value)
    assert name in msg
    assert 'PartialCoherenceMCF' in msg
    # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the rejection
    # message no longer cites the stale "v4.16+" scope marker;
    # it now points at the propagate_ensemble helper.
    assert 'propagate_ensemble' in msg


@pytest.mark.parametrize('name,fn', _CORE_ANALYZERS)
def test_core_analyzer_rejects_3d_ensemble(name, fn, ensemble_3d):
    """Parametrized: every guarded ``analysis/core.py`` analyzer must
    reject a 3-D ensemble with the iterate-over-ensemble hint.
    """
    with pytest.raises(ValueError) as excinfo:
        fn(ensemble_3d)
    msg = str(excinfo.value)
    assert name in msg
    assert '3-D' in msg
    assert 'for k in range(ensemble.shape[0])' in msg


# ============================================================================
# Section C: ``analysis/coherence.py`` guards.
# ============================================================================


def _trivial_prescription():
    """Minimal lens prescription dict for ``koehler_image`` /
    ``extended_source_image`` smoke tests.  Uses a small thin-lens
    surface to avoid the long-path real-lens machinery.
    """
    return {
        'wavelength': 633e-9,
        'surfaces': [
            {'type': 'thin_lens', 'focal_length': 0.05},
        ],
    }


def test_koehler_image_rejects_mcf(mcf_2_realisations):
    """``koehler_image(object_field=MCF, ...)`` must reject MCF."""
    import lumenairy as la
    with pytest.raises(TypeError) as excinfo:
        la.koehler_image(
            mcf_2_realisations,
            prescription=_trivial_prescription(),
            wavelength=633e-9, dx=2e-6, n_source_points=2)
    msg = str(excinfo.value)
    assert 'koehler_image' in msg
    assert 'PartialCoherenceMCF' in msg
    # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the rejection
    # message no longer cites the stale "v4.16+" scope marker;
    # it now points at the propagate_ensemble helper.
    assert 'propagate_ensemble' in msg


def test_koehler_image_rejects_3d_ensemble(ensemble_3d):
    """``koehler_image(object_field=3-D ensemble, ...)`` must reject."""
    import lumenairy as la
    with pytest.raises(ValueError) as excinfo:
        la.koehler_image(
            ensemble_3d,
            prescription=_trivial_prescription(),
            wavelength=633e-9, dx=2e-6, n_source_points=2)
    msg = str(excinfo.value)
    assert 'koehler_image' in msg
    assert '3-D' in msg


def test_extended_source_image_rejects_mcf(mcf_2_realisations):
    """``extended_source_image(object_field=MCF, ...)`` must reject."""
    import lumenairy as la
    with pytest.raises(TypeError) as excinfo:
        la.extended_source_image(
            mcf_2_realisations,
            prescription=_trivial_prescription(),
            wavelength=633e-9, dx=2e-6,
            source_angles=[(0.0, 0.0)])
    msg = str(excinfo.value)
    assert 'extended_source_image' in msg
    assert 'PartialCoherenceMCF' in msg
    # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the rejection
    # message no longer cites the stale "v4.16+" scope marker;
    # it now points at the propagate_ensemble helper.
    assert 'propagate_ensemble' in msg


def test_extended_source_image_rejects_3d_ensemble(ensemble_3d):
    """``extended_source_image(object_field=3-D, ...)`` must reject."""
    import lumenairy as la
    with pytest.raises(ValueError) as excinfo:
        la.extended_source_image(
            ensemble_3d,
            prescription=_trivial_prescription(),
            wavelength=633e-9, dx=2e-6,
            source_angles=[(0.0, 0.0)])
    msg = str(excinfo.value)
    assert 'extended_source_image' in msg
    assert '3-D' in msg


# ============================================================================
# Section D: ``analysis/detector.py`` guards.
# ============================================================================

def test_shack_hartmann_rejects_mcf(mcf_2_realisations):
    """``shack_hartmann(E=MCF, ...)`` must reject ``PartialCoherenceMCF``."""
    import lumenairy as la
    with pytest.raises(TypeError) as excinfo:
        la.shack_hartmann(
            mcf_2_realisations,
            dx=2e-6, wavelength=633e-9,
            lenslet_pitch=4e-6, lenslet_focal=1e-3)
    msg = str(excinfo.value)
    assert 'shack_hartmann' in msg
    assert 'PartialCoherenceMCF' in msg
    # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the rejection
    # message no longer cites the stale "v4.16+" scope marker;
    # it now points at the propagate_ensemble helper.
    assert 'propagate_ensemble' in msg


def test_shack_hartmann_rejects_3d_ensemble(ensemble_3d):
    """``shack_hartmann(E=3-D ensemble, ...)`` must reject."""
    import lumenairy as la
    with pytest.raises(ValueError) as excinfo:
        la.shack_hartmann(
            ensemble_3d,
            dx=2e-6, wavelength=633e-9,
            lenslet_pitch=4e-6, lenslet_focal=1e-3)
    msg = str(excinfo.value)
    assert 'shack_hartmann' in msg
    assert '3-D' in msg


# ============================================================================
# Section E: ``raytrace/from_field.py`` and ``propagators/propagation.py``
# guards.
# ============================================================================

def test_rays_from_field_rejects_mcf(mcf_2_realisations):
    """``rays_from_field(E=MCF, ...)`` must reject ``PartialCoherenceMCF``.

    v4.15.1 added an inline ``E.ndim != 2`` check on the
    ``np.asarray(E)`` result; the hoisted v4.15.5 guard runs FIRST
    so MCF inputs get the canonical v4.16 message instead of the
    Python-level ``TypeError`` from ``np.asarray``.
    """
    import lumenairy as la
    with pytest.raises(TypeError) as excinfo:
        la.rays_from_field(
            mcf_2_realisations,
            dx=2e-6, wavelength=633e-9, n_rays=4)
    msg = str(excinfo.value)
    assert 'rays_from_field' in msg
    assert 'PartialCoherenceMCF' in msg
    # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the rejection
    # message no longer cites the stale "v4.16+" scope marker;
    # it now points at the propagate_ensemble helper.
    assert 'propagate_ensemble' in msg


def test_rays_from_field_rejects_3d_ensemble(ensemble_3d):
    """``rays_from_field(E=3-D, ...)`` must reject 3-D ensembles via
    the new shared guard (preserves the v4.15.1 inline check for the
    ``E.ndim != 2`` error message, but now reaches the canonical
    iterate-over-ensemble hint via the helper).
    """
    import lumenairy as la
    with pytest.raises(ValueError) as excinfo:
        la.rays_from_field(
            ensemble_3d,
            dx=2e-6, wavelength=633e-9, n_rays=4)
    msg = str(excinfo.value)
    assert 'rays_from_field' in msg
    assert '3-D' in msg


def test_resample_field_rejects_mcf(mcf_2_realisations):
    """``resample_field(E_in=MCF, ...)`` must reject MCF inputs."""
    import lumenairy as la
    with pytest.raises(TypeError) as excinfo:
        la.resample_field(mcf_2_realisations, dx_in=2e-6, dx_out=4e-6)
    msg = str(excinfo.value)
    assert 'resample_field' in msg
    assert 'PartialCoherenceMCF' in msg
    # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the rejection
    # message no longer cites the stale "v4.16+" scope marker;
    # it now points at the propagate_ensemble helper.
    assert 'propagate_ensemble' in msg


def test_resample_field_rejects_3d_ensemble(ensemble_3d):
    """``resample_field(E_in=3-D, ...)`` must reject 3-D ensembles."""
    import lumenairy as la
    with pytest.raises(ValueError) as excinfo:
        la.resample_field(ensemble_3d, dx_in=2e-6, dx_out=4e-6)
    msg = str(excinfo.value)
    assert 'resample_field' in msg
    assert '3-D' in msg


# ============================================================================
# Section F: V6 walker structural pins (cross-checks the dispatcher
# meta-pin from a different angle).
# ============================================================================

def test_v6_walker_catches_first_positional_param_name():
    """End-to-end pin: the V6 walker imported from the dispatcher
    pin must catch a synthetic ``my_new_analyzer(E, ...)``.

    Cross-checks the dispatcher pin's structural-upgrade counter-
    pin from the regression suite.  If the V6 filter regresses to
    legacy-name-prefix-only the walker would miss this synthetic
    function and the dispatcher-pin counter test would also fail,
    but having the cross-check in the regression suite gives a
    clearer signal during triage.
    """
    from tests.unit.test_v4_15_3_dispatcher_pin_2d_scalar_field import (
        _function_matches_entry_point_filter,
        _legacy_name_matches_entry_point_filter,
    )
    fake_source = (
        'def my_new_analyzer(E, dx, wavelength):\n'
        '    """Synthetic V6-only candidate."""\n'
        '    return E.sum()\n'
    )
    fake_tree = ast.parse(fake_source, filename='<v6-cross>')
    fake_node = next(
        stmt for stmt in fake_tree.body
        if isinstance(stmt, ast.FunctionDef))
    # legacy filter says no (name doesn't match)
    assert not _legacy_name_matches_entry_point_filter('my_new_analyzer')
    # V6 filter says yes (first positional is ``E``)
    assert _function_matches_entry_point_filter(fake_node)


def test_class_method_walker_descent_catches_non_delegating_method():
    """Cross-check: the v4.15.5 P2-NEW-F2-4 class-method walker must
    descend into class bodies and yield ``apply`` / ``propagate``
    methods whose first non-self positional is in
    ``_FIELD_PARAM_NAMES``.
    """
    from tests.unit.test_v4_15_3_dispatcher_pin_2d_scalar_field import (
        _class_method_matches_filter,
    )
    fake_source = (
        'class FakeAnalyzer:\n'
        '    def apply(self, E_in, scale=1.0):\n'
        '        """Synthetic non-delegating class method."""\n'
        '        return E_in * 2.0\n'
    )
    fake_tree = ast.parse(fake_source, filename='<class-cross>')
    fake_class = next(
        stmt for stmt in fake_tree.body
        if isinstance(stmt, ast.ClassDef))
    fake_method = next(
        sub for sub in fake_class.body
        if isinstance(sub, ast.FunctionDef))
    assert _class_method_matches_filter(fake_method)


def test_operator_algebra_apply_methods_exempt():
    """Cross-check: every ``lumenairy.algebra.Operator`` subclass's
    ``apply`` method MUST resolve to the ``Operator.apply``
    delegating wrapper (or be exempt via
    ``_DELEGATING_CLASS_METHODS``).

    Without this pin, a future subclass that overrode ``apply``
    with a non-delegating implementation (e.g. a direct field
    multiply) would silently bypass the guard chain.
    """
    import lumenairy.algebra as alg
    from tests.unit.test_v4_15_3_dispatcher_pin_2d_scalar_field import (
        _DELEGATING_CLASS_METHODS,
    )
    Operator = alg.Operator
    # ``Operator.apply`` is what every subclass inherits.  Confirm
    # the inherited method's qualname points at Operator, not at
    # the subclass.
    for cls_name in ('ThinLens', 'FreeSpace', 'CylindricalLens',
                      'Magnify', 'FourierTransform', 'Aperture',
                      'GaussianAperture'):
        cls = getattr(alg, cls_name, None)
        if cls is None:
            continue
        method = getattr(cls, 'apply')
        # The bound method's __func__ should be Operator.apply.
        # If a subclass overrides apply (i.e. it's NOT delegating)
        # this check fails -- the user would have to add the
        # subclass's (file, classname, 'apply') triple to
        # ``_DELEGATING_CLASS_METHODS`` or add the inline guard.
        assert method is Operator.apply, (
            f"{cls_name}.apply overrides Operator.apply -- if this "
            f"is intentional and the subclass legitimately delegates "
            f"to a guarded scalar function, add "
            f"('lumenairy/algebra/<file>.py', '{cls_name}', 'apply') "
            f"to _DELEGATING_CLASS_METHODS.  If the override does "
            f"NOT delegate (direct field multiply / phase add), add "
            f"an inline _check_2d_scalar_field call instead.")
    # And the base ``Operator.apply`` itself must be in the
    # delegating-class-methods set, so the dispatcher pin's
    # ``test_all_class_methods_call_helper_first`` passes.
    assert ('lumenairy/algebra/base.py', 'Operator', 'apply') in (
        _DELEGATING_CLASS_METHODS)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

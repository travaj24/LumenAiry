"""v4.15.4 / Agent A — regression tests for the 6 newly-guarded
entry points that the v4.15.3 audit (3-way agent convergence)
identified as siblings missed by the v4.15.3 walker scope.

Closes:
- P1-NEW-3WAY-1: ``system.py::propagate_through_system_jax``
  unguarded sibling of the already-guarded
  ``propagate_through_system`` in the SAME file
- P2-NEW-3WAY-2: ``analysis/ao.py::apply_dm``,
  ``analysis/detector.py::apply_detector``,
  ``analysis/through_focus.py::apply_perturbations``
- P2-NEW-V2-1: ``propagators/vector_diffraction.py``'s
  ``richards_wolf_focus`` and ``debye_wolf_psf`` (missed by the
  walker's v4.15.3 name regex)

Five of the six new sites accept 2-D scalar fields and now call
``_check_2d_scalar_field`` directly (matching the v4.15.3
guard-the-first-statement pattern).  ``apply_perturbations``
takes a prescription DICT (not a 2-D scalar field), so it is
documented as an exempt entry in the dispatcher meta-pin's
``_GUARD_EXEMPTIONS`` registry rather than guarded; the
regression test for it verifies the exempt registration.

Plus one walker-scope pin (``test_walker_now_visits_*``)
asserting v4.15.4's package-walk scope extension catches the
``system.py`` + ``analysis/`` siblings the v4.15.3 walker had
missed.

Author: Andrew Traverso -- v4.15.4 / Agent A
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# Helper: build a v4.16-deferred PartialCoherenceMCF.
# Same construction pattern as the v4.15.3 dispatcher-pin tests, factored
# out so each regression test can stay focused on the sibling-specific
# behaviour.
# ============================================================================

def _make_mcf():
    """Return a small ``PartialCoherenceMCF`` instance for guard tests."""
    from lumenairy.sources.core import create_gaussian_schell_source
    return create_gaussian_schell_source(
        N=16, dx=2e-6, wavelength=633e-9,
        w0=10e-6, sigma_g=5e-6,
        n_realizations=2, seed=0, return_kind='mcf')


def _make_3d_ensemble(n=4, ny=16, nx=16):
    """Return a 3-D ensemble of shape ``(n, ny, nx)`` (the
    Schell-source ensemble shape post-v4.15.1)."""
    return np.zeros((n, ny, nx), dtype=np.complex128)


# ============================================================================
# P1-NEW-3WAY-1: propagate_through_system_jax
# ============================================================================

def test_propagate_through_system_jax_rejects_mcf():
    """``propagate_through_system_jax`` must raise ``TypeError`` with
    the v4.16-roadmap message when handed a ``PartialCoherenceMCF``.

    Pre-v4.15.4 this function had no guard (walker scope didn't visit
    ``system.py``), so passing an MCF would leak a confusing
    ``TracerArrayConversionError`` from the downstream
    ``jnp.asarray(E_in, ...)`` cast.  Post-v4.15.4 the
    ``_check_2d_scalar_field`` helper runs FIRST and returns the
    canonical v4.16-deferred message.
    """
    pytest.importorskip('jax')
    from lumenairy.system import propagate_through_system_jax

    mcf = _make_mcf()
    elements = [{'type': 'propagate', 'z': 0.0}]
    with pytest.raises(TypeError) as excinfo:
        propagate_through_system_jax(
            mcf, elements, wavelength=633e-9, dx=2e-6)
    msg = str(excinfo.value)
    assert 'PartialCoherenceMCF' in msg
    assert 'v4.16' in msg
    assert 'propagate_through_system_jax' in msg


def test_propagate_through_system_jax_rejects_3d_ensemble():
    """``propagate_through_system_jax`` must raise ``ValueError``
    with the iterate-over-ensemble hint when handed a 3-D ensemble.
    """
    pytest.importorskip('jax')
    from lumenairy.system import propagate_through_system_jax

    ensemble = _make_3d_ensemble()
    elements = [{'type': 'propagate', 'z': 0.0}]
    with pytest.raises(ValueError) as excinfo:
        propagate_through_system_jax(
            ensemble, elements, wavelength=633e-9, dx=2e-6)
    msg = str(excinfo.value)
    assert '3-D' in msg
    assert 'propagate_through_system_jax' in msg
    assert 'for k in range(ensemble.shape[0])' in msg


# ============================================================================
# P2-NEW-3WAY-2: apply_dm
# ============================================================================

def test_apply_dm_rejects_mcf_and_3d_ensemble():
    """``apply_dm`` must reject ``PartialCoherenceMCF`` (TypeError,
    v4.16 marker) and 3-D ensemble (ValueError, ensemble-iteration
    hint).  Both routed through ``_check_2d_scalar_field``.
    """
    from lumenairy.analysis.ao import apply_dm, DeformableMirror

    # Small DM compatible with the field grids below.
    N = 16
    dx = 2e-6
    dm = DeformableMirror(n_actuators=3, pitch=4 * dx, dx=dx, N=N)

    # ---- MCF rejection (TypeError, v4.16 marker) -----------------------
    mcf = _make_mcf()
    with pytest.raises(TypeError) as excinfo:
        apply_dm(mcf, dm)
    msg = str(excinfo.value)
    assert 'PartialCoherenceMCF' in msg
    assert 'v4.16' in msg
    assert 'apply_dm' in msg

    # ---- 3-D ensemble rejection (ValueError, iteration hint) -----------
    ensemble = _make_3d_ensemble(n=4, ny=N, nx=N)
    with pytest.raises(ValueError) as excinfo:
        apply_dm(ensemble, dm)
    msg = str(excinfo.value)
    assert '3-D' in msg
    assert 'apply_dm' in msg
    assert 'for k in range(ensemble.shape[0])' in msg

    # ---- Smoke test: legitimate 2-D input must NOT raise --------------
    E_in = np.ones((N, N), dtype=np.complex128)
    E_out = apply_dm(E_in, dm)
    assert E_out.shape == (N, N)


# ============================================================================
# P2-NEW-3WAY-2: apply_detector
# ============================================================================

def test_apply_detector_rejects_mcf_and_3d_ensemble():
    """``apply_detector`` must reject ``PartialCoherenceMCF`` and
    3-D ensemble via the shared helper.
    """
    from lumenairy.analysis.detector import apply_detector

    # ---- MCF rejection (TypeError, v4.16 marker) -----------------------
    mcf = _make_mcf()
    with pytest.raises(TypeError) as excinfo:
        apply_detector(mcf, dx_field=2e-6, pixel_pitch=4e-6)
    msg = str(excinfo.value)
    assert 'PartialCoherenceMCF' in msg
    assert 'v4.16' in msg
    assert 'apply_detector' in msg

    # ---- 3-D ensemble rejection (ValueError, iteration hint) -----------
    ensemble = _make_3d_ensemble()
    with pytest.raises(ValueError) as excinfo:
        apply_detector(ensemble, dx_field=2e-6, pixel_pitch=4e-6)
    msg = str(excinfo.value)
    assert '3-D' in msg
    assert 'apply_detector' in msg
    assert 'for k in range(ensemble.shape[0])' in msg


# ============================================================================
# P2-NEW-3WAY-2: apply_perturbations
# ----------------------------------------------------------------------------
# Note: ``apply_perturbations`` takes a prescription DICT (not a 2-D
# scalar field), so it is NOT guarded with the helper.  Instead it is
# documented in the dispatcher meta-pin's ``_GUARD_EXEMPTIONS``
# registry with the reason cited inline.  The regression test below
# pins that exempt registration so a future contributor who tries to
# guard the function in error gets the meta-pin assertion to flip
# them back to the exemption-update path.
# ============================================================================

def test_apply_perturbations_is_documented_exempt():
    """``apply_perturbations`` operates on a prescription dict and is
    NOT a 2-D-scalar-field entry point.  The dispatcher meta-pin's
    ``_GUARD_EXEMPTIONS`` registry MUST list it (with cited reason)
    so the meta-pin doesn't trip on a function whose first arg has
    no ``.ndim``.

    Counter-pin: if a future contributor deletes the exemption (and
    the function still has no guard) the v4.15.4 dispatcher-pin
    ``test_all_entry_points_call_helper_first`` would fail; this
    test surfaces the exempt-registration as the canonical reason.
    """
    # Import the exemption set from the dispatcher pin module.
    # Import-by-path so this regression test stays self-contained
    # even if the pin module's filename changes.
    from tests.unit.test_v4_15_3_dispatcher_pin_2d_scalar_field import (
        _GUARD_EXEMPTIONS,
    )

    assert (
        'lumenairy/analysis/through_focus.py', 'apply_perturbations'
    ) in _GUARD_EXEMPTIONS, (
        "apply_perturbations must be in _GUARD_EXEMPTIONS: it takes "
        "a prescription dict, not a 2-D scalar field, and guarding "
        "it would break every legitimate caller. If you are removing "
        "this exemption, you must either guard the function (which "
        "will break the legitimate dict-callers) or change the "
        "function signature.")


# ============================================================================
# P2-NEW-V2-1: richards_wolf_focus
# ============================================================================

def test_richards_wolf_focus_rejects_mcf_and_3d_ensemble():
    """``richards_wolf_focus`` must reject ``PartialCoherenceMCF``
    and 3-D ensemble via the shared helper.  Pre-v4.15.4 the
    walker visited ``propagators/vector_diffraction.py`` but the
    name regex (``apply_*`` / ``*_propagate*`` / ``propagate``)
    didn't match ``richards_wolf_focus``.  v4.15.4 broadens the
    regex via ``__all__``-membership discovery so the function is
    now in the meta-pin's scope, and the function now calls the
    helper as its first executable statement.
    """
    from lumenairy.propagators.vector_diffraction import richards_wolf_focus

    # ---- MCF rejection (TypeError, v4.16 marker) -----------------------
    mcf = _make_mcf()
    with pytest.raises(TypeError) as excinfo:
        richards_wolf_focus(
            mcf, wavelength=633e-9, NA=0.5, f=1e-3, dx_pupil=2e-6)
    msg = str(excinfo.value)
    assert 'PartialCoherenceMCF' in msg
    assert 'v4.16' in msg
    assert 'richards_wolf_focus' in msg

    # ---- 3-D ensemble rejection (ValueError, iteration hint) -----------
    ensemble = _make_3d_ensemble()
    with pytest.raises(ValueError) as excinfo:
        richards_wolf_focus(
            ensemble, wavelength=633e-9, NA=0.5, f=1e-3, dx_pupil=2e-6)
    msg = str(excinfo.value)
    assert '3-D' in msg
    assert 'richards_wolf_focus' in msg
    assert 'for k in range(ensemble.shape[0])' in msg


# ============================================================================
# P2-NEW-V2-1: debye_wolf_psf
# ============================================================================

def test_debye_wolf_psf_rejects_mcf_and_3d_ensemble():
    """``debye_wolf_psf`` must reject ``PartialCoherenceMCF`` and
    3-D ensemble via the shared helper, surfacing the function's
    own name (not the underlying ``richards_wolf_focus``) in the
    error message.
    """
    from lumenairy.propagators.vector_diffraction import debye_wolf_psf

    # ---- MCF rejection (TypeError, v4.16 marker) -----------------------
    mcf = _make_mcf()
    with pytest.raises(TypeError) as excinfo:
        debye_wolf_psf(
            mcf, wavelength=633e-9, NA=0.5, f=1e-3, dx_pupil=2e-6)
    msg = str(excinfo.value)
    assert 'PartialCoherenceMCF' in msg
    assert 'v4.16' in msg
    # Crucially: ``debye_wolf_psf`` raises with its OWN name, not
    # the downstream ``richards_wolf_focus``.  This is the value of
    # guarding both functions independently rather than only the
    # delegate.
    assert 'debye_wolf_psf' in msg

    # ---- 3-D ensemble rejection (ValueError, iteration hint) -----------
    ensemble = _make_3d_ensemble()
    with pytest.raises(ValueError) as excinfo:
        debye_wolf_psf(
            ensemble, wavelength=633e-9, NA=0.5, f=1e-3, dx_pupil=2e-6)
    msg = str(excinfo.value)
    assert '3-D' in msg
    assert 'debye_wolf_psf' in msg
    assert 'for k in range(ensemble.shape[0])' in msg


# ============================================================================
# Walker scope extension pin: v4.15.4's _TARGET_PACKAGES now includes
# ``lumenairy/analysis`` and the file-at-package-root ``lumenairy/system.py``.
# This test pins that the package-walk fallback actually discovers files
# from both new scopes (so a future revert of _TARGET_PACKAGES would fail).
# ============================================================================

def test_walker_now_visits_system_py_and_analysis_package():
    """The dispatcher meta-pin's walker must now visit
    ``lumenairy/system.py`` and the ``lumenairy/analysis/``
    subpackage.

    Pin enforced via two channels:
    (1) ``_iter_target_py_files`` yields at least one file under
        ``analysis/`` and the ``system.py`` file itself.
    (2) ``_walk_entry_points`` (full walk, after dedup) yields at
        least one entry-point function from each new scope.

    Without this pin a one-line revert of ``_TARGET_PACKAGES`` to
    the v4.15.3 ``('lumenairy/propagators', 'lumenairy/elements')``
    pair would silently restore the v4.15.3 sibling gap and pass the
    pre-existing tests in the dispatcher-pin file.
    """
    from tests.unit.test_v4_15_3_dispatcher_pin_2d_scalar_field import (
        _TARGET_PACKAGES,
        _iter_target_py_files,
        _walk_entry_points,
    )

    # ---- Pin (a): _TARGET_PACKAGES contents -----------------------------
    assert 'lumenairy/analysis' in _TARGET_PACKAGES, (
        "_TARGET_PACKAGES must include 'lumenairy/analysis' so the "
        "v4.15.3 sibling-gap finding (analysis/ao.py::apply_dm etc.) "
        "cannot recur.")
    assert 'lumenairy/system.py' in _TARGET_PACKAGES, (
        "_TARGET_PACKAGES must include the file-at-package-root "
        "'lumenairy/system.py' so the v4.15.3 sibling-gap finding "
        "(propagate_through_system_jax in the SAME file as the "
        "guarded propagate_through_system) cannot recur.")

    # ---- Pin (b): _iter_target_py_files reaches both scopes -------------
    visited = list(_iter_target_py_files())
    visited_rel = [p.relative_to(_REPO_ROOT).as_posix() for p in visited]
    # At least one file in analysis/
    assert any(rel.startswith('lumenairy/analysis/')
               for rel in visited_rel), (
        f"Package walk did not yield any file in lumenairy/analysis/. "
        f"Got: {visited_rel}")
    # The system.py file itself
    assert 'lumenairy/system.py' in visited_rel, (
        f"Package walk did not yield lumenairy/system.py as a "
        f"file-at-package-root target. Got: {visited_rel}")

    # ---- Pin (c): _walk_entry_points discovers known new entries -------
    discovered_names = set()
    for py, node in _walk_entry_points():
        rel = py.relative_to(_REPO_ROOT).as_posix()
        discovered_names.add((rel, node.name))
    # propagate_through_system_jax (system.py)
    assert ('lumenairy/system.py',
            'propagate_through_system_jax') in discovered_names, (
        "Walker must discover propagate_through_system_jax in "
        "lumenairy/system.py.")
    # apply_dm (analysis/ao.py)
    assert ('lumenairy/analysis/ao.py',
            'apply_dm') in discovered_names, (
        "Walker must discover apply_dm in lumenairy/analysis/ao.py.")
    # apply_detector (analysis/detector.py)
    assert ('lumenairy/analysis/detector.py',
            'apply_detector') in discovered_names, (
        "Walker must discover apply_detector in "
        "lumenairy/analysis/detector.py.")
    # richards_wolf_focus, debye_wolf_psf (propagators/vector_diffraction.py)
    assert ('lumenairy/propagators/vector_diffraction.py',
            'richards_wolf_focus') in discovered_names, (
        "Walker must discover richards_wolf_focus via the v4.15.4 "
        "broadened name filter (richards_wolf_*).")
    assert ('lumenairy/propagators/vector_diffraction.py',
            'debye_wolf_psf') in discovered_names, (
        "Walker must discover debye_wolf_psf via the v4.15.4 "
        "broadened name filter (debye_wolf_*).")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

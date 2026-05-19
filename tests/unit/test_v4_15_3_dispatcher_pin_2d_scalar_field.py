"""Dispatcher meta-pin (v4.15.3): every public ``def apply_*`` and
``def *_propagate*`` in ``lumenairy/propagators/`` and
``lumenairy/elements/`` must call ``_check_2d_scalar_field(E, ...)``
as its first executable statement (after the docstring and any
helper-import statements).

Counter-measure to the recurring 'fix N, miss N+1' meta-pattern
the v4.15.0 -> v4.15.2 audit rounds repeatedly identified.
v4.15.2 hand-guarded 10 entry points; v4.15.3 closes 9 more
siblings and pins the invariant structurally so the 20th entry
point can't be added unguarded.

The walker discovers top-level public ``def apply_*`` /
``def *_propagate*`` functions across the propagator and lens
packages.  Every discovered function must EITHER call
``_check_2d_scalar_field`` early in its body OR appear in the
documented ``_GUARD_EXEMPTIONS`` set with a cited reason.

The exemption set lists every discovered function that legitimately
does NOT accept a bare 2-D coherent scalar field (e.g.
``apply_thin_lens_to_beamlets`` takes a ``BeamletBundle``,
``angular_spectrum_propagate_batch`` requires a 3-D stack,
JonesField methods operate on Jones-vector fields).  Adding the
exemption with a comment is how a new non-scalar entry point lands
without tripping the pin; adding a new scalar entry point WITHOUT
guarding it is the failure mode this pin catches.

Author: Andrew Traverso -- v4.15.3 / Agent A
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TARGET_PACKAGES = ('lumenairy/propagators', 'lumenairy/elements')


# ============================================================================
# Documented exemptions.  Each entry is a (module_relative_posix_path,
# function_name) tuple.  Every exemption MUST cite the reason it
# legitimately bypasses the 2-D scalar-field guard.
# ============================================================================

_GUARD_EXEMPTIONS = frozenset({
    # ---- propagators/dispatch.py --------------------------------------------
    # ``asm_propagate`` is a thin wrapper that delegates to one of the
    # five guarded ASM-family propagators (asm / asm_tilted / asm_mft /
    # sas / fraunhofer).  Every code path through this function reaches a
    # guarded entry point, so the wrapper itself is structurally
    # delegated.  Not in the v4.15.3 P0-NEW-F2-1 list.
    ('lumenairy/propagators/dispatch.py', 'asm_propagate'),

    # ---- propagators/propagation.py ----------------------------------------
    # ``apply_fresnel_curvature`` takes a 2-D field but is a phase
    # multiplication helper (not a propagator); it is not in the
    # v4.15.3 P0-NEW-F2-1 list.  Adding the guard would be in-scope
    # for a future audit round (re-evaluate at v4.16).
    ('lumenairy/propagators/propagation.py', 'apply_fresnel_curvature'),
    # ``angular_spectrum_propagate_batch`` is the batched 3-D-stack
    # variant; its contract REQUIRES ``E_stack.ndim == 3``.  Routing
    # this through ``_check_2d_scalar_field`` would invert the
    # invariant and reject the only valid input shape.  The function
    # has its own dedicated 3-D-shape check.
    ('lumenairy/propagators/propagation.py',
     'angular_spectrum_propagate_batch'),

    # ---- propagators/gbd.py ------------------------------------------------
    # The Gaussian-beam-decomposition propagators accept BeamletBundle
    # / GBDPropagationState objects -- not 2-D scalar fields.  They
    # have their own type checks at entry; the 2-D scalar-field guard
    # would reject every legitimate input.
    ('lumenairy/propagators/gbd.py', 'propagate_beamlets_freespace'),
    ('lumenairy/propagators/gbd.py', 'apply_thin_lens_to_beamlets'),
    ('lumenairy/propagators/gbd.py', 'propagate_gbd'),
    ('lumenairy/propagators/gbd.py', 'propagate_gbd_freespace'),
    ('lumenairy/propagators/gbd.py', 'propagate_gbd_thin_lens'),
    ('lumenairy/propagators/gbd.py', 'apply_abcd_to_beamlets'),
    ('lumenairy/propagators/gbd.py', 'propagate_gbd_through_prescription'),

    # ---- propagators/hf.py -------------------------------------------------
    # The Huygens-Fresnel propagators accept Huygens-Fresnel state
    # objects (path-bundle / opl-callable), not 2-D scalar fields.
    ('lumenairy/propagators/hf.py', 'propagate_huygens_fresnel'),
    ('lumenairy/propagators/hf.py', 'propagate_huygens_fresnel_freespace'),
    ('lumenairy/propagators/hf.py',
     'propagate_huygens_fresnel_with_opl_callable'),
    ('lumenairy/propagators/hf.py',
     'propagate_huygens_fresnel_through_prescription'),

    # ---- propagators/hfpi.py -----------------------------------------------
    # HFPI propagators / aperture-diffraction helpers operate on
    # PathBundle objects, not 2-D scalar fields.
    ('lumenairy/propagators/hfpi.py', 'propagate_to_plane'),
    ('lumenairy/propagators/hfpi.py', 'apply_aperture_diffraction'),
    ('lumenairy/propagators/hfpi.py', 'propagate_hfpi'),
    ('lumenairy/propagators/hfpi.py', 'propagate_hfpi_freespace_aperture'),
    ('lumenairy/propagators/hfpi.py', 'propagate_hfpi_through_prescription'),

    # ---- propagators/vectorial_hfpi.py -------------------------------------
    # Vectorial-HFPI propagators operate on VectorPathBundle objects,
    # not 2-D scalar fields.
    ('lumenairy/propagators/vectorial_hfpi.py', 'propagate_vector_to_plane'),
    ('lumenairy/propagators/vectorial_hfpi.py',
     'apply_vector_aperture_diffraction'),
    ('lumenairy/propagators/vectorial_hfpi.py',
     'propagate_vector_hfpi_freespace_aperture'),

    # ---- propagators/asymptotic.py -----------------------------------------
    # The asymptotic / modal propagators take modal-coefficient
    # representations and ray bundles, not 2-D scalar fields.
    ('lumenairy/propagators/asymptotic.py', 'propagate_modal_asymptotic'),
    ('lumenairy/propagators/asymptotic.py',
     'propagate_hf_chebyshev_quadrature'),
    ('lumenairy/propagators/asymptotic.py',
     'propagate_modal_asymptotic_lg00_jax'),

    # ---- propagators/subaperture.py ----------------------------------------
    # Subaperture propagator takes a subaperture-state object (dict +
    # tile-local field tuple), not a single 2-D scalar field.  Its own
    # entry-point validation surfaces the appropriate error.
    ('lumenairy/propagators/subaperture.py',
     'propagate_subaperture_asymptotic'),

    # ---- elements/elements.py ----------------------------------------------
    # The generic-element ``apply_*`` helpers (mirror, aperture, mask,
    # zernike-aberration, Lyot stop, vortex-phase mask, apodized-pupil)
    # DO take 2-D scalar fields but are NOT in the v4.15.3 P0-NEW-F2-1
    # closure scope.  Re-evaluate at v4.16 (likely candidates for the
    # next round of guard migration).
    ('lumenairy/elements/elements.py', 'apply_mirror'),
    ('lumenairy/elements/elements.py', 'apply_aperture'),
    ('lumenairy/elements/elements.py', 'apply_gaussian_aperture'),
    ('lumenairy/elements/elements.py', 'apply_mask'),
    ('lumenairy/elements/elements.py', 'apply_zernike_aberration'),
    ('lumenairy/elements/elements.py', 'apply_lyot_focal_plane_mask'),
    ('lumenairy/elements/elements.py', 'apply_vortex_phase_mask'),
    ('lumenairy/elements/elements.py', 'apply_lyot_stop'),
    ('lumenairy/elements/elements.py', 'apply_apodized_pupil'),

    # ---- elements/_lens_jax.py ---------------------------------------------
    # JAX-traceable lens variants are NOT in the v4.15.3 P0-NEW-F2-1
    # closure scope.  Their guard plan is coupled to the v4.16 JAX
    # autodiff partial-coherence story.
    ('lumenairy/elements/_lens_jax.py', 'apply_real_lens_traced_jax'),
    ('lumenairy/elements/_lens_jax.py', 'apply_real_lens_maslov_jax'),

    # ---- elements/polarization.py ------------------------------------------
    # Top-level polarization helpers accept a ``JonesField`` object,
    # NOT a 2-D scalar field.  The JonesField wraps two 2-D fields
    # (Ex, Ey) and these helpers operate on the pair, dispatching
    # element-wise through the already-guarded scalar lens / mirror
    # / aperture / mask kernels for the non-polarizing variants.  A
    # 2-D scalar input to these would fail at attribute access.
    ('lumenairy/elements/polarization.py', 'apply_jones_matrix'),
    ('lumenairy/elements/polarization.py', 'apply_polarizer'),
    ('lumenairy/elements/polarization.py', 'apply_waveplate'),
    ('lumenairy/elements/polarization.py', 'apply_half_wave_plate'),
    ('lumenairy/elements/polarization.py', 'apply_quarter_wave_plate'),
    ('lumenairy/elements/polarization.py', 'apply_rotator'),
})


# ============================================================================
# Discovery walker -- top-level public defs only
# ============================================================================

def _walk_entry_points():
    """Yield (path, ast.FunctionDef) for every public top-level def in
    scope.  Class methods are intentionally excluded -- the guard
    contract is for the top-level public API; per-class wrapper methods
    delegate to the guarded scalar functions.
    """
    for pkg_rel in _TARGET_PACKAGES:
        pkg_root = _REPO_ROOT / pkg_rel
        for py in sorted(pkg_root.rglob('*.py')):
            if py.name.startswith('_') and 'lens' not in py.name:
                # Skip private modules but keep _lens_*.py
                continue
            src = py.read_text(encoding='utf-8', errors='replace')
            tree = ast.parse(src, filename=str(py))
            # Top-level FunctionDef only.  Class methods are walked
            # separately by the per-class delegation pattern and are
            # not in scope for this pin.
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                name = node.name
                if name.startswith('_'):
                    continue
                if not (name.startswith('apply_')
                        or '_propagate' in name
                        or name == 'propagate'):
                    continue
                yield py, node


def _first_executable_statement(body):
    """Return the first executable statement in ``body``, skipping a
    leading docstring AND any leading ``Import`` / ``ImportFrom``
    statements (which support the lazy-import pattern the guard uses
    to avoid circular dependencies).
    """
    skip_leading_imports = True
    saw_docstring = False
    for stmt in body:
        # Skip the docstring (first Expr containing a Constant str).
        if (not saw_docstring
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)):
            saw_docstring = True
            continue
        # Skip leading import statements (helper's lazy-import lives
        # alongside the call).
        if skip_leading_imports and isinstance(
                stmt, (ast.Import, ast.ImportFrom)):
            continue
        skip_leading_imports = False
        return stmt
    return None


def _is_helper_call(stmt):
    """True iff ``stmt`` is a bare ``_check_2d_scalar_field(...)``
    expression statement.
    """
    if not isinstance(stmt, ast.Expr):
        return False
    if not isinstance(stmt.value, ast.Call):
        return False
    func = stmt.value.func
    if isinstance(func, ast.Name) and func.id == '_check_2d_scalar_field':
        return True
    if (isinstance(func, ast.Attribute)
            and func.attr == '_check_2d_scalar_field'):
        return True
    return False


# ============================================================================
# Pre-flight: count floor.  Catches the case where the walker silently
# collapses (e.g. a refactor renames the package layout).
# ============================================================================

def test_minimum_entry_points_discovered():
    """Sanity floor: at least 19 entry points must be discoverable.

    The v4.15.3 closure migrates 10 v4.15.2-guarded sites + 9 new
    siblings, plus the walker will also discover roughly 30 exempt
    siblings (BeamletBundle / Jones-field / generic-element /
    JAX-lens etc.).  Below 19 means the walker is broken.
    """
    count = sum(1 for _ in _walk_entry_points())
    assert count >= 19, (
        f"Walker found only {count} entry points; expected >= 19. "
        f"The discovery walker may need updating if the package "
        f"layout has changed.")


# ============================================================================
# Main pin: every discovered entry point either calls the helper OR
# appears in the exemption set with a documented reason.
# ============================================================================

def test_all_entry_points_call_helper_first():
    """Every public propagator/lens entry point must call
    ``_check_2d_scalar_field`` as its first executable statement
    (after the docstring and any leading import statements).
    Exempted functions must be listed in ``_GUARD_EXEMPTIONS`` with
    a cited reason.
    """
    failures = []
    for py, node in _walk_entry_points():
        rel = py.relative_to(_REPO_ROOT).as_posix()
        if (rel, node.name) in _GUARD_EXEMPTIONS:
            continue
        first_stmt = _first_executable_statement(node.body)
        if first_stmt is None:
            failures.append(
                f"{rel}:{node.lineno} {node.name}: empty body")
            continue
        if not _is_helper_call(first_stmt):
            failures.append(
                f"{rel}:{node.lineno} {node.name}: "
                f"first executable statement is not "
                f"`_check_2d_scalar_field(...)`. Either add the "
                f"guard call (see lumenairy/_validation.py) or, if "
                f"this function legitimately does not accept a 2-D "
                f"scalar field, add it to ``_GUARD_EXEMPTIONS`` "
                f"with a comment explaining why.")
    if failures:
        raise AssertionError(
            "Entry points missing _check_2d_scalar_field guard:\n  - "
            + "\n  - ".join(failures))


# ============================================================================
# Counter-pins on the helper itself: smoke-test the message contract.
# ============================================================================

def test_helper_rejects_mcf_with_v4_16_message():
    """``_check_2d_scalar_field`` must raise ``TypeError`` with a
    ``v4.16``-scope marker when handed a ``PartialCoherenceMCF``.
    """
    from lumenairy._validation import _check_2d_scalar_field
    from lumenairy.sources.core import create_gaussian_schell_source

    mcf = create_gaussian_schell_source(
        N=16, dx=2e-6, wavelength=633e-9,
        w0=10e-6, sigma_g=5e-6,
        n_realizations=2, seed=0, return_kind='mcf')
    with pytest.raises(TypeError) as excinfo:
        _check_2d_scalar_field(mcf, 'unit_test_fn')
    msg = str(excinfo.value)
    assert 'PartialCoherenceMCF' in msg
    assert 'v4.16' in msg
    assert 'unit_test_fn' in msg


def test_helper_rejects_3d_ensemble_with_iteration_hint():
    """``_check_2d_scalar_field`` must raise ``ValueError`` with the
    canonical iterate-over-ensemble hint when handed a 3-D ensemble.
    """
    import numpy as np

    from lumenairy._validation import _check_2d_scalar_field

    ensemble = np.zeros((4, 16, 16), dtype=np.complex128)
    with pytest.raises(ValueError) as excinfo:
        _check_2d_scalar_field(ensemble, 'unit_test_fn')
    msg = str(excinfo.value)
    assert '3-D' in msg
    assert 'unit_test_fn' in msg
    assert 'for k in range(ensemble.shape[0])' in msg


def test_helper_rejects_1d_with_tailored_hint():
    """``_check_2d_scalar_field`` must raise ``ValueError`` with the
    tailored 1-D / scalar diagnostic when handed a non-2-D ndarray.
    """
    import numpy as np

    from lumenairy._validation import _check_2d_scalar_field

    one_d = np.zeros(16, dtype=np.complex128)
    with pytest.raises(ValueError) as excinfo:
        _check_2d_scalar_field(one_d, 'unit_test_fn')
    msg = str(excinfo.value)
    assert '1-D' in msg
    assert 'unit_test_fn' in msg
    # The tailored hint should mention the wrong-object guidance.
    assert ('2-D NumPy/CuPy complex array' in msg
            or 'scalar' in msg.lower())


def test_helper_accepts_2d_complex_array():
    """Sanity: the helper must NOT raise on a legitimate 2-D complex
    ndarray.  Without this counter-pin a too-strict helper would
    block every legitimate caller silently.
    """
    import numpy as np

    from lumenairy._validation import _check_2d_scalar_field

    E = np.zeros((16, 16), dtype=np.complex128)
    # Should not raise.
    _check_2d_scalar_field(E, 'unit_test_fn')


# ============================================================================
# Diagnostic helper (printed when -v is on; useful when adding a new
# entry point and triaging discovery).
# ============================================================================

def test_discovered_entry_points_list_for_diagnostics():
    """Emit the discovered entry-point list to stdout for triage.

    Not really a test -- always passes -- but provides a single
    inspection point so ``pytest -v`` shows the discovery walker's
    output without needing to run the full pin.
    """
    discovered = []
    for py, node in _walk_entry_points():
        rel = py.relative_to(_REPO_ROOT).as_posix()
        tag = ('exempt' if (rel, node.name) in _GUARD_EXEMPTIONS
               else 'guarded')
        discovered.append((rel, node.lineno, node.name, tag))
    print(f"\nv4.15.3 _check_2d_scalar_field meta-pin discovered "
          f"{len(discovered)} entry points:")
    for rel, lineno, name, tag in discovered:
        print(f"  [{tag}] {rel}:{lineno}  {name}")
    assert isinstance(discovered, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""Dispatcher meta-pin (v4.15.3 / extended v4.15.4): every public
``def apply_*`` / ``def *_propagate*`` / ``def richards_wolf*`` /
``def debye_wolf*`` in ``lumenairy/propagators/``,
``lumenairy/elements/``, ``lumenairy/analysis/``, and the
file-at-package-root ``lumenairy/system.py`` must call
``_check_2d_scalar_field(E, ...)`` as its first executable
statement (after the docstring and any helper-import statements).

Counter-measure to the recurring 'fix N, miss N+1' meta-pattern
the v4.15.0 -> v4.15.2 audit rounds repeatedly identified.
v4.15.2 hand-guarded 10 entry points; v4.15.3 closes 9 more
siblings and pins the invariant structurally so the 20th entry
point can't be added unguarded.

v4.15.4 closes the meta-pattern recurrence the v4.15.3 audit
identified one level of indirection higher: the walker had scoped
only to ``propagators/`` + ``elements/``, missing 4 public entry
points in ``system.py`` (``propagate_through_system_jax``) and
``analysis/`` (``apply_dm``, ``apply_detector``,
``apply_perturbations``) plus 2 functions in scope but missed by
the name regex (``richards_wolf_focus``, ``debye_wolf_psf``).
v4.15.4 broadens discovery via ``lumenairy.__all__`` membership
(robust against future module reorganisation) AND extends the
package-walk scope so non-``__all__`` siblings are still caught.

The walker discovers top-level public ``def apply_*`` /
``def *_propagate*`` / ``def richards_wolf*`` / ``def debye_wolf*``
functions across the propagator, lens, analysis, and
system-at-root packages.  Every discovered function must EITHER
call ``_check_2d_scalar_field`` early in its body OR appear in the
documented ``_GUARD_EXEMPTIONS`` set with a cited reason.

The exemption set lists every discovered function that legitimately
does NOT accept a bare 2-D coherent scalar field (e.g.
``apply_thin_lens_to_beamlets`` takes a ``BeamletBundle``,
``angular_spectrum_propagate_batch`` requires a 3-D stack,
``apply_perturbations`` takes a prescription dict, JonesField
methods operate on Jones-vector fields).  Adding the exemption
with a comment is how a new non-scalar entry point lands
without tripping the pin; adding a new scalar entry point WITHOUT
guarding it is the failure mode this pin catches.

Author: Andrew Traverso -- v4.15.3 / Agent A
                          v4.15.4 / Agent A (scope extension)
"""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]

# v4.15.4 (P1-NEW-3WAY-1 + P2-NEW-3WAY-2): walker scope extended.
# v4.15.3 was ``('lumenairy/propagators', 'lumenairy/elements')``;
# the v4.15.3 audit found 4 public entry points outside that scope
# accepting 2-D scalar fields.  v4.15.4 adds ``lumenairy/analysis``
# (subpackage) and ``lumenairy/system.py`` (a file at package
# root -- the walker handles both cases via ``rglob`` of a
# directory and direct visit of a file).
#
# This is now a FALLBACK scope: primary discovery uses
# ``lumenairy.__all__`` (see ``_walk_entry_points``); the package
# walk catches any sibling not exported via ``__all__``.
_TARGET_PACKAGES = ('lumenairy/propagators',
                    'lumenairy/elements',
                    'lumenairy/analysis',
                    'lumenairy/system.py')


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

    # ---- analysis/through_focus.py (v4.15.4 NEW) ---------------------------
    # ``apply_perturbations(prescription, perturbations, N, dx)`` takes
    # a prescription DICT (not a 2-D scalar field) and returns a
    # deep-copied perturbed prescription.  It is matched by the
    # walker's ``apply_*`` name filter, but the first positional arg
    # has no ``.ndim`` and is not the kind of object the guard checks.
    # Adding the guard would make every legitimate call raise
    # ``ValueError`` at the first line.  Documented exempt.
    ('lumenairy/analysis/through_focus.py', 'apply_perturbations'),

    # ---- system.py (v4.15.4 NEW) -------------------------------------------
    # ``clear_propagate_system_jax_cache()`` takes NO arguments -- it
    # is a cache-clearing helper, not a propagator.  The walker
    # discovers it because the name contains the substring
    # ``_propagate``.  Documented exempt.
    ('lumenairy/system.py', 'clear_propagate_system_jax_cache'),

    # ---- raytrace/core.py (v4.15.4 NEW) ------------------------------------
    # ``apply_doe_phase_traced(rays, ...)`` takes a ``RayBundle``
    # (geometric-ray representation, NOT a 2-D scalar field).  The
    # v4.15.4 ``__all__``-membership pass discovers it because
    # ``apply_doe_phase_traced`` is in ``lumenairy.__all__``.  The
    # 2-D scalar guard would reject every legitimate caller --
    # ``RayBundle`` has no ``.ndim``.  Documented exempt.
    ('lumenairy/raytrace/core.py', 'apply_doe_phase_traced'),

    # ---- _context.py (v4.15.4 NEW) -----------------------------------------
    # ``apply_globals(state: dict) -> None`` applies a globals
    # snapshot to library-wide knobs (FFT threads, max RAM, cache
    # sizes).  Takes a dict of name -> value, NOT a 2-D scalar
    # field.  The v4.15.4 ``__all__``-membership pass discovers it
    # because ``apply_globals`` is in ``lumenairy.__all__``.
    # Documented exempt.
    ('lumenairy/_context.py', 'apply_globals'),
})


# ============================================================================
# Discovery walker -- top-level public defs only
# ============================================================================

def _name_matches_entry_point_filter(name: str) -> bool:
    """True iff ``name`` should be treated as a 2-D-scalar-field
    entry-point candidate by the walker.

    v4.15.4 (P2-NEW-V2-1) broadens the v4.15.3 filter to include
    ``richards_wolf_*`` and ``debye_wolf_*`` -- the
    ``propagators/vector_diffraction.py`` siblings that accept
    2-D pupils but didn't match the v4.15.3 regex.
    """
    if name.startswith('_'):
        return False
    if name.startswith('apply_'):
        return True
    if '_propagate' in name:
        return True
    if name == 'propagate':
        return True
    # v4.15.4 follow-up: ``propagate_through_system`` and its JAX
    # sibling ``propagate_through_system_jax`` don't contain the
    # underscore-prefix ``_propagate`` substring, so the v4.15.3
    # filter missed them.  Catch the ``propagate_<something>``
    # family explicitly.
    if name.startswith('propagate_'):
        return True
    if name.startswith('richards_wolf'):
        return True
    if name.startswith('debye_wolf'):
        return True
    return False


def _iter_target_py_files():
    """Yield every ``.py`` file in the walker's scope.

    Handles both directory-shaped targets (``lumenairy/propagators``)
    and file-shaped targets (``lumenairy/system.py``).  The latter is
    new in v4.15.4: ``system.py`` is a single file at the package
    root, not a subpackage.
    """
    for pkg_rel in _TARGET_PACKAGES:
        pkg_path = _REPO_ROOT / pkg_rel
        if pkg_path.is_file():
            # File-shaped target (e.g. ``lumenairy/system.py``).
            yield pkg_path
            continue
        if not pkg_path.is_dir():
            # Misconfigured path -- emit nothing for this entry rather
            # than raising; the count-floor test will catch a
            # walker-wide collapse.
            continue
        for py in sorted(pkg_path.rglob('*.py')):
            if py.name.startswith('_') and 'lens' not in py.name:
                # Skip private modules but keep _lens_*.py
                continue
            yield py


def _file_to_ast(py):
    """Parse ``py`` (a ``pathlib.Path``) and return the module AST.

    Uses cp1252-tolerant decoding to match the file-write contract.
    """
    src = py.read_text(encoding='utf-8', errors='replace')
    return ast.parse(src, filename=str(py))


def _public_top_level_function_defs(tree, py):
    """Yield ``(py, node)`` pairs for every public top-level
    ``FunctionDef`` in ``tree`` whose name passes the entry-point
    filter.  Class methods are intentionally excluded -- the guard
    contract is for the top-level public API; per-class wrapper
    methods delegate to the guarded scalar functions.
    """
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not _name_matches_entry_point_filter(node.name):
            continue
        yield py, node


def _walk_via_all_membership():
    """Yield (py, ast.FunctionDef) for every entry point reachable
    via ``lumenairy.__all__`` whose name passes the entry-point
    filter.

    v4.15.4 (P2-NEW-V2-1 fix): refactored from package-only walk to
    ``__all__``-based discovery so future module reorganisations
    that move a function between submodules don't silently drop it
    from the walker's scope.  Pairs with the fallback package walk
    below to also catch non-``__all__`` siblings.
    """
    try:
        la = importlib.import_module('lumenairy')
    except Exception:  # pragma: no cover -- catastrophic import failure
        return
    all_names = getattr(la, '__all__', None)
    if not all_names:
        return
    for name in all_names:
        if not _name_matches_entry_point_filter(name):
            continue
        obj = getattr(la, name, None)
        if obj is None or not callable(obj):
            continue
        try:
            src_file = inspect.getsourcefile(obj)
        except TypeError:
            continue
        if src_file is None:
            continue
        py = Path(src_file).resolve()
        # Only yield if the source file is inside the repo (so
        # vendored / installed copies don't leak in).
        try:
            py.relative_to(_REPO_ROOT)
        except ValueError:
            continue
        try:
            tree = _file_to_ast(py)
        except OSError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != name:
                continue
            yield py, node
            break


def _walk_via_package_scan():
    """Yield (py, ast.FunctionDef) for every entry point found by
    walking ``_TARGET_PACKAGES``.

    Kept as a fallback (after v4.15.4) so a function that is NOT in
    ``lumenairy.__all__`` but still lives in a target package and
    matches the name filter is still caught.  Example future use:
    a public-by-convention helper at top of module that wasn't yet
    promoted to the package-level ``__all__``.
    """
    for py in _iter_target_py_files():
        try:
            tree = _file_to_ast(py)
        except OSError:
            continue
        for pair in _public_top_level_function_defs(tree, py):
            yield pair


def _walk_entry_points():
    """Yield (path, ast.FunctionDef) for every public top-level def in
    scope, deduped across the ``__all__``-membership pass and the
    package-walk fallback.

    Class methods are intentionally excluded -- the guard contract
    is for the top-level public API; per-class wrapper methods
    delegate to the guarded scalar functions.
    """
    seen: set = set()
    for py, node in _walk_via_all_membership():
        key = (str(py.resolve()), node.name)
        if key in seen:
            continue
        seen.add(key)
        yield py, node
    for py, node in _walk_via_package_scan():
        key = (str(py.resolve()), node.name)
        if key in seen:
            continue
        seen.add(key)
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
    """Sanity floor: at least 25 entry points must be discoverable.

    The v4.15.3 closure migrated 10 v4.15.2-guarded sites + 9 new
    siblings (~19 total).  v4.15.4 broadens the walker scope to
    cover ``analysis/`` + ``system.py`` (file-at-root) plus the
    ``richards_wolf_*`` / ``debye_wolf_*`` name family in
    ``propagators/vector_diffraction.py`` -- adding 6 new sites
    (5 guarded + 1 exempt ``apply_perturbations``) plus the
    cache-clearer ``clear_propagate_system_jax_cache`` (exempt).
    So the post-v4.15.4 floor is ~25 (was 19).  Below this means
    the walker is broken.
    """
    count = sum(1 for _ in _walk_entry_points())
    assert count >= 25, (
        f"Walker found only {count} entry points; expected >= 25. "
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
# Counter-pin: a fake unguarded propagator must trip the meta-pin.
# v4.15.4 (P3-NEW-F2-3) -- pins that the walker's correctness relies
# on a POSITIVE signal (helper-call detection), not just on
# ``_walk_entry_points`` returning enough entries.  Modeled on the
# v4.15.0 ``_validate_grid_params`` pin's
# ``test_counter_pin_fake_unvalidated_factory_fails``.
# ============================================================================

def test_counter_pin_fake_unguarded_propagate_fails(monkeypatch):
    """Inject a synthetic unguarded propagator AST node into the
    walker output and assert the meta-pin assertion fires.

    Without this counter-pin a silent walker regression -- e.g. a
    name-regex bug that excludes ``apply_*`` siblings -- could
    pass even though it would have hidden every real
    unguarded-sibling failure.  This test verifies that the
    meta-pin's ``test_all_entry_points_call_helper_first`` only
    passes because the discovered functions actually call the
    helper, not because the walker silently elided them.
    """
    # ---- 1. Build a fake AST representing an unguarded propagator. -----
    fake_source = (
        '"""Fake module for counter-pin."""\n'
        'def fake_unguarded_propagate(E, wavelength, dx):\n'
        '    """Fake docstring."""\n'
        '    return E * 2.0\n'  # NO _check_2d_scalar_field call
    )
    fake_tree = ast.parse(fake_source, filename='<fake>')
    fake_node = None
    for stmt in fake_tree.body:
        if (isinstance(stmt, ast.FunctionDef)
                and stmt.name == 'fake_unguarded_propagate'):
            fake_node = stmt
            break
    assert fake_node is not None, (
        "Counter-pin setup error: failed to build fake FunctionDef "
        "AST node.")

    fake_path = _REPO_ROOT / 'lumenairy' / 'propagators' / 'fake_propagator.py'

    # ---- 2. Monkey-patch _walk_entry_points to include the fake. -------
    # Patch the module-level binding so when
    # ``test_all_entry_points_call_helper_first`` resolves the name
    # ``_walk_entry_points`` at call time it picks up the wrapper.
    import sys as _sys
    this_module = _sys.modules[__name__]
    real_walker = this_module._walk_entry_points

    def _walker_with_fake():
        # Yield everything the real walker yields...
        for py, node in real_walker():
            yield py, node
        # ...then append the synthetic unguarded entry.
        yield fake_path, fake_node

    monkeypatch.setattr(this_module, '_walk_entry_points',
                        _walker_with_fake)

    # ---- 3. Re-run the main pin and assert it now fails. ---------------
    with pytest.raises(AssertionError) as excinfo:
        test_all_entry_points_call_helper_first()
    msg = str(excinfo.value)
    assert 'fake_unguarded_propagate' in msg, (
        f"Counter-pin: meta-pin failed but its error message does "
        f"not name the injected unguarded function. Got:\n{msg}")
    assert 'fake_propagator.py' in msg, (
        f"Counter-pin: meta-pin failed but its error message does "
        f"not name the injected file. Got:\n{msg}")


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

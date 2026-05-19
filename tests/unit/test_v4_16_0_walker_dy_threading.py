"""v4.16.0 meta-pin: ``dy`` parameter threading walker.

Audit context
-------------

v4.13.0 (audit L6, ``test_audit_fixes_v4_13_0_jax_dtype_dy_siblings.py``)
introduced ``dy: Optional[float] = None`` as the canonical anamorphic
grid-spacing keyword across the ``apply_*`` family.  The historical
LumenAiry contract was a single ``dx`` argument with implicit square
pixels; threading ``dy`` lets callers chain anamorphic optics (e.g.
``apply_cylindrical_lens`` followed by an ASM propagation on the
resulting rectangular grid).

The v4.13.0 sweep landed ``dy`` on the bulk of the family but missed
a handful of siblings; each subsequent audit cycle (v4.14.0,
v4.14.1, v4.15.0) hand-patched one or two more.  This meta-pin
walks ``lumenairy.__all__`` for every name starting with ``apply_``
and asserts each function signature carries ``dy=None`` -- EXCEPT
the documented exemption set, which lists ``apply_*`` siblings whose
public contract intentionally does NOT take a ``dy`` (e.g.
operates on a non-pixel-grid object, ships its own grid metadata
on the input, or is geometrically square by construction).

Walker design
~~~~~~~~~~~~~

* AST-based: each candidate function is parsed via
  :func:`ast.parse` of its source file (cached via :func:`lru_cache`
  for repeat hits within the suite).  The AST inspection asks "does
  the function signature include a parameter named ``dy``?" -- it
  does NOT require any particular default value, position, or
  type annotation, because the historical landings have been
  inconsistent (some are positional-or-keyword, some are kw-only).

* Each function is also resolved via :func:`getattr` + :func:`inspect.signature`
  as a cross-check.  If the AST says ``dy`` is present but
  ``inspect`` disagrees, the function is flagged with a diagnostic
  (this would mean a decorator is hiding the parameter -- a real
  bug worth surfacing).

Exemption registry
~~~~~~~~~~~~~~~~~~

Each exemption MUST cite WHY ``dy`` is legitimately absent:

* ``apply_perturbations`` -- takes a prescription dict + sequence of
  ``Perturbation`` objects.  No pixel grid; returns a deep-copied
  prescription with perturbation fields populated.
* ``apply_mask`` -- pure element-wise field * mask multiplication.
  Does NOT consume any grid spacing (caller supplies a precomputed
  mask aligned with the field).
* ``apply_globals`` -- takes a globals-snapshot dict (output of
  :func:`snapshot_globals`).  Configuration-knob mutator, not a
  field-domain operation.
* ``apply_jones_matrix`` / ``apply_polarizer`` / ``apply_waveplate``
  / ``apply_half_wave_plate`` / ``apply_quarter_wave_plate`` /
  ``apply_rotator`` -- consume a :class:`JonesField` object whose
  ``dx`` / ``dy`` attributes carry the grid metadata.  Adding a
  module-level ``dy`` kwarg would silently shadow the field-level
  metadata.
* ``apply_thin_lens_to_beamlets`` / ``apply_abcd_to_beamlets`` --
  take a :class:`BeamletBundle`; the bundle's ``positions`` array
  is the grid, not a pixel raster.
* ``apply_aperture_diffraction`` -- takes a :class:`PathBundle`
  (HFPI).  Same as above: per-path position arrays, not a pixel
  raster.
* ``apply_vector_aperture_diffraction`` -- takes a
  :class:`VectorPathBundle` (vectorial HFPI).  Same as above.
* ``apply_doe_phase_traced`` -- takes a :class:`RayBundle`; the
  diffraction-order shift acts on per-ray direction cosines, not
  on a pixel raster.
* ``apply_detector`` -- takes ``dx_field`` (the field pixel pitch)
  AND ``pixel_pitch`` (the detector pixel pitch); both are scalar
  because the function's contract is a square-pixel detector with
  square field-domain sampling.  An anamorphic variant would need
  ``dx_field``, ``dy_field``, ``pixel_pitch_x``, ``pixel_pitch_y``;
  the v4.16.0 contract is square-only.  Re-evaluate at v4.17+ if
  anamorphic detector simulation lands.
* ``apply_dm`` -- module-level convenience that delegates to
  :meth:`DeformableMirror.apply`; the mirror geometry is described
  by the DM's own grid (``mirror.dx_actuator`` / ``mirror.n_actuators``)
  which is intrinsically square by the DM design.  An anamorphic DM
  would need a DM-level dx/dy split first.

Counter-pin
~~~~~~~~~~~

``test_counter_pin_fake_missing_dy_threading_fails`` injects a
synthetic ``apply_*`` candidate WITHOUT a ``dy`` parameter into the
discovery walker and asserts the main pin fires.  Modeled on the
v4.15.3 ``test_counter_pin_fake_unguarded_propagate_fails`` /
v4.15.4 P3-NEW-F2-3 counter-pin pattern: without this positive
signal, a silent walker regression (e.g. the ``apply_*`` name
filter degrading to ``apply``-prefix-only and missing
``apply_real_lens``) would let the meta-pin pass on zero
parametrized cases.

Author: Andrew Traverso -- v4.16.0 / Agent A
"""
from __future__ import annotations

import ast
import inspect
from functools import lru_cache
from pathlib import Path

import pytest

import lumenairy as la


_REPO_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# Documented exemption registry.  Each entry is the bare function name as it
# appears in ``lumenairy.__all__``.  See the module docstring for the cited
# rationale per entry.
# ============================================================================

_KNOWN_DY_THREADING_EXEMPTIONS = frozenset({
    # ---- non-field consumers ----------------------------------------------
    # ``apply_perturbations`` takes a prescription dict; not a field.
    'apply_perturbations',
    # ``apply_globals`` takes a globals-snapshot dict; not a field.
    'apply_globals',

    # ---- pure element-wise (no grid spacing) ------------------------------
    # ``apply_mask`` is a bare element-wise multiplication; mask is
    # pre-aligned with field, no dx/dy consumed.
    'apply_mask',

    # ---- JonesField consumers (grid metadata lives on the input) ----------
    # The polarization helpers consume a ``JonesField`` whose ``dx``/``dy``
    # attributes carry the anamorphic grid metadata.  Re-adding a top-level
    # ``dy`` kwarg would silently shadow the field-side metadata.
    'apply_jones_matrix',
    'apply_polarizer',
    'apply_waveplate',
    'apply_half_wave_plate',
    'apply_quarter_wave_plate',
    'apply_rotator',

    # ---- BeamletBundle / PathBundle / RayBundle consumers -----------------
    # Bundles carry per-element position arrays; there is no underlying
    # pixel raster, so dx/dy do not apply.
    'apply_thin_lens_to_beamlets',
    'apply_abcd_to_beamlets',
    'apply_aperture_diffraction',
    'apply_vector_aperture_diffraction',
    'apply_doe_phase_traced',

    # ---- square-only by contract ------------------------------------------
    # ``apply_detector`` ships a single ``dx_field`` and ``pixel_pitch``;
    # square-only by v4.16.0 contract.  Re-evaluate at v4.17+ if anamorphic
    # detector simulation is added.
    'apply_detector',
    # ``apply_dm`` delegates to ``DeformableMirror.apply``; the DM geometry
    # is square by the mirror's intrinsic actuator pitch.  An anamorphic
    # DM would need a DM-level dx/dy split first.
    'apply_dm',
})


# ============================================================================
# Discovery walker
# ============================================================================

@lru_cache(maxsize=None)
def _file_to_ast(path):
    """Parse ``path`` (a ``pathlib.Path``) and return the module AST.

    Cached so multiple ``apply_*`` lookups against the same module
    (e.g. ten siblings in ``elements/_lens_thin.py``) parse the file
    once.  Mirrors the v4.15.5 P3-NEW-F1-2 lru_cache contract.
    """
    src = path.read_text(encoding='utf-8', errors='replace')
    return ast.parse(src, filename=str(path))


def _discover_apply_entry_points():
    """Walk ``lumenairy.__all__`` for every ``apply_*`` name and yield
    ``(name, source_file_path, ast.FunctionDef)`` triples.

    The walker resolves each name via ``getattr(la, name)``, then
    ``inspect.getsourcefile`` to find the home file, then parses that
    file's AST and locates the ``FunctionDef`` matching the name.
    Names whose source file lives outside the repo (vendored copies)
    are skipped.
    """
    all_names = list(getattr(la, '__all__', []))
    for name in all_names:
        if not name.startswith('apply_'):
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
        try:
            py.relative_to(_REPO_ROOT)
        except ValueError:
            continue
        try:
            tree = _file_to_ast(py)
        except OSError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                yield name, py, node
                break


def _function_has_dy_param(node):
    """True iff ``node`` (an ``ast.FunctionDef``) declares a parameter
    named ``dy`` in any of: positional-or-keyword, keyword-only,
    positional-only, or ``**kwargs`` (the latter via name match on
    ``kwarg.arg``).

    Walks ``node.args.args``, ``node.args.kwonlyargs``,
    ``node.args.posonlyargs``, and the lone ``**kwargs`` slot.  Pure
    name match; default value and annotation are NOT inspected (the
    historical landings have been inconsistent and a fresh-from-trunk
    ``dy: float | None = None`` reads identically to the older
    ``dy=None`` form -- both satisfy the threading contract).
    """
    args = node.args
    for a in args.args:
        if a.arg == 'dy':
            return True
    for a in args.kwonlyargs:
        if a.arg == 'dy':
            return True
    for a in getattr(args, 'posonlyargs', []) or []:
        if a.arg == 'dy':
            return True
    # ``**kwargs`` would absorb a dy kwarg, but the static-analysis
    # cleanliness contract is that ``dy`` be NAMED in the signature
    # so it shows up in ``inspect.signature(...)``.  Don't accept
    # **kwargs-absorption.
    return False


# ============================================================================
# Pre-flight: count floor.  Catches the case where the walker silently
# collapses (e.g. ``apply_*`` filter regression).
# ============================================================================

def test_walker_minimum_apply_entry_points():
    """Sanity floor: the walker MUST discover at least 30 ``apply_*``
    candidates.

    v4.15.5 baseline: 36 ``apply_*`` entries in ``lumenairy.__all__``
    (counted at design time of this pin).  A floor of 30 leaves
    headroom for legitimate consolidation while catching a
    regression where the walker silently degrades to ``apply_``-
    prefix-only filtering and misses ``apply_real_lens``.
    """
    count = sum(1 for _ in _discover_apply_entry_points())
    assert count >= 30, (
        f"dy-threading walker found only {count} apply_* entry points; "
        f"expected >= 30 from lumenairy.__all__.  The discovery walker "
        f"may have regressed -- check the name filter and ``__all__`` "
        f"resolution.")


# ============================================================================
# Main pin: every discovered apply_* either declares dy or is exempt.
# ============================================================================

def test_all_apply_functions_thread_dy():
    """Every ``apply_*`` name in ``lumenairy.__all__`` must accept a
    ``dy`` parameter, OR appear in
    :data:`_KNOWN_DY_THREADING_EXEMPTIONS` with a cited rationale in
    the module docstring.
    """
    failures = []
    for name, py, node in _discover_apply_entry_points():
        if name in _KNOWN_DY_THREADING_EXEMPTIONS:
            continue
        if _function_has_dy_param(node):
            continue
        rel = py.relative_to(_REPO_ROOT).as_posix()
        failures.append(
            f"{rel}:{node.lineno} {name}: signature is missing the "
            f"``dy`` parameter.  Either thread ``dy=None`` through "
            f"the call (mirroring ``apply_thin_lens`` / "
            f"``apply_aperture`` / ``apply_mirror``), or, if this "
            f"function's API legitimately does not accept anamorphic "
            f"grid spacing, add the bare name to "
            f"``_KNOWN_DY_THREADING_EXEMPTIONS`` in this file with a "
            f"cited rationale in the module docstring.")
    if failures:
        raise AssertionError(
            "apply_* entries missing dy-threading:\n  - "
            + "\n  - ".join(failures))


# ============================================================================
# Cross-check: every AST-discovered ``dy`` should match
# inspect.signature.  A mismatch would mean a decorator is hiding the
# parameter from the runtime view -- worth surfacing.
# ============================================================================

def test_ast_dy_matches_inspect_signature():
    """For every non-exempt ``apply_*`` whose AST contains ``dy``, the
    runtime ``inspect.signature`` view MUST also contain ``dy``.

    Without this cross-check a future regression that wraps an
    ``apply_*`` in a decorator that strips kwargs would let the AST
    walker pass while the runtime contract has degraded.
    """
    mismatches = []
    for name, py, node in _discover_apply_entry_points():
        if name in _KNOWN_DY_THREADING_EXEMPTIONS:
            continue
        if not _function_has_dy_param(node):
            continue  # main pin will catch these
        obj = getattr(la, name, None)
        if obj is None:
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        if 'dy' in sig.parameters:
            continue
        rel = py.relative_to(_REPO_ROOT).as_posix()
        mismatches.append(
            f"{rel}:{node.lineno} {name}: AST signature lists ``dy`` "
            f"but ``inspect.signature(la.{name})`` does NOT -- a "
            f"decorator may be stripping kwargs.  Check the import "
            f"chain in lumenairy/__init__.py for wrappers / aliases.")
    if mismatches:
        raise AssertionError(
            "AST / inspect.signature mismatches on dy threading:\n  - "
            + "\n  - ".join(mismatches))


# ============================================================================
# Counter-pin: synthetic ``apply_*`` without ``dy`` must trip the main pin.
# Modeled on the v4.15.4/v4.15.5 P3-NEW-F2-3 counter-pin pattern.
# ============================================================================

def test_counter_pin_fake_missing_dy_threading_fails():
    """Inject a synthetic ``apply_fake_missing_dy(E, dx)`` AST node
    into the walker output and assert the main pin's assertion fires.

    Without this counter-pin a silent walker regression -- e.g. the
    AST inspector accepting ``**kwargs`` as a ``dy`` proxy -- could
    let the main pin pass even when real apply_* entries silently
    drop ``dy`` threading.  The counter-pin verifies the main pin
    only passes because the discovered functions actually declare
    ``dy``, not because the walker silently elided them.
    """
    # ---- 1. Build a fake AST representing an apply_* without dy. -------
    fake_source = (
        '"""Fake module for dy-threading counter-pin."""\n'
        'def apply_fake_missing_dy(E_in, dx, wavelength):\n'
        '    """Synthetic apply_* that intentionally drops dy."""\n'
        '    return E_in * 2.0\n'
    )
    fake_tree = ast.parse(fake_source, filename='<dy-fake>')
    fake_node = next(
        stmt for stmt in fake_tree.body
        if isinstance(stmt, ast.FunctionDef)
        and stmt.name == 'apply_fake_missing_dy'
    )
    assert not _function_has_dy_param(fake_node), (
        "Counter-pin setup error: synthetic ``apply_fake_missing_dy`` "
        "unexpectedly tests positive for a dy parameter.  Adjust the "
        "synthetic source so the test exercises the real failure mode.")

    fake_path = (
        _REPO_ROOT / 'lumenairy' / 'elements' / 'fake_missing_dy.py'
    )

    # ---- 2. Monkey-patch _discover_apply_entry_points to include the fake. ---
    import sys as _sys
    this_module = _sys.modules[__name__]
    real_walker = this_module._discover_apply_entry_points

    def _walker_with_fake():
        for name, py, node in real_walker():
            yield name, py, node
        yield 'apply_fake_missing_dy', fake_path, fake_node

    try:
        this_module._discover_apply_entry_points = _walker_with_fake
        with pytest.raises(AssertionError) as excinfo:
            test_all_apply_functions_thread_dy()
        msg = str(excinfo.value)
        assert 'apply_fake_missing_dy' in msg, (
            f"dy-threading counter-pin: main pin failed but its error "
            f"message does not name the injected unguarded function. "
            f"Got:\n{msg}")
        assert 'fake_missing_dy.py' in msg, (
            f"dy-threading counter-pin: main pin failed but its error "
            f"message does not name the injected file path. "
            f"Got:\n{msg}")
    finally:
        this_module._discover_apply_entry_points = real_walker


# ============================================================================
# Counter-pin: synthetic ``apply_*`` WITH ``dy`` must NOT trip the pin.
# Pins that the walker's positive signal (dy detection) actually works.
# ============================================================================

def test_counter_pin_fake_apply_with_dy_does_not_trip(monkeypatch):
    """Inject a synthetic ``apply_fake_with_dy(E, dx, ..., dy=None)``
    AST node and assert the main pin does NOT fire on it.

    Pairs with the missing-dy counter-pin above: this verifies the
    main pin's NEGATIVE half (when a function correctly threads ``dy``
    the walker accepts it without flagging).  Without this counter-
    pin a future regression that incorrectly REJECTED valid dy
    threading (e.g. a name-equality bug requiring an exact
    ``dy:Optional[float]=None`` literal) would silently break every
    legitimate caller while passing CI.
    """
    fake_source = (
        '"""Fake module for dy-threading positive-signal counter-pin."""\n'
        'def apply_fake_with_dy(E_in, dx, wavelength, dy=None):\n'
        '    """Synthetic apply_* that correctly threads dy."""\n'
        '    if dy is None: dy = dx\n'
        '    return E_in * 2.0\n'
    )
    fake_tree = ast.parse(fake_source, filename='<dy-fake-with>')
    fake_node = next(
        stmt for stmt in fake_tree.body
        if isinstance(stmt, ast.FunctionDef)
    )
    assert _function_has_dy_param(fake_node), (
        "Counter-pin setup error: synthetic apply_* with a literal "
        "``dy=None`` parameter does NOT register as having dy via "
        "_function_has_dy_param.  AST inspector regression.")

    fake_path = (
        _REPO_ROOT / 'lumenairy' / 'elements' / 'fake_with_dy.py'
    )

    import sys as _sys
    this_module = _sys.modules[__name__]
    real_walker = this_module._discover_apply_entry_points

    def _walker_with_fake():
        for name, py, node in real_walker():
            yield name, py, node
        yield 'apply_fake_with_dy', fake_path, fake_node

    monkeypatch.setattr(
        this_module, '_discover_apply_entry_points', _walker_with_fake)

    # Should NOT raise -- the fake correctly threads dy.
    # (If the rest of the codebase has a real failure the assertion
    # would still fire with a message naming THAT function.  Check that
    # the message does not blame ``apply_fake_with_dy``.)
    try:
        test_all_apply_functions_thread_dy()
    except AssertionError as exc:
        assert 'apply_fake_with_dy' not in str(exc), (
            f"Positive-signal counter-pin: ``apply_fake_with_dy`` "
            f"(which correctly threads dy) was flagged as missing it. "
            f"Got:\n{exc}")


# ============================================================================
# Diagnostic helper (printed when -v is on).
# ============================================================================

def test_discovered_apply_entries_for_diagnostics():
    """Print the discovered apply_* surface for triage."""
    discovered = []
    for name, py, node in _discover_apply_entry_points():
        rel = py.relative_to(_REPO_ROOT).as_posix()
        if name in _KNOWN_DY_THREADING_EXEMPTIONS:
            tag = 'exempt'
        elif _function_has_dy_param(node):
            tag = 'dy-ok'
        else:
            tag = 'MISSING-DY'
        discovered.append((rel, node.lineno, name, tag))
    print(f"\nv4.16.0 dy-threading walker discovered "
          f"{len(discovered)} apply_* entry points:")
    for rel, lineno, name, tag in discovered:
        print(f"  [{tag}] {rel}:{lineno}  {name}")
    assert isinstance(discovered, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

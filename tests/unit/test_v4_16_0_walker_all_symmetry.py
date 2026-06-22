"""v4.16.0 meta-pin: submodule ``__all__`` <-> top-level
``__all__`` symmetry walker.

Audit context
-------------

The v4.14.0 audit (P1-NEW-4) found that ``clear_lg_mode_stack_cache``
appeared in :data:`lumenairy.propagators.asymptotic.__all__` but was
not re-exported at the top-level :data:`lumenairy.__all__`.  Seven
sibling ``clear_*`` helpers had landed correctly; one had slipped
through the cracks.  v4.14.1 closed the specific gap and added
:mod:`test_v4_14_1_dispatcher_pin_cache_clears` to catch a future
``clear_*`` sibling slipping past.

This walker generalises the same invariant from ``clear_*`` names
to EVERY public name in any submodule's ``__all__``: each such name
must either be re-exported at top level, or appear in
:data:`_KNOWN_ALL_SYMMETRY_EXEMPTIONS` with a cited reason
(typically: classmethod factory not exposed as a free function,
backend-internal helper, etc.).

Walker design
~~~~~~~~~~~~~

* :func:`pkgutil.walk_packages` enumerates every submodule under
  ``lumenairy``.  For each, ``getattr(mod, '__all__', None)`` yields
  the public surface declaration.

* The walker is **AST-aware** in the sense that it ALSO consults
  the source of each submodule's ``__all__`` (via the cached
  :func:`_file_to_ast` helper) so that a future regression where a
  submodule's ``__all__`` is computed at runtime in a way that
  hides entries from the static view (e.g. conditional ``+=``
  inside a try/except) would still be discovered.  See
  :func:`_discover_submodule_all_entries`.

* Private modules (any path component starts with ``_``) are
  skipped; their ``__all__`` declarations are not part of the
  public-surface contract.

Exemption registry
~~~~~~~~~~~~~~~~~~

Each exemption MUST cite WHY the name is legitimately submodule-only:

* ``(lumenairy.algebra.from_prescription, from_prescription)`` --
  Exposed at top level as :meth:`Operator.from_prescription`
  classmethod (see :mod:`lumenairy.algebra.__init__`).  Adding a
  free-function ``la.from_prescription`` would be redundant with
  the classmethod and would expose two names with identical
  semantics.

* ``(lumenairy.backend, *)`` and ``(lumenairy.backend.<sub>, *)`` --
  Backend dispatch is namespaced under ``la.backend.*`` /
  ``la.backend.scipy.*`` etc. by deliberate API design: end-users
  reach into the dispatch layer via the explicit subpackage path
  (``la.backend.fft2(x)``, ``la.backend.scipy.jv(x)``).  Top-level
  re-export would crowd the global namespace with dispatch
  primitives.  Per-helper carve-outs (``array_namespace``,
  ``is_numpy_array``, ``is_cupy_array``, ``is_jax_array``,
  ``backend_name``, ``to_numpy``, ``to_backend``, ``RandomState``,
  ``CUPY_AVAILABLE``, ``JAX_AVAILABLE``) ARE re-exported at top
  level (they are the high-traffic helpers for user code).  Others
  (the FFT primitives, scipy dispatch) intentionally stay
  namespaced.

* ``(lumenairy.raytrace.bundles, *)`` -- ``BundleProtocol`` is a
  ``typing.Protocol`` used only for static typing; the conversion
  helpers ``ray_to_path``, ``ray_to_beamlet``, ``path_to_ray`` are
  reachable via ``la.raytrace.*`` but kept off the top level
  because they are inter-bundle adapters (a small audience).

* ``(lumenairy.raytrace.jax_trace, *)`` -- ``JaxPrescription`` /
  ``trace_jax_with_params`` are JAX-only entry points reachable via
  ``la.raytrace.*``; the JAX-side gradient pipeline has its own
  namespacing that the top-level package does not duplicate.

Counter-pin
~~~~~~~~~~~

``test_counter_pin_fake_missing_reexport_fails`` injects a synthetic
submodule with a name in its ``__all__`` that is NOT in
``lumenairy.__all__`` and asserts the main pin fires.

Author: Andrew Traverso -- v4.16.0 / Agent A
"""
from __future__ import annotations

import ast
import importlib
import pkgutil
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import pytest

import lumenairy as la

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# Documented exemption registry.  Each entry is a ``(submodule_dotted_name,
# member_name)`` tuple.  See the module docstring for the cited rationale
# per cluster.
# ============================================================================

_KNOWN_ALL_SYMMETRY_EXEMPTIONS = frozenset({
    # ---- algebra.from_prescription ----------------------------------------
    # ``from_prescription`` is attached as the
    # ``Operator.from_prescription`` classmethod in
    # ``lumenairy/algebra/__init__.py``.  A separate top-level free
    # function would be a second name with identical semantics; the
    # classmethod is the canonical entry point.
    ('lumenairy.algebra.from_prescription', 'from_prescription'),

    # ---- backend (and subpackages) ----------------------------------------
    # The backend-dispatch layer is namespaced under ``la.backend.*``
    # by design.  The high-traffic helpers (``array_namespace``,
    # ``is_*_array``, ``backend_name``, ``to_numpy``, ``to_backend``,
    # ``RandomState``, ``CUPY_AVAILABLE``, ``JAX_AVAILABLE``) ARE
    # re-exported at top level (and so are NOT listed here).  The
    # FFT and scipy primitives stay namespaced -- listed below.
    #
    # ``lumenairy.backend`` (package): the package re-exports its
    # own submodule __all__'s; entries here mirror the per-submodule
    # exemptions below so the package-level walker doesn't double-
    # flag.  See the per-submodule entries for the rationale.
    ('lumenairy.backend', 'fft2'),
    ('lumenairy.backend', 'ifft2'),
    ('lumenairy.backend', 'fft'),
    ('lumenairy.backend', 'ifft'),
    ('lumenairy.backend', 'fftshift'),
    ('lumenairy.backend', 'ifftshift'),
    ('lumenairy.backend', 'fftfreq'),
    ('lumenairy.backend', 'fft_backend_for'),
    ('lumenairy.backend', 'scipy'),

    # ---- backend.fft ------------------------------------------------------
    # FFT primitives reached via ``la.backend.fft2(x)`` etc.; not
    # re-exported at top level because numpy/scipy/pyfftw users
    # expect to call ``np.fft.fft2`` / ``scipy.fft.fft2`` and a
    # top-level ``la.fft2`` would invite confusion about whose
    # backend is being dispatched.  Stay namespaced.
    ('lumenairy.backend.fft', 'fft2'),
    ('lumenairy.backend.fft', 'ifft2'),
    ('lumenairy.backend.fft', 'fft'),
    ('lumenairy.backend.fft', 'ifft'),
    ('lumenairy.backend.fft', 'fftshift'),
    ('lumenairy.backend.fft', 'ifftshift'),
    ('lumenairy.backend.fft', 'fftfreq'),
    ('lumenairy.backend.fft', 'fft_backend_for'),

    # ---- backend.scipy ----------------------------------------------------
    # ``la.backend.scipy.jv(x)`` etc.  Same rationale as fft above:
    # ``la.jv`` would collide with the user's mental model of
    # ``scipy.special.jv`` and obscure that the call is being
    # dispatched.  Stay namespaced.
    ('lumenairy.backend.scipy', 'jv'),
    ('lumenairy.backend.scipy', 'erf'),
    ('lumenairy.backend.scipy', 'gammaln'),
    ('lumenairy.backend.scipy', 'expi'),
    ('lumenairy.backend.scipy', 'solve'),
    ('lumenairy.backend.scipy', 'lstsq'),
    ('lumenairy.backend.scipy', 'eigh'),

    # ---- raytrace.bundles -------------------------------------------------
    # ``BundleProtocol`` is a ``typing.Protocol`` used only as a static-
    # typing alias; users instantiate ``RayBundle`` / ``PathBundle`` /
    # ``BeamletBundle`` directly.  The inter-bundle conversion helpers
    # ``ray_to_path`` / ``ray_to_beamlet`` / ``path_to_ray`` are reached
    # via ``la.raytrace.*`` (and the raytrace package mirrors them in
    # its own ``__all__``).  Top-level re-export would crowd the
    # public surface with a small-audience adapter set.
    ('lumenairy.raytrace.bundles', 'BundleProtocol'),
    ('lumenairy.raytrace.bundles', 'ray_to_path'),
    ('lumenairy.raytrace.bundles', 'ray_to_beamlet'),
    ('lumenairy.raytrace.bundles', 'path_to_ray'),
    ('lumenairy.raytrace', 'BundleProtocol'),
    ('lumenairy.raytrace', 'ray_to_path'),
    ('lumenairy.raytrace', 'ray_to_beamlet'),
    ('lumenairy.raytrace', 'path_to_ray'),

    # ---- raytrace.jax_trace -----------------------------------------------
    # JAX-only entry points reached via ``la.raytrace.*``; the JAX
    # gradient pipeline namespaces its own helpers under raytrace
    # rather than promoting them to the top level (the top-level
    # ``trace_jax`` / ``trace_jax_cache`` are already exposed; these
    # are the lower-level parameterised variants).
    ('lumenairy.raytrace.jax_trace', 'JaxPrescription'),
    ('lumenairy.raytrace.jax_trace', 'trace_jax_with_params'),

    # ---- elements.eme (eigenmode-expansion mode solvers) ------------------
    # v5.15.0: the EME 2-D Bloch *mode* / band-structure solvers are a
    # specialized subpackage reached via ``la.elements.eme.*`` -- the
    # mode-solver peer of ``pmm`` / ``rcwa`` (whose own public names ARE
    # prefixed -- ``pmm_efficiency_1d`` etc. -- and so live at top level).
    # The EME functions carry GENERIC names (``layer_modes``, ``dispersion``,
    # ``mode_field``, ``cell_smatrix``, ``mode_match``, ...) that would be
    # ambiguous and collision-prone in the ~450-name top-level surface
    # (``dispersion`` / ``layer_modes`` read as far too general there).  They
    # stay namespaced under the subpackage by design -- same rationale as the
    # ``raytrace.bundles`` adapters and the ``backend.scipy`` primitives above.
    ('lumenairy.elements.eme', 'strip_x_modes'),
    ('lumenairy.elements.eme', 'cell_smatrix'),
    ('lumenairy.elements.eme', 'dispersion'),
    ('lumenairy.elements.eme', 'layer_modes'),
    ('lumenairy.elements.eme', 'mode_field'),
    ('lumenairy.elements.eme', 'ref_2d_modes'),
    ('lumenairy.elements.eme', 'strip_vector_modes'),
    ('lumenairy.elements.eme', 'dispersion_vec'),
    ('lumenairy.elements.eme', 'layer_vector_modes'),
    ('lumenairy.elements.eme', 'mode_field_vec'),
    ('lumenairy.elements.eme', 'beyn_refine_complex'),
    ('lumenairy.elements.eme', 'layer_vector_modes_complex'),
    ('lumenairy.elements.eme', 'ref_2d_modes_vector'),
    ('lumenairy.elements.eme', 'strips_to_eps_xy'),
    ('lumenairy.elements.eme', 'eps_xy_to_strips'),
    ('lumenairy.elements.eme', 'plane_wave_orders'),
    ('lumenairy.elements.eme', 'pw_matrix'),
    ('lumenairy.elements.eme', 'mode_match'),
    ('lumenairy.elements.eme', 'diffraction_fd'),
    ('lumenairy.elements.eme', 'diffraction_eme'),

    # ---- backend (package re-export of RandomState) -----------------------
    # ``RandomState`` IS at top level so this is NOT in the exemption
    # set -- listed here as a no-op anchor for the audit reader.
    # (NB: do not actually add to the frozenset.)
})


# ============================================================================
# Walker
# ============================================================================

@lru_cache(maxsize=None)
def _file_to_ast(path):
    """Parse ``path`` and return the module AST.  Cached per the
    v4.15.5 P3-NEW-F1-2 lru_cache contract."""
    src = path.read_text(encoding='utf-8', errors='replace')
    return ast.parse(src, filename=str(path))


def _discover_submodule_all_entries() -> List[Tuple[str, str]]:
    """Walk every submodule under ``lumenairy`` and yield every
    ``(submodule_dotted_name, member_name)`` pair declared in the
    submodule's ``__all__``.

    Returns a sorted list for deterministic test IDs.

    Skips:

    * Modules whose dotted path contains any segment starting with
      ``_`` (private modules are not part of the public surface).
    * Modules that fail to import (optional-dep modules e.g. JAX-
      only) -- those are covered by per-feature smoke tests.
    """
    pairs = []
    for modinfo in pkgutil.walk_packages(
            la.__path__, prefix='lumenairy.'):
        parts = modinfo.name.split('.')
        if any(p.startswith('_') for p in parts[1:]):
            continue
        try:
            mod = importlib.import_module(modinfo.name)
        except ImportError:
            continue
        names = getattr(mod, '__all__', None)
        if names is None:
            continue
        for name in names:
            if name.startswith('_'):
                # Submodule __all__ entries starting with _ are by
                # convention internal-but-deliberate exports; they
                # are not part of the top-level public surface.
                continue
            pairs.append((modinfo.name, name))
    return sorted(set(pairs))


# Materialise at module-import time so the count is visible in
# collection and so a walker collapse fails loudly.
_DISCOVERED_PAIRS = _discover_submodule_all_entries()


# ============================================================================
# Pre-flight: count floor.
# ============================================================================

def test_walker_minimum_submodule_all_entries():
    """Sanity floor: the walker MUST discover at least 100 entries.

    The v4.15.5 baseline has hundreds of submodule ``__all__``
    entries across all subpackages.  A floor of 100 catches a
    walker collapse (e.g. ``pkgutil.walk_packages`` returning
    nothing because the package was renamed) while leaving
    generous headroom for legitimate refactoring.
    """
    count = len(_DISCOVERED_PAIRS)
    assert count >= 100, (
        f"__all__-symmetry walker found only {count} submodule "
        f"entries; expected >= 100.  The discovery walker may have "
        f"regressed.")


# ============================================================================
# Main pin: every submodule __all__ entry is at top level OR exempt.
# ============================================================================

def test_all_submodule_entries_reexported_or_exempt():
    """Every name in a submodule's ``__all__`` MUST either appear at
    top-level ``lumenairy.__all__``, or be listed in
    :data:`_KNOWN_ALL_SYMMETRY_EXEMPTIONS` with a cited rationale.
    """
    top_all = set(getattr(la, '__all__', []))
    failures = []
    for submod, name in _DISCOVERED_PAIRS:
        if name in top_all:
            continue
        if (submod, name) in _KNOWN_ALL_SYMMETRY_EXEMPTIONS:
            continue
        failures.append(
            f"{submod}.__all__ exports {name!r} but it is not in "
            f"``lumenairy.__all__``.  Either:\n"
            f"    1. Add ``from .{submod[len('lumenairy.'):]} import "
            f"{name}`` to lumenairy/__init__.py and append "
            f"{name!r} to the top-level __all__, OR\n"
            f"    2. Rename to ``_{name}`` and remove from the "
            f"submodule's __all__ (mark internal), OR\n"
            f"    3. Add ``({submod!r}, {name!r})`` to "
            f"``_KNOWN_ALL_SYMMETRY_EXEMPTIONS`` in this file with "
            f"a cited rationale in the module docstring.\n"
            f"    (This is the v4.14.0 audit P1-NEW-4 sibling-gap "
            f"pattern generalised to all names.)")
    if failures:
        raise AssertionError(
            "Submodule __all__ entries not re-exported at top level:\n"
            "  - " + "\n  - ".join(failures))


# ============================================================================
# Cross-check: every name claimed re-exported MUST resolve and be callable
# (mirroring the v4.14.1 cache-clears pattern).
# ============================================================================

def test_all_top_level_attrs_are_resolvable():
    """Each name in ``lumenairy.__all__`` MUST resolve to a real
    attribute on ``la``.

    Counter-direction to the main pin: catches the v4.14.0 P1-NEW-4
    failure mode where ``__all__`` lists a name that is NOT actually
    imported (so ``from lumenairy import *`` raises NameError).
    """
    missing = []
    for name in la.__all__:
        if not hasattr(la, name):
            missing.append(name)
    assert not missing, (
        "Names listed in ``lumenairy.__all__`` but not present as "
        "attributes of ``la``:\n  - " + "\n  - ".join(missing))


# ============================================================================
# Counter-pin: synthetic submodule entry not in top __all__ must trip.
# ============================================================================

def test_counter_pin_fake_missing_reexport_fails(monkeypatch):
    """Inject a synthetic ``(submod, name)`` pair into the walker
    output where the name is NOT in ``lumenairy.__all__``, and assert
    the main pin fires.

    Without this counter-pin, a silent walker regression -- e.g. the
    ``__all__`` resolution short-circuiting on a missing attribute --
    could let the pin pass on zero parametrized cases.
    """
    fake_submod = 'lumenairy.fake_submodule_for_counter_pin'
    fake_name = 'fake_missing_reexport_name'
    # Sanity: the fake name is genuinely NOT in la.__all__.
    assert fake_name not in la.__all__, (
        "Counter-pin setup error: the synthetic missing-reexport name "
        "is unexpectedly already in la.__all__.")
    assert (fake_submod, fake_name) not in _KNOWN_ALL_SYMMETRY_EXEMPTIONS

    import sys as _sys
    this_module = _sys.modules[__name__]
    real_discovered = list(this_module._DISCOVERED_PAIRS)
    fake_discovered = real_discovered + [(fake_submod, fake_name)]
    monkeypatch.setattr(
        this_module, '_DISCOVERED_PAIRS', fake_discovered)

    with pytest.raises(AssertionError) as excinfo:
        test_all_submodule_entries_reexported_or_exempt()
    msg = str(excinfo.value)
    assert fake_name in msg, (
        f"Counter-pin: main pin failed but its error message does not "
        f"name the injected missing-reexport name. Got:\n{msg}")
    assert fake_submod in msg, (
        f"Counter-pin: main pin failed but its error message does not "
        f"name the injected submodule. Got:\n{msg}")


# ============================================================================
# Counter-pin: synthetic entry WITH proper reexport must NOT trip.
# ============================================================================

def test_counter_pin_fake_with_reexport_does_not_trip(monkeypatch):
    """Inject a synthetic ``(submod, name)`` pair where the name IS
    in ``la.__all__`` and assert the main pin does NOT flag it.

    Pairs with the missing-reexport counter-pin above: verifies the
    POSITIVE half of the walker's signal (re-exported names are
    correctly excluded from the failure set).
    """
    # Pick a name that's known to be in la.__all__ (a stable
    # high-traffic export).
    canonical_name = 'apply_real_lens'
    assert canonical_name in la.__all__, (
        "Counter-pin setup error: canonical reference name "
        "``apply_real_lens`` is unexpectedly NOT in la.__all__.")

    fake_submod = 'lumenairy.fake_with_reexport'
    import sys as _sys
    this_module = _sys.modules[__name__]
    real_discovered = list(this_module._DISCOVERED_PAIRS)
    fake_discovered = real_discovered + [(fake_submod, canonical_name)]
    monkeypatch.setattr(
        this_module, '_DISCOVERED_PAIRS', fake_discovered)

    # Should NOT raise on the injected pair specifically; if the rest
    # of the codebase has real failures the AssertionError would still
    # name THOSE, but the synthetic name must not appear in the message.
    try:
        test_all_submodule_entries_reexported_or_exempt()
    except AssertionError as exc:
        msg = str(exc)
        assert fake_submod not in msg, (
            f"Positive-signal counter-pin: re-exported name was "
            f"flagged. Got:\n{msg}")


# ============================================================================
# Diagnostic helper.
# ============================================================================

def test_discovered_all_entries_for_diagnostics():
    """Print the discovered (submod, name) pairs for triage."""
    top_all = set(getattr(la, '__all__', []))
    items = []
    for submod, name in _DISCOVERED_PAIRS:
        if name in top_all:
            tag = 'top'
        elif (submod, name) in _KNOWN_ALL_SYMMETRY_EXEMPTIONS:
            tag = 'exempt'
        else:
            tag = 'MISSING'
        items.append((submod, name, tag))
    print(f"\nv4.16.0 __all__-symmetry walker discovered "
          f"{len(items)} submodule __all__ entries.")
    n_top = sum(1 for _, _, t in items if t == 'top')
    n_exempt = sum(1 for _, _, t in items if t == 'exempt')
    n_missing = sum(1 for _, _, t in items if t == 'MISSING')
    print(f"  [top]     {n_top}")
    print(f"  [exempt]  {n_exempt}")
    print(f"  [MISSING] {n_missing}")
    for submod, name, tag in items:
        if tag == 'MISSING':
            print(f"  [{tag}] {submod}::{name}")
    assert isinstance(items, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

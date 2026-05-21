"""v4.15.4 / Agent B -- regression tests for the three v4.15.3 hygiene
closures.

Closes:

* **P2-NEW-F1-D** -- ``lumenairy/_validation.py`` shipped a 102-LOC
  helper with a lazy import inside the function body, claiming a
  circular dependency with ``lumenairy.sources.core``.  The audit
  grep-verified that ``sources.core`` does NOT import from
  ``_validation`` -- the circular-dep claim was hypothetical, not
  real.  The lazy import wasted ~1 us per propagator call (1-10 ms
  per merit eval in optimisation loops).  v4.15.4 hoists the
  ``PartialCoherenceMCF as _MCF`` import to module scope and updates
  the inaccurate comment.

* **P2-NEW-F1-B option a** -- ``lumenairy/optimize/core.py`` shipped
  ``_PerturbedABCDFallbackSentinel`` + its singleton in v4.15.2 but
  the class was never wired at the intended callsite (the
  ``ToleranceAwareMerit.evaluate`` perturbed-ABCD failure branch
  writes a 2-tuple ``(efl_p, bfl_p)`` fallback, which is not single-
  sentinel-friendly).  v4.15.3 marked the class as dead code via a
  ``_v4_15_3_dead_code = True`` class attribute, but the marker was
  data-only and not honoured by any static analyser.  v4.15.4
  DELETES the class and its singleton outright.

* **P3-NEW-F2-4** -- 57 v4.15.x tests failed under
  ``pytest -W error::DeprecationWarning`` because they triggered the
  documented v4.15.1 Schell-family ``return_kind`` default-path
  ``DeprecationWarning`` without ``pytest.warns`` shielding.
  v4.15.4 adds per-test-file ``pytestmark`` filterwarnings markers
  so the suite stays clean under strict ``-W error::DeprecationWarning``
  while leaving every other module's promotion-to-error untouched.

Author: Andrew Traverso -- v4.15.4 / Agent B
"""
from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# B.1 -- _validation.py lazy-import hoist (P2-NEW-F1-D)
# ============================================================================


def test_validation_helper_no_circular_import_on_module_load():
    """``python -c "import lumenairy"`` must succeed after the
    v4.15.4 hoist of ``PartialCoherenceMCF as _MCF`` to module scope.

    The v4.15.3 helper used a function-body lazy import that cited a
    (hypothetical) circular dep with ``sources.core``.  The hoist
    inverts that claim -- if a real circular dep had existed the
    module load itself would now raise ``ImportError``.  This pin
    runs the import in a fresh subprocess so we measure cold-load
    behaviour (the parent test process has already imported the
    package).
    """
    proc = subprocess.run(
        [sys.executable, '-c', 'import lumenairy'],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"Cold ``python -c 'import lumenairy'`` returned non-zero "
        f"after the v4.15.4 _validation.py hoist; expected a clean "
        f"load.\nstdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
    )
    # Defensive: no ImportError / circular-import message in stderr.
    assert 'ImportError' not in proc.stderr, (
        f"Cold import surfaced ImportError after the v4.15.4 hoist; "
        f"the (hypothetical) circular dep is now real.  Switch to a "
        f"TYPE_CHECKING import + duck-typed predicate per the comment "
        f"block in _validation.py.\nstderr: {proc.stderr}"
    )


def test_validation_helper_lazy_import_removed():
    """v4.15.4 (P2-NEW-F1-D): ``PartialCoherenceMCF`` is imported at
    MODULE scope in ``lumenairy/_validation.py``, not inside the
    ``_check_2d_scalar_field`` function body.

    AST-walk the source so the assertion survives line-by-line
    reformatting; we verify (a) at least one module-level
    ``from lumenairy.sources.core import ...`` exists and includes
    ``PartialCoherenceMCF``, and (b) NO ``from lumenairy.sources.core
    import ...`` lives inside the body of ``_check_2d_scalar_field``.
    """
    src_path = _REPO_ROOT / 'lumenairy' / '_validation.py'
    src = src_path.read_text(encoding='utf-8')
    tree = ast.parse(src)

    module_level_from_sources = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and \
                node.module == 'lumenairy.sources.core':
            module_level_from_sources.append(node)
    assert module_level_from_sources, (
        "Expected a module-level ``from lumenairy.sources.core "
        "import PartialCoherenceMCF as _MCF`` in _validation.py after "
        "the v4.15.4 hoist; found none."
    )
    names_imported = []
    for node in module_level_from_sources:
        for alias in node.names:
            names_imported.append(alias.name)
    assert 'PartialCoherenceMCF' in names_imported, (
        f"Expected ``PartialCoherenceMCF`` in the module-level "
        f"``from lumenairy.sources.core import ...`` of _validation.py "
        f"after the v4.15.4 hoist; got names={names_imported!r}."
    )

    # Scan inside _check_2d_scalar_field for any nested ImportFrom
    # node whose module is lumenairy.sources.core -- the v4.15.3 lazy
    # import lived exactly here.  Post-v4.15.4 there must be none.
    lazy_imports = []
    for func in ast.walk(tree):
        if isinstance(func, ast.FunctionDef) and \
                func.name == '_check_2d_scalar_field':
            for inner in ast.walk(func):
                if isinstance(inner, ast.ImportFrom) and \
                        inner.module == 'lumenairy.sources.core':
                    lazy_imports.append(inner.lineno)
    assert not lazy_imports, (
        f"Found lazy ``from lumenairy.sources.core import ...`` "
        f"statements inside ``_check_2d_scalar_field`` at line(s) "
        f"{lazy_imports!r} after the v4.15.4 hoist; the import must "
        f"live at module scope so it costs zero per propagator call."
    )


def test_validation_helper_mcf_alias_is_at_module_scope_runtime():
    """Runtime cross-check of the AST pin: ``_MCF`` is importable from
    ``lumenairy._validation`` at module scope and equals
    ``lumenairy.sources.core.PartialCoherenceMCF``."""
    from lumenairy._validation import _MCF as validation_mcf
    from lumenairy.sources.core import PartialCoherenceMCF
    assert validation_mcf is PartialCoherenceMCF, (
        "lumenairy._validation._MCF must alias "
        "lumenairy.sources.core.PartialCoherenceMCF -- the v4.15.4 "
        "hoist preserved the alias.  Got "
        f"{validation_mcf!r} (id={id(validation_mcf)}) vs "
        f"PartialCoherenceMCF={PartialCoherenceMCF!r} "
        f"(id={id(PartialCoherenceMCF)})."
    )


def test_validation_helper_doc_comment_does_not_claim_circular_dep():
    """The v4.15.4 comment must NOT repeat the inaccurate
    "circular dependency" claim from the v4.15.3 lazy-import comment.

    The pin is a string match (the comment is hand-edited per the
    audit's option-a recommendation; if a future refactor reintroduces
    a circular dep the comment block instructs maintainers to switch
    to a ``TYPE_CHECKING`` import + duck-typed predicate instead of
    re-introducing the lazy import).
    """
    src_path = _REPO_ROOT / 'lumenairy' / '_validation.py'
    src = src_path.read_text(encoding='utf-8')
    # Find the import statement region (first 60 lines).
    head = '\n'.join(src.splitlines()[:60])
    forbidden_phrases = (
        'Lazy import avoids a circular dependency',
    )
    for phrase in forbidden_phrases:
        assert phrase not in head, (
            f"_validation.py module-head still contains the v4.15.3 "
            f"inaccurate phrase {phrase!r}; the v4.15.4 audit "
            f"verified ``sources.core`` does NOT import from "
            f"``_validation`` and the lazy import was therefore "
            f"unjustified.  Refresh the comment block."
        )


# ============================================================================
# B.2 -- _PerturbedABCDFallbackSentinel deletion (P2-NEW-F1-B option a)
# ============================================================================


def test_perturbed_abcd_fallback_sentinel_class_deleted():
    """v4.15.4 (P2-NEW-F1-B option a): ``_PerturbedABCDFallbackSentinel``
    class is DELETED from ``lumenairy.optimize.core``.

    AST-walk the module source for any ``class
    _PerturbedABCDFallbackSentinel`` definition.  The class was
    documented as dead code in v4.15.3 via a ``_v4_15_3_dead_code =
    True`` class attribute, but the marker was data-only.  v4.15.4
    deletes it outright.
    """
    src_path = _REPO_ROOT / 'lumenairy' / 'optimize' / 'core.py'
    src = src_path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and \
                node.name == '_PerturbedABCDFallbackSentinel':
            offenders.append(node.lineno)
    assert not offenders, (
        f"Found ``class _PerturbedABCDFallbackSentinel`` definition(s) "
        f"in lumenairy/optimize/core.py at line(s) {offenders!r}; "
        f"v4.15.4 (AUDIT_V4_15_3 P2-NEW-F1-B option a) deleted this "
        f"class as dead code (never wired; the perturbed-ABCD fallback "
        f"is a 2-tuple, which is not single-sentinel-friendly)."
    )


def test_perturbed_abcd_fallback_sentinel_singleton_deleted():
    """v4.15.4 (P2-NEW-F1-B option a): the
    ``_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ`` singleton is DELETED
    from ``lumenairy.optimize.core``.

    Walk the module's module-level ``Assign`` targets for a binding
    to ``_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ``.  Post-v4.15.4 there
    must be none.
    """
    src_path = _REPO_ROOT / 'lumenairy' / 'optimize' / 'core.py'
    src = src_path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    offenders = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and \
                        tgt.id == '_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ':
                    offenders.append(node.lineno)
    assert not offenders, (
        f"Found module-level binding(s) of "
        f"``_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ = ...`` in "
        f"lumenairy/optimize/core.py at line(s) {offenders!r}; "
        f"v4.15.4 deleted the singleton as dead code."
    )


def test_perturbed_abcd_fallback_sentinel_names_not_importable():
    """Runtime cross-check: neither the class nor the singleton is
    accessible from ``lumenairy.optimize.core`` after the v4.15.4
    deletion.

    A failure here would mean the deletion landed in source but a
    stale ``.pyc`` cache was loaded; pytest should fail the import
    before this test even runs.  Kept as a cheap belt-and-braces pin.
    """
    from lumenairy.optimize import core as oc
    for name in (
        '_PerturbedABCDFallbackSentinel',
        '_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ',
    ):
        assert not hasattr(oc, name), (
            f"lumenairy.optimize.core still exposes {name!r} after "
            f"the v4.15.4 deletion; check for stale .pyc caches or a "
            f"partial revert of the deletion edit."
        )


def test_perturbed_abcd_fallback_sentinel_not_in_registry():
    """v4.15.4: with the class deleted, ``_SENTINEL_REGISTRY`` must
    NOT contain a ``'_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ'`` key.

    The registry is populated lazily on first ``__init__`` of each
    ``_Sentinel`` subclass; with the class deleted, no
    instantiation ever happens, so the registry entry never gets
    created.  Sanity pin: confirm the registry is clean of the
    deleted name.
    """
    from lumenairy._deprecation import _SENTINEL_REGISTRY
    assert '_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ' not in \
        _SENTINEL_REGISTRY, (
        "_SENTINEL_REGISTRY still carries a "
        "'_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ' entry after the "
        "v4.15.4 class deletion; check that no other module "
        "re-registers the singleton at import time."
    )


# ============================================================================
# B.3 -- DeprecationWarning test hygiene (P3-NEW-F2-4)
# ============================================================================


def test_no_deprecation_warning_errors_in_v4_15_1_agent_b_module():
    """A representative sample of v4.15.1 Agent-B tests must pass
    under ``-W error::DeprecationWarning`` after v4.15.4 added the
    per-module ``pytestmark = pytest.mark.filterwarnings(
    'default::DeprecationWarning')`` shield.

    The shield permits the documented Schell-family ``return_kind``
    default-path ``DeprecationWarning`` to fire without promoting it
    to an exception, while leaving every other module's strict
    promotion intact.
    """
    proc = subprocess.run(
        [
            sys.executable, '-m', 'pytest',
            'tests/unit/test_v4_15_1_agent_b.py',
            '-W', 'error::DeprecationWarning',
            '--tb=no', '-q',
        ],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
        timeout=120,
    )
    # pytest exit code 0 = all passed (1 = test failures, 2 = collect
    # error, 5 = no tests collected).  We accept 0 only.
    assert proc.returncode == 0, (
        f"tests/unit/test_v4_15_1_agent_b.py failed under "
        f"``-W error::DeprecationWarning`` after the v4.15.4 hygiene "
        f"closure.  Exit code: {proc.returncode}.\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )


def test_no_deprecation_warning_errors_in_v4_15_3_agent_a_module():
    """A second representative module: v4.15.3 Agent-A tests must
    pass under ``-W error::DeprecationWarning``.  The Agent-A file
    uses ``create_gaussian_schell_source`` in helper fixtures to
    build the 3-D ensemble for negative-path tests; the legacy
    positional call triggers the same Schell ``DeprecationWarning``.
    """
    proc = subprocess.run(
        [
            sys.executable, '-m', 'pytest',
            'tests/unit/test_v4_15_3_agent_a.py',
            '-W', 'error::DeprecationWarning',
            '--tb=no', '-q',
        ],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"tests/unit/test_v4_15_3_agent_a.py failed under "
        f"``-W error::DeprecationWarning`` after the v4.15.4 hygiene "
        f"closure.  Exit code: {proc.returncode}.\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )


def test_representative_v4_15_x_modules_pass_under_deprecation_error():
    """Sample-based pin: a representative slice of the v4.15.x test
    files that previously failed under ``-W error::DeprecationWarning``
    must all pass after the v4.15.4 hygiene closure.

    The baseline failure count was 57 across these modules (per the
    v4.15.4 dispatch brief).  Rather than re-running the whole
    ~1822-test suite under ``-W error`` for each invocation of this
    pin (~3-5 minutes), we sample 4 high-failure-count modules and
    require all 4 to pass cleanly.  The full-suite verification is
    documented as a manual release-gate step in the v4.15.4
    Agent-B release notes (see ``docs/release_notes/
    .release_notes_v4_15_4_agent_b.md``).
    """
    representative_modules = (
        # v4.15.1 Agent-B: 14 Schell-warning failures (largest single)
        'tests/unit/test_v4_15_1_agent_b.py',
        # v4.15.2 Agent-C: 12 ensemble-rejection failures
        'tests/unit/test_v4_15_2_agent_c.py',
        # v4.15.3 Agent-A: 20 sibling-rejection failures
        'tests/unit/test_v4_15_3_agent_a.py',
        # v4.15 Agent-F: positional LED-warning failure (different
        # source factory; same warning style)
        'tests/unit/test_v4_15_agent_f.py',
    )
    proc = subprocess.run(
        [
            sys.executable, '-m', 'pytest',
            *representative_modules,
            '-W', 'error::DeprecationWarning',
            '--tb=no', '-q', '--no-header',
        ],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"Representative v4.15.x sample failed under "
        f"``-W error::DeprecationWarning`` after the v4.15.4 hygiene "
        f"closure.  Exit code: {proc.returncode}.\n"
        f"Modules sampled: {representative_modules!r}\n"
        f"stdout (tail):\n"
        + '\n'.join(proc.stdout.splitlines()[-25:])
        + "\nstderr (tail):\n"
        + '\n'.join(proc.stderr.splitlines()[-25:])
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

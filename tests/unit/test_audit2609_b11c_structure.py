"""WP-B11c -- the STRUCTURE gates for the Wave-5.3 hygiene split.

Three claims, each a DECISION about the import graph and the monkeypatch
surface rather than a reading of any number:

1. ``rcwa/_core.py``'s BLAS-thread cap lives in the ``rcwa._blas`` leaf, every
   name ``_core.__all__`` promises still resolves out of ``_core`` and names the
   SAME object, and the cap's mutable state is read out of ``_blas``'s own
   globals -- so a test that substitutes it must patch ``_blas``, and a stale
   ``setattr`` on ``_core`` fails loudly instead of binding a shadow attribute.
2. The ``_lens_real <-> lenses`` import cycle is gone, and the two live things
   the move needed -- a ``lenses._NUMBA_AVAILABLE = False`` monkeypatch that the
   kernel must still SEE, and the lazily populated ``cp`` / ``_ne`` slots -- work
   through the PEP 562 ``__getattr__`` forward rather than binding a stale
   ``None``.
3. The ``lenses <-> lenses_maslov`` back-edge is gone: the four names it carried
   live in the leaf, ``lenses`` re-exports them by identity, and
   ``lenses_maslov`` no longer imports ``lenses`` at module scope.

The import-graph half of each claim is measured TWICE, by two independent
instruments, because either alone can be fooled.  Structurally, by the
module-level-only AST walk WP-B11 part a used (what the source SAYS).
Dynamically, by ``validation/probe_wp_b11c/import_graph.py``, an ``__import__``
hook in a child process that records only module-body imports (what the
interpreter DOES) -- the one instrument that sees the edge a cycle is made of,
a module-level ``from .lenses import x`` whose target is already in
``sys.modules`` half-initialised and which therefore executes no loader and
leaves no trace in ``-X importtime`` or in ``sys.modules``.  Both fail on the
pre-refactor tree: the AST walk and the recorder each report the same two
2-cycles part a reported, and the ``_core -> _blas`` edge does not exist.

Every bar here is a structural fact with no build-dependent component: nothing
in this file reads a float, a count of modes, or a timing.
"""
from __future__ import annotations

import ast
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import lumenairy

PKG = Path(lumenairy.__file__).resolve().parent
ELEM = PKG / "elements"
RCWA = ELEM / "rcwa"

# The lens family part a walked (11 modules), plus the leaves it created.
LENS_FAMILY = (
    "lenses", "lenses_maslov", "lenses_gbd", "lens_config",
    "_lens_real", "_lens_thin", "_lens_traced", "_lens_traced_multibranch",
    "_lens_traced_uniform", "_lens_imap", "_lens_jax", "_lens_kernels",
)


def _module_level_imports(path: Path, package_names) -> set:
    """Sibling modules ``path`` imports AT MODULE SCOPE (the edges that execute
    at import time).  In-function imports are deliberately NOT edges: they are
    how the family already breaks cycles it cannot otherwise break."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in tree.body:                       # module scope only
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module and node.module in package_names:
                out.add(node.module)
            elif node.module:
                tail = node.module.rsplit(".", 1)[-1]
                if tail in package_names and node.level:
                    out.add(tail)
        elif isinstance(node, ast.Import):
            for a in node.names:
                tail = a.name.rsplit(".", 1)[-1]
                if tail in package_names:
                    out.add(tail)
    return out


def _family_graph(directory: Path, names) -> dict:
    return {n: _module_level_imports(directory / f"{n}.py", set(names))
            for n in names if (directory / f"{n}.py").exists()}


def _two_cycles(graph: dict) -> set:
    return {tuple(sorted((a, b)))
            for a, outs in graph.items() for b in outs
            if b in graph and a in graph[b]}


RECORDER = PKG.parent / "validation" / "probe_wp_b11c" / "import_graph.py"


@functools.lru_cache(maxsize=1)
def _executed_edges() -> tuple:
    """The module-level import edges that ACTUALLY EXECUTE, recorded in a
    child process by ``validation/probe_wp_b11c/import_graph.py``.

    A child, not an import here: this process has long since imported the
    whole library, so nothing would execute.  The recorder wraps
    ``__import__`` and keeps only the calls whose frame is a MODULE BODY
    (``f_locals is f_globals``, true in no function, comprehension or class
    body), which makes it the one instrument that sees the edge a cycle is
    made of -- a module-level ``from .lenses import x`` whose target is
    already in ``sys.modules`` half-initialised, and which therefore executes
    no loader and leaves no trace in ``-X importtime`` or in ``sys.modules``
    afterwards.
    """
    if not RECORDER.exists():            # an installed wheel, not a checkout
        pytest.skip(f"the recorder is not in this tree: {RECORDER}")
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(PKG.parent)
    out = subprocess.run([sys.executable, str(RECORDER), "lumenairy"],
                         capture_output=True, text=True, env=env,
                         cwd=str(PKG.parent))
    assert out.returncode == 0, (
        f"recorder failed:\n{out.stdout[-2000:]}\n{out.stderr[-4000:]}")
    data = json.loads(out.stdout[out.stdout.index("{"):])
    return tuple((a, b) for a, b, _fl in data["edges"] if a)


def _executed_family_graph(prefix: str, names) -> dict:
    """The recorded edges, restricted to one family and stripped to bare
    module names."""
    keep = set(names)
    graph = {n: set() for n in keep}
    for a, b in _executed_edges():
        if not (a.startswith(prefix) and b.startswith(prefix)):
            continue
        sa, sb = a[len(prefix):], b[len(prefix):]
        if sa in keep and sb in keep:
            graph[sa].add(sb)
    return graph


def _defining_globals(fn):
    """The module namespace a callable's BODY reads its globals out of.

    ``@contextlib.contextmanager`` replaces the function with a helper defined
    in ``contextlib``, so the honest question is asked of ``__wrapped__``.
    """
    return getattr(fn, "__wrapped__", fn).__globals__


# ===========================================================================
# 1. rcwa/_core.py -- the BLAS leaf and its monkeypatch surface
# ===========================================================================

BLAS_REEXPORTED = ("_BLAS_STATE", "_get_blas_threads", "set_blas_threads",
                   "rcwa_blas_threads", "_blas_threads_quiet", "_blas_limit",
                   "_with_blas_limit")
# Read through ``_blas``'s globals at call time -- the definition site is the
# only place a substitution is seen.
BLAS_DEFINITION_SITE_ONLY = ("_BLAS_CONTROLLER", "_BLAS_CONTROLLER_UNAVAILABLE",
                             "_BLAS_CONTROLLER_LOCK",
                             "_BLAS_WARNED_UNCONTROLLABLE",
                             "_threadpoolctl_available",
                             "_warn_blas_uncontrollable",
                             "_get_blas_controller")


def test_the_blas_cap_is_a_leaf_module():
    """``rcwa._blas`` imports the standard library and ``lumenairy._knobs``
    (itself a leaf) and NOTHING else from the package, so every engine can
    depend on it without an edge back."""
    src = (RCWA / "_blas.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    pkg_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            pkg_imports.append("." * node.level + (node.module or ""))
        elif isinstance(node, ast.Import):
            pkg_imports += [a.name for a in node.names
                            if a.name.startswith("lumenairy")]
    assert pkg_imports == ["..._knobs"], pkg_imports


def test_every_promised_core_name_still_resolves_and_is_the_same_object():
    """``__all__`` is an import-path contract even for the private names, so
    the split may not drop one -- and the re-export must be the SAME object,
    not a copy (``_BLAS_STATE`` is mutated through whichever path finds it)."""
    from lumenairy.elements import rcwa as facade
    from lumenairy.elements.rcwa import _blas, _core

    missing = [n for n in _core.__all__ if not hasattr(_core, n)]
    assert missing == [], missing
    for n in BLAS_REEXPORTED:
        assert n in _core.__all__, n
        assert getattr(_core, n) is getattr(_blas, n), n
        assert getattr(facade, n) is getattr(_blas, n), n


def test_the_cap_state_is_read_out_of_the_leaf_so_the_patch_site_is_the_leaf():
    """The MEASUREMENT behind "patch the definition site": every function that
    reads the cap's mutable state has ``_blas``'s ``__dict__`` as its global
    namespace, so only a substitution made THERE is read."""
    from lumenairy.elements.rcwa import _blas

    for fn in ("_threadpoolctl_available", "_warn_blas_uncontrollable",
               "_get_blas_controller", "_blas_limit", "set_blas_threads",
               "_get_blas_threads", "rcwa_blas_threads",
               "_blas_threads_quiet"):
        assert _defining_globals(getattr(_blas, fn)) is vars(_blas), fn


def test_the_state_names_are_not_re_exported_so_a_stale_patch_fails_loudly():
    """A silent no-op is the hazard this split had to avoid: if ``_core``
    carried a copy of ``_BLAS_CONTROLLER``, an old ``setattr(_core, ...)``
    would bind a shadow attribute nothing reads and the test asserting the
    cap's behaviour would pass while measuring the unpatched code.  Absent
    from ``_core``, the same line raises AttributeError."""
    from lumenairy.elements.rcwa import _blas, _core

    for n in BLAS_DEFINITION_SITE_ONLY:
        assert hasattr(_blas, n), f"{n} vanished from the definition site"
        assert not hasattr(_core, n), (
            f"_core re-exports {n}; a stale monkeypatch on _core would now "
            f"succeed silently instead of raising AttributeError")
    with pytest.raises(AttributeError):
        getattr(_core, "_BLAS_CONTROLLER")


def test_the_four_monkeypatching_files_patch_the_definition_site():
    """Fail-closed inventory: the four test files that substitute the cap's
    state must name ``rcwa._blas``.  If a fifth file starts patching one of
    these names on ``_core``, this test names it."""
    here = Path(__file__).resolve().parent
    offenders = []
    for p in sorted(here.glob("test_*.py")):
        if p.name == Path(__file__).name:
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        if not any(n in src for n in BLAS_DEFINITION_SITE_ONLY):
            continue
        tree = ast.parse(src)
        core_aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and \
                    node.module.endswith("elements.rcwa"):
                for a in node.names:
                    if a.name == "_core":
                        core_aliases.add(a.asname or a.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and \
                    isinstance(node.func, ast.Attribute) and \
                    node.func.attr in ("setattr", "delattr") and node.args and \
                    isinstance(node.args[0], ast.Name) and \
                    node.args[0].id in core_aliases and len(node.args) > 1 and \
                    isinstance(node.args[1], ast.Constant) and \
                    node.args[1].value in BLAS_DEFINITION_SITE_ONLY:
                offenders.append(f"{p.name}:{node.lineno} {node.args[1].value}")
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and \
                            isinstance(t.value, ast.Name) and \
                            t.value.id in core_aliases and \
                            t.attr in BLAS_DEFINITION_SITE_ONLY:
                        offenders.append(f"{p.name}:{t.lineno} {t.attr}")
    assert offenders == [], (
        "these sites patch the cap's state on rcwa._core, which does not "
        "re-export it: " + "; ".join(offenders))


def test_the_executed_rcwa_graph_has_the_leaf_edge_and_no_back_edge():
    """The dynamic half: among the edges that ACTUALLY EXECUTE, ``_core``
    imports ``_blas`` and ``_blas`` imports nothing in the rcwa package.
    FAILS on the pre-refactor tree, where the ``_core -> _blas`` edge does not
    exist at all because the cap lives inside ``_core``."""
    graph = _executed_family_graph(
        "lumenairy.elements.rcwa.",
        ("_core", "_blas", "_geometry", "oned", "twod", "stack"))
    assert "_blas" in graph["_core"], sorted(graph["_core"])
    assert graph["_blas"] == set(), sorted(graph["_blas"])
    assert _two_cycles(graph) == set(), sorted(_two_cycles(graph))


# ===========================================================================
# 2. the _lens_real <-> lenses cycle
# ===========================================================================

#: The cycles still open at THIS commit -- EMPTY, and an equality rather than
#: an upper bound so that a cycle which closes forces this line to be revisited
#: in the same commit.  The audit's TESTS-ARCH section counted four; part a
#: re-measured three and closed two, this package closed the last two.
_OPEN_CYCLES = set()


def test_the_lens_family_carries_only_the_cycles_still_open():
    """The structural half: what the SOURCE says, by the module-level-only AST
    walk WP-B11 part a used."""
    graph = _family_graph(ELEM, LENS_FAMILY)
    assert _two_cycles(graph) == _OPEN_CYCLES, sorted(_two_cycles(graph))


def test_lens_real_does_not_import_the_hub_at_module_scope():
    """The specific edge: ``_lens_real`` used to read ``surface_sag_general``
    and ``surface_sag_biconic`` out of ``lenses``."""
    assert "lenses" not in _module_level_imports(ELEM / "_lens_real.py",
                                                 set(LENS_FAMILY))


def test_the_executed_lens_graph_agrees_with_the_source():
    """The dynamic half of the same claim, on the edges that actually EXECUTE.
    FAILS on the pre-refactor tree, where the recorder reports the same two
    pairs the AST walk does."""
    graph = _executed_family_graph("lumenairy.elements.", LENS_FAMILY)
    assert _two_cycles(graph) == _OPEN_CYCLES, sorted(_two_cycles(graph))
    assert "lenses" not in graph["_lens_real"], sorted(graph["_lens_real"])


def test_the_numba_gate_monkeypatch_still_reaches_the_kernel():
    """The live-state half of the move.  The suite reaches the pure-NumPy sag
    arm by setting ``lenses._NUMBA_AVAILABLE = False`` on a box that HAS numba;
    the kernel reads the flag at call time.  After the move the flag lives in
    the leaf, so ``lenses`` forwards the write -- and this test proves the
    kernel SEES it, by making the two arms disagree at the flag rather than by
    reading a number the build is entitled to move."""
    from lumenairy.elements import _lens_kernels, lenses

    assert lenses._NUMBA_AVAILABLE is _lens_kernels._NUMBA_AVAILABLE
    saved = _lens_kernels._NUMBA_AVAILABLE
    try:
        lenses._NUMBA_AVAILABLE = False
        assert _lens_kernels._NUMBA_AVAILABLE is False, (
            "a write through the lenses facade no longer reaches the kernel's "
            "own global -- every test that forces the pure-NumPy arm by "
            "setting lenses._NUMBA_AVAILABLE = False is now silently "
            "exercising the numba arm instead")
        assert lenses._NUMBA_AVAILABLE is False
    finally:
        _lens_kernels._NUMBA_AVAILABLE = saved
        lenses.__dict__.pop("_NUMBA_AVAILABLE", None)
    assert lenses._NUMBA_AVAILABLE is saved


def test_the_lazy_backend_slots_forward_live_and_not_a_stale_none():
    """``cp`` and ``_ne`` are populated on FIRST USE, so a plain
    ``from ._lens_kernels import cp`` in ``lenses`` would bind whatever the slot
    held at import time (``None``) forever.  The PEP 562 forward re-reads the
    leaf on every attribute access; this test proves it by moving the leaf's
    slot and watching the facade follow."""
    from lumenairy.elements import _lens_kernels, lenses

    for slot in ("cp", "_ne"):
        saved = getattr(_lens_kernels, slot)
        sentinel = object()
        try:
            setattr(_lens_kernels, slot, sentinel)
            assert getattr(lenses, slot) is sentinel, (
                f"lenses.{slot} is a stale snapshot, not a live view of the "
                f"leaf's lazily populated slot")
        finally:
            setattr(_lens_kernels, slot, saved)
        assert getattr(lenses, slot) is saved


def test_the_module_getattr_still_raises_for_a_name_nobody_defines():
    """A forward that answers everything hides typos and breaks ``hasattr``
    probes, so the fallthrough must still raise ``AttributeError``."""
    from lumenairy.elements import lenses

    with pytest.raises(AttributeError):
        lenses._this_name_exists_nowhere_at_all
    assert not hasattr(lenses, "_this_name_exists_nowhere_at_all")


# ===========================================================================
# 3. the lenses <-> lenses_maslov back-edge
# ===========================================================================

MASLOV_MOVED = ("NUMEXPR_AVAILABLE", "_ensure_numexpr_loaded",
                "_fit_normaliser", "_multi_indices_total_degree")


def test_lenses_maslov_does_not_import_the_hub_at_module_scope():
    assert "lenses" not in _module_level_imports(ELEM / "lenses_maslov.py",
                                                 set(LENS_FAMILY))


def test_the_executed_graph_carries_no_maslov_back_edge():
    graph = _executed_family_graph("lumenairy.elements.", LENS_FAMILY)
    assert "lenses" not in graph["lenses_maslov"], \
        sorted(graph["lenses_maslov"])
    # ... while the FORWARD edge, which is the hub's whole job, is still there.
    assert "lenses_maslov" in graph["lenses"], sorted(graph["lenses"])


def test_the_four_moved_names_resolve_from_both_ends_and_are_identical():
    """The re-export contract: every existing ``from ...lenses import
    _fit_normaliser`` keeps working, and the object is the leaf's."""
    from lumenairy.elements import _lens_kernels, lenses, lenses_maslov

    for n in MASLOV_MOVED:
        assert hasattr(_lens_kernels, n), n
        assert getattr(lenses, n) is getattr(_lens_kernels, n), n
        assert getattr(lenses_maslov, n) is getattr(_lens_kernels, n), n


def test_the_numexpr_gate_is_live_through_the_facade_too():
    """``NUMEXPR_AVAILABLE`` has the same shape as ``_NUMBA_AVAILABLE``: it is
    a gate the suite flips, and ``_ensure_numexpr_loaded`` reads it at call
    time out of the leaf."""
    from lumenairy.elements import _lens_kernels, lenses

    assert _lens_kernels._ensure_numexpr_loaded.__globals__ is \
        vars(_lens_kernels)
    saved = _lens_kernels.NUMEXPR_AVAILABLE
    try:
        _lens_kernels.NUMEXPR_AVAILABLE = not saved
        assert lenses.NUMEXPR_AVAILABLE is (not saved)
    finally:
        _lens_kernels.NUMEXPR_AVAILABLE = saved

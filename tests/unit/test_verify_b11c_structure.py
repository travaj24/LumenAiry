"""VERIFY-B11c -- the independent re-verification's own DECISION gates.

These are the gaps found while re-measuring WP-B11c
(``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B11c.md``).
Every one of them is a structural fact -- an identity, an attribute protocol, an
import edge or a source-level inventory.  Nothing in this file reads a float, a
mode count or a timing, so nothing here can be per-build.

What each group closes, and why the WP's own file does not already close it:

1. **The patch-site inventory is form-specific.**
   ``test_audit2609_b11c_structure.py::test_the_four_monkeypatching_files_patch_the_definition_site``
   recognises exactly two spellings of a stale patch -- ``monkeypatch.setattr(_core, 'NAME', ...)``
   and ``_core.NAME = ...`` -- and only when ``_core`` arrived through
   ``from ...elements.rcwa import _core``.  The string-target form
   (``monkeypatch.setattr('lumenairy.elements.rcwa._core._BLAS_CONTROLLER', ...)``),
   the ``import lumenairy.elements.rcwa._core as C`` form and
   ``sys.modules[...]`` are not seen, and ``tests/integration`` is not walked at
   all.  Those are the forms a future patch is most likely to arrive in.

2. **The lens side has the opposite failure mode from the BLAS side, and it is
   not fenced.**  WP-B11c's stated principle is that a patch target which moved
   should fail LOUDLY: the cap's state is deliberately absent from ``_core``, so
   a stale ``setattr`` raises.  On the lens side, every moved name is
   re-exported into ``lenses`` with ``X as X``, so a stale
   ``monkeypatch.setattr(lenses, '_is_cupy_array', fake)`` SUCCEEDS, binds a
   shadow in ``lenses.__dict__``, and is read by nothing -- the silent no-op the
   WP set out to avoid.  Only eight names are live-forwarded; the other
   call-time-read names are not.  No test in the suite does this today, which is
   why nothing is red; the inventory below keeps it that way.

3. **The closure walk behind the restated leaf property mis-resolves a package
   ``__init__``.**  ``test_audit2609_b11_hygiene.py``'s walker computes a
   relative import's base from ``mod.split('.')[:-1]``, which is right for a
   module and wrong for a package: inside ``lumenairy/backend/__init__.py`` a
   ``from ._optional import x`` resolves to ``lumenairy._optional``, a path that
   does not exist, so the walk silently drops that whole subtree.  Today's
   closure is ``{backend._optional}``, a plain module, so the answer is right;
   the walk that produced it is not, and a future leaf import through any
   package would go unchecked.  Re-derived here with a resolver that handles
   both.

4. **Four of the WP's sixteen ids depend on a child process running a file in
   ``validation/``, and skip when it is missing.**  The same claims are restated
   here from the source alone, so an installed wheel still gates them.

5. **The facade's attribute protocol is asserted for reads and one write.**
   Delete, ``dir``, ``importlib.reload``, pickling of the moved callables and
   the monkeypatch save/undo cycle over ALL eight forwarded names are not.
"""
from __future__ import annotations

import ast
import importlib
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_kernels, lenses
from lumenairy.elements.rcwa import _core as _rcore

# ``rcwa._blas`` is imported INSIDE each test that needs it, not here: the file
# has to be collectable on the pre-refactor tree, where the module does not
# exist, so that the fail-before reads per-id rather than as one collection
# error.

PKG = Path(la.__file__).resolve().parent
ELEM = PKG / "elements"
TESTS = Path(__file__).resolve().parent.parent          # tests/

#: The cap's mutable state and the three functions that read it at call time --
#: deliberately absent from ``_core`` so a stale patch raises.
BLAS_LEAF_ONLY = ("_BLAS_CONTROLLER", "_BLAS_CONTROLLER_UNAVAILABLE",
                  "_BLAS_CONTROLLER_LOCK", "_BLAS_WARNED_UNCONTROLLABLE",
                  "_threadpoolctl_available", "_warn_blas_uncontrollable",
                  "_get_blas_controller")

#: The eight names ``lenses`` forwards LIVE, in both directions.
LIVE_FORWARD = ("cp", "_ne", "NUMEXPR_AVAILABLE", "_NUMBA_AVAILABLE",
                "_numba", "_njit", "_prange", "_NUMBA_KERNELS")

#: Moved to the leaf, re-exported into ``lenses`` BY VALUE, and read at call
#: time out of the leaf's globals by the code that moved with them.  Patching
#: one of these on ``lenses`` succeeds and reaches nothing.
LEAF_READ_BY_VALUE = ("CUPY_AVAILABLE", "_is_cupy_array", "_ensure_cupy_loaded",
                      "_load_numba", "_get_aspheric_sag_accum_numba",
                      "_ensure_numexpr_loaded", "_collect_semi_diameters",
                      "_warn_if_aperture_exceeds_grid")


# ===========================================================================
# 1. The BLAS cap: the patch-site inventory, in every form a patch can take
# ===========================================================================

def _core_aliases(tree: ast.AST) -> set:
    """Every local name in one test file that refers to the ``rcwa._core``
    MODULE, by any of the four spellings a test can use to get it."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("elements.rcwa"):
                out |= {a.asname or a.name for a in node.names
                        if a.name == "_core"}
            # ``from ...elements import rcwa`` then ``rcwa._core`` is an
            # attribute chain, handled by the ``_dotted`` walk below.
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "lumenairy.elements.rcwa._core" and a.asname:
                    out.add(a.asname)
    return out


def _dotted(node) -> str:
    """``a.b.c`` for an Attribute/Name chain, else ''."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def test_no_test_file_patches_the_cap_state_on_core_in_any_form():
    """Fail-closed, and form-agnostic.  Walks BOTH test directories and catches

    * ``monkeypatch.setattr(alias, 'NAME', ...)`` / ``alias.NAME = ...`` for an
      alias bound by ``from ...rcwa import _core`` OR
      ``import lumenairy.elements.rcwa._core as C``;
    * ``anything.rcwa._core.NAME = ...`` (the attribute chain through the
      package -- and NOT a local alias merely SPELLED ``_core``, which three of
      the four re-pointed files use for ``rcwa._blas``);
    * the STRING target form
      ``monkeypatch.setattr('lumenairy.elements.rcwa._core.NAME', ...)``,
      which the WP's inventory does not see and which is the spelling a future
      patch is most likely to arrive in.

    A hit here is a SILENT no-op waiting to happen -- except that ``_core`` does
    not carry the name, so it is in fact a loud ``AttributeError``; this test
    names the file and line before a reader has to read the traceback.
    """
    offenders = []
    for path in sorted(TESTS.rglob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        if not any(n in src for n in BLAS_LEAF_ONLY):
            continue
        tree = ast.parse(src)
        aliases = _core_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if not isinstance(t, ast.Attribute):
                        continue
                    if t.attr not in BLAS_LEAF_ONLY:
                        continue
                    owner = _dotted(t.value)
                    if owner in aliases or owner.endswith("rcwa._core"):
                        offenders.append(f"{path.name}:{t.lineno} {t.attr}")
            if isinstance(node, ast.Call) and isinstance(node.func,
                                                         ast.Attribute) \
                    and node.func.attr in ("setattr", "delattr") and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and \
                        isinstance(first.value, str):
                    tgt = first.value
                    if tgt.startswith("lumenairy.elements.rcwa._core.") and \
                            tgt.rsplit(".", 1)[-1] in BLAS_LEAF_ONLY:
                        offenders.append(f"{path.name}:{node.lineno} {tgt}")
                    continue
                owner = _dotted(first)
                if (owner in aliases or owner.endswith("rcwa._core")) and \
                        len(node.args) > 1 and \
                        isinstance(node.args[1], ast.Constant) and \
                        node.args[1].value in BLAS_LEAF_ONLY:
                    offenders.append(
                        f"{path.name}:{node.lineno} {node.args[1].value}")
    assert offenders == [], (
        "these sites substitute the BLAS cap's state on rcwa._core, which "
        "does not carry it since WP-B11c; patch rcwa._blas instead: "
        + "; ".join(offenders))


@pytest.mark.parametrize("name", BLAS_LEAF_ONLY)
def test_a_stale_monkeypatch_on_core_raises_rather_than_binding_a_shadow(
        name, monkeypatch):
    """The WP asserts the ABSENCE of the attribute; this asserts the
    CONSEQUENCE the absence exists for, in the exact shape a test would use.
    ``monkeypatch.setattr`` with the default ``raising=True`` must refuse."""
    from lumenairy.elements.rcwa import _blas as _rblas

    assert hasattr(_rblas, name), f"{name} vanished from the definition site"
    with pytest.raises(AttributeError):
        monkeypatch.setattr(_rcore, name, object())
    assert name not in vars(_rcore), (
        f"a refused patch still bound {name} on _core")


def test_the_blas_leaf_and_the_only_module_it_imports_are_both_leaves():
    """The leaf claim is transitive or it is nothing: ``_blas`` imports
    ``lumenairy._knobs`` (the WP's own table row says "nothing in the package",
    which is loose -- ``_knobs`` IS in the package), so ``_knobs`` has to be a
    leaf too or the edge is a dependency in disguise."""
    knobs = (PKG / "_knobs.py").read_text(encoding="utf-8")
    pkg_imports = []
    for node in ast.walk(ast.parse(knobs)):
        if isinstance(node, ast.ImportFrom) and node.level:
            pkg_imports.append("." * node.level + (node.module or ""))
        elif isinstance(node, ast.ImportFrom) and (
                node.module or "").startswith("lumenairy"):
            pkg_imports.append(node.module)
        elif isinstance(node, ast.Import):
            pkg_imports += [a.name for a in node.names
                            if a.name.startswith("lumenairy")]
    assert pkg_imports == [], (
        f"lumenairy._knobs is no longer a leaf ({pkg_imports}), so "
        f"rcwa._blas's one package import is no longer harmless")


def test_every_capped_engine_reaches_the_leafs_controller():
    """The functional half of "``_core``'s consumers still reach the leaf's
    state": substitute the LEAF's ``_get_blas_controller`` with a counter and
    run one call through each family that applies the cap -- the 1-D and 2-D
    RCWA entry points (decorated with ``_with_blas_limit``), the threaded
    ``RCWAStack`` sweep, and the two PMM stack sweeps (which cap per worker
    through ``_blas_threads_quiet`` + ``_blas_limit``).  A count that does not
    move means that family reads a controller from somewhere else -- which is
    the regression a stale re-export of the state would have produced.

    Decisions, not numbers: every assertion is "the counter moved".  Works
    with or without ``threadpoolctl``: the counter sits in front of the
    availability question, not behind it.
    """
    from lumenairy.elements.rcwa import _blas as _rblas

    calls = []
    real = _rblas._get_blas_controller()

    def _counting():
        calls.append(1)
        return real

    cell = np.full((8, 8), 1.0 + 0j)
    cell[2:6, 3:7] = 6.1
    saved = _rblas._get_blas_controller
    _rblas._get_blas_controller = _counting
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with la.rcwa_blas_threads(1):
                before = len(calls)
                la.rcwa_efficiency_1d(0.62e-6, 2.05, 1.0, 1.46, 1.0, 0.31e-6,
                                      0.42, 0.78e-6, n_orders=5)
                assert len(calls) > before, "rcwa 1-D never asked the leaf"

                before = len(calls)
                la.rcwa_efficiency_2d(0.62e-6, 0.58e-6, cell, 1.46, 1.0,
                                      0.16e-6, 0.78e-6, n_orders_x=1,
                                      n_orders_y=1)
                assert len(calls) > before, "rcwa 2-D never asked the leaf"

            st = la.RCWAStack(period=0.62e-6, n_superstrate=1.0,
                              n_substrate=1.46, n_orders=2)
            st.add_layer(0.12e-6, eps=4.2)
            st.set_source(0.78e-6, theta=0.0)
            before = len(calls)
            st.solve_vs_wavelength(np.array([0.77e-6, 0.79e-6]),
                                   max_workers=2)
            assert len(calls) > before, "the threaded RCWA sweep never asked"

            ps = la.PMMStack(0.62e-6, n_substrate=1.46, degree=6,
                             far_field_orders=3)
            ps.add_layer(0.12e-6, segments=[(0.42, 4.2 + 0j),
                                            (0.58, 1.0 + 0j)])
            before = len(calls)
            ps.solve_vs_wavelength(np.array([0.77e-6, 0.79e-6]), theta=0.0,
                                   max_workers=2)
            assert len(calls) > before, "the threaded PMM sweep never asked"

            hy = la.PMM2DStackHybrid(0.62e-6, 0.58e-6, n_substrate=1.46,
                                     n_superstrate=1.0, n_orders=1, degree=5)
            hy.add_layer(0.09e-6, eps_cell=cell)
            before = len(calls)
            hy.solve_vs_wavelength(np.array([0.77e-6, 0.79e-6]), theta=0.0,
                                   max_workers=1)
            assert len(calls) > before, "the PMM-2D hybrid sweep never asked"
    finally:
        _rblas._get_blas_controller = saved


# ===========================================================================
# 2. The lens facade: the whole attribute protocol, and the by-value asymmetry
# ===========================================================================

@pytest.mark.parametrize("name", LIVE_FORWARD)
def test_the_facade_forwards_read_write_and_delete_for_every_live_name(name):
    """Read, WRITE and DELETE, for all eight -- the WP asserts the read for all
    eight, the write for ``cp`` and ``_NUMBA_AVAILABLE``, and the delete for
    none.  ``__delattr__`` is in the facade's protocol, so it is part of the
    contract whether or not a test uses it today."""
    assert name not in vars(lenses), (
        f"lenses carries its own {name!r}; the forward is shadowed")
    assert getattr(lenses, name) is getattr(_lens_kernels, name)

    saved = getattr(_lens_kernels, name)
    sentinel = object()
    try:
        setattr(lenses, name, sentinel)
        assert getattr(_lens_kernels, name) is sentinel, (
            f"a write of {name} through the facade did not reach the leaf")
        assert name not in vars(lenses), (
            f"the write bound a shadow copy of {name} on the facade")
        delattr(lenses, name)
        assert not hasattr(_lens_kernels, name), (
            f"a delete of {name} through the facade did not reach the leaf")
        assert not hasattr(lenses, name)
    finally:
        setattr(_lens_kernels, name, saved)
    assert getattr(lenses, name) is saved


@pytest.mark.parametrize("name", LIVE_FORWARD)
def test_monkeypatch_save_set_undo_leaves_no_shadow_for_every_live_name(
        name, monkeypatch):
    """The exact cycle the suite runs.  ``monkeypatch`` reads the old value
    through the forward, writes, and restores by writing again; if any of the
    three landed in ``lenses.__dict__`` the next test in the process would read
    a stale value that nothing maintains."""
    before = getattr(_lens_kernels, name)
    monkeypatch.setattr(lenses, name, "VERIFY-B11c")
    assert getattr(_lens_kernels, name) == "VERIFY-B11c"
    monkeypatch.undo()
    assert getattr(_lens_kernels, name) is before
    assert name not in vars(lenses), (
        f"monkeypatch's undo left a shadow {name!r} on the facade")


def test_dir_lenses_lists_every_forwarded_name_exactly_once():
    """A module's default ``__dir__`` lists ``__dict__`` only, so the forward
    needs its own -- and it must not double-count a name that is genuinely in
    the facade's dict."""
    listing = dir(lenses)
    assert len(listing) == len(set(listing)), "dir(lenses) has duplicates"
    missing = [n for n in LIVE_FORWARD if n not in listing]
    assert missing == [], missing
    assert listing == sorted(listing), "dir(lenses) is no longer sorted"


def test_the_facade_survives_importlib_reload_and_is_still_live():
    """``__class__`` assignment on a module is unusual enough that the reload
    path deserves a pin: reload re-executes the body against the SAME module
    object, which already has the facade's type and whose ``__setattr__``
    therefore intercepts every module-level binding the body performs."""
    reloaded = importlib.reload(lenses)
    assert reloaded is lenses
    assert type(lenses).__name__ == "_LensesFacade"
    assert type(lenses).__mro__[1].__name__ == "module"
    for name in LIVE_FORWARD:
        assert name not in vars(lenses), (
            f"reload bound a shadow {name!r} on the facade")
        assert getattr(lenses, name) is getattr(_lens_kernels, name)
    # ... and still live afterwards
    saved = _lens_kernels.cp
    try:
        _lens_kernels.cp = "AFTER-RELOAD"
        assert lenses.cp == "AFTER-RELOAD"
    finally:
        _lens_kernels.cp = saved


@pytest.mark.parametrize("name", ("surface_sag_general", "surface_sag_biconic",
                                  "_fit_normaliser",
                                  "_multi_indices_total_degree",
                                  "check_grid_vs_apertures"))
def test_the_moved_callables_pickle_and_resolve_by_identity(name):
    """``pickle`` writes ``__module__`` + ``__qualname__``, so a moved function
    changes what a pickle payload says -- it now names ``_lens_kernels``.  The
    contract that matters is that BOTH spellings still resolve, to the SAME
    object, and that a round trip returns that object."""
    fn = getattr(lenses, name)
    assert fn is getattr(_lens_kernels, name)
    assert fn.__module__ == "lumenairy.elements._lens_kernels", (
        f"{name}.__module__ is {fn.__module__!r}; the report's re-export table "
        f"says the definition lives in the leaf")
    assert pickle.loads(pickle.dumps(fn)) is fn


def test_the_names_lenses_re_exports_by_value_are_read_out_of_the_leaf():
    """The asymmetry this verification found, pinned rather than argued.

    Eight names are live-forwarded.  These eight are NOT: they are re-exported
    with ``X as X``, so ``lenses.<name>`` resolves -- and a
    ``monkeypatch.setattr(lenses, <name>, fake)`` therefore SUCCEEDS and binds
    a shadow that the kernel, which reads the leaf's globals, never sees.  At
    the base commit the same line DID change the kernel's behaviour, because
    the kernel lived in ``lenses``.  The property asserted here is the factual
    half: each one is read out of the leaf's namespace, so ``lenses`` is the
    wrong place to substitute it.
    """
    for name in LEAF_READ_BY_VALUE:
        assert hasattr(_lens_kernels, name), name
        assert getattr(lenses, name) is getattr(_lens_kernels, name), name
        assert name not in LIVE_FORWARD, (
            f"{name} is live-forwarded after all; move it out of this list")
        obj = getattr(_lens_kernels, name)
        if callable(obj):
            fn = getattr(obj, "__wrapped__", obj)
            assert fn.__globals__ is vars(_lens_kernels), (
                f"{name} does not read its globals out of the leaf")


def test_no_test_file_substitutes_a_by_value_reexport_on_the_lenses_facade():
    """Fail-closed inventory, the lens counterpart of the BLAS one.

    A patch of one of these names on ``lenses`` is the silent no-op WP-B11c's
    design note says it set out to avoid -- and unlike the BLAS state it cannot
    announce itself, because the re-export makes the ``setattr`` succeed.  No
    file does this today; this names the first one that tries.
    """
    offenders = []
    for path in sorted(TESTS.rglob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        if "lenses" not in src:
            continue
        tree = ast.parse(src)
        aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                    node.module or "").endswith("elements"):
                aliases |= {a.asname or a.name for a in node.names
                            if a.name == "lenses"}
            elif isinstance(node, ast.Import):
                aliases |= {a.asname for a in node.names
                            if a.name == "lumenairy.elements.lenses"
                            and a.asname}
        if not aliases:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and \
                            _dotted(t.value) in aliases and \
                            t.attr in LEAF_READ_BY_VALUE:
                        offenders.append(f"{path.name}:{t.lineno} {t.attr}")
            if isinstance(node, ast.Call) and isinstance(node.func,
                                                         ast.Attribute) and \
                    node.func.attr == "setattr" and len(node.args) > 1:
                first, second = node.args[0], node.args[1]
                if isinstance(first, ast.Constant) and isinstance(first.value,
                                                                  str):
                    tgt = first.value
                    if tgt.startswith("lumenairy.elements.lenses.") and \
                            tgt.rsplit(".", 1)[-1] in LEAF_READ_BY_VALUE:
                        offenders.append(f"{path.name}:{node.lineno} {tgt}")
                elif _dotted(first) in aliases and \
                        isinstance(second, ast.Constant) and \
                        second.value in LEAF_READ_BY_VALUE:
                    offenders.append(
                        f"{path.name}:{node.lineno} {second.value}")
    assert offenders == [], (
        "these sites substitute a by-value re-export on the lenses facade; "
        "the kernel reads the leaf, so the patch is a silent no-op -- patch "
        "lumenairy.elements._lens_kernels instead: " + "; ".join(offenders))


def test_the_numba_gate_flip_through_the_facade_changes_which_arm_runs():
    """The end-to-end shape, one level below the WP's ``_load_numba()`` pin: the
    gate is flipped THROUGH the facade and the builder that reads it is run, and
    what is asserted is which ARM the builder took -- by counting calls to the
    kernel factory, not by comparing two floating-point answers.

    Two arms are compared only for their DECISION.  Bit equality between the
    fused numba accumulation and the pure-NumPy term loop is a per-fixture
    reading, not a property: numba's LLVM is free to contract ``sag + c*h**k``
    into an FMA, and it does -- MEASURED here 2026-09-15 on Windows py3.14 /
    numba, ``{4: 3.1e2, 6: -8.4e6, 8: 1.7e11}`` on a 64 x 64 radius grid, the
    two arms differ in the last bits.  (The WP-B11c report's section 2.4 states
    the two as "equal bit for bit"; that holds for ITS coefficient set, not in
    general -- see VERIFY_WP-B11c.md.)  The relative gap is bounded below
    against the FMA floor so a genuinely different POLYNOMIAL still fails.
    """
    h_sq = (np.linspace(-9e-3, 9e-3, 64)[:, None] ** 2
            + np.linspace(-9e-3, 9e-3, 64)[None, :] ** 2)
    coeffs = {4: 3.1e2, 6: -8.4e6, 8: 1.7e11}
    asked = []
    real_factory = _lens_kernels._get_aspheric_sag_accum_numba

    def _counting():
        asked.append(1)
        return real_factory()

    saved_gate = _lens_kernels._NUMBA_AVAILABLE
    saved_numba = _lens_kernels._numba
    _lens_kernels._get_aspheric_sag_accum_numba = _counting
    try:
        fast = lenses.surface_sag_general(h_sq, 3.7e-2, -0.62, coeffs)
        assert asked, (
            "the sag builder never asked for the numba kernel while the gate "
            "was True -- there is no fast arm left to switch off")

        lenses._NUMBA_AVAILABLE = False            # the write, via the facade
        lenses._numba = None
        assert _lens_kernels._NUMBA_AVAILABLE is False
        assert _lens_kernels._load_numba() is False
        asked.clear()
        slow = lenses.surface_sag_general(h_sq, 3.7e-2, -0.62, coeffs)
        assert asked == [], (
            "the gate written through the facade did not reach the kernel: "
            "the builder still asked for the numba kernel, so every test that "
            "forces the pure-NumPy arm this way is a silent no-op")
    finally:
        _lens_kernels._get_aspheric_sag_accum_numba = real_factory
        _lens_kernels._NUMBA_AVAILABLE = saved_gate
        _lens_kernels._numba = saved_numba

    # The two arms evaluate the same polynomial; only the contraction differs.
    # Bar: 1e-12 relative, eight decades above the measured 2026-09-15 gap
    # (max relative 1.3e-16, i.e. one ULP) and far below a wrong term, which
    # would move the sag by O(1) of its own size.
    scale = float(np.nanmax(np.abs(fast)))
    gap = float(np.nanmax(np.abs(fast - slow)))
    assert gap <= 1e-12 * scale, (
        f"the two aspheric arms disagree by {gap / scale:.3e} relative, far "
        f"beyond a fused-multiply-add: they are not the same polynomial")


# ===========================================================================
# 3. The import graph, re-derived with a resolver that handles packages
# ===========================================================================

def _abs_import(level: int, module: str, owner: str, is_pkg: bool) -> str:
    """PEP 328's rule, for a MODULE and for a PACKAGE ``__init__`` alike.  One
    dot names the importer's package; each further dot climbs one level."""
    if not level:
        return module
    parts = owner.split(".")
    pkg = parts if is_pkg else parts[:-1]
    base = pkg[:len(pkg) - (level - 1)] if level > 1 else pkg
    head = ".".join(base)
    return f"{head}.{module}" if module else head


def _module_path(mod: str):
    """The file for a dotted name, module first then package ``__init__``."""
    rel = mod.split(".")[1:]
    direct = PKG.joinpath(*rel).with_suffix(".py")
    if direct.exists():
        return direct, False
    pkg_init = PKG.joinpath(*rel, "__init__.py")
    if pkg_init.exists():
        return pkg_init, True
    return None, False


def _package_imports(path: Path, mod: str, is_pkg: bool) -> set:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                out.add(_abs_import(node.level, node.module or "", mod,
                                    is_pkg))
            elif (node.module or "").startswith("lumenairy"):
                out.add(node.module)
        elif isinstance(node, ast.Import):
            out |= {a.name for a in node.names
                    if a.name.startswith("lumenairy")}
    return out


def test_the_leaf_closure_reaches_no_elements_module_with_packages_resolved():
    """The restated leaf property, re-derived.  Unlike the walk in
    ``test_audit2609_b11_hygiene.py`` this resolves a relative import inside a
    package ``__init__`` correctly, so a leaf import that goes through a
    subpackage is followed rather than silently dropped -- and it FAILS if any
    node in the closure cannot be resolved to a file, instead of skipping it.
    """
    start = "lumenairy.elements._lens_kernels"
    seen, pending, unresolved = set(), [start], []
    while pending:
        mod = pending.pop()
        if mod in seen:
            continue
        seen.add(mod)
        path, is_pkg = _module_path(mod)
        if path is None:
            unresolved.append(mod)
            continue
        pending += sorted(_package_imports(path, mod, is_pkg) - seen)
    assert unresolved == [], (
        f"the closure walk could not resolve {unresolved} to a file -- a "
        f"dropped node is an unchecked dependency")
    offenders = sorted(m for m in seen - {start}
                       if m.startswith("lumenairy.elements"))
    assert offenders == [], (
        f"_lens_kernels can reach {offenders}, a module in the family it "
        f"exists to keep acyclic.  Closure: {sorted(seen)}")


def test_the_lens_family_carries_no_module_level_two_cycle_by_a_second_walk():
    """The zero-cycle claim, from an independent walker: every
    ``lumenairy/elements/*.py`` in the family, module-body statements only,
    with ``ClassDef`` bodies excluded as well as functions (an import in a
    class body executes at import time, but none exists and counting it either
    way must not change the answer -- asserted by running both)."""
    fam = sorted(p.stem for p in ELEM.glob("*.py")
                 if p.stem == "lenses" or p.stem.startswith("lenses_")
                 or p.stem.startswith("_lens"))

    def edges(path, skip_classes):
        out = set()

        class V(ast.NodeVisitor):
            def visit_FunctionDef(self, node):      # noqa: N802
                return

            visit_AsyncFunctionDef = visit_FunctionDef
            visit_Lambda = visit_FunctionDef

            def visit_ClassDef(self, node):         # noqa: N802
                if not skip_classes:
                    self.generic_visit(node)

            def visit_ImportFrom(self, node):       # noqa: N802
                if node.level == 1 and node.module in fam:
                    out.add(node.module)

        V().visit(ast.parse(path.read_text(encoding="utf-8")))
        return out

    for skip in (True, False):
        graph = {m: edges(ELEM / f"{m}.py", skip) for m in fam}
        cycles = {tuple(sorted((a, b))) for a, outs in graph.items()
                  for b in outs if a in graph.get(b, ())}
        assert cycles == set(), (
            f"the lens family carries {sorted(cycles)} "
            f"(class bodies {'excluded' if skip else 'included'})")
        assert "lenses" not in graph["_lens_real"]
        assert "lenses" not in graph["lenses_maslov"]
        assert graph["_lens_kernels"] == set(), sorted(graph["_lens_kernels"])
        # the FORWARD edge, which is the hub's job, is still there
        assert {"_lens_real", "lenses_maslov"} <= graph["lenses"]


def test_the_rcwa_core_reads_the_blas_leaf_and_the_leaf_has_no_back_edge():
    """The ``_core -> _blas`` edge and the leaf's emptiness, from the SOURCE --
    so the claim still gates when the child-process recorder in ``validation/``
    is not shipped (an installed wheel), where the WP's dynamic ids skip."""
    rcwa_dir = ELEM / "rcwa"
    core = _package_imports(rcwa_dir / "_core.py",
                            "lumenairy.elements.rcwa._core", False)
    blas = _package_imports(rcwa_dir / "_blas.py",
                            "lumenairy.elements.rcwa._blas", False)
    assert "lumenairy.elements.rcwa._blas" in core, sorted(core)
    assert blas == {"lumenairy._knobs"}, sorted(blas)
    assert not any(m.startswith("lumenairy.elements") for m in blas), sorted(
        blas)

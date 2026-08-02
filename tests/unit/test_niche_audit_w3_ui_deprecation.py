"""AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 -- Territory A, wave 3.

Pins for the UI breadth pass (six user-reachable DEAD actions) and the
deprecation-registry rot.  Everything here is HEADLESS: PySide6 is not
installed on CI (nor on the audit box), so the dock code paths are pinned
the way the audit measured them -- ``importlib`` on the import TARGETS and
``inspect.signature(...).bind`` on the kwarg names, both read out of the
dock source with ``ast`` so the pin tracks the code rather than a
hand-copied duplicate of it.

Findings pinned
---------------
UI (all six were swallowed by ``except Exception`` and shipped green):

1. ``waveoptics_dock.py`` imported four whole-prescription propagators
   from ``propagators.propagation`` -- the v5.1.0 re-export shell for the
   ASM/Fresnel/RS/SAS/MFT family, which has never exported them.  All
   four menu choices (GBD / HFPI / Huygens-Fresnel / Subaperture) died
   with ``ImportError``.  Measured, and fixed to the owning submodules.
2. ``waveoptics_dock.py`` imported a nonexistent ``..detector`` (the
   module is ``..analysis.detector``): the detector checkbox was a no-op.
   Behind it, ``apply_detector`` returns ``(image, x_det, y_det)`` and the
   dock bound the 3-tuple to ``E_focus`` -- fixing only the import would
   have produced a ragged-array ``ValueError`` one line later.
3. ``coherence_dock.py`` / ``shack_hartmann_dock.py`` /
   ``lg_aberration_dock.py`` passed kwargs that do not exist on
   ``koehler_image`` / ``shack_hartmann`` / ``aberration_tensor``.
4. ``optimizer_dock.py`` built ``ToleranceAwareMerit(inner_merit=...)``;
   the parameter is ``sub_merit`` and ``perturbation_spec`` is required --
   selecting that merit aborted the whole optimizer run.
5. ``ui/surface_table.py`` (370 lines, ``SurfaceTableEditor``) had zero
   references repo-wide and is DELETED.
6. The specific handlers behind the fixed actions must report through the
   UI's diagnostics sink instead of ``pass`` (the class of silence that
   hid 1-4 for 20+ releases).  This is NOT a mass edit of the 92 empty
   handlers -- only the ones at the six dead actions.

Deprecation registry:

7. The removed-in banner emitted ``will be removed in v5.27`` FROM
   v5.29.0.  ``lumenairy/_deprecation.py`` now owns a removal-schedule
   registry (``REMOVAL_SCHEDULE`` / ``NEXT_REMOVAL_VERSION`` /
   ``resolve_removal_version``) that no banner can bypass, and
   ``check_removal_schedule`` makes it self-checking.  NO shim is removed
   here -- removal stays a release decision for each module owner.
8. ``_lens_jax.py`` documented a parameter ``lens_prescription``; the real
   name is ``prescription`` (``lens_prescription`` is the function's
   internal alias), so the documented call form raised ``TypeError``.
9. ``sources/core.py``'s Schell ``return_kind`` shim has ZERO production
   call sites -- pinned as measured (the fix is the owner's; see the
   report), together with the guarantee that when it IS called (tests,
   external wrappers) its banner names a future version.

Author: audit fix wave 3 (Territory A).
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import re
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import _deprecation as dep

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LUM_DIR = os.path.join(REPO_ROOT, 'lumenairy')
UI_DIR = os.path.join(LUM_DIR, 'ui')


# ===========================================================================
# Helpers -- read the dock source, resolve what it names
# ===========================================================================

def _parse(path: str) -> ast.Module:
    with open(path, 'r', encoding='utf-8') as fh:
        return ast.parse(fh.read(), filename=path)


def _ui_files() -> list[str]:
    return sorted(os.path.join(UI_DIR, fn) for fn in os.listdir(UI_DIR)
                  if fn.endswith('.py'))


def _relative_import_targets(path: str) -> list[tuple[int, str, tuple[str, ...]]]:
    """Return ``(lineno, absolute_module, names)`` for every relative
    ``from ... import ...`` in ``path`` (``lumenairy.ui.x`` is level 1,
    ``lumenairy.x`` is level 2)."""
    out = []
    for node in ast.walk(_parse(path)):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        base = 'lumenairy.ui' if node.level == 1 else 'lumenairy'
        mod = f'{base}.{node.module}' if node.module else base
        out.append((node.lineno, mod,
                    tuple(a.name for a in node.names)))
    return out


class _Any:
    """Stand-in for a real argument in a ``Signature.bind`` probe."""


_ANY = _Any()


def _calls_named(path: str, func_name: str) -> list[tuple[int, int, tuple[str, ...]]]:
    """Return ``(lineno, n_positional, kwarg_names)`` for every call in
    ``path`` whose callee's final name is ``func_name`` (matches both
    ``f(...)`` and ``mod.f(...)``)."""
    found = []
    for node in ast.walk(_parse(path)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, 'id', None) or getattr(fn, 'attr', None)
        if name != func_name:
            continue
        found.append((node.lineno, len(node.args),
                      tuple(k.arg for k in node.keywords if k.arg)))
    return found


def _owns_call(try_node: ast.Try, func_name: str) -> bool:
    """True when ``func_name`` is called in ``try_node``'s body and NOT
    inside a nested ``try`` -- i.e. this is the innermost handler that
    would catch the call's exception."""
    def hits(node) -> bool:
        if isinstance(node, ast.Try):
            return False          # a deeper handler owns anything in there
        if isinstance(node, ast.Call):
            name = (getattr(node.func, 'id', None)
                    or getattr(node.func, 'attr', None))
            if name == func_name:
                return True
        return any(hits(child) for child in ast.iter_child_nodes(node))

    return any(hits(stmt) for stmt in try_node.body)


def _assert_binds(callable_obj, path: str, func_name: str, *,
                  skip_self: bool = False) -> int:
    """Every call to ``func_name`` in ``path`` must bind against the real
    signature.  Returns the number of call sites checked (>= 1)."""
    sig = inspect.signature(callable_obj)
    sites = _calls_named(path, func_name)
    assert sites, (f'{os.path.basename(path)}: expected at least one call '
                   f'to {func_name}(); found none -- the pin has drifted '
                   f'from the code it guards.')
    for lineno, n_pos, kwargs in sites:
        args = [_ANY] * (n_pos + (1 if skip_self else 0))
        try:
            sig.bind(*args, **{k: _ANY for k in kwargs})
        except TypeError as exc:
            pytest.fail(
                f'{os.path.basename(path)}:{lineno}: '
                f'{func_name}({", ".join(kwargs)}) does not bind against '
                f'the real signature {sig}: {exc}')
    return len(sites)


# ===========================================================================
# Finding 1 + 2 -- every import target a dock names must exist
# ===========================================================================

# Sibling ui modules need PySide6 to import; the library targets do not.
_OPTIONAL_LIBRARY_MODULES = ('lumenairy.ui',)


class TestUIImportTargetsResolve:
    """A dock import naming a symbol that does not exist is a dead user
    action, and ``except Exception`` guarantees it ships green.  Sweep
    every relative import in ``lumenairy/ui/`` rather than only the five
    known-dead ones, so the next reorg cannot re-create the class."""

    def test_every_relative_ui_import_resolves(self):
        broken = []
        for path in _ui_files():
            for lineno, mod, names in _relative_import_targets(path):
                rel = os.path.relpath(path, REPO_ROOT)
                if mod.startswith(_OPTIONAL_LIBRARY_MODULES):
                    # ui/* sibling: presence-only (importing needs Qt).
                    if importlib.util.find_spec(mod) is None:
                        broken.append(f'{rel}:{lineno}: no module {mod!r}')
                    continue
                try:
                    spec = importlib.util.find_spec(mod)
                except (ImportError, ModuleNotFoundError) as exc:
                    broken.append(f'{rel}:{lineno}: find_spec({mod!r}): {exc}')
                    continue
                if spec is None:
                    broken.append(f'{rel}:{lineno}: no module {mod!r}')
                    continue
                obj = importlib.import_module(mod)
                for name in names:
                    if name == '*':
                        continue
                    if not hasattr(obj, name):
                        broken.append(
                            f'{rel}:{lineno}: {mod} does not export {name!r}')
        assert not broken, (
            'Dead import target(s) in lumenairy/ui/ -- each one is a user '
            'action that can only fail:\n  ' + '\n  '.join(broken))

    def test_whole_prescription_propagators_are_not_in_propagation_shell(self):
        """Discriminator for finding 1: the four names are NOT re-exported
        by ``propagators.propagation`` (so the pre-fix import could not
        have worked), and they ARE in the submodules the dock now names."""
        shell = importlib.import_module('lumenairy.propagators.propagation')
        owners = {
            'propagate_gbd_through_prescription':
                'lumenairy.propagators.gbd',
            'propagate_hfpi_through_prescription':
                'lumenairy.propagators.hfpi',
            'propagate_huygens_fresnel_through_prescription':
                'lumenairy.propagators.hf',
            'propagate_subaperture_asymptotic':
                'lumenairy.propagators.subaperture',
        }
        for name, owner in owners.items():
            assert not hasattr(shell, name), (
                f'propagators.propagation now exports {name}; a shim would '
                f'mask this class of reorg rot rather than fix it (cf. '
                f'test_audit_p1_gui_dead_import.py).')
            assert hasattr(importlib.import_module(owner), name)

    def test_waveoptics_dock_imports_the_four_from_their_owners(self):
        """Source-level: the dock's four import statements name the owning
        submodules."""
        path = os.path.join(UI_DIR, 'waveoptics_dock.py')
        want = {
            'lumenairy.propagators.gbd',
            'lumenairy.propagators.hfpi',
            'lumenairy.propagators.hf',
            'lumenairy.propagators.subaperture',
        }
        got = {mod for _l, mod, _n in _relative_import_targets(path)}
        assert want <= got, f'missing {sorted(want - got)}'
        assert 'lumenairy.detector' not in got

    def test_whole_prescription_calls_bind(self):
        """Finding 1, second half: the dock's call kwargs bind against the
        real signatures (the audit's signature-bind probe)."""
        path = os.path.join(UI_DIR, 'waveoptics_dock.py')
        for mod, fname in (
            ('lumenairy.propagators.gbd',
             'propagate_gbd_through_prescription'),
            ('lumenairy.propagators.hfpi',
             'propagate_hfpi_through_prescription'),
            ('lumenairy.propagators.hf',
             'propagate_huygens_fresnel_through_prescription'),
            ('lumenairy.propagators.subaperture',
             'propagate_subaperture_asymptotic'),
        ):
            fn = getattr(importlib.import_module(mod), fname)
            _assert_binds(fn, path, fname)


# ===========================================================================
# Finding 2 -- the detector unpack bug behind the dead import
# ===========================================================================

class TestWaveOpticsDetector:

    def test_apply_detector_returns_a_triple(self):
        from lumenairy.analysis.detector import apply_detector
        out = apply_detector(np.ones((16, 16), dtype=complex), 1e-6,
                             pixel_pitch=2e-6, quantum_efficiency=0.8)
        assert isinstance(out, tuple) and len(out) == 3
        img, x_det, y_det = out
        assert img.ndim == 2 and x_det.ndim == 1 and y_det.ndim == 1

    def test_the_pre_fix_single_assignment_would_have_died(self):
        """Discriminator: binding the 3-tuple to a field variable (the
        pre-fix code) cannot survive the very next line of the dock,
        ``I_focus = np.abs(E_focus) ** 2``."""
        from lumenairy.analysis.detector import apply_detector
        E_focus = apply_detector(np.ones((16, 16), dtype=complex), 1e-6,
                                 pixel_pitch=2e-6)
        with pytest.raises(ValueError):
            np.abs(E_focus) ** 2

    def test_dock_unpacks_three_targets_and_binds(self):
        path = os.path.join(UI_DIR, 'waveoptics_dock.py')
        from lumenairy.analysis.detector import apply_detector
        _assert_binds(apply_detector, path, 'apply_detector')
        # The assignment target must be a 3-element tuple.
        tree = _parse(path)
        targets = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            val = node.value
            if isinstance(val, ast.Call) and \
                    getattr(val.func, 'attr', getattr(val.func, 'id', None)) \
                    == 'apply_detector':
                targets.append(node.targets[0])
        assert targets, 'no apply_detector assignment found in the dock'
        for tgt in targets:
            assert isinstance(tgt, ast.Tuple) and len(tgt.elts) == 3, (
                'waveoptics_dock: apply_detector returns (image, x_det, '
                'y_det); binding it to a single name re-creates the latent '
                'unpack bug.')

    def test_amplitude_equivalent_convention(self):
        """The dock carries the electron image downstream as sqrt(clip(I)),
        so ``|E_focus|**2`` IS the detected image (one convention for
        I_focus / beam_power / d4sigma / the saved plane)."""
        from lumenairy.analysis.detector import apply_detector
        img, _x, _y = apply_detector(
            np.ones((32, 32), dtype=complex), 1e-6, pixel_pitch=2e-6,
            quantum_efficiency=0.5, read_noise_e=2.0, seed=0)
        E = np.sqrt(np.clip(np.asarray(img, dtype=np.float64),
                            0.0, None)).astype(np.complex128)
        assert np.allclose(np.abs(E) ** 2, np.clip(img, 0.0, None))

    def test_detector_handler_reports_instead_of_pass(self):
        """Finding 6, scoped to this action: the handler around the
        detector call must route through the UI diagnostics sink."""
        path = os.path.join(UI_DIR, 'waveoptics_dock.py')
        tree = _parse(path)
        checked = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            # Only the INNERMOST try owning the call (the whole
            # ``_run_impl`` body sits inside an outer try as well).
            if not _owns_call(node, 'apply_detector'):
                continue
            checked += 1
            for handler in node.handlers:
                assert not all(isinstance(s, ast.Pass) for s in handler.body), (
                    'waveoptics_dock: the detector handler still swallows '
                    'the exception silently.')
                dumped = ast.dump(ast.Module(body=handler.body,
                                             type_ignores=[]))
                assert 'report' in dumped, (
                    'waveoptics_dock: the detector handler must report the '
                    'exception (diag.report) so a broken detector option '
                    'cannot silently no-op again.')
        assert checked == 1, (
            f'expected exactly one try/except around apply_detector; '
            f'found {checked}')


# ===========================================================================
# Finding 3 -- stale kwargs in three docks
# ===========================================================================

class TestStaleDockKwargs:

    def test_coherence_dock_koehler_image_binds(self):
        path = os.path.join(UI_DIR, 'coherence_dock.py')
        n = _assert_binds(la.koehler_image, path, 'koehler_image')
        assert n >= 2, ('expected both the Schell tab and the Koehler tab '
                        'call sites')

    def test_koehler_image_has_no_legacy_kwargs(self):
        """Discriminator: the three names the dock used to pass are not
        parameters, so the pre-fix call could only raise."""
        params = inspect.signature(la.koehler_image).parameters
        for gone in ('source_sigma', 'N', 'n_modes'):
            assert gone not in params
        assert 'object_field' in params

    def test_shack_hartmann_dock_binds(self):
        path = os.path.join(UI_DIR, 'shack_hartmann_dock.py')
        _assert_binds(la.shack_hartmann, path, 'shack_hartmann')
        params = inspect.signature(la.shack_hartmann).parameters
        assert 'lenslet_focal' in params
        for gone in ('lenslet_focal_length', 'n_zernike'):
            assert gone not in params

    def test_shack_hartmann_returns_a_5_tuple_not_an_object(self):
        """The dock's ``res.slopes_x`` reads were dead too: the function
        returns a tuple.  Pin the arity + the dock's Zernike fit path."""
        E = np.ones((64, 64), dtype=complex)
        out = la.shack_hartmann(E, 1e-5, wavelength=550e-9,
                                lenslet_pitch=1e-4, lenslet_focal=5e-3)
        assert isinstance(out, tuple) and len(out) == 5
        sx, sy, wf, cx, cy = out
        assert not hasattr(out, 'slopes_x')
        coeffs, _labels = la.zernike_decompose(
            wf, 1e-4, wf.shape[0] * 1e-4, n_modes=6)
        assert np.shape(coeffs) == (6,)

    def test_lg_dock_uses_aberration_summary_and_binds(self):
        path = os.path.join(UI_DIR, 'lg_aberration_dock.py')
        _assert_binds(la.aberration_summary, path, 'aberration_summary')
        assert not _calls_named(path, 'aberration_tensor'), (
            'lg_aberration_dock calls aberration_tensor directly again; it '
            'takes a CanonicalPolyFit + s2_image, not a prescription.')

    def test_aberration_tensor_never_took_the_docks_kwargs(self):
        params = inspect.signature(la.aberration_tensor).parameters
        for gone in ('wavelength', 'w0', 'p_max', 'l_max'):
            assert gone not in params
        assert list(params)[:2] == ['fit', 's2_image']

    def test_lg_dock_renders_the_L_matrix(self):
        """``AberrationTensorResult`` carries the matrix on ``.L``; the
        pre-fix ``getattr(T, 'tensor', T)`` fell through to the dataclass."""
        from lumenairy.propagators.asymptotic import AberrationTensorResult
        assert not hasattr(AberrationTensorResult, 'tensor')
        assert 'L' in AberrationTensorResult.__dataclass_fields__
        with open(os.path.join(UI_DIR, 'lg_aberration_dock.py'),
                  encoding='utf-8') as fh:
            src = fh.read()
        assert "getattr(T, 'L'" in src


# ===========================================================================
# Finding 4 -- ToleranceAwareMerit(inner_merit=) aborted the run
# ===========================================================================

class TestOptimizerDockToleranceMerit:

    def test_optimizer_dock_call_binds(self):
        from lumenairy.optimize.wrapper_merits import ToleranceAwareMerit
        path = os.path.join(UI_DIR, 'optimizer_dock.py')
        _assert_binds(ToleranceAwareMerit.__init__, path,
                      'ToleranceAwareMerit', skip_self=True)

    def test_inner_merit_is_not_a_parameter(self):
        from lumenairy.optimize.wrapper_merits import ToleranceAwareMerit
        params = inspect.signature(ToleranceAwareMerit.__init__).parameters
        assert 'inner_merit' not in params
        assert 'sub_merit' in params
        # ...and perturbation_spec is REQUIRED, so omitting it (pre-fix)
        # could not have worked either.
        assert params['perturbation_spec'].default is inspect._empty
        for gone in ('radius_sigma_frac', 'thickness_sigma'):
            assert gone not in params

    def test_dock_passes_a_usable_perturbation_spec(self):
        """The spec the dock builds must be the shape the merit consumes
        (``surface_index`` is read unguarded by ``evaluate``)."""
        from lumenairy.optimize.merit_terms import FocalLengthMerit
        from lumenairy.optimize.wrapper_merits import ToleranceAwareMerit
        spec = [{'surface_index': i, 'decenter_std': 10e-6,
                 'tilt_std': 1e-4, 'form_error_rms': 0.0} for i in range(2)]
        m = ToleranceAwareMerit(sub_merit=FocalLengthMerit(target=0.05),
                                perturbation_spec=spec, n_trials=2, seed=1)
        assert m.n_trials == 2 and len(m.perturbation_spec) == 2
        assert all('surface_index' in s for s in m.perturbation_spec)


# ===========================================================================
# Finding 5 -- ui/surface_table.py deleted
# ===========================================================================

class TestSurfaceTableDeleted:

    def test_file_is_gone(self):
        assert not os.path.exists(os.path.join(UI_DIR, 'surface_table.py'))

    def test_no_references_remain_in_code_or_tests(self):
        pat = re.compile(r'surface_table|SurfaceTable')
        hits = []
        for root in (LUM_DIR, os.path.join(REPO_ROOT, 'tests')):
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames
                               if not d.startswith(('__', '.'))]
                for fn in filenames:
                    if not fn.endswith('.py'):
                        continue
                    p = os.path.join(dirpath, fn)
                    if os.path.abspath(p) == os.path.abspath(__file__):
                        continue
                    with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                        for i, line in enumerate(fh, 1):
                            if pat.search(line):
                                hits.append(
                                    f'{os.path.relpath(p, REPO_ROOT)}:{i}')
        assert not hits, ('dangling surface_table reference(s): '
                          + ', '.join(hits))


# ===========================================================================
# Finding 7 -- the deprecation removal-schedule registry
# ===========================================================================

_REMOVAL_RE = re.compile(r'will be (?:removed|required) in v([0-9][0-9.]*)')


def _module_level_str_constants(tree: ast.Module) -> dict[str, str]:
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = node.value.value
    return out


def _version_removed_sites() -> list[tuple[str, int, str]]:
    """Every ``version_removed=`` argument in ``lumenairy/`` resolved to a
    literal (constants defined at module level are followed)."""
    sites = []
    for dirpath, dirnames, filenames in os.walk(LUM_DIR):
        dirnames[:] = [d for d in dirnames if not d.startswith(('__', '.'))]
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(dirpath, fn)
            tree = _parse(path)
            consts = _module_level_str_constants(tree)
            rel = os.path.relpath(path, REPO_ROOT)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for kw in node.keywords:
                    if kw.arg != 'version_removed':
                        continue
                    val = kw.value
                    if isinstance(val, ast.Constant) and \
                            isinstance(val.value, str):
                        sites.append((rel, node.lineno, val.value))
                    elif isinstance(val, ast.Name) and val.id in consts:
                        sites.append((rel, node.lineno, consts[val.id]))
    return sites


class TestDeprecationRemovalSchedule:

    def test_registry_is_self_consistent(self):
        problems = dep.check_removal_schedule()
        assert problems == [], (
            'lumenairy._deprecation removal schedule is inconsistent:\n  '
            + '\n  '.join(problems))

    def test_next_removal_version_is_in_the_future(self):
        cur = dep._version_tuple(la.__version__)
        assert dep._version_tuple(dep.NEXT_REMOVAL_VERSION) > cur, (
            f'NEXT_REMOVAL_VERSION={dep.NEXT_REMOVAL_VERSION} has already '
            f'shipped (v{la.__version__}).')

    def test_shipped_horizons_resolve_forward(self):
        cur = dep._version_tuple(la.__version__)
        # The two horizons the audit measured as rotten.  v5.30 (W5) EXECUTED
        # the '5.27' removals and deleted the registry entry, so this now
        # exercises the ``resolve_removal_version`` BACKSTOP rather than an
        # explicit ``REMOVAL_SCHEDULE`` mapping -- which is the stronger
        # property: deleting an entry must not resurrect a past-horizon
        # banner for any site that still states the old version.
        for shipped in ('5.0', '5.27'):
            live = dep.resolve_removal_version(shipped)
            assert dep._version_tuple(live) > cur, (
                f'stated removal v{shipped} resolved to v{live}, which is '
                f'not after the running v{la.__version__}')
        # A future horizon is passed through untouched.
        assert dep.resolve_removal_version('6.0') == '6.0'

    def test_executed_removals_leave_no_registry_entry(self):
        """v5.30 (W5) removal-bookkeeping invariant.

        ``check_removal_schedule`` requires every ``REMOVAL_SCHEDULE``
        VALUE to lie in the future (invariant 2), so an entry for a
        completed removal can never be satisfied -- it would turn the
        self-check permanently red.  The registry's convention is
        therefore to DELETE the entry and tombstone it in a comment; the
        backstop above keeps the banner safe either way."""
        assert dep.REMOVAL_SCHEDULE == {}, (
            f'REMOVAL_SCHEDULE should be empty after the v5.30 W5 wave '
            f'executed its only entry; got {dep.REMOVAL_SCHEDULE!r}')
        assert dep.check_removal_schedule() == []

    def test_the_module_itself_stays_fully_functional(self):
        """Removing the shims must not gut the registry: the next
        deprecation cycle registers here exactly as before, and the P5
        return-contract TRANSITION is still scheduled (explicitly out of
        the W5 removal scope -- it flips a default, it deletes nothing)."""
        cur = dep._version_tuple(la.__version__)
        assert dep._version_tuple(dep.API_TRANSITION_VERSION) > cur
        assert dep.API_TRANSITION_VERSION == dep.NEXT_REMOVAL_VERSION
        for name in ('warn_deprecated_kwarg', 'warn_deprecated_alias',
                     'warn_renamed_function', 'warn_deprecated_default',
                     'warn_deprecated_signature', 'deprecated_alias'):
            assert callable(getattr(dep, name)), name
        # A fresh registry entry still resolves (simulated, not written).
        dep.REMOVAL_SCHEDULE['5.1'] = dep.NEXT_REMOVAL_VERSION
        try:
            assert dep.check_removal_schedule() == []
            assert dep.resolve_removal_version('5.1') == \
                dep.NEXT_REMOVAL_VERSION
        finally:
            del dep.REMOVAL_SCHEDULE['5.1']
        assert dep.resolve_removal_version(None) is None

    def test_no_call_site_advertises_a_shipped_removal_version(self):
        """The registry-level pin the audit asked for: no reachable
        ``version_removed=`` may resolve to a version <= __version__."""
        cur = dep._version_tuple(la.__version__)
        sites = _version_removed_sites()
        assert sites, 'no version_removed= sites found; the scan broke'
        bad = [f'{rel}:{line} version_removed={ver!r} -> '
               f'{dep.resolve_removal_version(ver)!r}'
               for rel, line, ver in sites
               if dep._version_tuple(
                   dep.resolve_removal_version(ver)) <= cur]
        assert not bad, ('deprecation banner(s) would advertise an '
                         'already-shipped removal version:\n  '
                         + '\n  '.join(bad))

    def test_emitted_banner_names_a_future_version(self):
        """End-to-end through a production path (not the helper).

        SUPERSEDED CARRIER: this pin used to drive the v5.25 ``sigma=`` ->
        ``w0=`` source shim, which is REMOVED in v5.30 (W5).  The
        PROPERTY under test is unchanged -- a banner emitted from the
        library must name a future version -- so the pin is re-pointed at
        a live production deprecation instead of being deleted:
        ``load_zmx_prescription``, the v4.7 Zemax-loader alias whose
        horizon was realigned to v6.0 in S4-17."""
        cur = dep._version_tuple(la.__version__)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                la.load_zmx_prescription('does-not-exist.zmx')
            except Exception:
                pass          # the warning fires before any file access
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, DeprecationWarning)]
        assert msgs, 'no live production deprecation banner fires at all'
        named = _REMOVAL_RE.findall(msgs[0])
        assert named, f'no removed-in clause in {msgs[0]!r}'
        assert dep._version_tuple(named[0]) > cur, (
            f'banner advertises v{named[0]} from a v{la.__version__} '
            f'library: {msgs[0]!r}')

    def test_the_removed_sigma_shim_emits_nothing_at_all(self):
        """Counter-pin to the supersession above: the carrier this test
        used to drive is gone, so it must now raise rather than warn."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(TypeError, match='sigma'):
                la.create_gaussian_beam(N=16, dx=1e-6, wavelength=550e-9,
                                        sigma=3e-6)
        assert not [w for w in caught
                    if issubclass(w.category, DeprecationWarning)], (
            [str(w.message) for w in caught])

    def test_rescheduled_banner_keeps_the_original_horizon_visible(self):
        """A slip is reported as a slip -- the message names the live
        version AND the one originally promised, so the goalpost is not
        moved silently (and the v5.25 pins asserting '5.27' still read
        the number they were written against)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            dep.warn_deprecated_kwarg('seed', 'rng', function='probe',
                                      version_added='5.25',
                                      version_removed='5.27')
        msg = str(caught[0].message)
        assert 'will be removed in v5.36' in msg
        assert 'rescheduled from v5.27' in msg

    def test_all_four_message_builders_route_through_the_registry(self):
        """The rot survived because four independent f-strings each
        interpolated ``version_removed`` verbatim."""
        cur = dep._version_tuple(la.__version__)
        probes = (
            lambda: dep.warn_deprecated_kwarg(
                'a', 'b', function='f', version_removed='5.0'),
            lambda: dep.warn_deprecated_alias(
                'a', 'b', version_removed='5.0'),
            lambda: dep.warn_deprecated_default(
                'a', 1.0, function='f', version_removed='5.0'),
            lambda: dep.warn_deprecated_signature(
                function='f', old_signature='f(a)', new_signature='f(*, a)',
                version_removed='5.0'),
        )
        for probe in probes:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                probe()
            msg = str(caught[0].message)
            named = _REMOVAL_RE.findall(msg)
            assert named, f'no removed-in clause in {msg!r}'
            assert dep._version_tuple(named[0]) > cur, msg

    def test_no_horizon_is_stated_when_none_is_known(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            dep.warn_deprecated_kwarg('a', 'b', function='f')
        assert 'will be removed' not in str(caught[0].message)

    def test_version_tuple_parsing(self):
        assert dep._version_tuple('5.29.0') == (5, 29, 0)
        assert dep._version_tuple('v5.27') == (5, 27, 0)
        assert dep._version_tuple('5.30.0rc1') == (5, 30, 0)
        assert dep._version_tuple('5.27') < dep._version_tuple('5.29.0')
        assert dep._version_tuple('5.32') > dep._version_tuple('5.29.0')

    def test_current_version_is_readable_without_a_circular_import(self):
        assert dep._current_version() == la.__version__


# ===========================================================================
# Finding 8 -- _lens_jax.py documented a parameter that does not exist
# ===========================================================================

_PARAM_LINE = re.compile(r'^\s{4}(\*{0,2}\w+)\s*:\s')


class TestLensJaxDocstringParams:
    """Docstring-only defect (the E-C1/E-C2 class): a documented parameter
    name that the signature does not accept.  Parsed with ``ast`` so the
    pin needs neither JAX nor an import."""

    def test_every_documented_param_exists(self):
        path = os.path.join(LUM_DIR, 'elements', '_lens_jax.py')
        tree = _parse(path)
        bad = []
        seen = 0
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            # clean=False: keep the original indentation the 4-space
            # parameter-line regex below matches on (cleandoc would dedent
            # it and make this scan silently vacuous).
            doc = ast.get_docstring(node, clean=False)
            if not doc or 'Parameters' not in doc:
                continue
            real = {a.arg for a in node.args.args + node.args.kwonlyargs}
            if node.args.vararg:
                real.add(node.args.vararg.arg)
            if node.args.kwarg:
                real.add(node.args.kwarg.arg)
            block = doc.split('Parameters', 1)[1]
            for section in ('Returns', 'Notes', 'Examples', 'References',
                            'Raises'):
                block = block.split(section, 1)[0]
            for line in block.splitlines():
                m = _PARAM_LINE.match(line)
                if not m:
                    continue
                for name in m.group(1).lstrip('*').split(','):
                    name = name.strip()
                    if not name:
                        continue
                    seen += 1
                    if name not in real:
                        bad.append(f'{node.name}(): documents {name!r}, '
                                   f'signature has {sorted(real)}')
        # Anti-vacuity floor: ``apply_real_lens_traced_jax`` alone
        # documents 10 parameters (the only numpydoc Parameters block in
        # the module -- its Maslov sibling defers to it in prose).
        assert seen >= 10, (f'the parameter-line scan matched only {seen} '
                            f'documented parameters; it has gone vacuous')
        assert not bad, ('_lens_jax.py documents parameter(s) that do not '
                         'exist:\n  ' + '\n  '.join(bad))

    def test_traced_jax_docstring_names_prescription(self):
        path = os.path.join(LUM_DIR, 'elements', '_lens_jax.py')
        for node in _parse(path).body:
            if isinstance(node, ast.FunctionDef) and \
                    node.name == 'apply_real_lens_traced_jax':
                doc = ast.get_docstring(node, clean=False) or ''
                assert '\n    prescription : dict' in doc
                assert '\n    lens_prescription : ' not in doc
                return
        pytest.fail('apply_real_lens_traced_jax not found')


# ===========================================================================
# Finding 9 -- the Schell return_kind shim cannot fire from production
# ===========================================================================

class TestSchellReturnKindShimReachability:
    """W3 MEASURED that the helper had zero production call sites -- the
    factories default ``return_kind='ensemble'`` outright since v4.16.1,
    so only an explicit ``return_kind=_RETURN_KIND_UNSET`` reached the
    sentinel branch, and even that branch did not warn.

    v5.30 (W5 shim-removal wave) acted on that measurement: the helper,
    the ``_RETURN_KIND_UNSET`` singleton, the
    ``_SchellReturnKindUnsetSentinel`` subclass and all five no-op
    branches are REMOVED.  The zero-call-sites scan below is kept (it now
    passes trivially AND guards against reintroduction); the
    'banner is not stale when invoked directly' pin is superseded by an
    absence check."""

    def test_zero_production_call_sites(self):
        hits = []
        for dirpath, dirnames, filenames in os.walk(LUM_DIR):
            dirnames[:] = [d for d in dirnames if not d.startswith(('__', '.'))]
            for fn in filenames:
                if not fn.endswith('.py'):
                    continue
                path = os.path.join(dirpath, fn)
                for node in ast.walk(_parse(path)):
                    if isinstance(node, ast.Call) and \
                            getattr(node.func, 'id', None) == \
                            '_warn_schell_return_kind_default':
                        hits.append(
                            f'{os.path.relpath(path, REPO_ROOT)}:{node.lineno}')
        assert hits == [], (
            'the Schell return_kind shim now HAS production call sites: '
            + ', '.join(hits) + '.  That is a behaviour change (the default '
            'path warned again); re-read the v4.16.1 rationale in '
            'sources/core.py before accepting it.')

    def test_default_path_is_silent_and_returns_the_ensemble(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = la.create_gaussian_schell_source(
                N=16, dx=1e-6, wavelength=550e-9, w0=4e-6, sigma_g=2e-6,
                n_realizations=2)
        assert not [w for w in caught
                    if issubclass(w.category, DeprecationWarning)]
        assert isinstance(out, tuple) and np.shape(out[0]) == (2, 16, 16)

    def test_helper_is_removed(self):
        """SUPERSEDES ``test_helper_banner_is_not_stale_when_invoked
        _directly``.

        A banner that can only be reached by a caller reaching into a
        private helper, for a transition that completed in v4.15.1, has no
        stale-ness left to guard -- v5.30 (W5) deletes it.  The stale-banner
        property is still covered library-wide by
        ``test_no_call_site_advertises_a_shipped_removal_version``."""
        import lumenairy.sources.core as core
        for name in ('_warn_schell_return_kind_default',
                     '_RETURN_KIND_UNSET',
                     '_SchellReturnKindUnsetSentinel'):
            assert not hasattr(core, name), name

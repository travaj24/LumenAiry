"""Lens-family architecture items (audit 2026-09-11, WP-A16 addenda 8-14).

Five mechanical changes, each with a property that can regress silently:

1. **One optional-dependency helper, five consumers.**  ``_lens_real``,
   ``_lens_traced``, ``lenses``, ``_lens_imap`` and ``propagators.fga`` had
   hand-copied ``_ensure_cupy_loaded`` / ``_is_cupy_array`` / ``_load_numba``
   (TESTS-ARCH P2-9: 5 copies of each).  They now delegate to
   ``lumenairy.backend._optional``.  Three things must survive the dedupe and
   are easy to lose: the module-level ``cp`` alias (the GPU branches read the
   NAME), ``_NUMBA_AVAILABLE`` as a module ATTRIBUTE (several tests
   monkeypatch it to reach the NumPy arm on a numba box), and ``fga``'s
   deliberate ``ImportError`` (it is the one consumer with no NumPy fallback).

2. **Three lens knobs in the registry.**  ``lens_sag_dtype``,
   ``lens_parallel_amp`` and ``pointwise_cos_grid_cache_budget`` join
   ``lumenairy._knobs`` so ``override()`` and the suite's autouse
   snapshot/restore fixture reach them (TESTS-ARCH P2-5: 53 ``set_`` verbs, 0
   context-manager forms, 0 resets, in a serially-run suite).

3. **``lumenairy.backend`` loads ``backend.scipy`` lazily** (PEP 562) and
   ``_lens_traced_multibranch`` reaches ``scipy.special.airy`` through a cached
   accessor -- the two module-level ``scipy`` importers the import-time item
   named.  ``la.backend.scipy.X`` must keep working, and ``ludwig_fold`` must
   still produce the Airy-uniform field.

4. **A narrowed except** in ``_lens_imap.build_inverse_map``.

5. **No F541** in ``_lens_real``'s mirror-guard warning -- and the warning text
   must still carry the placeholders it is supposed to.

FAIL-BEFORE.  Each test below was confirmed to fail on the pre-change tree by
the mechanism named in its docstring (the pre-change spelling is asserted
against directly where the old code is still reachable, and by in-process
monkeypatch where it is not); the WP-A16 report records the transcripts.

No wall-clock is asserted anywhere -- the import-time item's before/after is a
MEASUREMENT in the report, not a bar (TESTING_STANDARDS S1).
"""
from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import lumenairy as la
from lumenairy import _knobs
from lumenairy.backend import _optional
from lumenairy.elements import _lens_imap, _lens_real, _lens_traced, lenses
from lumenairy.elements import _lens_traced_multibranch as _mb
from lumenairy.propagators import fga

REPO = pathlib.Path(la.__file__).resolve().parent

#: The five modules addendum 8 names, and which helpers each must still expose.
_CUPY_CONSUMERS = (_lens_real, _lens_traced, lenses)
_NUMBA_CONSUMERS = (_lens_traced, lenses, _lens_imap)


# ---------------------------------------------------------------------------
# 1. Optional-dependency dedupe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('mod', _CUPY_CONSUMERS,
                         ids=[m.__name__.rsplit('.', 1)[-1]
                              for m in _CUPY_CONSUMERS])
def test_cupy_probe_is_the_shared_one_and_keeps_the_module_cp_alias(mod):
    """The dedupe must not cost the ``cp`` alias.

    ``xp = cp if _is_cupy_array(E) else np`` reads the module-level NAME, so a
    version that delegated the isinstance test but stopped binding ``cp``
    would raise ``NameError`` on the first GPU call -- on a box with CUDA,
    which CI does not have.  Pinning the coupling structurally is the only way
    to catch it here.
    """
    assert mod.CUPY_AVAILABLE is _optional.CUPY_AVAILABLE, (
        f'{mod.__name__}.CUPY_AVAILABLE is a second probe, not the shared '
        f'one; the accelerator-absent path is back to N implementations.')
    # ``is`` between two False bools is True on a CuPy-free box, so the value
    # check above cannot by itself tell a shared constant from a second
    # ``find_spec``.  This is the half that can: the module must IMPORT the
    # constant, not compute it.
    modsrc = pathlib.Path(mod.__file__).read_text(encoding='utf-8')
    assert "find_spec('cupy')" not in modsrc, (
        f'{mod.__name__} still probes for cupy itself; the probe belongs in '
        f'lumenairy.backend._optional (audit TESTS-ARCH P2-9).')
    # Structural, not textual: the name must arrive through an ImportFrom of
    # ``..backend._optional`` (two dots up from elements/).  Whether that
    # statement stands alone or is combined with the module's other
    # ``_optional`` imports is a formatter decision (``combine-as-imports``),
    # not a contract.
    imports_shared = any(
        isinstance(node, ast.ImportFrom)
        and node.module == 'backend._optional' and node.level == 2
        and any(alias.name == 'CUPY_AVAILABLE' for alias in node.names)
        for node in ast.walk(ast.parse(modsrc)))
    assert imports_shared, (
        f'{mod.__name__} does not import the shared CUPY_AVAILABLE from '
        f'..backend._optional.')
    assert hasattr(mod, 'cp'), (
        f'{mod.__name__} lost its module-level ``cp`` alias.')
    # the coupling: a True answer must imply ``cp`` is bound
    src = inspect.getsource(mod._is_cupy_array)
    assert '_ensure_cupy_loaded' in src, (
        f'{mod.__name__}._is_cupy_array no longer binds ``cp`` on a True '
        f'answer; the GPU branches read the module-level name.')
    # ...and the NumPy answer is correct and cheap (no import, no exception)
    assert mod._is_cupy_array(np.zeros(3)) is False
    assert mod._is_cupy_array(None) is False
    assert mod._is_cupy_array([1, 2, 3]) is False


def test_the_numpy_two_device_trap_is_live_so_the_isinstance_test_matters():
    """The defect every deleted copy's comment cited, re-measured.

    ``hasattr(x, 'device')`` was the old duck-type test.  NumPy 2.x gives
    every ndarray a ``.device``, so that test sends every NumPy array into the
    CuPy branch.  If this assertion ever fails the trap is gone and the
    isinstance requirement can be relaxed -- but until then it is load-bearing.
    """
    assert hasattr(np.zeros(3), 'device'), (
        'this NumPy has no ndarray.device, so the duck-type trap the shared '
        'helper exists to avoid is not live here; re-derive before relaxing '
        'anything.')


@pytest.mark.parametrize('mod', _NUMBA_CONSUMERS,
                         ids=[m.__name__.rsplit('.', 1)[-1]
                              for m in _NUMBA_CONSUMERS])
def test_numba_gate_stays_a_module_attribute_that_monkeypatch_reaches(
        mod, monkeypatch):
    """``_NUMBA_AVAILABLE`` must be read at CALL time from the module.

    Several existing tests set it to ``False`` to take the pure-NumPy arm on a
    box where numba IS installed.  A dedupe that read the shared constant
    directly would make that monkeypatch a no-op and silently stop testing the
    fallback.
    """
    assert mod._NUMBA_AVAILABLE == _optional.NUMBA_AVAILABLE
    monkeypatch.setattr(mod, '_NUMBA_AVAILABLE', False)
    monkeypatch.setattr(mod, '_numba', None)
    assert mod._load_numba() is False, (
        f'{mod.__name__}._load_numba() ignored a monkeypatched '
        f'_NUMBA_AVAILABLE=False, so every test that fakes an absent '
        f'accelerator on this module is now vacuous.')


def test_load_numba_returns_true_and_binds_the_handles_when_available():
    """Counter-pin to the monkeypatch test: the real answer must be real."""
    if not _optional.NUMBA_AVAILABLE:
        # Not a resource SKIP: the assertion below is simply the other half of
        # the contract, and it is the one that holds on a numba-free box.
        assert _optional.load_numba() is False
        assert _optional.numba_handles() == (None, None, None)
        return
    assert _optional.load_numba() is True
    nb, njit, prange = _optional.numba_handles()
    assert nb is not None and njit is not None and prange is not None
    for mod in _NUMBA_CONSUMERS:
        importlib.reload  # (not reloading: just asserting the live state)
        assert mod._load_numba() is True
        assert mod._njit is njit, (
            f'{mod.__name__} holds a DIFFERENT njit than the shared loader; '
            f'the dedupe did not actually dedupe.')


def test_fga_keeps_its_deliberate_importerror_for_the_missing_accelerator(
        monkeypatch):
    """``fga`` is the one consumer with no NumPy fallback.

    CONVENTIONS.md section 10 says absence is a VALUE for every helper that
    has a fallback and an ``ImportError`` with the install hint for an entry
    point that does not.  The dedupe must not convert the raise into a silent
    ``None``.
    """
    monkeypatch.setattr(fga, '_NUMBA', False)
    monkeypatch.setattr(fga, '_KERNELS', None)
    with pytest.raises(ImportError) as exc:
        fga._kernels()
    msg = str(exc.value)
    assert 'apply_real_lens_fga' in msg
    assert 'pip install lumenairy[numba]' in msg


def test_no_module_still_defines_its_own_copy_of_the_three_helpers():
    """The census that keeps the dedupe from silently regrowing.

    An AST walk for a module-level ``def`` of one of the three helper names
    whose BODY contains ``import cupy`` / ``import numba`` -- i.e. a real
    second implementation, not the thin wrapper that keeps the ``cp`` alias.
    """
    offenders = []
    for py in sorted(REPO.rglob('*.py')):
        if 'backend' in py.parts and py.name == '_optional.py':
            continue
        try:
            tree = ast.parse(py.read_text(encoding='utf-8'), filename=str(py))
        except SyntaxError:                       # pragma: no cover
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name not in ('_ensure_cupy_loaded', '_is_cupy_array',
                                 '_load_numba'):
                continue
            body = ast.dump(node)
            for pkg in ('cupy', 'numba'):
                if (f"names=[alias(name='{pkg}'" in body
                        or f"module='{pkg}'" in body):
                    offenders.append(
                        f'{py.relative_to(REPO.parent)}:{node.lineno} '
                        f'{node.name} imports {pkg} itself')
    assert not offenders, (
        'these functions re-implement the shared optional-dependency probe '
        'instead of delegating to lumenairy.backend._optional:\n  '
        + '\n  '.join(offenders))


# ---------------------------------------------------------------------------
# 2. The three lens knobs
# ---------------------------------------------------------------------------

_LENS_KNOBS = ('lens_sag_dtype', 'lens_parallel_amp',
               'pointwise_cos_grid_cache_budget')


def test_the_three_lens_knobs_are_registered():
    missing = [k for k in _LENS_KNOBS if k not in _knobs.knobs()]
    assert not missing, (
        f'{missing} are process-global lens knobs that override() and the '
        f'suite restore fixture cannot reach.  Add register_knob(...) beside '
        f'each set_*/get_* pair.')
    for k in _LENS_KNOBS:
        doc = _knobs.knob_doc(k)
        assert isinstance(doc, str) and len(doc) > 40, (
            f'knob {k!r} carries no usable doc: {doc!r}')


@pytest.mark.parametrize('knob,value', [
    ('lens_sag_dtype', np.float32),
    ('lens_parallel_amp', False),
    ('pointwise_cos_grid_cache_budget', 64 * 1024 * 1024),
])
def test_override_applies_and_restores_each_lens_knob(knob, value):
    """The property the whole registry exists for: a knob set inside a
    ``with`` is back to its previous value on exit, including on the
    exception path."""
    getter = _knobs._REGISTRY[knob].getter
    before = getter()
    assert not _knobs._same(before, value), (
        f'{knob} is already {value!r}, so this test cannot tell an applied '
        f'override from a no-op.  Another test leaked it, or the default '
        f'changed.')
    with la.override(**{knob: value}):
        assert _knobs._same(getter(), value), (
            f'override({knob}={value!r}) did not apply.')
    assert _knobs._same(getter(), before), (
        f'override({knob}=...) did not restore {before!r}.')

    # ...and on the exception path
    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with la.override(**{knob: value}):
            raise _Boom
    assert _knobs._same(getter(), before), (
        f'override({knob}=...) leaked {getter()!r} when the block raised.')


@pytest.mark.parametrize('knob', _LENS_KNOBS)
def test_each_lens_knob_round_trips_through_its_own_pair(knob):
    """``setter(getter())`` must be a no-op.

    ``pointwise_cos_grid_cache_budget`` is the reason this is not obvious:
    its PUBLIC setter takes megabytes while its public getter returns BYTES,
    so registering the public pair would have multiplied the budget by 2**20
    on every restore.  The registry uses a private byte/byte pair; this is the
    test that says why.
    """
    k = _knobs._REGISTRY[knob]
    before = k.getter()
    k.setter(k.getter())
    assert _knobs._same(k.getter(), before), (
        f'{knob}: setter(getter()) changed the value from {before!r} to '
        f'{k.getter()!r}.')


def test_the_megabyte_byte_asymmetry_that_forced_a_private_pair_is_real():
    """Fail-before evidence for the test above, re-measured every run.

    If the public pair ever becomes symmetric, the private accessors can be
    retired -- and this test is where that is noticed.
    """
    before = _lens_real.get_pointwise_cos_grid_cache_budget()
    try:
        _lens_real.set_pointwise_cos_grid_cache_budget(64)     # megabytes
        assert _lens_real.get_pointwise_cos_grid_cache_budget() == 64 * 1024 * 1024, (
            'the public getter no longer returns BYTES for a megabyte setter; '
            'the private knob pair in _lens_real may be retirable.')
    finally:
        _lens_real._set_pointwise_cos_grid_cache_budget_bytes(before)
    assert _lens_real.get_pointwise_cos_grid_cache_budget() == before


def test_set_low_memory_is_now_fully_covered_by_snapshot_restore():
    """``set_low_memory`` is a MACRO over four knobs and is deliberately not
    registered itself.  ``lens_parallel_amp`` was the last of the four not
    restorable, so the macro is now completely undone by restore()."""
    macro_knobs = ('fft_plan_cache_size', 'fft_double_buffer',
                   'fft_auto_promote', 'lens_parallel_amp',
                   'low_memory_prior')
    registered = set(_knobs.knobs())
    missing = [k for k in macro_knobs if k not in registered]
    assert not missing, (
        f'set_low_memory flips {missing}, which snapshot/restore cannot put '
        f'back -- a test that calls it leaks into every later test.')
    from lumenairy import memory as _memory
    # 5.47.1 (release run 34939783790, shard 6/8): restoring the four knobs
    # while leaving the macro's own first-enable stash in place made the
    # NEXT set_low_memory(True) keep the stale stash and the next disable
    # restore it (plan cache 8) over the caller's live values (16) -- in
    # whichever later test shared this process.  The stash is now the
    # registered knob 'low_memory_prior', so restore() clears it.
    state = _knobs.snapshot()
    assert state['low_memory_prior'] is None, (
        'an earlier test left set_low_memory(True) on record; the macro '
        'must be undone by set_low_memory(False) or _knobs.restore().')
    try:
        la.set_low_memory(True)
        assert _lens_traced.get_lens_parallel_amp() is False, (
            'set_low_memory(True) no longer flips lens_parallel_amp; this '
            'test is pinning the wrong macro.')
        assert _memory._LOW_MEMORY_PRIOR is not None, (
            'set_low_memory(True) no longer stashes its prior')
    finally:
        _knobs.restore(state)
    assert _lens_traced.get_lens_parallel_amp() is True
    assert _memory._LOW_MEMORY_PRIOR is None, (
        'restore() put the four knobs back but left the low-memory stash '
        'in place -- the next enable/disable cycle in this process would '
        'restore stale values')


# ---------------------------------------------------------------------------
# 3. Lazy scipy
# ---------------------------------------------------------------------------

def test_importing_lumenairy_does_not_import_backend_scipy_or_scipy_linalg():
    """The import-time item, asserted as a STRUCTURAL property (what is in
    ``sys.modules``), never as a time.

    A fresh interpreter is the only honest way to ask: this process has long
    since touched ``backend.scipy``.  ``scipy.linalg`` is the module the lazy
    forward actually saves -- ``scipy.special`` is still pulled in by
    ``scipy.fft``, which ``propagators/fft_infra.py`` imports eagerly and
    which is outside this work package (recorded in the WP-A16 report).
    """
    code = (
        'import sys; sys.path.insert(0, %r)\n'
        'import lumenairy\n'
        'print("backend.scipy", "lumenairy.backend.scipy" in sys.modules)\n'
        'print("scipy.linalg", "scipy.linalg" in sys.modules)\n'
        % str(REPO.parent))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, stdin=subprocess.DEVNULL,
                         cwd=str(REPO.parent))
    assert out.returncode == 0, out.stderr[-2000:]
    lines = dict(ln.rsplit(' ', 1) for ln in out.stdout.strip().splitlines())
    assert lines['backend.scipy'] == 'False', (
        'import lumenairy still loads lumenairy.backend.scipy eagerly; the '
        'PEP 562 forward in backend/__init__.py is not doing its job.')
    assert lines['scipy.linalg'] == 'False', (
        'import lumenairy still loads scipy.linalg; something imports '
        'lumenairy.backend.scipy (or scipy.linalg) at module scope again.')


def test_backend_scipy_still_resolves_every_way_it_used_to():
    mod = la.backend.scipy
    assert mod.__name__ == 'lumenairy.backend.scipy'
    from lumenairy.backend import scipy as by_from_import
    assert by_from_import is mod
    assert importlib.import_module('lumenairy.backend.scipy') is mod
    assert 'scipy' in dir(la.backend)
    assert 'scipy' in la.backend.__all__
    # caching: the second access is a plain globals() hit
    assert la.backend.__dict__['scipy'] is mod


def test_backend_getattr_raises_attributeerror_not_importerror():
    with pytest.raises(AttributeError):
        la.backend.definitely_not_a_submodule
    assert getattr(la.backend, 'nope', 'fallback') == 'fallback'
    assert hasattr(la.backend, 'scipy')


def test_the_airy_accessor_binds_once_and_gives_scipys_answer():
    """``ludwig_fold`` must still be the Ludwig uniform-Airy field.

    Bar: exact equality against ``scipy.special.airy`` itself.  There is no
    numerical content in a rebinding, so a tolerance would only hide a wrong
    binding (say, ``airye``).
    """
    from scipy.special import airy as reference
    z = np.linspace(-4.0, 2.0, 17)
    got = _mb._airy(z)
    want = reference(z)
    for a, b in zip(got, want):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert _mb._scipy_airy is not None, (
        'the accessor did not cache the handle, so every fold pays a '
        'sys.modules lookup.')
    assert _mb._scipy_airy is reference


def test_ludwig_fold_stays_bounded_where_the_branch_sum_diverges():
    """End-to-end counter-pin on the accessor, against a CLOSED-FORM oracle.

    Signature: ``ludwig_fold(k, S_plus, S_minus, A_plus, A_minus)``.  Approach
    the fold with the PHYSICAL branch pair -- amplitudes that diverge as
    ``rho**-1/4`` and carry the KMAH ``-pi/2`` on the branch that has touched
    the caustic, i.e. ``A+ = -i A-`` -- so that ``g1 = (A+ + iA-)/(rho**1/4
    sqrt2)`` vanishes and ``g0 = rho**1/4/sqrt2 (A+ - iA-) -> -i sqrt2 A0``.
    The Ludwig field then has the exact limit

        u(0) = sqrt(2 pi) k**(1/6) exp(i pi/4) exp(i k S0) (-i sqrt2 A0) Ai(0)

    with ``Ai(0) = 3**(-2/3)/Gamma(2/3) = 0.3550280538878172`` -- a number from
    the Airy function's own series, not from this library.

    BARS, derived (k = 2 pi / 632.8 nm, S0 = 1 mm, A0 = 1, so |u(0)| =
    18.4509969...).  MEASURED 2026-09-12 at half-separation delta = 1e-12 m:
    |u| = 18.4591 (4.4e-4 relative to the limit) while the plain branch sum is
    132.2 and climbing as delta**(-1/6).  The bars are 1e-2 on the relative
    distance to the closed form (a 23x margin over the measured 4.4e-4, and
    the residual is the genuine O(k**(2/3) rho) curvature of Ai, not noise)
    and 4x on the plain-sum-to-|u| ratio (measured 7.16, and the ratio grows
    without bound as delta -> 0, so the gap to a real regression -- a Ludwig
    form that tracked the divergent sum -- is unbounded above).
    """
    k = 2.0 * np.pi / 632.8e-9
    s0, a0 = 1.0e-3, 1.0 + 0.0j
    ai0 = 0.3550280538878172          # Ai(0), DLMF 9.2.3
    limit = (np.sqrt(2.0 * np.pi) * k ** (1.0 / 6.0) * np.exp(1j * np.pi / 4.0)
             * np.exp(1j * k * s0) * (-1j * np.sqrt(2.0) * a0) * ai0)

    delta = np.full(8, 1.0e-12)
    rho = (1.5 * delta) ** (2.0 / 3.0)
    a_minus = a0 * rho ** -0.25
    a_plus = -1j * a_minus
    out = _mb.ludwig_fold(k, s0 + delta, s0 - delta, a_plus, a_minus)
    plain = (a_plus * np.exp(1j * k * (s0 + delta))
             + a_minus * np.exp(1j * k * (s0 - delta)))

    assert np.all(np.isfinite(out)), (
        f'ludwig_fold returned {int(np.count_nonzero(~np.isfinite(out)))} '
        f'non-finite samples next to the fold -- the uniform replacement is '
        f'the whole point of the function.')
    rel = float(np.max(np.abs(out - limit)) / np.abs(limit))
    assert rel < 1.0e-2, (
        f'ludwig_fold is {rel:.3e} away from the closed-form on-fold limit '
        f'{limit!r}; measured 4.4e-4 on 2026-09-12.  A wrong Airy binding '
        f'(airye, or Bi for Ai) lands decades from here.')
    ratio = float(np.max(np.abs(plain)) / np.abs(limit))
    assert ratio > 4.0, (
        f'the plain two-branch sum is only {ratio:.2f}x the uniform field at '
        f'delta = 1e-12 m, so this fixture is not actually near a fold and '
        f'the test cannot tell a uniform field from the divergent sum.')


def test_ludwig_fold_reduces_to_the_plain_branch_sum_far_from_the_fold():
    """The other half of the uniform form, and a second independent oracle for
    the Airy accessor.

    Well outside the Kravtsov-Orlov band the Airy asymptotics turn the Ludwig
    expression back into ``A+ exp(i k S+) + A- exp(i k S-)`` exactly -- the
    library's own docstring says so, and the plain sum is an oracle this file
    computes itself.  MEASURED 2026-09-12 with the physical (KMAH) branch pair:
    relative distance 1.11e-3 at 20 wavelengths of eikonal separation and
    4.42e-5 at 500, i.e. it falls like the expected O(1/(k dS)) correction.
    BAR 5e-3 at 20 wavelengths: 4.5x the measured value, and three decades
    above float64 noise (~1e-15), while a mis-bound special function
    (``airye`` differs by exp(2/3 z**3/2), ``Bi`` by a growing exponential)
    lands at O(1) or larger.
    """
    k = 2.0 * np.pi / 632.8e-9
    s0, a0 = 1.0e-3, 1.0 + 0.0j
    for n_wave, bar in ((20.0, 5.0e-3), (500.0, 2.0e-4)):
        d_s = n_wave * 632.8e-9
        rho = (0.75 * d_s) ** (2.0 / 3.0)
        a_minus = a0 * rho ** -0.25
        a_plus = -1j * a_minus
        s_plus, s_minus = s0 + d_s / 2.0, s0 - d_s / 2.0
        uni = _mb.ludwig_fold(k, s_plus, s_minus, a_plus, a_minus)
        plain = (a_plus * np.exp(1j * k * s_plus)
                 + a_minus * np.exp(1j * k * s_minus))
        rel = float(abs(uni - plain) / abs(plain))
        assert rel < bar, (
            f'{n_wave:.0f} wavelengths past the fold the uniform field is '
            f'{rel:.3e} from the plain branch sum (bar {bar:.0e}); the Airy '
            f'accessor or the uniform form is wrong.')


# ---------------------------------------------------------------------------
# 4/5. The narrowed except and the F541
# ---------------------------------------------------------------------------

def test_build_inverse_map_ram_probe_catches_only_importerror():
    """WP-A15a section 5.7.  A broad clause here would swallow a real error
    raised while ``lumenairy.memory`` was being imported for the first time.
    """
    src = inspect.getsource(_lens_imap)
    idx = src.index('from .. import memory as _mem')
    window = src[idx:idx + 400]
    assert 'except ImportError:' in window, (
        'the RAM-budget probe in build_inverse_map is not narrowed to '
        'ImportError:\n' + window[:300])
    assert 'except Exception' not in window, window[:300]


def test_no_broad_except_remains_at_that_site():
    """Structural counter-pin: the whole module's ``except Exception`` census
    is down to the two the WP-A15a report justifies (``_glass_key_value`` and
    ``_sag_callable_fingerprint`` live in ``_lens_real``, not here), so
    ``_lens_imap`` should now carry none."""
    tree = ast.parse(pathlib.Path(_lens_imap.__file__).read_text(
        encoding='utf-8'))
    broad = [h.lineno for node in ast.walk(tree)
             if isinstance(node, ast.Try)
             for h in node.handlers
             if (h.type is None
                 or (isinstance(h.type, ast.Name) and h.type.id == 'Exception'))]
    assert not broad, (
        f'_lens_imap.py carries broad except clauses at lines {broad}; the '
        f'per-file census in test_audit_except_budget.py allows zero.')


def test_the_mirror_guard_warning_has_no_empty_f_strings_and_keeps_its_values():
    """F541 in ``_lens_real``'s unfolded-equivalent warning.

    Dropping an ``f`` prefix is only safe if that literal had no placeholders,
    so this asserts BOTH halves: that the two literals ruff flagged are no
    longer f-prefixed, and that the warning still interpolates the surface
    indices it is about.

    Checked on the EXACT pair ruff named rather than by a line heuristic:
    F541 is a property of the whole implicitly-concatenated literal, not of
    a line, so the two ``f"  DROPPED: ..."`` parts a few lines above are
    legal (each is concatenated with a part that carries ``{curved}`` /
    ``{sorted(set(shifted))}``) while the closing pair -- which has no
    placeholder in either part -- is not.  RE-MEASURED on the committed HEAD
    blob: ``ruff --isolated --select F541`` reports exactly 2 there and 0
    here.
    """
    src = inspect.getsource(_lens_real._unfold_mirror_surfaces)
    for literal in (
            '"  Use lumenairy.io.split_prescription_at_mirrors(rx) with "',
            '"apply_mirror at each fold to carry them."'):
        assert literal in src, (
            f'the mirror-guard warning no longer contains {literal!r}; this '
            f'test is pinning the wrong text.')
        assert ('f' + literal) not in src, (
            f'{literal!r} is f-prefixed again and carries no placeholder '
            f'(ruff F541).  Once both are plain the per-file ignore for '
            f'lumenairy/elements/_lens_real.py in pyproject.toml should be '
            f'DELETED -- see the WP-A16 report.')

    # ...and the warning still says which surfaces it dropped
    rx = {'surfaces': [
        dict(radius=0.05, glass_before='AIR', glass_after='N-BK7'),
        dict(radius=-0.2, glass_before='N-BK7', glass_after='MIRROR',
             is_mirror=True),
    ], 'thicknesses': [3.0e-3], 'aperture_diameter': 4.0e-4,
        'allow_unfolded_equivalent': True}
    with pytest.warns(RuntimeWarning) as rec:
        _lens_real._unfold_mirror_surfaces(rx, 'apply_real_lens')
    msg = str(rec[0].message)
    assert msg.startswith('apply_real_lens:'), msg[:120]
    assert 'UNFOLDED EQUIVALENT' in msg
    assert 'CURVED mirror(s) at [1]' in msg, (
        'the curved-mirror clause lost its interpolated index list; '
        'dropping the f prefix took a placeholder with it.  Message was: '
        + msg)
    assert 'split_prescription_at_mirrors' in msg

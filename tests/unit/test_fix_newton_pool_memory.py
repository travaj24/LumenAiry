"""The Newton process pool must not out-spend the box.

FAIL-BEFORE (design-121 capstone, docs/audits/CAPSTONE_D121_2026_08_06.md sec 5
and this fix's own docs/audits/FIX_POOL_MEMORY_2026_08_06.md).
``_invert_newton_parallel`` split the wave grid into ``n_workers`` chunks and
submitted every one of them to a persistent ``ProcessPoolExecutor`` with NO
memory accounting anywhere on the path, while the fine grid those chunks come
from is sized by ``carrier._memory_bounded_n_fine`` against a SINGLE-PROCESS
cost model (4 x 16 B x n^2).  Measured on the design-121 acceptance at the
shipped ``n_workers=8``: 22.1 GB committed PER WORKER, ~177 GB across the pool,
system commit 205.7 / 227.5 GB, free physical RAM 0.0 GB, run killed at 9.7 min.
The serial peak for the identical physics was 20.86 GB.

The same library already fixed the identical bug for the OTHER pool --
``carrier._multi_resolve_workers``, whose comment records it -- and this file
guards the same treatment for the Newton pool.

TWO rules ship, because the measurement found TWO terms and only one of them is
chunk-shaped:

  * the INTRINSIC per-worker cost, measured DEAD linear at 267.2 B per Newton
    point on a 1.728 GB import intercept (4-point sweep, 0.1 % spread over an
    8x range) -- this is what the memory clamp prices; and
  * the term that actually killed design 121: a ``spawn`` worker RE-EXECUTES an
    unguarded ``__main__`` module body in full before serving its chunk, so each
    worker paid the caller's WHOLE acceptance chain (~20.9 GB) rather than its
    1.9 GB chunk.  No worker count makes that acceptable, so it returns SERIAL.

Everything here is arithmetic on mocked memory or on temp files: no test in this
file allocates the memory it reasons about.  The psutil pin follows the idiom in
``test_fga_h4_h5.py::test_c2_env_budget_override`` -- freeze a snapshot so the
assertions test the CONTRACT rather than racing a busy box.

The safety argument for clamping at all is that it cannot move a number:
``_newton_invert_chunk`` is documented and tested (``test_niche_newton_pool_
both_fits.py``) to be BIT-IDENTICAL to the serial closure, so worker count is a
speed knob only.  ``test_a_capped_pool_is_bit_identical_to_serial`` below re-
proves that specifically for a pool the CLAMP sized, which is the new path.
"""
import inspect
import os
import subprocess
import sys
import types
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

# Design 121's own Stage-B numbers, so the arithmetic below is about the
# workload that produced the failure rather than an invented one:
# 2048 / ray_subsample 4 -> 512^2 Newton points per group, and the biggest
# measured Chebyshev fit grid on that chain was 531^2 (spy over the real run).
_D121_POINTS = 262_144
_D121_FIT = 531 * 531
_D121_NW = 8


def _budget(free_b):
    """The shipped budget rule, spelled out here so the tests assert the
    ARITHMETIC and not merely the function's agreement with itself."""
    return (LT._NEWTON_POOL_RAM_FRAC * free_b
            - LT._NEWTON_POOL_MIN_FREE_GB * 1e9)


# ---------------------------------------------------------------------------
# (a) FAIL-BEFORE: scarce memory must cap the worker count
# ---------------------------------------------------------------------------

def test_the_pre_fix_path_would_have_submitted_every_chunk():
    """FAIL-BEFORE, as arithmetic on the shipped model.

    The pre-fix gate was ``n_cpu = n_workers if n_workers is not None else
    available_cpus()`` and nothing else -- there was no memory term anywhere on
    the path, so the requested count WAS the dispatched count at any free-RAM
    reading whatsoever.  At the scarcity mocked here that dispatch does not fit
    the box; the shipped resolver returns a count that does.
    """
    free_b = 12.0e9
    pre_fix = _D121_NW                      # the old gate, verbatim
    per_worker = LT._newton_worker_bytes(_D121_POINTS / pre_fix, _D121_FIT)
    assert pre_fix * per_worker > _budget(free_b), (
        'the mocked scarcity is not actually scarce for the pre-fix dispatch, '
        'so this test would pass for free')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = LT._newton_resolve_workers(_D121_NW, _D121_POINTS, _D121_FIT,
                                         _free_b=free_b)
    assert 1 <= got < pre_fix, (
        f'the resolver returned {got} workers where the pre-fix path submitted '
        f'{pre_fix}; the clamp did not bind')
    # ...and what it returned actually FITS, priced at the chunk size that
    # count implies (fewer workers = bigger chunks, so re-pricing is required).
    assert (got * LT._newton_worker_bytes(_D121_POINTS / got, _D121_FIT)
            <= _budget(free_b)), (
        'the capped count still over-spends the budget: the resolver priced '
        'the cap at the REQUESTED chunk size instead of the one it will run')


def test_the_cap_degrades_all_the_way_to_serial():
    """Graceful degradation, not a cliff: as memory tightens the count walks
    down and bottoms out at 1 (serial), which is the numerically identical
    path.  A resolver that returned 0 -- or that raised -- would turn a
    resource problem into a failed run."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        counts = [LT._newton_resolve_workers(_D121_NW, _D121_POINTS, _D121_FIT,
                                             _free_b=f)
                  for f in (200e9, 60e9, 30e9, 12e9, 6e9, 1e9, 0.0)]
    assert counts[0] == _D121_NW
    assert counts[-1] == 1
    assert all(1 <= c <= _D121_NW for c in counts)
    assert counts == sorted(counts, reverse=True), (
        f'worker count {counts} is not monotone in available memory')


def test_the_clamp_reads_live_available_memory(monkeypatch):
    """The clamp must consult the box AT DISPATCH TIME, not a constant.

    Pinned-snapshot idiom (test_fga_h4_h5.py::test_c2_env_budget_override): a
    live ``psutil.virtual_memory()`` read differs between two calls on a busy
    box, so freeze one and edit the single field under test."""
    import psutil
    vm = psutil.virtual_memory()
    monkeypatch.setattr(psutil, 'virtual_memory',
                        lambda: vm._replace(available=int(12e9)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scarce = LT._newton_resolve_workers(_D121_NW, _D121_POINTS, _D121_FIT)
    monkeypatch.setattr(psutil, 'virtual_memory',
                        lambda: vm._replace(available=int(400e9)))
    abundant = LT._newton_resolve_workers(_D121_NW, _D121_POINTS, _D121_FIT)
    assert scarce < abundant == _D121_NW, (
        f'the resolver read {scarce} workers at 12 GB and {abundant} at '
        f'400 GB; it is not reading live memory')


# ---------------------------------------------------------------------------
# (b) abundant memory must change nothing
# ---------------------------------------------------------------------------

def test_abundant_memory_leaves_the_requested_worker_count_alone():
    """The clamp is a ceiling, never a policy: on a box that fits the pool it
    must return exactly what the caller asked for, and say nothing.  A clamp
    that trimmed a worker on a healthy box would be a silent, permanent
    slowdown on the default path."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        got = LT._newton_resolve_workers(_D121_NW, _D121_POINTS, _D121_FIT,
                                         _free_b=400e9)
    assert got == _D121_NW
    assert [str(w.message) for w in rec] == [], (
        'the clamp warned on a box with 400 GB free; the warning must be '
        'reserved for a cap that actually binds')


def test_a_call_that_cannot_reach_the_pool_is_clamped_silently():
    """The clamp runs UPSTREAM of the size gate, so it also sees calls that
    gate is about to answer serially at any worker count.  Announcing a cap on
    a dispatch that will never happen is noise that trains a reader to ignore
    the warning.

    Observed for real on ``test_fga.py``'s 384-point dispatcher probe: a
    24-CPU default against a 1125^2 ray-fit grid gave a perfectly CORRECT
    24 -> 20 clamp (24 x 2.83 GB = 68 GB against a 56.8 GB budget) for a
    16-points-per-chunk call that then ran in-process anyway.  The count must
    still be clamped -- it feeds the gate -- but the warning must not fire."""
    tiny = LT._POOL_MIN_PIXELS_WARM - 1
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        got = LT._newton_resolve_workers(24, tiny, 1125 * 1125,
                                         min_pool_points=LT._POOL_MIN_PIXELS_WARM,
                                         _free_b=118e9)
    assert got < 24, 'the clamp must still bind; only the warning is suppressed'
    assert [w for w in rec if 'Newton process pool' in str(w.message)] == [], (
        'the clamp announced a cap on a call that cannot reach the pool at any '
        'worker count')
    # ...and the SAME call above the bar does announce
    with pytest.warns(RuntimeWarning, match='Newton process pool'):
        LT._newton_resolve_workers(24, LT._POOL_MIN_PIXELS_WARM, 1125 * 1125,
                                   min_pool_points=LT._POOL_MIN_PIXELS_WARM,
                                   _free_b=118e9)


def test_the_clamp_can_only_ever_lower_the_count():
    """Even with absurd headroom the resolver must not invent workers: the
    caller's ``n_workers`` (or ``available_cpus()``) is the ceiling, and this
    function's whole contract is that it is monotone downward."""
    for req in (2, 4, 8, 24):
        assert LT._newton_resolve_workers(req, 1024, 64,
                                          _free_b=1e15) == req
    assert LT._newton_resolve_workers(1, _D121_POINTS, _D121_FIT,
                                      _free_b=0.0) == 1
    assert LT._newton_resolve_workers(0, _D121_POINTS, _D121_FIT,
                                      _free_b=0.0) == 1


def test_a_missing_memory_oracle_keeps_the_historical_behaviour(monkeypatch):
    """No psutil -> no clamp, exactly as ``_multi_resolve_workers`` does.  A
    resource guard that cannot measure must not guess; guessing here would
    serialise every install without psutil."""
    import psutil

    def _boom():
        raise OSError('no memory oracle')

    monkeypatch.setattr(psutil, 'virtual_memory', _boom)
    assert LT._newton_resolve_workers(_D121_NW, _D121_POINTS,
                                      _D121_FIT) == _D121_NW


# ---------------------------------------------------------------------------
# (d) the warning: exactly once, and it names the numbers
# ---------------------------------------------------------------------------

def test_the_cap_warning_fires_once_and_names_the_numbers():
    """A memory-limited result must never be returned as if it were the
    requested configuration -- the same rule ``_memory_bounded_n_fine`` follows
    for the fine grid.  One warning per binding dispatch, carrying every number
    a reader needs to act: what was asked for, what one worker costs, what the
    box has, what the budget rule allowed, and what will actually run."""
    free_b = 12.0e9
    with pytest.warns(RuntimeWarning) as rec:
        got = LT._newton_resolve_workers(_D121_NW, _D121_POINTS, _D121_FIT,
                                         _free_b=free_b)
    msgs = [str(w.message) for w in rec
            if 'Newton process pool' in str(w.message)]
    assert len(msgs) == 1, (
        f'{len(msgs)} cap warnings for one binding dispatch; expected exactly '
        f'one')
    msg = msgs[0]
    per_worker = LT._newton_worker_bytes(_D121_POINTS / _D121_NW, _D121_FIT)
    for needle in (
            f'asked for {_D121_NW} workers',
            f'{per_worker / 1e9:.2f} GB per worker',
            f'{_D121_POINTS // _D121_NW} Newton points/chunk',
            f'{_D121_FIT}-point ray-fit grid',
            f'{free_b / 1e9:.1f} GB is available',
            f'{_budget(free_b) / 1e9:.1f} GB',
            f'running {got} worker(s) instead'):
        assert needle in msg, f'the cap warning does not name {needle!r}:\n{msg}'
    # ...and it must say the answer did not move, or a reader will assume a
    # memory-limited run is a DIFFERENT run.
    assert 'bit-identical to serial' in msg


# ---------------------------------------------------------------------------
# the measured model itself
# ---------------------------------------------------------------------------

def test_the_per_worker_model_reproduces_the_measurements():
    """Pin the SCALING LAW, not just its use.

    Measured (fresh interpreter per point, psutil ``peak_pagefile``; the full
    table is in docs/audits/FIX_POOL_MEMORY_2026_08_06.md).  The model is
    required to bound every measurement from ABOVE -- a resource clamp that
    under-predicts is not a clamp -- and to stay within 25 %, or it is not a
    model of anything.
    """
    measured = [  # (chunk points, fit points, measured peak commit bytes)
        (32_768, _D121_FIT, 1_915_604_992),
        (2_097_152, _D121_FIT, 2_288_332_800),
        (4_194_304, _D121_FIT, 2_849_161_216),
        (8_388_608, _D121_FIT, 3_969_617_920),
        (16_777_216, _D121_FIT, 6_210_768_896),
        (2_097_152, 1024 * 1024, 2_626_211_840),
        (2_097_152, 2048 * 2048, 5_274_189_824),
    ]
    for chunk, fit, meas in measured:
        model = LT._newton_worker_bytes(chunk, fit)
        assert model >= meas, (
            f'model {model / 1e9:.3f} GB UNDER-predicts the measured '
            f'{meas / 1e9:.3f} GB at chunk={chunk} fit={fit}')
        assert model <= 1.25 * meas, (
            f'model {model / 1e9:.3f} GB is more than 25 % above the measured '
            f'{meas / 1e9:.3f} GB at chunk={chunk} fit={fit}')


def test_the_chunk_slope_is_the_measured_one():
    """The 267.2 B/point slope is the measurement the clamp rests on (0.1 %
    spread across a 8x chunk range), so a refactor that turns it into a round
    allowance has to fail here."""
    d = (LT._newton_worker_bytes(2_000_000, 0)
         - LT._newton_worker_bytes(1_000_000, 0)) / 1_000_000
    assert 267.0 <= d <= 275.0, (
        f'per-point term is {d:.1f} B/pt; the measured law is 267.2 B/pt')
    assert 1.5e9 <= LT._NEWTON_WORKER_BASE_BYTES <= 2.5e9, (
        'the per-process intercept left the range the import measurements '
        'support (bare python 0.012 GB, numpy 0.831 GB, lumenairy 1.65 GB, '
        '+0.07 GB numba JIT)')


def test_the_budget_rule_matches_the_sibling_pool():
    """The two clamps that meet on the exact final leg must speak the same
    language: this pool's fraction is the fine grid's own
    ``_FINE_GRID_RAM_FRAC``, and the reserve idiom is
    ``_multi_resolve_workers``'.  Drifting them apart is how one guard starts
    approving what the other refuses."""
    from lumenairy.propagators import carrier as C
    assert LT._NEWTON_POOL_RAM_FRAC == C._FINE_GRID_RAM_FRAC
    assert 0.5 <= LT._NEWTON_POOL_MIN_FREE_GB <= 8.0


# ---------------------------------------------------------------------------
# the term the chunk model cannot see: spawn re-executing an unguarded __main__
# ---------------------------------------------------------------------------

# The two scripts differ in ONE thing: whether the "expensive" top-level work
# (here, one line appended to a log -- design 121's version of it was the whole
# acceptance chain) sits inside the guard.  The imports are identical and run in
# the child either way, which is exactly why the shipped rule is about the
# GUARD and not about the import.
#
# The nested-dispatch ``try/except`` mirrors the library: a spawn child that
# re-runs an unguarded body and then reaches its own pool dispatch raises
# ``RuntimeError('...bootstrapping phase...')``, which ``_invert_newton_
# parallel`` catches and answers serially.  Without the catch here the child
# dies during bootstrap and the parent sees ``BrokenProcessPool`` -- also the
# mechanism, but it would make this test about error handling.
_BODY = """
import os, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def _noop(x):
    return x


def _work():
    with open(sys.argv[1], 'a') as fh:
        fh.write('BODY %d %s\\n' % (os.getpid(), __name__))
    ex = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'))
    try:
        print(ex.submit(_noop, 1).result())
    except (RuntimeError, OSError) as exc:
        print('nested dispatch refused:', type(exc).__name__)
    ex.shutdown(wait=True)
"""

_UNGUARDED = _BODY + "\n_work()\n"
_GUARDED = _BODY + "\nif __name__ == '__main__':\n    _work()\n"


def _run_script(tmp_path, src, name):
    script = tmp_path / name
    script.write_text(src)
    log = tmp_path / (name + '.log')
    out = subprocess.run([sys.executable, str(script), str(log)],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr + out.stdout
    return [ln for ln in log.read_text().splitlines() if ln.startswith('BODY')]


@pytest.mark.slow
def test_a_spawn_worker_really_does_rerun_an_unguarded_main(tmp_path):
    """PREMISE, proved by running it -- the rule below guards a real mechanism,
    not a suspicion.

    ``multiprocessing.spawn._fixup_main_from_path`` does
    ``runpy.run_path(main_path, run_name='__mp_main__')`` in every spawned
    child, so a ``__main__`` with no guard executes its ENTIRE body once per
    worker.  That is why design 121's eight Newton workers each committed
    22.1 GB (the whole acceptance chain, re-run) instead of the 1.9 GB their
    chunk costs.  The guarded script is the control: identical file, identical
    imports, one line moved."""
    lines = _run_script(tmp_path, _UNGUARDED, 'unguarded.py')
    assert len(lines) == 2, (
        f'expected the top-level body to run twice (parent + one spawn '
        f'worker), got {lines}')
    assert '__mp_main__' in lines[1], lines
    ctrl = _run_script(tmp_path, _GUARDED, 'guarded.py')
    assert len(ctrl) == 1, (
        f'the guarded control ran its body {len(ctrl)} times; the guard is '
        f'the whole remedy the warning recommends, so if it does not work the '
        f'advice is wrong: {ctrl}')


def _fake_main(monkeypatch, path):
    mod = types.ModuleType('__main__')
    mod.__file__ = str(path)
    mod.__spec__ = None
    monkeypatch.setitem(sys.modules, '__main__', mod)
    LT._reset_newton_pool_resource_state()


def test_an_unguarded_main_forces_serial_and_says_why(tmp_path, monkeypatch):
    """FAIL-BEFORE for the term that actually killed the capstone run.

    Pre-fix this returned the caller's 8 and every worker re-ran the caller's
    program.  There is no smaller pool that fixes it -- the caller's side
    effects would still run K extra times -- so the rule returns SERIAL, and it
    has to name the file and the remedy or the user cannot act on it."""
    script = tmp_path / 'runner_no_guard.py'
    script.write_text('import numpy as np\nx = np.zeros(3)\nprint(x.sum())\n')
    _fake_main(monkeypatch, script)
    with pytest.warns(RuntimeWarning) as rec:
        got = LT._newton_resolve_workers(_D121_NW, _D121_POINTS, _D121_FIT,
                                         _free_b=1e15)
    assert got == 1, (
        f'{got} workers on an unguarded __main__ with infinite RAM; the rule '
        f'is not about memory headroom, it is about the child re-running the '
        f'caller')
    msgs = [str(w.message) for w in rec if '__main__' in str(w.message)]
    assert len(msgs) == 1
    for needle in (str(script), "if __name__ ==", 'RE-EXECUTE',
                   '22.1 GB', 'bit-identical'):
        assert needle in msgs[0], f'missing {needle!r} in:\n{msgs[0]}'


def test_the_unguarded_warning_fires_once_per_process(tmp_path, monkeypatch):
    """It is a property of the PROCESS, not of the dispatch: a 6-group chain
    must not emit six copies of a paragraph.  ``close_worker_pool`` -- the
    documented return-to-cold entry point -- re-arms it, so a driver that
    swaps ``__main__`` is not silently answered from the old verdict."""
    script = tmp_path / 'runner_no_guard2.py'
    script.write_text('print("top level")\n')
    _fake_main(monkeypatch, script)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        for _ in range(6):
            assert LT._newton_resolve_workers(4, _D121_POINTS, _D121_FIT,
                                              _free_b=1e15) == 1
    assert len([w for w in rec if '__main__' in str(w.message)]) == 1
    LT.close_worker_pool()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        LT._newton_resolve_workers(4, _D121_POINTS, _D121_FIT, _free_b=1e15)
    assert len([w for w in rec if '__main__' in str(w.message)]) == 1, (
        'close_worker_pool did not return the process to a cold state for the '
        'guard verdict, so it is not the return-to-cold entry point it claims')


def test_a_guarded_main_is_left_alone(tmp_path, monkeypatch):
    """The control.  A caller who did the standard thing keeps the pool."""
    script = tmp_path / 'runner_guarded.py'
    script.write_text('import numpy as np\n\n\ndef main():\n    return 1\n\n\n'
                      "if __name__ == '__main__':\n    main()\n")
    _fake_main(monkeypatch, script)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        got = LT._newton_resolve_workers(_D121_NW, _D121_POINTS, _D121_FIT,
                                         _free_b=1e15)
    assert got == _D121_NW
    assert [w for w in rec if '__main__' in str(w.message)] == []


@pytest.mark.parametrize('body,guarded', [
    ("if __name__ == '__main__':\n    pass\n", True),
    ('if __name__ == "__main__":\n    pass\n', True),
    ("if '__main__' == __name__:\n    pass\n", True),
    ("if __name__ in ('__main__', '__mp_main__'):\n    pass\n", True),
    ("def f():\n    if __name__ == '__main__':\n        pass\n", False),
    ("x = 1\n", False),
    ("# if __name__ == '__main__':\nx = 1\n", False),
])
def test_the_guard_detector_reads_the_ast_not_the_text(tmp_path, body,
                                                       guarded):
    """Spelling-tolerant where it must be, and NOT text-matching where that
    would be wrong: a guard nested inside a function does not protect the
    module body, and one inside a comment protects nothing at all.  Top-level
    only, deliberately -- the failure is 'the module body re-runs'."""
    LT._reset_newton_pool_resource_state()
    p = tmp_path / f'g{abs(hash(body))}.py'
    p.write_text(body)
    assert LT._script_has_main_guard(str(p)) is guarded


def test_an_unreadable_main_keeps_the_pool(tmp_path, monkeypatch):
    """"Cannot prove it is unguarded" must not mean "assume the worst": a file
    we cannot read or parse gets the historical behaviour, so a zipapp / a
    frozen bundle / a syntax error in an unrelated file cannot silently
    serialise a working pipeline."""
    LT._reset_newton_pool_resource_state()
    bad = tmp_path / 'not_python.py'
    bad.write_text('this is (not python\n')
    assert LT._script_has_main_guard(str(bad)) is True
    assert LT._script_has_main_guard(str(tmp_path / 'missing.py')) is True


@pytest.mark.parametrize('spec_name,file_name,expect_path', [
    ('pytest.__main__', 'whatever.py', False),   # python -m pytest
    ('__main__', 'pytest.exe/__main__.py', False),   # pytest console script
    (None, 'ipython', False),                    # multiprocessing's carve-out
    (None, 'plain_runner.py', True),             # python plain_runner.py
    ('yourscript', 'yourscript.py', True),       # python -m yourscript
])
def test_the_predicate_mirrors_multiprocessing(tmp_path, monkeypatch,
                                               spec_name, file_name,
                                               expect_path):
    """The predicate must mirror ``multiprocessing.spawn`` rather than guess,
    or it will serialise every pytest run (``__main__.__spec__.name`` is
    ``pytest.__main__`` under ``python -m pytest`` and ``__main__`` under the
    console script -- both take ``_fixup_main_from_name``'s early return and
    re-run NOTHING).  MEASURED on this box for every row, and recorded in
    docs/audits/FIX_POOL_MEMORY_2026_08_06.md sec 5.3.

    The ``python -m yourscript`` row is the one that is easy to get wrong in
    the other direction: ``__spec__.name`` IS set there, but it does not end in
    ``__main__``, so ``_fixup_main_from_name`` falls through to
    ``runpy.run_module(..., run_name='__mp_main__')`` and the body re-runs just
    as it does for a path."""
    LT._reset_newton_pool_resource_state()
    path = tmp_path / os.path.basename(file_name)
    path.write_text('x = 1\n')          # unguarded on purpose
    mod = types.ModuleType('__main__')
    mod.__file__ = str(path) if file_name != 'ipython' else str(
        tmp_path / 'ipython')
    if file_name == 'ipython':
        (tmp_path / 'ipython').write_text('x = 1\n')
    mod.__spec__ = (types.SimpleNamespace(name=spec_name)
                    if spec_name is not None else None)
    monkeypatch.setitem(sys.modules, '__main__', mod)
    got = LT._spawn_reexecuted_main_script()
    assert (got is not None) is expect_path, (
        f'spec_name={spec_name!r} file={file_name!r} -> {got!r}')


# ---------------------------------------------------------------------------
# (c) BIT-IDENTITY: a pool the CLAMP sized must still equal serial
# ---------------------------------------------------------------------------

_WL = 1.31e-6
_W0 = 1.0e-3
_N, _RS = 1024, 2          # 262144 Newton points: past the cold pool bar


def _doublet(ap):
    surfs, before = [], 'air'
    for R, g in ((61.5e-3, 'N-BK7'), (-45.0e-3, 'N-SF5'), (-128.0e-3, 'air')):
        surfs.append({'radius': R, 'glass_before': before, 'glass_after': g,
                      'conic': 0.0, 'radius_y': None, 'conic_y': None,
                      'aspheric_coeffs': None, 'aspheric_coeffs_y': None})
        before = g
    return {'name': 'doublet', 'aperture_diameter': ap,
            'thicknesses': [4.0e-3, 2.5e-3], 'surfaces': surfs}


def _run(n_workers):
    ap = 1.2 * 2.0 * _W0
    dx = float(2.2 * max(ap, 3.0 * _W0) / _N)
    x = (np.arange(_N) - _N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env0 = np.exp(-(X ** 2 + Y ** 2) / _W0 ** 2).astype(np.complex128)
    res = la.propagate_traced_carrier_chain(
        env0, [{'prescription': _doublet(ap), 'gap_before': 0.0}],
        _WL, dx, r_in=np.inf, ray_subsample=_RS, n_workers=n_workers,
        final_distance=0.0,
        traced_kwargs=dict(parallel_amp=False, on_undersample='silent',
                           newton_fit='polynomial'))
    return np.asarray(res.field)


class _WidthSpy:
    """Record the worker count each pool dispatch actually ran at."""

    def __init__(self):
        self.widths = []

    def __enter__(self):
        LT.close_worker_pool()
        self._orig = LT._get_persistent_worker_pool

        def _spy(n_workers):
            self.widths.append(int(n_workers))
            return self._orig(n_workers)

        LT._get_persistent_worker_pool = _spy
        return self

    def __exit__(self, *exc):
        LT._get_persistent_worker_pool = self._orig
        LT.close_worker_pool()
        return False


def _cheb_backend_note():
    """What the PARENT process resolves for the Chebyshev evaluator.

    A spawn worker rebuilds the fit in a fresh interpreter, so pool/serial
    identity needs the parent and the worker to land on the SAME backend.
    Nothing in the pickled payload pins that, and the parent's answer is
    process state a long test session can change -- so when the serial arm
    below disagrees, this is the first thing to read."""
    try:
        kern = LT._get_cheb2d_val_grad_numba()
    except Exception as exc:                       # pragma: no cover - defensive
        kern = f'<raised {type(exc).__name__}>'
    return (f'parent _NUMBA_AVAILABLE={LT._NUMBA_AVAILABLE!r}, '
            f'cheb kernel={"present" if callable(kern) else kern!r}')


@pytest.mark.slow
def test_a_capped_pool_is_bit_identical_to_serial(monkeypatch):
    """The safety argument for clamping, re-proved on the path the clamp
    creates: a pool of a size NOBODY asked for.

    Two DIFFERENT contracts meet here and the test keeps them apart, because
    conflating them makes a pre-existing library defect look like a clamp bug
    (it did, on the first full-tree sweep of this fix -- see below).

    1.  THE CLAMP'S OWN CONTRACT, asserted unconditionally: a pool the clamp
        sized to 2 must equal a pool the caller sized to 4.  That is a
        different ``np.array_split`` of the same Newton lattice, so it is
        exactly what the clamp could break and nothing else is.  Both arms run
        their fit in fresh workers, so the backend question below cannot reach
        this comparison.

    2.  THE LIBRARY'S CONTRACT (serial == pool), which
        ``test_niche_newton_pool_both_fits.py`` also asserts.  It is NOT
        unconditional, and this fix's sweep is what showed it:

            MEASURED, one process, same shape, N=1024/rs=2 --
              parent _NUMBA_AVAILABLE=True   -> serial == pool, max|delta| 0
              parent _NUMBA_AVAILABLE=False  -> DIFFER,  max|delta| 5.167e-14

        The worker rebuilds ``_Cheb2DEvaluator`` in a fresh interpreter and
        takes the njit Chebyshev kernel if numba imports THERE; the payload
        carries ``newton_fit`` / ``fit_poly_order`` / ``fit_weights`` /
        ``newton_max_iters`` but no backend flag.  So a parent that has resolved
        a different evaluator than its workers runs a different floating-point
        ORDER, and the two paths part company in the last bits (1.8e-12 observed
        on the full-tree sweep).  That is a PRE-EXISTING gap in the payload --
        the same class as audit E-H2's ``newton_max_iters`` -- not something the
        resource clamp can cause or cure, and fixing it means touching the
        numerical path, which a resource-safety change must not do.

        So arm 2 is checked against the UNCAPPED pool first.  If serial already
        disagrees with a pool the clamp never touched, this process cannot
        honour the library's contract at all and the test says so precisely
        instead of blaming the clamp.

    The clamp is forced to land on 2 by pinning BOTH inputs: available memory
    (frozen psutil snapshot) and the per-worker cost, so the assertion is
    arithmetic rather than a statement about this box's free RAM.
    """
    import psutil
    vm = psutil.virtual_memory()
    if vm.available < 6 * (1 << 30):
        pytest.skip('insufficient free RAM to run two Newton workers for real')
    monkeypatch.setattr(psutil, 'virtual_memory',
                        lambda: vm._replace(available=int(20e9)))
    # budget = 0.5 * 20 GB - 2 GB = 8 GB; a flat 3 GB/worker => exactly 2.
    monkeypatch.setattr(LT, '_NEWTON_WORKER_BASE_BYTES', 3.0e9)
    monkeypatch.setattr(LT, '_NEWTON_WORKER_BYTES_PER_POINT', 0.0)
    monkeypatch.setattr(LT, '_NEWTON_WORKER_FIT_BYTES_PER_POINT', 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert LT._newton_resolve_workers(4, _D121_POINTS, _D121_FIT) == 2, (
            'the pinned arithmetic no longer lands on 2 workers, so this test '
            'would not exercise a capped pool')

    with _WidthSpy() as spy:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            capped = _run(4)
            widths = list(spy.widths)
    assert widths and set(widths) == {2}, (
        f'the capped run dispatched at widths {widths}; expected every '
        f'dispatch at the clamped 2, so this comparison does not exercise a '
        f'capped pool')

    # ...the same request, with the clamp unable to bind: 4 workers, 4 chunks.
    monkeypatch.setattr(LT, '_NEWTON_WORKER_BASE_BYTES', 1.0)
    with _WidthSpy() as spy4:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            uncapped = _run(4)
            widths4 = list(spy4.widths)
    assert widths4 and set(widths4) == {4}, (
        f'the uncapped reference dispatched at widths {widths4}, not 4, so it '
        f'is not a control for the clamped run')

    # (1) THE CLAMP'S CONTRACT -- unconditional.
    assert np.array_equal(capped, uncapped), (
        f'the memory clamp changed the answer: a pool it sized to 2 differs '
        f'from the pool of 4 the caller asked for, max|delta| = '
        f'{np.abs(capped - uncapped).max():.3e}.  Worker count must be a speed '
        f'knob only, so this is the clamp being wrong, not the backend.')

    # (2) THE LIBRARY'S CONTRACT -- conditional on parent/worker backend.
    with _WidthSpy() as spy_s:
        serial = _run(1)
        assert spy_s.widths == [], 'the n_workers=1 reference used a pool'
    if not np.array_equal(serial, uncapped):
        pytest.skip(
            f'this process cannot honour the library serial==pool contract: '
            f'an UNCAPPED 4-worker pool already differs from serial by '
            f'{np.abs(uncapped - serial).max():.3e}, so the clamp is not '
            f'involved.  Pre-existing: the worker re-derives the Chebyshev fit '
            f'in a fresh interpreter and the payload pins no backend '
            f'({_cheb_backend_note()}).  See '
            f'docs/audits/FIX_POOL_MEMORY_2026_08_06.md sec 8.')
    assert np.array_equal(serial, capped), (
        f'serial matches an uncapped pool but not the CAPPED one, max|delta| = '
        f'{np.abs(capped - serial).max():.3e} -- that is the clamp, and only '
        f'the clamp, changing the answer')


# ---------------------------------------------------------------------------
# mechanism pins -- a refactor must not be able to drop the clamp silently
# ---------------------------------------------------------------------------

def test_the_dispatch_path_consults_the_resolver():
    """Pin the WIRING the way the sibling tests pin theirs.  Bit-identity
    cannot notice a missing clamp (an unclamped pool returns the same numbers,
    right up until it does not return at all), so the only durable guard is
    that the dispatch path still asks."""
    src = inspect.getsource(LT.apply_real_lens_traced)
    i = src.find('def _invert_newton_parallel')
    assert i != -1
    body = src[i:i + 6000]
    assert '_newton_resolve_workers(_n_cpu_req, n_total, _fit_points' in body, (
        'the Newton dispatch no longer resolves a memory-clamped worker count; '
        'this is the defect the design-121 capstone measured at 205.7 / '
        '227.5 GB commit')
    assert 'min_pool_points=_POOL_MIN_PIXELS_WARM' in body, (
        'the clamp no longer receives the pool size bar, so it will announce '
        'caps on calls that run in-process anyway')
    # ...and it must be resolved BEFORE the cost gate reads the worker count,
    # so the promotion evidence is keyed by the count that would actually run.
    assert body.index('_newton_resolve_workers') < body.index(
        '_pool_reuse_is_likely(n_cpu'), (
        'the resource clamp now runs AFTER the cost gate, so the promotion '
        'evidence is keyed by a worker count the dispatch will not use')
    # the split and the pool must both use the CLAMPED count, not the request
    assert 'np.array_split(np.arange(n_total), n_cpu)' in body
    assert '_get_persistent_worker_pool(n_cpu)' in body


def test_the_fine_grid_ceiling_is_still_single_process():
    """The other half of the capstone's finding, recorded as a live fact rather
    than a note: ``_memory_bounded_n_fine`` sizes the fine grid with a
    single-process model and does NOT know about ``n_workers``.  That is why
    the clamp has to live on the pool side -- this test exists so that if the
    fine-grid model ever DOES learn about workers, whoever changes it finds the
    pool-side clamp and decides deliberately rather than double-counting."""
    from lumenairy.propagators import carrier as C
    sig = inspect.signature(C._memory_bounded_n_fine)
    assert 'n_workers' not in sig.parameters

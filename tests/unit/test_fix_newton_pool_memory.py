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
    Since v5.32.3 the payload PINS it (``cheb_backend``) so they always do;
    this note is what a residual mismatch would have to be read against."""
    try:
        kern = LT._get_cheb2d_val_grad_numba()
    except Exception as exc:                       # pragma: no cover - defensive
        kern = f'<raised {type(exc).__name__}>'
    return (f'parent _NUMBA_AVAILABLE={LT._NUMBA_AVAILABLE!r}, '
            f'cheb kernel={"present" if callable(kern) else kern!r}, '
            f"resolved backend={LT._resolved_cheb_backend('polynomial')!r}")


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
        ``test_niche_newton_pool_both_fits.py`` also asserts.  Also
        unconditional SINCE v5.32.3 (docs/audits/FIX_CI_POOL_2026_08_06.md),
        and it was not before:

            MEASURED, one process, same shape, N=1024/rs=2 --
              parent _NUMBA_AVAILABLE=True   -> serial == pool, max|delta| 0
              parent _NUMBA_AVAILABLE=False  -> DIFFER,  max|delta| 5.167e-14

        The worker rebuilds ``_Cheb2DEvaluator`` in a fresh interpreter and used
        to take the njit Chebyshev kernel if numba imported THERE, while the
        payload carried ``newton_fit`` / ``fit_poly_order`` / ``fit_weights`` /
        ``newton_max_iters`` but no backend flag -- so a parent that resolved a
        different evaluator than its workers ran a different floating-point
        ORDER and the two parted company in the last bits (1.8e-12 on the
        full-tree sweep, 1.358e-11 on CI).  The payload now pins
        ``cheb_backend`` and the worker honours it (or refuses the chunk), so
        the condition is gone and arm 2 asserts rather than skips.

        The UNCAPPED pool is still measured first, because it localises a
        failure: a break there is the library's identity contract, a break in
        the capped-vs-serial line below is the clamp.

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

    # (2) THE LIBRARY'S CONTRACT -- unconditional since the v5.32.3 pin.
    with _WidthSpy() as spy_s:
        serial = _run(1)
        assert spy_s.widths == [], 'the n_workers=1 reference used a pool'
    assert np.array_equal(serial, uncapped), (
        f'an UNCAPPED 4-worker pool differs from serial by '
        f'{np.abs(uncapped - serial).max():.3e}, so the clamp is not involved: '
        f'this is the library serial==pool contract.  Until v5.32.3 that was '
        f'conditional on the parent and its workers resolving the same '
        f'Chebyshev evaluator; the payload now pins it ({_cheb_backend_note()})'
        f'.  See docs/audits/FIX_CI_POOL_2026_08_06.md.')
    assert np.array_equal(serial, capped), (
        f'serial matches an uncapped pool but not the CAPPED one, max|delta| = '
        f'{np.abs(capped - serial).max():.3e} -- that is the clamp, and only '
        f'the clamp, changing the answer')


# ---------------------------------------------------------------------------
# mechanism pins -- a refactor must not be able to drop the clamp silently
# ---------------------------------------------------------------------------

def _dispatch_closure_source():
    """Source of the ``_invert_newton_parallel`` closure, bounded by the NEXT
    closure rather than by a character count.

    A fixed ``src[i:i + 6000]`` window was what this test used, and it broke on
    the very next change to the dispatcher -- the v5.32.3 backend pin added
    comments ahead of the pool call and pushed
    ``_get_persistent_worker_pool(n_cpu)`` past 6000 characters, failing a
    wiring pin whose subject had not moved.  A structural bound cannot rot that
    way, and it strictly widens what the pin covers.
    """
    src = inspect.getsource(LT.apply_real_lens_traced)
    i = src.find('def _invert_newton_parallel')
    assert i != -1, 'could not locate _invert_newton_parallel'
    j = src.find('\n    def ', i + 1)          # next closure at the same indent
    return src[i:] if j == -1 else src[i:j]


def test_the_dispatch_path_consults_the_resolver():
    """Pin the WIRING the way the sibling tests pin theirs.  Bit-identity
    cannot notice a missing clamp (an unclamped pool returns the same numbers,
    right up until it does not return at all), so the only durable guard is
    that the dispatch path still asks."""
    body = _dispatch_closure_source()
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


# ---------------------------------------------------------------------------
# v5.32.3 (docs/audits/FIX_CI_POOL_2026_08_06.md) -- the two follow-ups this
# file's own sec 8 left open, closed here because both are pool-payload facts:
#
#   * sec 8.1  serial == pool was CONDITIONAL on the parent and its spawn
#              workers resolving the same Chebyshev evaluator.  The payload now
#              pins ``cheb_backend``.
#   * the cap notice had no policy surface, so a physics-guard suppression test
#     broke on a 12 GB CI runner and passed on a 128 GB box.  ``on_pool_memory``
#     is that surface.
#
# Everything below is arithmetic, source inspection or a direct call to
# ``_newton_invert_chunk``: no test here spawns a pool.
# ---------------------------------------------------------------------------

def _fit_payload(order=6, n=25):
    """A minimal, REAL ``_newton_invert_chunk`` payload over a smooth map, so
    the worker's own gates can be driven without running a lens."""
    xs = np.linspace(-1e-3, 1e-3, n)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    return {
        'xs_in': xs,
        'x_out_grid': 0.9 * X + 3.0e2 * X * (X ** 2 + Y ** 2),
        'y_out_grid': 0.9 * Y + 3.0e2 * Y * (X ** 2 + Y ** 2),
        'opl_grid': 1.0e-3 + 5.0e2 * (X ** 2 + Y ** 2),
        'launch_radius': 1.0e-3,
        'dx': 1.0e-5,
        'bound': 0.999e-3,
        'inv_M_x': 1.0 / 0.9,
        'inv_M_y': 1.0 / 0.9,
        'newton_fit': 'polynomial',
        'fit_poly_order': order,
        'fit_weights': None,
        'newton_max_iters': 12,
    }


def _chunk_pts(m=64):
    r = np.linspace(-4e-4, 4e-4, m)
    Xw, Yw = np.meshgrid(r, r, indexing='ij')
    return Xw.ravel().copy(), Yw.ravel().copy()


def test_the_two_evaluator_backends_are_not_bit_identical():
    """THE PREMISE, measured rather than assumed.

    The pin is only worth its complexity if the two branches of
    ``_Cheb2DEvaluator.ev_value_and_grad`` really do differ.  They compute one
    polynomial two ways -- an njit Chebyshev recurrence and a pure-xp
    Vandermonde contraction -- so they agree to ~1e-16 RELATIVE and not bit for
    bit.  If a future change ever made them identical this test fails, and
    whoever sees it can retire the pin deliberately instead of finding out from
    a 1e-11 CI failure.
    """
    if LT._resolved_cheb_backend() != 'numba':
        pytest.skip('no numba kernel in this process: there is only one branch')
    p = _fit_payload()
    xa, ya = _chunk_pts(48)
    ev_nb = LT._Cheb2DEvaluator(p['xs_in'], p['xs_in'], p['opl_grid'],
                                order=6, xp=np, backend='numba')
    ev_np = LT._Cheb2DEvaluator(p['xs_in'], p['xs_in'], p['opl_grid'],
                                order=6, xp=np, backend='numpy')
    f_nb, _, _ = ev_nb.ev_value_and_grad(xa, ya)
    f_np, _, _ = ev_np.ev_value_and_grad(xa, ya)
    assert np.allclose(f_nb, f_np, rtol=1e-12, atol=0.0), (
        'the two evaluator backends disagree by more than rounding; that is a '
        'bug in one of them, not the ordering difference this pin is about')
    assert not np.array_equal(f_nb, f_np), (
        'the numba and pure-NumPy Chebyshev evaluators are now bit-identical.  '
        'If that is deliberate, the cheb_backend pin can be retired -- but '
        'retire it on purpose, with this test as the evidence')


def test_the_evaluator_honours_a_pinned_backend():
    """The pin has to actually select the branch, not merely be carried."""
    p = _fit_payload()
    xa, ya = _chunk_pts(16)
    seen = []
    orig = LT._get_cheb2d_val_grad_numba

    def _spy():
        k = orig()
        seen.append(k is not None)
        return k

    LT._get_cheb2d_val_grad_numba = _spy
    try:
        LT._Cheb2DEvaluator(p['xs_in'], p['xs_in'], p['opl_grid'], order=6,
                            xp=np, backend='numpy').ev_value_and_grad(xa, ya)
        assert seen == [], (
            "backend='numpy' still consulted the numba kernel getter, so the "
            "pin does not decide the branch")
        LT._Cheb2DEvaluator(p['xs_in'], p['xs_in'], p['opl_grid'], order=6,
                            xp=np, backend='numba').ev_value_and_grad(xa, ya)
        assert seen, "backend='numba' did not resolve the kernel"
    finally:
        LT._get_cheb2d_val_grad_numba = orig
    # and an unknown pin is refused rather than silently meaning 'auto'
    with pytest.raises(ValueError, match='backend'):
        LT._Cheb2DEvaluator(p['xs_in'], p['xs_in'], p['opl_grid'], order=6,
                            xp=np, backend='numexpr')


def test_the_worker_answers_in_the_pinned_order_not_its_own():
    """FAIL-BEFORE, run through the real worker entry point.

    ``_newton_invert_chunk`` IS the spawn worker's body, so calling it with two
    payloads that differ only in ``cheb_backend`` reproduces the parent/worker
    split without a pool: pre-fix both payloads took whatever branch the
    process had, which is exactly why a parent on the other branch disagreed
    with them.
    """
    if LT._resolved_cheb_backend() != 'numba':
        pytest.skip('no numba kernel in this process: there is only one branch')
    p = _fit_payload()
    xa, ya = _chunk_pts(64)
    opl_nb, _ = LT._newton_invert_chunk((dict(p, cheb_backend='numba'), xa, ya))
    opl_np, _ = LT._newton_invert_chunk((dict(p, cheb_backend='numpy'), xa, ya))
    # A payload with NO key keeps the historical behaviour: resolve locally.
    # In this process that is numba, so it must reproduce the numba pin.
    opl_old, _ = LT._newton_invert_chunk((dict(p), xa, ya))
    assert np.array_equal(opl_old, opl_nb), (
        'a payload written before the pin existed no longer behaves as it did; '
        'the compatibility default must be "resolve locally", not "numpy"')
    assert not np.array_equal(opl_nb, opl_np), (
        'the two pins produced identical OPL, so this call is not sensitive to '
        'the backend and proves nothing about the pin')
    assert np.allclose(opl_nb, opl_np, rtol=1e-11, atol=0.0)


def test_a_worker_that_cannot_honour_the_pin_refuses_the_chunk():
    """A worker cannot conjure the parent's backend.  Answering in the other
    floating-point order would be silently wrong -- the pool's entire safety
    argument is bit-identity -- so it raises, and the parent runs the chunk
    itself.  Emulated by making the kernel getter unavailable INSIDE the worker
    call, which is what a fresh interpreter without numba does."""
    p = dict(_fit_payload(), cheb_backend='numba')
    xa, ya = _chunk_pts(16)
    orig = LT._get_cheb2d_val_grad_numba
    LT._get_cheb2d_val_grad_numba = lambda: None
    try:
        with pytest.raises(LT.NewtonWorkerBackendUnavailable):
            LT._newton_invert_chunk((p, xa, ya))
        # ...but a payload pinned to 'numpy' is served happily by the same
        # worker: the refusal is about the pin it cannot honour, not about numba
        out, _ = LT._newton_invert_chunk(
            (dict(p, cheb_backend='numpy'), xa, ya))
        assert np.isfinite(out).any()
    finally:
        LT._get_cheb2d_val_grad_numba = orig
    # the refusal must survive the pool's own except-clause ordering: it is a
    # RuntimeError subclass, so a refactor that drops the specific handler still
    # falls back to serial rather than propagating
    assert issubclass(LT.NewtonWorkerBackendUnavailable, RuntimeError)


def test_the_dispatch_path_pins_the_backend_and_handles_the_refusal():
    """Wiring pin, same reasoning as ``test_the_dispatch_path_consults_the_
    resolver``: bit-identity cannot notice a MISSING pin on a box where the
    parent and its workers happen to agree -- which is every dev box that has
    numba -- so the only durable guard is that the dispatch path still pins."""
    body = _dispatch_closure_source()
    assert ("_spline_data['cheb_backend'] = _resolved_cheb_backend(newton_fit)"
            in body), (
        'the Newton payload no longer pins the parent-resolved Chebyshev '
        'backend: serial == pool goes back to being conditional on the workers '
        'resolving the same evaluator (FIX_POOL_MEMORY sec 8.1)')
    assert 'except NewtonWorkerBackendUnavailable' in body, (
        'a worker refusal is no longer handled specifically, so it would be '
        'absorbed by the pool-infrastructure clause with no diagnostic')
    assert body.index('except NewtonWorkerBackendUnavailable') < body.index(
        'except (BrokenProcessPool'), (
        'the generic pool-infrastructure handler now shadows the backend '
        'refusal, so the specific remedy is never printed')
    # The payload is pinned BEFORE it is FROZEN for the wire.  Pre-2026-08-10
    # the freeze was ``args_list = [...]`` (the dict embedded in every chunk
    # tuple); since FIX_PERF_ROUND2 item 3 it is the single
    # ``_newton_payload_blob`` call, whose bytes AND content digest are what
    # every worker then answers from -- so a stamp landing after it would be
    # invisible to the pool in exactly the way this pin exists to prevent.
    assert '_newton_payload_blob(_spline_data)' in body, (
        'the dispatch no longer freezes the payload through '
        '_newton_payload_blob, so this ordering pin has lost its anchor')
    assert body.index("_spline_data['cheb_backend']") < body.index(
        '_newton_payload_blob(_spline_data)'), (
        'the backend is pinned after the payload has already been serialised '
        'for the workers')
    assert body.index("_spline_data['cheb_fit']") < body.index(
        '_newton_payload_blob(_spline_data)'), (
        'the built fit is stamped after the payload has already been '
        'serialised for the workers')
    wsrc = inspect.getsource(LT._newton_invert_chunk)
    assert "knot_data.get('cheb_backend', None)" in wsrc, (
        'the worker no longer reads the pin')
    assert 'backend=_backend' in wsrc, (
        'the worker reads the pin but does not pass it to the evaluator')


def test_the_backend_refusal_latch_says_it_once_and_close_rearms_it():
    """The refusal is a fact about this process's workers, so a 6-group chain
    must not print the paragraph six times -- and ``close_worker_pool`` (which
    tears those workers down) must re-arm it, exactly as it re-arms the
    unguarded-``__main__`` warning."""
    LT.close_worker_pool()
    assert LT._note_pool_backend_refusal() is True
    assert LT._note_pool_backend_refusal() is False
    assert LT._POOL_BACKEND_REFUSED is True
    LT.close_worker_pool()
    assert LT._POOL_BACKEND_REFUSED is False
    assert LT._note_pool_backend_refusal() is True
    LT.close_worker_pool()


def test_the_cap_notice_has_a_policy_knob_that_only_moves_the_report():
    """``on_pool_memory``: the cap fires on a 12 GB CI runner and never on a
    128 GB workstation, so a test that asserts "this policy leaves no warnings"
    was passing locally and failing on CI for a reason that had nothing to do
    with the guard under test.  The knob is the routing every other guard in
    this signature already has.

    Both halves matter: the notice must be suppressible, and suppressing it
    must not change the CLAMP -- a knob that quietly restored the unclamped
    worker count would re-open the OOM this whole file exists for.
    """
    free_b = 12.0e9
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        loud = LT._newton_resolve_workers(_D121_NW, _D121_POINTS, _D121_FIT,
                                          _free_b=free_b)
    caps = [w for w in rec if 'Newton process pool asked for' in str(w.message)]
    assert len(caps) == 1, f'expected exactly one cap notice, got {len(caps)}'
    assert loud < _D121_NW, 'the premise is wrong: the clamp did not bind'

    for action in ('silent', 'ignore', 'off'):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            quiet = LT._newton_resolve_workers(
                _D121_NW, _D121_POINTS, _D121_FIT, on_pool_memory=action,
                _free_b=free_b)
        assert quiet == loud, (
            f'on_pool_memory={action!r} changed the clamped worker count '
            f'{loud} -> {quiet}; it may only change what is REPORTED')


def test_the_cap_knob_is_gated_at_entry_not_inside_the_warning_branch():
    """The D5 house rule (``_KNOWN_UNGATED``), applied to the knob this fix
    adds.  ``on_undersample`` is in that ledger precisely because its
    validation sits inside the branch that only runs when the condition trips;
    the memory cap has the same shape (it binds on a small box and never on a
    big one), so validating it there would mean the knob is checked on CI and
    not on a workstation.  Assert both the resolver-level gate and the
    signature default."""
    sig = inspect.signature(la.apply_real_lens_traced)
    assert sig.parameters['on_pool_memory'].default == 'warn'
    assert LT._traced_kwarg_defaults()['on_pool_memory'] == 'warn', (
        'the chain-forwarded default drifted from the signature default')
    # abundant memory: the warning branch is unreachable, and junk STILL raises
    for bad in ('zzz', 'Silent', None, 1):
        with pytest.raises(ValueError, match='on_pool_memory'):
            LT._newton_resolve_workers(_D121_NW, _D121_POINTS, _D121_FIT,
                                       on_pool_memory=bad, _free_b=400e9)
    # ...and the entry gate is in apply_real_lens_traced itself, unconditional
    body = inspect.getsource(la.apply_real_lens_traced)
    assert 'on_pool_memory = _pool_memory_policy(on_pool_memory)' in body


def test_the_unguarded_main_refusal_is_not_routed_through_the_knob(
        tmp_path, monkeypatch):
    """The line this fix draws, pinned so it is a decision and not an accident.

    Rule 1 is not a resource notice: an unguarded ``__main__`` makes every
    spawn worker re-run the caller's whole program, side effects included, and
    there is no worker count at which that is acceptable.  So
    ``on_pool_memory='silent'`` must NOT silence it -- a caller who quietened
    the cap has not asked to be told nothing about correctness."""
    script = tmp_path / 'runner_no_guard_knob.py'
    script.write_text('print("no guard here")\n')
    _fake_main(monkeypatch, script)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        got = LT._newton_resolve_workers(_D121_NW, _D121_POINTS, _D121_FIT,
                                         on_pool_memory='silent',
                                         _free_b=400e9)
    assert got == 1, 'the unguarded-__main__ rule stopped returning serial'
    assert [w for w in rec if '__main__' in str(w.message)], (
        "on_pool_memory='silent' silenced the unguarded-__main__ refusal; "
        "that is a correctness hazard (the caller's side effects run K extra "
        'times), not a resource notice')


# ---------------------------------------------------------------------------
# v5.33.0 (docs/audits/FIX_POOL_REBUILD_2026_08_08.md) -- THE SECOND HALF.
#
# The ``cheb_backend`` pin above made the parent and its workers EVALUATE in
# the same floating-point order.  ``test_pool_result_is_bit_identical_to_
# serial[polynomial]`` went on failing on CI afterwards, at 1.341e-11 /
# 1.358e-11, in ALL FOUR python lanes -- because the worker was still REBUILDING
# the fit.  ``_Cheb2DEvaluator.__init__`` runs ``_solve_lstsq_thread_safe``,
# i.e. ``A^T A`` and ``A^T b`` over a ~78 000-row design matrix, and OpenBLAS
# reduces those in a thread-count-dependent order.  A spawn worker does not
# inherit its parent's BLAS width (``threadpoolctl``'s cap is process-global, so
# a long-lived pytest parent that has been through a capped section is not at
# the environment default a fresh interpreter starts at), so the two recovered
# DIFFERENT coefficients from identical data.
#
# The parent now SHIPS its built fit and the worker evaluates it.  The tests
# below are the premise, the fail-before, the wiring, and one real spawned
# worker.
# ---------------------------------------------------------------------------

# ~78 000 rows: the traced doublet's own ray-fit grid at this file's N=1024 /
# ray_subsample=2 shape.  Big enough that OpenBLAS threads the reduction, which
# is the whole phenomenon -- a 25x25 toy grid is single-threaded on every build
# and would make the premise test vacuously green.
_BLAS_FIT_N = 279


def _blas_widths():
    """BLAS caps to compare a fit rebuild across, or ``()`` when this build
    cannot be capped at all."""
    try:
        from threadpoolctl import threadpool_info, threadpool_limits  # noqa: F401
    except ImportError:
        return ()
    return (1, 2, 4, None)


def _blas_payload(n=_BLAS_FIT_N):
    """``_fit_payload`` with an OPL map that is NOT exactly representable in
    the order-6 basis.

    The toy payload's OPL is a paraboloid, which an order-6 total-degree fit
    reproduces exactly -- every coefficient above degree 2 comes out at ~0, so
    a reduction-order difference has nothing to show up in.  A degree-6 radial
    term makes the residual real and the coefficients O(1), which is the regime
    the traced doublet's own fit is in.
    """
    p = _fit_payload(order=6, n=n)
    xs = p['xs_in']
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    r2 = X ** 2 + Y ** 2
    p['opl_grid'] = (1.0e-3 + 5.0e2 * r2 + 2.1e8 * r2 * r2
                     + 3.0e2 * X * r2 - 7.7e10 * r2 ** 3)
    return p


def _rebuild_coeffs(p, width):
    """Re-FIT the OPL evaluator from the payload's grids under a BLAS cap."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=width):
        return np.asarray(
            LT._Cheb2DEvaluator(p['xs_in'], p['xs_in'], p['opl_grid'],
                                order=int(p['fit_poly_order']), xp=np,
                                backend='numpy').coeffs)


def _shipped_payload(p):
    """``p`` plus the built fit the dispatch site now ships, PICKLE-ROUND-
    TRIPPED, because that is what a real worker receives."""
    import pickle
    evs = [LT._Cheb2DEvaluator(p['xs_in'], p['xs_in'], p[k], xp=np,
                               order=int(p['fit_poly_order']))
           for k in ('x_out_grid', 'y_out_grid', 'opl_grid')]
    q = dict(p)
    q['cheb_fit'] = LT._cheb_fit_payload(*evs, newton_fit='polynomial')
    return pickle.loads(pickle.dumps(q))


def test_a_rebuilt_fit_is_not_blas_width_stable():
    """THE PREMISE, measured rather than argued -- and the FAIL-BEFORE.

    If ``A^T A`` were reduced identically at every thread width there would be
    nothing here to fix and shipping the fit would be dead weight.  MEASURED on
    the traced doublet's own 77 841-point OPL fit (Windows 11, py3.14.6,
    numpy 2.4.4 / scipy-openblas 0.3.31, 24 cores), coefficients at BLAS width
    W against the process default: 4.596e-15 / 4.243e-15 / 5.484e-15 /
    8.766e-16 at W = 1 / 2 / 4 / 8, while 30/30 rebuilds at a FIXED width are
    bit-identical and so is a real spawned worker at that same width.  So this
    is the width and nothing else.  Full table in
    docs/audits/FIX_POOL_REBUILD_2026_08_08.md sec 2.

    A build whose BLAS cannot be capped, or is single-threaded on this shape,
    skips -- on such a build a worker that re-fits was already safe, which is
    exactly why the contract passed on some boxes and failed on CI.
    """
    widths = _blas_widths()
    if not widths:
        pytest.skip('threadpoolctl absent: this build cannot vary BLAS width')
    p = _blas_payload()
    ref = _rebuild_coeffs(p, None)
    deltas = {w: float(np.abs(ref - _rebuild_coeffs(p, w)).max())
              for w in widths if w is not None}
    # whatever it does, it must do it REPEATABLY at a fixed width, or the
    # measurement below is noise rather than a reduction-order fact
    assert np.array_equal(ref, _rebuild_coeffs(p, None)), (
        'the fit is not even reproducible in ONE process at ONE BLAS width; '
        'that is a different (and worse) defect than the one this fix closes')
    if not any(d > 0.0 for d in deltas.values()):
        pytest.skip(
            f'this BLAS reduces identically at every width tried ({deltas}), '
            f'so a worker that re-fits was already safe here.  The shipped '
            f'fit is still the durable answer -- see the sibling tests -- but '
            f'this box cannot witness the defect')
    assert max(deltas.values()) < 1e-12, (
        f'coefficients moved by {max(deltas.values()):.3e} across BLAS widths; '
        f'that is far beyond a reduction-order difference and means one of '
        f'these solves is wrong, not merely ordered differently')


def test_the_worker_evaluates_the_shipped_fit_and_never_re_fits():
    """The mechanism, at the worker's own entry point.

    ``_newton_invert_chunk`` IS the spawn worker's body.  With the fit shipped
    it must not reach the least-squares solver at all -- not "reach it and get
    the same answer", because whether it gets the same answer is precisely the
    thing that depends on a BLAS regime it does not control.  The keyless arm
    is the FAIL-BEFORE: that is the old worker, and it solves three times.
    """
    p = _fit_payload(n=64)
    xa, ya = _chunk_pts(24)
    calls = []
    orig = LT._solve_lstsq_thread_safe

    # ``**kw``: the real solver takes ``deterministic=`` (D14/D15), and a
    # stub that refuses the caller's keyword tests the stub, not the library.
    def _spy(A, b, **kw):
        calls.append(np.shape(A))
        return orig(A, b, **kw)

    LT._solve_lstsq_thread_safe = _spy
    try:
        shipped = _shipped_payload(p)
        n_before = len(calls)          # the parent's own three, built above
        opl_ship, _ = LT._newton_invert_chunk((shipped, xa, ya))
        assert len(calls) == n_before, (
            f'the worker ran {len(calls) - n_before} least-squares solves on a '
            f'payload that already carries the fit; it must EVALUATE the '
            f"parent's coefficients, not re-derive them")
        # ...and the historical payload still behaves as it did: three solves,
        # one per evaluator.  This is the pre-fix worker, and the reason CI
        # failed.
        old = dict(p)
        old.pop('cheb_fit', None)
        n_before = len(calls)
        opl_old, _ = LT._newton_invert_chunk((old, xa, ya))
        assert len(calls) - n_before == 3, (
            f'a payload with no cheb_fit key ran {len(calls) - n_before} '
            f'solves; the backwards-compatible path must still rebuild all '
            f'three evaluators (see the newton_fit / newton_max_iters '
            f'tolerances beside it)')
    finally:
        LT._solve_lstsq_thread_safe = orig
    # same numbers, both ways, on a box where the two BLAS regimes agree
    assert np.allclose(np.nan_to_num(opl_ship), np.nan_to_num(opl_old),
                       rtol=0.0, atol=1e-12)


def test_the_shipped_fit_is_evaluated_identically_at_any_blas_width():
    """THE CONTRACT, unconditional: the answer a worker returns may not depend
    on the BLAS width its interpreter happened to start at.

    Pre-fix this is the CI failure, measured end-to-end on the contract's own
    shape at 1.370e-11 for a worker at width 4 against a parent at width 24 --
    i.e. the 1.341e-11 / 1.358e-11 that
    ``test_pool_result_is_bit_identical_to_serial[polynomial]`` reported.  Here
    it is asserted at the worker entry point, where it costs milliseconds.
    """
    widths = _blas_widths()
    if not widths:
        pytest.skip('threadpoolctl absent: this build cannot vary BLAS width')
    from threadpoolctl import threadpool_limits
    p = _shipped_payload(_blas_payload())
    xa, ya = _chunk_pts(48)
    ref = None
    for w in widths:
        with threadpool_limits(limits=w):
            opl, _ = LT._newton_invert_chunk((p, xa, ya))
        if ref is None:
            ref = opl
            continue
        assert np.array_equal(np.nan_to_num(ref), np.nan_to_num(opl)), (
            f'the worker answered differently at BLAS width {w}, max|delta| = '
            f'{np.abs(np.nan_to_num(opl - ref)).max():.3e}.  The shipped fit '
            f'is supposed to remove every BLAS-dependent step from this path')


def test_from_state_reproduces_the_parents_evaluator_bitwise():
    """The round trip that the whole fix rests on: state -> pickle -> evaluator
    must give BYTE-IDENTICAL evaluation to the object it came from, and must
    perform no fit of its own."""
    import pickle
    p = _fit_payload(n=64)
    xa, ya = _chunk_pts(32)
    parent = LT._Cheb2DEvaluator(p['xs_in'], p['xs_in'], p['opl_grid'],
                                 order=6, xp=np, backend='numpy')
    state = pickle.loads(pickle.dumps(LT._cheb_fit_state(parent)))
    calls = []
    orig = LT._solve_lstsq_thread_safe
    LT._solve_lstsq_thread_safe = (
        lambda A, b, **kw: calls.append(1) or orig(A, b, **kw))
    try:
        child = LT._Cheb2DEvaluator.from_state(state, xp=np, backend='numpy')
    finally:
        LT._solve_lstsq_thread_safe = orig
    assert calls == [], 'from_state ran a least-squares solve'
    assert np.array_equal(np.asarray(parent.coeffs), np.asarray(child.coeffs))
    for a, b in zip(parent.ev_value_and_grad(xa, ya),
                    child.ev_value_and_grad(xa, ya)):
        assert np.array_equal(a, b), 'from_state does not evaluate identically'
    # the backend is a fact about WHERE it runs, so it is passed in rather than
    # carried in the state -- and it is validated on this path too
    assert 'backend' not in state
    with pytest.raises(ValueError, match='backend'):
        LT._Cheb2DEvaluator.from_state(state, xp=np, backend='numexpr')


def test_the_shipped_state_carries_the_fit_and_not_the_grids():
    """What travels, and what does not.

    The state is coefficients + multi-indices + the normalisation domain: the
    things an EVALUATION needs.  It must not re-ship the sample grids -- the
    payload already carries those for the spline path, and the per-worker
    memory model (``_newton_worker_bytes``) prices exactly one copy of them.
    A regression that shipped a second copy would silently move the clamp.
    """
    p = _blas_payload()
    ev = LT._Cheb2DEvaluator(p['xs_in'], p['xs_in'], p['opl_grid'], order=6,
                             xp=np)
    state = LT._cheb_fit_state(ev)
    assert set(state) == {'order', 'mi', 'coeffs', 'K1', 'K2',
                          'xmin', 'xmax', 'ymin', 'ymax'}
    n_terms = len(state['mi'])
    assert state['coeffs'].shape == state['K1'].shape == (n_terms,)
    import pickle
    n_bytes = len(pickle.dumps(
        LT._cheb_fit_payload(ev, ev, ev, newton_fit='polynomial')))
    grid_bytes = p['opl_grid'].nbytes
    assert n_bytes < 16_384 and n_bytes < grid_bytes / 20, (
        f'the shipped fit is {n_bytes} B against {grid_bytes} B of ONE grid '
        f'the payload already carries; it is meant to be free')
    # spline has no Chebyshev fit to ship: its worker rebuilds a single-threaded
    # FITPACK spline, which has no BLAS width to disagree about (and is why
    # [spline] passed on CI in every lane it ran in)
    assert LT._cheb_fit_payload(ev, ev, ev, newton_fit='spline') is None


def test_a_truncated_fit_state_is_refused_not_broadcast():
    """A payload that lost a coefficient must fail loudly.  Silent broadcasting
    would produce a plausible field from the wrong polynomial, which is the
    exact failure class this whole file is about."""
    p = _fit_payload(n=48)
    ev = LT._Cheb2DEvaluator(p['xs_in'], p['xs_in'], p['opl_grid'], order=6,
                             xp=np)
    good = LT._cheb_fit_state(ev)
    for key in ('coeffs', 'K1', 'K2'):
        bad = dict(good)
        bad[key] = np.asarray(good[key])[:-1]
        with pytest.raises(ValueError, match='inconsistent fit state'):
            LT._Cheb2DEvaluator.from_state(bad, xp=np)


def test_the_dispatch_path_ships_the_built_fit():
    """Wiring pin, same reasoning as its two siblings above: on a box whose
    parent and workers share a BLAS regime, bit-identity cannot notice that the
    fit stopped travelling -- which is every box this defect did not reproduce
    on.  So pin that the dispatch still ships it, from the SERIAL closure's own
    evaluators, before the payload is chunked."""
    body = _dispatch_closure_source()
    assert ("_spline_data['cheb_fit'] = _cheb_fit_payload(Sx, Sy, So, "
            'newton_fit)' in body), (
        "the Newton payload no longer ships the parent's built Chebyshev fit, "
        'so every worker re-solves the least squares in its own interpreter '
        'and pool == serial is conditional on the two sharing a BLAS width')
    # Same anchor move as its sibling above: the payload's contents are frozen
    # for the wire by ``_newton_payload_blob`` since FIX_PERF_ROUND2 item 3,
    # not by the old ``args_list`` comprehension.
    assert body.index("_spline_data['cheb_fit']") < body.index(
        '_newton_payload_blob(_spline_data)'), (
        'the fit is attached after the payload has already been serialised '
        'for the workers')
    wsrc = inspect.getsource(LT._newton_invert_chunk)
    assert "knot_data.get('cheb_fit', None)" in wsrc, (
        'the worker no longer reads the shipped fit')
    assert '_Cheb2DEvaluator.from_state(' in wsrc, (
        'the worker reads the shipped fit but does not construct from it')
    # ...and the keyless tolerance stays, exactly as it does for newton_fit,
    # newton_max_iters and cheb_backend
    assert '_Cheb2DEvaluator(xs_in, xs_in, x_out_grid' in wsrc, (
        'the historical rebuild path is gone, so a payload written before this '
        'key existed can no longer run')


_PROBE = '''
import multiprocessing as mp, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from lumenairy.elements import _lens_traced as LT
from threadpoolctl import threadpool_limits


def probe(payload):
    """Runs in the SPAWNED worker."""
    n = []
    orig = LT._solve_lstsq_thread_safe
    LT._solve_lstsq_thread_safe = (
        lambda A, b, **kw: n.append(1) or orig(A, b, **kw))
    try:
        ev = LT._Cheb2DEvaluator.from_state(payload['cheb_fit']['opl'], xp=np,
                                            backend='numpy')
        n_ship = len(n)
        refit = LT._Cheb2DEvaluator(payload['xs_in'], payload['xs_in'],
                                    payload['opl_grid'], order=6, xp=np,
                                    backend='numpy')
    finally:
        LT._solve_lstsq_thread_safe = orig
    return (np.asarray(ev.coeffs), np.asarray(refit.coeffs), n_ship)


def main():
    xs = np.linspace(-1e-3, 1e-3, 279)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    r2 = X ** 2 + Y ** 2
    g = 1.0e-3 + 5.0e2 * r2 + 2.1e8 * r2 * r2 + 3.0e2 * X * r2
    # The PARENT fits under a BLAS cap the freshly spawned child will not have
    # -- process-global on OpenBLAS, and exactly the asymmetry CI hits.
    with threadpool_limits(limits=1):
        ev = LT._Cheb2DEvaluator(xs, xs, g, order=6, xp=np, backend='numpy')
    payload = {'xs_in': xs, 'opl_grid': g,
               'cheb_fit': {'opl': LT._cheb_fit_state(ev)}}
    with ProcessPoolExecutor(max_workers=1,
                             mp_context=mp.get_context('spawn')) as ex:
        c_ship, c_refit, n_solves = ex.submit(probe, payload).result()
    parent = np.asarray(ev.coeffs)
    print('SHIPPED_IDENTICAL', bool(np.array_equal(parent, c_ship)))
    print('WORKER_SOLVES', n_solves)
    print('REFIT_IDENTICAL', bool(np.array_equal(parent, c_refit)))
    print('REFIT_DELTA', '%.3e' % float(np.abs(parent - c_refit).max()))


if __name__ == '__main__':
    main()
'''


@pytest.mark.slow
def test_a_real_spawned_worker_uses_the_parents_fit(tmp_path):
    """The probe the emulations stand in for: a REAL spawned worker, with the
    parent's fit built under a BLAS cap the child does not inherit.

    Two assertions, both unconditional:
      * the worker's evaluator holds the parent's coefficients, bit for bit;
      * it ran ZERO least-squares solves to get them.

    The third line it prints -- whether the child's own REBUILD matches -- is
    the fail-before, reported rather than asserted: on a box whose BLAS reduces
    identically at every width it legitimately matches, and that is the box
    this defect never reproduced on.
    """
    pytest.importorskip('threadpoolctl')
    script = tmp_path / 'probe_fit.py'
    script.write_text(_PROBE)
    out = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr + out.stdout
    got = {}
    for ln in out.stdout.splitlines():
        parts = ln.split(' ', 1)
        if len(parts) == 2 and parts[0].isupper():
            got[parts[0]] = parts[1].strip()
    assert got.get('SHIPPED_IDENTICAL') == 'True', (
        f'a spawned worker built from the shipped state does not hold the '
        f"parent's coefficients: {out.stdout}")
    assert got.get('WORKER_SOLVES') == '0', (
        f'the worker ran a least-squares solve to construct from a state that '
        f'already carries the answer: {out.stdout}')
    print(f'  [probe] worker re-fit matches parent: '
          f'{got.get("REFIT_IDENTICAL")} (delta {got.get("REFIT_DELTA")})')

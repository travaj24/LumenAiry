"""VERIFY WP-B13 follow-ups -- the mutation matrix.

A pytest PLUGIN (``-p``) that puts one chosen regression back in memory at
collection time, plus a DRIVER (``--drive``) that runs every mutation on this
build and tabulates which tests caught it.

Every mutation is a *silent* regression: it changes behaviour that no field
value reflects, so bit-identity cannot see it.  A mutation that no test
catches is a gap; a mutation that makes a test HANG rather than fail is also
a gap, so every arm runs under ``pytest-timeout``'s ``--timeout`` and a
subprocess deadline on top of it.

Mutations
  m1_dump_stringio     `_thread_dump` back to `io.StringIO` (D1)
  m2_census_neverdrop  `_shutdown_pool_bounded`: append, never remove (D2, pre)
  m3_census_oneline    the one-line repair: remove, THEN publish (D2, adverse)
  m4_census_noappend   never append at all (the census stops being a census)
  m5_detector_calls    the join detector back to the `ast.Call`-only sweep (D7)
  m6_detector_nowait   the detector ignores `wait=False` (false positives)
  m7_inflight_perchunk `_note_pool_inflight` claims once per CHUNK (D4)
  m8_inflight_never    the dispatcher never claims the pool at all
  m9_helper_no_finally the helper publishes `done` OUTSIDE the `finally`
  m10_lazy_eager       the pool pre-spawns its workers (D3's lazy-spawn fact)

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=<tree> python validation/probe_verify_b13_followups/\
vf8_mutations.py --drive --out vf8_win.json
"""
from __future__ import annotations

import faulthandler
import io
import json
import os
import subprocess
import sys
import threading
import time

MUTATIONS = (
    'm1_dump_stringio', 'm2_census_neverdrop', 'm3_census_oneline',
    'm4_census_noappend', 'm5_detector_calls', 'm6_detector_nowait',
    'm7_inflight_perchunk', 'm8_inflight_never', 'm9_helper_no_finally',
    'm10_lazy_eager',
)

_TARGET_FILES = (
    'tests/unit/test_fix_newton_pool_broken_fallback.py',
    'tests/unit/test_verify_b13_followups.py',
)

# The subset of the two files that could possibly decide any of these
# mutations: the dump helper, the census, the in-flight counter, the join
# detector, the lazy-spawn fact and the timeout clause.  Narrowing keeps each
# arm under a couple of minutes; the BASELINE arm is run with the same
# selection, so "caught" means "caught by a test that was actually run".
_SELECT = ('dump or wedge or census or expir or teardown or helper or '
           'in_flight or magnitude or detector or joins_an_executor or '
           'sibling or surplus or bootstrap_bar or '
           'timeout_on_the_dispatch or claim')


# ==========================================================================
# the injections
# ==========================================================================
def _stringio_thread_dump():
    buf = io.StringIO()
    faulthandler.dump_traceback(file=buf, all_threads=True)
    return buf.getvalue()


def _make_bounded(LT, flavour):
    """A replacement ``_shutdown_pool_bounded`` with one property broken."""
    def _bounded(ex, timeout=None):
        if ex is None:
            return True
        if timeout is None:
            timeout = LT._POOL_SHUTDOWN_TIMEOUT
        done = threading.Event()

        def _run():
            try:
                ex.shutdown(wait=True)
            except (RuntimeError, OSError, ValueError):
                pass
            finally:
                if flavour == 'neverdrop':
                    done.set()
                elif flavour == 'oneline':
                    with LT._ABANDONED_POOLS_LOCK:
                        try:
                            LT._ABANDONED_POOLS.remove(ex)
                        except ValueError:
                            pass
                    done.set()
                elif flavour == 'noappend':
                    done.set()
                    with LT._ABANDONED_POOLS_LOCK:
                        try:
                            LT._ABANDONED_POOLS.remove(ex)
                        except ValueError:
                            pass

        def _run_no_finally():
            try:
                ex.shutdown(wait=True)
            except (RuntimeError, OSError, ValueError):
                pass
            # NO finally: an uncaught class never publishes `done`
            done.set()
            with LT._ABANDONED_POOLS_LOCK:
                try:
                    LT._ABANDONED_POOLS.remove(ex)
                except ValueError:
                    pass

        target = _run_no_finally if flavour == 'no_finally' else _run
        try:
            threading.Thread(target=target,
                             name='lumenairy-newton-pool-close',
                             daemon=True).start()
        except RuntimeError:
            LT._abandon_pool(ex)
            return False
        if done.wait(timeout):
            return True
        with LT._ABANDONED_POOLS_LOCK:
            if flavour not in ('neverdrop', 'oneline', 'noappend',
                               'no_finally'):
                if done.is_set():
                    return True
            LT._POOL_SHUTDOWN_TIMEOUTS += 1
            if flavour != 'noappend':
                LT._ABANDONED_POOLS.append(ex)
        return False
    return _bounded


def _shipped_detector_factory():
    import ast

    def _det(src, exempt=('_shutdown_pool_bounded',)):
        tree = ast.parse(src)
        owner, enclosing = {}, {}
        for parent in ast.walk(tree):
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(parent):
                    if child is not parent:
                        owner[child] = parent.name
                        enclosing.setdefault(child, set()).add(parent.name)
        offenders = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'shutdown'):
                continue
            waits = [kw.value for kw in node.keywords if kw.arg == 'wait']
            joining = (not waits) or any(
                not (isinstance(v, ast.Constant) and v.value is False)
                for v in waits)
            if joining and not (set(exempt) & enclosing.get(node, set())):
                offenders.append(f'{owner.get(node)}:{node.lineno}:shutdown')
        return sorted(offenders), []
    return _det


def _nowait_detector_factory(real):
    def _det(src, exempt=('_shutdown_pool_bounded',)):
        off, info = real(src, exempt=exempt)
        # a detector that no longer exempts wait=False: add every shutdown
        import ast
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'shutdown'):
                tag = f'?:{node.lineno}:shutdown'
                if not any(o.endswith(f':{node.lineno}:shutdown')
                           for o in off):
                    off.append(tag)
        return sorted(off), info
    return _det


def _mutation():
    return os.environ.get('VF8_MUTATE', '').strip()


def pytest_configure(config):                      # noqa: ARG001 -- plugin
    """LIBRARY-side mutations, installed before collection."""
    which = _mutation()
    if not which:
        return
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    from lumenairy.elements import _lens_traced as LT

    if which == 'm2_census_neverdrop':
        LT._shutdown_pool_bounded = _make_bounded(LT, 'neverdrop')
    elif which == 'm3_census_oneline':
        LT._shutdown_pool_bounded = _make_bounded(LT, 'oneline')
    elif which == 'm4_census_noappend':
        LT._shutdown_pool_bounded = _make_bounded(LT, 'noappend')
    elif which == 'm9_helper_no_finally':
        LT._shutdown_pool_bounded = _make_bounded(LT, 'no_finally')
    elif which == 'm7_inflight_perchunk':
        _real = LT._note_pool_inflight

        def _per_chunk(delta):
            """One claim per CHUNK: eight movements where there was one."""
            out = _real(delta)
            for _ in range(7):
                out = _real(delta)
            return out
        LT._note_pool_inflight = _per_chunk
    elif which == 'm8_inflight_never':
        LT._note_pool_inflight = lambda delta: 0
    elif which == 'm10_lazy_eager':
        _real_get = LT._get_persistent_worker_pool

        def _eager(n_workers):
            ex = _real_get(n_workers)
            try:
                for _ in range(int(n_workers) + 1):
                    ex._adjust_process_count()
            except Exception:                      # noqa: BLE001
                pass
            return ex
        LT._get_persistent_worker_pool = _eager
    elif which in ('m1_dump_stringio', 'm5_detector_calls',
                   'm6_detector_nowait'):
        pass                                       # test-module side, below
    else:
        raise SystemExit(f'vf8: unknown mutation {which!r}')
    print(chr(10) + f'VF8 MUTATION ACTIVE (library): {which}', flush=True)


def pytest_collection_modifyitems(session, config, items):  # noqa: ARG001
    """TEST-module mutations, which have to run after the modules import."""
    which = _mutation()
    if which not in ('m1_dump_stringio', 'm5_detector_calls',
                     'm6_detector_nowait'):
        return
    patched = set()
    for item in items:
        mod = getattr(item, 'module', None)
        if mod is None or mod.__name__ in patched:
            continue
        if which == 'm1_dump_stringio' and hasattr(mod, '_thread_dump'):
            mod._thread_dump = _stringio_thread_dump
            patched.add(mod.__name__)
        elif which == 'm5_detector_calls' and hasattr(
                mod, '_unbounded_executor_joins'):
            mod._unbounded_executor_joins = _shipped_detector_factory()
            patched.add(mod.__name__)
        elif which == 'm6_detector_nowait' and hasattr(
                mod, '_unbounded_executor_joins'):
            mod._unbounded_executor_joins = _nowait_detector_factory(
                mod._unbounded_executor_joins)
            patched.add(mod.__name__)
    print(chr(10) + f'VF8 MUTATION ACTIVE (test module): {which} in '
          f'{sorted(patched)}', flush=True)


# ==========================================================================
# the driver
# ==========================================================================
def drive(args):
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    rows = []
    baseline = _one(root, args, None)
    rows.append(baseline)
    print('baseline', json.dumps(baseline), flush=True)
    for m in (args.only or MUTATIONS):
        r = _one(root, args, m)
        rows.append(r)
        print(m, json.dumps(r), flush=True)
    return rows


try:
    import pytest_timeout as _pt  # noqa: F401
    _HAS_PYTEST_TIMEOUT = True
except Exception:                                  # noqa: BLE001
    _HAS_PYTEST_TIMEOUT = False


def _one(root, args, mutation):
    env = dict(os.environ)
    env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[var] = '1'
    if mutation:
        env['VF8_MUTATE'] = mutation
    else:
        env.pop('VF8_MUTATE', None)
    probedir = os.path.dirname(os.path.abspath(__file__))
    env['PYTHONPATH'] = probedir + os.pathsep + env['PYTHONPATH']
    cmd = [sys.executable, '-m', 'pytest', *_TARGET_FILES, '-q',
           '-p', 'no:randomly', '-p', 'vf8_mutations',
           '-W', 'ignore::ResourceWarning', '-k', _SELECT,
           '-rf', '--no-header']
    # pytest-timeout is not installed in every venv this runs in (the WSL one
    # does not have it), and passing --timeout there makes pytest EXIT ON THE
    # COMMAND LINE -- which an earlier version of this driver read as "the
    # mutation was not caught".  The subprocess deadline below is the bound
    # that is always present; --timeout is added only when it exists.
    if _HAS_PYTEST_TIMEOUT:
        cmd.append(f'--timeout={args.timeout}')
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=root, env=env, timeout=args.deadline)
        timed_out = False
        out = proc.stdout
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        out = exc.stdout if isinstance(exc.stdout, str) else (
            (exc.stdout or b'').decode('utf-8', 'replace'))
    el = round(time.monotonic() - t0, 1)
    failed = sorted({ln.split(' ')[1].split('::')[-1]
                     for ln in out.splitlines()
                     if ln.startswith('FAILED ')})
    tail = [ln for ln in out.splitlines()
            if (' passed' in ln or ' failed' in ln or ' error' in ln
                or 'no tests ran' in ln)]
    ran = bool(tail)
    return {'mutation': mutation or '(none)',
            'wall_seconds': el,
            'driver_timed_out': timed_out,
            'pytest_timeout_plugin': _HAS_PYTEST_TIMEOUT,
            'pytest_actually_ran': ran,
            'failed_tests': failed,
            'n_failed': len(failed),
            # An arm whose pytest never produced a summary line is an ERROR,
            # never a "not caught": reading a failed invocation as a gap is
            # exactly the false-green this matrix exists to prevent.
            'caught': (None if not mutation
                       else ('ERROR -- pytest did not run' if not ran
                             else bool(failed))),
            'summary': tail[-1] if tail else '(no summary line)',
            'stdout_tail': '' if ran else out[-1200:]}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=None,
                    help='run only these mutations (plus the baseline)')
    ap.add_argument('--drive', action='store_true')
    ap.add_argument('--timeout', type=int, default=180)
    ap.add_argument('--deadline', type=float, default=900.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if not args.drive:
        print(__doc__)
        return 0
    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    rows = drive(args)
    out = {'python': sys.version, 'platform': sys.platform,
           'lumenairy_file': lumenairy.__file__,
           'rows': rows,
           'uncaught': [r['mutation'] for r in rows
                        if r['mutation'] != '(none)' and r['caught'] is False],
           'errored': [r['mutation'] for r in rows
                       if isinstance(r['caught'], str)],
           'hung': [r['mutation'] for r in rows if r['driver_timed_out']],
           'pytest_timeout_plugin': _HAS_PYTEST_TIMEOUT,
           'baseline_clean': (rows[0]['n_failed'] == 0
                              and rows[0]['pytest_actually_ran'])}
    if args.out:
        with open(args.out, 'w', encoding='cp1252', errors='replace') as fh:
            json.dump(out, fh, indent=1)
        print('wrote', args.out)
    print('UNCAUGHT', out['uncaught'])
    print('ERRORED', out['errored'])
    print('HUNG', out['hung'])
    print('baseline_clean', out['baseline_clean'],
          'pytest_timeout_plugin', _HAS_PYTEST_TIMEOUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())

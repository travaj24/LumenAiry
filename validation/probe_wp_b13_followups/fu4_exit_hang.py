"""FU4 -- what the interpreter-exit hang costs, and what would close it (D6).

VERIFY-WP-B13 D6, a MAINTAINER DECISION left open here: WP-B13 moved the
wedge from the MIDDLE of a computation (where the caller's result is lost) to
AFTER the call has returned its bit-identical serial answer -- but the
PROCESS still cannot exit.  Once CPython's `_ExecutorManagerThread` is stuck
in `_terminate_broken`, interpreter shutdown joins it twice over:
`concurrent.futures.process._python_exit`, registered through
`threading._register_atexit`, joins every thread in `_threads_wakeups`; and
`threading._shutdown` then joins every non-daemon thread's `_tstate_lock`.

Nothing in `lumenairy/` is changed by this probe.  The prototype below lives
HERE, behind `--daemonise on`, defaulting OFF, so the cost of the hang and
the effect of the candidate repair are both measured without moving a
default or shipping an unexercised branch.

MODES

  ``sigign``   POSIX only.  The chunk installs `SIG_IGN` for SIGTERM and
               holds; the probe then SIGKILLs exactly ONE worker, so the pool
               breaks and CPython tries to terminate the survivors --
               `p.terminate()` does nothing and `p.join()` is unbounded.
               This is the verification's NATURAL wedge, not a stub.
  ``control``  nothing is broken.  The same script, the same pool, the same
               teardown -- this is the gap the hang has to be measured
               against.

Run the harness (it launches the child and times it):

    PYTHONPATH=<tree> python validation/probe_wp_b13_followups/
    fu4_exit_hang.py --harness --out fu4_wsl.json
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import signal
import subprocess
import sys
import threading
import time
import warnings

# Literal constants only at top level: a spawn worker re-executes this module
# body, and the library REFUSES the pool for a script whose top level does
# real work (`_script_has_main_guard`).
T0 = 0.0


def emit(event, **kw):
    kw['event'] = event
    kw['t'] = round(time.monotonic() - T0, 4)
    sys.stdout.write(json.dumps(kw, sort_keys=True) + '\n')
    sys.stdout.flush()


def sigign_chunk(args):
    """Ignore SIGTERM, then hold: `p.terminate()` cannot end this worker."""
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except (ValueError, OSError, AttributeError):
        pass
    time.sleep(900.0)
    return args


def _singlet(ap=3e-3, r=9e-3):
    return {'name': 'fu4_singlet', 'aperture_diameter': ap,
            'thicknesses': [3e-3],
            'surfaces': [
                {'radius': r, 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'conic': 0.0, 'aspheric_coeffs': None},
                {'radius': -r, 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'conic': 0.0, 'aspheric_coeffs': None}]}


def _traced(la, n_workers, N):
    import numpy as np
    ap = 3e-3
    dx = 2.2 * ap / N
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / 1.2e-3 ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E0, prescription=_singlet(ap), wavelength=1.31e-6, dx=dx,
            ray_subsample=2, newton_fit='spline', n_workers=n_workers,
            parallel_amp=False, newton_amp_mask_rel=0.0,
            on_undersample='silent', on_aperture_beam='silent'))


# ---------------------------------------------------------------------------
# THE PROTOTYPE -- a switch, default OFF, NOT in the library
# ---------------------------------------------------------------------------

def install_daemonising_abandon(LT, record):
    """Wrap `_abandon_pool` so a retired pool cannot hold interpreter EXIT.

    Three steps, because CPython joins the manager thread twice:

      1. drop the executor from `concurrent.futures.process._threads_wakeups`
         so `_python_exit` does not join it;
      2. mark the manager thread daemonic (`_daemonic`, because the public
         setter refuses a started thread) so `threading._shutdown` skips it;
      3. discard its `_tstate_lock` from `threading._shutdown_locks`, which
         was added at start time while the thread was still non-daemon and
         is NOT removed by step 2.

    All three are private CPython attributes.  That is the honest cost of
    this repair and the reason it is measured here rather than shipped: it
    is a maintainer decision, not a defect fix.
    """
    import concurrent.futures.process as cfp

    real = LT._abandon_pool

    def _abandon(ex):
        note = {'threads_wakeups_popped': False, 'daemonised': False,
                'shutdown_lock_discarded': False, 'errors': []}
        mgr = getattr(ex, '_executor_manager_thread', None)
        try:
            if mgr is not None:
                note['threads_wakeups_popped'] = (
                    cfp._threads_wakeups.pop(mgr, None) is not None)
                mgr._daemonic = True
                note['daemonised'] = True
                lock = getattr(mgr, '_tstate_lock', None)
                if lock is not None:
                    before = len(threading._shutdown_locks)
                    threading._shutdown_locks.discard(lock)
                    note['shutdown_lock_discarded'] = (
                        len(threading._shutdown_locks) < before)
        except Exception as exc:                  # noqa: BLE001 -- reported
            note['errors'].append(f'{type(exc).__name__}: {exc}')
        record.append(note)
        return real(ex)

    LT._abandon_pool = _abandon


def kill_surviving_children():
    """SIGKILL every multiprocessing child still alive.

    The DIAGNOSTIC arm.  If removing the surviving workers lets the process
    exit while the daemonising prototype does not, then the join that holds
    the interpreter is ``multiprocessing.util._exit_function``'s
    ``for p in active_children(): p.join()`` -- the workers themselves --
    and not the executor manager thread the prototype targets.
    """
    import multiprocessing as mp
    killed = []
    for p in mp.active_children():
        try:
            os.kill(p.pid, signal.SIGKILL)
            killed.append(p.pid)
        except (OSError, AttributeError, ValueError):
            pass
    return killed


def killer_thread(LT, stop, out):
    """SIGKILL exactly ONE worker once the pool has real children."""
    end = time.monotonic() + 120.0
    while not stop.is_set() and time.monotonic() < end:
        ex = LT._PERSISTENT_POOL
        procs = list(getattr(ex, '_processes', {}).values()) if ex else []
        alive = [p for p in procs if p.is_alive()]
        if len(alive) >= 2:
            time.sleep(1.0)                 # let a chunk reach the worker
            victim = alive[0]
            out['victim_pid'] = victim.pid
            out['workers_at_kill'] = len(alive)
            try:
                os.kill(victim.pid, signal.SIGKILL)
            except (OSError, AttributeError, ValueError) as exc:
                out['kill_error'] = str(exc)
            emit('worker_killed', pid=victim.pid, of=len(alive))
            return
        time.sleep(0.05)
    emit('killer_gave_up')


def child_main(a):
    global T0
    T0 = time.monotonic()
    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    emit('import', lumenairy=la.__file__, python=sys.version.split()[0],
         pid=os.getpid(), mode=a.mode, daemonise=a.daemonise)
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1

    record: list = []
    if a.daemonise in ('on', 'both'):
        install_daemonising_abandon(LT, record)

    LT.close_worker_pool()
    ref = _traced(la, 1, a.N)
    emit('reference_done')

    out = {'mode': a.mode, 'daemonise': a.daemonise, 'workers': a.workers}
    stop = threading.Event()
    if a.mode == 'sigign':
        LT._newton_invert_chunk = sigign_chunk
        threading.Thread(target=killer_thread, args=(LT, stop, out),
                         name='fu4-killer', daemon=True).start()

    import numpy as np
    emit('dispatch_enter')
    t = time.monotonic()
    got = _traced(la, a.workers, a.N)
    out['call_seconds'] = round(time.monotonic() - t, 3)
    stop.set()
    out['identical'] = bool(np.array_equal(got, ref))
    out['max_delta'] = float(np.abs(got - ref).max())
    out['shutdown_timeouts'] = int(LT._POOL_SHUTDOWN_TIMEOUTS)
    out['abandoned_len'] = len(LT._ABANDONED_POOLS)
    out['inflight'] = int(LT._POOL_INFLIGHT)
    out['daemonise_record'] = record
    if a.daemonise in ('kill', 'both'):
        out['killed_children'] = kill_surviving_children()
    emit('dispatch_returned', **out)
    # Dump every thread a while AFTER the exit line, so whatever join is
    # holding the interpreter is READ rather than inferred.  exit=False:
    # killing the process here would destroy the measurement.
    faulthandler.enable()
    faulthandler.dump_traceback_later(a.dump_after, exit=False)
    emit('exiting')
    return 0


def run_one(mode, daemonise, deadline, workers, N):
    """Launch the child and time BOTH its answer and its process exit."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..', '..'))
    env = dict(os.environ)
    env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS'):
        env[var] = '1'
    cmd = [sys.executable, '-u', os.path.abspath(__file__),
           '--mode', mode, '--daemonise', daemonise,
           '--workers', str(workers), '--N', str(N),
           '--dump-after', str(max(5.0, deadline * 0.4))]
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=root, env=env, text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    lines, row = [], {'mode': mode, 'daemonise': daemonise,
                      'deadline_seconds': deadline}
    reader_done = threading.Event()

    def _read():
        for line in proc.stdout:
            lines.append(line.rstrip())
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get('event') == 'dispatch_returned':
                row['dispatch'] = ev
                row['t_dispatch_returned'] = round(time.monotonic() - t0, 3)
            elif ev.get('event') == 'exiting':
                row['t_exiting'] = round(time.monotonic() - t0, 3)
        reader_done.set()

    threading.Thread(target=_read, daemon=True).start()
    try:
        proc.wait(timeout=deadline)
        row['exited_on_its_own'] = True
        row['returncode'] = proc.returncode
    except subprocess.TimeoutExpired:
        row['exited_on_its_own'] = False
        row['returncode'] = None
        proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    row['t_process_gone'] = round(time.monotonic() - t0, 3)
    if 't_exiting' in row:
        row['exit_hang_seconds'] = round(
            row['t_process_gone'] - row['t_exiting'], 3)
    reader_done.wait(5.0)
    row['lines'] = lines[-8:]
    row['thread_dump'] = [ln for ln in lines
                          if not ln.startswith('{')][:120]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--harness', action='store_true')
    ap.add_argument('--mode', default='sigign',
                    choices=('sigign', 'control'))
    ap.add_argument('--daemonise', default='off',
                    choices=('off', 'on', 'kill', 'both'),
                    help='off = measure the hang; on = the daemonising '
                         'prototype; kill = SIGKILL the surviving workers '
                         'after the answer is in; both = both')
    ap.add_argument('--dump-after', type=float, default=25.0,
                    help='seconds after the exit line to dump every thread, '
                         'so the blocking join is READ rather than guessed')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--N', type=int, default=256)
    ap.add_argument('--deadline', type=float, default=120.0)
    ap.add_argument('--out', default='fu4.json')
    a = ap.parse_args()

    if not a.harness:
        return child_main(a)

    out = {'python': sys.version, 'platform': sys.platform, 'runs': []}
    plan = [('control', 'off', 60.0),
            ('sigign', 'off', a.deadline),
            ('sigign', 'on', a.deadline),
            ('sigign', 'kill', a.deadline),
            ('sigign', 'both', a.deadline)]
    for mode, daemonise, deadline in plan:
        print(f'--- {mode} daemonise={daemonise} ---', flush=True)
        row = run_one(mode, daemonise, deadline, a.workers, a.N)
        print(json.dumps(row, indent=1), flush=True)
        out['runs'].append(row)
    here = os.path.dirname(os.path.abspath(__file__))
    path = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('WROTE', path, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

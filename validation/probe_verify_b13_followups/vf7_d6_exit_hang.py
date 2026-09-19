"""VERIFY WP-B13 follow-ups, D6 -- the interpreter-exit hang, reproduced.

Five arms, run as CHILD processes under a hard deadline by
``--drive`` (the parent).  Each child runs one traced-lens call on a pool it
then wedges, prints an EXIT line, and returns; the parent times how long the
process takes to actually disappear after that line.

My wedge differs from ``fu4_exit_hang.py``'s in how the worker is parked.
There the chunk installs ``SIG_IGN`` for SIGTERM and holds.  Here the chunk
installs a SIGTERM HANDLER that is a no-op (so the signal is delivered,
caught and ignored -- a different kernel path from SIG_IGN, and the shape a
real worker with a cleanup handler would have) and then blocks in
``signal.pause``-equivalent ``time.sleep`` loop.  One worker is then SIGKILLed
so the pool marks itself broken and CPython's ``_terminate_broken`` runs its
unbounded ``p.terminate()`` / ``p.join()`` over the survivors.

  control          nothing broken
  handler          the wedge, nothing else
  handler+daemon   the wedge plus the three-step daemonising prototype
  handler+kill     the wedge, surviving workers SIGKILLed after the answer
  handler+both     both

``faulthandler`` is armed in the child AFTER its exit line so the parent can
read WHERE the main thread is parked, without the dump ending the process.

    wsl -e bash -lc 'cd /mnt/c/tmp/lum_vpool2 && OMP_NUM_THREADS=1 ... \
      python validation/probe_verify_b13_followups/vf7_d6_exit_hang.py \
      --drive --deadline 60 --out vf7_wsl.json'
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import warnings


# ==========================================================================
# worker-side
# ==========================================================================
def _wedge_chunk(_payload):
    """Install a no-op SIGTERM HANDLER and never return."""
    import signal as _sig
    import time as _t

    def _swallow(_s, _f):
        return None

    try:
        _sig.signal(_sig.SIGTERM, _swallow)
    except Exception:                              # noqa: BLE001
        pass
    while True:
        _t.sleep(3600)


def _fast_singlet(ap=3e-3, r=9e-3):
    return {'name': 'fast_singlet', 'aperture_diameter': ap,
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
            E0, prescription=_fast_singlet(ap), wavelength=1.31e-6, dx=dx,
            ray_subsample=2, newton_fit='spline', n_workers=n_workers,
            parallel_amp=False, newton_amp_mask_rel=0.0,
            on_undersample='silent', on_aperture_beam='silent'))


def daemonise_manager(ex):
    """The three-step prototype, applied to a LIVE executor.

    Returns which steps actually took effect, so an arm cannot silently be
    a no-op arm.
    """
    import concurrent.futures.process as cfp
    import threading
    rec = {}
    mgr = getattr(ex, '_executor_manager_thread', None)
    rec['manager_thread_found'] = mgr is not None
    if mgr is None:
        return rec
    rec['step1_removed_from_threads_wakeups'] = bool(
        cfp._threads_wakeups.pop(mgr, None) is not None)
    try:
        mgr._daemonic = True
        rec['step2_daemonic_set'] = bool(mgr.daemon)
    except Exception as exc:                       # noqa: BLE001
        rec['step2_daemonic_set'] = f'failed: {exc}'
    lock = getattr(mgr, '_tstate_lock', None)
    before = lock in threading._shutdown_locks if lock is not None else None
    if lock is not None:
        threading._shutdown_locks.discard(lock)
    rec['step3_tstate_lock_was_registered'] = before
    rec['step3_tstate_lock_now_registered'] = (
        lock in threading._shutdown_locks if lock is not None else None)
    return rec


def kill_survivors(ex):
    killed = []
    for p in list(getattr(ex, '_processes', {}).values()):
        try:
            if p.is_alive():
                os.kill(p.pid, signal.SIGKILL)
                killed.append(p.pid)
        except Exception:                          # noqa: BLE001
            pass
    return killed


def run_child(mode, N=256, workers=6):
    """One arm, in THIS process.  Prints a JSON line then an EXIT line."""
    import numpy as np

    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1
    print('lumenairy.__file__ =', la.__file__, flush=True)

    LT.close_worker_pool()
    ref = _traced(la, 1, N)

    out = {'mode': mode, 'workers': workers, 'N': N}
    ex = LT._get_persistent_worker_pool(workers)
    rec = {}
    if mode != 'control':
        # park every worker in the no-op-SIGTERM-handler frame
        for i in range(workers):
            ex.submit(_wedge_chunk, i)
        time.sleep(4.0)
        pids = [p.pid for p in ex._processes.values()]
        out['worker_pids'] = pids
        # SIGKILL exactly one, so the pool marks itself broken and CPython's
        # _terminate_broken runs its unbounded join over the survivors
        if pids:
            os.kill(pids[0], signal.SIGKILL)
            out['sigkilled_one'] = pids[0]
        time.sleep(2.0)
        out['pool_is_broken'] = bool(LT._pool_is_broken(ex))

    t0 = time.monotonic()
    got = _traced(la, workers, N)
    out['library_call_seconds'] = round(time.monotonic() - t0, 3)
    out['identical'] = bool(np.array_equal(got, ref))
    out['max_abs_delta'] = float(np.nanmax(np.abs(
        np.asarray(got) - np.asarray(ref))))
    out['abandoned_len'] = len(LT._ABANDONED_POOLS)
    out['shutdown_timeouts'] = LT._POOL_SHUTDOWN_TIMEOUTS

    if 'daemon' in mode or mode == 'handler+both':
        rec = daemonise_manager(ex)
    if 'kill' in mode or mode == 'handler+both':
        out['killed'] = kill_survivors(ex)
    out['daemonise_record'] = rec

    out['main_thread_ident'] = threading.main_thread().ident
    out['manager_thread_ident'] = getattr(
        getattr(ex, '_executor_manager_thread', None), 'ident', None)
    print('RESULT ' + json.dumps(out), flush=True)
    print('EXITLINE %.6f' % time.monotonic(), flush=True)
    sys.stdout.flush()
    # arm faulthandler so the parent can see WHERE we are parked if we hang.
    # The thread IDS above are what lets the parent pick the MAIN thread's
    # section out of the dump: faulthandler prints `Thread 0x<hex>` with no
    # name, so matching on the identifier is the only reliable way.
    import faulthandler
    faulthandler.dump_traceback_later(12.0, repeat=True, exit=False,
                                      file=sys.stderr)
    return 0


# ==========================================================================
# parent
# ==========================================================================
def drive(args):
    rows = []
    for mode in ('control', 'handler', 'handler+daemon', 'handler+kill',
                 'handler+both'):
        cmd = [sys.executable, os.path.abspath(__file__), '--mode', mode,
               '--n', str(args.n), '--workers', str(args.workers)]
        env = dict(os.environ)
        env.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1'})
        t0 = time.monotonic()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env)
        out_lines, err_lines = [], []
        exit_at = None
        result = None
        deadline = t0 + args.deadline
        # read stdout until EOF or deadline
        import threading as _th

        def _pump(stream, sink):
            for line in stream:
                sink.append(line)

        to = _th.Thread(target=_pump, args=(p.stdout, out_lines), daemon=True)
        te = _th.Thread(target=_pump, args=(p.stderr, err_lines), daemon=True)
        to.start()
        te.start()
        rc = None
        while time.monotonic() < deadline:
            rc = p.poll()
            if rc is not None:
                break
            time.sleep(0.05)
        gone_at = time.monotonic()
        killed = False
        if rc is None:
            p.kill()
            killed = True
            try:
                p.wait(20)
            except Exception:                      # noqa: BLE001
                pass
            gone_at = time.monotonic()
        to.join(3.0)
        te.join(3.0)
        for line in out_lines:
            if line.startswith('RESULT '):
                result = json.loads(line[len('RESULT '):])
            if line.startswith('EXITLINE '):
                exit_at = float(line.split()[1])
        # the child's monotonic clock is its own; use the parent's
        # observation of when the EXITLINE arrived instead
        blob = ''.join(err_lines)
        row = {'mode': mode,
               'process_gone_after_seconds': round(gone_at - t0, 3),
               'sigkilled_at_deadline': killed,
               'returncode': rc if rc is not None else 'KILLED',
               'library_call_seconds': (result or {}).get(
                   'library_call_seconds'),
               'identical': (result or {}).get('identical'),
               'max_abs_delta': (result or {}).get('max_abs_delta'),
               'pool_is_broken': (result or {}).get('pool_is_broken'),
               'daemonise_record': (result or {}).get('daemonise_record'),
               'killed_survivors': (result or {}).get('killed'),
               'reached_exit_line': exit_at is not None,
               'stderr_has_dump': 'Timeout (' in blob,
               'main_thread_stack': _stack_for(
                   blob, (result or {}).get('main_thread_ident')),
               'manager_thread_stack': _stack_for(
                   blob, (result or {}).get('manager_thread_ident')),
               'raw_first_dump': _all_stacks(blob)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        # sweep anything this arm left behind
        row['swept'] = sweep()
    return rows


def _stack_for(blob, ident):
    """The frames of ONE thread out of a faulthandler dump.

    ``faulthandler`` prints ``Thread 0x<hex> (most recent call first):`` with
    no thread name, so the only reliable selector is the identifier, which
    the child reports for both the main thread and the executor manager
    thread.
    """
    if ident is None or 'Timeout (' not in blob:
        return None
    frames, keep = [], False
    for line in blob.splitlines():
        s2 = line.strip()
        if s2.startswith('Thread 0x') or s2.startswith('Current thread 0x'):
            tok = s2.split()[2] if s2.startswith('Current') else s2.split()[1]
            keep = (int(tok, 16) == int(ident))
            if keep and frames:
                break                       # first dump only
            continue
        if keep and s2.startswith('File '):
            frames.append(s2)
    return frames[:14] or None


def _all_stacks(blob):
    """Everything, for the record."""
    if 'Timeout (' not in blob:
        return None
    end = blob.find('Timeout (', blob.find('Timeout (') + 1)
    return blob[:end if end > 0 else len(blob)][:6000]


def sweep():
    """Kill any worker this probe left behind, and count them."""
    import subprocess as sp
    found = []
    try:
        ps = sp.run(['ps', '-eo', 'pid,ppid,cmd'], capture_output=True,
                    text=True, timeout=30).stdout
    except Exception:                              # noqa: BLE001
        return {'error': 'ps unavailable'}
    for line in ps.splitlines():
        if ('multiprocessing.spawn_main' in line
                or 'vf7_d6_exit_hang' in line):
            parts = line.split(None, 2)
            if len(parts) >= 1 and parts[0].isdigit():
                found.append(int(parts[0]))
    me = os.getpid()
    killed = []
    for pid in found:
        if pid == me:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except Exception:                          # noqa: BLE001
            pass
    return {'found': len(found), 'killed': killed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default=None)
    ap.add_argument('--drive', action='store_true')
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--deadline', type=float, default=60.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.drive:
        rows = drive(args)
        out = {'python': sys.version, 'platform': sys.platform,
               'deadline_seconds': args.deadline, 'arms': rows,
               'final_sweep': sweep()}
        if args.out:
            with open(args.out, 'w', encoding='cp1252',
                      errors='replace') as fh:
                json.dump(out, fh, indent=1)
            print('wrote', args.out)
        return 0
    return run_child(args.mode or 'control', N=args.n, workers=args.workers)


if __name__ == '__main__':
    sys.exit(main())

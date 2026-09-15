"""P1 -- synthetic reproduction of the broken-pool shutdown deadlock.

No lumenairy code.  Builds a ``ProcessPoolExecutor`` on the ``spawn`` context
exactly as ``lumenairy.elements._lens_traced._get_persistent_worker_pool``
does, then breaks it on purpose and measures what ``shutdown`` does.

Terms used throughout
---------------------
call queue      the ``multiprocessing.Queue`` the executor's manager thread
                writes tasks into; the workers read it.  Its parent-side
                writer is fed by a QueueFeederThread over an 8 KiB pipe, so a
                task payload larger than 8 KiB blocks that thread until a
                worker reads.  The library ships a ~1.9 MB pickled Newton
                payload per chunk, so it is always in that regime.
manager thread  ``concurrent.futures.process._ExecutorManagerThread``; on a
                dead worker it runs ``_terminate_broken``, which terminates
                the remaining workers, closes the call queue and JOINS the
                feeder thread and every worker process, all unbounded.
broken pool     ``BrokenProcessPool`` raised out of ``future.result()``.

Scenarios (``--scenario``)
--------------------------
early_die       task 0 kills its worker immediately; the feeder is typically
                idle in ``notempty.wait()`` when the pool breaks.
feeder_blocked  every worker is first occupied by a long SMALL task, so the
                feeder is blocked inside ``Connection._send_bytes`` writing
                the first LARGE task when a busy worker dies.  This is the
                state the 2026-09-14 release-gate dumps captured.
wedged_worker   as feeder_blocked, but the surviving workers ignore
                ``SIGTERM``/``TerminateProcess`` for a while, so the manager
                thread's ``p.join()`` inside ``_terminate_broken`` is the
                unbounded wait.

Modes (``--mode``)
------------------
wait_true   shutdown(wait=True)   -- the shape ``close_worker_pool`` ships
wait_false  shutdown(wait=False, cancel_futures=True)
none        no shutdown at all; go straight to interpreter exit

Every phase prints one JSON object with a monotonic timestamp, so a phase
that never completes is visible as a MISSING line.  ``--exit-probe`` also
measures whether the interpreter can exit after the chosen shutdown, which is
the half of the question ``shutdown(wait=False)`` alone does not answer:
``concurrent.futures.process._python_exit`` is registered through
``threading._register_atexit`` and joins every live manager thread, so a
manager thread left stuck in ``_terminate_broken`` moves the hang from the
fallback to interpreter exit rather than removing it.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

T0 = time.monotonic()


def emit(phase, **kw):
    kw['phase'] = phase
    kw['t'] = round(time.monotonic() - T0, 4)
    sys.stdout.write(json.dumps(kw) + '\n')
    sys.stdout.flush()


def task_ok(blob, i):
    time.sleep(0.05)
    return i


def task_die(blob, i):
    # A worker that vanishes mid-flight: no traceback, no result item, just a
    # closed sentinel.  This is what a segfaulting / OOM-killed / antivirus-
    # terminated worker looks like to the executor.
    os._exit(7)


def task_hold(blob, i, seconds, die_after=-1.0):
    """Occupy a worker without reading anything more from the call queue.

    With every worker holding one of these, the feeder thread blocks inside
    ``Connection._send_bytes`` on the next large task.  ``die_after`` >= 0
    makes this worker vanish at that offset, which is what breaks the pool.
    """
    t_end = time.monotonic() + seconds
    if die_after >= 0.0:
        time.sleep(die_after)
        os._exit(7)
    while time.monotonic() < t_end:
        time.sleep(0.05)
    return i


def task_wedge(blob, i, seconds, die_after=-1.0):
    """Like ``task_hold`` but ignores SIGTERM, so ``Process.terminate`` needs
    the OS to force it down.  On POSIX this makes the manager thread's
    ``p.join()`` the unbounded wait; on Windows ``TerminateProcess`` is not
    catchable, so this degenerates to ``task_hold``."""
    import signal
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except (ValueError, AttributeError, OSError):
        pass
    return task_hold(blob, i, seconds, die_after)


def build_tasks(ex, args, blob, small):
    """Submit the scenario's tasks; return {future: index}."""
    futs = {}
    if args.scenario == 'early_die':
        futs[ex.submit(task_die, blob, 0)] = 0
        for i in range(1, args.tasks):
            futs[ex.submit(task_ok, blob, i)] = i
        return futs

    hold = task_wedge if args.scenario == 'wedged_worker' else task_hold
    # One holder per worker, SMALL payload so they are consumed at once.
    for w in range(args.workers):
        die_after = args.die_after if w == 0 else -1.0
        futs[ex.submit(hold, small, w, args.hold_seconds, die_after)] = w
    # ...then large tasks that cannot fit the 8 KiB pipe, so the feeder
    # thread blocks inside _send_bytes while every worker is busy.
    for i in range(args.workers, args.tasks):
        futs[ex.submit(task_ok, blob, i)] = i
    return futs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('wait_true', 'wait_false', 'none'),
                    default='wait_true')
    ap.add_argument('--scenario',
                    choices=('early_die', 'feeder_blocked', 'wedged_worker'),
                    default='feeder_blocked')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--tasks', type=int, default=24)
    ap.add_argument('--payload-mb', type=float, default=8.0)
    ap.add_argument('--hold-seconds', type=float, default=30.0)
    ap.add_argument('--die-after', type=float, default=3.0)
    ap.add_argument('--dump-after', type=float, default=60.0)
    ap.add_argument('--exit-probe', action='store_true')
    args = ap.parse_args()

    faulthandler.enable()
    # Hard stop: if any phase blocks, dump every thread and abort the process
    # rather than joining the family of hangs this probe exists to study.
    faulthandler.dump_traceback_later(args.dump_after, exit=True)

    blob = b'\x5a' * int(args.payload_mb * (1 << 20))
    small = b'\x5a' * 64
    ctx = mp.get_context('spawn')
    emit('start', mode=args.mode, scenario=args.scenario,
         workers=args.workers, tasks=args.tasks, payload_mb=args.payload_mb,
         python=sys.version.split()[0])

    ex = ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx)
    futs = build_tasks(ex, args, blob, small)
    emit('submitted', n=len(futs))

    broken = None
    try:
        for fut in as_completed(futs):
            fut.result()
    except BrokenProcessPool as exc:
        broken = exc
        emit('broken', exc=type(exc).__name__, text=str(exc)[:600],
             cause=type(exc.__cause__).__name__ if exc.__cause__ else None,
             cause_text=(str(exc.__cause__)[:1200]
                         if exc.__cause__ else None))
    except BaseException as exc:  # noqa: BLE001 -- probe reports everything
        emit('other_exception', exc=type(exc).__name__, text=str(exc)[:400])

    if broken is None:
        emit('not_broken')

    emit('threads_before_shutdown',
         threads=sorted(t.name for t in threading.enumerate()))
    emit('shutdown_enter', mode=args.mode)
    t_sd = time.monotonic()
    if args.mode == 'wait_true':
        ex.shutdown(wait=True)
    elif args.mode == 'wait_false':
        ex.shutdown(wait=False, cancel_futures=True)
    emit('shutdown_return', seconds=round(time.monotonic() - t_sd, 4))

    # Reaching here at all is the property the library's broken-pool branch
    # needs: it is where the serial fallback would run.
    emit('fallback_reached')
    emit('threads_after_shutdown',
         threads=sorted(t.name for t in threading.enumerate()))

    if args.exit_probe:
        emit('exit_probe_arm')
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(args.dump_after, exit=True)
    else:
        faulthandler.cancel_dump_traceback_later()
        os._exit(0)


if __name__ == '__main__':
    main()

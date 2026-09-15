"""VP1 -- is the shutdown lock really held across an unbounded join?

CLAIM (1) of the WP-B13 report, re-measured rather than read.  Two arms, both
on a REAL ``ProcessPoolExecutor``, no lumenairy anywhere:

broken  A worker ``os._exit``s, so CPython's manager thread runs
        ``terminate_broken`` -> ``_terminate_broken``, which this probe makes
        SLOW in a controlled way by wrapping ``_join_executor_internals``
        (the step that ends in ``call_queue.join_thread()`` and ``p.join()``)
        with a fixed sleep.  The parent then calls
        ``shutdown(wait=False, cancel_futures=True)`` -- the handoff's
        one-liner -- and this probe times it.  If the report is right the call
        blocks for about the sleep; if it is wrong it returns at once.

healthy No break: the same ``shutdown(wait=False, cancel_futures=True)`` on a
        live pool, to show the block is the broken state and not the flag.

Also asserts the source-level facts on whichever interpreter runs it, so the
3.12 and 3.14 readings are produced by the same code.
"""
from __future__ import annotations

import argparse
import concurrent.futures.process as cfp
import faulthandler
import inspect
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor


def _suicide(_):
    os._exit(7)


def _sleepy(_):
    time.sleep(30.0)
    return 1


def source_facts():
    src = inspect.getsource(cfp).splitlines()
    out = {}
    for i, line in enumerate(src, 1):
        if re.match(r'\s+def terminate_broken\(', line):
            out['terminate_broken_line'] = i
            out['terminate_broken_body'] = [s.strip() for s in src[i:i + 2]]
        if re.match(r'\s+def _join_executor_internals\(', line):
            out['join_internals_line'] = i
        if re.match(r'\s+def shutdown\(self, wait=True', line):
            out['shutdown_line'] = i
            out['shutdown_first_stmt'] = src[i].strip()
    body = inspect.getsource(cfp._ExecutorManagerThread._terminate_broken)
    ji = inspect.getsource(cfp._ExecutorManagerThread._join_executor_internals)
    out['terminate_broken_calls_join_internals'] = (
        '_join_executor_internals(broken=True)' in body)
    out['join_internals_has_join_thread'] = 'join_thread()' in ji
    out['join_internals_has_p_join'] = 'p.join()' in ji
    out['join_thread_has_timeout'] = bool(
        re.search(r'join_thread\([^)]+\)', ji))
    out['p_join_has_timeout'] = bool(re.search(r'\bp\.join\([^)]+\)', ji))
    out['terminate_broken_holds_lock'] = (
        out.get('terminate_broken_body', [''])[0] == 'with self.shutdown_lock:')
    out['shutdown_takes_lock_first'] = (
        out.get('shutdown_first_stmt') == 'with self._shutdown_lock:')
    return out


def arm_broken(block, workers):
    """One executor per process, so nothing else can enter the patched join."""
    orig = cfp._ExecutorManagerThread._join_executor_internals
    marks = {}

    def slow(self, broken=False):
        marks.setdefault('entered', time.monotonic())
        marks['broken_flag'] = bool(broken)
        time.sleep(block)
        marks['left'] = time.monotonic()
        return orig(self, broken=broken)

    cfp._ExecutorManagerThread._join_executor_internals = slow
    ex = ProcessPoolExecutor(max_workers=workers,
                             mp_context=__import__('multiprocessing')
                             .get_context('spawn'))
    # warm the workers so the break is mid-flight, not at spawn
    list(ex.map(abs, range(workers)))
    futs = [ex.submit(_suicide, 0)]
    futs += [ex.submit(_sleepy, i) for i in range(workers * 2)]
    t_wait = time.monotonic()
    while 'entered' not in marks and time.monotonic() - t_wait < 120:
        time.sleep(0.01)
    entered = 'entered' in marks
    # the manager is now inside _terminate_broken, holding _shutdown_lock
    t0 = time.monotonic()
    ex.shutdown(wait=False, cancel_futures=True)
    dt = time.monotonic() - t0
    return {'manager_entered_join': entered,
            'join_saw_broken_true': marks.get('broken_flag'),
            'executor_broken_flag': bool(getattr(ex, '_broken', None)),
            'shutdown_wait_false_cancel_true_seconds': round(dt, 3),
            'blocked_for': round(block, 3),
            'blocked': dt > 0.5 * block}


def arm_healthy(workers):
    ex = ProcessPoolExecutor(max_workers=workers,
                             mp_context=__import__('multiprocessing')
                             .get_context('spawn'))
    list(ex.map(abs, range(workers)))
    t0 = time.monotonic()
    ex.shutdown(wait=False, cancel_futures=True)
    dt = time.monotonic() - t0
    return {'shutdown_wait_false_cancel_true_seconds': round(dt, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', type=float, default=20.0)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--dump-after', type=float, default=240.0)
    ap.add_argument('--arm', choices=('broken', 'healthy', 'source'),
                    default='broken')
    ap.add_argument('--json-out', default=None)
    a = ap.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(a.dump_after, exit=True)
    out = {'python': sys.version.split()[0],
           'process_py': cfp.__file__,
           'arm': a.arm,
           'source': source_facts()}
    if a.arm == 'healthy':
        out['healthy'] = arm_healthy(a.workers)
    elif a.arm == 'broken':
        out['broken'] = arm_broken(a.block, a.workers)
    print(json.dumps(out, indent=1, sort_keys=True))
    if a.json_out:
        with open(a.json_out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    faulthandler.cancel_dump_traceback_later()
    return 0


if __name__ == '__main__':
    sys.exit(main())

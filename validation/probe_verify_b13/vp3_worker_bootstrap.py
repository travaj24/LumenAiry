"""VP3 -- how long does a Newton pool worker take to become usable?

The WP report's section 7.4 records that on WSL "exactly one worker at a time
makes progress, each taking three to six minutes of CPU to finish its
bootstrap", and files the cause (the /mnt/c 9p mount) as an INFERENCE.  This
probe measures the ladder directly and, with ``--tree`` pointed at a copy on
the Linux filesystem, tests that inference.

Three stages per rung, all on a real spawn ``ProcessPoolExecutor``:

  spawn     time until every worker has answered a trivial task (interpreter
            up, no library import)
  import    time until every worker has imported lumenairy (the real cost a
            Newton chunk pays first)
  chunk     time until every worker has imported the traced-lens module's
            heavy dependencies the way ``_newton_invert_chunk`` does

Per-worker first-answer times are recorded too, which is what shows whether
the starts are SERIALISED (report's claim) or concurrent.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

T0 = 0.0


def emit(event, **kw):
    kw['event'] = event
    kw['t'] = round(time.monotonic() - T0, 4)
    sys.stdout.write(json.dumps(kw, sort_keys=True) + "\n")
    sys.stdout.flush()


def task_trivial(i):
    return (i, os.getpid(), time.time())


def task_import(i):
    t0 = time.time()
    import lumenairy  # noqa: F401
    return (i, os.getpid(), time.time(), time.time() - t0, lumenairy.__file__)


def task_chunkdeps(i):
    t0 = time.time()
    import numpy  # noqa: F401
    import scipy.interpolate  # noqa: F401

    from lumenairy.elements import _lens_traced  # noqa: F401
    return (i, os.getpid(), time.time(), time.time() - t0,
            _lens_traced.__file__)


_TASKS = {'spawn': task_trivial, 'import': task_import,
          'chunk': task_chunkdeps}


def rung(workers, stage):
    fn = _TASKS[stage]
    ctx = mp.get_context('spawn')
    t0 = time.monotonic()
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    futs = [ex.submit(fn, i) for i in range(workers)]
    firsts = []
    for f in as_completed(futs):
        firsts.append(round(time.monotonic() - t0, 3))
    total = round(time.monotonic() - t0, 3)
    pids = set()
    inner = []
    for f in futs:
        r = f.result()
        pids.add(r[1])
        if len(r) > 3:
            inner.append(round(r[3], 3))
    ex.shutdown(wait=True)
    return {'workers': workers, 'stage': stage, 'total_seconds': total,
            'answer_seconds': sorted(firsts),
            'distinct_pids': len(pids),
            'worker_inner_seconds': sorted(inner),
            'shutdown_included': False}


def main():
    global T0
    T0 = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument('--ladder', default='1,2,4,8')
    ap.add_argument('--stages', default='spawn,import,chunk')
    ap.add_argument('--dump-after', type=float, default=1800.0)
    ap.add_argument('--json-out', default=None)
    a = ap.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(a.dump_after, exit=True)
    out = {'python': sys.version.split()[0], 'pid': os.getpid(),
           'sys_path0': sys.path[0], 'pythonpath': os.environ.get('PYTHONPATH'),
           'rungs': []}
    emit('start', **{k: v for k, v in out.items() if k != 'rungs'})
    for stage in a.stages.split(','):
        for w in [int(x) for x in a.ladder.split(',')]:
            r = rung(w, stage)
            out['rungs'].append(r)
            emit('rung', **r)
    if a.json_out:
        with open(a.json_out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    faulthandler.cancel_dump_traceback_later()
    emit('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())

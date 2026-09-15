"""VP runner -- launch a probe against a chosen TREE under a hard timeout.

Nothing in this file imports lumenairy.  It exists so that no probe that
drives a process pool can ever hang the verifier: the child gets
``subprocess.run(..., timeout=...)``, and on expiry its whole process tree is
killed.  The child pins its own tree through ``PYTHONPATH`` and prints
``lumenairy.__file__`` as its first JSON line, which this runner echoes.

It also separates two outcomes the report treats as one:

  returned   the probe printed its ``dispatch_returned`` line -- the library
             call came back.
  exited     the child process itself terminated inside the deadline.  A run
             that RETURNED but did not EXIT is the "wedge moves to interpreter
             exit" state the WP report claims in section 5.3.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def kill_tree(pid):
    n = 0
    try:
        import psutil
    except ImportError:
        return None
    try:
        p = psutil.Process(pid)
    except Exception:
        return 0
    kids = []
    try:
        kids = p.children(recursive=True)
    except Exception:
        pass
    for c in kids + [p]:
        try:
            c.kill()
            n += 1
        except Exception:
            pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--timeout', type=float, default=300.0)
    ap.add_argument('--label', default='')
    ap.add_argument('--jsonl', default=None)
    ap.add_argument('rest', nargs=argparse.REMAINDER)
    a = ap.parse_args()

    env = dict(os.environ)
    env['PYTHONPATH'] = a.tree
    env['OMP_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['PYTHONUNBUFFERED'] = '1'

    argv = [a.python] + [x for x in a.rest if x != '--']
    t0 = time.monotonic()
    timed_out = False
    killed = 0
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=a.timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        # kill the child AND every worker it spawned, then drain.
        killed = kill_tree(proc.pid) or 0
        try:
            proc.kill()
        except Exception:
            pass
        try:
            out, err = proc.communicate(timeout=30)
        except Exception:
            out, err = '', ''
        rc = proc.returncode
    dt = time.monotonic() - t0

    events = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('{'):
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    names = [e.get('event') for e in events]
    rec = {'label': a.label, 'tree': a.tree, 'argv': argv,
           'seconds': round(dt, 3), 'returncode': rc,
           'timed_out': timed_out,
           'returned': 'dispatch_returned' in names or 'summary' in names,
           'exited': not timed_out,
           'events': names,
           'lumenairy': next((e.get('lumenairy') for e in events
                              if e.get('event') == 'import'), None),
           'python': next((e.get('python') for e in events
                           if e.get('event') == 'import'), None),
           'result': next((e for e in events
                           if e.get('event') == 'dispatch_returned'), None),
           'killed_processes': killed,
           'stderr_tail': err[-4000:] if err else ''}
    print(json.dumps(rec, indent=1, sort_keys=True))
    if a.jsonl:
        with open(a.jsonl, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec, sort_keys=True) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

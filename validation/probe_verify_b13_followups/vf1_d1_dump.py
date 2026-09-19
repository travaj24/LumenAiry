"""VERIFY WP-B13 follow-ups, D1 -- the wedge report, with MY OWN wedge.

The branch's pin parks a daemon thread in ``_b13_wedged_frame_for_the_dump``
on a ``threading.Event``.  This probe uses a DIFFERENT frame name and a
DIFFERENT park mechanism -- a blocking ``queue.Queue.get()`` -- so that
"the dump names the frame that is stuck" is re-decided rather than re-read,
and so that a dump helper that happened to work only for ``Event.wait``
frames would be caught.

Three arms per build:

  fixed      the branch's ``_thread_dump`` (tempfile), through the branch's
             ``_with_deadline``: the failure must carry _with_deadline's own
             message AND the name of my parked frame.
  shipped    the base-commit helper (``io.StringIO``) rebound onto the test
             module in memory: the same wedge, same deadline.
  premise    what ``faulthandler.dump_traceback(file=io.StringIO())`` does on
             this build, measured rather than quoted.

Also checks that the dump reaches ALL threads (``all_threads=True`` is the
only reason the helper is useful: the wedged frame is never in the raising
thread's traceback).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=<tree> python validation/probe_verify_b13_followups/\
vf1_d1_dump.py --out vf1_win.json
"""
from __future__ import annotations

import argparse
import faulthandler
import importlib.util
import io
import json
import os
import queue
import sys
import threading


def _vf1_parked_on_a_queue_get(q):
    """A uniquely named module-level frame that parks on ``Queue.get()``.

    Different name and different mechanism from the branch's own pin, on
    purpose: the claim under test is that the dump names WHATEVER frame is
    stuck, not that it names one particular one.
    """
    return q.get()


def _import_test_module(root):
    path = os.path.join(root, 'tests', 'unit',
                        'test_fix_newton_pool_broken_fallback.py')
    spec = importlib.util.spec_from_file_location('_b13_testmod', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_b13_testmod'] = mod
    spec.loader.exec_module(mod)
    return mod


def _shipped_thread_dump():
    """base b631ce79's helper, verbatim."""
    buf = io.StringIO()
    faulthandler.dump_traceback(file=buf, all_threads=True)
    return buf.getvalue()


def _arm(tm, label, deadline=1.0):
    q = queue.Queue()
    err = None
    try:
        tm._with_deadline(lambda: _vf1_parked_on_a_queue_get(q), deadline,
                          'a VF1 engineered wedge')
        outcome = 'returned (no wedge detected -- the arm is invalid)'
    except BaseException as exc:                   # noqa: BLE001
        outcome = 'failed'
        err = f'{type(exc).__name__}: {exc}'
    finally:
        q.put(None)
    msg = err or ''
    return {'arm': label,
            'outcome': outcome,
            'exception_type': msg.split(':', 1)[0] if msg else None,
            'carries_with_deadline_message': (
                'a VF1 engineered wedge did not return within' in msg),
            'carries_thread_dump_header': 'Thread dump:' in msg,
            'names_the_parked_frame': '_vf1_parked_on_a_queue_get' in msg,
            'names_the_deadline_thread': 'deadline-a VF1' in msg,
            'message_head': msg[:400],
            'message_tail': msg[-600:] if len(msg) > 600 else msg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    tm = _import_test_module(root)

    out = {'python': sys.version, 'platform': sys.platform,
           'lumenairy_file': lumenairy.__file__}

    # --- premise, measured ------------------------------------------------
    try:
        faulthandler.dump_traceback(file=io.StringIO(), all_threads=True)
        out['stringio_premise'] = {'raises': False, 'exception': None}
    except BaseException as exc:                   # noqa: BLE001
        out['stringio_premise'] = {'raises': True,
                                   'exception': type(exc).__name__,
                                   'message': str(exc)}
    print('premise', json.dumps(out['stringio_premise']), flush=True)

    # --- the helper on its own, with a SECOND thread parked ---------------
    # ``all_threads=True`` is the only reason the helper is worth having:
    # the wedged frame is never in the raising thread's traceback.  So the
    # claim has to be decided with more than one thread alive.
    side_q = queue.Queue()
    side = threading.Thread(target=_vf1_parked_on_a_queue_get,
                            args=(side_q,), name='vf1-side', daemon=True)
    side.start()
    import time as _t
    _t.sleep(0.2)
    dump = tm._thread_dump()
    side_q.put(None)
    headers = dump.count('Thread 0x') + dump.count('Current thread 0x')
    out['helper_direct'] = {
        'chars': len(dump),
        'names_this_frame': 'main' in dump,
        'thread_headers': headers,
        'names_the_side_thread_frame': '_vf1_parked_on_a_queue_get' in dump,
        'covers_more_than_one_thread': headers > 1,
    }
    print('helper_direct', json.dumps(out['helper_direct']), flush=True)

    # --- arm 1: the branch as it stands -----------------------------------
    out['arm_fixed'] = _arm(tm, 'fixed (branch tempfile helper)')
    print('arm_fixed', json.dumps(
        {k: v for k, v in out['arm_fixed'].items()
         if not k.startswith('message')}), flush=True)

    # --- arm 2: the shipped helper rebound in memory ----------------------
    real = tm._thread_dump
    tm._thread_dump = _shipped_thread_dump
    try:
        out['arm_shipped'] = _arm(tm, 'shipped (io.StringIO helper)')
    finally:
        tm._thread_dump = real
    print('arm_shipped', json.dumps(
        {k: v for k, v in out['arm_shipped'].items()
         if not k.startswith('message')}), flush=True)

    out['verdict'] = {
        'fixed_arm_reports_its_own_message':
            out['arm_fixed']['carries_with_deadline_message'],
        'fixed_arm_names_the_wedged_frame':
            out['arm_fixed']['names_the_parked_frame'],
        'shipped_arm_loses_the_message':
            not out['arm_shipped']['carries_with_deadline_message'],
        'shipped_arm_raises_unsupported_operation':
            'UnsupportedOperation' in (out['arm_shipped']['message_head']
                                       or ''),
        'dump_covers_all_threads':
            out['helper_direct']['covers_more_than_one_thread'],
    }
    with open(args.out, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', json.dumps(out['verdict'], indent=1))
    print('wrote', args.out)


if __name__ == '__main__':
    main()

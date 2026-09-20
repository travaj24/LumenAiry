"""FIX WP-B13 FOLLOW-UPS round 2, VD2 -- the census's unconditional remove.

VERIFY_WP-B13_FOLLOWUPS.md section 8 defect VD2: ``_shutdown_pool_bounded``'s
helper ran ``_ABANDONED_POOLS.remove(ex)`` in its ``finally`` WHETHER OR NOT
its caller had appended, so a bounded teardown that completed inside its
bound would drop an entry :func:`_abandon_pool` had put there.  The verifier
measured that at **6/6 by construction** against a COPY of the shipped
ordering (``vf2_d2_census.py::attack_unconditional_remove``).

This probe reuses the verifier's own machinery -- its ``Census``,
``_StubPool``, ``_OrderedLock``, its three ordering copies and its attack --
by importing ``vf2_d2_census`` from ``validation/probe_verify_b13_followups``,
so the fail-before arm is literally the verifier's reproducer rather than a
re-written one.  Arms:

  fail_before   the verifier's ``bounded_shipped`` copy (= the pre-VD2
                library ordering) under the verifier's attack.  Expected:
                6/6 foreign entries dropped.
  fail_after    the SAME attack driven against the real, fixed
                ``lumenairy.elements._lens_traced._shutdown_pool_bounded``
                and the real ``_ABANDONED_POOLS``.  Expected: 0/6.
  unchanged     the whole D2 race matrix re-run -- three copies x two built
                arrival orders, plus the real library in both orders -- so
                the 8/8 (one-liner, helper-first) vs 0/8 (shipped, both
                orders) readings can be compared with the verifier's.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=<tree> python validation/probe_fix_b13_followups_r2/\
r2_vd2_census.py --out r2_vd2_win.json
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_VF2 = os.path.join(os.path.dirname(_HERE), 'probe_verify_b13_followups',
                    'vf2_d2_census.py')


def _load_verifier_module():
    """Import the verifier's probe AS IT STANDS -- never a copy of it."""
    spec = importlib.util.spec_from_file_location('vf2_d2_census', _VF2)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['vf2_d2_census'] = mod
    spec.loader.exec_module(mod)
    return mod


def attack_unconditional_remove_real(VF2, LT, reps=6):
    """The verifier's attack, driven against the REAL library function.

    Same construction as ``vf2_d2_census.attack_unconditional_remove``: a
    reaper appends and stays outstanding, then a bounded teardown of the SAME
    executor runs with a bound it comfortably meets (so its caller never
    appends), and the census is read while the reaper is still outstanding.
    The only difference is that the subject is ``LT._shutdown_pool_bounded``
    and the census is the module's own ``_ABANDONED_POOLS``.
    """
    dropped = 0
    returns = []
    base_len = len(LT._ABANDONED_POOLS)
    base_to = LT._POOL_SHUTDOWN_TIMEOUTS
    for _ in range(reps):
        pool = VF2._StubPool()
        pool.release()                  # a bounded shutdown returns AT ONCE
        gate = threading.Event()

        def _reap(_p=pool, _g=gate):
            with LT._ABANDONED_POOLS_LOCK:
                LT._ABANDONED_POOLS.append(_p)   # _abandon_pool appends now
            _g.wait(30.0)                        # ... and stays outstanding
            with LT._ABANDONED_POOLS_LOCK:
                try:
                    LT._ABANDONED_POOLS.remove(_p)
                except ValueError:
                    pass

        r = threading.Thread(target=_reap,
                             name='lumenairy-newton-pool-reaper', daemon=True)
        r.start()
        time.sleep(0.05)
        returns.append(LT._shutdown_pool_bounded(pool, 10.0))
        time.sleep(0.15)
        with LT._ABANDONED_POOLS_LOCK:
            still_there = pool in LT._ABANDONED_POOLS
        if not still_there:
            dropped += 1
        gate.set()
        r.join(30.0)
    return {'reps': reps,
            'reapers_entry_dropped_by_the_bounded_helper': dropped,
            'bounded_teardown_returned_in_bound': all(returns),
            'returns': sorted({str(x) for x in returns}),
            'census_len_before': base_len,
            'census_len_after': len(LT._ABANDONED_POOLS),
            'timeouts_before': base_to,
            'timeouts_after': LT._POOL_SHUTDOWN_TIMEOUTS}


def precondition_pin(LT):
    """Re-read, as numbers, the source facts the VD2 precondition pin makes."""
    tree = ast.parse(inspect.getsource(LT))
    fns = {n.name: n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef)}
    out = {}

    def _called_names(stmts):
        mod = ast.Module(body=stmts, type_ignores=[])
        return {c.func.id for c in ast.walk(mod)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}

    exclusive = False
    cw = fns.get('close_worker_pool')
    if cw is not None:
        for node in ast.walk(cw):
            if not isinstance(node, ast.If):
                continue
            b, o = _called_names(node.body), _called_names(node.orelse)
            if ('_abandon_pool' in b and '_shutdown_pool_bounded' in o) or \
               ('_shutdown_pool_bounded' in b and '_abandon_pool' in o):
                exclusive = True
    out['close_worker_pool_routes_to_exactly_one_mechanism'] = exclusive
    gp = fns.get('_get_persistent_worker_pool')
    gp_names = _called_names(gp.body) if gp is not None else set()
    out['_get_persistent_worker_pool_calls_shutdown_pool_bounded'] = (
        '_shutdown_pool_bounded' in gp_names)
    src_fn = inspect.getsource(LT._shutdown_pool_bounded)
    out['helper_removal_is_guarded_by_added'] = 'if added:' in src_fn
    out['caller_sets_added_under_the_lock'] = 'added.append(True)' in src_fn
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    from lumenairy.elements import _lens_traced as LT

    VF2 = _load_verifier_module()
    print('verifier probe =', VF2.__file__, flush=True)

    out = {'python': sys.version, 'platform': sys.platform,
           'lumenairy_file': lumenairy.__file__,
           'verifier_probe': VF2.__file__, 'reps': args.reps}

    # ---- fail-before: the verifier's own reproducer, unchanged -----------
    out['fail_before_copy_of_the_pre_vd2_ordering'] = \
        VF2.attack_unconditional_remove(6)
    print('fail_before',
          json.dumps(out['fail_before_copy_of_the_pre_vd2_ordering']),
          flush=True)

    # ---- fail-after: the same attack on the real, fixed function ---------
    out['fail_after_real_library'] = attack_unconditional_remove_real(
        VF2, LT, 6)
    print('fail_after', json.dumps(out['fail_after_real_library']), flush=True)

    # ---- unchanged: the D2 race matrix -----------------------------------
    races = []
    for impl in VF2.IMPLS:
        for order in ('helper', 'caller'):
            r = VF2.run_race(impl, order, args.reps)
            races.append(r)
            print('race', json.dumps(r), flush=True)
    out['races'] = races
    out['real_library'] = [VF2.run_real_library(LT, o, args.reps)
                           for o in ('helper', 'caller')]
    print('real_library', json.dumps(out['real_library']), flush=True)

    # ---- the other D2 attacks, so nothing regressed elsewhere ------------
    out['attack_helper_crash'] = VF2.attack_helper_crash(3)
    out['attack_double_abandon'] = VF2.attack_double_abandon(6)
    out['attack_boundary'] = VF2.attack_boundary(args.reps)
    out['precondition_pin'] = precondition_pin(LT)

    out['verdict'] = {
        'vd2_before_drops_per_6': out[
            'fail_before_copy_of_the_pre_vd2_ordering'][
            'reapers_entry_dropped_by_the_bounded_helper'],
        'vd2_after_drops_per_6': out['fail_after_real_library'][
            'reapers_entry_dropped_by_the_bounded_helper'],
        'oneline_leaks_helper_first': [
            r['leaked'] for r in races
            if r['impl'] == 'oneline' and r['order'] == 'helper'][0],
        'oneline_leaks_caller_first': [
            r['leaked'] for r in races
            if r['impl'] == 'oneline' and r['order'] == 'caller'][0],
        'shipped_copy_leaks_any_order': sum(
            r['leaked'] for r in races if r['impl'] == 'shipped'),
        'real_library_leaks_any_order': sum(
            r['leaked'] for r in out['real_library']),
        'real_library_census_empty_at_end': len(LT._ABANDONED_POOLS) == 0,
        'helper_crash_returns_to_empty': all(
            r['returns_to_empty'] for r in out['attack_helper_crash']),
        'boundary_leaks': out['attack_boundary']['leaks'],
        'double_abandon_premature_removals': out['attack_double_abandon'][
            'premature_removals_of_the_reapers_entry'],
        'precondition_pin': out['precondition_pin'],
    }
    with open(args.out, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', json.dumps(out['verdict'], indent=1))
    print('wrote', args.out)


if __name__ == '__main__':
    main()

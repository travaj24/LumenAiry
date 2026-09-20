"""VERIFY WP-B13 follow-ups, D2 -- the abandoned-pool census, attacked.

Three implementations of ``_shutdown_pool_bounded``'s ordering are driven
against the SAME census list and the SAME adverse interleaver:

  ``pre``      what shipped at base b631ce79: expiry appends, nothing ever
               removes.
  ``oneline``  the repair VERIFY-WP-B13 asked for: a bare
               ``finally: remove; done.set()`` in the helper.
  ``shipped``  this branch: ``done.set()`` BEFORE the lock, and the caller
               re-reads ``done`` under the lock before appending.  Driven
               here as a COPY (so all three sit under one interleaver) AND
               as the real ``lumenairy.elements._lens_traced`` function, so
               the copy is never the only evidence.

The adverse order is produced BY CONSTRUCTION, not by sleeping: a census
lock that admits the teardown-helper thread first and makes every other
thread wait until the helper has left its critical section.  A second
interleaver (``append_first``) produces the opposite order, so both arms of
the race are covered rather than one.

Attacks on the shipped ordering, each run REPS times:
  helper_crash       the helper's shutdown raises a class the helper does
                     NOT catch -- does ``done`` still get published, and is
                     the census still clean?
  double_abandon     ``_abandon_pool(ex)`` and a bounded teardown of the
                     SAME executor, concurrently: can the census leak, and
                     can an entry be removed twice / prematurely?
  boundary           the caller's wait expiring in the same instant the
                     helper publishes, with no interleaver at all.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=<tree> python validation/probe_verify_b13_followups/\
vf2_d2_census.py --out vf2_win.json
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time


# ==========================================================================
# The subject under test: a census, a lock, and three orderings.
# ==========================================================================
class Census:
    def __init__(self, lock=None):
        self.items = []
        self.lock = lock if lock is not None else threading.Lock()
        self.timeouts = 0


class _StubPool:
    """Joins forever until released.  No child processes involved."""

    def __init__(self, raise_cls=None):
        self._gate = threading.Event()
        self._raise = raise_cls
        self.shutdown_calls = []

    def shutdown(self, wait=True, **kw):
        self.shutdown_calls.append(bool(wait))
        if wait:
            self._gate.wait(120.0)
        if self._raise is not None:
            raise self._raise('engineered: the helper did not catch this')

    def release(self):
        self._gate.set()


class _Uncatchable(BaseException):
    """Not in ``(RuntimeError, OSError, ValueError)`` and not an Exception.

    The question this answers is whether ``done`` is published on a path the
    helper's ``except`` does not cover -- i.e. whether the ``finally`` is
    load-bearing.
    """


def bounded_pre(census, ex, timeout):
    """base b631ce79: append on expiry, never remove."""
    done = threading.Event()

    def _run():
        try:
            ex.shutdown(wait=True)
        except (RuntimeError, OSError, ValueError):
            pass
        finally:
            done.set()

    threading.Thread(target=_run, name='lumenairy-newton-pool-close',
                     daemon=True).start()
    if done.wait(timeout):
        return True
    with census.lock:
        census.timeouts += 1
        census.items.append(ex)
    return False


def bounded_oneline(census, ex, timeout):
    """The one-liner VERIFY-WP-B13 requested: remove, THEN publish."""
    done = threading.Event()

    def _run():
        try:
            ex.shutdown(wait=True)
        except (RuntimeError, OSError, ValueError):
            pass
        finally:
            with census.lock:
                try:
                    census.items.remove(ex)
                except ValueError:
                    pass
            done.set()

    threading.Thread(target=_run, name='lumenairy-newton-pool-close',
                     daemon=True).start()
    if done.wait(timeout):
        return True
    with census.lock:
        census.timeouts += 1
        census.items.append(ex)
    return False


def bounded_shipped(census, ex, timeout):
    """This branch: publish, THEN take the lock; caller re-reads under it."""
    done = threading.Event()

    def _run():
        try:
            ex.shutdown(wait=True)
        except (RuntimeError, OSError, ValueError):
            pass
        finally:
            done.set()
            with census.lock:
                try:
                    census.items.remove(ex)
                except ValueError:
                    pass

    threading.Thread(target=_run, name='lumenairy-newton-pool-close',
                     daemon=True).start()
    if done.wait(timeout):
        return True
    with census.lock:
        if done.is_set():
            return True
        census.timeouts += 1
        census.items.append(ex)
    return False


IMPLS = {'pre': bounded_pre, 'oneline': bounded_oneline,
         'shipped': bounded_shipped}


# ==========================================================================
# Interleavers: both arrival orders, by construction.
# ==========================================================================
class _OrderedLock:
    """A lock that forces one named thread through its section FIRST.

    ``threading.Lock`` makes no fairness promise, so the order cannot be
    obtained by sleeping.  ``first='helper'`` admits the
    ``lumenairy-newton-pool-close`` thread immediately and parks everyone
    else until it has released; ``first='caller'`` does the opposite.
    """

    def __init__(self, real, first='helper', patience=6.0):
        self._real = real
        self._first = first
        self._patience = patience
        self._passed = threading.Event()

    @staticmethod
    def _is_helper():
        return threading.current_thread().name == 'lumenairy-newton-pool-close'

    def _privileged(self):
        return self._is_helper() if self._first == 'helper' else (
            not self._is_helper())

    def acquire(self, *a, **kw):
        if not self._privileged():
            self._passed.wait(self._patience)
        return self._real.acquire(*a, **kw)

    def release(self):
        priv = self._privileged()
        out = self._real.release()
        if priv:
            self._passed.set()
        return out

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def _settle(pred, seconds=8.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def run_race(impl_name, order, reps, bound=0.05, park=0.30):
    """One implementation x one arrival order, ``reps`` times."""
    fn = IMPLS[impl_name]
    leaked, rets, errs = 0, [], []
    for _ in range(reps):
        census = Census(lock=_OrderedLock(threading.Lock(), first=order))
        pool = _StubPool()
        box = {}

        def _call(_c=census, _p=pool, _b=box):
            try:
                _b['ret'] = fn(_c, _p, bound)
            except BaseException as exc:           # noqa: BLE001
                _b['exc'] = f'{type(exc).__name__}: {exc}'

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        time.sleep(park)                 # the bound has certainly expired
        pool.release()                   # helper's join returns NOW
        t.join(30.0)
        ok = _settle(lambda _c=census, _p=pool: _p not in _c.items)
        if not ok:
            leaked += 1
        rets.append(box.get('ret'))
        if 'exc' in box:
            errs.append(box['exc'])
    return {'impl': impl_name, 'order': order, 'reps': reps,
            'leaked': leaked, 'leak_rate': round(leaked / reps, 3),
            'returns': sorted({str(r) for r in rets}),
            'errors': errs}


def run_plain_expiry(impl_name, reps, bound=0.05, park=0.30):
    """No interleaver: the ordinary expiry the defect was measured on."""
    fn = IMPLS[impl_name]
    census = Census()
    pools = []
    for _ in range(reps):
        p = _StubPool()
        pools.append(p)
        fn(census, p, bound)
    in_list_before = len(census.items)
    for p in pools:
        p.release()
    settled = _settle(lambda: len(census.items) == 0)
    return {'impl': impl_name, 'reps': reps,
            'census_while_outstanding': in_list_before,
            'census_after_completion': len(census.items),
            'timeouts_counted': census.timeouts,
            'returns_to_empty': settled}


# ==========================================================================
# Attacks on the SHIPPED ordering
# ==========================================================================
def attack_helper_crash(reps):
    """The helper's shutdown raises something it does not catch."""
    out = []
    for cls in (_Uncatchable, KeyboardInterrupt, MemoryError):
        census = Census()
        pools = []
        for _ in range(reps):
            p = _StubPool(raise_cls=cls)
            pools.append(p)
            bounded_shipped(census, p, 0.05)
        outstanding = len(census.items)
        for p in pools:
            p.release()
        settled = _settle(lambda: len(census.items) == 0)
        out.append({'raised_by_helper': cls.__name__,
                    'census_while_outstanding': outstanding,
                    'census_after': len(census.items),
                    'returns_to_empty': settled})
    return out


def attack_double_abandon(reps):
    """The same executor abandoned and bounded-torn-down at once.

    Models ``_abandon_pool``'s reaper (append now, remove when its
    non-joining shutdown returns) racing an expiring bounded teardown of the
    SAME object.  Counts BOTH failure modes: a leak (entry survives) and a
    premature/double removal (the census goes NEGATIVE relative to the
    outstanding teardowns, i.e. an entry that is still outstanding is gone).
    """
    leaks = premature = 0
    for _ in range(reps):
        census = Census()
        pool = _StubPool()
        reaper_gate = threading.Event()

        def _reap(_c=census, _p=pool, _g=reaper_gate):
            with _c.lock:
                _c.items.append(_p)      # _abandon_pool appends immediately
            _g.wait(30.0)                # its shutdown(wait=False) returns
            with _c.lock:
                try:
                    _c.items.remove(_p)
                except ValueError:
                    pass

        r = threading.Thread(target=_reap, name='lumenairy-newton-pool-reaper',
                             daemon=True)
        r.start()
        time.sleep(0.02)
        box = {}

        def _call(_c=census, _p=pool, _b=box):
            _b['ret'] = bounded_shipped(_c, _p, 0.05)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        time.sleep(0.30)
        # while the reaper is STILL outstanding, is its entry still there?
        with census.lock:
            present_while_reaper_outstanding = pool in census.items
        if not present_while_reaper_outstanding:
            premature += 1
        pool.release()
        reaper_gate.set()
        t.join(30.0)
        r.join(30.0)
        if not _settle(lambda _c=census, _p=pool: _p not in _c.items):
            leaks += 1
    return {'reps': reps, 'leaks': leaks,
            'premature_removals_of_the_reapers_entry': premature,
            'note': ('a premature removal means the bounded helper dropped '
                     'an entry it never added -- the census then '
                     'under-reports a teardown that is still outstanding')}


def attack_unconditional_remove(reps=6):
    """The helper removes UNCONDITIONALLY -- can it drop someone else's entry?

    ``_shutdown_pool_bounded``'s helper runs ``_ABANDONED_POOLS.remove(ex)``
    in its ``finally`` whether or not the caller ever appended.  If the same
    executor is in the census for ANOTHER reason -- an ``_abandon_pool``
    whose reaper is still outstanding -- a bounded teardown of that executor
    that COMPLETES INSIDE ITS BOUND (so the caller never appends) still
    removes an entry, and the entry it removes is the reaper's.

    Driven here: the reaper appends and stays outstanding; a bounded
    teardown with a bound it comfortably meets runs against the same object;
    the census is then read while the reaper is STILL outstanding.
    """
    dropped = 0
    for _ in range(reps):
        census = Census()
        pool = _StubPool()
        pool.release()                  # a bounded shutdown will return AT ONCE
        gate = threading.Event()

        def _reap(_c=census, _p=pool, _g=gate):
            with _c.lock:
                _c.items.append(_p)     # _abandon_pool appends immediately
            _g.wait(30.0)               # ... and stays outstanding
            with _c.lock:
                try:
                    _c.items.remove(_p)
                except ValueError:
                    pass

        r = threading.Thread(target=_reap, name='lumenairy-newton-pool-reaper',
                             daemon=True)
        r.start()
        time.sleep(0.05)
        bounded_shipped(census, pool, 10.0)            # completes in-bound
        time.sleep(0.15)
        with census.lock:
            still_there = pool in census.items
        if not still_there:
            dropped += 1
        gate.set()
        r.join(30.0)
    return {'reps': reps,
            'reapers_entry_dropped_by_the_bounded_helper': dropped,
            'bounded_teardown_returned_in_bound': True,
            'severity': ('the census UNDER-reports an outstanding teardown; '
                         'no library path passes one executor to both '
                         'mechanisms today, so this is latent'),
            'suggested_edit': ("have the caller record, under "
                               "_ABANDONED_POOLS_LOCK, that it appended, and "
                               "have the helper remove only then")}


def attack_boundary(reps, bound=0.05):
    """No interleaver; the wait expires as the helper publishes."""
    leaks, rets = 0, []
    for i in range(reps):
        census = Census()
        pool = _StubPool()
        box = {}

        def _call(_c=census, _p=pool, _b=box):
            _b['ret'] = bounded_shipped(_c, _p, bound)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        # release at exactly the bound, scanned across it
        time.sleep(bound * (0.85 + 0.03 * (i % 11)))
        pool.release()
        t.join(30.0)
        if not _settle(lambda _c=census, _p=pool: _p not in _c.items):
            leaks += 1
        rets.append(box.get('ret'))
    return {'reps': reps, 'leaks': leaks,
            'returns': {str(k): rets.count(k) for k in set(rets)}}


# ==========================================================================
# The REAL library function, under the same adverse interleaver.
# ==========================================================================
def run_real_library(LT, order, reps, bound=0.05, park=0.30):
    real_lock = LT._ABANDONED_POOLS_LOCK
    leaked, rets = 0, []
    base_len = len(LT._ABANDONED_POOLS)
    base_to = LT._POOL_SHUTDOWN_TIMEOUTS
    try:
        for _ in range(reps):
            LT._ABANDONED_POOLS_LOCK = _OrderedLock(threading.Lock(),
                                                    first=order)
            pool = _StubPool()
            box = {}

            def _call(_p=pool, _b=box):
                _b['ret'] = LT._shutdown_pool_bounded(_p, bound)

            t = threading.Thread(target=_call, daemon=True)
            t.start()
            time.sleep(park)
            pool.release()
            t.join(30.0)
            if not _settle(lambda _p=pool: _p not in LT._ABANDONED_POOLS):
                leaked += 1
            rets.append(box.get('ret'))
    finally:
        LT._ABANDONED_POOLS_LOCK = real_lock
    return {'order': order, 'reps': reps, 'leaked': leaked,
            'returns': sorted({str(r) for r in rets}),
            'census_len_before': base_len,
            'census_len_after': len(LT._ABANDONED_POOLS),
            'timeouts_before': base_to,
            'timeouts_after': LT._POOL_SHUTDOWN_TIMEOUTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=12)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    from lumenairy.elements import _lens_traced as LT

    out = {'python': sys.version, 'platform': sys.platform,
           'lumenairy_file': lumenairy.__file__, 'reps': args.reps}

    out['plain_expiry'] = [run_plain_expiry(k, 3) for k in IMPLS]
    print('plain_expiry', json.dumps(out['plain_expiry'], indent=1),
          flush=True)

    races = []
    for impl in IMPLS:
        for order in ('helper', 'caller'):
            r = run_race(impl, order, args.reps)
            races.append(r)
            print('race', json.dumps(r), flush=True)
    out['races'] = races

    out['attack_helper_crash'] = attack_helper_crash(3)
    print('helper_crash', json.dumps(out['attack_helper_crash']), flush=True)
    out['attack_double_abandon'] = attack_double_abandon(max(4, args.reps // 2))
    print('double_abandon', json.dumps(out['attack_double_abandon']),
          flush=True)
    out['attack_boundary'] = attack_boundary(args.reps)
    print('boundary', json.dumps(out['attack_boundary']), flush=True)
    out['attack_unconditional_remove'] = attack_unconditional_remove(6)
    print('uncond_remove', json.dumps(out['attack_unconditional_remove']),
          flush=True)

    out['real_library'] = [run_real_library(LT, o, max(4, args.reps // 2))
                           for o in ('helper', 'caller')]
    print('real_library', json.dumps(out['real_library'], indent=1),
          flush=True)

    out['verdict'] = {
        'pre_leaks_on_a_plain_expiry': any(
            r['impl'] == 'pre' and r['census_after_completion'] > 0
            for r in out['plain_expiry']),
        'oneline_leaks_under_helper_first': any(
            r['impl'] == 'oneline' and r['order'] == 'helper'
            and r['leaked'] > 0 for r in races),
        'shipped_leaks_under_either_order': any(
            r['impl'] == 'shipped' and r['leaked'] > 0 for r in races),
        'shipped_real_library_leaks': any(
            r['leaked'] > 0 for r in out['real_library']),
        'shipped_survives_helper_crash': all(
            r['returns_to_empty'] for r in out['attack_helper_crash']),
        'shipped_boundary_leaks': out['attack_boundary']['leaks'],
        'shipped_premature_removal_on_double_abandon':
            out['attack_double_abandon'][
                'premature_removals_of_the_reapers_entry'],
        'shipped_unconditional_remove_drops_a_foreign_entry':
            out['attack_unconditional_remove'][
                'reapers_entry_dropped_by_the_bounded_helper'],
    }
    with open(args.out, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', json.dumps(out['verdict'], indent=1))
    print('wrote', args.out)


if __name__ == '__main__':
    main()

"""P2 -- drive the two-arm traced-carrier chain WITHOUT pytest, instrumented.

Reproduces the b4 gate-(c) fixture (``tests/unit/test_audit2609_b4_collins
_transport.py::p5_arms``) as a plain script, with every Newton-pool decision
logged with a thread id and a monotonic timestamp:

  resolve   ``_newton_resolve_workers`` -- the RAM/``__main__`` clamp; its
            answer is what keys the pool, so two threads that read different
            free memory ask for different pools
  getpool   ``_get_persistent_worker_pool`` -- entry, whether it REBUILT (the
            cached pool had a different worker count), and exit
  close     ``close_worker_pool`` -- entry and exit; the release-gate dump
            caught a thread wedged between these two
  defer     ``_note_pool_deferral`` / ``_pool_reuse_is_likely`` -- the cost
            gate that decides whether a pool is used at all

The script prints one JSON object per event on stdout, so a wedged phase is a
missing exit line.  ``faulthandler.dump_traceback_later(..., exit=True)`` is
armed throughout, so the probe can never join the hang it is studying.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import threading
import time
import warnings

# Module body kept free of top-level WORK: a spawn worker re-imports
# __main__, and ``_script_has_main_guard`` (rightly) refuses the pool for
# a script whose body does anything but imports, defs and literals.
T0 = 0.0
_EMIT_LOCK = None


def emit(event, **kw):
    kw['event'] = event
    kw['t'] = round(time.monotonic() - T0, 4)
    kw['tid'] = threading.get_ident()
    kw['tname'] = threading.current_thread().name
    line = json.dumps(kw)
    with _EMIT_LOCK:
        sys.stdout.write(line + '\n')
        sys.stdout.flush()


def instrument(lt):
    """Wrap the pool entry points of ``lumenairy.elements._lens_traced``.

    ``_invert_newton_parallel`` is a closure that looks its collaborators up
    as MODULE globals at call time, so replacing the module attributes is
    enough -- no library edit is needed to observe the shipped path.
    """
    _get = lt._get_persistent_worker_pool
    _close = lt.close_worker_pool
    _resolve = lt._newton_resolve_workers
    _note = lt._note_pool_deferral
    _likely = lt._pool_reuse_is_likely

    def get_persistent_worker_pool(n_workers):
        cached = lt._PERSISTENT_POOL_NWORKERS
        rebuild = (lt._PERSISTENT_POOL is not None and cached != n_workers)
        emit('getpool_enter', n_workers=int(n_workers), cached=cached,
             rebuild=bool(rebuild),
             pool_alive=lt._PERSISTENT_POOL is not None)
        try:
            ex = _get(n_workers)
        except BaseException as exc:  # noqa: BLE001
            emit('getpool_raise', exc=type(exc).__name__, text=str(exc)[:300])
            raise
        emit('getpool_exit', n_workers=int(n_workers), pool=id(ex),
             broken=bool(getattr(ex, '_broken', None)))
        return ex

    def close_worker_pool():
        emit('close_enter', pool=id(lt._PERSISTENT_POOL),
             nworkers=lt._PERSISTENT_POOL_NWORKERS)
        t = time.monotonic()
        try:
            _close()
        finally:
            emit('close_exit', seconds=round(time.monotonic() - t, 4))

    def newton_resolve_workers(requested, n_total, fit_points, **kw):
        n = _resolve(requested, n_total, fit_points, **kw)
        emit('resolve', requested=int(requested), n_total=int(n_total),
             fit_points=int(fit_points), resolved=int(n))
        return n

    def note_pool_deferral(n_workers, cost_class, n_points, seconds):
        emit('defer', n_workers=int(n_workers), cost_class=str(cost_class),
             n_points=int(n_points), seconds=round(float(seconds), 4))
        return _note(n_workers, cost_class, n_points, seconds)

    def pool_reuse_is_likely(n_workers, cost_class, n_points):
        r = _likely(n_workers, cost_class, n_points)
        emit('promote_query', n_workers=int(n_workers),
             cost_class=str(cost_class), n_points=int(n_points),
             promoted=bool(r))
        return r

    lt._get_persistent_worker_pool = get_persistent_worker_pool
    lt.close_worker_pool = close_worker_pool
    lt._newton_resolve_workers = newton_resolve_workers
    lt._note_pool_deferral = note_pool_deferral
    lt._pool_reuse_is_likely = pool_reuse_is_likely


def _gauss_env(n, dx, w):
    import numpy as np
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


def run_chain(args):
    import numpy as np

    from lumenairy.elements import apply_real_lens_traced
    from lumenairy.glass import GLASS_REGISTRY
    from lumenairy.propagators import carrier as C
    from lumenairy.propagators.propagation import angular_spectrum_propagate
    from lumenairy.raytrace.seidel import system_abcd_prescription

    lam, ng = 1.31e-6, 1.5168
    GLASS_REGISTRY['_B4GLASS'] = (lambda wl: ng)
    sd = 10e-3

    def presc():
        return {'surfaces': [
            {'radius': 51.68e-3, 'glass_before': 'air',
             'glass_after': '_B4GLASS', 'semi_diameter': sd},
            {'radius': -51.68e-3, 'glass_before': '_B4GLASS',
             'glass_after': 'air', 'semi_diameter': sd}],
            'thicknesses': [5e-3], 'aperture_diameter': 2 * sd,
            'stop_index': 0}

    M, _, _, _ = system_abcd_prescription(presc(), lam)
    w0, z1 = 6.0e-6, 30e-3
    zR = np.pi * w0 ** 2 / lam
    r_in = z1 * (1.0 + (zR / z1) ** 2)
    w_l = w0 * np.sqrt(1.0 + (z1 / zR) ** 2)
    n = args.n
    dx = 2 * 3.0 * w_l / n
    env0 = _gauss_env(n, dx, w_l)
    tk = dict(amplitude_model='ray_density', preserve_input_phase='remap',
              remap_sampling='full')
    gap = 40e-3
    R_a = (M[0, 0] * r_in + M[0, 1]) / (M[1, 0] * r_in + M[1, 1])
    R_b = R_a + gap
    R_c = (M[0, 0] * R_b + M[0, 1]) / (M[1, 0] * R_b + M[1, 1])
    groups = [{'prescription': presc(), 'gap_before': 0.0},
              {'prescription': presc(), 'gap_before': gap}]

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if args.brute:
            emit('brute_enter')
            Eb = apply_real_lens_traced(
                np.asarray(C.carrier_referenced_reconstruct(
                    env0, r_in, lam, dx)),
                prescription=presc(), wavelength=lam, dx=dx, carrier=r_in,
                ray_subsample=args.ray_subsample, **tk)
            Eb = angular_spectrum_propagate(np.asarray(Eb), gap, lam, dx)
            Eb = apply_real_lens_traced(
                Eb, prescription=presc(), wavelength=lam, dx=dx, carrier=R_b,
                ray_subsample=args.ray_subsample, **tk)
            Eb = angular_spectrum_propagate(np.asarray(Eb), -R_c * 0.5, lam,
                                            dx)
            out['brute'] = float(np.abs(np.asarray(Eb)).sum())
            emit('brute_exit', sum_abs=out['brute'])
        for tr in args.transports:
            emit('chain_enter', transport=tr)
            r = C.propagate_traced_carrier_chain(
                env0, groups, lam, dx, r_in=r_in,
                ray_subsample=args.ray_subsample,
                final_distance=-R_c * 0.5, final_leg='paraxial',
                traced_kwargs=tk, carrier_reference='sphere', transport=tr)
            out[tr] = float(np.abs(np.asarray(r.field)).sum())
            emit('chain_exit', transport=tr, sum_abs=out[tr])
    return out


def main():
    global T0, _EMIT_LOCK
    T0 = time.monotonic()
    _EMIT_LOCK = threading.Lock()
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2048)
    ap.add_argument('--ray-subsample', type=int, default=2)
    ap.add_argument('--transports', default='sziklas,collins')
    ap.add_argument('--brute', action='store_true')
    ap.add_argument('--dump-after', type=float, default=900.0)
    ap.add_argument('--no-instrument', action='store_true')
    args = ap.parse_args()
    args.transports = [t for t in args.transports.split(',') if t]

    faulthandler.enable()
    faulthandler.dump_traceback_later(args.dump_after, exit=True)

    import lumenairy
    from lumenairy.elements import _lens_traced as lt
    emit('import', lumenairy=lumenairy.__file__,
         version=lumenairy.__version__, python=sys.version.split()[0],
         pid=os.getpid())
    if not args.no_instrument:
        instrument(lt)

    t = time.monotonic()
    try:
        out = run_chain(args)
    except BaseException as exc:  # noqa: BLE001
        import traceback
        emit('chain_raise', exc=type(exc).__name__, text=str(exc)[:600],
             tb=traceback.format_exc()[-2000:])
        faulthandler.cancel_dump_traceback_later()
        os._exit(3)
    emit('done', seconds=round(time.monotonic() - t, 3), out=out)
    faulthandler.cancel_dump_traceback_later()
    # Leave the interpreter to finalize normally so a wedged manager thread
    # would show up as an exit hang; the caller's `timeout` bounds it.
    return 0


if __name__ == '__main__':
    sys.exit(main())

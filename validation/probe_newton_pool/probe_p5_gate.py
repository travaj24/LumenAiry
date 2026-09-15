"""P5 -- which gate keeps a 262 144-point traced call off the pool?

Runs the ``test_fix_newton_pool_memory`` chain fixture once at
``n_workers=8`` and reports every input the pool gate in
``_invert_newton_parallel`` reads, so the answer is a measurement rather than
a reading of the source.
"""
from __future__ import annotations

import faulthandler
import json
import sys
import warnings

import numpy as np

_WL = 1.31e-6
_W0 = 1.0e-3
_N, _RS = 1024, 2


def emit(event, **kw):
    kw['event'] = event
    sys.stdout.write(json.dumps(kw) + '\n')
    sys.stdout.flush()


def _doublet(ap):
    surfs, before = [], 'air'
    for R, g in ((61.5e-3, 'N-BK7'), (-45.0e-3, 'N-SF5'), (-128.0e-3, 'air')):
        surfs.append({'radius': R, 'glass_before': before, 'glass_after': g,
                      'conic': 0.0, 'radius_y': None, 'conic_y': None,
                      'aspheric_coeffs': None, 'aspheric_coeffs_y': None})
        before = g
    return {'name': 'doublet', 'aperture_diameter': ap,
            'thicknesses': [4.0e-3, 2.5e-3], 'surfaces': surfs}


def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(600.0, exit=True)
    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    emit('import', file=la.__file__)

    _resolve = LT._newton_resolve_workers
    _class = LT._newton_cost_class
    _likely = LT._pool_reuse_is_likely
    _get = LT._get_persistent_worker_pool
    seen = []

    def resolve(requested, n_total, fit_points, **kw):
        n = _resolve(requested, n_total, fit_points, **kw)
        seen.append(('resolve', int(requested), int(n_total),
                     int(fit_points), int(n)))
        emit('resolve', requested=int(requested), n_total=int(n_total),
             fit_points=int(fit_points), resolved=int(n),
             backend_refused=bool(LT._POOL_BACKEND_REFUSED),
             pool_alive=LT._PERSISTENT_POOL is not None,
             pool_n=LT._PERSISTENT_POOL_NWORKERS,
             min_pixels=LT._POOL_MIN_PIXELS,
             min_pixels_warm=LT._POOL_MIN_PIXELS_WARM)
        return n

    def cost_class(fit):
        c = _class(fit)
        emit('cost_class', cls=str(c))
        return c

    def likely(n_workers, cls, n_points):
        r = _likely(n_workers, cls, n_points)
        emit('promote_query', n_workers=int(n_workers), cls=str(cls),
             n_points=int(n_points), promoted=bool(r))
        return r

    def get_pool(n):
        emit('getpool', n=int(n))
        return _get(n)

    LT._newton_resolve_workers = resolve
    LT._newton_cost_class = cost_class
    LT._pool_reuse_is_likely = likely
    LT._get_persistent_worker_pool = get_pool

    ap = 1.2 * 2.0 * _W0
    dx = float(2.2 * max(ap, 3.0 * _W0) / _N)
    x = (np.arange(_N) - _N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env0 = np.exp(-(X ** 2 + Y ** 2) / _W0 ** 2).astype(np.complex128)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = la.propagate_traced_carrier_chain(
            env0, [{'prescription': _doublet(ap), 'gap_before': 0.0}],
            _WL, dx, r_in=np.inf, ray_subsample=_RS, n_workers=8,
            final_distance=0.0,
            traced_kwargs=dict(parallel_amp=False, on_undersample='silent',
                               newton_fit='polynomial'))
    emit('warnings', msgs=[str(w.message)[:220] for w in rec])
    emit('done', shape=list(np.asarray(res.field).shape),
         backend_refused=bool(LT._POOL_BACKEND_REFUSED),
         pool_alive=LT._PERSISTENT_POOL is not None)
    LT.close_worker_pool()
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()

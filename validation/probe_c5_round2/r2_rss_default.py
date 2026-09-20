"""WP-C5 round 2 -- "warn, not refuse", measured on the DEFAULT path.

The decision recorded in the CHANGELOG is that the dense reconstruction WARNS
rather than raises below :func:`_dense_budget_floor_bytes`, because the floor
crosses the shipped ``mem_budget_mb`` default at N = 1706 and a refusal would
turn a call that completes today into a hard error.  This probe runs that
call -- default budget, default accounting, a grid past the crossing -- and
records what it costs in RESIDENT SET beside what the same call costs under
``'legacy'``, which is the 5.48.x behaviour.

RSS is sampled in a thread: a before/after difference is not a peak, because
the transient is freed before the call returns and the allocator keeps the
arena.  The reading is process-wide, so it is an UPPER bound on what the call
itself holds.

Usage (BLAS pinned on the COMMAND LINE)::

    PYTHONPATH=<tree> python validation/probe_c5_round2/r2_rss_default.py OUT.json
"""
import gc
import inspect
import os
import sys
import threading
import warnings

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), 'probe_verify_c5'))

import numpy as np                                            # noqa: E402
import psutil                                                 # noqa: E402

from _vd import write                                         # noqa: E402

from lumenairy.propagators import gbd as G                    # noqa: E402

_PROC = psutil.Process()


class RssPeak:
    def __init__(self, interval=0.002):
        self.interval, self.peak, self.base = float(interval), 0, 0
        self._stop, self._th = threading.Event(), None

    def __enter__(self):
        gc.collect()
        self.base = self.peak = _PROC.memory_info().rss
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, _PROC.memory_info().rss)
            except Exception:
                return
            self._stop.wait(self.interval)

    def __exit__(self, *a):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)
        return False

    @property
    def delta(self):
        return int(self.peak - self.base)


def bundle(n=48, seed=5):
    rng = np.random.default_rng(seed)
    return G.BeamletBundle(
        positions=rng.normal(0.0, 1.5e-4, size=(n, 3)),
        directions=np.zeros((n, 3)),
        Q=np.full(n, 1.0 / (1.2e-3 - 0.03j), dtype=np.complex128),
        amplitude=(rng.normal(size=n)
                   + 1j * rng.normal(size=n)).astype(np.complex128),
        waist0=np.full(n, 1.2e-3))


def run(mode, N, b):
    """One DEFAULT-path call -- no ``mem_budget_mb`` passed at all."""
    old = G.DENSE_MEM_BUDGET_ACCOUNTING
    if mode is not None:
        G.DENSE_MEM_BUDGET_ACCOUNTING = mode
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            with RssPeak() as rw:
                E = np.asarray(G.reconstruct_field_from_beamlets(
                    b, Ny=N, Nx=N, dx=5.0e-7, wavelength=1.064e-6))
            msgs = [str(w.message) for w in rec]
        return dict(
            accounting=G.DENSE_MEM_BUDGET_ACCOUNTING,
            completes=True, finite=bool(np.all(np.isfinite(E))),
            shape=list(E.shape), rss_peak_mb=rw.peak / 1e6,
            rss_delta_mb=rw.delta / 1e6,
            floor_notices=sum(('mem_budget_mb' in m and 'floor' in m)
                              for m in msgs),
            notice=next((m for m in msgs
                         if 'mem_budget_mb' in m and 'floor' in m), None),
            peak_abs_field=float(np.abs(E).max()))
    finally:
        G.DENSE_MEM_BUDGET_ACCOUNTING = old


def main(out_path):
    default_budget = inspect.signature(
        G.reconstruct_field_from_beamlets).parameters['mem_budget_mb'].default
    per_cell = G._DENSE_FIXED_CELL_BYTES + G._DENSE_CELL_BYTES_MEASURED
    n_star = float(np.sqrt(default_budget * 1e6 / per_cell))
    N = 2048
    b = bundle()
    res = dict(
        default_mem_budget_mb=float(default_budget),
        floor_bytes_per_cell=float(per_cell),
        n_star=n_star,
        first_binding_square_grid=int(np.floor(n_star)) + 1,
        floor_at_1705_mb=G._dense_budget_floor_bytes(1705, 1705) / 1e6,
        floor_at_1706_mb=G._dense_budget_floor_bytes(1706, 1706) / 1e6,
        N=N, n_beamlets=48,
        floor_at_N_mb=G._dense_budget_floor_bytes(N, N) / 1e6,
        shipped_accounting=G.DENSE_MEM_BUDGET_ACCOUNTING)
    # the honest arm FIRST, so its pages are fresh and the reading is not
    # served out of an arena the loud arm already grew
    res['measured'] = run('measured', N, b)
    res['legacy'] = run('legacy', N, b)
    res['ratio_legacy_over_measured'] = (
        res['legacy']['rss_peak_mb'] / res['measured']['rss_peak_mb'])
    write(out_path, res)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1
         else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'r2_rss_default_win.json'))

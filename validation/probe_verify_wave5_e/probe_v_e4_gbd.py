"""VERIFY-WAVE5-E / E4 (D3): the 'measured' budget bound, re-measured
independently.

Nothing is imported from the builder's probes: the bundle, the chunk formula
and the tracemalloc harness are written here from the public API and the two
module constants, so the numbers are a re-measurement and not a re-run.

A second bundle (different beamlet count, different Q, different waist) and a
second grid (N = 192) are swept as well, so the "bound only above one beamlet
column" statement is checked as a FUNCTION of the floor rather than at one N.
"""
import json
import os
import sys
import tracemalloc
import warnings

import numpy as np

import lumenairy as la
from lumenairy.propagators import gbd as G


def _bundle(n, seed=4242, waist=1.2e-3):
    rng = np.random.default_rng(seed)
    kw = dict(positions=rng.normal(0.0, 1.7e-4, size=(n, 3)),
              Q=np.full(n, 1.0 / (0.8e-3 - 0.03j), dtype=np.complex128),
              amplitude=(rng.normal(size=n)
                         + 1j * rng.normal(size=n)).astype(np.complex128),
              waist0=np.full(n, waist))
    try:
        return G.BeamletBundle(directions=np.zeros((n, 3)), **kw)
    except TypeError:
        return G.BeamletBundle(**kw)


def _chunk(N, budget_mb, mode, requested=4096, n=None):
    cell = (G._DENSE_CELL_BYTES_MEASURED if mode == 'measured'
            else G._DENSE_CELL_BYTES_LEGACY)
    per_col = N * N * cell
    ch = min(requested, max(1, int(budget_mb * 1e6 / max(1.0, per_col))))
    return ch if n is None else min(ch, n)


def _run(mode, bundle, N, budget_mb):
    old = G.DENSE_MEM_BUDGET_ACCOUNTING
    G.DENSE_MEM_BUDGET_ACCOUNTING = mode
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            G.reconstruct_field_from_beamlets(
                bundle, Ny=N, Nx=N, dx=2.0e-6, wavelength=1.0e-6,
                chunk_beamlets=4096, mem_budget_mb=budget_mb)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        G.DENSE_MEM_BUDGET_ACCOUNTING = old


def main():
    res = dict(lumenairy_file=la.__file__, version=la.__version__,
               python=sys.version.split()[0], numpy=np.__version__,
               platform=sys.platform,
               shipped_default=G.DENSE_MEM_BUDGET_ACCOUNTING,
               legacy_const=G._DENSE_CELL_BYTES_LEGACY,
               measured_const=G._DENSE_CELL_BYTES_MEASURED,
               env_budget=os.environ.get('LUMENAIRY_MEM_BUDGET_MB'),
               cells=[])
    print('lumenairy.__file__ =', la.__file__, flush=True)
    for N, nb in ((256, 1024), (192, 768)):
        floor_mb = N * N * (48.0 + G._DENSE_CELL_BYTES_MEASURED) / 1e6
        b = _bundle(nb)
        for budget in (512.0, 16.0, 4.0, 1.0):
            peak = _run('measured', b, N, budget)
            row = dict(N=N, n_beamlets=nb, budget_mb=budget,
                       chunk=_chunk(N, budget, 'measured', n=nb),
                       one_column_floor_mb=floor_mb,
                       peak_mb=peak / 1e6, ratio=peak / (budget * 1e6),
                       budget_above_floor=bool(budget > floor_mb))
            res['cells'].append(row)
            print('N=%d budget %7.1f MB floor %.2f MB -> chunk %3d peak '
                  '%8.2f MB %7.2fx  %s'
                  % (N, budget, floor_mb, row['chunk'], row['peak_mb'],
                     row['ratio'],
                     'bounded' if row['ratio'] < 1.0 else 'NOT bounded'),
                  flush=True)
    # the scope statement, as a two-sided decision
    by = {(c['N'], c['budget_mb']): c for c in res['cells']}
    res['ratio_4mb_N256'] = by[(256, 4.0)]['ratio']
    res['ratio_1mb_N256'] = by[(256, 1.0)]['ratio']
    res['floor_mb_N256'] = by[(256, 4.0)]['one_column_floor_mb']
    res['scope_holds_every_N'] = all(
        (c['ratio'] < 1.0) == c['budget_above_floor'] or
        not c['budget_above_floor']
        for c in res['cells'])
    res['bounded_iff_above_floor'] = [
        dict(N=c['N'], budget=c['budget_mb'], above_floor=c['budget_above_floor'],
             bounded=bool(c['ratio'] < 1.0)) for c in res['cells']]
    print(json.dumps(res, indent=1))


main()

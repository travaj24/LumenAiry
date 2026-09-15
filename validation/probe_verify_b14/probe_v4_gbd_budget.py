"""VERIFY-B14 arm 4 -- the dense GBD reconstruction's memory budget, re-measured.

Three independent things are read here, not one:

1. THE ANALYTIC BYTE COUNT.  The scalar ``has_dirs`` branch of
   ``reconstruct_field_from_beamlets`` holds, per (output cell x
   beamlet-column), ``dX`` + ``dY`` + ``rho2`` (3 x float64 = 24 B) and, at the
   peak of the ``arg`` / ``phase`` construction, the complex128 ``arg``, the
   ``1j*k*arg`` temporary and ``phase`` (3 x complex128 = 48 B) -- 72 B.  That
   is a derivation, not a reading, and it is what the tracemalloc ladder is
   compared against.

2. THE MEASURED PEAK, over a grid x budget x beamlet-count ladder, in both
   accounting modes, with the effective chunk recorded so the peak can be
   expressed per cell-column.

3. WHETHER ``'measured'`` MAKES THE BUDGET A BOUND *IN GENERAL* or only on
   the one cell the shipped test asserts -- the small-budget arms, where the
   chunk floors at 1 and the per-column transient alone exceeds the request.
"""
from __future__ import annotations

import json
import sys
import tracemalloc

import numpy as np

import lumenairy as la
from lumenairy.propagators import gbd as G

_DIRS = True


def _bundle(n, seed=0):
    rng = np.random.default_rng(seed)
    return G.BeamletBundle(
        positions=rng.normal(0.0, 2.0e-4, size=(n, 3)),
        directions=np.zeros((n, 3)) if _DIRS else None,
        Q=np.full(n, 1.0 / (1.0e-3 - 0.02j), dtype=np.complex128),
        amplitude=(rng.normal(size=n)
                   + 1j * rng.normal(size=n)).astype(np.complex128),
        waist0=np.full(n, 1.0e-3))


def _effective_chunk(N, budget_mb, mode, requested=4096, n=None):
    cell = (G._DENSE_CELL_BYTES_MEASURED if mode == 'measured'
            else G._DENSE_CELL_BYTES_LEGACY)
    per_col = N * N * cell
    mx = max(1, int(budget_mb * 1e6 / max(1.0, per_col)))
    ch = min(requested, mx)
    return ch if n is None else min(ch, n)


def _run(mode, bundle, N, budget_mb):
    G.DENSE_MEM_BUDGET_ACCOUNTING = mode
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        out = G.reconstruct_field_from_beamlets(
            bundle, Ny=N, Nx=N, dx=2.0e-6, wavelength=1.0e-6,
            chunk_beamlets=4096, mem_budget_mb=budget_mb)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return np.asarray(out), int(peak - base)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'v4_gbd.json'
    old = G.DENSE_MEM_BUDGET_ACCOUNTING
    res = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'shipped_default': old,
           'legacy_const': G._DENSE_CELL_BYTES_LEGACY,
           'measured_const': G._DENSE_CELL_BYTES_MEASURED,
           'windowed_const': G._WINDOWED_CELL_BYTES,
           'analytic_bytes_per_cell_col': 72.0,
           'rows': [], 'identity': {}}
    try:
        ladder = [(64, 512, 512.0), (64, 512, 64.0), (128, 1024, 512.0),
                  (192, 512, 512.0), (192, 512, 64.0), (256, 512, 64.0),
                  (256, 1024, 512.0), (256, 1024, 16.0), (256, 512, 4.0),
                  (256, 512, 1.0)]
        for (N, nb, budget) in ladder:
            b = _bundle(nb)
            for mode in ('legacy', 'measured'):
                ch = _effective_chunk(N, budget, mode, n=nb)
                _f, peak = _run(mode, b, N, budget)
                row = {'N': N, 'beamlets': nb, 'budget_mb': budget,
                       'mode': mode, 'chunk': ch,
                       'peak_mb': peak / 1e6,
                       'overrun': peak / (budget * 1e6),
                       'B_per_cell_col': peak / float(N * N * ch)}
                res['rows'].append(row)
                print('N=%-4d nb=%-5d budget=%-7.1f %-9s chunk=%-5d '
                      'peak=%8.1f MB  overrun=%6.2fx  B/cell-col=%6.1f'
                      % (N, nb, budget, mode, ch, row['peak_mb'],
                         row['overrun'], row['B_per_cell_col']), flush=True)
        # byte identity of the two accountings on the worst cell
        b = _bundle(1024)
        a, pa = _run('legacy', b, 256, 512.0)
        c, pc = _run('measured', b, 256, 512.0)
        scale = float(np.max(np.abs(a)))
        res['identity'] = {
            'equal_bytes': bool(np.array_equal(a, c)),
            'rel': float(np.max(np.abs(a - c))) / scale,
            'legacy_peak_mb': pa / 1e6, 'measured_peak_mb': pc / 1e6}
        print('identity', res['identity'])
        # an unknown value must behave as legacy
        b2 = _bundle(128)
        d, _ = _run('legacy', b2, 64, 512.0)
        e, _ = _run('not-a-mode', b2, 64, 512.0)
        res['unknown_is_legacy'] = bool(np.array_equal(d, e))
        print('unknown_is_legacy', res['unknown_is_legacy'])
    finally:
        G.DENSE_MEM_BUDGET_ACCOUNTING = old
    print('lumenairy:', la.__file__)
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()

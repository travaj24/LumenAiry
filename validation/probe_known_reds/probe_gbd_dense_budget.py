"""Bound the dense GBD reconstruction's working set against its own budget.

WHY.  Handoff section 5 records that a long pytest run on this box has twice
died with ``Windows fatal exception: access violation``, once inside the dense
``reconstruct_field_from_beamlets`` path of ``lumenairy/propagators/gbd.py``,
"which passes alone in 30 s".  A native-level fault in a pure-numpy loop that
only appears beside other heavy jobs is an allocation story, so the question
this probe answers is whether that loop's memory budget BOUNDS what it
allocates.

WHAT IS MEASURED.  ``reconstruct_field_from_beamlets`` sizes its chunk from

    _bytes_per_col = Ny * Nx * 16.0
    chunk = mem_budget_mb * 1e6 / _bytes_per_col

with the comment "bytes per beamlet-column of the dense working set ~
Ny*Nx*16 (the dX/dY/rho2/phase buffers)".  16 B/cell is the size of ONE
complex128 element; the four buffers named are three float64 (``dX``, ``dY``,
``rho2``) and one complex128 (``phase``), so the arithmetic does not match the
sentence.  The sibling WINDOWED path does this properly: ``_WINDOWED_CELL_BYTES
= 32.0`` with a written per-array tally (~26 B/cell measured, 32 with margin).

This probe runs the dense path under ``tracemalloc`` over a ladder of grid
sizes and chunk counts and reports the measured peak against the budget, as
BYTES PER (cell x beamlet-column), which is the quantity the constant is.
"""
import json
import os
import pathlib
import sys
import tracemalloc

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lumenairy as la  # noqa: E402
from lumenairy.propagators import gbd as G  # noqa: E402


def _bundle(n, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(0.0, 2.0e-4, size=(n, 3))
    Q = np.full(n, 1.0 / (1.0e-3 - 0.02j), dtype=np.complex128)
    amp = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)
    return G.BeamletBundle(positions=pos, Q=Q, amplitude=amp,
                           directions=np.zeros((n, 3)),
                           waist0=np.full(n, 1.0e-3))


def _measure(N, n_beamlets, chunk, budget_mb):
    b = _bundle(n_beamlets)
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    out = G.reconstruct_field_from_beamlets(
        b, Ny=N, Nx=N, dx=2.0e-6, wavelength=1.0e-6,
        chunk_beamlets=chunk, mem_budget_mb=budget_mb)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # the chunk the function ACTUALLY used, reproducing its own arithmetic
    eff_chunk = min(chunk, max(1, int(budget_mb * 1e6 / max(1.0, N * N * 16.0))))
    eff_chunk = min(eff_chunk, n_beamlets)
    return {
        'N': N, 'n_beamlets': n_beamlets, 'chunk_requested': chunk,
        'budget_mb': budget_mb, 'effective_chunk': eff_chunk,
        'peak_bytes': int(peak - base),
        'peak_mb': round((peak - base) / 1e6, 2),
        'budget_overrun_x': round((peak - base) / (budget_mb * 1e6), 3),
        'bytes_per_cell_col': round((peak - base) / (N * N * eff_chunk), 2),
        'out_nonzero': bool(np.any(out)),
    }


def main():
    assert 'lum_reds' in la.__file__, la.__file__
    rows = []
    for N in (64, 128, 192, 256):
        for n_b, chunk, budget in ((512, 2048, 512.0), (512, 2048, 64.0),
                                   (1024, 4096, 512.0)):
            try:
                rows.append(_measure(N, n_b, chunk, budget))
            except Exception as exc:                      # noqa: BLE001
                rows.append({'N': N, 'n_beamlets': n_b, 'chunk': chunk,
                             'budget_mb': budget,
                             'error': f'{type(exc).__name__}: {exc}'})
            print(json.dumps(rows[-1]))
    good = [r for r in rows if 'bytes_per_cell_col' in r]
    out = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'declared_bytes_per_cell_col': 16.0,
           'windowed_sibling_constant': float(G._WINDOWED_CELL_BYTES),
           'rows': rows,
           'measured_bytes_per_cell_col': {
               'min': min(r['bytes_per_cell_col'] for r in good),
               'max': max(r['bytes_per_cell_col'] for r in good)},
           'max_budget_overrun_x': max(r['budget_overrun_x'] for r in good),
           'env': {k: os.environ.get(k) for k in
                   ('OPENBLAS_CORETYPE', 'OPENBLAS_NUM_THREADS')}}
    print('declared bytes/cell-col: 16.0   measured:',
          out['measured_bytes_per_cell_col'])
    print('max budget overrun:', out['max_budget_overrun_x'], 'x')
    tag = os.environ.get('PROBE_TAG', 'default')
    dest = pathlib.Path(__file__).parent / f'gbd_dense_budget_{tag}.json'
    dest.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('wrote', dest)


if __name__ == '__main__':
    main()

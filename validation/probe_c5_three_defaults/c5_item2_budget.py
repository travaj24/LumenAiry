"""WP-C5 item 2 -- the dense GBD memory budget, measured.

Three things, in this order, because each depends on the one before it:

 1. **The two per-cell terms, DERIVED rather than quoted.**  The dense loop's
    live peak is affine in the chunk: ``peak = Ny*Nx*(fixed + c*chunk)``.  The
    ``fixed`` term is the grid and accumulator arrays that live outside the
    chunk loop (``Xg``, ``Yg``, ``out`` and the per-chunk reduction's surviving
    temporary); ``c`` is the per (cell x beamlet-column) cost the budget
    arithmetic uses.  Both are fitted here by least squares over a chunk
    ladder at a budget large enough never to bind, on the running build.
 2. **The floor**, ``Ny*Nx*(fixed + c)`` -- the peak at ``chunk = 1``, which is
    the smallest chunk the loop can take, so no accounting constant can put it
    under a smaller budget.  Compared against the library's published
    ``_dense_budget_floor_bytes``.
 3. **The two-sided reading**: a budget 1.5x the floor must BOUND the loop; a
    budget 0.5x the floor must be LOUD about it and must not pretend.

Plus the byte readings: ``'legacy'`` against the parent archive, ``'measured'``
against ``'legacy'`` (relative spread only), and one budget run twice.

Usage (BLAS and the env budget pinned on the COMMAND LINE)::

    PYTHONPATH=<tree> python validation/probe_c5_three_defaults/c5_item2_budget.py OUT.json
"""
import os
import sys
import tracemalloc
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                             # noqa: E402

from _digest import dig, write                                 # noqa: E402

from lumenairy.propagators import gbd as G                     # noqa: E402

DX, WL = 2.0e-6, 1.0e-6


def bundle(n=1024, seed=0):
    rng = np.random.default_rng(seed)
    return G.BeamletBundle(
        positions=rng.normal(0.0, 2.0e-4, size=(n, 3)),
        directions=np.zeros((n, 3)),
        Q=np.full(n, 1.0 / (1.0e-3 - 0.02j), dtype=np.complex128),
        amplitude=(rng.normal(size=n)
                   + 1j * rng.normal(size=n)).astype(np.complex128),
        waist0=np.full(n, 1.0e-3))


def run(mode, b, N, budget_mb, chunk=4096, record=False):
    """``(field, peak_bytes, warnings)`` for one accounting mode."""
    old = G.DENSE_MEM_BUDGET_ACCOUNTING
    G.DENSE_MEM_BUDGET_ACCOUNTING = mode
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            tracemalloc.start()
            try:
                tracemalloc.reset_peak()
                base = tracemalloc.get_traced_memory()[0]
                out = G.reconstruct_field_from_beamlets(
                    b, Ny=N, Nx=N, dx=DX, wavelength=WL,
                    chunk_beamlets=chunk, mem_budget_mb=budget_mb)
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
        msgs = [str(w.message) for w in rec]
    finally:
        G.DENSE_MEM_BUDGET_ACCOUNTING = old
    return np.asarray(out), int(peak - base), (msgs if record else [])


def fit_terms(N, b, chunks):
    """Least-squares ``peak/(Ny*Nx) = fixed + c*chunk`` over a chunk ladder.

    The budget is set far above anything the ladder can need, so the chunk is
    the one the caller passed and the fit reads the loop's own arithmetic
    rather than the budget's.
    """
    rows = []
    # WARM-UP, discarded.  The FIRST tracemalloc window in a process also
    # captures whatever the import graph allocates lazily on the first call
    # (measured: 838 B/cell at N = 256 against a model 144.5), which lands on
    # whichever chunk happens to be measured first and destroys the fit.  One
    # discarded run removes it; the ladder below then fits to 1e-5.
    run('legacy', b, N, 1.0e7, chunk=int(chunks[0]))
    for ch in chunks:
        _f, peak, _w = run('legacy', b, N, 1.0e7, chunk=ch)
        rows.append((ch, peak / float(N * N)))
    A = np.array([[1.0, float(ch)] for ch, _ in rows])
    y = np.array([v for _, v in rows])
    (fixed, c), res, _rank, _sv = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([fixed, c])
    worst = float(np.max(np.abs(pred - y) / y))
    return dict(chunks=[int(ch) for ch, _ in rows],
                bytes_per_cell=[float(v) for _, v in rows],
                fixed_b_per_cell=float(fixed), per_col_b_per_cell=float(c),
                worst_rel_dev=worst)


def main(out):
    res = {'shipped_default': G.DENSE_MEM_BUDGET_ACCOUNTING,
           'cell_bytes': {
               'legacy': G._DENSE_CELL_BYTES_LEGACY,
               'measured': G._DENSE_CELL_BYTES_MEASURED,
               'fixed': getattr(G, '_DENSE_FIXED_CELL_BYTES', None),
               'windowed': G._WINDOWED_CELL_BYTES}}
    digests = {}
    b = bundle()

    # --- 1. the two terms, fitted -----------------------------------------
    res['fit'] = {}
    for N in (256, 512):
        res['fit'][str(N)] = fit_terms(N, b, (1, 2, 4, 8, 16, 32))

    # --- 2. the floor ------------------------------------------------------
    floors = {}
    for N in (256, 512):
        f = res['fit'][str(N)]
        emp = float(N * N * (f['fixed_b_per_cell'] + f['per_col_b_per_cell']))
        lib = (G._dense_budget_floor_bytes(N, N)
               if hasattr(G, '_dense_budget_floor_bytes') else None)
        floors[str(N)] = {'empirical_bytes': emp, 'library_bytes': lib,
                          'library_over_empirical':
                              (None if lib is None else float(lib) / emp)}
    res['floor'] = floors

    # --- 3. two-sided, at 1.5x and 0.5x the LIBRARY floor ------------------
    two_sided = {}
    for N in (256, 512):
        lib = floors[str(N)]['library_bytes']
        if lib is None:
            continue
        row = {}
        for tag, mult in (('above_1.5x', 1.5), ('below_0.5x', 0.5)):
            mb = float(lib) * mult / 1e6
            _f, peak, msgs = run('measured', b, N, mb, record=True)
            row[tag] = {'budget_mb': mb, 'peak_bytes': peak,
                        'peak_over_budget': peak / (mb * 1e6),
                        'n_warnings': len(msgs),
                        'floor_named': any('floor' in m.lower()
                                           for m in msgs),
                        'messages': [m[:400] for m in msgs]}
        # and the legacy arm at the SAME budgets, for the contrast
        for tag, mult in (('legacy_above_1.5x', 1.5), ('legacy_below_0.5x',
                                                       0.5)):
            mb = float(lib) * mult / 1e6
            _f, peak, msgs = run('legacy', b, N, mb, record=True)
            row[tag] = {'budget_mb': mb, 'peak_bytes': peak,
                        'peak_over_budget': peak / (mb * 1e6),
                        'n_warnings': len(msgs)}
        two_sided[str(N)] = row
    res['two_sided'] = two_sided

    # --- 3b. the SWEEP: the worst ratio anywhere above the floor ----------
    # A single budget flatters the arithmetic -- the ratio peaks just after a
    # chunk increment -- so the honest statement is the worst reading over a
    # span.  Only meaningful where the library publishes a floor.
    sweep = {}
    for N in (256, 512):
        lib = floors[str(N)]['library_bytes']
        if lib is None:
            continue
        rows = []
        for mult in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0):
            mb = float(lib) * mult / 1e6
            _f, peak, msgs = run('measured', b, N, mb, record=True)
            rows.append({'mult': mult, 'budget_mb': mb, 'peak_bytes': peak,
                         'ratio': peak / (mb * 1e6),
                         'floor_notice': any('floor' in m for m in msgs)})
        sweep[str(N)] = {'rows': rows,
                         'worst_ratio': max(r['ratio'] for r in rows),
                         'worst_at_mult': max(rows,
                                              key=lambda r: r['ratio'])['mult'],
                         'any_notice': any(r['floor_notice'] for r in rows)}
    res['budget_sweep'] = sweep

    # --- 4. the legacy overrun at the shipped default budget --------------
    over = {}
    for N in (256, 512):
        for mode in ('legacy', 'measured'):
            _f, peak, _w = run(mode, b, N, 512.0)
            over['%s/N%d/512MB' % (mode, N)] = {
                'peak_bytes': peak, 'peak_over_budget': peak / 512.0e6}
    res['at_the_default_budget'] = over

    # --- 5. what the flip costs in bytes ----------------------------------
    spread = {}
    for N in (256, 512):
        a, _p, _w = run('legacy', b, N, 512.0)
        c, _p, _w = run('measured', b, N, 512.0)
        sc = float(np.max(np.abs(a)))
        spread[str(N)] = {
            'rel_max_abs_diff': float(np.max(np.abs(a - c))) / sc,
            'identical': bool(np.array_equal(a, c))}
        digests['legacy/N%d/512MB' % N] = dig(a)
        digests['measured/N%d/512MB' % N] = dig(c)
    res['legacy_vs_measured'] = spread

    # --- 6. determinism: the same budget twice ----------------------------
    det = {}
    for mode in ('legacy', 'measured'):
        a, _p, _w = run(mode, b, 256, 37.0)
        c, _p, _w = run(mode, b, 256, 37.0)
        det[mode] = bool(np.array_equal(a, c))
        digests['repeat/%s/N256/37MB' % mode] = dig(a)
    res['repeat_run_identical'] = det

    # --- 7. a fixed digest set for the archive-to-archive reading ---------
    for N in (64, 128, 256):
        for mb in (512.0, 64.0, 8.0):
            for mode in ('legacy', 'measured'):
                f, _p, _w = run(mode, b, N, mb)
                digests['%s/N%d/%gMB' % (mode, N, mb)] = dig(f)
    # the windowed sibling must not move at all
    for N in (128, 256):
        f = G.reconstruct_field_from_beamlets(
            b, Ny=N, Nx=N, dx=DX, wavelength=WL, window=5.0,
            mem_budget_mb=512.0)
        digests['windowed/N%d' % N] = dig(f)

    # --- 8. the vocabulary gate -------------------------------------------
    try:
        run('not-a-mode', b, 64, 512.0)
        res['unknown_mode'] = 'accepted'
    except Exception as exc:                               # noqa: BLE001
        res['unknown_mode'] = '%s: %s' % (type(exc).__name__, str(exc)[:300])

    res['digests'] = digests
    write(out, res)


if __name__ == '__main__':
    main(sys.argv[1])

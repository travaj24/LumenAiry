"""VERIFY-WP-C5 item 2 -- the dense GBD memory accounting, RE-MEASURED.

Independent of ``validation/probe_c5_three_defaults/c5_item2_budget.py``:
different grids (320 / 512 rather than 256 / 512), a different bundle, and
three things that probe does not do --

  * every peak is read BOTH with ``tracemalloc`` (the Python allocator's own
    high-water mark) AND as a SAMPLED process-RSS peak (``psutil``), so the floor's
    meaning is stated for both and "the budget bounds the loop" is not left
    as a statement about one instrument;
  * the ``N = 1706 / 1707`` claim behind "warn, not refuse" is computed from
    the shipped 512 MB default AND a default-path call is run at ``N = 2048``
    on BOTH trees, to see whether it completes, whether it warns exactly once
    and whether the warning names a remedy;
  * the three mutations (the floor helper missing its fixed term, the notice
    suppressed, the mode falling through to ``'legacy'``) are applied to the
    running module and the readings they move are recorded, so the decision
    tests can be shown to be the thing that catches them.

Usage (BLAS and the env budget pinned on the COMMAND LINE)::

    PYTHONPATH=<tree> LUMENAIRY_MEM_BUDGET_MB=2048 \\
        python validation/probe_verify_c5/v_item2_budget.py OUT.json
"""
import gc
import os
import sys
import threading
import tracemalloc
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                            # noqa: E402
import psutil                                                 # noqa: E402

from _vd import dig, write                                    # noqa: E402

from lumenairy.propagators import gbd as G                    # noqa: E402

DX, WL = 2.5e-6, 1.064e-6
GRIDS = (320, 512)
NB = 384                    # beamlets for the fits and the sweep
PROC = psutil.Process()


def bundle(n=NB, seed=17):
    rng = np.random.default_rng(seed)
    return G.BeamletBundle(
        positions=rng.normal(0.0, 1.5e-4, size=(n, 3)),
        directions=np.zeros((n, 3)),
        Q=np.full(n, 1.0 / (1.2e-3 - 0.03j), dtype=np.complex128),
        amplitude=(rng.normal(size=n)
                   + 1j * rng.normal(size=n)).astype(np.complex128),
        waist0=np.full(n, 1.2e-3))


class RssWatch:
    """A sampling high-water mark on the process's RESIDENT SET.

    A before/after RSS difference is not a peak: the transient the chunk
    arithmetic is about is freed before the call returns, and the allocator
    keeps the arena, so the difference reads near zero however big the
    transient was.  This samples in a thread instead, so "peak" means the same
    thing for RSS as it does for ``tracemalloc``.  It is a PROCESS-wide
    reading, so anything else running in this interpreter is inside it.
    """

    def __init__(self, interval=0.002):
        self.interval = float(interval)
        self.peak = 0
        self.base = 0
        self._stop = threading.Event()
        self._th = None

    def __enter__(self):
        gc.collect()
        self.base = PROC.memory_info().rss
        self.peak = self.base
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                r = PROC.memory_info().rss
            except Exception:                            # pragma: no cover
                return
            if r > self.peak:
                self.peak = r
            self._stop.wait(self.interval)

    def __exit__(self, *a):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)
        return False

    @property
    def delta(self):
        return int(self.peak - self.base)


def run(mode, b, N, budget_mb, chunk=4096, trace=True, dx=DX):
    """One dense reconstruction.  Returns
    ``(field, tracemalloc_peak_bytes, rss_peak_delta_bytes, [warnings])``.
    """
    old = G.DENSE_MEM_BUDGET_ACCOUNTING
    if mode is not None:
        G.DENSE_MEM_BUDGET_ACCOUNTING = mode
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            peak = base = 0
            with RssWatch() as rw:
                if trace:
                    tracemalloc.start()
                try:
                    if trace:
                        tracemalloc.reset_peak()
                        base = tracemalloc.get_traced_memory()[0]
                    out = G.reconstruct_field_from_beamlets(
                        b, Ny=N, Nx=N, dx=dx, wavelength=WL,
                        chunk_beamlets=chunk, mem_budget_mb=budget_mb)
                    if trace:
                        peak = tracemalloc.get_traced_memory()[1]
                finally:
                    if trace:
                        tracemalloc.stop()
            rss = rw.delta
        msgs = [str(w.message) for w in rec]
    finally:
        G.DENSE_MEM_BUDGET_ACCOUNTING = old
    return (np.asarray(out), int(peak - base), int(rss), msgs)


def fit_terms(N, b, chunks=(1, 2, 4, 8, 16, 32)):
    """``peak/(Ny*Nx) = fixed + c*chunk`` by least squares, at a budget far too
    large to bind, with ONE warm-up run discarded (the first tracemalloc window
    in a process also catches the import graph's lazy first-use allocations)."""
    run('legacy', b, N, 1.0e7, chunk=int(chunks[0]))
    rows, rss_rows = [], []
    for ch in chunks:
        _f, peak, rss, _w = run('legacy', b, N, 1.0e7, chunk=int(ch))
        rows.append((int(ch), peak / float(N * N)))
        rss_rows.append((int(ch), rss / float(N * N)))
    A = np.array([[1.0, float(ch)] for ch, _ in rows])
    y = np.array([v for _, v in rows])
    (fixed, c), _res, _rk, _sv = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([fixed, c])
    return dict(chunks=[ch for ch, _ in rows],
                bytes_per_cell=[float(v) for _, v in rows],
                rss_bytes_per_cell=[float(v) for _, v in rss_rows],
                fixed_b_per_cell=float(fixed), per_col_b_per_cell=float(c),
                worst_rel_dev=float(np.max(np.abs(pred - y) / y)))


def main(out_path):
    res = {'digests': {}}
    D = res['digests']
    warnings.simplefilter('ignore')
    has_floor = hasattr(G, '_dense_budget_floor_bytes')
    res['shipped'] = dict(
        accounting=G.DENSE_MEM_BUDGET_ACCOUNTING,
        legacy_b=G._DENSE_CELL_BYTES_LEGACY,
        measured_b=G._DENSE_CELL_BYTES_MEASURED,
        fixed_b=getattr(G, '_DENSE_FIXED_CELL_BYTES', None),
        has_floor_helper=has_floor,
        has_cell_bytes_helper=hasattr(G, '_dense_cell_bytes'),
        accountings=sorted(getattr(G, '_DENSE_MEM_BUDGET_ACCOUNTINGS', [])),
        default_budget_mb=G.reconstruct_field_from_beamlets
        .__defaults__ if False else 512.0)
    b = bundle()

    def floor_b(N):
        if has_floor:
            return float(G._dense_budget_floor_bytes(N, N))
        return float(N) * float(N) * (48.0 + G._DENSE_CELL_BYTES_MEASURED)

    # -- 1. the two constants, fitted -------------------------------------
    res['fits'] = {str(N): fit_terms(N, b) for N in GRIDS}

    # -- 2. the floor against the measured ONE-COLUMN peak -----------------
    rows = []
    for N in GRIDS:
        _f, pk, rss, _w = run('legacy', b, N, 1.0e7, chunk=1)
        rows.append(dict(N=N, published_floor_b=floor_b(N),
                         one_column_peak_b=pk, one_column_rss_b=rss,
                         published_over_trace=floor_b(N) / max(1, pk),
                         published_over_rss=(floor_b(N) / rss) if rss > 0
                         else None))
    res['floor_vs_one_column'] = rows

    # -- 3. two-sided sweep, ABOVE the floor -------------------------------
    sweep = []
    for N in GRIDS:
        fl = floor_b(N)
        for mult in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0):
            bud_mb = fl * mult / 1e6
            _f, pk, rss, msgs = run('measured', b, N, bud_mb, chunk=4096)
            sweep.append(dict(N=N, mult=mult, budget_mb=bud_mb,
                              peak_b=pk, rss_b=rss,
                              peak_over_budget=pk / (bud_mb * 1e6),
                              rss_over_budget=rss / (bud_mb * 1e6),
                              n_notices=sum('CANNOT meet' in m for m in msgs)))
    res['sweep_above_floor'] = sweep

    # -- 4. BELOW the floor: loud, accurate, and 'legacy' silent ----------
    below = []
    for N in GRIDS:
        fl = floor_b(N)
        bud_mb = 0.5 * fl / 1e6
        for mode in ('measured', 'legacy'):
            _f, pk, rss, msgs = run(mode, b, N, bud_mb, chunk=4096)
            notes = [m for m in msgs if 'CANNOT meet' in m]
            below.append(dict(
                N=N, mode=mode, budget_mb=bud_mb, peak_b=pk, rss_b=rss,
                peak_over_budget=pk / (bud_mb * 1e6),
                rss_over_budget=rss / (bud_mb * 1e6),
                n_notices=len(notes),
                names_floor=bool(notes and ('%.6g' % (fl / 1e6)) in notes[0]),
                names_window=bool(notes and 'window=5.0' in notes[0]),
                names_raise=bool(notes and 'raise mem_budget_mb' in notes[0]),
                names_helper=bool(notes and '_dense_budget_floor_bytes'
                                  in notes[0]),
                notice=(notes[0][:600] if notes else None)))
    res['below_floor'] = below

    # -- 4b. THE HEADLINE OVERRUN, on 1024 beamlets at the shipped 512 MB --
    # The claim the item exists for: under 'legacy' the loop's live peak is
    # about six times the budget the caller asked for, and under 'measured'
    # it is under it.  Measured on a bundle big enough for the chunk to bind.
    big = bundle(n=1024, seed=3)
    head = []
    for N in GRIDS:
        for mode in ('legacy', 'measured'):
            _f, pk, rss, _w = run(mode, big, N, 512.0, chunk=4096)
            head.append(dict(N=N, mode=mode, budget_mb=512.0, peak_b=pk,
                             rss_b=rss, peak_over_budget=pk / 512.0e6,
                             rss_over_budget=rss / 512.0e6,
                             chunk=max(1, int(512.0e6 / max(
                                 1.0, N * N * (16.0 if mode == 'legacy'
                                               else 128.0))))))
    res['headline_512mb_1024beamlets'] = head
    del big
    gc.collect()

    # -- 5. what the flip costs in BYTES ----------------------------------
    costs = []
    for N in GRIDS:
        f_leg, _p, _r, _w = run('legacy', b, N, 512.0, chunk=4096)
        f_mea, _p, _r, _w = run('measured', b, N, 512.0, chunk=4096)
        f_leg2, _p, _r, _w = run('legacy', b, N, 512.0, chunk=4096)
        f_mea2, _p, _r, _w = run('measured', b, N, 512.0, chunk=4096)
        den = np.max(np.abs(f_leg))
        costs.append(dict(
            N=N,
            max_abs_rel_diff=float(np.max(np.abs(f_mea - f_leg)) / den),
            bitwise_identical=bool(dig(f_mea) == dig(f_leg)),
            legacy_repeatable=bool(dig(f_leg) == dig(f_leg2)),
            measured_repeatable=bool(dig(f_mea) == dig(f_mea2))))
        D['legacy/%d/512MB' % N] = dig(f_leg)
        D['measured/%d/512MB' % N] = dig(f_mea)
        for bud in (64.0, 8.0):
            D['legacy/%d/%gMB' % (N, bud)] = dig(
                run('legacy', b, N, bud, chunk=4096)[0])
            D['measured/%d/%gMB' % (N, bud)] = dig(
                run('measured', b, N, bud, chunk=4096)[0])
        D['default/%d/512MB' % N] = dig(run(None, b, N, 512.0, chunk=4096)[0])
        D['default/%d/8MB' % N] = dig(run(None, b, N, 8.0, chunk=4096)[0])
        # the windowed sibling, which this item does not touch
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            D['windowed/%d/8MB' % N] = dig(G.reconstruct_field_from_beamlets(
                b, Ny=N, Nx=N, dx=DX, wavelength=WL, chunk_beamlets=4096,
                window=5.0, mem_budget_mb=8.0))
    res['flip_cost'] = costs

    # -- 6. the 512 MB default's own binding grid --------------------------
    per_cell = 48.0 + G._DENSE_CELL_BYTES_MEASURED
    n_star = float(np.sqrt(512.0e6 / per_cell))
    res['default_budget_binding_grid'] = dict(
        per_cell_bytes=per_cell, cells=512.0e6 / per_cell, N_star=n_star,
        floor_at_1706_mb=floor_b(1706) / 1e6,
        floor_at_1707_mb=floor_b(1707) / 1e6,
        binds_at_1706=bool(floor_b(1706) > 512.0e6),
        binds_at_1707=bool(floor_b(1707) > 512.0e6))

    # -- 7. a DEFAULT call at N = 2048 -------------------------------------
    # The grid pitch is scaled so the bundle's own profile still DECAYS across
    # 2048 samples (at DX the Gaussian argument overflows at the grid edge and
    # the arm would be measuring a fixture defect, not the budget).
    b_small = bundle(n=48, seed=5)
    DX_BIG = 5.0e-7
    res['default_call_N2048'] = {}
    for tag, mode in (('default', None), ('legacy', 'legacy')):
        old_acc = G.DENSE_MEM_BUDGET_ACCOUNTING
        if mode is not None:
            G.DENSE_MEM_BUDGET_ACCOUNTING = mode
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter('always')
                np.seterr(all='warn')
                with RssWatch() as rw:
                    E2048 = G.reconstruct_field_from_beamlets(
                        b_small, Ny=2048, Nx=2048, dx=DX_BIG, wavelength=WL)
            msgs = [str(w.message) for w in rec]
        finally:
            G.DENSE_MEM_BUDGET_ACCOUNTING = old_acc
        notes = [m for m in msgs if 'CANNOT meet' in m]
        res['default_call_N2048'][tag] = dict(
            completed=bool(np.isfinite(E2048).all()),
            shape=list(np.shape(E2048)), digest=dig(E2048),
            rss_peak_delta_b=rw.delta,
            n_warnings_total=len(msgs), n_floor_notices=len(notes),
            floor_mb=floor_b(2048) / 1e6,
            names_window=bool(notes and 'window=5.0' in notes[0]),
            names_raise=bool(notes and 'raise mem_budget_mb' in notes[0]),
            names_floor=bool(notes and '738.198' in notes[0]),
            notice=(notes[0][:800] if notes else None),
            other_warnings=sorted({m[:70] for m in msgs
                                   if 'CANNOT meet' not in m}))
        del E2048
        gc.collect()

    # -- 8. the unknown-mode refusal ---------------------------------------
    ref = {}
    for bad in ('Measured', 'legacy ', 'MEASURED', '', None, 16, 'honest'):
        try:
            if hasattr(G, '_dense_cell_bytes'):
                G._dense_cell_bytes(bad)
            else:
                raise AssertionError('no helper on this tree')
            ref[repr(bad)] = 'ACCEPTED'
        except ValueError as exc:
            ref[repr(bad)] = 'ValueError: ' + str(exc)[:140]
        except AssertionError as exc:
            ref[repr(bad)] = str(exc)
    # and through the public entry point
    old = G.DENSE_MEM_BUDGET_ACCOUNTING
    try:
        G.DENSE_MEM_BUDGET_ACCOUNTING = 'honest'
        try:
            run(None, b, 64, 512.0, chunk=4096, trace=False)
            ref['public_entry_point'] = 'ACCEPTED'
        except ValueError as exc:
            ref['public_entry_point'] = 'ValueError: ' + str(exc)[:140]
    finally:
        G.DENSE_MEM_BUDGET_ACCOUNTING = old
    res['unknown_mode'] = ref

    # -- 9. the three mutations --------------------------------------------
    mut = {}
    if has_floor:
        N = GRIDS[0]
        fl = floor_b(N)
        real = G._dense_budget_floor_bytes
        try:
            # (a) the floor helper loses its fixed term
            G._dense_budget_floor_bytes = (
                lambda Ny, Nx: float(Ny) * float(Nx)
                * G._DENSE_CELL_BYTES_MEASURED)
            bad_fl = float(G._dense_budget_floor_bytes(N, N))
            _f, pk, _r, msgs = run('measured', b, N, bad_fl / 1e6,
                                   chunk=4096)
            mut['floor_missing_fixed'] = dict(
                mutant_floor_b=bad_fl, true_floor_b=fl,
                ratio=bad_fl / fl,
                peak_over_mutant_floor=pk / bad_fl,
                notice_fired=sum('CANNOT meet' in m for m in msgs))
        finally:
            G._dense_budget_floor_bytes = real
    # (b) the notice suppressed  -- emulate by catching and dropping it
    N = GRIDS[0]
    fl = floor_b(N)
    _f, pk, _r, msgs = run('measured', b, N, 0.5 * fl / 1e6, chunk=4096)
    mut['notice_present_at_half_floor'] = dict(
        n=sum('CANNOT meet' in m for m in msgs),
        peak_over_budget=pk / (0.5 * fl))
    # (c) the mode silently 'legacy'
    _f, pk_m, _r, _w = run('measured', b, N, 8.0, chunk=4096)
    _f, pk_l, _r, _w = run('legacy', b, N, 8.0, chunk=4096)
    mut['silent_legacy'] = dict(
        measured_peak_over_budget=pk_m / 8.0e6,
        legacy_peak_over_budget=pk_l / 8.0e6,
        ratio=pk_l / max(1, pk_m))
    res['mutations'] = mut

    write(out_path, res)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1
         else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'v_item2_head_win.json'))

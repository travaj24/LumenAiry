"""H2-1 (item 14) -- the direct-matrix MFT branch: crossover and tolerance.

Run as a CHILD process bound to ONE tree:

    python probe_mft_direct.py <tree> <out.json> [--quick]

Produces a measurement JSON (NOT a bit-identity map -- ``probe_mft_bitid.py``
carries that) with four sections:

``shapes``
    For every ``(N, M)`` in ``{64,128,256,512,1024} x {32,64,128,256,512}`` and
    every route in ``{bluestein, separable, direct}``: best-of-``reps`` wall
    time and ``tracemalloc`` peak in TWO cache regimes (``cold``, every library
    cache dropped before each repeat; ``warm``, nothing cleared), plus the
    ANALYTIC byte count derived from the source below.  Both regimes are
    published because they differ by an order of magnitude for the chirp-Z
    route and not at all for the dense one -- see :func:`measure_shapes`.

``tolerance``
    The two chirp-Z reductions and the dense route against a pairwise-summed
    float64 reference on a small case, and against each other, with the
    summation condition number the bar is derived from.

``phase_budget``
    The one regime where the dense route is not merely an alternative: a chirp
    phase argument past float64's reach, where ``_bluestein_2d`` warns and
    loses digits and the dense route -- which reduces modulo one turn -- does
    not.

``env``
    Build, versions, FFT backend switches.

THE ANALYTIC BYTE COUNTS, derived by reading ``_bluestein.py``:

* ``bluestein`` (2-D convolution).  ``L{x,y} = next_fast_len(N + M - 1)``.
  Live at the peak (step 5 of the source): ``g_pad``, ``G_FFT``, ``h_2d``,
  ``H_FFT``, the product ``G_FFT*H_FFT`` and ``ifft2``'s output --
  six ``(Ly, Lx)`` complex arrays, plus the pre-chirped ``g`` at
  ``(Ny, Nx)``.  So ``16*(6*Ly*Lx + Ny*Nx)`` bytes for complex128.
* ``separable`` (two 1-D passes).  The x pass holds ``g`` and ``G`` at
  ``(Ny, Lx)`` plus the transform output at ``(Ny, Lx)``; the y pass the same
  at ``(Ly, M)``.  So ``16*3*max(Ny*Lx, Ly*M)`` bytes.
* ``direct`` (two matrix products).  ``Wx`` at ``(M, N)``, ``Wy`` at
  ``(M, N)``, the float64 phase scratch ``t`` at ``8*M*N`` while each is
  built, the intermediate at ``(M, N)`` or ``(N, M)`` and the output at
  ``(M, M)``.  So ``16*(2*Mx*Nx_+ My*Ny_ ... )`` -- written out in
  :func:`_analytic_bytes`.

The counts are MODELS of the source, not fits to the measurement; the report
prints both and the ratio, so where the model and ``tracemalloc`` disagree the
disagreement is visible rather than hidden.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import tracemalloc
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import hlib  # noqa: E402

import numpy as np  # noqa: E402
from scipy.fft import next_fast_len  # noqa: E402

hlib.anchor(_TREE)

from lumenairy.propagators._bluestein import (  # noqa: E402
    _bluestein_2d, _clear_h_fft_cache)
from lumenairy.propagators.fft_infra import _fft2, _ifft2  # noqa: E402

QUICK = '--quick' in sys.argv
EXTEND = '--extend' in sys.argv
N_LADDER = (64, 128, 256, 512, 1024)
M_LADDER = (32, 64, 128, 256, 512)
if QUICK:
    N_LADDER = (64, 256)
    M_LADDER = (32, 128)

#: Shapes BEYOND the requested box, run only with ``--extend``.  The requested
#: ladder tops out at ``(N, M) = (1024, 512)`` and the dense route still wins
#: there on both axes, so the crossover -- the thing the item asks to be
#: measured -- is not inside the box.  These four shapes bracket it.  They are
#: a separate list, and a separate JSON section, so the requested box is
#: reported exactly as asked and the extension is visibly an extension.
EXT_SHAPES = ((1024, 1024), (2048, 512), (2048, 1024), (1448, 1448))

#: ``alpha`` is the sampling-rate parameter the MFT propagators form as
#: ``dx_in*dx_out/(lambda*z)``.  Pinned here (not derived from a lens) so the
#: ladder measures the TRANSFORM and nothing else; the value is the natural
#: Fresnel one for a 512-point input at unit zoom, ``1/512``.
ALPHA = 1.0 / 512.0
#: Best-of-N.  The box this was measured on runs other heavy python work
#: concurrently (44 interpreters alive during the first ladder), so a single
#: timing is a sample of the CONTENTION as much as of the routine.  Five
#: repeats, the minimum reported, and every repeat kept in
#: ``seconds_cold_all`` / ``seconds_warm_all`` so the spread is visible in the
#: JSON rather than hidden behind the minimum.
REPS = 1 if QUICK else 5


def _field(Ny, Nx, seed=20260915):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((Ny, Nx))
            + 1j * rng.standard_normal((Ny, Nx))).astype(np.complex128)


def _call(E, N_out, method, alpha=ALPHA, sign=-1):
    if method == 'bluestein':
        kw = dict(separable=False, method='auto')
    elif method == 'separable':
        kw = dict(separable=True, method='auto')
    elif method == 'direct':
        kw = dict(separable=False, method='direct')
    else:
        raise SystemExit(method)
    return _bluestein_2d(E, alpha, alpha, N_out, N_out,
                         sign=sign, xp=np, fft2=_fft2, ifft2=_ifft2, **kw)


def _analytic_bytes(N, M, method, itemsize=16):
    """Bytes the route's own source says it holds live at its peak."""
    if method == 'bluestein':
        L = int(next_fast_len(int(N + M - 1)))
        return itemsize * (6 * L * L + N * N)
    if method == 'separable':
        Lx = int(next_fast_len(int(N + M - 1)))
        Ly = Lx
        return itemsize * 3 * max(N * Lx, Ly * M)
    if method == 'direct':
        # Wx (M,N) + Wy (M,N) complex, the float64 phase scratch 8*M*N live
        # while the second is built, the intermediate and the output.
        cost_y_first = M * N * N + M * N * M
        cost_x_first = N * N * M + M * N * M
        inter = M * N if cost_y_first <= cost_x_first else N * M
        return itemsize * (2 * M * N + inter + M * M) + 8 * M * N
    raise SystemExit(method)


def _clear_caches():
    _clear_h_fft_cache()
    try:
        from lumenairy._cache_registry import clear_all_registered_caches
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clear_all_registered_caches()
    except Exception as exc:                       # noqa: BLE001 -- recorded
        print(f"[warn] cache clear: {exc}", file=sys.stderr)
    gc.collect()


def measure_shapes():
    """The ladder, in TWO cache regimes, because the two answer different
    questions and the chirp-Z route's answer differs by an order of magnitude
    between them.

    ``cold`` -- every library cache dropped before the call (the chirp-kernel
    FFT cache AND, through the registry, the pyFFTW plan / buffer cache).  This
    is the FIRST call at a shape, and it is the production regime for the MFT
    propagators as ``_bluestein.py``'s own byte-cap comment records it: alpha
    carries ``dx_out/(N_in*dx_in)``, which moves per congruence, so a
    production order measured ``hits = 0`` across the whole fan.  The dense
    route has no plan and no kernel cache, so cold and warm are the same call
    for it -- which is itself part of the answer.

    ``warm`` -- nothing cleared, after a warm-up call at the same shape.  This
    is a repeated readout on ONE geometry, where the chirp kernel and the FFT
    plan are both hits.

    ``tracemalloc`` peak is recorded for both: the cold peak includes whatever
    persistent buffers the first call allocates, the warm peak is the marginal
    cost of one more call.
    """
    rows = []
    for N in N_LADDER:
        E = _field(N, N)
        for M in M_LADDER:
            ref = None
            for method in ('bluestein', 'separable', 'direct'):
                _clear_caches()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        _call(E, M, method)
                except Exception as exc:           # noqa: BLE001 -- recorded
                    rows.append({'N': N, 'M': M, 'method': method,
                                 'error': f'{type(exc).__name__}: {exc}'})
                    continue
                row = {'N': N, 'M': M, 'method': method,
                       'analytic_bytes': int(_analytic_bytes(N, M, method))}
                # PASS 1 -- wall time, with tracemalloc OFF.  tracemalloc
                # charges per ALLOCATION, and the chirp-Z route allocates far
                # more objects than two matrix products do, so timing inside it
                # would bias the very comparison this table is for.
                for regime in ('cold', 'warm'):
                    best = float('inf')
                    if regime == 'warm':
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            _call(E, M, method)
                    every = []
                    for _ in range(REPS):
                        if regime == 'cold':
                            _clear_caches()
                        t0 = time.perf_counter()
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            F = _call(E, M, method)
                        dt = time.perf_counter() - t0
                        best = min(best, dt)
                        every.append(dt)
                        del F
                    row[f'seconds_{regime}'] = best
                    row[f'seconds_{regime}_all'] = every
                # PASS 2 -- tracemalloc peak, untimed.
                for regime in ('cold', 'warm'):
                    peak = 0
                    if regime == 'warm':
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            _call(E, M, method)
                    for _ in range(1 if regime == 'warm' else REPS):
                        if regime == 'cold':
                            _clear_caches()
                        tracemalloc.start()
                        tracemalloc.reset_peak()
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            F = _call(E, M, method)
                        _cur, pk = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        peak = max(peak, pk)
                        del F
                    row[f'tracemalloc_peak_bytes_{regime}'] = int(peak)
                _clear_caches()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    F = _call(E, M, method)
                if ref is None:
                    ref = F
                    row['rel_L2_vs_bluestein'] = 0.0
                    row['max_abs_rel_vs_bluestein'] = 0.0
                else:
                    d = F - ref
                    row['rel_L2_vs_bluestein'] = float(
                        np.linalg.norm(d) / np.linalg.norm(ref))
                    row['max_abs_rel_vs_bluestein'] = float(
                        np.max(np.abs(d)) / np.max(np.abs(ref)))
                    del F
                rows.append(row)
            del ref
        del E
    return rows


# ---------------------------------------------------------------------------
# Tolerance: the pairwise-summed float64 reference
# ---------------------------------------------------------------------------

def _pairwise_reference(E, alpha, M, sign):
    """The SAME sum, evaluated one output point at a time with NumPy's
    pairwise summation over the whole ``N*N`` product.

    ``np.sum`` on a contiguous float / complex array uses PAIRWISE summation
    with an unrolled block of 128, whose error bound is
    ``eps * (log2(n/128) + 8) * sum|x|`` -- against ``eps*n*sum|x|`` for a
    naive loop and ``eps*sqrt(n)`` typical for a BLAS ``matmul``'s blocked
    order.  No mpmath and no float128 is used or needed: the bound below is
    what makes it a reference, not extra digits.

    The phase is built with the same modulo-one-turn reduction the shipped
    dense route uses, so the reference and the routes differ in the SUMMATION
    only -- which is the quantity the bar is derived from.
    """
    Ny, Nx = E.shape
    n_x = np.arange(Nx, dtype=np.float64)
    n_y = np.arange(Ny, dtype=np.float64)
    out = np.empty((M, M), dtype=np.complex128)
    for ky in range(M):
        ty = alpha * float(ky) * n_y
        ty = ty - np.rint(ty)
        wy = np.exp(1j * sign * 2.0 * np.pi * ty)
        for kx in range(M):
            tx = alpha * float(kx) * n_x
            tx = tx - np.rint(tx)
            wx = np.exp(1j * sign * 2.0 * np.pi * tx)
            out[ky, kx] = np.sum(E * (wy[:, None] * wx[None, :]))
    return out


def measure_tolerance():
    """Small case, every route against the pairwise reference and each other.

    The DERIVED BAR.  Each output point is a sum of ``n = Ny*Nx`` terms whose
    magnitudes are ``|E[ny,nx]|`` (the kernel has unit modulus).  Its
    summation condition number is ``kappa = sum|E| / |F|``; a summation whose
    growth factor is ``g`` commits at most ``g * eps * kappa`` relative error.
    So the bar for any two routes against each other is
    ``(g_a + g_b) * eps * kappa``, and the report states it in exactly that
    form.  ``g`` is bounded above by: ``log2(n/128)+8`` for the pairwise
    reference, ``~3*log2(L^2)`` for the chirp-Z route (three FFTs of length
    ``L^2``, each ``log2`` deep), and ``N`` worst-case / ``sqrt(N)`` typical
    for the dense route's two BLAS products.
    """
    out = {}
    eps = float(np.finfo(np.float64).eps)
    for (N, M) in ((16, 8), (32, 16), (48, 24)):
        E = _field(N, N, seed=4242 + N)
        alpha = 1.0 / 64.0
        ref = _pairwise_reference(E, alpha, M, -1)
        sum_abs = float(np.sum(np.abs(E)))
        # per-point condition number, and the worst one over the block
        kappa = sum_abs / np.abs(ref)
        kappa_max = float(np.max(kappa))
        kappa_med = float(np.median(kappa))
        n = N * N
        L = int(next_fast_len(int(N + M - 1)))
        g_pair = float(np.log2(max(n / 128.0, 2.0)) + 8.0)
        g_chirp = float(3.0 * np.log2(L * L))
        g_dense = float(np.sqrt(n))
        row = {'N': N, 'M': M, 'alpha': alpha, 'n_terms': n, 'L': L,
               'sum_abs_E': sum_abs,
               'kappa_max': kappa_max, 'kappa_median': kappa_med,
               'g_pairwise': g_pair, 'g_chirp': g_chirp, 'g_dense': g_dense,
               'eps': eps}
        got = {}
        for method in ('bluestein', 'separable', 'direct'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                F = _call(E, M, method, alpha=alpha)
            got[method] = F
            d = F - ref
            row[f'{method}_rel_L2_vs_ref'] = float(
                np.linalg.norm(d) / np.linalg.norm(ref))
            row[f'{method}_max_abs_vs_ref'] = float(np.max(np.abs(d)))
            row[f'{method}_max_abs_rel_vs_ref'] = float(
                np.max(np.abs(d)) / np.max(np.abs(ref)))
            # the two-sided derived bar, per route, in ABSOLUTE terms
            g = {'bluestein': g_chirp, 'separable': g_chirp,
                 'direct': g_dense}[method]
            row[f'{method}_bar_abs'] = float((g + g_pair) * eps * sum_abs)
        for a, b in (('bluestein', 'separable'), ('bluestein', 'direct'),
                     ('separable', 'direct')):
            d = got[a] - got[b]
            row[f'{a}_vs_{b}_rel_L2'] = float(
                np.linalg.norm(d) / np.linalg.norm(ref))
            row[f'{a}_vs_{b}_max_abs'] = float(np.max(np.abs(d)))
            ga = g_chirp if a != 'direct' else g_dense
            gb = g_chirp if b != 'direct' else g_dense
            row[f'{a}_vs_{b}_bar_abs'] = float((ga + gb) * eps * sum_abs)
        out[f'N{N}_M{M}'] = row
        del E, ref, got
    return out


# ---------------------------------------------------------------------------
# The phase-budget regime
# ---------------------------------------------------------------------------

def measure_phase_budget():
    """``_bluestein_2d``'s own guard warns above ``alpha*N_max^2 > 1e15``.

    That guard is a statement about the CHIRP signals -- ``exp(i*pi*alpha*n^2)``
    with ``n`` up to ``N`` -- and its advice is "fall back to a regular FFT
    propagator".  The dense route reduces modulo one turn, so it has no such
    phase.  Measured here as a DECISION (does the route warn, and does its
    answer still track the pairwise reference), not as a pinned residual.
    """
    out = {}
    N, M = 24, 12
    E = _field(N, N, seed=77)
    for label, alpha in (('nominal', 1.0 / 64.0),
                         ('budget_1e12', 1e12 / float(N) ** 2),
                         ('budget_1e15', 1e15 / float(N) ** 2),
                         ('budget_1e17', 1e17 / float(N) ** 2)):
        ref = _pairwise_reference(E, alpha, M, -1)
        row = {'alpha': alpha, 'phase_budget': alpha * float(N) ** 2}
        for method in ('bluestein', 'direct'):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                F = _call(E, M, method, alpha=alpha)
            d = F - ref
            row[f'{method}_rel_L2_vs_ref'] = float(
                np.linalg.norm(d) / np.linalg.norm(ref))
            row[f'{method}_warned'] = [str(w.message)[:80] for w in caught]
        out[label] = row
    return out


def measure_extension():
    """The crossover itself, outside the requested box.

    One repeat, cold only, time and peak: at these shapes the dense route costs
    tens of seconds and the point is WHERE the two curves cross, not a
    best-of-three timing.
    """
    rows = []
    for (N, M) in EXT_SHAPES:
        E = _field(N, N)
        for method in ('bluestein', 'separable', 'direct'):
            _clear_caches()
            row = {'N': N, 'M': M, 'method': method,
                   'analytic_bytes': int(_analytic_bytes(N, M, method))}
            try:
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    F = _call(E, M, method)
                row['seconds_cold'] = time.perf_counter() - t0
                del F
                _clear_caches()
                tracemalloc.start()
                tracemalloc.reset_peak()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    F = _call(E, M, method)
                _cur, pk = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                row['tracemalloc_peak_bytes_cold'] = int(pk)
                del F
            except Exception as exc:               # noqa: BLE001 -- recorded
                row['error'] = f'{type(exc).__name__}: {exc}'
            rows.append(row)
            print(f"[ext] {row}", file=sys.stderr)
        del E
    return rows


def main():
    out_path = sys.argv[2]
    import scipy
    from lumenairy.propagators import fft_infra as fi
    import lumenairy
    env = {
        'build': hlib.build_tag(),
        'python': sys.version.split()[0],
        'platform': sys.platform,
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'lumenairy': lumenairy.__version__,
        'lumenairy_file': os.path.realpath(lumenairy.__file__),
        'tree': _TREE,
        'USE_PYFFTW': bool(fi.USE_PYFFTW),
        'PYFFTW_AVAILABLE': bool(fi.PYFFTW_AVAILABLE),
        'USE_SCIPY_FFT': bool(fi.USE_SCIPY_FFT),
        'SCIPY_FFT_WORKERS': getattr(fi, 'SCIPY_FFT_WORKERS', None),
        'FFTW_THREADS': getattr(fi, 'FFTW_THREADS', None),
        'get_fft_threads': fi.get_fft_threads(),
        'registered_cache_clearers': len(
            __import__('lumenairy._cache_registry', fromlist=['x'])
            .list_registered_cache_clearers()),
        'threads_env': {k: os.environ.get(k) for k in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'MKL_NUM_THREADS')},
        'quick': QUICK,
        'extend': EXTEND,
        'reps': REPS,
    }
    result = {
        'env': env,
        'tolerance': measure_tolerance(),
        'phase_budget': measure_phase_budget(),
        'shapes': measure_shapes(),
    }
    if EXTEND:
        result['extension'] = measure_extension()
    hlib.write_json(result, out_path)
    print(json.dumps({'n_shape_rows': len(result['shapes']),
                      'build': env['build']}))


if __name__ == '__main__':
    main()

"""Phase R3 (roadmap B3 + B7): GBD windowed-reconstruct memory + traced-fit
thread-safe least-squares.

B3 -- GBD decenter reconstruct memory
-------------------------------------
``_reconstruct_windowed`` (the NumPy windowed beamlet scatter that the GBD
decenter / per-surface-astigmatic path uses) accumulated each ``(n_g, by, bx)``
beamlet x window-box tile through two extra ``np.where`` copies and let the
previous chunk's big buffers coexist with the next chunk's -- the peak ran
~1.8x over ``mem_budget_mb`` (~0.94 GB at a 512 MB budget, ~1.6 GB in the full
pipeline), a 16 GB-runner OOM risk on large decentered GBD runs.

The B3 change accumulates in memory-budgeted tiles that honor
``LUMENAIRY_MEM_BUDGET_MB`` (mirroring the FGA chunk loop): out-of-grid cells
are masked IN PLACE instead of via two ``np.where`` copies, and each tile's big
buffers are freed before the next allocates.  With the env unset and the
default budget the chunk boundaries -- and hence the output -- are UNCHANGED, so
the result is byte-identical to the pre-B3 path (asserted here against a verbatim
copy of that path); the win is purely lower peak memory.  These tests measure the
peak-bytes reduction with ``tracemalloc`` and pin byte-identity.

B7 -- jax x OpenBLAS lstsq mitigation
-------------------------------------
The traced Chebyshev / coordinate fits used ``np.linalg.lstsq`` (LAPACK
``gelsd`` SVD), whose multi-threaded OpenBLAS OpenMP pool deadlocks against JAX's
OpenMP runtime in a shared process (the CI worked around it with
``OMP_NUM_THREADS=1`` pins).  ``_solve_lstsq_thread_safe`` replaces it with a
normal-equations Cholesky/LU solve on the tiny ``M x M`` Gram matrix, which never
takes the ``gelsd`` path so the deadlock cannot recur.  These tests pin the
solution to ``lstsq`` to tolerance and exercise the traced fit under
multi-thread BLAS in a fresh JAX-loaded subprocess.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import tracemalloc
import warnings
from typing import Any, Tuple

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements._lens_traced import _solve_lstsq_thread_safe
from lumenairy.glass import GLASS_REGISTRY
from lumenairy.memory import available_memory_bytes
from lumenairy.propagators.gbd import (
    BeamletBundle,
    _fft_reconstruct_applicable,
    _q_is_tensor,
    _reconstruct_windowed,
)

GLASS_REGISTRY['_R3A'] = lambda wl: 1.5168
_WL = 1.31e-6


# ===========================================================================
# Pre-B3 reference: the windowed reconstruct BEFORE the memory change, copied
# verbatim (two np.where copies, no cross-chunk free).  Serves both as the
# byte-identity oracle and as the "peak before" measurement.
# ===========================================================================
def _reconstruct_windowed_preB3(
    beamlets: BeamletBundle, *, Ny: int, Nx: int, dx: float, dy: float,
    centre: Tuple[float, float], wavelength: float,
    n_sigma: float = 5.0, mem_budget_mb: float = 512.0,
) -> np.ndarray:
    cx, cy = centre
    k = 2.0 * float(np.pi) / wavelength
    out_dtype = beamlets.amplitude.dtype
    x_b = np.asarray(beamlets.positions[:, 0], dtype=np.float64)
    y_b = np.asarray(beamlets.positions[:, 1], dtype=np.float64)
    a_b = np.asarray(beamlets.amplitude)
    is_tensor = _q_is_tensor(beamlets.Q)
    has_dirs = (getattr(beamlets, 'directions', None) is not None)
    if has_dirs:
        L_all = np.asarray(beamlets.directions[:, 0], dtype=np.float64)
        M_all = np.asarray(beamlets.directions[:, 1], dtype=np.float64)
    if is_tensor:
        Q_all = np.asarray(beamlets.Q)
        D = -np.imag(Q_all)
        a11 = D[:, 0, 0]
        a22 = D[:, 1, 1]
        a12 = D[:, 0, 1]
        tr = a11 + a22
        disc = np.sqrt(np.maximum((a11 - a22) ** 2 + 4.0 * a12 * a12, 0.0))
        lam_min = 0.5 * (tr - disc)
    else:
        Q_all = np.asarray(beamlets.Q)
        lam_min = -np.imag(Q_all)
    alpha_min = np.maximum(0.5 * k * lam_min, 1e-30)
    R_cut = n_sigma / np.sqrt(alpha_min)
    Wx_f = R_cut / dx
    Wy_f = R_cut / dy
    out_r = np.zeros(Ny * Nx + 1, dtype=np.float64)
    out_i = np.zeros(Ny * Nx + 1, dtype=np.float64)
    sentinel = Ny * Nx

    def _quantize(Wf, cap):
        W = np.ceil(Wf).astype(np.int64)
        W = np.clip(W, 1, cap)
        lg2 = np.ceil(2.0 * np.log2(np.maximum(W, 1))).astype(np.int64)
        Wq = np.ceil(np.power(2.0, 0.5 * lg2)).astype(np.int64)
        Wq = np.where(W <= 1, 1, Wq)
        return np.clip(Wq, 1, cap).astype(np.int64)

    Wx_q = _quantize(Wx_f, Nx)
    Wy_q = _quantize(Wy_f, Ny)
    ix0 = np.round((x_b - cx) / dx + Nx / 2.0).astype(np.int64)
    iy0 = np.round((y_b - cy) / dy + Ny / 2.0).astype(np.int64)
    skew = bool(is_tensor and np.any(Q_all[:, 0, 1] != 0.0))
    keys = Wy_q.astype(np.int64) * (Nx + 1) + Wx_q.astype(np.int64)
    for key in np.unique(keys):
        sel = np.nonzero(keys == key)[0]
        Wx = int(Wx_q[sel[0]])
        Wy = int(Wy_q[sel[0]])
        bx = 2 * Wx + 1
        by = 2 * Wy + 1
        cells = by * bx
        chunk = max(1, int(mem_budget_mb * 1e6 / max(1.0, cells * 32.0)))
        off_x = np.arange(-Wx, Wx + 1, dtype=np.int64)
        off_y = np.arange(-Wy, Wy + 1, dtype=np.int64)
        for cs in range(0, sel.size, chunk):
            g = sel[cs:cs + chunk]
            ixg = ix0[g][:, None] + off_x[None, :]
            iyg = iy0[g][:, None] + off_y[None, :]
            Xloc = (ixg - Nx / 2.0) * dx + cx
            Yloc = (iyg - Ny / 2.0) * dy + cy
            dX = Xloc - x_b[g][:, None]
            dY = Yloc - y_b[g][:, None]
            if skew:
                dX2 = dX[:, None, :] ** 2
                dY2 = dY[:, :, None] ** 2
                Qxx = np.conj(Q_all[g, 0, 0])[:, None, None]
                Qyy = np.conj(Q_all[g, 1, 1])[:, None, None]
                Qxy = np.conj(Q_all[g, 0, 1])[:, None, None]
                arg = 0.5 * (Qxx * dX2 + Qyy * dY2
                             + 2.0 * Qxy * (dX[:, None, :] * dY[:, :, None]))
                if has_dirs:
                    arg = (arg + L_all[g][:, None, None] * dX[:, None, :]
                           + M_all[g][:, None, None] * dY[:, :, None])
                contrib = a_b[g][:, None, None] * np.exp(1j * k * arg)
            else:
                if is_tensor:
                    cqx = np.conj(Q_all[g, 0, 0])
                    cqy = np.conj(Q_all[g, 1, 1])
                else:
                    cqx = np.conj(Q_all[g])
                    cqy = cqx
                argx = 0.5 * cqx[:, None] * (dX * dX)
                argy = 0.5 * cqy[:, None] * (dY * dY)
                if has_dirs:
                    argx = argx + L_all[g][:, None] * dX
                    argy = argy + M_all[g][:, None] * dY
                gx = np.exp(1j * k * argx)
                gy = np.exp(1j * k * argy)
                contrib = (a_b[g][:, None, None]
                           * gy[:, :, None]) * gx[:, None, :]
            idx = iyg[:, :, None] * Nx + ixg[:, None, :]
            valid = ((iyg[:, :, None] >= 0) & (iyg[:, :, None] < Ny)
                     & (ixg[:, None, :] >= 0) & (ixg[:, None, :] < Nx))
            idx = np.where(valid, idx, sentinel).ravel()
            cflat = np.where(valid, contrib, 0.0).ravel()
            out_r += np.bincount(idx, weights=cflat.real, minlength=Ny * Nx + 1)
            out_i += np.bincount(idx, weights=cflat.imag, minlength=Ny * Nx + 1)
    out = (out_r[:Ny * Nx] + 1j * out_i[:Ny * Nx]).reshape(Ny, Nx)
    return out.astype(out_dtype)


def _windowed_bundle(N, dx, n_side, w0, kind='scalar', seed=0):
    """A decentered-GBD-like beamlet bundle that forces the windowed path
    (non-uniform per-beamlet Q + per-beamlet tilt -> not FFT-applicable)."""
    rng = np.random.default_rng(seed)
    ax = np.linspace(-0.4 * N * dx, 0.4 * N * dx, n_side)
    X, Y = np.meshgrid(ax, ax)
    pos = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    n = pos.shape[0]
    dirs = np.zeros((n, 3))
    dirs[:, 0] = 0.02 * (pos[:, 0] / (0.4 * N * dx))
    dirs[:, 1] = 0.015 * (pos[:, 1] / (0.4 * N * dx))
    lam = _WL
    zR = np.pi * w0 ** 2 / lam
    j1 = 1.0 + 0.05 * rng.standard_normal(n)
    if kind == 'scalar':
        Q: Any = (-1j / (zR * j1)).astype(np.complex128)
    else:
        j2 = 1.0 + 0.07 * rng.standard_normal(n)
        Q = np.zeros((n, 2, 2), dtype=np.complex128)
        Q[:, 0, 0] = -1j / (zR * j1)
        Q[:, 1, 1] = -1j / (zR * j2)
        if kind == 'skew':
            Q[:, 0, 1] = Q[:, 1, 0] = 0.15 * (-1j / (zR * j1))
    return BeamletBundle(positions=pos, directions=dirs, Q=Q,
                         amplitude=np.ones(n, dtype=np.complex128),
                         waist0=np.full(n, w0))


def _measure_peak(fn):
    tracemalloc.start()
    try:
        out = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return out, peak


# ---------------------------------------------------------------------------
# B3 -- byte-identity of the windowed reconstruct (default budget, env unset)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('kind', ['scalar', 'tensor', 'skew'])
def test_windowed_reconstruct_byte_identical_to_preB3(kind, monkeypatch):
    """The B3 in-place-masking reconstruct is BYTE-IDENTICAL to the pre-B3
    two-np.where path at the default budget (env unset): same chunk boundaries,
    same bincount inputs, so no numerical drift."""
    monkeypatch.delenv('LUMENAIRY_MEM_BUDGET_MB', raising=False)
    N, dx = 512, 8e-6
    b = _windowed_bundle(N, dx, 72, 130e-6, kind=kind)
    assert not _fft_reconstruct_applicable(b, N, N, dx, dx, (0.0, 0.0))
    new = _reconstruct_windowed(b, Ny=N, Nx=N, dx=dx, dy=dx, centre=(0.0, 0.0),
                                wavelength=_WL, n_sigma=5.0, mem_budget_mb=512.0)
    ref = _reconstruct_windowed_preB3(
        b, Ny=N, Nx=N, dx=dx, dy=dx, centre=(0.0, 0.0),
        wavelength=_WL, n_sigma=5.0, mem_budget_mb=512.0)
    assert new.shape == ref.shape and new.dtype == ref.dtype
    assert np.array_equal(new, ref), (
        f"{kind}: max abs diff {np.max(np.abs(new - ref)):.3e}")


def test_windowed_reconstruct_matches_dense_sum(monkeypatch):
    """Independent oracle: the windowed reconstruct matches the DENSE coherent
    sum (window=None, a different code path) to the window truncation tail."""
    monkeypatch.delenv('LUMENAIRY_MEM_BUDGET_MB', raising=False)
    N, dx = 256, 8e-6
    b = _windowed_bundle(N, dx, 40, 150e-6, kind='scalar')
    win = _reconstruct_windowed(b, Ny=N, Nx=N, dx=dx, dy=dx, centre=(0.0, 0.0),
                                wavelength=_WL, n_sigma=6.0, mem_budget_mb=512.0)
    dense = np.asarray(la.reconstruct_field_from_beamlets(
        b, Ny=N, Nx=N, dx=dx, dy=dx, wavelength=_WL, centre=(0.0, 0.0),
        window=None))
    scale = float(np.max(np.abs(dense)))
    assert np.max(np.abs(win - dense)) < 1e-8 * scale


def test_windowed_env_budget_ceiling_matches_to_tolerance(monkeypatch):
    """LUMENAIRY_MEM_BUDGET_MB is honored as a ceiling: a tight budget re-chunks
    the accumulation (different float summation grouping) but the result matches
    the default-budget reconstruct to ~machine precision."""
    N, dx = 512, 8e-6
    b = _windowed_bundle(N, dx, 72, 130e-6, kind='scalar')
    monkeypatch.delenv('LUMENAIRY_MEM_BUDGET_MB', raising=False)
    base = _reconstruct_windowed(b, Ny=N, Nx=N, dx=dx, dy=dx, centre=(0.0, 0.0),
                                 wavelength=_WL, n_sigma=5.0, mem_budget_mb=512.0)
    monkeypatch.setenv('LUMENAIRY_MEM_BUDGET_MB', '16')
    tight = _reconstruct_windowed(b, Ny=N, Nx=N, dx=dx, dy=dx, centre=(0.0, 0.0),
                                  wavelength=_WL, n_sigma=5.0, mem_budget_mb=512.0)
    scale = float(np.max(np.abs(base)))
    assert np.max(np.abs(tight - base)) <= 1e-10 * scale


@pytest.mark.parametrize('bad', ['not-a-number', '-5', '0'])
def test_windowed_env_budget_invalid_ignored(bad, monkeypatch):
    """An unparseable / non-positive LUMENAIRY_MEM_BUDGET_MB is ignored with a
    warning -- the reconstruct still runs and matches the default."""
    N, dx = 256, 8e-6
    b = _windowed_bundle(N, dx, 40, 150e-6, kind='scalar')
    monkeypatch.delenv('LUMENAIRY_MEM_BUDGET_MB', raising=False)
    base = _reconstruct_windowed(b, Ny=N, Nx=N, dx=dx, dy=dx, centre=(0.0, 0.0),
                                 wavelength=_WL, n_sigma=5.0, mem_budget_mb=512.0)
    monkeypatch.setenv('LUMENAIRY_MEM_BUDGET_MB', bad)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = _reconstruct_windowed(b, Ny=N, Nx=N, dx=dx, dy=dx,
                                    centre=(0.0, 0.0), wavelength=_WL,
                                    n_sigma=5.0, mem_budget_mb=512.0)
    assert np.array_equal(got, base)


@pytest.mark.skipif(
    available_memory_bytes() < (4 << 30),
    reason="needs ~4 GB free to hold both the pre-B3 and B3 N=1024 peaks")
def test_windowed_reconstruct_peak_reduced_large_N():
    """RAM-guarded: at N=1024 the B3 reconstruct's tracemalloc peak is markedly
    lower than the pre-B3 path (and no longer overshoots mem_budget_mb ~2x),
    while staying byte-identical."""
    N, dx = 1024, 8e-6
    b = _windowed_bundle(N, dx, 96, 120e-6, kind='scalar')
    assert not _fft_reconstruct_applicable(b, N, N, dx, dx, (0.0, 0.0))
    budget = 512.0
    kw = dict(Ny=N, Nx=N, dx=dx, dy=dx, centre=(0.0, 0.0),
              wavelength=_WL, n_sigma=5.0, mem_budget_mb=budget)
    old, peak_old = _measure_peak(lambda: _reconstruct_windowed_preB3(b, **kw))
    new, peak_new = _measure_peak(lambda: _reconstruct_windowed(b, **kw))
    print(f"\nB3 peak @N=1024 budget={budget:.0f}MB: "
          f"before={peak_old / 1e6:.1f} MB  after={peak_new / 1e6:.1f} MB  "
          f"ratio={peak_new / peak_old:.2f}")
    assert np.array_equal(new, old)
    # measurably lower, and within ~1.2x of the nominal budget (was ~1.8x)
    assert peak_new < 0.75 * peak_old
    assert peak_new < 1.25 * budget * 1e6


@pytest.mark.skipif(
    available_memory_bytes() < (3 << 30),
    reason="needs ~3 GB free for the decentered GBD end-to-end run")
def test_apply_real_lens_gbd_decentered_runs_and_env_invariant(monkeypatch):
    """End-to-end: a decentered singlet through apply_real_lens_gbd (the path
    that drives _reconstruct_windowed) produces a finite field, and the tight
    env budget does not change the result beyond machine precision."""
    N, dx = 384, 8e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (2.6e-3) ** 2).astype(np.complex128)
    presc = {'wavelength': _WL, 'aperture_diameter': 10e-3, 'surfaces': [
        {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
         'glass_after': '_R3A', 'semi_diameter': 6e-3,
         'decenter': (0.4e-3, 0.0)},
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': '_R3A',
         'glass_after': 'air', 'semi_diameter': 6e-3}],
        'thicknesses': [5e-3], 'stop_index': 0}
    common = dict(prescription=presc, wavelength=_WL, dx=dx)
    monkeypatch.delenv('LUMENAIRY_MEM_BUDGET_MB', raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = np.asarray(la.apply_real_lens_gbd(E0, **common))
        monkeypatch.setenv('LUMENAIRY_MEM_BUDGET_MB', '32')
        out_env = np.asarray(la.apply_real_lens_gbd(E0, **common))
    assert out.shape == (N, N)
    assert np.isfinite(out).all()
    assert float(np.sum(np.abs(out) ** 2)) > 0.0
    scale = float(np.max(np.abs(out)))
    assert np.max(np.abs(out_env - out)) <= 1e-9 * scale


# ---------------------------------------------------------------------------
# B7 -- thread-safe normal-equations solve == lstsq to tolerance
# ---------------------------------------------------------------------------
def _cheb_design(n_pts, order, seed=0):
    """A normalised 2-D tensor-Chebyshev Vandermonde on scattered [-1,1] coords
    (well-conditioned, oversampled) -- the shape the traced fits solve."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, n_pts)
    v = rng.uniform(-1.0, 1.0, n_pts)
    mi = [(kx, ky) for kx in range(order + 1)
          for ky in range(order + 1 - kx)]
    import numpy.polynomial.chebyshev as C
    Tu = C.chebvander(u, order)
    Tv = C.chebvander(v, order)
    A = np.stack([Tu[:, kx] * Tv[:, ky] for (kx, ky) in mi], axis=1)
    return A


def test_normal_eq_solve_matches_lstsq_single_rhs():
    """_solve_lstsq_thread_safe reproduces np.linalg.lstsq to ~1e-9 on the
    well-conditioned Chebyshev design the traced fits use (single RHS)."""
    A = _cheb_design(600, 6)
    rng = np.random.default_rng(1)
    c_true = rng.standard_normal(A.shape[1])
    b = A @ c_true + 1e-6 * rng.standard_normal(A.shape[0])
    x_ne = _solve_lstsq_thread_safe(A, b)
    x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
    assert x_ne.shape == x_ls.shape
    assert np.allclose(x_ne, x_ls, rtol=1e-7, atol=1e-9)
    # and it actually solves the least-squares problem (recovers c_true)
    assert np.allclose(x_ne, c_true, atol=1e-4)


def test_normal_eq_solve_matches_lstsq_multi_rhs():
    """Multi-RHS (the coordinate-fit shape) is handled and matches lstsq."""
    A = _cheb_design(500, 5, seed=3)
    rng = np.random.default_rng(4)
    B = A @ rng.standard_normal((A.shape[1], 3)) + 1e-7 * rng.standard_normal(
        (A.shape[0], 3))
    x_ne = _solve_lstsq_thread_safe(A, B)
    x_ls, *_ = np.linalg.lstsq(A, B, rcond=None)
    assert x_ne.shape == (A.shape[1], 3)
    assert np.allclose(x_ne, x_ls, rtol=1e-7, atol=1e-8)


def test_solve_never_calls_gelsd_lstsq_for_wellconditioned(monkeypatch):
    """The default (well-conditioned) path never reaches np.linalg.lstsq -- the
    deadlock-prone gelsd SVD.  Poison lstsq and confirm the solve still works."""
    import lumenairy.elements._lens_traced as lt

    def _poisoned(*a, **k):
        raise AssertionError("np.linalg.lstsq (gelsd) must not be reached")

    monkeypatch.setattr(lt.np.linalg, 'lstsq', _poisoned)
    A = _cheb_design(400, 6, seed=5)
    rng = np.random.default_rng(6)
    b = A @ rng.standard_normal(A.shape[1])
    x = lt._solve_lstsq_thread_safe(A, b)
    assert np.isfinite(x).all()


@pytest.mark.parametrize('kw', [
    {},
    {'inversion_method': 'fit'},
    {'carrier': 'auto'},
])
def test_traced_field_matches_lstsq_reference(kw, monkeypatch):
    """apply_real_lens_traced with the production normal-equations solve matches
    the same run with a lstsq solve to tolerance, across the fit sites (default
    Cheb2D interp, the direct-inverse 'fit' path, and the carrier fit)."""
    import lumenairy.elements._lens_traced as lt
    N, dx = 320, 8e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (3e-3) ** 2).astype(np.complex128)
    presc = {'wavelength': _WL, 'aperture_diameter': 10e-3, 'surfaces': [
        {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
         'glass_after': '_R3A', 'semi_diameter': 6e-3},
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': '_R3A',
         'glass_after': 'air', 'semi_diameter': 6e-3}],
        'thicknesses': [5e-3], 'stop_index': 0}
    base = dict(prescription=presc, wavelength=_WL, dx=dx,
                on_undersample='silent', **kw)

    def _lstsq_solve(A, b):
        A = np.ascontiguousarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        out, *_ = np.linalg.lstsq(A, b, rcond=None)
        return out

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prod = np.asarray(la.apply_real_lens_traced(E0, **base))
        monkeypatch.setattr(lt, '_solve_lstsq_thread_safe', _lstsq_solve)
        ref = np.asarray(la.apply_real_lens_traced(E0, **base))
    assert np.isfinite(prod).all()
    scale = float(np.max(np.abs(ref)))
    assert np.max(np.abs(prod - ref)) <= 1e-8 * scale


def test_traced_fit_runs_under_multithread_blas_with_jax():
    """The traced fit runs to completion under multi-thread OpenBLAS with JAX
    loaded in the SAME process -- the configuration the pre-B7 gelsd lstsq
    deadlocked on.  Runs in a fresh subprocess (multi-thread env, jax imported
    first) with a hard timeout; a hang (regression) trips the timeout."""
    pytest.importorskip('jax')
    code = textwrap.dedent(
        """
        import os, numpy as np
        import jax, jax.numpy as jnp
        # spin up JAX's runtime / OpenMP pool before the BLAS fit
        _ = float(jnp.ones(256).sum())
        import lumenairy as la
        from lumenairy.glass import GLASS_REGISTRY
        GLASS_REGISTRY['_R3B'] = lambda wl: 1.5168
        N, dx, wl = 256, 8e-6, 1.31e-6
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        E0 = np.exp(-(X**2 + Y**2) / (3e-3)**2).astype(np.complex128)
        presc = {'wavelength': wl, 'aperture_diameter': 10e-3, 'surfaces': [
            {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
             'glass_after': '_R3B', 'semi_diameter': 6e-3},
            {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': '_R3B',
             'glass_after': 'air', 'semi_diameter': 6e-3}],
            'thicknesses': [5e-3], 'stop_index': 0}
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = np.asarray(la.apply_real_lens_traced(
                E0, prescription=presc, wavelength=wl, dx=dx,
                on_undersample='silent'))
        assert np.isfinite(out).all() and out.shape == (N, N)
        print('OK', float(np.sum(np.abs(out)**2)))
        """
    )
    env = dict(os.environ)
    # force the multi-thread BLAS + OpenMP nesting condition (NOT the CI pin)
    env['OMP_NUM_THREADS'] = '4'
    env['OPENBLAS_NUM_THREADS'] = '4'
    env.pop('LUMENAIRY_MEM_BUDGET_MB', None)
    try:
        proc = subprocess.run([sys.executable, '-c', code], env=env,
                              capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        pytest.fail("traced fit hung under multi-thread BLAS + JAX (deadlock)")
    assert proc.returncode == 0, (
        f"subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    assert 'OK' in proc.stdout

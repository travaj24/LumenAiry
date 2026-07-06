"""Maslov-method (phase-space asymptotic) propagator through a thick-lens
prescription.

Originally inlined in :mod:`lumenairy.elements.lenses` (and before that
in a now-removed top-level ``lens_maslov.py`` module).  Split out into
its own file in v3.5.5 to reduce ``lenses.py`` bloat.  Imports remain
backwards-compatible -- ``apply_real_lens_maslov`` is still re-exported
from :mod:`lumenairy.elements.lenses` for callers that import it from
there.

This module owns the Maslov NumPy implementation (the JAX variant
``apply_real_lens_maslov_jax`` lives in
:mod:`lumenairy.elements._lens_jax` because it shares the ``_cheb_*``
Chebyshev evaluators there with ``apply_real_lens_traced_jax``; it is
re-exported from :mod:`lumenairy.elements.lenses`).

Author: Andrew Traverso
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from .. import raytrace as rt
from .._math.chebyshev import (
    chebyshev_derivative_vandermonde as _chebyshev_derivative_vandermonde,
)
from .._math.chebyshev import (
    chebyshev_second_derivative_vandermonde as _chebyshev_second_derivative_vandermonde,
)

# v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
# The three Chebyshev Vandermonde helpers moved from
# ``lumenairy.elements.lenses`` to ``lumenairy._math.chebyshev``.  We
# import the new public names and bind them to the legacy
# underscore-prefixed locals so the rest of this module's call sites
# (~10 references) keep working unchanged.
from .._math.chebyshev import (
    chebyshev_vandermonde as _chebyshev_vandermonde,
)
from ..progress import call_progress

# Pixel-band size for the stationary_phase integrator's _opd_and_derivs
# evaluations.  None -> auto (memory-budgeted from the basis count).  A test
# seam: setting it to a small int forces maximal banding, which must produce
# byte-identical output to the unbanded path (the per-pixel work is
# independent).  Analogous to the quadrature integrator's ``chunk_v2`` param.
_SP_PIXEL_CHUNK = None

# Output-ROW band for the uniform-quadrature integrator (F2 follow-up).  The
# (N_out^2, M) design matrix G is the quadrature path's dominant allocation
# (451 GB at N=16384 / output_subsample=1), so instead of materialising it
# whole we band the output rows and build only a G-band per band.  None ->
# auto (memory-budgeted rows/band); a small int is a test seam forcing many
# bands (output is ~ULP-close; exactly byte-identical for the numpy path).
_QUAD_ROW_BAND = None

# M-P2 follow-up: the (N_out^2, M) design matrix G is a Kronecker product
# G[(iy,ix),m] = Ty[k2,iy]*Tx[k1,ix], so G @ H factorizes -- scatter H's rows
# by their (k1,k2) pair into a (P,P,.) tensor (P=poly_order+1), then a single
# einsum per chunk -- eliminating the G build and cutting the integration
# FLOPs by ~M/P (14-30x).  True (default) uses the factorized path; False
# falls back to the explicit per-row-band G (the ULP-level validation
# reference).  A tiny seam for A/B checks.
_QUAD_FACTORIZE = True

# A1 (v5.20): auto-resolution bounds for the uniform-quadrature v2 sampling.
# When the caller leaves ``n_v2`` unset (the new default), the quadrature path
# sizes it from the same v2-oscillation estimate the N2 under-resolution guard
# uses (want n_v2 >~ 4 * v2-oscillations), clamped to [_N_V2_AUTO_MIN,
# _N_V2_AUTO_MAX].  This makes the robust default integrator *properly
# resolved* out of the box instead of speckling at the old fixed n_v2=32
# (a demanding tight-focus chart wants ~150-200; see the 2026-07 audit
# remediation, A1 diagnosis).  The floor keeps low-NA charts byte-identical to
# the historical default; past the ceiling the N2 warning still fires and
# points at local_quadrature / stationary_phase (the cheap asymptotic
# evaluators, which are the correct choice at high production NA).
_N_V2_AUTO_MIN = 32
_N_V2_AUTO_MAX = 256

# Other shared helpers still live in lenses.py.
from .lenses import (
    NUMEXPR_AVAILABLE,
    _ensure_numexpr_loaded,
    _fit_normaliser,
    _multi_indices_total_degree,
    _warn_if_aperture_exceeds_grid,
)

# ---------------------------------------------------------------------------
# M-P4 (audit perf): optional Numba kernel for the 4-variable Chebyshev
# value+derivative sum ``_opd_and_derivs``.  The NumPy path materialises eight
# (M, n_px) basis-gathered arrays and six full-array reductions per call; a
# single @njit(parallel) kernel collapses that to O(poly_order) stack work per
# sample via 3-term Chebyshev recurrences (T, T'=n*U_{n-1}, and the
# differentiated T'' recurrence -- byte-for-byte the same recurrences as
# lumenairy._math.chebyshev, so the only deviation from NumPy is the
# term-reduction order -> ULP).  Lazily compiled on first use (numba import is
# ~1.8 s); pure-NumPy fallback when numba is absent.  ``_MASLOV_USE_NUMBA`` is
# a test seam (set False to force the NumPy reference).
_MASLOV_USE_NUMBA = True

import importlib.util as _mz_ilu  # noqa: E402

_MZ_NUMBA_AVAILABLE = _mz_ilu.find_spec("numba") is not None
_mz_njit = None
_mz_prange = None
_MZ_KERNELS: dict = {}


def _mz_load_numba():
    global _mz_njit, _mz_prange
    if _mz_njit is not None:
        return True
    if not _MZ_NUMBA_AVAILABLE:
        return False
    from numba import njit as _nj
    from numba import prange as _pr
    _mz_njit, _mz_prange = _nj, _pr
    return True


def _get_cheb4d_numba():
    """Compile (once) and return the 4-var Chebyshev value+deriv kernel, or
    None if numba is unavailable."""
    if "cheb4d" in _MZ_KERNELS:
        return _MZ_KERNELS["cheb4d"]
    if not _mz_load_numba():
        _MZ_KERNELS["cheb4d"] = None
        return None

    @_mz_njit(cache=True, parallel=True, fastmath=True)
    def _cheb4d_opd_derivs(coef, K1, K2, K3, K4, u1, u2, u3, u4, P):
        n = u1.shape[0]
        M = coef.shape[0]
        f = np.zeros(n)
        df3 = np.zeros(n)
        df4 = np.zeros(n)
        d233 = np.zeros(n)
        d234 = np.zeros(n)
        d244 = np.zeros(n)
        for i in _mz_prange(n):
            a1 = u1[i]
            a2 = u2[i]
            a3 = u3[i]
            a4 = u4[i]
            # T_n (first kind) for all four variables
            Tu1 = np.empty(P + 1)
            Tu2 = np.empty(P + 1)
            Tu3 = np.empty(P + 1)
            Tu4 = np.empty(P + 1)
            Tu1[0] = 1.0
            Tu2[0] = 1.0
            Tu3[0] = 1.0
            Tu4[0] = 1.0
            if P >= 1:
                Tu1[1] = a1
                Tu2[1] = a2
                Tu3[1] = a3
                Tu4[1] = a4
            for m in range(2, P + 1):
                Tu1[m] = 2.0 * a1 * Tu1[m - 1] - Tu1[m - 2]
                Tu2[m] = 2.0 * a2 * Tu2[m - 1] - Tu2[m - 2]
                Tu3[m] = 2.0 * a3 * Tu3[m - 1] - Tu3[m - 2]
                Tu4[m] = 2.0 * a4 * Tu4[m - 1] - Tu4[m - 2]
            # U_n (second kind) for u3, u4 -> first derivative T'_n = n*U_{n-1}
            Uu3 = np.empty(P + 1)
            Uu4 = np.empty(P + 1)
            Uu3[0] = 1.0
            Uu4[0] = 1.0
            if P >= 1:
                Uu3[1] = 2.0 * a3
                Uu4[1] = 2.0 * a4
            for m in range(2, P + 1):
                Uu3[m] = 2.0 * a3 * Uu3[m - 1] - Uu3[m - 2]
                Uu4[m] = 2.0 * a4 * Uu4[m - 1] - Uu4[m - 2]
            dTu3 = np.zeros(P + 1)
            dTu4 = np.zeros(P + 1)
            for m in range(1, P + 1):
                dTu3[m] = float(m) * Uu3[m - 1]
                dTu4[m] = float(m) * Uu4[m - 1]
            # T''_n via differentiated recurrence: T''_2=4,
            # T''_{n+1} = 2u T''_n + 4 T'_n - T''_{n-1}
            d2Tu3 = np.zeros(P + 1)
            d2Tu4 = np.zeros(P + 1)
            if P >= 2:
                d2Tu3[2] = 4.0
                d2Tu4[2] = 4.0
            for m in range(2, P):
                d2Tu3[m + 1] = 2.0 * a3 * d2Tu3[m] + 4.0 * dTu3[m] - d2Tu3[m - 1]
                d2Tu4[m + 1] = 2.0 * a4 * d2Tu4[m] + 4.0 * dTu4[m] - d2Tu4[m - 1]
            sf = 0.0
            sdf3 = 0.0
            sdf4 = 0.0
            sd233 = 0.0
            sd234 = 0.0
            sd244 = 0.0
            for mm in range(M):
                k1 = K1[mm]
                k2 = K2[mm]
                k3 = K3[mm]
                k4 = K4[mm]
                t12 = Tu1[k1] * Tu2[k2]
                base = coef[mm] * t12       # matches NumPy's c * (T1b*T2b)
                t3 = Tu3[k3]
                t4 = Tu4[k4]
                dt3 = dTu3[k3]
                dt4 = dTu4[k4]
                sf += base * t3 * t4
                sdf3 += base * dt3 * t4
                sdf4 += base * t3 * dt4
                sd233 += base * d2Tu3[k3] * t4
                sd244 += base * t3 * d2Tu4[k4]
                sd234 += base * dt3 * dt4
            f[i] = sf
            df3[i] = sdf3
            df4[i] = sdf4
            d233[i] = sd233
            d234[i] = sd234
            d244[i] = sd244
        return f, df3, df4, d233, d234, d244

    _MZ_KERNELS["cheb4d"] = _cheb4d_opd_derivs
    return _cheb4d_opd_derivs


def _opd6_numpy(coef, K1, K2, K3, K4, u1, u2, u3, u4, P):
    """NumPy reference for the 4-var Chebyshev value + v2-derivatives.  Returns
    (f, df_du3, df_du4, d2f_33, d2f_34, d2f_44)."""
    T1 = _chebyshev_vandermonde(u1, P)
    T2 = _chebyshev_vandermonde(u2, P)
    T3 = _chebyshev_vandermonde(u3, P)
    T4 = _chebyshev_vandermonde(u4, P)
    dT3 = _chebyshev_derivative_vandermonde(u3, P)
    dT4 = _chebyshev_derivative_vandermonde(u4, P)
    d2T3 = _chebyshev_second_derivative_vandermonde(u3, P)
    d2T4 = _chebyshev_second_derivative_vandermonde(u4, P)
    T1b = T1[K1]
    T2b = T2[K2]
    T3b = T3[K3]
    T4b = T4[K4]
    dT3b = dT3[K3]
    dT4b = dT4[K4]
    d2T3b = d2T3[K3]
    d2T4b = d2T4[K4]
    T12 = T1b * T2b
    c = coef[:, None]
    f = np.sum(c * T12 * T3b * T4b, axis=0)
    df_du3 = np.sum(c * T12 * dT3b * T4b, axis=0)
    df_du4 = np.sum(c * T12 * T3b * dT4b, axis=0)
    d2f_33 = np.sum(c * T12 * d2T3b * T4b, axis=0)
    d2f_44 = np.sum(c * T12 * T3b * d2T4b, axis=0)
    d2f_34 = np.sum(c * T12 * dT3b * dT4b, axis=0)
    return f, df_du3, df_du4, d2f_33, d2f_34, d2f_44


def _opd6(coef, K1, K2, K3, K4, u1, u2, u3, u4, P):
    """Dispatch the 4-var Chebyshev value+deriv sum to the Numba kernel
    (default, when available) or the NumPy reference.  Result-identical to
    ULP; the kernel avoids the eight (M, n) basis arrays + six reductions."""
    if _MASLOV_USE_NUMBA:
        kern = _get_cheb4d_numba()
        if kern is not None:
            return kern(
                np.ascontiguousarray(coef, dtype=np.float64),
                np.ascontiguousarray(K1, dtype=np.int64),
                np.ascontiguousarray(K2, dtype=np.int64),
                np.ascontiguousarray(K3, dtype=np.int64),
                np.ascontiguousarray(K4, dtype=np.int64),
                np.ascontiguousarray(u1, dtype=np.float64),
                np.ascontiguousarray(u2, dtype=np.float64),
                np.ascontiguousarray(u3, dtype=np.float64),
                np.ascontiguousarray(u4, dtype=np.float64),
                int(P))
    return _opd6_numpy(coef, K1, K2, K3, K4, u1, u2, u3, u4, P)


def _opd_vd3_numpy(cop, csx, csy, K1, K2, K3, K4, u1, u2, u3, u4, P):
    """Value (+ v2 first-derivatives for s1x/s1y) of the SHARED 4-var Chebyshev
    basis for the three coefficient sets at once.  Returns
    ``(opd_v, s1x_v, ds1x3, ds1x4, s1y_v, ds1y3, ds1y4)``.  The local_quadrature
    integrand loop needs opd VALUE and s1x/s1y value + first derivatives, never
    second derivatives (those are the one-time per-pixel Hessian), and evaluates
    all three at the SAME query points -- so this builds the basis once, skips
    the T'' recurrence entirely, and shares it across opd/s1x/s1y (vs three
    separate 6-output ``_opd6`` calls that rebuild the basis 3x)."""
    T1 = _chebyshev_vandermonde(u1, P)
    T2 = _chebyshev_vandermonde(u2, P)
    T3 = _chebyshev_vandermonde(u3, P)
    T4 = _chebyshev_vandermonde(u4, P)
    dT3 = _chebyshev_derivative_vandermonde(u3, P)
    dT4 = _chebyshev_derivative_vandermonde(u4, P)
    T12 = T1[K1] * T2[K2]
    T3b = T3[K3]
    T4b = T4[K4]
    val = T12 * T3b * T4b            # (M, n) -- shared value basis
    d3 = T12 * dT3[K3] * T4b
    d4 = T12 * T3b * dT4[K4]
    return (np.sum(cop[:, None] * val, axis=0),
            np.sum(csx[:, None] * val, axis=0),
            np.sum(csx[:, None] * d3, axis=0),
            np.sum(csx[:, None] * d4, axis=0),
            np.sum(csy[:, None] * val, axis=0),
            np.sum(csy[:, None] * d3, axis=0),
            np.sum(csy[:, None] * d4, axis=0))


def _get_cheb4d_vd3_numba():
    """Numba twin of :func:`_opd_vd3_numpy` (value + 1st-deriv, three coef sets,
    shared basis, no T'' recurrence), or None if numba is unavailable."""
    if "cheb4d_vd3" in _MZ_KERNELS:
        return _MZ_KERNELS["cheb4d_vd3"]
    if not _mz_load_numba():
        _MZ_KERNELS["cheb4d_vd3"] = None
        return None

    @_mz_njit(cache=True, parallel=True, fastmath=True)
    def _cheb4d_vd3(cop, csx, csy, K1, K2, K3, K4, u1, u2, u3, u4, P):
        n = u1.shape[0]
        M = cop.shape[0]
        opd_v = np.zeros(n)
        s1x_v = np.zeros(n)
        ds1x3 = np.zeros(n)
        ds1x4 = np.zeros(n)
        s1y_v = np.zeros(n)
        ds1y3 = np.zeros(n)
        ds1y4 = np.zeros(n)
        for i in _mz_prange(n):
            a1 = u1[i]; a2 = u2[i]; a3 = u3[i]; a4 = u4[i]
            Tu1 = np.empty(P + 1); Tu2 = np.empty(P + 1)
            Tu3 = np.empty(P + 1); Tu4 = np.empty(P + 1)
            Tu1[0] = 1.0; Tu2[0] = 1.0; Tu3[0] = 1.0; Tu4[0] = 1.0
            if P >= 1:
                Tu1[1] = a1; Tu2[1] = a2; Tu3[1] = a3; Tu4[1] = a4
            for m in range(2, P + 1):
                Tu1[m] = 2.0 * a1 * Tu1[m - 1] - Tu1[m - 2]
                Tu2[m] = 2.0 * a2 * Tu2[m - 1] - Tu2[m - 2]
                Tu3[m] = 2.0 * a3 * Tu3[m - 1] - Tu3[m - 2]
                Tu4[m] = 2.0 * a4 * Tu4[m - 1] - Tu4[m - 2]
            Uu3 = np.empty(P + 1); Uu4 = np.empty(P + 1)
            Uu3[0] = 1.0; Uu4[0] = 1.0
            if P >= 1:
                Uu3[1] = 2.0 * a3; Uu4[1] = 2.0 * a4
            for m in range(2, P + 1):
                Uu3[m] = 2.0 * a3 * Uu3[m - 1] - Uu3[m - 2]
                Uu4[m] = 2.0 * a4 * Uu4[m - 1] - Uu4[m - 2]
            dTu3 = np.zeros(P + 1); dTu4 = np.zeros(P + 1)
            for m in range(1, P + 1):
                dTu3[m] = float(m) * Uu3[m - 1]
                dTu4[m] = float(m) * Uu4[m - 1]
            sopd = 0.0; sxv = 0.0; sx3 = 0.0; sx4 = 0.0
            syv = 0.0; sy3 = 0.0; sy4 = 0.0
            for mm in range(M):
                t12 = Tu1[K1[mm]] * Tu2[K2[mm]]
                t3 = Tu3[K3[mm]]; t4 = Tu4[K4[mm]]
                dt3 = dTu3[K3[mm]]; dt4 = dTu4[K4[mm]]
                vv = t12 * t3 * t4
                v3 = t12 * dt3 * t4
                v4 = t12 * t3 * dt4
                sopd += cop[mm] * vv
                sxv += csx[mm] * vv; sx3 += csx[mm] * v3; sx4 += csx[mm] * v4
                syv += csy[mm] * vv; sy3 += csy[mm] * v3; sy4 += csy[mm] * v4
            opd_v[i] = sopd
            s1x_v[i] = sxv; ds1x3[i] = sx3; ds1x4[i] = sx4
            s1y_v[i] = syv; ds1y3[i] = sy3; ds1y4[i] = sy4
        return opd_v, s1x_v, ds1x3, ds1x4, s1y_v, ds1y3, ds1y4

    _MZ_KERNELS["cheb4d_vd3"] = _cheb4d_vd3
    return _cheb4d_vd3


def _opd_vd3(cop, csx, csy, K1, K2, K3, K4, u1, u2, u3, u4, P):
    """Dispatch the shared value+1st-deriv 3-coef kernel to Numba (default) or
    the NumPy reference; ULP-equal, ~2x cheaper than three ``_opd6`` calls."""
    if _MASLOV_USE_NUMBA:
        kern = _get_cheb4d_vd3_numba()
        if kern is not None:
            return kern(
                np.ascontiguousarray(cop, dtype=np.float64),
                np.ascontiguousarray(csx, dtype=np.float64),
                np.ascontiguousarray(csy, dtype=np.float64),
                np.ascontiguousarray(K1, dtype=np.int64),
                np.ascontiguousarray(K2, dtype=np.int64),
                np.ascontiguousarray(K3, dtype=np.int64),
                np.ascontiguousarray(K4, dtype=np.int64),
                np.ascontiguousarray(u1, dtype=np.float64),
                np.ascontiguousarray(u2, dtype=np.float64),
                np.ascontiguousarray(u3, dtype=np.float64),
                np.ascontiguousarray(u4, dtype=np.float64),
                int(P))
    return _opd_vd3_numpy(cop, csx, csy, K1, K2, K3, K4, u1, u2, u3, u4, P)


# CuPy fused kernel for the 4-var Chebyshev value+derivs -- the device twin of
# the Numba ``_cheb4d_opd_derivs``.  One thread per query point runs the O(P)
# T/U/T'/T'' recurrences in local memory then loops over the M multi-indices,
# so it avoids the (M, n) global temporaries the numpy-style ``_opd6_xp`` path
# materializes (~1.7 GB at n~1e6) -- those temporaries make the asymptotic
# evaluators MEMORY-BOUND and slower than the CPU on the GPU.  PMAX bounds the
# per-thread local arrays (poly_order + 1 <= PMAX).
_MZ_CUPY_KERNELS = {}
_MZ_CUPY_PMAX = 24
_MZ_CHEB4D_CUDA = r'''
extern "C" __global__ void cheb4d_opd_derivs(
    const double* coef, const long long* K1, const long long* K2,
    const long long* K3, const long long* K4,
    const double* u1, const double* u2, const double* u3, const double* u4,
    const int P, const int M, const long long n,
    double* f, double* df3, double* df4,
    double* d233, double* d234, double* d244)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int PM = %d;
    double a1 = u1[i], a2 = u2[i], a3 = u3[i], a4 = u4[i];
    double Tu1[PM], Tu2[PM], Tu3[PM], Tu4[PM];
    double Uu3[PM], Uu4[PM], dTu3[PM], dTu4[PM], d2Tu3[PM], d2Tu4[PM];
    Tu1[0] = Tu2[0] = Tu3[0] = Tu4[0] = 1.0;
    if (P >= 1) { Tu1[1] = a1; Tu2[1] = a2; Tu3[1] = a3; Tu4[1] = a4; }
    for (int m = 2; m <= P; ++m) {
        Tu1[m] = 2.0 * a1 * Tu1[m-1] - Tu1[m-2];
        Tu2[m] = 2.0 * a2 * Tu2[m-1] - Tu2[m-2];
        Tu3[m] = 2.0 * a3 * Tu3[m-1] - Tu3[m-2];
        Tu4[m] = 2.0 * a4 * Tu4[m-1] - Tu4[m-2];
    }
    Uu3[0] = Uu4[0] = 1.0;
    if (P >= 1) { Uu3[1] = 2.0 * a3; Uu4[1] = 2.0 * a4; }
    for (int m = 2; m <= P; ++m) {
        Uu3[m] = 2.0 * a3 * Uu3[m-1] - Uu3[m-2];
        Uu4[m] = 2.0 * a4 * Uu4[m-1] - Uu4[m-2];
    }
    for (int m = 0; m <= P; ++m) { dTu3[m] = 0.0; dTu4[m] = 0.0;
                                   d2Tu3[m] = 0.0; d2Tu4[m] = 0.0; }
    for (int m = 1; m <= P; ++m) { dTu3[m] = (double)m * Uu3[m-1];
                                   dTu4[m] = (double)m * Uu4[m-1]; }
    if (P >= 2) { d2Tu3[2] = 4.0; d2Tu4[2] = 4.0; }
    for (int m = 2; m < P; ++m) {
        d2Tu3[m+1] = 2.0 * a3 * d2Tu3[m] + 4.0 * dTu3[m] - d2Tu3[m-1];
        d2Tu4[m+1] = 2.0 * a4 * d2Tu4[m] + 4.0 * dTu4[m] - d2Tu4[m-1];
    }
    double sf = 0.0, sdf3 = 0.0, sdf4 = 0.0;
    double sd233 = 0.0, sd234 = 0.0, sd244 = 0.0;
    for (int mm = 0; mm < M; ++mm) {
        int k1 = (int)K1[mm], k2 = (int)K2[mm];
        int k3 = (int)K3[mm], k4 = (int)K4[mm];
        double base = coef[mm] * Tu1[k1] * Tu2[k2];
        double t3 = Tu3[k3], t4 = Tu4[k4];
        double dt3 = dTu3[k3], dt4 = dTu4[k4];
        sf   += base * t3 * t4;
        sdf3 += base * dt3 * t4;
        sdf4 += base * t3 * dt4;
        sd233 += base * d2Tu3[k3] * t4;
        sd244 += base * t3 * d2Tu4[k4];
        sd234 += base * dt3 * dt4;
    }
    f[i] = sf; df3[i] = sdf3; df4[i] = sdf4;
    d233[i] = sd233; d234[i] = sd234; d244[i] = sd244;
}
''' % _MZ_CUPY_PMAX


def _get_cheb4d_cupy(cp):
    """Compile (once, cached) and return the CuPy RawKernel twin of the Numba
    4-var Chebyshev value+deriv kernel."""
    if 'cheb4d' not in _MZ_CUPY_KERNELS:
        _MZ_CUPY_KERNELS['cheb4d'] = cp.RawKernel(
            _MZ_CHEB4D_CUDA, 'cheb4d_opd_derivs')
    return _MZ_CUPY_KERNELS['cheb4d']


def _opd6_cupy(cp, coef, K1, K2, K3, K4, u1, u2, u3, u4, P):
    """Device evaluation of the 4-var Chebyshev value + v2 derivatives via the
    fused RawKernel (one thread per query point).  Result-close to the Numba /
    NumPy kernels (~1e-13; strict-IEEE vs numba fastmath reassociation)."""
    if P + 1 > _MZ_CUPY_PMAX:
        raise ValueError(
            f"apply_real_lens_maslov: use_gpu asymptotic evaluators support "
            f"poly_order <= {_MZ_CUPY_PMAX - 1} (got {P}); raise _MZ_CUPY_PMAX "
            f"or use use_gpu=False.")
    coef = cp.ascontiguousarray(coef, dtype=cp.float64)
    K1 = cp.ascontiguousarray(K1, dtype=cp.int64)
    K2 = cp.ascontiguousarray(K2, dtype=cp.int64)
    K3 = cp.ascontiguousarray(K3, dtype=cp.int64)
    K4 = cp.ascontiguousarray(K4, dtype=cp.int64)
    u1 = cp.ascontiguousarray(u1, dtype=cp.float64)
    u2 = cp.ascontiguousarray(u2, dtype=cp.float64)
    u3 = cp.ascontiguousarray(u3, dtype=cp.float64)
    u4 = cp.ascontiguousarray(u4, dtype=cp.float64)
    n = int(u1.shape[0])
    M = int(coef.shape[0])
    outs = [cp.empty(n, dtype=cp.float64) for _ in range(6)]
    kern = _get_cheb4d_cupy(cp)
    threads = 128
    blocks = (n + threads - 1) // threads
    kern((blocks,), (threads,),
         (coef, K1, K2, K3, K4, u1, u2, u3, u4,
          np.int32(P), np.int32(M), np.int64(n),
          outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]))
    return tuple(outs)


def _opd6_xp(xp, coef, K1, K2, K3, K4, u1, u2, u3, u4, P):
    """xp-dispatched (NumPy or CuPy) twin of :func:`_opd6_numpy` -- the 4-var
    Chebyshev value + v2 first/second derivatives.  With ``xp=np`` it is
    byte-identical to ``_opd6_numpy`` (the numpy (M, n) path); with ``xp=cupy``
    it dispatches to the fused :func:`_opd6_cupy` RawKernel (per-pixel, no
    (M, n) temporaries) so the asymptotic evaluators are actually FAST on the
    GPU rather than memory-bound.  Returns
    ``(f, df_du3, df_du4, d2f_33, d2f_34, d2f_44)``.
    """
    if xp is not np:
        return _opd6_cupy(xp, coef, K1, K2, K3, K4, u1, u2, u3, u4, P)
    T1 = _chebyshev_vandermonde(u1, P, xp=xp)
    T2 = _chebyshev_vandermonde(u2, P, xp=xp)
    T3 = _chebyshev_vandermonde(u3, P, xp=xp)
    T4 = _chebyshev_vandermonde(u4, P, xp=xp)
    dT3 = _chebyshev_derivative_vandermonde(u3, P, xp=xp)
    dT4 = _chebyshev_derivative_vandermonde(u4, P, xp=xp)
    d2T3 = _chebyshev_second_derivative_vandermonde(u3, P, xp=xp)
    d2T4 = _chebyshev_second_derivative_vandermonde(u4, P, xp=xp)
    T1b = T1[K1]
    T2b = T2[K2]
    T3b = T3[K3]
    T4b = T4[K4]
    dT3b = dT3[K3]
    dT4b = dT4[K4]
    d2T3b = d2T3[K3]
    d2T4b = d2T4[K4]
    T12 = T1b * T2b
    c = coef[:, None]
    f = xp.sum(c * T12 * T3b * T4b, axis=0)
    df_du3 = xp.sum(c * T12 * dT3b * T4b, axis=0)
    df_du4 = xp.sum(c * T12 * T3b * dT4b, axis=0)
    d2f_33 = xp.sum(c * T12 * d2T3b * T4b, axis=0)
    d2f_44 = xp.sum(c * T12 * T3b * d2T4b, axis=0)
    d2f_34 = xp.sum(c * T12 * dT3b * dT4b, axis=0)
    return f, df_du3, df_du4, d2f_33, d2f_34, d2f_44


def _maslov_newton_saddle_xp(xp, opd6, coef_opd, u_s2x, u_s2y, inbox_flat,
                             newton_iter, newton_tol, lin_v3, lin_v4):
    """Per-pixel Newton solve for the v2 stationary point, shared by the
    stationary_phase / local_quadrature GPU twins.

    Unlike the CPU integrators (which shrink an ``active`` boolean subset each
    iteration) this evaluates ALL pixels every iteration and freezes the step
    of already-converged pixels (``dv = 0``) -- the SIMD-friendly form for the
    GPU, and numerically equivalent to the CPU active-subset loop (a frozen
    pixel's ``u_v2`` never changes, so its later gradients are irrelevant).
    Out-of-box pixels start ``converged`` (frozen at ``u_v2 = 0``) and are
    zeroed by the caller.  Returns ``(u_v2x, u_v2y, converged)``.
    """
    n_px = u_s2x.shape[0]
    u_v2x = xp.zeros(n_px, dtype=xp.float64)
    u_v2y = xp.zeros(n_px, dtype=xp.float64)
    converged = ~inbox_flat
    for _it in range(newton_iter):
        _, g3, g4, H33, H34, H44 = opd6(coef_opd, u_s2x, u_s2y, u_v2x, u_v2y)
        g3 = g3 + lin_v3
        g4 = g4 + lin_v4
        det_H = H33 * H44 - H34 * H34
        det_safe = xp.where(xp.abs(det_H) < 1e-30,
                            xp.sign(det_H) * 1e-30 + 1e-30, det_H)
        dv3 = -(H44 * g3 - H34 * g4) / det_safe
        dv4 = -(-H34 * g3 + H33 * g4) / det_safe
        step_size = xp.sqrt(dv3 ** 2 + dv4 ** 2)
        damp = xp.where(step_size > 0.5,
                        0.5 / xp.maximum(step_size, 1e-30), 1.0)
        dv3 = dv3 * damp
        dv4 = dv4 * damp
        # Freeze already-converged (incl. out-of-box) pixels.
        dv3 = xp.where(converged, 0.0, dv3)
        dv4 = xp.where(converged, 0.0, dv4)
        u_v2x = xp.clip(u_v2x + dv3, -1.0, 1.0)
        u_v2y = xp.clip(u_v2y + dv4, -1.0, 1.0)
        grad_mag = xp.sqrt(g3 ** 2 + g4 ** 2)
        converged = converged | (grad_mag < newton_tol)
    return u_v2x, u_v2y, converged


def _solve_fit(A, RHS, gram_factor=None):
    """Least-squares solve for the Maslov Chebyshev fit ``A @ coef ~= RHS``.

    v5.21 (M-P5 follow-up): normal-equations Cholesky (``G = A^T A``; solve
    ``G coef = A^T RHS``) instead of the ``gelsd`` full-SVD ``lstsq``.  ``A`` is
    a normalized tensor-Chebyshev Vandermonde -- well-conditioned and ~1.5x
    oversampled -- so squaring the condition number in ``G`` is safe, and
    ``cho_factor(G)`` is O(M^3) with tiny ``M`` (70 at poly_order=4) vs the
    O(n_rays M^2) SVD.  ``G`` (and its Cholesky factor) depend ONLY on the ray
    node grid + poly_order, not the field/wavelength, so a caller sweeping the
    SAME optic can precompute ``gram_factor`` once and pass it in (only the
    cheap ``A^T RHS`` GEMM + back-substitution then re-run per field).  Falls
    back to LU solve, then to ``lstsq`` (SVD), if ``G`` is not positive-definite
    (a rank-deficient / ill-conditioned freeform).  Returns ``coef`` (M, k).
    """
    b = A.T @ RHS
    if gram_factor is not None:
        try:
            from scipy.linalg import cho_solve
            return cho_solve(gram_factor, b, check_finite=False)
        except Exception:
            pass
    G = A.T @ A
    try:
        from scipy.linalg import cho_factor, cho_solve
        return cho_solve(cho_factor(G, check_finite=False), b,
                         check_finite=False)
    except Exception:
        try:
            return np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(A, RHS, rcond=None)
            return coef


def _gram_cho_factor(A):
    """Cholesky factor of ``A^T A`` for :func:`_solve_fit` reuse across a
    same-optic sweep, or ``None`` if scipy is absent / ``G`` is not PD."""
    try:
        from scipy.linalg import cho_factor
        return cho_factor(A.T @ A, check_finite=False)
    except Exception:
        return None


def apply_real_lens_maslov(
    E_in: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    ray_field_samples: int = 16,
    ray_pupil_samples: int = 16,
    poly_order: int = 4,
    n_v2: Optional[int] = None,
    output_subsample: int = 1,
    roi: Optional[Any] = None,
    output_plane_distance: float = 0.0,
    output_plane_n: float = 1.0,
    extract_linear_phase: bool = True,
    chunk_v2: int = 64,
    use_numexpr: Optional[bool] = None,
    integration_method: str = 'auto',
    stationary_newton_iter: int = 12,
    stationary_newton_tol: float = 1e-10,
    local_n_samples: int = 8,
    local_window_sigma: float = 3.0,
    collimated_input: bool = False,
    input_na: Optional[float] = None,
    normalize_output: str = 'power',
    verbose: bool = False,
    progress: Optional[Any] = None,
    use_gpu: bool = False,
    fold_split: bool = False,
) -> np.ndarray:
    """
    Phase-space / Maslov propagator through a thick-lens prescription.

    See Also
    --------
    apply_real_lens :
        Analytic split-step thin-element model.  Default fast path
        when the output plane is well away from any caustic and
        autodiff gradients aren't required.
    apply_real_lens_traced :
        Per-pixel ray-traced OPL + wave-optics amplitude envelope.
        Achieves sub-nm OPD on cemented doublets, but is **not**
        differentiable (uses Newton inversion of the
        entrance->exit map) and breaks down at caustics where the
        per-pixel ray map becomes multi-valued.
    apply_real_lens_maslov_jax :
        JAX-traced twin of this function for autodiff /
        gradient-based design optimisation.

    Quick decision guide
    --------------------
    * Default / fast wave model -> ``apply_real_lens``.
    * Sub-nm OPD on cemented doublets / multi-surface curved interfaces
      -> ``apply_real_lens_traced``.
    * Inside a JAX-autodiff design optimisation, or near a caustic
      -> ``apply_real_lens_maslov`` (this function) /
      ``apply_real_lens_maslov_jax``.

    Description
    -----------
    Traces a Chebyshev-node grid of rays from the entrance plane of
    ``lens_prescription`` to the exit plane, fits a 4-variable
    Chebyshev tensor-product polynomial to ``s1(s2, v2)`` and
    ``OPD(s2, v2)``, then evaluates the Maslov integral

        E(s2) = integral E_in(s1(s2, v2)) * exp(2 pi i OPD(s2, v2))
                          * |det(ds1/dv2)|  d^2 v2

    at each output pixel.  See the v3.4.x release notes (or the
    ``Phase-Space Asymptotic Propagator`` wiki page) for the full
    physics derivation and quadrature/stationary-phase trade-offs.

    Parameters mirror the inline-in-lenses.py predecessor exactly so
    no caller-side changes are required.

    Anamorphic pixels (``dy != dx``) are supported (v5.20): the
    entrance/exit sampler, output axes, and angular-content estimate
    use the separate ``dx``/``dy`` pitches, and the Chebyshev fit +
    per-axis quadrature Vandermondes already normalise x and y
    independently.  ``dy`` resolves ``None -> get_default_dy() -> dx``
    like ``apply_real_lens``.  The array itself must still be **square**
    (``N x N``); a rectangular *array* (``Ny != Nx``) and the ``roi=``
    window under anamorphic pixels are not yet supported and raise --
    use ``apply_real_lens`` for those.

    ``n_v2`` (uniform-quadrature v2 sampling) defaults to ``None`` ->
    **auto-resolution**: the ``integration_method='quadrature'`` path
    sizes it from the fitted OPD's v2-oscillation count (want
    ``n_v2 >~ 4 * v2-oscillations``), clamped to
    ``[_N_V2_AUTO_MIN, _N_V2_AUTO_MAX]``.  This keeps the robust default
    integrator properly resolved (a demanding tight-focus chart wants
    ~150-200 samples; the old fixed default of 32 speckled).  Low-NA
    charts clamp to the floor and stay byte-identical to the historical
    default; past the ceiling the N2 warning fires and steers you to the
    cheap asymptotic evaluators.  ``n_v2`` is ignored by
    ``local_quadrature`` / ``stationary_phase`` (they window around the
    per-pixel saddle rather than sample a uniform v2 grid); pass an
    explicit int to pin the sampling for reproducibility.

    ``integration_method='auto'`` (v5.21; the **default**) resolves
    to a concrete integrator from the fitted chart's v2-oscillation count:
    **uniform 'quadrature'** when it is well-resolved
    (``4 * v2_osc <= _N_V2_AUTO_MAX``) -- exact and caustic-safe, and where
    low-oscillation / near-caustic charts fall -- and the fast asymptotic
    **'local_quadrature'** only when uniform quadrature would need more than the
    sample cap (the very oscillatory / high-NA regime where quadrature is both
    slow and speckles).  Byte-identical to the method it picks in the
    well-resolved regime (auto -> quadrature at the same auto-sized ``n_v2``);
    it only diverges from the old ``'quadrature'`` default in the under-resolved
    near-caustic regime, where that default clamped ``n_v2`` and emitted an
    "under-resolved" warning anyway.  Measured **357x** faster (and no
    multi-GB / minute-scale near-focus quadrature) than the old default on a
    high-NA singlet chart while staying on the safe quadrature elsewhere.  Pass
    ``integration_method='quadrature'`` explicitly to force the exact uniform
    quadrature everywhere.

    ``fold_split=True`` (v5.21) auto-handles a **folded** prescription (one with
    fold mirrors) instead of raising: it splits at every fold
    (:func:`lumenairy.io.split_prescription_at_mirrors`) and chains this Maslov
    propagator over each refractive leg with a free-space + :func:`apply_mirror`
    (flat -> field-preserving; curved -> ``f = R/2`` focus) over each fold --
    the documented per-segment pattern, in one call.  No fold -> the single-call
    path (byte-identical).

    ``output_plane_distance`` (v5.21; M-P6 follow-up) composes a **free-space
    leg** of that axial distance (in ``output_plane_n``, air = 1) into the
    canonical entrance->exit map, so the fit lands on a DOWNSTREAM plane (e.g.
    the focus / image plane a back-focal-distance past a prescription that ends
    at the last lens vertex) WITHOUT re-tracing the optics.  Combined with
    ``roi=(cx, cy, half_width)`` this places the ROI directly on that plane --
    an ``O(roi_n^2)`` integrand cost at the focus (measured ~21x vs the full
    grid here, up to ~1e3-1e4x for a tight spot on a large grid) -- and a
    through-focus scan (many ``output_plane_distance`` values) re-uses the single
    ray trace, only re-propagating + refitting (cheap).  The composed field is
    exact: it matches baking the same distance into the prescription's last
    thickness to ~1e-10, and the ROI window is identical to the full-grid slice.
    Not yet combined with ``fold_split``.

    ``use_gpu`` (opt-in) runs the per-pixel integrand on the GPU via
    CuPy -- the same ``use_gpu=True`` / cupy-array entry as
    ``apply_real_lens``.  The cheap ray trace + Chebyshev fit stay on the
    host; only the O(N^2 * n_v2) integrand evaluation moves to the
    device.  Supported for **all three** integrators: ``quadrature``
    (v5.20; the Kronecker-factorized uniform quadrature) and, as of the
    next release, the asymptotic evaluators ``stationary_phase`` /
    ``local_quadrature`` (the per-pixel Newton saddle + Hessian signature
    on an xp-dispatched Chebyshev kernel).  Requires the ``cupy``
    package; returns a CuPy device array (call ``cupy.asnumpy`` to pull it
    back to the host).  GPU results match the CPU integrator to ~1e-6
    (device reduction order, not byte-identical -- like the existing
    numexpr-vs-numpy ULP delta).
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper -- siblings missed by the
    # v4.15.2 closure now share the same first-line guard.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_maslov')

    # v4.13.0 (audit L4a): port the explicit mirror-in-surfaces guard
    # from ``apply_real_lens_traced``.  Pre-fix a hand-built prescription
    # with ``surfaces[i]['is_mirror']=True`` (or ``glass_after='MIRROR'``)
    # would slip past the shared ``_check_no_silent_fold_drop`` (which
    # only inspects ``prescription['elements']``), and the Maslov leg
    # would silently treat the mirror as a refractor with the wrong
    # sign.  Fail loudly with the same mirror-specific message as
    # ``apply_real_lens_traced``.
    # v5.21: fold_split=True auto-handles a folded prescription instead of
    # raising -- split at every fold and alternate this Maslov propagator (each
    # refractive leg, mirror-free -> the normal path) with apply_mirror (each
    # fold), chaining the field.  The apply_mirror focusing phase folds the
    # frame; each leg is evaluated in its own local +z (see the split helper's
    # frame note).  Reduces to the single-call path when the prescription has no
    # fold.
    if fold_split:
        from ..io.prescriptions_transforms import split_prescription_at_mirrors
        _legs = split_prescription_at_mirrors(prescription)
        if len(_legs) > 1:
            from .elements import apply_mirror
            _leg_kw = dict(
                wavelength=wavelength, dx=dx, dy=dy,
                ray_field_samples=ray_field_samples,
                ray_pupil_samples=ray_pupil_samples, poly_order=poly_order,
                n_v2=n_v2, output_subsample=output_subsample,
                extract_linear_phase=extract_linear_phase, chunk_v2=chunk_v2,
                use_numexpr=use_numexpr, integration_method=integration_method,
                stationary_newton_iter=stationary_newton_iter,
                stationary_newton_tol=stationary_newton_tol,
                local_n_samples=local_n_samples,
                local_window_sigma=local_window_sigma,
                collimated_input=collimated_input, input_na=input_na,
                normalize_output=normalize_output, verbose=verbose,
                use_gpu=use_gpu)
            from ..propagators.asm import angular_spectrum_propagate
            E = E_in
            for _leg in _legs:
                if _leg['kind'] == 'refractive':
                    E = apply_real_lens_maslov(
                        E, prescription=_leg['prescription'], **_leg_kw)
                else:
                    # free-space to the mirror, reflect (curved -> f = R/2 focus
                    # phase; flat -> field unchanged), then free-space out.  The
                    # split helper carries these gaps on the mirror leg (they are
                    # NOT in the refractive segments).
                    _m = _leg['element']
                    _din = float(_leg.get('distance_in', 0.0) or 0.0)
                    _dout = float(_leg.get('distance_out', 0.0) or 0.0)
                    if abs(_din) > 0.0:
                        E = angular_spectrum_propagate(E, _din, wavelength, dx)
                    E = apply_mirror(
                        E, wavelength=wavelength, dx=dx, dy=dy,
                        radius=_m.get('radius'), conic=_m.get('conic', 0.0),
                        aperture_diameter=_m.get('clear_aperture'))
                    if abs(_dout) > 0.0:
                        E = angular_spectrum_propagate(E, _dout, wavelength, dx)
            return E

    _surfaces_list = prescription.get('surfaces') or []
    _mirror_surf_idx = []
    for _i, _s in enumerate(_surfaces_list):
        if not isinstance(_s, dict):
            continue
        _gl_after = _s.get('glass_after')
        _is_mirror = bool(_s.get('is_mirror', False)) or (
            isinstance(_gl_after, str)
            and _gl_after.upper() == 'MIRROR'
        )
        if _is_mirror:
            _mirror_surf_idx.append(_i)
    if _mirror_surf_idx:
        raise ValueError(
            f"apply_real_lens_maslov: prescription has "
            f"{len(_mirror_surf_idx)} mirror surface(s) at "
            f"indices {_mirror_surf_idx} -- apply_real_lens_maslov "
            f"only walks refracting surfaces.  Running this "
            f"prescription as-is would silently treat the mirror as "
            f"a refractor (wrong sign / wrong focusing phase) and "
            f"propagate along the unfolded-equivalent axis.  Use "
            f"the per-segment trace + apply_mirror pattern for "
            f"folded designs: call "
            f"lumenairy.io.split_prescription_at_mirrors(rx) to "
            f"split the prescription at each fold, then alternate "
            f"apply_real_lens_maslov (each segment) with "
            f"apply_mirror (each fold).  See Guide-Folded-Designs "
            f"section 'Wave-optics through a fold'.")

    # Folded-design silent-drop guard: same as apply_real_lens.
    from ._lens_real import _check_no_silent_fold_drop
    _check_no_silent_fold_drop(
        prescription, fn_name='apply_real_lens_maslov')

    # Internal references keep the legacy local name to avoid a
    # sprawling rename across the function body.
    lens_prescription = prescription

    # Local references to numexpr (if available) -- the parent module
    # (lenses.py) holds the lazy module slot.
    from . import lenses as _lenses_module
    t0 = time.perf_counter()

    # v5.20 (GPU): CuPy dispatch mirrors apply_real_lens -- opt in via
    # use_gpu=True OR by passing a CuPy input array.  Only the O(N^2 * n_v2)
    # integrand evaluation runs on the device; the cheap ray trace + Chebyshev
    # fit stay on the host, so E_in is normalised to a host copy for that
    # pipeline and a device copy is uploaded for the integrator.
    from ._lens_real import _ensure_cupy_loaded, _is_cupy_array
    _cupy_in = _is_cupy_array(E_in)
    _use_gpu = bool(use_gpu) or _cupy_in
    _cp = None
    if _use_gpu:
        if not _ensure_cupy_loaded():
            raise ImportError(
                "apply_real_lens_maslov: use_gpu=True (or a CuPy input array) "
                "requires the 'cupy' package.  Install cupy-cuda12x (NVIDIA, "
                "matching your CUDA version) or cupy-rocm-6-1 (AMD ROCm); or "
                "call with use_gpu=False for the CPU path.")
        import cupy as _cp
        E_in = _cp.asnumpy(E_in) if _cupy_in else np.asarray(E_in)
    else:
        E_in = np.asarray(E_in)
    if E_in.ndim != 2 or E_in.shape[0] != E_in.shape[1]:
        raise ValueError(
            f"E_in must be square 2D, got shape {E_in.shape}")
    N = E_in.shape[0]

    # v5.20: anamorphic (dy != dx) support.  Resolve dy on the same
    # ``None -> get_default_dy() -> dx`` chain as apply_real_lens, then thread
    # the separate x/y pitches through the input sampler, the output axes, and
    # the angular-content FFT.  The Chebyshev entrance->exit fit already
    # normalises s1x/s1y (and v2x/v2y) on independent axes, and the quadrature
    # integrator already receives separate per-axis Vandermondes (Tx_1d from
    # out_axis_x at dx, Ty_1d from out_axis_y at dy), so anisotropic *physical*
    # spacing needs no integrator change -- only the axes feeding it.  The
    # array itself must still be square N x N (the integrators' output pixel
    # COUNT is a single N_out per axis); a rectangular ARRAY (Ny != Nx) remains
    # apply_real_lens territory and is rejected by the square-2D guard above.
    if dy is None:
        from ..propagators.propagation import get_default_dy
        dy = get_default_dy()
        if dy is None:
            dy = dx
    dy = float(dy)
    _anamorphic = abs(dy - float(dx)) > 1e-12 * max(abs(float(dx)), 1.0)

    # Pre-flight grid vs prescription-aperture check.
    try:
        _warn_if_aperture_exceeds_grid(
            lens_prescription, N, dx, source='apply_real_lens_maslov')
    except (KeyError, ValueError, TypeError, AttributeError):
        # Aperture-check failure is informational only; the
        # propagator still runs.
        pass

    def _progress(phase, frac, note=''):
        dt = time.perf_counter() - t0
        if progress is not None:
            # F3 (audit): emit the suite-standard (stage, fraction,
            # message) signature via call_progress instead of the old
            # bespoke keyword/4-positional protocol, which raised
            # TypeError on a standard (label, frac[, msg]) callback and
            # crashed the propagator mid-lens.  ``phase`` becomes the
            # stage label; the note + elapsed time fold into the message
            # so no information is lost.  call_progress swallows broken-
            # callback exceptions so a progress bar can never crash the run.
            msg = f'{note} ({dt:.1f}s)' if note else f'({dt:.1f}s)'
            call_progress(progress, phase, float(frac), msg)
        if verbose:
            print(f"  maslov {phase:>10s}  {frac*100:5.1f}%  "
                  f"({dt:6.1f}s) {note}", flush=True)

    # -----------------------------------------------------------------
    # Step 1: Trace rays on a Chebyshev-node (h, p) grid
    # -----------------------------------------------------------------
    _progress('trace', 0.0, 'building ray bundle')

    surfaces = rt.surfaces_from_prescription(lens_prescription)
    if not surfaces:
        raise ValueError("Lens prescription has no surfaces.")

    # 4.11.2: warn if a non-entrance or decentered stop is configured.
    # ``apply_real_lens`` honours ``stop_index`` and per-surface
    # ``decenter`` on the stop; the Maslov path traces a Chebyshev-node
    # ray bundle launched on a centred (h, p) grid scaled by the
    # entrance aperture, so a non-zero stop_index is silently moved to
    # the entrance.
    _stop_index = lens_prescription.get('stop_index')
    if _stop_index is not None and int(_stop_index) != 0:
        import warnings
        warnings.warn(
            f"apply_real_lens_maslov: prescription specifies "
            f"stop_index={_stop_index}, but the Maslov ray bundle is "
            "launched on a centred (h, p) Chebyshev grid scaled by the "
            "entrance aperture; the aperture stop is effectively "
            "applied at the entrance (index 0).  For physically-correct "
            "stop behaviour on a non-entrance stop, use apply_real_lens.",
            RuntimeWarning, stacklevel=2,
        )
    else:
        _surfs_chk = lens_prescription.get('surfaces') or []
        if _surfs_chk:
            _stop_surf_idx = int(_stop_index) if _stop_index is not None else 0
            if 0 <= _stop_surf_idx < len(_surfs_chk):
                _dec = _surfs_chk[_stop_surf_idx].get('decenter') or (0.0, 0.0)
                if _dec[0] != 0.0 or _dec[1] != 0.0:
                    import warnings
                    warnings.warn(
                        f"apply_real_lens_maslov: stop surface "
                        f"{_stop_surf_idx} has decenter={_dec}; the "
                        "Maslov ray bundle is launched on a centred "
                        "(h, p) grid and will not see the off-axis stop "
                        "correctly.  Use apply_real_lens for "
                        "decentered-stop systems.",
                        RuntimeWarning, stacklevel=2,
                    )

    aperture_m = lens_prescription.get('aperture_diameter', None)
    if aperture_m is None:
        sds = [s.semi_diameter for s in surfaces if np.isfinite(s.semi_diameter)]
        if sds:
            aperture_m = 2.0 * min(sds)
        else:
            # Circular-aperture fallback: use the smaller grid half-extent so
            # the launched bundle stays inside the (possibly anamorphic) grid
            # (min(dx, dy) == dx for the square-pixel case, so unchanged there).
            aperture_m = N * min(float(dx), float(dy)) * 0.5
    r_aperture = 0.5 * aperture_m

    def cheb_nodes(n):
        i = np.arange(n)
        return np.cos((i + 0.5) * np.pi / n)

    hx = cheb_nodes(ray_field_samples)
    hy = cheb_nodes(ray_field_samples)
    px = cheb_nodes(ray_pupil_samples)
    py = cheb_nodes(ray_pupil_samples)

    HX, HY, PX, PY = np.meshgrid(hx, hy, px, py, indexing='ij')
    HX = HX.ravel()
    HY = HY.ravel()
    PX = PX.ravel()
    PY = PY.ravel()

    keep = (PX**2 + PY**2) <= 1.0
    HX, HY, PX, PY = HX[keep], HY[keep], PX[keep], PY[keep]
    n_rays = len(HX)
    if n_rays < 1.5 * _count_multi_indices_4d(poly_order):
        raise ValueError(
            f"Only {n_rays} rays survived pupil masking; need at least "
            f"~{int(1.5 * _count_multi_indices_4d(poly_order))} "
            f"for a well-conditioned order-{poly_order} fit.")

    s1x = HX * r_aperture
    s1y = HY * r_aperture

    # N3 (audit): the pupil-direction chart must span BOTH the lens
    # acceptance NA and the INPUT field's angular content.  Sizing from
    # the lens EFL alone (the pre-fix na_proxy) drops any divergent /
    # tilted input source off the traced ray chart, so its wide-angle
    # rays are extrapolated or clip at |u_v2| = 1 -- silently dim / wrong
    # output at ANY resolution.  Split the sizing into a lens term and an
    # input term.
    if collimated_input:
        na_lens = 1e-5
    else:
        try:
            _M, _efl, _bfl, _ffl = rt.system_abcd_prescription(
                lens_prescription, wavelength)
            efl_abs = float(abs(_efl))
            if np.isfinite(efl_abs) and efl_abs > 0:
                na_lens = r_aperture / max(efl_abs, r_aperture * 10)
            else:
                lens_total_thickness = sum(s.thickness for s in surfaces)
                na_lens = r_aperture / max(lens_total_thickness,
                                           r_aperture * 10)
        except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                np.linalg.LinAlgError, IndexError, TypeError):
            # system_abcd_prescription failure -- fall back to a
            # thickness-based NA proxy (geometric heuristic).
            lens_total_thickness = sum(s.thickness for s in surfaces)
            na_lens = r_aperture / max(lens_total_thickness,
                                       r_aperture * 10)

    # Divergence NA of the input field: measured from the second moment
    # of its angular spectrum (a single FFT; direction cosine v =
    # wavelength * fx in the paraxial regime), unless the caller supplies
    # input_na explicitly (or the field is declared collimated).
    _na_meas = 0.0
    if not collimated_input:
        _F = np.fft.fft2(E_in)
        _P = np.abs(_F) ** 2
        del _F
        _fx = np.fft.fftfreq(N, d=dx)
        _fy = np.fft.fftfreq(N, d=dy)   # v5.20: anamorphic y-axis pitch
        _FX, _FY = np.meshgrid(_fx, _fy, indexing='xy')
        _Ptot = float(_P.sum())
        if _Ptot > 0.0:
            _v2 = (wavelength ** 2) * (_FX ** 2 + _FY ** 2)
            _rms = float(np.sqrt(float((_v2 * _P).sum()) / _Ptot))
            _na_meas = 3.0 * _rms   # ~3-sigma coverage of the spectrum
            del _v2
        del _P, _FX, _FY, _fx, _fy
    if input_na is not None:
        na_input = float(input_na)
        # Explicit input_na must be a finite, non-negative direction cosine.
        # A NaN slips past the na_proxy>=1 clamp below (NaN comparisons are
        # False), reaching the trace as N_dir=NaN and dying with a
        # misleading "0 rays survived" TIR message (adversarial review) --
        # fail fast here with the real cause instead.
        if not (np.isfinite(na_input) and na_input >= 0.0):
            raise ValueError(
                f"apply_real_lens_maslov: input_na must be a finite, "
                f"non-negative number (an input-side NA / direction cosine); "
                f"got {input_na!r}.  Omit input_na to auto-size the pupil "
                f"chart from the field's angular spectrum.")
        # Coverage guard: warn if the caller under-specified input_na
        # relative to the measured angular spread (the field will clip).
        if (not collimated_input) and na_input < 0.7 * _na_meas:
            import warnings
            warnings.warn(
                f"apply_real_lens_maslov: input_na={na_input:.4f} is well "
                f"below the measured input angular spread "
                f"(~{_na_meas:.4f}); the pupil chart may not cover the "
                f"field and wide-angle content will be lost.  Omit "
                f"input_na to auto-size from the field.",
                RuntimeWarning, stacklevel=2)
    elif collimated_input:
        na_input = 0.0
    else:
        na_input = _na_meas

    # Chart spans the lens acceptance plus the input divergence.
    na_proxy = na_lens + na_input

    # Clamp to a physical direction cosine (< 1).  A speckled / hard-aperture
    # input can have a 3-sigma angular estimate na_input > 1 (measured
    # 1.3-4.1 on white-noise fields, adversarial review P2); leaving
    # na_proxy > 1 forces every pupil ray to v1x^2+v1y^2 > 1, so
    # N_dir = sqrt(max(1 - v1x^2 - v1y^2, 0)) = 0 and the whole chart is
    # grazing -> the wide-angle content it was meant to capture is dropped.
    # Cap just below unity and tell the caller the estimate is being
    # trusted only up to the horizon.  Use ``not (na_proxy < 1.0)`` rather
    # than ``na_proxy >= 1.0`` so a non-finite proxy (e.g. an inf leaking in
    # from na_lens) is also caught -- NaN would already have been rejected
    # for explicit input_na above, but this keeps N_dir strictly real.
    if not (na_proxy < 1.0):
        import warnings
        warnings.warn(
            f"apply_real_lens_maslov: NA proxy {na_proxy:.3f} (lens "
            f"{na_lens:.3f} + input {na_input:.3f}) exceeds 1; the input "
            f"angular-spread estimate is likely inflated by high-frequency / "
            f"aperture-edge content.  Clamping the pupil chart to NA=0.999 "
            f"(the physical horizon).  Pass input_na explicitly to size the "
            f"chart deliberately.",
            RuntimeWarning, stacklevel=2)
        na_proxy = 0.999

    if verbose:
        print(f"  NA_proxy = {na_proxy:.5f}  (lens {na_lens:.5f} + "
              f"input {na_input:.5f}; collimated_input={collimated_input})")

    v1x = PX * na_proxy
    v1y = PY * na_proxy
    N_dir = np.sqrt(np.maximum(1.0 - v1x**2 - v1y**2, 0.0))
    _progress('trace', 0.05, f'{n_rays} rays prepared')

    rays = rt.RayBundle(
        x=s1x.copy(), y=s1y.copy(), z=np.zeros_like(s1x),
        L=v1x.copy(), M=v1y.copy(), N=N_dir,
        wavelength=wavelength,
        alive=np.ones(n_rays, dtype=bool),
        opd=np.zeros(n_rays),
    )

    tr = rt.trace(rays, surfaces, wavelength)
    exit_rays = tr.image_rays
    alive = exit_rays.alive
    if alive.sum() < 1.5 * _count_multi_indices_4d(poly_order):
        raise ValueError(
            f"Only {alive.sum()}/{n_rays} rays survived the trace; "
            f"likely aperture / TIR issue.  Check prescription.")

    # #2 (M-P6 follow-up): compose a free-space leg into the canonical map so
    # the fit maps entrance coords to a DOWNSTREAM (e.g. focus / image) plane a
    # distance ``output_plane_distance`` past the prescription's exit vertex,
    # WITHOUT re-tracing the optics.  Each exit ray advances by that axial gap
    # (direction unchanged in free space; OPL += n * geometric-path); the fit +
    # ROI machinery is then unchanged but lands on the requested plane -- so a
    # tiny ``roi`` window at the focus costs O(roi_n^2) integrand evals, and a
    # through-focus scan re-uses the single ray trace (only re-propagate + refit,
    # which is cheap vs re-tracing).  ``output_plane_n`` is the index of that
    # output space (air = 1).
    ex_x = exit_rays.x
    ex_y = exit_rays.y
    ex_opd = exit_rays.opd
    if output_plane_distance:
        _Nz = exit_rays.N
        _t = output_plane_distance / np.where(np.abs(_Nz) > 1e-30, _Nz, 1e-30)
        ex_x = ex_x + _t * exit_rays.L
        ex_y = ex_y + _t * exit_rays.M
        ex_opd = ex_opd + float(output_plane_n) * _t

    s2x = ex_x[alive]
    s2y = ex_y[alive]
    v2x = exit_rays.L[alive]
    v2y = exit_rays.M[alive]
    opd_m = ex_opd[alive] - rays.opd[alive]
    opd_w = opd_m / wavelength
    s1x_live = s1x[alive]
    s1y_live = s1y[alive]
    _progress('trace', 0.15, f'{alive.sum()} alive rays; '
              f'OPD p-v = {opd_w.max()-opd_w.min():.3f} waves')

    # -----------------------------------------------------------------
    # Step 2: Normalise (s2, v2) to [-1, 1]^4 and fit Chebyshev polys
    # -----------------------------------------------------------------
    _progress('fit', 0.15, 'normalising inputs')
    s2x_c, s2x_h = _fit_normaliser(s2x)
    s2y_c, s2y_h = _fit_normaliser(s2y)
    v2x_c, v2x_h = _fit_normaliser(v2x)
    v2y_c, v2y_h = _fit_normaliser(v2y)

    u_s2x = (s2x - s2x_c) / s2x_h
    u_s2y = (s2y - s2y_c) / s2y_h
    u_v2x = (v2x - v2x_c) / v2x_h
    u_v2y = (v2y - v2y_c) / v2y_h

    linear_coeffs = None
    if extract_linear_phase:
        X5 = np.column_stack([
            np.ones_like(u_s2x),
            u_s2x, u_s2y, u_v2x, u_v2y,
        ])
        linear_coeffs, *_ = np.linalg.lstsq(X5, opd_w, rcond=None)
        opd_linear = X5 @ linear_coeffs
        opd_residual = opd_w - opd_linear
    else:
        opd_residual = opd_w.copy()

    # N4 (audit): the fitted linear OPD term was subtracted for fit
    # conditioning but never re-applied -- silently dropping output tilt
    # and shifting the stationary point for decentered / tilted / off-axis
    # systems (benign piston for a centered lens).  Re-apply it EXACTLY by
    # splitting it: the s2 part (c0 + c1*u_s2x + c2*u_s2y) is constant in
    # the pupil-momentum integration variable v2, so it factors out of the
    # canonical integral and is re-applied as an output post-multiply after
    # dispatch; the v2 part (c3*u_v2x + c4*u_v2y) lives inside the integral
    # (it shifts the stationary point) and is threaded into every
    # integrator's OPD + saddle-point gradient.  linear_coeffs are in WAVES
    # (same units as opd), so they add directly with no scaling.
    if linear_coeffs is None:
        linear_coeffs = np.zeros(5, dtype=np.float64)
    _lin = np.asarray(linear_coeffs, dtype=np.float64)
    _lin_v3 = float(_lin[3])
    _lin_v4 = float(_lin[4])

    mi = _multi_indices_total_degree(4, poly_order)
    M = len(mi)
    _progress('fit', 0.25, f'building design matrix ({n_rays} x {M})')
    T1 = _chebyshev_vandermonde(u_s2x, poly_order)
    T2 = _chebyshev_vandermonde(u_s2y, poly_order)
    T3 = _chebyshev_vandermonde(u_v2x, poly_order)
    T4 = _chebyshev_vandermonde(u_v2y, poly_order)
    A = np.empty((len(u_s2x), M), dtype=np.float64)
    for j, (k1, k2, k3, k4) in enumerate(mi):
        A[:, j] = T1[k1] * T2[k2] * T3[k3] * T4[k4]

    # Perf (M-P5-adjacent): the OPD, s1x and s1y fits all share the SAME
    # design matrix A, so solve them with a single stacked right-hand side --
    # one SVD of A instead of three.  ~2.9x cheaper on the fit stage (which
    # dominates the Maslov runtime once M-P4 has accelerated the integrate
    # step).  LAPACK's multi-RHS gelsd path reorders slightly vs three
    # single-RHS solves, so the coefficients differ at ULP (~1e-15 relative,
    # far below complex64 output precision) -- not byte-identical.
    _progress('fit', 0.35, 'solving normal equations for OPD + s1x + s1y (stacked RHS)')
    _coef3 = _solve_fit(
        A, np.column_stack([opd_residual, s1x_live, s1y_live]))
    coef_opd = _coef3[:, 0]
    coef_s1x = _coef3[:, 1]
    coef_s1y = _coef3[:, 2]

    # v5.21 (M-P follow-up): the fit-residual RMS diagnostics are only ever read
    # into the progress/verbose string below -- three A@coef GEMVs + reductions
    # of pure waste on a headless production sweep.  Compute them only when a
    # consumer exists.
    if progress is not None or verbose:
        opd_pred = A @ coef_opd
        s1x_pred = A @ coef_s1x
        s1y_pred = A @ coef_s1y
        res_opd = np.sqrt(np.mean((opd_residual - opd_pred)**2))
        res_s1x = np.sqrt(np.mean((s1x_live - s1x_pred)**2)) * 1e6
        res_s1y = np.sqrt(np.mean((s1y_live - s1y_pred)**2)) * 1e6
        _progress('fit', 0.60,
                  f'RMS OPD residual = {res_opd:.2e} waves; '
                  f's1x RMS = {res_s1x:.2e} um, s1y RMS = {res_s1y:.2e} um')

    # -----------------------------------------------------------------
    # Step 3: Build output grids
    # -----------------------------------------------------------------
    _progress('grid', 0.60, 'setting up output and v2 grids')
    if output_subsample < 1:
        output_subsample = 1
    N_out_coarse = N // output_subsample

    if roi is None:
        _idx = np.arange(N_out_coarse) - N_out_coarse / 2
        out_axis_x = _idx * (dx * output_subsample)
        out_axis_y = _idx * (dy * output_subsample)   # v5.20: anamorphic
        _roi_active = False
    else:
        if _anamorphic:
            # A square physical ROI window at native pitch resolves to
            # different pixel counts in x (roi_hw/dx) and y (roi_hw/dy) --
            # a rectangular output the square integrators don't take.  ROI is
            # square-pixel only for now; the full-grid path is anamorphic.
            raise NotImplementedError(
                "apply_real_lens_maslov: roi= is not yet supported together "
                f"with anamorphic pixels (dx={dx!r} != dy={dy!r}); a square "
                "ROI window maps to a rectangular pixel grid.  Use the full "
                "grid (roi=None) for anamorphic runs, or square pixels for "
                "ROI.")
        # M-P6 (audit perf): evaluate only a region of interest -- a square
        # window of ``roi_n`` pixels at the native ``dx`` spacing centred at
        # physical ``(roi_cx, roi_cy)`` = ``roi[:2]`` with half-width
        # ``roi[2]``.  The integrators evaluate each output pixel
        # independently, so the returned (roi_n, roi_n) field is identical to
        # the ROI slice of the full-grid field, but costs O(roi_n^2) instead
        # of O(N^2) integrand evaluations -- 10^3-10^4x fewer for spot
        # studies.  Full resolution only (output_subsample forced to 1); the
        # power normalisation is skipped (ill-defined on a sub-window -- the
        # ROI captures only part of the output power), so ROI returns the
        # raw field, matching a normalize_output='none' full run.
        roi_cx, roi_cy, roi_hw = float(roi[0]), float(roi[1]), float(roi[2])
        output_subsample = 1
        N_out_coarse = max(1, int(round(2.0 * roi_hw / dx)))
        _ax = (np.arange(N_out_coarse) - N_out_coarse / 2) * dx
        out_axis_x = _ax + roi_cx
        out_axis_y = _ax + roi_cy
        normalize_output = 'none'
        _roi_active = True
    s2x_grid, s2y_grid = np.meshgrid(out_axis_x, out_axis_y, indexing='xy')

    u_s2x_out = (s2x_grid - s2x_c) / s2x_h
    u_s2y_out = (s2y_grid - s2y_c) / s2y_h
    inbox = (np.abs(u_s2x_out) <= 1.0) & (np.abs(u_s2y_out) <= 1.0)

    # v5.21: integration_method='auto' -- resolve to a concrete integrator from
    # the fitted chart's v2-oscillation count.  Uniform 'quadrature' is exact
    # AND caustic-safe (its integrand amplitude |det ds1/dv2| is finite through
    # focus) but costs O(N^2 * n_v2); the asymptotic 'local_quadrature' is
    # 77-386x faster but is singular AT a caustic and only accurate when the
    # integrand is oscillatory (the saddle dominates).  So: use quadrature when
    # it is well-resolved (need n_v2 <= _N_V2_AUTO_MAX -- this covers low-
    # oscillation AND near-caustic charts, which are low-v2-oscillation and thus
    # stay on the safe path), and switch to the fast asymptotic only when
    # quadrature would need MORE samples than the cap (the very oscillatory /
    # high-NA regime where uniform quadrature is both slow and would speckle,
    # exactly where the saddle approximation is the intended tool).
    if integration_method == 'auto':
        _v2m_a = np.array(
            [1.0 if (k[2] > 0 or k[3] > 0) else 0.0 for k in mi],
            dtype=np.float64)
        _osc_a = float(np.sum(np.abs(coef_opd) * _v2m_a))
        _need_a = int(np.ceil(4.0 * _osc_a)) + 1
        integration_method = ('local_quadrature' if _need_a > _N_V2_AUTO_MAX
                              else 'quadrature')
        _progress('integrate', 0.595,
                  f"auto -> {integration_method} (need n_v2~{_need_a})")

    # A1 (v5.20): auto-resolve the uniform-quadrature v2 sampling when the
    # caller left n_v2 unset.  n_v2 drives ONLY integration_method='quadrature'
    # (the local_quadrature / stationary_phase paths window around the per-pixel
    # saddle via the v2-box half-width v2x_h/v2y_h, and never read this uniform
    # sample count), so size it from the same _v2_osc estimate the N2 guard
    # below uses and leave the asymptotic paths at the floor.  Both ``mi`` and
    # ``coef_opd`` are already fitted at this point.
    if n_v2 is None:
        if integration_method == 'quadrature':
            _v2_mask_auto = np.array(
                [1.0 if (k[2] > 0 or k[3] > 0) else 0.0 for k in mi],
                dtype=np.float64)
            _v2_osc_auto = float(np.sum(np.abs(coef_opd) * _v2_mask_auto))
            n_v2 = int(np.clip(int(np.ceil(4.0 * _v2_osc_auto)) + 1,
                               _N_V2_AUTO_MIN, _N_V2_AUTO_MAX))
        else:
            n_v2 = _N_V2_AUTO_MIN

    u_v2x_samples = np.linspace(-1.0, 1.0, n_v2)
    u_v2y_samples = np.linspace(-1.0, 1.0, n_v2)
    du = u_v2x_samples[1] - u_v2x_samples[0]

    def tukey(n, alpha=0.2):
        u = np.linspace(-1, 1, n)
        abs_u = np.abs(u)
        w = np.ones_like(u)
        taper_start = 1.0 - alpha
        tmask = abs_u > taper_start
        w[tmask] = 0.5 * (1 + np.cos(np.pi * (abs_u[tmask] - taper_start) / alpha))
        return w
    # v5.21: both axes use n_v2, so the two Tukey windows are identical --
    # compute once.
    tuk_x = tukey(n_v2)
    tuk_y = tuk_x
    tuk_2d = tuk_x[None, :] * tuk_y[:, None]

    # v5.2.1: ``v2x_samples`` / ``v2y_samples`` were computed but never
    # used -- downstream code reads ``u_v2x_samples`` / ``u_v2y_samples``
    # (the unitless Chebyshev-node coords) instead.  Removed dead assigns.


    # v5.21: the sampler used only in_axis[0] = -(N/2)*pitch, but allocated two
    # length-N arrays every chunk to read that one scalar.  Precompute the
    # scalar origins (matching the GPU twin, which already does this).
    _in0x = -(N / 2) * dx
    _in0y = -(N / 2) * dy   # v5.20: anamorphic y pitch

    def sample_E_bilinear(s1x_q: np.ndarray, s1y_q: np.ndarray) -> np.ndarray:
        fx = (s1x_q - _in0x) / dx
        fy = (s1y_q - _in0y) / dy
        ix = np.floor(fx).astype(np.int64)
        iy = np.floor(fy).astype(np.int64)
        wx = fx - ix
        wy = fy - iy
        ok = (ix >= 0) & (ix < N - 1) & (iy >= 0) & (iy < N - 1)
        ix_c = np.clip(ix, 0, N - 2)
        iy_c = np.clip(iy, 0, N - 2)
        e00 = E_in[iy_c, ix_c]
        e10 = E_in[iy_c, ix_c + 1]
        e01 = E_in[iy_c + 1, ix_c]
        e11 = E_in[iy_c + 1, ix_c + 1]
        val = ((1 - wx) * (1 - wy) * e00
               + wx * (1 - wy) * e10
               + (1 - wx) * wy * e01
               + wx * wy * e11)
        # v4.14.1 (audit P2-6): dtype-aware out-of-bounds sentinel so
        # a complex64 E_in stays complex64 through the bilinear sample
        # (was silently upcasting via the ``0.0 + 0.0j`` complex128
        # literal).  Matches the v4.13.2 canonical pattern.
        val = np.where(ok, val, np.zeros((), dtype=val.dtype))
        return val

    # -----------------------------------------------------------------
    # Step 4: Integrate
    # -----------------------------------------------------------------
    if integration_method not in ('quadrature', 'stationary_phase',
                                    'local_quadrature'):
        raise ValueError(
            f"integration_method must be one of 'quadrature', "
            f"'stationary_phase', 'local_quadrature', "
            f"got {integration_method!r}")

    _progress('integrate', 0.60,
              f'method={integration_method}')

    K1_arr = np.array([k[0] for k in mi], dtype=np.int64)
    K2_arr = np.array([k[1] for k in mi], dtype=np.int64)
    K3_arr = np.array([k[2] for k in mi], dtype=np.int64)
    K4_arr = np.array([k[3] for k in mi], dtype=np.int64)

    inbox_flat = inbox.ravel()

    # Upload the input field to the device once for whichever GPU integrator
    # runs below (the coarse result is pulled back to the host for the
    # numpy post-processing, then re-uploaded at return).
    if _use_gpu:
        _E_in_gpu = _cp.asarray(E_in)

    if integration_method == 'stationary_phase':
        if _use_gpu:
            E_out_coarse = _cp.asnumpy(_integrate_stationary_phase_cupy(
                _cp, coef_opd, coef_s1x, coef_s1y,
                K1_arr, K2_arr, K3_arr, K4_arr,
                poly_order, N_out_coarse,
                u_s2x_out, u_s2y_out, inbox_flat,
                v2x_h, v2y_h,
                _E_in_gpu, N, dx, dy,
                stationary_newton_iter, stationary_newton_tol,
                out_dtype=E_in.dtype, lin_v3=_lin_v3, lin_v4=_lin_v4,
            ))
        else:
            E_out_coarse = _integrate_stationary_phase(
                coef_opd, coef_s1x, coef_s1y, mi,
                K1_arr, K2_arr, K3_arr, K4_arr,
                poly_order, N_out_coarse,
                u_s2x_out, u_s2y_out, inbox_flat,
                v2x_c, v2y_c, v2x_h, v2y_h,
                sample_E_bilinear,
                stationary_newton_iter, stationary_newton_tol,
                _progress, verbose,
                out_dtype=E_in.dtype,
                lin_v3=_lin_v3, lin_v4=_lin_v4,
            )
    elif integration_method == 'local_quadrature':
        if _use_gpu:
            E_out_coarse = _cp.asnumpy(_integrate_local_quadrature_cupy(
                _cp, coef_opd, coef_s1x, coef_s1y,
                K1_arr, K2_arr, K3_arr, K4_arr,
                poly_order, N_out_coarse,
                u_s2x_out, u_s2y_out, inbox_flat,
                v2x_h, v2y_h,
                _E_in_gpu, N, dx, dy,
                stationary_newton_iter, stationary_newton_tol,
                local_n_samples, local_window_sigma,
                out_dtype=E_in.dtype, lin_v3=_lin_v3, lin_v4=_lin_v4,
            ))
        else:
            E_out_coarse = _integrate_local_quadrature(
                coef_opd, coef_s1x, coef_s1y, mi,
                K1_arr, K2_arr, K3_arr, K4_arr,
                poly_order, N_out_coarse,
                u_s2x_out, u_s2y_out, inbox_flat,
                v2x_c, v2y_c, v2x_h, v2y_h,
                sample_E_bilinear,
                stationary_newton_iter, stationary_newton_tol,
                local_n_samples, local_window_sigma,
                _progress, verbose,
                out_dtype=E_in.dtype,
                lin_v3=_lin_v3, lin_v4=_lin_v4,
            )
    else:
        # N2 (audit): estimate the v2 oscillation count of the integrand
        # phase 2*pi*OPD(s2,v2) from the fitted coefficients.  Chebyshev
        # polynomials are bounded by 1 on [-1, 1], so the sum of
        # |coef_opd| over v2-dependent terms (k3>0 or k4>0) upper-bounds
        # the OPD excursion in WAVES = cycles along v2.  Uniform n_v2-point
        # quadrature needs a few samples per cycle; when under-resolved the
        # result speckles regardless of grid/memory (no output-resolution
        # fix helps) -- warn and point at the asymptotic evaluators, which
        # are the correct choice at production NA.
        _v2_mask = np.array(
            [1.0 if (k[2] > 0 or k[3] > 0) else 0.0 for k in mi],
            dtype=np.float64)
        _v2_osc = float(np.sum(np.abs(coef_opd) * _v2_mask))
        if n_v2 < 4.0 * _v2_osc:
            import warnings
            warnings.warn(
                f"apply_real_lens_maslov: integration_method='quadrature' "
                f"with n_v2={n_v2} is under-resolved for this chart "
                f"(~{_v2_osc:.0f} v2 oscillations; want n_v2 >~ "
                f"{int(4 * _v2_osc)}).  Uniform quadrature will speckle "
                f"regardless of output resolution or memory.  Increase "
                f"n_v2, or use integration_method='local_quadrature' / "
                f"'stationary_phase' (the correct evaluators at "
                f"production NA).",
                RuntimeWarning, stacklevel=2)
        # N1 + F2 (audit): the (N_out^2, M) Chebyshev design matrix G is used
        # ONLY by the quadrature integrator (its G @ H GEMMs).  The
        # stationary_phase / local_quadrature integrators evaluate the
        # Chebyshev basis per pixel-chunk and never touch it (N1: don't build
        # it for them).  For the quadrature path itself, materialising the
        # whole G forced a 451 GB allocation at N=16384 / output_subsample=1
        # (F2); instead we pass only the two cheap per-axis Vandermondes
        # ((poly_order+1, N_out) each) and let the integrator build a
        # per-output-row-band G on the fly.
        _progress('integrate', 0.61,
                  f'precomputing (s2)-axis basis on {N_out_coarse} points')
        Tx_1d = _chebyshev_vandermonde(
            (out_axis_x - s2x_c) / s2x_h, poly_order)
        Ty_1d = _chebyshev_vandermonde(
            (out_axis_y - s2y_c) / s2y_h, poly_order)
        if _use_gpu:
            # GPU quadrature.  Run the O(N^2 * n_v2) integrand on the device
            # (E_in already uploaded to _E_in_gpu above), pull the coarse field
            # back to the host for the (numpy) upsample / linear-phase /
            # normalize steps; the final return re-uploads to a device array
            # (apply_real_lens convention) when use_gpu / a CuPy input was given.
            E_out_coarse = _cp.asnumpy(_integrate_quadrature_cupy(
                _cp,
                coef_opd, coef_s1x, coef_s1y, mi,
                K3_arr, K4_arr,
                poly_order, Tx_1d, Ty_1d, N_out_coarse,
                u_v2x_samples, u_v2y_samples, tuk_2d, du,
                v2x_h, v2y_h, chunk_v2, inbox_flat,
                _E_in_gpu, N, dx, dy,
                _progress,
                out_dtype=E_in.dtype,
                lin_v3=_lin_v3, lin_v4=_lin_v4,
            ))
        else:
            E_out_coarse = _integrate_quadrature(
                coef_opd, coef_s1x, coef_s1y, mi,
                K1_arr, K2_arr, K3_arr, K4_arr,
                poly_order, Tx_1d, Ty_1d, N_out_coarse,
                u_v2x_samples, u_v2y_samples, tuk_2d, du,
                v2x_h, v2y_h, chunk_v2, inbox_flat,
                sample_E_bilinear,
                use_numexpr, _progress,
                _lenses_module,
                out_dtype=E_in.dtype,
                lin_v3=_lin_v3, lin_v4=_lin_v4,
            )

    # -----------------------------------------------------------------
    # Step 5: Upsample to the full grid if output_subsample > 1
    # -----------------------------------------------------------------
    if output_subsample > 1:
        _progress('upsample', 0.95,
                  f'interpolating {N_out_coarse}^2 -> {N}^2 (cubic)')
        from scipy.ndimage import zoom
        zoom_factor = float(N) / float(N_out_coarse)
        amp = np.abs(E_out_coarse)
        amp_z = zoom(amp, zoom_factor, order=3, mode='nearest')
        # Phase upsampling: pre-3.5.6 used line-by-line np.unwrap then
        # cubic zoom of the unwrapped phase.  Line-by-line unwrap is
        # fragile near caustics / focal saddles where the phase wraps
        # along both axes; the resulting cubic-interpolated phase had
        # ~4% RMS errors from line-mismatched seams.
        #
        # 3.5.6 fix: interpolate the COMPLEX exp(i*phase) directly via
        # cubic zoom of its real and imaginary parts, then take
        # ``angle()``.  This avoids any 2-D phase-unwrap step
        # (and therefore any unwrap-induced seams) at the cost of
        # only being well-behaved when the local phase variation
        # between adjacent coarse pixels is < pi -- which is the same
        # condition the original line-unwrap silently relied on.
        # For Maslov outputs that satisfy that bound (typical
        # refractive systems with output_subsample <= 8), the new
        # path agrees with the OLD output to ~0.3% RMS while
        # eliminating the caustic-seam artifact.
        phase_c = np.angle(E_out_coarse)
        cos_z = zoom(np.cos(phase_c), zoom_factor, order=3, mode='nearest')
        sin_z = zoom(np.sin(phase_c), zoom_factor, order=3, mode='nearest')
        E_out_re = amp_z * cos_z
        E_out_im = amp_z * sin_z

        def _fit(a):
            if a.shape == (N, N):
                return a
            out = np.zeros((N, N), dtype=a.dtype)
            rows = min(a.shape[0], N)
            cols = min(a.shape[1], N)
            out[:rows, :cols] = a[:rows, :cols]
            return out
        # v4.14.0: ``1j * float64`` returns complex128; cast back to
        # E_in.dtype so complex64 inputs are preserved through the
        # final re-fit step.
        E_out = (_fit(E_out_re) + 1j * _fit(E_out_im)).astype(E_in.dtype)
    else:
        E_out = E_out_coarse

    # N4 (audit) re-apply the s2 part of the fitted linear OPD that was
    # subtracted before fitting.  Split by cost + Nyquist-safety:
    #
    #  * Piston (_lin[0]) is a GLOBAL phase -> apply as a scalar.  It is
    #    grid-invariant, so this avoids building an N x N temporary just to
    #    add a constant (the piston is ~10^3 waves and is the ONLY term
    #    that is ever appreciable here -- see below).
    #  * The s2-slope terms (_lin[1], _lin[2]) are ~0 for a rotationally-
    #    symmetric prescription (the OPL is then even in output position:
    #    measured |_lin[1]| ~ 1e-10 for a symmetric singlet, < 0.04 waves
    #    for a 0.04 rad tilted input; literal decenter/tilt dict keys are
    #    dropped by the centred trace).  But a FREEFORM surface
    #    (xy_polynomial / zernike odd terms are honored by the trace) makes
    #    them genuinely large -- a wedge/prism deviates the beam, giving a
    #    real output-position OPL slope of up to ~10^4 waves (adversarial
    #    review; verified prism |_lin[1]| = 15.6 waves >> coarse Nyquist,
    #    still subsample-invariant here).  So this branch is load-bearing,
    #    not defensive: the slope MUST be applied on the FINE (post-upsample)
    #    grid, because a slope above the coarse Nyquist (c1 > N_out_coarse/4)
    #    aliases / flips under the cubic phase-zoom if applied on the coarse
    #    field first.  The fine-pixel coordinate is reproduced by zooming the
    #    coarse output axis with the SAME zoom call, so the tilt lands
    #    exactly where the zoomed content lives (convention-independent;
    #    avoids the grid_mode=False edge-stretch of a nominal fine axis).
    #    The abs()>1e-6 gate skips the N x N coordinate build for the common
    #    symmetric case (where the slope is a negligible ~1e-10 waves and
    #    the meshgrid would otherwise cost ~17 GB at N=32768).
    #
    # NB when a large real output tilt ALSO has an in-integral (pupil, v2)
    # component -- e.g. a strongly-powered freeform lens rather than a flat
    # wedge -- that component lives INSIDE the canonical integral (via the
    # _lin_v3/_v4 terms) and is coarse-resolved, so it aliases for output
    # tilts above the coarse Nyquist regardless of where this post-multiply
    # runs.  That is the N2 under-resolution regime (warned separately);
    # reduce output_subsample.
    if _lin[0]:
        E_out = (E_out * np.exp(2j * np.pi * _lin[0])).astype(E_in.dtype)
    if abs(_lin[1]) > 1e-6 or abs(_lin[2]) > 1e-6:
        if output_subsample > 1:
            # subsample>1 is non-ROI.  v5.20: zoom BOTH axes independently --
            # for anamorphic pixels out_axis_x (dx) != out_axis_y (dy), so the
            # old ``out_axis_fy = out_axis_fx`` would place the y-tilt at the
            # wrong pitch.  (Square pixels: the two zooms are identical.)
            from scipy.ndimage import zoom as _zoom1d

            def _zoom_axis(_ax):
                _f = _zoom1d(_ax, float(N) / float(N_out_coarse),
                             order=1, mode='nearest')
                if _f.shape[0] != N:  # non-divisible safety (matches _fit)
                    _tmp = np.zeros(N, dtype=_f.dtype)
                    _n = min(_f.shape[0], N)
                    _tmp[:_n] = _f[:_n]
                    _f = _tmp
                return _f

            out_axis_fx = _zoom_axis(out_axis_x)
            out_axis_fy = _zoom_axis(out_axis_y)
        else:
            out_axis_fx = out_axis_x       # coarse grid == fine grid (ROI-safe)
            out_axis_fy = out_axis_y
        _s2x_f, _s2y_f = np.meshgrid(out_axis_fx, out_axis_fy, indexing='xy')
        _u_s2x_f = (_s2x_f - s2x_c) / s2x_h
        _u_s2y_f = (_s2y_f - s2y_c) / s2y_h
        E_out = (E_out * np.exp(
            2j * np.pi * (_lin[1] * _u_s2x_f
                          + _lin[2] * _u_s2y_f))).astype(E_in.dtype)
        del _s2x_f, _s2y_f, _u_s2x_f, _u_s2y_f

    # -----------------------------------------------------------------
    # Step 6: Absolute-amplitude normalization.
    # -----------------------------------------------------------------
    if normalize_output == 'power':
        p_in = float((np.abs(E_in)**2).sum())
        p_out = float((np.abs(E_out)**2).sum())
        if p_out > 0 and p_in > 0:
            scale = np.sqrt(p_in / p_out)
            E_out = E_out * scale
    elif normalize_output == 'peak':
        a_in = float(np.abs(E_in).max())
        a_out = float(np.abs(E_out).max())
        if a_out > 0 and a_in > 0:
            E_out = E_out * (a_in / a_out)
    elif normalize_output == 'none':
        pass
    elif isinstance(normalize_output, (int, float, complex)):
        E_out = E_out * normalize_output
    else:
        raise ValueError(f"normalize_output={normalize_output!r}; "
                          f"expected 'power', 'peak', 'none', or scalar")

    # v4.14.0: final dtype cast back to E_in.dtype.  The normalization
    # multiplies above promote complex64 -> complex128 because the
    # scalar scale factor is a python float (float64).  Cast once at
    # the end to preserve the input-dtype contract.
    if E_out.dtype != E_in.dtype:
        E_out = E_out.astype(E_in.dtype)

    _progress('done', 1.0,
              f'total {time.perf_counter()-t0:.1f}s')
    if _use_gpu:
        # Match apply_real_lens: use_gpu / CuPy-input -> return a device array
        # (the host-side post-processing above is O(N^2), cheap vs the
        # integration).  Call cupy.asnumpy on the result to pull it to host.
        return _cp.asarray(E_out)
    return E_out


def apply_real_lens_maslov_vector(
    E_vec: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    **maslov_kwargs: Any,
) -> np.ndarray:
    """Vector (Jones) Maslov lens propagator: caustic-safe phase-space
    propagation *with* polarization ray tracing.

    Applies the per-pixel transverse Jones matrix of the prescription's
    base-ray Fresnel s/p (transmission -- reusing the GBD polarization ray
    tracing, :func:`lumenairy.propagators.gbd._fresnel_jones_matrix_per_beamlet`,
    incl. per-surface refraction ``t_s`` / ``t_p`` and multilayer coatings) to
    the input ``(E_x, E_y)`` Jones field, then propagates each mixed component
    through the lens with the scalar :func:`apply_real_lens_maslov` (which keeps
    the field finite through a caustic).  This closes the "Maslov is scalar-only"
    gap: polarization-resolved study through a focus, which GBD's paraxial
    beamlets cannot do with caustic fidelity.

    Parameters
    ----------
    E_vec : array ``(2, Ny, Nx)`` complex -- the ``(E_x, E_y)`` Jones field.
    prescription, wavelength, dx, dy : as :func:`apply_real_lens_maslov`.
    **maslov_kwargs : forwarded to :func:`apply_real_lens_maslov` for each
        component (``integration_method``, ``poly_order``, ``use_gpu``, ...).

    Returns
    -------
    array ``(2, Ny, Nx)`` complex -- the output ``(E_x, E_y)`` Jones field.

    Notes
    -----
    Transmission Jones only (the base-ray Fresnel is applied at the input plane
    then the scalar envelope is propagated), matching the GBD vector convention;
    reflection at fold mirrors is handled by ``apply_mirror`` / ``fold_split``.
    """
    from ..propagators.gbd import _fresnel_jones_matrix_per_beamlet

    E_vec = np.asarray(E_vec)
    if E_vec.ndim != 3 or E_vec.shape[0] != 2:
        raise ValueError(
            "apply_real_lens_maslov_vector: E_vec must be (2, Ny, Nx) (the "
            f"E_x, E_y Jones components); got shape {E_vec.shape}.")
    Ny, Nx = E_vec.shape[-2], E_vec.shape[-1]
    if dy is None:
        dy = dx
    ix = np.arange(Nx)
    iy = np.arange(Ny)
    Ix, Iy = np.meshgrid(ix, iy, indexing='xy')
    xb = (Ix.ravel() - Nx / 2.0) * dx
    yb = (Iy.ravel() - Ny / 2.0) * float(dy)
    zc = np.zeros_like(xb)
    P, _alive = _fresnel_jones_matrix_per_beamlet(
        xb, yb, zc, zc, prescription, wavelength)
    ExS = np.asarray(E_vec[0]).ravel()
    EyS = np.asarray(E_vec[1]).ravel()
    ExM = (P[:, 0, 0] * ExS + P[:, 0, 1] * EyS).reshape(Ny, Nx)
    EyM = (P[:, 1, 0] * ExS + P[:, 1, 1] * EyS).reshape(Ny, Nx)
    out_x = apply_real_lens_maslov(
        ExM, prescription=prescription, wavelength=wavelength, dx=dx, dy=dy,
        **maslov_kwargs)
    out_y = apply_real_lens_maslov(
        EyM, prescription=prescription, wavelength=wavelength, dx=dx, dy=dy,
        **maslov_kwargs)
    from ..backend.array import array_namespace
    xp = array_namespace(out_x)
    return xp.stack([out_x, out_y], axis=0)


def _count_multi_indices_4d(max_order: int) -> int:
    """Number of 4-variable multi-indices with total degree <= max_order
    (== C(n+4, 4) for n = max_order)."""
    from math import comb
    return comb(max_order + 4, 4)


# ---------------------------------------------------------------------------
# Integration method helpers
# ---------------------------------------------------------------------------

def _integrate_quadrature(
    coef_opd, coef_s1x, coef_s1y, mi,
    K1_arr, K2_arr, K3_arr, K4_arr,
    poly_order, Tx_1d, Ty_1d, N_out_coarse,
    u_v2x_samples, u_v2y_samples, tuk_2d, du,
    v2x_h, v2y_h, chunk_v2, inbox_flat,
    sample_E_bilinear,
    use_numexpr, _progress,
    _lenses_module,
    out_dtype=np.complex128,
    lin_v3=0.0, lin_v4=0.0,
):
    """Uniform Tukey-windowed quadrature on the (v2x, v2y) grid.

    v4.14.0: ``out_dtype`` defaults to ``np.complex128`` for back-
    compat; callers pass ``E_in.dtype`` to preserve complex64 inputs.

    F2 follow-up (audit remediation): the (N_out^2, M) design matrix ``G``
    is the quadrature path's dominant allocation and OOMs at scale (451 GB
    at N=16384, output_subsample=1).  Rather than take a prebuilt ``G``, this
    now takes the two cheap per-axis Chebyshev Vandermondes
    ``Tx_1d``/``Ty_1d`` ((poly_order+1, N_out) each) and builds only a
    ``G_band`` for a band of output ROWS at a time
    (``G[iy*N_out+ix, m] = Ty_1d[k2, iy] * Tx_1d[k1, ix]``), capping peak
    memory to O(rows_per_band * (M + n_v2)).
    """
    n_v2 = len(u_v2x_samples)
    n_v2_total = n_v2 * n_v2
    M = len(mi)
    k1k2 = [(k[0], k[1]) for k in mi]

    Tu3_all  = _chebyshev_vandermonde(u_v2x_samples, poly_order)
    Tu4_all  = _chebyshev_vandermonde(u_v2y_samples, poly_order)
    dTu3_all = _chebyshev_derivative_vandermonde(u_v2x_samples, poly_order)
    dTu4_all = _chebyshev_derivative_vandermonde(u_v2y_samples, poly_order)

    iy_grid, ix_grid = np.meshgrid(np.arange(n_v2), np.arange(n_v2),
                                     indexing='ij')
    v2x_idx = ix_grid.ravel()
    v2y_idx = iy_grid.ravel()

    T3bj  = Tu3_all [K3_arr[:, None], v2x_idx[None, :]]
    T4bj  = Tu4_all [K4_arr[:, None], v2y_idx[None, :]]
    dT3bj = dTu3_all[K3_arr[:, None], v2x_idx[None, :]]
    dT4bj = dTu4_all[K4_arr[:, None], v2y_idx[None, :]]
    T3_T4  = T3bj * T4bj
    dT3_T4 = dT3bj * T4bj
    T3_dT4 = T3bj * dT4bj

    H_opd      = coef_opd[:, None] * T3_T4
    H_s1x      = coef_s1x[:, None] * T3_T4
    H_s1y      = coef_s1y[:, None] * T3_T4
    H_ds1x_du3 = coef_s1x[:, None] * dT3_T4
    H_ds1x_du4 = coef_s1x[:, None] * T3_dT4
    H_ds1y_du3 = coef_s1y[:, None] * dT3_T4
    H_ds1y_du4 = coef_s1y[:, None] * T3_dT4

    P_ord = poly_order + 1
    if _QUAD_FACTORIZE:
        # M-P2: G[(iy,ix),m] = Ty[k2,iy]*Tx[k1,ix] is a Kronecker product, so
        # (G @ H)[(iy,ix),j] = sum_{k1,k2} Ty[k2,iy] Tx[k1,ix] Hh[k1,k2,j]
        # where Hh scatter-sums H's rows by their (k1,k2) pair.  The (P,P,.)
        # tensors are tiny; the per-band contraction is one einsum, no G.
        _S = np.zeros((P_ord * P_ord, M), dtype=np.float64)
        for _m, (_k1, _k2) in enumerate(k1k2):
            _S[_k1 * P_ord + _k2, _m] = 1.0

        def _hat(H):
            return (_S @ H).reshape(P_ord, P_ord, H.shape[1])

        Hh_opd = _hat(H_opd)
        Hh_s1x = _hat(H_s1x)
        Hh_s1y = _hat(H_s1y)
        Hh_ds1x_du3 = _hat(H_ds1x_du3)
        Hh_ds1x_du4 = _hat(H_ds1x_du4)
        Hh_ds1y_du3 = _hat(H_ds1y_du3)
        Hh_ds1y_du4 = _hat(H_ds1y_du4)

        def _factor_contract(Hh, Tyb, cs, ce, bw):
            # result[R,i,j] = sum_{a,b} Ty[b,R] Hh[a,b,j] Tx[a,i]
            return np.einsum('bR,abj,ai->Rij', Tyb, Hh[:, :, cs:ce], Tx_1d,
                             optimize=True).reshape(bw * N_out_coarse, ce - cs)

    weight_per_sample = tuk_2d.ravel() * du * du * (v2x_h * v2y_h)

    # N4: linear-in-v2 OPD term (c3*u_v2x + c4*u_v2y), one value per v2
    # sample; added to the residual-fit opd_c in the chunk loop below.
    lin_v = (lin_v3 * u_v2x_samples[v2x_idx]
             + lin_v4 * u_v2y_samples[v2y_idx])

    if use_numexpr is None:
        use_numexpr = NUMEXPR_AVAILABLE
    use_numexpr = (bool(use_numexpr) and NUMEXPR_AVAILABLE
                    and _ensure_numexpr_loaded())
    _progress('integrate', 0.65,
              f'quadrature: {n_v2_total} v2 samples, chunk={chunk_v2}, '
              f'numexpr={use_numexpr}')

    if chunk_v2 <= 0:
        chunk_v2 = n_v2_total
    chunk_v2 = min(chunk_v2, n_v2_total)

    # Output-row band: full rows only, so band pixels = rows_per_band * N_out
    # align to the (iy, ix) row-major layout.  Auto-size to keep the G-band
    # plus the (band_px, chunk_v2) working set bounded (~budget bytes).
    if _QUAD_ROW_BAND:
        rows_per_band = max(1, int(_QUAD_ROW_BAND))
    else:
        _budget_px = 4_000_000  # ~ band_px cap -> G_band ~ band_px*M*8 bytes
        rows_per_band = max(1, min(N_out_coarse, _budget_px // max(1, N_out_coarse)))

    E_out_flat = np.zeros(N_out_coarse * N_out_coarse, dtype=out_dtype)
    t_int_start = time.perf_counter()

    _ne = _lenses_module._ne if use_numexpr else None
    n_bands = (N_out_coarse + rows_per_band - 1) // rows_per_band

    for iy0 in range(0, N_out_coarse, rows_per_band):
        iy1 = min(iy0 + rows_per_band, N_out_coarse)
        p0 = iy0 * N_out_coarse
        p1 = iy1 * N_out_coarse
        inbox_b = inbox_flat[p0:p1]
        _bw = iy1 - iy0

        if _QUAD_FACTORIZE:
            _Tyb = Ty_1d[:, iy0:iy1]          # (P, band_rows); no G materialized
        else:
            # Explicit per-row-band design matrix (validation reference):
            # G_band[(iy-iy0)*N_out + ix, m] = Ty_1d[k2, iy] * Tx_1d[k1, ix].
            G_band = np.empty((p1 - p0, M), dtype=np.float64)
            for m_, (k1, k2) in enumerate(k1k2):
                G_band[:, m_] = np.outer(Ty_1d[k2, iy0:iy1], Tx_1d[k1]).ravel()

        acc = np.zeros(p1 - p0, dtype=out_dtype)
        for c_start in range(0, n_v2_total, chunk_v2):
            c_end = min(c_start + chunk_v2, n_v2_total)

            if _QUAD_FACTORIZE:
                opd_c      = _factor_contract(Hh_opd, _Tyb, c_start, c_end, _bw)
                s1x_c      = _factor_contract(Hh_s1x, _Tyb, c_start, c_end, _bw)
                s1y_c      = _factor_contract(Hh_s1y, _Tyb, c_start, c_end, _bw)
                ds1x_du3_c = _factor_contract(Hh_ds1x_du3, _Tyb, c_start, c_end, _bw)
                ds1x_du4_c = _factor_contract(Hh_ds1x_du4, _Tyb, c_start, c_end, _bw)
                ds1y_du3_c = _factor_contract(Hh_ds1y_du3, _Tyb, c_start, c_end, _bw)
                ds1y_du4_c = _factor_contract(Hh_ds1y_du4, _Tyb, c_start, c_end, _bw)
            else:
                opd_c      = G_band @ H_opd     [:, c_start:c_end]
                s1x_c      = G_band @ H_s1x     [:, c_start:c_end]
                s1y_c      = G_band @ H_s1y     [:, c_start:c_end]
                ds1x_du3_c = G_band @ H_ds1x_du3[:, c_start:c_end]
                ds1x_du4_c = G_band @ H_ds1x_du4[:, c_start:c_end]
                ds1y_du3_c = G_band @ H_ds1y_du3[:, c_start:c_end]
                ds1y_du4_c = G_band @ H_ds1y_du4[:, c_start:c_end]
            opd_c      = opd_c + lin_v[None, c_start:c_end]

            det_J_c = (ds1x_du3_c * ds1y_du4_c
                       - ds1x_du4_c * ds1y_du3_c)
            abs_J_c = np.abs(det_J_c) / (v2x_h * v2y_h)

            Eobj_c = sample_E_bilinear(s1x_c, s1y_c)
            weights_c = weight_per_sample[c_start:c_end]

            if use_numexpr:
                # v5.2.1: numexpr's ``evaluate(expr)`` reads variable names
                # from the caller's stack frame via introspection, which
                # makes ``twopi`` / ``cos_term`` / etc. invisible to static
                # analysis (ruff F841).  Pass an explicit ``local_dict=``
                # so the locals appear in the surrounding code's AST.
                # Matches the canonical pattern at ``_lens_real.py:882``.
                twopi = 2.0 * np.pi
                cos_term = _ne.evaluate(
                    "cos(twopi * opd_c)",
                    local_dict={'twopi': twopi, 'opd_c': opd_c})
                sin_term = _ne.evaluate(
                    "sin(twopi * opd_c)",
                    local_dict={'twopi': twopi, 'opd_c': opd_c})
                Er = Eobj_c.real
                Ei = Eobj_c.imag
                contrib_r = _ne.evaluate(
                    "(Er*cos_term - Ei*sin_term) * abs_J_c * weights_c",
                    local_dict={'Er': Er, 'Ei': Ei,
                                'cos_term': cos_term, 'sin_term': sin_term,
                                'abs_J_c': abs_J_c, 'weights_c': weights_c})
                contrib_i = _ne.evaluate(
                    "(Ei*cos_term + Er*sin_term) * abs_J_c * weights_c",
                    local_dict={'Er': Er, 'Ei': Ei,
                                'cos_term': cos_term, 'sin_term': sin_term,
                                'abs_J_c': abs_J_c, 'weights_c': weights_c})
                contrib_sum = contrib_r.sum(axis=1) + 1j * contrib_i.sum(axis=1)
            else:
                contrib_c = (Eobj_c
                              * np.exp(2j * np.pi * opd_c)
                              * abs_J_c
                              * weights_c)
                contrib_sum = contrib_c.sum(axis=1)

            acc += contrib_sum

        # Write only in-box pixels (reads acc only where inbox, so any
        # out-of-box garbage from sample_E_bilinear never propagates).
        rel_inbox = np.nonzero(inbox_b)[0]
        E_out_flat[p0 + rel_inbox] = acc[rel_inbox]

        if n_bands > 1:
            _progress('integrate', 0.65 + 0.30 * (iy1 / N_out_coarse),
                      f'quadrature output-row band {iy1}/{N_out_coarse}')

    t_int = time.perf_counter() - t_int_start
    _progress('integrate', 0.95,
              f'quadrature: {n_v2_total} v2 samples, {n_bands} row band(s), '
              f'in {t_int:.1f}s '
              f'({"numexpr" if use_numexpr else "numpy"}, '
              f'chunk={chunk_v2})')

    return E_out_flat.reshape(N_out_coarse, N_out_coarse)


def _integrate_quadrature_cupy(
    cp,
    coef_opd, coef_s1x, coef_s1y, mi,
    K3_arr, K4_arr,
    poly_order, Tx_1d, Ty_1d, N_out_coarse,
    u_v2x_samples, u_v2y_samples, tuk_2d, du,
    v2x_h, v2y_h, chunk_v2, inbox_flat,
    E_in_gpu, N, dx, dy,
    _progress,
    out_dtype=np.complex128,
    lin_v3=0.0, lin_v4=0.0,
):
    """CuPy GPU twin of the factorized :func:`_integrate_quadrature`.

    Same phase-space quadrature math as the CPU factorized (non-numexpr)
    path -- ``E_out(s2) = sum_v2 E_in(s1(s2,v2)) exp(2 pi i OPD) |det ds1/dv2|
    w(v2)`` -- with the Kronecker ``G = Ty (x) Tx`` factorization
    (``G @ H`` = one einsum per row band, no ``G`` materialized) evaluated on
    the device.  Only the O(N^2 * n_v2) integrand touches the GPU; the trace,
    fit, and coefficient arrays arrive from the host.  Output-row banding is
    retained for device-memory safety; numexpr and the CPU byte-budget
    heuristics are dropped (the GPU reduces directly).  Validated against the
    CPU integrator to ~1e-6 (device BLAS/reduction order -> not byte-identical).
    """
    xp = cp
    n_v2 = len(u_v2x_samples)
    n_v2_total = n_v2 * n_v2
    M = len(mi)
    k1k2 = [(int(k[0]), int(k[1])) for k in mi]
    P_ord = poly_order + 1

    # Small per-v2-axis Vandermondes: build on host, move to device.
    Tu3 = xp.asarray(_chebyshev_vandermonde(u_v2x_samples, poly_order))
    Tu4 = xp.asarray(_chebyshev_vandermonde(u_v2y_samples, poly_order))
    dTu3 = xp.asarray(_chebyshev_derivative_vandermonde(u_v2x_samples, poly_order))
    dTu4 = xp.asarray(_chebyshev_derivative_vandermonde(u_v2y_samples, poly_order))
    Tx_g = xp.asarray(Tx_1d)
    Ty_g = xp.asarray(Ty_1d)

    iy_grid, ix_grid = xp.meshgrid(xp.arange(n_v2), xp.arange(n_v2),
                                   indexing='ij')
    v2x_idx = ix_grid.ravel()
    v2y_idx = iy_grid.ravel()

    K3g = xp.asarray(K3_arr)
    K4g = xp.asarray(K4_arr)
    T3bj = Tu3[K3g[:, None], v2x_idx[None, :]]
    T4bj = Tu4[K4g[:, None], v2y_idx[None, :]]
    dT3bj = dTu3[K3g[:, None], v2x_idx[None, :]]
    dT4bj = dTu4[K4g[:, None], v2y_idx[None, :]]
    T3_T4 = T3bj * T4bj
    dT3_T4 = dT3bj * T4bj
    T3_dT4 = T3bj * dT4bj

    cop = xp.asarray(coef_opd)[:, None]
    csx = xp.asarray(coef_s1x)[:, None]
    csy = xp.asarray(coef_s1y)[:, None]
    H_opd = cop * T3_T4
    H_s1x = csx * T3_T4
    H_s1y = csy * T3_T4
    H_ds1x_du3 = csx * dT3_T4
    H_ds1x_du4 = csx * T3_dT4
    H_ds1y_du3 = csy * dT3_T4
    H_ds1y_du4 = csy * T3_dT4

    # Scatter H rows by (k1, k2) into a (P, P, .) tensor; factor-contract.
    _S = xp.zeros((P_ord * P_ord, M), dtype=xp.float64)
    for _m, (_k1, _k2) in enumerate(k1k2):
        _S[_k1 * P_ord + _k2, _m] = 1.0

    def _hat(H):
        return (_S @ H).reshape(P_ord, P_ord, H.shape[1])

    Hh_opd = _hat(H_opd)
    Hh_s1x = _hat(H_s1x)
    Hh_s1y = _hat(H_s1y)
    Hh_ds1x_du3 = _hat(H_ds1x_du3)
    Hh_ds1x_du4 = _hat(H_ds1x_du4)
    Hh_ds1y_du3 = _hat(H_ds1y_du3)
    Hh_ds1y_du4 = _hat(H_ds1y_du4)

    def _factor_contract(Hh, Tyb, cs, ce, bw):
        return xp.einsum('bR,abj,ai->Rij', Tyb, Hh[:, :, cs:ce], Tx_g,
                         optimize=True).reshape(bw * N_out_coarse, ce - cs)

    weight_per_sample = xp.asarray(tuk_2d.ravel()) * du * du * (v2x_h * v2y_h)
    u_v2x_g = xp.asarray(u_v2x_samples)
    u_v2y_g = xp.asarray(u_v2y_samples)
    lin_v = (lin_v3 * u_v2x_g[v2x_idx] + lin_v4 * u_v2y_g[v2y_idx])

    inbox_g = xp.asarray(inbox_flat)
    in0x = -(N / 2) * dx        # in_axis_x[0]
    in0y = -(N / 2) * dy        # in_axis_y[0] (anamorphic)

    def _sample(s1x_q, s1y_q):
        fx = (s1x_q - in0x) / dx
        fy = (s1y_q - in0y) / dy
        ix = xp.floor(fx).astype(xp.int64)
        iy = xp.floor(fy).astype(xp.int64)
        wx = fx - ix
        wy = fy - iy
        ok = (ix >= 0) & (ix < N - 1) & (iy >= 0) & (iy < N - 1)
        ixc = xp.clip(ix, 0, N - 2)
        iyc = xp.clip(iy, 0, N - 2)
        e00 = E_in_gpu[iyc, ixc]
        e10 = E_in_gpu[iyc, ixc + 1]
        e01 = E_in_gpu[iyc + 1, ixc]
        e11 = E_in_gpu[iyc + 1, ixc + 1]
        val = ((1 - wx) * (1 - wy) * e00 + wx * (1 - wy) * e10
               + (1 - wx) * wy * e01 + wx * wy * e11)
        return xp.where(ok, val, xp.zeros((), dtype=val.dtype))

    if chunk_v2 <= 0:
        chunk_v2 = n_v2_total
    chunk_v2 = min(chunk_v2, n_v2_total)
    _budget_px = 4_000_000
    rows_per_band = max(1, min(N_out_coarse,
                               _budget_px // max(1, N_out_coarse)))
    E_out_flat = xp.zeros(N_out_coarse * N_out_coarse, dtype=out_dtype)
    n_bands = (N_out_coarse + rows_per_band - 1) // rows_per_band

    for iy0 in range(0, N_out_coarse, rows_per_band):
        iy1 = min(iy0 + rows_per_band, N_out_coarse)
        _bw = iy1 - iy0
        p0 = iy0 * N_out_coarse
        p1 = iy1 * N_out_coarse
        _Tyb = Ty_g[:, iy0:iy1]
        acc = xp.zeros(p1 - p0, dtype=out_dtype)
        for cs in range(0, n_v2_total, chunk_v2):
            ce = min(cs + chunk_v2, n_v2_total)
            opd_c = _factor_contract(Hh_opd, _Tyb, cs, ce, _bw) \
                + lin_v[None, cs:ce]
            s1x_c = _factor_contract(Hh_s1x, _Tyb, cs, ce, _bw)
            s1y_c = _factor_contract(Hh_s1y, _Tyb, cs, ce, _bw)
            d13 = _factor_contract(Hh_ds1x_du3, _Tyb, cs, ce, _bw)
            d14 = _factor_contract(Hh_ds1x_du4, _Tyb, cs, ce, _bw)
            d23 = _factor_contract(Hh_ds1y_du3, _Tyb, cs, ce, _bw)
            d24 = _factor_contract(Hh_ds1y_du4, _Tyb, cs, ce, _bw)
            det_J_c = d13 * d24 - d14 * d23
            abs_J_c = xp.abs(det_J_c) / (v2x_h * v2y_h)
            Eobj_c = _sample(s1x_c, s1y_c)
            contrib = (Eobj_c * xp.exp(2j * xp.pi * opd_c)
                       * abs_J_c * weight_per_sample[cs:ce])
            acc += contrib.sum(axis=1)
        rel = xp.nonzero(inbox_g[p0:p1])[0]
        E_out_flat[p0 + rel] = acc[rel]
        if n_bands > 1:
            _progress('integrate', 0.65 + 0.30 * (iy1 / N_out_coarse),
                      f'quadrature[gpu] output-row band {iy1}/{N_out_coarse}')

    return E_out_flat.reshape(N_out_coarse, N_out_coarse)


def _integrate_stationary_phase(
    coef_opd, coef_s1x, coef_s1y, mi,
    K1_arr, K2_arr, K3_arr, K4_arr,
    poly_order, N_out_coarse,
    u_s2x_out, u_s2y_out, inbox_flat,
    v2x_c, v2y_c, v2x_h, v2y_h,
    sample_E_bilinear,
    newton_iter, newton_tol,
    _progress, verbose,
    out_dtype=np.complex128,
    lin_v3=0.0, lin_v4=0.0,
):
    """Leading-order stationary-phase (Gaussian-moment) evaluation.

    v4.14.0: ``out_dtype`` defaults to ``np.complex128`` for back-
    compat; callers pass ``E_in.dtype`` to preserve complex64 inputs.
    """
    t_int_start = time.perf_counter()
    _progress('integrate', 0.65,
              f'stationary-phase Newton ({newton_iter} max iters)')

    N_px = N_out_coarse * N_out_coarse

    u_s2x_flat = u_s2x_out.ravel()
    u_s2y_flat = u_s2y_out.ravel()

    u_v2x = np.zeros(N_px, dtype=np.float64)
    u_v2y = np.zeros(N_px, dtype=np.float64)

    def _opd_and_derivs(coef, u1, u2, u3, u4):
        # M-P4: dispatch to the Numba kernel (default, ULP-equal) or the
        # NumPy reference; returns (f, df_du3, df_du4, d2f_33, d2f_34, d2f_44).
        return _opd6(coef, K1_arr, K2_arr, K3_arr, K4_arr,
                     u1, u2, u3, u4, poly_order)

    # Deferred follow-up (audit remediation): _opd_and_derivs builds the
    # (M, n_px) Chebyshev basis for ALL its input pixels at once, so a
    # full-resolution call (n_px = N_out_coarse^2) peaks at ~M * n_px *
    # O(10) * 8 bytes -- ~133 GB at N=16384, the OOM the audit flagged for
    # this integrator (unlike local_quadrature it was not pixel-banded).
    # Band it here: the per-pixel work is independent (the only reduction
    # is np.sum over the basis axis WITHIN a pixel), so evaluating in
    # contiguous pixel chunks and concatenating is BYTE-IDENTICAL to the
    # unbanded call while capping peak memory to ~0.5 GB.
    _PX_CHUNK_SP = (int(_SP_PIXEL_CHUNK) if _SP_PIXEL_CHUNK
                    else max(1, 4_000_000 // max(1, len(mi))))

    def _opd_and_derivs_banded(coef, u1, u2, u3, u4):
        n = u1.shape[0]
        if n <= _PX_CHUNK_SP:
            return _opd_and_derivs(coef, u1, u2, u3, u4)
        outs = tuple(np.empty(n, dtype=np.float64) for _ in range(6))
        for s in range(0, n, _PX_CHUNK_SP):
            e = min(s + _PX_CHUNK_SP, n)
            res = _opd_and_derivs(coef, u1[s:e], u2[s:e], u3[s:e], u4[s:e])
            for k in range(6):
                outs[k][s:e] = res[k]
        return outs

    converged_mask = np.zeros(N_px, dtype=bool)
    converged_mask[~inbox_flat] = True

    for it in range(newton_iter):
        if converged_mask.all():
            break
        active = ~converged_mask
        u1 = u_s2x_flat[active]
        u2 = u_s2y_flat[active]
        u3 = u_v2x[active]
        u4 = u_v2y[active]
        _, g3, g4, H33, H34, H44 = _opd_and_derivs_banded(
            coef_opd, u1, u2, u3, u4)
        # N4: the linear-in-v2 OPD term (c3*u_v2x + c4*u_v2y) has constant
        # v2-gradient (c3, c4) and zero Hessian, so it shifts the saddle
        # point but not its curvature.  Add it to the gradient here.
        g3 = g3 + lin_v3
        g4 = g4 + lin_v4
        det_H = H33 * H44 - H34 * H34
        det_safe = np.where(np.abs(det_H) < 1e-30,
                             np.sign(det_H) * 1e-30 + 1e-30, det_H)
        dv3 = -(H44 * g3 - H34 * g4) / det_safe
        dv4 = -(-H34 * g3 + H33 * g4) / det_safe
        step_limit = 0.5
        step_size = np.sqrt(dv3**2 + dv4**2)
        damp = np.where(step_size > step_limit,
                         step_limit / np.maximum(step_size, 1e-30),
                         1.0)
        dv3 *= damp
        dv4 *= damp
        u_v2x_new = u_v2x[active] + dv3
        u_v2y_new = u_v2y[active] + dv4
        u_v2x_new = np.clip(u_v2x_new, -1.0, 1.0)
        u_v2y_new = np.clip(u_v2y_new, -1.0, 1.0)
        u_v2x[active] = u_v2x_new
        u_v2y[active] = u_v2y_new
        grad_mag = np.sqrt(g3**2 + g4**2)
        newly = np.zeros(N_px, dtype=bool)
        newly[active] = grad_mag < newton_tol
        converged_mask |= newly
        if verbose and (it == 0 or it == newton_iter - 1 or
                         it % max(1, newton_iter // 4) == 0):
            n_conv = converged_mask.sum()
            _progress('integrate', 0.65 + 0.15 * it / newton_iter,
                      f'Newton iter {it+1}/{newton_iter}, '
                      f'{n_conv}/{N_px} pixels converged '
                      f'(max grad {grad_mag.max():.2e})')

    _progress('integrate', 0.85, 'evaluating saddle-point formula')

    opd_star, g3, g4, H33, H34, H44 = _opd_and_derivs_banded(
        coef_opd, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)
    # N4: add the linear-in-v2 OPD contribution at the (shifted) saddle.
    opd_star = opd_star + lin_v3 * u_v2x + lin_v4 * u_v2y
    s1x_star, ds1x_du3, ds1x_du4, _, _, _ = _opd_and_derivs_banded(
        coef_s1x, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)
    s1y_star, ds1y_du3, ds1y_du4, _, _, _ = _opd_and_derivs_banded(
        coef_s1y, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)

    det_J_norm = ds1x_du3 * ds1y_du4 - ds1x_du4 * ds1y_du3
    abs_J = np.abs(det_J_norm) / (v2x_h * v2y_h)

    H33_phys = H33 / (v2x_h * v2x_h)
    H34_phys = H34 / (v2x_h * v2y_h)
    H44_phys = H44 / (v2y_h * v2y_h)
    det_H_phys = H33_phys * H44_phys - H34_phys * H34_phys
    trace_H = H33_phys + H44_phys
    sig = np.where(det_H_phys > 0,
                    np.where(trace_H > 0, 2, -2),
                    0)
    amp_sp = 1.0 / np.sqrt(np.maximum(np.abs(det_H_phys), 1e-300))
    phase_sp = np.exp(1j * (np.pi / 4.0) * sig)

    Eobj_star = sample_E_bilinear(s1x_star, s1y_star)

    # v4.14.0: cast to ``out_dtype`` (=E_in.dtype from caller) so a
    # complex64 input doesn't get silently upcast to complex128 by
    # the float64-phase * complex128-exp multiply.
    E_flat = (Eobj_star
              * np.exp(2j * np.pi * opd_star)
              * abs_J
              * amp_sp
              * phase_sp).astype(out_dtype)

    not_conv = ~converged_mask
    if not_conv.any():
        E_flat[not_conv] = 0.0
        if verbose:
            _progress('integrate', 0.92,
                      f'{not_conv.sum()}/{N_px} pixels did not converge, '
                      f'zeroed')

    E_flat[~inbox_flat] = 0.0

    t_int = time.perf_counter() - t_int_start
    _progress('integrate', 0.95,
              f'stationary_phase: {N_px} pixels in {t_int:.1f}s')

    return E_flat.reshape(N_out_coarse, N_out_coarse)


def _integrate_local_quadrature(
    coef_opd, coef_s1x, coef_s1y, mi,
    K1_arr, K2_arr, K3_arr, K4_arr,
    poly_order, N_out_coarse,
    u_s2x_out, u_s2y_out, inbox_flat,
    v2x_c, v2y_c, v2x_h, v2y_h,
    sample_E_bilinear,
    newton_iter, newton_tol,
    n_samples, window_sigma,
    _progress, verbose,
    out_dtype=np.complex128,
    lin_v3=0.0, lin_v4=0.0,
):
    """Hybrid stationary-phase + local quadrature.

    v4.14.0: ``out_dtype`` defaults to ``np.complex128`` for back-
    compat; callers pass ``E_in.dtype`` to preserve complex64 inputs.
    """
    t_int_start = time.perf_counter()
    _progress('integrate', 0.60,
              f'local_quadrature: Newton phase ({newton_iter} max iters)')

    N_px = N_out_coarse * N_out_coarse
    u_s2x_flat = u_s2x_out.ravel()
    u_s2y_flat = u_s2y_out.ravel()

    u_v2x = np.zeros(N_px, dtype=np.float64)
    u_v2y = np.zeros(N_px, dtype=np.float64)

    def _opd_and_derivs(coef, u1, u2, u3, u4):
        # M-P4: dispatch to the Numba kernel (default, ULP-equal) or the
        # NumPy reference; returns (f, df_du3, df_du4, d2f_33, d2f_34, d2f_44).
        return _opd6(coef, K1_arr, K2_arr, K3_arr, K4_arr,
                     u1, u2, u3, u4, poly_order)

    converged = np.zeros(N_px, dtype=bool)
    converged[~inbox_flat] = True
    for it in range(newton_iter):
        if converged.all():
            break
        active = ~converged
        u1 = u_s2x_flat[active]
        u2 = u_s2y_flat[active]
        u3 = u_v2x[active]
        u4 = u_v2y[active]
        _, g3, g4, H33, H34, H44 = _opd_and_derivs(coef_opd, u1, u2, u3, u4)
        # N4: linear-in-v2 OPD term shifts the saddle gradient (c3, c4).
        g3 = g3 + lin_v3
        g4 = g4 + lin_v4
        det_H = H33 * H44 - H34 * H34
        det_safe = np.where(np.abs(det_H) < 1e-30,
                             np.sign(det_H) * 1e-30 + 1e-30, det_H)
        dv3 = -(H44 * g3 - H34 * g4) / det_safe
        dv4 = -(-H34 * g3 + H33 * g4) / det_safe
        step_size = np.sqrt(dv3 ** 2 + dv4 ** 2)
        damp = np.where(step_size > 0.5,
                         0.5 / np.maximum(step_size, 1e-30), 1.0)
        dv3 *= damp
        dv4 *= damp
        u_v2x[active] = np.clip(u_v2x[active] + dv3, -1.0, 1.0)
        u_v2y[active] = np.clip(u_v2y[active] + dv4, -1.0, 1.0)
        grad_mag = np.sqrt(g3 ** 2 + g4 ** 2)
        newly = np.zeros(N_px, dtype=bool)
        newly[active] = grad_mag < newton_tol
        converged |= newly

    _progress('integrate', 0.72, 'computing Hessian eigen-scales')
    _, _, _, H33, H34, H44 = _opd_and_derivs(
        coef_opd, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)
    H33_phys = H33 / (v2x_h ** 2)
    H34_phys = H34 / (v2x_h * v2y_h)
    H44_phys = H44 / (v2y_h ** 2)
    tau = H33_phys + H44_phys
    detH = H33_phys * H44_phys - H34_phys ** 2
    disc = np.maximum(tau ** 2 / 4.0 - detH, 0.0)
    sqrt_disc = np.sqrt(disc)
    lam1 = tau / 2.0 + sqrt_disc
    lam2 = tau / 2.0 - sqrt_disc
    sigma1_phys = 1.0 / np.sqrt(np.maximum(np.abs(lam1), 1e-30) * np.pi)
    sigma2_phys = 1.0 / np.sqrt(np.maximum(np.abs(lam2), 1e-30) * np.pi)
    sigma1_norm = sigma1_phys / v2x_h
    sigma2_norm = sigma2_phys / v2y_h

    _progress('integrate', 0.75,
              f'local uniform sampling: {n_samples}x{n_samples} pts, '
              f'window={window_sigma}sigma')
    lin = np.linspace(-window_sigma, window_sigma, n_samples)
    dxi = lin[1] - lin[0]
    Xlin, Ylin = np.meshgrid(lin, lin, indexing='xy')
    Xlin_flat = Xlin.ravel()
    Ylin_flat = Ylin.ravel()

    u_v2x_samp = (u_v2x[:, None]
                   + (sigma1_norm[:, None]) * Xlin_flat[None, :])
    u_v2y_samp = (u_v2y[:, None]
                   + (sigma2_norm[:, None]) * Ylin_flat[None, :])
    u_v2x_samp = np.clip(u_v2x_samp, -1.0, 1.0)
    u_v2y_samp = np.clip(u_v2y_samp, -1.0, 1.0)

    n_s2 = n_samples * n_samples
    u_s2x_tile = np.broadcast_to(u_s2x_flat[:, None], (N_px, n_s2))
    u_s2y_tile = np.broadcast_to(u_s2y_flat[:, None], (N_px, n_s2))

    _progress('integrate', 0.78,
              f'evaluating integrand on {N_px*n_s2:,} (pixel,sample) pairs')

    E_flat = np.zeros(N_px, dtype=out_dtype)
    w2d_phys = (sigma1_phys * sigma2_phys) * (dxi ** 2)

    PX_CHUNK = max(1, min(N_px, 1024 * 64 // max(1, n_s2 // 16)))
    for p_start in range(0, N_px, PX_CHUNK):
        p_end = min(p_start + PX_CHUNK, N_px)
        u3 = u_v2x_samp[p_start:p_end].ravel()
        u4 = u_v2y_samp[p_start:p_end].ravel()
        u1 = u_s2x_tile[p_start:p_end].ravel()
        u2 = u_s2y_tile[p_start:p_end].ravel()
        # v5.21 (M-P8): one shared-basis value+1st-deriv kernel for the three
        # coef sets (opd value only; s1x/s1y value + du3/du4), skipping the
        # unused second derivatives and the 2x redundant basis rebuild.
        (opd_v, s1x_v, ds1x_du3, ds1x_du4,
         s1y_v, ds1y_du3, ds1y_du4) = _opd_vd3(
            coef_opd, coef_s1x, coef_s1y, K1_arr, K2_arr, K3_arr, K4_arr,
            u1, u2, u3, u4, poly_order)
        # N4: linear-in-v2 OPD contribution at each window sample.
        opd_v = opd_v + lin_v3 * u3 + lin_v4 * u4
        det_J = ds1x_du3 * ds1y_du4 - ds1x_du4 * ds1y_du3
        abs_J = np.abs(det_J) / (v2x_h * v2y_h)

        Eobj_v = sample_E_bilinear(s1x_v, s1y_v)

        contrib = (Eobj_v
                    * np.exp(2j * np.pi * opd_v)
                    * abs_J)
        contrib_r = contrib.reshape(p_end - p_start, n_s2)
        E_flat[p_start:p_end] = contrib_r.sum(axis=1) * \
                                  w2d_phys[p_start:p_end]
        if verbose and (p_start % (PX_CHUNK * 8) == 0):
            _progress('integrate',
                      0.78 + 0.15 * (p_end / N_px),
                      f'pixel chunk {p_end}/{N_px}')

    E_flat[~converged] = 0.0
    E_flat[~inbox_flat] = 0.0

    t_int = time.perf_counter() - t_int_start
    _progress('integrate', 0.95,
              f'local_quadrature: {N_px} pixels, '
              f'{n_s2} samples/pixel, {t_int:.1f}s')

    return E_flat.reshape(N_out_coarse, N_out_coarse)


def _sample_bilinear_xp(xp, E_in_gpu, N, dx, dy, s1x_q, s1y_q):
    """xp bilinear sample of the (device) input field at physical (s1x, s1y).
    Mirrors the CPU ``sample_E_bilinear`` closure exactly (anamorphic dx/dy,
    dtype-preserving out-of-bounds zero) for the GPU asymptotic evaluators."""
    in0x = -(N / 2) * dx        # in_axis_x[0]
    in0y = -(N / 2) * dy        # in_axis_y[0]
    fx = (s1x_q - in0x) / dx
    fy = (s1y_q - in0y) / dy
    ix = xp.floor(fx).astype(xp.int64)
    iy = xp.floor(fy).astype(xp.int64)
    wx = fx - ix
    wy = fy - iy
    ok = (ix >= 0) & (ix < N - 1) & (iy >= 0) & (iy < N - 1)
    ixc = xp.clip(ix, 0, N - 2)
    iyc = xp.clip(iy, 0, N - 2)
    e00 = E_in_gpu[iyc, ixc]
    e10 = E_in_gpu[iyc, ixc + 1]
    e01 = E_in_gpu[iyc + 1, ixc]
    e11 = E_in_gpu[iyc + 1, ixc + 1]
    val = ((1 - wx) * (1 - wy) * e00 + wx * (1 - wy) * e10
           + (1 - wx) * wy * e01 + wx * wy * e11)
    return xp.where(ok, val, xp.zeros((), dtype=val.dtype))


def _integrate_stationary_phase_cupy(
    cp, coef_opd, coef_s1x, coef_s1y,
    K1_arr, K2_arr, K3_arr, K4_arr,
    poly_order, N_out_coarse,
    u_s2x_out, u_s2y_out, inbox_flat,
    v2x_h, v2y_h,
    E_in_gpu, N, dx, dy,
    newton_iter, newton_tol,
    out_dtype=np.complex128, lin_v3=0.0, lin_v4=0.0,
):
    """CuPy GPU twin of :func:`_integrate_stationary_phase`.

    Same leading-order saddle-point (Gaussian-moment) evaluation -- per-pixel
    Newton solve for the v2 stationary point, then the Hessian-signature
    amplitude ``exp(i pi sig / 4) / sqrt|det H|`` -- evaluated on the device via
    the xp-dispatched Chebyshev kernel :func:`_opd6_xp`.  The CPU integrator is
    left untouched.  Validated against it on a NumPy backend (ULP) and on-device
    to ~1e-6 (device reduction order + the SIMD all-pixel Newton loop vs the CPU
    active-subset loop, numerically equivalent).
    """
    xp = cp
    K1 = xp.asarray(K1_arr)
    K2 = xp.asarray(K2_arr)
    K3 = xp.asarray(K3_arr)
    K4 = xp.asarray(K4_arr)
    cop = xp.asarray(coef_opd)
    csx = xp.asarray(coef_s1x)
    csy = xp.asarray(coef_s1y)
    u_s2x = xp.asarray(u_s2x_out.ravel())
    u_s2y = xp.asarray(u_s2y_out.ravel())
    inbox = xp.asarray(inbox_flat)

    def opd6(coef, u1, u2, u3, u4):
        return _opd6_xp(xp, coef, K1, K2, K3, K4, u1, u2, u3, u4, poly_order)

    u_v2x, u_v2y, converged = _maslov_newton_saddle_xp(
        xp, opd6, cop, u_s2x, u_s2y, inbox, newton_iter, newton_tol,
        lin_v3, lin_v4)

    opd_star, g3, g4, H33, H34, H44 = opd6(cop, u_s2x, u_s2y, u_v2x, u_v2y)
    opd_star = opd_star + lin_v3 * u_v2x + lin_v4 * u_v2y
    s1x_star, ds1x_du3, ds1x_du4, _, _, _ = opd6(csx, u_s2x, u_s2y, u_v2x, u_v2y)
    s1y_star, ds1y_du3, ds1y_du4, _, _, _ = opd6(csy, u_s2x, u_s2y, u_v2x, u_v2y)

    det_J_norm = ds1x_du3 * ds1y_du4 - ds1x_du4 * ds1y_du3
    abs_J = xp.abs(det_J_norm) / (v2x_h * v2y_h)

    H33_phys = H33 / (v2x_h * v2x_h)
    H34_phys = H34 / (v2x_h * v2y_h)
    H44_phys = H44 / (v2y_h * v2y_h)
    det_H_phys = H33_phys * H44_phys - H34_phys * H34_phys
    trace_H = H33_phys + H44_phys
    sig = xp.where(det_H_phys > 0,
                   xp.where(trace_H > 0, 2.0, -2.0), 0.0)
    amp_sp = 1.0 / xp.sqrt(xp.maximum(xp.abs(det_H_phys), 1e-300))
    phase_sp = xp.exp(1j * (xp.pi / 4.0) * sig)

    Eobj_star = _sample_bilinear_xp(xp, E_in_gpu, N, dx, dy, s1x_star, s1y_star)
    E_flat = (Eobj_star * xp.exp(2j * xp.pi * opd_star)
              * abs_J * amp_sp * phase_sp).astype(out_dtype)
    # Zero in-box-non-converged (converged incl. out-of-box) then out-of-box.
    E_flat = xp.where(converged, E_flat, xp.asarray(0, dtype=out_dtype))
    E_flat = xp.where(inbox, E_flat, xp.asarray(0, dtype=out_dtype))
    return E_flat.reshape(N_out_coarse, N_out_coarse)


def _integrate_local_quadrature_cupy(
    cp, coef_opd, coef_s1x, coef_s1y,
    K1_arr, K2_arr, K3_arr, K4_arr,
    poly_order, N_out_coarse,
    u_s2x_out, u_s2y_out, inbox_flat,
    v2x_h, v2y_h,
    E_in_gpu, N, dx, dy,
    newton_iter, newton_tol,
    n_samples, window_sigma,
    out_dtype=np.complex128, lin_v3=0.0, lin_v4=0.0,
):
    """CuPy GPU twin of :func:`_integrate_local_quadrature`.

    Same hybrid stationary-phase + local windowed quadrature -- Newton saddle,
    Hessian eigen-scale window (`sigma1`, `sigma2`), then an
    ``n_samples x n_samples`` local grid integrated per pixel -- on the device.
    CPU integrator untouched; validated NumPy-backend ULP and on-device ~1e-6.
    """
    xp = cp
    K1 = xp.asarray(K1_arr)
    K2 = xp.asarray(K2_arr)
    K3 = xp.asarray(K3_arr)
    K4 = xp.asarray(K4_arr)
    cop = xp.asarray(coef_opd)
    csx = xp.asarray(coef_s1x)
    csy = xp.asarray(coef_s1y)
    u_s2x = xp.asarray(u_s2x_out.ravel())
    u_s2y = xp.asarray(u_s2y_out.ravel())
    inbox = xp.asarray(inbox_flat)
    N_px = N_out_coarse * N_out_coarse

    def opd6(coef, u1, u2, u3, u4):
        return _opd6_xp(xp, coef, K1, K2, K3, K4, u1, u2, u3, u4, poly_order)

    u_v2x, u_v2y, converged = _maslov_newton_saddle_xp(
        xp, opd6, cop, u_s2x, u_s2y, inbox, newton_iter, newton_tol,
        lin_v3, lin_v4)

    _, _, _, H33, H34, H44 = opd6(cop, u_s2x, u_s2y, u_v2x, u_v2y)
    H33_phys = H33 / (v2x_h ** 2)
    H34_phys = H34 / (v2x_h * v2y_h)
    H44_phys = H44 / (v2y_h ** 2)
    tau = H33_phys + H44_phys
    detH = H33_phys * H44_phys - H34_phys ** 2
    disc = xp.maximum(tau ** 2 / 4.0 - detH, 0.0)
    sqrt_disc = xp.sqrt(disc)
    lam1 = tau / 2.0 + sqrt_disc
    lam2 = tau / 2.0 - sqrt_disc
    sigma1_phys = 1.0 / xp.sqrt(xp.maximum(xp.abs(lam1), 1e-30) * xp.pi)
    sigma2_phys = 1.0 / xp.sqrt(xp.maximum(xp.abs(lam2), 1e-30) * xp.pi)
    sigma1_norm = sigma1_phys / v2x_h
    sigma2_norm = sigma2_phys / v2y_h

    lin = xp.linspace(-window_sigma, window_sigma, n_samples)
    dxi = lin[1] - lin[0]
    Xlin, Ylin = xp.meshgrid(lin, lin, indexing='xy')
    Xlin_flat = Xlin.ravel()
    Ylin_flat = Ylin.ravel()
    u_v2x_samp = xp.clip(
        u_v2x[:, None] + sigma1_norm[:, None] * Xlin_flat[None, :], -1.0, 1.0)
    u_v2y_samp = xp.clip(
        u_v2y[:, None] + sigma2_norm[:, None] * Ylin_flat[None, :], -1.0, 1.0)
    n_s2 = n_samples * n_samples
    w2d_phys = (sigma1_phys * sigma2_phys) * (dxi ** 2)

    E_flat = xp.zeros(N_px, dtype=out_dtype)
    PX_CHUNK = max(1, min(N_px, 1024 * 64 // max(1, n_s2 // 16)))
    for p0 in range(0, N_px, PX_CHUNK):
        p1 = min(p0 + PX_CHUNK, N_px)
        bw = p1 - p0
        u3 = u_v2x_samp[p0:p1].ravel()
        u4 = u_v2y_samp[p0:p1].ravel()
        u1 = xp.broadcast_to(u_s2x[p0:p1, None], (bw, n_s2)).ravel()
        u2 = xp.broadcast_to(u_s2y[p0:p1, None], (bw, n_s2)).ravel()
        opd_v, _, _, _, _, _ = opd6(cop, u1, u2, u3, u4)
        opd_v = opd_v + lin_v3 * u3 + lin_v4 * u4
        s1x_v, ds1x_du3, ds1x_du4, _, _, _ = opd6(csx, u1, u2, u3, u4)
        s1y_v, ds1y_du3, ds1y_du4, _, _, _ = opd6(csy, u1, u2, u3, u4)
        det_J = ds1x_du3 * ds1y_du4 - ds1x_du4 * ds1y_du3
        abs_J = xp.abs(det_J) / (v2x_h * v2y_h)
        Eobj_v = _sample_bilinear_xp(xp, E_in_gpu, N, dx, dy, s1x_v, s1y_v)
        contrib = (Eobj_v * xp.exp(2j * xp.pi * opd_v) * abs_J)
        E_flat[p0:p1] = contrib.reshape(bw, n_s2).sum(axis=1) * w2d_phys[p0:p1]

    E_flat = xp.where(converged, E_flat, xp.asarray(0, dtype=out_dtype))
    E_flat = xp.where(inbox, E_flat, xp.asarray(0, dtype=out_dtype))
    return E_flat.reshape(N_out_coarse, N_out_coarse)


__all__ = [
    'apply_real_lens_maslov',
    'apply_real_lens_maslov_vector',
]

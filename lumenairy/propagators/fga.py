"""Frozen Gaussian Approximation (FGA) lens propagator -- caustic-accurate
Gaussian-beam summation.

FGA (Lu & Yang, *Commun. Math. Sci.* 9(3):663, 2011, arXiv:1010.1968 -- the
wave-equation transplant of the Herman-Kluk semiclassical propagator, Herman &
Kluk, *Chem. Phys.* 91:27, 1984) is a Gaussian-beam-summation propagator that
is **caustic-accurate**.  Unlike ordinary Gaussian Beamlet Decomposition
(:mod:`lumenairy.propagators.gbd`), which carries an evolving (thawed) complex
curvature and whose paraxial per-beamlet phase mis-renders the interference
structure at foci and fold/cusp caustics, FGA:

* **freezes** each beamlet's width and instead reconstructs the caustic by the
  interference of a dense phase-space ``(q, p)`` swarm (position + momentum);
* weights each beamlet by the **Herman-Kluk prefactor** ``a = sqrt(det Z)`` with
  ``Z = (A + D) + i (k w0^2 C - B / (k w0^2))`` built from the ray-transfer /
  monodromy blocks ``A, B, C, D``.  Because ``C`` (the position->momentum block
  that vanishes at a focus) enters *additively*, ``a`` never blows up -- the
  method is regular at caustics by construction (Baranger et al., *J. Phys. A*
  34:7227, 2001).

Validation (vs the exact angular-spectrum field): reproduces free-space
diffraction to fidelity 0.9998, matches :func:`apply_real_lens_gbd` and the
angular-spectrum oracle through a real lens to 0.997-0.999, and **beats GBD at a
spherical-aberration caustic** on both field fidelity and peak-intensity error
(GBD peak error 0.03-0.34 vs FGA 0.01-0.07).

Accuracy knobs.  The reconstruction is normalized so the ``t=0`` resolution of
identity conserves energy (ratio ~1.0) PROVIDED the momentum swarm covers each
beamlet's own momentum width ~``2/(k*w0)`` -- the auto sampler enforces this
completeness floor (an explicit ``p_max`` below it under-normalizes: a narrow
``p_max`` reproduces the field SHAPE, fidelity ~1, but at reduced power, so use
``normalize_output='power'`` if you must under-sample).  The leading FGA is exact
for the paraxial quadratic Hamiltonian, so free-space propagation conserves
energy to ~1.0.
The frozen width ``w0`` is the FGA convergence parameter: a wider beamlet is more
paraxial and gives a cleaner frame, while caustic *resolution* wants a smaller
``w0`` -- a standard FGA tradeoff.  The momentum half-range ``p_max`` must cover
the field's angular content (auto-set from the prescription NA); ``n_p`` sets the
momentum samples per axis and ``dq_step`` the position-lattice stride.

Energy caveat (near-axial inputs through strong focusing).  When the input is
near-collimated (its FBI/local spectrum concentrates near ``p=0``) AND the system
focuses it strongly, flooring ``p_max`` at the beamlet completeness width
``~3/(k*w0)`` -- far above the field's real angular content -- launches
excess-momentum beamlets that the strong lens sprays into a halo of radius
``~efl*p_max``, corrupting the SHAPE (not just the scale).  The auto sampler now
DETECTS this (angular content far below the completeness floor) and instead sizes
``p_max`` to the field content, restoring the beamlet-completeness energy by
power-normalizing the output automatically (a ``RuntimeWarning`` reports the
choice; H5).  Passing an explicit ``p_max`` (~ the angular content) with
``normalize_output='power'`` reproduces this manually.  This is a
representation-regime effect, not the O(eps) FGA transport error (which is
negligible for lens-like Hamiltonians).  Note ``gbd``/``traced`` remain faster
and equally accurate for a smooth single-valued focusing caustic; reach for FGA
where the interference structure (fold/cusp) actually needs it.

Diverging / expanding inputs.  Leading-order FGA/Herman-Kluk is EXACT for the
(quadratic) free-space Hamiltonian, so a diverging beam's only error is the
phase-space QUADRATURE -- and the historical ~0.93 fidelity cap was a
too-coarse MOMENTUM spacing ``dp`` (the diverging beam's broad phase-space
footprint needs ``dp <~ lambda / beam-extent``), NOT a frozen-approximation
wall (Lasser & Lubich, *Acta Numerica* 29, 2020; Kroninger, Lasser & Vanicek,
*Front. Phys.* 11, 2023).  ``p_max=None`` / ``n_p=None`` (the defaults) now
AUTO-SIZE the swarm to the field -- ``p_max`` to the field's angular content and
``n_p`` to make ``dp`` fine enough -- so a diverging beam reconstructs to
fidelity ~1 (e.g. 0.9995 for a beam diverging at ~0.1 rad, up from ~0.93) at a
modest ``n_p``.  Pass explicit ``p_max``/``n_p`` to override.
``momentum_sampling='adaptive'`` (opt-in; default ``'uniform'`` is byte-identical)
concentrates the ``n_p`` nodes where the field's angular spectrum lives (importance
sampling from the marginal Wigner / local-frequency estimate, with the matching
per-node quadrature weights + auto power-normalization).  NOTE (measured, N6b): it
is a NON-improvement in practice -- the reconstruction integrand is the spectrum
convolved with the beamlet momentum width ``~1/(k w0)``, so it is too broad for
importance sampling to beat uniform (which already converges: e.g. fidelity 1.000
at ``n_p=21`` on a smooth 0.23-NA diverging transport).  It is provided per the
accuracy-niche plan with a documented fidelity-vs-cost curve; reach for it only for
a genuinely SPARSE / multi-peak spectrum.

Requires the optional ``numba`` accelerator (the pure-NumPy swarm sum is
impractically slow); install with ``pip install lumenairy[numba]``.
"""
from __future__ import annotations

import copy as _copy
import math
import threading
import warnings
from typing import Any, Dict, Optional

import numpy as np

from ..backend import array_namespace

_NUMBA = None


def _load_numba():
    """Lazy numba import (mirrors the Maslov accelerator pattern)."""
    global _NUMBA
    if _NUMBA is not None:
        return _NUMBA
    try:
        import numba  # noqa: F401
        _NUMBA = True
    except ImportError:
        _NUMBA = False
    return _NUMBA


def _build_kernels():
    """Compile the windowed Gabor-coefficient and frozen-Gaussian-scatter numba
    kernels on first use.  Returns ``(coeff, scatter, coeff_sep, scatter_sep)`` --
    the ``*_sep`` variants are the faster separable/recurrence forms, numerically
    identical (ULP) to their direct counterparts."""
    from numba import njit, prange

    @njit(cache=True, parallel=True, fastmath=True)
    def _coeff(u0r, u0i, qx, qy, px, py, x0, y0, dx, dyg, w0, k, Ag, nsig,
               cr, ci):
        Ny, Nx = u0r.shape
        Nq = qx.shape[0]
        Np = px.shape[0]
        R = nsig * w0
        inv2w2 = 1.0 / (2.0 * w0 * w0)
        for iq in prange(Nq):
            cxq = qx[iq]
            cyq = qy[iq]
            # Pad the box by one pixel each side so the circular gate
            # (``r2 > R*R``) is the SOLE selector -- a guaranteed superset
            # matching ``_coeff_sep``'s ``jx +- int(R/dx) +- 1`` box.  The clamps
            # and gate discard the padding, so the output is byte-identical for
            # non-integer R/dx and picks up the legitimately-in-window boundary
            # pixel at integer R/dx (the default nsig*w0_factor=15), where int()
            # rounding of the tight bound would otherwise drop it (seam S2-7).
            j0 = int((cxq - R - x0) / dx) - 1
            j1 = int((cxq + R - x0) / dx) + 2
            i0 = int((cyq - R - y0) / dyg) - 1
            i1 = int((cyq + R - y0) / dyg) + 2
            if j0 < 0:
                j0 = 0
            if i0 < 0:
                i0 = 0
            if j1 > Nx:
                j1 = Nx
            if i1 > Ny:
                i1 = Ny
            for ip in range(Np):
                ppx = px[ip]
                ppy = py[ip]
                sr = 0.0
                si = 0.0
                for i in range(i0, i1):
                    dy = (y0 + i * dyg) - cyq
                    for j in range(j0, j1):
                        dxr = (x0 + j * dx) - cxq
                        r2 = dxr * dxr + dy * dy
                        if r2 > R * R:
                            continue
                        mag = Ag * math.exp(-r2 * inv2w2)
                        ph = k * (ppx * dxr + ppy * dy)
                        cph = math.cos(ph)
                        sph = math.sin(ph)
                        ur = u0r[i, j]
                        ui = u0i[i, j]
                        sr += mag * (cph * ur + sph * ui)
                        si += mag * (cph * ui - sph * ur)
                cr[iq, ip] = sr * dx * dyg
                ci[iq, ip] = si * dx * dyg

    @njit(cache=True, parallel=True, fastmath=True)
    def _coeff_sep(u0r, u0i, qx, qy, qix, qiy, x0, y0, dx, dyg, w0, k, Ag, nsig,
                   n_p, moff, loff, gx, gy, Exr, Exi, Eyr, Eyi, cr, ci):
        """Separable (tensor-momentum) Gabor analysis -- numerically identical to
        ``_coeff`` (ULP-level, `fastmath`) at ~``n_p`` x less work.  The momentum
        grid is the tensor product ``pv (x) pv`` (``ip = a*n_p + b``,
        ``px=pv[b]``, ``py=pv[a]``), and both the Gaussian window and the phase
        factor over ``exp(-i k (px dxr + py dy))`` are separable, so the 2-D
        windowed sum factors into an x-transform reused across every ``py``:

            inner_x[l, b] = sum_m gx[m] u0(i,j) exp(-i k pv[b] dxr)      (per row l)
            c[iq, a, b]   = Ag dx dyg sum_l gy[l] exp(-i k pv[a] dy) inner_x[l, b]

        The window box (``i0..i1``, ``j0..j1``) and the CIRCULAR truncation
        (``r2 > R^2``) are computed exactly as in ``_coeff`` (same ``int()``
        bounds, same ``dxr=m*dx`` / ``dy=l*dyg`` offsets that hold because the
        launch lattice is on-grid), so the two kernels select the identical sample
        set.  ``gx/gy`` and the phase tables ``Ex/Ey`` are precomputed once and
        SHARED across every lattice point (``dxr`` depends only on ``m=j-qix``)."""
        Nq = qix.shape[0]
        R = nsig * w0
        R2 = R * R
        Ny, Nx = u0r.shape
        Adxy = Ag * dx * dyg
        for iq in prange(Nq):
            jx = qix[iq]
            jy = qiy[iq]
            cxq = qx[iq]
            cyq = qy[iq]
            iR = int(R / dyg)
            jR = int(R / dx)
            i0 = jy - iR - 1
            i1 = jy + iR + 2
            j0 = jx - jR - 1
            j1 = jx + jR + 2
            if i0 < 0:
                i0 = 0
            if j0 < 0:
                j0 = 0
            if i1 > Ny:
                i1 = Ny
            if j1 > Nx:
                j1 = Nx
            # x-transform per row -> inner_x[row-local, b]
            nrow = i1 - i0
            ixr = np.zeros((nrow, n_p))
            ixi = np.zeros((nrow, n_p))
            for i in range(i0, i1):
                lrow = i - i0
                # circular-window GATE computed exactly as in _coeff (dxr via the
                # grid position minus cxq, not m*dx) so the two kernels select the
                # bit-identical sample set even at r2 == R^2 boundary ties; the
                # m*dx / l*dyg table lookups differ only by ULP in the weights.
                dyg_gate = (y0 + i * dyg) - cyq
                dy2 = dyg_gate * dyg_gate
                for j in range(j0, j1):
                    m = j - jx
                    dxr_gate = (x0 + j * dx) - cxq
                    if dxr_gate * dxr_gate + dy2 > R2:
                        continue
                    g = gx[m + moff]
                    ur = g * u0r[i, j]
                    ui = g * u0i[i, j]
                    mi = m + moff
                    for b in range(n_p):
                        exr = Exr[mi, b]
                        exi = Exi[mi, b]
                        ixr[lrow, b] += ur * exr - ui * exi
                        ixi[lrow, b] += ur * exi + ui * exr
            # y-transform: combine rows over exp(-i k pv[a] dy) gy
            for a in range(n_p):
                for b in range(n_p):
                    sr = 0.0
                    si = 0.0
                    for i in range(i0, i1):
                        lrow = i - i0
                        li = (i - jy) + loff
                        gyl = gy[li]
                        eyr = Eyr[li, a]
                        eyi = Eyi[li, a]
                        xr = ixr[lrow, b]
                        xi = ixi[lrow, b]
                        sr += gyl * (eyr * xr - eyi * xi)
                        si += gyl * (eyr * xi + eyi * xr)
                    ip = a * n_p + b
                    cr[iq, ip] = Adxy * sr
                    ci[iq, ip] = Adxy * si

    @njit(cache=True, parallel=True, fastmath=True)
    def _scatter(Qx, Qy, Px, Py, Wr, Wi, x0, y0, dx, dyg, Ny, Nx,
                 w0, k, Ag, nsig, outr, outi):
        Nb = Qx.shape[0]
        R = nsig * w0
        inv2w2 = 1.0 / (2.0 * w0 * w0)
        for i in prange(Ny):        # rows: each written once -> no scatter race
            yy = y0 + i * dyg
            for b in range(Nb):
                wr = Wr[b]
                wi = Wi[b]
                if wr == 0.0 and wi == 0.0:
                    continue
                dy = yy - Qy[b]
                if dy > R or dy < -R:
                    continue
                ppx = Px[b]
                ppy = Py[b]
                j0 = int((Qx[b] - R - x0) / dx)
                j1 = int((Qx[b] + R - x0) / dx) + 1
                if j0 < 0:
                    j0 = 0
                if j1 > Nx:
                    j1 = Nx
                for j in range(j0, j1):
                    dxr = (x0 + j * dx) - Qx[b]
                    r2 = dxr * dxr + dy * dy
                    if r2 > R * R:
                        continue
                    mag = Ag * math.exp(-r2 * inv2w2)
                    ph = k * (ppx * dxr + ppy * dy)
                    cph = math.cos(ph)
                    sph = math.sin(ph)
                    outr[i, j] += mag * (wr * cph - wi * sph)
                    outi[i, j] += mag * (wr * sph + wi * cph)

    @njit(cache=True, parallel=True, fastmath=True)
    def _scatter_sep(Qx, Qy, Px, Py, Wr, Wi, x0, y0, dx, dyg, Ny, Nx,
                     w0, k, Ag, nsig, outr, outi):
        """Frozen-Gaussian scatter -- numerically identical to ``_scatter``
        (ULP-level, `fastmath`) with the transcendentals HOISTED out of the inner
        ``j`` loop.  Post-transport each beamlet has its own ``(Q, P)`` (no shared
        grid), so the tensor-separable trick can't apply; instead, along a row the
        phase ``exp(+i k (px dxr + py dy))`` advances by a CONSTANT rotation
        ``exp(+i k px dx)`` per step and the Gaussian ``exp(-dxr^2/2w0^2)`` by a
        two-term recurrence (ratio *= ``exp(-dx^2/w0^2)``), so each ``j`` costs a
        few mults instead of a cos+sin+exp.  Same row-parallel structure (each row
        written by one thread -> race-free), same box / circular-``r2`` gate / ``j``
        order / beamlet order as ``_scatter``, so the accumulation matches to ULP.
        The recurrences are seeded once per (row, beamlet) with exact transcendental
        calls, so drift over the ``<~2 nsig w0/dx`` window steps is ~machine-eps."""
        Nb = Qx.shape[0]
        R = nsig * w0
        R2 = R * R
        inv2w2 = 1.0 / (2.0 * w0 * w0)
        cc = math.exp(-2.0 * dx * dx * inv2w2)     # Gaussian ratio-of-ratios (const)
        for i in prange(Ny):        # rows: each written once -> no scatter race
            yy = y0 + i * dyg
            for b in range(Nb):
                wr = Wr[b]
                wi = Wi[b]
                if wr == 0.0 and wi == 0.0:
                    continue
                dy = yy - Qy[b]
                if dy > R or dy < -R:
                    continue
                ppx = Px[b]
                ppy = Py[b]
                qxb = Qx[b]
                j0 = int((qxb - R - x0) / dx)
                j1 = int((qxb + R - x0) / dx) + 1
                if j0 < 0:
                    j0 = 0
                if j1 > Nx:
                    j1 = Nx
                if j1 <= j0:
                    continue
                dy2 = dy * dy
                agy = Ag * math.exp(-dy2 * inv2w2)          # Ag * gy(dy)
                dxr = (x0 + j0 * dx) - qxb                   # dxr at j0
                # phase exp(+i*(k*ppx*dxr + k*ppy*dy)); step rotation exp(+i*k*ppx*dx)
                ph0 = k * (ppx * dxr + ppy * dy)
                cph = math.cos(ph0)
                sph = math.sin(ph0)
                dphx = k * ppx * dx
                cstep = math.cos(dphx)
                sstep = math.sin(dphx)
                gxj = math.exp(-dxr * dxr * inv2w2)          # Gaussian at j0
                ratio = math.exp(-(2.0 * dxr * dx + dx * dx) * inv2w2)
                for j in range(j0, j1):
                    # Gate on a FRESH dxr (bit-identical to _scatter's gate) so
                    # the two kernels select the SAME pixel set even at the
                    # integer-R/dx boundary tie (the default nsig*w0_factor=15);
                    # the recurrence dxr drifts by ~ULP and must not decide the
                    # discrete window membership (seam S2-7, scatter half).
                    dxrf = (x0 + j * dx) - qxb
                    if dxrf * dxrf + dy2 <= R2:
                        mag = agy * gxj
                        outr[i, j] += mag * (wr * cph - wi * sph)
                        outi[i, j] += mag * (wr * sph + wi * cph)
                    # advance the Gaussian + phase recurrences (every step, to
                    # stay phase-locked)
                    gxj *= ratio
                    ratio *= cc
                    ncph = cph * cstep - sph * sstep
                    sph = sph * cstep + cph * sstep
                    cph = ncph

    return _coeff, _scatter, _coeff_sep, _scatter_sep


_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is None:
        if not _load_numba():
            raise ImportError(
                "apply_real_lens_fga requires the optional 'numba' accelerator "
                "(the pure-NumPy phase-space swarm sum is impractically slow). "
                "Install with `pip install lumenairy[numba]`.")
        _KERNELS = _build_kernels()
    return _KERNELS


def _det2(M: np.ndarray) -> np.ndarray:
    return M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]


def _hk_prefactor(M, go, gi, kw2):
    """Herman-Kluk prefactor ``a = sqrt(det Z)`` (continuous branch) for one
    momentum column -- shared by the scalar, vector, and coarse FGA paths.

    ``M`` is the (Nq, 4, 4) ray-transfer monodromy in SLOPE state
    ``[x, y, u, v]``; ``go`` (Nq,) and ``gi`` (scalar) are the output / input
    slope -> direction-cosine Jacobian factors the CALLER computes.  The
    scalar-vs-2x2 Jacobian choice (audit S2-16) stays in the caller: this helper
    only assembles the complex Herman-Kluk matrix
    ``Z = (A + D) + i (kw2 * C - B / kw2)`` from the go/gi-scaled monodromy blocks
    (``kw2 = k * w0**2``) and returns ``a = sqrt(det Z)`` on the branch that is
    continuous through the identity (``Re a >= 0``)."""
    A = M[:, 0:2, 0:2]
    B = M[:, 0:2, 2:4] * gi
    Cc = M[:, 2:4, 0:2] * go[:, None, None]
    D = M[:, 2:4, 2:4] * (go[:, None, None] * gi)
    Z = (A + D) + 1j * (kw2 * Cc - B / kw2)
    a = np.sqrt(_det2(Z))
    a = np.where(a.real < 0, -a, a)
    return a


def _coeff_keep_mask(qbounds, Np, coeff_frac, chunk_peak):
    """Global per-momentum-peak coeff-prune PRE-PASS (multi-lattice-chunk).

    Under Nq (position-lattice) chunking the coefficient-prune decision needs the
    GLOBAL per-momentum peak over ALL q -- a per-chunk peak under-counts and would
    drop real momenta -- so a light analysis-only pre-pass accumulates it.
    Returns the boolean keep-mask over the ``Np`` momenta, or ``None`` when
    pruning is off (``coeff_frac <= 0``) or the lattice is a single chunk (the
    caller then uses the cheaper running-max path).  ``chunk_peak(qs, qe)``
    returns the (Np,) max coefficient magnitude over the q-chunk ``[qs, qe)``;
    the scalar and vector paths differ only in that magnitude (single-channel
    ``|c|`` vs combined ``sqrt(|cx|**2 + |cy|**2)``), which stays in the caller."""
    if not (coeff_frac > 0.0 and len(qbounds) > 1):
        return None
    gpeak = np.zeros(Np)
    for qs, qe in qbounds:
        gpeak = np.maximum(gpeak, chunk_peak(qs, qe))
    return gpeak >= coeff_frac * float(gpeak.max() + 1e-300)


def _swarm_lattice(Ny, Nx, dx, dyg, x0, y0, dq_step, p_max, n_p, nodes=None):
    """Build the phase-space launch swarm.  ``nodes=None`` (default) is the
    UNIFORM momentum grid ``pv = linspace(-p_max, p_max, n_p)`` with the constant
    cell spacing ``dp`` and ``wp=None`` (the caller uses the scalar ``dp**2``
    quadrature measure -- byte-identical to prior releases).  ``nodes=(pv, wv)``
    is a NON-uniform (content-adaptive) 1-D momentum node set with per-node
    trapezoidal quadrature weights ``wv``; the returned ``wp = outer(wv, wv)`` is
    the per-momentum 2-D quadrature cell area the caller MUST fold into the
    beamlet weight in place of the uniform ``dp**2`` (adversarial: a non-uniform
    p-measure requires the matching per-momentum weights or the reconstruction
    mis-normalizes)."""
    qi = np.arange(0, Nx, dq_step)
    qj = np.arange(0, Ny, dq_step)
    qxg, qyg = np.meshgrid(x0 + qi * dx, y0 + qj * dyg)
    qx = qxg.ravel().astype(np.float64)
    qy = qyg.ravel().astype(np.float64)
    # momentum (direction-cosine) grid is ISOTROPIC: p is the physical transverse
    # direction cosine, bounded by the system NA, not by the (possibly
    # anamorphic) pixel pitch.  The separable analysis / recurrence-scatter
    # kernels are spacing-AGNOSTIC (they read pv via precomputed phase tables), so
    # a non-uniform pv drops in with no kernel change.
    if nodes is None:
        pv = np.linspace(-p_max, p_max, n_p)
        wp = None
    else:
        pv, wv = nodes
        pv = np.asarray(pv, dtype=np.float64)
        wp = np.outer(wv, wv).ravel().astype(np.float64)   # per-momentum cell area
    pxg, pyg = np.meshgrid(pv, pv)
    px = pxg.ravel().astype(np.float64)
    py = pyg.ravel().astype(np.float64)
    dp = (pv[1] - pv[0]) if (nodes is None and n_p > 1) else 2.0 * p_max
    return qx, qy, px, py, dp, wp


def _adaptive_momentum_nodes(E_in, dx, dyg, wavelength, w0, p_max, n_p,
                             beta=0.7):
    """Content-adaptive (importance-sampled) 1-D momentum nodes on
    ``[-p_max, p_max]`` for a SEPARABLE ``pv (x) pv`` swarm, concentrated where the
    input field's angular spectrum lives, with midpoint-cell quadrature weights.

    The reconstruction integrates the Gabor coefficient ``c(q, p)`` -- the field's
    LOCAL spectrum BROADENED by each frozen beamlet's own momentum resolution
    ``sigma_p ~ 1/(k w0)`` -- against the beamlet kernel over momentum.  The
    importance density is therefore the (symmetrized) marginal angular power
    CONVOLVED with that beamlet momentum Gaussian (so the beamlet-completeness
    skirt is not starved), and nodes are placed by its inverse-CDF -- putting
    samples where the integrand has mass, improving the phase-space quadrature at
    fixed ``n_p`` (the ``n_p``-capped fidelity lever, S2-15).  ``beta`` blends the
    importance CDF with the uniform CDF (``beta=1`` pure importance, ``0``
    uniform) so the node map is STRICTLY monotone (no duplicate nodes) and always
    spans the full ``[-p_max, p_max]`` (the completeness-floor coverage is
    preserved; only the density is redistributed).  Isotropic: x / y marginals are
    averaged so a single ``pv`` serves both tensor axes (the separable kernel
    needs one pv)."""
    from scipy.ndimage import gaussian_filter1d
    E = np.asarray(E_in)
    ny, nx = E.shape
    P = np.abs(np.fft.fft2(E)) ** 2
    fxp = np.fft.fftfreq(nx, dx) * wavelength          # p_x direction cosine
    fyp = np.fft.fftfreq(ny, dyg) * wavelength
    pg = np.linspace(-p_max, p_max, 4096)
    ox = np.argsort(fxp)
    oy = np.argsort(fyp)
    Sx = np.interp(pg, fxp[ox], P.sum(axis=0)[ox], left=0.0, right=0.0)
    Sy = np.interp(pg, fyp[oy], P.sum(axis=1)[oy], left=0.0, right=0.0)
    dens = 0.5 * (Sx + Sy)                             # isotropic marginal density
    # broaden by the beamlet momentum resolution sigma_p ~ 1/(k w0) so the
    # importance density matches the actual reconstruction integrand |c(q, p)|
    # (field content convolved with the frozen beamlet), keeping the completeness
    # skirt sampled rather than starved.
    k = 2.0 * np.pi / wavelength
    sig_samp = (1.0 / (k * w0)) / ((pg[-1] - pg[0]) / (pg.size - 1))
    if sig_samp >= 0.5:
        dens = gaussian_filter1d(dens, sig_samp, mode="constant")
    tot = float(dens.sum())
    if not np.isfinite(tot) or tot <= 0.0:
        pv = np.linspace(-p_max, p_max, n_p)           # no content -> uniform
    else:
        cdf_imp = np.cumsum(dens)
        cdf_imp /= cdf_imp[-1]
        cdf_uni = (pg - pg[0]) / (pg[-1] - pg[0])
        cdf = beta * cdf_imp + (1.0 - beta) * cdf_uni  # strictly monotone
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
        pv = np.interp(np.linspace(0.0, 1.0, n_p), cdf, pg)
        pv[0], pv[-1] = -p_max, p_max                  # pin endpoints (coverage)
    return pv, _midpoint_cell_weights(pv)


def _midpoint_cell_weights(pv):
    """MIDPOINT-cell quadrature weights for 1-D nodes ``pv``: each node owns the
    interval to its neighbours' midpoints, the two endpoint cells extending half
    a spacing beyond.  For a UNIFORM grid this returns the constant spacing ``dp``
    for EVERY node (endpoints included) -- exactly the rectangle-rule ``dp`` the
    scalar-``C`` uniform path uses -- so ``outer(wv, wv) == dp**2`` and the
    adaptive plumbing reduces to the byte-identical uniform normalization; for a
    non-uniform grid it is the consistent generalization."""
    pv = np.asarray(pv, dtype=np.float64)
    n = pv.size
    if n == 1:
        return np.array([1.0])
    mids = 0.5 * (pv[:-1] + pv[1:])                    # n-1 interior boundaries
    left = pv[0] - 0.5 * (pv[1] - pv[0])
    right = pv[-1] + 0.5 * (pv[-1] - pv[-2])
    bounds = np.concatenate(([left], mids, [right]))
    return np.diff(bounds)


def _lattice_support_mask(u0, dx, dyg, dq_step, w0, nsig, frac):
    """Boolean keep-mask over the position lattice (in the raveled ``qx``/``qy``
    order): drop lattice points whose windowed ``|u0|`` is below ``frac`` of the
    peak.  Provably ~no-loss: by Cauchy-Schwarz the Gabor coefficient obeys
    ``|c(q, p)| <= ||g|| * ||u0 restricted to the window||`` for ALL ``p``, so a
    dropped beamlet contributes below ``frac`` to the reconstruction -- set
    ``frac`` under the FGA error floor.  Biggest win for concentrated fields."""
    from scipy.ndimage import uniform_filter
    Ny, Nx = u0.shape
    a2 = np.abs(u0) ** 2
    wx = max(1, int(round(2.0 * nsig * w0 / dx)))
    wy = max(1, int(round(2.0 * nsig * w0 / dyg)))
    amp = np.sqrt(np.maximum(uniform_filter(a2, size=(wy, wx),
                                            mode='constant'), 0.0))
    sub = amp[np.ix_(np.arange(0, Ny, dq_step), np.arange(0, Nx, dq_step))]
    return (sub > frac * float(amp.max() + 1e-300)).ravel()


def _gabor_coeff(u0, qx, qy, px, py, x0, y0, dx, dyg, w0, k, Ag, nsig):
    coeff = _kernels()[0]
    cr = np.zeros((qx.shape[0], px.shape[0]))
    ci = np.zeros((qx.shape[0], px.shape[0]))
    coeff(np.ascontiguousarray(u0.real), np.ascontiguousarray(u0.imag),
          qx, qy, px, py, x0, y0, dx, dyg, w0, k, Ag, nsig, cr, ci)
    return cr + 1j * ci


def _gabor_coeff_sep(u0, qx, qy, pv, x0, y0, dx, dyg, w0, k, Ag, nsig):
    """Separable Gabor analysis over the FULL tensor momentum grid ``pv (x) pv``
    (``ip = a*n_p + b``).  Numerically identical to :func:`_gabor_coeff` on that
    grid (ULP-level) at ~``n_p`` x less work.  Precomputes the shared Gaussian /
    phase tables (``dxr`` depends only on ``m = j - qix``, so they are the same
    for every launch lattice point) and dispatches the separable kernel."""
    coeff_sep = _kernels()[2]
    n_p = pv.shape[0]
    Np = n_p * n_p
    Nq = qx.shape[0]
    R = nsig * w0
    inv2w2 = 1.0 / (2.0 * w0 * w0)
    moff = int(R / dx) + 1
    loff = int(R / dyg) + 1
    marr = np.arange(-moff, moff + 1) * dx           # dxr for each m
    larr = np.arange(-loff, loff + 1) * dyg          # dy for each l
    gx = np.exp(-(marr ** 2) * inv2w2)
    gy = np.exp(-(larr ** 2) * inv2w2)
    phx = k * np.outer(marr, pv)                     # (2moff+1, n_p): k*dxr*pv[b]
    phy = k * np.outer(larr, pv)
    Exr = np.cos(phx)            # exp(-i k pv dxr): real / imag
    Exi = -np.sin(phx)
    Eyr = np.cos(phy)
    Eyi = -np.sin(phy)
    # integer lattice indices: the launch lattice is on-grid (cxq = x0 + qix*dx)
    qix = np.rint((qx - x0) / dx).astype(np.int64)
    qiy = np.rint((qy - y0) / dyg).astype(np.int64)
    cr = np.zeros((Nq, Np))
    ci = np.zeros((Nq, Np))
    coeff_sep(np.ascontiguousarray(u0.real), np.ascontiguousarray(u0.imag),
              qx, qy, qix, qiy, x0, y0, dx, dyg, w0, k, Ag, nsig, n_p, moff,
              loff, gx, gy, Exr, Exi, Eyr, Eyi, cr, ci)
    return cr + 1j * ci


def _reconstruct_into(outr, outi, Qx, Qy, Px, Py, W, x0, y0, dx, dyg, Ny, Nx,
                      w0, k, Ag, nsig, sep=False):
    """Scatter this batch of beamlets into the EXISTING ``outr``/``outi``
    accumulators (the scatter kernel does ``+=``).  Lets the transport chunk the
    momentum swarm and add each chunk's contribution in place -- the memory
    lever.  ``sep=True`` uses the faster recurrence scatter kernel (ULP-identical
    to the direct one)."""
    scatter = _kernels()[3] if sep else _kernels()[1]
    scatter(Qx.ravel(), Qy.ravel(), Px.ravel(), Py.ravel(),
            np.ascontiguousarray(W.real).ravel(),
            np.ascontiguousarray(W.imag).ravel(),
            x0, y0, dx, dyg, Ny, Nx, w0, k, Ag, nsig, outr, outi)


def _default_p_max(prescription, wavelength, powerless_uncapped=False):
    """Momentum half-range from the system NA (falls back to a moderate cone).

    ``powerless_uncapped=True`` (the momentum-sizing caller) returns ``np.inf``
    for a power-less prescription so no NA cap is imposed; the NA-*estimate*
    callers leave it False and keep the small-NA floor (see below)."""
    try:
        from ..raytrace import system_abcd_prescription  # noqa: F401
    except ImportError:
        return 0.15
    # aperture radius / effective focal length ~ marginal-ray NA; pad x1.6.
    # Narrow catch (not bare Exception): a malformed prescription surfaces as a
    # LookupError (missing key/index), a TypeError/ValueError (bad numeric), an
    # AttributeError (wrong object), or an ArithmeticError (degenerate efl) -- any
    # of which falls back to the moderate cone; anything else is a real bug and
    # propagates.
    try:
        semis = [float(s.get('semi_diameter', 0.0))
                 for s in prescription.get('surfaces', [])]
        ap = prescription.get('aperture_diameter', 0.0)
        r = max([ap / 2.0] + semis) if (semis or ap) else 0.0
        efl = abs(float(system_abcd_prescription(prescription, wavelength)[3]))
        if not np.isfinite(efl):
            # Power-less prescription (flat plate / free space): the marginal-ray
            # NA is 0, so a physical NA cap is meaningless and truncates a
            # wide-angle free-space beam (audit S2-6).  The momentum-sizing caller
            # asks for NO cap (np.inf) so p_max follows the field's own angular
            # content; the NA-estimate callers keep the small-NA floor (0.05).
            return np.inf if powerless_uncapped else 0.05
        na = (r / efl) if (r > 0 and efl > 0) else 0.1
        return float(min(0.6, max(0.05, 1.6 * na)))
    except (LookupError, TypeError, ValueError, AttributeError,
            ArithmeticError):
        return 0.15


_DP_TARGET = 0.008          # auto momentum spacing (direction cosine); see below
_PMAX_BEAMLET_C = 3.0       # p_max completeness floor = C/(k*w0) (beamlet momentum
#                             width); C~2-3 keeps the t=0 identity power >0.99

# H5 (near-collimated input through strong focusing): when the beamlet
# completeness floor is far ABOVE the field's real angular content, flooring
# p_max at it launches excess-momentum beamlets that a strong lens sprays into a
# halo of radius ~efl*p_max, corrupting the SHAPE (not just the scale).  Above
# this floor/content ratio, size p_max to the field content (x the tail factor)
# instead, and recover the beamlet-completeness energy by output power
# normalization (auto-enabled on that path).
_PMAX_CONTENT_OVERRIDE_RATIO = 10.0   # floor/content above this -> content-size p_max
_PMAX_CONTENT_TAIL = 3.0              # spectral tail factor on the content-sized p_max

# Default FGA memory budget (H4a): when the caller leaves ``mem_budget_mb``
# unset, default it to this fraction of the available RAM budget (floored) so
# the byte-identical position-lattice chunk loop engages for a large aperture
# instead of a single huge (Nq, Np) allocation that OOMs.
_DEFAULT_MEM_BUDGET_FRAC = 0.25
_DEFAULT_MEM_BUDGET_FLOOR_MB = 1000.0

# Per-lattice-point transient ray-trace memory (H4b).  The finite-difference
# ``ray_transfer_jacobian`` threads a bundle of ``_FD_BUNDLE_RAYS`` rays (base +
# +/-h in x, y, ux, uy) through the whole system and assembles the 4x4 Jacobian,
# peaking at ~3.2 kB per base lattice point (measured, few-surface conic lens) --
# the old flat 500 B trace term under-counted this ~6x, so a ``mem_budget_mb``
# did NOT bound the FD peak (measured 6000 MB budget -> ~60 GB).  The analytic
# (single forward-AD dual ray) Jacobian threads ``_ADRT_DUAL_SLOTS`` value/deriv
# slots, ~1.8 kB/point.  Sizing the chunk with the ACTUAL per-point trace cost
# makes the budget honored on both paths.
_FD_BUNDLE_RAYS = 9         # base + +/-h in x, y, ux, uy (finite-difference bundle)
_ADRT_DUAL_SLOTS = 5        # analytic forward-AD ray: value + d/d(x, y, ux, uy)
_TRACE_STATE_BYTES = 360.0  # per-point ray state threaded through the system (measured)

# Fixed full-grid array floor for the memory estimate (C2): the per-pixel
# bytes of the persistent + peak-transient grid arrays that do NOT shrink with
# the swarm chunking -- input field (complex128, 16 B) + outr/outi accumulators
# (2x float64, 16 B) + the returned complex output (16 B).  ~48 B/px reproduces
# the H8 empirical "~40 GB raw grid arrays at N=28672" (822 Mpx x 48 B ~ 39 GB);
# the vector path carries two input fields, hence x2.
_FGA_GRID_BYTES_PER_PX_SCALAR = 48.0

# Gabor-window pool for the memory estimate (audit P9).  ``nsig`` sets the
# truncation radius ``R = nsig * w0`` of the Gabor analysis/synthesis window,
# which fixes the half-width of the window in samples,
# ``m = int(R / dx) + 1``.  Two allocations scale with that half-width and
# with NOTHING else in the model, so ``nsig`` is exactly the knob that moves
# them:
#
#   * the SHARED separable tables built once per call by
#     :func:`_gabor_coeff_sep` -- ``gx`` / ``gy`` (length ``2m+1``) plus the
#     phase tables ``Exr``/``Exi``/``Eyr``/``Eyi`` of shape ``(2m+1, n_p)``.
#     The build also holds ``phx`` / ``phy`` and the ``np.sin`` / unary-negate
#     temporaries at peak, so the count is calibrated rather than derived:
#     ``_GABOR_TABLES_PER_AXIS`` arrays of ``(2m+1) * n_p`` float64 per axis.
#     MEASURED by tracemalloc on the peak DELTA between nsig=3 and nsig=12
#     (the nsig-independent ``ascontiguousarray(u0.real/.imag)`` copies and
#     the njit compilation cancel in the delta): 54.72 kB at both N=128 and
#     N=256 (dx=3 um, w0=5*dx, n_p=9), i.e. 8.44 table-equivalents across the
#     two axes -> 4.22 per axis, so 4.25 is the calibrated, slightly
#     conservative value (model 55.08 kB, 0.7% high);
#   * the per-thread inner-x workspace ``ixr`` / ``ixi`` of shape
#     ``(nrow, n_p)`` float64 inside the ``prange`` body of ``_coeff_sep``,
#     with ``nrow <= 2*int(R/dy) + 3``.  numba's ``parallel=True`` gives each
#     worker its own, so it multiplies by the thread count.
#
# The direct (non-separable, ``n_p < 5``) kernel accumulates into scalars and
# allocates nothing window-sized, so this pool is zero there.
_GABOR_TABLES_PER_AXIS = 4.25     # measured (see above), per axis
_GABOR_WORKSPACE_ARRAYS = 2.0     # ixr + ixi, per numba worker


def _fga_numba_threads():
    """Worker count numba's ``prange`` will use (for the P9 window-pool term).

    Reads numba's own configured thread count when numba is importable so the
    estimate matches the run; falls back to ``os.cpu_count()`` (numba's own
    default) when it is not, so the estimate stays available on a
    numba-less box.
    """
    try:
        import numba
        return max(1, int(numba.get_num_threads()))
    except (ImportError, AttributeError, RuntimeError, ValueError):
        # ImportError: numba absent.  AttributeError: an old / stub numba
        # without the threading-layer API.  RuntimeError / ValueError:
        # numba present but its threading layer failed to initialise.
        import os
        return max(1, int(os.cpu_count() or 1))


def _fga_gabor_window_bytes(nsig, w0, dx, dyg, n_p, use_sep, cfull_mult=1.0,
                            n_threads=None):
    """Peak bytes of the ``nsig``-sized Gabor-window pool (audit P9).

    ``nsig`` was documented on :func:`fga_memory_estimate` as "as on
    :func:`apply_real_lens_fga`" but was never read, while the real path reads
    it six times -- so the estimate was invariant to a 12x change in the
    window radius.  This is the term that makes it live.  Returns 0.0 on the
    direct (non-separable) kernel, which allocates nothing window-sized.
    """
    if not use_sep:
        return 0.0
    R = float(nsig) * float(w0)
    mx = int(R / float(dx)) + 1
    my = int(R / float(dyg)) + 1
    npf = float(n_p)
    tables = _GABOR_TABLES_PER_AXIS * 8.0 * npf * ((2 * mx + 1) + (2 * my + 1))
    threads = _fga_numba_threads() if n_threads is None else int(n_threads)
    workspace = (_GABOR_WORKSPACE_ARRAYS * 8.0 * npf
                 * (2 * my + 3) * threads)
    return float(cfull_mult) * (tables + workspace)


# Lever #3 (coarse-lattice trace + interpolation) tunables.
_COARSE_INTERP_ORDER = 3    # cubic map_coordinates upsample (opd ~2e-5 waves, well
#                             under the ~1e-3 FGA floor; ~2x faster than quintic)
_RIM_CELLS = 3              # dilation (coarse cells) of the vignetting boundary
_COARSE_SUPP_FRAC = 1e-3    # rim direct-trace only where windowed |u0| is >= this


def _is_all_conic(surfaces):
    """True when EVERY surface is a rotationally-symmetric conic (sphere / conic,
    no aspheric-polynomial / freeform / biconic terms) -- the class the analytic
    (forward-mode-AD) differential Jacobian handles exactly."""
    for s in surfaces:
        if (getattr(s, 'aspheric_coeffs', None) or getattr(s, 'freeform', None)
                or getattr(s, 'radius_y', None) is not None
                or getattr(s, 'conic_y', None) is not None
                or getattr(s, 'aspheric_coeffs_y', None) is not None):
            return False
    return True


def _pick_ray_transfer(surfaces, exact):
    """Differential ray-transfer Jacobian primitive (lever #1).  ``exact=True``
    uses the truncation-free analytic (forward-mode-AD) Jacobian when EVERY
    surface is a rotationally-symmetric conic -- exact vs the finite-difference
    ~1e-8, pure NumPy, ~1.2x faster on a large aperture AND ~5x lighter (one
    traced ray vs the FD 9-ray bundle, so a large-N full trace fits a memory
    budget the FD path OOMs -- H4c).  Falls back to the FD primitive for aspheric
    / freeform / biconic surfaces (which the analytic form does not handle) and
    when ``exact=False``.  ``exact=None`` (the default) AUTO-selects: analytic for
    an all-conic prescription, FD otherwise."""
    from ..raytrace.differential import ray_transfer_jacobian, ray_transfer_jacobian_analytic
    all_conic = _is_all_conic(surfaces)
    if exact is None:
        exact = all_conic       # H4c: default to the analytic Jacobian when it applies
    if not exact or not all_conic:
        return ray_transfer_jacobian
    return ray_transfer_jacobian_analytic


# Lever #2: cross-call cache of the FIELD-INDEPENDENT coarse pre-trace (the
# per-momentum coarse ray-data grids + coarse alive masks depend only on the
# geometry, not on the input field), so a repeated propagation through the SAME
# optics -- e.g. a design / optimization loop that varies only the field -- reuses
# it.  Opt-in (``cache_trace``); bounded, FIFO-evicted; each entry is the coarse
# swarm (~ full swarm / coarse_stride^2), so keep the bound small.  MEASURED
# footprint (G1 cache audit, 2026-07-19; per-momentum grids (7, N//stride,
# N//stride) float64 + bool alive, Np momenta): at coarse_stride=3, Np=49 the
# entry is ~311 MB at N=2048 and ~1.25 GB at N=4096, so the FIFO bound of 2 caps
# the retained coarse cache at ~0.6 GB / ~2.5 GB respectively.  Drain any time
# with clear_coarse_trace_cache() / clear_asm_caches() (registered below).
_COARSE_CACHE: Dict[Any, Any] = {}
_COARSE_CACHE_MAX = 2
# v5.24.4 (audit hygiene): thread-safety lock for ``_COARSE_CACHE``.  The
# read (``get``) -> ``_coarse_cache_put`` (len-check -> pop -> __setitem__)
# sequence is a read-modify-write two threads propagating through the SAME
# optics could tear.  Follows the ``_THROUGH_FOCUS_SCAN_JAX_CACHE_LOCK``
# precedent in :mod:`analysis.through_focus`.  Lock-scope discipline: the
# expensive per-momentum coarse trace runs OUTSIDE the lock so a concurrent
# cache hit on a different key isn't blocked on it (mirrors the jit-compile
# discipline of the through-focus lock).
_COARSE_CACHE_LOCK = threading.Lock()


def clear_coarse_trace_cache() -> None:
    """Drop every cached FGA coarse pre-trace (lever #2).

    Forces the next ``cache_trace=True`` propagation to recompute the
    field-independent coarse ray-data grids from scratch.  Use in unit
    tests that pin cache mechanics or in long-running pipelines that want
    to release the retained coarse swarms.
    """
    with _COARSE_CACHE_LOCK:
        _COARSE_CACHE.clear()


# v5.24.4 (audit hygiene): enroll the coarse-trace clearer with the central
# registry at import time so ``clear_all_registered_caches`` drains it (the
# v4.16.1 enrollment meta-pin requires every module-level ``_*_CACHE`` to be
# registered).  Late-binding closure preserves ``mock.patch.object`` test
# semantics (mirrors the through_focus / propagation enrollment pattern).
try:
    import sys as _sys

    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        'fga_coarse_trace',
        lambda: getattr(_this_mod, 'clear_coarse_trace_cache')(),
    )
except ImportError:
    pass


def _coarse_geom_key(surfs, wavelength, Nx, Ny, dx, dyg, dq_step, coarse_stride,
                     px, w0, z_image):
    """Hashable identity of the FIELD-INDEPENDENT coarse pre-trace (surfaces +
    grid + momentum lattice + frozen width + image leg).  Two calls with equal
    keys produce identical coarse grids/alive, so the trace is reusable."""
    surf_key = tuple((float(getattr(s, 'radius', np.inf)),
                      float(getattr(s, 'conic', 0.0)),
                      float(getattr(s, 'thickness', 0.0)),
                      str(getattr(s, 'glass', getattr(s, 'material', ''))),
                      str(getattr(s, 'aspheric_coeffs', None)),
                      float(getattr(s, 'semi_diameter', 0.0) or 0.0))
                     for s in surfs)
    return (surf_key, float(wavelength), int(Nx), int(Ny), float(dx), float(dyg),
            int(dq_step), int(coarse_stride), int(px.size), float(px[0]),
            float(px[-1]), float(w0), float(z_image))


def _coarse_cache_put(key, val):
    with _COARSE_CACHE_LOCK:
        if len(_COARSE_CACHE) >= _COARSE_CACHE_MAX:
            _COARSE_CACHE.pop(next(iter(_COARSE_CACHE)))    # FIFO evict oldest
        _COARSE_CACHE[key] = val


def _field_angular_content(E_in, dx, dyg, wavelength, frac=0.999):
    """Half-range (direction cosine) of the field's angular spectrum holding
    ``frac`` of the energy.  ``p_max`` must COVER the field's real angular
    content -- a diverging / structured field carries content well beyond
    ``p=0`` (~ its divergence half-angle), a collimated beam ~0.  Momenta the
    field does NOT contain get a ~zero Gabor coefficient, so sizing ``p_max`` to
    the field (not to an over-wide fixed cone) keeps the momentum quadrature
    matched to the field."""
    E = np.asarray(E_in)
    ny, nx = E.shape
    P = np.abs(np.fft.fft2(E)) ** 2
    px = np.fft.fftfreq(nx, dx) * wavelength                # p_x direction cosine
    py = np.fft.fftfreq(ny, dyg) * wavelength
    PX, PY = np.meshgrid(px, py)
    pr = np.sqrt(PX ** 2 + PY ** 2).ravel()
    Pf = P.ravel()
    tot = float(Pf.sum())
    if not np.isfinite(tot) or tot <= 0.0:
        return 0.0
    order = np.argsort(pr)
    cum = np.cumsum(Pf[order]) / tot
    return float(pr[order][min(int(np.searchsorted(cum, frac)), pr.size - 1)])


def _resolve_sampling(E_in, dx, dyg, wavelength, w0, prescription, p_max, n_p,
                      info=None):
    """Resolve ``(p_max, n_p)``, auto-sizing either when ``None`` so the
    phase-space swarm is MATCHED to the field: ``p_max`` covers the field's
    angular content AND each beamlet's own momentum width (for completeness),
    capped by the system NA, and ``n_p`` makes the momentum spacing
    ``dp = 2*p_max/(n_p-1)`` fine enough (``~<= _DP_TARGET``) to resolve the
    phase-space quadrature -- which is what makes FGA accurate on diverging beams
    (whose broad phase-space footprint the fixed old default under-sampled;
    leading FGA is EXACT for free space, so with dp matched a diverging beam
    reconstructs to ~round-off).  An explicit ``p_max`` / ``n_p`` is honoured.

    When a mutable ``info`` dict is passed, ``info['content_override']`` reports
    whether the H5 near-collimated content-sizing path was taken (the caller then
    power-normalizes the output to restore the beamlet-completeness energy)."""
    content_override = False
    if p_max is None:
        na_cap = _default_p_max(prescription, wavelength, powerless_uncapped=True)
        content = _field_angular_content(E_in, dx, dyg, wavelength)
        # COMPLETENESS FLOOR (audit S2-2): the swarm must cover each beamlet's OWN
        # momentum width ~2/(k*w0), else the t=0 resolution of identity loses
        # power -- a collimated Gaussian has narrow angular CONTENT but the frozen
        # beamlet still spans +-~2/(k*w0), so the old 0.03 floor gave up to 34%
        # power deficit (while fidelity stayed ~1, a pure under-normalization).
        # Floor at _PMAX_BEAMLET_C/(k*w0) (power >0.99).  Cover the field's content
        # generously (x1.5 spectral tail; over-wide p_max is harmless at fine dp),
        # never wider than the system NA cone (x1.5; content beyond it is clipped)
        # -- but a power-less prescription (flat plate / free space) has na_cap=inf
        # so p_max follows the field's own content uncapped (audit S2-6: the NA cap
        # is physically meaningless for free space and truncated wide-angle beams).
        k = 2.0 * np.pi / wavelength
        beamlet_floor = _PMAX_BEAMLET_C / (k * w0)
        # H5: for a NEAR-COLLIMATED input the field's angular CONTENT is far below
        # the completeness floor; flooring p_max there launches excess-momentum
        # beamlets that a strong-focusing system sprays into a halo of radius
        # ~efl*p_max, corrupting the SHAPE (measured: f/5 r2m 187/304 um vs oracle
        # 65).  When floor/content exceeds _PMAX_CONTENT_OVERRIDE_RATIO, size
        # p_max to the field's own content (x tail) instead of the floor, and
        # signal the caller to power-normalize (recovering the beamlet-
        # completeness energy the floor would have supplied).  The identity /
        # diverging cases (ratio ~1-7) keep the floor path unchanged.
        if content > 0.0 and beamlet_floor > _PMAX_CONTENT_OVERRIDE_RATIO * content:
            content_override = True
            p_max = float(min(1.5 * na_cap, _PMAX_CONTENT_TAIL * content))
            warnings.warn(
                "FGA auto momentum sampling: the input is near-collimated (its "
                f"angular content {content:.3g} is <{1.0 / _PMAX_CONTENT_OVERRIDE_RATIO:.2g} "
                f"of the beamlet completeness floor {beamlet_floor:.3g}). Sizing "
                f"p_max to the field content ({p_max:.3g}) instead of the floor "
                "to avoid a spurious focused-halo artifact, and power-normalizing "
                "the output to restore the completeness energy. Pass an explicit "
                "p_max / normalize_output to override.",
                RuntimeWarning, stacklevel=2)
        else:
            p_max = float(min(1.5 * na_cap, max(beamlet_floor, 1.5 * content)))
    if info is not None:
        info['content_override'] = content_override
    if n_p is None:
        # momentum SPACING (direction cosine) fine enough to converge the
        # phase-space quadrature.  Empirically ~0.008 works across beam sizes
        # (the historical diverging-beam cap was dp ~0.02-0.03, too coarse); it
        # is roughly field-independent for smooth beams -- a wide collimated beam
        # has a narrow p_max (few samples) and a small diverging beam a wider
        # p_max (more), both converging at this spacing -- so n_p scales with
        # p_max, not with the beam size.  The frame stays hugely over-complete
        # (Nyquist allows dp up to ~lambda/dq >> this).
        n_p_want = int(np.ceil(2.0 * p_max / _DP_TARGET)) | 1
        n_p = int(min(61, max(7, n_p_want)))  # clamp: >=7 samples, <=61 (cost)
        # audit S2-15: the <=61 cap silently re-enters the under-sampled dp
        # regime for a wide p_max (p_max > ~0.24 at _DP_TARGET=0.008): after the
        # cap the ACHIEVED spacing dp = 2*p_max/(n_p-1) can exceed the target,
        # degrading the phase-space quadrature that makes FGA accurate on
        # diverging / high-NA fields (probe: fid 0.917 clamped vs 1.0000
        # unclamped).  Warn so the caller can restore dp <= target by passing an
        # explicit larger n_p (cost) or a smaller p_max, rather than trust a
        # quietly down-sampled result.
        if n_p < n_p_want:
            dp_ach = 2.0 * p_max / (n_p - 1)
            if info is not None:
                # expose the target-meeting n_p so callers / fga_memory_estimate
                # can size the exact run the clamp declined.
                info['n_p_target'] = int(n_p_want)
                info['n_p_clamped'] = int(n_p)
            if dp_ach > _DP_TARGET * (1.0 + 1e-9):
                # the momentum-swarm memory scales with Np = n_p**2, so meeting
                # the target costs (n_p_want / n_p)**2 x the swarm arrays.  Give
                # the caller the explicit n_p AND that factor so they can weigh
                # the fidelity vs cost; fga_memory_estimate turns it into an
                # absolute MB figure for their grid.
                mem_factor = (float(n_p_want) / float(n_p)) ** 2
                warnings.warn(
                    "FGA auto momentum sampling: n_p clamped to the cost cap "
                    f"({n_p}) yields dp={dp_ach:.4g} > target {_DP_TARGET:.4g} "
                    f"for p_max={p_max:.4g}; the phase-space quadrature is "
                    "under-sampled and a diverging / high-NA field may lose "
                    f"fidelity. To restore dp <= target pass n_p={n_p_want} "
                    f"(grows the momentum swarm ~{mem_factor:.1f}x in memory -- "
                    "call fga_memory_estimate(...) for the absolute MB on your "
                    "grid), or lower p_max.",
                    RuntimeWarning, stacklevel=2)
    return float(p_max), int(n_p)


def _env_mem_budget_mb():
    """Read the ``LUMENAIRY_MEM_BUDGET_MB`` environment override (C2), or
    ``None`` when unset / invalid.  A positive number pins the FGA default
    memory budget (MB) directly -- the escape hatch for a SHARED box where the
    25%-of-RAM auto-default (which sees the whole machine, not this process's
    share) is too generous.  A non-positive / unparseable value is ignored with
    a one-line warning so a typo does not silently disable the guard."""
    import os
    raw = os.environ.get('LUMENAIRY_MEM_BUDGET_MB')
    if raw is None or str(raw).strip() == '':
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        warnings.warn(
            f"LUMENAIRY_MEM_BUDGET_MB={raw!r} is not a number; ignoring it and "
            "falling back to the 25%-of-RAM FGA default.",
            RuntimeWarning, stacklevel=2)
        return None
    if not (val > 0.0):
        warnings.warn(
            f"LUMENAIRY_MEM_BUDGET_MB={raw!r} is not positive; ignoring it.",
            RuntimeWarning, stacklevel=2)
        return None
    return val


def _default_mem_budget_mb():
    """Default FGA memory budget (MB) when the caller leaves ``mem_budget_mb``
    unset (H4a): the ``LUMENAIRY_MEM_BUDGET_MB`` environment override when set
    (C2), else a fraction (``_DEFAULT_MEM_BUDGET_FRAC``) of the available RAM
    budget, floored at ``_DEFAULT_MEM_BUDGET_FLOOR_MB``.  A concrete budget lets
    the (byte-identical, max|diff|=0.0) position-lattice chunk loop engage for a
    large aperture instead of a single huge (Nq, Np) allocation that OOMs."""
    env = _env_mem_budget_mb()
    if env is not None:
        return env
    from ..memory import get_ram_budget
    mb = _DEFAULT_MEM_BUDGET_FRAC * float(get_ram_budget()) / 1e6
    return max(_DEFAULT_MEM_BUDGET_FLOOR_MB, mb)


def _fga_lattice_point_bytes(Np, cw, use_sep, cfull_mult, fd_bundle):
    """Peak transient bytes per POSITION-lattice point of the FGA swarm (H4
    memory model) -- the single source of truth shared by the chunk sizer
    (:func:`_resolve_nq_chunk`) and the public sizing helper
    (:func:`fga_memory_estimate`) so they can never drift.

    * separable coefficient array ``c_full`` -- ``cfull_mult*Np*16`` B (whole
      momentum grid for this lattice point; ``cfull_mult`` = 1 scalar, 2 for the
      vector ``Ex``/``Ey`` path);
    * beamlet transport arrays ``QX/QY/PX/PY + AW + coeff (+ Jones)`` --
      ``cw*~100`` B over the ``cw`` momenta in flight;
    * the differential ray-trace bundle: the FD Jacobian threads a
      ``_FD_BUNDLE_RAYS``-ray bundle (~3.2 kB/pt), the analytic single-ray
      Jacobian ``_ADRT_DUAL_SLOTS`` slots (~1.8 kB/pt)."""
    trace_bytes = ((_FD_BUNDLE_RAYS if fd_bundle else _ADRT_DUAL_SLOTS)
                   * _TRACE_STATE_BYTES)
    return (cfull_mult * Np * 16.0 if use_sep else 0.0) + cw * 100.0 + trace_bytes


def _chunk_from_budget(shape, dq_step, chunk, mem_budget_mb):
    """Resolve the momentum-chunk size.  An explicit ``chunk`` wins; otherwise a
    ``mem_budget_mb`` caps peak beamlet memory by sizing the chunk from the
    position-lattice count ``Nq`` (~100 B per ``(q, p)`` beamlet across the
    QX/QY/PX/PY + Gabor-coefficient + weight + scatter arrays).  Both unset ->
    the whole swarm at once."""
    if chunk is not None:
        return int(chunk)
    if mem_budget_mb is None:
        return None
    ny, nx = shape
    nq = len(range(0, nx, dq_step)) * len(range(0, ny, dq_step))
    return max(1, int(float(mem_budget_mb) * 1e6 / (nq * 100.0)))


def _resolve_nq_chunk(Nq, Np, use_sep, cw, mem_budget_mb, fn, cfull_mult=1.0,
                      fd_bundle=True):
    """POSITION-lattice (``Nq``) chunk size from ``mem_budget_mb``, or ``None`` if
    the whole lattice fits / no budget.  ``mem_budget_mb`` now bounds BOTH
    dimensions: momenta via ``cw`` and the position lattice via this chunk, so a
    large aperture runs within the budget (additive over lattice chunks) instead
    of OOMing.  Peak per lattice-chunk of size ``nqc``:

    * separable coeff array ``c_full`` -- ``cfull_mult*nqc*Np*16`` B (whole
      momentum grid; ``cfull_mult`` = 1 scalar, 2 for the vector ``Ex``/``Ey``);
    * beamlet transport arrays ``QX/QY/PX/PY + AW + coeff (+ Jones)`` --
      ``nqc*cw*~100`` B;
    * the per-momentum differential ray-trace temporaries.  ``fd_bundle=True``
      (the finite-difference Jacobian) threads a ``_FD_BUNDLE_RAYS``-ray bundle
      through the system (~``_FD_BUNDLE_RAYS*_TRACE_STATE_BYTES`` ~3.2 kB/point);
      ``fd_bundle=False`` (the analytic single-ray Jacobian) is
      ``_ADRT_DUAL_SLOTS*_TRACE_STATE_BYTES`` ~1.8 kB/point.  Counting the ACTUAL
      trace cost (not the old flat 500 B, which under-counted the FD bundle ~6x)
      is what makes ``mem_budget_mb`` HONORED on the FD path (H4b).

    Only raises if even a SINGLE lattice point exceeds the budget (absurdly tiny
    budget / huge ``n_p``) -- otherwise it always finds a workable chunk."""
    if mem_budget_mb is None:
        return None
    budget = float(mem_budget_mb) * 1e6
    per = _fga_lattice_point_bytes(Np, cw, use_sep, cfull_mult, fd_bundle)
    nqc = int(budget / per)
    if nqc < 1:
        raise MemoryError(
            f"{fn}: a single position-lattice point needs ~{per / 1e6:.2f} MB "
            f"(Np={Np}, chunk={cw}), exceeding mem_budget_mb={mem_budget_mb} MB.  "
            f"Raise the budget, lower n_p, or use a smaller momentum chunk.")
    return None if nqc >= Nq else nqc


def _fga_coarse(u0, dx, dyg, x0, y0, Ny, Nx, k, w0, nsig, Ag, C, kw2, surfs,
                wavelength, z_image, dq_step, px, py, n_p, Np,
                use_sep, cw, nq_chunk, coarse_stride, cache_trace=False, Cp=None):
    """Lever #3: coarse-lattice trace + spline ``map_coordinates`` interpolation.

    The ray trace is ~80% of FGA cost, and the exit-vertex map
    ``(q, p) -> (xv, yv, ux, uy, opd, a)`` is smooth and single-valued at the
    launch plane (the caustic fold is DOWNSTREAM), so trace only a
    stride-``coarse_stride`` position sub-lattice per momentum and interpolate the
    smooth ray quantities to the full lattice with an order-``_COARSE_INTERP_ORDER``
    (currently 3 = cubic; the public ``coarse_stride`` docstring and the constant
    agree -- this docstring said "order-5 (quintic)" pre-S9)
    ``scipy.ndimage.map_coordinates`` -- C-backed and grid-regular, ~3.6-5.5x
    faster than the full trace (scaling up with N).  ``AW = C*a*exp(i k opd)`` is
    reconstructed at full resolution; the oscillating ``AW`` is NEVER
    interpolated.  Validated no-loss (fid 1.0; opd ~1e-8 waves, position ~1e-9 m,
    prefactor ~1e-12 rel).  RIM: where the aperture vignettes the beam,
    interpolation across the alive/dead edge is invalid, so beamlets within a
    ``_RIM_CELLS``-cell band of a dead coarse node that ALSO carry beam support
    are traced DIRECTLY (dark exterior points interp-marked alive are harmless --
    their Gabor coefficient is ~0, so weight ~0).  Keeps the separable analysis
    and the batched numba scatter; position pruning is not applied (the coarse
    trace already visits only ``O(Nq / coarse_stride^2)`` points).

    .. versionchanged:: 5.30
        Dropped the ``coeff_frac`` parameter (audit P13): it appeared only in
        the signature and was never read in the body, so a caller's
        momentum-coefficient pruning request was silently discarded on the
        coarse path.  Momentum pruning is simply not implemented here -- the
        signature now says so.  Module-private, one call site."""
    from scipy.ndimage import (
        binary_dilation,
        distance_transform_edt,
        map_coordinates,
        uniform_filter,
    )

    from ..raytrace.differential import ray_transfer_jacobian
    M = int(coarse_stride)
    ii = np.arange(0, Nx, dq_step)
    jj = np.arange(0, Ny, dq_step)
    xs_f = x0 + ii * dx
    ys_f = y0 + jj * dyg
    nxf, nyf = xs_f.size, ys_f.size
    # coarse sub-lattice indices into the full axes (include the last so the
    # interpolation never extrapolates past the traced support)
    cix = np.unique(np.r_[np.arange(0, nxf, M), nxf - 1])
    ciy = np.unique(np.r_[np.arange(0, nyf, M), nyf - 1])
    xs_c, ys_c = xs_f[cix], ys_f[ciy]
    nxc, nyc = xs_c.size, ys_c.size
    QXc, QYc = np.meshgrid(xs_c, ys_c)
    qxc = QXc.ravel().astype(np.float64)
    qyc = QYc.ravel().astype(np.float64)
    QXf, QYf = np.meshgrid(xs_f, ys_f)         # full lattice (C-order: y outer)
    qxf = QXf.ravel().astype(np.float64)
    qyf = QYf.ravel().astype(np.float64)
    Nq = qxf.size
    # fractional coarse-index coordinate of every full grid LINE (map_coordinates)
    fx = np.interp(xs_f, xs_c, np.arange(nxc, dtype=np.float64))
    fy = np.interp(ys_f, ys_c, np.arange(nyc, dtype=np.float64))
    cxi = np.clip(np.round(fx).astype(int), 0, nxc - 1)   # nearest coarse cell
    cyi = np.clip(np.round(fy).astype(int), 0, nyc - 1)
    # beam-support mask over the full lattice (rim direct-trace only where the
    # beam lives, so the dark exterior aperture edge is never traced)
    a2 = np.abs(u0) ** 2
    wwx = max(1, int(round(2.0 * nsig * w0 / dx)))
    wwy = max(1, int(round(2.0 * nsig * w0 / dyg)))
    amp = np.sqrt(np.maximum(uniform_filter(a2, size=(wwy, wwx),
                                            mode='constant'), 0.0))
    supp2d = amp[np.ix_(jj, ii)] > _COARSE_SUPP_FRAC * float(amp.max() + 1e-300)
    pv = px[:n_p]

    def _trace(qxp, qyp, pxi, pyi, pz):
        uxin = np.full(qxp.shape, pxi / pz)
        uyin = np.full(qxp.shape, pyi / pz)
        dt = ray_transfer_jacobian(qxp.copy(), qyp.copy(), uxin, uyin, surfs,
                                   wavelength, per_surface=False)
        uxo, uyo = dt.ux, dt.uy
        xv = dt.x + z_image * uxo
        yv = dt.y + z_image * uyo
        opd = dt.opd + z_image * np.sqrt(1.0 + uxo ** 2 + uyo ** 2)
        Ml = np.tile(np.eye(4), (qxp.shape[0], 1, 1))
        Ml[:, 0, 2] = z_image
        Ml[:, 1, 3] = z_image
        Mm = Ml @ dt.jacobian
        # ------------------------------------------------------------------
        # slope -> direction-cosine monodromy conversion (audit S2-16).
        # ------------------------------------------------------------------
        # The Herman-Kluk prefactor a = sqrt(det Z) needs the ray-transfer
        # Jacobian Mm (in SLOPE state [x, y, u, v]) expressed in DIRECTION-
        # COSINE momentum p = u / sqrt(1 + u^2 + v^2).  The exact output
        # conversion is the FULL 2x2 Jacobian
        #     d(p_x, p_y)/d(u_x, u_y)
        #       = (1 + u^2 + v^2)^-1.5 * [[1 + v^2, -u v], [-u v, 1 + u^2]],
        # and the input the analogous 2x2 d(u)/d(p).  Here we apply only its
        # ISOTROPIC (trace/2-like) part as a SCALAR -- ``go`` on the output
        # block, ``gi`` on the input block -- dropping the off-diagonal
        # ``-u v`` skew term and the diagonal ``u^2`` vs ``v^2`` anisotropy.
        # That approximation is exact on-axis and O(u^2) in the prefactor
        # otherwise, growing with NA and skew.  It is BELOW the noise of the
        # moderate-NA regime the library validates (measured FGA fidelity
        # 0.997-0.999 there); the full 2x2 upgrade would change the validated
        # high-NA prefactor and is deferred (see audit AUDIT_V5_24_2 S2-16 and
        # tests/unit/test_audit_g06_perf.py::test_s2_16_*).
        go = 1.0 / (1.0 + uxo ** 2 + uyo ** 2) ** 1.5
        gi = (1.0 + (pxi / pz) ** 2 + (pyi / pz) ** 2) ** 1.5
        a = _hk_prefactor(Mm, go, gi, kw2)
        return xv, yv, uxo, uyo, opd, a, np.asarray(dt.alive, bool)

    # ---- FIELD-INDEPENDENT coarse pre-trace (per-momentum coarse ray-data grids
    # + coarse alive masks): a pure function of the geometry, so cacheable across
    # calls (lever #2).  Grids are dead-node-filled (nearest-alive) so the interp
    # stays stable; the rim (which depends on the FIELD's support) overrides them.
    ckey = (_coarse_geom_key(surfs, wavelength, Nx, Ny, dx, dyg, dq_step,
                             coarse_stride, px, w0, z_image)
            if cache_trace else None)
    if ckey is not None:
        with _COARSE_CACHE_LOCK:
            cached = _COARSE_CACHE.get(ckey)
    else:
        cached = None
    if cached is not None:
        grids, alv_list = cached
    else:
        grids = [None] * Np                # (7, nyc, nxc) per momentum
        alv_list = [None] * Np             # coarse alive (nyc, nxc) per momentum
        for jm in range(Np):
            pxi, pyi = float(px[jm]), float(py[jm])
            pz = math.sqrt(max(1.0 - pxi * pxi - pyi * pyi, 1e-12))
            xvc, yvc, uxc, uyc, opdc, ac, alvc = _trace(qxc, qyc, pxi, pyi, pz)
            alv2 = alvc.reshape(nyc, nxc)
            stack = np.stack([xvc, yvc, uxc, uyc, opdc, ac.real, ac.imag]
                             ).reshape(7, nyc, nxc)
            if not alv2.all():
                idx = distance_transform_edt(~alv2, return_distances=False,
                                             return_indices=True)
                stack = stack[:, idx[0], idx[1]]
            grids[jm] = stack
            alv_list[jm] = alv2
        if ckey is not None:
            _coarse_cache_put(ckey, (grids, alv_list))

    # ---- FIELD-DEPENDENT rim: support-selected direct traces at the vignetting
    # boundary (cheap, O(boundary); recomputed per call from the coarse alive) ----
    rim = [None] * Np                      # (ridx, deadns, traced-values) or None
    for jm in range(Np):
        alv2 = alv_list[jm]
        if alv2.all():
            continue
        pxi, pyi = float(px[jm]), float(py[jm])
        pz = math.sqrt(max(1.0 - pxi * pxi - pyi * pyi, 1e-12))
        dfull = binary_dilation(~alv2, iterations=_RIM_CELLS)[np.ix_(cyi, cxi)]
        ridx = np.flatnonzero(dfull.ravel() & supp2d.ravel())
        deadns = np.flatnonzero(dfull.ravel() & ~supp2d.ravel())
        if ridx.size:
            rv = _trace(qxf[ridx], qyf[ridx], pxi, pyi, pz)
            rim[jm] = (ridx, deadns, rv)
        elif deadns.size:
            rim[jm] = (np.empty(0, int), deadns, None)

    # ---- reconstruct: lattice-chunk (memory) x momentum-batch (batched scatter)
    outr = np.zeros((Ny, Nx))
    outi = np.zeros((Ny, Nx))
    qbounds = ([(0, Nq)] if nq_chunk is None
               else [(s, min(s + nq_chunk, Nq)) for s in range(0, Nq, nq_chunk)])
    for qs, qe in qbounds:
        nqc = qe - qs
        # fractional coarse coords of this chunk's full-lattice points
        pidx = np.arange(qs, qe)
        iy = pidx // nxf
        ix = pidx - iy * nxf
        coords = np.vstack([fy[iy], fx[ix]])            # (2, nqc)
        c_full = (_gabor_coeff_sep(u0, qxf[qs:qe], qyf[qs:qe], pv, x0, y0, dx,
                                   dyg, w0, k, Ag, nsig) if use_sep else None)
        for cs in range(0, Np, cw):
            ce = min(cs + cw, Np)
            mm = ce - cs
            if use_sep:
                cco = c_full[:, cs:ce]
            else:
                cco = _gabor_coeff(u0, qxf[qs:qe], qyf[qs:qe], px[cs:ce],
                                   py[cs:ce], x0, y0, dx, dyg, w0, k, Ag, nsig)
            QX = np.empty((nqc, mm))
            QY = np.empty((nqc, mm))
            PX = np.empty((nqc, mm))
            PY = np.empty((nqc, mm))
            AW = np.zeros((nqc, mm), dtype=np.complex128)
            ALV = np.ones((nqc, mm), dtype=bool)
            for j in range(mm):
                jm = cs + j
                g = grids[jm]
                vals = [map_coordinates(g[t], coords, order=_COARSE_INTERP_ORDER,
                                        mode='nearest') for t in range(7)]
                xv, yv, uxo, uyo, opd = vals[0], vals[1], vals[2], vals[3], vals[4]
                a = vals[5] + 1j * vals[6]
                if rim[jm] is not None:                 # rim direct-trace overwrite
                    ridx, deadns, rv = rim[jm]
                    sel = (ridx >= qs) & (ridx < qe)
                    if sel.any():
                        loc = ridx[sel] - qs
                        rvs = tuple(v[sel] for v in rv[:6])
                        xv[loc], yv[loc] = rvs[0], rvs[1]
                        uxo[loc], uyo[loc] = rvs[2], rvs[3]
                        opd[loc] = rvs[4]
                        a[loc] = rvs[5]
                        ALV[loc, j] = rv[6][sel]
                    dsel = (deadns >= qs) & (deadns < qe)
                    if dsel.any():
                        ALV[deadns[dsel] - qs, j] = False
                invo = 1.0 / np.sqrt(1.0 + uxo ** 2 + uyo ** 2)
                QX[:, j] = xv
                QY[:, j] = yv
                PX[:, j] = uxo * invo
                PY[:, j] = uyo * invo
                cj = C if Cp is None else Cp[jm]        # per-momentum quad weight
                AW[:, j] = cj * a * np.exp(1j * k * opd)
            W = cco * AW
            W[~ALV] = 0.0
            _reconstruct_into(outr, outi, QX, QY, PX, PY, W, x0, y0, dx, dyg,
                              Ny, Nx, w0, k, Ag, nsig, sep=use_sep)
    return outr + 1j * outi


def _fga_through_lens(u0, dx, dyg, prescription, wavelength, w0, z_image,
                      dq_step, p_max, n_p, nsig, chunk=None, prune_frac=0.0,
                      coeff_frac=0.0, separable="auto", mem_budget_mb=None,
                      coarse_stride=1, exact_jacobian=None, cache_trace=False,
                      momentum_nodes=None):
    """Core FGA transport through a prescription to (last vertex + z_image).

    ``dx`` / ``dyg`` are the (possibly anamorphic) x / y pixel pitches.  ``chunk``
    (momenta per batch, ``None`` = all at once) bounds the MOMENTUM dimension;
    ``mem_budget_mb`` additionally chunks the POSITION lattice (``nq_chunk``) so
    peak memory is ``O(nq_chunk * (Np + chunk))`` -- a large aperture runs within
    the budget as an additive sum over lattice chunks instead of OOMing.  The
    reconstruction is an additive sum over independent beamlets, so both chunked
    results match the full swarm to float round-off (coeff pruning stays matched
    via a global-per-momentum-peak pre-pass when the lattice is chunked).
    ``prune_frac`` drops lattice points with negligible windowed ``|u0|``.
    ``separable`` (True / False / 'auto') uses the tensor-momentum analysis kernel
    + recurrence scatter kernel -- both ULP-identical to the direct kernels but
    faster; 'auto' enables them when ``n_p >= 5`` (below that the ~``n_p`` x
    analysis win is negligible)."""
    from ..raytrace import surfaces_from_prescription

    k = 2.0 * np.pi / wavelength
    Ny, Nx = u0.shape
    x0 = -(Nx / 2) * dx
    y0 = -(Ny / 2) * dyg
    Ag = (1.0 / (np.pi * w0 ** 2)) ** 0.5

    qx, qy, px, py, dp, wp = _swarm_lattice(Ny, Nx, dx, dyg, x0, y0, dq_step,
                                            p_max, n_p, nodes=momentum_nodes)
    if prune_frac > 0.0:
        keep = _lattice_support_mask(u0, dx, dyg, dq_step, w0, nsig, prune_frac)
        qx = qx[keep]
        qy = qy[keep]
    Nq = qx.shape[0]
    Np = px.shape[0]
    # phase-space measure * the FGA normalization.  The position measure is the
    # anamorphic lattice cell (dq_step*dx)(dq_step*dyg).  The /2^{d/2} (d=2
    # transverse) removes the double-counted Herman-Kluk identity factor
    # a(0)=2^{d/2}: without it the t=0 resolution of identity over-counts by
    # 2^d=4 in power (verified: the flat-prescription output=0 power ratio -> 4.0
    # in the well-sampled limit, -> 1.0 with this factor).  UNIFORM swarm: the
    # scalar ``dp**2`` momentum cell (byte-identical to prior releases).  ADAPTIVE
    # swarm: ``Cp`` carries the PER-MOMENTUM cell area ``wp`` (outer(wv, wv)) so a
    # non-uniform p-measure is quadrature-correct (else the power mis-normalizes).
    C = ((k / (2.0 * np.pi)) ** 2 * (dq_step ** 2 * dx * dyg) * (dp ** 2)) / 2.0
    Cp = (None if wp is None else
          ((k / (2.0 * np.pi)) ** 2 * (dq_step ** 2 * dx * dyg)) / 2.0 * wp)

    # trace to the LAST SURFACE VERTEX; the image-side leg is added manually.
    surfs = [_copy.copy(s) for s in surfaces_from_prescription(prescription)]
    surfs[-1].thickness = 0.0
    kw2 = k * w0 * w0
    ray_transfer_jacobian = _pick_ray_transfer(surfs, exact_jacobian)  # lever #1
    from ..raytrace.differential import (
        ray_transfer_jacobian_analytic as _rtja,
    )
    # the FULL-trace path uses the resolved Jacobian (FD 9-ray bundle vs analytic
    # single ray); the coarse path always traces a thinned coarse lattice via FD,
    # bounded separately -- so its per-chunk reconstruction carries no live trace.
    uses_fd = (ray_transfer_jacobian is not _rtja) and int(coarse_stride) == 1

    cw = Np if (chunk is None or int(chunk) <= 0) else min(int(chunk), Np)
    use_sep = bool(separable) if separable != "auto" else (n_p >= 5)
    # mem_budget_mb now bounds BOTH dimensions: momenta via cw AND the POSITION
    # lattice via nq_chunk, so a large aperture runs within the budget as an
    # additive sum over lattice chunks instead of OOMing (audit F3).  The
    # separable c_full is then per-lattice-chunk (nqc*Np), not the whole Nq*Np.
    # fd_bundle sizes the trace term to the ACTUAL Jacobian bundle so the FD path
    # is bounded, not ~10x over budget (H4b).
    nq_chunk = _resolve_nq_chunk(Nq, Np, use_sep, cw, mem_budget_mb,
                                 "apply_real_lens_fga", fd_bundle=uses_fd)
    if coarse_stride and int(coarse_stride) > 1:
        return _fga_coarse(u0, dx, dyg, x0, y0, Ny, Nx, k, w0, nsig, Ag, C, kw2,
                           surfs, wavelength, z_image, dq_step, px, py, n_p, Np,
                           use_sep, cw, nq_chunk, int(coarse_stride),
                           cache_trace=bool(cache_trace), Cp=Cp)
    qbounds = ([(0, Nq)] if nq_chunk is None
               else [(s, min(s + nq_chunk, Nq)) for s in range(0, Nq, nq_chunk)])

    # Coefficient pruning under Nq-chunking needs the GLOBAL per-momentum peak
    # (over ALL q): a per-lattice-chunk peak under-counts and would prune real
    # momenta.  With >1 lattice chunk, a light analysis-only PRE-PASS accumulates
    # that global peak so the prune decision (and the field) matches the
    # un-chunked result to round-off.  Single chunk keeps the running-max path.
    def _chunk_peak(qs, qe):
        cc = (_gabor_coeff_sep(u0, qx[qs:qe], qy[qs:qe], px[:n_p], x0, y0,
                               dx, dyg, w0, k, Ag, nsig) if use_sep else
              _gabor_coeff(u0, qx[qs:qe], qy[qs:qe], px, py, x0, y0, dx, dyg,
                           w0, k, Ag, nsig))
        return np.max(np.abs(cc), axis=0)
    keep_mom = _coeff_keep_mask(qbounds, Np, coeff_frac, _chunk_peak)

    outr = np.zeros((Ny, Nx))
    outi = np.zeros((Ny, Nx))
    for qs, qe in qbounds:                              # POSITION-lattice chunk
        qxc = qx[qs:qe]
        qyc = qy[qs:qe]
        nqc = qe - qs
        c_full_c = (_gabor_coeff_sep(u0, qxc, qyc, px[:n_p], x0, y0, dx, dyg, w0,
                                     k, Ag, nsig) if use_sep else None)
        gmax_c = 0.0                                    # running max (single-chunk)
        for cs in range(0, Np, cw):                     # momentum chunk
            ce = min(cs + cw, Np)
            m = ce - cs
            pxc = px[cs:ce]
            pyc = py[cs:ce]
            if use_sep:
                c = c_full_c[:, cs:ce]
            else:
                c = _gabor_coeff(u0, qxc, qyc, pxc, pyc, x0, y0, dx, dyg, w0, k,
                                 Ag, nsig)
            cmax_p = None
            if coeff_frac > 0.0 and keep_mom is None:
                cmax_p = np.max(np.abs(c), axis=0)
                gmax_c = max(gmax_c, float(cmax_p.max()))
            QX = np.empty((nqc, m))
            QY = np.empty((nqc, m))
            PX = np.empty((nqc, m))
            PY = np.empty((nqc, m))
            AW = np.zeros((nqc, m), dtype=np.complex128)
            ALV = np.zeros((nqc, m), dtype=bool)
            for j in range(m):
                if coeff_frac > 0.0:
                    if keep_mom is not None:
                        if not keep_mom[cs + j]:
                            continue
                    elif cmax_p[j] < coeff_frac * gmax_c:
                        continue                        # negligible momentum
                pxi = float(pxc[j])
                pyi = float(pyc[j])
                pz_in = math.sqrt(max(1.0 - pxi * pxi - pyi * pyi, 1e-12))
                uxin = np.full(nqc, pxi / pz_in)
                uyin = np.full(nqc, pyi / pz_in)
                dt = ray_transfer_jacobian(qxc.copy(), qyc.copy(), uxin, uyin,
                                           surfs, wavelength, per_surface=False)
                uxo = dt.ux
                uyo = dt.uy
                # manual image-side free-space leg z_image (slope coordinates)
                xv = dt.x + z_image * uxo
                yv = dt.y + z_image * uyo
                opd_tot = dt.opd + z_image * np.sqrt(1.0 + uxo ** 2 + uyo ** 2)
                Mleg = np.tile(np.eye(4), (nqc, 1, 1))
                Mleg[:, 0, 2] = z_image
                Mleg[:, 1, 3] = z_image
                M = Mleg @ dt.jacobian
                # slope -> direction-cosine monodromy: SCALAR isotropic part of
                # the true 2x2 d(p)/d(u) Jacobian (O(u^2), NA/skew-limited --
                # see the full note at the ``_fga_coarse._trace`` site, audit
                # S2-16).
                go = 1.0 / (1.0 + uxo ** 2 + uyo ** 2) ** 1.5     # dp/du at output
                gi = (1.0 + (pxi / pz_in) ** 2 + (pyi / pz_in) ** 2) ** 1.5  # du/dp
                a = _hk_prefactor(M, go, gi, kw2)             # continuous branch
                invo = 1.0 / np.sqrt(1.0 + uxo ** 2 + uyo ** 2)
                QX[:, j] = xv
                QY[:, j] = yv
                PX[:, j] = uxo * invo
                PY[:, j] = uyo * invo
                cj = C if Cp is None else Cp[cs + j]   # per-momentum quad weight
                AW[:, j] = cj * a * np.exp(1j * k * opd_tot)
                ALV[:, j] = np.asarray(dt.alive, bool)
            W = c * AW
            W[~ALV] = 0.0
            _reconstruct_into(outr, outi, QX, QY, PX, PY, W, x0, y0, dx, dyg,
                              Ny, Nx, w0, k, Ag, nsig, sep=use_sep)
    return outr + 1j * outi


def apply_real_lens_fga(
    E_in: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    output_plane_distance: float = 0.0,
    w0_factor: float = 5.0,
    dq_step: int = 2,
    p_max: Optional[float] = None,
    n_p: Optional[int] = None,
    nsig: float = 3.0,
    mem_budget_mb: Optional[float] = None,
    chunk: Optional[int] = None,
    prune_frac: float = 1e-4,
    coeff_frac: float = 1e-4,
    separable: Any = "auto",
    coarse_stride: int = 1,
    exact_jacobian: Optional[bool] = None,
    cache_trace: bool = False,
    normalize_output: str = "none",
    momentum_sampling: str = "uniform",
) -> np.ndarray:
    """Propagate ``E_in`` through a real lens ``prescription`` by the
    **Frozen Gaussian Approximation** and return the field at
    ``last-surface-vertex + output_plane_distance`` on the input grid.

    Caustic-accurate peer of :func:`lumenairy.elements.apply_real_lens_gbd`;
    reach for it near foci and fold/cusp caustics where GBD's thawed-beam phase
    mis-renders the interference (see the module docstring).  NumPy only;
    requires the optional ``numba`` accelerator.

    Parameters
    ----------
    E_in : (Ny, Nx) complex ndarray
        Input field.  Rectangular grids and anamorphic (``dx != dy``) pixel
        pitch are supported.
    prescription : dict
        Surface prescription (as consumed by
        :func:`lumenairy.raytrace.surfaces_from_prescription`).
    wavelength, dx : float
        Vacuum wavelength and x grid pitch [m].
    dy : float, optional
        y grid pitch [m] for anamorphic grids; ``None`` (default) uses ``dx``
        (square pixels).  The frozen beamlet stays isotropic with width
        ``w0 = w0_factor * sqrt(dx*dy)``; strong anisotropy (``>~ 3:1``) may need
        a larger ``w0_factor`` to keep the coarse-axis frame well sampled.  The
        momentum swarm is unchanged (``p`` is a physical direction cosine bounded
        by the NA, not by the pixel pitch).
    output_plane_distance : float
        Axial distance past the last surface vertex to evaluate [m].
    w0_factor : float
        Frozen beamlet width in units of the pixel pitch
        (``w0 = w0_factor * sqrt(dx*dy)``).  The FGA convergence parameter:
        larger conserves energy / smooth-field accuracy better, smaller resolves
        finer caustic structure.
    dq_step : int
        Position-lattice stride (beamlet spacing in pixels).
    p_max : float, optional
        Momentum (direction-cosine) half-range of the swarm.  ``None`` (default)
        auto-sizes it to the FIELD's angular content (capped by the system NA) --
        a diverging beam needs its real content covered, a collimated beam ~0.
    n_p : int, optional
        Momentum samples per transverse axis (swarm has ``n_p**2`` directions).
        ``None`` (default) auto-sizes it so the momentum spacing
        ``dp = 2*p_max/(n_p-1)`` is fine enough (``<~ lambda / field-extent``) to
        resolve the phase-space quadrature -- the lever that makes FGA accurate on
        diverging beams (their broad phase-space footprint under-samples at a
        fixed small ``n_p``).  Clamped to ``[7, 61]`` when auto.
    nsig : float
        Frozen-beamlet window radius in sigmas (per-beamlet cost scales as
        ``nsig**2``).  Default ``3.0``: overlapping beamlets fill the >3-sigma
        tail (`exp(-4.5)`), so the reconstruction is unchanged from the old
        ``4.0`` while ~1.8x faster (verified: fidelity and caustic peak-intensity
        error identical).
    mem_budget_mb : float, optional
        Cap peak memory (~MB) by processing the swarm in chunks.  It bounds BOTH
        dimensions: the MOMENTUM swarm (``chunk`` momenta per batch) AND the
        POSITION lattice (``Nq`` = aperture-support / ``dq_step^2``, chunked into
        ``nq_chunk`` blocks).  So a LARGE aperture runs within the budget as an
        additive sum over lattice chunks instead of OOMing -- the separable
        ``(nq_chunk*Np)`` coefficient array, the per-momentum ray trace (sized to
        the actual FD 9-ray / analytic single-ray Jacobian bundle), and the
        scatter are all bounded by ``nq_chunk``.  The chunked result is identical
        to the full swarm to float round-off (it only reorders an additive sum),
        so this is a pure memory lever.  ``None`` (the default) does NOT process
        the whole swarm unbounded: it defaults to ~25% of the available RAM
        (floored at 1000 MB) so the chunk loop engages and a large-N run does not
        OOM; pass an explicit value to pin a tighter budget, or a large one to
        effectively disable chunking.  Only an absurd budget (a single lattice
        point can't fit) raises.  (A 24 mm-aperture FGA is still impractically
        slow -- the ray trace scales with ``Nq`` -- so control ``Nq`` with
        ``dq_step`` / ``prune_frac`` / ``n_p`` and prefer a lighter propagator
        there, but it no longer OOMs.)

        On a SHARED box the 25%-of-RAM default is measured against the WHOLE
        machine, not this process's share, so it can over-commit.  Set the
        ``LUMENAIRY_MEM_BUDGET_MB`` environment variable to pin the default
        budget (MB) process-wide without threading ``mem_budget_mb`` through
        every call site, and call :func:`fga_memory_estimate` to size a run
        (grid floor + swarm peak, chunked and unchunked) before launching it.
    chunk : int, optional
        Explicit momenta-per-batch (overrides ``mem_budget_mb``).
    prune_frac : float
        Drop launch-lattice points whose windowed ``|E_in|`` is below this
        fraction of the peak -- those beamlets carry a negligible Gabor
        coefficient for every momentum (Cauchy-Schwarz), so the reconstruction is
        unchanged.  Default ``1e-4`` (a dropped beamlet contributes ``< 1e-4`` in
        amplitude / ``1e-8`` in energy): 3-5x faster on concentrated fields, a
        no-op on grid-filling ones.  ``0`` disables it.
    coeff_frac : float
        Skip whole momenta whose peak Gabor coefficient ``max_q |c(q, p)|`` is
        below this fraction of the running global peak -- the field carries ~no
        energy at that direction, so its beamlets (the whole ray trace + scatter)
        are dropped.  Default ``1e-4``: faster for spectrally-concentrated
        (smooth) fields, a no-op for broadband ones.  Conservative/no-loss (the
        running global peak only grows, so it never over-prunes).  ``0`` disables.
    separable : {'auto', True, False}
        Use the faster separable analysis (the tensor-``pv (x) pv`` momentum grid
        factorizes the 2-D Gabor transform into an x-transform reused across every
        ``py`` -- ~``n_p`` x on the analysis) and the recurrence scatter (the
        window phase / Gaussian advance by recurrences, hoisting the cos/sin/exp
        out of the inner loop).  Both are numerically equivalent to the direct
        kernels to well within the FGA error floor (each matches the exact oracle
        to the same fidelity), for a ~1.5-1.8x combined speedup.  ``'auto'``
        (default) enables them when ``n_p >= 5``.
    coarse_stride : int
        Opt-in (default ``1`` = off, exact) coarse-lattice trace.  The ray trace
        is the dominant FGA cost and the exit-vertex map ``(q,p) -> (xv,yv,ux,uy,
        opd,a)`` is smooth and single-valued at the launch plane (the caustic
        fold is DOWNSTREAM), so with ``coarse_stride=M`` only a stride-``M``
        position sub-lattice is traced per momentum and the smooth ray quantities
        are cubic-``map_coordinates``-interpolated to the full lattice (``AW`` is
        reconstructed, never interpolated).  No-loss (fidelity ~``1`` vs the full
        trace; validated), with a direct-trace RIM fallback at the vignetting
        boundary.  The interpolation has a fixed per-point overhead, so the NET
        speedup depends on the trace cost: it grows with the aperture (N) and the
        prescription complexity -- worthwhile for LARGE apertures / multi-surface
        / aspheric lenses (measured ~1.2-1.6x at N=256-384, growing with N), and
        marginal or negative for small grids or cheap (near-free-space) traces,
        where the default ``1`` is best.  Position pruning is not applied on this
        path (``M`` already thins the trace).  Use ``M`` ~ ``4-8``.
    exact_jacobian : bool, optional
        ``None`` (the default) AUTO-selects: the truncation-free analytic
        (forward-mode-AD) differential ray-transfer Jacobian for an ALL-CONIC
        prescription (every surface a rotationally-symmetric sphere / conic), the
        finite-difference Jacobian otherwise.  The analytic form is exact vs the
        FD ``~1e-8`` truncation, ~``1.2x`` faster on a large aperture, AND ~``5x``
        lighter (one traced ray vs the FD 9-ray bundle), so an all-conic large-N
        full trace fits a memory budget that the FD path OOMs (H4c).  Pass
        ``True`` to force it (still falls back to FD for aspheric / freeform /
        biconic, which the analytic form does not handle) or ``False`` to force
        the FD Jacobian.  Applies to the full-trace path (``coarse_stride=1``; the
        coarse path uses FD, where the interpolation dominates the accuracy
        budget anyway).
    cache_trace : bool
        Opt-in (default ``False``), effective only with ``coarse_stride > 1``.
        The coarse pre-trace (per-momentum coarse ray-data grids + alive masks)
        is FIELD-INDEPENDENT given the geometry AND the momentum lattice, so
        caching it lets a repeated propagation through the SAME optics skip the
        trace on subsequent calls (lever #2).  Reuse requires the SAME momentum
        lattice: pass EXPLICIT ``p_max`` / ``n_p`` (or vary only the field PHASE),
        since the default auto sampler sizes ``p_max`` from each field's angular
        content -- differing-content fields then get differing swarms and MISS
        the cache.  Bounded, FIFO-evicted cache; each entry is
        the coarse swarm (~ full swarm / ``coarse_stride^2``), so it trades memory
        for the trace time -- worthwhile for repeated same-optics propagations,
        pointless for a one-shot call.  The field-dependent rim is recomputed each
        call, so the result is identical whether cached or not.
    normalize_output : {'none', 'power'}
        ``'none'`` returns the raw FGA field
        ``'power'`` rescales it so the
        output power equals the input power (energy conservation improves with
        ``w0_factor`` -- see the module docstring).  When the auto sampler takes
        the H5 near-collimated content-sizing path (see ``p_max``), power
        normalization is applied AUTOMATICALLY even under ``'none'`` (the
        content-sized swarm is under-complete, so the raw scale is wrong); a
        ``RuntimeWarning`` reports it.
    momentum_sampling : {'uniform', 'adaptive'}
        Momentum-swarm node placement.  ``'uniform'`` (default) is the equally
        spaced ``pv = linspace(-p_max, p_max, n_p)`` grid -- byte-identical to
        prior releases.  ``'adaptive'`` is CONTENT-adaptive: the ``n_p`` momentum
        nodes are IMPORTANCE-sampled by the inverse-CDF of the input field's
        (symmetrized) marginal angular power, concentrating samples where the
        field's spectrum -- hence the reconstruction integrand -- has mass, with
        the matching per-node trapezoidal quadrature weights folded into the
        beamlet normalization.  MEASURED (N6b): a NON-improvement -- uniform
        already converges (the reconstruction integrand is the spectrum broadened
        by the beamlet momentum width, so it is not concentrated enough for
        importance sampling to win), so this is opt-in for a genuinely SPARSE /
        multi-peak spectrum, NOT a general fidelity lever.  The nodes still span
        the full ``[-p_max, p_max]`` (beamlet-completeness coverage preserved).
        Because the non-uniform quadrature carries the correct RELATIVE per-node
        weights (the SHAPE) but not the calibrated uniform absolute scale, the
        adaptive output is AUTO power-normalized to the input power (as the H5 path
        is).

    Returns
    -------
    (Ny, Nx) complex ndarray
        The field at the output plane.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_fga', input_kind='field')
    xp = array_namespace(E_in)
    if xp.__name__ != "numpy":
        raise NotImplementedError(
            "apply_real_lens_fga is NumPy-only (the ray trace + numba swarm "
            "sum). Pass a NumPy array.")
    E_in = np.asarray(E_in, dtype=np.complex128)
    if normalize_output not in ("none", "power"):
        raise ValueError(
            "normalize_output must be 'none' or 'power', got "
            f"{normalize_output!r}.")
    if momentum_sampling not in ("uniform", "adaptive"):
        raise ValueError(
            "momentum_sampling must be 'uniform' or 'adaptive', got "
            f"{momentum_sampling!r}.")
    if mem_budget_mb is None:                       # H4a: engage the chunk loop
        mem_budget_mb = _default_mem_budget_mb()
    dyg = float(dx) if dy is None else float(dy)
    w0 = float(w0_factor) * math.sqrt(float(dx) * dyg)
    _samp = {}
    p_max, n_p = _resolve_sampling(E_in, float(dx), dyg, float(wavelength), w0,
                                   prescription, p_max, n_p, info=_samp)
    nodes = (None if momentum_sampling == "uniform" else
             _adaptive_momentum_nodes(E_in, float(dx), dyg, float(wavelength),
                                      w0, float(p_max), int(n_p)))
    chunk_eff = _chunk_from_budget(E_in.shape, int(dq_step), chunk, mem_budget_mb)
    out = _fga_through_lens(
        E_in, float(dx), dyg, prescription, float(wavelength), w0,
        float(output_plane_distance), int(dq_step), float(p_max), int(n_p),
        float(nsig), chunk=chunk_eff, prune_frac=float(prune_frac),
        coeff_frac=float(coeff_frac), separable=separable,
        mem_budget_mb=mem_budget_mb, coarse_stride=int(coarse_stride),
        exact_jacobian=exact_jacobian, cache_trace=bool(cache_trace),
        momentum_nodes=nodes)
    # H5: the content-sized swarm is intentionally under-complete, so restore the
    # absolute scale by power normalization even under the 'none' default.  The
    # ADAPTIVE swarm likewise does not preserve the calibrated uniform absolute
    # scale (its non-uniform quadrature carries the correct RELATIVE weights for
    # the shape, but the overall FGA power scale is convention-set for the uniform
    # rectangle rule), so it too auto-normalizes -- the reconstruction SHAPE is
    # what the adaptive nodes improve.
    do_norm = (normalize_output == "power" or _samp.get('content_override', False)
               or momentum_sampling == "adaptive")
    if do_norm:
        pin = float(np.sum(np.abs(E_in) ** 2))
        pout = float(np.sum(np.abs(out) ** 2))
        if pout > 0.0:
            out = out * math.sqrt(pin / pout)
    return out


def fga_memory_estimate(
    N: Any,
    dx: float,
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    dy: Optional[float] = None,
    E_in: Optional[np.ndarray] = None,
    p_max: Optional[float] = None,
    n_p: Optional[int] = None,
    dq_step: int = 2,
    w0_factor: float = 5.0,
    nsig: float = 3.0,
    vector: bool = False,
    exact_jacobian: Optional[bool] = None,
    mem_budget_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """Predict FGA peak memory for a run BEFORE launching it (C2 / H4 model).

    Exposes the same memory model the chunk sizer uses (via the shared
    :func:`_fga_lattice_point_bytes`), so a caller can size a large-aperture
    :func:`apply_real_lens_fga` / :func:`apply_real_lens_fga_vector` on a shared
    box -- pick ``dq_step`` / ``n_p`` / an explicit ``mem_budget_mb`` (or the
    ``LUMENAIRY_MEM_BUDGET_MB`` env override) -- instead of discovering an OOM at
    runtime (the H8 agent had to reverse-engineer these numbers).

    Three memory pools:

    * a FIXED full-grid floor (``grid_arrays_mb``) -- the input field, the
      ``outr``/``outi`` accumulators and the complex output; it does NOT shrink
      with the swarm chunking and is not bounded by ``mem_budget_mb``;
    * a FIXED Gabor-window pool (``gabor_window_mb``, v5.30 / audit P9) -- the
      shared separable window + phase tables and the per-numba-worker inner-x
      workspace, both sized by the window half-width ``int(nsig*w0/dx) + 1``.
      This is the only term ``nsig`` moves, and it is why ``nsig`` is read at
      all (see below).  Zero on the direct kernel (``n_p < 5``), which
      accumulates into scalars;
    * the beamlet SWARM (``swarm_peak_mb``) -- ``Nq`` position-lattice points x
      ``Np = n_p**2`` momenta; THIS is what ``mem_budget_mb`` bounds by chunking
      the ``Nq`` lattice into ``nq_chunk`` blocks, giving ``chunked_swarm_peak_mb``.

    ``peak_unchunked_mb`` / ``peak_chunked_mb`` are the two fixed pools plus the
    corresponding swarm term.

    Parameters
    ----------
    N : int or (Ny, Nx)
        Grid size (square when an int).  Ignored (and overridden) when ``E_in``
        is given.
    dx, wavelength : float
        Grid pitch and vacuum wavelength [m].
    prescription : dict
        Same surface prescription ``apply_real_lens_fga`` consumes (used for the
        NA cap and the analytic-vs-FD Jacobian selection).
    dy : float, optional
        y pitch for anamorphic grids; ``None`` -> ``dx``.
    E_in : ndarray, optional
        The actual input field.  When given, ``p_max`` / ``n_p`` are auto-sized
        from its true angular content (the accurate estimate); its shape sets
        ``N``.  When omitted, the auto momentum sizing uses the system-NA cone as
        an UPPER BOUND (``sampling_basis='na_cap_upper_bound'``) -- a diverging
        field's real ``p_max`` never exceeds it, so the estimate is conservative.
    p_max, n_p : optional
        Explicit swarm sampling (bypasses auto-sizing), matching the same kwargs
        on ``apply_real_lens_fga``.
    dq_step, w0_factor, nsig, exact_jacobian, mem_budget_mb :
        As on :func:`apply_real_lens_fga`.  ``mem_budget_mb=None`` uses the
        library default (env override or 25%-of-RAM).  ``nsig`` sizes the
        Gabor-window pool (``gabor_window_mb``) -- see the pool list above.
    vector : bool, default False
        Size the vector (``apply_real_lens_fga_vector``) path: two input fields
        and a ``cfull_mult=2`` coefficient array.

    Returns
    -------
    dict
        ``Ny, Nx, dx, dy, p_max, n_p, Np, dq_step, Nq_full, use_separable,
        exact_jacobian, bytes_per_lattice_point, swarm_peak_mb,
        chunked_swarm_peak_mb, grid_arrays_mb, nsig, gabor_window_mb,
        peak_unchunked_mb, peak_chunked_mb, budget_mb, nq_chunk,
        swarm_fits_budget, sampling_basis, n_p_target``.
        ``Nq_full`` is the UN-pruned lattice
        (upper bound; ``prune_frac`` reduces it for a concentrated field), and
        ``n_p_target`` (when set) is the un-clamped ``n_p`` that would meet the
        ``dp`` target -- the memory of that finer swarm scales by
        ``(n_p_target / n_p)**2``.

    .. versionchanged:: 5.30
        ``nsig`` is now READ (audit P9).  Pre-v5.30 it was documented as
        "as on :func:`apply_real_lens_fga`" but never consulted, so the
        whole estimate was bit-identical for ``nsig`` = 1.0 and 12.0 while
        the real path reads it six times.  It now sizes the new
        ``gabor_window_mb`` pool, and ``nsig`` itself is echoed back in the
        result dict.  Every other key is unchanged;
        ``peak_unchunked_mb`` / ``peak_chunked_mb`` grow by
        ``gabor_window_mb``.
    """
    from ..raytrace import surfaces_from_prescription

    if E_in is not None:
        Ny, Nx = int(np.shape(E_in)[0]), int(np.shape(E_in)[1])
    elif isinstance(N, (tuple, list)) and len(N) == 2:
        Ny, Nx = int(N[0]), int(N[1])
    else:
        Ny = Nx = int(N)
    dyg = float(dx) if dy is None else float(dy)
    w0 = float(w0_factor) * math.sqrt(float(dx) * dyg)
    k = 2.0 * np.pi / float(wavelength)

    # ---- resolve the momentum swarm (p_max, n_p) -----------------------
    n_p_target = None
    if p_max is not None and n_p is not None:
        p_max_r, n_p_r = float(p_max), int(n_p)
        sampling_basis = 'explicit'
    elif E_in is not None:
        _info: Dict[str, Any] = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')      # sizing call: no side warnings
            p_max_r, n_p_r = _resolve_sampling(
                np.asarray(E_in, dtype=np.complex128), float(dx), dyg,
                float(wavelength), w0, prescription, p_max, n_p, info=_info)
        n_p_target = _info.get('n_p_target')
        sampling_basis = 'field_content'
    else:
        # No field: size p_max from the system-NA cone (x1.5 tail) as an upper
        # bound, floored at the beamlet-completeness width, then n_p to the dp
        # target -- the same arithmetic _resolve_sampling uses, minus the
        # (field-dependent) content term.
        na_cap = _default_p_max(prescription, float(wavelength),
                                powerless_uncapped=True)
        beamlet_floor = _PMAX_BEAMLET_C / (k * w0)
        if p_max is not None:
            p_max_r = float(p_max)
        elif np.isfinite(na_cap):
            p_max_r = float(max(beamlet_floor, 1.5 * na_cap))
        else:
            p_max_r = float(beamlet_floor)
        if n_p is not None:
            n_p_r = int(n_p)
        else:
            n_p_want = int(np.ceil(2.0 * p_max_r / _DP_TARGET)) | 1
            n_p_r = int(min(61, max(7, n_p_want)))
            if n_p_r < n_p_want:
                n_p_target = int(n_p_want)
        sampling_basis = 'na_cap_upper_bound'

    Np = int(n_p_r) * int(n_p_r)
    use_sep = n_p_r >= 5
    cfull_mult = 2.0 if vector else 1.0

    # ---- analytic (single-ray) vs FD (9-ray bundle) Jacobian -----------
    surfs = surfaces_from_prescription(prescription)
    all_conic = _is_all_conic(surfs)
    exact = all_conic if exact_jacobian is None else bool(exact_jacobian)
    fd_bundle = not (exact and all_conic)

    # ---- lattice + per-point memory ------------------------------------
    Nq_full = (len(range(0, Nx, int(dq_step)))
               * len(range(0, Ny, int(dq_step))))
    per = _fga_lattice_point_bytes(Np, Np, use_sep, cfull_mult, fd_bundle)
    swarm_peak_mb = Nq_full * per / 1e6
    grid_bytes = _FGA_GRID_BYTES_PER_PX_SCALAR * (2.0 if vector else 1.0)
    grid_arrays_mb = grid_bytes * float(Ny) * float(Nx) / 1e6

    # ---- audit P9: the nsig-sized Gabor-window pool ---------------------
    # ``nsig`` was documented here but never read, while ``_fga_coarse`` /
    # ``_gabor_coeff*`` / ``_lattice_support_mask`` read it 6x -- so the
    # estimate was bit-identical for nsig = 1 and nsig = 12 (measured).
    # This is a third FIXED pool alongside the grid floor: it does not
    # shrink with the lattice chunking and is not bounded by
    # ``mem_budget_mb`` (the chunk sizer bounds the swarm only).
    gabor_window_mb = _fga_gabor_window_bytes(
        float(nsig), w0, float(dx), dyg, int(n_p_r), use_sep,
        cfull_mult=cfull_mult) / 1e6
    fixed_mb = grid_arrays_mb + gabor_window_mb

    budget = (float(mem_budget_mb) if mem_budget_mb is not None
              else _default_mem_budget_mb())
    nq_chunk = _resolve_nq_chunk(Nq_full, Np, use_sep, Np, budget,
                                 'fga_memory_estimate', cfull_mult=cfull_mult,
                                 fd_bundle=fd_bundle)
    chunk_nq = Nq_full if nq_chunk is None else int(nq_chunk)
    chunked_swarm_peak_mb = chunk_nq * per / 1e6

    return {
        'Ny': Ny, 'Nx': Nx, 'dx': float(dx), 'dy': dyg,
        'p_max': float(p_max_r), 'n_p': int(n_p_r), 'Np': int(Np),
        'dq_step': int(dq_step), 'Nq_full': int(Nq_full),
        'use_separable': bool(use_sep), 'exact_jacobian': bool(exact),
        'bytes_per_lattice_point': float(per),
        'swarm_peak_mb': float(swarm_peak_mb),
        'chunked_swarm_peak_mb': float(chunked_swarm_peak_mb),
        'grid_arrays_mb': float(grid_arrays_mb),
        'nsig': float(nsig),
        'gabor_window_mb': float(gabor_window_mb),
        'peak_unchunked_mb': float(fixed_mb + swarm_peak_mb),
        'peak_chunked_mb': float(fixed_mb + chunked_swarm_peak_mb),
        'budget_mb': float(budget),
        'nq_chunk': (None if nq_chunk is None else int(nq_chunk)),
        'swarm_fits_budget': bool(nq_chunk is None),
        'sampling_basis': sampling_basis,
        'n_p_target': (None if n_p_target is None else int(n_p_target)),
    }


def _fga_vector_through_lens(Ex, Ey, dx, dyg, prescription, wavelength, w0,
                             z_image, dq_step, p_max, n_p, nsig, chunk=None,
                             prune_frac=0.0, coeff_frac=0.0, separable="auto",
                             mem_budget_mb=None, momentum_nodes=None):
    """Vector (Jones) FGA transport: returns ``(Ex, Ey, Ez)`` at the output
    plane.  The scalar transport (ray map + HK weight + OPL) is shared by both
    polarization channels; each beamlet additionally carries the per-beamlet 2x2
    Fresnel Jones matrix (polarization ray tracing, s/p per surface -- the s/p
    frame rotation IS the geometric phase), and the longitudinal ``Ez`` is added
    from the exit-ray directions (``E.k = 0``).  ``dx`` / ``dyg`` are the
    (possibly anamorphic) x / y pixel pitches.  ``separable`` (True/False/'auto')
    uses the faster ULP-identical separable analysis + recurrence scatter."""
    from ..propagators.gbd import _fresnel_jones_matrix_per_beamlet
    from ..raytrace import surfaces_from_prescription
    from ..raytrace.differential import ray_transfer_jacobian

    k = 2.0 * np.pi / wavelength
    Ny, Nx = Ex.shape
    x0 = -(Nx / 2) * dx
    y0 = -(Ny / 2) * dyg
    Ag = (1.0 / (np.pi * w0 ** 2)) ** 0.5
    qx, qy, px, py, dp, wp = _swarm_lattice(Ny, Nx, dx, dyg, x0, y0, dq_step,
                                            p_max, n_p, nodes=momentum_nodes)
    if prune_frac > 0.0:
        supp = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2)
        keep = _lattice_support_mask(supp, dx, dyg, dq_step, w0, nsig,
                                     prune_frac)
        qx = qx[keep]
        qy = qy[keep]
    Nq = qx.shape[0]
    Np = px.shape[0]
    C = ((k / (2.0 * np.pi)) ** 2 * (dq_step ** 2 * dx * dyg) * (dp ** 2)) / 2.0
    Cp = (None if wp is None else
          ((k / (2.0 * np.pi)) ** 2 * (dq_step ** 2 * dx * dyg)) / 2.0 * wp)

    surfs = [_copy.copy(s) for s in surfaces_from_prescription(prescription)]
    surfs[-1].thickness = 0.0
    kw2 = k * w0 * w0

    cw = Np if (chunk is None or int(chunk) <= 0) else min(int(chunk), Np)
    use_sep = bool(separable) if separable != "auto" else (n_p >= 5)
    # Nq-chunking bounds mem_budget_mb (vector holds TWO whole-grid coeff arrays
    # cx + cy -> cfull_mult=2); additive over lattice chunks (audit F3).
    nq_chunk = _resolve_nq_chunk(Nq, Np, use_sep, cw, mem_budget_mb,
                                 "apply_real_lens_fga_vector", cfull_mult=2.0)
    qbounds = ([(0, Nq)] if nq_chunk is None
               else [(s, min(s + nq_chunk, Nq)) for s in range(0, Nq, nq_chunk)])

    def _cxy(qxc, qyc, pxc, pyc, full):
        # separable ALWAYS computes the whole tensor grid (pv = px[:n_p]); direct
        # takes the full 2-D grid (px, py) for the pre-pass or the chunk otherwise.
        if use_sep:
            return (_gabor_coeff_sep(Ex, qxc, qyc, px[:n_p], x0, y0, dx, dyg, w0,
                                     k, Ag, nsig),
                    _gabor_coeff_sep(Ey, qxc, qyc, px[:n_p], x0, y0, dx, dyg, w0,
                                     k, Ag, nsig))
        pa, pb = (px, py) if full else (pxc, pyc)
        return (_gabor_coeff(Ex, qxc, qyc, pa, pb, x0, y0, dx, dyg, w0, k, Ag,
                             nsig),
                _gabor_coeff(Ey, qxc, qyc, pa, pb, x0, y0, dx, dyg, w0, k, Ag,
                             nsig))

    # combined-|cx,cy| global per-momentum peak pre-pass (see the scalar path)
    def _chunk_peak(qs, qe):
        cxc, cyc = _cxy(qx[qs:qe], qy[qs:qe], px, py, True)
        return np.max(np.sqrt(np.abs(cxc) ** 2 + np.abs(cyc) ** 2), axis=0)
    keep_mom = _coeff_keep_mask(qbounds, Np, coeff_frac, _chunk_peak)

    exr = np.zeros((Ny, Nx))
    exi = np.zeros((Ny, Nx))
    eyr = np.zeros((Ny, Nx))
    eyi = np.zeros((Ny, Nx))
    ezr = np.zeros((Ny, Nx))
    ezi = np.zeros((Ny, Nx))
    for qs, qe in qbounds:                              # POSITION-lattice chunk
        qxc0 = qx[qs:qe]
        qyc0 = qy[qs:qe]
        nqc = qe - qs
        cx_full_c, cy_full_c = _cxy(qxc0, qyc0, px, py, True) if use_sep \
            else (None, None)
        gmax_c = 0.0
        for cs in range(0, Np, cw):                     # momentum chunk
            ce = min(cs + cw, Np)
            m = ce - cs
            pxc = px[cs:ce]
            pyc = py[cs:ce]
            if use_sep:
                cx = cx_full_c[:, cs:ce]
                cy = cy_full_c[:, cs:ce]
            else:
                cx, cy = _cxy(qxc0, qyc0, pxc, pyc, False)
            cmax_p = None
            if coeff_frac > 0.0 and keep_mom is None:
                cmax_p = np.max(np.sqrt(np.abs(cx) ** 2 + np.abs(cy) ** 2),
                                axis=0)
                gmax_c = max(gmax_c, float(cmax_p.max()))
            QX = np.empty((nqc, m))
            QY = np.empty((nqc, m))
            # PX/PY zero-init: coeff-pruned (skipped) columns must stay FINITE so
            # Wz = -(PX*Wx + PY*Wy)/PZ is 0 (Wx=Wy=0 there) not NaN from inf*0.
            PX = np.zeros((nqc, m))
            PY = np.zeros((nqc, m))
            Wx = np.zeros((nqc, m), dtype=np.complex128)
            Wy = np.zeros((nqc, m), dtype=np.complex128)
            for j in range(m):
                if coeff_frac > 0.0:
                    if keep_mom is not None:
                        if not keep_mom[cs + j]:
                            continue
                    elif cmax_p[j] < coeff_frac * gmax_c:
                        continue
                pxi = float(pxc[j])
                pyi = float(pyc[j])
                pz_in = math.sqrt(max(1.0 - pxi * pxi - pyi * pyi, 1e-12))
                uxin = np.full(nqc, pxi / pz_in)
                uyin = np.full(nqc, pyi / pz_in)
                dt = ray_transfer_jacobian(qxc0.copy(), qyc0.copy(), uxin, uyin,
                                           surfs, wavelength, per_surface=False)
                J, jalive = _fresnel_jones_matrix_per_beamlet(
                    qxc0.copy(), qyc0.copy(), uxin, uyin, prescription,
                    wavelength)
                uxo = dt.ux
                uyo = dt.uy
                xv = dt.x + z_image * uxo
                yv = dt.y + z_image * uyo
                opd_tot = dt.opd + z_image * np.sqrt(1.0 + uxo ** 2 + uyo ** 2)
                Mleg = np.tile(np.eye(4), (nqc, 1, 1))
                Mleg[:, 0, 2] = z_image
                Mleg[:, 1, 3] = z_image
                M = Mleg @ dt.jacobian
                # slope -> direction-cosine monodromy: SCALAR isotropic part of
                # the true 2x2 d(p)/d(u) Jacobian (O(u^2), NA/skew-limited --
                # see the full note at the ``_fga_coarse._trace`` site, audit
                # S2-16).
                go = 1.0 / (1.0 + uxo ** 2 + uyo ** 2) ** 1.5
                gi = (1.0 + (pxi / pz_in) ** 2 + (pyi / pz_in) ** 2) ** 1.5
                a = _hk_prefactor(M, go, gi, kw2)
                cj = C if Cp is None else Cp[cs + j]     # per-momentum quad weight
                base = cj * a * np.exp(1j * k * opd_tot)  # scalar beamlet weight
                invo = 1.0 / np.sqrt(1.0 + uxo ** 2 + uyo ** 2)
                QX[:, j] = xv
                QY[:, j] = yv
                PX[:, j] = uxo * invo
                PY[:, j] = uyo * invo
                cxi = cx[:, j]
                cyi = cy[:, j]
                # apply the 2x2 Jones to (Ex, Ey), weight by the scalar amplitude
                ex_out = J[:, 0, 0] * cxi + J[:, 0, 1] * cyi
                ey_out = J[:, 1, 0] * cxi + J[:, 1, 1] * cyi
                alv = np.asarray(dt.alive, bool) & np.asarray(jalive, bool)
                Wx[:, j] = np.where(alv, base * ex_out, 0.0)
                Wy[:, j] = np.where(alv, base * ey_out, 0.0)
            # longitudinal Ez per beamlet: E.k = 0 -> Ez = -(px*Ex + py*Ey)/pz
            PZ = np.sqrt(np.maximum(1.0 - PX ** 2 - PY ** 2, 1e-12))
            Wz = -(PX * Wx + PY * Wy) / PZ
            _reconstruct_into(exr, exi, QX, QY, PX, PY, Wx, x0, y0, dx, dyg,
                              Ny, Nx, w0, k, Ag, nsig, sep=use_sep)
            _reconstruct_into(eyr, eyi, QX, QY, PX, PY, Wy, x0, y0, dx, dyg,
                              Ny, Nx, w0, k, Ag, nsig, sep=use_sep)
            _reconstruct_into(ezr, ezi, QX, QY, PX, PY, Wz, x0, y0, dx, dyg,
                              Ny, Nx, w0, k, Ag, nsig, sep=use_sep)
    return exr + 1j * exi, eyr + 1j * eyi, ezr + 1j * ezi


def apply_real_lens_fga_vector(
    E_vec: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    output_plane_distance: float = 0.0,
    w0_factor: float = 5.0,
    dq_step: int = 2,
    p_max: Optional[float] = None,
    n_p: Optional[int] = None,
    nsig: float = 3.0,
    mem_budget_mb: Optional[float] = None,
    chunk: Optional[int] = None,
    prune_frac: float = 1e-4,
    coeff_frac: float = 1e-4,
    separable: Any = "auto",
    return_longitudinal: bool = False,
    normalize_output: str = "none",
    momentum_sampling: str = "uniform",
) -> np.ndarray:
    """Vector (Jones) FGA lens propagator -- the polarization-carrying,
    caustic-accurate peer of :func:`apply_real_lens_fga`.

    ``E_vec`` is a ``(2, Ny, Nx)`` transverse Jones field ``(E_x, E_y)``.  Each
    frozen beamlet carries the per-surface Fresnel s/p Jones matrix (polarization
    ray tracing, exactly as the GBD vector propagator), so the returned field
    captures diattenuation, retardance, and the geometric s/p frame rotation
    through the system, while remaining caustic-accurate.

    Returns ``(2, Ny, Nx)`` ``(E_x, E_y)`` by default, or ``(3, Ny, Nx)``
    ``(E_x, E_y, E_z)`` when ``return_longitudinal=True`` -- the longitudinal
    ``E_z`` (from ``E . k = 0``) is the high-NA piece a transverse-only model
    misses.  NumPy-only; requires ``numba``.  Other parameters as
    :func:`apply_real_lens_fga`.
    """
    E_vec = np.asarray(E_vec, dtype=np.complex128)
    if E_vec.ndim != 3 or E_vec.shape[0] != 2:
        raise ValueError(
            "apply_real_lens_fga_vector: E_vec must be (2, Ny, Nx) = (Ex, Ey).")
    if normalize_output not in ("none", "power"):
        raise ValueError(
            "normalize_output must be 'none' or 'power', got "
            f"{normalize_output!r}.")
    if momentum_sampling not in ("uniform", "adaptive"):
        raise ValueError(
            "momentum_sampling must be 'uniform' or 'adaptive', got "
            f"{momentum_sampling!r}.")
    if mem_budget_mb is None:                       # H4a: engage the chunk loop
        mem_budget_mb = _default_mem_budget_mb()
    dyg = float(dx) if dy is None else float(dy)
    # auto-size the swarm from the higher-energy Jones component (both share the
    # beam geometry, so its extent/angular-content set the sampling for both)
    _rep = E_vec[0] if (np.sum(np.abs(E_vec[0]) ** 2)
                        >= np.sum(np.abs(E_vec[1]) ** 2)) else E_vec[1]
    w0 = float(w0_factor) * math.sqrt(float(dx) * dyg)
    _samp = {}
    p_max, n_p = _resolve_sampling(_rep, float(dx), dyg, float(wavelength), w0,
                                   prescription, p_max, n_p, info=_samp)
    nodes = (None if momentum_sampling == "uniform" else
             _adaptive_momentum_nodes(_rep, float(dx), dyg, float(wavelength),
                                      w0, float(p_max), int(n_p)))
    chunk_eff = _chunk_from_budget(E_vec[0].shape, int(dq_step), chunk,
                                   mem_budget_mb)
    ex, ey, ez = _fga_vector_through_lens(
        E_vec[0], E_vec[1], float(dx), dyg, prescription, float(wavelength), w0,
        float(output_plane_distance), int(dq_step), float(p_max), int(n_p),
        float(nsig), chunk=chunk_eff, prune_frac=float(prune_frac),
        coeff_frac=float(coeff_frac), separable=separable,
        mem_budget_mb=mem_budget_mb, momentum_nodes=nodes)
    # H5: the content-sized swarm (near-collimated auto path) is under-complete,
    # so restore the absolute scale by power normalization even under 'none'.  The
    # adaptive swarm's non-uniform quadrature likewise does not preserve the
    # calibrated uniform absolute scale (correct RELATIVE weights = shape), so it
    # too auto-normalizes.
    do_norm = (normalize_output == "power" or _samp.get('content_override', False)
               or momentum_sampling == "adaptive")
    if do_norm:
        # rescale the transverse (Ex, Ey) to the input power (lossless
        # assumption; the raw FGA scale is shape-correct but w0/sampling-
        # dependent -- see the module docstring on the FGA convergence knob).
        pin = float(np.sum(np.abs(E_vec[0]) ** 2 + np.abs(E_vec[1]) ** 2))
        pout = float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2))
        if pout > 0.0:
            s = math.sqrt(pin / pout)
            ex, ey, ez = ex * s, ey * s, ez * s
    if return_longitudinal:
        return np.stack([ex, ey, ez], axis=0)
    return np.stack([ex, ey], axis=0)


def _caustic_zone(E_in, dx, prescription, wavelength, n_rays=25):
    """Geometric-caustic axial extent [z_near, z_far] PAST the last vertex, or
    ``None`` if the field is not converging to a caustic.

    Traces a meridional ray fan whose launch DIRECTIONS follow the input field's
    local wavefront (phase gradient), so a converging/diverging input is handled
    -- not just the system's collimated focus.  Each converging exit ray crosses
    the axis at ``z = -x_exit / u_exit``; the 5th-95th percentile spread of those
    crossings is the (spherical-aberration-broadened) caustic zone.
    """
    from ..raytrace import surfaces_from_prescription
    from ..raytrace.differential import ray_transfer_jacobian

    N = E_in.shape[-1]
    k = 2.0 * np.pi / wavelength
    cx = N // 2                       # column (x) centre -- for the x grid
    cy = E_in.shape[0] // 2           # row (y) centre -- THE meridional row
    xgrid = (np.arange(N) - cx) * dx
    row = E_in[cy, :]                 # audit S2-21: was E_in[cx,:] (wrong row / an
    #                                   IndexError when Nx//2 >= Ny for a tall field)
    amp = np.abs(row)
    if amp.max() <= 0.0:
        return None
    # S11-6f NOT-CHANGED (AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1): the
    # caustic zone below is the 5th-95th percentile of the axial crossings of
    # an UNWEIGHTED, equally-radius-spaced meridional fan, so it weights every
    # pupil zone alike rather than by annular area (2*pi*r*dr) or by the local
    # |E|^2.  An area/amplitude-weighted percentile is the physically better
    # metric but is NOT bit-identical on any input, so it is left for a
    # deliberate metric-definition pass rather than folded into a hygiene fix.
    # local slope u = (1/k) d(arg E)/dx on the illuminated support (+x half)
    phase = np.unwrap(np.angle(np.where(amp > 1e-6 * amp.max(), row, 1.0)))
    slope = np.gradient(phase, dx) / k
    half = slice(cx, N)
    xs_h = xgrid[half]
    amp_h = amp[half]
    sl_h = slope[half]
    good = amp_h > 0.05 * amp_h.max()
    if good.sum() < 3:
        return None
    rr = np.linspace(xs_h[good][0], xs_h[good][-1], n_rays)
    u_in = np.interp(rr, xs_h[good], sl_h[good])
    surfs = [_copy.copy(s) for s in surfaces_from_prescription(prescription)]
    surfs[-1].thickness = 0.0
    zeros = np.zeros_like(rr)
    dt = ray_transfer_jacobian(rr, zeros, u_in, zeros, surfs, wavelength,
                               per_surface=False)
    xo, uo, alive = dt.x, dt.ux, np.asarray(dt.alive, bool)
    conv = alive & (xo * uo < 0.0) & (np.abs(uo) > 1e-9)
    if conv.sum() < max(3, n_rays // 4):        # not meaningfully converging
        return None
    zf = -xo[conv] / uo[conv]
    zf = zf[zf > 0.0]
    if zf.size < 3:
        return None
    return float(np.percentile(zf, 5)), float(np.percentile(zf, 95))


def apply_real_lens_auto(
    E_in: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    output_plane_distance: float = 0.0,
    method: str = "auto",
    return_method: bool = False,
    caustic_pad_dof: float = 3.0,
    fga_kwargs: Optional[Dict[str, Any]] = None,
    gbd_kwargs: Optional[Dict[str, Any]] = None,
):
    """GBD / FGA auto-dispatching lens propagator.

    ``method='auto'`` (default) routes the OUTPUT PLANE to the right beamlet
    method: the fast thawed-beamlet :func:`apply_real_lens_gbd` in smooth
    regions, and the caustic-accurate frozen-beamlet
    :func:`apply_real_lens_fga` when the plane lies inside (or within a
    depth-of-focus pad of) the field's geometric caustic zone -- i.e. near a
    focus / fold / cusp, where GBD's thawed-beam phase mis-renders the
    interference.  Both are ray-based (no thin-screen obliquity ceiling), so the
    dispatched result is high-NA-accurate as well as caustic-accurate.  Because
    FGA matches GBD in smooth regions, the dispatch is a SPEED choice biased
    toward accuracy: when the caustic detector is uncertain it prefers FGA.

    .. note::
       :func:`apply_real_lens_universal` is the **canonical** dispatcher -- a
       superset that also routes low-NA planes to the wave-exact phase-screen and
       high-NA collimated planes to the sub-nm ``traced`` OPL.  This 2-way GBD/FGA
       router is the beamlet-only subset (both members launch beamlets along the
       local phase gradient, so BOTH already handle a single-valued diverging beam
       -- unlike bare ``traced``, which is why ``universal`` needs its collimation
       split but this router does not).  Prefer ``universal`` unless you
       specifically want the GBD-vs-FGA choice.

       This 2-way router never selects the analytic phase-screen model (only
       ``gbd`` / ``fga``, both ray-based with no thin-screen obliquity ceiling), so
       ``universal``'s H2 sag-screen aberration gate does not apply here -- there
       is no analytic route to steer away from.

    ``method='gbd'`` / ``'fga'`` force the choice.  ``return_method=True`` also
    returns the ``'gbd'``/``'fga'`` string actually used.  ``fga_kwargs`` /
    ``gbd_kwargs`` forward extra arguments to the chosen propagator (e.g.
    ``fga_kwargs={'w0_factor': 4.0, 'n_p': 21}``).  ``dy`` (anamorphic y pitch)
    is forwarded to whichever propagator is chosen.

    Notes
    -----
    The caustic zone is detected geometrically (a meridional ray fan whose
    directions follow the input wavefront); ``caustic_pad_dof`` widens it by that
    many diffraction depth-of-focus units ``lambda / NA^2`` so the frozen method
    also covers the wave boundary layer around the geometric caustic.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_auto', input_kind='field')
    if method not in ("auto", "gbd", "fga"):
        raise ValueError(f"method must be 'auto', 'gbd' or 'fga', got {method!r}.")
    fga_kwargs = dict(fga_kwargs or {})
    gbd_kwargs = dict(gbd_kwargs or {})

    chosen = method
    if method == "auto":
        chosen = "gbd"
        zone = _caustic_zone(np.asarray(E_in), float(dx), prescription,
                             float(wavelength))
        if zone is not None:
            z_near, z_far = zone
            # depth-of-focus pad from the marginal NA (aperture / focal distance)
            na = _default_p_max(prescription, wavelength) / 1.6   # undo the pad
            na = max(na, 1e-3)
            dof = float(wavelength) / (na * na)
            pad = caustic_pad_dof * dof
            if (z_near - pad) <= float(output_plane_distance) <= (z_far + pad):
                chosen = "fga"

    if chosen == "fga":
        out = apply_real_lens_fga(
            E_in, prescription=prescription, wavelength=wavelength, dx=dx, dy=dy,
            output_plane_distance=output_plane_distance, **fga_kwargs)
    else:
        from ..elements import apply_real_lens_gbd  # lazy: avoid import cycle
        out = apply_real_lens_gbd(
            E_in, prescription=prescription, wavelength=wavelength, dx=dx, dy=dy,
            output_plane_distance=output_plane_distance, **gbd_kwargs)
    return (out, chosen) if return_method else out


def _system_na(prescription, wavelength):
    """Marginal-ray NA estimate ~ aperture-radius / effective-focal-length."""
    return max(_default_p_max(prescription, wavelength) / 1.6, 1e-3)


def _tilt_dispersion(E_in, dx, dyg, wavelength, na):
    """Multi-valuedness score: the amplitude-weighted RMS spread of the LOCAL
    wavevector about its per-region mean, normalized by the system NA.

    A *single-valued* field has ONE well-defined ray direction at every pixel
    (plane wave, Gaussian, a single diverging/converging point source, an
    MLA-tilted beamlet, ANY smooth aberrated single beam): its local wavevector
    ``k_local = grad(arg E)/k0`` is a smooth function of position, so it equals
    its own local (few-pixel) mean and the score is ~0 (empirically <0.006 even
    for strong spherical/coma/astigmatism, strong divergence, and 6x6-MLA fields).
    A *multi-valued* field -- several wave components crossing the same region
    (multi-emitter, post-DOE diffraction orders, speckle) -- has NO single local
    direction: ``grad(arg E)`` swings pixel-to-pixel across the interference
    fringes while the mean sits near the (amplitude-weighted) average, so the
    residual is a real fraction of NA and the score is ~0.09-0.4 (a >10x
    separation from the single-valued ceiling).

    Note this is a LOCAL residual about the LOCAL mean, NOT the global
    ``|mean tilt|/rms tilt`` "coherence" -- a radially symmetric single
    diverging/converging beam has +/- local tilts that cancel in the global mean
    (coherence ~0) yet is perfectly single-valued (local residual ~0), so the
    global coherence would misclassify it; this local measure does not.

    It is exactly the quantity :func:`apply_real_lens_traced`'s input-aware ray
    launch smooths away -- i.e. it directly measures how far traced's
    single-direction-per-pixel model is from valid.  When it is large, traced
    silently collapses the crossing components to their mean direction and
    applies the wrong angle-dependent OPD; the dispatcher then prefers FGA, whose
    phase-space swarm transports every direction independently.

    Returns 0.0 for an empty/zero field.  The local mean uses an
    amplitude-weighted Gaussian (sigma ~2 px) so low-amplitude pixels (fringe
    nulls, gaps between orders) can't drag the reading toward their noisy phase.
    """
    from scipy.ndimage import gaussian_filter
    E = np.asarray(E_in)
    a2 = np.abs(E) ** 2
    tot = float(a2.sum())
    if not np.isfinite(tot) or tot <= 0.0:
        return 0.0
    k0 = 2.0 * np.pi / float(wavelength)
    # per-pixel local wavevector via the conjugate-product trick (wraps once to
    # (-pi, pi] without a global unwrap); direction cosine p = d(arg E)/(k0*d).
    px = np.zeros_like(a2)
    py = np.zeros_like(a2)
    px[:, :-1] = np.angle(E[:, 1:] * np.conj(E[:, :-1])) / (k0 * float(dx))
    py[:-1, :] = np.angle(E[1:, :] * np.conj(E[:-1, :])) / (k0 * float(dyg))
    sig = 2.0

    def _wmean(f):                      # amplitude-weighted local Gaussian mean
        num = gaussian_filter(a2 * f, sig, mode='nearest')
        den = gaussian_filter(a2, sig, mode='nearest')
        return num / np.maximum(den, 1e-300)

    dpx = px - _wmean(px)
    dpy = py - _wmean(py)
    var = float(np.sum(a2 * (dpx * dpx + dpy * dpy)) / tot)
    rms = np.sqrt(max(var, 0.0))
    return rms / max(float(na), 1e-6)


# H2 aberration gate (hammer audit, docs/audit_real_lens_hammer_2026_07_19.md).
# The analytic sag-screen model ('phase_screen') is a per-surface plane-projection
# phase mask; it structurally cannot represent the transverse ray-displacement of
# high-order / orientation-dependent aberration, so its absolute spot size / EE
# drift once the per-surface spherical term grows (H2: the f/5 dual-oracle singlet
# converged to 40.5 um vs the 65 um truth; PCX orientation errs in BOTH directions,
# 60.4/60.9 um where truth is 43/128).  The dispatcher estimates that error cheaply
# at routing time and steers a heavily-aberrated prescription AWAY from
# 'phase_screen' (to the ray-based traced / fga members) even at low NA, and warns
# when 'phase_screen' is explicitly forced outside its envelope.
_ABERRATION_MAX_RAD = 2.0   # sag-screen wavefront-error budget (rad); see below


def _input_marginal_slope(E, a2, X, Y, dx, dyg, wavelength, r_beam):
    """Signed paraxial marginal-ray slope ``u_in`` (rad) of the input carrier at
    the first surface, from a least-squares fit of the field's local radial
    wavevector ``u_r(rho) = (1/R) * rho``::

        1/R  = sum(a2 * (px*X + py*Y)) / sum(a2 * (X^2 + Y^2)),   u_in = r_beam / R

    with ``px, py`` the wrapping-safe per-pixel direction cosines
    (``angle(E_i * conj(E_{i-1})) / (k*d)``).  Returns ``0.0`` **exactly** for a
    real / globally-phased (flat-wavefront / collimated) field, so a collimated
    input leaves the paraxial height trace at ``h = r_beam`` -- byte-identical to
    the historical "input beam radius at every surface" for a single powered
    surface.  Handles a converging (``u_in < 0``) or diverging (``u_in > 0``)
    input so the marginal heights inside the lens are correct (G1 item 1c).

    The wrapping-safe per-pixel increments require the input carrier fringe to be
    RESOLVED by the grid (``dx < lambda*R/r`` at the beam edge); on a grid too
    coarse to sample a strongly curved carrier the increments alias and ``u_in``
    collapses toward 0 -- i.e. it falls back to the collimated (``h = r_beam``)
    trace, the CONSERVATIVE (never-under-trips) behaviour, so an under-sampled
    curved input is safe (routed to a ray member on the r_beam estimate), just not
    refined.
    """
    if not np.iscomplexobj(E):
        return 0.0
    k = 2.0 * np.pi / float(wavelength)
    px = np.zeros_like(a2)
    py = np.zeros_like(a2)
    px[:, :-1] = np.angle(E[:, 1:] * np.conj(E[:, :-1])) / (k * float(dx))
    py[:-1, :] = np.angle(E[1:, :] * np.conj(E[:-1, :])) / (k * float(dyg))
    den = float(np.sum(a2 * (X * X + Y * Y)))
    if den <= 0.0:
        return 0.0
    inv_R = float(np.sum(a2 * (px * X + py * Y))) / den
    return float(r_beam * inv_R)


def _sag_screen_aberration_rad(E_in, dx, dyg, prescription, wavelength):
    """Cheap estimate (radians of wavefront) of the analytic sag-screen model's
    leading spherical-aberration error over the whole prescription.

    Uses the audit's own error scaling -- the per-surface marginal-ray obliquity
    term ``k * |sag(r)| * theta^2`` with ``sag(r) = r^2/(2R)`` and ``theta = r/R``
    the local surface-normal tilt at the beam marginal radius ``r`` -- i.e.
    ``k * r^4 / (2 |R|^3)`` per powered *spherical* surface (the leading ``r^4``
    wavefront term the plane-projection phase screen gets wrong; H2).  Magnitudes
    are summed over surfaces (a symmetric biconvex's two surfaces contribute the
    SAME sign of physical spherical aberration; signed cancellation of the
    ``|sag|*theta^2`` obliquity magnitude would falsely exonerate it -- the H1
    sign trap).

    G1 generality extensions (docs/audit_real_lens_hammer_2026_07_19.md follow-up):

    * **Conic / aspheric surfaces (item 1a).**  The spherical ``r^4`` SAG
      coefficient ``1/(8R^3)`` in the obliquity term is generalized to the TRUE
      even-conic + aspheric coefficient ``c4 = (1+k)/(8R^3) + A4`` (A4 = the ``r^4``
      aspheric term).  ``k * r^4 / (2|R|^3) == k * r^4 * 4 * |1/(8R^3)|``, so a
      plain sphere (``k == 0``, no ``A4``) is byte-identical; a conic / asphere
      shifts the estimate to match its actual leading ``r^4`` sag.

    * **Multi-element per-surface height (item 1b).**  ``r`` is no longer the
      INPUT beam radius at every surface; a cheap paraxial marginal-ray y-nu trace
      (refraction + transfer) gives the marginal height ``h`` AT each surface, so a
      doublet / relay uses the converging beam's true per-surface radius.  For a
      single powered surface ``h == r_beam`` (unchanged).

    * **Converging / diverging input (item 1c).**  The marginal ray starts at the
      measured signed input slope ``u_in`` (:func:`_input_marginal_slope`), 0 for
      a collimated field, so the height trace is correct for a curved-wavefront
      input.

    ``r_beam`` is the input field's 1/e^2 radius (``sqrt(2 * <r^2>)``).  NOTE this
    remains a CONSERVATIVE per-surface curvature x beam-size bound on the
    sag-screen MODEL error (an obliquity upper bound), not a faithful system-SA
    estimate: it over-trips (routes to the accurate ray members, never wrong) on a
    well-corrected fast element -- see the G1 audit doc.

    Calibrated against the dual-oracle f/5 biconvex (R = +/-51.68 mm, w0 = 5 mm,
    lambda = 1.31 um): the sum is ~20 rad -- far above the ~2 rad budget, so the
    router steers it off 'phase_screen' -- while the benign small-beam regime
    (w0 = 0.5 mm) sums ~0.002 rad, well under budget.

    Returns 0.0 (never trips the gate) on an empty/zero field or a malformed
    prescription -- the same defensive fallback as :func:`_default_p_max`.
    """
    try:
        from ..glass import get_glass_index
        from ..raytrace import surfaces_from_prescription
        E = np.asarray(E_in)
        a2 = np.abs(E) ** 2
        tot = float(a2.sum())
        if not np.isfinite(tot) or tot <= 0.0:
            return 0.0
        ny, nx = E.shape
        xs = (np.arange(nx) - nx // 2) * float(dx)
        ys = (np.arange(ny) - ny // 2) * float(dyg)
        X, Y = np.meshgrid(xs, ys)
        r_rms2 = float(np.sum(a2 * (X ** 2 + Y ** 2)) / tot)
        r_beam = math.sqrt(max(2.0 * r_rms2, 0.0))      # 1/e^2 radius (Gaussian)
        k = 2.0 * np.pi / float(wavelength)
        wl = float(wavelength)
        # Paraxial marginal-ray y-nu trace: start at the measured input slope.
        u_in = _input_marginal_slope(E, a2, X, Y, dx, dyg, wl, r_beam)
        surfs = list(surfaces_from_prescription(prescription))
        h = r_beam
        u = u_in
        total = 0.0
        n_last = len(surfs) - 1
        for i, s in enumerate(surfs):
            R = float(getattr(s, 'radius', np.inf))
            kc = float(getattr(s, 'conic', 0.0) or 0.0)
            asph = getattr(s, 'aspheric_coeffs', None) or {}
            A4 = float(asph.get(4, asph.get('4', 0.0)) or 0.0) if asph else 0.0
            try:
                n1 = float(get_glass_index(getattr(s, 'glass_before', 'air'), wl))
                n2 = float(get_glass_index(getattr(s, 'glass_after', 'air'), wl))
            except (ValueError, KeyError, LookupError, TypeError):
                n1 = n2 = 1.0       # unknown glass: no paraxial bend (heights hold)
            finite = np.isfinite(R) and R != 0.0
            # Per-surface leading r^4 sag-screen error at the marginal height h.
            # Spherical (k==0, no A4) uses the byte-identical historical form;
            # a conic / asphere uses the true r^4 sag coefficient c4.
            if finite and kc == 0.0 and A4 == 0.0:
                total += k * h ** 4 / (2.0 * abs(R) ** 3)
            elif finite:
                c4 = (1.0 + kc) / (8.0 * R ** 3) + A4
                total += k * h ** 4 * 4.0 * abs(c4)
            elif A4 != 0.0:
                total += k * h ** 4 * 4.0 * abs(A4)          # flat + r^4 asphere
            # Paraxial refraction (n2 u2 = n1 u1 - h phi) + transfer to next surf.
            phi = ((n2 - n1) / R) if finite else 0.0
            if n2 != 0.0:
                u = (n1 * u - h * phi) / n2
            if i < n_last:
                h = h + u * float(getattr(s, 'thickness', 0.0) or 0.0)
        return float(total)
    except (LookupError, TypeError, ValueError, AttributeError,
            ArithmeticError):
        return 0.0


# P7 Seidel true-SA gate (docs/plan_accuracy_niches_2026_07_19.md N8).
# ``_sag_screen_aberration_rad`` above is a per-surface |c4|*h^4 MAGNITUDE bound on
# the THIN sag-screen's obliquity error -- it cannot see cancellation, so a
# well-corrected (aplanatic / balanced-asphere) element trips it even though its
# SYSTEM spherical aberration is nulled (a false positive: safe, but routes to the
# slow ray members).  ``_seidel_sa_wfe_rad`` below is the SIGNED system Seidel S1
# (spherical) sum along the SAME paraxial marginal-ray trace, so a nulled system
# reads ~0.  When it is within the near-diffraction-limited budget the analytic
# member is accurate and the router steers the plane to the fast (obliquity-correct
# ``displaced``) analytic model instead of rays.  The Seidel machinery's S4/S5
# sign convention (v5.24.4) is trustworthy AND the S1 prediction is INDEPENDENTLY
# re-checked against a Debye-oracle W(h) r^4 fit (tests/unit/test_niche_p7_*.py):
# base-sphere S1 reproduces lumenairy.seidel_coefficients to machine precision and
# the base+conic/aspheric total matches the oracle r^4 coefficient to <1% on the
# near-paraxial designs (16-20% on an f/0.85 extreme where 3rd-order theory itself
# breaks down -- that regime is high-NA and never reaches this low-NA gate).
# Conservative "comfortably diffraction-limited" W_rms budget for reclaiming a
# plane onto the analytic model: lambda/20 RMS (Strehl ~ 0.91), a bar TIGHTER than
# the Marechal lambda/14 (Strehl 0.8) diffraction limit -- so only a clearly
# well-corrected (nulled-SA) element earns the fast analytic route; a borderline
# design stays on the exact ray members (safe, only slower).
_SEIDEL_SA_MAX_RAD = 2.0 * np.pi / 20.0     # lambda/20 W_rms budget (rad)
# W_rms of a pure 3rd-order spherical W = (1/8) S1 rho^4 at the paraxial focus is
# |S1|/(12 sqrt(5)); k * that is the wavefront RMS in radians.
_SEIDEL_RMS_C = 1.0 / (12.0 * math.sqrt(5.0))


def _seidel_sa_wfe_rad(E_in, dx, dyg, prescription, wavelength):
    """Signed system Seidel S1 (third-order spherical) wavefront RMS [rad] along
    the gate's paraxial marginal-ray trace, at the field's measured input
    congruence.  Returns ``(wfe_rad, all_classified)``.

    Per surface the marginal-ray refraction invariant ``A = n1 * (c*h + u)`` and
    ``Delta(u/n)`` give the base-sphere sum ``S1_base = -A^2 * h * Delta(u/n)``
    (byte-identical to :func:`lumenairy.seidel_coefficients` for a plain sphere),
    plus the conic / aspheric r^4 departure ``S1_asph = 8*(n2-n1)*G*h^4`` with
    ``G = k_conic/(8 R^3) + A4`` (the departure of the surface figure from its
    vertex sphere; ``G == 0`` for a plain sphere).  The signed contributions are
    summed -- so a system whose surfaces' spherical aberration CANCELS (an
    aplanatic conic, a balanced doublet) reads ~0, where the |c4| magnitude bound
    of :func:`_sag_screen_aberration_rad` cannot.

    The marginal ray starts at ``h = r_beam`` (the field 1/e^2 radius) with slope
    ``u_in`` (:func:`_input_marginal_slope`, 0 for a collimated field), matching
    the sag-screen gate's trace exactly, so the conjugate rides through.

    ``all_classified`` is False (and the c4 MAGNITUDE term is added as a fallback
    for that surface) when a surface cannot be Seidel-classified -- a mirror, a
    biconic / freeform / form-error surface, or an unresolvable glass -- so the
    caller can refuse the analytic route when the walk is incomplete (the c4 bound
    is retained as the conservative fallback the plan requires).

    Returns ``(0.0, False)`` on an empty / zero field or malformed prescription
    (never trips a rescue).
    """
    try:
        from ..glass import get_glass_index
        from ..raytrace import surfaces_from_prescription
        E = np.asarray(E_in)
        a2 = np.abs(E) ** 2
        tot = float(a2.sum())
        if not np.isfinite(tot) or tot <= 0.0:
            return 0.0, False
        ny, nx = E.shape
        xs = (np.arange(nx) - nx // 2) * float(dx)
        ys = (np.arange(ny) - ny // 2) * float(dyg)
        X, Y = np.meshgrid(xs, ys)
        r_rms2 = float(np.sum(a2 * (X ** 2 + Y ** 2)) / tot)
        r_beam = math.sqrt(max(2.0 * r_rms2, 0.0))
        k = 2.0 * np.pi / float(wavelength)
        wl = float(wavelength)
        u_in = _input_marginal_slope(E, a2, X, Y, dx, dyg, wl, r_beam)
        surfs = list(surfaces_from_prescription(prescription))
        h = r_beam
        u = u_in
        s1_signed = 0.0
        fallback_rad = 0.0
        all_classified = True
        n_last = len(surfs) - 1
        for i, s in enumerate(surfs):
            R = float(getattr(s, 'radius', np.inf))
            kc = float(getattr(s, 'conic', 0.0) or 0.0)
            asph = getattr(s, 'aspheric_coeffs', None) or {}
            A4 = float(asph.get(4, asph.get('4', 0.0)) or 0.0) if asph else 0.0
            finite = np.isfinite(R) and R != 0.0
            c = (1.0 / R) if finite else 0.0
            classifiable = not bool(getattr(s, 'is_mirror', False))
            try:
                n1 = float(get_glass_index(getattr(s, 'glass_before', 'air'), wl))
                n2 = float(get_glass_index(getattr(s, 'glass_after', 'air'), wl))
            except (ValueError, KeyError, LookupError, TypeError):
                n1 = n2 = 1.0
                classifiable = False
            if classifiable and n2 != 0.0:
                i_ang = c * h + u
                A = n1 * i_ang
                u_after = (n1 * u - h * (n2 - n1) * c) / n2
                delta_un = u_after / n2 - u / n1
                s1_base = -(A ** 2) * h * delta_un
                G = (kc / (8.0 * R ** 3) + A4) if finite else A4
                s1_asph = 8.0 * (n2 - n1) * G * h ** 4
                s1_signed += s1_base + s1_asph
                u = u_after
            else:
                # Unclassifiable surface: keep the conservative c4 MAGNITUDE bound
                # (the plan's fallback) and flag the walk incomplete.
                all_classified = False
                if finite:
                    c4 = (1.0 + kc) / (8.0 * R ** 3) + A4
                    fallback_rad += k * h ** 4 * 4.0 * abs(c4)
                elif A4 != 0.0:
                    fallback_rad += k * h ** 4 * 4.0 * abs(A4)
                # no reliable paraxial bend for this surface: heights hold (u kept)
            if i < n_last:
                h = h + u * float(getattr(s, 'thickness', 0.0) or 0.0)
        wfe_rad = k * abs(s1_signed) * _SEIDEL_RMS_C + fallback_rad
        return float(wfe_rad), bool(all_classified)
    except (LookupError, TypeError, ValueError, AttributeError,
            ArithmeticError):
        return 0.0, False


def _displaced_compatible(prescription):
    """True when ``prescription`` is within ``surface_model='displaced'``'s
    rotationally-symmetric conic / aspheric envelope, so the dispatcher may route
    a Seidel-rescued plane to the obliquity-correct displaced analytic model
    instead of the ray members.  Mirrors, biconic (``radius_y``), analytic
    ``freeform_type``, ``form_error`` maps, and per-surface ``decenter`` / ``tilt``
    fall OUTSIDE the fast meridional path, so those refuse the rescue (-> rays)."""
    try:
        surfaces = prescription.get('surfaces') or []
    except AttributeError:
        return False
    if not surfaces:
        return False
    for s in surfaces:
        if not isinstance(s, dict):
            return False
        if bool(s.get('is_mirror', False)) or (
                isinstance(s.get('glass_after'), str)
                and s['glass_after'].upper() == 'MIRROR'):
            return False
        for _k in ('radius_y', 'freeform_type', 'form_error', 'sag_callable'):
            if s.get(_k) is not None:
                return False
        for _k in ('decenter', 'tilt'):
            _v = s.get(_k)
            if _v is not None and tuple(_v) != (0.0, 0.0):
                return False
    return True


def _universal_route(E_in, prescription, wavelength, dx, dyg, opd, na_threshold,
                     caustic_pad_dof, multivalued, multivalued_threshold,
                     aberration_threshold,
                     seidel_sa_threshold=_SEIDEL_SA_MAX_RAD):
    """Pure routing decision for :func:`apply_real_lens_universal` (no
    propagation, so the choice is cheaply unit-testable).  Returns the member name
    ('phase_screen' / 'displaced' / 'fga' / 'traced').  See the dispatcher docstring
    for the full regime map; the H2 aberration gate
    (:func:`_sag_screen_aberration_rad`) keeps a heavily-aberrated prescription off
    'phase_screen' even at low NA, and the P7 Seidel gate
    (:func:`_seidel_sa_wfe_rad`) reclaims a well-corrected (nulled-SA) element from
    the ray members onto the fast obliquity-correct 'displaced' analytic model."""
    na = _system_na(prescription, wavelength)
    aberrated = (_sag_screen_aberration_rad(E_in, dx, dyg, prescription,
                                            wavelength)
                 > float(aberration_threshold))
    if na < float(na_threshold) and not aberrated:
        # low NA + benign: the thin-element phase screen is angle-independent (a
        # linear operator on the field) + exact ASM, so it is wave-exact for ANY
        # (single- or multi-valued) field, and the fastest option.  H2 gate: a
        # low-NA-but-heavily-aberrated prescription (below na_threshold yet with a
        # large per-surface r^4 term) falls through to the ray-based members
        # instead of this sag-screen.
        return "phase_screen"
    if na < float(na_threshold) and aberrated:
        # P7 Seidel refinement.  The c4 obliquity bound above is a per-surface
        # MAGNITUDE sum, so it FALSE-POSITIVES on an element whose SYSTEM spherical
        # aberration is nulled across surfaces (an aplanatic conic / balanced
        # asphere) even though each |c4| is large.  The signed Seidel S1 sum sees
        # that cancellation; when the implied W_rms is within the near-diffraction-
        # limited budget AND every surface was Seidel-classifiable, the analytic
        # model IS accurate -- so route to the obliquity-correct 'displaced' model
        # (the thin sag-screen mis-renders the strong sag departure, but displaced
        # carries the true per-surface incidence cosines: measured ~2% vs the
        # diffraction oracle where thin reads ~40%).  Restricted to LOW NA, where
        # the 3rd-order term dominates and 5th-order (~NA^6) is negligible, and to
        # the displaced envelope (rotationally-symmetric conic / aspheric).  A
        # false negative here only costs speed (falls through to the exact ray
        # members); a false positive is prevented by requiring BOTH the classified
        # walk and the tight W_rms budget.
        seidel_rad, classified = _seidel_sa_wfe_rad(E_in, dx, dyg, prescription,
                                                    wavelength)
        if (classified and seidel_rad < float(seidel_sa_threshold)
                and _displaced_compatible(prescription)):
            return "displaced"
    # high NA, OR low-NA-but-aberrated: use a ray/beamlet member (no thin-screen
    # obliquity ceiling).  Multi-valued fields never route to traced.
    if multivalued is None:
        mv = (_tilt_dispersion(E_in, dx, dyg, wavelength, na)
              > float(multivalued_threshold))
    else:
        mv = bool(multivalued)
    if mv:
        return "fga"
    zone = _caustic_zone(E_in, dx, prescription, wavelength)
    if zone is not None:
        pad = caustic_pad_dof * float(wavelength) / (na * na)
        if (zone[0] - pad) <= opd <= (zone[1] + pad):
            return "fga"
    # smooth plane: the sub-nm traced OPL, but traced launches rays along the local
    # phase gradient and is valid only for a ~collimated beam.  A single-valued but
    # DIVERGING beam would be blurred -> route a BENIGN diverging beam to the
    # wave-exact phase_screen (bounded thin-screen OPD, never a blur); an ABERRATED
    # beam must avoid the sag-screen -> traced, the H6-fixed reference for the
    # diverging real-surface class.  Uses traced's own collimation discriminator.
    from ..elements._lens_traced import (
        _NONCOLLIMATED_RESID_THRESH,
        _carrier_residual_rms,
    )
    spread = _carrier_residual_rms(E_in, None, wavelength, dx)
    if spread > _NONCOLLIMATED_RESID_THRESH and not aberrated:
        return "phase_screen"
    return "traced"


def apply_real_lens_universal(
    E_in: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    output_plane_distance: float = 0.0,
    method: str = "auto",
    na_threshold: float = 0.12,
    caustic_pad_dof: float = 3.0,
    multivalued: Optional[bool] = None,
    multivalued_threshold: float = 0.06,
    aberration_threshold: float = _ABERRATION_MAX_RAD,
    seidel_sa_threshold: float = _SEIDEL_SA_MAX_RAD,
    return_method: bool = False,
    method_kwargs: Optional[Dict[str, Any]] = None,
):
    """Universal (5-way) auto-dispatching lens propagator -- routes each output
    plane to the MOST ACCURATE propagator for its regime.

    ``method='auto'`` (default) picks:

    * ``'phase_screen'`` (:func:`lumenairy.elements.apply_real_lens`) -- LOW NA
      (``< na_threshold``) **and within the sag-screen aberration envelope**: the
      thin-element phase-screen model is accurate there and the exact
      angular-spectrum propagation handles focus/caustics, so it is wave-exact and
      fast, with no beamlet-discretization or ray-model cost;
    * ``'fga'`` (:func:`apply_real_lens_fga`) -- HIGH NA **and** near a caustic:
      the only caustic-accurate *and* ray-based (no thin-screen obliquity) option;
    * ``'traced'`` (:func:`lumenairy.elements.apply_real_lens_traced`) -- HIGH NA,
      smooth, single-valued AND **~collimated**: per-pixel ray-traced OPL, sub-nm,
      no thin-screen ceiling.  A single-valued but **diverging** beam (large
      residual angular spread -- not multi-valued, but not collimated either) is
      routed to ``'phase_screen'`` instead: traced launches rays along the local
      phase gradient and would silently blur it, whereas the phase-screen +
      exact-ASM path is wave-exact in propagation (bounded thin-screen OPD error,
      never a blur).  The split uses traced's own collimation threshold.

    Multi-valued fields never route to ``traced``
    ---------------------------------------------
    ``traced`` launches one ray per pixel along the LOCAL phase gradient, so it is
    only valid where the field has a single well-defined direction at each pixel.
    A **multi-emitter / post-DOE / speckle** field is *multi-valued* -- several
    wave components cross the same region, so there is no single local direction,
    and traced silently collapses them to their amplitude-weighted MEAN direction
    (applying the wrong angle-dependent OPD to every component).  ``'auto'``
    therefore measures the field's multi-valuedness (:func:`_tilt_dispersion`, the
    NA-normalized spread of the local wavevector about its per-region mean) and
    routes multi-valued high-NA fields to ``'fga'``, whose phase-space swarm
    transports every direction independently.  ``multivalued`` overrides the
    detector: ``True`` forces the multi-valued path (never ``traced``), ``False``
    trusts the field as single-valued (allows ``traced``), ``None`` (default)
    auto-detects with cutoff ``multivalued_threshold`` (score above it => FGA).
    A false positive only costs speed (FGA is never *wrong*, just slower than
    traced on a truly single-valued field), so the detector is biased to prefer
    FGA when uncertain.

    Aberration gate: never route a heavily-aberrated prescription to the sag-screen
    -------------------------------------------------------------------------------
    The analytic ``'phase_screen'`` model is a per-surface plane-projection phase
    mask, which structurally cannot represent the transverse ray-displacement of
    high-order / orientation-dependent aberration (hammer audit H2, now
    dual-oracle-quantified: the f/5 biconvex converged to 40.5 um vs the 65 um
    truth; a plano-convex errs in BOTH orientations, 60.4/60.9 um where the truth is
    43/128 um).  ``'auto'`` therefore estimates that error cheaply at routing time
    (:func:`_sag_screen_aberration_rad`: the per-surface marginal-ray obliquity sum
    ``k * r^4 / (2|R|^3)``, the audit's own error scaling) and, when it exceeds
    ``aberration_threshold`` radians, routes AWAY from ``'phase_screen'`` even at
    low NA -- into the ray-based members via the same regime logic (near a caustic
    -> ``'fga'``; a smooth plane -> ``'traced'``, the H6-fixed reference for the
    diverging real-surface class).  The benign small-beam / low-NA well-corrected
    regime (estimate << the budget) keeps the fast wave-exact phase-screen dispatch
    unchanged.  When ``method='phase_screen'`` is explicitly FORCED outside the
    envelope, the call still runs but emits a ``RuntimeWarning`` citing the budget
    (the absolute spot size / EE are unreliable there).

    Seidel true-SA gate: reclaim a nulled-SA element from the ray members (P7)
    -------------------------------------------------------------------------
    The ``_sag_screen_aberration_rad`` bound above sums per-surface |c4| MAGNITUDES,
    so it cannot see cancellation: an aplanatic conic / balanced asphere whose
    SYSTEM spherical aberration is nulled still trips it (a false positive -- safe,
    but it pays for the slow ray members).  ``'auto'`` therefore also computes the
    SIGNED system Seidel S1 (:func:`_seidel_sa_wfe_rad`) along the same paraxial
    marginal-ray trace; a nulled system reads ~0.  When -- at LOW NA, where the
    3rd-order term dominates -- that trip is paired with a Seidel W_rms inside the
    near-diffraction-limited budget ``seidel_sa_threshold`` (lambda/20, Strehl ~0.91)
    AND every surface was classifiable AND the prescription is within the displaced
    envelope, the plane routes to ``'displaced'`` (:func:`lumenairy.apply_real_lens`
    ``surface_model='displaced'``, ``conjugate='auto'``) -- the obliquity-correct
    analytic model, which renders the strong corrected sag departure the thin screen
    cannot (measured ~2% vs the diffraction oracle where the thin screen reads
    ~40%).  A ``'displaced'`` plane, like ``'phase_screen'`` / ``'traced'``, returns
    at the exit vertex and finishes the leg with an exact angular-spectrum step.

    Exit-NA Nyquist guard (H3): orthogonal to this router.  When ``'auto'`` selects
    ``'traced'``, that member carries its OWN amplitude-aware exit-NA Nyquist guard
    (warns when the grid ``dx`` cannot resolve the converging exit wavefront; H3),
    which fires unchanged however traced is invoked -- the dispatcher neither
    suppresses nor duplicates it.  Pass ``method_kwargs={'traced':
    {'on_undersample': 'silent'}}`` to quiet it for a deliberately coarse grid.

    Aperture:beam cliff guard on the ``'traced'`` route (v5.31, audit W9-13)
    ----------------------------------------------------------------------
    This router now runs ``'traced'`` with ``fit_radius_beam_factor=2.0`` -- the
    P2 guard that is :func:`~lumenairy.propagate_traced_carrier_chain`'s default --
    rather than inheriting the element's historical ``None``.  Without it a
    physical aperture much larger than the beam lets wildly-aberrated marginal
    rays into the traced OPL fit and the focus collapses; MEASURED on the E4
    corrected relay at the element's own defaults, exit-wavefront Strehl
    0.9701 / 0.1085 / 0.0384 at 1.50x / 1.75x / 2.50x the beam diameter without
    the guard versus 0.9874 / 0.9820 / 0.9816 with it.  Only the ray-FIT domain
    is restricted -- no field energy is vignetted -- and the pre-cliff regime is
    unharmed.  Pass ``method_kwargs={'traced': {'fit_radius_beam_factor': None}}``
    for the pre-v5.31 aperture-only fit domain.  The chain's other three
    validated options are deliberately NOT adopted here: they are carrier-regime
    options and this router supplies no carrier (measured equal-or-worse without
    one).

    The two ray/wave-exact-surface members that are NOT caustic-native
    (``phase_screen``, ``traced``) return the field at the exit vertex, so the
    output-plane leg is finished with an exact angular-spectrum propagation.
    Force any member with ``method='phase_screen'|'gbd'|'traced'|'fga'``
    (``'gbd'`` -- the fast, differentiable, polarization-capable thawed beamlet --
    is not auto-selected because ``traced``/``fga`` dominate it on accuracy, but
    is available for those other strengths).  ``return_method=True`` also returns
    the chosen name; ``method_kwargs={'traced': {...}, 'fga': {...}, ...}``
    forwards per-method extra arguments.  ``dy`` (anamorphic y pitch) is
    forwarded to whichever member runs, including the exact angular-spectrum
    output leg.

    This is the **canonical** auto-dispatcher (:func:`apply_real_lens_auto` is the
    older GBD/FGA-only 2-way subset).

    .. note::
       **Split-step callers.**  The near-caustic -> ``'fga'`` decision keys on
       ``output_plane_distance`` (the geometric caustic lies DOWNSTREAM of the
       exit vertex).  A caller that applies the lens at ``output_plane_distance=0``
       and does its OWN downstream free-space propagation (a split-step BPM /
       manual ASM) therefore never triggers the ``'fga'`` branch -- at the vertex a
       single-valued field routes only to ``'phase_screen'`` / ``'traced'``
       (or ``'fga'`` if it is itself multi-valued).  If you split the lens and the
       propagation and want caustic-accurate rendering, pass the full
       ``output_plane_distance`` here (let the dispatcher finish the leg) or force
       ``method='fga'``.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_universal',
                           input_kind='field')
    valid = ("auto", "phase_screen", "displaced", "gbd", "traced", "fga")
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got {method!r}.")
    mkw = dict(method_kwargs or {})
    E_in = np.asarray(E_in)
    opd = float(output_plane_distance)
    dyg = float(dx) if dy is None else float(dy)

    chosen = method
    if method == "auto":
        chosen = _universal_route(
            E_in, prescription, float(wavelength), float(dx), dyg, opd,
            na_threshold, caustic_pad_dof, multivalued, multivalued_threshold,
            aberration_threshold, seidel_sa_threshold)
    elif method == "phase_screen":
        # H2 gate: 'phase_screen' (analytic sag-screen) FORCED.  'auto' would have
        # re-routed a heavily-aberrated prescription away; when the user forces it,
        # honour the choice but warn if the estimated per-surface spherical-
        # aberration wavefront error is outside the sag-screen validity envelope,
        # so an unreliable absolute spot / EE is never returned silently.
        aberr = _sag_screen_aberration_rad(E_in, float(dx), dyg, prescription,
                                           float(wavelength))
        if aberr > float(aberration_threshold):
            warnings.warn(
                "apply_real_lens_universal: method='phase_screen' (analytic "
                "sag-screen) forced, but the estimated per-surface spherical-"
                f"aberration wavefront error is {aberr:.1f} rad, above the "
                f"{float(aberration_threshold):.1f} rad validity budget.  The "
                "plane-projection phase screen cannot represent this high-order / "
                "orientation-dependent aberration (audit H2: the f/5 singlet "
                "converged to 40.5 um vs the 65 um dual-oracle truth), so the "
                "absolute spot size / encircled energy will be unreliable.  Use "
                "method='traced' or 'fga' for absolute fidelity, or method='auto' "
                "to route automatically.",
                RuntimeWarning, stacklevel=2)

    if chosen == "fga":
        out = apply_real_lens_fga(
            E_in, prescription=prescription, wavelength=wavelength, dx=dx, dy=dy,
            output_plane_distance=opd, **mkw.get("fga", {}))
    elif chosen == "gbd":
        from ..elements import apply_real_lens_gbd
        out = apply_real_lens_gbd(
            E_in, prescription=prescription, wavelength=wavelength, dx=dx, dy=dy,
            output_plane_distance=opd, **mkw.get("gbd", {}))
    else:
        # phase_screen / displaced / traced all return at the exit vertex -> finish
        # the output leg with an exact angular-spectrum propagation (the field is
        # smooth here, so ASM is wave-exact even through an intervening focus).
        from ..elements import apply_real_lens, apply_real_lens_traced
        extra = dict(mkw.get(chosen, {}))
        if chosen == "traced":
            # v5.31 (audit W9-13): adopt the P2 aperture:beam cliff GUARD as
            # this router's traced default.  ``apply_real_lens_traced``'s own
            # signature default is ``fit_radius_beam_factor=None`` (the
            # historical aperture-only ray-fit domain, kept for byte-identity
            # on the direct-call path), and this branch inherited it -- so
            # 'auto' routed high-NA collimated beams straight into the regime
            # the v5.29 P2 work exists to survive, with only the element's
            # warn-only ``on_aperture_beam`` RuntimeWarning for company.
            #
            # MEASURED (E4 corrected relay, N=1536, dx=7 um, beam w=2 mm, every
            # other option at the ELEMENT defaults -- i.e. exactly this branch's
            # configuration -- exit-wavefront Strehl in the carrier-referenced
            # envelope):
            #
            #   aperture   1.50x beam   1.75x beam   2.50x beam
            #   frbf=None      0.9701       0.1085       0.0384
            #   frbf=2.0       0.9874       0.9820       0.9816
            #
            # i.e. +0.87 and +0.94 Strehl past the cliff, and +0.017 (no harm)
            # before it.  The guard restricts only the ray-FIT domain -- no
            # field energy is vignetted -- so this is not an accuracy/energy
            # trade.  ``setdefault``: ``method_kwargs={'traced':
            # {'fit_radius_beam_factor': None}}`` still opts back out.
            #
            # NOT adopted: the chain's other three validated options
            # (``amplitude_model='ray_density'``, ``preserve_input_phase=
            # 'remap'``, ``remap_sampling='full'``).  Those are CARRIER-regime
            # options -- ``propagate_traced_carrier_chain`` applies them
            # together with ``carrier_reference='sphere'`` and an
            # element-supplied ``carrier=R_in``, which this router cannot
            # provide (it has no q-trace).  MEASURED on a single element with no
            # carrier they do not pay: exit Strehl deltas of -0.0025 / -0.0249 /
            # -0.1912 at 1.0x / 1.5x / 1.75x beam, i.e. equal or worse.
            from ..elements._lens_traced import _FIT_RADIUS_BEAM_FACTOR_DEFAULT
            extra.setdefault('fit_radius_beam_factor',
                             _FIT_RADIUS_BEAM_FACTOR_DEFAULT)
            exitf = apply_real_lens_traced(
                E_in, prescription=prescription, wavelength=wavelength, dx=dx,
                dy=dy, **extra)
        elif chosen == "displaced":
            # P7 Seidel-rescued analytic member: the obliquity-correct sag screen.
            # 'auto' fits the input congruence from E_in (collimated -> flat), so a
            # curved-wavefront input gets the correct per-surface incidence fan.
            extra.setdefault("surface_model", "displaced")
            extra.setdefault("conjugate", "auto")
            exitf = apply_real_lens(
                E_in, prescription=prescription, wavelength=wavelength, dx=dx,
                dy=dy, **extra)
        else:   # phase_screen (thin sag-screen)
            exitf = apply_real_lens(
                E_in, prescription=prescription, wavelength=wavelength, dx=dx,
                dy=dy, **extra)
        if opd != 0.0:
            from .asm import angular_spectrum_propagate
            exitf = angular_spectrum_propagate(exitf, opd, wavelength, dx, dy)
        out = exitf
    return (out, chosen) if return_method else out

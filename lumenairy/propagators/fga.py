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
identity is exact (energy ratio 1.0; the leading FGA is exact for the paraxial
quadratic Hamiltonian, so free-space propagation conserves energy to ~1.0).
The frozen width ``w0`` is the FGA convergence parameter: a wider beamlet is more
paraxial and gives a cleaner frame, while caustic *resolution* wants a smaller
``w0`` -- a standard FGA tradeoff.  The momentum half-range ``p_max`` must cover
the field's angular content (auto-set from the prescription NA); ``n_p`` sets the
momentum samples per axis and ``dq_step`` the position-lattice stride.

Energy caveat (near-axial inputs through strong focusing).  When the input is
near-collimated (its FBI/local spectrum concentrates near ``p=0``) AND the system
focuses it strongly, the raw output amplitude can be off by a large scale factor
(the FGA high-frequency assumption is strained near ``p=0``, and OVER-sampling
high-``p`` beamlets injects spurious energy).  The field SHAPE stays correct
(fidelity ~0.9998 vs the exact field), so pass ``normalize_output='power'`` to
recover the absolute scale, and do not set ``p_max`` wider than the actual
angular content.  This is a representation-regime effect, not the O(eps) FGA
transport error (which is negligible for lens-like Hamiltonians).

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

Requires the optional ``numba`` accelerator (the pure-NumPy swarm sum is
impractically slow); install with ``pip install lumenairy[numba]``.
"""
from __future__ import annotations

import copy as _copy
import math
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
            j0 = int((cxq - R - x0) / dx)
            j1 = int((cxq + R - x0) / dx) + 1
            i0 = int((cyq - R - y0) / dyg)
            i1 = int((cyq + R - y0) / dyg) + 1
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
                    if dxr * dxr + dy2 <= R2:
                        mag = agy * gxj
                        outr[i, j] += mag * (wr * cph - wi * sph)
                        outi[i, j] += mag * (wr * sph + wi * cph)
                    # advance recurrences (every step, to stay phase-locked)
                    dxr += dx
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


def _swarm_lattice(Ny, Nx, dx, dyg, x0, y0, dq_step, p_max, n_p):
    qi = np.arange(0, Nx, dq_step)
    qj = np.arange(0, Ny, dq_step)
    qxg, qyg = np.meshgrid(x0 + qi * dx, y0 + qj * dyg)
    qx = qxg.ravel().astype(np.float64)
    qy = qyg.ravel().astype(np.float64)
    # momentum (direction-cosine) grid is ISOTROPIC: p is the physical transverse
    # direction cosine, bounded by the system NA, not by the (possibly
    # anamorphic) pixel pitch.
    pv = np.linspace(-p_max, p_max, n_p)
    pxg, pyg = np.meshgrid(pv, pv)
    px = pxg.ravel().astype(np.float64)
    py = pyg.ravel().astype(np.float64)
    dp = (pv[1] - pv[0]) if n_p > 1 else 2.0 * p_max
    return qx, qy, px, py, dp


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


def _reconstruct(Qx, Qy, Px, Py, W, x0, y0, dx, dyg, Ny, Nx, w0, k, Ag, nsig):
    outr = np.zeros((Ny, Nx))
    outi = np.zeros((Ny, Nx))
    _reconstruct_into(outr, outi, Qx, Qy, Px, Py, W, x0, y0, dx, dyg, Ny, Nx,
                      w0, k, Ag, nsig)
    return outr + 1j * outi


def _default_p_max(prescription, wavelength):
    """Momentum half-range from the system NA (falls back to a moderate cone)."""
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
        na = (r / efl) if (r > 0 and efl > 0) else 0.1
        return float(min(0.6, max(0.05, 1.6 * na)))
    except (LookupError, TypeError, ValueError, AttributeError,
            ArithmeticError):
        return 0.15


_DP_TARGET = 0.008          # auto momentum spacing (direction cosine); see below


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


def _resolve_sampling(E_in, dx, dyg, wavelength, prescription, p_max, n_p):
    """Resolve ``(p_max, n_p)``, auto-sizing either when ``None`` so the
    phase-space swarm is MATCHED to the field: ``p_max`` covers the field's
    angular content (capped by the system NA), and ``n_p`` makes the momentum
    spacing ``dp = 2*p_max/(n_p-1)`` fine enough (``~<= _DP_TARGET``) to resolve
    the phase-space quadrature -- which is what makes FGA accurate on diverging
    beams (whose broad phase-space footprint the fixed old default under-sampled;
    leading FGA is EXACT for free space, so with dp matched a diverging beam
    reconstructs to ~round-off).  An explicit ``p_max`` / ``n_p`` is honoured."""
    if p_max is None:
        na_cap = _default_p_max(prescription, wavelength)
        content = _field_angular_content(E_in, dx, dyg, wavelength)
        # cover the field's content generously (x1.5 for the spectral tail --
        # over-wide p_max is harmless at fine dp, only truncation hurts), but
        # never wider than the system NA cone (x1.5): content beyond the NA is
        # clipped by the optic anyway, and a wider p_max just inflates n_p.
        p_max = float(min(1.5 * na_cap, max(0.03, 1.5 * content)))
    if n_p is None:
        # momentum SPACING (direction cosine) fine enough to converge the
        # phase-space quadrature.  Empirically ~0.008 works across beam sizes
        # (the historical diverging-beam cap was dp ~0.02-0.03, too coarse); it
        # is roughly field-independent for smooth beams -- a wide collimated beam
        # has a narrow p_max (few samples) and a small diverging beam a wider
        # p_max (more), both converging at this spacing -- so n_p scales with
        # p_max, not with the beam size.  The frame stays hugely over-complete
        # (Nyquist allows dp up to ~lambda/dq >> this).
        n_p = int(np.ceil(2.0 * p_max / _DP_TARGET)) | 1
        n_p = int(min(61, max(7, n_p)))       # clamp: >=7 samples, <=61 (cost)
    return float(p_max), int(n_p)


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


def _resolve_nq_chunk(Nq, Np, use_sep, cw, mem_budget_mb, fn, cfull_mult=1.0):
    """POSITION-lattice (``Nq``) chunk size from ``mem_budget_mb``, or ``None`` if
    the whole lattice fits / no budget.  ``mem_budget_mb`` now bounds BOTH
    dimensions: momenta via ``cw`` and the position lattice via this chunk, so a
    large aperture runs within the budget (additive over lattice chunks) instead
    of OOMing.  Peak per lattice-chunk of size ``nqc``:

    * separable coeff array ``c_full`` -- ``cfull_mult*nqc*Np*16`` B (whole
      momentum grid; ``cfull_mult`` = 1 scalar, 2 for the vector ``Ex``/``Ey``);
    * beamlet transport arrays ``QX/QY/PX/PY + AW + coeff (+ Jones)`` --
      ``nqc*cw*~100`` B;
    * the per-momentum ``ray_transfer_jacobian`` temporaries -- ``nqc*~500`` B.

    Only raises if even a SINGLE lattice point exceeds the budget (absurdly tiny
    budget / huge ``n_p``) -- otherwise it always finds a workable chunk."""
    if mem_budget_mb is None:
        return None
    budget = float(mem_budget_mb) * 1e6
    per = (cfull_mult * Np * 16.0 if use_sep else 0.0) + cw * 100.0 + 500.0
    nqc = int(budget / per)
    if nqc < 1:
        raise MemoryError(
            f"{fn}: a single position-lattice point needs ~{per / 1e6:.2f} MB "
            f"(Np={Np}, chunk={cw}), exceeding mem_budget_mb={mem_budget_mb} MB.  "
            f"Raise the budget, lower n_p, or use a smaller momentum chunk.")
    return None if nqc >= Nq else nqc


def _fga_through_lens(u0, dx, dyg, prescription, wavelength, w0, z_image,
                      dq_step, p_max, n_p, nsig, chunk=None, prune_frac=0.0,
                      coeff_frac=0.0, separable="auto", mem_budget_mb=None):
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
    from ..raytrace.differential import ray_transfer_jacobian

    k = 2.0 * np.pi / wavelength
    Ny, Nx = u0.shape
    x0 = -(Nx / 2) * dx
    y0 = -(Ny / 2) * dyg
    Ag = (1.0 / (np.pi * w0 ** 2)) ** 0.5

    qx, qy, px, py, dp = _swarm_lattice(Ny, Nx, dx, dyg, x0, y0, dq_step,
                                        p_max, n_p)
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
    # in the well-sampled limit, -> 1.0 with this factor).
    C = ((k / (2.0 * np.pi)) ** 2 * (dq_step ** 2 * dx * dyg) * (dp ** 2)) / 2.0

    # trace to the LAST SURFACE VERTEX; the image-side leg is added manually.
    surfs = [_copy.copy(s) for s in surfaces_from_prescription(prescription)]
    surfs[-1].thickness = 0.0
    kw2 = k * w0 * w0

    cw = Np if (chunk is None or int(chunk) <= 0) else min(int(chunk), Np)
    use_sep = bool(separable) if separable != "auto" else (n_p >= 5)
    # mem_budget_mb now bounds BOTH dimensions: momenta via cw AND the POSITION
    # lattice via nq_chunk, so a large aperture runs within the budget as an
    # additive sum over lattice chunks instead of OOMing (audit F3).  The
    # separable c_full is then per-lattice-chunk (nqc*Np), not the whole Nq*Np.
    nq_chunk = _resolve_nq_chunk(Nq, Np, use_sep, cw, mem_budget_mb,
                                 "apply_real_lens_fga")
    qbounds = ([(0, Nq)] if nq_chunk is None
               else [(s, min(s + nq_chunk, Nq)) for s in range(0, Nq, nq_chunk)])

    # Coefficient pruning under Nq-chunking needs the GLOBAL per-momentum peak
    # (over ALL q): a per-lattice-chunk peak under-counts and would prune real
    # momenta.  With >1 lattice chunk, a light analysis-only PRE-PASS accumulates
    # that global peak so the prune decision (and the field) matches the
    # un-chunked result to round-off.  Single chunk keeps the running-max path.
    keep_mom = None
    if coeff_frac > 0.0 and len(qbounds) > 1:
        gpeak = np.zeros(Np)
        for qs, qe in qbounds:
            cc = (_gabor_coeff_sep(u0, qx[qs:qe], qy[qs:qe], px[:n_p], x0, y0,
                                   dx, dyg, w0, k, Ag, nsig) if use_sep else
                  _gabor_coeff(u0, qx[qs:qe], qy[qs:qe], px, py, x0, y0, dx, dyg,
                               w0, k, Ag, nsig))
            gpeak = np.maximum(gpeak, np.max(np.abs(cc), axis=0))
        keep_mom = gpeak >= coeff_frac * float(gpeak.max() + 1e-300)

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
                # slope -> direction-cosine conjugation (canonical monodromy)
                go = 1.0 / (1.0 + uxo ** 2 + uyo ** 2) ** 1.5     # dp/du at output
                gi = (1.0 + (pxi / pz_in) ** 2 + (pyi / pz_in) ** 2) ** 1.5  # du/dp
                A = M[:, 0:2, 0:2]
                B = M[:, 0:2, 2:4] * gi
                Cc = M[:, 2:4, 0:2] * go[:, None, None]
                D = M[:, 2:4, 2:4] * (go[:, None, None] * gi)
                Z = (A + D) + 1j * (kw2 * Cc - B / kw2)
                a = np.sqrt(_det2(Z))
                a = np.where(a.real < 0, -a, a)               # continuous branch
                invo = 1.0 / np.sqrt(1.0 + uxo ** 2 + uyo ** 2)
                QX[:, j] = xv
                QY[:, j] = yv
                PX[:, j] = uxo * invo
                PY[:, j] = uyo * invo
                AW[:, j] = C * a * np.exp(1j * k * opd_tot)
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
    normalize_output: str = "none",
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
        ``(nq_chunk*Np)`` coefficient array, the per-momentum ray trace, and the
        scatter are all bounded by ``nq_chunk``.  The chunked result is identical
        to the full swarm to float round-off (it only reorders an additive sum),
        so this is a pure memory lever.  ``None`` processes the whole swarm at
        once.  Only an absurd budget (a single lattice point can't fit) raises.
        (A 24 mm-aperture FGA is still impractically slow -- the ray trace scales
        with ``Nq`` -- so control ``Nq`` with ``dq_step`` / ``prune_frac`` / ``n_p``
        and prefer a lighter propagator there, but it no longer OOMs.)
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
    normalize_output : {'none', 'power'}
        ``'none'`` returns the raw FGA field
        ``'power'`` rescales it so the
        output power equals the input power (energy conservation improves with
        ``w0_factor`` -- see the module docstring).

    Returns
    -------
    (Ny, Nx) complex ndarray
        The field at the output plane.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_fga')
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
    dyg = float(dx) if dy is None else float(dy)
    p_max, n_p = _resolve_sampling(E_in, float(dx), dyg, float(wavelength),
                                   prescription, p_max, n_p)
    w0 = float(w0_factor) * math.sqrt(float(dx) * dyg)
    chunk_eff = _chunk_from_budget(E_in.shape, int(dq_step), chunk, mem_budget_mb)
    out = _fga_through_lens(
        E_in, float(dx), dyg, prescription, float(wavelength), w0,
        float(output_plane_distance), int(dq_step), float(p_max), int(n_p),
        float(nsig), chunk=chunk_eff, prune_frac=float(prune_frac),
        coeff_frac=float(coeff_frac), separable=separable,
        mem_budget_mb=mem_budget_mb)
    if normalize_output == "power":
        pin = float(np.sum(np.abs(E_in) ** 2))
        pout = float(np.sum(np.abs(out) ** 2))
        if pout > 0.0:
            out = out * math.sqrt(pin / pout)
    return out


def _fga_vector_through_lens(Ex, Ey, dx, dyg, prescription, wavelength, w0,
                             z_image, dq_step, p_max, n_p, nsig, chunk=None,
                             prune_frac=0.0, coeff_frac=0.0, separable="auto",
                             mem_budget_mb=None):
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
    qx, qy, px, py, dp = _swarm_lattice(Ny, Nx, dx, dyg, x0, y0, dq_step,
                                        p_max, n_p)
    if prune_frac > 0.0:
        supp = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2)
        keep = _lattice_support_mask(supp, dx, dyg, dq_step, w0, nsig,
                                     prune_frac)
        qx = qx[keep]
        qy = qy[keep]
    Nq = qx.shape[0]
    Np = px.shape[0]
    C = ((k / (2.0 * np.pi)) ** 2 * (dq_step ** 2 * dx * dyg) * (dp ** 2)) / 2.0

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
    keep_mom = None
    if coeff_frac > 0.0 and len(qbounds) > 1:
        gpeak = np.zeros(Np)
        for qs, qe in qbounds:
            cxc, cyc = _cxy(qx[qs:qe], qy[qs:qe], px, py, True)
            gpeak = np.maximum(gpeak, np.max(
                np.sqrt(np.abs(cxc) ** 2 + np.abs(cyc) ** 2), axis=0))
        keep_mom = gpeak >= coeff_frac * float(gpeak.max() + 1e-300)

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
                go = 1.0 / (1.0 + uxo ** 2 + uyo ** 2) ** 1.5
                gi = (1.0 + (pxi / pz_in) ** 2 + (pyi / pz_in) ** 2) ** 1.5
                A = M[:, 0:2, 0:2]
                B = M[:, 0:2, 2:4] * gi
                Cc = M[:, 2:4, 0:2] * go[:, None, None]
                D = M[:, 2:4, 2:4] * (go[:, None, None] * gi)
                Z = (A + D) + 1j * (kw2 * Cc - B / kw2)
                a = np.sqrt(_det2(Z))
                a = np.where(a.real < 0, -a, a)
                base = C * a * np.exp(1j * k * opd_tot)  # scalar beamlet weight
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
    dyg = float(dx) if dy is None else float(dy)
    # auto-size the swarm from the higher-energy Jones component (both share the
    # beam geometry, so its extent/angular-content set the sampling for both)
    _rep = E_vec[0] if (np.sum(np.abs(E_vec[0]) ** 2)
                        >= np.sum(np.abs(E_vec[1]) ** 2)) else E_vec[1]
    p_max, n_p = _resolve_sampling(_rep, float(dx), dyg, float(wavelength),
                                   prescription, p_max, n_p)
    w0 = float(w0_factor) * math.sqrt(float(dx) * dyg)
    chunk_eff = _chunk_from_budget(E_vec[0].shape, int(dq_step), chunk,
                                   mem_budget_mb)
    ex, ey, ez = _fga_vector_through_lens(
        E_vec[0], E_vec[1], float(dx), dyg, prescription, float(wavelength), w0,
        float(output_plane_distance), int(dq_step), float(p_max), int(n_p),
        float(nsig), chunk=chunk_eff, prune_frac=float(prune_frac),
        coeff_frac=float(coeff_frac), separable=separable,
        mem_budget_mb=mem_budget_mb)
    if normalize_output == "power":
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
    cx = N // 2
    xgrid = (np.arange(N) - cx) * dx
    row = E_in[cx, :]
    amp = np.abs(row)
    if amp.max() <= 0.0:
        return None
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
    _check_2d_scalar_field(E_in, 'apply_real_lens_auto')
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
    return_method: bool = False,
    method_kwargs: Optional[Dict[str, Any]] = None,
):
    """Universal (4-way) auto-dispatching lens propagator -- routes each output
    plane to the MOST ACCURATE propagator for its regime.

    ``method='auto'`` (default) picks:

    * ``'phase_screen'`` (:func:`lumenairy.elements.apply_real_lens`) -- LOW NA
      (``< na_threshold``): the thin-element phase-screen model is accurate there
      and the exact angular-spectrum propagation handles focus/caustics, so it is
      wave-exact and fast, with no beamlet-discretization or ray-model cost;
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
    _check_2d_scalar_field(E_in, 'apply_real_lens_universal')
    valid = ("auto", "phase_screen", "gbd", "traced", "fga")
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got {method!r}.")
    mkw = dict(method_kwargs or {})
    E_in = np.asarray(E_in)
    opd = float(output_plane_distance)

    chosen = method
    if method == "auto":
        na = _system_na(prescription, wavelength)
        dyg = float(dx) if dy is None else float(dy)
        if na < float(na_threshold):
            # low NA: the thin-element phase screen is angle-independent (a linear
            # operator on the field) + exact ASM, so it is wave-exact for ANY
            # field, single- or multi-valued.
            chosen = "phase_screen"
        else:
            # high NA: traced is only valid for a SINGLE-VALUED wavefront.  Decide
            # multi-valuedness first (explicit override, else auto-detect); a
            # multi-valued field can never use traced -> FGA transports every
            # crossing direction independently.
            if multivalued is None:
                mv = _tilt_dispersion(E_in, float(dx), dyg, float(wavelength),
                                      na) > float(multivalued_threshold)
            else:
                mv = bool(multivalued)
            if mv:
                chosen = "fga"
            else:
                # single-valued.  Near the caustic (fold/cusp) -> FGA (ray-based,
                # handles both the divergence and the caustic).  (_caustic_zone's
                # single-row slope model is itself valid only for single-valued
                # fields, so it is reached only on this branch.)
                zone = _caustic_zone(E_in, float(dx), prescription,
                                     float(wavelength))
                near = False
                if zone is not None:
                    pad = caustic_pad_dof * float(wavelength) / (na * na)
                    near = (zone[0] - pad) <= opd <= (zone[1] + pad)
                if near:
                    chosen = "fga"
                else:
                    # smooth plane: the sub-nm traced OPL, BUT traced launches
                    # rays along the local phase gradient and is only valid for a
                    # ~collimated beam -- a single-valued but DIVERGING beam (large
                    # residual angular spread, e.g. a bare point-source relay)
                    # would be silently blurred.  Route those to the wave-exact
                    # phase_screen (apply_real_lens + exact ASM) instead -- bounded
                    # thin-screen OPD error, never a blur.  Uses traced's own
                    # collimation discriminator + threshold so the split matches
                    # exactly where traced stops being valid.
                    from ..elements._lens_traced import (
                        _NONCOLLIMATED_RESID_THRESH,
                        _carrier_residual_rms,
                    )
                    spread = _carrier_residual_rms(E_in, None, float(wavelength),
                                                   float(dx))
                    chosen = ("phase_screen"
                              if spread > _NONCOLLIMATED_RESID_THRESH
                              else "traced")

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
        # phase_screen / traced return at the exit vertex -> finish the output
        # leg with an exact angular-spectrum propagation (the field is smooth
        # here, so ASM is wave-exact even through an intervening focus).
        from ..elements import apply_real_lens, apply_real_lens_traced
        fn = apply_real_lens if chosen == "phase_screen" else apply_real_lens_traced
        exitf = fn(E_in, prescription=prescription, wavelength=wavelength, dx=dx,
                   dy=dy, **mkw.get(chosen, {}))
        if opd != 0.0:
            from .asm import angular_spectrum_propagate
            exitf = angular_spectrum_propagate(exitf, opd, wavelength, dx, dy)
        out = exitf
    return (out, chosen) if return_method else out

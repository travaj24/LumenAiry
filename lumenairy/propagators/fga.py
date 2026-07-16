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
    kernels on first use.  Returns ``(coeff_kernel, scatter_kernel)``."""
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

    return _coeff, _scatter


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


def _gabor_coeff(u0, qx, qy, px, py, x0, y0, dx, dyg, w0, k, Ag, nsig):
    coeff, _ = _kernels()
    cr = np.zeros((qx.shape[0], px.shape[0]))
    ci = np.zeros((qx.shape[0], px.shape[0]))
    coeff(np.ascontiguousarray(u0.real), np.ascontiguousarray(u0.imag),
          qx, qy, px, py, x0, y0, dx, dyg, w0, k, Ag, nsig, cr, ci)
    return cr + 1j * ci


def _reconstruct(Qx, Qy, Px, Py, W, x0, y0, dx, dyg, Ny, Nx, w0, k, Ag, nsig):
    _, scatter = _kernels()
    outr = np.zeros((Ny, Nx))
    outi = np.zeros((Ny, Nx))
    scatter(Qx.ravel(), Qy.ravel(), Px.ravel(), Py.ravel(),
            np.ascontiguousarray(W.real).ravel(),
            np.ascontiguousarray(W.imag).ravel(),
            x0, y0, dx, dyg, Ny, Nx, w0, k, Ag, nsig, outr, outi)
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


def _fga_through_lens(u0, dx, dyg, prescription, wavelength, w0, z_image,
                      dq_step, p_max, n_p, nsig):
    """Core FGA transport through a prescription to (last vertex + z_image).

    ``dx`` / ``dyg`` are the (possibly anamorphic) x / y pixel pitches."""
    from ..raytrace import surfaces_from_prescription
    from ..raytrace.differential import ray_transfer_jacobian

    k = 2.0 * np.pi / wavelength
    Ny, Nx = u0.shape
    x0 = -(Nx / 2) * dx
    y0 = -(Ny / 2) * dyg
    Ag = (1.0 / (np.pi * w0 ** 2)) ** 0.5

    qx, qy, px, py, dp = _swarm_lattice(Ny, Nx, dx, dyg, x0, y0, dq_step,
                                        p_max, n_p)
    Nq = qx.shape[0]
    Np = px.shape[0]
    # phase-space measure * the FGA normalization.  The position measure is the
    # anamorphic lattice cell (dq_step*dx)(dq_step*dyg).  The /2^{d/2} (d=2
    # transverse) removes the double-counted Herman-Kluk identity factor
    # a(0)=2^{d/2}: without it the t=0 resolution of identity over-counts by
    # 2^d=4 in power (verified: the flat-prescription output=0 power ratio -> 4.0
    # in the well-sampled limit, -> 1.0 with this factor).
    C = ((k / (2.0 * np.pi)) ** 2 * (dq_step ** 2 * dx * dyg) * (dp ** 2)) / 2.0

    c = _gabor_coeff(u0, qx, qy, px, py, x0, y0, dx, dyg, w0, k, Ag, nsig)

    # trace to the LAST SURFACE VERTEX; the image-side leg is added manually.
    surfs = [_copy.copy(s) for s in surfaces_from_prescription(prescription)]
    surfs[-1].thickness = 0.0

    QX = np.empty((Nq, Np))
    QY = np.empty((Nq, Np))
    PX = np.empty((Nq, Np))
    PY = np.empty((Nq, Np))
    AW = np.zeros((Nq, Np), dtype=np.complex128)
    ALV = np.zeros((Nq, Np), dtype=bool)
    kw2 = k * w0 * w0
    for ip in range(Np):
        pxi = float(px[ip])
        pyi = float(py[ip])
        pz_in = math.sqrt(max(1.0 - pxi * pxi - pyi * pyi, 1e-12))
        uxin = np.full(Nq, pxi / pz_in)
        uyin = np.full(Nq, pyi / pz_in)
        dt = ray_transfer_jacobian(qx.copy(), qy.copy(), uxin, uyin,
                                   surfs, wavelength, per_surface=False)
        uxo = dt.ux
        uyo = dt.uy
        # manual image-side free-space leg z_image (slope coordinates)
        xv = dt.x + z_image * uxo
        yv = dt.y + z_image * uyo
        opd_tot = dt.opd + z_image * np.sqrt(1.0 + uxo ** 2 + uyo ** 2)
        Mleg = np.tile(np.eye(4), (Nq, 1, 1))
        Mleg[:, 0, 2] = z_image
        Mleg[:, 1, 3] = z_image
        M = Mleg @ dt.jacobian
        # slope -> direction-cosine conjugation for the canonical monodromy
        go = 1.0 / (1.0 + uxo ** 2 + uyo ** 2) ** 1.5      # dp/du at output
        gi = (1.0 + (pxi / pz_in) ** 2 + (pyi / pz_in) ** 2) ** 1.5  # du/dp in
        A = M[:, 0:2, 0:2]
        B = M[:, 0:2, 2:4] * gi
        Cc = M[:, 2:4, 0:2] * go[:, None, None]
        D = M[:, 2:4, 2:4] * (go[:, None, None] * gi)
        Z = (A + D) + 1j * (kw2 * Cc - B / kw2)
        a = np.sqrt(_det2(Z))
        a = np.where(a.real < 0, -a, a)                    # continuous branch
        invo = 1.0 / np.sqrt(1.0 + uxo ** 2 + uyo ** 2)
        QX[:, ip] = xv
        QY[:, ip] = yv
        PX[:, ip] = uxo * invo
        PY[:, ip] = uyo * invo
        AW[:, ip] = C * a * np.exp(1j * k * opd_tot)
        ALV[:, ip] = np.asarray(dt.alive, bool)

    W = c * AW
    W[~ALV] = 0.0
    return _reconstruct(QX, QY, PX, PY, W, x0, y0, dx, dyg, Ny, Nx, w0, k, Ag,
                        nsig)


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
    n_p: int = 15,
    nsig: float = 4.0,
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
        Momentum (direction-cosine) half-range of the swarm.  ``None`` auto-sets
        it from the prescription NA.
    n_p : int
        Momentum samples per transverse axis (swarm has ``n_p**2`` directions).
    nsig : float
        Gaussian window radius in sigmas for the windowed sum (tail
        ``exp(-nsig**2)``).
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
    if p_max is None:
        p_max = _default_p_max(prescription, wavelength)
    w0 = float(w0_factor) * math.sqrt(float(dx) * dyg)
    out = _fga_through_lens(
        E_in, float(dx), dyg, prescription, float(wavelength), w0,
        float(output_plane_distance), int(dq_step), float(p_max), int(n_p),
        float(nsig))
    if normalize_output == "power":
        pin = float(np.sum(np.abs(E_in) ** 2))
        pout = float(np.sum(np.abs(out) ** 2))
        if pout > 0.0:
            out = out * math.sqrt(pin / pout)
    return out


def _fga_vector_through_lens(Ex, Ey, dx, dyg, prescription, wavelength, w0,
                             z_image, dq_step, p_max, n_p, nsig):
    """Vector (Jones) FGA transport: returns ``(Ex, Ey, Ez)`` at the output
    plane.  The scalar transport (ray map + HK weight + OPL) is shared by both
    polarization channels; each beamlet additionally carries the per-beamlet 2x2
    Fresnel Jones matrix (polarization ray tracing, s/p per surface -- the s/p
    frame rotation IS the geometric phase), and the longitudinal ``Ez`` is added
    from the exit-ray directions (``E.k = 0``).  ``dx`` / ``dyg`` are the
    (possibly anamorphic) x / y pixel pitches."""
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
    Nq = qx.shape[0]
    Np = px.shape[0]
    C = ((k / (2.0 * np.pi)) ** 2 * (dq_step ** 2 * dx * dyg) * (dp ** 2)) / 2.0

    cx = _gabor_coeff(Ex, qx, qy, px, py, x0, y0, dx, dyg, w0, k, Ag, nsig)
    cy = _gabor_coeff(Ey, qx, qy, px, py, x0, y0, dx, dyg, w0, k, Ag, nsig)

    surfs = [_copy.copy(s) for s in surfaces_from_prescription(prescription)]
    surfs[-1].thickness = 0.0
    kw2 = k * w0 * w0
    QX = np.empty((Nq, Np))
    QY = np.empty((Nq, Np))
    PX = np.empty((Nq, Np))
    PY = np.empty((Nq, Np))
    Wx = np.zeros((Nq, Np), dtype=np.complex128)
    Wy = np.zeros((Nq, Np), dtype=np.complex128)
    for ip in range(Np):
        pxi = float(px[ip])
        pyi = float(py[ip])
        pz_in = math.sqrt(max(1.0 - pxi * pxi - pyi * pyi, 1e-12))
        uxin = np.full(Nq, pxi / pz_in)
        uyin = np.full(Nq, pyi / pz_in)
        dt = ray_transfer_jacobian(qx.copy(), qy.copy(), uxin, uyin,
                                   surfs, wavelength, per_surface=False)
        J, jalive = _fresnel_jones_matrix_per_beamlet(
            qx.copy(), qy.copy(), uxin, uyin, prescription, wavelength)
        uxo = dt.ux
        uyo = dt.uy
        xv = dt.x + z_image * uxo
        yv = dt.y + z_image * uyo
        opd_tot = dt.opd + z_image * np.sqrt(1.0 + uxo ** 2 + uyo ** 2)
        Mleg = np.tile(np.eye(4), (Nq, 1, 1))
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
        base = C * a * np.exp(1j * k * opd_tot)          # scalar beamlet weight
        invo = 1.0 / np.sqrt(1.0 + uxo ** 2 + uyo ** 2)
        QX[:, ip] = xv
        QY[:, ip] = yv
        PX[:, ip] = uxo * invo
        PY[:, ip] = uyo * invo
        cxi = cx[:, ip]
        cyi = cy[:, ip]
        # apply the 2x2 Jones to the (Ex, Ey) coefficient, weight by the scalar
        ex_out = J[:, 0, 0] * cxi + J[:, 0, 1] * cyi
        ey_out = J[:, 1, 0] * cxi + J[:, 1, 1] * cyi
        alv = np.asarray(dt.alive, bool) & np.asarray(jalive, bool)
        Wx[:, ip] = np.where(alv, base * ex_out, 0.0)
        Wy[:, ip] = np.where(alv, base * ey_out, 0.0)

    ex = _reconstruct(QX, QY, PX, PY, Wx, x0, y0, dx, dyg, Ny, Nx, w0, k, Ag,
                      nsig)
    ey = _reconstruct(QX, QY, PX, PY, Wy, x0, y0, dx, dyg, Ny, Nx, w0, k, Ag,
                      nsig)
    # longitudinal Ez per beamlet: E.k = 0 -> Ez = -(px*Ex + py*Ey)/pz
    PZ = np.sqrt(np.maximum(1.0 - PX ** 2 - PY ** 2, 1e-12))
    Wz = -(PX * Wx + PY * Wy) / PZ
    ez = _reconstruct(QX, QY, PX, PY, Wz, x0, y0, dx, dyg, Ny, Nx, w0, k, Ag,
                      nsig)
    return ex, ey, ez


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
    n_p: int = 15,
    nsig: float = 4.0,
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
    if p_max is None:
        p_max = _default_p_max(prescription, wavelength)
    w0 = float(w0_factor) * math.sqrt(float(dx) * dyg)
    ex, ey, ez = _fga_vector_through_lens(
        E_vec[0], E_vec[1], float(dx), dyg, prescription, float(wavelength), w0,
        float(output_plane_distance), int(dq_step), float(p_max), int(n_p),
        float(nsig))
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
      smooth (single-valued, guaranteed by the no-caustic branch): per-pixel
      ray-traced OPL, sub-nm, no thin-screen ceiling.

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
        if _system_na(prescription, wavelength) < float(na_threshold):
            chosen = "phase_screen"
        else:
            zone = _caustic_zone(E_in, float(dx), prescription, float(wavelength))
            near = False
            if zone is not None:
                na = _system_na(prescription, wavelength)
                pad = caustic_pad_dof * float(wavelength) / (na * na)
                near = (zone[0] - pad) <= opd <= (zone[1] + pad)
            chosen = "fga" if near else "traced"

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

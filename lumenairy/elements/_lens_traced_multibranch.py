"""
lumenairy.elements._lens_traced_multibranch -- MULTI-BRANCH (multi-arrival)
ray-traced lens field, valid THROUGH focus and caustics.

:func:`apply_real_lens_traced` assumes the entrance->output ray map is
single-valued (one ray per output pixel), so it breaks down through focus /
at caustics where several rays arrive at the same point.  This module
constructs the full MULTI-VALUED geometric field by the wavefront-construction
/ triangulated-ray-map method of the seismology literature:

* Triangulate the regular entrance launch grid; map each triangle by its
  traced output-plane vertex positions; every output pixel covered by a
  mapped triangle receives ONE arrival branch from it (Lambare, Lucio &
  Hanyga 1996, Geophys. J. Int. 125, 584, Secs 5-6; Vinje, Iversen &
  Gjoystdal 1993, Geophysics 58, 1157).
* Per-branch OPL by the second-order "intrapolation" formula
  ``T(x) = sum_i alpha_i [ T_i + 1/2 (x - x_i) . p_i ]`` with ``p_i`` the
  transverse direction cosines (Kraaijpoel 2003, PhD thesis Utrecht,
  eq. 5.7 -- the 1/2 is exact for quadratic T, NOT a typo).
* Per-branch amplitude ``1/sqrt|J|`` from the SIGNED-AREA RATIO of the mapped
  to launch triangle (Chambers & Kendall 2008, Geophys. J. Int. 173, 1030,
  eq. 1 "ratio method" -- more robust near degenerate configurations than the
  differential Jacobian).
* KMAH (Maslov) index per ray: on the homogeneous exit leg the transverse
  Jacobian is LINEAR in distance, so ``det Q(z) = det Q0 + z tr(adj(Q0) Qdot)
  + z^2 det Qdot`` is an exact quadratic whose roots are the two astigmatic
  focal-line crossings -- caustic passages are counted analytically (Cerveny,
  Seismic Ray Theory, CUP 2001 dynamic ray tracing; Klimes 2010, Stud.
  Geophys. Geod. 54, 269, App. B; Cormier, Treatise on Geophysics ch. 3.04
  eq. 23).  Each fold crossing multiplies the branch by ``exp(-i pi/2)``
  (time convention ``exp(-i w t)``, field ``exp(+i k OPL)`` -- Klimes 2010
  Sec. 1 verbatim; = the Gouy phase in optics, Visser & Wolf 2010).
* In-glass caustics: each surface-to-surface leg is a homogeneous straight
  segment, so its ``det Q(z)`` is the same exact quadratic the exit leg uses
  -- the internal focal-line crossings are counted per leg and added to the
  index (0 for an air-focus system).  A runtime parity assertion
  ``sgn det J = (-1)^m`` (Mitrofanov & Priimenko 2013 eq. 1) then guarantees
  the EXACT parity; if the direct in-glass count disagrees with it (an
  internal focus the global-frame leg quadratic under-resolves), the parity
  is still enforced and a warning is raised.
* AT the caustic itself (degenerate mapped triangles) the ART amplitude is
  not evaluated -- the literature-standard choice (Vinje 1993 leaves it
  undefined; Lambare 1996 notes the singularity is inherent to ART); the
  caustic band is O(k^-2/3) wide (~1 output pixel at optical wavelengths,
  Kravtsov & Orlov criterion k|dS| <~ pi), and the plain two-branch sum is
  accurate "up to the first Airy peak" outside it.

Author: Andrew Traverso
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np

from .. import raytrace as rt

__all__ = ['apply_real_lens_traced_multibranch', 'ludwig_fold']


def ludwig_fold(k, S_plus, S_minus, A_plus, A_minus):
    """Uniform (Ludwig 1966 / Kravtsov 1964) FOLD-caustic field from the two
    coalescing ray branches -- finite exactly where the branch amplitudes
    diverge, in the RAY-NATIVE parametrization (branch eikonals + amplitudes).

    For two branches with eikonals ``S+ >= S-`` [m] and COMPLEX amplitudes
    ``A+-`` (each INCLUDING its own Maslov/SPA phase; '+' = the branch that has
    touched the caustic)::

        phi = (S+ + S-)/2,   rho = [3/4 (S+ - S-)]^(2/3)
        g0  = rho^{1/4}/sqrt2 (A+ - i A-)
        g1  = rho^{-1/4}/sqrt2 (A+ + i A-)
        u   = sqrt(2 pi) k^{1/6} e^{i pi/4} e^{i k phi}
              [ g0 Ai(-k^{2/3} rho) + i k^{-1/3} g1 Ai'(-k^{2/3} rho) ]

    (Ludwig, Comm. Pure Appl. Math. 19, 215 (1966), as reproduced in
    Giannopoulou/Makrakis arXiv:1706.02958 eqs. 2.30-2.39 and Grillo & Cordes
    arXiv:1810.09058 eqs. 31-44.)  The combinations ``(A+ - iA-)rho^{1/4}`` and
    ``(A+ + iA-)rho^{-1/4}`` stay FINITE at the fold although ``A+-`` diverge;
    on the bright side the formula reduces exactly to the plain two-branch sum
    ``A+ e^{ikS+} + A- e^{ikS-}``.  Validated to machine precision (~3e-16)
    against the exact cubic-phase integral for both regimes, including exactly
    on the caustic.  Use inside the Kravtsov-Orlov band ``k|S+ - S-| <~ pi`` to
    replace a coalescing pair in a multi-branch sum (the plain sum is accurate
    outside it, "up to the first Airy peak").  Amplitudes must be COMPLEX
    (real amplitudes without their Maslov phases give wrong interference).
    """
    from scipy.special import airy
    phi = 0.5 * (S_plus + S_minus)
    rho = (0.75 * (S_plus - S_minus)) ** (2.0 / 3.0)
    r14 = rho ** 0.25
    g0 = r14 / np.sqrt(2.0) * (A_plus - 1j * A_minus)
    g1 = (A_plus + 1j * A_minus) / (r14 * np.sqrt(2.0) + 1e-300)
    ai, aip, _, _ = airy(-(k ** (2.0 / 3.0)) * rho)
    return (np.sqrt(2 * np.pi) * k ** (1.0 / 6.0) * np.exp(1j * np.pi / 4)
            * np.exp(1j * k * phi)
            * (g0 * ai + 1j * k ** (-1.0 / 3.0) * g1 * aip))


def _trace_launch_grid(prescription, wavelength, launch_radius, n_launch,
                       output_plane_distance, output_plane_n,
                       L0=0.0, M0=0.0):
    """Trace a regular entrance grid to the output plane.

    ``L0``/``M0`` are the launch direction cosines of the input congruence
    (0 = collimated on-axis; a tilted/carrier input launches every ray along
    its input phase plane).  The input eikonal on the ``z = 0`` entrance
    plane, ``T_in = L0 x + M0 y``, is included in the returned OPL so the
    branch phases carry the full input carrier exactly.

    Returns dict of (n_launch, n_launch) grids: entrance coords, output-plane
    transverse positions, OPL [m], exit direction cosines, alive mask.
    """
    surfaces = rt.surfaces_from_prescription(prescription)
    xs_in = np.linspace(-launch_radius, launch_radius, n_launch)
    Xi, Yi = np.meshgrid(xs_in, xs_in, indexing='ij')
    n_rays = Xi.size
    N0 = np.sqrt(max(0.0, 1.0 - L0 * L0 - M0 * M0))
    rays = rt.RayBundle(
        x=Xi.ravel().copy(), y=Yi.ravel().copy(), z=np.zeros(n_rays),
        L=np.full(n_rays, float(L0)), M=np.full(n_rays, float(M0)),
        N=np.full(n_rays, N0),
        wavelength=wavelength,
        alive=np.ones(n_rays, dtype=bool),
        opd=np.zeros(n_rays),
    )
    tr = rt.trace(rays, surfaces, wavelength)
    ex = tr.image_rays
    alive = ex.alive.copy()
    # Per-surface state (x, y, z, outgoing slopes) on the launch grid -- for the
    # in-glass KMAH count (D3): each surface-to-surface leg is a homogeneous
    # straight segment, so det Q(z) is the same exact quadratic the exit leg
    # uses, and its roots are the in-glass focal-line crossings.
    shape = (n_launch, n_launch)
    history = []
    for hb in (tr.ray_history or []):
        hNz = np.where(np.abs(hb.N) > 1e-30, hb.N, 1e-30)
        history.append({
            'x': np.asarray(hb.x, float).reshape(shape),
            'y': np.asarray(hb.y, float).reshape(shape),
            'z': np.asarray(hb.z, float).reshape(shape),
            'sx': (np.asarray(hb.L, float) / hNz).reshape(shape),
            'sy': (np.asarray(hb.M, float) / hNz).reshape(shape),
        })
    # advance the exit rays to the output plane a distance d past the exit
    # vertex (free space, index output_plane_n): path length t = d / N_z.
    Nz = np.where(np.abs(ex.N) > 1e-30, ex.N, 1e-30)
    t = output_plane_distance / Nz
    x_out = ex.x + t * ex.L
    y_out = ex.y + t * ex.M
    opl = (ex.opd + float(output_plane_n) * t
           + (L0 * Xi + M0 * Yi).ravel())
    bad = ~alive
    for arr in (x_out, y_out, opl):
        arr[bad] = np.nan
    return {
        'xs_in': xs_in,
        'Xi': Xi, 'Yi': Yi,
        'x_out': x_out.reshape(shape), 'y_out': y_out.reshape(shape),
        'opl': opl.reshape(shape),
        'L': ex.L.reshape(shape), 'M': ex.M.reshape(shape),
        'N': ex.N.reshape(shape),
        'x_exit': ex.x.reshape(shape), 'y_exit': ex.y.reshape(shape),
        'alive': alive.reshape(shape),
        'history': history,
    }


def _count_fold_roots(detQ0, trK, detQd, zmax):
    """Count the roots of the exact quadratic ``det Q(z) = detQ0 + z trK +
    z^2 detQd`` in ``(0, zmax]`` (with multiplicity) per grid node -- the
    astigmatic focal-line crossings on one homogeneous leg.  Shared by the
    exit-leg and the in-glass-leg KMAH counts."""
    A, B, C = detQd, trK, detQ0
    m = np.zeros(np.shape(detQ0), dtype=np.int64)
    with np.errstate(invalid='ignore', divide='ignore'):
        disc = B * B - 4.0 * A * C
        sq = np.sqrt(np.maximum(disc, 0.0))
        quad = np.abs(A) > 1e-300
        for sgn in (-1.0, 1.0):
            z = np.where(quad, (-B + sgn * sq) / (2.0 * A), np.nan)
            m += ((disc >= 0.0) & quad & (z > 0.0)
                  & (z <= zmax)).astype(int)
        # linear degenerate case (det Qdot ~ 0): the single root -C/B
        lin = (~quad) & (np.abs(B) > 1e-300)
        zlin = np.where(lin, -C / np.where(lin, B, 1.0), np.nan)
        m += (lin & (zlin > 0.0) & (zlin <= zmax)).astype(int)
    return m


def _kmah_in_glass(g):
    """In-glass (surface-to-surface) fold-caustic count per launch node (D3).

    The exit-leg count (:func:`_kmah_free_leg`) plus a mod-2 parity closure
    left an EVEN number of internal caustic crossings invisible (a full focus
    between two elements = 2 crossings => a pi Maslov-index error).  Each
    surface-to-surface leg is a homogeneous straight segment, so its transverse
    Jacobian is the same exact quadratic ``det Q(z)`` the exit leg uses; sum its
    roots over every internal leg.  Returns the per-node in-glass count (0 when
    no ray focuses inside the prescription -- so an air-focus system is
    byte-identical to the prior release)."""
    hist = g.get('history') or []
    if len(hist) < 2:
        return np.zeros_like(g['x_exit'], dtype=np.int64)
    h = g['xs_in'][1] - g['xs_in'][0]

    def _grad(F):
        return np.gradient(F, h, h)          # d/d(x_in), d/d(y_in)

    m_ig = np.zeros(g['x_exit'].shape, dtype=np.int64)
    for k in range(len(hist) - 1):
        s0, s1 = hist[k], hist[k + 1]
        a11, a12 = _grad(s0['x'])
        a21, a22 = _grad(s0['y'])
        b11, b12 = _grad(s0['sx'])
        b21, b22 = _grad(s0['sy'])
        detQ0 = a11 * a22 - a12 * a21
        detQd = b11 * b22 - b12 * b21
        trK = a22 * b11 - a12 * b21 - a21 * b12 + a11 * b22
        leg = np.sqrt((s1['x'] - s0['x']) ** 2 + (s1['y'] - s0['y']) ** 2
                      + (s1['z'] - s0['z']) ** 2)
        with np.errstate(invalid='ignore'):
            m_ig += np.where(np.isfinite(leg),
                             _count_fold_roots(detQ0, trK, detQd, leg), 0)
    return m_ig


def _kmah_free_leg(g, d_out):
    """Analytic KMAH count on the homogeneous exit leg (Klimes/Cerveny).

    The transverse Jacobian is linear in the leg distance ``z``:
    ``Q(z) = Q0 + z Qdot`` with ``Q0 = d(x_exit)/d(gamma)`` and
    ``Qdot = d(slope)/d(gamma)`` (slope = (L/N, M/N)); hence
    ``det Q(z) = det Q0 + z tr(adj(Q0) Qdot) + z^2 det Qdot`` -- an exact
    quadratic whose roots in ``(0, d_out]`` are the astigmatic focal-line
    crossings.  Count them (with multiplicity) per launch node.  ``Q0``/
    ``Qdot`` are central finite differences on the launch grid.
    """
    h = g['xs_in'][1] - g['xs_in'][0]

    def _grad(F):
        dF_di, dF_dj = np.gradient(F, h, h)
        return dF_di, dF_dj   # d/d(x_in), d/d(y_in)  (indexing='ij')

    Nz = np.where(np.abs(g['N']) > 1e-30, g['N'], 1e-30)
    sx = g['L'] / Nz
    sy = g['M'] / Nz
    # Q0 (exit positions) and Qdot (slopes) w.r.t. launch coords
    a11, a12 = _grad(g['x_exit'])
    a21, a22 = _grad(g['y_exit'])
    b11, b12 = _grad(sx)
    b21, b22 = _grad(sy)
    detQ0 = a11 * a22 - a12 * a21
    detQd = b11 * b22 - b12 * b21
    # tr(adj(Q0) Qdot) with adj([[a,b],[c,d]]) = [[d,-b],[-c,a]]
    trK = a22 * b11 - a12 * b21 - a21 * b12 + a11 * b22
    # exit-leg roots of detQ0 + z trK + z^2 detQd in (0, d_out]
    m = _count_fold_roots(detQ0, trK, detQd, d_out)
    # IN-GLASS legs (D3): the exit-leg count plus a mod-2 parity closure left an
    # EVEN internal caustic count invisible (a focus between two elements = 2
    # crossings => a pi error).  Add the per-leg quadratic count over every
    # surface-to-surface leg; 0 for an air-focus system (byte-identical to the
    # prior release), so existing validated charts are unchanged.
    m_ig = _kmah_in_glass(g)
    m = m + m_ig
    # EXACT parity guarantee (Mitrofanov & Priimenko eq. 1), retained: sgn
    # det J(out) must be (-1)^m relative to the launch (det=+1).  If the
    # combined exit+in-glass count still has the wrong parity, the in-glass
    # magnitude is off by an odd amount -> restore the exact parity by +1
    # (what the old closure did).  Where that correction fires on a node the
    # in-glass detector ALSO flagged (m_ig>0), the two disagree -> the
    # global-frame in-glass approximation is unreliable there; warn once.
    c11, c12 = _grad(g['x_out'])
    c21, c22 = _grad(g['y_out'])
    detJ_out = c11 * c22 - c12 * c21
    parity_bad = (np.sign(detJ_out) != (-1.0) ** m) & np.isfinite(detJ_out)
    m = m + parity_bad.astype(int)
    if bool(np.any(parity_bad & (m_ig > 0))):
        warnings.warn(
            "apply_real_lens_traced_multibranch: the in-glass KMAH count and "
            "the exact parity closure disagree on some launch nodes (an "
            "internal focus inside the prescription that the global-frame "
            "leg quadratic under/over-resolves); the branch Maslov index may "
            "carry a residual pi error there.  Validate against the GBD or "
            "Maslov propagator for internal-focus prescriptions.",
            stacklevel=2)
    # fill nodes whose FD Jacobian was NaN-contaminated by a dead neighbour
    # (their own ray may be alive and used by adjacent triangles): nearest
    # valid neighbour, iteratively (KMAH is piecewise constant per sheet).
    bad = ~(np.isfinite(detQ0) & np.isfinite(detJ_out))
    it = 0
    while bad.any() and it < max(m.shape):
        it += 1
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            src_m = np.roll(m, sh, axis=ax)
            src_ok = np.roll(~bad, sh, axis=ax)
            take = bad & src_ok
            m[take] = src_m[take]
            bad[take] = False
    return m, detJ_out


def apply_real_lens_traced_multibranch(
    E_in: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    output_plane_distance: float = 0.0,
    output_plane_n: float = 1.0,
    ray_subsample: int = 2,
    min_area_ratio: float = 1e-6,
    caustic_band: str = 'ludwig',
    input_carrier: Any = None,
    return_diagnostics: bool = False,
) -> Any:
    """Multi-branch ray-traced lens field, valid THROUGH focus / caustics.

    Constructs the coherent multi-arrival geometric field at the output plane
    ``output_plane_distance`` past the prescription's exit vertex::

        E(x) = sum_branches  a_br(x) sqrt(1/|J_br|)
               * exp(i k OPL_br(x) - i (pi/2) m_br)

    by the triangulated-ray-map / wavefront-construction method (see module
    docstring for the full literature provenance).  Where the single-valued
    :func:`apply_real_lens_traced` breaks down (several rays per pixel through
    focus), every arrival is summed with its ``1/sqrt|J|`` amplitude and its
    KMAH (Maslov / Gouy) phase ``exp(-i pi m/2)`` -- fold caustics counted
    ANALYTICALLY on the exit leg via the exact quadratic ``det Q(z)``.

    Scope: slowly-varying input ENVELOPE with an optional linear carrier /
    tilt (``input_carrier``; the launch congruence follows the input phase
    plane and the carrier phase is carried exactly by the branch eikonals).
    ``E_in`` supplies per-ray complex envelope amplitude.  The ART amplitude
    is undefined ON the caustic itself (degenerate triangles are skipped --
    literature-standard); the caustic band is ~1 output pixel wide at
    optical wavelengths.

    ``caustic_band='ludwig'`` (default) applies the literature-standard local
    uniform replacement in the Kravtsov-Orlov band: at pixels where a
    coalescing branch pair has ``k |S+ - S-| <= pi``, the pair's plain sum
    (which diverges toward the fold) is swapped for the Ludwig uniform-Airy
    field :func:`ludwig_fold` -- finite exactly where the branch amplitudes
    diverge, and reducing to the plain sum outside the band by construction
    (Kravtsov 1964; Ludwig 1966; Grillo & Cordes 2018 eq. 47).  All other
    branches at the pixel keep their plain contributions.
    ``caustic_band='plain'`` keeps the raw branch sum everywhere.

    Parameters
    ----------
    E_in : (N, N) complex -- input field at the lens entrance plane.
    output_plane_distance : float -- output plane distance past the exit
        vertex [m] (e.g. near the focal distance for through-focus study).
    ray_subsample : int -- launch-grid spacing in units of ``dx`` (2 -> one
        ray per 2 pixels).  Smaller = denser branches = more accurate.
    min_area_ratio : float -- skip triangles whose mapped area is below this
        fraction of their launch area (the degenerate caustic set).
    input_carrier : None | 'auto' | (kx, ky) -- transverse carrier
        wavevector of the input field [rad/m].  ``None`` (default) assumes a
        collimated on-axis congruence (v1 behaviour).  ``'auto'`` estimates
        the mean carrier from ``E_in`` by the lag-1 correlation phase (exact
        for carrier x smooth envelope, subpixel; the carrier must be
        resolvable on the pixel grid).  A ``(kx, ky)`` tuple uses the given
        carrier directly (works even for super-Nyquist carriers, since the
        envelope is sampled carrier-stripped).  The launch congruence is
        tilted along the input phase plane (direction cosines ``kx/k0``,
        ``ky/k0``), the input eikonal is carried exactly by the branch
        phases, and the envelope is bilinearly sampled with the carrier
        removed (no aliasing).  Small-tilt scope: the ART amplitude keeps
        the v1 transverse-area ratio (obliquity ``cos theta`` factors are
        neglected -- <0.1% below ~2.5 deg).
    return_diagnostics : bool -- also return a dict with the per-node KMAH
        map, det J, branch-count image, and the resolved ``input_carrier``.

    Returns
    -------
    (N, N) complex output field (plus diagnostics dict if requested).
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_traced_multibranch')
    E_in = np.asarray(E_in)
    N = E_in.shape[0]
    if E_in.shape[0] != E_in.shape[1]:
        raise ValueError("apply_real_lens_traced_multibranch: square grid "
                         f"required; got {E_in.shape}.")
    k0 = 2.0 * np.pi / wavelength

    # ---- input carrier / tilt (v2): launch along the input phase plane ---
    if input_carrier is None or (isinstance(input_carrier, str)
                                 and input_carrier == 'none'):
        kcx = kcy = 0.0
    elif isinstance(input_carrier, str) and input_carrier == 'auto':
        # lag-1 correlation phase: exact for carrier x smooth envelope
        # (E_in is indexed [y, x]: axis 1 = x, axis 0 = y)
        kcx = float(np.angle(np.nansum(
            E_in[:, 1:] * np.conj(E_in[:, :-1])))) / dx
        kcy = float(np.angle(np.nansum(
            E_in[1:, :] * np.conj(E_in[:-1, :])))) / dx
    else:
        kcx, kcy = (float(input_carrier[0]), float(input_carrier[1]))
    L0 = kcx / k0
    M0 = kcy / k0
    if L0 * L0 + M0 * M0 >= 1.0:
        raise ValueError(
            "apply_real_lens_traced_multibranch: input_carrier "
            f"(kx={kcx:.4g}, ky={kcy:.4g} rad/m) is evanescent at this "
            "wavelength.")
    if kcx != 0.0 or kcy != 0.0:
        # strip the carrier before envelope sampling (re-added exactly via
        # the launch eikonal T_in = L0 x + M0 y in the branch phases)
        xs_px = (np.arange(N) - N / 2.0) * dx
        XP, YP = np.meshgrid(xs_px, xs_px)          # [y, x]: XP cols = x
        E_work = E_in * np.exp(-1j * (kcx * XP + kcy * YP))
    else:
        E_work = E_in

    aperture = prescription.get('aperture_diameter')
    if aperture is not None:
        # stay strictly inside the aperture rim: rays AT the rim can die
        # (vignetting / missed caps) and NaN-poison the finite-difference
        # Jacobians of their neighbours -- exactly the marginal rays whose
        # caustic crossings matter most.
        launch_radius = 0.5 * float(aperture) * 0.98
    else:
        launch_radius = 0.5 * N * dx
    sub = max(1, int(ray_subsample))
    n_launch = max(9, int(2 * launch_radius / (dx * sub)))
    if n_launch % 2 == 0:
        n_launch += 1

    g = _trace_launch_grid(prescription, wavelength, launch_radius, n_launch,
                           output_plane_distance, output_plane_n,
                           L0=L0, M0=M0)
    m_grid, detJ = _kmah_free_leg(g, output_plane_distance)

    # sample the input field at the launch nodes (bilinear)
    xs_in = g['xs_in']
    fi = np.clip((xs_in / dx) + N / 2.0, 0.0, N - 1.0 - 1e-9)
    i0 = np.floor(fi).astype(int)
    w = fi - i0
    # E_work[y, x] (carrier-stripped envelope) with our launch grids indexed
    # (i=x_in, j=y_in)
    Exx = ((1 - w)[None, :] * ((1 - w)[:, None] * E_work[np.ix_(i0, i0)]
                               + w[:, None] * E_work[np.ix_(i0 + 1, i0)])
           + w[None, :] * ((1 - w)[:, None] * E_work[np.ix_(i0, i0 + 1)]
                           + w[:, None] * E_work[np.ix_(i0 + 1, i0 + 1)]))
    # Exx[j_y, i_x] -> transpose to [i_x, j_y] to match Xi/Yi (indexing='ij')
    E_launch = Exx.T

    # per-node data for the triangulation
    XO = g['x_out']
    YO = g['y_out']
    OPL = g['opl']
    L = g['L']
    M = g['M']
    ok = g['alive'] & np.isfinite(XO) & np.isfinite(OPL)

    h = xs_in[1] - xs_in[0]
    tri_launch_area = 0.5 * h * h

    if caustic_band not in ('ludwig', 'plain'):
        raise ValueError(
            "apply_real_lens_traced_multibranch: caustic_band must be "
            f"'ludwig' or 'plain', got {caustic_band!r}")

    # per-branch contribution lists (assembled after the loop so the caustic
    # band can swap coalescing PAIRS for the Ludwig uniform-fold field)
    br_idx: list = []      # flat pixel index (x-major)
    br_T: list = []        # branch eikonal OPL [m]
    br_A: list = []        # COMPLEX branch amplitude incl. its Maslov phase
    n_branch = np.zeros((N, N), dtype=np.int32)

    # triangulate each launch cell into 2 triangles (fixed diagonal), map,
    # rasterize output pixels via barycentric coords (Lambare Fig. 5).
    # VECTORIZED: all per-triangle setup on flat arrays, then rasterization
    # batched in power-of-2 bounding-box buckets -- the same math and the
    # same contribution set as a per-triangle loop, only the float summation
    # order inside np.add.at differs.
    ok4 = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    ci, cj = np.nonzero(ok4)
    # two triangle families per cell:
    # A = (i, j), (i+1, j), (i, j+1);  B = (i+1, j), (i+1, j+1), (i, j+1)
    V0i = np.concatenate([ci, ci + 1])
    V0j = np.concatenate([cj, cj])
    V1i = np.concatenate([ci + 1, ci + 1])
    V1j = np.concatenate([cj, cj + 1])
    V2i = np.concatenate([ci, ci])
    V2j = np.concatenate([cj + 1, cj + 1])
    x0 = XO[V0i, V0j]
    y0 = YO[V0i, V0j]
    x1 = XO[V1i, V1j]
    y1 = YO[V1i, V1j]
    x2 = XO[V2i, V2j]
    y2 = YO[V2i, V2j]
    # signed area of the mapped triangle (fold parity + Jacobian)
    area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    with np.errstate(invalid='ignore'):
        ratio = np.abs(0.5 * area2) / tri_launch_area
        good = np.isfinite(area2) & (ratio >= min_area_ratio)
    if good.any():
        xmn = np.minimum(np.minimum(x0, x1), x2)[good]
        xmx = np.maximum(np.maximum(x0, x1), x2)[good]
        ymn = np.minimum(np.minimum(y0, y1), y2)[good]
        ymx = np.maximum(np.maximum(y0, y1), y2)[good]
        pxmin = np.maximum(0, np.floor(xmn / dx + N / 2.0).astype(np.int64))
        pxmax = np.minimum(N - 1,
                           np.ceil(xmx / dx + N / 2.0).astype(np.int64))
        pymin = np.maximum(0, np.floor(ymn / dx + N / 2.0).astype(np.int64))
        pymax = np.minimum(N - 1,
                           np.ceil(ymx / dx + N / 2.0).astype(np.int64))
        keep = (pxmax >= pxmin) & (pymax >= pymin)

        def _g(a):
            return a[good][keep]

        x0k, y0k, x1k, y1k, x2k, y2k = (
            _g(x0), _g(y0), _g(x1), _g(y1), _g(x2), _g(y2))
        den = _g(area2)
        amp_J = 1.0 / np.sqrt(_g(ratio))
        # KMAH: vertices of one branch share m; take the majority (median)
        mtri = np.stack([m_grid[V0i, V0j], m_grid[V1i, V1j],
                         m_grid[V2i, V2j]], axis=1).astype(np.int64)
        m_br = np.sort(mtri, axis=1)[:, 1][good][keep]
        ph_br = np.exp(-0.5j * np.pi * m_br)
        T0 = _g(OPL[V0i, V0j])
        T1 = _g(OPL[V1i, V1j])
        T2 = _g(OPL[V2i, V2j])
        # eikonal-gradient transverse components p = n*(L, M) for the
        # intrapolation below.  In an index-n output space the phase slowness
        # is n*(L, M), NOT the bare direction cosines (D4); pn = 1 for a
        # vacuum output plane, so this is byte-identical there.  (The OPL
        # advance already carries the n factor -- output_plane_n * t above.)
        pn = float(output_plane_n)
        L0 = pn * _g(L[V0i, V0j])
        L1 = pn * _g(L[V1i, V1j])
        L2 = pn * _g(L[V2i, V2j])
        M0 = pn * _g(M[V0i, V0j])
        M1 = pn * _g(M[V1i, V1j])
        M2 = pn * _g(M[V2i, V2j])
        E0 = _g(E_launch[V0i, V0j])
        E1 = _g(E_launch[V1i, V1j])
        E2 = _g(E_launch[V2i, V2j])
        pxmin, pxmax = pxmin[keep], pxmax[keep]
        pymin, pymax = pymin[keep], pymax[keep]
        nb_flat = n_branch.reshape(-1)

        wmax = np.maximum(pxmax - pxmin, pymax - pymin) + 1
        cls = np.ceil(np.log2(wmax)).astype(np.int64)
        for c in np.unique(cls):
            s = np.nonzero(cls == c)[0]
            wb = int(2 ** c)
            off = np.arange(wb, dtype=np.int64)
            gx = pxmin[s, None, None] + off[None, :, None]
            gy = pymin[s, None, None] + off[None, None, :]
            vmask = ((gx <= pxmax[s, None, None])
                     & (gy <= pymax[s, None, None]))
            PX = (gx - N / 2.0) * dx
            PY = (gy - N / 2.0) * dx
            X0 = x0k[s, None, None]
            Y0 = y0k[s, None, None]
            # barycentric coordinates
            d00x = (x1k - x0k)[s, None, None]
            d00y = (y1k - y0k)[s, None, None]
            d01x = (x2k - x0k)[s, None, None]
            d01y = (y2k - y0k)[s, None, None]
            wx = PX - X0
            wy = PY - Y0
            dn = den[s, None, None]
            a1 = (wx * d01y - wy * d01x) / dn
            a2 = (d00x * wy - d00y * wx) / dn
            a0 = 1.0 - a1 - a2
            inside = (a0 >= 0.0) & (a1 >= 0.0) & (a2 >= 0.0) & vmask
            if not inside.any():
                continue
            # second-order intrapolated OPL (Kraaijpoel eq. 5.7):
            # T = sum_i a_i [T_i + 1/2 (x - x_i).p_i], p = n*(L, M) (D4)
            T = (a0 * (T0[s, None, None]
                       + 0.5 * ((PX - X0) * L0[s, None, None]
                                + (PY - Y0) * M0[s, None, None]))
                 + a1 * (T1[s, None, None]
                         + 0.5 * ((PX - x1k[s, None, None])
                                  * L1[s, None, None]
                                  + (PY - y1k[s, None, None])
                                  * M1[s, None, None]))
                 + a2 * (T2[s, None, None]
                         + 0.5 * ((PX - x2k[s, None, None])
                                  * L2[s, None, None]
                                  + (PY - y2k[s, None, None])
                                  * M2[s, None, None])))
            Ein_tri = (a0 * E0[s, None, None] + a1 * E1[s, None, None]
                       + a2 * E2[s, None, None])
            # branch record: complex amplitude WITH its Maslov phase, and
            # the eikonal separately (needed by the Ludwig pair-swap)
            flat = (gx * N + gy)[inside]
            br_idx.append(flat)
            br_T.append(T[inside])
            br_A.append((Ein_tri * amp_J[s, None, None]
                         * ph_br[s, None, None])[inside])
            np.add.at(nb_flat, flat, 1)

    # ---- assemble the coherent branch sum ------------------------------
    E_flat = np.zeros(N * N, dtype=np.complex128)
    if br_idx:
        bi = np.concatenate(br_idx)
        bT = np.concatenate(br_T)
        bA = np.concatenate(br_A)
        plain = bA * np.exp(1j * k0 * bT)
        np.add.at(E_flat, bi, plain)
        if caustic_band == 'ludwig':
            # In the Kravtsov-Orlov caustic band (k|S+ - S-| <= pi between a
            # coalescing branch pair) the plain two-branch sum is invalid --
            # swap the pair for the Ludwig uniform fold field (finite exactly
            # where the branch amplitudes diverge; reduces to the plain sum
            # outside the band by construction).  Grillo & Cordes eq. 47:
            # uniform-Airy for the coalescing pair, plain GO for the rest.
            order = np.argsort(bi, kind='stable')
            bi_s = bi[order]
            starts = np.flatnonzero(np.r_[True, bi_s[1:] != bi_s[:-1]])
            ends = np.r_[starts[1:], bi_s.size]
            band = np.pi / k0
            for s, e in zip(starts, ends):
                if e - s < 2:
                    continue
                sel = order[s:e]
                Ts = bT[sel]
                # the closest-eikonal pair
                o2 = np.argsort(Ts)
                dT = np.diff(Ts[o2])
                jmin = int(np.argmin(dT))
                if dT[jmin] > band:
                    continue
                ia = sel[o2[jmin]]        # lower-S branch  (S-)
                ib = sel[o2[jmin + 1]]    # higher-S branch (S+)
                # floor the eikonal split (removable 0/0 in the g1 term
                # exactly at coalescence)
                Sm, Sp = bT[ia], bT[ib]
                if Sp - Sm < 1e-4 * wavelength:
                    Sp = Sm + 1e-4 * wavelength
                uni = ludwig_fold(k0, Sp, Sm, bA[ib], bA[ia])
                E_flat[bi_s[s]] += uni - (plain[ia] + plain[ib])
    E_out = E_flat.reshape(N, N)

    # our accumulation is indexed [x, y]; the library field convention is
    # E[y, x] -- transpose on return.
    E_out = E_out.T
    n_branch = n_branch.T
    if return_diagnostics:
        return E_out, {'kmah': m_grid, 'detJ': detJ, 'n_branch': n_branch,
                       'input_carrier': (kcx, kcy)}
    return E_out

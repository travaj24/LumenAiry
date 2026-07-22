"""
lumenairy.elements._lens_traced_uniform -- UNIFORM (Airy) dark-side completion
of the multibranch ray-density caustic field (niche N16 / K4).

The K1 multibranch KMAH ray-density sum
(:func:`lumenairy.elements._lens_traced_multibranch.apply_real_lens_traced_multibranch`)
is a purely GEOMETRIC (ART) construction: on the DARK side of a fold caustic no
real ray branch exists, so the sum is identically ZERO there and misses the
exponentially-decaying Airy tail (fold-truth: windowed r2m ~15% low, ~20% of the
caustic energy missing).  This module closes that gap by the Chester-Friedman-
Ursell / Ludwig UNIFORM asymptotic: near a fold the field is one Airy function
``Ai(-k^{2/3} zeta)`` -- oscillatory on the bright side (``zeta > 0``) and an
``Ai(+)`` exponential tail on the DARK side (``zeta < 0``).

Method (the ROBUST fit-and-continue route of plan N16, for a rotationally-
symmetric fold RING -- the caustic_fold_ref regime):

  1. Run the K1 multibranch to get the finite, ludwig-regularized BRIGHT-side
     field (interior + near-caustic).  This module only ADDS the dark tail.
  2. A meridional ray trace (via :mod:`lumenairy.raytrace`) of the same
     prescription to the output plane gives the fold geometry EXACTLY: the
     caustic-ring radius ``r_c`` (the turning point of the ray height
     ``y_obs(h)``), the fold parameter ``zeta(r) = [3/4 (S+ - S-)]^{2/3}``
     (linear in ``r_c - r`` near the fold -- ``zeta = kappa (r_c - r)``, from the
     eikonal difference of the two coalescing branches), and the mean phase
     ``A(r) = (S+ + S-)/2`` (a smooth quadratic).
  3. The two CFU amplitude coefficients ``a0, a1`` are SMOOTH (real-analytic)
     through the caustic but are hard to get accurately from the raw geometric
     ray-tube amplitudes near a sharp/asymmetric fold, so they are FIT (complex
     least squares) to the multibranch BRIGHT field over a band just inside
     ``r_c``, using the exact ``uniform_fold_airy`` CFU kernel
     (:func:`lumenairy.elements.lenses_maslov._fold_airy_eval`) as the basis.
  4. The SAME CFU kernel is then evaluated at ``zeta < 0`` (analytic
     continuation through the caustic) to fill the dark-side pixels with the
     exponential Airy tail -- reusing ``_fold_airy_eval`` verbatim for both
     sides (no divergent reimplementation).

A rotationally-symmetric SINGLE **CUSP** ring (``n_turn == 2`` -- a 3-branch
coalescence at finite radius) is ALSO completed, via the :func:`pearcey`
generalisation (niche R2 / A1): the local ray geometry is mapped to the Pearcey
normal form ``t^4 + x t^2 + y t`` and the cusp-finite Pearcey field replaces the
multibranch in the cusp zone (see the CUSP section below).  Everything else
falls back.

Scope / fallbacks (documented, never inf/nan):
  * A rotationally-symmetric SINGLE fold RING (``n_turn == 1``, the
    caustic_fold_ref class) is completed by the fold-Airy path; a single finite-
    radius CUSP ring (``n_turn == 2``) by the Pearcey path.  A decentered/tilted
    prescription, a non-rotationally-symmetric input, a carrier tilt, a
    near-axial (Bessoid) cusp, three-or-more coalescing rings, or no clean fold
    -> the module DETECTS the case and falls back to the plain multibranch field
    with a one-time warning.
  * If the bright-side fit residual is large (the uniform fold model does not
    describe the field), or the cusp control-solve / linear-map residual is too
    large (not a clean single cusp), it falls back too.

Author: Andrew Traverso
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np

from .. import raytrace as rt
from ._lens_traced_multibranch import apply_real_lens_traced_multibranch
from .lenses_maslov import _fold_airy_eval, pearcey

__all__ = ['apply_real_lens_traced_uniform']

# Airy argument cap for the dark-side fill: ``Ai(50) ~ 1e-73`` (numerically
# zero) so clamping ``-k^{2/3} zeta`` at this bound leaves the physical tail
# untouched while preventing any overflow far out on the grid.
_AIRY_ARG_CAP = 50.0

# Pearcey series (:func:`pearcey`) converges everywhere but SLOWS / overflows
# for large ``|x|, |y|``; clamp the control coordinates to this box when
# building the CUSP field (far outside the box geometric optics is exact, so
# the clamped value only affects a low-amplitude tail).  ``x`` is the
# defocus-like axis (the cusp opens toward ``x < 0``), ``y`` the transverse one.
_PEARCEY_X_LO, _PEARCEY_X_HI = -12.0, 6.0
_PEARCEY_Y_CAP = 11.0


# ==========================================================================
# Catastrophe classification (niche R5 / roadmap A4).
#
# The number of INTERIOR turning points of the meridional map ``x_out(h)`` (the
# fold caustics crossed along a radial cut) names the catastrophe class in the
# Thom hierarchy:  1 -> FOLD (A_2, the Airy path), 2 -> CUSP (A_3, the Pearcey
# path), 3 -> SWALLOWTAIL (A_4), 4 -> BUTTERFLY (A_5), >=5 -> a higher /
# non-classifiable coalescence.  Only the fold and cusp have canonical
# completions here; the higher catastrophes are RARE in real lenses and their
# canonical integrals are high-effort, so A4 makes the dispatcher DETECT the
# class and ROUTE cleanly to the finite multibranch / GBD fallback with a
# one-time NAMED warning -- keeping 'seamless' true by routing, never by
# emitting inf/nan.
# ==========================================================================
_CATASTROPHE_NAMES = {
    1: 'fold (A2)',
    2: 'cusp (A3)',
    3: 'swallowtail (A4)',
    4: 'butterfly (A5)',
}


def _count_interior_turning_points(x_out):
    """Number of interior turning points (sign changes of ``d x_out / d h``) of a
    monotone-``h`` meridional map ``x_out(h)`` -- each is a fold caustic, and the
    count names the catastrophe class (see :data:`_CATASTROPHE_NAMES`).

    Same intent as the ``diff(sign(diff(x_out)))`` construction in
    :func:`_trace_meridional_fold` / :func:`_trace_meridional_cusp`, but robust
    to an isolated zero-slope SAMPLE at an extremum (which would otherwise
    double-count a ``+ -> 0 -> -`` transition): the zero-sign entries are dropped
    before counting the residual sign changes.  On clean ray-traced maps (no
    exactly-flat slope sample) this equals the live traces' ``turns.size``."""
    xo = np.asarray(x_out, dtype=float)
    if xo.size < 3:
        return 0
    s = np.sign(np.diff(xo))
    s = s[s != 0]
    if s.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(s) != 0))


def _classify_catastrophe(n_turn):
    """Name the catastrophe class from the interior-turning-point count
    ``n_turn``.  ``1`` -> fold, ``2`` -> cusp (both have canonical completions),
    ``3`` -> swallowtail, ``4`` -> butterfly, ``>=5`` -> a non-classifiable
    higher catastrophe.  The swallowtail/butterfly/non-classifiable classes are
    ROUTED (never analytically evaluated) to the finite fallback."""
    name = _CATASTROPHE_NAMES.get(int(n_turn))
    if name is not None:
        return name
    if int(n_turn) >= 5:
        return f'non-classifiable higher catastrophe ({int(n_turn)} folds)'
    return f'degenerate map ({int(n_turn)} folds)'


def _is_rotationally_symmetric(E_in, dx, *, tol=0.12, n_rings=6, n_theta=48):
    """True when ``|E_in|`` is (approximately) rotationally symmetric about the
    grid centre -- the necessary condition for the radial fold-ring completion.

    For ``n_rings`` radii spanning the significant-power support (kept inside
    the inscribed circle so every azimuth is on-grid), ``|E_in|`` is bilinearly
    resampled at ``n_theta`` AZIMUTHS and the per-ring azimuthal coefficient of
    variation (std/mean over theta, at FIXED radius -- so a steep radial profile
    does not trip it) must stay below ``tol``.  Cheap and grid-robust; a false
    negative only steers the caller to the (still-finite) multibranch
    fallback."""
    E_in = np.asarray(E_in)
    N = E_in.shape[0]
    a = np.abs(E_in)
    amax = float(a.max())
    if amax <= 0.0:
        return False
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X * X + Y * Y)
    sig = a > 0.05 * amax
    if not sig.any():
        return False
    # cap the support radius at the inscribed circle so every sampled ring is
    # fully on-grid (avoids partial-annulus artefacts at the corners)
    r_max_grid = 0.49 * N * dx
    r_sup = float(min(r[sig].max(), r_max_grid))
    if r_sup <= 2.0 * dx:
        return False
    c = 0.5 * N                          # grid-centre index (pixel-centre conv.)
    th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    for rr in np.linspace(0.15 * r_sup, r_sup, n_rings):
        # sample |E| at (rr, theta); grid index = centre + coord/dx
        gi = c + (rr * ct) / dx          # column (x) index
        gj = c + (rr * st) / dx          # row (y) index
        i0 = np.floor(gi).astype(int)
        j0 = np.floor(gj).astype(int)
        if i0.min() < 0 or j0.min() < 0 or i0.max() + 1 >= N or j0.max() + 1 >= N:
            continue
        fi = gi - i0
        fj = gj - j0
        vals = ((1 - fi) * (1 - fj) * a[j0, i0]
                + fi * (1 - fj) * a[j0, i0 + 1]
                + (1 - fi) * fj * a[j0 + 1, i0]
                + fi * fj * a[j0 + 1, i0 + 1])
        mu = float(vals.mean())
        if mu <= 0.02 * amax:
            continue
        if float(vals.std()) / mu > tol:
            return False
    return True


def _trace_meridional_fold(prescription, wavelength, output_plane_distance,
                           output_plane_n, launch_radius, n_fan):
    """Meridional ray trace of ``prescription`` to the output plane; extract the
    single fold-ring geometry.

    Launches a dense on-axis-collimated meridional fan along ``+x`` (``y = 0``)
    through the SAME surfaces the multibranch uses, advances the exit rays a
    distance ``output_plane_distance`` past the exit vertex (index
    ``output_plane_n``), and returns::

        {ok, reason, r_c, kappa, cphi, n_turn}

    ``ok`` is True only for exactly ONE interior turning point of ``y_obs(h)``
    (a single fold ring).  ``zeta(r) = kappa (r_c - r)`` (linear fit of
    ``[3/4 |S_A - S_B|]^{2/3}`` over the two-branch band) and
    ``A(r) = polyval(cphi, r - r_c)`` (quadratic mean-eikonal fit).  ``reason``
    names the fallback cause when ``ok`` is False."""
    fail = dict(ok=False, r_c=0.0, kappa=0.0, cphi=None, n_turn=0)
    surfaces = rt.surfaces_from_prescription(prescription)
    xs = np.linspace(launch_radius / n_fan, launch_radius, n_fan)
    rays = rt.RayBundle(
        x=xs.copy(), y=np.zeros(n_fan), z=np.zeros(n_fan),
        L=np.zeros(n_fan), M=np.zeros(n_fan), N=np.ones(n_fan),
        wavelength=wavelength, alive=np.ones(n_fan, dtype=bool),
        opd=np.zeros(n_fan))
    tr = rt.trace(rays, surfaces, wavelength)
    ex = tr.image_rays
    Nz = np.where(np.abs(ex.N) > 1e-30, ex.N, 1e-30)
    t = output_plane_distance / Nz
    x_out = ex.x + t * ex.L
    opl = ex.opd + float(output_plane_n) * t
    alive = ex.alive & np.isfinite(x_out) & np.isfinite(opl)
    h = xs[alive]
    xo = x_out[alive]
    S = opl[alive]
    if h.size < 64:
        return {**fail, 'reason': 'too_few_rays'}

    # interior turning points of x_out(h) (fold caustics = dx_out/dh sign change)
    dxo = np.diff(xo)
    sgn = np.sign(dxo)
    turns = np.where(np.diff(sgn) != 0)[0] + 1     # index into h of the turn
    n_turn = int(turns.size)
    if n_turn == 0:
        return {**fail, 'reason': 'no_fold', 'n_turn': 0}
    if n_turn > 1:
        # >1 interior turning point -> cusp / multiple rings (Pearcey regime,
        # out of scope): detect + fall back.
        return {**fail, 'reason': 'cusp_or_multiple', 'n_turn': n_turn}

    i_f = int(turns[0])
    r_c = float(abs(xo[i_f]))
    if r_c <= 0.0:
        return {**fail, 'reason': 'degenerate_rc', 'n_turn': n_turn}

    # two branches around the fold; |x_out| is monotonic on each
    ya = np.abs(xo[:i_f + 1])
    Sa = S[:i_f + 1]
    yb = np.abs(xo[i_f:])
    Sb = S[i_f:]
    from scipy.interpolate import CubicSpline

    def _mono_spline(yv, sv):
        yu, idx = np.unique(yv, return_index=True)
        if yu.size < 4:
            return None
        return CubicSpline(yu, sv[idx])
    spa = _mono_spline(ya, Sa)
    spb = _mono_spline(yb, Sb)
    if spa is None or spb is None:
        return {**fail, 'reason': 'branch_undersampled', 'n_turn': n_turn}

    r_lo = float(max(ya.min(), yb.min()))
    band = r_c - r_lo
    if band <= 0.0:
        return {**fail, 'reason': 'no_two_branch_band', 'n_turn': n_turn}
    rb = np.linspace(r_lo + 0.02 * band, r_c - 0.02 * band, 256)
    dS = np.abs(spa(rb) - spb(rb))
    zeta = (0.75 * dS) ** (2.0 / 3.0)
    # kappa = slope of the linear zeta(r_c - r); intercept must be ~0 (fold)
    u = r_c - rb
    kfit = np.polyfit(u, zeta, 1)
    kappa = float(kfit[0])
    if not np.isfinite(kappa) or kappa <= 0.0:
        return {**fail, 'reason': 'bad_kappa', 'n_turn': n_turn}
    zpred = np.polyval(kfit, u)
    resid = float(np.max(np.abs(zpred - zeta)) / (np.max(zeta) + 1e-300))
    if resid > 0.15:
        return {**fail, 'reason': 'zeta_nonlinear', 'n_turn': n_turn}
    cphi = np.polyfit(rb - r_c, 0.5 * (spa(rb) + spb(rb)), 2)
    return dict(ok=True, reason='fold_ring', r_c=r_c, kappa=kappa,
                cphi=cphi, n_turn=n_turn)


# ==========================================================================
# CUSP (Pearcey) completion -- A1 / niche R2.
#
# A CUSP is a 3-branch coalescence: the meridional map ``x_out(h)`` has TWO
# interior turning points (``n_turn == 2``), so a band of radii ``r in
# (r1, r2)`` is reached by THREE ray branches whose two folds (at ``r1``, ``r2``)
# sit within a diffraction width of each other -- the fold-Airy of the fold
# completion cannot resolve the pair and the correct local diffraction pattern
# is the PEARCEY (cusp) canonical integral
# ``P(x, y) = int exp(i(t^4 + x t^2 + y t)) dt`` (:func:`pearcey`, REUSED here --
# no reimplementation).  This module maps the local ray geometry to the Pearcey
# normal form, evaluates ``P`` (and its two first derivatives) on the mapped
# control coordinates, and combines them with amplitude coefficients derived
# from the three branches' stationary-phase ray amplitudes -- the exact cusp
# analog of :func:`lumenairy.elements.lenses_maslov.uniform_fold_airy`.
#
# THE MAPPING (validated to machine precision against 1-D quartic-phase
# integrals, and to Icorr ~ 0.97 / windowed r2m ~ few-% against a direct
# Rayleigh-Sommerfeld cusp ground truth):
#
#   * Control coordinates ``(x, y)`` from the three branch PHASES.  The three
#     stationary points ``w_j`` of ``t^4 + x t^2 + y t`` are the roots of
#     ``4 w^3 + 2 x w + y = 0`` and their critical VALUES are
#     ``g(w_j) = (1/2) x w_j^2 + (3/4) y w_j`` with ``sum_j g(w_j) = -(1/2) x^2``.
#     Given the three branch phases ``Phi_j`` (radians = ``k * OPL_j``), solve
#     the 2-DOF system matching the sorted, mean-removed ``g(w_j)`` to the
#     sorted, mean-removed ``Phi_j`` for ``(x, y)`` (a small least squares).  The
#     scale is carried in radians so the Pearcey ``k``-scaling is implicit; the
#     reference phase is ``Phi0 = mean(Phi_j) + x^2 / 6``.
#   * At a FIXED observation plane the map is (to leading order) LINEAR: ``x`` is
#     constant (the plane's distance from the cusp point) and ``y`` is linear in
#     the transverse radius -- so ``x = x0``, ``y = gamma (r - r_sym)``, robustly
#     extrapolatable into the 1-branch regions where the phase solve has no
#     three real branches.
#   * Amplitude coefficients ``(b0, b1, b2)`` of ``(P, dP/dy, dP/dx)`` from the
#     branch ray-tube amplitudes: matching the stationary-phase limit of the
#     uniform form to the geometric ray sum gives, per branch,
#     ``b0 + b1 w_j + b2 w_j^2 = amp_j sqrt(|g''(w_j)| / 2pi)
#        exp(i (sgn(Psi''_j) - sgn(g''(w_j))) pi/4)`` with ``g''(w) = 12 w^2 +
#     2 x`` -- a 3x3 Vandermonde solve (the cusp analog of the fold's ``a0, a1``).
#   * The completed field is ``E = e^{i Phi0} (b0 P - i b1 dP/dy - i b2 dP/dx)``,
#     finite through the caustic (``P`` is entire), with the correct cusp fringe
#     structure and the exponential dark-side tail built in.
#
# SCOPE: rotationally-symmetric SINGLE cusp ring (``n_turn == 2``), the same
# collimated / rot-sym / centred gate as the fold path.  Anything the mapping
# cannot resolve (bad control solve, non-linear map, under-resolved fringe)
# DETECTS + falls back to the plain multibranch field (finite, never inf/nan).
# ==========================================================================


def _pearcey_cubic_roots(x, y):
    """The three real roots of ``4 w^3 + 2 x w + y = 0`` (the stationary points
    of the Pearcey phase ``t^4 + x t^2 + y t``), sorted ascending, or ``None``
    when fewer than three real roots exist (outside the cusp -- 1 branch)."""
    # depressed cubic w^3 + p w + q = 0 with p = x/2, q = y/4
    p = 0.5 * x
    q = 0.25 * y
    if p >= 0.0:
        return None
    disc = (0.5 * q) ** 2 + (p / 3.0) ** 3
    if disc > 0.0:
        return None
    m = 2.0 * np.sqrt(-p / 3.0)
    arg = np.clip(3.0 * q / (p * m), -1.0, 1.0)
    th = np.arccos(arg) / 3.0
    roots = np.array([m * np.cos(th - 2.0 * np.pi * kk / 3.0) for kk in range(3)])
    return np.sort(roots)


def _pearcey_crit_values(x, y):
    """Critical values ``g(w_j) = (1/2) x w_j^2 + (3/4) y w_j`` at the three real
    stationary points, or ``None`` when there are not three real roots."""
    w = _pearcey_cubic_roots(x, y)
    if w is None:
        return None
    return 0.5 * x * w ** 2 + 0.75 * y * w


def _solve_pearcey_control(phis, guess=(-2.0, 0.3)):
    """Solve the Pearcey control coordinates ``(x, |y|)`` from three branch
    phases (radians).  Matches the sorted, mean-removed critical values of the
    normal form to the sorted, mean-removed phases.  ``y`` is returned as its
    magnitude (the sorted-phase data cannot distinguish ``y`` from ``-y`` -- the
    sign is fixed downstream by the transverse position).  Returns
    ``(x, abs_y, cost)``; ``cost`` (the residual) flags a poor / non-cusp
    match."""
    from scipy.optimize import least_squares
    d = np.sort(phis)
    d = d - d.mean()

    def resid(v):
        cv = _pearcey_crit_values(v[0], v[1])
        if cv is None:
            return [1e3, 1e3, 1e3]
        c = np.sort(cv)
        return (c - c.mean()) - d

    sol = least_squares(resid, np.asarray(guess, dtype=float),
                        method='lm', xtol=1e-13, ftol=1e-13)
    return float(sol.x[0]), abs(float(sol.x[1])), float(sol.cost)


def _pearcey_cusp_amp_coeffs(x, y, phis, amps, dxdh):
    """The three amplitude coefficients ``(b0, b1, b2)`` of ``(P, dP/dy, dP/dx)``
    from the three branches' ray-tube amplitudes -- the cusp analog of the
    fold's ``(a0, a1)``.

    ``phis, amps, dxdh`` are the branch phases (radians), ray-tube amplitude
    magnitudes and signed ``d x_out / d h`` (its SIGN is the stationary-phase
    Maslov indicator).  Pairs each branch (ordered by phase) with a root
    (ordered by critical value), then solves the 3x3 Vandermonde
    ``[1, w_j, w_j^2] . (b0, b1, b2) = rhs_j`` with
    ``rhs_j = amp_j sqrt(|g''(w_j)| / 2pi) exp(i (sgn(Psi''_j) - sgn(g''_j)) pi/4)``,
    ``g''(w) = 12 w^2 + 2 x``.  Returns the complex ``(b0, b1, b2)`` or ``None``
    (degenerate)."""
    w = _pearcey_cubic_roots(x, y)
    cv = _pearcey_crit_values(x, y)
    if w is None or cv is None:
        return None
    ob = np.argsort(phis)          # branch order by phase
    oc = np.argsort(cv)            # root order by critical value
    wp = w[oc]                     # roots in critical-value order
    amp_p = np.asarray(amps)[ob]   # branch amps in phase order
    dxdh_p = np.asarray(dxdh)[ob]
    gpp = 12.0 * wp ** 2 + 2.0 * x
    if np.any(np.abs(gpp) < 1e-12):
        return None
    rhs = (amp_p * np.sqrt(np.abs(gpp) / (2.0 * np.pi))
           * np.exp(1j * (np.sign(dxdh_p) - np.sign(gpp)) * np.pi / 4.0))
    vander = np.stack([np.ones(3), wp, wp ** 2], axis=1)
    try:
        b = np.linalg.solve(vander, rhs)
    except np.linalg.LinAlgError:
        return None
    return b


def _pearcey_basis(x, y, step=3e-3):
    """``(P, dP/dx, dP/dy)`` at ``(x, y)`` -- REUSING :func:`pearcey` (central
    finite differences for the two derivatives; a reimplementation of the kernel
    would be a defect).  The control coordinates are clamped to the Pearcey
    series' well-behaved box so a far-tail pixel never overflows."""
    xe = float(np.clip(x, _PEARCEY_X_LO, _PEARCEY_X_HI))
    ye = float(np.clip(y, -_PEARCEY_Y_CAP, _PEARCEY_Y_CAP))
    p0 = pearcey(xe, ye)
    px = (pearcey(xe + step, ye) - pearcey(xe - step, ye)) / (2.0 * step)
    py = (pearcey(xe, ye + step) - pearcey(xe, ye - step)) / (2.0 * step)
    return p0, px, py


def _cusp_geometry_from_branches(r_arr, phib, ampb, dxdhb, *,
                                 max_control_cost=5e-4, max_map_resid=0.25):
    """Reduce the three-branch meridional geometry over the cusp band to the
    LINEAR Pearcey control map + smooth amplitude / phase coefficients.

    Parameters (all with the branches in a CONSISTENT per-radius order, e.g.
    by launch height):

    * ``r_arr`` -- ``(M,)`` radii spanning the three-branch band ``(r1, r2)``.
    * ``phib``  -- ``(M, 3)`` branch phases (radians = ``k * OPL``).
    * ``ampb``  -- ``(M, 3)`` branch ray-tube amplitude magnitudes.
    * ``dxdhb`` -- ``(M, 3)`` branch signed ``d x_out / d h``.

    Returns a dict ``{ok, x0, gamma, r_sym, cP, cb, ...}`` (the resolved linear
    map ``x=x0``, ``y=gamma (r - r_sym)``, the quadratic mean-phase poly ``cP``
    and the three linear complex amplitude polys ``cb``), or ``{ok: False,
    reason}`` when the control solve or the linear map is too poor to trust."""
    r_arr = np.asarray(r_arr, dtype=float)
    phib = np.asarray(phib, dtype=float)
    ampb = np.asarray(ampb, dtype=float)
    dxdhb = np.asarray(dxdhb, dtype=float)
    M = r_arr.shape[0]
    x_l, y_l, phi0_l, b_l, cost_l = [], [], [], [], []
    guess = (-2.0, 0.3)
    for i in range(M):
        ph = phib[i]
        x, ay, cost = _solve_pearcey_control(ph, guess=guess)
        guess = (x, ay if ay > 1e-9 else 0.3)
        b = _pearcey_cusp_amp_coeffs(x, ay, ph, ampb[i], dxdhb[i])
        if b is None:
            continue
        x_l.append(x)
        y_l.append(ay)
        phi0_l.append(ph.mean() + x ** 2 / 6.0)
        b_l.append(b)
        cost_l.append(cost)
    if len(x_l) < 8:
        return {'ok': False, 'reason': 'cusp band undersampled'}
    x_a = np.array(x_l)
    y_abs = np.array(y_l)
    phi0_a = np.array(phi0_l)
    b_a = np.array(b_l)
    cost_a = np.array(cost_l)
    # control-solve quality: the phases must actually reduce to the cusp normal
    # form (a large residual => this is not a clean single cusp).
    phi_scale = float(np.median(np.abs(np.diff(np.sort(phib[M // 2]))))) + 1e-30
    if float(np.median(cost_a)) > max_control_cost * phi_scale ** 2 + 1e-12 \
            and float(np.median(np.sqrt(cost_a))) > max_control_cost * phi_scale:
        return {'ok': False, 'reason': 'cusp control-solve residual too large'}
    # symmetric centre = min |y|; linear map y = gamma (r - r_sym), x = const
    ic = int(np.argmin(y_abs))
    r_sym = float(r_arr[ic])
    u = r_arr - r_sym
    y_signed = y_abs * np.sign(u + 1e-30)
    x0 = float(np.median(x_a))
    gam = float(np.linalg.lstsq(u[:, None], y_signed, rcond=None)[0][0])
    if not np.isfinite(gam) or abs(gam) < 1e-30:
        return {'ok': False, 'reason': 'degenerate cusp transverse map'}
    # linear-map fidelity: solved y vs the linear model (relative to the span)
    y_model = gam * u
    map_resid = float(np.max(np.abs(y_model - y_signed))
                      / (np.max(np.abs(y_signed)) + 1e-30))
    if not np.isfinite(map_resid) or map_resid > max_map_resid:
        return {'ok': False, 'reason': 'cusp map not linear (astigmatic/tilted)'}
    cP = np.polyfit(u, phi0_a, 2)
    cb = [np.polyfit(u, b_a[:, j], 1) for j in range(3)]
    return {'ok': True, 'reason': 'cusp_ring', 'x0': x0, 'gamma': gam,
            'r_sym': r_sym, 'cP': cP, 'cb': cb, 'x_var': float(np.std(x_a)),
            'y_max': float(np.max(y_abs)), 'map_resid': map_resid,
            'n_band': int(M), 'u_lo': float(u.min()), 'u_hi': float(u.max())}


def _radial_amp_sampler(E_in, dx):
    """A callable ``r -> |E_in|(r)`` from the rotationally-symmetric input field,
    sampling ``|E_in|`` along the ``+x`` grid axis (pixel-centre convention)."""
    a = np.abs(np.asarray(E_in))
    N = a.shape[0]
    c = N // 2
    rp = np.arange(N - c) * dx
    ap = a[c, c:].astype(float)

    def sampler(r):
        return np.interp(np.abs(r), rp, ap, left=ap[0], right=0.0)
    return sampler


def _roots_in_segments(h, f):
    """All ``h`` where the sampled ``f(h)`` crosses zero, refined by bisection on
    the bracketing grid cells.  ``h`` monotone ascending."""
    s = np.sign(f)
    idx = np.where(np.diff(s) != 0)[0]
    out = []
    for i in idx:
        a, b = h[i], h[i + 1]
        fa = f[i]
        for _ in range(60):
            m = 0.5 * (a + b)
            fm = np.interp(m, h, f)
            if fa * fm <= 0.0:
                b = m
            else:
                a, fa = m, fm
        out.append(0.5 * (a + b))
    return np.array(out)


def _trace_meridional_cusp(prescription, wavelength, output_plane_distance,
                           output_plane_n, launch_radius, n_fan, E_in, dx,
                           n_band=120):
    """Meridional trace + Pearcey-cusp geometry for a ``n_turn == 2`` (single
    finite-radius CUSP ring) prescription.

    Traces the same collimated meridional fan as :func:`_trace_meridional_fold`,
    requires EXACTLY two interior turning points of the signed ``x_out(h)`` whose
    extrema have the SAME sign (a finite-radius ring, not the on-axis /
    near-axial focus -- that is the Bessoid regime, out of scope), samples the
    three-branch band, and reduces it to the linear Pearcey control map +
    amplitude coefficients via :func:`_cusp_geometry_from_branches`.

    Returns ``{ok, r1, r2, ...geometry...}`` or ``{ok: False, reason}``."""
    fail = dict(ok=False, r1=0.0, r2=0.0)
    surfaces = rt.surfaces_from_prescription(prescription)
    xs = np.linspace(launch_radius / n_fan, launch_radius, n_fan)
    rays = rt.RayBundle(
        x=xs.copy(), y=np.zeros(n_fan), z=np.zeros(n_fan),
        L=np.zeros(n_fan), M=np.zeros(n_fan), N=np.ones(n_fan),
        wavelength=wavelength, alive=np.ones(n_fan, dtype=bool),
        opd=np.zeros(n_fan))
    tr = rt.trace(rays, surfaces, wavelength)
    ex = tr.image_rays
    Nz = np.where(np.abs(ex.N) > 1e-30, ex.N, 1e-30)
    t = output_plane_distance / Nz
    x_out = ex.x + t * ex.L
    opl = ex.opd + float(output_plane_n) * t
    alive = ex.alive & np.isfinite(x_out) & np.isfinite(opl)
    h = xs[alive]
    xo = x_out[alive]
    S = opl[alive]
    if h.size < 128:
        return {**fail, 'reason': 'too_few_rays'}

    dxo = np.diff(xo)
    turns = np.where(np.diff(np.sign(dxo)) != 0)[0] + 1
    if turns.size != 2:
        return {**fail, 'reason': f'n_turn={turns.size} (not a single cusp)'}
    Xa, Xb = xo[int(turns[0])], xo[int(turns[1])]
    if Xa * Xb <= 0.0:
        # the band straddles the axis -> on-axis (Bessoid) cusp, out of scope
        return {**fail, 'reason': 'near-axial cusp (Bessoid, out of scope)'}
    band_lo, band_hi = (Xa, Xb) if Xa < Xb else (Xb, Xa)
    r1 = float(min(abs(Xa), abs(Xb)))
    r2 = float(max(abs(Xa), abs(Xb)))
    if (r2 - r1) <= 0.0 or r1 <= 0.0:
        return {**fail, 'reason': 'degenerate cusp band'}

    k0 = 2.0 * np.pi / wavelength
    amp_of = _radial_amp_sampler(E_in, dx)
    hc = 0.5 * (h[:-1] + h[1:])
    slope = dxo / np.diff(h)          # d x_out / d h (signed), on cell centres

    vs = np.linspace(band_lo + 0.02 * (band_hi - band_lo),
                     band_hi - 0.02 * (band_hi - band_lo), n_band)
    r_arr, phib, ampb, dxdhb = [], [], [], []
    for v in vs:
        hb = _roots_in_segments(h, xo - v)
        if hb.size != 3:
            continue
        rq = abs(v)
        ph = k0 * np.interp(hb, h, S)
        jj = np.interp(hb, hc, slope)
        amp = amp_of(hb) * np.sqrt(np.clip(hb / (rq * np.abs(jj) + 1e-300),
                                           0.0, None))
        r_arr.append(rq)
        phib.append(ph)
        ampb.append(amp)
        dxdhb.append(jj)
    if len(r_arr) < 12:
        return {**fail, 'reason': 'cusp band undersampled (branch tracing)'}
    r_arr = np.array(r_arr)
    order = np.argsort(r_arr)
    geom = _cusp_geometry_from_branches(
        r_arr[order], np.array(phib)[order], np.array(ampb)[order],
        np.array(dxdhb)[order])
    if not geom['ok']:
        return {**fail, 'reason': geom['reason']}
    geom = dict(geom)
    geom.update(r1=r1, r2=r2)
    return geom


def _build_pearcey_cusp_field(E_mb, geom, wavelength, dx):
    """Build the 2-D Pearcey-cusp completed field from the multibranch base
    ``E_mb`` and the resolved cusp geometry.

    The Pearcey envelope depends only on the RADIUS (the control map is
    ``x=x0``, ``y=gamma (r - r_sym)``), so it is evaluated on a dense 1-D radial
    LUT and interpolated onto the grid (fast; mirrors the fold path's radial
    evaluation).  The rapidly-oscillating reference phase ``exp(i Phi0(r))`` is
    applied per-pixel (no LUT aliasing).  The overall (diffraction-prefactor)
    scale is fixed by matching to ``E_mb`` in a clean single-branch annulus just
    inside ``r1``; the completed field REPLACES ``E_mb`` only in the cusp zone
    ``[r1 - margin, r2 + margin]`` (the multibranch stays elsewhere).  Returns
    the completed field (finite everywhere) or ``None`` (guarded degenerate)."""
    N = E_mb.shape[0]
    x0 = geom['x0']
    gam = geom['gamma']
    r_sym = geom['r_sym']
    cP = geom['cP']
    cb = geom['cb']
    r1, r2 = geom['r1'], geom['r2']
    band = r2 - r1
    margin = 2.5 * band
    # the amplitude / mean-phase polynomials are fitted ONLY over the three-
    # branch band; beyond it (the 1-branch regions) the transverse control ``y``
    # keeps growing linearly (physical -- the single-ray asymptotic) but the
    # amplitude coefficients must NOT be extrapolated past the band (a linear
    # fit would blow up).  Clamp ``u`` for the coefficient / phase evaluation to
    # the band range, so they hold their band-edge value in the tails.
    u_lo = geom.get('u_lo', -0.5 * band)
    u_hi = geom.get('u_hi', 0.5 * band)

    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    rg = np.sqrt(X * X + Y * Y)

    def envelope(r):
        u = r - r_sym
        uc = min(max(u, u_lo), u_hi)
        yq = gam * u
        p0, px, py = _pearcey_basis(x0, yq)
        return (np.polyval(cb[0], uc) * p0
                - 1j * np.polyval(cb[1], uc) * py
                - 1j * np.polyval(cb[2], uc) * px)

    # dense radial LUT of the slowly-varying complex envelope over the zone
    r_lo = max(0.0, r1 - margin)
    r_hi = r2 + margin
    n_lut = max(256, int(4.0 * (r_hi - r_lo) / dx))
    r_lut = np.linspace(r_lo, r_hi, n_lut)
    env_lut = np.array([envelope(rr) for rr in r_lut])
    if not np.all(np.isfinite(env_lut)):
        return None

    zone = (rg >= r_lo) & (rg <= r_hi)
    rz = rg[zone]
    env_z = (np.interp(rz, r_lut, env_lut.real)
             + 1j * np.interp(rz, r_lut, env_lut.imag))
    phi0_z = np.polyval(cP, rz - r_sym)
    F = env_z * np.exp(1j * phi0_z)

    # prefactor: match F to E_mb in a clean single-branch annulus inside r1
    ann_lo = max(0.0, r1 - 3.0 * band)
    ann_hi = max(ann_lo + dx, r1 - 0.5 * band)
    ann = (rz >= ann_lo) & (rz <= ann_hi)
    if int(ann.sum()) < 16:
        ann = (rz >= max(0.0, r1 - 6.0 * band)) & (rz <= r1 - 0.5 * band)
    denom = np.vdot(F[ann], F[ann])
    if int(ann.sum()) < 8 or abs(denom) <= 0.0:
        return None
    Emb_ann = np.asarray(E_mb)[zone][ann]
    pref = np.vdot(F[ann], Emb_ann) / denom
    if not np.isfinite(pref):
        return None

    E_out = np.array(E_mb, dtype=np.complex128, copy=True)
    E_out[zone] = pref * F
    if not np.all(np.isfinite(E_out)):
        return None
    return E_out


def apply_real_lens_traced_uniform(
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
    uniform_fit_halfwidth: Any = None,
    n_fan: int = 4000,
    return_diagnostics: bool = False,
) -> Any:
    """Uniform (Airy) dark-side completion of the multibranch ray-density field.

    Runs :func:`apply_real_lens_traced_multibranch` for the finite, ludwig-
    regularized BRIGHT-side field, then -- for a rotationally-symmetric SINGLE
    fold RING -- fills the DARK side (``r > r_c``) with the CFU uniform Airy tail
    (see the module docstring).  Falls back to the plain multibranch field (with
    a one-time warning) for any non-fold / cusp / non-symmetric case.  The
    output is FINITE everywhere (never inf/nan).

    Parameters mirror :func:`apply_real_lens_traced_multibranch`, plus
    ``uniform_fit_halfwidth`` (bright-band fit width [m]; ``None`` -> auto from
    the Airy scale) and ``n_fan`` (meridional-trace ray count).

    Returns the completed ``(N, N)`` complex field (plus a diagnostics dict when
    ``return_diagnostics`` -- with the resolved ``r_c``, ``kappa``, the fitted
    Airy coefficients, the fit residual, and ``fell_back`` / ``reason``)."""
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_traced_uniform')
    E_in = np.asarray(E_in)
    N = E_in.shape[0]
    if E_in.ndim != 2 or E_in.shape[0] != E_in.shape[1]:
        raise ValueError("apply_real_lens_traced_uniform: square 2D field "
                         f"required; got {E_in.shape}.")
    target_cdtype = E_in.dtype if np.iscomplexobj(E_in) else np.complex128

    # 1. bright-side multibranch (finite through the fold via ludwig)
    E_mb, mb_diag = apply_real_lens_traced_multibranch(
        E_in, prescription=prescription, wavelength=wavelength, dx=dx,
        output_plane_distance=output_plane_distance,
        output_plane_n=output_plane_n, ray_subsample=ray_subsample,
        min_area_ratio=min_area_ratio, caustic_band=caustic_band,
        input_carrier=input_carrier, return_diagnostics=True)
    E_mb = np.asarray(E_mb)

    def _fallback(reason):
        warnings.warn(
            "apply_real_lens_traced(caustic='uniform'): the uniform Airy dark-"
            f"side completion does not apply here ({reason}); returning the "
            "plain multibranch field (bright-side only, no dark tail).  The "
            "uniform completion covers a rotationally-symmetric SINGLE fold "
            "RING (collimated / rot-sym input, centred prescription, one "
            "interior caustic); a cusp needs the Pearcey generalisation and a "
            "decentered / astigmatic fold is out of scope -- use "
            "apply_real_lens_gbd / apply_real_lens_fga or single-branch "
            "ray_density + ASM for those.", RuntimeWarning, stacklevel=3)
        out = E_mb.astype(target_cdtype) if E_mb.dtype != target_cdtype else E_mb
        if return_diagnostics:
            d = dict(mb_diag)
            d.update(fell_back=True, reason=reason, r_c=None, kappa=None,
                     c0=None, c1=None, fit_residual=None)
            return out, d
        return out

    # 2. rotational-symmetry / on-axis gate
    from ._lens_traced import _prescription_has_field_frame
    kcx, kcy = mb_diag.get('input_carrier', (0.0, 0.0))
    if _prescription_has_field_frame(prescription):
        return _fallback('decentered/tilted prescription')
    if abs(kcx) > 0.0 or abs(kcy) > 0.0:
        return _fallback('carrier tilt (fold not a centred ring)')
    if not _is_rotationally_symmetric(E_in, dx):
        return _fallback('non-rotationally-symmetric input')

    # 3. meridional fold geometry (r_c, kappa, phi)
    aperture = prescription.get('aperture_diameter')
    if aperture is not None:
        launch_radius = 0.5 * float(aperture) * 0.98
    else:
        launch_radius = 0.5 * N * dx
    fold = _trace_meridional_fold(
        prescription, wavelength, float(output_plane_distance),
        float(output_plane_n), launch_radius, int(n_fan))
    if not fold['ok']:
        # ``n_turn == 2`` is a CUSP (3-branch coalescence); try the Pearcey
        # cusp completion (A1 / niche R2) before falling back to multibranch.
        if fold.get('reason') == 'cusp_or_multiple' and fold.get('n_turn') == 2:
            cusp = _trace_meridional_cusp(
                prescription, wavelength, float(output_plane_distance),
                float(output_plane_n), launch_radius, int(n_fan), E_in, dx)
            if cusp['ok']:
                E_cusp = _build_pearcey_cusp_field(E_mb, cusp, wavelength, dx)
                if E_cusp is not None and np.all(np.isfinite(E_cusp)):
                    E_out = E_cusp.astype(target_cdtype)
                    if return_diagnostics:
                        d = dict(mb_diag)
                        d.update(fell_back=False, reason='cusp_ring',
                                 r_c=None, kappa=None, c0=None, c1=None,
                                 fit_residual=None, cusp_r1=cusp['r1'],
                                 cusp_r2=cusp['r2'], cusp_x0=cusp['x0'],
                                 cusp_gamma=cusp['gamma'],
                                 cusp_map_resid=cusp['map_resid'])
                        return E_out, d
                    return E_out
                return _fallback('cusp: non-finite field (guarded)')
            return _fallback('cusp: ' + cusp['reason'])
        # A4 (niche R5): a map with THREE OR MORE interior turning points is a
        # HIGHER catastrophe (swallowtail / butterfly / non-classifiable) -- NOT
        # a clean fold or cusp.  There is no canonical completion here (rare in
        # real lenses, high effort); DETECT the class from the turning-point
        # structure and ROUTE cleanly to the finite multibranch/GBD fallback
        # with a one-time NAMED warning (never inf/nan).
        if fold.get('reason') == 'cusp_or_multiple' \
                and int(fold.get('n_turn') or 0) >= 3:
            n_turn = int(fold['n_turn'])
            cls = _classify_catastrophe(n_turn)
            return _fallback(
                f'{cls}: {n_turn} coalescing fold branches -- a higher '
                f'catastrophe with no canonical analytic completion; routing '
                f'to the finite multibranch field (use apply_real_lens_gbd / '
                f'apply_real_lens_fga for the diffraction-correct field here)')
        return _fallback(fold['reason'])
    r_c = fold['r_c']
    kappa = fold['kappa']
    cphi = fold['cphi']
    k0 = 2.0 * np.pi / wavelength

    # radial coordinate on the wave grid (library centre convention)
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    rgrid = np.sqrt(X * X + Y * Y)

    def _A(rq):
        return np.polyval(cphi, rq - r_c)

    def _zeta(rq):
        return kappa * (r_c - rq)

    # 4. fit the two smooth CFU coefficients (c0 -> Ai term, c1 -> Ai' term) to
    #    the BRIGHT multibranch field over a band just inside r_c, using the
    #    exact CFU kernel as the (linear) basis -- REUSE, no reimplementation.
    # The Airy boundary layer around the caustic has radial scale
    # ``l_airy = 1/(k0^{2/3} kappa)``.  The grid must resolve it (>~1.2 px) for
    # the fit to be meaningful; otherwise fall back (a coarse grid under-samples
    # the fold ring itself).
    l_airy = 1.0 / (k0 ** (2.0 / 3.0) * kappa)
    if l_airy < 1.2 * dx:
        return _fallback('fold Airy scale under-resolved by the grid')
    # Fit over a band ~1 l_airy wide JUST inside r_c: a wider band pulls in the
    # cone-dominated interior and over-fits the Ai' term -> a too-fat tail
    # (validated ~1 l_airy across N=512..1024, dx 1.5-3 um).
    W = float(uniform_fit_halfwidth) if uniform_fit_halfwidth is not None \
        else l_airy
    gap = 0.15 * l_airy
    if r_c - W <= gap:
        return _fallback('fit band collapses (fold too small vs grid)')
    bright = (rgrid >= r_c - W) & (rgrid <= r_c - gap)
    if int(bright.sum()) < 24:
        return _fallback('too few bright-band pixels for the fit')
    rb = rgrid[bright]
    Eb = E_mb[bright]
    # Basis = the EXACT CFU kernel (reused, no reimplementation), evaluated at
    # (a0, a1) = (1, 0) and (0, 1); linear in the coefficients, so fitting
    # (c0, c1) here and evaluating _fold_airy_eval(..., c0, c1) at zeta<0 below
    # continues the SAME uniform field analytically to the dark side.  The
    # per-pixel rows are weighted by 1/sqrt(r) so each RADIUS carries equal
    # weight (undoing the circumference ~r pixel-count bias, which otherwise
    # over-weights the pixels nearest r_c and over-fits the Ai' term); this is
    # a binning-free, grid-robust radially-uniform least squares.
    wt = 1.0 / np.sqrt(rb)
    basis0 = _fold_airy_eval(k0, _A(rb), _zeta(rb), 1.0, 0.0)
    basis1 = _fold_airy_eval(k0, _A(rb), _zeta(rb), 0.0, 1.0)
    G = np.stack([basis0, basis1], axis=1)
    Gw = G * wt[:, None]
    Ew = Eb * wt
    coef, *_ = np.linalg.lstsq(Gw, Ew, rcond=None)
    c0, c1 = complex(coef[0]), complex(coef[1])
    fit_resid = float(np.linalg.norm(Gw @ coef - Ew)
                      / (np.linalg.norm(Ew) + 1e-300))
    if not (np.isfinite(c0) and np.isfinite(c1)) or fit_resid > 0.5:
        return _fallback(f'uniform fit residual too large ({fit_resid:.2f})')

    # 5. fill the DARK side (r > r_c): analytic continuation to zeta < 0 -> the
    #    exponential Airy tail.  Clamp the Airy argument far out (no overflow;
    #    the physical tail -- arg <~ 15 -- is untouched).
    E_out = E_mb.astype(np.complex128, copy=True)
    dark = rgrid > r_c
    rd = rgrid[dark]
    zd = _zeta(rd)
    # -k0^{2/3} zeta = k0^{2/3} kappa (r - r_c) >= 0 on the dark side; cap it.
    zfloor = -_AIRY_ARG_CAP / (k0 ** (2.0 / 3.0))
    zd = np.maximum(zd, zfloor)
    E_out[dark] = _fold_airy_eval(k0, _A(rd), zd, c0, c1)
    if not np.all(np.isfinite(E_out)):
        return _fallback('non-finite dark-side fill (guarded)')
    E_out = E_out.astype(target_cdtype)

    if return_diagnostics:
        d = dict(mb_diag)
        d.update(fell_back=False, reason='fold_ring', r_c=r_c, kappa=kappa,
                 c0=c0, c1=c1, fit_residual=fit_resid, fit_halfwidth=W,
                 n_turn=fold['n_turn'])
        return E_out, d
    return E_out

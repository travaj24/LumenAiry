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

Scope / fallbacks (documented, never inf/nan):
  * Only a rotationally-symmetric SINGLE fold RING is completed (the
    caustic_fold_ref class).  A decentered/tilted prescription, a
    non-rotationally-symmetric input, a carrier tilt, a CUSP (3-branch
    coalescence / >1 interior turning point), or no clean fold -> the module
    DETECTS the case and falls back to the plain multibranch field with a
    one-time warning (a cusp would need the :func:`pearcey` generalisation,
    out of scope here).
  * If the bright-side fit residual is large (the uniform fold model does not
    describe the field), it falls back too.

Author: Andrew Traverso
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np

from .. import raytrace as rt
from ._lens_traced_multibranch import apply_real_lens_traced_multibranch
from .lenses_maslov import _fold_airy_eval

__all__ = ['apply_real_lens_traced_uniform']

# Airy argument cap for the dark-side fill: ``Ai(50) ~ 1e-73`` (numerically
# zero) so clamping ``-k^{2/3} zeta`` at this bound leaves the physical tail
# untouched while preventing any overflow far out on the grid.
_AIRY_ARG_CAP = 50.0


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

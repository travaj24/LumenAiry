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
import warnings
from functools import lru_cache
from typing import Any, Dict, Optional, Union

import numpy as np

from .. import raytrace as rt

# The three Chebyshev Vandermonde helpers live in
# ``lumenairy._math.chebyshev``.  The new public names are imported here
# and bound to the legacy underscore-prefixed locals so this module's
# ~10 call sites keep working unchanged.
from .._math.chebyshev import (
    chebyshev_derivative_vandermonde as _chebyshev_derivative_vandermonde,
    chebyshev_second_derivative_vandermonde as _chebyshev_second_derivative_vandermonde,
    chebyshev_vandermonde as _chebyshev_vandermonde,
)
from ..progress import call_progress

# Configuration objects (audit 2026-09-11 TESTS-ARCH section 14 item 13).
# ``lens_config`` is a LEAF -- it imports nothing from lumenairy at module
# scope -- so this edge is one-way and adds no import cost.
from .lens_config import (
    LensConfig,
    LensGeometry,
    LensNumerics,
    LensResources,
    _wants_config,
    resolve_entry_point_kwargs as _resolve_lens_config,
)

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

# S6 (audit): the input WAVEFRONT spread (3-sigma local-direction-cosine NA)
# above which the OPD-only saddle of the two asymptotic evaluators is no longer
# the stationary point of the full integrand -- so above it the driver fits the
# input's local wavevector into the saddle, and below it the OPD-only saddle
# ships unchanged.  1e-3 rad is ~1/1000 of the horizon: far below any real
# divergence, far above the 1e-10..1e-12 floor a numerically flat wave leaves
# in a phase-gradient estimate, so a genuinely collimated input never trips it.
# This bar is a FLATNESS DECLARATION, not a claim that the sub-bar error is
# negligible: the OPD-only saddle's launch direction is wrong by the input's
# own |v1_in|, so an input tilt theta misplaces the spot by ~f*theta whatever
# f is, and at the bar that is 2 um on the f = 6 mm fixture (one
# diffraction-limited spot there, 0.3 mm on an f = 1 m system).  MEASURED
# across it (f = 6 mm N-BK7 singlet, NA 0.05 chart, readout 0.6 mm past best
# focus, 'stationary_phase'), the relative L2 the fitted saddle moves the
# field by is 4.3e-02 at NA_wf = 3.0e-04, 1.46e-01 at the bar, 4.7e-01 at
# 3.0e-03 and O(1) beyond: linear in the tilt, with no natural knee.  What the
# bar really separates is a field that HAS a wavefront from one that is flat
# up to numerical noise (a numerically real field measures EXACTLY 0 here, and
# a field carrying float64 phase dirt measures ~1e-16), which matters because
# fitting k1 out of phase dirt would trip the residual gate below and warn
# about nothing.  A caller whose optic makes 3e-04 rad matter sets
# ``_S6_INPUT_WAVEVECTOR_SADDLE = True``.
_SADDLE_FLAT_INPUT_NA = 1e-3

# S6 fallback: the largest intensity-weighted RMS residual of the (k1x, k1y)
# fit, as a fraction of their own intensity-weighted RMS, at which the fitted
# local wavevector is still trusted to place the saddle.  0.5 is "the fit
# explains at least 75 % of the local wavevector's power".  Above it the
# driver keeps the OPD-only saddle and warns.
#
# MEASURED on the f = 6 mm N-BK7 singlet chart (order 4, 16^4 rays, readout
# 0.6 mm past best focus), as the residual and the field fidelity against a
# lumenairy-free conic-raytrace + Rayleigh-Sommerfeld oracle, OPD-only saddle
# -> fitted saddle:
#
#   residual 1.5e-12 (pure tilt) ......................  0.192 -> 0.907
#   residual 1.1e-05..1.7e-05 (converging / diverging) . 0.481 -> 0.991
#   residual 8.0e-05..5.7e-02 (10-20 waves of coma /
#            astigmatism / trefoil / spherical) ........ (all improve)
#   residual 7.2e-02..2.9e-01 (hard-edged aperture at
#            0.4..0.95 of the traced pupil) ............ 0.164 -> 0.812,
#                                            0.118 -> 0.279, 0.086 -> 0.190
#   residual 1.1e-01..9.1e-01 (speckle, 0.002..0.05 rad
#            rms phase noise on a tilt) ................ 0.192 -> 0.911,
#                                            0.192 -> 0.439, 0.191 -> 0.251
#   residual 9.6e-01 (0.1 rad rms speckle) ............. 0.191 -> 0.105 (WORSE)
#   residual 9.8e-01 (0.3 rad rms speckle) ............. 0.572 -> 0.000 (WORSE)
#   residual 9.9e-01 (uniform white-noise phase) ....... 0.014 -> 0.009 (WORSE)
#
# The bar is placed at 0.5 because engaging a poor fit is still better than
# the OPD-only saddle whenever the carrier is NOT flat: that saddle then puts
# the spot on the wrong ray entirely, so the comparison above has almost no
# turning point to find.  What the bar cannot do is say whether the ANSWER is
# trustworthy, and it is loose in that sense: it is the fit's VALUE residual,
# while the saddle also consumes the fit's two DERIVATIVES, whose error is not
# bounded by it.  MEASURED (VERIFY-B1, 2026-09-13) with this same chart's
# driver reporting the residual, and the field fidelity taken against the
# EXACT pointwise 'quadrature' on the same chart rather than against a
# geometric-optics oracle (which is not a truth for speckle), speckle on a
# tilted carrier:
#
#   speckle 0.002 / 0.005 rad rms -> residual 4.7e-03 / 1.2e-02, fidelity at
#            the collimated floor;
#   speckle 0.01 / 0.02 / 0.05 ... -> residual 2.3e-02 / 4.6e-02 / 1.1e-01,
#            fidelity 0.66 / 0.10 / 4e-04 of the floor's 0.91.
#
# So on a tilted carrier the answer is gone two decades below this bar and the
# bar never fires; the remedy the warning should have named is
# ``integration_method='quadrature'``.  The hard-edged-aperture and
# converging / diverging rows above reproduce exactly; only the speckle
# residuals differ (the values above are ~20x higher than this chart produces
# for the speckle amplitudes they name).  Changing the bar is a shipped
# behaviour decision and is left to the owner; the VERIFY-B1 report carries
# the full ladder.
_K1_FIT_RESIDUAL_MAX = 0.5

# S6 fallback, the chart half: the largest relative RMS residual of the
# ENTRANCE-COORDINATE fit ``s1(s2, v2)`` at which the fitted local wavevector
# is still trusted to place the saddle.  The S6 term is ``k1 . ds1/dv2``, so
# the chart has to carry BOTH factors; ``_K1_FIT_RESIDUAL_MAX`` above scores
# only the first, and a uniform tilt passes it perfectly (k1 is a constant)
# however badly the chart is fitted.
#
# MEASURED (VERIFY-B1, 2026-09-13) as this residual against the field fidelity
# of the two asymptotic evaluators, scored against an exact conic-raytrace +
# Kirchhoff oracle, over a tilt sweep that drives ``na_proxy`` (and so the
# pupil box the order-4 chart has to span) upwards -- on the f = 6 mm N-BK7 /
# 1.0 um chart and on an independent f = 13.3 mm N-SF11 / 1.55 um one:
#
#   s1 residual 5.5e-05 .. 1.8e-03 (tilt 0.5 .. 1.5 x the lens NA)
#            ... fidelity at the collimated floor on both charts
#   s1 residual 3.3e-03 (f = 13.3 mm, tilt 1.75x) ..... 0.92 -> 0.04 COLLAPSE
#   s1 residual 4.1e-03 (f = 6 mm, tilt 2x) ........... 0.93 -> 0.00 COLLAPSE
#   s1 residual 5.8e-03 (f = 13.3 mm, tilt 2x) ........ 0.92 -> 0.08
#   s1 residual 6.7e-03 (f = 6 mm, tilt 4x) ........... 0.93 -> 0.00
#
# and the same chart at ``poly_order=6`` (s1 residual 2.7e-04) returns 0.91.
# The exact 'quadrature' integrator is still 0.92..0.99 across that whole
# band, so what fails is the SADDLE riding on the fit, not the chart's use by
# the pointwise integrators.  The bar is the geometric mean of the bracket
# 1.8e-03 (last good) .. 3.3e-03 (first collapse), which also sits a factor
# 1.9 above the other chart's last good row and 1.6 below its first bad one.
# Above it the OPD-only saddle is kept and the warning names ``poly_order``
# and ``input_na``; ``input_wavevector_saddle=True`` overrides, as it does for
# the k1 bar.
_S1_FIT_RESIDUAL_MAX = 2.5e-3

# S6 A/B seam, in the style of ``_QUAD_FACTORIZE`` above.  ``None`` (default)
# is the decision described at ``_SADDLE_FLAT_INPUT_NA`` and
# ``_K1_FIT_RESIDUAL_MAX``; ``False`` always solves ``grad_v2 OPD = 0`` -- the
# OPD-only saddle, and its warning, for every input -- and ``True`` uses the
# fitted local wavevector whenever the input is not flat, faithful fit or not.
# It is how the regression tests hold the two saddles side by side on one
# build, and how a caller asks a whole process for the OPD-only answer on a
# non-collimated input; the changelog's Migration note says when that is what
# someone wants.  Process-global and private, so it sets the DEFAULT only: the
# ``input_wavevector_saddle=`` keyword takes precedence per call, and is the
# spelling to reach for unless you mean every call in the process.
_S6_INPUT_WAVEVECTOR_SADDLE = None

# poly_order='auto' (v5.21): raise the tensor-Chebyshev OPD-fit order until the
# held-out fit residual stops improving (plateau) or reaches a good-enough
# target, so a smooth optic uses a cheap low order and a strongly-aberrated one
# gets the order it needs -- no manual tuning, and no over-fit (order is scored
# on a held-out ray split, not the training residual which only ever decreases).
_MZ_POLY_AUTO_MIN = 3
_MZ_POLY_AUTO_MAX = 8
_MZ_POLY_AUTO_TARGET = 1e-4   # waves RMS -- below this, extra order is wasted
_MZ_POLY_AUTO_RTOL = 0.10     # stop once an order step improves residual < 10%

# Other shared helpers still live in lenses.py.
from ._lens_real import _normalise_stop_index
from .lenses import (
    NUMEXPR_AVAILABLE,
    _ensure_numexpr_loaded,
    _fit_normaliser,
    _multi_indices_total_degree,
    _warn_if_aperture_exceeds_grid,
)

# ---------------------------------------------------------------------------
# Van Vleck-Maslov normalisation of the mixed-representation integral.
#
# The semiclassical (Van Vleck-Morette) kernel in d = 2 transverse dimensions
# is
#
#     K = (k / (2 pi i)) * |det d2S/ds1 ds2|^(1/2) * exp(i k S)
#
# and the change of variable s1 -> v2 at fixed s2 contributes
# d^2 s1 = |det(ds1/dv2)| d^2 v2 while |det d2S/ds1 ds2|^(1/2) =
# |det(ds1/dv2)|^(-1/2), so the two combine to the SQUARE ROOT:
#
#     E(s2) = 1/(i lambda) INT E_in(s1(s2, v2)) |det(ds1/dv2)|^(1/2)
#                                 exp(2 pi i OPD_waves) d^2 v2
#
# The integrators hold the Jacobian in NORMALISED chart coordinates
# (det_J_norm = det(ds1/du_v2)); ds1/dv2 = ds1/du_v2 / (v2x_h v2y_h).
# ``_van_vleck_density`` is that conversion + square root in one place so the
# four NumPy integrators, the three CuPy twins and the JAX/asymptotic siblings
# cannot drift apart again, and ``_maslov_kernel_prefactor`` is the 1/(i*lambda)
# that closes the absolute scale.  Verified on a two-flat-surface free-space
# chart (ds1/dv2 = -z I exactly): E_maslov / E_ASM = 1.000 with these, versus
# i*lambda*z without them.
# ---------------------------------------------------------------------------

def _van_vleck_density(abs_det_J_norm, v2x_h, v2y_h):
    """``|det(ds1/dv2)|**0.5`` from the normalised-chart Jacobian modulus.

    ``abs_det_J_norm`` is ``|det(ds1/du_v2)|`` (already an absolute value, in
    whatever array namespace the caller uses -- NumPy, CuPy or JAX); the
    returned array is the Van Vleck amplitude density that multiplies the
    integrand, in physical ``d^2 v2`` measure.
    """
    return (abs_det_J_norm / (v2x_h * v2y_h)) ** 0.5


def _maslov_kernel_prefactor(wavelength: float) -> complex:
    """``k / (2 pi i) = 1 / (i lambda)`` -- the d = 2 Van Vleck prefactor.

    Applied once to the assembled field.  It is a constant, so it commutes
    with the coarse->fine upsample and the linear-phase re-application; it is
    applied just before the ``normalize_output`` step so that
    ``normalize_output='none'`` (which ``roi=`` runs are forced onto) returns
    a physically-scaled field.
    """
    return 1.0 / (1j * float(wavelength))


# ---------------------------------------------------------------------------
# local_quadrature window geometry (audit S2).
#
# The integrator evaluates  INT A(v2) exp(2 pi i OPD(v2)) d^2 v2  on a finite
# lattice around the per-pixel saddle.  Three things have to be right:
#
# 1. PRINCIPAL AXES.  The natural widths come from the EIGENVALUES of the
#    phase Hessian, so the lattice has to be laid out on its EIGENVECTORS.
#    Scaling the coordinate axes by eigenvalues instead
#    swaps the two widths whenever ``H44 > H33`` and ignores ``H34``
#    entirely -- measured relative error 9.1 on a chart with A = 4, B = 40
#    and 2.5-3.1 with a cross term.
#
# 2. A SMOOTH WINDOW.  A hard-truncated uniform sum of a chirp leaves the
#    Fresnel endpoint oscillation, an O(1/extent) error that does NOT shrink
#    with the sample count -- measured 40 % floor at the shipped default,
#    flat from n = 32 to n = 128.  A Gaussian taper removes the endpoint.
#
# 3. THE WINDOW DIVIDED BACK OUT.  The taper is not part of the integral, so
#    its effect is removed by dividing by what it does to the QUADRATIC MODEL
#    of the phase, evaluated on the SAME finite lattice:
#
#        corr(s) = INT exp(i s xi^2) dxi  /  SUM_k exp(i s xi_k^2) w(xi_k) dxi
#                = sqrt(pi) e^(i s pi/4)  /  SUM_k ...
#
#    with ``s = sign(lambda_j)`` (in the principal-axis coordinate scaled by
#    ``sigma_j = 1/sqrt(pi |lambda_j|)`` the model phase is exactly
#    ``s * xi^2``).  Because the correction is computed on the same lattice,
#    the scheme is EXACT for a quadratic chart with a constant amplitude at
#    any ``local_n_samples`` / ``local_window_sigma`` WHILE THE TAPERED
#    LATTICE FITS INSIDE THE FITTED CHART BOX, and for a real chart it is the
#    Gaussian-regularised saddle with the model divided out -- i.e. no worse
#    than ``stationary_phase``, plus whatever non-quadratic content the
#    lattice resolves.
#
#    The qualifier is load-bearing (VERIFY-A4, O-1): out-of-box samples are
#    dropped while the correction is still computed on the FULL lattice, so
#    the exactness goes with them -- measured 8.09e-02 at
#    ``window_sigma = 5`` and 8.17e-01 at the shipped defaults on a chart
#    whose small Hessian eigenvalue puts ``sigma2_norm`` at 1.785, against
#    1e-15 when nothing is dropped.  ``_warn_local_window_truncation`` says
#    so once, above a 1 % dropped fraction.
# ---------------------------------------------------------------------------

# Number of Gaussian standard deviations of taper that fit inside the sampled
# half-extent ``local_window_sigma``.  3.0 puts the window at exp(-4.5) at the
# lattice edge, which is what kills the endpoint oscillation; the residual
# truncation is divided out exactly by the model correction above.
_LOCAL_TAPER_SIGMAS = 3.0


@lru_cache(maxsize=32)
def _local_window_1d(n_samples: int, window_sigma: float):
    """The shared 1-D lattice, Gaussian taper and model corrections.

    Returns ``(xi, dxi, taper1d, corr_plus, corr_minus)`` where ``corr_pm``
    are the complex factors that turn the tapered discrete sum of
    ``exp(+-i xi^2)`` back into its exact infinite-range value
    ``sqrt(pi) exp(+-i pi/4)``.  Depends only on the two integrator knobs, so
    it is cached (and therefore bit-identical between pixels and calls).
    """
    xi = np.linspace(-float(window_sigma), float(window_sigma), int(n_samples))
    dxi = float(xi[1] - xi[0]) if len(xi) > 1 else 2.0 * float(window_sigma)
    s_taper = float(window_sigma) / _LOCAL_TAPER_SIGMAS
    taper1d = np.exp(-0.5 * (xi / s_taper) ** 2)
    m_plus = complex(np.sum(np.exp(1j * xi ** 2) * taper1d) * dxi)
    m_minus = complex(np.sum(np.exp(-1j * xi ** 2) * taper1d) * dxi)
    ex_plus = np.sqrt(np.pi) * np.exp(1j * np.pi / 4.0)
    # A pathological (n_samples, window_sigma) pair could in principle put the
    # model sum at the origin; fall back to the infinite-range analytic ratio
    # rather than dividing by ~0 (CONVENTIONS Section 9: no silent inf).
    floor = 1e-3 * np.sqrt(np.pi)
    corr_plus = (ex_plus / m_plus) if abs(m_plus) > floor else 1.0 + 0j
    corr_minus = (np.conj(ex_plus) / m_minus) if abs(m_minus) > floor else 1.0 + 0j
    return xi, dxi, taper1d, corr_plus, corr_minus


def _local_window_geometry(xp, H33, H34, H44, v2x_h, v2y_h,
                           n_samples, window_sigma):
    """Principal-axis widths, rotation and window correction per pixel.

    ``H33 / H34 / H44`` are the second derivatives of the OPD (waves) with
    respect to the NORMALISED chart coordinates; they are de-normalised to the
    physical direction-cosine frame here, exactly as
    :func:`_integrate_stationary_phase` does.

    Returns ``(sigma1_phys, sigma2_phys, cos_theta, sin_theta, window_corr)``:
    the two principal half-widths, the eigenvector rotation, and the complex
    per-pixel factor that divides the Gaussian taper back out.
    """
    _, _, _, corr_plus, corr_minus = _local_window_1d(
        int(n_samples), float(window_sigma))
    a = H33 / (v2x_h ** 2)
    b = H34 / (v2x_h * v2y_h)
    d = H44 / (v2y_h ** 2)
    tau = a + d
    det_h = a * d - b ** 2
    disc = xp.maximum(tau ** 2 / 4.0 - det_h, 0.0)
    sqrt_disc = xp.sqrt(disc)
    lam1 = tau / 2.0 + sqrt_disc
    lam2 = tau / 2.0 - sqrt_disc
    # Eigenvector angle of a real symmetric 2x2: tan(2 theta) = 2b / (a - d),
    # with lam1 (the larger eigenvalue) along (cos theta, sin theta).
    theta = 0.5 * xp.arctan2(2.0 * b, a - d)
    cos_t = xp.cos(theta)
    sin_t = xp.sin(theta)
    sigma1 = 1.0 / xp.sqrt(xp.maximum(xp.abs(lam1), 1e-30) * xp.pi)
    sigma2 = 1.0 / xp.sqrt(xp.maximum(xp.abs(lam2), 1e-30) * xp.pi)
    c1 = xp.where(lam1 >= 0.0, corr_plus, corr_minus)
    c2 = xp.where(lam2 >= 0.0, corr_plus, corr_minus)
    return sigma1, sigma2, cos_t, sin_t, c1 * c2


# Fraction of a pixel's tapered lattice that may fall outside the fitted
# chart before ``local_quadrature`` stops being exact on a quadratic chart.
# Derivation in :func:`_warn_local_window_truncation`.
_LOCAL_WINDOW_DROP_WARN_FRAC = 0.01


def _warn_local_window_truncation(in_chart, inbox_flat, n_samples,
                                  window_sigma) -> None:
    """One RuntimeWarning when the tapered lattice leaves the chart box.

    VERIFY-A4 (O-1).  ``_integrate_local_quadrature`` is exact on a quadratic
    chart because it divides its Gaussian taper back out of the QUADRATIC
    MODEL computed on the SAME finite lattice -- but the samples that land
    outside the fitted Chebyshev box are DROPPED (correctly: the recurrences
    are not accurate there) while the correction is still computed on the
    FULL lattice, so the two stop matching as soon as any sample is lost.

    MEASURED on a synthetic quadratic chart whose larger principal width is
    ``sigma2_norm = 0.2524`` (a = 37, d = 5, hx = 0.031, hy = 0.019),
    against the closed-form Fresnel value:

        window_sigma  lattice reach in u   relative error
        3.0           0.757  (inside)      6.60e-15
        3.7           0.934  (inside)      2.75e-15
        5.0           1.262  (OUTSIDE)     8.09e-02
        7.5           1.893  (OUTSIDE)     8.79e-02

    and **8.17e-01** at the SHIPPED defaults on a chart with a small Hessian
    eigenvalue (a = 5, b = 4.9, d = 5 -> ``sigma2_norm = 1.785``), where
    ``stationary_phase`` is exact to 1.4e-15.  That is bounded -- the pre-S2
    ``np.clip`` over-count reached 2.0e+03 -- but it was SILENT, which is the
    same class of defect audit Y4 closed for ``propagate_modal_asymptotic``.

    The 1 % trigger is where the dropped weight stops being round-off: with
    the taper at ``exp(-4.5)`` on the lattice edge, losing 1 % of the samples
    costs ~1e-3 of the summed weight -- three decades above the 1e-15 the
    scheme reaches when nothing is dropped, and three decades below the
    8e-02 measured at ``window_sigma = 5``.
    """
    live = np.asarray(inbox_flat, dtype=bool).ravel()
    if not np.any(live):
        return
    ok = np.asarray(in_chart, dtype=bool)[live]
    n_total = int(ok.size)
    if n_total <= 0:
        return
    n_drop = n_total - int(np.count_nonzero(ok))
    if n_drop <= 0:
        return
    frac = n_drop / n_total
    if frac < _LOCAL_WINDOW_DROP_WARN_FRAC:
        return
    n_px = int(np.count_nonzero(live))
    worst = float(1.0 - ok.reshape(n_px, -1).mean(axis=1).min())
    import warnings
    warnings.warn(
        f"apply_real_lens_maslov: integration_method='local_quadrature' "
        f"dropped {n_drop}/{n_total} window samples ({100.0 * frac:.1f} %; "
        f"worst pixel {100.0 * worst:.1f} %) because the tapered lattice "
        f"reaches OUTSIDE the fitted Chebyshev chart at "
        f"local_window_sigma={window_sigma:g}, "
        f"local_n_samples={int(n_samples):d}.  The scheme is exact on a "
        f"quadratic chart only while the lattice fits INSIDE the box -- it "
        f"divides the taper back out of a model evaluated on the FULL "
        f"lattice, so a truncated lattice leaves a residual (measured "
        f"relative error 8.1e-02 at 26 % dropped, and 8.2e-01 on a chart "
        f"whose small Hessian eigenvalue puts sigma2_norm at 1.8, against "
        f"1e-15 when nothing is dropped).  Reduce local_window_sigma, widen "
        f"the chart (larger aperture / input_na), or use "
        f"integration_method='stationary_phase', which is exact on a "
        f"quadratic chart at any window.",
        RuntimeWarning, stacklevel=3)


def _v2_oscillation_bound(mi, coef_opd) -> float:
    """Upper bound on the phase CYCLE COUNT of the integrand along v2.

    The integrand phase is ``2 pi * OPD(s2, v2)`` with OPD in waves, so the
    number of oscillations a uniform quadrature has to resolve along one v2
    axis is bounded by the TOTAL VARIATION of the v2-dependent part of OPD,
    not by its excursion.  For a Chebyshev term ``c * T_k(u)`` the excursion
    is ``|c|`` but the total variation on [-1, 1] is ``2 k |c|`` -- the
    polynomial sweeps the interval ``k`` times.  Hence the bound

        osc = sum_k |c_k| * max(k3, k4)

    (terms with ``k3 = k4 = 0`` are constant in v2 and contribute nothing).

    N2 / S9 (audit): all three consumers -- the ``'auto'`` integrator choice,
    the ``'auto'`` ``n_v2`` resolution and the under-resolution warning --
    used the plain coefficient sum, which under-counts by up to 2.5x at the
    orders ``poly_order='auto'`` can select (measured: order 8 on an
    f = 6 mm biconvex, 634.0 -> 1539.7; f = 2 mm / 0.3 mm aperture,
    136.2 -> 337.1).  Under-counting makes ``auto`` pick ``quadrature`` for
    a chart that then speckles, and silences the warning that would have
    said so.
    """
    tv = np.array([float(max(k[2], k[3])) for k in mi], dtype=np.float64)
    return float(np.sum(np.abs(np.asarray(coef_opd, dtype=np.float64)) * tv))

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
    from numba import njit as _nj, prange as _pr
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
            a1 = u1[i]
            a2 = u2[i]
            a3 = u3[i]
            a4 = u4[i]
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
            sopd = 0.0
            sxv = 0.0
            sx3 = 0.0
            sx4 = 0.0
            syv = 0.0
            sy3 = 0.0
            sy4 = 0.0
            for mm in range(M):
                t12 = Tu1[K1[mm]] * Tu2[K2[mm]]
                t3 = Tu3[K3[mm]]
                t4 = Tu4[K4[mm]]
                dt3 = dTu3[K3[mm]]
                dt4 = dTu4[K4[mm]]
                vv = t12 * t3 * t4
                v3 = t12 * dt3 * t4
                v4 = t12 * t3 * dt4
                sopd += cop[mm] * vv
                sxv += csx[mm] * vv
                sx3 += csx[mm] * v3
                sx4 += csx[mm] * v4
                syv += csy[mm] * vv
                sy3 += csy[mm] * v3
                sy4 += csy[mm] * v4
            opd_v[i] = sopd
            s1x_v[i] = sxv
            ds1x3[i] = sx3
            ds1x4[i] = sx4
            s1y_v[i] = syv
            ds1y3[i] = sy3
            ds1y4[i] = sy4
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


def _opd_vd9_numpy(cop, csx, csy, K1, K2, K3, K4, u1, u2, u3, u4, P):
    """Value + v2 first-derivatives of the SHARED 4-var Chebyshev basis for
    ALL THREE coefficient sets.  Returns
    ``(opd_v, dopd3, dopd4, s1x_v, ds1x3, ds1x4, s1y_v, ds1y3, ds1y4)``.
    The Levin integrator needs exactly this at every query set (phase
    gradient from opd, Jacobian + landing point from s1x/s1y) and never the
    second derivatives -- so one basis build replaces three ``_opd6`` calls
    that each also run the T'' recurrence."""
    T1 = _chebyshev_vandermonde(u1, P)
    T2 = _chebyshev_vandermonde(u2, P)
    T3 = _chebyshev_vandermonde(u3, P)
    T4 = _chebyshev_vandermonde(u4, P)
    dT3 = _chebyshev_derivative_vandermonde(u3, P)
    dT4 = _chebyshev_derivative_vandermonde(u4, P)
    T12 = T1[K1] * T2[K2]
    T3b = T3[K3]
    T4b = T4[K4]
    val = T12 * T3b * T4b
    d3 = T12 * dT3[K3] * T4b
    d4 = T12 * T3b * dT4[K4]
    return (np.sum(cop[:, None] * val, axis=0),
            np.sum(cop[:, None] * d3, axis=0),
            np.sum(cop[:, None] * d4, axis=0),
            np.sum(csx[:, None] * val, axis=0),
            np.sum(csx[:, None] * d3, axis=0),
            np.sum(csx[:, None] * d4, axis=0),
            np.sum(csy[:, None] * val, axis=0),
            np.sum(csy[:, None] * d3, axis=0),
            np.sum(csy[:, None] * d4, axis=0))


def _get_cheb4d_vd9_numba():
    """Numba twin of :func:`_opd_vd9_numpy` (value + 1st-deriv, three coef
    sets, shared basis, no T'' recurrence), or None if numba is
    unavailable."""
    if "cheb4d_vd9" in _MZ_KERNELS:
        return _MZ_KERNELS["cheb4d_vd9"]
    if not _mz_load_numba():
        _MZ_KERNELS["cheb4d_vd9"] = None
        return None

    @_mz_njit(cache=True, parallel=True, fastmath=True)
    def _cheb4d_vd9(cop, csx, csy, K1, K2, K3, K4, u1, u2, u3, u4, P):
        n = u1.shape[0]
        M = cop.shape[0]
        opd_v = np.zeros(n)
        dopd3 = np.zeros(n)
        dopd4 = np.zeros(n)
        s1x_v = np.zeros(n)
        ds1x3 = np.zeros(n)
        ds1x4 = np.zeros(n)
        s1y_v = np.zeros(n)
        ds1y3 = np.zeros(n)
        ds1y4 = np.zeros(n)
        for i in _mz_prange(n):
            a1 = u1[i]
            a2 = u2[i]
            a3 = u3[i]
            a4 = u4[i]
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
            sov = 0.0
            so3 = 0.0
            so4 = 0.0
            sxv = 0.0
            sx3 = 0.0
            sx4 = 0.0
            syv = 0.0
            sy3 = 0.0
            sy4 = 0.0
            for mm in range(M):
                t12 = Tu1[K1[mm]] * Tu2[K2[mm]]
                t3 = Tu3[K3[mm]]
                t4 = Tu4[K4[mm]]
                dt3 = dTu3[K3[mm]]
                dt4 = dTu4[K4[mm]]
                vv = t12 * t3 * t4
                v3 = t12 * dt3 * t4
                v4 = t12 * t3 * dt4
                sov += cop[mm] * vv
                so3 += cop[mm] * v3
                so4 += cop[mm] * v4
                sxv += csx[mm] * vv
                sx3 += csx[mm] * v3
                sx4 += csx[mm] * v4
                syv += csy[mm] * vv
                sy3 += csy[mm] * v3
                sy4 += csy[mm] * v4
            opd_v[i] = sov
            dopd3[i] = so3
            dopd4[i] = so4
            s1x_v[i] = sxv
            ds1x3[i] = sx3
            ds1x4[i] = sx4
            s1y_v[i] = syv
            ds1y3[i] = sy3
            ds1y4[i] = sy4
        return (opd_v, dopd3, dopd4, s1x_v, ds1x3, ds1x4,
                s1y_v, ds1y3, ds1y4)

    _MZ_KERNELS["cheb4d_vd9"] = _cheb4d_vd9
    return _cheb4d_vd9


def _opd_vd9(cop, csx, csy, K1, K2, K3, K4, u1, u2, u3, u4, P):
    """Dispatch the shared value+1st-deriv 3-coef 9-output kernel to Numba
    (default) or the NumPy reference; ULP-equal, ~2x cheaper than three
    ``_opd6`` calls (one basis build, no T'' recurrence)."""
    if _MASLOV_USE_NUMBA:
        kern = _get_cheb4d_vd9_numba()
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
    return _opd_vd9_numpy(cop, csx, csy, K1, K2, K3, K4, u1, u2, u3, u4, P)


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


# ---------------------------------------------------------------------------
# The INPUT field's phase in the saddle condition (finding S6).
#
# The integrand is  E_in(s1(s2, v2)) |det ds1/dv2|^(1/2) exp(2 pi i OPD_waves),
# so the phase that is stationary in v2 is the TOTAL
#
#     Psi(v2) = OPD_waves(s2, v2) + arg E_in(s1(s2, v2)) / (2 pi).
#
# Writing k1 = (1/k0) grad arg E_in -- the input's local wavevector, which is
# ``n1`` times its direction cosine and is exactly what
# ``_local_direction_cosines`` returns -- the second term's v2-gradient is
# ``(k1 . ds1/dv2) / lambda`` in waves.  Dropping it leaves grad_v2 OPD = 0,
# and the symplectic identity ``dOPD/dv2 = -n1 (v1 . ds1/dv2)`` makes THAT the
# v1 = 0 (on-axis collimated) launch ray at every pixel and for every input.
# Keeping it makes the saddle condition ``(v1_in - v1) . ds1/dv2 = 0``: the ray
# whose LAUNCH direction matches the input's local wavevector, which is the
# ray the leading-order expansion is supposed to be about.
#
# ``k1`` reaches here as two more Chebyshev fits over the SAME chart
# coordinates the OPD and s1 fits use (``apply_real_lens_maslov`` builds them),
# so the term and its derivative come out of the same 4-variable evaluator and
# the CPU / GPU twins cannot drift apart.  E_in's own amplitude AND its phase
# VALUE stay where they are -- the integrand still samples the complex field,
# so the input phase enters the answer exactly rather than through the fit;
# the fit is used only to say WHERE the saddle is and how sharp it is.
# ---------------------------------------------------------------------------

def _input_phase_terms(s1x_ev, s1y_ev, k1x_ev, k1y_ev, inv_wavelength):
    """Gradient and Hessian of the input field's phase, in WAVES per unit of
    the normalised chart coordinate -- the S6 term.

    ``s1x_ev`` / ``s1y_ev`` are the full 6-tuples
    ``(f, df_du3, df_du4, d2f_33, d2f_34, d2f_44)`` of the entrance-coordinate
    fits; ``k1x_ev`` / ``k1y_ev`` the same 6-tuples of the local-wavevector
    fits (only the value and the two first derivatives are read).  Returns
    ``(g3, g4, a33, a34, a44)``.

    The mixed second derivative is SYMMETRISED.  ``d(k1 . ds1/du3)/du4`` and
    ``d(k1 . ds1/du4)/du3`` differ by ``(dk1a/ds1b - dk1b/ds1a)`` contracted
    with the two chart tangents, which vanishes for the true field (``k1`` is
    a gradient, so its Jacobian is symmetric) and is pure fit residual here.
    The Newton step solves a symmetric 2x2, so taking the mean is both the
    consistent reading and the one that keeps the two twins' step identical.
    """
    s1x, ds1x_3, ds1x_4, d2x_33, d2x_34, d2x_44 = s1x_ev
    s1y, ds1y_3, ds1y_4, d2y_33, d2y_34, d2y_44 = s1y_ev
    k1x, dkx_3, dkx_4 = k1x_ev[0], k1x_ev[1], k1x_ev[2]
    k1y, dky_3, dky_4 = k1y_ev[0], k1y_ev[1], k1y_ev[2]
    g3 = (k1x * ds1x_3 + k1y * ds1y_3) * inv_wavelength
    g4 = (k1x * ds1x_4 + k1y * ds1y_4) * inv_wavelength
    a33 = (dkx_3 * ds1x_3 + k1x * d2x_33
           + dky_3 * ds1y_3 + k1y * d2y_33) * inv_wavelength
    a44 = (dkx_4 * ds1x_4 + k1x * d2x_44
           + dky_4 * ds1y_4 + k1y * d2y_44) * inv_wavelength
    a34 = (0.5 * (dkx_4 * ds1x_3 + dkx_3 * ds1x_4
                  + dky_4 * ds1y_3 + dky_3 * ds1y_4)
           + k1x * d2x_34 + k1y * d2y_34) * inv_wavelength
    return g3, g4, a33, a34, a44


def _eval_input_phase_terms(evalf, k1_fit, u1, u2, u3, u4):
    """:func:`_input_phase_terms` with the four fits evaluated by ``evalf``
    (the caller's banded / Numba / CuPy 4-variable Chebyshev kernel).

    ``k1_fit`` is the 5-tuple
    ``(coef_s1x, coef_s1y, coef_k1x, coef_k1y, 1 / wavelength)`` that
    :func:`apply_real_lens_maslov` assembles, or ``None`` when the input's
    local wavevector is flat (or was refused) -- in which case no caller
    reaches this function and the arithmetic is the pre-S6 one, bit for bit.
    """
    csx, csy, ckx, cky, inv_wl = k1_fit
    return _input_phase_terms(evalf(csx, u1, u2, u3, u4),
                              evalf(csy, u1, u2, u3, u4),
                              evalf(ckx, u1, u2, u3, u4),
                              evalf(cky, u1, u2, u3, u4), inv_wl)


def _k1_fit_to_device(xp, k1_fit):
    """The four ``k1_fit`` coefficient vectors as ``xp`` arrays.

    The GPU twins take the fit on the host (the driver builds it with NumPy)
    and upload it once, exactly as they do for ``coef_opd`` / ``coef_s1*``.
    ``None`` passes straight through, so a flat input costs nothing.
    """
    if k1_fit is None:
        return None
    return (xp.asarray(k1_fit[0]), xp.asarray(k1_fit[1]),
            xp.asarray(k1_fit[2]), xp.asarray(k1_fit[3]), k1_fit[4])


def _maslov_newton_saddle_xp(xp, opd6, coef_opd, u_s2x, u_s2y, inbox_flat,
                             newton_iter, newton_tol, lin_v3, lin_v4,
                             k1_fit=None):
    """Per-pixel Newton solve for the v2 stationary point, shared by the
    stationary_phase / local_quadrature GPU twins.

    Unlike the CPU integrators (which shrink an ``active`` boolean subset each
    iteration) this evaluates ALL pixels every iteration and freezes the step
    of already-converged pixels (``dv = 0``) -- the SIMD-friendly form for the
    GPU, and numerically equivalent to the CPU active-subset loop (a frozen
    pixel's ``u_v2`` never changes, so its later gradients are irrelevant).
    Out-of-box pixels start ``converged`` (frozen at ``u_v2 = 0``) and are
    zeroed by the caller.  Returns ``(u_v2x, u_v2y, converged)``.

    ``k1_fit`` (S6) adds the input field's own phase to the stationary
    condition -- see :func:`_input_phase_terms`.  ``None`` (a flat input, or
    the refused fallback) leaves every operation below exactly as it was.
    """
    n_px = u_s2x.shape[0]
    u_v2x = xp.zeros(n_px, dtype=xp.float64)
    u_v2y = xp.zeros(n_px, dtype=xp.float64)
    converged = ~inbox_flat
    for _it in range(newton_iter):
        _, g3, g4, H33, H34, H44 = opd6(coef_opd, u_s2x, u_s2y, u_v2x, u_v2y)
        g3 = g3 + lin_v3
        g4 = g4 + lin_v4
        if k1_fit is not None:
            e3, e4, a33, a34, a44 = _eval_input_phase_terms(
                opd6, k1_fit, u_s2x, u_s2y, u_v2x, u_v2y)
            g3 = g3 + e3
            g4 = g4 + e4
            H33 = H33 + a33
            H34 = H34 + a34
            H44 = H44 + a44
        det_H = H33 * H44 - H34 * H34
        # MSL-1 (AUDIT_MASLOV): sign-preserving floor that NEVER returns 0.
        # The old ``sign(det_H)*1e-30 + 1e-30`` cancelled to exactly 0 for a
        # tiny NEGATIVE determinant (det_H in (-1e-30, 0): -1e-30 + 1e-30 = 0)
        # -> division by zero -> inf/NaN saddle step -> a NaN-poisoned pixel.
        # A saddle's OPD-Hessian determinant crosses zero at a fold caustic,
        # so near-caustic pixels are exactly where near-zero det_H arises.
        det_safe = xp.where(xp.abs(det_H) < 1e-30,
                            xp.where(det_H < 0, -1e-30, 1e-30), det_H)
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


def _maslov_newton_saddle_cpu(opd_eval, coef_opd, u_s2x_flat, u_s2y_flat,
                              inbox_flat, newton_iter, newton_tol,
                              lin_v3, lin_v4, progress=None, k1_fit=None):
    """CPU active-subset Newton solve for the per-pixel v2 stationary point --
    the shared engine of the stationary-phase and local-quadrature CPU
    integrators (audit S2-14: the loop was written out identically at both
    sites).  Each iteration shrinks the ``active`` (not-yet-converged) subset and
    evaluates the OPD only there (the CPU-optimal form -- vs the SIMD freeze-all
    GPU twin :func:`_maslov_newton_saddle_xp`, which is numerically equivalent
    but processes every pixel each step).

    ``opd_eval(coef, u1, u2, u3, u4)`` returns the OPD 6-tuple
    ``(f, g3, g4, H33, H34, H44)`` (the caller binds its banded/plain kernel).
    ``progress(it, converged, grad_mag)`` is invoked once per iteration when not
    ``None`` (the stationary-phase caller uses it for its verbose banner).
    ``u_v2x``/``u_v2y`` start at 0; out-of-box pixels start converged (frozen at
    0).  Returns ``(u_v2x, u_v2y, converged)``.  Reproduces the former inline
    loops operation-for-operation, so both routed sites are bit-identical.

    ``k1_fit`` (S6) adds the input field's own phase to the stationary
    condition -- see :func:`_input_phase_terms`.  ``None`` (a flat input, or
    the refused fallback) leaves every operation below exactly as it was."""
    N_px = u_s2x_flat.shape[0]
    u_v2x = np.zeros(N_px, dtype=np.float64)
    u_v2y = np.zeros(N_px, dtype=np.float64)
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
        _, g3, g4, H33, H34, H44 = opd_eval(coef_opd, u1, u2, u3, u4)
        # N4: the linear-in-v2 OPD term (c3*u_v2x + c4*u_v2y) has constant
        # v2-gradient (c3, c4) and zero Hessian, so it shifts the saddle
        # point but not its curvature.  Add it to the gradient here.
        g3 = g3 + lin_v3
        g4 = g4 + lin_v4
        if k1_fit is not None:
            e3, e4, a33, a34, a44 = _eval_input_phase_terms(
                opd_eval, k1_fit, u1, u2, u3, u4)
            g3 = g3 + e3
            g4 = g4 + e4
            H33 = H33 + a33
            H34 = H34 + a34
            H44 = H44 + a44
        det_H = H33 * H44 - H34 * H34
        # MSL-1 (AUDIT_MASLOV): sign-preserving floor that never returns 0
        # (the old ``sign(det_H)*1e-30 + 1e-30`` cancelled to 0 for a tiny
        # negative det_H -> NaN saddle step near fold caustics).
        det_safe = np.where(np.abs(det_H) < 1e-30,
                            np.where(det_H < 0, -1e-30, 1e-30), det_H)
        dv3 = -(H44 * g3 - H34 * g4) / det_safe
        dv4 = -(-H34 * g3 + H33 * g4) / det_safe
        step_size = np.sqrt(dv3 ** 2 + dv4 ** 2)
        damp = np.where(step_size > 0.5,
                        0.5 / np.maximum(step_size, 1e-30), 1.0)
        dv3 = dv3 * damp
        dv4 = dv4 * damp
        u_v2x[active] = np.clip(u_v2x[active] + dv3, -1.0, 1.0)
        u_v2y[active] = np.clip(u_v2y[active] + dv4, -1.0, 1.0)
        grad_mag = np.sqrt(g3 ** 2 + g4 ** 2)
        newly = np.zeros(N_px, dtype=bool)
        newly[active] = grad_mag < newton_tol
        converged |= newly
        if progress is not None:
            progress(it, converged, grad_mag)
    return u_v2x, u_v2y, converged


def _tukey_taper(u, alpha=0.2):
    """Symmetric Tukey (cosine-tapered) window evaluated at normalized coords
    ``u`` on [-1, 1] (audit S2-14).  Unity in the interior ``|u| <= 1 - alpha``
    with a raised-cosine roll-off to 0 at ``|u| = 1``.

    This is the ONE definition of the Maslov quadrature window.  It
    reproduces the ``1.0 - alpha`` taper start and the
    ``0.5*(1 + cos(pi*(|u| - (1 - alpha))/alpha))`` roll-off operation-for-
    operation, so every routed call is bit-identical."""
    au = np.abs(u)
    w = np.ones_like(u)
    m = au > 1.0 - alpha
    w[m] = 0.5 * (1.0 + np.cos(np.pi * (au[m] - (1.0 - alpha)) / alpha))
    return w


# Largest Gram condition number ``cond(A^T A) = lam_max/lam_min`` for which the
# normal-equations Cholesky in :func:`_solve_fit` is trusted.  Above it the
# solve falls through to the min-norm SVD, and above _GRAM_COND_SINGULAR the
# caller is warned.  Derivation in :func:`_solve_fit`.
_GRAM_COND_MAX = 1.0e12
_GRAM_COND_SINGULAR = 1.0 / np.finfo(np.float64).eps    # ~4.5e15


def _solve_fit(A, RHS, gram_factor=None):
    """Least-squares solve for the Maslov Chebyshev fit ``A @ coef ~= RHS``.

    Normal-equations Cholesky (``G = A^T A``; solve ``G coef = A^T RHS``)
    instead of the ``gelsd`` full-SVD ``lstsq``, which is O(M^3) with
    tiny ``M`` (70 at poly_order=4) rather than O(n_rays M^2).  A caller
    sweeping the SAME optic can precompute ``gram_factor`` and pass it
    in (only the cheap ``A^T RHS`` GEMM + back-substitution then re-run
    per field).  Returns ``coef`` (M, k).

    Conditioning gate
    -----------------
    ``A`` is a normalized tensor-Chebyshev Vandermonde, ~1.5x
    oversampled, so squaring its condition number in ``G`` is safe on a
    well-conditioned chart -- and NOT on every chart, which the
    ``LinAlgError`` fallback ladder below cannot see: a numerically
    positive-semidefinite but RANK-DEFICIENT ``G`` factors happily and
    returns an arbitrary member of the solution set.

    Measured (audit follow-up, f = 6 mm N-BK7 biconvex, 0.2 mm aperture,
    poly_order = 4): ``rank(A) = 65`` of 70 columns, ``cond(A) = 1.81e+15``,
    ``cond(G) = 6.18e+18``.  Two runs of the SAME optic whose design matrices
    agree to 3.1e-15 and whose right-hand sides agree to 6.8e-13 waves came
    back with coefficients **0.869 waves apart** -- both with the same fit
    residual (1.613e-09 vs 1.756e-09 waves, and 1.756e-09 cross-evaluated), so
    the difference lives entirely in the null space.  That is invisible on the
    training manifold and NOT invisible inside the v2 integral, which samples
    ``(s2, v2)`` combinations off it: the two fields differed by relL2 0.45.
    (The audit's own conditioning note measured ``cond(G)`` = 1.24e9 / 8.33e13 /
    4.09e17 / 1.41e19 at poly_order 4 / 6 / 8 / 10 on a 1.5 mm-aperture chart;
    a small, fast chart reaches those numbers at a much lower order.)

    So: measure ``cond(G)`` once (an ``M x M`` ``eigvalsh``, microseconds beside
    the ``A^T A`` GEMM) and

    * ``cond(G) <= _GRAM_COND_MAX`` (1e12, still ~4 float64 digits of margin in
      the squared system) -- take the fast Cholesky, byte-identical to v5.21;
    * above it -- take ``np.linalg.lstsq``, whose minimum-norm solution is a
      deterministic, unique function of ``(A, RHS)`` -- the conservative
      answer, and the one that matters here;
    * above ``_GRAM_COND_SINGULAR`` (``1/eps``) -- also WARN, because there the
      fit is genuinely rank-deficient and even the min-norm answer depends on
      ``lstsq``'s ``rcond`` cut: the extra columns are not determined by the
      data and a lower ``poly_order`` (or a wider chart) is the real fix.
    """
    b = A.T @ RHS
    if gram_factor is not None:
        try:
            from scipy.linalg import cho_solve
            return cho_solve(gram_factor, b, check_finite=False)
        except (ImportError, ValueError, np.linalg.LinAlgError):
            pass                       # scipy absent / stale-shape factor
    G = A.T @ A
    try:
        _ev = np.linalg.eigvalsh(G)
        _hi = float(_ev[-1])
        _lo = float(_ev[0])
        _cond = (np.inf if not (_hi > 0.0) or _lo <= 0.0 else _hi / _lo)
    except np.linalg.LinAlgError:
        _cond = np.inf
    if _cond > _GRAM_COND_MAX:
        if _cond > _GRAM_COND_SINGULAR:
            import warnings
            warnings.warn(
                f"apply_real_lens_maslov: the canonical-map design matrix is "
                f"RANK-DEFICIENT at float64 -- cond(A^T A) = {_cond:.2e} over "
                f"{A.shape[1]} basis terms on {A.shape[0]} rays.  The "
                f"minimum-norm least-squares solution is returned (unique and "
                f"reproducible), but the undetermined directions are set by "
                f"the solver's rcond cut, not by the ray data: two charts that "
                f"agree to float64 noise can still produce coefficients that "
                f"differ off the training manifold, and the v2 integral "
                f"samples exactly there.  Reduce poly_order, widen the chart "
                f"(larger aperture / input_na), or raise "
                f"ray_field_samples / ray_pupil_samples.",
                RuntimeWarning, stacklevel=3)
        coef, *_ = np.linalg.lstsq(A, RHS, rcond=None)
        return coef
    try:
        from scipy.linalg import cho_factor, cho_solve
        return cho_solve(cho_factor(G, check_finite=False), b,
                         check_finite=False)
    except (ImportError, ValueError, np.linalg.LinAlgError):
        # scipy absent, or G not positive-definite (rank-deficient freeform)
        try:
            return np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(A, RHS, rcond=None)
            return coef


def _select_poly_order_auto(u1, u2, u3, u4, opd, *, order_min, order_max,
                            target_waves, rtol, holdout_stride=5):
    """Pick the lowest tensor-Chebyshev total-degree that fits ``opd`` (waves)
    for ``poly_order='auto'`` -- WITHOUT over-fitting.

    The four ``u*`` are the [-1,1]-normalised ``(s2x, s2y, v2x, v2y)`` fit coords
    (same coords the production fit uses); ``opd`` is the linear-detrended OPD
    residual in waves.  Order is scored on a **held-out** strided ray subset
    (not the training residual, which decreases monotonically with order and
    would always pick ``order_max``), so an order that only fits ray-node noise
    is rejected.  Ascends from ``order_min``; returns the first order whose
    held-out RMS residual is below ``target_waves``, else the lowest order whose
    residual is within ``(1+rtol)`` of the best seen up to ``order_max``.  The
    tensor-Chebyshev total-degree sets nest (degree<=p columns are a subset of
    degree<=p+1), so one ``order_max`` Vandermonde + column-subset per trial
    order costs a single basis build.  Returns ``(order:int, residual:float)``.
    """
    n = int(u1.shape[0])
    val_mask = np.zeros(n, dtype=bool)
    val_mask[::holdout_stride] = True
    fit_mask = ~val_mask
    # A held-out split is only trustworthy if the training set still over-
    # samples the largest candidate order; otherwise score in-sample (the
    # plateau/target logic still gives a sensible diminishing-returns stop).
    if (fit_mask.sum() < 1.5 * _count_multi_indices_4d(order_max)
            or val_mask.sum() < 8):
        fit_mask = np.ones(n, dtype=bool)
        val_mask = np.ones(n, dtype=bool)
    T1 = _chebyshev_vandermonde(u1, order_max)
    T2 = _chebyshev_vandermonde(u2, order_max)
    T3 = _chebyshev_vandermonde(u3, order_max)
    T4 = _chebyshev_vandermonde(u4, order_max)
    mi = _multi_indices_total_degree(4, order_max)
    Afull = np.empty((n, len(mi)), dtype=np.float64)
    deg = np.empty(len(mi), dtype=np.int64)
    for j, (k1, k2, k3, k4) in enumerate(mi):
        Afull[:, j] = T1[k1] * T2[k2] * T3[k3] * T4[k4]
        deg[j] = k1 + k2 + k3 + k4
    res_by_p = {}
    for p in range(order_min, order_max + 1):
        cols = deg <= p
        A = Afull[:, cols]
        coef, *_ = np.linalg.lstsq(A[fit_mask], opd[fit_mask], rcond=None)
        res = float(np.sqrt(np.mean((opd[val_mask] - A[val_mask] @ coef) ** 2)))
        res_by_p[p] = res
        if res < target_waves:
            return p, res
    best = min(res_by_p.values())
    for p in range(order_min, order_max + 1):
        if res_by_p[p] <= best * (1.0 + rtol):
            return p, res_by_p[p]
    return order_max, res_by_p[order_max]


def _fold_airy_eval(k, A, zeta, a0, a1):
    """Evaluate the Chester-Friedman-Ursell fold-Airy uniform form given the
    mapped mean phase ``A``, the fold parameter ``zeta`` and the two CFU
    amplitude coefficients ``a0, a1``::

        I = e^{ikA} 2pi [ a0 k^{-1/3} Ai(-k^{2/3} zeta)
                          - i a1 k^{-2/3} Ai'(-k^{2/3} zeta) ]

    This is the closing expression of :func:`uniform_fold_airy` (which computes
    ``A, zeta, a0, a1`` from the two coalescing real saddles), factored out so
    the DARK side of a fold can be reached by analytic continuation: ``zeta`` may
    be **negative** (no real saddles -- the coalesced rays have become a
    complex-conjugate pair), whereupon the Airy argument ``-k^{2/3} zeta`` is
    POSITIVE and ``Ai``/``Ai'`` give the exponentially-decaying dark-side tail
    (the missing evanescent diffraction the pure geometric multibranch sum drops
    at ``zeta = 0``).  ``a0, a1`` are the SMOOTH (real-analytic through the
    caustic) coefficients, so the traced ``caustic='uniform'`` completion fits
    them on the BRIGHT side (``zeta > 0``) and evaluates HERE at ``zeta < 0`` --
    reusing this one CFU kernel for both sides.  ``A``/``zeta``/``a0``/``a1`` may
    be scalars or broadcastable arrays.  Convention ``exp(-i w t)`` /
    ``exp(+i k f)`` (matches the rest of the library)."""
    from scipy.special import airy
    ai, aip, _, _ = airy(-(k ** (2.0 / 3.0)) * zeta)
    return np.exp(1j * k * A) * 2 * np.pi * (
        a0 * k ** (-1.0 / 3.0) * ai - 1j * a1 * k ** (-2.0 / 3.0) * aip)


def uniform_fold_airy(k, t1, t2, f1, f2, fpp1, fpp2, g1=1.0, g2=1.0):
    """Uniform (Chester-Friedman-Ursell) value of the oscillatory integral
    ``I(k) = int g(t) exp(i k f(t)) dt`` near a **FOLD** caustic -- where two
    stationary points ``t1, t2`` of ``f`` coalesce and ordinary stationary phase
    diverges (``f'' -> 0``).  This stays finite through the caustic and reduces
    to the two-saddle stationary-phase sum away from it.

    Given the two real stationary points with their phase ``f``, curvature
    ``f''`` and amplitude ``g`` (need NOT be pre-sorted), maps ``f`` to the cubic
    normal form and returns the Airy-uniform result::

        f1<=f2 (sort),  A=(f1+f2)/2,  zeta=[3/4 (f2-f1)]^(2/3) >= 0
        beta_j = g_j sqrt(2/|f''_j|) exp(i sgn(f''_j) pi/4)     (SPA amplitude+phase)
        a0 =  [1/2(e^{ipi/4}beta2 + e^{-ipi/4}beta1)] zeta^{1/4}
        a1 = -[1/2(e^{ipi/4}beta2 - e^{-ipi/4}beta1)] zeta^{-1/4}
        I  = e^{ikA} 2pi [ a0 k^{-1/3} Ai(-k^{2/3}zeta) - i a1 k^{-2/3} Ai'(-k^{2/3}zeta) ]

    The branch discipline (the crux of the method) is pinned by the
    stationary-phase Maslov phase ``exp(i sgn(f'') pi/4)`` per saddle -- NOT by a
    ``sqrt(f''/2u)`` root, whose sign is ambiguous.  Validated to machine
    precision (~1e-14) against the exact cubic-phase integrals
    ``int (1 + a t) exp(i k (t^3/3 - c t)) dt =
    2pi k^{-1/3} Ai(-k^{2/3}c) - a 2pi i k^{-2/3} Ai'(-k^{2/3}c)`` for both
    ``a1 = 0`` and ``a1 != 0``, and stays finite on the caustic (``zeta -> 0``).

    This is the caustic-finite integrator underlying a uniform Maslov evaluator
    and the Airy hand-off for a multi-branch geometric-optics field; a cusp
    caustic needs the Pearcey generalisation (not implemented here).  Convention:
    ``exp(-i w t)`` / ``exp(+i k f)`` (matches the rest of the library).

    Parameters
    ----------
    k : float -- wavenumber (large-parameter of the asymptotics).
    t1, t2, f1, f2, fpp1, fpp2 : the two stationary points and their ``f``,
        ``f''`` values.
    g1, g2 : amplitude ``g`` at each stationary point (default 1).

    Returns
    -------
    complex -- the uniform integral value.
    """
    if f2 < f1:                        # sort so saddle 2 has the higher phase
        t1, t2, f1, f2, fpp1, fpp2, g1, g2 = t2, t1, f2, f1, fpp2, fpp1, g2, g1
    A = 0.5 * (f1 + f2)
    zeta = (0.75 * (f2 - f1)) ** (2.0 / 3.0)
    b1 = g1 * np.sqrt(2.0 / abs(fpp1)) * np.exp(1j * np.sign(fpp1) * np.pi / 4)
    b2 = g2 * np.sqrt(2.0 / abs(fpp2)) * np.exp(1j * np.sign(fpp2) * np.pi / 4)
    ep, em = np.exp(1j * np.pi / 4), np.exp(-1j * np.pi / 4)
    a0 = 0.5 * (ep * b2 + em * b1) * zeta ** 0.25
    a1 = -0.5 * (ep * b2 - em * b1) * zeta ** (-0.25)
    # The closing CFU expression is shared with the dark-side continuation via
    # ``_fold_airy_eval`` (byte-identical to the former inline evaluation).
    return _fold_airy_eval(k, A, zeta, a0, a1)


def pearcey(x, y, *, mmax=60, pmax=60, tol=1e-16):
    """Canonical Pearcey integral -- the CUSP-caustic diffraction special
    function (the cusp analogue of the Airy function for folds)::

        P(x, y) = int_{-inf}^{inf} exp(i (t^4 + x t^2 + y t)) dt      (DLMF Psi_2)

    Evaluated by its everywhere-convergent double series (``P`` is even in ``y``,
    so only even powers survive), using the quartic Gaussian moment
    ``int t^{2k} exp(i t^4) dt = 1/2 Gamma((2k+1)/4) exp(i pi (2k+1)/8)``::

        P(x,y) = sum_{m,p>=0} (i x)^m/m! * (-1)^p y^{2p}/(2p)!
                             * 1/2 Gamma((2(m+p)+1)/4) exp(i pi (2(m+p)+1)/8)

    Validated to machine precision against the exact cusp value
    ``P(0,0) = 1/2 Gamma(1/4) exp(i pi/8)`` (``|P|=1.812804``, ``arg=22.5 deg``),
    the even-in-``y`` symmetry, and a contour-rotated quadrature at ``y=0``.  The
    series converges everywhere but slows for large ``|x|, |y|``; raise
    ``mmax/pmax`` past the default (good to ``|x|,|y| ~ 15``).  Convention
    ``exp(-i w t)`` / ``exp(+i phase)`` matches the rest of the library.

    This is the caustic-finite kernel for a uniform cusp (astigmatic-focus)
    evaluator -- the cusp peer of :func:`uniform_fold_airy` -- whose CFU quartic
    mapping (3 coalescing saddles) is the integration step layered on top.
    """
    from math import factorial

    from scipy.special import gamma
    xj, yj = complex(x), complex(y)
    total = 0j
    for m in range(mmax + 1):
        xm = (1j * xj) ** m / factorial(m)
        row = 0j
        for p in range(pmax + 1):
            k = m + p
            term = (xm * ((-1) ** p) * (yj ** (2 * p)) / factorial(2 * p)
                    * 0.5 * gamma((2 * k + 1) / 4.0)
                    * np.exp(1j * np.pi * (2 * k + 1) / 8.0))
            row += term
            if p > 3 and abs(term) < tol * (abs(total) + 1e-30):
                break
        total += row
        if m > 3 and abs(row) < tol * (abs(total) + 1e-30):
            break
    return total


def apply_real_lens_maslov(
    E_in: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    ray_field_samples: int = 16,
    ray_pupil_samples: int = 16,
    poly_order: Union[int, str] = 4,
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
    levin_tol: float = 1e-3,
    collimated_input: bool = False,
    input_na: Optional[float] = None,
    input_wavevector_saddle: Optional[bool] = None,
    normalize_output: str = 'power',
    verbose: bool = False,
    progress: Optional[Any] = None,
    use_gpu: bool = False,
    fold_split: bool = False,
    geometry: Optional['LensGeometry'] = None,
    numerics: Optional['LensNumerics'] = None,
    resources: Optional['LensResources'] = None,
    config: Optional['LensConfig'] = None,
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
    ``lens_prescription`` to the EXIT VERTEX plane of its last surface
    (``rt.trace`` leaves rays on the curved surface itself; the signed
    ``t = -z/N`` transfer is applied before anything is fitted), fits a
    4-variable Chebyshev tensor-product polynomial to ``s1(s2, v2)`` and
    ``OPD(s2, v2)``, then evaluates the Van Vleck-Maslov integral

        E(s2) = k/(2 pi i) *
                integral E_in(s1(s2, v2)) * exp(2 pi i OPD(s2, v2))
                          * |det(ds1/dv2)|^(1/2)  d^2 v2

    at each output pixel.  The SQUARE ROOT of the Jacobian and the
    ``k/(2 pi i) = 1/(i lambda)`` prefactor are what make the result an
    absolutely-scaled field (audit S4): on a free-space chart, where
    ``ds1/dv2 = -z I`` exactly, ``normalize_output='none'`` reproduces the
    angular-spectrum field to 0.3 % at two wavelengths and three distances.  See the v3.4.x release notes (or the
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
    to a concrete integrator from the fitted chart's v2-oscillation count
    (:func:`_v2_oscillation_bound`): **uniform 'quadrature'** when it is
    well-resolved (``4 * v2_osc <= _N_V2_AUTO_MAX``) -- exact and
    caustic-safe, and where low-oscillation / near-caustic charts fall --
    and the asymptotic **'stationary_phase'** only when uniform quadrature
    would need more than the sample cap (the very oscillatory / high-NA
    regime where quadrature is both slow and speckles).  Byte-identical to
    the method it picks in the well-resolved regime (auto -> quadrature at
    the same auto-sized ``n_v2``).

    The asymptotic fallback WARNS (audit S2).  A leading-order saddle
    evaluation is only accurate where the integrand really is oscillatory;
    at or near the lens EXIT plane of a focusing system it is not -- both
    the v2-Hessian and ``ds1/dv2`` collapse there -- and both asymptotic
    evaluators are then O(1) wrong (measured relL2 0.84 for
    'stationary_phase' and 2.6 for 'local_quadrature' at its defaults,
    against a converged uniform quadrature on an f = 6 mm singlet).  Pass
    ``integration_method='quadrature'`` with an explicit ``n_v2`` for an
    exit-plane field, or ``output_plane_distance=`` to put the observation
    plane where the asymptotics belong.

    ``integration_method='levin'`` (v5.21) evaluates the v2 integral by the
    adaptive delaminating Levin method (:mod:`lumenairy._math.levin`, after
    Chen-Serkh-Bremer-Aubry arXiv:2506.02424): caustic-UNIFORM with **no
    saddle finding** (finite and accurate through folds where
    'stationary_phase' / 'local_quadrature' diverge) at a per-pixel cost
    INDEPENDENT of the v2 oscillation count (unlike 'quadrature', whose cost
    grows as ``n_v2^2``).  ``levin_tol`` sets the absolute per-pixel tolerance
    (a rigorous residual bound is refined until met).  Pure NumPy and adaptive
    per pixel -- suited to caustic-band ROI studies and hard high-NA charts,
    not (yet) to full-grid production sweeps.

    **The two ASYMPTOTIC evaluators follow the INPUT field's local wavevector**
    (audit finding S6).  The v2 integrand's phase is
    ``arg E_in(s1(v2)) + 2 pi OPD_waves``; solving ``grad_v2 OPD = 0`` alone
    selects, by the symplectic identity ``dOPD/dv2 = -n1 (v1 . ds1/dv2)``, the
    ``v1 = 0`` collimated launch ray at every pixel -- right for a flat input,
    wrong for the diverging / converging / tilted one the pupil chart is
    deliberately sized to cover (``na_proxy = na_lens + na_input``).  So when
    the input's wavefront spread over the traced aperture exceeds
    ``_SADDLE_FLAT_INPUT_NA``, its local wavevector
    ``k1 = (1/k0) grad arg E_in`` is fitted over the same chart coordinates as
    the OPD and the entrance coordinates, and ``(k1 . ds1/dv2) / lambda`` joins
    the saddle gradient (and its derivative the Hessian, which sets the
    Gaussian-moment amplitude, the Maslov signature and the local_quadrature
    window).  The saddle condition becomes ``(v1_in - v1) . ds1/dv2 = 0``: the
    ray whose LAUNCH direction is the input's own.  ``E_in`` is still sampled
    as the complex field, so its phase enters the answer exactly and the fit
    only places the saddle.  A flat input (``k1 = 0`` everywhere, which is what
    a real non-negative ``E_in`` gives EXACTLY) and ``collimated_input=True``
    never build the fit at all, so they run the OPD-only arithmetic bit for
    bit -- the S6 term is not small there, it is absent.  When the local
    wavevector is not
    representable on the chart -- speckle, or a hard-edged aperture, whose dark
    side reports a launch direction of 0 that the illuminated side contradicts
    -- the fit's intensity-weighted residual exceeds
    ``_K1_FIT_RESIDUAL_MAX``, the OPD-only saddle is kept, and the S6
    ``RuntimeWarning`` fires naming both measurements.  The term is
    ``k1 . ds1/dv2``, so the same gate also refuses a chart that cannot carry
    the OTHER factor: when the order-``poly_order`` fit to the entrance
    coordinates has relative RMS residual above ``_S1_FIT_RESIDUAL_MAX`` the
    fitted saddle is placed on a ray the chart has itself misplaced, which a
    larger ``poly_order`` -- or an explicit ``input_na``, since ``na_proxy``
    sizes the pupil box from the input's angular spectrum and a uniform tilt
    inflates it threefold -- repairs.  A wavefront steeper
    than the grid's own Nyquist angle ``lambda / (2 dx)`` is a DIFFERENT
    failure and this gate does not see it: the phase-difference estimator
    aliases to a wrapped direction that is perfectly smooth (measured
    residual 1.2e-10 at a 1.2 x Nyquist tilt), so the whole sampled field --
    not just this saddle -- is the aliased one.  Sample the input finely
    enough that ``max|grad arg E_in| dx < pi``.

    ``input_wavevector_saddle`` chooses that stationary point PER CALL.  The
    two candidates are ``grad_v2 OPD = 0`` (the ``v1 = 0`` collimated launch
    ray at every pixel) and ``grad_v2[arg E_in(s1(v2)) + 2 pi OPD_waves] = 0``
    (the ray whose launch direction is the input's own).  ``None`` (the
    default) decides as described above -- the input's own wavefront spread
    and the fit's own residual pick; ``True`` uses the fitted local wavevector
    whenever the input is not flat, faithful fit or not; ``False`` always
    solves ``grad_v2 OPD = 0``, and warns, which is what a caller who wants
    the OPD-only answer on a non-collimated input passes (the changelog's
    Migration note says when that is what someone wants).  It is
    keyword-only and NOT a ``LensNumerics`` field on purpose: it is a property
    of the INPUT FIELD, not of the optic or the machine, so it cannot travel
    in a config object that is reused across fields.  The module-level
    ``_S6_INPUT_WAVEVECTOR_SADDLE`` sets the process default this keyword
    overrides.  An input tilt BELOW the engagement bar (about 3.3e-4 rad,
    ``_SADDLE_FLAT_INPUT_NA / 3``) is left on the OPD-only saddle and its
    spot lands ``f * theta`` off the truth with no warning -- measured 4.7 um
    at f = 13 mm, 35 um at 100 mm and 354 um at 1 m; pass
    ``input_wavevector_saddle=True`` if your focal length makes that matter.
    ``'quadrature'`` and
    ``'levin'`` integrate the true integrand pointwise, have no saddle, and
    are untouched by all of this.

    ``poly_order`` (default ``4``) accepts ``'auto'`` (v5.21): the tensor-
    Chebyshev fit order is raised from ``_MZ_POLY_AUTO_MIN`` until the OPD-fit
    residual, scored on a **held-out** ray split (not the training residual,
    which only ever decreases and would always pick the ceiling), stops
    improving by more than ``_MZ_POLY_AUTO_RTOL`` or drops below
    ``_MZ_POLY_AUTO_TARGET`` waves RMS -- capped at ``_MZ_POLY_AUTO_MAX``.  A
    smooth optic then fits at a cheap low order and a strongly-aberrated / near-
    caustic chart is given the order it needs, with no manual tuning and no
    over-fit.  ``'auto'`` sizes the ray-count guards for its *maximum* candidate
    order, so it wants the default (dense) ray sampling; pass an explicit int to
    pin the order (and the runtime) for a production sweep.

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
    ray trace, only re-propagating + refitting (cheap).  The composed field
    matches baking the same distance into the prescription's last thickness --
    the two produce the SAME canonical chart, verified to 6.8e-13 waves in OPD
    and 0 m in s1 at d = 0.5 ... 5 mm -- and the ROI window is identical to the
    full-grid slice (measured bit-identical, max |dE| = 0.0).

    The composed FIELD agrees to relL2 1.7e-05 ... 9.3e-04 on an f = 6 mm,
    0.2 mm-aperture singlet at d = 0.5 ... 5 mm.  The residual gap is not
    the composition: it is the non-uniqueness of
    the Chebyshev fit itself on a rank-deficient chart, where two charts
    agreeing to float64 noise can land on different members of the same
    solution set.  :func:`_solve_fit` now routes such charts to the
    deterministic minimum-norm SVD (and warns when they are rank-deficient at
    float64), which is what brought this from relL2 0.45 ... 1.38 down to the
    numbers above.
    Not yet combined with ``fold_split`` (raises ``NotImplementedError`` rather
    than silently dropping the requested observation plane).

    ``normalize_output`` (default ``'power'``) sets the returned field's
    absolute amplitude scale.  The raw integral is ALREADY absolutely
    normalised (audit S4) -- it carries the Van Vleck density
    ``|det(ds1/dv2)|^(1/2)`` and the d = 2 prefactor ``k/(2 pi i)``, and
    reproduces the exact free-space field to 0.3 % with
    ``normalize_output='none'`` -- so ``'power'`` / ``'peak'`` are now
    diagnostics (they also absorb aperture clipping and chart truncation)
    rather than the only way to get a meaningful amplitude:

    * ``'power'`` -- rescale so ``sum |E_out|^2 == sum |E_in|^2``.
    * ``'peak'``  -- rescale so ``max |E_out| == max |E_in|``.
    * ``'none'``  -- return the raw integral (no rescale).
    * a scalar (int/float/complex) -- multiply by exactly that factor.

    With ``roi=`` the two *global* modes are unavailable: ``'power'`` and
    ``'peak'`` are reductions over the FULL output grid, which the ROI path
    never evaluates.  Requesting either with ``roi=`` warns and falls back to
    the raw ``'none'`` scale (which is what makes the ROI byte-identical to the
    corresponding slice of a ``normalize_output='none'`` full-grid run); pass
    ``'none'`` explicitly to silence the warning, or a scalar factor (which
    *is* window-independent and is applied).

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

    Configuration objects
    ---------------------
    geometry, numerics, resources, config : optional
        :class:`~lumenairy.LensGeometry` / :class:`~lumenairy.LensNumerics` /
        :class:`~lumenairy.LensResources`, or the
        :class:`~lumenairy.LensConfig` that holds all three, as an alternative
        to spelling the settings out as keywords.  Purely ADDITIVE: every
        keyword above still works with the same default, and a call that
        passes none of the four runs exactly the code it ran before.  A set
        field and a keyword for the SAME setting must agree or the call
        raises; a set field this function has no parameter for also raises
        (``config.narrowed_to('apply_real_lens_maslov')`` drops those
        deliberately).  Most of this engine's tuning constants
        (``levin_tol``, ``poly_order``, ``integration_method``, ...) are
        deliberately NOT config fields; see ``docs/lens_configuration.md``.
    """
    # Defensive guard via the shared
    # ``_check_2d_scalar_field`` helper -- siblings missed by the
    # v4.15.2 closure now share the same first-line guard.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_maslov', input_kind='field')
    # Config objects, if any, are merged into the keywords and the call is
    # re-entered with them -- so the configured path is the SAME code as the
    # equivalent keyword call, by construction rather than by review.  Four
    # ``is not None`` tests when nothing is configured; nothing else changes.
    if _wants_config(geometry, numerics, resources, config):
        return apply_real_lens_maslov(E_in, **_resolve_lens_config(
            apply_real_lens_maslov, locals(), geometry=geometry,
            numerics=numerics, resources=resources, config=config))

    # Explicit mirror-in-surfaces guard, mirroring
    # ``apply_real_lens_traced``.  A hand-built prescription with
    # ``surfaces[i]['is_mirror']=True`` (or ``glass_after='MIRROR'``)
    # slips past the shared ``_check_no_silent_fold_drop`` (which only
    # inspects ``prescription['elements']``), and the Maslov leg would
    # then silently treat the mirror as a refractor with the wrong
    # sign.  Fail loudly with the same mirror-specific message as
    # ``apply_real_lens_traced``.
    # fold_split=True auto-handles a folded prescription instead of
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
            # E-L20 (audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25): ``_leg_kw``
            # below forwards most of the signature but omitted
            # output_plane_distance / output_plane_n / roi / progress, so a
            # folded run SILENTLY evaluated the last lens vertex instead of
            # the requested observation plane (and silently ignored a
            # requested ROI window).  The docstring already says these are
            # "not yet combined with fold_split"; make that a hard failure
            # naming the dropped kwargs rather than a wrong answer.
            # ``progress`` IS forwarded (it is a pure callback, no numerics --
            # each leg simply reports its own 0->1 fraction).
            _dropped = []
            if roi is not None:
                _dropped.append(f'roi={roi!r}')
            if output_plane_distance:   # same truthiness gate the main path uses
                _dropped.append(
                    f'output_plane_distance={output_plane_distance!r}')
                if output_plane_n != 1.0:
                    _dropped.append(f'output_plane_n={output_plane_n!r}')
            if _dropped:
                raise NotImplementedError(
                    f"apply_real_lens_maslov: fold_split=True is not yet "
                    f"combined with {', '.join(_dropped)} -- the folded path "
                    f"chains one Maslov call per refractive leg and has no "
                    f"way to place a downstream observation plane / ROI "
                    f"window on the final leg, so these were previously "
                    f"DROPPED silently (the field came back at the last lens "
                    f"vertex of the last leg).  Split the prescription "
                    f"yourself with "
                    f"lumenairy.io.split_prescription_at_mirrors(rx) and pass "
                    f"these kwargs to the final leg's "
                    f"apply_real_lens_maslov call.")
            from .elements import apply_mirror
            _leg_kw = dict(
                wavelength=wavelength, dx=dx, dy=dy, progress=progress,
                ray_field_samples=ray_field_samples,
                ray_pupil_samples=ray_pupil_samples, poly_order=poly_order,
                n_v2=n_v2, output_subsample=output_subsample,
                extract_linear_phase=extract_linear_phase, chunk_v2=chunk_v2,
                use_numexpr=use_numexpr, integration_method=integration_method,
                stationary_newton_iter=stationary_newton_iter,
                stationary_newton_tol=stationary_newton_tol,
                local_n_samples=local_n_samples,
                local_window_sigma=local_window_sigma,
                levin_tol=levin_tol,
                collimated_input=collimated_input, input_na=input_na,
                input_wavevector_saddle=input_wavevector_saddle,
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
                        # S11 (audit): pass ``dy``.  These two mirror-leg
                        # gaps were the only anamorphic-unaware calls left
                        # in this driver -- every other site threads dx/dy
                        # separately -- so an anamorphic grid silently
                        # propagated the fold gaps with dy = dx.
                        E = angular_spectrum_propagate(E, _din, wavelength,
                                                       dx, dy)
                    E = apply_mirror(
                        E, wavelength=wavelength, dx=dx, dy=dy,
                        radius=_m.get('radius'), conic=_m.get('conic', 0.0),
                        aperture_diameter=_m.get('clear_aperture'))
                    if abs(_dout) > 0.0:
                        E = angular_spectrum_propagate(E, _dout, wavelength,
                                                       dx, dy)
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

    # CuPy dispatch mirrors apply_real_lens -- opt in via
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
        # WP-A2: pass the y extent too, so an anamorphic grid is checked
        # against the axis that truncates FIRST rather than against a
        # semi-extent that exists on neither axis.
        _warn_if_aperture_exceeds_grid(
            lens_prescription, N, dx, source='apply_real_lens_maslov',
            N_y=int(E_in.shape[0]), dy=dy)
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
    # Validate the key the way ``apply_real_lens`` does -- an out-of-range
    # or non-integer ``stop_index`` RAISES with a Section 2 prefix rather
    # than being int()-ed into a silent warning path (where a negative
    # index reads as "non-entrance stop" and a float raises a bare
    # TypeError).  Normalising also makes ``stop_index=-1`` on a 2-surface
    # lens mean surface 1, as Python indexing does, instead of tripping
    # the non-entrance warning.
    _stop_index = _normalise_stop_index(
        lens_prescription.get('stop_index'),
        len(lens_prescription.get('surfaces') or surfaces),
        fn_name='apply_real_lens_maslov')
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

    # v5.21: poly_order='auto' -- provisionally size the ray-count guards and
    # arrays for the largest candidate order, then (once the trace + fit inputs
    # exist) select the lowest order that fits the OPD on a held-out ray split.
    # Resolved to a concrete int at the fit stage below.
    if isinstance(poly_order, str):
        if poly_order != 'auto':
            raise ValueError(
                f"apply_real_lens_maslov: poly_order must be an int or 'auto'; "
                f"got {poly_order!r}.")
        _poly_auto = True
        poly_order = _MZ_POLY_AUTO_MAX
    else:
        _poly_auto = False
        poly_order = int(poly_order)

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
    # the lens EFL alone drops any divergent /
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
    # ``rt.trace`` leaves every ray ON the last surface, at z = sag(rho) in
    # that surface's local frame -- NOT on the exit vertex plane.  The
    # canonical map (s2, v2) -> OPD built below is documented (and consumed)
    # as an EXIT-PLANE chart, so the rays have to be carried the signed
    # straight-line leg t = -z/N first; skipping it injects a pure rho^2
    # (defocus) OPD of n_exit*sag(rho), which the Chebyshev fit absorbs
    # silently.  ``at_exit_vertex`` is the single shared operator for that
    # transfer (it also resolves n_exit from the prescription, kills grazing
    # rays instead of teleporting them, and is idempotent).
    exit_rays = tr.at_exit_vertex()
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
    #
    # ``exit_rays`` already sits on the vertex plane (z = 0), so this leg is
    # the full ``d/N``; composed with the vertex transfer above it is the
    # ``(d - z_sag)/N`` the requested plane actually needs, with the sag leg
    # correctly priced at the EXIT medium's index and the free leg at
    # ``output_plane_n``.
    ex_x = exit_rays.x
    ex_y = exit_rays.y
    ex_opd = exit_rays.opd
    if output_plane_distance:
        _Nz = exit_rays.N
        _t = np.where(alive & (np.abs(_Nz) > rt.EXIT_VERTEX_GRAZING_TOL),
                      output_plane_distance / np.where(
                          np.abs(_Nz) > rt.EXIT_VERTEX_GRAZING_TOL, _Nz, 1.0),
                      0.0)
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

    # v5.21: resolve poly_order='auto' now that the fit inputs exist -- pick the
    # lowest tensor-Chebyshev order that fits the (linear-detrended) OPD residual
    # on a held-out ray split, so a smooth optic uses a cheap low order and a
    # strongly-aberrated one is given the order it needs (no manual tuning, no
    # over-fit).  The rest of the fit + integrate path below is unchanged and
    # runs at the resolved int order.
    if _poly_auto:
        poly_order, _auto_res = _select_poly_order_auto(
            u_s2x, u_s2y, u_v2x, u_v2y, opd_residual,
            order_min=_MZ_POLY_AUTO_MIN, order_max=_MZ_POLY_AUTO_MAX,
            target_waves=_MZ_POLY_AUTO_TARGET, rtol=_MZ_POLY_AUTO_RTOL)
        _progress('fit', 0.22,
                  f"poly_order='auto' -> {poly_order} "
                  f"(held-out OPD fit RMS {_auto_res:.2e} waves)")

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

    # The fit-residual RMS diagnostics are only ever read
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
        # studies.  Full resolution only (output_subsample forced to 1).
        roi_cx, roi_cy, roi_hw = float(roi[0]), float(roi[1]), float(roi[2])
        output_subsample = 1
        N_out_coarse = max(1, int(round(2.0 * roi_hw / dx)))
        _ax = (np.arange(N_out_coarse) - N_out_coarse / 2) * dx
        out_axis_x = _ax + roi_cx
        out_axis_y = _ax + roi_cy
        # E-H5 (audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25): this used to be
        # an unconditional ``normalize_output = 'none'`` -- a SILENT overwrite
        # of the caller's request.  With the default normalize_output='power'
        # a roi= call returned a patch ~1e8 below the scale the very same call
        # returns on the full grid (measured median |roi|/|full patch| =
        # 0.000000), with no warning anywhere.  Split by whether the requested
        # scale is computable from the window alone:
        #
        #  * scalar factor -> window-independent, so APPLY it (Step 6 below).
        #    It was silently dropped too (measured ratio 0.500 for
        #    normalize_output=2.0); that is a plain bug and is now fixed.
        #  * 'none' -> nothing to do.
        #  * 'power' / 'peak' -> genuinely NOT computable here: both are
        #    global reductions (sum |E_out|^2 / max |E_out|) over the FULL
        #    output grid, which the ROI path by construction never evaluates
        #    -- that is the entire point of the O(roi_n^2) window.  Keep the
        #    raw scale (which is what preserves the documented "identical to
        #    the ROI slice of the full-grid field" contract against a
        #    normalize_output='none' full run) but say so OUT LOUD, and
        #    document it on ``normalize_output`` in the docstring.  Raising
        #    was considered and rejected: 'power' is the default, so raising
        #    would make every existing roi= call fail.
        #
        # An unrecognised normalize_output now also reaches Step 6's
        # ValueError instead of being swallowed by the overwrite.
        if normalize_output in ('power', 'peak'):
            import warnings
            warnings.warn(
                f"apply_real_lens_maslov: roi= cannot honour "
                f"normalize_output={normalize_output!r} -- that scale is a "
                f"global reduction over the FULL output grid, and the ROI "
                f"path evaluates only the window.  Returning the RAW field "
                f"(the normalize_output='none' scale), which for a typical "
                f"chart is many orders of magnitude below the "
                f"power-normalised full-grid field.  Pass "
                f"normalize_output='none' (explicit -- silences this "
                f"warning), or an explicit scalar factor, or drop roi= for a "
                f"normalised full grid.",
                UserWarning, stacklevel=2)
            normalize_output = 'none'
        _roi_active = True
    s2x_grid, s2y_grid = np.meshgrid(out_axis_x, out_axis_y, indexing='xy')

    u_s2x_out = (s2x_grid - s2x_c) / s2x_h
    u_s2y_out = (s2y_grid - s2y_c) / s2y_h
    inbox = (np.abs(u_s2x_out) <= 1.0) & (np.abs(u_s2y_out) <= 1.0)

    # v5.21: integration_method='auto' -- resolve to a concrete integrator from
    # the fitted chart's v2-oscillation count.  Uniform 'quadrature' is exact
    # AND caustic-safe (its integrand amplitude is finite through focus) but
    # costs O(N^2 * n_v2); the asymptotic evaluators are 77-386x faster but are
    # singular AT a caustic and only accurate when the integrand is oscillatory
    # (the saddle dominates).  So: use quadrature when it is well-resolved
    # (need n_v2 <= _N_V2_AUTO_MAX -- this covers low-oscillation AND
    # near-caustic charts, which are low-v2-oscillation and thus stay on the
    # safe path), and switch to the asymptotic only when quadrature would need
    # MORE samples than the cap -- with a warning, because that regime is not
    # automatically one where the saddle approximation holds.
    if integration_method == 'auto':
        _osc_a = _v2_oscillation_bound(mi, coef_opd)
        _need_a = int(np.ceil(4.0 * _osc_a)) + 1
        # S2 (audit): the asymptotic fallback is 'stationary_phase', not
        # 'local_quadrature'.  Measured on the auditor's synthetic quadratic
        # charts (where the exact value is closed-form) BOTH are now exact;
        # on a real f = 6 mm singlet scored against a converged uniform
        # quadrature, stationary_phase is relL2 8.4e-01 against
        # local_quadrature's 2.6e+00 at its shipped defaults -- the local
        # window reaches into pupil zones where the order-4 fit is
        # extrapolating.  stationary_phase is also the cheaper of the two.
        integration_method = ('stationary_phase' if _need_a > _N_V2_AUTO_MAX
                              else 'quadrature')
        _progress('integrate', 0.595,
                  f"auto -> {integration_method} (need n_v2~{_need_a})")
        if integration_method != 'quadrature':
            import warnings  # function-local, matching this driver
            warnings.warn(
                f"apply_real_lens_maslov: integration_method='auto' resolved "
                f"to 'stationary_phase' because uniform quadrature would need "
                f"n_v2 ~ {_need_a} samples (cap {_N_V2_AUTO_MAX}).  The "
                f"leading-order saddle evaluation is only accurate where the "
                f"integrand really is oscillatory -- near the lens EXIT plane "
                f"of a focusing system it is not (the v2-Hessian and the "
                f"Jacobian both collapse there) and it can be O(1) wrong.  "
                f"For an observation plane at or near the exit vertex pass "
                f"integration_method='quadrature' with an explicit n_v2, or "
                f"use apply_real_lens_traced / apply_real_lens_fga; for a "
                f"plane near focus pass output_plane_distance= and keep this "
                f"evaluator.",
                RuntimeWarning, stacklevel=2)

    # A1 (v5.20): auto-resolve the uniform-quadrature v2 sampling when the
    # caller left n_v2 unset.  n_v2 drives ONLY integration_method='quadrature'
    # (the local_quadrature / stationary_phase paths window around the per-pixel
    # saddle via the v2-box half-width v2x_h/v2y_h, and never read this uniform
    # sample count), so size it from the same _v2_osc estimate the N2 guard
    # below uses and leave the asymptotic paths at the floor.  Both ``mi`` and
    # ``coef_opd`` are already fitted at this point.
    if n_v2 is None:
        if integration_method == 'quadrature':
            _v2_osc_auto = _v2_oscillation_bound(mi, coef_opd)
            n_v2 = int(np.clip(int(np.ceil(4.0 * _v2_osc_auto)) + 1,
                               _N_V2_AUTO_MIN, _N_V2_AUTO_MAX))
        else:
            n_v2 = _N_V2_AUTO_MIN

    u_v2x_samples = np.linspace(-1.0, 1.0, n_v2)
    u_v2y_samples = np.linspace(-1.0, 1.0, n_v2)
    du = u_v2x_samples[1] - u_v2x_samples[0]

    def tukey(n, alpha=0.2):                 # shared window (S2-14)
        return _tukey_taper(np.linspace(-1, 1, n), alpha)
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
        # Dtype-aware out-of-bounds sentinel so
        # a complex64 E_in stays complex64 through the bilinear sample
        # (was silently upcasting via the ``0.0 + 0.0j`` complex128
        # literal).  Matches the v4.13.2 canonical pattern.
        val = np.where(ok, val, np.zeros((), dtype=val.dtype))
        return val

    # -----------------------------------------------------------------
    # Step 4: Integrate
    # -----------------------------------------------------------------
    if integration_method not in ('quadrature', 'stationary_phase',
                                    'local_quadrature', 'levin'):
        # E-M14 (audit): 'auto' -- the DEFAULT -- was missing from this
        # message, so a typo'd method was told to use a set that excludes the
        # value it already had.  ('auto' never reaches here: it is resolved to
        # a concrete integrator above.)
        raise ValueError(
            f"integration_method must be one of 'auto' (default), "
            f"'quadrature', 'stationary_phase', 'local_quadrature', 'levin', "
            f"got {integration_method!r}")

    # -----------------------------------------------------------------
    # S6: the input field's local wavevector in the asymptotic saddle
    # -----------------------------------------------------------------
    # The two asymptotic evaluators expand the v2 integral about the saddle of
    # its TOTAL phase, arg E_in(s1(v2)) + 2 pi OPD_waves.  Keeping only the OPD
    # half makes the saddle condition grad_v2 OPD = 0, and the symplectic
    # identity dOPD/dv2 = -n1 (v1 . ds1/dv2) (measured closing to 5.8e-7
    # relative on a real singlet chart) makes THAT the v1 = 0 on-axis
    # collimated launch ray, at every pixel and for every input -- while the
    # chart is deliberately sized to cover a diverging / tilted one
    # (na_proxy = na_lens + na_input).  So the input's local wavevector
    # k1 = (1/k0) grad arg E_in is fitted here over the SAME chart coordinates
    # as the OPD and s1 fits and handed to the integrators, which put
    # (k1 . ds1/dv2) / lambda into the saddle gradient and its derivative into
    # the Hessian: the saddle condition becomes (v1_in - v1) . ds1/dv2 = 0.
    # 'quadrature' and 'levin' integrate the true integrand pointwise and have
    # no saddle, so they never build this fit.
    #
    # The MEASUREMENT of "is the input flat" is the WAVEFRONT NA (the spread of
    # the local wavevector), NOT the second moment of |FFT(E_in)|^2 that sizes
    # the pupil chart (VERIFY-A4): a COLLIMATED beam of finite width has a real
    # angular spectrum -- a Gaussian of waist w spreads by lambda/(pi w),
    # measured 3-sigma NA 3.54e-03 / 1.10e-03 at waist 0.25 / 0.8 mm
    # (lambda = 1.31 um) -- while its geometric launch direction is v1 = 0
    # everywhere, where ``_wavefront_na`` returns EXACTLY 0 and the OPD-only
    # saddle is already the right one.  The decision below reads that spread
    # over the TRACED RAY entrance points, weighted by the input intensity
    # there: that is the part of the wavefront the integral actually samples,
    # and it keeps the denominator of the fit-quality ratio away from zero.
    _k1_fit = None
    _na_wf = 0.0
    _k1_na_rays = 0.0
    _k1_res_rel = 0.0
    _s1_res_rel = 0.0
    _s6_mode = (_S6_INPUT_WAVEVECTOR_SADDLE if input_wavevector_saddle is None
                else input_wavevector_saddle)
    _asymptotic = integration_method in ('stationary_phase',
                                         'local_quadrature')
    if _asymptotic and not collimated_input:
        _intensity = np.abs(E_in) ** 2
        _k1x_grid, _k1y_grid = _local_direction_cosines(
            E_in, dx, dy, wavelength)
        _na_wf = _wavefront_na_from_cosines(_intensity, _k1x_grid, _k1y_grid)
    if _asymptotic and not collimated_input and _s6_mode is not False:
        _k1x_rays = _sample_real_bilinear(_k1x_grid, N, dx, dy,
                                          s1x_live, s1y_live)
        _k1y_rays = _sample_real_bilinear(_k1y_grid, N, dx, dy,
                                          s1x_live, s1y_live)
        _w_rays = _sample_real_bilinear(_intensity, N, dx, dy,
                                        s1x_live, s1y_live)
        _w_tot = float(_w_rays.sum())
        _k1_pow = (float((_w_rays * (_k1x_rays ** 2
                                     + _k1y_rays ** 2)).sum()) / _w_tot
                   if _w_tot > 0.0 else 0.0)
        _k1_na_rays = 3.0 * float(np.sqrt(max(_k1_pow, 0.0)))
        _engage = (_k1_na_rays > _SADDLE_FLAT_INPUT_NA if _s6_mode is None
                   else _k1_na_rays > 0.0)
        if _engage:
            # One more least-squares solve against the SAME design matrix.  It
            # is a SEPARATE solve rather than two more columns on the stacked
            # OPD/s1x/s1y right-hand side (~20 ms on a 40k-ray chart, measured)
            # because widening a GEMM's right-hand side is entitled to move the
            # existing columns in the last bits, and the price of that is that
            # no run of the OTHER three integrators would be reproducible
            # across this release.  This way every call that does not engage
            # the S6 saddle is bit-for-bit the OPD-only answer.
            _k1_rhs = np.column_stack([_k1x_rays, _k1y_rays])
            _coef_k1 = _solve_fit(A, _k1_rhs)
            _k1_res = A @ _coef_k1 - _k1_rhs
            _k1_res_pow = float((_w_rays * (_k1_res[:, 0] ** 2
                                            + _k1_res[:, 1] ** 2)).sum())
            _k1_res_rel = (float(np.sqrt(_k1_res_pow / _w_tot / _k1_pow))
                           if (_w_tot > 0.0 and _k1_pow > 0.0) else np.inf)
            # FALLBACK CRITERION, part 1 of 2.  The saddle correction is only
            # as good as the chart's ability to REPRESENT the input's local
            # wavevector: a speckled input, or one cut by a hard-edged
            # aperture (where ``_local_direction_cosines`` reports 0 in the
            # dark and the true wavefront in the light), is not a
            # degree-``poly_order`` tensor polynomial of the chart coordinates
            # in any useful sense, and a saddle placed by a bad fit is a
            # different wrong ray, not a better one.  Measure it: the
            # intensity-weighted RMS fit residual of (k1x, k1y) as a fraction
            # of their intensity-weighted RMS.
            #
            # Part 2: the SAME contraction reads the ENTRANCE-COORDINATE fit,
            # because the term is k1 . ds1/dv2 -- so the chart has to carry
            # BOTH factors.  ``_s1_res_rel`` is the entrance-coordinate fit's
            # own relative RMS residual over the traced rays; the constant it
            # is compared against carries the ladder.
            _s1_res = A @ np.column_stack([coef_s1x, coef_s1y]) \
                - np.column_stack([s1x_live, s1y_live])
            _s1_pow = float(np.mean(s1x_live ** 2 + s1y_live ** 2))
            _s1_res_rel = (float(np.sqrt(
                float(np.mean(_s1_res[:, 0] ** 2 + _s1_res[:, 1] ** 2))
                / _s1_pow)) if _s1_pow > 0.0 else np.inf)
            # Below both bars, use the fit; above either, keep the OPD-only
            # saddle and say so, which is the honest answer when the term the
            # saddle would be moved by is not a chart-representable field.
            if ((_k1_res_rel <= _K1_FIT_RESIDUAL_MAX
                 and _s1_res_rel <= _S1_FIT_RESIDUAL_MAX) or _s6_mode):
                _k1_fit = (coef_s1x, coef_s1y, _coef_k1[:, 0], _coef_k1[:, 1],
                           1.0 / float(wavelength))
            # The ``k1 fit residual <x> (engaged|refused)`` tail is parsed by
            # tests/unit/test_audit2609_b1_maslov_input_wavevector.py; new
            # fields go BEFORE it and do not spell the word "residual".
            _progress('integrate', 0.598,
                      f'S6 input-wavevector saddle: ray NA {_k1_na_rays:.4f}, '
                      f's1 chart fit {_s1_res_rel:.2e}, '
                      f'k1 fit residual {_k1_res_rel:.2e} '
                      f'({"engaged" if _k1_fit is not None else "refused"})')
    if (_asymptotic and not collimated_input and _k1_fit is None
            and (_na_wf > _SADDLE_FLAT_INPUT_NA
                 or _k1_na_rays > _SADDLE_FLAT_INPUT_NA)):
        import warnings  # function-local, matching this driver
        if _s6_mode is False:
            _why = ("it was asked for the OPD-only saddle "
                    + ("(input_wavevector_saddle=False)"
                       if input_wavevector_saddle is not None
                       else "(_S6_INPUT_WAVEVECTOR_SADDLE = False)")
                    + ", so the saddle of the OPD alone is being solved")
        elif _k1_na_rays <= _SADDLE_FLAT_INPUT_NA:
            _why = (f"the wavefront is flat ACROSS THE TRACED APERTURE "
                    f"(ray-sampled NA {_k1_na_rays:.4f}), so nothing was "
                    f"fitted and the saddle of the OPD alone is solved")
        elif _s1_res_rel > _S1_FIT_RESIDUAL_MAX:
            _why = (f"the CHART cannot carry the other factor of the S6 term "
                    f"-- the order-{poly_order} fit to the entrance "
                    f"coordinates s1(s2, v2), which k1 is contracted "
                    f"against, has relative RMS residual {_s1_res_rel:.2e}, "
                    f"above the {_S1_FIT_RESIDUAL_MAX:g} bar, so the fitted "
                    f"saddle would be placed on a ray the chart has "
                    f"misplaced (a larger poly_order, or an explicit "
                    f"input_na that stops na_proxy over-sizing the pupil "
                    f"box, fixes this) -- so the saddle of the OPD alone is "
                    f"being solved")
        else:
            _why = (f"its local wavevector could not be represented on the "
                    f"chart -- the intensity-weighted RMS residual of the "
                    f"order-{poly_order} fit to (k1x, k1y) is "
                    f"{_k1_res_rel:.2f} of their own RMS, above the "
                    f"{_K1_FIT_RESIDUAL_MAX:g} bar (speckle and a hard-edged "
                    f"aperture do this; so does an input wavefront the fit "
                    f"order cannot carry, which a larger poly_order fixes) "
                    f"-- so the saddle of the OPD alone is being solved")
        warnings.warn(
            f"apply_real_lens_maslov: integration_method="
            f"{integration_method!r} expands about the stationary point of "
            f"the integrand's total phase, but {_why}, which selects the "
            f"v1 = 0 (collimated) launch ray at every pixel.  The input's "
            f"measured WAVEFRONT spread is NA ~ {_na_wf:.4f} over the grid "
            f"and {_k1_na_rays:.4f} over the traced aperture "
            f"(> {_SADDLE_FLAT_INPUT_NA:g}; its angular spectrum spans "
            f"{_na_meas:.4f}, which for a collimated beam is just "
            f"diffraction and is NOT what this gate tests), so the result is "
            f"a leading-order expansion about the wrong ray.  Use "
            f"integration_method='quadrature' (exact) or 'levin' "
            f"(caustic-uniform) for a diverging / converging / tilted input "
            f"whose wavefront this chart cannot fit, or pass "
            f"collimated_input=True if the input really is flat and the "
            f"measured spread is aperture-edge content.",
            RuntimeWarning, stacklevel=2)

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

    if integration_method == 'levin':
        E_out_coarse = _integrate_levin(
            coef_opd, coef_s1x, coef_s1y,
            K1_arr, K2_arr, K3_arr, K4_arr,
            poly_order, N_out_coarse,
            u_s2x_out, u_s2y_out, inbox_flat,
            v2x_h, v2y_h,
            sample_E_bilinear,
            levin_tol, _progress, verbose,
            out_dtype=E_in.dtype,
            lin_v3=_lin_v3, lin_v4=_lin_v4,
        )
    elif integration_method == 'stationary_phase':
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
                k1_fit=_k1_fit,
            ))
        else:
            E_out_coarse = _integrate_stationary_phase(
                coef_opd, coef_s1x, coef_s1y, mi,
                K1_arr, K2_arr, K3_arr, K4_arr,
                poly_order, N_out_coarse,
                u_s2x_out, u_s2y_out, inbox_flat,
                v2x_h, v2y_h,
                sample_E_bilinear,
                stationary_newton_iter, stationary_newton_tol,
                _progress, verbose,
                out_dtype=E_in.dtype,
                lin_v3=_lin_v3, lin_v4=_lin_v4, k1_fit=_k1_fit,
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
                k1_fit=_k1_fit,
            ))
        else:
            E_out_coarse = _integrate_local_quadrature(
                coef_opd, coef_s1x, coef_s1y,
                K1_arr, K2_arr, K3_arr, K4_arr,
                poly_order, N_out_coarse,
                u_s2x_out, u_s2y_out, inbox_flat,
                v2x_h, v2y_h,
                sample_E_bilinear,
                stationary_newton_iter, stationary_newton_tol,
                local_n_samples, local_window_sigma,
                _progress, verbose,
                out_dtype=E_in.dtype,
                lin_v3=_lin_v3, lin_v4=_lin_v4, k1_fit=_k1_fit,
            )
    else:
        # N2 (audit): estimate the v2 oscillation count of the integrand
        # phase 2*pi*OPD(s2,v2) from the fitted coefficients -- see
        # :func:`_v2_oscillation_bound` for why it is the TOTAL VARIATION
        # (sum |c_k| * max(k3, k4)) and not the excursion (sum |c_k|).
        # Uniform n_v2-point quadrature needs a few samples per cycle; when
        # under-resolved the result speckles regardless of grid/memory (no
        # output-resolution fix helps) -- warn and point at the asymptotic
        # evaluators, which are the correct choice at production NA.
        _v2_osc = _v2_oscillation_bound(mi, coef_opd)
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
                K3_arr, K4_arr,
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
        # E-H1 (audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25): the coarse
        # output grid built at Step 3 is a STRIDE subsample of the standard
        # lattice -- coarse sample j sits at physical
        # ``(j - N_c/2) * sub * dx``, i.e. at FINE index
        # ``j*sub + (N - N_c*sub)/2`` (the offset is 0 whenever sub divides
        # N).  So the exact fine->coarse map is
        #     j = (i - off) / sub,     off = (N - N_c*sub) / 2
        # ``scipy.ndimage.zoom`` (grid_mode=False) instead maps
        # ``i -> i*(N_c-1)/(N-1)``: EDGE-anchored, which both displaces the
        # content and mis-magnifies it by ``(N-1)/(sub*(N_c-1))``.  Measured
        # on a rotationally-symmetric on-axis singlet (true SPATIAL intensity
        # centroid = 0, N=96): +0.5018 / +1.5359 / +3.7011 fine px at
        # sub=2/4/8 against the closed form +0.5106 / +1.5652 / +3.8182, with
        # a x1.010073 second-moment width (magnification) error at sub=2;
        # post-fix +0.0024 / +0.0191 / +0.1061 px and x1.000502.  The coarse
        # samples also now come back UNCHANGED at their own fine indices
        # (i = sub*j, to 6e-16 relative; the zoom path was off by 2.7e-2 of
        # the field peak there, 34% on a strong singlet).  A large real output
        # tilt is rescaled by the same factor: a flat prism's recovered
        # spectral tilt reads +4.61% at sub=2 and +23.02% at sub=6 through
        # ``zoom`` against 0.02% / 0.16% on the true lattice.  This is the
        # ``ii*Ns/N`` traced-upsample bug fixed at 0a743a6, and takes the same
        # remedy: resample on the true lattice.
        #
        # ``affine_transform`` with a diagonal
        # ``1/sub`` matrix + ``-off/sub`` shift is the same NI_ZoomShift
        # spline kernel ``zoom`` uses (verified bit-identical to
        # ``map_coordinates`` on these coordinates, and identical to
        # ``zoom`` to ~1e-14 when evaluated at *zoom's* coordinates -- so the
        # whole behaviour change here is the lattice), and unlike
        # ``map_coordinates`` it needs no O(N^2) coordinate stack (which
        # would be 17 GB at N=32768).  Output shape is exactly (N, N) by
        # construction.  There is no CuPy twin to keep in lockstep here: all
        # three GPU integrators ``_cp.asnumpy`` their coarse result above, so
        # this host-side upsample is the single shared path.
        from scipy.ndimage import affine_transform
        _up_mat = np.array([1.0 / output_subsample, 1.0 / output_subsample])
        _up_off = -((N - N_out_coarse * output_subsample) / 2.0
                    / output_subsample)

        def _upsample(_a):
            return affine_transform(_a, _up_mat, offset=_up_off,
                                    output_shape=(N, N), order=3,
                                    mode='nearest')
        amp = np.abs(E_out_coarse)
        amp_z = _upsample(amp)
        # Phase upsampling: interpolate the COMPLEX ``exp(i*phase)``
        # directly via cubic zoom of its real and imaginary parts, then
        # take ``angle()``.  That avoids any 2-D phase-unwrap step -- a
        # line-by-line ``np.unwrap`` is fragile near caustics / focal
        # saddles where the phase wraps along both axes, and the
        # cubic-interpolated phase then carries ~4% RMS errors from
        # line-mismatched seams.  The cost is that this form is
        # well-behaved only when the local phase variation between
        # adjacent coarse pixels is < pi -- the same condition the
        # line-unwrap silently relied on.  For Maslov outputs that
        # satisfy it (typical refractive systems with
        # output_subsample <= 8) the two agree to ~0.3% RMS.
        phase_c = np.angle(E_out_coarse)
        cos_z = _upsample(np.cos(phase_c))
        sin_z = _upsample(np.sin(phase_c))
        # v4.14.0: ``1j * float64`` returns complex128; cast back to
        # E_in.dtype so complex64 inputs are preserved.  (The old zoom()
        # path needed a shape re-fit here because ``zoom`` sizes its output
        # by rounding; ``affine_transform`` is handed ``output_shape=(N, N)``
        # explicitly, so the re-fit was provably dead and is gone.)
        E_out = (amp_z * cos_z + 1j * (amp_z * sin_z)).astype(E_in.dtype)
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
    #    field first.  The fine-pixel coordinate is reproduced by resampling
    #    the coarse output axis with the SAME lattice map as the content, so
    #    the tilt lands exactly where the resampled content lives
    #    (convention-independent).
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
            # subsample>1 is non-ROI.  v5.20: resample BOTH axes
            # independently -- for anamorphic pixels out_axis_x (dx) !=
            # out_axis_y (dy), so the old ``out_axis_fy = out_axis_fx`` would
            # place the y-tilt at the wrong pitch.  (Square pixels: the two
            # are identical.)
            # E-H1: use the SAME lattice map as the Step-5 content upsample
            # above (``affine_transform``, 1/sub + -off/sub), not the
            # edge-anchored ``zoom``, so the tilt still lands exactly where
            # the resampled content lives.  On the true lattice this
            # reproduces the standard fine axis ``(arange(N) - N/2)*dx`` to
            # ~1e-16 over the interpolated interior, and follows the
            # ``mode='nearest'`` clamp over the (sub-1)-wide extrapolated
            # tail -- exactly where the content is clamped too.
            from scipy.ndimage import affine_transform as _affine1d

            def _resample_axis(_ax):
                return _affine1d(_ax, _up_mat[:1], offset=_up_off,
                                 output_shape=(N,), order=1, mode='nearest')

            out_axis_fx = _resample_axis(out_axis_x)
            out_axis_fy = _resample_axis(out_axis_y)
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

    # S4 (audit): the d = 2 Van Vleck prefactor k/(2 pi i) = 1/(i lambda).
    # It closes the ABSOLUTE scale of the canonical integral -- without it the
    # returned field is off by 1/(i*lambda) (and, before the sqrt fix in the
    # integrands, by a further factor of |det ds1/dv2|).  A constant, so it
    # commutes with the upsample and the linear-phase re-application above;
    # applied here so that normalize_output='none' -- which every ``roi=`` run
    # is forced onto -- is physically scaled.
    E_out = (E_out * _maslov_kernel_prefactor(wavelength)).astype(E_in.dtype)

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
    gap for the TRANSVERSE Jones field: per-surface diattenuation and retardance
    carried through a caustic, which GBD's paraxial beamlets cannot do with
    caustic fidelity.  It is NOT a full vector focus -- there is no ``E_z`` and
    no exit-frame transport of the Jones vector (audit S10; see the Notes) --
    so do not read it as a "polarization-resolved study through a focus" at
    high NA.

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

    **``normalize_output`` is applied ONCE, to the pair** (audit S10, third
    sub-item).  Before v5.46 the requested mode was forwarded to the two
    scalar legs, which each normalised INDEPENDENTLY -- so ``'power'`` (the
    scalar default, hence the default here) rescaled ``E_x`` and ``E_y`` by
    different factors and forced the output polarization ratio back to the
    post-Fresnel INPUT ratio, deleting exactly the diattenuation this wrapper
    exists to compute.  Both legs now run at ``normalize_output='none'`` and a
    single scale is applied to the pair, as
    :func:`~lumenairy.propagators.fga.apply_real_lens_fga_vector` already does:

    * ``'power'`` (default) -- one factor so ``sum(|E_x|^2 + |E_y|^2)`` equals
      the POST-FRESNEL input pair's total power, i.e. the scalar
      ``normalize_output='power'`` contract applied ONCE to the pair.  The
      surface Fresnel transmission therefore stays in the absolute scale
      (``T1*T2`` on a two-surface singlet), unlike
      :func:`~lumenairy.propagators.fga.apply_real_lens_fga_vector`, which
      normalises to the RAW input pair under a lossless assumption;
    * ``'peak'`` -- one factor so ``max sqrt(|E_x|^2 + |E_y|^2)`` matches the
      post-Fresnel input pair's;
    * ``'none'`` -- no scale at all (the Van Vleck-normalised absolute field).

    ``P_x / P_y`` is therefore identical under all three modes.

    **Longitudinal field and frame transport are still missing** (audit S10,
    remaining sub-item): the Jones vector is not parallel-transported /
    Richards-Wolf-rotated into the exit-ray frame and there is no ``E_z``, so
    at high NA this wrapper is a TRANSVERSE model with per-surface Fresnel
    weights -- not a full vector focus.  For the longitudinal component use
    :func:`~lumenairy.propagators.fga.apply_real_lens_fga_vector`
    (``return_longitudinal=True``) or GBD's
    ``reconstruct_vector_field_with_ez``.
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
    _norm = maslov_kwargs.pop('normalize_output', 'power')
    if _norm not in ('power', 'peak', 'none'):
        raise ValueError(
            f"apply_real_lens_maslov_vector: normalize_output must be one of "
            f"'power', 'peak', 'none'; got {_norm!r}.")
    ix = np.arange(Nx)
    iy = np.arange(Ny)
    Ix, Iy = np.meshgrid(ix, iy, indexing='xy')
    xb = (Ix.ravel() - Nx / 2.0) * dx
    yb = (Iy.ravel() - Ny / 2.0) * float(dy)
    # S10 (audit): launch the polarization-ray-tracing base rays along the
    # INPUT FIELD's own local wavevector rather than axially.  The GBD helper
    # documents ``ux = uy = 0`` as its convention because the GBD vector
    # driver hands it a collimated frame; this wrapper is offered for
    # diverging / converging / tilted inputs, where the s/p split and the
    # per-surface incidence angles genuinely depend on the incoming direction.
    # The estimator is the same per-pixel conjugate-product local wavevector
    # the FGA router uses (``_global_mean_tilt`` / ``_tilt_dispersion``), so
    # the two families read "which way is this pixel going" identically, and
    # it returns EXACTLY zero on a real, non-negative (flat-phase) input
    # -- the collimated case is bit-identical to an axial launch.
    ux, uy = _input_direction_cosines(E_vec, dx, float(dy), wavelength)
    P, _alive = _fresnel_jones_matrix_per_beamlet(
        xb, yb, ux.ravel(), uy.ravel(), prescription, wavelength)
    ExS = np.asarray(E_vec[0]).ravel()
    EyS = np.asarray(E_vec[1]).ravel()
    ExM = (P[:, 0, 0] * ExS + P[:, 0, 1] * EyS).reshape(Ny, Nx)
    EyM = (P[:, 1, 0] * ExS + P[:, 1, 1] * EyS).reshape(Ny, Nx)
    out_x = apply_real_lens_maslov(
        ExM, prescription=prescription, wavelength=wavelength, dx=dx, dy=dy,
        normalize_output='none', **maslov_kwargs)
    out_y = apply_real_lens_maslov(
        EyM, prescription=prescription, wavelength=wavelength, dx=dx, dy=dy,
        normalize_output='none', **maslov_kwargs)
    from ..backend.array import array_namespace
    xp = array_namespace(out_x)
    # ONE joint scale for the pair -- see the Notes above.  The reference is
    # the POST-FRESNEL pair (ExM, EyM), i.e. the field the two scalar legs
    # were actually handed: that is the literal joint form of the scalar
    # ``normalize_output`` contract ("sum |E_out|^2 == sum |E_in|^2" applied
    # once to the pair) and it keeps the surface Fresnel TRANSMISSION in the
    # absolute scale, where dividing by the RAW input power -- the FGA peer's
    # lossless convention -- would normalise it away.  Only the s/p ratio is
    # common to both conventions; this one keeps T1*T2 as well.
    scale = 1.0
    if _norm == 'power':
        p_in = float(np.sum(np.abs(ExM) ** 2 + np.abs(EyM) ** 2))
        p_out = float(xp.sum(xp.abs(out_x) ** 2 + xp.abs(out_y) ** 2))
        if p_out > 0.0 and np.isfinite(p_out) and p_in > 0.0:
            scale = float(np.sqrt(p_in / p_out))
    elif _norm == 'peak':
        a_in = float(np.max(np.abs(ExM) ** 2 + np.abs(EyM) ** 2)) ** 0.5
        a_out = float(xp.max(xp.abs(out_x) ** 2 + xp.abs(out_y) ** 2)) ** 0.5
        if a_out > 0.0 and np.isfinite(a_out) and a_in > 0.0:
            scale = a_in / a_out
    if scale != 1.0:
        out_x = (out_x * scale).astype(out_x.dtype)
        out_y = (out_y * scale).astype(out_y.dtype)
    return xp.stack([out_x, out_y], axis=0)


def _local_direction_cosines(E, dx, dy, wavelength):
    """Per-pixel local direction cosines ``(ux, uy)`` of a 2-D scalar field.

    ``u = (1/k0) grad(arg E)`` by the conjugate-product forward difference
    ``arg(E[i+1] conj(E[i])) / (k0 dx)``: no unwrap, and it wraps only at the
    grid's own Nyquist angle ``lambda / (2 dx)``, the largest direction the
    grid can carry.  The last column / row repeats its neighbour so an edge
    pixel is not reported as axial, and a pixel with no amplitude is set to
    zero.  A real, non-negative field gives EXACTLY ``(0, 0)``.
    """
    E = np.asarray(E)
    k0 = 2.0 * np.pi / float(wavelength)
    ux = np.zeros(E.shape, dtype=np.float64)
    uy = np.zeros(E.shape, dtype=np.float64)
    if E.shape[-1] > 1:
        ux[:, :-1] = np.angle(E[:, 1:] * np.conj(E[:, :-1])) / (k0 * float(dx))
        ux[:, -1] = ux[:, -2]
    if E.shape[-2] > 1:
        uy[:-1, :] = np.angle(E[1:, :] * np.conj(E[:-1, :])) / (k0 * float(dy))
        uy[-1, :] = uy[-2, :]
    dead = np.abs(E) <= 0.0
    ux[dead] = 0.0
    uy[dead] = 0.0
    return ux, uy


def _wavefront_na(E, dx, dy, wavelength):
    """3-sigma intensity-weighted spread of the input's LOCAL WAVEVECTOR.

    ``3 * sqrt(<ux^2 + uy^2>)`` from :func:`_local_direction_cosines`.  This
    is the quantity the S6 saddle warning has to test, and it is NOT the
    second moment of ``|FFT(E)|^2``: a COLLIMATED beam of finite width has a
    genuine angular spectrum (a Gaussian of waist ``w`` spreads by
    ``lambda / (pi w)``) while its geometric launch direction is ``v1 = 0``
    everywhere, which is exactly the case the OPD-only saddle gets right.

    MEASURED (lambda = 1.31 um, dx = 10 um) on flat-phase Gaussians of waist
    0.25 / 0.8 / 2 / 4 mm: the FFT second moment gives 3-sigma NA
    3.54e-03 / 1.10e-03 / 4.2e-04 / 2.1e-04 -- the first two ABOVE
    ``_SADDLE_FLAT_INPUT_NA``, i.e. a false alarm on any collimated beam
    narrower than ~1 mm -- while this estimator gives EXACTLY 0.000e+00 at
    every width.  On inputs that really are non-flat the two agree: tilt
    0.002 / 0.01 rad -> 6.00e-03 / 3.00e-02 here against 6.11e-03 /
    3.00e-02 spectrally; diverging f = -20 mm -> 8.42e-02 both; converging
    f = +50 mm -> 3.37e-02 both.
    """
    ux, uy = _local_direction_cosines(E, dx, dy, wavelength)
    return _wavefront_na_from_cosines(np.abs(np.asarray(E)) ** 2, ux, uy)


def _wavefront_na_from_cosines(intensity, ux, uy):
    """:func:`_wavefront_na` given the intensity and the cosines already in
    hand -- the form the driver uses, which needs the cosine GRIDS themselves
    to sample the input's local wavevector at the traced ray entrance points
    (S6) and must not pay for a second ``np.angle`` pass over the field."""
    tot = float(np.asarray(intensity).sum())
    if not np.isfinite(tot) or tot <= 0.0:
        return 0.0
    return 3.0 * float(np.sqrt(
        float((intensity * (ux * ux + uy * uy)).sum()) / tot))


def _sample_real_bilinear(G, N, dx, dy, xq, yq):
    """Bilinear sample of a REAL (N, N) grid at physical ``(xq, yq)``.

    The grid origin convention is the driver's own
    (``x[0] = -(N/2) dx``), matching ``sample_E_bilinear``.  Query points
    outside the grid are CLAMPED to the edge cell rather than zeroed: this
    samples the input's local wavevector at ray entrance points, and a zero
    there would assert "this ray launches on axis", which is a statement about
    the field the grid does not make.  The aperture-vs-grid mismatch that puts
    a ray outside is announced separately by
    :func:`_warn_if_aperture_exceeds_grid`.
    """
    fx = (np.asarray(xq) + (N / 2) * dx) / dx
    fy = (np.asarray(yq) + (N / 2) * dy) / dy
    ix = np.clip(np.floor(fx).astype(np.int64), 0, N - 2)
    iy = np.clip(np.floor(fy).astype(np.int64), 0, N - 2)
    wx = np.clip(fx - ix, 0.0, 1.0)
    wy = np.clip(fy - iy, 0.0, 1.0)
    return ((1.0 - wx) * (1.0 - wy) * G[iy, ix]
            + wx * (1.0 - wy) * G[iy, ix + 1]
            + (1.0 - wx) * wy * G[iy + 1, ix]
            + wx * wy * G[iy + 1, ix + 1])


def _input_direction_cosines(E_vec, dx, dy, wavelength):
    """Per-pixel local direction cosines ``(ux, uy)`` of a Jones field.

    ``u = (1/k0) grad(arg E)`` evaluated by the conjugate-product forward
    difference ``arg(E[i+1] conj(E[i])) / (k0 dx)`` -- the same estimator
    :func:`lumenairy.propagators.fga._global_mean_tilt` uses, so the Maslov
    and FGA vector paths agree on the launch direction by construction.  It
    needs no unwrap and wraps only at the grid's own Nyquist angle
    ``lambda / (2 dx)``, which is the largest direction the grid can carry.

    The phase is read from the higher-power Jones component (the two share one
    base ray per pixel, and the stronger one carries the better-conditioned
    phase), matching ``apply_real_lens_fga_vector``'s ``_rep`` choice.  The
    last column / row repeats its neighbour so an edge pixel is not reported
    as axial.  A real, non-negative input gives EXACTLY ``(0, 0)``.
    """
    Ex = np.asarray(E_vec[0])
    Ey = np.asarray(E_vec[1])
    E = Ex if float(np.sum(np.abs(Ex) ** 2)) >= float(
        np.sum(np.abs(Ey) ** 2)) else Ey
    return _local_direction_cosines(E, dx, dy, wavelength)


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
    # E-L9 (audit): K1_arr / K2_arr were dead here -- the s2 (k1, k2) part of
    # the basis reaches this integrator pre-evaluated as the Tx_1d / Ty_1d
    # Vandermondes, so only the v2 exponents are needed.  Dropped; this now
    # matches the arg list of the CuPy twin _integrate_quadrature_cupy, which
    # never took them (audit-noted NumPy/CuPy twin drift).
    K3_arr, K4_arr,
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
            abs_J_c = _van_vleck_density(np.abs(det_J_c), v2x_h, v2y_h)

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
            abs_J_c = _van_vleck_density(xp.abs(det_J_c), v2x_h, v2y_h)
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
    # E-L9 (audit): v2x_c / v2y_c were dead -- the saddle is solved in the
    # NORMALISED u_v2 coordinates and only the half-widths v2x_h / v2y_h enter
    # (the Hessian de-normalisation).  Dropped; the CuPy twin
    # _integrate_stationary_phase_cupy never took them either.
    v2x_h, v2y_h,
    sample_E_bilinear,
    newton_iter, newton_tol,
    _progress, verbose,
    out_dtype=np.complex128,
    lin_v3=0.0, lin_v4=0.0, k1_fit=None,
):
    """Leading-order stationary-phase (Gaussian-moment) evaluation.

    v4.14.0: ``out_dtype`` defaults to ``np.complex128`` for back-
    compat; callers pass ``E_in.dtype`` to preserve complex64 inputs.

    ``k1_fit`` (S6) puts the input field's own phase into the stationary
    condition and into the Hessian that sets the Gaussian-moment amplitude and
    the Maslov signature; see :func:`_input_phase_terms`.  ``None`` -- a flat
    input, or the refused fallback -- runs the pre-S6 arithmetic bit for bit.
    """
    t_int_start = time.perf_counter()
    _progress('integrate', 0.65,
              f'stationary-phase Newton ({newton_iter} max iters)')

    N_px = N_out_coarse * N_out_coarse

    u_s2x_flat = u_s2x_out.ravel()
    u_s2y_flat = u_s2y_out.ravel()

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

    def _sp_progress(it, converged, grad_mag):    # verbose banner (S2-14)
        if verbose and (it == 0 or it == newton_iter - 1 or
                        it % max(1, newton_iter // 4) == 0):
            n_conv = converged.sum()
            _progress('integrate', 0.65 + 0.15 * it / newton_iter,
                      f'Newton iter {it+1}/{newton_iter}, '
                      f'{n_conv}/{N_px} pixels converged '
                      f'(max grad {grad_mag.max():.2e})')

    # Shared CPU active-subset Newton (S2-14); banded OPD eval caps peak memory.
    u_v2x, u_v2y, converged_mask = _maslov_newton_saddle_cpu(
        _opd_and_derivs_banded, coef_opd, u_s2x_flat, u_s2y_flat, inbox_flat,
        newton_iter, newton_tol, lin_v3, lin_v4, progress=_sp_progress,
        k1_fit=k1_fit)

    _progress('integrate', 0.85, 'evaluating saddle-point formula')

    opd_star, g3, g4, H33, H34, H44 = _opd_and_derivs_banded(
        coef_opd, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)
    # N4: add the linear-in-v2 OPD contribution at the (shifted) saddle.
    opd_star = opd_star + lin_v3 * u_v2x + lin_v4 * u_v2y
    _s1x_ev = _opd_and_derivs_banded(
        coef_s1x, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)
    _s1y_ev = _opd_and_derivs_banded(
        coef_s1y, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)
    s1x_star, ds1x_du3, ds1x_du4 = _s1x_ev[0], _s1x_ev[1], _s1x_ev[2]
    s1y_star, ds1y_du3, ds1y_du4 = _s1y_ev[0], _s1y_ev[1], _s1y_ev[2]
    if k1_fit is not None:
        # S6: the Gaussian-moment amplitude and the Maslov signature both read
        # the curvature of the TOTAL stationary phase, not of the OPD alone.
        _, _, _a33, _a34, _a44 = _input_phase_terms(
            _s1x_ev, _s1y_ev,
            _opd_and_derivs_banded(k1_fit[2], u_s2x_flat, u_s2y_flat,
                                   u_v2x, u_v2y),
            _opd_and_derivs_banded(k1_fit[3], u_s2x_flat, u_s2y_flat,
                                   u_v2x, u_v2y), k1_fit[4])
        H33 = H33 + _a33
        H34 = H34 + _a34
        H44 = H44 + _a44

    det_J_norm = ds1x_du3 * ds1y_du4 - ds1x_du4 * ds1y_du3
    abs_J = _van_vleck_density(np.abs(det_J_norm), v2x_h, v2y_h)

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


def _warn_levin_over_tolerance(achieved_bounds, tolerances, n_total,
                               levin_tol) -> int:
    """Audit S2-18: warn when Levin per-pixel fallback pixels still miss their
    residual tolerance.

    The delaminating Levin evaluator accepts each leaf on a rigorous residual
    bound; pixels that fail at the batched depth caps get a depth-6 per-pixel
    re-pass as the final safety net.  When even that cannot meet the bound the
    depth-12 deep pass already failed, the pixel value carries a
    larger-than-requested error -- which the old code returned SILENTLY (only a
    progress string mentioned the count).  Emit a ``RuntimeWarning`` so a
    caller can distrust those pixels.  Returns the number of over-tolerance
    pixels.  ``tolerances <= 0`` (dark pixels, ``f_scale == 0``) are ignored.
    """
    ab = np.asarray(achieved_bounds, dtype=float)
    tl = np.asarray(tolerances, dtype=float)
    valid = tl > 0.0
    over = valid & (ab > tl)
    n_over = int(np.count_nonzero(over))
    if n_over:
        worst = float(np.max(ab[over] / tl[over]))
        warnings.warn(
            f"Maslov Levin integrator: {n_over}/{int(n_total)} output "
            f"pixel(s) did not meet the residual tolerance "
            f"(levin_tol={levin_tol:g}) even after the depth-6 per-pixel "
            f"adaptive fallback (worst achieved bound ~{worst:.2g}x the "
            "target). These pixels carry a larger-than-requested error; treat "
            "their values as approximate. Consider a coarser levin_tol, or "
            "integration_method='traced_multibranch' for this chart.",
            RuntimeWarning, stacklevel=2)
    return n_over


def _integrate_levin(
    # E-L9 (audit): ``mi`` is dead here -- the exponent tuples are consumed
    # only as the K*_arr int64 arrays.  The physical half-widths v2x_h / v2y_h
    # WERE dropped as dead by E-L9 and are back (S4): the Van Vleck density
    # sqrt(|det ds1/dv2|) does not cancel against the d^2 v2 measure the way
    # the pre-S4 |det ds1/dv2| did, so the normalised-box integrand carries a
    # sqrt(v2x_h * v2y_h) that the engine's own unit box cannot supply.
    coef_opd, coef_s1x, coef_s1y,
    K1_arr, K2_arr, K3_arr, K4_arr,
    poly_order, N_out_coarse,
    u_s2x_out, u_s2y_out, inbox_flat,
    v2x_h, v2y_h,
    sample_E_bilinear,
    levin_tol, _progress, verbose,
    out_dtype=np.complex128,
    lin_v3=0.0, lin_v4=0.0,
):
    """Adaptive delaminating Levin evaluation of the Maslov v2 integral --
    caustic-UNIFORM (no saddle finding, finite and accurate through folds /
    cusps) at a per-pixel cost independent of the oscillation count.

    Evaluates, per output pixel, the same integrand as
    :func:`_integrate_quadrature` (Tukey-windowed unit v2 box)::

        E(s2) = int int  E_obj(s1(u)) |det ds1/du| tuk(u3) tuk(u4)
                          exp(2 pi i OPD(u)) du3 du4

    with the phase/amplitude and their derivatives supplied pointwise by
    the Chebyshev fits (`_opd6`).  ``levin_tol`` is RELATIVE (scaled per
    pixel by ``max|f| x domain area``); every accepted box carries a
    rigorous residual bound ``|I_hat - I| <= int int |r|``.

    Implementation: lockstep per-pixel quadtrees, wave-batched over
    (pixel, box) pairs -- each refinement wave evaluates ALL pairs in a
    handful of large vectorized ``_opd6`` / batched-solve calls
    (regularized normal-equations collocation standing in for the
    delaminating TSVD; the residual bound measures the actual solution, so
    regularization can only cost refinement, never accuracy).  Leaves are
    accepted by residual-bound equidistribution; pixels that fail at the
    depth cap get a deeper batched re-pass, then a per-pixel adaptive
    engine (:func:`lumenairy._math.levin.levin2d`) as the final safety
    net.  Unlike 'stationary_phase' / 'local_quadrature' it does not
    diverge at caustics, and unlike 'quadrature' its cost does not scale
    with ``n_v2^2``; on a hard high-NA chart expect ~0.04 s/pixel at
    ``levin_tol=1e-2`` and ~0.3 s/pixel at ``1e-3`` (~300-2000x the
    per-pixel adaptive engine).  Peak memory is chunk-bounded
    (~hundreds of MB of transients) independent of grid size.
    """
    from .._math.levin import _cheb_D as _cheb_D_phys, levin2d
    t0 = time.perf_counter()
    N_px = N_out_coarse * N_out_coarse
    u1f = u_s2x_out.ravel()
    u2f = u_s2y_out.ravel()
    E_flat = np.zeros(N_px, dtype=out_dtype)

    def _tuk(u, alpha=0.2):                   # shared window (S2-14)
        return _tukey_taper(u, alpha)

    # S4: the physical d^2 v2 measure that the unit-box Levin engine does not
    # carry; see the ``f`` closure below.  VERIFY-A4 (O-2): the DENSITY
    # itself comes from the shared :func:`_van_vleck_density`, like the six
    # other integrand sites -- the two closures below used to write
    # ``sqrt(|det J_norm|)`` out by hand, which is exactly the drift the
    # shared helper exists to prevent.  Composing them is an identity:
    # ``_van_vleck_density(d, 1, 1) == d ** 0.5``, and NumPy's ``** 0.5`` is
    # ``sqrt`` bit-for-bit (verified over 1e5 samples), so composing the
    # helper with the box Jacobian ``sqrt(hx * hy)`` reproduces the previous
    # expression EXACTLY -- pinned in
    # tests/unit/test_audit2609_a4_verify_maslov_asymptotic.py.
    #
    # Unit half-widths are the right call here, not ``(v2x_h, v2y_h)``: the
    # Levin engine works in the NORMALISED unit box from end to end, so it
    # wants the normalised-chart density and carries the box map's Jacobian
    # separately.  (``_van_vleck_density(d, hx, hy) * hx * hy`` is the same
    # number to 1-2 ULP but re-associates the products, which would move the
    # returned field off bit-identity for no reason.)
    _vv_measure = float(np.sqrt(v2x_h * v2y_h))

    idx = np.where(inbox_flat)[0]
    twopi = 2.0 * np.pi

    # ---- per-pixel closures (probe + fallback path) ----------------------
    def _pixel_funcs(p):
        u1p = u1f[p]
        u2p = u2f[p]

        def _ev(coef, u3, u4):
            sh = np.shape(u3)
            u3f = np.asarray(u3, dtype=np.float64).ravel()
            u4f = np.asarray(u4, dtype=np.float64).ravel()
            o1 = np.full_like(u3f, u1p)
            o2 = np.full_like(u3f, u2p)
            out = _opd6(coef, K1_arr, K2_arr, K3_arr, K4_arr,
                        o1, o2, u3f, u4f, poly_order)
            return [o.reshape(sh) for o in out]

        def g(u3, u4):
            return twopi * (_ev(coef_opd, u3, u4)[0]
                            + lin_v3 * u3 + lin_v4 * u4)

        def gx(u3, u4):
            return twopi * (_ev(coef_opd, u3, u4)[1] + lin_v3)

        def gy(u3, u4):
            return twopi * (_ev(coef_opd, u3, u4)[2] + lin_v4)

        def f(u3, u4):
            s1x, dx3, dx4 = _ev(coef_s1x, u3, u4)[:3]
            s1y, dy3, dy4 = _ev(coef_s1y, u3, u4)[:3]
            # S4: sqrt(|det ds1/dv2|) * d^2 v2 -- the Levin engine
            # integrates over the normalised unit box, so the physical
            # measure ``v2x_h * v2y_h`` rides on ``f`` beside the shared
            # Van Vleck density.
            detJ = _vv_measure * _van_vleck_density(
                np.abs(dx3 * dy4 - dx4 * dy3), 1.0, 1.0)
            Eo = sample_E_bilinear(
                s1x.ravel(), s1y.ravel()).reshape(np.shape(u3))
            return Eo * detJ * _tuk(np.asarray(u3)) * _tuk(np.asarray(u4))

        return g, gx, gy, f

    # levin_tol is RELATIVE; the engine's tolerance is absolute -- scale by
    # the integrand magnitude x box area (the natural size of the
    # non-oscillatory integral; the paper's eps is relative to ||f||_inf).
    def _pixel_scale(f):
        _up = np.linspace(-1.0, 1.0, 9)
        _U3p, _U4p = np.meshgrid(_up, _up, indexing='ij')
        return float(np.abs(f(_U3p, _U4p)).max()) * 4.0

    def _pixel_adaptive(p, max_depth=6, tol_factor=1.0, want_leaves=False,
                        return_est=False):
        g, gx, gy, f = _pixel_funcs(p)
        f_scale = _pixel_scale(f)
        tol_used = levin_tol * f_scale * tol_factor
        if f_scale <= 0.0:
            if return_est:
                return 0.0j, 0.0, 0.0
            return (0.0j, []) if want_leaves else 0.0j
        out = levin2d(g, gx, gy, f, (-1.0, 1.0, -1.0, 1.0),
                      tol=tol_used, k=7,
                      max_depth=max_depth, return_leaves=want_leaves)
        if want_leaves:
            return out[0], [leaf[0] for leaf in out[3]]
        if return_est:
            # levin2d returns (val, n_leaves, achieved_residual_bound); the
            # bound lets the caller see when even this per-pixel fallback could
            # not meet ``tol_used`` (audit S2-18).
            return out[0], float(out[2]), float(tol_used)
        return out[0]

    if len(idx) == 0:
        return E_flat.reshape(N_out_coarse, N_out_coarse)

    # ---- batched machinery shared by all boxes ---------------------------
    kk = 7
    n_act = len(idx)
    u1a = u1f[idx]
    u2a = u2f[idx]
    us = np.cos(np.pi * np.arange(kk) / (kk - 1))[::-1]      # std nodes
    from numpy.polynomial import chebyshev as _Ch
    Vs = _Ch.chebvander(us, kk - 1)                          # (k, k)
    Vs_inv = np.linalg.inv(Vs)
    nf = 2 * kk
    uf = np.cos(np.pi * np.arange(nf) / (nf - 1))[::-1]
    Ef_mat = _Ch.chebvander(uf, kk - 1)                      # (2k, k)
    # derivative eval matrix on std interval (chain-ruled per box below)
    Ed_mat = np.stack([_Ch.chebval(uf, _Ch.chebder(np.eye(kk)[j]))
                       for j in range(kk)], axis=1)          # (2k, k)
    k_b = 12
    us_b = np.cos(np.pi * np.arange(k_b) / (k_b - 1))[::-1]

    def _pair_ev9(o1, o2, u3v, u4v):
        """ONE shared-basis kernel call for a batch of (pixel, box) pairs:
        value + d3 + d4 for opd, s1x, s1y (everything the Levin integrand
        needs, no wasted T'' work).  o1/o2: (m,) pixel coords; u3v/u4v:
        (m, nn) chart points -> list of nine (m, nn) arrays."""
        mm, nn = u3v.shape
        out = _opd_vd9(coef_opd, coef_s1x, coef_s1y,
                       K1_arr, K2_arr, K3_arr, K4_arr,
                       np.repeat(o1, nn), np.repeat(o2, nn),
                       np.ascontiguousarray(u3v, dtype=np.float64).ravel(),
                       np.ascontiguousarray(u4v, dtype=np.float64).ravel(),
                       poly_order)
        return [o.reshape(mm, nn) for o in out]

    def _pairs_f(u3v, u4v, sx, dx3, dx4, sy, dy3, dy4):
        """Integrand amplitude f from the s1x/s1y outputs of _pair_ev9."""
        detJ = _vv_measure * _van_vleck_density(
            np.abs(dx3 * dy4 - dx4 * dy3), 1.0, 1.0)
        Eo = sample_E_bilinear(sx.ravel(), sy.ravel()).reshape(sx.shape)
        return (Eo * detJ * _tuk(np.asarray(u3v, dtype=np.float64))
                * _tuk(np.asarray(u4v, dtype=np.float64)))

    def _pair_f(o1, o2, u3v, u4v):
        out = _pair_ev9(o1, o2, u3v, u4v)
        return _pairs_f(u3v, u4v, *out[3:9])

    def _tik_batch(D, gdiag, rhs):
        """Batched regularized solve of (D + i diag(g)) p = rhs.

        D: (k, k) standard-interval differentiation matrix; gdiag/rhs:
        (B, k).  Normal-equations Tikhonov at ~1e-8 relative approximates
        the delaminating TSVD's minimal-norm behaviour at the level double
        precision supports through A^H A; the rigorous residual bound
        downstream measures the ACTUAL solution, so a poorly regularized
        fiber can only cost refinement, never accuracy.
        """
        k = D.shape[0]
        A = np.broadcast_to(D.astype(complex), (gdiag.shape[0], k, k)).copy()
        ii = np.arange(k)
        A[:, ii, ii] += 1j * gdiag
        nrm = np.abs(A).sum(axis=2).max(axis=1)
        lam2 = (1e-8 * nrm) ** 2
        M = A.conj().transpose(0, 2, 1) @ A
        M[:, ii, ii] += lam2[:, None]
        y = np.einsum('bji,bj->bi', A.conj(), rhs)
        return np.linalg.solve(M, y[..., None])[..., 0]

    D_std_s = _cheb_D_phys(kk, -1.0, 1.0)
    D_std_b = _cheb_D_phys(k_b, -1.0, 1.0)

    def _eval_pairs_chunk(pp, bxa):
        """Levin-evaluate a batch of (pixel, box) PAIRS in one shot.

        pp: (m,) indices into the active-pixel arrays; bxa: (m, 4) boxes
        (a, b, c, d).  Returns ``(val, est)``: per-pair box integrals
        (complex) and rigorous residual bounds ``area * mean|r|``.
        Collocation is standardized -- on a box of s-half-width hs,
        ``A_phys = (1/hs)(D_std + i hs diag(g_s))`` -- so pairs with
        different box sizes and delamination axes share one batch.
        """
        m = len(pp)
        o1 = u1a[pp]
        o2 = u2a[pp]
        mid_x, hx = (0.5 * (bxa[:, 0] + bxa[:, 1]),
                     0.5 * (bxa[:, 1] - bxa[:, 0]))
        mid_y, hy = (0.5 * (bxa[:, 2] + bxa[:, 3]),
                     0.5 * (bxa[:, 3] - bxa[:, 2]))
        xn = mid_x[:, None] + hx[:, None] * us[None, :]      # (m, k)
        yn = mid_y[:, None] + hy[:, None] * us[None, :]
        U3 = np.broadcast_to(xn[:, :, None], (m, kk, kk)).reshape(m, -1)
        U4 = np.broadcast_to(yn[:, None, :], (m, kk, kk)).reshape(m, -1)
        ev = _pair_ev9(o1, o2, U3, U4)
        g3 = (twopi * (ev[1] + lin_v3)).reshape(m, kk, kk)
        g4 = (twopi * (ev[2] + lin_v4)).reshape(m, kk, kk)
        fv = _pairs_f(U3, U4, *ev[3:9]).reshape(m, kk, kk)
        # per-pair delamination axis; normalize arrays to (s, t) order
        m3 = np.abs(g3).mean(axis=(1, 2)) >= np.abs(g4).mean(axis=(1, 2))
        M3 = m3[:, None, None]
        G_s = np.where(M3, g3, g4.transpose(0, 2, 1))
        G_t = np.where(M3, g4, g3.transpose(0, 2, 1))
        Fv = np.where(M3, fv, fv.transpose(0, 2, 1))
        hs = np.where(m3, hx, hy)
        ht = np.where(m3, hy, hx)
        mid_s = np.where(m3, mid_x, mid_y)
        mid_t = np.where(m3, mid_y, mid_x)
        # interior fiber solves: per t-fiber j, (D_s + i diag(g_s)) p = f
        gd = (G_s.transpose(0, 2, 1) * hs[:, None, None]).reshape(-1, kk)
        rhs = (Fv.transpose(0, 2, 1)
               * hs[:, None, None]).reshape(-1, kk).astype(complex)
        P = _tik_batch(D_std_s, gd, rhs).reshape(m, kk, kk)  # [m, j_t, i_s]
        p_hi = P[:, :, -1]
        p_lo = P[:, :, 0]
        # ---- boundary 1-D Levin along t at s = +/-1 (composite segs) -----
        val = np.zeros(m, dtype=np.complex128)
        for sign_, s_off, p_end in ((1.0, 1.0, p_hi), (-1.0, -1.0, p_lo)):
            gt_edge = G_t[:, -1 if s_off > 0 else 0, :]      # (m, k)
            span = np.abs(gt_edge).max(axis=1) * (2.0 * ht)
            n_seg = np.clip(np.ceil(span / 8.0).astype(np.int64), 1, 48)
            S = int(n_seg.sum())
            par = np.repeat(np.arange(m), n_seg)
            base = np.repeat(np.cumsum(n_seg) - n_seg, n_seg)
            loc = np.arange(S) - base
            w = (2.0 * ht / n_seg)[par]
            h_seg = 0.5 * w
            mid_seg = (mid_t - ht)[par] + (loc + 0.5) * w
            tseg = mid_seg[:, None] + h_seg[:, None] * us_b[None, :]
            s_fix = (mid_s + s_off * hs)[par]
            m3s = m3[par]
            u3s = np.where(m3s[:, None], s_fix[:, None], tseg)
            u4s = np.where(m3s[:, None], tseg, s_fix[:, None])
            oseg, e3, e4 = _pair_ev9(o1[par], o2[par], u3s, u4s)[:3]
            gseg = twopi * (oseg + lin_v3 * u3s + lin_v4 * u4s)
            gts = twopi * np.where(m3s[:, None], e4 + lin_v4, e3 + lin_v3)
            useg = (tseg - mid_t[par][:, None]) / ht[par][:, None]
            Mv = (_Ch.chebvander(useg.reshape(-1), kk - 1)
                  .reshape(S, k_b, kk) @ Vs_inv)
            amps = np.einsum('sik,sk->si', Mv, p_end[par]).astype(complex)
            q = _tik_batch(D_std_b, gts * h_seg[:, None],
                           amps * h_seg[:, None])
            contrib = sign_ * (q[:, -1] * np.exp(1j * gseg[:, -1])
                               - q[:, 0] * np.exp(1j * gseg[:, 0]))
            np.add.at(val, par, contrib)
        # ---- rigorous residual bound: |I_hat - I| <= int int |r| ---------
        C = np.einsum('mi,pji,kj->pmk', Vs_inv, P, Vs_inv)   # [p, m_s, k_t]
        Pf = np.einsum('am,pmk,bk->pab', Ef_mat, C, Ef_mat)
        Pd = (np.einsum('am,pmk,bk->pab', Ed_mat, C, Ef_mat)
              / hs[:, None, None])
        sfn = mid_s[:, None] + hs[:, None] * uf[None, :]
        tfn = mid_t[:, None] + ht[:, None] * uf[None, :]
        SF = np.broadcast_to(sfn[:, :, None], (m, nf, nf))
        TF = np.broadcast_to(tfn[:, None, :], (m, nf, nf))
        M3f = m3[:, None, None]
        u3r = np.where(M3f, SF, TF).reshape(m, -1)
        u4r = np.where(M3f, TF, SF).reshape(m, -1)
        evr = _pair_ev9(o1, o2, u3r, u4r)
        gsr = twopi * np.where(m3[:, None], evr[1] + lin_v3,
                               evr[2] + lin_v4)
        fr = _pairs_f(u3r, u4r, *evr[3:9])
        r = Pd.reshape(m, -1) + 1j * gsr * Pf.reshape(m, -1) - fr
        est = np.abs(r).mean(axis=1) * (4.0 * hs * ht)
        return val, est

    def _eval_pairs(pp, bxa):
        m = len(pp)
        chunk = max(64, int(2.5e6 // (nf * nf)))
        val = np.empty(m, dtype=np.complex128)
        est = np.empty(m, dtype=np.float64)
        for i0 in range(0, m, chunk):
            sl = slice(i0, min(m, i0 + chunk))
            val[sl], est[sl] = _eval_pairs_chunk(pp[sl], bxa[sl])
        return val, est

    # ---- tolerance scale: batched 9x9 probe (same semantics as the
    # per-pixel path -- levin_tol is RELATIVE to max|f| x domain area) -----
    _up = np.linspace(-1.0, 1.0, 9)
    _U3p, _U4p = np.meshgrid(_up, _up, indexing='ij')
    fmax_b = np.empty(n_act, dtype=np.float64)
    _pc = 20000
    for i0 in range(0, n_act, _pc):
        sl = slice(i0, min(n_act, i0 + _pc))
        nsl = sl.stop - sl.start
        fmax_b[sl] = np.abs(_pair_f(
            u1a[sl], u2a[sl],
            np.broadcast_to(_U3p.ravel(), (nsl, 81)),
            np.broadcast_to(_U4p.ravel(), (nsl, 81)))).max(axis=1)
    tol_px = levin_tol * fmax_b * 4.0
    live = fmax_b > 0.0
    tol_safe = np.where(live, tol_px, np.inf)

    # ---- lockstep per-pixel quadtrees, wave-batched over (pixel, box) ----
    # The phase surface differs per pixel (u1/u2 enter the 4-D Chebyshev
    # fit), so each active pixel refines its OWN quadtree; all (pixel, box)
    # pairs of a refinement wave are evaluated together in a handful of
    # large vectorized calls.  A leaf is accepted by residual-bound
    # EQUIDISTRIBUTION -- est <= tol_px * (box area / domain area) -- which
    # guarantees the per-pixel sum over leaves meets tol_px.
    accum_val = np.zeros(n_act, dtype=np.complex128)
    accum_est = np.zeros(n_act, dtype=np.float64)
    n_pairs_tot = 0

    def _run_waves(px, max_depth, max_pairs, tag):
        nonlocal n_pairs_tot
        pp = px.copy()
        bxa = np.tile(np.array([[-1.0, 1.0, -1.0, 1.0]]), (len(pp), 1))
        dep = np.zeros(len(pp), dtype=np.int64)
        # ADAPTIVE budget: a leaf is accepted when its bound fits the
        # pixel's REMAINING tolerance re-equidistributed over its remaining
        # open area.  Accepted leaves consume tolerance and release area, so
        # the budget density rem_tol/open_area is monotone non-decreasing
        # (rigor preserved: per-pixel sum of accepted bounds <= tol_px) and
        # slack from smooth regions flows to the hard caustic-band leaves --
        # which fixed equidistribution starves into deep refinement.
        rem_tol = tol_safe.copy()
        open_area = np.zeros(n_act, dtype=np.float64)
        open_area[pp] = 4.0
        wave = 0
        n_pairs = 0
        while len(pp):
            v_w, e_w = _eval_pairs(pp, bxa)
            n_pairs += len(pp)
            area = ((bxa[:, 1] - bxa[:, 0]) * (bxa[:, 3] - bxa[:, 2]))
            need = ((e_w > rem_tol[pp] * area / open_area[pp])
                    & (dep < max_depth) & (n_pairs < max_pairs))
            keep = ~need
            np.add.at(accum_val, pp[keep], v_w[keep])
            np.add.at(accum_est, pp[keep], e_w[keep])
            np.add.at(rem_tol, pp[keep], -e_w[keep])
            np.add.at(open_area, pp[keep], -area[keep])
            np.maximum(rem_tol, 0.0, out=rem_tol)
            np.maximum(open_area, 1e-300, out=open_area)
            pn = pp[need]
            a_, b_, c_, d_ = bxa[need].T
            mx, my = 0.5 * (a_ + b_), 0.5 * (c_ + d_)
            bxa = np.concatenate([
                np.stack([a_, mx, c_, my], axis=1),
                np.stack([mx, b_, c_, my], axis=1),
                np.stack([a_, mx, my, d_], axis=1),
                np.stack([mx, b_, my, d_], axis=1)], axis=0)
            pp = np.concatenate([pn, pn, pn, pn])
            dep = np.tile(dep[need] + 1, 4)
            wave += 1
            if len(pp):
                _progress('integrate', min(0.6 + 0.02 * wave, 0.78),
                          f'levin: {tag} wave {wave}, {len(pp)} pair-boxes '
                          f'queued ({n_pairs} evaluated, '
                          f'{time.perf_counter() - t0:.1f}s)')
        n_pairs_tot += n_pairs

    _run_waves(np.where(live)[0], max_depth=8,
               max_pairs=20000 * max(1, n_act), tag='main')

    # ---- deep re-pass for pixels whose bound failed at the depth cap -----
    bad = np.where((accum_est > tol_px) & live)[0]
    if len(bad):
        _progress('integrate', 0.79,
                  f'levin: deep re-pass for {len(bad)} pixels '
                  f'({time.perf_counter() - t0:.1f}s)')
        accum_val[bad] = 0.0
        accum_est[bad] = 0.0
        _run_waves(bad, max_depth=12, max_pairs=200000 * len(bad),
                   tag='deep')
        bad = np.where((accum_est > tol_px) & live)[0]

    E_flat[idx] = accum_val
    # ---- per-pixel adaptive safety net (rarely reached) ------------------
    _progress('integrate', 0.8,
              f'levin: batched pass done ({n_pairs_tot} pair-boxes, '
              f'{time.perf_counter() - t0:.1f}s), '
              f'{len(bad)}/{len(idx)} pixels -> adaptive fallback')
    # audit S2-18: track pixels whose residual bound STILL exceeds tolerance
    # after the depth-6 per-pixel fallback (the depth-12 deep re-pass already
    # failed them).  Those pixels carry a larger-than-requested error; the old
    # code returned them silently (only a progress string mentioned the count).
    fb_est = np.empty(len(bad), dtype=np.float64)
    fb_tol = np.empty(len(bad), dtype=np.float64)
    for n_done, ib in enumerate(bad):
        val, est, tol_used = _pixel_adaptive(idx[ib], return_est=True)
        E_flat[idx[ib]] = val
        fb_est[n_done] = est
        fb_tol[n_done] = tol_used
        if verbose:
            _progress('integrate', 0.8 + 0.15 * n_done / max(1, len(bad)),
                      f'levin fallback: pixel {n_done + 1}/{len(bad)}')

    n_over = _warn_levin_over_tolerance(fb_est, fb_tol, len(idx), levin_tol)

    _progress('integrate', 0.95,
              f'levin: {len(idx)} pixels ({n_pairs_tot} pair-boxes, '
              f'{len(bad)} adaptive fallbacks, {n_over} over-tolerance) in '
              f'{time.perf_counter() - t0:.1f}s')
    return E_flat.reshape(N_out_coarse, N_out_coarse)


def _integrate_local_quadrature(
    # E-L9 (audit): ``mi`` (consumed only as the K*_arr arrays) and the box
    # centres v2x_c / v2y_c (the saddle + window live in normalised u_v2; only
    # the half-widths de-normalise the Hessian) were dead.  Dropped; the CuPy
    # twin _integrate_local_quadrature_cupy never took any of the three.
    coef_opd, coef_s1x, coef_s1y,
    K1_arr, K2_arr, K3_arr, K4_arr,
    poly_order, N_out_coarse,
    u_s2x_out, u_s2y_out, inbox_flat,
    v2x_h, v2y_h,
    sample_E_bilinear,
    newton_iter, newton_tol,
    n_samples, window_sigma,
    _progress, verbose,
    out_dtype=np.complex128,
    lin_v3=0.0, lin_v4=0.0, k1_fit=None,
):
    """Gaussian-windowed local quadrature about the per-pixel saddle.

    Samples the integrand on a lattice that is laid out on the PRINCIPAL
    AXES of the phase Hessian (not the coordinate axes), tapers it with a
    Gaussian window, and divides out that window's exact effect on the
    quadratic model of the phase.  The three pieces are what make it
    converge; see :func:`_local_window_geometry` for the algebra and the
    measured convergence ladder.

    Exact on a quadratic chart at any ``n_samples`` / ``window_sigma`` WHILE
    THE TAPERED LATTICE FITS INSIDE THE FITTED CHART BOX; out-of-box samples
    are dropped and the exactness goes with them, which
    :func:`_warn_local_window_truncation` announces once above a 1 % dropped
    fraction.

    v4.14.0: ``out_dtype`` defaults to ``np.complex128`` for back-
    compat; callers pass ``E_in.dtype`` to preserve complex64 inputs.

    ``k1_fit`` (S6) puts the input field's own phase into the stationary
    condition -- which re-CENTRES the window -- and into the Hessian that sets
    the window's principal axes, widths and taper correction; see
    :func:`_input_phase_terms`.  The integrand itself is unchanged: it samples
    the complex ``E_in``, which already carries that phase exactly.  ``None``
    -- a flat input, or the refused fallback -- runs the pre-S6 arithmetic bit
    for bit.
    """
    t_int_start = time.perf_counter()
    _progress('integrate', 0.60,
              f'local_quadrature: Newton phase ({newton_iter} max iters)')

    N_px = N_out_coarse * N_out_coarse
    u_s2x_flat = u_s2x_out.ravel()
    u_s2y_flat = u_s2y_out.ravel()

    def _opd_and_derivs(coef, u1, u2, u3, u4):
        # M-P4: dispatch to the Numba kernel (default, ULP-equal) or the
        # NumPy reference; returns (f, df_du3, df_du4, d2f_33, d2f_34, d2f_44).
        return _opd6(coef, K1_arr, K2_arr, K3_arr, K4_arr,
                     u1, u2, u3, u4, poly_order)

    u_v2x, u_v2y, converged = _maslov_newton_saddle_cpu(   # shared CPU (S2-14)
        _opd_and_derivs, coef_opd, u_s2x_flat, u_s2y_flat, inbox_flat,
        newton_iter, newton_tol, lin_v3, lin_v4, k1_fit=k1_fit)

    _progress('integrate', 0.72, 'computing Hessian eigen-scales')
    _, _, _, H33, H34, H44 = _opd_and_derivs(
        coef_opd, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)
    if k1_fit is not None:
        # S6: the window is laid out on the principal axes of the TOTAL
        # stationary phase, and its taper is divided back out of the SAME
        # quadratic model -- so both read the input phase's curvature too.
        _, _, _a33, _a34, _a44 = _eval_input_phase_terms(
            _opd_and_derivs, k1_fit, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)
        H33 = H33 + _a33
        H34 = H34 + _a34
        H44 = H44 + _a44
    (sigma1_phys, sigma2_phys, cos_t, sin_t,
     window_corr) = _local_window_geometry(
        np, H33, H34, H44, v2x_h, v2y_h, n_samples, window_sigma)

    _progress('integrate', 0.75,
              f'local principal-axis sampling: {n_samples}x{n_samples} pts, '
              f'window={window_sigma}sigma (Gaussian-tapered)')
    lin, dxi, taper1d, _, _ = _local_window_1d(int(n_samples),
                                               float(window_sigma))
    Xlin, Ylin = np.meshgrid(lin, lin, indexing='xy')
    Xlin_flat = Xlin.ravel()
    Ylin_flat = Ylin.ravel()
    taper = np.outer(taper1d, taper1d).ravel()   # 'xy' meshgrid -> (y, x)

    # Rotate the principal-axis offsets (sigma1*xi1, sigma2*xi2) back into the
    # physical (v2x, v2y) frame, then normalise by the box half-widths.  The
    # rotation is orthogonal, so the area element sigma1*sigma2*dxi^2 below is
    # unchanged by it.
    off1 = sigma1_phys[:, None] * Xlin_flat[None, :]
    off2 = sigma2_phys[:, None] * Ylin_flat[None, :]
    u_v2x_samp = u_v2x[:, None] + (cos_t[:, None] * off1
                                   - sin_t[:, None] * off2) / v2x_h
    u_v2y_samp = u_v2y[:, None] + (sin_t[:, None] * off1
                                   + cos_t[:, None] * off2) / v2y_h
    del off1, off2
    # S2/P2 (audit): a bare ``np.clip`` folds every out-of-box sample onto
    # the box edge and still counts it at the full unclipped cell area,
    # over-counting by up to 3 decades on a weakly-curved chart.  Samples
    # outside the fitted chart carry no information (the Chebyshev
    # recurrences are not even accurate there), so DROP them: clip to keep
    # the polynomial evaluation in its accurate range, and zero the
    # contribution.  See ``docs/history/lumenairy.elements.lenses_maslov.md``.
    in_chart = ((np.abs(u_v2x_samp) <= 1.0) & (np.abs(u_v2y_samp) <= 1.0))
    np.clip(u_v2x_samp, -1.0, 1.0, out=u_v2x_samp)
    np.clip(u_v2y_samp, -1.0, 1.0, out=u_v2y_samp)
    _warn_local_window_truncation(in_chart, inbox_flat, n_samples,
                                  window_sigma)

    n_s2 = n_samples * n_samples
    u_s2x_tile = np.broadcast_to(u_s2x_flat[:, None], (N_px, n_s2))
    u_s2y_tile = np.broadcast_to(u_s2y_flat[:, None], (N_px, n_s2))

    _progress('integrate', 0.78,
              f'evaluating integrand on {N_px*n_s2:,} (pixel,sample) pairs')

    E_flat = np.zeros(N_px, dtype=out_dtype)
    w2d_phys = (sigma1_phys * sigma2_phys) * (dxi ** 2) * window_corr

    PX_CHUNK = max(1, min(N_px, 1024 * 64 // max(1, n_s2 // 16)))
    for p_start in range(0, N_px, PX_CHUNK):
        p_end = min(p_start + PX_CHUNK, N_px)
        u3 = u_v2x_samp[p_start:p_end].ravel()
        u4 = u_v2y_samp[p_start:p_end].ravel()
        u1 = u_s2x_tile[p_start:p_end].ravel()
        u2 = u_s2y_tile[p_start:p_end].ravel()
        # One shared-basis value+1st-deriv kernel for the three
        # coef sets (opd value only; s1x/s1y value + du3/du4), skipping the
        # unused second derivatives and the 2x redundant basis rebuild.
        (opd_v, s1x_v, ds1x_du3, ds1x_du4,
         s1y_v, ds1y_du3, ds1y_du4) = _opd_vd3(
            coef_opd, coef_s1x, coef_s1y, K1_arr, K2_arr, K3_arr, K4_arr,
            u1, u2, u3, u4, poly_order)
        # N4: linear-in-v2 OPD contribution at each window sample.
        opd_v = opd_v + lin_v3 * u3 + lin_v4 * u4
        det_J = ds1x_du3 * ds1y_du4 - ds1x_du4 * ds1y_du3
        abs_J = _van_vleck_density(np.abs(det_J), v2x_h, v2y_h)

        Eobj_v = sample_E_bilinear(s1x_v, s1y_v)

        contrib = (Eobj_v
                    * np.exp(2j * np.pi * opd_v)
                    * abs_J)
        contrib_r = contrib.reshape(p_end - p_start, n_s2)
        # Gaussian taper (broadcast over the shared sample lattice) and the
        # out-of-chart drop; the taper is divided back out by ``window_corr``
        # inside ``w2d_phys``.
        contrib_r = np.where(in_chart[p_start:p_end],
                             contrib_r * taper[None, :], 0.0)
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
    out_dtype=np.complex128, lin_v3=0.0, lin_v4=0.0, k1_fit=None,
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

    k1_dev = _k1_fit_to_device(xp, k1_fit)
    u_v2x, u_v2y, converged = _maslov_newton_saddle_xp(
        xp, opd6, cop, u_s2x, u_s2y, inbox, newton_iter, newton_tol,
        lin_v3, lin_v4, k1_fit=k1_dev)

    opd_star, g3, g4, H33, H34, H44 = opd6(cop, u_s2x, u_s2y, u_v2x, u_v2y)
    opd_star = opd_star + lin_v3 * u_v2x + lin_v4 * u_v2y
    _s1x_ev = opd6(csx, u_s2x, u_s2y, u_v2x, u_v2y)
    _s1y_ev = opd6(csy, u_s2x, u_s2y, u_v2x, u_v2y)
    s1x_star, ds1x_du3, ds1x_du4 = _s1x_ev[0], _s1x_ev[1], _s1x_ev[2]
    s1y_star, ds1y_du3, ds1y_du4 = _s1y_ev[0], _s1y_ev[1], _s1y_ev[2]
    if k1_dev is not None:
        # S6: total-phase curvature, matching the CPU twin operation for
        # operation (the s1 fits are the SAME two ``opd6`` calls above).
        _, _, _a33, _a34, _a44 = _input_phase_terms(
            _s1x_ev, _s1y_ev,
            opd6(k1_dev[2], u_s2x, u_s2y, u_v2x, u_v2y),
            opd6(k1_dev[3], u_s2x, u_s2y, u_v2x, u_v2y), k1_dev[4])
        H33 = H33 + _a33
        H34 = H34 + _a34
        H44 = H44 + _a44

    det_J_norm = ds1x_du3 * ds1y_du4 - ds1x_du4 * ds1y_du3
    abs_J = _van_vleck_density(xp.abs(det_J_norm), v2x_h, v2y_h)

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
    out_dtype=np.complex128, lin_v3=0.0, lin_v4=0.0, k1_fit=None,
):
    """CuPy GPU twin of :func:`_integrate_local_quadrature`.

    Same Gaussian-windowed principal-axis local quadrature -- Newton saddle,
    Hessian EIGENBASIS window (`sigma1`, `sigma2` along the eigenvectors),
    an ``n_samples x n_samples`` tapered lattice per pixel, and the tapered
    quadratic-model correction divided back out (audit S2) -- on the device.
    The 1-D lattice / taper / correction come from the same host-side
    :func:`_local_window_1d` the CPU integrator uses, so the two cannot drift.
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

    k1_dev = _k1_fit_to_device(xp, k1_fit)
    u_v2x, u_v2y, converged = _maslov_newton_saddle_xp(
        xp, opd6, cop, u_s2x, u_s2y, inbox, newton_iter, newton_tol,
        lin_v3, lin_v4, k1_fit=k1_dev)

    _, _, _, H33, H34, H44 = opd6(cop, u_s2x, u_s2y, u_v2x, u_v2y)
    if k1_dev is not None:
        # S6: total-phase curvature -- the window's axes, widths and taper
        # correction, exactly as the CPU twin computes them.
        _, _, _a33, _a34, _a44 = _eval_input_phase_terms(
            opd6, k1_dev, u_s2x, u_s2y, u_v2x, u_v2y)
        H33 = H33 + _a33
        H34 = H34 + _a34
        H44 = H44 + _a44
    (sigma1_phys, sigma2_phys, cos_t, sin_t,
     window_corr) = _local_window_geometry(
        xp, H33, H34, H44, v2x_h, v2y_h, n_samples, window_sigma)

    lin_h, dxi, taper1d_h, _, _ = _local_window_1d(int(n_samples),
                                                   float(window_sigma))
    lin = xp.asarray(lin_h)
    Xlin, Ylin = xp.meshgrid(lin, lin, indexing='xy')
    Xlin_flat = Xlin.ravel()
    Ylin_flat = Ylin.ravel()
    taper = xp.asarray(np.outer(taper1d_h, taper1d_h).ravel())
    off1 = sigma1_phys[:, None] * Xlin_flat[None, :]
    off2 = sigma2_phys[:, None] * Ylin_flat[None, :]
    u_v2x_samp = u_v2x[:, None] + (cos_t[:, None] * off1
                                   - sin_t[:, None] * off2) / v2x_h
    u_v2y_samp = u_v2y[:, None] + (sin_t[:, None] * off1
                                   + cos_t[:, None] * off2) / v2y_h
    del off1, off2
    in_chart = ((xp.abs(u_v2x_samp) <= 1.0) & (xp.abs(u_v2y_samp) <= 1.0))
    u_v2x_samp = xp.clip(u_v2x_samp, -1.0, 1.0)
    u_v2y_samp = xp.clip(u_v2y_samp, -1.0, 1.0)
    # One device->host transfer of the boolean mask so the GPU integrator is
    # as audible as the CPU one; see _warn_local_window_truncation.
    _warn_local_window_truncation(
        xp.asnumpy(in_chart) if hasattr(xp, 'asnumpy') else np.asarray(
            in_chart),
        np.asarray(inbox_flat), n_samples, window_sigma)
    n_s2 = n_samples * n_samples
    w2d_phys = (sigma1_phys * sigma2_phys) * (dxi ** 2) * window_corr

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
        abs_J = _van_vleck_density(xp.abs(det_J), v2x_h, v2y_h)
        Eobj_v = _sample_bilinear_xp(xp, E_in_gpu, N, dx, dy, s1x_v, s1y_v)
        contrib = (Eobj_v * xp.exp(2j * xp.pi * opd_v)
                   * abs_J).reshape(bw, n_s2)
        contrib = xp.where(in_chart[p0:p1], contrib * taper[None, :], 0.0)
        E_flat[p0:p1] = contrib.sum(axis=1) * w2d_phys[p0:p1]

    E_flat = xp.where(converged, E_flat, xp.asarray(0, dtype=out_dtype))
    E_flat = xp.where(inbox, E_flat, xp.asarray(0, dtype=out_dtype))
    return E_flat.reshape(N_out_coarse, N_out_coarse)


def clear_maslov_local_window_cache() -> None:
    """Drop the cached ``local_quadrature`` sample lattices.

    :func:`_local_window_1d` memoises the 1-D lattice, Gaussian taper and
    quadratic-model corrections on ``(local_n_samples, local_window_sigma)``.
    The entries are tiny -- three float/complex arrays of length
    ``local_n_samples`` (<= a few kB at any sane sample count, times an LRU
    cap of 32) -- so there is no byte-budget hook; the clearer exists so the
    cache participates in the central registry contract (``clear_asm_caches``
    / ``lumenairy_context(clear_caches_on_exit=True)``) like every other
    module-level cache in the package.

    Clearing is always safe: the cache is a pure function of its key, so a
    drained cache only costs the (microsecond) rebuild.
    """
    _local_window_1d.cache_clear()


# v4.16.0 central cache registry: register at module-import time so
# ``clear_asm_caches`` picks this cache up by walking the registry instead of
# enumerating clear calls by hand (see ``lumenairy/_cache_registry.py``).
try:
    import sys as _sys

    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _mz_this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        'maslov_local_window',
        lambda: getattr(_mz_this_mod, 'clear_maslov_local_window_cache')(),
    )
except ImportError:
    # Defensive, mirroring the sibling registrations: a partial install or an
    # odd reload sequence must not make this module unimportable.
    pass


__all__ = [
    'apply_real_lens_maslov',
    'apply_real_lens_maslov_vector',
    'clear_maslov_local_window_cache',
    'uniform_fold_airy',
    'pearcey',
]

"""W6-ASYM -- numerical interior of the asymptotic propagator family.

Territory: ``lumenairy/propagators/asymptotic.py``,
``asymptotic_canonical_fit.py``, ``asymptotic_modes.py``,
``asymptotic_maslov.py`` plus the parity-relevant parts of
``asymptotic_jax_twin.py`` -- the ~4400 lines the 2026-07-25 adversarial
audit's Territory P explicitly listed as "Not reached: asymptotic family
interior".  ``asymptotic_aberration_tensor.py`` is OUT of scope here (it
was validated in W3-T3/W4; its oracle machinery is reused below).

Findings pinned in this file
---------------------------

======  ========================================  ==========================
id      site                                      claim
======  ========================================  ==========================
W6-A1   ``asymptotic.py`` ``maslov_tracking``      The Keller-Maslov raster
                                                  unwrap is PROVABLY wrong
                                                  for this ``M`` and made
                                                  the returned field
                                                  GRID-DEPENDENT.  FIXED
                                                  (default -> ``'principal'``
                                                  + warning on the legacy
                                                  modes).
W6-A2   ``asymptotic_maslov.py`` batch Newton      ``converged`` used an
                                                  ABSOLUTE ``tol`` on a
                                                  DIMENSIONAL residual, so
                                                  it read False for pixels
                                                  converged to machine
                                                  precision.  FIXED
                                                  (scale-relative verdict;
                                                  ``v2*`` bit-unchanged).
W6-A3   whole family                               Leading-order accuracy
                                                  band vs a brute-force
                                                  quadrature of the SAME
                                                  integrand -- MEASURED, no
                                                  defect.
W6-A4   ``include_linear=False`` x 4 sites         The returned PHASE is
                                                  referenced to the fit's
                                                  linear ramp, so
                                                  ``extract_linear_phase``
                                                  silently changes it.
                                                  Documented + pinned
                                                  (amplitude unaffected).
W6-A5   ``asymptotic.py`` shape handling           ndim >= 3 grids raised a
                                                  bare "too many values to
                                                  unpack".  FIXED (named
                                                  guard).
W6-A6   whole family, fold caustic                 The field stays FINITE
                                                  through a caustic and the
                                                  KMAH pi/2 is delivered by
                                                  the PRINCIPAL branch --
                                                  premise of the branch
                                                  machinery REFUTED.
                                                  Accuracy envelope
                                                  MEASURED.
W6-A7   ``propagate_modal_asymptotic``             No radiometric
                                                  normalisation and
                                                  ``fit.wavelength`` never
                                                  read.  Documented +
                                                  pinned (exact linearity,
                                                  grid-converged power).
W6-A8   ``asymptotic_jax_twin.py`` evaluators      NumPy/JAX parity floor
                                                  MEASURED (not assumed):
                                                  3.3e-10 worst.
W6-A9   ``asymptotic_maslov.py`` batched poly      Batched multi-mode
        algebra                                   algebra EXACT to 4.7e-14
                                                  for a quadratic
                                                  exponent; leading-order
                                                  error grows 362x from
                                                  the piston channel to an
                                                  order-6 channel.
W6-A10  ``fit_canonical_polynomials_jax``          Row-weight dead-ray
                                                  masking is SOUND (NaN
                                                  hypothesis refuted);
                                                  parity floor under
                                                  vignetting MEASURED.
W6-A11  ``_lg_mode_conj_stack`` /                  ``indexing='xy'`` and
        ``_hg_mode_conj_stack`` cache key         ``indexing='ij'``
                                                  meshgrids of the same
                                                  axes COLLIDED on one
                                                  cache entry -> the 2nd
                                                  caller got TRANSPOSED
                                                  modes (8.232e+00 rel).
                                                  FIXED (corner
                                                  fingerprint).
W6-A12  ``lg_polynomial`` / ``hg_polynomial``      Monomial-basis
        + ``_evaluate_poly2d``                    high-order cancellation
                                                  hypothesis REFUTED
                                                  (<=5.7e-11 to m=n=14,
                                                  w-independent).
W6-A13  mode-stack cache values                    Returned BY IDENTITY and
                                                  WRITEABLE -> caller
                                                  mutation poisoned the
                                                  cache.  FIXED
                                                  (``setflags(write=
                                                  False)``).
W6-A14  ``decompose_lg`` docstring                 Said "trapezoidal
                                                  quadrature" (it is a
                                                  rectangle sum) and
                                                  ``(Nx, Ny)`` (it is
                                                  ``(Ny, Nx)``).  FIXED
                                                  (docs).
W6-A15  ``auto_bump_threshold_waves``              The ``+2`` bump
                                                  OVERSHOT
                                                  ``max_auto_poly_order``
                                                  by 1 on opposite
                                                  parity.  FIXED (clamp).
W6-A16  ``propagate_hf_chebyshev_quadrature``      NO validity guard at
                                                  all: a grid 10x outside
                                                  the fit box inflated
                                                  max|E| 44.9x silently;
                                                  non-uniform axes
                                                  silently integrated
                                                  with a mean pitch.
                                                  FIXED (two warnings) +
                                                  dead statement removed.
W6-A17  ``fit_canonical_polynomials``              CONDITIONING interior
        conditioning                              MEASURED: the DEFAULT
                                                  ``poly_order=6`` fit is
                                                  rank-deficient
                                                  (185/210,
                                                  cond 7.4e+15) and the
                                                  samples cover 1.00% of
                                                  the box.  No cliff near
                                                  the edge (REFUTED);
                                                  derivatives clean.
======  ========================================  ==========================

W6-A17 -- what was REFUTED near the box edge
--------------------------------------------
There is NO conditioning cliff just inside the edge, and there is no
reachable "just outside" at all.  ``_fit_normaliser`` pads the observed
extent by 5%, so a physically-traced ray can never exceed
``|u| = 1/1.05 = 0.9524`` -- measured exactly that as the maximum over
20 000 scattered launches.  Truth error against a direct ray trace, by
``|u|`` shell (max over the shell):

    |u| shell   count   max |ds1| [m]   max |dPhi| [waves]
    0.00-0.25    1371     1.6051e-10        6.2201e-06
    0.25-0.50    3968     1.6466e-10        3.6663e-06
    0.50-0.75    6660     1.4418e-10        3.0955e-06
    0.75-0.90    5354     1.2584e-10        2.8841e-06
    0.90-0.95    2021     1.8047e-10        4.0605e-06
    0.95-0.99     626     1.3290e-10        3.0280e-06
    0.99-1.00       0     (unreachable -- the 5% normaliser pad)

ratio (0.90-0.95)/(0.25-0.50) = 1.10x on s1 and 1.11x on Phi.  The
extrapolation danger is therefore NOT at the traced-ray edge; it is in
the 99.00% of the normalised box the training set never visits (W6-A17)
and in ``propagate_hf_chebyshev_quadrature``'s unguarded grids (W6-A16).

``_eval_4d_cross_deriv`` is CLEAN: all four ``(axis_a, axis_b)`` pairs
agree with central differences of ``HFPolyFit.eval_phi`` to <= 4.14e-07
absolute, i.e. <= 1.31e-06 relative to the largest analytic value
(3.164652e-01).  (A per-element relative metric reports 1.0 on the two
mixed x-y pairs only because they are numerically zero by the singlet's
rotational symmetry -- max|analytic| 5.11e-04 vs 3.16e-01 on the
like-axis pairs.)  ``eval_van_vleck_density`` returns ``sqrt(|det|)``;
on this fit ``det`` was positive at 200/200 scattered probes, so no sign
information is being discarded in practice.

``fit_hf_polynomials`` builds its design matrix with a Python loop while
``fit_canonical_polynomials`` uses the fancy-index form.  They are
BIT-IDENTICAL, and the loop is 4.3x FASTER at 4096x210 (0.0946 s vs
0.2847 s for 20 builds) because the vectorised form materialises a
``(n_basis, n_rays)`` temporary -- so the "vectorised is better"
premise is REFUTED at this size.


W6-A1 -- the theorem (this is what makes the unwrap indefensible)
-----------------------------------------------------------------
``_compute_M_b_batch`` builds

    M       = M_real - i pi H_phi
    M_real  = J^T J / w_s^2 + I / w_p^2

so ``Re M`` is STRICTLY positive definite for any finite ``w_p > 0`` (the
``I / w_p^2`` term alone is).  Factor ``M = R^{1/2} (I - i K) R^{1/2}``
with ``R = Re M`` and ``K = R^{-1/2} (pi H_phi) R^{-1/2}`` real symmetric
with eigenvalues ``k1, k2``.  Then

    det M     = det(R) (1 - i k1)(1 - i k2)
    arg det M = -atan(k1) - atan(k2)   in (-pi, +pi)  STRICTLY

because ``det R > 0`` and each ``atan`` lies in ``(-pi/2, +pi/2)``.  So
``det M`` NEVER reaches the principal branch cut: the principal ``sqrt``
is the unique globally analytic continuation, the Maslov index is
identically zero, and the physical pi/2 phase advance through a fold
caustic arrives CONTINUOUSLY as ``arg det M`` sweeps ``-pi -> 0``.

MEASURED (stock N-BK7 singlet fit of
``validation/propagators/test_asymptotic.py``, 17x17 output grid):

  * ``min eig Re M`` = 1.0158e+06 (w_s=20 um, w_p=0.02) / 1.6253e+05
    (w_s=50 um, w_p=0.05) -- never near zero;
  * the identity ``arg det M == -atan(k1) - atan(k2)`` holds to
    8.06e-14 / 7.77e-15 / 1.28e-15 across three waist settings;
  * ``max |arg det M|`` = 1.3152 / 1.5213 / 0.2660 rad, i.e. 1.83 / 1.62 /
    2.88 rad of headroom below pi.

What the raster unwrap actually detects is UNDERSAMPLING.  Weaken the
Gaussian regularisation and ``arg det M`` hugs +-pi (measured
``max |arg| = 3.141024`` at w_s=1e-2, w_p=0.02, i.e. ``pi - 5.7e-4``); its
sign then flips between adjacent pixels, giving a spurious
``|d_arg| ~ 2 pi`` (measured max jump 6.2810 vs ``2 pi = 6.2832``) and a
spurious factor of ``-1``.  Measured flip counts on the singlet, 17x17:

    w_s      w_p     max|arg|   max jump   flipped / in-box
    2e-05    0.05     1.3261     2.2821      0 / 281
    5e-05    0.05     1.5213     3.0137      0 / 281
    2e-04    0.05     2.2724     3.6497      8 / 281   <-- FIRES
    2e-04    0.10     2.2231     3.6521     14 / 281
    1e-03    0.10     2.9738     5.8779     18 / 289
    1e-02    1.00     3.1399     6.2781     18 / 289

and the count grows with the grid: at w_s=1e-2, w_p=50 the same fit gives
0/81 (n=9), 8/281 (n=17), 130/1073 (n=33), 648/4097 (n=65).

END-TO-END DAMAGE (w_s=1e-3, w_p=0.1, singlet):
  * 65x65 grid: ``'row_reset'`` differs from ``'principal'`` on 742 of
    4209 non-zero pixels, EVERY difference an exact sign flip, with
    ``max |dE| / max |E| = 1.012`` -- the brightest pixels are hit.
  * The SAME output point returns E on one grid and -E on another: of the
    289 points shared between a 17x17 and a 65x65 grid, 60 come back
    sign-flipped (relative difference exactly 2.000) under
    ``'row_reset'``, while ``'principal'`` reproduces them to 3.419e-11.
  * At the library-default waists all three modes are BIT-IDENTICAL
    (``max |dE| = 0.0``), so the default change is a no-op on
    well-regularised input.

W6-A2 -- MEASURED.  On the same singlet/grid the cold-start residual has
median 2.035e+07 (w_s=20 um, w_p=0.02); Newton drives it to median
6.799e-09, a RELATIVE reduction of 3.868e-16 (full machine precision), and
yet ``rn < tol = 1e-12`` held for exactly 1 of 289 pixels.  Post-fix the
verdict is ``rn < tol * max(r0, 1)`` and reads 257/289 with EVERY returned
``v2*`` bit-for-bit identical (2312/2312 values, ``max |delta| = 0.0``).
Counts across four settings, pre -> post: 1->257, 3->257, 1->265, 1->85.

W6-A4 -- MEASURED.  Amplitudes agree to 1.411e-08 between
``extract_linear_phase=True`` and ``False`` fits of the same system (pure
lstsq noise); the phase differs by a CONSTANT 2.5454 rad (spread 1.0e-06
rad) on a refractive singlet -- pure piston -- and by a genuine SPREAD of
5.369 rad = 0.8545 waves once a 2 um first-order grating puts
``a1 = 2.0172e+03`` waves of diffracted tilt into the ramp.  The
v2-linear part is negligible on every case measured
(``|a3| + |a4| <= 1.71e-09`` waves, amplitude impact <= 3.20e-11), so the
removal is phase-only.  NOT a bug -- removing piston and tilt before
quoting a wavefront is the standard Strehl/Seidel convention -- but it was
undocumented at the public API and is now pinned.
"""

from __future__ import annotations

import functools
import math
import warnings

import numpy as np
import pytest
from numpy.polynomial import chebyshev as _cheb

import lumenairy as la
from lumenairy.elements.lenses import _multi_indices_total_degree
from lumenairy.propagators.asymptotic import (
    CanonicalPolyFit,
    _compute_M_b_batch,
    _solve_envelope_stationary_batch,
    fit_canonical_polynomials,
    lg_polynomial,
    propagate_modal_asymptotic,
)

WL = 1.31e-6

# ===========================================================================
# The real-lens fixture -- the same singlet the validation suite fits
# (validation/propagators/test_asymptotic.py::_build_test_singlet).
# ===========================================================================


@functools.lru_cache(maxsize=2)
def _fit(n_field=8, n_pupil=8, poly_order=6):
    pres = la.make_singlet(51.5e-3, np.inf, 4.1e-3, 'N-BK7', aperture=12.0e-3)
    pres['object_distance'] = 200e-3
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fit_canonical_polynomials(
            pres, wavelength=WL, source_box_half=20e-6, pupil_box_half=0.02,
            n_field=n_field, n_pupil=n_pupil, poly_order=poly_order)


def _grid(fit, n, frac=0.9):
    L = fit.s2x_halfrange * frac
    ax = np.linspace(-L, L, n) + fit.s2x_centre
    return np.meshgrid(ax, ax, indexing='xy')


def _M_at(fit, X, Y, w_s, w_p, src=(0.0, 0.0)):
    """(M, valid, v2*) at the envelope-stationary point of every pixel."""
    v_c = (fit.v2x_centre, fit.v2y_centre)
    fx, fy = X.ravel(), Y.ravel()
    vx, vy, conv = _solve_envelope_stationary_batch(
        fit, fx, fy, src[0], src[1], w_s=w_s, w_p=w_p,
        v_cx=v_c[0], v_cy=v_c[1])
    M = _compute_M_b_batch(fit, fx, fy, vx, vy, src[0], src[1],
                           w_s, w_p, v_c[0], v_c[1])[0]
    u1 = (fx - fit.s2x_centre) / fit.s2x_halfrange
    u2 = (fy - fit.s2y_centre) / fit.s2y_halfrange
    u3 = (vx - fit.v2x_centre) / fit.v2x_halfrange
    u4 = (vy - fit.v2y_centre) / fit.v2y_halfrange
    valid = ((np.abs(u1) <= 1.0) & (np.abs(u2) <= 1.0)
             & (np.abs(u3) <= 1.0) & (np.abs(u4) <= 1.0))
    return M, valid, (vx, vy), conv


def _field(fit, X, Y, w_s, w_p, tracking=None, src=(0.0, 0.0)):
    kw = {} if tracking is None else {'maslov_tracking': tracking}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return propagate_modal_asymptotic(
            fit, source_point=src, w_s=w_s, w_p=w_p,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y, **kw)


def _field_amp(fit, X, Y, src_amp, pup_amp, w_s=20e-6, w_p=0.02):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return propagate_modal_asymptotic(
            fit, source_point=(0.0, 0.0), w_s=w_s, w_p=w_p,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            source_amplitudes=src_amp, pupil_amplitudes=pup_amp,
            s2_grid_x=X, s2_grid_y=Y, maslov_tracking='principal')


# ===========================================================================
# Synthetic CanonicalPolyFit -- a fit whose Phi is an EXACT low-order
# polynomial with a diallable phase scale, so a brute-force quadrature of
# the same integrand converges at a few hundred Gauss-Legendre nodes.
#
# A real-lens fit carries ~600 waves of residual Phi across the v2 box
# (measured: min -586.30, max +29.33 waves at s2 = 0 on the singlet
# above), which would need O(10^4) nodes per axis.
# ===========================================================================

_SYN_ORDER = 4
_SYN_MI = _multi_indices_total_degree(4, _SYN_ORDER)
_SYN_IDX = {tuple(m): j for j, m in enumerate(_SYN_MI)}
_MON2CHEB = [_cheb.poly2cheb([0.0] * n + [1.0]) for n in range(_SYN_ORDER + 1)]


def _coef_from_monomials(terms):
    """{(a,b,c,d): value} on monomials u1^a u2^b u3^c u4^d -> Chebyshev."""
    out = np.zeros(len(_SYN_MI), dtype=np.float64)
    for (a, b, c, d), val in terms.items():
        for k1, w1 in enumerate(_MON2CHEB[a]):
            if w1 == 0.0:
                continue
            for k2, w2 in enumerate(_MON2CHEB[b]):
                if w2 == 0.0:
                    continue
                for k3, w3 in enumerate(_MON2CHEB[c]):
                    if w3 == 0.0:
                        continue
                    for k4, w4 in enumerate(_MON2CHEB[d]):
                        if w4 == 0.0:
                            continue
                        out[_SYN_IDX[(k1, k2, k3, k4)]] += (
                            val * w1 * w2 * w3 * w4)
    return out


def _make_synth(phi_terms, s1x_terms, s1y_terms):
    """All centres 0 and all halfranges 1, so u_i IS the coordinate."""
    return CanonicalPolyFit(
        poly_order=_SYN_ORDER, multi_indices=_SYN_MI,
        coef_phi=_coef_from_monomials(phi_terms),
        coef_s1x=_coef_from_monomials(s1x_terms),
        coef_s1y=_coef_from_monomials(s1y_terms),
        s2x_centre=0.0, s2x_halfrange=1.0,
        s2y_centre=0.0, s2y_halfrange=1.0,
        v2x_centre=0.0, v2x_halfrange=1.0,
        v2y_centre=0.0, v2y_halfrange=1.0,
        wavelength=1.0e-6, linear_coeffs_phi=None,
        extract_linear_phase=False)


def _poly2d(coeffs, x, y):
    out = np.zeros(np.broadcast(x, y).shape, dtype=np.complex128)
    for (i, j), c in coeffs.items():
        out = out + c * x ** i * y ** j
    return out


def _quad_oracle(fit, s2x, s2y, *, src, w_s, w_p, v_c, v_star, n, half):
    """Brute-force Gauss-Legendre quadrature of the SAME phase-space
    integral the asymptotic engine approximates:

        E(s2) = int d^2 v |det J(s2,v)| Esrc(s1(s2,v) - src)
                          Apup(v - vc) exp(2 pi i Phi(s2,v))

    (the engine freezes ``det J`` at ``v*``, linearises ``s1`` about
    ``v*`` and truncates the exponent at quadratic order -- nothing else).
    ``half`` is the half-width of the integration window in v2 units and
    MUST keep the window inside the fit box, or the oracle integrates an
    extrapolated Chebyshev polynomial and is worthless.
    """
    t, wg = np.polynomial.legendre.leggauss(n)
    VX, VY = np.meshgrid(v_star[0] + half * t, v_star[1] + half * t,
                         indexing='xy')
    W = np.outer(wg, wg) * (half * half)
    S2X = np.full(VX.shape, float(s2x))
    S2Y = np.full(VX.shape, float(s2y))
    s1x, s1y, jxx, jxy, jyx, jyy = fit.eval_s1_with_v2_grad(S2X, S2Y, VX, VY)
    detJ = np.abs(jxx * jyy - jxy * jyx)
    phi = fit.eval_phi(S2X, S2Y, VX, VY, include_linear=False)
    rx, ry = s1x - src[0], s1y - src[1]
    dvx, dvy = VX - v_c[0], VY - v_c[1]
    integ = (detJ
             * _poly2d(lg_polynomial(0, 0, w_s), rx, ry)
             * np.exp(-(rx * rx + ry * ry) / (w_s * w_s))
             * _poly2d(lg_polynomial(0, 0, w_p), dvx, dvy)
             * np.exp(-(dvx * dvx + dvy * dvy) / (w_p * w_p))
             * np.exp(2j * math.pi * phi))
    return complex(np.sum(integ * W))


# ===========================================================================
# W6-A1 -- the theorem
# ===========================================================================

@pytest.mark.parametrize('w_s,w_p,min_eig_floor', [
    (20e-6, 0.02, 1.0e+06),
    (50e-6, 0.05, 1.0e+05),
    (20e-6, 0.002, 1.0e+07),
])
def test_w6_a1_re_M_positive_definite_and_arg_detM_strictly_inside_the_cut(
        w_s, w_p, min_eig_floor):
    """GREEN pre- and post-fix -- the ANCHOR the whole W6-A1 verdict rests
    on.  ``Re M = J^T J / w_s^2 + I / w_p^2`` is positive definite, hence

        arg det M = -atan(k1) - atan(k2)  in (-pi, +pi) STRICTLY

    with ``k1, k2`` the eigenvalues of ``R^{-1/2}(pi H_phi)R^{-1/2}``.
    Measured identity residual 8.06e-14 / 7.77e-15 / 1.28e-15 and
    ``min eig Re M`` = 1.0158e+06 / 1.6253e+05 / 1.8572e+07 for the three
    waist settings.
    """
    fit = _fit()
    X, Y = _grid(fit, 17)
    M, valid, _, _ = _M_at(fit, X, Y, w_s, w_p)
    R = np.real(M)
    Hpi = -np.imag(M)                     # == pi * H_phi
    eig_R = np.linalg.eigvalsh(R[valid])
    assert eig_R.min() > min_eig_floor, (
        f'Re M must be positive definite by construction; min eigenvalue '
        f'{eig_R.min():.4e} (measured floor {min_eig_floor:.1e})')

    det_M = (M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0])[valid]
    args = np.angle(det_M)
    assert np.all(np.abs(args) < math.pi), (
        f'arg det M reached the branch cut: max |arg| = '
        f'{np.abs(args).max():.9f} vs pi = {math.pi:.9f}')

    ks = []
    for i in np.where(valid)[0]:
        lam, V = np.linalg.eigh(R[i])
        Rm = V @ np.diag(lam ** -0.5) @ V.T
        ks.append(np.linalg.eigvalsh(Rm @ Hpi[i] @ Rm))
    ks = np.asarray(ks)
    resid = np.max(np.abs(args - (-np.arctan(ks[:, 0]) - np.arctan(ks[:, 1]))))
    assert resid < 1e-11, (
        f'arg det M != -atan(k1) - atan(k2): residual {resid:.3e} '
        f'(measured 8.06e-14 worst of three settings).  If this fails the '
        f'positive-definite factorisation no longer holds and the whole '
        f'W6-A1 argument must be re-derived.')


def test_w6_a1_default_tracking_makes_the_field_grid_independent():
    """PRE-FIX RED.  ``E(s2)`` is a POINTWISE functional of the fit -- an
    integral of an entire integrand against a Gaussian -- so it cannot
    depend on which other pixels were requested in the same call.  The
    pre-v5.30 default ``maslov_tracking='row_reset'`` made it depend on
    the raster neighbours: at ``w_s = 1e-3, w_p = 0.1`` on the stock
    singlet, 60 of the 289 output points shared between a 17x17 and a
    65x65 grid came back with the OPPOSITE SIGN on the two grids
    (relative difference exactly 2.000).  Post-fix (default
    ``'principal'``) the same points agree to 3.419e-11.
    """
    fit = _fit()
    w_s, w_p = 1e-3, 0.1
    n_c, n_f = 17, 65
    step = (n_f - 1) // (n_c - 1)
    Xc, Yc = _grid(fit, n_c)
    Xf, Yf = _grid(fit, n_f)
    assert np.allclose(Xf[0, ::step], Xc[0]), 'grid subset premise broken'

    Ec = _field(fit, Xc, Yc, w_s, w_p)                # DEFAULT tracking
    Ef = _field(fit, Xf, Yf, w_s, w_p)[::step, ::step]
    nz = (np.abs(Ec) > 0) & (np.abs(Ef) > 0)
    assert int(nz.sum()) > 200, f'need shared pixels, got {int(nz.sum())}'
    rel = np.abs(Ec[nz] - Ef[nz]) / np.abs(Ec[nz])
    n_flip = int(np.sum(np.abs(Ec[nz] + Ef[nz]) < 1e-9 * np.abs(Ec[nz])))
    assert n_flip == 0, (
        f'{n_flip} of {int(nz.sum())} shared output points changed SIGN '
        f'between a {n_c}x{n_c} and a {n_f}x{n_f} grid (measured pre-fix '
        f'60/289).  The Maslov raster unwrap must not reach the default '
        f'path: E(s2) is pointwise.')
    assert rel.max() < 1e-8, (
        f'default-tracking field is grid-dependent: max rel diff '
        f'{rel.max():.3e} between the two grids (measured post-fix '
        f'3.419e-11, pre-fix 2.000).')


def test_w6_a1_default_is_the_principal_branch():
    """PRE-FIX RED.  The default must BE ``'principal'`` bit-for-bit on a
    case where the legacy unwrap fires (pre-fix the default was
    ``'row_reset'``, which differs on 742 of 4209 non-zero pixels at
    ``w_s = 1e-3, w_p = 0.1``, 65x65)."""
    fit = _fit()
    w_s, w_p = 1e-3, 0.1
    X, Y = _grid(fit, 65)
    E_def = _field(fit, X, Y, w_s, w_p)
    E_pri = _field(fit, X, Y, w_s, w_p, 'principal')
    E_row = _field(fit, X, Y, w_s, w_p, 'row_reset')
    nz = np.abs(E_pri) > 0
    n_diff = int(np.sum(np.abs(E_row[nz] - E_pri[nz]) > 0))
    assert n_diff > 300, (
        f'premise check: this probe is only meaningful because the legacy '
        f'unwrap FIRES here; it changed only {n_diff} pixels (measured '
        f'742/4209).')
    assert np.array_equal(E_def, E_pri), (
        'propagate_modal_asymptotic default maslov_tracking must be '
        "'principal' -- the raster unwrap is provably spurious for this M "
        '(see the module docstring theorem).')
    # every legacy difference is an exact sign flip, and it hits the peak
    flipped = np.abs(E_row[nz] + E_pri[nz]) < 1e-9 * np.abs(E_pri[nz])
    assert int(flipped.sum()) == n_diff, (
        'every legacy-vs-principal difference must be an exact factor of '
        f'-1; {n_diff - int(flipped.sum())} were not')
    assert (np.max(np.abs(E_row - E_pri))
            / np.max(np.abs(E_pri))) > 0.9, (
        'the spurious flips hit the brightest pixels (measured '
        'max|dE|/max|E| = 1.012)')


def test_w6_a1_legacy_mode_warns_when_it_flips_a_pixel():
    """PRE-FIX RED -- the legacy modes are retained for reproducing
    pre-v5.30 output, but silently returning 742 sign-flipped pixels is
    what made this defect invisible for five minor versions."""
    fit = _fit()
    X, Y = _grid(fit, 65)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        propagate_modal_asymptotic(
            fit, source_point=(0.0, 0.0), w_s=1e-3, w_p=0.1,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y, maslov_tracking='row_reset')
    hits = [w for w in rec
            if issubclass(w.category, RuntimeWarning)
            and 'spurious sign flip' in str(w.message)]
    assert hits, (
        "maslov_tracking='row_reset' sign-flipped pixels without warning "
        '(measured 742 of 4209 in-box pixels at w_s=1e-3, w_p=0.1)')


def test_w6_a1_default_path_is_silent_on_a_well_regularised_case():
    """SCOPE GUARD (GREEN pre- and post-fix).  At the library-default
    waists all three tracking modes are BIT-IDENTICAL (measured
    ``max |dE| = 0.0``), so the default change is a no-op there and
    nothing may warn."""
    fit = _fit()
    X, Y = _grid(fit, 17)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        E_def = propagate_modal_asymptotic(
            fit, source_point=(0.0, 0.0), w_s=50e-6, w_p=0.05,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y)
    assert not [w for w in rec if 'maslov' in str(w.message).lower()], (
        'the default path must never warn about the Maslov branch')
    for mode in ('principal', '1d_raster', 'row_reset'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E = propagate_modal_asymptotic(
                fit, source_point=(0.0, 0.0), w_s=50e-6, w_p=0.05,
                v2_centre=(fit.v2x_centre, fit.v2y_centre),
                s2_grid_x=X, s2_grid_y=Y, maslov_tracking=mode)
        assert np.array_equal(E, E_def), (
            f'at the library-default waists maslov_tracking={mode!r} must '
            f'be bit-identical to the default (measured max |dE| = 0.0)')


# ===========================================================================
# W6-A2 -- the Newton convergence verdict
# ===========================================================================

@pytest.mark.parametrize('w_s,w_p,floor', [
    (20e-6, 0.02, 200),
    (50e-6, 0.05, 200),
    (20e-6, 0.002, 200),
])
def test_w6_a2_converged_verdict_is_scale_relative(w_s, w_p, floor):
    """PRE-FIX RED.  ``_solve_envelope_stationary_batch`` compared a
    DIMENSIONAL residual against an ABSOLUTE ``tol = 1e-12``.  Measured on
    the stock singlet, 17x17: the cold-start residual has median 2.035e+07
    and Newton drives it to median 6.799e-09 -- a relative reduction of
    3.868e-16, i.e. full machine precision -- yet ``rn < 1e-12`` held for
    exactly 1 of 289 pixels (3/289 at w_s=50 um, 1/289 at w_p=0.002).
    Post-fix the verdict is ``rn < tol * max(r0, 1)`` and reads
    257 / 257 / 265 of 289.
    """
    fit = _fit()
    X, Y = _grid(fit, 17)
    _, valid, _, conv = _M_at(fit, X, Y, w_s, w_p)
    assert int(conv.sum()) >= floor, (
        f'converged verdict True on only {int(conv.sum())}/{conv.size} '
        f'pixels ({int(valid.sum())} in-box).  tol is an ABSOLUTE '
        f'tolerance on a DIMENSIONAL residual -- see the '
        f'_solve_envelope_stationary_batch docstring.')


def test_w6_a2_newton_really_does_reach_machine_precision():
    """GREEN pre- and post-fix -- the measurement that proves W6-A2 is a
    VERDICT bug and not a solver bug.  The residual reduction relative to
    the cold-start value is at machine precision (measured median
    3.868e-16), so the pixels the pre-fix flag called "failed" had in fact
    converged perfectly."""
    fit = _fit()
    w_s, w_p = 20e-6, 0.02
    v_c = (fit.v2x_centre, fit.v2y_centre)
    X, Y = _grid(fit, 17)
    fx, fy = X.ravel(), Y.ravel()

    def _rn(vx, vy):
        s1x, s1y, jxx, jxy, jyx, jyy = fit.eval_s1_with_v2_grad(fx, fy, vx, vy)
        J = np.stack([np.stack([jxx, jxy], -1), np.stack([jyx, jyy], -1)], -2)
        ds = np.stack([s1x, s1y], axis=1)
        dv = np.stack([vx - v_c[0], vy - v_c[1]], axis=1)
        r = (np.einsum('kij,kj->ki', np.swapaxes(J, -1, -2), ds) / w_s ** 2
             + dv / w_p ** 2)
        return np.linalg.norm(r, axis=1)

    vx, vy, _ = _solve_envelope_stationary_batch(
        fit, fx, fy, 0.0, 0.0, w_s=w_s, w_p=w_p, v_cx=v_c[0], v_cy=v_c[1])
    r0 = _rn(np.full_like(fx, v_c[0]), np.full_like(fx, v_c[1]))
    r1 = _rn(vx, vy)
    _, valid, _, _ = _M_at(fit, X, Y, w_s, w_p)
    frac = np.median(r1[valid] / r0[valid])
    assert np.median(r0[valid]) > 1e6, (
        f'premise: the residual is dimensional and large at the cold start '
        f'(measured median 2.035e+07), got {np.median(r0[valid]):.3e}')
    assert frac < 1e-13, (
        f'median relative residual reduction {frac:.3e} (measured 3.868e-16 '
        f'-- machine precision).  If this regresses, the Newton solver '
        f'itself has a problem and W6-A2 is not just a verdict bug.')


def test_w6_a2_v2_star_is_untouched_by_the_verdict_fix():
    """SCOPE GUARD.  The verdict fix must not move the expansion point.
    ``done`` still keys off the ORIGINAL absolute ``rn < tol`` test, so
    every returned ``v2*`` is bit-for-bit pre-fix (verified against a
    3a1da2b worktree: 2312/2312 values identical, max |delta| = 0.0).
    Here we re-pin the hard numbers at the fit centre so a future change
    to the iteration is caught even without the worktree.
    """
    fit = _fit()
    vx, vy, _ = _solve_envelope_stationary_batch(
        fit, np.array([fit.s2x_centre]), np.array([fit.s2y_centre]),
        0.0, 0.0, w_s=20e-6, w_p=0.02,
        v_cx=fit.v2x_centre, v_cy=fit.v2y_centre)
    # On-axis source, on-axis image, rotationally symmetric singlet:
    # the envelope-stationary pupil point is the pupil centre.
    assert abs(float(vx[0])) < 1e-15 and abs(float(vy[0])) < 1e-15, (
        f'v2* at the fit centre moved: ({float(vx[0]):.6e}, '
        f'{float(vy[0]):.6e}) -- expected (0, 0) by symmetry')


# ===========================================================================
# W6-A3 -- exact-quadrature oracle for the leading-order engine
# ===========================================================================

_SYN_S1X = {(1, 0, 0, 0): 0.5, (0, 0, 1, 0): 0.30}
_SYN_S1Y = {(0, 1, 0, 0): 0.5, (0, 0, 0, 1): 0.30}


def _smooth_synth(k):
    """Smooth, non-caustic, quadratic-dominant phase with a controlled
    cubic perturbation -- the leading-order error is driven by that cubic.
    """
    return _make_synth(
        {(0, 0, 2, 0): 0.5 * k, (0, 0, 0, 2): 0.5 * k,
         (1, 0, 1, 0): 0.25 * k, (0, 1, 0, 1): 0.25 * k,
         (0, 0, 3, 0): 0.05 * k, (0, 0, 0, 3): 0.05 * k},
        _SYN_S1X, _SYN_S1Y)


_A3_W_S, _A3_W_P = 1.0, 0.10
_A3_S2 = (0.30, -0.20)


def _a3_probe(fit):
    """Engine value, converged oracle value, and the window bookkeeping.

    The quadrature window is sized on the EFFECTIVE Gaussian width
    ``sigma = 1/sqrt(2 Re M11)`` realised at the expansion point and is
    asserted to fit inside the fit box -- if it does not, the oracle
    integrates an extrapolated Chebyshev polynomial and is worthless
    (that mistake put a 1.3e-08 floor on the first version of this pin).
    """
    src, v_c = (0.0, 0.0), (0.0, 0.0)
    sx, sy = _A3_S2
    vx, vy, _ = _solve_envelope_stationary_batch(
        fit, np.array([sx]), np.array([sy]), src[0], src[1],
        w_s=_A3_W_S, w_p=_A3_W_P, v_cx=v_c[0], v_cy=v_c[1])
    M = _compute_M_b_batch(fit, np.array([sx]), np.array([sy]), vx, vy,
                           src[0], src[1], _A3_W_S, _A3_W_P,
                           v_c[0], v_c[1])[0]
    sigma = 1.0 / math.sqrt(2.0 * float(np.real(M[0, 0, 0])))
    room = 1.0 - max(abs(float(vx[0])), abs(float(vy[0])))
    half = 9.0 * sigma
    assert half < room, (
        f'quadrature window 9 sigma = {half:.4f} must fit inside the fit '
        f'box (room {room:.4f}); measured sigma = 0.0707, room = 1.000')
    got = complex(_field(fit, np.array([[sx]]), np.array([[sy]]),
                         _A3_W_S, _A3_W_P, 'principal', src=src)[0, 0])
    kw = dict(src=src, w_s=_A3_W_S, w_p=_A3_W_P, v_c=v_c,
              v_star=(float(vx[0]), float(vy[0])), half=half)
    lo = _quad_oracle(fit, sx, sy, n=360, **kw)
    ref = _quad_oracle(fit, sx, sy, n=520, **kw)
    conv = abs(ref - lo) / abs(ref)
    assert conv < 1e-11, (
        f'oracle not converged: 360 vs 520 nodes differ by {conv:.3e} '
        f'(measured 5.5e-14)')
    return got, ref, abs(got - ref) / abs(ref)


@pytest.mark.parametrize('k,lo_band,hi_band', [
    (1.0, 1.0e-06, 1.5e-05),
    (3.0, 1.0e-05, 1.2e-04),
    (10.0, 1.0e-04, 1.2e-03),
    (30.0, 4.0e-04, 5.0e-03),
    (100.0, 2.5e-03, 3.0e-02),
])
def test_w6_a3_engine_matches_brute_force_quadrature_of_its_own_integrand(
        k, lo_band, hi_band):
    """GREEN pre- and post-fix -- the ORACLE for the requested item (1).
    Brute-force Gauss-Legendre quadrature of the SAME phase-space integral
    the engine approximates, on a synthetic fit whose ``Phi`` is an exact
    polynomial with phase scale ``k`` waves.  The engine's only
    approximations are (a) freezing ``det J`` at ``v*``, (b) linearising
    ``s1`` about ``v*``, (c) truncating the exponent at quadratic order --
    so the residual IS the leading-order error and must track the
    neglected cubic (coefficient ``0.05 k`` on ``u3^3`` and ``u4^3``).

    MEASURED (w_s = 1.0, w_p = 0.10, probe s2 = (0.30, -0.20), 9-sigma
    window, oracle self-converged to 5.5e-14):

      k     |E| engine            |E| oracle            amp_rel    dphase     TOTAL
      1   1.73869205145004e-02  1.73868537177617e-02  3.842e-06  +2.47e-07  3.850e-06
      3   1.72106077113484e-02  1.72100308821811e-02  3.352e-05  +6.49e-06  3.414e-05
      10  1.54554218883423e-02  1.54512777352766e-02  2.682e-04  +1.82e-04  3.239e-04
      30  8.68815539017747e-03  8.68205744673534e-03  7.024e-04  +1.10e-03  1.307e-03
      100 2.54797554568797e-03  2.53871715618953e-03  3.647e-03  +6.99e-03  7.893e-03

    Fitted order: d(log error)/d(log k) = +1.99 / +1.87 / +1.27 / +1.49
    over the four intervals -- i.e. the leading-order error grows ~k^2 in
    this construction (the effective width is set by w_p until pi k
    overtakes 1/w_p^2, so BOTH the cubic coefficient and the phase weight
    grow with k).  Independently reproduced at w_p = 0.06:
    4.867e-07 / 4.366e-06 / 4.671e-05 / 3.036e-04 / 3.024e-03, slopes
    +2.00 / +1.97 / +1.70 / +1.91.

    BOTH ends are pinned: the upper band catches an accuracy regression,
    the lower band catches an oracle that has stopped being independent.
    """
    got, ref, tot = _a3_probe(_smooth_synth(k))
    assert tot < hi_band, (
        f'k={k}: engine vs exact quadrature total rel = {tot:.3e} exceeds '
        f'the measured band {hi_band:.1e}; |E| engine {abs(got):.14e} vs '
        f'oracle {abs(ref):.14e}, phase {float(np.angle(got / ref)):+.3e} '
        f'rad')
    assert tot > lo_band, (
        f'k={k}: engine and quadrature agree to {tot:.3e}, BELOW the '
        f'measured leading-order truncation {lo_band:.1e} -- the oracle '
        f'has probably stopped being independent of the engine')


@pytest.mark.parametrize('k', [10.0, 30.0])
def test_w6_a3_closed_form_contraction_is_exact_for_a_quadratic_exponent(k):
    """GREEN pre- and post-fix -- the anchor that says the W6-A3 band above
    measures the ASYMPTOTIC TRUNCATION and nothing else.  Delete the cubic
    term and the exponent is exactly quadratic, so the Wick /
    Gaussian-moment contraction is EXACT and the residual must fall to
    quadrature precision.

    MEASURED at w_p = 0.10: engine vs quadrature 6.778e-14 (k=10) and
    6.765e-14 (k=30), amplitude 6.77e-14 / 6.77e-14 and phase
    -5.9e-16 / +2.2e-15 rad -- i.e. the oracle's own 5.5e-14 noise floor.
    (At w_p = 0.06: 6.809e-14 / 6.788e-14.)
    """
    quadratic_only = _make_synth(
        {(0, 0, 2, 0): 0.5 * k, (0, 0, 0, 2): 0.5 * k,
         (1, 0, 1, 0): 0.25 * k, (0, 1, 0, 1): 0.25 * k},
        _SYN_S1X, _SYN_S1Y)
    _got, _ref, rel = _a3_probe(quadratic_only)
    assert rel < 1e-12, (
        f'with an exactly-quadratic exponent the Gaussian-moment formula '
        f'is EXACT; engine vs quadrature rel = {rel:.3e} (measured '
        f'6.78e-14).  A failure here means the closed-form contraction '
        f'itself is wrong, not the asymptotic truncation.')


# ===========================================================================
# W6-A9 -- the BATCHED MULTI-MODE polynomial algebra vs the same oracle
# ===========================================================================
#
# The LG_00 x LG_00 channel exercises none of
# ``_batched_polynomial_substitute_linear_2d``,
# ``_batched_polynomial_under_affine_shift``, ``_poly_dict_to_array`` or the
# moment contraction beyond order 0 -- the polynomial reduces to a constant
# whose zeroth Wick moment is 1.  These probes drive real higher-order
# source and pupil modes through the batched path.

_A9_MODE_PAIRS = [
    ({(0, 0): 1.0}, {(0, 0): 1.0}, 'LG00 x LG00'),
    ({(1, 0): 1.0}, {(0, 0): 1.0}, 'LG10 x LG00'),
    ({(0, 0): 1.0}, {(1, 0): 1.0}, 'LG00 x LG10'),
    ({(0, 1): 1.0}, {(0, 0): 1.0}, 'LG0+1 x LG00'),
    ({(0, -1): 1.0}, {(0, 1): 1.0}, 'LG0-1 x LG0+1'),
    ({(2, 0): 1.0}, {(0, 2): 1.0}, 'LG20 x LG0+2'),
    ({(1, 1): 1.0}, {(1, -1): 1.0}, 'LG1+1 x LG1-1'),
    ({(0, 0): 1.0, (1, 0): 0.4 - 0.2j, (0, 1): -0.3j},
     {(0, 0): 1.0, (0, -1): 0.5 + 0.1j}, 'mixed x mixed'),
    ({(3, 0): 1.0}, {(0, 3): 1.0}, 'LG30 x LG0+3'),
]


def _a9_quad(fit, v_star, half, n, src_modes, pup_modes):
    src, v_c = (0.0, 0.0), (0.0, 0.0)
    sx, sy = _A3_S2
    t, wg = np.polynomial.legendre.leggauss(n)
    VX, VY = np.meshgrid(v_star[0] + half * t, v_star[1] + half * t,
                         indexing='xy')
    W = np.outer(wg, wg) * (half * half)
    S2X = np.full(VX.shape, float(sx))
    S2Y = np.full(VX.shape, float(sy))
    s1x, s1y, jxx, jxy, jyx, jyy = fit.eval_s1_with_v2_grad(S2X, S2Y, VX, VY)
    detJ = np.abs(jxx * jyy - jxy * jyx)
    phi = fit.eval_phi(S2X, S2Y, VX, VY, include_linear=False)
    rx, ry = s1x - src[0], s1y - src[1]
    dvx, dvy = VX - v_c[0], VY - v_c[1]
    Es = np.zeros(VX.shape, dtype=np.complex128)
    for k, a in src_modes.items():
        Es = Es + a * _poly2d(lg_polynomial(k[0], k[1], _A3_W_S), rx, ry)
    Ap = np.zeros(VX.shape, dtype=np.complex128)
    for k, b in pup_modes.items():
        Ap = Ap + b * _poly2d(lg_polynomial(k[0], k[1], _A3_W_P), dvx, dvy)
    integ = (detJ * Es * np.exp(-(rx * rx + ry * ry) / (_A3_W_S ** 2))
             * Ap * np.exp(-(dvx * dvx + dvy * dvy) / (_A3_W_P ** 2))
             * np.exp(2j * math.pi * phi))
    return complex(np.sum(integ * W))


def _a9_run(fit, src_modes, pup_modes):
    src, v_c = (0.0, 0.0), (0.0, 0.0)
    sx, sy = _A3_S2
    vx, vy, _ = _solve_envelope_stationary_batch(
        fit, np.array([sx]), np.array([sy]), src[0], src[1],
        w_s=_A3_W_S, w_p=_A3_W_P, v_cx=v_c[0], v_cy=v_c[1])
    M = _compute_M_b_batch(fit, np.array([sx]), np.array([sy]), vx, vy,
                           src[0], src[1], _A3_W_S, _A3_W_P,
                           v_c[0], v_c[1])[0]
    half = 9.0 / math.sqrt(2.0 * float(np.real(M[0, 0, 0])))
    got = complex(_field_amp(fit, np.array([[sx]]), np.array([[sy]]),
                             src_modes, pup_modes,
                             w_s=_A3_W_S, w_p=_A3_W_P)[0, 0])
    vs = (float(vx[0]), float(vy[0]))
    lo = _a9_quad(fit, vs, half, 360, src_modes, pup_modes)
    ref = _a9_quad(fit, vs, half, 540, src_modes, pup_modes)
    assert abs(ref - lo) < 1e-11 * abs(ref), (
        f'oracle not converged: {abs(ref - lo) / abs(ref):.3e}')
    return got, ref, abs(got - ref) / abs(ref)


def test_w6_a9_batched_multimode_algebra_is_exact_for_a_quadratic_exponent():
    """GREEN pre- and post-fix -- the previously-UNREACHED batched
    polynomial interior.  With an exactly-quadratic ``Phi`` and ``s1``
    linear in ``v2``, the whole leading-order construction is EXACT (the
    Gaussian factor exactly, the polynomial prefactor exactly under the
    linear substitution), so every mode pair must reproduce the
    brute-force quadrature to quadrature precision.

    MEASURED worst over nine mode pairs up to total order 6
    (LG_{3,0} x LG_{0,+3}): 4.665e-14, against an oracle self-convergence
    floor of 5.8e-14.  Per-case: 4.464e-14 / 4.462e-14 / 4.403e-14 /
    4.459e-14 / 4.465e-14 / 4.436e-14 / 4.356e-14 / 4.459e-14 /
    4.665e-14.
    """
    k = 10.0
    fit = _make_synth(
        {(0, 0, 2, 0): 0.5 * k, (0, 0, 0, 2): 0.5 * k,
         (1, 0, 1, 0): 0.25 * k, (0, 1, 0, 1): 0.25 * k},
        _SYN_S1X, _SYN_S1Y)
    worst = 0.0
    detail = []
    for src_modes, pup_modes, label in _A9_MODE_PAIRS:
        _got, _ref, rel = _a9_run(fit, src_modes, pup_modes)
        detail.append((label, rel))
        worst = max(worst, rel)
    assert worst < 1e-12, (
        f'the batched multi-mode contraction must be EXACT for a quadratic '
        f'exponent; worst rel {worst:.3e} (measured 4.665e-14).  '
        f'Per-case: {detail}')


def test_w6_a9_leading_order_error_grows_sharply_with_mode_order():
    """GREEN pre- and post-fix -- pins a MEASURED, UNDOCUMENTED accuracy
    property.  Once the cubic is present the leading-order truncation
    error is NOT uniform across channels: it grows by ~2.5 orders from the
    piston channel to an order-6 channel, because the higher Wick moments
    weight the tails of the Gaussian where the neglected cubic dominates.

    MEASURED at k = 10, w_s = 1.0, w_p = 0.10, s2 = (0.30, -0.20):

        LG00  x LG00    3.239e-04
        LG10  x LG00    3.238e-04
        LG00  x LG10    1.738e-03
        LG0+1 x LG00    7.879e-04
        LG0-1 x LG0+1   9.575e-03
        LG20  x LG0+2   4.110e-02
        LG1+1 x LG1-1   1.744e-02
        mixed x mixed   2.285e-03
        LG30  x LG0+3   1.173e-01     <-- 362x the piston channel

    Anyone reading a high-order channel off this propagator (e.g. an LG
    aberration merit targeting (3, 0)) needs that number.
    """
    k = 10.0
    fit = _make_synth(
        {(0, 0, 2, 0): 0.5 * k, (0, 0, 0, 2): 0.5 * k,
         (1, 0, 1, 0): 0.25 * k, (0, 1, 0, 1): 0.25 * k,
         (0, 0, 3, 0): 0.05 * k, (0, 0, 0, 3): 0.05 * k},
        _SYN_S1X, _SYN_S1Y)
    rels = {}
    for src_modes, pup_modes, label in _A9_MODE_PAIRS:
        rels[label] = _a9_run(fit, src_modes, pup_modes)[2]
    assert 1e-4 < rels['LG00 x LG00'] < 1e-3, (
        f'piston channel {rels["LG00 x LG00"]:.3e} (measured 3.239e-04)')
    assert 3e-2 < rels['LG30 x LG0+3'] < 4e-1, (
        f'order-6 channel {rels["LG30 x LG0+3"]:.3e} (measured 1.173e-01)')
    ratio = rels['LG30 x LG0+3'] / rels['LG00 x LG00']
    assert ratio > 50.0, (
        f'the high-order channel must be MUCH worse than the piston '
        f'channel (measured 362x); got {ratio:.1f}.  All rels: {rels}')


# ===========================================================================
# W6-A6 -- through-caustic behaviour (requested item 2), verified NUMERICALLY
# ===========================================================================
#
# Synthetic FOLD caustic in the Airy normal form:
#
#     Phi = k (u3^3/3 - u1 u3 + u4^3/3 - u2 u4)
#     s1  = 0.5 u1 + 0.30 u3          (and the same in y)
#
# The phase Hessian d2Phi/du3^2 = 2 k u3 vanishes exactly on u3 = 0.  With
# w_s = 0.02 the SOURCE term dominates the envelope, so the
# envelope-stationary point u3*(u1) tracks -(a/b) u1 and sweeps THROUGH
# zero -- a genuine fold-Hessian sign change, not a static degenerate
# point.

def _fold_synth(k, a=0.5, b=0.30):
    return _make_synth(
        {(0, 0, 3, 0): k / 3.0, (1, 0, 1, 0): -k,
         (0, 0, 0, 3): k / 3.0, (0, 1, 0, 1): -k},
        {(1, 0, 0, 0): a, (0, 0, 1, 0): b},
        {(0, 1, 0, 0): a, (0, 0, 0, 1): b})


def _fold_sweep(k, us, w_s=0.02, w_p=0.10):
    fit = _fold_synth(k)
    fx = np.asarray(us, dtype=float)
    fy = np.zeros_like(fx)
    vx, vy, _ = _solve_envelope_stationary_batch(
        fit, fx, fy, 0.0, 0.0, w_s=w_s, w_p=w_p, v_cx=0.0, v_cy=0.0)
    M = _compute_M_b_batch(fit, fx, fy, vx, vy, 0.0, 0.0,
                           w_s, w_p, 0.0, 0.0)[0]
    det_M = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    E = _field(fit, fx.reshape(1, -1), fy.reshape(1, -1), w_s, w_p,
               'principal').ravel()
    H33 = -np.imag(M[:, 0, 0]) / math.pi          # the real phase Hessian
    return dict(arg=np.angle(det_M), H33=H33,
                k1=math.pi * H33 / np.real(M[:, 0, 0]), E=E, v3=vx)


def test_w6_a6_field_stays_finite_and_continuous_through_a_fold_caustic():
    """GREEN pre- and post-fix -- REFUTES the premise that a caustic makes
    this evaluator diverge.  ``Re M`` is positive definite, so the
    Gaussian regularisation keeps ``det M`` away from zero even where the
    PHASE Hessian vanishes identically; there is no singularity for a
    branch index to repair.

    MEASURED (k = 400, w_s = 0.02, w_p = 0.10, u1 in [-0.30, +0.30], 13
    samples): ``H33`` sweeps +2.769e+02 -> 0 (exactly, at u1 = 0) ->
    -2.769e+02, every ``|E|`` is finite and non-zero, ``arg det M`` runs
    -1.213278 -> 0 -> +1.213278 (max |arg| = 1.213278, i.e. 1.93 rad below
    pi), it is strictly monotone, and the largest jump between adjacent
    samples is 0.419642 rad -- so the raster unwrap CANNOT fire anywhere
    on the sweep.
    """
    us = np.linspace(-0.30, 0.30, 13)
    r = _fold_sweep(400.0, us)
    assert r['H33'][0] * r['H33'][-1] < 0, (
        f'premise: the phase Hessian must change SIGN across the sweep '
        f'(measured +2.769e+02 -> -2.769e+02), got {r["H33"][0]:.3e} -> '
        f'{r["H33"][-1]:.3e}')
    assert abs(r['H33'][len(us) // 2]) < 1e-9, (
        f'the caustic must sit exactly at u1 = 0 (H33 = 0), got '
        f'{r["H33"][len(us) // 2]:.3e}')
    assert np.all(np.isfinite(r['E'])), 'field went non-finite at a caustic'
    assert np.all(np.abs(r['E']) > 0), 'field was zeroed at a caustic'
    assert np.all(np.abs(r['arg']) < math.pi), (
        f'arg det M reached the cut: max |arg| = {np.abs(r["arg"]).max():.9f}')
    d = np.diff(r['arg'])
    assert np.all(d > 0), 'arg det M must advance monotonically through the fold'
    assert float(np.abs(d).max()) < math.pi, (
        f'largest adjacent jump {float(np.abs(d).max()):.6f} rad must stay '
        f'below pi so no unwrap can fire (measured 0.419642)')


def test_w6_a6_accuracy_is_best_ON_the_caustic_and_collapses_off_it():
    """GREEN pre- and post-fix -- pins a MEASURED, UNDOCUMENTED accuracy
    envelope.  The evaluator expands about the ENVELOPE-stationary point
    ``v*``, not about the complex saddle of the FULL exponent.  Right on
    the caustic the two coincide and the answer is good; a few effective
    widths off it they separate, the answer becomes dominated by
    ``exp(b_quad)``, and the leading-order value under-predicts by orders
    of magnitude.

    MEASURED (fold fit, w_s = 0.05, w_p = 0.10, sigma = 0.0606, 9-sigma
    quadrature window, oracle self-converged to 5.8e-14 at the points
    quoted):

        k    u1=0 (ON caustic)   |u1| = 1.2 sigma   |u1| = 2.5 sigma
        30       2.916e-03        2.8e-02/2.6e-02    3.4e-04/7.2e-03
        60       1.137e-02        4.4e-03/1.3e-02    4.4e-01/9.6e-01
        150      6.211e-02        4.2e-01/1.8e-01    9.8e-01/9.96e-01

    Worst off-caustic case measured with a still-converged oracle
    (k = 150, u1 = -0.10, oracle conv 2.5e-11): engine 2.258e-08 vs truth
    1.250e-06, i.e. a factor 55 UNDER-prediction.  Fixing that needs a
    genuine complex-saddle / steepest-descent evaluator, which is a
    feature, not a repair -- this pin exists so the limitation is on
    record and any improvement is visible.
    """
    for k, band_on in ((30.0, 1.0e-2), (60.0, 3.0e-2), (150.0, 1.5e-1)):
        r = _fold_sweep(k, np.array([0.0]), w_s=0.05, w_p=0.10)
        fit = _fold_synth(k)
        vs = (float(r['v3'][0]), 0.0)
        M = _compute_M_b_batch(fit, np.array([0.0]), np.array([0.0]),
                               np.array([vs[0]]), np.array([vs[1]]),
                               0.0, 0.0, 0.05, 0.10, 0.0, 0.0)[0]
        half = 9.0 / math.sqrt(2.0 * float(np.real(M[0, 0, 0])))
        got = complex(r['E'][0])
        kw = dict(src=(0.0, 0.0), w_s=0.05, w_p=0.10, v_c=(0.0, 0.0),
                  v_star=vs, half=half)
        lo = _quad_oracle(fit, 0.0, 0.0, n=360, **kw)
        ref = _quad_oracle(fit, 0.0, 0.0, n=540, **kw)
        assert abs(ref - lo) < 1e-11 * abs(ref), (
            f'k={k}: oracle not converged ({abs(ref - lo) / abs(ref):.3e})')
        rel = abs(abs(got) - abs(ref)) / abs(ref)
        assert rel < band_on, (
            f'k={k}: ON the caustic the engine must still be accurate; '
            f'amplitude rel {rel:.3e} exceeds the measured band '
            f'{band_on:.1e} (measured 2.916e-03 / 1.137e-02 / 6.211e-02)')


def test_w6_a6_principal_branch_delivers_the_kmah_pi_over_2():
    """GREEN pre- and post-fix -- the NUMERICAL end-to-end confirmation the
    audit asked for.  The Keller-Maslov pi/2 through one fold is produced
    by the PRINCIPAL branch of ``sqrt(det M)`` alone: as the Gaussian
    regularisation is weakened (``k`` up, so ``|k1| = |pi H33| / Re M11``
    up), the half-sweep of ``arg det M`` across the caustic converges to
    pi/2 from BELOW and never overshoots into a wrap.

    MEASURED (u1 = -0.30 -> +0.30, w_s = 0.02, w_p = 0.10):

        k         k1(-0.3)     k1(+0.3)    arg(-0.3)   arg(+0.3)  halfsweep/(pi/2)
        4e+01      +0.268       -0.268     -0.261554   +0.261554     0.166510
        4e+02      +2.677       -2.677     -1.213278   +1.213278     0.772397
        4e+03     +26.769      -26.769     -1.533456   +1.533456     0.976229
        4e+04    +267.686     -267.686     -1.567061   +1.567061     0.997622
        4e+05   +2676.860    -2676.860     -1.570423   +1.570423     0.999762
        4e+06  +26768.600   -26768.600     -1.570759   +1.570759     0.999976

    i.e. ``-atan(k1(-)) - atan(k1(+)) -> pi`` and half of it -> pi/2,
    exactly as the ``arg det M = -atan(k1) - atan(k2)`` identity predicts.
    Nothing here needs a branch counter.
    """
    ks = (4e1, 4e2, 4e3, 4e4, 4e5, 4e6)
    got = []
    for k in ks:
        r = _fold_sweep(k, np.array([-0.30, 0.30]))
        assert np.all(np.abs(r['arg']) < math.pi)
        got.append(0.5 * float(r['arg'][1] - r['arg'][0]) / (math.pi / 2))
    assert all(0.0 < g < 1.0 for g in got), (
        f'the half-sweep must approach pi/2 from BELOW and never wrap: {got}')
    assert all(got[i] < got[i + 1] for i in range(len(got) - 1)), (
        f'the half-sweep must increase monotonically with k: {got}')
    assert got[0] < 0.2, (
        f'strongly-regularised end should be far from pi/2 (measured '
        f'0.166510), got {got[0]:.6f}')
    assert got[-1] > 0.9999, (
        f'weakly-regularised end must reach pi/2 (measured 0.999976), got '
        f'{got[-1]:.6f} -- if this drops, the KMAH phase is no longer '
        f'being delivered by the principal branch and the W6-A1 default '
        f'change would need revisiting')


# ===========================================================================
# W6-A4 -- the piston/tilt phase reference
# ===========================================================================

@functools.lru_cache(maxsize=4)
def _fit_extract(extract, grating_period=None):
    pres = la.make_singlet(51.5e-3, np.inf, 4.1e-3, 'N-BK7', aperture=12.0e-3)
    pres['object_distance'] = 200e-3
    sd = (None if grating_period is None
          else {1: (1.0, 0.0, grating_period, grating_period)})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fit_canonical_polynomials(
            pres, wavelength=WL, source_box_half=20e-6, pupil_box_half=0.02,
            n_field=6, n_pupil=6, poly_order=6,
            extract_linear_phase=extract, surface_diffraction=sd)


def test_w6_a4_extract_linear_phase_changes_the_phase_not_the_amplitude():
    """GREEN pre- and post-fix -- pins the CONVENTION that the whole family
    evaluates ``Phi`` with ``include_linear=False``
    (``asymptotic_maslov.py`` ``_compute_M_b_batch``,
    ``asymptotic_aberration_tensor.py`` ``_compute_M_b``,
    ``asymptotic_jax_twin.py`` ``_compute_M_b_xp``), so the returned phase
    is referenced to the fit's linear ramp.

    Refractive singlet: amplitudes agree to 5.94e-09; the phase difference
    is a CONSTANT 2.5454 rad with spread 1.0e-06 rad -- pure piston.
    """
    fa, fb = _fit_extract(True), _fit_extract(False)
    assert fa.linear_coeffs_phi is not None
    assert fb.linear_coeffs_phi is None
    X, Y = _grid(fa, 9, frac=0.6)
    Ea = _field(fa, X, Y, 20e-6, 0.02, 'principal')
    Eb = _field(fb, X, Y, 20e-6, 0.02, 'principal')
    nz = (np.abs(Ea) > 0) & (np.abs(Eb) > 0)
    amp = np.abs(np.abs(Ea[nz]) - np.abs(Eb[nz])) / np.abs(Eb[nz])
    assert amp.max() < 1e-7, (
        f'extract_linear_phase must not change the AMPLITUDE; max rel '
        f'{amp.max():.3e} (measured 5.94e-09 = lstsq noise)')
    dph = np.angle(Ea[nz] / Eb[nz])
    spread = float(dph.max() - dph.min())
    assert spread < 1e-4, (
        f'on a REFRACTIVE system the dropped ramp is a pure PISTON, so the '
        f'phase offset must be constant; spread {spread:.3e} rad '
        f'(measured 1.0e-06)')
    assert abs(float(np.median(dph))) > 1.0, (
        f'premise: the piston offset is real and large (measured 2.5454 '
        f'rad), got {float(np.median(dph)):.4f}')


def test_w6_a4_diffracted_tilt_is_removed_from_the_returned_phase():
    """GREEN pre- and post-fix.  A 2 um first-order grating puts
    ``a1 = 2.0172e+03`` waves of diffracted tilt into the extracted ramp;
    the evaluators drop it, so the returned phase acquires a genuine
    SPREAD of 5.369 rad (0.8545 waves) relative to the
    ``extract_linear_phase=False`` fit -- while the amplitude still agrees
    to 5.93e-09.  The v2-linear part stays negligible
    (``|a3| + |a4| = 6.23e-10`` waves), which is why the amplitude
    survives."""
    fa, fb = _fit_extract(True, 2e-6), _fit_extract(False, 2e-6)
    a0, a1, a2, a3, a4 = (float(c) for c in fa.linear_coeffs_phi)
    assert abs(a1) > 1e3, (
        f'premise: the grating must put a large output-plane tilt in the '
        f'ramp; a1 = {a1:.6e} waves (measured 2.0172e+03)')
    assert abs(a3) + abs(a4) < 1e-8, (
        f'the v2-linear part of the ramp must stay negligible or the '
        f'omission would corrupt the AMPLITUDE too; |a3| + |a4| = '
        f'{abs(a3) + abs(a4):.3e} waves (measured 6.23e-10)')
    X, Y = _grid(fa, 9, frac=0.6)
    Ea = _field(fa, X, Y, 20e-6, 0.02, 'principal')
    Eb = _field(fb, X, Y, 20e-6, 0.02, 'principal')
    nz = (np.abs(Ea) > 0) & (np.abs(Eb) > 0)
    amp = np.abs(np.abs(Ea[nz]) - np.abs(Eb[nz])) / np.abs(Eb[nz])
    assert amp.max() < 1e-7, (
        f'amplitude must survive the ramp removal; max rel {amp.max():.3e} '
        f'(measured 5.93e-09)')
    dph = np.angle(Ea[nz] / Eb[nz])
    spread = float(dph.max() - dph.min())
    assert spread > 1.0, (
        f'the dropped diffracted tilt must show up as a phase SPREAD '
        f'(measured 5.369 rad = 0.8545 waves), got {spread:.4f} rad -- if '
        f'this vanishes the evaluators started including the ramp and the '
        f'documented convention changed')


# ===========================================================================
# W6-A7 -- power accounting and linearity (requested item 6)
# ===========================================================================

def test_w6_a7_modal_propagator_is_exactly_linear_in_the_mode_amplitudes():
    """GREEN pre- and post-fix.  ``propagate_modal_asymptotic`` sums
    ``a_src * b_pup * <contraction>`` over mode pairs, so it must be
    EXACTLY linear in each amplitude dict.  Measured: doubling the source
    amplitude reproduces ``2 E`` BITWISE; a complex pupil amplitude
    ``3 - 1j`` reproduces ``(3 - 1j) E`` to 6.440e-16."""
    fit = _fit()
    X, Y = _grid(fit, 17, frac=0.5)
    E1 = _field_amp(fit, X, Y, {(0, 0): 1.0 + 0.0j}, None)
    E2 = _field_amp(fit, X, Y, {(0, 0): 2.0 + 0.0j}, None)
    E3 = _field_amp(fit, X, Y, {(0, 0): 1.0 + 0.0j}, {(0, 0): 3.0 - 1.0j})
    assert np.array_equal(E2, 2.0 * E1), (
        f'source-amplitude scaling must be bitwise exact; max |dE| = '
        f'{np.max(np.abs(E2 - 2.0 * E1)):.3e}')
    nz = np.abs(E1) > 0
    rel = np.max(np.abs(E3[nz] - (3.0 - 1.0j) * E1[nz]) / np.abs(E1[nz]))
    assert rel < 1e-14, (
        f'pupil-amplitude scaling rel error {rel:.3e} (measured 6.440e-16)')


def test_w6_a7_output_power_converges_under_grid_refinement():
    """GREEN pre- and post-fix.  ``sum |E|^2 dA`` over the whole in-box
    region is a Riemann sum of a smooth integrand and must converge.
    Measured at w_s = 20 um, w_p = 0.002 (a fully-resolved spot):
    1.058617786707e-10 at n = 65, 129, 257 and 513 -- successive relative
    changes 1.5e-14 / 3.3e-15 / 1.6e-15.  At w_p = 0.02 the spot is
    narrower than the 65-pixel grid and the same sequence reads
    5.126e-11 -> 5.125e-11 -> 5.128e-11 -> 5.128e-11 (worst step
    6.5e-04), i.e. still converging."""
    fit = _fit()
    P = []
    for n in (65, 129, 257):
        X, Y = _grid(fit, n, frac=0.995)
        E = _field(fit, X, Y, 20e-6, 0.002, 'principal')
        d = float(X[0, 1] - X[0, 0])
        P.append(float(np.sum(np.abs(E) ** 2) * d * d))
    assert abs(P[1] / P[0] - 1.0) < 1e-10 and abs(P[2] / P[1] - 1.0) < 1e-10, (
        f'output power must be grid-converged on a resolved spot: {P} '
        f'(measured 1.058617786707e-10 at all three n)')
    assert P[-1] > 0.0


def test_w6_a7_output_field_carries_no_radiometric_normalisation():
    """GREEN pre- and post-fix -- pins a DOCUMENTATION GAP, not a bug.
    ``propagate_modal_asymptotic`` returns the BARE phase-space integral:
    it applies no ``1/(i lambda z)``-class prefactor and never reads
    ``fit.wavelength`` at all (the wavelength enters only through ``Phi``,
    which the fit stores in waves).  So the returned amplitude is NOT on a
    conserved-power scale and is NOT on the same scale as
    ``propagate_hf_chebyshev_quadrature`` (which does apply an explicit
    ``* (-1j)`` Maslov factor plus ``pixel_area`` -- see the F-21 comment
    at the end of that function).

    MEASURED on a 257x257 grid over 0.995 of the fit box: both the source
    and the pupil LG_{0,0} carry unit L2 norm, yet ``sum |E|^2 dA`` comes
    out 1.657151e-11 (w_s=50 um, w_p=0.05) / 5.128074e-11 (20 um, 0.02) /
    1.058618e-10 (20 um, 0.002) / 8.532070e-11 (20 um, 2e-4) -- a factor
    6.4 spread with no fixed relation to the unit input.  Anything that
    needs absolute radiometry must supply its own prefactor.
    """
    fit = _fit()
    powers = {}
    for w_s, w_p in ((50e-6, 0.05), (20e-6, 0.02), (20e-6, 0.002)):
        X, Y = _grid(fit, 257, frac=0.995)
        E = _field(fit, X, Y, w_s, w_p, 'principal')
        d = float(X[0, 1] - X[0, 0])
        powers[(w_s, w_p)] = float(np.sum(np.abs(E) ** 2) * d * d)
    vals = sorted(powers.values())
    assert vals[0] < 1e-9, (
        f'premise: the returned field is nowhere near unit power '
        f'(measured 1.66e-11 .. 1.06e-10), got {vals}')
    assert vals[-1] / vals[0] > 3.0, (
        f'premise: P_out has no fixed relation to the unit-norm input '
        f'(measured spread factor 6.4), got {vals[-1] / vals[0]:.3f}')


# ===========================================================================
# W6-A8 -- NumPy vs JAX parity (requested item 5): MEASURED floor
# ===========================================================================

@pytest.mark.parametrize('n_field,n_pupil,poly_order,phi_floor,s1_floor', [
    (8, 8, 6, 1.0e-13, 1.0e-10),
    (4, 4, 4, 1.0e-13, 1.0e-10),
])
def test_w6_a8_backend_aware_evaluators_match_numpy(n_field, n_pupil,
                                                    poly_order, phi_floor,
                                                    s1_floor):
    """GREEN pre- and post-fix.  ``eval_phi_xp`` / ``eval_s1_xp`` route
    through ``_evaluate_polynomial_4d_xp`` (a ``tensordot`` over the basis
    axis) while the NumPy ``eval_phi`` / ``eval_s1`` use
    ``_evaluate_polynomial_4d``.  MEASURED worst relative disagreement
    over 150 uniform-random in-box probes:

        fit                            eval_phi    eval_s1   pixel eval
        R1=51.5mm od=200mm order 6    3.834e-15  1.023e-12    3.278e-10
        R1=20mm  od=100mm order 4     1.794e-16  8.839e-14    8.987e-13
        R1=51.5mm order 8, n=8        0.000e+00  6.573e-14    3.870e-13

    On NumPy inputs the two paths are BIT-IDENTICAL (measured 0.0), so the
    floor above is JAX's XLA reassociation, not a formula difference.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    fit = _fit(n_field, n_pupil, poly_order)
    rng = np.random.default_rng(7)
    worst_phi = worst_s1 = worst_np = 0.0
    for _ in range(150):
        u = rng.uniform(-1, 1, 4)
        s2x = fit.s2x_centre + u[0] * fit.s2x_halfrange
        s2y = fit.s2y_centre + u[1] * fit.s2y_halfrange
        v2x = fit.v2x_centre + u[2] * fit.v2x_halfrange
        v2y = fit.v2y_centre + u[3] * fit.v2y_halfrange
        args_np = tuple(np.asarray(v) for v in (s2x, s2y, v2x, v2y))
        args_jx = tuple(jnp.asarray(v) for v in (s2x, s2y, v2x, v2y))
        pn = float(fit.eval_phi(*args_np))
        worst_phi = max(worst_phi,
                        abs(pn - float(fit.eval_phi_xp(*args_jx)))
                        / max(abs(pn), 1e-30))
        worst_np = max(worst_np,
                       abs(pn - float(fit.eval_phi_xp(*args_np)))
                       / max(abs(pn), 1e-30))
        for a, b in zip(fit.eval_s1(*args_np), fit.eval_s1_xp(*args_jx)):
            worst_s1 = max(worst_s1, abs(float(a) - float(b))
                           / max(abs(float(a)), 1e-12))
    assert worst_np == 0.0, (
        f'on NUMPY inputs eval_phi_xp must be bit-identical to eval_phi; '
        f'worst rel {worst_np:.3e}')
    assert worst_phi < phi_floor, (
        f'eval_phi NumPy vs JAX worst rel {worst_phi:.3e} exceeds the '
        f'measured floor {phi_floor:.1e}')
    assert worst_s1 < s1_floor, (
        f'eval_s1 NumPy vs JAX worst rel {worst_s1:.3e} exceeds the '
        f'measured floor {s1_floor:.1e}')


def test_w6_a8_pixel_evaluator_parity_numpy_vs_jax():
    """GREEN pre- and post-fix.  Full-pixel parity between
    ``propagate_modal_asymptotic`` (NumPy, batched, principal branch) and
    ``propagate_modal_asymptotic_lg00_jax`` (JAX, vmap'd, principal
    branch by construction) on the LG_{0,0} channel, with ``v*`` supplied
    from the NumPy scalar solver so the two paths differ ONLY in the
    arithmetic.  MEASURED worst relative 3.278e-10 (median 1.248e-10) on
    the stock singlet; 8.987e-13 and 3.870e-13 on two other fits."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from lumenairy.propagators.asymptotic import (
        propagate_modal_asymptotic_lg00_jax,
        solve_envelope_stationary,
    )
    fit = _fit()
    v_c = (fit.v2x_centre, fit.v2y_centre)
    X, Y = _grid(fit, 5, frac=0.5)
    vg = np.zeros(X.shape + (2,))
    for iy in range(X.shape[0]):
        for ix in range(X.shape[1]):
            vv, _, _ = solve_envelope_stationary(
                fit, (X[iy, ix], Y[iy, ix]), (0.0, 0.0),
                w_s=20e-6, w_p=0.02, v2_centre=v_c)
            vg[iy, ix] = vv
    E_np = _field(fit, X, Y, 20e-6, 0.02, 'principal')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_jx = np.asarray(propagate_modal_asymptotic_lg00_jax(
            fit, X, Y, vg, source_point=(0.0, 0.0),
            w_s=20e-6, w_p=0.02, v2_centre=v_c))
    nz = np.abs(E_np) > 0
    assert int(nz.sum()) >= 20
    rel = np.abs(E_np[nz] - E_jx[nz]) / np.abs(E_np[nz])
    assert rel.max() < 1e-8, (
        f'NumPy vs JAX pixel-evaluator worst rel {rel.max():.3e} exceeds '
        f'the measured floor (3.278e-10)')


def test_w6_a10_jax_fit_survives_partial_vignetting():
    """GREEN pre- and post-fix -- REFUTES a NaN-propagation hypothesis and
    records the measured parity floor.  The NumPy fit SLICES the design
    matrix to the live rays; ``fit_canonical_polynomials_jax`` keeps every
    row and zeroes the dead ones (``A_w = A * w[:, None]``).  That would
    poison the QR with ``nan * 0`` if a dead ray's recorded position were
    non-finite -- verified it never is: over apertures from 10 mm down to
    1 mm, NO dead ray has a non-finite ``x``, ``y``, ``L``, ``M`` or
    ``opd``.

    MEASURED at aperture 9 mm (384 of 576 rays alive), ``n_field=4,
    n_pupil=6, poly_order=4``: both fits finite; normalisers identical to
    2.2e-15 (s2x) / 2.9e-15 (v2x); fit residuals agree to 2.630e-06
    relative (1.4932369880e-06 NumPy vs 1.4932330614e-06 JAX), i.e. the
    two coefficient vectors describe the training data equally well;
    coefficient vectors 1.670e-04 apart on the scale of the largest
    coefficient -- the near-null-direction effect analysed in
    ``_differentiable_lstsq``.  (At full liveness the same number is
    1.588e-06, and 6.002e-07 at 512/576.)
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from lumenairy.propagators.asymptotic import fit_canonical_polynomials_jax
    pres = la.make_singlet(51.5e-3, np.inf, 4.1e-3, 'N-BK7', aperture=9e-3)
    pres['object_distance'] = 200e-3
    kw = dict(source_box_half=20e-6, pupil_box_half=0.02,
              n_field=4, n_pupil=6, poly_order=4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fn = fit_canonical_polynomials(pres, wavelength=WL, **kw)
        fj = fit_canonical_polynomials_jax(pres, wavelength=WL, **kw)
    assert 0 < int(fn.n_rays) < 576, (
        f'premise: this probe needs PARTIAL vignetting, got '
        f'{fn.n_rays}/576 alive (measured 384)')
    cj = np.asarray(fj.coef_phi)
    assert np.all(np.isfinite(cj)), (
        f'JAX fit returned {int(np.sum(~np.isfinite(cj)))} non-finite '
        f'coefficients under vignetting -- the row-weight masking let a '
        f'nan*0 through')
    for name in ('s2x_halfrange', 's2y_halfrange',
                 'v2x_halfrange', 'v2y_halfrange'):
        a = float(getattr(fn, name))
        b = float(getattr(fj, name))
        assert abs(b / a - 1.0) < 1e-12, (
            f'{name}: NumPy {a:.6e} vs JAX {b:.6e} (measured agreement '
            f'2.2e-15)')
    assert abs(float(fj.res_phi_rms_waves) / fn.res_phi_rms_waves
               - 1.0) < 1e-4, (
        f'fit residuals must match: NumPy {fn.res_phi_rms_waves:.10e} vs '
        f'JAX {float(fj.res_phi_rms_waves):.10e} (measured 2.630e-06 '
        f'relative)')
    scale = max(1e-30, float(np.max(np.abs(fn.coef_phi))))
    rel = float(np.max(np.abs(cj - fn.coef_phi))) / scale
    assert rel < 1e-3, (
        f'coefficient parity under vignetting {rel:.3e} exceeds the '
        f'measured 1.670e-04')


# ===========================================================================
# W6-A11 -- LG/HG mode-stack cache: indexing='xy' vs indexing='ij' collision
# ===========================================================================

def _lag_gen(p, a, x):
    out = np.zeros_like(x, dtype=np.float64)
    for k in range(p + 1):
        out = out + ((-1.0) ** k * math.comb(p + a, p - k)
                     / math.factorial(k)) * x ** k
    return out


def _lg_ref(p, ell, w, X, Y):
    """From-scratch polar-form LG oracle (the W3-T3 machinery, reused)."""
    a = abs(int(ell))
    r2 = X * X + Y * Y
    N = math.sqrt(2.0 * math.factorial(p)
                  / (math.pi * math.factorial(p + a))) / w
    radial = ((np.sqrt(2.0) * np.sqrt(r2) / w) ** a
              * _lag_gen(p, a, 2.0 * r2 / (w * w)))
    return (N * radial * np.exp(1j * ell * np.arctan2(Y, X))
            * np.exp(-r2 / (w * w))).astype(np.complex128)


_A11_MODES = [(0, 0), (0, 1), (1, 0), (1, 1), (0, -1)]


def _a11_setup():
    """Two genuinely DIFFERENT grids that the pre-fix key could not tell
    apart: same shape, same detected pitch, same (X[0,0], Y[0,0])."""
    n = 65
    x = np.linspace(-40e-6, 40e-6, n)
    y = np.linspace(-25e-6, 25e-6, n)      # different half-extent
    Xxy, Yxy = np.meshgrid(x, y, indexing='xy')
    Xij, Yij = np.meshgrid(x, y, indexing='ij')
    w = 12e-6

    def probe(X, Y):                        # asymmetric: coma on LG_{0,+1}
        u, v = X / w, Y / w
        return _lg_ref(0, 1, w, X, Y) * np.exp(
            1j * 0.9 * u * (u * u + v * v - 0.5))
    return (w, float(x[1] - x[0]), float(y[1] - y[0]),
            (Xxy, Yxy, probe(Xxy, Yxy)), (Xij, Yij, probe(Xij, Yij)))


def test_w6_a11_mode_stack_cache_key_separates_xy_from_ij_meshgrids():
    """PRE-FIX RED.  ``_lg_mode_conj_stack`` keyed on
    ``(p_max, ell_max, Ny, Nx, w, cx, cy, dx, dy, dtype, X.flat[0],
    Y.flat[0])``.  For an ``indexing='xy'`` grid and an ``indexing='ij'``
    grid built from the SAME two equal-length axes every one of those is
    identical -- ``_meshgrid_axis_step`` deliberately finds the varying
    axis so ``dx``/``dy`` match, and ``X[0,0] = x[0]``, ``Y[0,0] = y[0]``
    in both.  The stacks are transposes of each other, so the SECOND
    caller silently got the FIRST caller's modes.

    MEASURED on ``x = linspace(-40 um, 40 um, 65)``,
    ``y = linspace(-25 um, 25 um, 65)``, probe field
    ``LG_{0,+1} x exp(i 0.9 u (u^2+v^2-0.5))``:

        cache cleared between calls   xy 3.804e-15   ij 5.158e-15
        'ij' issued after 'xy'                       ij 8.232e+00
        cache entries after both      1  (=> one shared entry)

    a degradation factor of 1.596e+15.  Post-fix: 2 entries, warm error
    5.158e-15 == cold.  ``decompose_hg`` behaved identically.
    """
    from lumenairy.propagators.asymptotic_modes import (
        _LG_MODE_STACK_CACHE,
        clear_lg_mode_stack_cache,
        decompose_lg,
    )
    w, dx, dy, (Xxy, Yxy, Fxy), (Xij, Yij, Fij) = _a11_setup()

    def oracle(X, Y, F):
        return {k: complex(np.sum(np.conj(_lg_ref(k[0], k[1], w, X, Y)) * F)
                           * dx * dy) for k in _A11_MODES}

    ref_ij = oracle(Xij, Yij, Fij)

    clear_lg_mode_stack_cache()
    cold = decompose_lg(Fij, Xij, Yij, w=w, p_max=1, ell_max=1)
    err_cold = max(abs(cold[k] - ref_ij[k]) / abs(ref_ij[k])
                   for k in _A11_MODES)
    assert err_cold < 1e-12, (
        f'premise: the ij orientation must be RIGHT on a cold cache; '
        f'worst rel {err_cold:.3e} (measured 5.158e-15)')

    clear_lg_mode_stack_cache()
    decompose_lg(Fxy, Xxy, Yxy, w=w, p_max=1, ell_max=1)
    n_after_xy = len(_LG_MODE_STACK_CACHE)
    warm = decompose_lg(Fij, Xij, Yij, w=w, p_max=1, ell_max=1)
    n_after_ij = len(_LG_MODE_STACK_CACHE)
    err_warm = max(abs(warm[k] - ref_ij[k]) / abs(ref_ij[k])
                   for k in _A11_MODES)
    clear_lg_mode_stack_cache()

    assert n_after_ij == n_after_xy + 1, (
        f'the two orientations COLLIDED on one cache entry '
        f'({n_after_xy} -> {n_after_ij}); the key cannot distinguish '
        f"indexing='xy' from indexing='ij'")
    assert err_warm < 1e-12, (
        f"decompose_lg(indexing='ij') returned the CACHED 'xy' modes: "
        f'worst rel vs the independent overlap oracle {err_warm:.3e} '
        f'(measured pre-fix 8.232e+00, post-fix 5.158e-15)')


def test_w6_a11_hg_mode_stack_cache_key_separates_the_orientations():
    """PRE-FIX RED -- the HG twin of the above (measured 1 shared entry
    pre-fix, 2 post-fix)."""
    from lumenairy.propagators.asymptotic_modes import (
        _HG_MODE_STACK_CACHE,
        clear_lg_mode_stack_cache,
        decompose_hg,
    )
    _w, _dx, _dy, (Xxy, Yxy, Fxy), (Xij, Yij, Fij) = _a11_setup()
    wx, wy = 12e-6, 9e-6
    clear_lg_mode_stack_cache()
    decompose_hg(Fxy, Xxy, Yxy, wx, wy, 1, 1)
    n1 = len(_HG_MODE_STACK_CACHE)
    got = decompose_hg(Fij, Xij, Yij, wx, wy, 1, 1)
    n2 = len(_HG_MODE_STACK_CACHE)
    clear_lg_mode_stack_cache()
    ref = decompose_hg(Fij, Xij, Yij, wx, wy, 1, 1)
    clear_lg_mode_stack_cache()
    assert n2 == n1 + 1, (
        f'HG orientations collided on one cache entry ({n1} -> {n2})')
    worst = max(abs(got[k] - ref[k]) / max(abs(ref[k]), 1e-300)
                for k in ref)
    assert worst < 1e-12, (
        f'warm ij HG coefficients differ from the cold ones by '
        f'{worst:.3e} -- a cache collision')


def test_w6_a11_symmetric_square_grid_hides_the_collision():
    """GREEN pre- and post-fix -- explains WHY this survived the P3-53
    ``indexing='ij'`` fix.  On a SQUARE grid built from ONE axis,
    ``X_xy == X_ij.T`` bitwise, so with a rotationally symmetric field
    the transposed mode stack gives BIT-IDENTICAL coefficients (measured
    worst |delta| = 0.0 pre-fix).  Any regression test written on that
    geometry is self-fulfilling; the pin above deliberately uses
    different x and y half-extents AND an asymmetric field."""
    from lumenairy.propagators.asymptotic_modes import (
        clear_lg_mode_stack_cache,
        decompose_lg,
    )
    xs = np.linspace(-30e-6, 30e-6, 33)
    Sxy, Txy = np.meshgrid(xs, xs, indexing='xy')
    Sij, Tij = np.meshgrid(xs, xs, indexing='ij')
    assert np.array_equal(Sxy, Sij.T) and np.array_equal(Txy, Tij.T), (
        'premise: a square single-axis grid transposes exactly')
    w = 12e-6
    fsym = _lg_ref(0, 0, w, Sxy, Txy)
    clear_lg_mode_stack_cache()
    r1 = decompose_lg(fsym, Sxy, Txy, w=w, p_max=1, ell_max=1)
    r2 = decompose_lg(fsym, Sij, Tij, w=w, p_max=1, ell_max=1)
    clear_lg_mode_stack_cache()
    worst = max(abs(r1[k] - r2[k]) for k in r1)
    scale = max(abs(v) for v in r1.values())
    assert worst / scale < 1e-14, (
        f'premise: on this geometry a collision is invisible '
        f'(measured 0.0), got {worst / scale:.3e}')


def test_w6_a13_cached_mode_stack_is_read_only():
    """PRE-FIX RED.  The cached ``(keys, stack)`` tuple is returned BY
    IDENTITY on every hit, so a caller that writes into the array poisons
    the cache for the life of the process.  MEASURED pre-fix: setting
    ``stack[0, 0, 0] = 12345`` made the NEXT call return 12345 where the
    true mode value is 7.694599e-18.  Same class as the PMM geo-eig cache
    (adversarial audit M9); every in-library consumer only reads."""
    from lumenairy.propagators import asymptotic_modes as _am
    _am.clear_lg_mode_stack_cache()
    ax = np.linspace(-5e-5, 5e-5, 9)
    X, Y = np.meshgrid(ax, ax, indexing='xy')
    _keys, stack = _am._lg_mode_conj_stack(X, Y, 20e-6, 0, 0, 0.0, 0.0,
                                           1e-6, 1e-6)
    assert not stack.flags.writeable, (
        'the cached mode stack is writeable and is handed out by '
        'identity -- a caller mutation poisons the cache')
    with pytest.raises(ValueError):
        stack[0, 0, 0] = 12345.0
    _am.clear_lg_mode_stack_cache()


# ===========================================================================
# W6-A12 -- high-order mode evaluation: hypothesis REFUTED, band recorded
# ===========================================================================

@pytest.mark.parametrize('w', [1.0, 30e-6, 1e-9])
def test_w6_a12_monomial_basis_mode_evaluation_holds_to_high_order(w):
    """GREEN pre- and post-fix -- REFUTES a catastrophic-cancellation
    hypothesis.  ``lg_polynomial`` / ``hg_polynomial`` return CARTESIAN
    MONOMIAL coefficients carrying ``(sqrt(2)/w)**(2k)`` prefactors, and
    ``_evaluate_poly2d`` sums those monomials -- so high-order evaluation
    looked like a cancellation trap.  It is not, and the error is
    w-INDEPENDENT (the ``w`` powers cancel between the coefficient and
    the coordinate).

    MEASURED against a from-scratch polar/recurrence oracle on a 193^2
    grid over +-6w, as ``max|lib - oracle| / max|oracle|``
    (w = 1 / 30 um / 1e-9 give the same numbers to within a factor 2):

        LG (p, 0)  p=0..12 : 0.0, 3.5e-16 ... 4.1e-12, 1.2e-11
        LG (0, l)  l=0..12 : 0.0, 4.1e-16 ... 5.6e-15, 7.1e-15
        LG (p, p)  p=0..8  : 0.0, 6.4e-16 ... 4.4e-12, 1.8e-11
        HG (m, m)  m=0..14 : 1.4e-16 ...   1.8e-11, 5.7e-11

    and the LIBRARY modes' own Gram matrix stays orthonormal:
    max|G - I| = 1.73e-14 (p_max<=6), 2.73e-14 (7), 1.09e-13 (8),
    1.61e-13 (9), 1.03e-12 (10) for LG, and a flat 1.80e-14 for HG all
    the way to m_max = 10.
    """
    from lumenairy.propagators.asymptotic_modes import (
        evaluate_hg_mode,
        evaluate_lg_mode,
    )
    ax = np.linspace(-6.0 * w, 6.0 * w, 193)
    X, Y = np.meshgrid(ax, ax, indexing='xy')
    worst_lg = 0.0
    for (p, ell) in ([(p, 0) for p in range(13)]
                     + [(0, ell) for ell in range(13)]
                     + [(p, p) for p in range(9)]):
        ref = _lg_ref(p, ell, w, X, Y)
        got = evaluate_lg_mode(p, ell, w, X, Y)
        worst_lg = max(worst_lg, float(np.max(np.abs(got - ref)))
                       / float(np.max(np.abs(ref))))
    assert worst_lg < 1e-9, (
        f'LG monomial evaluation worst rel {worst_lg:.3e} up to total '
        f'order 16 (measured 1.9e-11)')

    def _hg_ref(m, n, wx, wy, Xg, Yg):
        def phi(k, u, ww):
            z = math.sqrt(2.0) * u / ww
            h0 = np.ones_like(z)
            if k == 0:
                h = h0
            else:
                h1 = 2.0 * z
                for kk in range(2, k + 1):
                    h0, h1 = h1, 2.0 * z * h1 - 2.0 * (kk - 1) * h0
                h = h1
            Nk = ((2.0 / (math.pi * ww * ww)) ** 0.25
                  / math.sqrt((2.0 ** k) * math.factorial(k)))
            return Nk * h * np.exp(-u * u / (ww * ww))
        return phi(m, Xg, wx) * phi(n, Yg, wy)

    worst_hg = 0.0
    for k in range(15):
        ref = _hg_ref(k, k, w, w, X, Y)
        got = evaluate_hg_mode(k, k, w, w, X, Y)
        worst_hg = max(worst_hg, float(np.max(np.abs(got - ref)))
                       / float(np.max(np.abs(ref))))
    assert worst_hg < 1e-9, (
        f'HG monomial evaluation worst rel {worst_hg:.3e} up to m=n=14 '
        f'(measured 5.7e-11)')


# ===========================================================================
# W6-A17 -- canonical-fit CONDITIONING interior (requested item 3)
# ===========================================================================

def _rebuild_design_matrix(poly_order, n_field=8, n_pupil=8):
    """Rebuild the exact ``A`` that ``fit_canonical_polynomials`` feeds to
    ``np.linalg.lstsq``, plus the realised normalised coordinates."""
    from lumenairy._math.chebyshev import chebyshev_vandermonde as _cv
    from lumenairy.elements.lenses import _fit_normaliser
    from lumenairy.raytrace import (
        _make_bundle,
        surfaces_from_prescription,
        trace,
    )
    pres = la.make_singlet(51.5e-3, np.inf, 4.1e-3, 'N-BK7', aperture=12e-3)
    pres['object_distance'] = 200e-3
    uf = np.cos(np.pi * (np.arange(n_field) + 0.5) / n_field)
    up = np.cos(np.pi * (np.arange(n_pupil) + 0.5) / n_pupil)
    S1X, S1Y, V1X, V1Y = np.meshgrid(uf * 20e-6, uf * 20e-6,
                                     up * 0.02, up * 0.02, indexing='ij')
    b = _make_bundle(x=S1X.ravel(), y=S1Y.ravel(), L=V1X.ravel(),
                     M=V1Y.ravel(), wavelength=WL)
    b.z = np.full(S1X.size, -200e-3)
    r = trace(b, surfaces_from_prescription(pres), WL, output_filter='last')
    f = r.image_rays
    al = np.asarray(f.alive, dtype=bool)
    obs = (np.asarray(f.x)[al], np.asarray(f.y)[al],
           np.asarray(f.L)[al], np.asarray(f.M)[al])
    u = [(v - c) / h for v, (c, h) in
         zip(obs, [_fit_normaliser(v) for v in obs])]
    mi = _multi_indices_total_degree(4, poly_order)
    K = np.asarray(mi, dtype=np.int64)
    T = [_cv(ui, poly_order) for ui in u]
    A = (T[0][K[:, 0]] * T[1][K[:, 1]] * T[2][K[:, 2]]
         * T[3][K[:, 3]]).T.astype(np.float64)
    return A, u, len(mi)


def test_w6_a17_default_canonical_fit_is_rank_deficient():
    """GREEN pre- and post-fix -- pins a MEASURED, previously-unrecorded
    conditioning fact about the LIBRARY DEFAULT.  The Chebyshev basis is
    orthogonal on Chebyshev-Gauss nodes in the INPUT ``(s1, v1)``
    coordinates, but the fit normalises the OUTPUT ``(s2, v2)``
    coordinates -- the ray-traced IMAGES of those nodes, which are not
    Chebyshev-distributed and which do not fill the box.

    MEASURED on the stock N-BK7 singlet (``n_field = n_pupil = 8``,
    4096 training rays):

        poly_order  n_basis  sigma_max   sigma_min   cond(A)     rank
             2          15   7.7335e+01  4.6537e-04  1.6618e+05   15/15
             4          70   9.0781e+01  2.6903e-09  3.3744e+10   70/70
             6         210   1.0674e+02  1.4423e-14  7.4005e+15  185/210
             8         495   1.2302e+02  2.4713e-15  4.9781e+16  376/495

    ``poly_order = 6`` is the LIBRARY DEFAULT, so the default fit is
    numerically rank-deficient by 25 of 210 directions at
    ``cond(A) = 7.4e+15`` (~1/eps).  ``np.linalg.lstsq(rcond=None)``
    silently truncates those directions and returns the minimum-norm
    solution, which is why the coefficients are NOT unique and why the
    NumPy-vs-JAX coefficient comparison in ``_differentiable_lstsq``
    lives entirely in the near-null space.  The residual is still tiny
    and the fit is still usable INSIDE the sampled manifold; the pin
    exists so nobody reads the coefficients as physical.

    Root cause, measured: the realised samples occupy 100 of the 10^4
    cells of a 10-bin-per-axis grid over ``[-1, 1]^4`` -- 1.00% -- and
    the basis Gram matrix on that distribution has
    ``max|off-diag| / min|diag| = 1.6640e+01`` against 3.1752e-14 for the
    same basis on a true 4-D Chebyshev product grid (whose cond is 4.0).
    """
    A, u, n_basis = _rebuild_design_matrix(6)
    s = np.linalg.svd(A, compute_uv=False)
    rank = int(np.linalg.matrix_rank(A))
    assert n_basis == 210 and A.shape[0] == 4096, (
        f'premise moved: A is {A.shape}, n_basis {n_basis}')
    assert s[0] / s[-1] > 1e14, (
        f'cond(A) = {s[0] / s[-1]:.4e} (measured 7.4005e+15)')
    assert rank < n_basis, (
        f'the default poly_order=6 fit should be numerically '
        f'rank-deficient; rank {rank}/{n_basis} (measured 185/210)')
    G = A.T @ A / A.shape[0]
    off = float(np.max(np.abs(G - np.diag(np.diag(G)))))
    assert off / float(np.min(np.abs(np.diag(G)))) > 1.0, (
        'the basis is far from orthogonal on the realised distribution '
        '(measured max|off|/min|diag| = 16.640)')
    cells = np.zeros((10, 10, 10, 10), dtype=bool)
    idx = [np.clip(((ui + 1) / 2 * 10).astype(int), 0, 9) for ui in u]
    cells[idx[0], idx[1], idx[2], idx[3]] = True
    frac = cells.sum() / 10000.0
    assert frac < 0.05, (
        f'the training samples cover {100 * frac:.2f}% of the normalised '
        f'box (measured 1.00%) -- everything else is extrapolation')


def test_w6_a17_endpoint_anchored_does_not_hurt_conditioning():
    """GREEN pre- and post-fix -- VERIFIES (and sharpens) the
    ``endpoint_anchored`` docstring claim that "the conditioning hit on
    the least-squares fit is negligible at typical poly_order (4 - 6)".
    Measured: it is not a hit at all, it is a small IMPROVEMENT --
    cond(A) 7.4005e+15 -> 6.4182e+15 at poly_order 6 (n=8x8) and
    3.2723e+10 -> 2.9197e+10 at poly_order 4 (n=6x6)."""
    from lumenairy._math.chebyshev import chebyshev_vandermonde as _cv
    from lumenairy.elements.lenses import _fit_normaliser
    from lumenairy.raytrace import (
        _make_bundle,
        surfaces_from_prescription,
        trace,
    )
    pres = la.make_singlet(51.5e-3, np.inf, 4.1e-3, 'N-BK7', aperture=12e-3)
    pres['object_distance'] = 200e-3
    conds = {}
    for anchored in (False, True):
        n = 8
        i = np.arange(n)
        uf = np.cos(np.pi * (i + 0.5) / n)
        if anchored:
            uf = uf / np.cos(np.pi / (2.0 * n))
        S1X, S1Y, V1X, V1Y = np.meshgrid(uf * 20e-6, uf * 20e-6,
                                         uf * 0.02, uf * 0.02, indexing='ij')
        b = _make_bundle(x=S1X.ravel(), y=S1Y.ravel(), L=V1X.ravel(),
                         M=V1Y.ravel(), wavelength=WL)
        b.z = np.full(S1X.size, -200e-3)
        r = trace(b, surfaces_from_prescription(pres), WL,
                  output_filter='last')
        f = r.image_rays
        al = np.asarray(f.alive, dtype=bool)
        obs = (np.asarray(f.x)[al], np.asarray(f.y)[al],
               np.asarray(f.L)[al], np.asarray(f.M)[al])
        u = [(v - c) / h for v, (c, h) in
             zip(obs, [_fit_normaliser(v) for v in obs])]
        mi = _multi_indices_total_degree(4, 6)
        K = np.asarray(mi, dtype=np.int64)
        T = [_cv(ui, 6) for ui in u]
        A = (T[0][K[:, 0]] * T[1][K[:, 1]] * T[2][K[:, 2]] * T[3][K[:, 3]]).T
        s = np.linalg.svd(A, compute_uv=False)
        conds[anchored] = float(s[0] / s[-1])
    assert conds[True] < 2.0 * conds[False], (
        f'endpoint_anchored must not blow up the conditioning: '
        f'{conds[False]:.4e} -> {conds[True]:.4e} (measured 7.4005e+15 '
        f'-> 6.4182e+15, an improvement)')


def test_w6_a17_analytic_v2_gradients_match_finite_differences():
    """GREEN pre- and post-fix.  ``eval_s1_with_v2_grad`` /
    ``eval_phi_with_v2_grad`` apply the chain-rule factor
    ``1/v2x_halfrange`` to the Chebyshev derivative tables, and with
    ``include_linear=True`` they add ``a3`` / ``a4`` to the du-gradient
    BEFORE that multiply -- which is the correct order (``d/du3`` of
    ``a3 u3`` is ``a3``, then ``d/dv2x = (1/hx) d/du3``).  Verified by
    central differences of the fit's own ``eval_s1`` / ``eval_phi`` over
    400 uniform-random in-box probes, with clean O(h^2) scaling:

        FD step (in u)   ds1/dv2      dPhi/dv2(lin=F)  dPhi/dv2(lin=T)
            1e-2         2.6000e-01     1.3789e-01       1.3789e-01
            1e-3         2.6000e-03     1.3790e-03       1.3790e-03
            1e-4         2.6010e-05     1.3785e-05       1.3772e-05
            1e-5         1.0868e-06     1.9520e-07       6.1690e-06
            1e-6         3.2825e-06     7.3856e-07       4.2183e-05

    (the 1e-5/1e-6 rows are FD roundoff, worse for ``include_linear=True``
    because the 1.5735e+05-wave piston has to cancel).
    """
    fit = _fit()
    rng = np.random.default_rng(11)
    u = rng.uniform(-0.85, 0.85, (4, 400))
    s2x = fit.s2x_centre + u[0] * fit.s2x_halfrange
    s2y = fit.s2y_centre + u[1] * fit.s2y_halfrange
    v2x = fit.v2x_centre + u[2] * fit.v2x_halfrange
    v2y = fit.v2y_centre + u[3] * fit.v2y_halfrange
    h3 = 1e-4 * fit.v2x_halfrange
    h4 = 1e-4 * fit.v2y_halfrange
    _, _, jxx, jxy, jyx, jyy = fit.eval_s1_with_v2_grad(s2x, s2y, v2x, v2y)
    sxp, syp = fit.eval_s1(s2x, s2y, v2x + h3, v2y), fit.eval_s1(
        s2x, s2y, v2x, v2y + h4)
    sxm, sym = fit.eval_s1(s2x, s2y, v2x - h3, v2y), fit.eval_s1(
        s2x, s2y, v2x, v2y - h4)
    for an, fd in ((jxx, (sxp[0] - sxm[0]) / (2 * h3)),
                   (jxy, (syp[0] - sym[0]) / (2 * h4)),
                   (jyx, (sxp[1] - sxm[1]) / (2 * h3)),
                   (jyy, (syp[1] - sym[1]) / (2 * h4))):
        rel = float(np.max(np.abs(an - fd) / (np.abs(an) + 1e-30)))
        assert rel < 1e-3, (
            f'ds1/dv2 analytic vs FD max rel {rel:.3e} at h=1e-4 in u '
            f'(measured 2.601e-05)')
    for lin in (False, True):
        _p, gx, gy = fit.eval_phi_with_v2_grad(s2x, s2y, v2x, v2y,
                                               include_linear=lin)
        pxp = fit.eval_phi(s2x, s2y, v2x + h3, v2y, include_linear=lin)
        pxm = fit.eval_phi(s2x, s2y, v2x - h3, v2y, include_linear=lin)
        pyp = fit.eval_phi(s2x, s2y, v2x, v2y + h4, include_linear=lin)
        pym = fit.eval_phi(s2x, s2y, v2x, v2y - h4, include_linear=lin)
        rx = float(np.max(np.abs((pxp - pxm) / (2 * h3) - gx)
                          / (np.abs(gx) + 1e-30)))
        ry = float(np.max(np.abs((pyp - pym) / (2 * h4) - gy)
                          / (np.abs(gy) + 1e-30)))
        assert max(rx, ry) < 1e-3, (
            f'dPhi/dv2 (include_linear={lin}) analytic vs FD max rel '
            f'{max(rx, ry):.3e} at h=1e-4 in u (measured 1.3785e-05 / '
            f'1.3772e-05).  A failure with lin=True and not lin=False '
            f'would mean the a3/a4 chain-rule order is wrong.')


# ===========================================================================
# W6-A15 -- auto_bump_threshold_waves overshoots max_auto_poly_order
# ===========================================================================

@pytest.mark.parametrize('start,cap', [(9, 10), (5, 6), (3, 4), (6, 10)])
def test_w6_a15_auto_bump_respects_max_auto_poly_order(start, cap):
    """PRE-FIX RED for the odd-parity rows.  The gate is
    ``poly_order < max_auto_poly_order`` but the bump was an
    unconditional ``+2``, so a cap of the opposite parity was OVERSHOT.
    MEASURED with a deliberately unreachable threshold (1e-12 waves) on
    the stock N-BK7 singlet: start=9 cap=10 returned poly_order=11;
    start=5 cap=6 returned 7; start=3 cap=4 returned 5.  Only
    start=6 cap=10 landed on the cap."""
    pres = la.make_singlet(51.5e-3, np.inf, 4.1e-3, 'N-BK7', aperture=12e-3)
    pres['object_distance'] = 200e-3
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f = fit_canonical_polynomials(
            pres, wavelength=WL, source_box_half=20e-6, pupil_box_half=0.02,
            n_field=6, n_pupil=6, poly_order=start,
            auto_bump_threshold_waves=1e-12, max_auto_poly_order=cap)
    assert f.poly_order <= cap, (
        f'auto-bump from {start} overshot max_auto_poly_order={cap}: '
        f'returned poly_order={f.poly_order}')
    assert f.poly_order == cap, (
        f'auto-bump should climb all the way to the cap when the '
        f'threshold is unreachable; got {f.poly_order}, cap {cap}')


# ===========================================================================
# W6-A16 -- propagate_hf_chebyshev_quadrature validity guards
# ===========================================================================

@functools.lru_cache(maxsize=1)
def _hf_fit():
    from lumenairy.propagators.asymptotic import fit_hf_polynomials
    pres = la.make_singlet(51.5e-3, np.inf, 4.1e-3, 'N-BK7', aperture=12e-3)
    pres['object_distance'] = 200e-3
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fit_hf_polynomials(pres, wavelength=WL, source_box_half=20e-6,
                                  pupil_box_half=0.02, n_field=6, n_pupil=6,
                                  poly_order=4)


def test_w6_a16_hf_quadrature_warns_when_the_grid_leaves_the_fit_box():
    """PRE-FIX RED.  ``propagate_hf_chebyshev_quadrature`` had NO validity
    guard of any kind: ``Phi`` is a Chebyshev tensor product evaluated at
    whatever normalised coordinate the caller's grid produces, and
    outside the box it is unconstrained extrapolation.

    MEASURED on the stock N-BK7 singlet HF fit (s1 half-box 2.0284e-05 m),
    unit input field, in-box reference ``max|E_out| = 6.879511e-03``:

        input grid  1.5x the s1 half-box -> 1.533764e-02  (2.229x)
        input grid  3x                   -> 5.836258e-02  (8.484x)
        input grid 10x                   -> 3.089184e-01  (44.90x)

    all finite, all silent.  (An OUTPUT grid outside the s2 box is milder
    -- 2x gives 0.895x and 8x gives 0.0888x -- but is equally unguarded.)
    """
    from lumenairy.propagators.asymptotic import (
        propagate_hf_chebyshev_quadrature,
    )
    hf = _hf_fit()
    n = 24
    base = np.linspace(-hf.s1x_halfrange, hf.s1x_halfrange, n) + hf.s1x_centre
    oy = (np.linspace(-0.5 * hf.s2x_halfrange, 0.5 * hf.s2x_halfrange, 8)
          + hf.s2x_centre)
    E = np.ones((n, n), dtype=np.complex128)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        ref = propagate_hf_chebyshev_quadrature(hf, E, base, base, oy, oy)
    assert not [w for w in rec if 'validity box' in str(w.message)], (
        'an in-box call must NOT warn')

    wide = np.linspace(-3.0 * hf.s1x_halfrange, 3.0 * hf.s1x_halfrange,
                       n) + hf.s1x_centre
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = propagate_hf_chebyshev_quadrature(hf, E, wide, wide, oy, oy)
    hits = [w for w in rec
            if issubclass(w.category, RuntimeWarning)
            and 'validity box' in str(w.message)]
    assert hits, (
        f'no warning when the input grid ran 3x outside the s1 box; '
        f'max|E_out| inflated from {np.max(np.abs(ref)):.6e} to '
        f'{np.max(np.abs(out)):.6e} (measured 8.484x)')
    assert np.max(np.abs(out)) / np.max(np.abs(ref)) > 2.0, (
        'premise: extrapolation must visibly inflate the answer '
        '(measured 8.484x)')


def test_w6_a16_hf_quadrature_warns_on_a_non_uniform_input_grid():
    """PRE-FIX RED.  ``dx = mean(diff(input_grid_x))`` silently accepts a
    non-uniform axis and uses the MEAN pitch as the area element for
    every sample.  MEASURED with a cubic-spaced axis over the same
    endpoints (true steps span 3.3343e-09 .. 4.8448e-06, mean
    1.7639e-06): accepted silently, answer 1.004x the uniform-grid
    value -- close only because the integrand is smooth."""
    from lumenairy.propagators.asymptotic import (
        propagate_hf_chebyshev_quadrature,
    )
    hf = _hf_fit()
    n = 24
    t = np.linspace(-1, 1, n) ** 3
    nu = hf.s1x_centre + hf.s1x_halfrange * t
    oy = (np.linspace(-0.5 * hf.s2x_halfrange, 0.5 * hf.s2x_halfrange, 8)
          + hf.s2x_centre)
    E = np.ones((n, n), dtype=np.complex128)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        propagate_hf_chebyshev_quadrature(hf, E, nu, nu, oy, oy)
    hits = [w for w in rec
            if issubclass(w.category, RuntimeWarning)
            and 'uniformly spaced' in str(w.message)]
    assert hits, (
        'a non-uniform input axis (steps spanning 1450x) was accepted '
        'silently and integrated with a single mean pitch')


# ===========================================================================
# W6-A5 -- shape / rank guard
# ===========================================================================

@pytest.mark.parametrize('shape', [(2, 3, 4), (2, 2, 2, 2)])
def test_w6_a5_rank_guard_names_the_offending_argument(shape):
    """PRE-FIX RED.  Pre-fix, an ndim >= 3 grid reached the internal
    row-major unpack ``Ny, Nx = s2x_arr.shape`` and raised a bare
    ``ValueError: too many values to unpack (expected 2, got 3)`` from the
    middle of the function -- no mention of which argument was wrong."""
    fit = _fit()
    with pytest.raises(ValueError, match=r's2_grid_x'):
        propagate_modal_asymptotic(
            fit, w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=np.zeros(shape), s2_grid_y=np.zeros(shape))


def test_w6_a5_low_rank_grids_still_work():
    """SCOPE GUARD -- 0-D, 1-D and 2-D grids must keep working, and on a
    1-D grid ``'row_reset'`` is INERT (there are no rows to reset, so it
    is bit-identical to ``'1d_raster'``)."""
    fit = _fit()
    v_c = (fit.v2x_centre, fit.v2y_centre)
    ax = np.linspace(-1, 1, 5) * fit.s2x_halfrange * 0.5 + fit.s2x_centre
    for gx, gy, want in ((np.array(fit.s2x_centre),
                          np.array(fit.s2y_centre), ()),
                         (ax, np.zeros_like(ax), (5,)),
                         (np.zeros((0,)), np.zeros((0,)), (0,))):
        out = propagate_modal_asymptotic(
            fit, w_s=20e-6, w_p=0.02, v2_centre=v_c,
            s2_grid_x=gx, s2_grid_y=gy)
        assert out.shape == want, f'shape {out.shape} != {want}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = propagate_modal_asymptotic(
            fit, w_s=20e-6, w_p=0.02, v2_centre=v_c, s2_grid_x=ax,
            s2_grid_y=np.zeros_like(ax), maslov_tracking='row_reset')
        b = propagate_modal_asymptotic(
            fit, w_s=20e-6, w_p=0.02, v2_centre=v_c, s2_grid_x=ax,
            s2_grid_y=np.zeros_like(ax), maslov_tracking='1d_raster')
    assert np.array_equal(a, b), (
        "on a 1-D grid 'row_reset' has no rows to reset and must be "
        "bit-identical to '1d_raster'")

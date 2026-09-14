"""LG aberration-tensor evaluator + sphere/cone-form polynomial helpers
for the phase-space asymptotic propagator.

v5.1.0 file-split (Agent D):  extracted from
``lumenairy.propagators.asymptotic`` with NO public-API or physics
change.  Holds:

* :class:`AberrationTensorResult` -- container for the LG tensor
  evaluated at one chief-ray image point.
* :func:`aberration_tensor` -- the closed-form Wick-contraction
  evaluator (used for a pure ``(0, 0)`` output request; every other
  output-mode request routes to the σ-grid projection -- audit W3-T3).
* The 2-D polynomial algebra (``_multiply_polys_2d``,
  ``_polynomial_under_affine_shift``,
  ``_polynomial_substitute_linear_2d``,
  ``_contract_against_moment_table``) used by the tensor build.
* The scalar saddle-point helpers (``_compute_M_b``,
  ``_phi_v2_hessian``) -- called once per pixel for the
  aberration-tensor closed-form path.

All re-exported through :mod:`lumenairy.propagators.asymptotic`
so existing call sites continue to work unchanged.
"""

# Version history for this module: ``docs/history/lumenairy.propagators.asymptotic_aberration_tensor.md``.

from __future__ import annotations

import math
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# The Chebyshev helpers live in ``lumenairy._math.chebyshev``; these aliases
# bind the public names to the underscore-prefixed locals this module's call
# sites use.
from .._math.chebyshev import (
    chebyshev_derivative_vandermonde as _chebyshev_derivative_vandermonde,
    chebyshev_second_derivative_vandermonde as _chebyshev_second_derivative_vandermonde,
    chebyshev_vandermonde as _chebyshev_vandermonde,
)
from .asymptotic_canonical_fit import (
    CanonicalPolyFit,
    solve_envelope_stationary,
)
from .asymptotic_maslov import (
    B_QUAD_EXP_MAX,
    _maslov_branch_corrected_sqrt,
    lg00_sampling_waist_from_M,
    van_vleck_weight,
)
from .asymptotic_modes import (
    decompose_lg,
    gaussian_moment_table_2d,
    lg_polynomial,
)

__all__ = [
    'AberrationTensorResult',
    'aberration_tensor',
]


# ===========================================================================
# Section 5 -- Aberration tensor and modal asymptotic propagator
# ===========================================================================

@dataclass
class AberrationTensorResult:
    """Result of an LG aberration-tensor evaluation at one image point.

    Fields
    ------
    L : ndarray, shape (n_output_modes, n_source_modes)
        Aberration matrix L_{k,n} = sum_m b_m T_{k;n,m} after pupil
        contraction.  Rows index output LG modes; columns index source
        LG modes.
    output_modes : list of (p, ell)
    source_modes : list of (p, ell)
    pupil_modes : list of (p, ell)
    s2_image : (float, float)
        Chief-ray image point [m] at which this tensor was evaluated.
    w_s, w_p, w_o : float
        Source, pupil, output Gaussian waists.
    v_star : (float, float)
        Envelope-stationary v_2* at s2_image.
    sigma_grid_n : int or None
        σ-grid size actually used [audit W4-T1].  ``None`` on the
        pure-``(0, 0)`` closed-form branch, which builds no grid.  Read it
        to see what the ADAPTIVE default resolved to.
    sigma_curvature : ndarray (2, 2) or None
        Measured quadratic-phase curvature ``C = dv*/dσ`` [direction
        cosine per metre] that the curvature-matched output basis was
        built from [audit W4-T2].  ``None`` unless
        ``curvature_matched_basis=True`` produced a usable measurement --
        so it doubles as the flag's "did it actually engage?" report.  The
        basis carried ``exp(+i·π·σᵀCσ/λ)``.
    van_vleck_weight : complex or None
        The Van Vleck-Maslov integrand weight ``-1j sqrt(|det ds1/dv2|) /
        lambda`` actually applied at the envelope-stationary point (audit
        Y2).  Exposed so a caller that needs the pre-v5.46 scale can divide
        it out and multiply back the old ``|det ds1/dv2|``:
        ``L_legacy = L * |det J| / van_vleck_weight``, i.e.
        ``|L_legacy| = |L| * lambda * sqrt(|det J|)`` -- equivalently
        ``|L_legacy|**2 = |L|**2 * lambda**2 * |det J|``, which is the
        ``1/(lambda**2 |det J|)`` factor the two LG-merit re-pins quote.
        ``None`` on the sigma-grid branch, which evaluates
        the weight per grid point.

    Notes
    -----
    Indices of L are NAMED after physical aberrations via
    ``lg_seidel_label(p, ell)``:  (1, 0) is defocus, (2, 0) is
    spherical, (1, +-1) is coma, (0, +-2) is astigmatism, etc.

    The names are a radial-order correspondence (``n = 2p + |l|``,
    ``m = l``), not a Seidel decomposition (audit Y5): ``L`` is an
    overlap of the IMAGE-PLANE FIELD onto a real-waist LG basis, and LG
    modes are not Zernike polynomials.  The W4-T2 note below records,
    measured, that the ``(2, 0)`` channel is an interference residue
    whose phase rotates with the design (5 of 6 sign flips across
    adjacent designs).  So driving ``|L_{(2,0), 0}|^2`` toward zero is a
    merit on that mode overlap -- useful, and monotone in aberration on
    many designs -- but it is NOT the Seidel spherical coefficient.
    """
    L: np.ndarray
    output_modes: List[Tuple[int, int]]
    source_modes: List[Tuple[int, int]]
    pupil_modes: List[Tuple[int, int]]
    s2_image: Tuple[float, float]
    w_s: float
    w_p: float
    w_o: float
    v_star: Tuple[float, float]
    # v5.29 (audit W4-T1 / W4-T2).  Appended WITH defaults so every existing
    # construction -- positional or keyword -- keeps working unchanged.
    sigma_grid_n: Optional[int] = None
    sigma_curvature: Optional[np.ndarray] = None
    # v5.46 (audit Y2).  Appended WITH a default, like the two above.
    van_vleck_weight: Optional[complex] = None


def _multiply_polys_2d(p_a: Dict[Tuple[int, int], complex],
                        p_b: Dict[Tuple[int, int], complex]
                        ) -> Dict[Tuple[int, int], complex]:
    """Multiply two 2-D polynomial dicts."""
    out: Dict[Tuple[int, int], complex] = {}
    for (i_a, j_a), c_a in p_a.items():
        for (i_b, j_b), c_b in p_b.items():
            key = (i_a + i_b, j_a + j_b)
            out[key] = out.get(key, 0.0 + 0.0j) + c_a * c_b
    return out


def _polynomial_under_affine_shift(
    coeffs: Dict[Tuple[int, int], complex],
    shift_x: complex, shift_y: complex,
    var_name: str = 'eta',
) -> Dict[Tuple[int, int], complex]:
    """Substitute (x, y) -> (x + shift_x, y + shift_y) in a 2-D polynomial.

    Used to produce the ``eta``-polynomial after the (s_1, v_2) -> eta
    coordinate change.
    """
    if not coeffs:
        return {}
    max_i = max(k[0] for k in coeffs)
    max_j = max(k[1] for k in coeffs)
    # Pre-compute (x + shift_x)^i = sum_k C(i, k) shift_x^(i-k) x^k
    # as a coefficient table indexed by k.
    bin_x: List[Dict[int, complex]] = []
    for i in range(max_i + 1):
        row: Dict[int, complex] = {}
        for k in range(i + 1):
            row[k] = math.comb(i, k) * (shift_x ** (i - k))
        bin_x.append(row)
    bin_y: List[Dict[int, complex]] = []
    for j in range(max_j + 1):
        row = {}
        for k in range(j + 1):
            row[k] = math.comb(j, k) * (shift_y ** (j - k))
        bin_y.append(row)

    out: Dict[Tuple[int, int], complex] = {}
    for (i, j), c in coeffs.items():
        for kx, bx in bin_x[i].items():
            for ky, by in bin_y[j].items():
                key = (kx, ky)
                out[key] = out.get(key, 0.0 + 0.0j) + c * bx * by
    return out


def _contract_against_moment_table(
    poly: Dict[Tuple[int, int], complex],
    moments: Dict[Tuple[int, int], complex],
) -> complex:
    """Compute ``<P(eta)>_M = sum_{ij} c_{ij} <eta_x^i eta_y^j>``."""
    total = 0.0 + 0.0j
    for (i, j), c in poly.items():
        total += c * moments.get((i, j), 0.0 + 0.0j)
    return total


def _compute_M_b(
    fit: CanonicalPolyFit,
    s2x: float, s2y: float,
    v2x: float, v2y: float,
    src_x: float, src_y: float,
    w_s: float, w_p: float,
    v_cx: float, v_cy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, complex,
           float, float]:
    """Build the complex beam matrix M, the linear
    term b (eq. 17), the Jacobian J*, and the OPL piston Phi*."""
    s2x_arr = np.asarray(s2x).reshape(())
    s2y_arr = np.asarray(s2y).reshape(())
    v2x_arr = np.asarray(v2x).reshape(())
    v2y_arr = np.asarray(v2y).reshape(())

    s1x, s1y, dS1x_dv2x, dS1x_dv2y, dS1y_dv2x, dS1y_dv2y = (
        fit.eval_s1_with_v2_grad(s2x_arr, s2y_arr, v2x_arr, v2y_arr)
    )
    phi, dPhi_dv2x, dPhi_dv2y = fit.eval_phi_with_v2_grad(
        s2x_arr, s2y_arr, v2x_arr, v2y_arr,
        include_linear=False,
    )

    s1x_v = float(s1x)
    s1y_v = float(s1y)
    phi_v = float(phi)
    J = np.array([
        [float(dS1x_dv2x), float(dS1x_dv2y)],
        [float(dS1y_dv2x), float(dS1y_dv2y)],
    ])
    g = np.array([float(dPhi_dv2x), float(dPhi_dv2y)])

    # Hessian of phi w.r.t. v2 (use finite differences of analytic 1st
    # derivatives; cleaner to differentiate the polynomial twice).
    H_phi = _phi_v2_hessian(fit, s2x, s2y, v2x, v2y)

    inv_ws2 = 1.0 / (w_s * w_s)
    inv_wp2 = 1.0 / (w_p * w_p)
    M_real = inv_ws2 * (J.T @ J) + inv_wp2 * np.eye(2)
    M = M_real - 1j * math.pi * H_phi
    r_star = np.array([s1x_v - src_x, s1y_v - src_y])
    delta_v = np.array([v2x - v_cx, v2y - v_cy])
    b = (2.0j * math.pi * g
         - 2.0 * inv_ws2 * (J.T @ r_star)
         - 2.0 * inv_wp2 * delta_v)
    detJ = float(np.abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]))

    G0 = math.exp(
        -(r_star[0] ** 2 + r_star[1] ** 2) / (w_s * w_s)
        - (delta_v[0] ** 2 + delta_v[1] ** 2) / (w_p * w_p)
    )
    return M, b, np.array([s1x_v, s1y_v]), J, complex(phi_v), G0, detJ


def _phi_v2_hessian(fit: CanonicalPolyFit, s2x: float, s2y: float,
                     v2x: float, v2y: float) -> np.ndarray:
    """Compute the 2x2 Hessian d^2 Phi / d v_2 d v_2 by analytic
    differentiation of the Chebyshev fit, in *physical* (waves per
    direction-cosine^2) units."""
    # Normalised coords
    u1 = (s2x - fit.s2x_centre) / fit.s2x_halfrange
    u2 = (s2y - fit.s2y_centre) / fit.s2y_halfrange
    u3 = (v2x - fit.v2x_centre) / fit.v2x_halfrange
    u4 = (v2y - fit.v2y_centre) / fit.v2y_halfrange

    T1 = _chebyshev_vandermonde(np.array(u1), fit.poly_order)
    T2 = _chebyshev_vandermonde(np.array(u2), fit.poly_order)
    T3 = _chebyshev_vandermonde(np.array(u3), fit.poly_order)
    T4 = _chebyshev_vandermonde(np.array(u4), fit.poly_order)
    dT3 = _chebyshev_derivative_vandermonde(np.array(u3), fit.poly_order)
    dT4 = _chebyshev_derivative_vandermonde(np.array(u4), fit.poly_order)
    d2T3 = _chebyshev_second_derivative_vandermonde(np.array(u3),
                                                       fit.poly_order)
    d2T4 = _chebyshev_second_derivative_vandermonde(np.array(u4),
                                                       fit.poly_order)

    h33 = 0.0
    h34 = 0.0
    h44 = 0.0
    for c, (k1, k2, k3, k4) in zip(fit.coef_phi, fit.multi_indices):
        if c == 0.0:
            continue
        T12 = float(T1[k1]) * float(T2[k2])
        h33 += c * T12 * float(d2T3[k3]) * float(T4[k4])
        h34 += c * T12 * float(dT3[k3]) * float(dT4[k4])
        h44 += c * T12 * float(T3[k3]) * float(d2T4[k4])
    # Add linear-phase Hessian contribution -- linear terms have zero
    # second derivative, so nothing to add.
    invhx = 1.0 / fit.v2x_halfrange
    invhy = 1.0 / fit.v2y_halfrange
    return np.array([
        [h33 * invhx * invhx, h34 * invhx * invhy],
        [h34 * invhx * invhy, h44 * invhy * invhy],
    ])


# Coarse probe grid (per axis) used to measure the image-plane field waist
# that becomes the DEFAULT sigma-basis ``w_o``.  32 is measured to give the
# waist to <= 4.4e-3 relative against a 201x201 reference on the validation
# singlets (16 -> 4.9e-2, 24 -> 4.6e-2, 48 -> 3.0e-3); a few percent on a
# BASIS scale is immaterial, and the probe costs ~25 % of one 64x64
# projection grid.
_W_O_PROBE_N = 32


def _lg00_sampling_waist(M: np.ndarray) -> float:
    """Default ``w_o`` for the pure-``(0, 0)`` CLOSED-FORM branch.

    NOT an image-plane length.  That branch point-samples the field at the
    chief ray and multiplies by ``conj(LG_00(0)) = sqrt(2/(pi w_o^2))``, so
    ``w_o`` only fixes that normalisation constant; ``1/sqrt(lambda_max(Re
    M))`` (the effective pupil acceptance, in direction cosines) is the
    historical convention and is BIT-FOR-BIT the cross-backend contract of
    ``asymptotic_jax_twin.aberration_tensor_lg00_jax``, which hardcodes the
    identical expression.  Change one and you MUST change the other:
    ``tests/unit/test_audit_raytrace.py::…lg00_jax_matches_numpy_0_0`` and
    the W3-3 coupling pin in
    ``tests/unit/test_niche_audit_w3_oracles.py`` hold them together.

    The σ-integration branch, where ``w_o`` IS a length, uses
    :func:`_measure_image_plane_waist` instead (audit W3-T3b).
    """
    # v5.46 (audit Y3/Y5): one shared, eigensolver-free implementation, so
    # the cross-backend contract above is enforced by construction and the
    # near-degenerate ``eigvalsh`` JVP is out of the JAX twin's gradient.
    return float(lg00_sampling_waist_from_M(M, np))


def _s2_validity_room(fit: CanonicalPolyFit,
                      s2x_img: float, s2y_img: float) -> float:
    """Half-width of the largest centred square around ``s2_image`` that
    still lies inside the fit's ``s2`` validity box.  The asymptotic
    propagator is identically ZERO outside that box, so nothing is lost by
    never sampling beyond it."""
    return min(
        fit.s2x_centre + fit.s2x_halfrange - s2x_img,
        s2x_img - (fit.s2x_centre - fit.s2x_halfrange),
        fit.s2y_centre + fit.s2y_halfrange - s2y_img,
        s2y_img - (fit.s2y_centre - fit.s2y_halfrange),
    )


# ---------------------------------------------------------------------------
# v5.29 (audit W4-T1) -- adaptive default σ-grid size
# ---------------------------------------------------------------------------
# The image-plane field is a CHIRP.  Its phase is ``2π·Φ(s2, v*(s2))`` and the
# envelope-stationary relation gives ``dΦ/ds2 = v*/λ`` (Φ in waves, v* a
# direction cosine), so the LOCAL FRINGE RATE at σ is ``|v*(σ)|/λ`` cycles per
# metre.  Nyquist on a grid of ``n`` samples across ``2·extent`` therefore
# needs ``2·extent/(n - 1) <= λ/(2·v_max)``, i.e.
#
#     n >= 4·extent·v_max/λ                                        (W4-T1)
#
# MEASURED (validation singlet R1 = 51.5 mm, extent 4.034e-3 m, λ 1.31 µm):
# the amplitude-weighted 99.9th percentile of the field's own local fringe
# rate, obtained by differentiating the unwrapped phase of a 1-D cut and
# converged in the cut's own sampling (4497 -> 16385 -> 65537 points gives
# 4.4938e+3 / 4.4932e+3 / 4.4941e+3 cycles/m), agrees with ``max|v*|/λ`` over
# the pixels the propagator declares ALIVE (4.4290e+3 cycles/m) to 1.5 %.
# (The raw MAXIMUM of the phase gradient diverges with the cut's sampling --
# 1.19e5 -> 3.79e5 -> 6.37e5 -> it is the π-jump at the amplitude zeros, not
# bandwidth.)  On R1 = 60 mm the same pair reads 6.8342e+3 vs 6.7996e+3
# cycles/m (0.5 %).  So the chirp model is the right predictor and ``v_max``
# is the only free quantity.
#
# ``v_max`` is taken from the fit's ``v2`` VALIDITY BOX, which is a STRICT
# upper bound on the alive region: ``propagate_modal_asymptotic`` zeroes any
# pixel whose Newton solve leaves ``|u3| <= 1, |u4| <= 1`` (see its ``in_box_v``
# mask), so no non-zero pixel can carry a fringe rate above
# ``(|v2c| + v2_halfrange)/λ``.  It costs nothing (no extra propagate), it is
# deterministic and platform-stable (fit outputs), and it reproduces the 246
# quoted in the ``sigma_grid_n`` docstring.  It IS loose -- the alive region's
# actual max |v*| is 0.00580 / 0.00891 against box bounds 0.0199 / 0.01415,
# so the bound over-asks by 3.4x / 1.6x -- and it even ORDERS the two designs
# the wrong way round (245 vs 175 required, while the measured n = 64 error is
# +2.3 % vs +33 %).  Conservative is the correct failure direction for a
# default whose old value UNDER-resolved, and a tighter bound would need the
# alive mask, i.e. the propagate we are trying to size.
_SIGMA_GRID_N_MIN = 64

# Cost/accuracy ladder.  Powers of two and their 1.5x midpoints: each rung is
# a modest cost step, and every rung in the measured table below is on it.
_SIGMA_GRID_LADDER = (64, 96, 128, 192, 256, 384, 512, 768, 1024)

# Default cap.  Chosen from the MEASURED accuracy/cost table below (this box,
# 11-mode request on the two validation singlets; error is relative to
# ``n = 1024``, cost is wall-clock relative to ``n = 64`` = 0.726 s, best of
# two reps with nothing else running):
#
#     n     cost    (0,0)               (1,0)               (2,0)
#                   51.5mm   60.0mm     51.5mm   60.0mm     51.5mm   60.0mm
#     64    1.0x    1.1e-1   5.7e-2     1.6e-1   1.4e-1     2.0e-2   1.5e-1
#     96    2.3x    1.5e-1   4.9e-2     5.8e-2   6.1e-2     1.3e-1   2.2e-2
#     128   4.7x    9.0e-2   1.9e-2     6.0e-3   2.7e-2     6.1e-2   4.5e-3
#     192   11.1x   2.0e-2   2.6e-2     4.7e-2   2.9e-2     3.7e-3   2.3e-2
#     256   15.6x   1.8e-2   1.4e-2     6.1e-3   9.9e-3     4.9e-3   1.6e-2
#     384   19.7x   8.9e-3   2.7e-4     7.3e-4   2.0e-4     8.2e-3   2.5e-3
#     512   40.1x   4.0e-3   1.9e-4     3.2e-3   1.1e-3     6.8e-4   1.8e-4
#
# (Cost grows SUB-quadratically past n = 256 -- 19.7x / 40.1x against 36x /
# 64x pixels -- because the fixed w_o probe and the serial Maslov unwrap stop
# dominating; do not extrapolate the low rungs.)
#
# Convergence is OSCILLATORY (a chirp integral: each rung moves the sample
# phases, not just their density), so no rung is monotonically better than the
# one below -- but the ~1e-1 plateau of ``n = 64`` is gone by ``n = 256`` on
# every (p, 0) channel, and the W4-T1 formula asks for 245 / 175 on these two
# designs, i.e. ladder 256 / 192.  256 therefore buys the order-of-magnitude
# accuracy the flagged defect was about at the 15.6x that the flag already
# measured, without ever truncating on the reference designs.  Push it to 512
# when you need ~1e-3 magnitudes and can pay 63x.  ``sigma_grid_n_max``
# exposes it per call; an optimiser loop that wants the old speed should pass
# ``sigma_grid_n=64`` explicitly (bit-for-bit the pre-W4-T1 default).
_SIGMA_GRID_N_MAX_DEFAULT = 256


def _sigma_chirp_v_max(fit: CanonicalPolyFit) -> float:
    """Strict upper bound on ``|v2*|`` over the pixels the σ-branch's
    propagator can return a NON-ZERO field at [direction cosine].

    ``propagate_modal_asymptotic`` masks out every pixel whose
    envelope-stationary ``v2*`` falls outside the fit's ``v2`` box, so this
    is also a strict upper bound on the field's local fringe rate times λ.
    """
    return max(abs(float(fit.v2x_centre)) + abs(float(fit.v2x_halfrange)),
               abs(float(fit.v2y_centre)) + abs(float(fit.v2y_halfrange)))


def _required_sigma_grid_n(fit: CanonicalPolyFit, extent: float) -> int:
    """Nyquist sample count for the chirped image-plane field on a σ grid of
    half-extent ``extent``:  ``n >= 4·extent·v_max/λ`` (audit W4-T1).

    Returns :data:`_SIGMA_GRID_N_MIN` if any input is degenerate.
    """
    lam = float(fit.wavelength)
    v_max = _sigma_chirp_v_max(fit)
    if not (math.isfinite(lam) and lam > 0.0
            and math.isfinite(extent) and extent > 0.0
            and math.isfinite(v_max) and v_max > 0.0):
        return _SIGMA_GRID_N_MIN
    n_req = 4.0 * extent * v_max / lam
    if not math.isfinite(n_req):
        return _SIGMA_GRID_N_MIN
    return max(_SIGMA_GRID_N_MIN, int(math.ceil(n_req)))


def _sigma_grid_n_on_ladder(n_req: int) -> int:
    """Round ``n_req`` UP onto :data:`_SIGMA_GRID_LADDER` (doubling past its
    top rung, so the result is never below what was asked for)."""
    for cand in _SIGMA_GRID_LADDER:
        if cand >= n_req:
            return cand
    n = _SIGMA_GRID_LADDER[-1]
    while n < n_req:
        n *= 2
    return n


def _fit_fingerprint(fit: CanonicalPolyFit) -> Tuple:
    """A hashable fingerprint of everything in ``fit`` a propagate reads.

    The coefficient vectors go in as raw bytes of a canonical float64 copy,
    so two structurally identical fits from separate traces share an entry and
    a refit never serves the old one.  ``id(fit)`` would be cheaper and wrong:
    CPython reuses the address of a collected object, so a cache keyed on it
    can answer for a fit that no longer exists.
    """
    def _b(a):
        return None if a is None else np.ascontiguousarray(
            np.asarray(a, dtype=np.float64)).tobytes()

    return (int(fit.poly_order), tuple(map(tuple, fit.multi_indices)),
            _b(fit.coef_phi), _b(fit.coef_s1x), _b(fit.coef_s1y),
            _b(fit.linear_coeffs_phi), bool(fit.extract_linear_phase),
            float(fit.s2x_centre), float(fit.s2x_halfrange),
            float(fit.s2y_centre), float(fit.s2y_halfrange),
            float(fit.v2x_centre), float(fit.v2x_halfrange),
            float(fit.v2y_centre), float(fit.v2y_halfrange),
            float(fit.wavelength))


def _callable_identity(fn) -> Tuple:
    """The ``propagate`` half of the waist-cache key: the OBJECT identity of
    the evaluator, made safe by the entry that holds it.

    A callable's answer is its code PLUS whatever state it captured, and no
    structural key can see the second half:

    * ``__qualname__`` alone separates nothing useful -- two closures from one
      factory share ``factory.<locals>.propagate``, two lambdas share
      ``<lambda>``, and a :class:`functools.partial` has none at all, so it
      degrades to a ``repr`` carrying a memory address.
    * the code object's ``(module, qualname, file, first line)`` -- the shape
      ``lumenairy._cache_registry._clearer_identity`` uses -- separates two
      DEFINITIONS but still not two closures over one ``def`` nor two
      instances of one callable class, as that function's own docstring says.

    So the key is ``id(fn)``, and :func:`_w_o_cache_put` stores ``fn`` itself
    beside the value.  That is what makes the id sound: the ``id()``-reuse
    hazard :func:`_fit_fingerprint` exists to avoid needs the keyed object to
    have been COLLECTED, and every live entry holds a strong reference to its
    own callable, so no two live entries can share an id.  The cost is 64
    references at the cache bound; the price is that a caller which rebuilds
    an equivalent wrapper on every call never hits (correct, just uncached),
    while the shipped caller -- which passes the module-level
    ``propagate_modal_asymptotic`` object -- always does.  The qualified name
    rides along so a dumped key is readable.
    """
    return (id(fn), getattr(fn, '__qualname__', type(fn).__name__))


def _propagate_seams() -> Tuple:
    """The process-global A/B seams the probe's ``propagate`` reads.

    ``_measure_image_plane_waist``'s answer is a pure function of its
    arguments AND of whatever module seams the evaluator consults, so the
    seams belong in the key: without them an A/B measurement of a seam is
    order-dependent -- warm the cache with the seam off, flip it, and the
    stale width comes back.  MEASURED (VERIFY-B7, 2026-09-14) on a
    N-SF11 / 0.85 um chart: ``_NEWTON_SCALE_RELATIVE_STOP`` moves ``w_o`` from
    1.4163857683404920e-04 to 1.4163857683413807e-04, 6.3e-11 relative.

    A future seam that ``propagate_modal_asymptotic`` reads must join this
    tuple, or flipping it must drain the cache.
    """
    from . import asymptotic_maslov as _am
    return (bool(getattr(_am, '_NEWTON_SCALE_RELATIVE_STOP', False)),)


# Cross-call cache of the image-plane waist probe.  The probe is one (rarely
# two) coarse ``propagate_modal_asymptotic`` calls on a ``_W_O_PROBE_N`` grid
# -- MEASURED 0.19 s of a 6.6 s default ``aberration_tensor`` on the
# validation singlet, and repeated verbatim by every call that shares a fit,
# an image point and a pupil weighting (a merit evaluated over source modes,
# or an optimiser loop whose step did not move the optic).  Bounded and
# FIFO-evicted; each entry is one float beside a reference to the evaluator
# that produced it (see :func:`_callable_identity`).
_W_O_CACHE: 'OrderedDict[Any, Tuple[Optional[float], Any]]' = OrderedDict()
_W_O_CACHE_MAX = 64
_W_O_CACHE_LOCK = threading.Lock()


def clear_image_plane_waist_cache() -> None:
    """Drain the :func:`_measure_image_plane_waist` cross-call cache."""
    with _W_O_CACHE_LOCK:
        _W_O_CACHE.clear()


try:
    import sys as _sys

    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        'image_plane_waist',
        lambda: getattr(_this_mod, 'clear_image_plane_waist_cache')(),
    )
except ImportError:      # pragma: no cover - registry always present in-tree
    pass


def _measure_image_plane_waist(
    fit: CanonicalPolyFit,
    s2x_img: float, s2y_img: float,
    source_point: Tuple[float, float],
    pupil_amplitudes: Dict[Tuple[int, int], complex],
    w_s: float, w_p: float,
    v2_centre: Tuple[float, float],
    propagate,
    n: int = _W_O_PROBE_N,
) -> Optional[float]:
    """Measure the propagated field's image-plane Gaussian waist [m].

    This is the DEFAULT output-basis scale for the σ-integration path.
    It is measured, not modelled, because the quantity the LG projection
    needs is by definition the field's own width, and that width is
    dominated by the DEFOCUS / aberration blur -- which no function of the
    pupil-space beam matrix ``M`` alone can see (audit W3-T3b: between the
    two validation singlets the true waist moves 71.7 % while every
    ``M``-only construction moves < 0.3 %).

    Estimator: intensity second moment (D4σ/2) of ``|U|²`` on a coarse
    grid spanning the fit's ``s2`` validity box, i.e. for an amplitude
    ``exp(-r²/w²)`` -- whose intensity has per-axis variance ``w²/4`` --
    ``w = 2·sqrt(var_per_axis)``.  One refinement pass fires only if the
    field turns out to be sampled by fewer than 3 cells (a field far
    narrower than the validity box), so the common case costs exactly one
    coarse propagate.  The probe uses the FUNDAMENTAL source mode so the
    basis scale cannot depend on the caller's ``source_modes`` ordering.

    Returns ``None`` if the probe cannot produce a finite positive width
    (dead field, degenerate box); the caller then falls back.

    The answer is a pure function of ``(fit, s2_image, source_point,
    pupil_amplitudes, w_s, w_p, v2_centre, n)`` -- every one of which reaches
    the probe's ``propagate`` call or its grid -- plus ``propagate`` itself
    and the process-global seams it reads, so it is memoised on exactly that
    tuple in ``_W_O_CACHE``.  ``propagate`` enters by
    :func:`_callable_identity` and the seams by :func:`_propagate_seams`.
    """
    _key = (_fit_fingerprint(fit), float(s2x_img), float(s2y_img),
            float(source_point[0]), float(source_point[1]),
            tuple(sorted((tuple(k), complex(v))
                         for k, v in (pupil_amplitudes or {}).items())),
            float(w_s), float(w_p),
            float(v2_centre[0]), float(v2_centre[1]), int(n),
            _callable_identity(propagate), _propagate_seams())
    with _W_O_CACHE_LOCK:
        if _key in _W_O_CACHE:
            _W_O_CACHE.move_to_end(_key)
            return _W_O_CACHE[_key][0]

    room = _s2_validity_room(fit, s2x_img, s2y_img)
    if not (math.isfinite(room) and room > 0.0):
        return _w_o_cache_put(_key, None, propagate)
    ext = 0.98 * room
    w = None
    for _pass in range(2):
        xs = np.linspace(s2x_img - ext, s2x_img + ext, n)
        ys = np.linspace(s2y_img - ext, s2y_img + ext, n)
        SX, SY = np.meshgrid(xs, ys, indexing='xy')
        try:
            U = propagate(
                fit,
                source_point=(float(source_point[0]), float(source_point[1])),
                source_amplitudes={(0, 0): 1.0 + 0.0j},
                pupil_amplitudes=pupil_amplitudes,
                w_s=w_s, w_p=w_p, v2_centre=v2_centre,
                s2_grid_x=SX, s2_grid_y=SY,
            )
        except (ValueError, RuntimeError, ZeroDivisionError, IndexError,
                np.linalg.LinAlgError):
            return _w_o_cache_put(_key, None, propagate)
        inten = np.abs(np.asarray(U)) ** 2
        inten = np.where(np.isfinite(inten), inten, 0.0)
        tot = float(inten.sum())
        if not (tot > 0.0):
            return _w_o_cache_put(_key, None, propagate)
        lx = SX - s2x_img
        ly = SY - s2y_img
        cx = float((inten * lx).sum() / tot)
        cy = float((inten * ly).sum() / tot)
        var = float(
            (inten * ((lx - cx) ** 2 + (ly - cy) ** 2)).sum() / tot) / 2.0
        if not (math.isfinite(var) and var > 0.0):
            return _w_o_cache_put(_key, None, propagate)
        w = 2.0 * math.sqrt(var)
        cell = 2.0 * ext / (n - 1)
        if w >= 3.0 * cell:
            return _w_o_cache_put(_key, w, propagate)
        # Under-sampled: the field is far narrower than the validity box.
        ext_next = min(6.0 * w, 0.98 * room)
        if not (ext_next > 0.0) or ext_next >= ext:
            return _w_o_cache_put(_key, w, propagate)
        ext = ext_next
    return _w_o_cache_put(_key, w, propagate)


def _w_o_cache_put(key, value, fn):
    """Record ``value`` under ``key`` in the bounded waist cache and return
    it, so every exit of :func:`_measure_image_plane_waist` caches exactly
    what it returns -- including the ``None`` verdicts, which cost the same
    probe to reach.

    The entry is ``(value, fn)``: holding the evaluator alive is what makes
    the ``id(fn)`` in the key sound (:func:`_callable_identity`), since an id
    can only be reused once its object has been collected."""
    with _W_O_CACHE_LOCK:
        _W_O_CACHE[key] = (value, fn)
        while len(_W_O_CACHE) > _W_O_CACHE_MAX:
            _W_O_CACHE.popitem(last=False)
    return value


# ---------------------------------------------------------------------------
# v5.29 (audit W4-T2) -- curvature-matched (complex-q) output basis
# ---------------------------------------------------------------------------
def _measure_sigma_phase_curvature(
    fit: CanonicalPolyFit,
    s2x_img: float, s2y_img: float,
    src_x: float, src_y: float,
    w_s: float, w_p: float,
    v2_centre: Tuple[float, float],
    extent: float,
) -> Optional[np.ndarray]:
    """Measure the image-plane field's QUADRATIC phase curvature at the
    chief ray:  ``C = d v*/d σ`` [direction cosine per metre], symmetric 2x2.

    The field's phase is ``2π·Φ(s2, v*(s2))`` and ``dΦ/ds2 = v*/λ`` (Φ in
    waves), so ``C/λ`` is the Hessian of the phase in waves and the field
    carries ``exp(i·π·σᵀCσ/λ)`` to second order -- a defocus/curvature term
    of ``-30.1`` waves at the σ-grid edge on the validation singlet at
    R1 = 51.5 mm (``C_xx = -4.847``, equivalent radius -0.206 m), i.e. the
    field is a strongly curved wavefront, which is exactly why a REAL-waist
    LG basis cannot represent it in a few ``(p, 0)`` modes.

    Estimated by a central difference of :func:`solve_envelope_stationary`
    about ``s2_image`` with step ``extent/8`` -- four Newton solves, no
    propagate.  MEASURED against the alternative (an amplitude-weighted
    least-squares fit of ``v*(σ)`` over a 33x33 probe grid, which costs a
    full coarse propagate + Newton batch): ``C_xx`` agrees to 0.11 % - 0.28 %
    and the resulting ``|L(2, 0)|`` to 0.9 % - 3.4 % across eight designs,
    and BOTH make the channel STRICTLY MONOTONE in R1 (0 of 6 sign flips) --
    so the cheap estimator is the one that ships.

    Returns ``None`` if the probe leaves the fit's validity box or the
    Newton solves do not produce a finite symmetric result; the caller then
    keeps the flat-phase basis.
    """
    for shrink in (1.0, 0.25):
        h = abs(float(extent)) * 0.125 * shrink
        if not (math.isfinite(h) and h > 0.0):
            return None
        pts = ((s2x_img + h, s2y_img), (s2x_img - h, s2y_img),
               (s2x_img, s2y_img + h), (s2x_img, s2y_img - h))
        if any(not bool(fit.in_box(np.asarray(px), np.asarray(py),
                                   np.asarray(v2_centre[0]),
                                   np.asarray(v2_centre[1])))
               for (px, py) in pts):
            continue
        vs = []
        ok = True
        for (px, py) in pts:
            try:
                v, _n, _r = solve_envelope_stationary(
                    fit, (px, py), (src_x, src_y),
                    w_s=w_s, w_p=w_p, v2_centre=v2_centre,
                )
            except (ValueError, RuntimeError, ZeroDivisionError,
                    np.linalg.LinAlgError):
                ok = False
                break
            vx, vy = float(v[0]), float(v[1])
            if not (math.isfinite(vx) and math.isfinite(vy)):
                ok = False
                break
            # Outside the v2 box the propagator returns an identically
            # ZERO field, so a derivative taken there describes nothing.
            if not bool(fit.in_box(np.asarray(px), np.asarray(py),
                                   np.asarray(vx), np.asarray(vy))):
                ok = False
                break
            vs.append((vx, vy))
        if not ok:
            continue
        (vpx_x, vpx_y), (vmx_x, vmx_y), (vpy_x, vpy_y), (vmy_x, vmy_y) = vs
        c_xx = (vpx_x - vmx_x) / (2.0 * h)
        c_yy = (vpy_y - vmy_y) / (2.0 * h)
        # Symmetrise: C is a Hessian, so the mixed partials must agree;
        # averaging both estimates halves the finite-difference error.
        c_xy = 0.5 * ((vpy_x - vmy_x) / (2.0 * h)
                      + (vpx_y - vmx_y) / (2.0 * h))
        C = np.array([[c_xx, c_xy], [c_xy, c_yy]], dtype=np.float64)
        if not np.all(np.isfinite(C)):
            continue
        return C
    return None


def aberration_tensor(
    fit: CanonicalPolyFit,
    s2_image: Tuple[float, float],
    *,
    source_point: Tuple[float, float] = (0.0, 0.0),
    source_modes: Optional[List[Tuple[int, int]]] = None,
    pupil_modes: Optional[List[Tuple[int, int]]] = None,
    output_modes: Optional[List[Tuple[int, int]]] = None,
    pupil_amplitudes: Optional[Dict[Tuple[int, int], complex]] = None,
    w_s: float = 50e-6,
    w_p: float = 0.05,
    w_o: Optional[float] = None,
    v2_centre: Tuple[float, float] = (0.0, 0.0),
    sigma_grid_n: Optional[int] = None,
    sigma_grid_n_max: Optional[int] = None,
    sigma_grid_extent: Optional[float] = None,
    curvature_matched_basis: bool = False,
) -> AberrationTensorResult:
    """LG aberration tensor at a single chief-ray image point.

    Expand source, pupil, and output
    fields in Laguerre-Gaussian bases, evaluate the leading-order
    asymptotic propagator analytically as a Wick-contracted Gaussian
    moment, and project onto the output basis to read off the
    coefficient ``L_{k, n} = sum_m b_m T_{k;n,m}`` of each named
    aberration channel.

    Parameters
    ----------
    fit : CanonicalPolyFit
        4-D Chebyshev fit of the prescription.
    s2_image : (float, float)
        Image-plane point [m] at which to evaluate the tensor.  Should
        be the chief-ray landing of ``source_point`` for the
        Seidel-name interpretation to apply.
    source_point : (float, float)
        Source-plane point [m].
    source_modes, pupil_modes, output_modes : list of (p, ell)
        LG mode indices to retain.  Defaults below give a useful
        baseline.
    pupil_amplitudes : dict, optional
        Pupil expansion coefficients ``{(p, ell): complex}``.  If None,
        defaults to a clean LG_{0,0} pupil (b_{0,0} = 1) -- the
        "ideal-Gaussian-pupil" convention used for design merit
        functions, where higher pupil modes are not needed because the
        Seidel content lives entirely on the output side.
    w_s, w_p : float
        Source and pupil Gaussian waists [m and direction-cosine].
    w_o : float, optional
        Output Gaussian waist [m].  An explicit value is honoured
        verbatim on both branches.  The DEFAULT differs per branch,
        because ``w_o`` plays two different roles (audit W3-T3b):

        * σ-integration (any output mode other than ``(0, 0)``) --
          ``w_o`` is a real IMAGE-PLANE LENGTH: the LG basis waist and,
          via ``extent = 4·w_o``, the σ-grid span.  Defaults to the
          MEASURED waist of the propagated field (intensity second
          moment on a coarse probe over the fit's ``s2`` validity box;
          see :func:`_measure_image_plane_waist`).  It has to be
          measured: the width is set by the defocus/aberration blur,
          which no function of the pupil-space beam matrix ``M`` can
          see -- across the two validation singlets the true waist
          moves +71.7 % while every ``M``-only construction moves
          < 0.3 %.
        * pure ``[(0, 0)]`` -- the closed form point-samples the field
          and ``w_o`` only sets the ``sqrt(2/(π w_o²))`` normalisation.
          Keeps the historical ``1/sqrt(λ_max(Re M))`` convention
          BIT-FOR-BIT; it is the cross-backend contract of
          ``aberration_tensor_lg00_jax`` (see
          :func:`_lg00_sampling_waist`).
    v2_centre : (float, float), optional
        Pupil centre.
    sigma_grid_n : int, optional
        Output-plane grid size used for the σ-integration path, i.e.
        whenever ``output_modes`` requests anything other than the
        single mode ``(0, 0)``.  Ignored for a pure ``[(0, 0)]`` request
        (the fast closed-form chief-ray sampling is used).

        An explicit value is honoured VERBATIM, and ``sigma_grid_n=64``
        reproduces the pre-v5.29 flat default bit-for-bit -- pass it in
        an optimiser loop that wants the old speed.

        **The DEFAULT is ADAPTIVE since v5.29 (audit W4-T1).**  The
        image-plane field is a CHIRP: the envelope-stationary relation
        ``dΦ/ds2 = v*/λ`` makes its local fringe rate at offset σ equal
        to ``|v*(σ)|/λ`` cycles/m, so Nyquist needs

            n >= 4·extent·v_max/λ

        with ``v_max`` bounded by the fit's ``v2`` validity box (the
        propagator zeroes every pixel whose ``v2*`` leaves it, so no
        non-zero pixel can chirp faster).  That is 245 / 175 on the two
        validation singlets, against the old flat default of 64 -- which
        therefore UNDER-resolved by 1.1x - 1.7x and cost +2.3 % / +33 %
        on the ``(2, 0)`` magnitude.  The default now rounds the required
        ``n`` up onto the cost ladder ``64, 96, 128, 192, 256, 384, 512,
        768, 1024`` and clamps it to ``sigma_grid_n_max``.

        Measured accuracy (relative to ``n = 1024``) and cost (relative
        to ``n = 64``, best of two clean reps on this box):

            n     cost   |(0,0)| err      |(2,0)| err
                         51.5mm  60.0mm   51.5mm  60.0mm
            64    1.0x   1.1e-1  5.7e-2   2.0e-2  1.5e-1
            128   4.7x   9.0e-2  1.9e-2   6.1e-2  4.5e-3
            192   11.1x  2.0e-2  2.6e-2   3.7e-3  2.3e-2
            256   15.6x  1.8e-2  1.4e-2   4.9e-3  1.6e-2
            384   19.7x  8.9e-3  2.7e-4   8.2e-3  2.5e-3
            512   40.1x  4.0e-3  1.9e-4   6.8e-4  1.8e-4

        Convergence is OSCILLATORY -- a chirp integral moves its sample
        phases, not just their density, at every rung -- so no single
        rung is monotonically better than the one below; what the
        adaptive default buys is the exit from the ~1e-1 plateau that
        ``n = 64`` sat on.  (Cost grows SUB-quadratically past 256 --
        19.7x / 40.1x against 36x / 64x pixels -- because the fixed
        ``w_o`` probe and the serial Maslov unwrap stop dominating.)
        Channel-to-channel and design-to-design
        RESPONSES were always robust to it (the curvature discriminator
        reads 2.09e-1 relative at n = 64 and 4.04e-1 at n = 256).  This
        is a property of the σ quadrature, not of the W3-T3b waist: the
        same aliasing is present at any explicit ``w_o`` that spans the
        same box.

        Residual limitation: the amplitude is HARD-TRUNCATED where the
        fit's validity box ends (only 30 % / 65 % of the default grid
        carries a non-zero field on the two singlets), and that step has
        no band limit -- so no finite ``n`` converges the last ~1e-3.
    sigma_grid_n_max : int, optional
        Cap on the ADAPTIVE ``sigma_grid_n`` default; ignored when
        ``sigma_grid_n`` is given explicitly.  Default 256, picked from
        the table above (the accuracy elbow that the W4-T1 formula
        already asks for on the reference designs, at the 15.6x that the
        W3-T3b flag measured).  When the cap truncates the required
        ``n``, a ``UserWarning`` naming BOTH numbers is emitted.  Raise
        it to 512 for ~1e-3 magnitudes at 31x; lower it (or pass
        ``sigma_grid_n=64``) inside an optimiser loop.
    sigma_grid_extent : float, optional
        Half-extent of the σ-grid [m].  Default ``4 · w_o``, clamped to
        the fit's ``s2`` validity half-box measured from ``s2_image``
        (the propagator is identically zero outside it, so an
        overshooting grid samples no field at all -- see the W3-T3 note
        at the branch).  An explicit value is honoured verbatim, so
        pass one if you want the raw ``4 · w_o`` behaviour.

        When the clamp binds (the field fills the validity box), the σ
        grid spans FEWER than ``4 · w_o`` and the LG basis is truncated:
        measured on the validation singlets, the discrete basis Gram
        matrix departs from the identity by 2.7e-3 at R1 = 51.5 mm but by
        0.455 at R1 = 60 mm (where ``±extent`` is only ``±1.51 w_o``).
        The channel values are still exact overlaps of that discrete
        basis -- what degrades is the basis's own orthonormality, so
        cross-channel interpretation gets softer as the clamp bites.
    curvature_matched_basis : bool, optional
        Opt-in (default ``False`` = the historical flat-phase basis,
        BIT-FOR-BIT).  When ``True``, project onto a curvature-matched
        COMPLEX-q LG basis: the same radial profile times the field's own
        MEASURED quadratic phase, ``LG_k(σ; w_o) · exp(i·π·σᵀCσ/λ)`` with
        ``C = dv*/dσ`` from :func:`_measure_sigma_phase_curvature`.
        Ignored on the pure-``(0, 0)`` closed-form branch (no σ grid, so
        no σ-dependent phase to match) and when the curvature cannot be
        measured (a ``UserWarning`` says so and the flat basis is kept).

        **Use it when you need a channel that is a SMOOTH function of the
        design** -- an optimiser gradient, a merit trend, a tolerance
        sweep.  Measured (audit W4-T2, eight singlets R1 = 51.5 -> 60 mm,
        ``sigma_grid_n = 256``, ``|L(2, 0)|``):

            basis                          sign flips   ptp/mean
            flat (default)                    5 of 6      0.307
            curvature-matched (this flag)     0 of 6      2.197

        i.e. the flag turns a channel that reversed direction on five of
        six adjacent design steps into a STRICTLY MONOTONE one, and makes
        it a 7x stronger discriminator at the same time.  The physics: the
        field is a strongly curved wavefront (-30.1 waves of quadratic
        phase at the σ-grid edge at R1 = 51.5 mm, equivalent radius
        -0.206 m, sweeping to -21.5 waves at R1 = 60 mm), and expanding
        that in a REAL-waist basis needs far more than three ``(p, 0)``
        modes -- so the truncated ``(2, 0)`` coefficient is an
        interference residue whose phase rotates with the design rather
        than a measure of spherical aberration.  Matching the basis
        curvature removes the rotation and leaves the residual (genuinely
        aberration) content.

        **Do NOT use it** when you need the raw overlap onto the standard
        flat LG basis -- the cross-backend ``(0, 0)`` contract, the
        oracle-comparable overlaps, and every pinned channel VALUE are
        defined on the default basis.  The two bases are different
        expansions of the same field; their coefficients are not
        interchangeable.

        Rejected alternative, MEASURED: "exclude defocus from the
        channel" by algebraically removing the ``(1, 0)`` projection
        before reading ``(2, 0)``.  In an orthonormal basis that is
        identically a no-op, and on the discrete grid it moves
        ``|L(2, 0)|`` by only 0.4 % - 20 % (via the Gram off-diagonal,
        4.4e-4 to 0.219) -- not enough: 2 of 6 sign flips remain.  The
        rotating phase, not the defocus AMPLITUDE, is what makes the
        channel non-smooth.

    Returns
    -------
    AberrationTensorResult

    Notes
    -----
    The projection at the chief image is a point-sampling functional on
    the closed-form branch: its whole output-mode dependence is
    ``conj(LG_k)`` evaluated at ONE point.  That equals
    ``N_{p,0} = sqrt(2/(pi w_o^2))`` for every ``(p, 0)`` mode, and is
    identically zero for every ``l != 0`` mode -- ``(sigma_x + j
    sigma_y)^|l|`` times a Laguerre polynomial has no constant term.  So
    only the pure ``[(0, 0)]`` request takes that branch: its LG
    polynomial genuinely IS that constant, and it is the documented
    cross-backend contract of ``aberration_tensor_lg00_jax``.  Do NOT
    widen the branch -- it would return a BIT-IDENTICAL ``L`` for piston
    / defocus / spherical / every higher ``(p, 0)`` channel and an
    identically zero one for coma, astigmatism and tilt, silently.

    Every other request routes to the sigma-integration (a small
    output-plane grid, ``propagate_modal_asymptotic`` + ``decompose_lg``),
    whose overlaps an independent from-scratch LG quadrature reproduces to
    ~1e-14 relative.  The two paths carry different overall scales
    (sampling vs. overlap integral); see the note at the branch.
    """
    # Late import to avoid the propagators.asymptotic shell ↔ this
    # submodule import cycle; ``propagate_modal_asymptotic`` lives in
    # the shell so it can resolve ``_solve_envelope_stationary_batch``
    # against the shell module's globals (test-monkey-patch contract).
    from .asymptotic import propagate_modal_asymptotic

    if source_modes is None:
        source_modes = [(0, 0)]
    if pupil_modes is None:
        pupil_modes = [(0, 0)]
    if output_modes is None:
        # Default:  the named Seidel/Zernike aberrations through 4th order
        output_modes = [
            (0, 0),  # piston / Strehl
            (1, 0),  # defocus
            (2, 0),  # primary spherical
            (0, 1), (0, -1),  # tilt
            (1, 1), (1, -1),  # coma
            (0, 2), (0, -2),  # astigmatism
            (0, 3), (0, -3),  # trefoil
        ]
    if pupil_amplitudes is None:
        pupil_amplitudes = {(0, 0): 1.0 + 0.0j}

    # Y5 (audit): ``pupil_modes`` is very nearly INERT -- its only real use
    # is sizing the Wick moment table, and the actual pupil CONTENT comes
    # exclusively from ``pupil_amplitudes``.  A caller who passes
    # ``pupil_modes=[(0, 0), (1, 0)]`` without matching amplitudes got an
    # LG_{0,0} pupil and a result object that claimed otherwise, silently.
    # Say so rather than change the default content under them.
    _missing = [k for k in (tuple(m) for m in pupil_modes)
                if abs(complex(pupil_amplitudes.get(k, 0.0))) < 1e-300]
    if _missing:
        import warnings as _w
        _w.warn(
            f"aberration_tensor: pupil_modes requests {_missing} but "
            f"pupil_amplitudes carries no (non-zero) coefficient for "
            f"them, so the pupil content is only "
            f"{sorted(k for k, v in pupil_amplitudes.items() if abs(v))}.  "
            f"``pupil_modes`` sizes the moment table and is echoed on the "
            f"result; it does NOT create pupil content.  Pass "
            f"pupil_amplitudes={{mode: coefficient}} for every mode you "
            f"want in the pupil.",
            RuntimeWarning, stacklevel=2)

    s2x_img, s2y_img = float(s2_image[0]), float(s2_image[1])
    src_x, src_y = float(source_point[0]), float(source_point[1])

    # Solve the envelope-stationary equation at s2_image
    v_star, _n_iter, _resid = solve_envelope_stationary(
        fit, (s2x_img, s2y_img), (src_x, src_y),
        w_s=w_s, w_p=w_p, v2_centre=v2_centre,
    )
    v2x_star, v2y_star = v_star

    # Build M, b at v_star
    M, b, s1_star, J_star, phi_star, G0, detJ = _compute_M_b(
        fit, s2x_img, s2y_img, v2x_star, v2y_star,
        src_x, src_y, w_s, w_p, v2_centre[0], v2_centre[1]
    )

    # Which branch will run?  (Only the output modes decide -- see the
    # routing note further down.)  The DEFAULT ``w_o`` differs between the
    # two because ``w_o`` plays two completely different roles.
    needs_sigma_integration = any(tuple(k_out) != (0, 0)
                                  for k_out in output_modes)

    # Choose output waist if not supplied
    if w_o is None:
        if needs_sigma_integration:
            # σ-INTEGRATION PATH:  here ``w_o`` is a genuine IMAGE-PLANE
            # LENGTH -- the waist of the LG basis the field is projected
            # onto, and (via ``extent = 4·w_o``) the span of the σ grid.
            # It must therefore match the field's own image-plane width.
            #
            # No function of ``M`` alone can supply it: the image-plane
            # width is dominated by the defocus/aberration blur, which
            # lives in the σ↔v coupling, not in the pupil-space Hessian.
            # Measured across the two validation singlets (R1 = 51.5 mm vs
            # 60 mm): true waist 1.559e-3 -> 2.677e-3 m (+71.7 %) while
            # ``1/sqrt(lambda_max(Re M))`` moves 1.0116e-4 -> 1.0086e-4
            # (-0.3 %) and the diffraction image of that acceptance,
            # ``lambda·sqrt(lambda_max)/pi``, moves +0.3 %.  A basis pinned
            # to a design-independent scale makes every merit channel
            # design-independent too -- exactly the CI symptom this fixes
            # (``LGAberrationMerit`` responded 4.0e-3 relative to a 17 %
            # curvature change; post-fix 2.0e-1).
            #
            # So measure it.  See :func:`_measure_image_plane_waist`.
            w_o = _measure_image_plane_waist(
                fit, s2x_img, s2y_img, (src_x, src_y), pupil_amplitudes,
                w_s, w_p, v2_centre, propagate_modal_asymptotic,
            )
            if w_o is None:
                # Probe could not produce a finite width (dead field or
                # degenerate box).  Fall back to a quarter of the validity
                # room, so the ``4·w_o`` grid spans the box exactly -- the
                # know-nothing choice -- and only then to the legacy scale.
                _room = _s2_validity_room(fit, s2x_img, s2y_img)
                if math.isfinite(_room) and _room > 0.0:
                    w_o = 0.25 * _room
                else:
                    w_o = _lg00_sampling_waist(M)
        else:
            # PURE-(0, 0) CLOSED-FORM PATH:  ``w_o`` is NOT a length here.
            # The branch point-samples the field at the chief ray and
            # multiplies by ``conj(LG_00(0)) = sqrt(2/(pi w_o^2))``, so
            # ``w_o`` only sets that normalisation constant.  Its value is
            # a CONVENTION, and it is the documented cross-backend contract
            # of ``aberration_tensor_lg00_jax`` (which hardcodes
            # ``A_lead·N_s·N_p·N_o`` with the identical default).  Left
            # bit-for-bit by W3-T3b for exactly that reason; the twin
            # carries the same formula and the pair is pinned.
            w_o = _lg00_sampling_waist(M)

    # Stationary shift delta* = 0.5 M^-1 b
    M_inv = np.linalg.inv(M)
    delta_star = 0.5 * (M_inv @ b)
    # 4.10: removed unused `Sigma = 0.5 * M_inv` (dead code).
    # 4.11.2: route through the shared Maslov branch helper so this
    # site stays consistent with ``propagate_modal_asymptotic`` and
    # the JAX sibling evaluators.  Single-point evaluation defaults
    # to the principal sqrt (caller has no branch history at one
    # image point); see ``_maslov_branch_corrected_sqrt``.
    sqrt_detM, _, _ = _maslov_branch_corrected_sqrt(np.linalg.det(M))

    # Leading amplitude
    # S4/Y2: |det J| -> -1j sqrt(|det J|)/lambda (Van Vleck-Maslov).
    # P3 (audit): the batched path masks |Re(b_quad)| > 700 before the
    # exponential; do the same here instead of letting it overflow to inf
    # with a bare NumPy warning.
    _bq = complex(0.25 * (b @ M_inv @ b))
    _vv_weight_reported = complex(van_vleck_weight(detJ, fit.wavelength))
    if not np.isfinite(abs(_bq)) or abs(_bq.real) > B_QUAD_EXP_MAX:
        A_lead = 0.0j
    else:
        A_lead = (van_vleck_weight(detJ, fit.wavelength)
                  * (math.pi / sqrt_detM) * G0
                  * np.exp(2j * math.pi * phi_star)
                  * np.exp(_bq))

    # Pre-tabulate eta-moments for max polynomial order needed
    max_order_needed = max(
        max((2 * p + abs(ell) for (p, ell) in source_modes), default=0)
        + max((2 * p + abs(ell) for (p, ell) in pupil_modes), default=0),
        4  # always have enough for low output orders
    )
    eta_moments = gaussian_moment_table_2d(M, max_order_needed)

    # Compute T_{k;n,m} for each (k, n, m).
    # The integrand polynomial is
    #     P_{n,m}(eta) = p^src_n(s_1(s_2*, v_2*+eta) - s_src; w_s)
    #                   * p^pup_m(v_2* - v_2c + eta; w_p)
    # at leading order s_1(s_2*, v_2*+eta) = s_1* + J* delta_star + J* eta;
    # so the source argument is r* + J* delta_star + J* eta.
    # The pupil argument is (v_2* - v_2c + delta_star + eta).
    n_out = len(output_modes)
    n_src = len(source_modes)
    L = np.zeros((n_out, n_src), dtype=np.complex128)

    # Source argument shift = r* + J* delta_star ; J* eta is the
    # eta-dependent piece.  But J* couples eta -> source-r-space, so
    # we need to substitute (eta_1, eta_2) -> J* (eta_1, eta_2) into
    # the source polynomial.  This is a linear coordinate change in
    # the source polynomial whose result is *another* polynomial in
    # eta of the same total degree.  Paper 2 leading order keeps this
    # affine substitution exactly.
    #
    # We implement it generically:  given p^src_n(r1, r2), substitute
    # r = (J* eta) + (r* + J* delta_star) -- the full affine
    # transformation -- and re-collect as polynomial in eta.

    r_const = s1_star + J_star @ np.array([delta_star[0], delta_star[1]]) - np.array([src_x, src_y])
    # Linear transform matrix on eta: r1 = J11 eta1 + J12 eta2 + r_const[0],
    # r2 = J21 eta1 + J22 eta2 + r_const[1].
    pupil_const = (np.array([v2x_star, v2y_star])
                   - np.array(v2_centre)
                   + np.array([delta_star[0], delta_star[1]]))

    # ------------------------------------------------------------------
    # 4.9 -- output-mode projection
    # ------------------------------------------------------------------
    # Pre-4.9 collapsed the output projection to ``out_poly.get((0, 0))``,
    # which is the constant term of the LG output polynomial.  This
    # zeroes out every ℓ ≠ 0 mode because
    # ``(σ_x + j·σ_y)^|ℓ| · Laguerre`` has no constant term for |ℓ| ≥ 1
    # -- silently producing zero coma/astigmatism/tilt tensor entries
    # even when the physical aberrations were present.  The audit's
    # action item #2.5 called this out and recommended either (a)
    # restricting the API to ℓ = 0 or (b) implementing the actual
    # σ-integration.  4.9 takes path (b).
    #
    # The σ-projection is exact at the leading-order asymptotic level:
    # the field at ``s2 = s2_image + σ`` is given by the same
    # saddle-point machinery that ``propagate_modal_asymptotic`` already
    # evaluates pixel-by-pixel.  Building a small grid around the chief
    # image and projecting numerically with ``decompose_lg`` does the
    # output Gaussian-moment integral against the LG_o basis without
    # collapsing the σ-dependence.
    #
    # v5.28.x (audit W3-T3):  the closed-form branch is a *point-sampling*
    # functional, not a projection -- its ENTIRE output-mode dependence is
    # the scalar ``out_const`` below, i.e. the output LG polynomial
    # evaluated at ONE point.  Measured on a 500 mm N-BK7 singlet
    # (w_s = 20 um, w_p = 0.05, on-axis):  the branch returns
    # ``L = U(chief) * N_{p,0}(w_o)`` to 6.9e-17, and
    # ``N_{p,0} = sqrt(2/(pi w_o^2))`` is INDEPENDENT of p, so
    # ``output_modes = [(0, 0), (1, 0), (2, 0), (3, 0), (5, 0)]`` all came
    # back BIT-IDENTICAL (max spread 2.0e-17) -- and separate single-mode
    # calls did too, with no warning emitted.  The pre-4.10.3 escape
    # ("go off-axis to lift the degeneracy") is a phantom: evaluating the
    # output polynomial at the ABSOLUTE image coordinate (which is not the
    # output-basis coordinate -- the sigma-grid branch below correctly
    # centres the basis ON s2_image) only perturbs the shared constant by
    # O((s2_img/w_o)^2), measured 1.9e-5 relative at s2 = 30 um, and that
    # perturbation is not the mode's overlap integral.
    #
    # So (0, 0) -- whose LG polynomial IS the bare constant N_{0,0} -- is
    # the only output mode the closed form can represent.  Everything else
    # routes to the sigma-integration, which an independent from-scratch
    # LG quadrature oracle reproduces to ~1e-14 relative.  Keeping the
    # pure-(0, 0) request on the closed form preserves the documented
    # cross-backend contract with ``aberration_tensor_lg00_jax`` (which
    # hardcodes exactly ``A_lead * N_s * N_p * N_o``).
    #
    # NOTE (known, pre-existing): the two branches do not share an overall
    # scale -- the closed form returns ``U(chief) * conj(LG_k(0))`` (units
    # of field/length) while the sigma branch returns the true overlap
    # ``integral conj(LG_k) U`` (units of field*length).  A request for
    # ``[(0, 0)]`` alone is therefore NOT on the same scale as the (0, 0)
    # entry of a multi-mode request.  Unifying them requires changing the
    # JAX twin's convention too; until then prefer multi-mode requests
    # (all entries mutually consistent overlaps) for anything but the
    # single Strehl-amplitude channel.
    #
    # ``needs_sigma_integration`` was decided above the ``w_o`` default,
    # which branches on it (audit W3-T3b: the two paths need different
    # defaults because ``w_o`` means different things in them).

    # Reported back on the result.  The closed-form branch has no σ grid and
    # no σ-dependent basis phase, so both stay None there (audit W4-T1/T2).
    n_grid: Optional[int] = None
    C_sigma: Optional[np.ndarray] = None

    if not needs_sigma_integration:
        # ---- Closed-form chief-ray sampling (output mode (0, 0) only) ---
        for io, k_out in enumerate(output_modes):
            out_poly_full = lg_polynomial(k_out[0], k_out[1], w_o)
            out_poly = {key: c.conjugate()
                        for key, c in out_poly_full.items()}
            for js, k_src in enumerate(source_modes):
                src_poly_r = lg_polynomial(k_src[0], k_src[1], w_s)
                src_poly_eta = _polynomial_substitute_linear_2d(
                    src_poly_r,
                    A_xx=J_star[0, 0], A_xy=J_star[0, 1],
                    A_yx=J_star[1, 0], A_yy=J_star[1, 1],
                    b_x=r_const[0], b_y=r_const[1],
                )
                T_acc = 0.0 + 0.0j
                for k_pup, b_pup in pupil_amplitudes.items():
                    if abs(b_pup) < 1e-300:
                        continue
                    pup_poly_r = lg_polynomial(k_pup[0], k_pup[1], w_p)
                    pup_poly_eta = _polynomial_under_affine_shift(
                        pup_poly_r,
                        shift_x=complex(pupil_const[0]),
                        shift_y=complex(pupil_const[1]),
                    )
                    P_eta = _multiply_polys_2d(src_poly_eta, pup_poly_eta)
                    exp_val = _contract_against_moment_table(
                        P_eta, eta_moments)
                    # Point sampling of the output basis at the chief
                    # ray.  Reachable ONLY for output mode (0, 0) since
                    # v5.28.x (audit W3-T3) -- see the routing note
                    # above -- and LG_{0,0}'s polynomial is the bare
                    # constant N_{0,0}, so this loop collapses to
                    # ``out_const = conj(N_{0,0}) = sqrt(2/(pi w_o^2))``
                    # regardless of (s2x_img, s2y_img).  The 4.10.3
                    # "evaluate at the absolute image coordinate"
                    # heuristic (which pretended to lift the p
                    # degeneracy but only perturbed the shared constant
                    # by O((s2_img/w_o)^2)) is therefore inert here; it
                    # is kept only so the expression stays a faithful
                    # generic polynomial evaluation.
                    out_const = 0.0 + 0.0j
                    for (ii, jj), c in out_poly.items():
                        out_const += c * (s2x_img ** ii) * (s2y_img ** jj)
                    T_acc += b_pup * out_const * exp_val
                L[io, js] = A_lead * T_acc
    else:
        # ---- Grid-based σ-integration (handles all ℓ) ------------------
        # Grid extent: ~4·w_o each side captures > 99.9 % of the LG basis
        # tail (the dominant ℓ_max scaling).  Sampling: 64 points across
        # 8·w_o gives ~ w_o/8 resolution, plenty for accurate trapezoidal
        # projection of LG modes up to (p, ℓ) ~ (3, 3).  Both knobs can
        # be tuned via the new ``sigma_grid_n`` / ``sigma_grid_extent``
        # kwargs added below for users who need higher orders or
        # tighter accuracy.
        # NOTE (audit W4-T1):  ``extent`` is resolved BEFORE ``n_grid``,
        # because the adaptive default for ``n_grid`` is a Nyquist condition
        # on the chirp across ``extent``.  Nothing in the extent branch reads
        # ``n_grid``, so the reorder is value-neutral.
        if sigma_grid_extent is not None:
            extent = float(sigma_grid_extent)
        else:
            extent = 4.0 * w_o
            # v5.28.x (audit W3-T3):  ``propagate_modal_asymptotic`` is
            # identically ZERO outside the fit's s2 validity box, so a
            # default grid that overshoots the box does not merely waste
            # samples -- it starves the quadrature completely.  With the
            # pre-W3-T3b pupil-scale ``w_o`` this fired hard (measured
            # 9.83e-3 "m" against a 1.54e-4 m half-box: +-4 w_o was 255x
            # outside the box, the 64x64 grid landed ZERO valid pixels and
            # EVERY entry of L came back exactly 0.0, silently resurrecting
            # the pre-4.9 all-zero coma/astigmatism/tilt bug).  The W3-T3b
            # default is measured INSIDE this box so the clamp is now a
            # guard rather than a load-bearing correction -- it still binds
            # when the field fills the box (4 x 1.559e-3 > 4.034e-3 room on
            # the validation singlet).  An explicit ``sigma_grid_extent``
            # is honoured verbatim.  Nothing is lost by the clamp: the
            # integrand vanishes outside the box.
            room = _s2_validity_room(fit, s2x_img, s2y_img)
            if room > 0.0:
                extent = min(extent, float(room))

        # ---- σ-grid size: ADAPTIVE default (audit W4-T1) ---------------
        # An explicit ``sigma_grid_n`` is honoured verbatim -- BIT-FOR-BIT
        # the pre-W4-T1 behaviour, including ``sigma_grid_n=64`` which
        # reproduces the old default exactly.
        n_req_chirp = None
        if sigma_grid_n is not None:
            n_grid = int(sigma_grid_n)
        else:
            # The old flat default 64 was measured to under-resolve this
            # chirp by 1.1x - 1.7x on the validation singlets, costing
            # +2.3 % to +33 % on the (2, 0) magnitude.  Size the grid from
            # the chirp instead:  n >= 4·extent·v_max/λ, rounded up onto
            # the cost ladder, clamped to ``sigma_grid_n_max``.
            n_req_chirp = _required_sigma_grid_n(fit, extent)
            n_cap = (_SIGMA_GRID_N_MAX_DEFAULT if sigma_grid_n_max is None
                     else max(_SIGMA_GRID_N_MIN, int(sigma_grid_n_max)))
            n_grid = _sigma_grid_n_on_ladder(n_req_chirp)
            if n_grid > n_cap:
                # ASCII-only: warning text reaches consoles/log handlers
                # whose encoding may be cp1252 (a 'sigma'/'lambda' glyph
                # here raised UnicodeEncodeError on this box's default
                # stdout).  The docstrings keep the maths glyphs.
                warnings.warn(
                    f"aberration_tensor: the image-plane field on this "
                    f"sigma grid is a chirp whose local fringe rate "
                    f"reaches v_max/lambda = "
                    f"{_sigma_chirp_v_max(fit) / float(fit.wavelength):.4g}"
                    f" cycles/m over half-extent {extent:.4g} m, so "
                    f"Nyquist needs sigma_grid_n >= {n_req_chirp} "
                    f"({n_grid} on the cost ladder); TRUNCATED to the cap "
                    f"sigma_grid_n_max = {n_cap}.  The (p, 0) channel "
                    f"MAGNITUDES are therefore aliasing-limited (measured "
                    f"~1e-1 relative at n = 64 against n = 1024); "
                    f"channel-to-channel and design-to-design RESPONSES "
                    f"are robust to it.  Raise sigma_grid_n_max, pass an "
                    f"explicit sigma_grid_n, or shrink "
                    f"sigma_grid_extent.",
                    UserWarning, stacklevel=2)
                n_grid = n_cap
        sx_arr = np.linspace(s2x_img - extent, s2x_img + extent, n_grid)
        sy_arr = np.linspace(s2y_img - extent, s2y_img + extent, n_grid)
        S2X, S2Y = np.meshgrid(sx_arr, sy_arr, indexing='xy')
        SX_local = S2X - s2x_img
        SY_local = S2Y - s2y_img

        max_p_o = max((k_out[0] for k_out in output_modes), default=0)
        max_ell_o = max((abs(k_out[1]) for k_out in output_modes),
                        default=0)

        # ---- Optional curvature-matched (complex-q) basis (W4-T2) -------
        # Projecting onto ``LG_k · exp(+i·π·σᵀCσ/λ)`` is identical to
        # projecting the PHASE-FLATTENED field ``U · exp(-i·π·σᵀCσ/λ)`` onto
        # the plain LG basis, because the overlap conjugates the mode.  Doing
        # it that way reuses ``decompose_lg`` (and its mode-stack cache)
        # unchanged, so the default path is untouched.
        if curvature_matched_basis:
            C_sigma = _measure_sigma_phase_curvature(
                fit, s2x_img, s2y_img, src_x, src_y, w_s, w_p,
                v2_centre, extent,
            )
            if C_sigma is None:
                warnings.warn(
                    "aberration_tensor: curvature_matched_basis=True but "
                    "the field's quadratic phase could not be measured "
                    "(the finite-difference probe left the fit's validity "
                    "box, or the Newton solves failed).  Falling back to "
                    "the default FLAT-phase LG basis, whose (p, 0) "
                    "channels are the non-smooth ones this flag exists to "
                    "fix (audit W4-T2).",
                    UserWarning, stacklevel=2)
            else:
                q_waves = (
                    (C_sigma[0, 0] * SX_local * SX_local
                     + 2.0 * C_sigma[0, 1] * SX_local * SY_local
                     + C_sigma[1, 1] * SY_local * SY_local)
                    / (2.0 * float(fit.wavelength))
                )
                basis_phase = np.exp(-2j * math.pi * q_waves)

        for js, k_src in enumerate(source_modes):
            # Field on the grid for THIS source mode (with the
            # caller's pupil_amplitudes weighting).
            src_amps = {k_src: 1.0 + 0.0j}
            field = propagate_modal_asymptotic(
                fit,
                source_point=(src_x, src_y),
                source_amplitudes=src_amps,
                pupil_amplitudes=pupil_amplitudes,
                w_s=w_s, w_p=w_p, v2_centre=v2_centre,
                s2_grid_x=S2X, s2_grid_y=S2Y,
            )
            if C_sigma is not None:
                field = np.asarray(field) * basis_phase
            # Only the caller's ``output_modes`` are read out of the result,
            # so only those are built: the (max_p_o, max_ell_o) rectangle the
            # two maxima span is the enclosing box of a set that is usually
            # much smaller than it (a (2, 0) / (1, 1) / (0, 3) selection spans
            # 21 modes and reads 3).  The overlaps are per-mode reductions
            # against independently-built modes, so this is the same number
            # for every mode that IS read.
            overlaps = decompose_lg(
                field, SX_local, SY_local,
                w=w_o, p_max=max_p_o, ell_max=max_ell_o,
                only=tuple(tuple(k_out) for k_out in output_modes),
            )
            for io, k_out in enumerate(output_modes):
                L[io, js] = overlaps.get(k_out, 0.0 + 0.0j)

    return AberrationTensorResult(
        L=L,
        output_modes=list(output_modes),
        source_modes=list(source_modes),
        pupil_modes=list(pupil_modes),
        s2_image=(s2x_img, s2y_img),
        w_s=w_s, w_p=w_p, w_o=w_o,
        v_star=(v2x_star, v2y_star),
        sigma_grid_n=n_grid,
        sigma_curvature=C_sigma,
        van_vleck_weight=_vv_weight_reported,
    )


def _polynomial_substitute_linear_2d(
    coeffs: Dict[Tuple[int, int], complex],
    A_xx: float, A_xy: float, A_yx: float, A_yy: float,
    b_x: float, b_y: float,
) -> Dict[Tuple[int, int], complex]:
    """Substitute (r_x, r_y) -> A * (eta_x, eta_y) + (b_x, b_y) in a
    2-D polynomial, returning the resulting polynomial in (eta_x, eta_y).

    Used to push the source polynomial through the linear J* map at
    the envelope-stationary point.
    """
    if not coeffs:
        return {}
    # First pre-compute (a x + b y + c)^n expansion as polynomial in (x, y).
    # We need (A_xx eta_x + A_xy eta_y + b_x)^i and similarly for y.

    def axes_pow(coef_a: complex, coef_b: complex, coef_c: complex,
                 n: int) -> Dict[Tuple[int, int], complex]:
        """Expand (a x + b y + c)^n via multinomial."""
        out: Dict[Tuple[int, int], complex] = {}
        # multinomial:  sum over (i, j, k) with i + j + k = n of
        #     n!/(i! j! k!) * a^i * b^j * c^k * x^i * y^j
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                w = (math.factorial(n)
                     // (math.factorial(i) * math.factorial(j)
                         * math.factorial(k)))
                key = (i, j)
                out[key] = out.get(key, 0.0 + 0.0j) + (
                    w * (coef_a ** i) * (coef_b ** j) * (coef_c ** k)
                )
        return out

    out: Dict[Tuple[int, int], complex] = {}
    # Cache the expansions of the linear forms raised to each needed power
    max_i = max(k[0] for k in coeffs)
    max_j = max(k[1] for k in coeffs)

    # (A_xx eta_x + A_xy eta_y + b_x)^i ; build for i = 0..max_i
    cache_x: List[Dict[Tuple[int, int], complex]] = []
    for n in range(max_i + 1):
        cache_x.append(axes_pow(
            complex(A_xx), complex(A_xy), complex(b_x), n
        ))
    # (A_yx eta_x + A_yy eta_y + b_y)^j ; build for j = 0..max_j
    cache_y: List[Dict[Tuple[int, int], complex]] = []
    for n in range(max_j + 1):
        cache_y.append(axes_pow(
            complex(A_yx), complex(A_yy), complex(b_y), n
        ))

    for (i, j), c in coeffs.items():
        # Multiply cache_x[i] * cache_y[j] and accumulate into out.
        prod = _multiply_polys_2d(cache_x[i], cache_y[j])
        for key, pc in prod.items():
            out[key] = out.get(key, 0.0 + 0.0j) + c * pc
    return out

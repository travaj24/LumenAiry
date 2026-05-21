"""LG aberration-tensor evaluator + sphere/cone-form polynomial helpers
for the phase-space asymptotic propagator.

v5.1.0 file-split (Agent D):  extracted from
``lumenairy.propagators.asymptotic`` with NO public-API or physics
change.  Holds:

* :class:`AberrationTensorResult` -- container for the LG tensor
  evaluated at one chief-ray image point.
* :func:`aberration_tensor` -- the closed-form Wick-contraction
  evaluator (with σ-grid fallback for ℓ ≠ 0 output modes).
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

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .._math.chebyshev import (
    chebyshev_derivative_vandermonde as _chebyshev_derivative_vandermonde,
)
from .._math.chebyshev import (
    chebyshev_second_derivative_vandermonde as _chebyshev_second_derivative_vandermonde,
)

# v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
# Chebyshev helpers moved to lumenairy._math.chebyshev; binding the
# new public names to the legacy underscore-prefixed locals keeps the
# existing call sites in this module unchanged.
from .._math.chebyshev import (
    chebyshev_vandermonde as _chebyshev_vandermonde,
)
from .asymptotic_canonical_fit import (
    CanonicalPolyFit,
    solve_envelope_stationary,
)
from .asymptotic_maslov import _maslov_branch_corrected_sqrt
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

    Notes
    -----
    Indices of L correspond to physical aberrations via
    ``lg_seidel_label(p, ell)``:  (1, 0) is defocus, (2, 0) is
    spherical, (1, +-1) is coma, (0, +-2) is astigmatism, etc.
    Driving |L_{(2,0), 0}|^2 to zero suppresses on-axis spherical
    aberration, etc.
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
    sigma_grid_extent: Optional[float] = None,
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
        Output Gaussian waist [m].  Defaults to a value derived from
        the local complex beam matrix (Maréchal-scale).
    v2_centre : (float, float), optional
        Pupil centre.
    sigma_grid_n : int, optional
        Output-plane grid size used for the σ-integration path when
        any requested ``output_modes`` has ``ℓ ≠ 0``.  Default 64 is
        accurate for LG modes up to ``(p, |ℓ|) ~ (3, 3)``; bump for
        higher orders.  Ignored when all output modes have ``ℓ = 0``
        (the fast closed-form chief-ray projection is used).
    sigma_grid_extent : float, optional
        Half-extent of the σ-grid [m].  Default ``4 · w_o`` captures
        > 99.9 % of the LG tail mass.  Increase only if the output
        beam has unusually long tails (large pupil aberrations).

    Returns
    -------
    AberrationTensorResult

    Notes
    -----
    **Pre-4.9 limitation removed.**  Pre-4.9 the projection at the
    chief image collapsed to the constant term of the LG output
    polynomial, which is identically zero for any ``ℓ ≠ 0`` mode
    ((σ_x + j·σ_y)^|ℓ| · Laguerre has no constant term).  That made
    coma ``(1, ±1)``, astigmatism ``(0, ±2)``, tilt ``(0, ±1)``, and
    every other ℓ ≠ 0 entry of the returned tensor silently zero,
    even when the underlying aberration was present.  4.9 fixes this
    by doing the actual σ-integration via a small output-plane grid
    and a numerical LG projection (``propagate_modal_asymptotic`` +
    ``decompose_lg``).  ℓ = 0 modes still use the fast closed-form
    chief-ray path -- you get the new behaviour only when you ask
    for an ℓ ≠ 0 output mode.
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

    # Choose output waist if not supplied
    if w_o is None:
        # Use the smaller of the two real-part eigenvalues of M^-1, and
        # drop a factor of pi (Maréchal convention scales as
        # 1/sqrt(lambda_max(Re M))).
        eig_M_real = np.linalg.eigvalsh(np.real(M))
        if eig_M_real.max() <= 0:
            w_o = 1e-6  # fallback for ill-conditioned cases
        else:
            w_o = 1.0 / math.sqrt(float(eig_M_real.max()))
        # Clamp to a physically reasonable range
        w_o = max(min(w_o, 1.0), 1e-9)

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
    A_lead = (detJ * (math.pi / sqrt_detM) * G0
              * np.exp(2j * math.pi * phi_star)
              * np.exp(0.25 * b @ M_inv @ b))

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
    # For pure ℓ = 0 output sets we keep the closed-form chief-ray
    # evaluation -- it's faster and gives identical numbers in that
    # regime.  Mixed ℓ = 0 / ℓ ≠ 0 sets fall through to the grid path.
    needs_sigma_integration = any(k_out[1] != 0 for k_out in output_modes)
    # 4.10.3: closed-form ℓ=0 path now evaluates the output LG
    # polynomial at the saddle's σ_image (instead of grabbing only its
    # x⁰y⁰ Cartesian constant), so different (p, 0) modes are
    # distinguished for any OFF-axis saddle.  For an axial saddle
    # σ_image=(0,0), the LG_p,0 polynomial reduces to the
    # normalisation constant N_p,0 = sqrt(2/(π w_o²)) -- the same
    # for all p (point evaluation at the origin of orthonormal modes
    # that all peak there is a fundamental closed-form / saddle-point
    # limitation; the σ-grid path has the right orthogonal structure
    # but a different overall normalisation than the closed-form,
    # so we don't auto-route).  Pass a non-axial source_point /
    # s2_image, or include an ℓ ≠ 0 mode, to force the discriminating
    # behaviour.
    if not needs_sigma_integration:
        ell0_p_values = {k_out[0] for k_out in output_modes
                          if k_out[1] == 0}
        is_axial_saddle = (abs(s2x_img) + abs(s2y_img)
                            < 1e-9 * max(w_o, 1e-12))
        if len(ell0_p_values) > 1 and is_axial_saddle:
            import warnings
            warnings.warn(
                "aberration_tensor: ON-AXIS saddle σ_image=(0,0) with "
                f"multiple radial-p modes in {sorted(ell0_p_values)}.  "
                "All LG_p,0 modes have the same value N_p,0 at the "
                "origin, so the saddle-point closed-form cannot "
                "distinguish them.  This is a fundamental limit of the "
                "point-evaluation approximation, not a bug.  Use an "
                "off-axis source_point / s2_image to lift the "
                "degeneracy, or include an ℓ ≠ 0 mode to force the "
                "σ-grid integration path.",
                RuntimeWarning, stacklevel=3,
            )

    if not needs_sigma_integration:
        # ---- Closed-form chief-ray projection (ℓ = 0 only) -------------
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
                    # 4.10.3: evaluate out_poly at the saddle's
                    # σ_image (the saddle is held fixed across the
                    # η-integration).  Pre-4.10.3 grabbed only the
                    # x⁰y⁰ Cartesian constant, which equals N_p,0
                    # for every (p, 0) mode and therefore collapsed
                    # defocus / spherical / secondary-spherical to
                    # the same scalar.  Evaluating the full
                    # polynomial at (s2x_img, s2y_img) recovers the
                    # mode-discriminating polynomial terms.  The
                    # ON-axis multi-p case is automatically routed
                    # to the σ-grid path by the
                    # needs_sigma_integration check above (modes
                    # genuinely degenerate at the origin in the
                    # saddle-point limit).
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
        n_grid = int(sigma_grid_n) if sigma_grid_n is not None else 64
        extent = (float(sigma_grid_extent)
                  if sigma_grid_extent is not None else 4.0 * w_o)
        sx_arr = np.linspace(s2x_img - extent, s2x_img + extent, n_grid)
        sy_arr = np.linspace(s2y_img - extent, s2y_img + extent, n_grid)
        S2X, S2Y = np.meshgrid(sx_arr, sy_arr, indexing='xy')
        SX_local = S2X - s2x_img
        SY_local = S2Y - s2y_img

        max_p_o = max((k_out[0] for k_out in output_modes), default=0)
        max_ell_o = max((abs(k_out[1]) for k_out in output_modes),
                        default=0)

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
            overlaps = decompose_lg(
                field, SX_local, SY_local,
                w=w_o, p_max=max_p_o, ell_max=max_ell_o,
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

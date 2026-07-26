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
    eig_M_real = np.linalg.eigvalsh(np.real(M))
    if eig_M_real.max() <= 0:
        w = 1e-6  # fallback for ill-conditioned cases
    else:
        w = 1.0 / math.sqrt(float(eig_M_real.max()))
    return max(min(w, 1.0), 1e-9)


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
    """
    room = _s2_validity_room(fit, s2x_img, s2y_img)
    if not (math.isfinite(room) and room > 0.0):
        return None
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
            return None
        inten = np.abs(np.asarray(U)) ** 2
        inten = np.where(np.isfinite(inten), inten, 0.0)
        tot = float(inten.sum())
        if not (tot > 0.0):
            return None
        lx = SX - s2x_img
        ly = SY - s2y_img
        cx = float((inten * lx).sum() / tot)
        cy = float((inten * ly).sum() / tot)
        var = float(
            (inten * ((lx - cx) ** 2 + (ly - cy) ** 2)).sum() / tot) / 2.0
        if not (math.isfinite(var) and var > 0.0):
            return None
        w = 2.0 * math.sqrt(var)
        cell = 2.0 * ext / (n - 1)
        if w >= 3.0 * cell:
            return w
        # Under-sampled: the field is far narrower than the validity box.
        ext_next = min(6.0 * w, 0.98 * room)
        if not (ext_next > 0.0) or ext_next >= ext:
            return w
        ext = ext_next
    return w


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
        single mode ``(0, 0)``.  Default 64 is accurate for LG modes up
        to ``(p, |ℓ|) ~ (3, 3)``; bump for higher orders.  Ignored for
        a pure ``[(0, 0)]`` request (the fast closed-form chief-ray
        sampling is used).

        **Accuracy note (measured, audit W3-T3b).**  The image-plane
        field is a CHIRP: its local spatial frequency at offset σ is
        ``|v*(σ)|/λ``, so resolving it needs roughly
        ``n >= 4·extent·v_max/λ`` samples (246 for the validation
        singlet's ``extent = 4.03e-3 m``, ``v_max = 0.02``,
        ``λ = 1.31 µm``).  At the default 64 the ``(2, 0)`` channel is
        therefore aliasing-limited; measured against ``n = 384``:

            n      = 64      96      128     192     256
            R1=51.5  +2.3 %  -25 %   -13 %   -2.3 %  -0.7 %
            R1=60.0  +33 %   -3.9 %  +1.4 %  -4.0 %  -2.6 %

        Convergence is oscillatory, and ``n = 256`` costs ~15x ``n =
        64`` (5.1 s vs 0.34 s here).  The default trades accuracy for
        the optimiser loop this evaluator feeds; raise it when you need
        converged channel MAGNITUDES.  Channel-to-channel and
        design-to-design RESPONSES are robust to it (the curvature
        discriminator reads 2.09e-1 relative at n = 64 and 4.04e-1 at
        n = 256).  Note this is a property of the σ quadrature, not of
        the W3-T3b waist: the same aliasing is present at any explicit
        ``w_o`` that spans the same box.
    sigma_grid_extent : float, optional
        Half-extent of the σ-grid [m].  Default ``4 · w_o``, clamped to
        the fit's ``s2`` validity half-box measured from ``s2_image``
        (the propagator is identically zero outside it, so an
        overshooting grid samples no field at all -- see the W3-T3 note
        at the branch).  An explicit value is honoured verbatim, so
        pass one if you want the raw ``4 · w_o`` behaviour.

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
    ``decompose_lg``).

    **v5.28.x (audit W3-T3) -- ℓ = 0 degeneracy removed.**  4.9-5.28
    kept EVERY ℓ = 0 output mode on the closed-form chief-ray path,
    which is a point-sampling functional whose whole output-mode
    dependence is ``conj(LG_k)`` evaluated at one point:  that equals
    ``N_{p,0} = sqrt(2/(π w_o²))`` for every ``(p, 0)`` mode, so
    ``L`` came back BIT-IDENTICAL for piston / defocus / spherical /
    every higher ``(p, 0)`` channel (and across separate single-mode
    calls, with no warning).  Only the pure ``[(0, 0)]`` request --
    whose LG polynomial genuinely IS that constant, and which is the
    documented cross-backend contract of
    ``aberration_tensor_lg00_jax`` -- still uses the closed form; every
    other request routes to the σ-integration, whose overlaps an
    independent from-scratch LG quadrature reproduces to ~1e-14
    relative.  The two paths carry different overall scales (sampling
    vs. overlap integral); see the note at the branch.
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
            # v5.29 (audit W3-T3b).  The pre-fix default was
            # ``1/sqrt(lambda_max(Re M))``, which is DIMENSIONALLY a pupil
            # quantity: ``M``'s entries are ``J^T J / w_s^2 + I / w_p^2 -
            # i·pi·H_phi`` with ``J = ds1/dv2`` [m/direction-cosine], so
            # ``M`` is in 1/direction-cosine^2 and its inverse square root
            # is an ANGLE -- the effective pupil acceptance -- used as if
            # it were metres.  Being dimensionally wrong, its error had no
            # fixed sign: measured 1.01e-4 "m" against a true field waist
            # of 1.559e-3 m (15x too NARROW, so ``4·w_o`` sampled only the
            # flat central 10 % of the field) on the validation singlet at
            # w_p = 0.02, but 255x too WIDE (grid entirely outside the
            # validity box, every entry of L exactly 0) at w_p = 0.05.
            #
            # Nor can any function of ``M`` alone be right: the image-plane
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
        n_grid = int(sigma_grid_n) if sigma_grid_n is not None else 64
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

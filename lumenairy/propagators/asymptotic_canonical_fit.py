"""Canonical-form Chebyshev polynomial fits driving the
phase-space asymptotic propagator.

v5.1.0 file-split (Agent D):  extracted from
``lumenairy.propagators.asymptotic`` with NO public-API or physics
change.  Holds:

* :class:`CanonicalPolyFit` -- dataclass for the 4-D
  Chebyshev fit of ``Phi(s2, v2)`` and ``s1(s2, v2)`` used by the
  Maslov saddle-point propagator.
* :class:`HFPolyFit` -- counterpart for direct Huygens-Fresnel
  quadrature parametrised on ``Phi(s1, s2)``.
* :func:`fit_canonical_polynomials` / :func:`fit_hf_polynomials` --
  trace-and-fit pipelines.
* :func:`solve_envelope_stationary` -- Newton solver for the
  envelope-stationary saddle point.
* :func:`propagate_hf_chebyshev_quadrature` -- direct HF
  quadrature evaluator using ``HFPolyFit``.

All re-exported through ``lumenairy.propagators.asymptotic`` so
existing call sites continue to work unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..elements.lenses import (
    _chebyshev_vandermonde,
    _chebyshev_derivative_vandermonde,
    _chebyshev_second_derivative_vandermonde,
    _multi_indices_total_degree,
    _evaluate_polynomial_4d,
    _evaluate_polynomial_4d_and_grad34,
    _fit_normaliser,
)


__all__ = [
    'CanonicalPolyFit',
    'HFPolyFit',
    'fit_canonical_polynomials',
    'fit_hf_polynomials',
    'solve_envelope_stationary',
    'propagate_hf_chebyshev_quadrature',
]


# ===========================================================================
# Section 3 -- Canonical polynomial fit (Phi(s2, v2) and s1(s2, v2))
# ===========================================================================

@dataclass
class CanonicalPolyFit:
    """Result of a Chebyshev tensor-product fit to ray-traced
    ``Phi(s2, v2)`` and ``s1(s2, v2)``.

    Fields
    ------
    poly_order : int
        Total-degree truncation used in the fit (``|k|_1 <= poly_order``).
    multi_indices : list of 4-tuples
        Multi-indices ``(k1, k2, k3, k4)`` enumerating the basis terms,
        in the same order as the coefficient vectors.
    coef_phi : ndarray
        Coefficients of the Phi polynomial (in waves), residual after
        linear-phase extraction.
    coef_s1x, coef_s1y : ndarray
        Coefficients of the s1 back-map components (in metres).
    linear_coeffs_phi : ndarray, optional
        ``(alpha_0, alpha_1, alpha_2, alpha_3, alpha_4)`` of the linear
        prefit of Phi.  None if the fit was done
        with ``extract_linear_phase=False``.
    s2x_centre, s2x_halfrange : float
        Normalisation parameters for s2x:  ``u_s2x = (s2x - centre) /
        halfrange``.
    s2y_centre, s2y_halfrange : float
        Same for s2y.
    v2x_centre, v2x_halfrange : float
        Same for v2x.
    v2y_centre, v2y_halfrange : float
        Same for v2y.
    wavelength : float
        Wavelength used for the fit [m].
    res_phi_rms_waves : float
        RMS fit residual on the Chebyshev-node training set, in waves.
    res_s1_rms_m : float
        RMS fit residual on s1, in metres.
    n_rays : int
        Number of training rays.
    """
    poly_order: int
    multi_indices: List[Tuple[int, int, int, int]]
    coef_phi: np.ndarray
    coef_s1x: np.ndarray
    coef_s1y: np.ndarray
    s2x_centre: float
    s2x_halfrange: float
    s2y_centre: float
    s2y_halfrange: float
    v2x_centre: float
    v2x_halfrange: float
    v2y_centre: float
    v2y_halfrange: float
    wavelength: float
    res_phi_rms_waves: float = 0.0
    res_s1_rms_m: float = 0.0
    n_rays: int = 0
    linear_coeffs_phi: Optional[np.ndarray] = None
    extract_linear_phase: bool = False

    # ------------------------------------------------------------------
    # Coordinate normalisation helpers
    # ------------------------------------------------------------------
    def to_normalised(self, s2x: np.ndarray, s2y: np.ndarray,
                      v2x: np.ndarray, v2y: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map physical (s2, v2) into normalised [-1, 1]^4."""
        u1 = (s2x - self.s2x_centre) / self.s2x_halfrange
        u2 = (s2y - self.s2y_centre) / self.s2y_halfrange
        u3 = (v2x - self.v2x_centre) / self.v2x_halfrange
        u4 = (v2y - self.v2y_centre) / self.v2y_halfrange
        return u1, u2, u3, u4

    def in_box(self, s2x: np.ndarray, s2y: np.ndarray,
               v2x: np.ndarray, v2y: np.ndarray) -> np.ndarray:
        """Boolean mask of points inside the fit's normalised box."""
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        return ((np.abs(u1) <= 1.0) & (np.abs(u2) <= 1.0)
                & (np.abs(u3) <= 1.0) & (np.abs(u4) <= 1.0))

    # ------------------------------------------------------------------
    # Polynomial evaluation helpers
    # ------------------------------------------------------------------
    def eval_s1(self, s2x: np.ndarray, s2y: np.ndarray,
                v2x: np.ndarray, v2y: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the back-map s1(s2, v2) -> (s1x, s1y) at arbitrary
        (s2, v2) samples in physical units."""
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        s1x = _evaluate_polynomial_4d(self.coef_s1x, self.multi_indices,
                                       u1, u2, u3, u4, self.poly_order)
        s1y = _evaluate_polynomial_4d(self.coef_s1y, self.multi_indices,
                                       u1, u2, u3, u4, self.poly_order)
        return s1x, s1y

    def eval_phi(self, s2x: np.ndarray, s2y: np.ndarray,
                 v2x: np.ndarray, v2y: np.ndarray,
                 *, include_linear: bool = True) -> np.ndarray:
        """Evaluate Phi(s2, v2) [waves] at arbitrary samples.

        Parameters
        ----------
        include_linear : bool, default True
            If True and the fit extracted a linear ramp, re-add it to
            give the *raw* OPD that the ray trace would report.  If
            False, return only the Chebyshev residual (the
            integrator-safe form).
        """
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        phi = _evaluate_polynomial_4d(self.coef_phi, self.multi_indices,
                                       u1, u2, u3, u4, self.poly_order)
        if include_linear and self.linear_coeffs_phi is not None:
            a0, a1, a2, a3, a4 = self.linear_coeffs_phi
            phi = phi + (a0 + a1 * u1 + a2 * u2 + a3 * u3 + a4 * u4)
        return phi

    def eval_s1_with_v2_grad(self, s2x: np.ndarray, s2y: np.ndarray,
                              v2x: np.ndarray, v2y: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate s1(s2, v2) and its (v2x, v2y)-gradient (Jacobian).

        Returns
        -------
        s1x, s1y, dS1x_dv2x, dS1x_dv2y, dS1y_dv2x, dS1y_dv2y
            All in physical units; gradients have the chain-rule
            normalisation factor 1/v2x_halfrange and 1/v2y_halfrange
            already applied.
        """
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        s1x, du3_s1x, du4_s1x = _evaluate_polynomial_4d_and_grad34(
            self.coef_s1x, self.multi_indices, u1, u2, u3, u4,
            self.poly_order)
        s1y, du3_s1y, du4_s1y = _evaluate_polynomial_4d_and_grad34(
            self.coef_s1y, self.multi_indices, u1, u2, u3, u4,
            self.poly_order)
        # Chain rule:  d/dv2x = (1/v2x_halfrange) d/du3 ; same for v2y
        invhx = 1.0 / self.v2x_halfrange
        invhy = 1.0 / self.v2y_halfrange
        return (s1x, s1y,
                du3_s1x * invhx, du4_s1x * invhy,
                du3_s1y * invhx, du4_s1y * invhy)

    def eval_phi_with_v2_grad(self, s2x: np.ndarray, s2y: np.ndarray,
                               v2x: np.ndarray, v2y: np.ndarray,
                               *, include_linear: bool = False
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate Phi and its v2-gradient.  ``include_linear``
        controls whether the linear-phase prefit is re-added (default
        False since the integrator drops it)."""
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        phi, du3_phi, du4_phi = _evaluate_polynomial_4d_and_grad34(
            self.coef_phi, self.multi_indices, u1, u2, u3, u4,
            self.poly_order)
        if include_linear and self.linear_coeffs_phi is not None:
            a0, a1, a2, a3, a4 = self.linear_coeffs_phi
            phi = phi + (a0 + a1 * u1 + a2 * u2 + a3 * u3 + a4 * u4)
            du3_phi = du3_phi + a3
            du4_phi = du4_phi + a4
        invhx = 1.0 / self.v2x_halfrange
        invhy = 1.0 / self.v2y_halfrange
        return phi, du3_phi * invhx, du4_phi * invhy


def fit_canonical_polynomials(
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    source_box_half: float = 50e-6,
    pupil_box_half: float = 0.05,
    n_field: int = 8,
    n_pupil: int = 8,
    poly_order: int = 6,
    extract_linear_phase: bool = True,
    object_distance: Optional[float] = None,
    surface_diffraction: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    endpoint_anchored: bool = False,
    auto_bump_threshold_waves: Optional[float] = None,
    max_auto_poly_order: int = 10,
    source_centre: Tuple[float, float] = (0.0, 0.0),
) -> CanonicalPolyFit:
    """Fit Chebyshev tensor-product polynomials to ``Phi(s2, v2)`` and
    ``s1(s2, v2)`` over a 4-D source x pupil grid.

    The trace-and-fit pipeline samples a Chebyshev-node grid in the
    source-plane coordinates ``(s1_x, s1_y)`` and the input direction
    cosines ``(v1_x, v1_y)``, traces each ray forward through the
    prescription, and records the landing position ``s2`` and direction
    cosine ``v2`` at the last surface (along with the accumulated
    optical path).  The 4-D scattered training set
    ``{(s2, v2, s1, Phi)}`` then drives a least-squares fit of
    ``s1(s2, v2)`` and ``Phi(s2, v2)`` as 4-variable Chebyshev tensor-
    product polynomials.

    Sampling source positions in addition to pupil directions is what
    gives the back-map ``s1(s2, v2)`` non-trivial dependence on (s2,
    v2):  with a single source point the trace produces a 2-D
    submanifold of (s2, v2) values along which ``s1 = const`` and the
    fit's J = d s1 / d v2 collapses to zero.

    Parameters
    ----------
    prescription : dict
        lumenairy prescription dict (e.g. from
        ``load_zemax_zmx`` or a builder).
    wavelength : float
        Wavelength [m].
    source_box_half : float
        Half-width of the source sampling box [m].  Set to a few times
        the source-Gaussian waist consumers will use (typical 50 um).
    pupil_box_half : float
        Half-width of the input direction-cosine box [dimensionless].
        Set to cover the system aperture as seen from the source plane.
    n_field : int
        Per-axis Chebyshev-node grid size for source sampling.
    n_pupil : int
        Per-axis grid size for input direction sampling.  Total ray
        count is ``n_field**2 * n_pupil**2``.
    poly_order : int
        Total-degree truncation of the Chebyshev fit.  Default 6.
    extract_linear_phase : bool
        If True (default), pre-fit and subtract a 5-parameter linear
        ramp from Phi before the Chebyshev fit -- this restores Nyquist
        sampling for diffractive surfaces at non-zero orders (
        Section 5).  No-op for refractive systems.
    object_distance : float, optional
        Source-plane to first-surface distance [m].  Defaults to
        ``prescription.get('object_distance', 0.0)``.
    surface_diffraction : dict, optional
        Per-surface diffractive-order kicks for DOE / grating elements
        embedded in the prescription.  Maps surface index to
        ``(order_x, order_y, period_x, period_y)``.  Each ray's
        direction cosines are shifted by ``m * lambda / Lambda`` after
        refraction at that surface, evanescent orders are flagged
        ``alive=False``.  Use this to pin the LG fit to a single
        diffraction order of a Dammann splitter or any thin grating in
        a sequential prescription -- the resulting fit's chief-ray
        landings span the chosen order's image-plane footprint, so
        :func:`aberration_tensor` evaluations at that order's frame
        centres reflect the true diffractive wavefront.  See also
        :func:`lumenairy.raytrace.apply_doe_phase_traced`.
    endpoint_anchored : bool, optional
        If True, the Chebyshev sampling nodes for the input source-
        plane and pupil-direction-cosine grids are rescaled so the
        outermost two nodes per axis sit exactly at ``+- box_half``
        instead of slightly inside (``cos(pi/(2*N)) * box_half``).
        Equivalent to ``cos((k + 1/2)*pi / N) / cos(pi/(2*N))``.
        Marginally extends the fit's training-data support to the box
        edge; the conditioning hit on the least-squares fit is
        negligible at typical ``poly_order`` (4 - 6) but the fit
        coefficients change very slightly so this is opt-in for
        bit-compatibility with prior fits.  Default ``False`` (use
        the standard Chebyshev-Gauss roots).

    Returns
    -------
    CanonicalPolyFit
        Fit container.  ``fit.res_phi_rms_waves`` is typically below
        1e-4 waves on a well-conditioned refractive design.

    Raises
    ------
    ValueError
        On invalid input.
    RuntimeError
        If too many rays die for a meaningful fit.
    """
    from ..raytrace import surfaces_from_prescription, _make_bundle, trace

    if wavelength <= 0:
        raise ValueError(f"wavelength must be > 0, got {wavelength}")
    if poly_order < 0:
        raise ValueError(f"poly_order must be >= 0, got {poly_order}")
    if source_box_half <= 0 or pupil_box_half <= 0:
        raise ValueError("source_box_half and pupil_box_half must be > 0")

    if object_distance is None:
        object_distance = float(prescription.get('object_distance', 0.0)) or 0.0

    surfaces = surfaces_from_prescription(prescription)

    # 4-D Chebyshev-node grid:  (s1_x, s1_y) x (v1_x, v1_y).
    def cheb_nodes(n: int) -> np.ndarray:
        i = np.arange(n)
        x = np.cos(np.pi * (i + 0.5) / n)
        if endpoint_anchored and n >= 2:
            # Rescale so outermost nodes sit at exactly +-1 instead of
            # +-cos(pi/(2n)).  Outermost training samples then graze the
            # box edge.  Conditioning hit negligible for poly_order <= 6
            # with 6+ samples per axis.
            x = x / np.cos(np.pi / (2.0 * n))
        return x

    u_field = cheb_nodes(n_field)        # in (-1, 1)
    u_pupil = cheb_nodes(n_pupil)
    # 4.11.2: shift source-sampling box by ``source_centre`` so the
    # subaperture / patch-decomposition propagator can fit a local
    # polynomial centred on each patch's object-plane footprint.
    # Pre-4.11.2 the source box was hard-pinned to the origin, so the
    # subaperture loop's per-patch fits were all identical and only
    # the on-axis patch had meaningful chief-ray landings inside the
    # output evaluation grid -- off-axis patches saw a fit whose
    # ``s2x_centre`` / ``s2y_centre`` straddled the on-axis image and
    # were therefore mostly out-of-box (``|u| > 1``) at the
    # off-axis evaluation pixels, producing zero field.
    cx_src, cy_src = float(source_centre[0]), float(source_centre[1])
    s1x_axis = u_field * source_box_half + cx_src
    s1y_axis = u_field * source_box_half + cy_src
    v1x_axis = u_pupil * pupil_box_half
    v1y_axis = u_pupil * pupil_box_half

    S1X, S1Y, V1X, V1Y = np.meshgrid(s1x_axis, s1y_axis,
                                      v1x_axis, v1y_axis,
                                      indexing='ij')
    s1x_in = S1X.ravel()
    s1y_in = S1Y.ravel()
    v1x_in = V1X.ravel()
    v1y_in = V1Y.ravel()
    n_rays = s1x_in.size

    # Reject any input direction with sin^2(theta) >= 1 (non-real
    # longitudinal cosine):  this means pupil_box_half is too large.
    sumsq = v1x_in * v1x_in + v1y_in * v1y_in
    if np.any(sumsq >= 1.0):
        raise ValueError(
            "pupil_box_half too large -- input direction cosines would "
            "be non-real.  Reduce pupil_box_half "
            f"(current = {pupil_box_half}).")

    # Build a ray bundle launched at z = -object_distance.  trace()
    # automatically propagates the rays the appropriate distance to the
    # first surface (since the surfaces' z positions are determined by
    # the prescription's accumulated thicknesses).
    bundle = _make_bundle(
        x=s1x_in.astype(np.float64),
        y=s1y_in.astype(np.float64),
        L=v1x_in.astype(np.float64),
        M=v1y_in.astype(np.float64),
        wavelength=wavelength,
    )
    bundle.z = np.full(n_rays, -object_distance, dtype=np.float64)

    res = trace(bundle, surfaces, wavelength, output_filter='last',
                surface_diffraction=surface_diffraction)
    final = res.image_rays
    alive = np.asarray(final.alive, dtype=bool)
    n_alive = int(alive.sum())
    if n_alive < max(64, 0.5 * n_rays):
        raise RuntimeError(
            f"Too many rays died during canonical-fit trace: "
            f"{n_alive} alive of {n_rays}.  Reduce pupil_box_half, "
            f"check prescription apertures, or rebalance the source/"
            f"pupil sampling boxes.")

    s2x_obs = np.asarray(final.x, dtype=np.float64)
    s2y_obs = np.asarray(final.y, dtype=np.float64)
    v2x_obs = np.asarray(final.L, dtype=np.float64)
    v2y_obs = np.asarray(final.M, dtype=np.float64)
    phi_obs = np.asarray(final.opd, dtype=np.float64) / wavelength

    s2x_live = s2x_obs[alive]
    s2y_live = s2y_obs[alive]
    v2x_live = v2x_obs[alive]
    v2y_live = v2y_obs[alive]
    phi_live = phi_obs[alive]
    s1x_live = s1x_in[alive]
    s1y_live = s1y_in[alive]

    # Normalise the OUTPUT (s2, v2) coordinates to [-1, 1]^4 from the
    # observed extents of the live rays.  Rays that end
    # up at extreme s2 or v2 set the box; the 5%-pad in _fit_normaliser
    # leaves room for evaluation at boundary pixels.
    s2x_c, s2x_h = _fit_normaliser(s2x_live)
    s2y_c, s2y_h = _fit_normaliser(s2y_live)
    v2x_c, v2x_h = _fit_normaliser(v2x_live)
    v2y_c, v2y_h = _fit_normaliser(v2y_live)

    u_s2x = (s2x_live - s2x_c) / s2x_h
    u_s2y = (s2y_live - s2y_c) / s2y_h
    u_v2x = (v2x_live - v2x_c) / v2x_h
    u_v2y = (v2y_live - v2y_c) / v2y_h

    linear_coeffs = None
    if extract_linear_phase:
        X5 = np.column_stack([
            np.ones_like(u_s2x),
            u_s2x, u_s2y, u_v2x, u_v2y,
        ])
        linear_coeffs, *_ = np.linalg.lstsq(X5, phi_live, rcond=None)
        opd_residual = phi_live - X5 @ linear_coeffs
    else:
        opd_residual = phi_live.copy()

    multi_indices = _multi_indices_total_degree(4, poly_order)
    n_basis = len(multi_indices)
    T1 = _chebyshev_vandermonde(u_s2x, poly_order)
    T2 = _chebyshev_vandermonde(u_s2y, poly_order)
    T3 = _chebyshev_vandermonde(u_v2x, poly_order)
    T4 = _chebyshev_vandermonde(u_v2y, poly_order)
    # 3.5.6: Vectorised design-matrix build via fancy indexing,
    # replacing the per-basis-term Python loop.  Each T_i has shape
    # (poly_order+1, n_rays); T_i[K_i] has shape (n_basis, n_rays);
    # the elementwise product has shape (n_basis, n_rays); transpose
    # to (n_rays, n_basis).  Avoids the n_basis-iteration Python
    # loop with no temporary allocations beyond the result.
    K_arr = np.asarray(multi_indices, dtype=np.int64)
    K1, K2, K3, K4 = K_arr[:, 0], K_arr[:, 1], K_arr[:, 2], K_arr[:, 3]
    A = (T1[K1] * T2[K2] * T3[K3] * T4[K4]).T.astype(
        np.float64, copy=False)

    coef_phi, *_ = np.linalg.lstsq(A, opd_residual, rcond=None)
    coef_s1x, *_ = np.linalg.lstsq(A, s1x_live, rcond=None)
    coef_s1y, *_ = np.linalg.lstsq(A, s1y_live, rcond=None)

    res_phi = float(np.sqrt(np.mean((opd_residual - A @ coef_phi) ** 2)))
    res_s1 = float(np.sqrt(0.5 * np.mean(
        (s1x_live - A @ coef_s1x) ** 2 + (s1y_live - A @ coef_s1y) ** 2
    )))

    # 3.5.6: optional automatic poly_order bump if the residual exceeds
    # the user-supplied threshold.  Useful for cemented multi-element
    # systems where order=6 (the default) under-fits.  Caller passes
    # ``auto_bump_threshold_waves=1e-3`` to opt in; default None
    # preserves the prior single-shot behaviour.
    if (auto_bump_threshold_waves is not None
            and res_phi > float(auto_bump_threshold_waves)
            and poly_order < int(max_auto_poly_order)):
        import warnings as _w
        _w.warn(
            f"fit_canonical_polynomials: order={poly_order} fit "
            f"residual {res_phi:.3e} waves exceeds threshold "
            f"{auto_bump_threshold_waves:.3e}; auto-bumping to "
            f"order={poly_order + 2} (max {max_auto_poly_order}).",
            RuntimeWarning, stacklevel=2)
        return fit_canonical_polynomials(
            prescription, wavelength,
            source_box_half=source_box_half,
            pupil_box_half=pupil_box_half,
            n_field=n_field, n_pupil=n_pupil,
            poly_order=poly_order + 2,
            extract_linear_phase=extract_linear_phase,
            object_distance=object_distance,
            surface_diffraction=surface_diffraction,
            endpoint_anchored=endpoint_anchored,
            auto_bump_threshold_waves=auto_bump_threshold_waves,
            max_auto_poly_order=max_auto_poly_order,
            source_centre=source_centre,
        )

    return CanonicalPolyFit(
        poly_order=poly_order,
        multi_indices=multi_indices,
        coef_phi=coef_phi,
        coef_s1x=coef_s1x,
        coef_s1y=coef_s1y,
        s2x_centre=s2x_c, s2x_halfrange=s2x_h,
        s2y_centre=s2y_c, s2y_halfrange=s2y_h,
        v2x_centre=v2x_c, v2x_halfrange=v2x_h,
        v2y_centre=v2y_c, v2y_halfrange=v2y_h,
        wavelength=wavelength,
        res_phi_rms_waves=res_phi,
        res_s1_rms_m=res_s1,
        n_rays=n_alive,
        linear_coeffs_phi=linear_coeffs,
        extract_linear_phase=extract_linear_phase,
    )


# ===========================================================================
# Section 4 -- Newton solver for the envelope-stationary point
# ===========================================================================

def solve_envelope_stationary(
    fit: CanonicalPolyFit,
    s2: Tuple[float, float],
    source_point: Tuple[float, float],
    *,
    w_s: float,
    w_p: float,
    v2_centre: Tuple[float, float] = (0.0, 0.0),
    v2_initial: Optional[Tuple[float, float]] = None,
    max_iter: int = 12,
    tol: float = 1e-12,
) -> Tuple[Tuple[float, float], int, float]:
    """Newton-solve the envelope-stationary equation.

    Find ``v_2^*`` such that the joint Gaussian envelope

        G(v_2) = exp(-|s_1(s_2, v_2) - s_src|^2 / w_s^2
                     - |v_2 - v_2_centre|^2 / w_p^2)

    is locally maximised at ``s_2``.  Equation (9) reads

        J^T (s_1 - s_src) / w_s^2 + (v_2 - v_2_centre) / w_p^2 = 0

    with ``J = ds_1 / dv_2``.

    Parameters
    ----------
    fit : CanonicalPolyFit
        Polynomial fit of the system.
    s2 : (float, float)
        Output-plane point [m].
    source_point : (float, float)
        Source-plane point [m].
    w_s, w_p : float
        Source and pupil Gaussian waists.  ``w_s`` in [m] (object plane);
        ``w_p`` in direction-cosine units [dimensionless].
    v2_centre : (float, float), optional
        Pupil centre in direction cosines.  Default origin.
    v2_initial : (float, float), optional
        Newton initial guess.  Defaults to ``v2_centre``.
    max_iter : int, optional
        Iteration cap.  Newton typically converges in 3-5; cap at 12.
    tol : float, optional
        Convergence tolerance on |residual|.  Default 1e-12.

    Returns
    -------
    v_star : (float, float)
        The stationary point.
    n_iter : int
        Iterations actually used.
    residual_norm : float
        Final |residual|.
    """
    s2x, s2y = float(s2[0]), float(s2[1])
    src_x, src_y = float(source_point[0]), float(source_point[1])
    v_cx, v_cy = float(v2_centre[0]), float(v2_centre[1])
    if v2_initial is None:
        v2x = v_cx
        v2y = v_cy
    else:
        v2x = float(v2_initial[0])
        v2y = float(v2_initial[1])

    inv_ws2 = 1.0 / (w_s * w_s)
    inv_wp2 = 1.0 / (w_p * w_p)

    last_norm = float('inf')
    for it in range(max_iter):
        # Evaluate s1, J at current v2
        s2x_arr = np.asarray(s2x).reshape(())
        s2y_arr = np.asarray(s2y).reshape(())
        v2x_arr = np.asarray(v2x).reshape(())
        v2y_arr = np.asarray(v2y).reshape(())
        s1x, s1y, dS1x_dv2x, dS1x_dv2y, dS1y_dv2x, dS1y_dv2y = (
            fit.eval_s1_with_v2_grad(s2x_arr, s2y_arr, v2x_arr, v2y_arr)
        )
        s1x = float(s1x)
        s1y = float(s1y)
        # Jacobian of s_1 w.r.t. v_2
        J = np.array([
            [float(dS1x_dv2x), float(dS1x_dv2y)],
            [float(dS1y_dv2x), float(dS1y_dv2y)],
        ])
        delta_s1 = np.array([s1x - src_x, s1y - src_y])
        delta_v = np.array([v2x - v_cx, v2y - v_cy])
        residual = inv_ws2 * (J.T @ delta_s1) + inv_wp2 * delta_v
        rn = float(np.linalg.norm(residual))
        # 4.10: stricter stall check.  Pre-4.10 the chained comparison
        # `rn < tol or rn > 0.99 * last_norm > 1e-300` could fire on
        # the FIRST iteration (since last_norm = inf initially makes
        # 0.99*inf > rn trivially true on the right end of the chain
        # while the left end is false, but the overall expression
        # depends on Python's chaining semantics).  More importantly,
        # the stall heuristic fired even when one of two consecutive
        # iterations was sub-quadratic during a normal Newton
        # warm-up.  Require AT LEAST 2 iterations before declaring
        # a stall, and require BOTH halves of the chain to be true
        # explicitly to avoid the chained-comparison surprise.
        is_converged = rn < tol
        is_stalling = (it >= 2 and rn > 0.9 * last_norm and last_norm > 1e-300)
        if is_converged or is_stalling:
            return (v2x, v2y), it, rn
        last_norm = rn

        # Approximate Hessian:  d residual / d v2.
        # First term:  d/dv2 [ J^T (s_1 - s_src) ] = J^T @ J + sum_k
        #                    (s_1 - s_src)_k * d^2 s_1_k / dv2 dv2.
        # Hessian-of-s1 piece is second-order in (s_1 - s_src) and is
        # neglected here (Gauss-Newton-like).  This is exact at the
        # stationary point and gives quadratic-ish convergence.
        # Second term:  d/dv2 [(v_2 - v_2c) / w_p^2] = I / w_p^2.
        H = inv_ws2 * (J.T @ J) + inv_wp2 * np.eye(2)
        try:
            step = np.linalg.solve(H, residual)
        except np.linalg.LinAlgError:
            # Singular Hessian -- back off and bail.
            return (v2x, v2y), it, rn
        v2x = v2x - float(step[0])
        v2y = v2y - float(step[1])

    return (v2x, v2y), max_iter, last_norm


# ===========================================================================
# Section 8 -- HF-form polynomial fit:  Phi(s1, s2)
# ===========================================================================
#
# The canonical fit (Section 3) parametrises Phi as a function of (s2, v2)
# -- the natural coordinates for the asymptotic / Maslov saddle-point
# evaluation.  For direct Huygens-Fresnel quadrature without any
# saddle-point approximation we want Phi expressed as a function of the
# endpoints (s1, s2) so that the HF integral
#
#     E_HF(s2) = integral E_in(s1) sqrt(|det d2 Phi / d s1 d s2|)
#                * exp(2 pi i Phi(s1, s2)) d^2 s1
#
# can be evaluated by 2-D quadrature over s1 with no inverse mapping.
#
# The fit traces a Chebyshev grid in (s1, v1), records the (s1, s2_observed,
# OPL) tuples, and least-squares fits Phi as a 4-D Chebyshev tensor
# product in (s1_x, s1_y, s2_x, s2_y).  The Van Vleck cross-Hessian
# determinant ``det d2 Phi / d s1 d s2`` is computed analytically from
# the polynomial coefficients.


@dataclass
class HFPolyFit:
    """4-D Chebyshev tensor-product fit of Phi(s1, s2).

    Counterpart to :class:`CanonicalPolyFit` for direct
    Huygens-Fresnel quadrature.  Fields mirror CanonicalPolyFit but
    the four normalisation axes are (s1x, s1y, s2x, s2y) instead of
    (s2x, s2y, v2x, v2y).
    """
    poly_order: int
    multi_indices: List[Tuple[int, int, int, int]]
    coef_phi: np.ndarray
    s1x_centre: float
    s1x_halfrange: float
    s1y_centre: float
    s1y_halfrange: float
    s2x_centre: float
    s2x_halfrange: float
    s2y_centre: float
    s2y_halfrange: float
    wavelength: float
    res_phi_rms_waves: float = 0.0
    n_rays: int = 0
    linear_coeffs_phi: Optional[np.ndarray] = None
    extract_linear_phase: bool = False

    def to_normalised(
        self,
        s1x: np.ndarray,
        s1y: np.ndarray,
        s2x: np.ndarray,
        s2y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u1 = (s1x - self.s1x_centre) / self.s1x_halfrange
        u2 = (s1y - self.s1y_centre) / self.s1y_halfrange
        u3 = (s2x - self.s2x_centre) / self.s2x_halfrange
        u4 = (s2y - self.s2y_centre) / self.s2y_halfrange
        return u1, u2, u3, u4

    def in_box(
        self,
        s1x: np.ndarray,
        s1y: np.ndarray,
        s2x: np.ndarray,
        s2y: np.ndarray,
    ) -> np.ndarray:
        u1, u2, u3, u4 = self.to_normalised(s1x, s1y, s2x, s2y)
        return ((np.abs(u1) <= 1.0) & (np.abs(u2) <= 1.0)
                & (np.abs(u3) <= 1.0) & (np.abs(u4) <= 1.0))

    def eval_phi(
        self,
        s1x: np.ndarray,
        s1y: np.ndarray,
        s2x: np.ndarray,
        s2y: np.ndarray,
        *,
        include_linear: bool = True,
    ) -> np.ndarray:
        """Evaluate Phi(s1, s2) [waves]."""
        u1, u2, u3, u4 = self.to_normalised(s1x, s1y, s2x, s2y)
        phi = _evaluate_polynomial_4d(self.coef_phi, self.multi_indices,
                                       u1, u2, u3, u4, self.poly_order)
        if include_linear and self.linear_coeffs_phi is not None:
            a0, a1, a2, a3, a4 = self.linear_coeffs_phi
            phi = phi + (a0 + a1 * u1 + a2 * u2 + a3 * u3 + a4 * u4)
        return phi

    def eval_van_vleck_density(
        self,
        s1x: np.ndarray,
        s1y: np.ndarray,
        s2x: np.ndarray,
        s2y: np.ndarray,
    ) -> np.ndarray:
        """Evaluate sqrt(|det d2 Phi / d s1 d s2|) -- the Van Vleck factor.

        The 2x2 cross-Hessian is computed analytically from the
        Chebyshev polynomial via term-wise derivatives.
        """
        u1, u2, u3, u4 = self.to_normalised(s1x, s1y, s2x, s2y)
        inv_h1x = 1.0 / self.s1x_halfrange
        inv_h1y = 1.0 / self.s1y_halfrange
        inv_h2x = 1.0 / self.s2x_halfrange
        inv_h2y = 1.0 / self.s2y_halfrange

        d13 = _eval_4d_cross_deriv(self.coef_phi, self.multi_indices,
                                    u1, u2, u3, u4, self.poly_order, 0, 2)
        d14 = _eval_4d_cross_deriv(self.coef_phi, self.multi_indices,
                                    u1, u2, u3, u4, self.poly_order, 0, 3)
        d23 = _eval_4d_cross_deriv(self.coef_phi, self.multi_indices,
                                    u1, u2, u3, u4, self.poly_order, 1, 2)
        d24 = _eval_4d_cross_deriv(self.coef_phi, self.multi_indices,
                                    u1, u2, u3, u4, self.poly_order, 1, 3)

        H11 = d13 * (inv_h1x * inv_h2x)
        H12 = d14 * (inv_h1x * inv_h2y)
        H21 = d23 * (inv_h1y * inv_h2x)
        H22 = d24 * (inv_h1y * inv_h2y)
        det = H11 * H22 - H12 * H21
        return np.sqrt(np.abs(det))


def _eval_4d_cross_deriv(coef, multi_indices, u1, u2, u3, u4,
                          poly_order, axis_a, axis_b):
    """Evaluate d2 phi / (d u_a d u_b) of a 4-D Chebyshev tensor product
    at scattered points.

    Uses Chebyshev derivative tables (T'_n = n U_{n-1}) on the two
    differentiation axes and ordinary tables on the other two.
    """
    tables = [
        _chebyshev_vandermonde(u1, poly_order),
        _chebyshev_vandermonde(u2, poly_order),
        _chebyshev_vandermonde(u3, poly_order),
        _chebyshev_vandermonde(u4, poly_order),
    ]
    deriv_tables = [
        _chebyshev_derivative_vandermonde(u1, poly_order),
        _chebyshev_derivative_vandermonde(u2, poly_order),
        _chebyshev_derivative_vandermonde(u3, poly_order),
        _chebyshev_derivative_vandermonde(u4, poly_order),
    ]
    use_tables = [
        deriv_tables[i] if i in (axis_a, axis_b) else tables[i]
        for i in range(4)
    ]

    out = np.zeros_like(np.asarray(u1, dtype=np.float64))
    for j, (k1, k2, k3, k4) in enumerate(multi_indices):
        out += (coef[j] * use_tables[0][k1] * use_tables[1][k2]
                * use_tables[2][k3] * use_tables[3][k4])
    return out


def fit_hf_polynomials(
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    source_box_half: float = 50e-6,
    pupil_box_half: float = 0.05,
    n_field: int = 8,
    n_pupil: int = 8,
    poly_order: int = 6,
    extract_linear_phase: bool = True,
    object_distance: Optional[float] = None,
    surface_diffraction: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    endpoint_anchored: bool = False,
) -> HFPolyFit:
    """Fit a 4-D Chebyshev tensor-product polynomial to Phi(s1, s2).

    HF counterpart to :func:`fit_canonical_polynomials`.  Traces a
    (s1, v1) Chebyshev-node grid through the prescription, records
    the (s1, s2_observed, OPL) tuples, and least-squares fits Phi
    as a function of the endpoint pair (s1, s2).

    Parameters
    ----------
    prescription : dict
    wavelength : float
    source_box_half, pupil_box_half : float
    n_field, n_pupil : int
        Per-axis Chebyshev sampling.
    poly_order : int
    extract_linear_phase : bool
    object_distance : float, optional
    surface_diffraction : dict, optional
    endpoint_anchored : bool

    Returns
    -------
    HFPolyFit
    """
    from ..raytrace import surfaces_from_prescription, _make_bundle, trace

    if wavelength <= 0:
        raise ValueError("wavelength must be > 0")
    if poly_order < 0:
        raise ValueError("poly_order must be >= 0")
    if source_box_half <= 0 or pupil_box_half <= 0:
        raise ValueError("source_box_half and pupil_box_half must be > 0")

    if object_distance is None:
        object_distance = float(prescription.get('object_distance', 0.0)) or 0.0

    surfaces = surfaces_from_prescription(prescription)

    def cheb_nodes(n):
        i = np.arange(n)
        x = np.cos(np.pi * (i + 0.5) / n)
        if endpoint_anchored and n >= 2:
            x = x / np.cos(np.pi / (2.0 * n))
        return x

    u_field = cheb_nodes(n_field)
    u_pupil = cheb_nodes(n_pupil)
    s1x_axis = u_field * source_box_half
    s1y_axis = u_field * source_box_half
    v1x_axis = u_pupil * pupil_box_half
    v1y_axis = u_pupil * pupil_box_half

    S1X, S1Y, V1X, V1Y = np.meshgrid(s1x_axis, s1y_axis,
                                      v1x_axis, v1y_axis,
                                      indexing='ij')
    s1x_in = S1X.ravel()
    s1y_in = S1Y.ravel()
    v1x_in = V1X.ravel()
    v1y_in = V1Y.ravel()
    n_rays = s1x_in.size

    sumsq = v1x_in * v1x_in + v1y_in * v1y_in
    if np.any(sumsq >= 1.0):
        raise ValueError(
            "pupil_box_half too large; reduce it.")

    bundle = _make_bundle(
        x=s1x_in.astype(np.float64),
        y=s1y_in.astype(np.float64),
        L=v1x_in.astype(np.float64),
        M=v1y_in.astype(np.float64),
        wavelength=wavelength,
    )
    bundle.z = np.full(n_rays, -object_distance, dtype=np.float64)

    res = trace(bundle, surfaces, wavelength, output_filter='last',
                surface_diffraction=surface_diffraction)
    final = res.image_rays
    alive = np.asarray(final.alive, dtype=bool)
    n_alive = int(alive.sum())
    if n_alive < max(64, 0.5 * n_rays):
        raise RuntimeError(
            f"Too many rays died during HF-fit trace: "
            f"{n_alive} alive of {n_rays}.  Reduce pupil_box_half "
            f"or check apertures.")

    s2x_obs = np.asarray(final.x, dtype=np.float64)
    s2y_obs = np.asarray(final.y, dtype=np.float64)
    phi_obs = np.asarray(final.opd, dtype=np.float64) / wavelength

    s1x_live = s1x_in[alive]
    s1y_live = s1y_in[alive]
    s2x_live = s2x_obs[alive]
    s2y_live = s2y_obs[alive]
    phi_live = phi_obs[alive]

    s1x_c, s1x_h = _fit_normaliser(s1x_live)
    s1y_c, s1y_h = _fit_normaliser(s1y_live)
    s2x_c, s2x_h = _fit_normaliser(s2x_live)
    s2y_c, s2y_h = _fit_normaliser(s2y_live)

    u_s1x = (s1x_live - s1x_c) / s1x_h
    u_s1y = (s1y_live - s1y_c) / s1y_h
    u_s2x = (s2x_live - s2x_c) / s2x_h
    u_s2y = (s2y_live - s2y_c) / s2y_h

    linear_coeffs = None
    if extract_linear_phase:
        X5 = np.column_stack([
            np.ones_like(u_s1x), u_s1x, u_s1y, u_s2x, u_s2y,
        ])
        linear_coeffs, *_ = np.linalg.lstsq(X5, phi_live, rcond=None)
        opd_residual = phi_live - X5 @ linear_coeffs
    else:
        opd_residual = phi_live.copy()

    multi_indices = _multi_indices_total_degree(4, poly_order)
    n_basis = len(multi_indices)
    T1 = _chebyshev_vandermonde(u_s1x, poly_order)
    T2 = _chebyshev_vandermonde(u_s1y, poly_order)
    T3 = _chebyshev_vandermonde(u_s2x, poly_order)
    T4 = _chebyshev_vandermonde(u_s2y, poly_order)
    A = np.empty((u_s1x.size, n_basis), dtype=np.float64)
    for j, (k1, k2, k3, k4) in enumerate(multi_indices):
        A[:, j] = T1[k1] * T2[k2] * T3[k3] * T4[k4]
    coef_phi, *_ = np.linalg.lstsq(A, opd_residual, rcond=None)
    res_phi = float(np.sqrt(np.mean((opd_residual - A @ coef_phi) ** 2)))

    return HFPolyFit(
        poly_order=poly_order,
        multi_indices=multi_indices,
        coef_phi=coef_phi,
        s1x_centre=s1x_c, s1x_halfrange=s1x_h,
        s1y_centre=s1y_c, s1y_halfrange=s1y_h,
        s2x_centre=s2x_c, s2x_halfrange=s2x_h,
        s2y_centre=s2y_c, s2y_halfrange=s2y_h,
        wavelength=wavelength,
        res_phi_rms_waves=res_phi,
        n_rays=n_alive,
        linear_coeffs_phi=linear_coeffs,
        extract_linear_phase=extract_linear_phase,
    )


def propagate_hf_chebyshev_quadrature(
    fit: HFPolyFit,
    E_in: np.ndarray,
    input_grid_x: np.ndarray,
    input_grid_y: np.ndarray,
    output_grid_x: np.ndarray,
    output_grid_y: np.ndarray,
    *,
    apply_van_vleck: bool = True,
    chunk_output: int = 64,
    max_chunk_memory_mb: float = 256.0,
) -> np.ndarray:
    """Direct 2-D HF quadrature using a tensor-product Chebyshev
    polynomial fit of Phi(s1, s2).

    For each output point s2, evaluates

        E_HF(s2) = sum over input grid of
                   E_in(s1) * sqrt(|det d2 Phi / d s1 d s2|)
                   * exp(2 pi i Phi(s1, s2)) * dx * dy

    using the polynomial fit's analytical Phi and Van Vleck factor.

    Parameters
    ----------
    fit : HFPolyFit
    E_in : ndarray (Ny_in, Nx_in) complex
    input_grid_x, input_grid_y : 1-D arrays
    output_grid_x, output_grid_y : 1-D arrays
    apply_van_vleck : bool
    chunk_output : int
        Output points processed per chunk to bound memory.  4.12.0:
        chunk dimension is now a true broadcast axis (was a Python
        loop pre-4.12.0); larger ``chunk_output`` directly improves
        throughput as long as the broadcast-tensor memory fits.
    max_chunk_memory_mb : float
        Soft cap on the per-chunk broadcast-tensor footprint
        (``chunk * Ny_in * Nx_in`` complex128 = ``16 * chunk *
        Ny_in * Nx_in`` bytes).  If ``chunk_output`` would exceed
        this, the chunk is shrunk to fit.  Default 256 MB.

    Returns
    -------
    ndarray (Ny_out, Nx_out) complex
    """
    Nx_in = input_grid_x.shape[0]
    Ny_in = input_grid_y.shape[0]
    Nx_out = output_grid_x.shape[0]
    Ny_out = output_grid_y.shape[0]
    if E_in.shape != (Ny_in, Nx_in):
        raise ValueError(
            f"E_in shape {E_in.shape} mismatches input grid "
            f"({Ny_in}, {Nx_in}).")

    dx = float(np.mean(np.diff(input_grid_x)))
    dy = float(np.mean(np.diff(input_grid_y)))
    pixel_area = dx * dy

    S1X, S1Y = np.meshgrid(input_grid_x, input_grid_y, indexing='xy')
    # 4.11.2: force a complex output dtype so a real-valued E_in (e.g. a
    # pure intensity mask) doesn't silently strip the imaginary part of
    # the HF kernel during ``kernel.astype(E_in.dtype)`` and the
    # subsequent multiply.  Pre-4.11.2 a real E_in produced a real-
    # valued "field" with the imaginary half of the HF integrand
    # summed away -- relative-intensity contrast was preserved but
    # interference structure was destroyed.
    if np.iscomplexobj(E_in):
        out_dtype = E_in.dtype
    elif E_in.dtype == np.float64:
        out_dtype = np.complex128
    else:
        out_dtype = np.complex64
    out = np.zeros((Ny_out, Nx_out), dtype=out_dtype)

    # ------------------------------------------------------------------
    # 4.12.0 (perf #6): vectorise across the output chunk.
    # ------------------------------------------------------------------
    # Pre-4.12.0 the outer ``for start in range(0, n_out, chunk_output)``
    # paired with an inner scalar ``for k in range(start, end)``
    # evaluated one output pixel at a time -- the chunking did nothing
    # because the inner loop was a Python loop, not a vector op.  Every
    # iteration rebuilt the full ``(Ny_in, Nx_in)`` Chebyshev Vandermonde
    # tables on the input grid and called ``np.exp(2j*pi*phi)`` on the
    # full grid.
    #
    # 4.12.0 hoists the input-grid Chebyshev tables outside the chunk
    # loop and vectorises the chunk axis as a true broadcast dim
    # ``(chunk, Ny_in, Nx_in)``.  Memory footprint at chunk=64 on
    # (256, 256) input is ~64 MB complex128 (fits comfortably under
    # the 256 MB default cap); at chunk=1024 it would be ~1 GB and is
    # capped by ``max_chunk_memory_mb``.
    n_out = Ny_out * Nx_out
    poly_order = fit.poly_order
    multi_indices = fit.multi_indices
    K = np.asarray(multi_indices, dtype=np.int64)
    K1, K2, K3, K4 = K[:, 0], K[:, 1], K[:, 2], K[:, 3]
    coef_phi = np.asarray(fit.coef_phi, dtype=np.float64)

    # Normalised input-grid coords -- shape (Ny_in, Nx_in)
    u1 = (S1X - fit.s1x_centre) / fit.s1x_halfrange
    u2 = (S1Y - fit.s1y_centre) / fit.s1y_halfrange
    # Chebyshev Vandermonde tables on input grid (and derivative tables
    # for the Van Vleck cross-Hessian, computed once and reused).
    T1 = _chebyshev_vandermonde(u1, poly_order)   # (k+1, Ny_in, Nx_in)
    T2 = _chebyshev_vandermonde(u2, poly_order)
    if apply_van_vleck:
        dT1 = _chebyshev_derivative_vandermonde(u1, poly_order)
        dT2 = _chebyshev_derivative_vandermonde(u2, poly_order)

    # Flatten output coordinates so the chunk axis is 1-D.
    out_iy, out_ix = np.divmod(np.arange(n_out), Nx_out)
    s2x_flat = output_grid_x[out_ix].astype(np.float64)
    s2y_flat = output_grid_y[out_iy].astype(np.float64)

    # Memory guard:  cap chunk so the (chunk, Ny_in, Nx_in) broadcast
    # tensor (complex128, 16 B/element) stays under the soft budget.
    bytes_per_pixel_per_output = 16 * Ny_in * Nx_in
    if bytes_per_pixel_per_output > 0:
        max_chunk_by_mem = max(
            1,
            int((max_chunk_memory_mb * 1024 * 1024)
                // bytes_per_pixel_per_output)
        )
    else:
        max_chunk_by_mem = chunk_output
    effective_chunk = max(1, min(int(chunk_output), max_chunk_by_mem))

    # Linear-phase coefficients (or zeros if none)
    if fit.linear_coeffs_phi is not None:
        a0, a1, a2, a3, a4 = (float(c) for c in fit.linear_coeffs_phi)
    else:
        a0 = a1 = a2 = a3 = a4 = 0.0
    has_linear = fit.linear_coeffs_phi is not None

    inv_h1x = 1.0 / fit.s1x_halfrange
    inv_h1y = 1.0 / fit.s1y_halfrange
    inv_h2x = 1.0 / fit.s2x_halfrange
    inv_h2y = 1.0 / fit.s2y_halfrange

    flat_out = np.zeros(n_out, dtype=out_dtype)

    for start in range(0, n_out, effective_chunk):
        end = min(start + effective_chunk, n_out)
        s2x_c = s2x_flat[start:end]    # (c,)
        s2y_c = s2y_flat[start:end]    # (c,)
        u3 = (s2x_c - fit.s2x_centre) / fit.s2x_halfrange
        u4 = (s2y_c - fit.s2y_centre) / fit.s2y_halfrange
        # Chebyshev Vandermonde tables on the chunk of output points
        # -- shape (k+1, c).  These need broadcasting against the input
        # tables along a new (Ny_in, Nx_in) axis pair when forming the
        # final basis.
        T3 = _chebyshev_vandermonde(u3, poly_order)   # (k+1, c)
        T4 = _chebyshev_vandermonde(u4, poly_order)

        # Per-basis-term coefficients on the input grid:
        #   tab_12[m] = T1[K1[m]] * T2[K2[m]]       (M, Ny_in, Nx_in)
        # and on the chunk:
        #   tab_34[m] = T3[K3[m]] * T4[K4[m]]       (M, c)
        # The full 4-D basis at every (chunk, Ny_in, Nx_in) point is
        # tab_12[m, y, x] * tab_34[m, c]; the phi sum reduces over m:
        #   phi[c, y, x] = sum_m coef[m] tab_12[m, y, x] tab_34[m, c]
        # which is a tensordot over the basis axis.
        tab_12 = T1[K1] * T2[K2]                       # (M, Ny_in, Nx_in)
        tab_34 = T3[K3] * T4[K4]                       # (M, c)

        # phi_poly[c, y, x] = sum_m (coef[m] * tab_34[m, c]) tab_12[m, y, x]
        #                   = (coef * tab_34) [c, m] @ tab_12 [m, y*x]
        # Fast path: weight tab_12 by (coef * tab_34) and reduce over m.
        cw = coef_phi[:, None] * tab_34                # (M, c)
        # tensordot over the basis axis: result shape (c, Ny_in, Nx_in)
        phi_poly = np.tensordot(cw, tab_12, axes=([0], [0]))

        if has_linear:
            # u1, u2 have shape (Ny_in, Nx_in); u3, u4 have shape (c,).
            # The linear term is a0 + a1 u1 + a2 u2 + a3 u3 + a4 u4
            # which broadcasts to (c, Ny_in, Nx_in).
            lin_xy = a0 + a1 * u1 + a2 * u2                # (Ny_in, Nx_in)
            lin_c = a3 * u3 + a4 * u4                       # (c,)
            phi = phi_poly + lin_xy[None, :, :] + lin_c[:, None, None]
        else:
            phi = phi_poly

        kernel = np.exp(2j * np.pi * phi).astype(out_dtype)   # (c, Ny_in, Nx_in)

        if apply_van_vleck:
            # Cross-Hessian d2 phi / (du_a du_b) for a, b in {(1,3),(1,4),(2,3),(2,4)}.
            # Use the same tensor-product pattern as phi but swap in derivative tables.
            tab_dT1 = dT1[K1]   # (M, Ny_in, Nx_in)
            tab_dT2 = dT2[K2]
            tab_T1 = T1[K1]
            tab_T2 = T2[K2]
            # Derivative tables on s2 coords -- chunked (M, c):
            dT3_arr = _chebyshev_derivative_vandermonde(u3, poly_order)
            dT4_arr = _chebyshev_derivative_vandermonde(u4, poly_order)
            tab_dT3 = dT3_arr[K3]
            tab_dT4 = dT4_arr[K4]
            tab_T3 = T3[K3]
            tab_T4 = T4[K4]

            # d13 = d2 phi / (du1 du3); inputs (M, Ny, Nx) x (M, c)
            xy_d13 = tab_dT1 * tab_T2                          # (M, Ny, Nx)
            c_d13 = coef_phi[:, None] * tab_dT3 * tab_T4       # (M, c)
            d13 = np.tensordot(c_d13, xy_d13, axes=([0], [0])) # (c, Ny, Nx)

            xy_d14 = tab_dT1 * tab_T2
            c_d14 = coef_phi[:, None] * tab_T3 * tab_dT4
            d14 = np.tensordot(c_d14, xy_d14, axes=([0], [0]))

            xy_d23 = tab_T1 * tab_dT2
            c_d23 = coef_phi[:, None] * tab_dT3 * tab_T4
            d23 = np.tensordot(c_d23, xy_d23, axes=([0], [0]))

            xy_d24 = tab_T1 * tab_dT2
            c_d24 = coef_phi[:, None] * tab_T3 * tab_dT4
            d24 = np.tensordot(c_d24, xy_d24, axes=([0], [0]))

            H11 = d13 * (inv_h1x * inv_h2x)
            H12 = d14 * (inv_h1x * inv_h2y)
            H21 = d23 * (inv_h1y * inv_h2x)
            H22 = d24 * (inv_h1y * inv_h2y)
            det = H11 * H22 - H12 * H21
            density = np.sqrt(np.abs(det))                      # (c, Ny, Nx)
            integrand_kernel = density * kernel
        else:
            integrand_kernel = kernel

        # Reduce over the input grid; result shape (c,).
        # einsum is comparable in speed to tensordot + sum here; tensordot
        # is fractionally faster on contiguous arrays.
        chunk_field = np.einsum('cyx,yx->c',
                                 integrand_kernel,
                                 E_in.astype(out_dtype, copy=False),
                                 optimize=False) * pixel_area
        flat_out[start:end] = chunk_field

    out = flat_out.reshape(Ny_out, Nx_out)
    # 4.10: apply the Van Vleck-Morette asymptotic prefactor (2π)^(-d/2)·i^(-d/2)
    # for d = 2: this is -i/(2π).  Pre-4.10 omitted the i^(-d/2) (the
    # Maslov phase) so the result was off by a global 90° phase relative
    # to the Fresnel kernel (which is 1/(iλz) = -i/(λz) in the paraxial
    # limit).  Without this factor, coherent superposition of HF-quadrature
    # output with ASM/Fresnel output is 90° out of phase.
    out = out * (-1j)
    return out

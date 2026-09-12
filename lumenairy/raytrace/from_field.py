"""
lumenairy.raytrace.from_field -- bridge a coherent field into a RayBundle.

This module ships v4.15.1 / Cluster B Item 6.  It introduces a single
public function, :func:`rays_from_field`, that samples a coherent
complex field ``E(x, y)`` into a geometric :class:`RayBundle`.  Each
ray's position is drawn from the field's intensity distribution and
its direction is computed from the local phase gradient
``k_perp = grad(phi)`` via the paraxial mapping ``(L, M) = (k_x, k_y)
/ k0``.  The output bundle can be handed to the rest of
:mod:`lumenairy.raytrace` (e.g. ``trace``, ``spot_diagram``,
``trace_world``) so users can overlay ray traces on coherent-field
plots, seed an HFPI / GBD path bundle from a measured pupil field, or
hand a coherent field into the geometric tracer for hybrid analysis.

Math reference
--------------
* Position: pixel ``(ix, iy)`` is sampled from ``|E|^2`` via one of
  three strategies (``cdf``, ``rejection``, ``uniform``); see
  :func:`_place_cdf`, :func:`_place_rejection`, :func:`_place_uniform`.
  All three placement modes apply ``intensity_threshold`` *per-pixel*
  against the global maximum (``|E|^2 / max(|E|^2) >= threshold``)
  so that sub-threshold noise cannot leak through marginal sums --
  see :func:`_place_cdf` for the historical pixel-vs-marginal
  v4.15.2 fix (P1-NEW-D in the v4.15.1 audit) and the v4.15.3
  ``>=`` consistency fix that aligned ``_place_cdf`` and
  ``_place_uniform`` with the ``_place_rejection`` convention.
* Direction: ``k_perp`` is computed from the *phase ratio* of
  adjacent field samples, symmetrised over the two ONE-PIXEL steps
  ``k_x = arg(E[i,j+1]*conj(E[i,j]) + E[i,j]*conj(E[i,j-1])) / dx``
  (the singularity-safe ``complex_gradient`` mode, default) or from
  the gradient of the unwrapped phase ``k_perp = grad(unwrap(angle
  E))`` (the ``unwrap_gradient`` mode).  The phase-ratio form is the
  *discrete analogue* of the continuous Madelung formula
  ``k_perp = Im(grad E / E)`` (CLUSTER_B_SPEC.md §4.4) but is exact
  for plane waves up to the FULL grid Nyquist ``|L| < lambda/(2 dx)``,
  whereas the literal central-difference ``Im(grad E / E)`` carries a
  ``sinc(k_x dx / pi)`` discretisation bias.  The phase-ratio form
  is also automatically singularity-safe because ``np.angle(.)`` is
  bounded in ``(-pi, pi]`` regardless of how small either factor is.
  v5.x / R6: the estimator was a TWO-pixel difference
  ``arg(E[j+1] conj(E[j-1]))/(2 dx)`` until the 2026-09-11 audit, which
  wrapped silently above HALF the grid Nyquist and gave boundary rays
  exactly half the correct direction cosine.  See the inline docstring
  on :func:`_angle_complex_gradient` for the measurements.
  Direction cosines follow from ``L = k_x / k0``, ``M = k_y / k0``,
  ``N = sqrt(1 - L^2 - M^2)``.  Rays whose ``L^2 + M^2 > 1`` correspond
  to evanescent k-vectors and are marked ``alive = False`` /
  ``error_code = RAY_EVANESCENT``.
* OPD: initialised to ``phi(x_ray, y_ray) / k0`` so subsequent
  geometric-ray OPD accumulation continues correctly from the
  wave-optical state.  ``np.angle`` is WRAPPED, so this seed lives in
  ``(-lambda/2, +lambda/2]``; pass ``opd_phase='unwrapped'`` when the
  consumer treats ``RayBundle.opd`` as a geometric path rather than a
  phase (R7).

Time convention: ``exp(-i omega t)``, matching the rest of the
library.  Wavelength is the vacuum wavelength in metres; the field is
defined on a regularly sampled 2-D grid with pitch ``(dx, dy)``.
Anamorphic grids (``dx != dy``) are supported transparently; default
``dy = dx`` for square grids.

Author: Andrew Traverso -- v4.15.1 / Agent F (Cluster B Item 6).
v4.15.2 -- Agent D: ``_place_cdf`` thresholds pixel-wise (was
marginal-wise); ``rays_from_field`` warns on short-return; top-of-file
docstring updated to phase-ratio formulation; ``n_rays=0`` returns an
empty bundle cleanly; ``'uniform'`` deduplicates collided sub-grid
pixels so ``n_rays > N^2`` is capped at ``N^2``.
The intensity-threshold comparison is consistent across the 3 placement
modes: all three use the inclusive
``|E|^2 / max(|E|^2) >= intensity_threshold`` convention.  A mode using
strict ``>`` drops boundary-exact pixels (see
docs/history/lumenairy.raytrace.from_field.md).
This matters for inputs at the exact threshold
boundary -- see ``docs/release_notes/.release_notes_v4_15_3_agent_d.md``.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np

from .core import RAY_EVANESCENT, RAY_OK, RayBundle

__all__ = ['rays_from_field']


# ============================================================================
# Public API
# ============================================================================

def rays_from_field(
    E: np.ndarray,
    *,
    dx: float,
    wavelength: float,
    dy: Optional[float] = None,
    n_rays: int = 200,
    placement: str = 'cdf',
    angle_method: str = 'complex_gradient',
    intensity_threshold: float = 1e-4,
    z0: float = 0.0,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    opd_phase: str = 'wrapped',
) -> RayBundle:
    """Sample a coherent field into a geometric :class:`RayBundle`.

    Each ray's position is drawn from the field's intensity
    distribution; each ray's direction comes from the local phase
    gradient ``k_perp = grad(phi)`` via the paraxial mapping
    ``(L, M) = (k_x, k_y) / k0``, ``N = sqrt(1 - L^2 - M^2)``.  Rays
    whose tangential ``k`` exceeds ``k0`` (evanescent components) are
    flagged with ``RAY_EVANESCENT`` and ``alive = False``.

    The OPD is initialised to ``phi(x_ray, y_ray) / k0`` so subsequent
    geometric-ray OPD accumulation continues from the wave-optical
    state.

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Coherent field amplitude.  Must be 2-D.  NumPy convention:
        first axis (rows) is ``y``, second axis (columns) is ``x``.
    dx : float
        Grid pitch in x [m].  Must be positive.
    wavelength : float
        Vacuum wavelength [m].  Must be positive.
    dy : float, optional
        Grid pitch in y [m].  Defaults to ``dx`` (square grid).
    n_rays : int, default 200
        Target number of rays.  Must be non-negative.  ``n_rays == 0``
        returns an empty bundle cleanly.  The returned bundle has
        *at most* ``n_rays`` rays (some may be flagged as evanescent
        and ``alive = False``).  If the placement step returns fewer
        than ``n_rays`` (e.g. ``'rejection'`` exhausting its
        50 * ``n_rays`` candidate budget, or ``'uniform'`` returning
        fewer survivors than the sub-grid count due to thresholding
        / pixel collision), a ``RuntimeWarning`` is emitted
        indicating ``n_actual`` vs the requested ``n_rays``; the
        partial bundle is still returned for downstream use.
    placement : {'cdf', 'rejection', 'uniform'}, default 'cdf'
        Strategy for placing ray origins.

        * ``'cdf'`` -- separable inverse-CDF sampling along the x and
          y marginals.  Fast.  Exact for separable intensity
          distributions (e.g. Gaussian beams aligned to the axes);
          approximate for strongly non-separable intensities (e.g.
          azimuthal lattices).  ``intensity_threshold`` is applied
          *pixel-wise* (``|E|^2 / max(|E|^2) >= threshold``) BEFORE
          the marginal sums are formed, so sub-threshold noise
          cannot accumulate across the marginal axis and contaminate
          cannot accumulate across the marginal axis and contaminate
          the CDF.  ``>=`` is the canonical convention.)  Use for
          visualisation.
        * ``'rejection'`` -- true 2-D rejection sampling from
          ``|E|^2``.  Slower but exact regardless of separability.
          ``intensity_threshold`` is applied pixel-wise.  Use for
          vortex beams, complicated holograms, etc.
        * ``'uniform'`` -- uniform placement on a regular sub-grid,
          dropping pixels whose intensity falls below
          ``intensity_threshold``.  ``intensity_threshold`` is
          applied pixel-wise.  Returns up to ``n_rays`` survivors
          (could be fewer if the supporting region is too small or
          the sub-grid count exceeds the number of unique grid
          pixels; see also the ``RuntimeWarning`` on short return).
          Use when you want a deterministic ray fan.

    angle_method : {'complex_gradient', 'unwrap_gradient'}, default 'complex_gradient'
        How to compute the local k-vector.

        * ``'complex_gradient'`` -- ``k_perp = Im(grad E / E)``.
          Avoids phase unwrapping entirely.  Singularity-safe -- the
          denominator ``E`` is clamped to ``intensity_threshold *
          max(|E|)`` to prevent division blowup near vortex cores.
          Recommended default.
        * ``'unwrap_gradient'`` -- ``k_perp = grad(np.unwrap(angle(E)))``
          unwrapped along each axis independently.  Fragile near
          phase singularities; use only for smooth phase profiles.

    intensity_threshold : float, default 1e-4
        Relative threshold ``|E|^2 / max(|E|^2)``.  Pixels strictly
        below the threshold are excluded from placement for ALL three
        modes (``'cdf'``, ``'rejection'``, ``'uniform'``); the
        threshold is applied pixel-wise rather than against any
        marginal sum.  The comparison is inclusive
        (``|E|^2 / max(|E|^2) >= threshold``) so a pixel whose
        normalised intensity is exactly ``intensity_threshold`` is
        retained.  All three modes use the same comparison.  For
        ``angle_method='complex_gradient'`` the
        threshold also clamps the denominator of ``grad E / E``.
    z0 : float, default 0.0
        Axial position of all ray origins [m].
    random_state : int or np.random.Generator or None, optional
        Seed or generator used for the random placement.  ``None``
        uses :func:`numpy.random.default_rng()` with fresh entropy.
        Pass an int (or a fixed Generator) for reproducibility.
    opd_phase : {'wrapped', 'unwrapped'}, default 'wrapped'
        How the returned ``opd`` is seeded from the field phase (R7).

        * ``'wrapped'`` (default, backward compatible) --
          ``np.angle(E) / k0``, therefore confined to
          ``(-lambda/2, +lambda/2]``.  Correct mod 2 pi, so any consumer
          that exponentiates it (``bundles.ray_to_beamlet``, HFPI) is
          unaffected; a consumer that treats ``opd`` as a GEOMETRIC path
          (an OPD fan, a wavefront fit, ``np.unwrap``, differencing
          across rays) sees a sawtooth.  Measured on a converging
          spherical wave with 0.8 waves of true spread: the returned
          ``opd`` covered the whole [-494.6, +498.3] nm wrap interval at
          lambda = 1 um.
        * ``'unwrapped'`` -- ``np.unwrap`` the phase along x then y over
          the whole grid before sampling, giving a continuous path
          (defined up to one global piston).  Exact for a smooth phase
          sampled above Nyquist; unreliable across a vortex core or a
          disconnected support, hence opt-in.

    Returns
    -------
    RayBundle
        Bundle with at most ``n_rays`` rays (typically exactly
        ``n_rays``, see the ``RuntimeWarning`` clause on the
        ``n_rays`` parameter for the short-return case).  Per-ray
        fields:

        * ``x, y`` -- sampled origin coordinates [m]
        * ``z`` -- ``np.full(n_actual, z0)`` [m]
        * ``L, M, N`` -- direction cosines, with ``L^2 + M^2 + N^2 =
          1`` for living rays; ``N = 0`` for evanescent rays.
        * ``alive`` -- ``True`` for non-evanescent rays, ``False``
          otherwise.
        * ``opd`` -- ``phi(x_ray, y_ray) / k0``  [m].  WRAPPED into
          ``(-lambda/2, +lambda/2]`` unless ``opd_phase='unwrapped'``
          (see that parameter).
        * ``error_code`` -- ``RAY_OK`` (0) or ``RAY_EVANESCENT`` (5).
        * ``wavelength`` -- vacuum wavelength [m].

    Raises
    ------
    ValueError
        If ``E`` is not a 2-D complex / real array, ``dx`` or
        ``wavelength`` is non-positive, ``n_rays < 0``, or
        ``placement`` / ``angle_method`` are unrecognised.

    Warns
    -----
    RuntimeWarning
        If the placement step returns fewer than ``n_rays`` rays
        (e.g. ``'rejection'`` exhausted its candidate budget or
        ``'uniform'`` returned fewer survivors than requested).

    Examples
    --------
    Seed a ray bundle from a (mock) measured pupil field for a
    downstream geometric trace::

        >>> import numpy as np
        >>> import lumenairy as la
        >>> N, dx, lam, f = 256, 2e-6, 633e-9, 0.05
        >>> src = la.Source.gaussian(N=N, dx=dx, wavelength=lam, w0=200e-6)
        >>> E_pupil = la.apply_thin_lens(src.E, f=f, wavelength=lam, dx=dx)
        >>> rays = la.rays_from_field(E_pupil, dx=dx, wavelength=lam,
        ...                            n_rays=80, placement='cdf')
        >>> bool(rays.alive.all())  # all rays propagate
        True

    See Also
    --------
    lumenairy.raytrace.RayBundle : output type.
    lumenairy.raytrace.trace : geometric trace through a Surface list.
    lumenairy.propagators.hfpi.init_paths_from_field : analogous
        bridge for the HFPI Monte-Carlo path representation.
    lumenairy.propagators.gbd.decompose_field_to_beamlets : analogous
        bridge for GBD beamlets.
    """
    # Defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  v4.15.1 added an inline
    # ``E.ndim != 2`` check below; that catches 3-D ensembles with a
    # tailored message, but a ``PartialCoherenceMCF`` input fails at
    # ``np.asarray(E)`` with the unhelpful Python TypeError instead
    # of the canonical v4.16-roadmap message.  Hoist the guard to
    # the absolute entry so MCF inputs are rejected first; the
    # downstream ``E.ndim != 2`` check is preserved for diagnostic
    # symmetry (its message is more specific to the rays-from-field
    # use case).  Input kind: 'field'.
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E, 'rays_from_field', input_kind='field')
    # ------------------------------------------------------------------
    # Input validation -- audit posture: fail loudly at the boundary.
    # ------------------------------------------------------------------
    E = np.asarray(E)
    if E.ndim != 2:
        raise ValueError(
            f'rays_from_field: E must be 2-D (Ny, Nx); got shape {E.shape}.'
        )
    if not np.issubdtype(E.dtype, np.complexfloating):
        # Accept real input by promoting to complex; many users pass
        # plain real-valued amplitude masks.
        E = E.astype(np.complex128)
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError(
            f'rays_from_field: dx must be positive and finite; got {dx!r}.'
        )
    if dy is None:
        dy = dx
    if not np.isfinite(dy) or dy <= 0:
        raise ValueError(
            f'rays_from_field: dy must be positive and finite; got {dy!r}.'
        )
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError(
            f'rays_from_field: wavelength must be positive and finite; '
            f'got {wavelength!r}.'
        )
    if not isinstance(n_rays, (int, np.integer)) or n_rays < 0:
        raise ValueError(
            f'rays_from_field: n_rays must be a non-negative integer; '
            f'got {n_rays!r}.'
        )
    if placement not in ('cdf', 'rejection', 'uniform'):
        raise ValueError(
            f"rays_from_field: placement must be one of "
            f"{{'cdf', 'rejection', 'uniform'}}; got {placement!r}."
        )
    if angle_method not in ('complex_gradient', 'unwrap_gradient'):
        raise ValueError(
            f"rays_from_field: angle_method must be one of "
            f"{{'complex_gradient', 'unwrap_gradient'}}; "
            f"got {angle_method!r}."
        )
    if not (0.0 <= intensity_threshold < 1.0):
        raise ValueError(
            f'rays_from_field: intensity_threshold must be in [0, 1); '
            f'got {intensity_threshold!r}.'
        )
    if opd_phase not in ('wrapped', 'unwrapped'):
        raise ValueError(
            f"rays_from_field: opd_phase must be one of "
            f"{{'wrapped', 'unwrapped'}}; got {opd_phase!r}."
        )

    # Normalise random_state -> Generator.
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    k0 = 2.0 * np.pi / wavelength
    Ny, Nx = E.shape

    # If the field is entirely zero (or below threshold everywhere)
    # we fall back to uniform placement at the centre so we don't
    # produce a NaN-filled bundle.
    max_abs = float(np.abs(E).max())
    if max_abs == 0.0:
        raise ValueError(
            'rays_from_field: input field is identically zero; '
            'cannot sample rays from |E|^2.'
        )

    # ------------------------------------------------------------------
    # Edge case -- n_rays == 0 returns an empty bundle cleanly.
    # ------------------------------------------------------------------
    if n_rays == 0:
        empty_f = np.array([], dtype=np.float64)
        empty_b = np.array([], dtype=bool)
        empty_u8 = np.array([], dtype=np.uint8)
        return RayBundle(
            x=empty_f, y=empty_f, z=empty_f,
            L=empty_f, M=empty_f, N=empty_f,
            wavelength=float(wavelength),
            alive=empty_b, opd=empty_f,
            error_code=empty_u8,
        )

    # ------------------------------------------------------------------
    # 1. Place rays in (x, y) according to |E|^2.
    # ------------------------------------------------------------------
    if placement == 'cdf':
        x_pos, y_pos, ix_arr, iy_arr = _place_cdf(
            E, dx, dy, n_rays, intensity_threshold, rng,
        )
    elif placement == 'rejection':
        x_pos, y_pos, ix_arr, iy_arr = _place_rejection(
            E, dx, dy, n_rays, intensity_threshold, rng,
        )
    else:  # placement == 'uniform'
        x_pos, y_pos, ix_arr, iy_arr = _place_uniform(
            E, dx, dy, n_rays, intensity_threshold,
        )

    n_actual = ix_arr.size
    if n_actual == 0:
        raise ValueError(
            'rays_from_field: no pixels survived intensity thresholding -- '
            'the field may be entirely below the threshold or the supporting '
            'region is too small for the requested n_rays.'
        )
    if n_actual < n_rays:
        warnings.warn(
            f"rays_from_field: placement '{placement}' returned "
            f"{n_actual} rays vs requested {n_rays}. This may indicate "
            f"intensity_threshold is too aggressive, or the placement "
            f"budget (rejection-mode tries = 50*n_rays; uniform-mode "
            f"sub-grid pixels deduplicated to <= N^2) was exhausted "
            f"before n_rays was reached.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # 2. Compute (L, M, N) at each ray-origin pixel.
    # ------------------------------------------------------------------
    if angle_method == 'complex_gradient':
        L, M, N_dir, evanescent = _angle_complex_gradient(
            E, dx, dy, ix_arr, iy_arr, k0, intensity_threshold,
        )
    else:  # angle_method == 'unwrap_gradient'
        L, M, N_dir, evanescent = _angle_unwrap_gradient(
            E, dx, dy, ix_arr, iy_arr, k0,
        )

    # ------------------------------------------------------------------
    # 3. Initialise OPD from the wave-optical phase at each ray origin.
    # ------------------------------------------------------------------
    # R7 (AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11): ``np.angle`` is
    # WRAPPED into (-pi, pi], so the default ``opd`` is a sawtooth, not a
    # geometric path -- measured on a converging spherical wave with
    # 0.8 waves of true OPL spread, the returned ``opd`` covered the full
    # wrap interval [-494.6, +498.3] nm at lambda = 1 um.  Consumers that
    # exponentiate (``bundles.ray_to_beamlet``, HFPI) are unaffected
    # mod 2 pi; anything that treats ``opd`` as a path (an OPD fan, a
    # wavefront fit, ``np.unwrap``, differencing across rays) is not.
    # ``opd_phase='unwrapped'`` removes the sawtooth over the sampled
    # support.
    phi = np.angle(E)
    if opd_phase == 'unwrapped':
        # Unwrap along each axis independently (the same two-pass scheme
        # ``angle_method='unwrap_gradient'`` uses), then read the ray
        # pixels.  Exact for a smooth phase sampled above Nyquist;
        # unreliable across a vortex core or a disconnected support,
        # which is why it is opt-in.
        phi = np.unwrap(np.unwrap(phi, axis=1), axis=0)
    opd_init = phi[iy_arr, ix_arr] / k0

    # ------------------------------------------------------------------
    # 4. Assemble RayBundle.
    # ------------------------------------------------------------------
    z = np.full(n_actual, float(z0), dtype=np.float64)
    alive = ~evanescent
    error_code = np.where(
        evanescent,
        np.uint8(RAY_EVANESCENT),
        np.uint8(RAY_OK),
    ).astype(np.uint8)

    return RayBundle(
        x=x_pos.astype(np.float64),
        y=y_pos.astype(np.float64),
        z=z,
        L=L.astype(np.float64),
        M=M.astype(np.float64),
        N=N_dir.astype(np.float64),
        wavelength=float(wavelength),
        alive=alive.astype(bool),
        opd=opd_init.astype(np.float64),
        error_code=error_code,
    )


# ============================================================================
# Placement helpers
# ============================================================================

def _place_cdf(
    E: np.ndarray,
    dx: float,
    dy: float,
    n_rays: int,
    threshold: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separable inverse-CDF placement.

    Builds 1-D cumulative distributions from the row- and column-sum
    intensity marginals (computed on a PIXEL-WISE-THRESHOLDED copy of
    ``|E|^2``), then inverts each via :func:`numpy.searchsorted`.
    Independent draws in x and y give a separable joint distribution
    -- exact for separable ``|E|^2`` (e.g. axially aligned Gaussians)
    and a fast approximation otherwise.

    The threshold is applied pixel-wise BEFORE the marginal sums.
    Applying it to the marginal sums themselves
    (``Ix.sum(axis=0) > threshold * Ix.max()``) lets sub-
    contaminate the CDF.  Pixel-wise thresholding makes ``'cdf'``
    consistent with ``'rejection'`` and ``'uniform'``.

    The threshold comparison is inclusive (``>=``) so pixels at exactly
    ``intensity_threshold`` are retained, matching the canonical "pixel
    intensity meets threshold" convention used in ``_place_rejection``
    and ``_place_uniform``.
    """
    I = np.abs(E) ** 2
    I_norm = I / I.max()
    # Pixel-wise threshold: zero out any pixel whose normalised
    # intensity is strictly below the threshold, BEFORE forming the
    # marginal sums.  This matches the spec docstring's claim of a
    # pixel-wise threshold and lines up with the other two modes.
    # v4.15.3 -- ``>=`` is the canonical "meets threshold" convention.
    I_thresh = np.where(I_norm >= threshold, I, 0.0)

    Ny, Nx = E.shape

    # x-marginal (sum over y) of the pixel-thresholded intensity.
    Ix = I_thresh.sum(axis=0)
    if Ix.sum() == 0.0:
        # All intensity below threshold -- nothing to sample.
        return (np.array([]), np.array([]),
                np.array([], dtype=np.intp),
                np.array([], dtype=np.intp))
    cdf_x = np.cumsum(Ix)
    cdf_x = cdf_x / cdf_x[-1]

    # y-marginal (sum over x) of the pixel-thresholded intensity.
    Iy = I_thresh.sum(axis=1)
    if Iy.sum() == 0.0:
        return (np.array([]), np.array([]),
                np.array([], dtype=np.intp),
                np.array([], dtype=np.intp))
    cdf_y = np.cumsum(Iy)
    cdf_y = cdf_y / cdf_y[-1]

    u = rng.random(n_rays)
    v = rng.random(n_rays)
    ix = np.searchsorted(cdf_x, u, side='right')
    iy = np.searchsorted(cdf_y, v, side='right')
    # Clamp to valid index range (searchsorted can return Nx if u == 1).
    ix = np.clip(ix, 0, Nx - 1).astype(np.intp)
    iy = np.clip(iy, 0, Ny - 1).astype(np.intp)

    # S11-6f NOT-CHANGED (AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1): the ray
    # origin is anchored on ``N // 2`` while the propagators' centred field
    # grid is ``(j - N/2)*dx``; the two agree for even N and differ by dx/2 for
    # odd N.  Every placement helper in this module shares the anchor (so they
    # are mutually consistent), and changing it moves every ray on every odd-N
    # field -- not bit-identical, left for a deliberate grid-convention pass.
    x = (ix - Nx // 2) * dx
    y = (iy - Ny // 2) * dy
    return x.astype(np.float64), y.astype(np.float64), ix, iy


def _place_rejection(
    E: np.ndarray,
    dx: float,
    dy: float,
    n_rays: int,
    threshold: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """True 2-D rejection sampling from ``|E|^2``.

    Slower than ``_place_cdf`` (~50 candidate draws per accepted ray
    in the worst case) but exact for any intensity distribution.
    Caps iterations at ``50 * n_rays`` to guarantee termination on
    pathological inputs; if the cap is reached the function returns
    the rays accepted so far (caller will resize the bundle).

    The threshold comparison is inclusive (``>=``) here as in all three
    placement modes; the canonical convention is "pixel intensity meets
    threshold" so a pixel whose normalised intensity is exactly
    ``intensity_threshold`` is retained.  The behaviour difference for
    inputs at the exact threshold
    """
    I = np.abs(E) ** 2
    I = I / I.max()
    Ny, Nx = E.shape

    x_list, y_list = [], []
    ix_list, iy_list = [], []
    max_tries = max(n_rays * 50, 100)
    tries = 0
    while len(x_list) < n_rays and tries < max_tries:
        ix_c = int(rng.integers(0, Nx))
        iy_c = int(rng.integers(0, Ny))
        v = I[iy_c, ix_c]
        if v >= threshold and rng.random() < v:
            x_list.append((ix_c - Nx // 2) * dx)
            y_list.append((iy_c - Ny // 2) * dy)
            ix_list.append(ix_c)
            iy_list.append(iy_c)
        tries += 1

    return (
        np.array(x_list, dtype=np.float64),
        np.array(y_list, dtype=np.float64),
        np.array(ix_list, dtype=np.intp),
        np.array(iy_list, dtype=np.intp),
    )


def _place_uniform(
    E: np.ndarray,
    dx: float,
    dy: float,
    n_rays: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniform sub-grid placement, dropping pixels below threshold.

    Builds a roughly ``sqrt(n_rays)`` x ``sqrt(n_rays)`` sub-grid over
    the field extent, takes the nearest pixel to each sub-grid point,
    and drops the ones whose ``|E|^2`` falls below ``threshold *
    max(|E|^2)``.  Deterministic -- no rng dependence.  Returns up to
    ``n_rays`` survivors (or ``N_pixels`` survivors, whichever is
    smaller -- duplicate sub-grid landings on the same pixel are
    deduplicated so this helper never returns more rays than there
    are unique grid pixels passing the threshold).

    When ``n_rays`` exceeds the
    number of unique pixels in the grid, the sub-grid pixelisation can
    produce duplicate ``(iy, ix)`` entries, so they are deduped in a
    stable manner.

    The threshold comparison is inclusive (``>=``) so pixels at exactly
    ``intensity_threshold`` are retained, matching the canonical "pixel
    intensity meets threshold" convention used in ``_place_rejection``
    and ``_place_cdf``.
    """
    I = np.abs(E) ** 2
    Imax = I.max()
    if Imax == 0:
        return (np.array([]), np.array([]),
                np.array([], dtype=np.intp),
                np.array([], dtype=np.intp))
    # Pixel-wise threshold mask (matches the docstring "Pixels below
    # the threshold are excluded from placement").  v4.15.3 -- ``>=``
    # is the canonical "meets threshold" convention.
    mask = (I / Imax) >= threshold

    Ny, Nx = E.shape

    # Use the larger of the two axes to anchor the grid count, scaled
    # so that the total cell count is close to n_rays.
    #
    # S9-RT3 (audit review4a, pattern #2 -- a pixel-count parameter whose
    # PHYSICAL meaning flips with the grid).  ``aspect`` must be the SIGNED
    # ratio ``Nx/Ny``, because it is divided into the ``y`` count only.  The
    # orientation-blind ``max/min`` form gave the right (near-isotropic) ray
    # pitch for ``Nx >= Ny`` and the INVERTED one for ``Ny > Nx``, putting
    # FEWER samples on the LONGER axis: measured at ``n_rays=64``,
    # ``(Nx, Ny) = (64, 256)`` gave nx_grid=16 / ny_grid=4 -> 4.20 px x-pitch
    # vs 85.00 px y-pitch (20.2x anisotropic; isotropic would be 16 px), and
    # (32, 512) gave 1.00 px vs 511.00 px (511x, i.e. TWO ray rows over 512
    # rows), while the transposed (256, 64) / (512, 32) grids came out at
    # 1.24x / 1.88x.  Same ``n_rays``, same physical field, ray density per
    # metre changing 20-500x purely from the grid's orientation.
    # ``Nx / Ny`` is the numerically identical expression to
    # ``max(Nx, Ny) / min(Nx, Ny)`` whenever ``Nx >= Ny``, so every
    # already-correct (wide or square) grid keeps a bit-identical placement.
    aspect = Nx / Ny
    ny_grid = int(np.ceil(np.sqrt(n_rays / aspect)))
    nx_grid = int(np.ceil(n_rays / max(ny_grid, 1)))
    ny_grid = max(ny_grid, 1)
    nx_grid = max(nx_grid, 1)

    # Pick the central pixel of each sub-cell -- monotonic spacing
    # across the masked region.
    #
    # S11-3 (AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1, the lattice /
    # cell-centring pattern).  ``np.linspace(0, N-1, n)`` is EDGE-
    # ANCHORED: it puts the first and last sub-grid point exactly ON the
    # array edges (pixels 0 and N-1), which contradicts this function's
    # own "central pixel of each sub-cell" contract and -- on any centred
    # beam -- lands those points where the intensity is far below
    # ``intensity_threshold``, so they are thresholded away.  Measured on
    # a centred w0 = 80 um Gaussian on a 64x64 / 10 um grid at
    # ``intensity_threshold = 1e-4``: n_rays = 1, 2 and 4 produced ZERO
    # survivors (``rays_from_field`` then RAISED "no pixels survived
    # intensity thresholding"), 9 -> 1 ray, 16 -> 4, 25 -> 5.
    #
    # The cell-centred lattice ``(arange(n) + 0.5) * N / n - 0.5`` places
    # sub-grid point k at the centre of the k-th equal-area sub-cell of
    # the pixel span [-0.5, N-0.5] -- the same convention
    # ``analysis/detector.py`` uses for field-sample centres and
    # ``elements/rcwa`` uses for period fractions.  For n = 1 it lands on
    # the centre pixel (N=64 -> 31.5 -> pixel 32) instead of the corner.
    # NOT bit-compatible at any n_rays: this is a deliberate placement
    # change, pinned in tests/unit/test_niche_s11_sibling_deferred.py.
    iy_pixels = np.clip(
        np.round((np.arange(ny_grid) + 0.5) * Ny / ny_grid - 0.5),
        0, Ny - 1).astype(np.intp)
    ix_pixels = np.clip(
        np.round((np.arange(nx_grid) + 0.5) * Nx / nx_grid - 0.5),
        0, Nx - 1).astype(np.intp)
    iy_grid, ix_grid = np.meshgrid(iy_pixels, ix_pixels, indexing='ij')
    iy_flat = iy_grid.ravel()
    ix_flat = ix_grid.ravel()

    # Dedupe sub-grid landings stably so the helper never returns
    # more rays than there are unique grid pixels.  np.unique with
    # return_index guarantees stable ordering by first-occurrence.
    if iy_flat.size > 0:
        composite = iy_flat.astype(np.int64) * np.int64(Nx) + \
            ix_flat.astype(np.int64)
        _, first_idx = np.unique(composite, return_index=True)
        first_idx.sort()  # preserve original sub-grid traversal order
        iy_flat = iy_flat[first_idx]
        ix_flat = ix_flat[first_idx]

    keep = mask[iy_flat, ix_flat]
    iy_arr = iy_flat[keep][:n_rays].astype(np.intp)
    ix_arr = ix_flat[keep][:n_rays].astype(np.intp)

    x = (ix_arr - Nx // 2) * dx
    y = (iy_arr - Ny // 2) * dy
    return x.astype(np.float64), y.astype(np.float64), ix_arr, iy_arr


# ============================================================================
# Angle helpers
# ============================================================================

def _angle_complex_gradient(
    E: np.ndarray,
    dx: float,
    dy: float,
    ix: np.ndarray,
    iy: np.ndarray,
    k0: float,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Singularity-safe complex-gradient angle estimator.

    Computes the local k-vector from the *phase ratio* of adjacent
    field samples, symmetrised over the two ONE-PIXEL steps that
    straddle the sample (R6, AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11):

    .. math::

        k_x(i,j) = \\frac{1}{\\Delta x}\\,
                  \\arg\\!\\bigl( E_{i,j+1}E_{i,j}^{*}
                                 + E_{i,j}E_{i,j-1}^{*}\\bigr)

    (and analogously for ``k_y``).  This is the discrete analogue of
    the Madelung formula :math:`k_\\perp = \\operatorname{Im}(\\nabla E
    / E)`, is *exact* for plane waves, is centred, and is unambiguous
    over the FULL grid Nyquist :math:`|L| < \\lambda / (2 \\Delta x)`.
    It is also automatically singularity-safe -- ``np.angle`` on a
    complex sum is bounded in :math:`(-\\pi, \\pi]` regardless of how
    small either factor is.

    Pre-R6 the estimator used the TWO-pixel step
    :math:`\\arg(E_{j+1}E_{j-1}^{*}) / (2\\Delta x)`, valid only to
    :math:`|L| < \\lambda / (4\\Delta x)` -- half the grid's own Nyquist
    -- and wrapping silently above it (measured at
    :math:`\\lambda = 1\\,\\mu m`, :math:`\\Delta x = 2\\,\\mu m`:
    ``L_true`` 0.150 recovered as -0.100, 0.200 as -0.050, 0.300 as
    +0.050, 0.490 as -0.010).  That also falsifies the claim that the
    form "can detect evanescent rays whose tangential k exceeds
    :math:`\\pi/\\Delta x`": an evanescent ``L = 0.49`` comes back as a
    benign ``L = -0.01``, so no evanescent ray is ever flagged.

    Boundary samples have one real neighbour only; the missing side is
    dropped (one-sided one-pixel difference), which is still exact for a
    plane wave.  Pre-R6 the clipped self-reference was kept while the
    divisor stayed at :math:`2\\Delta x`, so edge rays got exactly HALF
    the correct direction cosine (measured 0.5000 over 128 edge rays).

    The denominator clamp from ``intensity_threshold`` is still
    applied for the rare case where the neighbouring pixels are
    identically zero (e.g. far outside a hard aperture): there the
    product sum is zero and ``np.angle`` returns 0, which we accept as
    "no useful gradient information here".

    Deviates from CLUSTER_B_SPEC.md §4.4's literal
    ``Im(grad E / E)`` formula in favour of this phase-ratio form
    because the literal formula is biased by
    :math:`\\operatorname{sinc}(k_x \\Delta x / \\pi)` under central
    differences and fails the 5-deg tilted-plane-wave test at the
    spec's tolerance for the spec's sampling.  The phase-ratio form is
    the standard "computational k-vector" used in quantum-mechanics
    literature (Madelung continuity equation discretisation) and
    matches the spec's stated intent: recover the local k-vector with no
    explicit unwrap.
    """
    Ny, Nx = E.shape

    # Neighbour indices, clipped at the array boundary.  ``have_*``
    # records whether the neighbour is a REAL neighbour rather than the
    # clipped self-reference -- see the symmetrised-difference block
        # below (R6: a clipped self-reference halves the baseline while
        # the divisor stays at 2 dx).
    ix = np.asarray(ix, dtype=np.intp)
    iy = np.asarray(iy, dtype=np.intp)
    ix_plus = np.clip(ix + 1, 0, Nx - 1)
    ix_minus = np.clip(ix - 1, 0, Nx - 1)
    iy_plus = np.clip(iy + 1, 0, Ny - 1)
    iy_minus = np.clip(iy - 1, 0, Ny - 1)
    have_xp = ix_plus != ix
    have_xm = ix_minus != ix
    have_yp = iy_plus != iy
    have_ym = iy_minus != iy

    max_abs = np.abs(E).max()
    clamp_floor = threshold * max_abs

    # Sample neighbour values and clamp tiny |E| to avoid the rare
    # "both neighbours zero" case which would produce a meaningless
    # 0 + 0j product; lifting to clamp_floor preserves any non-zero
    # phase information present and otherwise gives a 0 gradient.
    def _safe_sample(arr, iy_idx, ix_idx):
        v = arr[iy_idx, ix_idx]
        abs_v = np.abs(v)
        if max_abs == 0:
            return v
        scale = np.where(abs_v >= clamp_floor,
                          1.0,
                          clamp_floor / np.maximum(abs_v, 1e-300))
        return v * scale

    E_c = _safe_sample(E, iy, ix)
    Ex_plus = _safe_sample(E, iy, ix_plus)
    Ex_minus = _safe_sample(E, iy, ix_minus)
    Ey_plus = _safe_sample(E, iy_plus, ix)
    Ey_minus = _safe_sample(E, iy_minus, ix)

    # R6 (AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11): SYMMETRISED ONE-PIXEL
    # difference,
    #
    #     kx = arg( E[j+1] conj(E[j]) + E[j] conj(E[j-1]) ) / dx
    #
    # replacing the two-pixel form ``arg(E[j+1] conj(E[j-1])) / (2 dx)``.
    # Both are centred and both are exact for a plane wave, but the
    # two-pixel form is unambiguous only for ``|kx * 2 dx| < pi``, i.e.
    # ``|L| < lambda / (4 dx)`` -- HALF of what the grid itself supports
    # (``lambda / (2 dx)``) -- and silently WRAPS above it.  Measured at
    # lambda = 1 um, dx = 2 um (grid Nyquist |L| <= 0.25): L_true 0.150
    # came back -0.100, 0.200 -> -0.050, 0.300 -> +0.050, 0.490 -> -0.010.
    # The one-pixel products each carry the phase step ``kx dx``, so the
    # sum's argument is unambiguous to the FULL grid Nyquist, and an
    # evanescent tangential k now genuinely reads out of range instead of
    # rolling back through zero into a benign-looking value.  Summing the
    # two products (rather than averaging their angles) keeps the
    # amplitude weighting and cannot wrap.
    #
    # Boundary columns/rows have only ONE real neighbour: the clipped
    # index collapses onto the pixel itself, whose product ``E conj(E)``
    # is a real positive number carrying no phase.  Including it halves
    # the recovered direction cosine exactly -- measured on a
    # uniform-amplitude tilted plane wave with L_true = 0.05 on a 64x64
    # grid: interior pixels +0.050000, edge columns +0.025000, ratio
    # 0.5000 over 128 edge rays.  Dropping the missing side leaves a
    # one-sided one-pixel difference, which is still exact for a plane
    # wave (only noisier), so edge rays now read the right angle.
    #
    # The two one-pixel phasors are NORMALISED before they are summed,
    # i.e. the estimator is the circular MEAN of the two half-step phase
    # increments.  Summing the raw products would weight each half-step
    # by its amplitude, which biases the result wherever |E| varies
    # across the pixel: measured on the converging-spherical-wave probe
    # (repro/RAYTRACE/p7_fromfield.py test 4) the amplitude-weighted sum
    # focused to 16.3 nm rms while the circular mean focuses to 0.002 nm
    # -- the central two-pixel difference is exact for a quadratic phase
    # and the circular mean inherits that, because for a quadratic the
    # mean of the two half-step derivatives IS the centre derivative.
    # Normalising also cannot wrap: the bisector of two unit phasors is
    # always the correct mean direction.
    def _unit(prod, have):
        mag = np.abs(prod)
        ok = have & (mag > 0.0)
        return np.where(ok, prod / np.where(ok, mag, 1.0), 0.0)

    px_plus = _unit(Ex_plus * np.conj(E_c), have_xp)
    px_minus = _unit(E_c * np.conj(Ex_minus), have_xm)
    py_plus = _unit(Ey_plus * np.conj(E_c), have_yp)
    py_minus = _unit(E_c * np.conj(Ey_minus), have_ym)

    kx = np.angle(px_plus + px_minus) / dx
    ky = np.angle(py_plus + py_minus) / dy

    L = kx / k0
    M = ky / k0
    return _direction_cosines(L, M)


def _angle_unwrap_gradient(
    E: np.ndarray,
    dx: float,
    dy: float,
    ix: np.ndarray,
    iy: np.ndarray,
    k0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unwrap-based angle estimator (fragile near singularities).

    ``k_perp = grad(unwrap(angle(E)))``, unwrapped along each axis
    independently with :func:`numpy.unwrap`.  Provided for cases
    where the user knows the phase is smooth (no vortices) and wants
    to side-step the complex-gradient clamp.
    """
    phi = np.angle(E)
    phi_unwrap_x = np.unwrap(phi, axis=1)
    phi_unwrap_y = np.unwrap(phi, axis=0)
    # Gradient of unwrapped phase along each axis.
    dphi_dx = np.gradient(phi_unwrap_x, dx, axis=1)
    dphi_dy = np.gradient(phi_unwrap_y, dy, axis=0)

    kx = dphi_dx[iy, ix]
    ky = dphi_dy[iy, ix]

    L = kx / k0
    M = ky / k0
    return _direction_cosines(L, M)


def _direction_cosines(
    L: np.ndarray,
    M: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``N = sqrt(1 - L^2 - M^2)`` with evanescent flagging.

    ``L^2 + M^2 > 1`` corresponds to evanescent k-vectors; we set
    ``N = 0`` for those rays and return a boolean mask so the caller
    can populate ``alive`` / ``error_code``.
    """
    L = np.asarray(L, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    sum_sq = L * L + M * M
    evanescent = sum_sq > 1.0
    N = np.where(
        evanescent,
        0.0,
        np.sqrt(np.maximum(1.0 - sum_sq, 0.0)),
    )
    return L, M, N, evanescent

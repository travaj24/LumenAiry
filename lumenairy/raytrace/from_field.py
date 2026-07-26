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
  adjacent field samples
  ``k_x = arg(E[i,j+1] * conj(E[i,j-1])) / (2 dx)``
  (the singularity-safe ``complex_gradient`` mode, default) or from
  the gradient of the unwrapped phase ``k_perp = grad(unwrap(angle
  E))`` (the ``unwrap_gradient`` mode).  The phase-ratio form is the
  *discrete analogue* of the continuous Madelung formula
  ``k_perp = Im(grad E / E)`` (CLUSTER_B_SPEC.md §4.4) but is exact
  for plane waves whenever ``|k_x dx| < pi`` whereas the literal
  central-difference ``Im(grad E / E)`` carries a
  ``sinc(k_x dx / pi)`` discretisation bias.  The phase-ratio form
  is also automatically singularity-safe because ``np.angle(.)`` is
  bounded in ``(-pi, pi]`` regardless of how small either factor is.
  See the inline docstring on :func:`_angle_complex_gradient` for
  the full derivation and the spec-deviation rationale.
  Direction cosines follow from ``L = k_x / k0``, ``M = k_y / k0``,
  ``N = sqrt(1 - L^2 - M^2)``.  Rays whose ``L^2 + M^2 > 1`` correspond
  to evanescent k-vectors and are marked ``alive = False`` /
  ``error_code = RAY_EVANESCENT``.
* OPD: initialised to ``phi(x_ray, y_ray) / k0`` so subsequent
  geometric-ray OPD accumulation continues correctly from the
  wave-optical state.

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
v4.15.3 -- Agent D: intensity-threshold comparison made consistent
across the 3 placement modes; all three now use the inclusive
``|E|^2 / max(|E|^2) >= intensity_threshold`` convention.  Pre-v4.15.3
only ``_place_rejection`` used ``>=``; ``_place_cdf`` and
``_place_uniform`` used strict ``>``, dropping boundary-exact pixels.
This is a numerical behaviour change for inputs at the exact threshold
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
          the CDF.  (v4.15.2 fix; see P1-NEW-D in the v4.15.1 audit.
          v4.15.3 -- ``>=`` is the canonical convention; pre-v4.15.3
          used strict ``>``.)  Use for visualisation.
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
        retained.  v4.15.3 -- the three modes are now consistent;
        pre-v4.15.3 only ``_place_rejection`` used the inclusive
        comparison while ``_place_cdf`` and ``_place_uniform`` used
        strict ``>``.  For ``angle_method='complex_gradient'`` the
        threshold also clamps the denominator of ``grad E / E``.
    z0 : float, default 0.0
        Axial position of all ray origins [m].
    random_state : int or np.random.Generator or None, optional
        Seed or generator used for the random placement.  ``None``
        uses :func:`numpy.random.default_rng()` with fresh entropy.
        Pass an int (or a fixed Generator) for reproducibility.

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
        * ``opd`` -- ``phi(x_ray, y_ray) / k0``  [m].
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
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
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
    phi = np.angle(E)
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

    v4.15.2 fix (P1-NEW-D in the v4.15.1 audit) -- threshold is now
    applied pixel-wise BEFORE the marginal sums.  Previously the
    threshold was applied to the marginal sums themselves
    (``Ix.sum(axis=0) > threshold * Ix.max()``), which let sub-
    threshold background noise accumulate across rows / columns and
    contaminate the CDF.  Pixel-wise thresholding makes ``'cdf'``
    consistent with ``'rejection'`` and ``'uniform'``.

    v4.15.3 -- threshold comparison is inclusive (``>=``) so pixels at
    exactly ``intensity_threshold`` are retained, matching the
    canonical "pixel intensity meets threshold" convention used in
    ``_place_rejection`` and now also ``_place_uniform``.  Pre-v4.15.3
    used strict ``>``.  Documented in the v4.15.3 release notes.
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

    v4.15.3 -- threshold comparison is inclusive (``>=``) here as in
    all three placement modes; the canonical convention is "pixel
    intensity meets threshold" so a pixel whose normalised intensity
    is exactly ``intensity_threshold`` is retained.  Pre-v4.15.3 only
    this mode used ``>=`` (which became the chosen convention); the
    other two modes (``_place_cdf``, ``_place_uniform``) used strict
    ``>``.  The behaviour change for inputs at the exact threshold
    boundary is documented in the v4.15.3 release notes.
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

    v4.15.2 (P3 from the v4.15.1 audit): when ``n_rays`` exceeds the
    number of unique pixels in the grid, the sub-grid pixelisation
    used to produce duplicate ``(iy, ix)`` entries; we now dedupe in
    a stable manner.

    v4.15.3 -- threshold comparison is inclusive (``>=``) so pixels at
    exactly ``intensity_threshold`` are retained, matching the
    canonical "pixel intensity meets threshold" convention used in
    ``_place_rejection`` and now also ``_place_cdf``.  Pre-v4.15.3
    used strict ``>``.  Documented in the v4.15.3 release notes.
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
    field samples:

    .. math::

        k_x(i,j) = \\frac{1}{2 \\Delta x}\\,
                  \\arg\\!\\bigl( E_{i,j+1}\\,E_{i,j-1}^*\\bigr)

    (and analogously for ``k_y``).  This is the discrete analogue of
    the Madelung formula :math:`k_\\perp = \\operatorname{Im}(\\nabla E
    / E)` but is *exact* for plane waves with :math:`|k_x \\Delta x|
    < \\pi/2` (whereas :math:`\\operatorname{Im}(\\nabla E / E)`
    computed with central differences carries an
    :math:`\\operatorname{sinc}(k_x \\Delta x / \\pi)` discretisation
    bias).  It is also automatically singularity-safe -- ``np.angle``
    on a complex product is bounded in :math:`(-\\pi, \\pi]`
    regardless of how small either factor is.

    The denominator clamp from ``intensity_threshold`` is still
    applied for the rare case where both neighbouring pixels are
    identically zero (e.g. far outside a hard aperture): there the
    product :math:`E_{j+1} E_{j-1}^*` is zero and ``np.angle`` returns
    0, which we accept as "no useful gradient information here".

    Deviates from CLUSTER_B_SPEC.md §4.4's literal
    ``Im(grad E / E)`` formula in favour of this phase-ratio form
    because the literal formula is biased by
    :math:`\\operatorname{sinc}(k_x \\Delta x / \\pi)` under central
    differences and therefore (a) fails the 5-deg tilted-plane-wave
    test at the spec's tolerance for the spec's sampling, and (b)
    cannot detect evanescent rays whose tangential :math:`k` exceeds
    :math:`\\pi / \\Delta x` because the central-difference output
    rolls back through zero.  The phase-ratio form is the standard
    "computational k-vector" used in quantum-mechanics literature
    (Madelung continuity equation discretisation) and matches the
    spec's stated intent: recover the local k-vector with no
    explicit unwrap.
    """
    Ny, Nx = E.shape

    # Phase-ratio neighbours: pad the array with the boundary value so
    # edge pixels still get a sensible value.  Use np.roll with
    # boundary trimming via clip on the neighbour indices.
    ix = np.asarray(ix, dtype=np.intp)
    iy = np.asarray(iy, dtype=np.intp)
    ix_plus = np.clip(ix + 1, 0, Nx - 1)
    ix_minus = np.clip(ix - 1, 0, Nx - 1)
    iy_plus = np.clip(iy + 1, 0, Ny - 1)
    iy_minus = np.clip(iy - 1, 0, Ny - 1)

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

    Ex_plus = _safe_sample(E, iy, ix_plus)
    Ex_minus = _safe_sample(E, iy, ix_minus)
    Ey_plus = _safe_sample(E, iy_plus, ix)
    Ey_minus = _safe_sample(E, iy_minus, ix)

    # arg(E[i,j+1] * conj(E[i,j-1])) / (2 dx)  -- exact for plane waves
    # when 2*dx < lambda (i.e. |kx*dx| < pi).
    kx = np.angle(Ex_plus * np.conj(Ex_minus)) / (2.0 * dx)
    ky = np.angle(Ey_plus * np.conj(Ey_minus)) / (2.0 * dy)

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

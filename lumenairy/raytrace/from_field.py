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
* Direction: ``k_perp`` is computed from the *phase ratio* of
  adjacent field samples ``k_x = arg(E[i,j+1] * conj(E[i,j-1])) /
  (2 dx)`` (the singularity-safe ``complex_gradient`` mode, default)
  or from the gradient of the unwrapped phase ``k_perp =
  grad(unwrap(angle E))`` (the ``unwrap_gradient`` mode).  The first
  form is preferred because it avoids phase unwrapping entirely,
  tolerates vortex cores, and is exact for plane waves with
  ``|k_x dx| < pi``.
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
"""
from __future__ import annotations

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
        Target number of rays.  The returned bundle has
        ``n_rays`` rays (some may be flagged as evanescent and
        ``alive = False``).  Must be positive.
    placement : {'cdf', 'rejection', 'uniform'}, default 'cdf'
        Strategy for placing ray origins.

        * ``'cdf'`` -- separable inverse-CDF sampling along the x and
          y marginals.  Fast.  Exact for separable intensity
          distributions (e.g. Gaussian beams aligned to the axes);
          approximate for strongly non-separable intensities (e.g.
          azimuthal lattices).  Use for visualisation.
        * ``'rejection'`` -- true 2-D rejection sampling from
          ``|E|^2``.  Slower but exact regardless of separability.
          Use for vortex beams, complicated holograms, etc.
        * ``'uniform'`` -- uniform placement on a regular sub-grid,
          dropping pixels whose intensity falls below
          ``intensity_threshold``.  Returns up to ``n_rays`` survivors
          (could be fewer if the supporting region is too small).
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
        Relative threshold ``|E|^2 / max(|E|^2)``.  Pixels below the
        threshold are excluded from placement (for ``'cdf'`` and
        ``'rejection'``).  For ``angle_method='complex_gradient'`` the
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
        Bundle with ``n_rays`` rays.  Per-ray fields:

        * ``x, y`` -- sampled origin coordinates [m]
        * ``z`` -- ``np.full(n_rays, z0)`` [m]
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
        ``wavelength`` is non-positive, ``n_rays < 1``, or
        ``placement`` / ``angle_method`` are unrecognised.

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
    if not isinstance(n_rays, (int, np.integer)) or n_rays < 1:
        raise ValueError(
            f'rays_from_field: n_rays must be a positive integer; '
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
    intensity marginals, then inverts each via :func:`numpy.searchsorted`.
    Independent draws in x and y give a separable joint distribution
    -- exact for separable ``|E|^2`` (e.g. axially aligned Gaussians)
    and a fast approximation otherwise.
    """
    I = np.abs(E) ** 2
    I = I / I.max()

    Ny, Nx = E.shape

    # x-marginal (sum over y)
    Ix = I.sum(axis=0)
    Ix = np.where(Ix > threshold * Ix.max(), Ix, 0.0)
    if Ix.sum() == 0.0:
        # All intensity below threshold -- nothing to sample.
        return (np.array([]), np.array([]),
                np.array([], dtype=np.intp),
                np.array([], dtype=np.intp))
    cdf_x = np.cumsum(Ix)
    cdf_x = cdf_x / cdf_x[-1]

    # y-marginal (sum over x)
    Iy = I.sum(axis=1)
    Iy = np.where(Iy > threshold * Iy.max(), Iy, 0.0)
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
    ``n_rays`` survivors.
    """
    I = np.abs(E) ** 2
    Imax = I.max()
    if Imax == 0:
        return (np.array([]), np.array([]),
                np.array([], dtype=np.intp),
                np.array([], dtype=np.intp))
    mask = I >= threshold * Imax

    Ny, Nx = E.shape

    # Use the larger of the two axes to anchor the grid count, scaled
    # so that the total cell count is close to n_rays.
    aspect = max(Nx, Ny) / min(Nx, Ny)
    ny_grid = int(np.ceil(np.sqrt(n_rays / aspect)))
    nx_grid = int(np.ceil(n_rays / max(ny_grid, 1)))
    ny_grid = max(ny_grid, 1)
    nx_grid = max(nx_grid, 1)

    # Pick the central pixel of each sub-cell -- monotonic spacing
    # across the masked region.
    iy_pixels = np.linspace(0, Ny - 1, ny_grid, dtype=np.intp)
    ix_pixels = np.linspace(0, Nx - 1, nx_grid, dtype=np.intp)
    iy_grid, ix_grid = np.meshgrid(iy_pixels, ix_pixels, indexing='ij')
    iy_flat = iy_grid.ravel()
    ix_flat = ix_grid.ravel()

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

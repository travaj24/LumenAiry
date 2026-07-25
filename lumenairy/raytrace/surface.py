"""
v5.1.0 split: surface representations + sag/normal math.

Extracted from ``lumenairy/raytrace/core.py`` as part of the v5.1.0
6-file split (ROADMAP Agent B).  Hosts the core data classes
(:class:`RayBundle`, :class:`Surface`, :class:`TraceResult`), the
per-ray error-code constants, and the surface-sag / surface-normal
helpers.  Every public name here is re-exported from
``lumenairy.raytrace.core`` so existing imports continue to resolve.

No physics change: contents are bit-for-bit copies of the original
implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..elements.lenses import surface_sag_biconic, surface_sag_general

# ============================================================================
# Data structures
# ============================================================================

# Error codes for RayBundle.error_code.  Codes are cumulative "first
# failure wins": once set non-zero, a ray's error_code sticks for the
# rest of the trace so downstream surfaces can't overwrite the root
# cause.  alive = (error_code == 0) by invariant -- helpers that
# vignette rays MUST set error_code at the same time.
RAY_OK              = 0    # ray is alive
RAY_TIR             = 1    # total internal reflection at refract
RAY_APERTURE        = 2    # clipped by a surface semi_diameter
RAY_MISSED_SURFACE  = 3    # intersection Newton failed / no real root
RAY_NAN             = 4    # arithmetic produced NaN/Inf (numerical fault)
RAY_EVANESCENT      = 5    # diffraction order does not propagate
                           # (L'^2 + M'^2 > 1 after a grating k-shift)


@dataclass
class RayBundle:
    """A bundle of rays represented as parallel numpy arrays.

    Each array has shape ``(N_rays,)``.  All spatial coordinates are in
    metres; direction cosines satisfy ``L**2 + M**2 + N**2 == 1``.

    Attributes
    ----------
    x, y, z : ndarray
        Ray positions [m].
    L, M, N : ndarray
        Direction cosines (x, y, z components).
    wavelength : float
        Vacuum wavelength [m].
    alive : ndarray of bool
        ``False`` for rays that have been vignetted or suffered TIR.
        Derived quantity: ``alive = (error_code == RAY_OK)``.
    opd : ndarray
        Accumulated optical path length [m] along each ray.
    error_code : ndarray of uint8
        Per-ray diagnostic code (``RAY_OK`` / ``RAY_TIR`` /
        ``RAY_APERTURE`` / ``RAY_MISSED_SURFACE`` / ``RAY_NAN``).
        First-failure-wins: once a ray is killed with a non-zero
        code, subsequent surfaces do NOT overwrite the root cause.
        Useful for post-trace diagnostics -- see
        :func:`trace_summary` for the breakdown.
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    L: np.ndarray
    M: np.ndarray
    N: np.ndarray
    wavelength: float
    alive: np.ndarray
    opd: np.ndarray
    # 3.1.9: per-ray diagnostic code.  Default-factory keeps older
    # constructors (pickled bundles, user code that creates bundles
    # directly without this field) working -- a missing error_code is
    # synthesised from alive as "alive -> OK, dead -> TIR (unknown)."
    error_code: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # Synthesise error_code if the caller didn't supply one.  This
        # keeps any pre-3.1.9 code paths (user-constructed bundles,
        # pickled objects from older versions) working transparently.
        if self.error_code is None:
            ec = np.zeros(len(self.x), dtype=np.uint8)
            ec[~np.asarray(self.alive, dtype=bool)] = RAY_TIR
            # NB: "unknown dead" defaults to RAY_TIR as a placeholder;
            # actual downstream codes are set at kill time by
            # _intersect_surface / _refract / _reflect.
            self.error_code = ec

    @property
    def n_rays(self) -> int:
        return len(self.x)

    def copy(self) -> 'RayBundle':
        return RayBundle(
            x=self.x.copy(), y=self.y.copy(), z=self.z.copy(),
            L=self.L.copy(), M=self.M.copy(), N=self.N.copy(),
            wavelength=self.wavelength,
            alive=self.alive.copy(), opd=self.opd.copy(),
            error_code=(self.error_code.copy()
                         if self.error_code is not None else None),
        )

    def to_jax_state(self) -> Any:
        """Convert this RayBundle to a :class:`JaxRayState`.

        The JAX state drops ``wavelength`` and ``error_code`` (which
        have no JAX-traceable analogue); ``alive`` is preserved as the
        JAX bool mask.  Useful when handing a NumPy-traced bundle into
        ``trace_jax`` for differentiable downstream optics.
        """
        from .jax_trace import make_jax_ray_state
        return make_jax_ray_state(
            x=self.x, y=self.y, z=self.z,
            L=self.L, M=self.M, N=self.N,
            opd=self.opd, alive=self.alive,
        )


@dataclass
class Surface:
    """A single optical surface in the sequential model.

    Attributes
    ----------
    radius : float
        Radius of curvature [m].  ``inf`` for flat.
    conic : float
        Conic constant (0 = sphere, -1 = paraboloid).
    aspheric_coeffs : dict or None
        Even polynomial coefficients ``{power: coeff}``
        (e.g. ``{4: A4, 6: A6}``).
    semi_diameter : float
        Clear semi-aperture [m].  Rays outside are vignetted.
    glass_before : str
        Glass name on the input side (e.g. ``'air'``).
    glass_after : str
        Glass name on the output side (e.g. ``'N-BK7'``).
    is_mirror : bool
        If True, the surface reflects rather than refracts.
    is_stop : bool
        If True, marks this surface as the aperture stop of the
        system.  Used by stop-aware helpers (``find_stop``,
        ``compute_pupils``, ``seidel_coefficients``, etc.) to anchor
        the chief ray.  Zemax ``.zmx`` / ``.txt`` loaders set this
        from the STOP keyword; the legacy fallback behaviour (first
        surface with a finite semi-diameter, else surface 0) is
        preserved by ``find_stop`` when no surface is flagged.
    thickness : float
        Axial distance to the *next* surface [m].
    label : str
        Human-readable label for the surface.
    surf_num : int
        Zemax surface number (for reference).
    """
    radius: float = np.inf
    conic: float = 0.0
    aspheric_coeffs: Optional[Dict] = None
    semi_diameter: float = np.inf
    glass_before: str = 'air'
    glass_after: str = 'air'
    is_mirror: bool = False
    is_stop: bool = False
    thickness: float = 0.0
    label: str = ''
    surf_num: int = -1
    # 3.7.0: Coordinate-break support.  When ``is_coordbrk`` is True
    # this Surface is treated as a Zemax-style COORDBRK -- the trace
    # loop transforms the ray bundle's frame (decenter then tilt, or
    # the reverse if ``coordbrk_order`` is non-zero) and skips
    # intersection / refraction / reflection.  Paraxial helpers
    # (system_abcd, find_paraxial_focus, compute_pupils,
    # seidel_coefficients, find_stop) skip these.  ``thickness`` on a
    # coord-break is the air gap from the break to the next surface
    # in the local (post-transform) frame, exactly like Zemax DISZ.
    is_coordbrk: bool = False
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0
    tilt_z_deg: float = 0.0
    decenter_x_m: float = 0.0
    decenter_y_m: float = 0.0
    # Zemax PARM 6: 0 = pre-decenter, then-tilt; 1 = pre-tilt, then-
    # decenter.  We match Zemax's default (0) for compatibility.
    coordbrk_order: int = 0
    # Biconic / anamorphic extensions (all optional; None => rotationally
    # symmetric surface using radius / conic / aspheric_coeffs above).
    radius_y: Optional[float] = None
    conic_y: Optional[float] = None
    aspheric_coeffs_y: Optional[Dict] = None
    # Optional freeform departure layered on top of the (biconic) base
    # sag.  Dict keys depend on ``kind``; see freeform.surface_sag_freeform.
    # Recognised forms:
    #   {'kind': 'xy_polynomial', 'coefficients': {(i,j): a_ij, ...}}
    #   {'kind': 'zernike',      'coefficients': {(n,m): c_nm, ...},
    #    'aperture_radius': r}
    #   {'kind': 'chebyshev',    'coefficients': {(i,j): c_ij, ...},
    #    'normalization_radius': r}
    # Surface normals through freeform departures use a finite-difference
    # gradient at the ray's intersection point.
    freeform: Optional[Dict] = None
    # Optional BSDF (bidirectional scattering distribution function) for
    # stray-light analysis.  Either a BSDFModel instance or a dict spec
    # consumed by :func:`bsdf.make_bsdf`.  Does not affect the specular
    # trace; invoke :func:`bsdf.sample_scatter_rays` to spawn scatter
    # rays from a surface that carries this field.
    bsdf: Optional[object] = None
    # Reflective-coating material for a mirror (``is_mirror=True``): a
    # complex refractive index ``n + i*kappa`` (metal) that drives the
    # Fresnel ``r_s`` / ``r_p`` used by polarization ray tracing
    # (``propagators.gbd._fresnel_jones_matrix_per_beamlet``), giving the
    # diattenuation + retardance of a real metal mirror.  Accepts a material
    # NAME (looked up via :func:`glass.get_glass_index_complex`), a callable
    # ``wavelength -> complex``, or a direct complex value.  ``None`` -> an
    # ideal reflector (``|r_s| = |r_p| = 1``).  Does not affect the geometric
    # trace (reflection is pure geometry).
    coating: Optional[object] = None

    # N10a (2026-07-20): per-surface FIELD-FRAME decenter / tilt / freeform,
    # the ``apply_real_lens`` displaced-pointwise (P3) convention -- DISTINCT
    # from the rigid-body coordinate-break block (``decenter_x_m`` /
    # ``tilt_x_deg``), which transforms the whole ray frame.  Field-frame means
    # the sag is evaluated at the shifted coordinate ``sag(x - dx, y - dy)``,
    # small-angle tilt adds the linear ramp ``tx*(x-dx) + ty*(y-dy)`` plus the
    # correspondingly tilted normal, and ``field_sag_callable(xs, ys)`` REPLACES
    # the base sag (matching ``_disp_surface_z_grad`` /
    # ``geom_spot_decenter_oracle`` exactly).  These are honoured by the shared
    # ``_surface_sag_xy`` / ``_surface_sag_derivatives_xy`` / ``_surface_normal``
    # helpers, so BOTH the traced (``trace``) and GBD
    # (``ray_transfer_jacobian``) ray models pick up the transverse ray
    # walk-off -- hence the true induced coma -- naturally.  All default None /
    # (0, 0), so a surface without them is byte-identical to pre-N10a.
    field_decenter: Optional[Any] = None       # (dx, dy) [m]
    field_tilt: Optional[Any] = None           # (tx, ty) [rad], small-angle
    field_sag_callable: Optional[Any] = None   # (xs, ys) -> sag [m]

    # 3.7.5: World-frame trace support.  When ``world_origin`` and
    # ``world_R`` are both populated, :func:`trace_world` propagates
    # rays in world coordinates between surfaces and transforms them
    # into this surface's local frame only for the intersection /
    # refraction / reflection step.  ``thickness`` is unused on the
    # world path -- the gap to the next surface is implicit in the
    # next surface's ``world_origin``.  ``is_coordbrk`` Surfaces are
    # not emitted on the world path (their tilts are baked into the
    # next surface's ``world_R``).  Untilted, axially symmetric
    # systems can still use the legacy local-frame :func:`trace` --
    # both paths give identical results when world frames equal
    # ``origin = (0, 0, cum_z), R = I``.
    world_origin: Optional[np.ndarray] = None
    world_R: Optional[np.ndarray] = None


@dataclass
class TraceResult:
    """Result of tracing a ray bundle through a sequential system.

    Attributes
    ----------
    surfaces : list of Surface
        The surface list that was traced.
    ray_history : list of RayBundle
        Ray state *after* each surface (index 0 = after surface 0).
        ``ray_history[-1]`` is the final image-plane intercept.
    input_rays : RayBundle
        The original input rays (before any surface).
    wavelength : float
        Vacuum wavelength [m].
    """
    surfaces: List[Surface]
    ray_history: List[RayBundle]
    input_rays: RayBundle
    wavelength: float

    @property
    def image_rays(self) -> RayBundle:
        """Rays at the final (image) surface."""
        return self.ray_history[-1]

    def rays_at(self, surface_index: int) -> RayBundle:
        """Return the ray bundle after the given surface."""
        return self.ray_history[surface_index]


# ============================================================================
# Surface sag and normal computation
# ============================================================================

def _field_frame_active(surface) -> bool:
    """True when this surface carries a P3-style FIELD-FRAME decenter / tilt /
    freeform ``field_sag_callable`` (the ``apply_real_lens`` displaced-pointwise
    convention).  DISTINCT from the rigid-body coordinate-break fields
    (``decenter_x_m`` / ``tilt_x_deg``), which transform the whole ray frame.

    When this returns False every geometry helper takes its pre-N10a code path
    byte-for-byte -- the zero-decenter regression pin."""
    if getattr(surface, 'field_sag_callable', None) is not None:
        return True
    fd = getattr(surface, 'field_decenter', None)
    if fd is not None and (float(fd[0]) != 0.0 or float(fd[1]) != 0.0):
        return True
    ft = getattr(surface, 'field_tilt', None)
    if ft is not None and (float(ft[0]) != 0.0 or float(ft[1]) != 0.0):
        return True
    return False


def _field_frame_sag_and_grad(x, y, surface):
    """SINGLE shared field-frame transformed-sag + gradient helper (N10a/N10b).

    Returns ``(f, dfdx, dfdy)`` -- the surface z-departure and its transverse
    gradient at FIELD coordinates ``(x, y)`` [m] -- honouring the per-surface
    ``field_decenter=(dx, dy)`` (sag evaluated at ``x - dx, y - dy``),
    small-angle ``field_tilt=(tx, ty)`` (linear ramp ``tx*(x-dx) + ty*(y-dy)``
    plus the correspondingly tilted normal ``dfdx += tx``), and a freeform
    ``field_sag_callable(xs, ys)`` (which REPLACES the base sag).  This is the
    SAME field-frame convention as the analytic ``_disp_surface_z_grad`` and the
    lumenairy-free ``geom_spot_decenter_oracle``, so the traced
    (``trace`` -> ``_intersect_surface`` / ``_refract``) and GBD
    (``ray_transfer_jacobian`` finite-difference) ray models agree with the
    analytic model on what a decenter MEANS (cross-model agreement pinned by
    ``tests/unit/test_niche_p9_decenter_tilt.py``).  Both ray models share this
    ONE definition -- there is no divergent second copy of the geometry.

    The surface normal follows from the caller: ``_surface_normal`` returns
    ``(-dfdx, -dfdy, 1) / |.|``, so the transverse walk-off between a thick
    element's surfaces (the true induced coma) emerges naturally once each ray
    is carried through the glass gaps."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fd = getattr(surface, 'field_decenter', None) or (0.0, 0.0)
    ft = getattr(surface, 'field_tilt', None) or (0.0, 0.0)
    fc = getattr(surface, 'field_sag_callable', None)
    dcx, dcy = float(fd[0]), float(fd[1])
    tx, ty = float(ft[0]), float(ft[1])
    xs = x - dcx
    ys = y - dcy
    if fc is not None:
        # Freeform hook (REPLACES the base sag): FD gradient at a fixed
        # sub-micron step (matches ``_disp_surface_z_grad``).
        step = 1.0e-7
        f = np.asarray(fc(xs, ys), dtype=np.float64)
        dfdx = (np.asarray(fc(xs + step, ys), dtype=np.float64)
                - np.asarray(fc(xs - step, ys), dtype=np.float64)) / (2.0 * step)
        dfdy = (np.asarray(fc(xs, ys + step), dtype=np.float64)
                - np.asarray(fc(xs, ys - step), dtype=np.float64)) / (2.0 * step)
    else:
        # Base conic / aspheric / biconic / freeform sag + analytic gradient,
        # evaluated at the decentered coordinates (d/dx of sag(x-dx) =
        # sag'(x-dx), so the base helpers at (xs, ys) give the shifted gradient).
        f = _base_surface_sag_xy(xs, ys, surface)
        dfdx, dfdy = _base_surface_sag_derivatives_xy(xs, ys, surface)
    if tx != 0.0 or ty != 0.0:
        f = f + tx * xs + ty * ys
        dfdx = dfdx + tx
        dfdy = dfdy + ty
    return f, dfdx, dfdy


def _surface_sag_xy(x, y, surface):
    """Sag z = f(x, y) for an arbitrary (possibly biconic / freeform /
    field-frame decentered / tilted) surface.

    When the surface carries a FIELD-FRAME decenter / tilt / ``field_sag_callable``
    (N10a) the shared :func:`_field_frame_sag_and_grad` helper is used; otherwise
    the byte-identical base dispatch:

    1. ``surface.freeform`` set       -> :func:`freeform.surface_sag_freeform`
       (uses the surface's own ``radius`` + ``conic`` as the base
       conic; departure is XY-poly / Zernike / Chebyshev per the spec).
    2. ``surface.radius_y`` set       -> :func:`surface_sag_biconic`.
    3. otherwise                      -> :func:`surface_sag_general`.
    """
    if _field_frame_active(surface):
        f, _, _ = _field_frame_sag_and_grad(x, y, surface)
        return f
    return _base_surface_sag_xy(x, y, surface)


def _base_surface_sag_xy(x, y, surface):
    """Base (non-field-frame) sag dispatch -- see :func:`_surface_sag_xy`."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    spec = getattr(surface, 'freeform', None)
    if spec:
        # Freeform surfaces include their own base sphere/conic, so
        # they're an "instead of" rather than "on top of" the biconic
        # base.  Rotationally-symmetric base only -- biconic-base
        # freeforms are not yet supported by freeform.surface_sag_freeform
        # and would silently drop the radius_y term, which is worse
        # than warning the user up front.
        if getattr(surface, 'radius_y', None) is not None:
            raise NotImplementedError(
                "Freeform departure on a biconic base is not supported "
                "yet: leave radius_y unset, or extend "
                "freeform.surface_sag_freeform with biconic support.")
        from ..elements.freeform import surface_sag_freeform
        sd = dict(spec)
        # surface_sag_freeform reads 'freeform_type', 'radius', 'conic',
        # plus per-kind keys.  The Surface dataclass already supplies
        # radius/conic, so merge them in if not overridden.
        sd.setdefault('radius', surface.radius)
        sd.setdefault('conic', surface.conic)
        return surface_sag_freeform(x, y, sd)

    if getattr(surface, 'radius_y', None) is None:
        return surface_sag_general(
            x * x + y * y, surface.radius, surface.conic,
            surface.aspheric_coeffs)
    # Biconic path
    return surface_sag_biconic(
        x, y, R_x=surface.radius, R_y=surface.radius_y,
        conic_x=surface.conic,
        conic_y=surface.conic_y,
        aspheric_coeffs=surface.aspheric_coeffs,
        aspheric_coeffs_y=surface.aspheric_coeffs_y)


def _surface_sag_derivatives_xy(x, y, surface):
    """Partial derivatives (dz/dx, dz/dy) at (x, y) on the given surface.

    Used for the surface normal in refraction / reflection and for the
    slant-correction formula.  Honours a FIELD-FRAME decenter / tilt /
    ``field_sag_callable`` (N10a) via the shared
    :func:`_field_frame_sag_and_grad` helper; otherwise the byte-identical base
    dispatch (biconic surfaces sum per-axis derivatives; freeform surfaces use a
    centred finite difference).
    """
    if _field_frame_active(surface):
        _, dfdx, dfdy = _field_frame_sag_and_grad(x, y, surface)
        return dfdx, dfdy
    return _base_surface_sag_derivatives_xy(x, y, surface)


def _base_surface_sag_derivatives_xy(x, y, surface):
    """Base (non-field-frame) sag derivatives -- see
    :func:`_surface_sag_derivatives_xy`."""
    # Freeform path: centred FD with h scaled to local feature size.
    if getattr(surface, 'freeform', None):
        # Step size: small fraction of typical aperture (use R or 1 mm).
        R = surface.radius if (surface.radius is not None
                               and np.isfinite(surface.radius)) else 1e-3
        h_step = max(abs(R) * 1e-6, 1e-9)
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        z_xp = _surface_sag_xy(x_arr + h_step, y_arr, surface)
        z_xm = _surface_sag_xy(x_arr - h_step, y_arr, surface)
        z_yp = _surface_sag_xy(x_arr, y_arr + h_step, surface)
        z_ym = _surface_sag_xy(x_arr, y_arr - h_step, surface)
        return ((z_xp - z_xm) / (2 * h_step),
                (z_yp - z_ym) / (2 * h_step))
    if getattr(surface, 'radius_y', None) is None:
        # Rotationally symmetric -- reuse the scalar helpers.
        h = np.sqrt(x * x + y * y)
        h_safe = np.maximum(h, 1e-30)
        dz_dh = _surface_sag_derivative(
            h, surface.radius, surface.conic, surface.aspheric_coeffs)
        dz_dx = np.where(h > 0, dz_dh * x / h_safe, 0.0)
        dz_dy = np.where(h > 0, dz_dh * y / h_safe, 0.0)
        return dz_dx, dz_dy

    # Biconic -- derivative of each axis independently.
    def _axis_deriv(u, R, K, asph):
        if R is None or np.isinf(R):
            d = np.zeros_like(u)
        else:
            h_sq = u * u
            norm = (1 + K) * h_sq / R ** 2
            valid = norm < 0.9999
            denom = np.where(valid, np.sqrt(np.maximum(1 - norm, 1e-30)), 1.0)
            # v5.4.7 (audit AUDIT_V5_4_6 gap #2): NaN (not 0.0) out of the
            # conic domain, matching the rot-sym _surface_sag_derivative
            # and the surface_sag_general / v5.4.6 F-19 sag fix.  A silent
            # 0.0 here gives a bogus axial (flat) surface normal where the
            # sag is already NaN -- the exact silent-wrong-geometry
            # inconsistency the F-19 fix targeted (biconic + out-of-domain
            # ray).
            d = np.where(valid, u / (R * denom), np.nan)
        if asph:
            for power, coeff in asph.items():
                # d/du of coeff * u^power = power * coeff * u^(power-1)
                d = d + power * coeff * u ** (power - 1)
        return d

    asph_y = (surface.aspheric_coeffs_y
              if surface.aspheric_coeffs_y is not None
              else surface.aspheric_coeffs)
    # S9-RT2 (audit review4a): honour the SAME ``conic_y is None -> conic_x``
    # default that the sag twin :func:`lenses.surface_sag_biconic` documents
    # and implements ("conic_y : float, optional -- Conic constant along y.
    # Defaults to ``conic_x`` if not given").  ``Surface.conic_y`` defaults to
    # ``None``, so without this the DERIVATIVE (surface-normal) path raised
    # ``TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'``
    # at ``(1 + K)`` for EVERY biconic / cylindrical surface whose ``conic_y``
    # was left unset, while the sag path computed fine -- i.e. a
    # ``Surface(radius=50e-3, radius_y=60e-3)`` could not be traced at all
    # (verified through the public ``trace()`` API).  Bit-identical whenever
    # ``conic_y`` is not ``None`` (the only inputs that previously worked).
    conic_y = (surface.conic_y if surface.conic_y is not None
               else surface.conic)
    dz_dx = _axis_deriv(x, surface.radius, surface.conic,
                        surface.aspheric_coeffs)
    dz_dy = _axis_deriv(y, surface.radius_y, conic_y, asph_y)
    return dz_dx, dz_dy


def _surface_sag_derivative(h, R, conic=0.0, aspheric_coeffs=None):
    """Derivative of sag with respect to radial distance h = sqrt(h_sq).

    Returns dz/dh for computing surface normals.
    """
    h = np.asarray(h, dtype=np.float64)

    dz_dh = np.zeros_like(h)

    if R is not None and not np.isinf(R):
        h_sq = h ** 2
        norm = (1 + conic) * h_sq / R ** 2
        valid = norm < 0.9999
        denom = np.where(valid, np.sqrt(np.maximum(1 - norm, 1e-30)), 1.0)
        # d(sag)/dh for conic: h / (R * sqrt(1 - (1+k)*h^2/R^2)).
        # v5.4.6 (audit P3-2): outside the conic domain ((1+k)h^2/R^2 >= 1)
        # there is no real surface, so the derivative is NaN (matching
        # surface_sag_general / the F-19 biconic fix) rather than a silent
        # 0.0 that yields a bogus axial (flat) normal and masks the bad
        # geometry.  Rays in that region now carry NaN normals and are
        # flagged downstream instead of refracting through nonsense.
        dz_dh = np.where(valid, h / (R * denom), np.nan)

    if aspheric_coeffs:
        for power, coeff in aspheric_coeffs.items():
            # d/dh of coeff * h^power = power * coeff * h^(power-1)
            dz_dh = dz_dh + power * coeff * h ** (power - 1)

    return dz_dh


def _surface_normal(x, y, surface):
    """Outward unit normal at point (x, y) on the given surface.

    Returns (nx, ny, nz) arrays.  The normal points from glass_before
    toward glass_after (i.e. in the +z direction for a flat surface).
    Handles biconic / anamorphic surfaces via ``_surface_sag_derivatives_xy``.
    """
    dz_dx, dz_dy = _surface_sag_derivatives_xy(x, y, surface)
    # Normal = (-dz/dx, -dz/dy, 1), normalised
    mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)
    return -dz_dx / mag, -dz_dy / mag, 1.0 / mag


# ============================================================================
# Surface utility: clone with field overrides
# ============================================================================

def _surface_copy_with(surf, **overrides):
    """Return a new Surface with the given fields overridden.

    Propagates ALL optional fields -- the biconic block (``radius_y``,
    ``conic_y``, ``aspheric_coeffs_y``), ``freeform``, ``bsdf``,
    ``is_stop``, the coord-break block (``is_coordbrk``,
    ``tilt_x/y/z_deg``, ``decenter_x/y_m``, ``coordbrk_order``), and
    the world-frame fields (``world_origin``, ``world_R``) -- so
    anamorphic, freeform, stop-flagged, coord-break, and world-frame
    surfaces survive the clone.  This is a lightweight drop-in for
    ``dataclasses.replace`` that keeps the fallback
    ``getattr(..., <default>)`` for bundles unpickled from older
    library versions.

    v5.17.1 (audit P3-60): pre-fix the hand-rolled field list dropped
    the coord-break and world-frame blocks, so a cloned coord-break
    Surface silently became a regular flat refracting surface and a
    cloned world-frame surface lost its frame.
    """
    return Surface(
        radius=overrides.get('radius', surf.radius),
        conic=overrides.get('conic', surf.conic),
        aspheric_coeffs=overrides.get('aspheric_coeffs', surf.aspheric_coeffs),
        semi_diameter=overrides.get('semi_diameter', surf.semi_diameter),
        glass_before=overrides.get('glass_before', surf.glass_before),
        glass_after=overrides.get('glass_after', surf.glass_after),
        is_mirror=overrides.get('is_mirror', surf.is_mirror),
        is_stop=overrides.get('is_stop', getattr(surf, 'is_stop', False)),
        thickness=overrides.get('thickness', surf.thickness),
        label=overrides.get('label', surf.label),
        surf_num=overrides.get('surf_num', surf.surf_num),
        is_coordbrk=overrides.get('is_coordbrk',
                                    getattr(surf, 'is_coordbrk', False)),
        tilt_x_deg=overrides.get('tilt_x_deg',
                                   getattr(surf, 'tilt_x_deg', 0.0)),
        tilt_y_deg=overrides.get('tilt_y_deg',
                                   getattr(surf, 'tilt_y_deg', 0.0)),
        tilt_z_deg=overrides.get('tilt_z_deg',
                                   getattr(surf, 'tilt_z_deg', 0.0)),
        decenter_x_m=overrides.get('decenter_x_m',
                                     getattr(surf, 'decenter_x_m', 0.0)),
        decenter_y_m=overrides.get('decenter_y_m',
                                     getattr(surf, 'decenter_y_m', 0.0)),
        coordbrk_order=overrides.get(
            'coordbrk_order', getattr(surf, 'coordbrk_order', 0)),
        radius_y=overrides.get('radius_y',
                                 getattr(surf, 'radius_y', None)),
        conic_y=overrides.get('conic_y',
                                getattr(surf, 'conic_y', None)),
        aspheric_coeffs_y=overrides.get(
            'aspheric_coeffs_y',
            getattr(surf, 'aspheric_coeffs_y', None)),
        freeform=overrides.get('freeform',
                                 getattr(surf, 'freeform', None)),
        bsdf=overrides.get('bsdf', getattr(surf, 'bsdf', None)),
        coating=overrides.get('coating', getattr(surf, 'coating', None)),
        field_decenter=overrides.get(
            'field_decenter', getattr(surf, 'field_decenter', None)),
        field_tilt=overrides.get(
            'field_tilt', getattr(surf, 'field_tilt', None)),
        field_sag_callable=overrides.get(
            'field_sag_callable', getattr(surf, 'field_sag_callable', None)),
        world_origin=overrides.get(
            'world_origin', getattr(surf, 'world_origin', None)),
        world_R=overrides.get('world_R', getattr(surf, 'world_R', None)),
    )


__all__ = [
    # Error codes
    'RAY_OK', 'RAY_TIR', 'RAY_APERTURE',
    'RAY_MISSED_SURFACE', 'RAY_NAN', 'RAY_EVANESCENT',
    # Data classes
    'RayBundle', 'Surface', 'TraceResult',
    # Sag / normal helpers (private but exported for cross-submodule
    # use; the underscore prefix preserves the original "private"
    # signalling -- callers should not rely on these from user code).
    '_surface_sag_xy',
    '_surface_sag_derivatives_xy', '_surface_sag_derivative',
    '_surface_normal',
    # N10a field-frame decenter / tilt geometry (shared by traced + GBD)
    '_field_frame_active', '_field_frame_sag_and_grad',
    # Surface utility
    '_surface_copy_with',
]

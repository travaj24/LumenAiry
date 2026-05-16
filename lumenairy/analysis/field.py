"""
lumenairy.analysis.field -- field-resolved scriptable analyses.

This module collects the classical lens-design analyses that, prior to
4.4.0, lived only inside ``lumenairy/ui/*_dock.py`` (and were therefore
unreachable from a script or notebook without copy-pasting the dock
body), plus several primitives that were missing entirely.

Lifted-from-GUI analyses
------------------------

* :func:`distortion_vs_field` -- chief-ray f-tan(theta) distortion sweep.
* :func:`distortion_grid` -- 2-D distortion grid for visualization.
* :func:`footprint_per_surface` -- ray footprints at every surface,
  grouped by field angle.
* :func:`spot_diagram_vs_field` -- spot diagrams at multiple field
  angles on a common image plane.
* :func:`sensitivity_ranking` -- central-difference d(merit)/d(var)
  ranking of an arbitrary parameter vector.

New analyses
------------

* :func:`relative_illumination` -- geometric vignetting / transmission
  fraction vs field.
* :func:`field_aberration_sweep` -- real-ray sagittal / tangential
  focus shifts and astigmatism vs field (companion to the paraxial
  :func:`seidel_field_sweep`).
* :func:`petzval_radius` -- paraxial Petzval surface radius
  (1/R_p = sum (n2 - n1) / (n1 n2 R)).

Strehl alternatives (live in :mod:`lumenairy.analysis.analysis`,
re-exported at top level): :func:`strehl_marechal` and
:func:`strehl_phase_integral`.

All functions accept either a prescription dict or a pre-built list of
:class:`Surface` objects.  Prescriptions are the common case; the
surface-list path exists so the GUI docks (which build world-frame
surface lists with custom image-plane placement) can call these
public functions while preserving their fold-aware geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field, fields as _dc_fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class _DictAttrMixin:
    """Mixin that makes a ``@dataclass`` instance subscriptable.

    ``obj['name']`` returns ``getattr(obj, 'name')`` so callers written
    against the 4.4 dict-return API of :func:`distortion_grid` /
    :func:`footprint_per_surface` / :func:`spot_diagram_vs_field` /
    :func:`petzval_radius` keep working while new code can use
    attribute access.

    ``'k' in obj`` reports whether the dataclass has field ``k``.
    Iteration yields the field names (matches ``dict.__iter__``).
    """

    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(key) from e

    def __contains__(self, key: str) -> bool:
        return any(f.name == key for f in _dc_fields(self))

    def __iter__(self):
        return (f.name for f in _dc_fields(self))

    def keys(self):
        return [f.name for f in _dc_fields(self)]

    def values(self):
        return [getattr(self, f.name) for f in _dc_fields(self)]

    def items(self):
        return [(f.name, getattr(self, f.name)) for f in _dc_fields(self)]

    def get(self, key: str, default=None):
        return getattr(self, key, default)

from ..raytrace import (
    Surface,
    TraceResult,
    surfaces_from_prescription,
    system_abcd,
    make_rings,
    make_fan,
    spot_rms,
    spot_geo_radius,
)
from ..raytrace.core import trace_world, _make_bundle
from ..raytrace import trace


def _trace(rays, surfaces, wavelength):
    """Trace via ``trace_world`` if every surface carries a world
    frame, else fall back to the local-frame :func:`trace`.

    Lets the same public function transparently support both
    prescription-built surface lists (no world frames) and
    GUI-built world-frame surface lists (folded designs)."""
    if all(getattr(s, 'world_origin', None) is not None
           and getattr(s, 'world_R', None) is not None
           for s in surfaces):
        return trace_world(rays, surfaces, wavelength)
    return trace(rays, surfaces, wavelength)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _resolve_surfaces(system_or_prescription) -> List[Surface]:
    """Accept either a prescription dict or a pre-built surface list."""
    if isinstance(system_or_prescription, dict):
        return surfaces_from_prescription(system_or_prescription)
    return list(system_or_prescription)


def _append_image_plane(surfaces: List[Surface],
                         image_distance: float) -> List[Surface]:
    """Return a new surface list with a flat image plane appended
    ``image_distance`` past the last surface along its local +z axis.

    Works for both local-frame (legacy) and world-frame surface lists.
    In local-frame mode the previous-last surface's ``thickness`` is
    bumped to ``image_distance`` so :func:`trace` actually propagates
    the rays the requested distance to the new image plane.
    """
    surfs = list(surfaces)
    if not surfs:
        raise ValueError("Cannot append image plane to an empty surface list.")
    last = surfs[-1]
    if getattr(last, 'world_origin', None) is None or \
            getattr(last, 'world_R', None) is None:
        # Local-frame path: clone the last surface so we can set its
        # thickness to the image distance without mutating the caller's
        # surfaces.  Local-frame :func:`trace` advances rays from
        # surface i to surface i+1 using surfaces[i].thickness.
        from copy import copy as _copy
        last_clone = _copy(last)
        last_clone.thickness = float(image_distance)
        img = Surface(
            radius=float('inf'), conic=0.0,
            glass_before=last.glass_after, glass_after=last.glass_after,
            semi_diameter=float('inf'),
            label='Image',
        )
        return surfs[:-1] + [last_clone, img]
    # World-frame: place along last surface's local +z.
    z_hat = last.world_R[:, 2]
    new_origin = last.world_origin + z_hat * float(image_distance)
    img = Surface(
        radius=float('inf'), conic=0.0,
        glass_before=last.glass_after, glass_after=last.glass_after,
        semi_diameter=float('inf'),
        label='Image',
    )
    img.world_origin = new_origin
    img.world_R = last.world_R.copy()
    return surfs + [img]


def _semi_aperture_from(system, semi_aperture: Optional[float]) -> float:
    """Resolve semi_aperture from either an explicit value or
    ``prescription['aperture_diameter']``."""
    if semi_aperture is not None:
        return float(semi_aperture)
    if isinstance(system, dict):
        ap = system.get('aperture_diameter')
        if ap is not None:
            return float(ap) / 2.0
    raise ValueError(
        "semi_aperture is required (no aperture_diameter in prescription).")


# ----------------------------------------------------------------------
# Distortion vs field (lifted from ui/distortion_dock.py)
# ----------------------------------------------------------------------

@dataclass
class DistortionVsField:
    """Chief-ray distortion vs field-angle sweep result.

    Attributes
    ----------
    theta_deg : ndarray
        Field angles evaluated [deg].
    h_chief : ndarray
        Chief-ray image-plane height at each theta [m].  NaN where the
        trace failed or the ray was vignetted.
    h_paraxial : ndarray
        Paraxial f*tan(theta) reference [m].
    distortion_pct : ndarray
        ``100 * (h_chief - h_paraxial) / h_paraxial``.  Positive =
        pincushion, negative = barrel.  Defined zero at theta=0.
    efl : float
        Paraxial effective focal length [m].
    bfl : float
        Paraxial back focal length [m] (defines the image-plane
        position if ``image_distance`` was not explicitly supplied).
    max_distortion_pct : float
        ``max(|distortion_pct|)`` over finite entries.
    sign : str
        ``'pincushion'`` / ``'barrel'`` / ``'mixed'`` / ``'unknown'``.
    """
    theta_deg: np.ndarray
    h_chief: np.ndarray
    h_paraxial: np.ndarray
    distortion_pct: np.ndarray
    efl: float
    bfl: float
    max_distortion_pct: float
    sign: str


def distortion_vs_field(
    system: Union[Dict[str, Any], List[Surface]],
    wavelength: float,
    max_field_deg: float,
    *,
    n_points: int = 21,
    image_distance: Optional[float] = None,
) -> DistortionVsField:
    """Sweep chief-ray field angle and compute f-tan(theta) distortion.

    Parameters
    ----------
    system : dict or list of Surface
        Prescription dict, or a pre-built surface list that already
        includes the image plane.
    wavelength : float
        Wavelength [m].
    max_field_deg : float
        Maximum field angle [deg]; sweep covers ``[0, max_field_deg]``.
    n_points : int, default 21
        Number of field-angle samples.
    image_distance : float, optional
        Image-plane distance from the last lens vertex [m].  Defaults
        to the paraxial BFL.  Ignored when ``system`` is already a
        surface list with an image plane appended.

    Returns
    -------
    DistortionVsField
    """
    surfaces = _resolve_surfaces(system)
    _, efl, bfl, _ = system_abcd(surfaces, wavelength)
    if not (np.isfinite(efl) and efl > 0):
        raise ValueError(
            f"Need a converging system with finite positive EFL; got {efl}")

    # Append image plane if it isn't already there.  Heuristic: the
    # last surface is "an image plane" if it has infinite radius AND
    # glass_before == glass_after.
    if not (surfaces[-1].label == 'Image' or
            (not np.isfinite(surfaces[-1].radius)
             and surfaces[-1].glass_before == surfaces[-1].glass_after)):
        if image_distance is None:
            image_distance = bfl
        surfaces = _append_image_plane(surfaces, image_distance)

    thetas_deg = np.linspace(0.0, float(max_field_deg), int(n_points))
    thetas_rad = np.radians(thetas_deg)

    h_chief = np.full_like(thetas_rad, np.nan)
    for i, theta in enumerate(thetas_rad):
        try:
            rays = _make_bundle(
                x=np.array([0.0]), y=np.array([0.0]),
                L=np.array([0.0]), M=np.array([np.sin(theta)]),
                wavelength=wavelength,
            )
            result = _trace(rays, surfaces, wavelength)
            if result.image_rays.alive[0]:
                h_chief[i] = float(result.image_rays.y[0])
        except Exception:
            pass

    h_paraxial = efl * np.tan(thetas_rad)
    with np.errstate(divide='ignore', invalid='ignore'):
        distortion_pct = np.where(
            np.abs(h_paraxial) > 1e-15,
            100.0 * (h_chief - h_paraxial) / h_paraxial,
            0.0,
        )

    finite = np.isfinite(distortion_pct)
    if finite.any():
        max_d = float(np.nanmax(np.abs(distortion_pct)))
        d_max = float(np.nanmax(distortion_pct))
        d_min = float(np.nanmin(distortion_pct))
        if d_max > 0 and d_min < 0:
            sign = 'mixed'
        elif d_max > 0:
            sign = 'pincushion'
        elif d_min < 0:
            sign = 'barrel'
        else:
            sign = 'unknown'
    else:
        max_d = float('nan')
        sign = 'unknown'

    return DistortionVsField(
        theta_deg=thetas_deg,
        h_chief=h_chief,
        h_paraxial=h_paraxial,
        distortion_pct=distortion_pct,
        efl=float(efl),
        bfl=float(bfl),
        max_distortion_pct=max_d,
        sign=sign,
    )


@dataclass
class DistortionGrid(_DictAttrMixin):
    """2-D chief-ray distortion-grid result.

    Returned by :func:`distortion_grid`.  Subscriptable for back-compat
    with the 4.4 dict-return API (``result['actual_x']`` works the
    same as ``result.actual_x``).

    Attributes
    ----------
    theta_x_deg, theta_y_deg : ndarray
        1-D axis angles [deg], length ``n_grid``.
    paraxial_x, paraxial_y : ndarray
        Paraxial f-tan(theta) reference image positions [m],
        shape ``(n_grid, n_grid)``.
    actual_x, actual_y : ndarray
        Traced chief-ray image positions [m], same shape.  NaN where
        the chief ray was vignetted.
    efl, bfl : float
        Paraxial effective focal length and back focal length [m].
    """
    theta_x_deg: np.ndarray
    theta_y_deg: np.ndarray
    paraxial_x: np.ndarray
    paraxial_y: np.ndarray
    actual_x: np.ndarray
    actual_y: np.ndarray
    efl: float
    bfl: float


def distortion_grid(
    system: Union[Dict[str, Any], List[Surface]],
    wavelength: float,
    max_field_deg: float,
    *,
    n_grid: int = 7,
    image_distance: Optional[float] = None,
) -> DistortionGrid:
    """2-D chief-ray distortion grid.

    For every (theta_x, theta_y) on an n_grid x n_grid mesh covering
    ``[-max_field_deg, +max_field_deg]`` in each axis, trace the chief
    ray and report its image-plane (x, y) position alongside the
    paraxial f-tan(theta) reference.

    Parameters
    ----------
    system : dict or list of Surface
    wavelength : float
    max_field_deg : float
        Half-width of the angular sweep [deg].
    n_grid : int, default 7
        Samples per axis (total n_grid**2 chief rays).
    image_distance : float, optional
        Image-plane distance from last vertex [m].  Defaults to BFL.

    Returns
    -------
    :class:`DistortionGrid`
        Dataclass with ``theta_x_deg``, ``theta_y_deg``,
        ``paraxial_x``, ``paraxial_y``, ``actual_x``, ``actual_y``,
        ``efl``, ``bfl`` attributes.  Subscriptable
        (``result['actual_x']``) for back-compat with the 4.4
        dict-return API.
    """
    surfaces = _resolve_surfaces(system)
    _, efl, bfl, _ = system_abcd(surfaces, wavelength)
    if not (surfaces[-1].label == 'Image' or
            (not np.isfinite(surfaces[-1].radius)
             and surfaces[-1].glass_before == surfaces[-1].glass_after)):
        if image_distance is None:
            image_distance = bfl
        surfaces = _append_image_plane(surfaces, image_distance)

    ths = np.linspace(-max_field_deg, max_field_deg, int(n_grid))
    ths_rad = np.radians(ths)

    actual_x = np.full((n_grid, n_grid), np.nan)
    actual_y = np.full((n_grid, n_grid), np.nan)
    for ix, tx in enumerate(ths_rad):
        for iy, ty in enumerate(ths_rad):
            try:
                rays = _make_bundle(
                    x=np.array([0.0]), y=np.array([0.0]),
                    L=np.array([np.sin(tx)]),
                    M=np.array([np.sin(ty)]),
                    wavelength=wavelength,
                )
                r = _trace(rays, surfaces, wavelength)
                if r.image_rays.alive[0]:
                    actual_x[iy, ix] = float(r.image_rays.x[0])
                    actual_y[iy, ix] = float(r.image_rays.y[0])
            except Exception:
                pass

    para_x = (efl * np.tan(ths_rad))[None, :].repeat(n_grid, axis=0)
    para_y = (efl * np.tan(ths_rad))[:, None].repeat(n_grid, axis=1)

    return DistortionGrid(
        theta_x_deg=ths,
        theta_y_deg=ths,
        paraxial_x=para_x,
        paraxial_y=para_y,
        actual_x=actual_x,
        actual_y=actual_y,
        efl=float(efl),
        bfl=float(bfl),
    )


# ----------------------------------------------------------------------
# Footprint per surface (lifted from ui/footprint_dock.py)
# ----------------------------------------------------------------------

@dataclass
class FieldFootprint(_DictAttrMixin):
    """Ray footprint at a single surface for a single field angle."""
    field_deg: float
    x: np.ndarray
    y: np.ndarray
    alive: np.ndarray


@dataclass
class SurfaceFootprint(_DictAttrMixin):
    """Per-surface ray footprint aggregated across all field angles."""
    surface_index: int
    semi_diameter: float
    label: str
    fields: List[FieldFootprint]


def footprint_per_surface(
    system: Union[Dict[str, Any], List[Surface]],
    wavelength: float,
    *,
    semi_aperture: Optional[float] = None,
    fields_deg: Sequence[float] = (0.0,),
    num_rings: int = 6,
    rays_per_ring: int = 12,
    image_distance: Optional[float] = None,
) -> List[SurfaceFootprint]:
    """Per-surface ray footprints over one or more field angles.

    For each surface in the system, return the (x, y) coordinates of
    every alive ray at that surface (in the surface's local frame),
    grouped by field angle.  This is the script-accessible form of
    the GUI's Footprint dock.

    Parameters
    ----------
    system : dict or list of Surface
    wavelength : float
    semi_aperture : float, optional
        Entrance-pupil semi-diameter [m].  Defaults to half of
        ``prescription['aperture_diameter']``.
    fields_deg : sequence of float, default (0.0,)
    num_rings, rays_per_ring : int
        Concentric-ring sampling (matches the standard spot-diagram
        convention).
    image_distance : float, optional
        Image-plane distance from last vertex [m].  Defaults to BFL.

    Returns
    -------
    list of :class:`SurfaceFootprint`
        One entry per surface (last entry = image plane if one was
        appended).  Each :class:`SurfaceFootprint` is also subscriptable
        (``entry['fields'][0]['x']``) for back-compat with the 4.4
        dict-return API.
    """
    surfaces = _resolve_surfaces(system)
    semi_ap = _semi_aperture_from(system, semi_aperture)
    if not (surfaces[-1].label == 'Image' or
            (not np.isfinite(surfaces[-1].radius)
             and surfaces[-1].glass_before == surfaces[-1].glass_after)):
        if image_distance is None:
            try:
                _, _, bfl, _ = system_abcd(surfaces, wavelength)
            except Exception:
                bfl = None
            if bfl is not None and np.isfinite(bfl) and bfl > 0:
                image_distance = bfl
        if image_distance is not None:
            surfaces = _append_image_plane(surfaces, image_distance)

    out: List[SurfaceFootprint] = []
    for si, surf in enumerate(surfaces):
        out.append(SurfaceFootprint(
            surface_index=si,
            semi_diameter=float(surf.semi_diameter)
                if np.isfinite(surf.semi_diameter) else float('nan'),
            label=surf.label or f'S{si + 1}',
            fields=[],
        ))

    for fa_deg in fields_deg:
        fa_rad = float(np.radians(fa_deg))
        try:
            rays = make_rings(semi_ap, num_rings, rays_per_ring,
                                fa_rad, wavelength)
            result = _trace(rays, surfaces, wavelength)
        except Exception:
            for entry in out:
                entry.fields.append(FieldFootprint(
                    field_deg=fa_deg,
                    x=np.array([]), y=np.array([]),
                    alive=np.array([], dtype=bool),
                ))
            continue
        for si, rb in enumerate(result.ray_history):
            if si >= len(out):
                break
            out[si].fields.append(FieldFootprint(
                field_deg=fa_deg,
                x=np.asarray(rb.x),
                y=np.asarray(rb.y),
                alive=np.asarray(rb.alive, dtype=bool),
            ))
    return out


# ----------------------------------------------------------------------
# Spot diagrams vs field (lifted from ui/spot_field_dock.py)
# ----------------------------------------------------------------------

@dataclass
class SpotDiagramField(_DictAttrMixin):
    """Spot-diagram statistics for a single field angle."""
    field_deg: float
    result: Any
    rms_radius: float
    geo_radius: float
    centroid: Tuple[float, float]


def spot_diagram_vs_field(
    system: Union[Dict[str, Any], List[Surface]],
    wavelength: float,
    fields_deg: Sequence[float],
    *,
    semi_aperture: Optional[float] = None,
    num_rings: int = 6,
    rays_per_ring: int = 12,
    image_distance: Optional[float] = None,
) -> List[SpotDiagramField]:
    """Trace spot diagrams at multiple field angles on a common image
    plane.

    Parameters
    ----------
    system : dict or list of Surface
    wavelength : float
    fields_deg : sequence of float
        Field angles to evaluate [deg].
    semi_aperture : float, optional
    num_rings, rays_per_ring : int
        Ring-sampling parameters.
    image_distance : float, optional
        Image-plane distance from last vertex [m].  Defaults to BFL.

    Returns
    -------
    list of :class:`SpotDiagramField`
        One :class:`SpotDiagramField` per field angle.  Each is
        subscriptable (``entry['rms_radius']``) for back-compat with
        the 4.4 dict-return API.
    """
    surfaces = _resolve_surfaces(system)
    semi_ap = _semi_aperture_from(system, semi_aperture)
    _, _, bfl, _ = system_abcd(surfaces, wavelength)
    if not (surfaces[-1].label == 'Image' or
            (not np.isfinite(surfaces[-1].radius)
             and surfaces[-1].glass_before == surfaces[-1].glass_after)):
        if image_distance is None:
            image_distance = bfl
        surfaces = _append_image_plane(surfaces, image_distance)

    out = []
    for fa_deg in fields_deg:
        fa_rad = float(np.radians(fa_deg))
        rays = make_rings(semi_ap, num_rings, rays_per_ring,
                            fa_rad, wavelength)
        result = _trace(rays, surfaces, wavelength)
        ir = result.image_rays
        alive = ir.alive
        if alive.any():
            cx = float(np.mean(ir.x[alive]))
            cy = float(np.mean(ir.y[alive]))
            try:
                rms, _ = spot_rms(result)
                rms = float(rms)
            except Exception:
                rms = float('nan')
            try:
                geo, _ = spot_geo_radius(result)
                geo = float(geo)
            except Exception:
                geo = float('nan')
        else:
            cx = cy = float('nan')
            rms = geo = float('nan')
        out.append(SpotDiagramField(
            field_deg=fa_deg,
            result=result,
            rms_radius=rms,
            geo_radius=geo,
            centroid=(cx, cy),
        ))
    return out


# ----------------------------------------------------------------------
# Relative illumination (new)
# ----------------------------------------------------------------------

@dataclass
class RelativeIllumination:
    """Geometric relative-illumination vs field-angle result.

    Attributes
    ----------
    fields_deg : ndarray
    n_launched : ndarray of int
        Rays launched at each field.
    n_transmitted : ndarray of int
        Rays still alive at the image plane.
    transmission : ndarray
        ``n_transmitted / n_launched`` per field.
    relative_illumination : ndarray
        ``transmission / transmission[axis]``.  1.0 on-axis by
        construction; falls toward 0 at the vignetting edge.
    """
    fields_deg: np.ndarray
    n_launched: np.ndarray
    n_transmitted: np.ndarray
    transmission: np.ndarray
    relative_illumination: np.ndarray


def relative_illumination(
    system: Union[Dict[str, Any], List[Surface]],
    wavelength: float,
    fields_deg: Sequence[float],
    *,
    semi_aperture: Optional[float] = None,
    num_rings: int = 12,
    rays_per_ring: int = 24,
    image_distance: Optional[float] = None,
) -> RelativeIllumination:
    """Geometric relative-illumination (vignetting) vs field angle.

    Launches a dense ring grid at the entrance pupil for each field
    angle, traces through the system, and counts how many rays survive
    to the image plane.  Returns the survival fraction normalised by
    the on-axis transmission.

    This is a **pure-geometric** vignetting analysis -- it does NOT
    include cos^4(theta) radiometric falloff, which a wave-optics or
    radiometry pipeline would compose with this result.

    Parameters
    ----------
    system : dict or list of Surface
    wavelength : float
    fields_deg : sequence of float
        Field angles to evaluate [deg].  Should include 0.0 for the
        normalisation; if missing, it is prepended.
    semi_aperture, num_rings, rays_per_ring : see other functions.
    image_distance : float, optional
        Image-plane distance from last vertex [m].

    Returns
    -------
    RelativeIllumination
    """
    surfaces = _resolve_surfaces(system)
    semi_ap = _semi_aperture_from(system, semi_aperture)
    _, _, bfl, _ = system_abcd(surfaces, wavelength)
    if not (surfaces[-1].label == 'Image' or
            (not np.isfinite(surfaces[-1].radius)
             and surfaces[-1].glass_before == surfaces[-1].glass_after)):
        if image_distance is None:
            image_distance = bfl
        surfaces = _append_image_plane(surfaces, image_distance)

    fields_deg = np.asarray(fields_deg, dtype=float)
    if not np.any(fields_deg == 0.0):
        fields_deg = np.concatenate([[0.0], fields_deg])

    n_launched = np.zeros_like(fields_deg, dtype=int)
    n_alive = np.zeros_like(fields_deg, dtype=int)
    for i, fa_deg in enumerate(fields_deg):
        fa_rad = float(np.radians(fa_deg))
        rays = make_rings(semi_ap, num_rings, rays_per_ring,
                            fa_rad, wavelength)
        result = _trace(rays, surfaces, wavelength)
        n_launched[i] = rays.alive.size
        n_alive[i] = int(np.count_nonzero(result.image_rays.alive))

    with np.errstate(divide='ignore', invalid='ignore'):
        transmission = n_alive / np.maximum(n_launched, 1)
    on_axis = transmission[np.argmin(np.abs(fields_deg))]
    if on_axis <= 0:
        rel = np.full_like(transmission, np.nan)
    else:
        rel = transmission / on_axis

    return RelativeIllumination(
        fields_deg=fields_deg,
        n_launched=n_launched,
        n_transmitted=n_alive,
        transmission=transmission,
        relative_illumination=rel,
    )


# ----------------------------------------------------------------------
# Field aberration sweep (real-ray sag/tan focus shifts)
# ----------------------------------------------------------------------

@dataclass
class FieldAberrationSweep:
    """Sagittal / tangential focus and astigmatism vs field angle.

    Attributes
    ----------
    fields_deg : ndarray
    sagittal_focus_shift : ndarray
        Axial focus offset of the sagittal fan from the paraxial image
        plane, per field [m].
    tangential_focus_shift : ndarray
        Same for the tangential fan [m].
    astigmatism : ndarray
        ``sagittal_focus_shift - tangential_focus_shift`` per field [m].
    petzval_radius : float
        Paraxial Petzval surface radius [m].
    """
    fields_deg: np.ndarray
    sagittal_focus_shift: np.ndarray
    tangential_focus_shift: np.ndarray
    astigmatism: np.ndarray
    petzval_radius: float


def _propagate_rms(result: TraceResult, z_to: float) -> float:
    """RMS spot radius of the exit rays propagated by free space to
    axial position ``z_to`` measured from the last lens vertex."""
    rays = result.image_rays
    alive = rays.alive
    if not alive.any():
        return float('nan')
    Lhat = rays.L[alive]
    Mhat = rays.M[alive]
    Nhat = rays.N[alive]
    x0 = rays.x[alive]
    y0 = rays.y[alive]
    z0 = rays.z[alive]
    nonzero = Nhat != 0
    if not nonzero.any():
        return float('nan')
    Lhat = Lhat[nonzero]
    Mhat = Mhat[nonzero]
    Nhat = Nhat[nonzero]
    x0 = x0[nonzero]
    y0 = y0[nonzero]
    z0 = z0[nonzero]
    dz = z_to - z0
    xz = x0 + dz * Lhat / Nhat
    yz = y0 + dz * Mhat / Nhat
    if xz.size == 0:
        return float('nan')
    cx = xz.mean()
    cy = yz.mean()
    return float(np.sqrt(np.mean((xz - cx) ** 2 + (yz - cy) ** 2)))


def field_aberration_sweep(
    system: Union[Dict[str, Any], List[Surface]],
    wavelength: float,
    fields_deg: Sequence[float],
    *,
    semi_aperture: Optional[float] = None,
    dz_search: Optional[float] = None,
    n_z: int = 51,
    n_fan: int = 15,
) -> FieldAberrationSweep:
    """Real-ray sagittal / tangential focus shifts and astigmatism vs field.

    For each field angle:

    1. Trace a tangential (y-axis) fan and a sagittal (x-axis) fan
       through the system.
    2. Sweep the image plane axially around the paraxial BFL and find
       the z that minimises the RMS spread of each fan separately.
    3. Report the focus offsets and the sagittal-tangential astigmatism.

    This is the **real-ray** companion to :func:`seidel_field_sweep`,
    which uses paraxial Hopkins sums (S3, S4) to predict the same
    quantities.  At small field angles and well-corrected systems the
    two should agree; differences grow as higher-order aberrations
    enter.

    Parameters
    ----------
    system : dict or list of Surface
    wavelength : float
    fields_deg : sequence of float
    semi_aperture : float, optional
    dz_search : float, optional
        Half-width of axial search [m].  Defaults to ``|BFL| / 20``.
    n_z : int, default 51
        Axial samples per focus search.
    n_fan : int, default 15
        Rays per fan (odd preferred so the chief ray is included).

    Returns
    -------
    FieldAberrationSweep

    See Also
    --------
    seidel_field_sweep : paraxial third-order Hopkins-sum version.
    petzval_radius : the Petzval surface radius reported here.
    """
    surfaces = _resolve_surfaces(system)
    semi_ap = _semi_aperture_from(system, semi_aperture)
    _, efl, bfl, _ = system_abcd(surfaces, wavelength)
    if dz_search is None:
        dz_search = abs(bfl) / 20.0 if np.isfinite(bfl) else 1e-3

    fields_deg = np.asarray(fields_deg, dtype=float)
    sag_shift = np.full_like(fields_deg, np.nan)
    tan_shift = np.full_like(fields_deg, np.nan)

    z_offsets = np.linspace(-dz_search, dz_search, int(n_z))

    for i, fa_deg in enumerate(fields_deg):
        fa_rad = float(np.radians(fa_deg))
        # 4.10: build sagittal/tangential fans at a +y field.  Pre-4.10
        # this used make_fan(axis='x', field_angle=fa) for "sag" and
        # make_fan(axis='y', field_angle=fa) for "tan" -- but make_fan
        # tilts the chief ray ALONG the fan axis, so both calls produced
        # tangential fans at two *different* fields (x and y),
        # reporting astigmatism between unrelated configurations.
        # Correct convention: chief tilted in +y; sagittal fan spreads
        # rays along x (perpendicular to meridional plane); tangential
        # fan spreads along y (in the meridional plane).
        sin_fa = float(np.sin(fa_rad))
        t = np.linspace(-1.0, 1.0, n_fan)
        sag_rays = _make_bundle(
            x=t * semi_ap,
            y=np.zeros(n_fan),
            L=np.zeros(n_fan),
            M=np.full(n_fan, sin_fa),
            wavelength=wavelength,
        )
        tan_rays = _make_bundle(
            x=np.zeros(n_fan),
            y=t * semi_ap,
            L=np.zeros(n_fan),
            M=np.full(n_fan, sin_fa),
            wavelength=wavelength,
        )
        try:
            sag_res = _trace(sag_rays, surfaces, wavelength)
            tan_res = _trace(tan_rays, surfaces, wavelength)
        except Exception:
            continue

        sag_rms = np.array([_propagate_rms(sag_res, bfl + dz)
                              for dz in z_offsets])
        tan_rms = np.array([_propagate_rms(tan_res, bfl + dz)
                              for dz in z_offsets])
        if np.isfinite(sag_rms).any():
            sag_shift[i] = float(z_offsets[np.nanargmin(sag_rms)])
        if np.isfinite(tan_rms).any():
            tan_shift[i] = float(z_offsets[np.nanargmin(tan_rms)])

    astig = sag_shift - tan_shift
    pr = petzval_radius(surfaces, wavelength)

    return FieldAberrationSweep(
        fields_deg=fields_deg,
        sagittal_focus_shift=sag_shift,
        tangential_focus_shift=tan_shift,
        astigmatism=astig,
        petzval_radius=pr,
    )


# ----------------------------------------------------------------------
# Petzval radius (new)
# ----------------------------------------------------------------------

def petzval_radius(
    system: Union[Dict[str, Any], List[Surface]],
    wavelength: float,
) -> float:
    """Paraxial Petzval surface radius [m].

    Computes ``1 / R_p = sum_i (n2_i - n1_i) / (n1_i n2_i R_i)`` over
    refracting surfaces.  Returns ``+inf`` if the sum is exactly zero
    (a Petzval-flat system) and ``nan`` if any glass index can't be
    resolved.

    Parameters
    ----------
    system : dict or list of Surface
    wavelength : float [m]

    Returns
    -------
    R_p : float
        Petzval radius [m].  Positive = curving toward the object
        side; negative = away.
    """
    from ..glass import get_glass_index
    surfaces = _resolve_surfaces(system)
    inv_R = 0.0
    for s in surfaces:
        R = s.radius
        if not np.isfinite(R) or R == 0:
            continue
        try:
            n1 = float(get_glass_index(s.glass_before, wavelength))
            n2 = float(get_glass_index(s.glass_after, wavelength))
        except Exception:
            return float('nan')
        if n1 == n2:
            continue
        inv_R += (n2 - n1) / (n1 * n2 * R)
    if inv_R == 0:
        return float('inf')
    # 4.10: Born & Wolf §4.4 gives the Petzval surface radius as
    #   1/R_p = -Σ (n2 - n1) / (n1 n2 R)
    # with the leading minus sign by convention (Petzval surface
    # curves AWAY from the lens for positive-power systems).  Pre-4.10
    # returned +1/inv_R, flipping the sign.
    return -1.0 / inv_R


# ----------------------------------------------------------------------
# Sensitivity ranking (lifted from ui/sensitivity_dock.py)
# ----------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Finite-difference sensitivity of a scalar metric to each
    variable in a parameter vector.

    Attributes
    ----------
    labels : list of str
    values : ndarray
        The parameter values at which the sensitivities were taken.
    derivative : ndarray
        ``d(metric)/d(var)``.
    relative_sensitivity : ndarray
        ``|d * value / f0|``, a dimensionless rank metric.
    base_value : float
        ``f(x0)``.
    order : ndarray of int
        Indices that sort variables from most to least sensitive.
    """
    labels: list
    values: np.ndarray
    derivative: np.ndarray
    relative_sensitivity: np.ndarray
    base_value: float
    order: np.ndarray


def sensitivity_ranking(
    merit_fn: Callable[[np.ndarray], float],
    x0: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    *,
    eps_rel: float = 1e-3,
    eps_abs_floor: float = 1e-9,
) -> SensitivityResult:
    """Central-difference sensitivity of a scalar merit to each
    variable.

    Lifted-from-GUI helper: the same algorithm previously buried in
    ``ui/sensitivity_dock.SensitivityDock._run_ranking`` is now
    callable from any script.

    Parameters
    ----------
    merit_fn : callable(x) -> float
        Scalar metric to differentiate.  Should be deterministic
        (callable side effects are tolerated but discouraged).
    x0 : sequence of float
        Operating point, length N.
    labels : sequence of str, optional
        Names for each variable.  Defaults to ``['x0', 'x1', ...]``.
    eps_rel : float, default 1e-3
        Relative step: ``step = |x_i| * eps_rel`` (with absolute
        floor).
    eps_abs_floor : float, default 1e-9
        Minimum absolute step when ``|x_i|`` is near zero.

    Returns
    -------
    SensitivityResult
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    if labels is None:
        labels = [f'x{i}' for i in range(n)]
    else:
        labels = list(labels)

    f0 = float(merit_fn(x0))
    deriv = np.full(n, np.nan)
    for i in range(n):
        v = x0[i]
        step = abs(v) * eps_rel if v != 0 else eps_abs_floor
        if step < eps_abs_floor:
            step = eps_abs_floor
        xp = x0.copy(); xp[i] = v + step
        xm = x0.copy(); xm[i] = v - step
        try:
            fp = float(merit_fn(xp))
            fm = float(merit_fn(xm))
        except Exception:
            continue
        if np.isfinite(fp) and np.isfinite(fm):
            deriv[i] = (fp - fm) / (2 * step)

    if abs(f0) > 1e-30:
        rel = np.abs(deriv * x0 / f0)
    else:
        rel = np.abs(deriv * x0)

    finite = np.isfinite(rel)
    order = np.argsort(-np.where(finite, rel, -np.inf))

    return SensitivityResult(
        labels=labels,
        values=x0,
        derivative=deriv,
        relative_sensitivity=rel,
        base_value=f0,
        order=order,
    )


__all__ = [
    # Distortion
    'DistortionVsField', 'distortion_vs_field', 'distortion_grid',
    # Footprint
    'footprint_per_surface',
    # Spot
    'spot_diagram_vs_field',
    # Vignetting
    'RelativeIllumination', 'relative_illumination',
    # Field-curvature / astigmatism (real-ray)
    'FieldAberrationSweep', 'field_aberration_sweep',
    # Petzval
    'petzval_radius',
    # Sensitivity
    'SensitivityResult', 'sensitivity_ranking',
]

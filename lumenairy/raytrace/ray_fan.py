"""
v5.1.0 split: ray-fan / spot / OPD / through-focus analytics + plots.

Extracted from ``lumenairy/raytrace/core.py`` as part of the v5.1.0
6-file split (ROADMAP Agent B).  Hosts the image-plane analytics that
rebuild a final ray bundle and report aberrations:

* :func:`spot_rms`, :func:`spot_geo_radius`, :func:`spot_diagram` --
  scalar spot metrics + matplotlib spot diagram.
* :func:`ray_fan_data`, :func:`ray_fan_data_world`,
  :func:`ray_fan_plot`, :func:`ray_fan_plot_prescription` -- transverse
  ray-aberration fans (tangential + sagittal).
* :func:`opd_fan_data`, :func:`opd_fan_data_world` -- wavefront error
  vs pupil coordinate.
* :func:`refocus`, :func:`through_focus_rms` -- closed-form image-plane
  refocus + best-focus search.

Every public name here is re-exported from
``lumenairy.raytrace.core`` so existing imports continue to resolve.

No physics change: contents are bit-for-bit copies of the original
implementations.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..glass import get_glass_index
from .seidel import first_order_data, system_abcd
from .surface import Surface, TraceResult
from .trace import (
    make_fan,
    make_ray,
    make_rings,
    surfaces_from_prescription,
    trace,
)
from .world_trace import trace_world


# ============================================================================
# Shared helper: entrance-pupil-centring launch offset
# ============================================================================

def _ep_offset(ep_z: float, field_angle: float) -> float:
    """Launch-height offset that puts a ``z = 0``-launched chief / fan ray
    through the entrance-pupil centre at ``z = ep_z``.

    ``-ep_z * tan(field_angle)`` -- but S11-6c
    (AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1) found that raw expression
    produces ``NaN`` for an OBJECT-SPACE TELECENTRIC system, where
    ``compute_pupils`` legitimately returns ``ep_z = inf`` (the stop sits
    at the pre-stop group's rear focal plane, so ``A_pre = 0``): on-axis
    that is ``inf * tan(0) = inf * 0 = NaN``, and the NaN then propagated
    into every launched ray height, so ``ray_fan_data`` /
    ``opd_fan_data`` returned all-NaN fans with no diagnostic.  An
    entrance pupil at infinity has no FINITE centring offset at any
    field, so fall back to the legacy origin-launched convention
    (``ep_off = 0``) -- exactly what the callers' ``except`` branches
    already do for every other pupil failure.

    Bit-identical whenever ``ep_z`` is finite, which is every
    non-telecentric system: the arithmetic is untouched there.
    """
    if not np.isfinite(ep_z):
        return 0.0
    return -ep_z * np.tan(field_angle)


# ============================================================================
# Analysis: spot diagram
# ============================================================================

def spot_rms(result: 'TraceResult') -> Tuple[float, Tuple[float, float]]:
    """Compute RMS spot radius from a trace result.

    Parameters
    ----------
    result : TraceResult

    Returns
    -------
    rms : float
        RMS spot radius [m] at the final surface.
    centroid : tuple (cx, cy)
        Spot centroid [m].
    """
    r = result.image_rays
    alive = r.alive
    if not np.any(alive):
        return np.inf, (0.0, 0.0)

    cx = np.mean(r.x[alive])
    cy = np.mean(r.y[alive])

    dx = r.x[alive] - cx
    dy = r.y[alive] - cy
    rms = np.sqrt(np.mean(dx ** 2 + dy ** 2))

    return rms, (cx, cy)


def spot_geo_radius(result: 'TraceResult') -> float:
    """Compute the geometric (maximum) spot radius.

    Parameters
    ----------
    result : TraceResult

    Returns
    -------
    geo_radius : float
        Maximum distance from centroid [m].
    """
    r = result.image_rays
    alive = r.alive
    if not np.any(alive):
        return np.inf

    cx = np.mean(r.x[alive])
    cy = np.mean(r.y[alive])
    dist = np.sqrt((r.x[alive] - cx) ** 2 + (r.y[alive] - cy) ** 2)
    return np.max(dist)


def spot_diagram(
    result: 'TraceResult',
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    units: str = 'um',
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Plot a spot diagram from a trace result.

    Parameters
    ----------
    result : TraceResult
    ax : matplotlib Axes or None
        If None, creates a new figure.
    title : str or None
    units : str
        ``'um'`` (micrometres) or ``'mm'`` (millimetres).
    **kwargs
        Passed to ``ax.scatter()``.

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    scale = {'um': 1e6, 'mm': 1e3, 'm': 1.0}[units]
    label = {'um': 'µm', 'mm': 'mm', 'm': 'm'}[units]

    r = result.image_rays
    alive = r.alive

    rms, (cx, cy) = spot_rms(result)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure

    scatter_kw = dict(s=4, alpha=0.6, edgecolors='none')
    scatter_kw.update(kwargs)

    ax.scatter((r.x[alive] - cx) * scale,
               (r.y[alive] - cy) * scale,
               **scatter_kw)

    ax.set_xlabel(f'x [{label}]')
    ax.set_ylabel(f'y [{label}]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if title is None:
        n_alive = np.sum(alive)
        title = (f'Spot Diagram  ({n_alive}/{r.n_rays} rays)\n'
                 f'RMS = {rms * scale:.3f} {label},  '
                 f'GEO = {spot_geo_radius(result) * scale:.3f} {label}')
    ax.set_title(title)

    # Draw Airy disc for reference.
    # 4.11.2: include the f_eff factor so the Airy disc is in image-
    # plane metres, not radians.  Pre-4.11.2 the formula was
    # ``1.22 * wavelength / (2 * semi_diameter)`` (a half-angle), which
    # for a typical f/4 100mm-EFL singlet under-reported the Airy
    # radius by ~25x relative to the spot-diagram axis (also in
    # image-plane metres).
    sd0 = result.surfaces[0].semi_diameter
    if np.isfinite(sd0):
        try:
            _, _f_eff, _, _ = system_abcd(result.surfaces, result.wavelength)
        except (ValueError, RuntimeError, ZeroDivisionError,
                np.linalg.LinAlgError, IndexError):
            # system_abcd can raise on degenerate prescriptions: a
            # mirror-only system has no usable paraxial focus, an
            # ill-conditioned ABCD product yields ZeroDivision/LinAlg
            # failures, and short prescriptions trip IndexError.
            _f_eff = float('nan')
        if np.isfinite(_f_eff):
            airy_r = 1.22 * result.wavelength * abs(_f_eff) / (2.0 * sd0)
        else:
            # Afocal/degenerate: the diffraction limit is a half-ANGLE
            # (radians); drawing it on a metre-scaled spot axis would be
            # meaningless, so skip the Airy circle entirely.
            airy_r = None
    else:
        airy_r = None
    if airy_r is not None and airy_r * scale < ax.get_xlim()[1] * 5:
        circle = plt.Circle((0, 0), airy_r * scale,
                             fill=False, color='red', linestyle='--',
                             linewidth=0.8, label=f'Airy ({airy_r*scale:.3f} {label})')
        ax.add_patch(circle)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig, ax


# ============================================================================
# Analysis: ray fan (transverse aberration) plots
# ============================================================================

def ray_fan_data(
    surfaces: List['Surface'],
    wavelength: float,
    semi_aperture: float,
    field_angle: float = 0.0,
    n_rays: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute transverse ray aberration vs normalised pupil coordinate.

    Parameters
    ----------
    surfaces : list of Surface
    wavelength : float
    semi_aperture : float
    field_angle : float
    n_rays : int

    Returns
    -------
    py : ndarray
        Normalised pupil coordinate in Y (tangential fan).
    ey : ndarray
        Transverse ray error in Y [m] (tangential).
    px : ndarray
        Normalised pupil coordinate in X (sagittal fan).
    ex : ndarray
        Transverse ray error in X [m] (sagittal).
    """
    # Reference: chief ray.  4.11.2 (audit H-AB-3 sibling): for off-
    # axis fields the chief is launched at angle from the EP centre,
    # not from (0,0,0).  Pre-4.11.2 the chief started at z=0 from the
    # axis, so ``y_ref / x_ref`` did not match the canonical "chief
    # passes through the EP centre" reference and the fan plots were
    # offset by O(field_angle * z_EP).  Mirrors the v4.10 fix in
    # ``eval_image_plane_wfe`` and v4.11.2-Track-F in
    # ``relative_illumination`` / ``field_aberration_sweep``.
    # RT-5 (AUDIT_RAYTRACE_CORE): EP-centre the FANS too, not just the
    # chief.  ``make_fan`` launches each fan at z=0 with a field tilt along
    # its OWN axis (y-fan: M=sin(fa); x-fan: L=sin(fa)), so for an off-axis
    # field every fan ray crosses the entrance pupil displaced by
    # ``ep_z*tan(fa)``.  The 4.11.2 fix moved only the reference chief to the
    # EP centre, so chief and fan then sampled DIFFERENT pupil zones and the
    # fan no longer passed through zero at py=0 (``ey(0)`` read the launch-
    # convention offset instead of 0).  We shift each fan's LAUNCH heights
    # by the same ``ep_off = -ep_z*tan(fa)`` used for the chief (so the fan
    # is centred on the chief's pupil crossing), and reference each fan
    # against a chief of the SAME orientation so ``ey(0) == ex(0) == 0``.
    try:
        fod = first_order_data(surfaces, wavelength)
        ep_off = _ep_offset(fod.ep_z, field_angle)
        # make_ray(x, y, L, M, *, wavelength): tangential chief tilts in M,
        # sagittal chief tilts in L; each launches at z=0 with the ep_off
        # offset along its axis so it crosses the EP centre at z=ep_z.
        chief_y = make_ray(0.0, ep_off, 0.0, np.sin(field_angle),
                           wavelength=wavelength)
        chief_x = make_ray(ep_off, 0.0, np.sin(field_angle), 0.0,
                           wavelength=wavelength)
    except (ValueError, RuntimeError, ZeroDivisionError, AttributeError,
            np.linalg.LinAlgError, IndexError):
        # No first-order pupil available (e.g. mirror-only stop-less
        # system) -- first_order_data raises ValueError on missing
        # stop, AttributeError on a stripped Surface dataclass,
        # ZeroDivisionError / LinAlgError on ill-conditioned ABCD.
        # Fall back to legacy origin-launched chiefs (ep_off = 0).
        ep_off = 0.0
        chief_y = make_ray(0, 0, 0, np.sin(field_angle),
                           wavelength=wavelength)
        chief_x = make_ray(0, 0, np.sin(field_angle), 0,
                           wavelength=wavelength)
    y_ref = trace(chief_y, surfaces, wavelength).image_rays.y[0]
    x_ref = trace(chief_x, surfaces, wavelength).image_rays.x[0]

    # Tangential fan (Y) -- launch EP-centred on the chief (RT-5).
    fan_y = make_fan('y', semi_aperture, n_rays, field_angle, wavelength)
    fan_y.y = fan_y.y + ep_off
    img_y = trace(fan_y, surfaces, wavelength).image_rays
    py = np.linspace(-1, 1, n_rays)
    ey = np.where(img_y.alive, img_y.y - y_ref, np.nan)

    # Sagittal fan (X) -- launch EP-centred on the chief (RT-5).
    fan_x = make_fan('x', semi_aperture, n_rays, field_angle, wavelength)
    fan_x.x = fan_x.x + ep_off
    img_x = trace(fan_x, surfaces, wavelength).image_rays
    px = np.linspace(-1, 1, n_rays)
    ex = np.where(img_x.alive, img_x.x - x_ref, np.nan)

    return py, ey, px, ex


def ray_fan_data_world(
    surfaces: List['Surface'],
    wavelength: float,
    semi_aperture: float,
    field_angle: float = 0.0,
    n_rays: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """3.7.8: world-frame version of :func:`ray_fan_data`.

    Identical signature and return shape to ``ray_fan_data``, but
    expects each ``Surface`` in ``surfaces`` to have populated
    ``world_origin`` and ``world_R`` fields and routes through
    :func:`trace_world` so the fan is geometry-accurate on folded
    designs (the chief-ray and per-pupil-coord ray traces all
    land at the correct world image-plane position).
    """
    # RT-5 (AUDIT_RAYTRACE_CORE): the world twin never received the 4.11.2
    # EP-centred chief fix (it launched from (0,0,0)) nor the fan-centring
    # + per-orientation-chief fix now in ``ray_fan_data``.  Apply both here.
    # The paraxial ``ep_z`` from the ABCD is an axial distance (frame-
    # independent); the ``ep_off`` launch shift is exact for a straight-axis
    # world trace and an approximation for a strongly-folded design (there
    # the EP offset direction rotates with the fold).  The fallback
    # ``ep_off = 0`` reproduces the previous origin-launched behaviour.
    try:
        fod = first_order_data(surfaces, wavelength)
        ep_off = _ep_offset(fod.ep_z, field_angle)
        chief_y = make_ray(0.0, ep_off, 0.0, np.sin(field_angle),
                           wavelength=wavelength)
        chief_x = make_ray(ep_off, 0.0, np.sin(field_angle), 0.0,
                           wavelength=wavelength)
    except (ValueError, RuntimeError, ZeroDivisionError, AttributeError,
            np.linalg.LinAlgError, IndexError):
        ep_off = 0.0
        chief_y = make_ray(0, 0, 0, np.sin(field_angle),
                           wavelength=wavelength)
        chief_x = make_ray(0, 0, np.sin(field_angle), 0,
                           wavelength=wavelength)
    y_ref = trace_world(chief_y, surfaces, wavelength).image_rays.y[0]
    x_ref = trace_world(chief_x, surfaces, wavelength).image_rays.x[0]

    fan_y = make_fan('y', semi_aperture, n_rays, field_angle, wavelength)
    fan_y.y = fan_y.y + ep_off
    img_y = trace_world(fan_y, surfaces, wavelength).image_rays
    py = np.linspace(-1, 1, n_rays)
    ey = np.where(img_y.alive, img_y.y - y_ref, np.nan)

    fan_x = make_fan('x', semi_aperture, n_rays, field_angle, wavelength)
    fan_x.x = fan_x.x + ep_off
    img_x = trace_world(fan_x, surfaces, wavelength).image_rays
    px = np.linspace(-1, 1, n_rays)
    ex = np.where(img_x.alive, img_x.x - x_ref, np.nan)

    return py, ey, px, ex


def ray_fan_plot(
    surfaces: List['Surface'],
    wavelength: float,
    semi_aperture: float,
    field_angles: Optional[Sequence[float]] = None,
    n_rays: int = 101,
    ax: Optional[Tuple[Any, Any]] = None,
    units: str = 'um',
) -> Tuple[Any, Tuple[Any, Any]]:
    """Plot transverse ray aberration fans.

    Parameters
    ----------
    surfaces : list of Surface
    wavelength : float
    semi_aperture : float
    field_angles : list of float or None
        Field angles [rad] to plot.  Default: [0].
    n_rays : int
    ax : pair of Axes or None
        ``(ax_tangential, ax_sagittal)``.
    units : str

    Returns
    -------
    fig : Figure
    axes : pair of Axes
    """
    import matplotlib.pyplot as plt

    if field_angles is None:
        field_angles = [0.0]

    scale = {'um': 1e6, 'mm': 1e3, 'm': 1.0}[units]
    label = {'um': 'µm', 'mm': 'mm', 'm': 'm'}[units]

    if ax is None:
        fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        ax_t, ax_s = ax
        fig = ax_t.figure

    for fa in field_angles:
        py, ey, px, ex = ray_fan_data(surfaces, wavelength, semi_aperture,
                                      fa, n_rays)
        fa_deg = np.degrees(fa)
        ax_t.plot(py, ey * scale, label=f'{fa_deg:.1f}°')
        ax_s.plot(px, ex * scale, label=f'{fa_deg:.1f}°')

    ax_t.set_xlabel('Normalised pupil (PY)')
    ax_t.set_ylabel(f'EY [{label}]')
    ax_t.set_title('Tangential ray fan')
    ax_t.axhline(0, color='k', linewidth=0.5)
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=8)

    ax_s.set_xlabel('Normalised pupil (PX)')
    ax_s.set_ylabel(f'EX [{label}]')
    ax_s.set_title('Sagittal ray fan')
    ax_s.axhline(0, color='k', linewidth=0.5)
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(fontsize=8)

    fig.tight_layout()
    return fig, (ax_t, ax_s)


def ray_fan_plot_prescription(
    prescription: dict,
    wavelength: float,
    field_angles: Optional[Sequence[float]] = None,
    n_rays: int = 101,
    units: str = 'um',
) -> Tuple[Any, Tuple[Any, Any]]:
    """Ray fan plot from a lens prescription dict."""
    surfaces = surfaces_from_prescription(prescription)
    ap = prescription.get('aperture_diameter')
    sa = ap / 2.0 if ap else 12.7e-3
    return ray_fan_plot(surfaces, wavelength, sa, field_angles, n_rays,
                        units=units)


# ============================================================================
# Analysis: OPD (wavefront error)
# ============================================================================

def opd_fan_data(
    surfaces: List['Surface'],
    wavelength: float,
    semi_aperture: float,
    field_angle: float = 0.0,
    n_rays: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute OPD vs pupil coordinate for tangential and sagittal fans.

    Parameters
    ----------
    surfaces : list of Surface
    wavelength : float
    semi_aperture : float
    field_angle : float
    n_rays : int

    Returns
    -------
    py, opd_y, px, opd_x : ndarray
        Normalised pupil and OPD [waves] for each fan.
    """
    # Chief ray reference OPD.  4.11.2 (audit H-AB-3 sibling): chief
    # is launched at the EP centre for off-axis fields, not (0,0,0).
    # RT-5 (AUDIT_RAYTRACE_CORE): EP-centre the FANS too (see
    # ``ray_fan_data``), and reference each fan's OPD against a chief of the
    # SAME orientation (tangential tilts in M, sagittal in L) so the on-axis
    # ray of each fan reads exactly 0 waves.
    try:
        fod = first_order_data(surfaces, wavelength)
        ep_off = _ep_offset(fod.ep_z, field_angle)
        chief_y = make_ray(0.0, ep_off, 0.0, np.sin(field_angle),
                           wavelength=wavelength)
        chief_x = make_ray(ep_off, 0.0, np.sin(field_angle), 0.0,
                           wavelength=wavelength)
    except (ValueError, RuntimeError, ZeroDivisionError, AttributeError,
            np.linalg.LinAlgError, IndexError):
        # See ``ray_fan_data`` for the same fallback rationale.
        ep_off = 0.0
        chief_y = make_ray(0, 0, 0, np.sin(field_angle),
                           wavelength=wavelength)
        chief_x = make_ray(0, 0, np.sin(field_angle), 0,
                           wavelength=wavelength)
    opd_ref_y = trace(chief_y, surfaces, wavelength).image_rays.opd[0]
    opd_ref_x = trace(chief_x, surfaces, wavelength).image_rays.opd[0]

    # Tangential fan -- launch EP-centred on the chief (RT-5).
    fan_y = make_fan('y', semi_aperture, n_rays, field_angle, wavelength)
    fan_y.y = fan_y.y + ep_off
    img_y = trace(fan_y, surfaces, wavelength).image_rays
    py = np.linspace(-1, 1, n_rays)
    opd_y = np.where(img_y.alive, (img_y.opd - opd_ref_y) / wavelength, np.nan)

    # Sagittal fan -- launch EP-centred on the chief (RT-5).
    fan_x = make_fan('x', semi_aperture, n_rays, field_angle, wavelength)
    fan_x.x = fan_x.x + ep_off
    img_x = trace(fan_x, surfaces, wavelength).image_rays
    px = np.linspace(-1, 1, n_rays)
    opd_x = np.where(img_x.alive, (img_x.opd - opd_ref_x) / wavelength, np.nan)

    return py, opd_y, px, opd_x


def opd_fan_data_world(
    surfaces: List['Surface'],
    wavelength: float,
    semi_aperture: float,
    field_angle: float = 0.0,
    n_rays: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """3.7.8: world-frame version of :func:`opd_fan_data`.

    Identical signature and return shape; routes through
    :func:`trace_world` for fold-accurate OPD residuals.
    """
    # RT-5: EP-centre the fans + reference each against a same-orientation
    # chief (see ``ray_fan_data_world`` for the straight-axis/folded caveat).
    try:
        fod = first_order_data(surfaces, wavelength)
        ep_off = _ep_offset(fod.ep_z, field_angle)
        chief_y = make_ray(0.0, ep_off, 0.0, np.sin(field_angle),
                           wavelength=wavelength)
        chief_x = make_ray(ep_off, 0.0, np.sin(field_angle), 0.0,
                           wavelength=wavelength)
    except (ValueError, RuntimeError, ZeroDivisionError, AttributeError,
            np.linalg.LinAlgError, IndexError):
        ep_off = 0.0
        chief_y = make_ray(0, 0, 0, np.sin(field_angle),
                           wavelength=wavelength)
        chief_x = make_ray(0, 0, np.sin(field_angle), 0,
                           wavelength=wavelength)
    opd_ref_y = trace_world(chief_y, surfaces, wavelength).image_rays.opd[0]
    opd_ref_x = trace_world(chief_x, surfaces, wavelength).image_rays.opd[0]

    fan_y = make_fan('y', semi_aperture, n_rays, field_angle, wavelength)
    fan_y.y = fan_y.y + ep_off
    img_y = trace_world(fan_y, surfaces, wavelength).image_rays
    py = np.linspace(-1, 1, n_rays)
    opd_y = np.where(img_y.alive,
                     (img_y.opd - opd_ref_y) / wavelength, np.nan)

    fan_x = make_fan('x', semi_aperture, n_rays, field_angle, wavelength)
    fan_x.x = fan_x.x + ep_off
    img_x = trace_world(fan_x, surfaces, wavelength).image_rays
    px = np.linspace(-1, 1, n_rays)
    opd_x = np.where(img_x.alive,
                     (img_x.opd - opd_ref_x) / wavelength, np.nan)

    return py, opd_y, px, opd_x


# ============================================================================
# Analysis: through-focus spot
# ============================================================================

def refocus(
    result: 'TraceResult',
    delta_z: float,
    wavelength: Optional[float] = None,
) -> 'TraceResult':
    """Project the final bundle of a traced result to an image plane
    at ``delta_z`` downstream of the last surface's vertex, returning
    a new ``TraceResult``.

    Equivalent to appending a flat image-plane surface at
    ``thickness=delta_z`` after the last refracting surface and
    re-tracing, but closed-form -- each ray is advanced in a straight
    line from its current (post-refraction) position to the target
    z-plane, using its (L, M, N) direction cosines.  Orders of
    magnitude cheaper than the retrace when used in a focus sweep.

    Parameters
    ----------
    result : TraceResult
        Output of a previous :func:`trace` call.  Must have the
        final ray bundle available (``result.image_rays``); works
        with both ``output_filter='all'`` and ``output_filter='last'``.
    delta_z : float
        Axial distance from the last surface's VERTEX to the image
        plane [m].  Signed -- pass a negative value to move *toward*
        the lens (pre-focus).  Note: rays after :func:`trace` end at
        ``z = sag(h)`` of the last surface (not at the vertex plane
        z=0), so the effective arc length each ray travels is
        ``(delta_z - sag(h)) / N``, not ``delta_z / N``.
        ``refocus`` handles the sag-to-vertex correction internally,
        so the caller just specifies the target image distance and
        the math Just Works on curved exit surfaces.
    wavelength : float, optional
        Wavelength for resolving the image-space refractive index.
        Defaults to ``result.wavelength`` if unset.  If the
        image-space medium is glass rather than air (rare -- only
        relevant for tests that place the "image" inside a refractive
        element), the OPL update uses the correct n.

    Returns
    -------
    new_result : TraceResult
        Same surface list, same input rays, same wavelength.
        ``new_result.image_rays`` is the refocused bundle at
        z = delta_z (in the last surface's frame).

    Notes
    -----
    * Rays that were at z = sag (off-axis on a curved exit surface)
      travel a slightly longer path than rays at z = 0 (on-axis).
      The correction ``(delta_z - z_start)`` in the transfer keeps
      both paths geometrically consistent -- this is what matches
      the full-retrace behaviour that appends a flat image plane
      at ``thickness=delta_z`` after the last surface.
    * The OPL update is ``n * arc_length`` where arc length is the
      ray-path distance from (x, y, z_start) to the image plane.
      Signed, so negative ``delta_z`` subtracts OPL as expected.
    * For GRIN or highly aberrated image spaces where the
      "last-medium-is-uniform" assumption fails, use a full
      :func:`trace` with the image plane inserted rather than
      ``refocus``.
    """
    if wavelength is None:
        wavelength = result.wavelength
    n_image = get_glass_index(result.surfaces[-1].glass_after, wavelength)

    last = result.image_rays.copy()
    # Advance each ray to the image plane at z = delta_z (measured
    # from the last surface's vertex).  Rays currently sit at
    # z = sag(h), so the axial distance to travel is
    # (delta_z - z_current), and the arc length along each ray is
    # (delta_z - z_current) / N.
    with np.errstate(divide='ignore', invalid='ignore'):
        dz_remaining = delta_z - last.z
        t = np.where(last.alive & (np.abs(last.N) > 1e-30),
                     dz_remaining / last.N, 0.0)
    last.x = last.x + last.L * t
    last.y = last.y + last.M * t
    last.z = last.z + last.N * t
    # Signed OPL update: +n*t moves forward; for t < 0 this is a
    # physical "undo" of part of the previous propagation leg.
    last.opd = last.opd + n_image * t

    # Splice the refocused bundle into ray_history.  When the source
    # result used output_filter='last', ray_history has a single
    # entry; we overwrite it.  Otherwise we replace the last entry
    # (image_rays) only, leaving upstream surface-by-surface state
    # intact for callers that want it.
    if len(result.ray_history) <= 1:
        new_history = [last]
    else:
        new_history = list(result.ray_history[:-1]) + [last]

    return TraceResult(
        surfaces=result.surfaces,
        ray_history=new_history,
        input_rays=result.input_rays,
        wavelength=result.wavelength,
    )


def through_focus_rms(
    surfaces: List['Surface'],
    wavelength: float,
    semi_aperture: float,
    focus_shifts: Union[Sequence[float], np.ndarray],
    field_angle: float = 0.0,
    num_rings: int = 6,
    rays_per_ring: int = 36,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute RMS spot size at a series of focus positions.

    Useful for finding best focus.

    Performance note (3.1.8)
    ------------------------
    Earlier versions rebuilt the entire surface list on every focus
    shift and retraced from surface 0.  This version traces once
    (through the real surfaces only) and uses :func:`refocus` for
    each shift -- effectively a closed-form straight-line transfer
    in the image-space medium.  Speedup is roughly proportional to
    the number of surfaces (typically 5-20x).  Numerical output is
    identical to the pre-3.1.8 path on well-behaved systems, since
    ``refocus`` is the exact operator that ``trace`` would apply for
    the extra image-plane transfer surface.

    Parameters
    ----------
    surfaces : list of Surface
    wavelength : float
    semi_aperture : float
    focus_shifts : array-like
        Image distances [m] from the last surface vertex.  Pass
        e.g. ``bfl + np.linspace(-1e-3, 1e-3, 51)`` to scan around
        the paraxial focus.
    field_angle : float
    num_rings, rays_per_ring : int

    Returns
    -------
    shifts : ndarray
        Image distances [m].
    rms_values : ndarray
        RMS spot radius [m] at each position.
    best_shift : float
        Image distance giving minimum RMS.
    """
    focus_shifts = np.asarray(focus_shifts, dtype=np.float64)
    # S11-6d (AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1): an empty
    # ``focus_shifts`` used to fall through the whole sweep and die at the
    # ``focus_shifts[best_idx]`` return with a bare
    # ``IndexError: index 0 is out of bounds for axis 0 with size 0``,
    # naming neither this function nor the offending argument.
    if focus_shifts.ndim != 1 or focus_shifts.size == 0:
        raise ValueError(
            f"through_focus_rms: focus_shifts must be a non-empty 1-D "
            f"sequence of image distances [m]; got shape "
            f"{focus_shifts.shape}.")
    if num_rings < 1 or rays_per_ring < 1:
        raise ValueError(
            f"through_focus_rms: num_rings and rays_per_ring must both be "
            f">= 1; got num_rings={num_rings}, "
            f"rays_per_ring={rays_per_ring}.  A zero count produces an "
            f"empty ring bundle whose spot RMS is identically 0.0 at every "
            f"focus, which reads as a perfect focus.")
    rms_values = np.zeros_like(focus_shifts)

    rays = make_rings(semi_aperture, num_rings, rays_per_ring,
                      field_angle, wavelength)

    # Single base trace through the surfaces as specified.  Use
    # output_filter='last' because we only need the final bundle
    # for refocus + spot_rms.  Saves ~N_surfaces memory copies on
    # large ring counts.
    base = trace(rays, surfaces, wavelength, output_filter='last')

    for j, img_dist in enumerate(focus_shifts):
        shifted = refocus(base, float(img_dist), wavelength=wavelength)
        rms_values[j], _ = spot_rms(shifted)

    # RT-nit (AUDIT_RAYTRACE_CORE): guard the all-dead / non-finite case.
    # A fully-vignetted or TIR'd ring bundle makes ``spot_rms`` non-finite
    # (inf/NaN) at every shift; a bare ``np.argmin`` then silently returns
    # ``focus_shifts[0]`` as "best focus".  Mask non-finite shifts before
    # picking, and warn if NONE are usable.
    finite = np.isfinite(rms_values)
    if not np.any(finite):
        import warnings
        warnings.warn(
            "through_focus_rms: every focus position produced a non-finite "
            "RMS (the ring bundle fully vignettes / TIRs at field_angle="
            f"{field_angle}); best_shift is meaningless (returning "
            "focus_shifts[0]).",
            RuntimeWarning, stacklevel=2)
        best_idx = 0
    else:
        best_idx = int(np.argmin(np.where(finite, rms_values, np.inf)))
    return focus_shifts, rms_values, focus_shifts[best_idx]


__all__ = [
    'spot_rms', 'spot_geo_radius', 'spot_diagram',
    'ray_fan_data',
    'ray_fan_plot', 'ray_fan_plot_prescription',
    'opd_fan_data',
    'refocus', 'through_focus_rms',
    # NB: ``ray_fan_data_world`` and ``opd_fan_data_world`` are
    # defined in this module but deliberately omitted from
    # ``__all__`` -- pre-v5.1.0 they were importable from
    # ``lumenairy.raytrace`` (via an explicit re-export in
    # ``raytrace/__init__.py``) but were NOT in
    # ``lumenairy.raytrace.__all__`` (the advertised public
    # surface).  Keeping them off this submodule's ``__all__``
    # preserves the same "module-attribute visible but not
    # advertised" status -- callers who imported them by name
    # still resolve them; the v4.16.0 walker symmetry test
    # (tests/unit/test_v4_16_0_walker_all_symmetry.py) doesn't
    # demand a top-level re-export entry.
]

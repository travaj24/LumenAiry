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

import numpy as np
from typing import Any, List, Optional, Sequence, Tuple, Union

from ..glass import get_glass_index
from .surface import Surface, TraceResult
from .trace import (
    make_fan,
    make_ray,
    make_rings,
    surfaces_from_prescription,
    trace,
)
from .world_trace import trace_world
from .seidel import first_order_data, system_abcd


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
            # Fall back to the half-angle form (legacy behaviour).
            airy_r = 1.22 * result.wavelength / (2.0 * sd0)
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
    # Tangential fan (Y)
    fan_y = make_fan('y', semi_aperture, n_rays, field_angle, wavelength)
    res_y = trace(fan_y, surfaces, wavelength)
    img_y = res_y.image_rays

    # Reference: chief ray.  4.11.2 (audit H-AB-3 sibling): for off-
    # axis fields the chief is launched at angle from the EP centre,
    # not from (0,0,0).  Pre-4.11.2 the chief started at z=0 from the
    # axis, so ``y_ref / x_ref`` did not match the canonical "chief
    # passes through the EP centre" reference and the fan plots were
    # offset by O(field_angle * z_EP).  Mirrors the v4.10 fix in
    # ``eval_image_plane_wfe`` and v4.11.2-Track-F in
    # ``relative_illumination`` / ``field_aberration_sweep``.
    try:
        fod = first_order_data(surfaces, wavelength)
        ep_y = -fod.ep_z * np.tan(field_angle)
        chief = make_ray(0, ep_y, fod.ep_z,
                         np.sin(field_angle),
                         wavelength=wavelength)
    except (ValueError, RuntimeError, ZeroDivisionError, AttributeError,
            np.linalg.LinAlgError, IndexError):
        # No first-order pupil available (e.g. mirror-only stop-less
        # system) -- first_order_data raises ValueError on missing
        # stop, AttributeError on a stripped Surface dataclass,
        # ZeroDivisionError / LinAlgError on ill-conditioned ABCD.
        # Fall back to legacy origin-launched chief.
        chief = make_ray(0, 0, 0, np.sin(field_angle),
                         wavelength=wavelength)
    res_chief = trace(chief, surfaces, wavelength)
    y_ref = res_chief.image_rays.y[0]
    x_ref = res_chief.image_rays.x[0]

    py = np.linspace(-1, 1, n_rays)
    ey = np.where(img_y.alive, img_y.y - y_ref, np.nan)

    # Sagittal fan (X)
    fan_x = make_fan('x', semi_aperture, n_rays, field_angle, wavelength)
    res_x = trace(fan_x, surfaces, wavelength)
    img_x = res_x.image_rays

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
    fan_y = make_fan('y', semi_aperture, n_rays, field_angle, wavelength)
    res_y = trace_world(fan_y, surfaces, wavelength)
    img_y = res_y.image_rays

    chief = make_ray(0, 0, 0, np.sin(field_angle), wavelength=wavelength)
    res_chief = trace_world(chief, surfaces, wavelength)
    y_ref = res_chief.image_rays.y[0]
    x_ref = res_chief.image_rays.x[0]

    py = np.linspace(-1, 1, n_rays)
    ey = np.where(img_y.alive, img_y.y - y_ref, np.nan)

    fan_x = make_fan('x', semi_aperture, n_rays, field_angle, wavelength)
    res_x = trace_world(fan_x, surfaces, wavelength)
    img_x = res_x.image_rays

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
    # Tangential fan
    fan_y = make_fan('y', semi_aperture, n_rays, field_angle, wavelength)
    res_y = trace(fan_y, surfaces, wavelength)
    img_y = res_y.image_rays

    # Chief ray reference OPD.  4.11.2 (audit H-AB-3 sibling): chief
    # is launched at the EP centre for off-axis fields, not (0,0,0).
    # See ``ray_fan_data`` for the matching rationale.
    try:
        fod = first_order_data(surfaces, wavelength)
        ep_y = -fod.ep_z * np.tan(field_angle)
        chief = make_ray(0, ep_y, fod.ep_z,
                         np.sin(field_angle),
                         wavelength=wavelength)
    except (ValueError, RuntimeError, ZeroDivisionError, AttributeError,
            np.linalg.LinAlgError, IndexError):
        # See ``ray_fan_data`` for the same fallback rationale.
        chief = make_ray(0, 0, 0, np.sin(field_angle),
                         wavelength=wavelength)
    res_chief = trace(chief, surfaces, wavelength)
    opd_ref = res_chief.image_rays.opd[0]

    py = np.linspace(-1, 1, n_rays)
    opd_y = np.where(img_y.alive, (img_y.opd - opd_ref) / wavelength, np.nan)

    # Sagittal fan
    fan_x = make_fan('x', semi_aperture, n_rays, field_angle, wavelength)
    res_x = trace(fan_x, surfaces, wavelength)
    img_x = res_x.image_rays

    px = np.linspace(-1, 1, n_rays)
    opd_x = np.where(img_x.alive, (img_x.opd - opd_ref) / wavelength, np.nan)

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
    fan_y = make_fan('y', semi_aperture, n_rays, field_angle, wavelength)
    res_y = trace_world(fan_y, surfaces, wavelength)
    img_y = res_y.image_rays

    chief = make_ray(0, 0, 0, np.sin(field_angle), wavelength=wavelength)
    res_chief = trace_world(chief, surfaces, wavelength)
    opd_ref = res_chief.image_rays.opd[0]

    py = np.linspace(-1, 1, n_rays)
    opd_y = np.where(img_y.alive,
                     (img_y.opd - opd_ref) / wavelength, np.nan)

    fan_x = make_fan('x', semi_aperture, n_rays, field_angle, wavelength)
    res_x = trace_world(fan_x, surfaces, wavelength)
    img_x = res_x.image_rays

    px = np.linspace(-1, 1, n_rays)
    opd_x = np.where(img_x.alive,
                     (img_x.opd - opd_ref) / wavelength, np.nan)

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

    best_idx = np.argmin(rms_values)
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

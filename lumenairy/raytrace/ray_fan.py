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

Contents are bit-for-bit copies of the implementations this module was
split out of; no physics change.  See
docs/history/lumenairy.raytrace.ray_fan.md.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..glass import get_glass_index
from .exit_vertex import resolve_exit_index, vertex_plane_transfer_t
from .seidel import first_order_data, system_abcd
from .surface import Surface, TraceResult
from .trace import (
    make_fan,
    make_ray,
    make_rings,
    seed_entrance_eikonal,
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
    that is ``inf * tan(0) = inf * 0 = NaN``, and the NaN then propagates
    into every launched ray height, so ``ray_fan_data`` /
    ``opd_fan_data`` return all-NaN fans with no diagnostic.  An
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
# Shared helper: the OPD-fan reference sphere (R1)
# ============================================================================

def _eikonal(bundle):
    """Re-reference an OPD-fan bundle's OPL to the incident WAVEFRONT.

    R2 (AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11).  ``make_fan`` /
    ``make_ray`` launch on the ``z = 0`` PLANE with ``opd = 0``.  For an
    off-axis field that plane is NOT a wavefront: a ray launched at
    height ``y`` with direction ``(0, sin(theta), cos(theta))`` has
    already travelled ``y sin(theta)`` relative to the common incident
    wavefront, so the fan carries an uncorrected linear term of exactly
    ``-y sin(theta)``.  Measured on an f/4 plano-convex singlet (25 mm
    pupil, 587.6 nm): the fitted linear term of the returned fan was
    ``-185.64`` waves at 0.5 deg against ``y_max sin(theta)/lambda =
    +185.64`` (ratio ``-1.0000``), ``-744.06`` at 2 deg and ``-1879.86``
    at 5 deg -- where the real (quartic) aberration is 37.6 waves, i.e. a
    50:1 contamination.  Adding the eikonal removes it exactly.

    Applied HERE rather than in ``_make_bundle`` because that launcher is
    shared with ~20 consumers, several of which add their own entrance
    eikonal downstream (``elements/_lens_traced.py``'s v5.25.1 H6
    ``_carrier_W_fn``) and would double-count.  Bit-identical on axis
    (``L == M == 0`` makes the term identically zero).
    """
    return seed_entrance_eikonal(bundle)


def _image_space_index_for_fan(surfaces, wavelength) -> float:
    """Index of the medium the fan's exit legs travel in.

    Thin wrapper over :func:`exit_vertex.resolve_exit_index` so both OPD
    fans name themselves in the error message.  It cannot realistically
    fail where the trace itself succeeded: ``trace`` already resolves
    ``glass_before`` / ``glass_after`` for every surface, so an
    unresolvable name raises there first.
    """
    return resolve_exit_index(surfaces, wavelength, fn_name='opd_fan_data')


def _reference_sphere_radius(
    surfaces: List['Surface'],
    wavelength: float,
    N_chief: float,
    override: Optional[float],
) -> float:
    """Radius [m] of the reference sphere centred on the image point.

    Textbook convention (Welford §4; the same one rayoptics / Optiland /
    Zemax use internally, and the one
    ``analysis.image_plane_wfe(sphere_tangent='exit_pupil')`` already
    implements): the sphere is centred on the chief ray's image point and
    passes through the EXIT PUPIL centre, so its radius is the chief
    ray's path length from the XP plane to the image.

    Parameters
    ----------
    N_chief : float
        The chief ray's LONGITUDINAL DIRECTION COSINE at the evaluation
        surface (``image_rays.N[0]`` of the chief trace) -- **not** a
        refractive index.  (It was spelled ``n_chief`` until VERIFY-WP-A1
        open item 3; the value passed has always been the cosine.)

    Notes
    -----
    ``opd_fan_data`` evaluates the wavefront at the LAST surface of the
    caller's list, and :attr:`FirstOrderData.xp_z` is measured from that
    same surface, so the axial separation is ``|xp_z|`` and the arc
    length along the chief is ``|xp_z / N_chief|``.

    **Deliberate refinement over the Zemax convention.**  OpticStudio
    documents the reference sphere as "centred on the chief ray intercept
    with the image surface, radius equal to the exit pupil distance",
    i.e. the AXIAL ``|xp_z|``.  Dividing by ``|N_chief|`` uses the chief's
    SLANT path instead, so the sphere passes through the chief ray's
    actual crossing of the XP plane rather than through a point ``|xp_z|``
    away along it.  The two agree exactly on axis and differ only in the
    second-order ``n*eps^2/(2R)`` term off axis: VERIFY-WP-A1 measured
    the whole difference at **0.088 waves out of a 135.7-wave fan (0.065 %)**
    at 3 deg on an f/2.4 meniscus with 4.1 mm of transverse aberration,
    and 8.1e-3 waves out of 52.8 on a cemented doublet at 2 deg.  Pass
    ``reference_sphere_radius=abs(fod.xp_z)`` to reproduce Zemax exactly.

    Returns ``inf`` -- the reference-PLANE limit, which keeps the exact
    first-order ``-n*eps*sin(theta')`` correction and drops only the
    second-order ``n*eps^2/(2R)`` term -- whenever the exit pupil is
    unavailable (degenerate / stop-less / afocal prescriptions, or an XP
    that lands on the evaluation surface itself).  That fallback is the
    conservative one: it is never worse than a wrong finite radius.
    """
    if override is not None:
        R = float(override)
        if np.isnan(R) or R <= 0.0:
            raise ValueError(
                f"opd_fan_data: reference_sphere_radius must be a positive "
                f"distance in metres (or np.inf for a reference PLANE, or "
                f"None to derive it from the exit pupil); got {override!r}.")
        return R
    try:
        fod = first_order_data(surfaces, wavelength)
        xp_z = float(fod.xp_z)
    except (ValueError, RuntimeError, ZeroDivisionError, AttributeError,
            np.linalg.LinAlgError, IndexError):
        return float('inf')
    nc = abs(float(N_chief))
    if nc < 1e-12:
        nc = 1.0
    R = abs(xp_z) / nc
    if not np.isfinite(R) or R <= 0.0:
        return float('inf')
    return R


def _reference_sphere_leg(vx, vy, vz, L, M, N, R):
    """Signed OPL leg [m] from each ray's intercept to the reference sphere.

    ``v = P - C`` points from the sphere centre ``C`` (the chief ray's
    image point) to the ray's intercept ``P``; ``(L, M, N)`` is the ray
    direction.  The upstream crossing of ``|X - C| = R`` is at
    ``X = P + t d`` with

    .. math::
        t = -(v \\cdot d) - \\sqrt{(v \\cdot d)^2 + R^2 - |v|^2}

    and the chief (``v = 0``) has ``t = -R`` exactly, so the
    chief-referenced leg is ``t + R``.  Evaluated in the
    cancellation-free form ``(A^2 - B^2)/(A + B)`` with
    ``A = R - v.d``, ``B = sqrt(R^2 + (v.d)^2 - |v|^2)``:

    .. math::
        t + R = \\frac{|v|^2 - 2 R (v \\cdot d)}
                     {R - v \\cdot d + \\sqrt{R^2 + (v\\cdot d)^2 - |v|^2}}

    which is exact and stays accurate when ``|v| << R`` (the direct form
    subtracts two numbers that agree to ~10 digits).

    ``R = inf`` gives the reference-PLANE limit ``-(v . d)`` -- the pure
    first-order correction ``-eps sin(theta')``.

    Returns NaN for a ray farther from the image point than the sphere
    radius (``|v| > R``), i.e. one that never crosses the sphere; such a
    ray has no defined wavefront error on this reference.
    """
    vd = vx * L + vy * M + vz * N
    if not np.isfinite(R):
        return -vd
    v2 = vx * vx + vy * vy + vz * vz
    rad = R * R + vd * vd - v2
    with np.errstate(invalid='ignore'):
        root = np.sqrt(np.where(rad > 0.0, rad, np.nan))
        denom = R - vd + root
        return np.where(np.abs(denom) > 0.0, (v2 - 2.0 * R * vd) / denom,
                        np.nan)


def _opd_fan_wfe(img, chief, n_img, R, wavelength, fn_name='opd_fan_data'):
    """Chief-referenced wavefront error [waves] for one fan.

    ``img`` is the fan's final :class:`RayBundle`, ``chief`` the
    single-ray bundle of the SAME orientation traced through the same
    surfaces.  Dead rays come back NaN.

    Guard: the reference sphere only exists if every ray is INSIDE it
    (``|P - C| < R``).  It is not when the evaluation surface sits far
    from the image -- e.g. a bare singlet with no image plane appended,
    where the chief's "image point" is its own intercept on the rear
    face, the exit pupil is ~2 mm behind it, and the marginal rays are
    2.5 mm off axis.  There is no wavefront error to report in that
    geometry (there is no image point), so rather than returning the
    NaN / 1000-wave garbage an unguarded sphere solve produces, fall
    back to the reference-PLANE limit and say why.
    """
    cx = float(chief.x[0])
    cy = float(chief.y[0])
    cz = float(chief.z[0])
    opd_ref = float(chief.opd[0])
    vx, vy, vz = img.x - cx, img.y - cy, img.z - cz
    if np.isfinite(R):
        alive = np.asarray(img.alive, dtype=bool)
        v_max = float(np.max(np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)[alive])) \
            if alive.any() else 0.0
        if v_max >= R:
            import warnings
            warnings.warn(
                f"{fn_name}: the reference sphere centred on the chief "
                f"ray's image point has radius {R:.6g} m (the chief's path "
                f"back to the exit pupil), but a ray of this fan sits "
                f"{v_max:.6g} m away from that point -- i.e. the LAST "
                f"surface of the prescription is nowhere near the image, so "
                f"there is no image point to reference the wavefront to.  "
                f"Append a flat image surface at the paraxial focus (see "
                f"find_paraxial_focus) to get a wavefront error.  Falling "
                f"back to a reference PLANE for this fan; the result still "
                f"carries the lens's own converging curvature and is NOT a "
                f"wavefront aberration.",
                RuntimeWarning, stacklevel=3)
            R = float('inf')
    leg = _reference_sphere_leg(vx, vy, vz, img.L, img.M, img.N, R)
    w = ((img.opd - opd_ref) + n_img * leg) / wavelength
    return np.where(img.alive, w, np.nan)


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
    # fan would not pass through zero at py=0 (``ey(0)`` reading the
    # launch-convention offset instead of 0).  We shift each fan's LAUNCH
    # heights
    # heights
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
    *,
    reference_sphere_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the WAVEFRONT ERROR vs pupil coordinate for both fans.

    Each ray is referenced to a **reference sphere centred on the chief
    ray's image point and passing through the exit pupil** -- the
    standard wavefront-aberration definition (Welford,
    *Aberrations of Optical Systems*, §4: "the optical path from the
    object to the Gaussian image point").

    Parameters
    ----------
    surfaces : list of Surface
        The system.  The wavefront is evaluated at the LAST surface, so
        for a meaningful OPD fan that surface should be at (or near) the
        image plane.
    wavelength : float
        Vacuum wavelength [m].
    semi_aperture : float
        Pupil semi-aperture [m] that ``rho = 1`` maps to.
    field_angle : float
        Field half-angle [rad].
    n_rays : int
        Rays per fan.
    reference_sphere_radius : float, optional
        Radius [m] of the reference sphere.  ``None`` (default) derives
        it from the exit pupil (``|xp_z / N_chief|``, see
        :func:`_reference_sphere_radius`); ``np.inf`` selects a reference
        PLANE, which keeps the exact first-order term and drops only the
        second-order ``n*eps**2/(2R)`` one; a positive float overrides
        both.

    Returns
    -------
    py, opd_y, px, opd_x : ndarray
        Normalised pupil coordinate and wavefront error [waves] for the
        tangential (y) and sagittal (x) fans.  Vignetted rays are NaN.

    Notes
    R1 (AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11).  Returning
    ``(img.opd - opd_chief) / wavelength`` -- the OPL to each ray's OWN
    intercept, with no reference sphere -- differs from the wavefront
    error at FIRST order in the transverse aberration:
    ``W_plane - W_true = eps * sin(theta')``.
    ``W_plane - W_true = eps * sin(theta')``.
    Measured on an f/4 plano-convex singlet (R1 = 51.68 mm N-BK7, 25 mm
    pupil, 587.6 nm) at ``rho = 1``: ``+36.194`` waves reported against
    ``-11.714`` waves true -- the wrong SIGN and 3.1x the magnitude.  The
    residual matched ``|eps sin(theta')|`` to 3 decimal places at f/20
    (0.0726 w) and f/50 (0.0019 w), identifying the mechanism
    unambiguously.  The function was right exactly when it did not
    matter: an aberration-free parabolic mirror (eps == 0) gave PV
    7.1e-11 waves before and after.

    The corrected value is cross-checked against the Seidel relation
    ``W(rho=1) ~ -S1/8``: ``-11.71`` waves here vs ``-11.30`` waves from
    :func:`seidel_coefficients`, the remainder being genuine 5th order
    (the real-ray rho**6 fit contributes ``-0.41`` waves).

    See Also
    --------
    lumenairy.analysis.image_plane_wfe.eval_image_plane_wfe :
        the full 2-D pupil version, with best-focus search.
    """
    # Chief ray reference.  4.11.2 (audit H-AB-3 sibling): chief
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
    _FN = 'opd_fan_data'
    ref_y = trace(_eikonal(chief_y), surfaces, wavelength).image_rays
    ref_x = trace(_eikonal(chief_x), surfaces, wavelength).image_rays

    n_img = _image_space_index_for_fan(surfaces, wavelength)
    R_y = _reference_sphere_radius(surfaces, wavelength, ref_y.N[0],
                                   reference_sphere_radius)
    R_x = _reference_sphere_radius(surfaces, wavelength, ref_x.N[0],
                                   reference_sphere_radius)

    # Tangential fan -- launch EP-centred on the chief (RT-5).
    fan_y = make_fan('y', semi_aperture, n_rays, field_angle, wavelength)
    fan_y.y = fan_y.y + ep_off
    img_y = trace(_eikonal(fan_y), surfaces, wavelength).image_rays
    py = np.linspace(-1, 1, n_rays)
    opd_y = _opd_fan_wfe(img_y, ref_y, n_img, R_y, wavelength,
                         fn_name=_FN)

    # Sagittal fan -- launch EP-centred on the chief (RT-5).
    fan_x = make_fan('x', semi_aperture, n_rays, field_angle, wavelength)
    fan_x.x = fan_x.x + ep_off
    img_x = trace(_eikonal(fan_x), surfaces, wavelength).image_rays
    px = np.linspace(-1, 1, n_rays)
    opd_x = _opd_fan_wfe(img_x, ref_x, n_img, R_x, wavelength,
                         fn_name=_FN)

    return py, opd_y, px, opd_x


def opd_fan_data_world(
    surfaces: List['Surface'],
    wavelength: float,
    semi_aperture: float,
    field_angle: float = 0.0,
    n_rays: int = 101,
    *,
    reference_sphere_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """3.7.8: world-frame version of :func:`opd_fan_data`.

    Identical signature and return shape; routes through
    :func:`trace_world` for fold-accurate wavefront residuals.  Carries
    the same R1 reference-sphere referencing -- ``trace_world`` records
    its history in each surface's LOCAL frame, so the chief-relative
    geometry the sphere needs is expressed in exactly the same frame as
    on the sequential path.
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
    _FN = 'opd_fan_data_world'
    ref_y = trace_world(_eikonal(chief_y), surfaces, wavelength).image_rays
    ref_x = trace_world(_eikonal(chief_x), surfaces, wavelength).image_rays

    n_img = _image_space_index_for_fan(surfaces, wavelength)
    R_y = _reference_sphere_radius(surfaces, wavelength, ref_y.N[0],
                                   reference_sphere_radius)
    R_x = _reference_sphere_radius(surfaces, wavelength, ref_x.N[0],
                                   reference_sphere_radius)

    fan_y = make_fan('y', semi_aperture, n_rays, field_angle, wavelength)
    fan_y.y = fan_y.y + ep_off
    img_y = trace_world(_eikonal(fan_y), surfaces, wavelength).image_rays
    py = np.linspace(-1, 1, n_rays)
    opd_y = _opd_fan_wfe(img_y, ref_y, n_img, R_y, wavelength,
                         fn_name=_FN)

    fan_x = make_fan('x', semi_aperture, n_rays, field_angle, wavelength)
    fan_x.x = fan_x.x + ep_off
    img_x = trace_world(_eikonal(fan_x), surfaces, wavelength).image_rays
    px = np.linspace(-1, 1, n_rays)
    opd_x = _opd_fan_wfe(img_x, ref_x, n_img, R_x, wavelength,
                         fn_name=_FN)

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
    # (delta_z - z_current) / N.  Shares the ``vertex_plane_transfer_t``
    # kernel with ``exit_vertex_transfer`` / ``_transfer`` /
    # ``_intersect_surface``'s flat branch, so ``refocus(result, 0.0)``
    # and ``result.at_exit_vertex()`` are the same arithmetic (they
    # differ only in the grazing-ray policy: ``refocus`` leaves a
    # grazing ray alive and unmoved, as it always has, because a focus
    # sweep must not mutate the alive mask under the caller).
    t, _unreachable = vertex_plane_transfer_t(last.z, last.N, last.alive,
                                              z_target=delta_z)
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
    # ``focus_shifts`` falls through the whole sweep and dies at the
    # ``focus_shifts[best_idx]`` return with a bare
    # ``IndexError: index 0 is out of bounds for axis 0 with size 0``,
    # naming neither this function nor the offending argument.
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
    # ``__all__`` -- they are importable from
    # ``lumenairy.raytrace`` (via an explicit re-export in
    # ``raytrace/__init__.py``) but are NOT in
    # ``lumenairy.raytrace.__all__`` (the advertised public
    # surface).  Keeping them off this submodule's ``__all__``
    # surface).  Keeping them off this submodule's ``__all__``
    # preserves the same "module-attribute visible but not
    # advertised" status -- callers who imported them by name
    # still resolve them; the v4.16.0 walker symmetry test
    # (tests/unit/test_v4_16_0_walker_all_symmetry.py) doesn't
    # demand a top-level re-export entry.
]

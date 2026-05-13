"""
lumenairy.raytrace.world -- world-frame surface construction.

A prescription with coord-break entries (e.g. a folded ``.zmx`` design
loaded via :func:`load_zmx_prescription`) carries tilt / decenter
information in ``prescription['coord_breaks']`` and surface order in
``surf_num``.  :func:`surfaces_from_prescription` (4.3 and earlier)
discards that info -- it returns a local-frame surface list suitable
only for non-folded designs.

This module (4.4.0) adds the missing prescription-to-world-frame
translator.  Walks the combined coord-break + optical-surface sequence
in ``surf_num`` order, accumulates per-surface position
``world_origin`` and rotation ``world_R`` (Zemax PARM convention --
decenter applied first or last depending on ``order``, tilts as
rotations about local x then y axes), and emits a list of
:class:`Surface` objects with the world-frame metadata populated.  Pair
with :func:`trace_world` for folded-design ray tracing from any
script:

    presc = la.load_zmx_prescription('folded_design.zmx')
    surfaces = la.world_surfaces_from_prescription(presc)
    result = la.trace_world(rays, surfaces, wavelength)

For prescriptions WITHOUT coord-breaks, the result is equivalent to
:func:`surfaces_from_prescription` with each surface decorated with an
identity rotation and a cumulative origin along the optical axis.
This is harmless when :func:`trace_world` is the consumer; the two
trace paths agree to numerical precision on straight-axis designs.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .core import Surface, surfaces_from_prescription, system_abcd


def _rot_x(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,  -s],
                     [0.0,   s,   c]])


def _rot_y(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[  c, 0.0,   s],
                     [0.0, 1.0, 0.0],
                     [ -s, 0.0,   c]])


def _rot_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[  c,  -s, 0.0],
                     [  s,   c, 0.0],
                     [0.0, 0.0, 1.0]])


def _apply_coord_break(origin: np.ndarray, R: np.ndarray,
                        cb: dict) -> tuple:
    """Apply a single Zemax-style coord-break to (origin, R) and return
    the new frame.  Honours the PARM 6 ``order`` field
    (0 = decenter then tilt, 1 = tilt then decenter)."""
    dx = float(cb.get('decenter_x_m', 0.0))
    dy = float(cb.get('decenter_y_m', 0.0))
    tx = np.radians(float(cb.get('tilt_x_deg', 0.0)))
    ty = np.radians(float(cb.get('tilt_y_deg', 0.0)))
    tz = np.radians(float(cb.get('tilt_z_deg', 0.0)))
    order = int(cb.get('order', 0) or 0)
    tilt_R = _rot_x(tx) @ _rot_y(ty) @ _rot_z(tz)
    if order == 0:
        # Decenter first (in current frame), then tilt.
        new_origin = origin + R @ np.array([dx, dy, 0.0])
        new_R = R @ tilt_R
    else:
        # Tilt first, then decenter (in the new frame).
        new_R = R @ tilt_R
        new_origin = origin + new_R @ np.array([dx, dy, 0.0])
    return new_origin, new_R


def world_surfaces_from_prescription(prescription) -> List[Surface]:
    """Build world-frame :class:`Surface` objects from a prescription.

    Walks the combined coord-break + optical-surface sequence in
    ``surf_num`` order and accumulates per-surface ``world_origin`` and
    ``world_R``.  The output is consumable by :func:`trace_world` for
    folded-design ray tracing from a script.

    Parameters
    ----------
    prescription : dict
        Prescription dict from :func:`load_zmx_prescription`,
        :func:`make_singlet`, etc.  May or may not contain
        ``'coord_breaks'``; absent or empty -> straight optical axis
        with identity rotations.

    Returns
    -------
    surfaces : list of Surface
        Each carries ``world_origin`` (m, shape ``(3,)``) and
        ``world_R`` (shape ``(3, 3)``).  ``world_origin`` of the first
        surface is the origin; subsequent surfaces advance along the
        local +z by the inter-surface thickness, modified by any
        intervening coord-breaks.

    Notes
    -----
    * Mirrors are emitted with ``is_mirror=True``; reflection is
      handled by :func:`trace_world` in the surface's local frame.
      Folded designs typically include a downstream coord-break that
      re-aligns the axis with the reflected ray direction.
    * The Zemax PARM convention is honoured for coord-break ordering
      (``parm6 = 0`` -> decenter then tilt, ``= 1`` -> tilt then
      decenter).
    * For prescriptions without coord-breaks the result is the same
      surface set as :func:`surfaces_from_prescription`, only with
      identity ``world_R`` and a cumulative origin populated.

    Examples
    --------
    Round-trip on a straight singlet (no folds): world trace and
    local trace agree to within numerical precision:

    >>> import numpy as np, lumenairy as la
    >>> presc = la.make_singlet(R1=50e-3, R2=np.inf, d=3e-3,
    ...                          glass='N-BK7', aperture=10e-3)
    >>> rays = la.make_rings(4e-3, 3, 8, 0.0, 1.31e-6)
    >>> wsurfs = la.world_surfaces_from_prescription(presc)
    >>> r = la.trace_world(rays, wsurfs, 1.31e-6)

    For a folded design loaded from a ``.zmx``:

    >>> # presc = la.load_zmx_prescription('folded.zmx')
    >>> # wsurfs = la.world_surfaces_from_prescription(presc)
    >>> # result = la.trace_world(rays, wsurfs, wavelength)
    """
    base_surfaces = surfaces_from_prescription(prescription)
    cbs = list(prescription.get('coord_breaks') or [])

    # Build an interleaved schedule of (surf_num, kind, payload).
    # Optical surfaces and coord-breaks each carry their own
    # ``surf_num`` from the .zmx loader.
    events = []
    for s in base_surfaces:
        sn = int(getattr(s, 'surf_num', 0) or 0)
        events.append((sn, 'surface', s))
    for cb in cbs:
        sn = int(cb.get('surf_num', 0) or 0)
        events.append((sn, 'coordbrk', cb))
    # Stable sort by surf_num.  When a cb and a surface share a
    # surf_num (rare but legal in some loaders) put the cb first so
    # it modifies the frame before the surface is placed.
    events.sort(key=lambda e: (e[0], 0 if e[1] == 'coordbrk' else 1))

    origin = np.zeros(3, dtype=float)
    R = np.eye(3, dtype=float)

    out: List[Surface] = []
    for sn, kind, payload in events:
        if kind == 'coordbrk':
            origin, R = _apply_coord_break(origin, R, payload)
            # Advance along the new local +z by the coord-break's
            # own thickness (PARM-DISZ, in m).
            t_cb = float(payload.get('thickness_m', 0.0) or 0.0)
            origin = origin + t_cb * R[:, 2]
            continue
        # Optical surface: clone the base Surface, populate world
        # frame, and advance the cursor by its thickness.
        s = payload
        from copy import copy as _copy
        ws = _copy(s)
        ws.world_origin = origin.copy()
        ws.world_R = R.copy()
        out.append(ws)
        origin = origin + float(s.thickness or 0.0) * R[:, 2]
    return out


def paraxial_focus_world(world_surfaces, wavelength, *,
                          aperture_radius=None):
    """Paraxial focus position and propagation direction in world coords.

    Traces a chief ray and a paraxial marginal ray through the world
    surfaces with :func:`trace_world` and finds their convergence
    point.  This approach is robust to folded prescriptions, mirror
    reflections, and Zemax-signed post-mirror thicknesses -- because
    it uses actual ray traces, the sign convention in the prescription
    doesn't have to be reconciled separately.

    Parameters
    ----------
    world_surfaces : list of Surface
        Output of :func:`world_surfaces_from_prescription`.  Every
        surface must carry ``world_origin`` and ``world_R``.
    wavelength : float
        Wavelength [m].
    aperture_radius : float, optional
        Marginal-ray launch height [m] at the entrance pupil.
        Defaults to 1/10 of the smallest finite ``semi_diameter`` in
        the surface list (or 1 mm if no finite semi-diameter is
        available).  Smaller values give a more strictly paraxial
        result but lose numerical conditioning.

    Returns
    -------
    focus_origin : ndarray (shape ``(3,)``)
        Paraxial image-plane vertex in world coordinates [m].
    focus_normal : ndarray (shape ``(3,)``)
        Unit vector along the chief ray at focus (the image-plane
        normal).  ``(0, 0, 1)`` for un-folded systems, ``(0, +1, 0)``
        for a 90-deg fold to +y, etc.

    Notes
    -----
    Returns the midpoint of the closest-approach segment between the
    two traced rays, which collapses to the geometric image for a
    paraxial system.  For systems with strong aberrations the focus
    location will be perturbed; for those use
    :func:`through_focus_rms` / :func:`find_best_focus`.

    Examples
    --------
    >>> # Periscope: singlet at z=0..3mm, fold mirror at z=53mm,
    >>> # detector 50mm past in +y.  Focus = singlet's image, 100mm
    >>> # past the singlet along the bent axis.
    >>> wsurfs = la.world_surfaces_from_prescription(periscope_presc)
    >>> focus, normal = la.paraxial_focus_world(wsurfs, 1.31e-6)
    """
    if not world_surfaces:
        raise ValueError(
            "paraxial_focus_world: empty world_surfaces list.")
    last = world_surfaces[-1]
    if (getattr(last, 'world_origin', None) is None
            or getattr(last, 'world_R', None) is None):
        raise ValueError(
            "paraxial_focus_world: last surface has no world_origin / "
            "world_R; pass surfaces produced by "
            "world_surfaces_from_prescription.")

    # Pick a marginal-ray launch height.
    if aperture_radius is None:
        finite_sds = [float(s.semi_diameter) for s in world_surfaces
                      if np.isfinite(s.semi_diameter)
                      and s.semi_diameter > 0]
        if finite_sds:
            aperture_radius = 0.1 * min(finite_sds)
        else:
            aperture_radius = 1e-3

    from .core import trace_world, _make_bundle

    # Both rays travel parallel to world +z at z = 0 (paraxial /
    # infinite-object setup).  Chief at (0, 0); marginal at
    # (0, aperture_radius).
    chief = _make_bundle(
        x=np.array([0.0]), y=np.array([0.0]),
        L=np.array([0.0]), M=np.array([0.0]),
        wavelength=float(wavelength),
    )
    marginal = _make_bundle(
        x=np.array([0.0]), y=np.array([float(aperture_radius)]),
        L=np.array([0.0]), M=np.array([0.0]),
        wavelength=float(wavelength),
    )
    res_chief = trace_world(chief, world_surfaces, wavelength)
    res_marg = trace_world(marginal, world_surfaces, wavelength)

    def _world_state(result):
        rb = result.image_rays
        local_pos = np.array([rb.x[0], rb.y[0], rb.z[0]], dtype=float)
        local_dir = np.array([rb.L[0], rb.M[0], rb.N[0]], dtype=float)
        world_pos = (np.asarray(last.world_origin, dtype=float)
                     + last.world_R @ local_pos)
        world_dir = last.world_R @ local_dir
        return world_pos, world_dir

    p1, d1 = _world_state(res_chief)
    p2, d2 = _world_state(res_marg)

    # Closest-approach point between two lines in 3D.  Solve the
    # 2x2 normal-equation system for (t1, t2).
    A = np.array([[d1 @ d1, -(d1 @ d2)],
                  [-(d2 @ d1), d2 @ d2]])
    b = np.array([d1 @ (p2 - p1), d2 @ (p1 - p2)])
    try:
        t1, t2 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Parallel rays (e.g. infinite focal length): no finite focus.
        raise ValueError(
            "paraxial_focus_world: chief and marginal rays do not "
            "converge (parallel post-system).  System may be afocal "
            "or aperture_radius too small.")
    pt1 = p1 + t1 * d1
    pt2 = p2 + t2 * d2
    focus_origin = 0.5 * (pt1 + pt2)
    chief_norm = float(np.linalg.norm(d1))
    focus_normal = d1 / chief_norm if chief_norm > 0 else d1
    return focus_origin, focus_normal


__all__ = [
    'world_surfaces_from_prescription',
    'paraxial_focus_world',
]

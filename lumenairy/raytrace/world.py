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

from .core import Surface, surfaces_from_prescription


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


__all__ = ['world_surfaces_from_prescription']

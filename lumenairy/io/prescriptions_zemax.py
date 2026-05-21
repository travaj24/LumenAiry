"""
Zemax ``.zmx`` / ``.txt`` prescription I/O.

Loaders for Zemax sequential ``.zmx`` text files and Zemax
*Analyze -> Reports -> Prescription Data* ``.txt`` exports, plus the
matching exporters (``export_zemax_lens_data``, ``export_zemax_zmx``).

v5.1.0 split (Agent F): extracted from ``lumenairy/io/prescriptions.py``
without any logic change.  Public API is unchanged: every name in this
module is re-exported through ``lumenairy.io.prescriptions`` and
``lumenairy``.

Author: Andrew Traverso
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..glass import GLASS_REGISTRY

# ============================================================================
# Zemax .zmx file parser
# ============================================================================

def load_zemax_zmx(filepath: str,
                   surface_range: Optional[Tuple[int, int]] = None,
                   name: Optional[str] = None) -> Dict[str, Any]:
    """Parse a Zemax ``.zmx`` text file and return a lens prescription dict.

    Reads the surface table from a Zemax sequential lens file and builds a
    prescription dict compatible with :func:`apply_real_lens`.  Handles
    standard spherical surfaces, even-aspheric surfaces (EVENASPH), and
    Forbes Q-type freeforms (QBFS / QCON, v4.15.1+).  Q-type surfaces
    are emitted with the canonical LumenAiry freeform keys
    ``freeform_type``, ``q_bfs_coeffs`` / ``q_con_coeffs`` and
    ``r_max`` -- consult :func:`surface_sag_q_bfs` /
    :func:`surface_sag_q_con` for the consumer side.

    Parameters
    ----------
    filepath : str
        Path to the .zmx file.
    surface_range : tuple of (int, int), optional
        Which Zemax surface numbers to include as lens surfaces, given as
        ``(first, last)`` inclusive.  For example, ``(2, 4)`` extracts
        surfaces 2, 3, and 4 from the Zemax file.  If *None*, all surfaces
        between the first and last glass surfaces are automatically detected.
    name : str, optional
        Human-readable label.  If *None*, derived from the filename.

    Returns
    -------
    result : dict
        ``'name'`` : str -- human-readable label

        ``'aperture_diameter'`` : float -- clear aperture [m]

        ``'surfaces'`` : list -- refracting surfaces only (for ``apply_real_lens``)

        ``'thicknesses'`` : list -- thicknesses between refracting surfaces [m]

        ``'elements'`` : list -- full element list including mirrors, each with
        ``'element_type'``: ``'surface'`` or ``'mirror'``

        ``'all_thicknesses'`` : list -- thicknesses between all elements [m]

    Notes
    -----
    The ``'surfaces'`` and ``'thicknesses'`` keys give a lens-only
    prescription that can be passed directly to :func:`apply_real_lens`.
    Mirrors and coordinate breaks are excluded from this.

    The ``'elements'`` key gives the full parsed sequence including mirrors
    (``element_type='mirror'``), which can be used with :func:`apply_mirror`
    for manual propagation through folded systems.

    Coordinate break surfaces (COORDBRK) are skipped entirely -- they
    represent geometric transforms that are not modeled by ASM.

    Conic constants are read only from dedicated ``CONI`` lines in the .zmx
    file.  The extra fields on the ``CURV`` line (which encode solve
    parameters like pickup scale factors) are ignored.

    Examples
    --------
    >>> rx = load_zemax_zmx('AC254-200-C.zmx')
    >>> E_out = apply_real_lens(E_in, prescription=rx, wavelength=1.3e-6, dx=2.1e-6)

    >>> rx = load_zemax_zmx('my_design.zmx', surface_range=(2, 5))
    """
    # Read file -- try UTF-16-LE first (Zemax default), then UTF-8
    for encoding in ('utf-16-le', 'utf-8', 'latin-1'):
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                text = f.read()
            if 'SURF' in text:
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise IOError(f"Could not read {filepath} with any supported encoding")

    # Remove BOM if present
    text = text.lstrip('﻿')
    lines = text.split('\n')

    # Determine unit scale factor (convert to meters)
    unit_scale = 1e-3  # default: mm
    for line in lines:
        tokens = line.strip().split()
        if tokens and tokens[0] == 'UNIT':
            unit_str = tokens[1].upper() if len(tokens) > 1 else 'MM'
            # 4.9 fix (audit #4.4): some Zemax exports use the long
            # spelling ``INCH`` / ``INCHES`` instead of the short
            # ``IN``.  Accept both.
            unit_map = {
                'MM': 1e-3, 'CM': 1e-2, 'M': 1.0,
                'IN': 25.4e-3, 'INCH': 25.4e-3, 'INCHES': 25.4e-3,
            }
            unit_scale = unit_map.get(unit_str, 1e-3)
            break

    # ------------------------------------------------------------------
    # Parse surfaces
    # ------------------------------------------------------------------
    surfaces_raw = []
    current_surf = None

    for line in lines:
        stripped = line.strip()
        tokens = stripped.split()
        if not tokens:
            continue

        keyword = tokens[0]

        if keyword == 'SURF':
            if current_surf is not None:
                surfaces_raw.append(current_surf)
            current_surf = {
                'surf_num': int(tokens[1]),
                'type': 'STANDARD',
                'curvature': 0.0,
                'conic': 0.0,
                'thickness': 0.0,
                'glass': None,
                'semi_diameter': 0.0,
                'aspheric_params': {},
                'is_stop': False,
                'is_mirror': False,
                'is_coordbrk': False,
                'comment': '',
            }

        elif current_surf is not None:
            if keyword == 'TYPE':
                stype = tokens[1] if len(tokens) > 1 else 'STANDARD'
                current_surf['type'] = stype
                if stype == 'COORDBRK':
                    current_surf['is_coordbrk'] = True

            elif keyword == 'STOP':
                current_surf['is_stop'] = True

            elif keyword == 'CURV':
                # Only read the curvature value (first token after keyword).
                # Remaining fields are solve parameters (pickup source,
                # scale factor, etc.) -- NOT conic constants.
                current_surf['curvature'] = float(tokens[1])

            elif keyword == 'CONI':
                current_surf['conic'] = float(tokens[1])

            elif keyword == 'DISZ':
                if tokens[1].upper() == 'INFINITY':
                    current_surf['thickness'] = float('inf')
                else:
                    current_surf['thickness'] = float(tokens[1])

            elif keyword == 'GLAS':
                glass_name = tokens[1]
                current_surf['glass'] = glass_name
                if glass_name.upper() == 'MIRROR':
                    current_surf['is_mirror'] = True

            elif keyword == 'MIRR':
                # Some files use MIRR flag instead of GLAS MIRROR
                try:
                    if int(tokens[1]) == 1:  # 1 = reflective
                        current_surf['is_mirror'] = True
                except (ValueError, IndexError):
                    pass

            elif keyword == 'DIAM':
                current_surf['semi_diameter'] = float(tokens[1])

            elif keyword == 'PARM':
                parm_num = int(tokens[1])
                parm_val = float(tokens[2])
                # PARM 0 is meaningful on Q-type freeforms (Norm Radius
                # convention; see Forbes 2007 sect. 5 / Zemax QBFS QCON
                # docs).  Pre-v4.15.1 the loader only stored non-zero
                # values and the parm_num >= 1 filter further dropped
                # any PARM 0 sourced from a Q-type freeform; v4.15.1
                # stores PARM 0 unconditionally and decides per-surface
                # how to consume it (Q-type r_max vs EVENASPH ignore).
                if parm_num == 0 or parm_val != 0.0:
                    current_surf['aspheric_params'][parm_num] = parm_val

            elif keyword == 'COMM':
                current_surf['comment'] = stripped[5:].strip().strip('"')

    # Don't forget the last surface
    if current_surf is not None:
        surfaces_raw.append(current_surf)

    # ------------------------------------------------------------------
    # Filter out coordinate breaks (non-optical surfaces)
    # ------------------------------------------------------------------
    optical_surfaces = [s for s in surfaces_raw if not s['is_coordbrk']]

    # ------------------------------------------------------------------
    # Determine which surfaces are part of the lens
    # ------------------------------------------------------------------
    if surface_range is not None:
        s_first, s_last = surface_range
        lens_surfaces = [s for s in optical_surfaces
                         if s_first <= s['surf_num'] <= s_last]
    else:
        # Auto-detect: find first and last surfaces with glass or mirror
        active = [s for s in optical_surfaces
                  if s['glass'] is not None or s['is_mirror']]
        if not active:
            raise ValueError(f"No glass/mirror surfaces found in {filepath}")
        s_first = active[0]['surf_num']
        s_last = active[-1]['surf_num'] + 1
        lens_surfaces = [s for s in optical_surfaces
                         if s_first <= s['surf_num'] <= s_last]

    if len(lens_surfaces) < 2:
        raise ValueError(
            f"Need at least 2 surfaces, got {len(lens_surfaces)} "
            f"in range ({s_first}, {s_last})")

    # ------------------------------------------------------------------
    # Object-space distance
    # ------------------------------------------------------------------
    # Zemax files typically have a chain of non-refractive surfaces
    # (OBJ plane, STOP, coordinate breaks, dummy reference planes, etc.)
    # before the first real lens surface.  These get filtered out of
    # ``lens_surfaces`` here, but their DISZ (z-thickness) values can
    # carry meaningful design geometry -- in particular, the distance
    # from the object/source plane to the first refractive surface.
    #
    # Without preserving that total, a downstream simulation that
    # propagates its own source field through the prescription will
    # implicitly place the source AT the first refractive surface,
    # collapsing the design's obj-space geometry.  For a field source
    # with finite angular spread (Gaussian beam, collimated array), this
    # causes a focal-plane defocus proportional to the dropped distance.
    #
    # Convention for ``object_distance``: the sum of DISZ values from
    # the STOP surface (treated as the TX / source plane) up to but not
    # including the first refractive surface of ``lens_surfaces``.  If
    # no STOP is present, sum from SURF 0 onward.  Non-finite DISZ
    # values (``INFINITY``) contribute 0 since Zemax uses INFINITY for
    # collimated-source configurations where the object is at infinity.
    stop_idx_in_raw = None
    for _idx, _s in enumerate(surfaces_raw):
        if _s.get('is_stop'):
            stop_idx_in_raw = _idx
            break
    # Find the index of the first lens surface in surfaces_raw.
    first_lens_surf_num = lens_surfaces[0]['surf_num']
    first_lens_idx_in_raw = next(
        (_idx for _idx, _s in enumerate(surfaces_raw)
         if _s['surf_num'] == first_lens_surf_num),
        None)
    obj_distance = 0.0
    if first_lens_idx_in_raw is not None:
        _start = stop_idx_in_raw if stop_idx_in_raw is not None else 0
        if _start < first_lens_idx_in_raw:
            for _idx in range(_start, first_lens_idx_in_raw):
                _t = surfaces_raw[_idx].get('thickness', 0.0)
                if np.isfinite(_t):
                    obj_distance += _t
    obj_distance *= unit_scale

    # ------------------------------------------------------------------
    # Build glass sequence: track current medium between surfaces
    # ------------------------------------------------------------------
    medium_between = []
    for s in lens_surfaces:
        if s['is_mirror']:
            medium_between.append(None)
        elif s['glass'] is not None and not s['is_mirror']:
            medium_between.append(s['glass'])
        else:
            medium_between.append(None)

    # ------------------------------------------------------------------
    # Build the output element list
    # ------------------------------------------------------------------
    elements = []
    for i, s in enumerate(lens_surfaces):
        # Radius from curvature (convert units)
        curv = s['curvature']
        if abs(curv) < 1e-15:
            radius = float('inf')
        else:
            radius = (1.0 / curv) * unit_scale

        # Per-surface clear semi-diameter [m]
        semi_dia_m = s['semi_diameter'] * unit_scale

        # ----- Freeform / aspheric coefficient extraction --------------
        # The PARM block of a surface carries different things on
        # different SURFTYPEs:
        #   * STANDARD     -- no PARM (sphere/conic only).
        #   * EVENASPH     -- PARM 1..N are even-asphere coefficients
        #                     a_4, a_6, ..., a_{2*(N+1)} in lens units.
        #   * QBFS / QCON  -- PARM 1..N are Forbes Q-bfs / Q-con
        #                     orthonormal polynomial coefficients a_m
        #                     (m = parm_num - 1) in lens units of sag.
        #                     PARM 0 is the normalisation radius
        #                     r_max in lens units (Zemax "Norm Radius"
        #                     field on the Q-type surface editor);
        #                     when absent, r_max falls back to DIAM
        #                     (the surface semi-diameter).
        # Pre-v4.15.1 the loader had no QBFS/QCON branch, so any
        # Q-type prescription silently degraded to base conic plus
        # an EVENASPH-mis-interpreted PARM table.  See
        # ``lumenairy.elements.freeform.surface_sag_q_bfs`` for the
        # canonical coefficient consumer.
        stype_u = (s.get('type') or 'STANDARD').upper()
        q_freeform_type = None
        q_coeffs = None
        q_r_max = None
        asph_coeffs = None

        if stype_u in ('QBFS', 'QCON'):
            q_freeform_type = 'q_bfs' if stype_u == 'QBFS' else 'q_con'
            # Build the coefficient list a_0, a_1, ... a_{N-1} indexed
            # by Forbes order m = parm_num - 1.  Coefficients are in
            # the same sag units as DISZ / DIAM (i.e. lens units like
            # mm), so multiply by ``unit_scale`` to convert to meters.
            q_parms = s.get('aspheric_params', {}) or {}
            if q_parms:
                # Find the maximum coefficient index (>= 1).
                max_m = max(
                    (parm_num for parm_num in q_parms if parm_num >= 1),
                    default=-1)
                if max_m >= 1:
                    # Dense list a_0..a_{max_m-1} (Forbes index =
                    # parm_num - 1).  Missing entries are zero.
                    q_coeffs = [0.0] * max_m
                    for parm_num, parm_val in q_parms.items():
                        if parm_num >= 1:
                            q_coeffs[parm_num - 1] = (
                                float(parm_val) * unit_scale)
            # Normalisation radius.  Zemax convention: PARM 0 carries
            # the Norm Radius in lens units.  Fall back to DIAM (semi-
            # diameter) when PARM 0 is absent OR zero, which is how
            # the Zemax exporter writes a Q-type surface that uses
            # its default ``DIAM`` as the normalisation radius.
            r_max_parm = float(q_parms.get(0, 0.0) or 0.0)
            if r_max_parm > 0:
                q_r_max = r_max_parm * unit_scale
            elif semi_dia_m > 0 and np.isfinite(semi_dia_m):
                q_r_max = float(semi_dia_m)
            else:
                # Final fallback: half the prescription's aperture
                # diameter -- only meaningful after the aperture-
                # detection block below has run, so use a conservative
                # 1.0 m placeholder here and let downstream callers
                # validate (surface_sag_q_bfs / q_con raise on a
                # non-positive r_max).
                q_r_max = 1.0
        else:
            # Aspheric coefficients (EVENASPH and silent on STANDARD).
            # Zemax EVENASPH convention: PARM 1 = alpha_4 (dominant
            # 4th-order term), PARM 2 = alpha_6, PARM N = alpha_(2*N+2).
            # Library canonical form is a dict keyed by total power:
            # {4: a4, 6: a6, ...}.  The exporter (export_zemax_zmx)
            # round-trips with parm_idx = power//2 - 1, so loader must
            # use power = 2 + 2*parm_num and accept parm_num >= 1.
            # (Pre-v4.11.2 dropped PARM 1 entirely and mis-labelled
            # every higher coefficient by one slot.)
            if s['aspheric_params']:
                asph_coeffs = {}
                for parm_num, parm_val in s['aspheric_params'].items():
                    if parm_num >= 1:
                        power = 2 + 2 * parm_num
                        asph_coeffs[power] = (
                            parm_val / (unit_scale ** (power - 1)))

        # v4.15.1 (P1-NEW-E): pack the Forbes Q-type freeform keys
        # into a small dict that gets spread onto either the mirror
        # or refractive surface element below.
        q_extra: Dict[str, Any] = {}
        if q_freeform_type is not None:
            q_extra['freeform_type'] = q_freeform_type
            q_extra[f"{q_freeform_type}_coeffs"] = list(q_coeffs or [])
            q_extra['r_max'] = float(q_r_max)

        if s['is_mirror']:
            elements.append({
                'element_type': 'mirror',
                'radius': radius,
                'conic': s['conic'],
                'aspheric_coeffs': asph_coeffs,
                'semi_diameter': semi_dia_m,
                'surf_num': s['surf_num'],
                'comment': s.get('comment', ''),
                'is_stop': bool(s.get('is_stop', False)),
                **q_extra,
            })
        else:
            # Determine glass before and after this surface
            if i == 0:
                glass_before = 'air'
            else:
                glass_before = medium_between[i - 1] or 'air'
            glass_after = medium_between[i] or 'air'

            elements.append({
                'element_type': 'surface',
                'radius': radius,
                'conic': s['conic'],
                'aspheric_coeffs': asph_coeffs,
                'glass_before': glass_before,
                'glass_after': glass_after,
                'semi_diameter': semi_dia_m,
                'surf_num': s['surf_num'],
                'comment': s.get('comment', ''),
                'is_stop': bool(s.get('is_stop', False)),
                **q_extra,
            })

    # ------------------------------------------------------------------
    # Thicknesses between consecutive lens surfaces (convert units)
    # ------------------------------------------------------------------
    thicknesses = []
    for i in range(len(lens_surfaces) - 1):
        t = lens_surfaces[i]['thickness']
        if np.isinf(t):
            t = 0.0
        thicknesses.append(t * unit_scale)

    # ------------------------------------------------------------------
    # Aperture from the stop surface or largest semi-diameter
    # ------------------------------------------------------------------
    stop_surfaces = [s for s in lens_surfaces if s['is_stop']]
    if stop_surfaces:
        aperture = stop_surfaces[0]['semi_diameter'] * 2 * unit_scale
    else:
        aperture = max(s['semi_diameter'] for s in lens_surfaces) * 2 * unit_scale

    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]

    # ------------------------------------------------------------------
    # Build lens-only prescription (refracting surfaces only)
    # ------------------------------------------------------------------
    refr_surfaces = [e for e in elements if e['element_type'] == 'surface']
    prescription_surfaces = []
    for e in refr_surfaces:
        ps_entry = {
            'radius': e['radius'],
            'conic': e['conic'],
            'aspheric_coeffs': e['aspheric_coeffs'],
            'glass_before': e['glass_before'],
            'glass_after': e['glass_after'],
        }
        # v4.15.1 (P1-NEW-E): forward Forbes Q-type freeform keys so
        # the lens-only prescription consumed by apply_real_lens_traced
        # / surface_sag_freeform sees the coefficients and r_max
        # (pre-v4.15.1 the Q-bfs / Q-con SURFTYPE was silently dropped
        # to base conic).
        if e.get('freeform_type') is not None:
            ps_entry['freeform_type'] = e['freeform_type']
            for _qkey in ('q_bfs_coeffs', 'q_con_coeffs', 'r_max'):
                if _qkey in e:
                    ps_entry[_qkey] = e[_qkey]
        prescription_surfaces.append(ps_entry)

    # Thicknesses for the lens-only prescription (between refracting surfaces)
    refr_indices = [i for i, e in enumerate(elements)
                    if e['element_type'] == 'surface']
    lens_thicknesses = []
    for j in range(len(refr_indices) - 1):
        # Sum thicknesses between consecutive refracting surfaces
        idx_start = refr_indices[j]
        idx_end = refr_indices[j + 1]
        total_t = 0
        for k in range(idx_start, idx_end):
            if k < len(thicknesses):
                total_t += thicknesses[k]
        lens_thicknesses.append(total_t)

    # ------------------------------------------------------------------
    # Warn about unknown glasses
    # ------------------------------------------------------------------
    unknown_glasses = set()
    for e in elements:
        if e['element_type'] == 'surface':
            for g in (e['glass_before'], e['glass_after']):
                if g != 'air' and g not in GLASS_REGISTRY:
                    unknown_glasses.add(g)
    if unknown_glasses:
        # v4.16.1 (audit ORG-2 / C.7): explicit UserWarning category +
        # stacklevel=2 so the warning points at the caller (the user
        # invoking load_zemax_zmx / similar) rather than at this line.
        warnings.warn(
            f"Glasses not in GLASS_REGISTRY: {unknown_glasses}. "
            f"Add them before calling apply_real_lens. Example:\n"
            f"  GLASS_REGISTRY['GLASS_NAME'] = ('specs', 'CATALOG', 'PAGE')\n"
            f"Browse refractiveindex.info to find the correct path.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Coordinate breaks
    # ------------------------------------------------------------------
    # Extract decenter/tilt parameters from every COORDBRK surface in
    # the raw surface list.  Zemax PARM 1-6 on a COORDBRK surface are:
    #   PARM 1:  Decenter X [lens units, e.g. mm]
    #   PARM 2:  Decenter Y [lens units]
    #   PARM 3:  Tilt X (rotation about x-axis) [degrees]
    #   PARM 4:  Tilt Y (rotation about y-axis) [degrees]
    #   PARM 5:  Tilt Z (rotation about z-axis, a.k.a. roll) [degrees]
    #   PARM 6:  Order  (0 = decenter then tilt, 1 = tilt then decenter)
    #
    # Decenters are converted to meters (multiplied by ``unit_scale``);
    # tilts remain in degrees.  The loader preserves COORDBRK DISZ
    # thickness via the usual ``all_thicknesses`` path (unchanged here).
    #
    # Downstream callers (wave-optics simulations) should iterate this
    # list and apply each break at its z-position in the propagation
    # chain -- see tx_design_study_sim.py's _apply_coord_break for an
    # example implementation.
    coord_breaks = []
    for s in surfaces_raw:
        if not s.get('is_coordbrk'):
            continue
        parms = s.get('aspheric_params', {})
        cb = {
            'surf_num': s['surf_num'],
            'decenter_x_m': float(parms.get(1, 0.0)) * unit_scale,
            'decenter_y_m': float(parms.get(2, 0.0)) * unit_scale,
            'tilt_x_deg':   float(parms.get(3, 0.0)),
            'tilt_y_deg':   float(parms.get(4, 0.0)),
            'tilt_z_deg':   float(parms.get(5, 0.0)),
            'order': int(float(parms.get(6, 0.0)) or 0),
            'thickness_m':  float(s.get('thickness', 0.0) or 0.0) * (
                unit_scale if np.isfinite(s.get('thickness', 0.0))
                else 1.0),
        }
        coord_breaks.append(cb)

    return {
        'name': name,
        'aperture_diameter': aperture,
        # Lens-only prescription (for apply_real_lens)
        'surfaces': prescription_surfaces,
        'thicknesses': lens_thicknesses,
        # Full element list including mirrors (for manual use)
        'elements': elements,
        'all_thicknesses': thicknesses,
        # Distance from the stop / source plane to the first refractive
        # surface.  Non-zero when the .zmx has dummy surfaces between
        # the object and the first lens; 0 when the first lens is the
        # first surface after the object.  Callers doing wave-optics
        # propagation should apply this as free space between their
        # source (or post-MLA) field and the first lens event.
        'object_distance': obj_distance,
        # List of coordinate breaks (decenters / tilts).  Each entry is
        # a dict with keys ``surf_num``, ``decenter_x_m``,
        # ``decenter_y_m``, ``tilt_x_deg``, ``tilt_y_deg``,
        # ``tilt_z_deg``, ``order``, ``thickness_m``.  Empty list when
        # the prescription has no COORDBRK surfaces.  Sorted in
        # .zmx surface order.
        'coord_breaks': coord_breaks,
    }


# ---------------------------------------------------------------------------
# Zemax Prescription Data text export parser
# ---------------------------------------------------------------------------

def load_zemax_prescription_data_txt(filepath: str,
                                     surface_range: Optional[Tuple[int, int]] = None,
                                     name: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse a Zemax "Prescription Data" text export and return a lens prescription.

    Zemax's *Analyze -> Reports -> Prescription Data* command exports a
    tab-separated text report containing the full surface table plus
    system parameters (wavelength, units, focal length, etc.).  This
    parser reads that format and produces the same output structure as
    :func:`load_zemax_zmx` so the two loaders are interchangeable.

    The file is typically UTF-16 encoded (both BOM-marked UTF-16 and
    UTF-8 are tried automatically).

    Parameters
    ----------
    filepath : str
        Path to the prescription text file.
    surface_range : tuple of (int, int), optional
        Which Zemax surface numbers to include as lens surfaces,
        inclusive on both ends.  If None, auto-detect the first and last
        surfaces with glass or mirror.
    name : str, optional
        Human-readable label.  If None, derived from the filename.

    Returns
    -------
    prescription : dict
        Dictionary with keys:

        - ``'name'``             : human-readable label
        - ``'aperture_diameter'``: clear aperture [m]
        - ``'surfaces'``         : refracting surfaces only (for apply_real_lens)
        - ``'thicknesses'``      : thicknesses between refracting surfaces [m]
        - ``'elements'``         : full element list including mirrors
        - ``'all_thicknesses'``  : thicknesses between all elements [m]
        - ``'wavelength'``       : primary wavelength [m] (if found in header)
        - ``'units'``            : lens unit string from the header

    Notes
    -----
    Supported column format (tab-separated, one row per surface)::

        Surf  Type   Radius  Thickness  Glass  Clear Diam  Chip Zone  Mech Diam  Conic  Comment

    Surfaces tagged as ``COORDBRK`` are filtered out (they represent
    geometric transforms).  ``MIRROR`` surfaces are tagged as mirror
    elements in the ``'elements'`` list but excluded from the lens-only
    ``'surfaces'`` list.  ``DGRATING`` surfaces are treated as flat
    optical surfaces (their diffractive behavior is not modeled here).

    Unlike ``.zmx`` files, prescription text reports give the radius
    directly (not as curvature), so there are no pickup-solve parameters
    to worry about.  The "Conic" column is read directly.

    Radii, thicknesses, and diameters are converted from the report's
    native units (Millimeters by default) to meters.

    Examples
    --------
    >>> rx = load_zemax_prescription_data_txt('TXdesign-prescription.txt')
    >>> print(f"Found {len(rx['elements'])} elements")
    >>> print(f"Wavelength: {rx.get('wavelength', 0)*1e9:.0f} nm")
    """
    # Try encodings in order: UTF-16 (with BOM, most common), UTF-16-LE,
    # UTF-8, and finally latin-1 as a fallback.
    text = None
    for encoding in ('utf-16', 'utf-16-le', 'utf-8', 'latin-1'):
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                candidate = f.read()
            if 'SURFACE DATA SUMMARY' in candidate:
                text = candidate
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        raise IOError(
            f"Could not read {filepath} with any supported encoding "
            f"(tried utf-16, utf-16-le, utf-8, latin-1)."
        )

    text = text.lstrip('﻿')  # strip BOM if present

    # ---------------------------------------------------------------
    # Parse header metadata (wavelength, units, stop radius, etc.)
    # ---------------------------------------------------------------
    wavelength_m = None
    unit_scale = 1e-3        # default: millimeters
    unit_name = 'Millimeters'
    for line in text.split('\n'):
        s = line.strip()
        if not s or ':' not in s:
            continue
        # "Primary Wavelength [µm] :  1.31"
        if 'Primary Wavelength' in s:
            try:
                val = s.split(':', 1)[1].strip().split()[0]
                wavelength_m = float(val) * 1e-6  # µm -> m
            except (ValueError, IndexError):
                pass
        # "Lens Units              :   Millimeters"
        elif 'Lens Units' in s:
            try:
                unit_name = s.split(':', 1)[1].strip()
                unit_map = {
                    'Millimeters': 1e-3,
                    'Centimeters': 1e-2,
                    'Meters': 1.0,
                    'Inches': 25.4e-3,
                }
                unit_scale = unit_map.get(unit_name, 1e-3)
            except (ValueError, IndexError):
                pass

    # ---------------------------------------------------------------
    # Locate the SURFACE DATA SUMMARY table
    # ---------------------------------------------------------------
    start = text.find('SURFACE DATA SUMMARY')
    if start < 0:
        raise ValueError(
            f"{filepath} does not contain a 'SURFACE DATA SUMMARY' section."
        )
    # End of table: the next "SURFACE DATA DETAIL" or "EDGE THICKNESS"
    end_markers = ('SURFACE DATA DETAIL', 'EDGE THICKNESS DATA',
                   'MULTI-CONFIGURATION DATA')
    end = len(text)
    for marker in end_markers:
        idx = text.find(marker, start + 1)
        if idx > 0 and idx < end:
            end = idx
    table = text[start:end]

    # Locate the column header row and parse rows after it
    lines = table.split('\n')
    header_idx = None
    for i, line in enumerate(lines):
        if 'Surf' in line and 'Type' in line and 'Radius' in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find column header row in surface table.")

    # ---------------------------------------------------------------
    # Parse each surface row
    # ---------------------------------------------------------------
    # Columns (tab-separated):
    #   Surf, Type, Radius, Thickness, Glass, Clear Diam, Chip Zone,
    #   Mech Diam, Conic, Comment
    surfaces_raw = []
    last_surf_num = -1

    for raw in lines[header_idx + 1:]:
        line = raw.rstrip()
        if not line.strip():
            continue

        fields = [f.strip() for f in line.split('\t')]
        if len(fields) < 9:
            # Not a surface row (blank separator, continuation, etc.)
            continue

        surf_label = fields[0]
        type_str = fields[1]
        radius_str = fields[2]
        thickness_str = fields[3]
        glass_str = fields[4] if len(fields) > 4 else ''
        clear_diam_str = fields[5] if len(fields) > 5 else ''
        conic_str = fields[8] if len(fields) > 8 else '0'
        comment = fields[9] if len(fields) > 9 else ''

        # Map OBJ / STO / IMA / numeric
        if surf_label == 'OBJ':
            surf_num = 0
        elif surf_label == 'STO':
            surf_num = last_surf_num + 1
        elif surf_label == 'IMA':
            surf_num = last_surf_num + 1
        else:
            try:
                surf_num = int(surf_label)
            except ValueError:
                continue  # skip malformed rows
        last_surf_num = max(last_surf_num, surf_num)

        # Parse numeric fields (handle "Infinity" and "-" placeholders)
        def _parse_float(s, default=0.0):
            s = s.strip()
            if not s or s == '-':
                return default
            if s.lower() in ('infinity', 'inf'):
                return float('inf')
            try:
                return float(s)
            except ValueError:
                return default

        radius = _parse_float(radius_str, float('inf'))
        thickness = _parse_float(thickness_str, 0.0)
        # The "Clear Diam" column in the prescription text report is the
        # full diameter, not semi-diameter.  Divide by 2 so the internal
        # representation matches the .zmx parser (which reads DIAM as
        # semi-diameter directly).
        semi_diameter = _parse_float(clear_diam_str, 0.0) / 2.0
        conic = _parse_float(conic_str, 0.0)

        glass = glass_str if glass_str else None
        is_mirror = glass is not None and glass.upper() == 'MIRROR'
        is_coordbrk = type_str.upper() == 'COORDBRK'
        is_stop = surf_label == 'STO'

        # Convert to meters
        if not np.isinf(radius):
            radius = radius * unit_scale
        thickness = thickness * unit_scale
        semi_diameter = semi_diameter * unit_scale

        surfaces_raw.append({
            'surf_num': surf_num,
            'surf_label': surf_label,
            'type': type_str,
            'radius': radius,
            'conic': conic,
            'thickness': thickness,
            'glass': glass,
            'semi_diameter': semi_diameter,
            'aspheric_params': {},
            'is_stop': is_stop,
            'is_mirror': is_mirror,
            'is_coordbrk': is_coordbrk,
            'comment': comment,
        })

    # ---------------------------------------------------------------
    # Filter out coordinate breaks (non-optical) and pick lens surfaces
    # ---------------------------------------------------------------
    optical_surfaces = [s for s in surfaces_raw if not s['is_coordbrk']]

    if surface_range is not None:
        s_first, s_last = surface_range
        lens_surfaces = [s for s in optical_surfaces
                         if s_first <= s['surf_num'] <= s_last]
    else:
        active = [s for s in optical_surfaces
                  if s['glass'] is not None or s['is_mirror']]
        if not active:
            raise ValueError(f"No glass/mirror surfaces found in {filepath}")
        s_first = active[0]['surf_num']
        s_last = active[-1]['surf_num'] + 1
        lens_surfaces = [s for s in optical_surfaces
                         if s_first <= s['surf_num'] <= s_last]

    if len(lens_surfaces) < 2:
        raise ValueError(
            f"Need at least 2 surfaces, got {len(lens_surfaces)} "
            f"in range ({s_first}, {s_last})"
        )

    # Object-space distance (see load_zemax_zmx for rationale).
    # Sum ``thickness`` values from the STOP surface (treated as the
    # source plane) up to but not including the first refractive
    # surface.  If no STOP is present, sum from SURF 0 onward.
    stop_idx_in_raw = None
    for _idx, _s in enumerate(surfaces_raw):
        if _s.get('is_stop'):
            stop_idx_in_raw = _idx
            break
    first_lens_surf_num = lens_surfaces[0]['surf_num']
    first_lens_idx_in_raw = next(
        (_idx for _idx, _s in enumerate(surfaces_raw)
         if _s['surf_num'] == first_lens_surf_num),
        None)
    obj_distance = 0.0
    if first_lens_idx_in_raw is not None:
        _start = stop_idx_in_raw if stop_idx_in_raw is not None else 0
        if _start < first_lens_idx_in_raw:
            for _idx in range(_start, first_lens_idx_in_raw):
                _t = surfaces_raw[_idx].get('thickness', 0.0)
                if np.isfinite(_t):
                    obj_distance += _t
    # Note: .txt-loader thicknesses are already in meters (no unit_scale).

    # Track the glass medium between each consecutive surface pair
    medium_between = []
    for s in lens_surfaces:
        if s['is_mirror']:
            medium_between.append(None)
        elif s['glass'] is not None and not s['is_mirror']:
            medium_between.append(s['glass'])
        else:
            medium_between.append(None)

    # Build the output element list
    elements = []
    for i, s in enumerate(lens_surfaces):
        if s['is_mirror']:
            elements.append({
                'element_type': 'mirror',
                'radius': s['radius'],
                'conic': s['conic'],
                'aspheric_coeffs': None,
                'semi_diameter': s['semi_diameter'],
                'surf_num': s['surf_num'],
                'comment': s.get('comment', ''),
                'is_stop': bool(s.get('is_stop', False)),
            })
        else:
            if i == 0:
                glass_before = 'air'
            else:
                glass_before = medium_between[i - 1] or 'air'
            glass_after = medium_between[i] or 'air'

            elements.append({
                'element_type': 'surface',
                'radius': s['radius'],
                'conic': s['conic'],
                'aspheric_coeffs': None,
                'glass_before': glass_before,
                'glass_after': glass_after,
                'semi_diameter': s['semi_diameter'],
                'surf_num': s['surf_num'],
                'comment': s.get('comment', ''),
                'is_stop': bool(s.get('is_stop', False)),
            })

    # All-element thicknesses (one fewer than elements)
    thicknesses = []
    for i in range(len(lens_surfaces) - 1):
        t = lens_surfaces[i]['thickness']
        if np.isinf(t):
            t = 0.0
        thicknesses.append(t)

    # Aperture from the stop surface or largest semi-diameter
    stop_surfaces = [s for s in lens_surfaces if s['is_stop']]
    if stop_surfaces:
        aperture = stop_surfaces[0]['semi_diameter'] * 2
    else:
        aperture = max(s['semi_diameter'] for s in lens_surfaces) * 2

    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]

    # Build the lens-only prescription (refracting surfaces only)
    refr_surfaces = [e for e in elements if e['element_type'] == 'surface']
    prescription_surfaces = [
        {
            'radius': e['radius'],
            'conic': e['conic'],
            'aspheric_coeffs': e['aspheric_coeffs'],
            'glass_before': e['glass_before'],
            'glass_after': e['glass_after'],
        }
        for e in refr_surfaces
    ]

    # Thicknesses between refracting surfaces only
    refr_indices = [i for i, e in enumerate(elements) if e['element_type'] == 'surface']
    lens_thicknesses = []
    for j in range(len(refr_indices) - 1):
        idx_start = refr_indices[j]
        idx_end = refr_indices[j + 1]
        total_t = sum(thicknesses[k] for k in range(idx_start, idx_end)
                      if k < len(thicknesses))
        lens_thicknesses.append(total_t)

    # Warn about unknown glasses
    unknown_glasses = set()
    for e in elements:
        if e['element_type'] == 'surface':
            for g in (e['glass_before'], e['glass_after']):
                if g != 'air' and g not in GLASS_REGISTRY:
                    unknown_glasses.add(g)
    if unknown_glasses:
        # v4.16.1 (audit ORG-2 / C.7): explicit UserWarning category +
        # stacklevel=2 so the warning points at the caller rather than
        # at this line.
        warnings.warn(
            f"Glasses not in GLASS_REGISTRY: {unknown_glasses}. "
            f"Add them before calling apply_real_lens. Example:\n"
            f"  GLASS_REGISTRY['GLASS_NAME'] = ('specs', 'CATALOG', 'PAGE')\n"
            f"Browse refractiveindex.info to find the correct path.",
            UserWarning,
            stacklevel=2,
        )

    return {
        'name': name,
        'aperture_diameter': aperture,
        # Lens-only prescription (for apply_real_lens)
        'surfaces': prescription_surfaces,
        'thicknesses': lens_thicknesses,
        # Full element list including mirrors (for manual use)
        'elements': elements,
        'all_thicknesses': thicknesses,
        # Distance from the stop / source plane to the first refractive
        # surface.  See load_zemax_zmx for rationale.
        'object_distance': obj_distance,
        # Metadata from header
        'wavelength': wavelength_m,
        'units': unit_name,
    }


# ============================================================================
# Zemax export
# ============================================================================
#
# These helpers write a lens prescription out in two forms that are
# useful for cross-verifying wave simulations against Zemax
# OpticStudio:
#
#   1. A human-readable LDE-style text table that can be typed (or
#      column-copy-pasted) into the Zemax Lens Data Editor.
#
#   2. A minimal ``.zmx`` sequential file that Zemax can import with
#      File > Open.  The generated file is intentionally minimal: it
#      defines only the surface table, wavelength, aperture and field,
#      using Zemax defaults for everything else.  After loading you may
#      want to verify the APERTURE settings (Clear Semi-Diameter
#      floating vs. fixed) and the STOP location to match your
#      experimental conditions.
#
# Sign convention matches Zemax's default (and our library's): positive
# radius of curvature means the centre of curvature lies to the right
# of the surface vertex.


def export_zemax_lens_data(prescription: Dict[str, Any], path: str, *,
                           wavelength: float,
                           stop_surface: int = 0,
                           aperture_diameter: Optional[float] = None,
                           back_focal_length: Optional[float] = None,
                           description: Optional[str] = None,
                           extra_notes: Optional[str] = None) -> None:
    """Write a human-readable Zemax-LDE-style text table for a lens
    prescription.

    The resulting file is easy to eyeball and can be transcribed into
    Zemax OpticStudio by hand.  For direct import, see
    :func:`export_zemax_zmx`.

    Parameters
    ----------
    prescription : dict
        Prescription dict with keys ``'surfaces'`` and ``'thicknesses'``
        (see :func:`make_singlet`).
    path : str
        Output file path (``.txt`` recommended).
    wavelength : float, default 1.31e-6
        Primary wavelength [m] to record in the file header.
    stop_surface : int, default 0
        Zero-based index of the aperture stop within the refracting
        surface list.
    aperture_diameter : float, optional
        Clear aperture diameter [m].  Falls back to
        ``prescription.get('aperture_diameter')``.
    back_focal_length : float, optional
        BFL [m] to insert between the last refracting surface and the
        image plane.  If ``None``, ``0.0`` is written (user should set
        by eye in Zemax).
    description : str, optional
        Free-form description written at the top of the file.
    extra_notes : list of str, optional
        Additional lines appended to the header as comments.

    Notes
    -----
    Column units: radii and thicknesses in *millimeters*, diameters in
    *millimeters*, conic dimensionless.  Glass strings are written as
    they appear in the prescription.  Infinite radii are rendered as
    ``Infinity`` (matching Zemax's text convention).
    """
    surfaces = prescription['surfaces']
    thicknesses = prescription['thicknesses']
    if aperture_diameter is None:
        aperture_diameter = prescription.get('aperture_diameter', 25.4e-3)
    semi_dia_mm = 0.5 * aperture_diameter * 1e3
    bfl_mm = (back_focal_length * 1e3) if back_focal_length else 0.0
    name = prescription.get('name', os.path.splitext(
        os.path.basename(path))[0])

    def _fmt_radius(R):
        return 'Infinity' if (R is None or np.isinf(R)) else f'{R*1e3:.6f}'

    lines = []
    lines.append(f'# Zemax-compatible lens data for: {name}')
    if description:
        lines.append(f'# Description: {description}')
    lines.append('#')
    lines.append('# Test conditions')
    lines.append(f'#   Primary wavelength: {wavelength*1e6:.4f} um')
    lines.append('#   Source: collimated on-axis plane wave')
    lines.append(f'#   Aperture: clear semi-diameter = '
                 f'{semi_dia_mm:.4f} mm (diameter {aperture_diameter*1e3:.4f} mm)')
    lines.append(f'#   Stop surface index: {stop_surface + 1}')
    if extra_notes:
        for note in extra_notes:
            lines.append(f'#   {note}')
    lines.append('#')
    lines.append('# Paste into the Zemax Lens Data Editor (Sequential mode)')
    lines.append('# Columns: SURF | TYPE | RADIUS [mm] | THICKNESS [mm] '
                 '| MATERIAL | SEMI-DIA [mm] | CONIC | COMMENT')
    lines.append('#')
    # Header row for the table
    lines.append(
        '# {0:4s} {1:11s} {2:>16s} {3:>16s} {4:>10s} {5:>10s} {6:>8s}  {7}'
        .format('SURF', 'TYPE', 'RADIUS', 'THICKNESS',
                'MATERIAL', 'SEMI-DIA', 'CONIC', 'COMMENT'))

    # Object surface
    lines.append(
        '  {0:4s} {1:11s} {2:>16s} {3:>16s} {4:>10s} {5:>10s} {6:>8s}  {7}'
        .format('OBJ', 'STANDARD', 'Infinity', 'Infinity',
                '--', '0.000', '0.000', 'Object at infinity'))

    # Refracting surfaces
    for i, surf in enumerate(surfaces):
        label = 'STO' if i == stop_surface else str(i + 1)
        stop_mark = ' * ' if i == stop_surface else '   '
        rad = _fmt_radius(surf.get('radius'))
        # Thickness after this surface: between-element spacing, or
        # BFL after the last surface.
        if i < len(thicknesses):
            t_mm = thicknesses[i] * 1e3
        else:
            t_mm = bfl_mm
        t_str = f'{t_mm:.6f}'
        glass = surf.get('glass_after', '')
        if not glass or glass.lower() in ('air', ''):
            glass = '--'
        conic = surf.get('conic', 0.0) or 0.0
        comment = surf.get('comment', '')
        # Per-surface semi-diameter wins over the global aperture/2.
        # This is what lets aperture overrides (commonly used to widen
        # individual lens housings for an off-axis fan-out simulation)
        # round-trip through the exported file.
        sd_surf = surf.get('semi_diameter')
        sd_mm = float(sd_surf) * 1e3 if sd_surf is not None else semi_dia_mm
        lines.append(
            '{mark}{surf:4s} {tp:11s} {rad:>16s} {th:>16s} '
            '{gl:>10s} {sd:>10.4f} {con:>8.4f}  {cm}'
            .format(mark=stop_mark, surf=label, tp='STANDARD',
                    rad=rad, th=t_str, gl=glass, sd=sd_mm,
                    con=float(conic), cm=comment or f'surface {i+1}'))

    # Image plane
    lines.append(
        '   {0:4s} {1:11s} {2:>16s} {3:>16s} {4:>10s} {5:>10.4f} {6:>8.4f}  {7}'
        .format('IMA', 'STANDARD', 'Infinity', '0.000000',
                '--', 0.0, 0.0, 'Image plane'))

    lines.append('#')
    lines.append('# Legend: "*" marks the aperture stop.')
    lines.append('# To verify OPD in Zemax: Analysis > Wavefront > '
                 'Wavefront Map, or Analysis > Aberrations > Optical '
                 'Path Difference (OPD fan).')
    lines.append('# Remember to set Aperture Type -> "Float By Stop '
                 'Size" and the primary wavelength to match the value '
                 'listed above.')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def _export_zemax_zmx_full(prescription, path, wavelength=1.31e-6,
                            stop_surface=0, aperture_diameter=None,
                            back_focal_length=None, name=None):
    """3.7.0: cb/mirror-aware .zmx writer.

    Walks ``prescription['elements']`` (full chronological list with
    mirrors) and ``prescription['coord_breaks']``, emitting
    ``TYPE COORDBRK`` surfaces with ``PARM 1..6`` filled in,
    ``GLAS MIRROR`` rows for reflective surfaces, and standard
    refractive surfaces with the appropriate ``GLAS`` and curvature.

    v4.11.2: the previous 3.6.1-hotfix-6 mirror-parity sign flip was
    removed.  The canonical thickness convention (since GUI v3.7.4)
    is **Zemax-signed**: thicknesses stored in ``all_thicknesses``
    and ``coord_breaks[i]['thickness_m']`` already carry Zemax sign
    (negative after a mirror).  Applying a parity flip on export
    therefore double-negated thicknesses on mirror legs (mirror DISZ
    came back positive instead of negative; coord-break DISZ after a
    mirror was inverted twice).

    Used by :func:`export_zemax_zmx` when the prescription dict
    carries the new keys; the pre-3.7 lens-only path remains the
    fallback.
    """
    full = prescription['elements']
    cbs = prescription.get('coord_breaks') or []
    all_thicknesses = (prescription.get('all_thicknesses')
                        or prescription.get('thicknesses') or [])
    if aperture_diameter is None:
        aperture_diameter = prescription.get('aperture_diameter', 25.4e-3)
    name = name or prescription.get('name',
        os.path.splitext(os.path.basename(path))[0])
    epd_mm = aperture_diameter * 1e3
    wvl_um = wavelength * 1e6
    semi_dia_mm = 0.5 * aperture_diameter * 1e3

    # Index cb's by surf_num (after which Zemax surface they sit).
    cb_at_surfnum = {}
    for cb in cbs:
        try:
            sn = int(cb.get('surf_num', -1))
        except (TypeError, ValueError):
            continue
        cb_at_surfnum.setdefault(sn, []).append(cb)

    lines = []
    lines.append('VERS 210000 0 123 0 0')
    lines.append('MODE SEQ')
    lines.append(f'NAME {name}')
    lines.append('UNIT MM X W X CM MR CPMM')
    lines.append(f'ENPD {epd_mm:.8f}')
    lines.append('ENVD 2.0e+01 1 0')
    lines.append('GFAC 0 0')
    lines.append('GCAT SCHOTT MISC')
    lines.append('RAIM 0 0 1 1 0 0 0 0 0')
    lines.append('PUSH 0 0 0 0 0 0')
    lines.append('SDMA 0 1 0')
    lines.append('FTYP 0 0 1 1 0 0 0')
    lines.append('ROPD 2')
    lines.append('PICB 1')
    lines.append('XFLN 0')
    lines.append('YFLN 0')
    lines.append('FWGN 1')
    lines.append('VDXN 0')
    lines.append('VDYN 0')
    lines.append('VCXN 0')
    lines.append('VCYN 0')
    lines.append('VANN 0')
    lines.append(f'WAVM 1 {wvl_um:.6f} 1.0')
    lines.append('PWAV 1')

    def _zemax_curv(R):
        return 0.0 if (R is None or np.isinf(R)) else 1.0 / (R * 1e3)

    # SURF 0: object at infinity
    lines.append('SURF 0')
    lines.append('  TYPE STANDARD')
    lines.append('  CURV 0 0 0 0 0 ""')
    lines.append('  DISZ INFINITY')
    lines.append(f'  DIAM {semi_dia_mm:.6f} 0 0 0 1 ""')

    # Walk the elements, interleaving cb's where they belong.
    # The GUI's to_prescription tags each cb with the surf_num it
    # sits at (i.e. the running counter increments AT the cb).
    surf_counter = 0   # we just emitted SURF 0 (object)
    # mirror_count is retained for diagnostics only.  Pre-v4.11.2 it
    # drove a thickness sign-flip; that flip was removed because the
    # canonical thicknesses are already Zemax-signed.
    mirror_count = 0

    # Build a flat list of (item_kind, payload, thickness_after_m)
    # in the order the importer's output puts them, then emit SURFs.
    flat = []
    cb_iter = list(cbs)
    cb_idx = 0
    for ei, e in enumerate(full):
        e_surf_num = int(e.get('surf_num', surf_counter + 1))
        # Drain any cb's that target a surf <= e_surf_num.
        while cb_idx < len(cb_iter):
            sn = int(cb_iter[cb_idx].get('surf_num', 0))
            if sn < e_surf_num:
                flat.append(('cb', cb_iter[cb_idx], None))
                cb_idx += 1
            else:
                break
        flat.append(('elem', e, None))
    while cb_idx < len(cb_iter):
        flat.append(('cb', cb_iter[cb_idx], None))
        cb_idx += 1

    # Emit surfaces.  The thickness AFTER an element/cb comes from
    # all_thicknesses indexed by element index; cb's get thickness 0
    # (they don't advance, just transform the frame).
    elem_idx_in_full = 0
    # Track the refracting-surface index separately so the STOP
    # marker lands on the requested refracting surface even when
    # coord-breaks and mirrors appear earlier in ``flat``.  The
    # ``stop_surface`` parameter is documented as "zero-based index
    # of the aperture stop **among refracting surfaces**".  Pre-v4.11.2
    # this compared the global ``surf_counter`` (which includes
    # coord-breaks and mirrors) so folded designs placed STOP on the
    # wrong row.
    refr_counter = -1
    for kind, payload, _ in flat:
        surf_counter += 1
        if kind == 'cb':
            cb = payload
            # COORDBRK entry.  PARM 1=decX_mm, 2=decY_mm,
            # 3=tiltX_deg, 4=tiltY_deg, 5=tiltZ_deg, 6=order.
            dx_mm = float(cb.get('decenter_x_m', 0.0)) * 1e3
            dy_mm = float(cb.get('decenter_y_m', 0.0)) * 1e3
            tx = float(cb.get('tilt_x_deg', 0.0))
            ty = float(cb.get('tilt_y_deg', 0.0))
            tz = float(cb.get('tilt_z_deg', 0.0))
            order = int(cb.get('order', 0) or 0)
            disz_mm = float(cb.get('thickness_m', 0.0)) * 1e3
            # v4.11.2: no mirror-parity flip -- ``thickness_m`` is
            # already Zemax-signed (loader copies raw DISZ * unit_scale
            # without sign manipulation).
            lines.append(f'SURF {surf_counter}')
            lines.append('  TYPE COORDBRK')
            lines.append('  CURV 0.0 0 0.0 0.0 0')
            lines.append(f'  PARM 1 {dx_mm:.10g}')
            lines.append(f'  PARM 2 {dy_mm:.10g}')
            lines.append(f'  PARM 3 {tx:.10g}')
            lines.append(f'  PARM 4 {ty:.10g}')
            lines.append(f'  PARM 5 {tz:.10g}')
            lines.append(f'  PARM 6 {order}')
            lines.append(f'  DISZ {disz_mm:.8f}')
            lines.append('  DIAM 0 0 0 0 1 ""')
            continue

        # Element: refractive or mirror.  Thickness AFTER this elem
        # comes from all_thicknesses[elem_idx_in_full].
        e = payload
        t_after_m = (all_thicknesses[elem_idx_in_full]
                      if elem_idx_in_full < len(all_thicknesses) else 0.0)
        elem_idx_in_full += 1
        # v4.11.2: no mirror-parity flip.  Pre-fix code converted
        # "physical-positive" back to Zemax-signed by negating every
        # thickness after each mirror; but the loader stores raw
        # Zemax-signed DISZ (no conversion) and the GUI (v3.7.4+)
        # keeps Zemax-signed canonical, so the flip was always
        # spurious here and destroyed mirror DISZ on round-trip.
        disz_mm = t_after_m * 1e3

        e_type = e.get('element_type', 'surface')
        R_m = e.get('radius', float('inf'))
        conic = float(e.get('conic', 0.0) or 0.0)
        sd_m = e.get('semi_diameter')
        sd_mm_e = (float(sd_m) * 1e3
                    if (sd_m is not None and np.isfinite(sd_m))
                    else semi_dia_mm)
        comment = (e.get('comment') or '').strip()
        curv_val = _zemax_curv(R_m)
        lines.append(f'SURF {surf_counter}')
        if comment:
            lines.append(f'  COMM {comment}')
        if e_type == 'mirror':
            lines.append('  TYPE STANDARD')
            lines.append(f'  CURV {curv_val:.10f} 0 0 0 0 ""')
            if conic != 0.0:
                lines.append(f'  CONI {conic:.6f}')
            lines.append('  GLAS MIRROR 0 0 1.5 50.0 0 0 0 0 0 0')
            lines.append(f'  DISZ {disz_mm:.8f}')
            lines.append(f'  DIAM {sd_mm_e:.6f} 0 0 0 1 ""')
            mirror_count += 1
        else:
            refr_counter += 1
            lines.append('  TYPE STANDARD')
            if refr_counter == stop_surface:
                lines.append('  STOP')
            lines.append(f'  CURV {curv_val:.10f} 0 0 0 0 ""')
            lines.append(f'  DISZ {disz_mm:.8f}')
            if conic != 0.0:
                lines.append(f'  CONI {conic:.6f}')
            glass_after = e.get('glass_after', 'air')
            if glass_after and glass_after.lower() not in ('air', ''):
                lines.append(
                    f'  GLAS {glass_after} 0 0 1.5 50.0 0 0 0 0 0 0')
            asph = e.get('aspheric_coeffs') or {}
            if asph:
                # Replace TYPE STANDARD line above with EVENASPH.
                for j in range(len(lines) - 1, -1, -1):
                    if lines[j] == '  TYPE STANDARD':
                        lines[j] = '  TYPE EVENASPH'
                        break
                for power in sorted(asph.keys()):
                    if power <= 0 or power % 2 != 0:
                        continue
                    parm_idx = power // 2 - 1
                    coeff_mm = asph[power] * (1e3 ** (1 - power))
                    lines.append(f'  PARM {parm_idx} {coeff_mm:.10e}')
            lines.append(f'  DIAM {sd_mm_e:.6f} 0 0 0 1 ""')

    # Image surface
    surf_counter += 1
    lines.append(f'SURF {surf_counter}')
    lines.append('  TYPE STANDARD')
    lines.append('  CURV 0 0 0 0 0 ""')
    lines.append('  DISZ 0.0')
    lines.append(f'  DIAM {semi_dia_mm:.6f} 0 0 0 1 ""')

    lines.append('BLNK ')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def export_zemax_zmx(prescription: Dict[str, Any], path: str, *,
                     wavelength: float,
                     stop_surface: int = 0,
                     aperture_diameter: Optional[float] = None,
                     back_focal_length: Optional[float] = None,
                     name: Optional[str] = None) -> None:
    """Write a minimal Zemax OpticStudio ``.zmx`` sequential file for a
    prescription.

    The produced file contains surface data, one wavelength, and an
    on-axis field.  Other Zemax settings use defaults; you may need to
    tweak glass catalogs or aperture conventions after opening the file.

    3.7.0: when the prescription carries the full ``elements`` list
    (with mirrors) and ``coord_breaks`` list, the writer emits
    matching ``GLAS MIRROR`` and ``TYPE COORDBRK`` entries with the
    correct PARM 1-6 fields so a round-trip ``load_zemax_zmx``
    -> GUI edit -> ``export_zemax_zmx`` preserves fold geometry.  The
    pre-3.7 lens-only path (``surfaces`` + ``thicknesses``) remains
    the fallback when these keys are absent.

    Parameters
    ----------
    prescription : dict
        As for :func:`export_zemax_lens_data`.
    path : str
        Output ``.zmx`` file path.
    wavelength : float
        Wavelength [m] recorded as the primary.
    stop_surface : int
        Zero-based index of the aperture stop among refracting surfaces.
    aperture_diameter : float, optional
        Entrance pupil diameter in meters; falls back to
        ``prescription['aperture_diameter']``.
    back_focal_length : float, optional
        BFL [m] between the last refracting surface and the image plane.
        Defaults to zero (user must adjust).
    name : str, optional
        Lens name recorded in the file header.

    Notes
    -----
    Zemax's ``.zmx`` format has evolved over versions; this writer
    targets a format accepted by recent OpticStudio releases for
    sequential systems.  If your version refuses the file, start a new
    session in Zemax and manually enter the rows from
    :func:`export_zemax_lens_data` instead.
    """
    # 3.7.0: prefer the full chronological list when present (it
    # carries mirrors and is aligned with all_thicknesses + the
    # coord_breaks list).  Fall back to the lens-only path otherwise.
    full_elements = prescription.get('elements')
    coord_breaks = prescription.get('coord_breaks') or []
    if full_elements is not None and (coord_breaks
                                       or any(e.get('element_type') == 'mirror'
                                              for e in full_elements)):
        return _export_zemax_zmx_full(
            prescription, path, wavelength=wavelength,
            stop_surface=stop_surface,
            aperture_diameter=aperture_diameter,
            back_focal_length=back_focal_length, name=name)

    surfaces = prescription['surfaces']
    thicknesses = prescription['thicknesses']
    if aperture_diameter is None:
        aperture_diameter = prescription.get('aperture_diameter', 25.4e-3)
    bfl = back_focal_length or 0.0

    name = name or prescription.get('name',
        os.path.splitext(os.path.basename(path))[0])
    # Zemax EPD is in mm
    epd_mm = aperture_diameter * 1e3
    wvl_um = wavelength * 1e6

    lines = []
    lines.append('VERS 210000 0 123 0 0')
    lines.append('MODE SEQ')
    lines.append(f'NAME {name}')
    lines.append('UNIT MM X W X CM MR CPMM')
    lines.append(f'ENPD {epd_mm:.8f}')
    lines.append('ENVD 2.0e+01 1 0')
    lines.append('GFAC 0 0')
    lines.append('GCAT SCHOTT MISC')
    lines.append('RAIM 0 0 1 1 0 0 0 0 0')
    lines.append('PUSH 0 0 0 0 0 0')
    lines.append('SDMA 0 1 0')
    lines.append('FTYP 0 0 1 1 0 0 0')
    lines.append('ROPD 2')
    lines.append('PICB 1')
    lines.append('XFLN 0')
    lines.append('YFLN 0')
    lines.append('FWGN 1')
    lines.append('VDXN 0')
    lines.append('VDYN 0')
    lines.append('VCXN 0')
    lines.append('VCYN 0')
    lines.append('VANN 0')
    lines.append(f'WAVM 1 {wvl_um:.6f} 1.0')
    lines.append('PWAV 1')

    def _zemax_curv(R):
        return 0.0 if (R is None or np.isinf(R)) else 1.0 / (R * 1e3)

    def _zemax_disz(t_m):
        return t_m * 1e3

    semi_dia_mm = 0.5 * aperture_diameter * 1e3

    # SURF 0: object at infinity
    lines.append('SURF 0')
    lines.append('  TYPE STANDARD')
    lines.append('  CURV 0 0 0 0 0 ""')
    lines.append('  DISZ INFINITY')
    lines.append(f'  DIAM {semi_dia_mm:.6f} 0 0 0 1 ""')

    # Refracting surfaces
    for i, surf in enumerate(surfaces):
        idx = i + 1
        R = surf.get('radius')
        conic = surf.get('conic', 0.0) or 0.0
        glass = surf.get('glass_after', '')
        if glass and glass.lower() not in ('air', ''):
            glass_line = f'  GLAS {glass} 0 0 1.5 50.0 0 0 0 0 0 0'
        else:
            glass_line = None

        t_m = thicknesses[i] if i < len(thicknesses) else bfl
        disz_val = _zemax_disz(t_m)
        curv_val = _zemax_curv(R)

        # Per-surface semi-diameter wins over the global aperture/2.
        sd_surf = surf.get('semi_diameter')
        sd_mm = float(sd_surf) * 1e3 if sd_surf is not None else semi_dia_mm

        lines.append(f'SURF {idx}')
        lines.append('  TYPE STANDARD')
        if i == stop_surface:
            lines.append('  STOP')
        lines.append(f'  CURV {curv_val:.10f} 0 0 0 0 ""')
        lines.append(f'  DISZ {disz_val:.8f}')
        if conic != 0.0:
            lines.append(f'  CONI {conic:.6f}')
        if glass_line is not None:
            lines.append(glass_line)
        # Even-aspheric polynomial coefficients (a_4 h^4 + a_6 h^6 + ...).
        # Zemax's even-aspheric uses TYPE EVENASPH and PARM 1..N.  When
        # the prescription has aspheric_coeffs, switch the surface type
        # so the coefficients survive a Zemax round-trip.
        asph = surf.get('aspheric_coeffs') or {}
        if asph:
            # Replace the TYPE STANDARD line above (just emitted).
            for j in range(len(lines) - 1, -1, -1):
                if lines[j] == '  TYPE STANDARD':
                    lines[j] = '  TYPE EVENASPH'
                    break
            for power in sorted(asph.keys()):
                if power <= 0 or power % 2 != 0:
                    continue
                # Zemax PARM index: a_4 -> 1, a_6 -> 2, a_8 -> 3, ...
                parm_idx = power // 2 - 1
                # Coefficient unit: input is 1/m^(power-1) for our convention,
                # Zemax expects 1/mm^(power-1).  Convert.
                coeff_mm = asph[power] * (1e3 ** (1 - power))
                lines.append(f'  PARM {parm_idx} {coeff_mm:.10e}')
        lines.append(f'  DIAM {sd_mm:.6f} 0 0 0 1 ""')

    # Image surface at BFL after last refracting surface
    last_idx = len(surfaces) + 1
    lines.append(f'SURF {last_idx}')
    lines.append('  TYPE STANDARD')
    lines.append('  CURV 0 0 0 0 0 ""')
    lines.append('  DISZ 0.0')
    lines.append(f'  DIAM {semi_dia_mm:.6f} 0 0 0 1 ""')

    lines.append('BLNK ')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

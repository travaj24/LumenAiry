"""
Quadoa Optikos ``.qos`` (JSON) prescription I/O -- best-effort.

Quadoa Optikos uses a JSON-based ``.qos`` system file.  The full schema
is not publicly documented at the level of every field, so the
exporter writes a self-defined JSON layout that captures every
field a lumenairy prescription holds (surfaces, glasses, thicknesses,
aperture, conics, asphere coefficients, biconic Y-axis radii, stop
index, wavelength, name, units, semi-diameters).  Importer round-trips
this layout exactly.  When Quadoa publishes a stable schema -- or when
users supply a reference ``.qos`` -- this can be tightened.  The
JSON-writer side is intentionally schema-versioned so future readers
can detect the layout.

v5.1.0 split (Agent F): extracted from ``lumenairy/io/prescriptions.py``
without any logic change.  Public API is unchanged: every name in this
module is re-exported through ``lumenairy.io.prescriptions`` and
``lumenairy``.

Author: Andrew Traverso
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional

import numpy as np

# ============================================================================
# Quadoa Optikos .qos (JSON) file I/O -- best-effort
# ============================================================================

QUADOA_SCHEMA_VERSION = '1.0'


def _quadoa_serialize_radius(R, scale):
    if R is None or not np.isfinite(R):
        return None
    return float(R) * scale


def _quadoa_serialize_aspheric(coeffs, scale=1.0):
    """Serialise a library aspheric_coeffs dict ``{4: a4, 6: a6, ...}``
    as a JSON-friendly dict with string keys (JSON requires string
    keys).  ``None`` -> ``None``.  Pre-v4.11.2 this iterated dict keys
    as if they were values, writing the powers [4.0, 6.0, ...] instead
    of the coefficients.

    v5.24.x (audit S4-19): unit-rescale each coefficient.  The library
    stores coefficients in meters (an even-asphere term ``A_p * r**p``
    is a sag length, so ``A_p`` has units ``length**(1 - p)``).  When
    the file body is written in ``units != M`` (``scale`` is the
    length-scale factor: 1e3 for MM, 1/0.0254 for IN), a coefficient of
    power ``p`` must scale by ``scale**(1 - p)`` so the written asphere
    is physically consistent with the (already scaled) radius / sag.
    Pre-fix the coefficients were written unscaled, so a MM file carried
    radii in mm but aspheres in per-meter -- an internally inconsistent
    prescription for any external Quadoa reader.  ``scale=1.0`` (the
    default, and the ``units='M'`` path) is a no-op, preserving byte-
    identical output for meter-unit exports.
    """
    if coeffs is None:
        return None
    if isinstance(coeffs, dict):
        return {str(int(p)): float(v) * (scale ** (1 - int(p)))
                for p, v in coeffs.items()}
    # Defensive: accept a list of (power, value) tuples or a sequence
    # of values (legacy callers).  A bare sequence of numbers cannot
    # be round-tripped without a power convention, so we refuse it.
    try:
        return {str(int(p)): float(v) * (scale ** (1 - int(p)))
                for p, v in coeffs}
    except (TypeError, ValueError):
        raise TypeError(
            "aspheric_coeffs must be a dict {power: value, ...}; got "
            f"{type(coeffs).__name__}.")


def _quadoa_deserialize_aspheric(obj, inv_scale=1.0):
    """Inverse of :func:`_quadoa_serialize_aspheric`.

    Accepts:

    * ``None`` -> ``None``
    * dict with string-or-int keys -> dict with int keys (canonical)
    * legacy list of values [a4, a6, a8] -> dict {4: a4, 6: a6, 8: a8}
      (the pre-v4.11.2 serializer wrote ``[4.0, 6.0, ...]`` -- those
      values are uninterpretable, so a legacy list is read at face
      value as coefficients starting from power=4).

    v5.24.x (audit S4-19): ``inv_scale`` (the length-scale factor that
    converts file units back to meters -- 1e-3 for MM, 0.0254 for IN)
    is applied per-coefficient as ``inv_scale**(1 - p)``, inverting the
    export-side ``scale**(1 - p)``.  With ``inv_scale == 1/scale`` the
    round-trip is exact for every power (``(scale * inv_scale)**(1-p) ==
    1``).  Default 1.0 preserves the meter-unit path byte-for-byte.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {int(k): float(v) * (inv_scale ** (1 - int(k)))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return {4 + 2 * i: float(v) * (inv_scale ** (1 - (4 + 2 * i)))
                for i, v in enumerate(obj)}
    raise TypeError(
        "aspheric_coeffs in Quadoa JSON must be dict, list, or null; "
        f"got {type(obj).__name__}.")


def export_quadoa_qos(prescription: Dict[str, Any], path: str, *,
                      wavelength: float,
                      stop_surface: Optional[int] = None,
                      aperture_diameter: Optional[float] = None,
                      back_focal_length: Optional[float] = None,
                      name: Optional[str] = None,
                      units: str = 'M') -> None:
    """Write a Quadoa Optikos-style ``.qos`` JSON system file.

    Quadoa's native file format is JSON-based.  The official schema is
    not fully publicly documented, so this writer emits a self-defined
    JSON layout (schema version :data:`QUADOA_SCHEMA_VERSION`) that
    captures every field a lumenairy prescription carries -- surfaces,
    radii (incl. biconic Y), conics, asphere coefficients, glasses,
    thicknesses, semi-diameters, aperture, stop index, wavelength,
    and units.  :func:`load_quadoa_qos` reads this layout back losslessly.

    .. warning::
        Quadoa-readability of the produced file is **not yet verified**.
        For pure round-tripping inside lumenairy this is exact; for
        external interchange with Quadoa Optikos itself, validate
        against a known-good reference ``.qos`` first.

    Parameters
    ----------
    prescription : dict
        Same format used by :func:`apply_real_lens`,
        :func:`export_zemax_zmx`, and :func:`export_codev_seq`.
    path : str
        Output ``.qos`` file path.
    wavelength : float
        Reference wavelength [m].
    stop_surface : int
        Zero-based index of the stop among refracting surfaces.
    aperture_diameter : float, optional
        Entrance-pupil diameter [m]; falls back to
        ``prescription['aperture_diameter']``.
    back_focal_length : float, optional
        BFL [m] from last surface to image plane.
    name : str, optional
        System name written into the JSON header.
    units : {'M', 'MM', 'IN'}, default 'M'
        Length units written in the header (file body is rescaled
        on write to preserve the chosen unit).

    See Also
    --------
    load_quadoa_qos
    export_zemax_zmx, export_codev_seq
    """
    import json

    surfaces = prescription['surfaces']
    thicknesses = prescription['thicknesses']
    # v5.4.6 (audit F-29): when the caller does not pass stop_surface
    # explicitly, default it to the PRESCRIPTION's own stop (stop_index,
    # else per-surface is_stop), not surface 0 -- otherwise a
    # load->export->load round trip relocates the aperture stop to
    # surface 0.
    if stop_surface is None:
        stop_surface = prescription.get('stop_index')
        if stop_surface is None:
            stop_surface = next(
                (i for i, s in enumerate(surfaces) if s.get('is_stop')), 0)
    stop_surface = int(stop_surface)
    if aperture_diameter is None:
        aperture_diameter = prescription.get('aperture_diameter', 25.4e-3)
    bfl = back_focal_length or 0.0

    name = name or prescription.get('name',
        os.path.splitext(os.path.basename(path))[0])

    units = str(units).upper()
    if units not in ('M', 'MM', 'IN'):
        raise ValueError(
            f"export_quadoa_qos: units must be M, MM, or IN (got {units!r})")
    scale = {'M': 1.0, 'MM': 1e3, 'IN': 1.0 / 0.0254}[units]

    surf_list = []
    for i, surf in enumerate(surfaces):
        t_m = thicknesses[i] if i < len(thicknesses) else bfl
        sd = surf.get('semi_diameter')
        entry = {
            'index': i,
            'radius': _quadoa_serialize_radius(
                surf.get('radius'), scale),
            'radius_y': _quadoa_serialize_radius(
                surf.get('radius_y'), scale),
            'conic': float(surf.get('conic', 0.0) or 0.0),
            'conic_y': (None if surf.get('conic_y') is None
                        else float(surf['conic_y'])),
            'aspheric_coeffs': _quadoa_serialize_aspheric(
                surf.get('aspheric_coeffs'), scale),
            'aspheric_coeffs_y': _quadoa_serialize_aspheric(
                surf.get('aspheric_coeffs_y'), scale),
            'glass_before': surf.get('glass_before', 'air'),
            'glass_after': surf.get('glass_after', 'air'),
            'thickness': float(t_m) * scale,
            'is_stop': bool(i == stop_surface),
            'semi_diameter': (None if sd is None or not np.isfinite(sd)
                              else float(sd) * scale),
            'comment': surf.get('comment', ''),
        }
        surf_list.append(entry)

    doc = {
        'format': 'quadoa-optikos-system',
        'schema_version': QUADOA_SCHEMA_VERSION,
        'generated_by': 'lumenairy.export_quadoa_qos',
        'name': name,
        'units': units,
        'wavelength_nm': float(wavelength) * 1e9,
        'aperture_diameter': float(aperture_diameter) * scale,
        'back_focal_length': float(bfl) * scale,
        'stop_surface': int(stop_surface),
        'surfaces': surf_list,
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)
        f.write('\n')


def load_quadoa_qos(filepath: str,
                    name: Optional[str] = None) -> Dict[str, Any]:
    """Parse a Quadoa Optikos-style ``.qos`` JSON file into a
    lumenairy prescription dict.

    Round-trips losslessly with :func:`export_quadoa_qos` (schema
    version :data:`QUADOA_SCHEMA_VERSION`).  If the file is missing
    the ``format`` / ``schema_version`` header but otherwise looks
    JSON-like with a ``surfaces`` array, the parser falls back to
    a permissive read; unknown fields are preserved on each surface
    under ``surf['_extras']`` so callers can inspect them without
    losing information.

    Parameters
    ----------
    filepath : str
    name : str, optional
        Override for the prescription name; defaults to the JSON
        ``name`` field or the file stem.

    Returns
    -------
    dict
        Standard lumenairy prescription dict
        (``{'name', 'aperture_diameter', 'surfaces', 'thicknesses',
            'wavelength', 'stop_index', ...}``).

    See Also
    --------
    export_quadoa_qos
    """
    import json

    with open(filepath, 'r', encoding='utf-8') as f:
        doc = json.load(f)

    if not isinstance(doc, dict) or 'surfaces' not in doc:
        raise ValueError(
            f"load_quadoa_qos: {filepath!r} is not a recognisable "
            f"Quadoa-style JSON system file (no 'surfaces' array).")

    units = str(doc.get('units', 'M')).upper()
    if units not in ('M', 'MM', 'IN'):
        warnings.warn(
            f"load_quadoa_qos: unknown units {units!r}, assuming meters",
            UserWarning, stacklevel=2)
        units = 'M'
    inv_scale = {'M': 1.0, 'MM': 1e-3, 'IN': 0.0254}[units]

    def _radius_in(v):
        if v is None:
            return float('inf')
        return float(v) * inv_scale

    raw = doc['surfaces']
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"load_quadoa_qos: {filepath!r} has empty 'surfaces' list.")

    surfaces = []
    thicknesses = []
    stop_index = None
    semi_diameters = []
    # v4.13.2 (C-P0-5): track the last surface's THI as a BFL
    # fallback for foreign `.qos` files that don't write the top-level
    # ``back_focal_length`` field.
    last_surface_thickness = 0.0
    known_keys = {
        'index', 'radius', 'radius_y', 'conic', 'conic_y',
        'aspheric_coeffs', 'aspheric_coeffs_y',
        'glass_before', 'glass_after',
        'thickness', 'is_stop', 'semi_diameter', 'comment',
    }
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            raise ValueError(
                f"load_quadoa_qos: surface {i} is not a JSON object.")
        sd = s.get('semi_diameter')
        surf = {
            'radius': _radius_in(s.get('radius')),
            'conic': float(s.get('conic', 0.0) or 0.0),
            'aspheric_coeffs': _quadoa_deserialize_aspheric(
                s.get('aspheric_coeffs'), inv_scale),
            'radius_y': (None if s.get('radius_y') is None
                         else _radius_in(s['radius_y'])),
            'conic_y': (None if s.get('conic_y') is None
                        else float(s['conic_y'])),
            'aspheric_coeffs_y': _quadoa_deserialize_aspheric(
                s.get('aspheric_coeffs_y'), inv_scale),
            'glass_before': s.get('glass_before', 'air'),
            'glass_after': s.get('glass_after', 'air'),
        }
        if sd is not None:
            surf['semi_diameter'] = float(sd) * inv_scale
            semi_diameters.append(surf['semi_diameter'])
        if s.get('comment'):
            surf['comment'] = s['comment']
        extras = {k: v for k, v in s.items() if k not in known_keys}
        if extras:
            surf['_extras'] = extras
        surfaces.append(surf)
        # v4.13.2 (C-P0-5): capture the last surface's THI as the
        # fallback BFL so externally-authored `.qos` files that encode
        # the BFL on the final surface (instead of the top-level
        # ``back_focal_length`` field) do not silently drop it on read.
        if i < len(raw) - 1:
            thicknesses.append(float(s.get('thickness', 0.0)) * inv_scale)
        else:
            last_surface_thickness = (
                float(s.get('thickness', 0.0)) * inv_scale)
        if s.get('is_stop'):
            stop_index = i

    aperture_diameter = doc.get('aperture_diameter')
    aperture_m = (
        25.4e-3 if aperture_diameter is None
        else float(aperture_diameter) * inv_scale)

    result = {
        'name': name or doc.get('name')
            or os.path.splitext(os.path.basename(filepath))[0],
        'aperture_diameter': aperture_m,
        'surfaces': surfaces,
        'thicknesses': thicknesses,
    }
    if 'wavelength_nm' in doc:
        result['wavelength'] = float(doc['wavelength_nm']) * 1e-9
    if stop_index is None and 'stop_surface' in doc:
        try:
            stop_index = int(doc['stop_surface'])
        except (TypeError, ValueError):
            stop_index = None
    if stop_index is not None:
        result['stop_index'] = stop_index
    if semi_diameters:
        result['has_semi_diameters'] = True
    # v4.13.2 (C-P0-5): preserve the BFL.  Prefer the top-level
    # ``back_focal_length`` field (what :func:`export_quadoa_qos`
    # writes); otherwise fall back to the last surface's THI for
    # foreign `.qos` files that follow the trailing-THI convention.
    bfl_raw = doc.get('back_focal_length')
    if bfl_raw is not None:
        try:
            bfl_val = float(bfl_raw) * inv_scale
            if np.isfinite(bfl_val) and bfl_val != 0.0:
                result['back_focal_length'] = bfl_val
        except (TypeError, ValueError):
            pass
    if 'back_focal_length' not in result:
        if (last_surface_thickness != 0.0
                and np.isfinite(last_surface_thickness)):
            result['back_focal_length'] = float(last_surface_thickness)
    return result


# A-13ish (AUDIT_ADVERSARIAL_CODEBASE 2026-07-25): declare the public
# surface explicitly, matching the convention every ``analysis/`` module
# already follows.  Every name here is re-exported through
#  ``lumenairy.io.prescriptions`` (v5.1.0 split) and the top-level facade.
__all__ = [
    'load_quadoa_qos',
    'export_quadoa_qos',
    'QUADOA_SCHEMA_VERSION',
]

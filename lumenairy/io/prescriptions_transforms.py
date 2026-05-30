"""
Prescription dict transforms: geometric scaling, schema normalisation,
folded-design splitting, mirror queries.

Operates on the canonical LumenAiry prescription-dict schema produced
by the builder factories and the various loaders.  None of these
helpers parse or write to disk; they are pure-Python dict transforms.

v5.1.0 split (Agent F): extracted from ``lumenairy/io/prescriptions.py``
without any logic change.  Public API is unchanged: every name in this
module is re-exported through ``lumenairy.io.prescriptions`` and
``lumenairy``.

Author: Andrew Traverso
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np

# ============================================================================
# Geometric scaling
# ============================================================================


def scale_prescription(prescription: Dict[str, Any],
                       factor: float) -> Dict[str, Any]:
    """Geometric self-similarity: return a deep-copied prescription
    whose every linear dimension is multiplied by ``factor``.

    The scaled system has the **same F-number, NA, paraxial
    magnification, and diffraction-limited spot size in absolute
    units** as the original (because wavelength is not scaled).  Use
    cases:

    * Unit conversion (e.g. ``factor=1e-3`` to convert a prescription
      built in millimetres to the SI metres convention).
    * Speeding up ray-trace + polynomial-fit-based diffraction
      methods (``fit_canonical_polynomials``,
      ``aberration_tensor``, ``propagate_modal_asymptotic``) where
      the absolute output pixel count for a given Nyquist
      sampling scales with the system's physical extent.  Optimisation
      loops that re-fit per merit-evaluation can be 4 - 16 times
      cheaper when the fit's source / output box shrinks.
    * Building geometrically-similar test prescriptions
      (e.g. a 0.25x-scale replica for fast smoke tests).

    The function scales:

    * ``aperture_diameter`` (top-level) and every ``semi_diameter``
      on ``elements`` and ``surfaces``;
    * ``object_distance``;
    * every entry in ``thicknesses`` and ``all_thicknesses``;
    * every ``radius`` / ``radius_y`` on each surface and element;
    * each entry in ``coord_breaks``'s ``decenter_x_m``,
      ``decenter_y_m``, and ``thickness_m``;
    * every aspheric coefficient ``A_n`` as ``A_n / factor**(n - 1)``,
      so the surface sag ``sum_n A_n * h**n`` scales linearly with
      ``factor`` when ``h`` does.

    The function does NOT scale (these are dimensionless or
    wavelength-relative): ``conic`` / ``conic_y`` constants, glass
    names, tilt angles in coord breaks, stop indices, wavelength
    metadata.  ``DAMMANN_PERIODX`` / ``DAMMANN_PERIODY`` aren't part
    of the prescription dict and are also not touched.

    Parameters
    ----------
    prescription : dict
        lumenairy prescription dict (any builder / loader output).
        Not modified.
    factor : float
        Linear scale factor.  Must be finite and positive.  ``> 1``
        enlarges; ``< 1`` shrinks.

    Returns
    -------
    dict
        Deep-copied scaled prescription.  Round-trips through
        ``apply_real_lens`` / ``fit_canonical_polynomials`` /
        ``aberration_tensor`` exactly the same as the original up to
        the chosen scale.

    Examples
    --------
    >>> import lumenairy as la
    >>> rx = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=10e-3)
    >>> rx_small = la.scale_prescription(rx, 0.25)
    >>> rx_small['surfaces'][0]['radius']  # 50 mm * 0.25 = 12.5 mm
    0.0125
    >>> rx_small['aperture_diameter']      # 10 mm * 0.25 = 2.5 mm
    0.0025

    See Also
    --------
    recommend_grid_for_prescription
        Pre-flight grid sizer that uses the same scaling identity
        when comparing simulation cost across scaled designs.
    """
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(
            f"factor must be finite and > 0, got {factor!r}")

    rx = copy.deepcopy(prescription)
    s = float(factor)

    # Top-level scalars
    if rx.get('aperture_diameter') is not None:
        rx['aperture_diameter'] = float(rx['aperture_diameter']) * s
    if rx.get('object_distance') is not None:
        rx['object_distance'] = float(rx['object_distance']) * s

    # Thickness lists
    for tkey in ('thicknesses', 'all_thicknesses'):
        if tkey in rx and rx[tkey]:
            rx[tkey] = [
                (float(t) * s if t is not None else t) for t in rx[tkey]
            ]

    # Per-surface and per-element scaling
    def _scale_surface_like(d):
        if not isinstance(d, dict):
            return
        for rkey in ('radius', 'radius_y'):
            if d.get(rkey) is not None:
                v = d[rkey]
                if np.isfinite(v):
                    d[rkey] = float(v) * s
                # Inf radii (flat surfaces) stay Inf
        if d.get('semi_diameter') is not None:
            sd = d['semi_diameter']
            if np.isfinite(sd):
                d['semi_diameter'] = float(sd) * s
        for ackey in ('aspheric_coeffs', 'aspheric_coeffs_y'):
            ac = d.get(ackey)
            if isinstance(ac, dict):
                d[ackey] = {
                    int(n): float(v) / (s ** (int(n) - 1))
                    for n, v in ac.items()
                }

    if isinstance(rx.get('surfaces'), list):
        for surf in rx['surfaces']:
            _scale_surface_like(surf)
    if isinstance(rx.get('elements'), list):
        for elem in rx['elements']:
            _scale_surface_like(elem)

    # Coord breaks: decenters and explicit thicknesses scale; tilts don't
    if isinstance(rx.get('coord_breaks'), list):
        for cb in rx['coord_breaks']:
            if not isinstance(cb, dict):
                continue
            for dkey in ('decenter_x_m', 'decenter_y_m', 'thickness_m'):
                if cb.get(dkey) is not None:
                    cb[dkey] = float(cb[dkey]) * s

    return rx


# ============================================================================
# Schema normalisation (4.0+)
# ============================================================================


def normalize_prescription(prescription: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``prescription`` with the canonical superset
    of schema keys filled in.

    The library's prescription dict has historically been built by
    several routes that emit slightly different schemas:

    * :func:`make_singlet` / :func:`make_doublet` / etc. return the
      minimal ``{'surfaces', 'thicknesses', 'aperture_diameter'}``.
    * :func:`load_zemax_zmx` adds ``'elements'`` and
      ``'all_thicknesses'`` (refractive-only ``'surfaces'`` plus the
      full element list including mirrors).
    * :func:`load_zemax_prescription_data_txt` additionally adds
      ``'wavelength'`` (primary), ``'units'`` (originating unit
      string), and ``'has_semi_diameters'``.
    * :func:`load_codev_seq` / :func:`load_quadoa_qos` match
      ``load_zemax_zmx``'s schema.

    Downstream functions (:func:`apply_real_lens`,
    :func:`monte_carlo_tolerancing`, :func:`eval_image_plane_wfe`, ...)
    each accept any of these schemas via silent fallback, but the
    fallback rules differ from function to function and have caught
    real users by surprise (for example,
    :func:`monte_carlo_tolerancing` perturbs only ``'surfaces'`` and
    silently skips mirrors in an ``'elements'`` list).

    This helper builds the **canonical superset**: every prescription
    is returned with both ``'surfaces'`` and ``'elements'`` populated
    (with ``elements`` mirroring ``surfaces`` if no ``elements``
    were provided), both ``'thicknesses'`` and ``'all_thicknesses'``
    populated, and the optional metadata fields (``'wavelength'``,
    ``'units'``, ``'object_distance'``, ``'stop_index'``,
    ``'has_semi_diameters'``) present (with safe defaults: ``None``
    for the metadata, ``0.0`` for ``object_distance`` if missing).

    The original dict is not modified -- a deep-copy is returned.

    Parameters
    ----------
    prescription : dict
        Any LumenAiry prescription dict (built by a make_*, loaded by
        any load_*, or hand-rolled).

    Returns
    -------
    dict
        Deep-copied prescription with canonical schema.

    Examples
    --------
    >>> import lumenairy as la
    >>> p = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=10e-3)
    >>> sorted(p.keys())                                   # before
    ['aperture_diameter', 'name', 'surfaces', 'thicknesses']
    >>> q = la.normalize_prescription(p)
    >>> sorted(q.keys())                                   # after
    ['aperture_diameter', 'all_thicknesses', 'elements', 'has_semi_diameters',
     'name', 'object_distance', 'stop_index', 'surfaces', 'thicknesses',
     'units', 'wavelength']
    >>> q['elements'] == q['surfaces']    # elements mirrors surfaces
    True

    Notes
    -----
    Callers can either run their prescription through
    ``normalize_prescription`` once at the top of a pipeline (the
    recommended idiom) or let each downstream function fall back to
    its existing schema-detection logic.  Both paths work; the
    explicit normalisation just removes ambiguity.
    """
    import copy
    if not isinstance(prescription, dict):
        raise TypeError(
            f"normalize_prescription expects a dict, got "
            f"{type(prescription).__name__}.")
    rx = copy.deepcopy(prescription)

    # surfaces / elements -- mirror whichever is missing.
    surfs = rx.get('surfaces')
    elems = rx.get('elements')
    if surfs is None and elems is None:
        raise ValueError(
            "normalize_prescription: prescription has neither "
            "'surfaces' nor 'elements'.")
    if surfs is None:
        # Build surfaces from elements (drop pure-mirror entries that
        # apply_real_lens cannot consume).  The canonical mirror flag
        # is ``element_type='mirror'`` -- pre-v4.11.2 this checked
        # ``e.get('mirror')`` which is never set, making the filter a
        # no-op (mirrors leaked through to apply_real_lens).
        surfs = [e for e in elems
                 if not (isinstance(e, dict)
                         and (e.get('element_type') == 'mirror'
                              or e.get('mirror')))]
        rx['surfaces'] = surfs
    if elems is None:
        # Elements mirror surfaces verbatim (no mirrors in pure
        # refractive prescriptions).
        rx['elements'] = list(surfs)

    # thicknesses / all_thicknesses
    th = rx.get('thicknesses')
    ath = rx.get('all_thicknesses')
    if th is None and ath is not None:
        rx['thicknesses'] = list(ath)
    elif ath is None and th is not None:
        rx['all_thicknesses'] = list(th)
    elif th is None and ath is None:
        rx['thicknesses'] = []
        rx['all_thicknesses'] = []

    # Optional metadata: ensure keys exist (with sensible defaults).
    rx.setdefault('aperture_diameter', None)
    rx.setdefault('object_distance', 0.0)
    rx.setdefault('stop_index', None)
    rx.setdefault('wavelength', None)
    rx.setdefault('units', None)
    rx.setdefault('has_semi_diameters', False)
    rx.setdefault('name', None)

    return rx


# ============================================================================
# Folded-design helpers
# ============================================================================


def split_prescription_at_mirrors(
    prescription: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Split a folded-design prescription into per-segment legs at every
    fold mirror.

    The Zemax-loader emits both ``'surfaces'`` (refracting-only, what
    :func:`apply_real_lens` consumes) and ``'elements'`` (full sequence
    including mirrors).  For a folded design, walking ``'surfaces'``
    propagates the wave along the *unfolded equivalent* axis -- correct
    for scalar on-axis fields when every mirror is flat, but silently
    wrong as soon as a curved mirror or a polarisation-sensitive field
    enters the picture.  This helper returns the segment list so the
    caller can alternate :func:`apply_real_lens` (each segment) with
    :func:`apply_mirror` (each fold), keeping the physics explicit.

    Parameters
    ----------
    prescription : dict
        A prescription dict as returned by :func:`load_zemax_zmx`,
        :func:`load_codev_seq`, or :func:`load_quadoa_qos` -- i.e.
        carrying both ``'elements'`` and ``'all_thicknesses'``.  A
        plain ``'surfaces'``-only prescription is returned unchanged
        wrapped in a single-element list.

    Returns
    -------
    legs : list of dict
        One entry per leg, in propagation order.  Each entry is either:

        - ``{'kind': 'refractive', 'prescription': sub_rx}`` -- a
          refracting-only sub-prescription consumable by
          :func:`apply_real_lens`.  ``sub_rx`` is the deep-copied
          minimal schema ``{'surfaces', 'thicknesses',
          'aperture_diameter', 'name'}``.
        - ``{'kind': 'mirror', 'element': mirror_dict}`` -- the raw
          mirror element from the original ``'elements'`` list.  Pass
          its radius / conic / aperture into :func:`apply_mirror`.

    Examples
    --------
    >>> rx = la.load_zemax_zmx('folded_design.zmx')
    >>> legs = la.split_prescription_at_mirrors(rx)
    >>> E = E_in
    >>> for leg in legs:
    ...     if leg['kind'] == 'refractive':
    ...         E = la.apply_real_lens(E, prescription=leg['prescription'],
    ...                                wavelength=wl, dx=dx)
    ...     else:
    ...         m = leg['element']
    ...         E = la.apply_mirror(E, wavelength=wl, dx=dx,
    ...                             radius=m.get('radius'),
    ...                             conic=m.get('conic', 0.0),
    ...                             aperture_diameter=m.get('clear_aperture'))

    Notes
    -----
    For a *flat* fold mirror, :func:`apply_mirror` returns the field
    unchanged apart from the aperture clip -- the propagation direction
    flips but the field's complex-amplitude distribution is unaffected
    in its own local +z frame.  For a *curved* fold mirror the focusing
    phase is applied.  Neither case automatically rotates the field's
    coordinate frame; callers writing 3-D world-frame analyses still
    need to track which world-axis "+z" points in for each leg (see
    :func:`world_surfaces_from_prescription` for the ray-side
    counterpart).

    Polarisation handling at fold mirrors (s/p phase / amplitude
    response from a real coating) is not done by :func:`apply_mirror`;
    if you need it, use a Jones-aware mirror wrapper of your own --
    the segment list this function returns is the right scaffold to
    insert it into.
    """
    elements = prescription.get('elements')
    all_th = prescription.get('all_thicknesses')
    if elements is None or all_th is None:
        # Plain prescription without mirrors -- return as a single leg.
        return [{'kind': 'refractive',
                 'prescription': copy.deepcopy(prescription)}]

    # Validate alignment of elements vs all_thicknesses.
    if len(all_th) != max(len(elements) - 1, 0):
        raise ValueError(
            f"split_prescription_at_mirrors: prescription has "
            f"{len(elements)} elements but {len(all_th)} thicknesses; "
            f"expected {len(elements) - 1}.  Was the prescription "
            f"loaded from a tool that doesn't emit a contiguous "
            f"thickness list?  Try normalize_prescription first.")

    aperture = prescription.get('aperture_diameter')
    name = prescription.get('name')

    legs: List[Dict[str, Any]] = []
    seg_surfaces: List[Dict[str, Any]] = []
    seg_thicknesses: List[float] = []
    last_was_surface = False

    def _flush_refractive() -> None:
        if not seg_surfaces:
            return
        sub_rx = {
            'surfaces': copy.deepcopy(seg_surfaces),
            'thicknesses': list(seg_thicknesses),
            'aperture_diameter': aperture,
            'name': f"{name}:seg{len(legs)}" if name else None,
        }
        legs.append({'kind': 'refractive', 'prescription': sub_rx})
        seg_surfaces.clear()
        seg_thicknesses.clear()

    for idx, el in enumerate(elements):
        kind = el.get('element_type', 'surface')
        if kind == 'mirror':
            _flush_refractive()
            # v5.4.6 (audit F-15): preserve the propagation distances INTO
            # and OUT OF the mirror (previously dropped), so the folded-
            # design walking workflow can reconstruct the inter-leg
            # geometry.  all_th[i] is the gap from element i to element i+1.
            d_in = float(all_th[idx - 1]) if idx > 0 else 0.0
            d_out = float(all_th[idx]) if idx < len(all_th) else 0.0
            legs.append({'kind': 'mirror',
                         'element': copy.deepcopy(el),
                         'distance_in': d_in,
                         'distance_out': d_out})
            last_was_surface = False
            continue
        # Refractive surface.
        if last_was_surface and idx > 0:
            # Carry the thickness *into* this segment from the prior
            # surface within the same refractive run.
            seg_thicknesses.append(float(all_th[idx - 1]))
        seg_surfaces.append(copy.deepcopy(el))
        last_was_surface = True

    _flush_refractive()
    return legs


def has_mirrors(prescription: Dict[str, Any]) -> bool:
    """Return True iff ``prescription['elements']`` carries any entry
    with ``element_type == 'mirror'``."""
    elements = prescription.get('elements')
    if elements is None:
        return False
    return any(el.get('element_type') == 'mirror' for el in elements)

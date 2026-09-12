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
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

# ============================================================================
# Geometric scaling
# ============================================================================

# I7 (AUDIT_ADVERSARIAL_EXHAUSTIVE 2026-09-11): ``scale_prescription`` handled
# radius / semi_diameter / aspheric_coeffs only, and its docstring listed what
# it scaled and what it deliberately did not -- so the three families it
# handled NEITHER way (Forbes-Q, diffractives, BFL) read as coverage.  The
# tables below are the explicit ledger: every key a dict of this kind may
# carry is either scaled, listed as deliberately dimensionless, or warned
# about.  Add to them when the schema grows.
_SURFACE_SCALED_KEYS = frozenset({
    'radius', 'radius_y', 'semi_diameter', 'aspheric_coeffs',
    'aspheric_coeffs_y', 'r_max', 'q_bfs_coeffs', 'q_con_coeffs',
})
_SURFACE_DIMENSIONLESS_KEYS = frozenset({
    'conic', 'conic_y', 'glass', 'glass_before', 'glass_after',
    'element_type', 'freeform_type', 'is_stop', 'is_mirror', 'surf_num',
    'comment', 'name', 'stop_index', 'tilt', 'diffractive',
})
# ``decenter`` is a LENGTH pair, but it lives on builder surfaces
# (make_off_axis_parabola) rather than loader ones; it is scaled explicitly.
_SURFACE_SCALED_KEYS = _SURFACE_SCALED_KEYS | {'decenter', 'clear_aperture'}
_DIFFRACTIVE_SCALED_KEYS = frozenset({
    'period', 'gap_before', 'gap_after', 'semi_diameter', 'origin',
    'lines_per_um',
})
_DIFFRACTIVE_DIMENSIONLESS_KEYS = frozenset({
    'type', 'order', 'angle_deg', 'surf_num', 'name', 'comment',
})
# Substrings that make a key name length-like.  Used only for the warning.
_LENGTH_LIKE_SUBSTRINGS = (
    'radius', 'diameter', 'thickness', 'distance', 'length', 'gap',
    'period', 'pitch', 'decenter', 'offset', 'sag', 'aperture', 'origin',
    'height', 'width', 'focal', 'depth', 'spacing', 'clearance',
)


def _warn_unscaled_length_keys(d, scaled_keys, what):
    """Warn once per unrecognised length-like key left unscaled."""
    unknown = sorted(
        k for k in d
        if k not in scaled_keys
        and any(sub in str(k).lower() for sub in _LENGTH_LIKE_SUBSTRINGS)
        and not any(sub in str(k).lower()
                    for sub in ('conic', 'aperture_type', 'is_'))
        and d.get(k) is not None
    )
    if unknown:
        warnings.warn(
            f"scale_prescription: {what} entry carries length-like key(s) "
            f"{unknown} that this transform does not recognise; they were "
            f"left UNSCALED, so the result is not geometrically similar.  "
            f"Scale them yourself or extend "
            f"prescriptions_transforms._SURFACE_SCALED_KEYS.",
            UserWarning, stacklevel=3)


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
      ``factor`` when ``h`` does;
    * the Forbes-Q freeform lengths ``r_max`` (normalisation radius) and
      ``q_bfs_coeffs`` / ``q_con_coeffs`` (each coefficient is a sag
      LENGTH, so they scale linearly, not as the aspheric rule);
    * the stored ``back_focal_length``;
    * every length on each entry of ``diffractives`` -- ``period``,
      ``origin``, ``gap_before``, ``gap_after``, ``semi_diameter``.

    The function does NOT scale (these are dimensionless or
    wavelength-relative): ``conic`` / ``conic_y`` constants, glass
    names, tilt angles in coord breaks, stop indices, wavelength
    metadata, and the diffraction ``order``.  ``DAMMANN_PERIODX`` /
    ``DAMMANN_PERIODY`` aren't part of the prescription dict and are
    also not touched.

    Any other key whose name looks length-like and is not in the handled
    set raises a :class:`UserWarning` rather than being left at its
    original size in silence.

    .. note::
       Scaling a grating ``period`` is what geometric self-similarity
       requires (every length times ``factor``), but the wavelength is
       deliberately NOT scaled -- so a scaled DOE diffracts at a
       different angle than the original.  That is the same trade the
       rest of this transform makes for every focusing surface (the
       F-number is preserved, the absolute diffraction-limited spot is
       not); pass the diffractives through unscaled by removing the key
       first if the DOE is meant to be kept at its as-built pitch.

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

    # Top-level scalars.  I7: ``back_focal_length`` (written by
    # ``load_codev_seq`` / ``load_quadoa_qos``) is a LENGTH and was left at
    # its original value, so a scaled prescription carried the unscaled
    # image distance -- measured at s = 0.25: 0.084 m instead of 0.021 m.
    for _lk in ('aperture_diameter', 'object_distance', 'back_focal_length'):
        if rx.get(_lk) is not None:
            rx[_lk] = float(rx[_lk]) * s

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
        # I7: Forbes-Q freeform.  ``r_max`` is a normalisation RADIUS and each
        # ``q_*_coeffs`` entry is a sag LENGTH, so both scale linearly -- the
        # aspheric ``A_n / s**(n-1)`` rule does NOT apply (the Q polynomials
        # are functions of the dimensionless u = r / r_max).  Pre-fix neither
        # was touched, so a scaled Q-type surface kept its original freeform
        # sag on a rescaled base conic (measured at s = 0.25: r_max stayed
        # 7.5 mm instead of 1.875 mm).
        if d.get('r_max') is not None and np.isfinite(d['r_max']):
            d['r_max'] = float(d['r_max']) * s
        for qkey in ('q_bfs_coeffs', 'q_con_coeffs'):
            qc = d.get(qkey)
            if isinstance(qc, (list, tuple)):
                d[qkey] = type(qc)(float(v) * s for v in qc)
            elif isinstance(qc, np.ndarray):
                d[qkey] = np.asarray(qc, dtype=float) * s
        # I7: the builder-side length keys (``make_off_axis_parabola`` writes
        # a decenter pair and a clear aperture) are lengths too; ``tilt`` is
        # an angle triple and is deliberately left alone.
        if d.get('clear_aperture') is not None and np.isfinite(
                d['clear_aperture']):
            d['clear_aperture'] = float(d['clear_aperture']) * s
        dec = d.get('decenter')
        if isinstance(dec, (list, tuple)):
            d['decenter'] = type(dec)(float(v) * s for v in dec)
        elif isinstance(dec, np.ndarray):
            d['decenter'] = np.asarray(dec, dtype=float) * s
        _warn_unscaled_length_keys(d, _SURFACE_SCALED_KEYS, 'surface')

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

    # I7: the v5.32 diffractive payload is entirely lengths and was untouched,
    # so a scaled system kept the original DOE pitch and axial gaps (measured
    # at s = 0.25: period 2 um instead of 0.5 um, gap_before 10 mm instead of
    # 2.5 mm) -- i.e. the "self-similar" result was not self-similar at all.
    if isinstance(rx.get('diffractives'), list):
        for dg in rx['diffractives']:
            if not isinstance(dg, dict):
                continue
            for dkey in ('period', 'gap_before', 'gap_after',
                         'semi_diameter'):
                if dg.get(dkey) is not None and np.isfinite(dg[dkey]):
                    dg[dkey] = float(dg[dkey]) * s
            org = dg.get('origin')
            if isinstance(org, (list, tuple)):
                dg['origin'] = type(org)(float(v) * s for v in org)
            elif isinstance(org, np.ndarray):
                dg['origin'] = np.asarray(org, dtype=float) * s
            # ``lines_per_um`` is the reciprocal of the pitch, so it scales
            # INVERSELY -- keep it consistent with ``period``.
            if dg.get('lines_per_um') is not None and dg['lines_per_um'] != 0:
                dg['lines_per_um'] = float(dg['lines_per_um']) / s
            _warn_unscaled_length_keys(dg, _DIFFRACTIVE_SCALED_KEYS,
                                       'diffractive')

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
    * :func:`load_codev_seq` emits the same ``'elements'`` /
      ``'all_thicknesses'`` pair (I7, v5.46 -- before that it emitted
      neither, although this docstring said it did).
    * :func:`load_quadoa_qos` still emits ``'surfaces'`` /
      ``'thicknesses'`` only; run it through this helper to get the
      superset.

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
    were provided, each entry carrying the canonical
    ``element_type='surface'`` discriminator -- v5.46, I7), both
    ``'thicknesses'`` and ``'all_thicknesses'`` populated, and the optional metadata fields (``'wavelength'``,
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
    >>> q['elements'][0]['element_type']  # canonical discriminator (v5.46)
    'surface'

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
        #
        # I7 (AUDIT_ADVERSARIAL_EXHAUSTIVE 2026-09-11): stamp
        # ``element_type='surface'`` on the mirrored entries.  Pre-fix they
        # were plain surface dicts with no ``element_type``, so
        # ``generate_simulation_script`` -- which subscripts
        # ``elem['element_type']`` -- raised ``KeyError: 'element_type'`` on
        # the output of the one helper documented as "the recommended idiom"
        # for making a builder prescription codegen-shaped.
        #
        # The stamp is applied IN PLACE on the shared dicts (``rx`` is
        # already a deep copy of the caller's input), so the documented
        # ``q['elements'] == q['surfaces']`` identity -- and the aliasing it
        # rests on -- is preserved exactly; only the canonical discriminator
        # is now present on both views instead of neither.
        for _s in surfs:
            if isinstance(_s, dict):
                _s.setdefault('element_type', 'surface')
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
        A prescription dict carrying both ``'elements'`` and
        ``'all_thicknesses'`` -- as returned by :func:`load_zemax_zmx`,
        :func:`load_codev_seq` (v5.46+), or :func:`normalize_prescription`.
        A plain ``'surfaces'``-only prescription (for example a
        :func:`load_quadoa_qos` result, or any ``make_*`` builder output)
        is returned unchanged wrapped in a single-element list, **with a
        :class:`UserWarning`** -- that path cannot see a fold, so a silent
        single-leg answer would be indistinguishable from a genuinely
        unfolded design.

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
        # I7: say so.  Pre-fix this early return was silent, so a
        # prescription whose loader simply does not emit ``elements``
        # reported "one refractive leg, no folds" -- indistinguishable from a
        # genuinely unfolded design, and the docstring above promises the
        # CODE V / Quadoa loaders carry both keys.
        _missing = [k for k, v in (('elements', elements),
                                   ('all_thicknesses', all_th)) if v is None]
        warnings.warn(
            f"split_prescription_at_mirrors: prescription "
            f"{prescription.get('name')!r} has no {' / '.join(_missing)} key, "
            f"so it is returned as ONE refractive leg without inspecting it "
            f"for mirrors.  If the design is folded, run it through "
            f"normalize_prescription (or reload it with a loader that emits "
            f"the full element list) first -- otherwise the fold is silently "
            f"flattened.",
            UserWarning, stacklevel=2)
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


# ============================================================================
# Combine multiple elements into one whole-group prescription
# ============================================================================


def combine_prescriptions(
    prescriptions: List[Dict[str, Any]],
    gaps: Any,
    *,
    aperture: Optional[float] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Concatenate several lens prescriptions separated by air gaps into ONE
    multi-element prescription, so an entire lens group can be propagated in a
    **single pass** by any real-lens propagator (:func:`apply_real_lens`,
    :func:`apply_real_lens_traced`, :func:`apply_real_lens_traced_multibranch`,
    :func:`apply_real_lens_gbd`, :func:`apply_real_lens_maslov`).

    This is the inverse of :func:`split_prescription_at_mirrors` for the
    all-refractive case, and the direct way to feed a stack of catalog elements
    (``make_singlet`` / ``make_doublet`` / ``thorlabs_lens`` / a Zemax lens
    group) to a whole-system propagator.  A single-pass trace avoids the
    per-element field reconstruction that accumulates error when a diverging
    beam is relayed through many elements -- and, with the multibranch
    propagator, stays caustic-safe through the focus.  (The output plane past
    the last vertex is set on the propagator via its ``output_plane_distance``,
    not here.)

    Parameters
    ----------
    prescriptions : sequence of prescription dicts
        Each a self-contained element that exits into the same medium its
        downstream neighbour enters from (normally ``air -> ... -> air``).
        Any builder / loader schema is accepted (run through
        :func:`normalize_prescription` internally).
    gaps : float or sequence of ``len(prescriptions) - 1`` floats
        Axial gap [m] between consecutive elements (last vertex of element *i*
        to first vertex of element *i+1*), in the medium element *i* exits into.
        A scalar is broadcast to every gap.
    aperture : float, optional
        Combined clear-aperture diameter [m].  Default: the max of the input
        elements' ``aperture_diameter``.
    name : str, optional
        Label for the combined prescription.

    Returns
    -------
    dict
        ``{name, aperture_diameter, surfaces, thicknesses}`` with the
        concatenated surfaces and the interleaved internal + gap thicknesses
        (``len(thicknesses) == len(surfaces) - 1``).

    Raises
    ------
    ValueError
        If no prescriptions are given, ``gaps`` has the wrong length, an element
        has no surfaces, or a junction is a glass discontinuity (element *i*'s
        exit medium != element *i+1*'s entry medium -- unphysical across a gap).

    Examples
    --------
    >>> import lumenairy as la
    >>> a = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=25e-3)
    >>> b = la.make_singlet(40e-3, -40e-3, 4e-3, 'N-BK7', aperture=25e-3)
    >>> group = la.combine_prescriptions([a, b], gaps=20e-3)   # 20 mm air gap
    >>> len(group['surfaces']), len(group['thicknesses'])
    (4, 3)
    """
    if not prescriptions:
        raise ValueError("combine_prescriptions: need at least one prescription.")
    n = len(prescriptions)
    if np.isscalar(gaps):
        gaps = [float(gaps)] * (n - 1)
    else:
        gaps = [float(g) for g in gaps]
    if len(gaps) != n - 1:
        raise ValueError(
            f"combine_prescriptions: {n} prescriptions but {len(gaps)} gaps "
            f"(expected {n - 1}).")

    surfaces: List[Dict[str, Any]] = []
    thicknesses: List[float] = []
    apertures: List[float] = []
    for i, rx in enumerate(prescriptions):
        rxn = normalize_prescription(rx)
        surfs = rxn.get('surfaces') or []
        if not surfs:
            raise ValueError(
                f"combine_prescriptions: prescription {i} has no surfaces.")
        if i > 0:
            prev_exit = surfaces[-1].get('glass_after') or 'air'
            this_entry = surfs[0].get('glass_before') or 'air'
            if prev_exit != this_entry:
                raise ValueError(
                    f"combine_prescriptions: glass discontinuity at junction "
                    f"{i} -- element {i - 1} exits into {prev_exit!r} but "
                    f"element {i} enters from {this_entry!r}.  A gap sits in a "
                    f"single medium; make glass_after/glass_before match.")
            thicknesses.append(gaps[i - 1])                 # the inter-element gap
        surfaces.extend(copy.deepcopy(s) for s in surfs)
        thicknesses.extend(float(t) for t in (rxn.get('thicknesses') or []))
        ap = rxn.get('aperture_diameter')
        if ap:
            apertures.append(float(ap))

    return {
        'name': name or ' + '.join(
            str(p.get('name') or f'element{i}')
            for i, p in enumerate(prescriptions)),
        'aperture_diameter': aperture if aperture is not None
        else (max(apertures) if apertures else None),
        'surfaces': surfaces,
        'thicknesses': thicknesses,
    }


# A-13ish (AUDIT_ADVERSARIAL_CODEBASE 2026-07-25): declare the public
# surface explicitly, matching the convention every ``analysis/`` module
# already follows.  Every name here is re-exported through
#  ``lumenairy.io.prescriptions`` (v5.1.0 split) and the top-level facade.
__all__ = [
    'scale_prescription',
    'normalize_prescription',
    'split_prescription_at_mirrors',
    'combine_prescriptions',
    'has_mirrors',
]

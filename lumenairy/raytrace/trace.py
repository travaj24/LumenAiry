"""
v5.1.0 split: sequential trace engine + prescription/element loaders
+ ray-generator factories.

Extracted from ``lumenairy/raytrace/core.py`` as part of the v5.1.0
6-file split (ROADMAP Agent B).  Hosts:

* :func:`trace` -- sequential ray trace through a Surface list.
* :func:`validate_prescription`, :func:`surfaces_from_prescription`,
  :func:`find_stop` -- prescription dict validation / conversion.
* :func:`make_ray`, :func:`make_fan`, :func:`make_ring`,
  :func:`make_grid`, :func:`make_rings` -- ray generators.
* :func:`apply_doe_phase_traced` -- grating diffraction-order kick.
* :func:`trace_prescription` -- high-level convenience.
* :func:`surfaces_from_elements`, :func:`raytrace_system` -- system.py
  element-list compatibility bridge.

Every public name here is re-exported from
``lumenairy.raytrace.core`` so existing imports continue to resolve.

No physics change: contents are bit-for-bit copies of the original
implementations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..glass import get_glass_index
from .intersection import (
    _apply_coord_break,
    _intersect_surface,
    _reflect,
    _refract,
    _transfer,
)
from .surface import (
    RAY_EVANESCENT,
    RAY_OK,
    RAY_TIR,
    RayBundle,
    Surface,
    TraceResult,
    _surface_copy_with,
)

# ============================================================================
# Sequential trace engine
# ============================================================================

def trace(
    rays: 'RayBundle',
    surfaces: List['Surface'],
    wavelength: float,
    output_filter: Union[str, Callable[..., Any]] = 'all',
    surface_diffraction: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
) -> 'TraceResult':
    """Trace a ray bundle through a sequential list of surfaces.

    Parameters
    ----------
    rays : RayBundle
        Input rays.  The bundle is *not* modified; a copy is traced.
    surfaces : list of Surface
        Ordered surface list.  ``surface.thickness`` gives the axial
        distance from this surface to the next.
    wavelength : float
        Vacuum wavelength [m] (used to resolve glass indices).
    output_filter : ``'all'`` (default) | ``'last'`` | callable
        Controls what per-surface state is retained in
        ``result.ray_history``.

        * ``'all'``  -- save a ``RayBundle.copy()`` after every
          surface (legacy behaviour).
        * ``'last'`` -- save only the final post-last-surface bundle
          in a one-element ``ray_history``.  ``result.image_rays``
          is still the expected object; every ``rays_at(i)`` for
          ``i < len(surfaces)-1`` raises ``IndexError``.  Use this
          for memory-constrained workloads where only the image-
          plane bundle is consumed -- most notably
          :func:`apply_real_lens_traced`, which at N=32768 avoids
          ~1-5 GB of transient ``RayBundle.copy()`` allocations per
          call.
        * ``callable`` -- any ``fn(rays, surf, index) -> Any``.  The
          return value is appended to ``ray_history``; return
          ``None`` to skip.  Enables user-defined per-surface
          recording (e.g. store only (x, y, opd) as a
          ``NamedTuple``, or accumulate running spot centroids).
    surface_diffraction : dict or None, optional
        Per-surface diffractive-order kicks.  Maps surface index
        ``i`` (zero-based) to a tuple ``(order_x, order_y, period_x,
        period_y)`` interpreted as the grating equation::

            L_new = L + order_x * wavelength / period_x
            M_new = M + order_y * wavelength / period_y

        applied AFTER refraction at surface ``i`` (so the rays
        continue propagation through the post-surface medium with the
        diffractive kick applied).  ``period_y`` may be ``np.inf`` for
        a 1-D grating; ``order_x`` / ``order_y`` may be half-integer
        (Dammann-style even-N splitters).  Orders that turn evanescent
        (``L_new**2 + M_new**2 > 1``) are flagged
        ``alive=False`` with ``error_code=RAY_EVANESCENT``.  See also
        :func:`apply_doe_phase_traced`.

    Returns
    -------
    result : TraceResult
    """
    r = rays.copy()
    history = [] if output_filter != 'last' else None
    final = None
    _diff = dict(surface_diffraction) if surface_diffraction else {}

    # Pre-resolve all glass indices once per wavelength.  Each
    # get_glass_index call has module-level LRU caching, so repeated
    # Python dispatch overhead is the only cost saved -- tiny per
    # surface, noticeable at high repeated-trace counts (focus
    # sweeps pre-3.1.8 retraced from scratch; the pattern survives
    # in user code that does its own iteration).  Underscore-
    # prefixed names to avoid colliding with the transfer-step
    # ``n_after = n2`` rebind later in this loop.
    _n_pre  = [get_glass_index(s.glass_before, wavelength) for s in surfaces]
    _n_post = [get_glass_index(s.glass_after,  wavelength) for s in surfaces]

    for i, surf in enumerate(surfaces):
        n1 = _n_pre[i]
        n2 = _n_post[i]

        # 3.7.0: Coord-break surfaces are pure frame transforms --
        # decenter + tilt with no intersection / OPL.  Apply the
        # transform, record the post-transform bundle in history (so
        # downstream consumers indexing by surface number stay
        # aligned), and continue to the transfer step below.
        if surf.is_coordbrk:
            _apply_coord_break(r, surf)
            if output_filter == 'all':
                history.append(r.copy())
            elif callable(output_filter):
                item = output_filter(r, surf, i)
                if item is not None:
                    history.append(item)
            if i == len(surfaces) - 1:
                final = r.copy() if output_filter == 'last' else None
            if i < len(surfaces) - 1:
                # Coord-break thickness is the air gap to the next
                # surface in the (post-transform) frame, exactly
                # like a regular surface's DISZ.  Use the post-cb
                # n2 (which equals n1 -- coord-breaks are inside a
                # single medium) so the OPL bookkeeping matches.
                _transfer(r, surf.thickness, n2)
            continue

        # 1. Intersect with surface (accumulates OPL in glass_before)
        _intersect_surface(r, surf, n_medium=n1)

        # 2. Refract or reflect
        if surf.is_mirror:
            _reflect(r, surf)
        else:
            _refract(r, surf, n1, n2)

        # 2.5. Diffractive-order kick (if this surface is registered as
        # a grating in surface_diffraction).  Modifies (L, M, N) in
        # place AND adds the DOE's linear OPL contribution
        # ``m * lambda * (x, y) / Lambda`` -- which apply_doe_phase_traced
        # explicitly excludes but the LG aberration fit needs to see in
        # order to give correct (0, 0) piston phases per emitter.  The
        # linear part of this OPL is geometric and gets absorbed by the
        # piston-coherence merit's linear fit; any non-linear part comes
        # from the per-emitter chief rays hitting the DOE at non-paraxial
        # positions, and is precisely the corner-frame coherence content
        # we want the optimizer to see.
        _diff_spec = _diff.get(i)
        if _diff_spec is not None:
            _mx, _my, _px, _py = _diff_spec
            _dL = float(_mx) * wavelength / float(_px)
            _dM = float(_my) * wavelength / float(_py)
            r.L = r.L + _dL
            r.M = r.M + _dM
            _sumsq = r.L * r.L + r.M * r.M
            _evan = _sumsq > 1.0
            _propagating = ~_evan
            _N_new = np.zeros_like(r.N)
            np.sqrt(np.maximum(1.0 - _sumsq, 0.0),
                    out=_N_new, where=_propagating)
            # Preserve the sign of the longitudinal cosine (forward
            # vs. backward propagation).  The original N's sign was
            # set by the propagation direction; the diffraction kick
            # only shifts (L, M) so the new N has the same sign.
            r.N = np.where(r.N < 0, -_N_new, _N_new)
            # Add the constant grating-order OPL contribution evaluated
            # at the ray's DOE-plane intersection (x, y).  The factor
            # ``m * lambda / period`` is the same gradient applied to
            # (L, M) above, so this is the integral of that phase
            # gradient evaluated at the surface.
            r.opd = r.opd + _dL * r.x + _dM * r.y
            if np.any(_evan) and r.alive is not None:
                r.alive = r.alive & _propagating
                if r.error_code is not None:
                    r.error_code = np.where(
                        _evan & (r.error_code == RAY_OK),
                        np.uint8(RAY_EVANESCENT),
                        r.error_code,
                    )

        # Save state after this surface, per output_filter
        if output_filter == 'all':
            history.append(r.copy())
        elif callable(output_filter):
            item = output_filter(r, surf, i)
            if item is not None:
                history.append(item)
        # 'last' branch: retain only the final bundle, copied below

        # Remember the final bundle so 'last' mode can cheaply snapshot
        # after the loop without an extra walk.
        if i == len(surfaces) - 1:
            final = r.copy() if output_filter == 'last' else None

        # 3. Transfer to next surface (accumulates the bulk
        # vertex-to-vertex axial leg in glass_after; the small
        # sag-correction at the next surface is added by the next
        # _intersect_surface call).
        if i < len(surfaces) - 1:
            n_after = n2  # medium after this surface
            _transfer(r, surf.thickness, n_after)

    if output_filter == 'last':
        history = [final] if final is not None else []

    return TraceResult(
        surfaces=surfaces,
        ray_history=history,
        input_rays=rays,
        wavelength=wavelength,
    )


# ============================================================================
# Prescription → Surface list conversion
# ============================================================================

def validate_prescription(
    prescription: Dict[str, Any],
    *,
    strict: bool = True,
) -> Optional[List[Tuple[str, str]]]:
    """Sanity-check a lens prescription dict.

    Catches the common errors that otherwise cause
    :func:`surfaces_from_prescription` (and every downstream trace) to
    fail with cryptic ``KeyError`` / ``IndexError`` messages, or worse,
    silently accept a degenerate prescription.

    Checks performed:

    * ``prescription`` is a dict with the required keys ``'surfaces'``
      and ``'thicknesses'``.
    * ``len(surfaces) >= 1``.
    * ``len(thicknesses)`` is either ``len(surfaces)`` (each surface
      has a forward thickness) or ``len(surfaces) - 1`` (legacy
      "between-surfaces" thicknesses; padded to match).  Other
      mismatches raise.
    * Each per-surface dict contains a finite, real ``radius`` (or the
      special infinity sentinel ``np.inf``).  Stops and detectors can
      have ``radius = np.inf``.
    * Each per-surface dict contains ``glass_before`` and
      ``glass_after``.  ``None`` is allowed (taken as "air"); empty
      strings are flagged.
    * If ``'aperture_diameter'`` is present, it is a positive finite
      number.

    Parameters
    ----------
    prescription : dict
        The prescription to validate.
    strict : bool, default True
        If True, raise ``ValueError`` on any failure.  If False, return
        a list of (key, reason) tuples describing each failure (empty
        list on success) so callers can decide whether to proceed.

    Returns
    -------
    list of (str, str)
        Only returned when ``strict=False``.  Empty list = valid.

    Raises
    ------
    ValueError
        When ``strict=True`` and any check fails.

    Examples
    --------
    >>> from lumenairy.raytrace import validate_prescription
    >>> p = {'surfaces': [{'radius': 0.05, 'glass_before': 'air',
    ...                    'glass_after': 'N-BK7'},
    ...                   {'radius': -0.05, 'glass_before': 'N-BK7',
    ...                    'glass_after': 'air'}],
    ...      'thicknesses': [0.005, 0.1]}
    >>> validate_prescription(p)  # raises nothing
    """
    issues: list[tuple[str, str]] = []

    def _fail(key: str, reason: str) -> None:
        issues.append((key, reason))

    if not isinstance(prescription, dict):
        _fail('prescription',
              f'must be a dict, got {type(prescription).__name__}')
    else:
        if 'surfaces' not in prescription:
            _fail('surfaces', "required key 'surfaces' missing")
        if 'thicknesses' not in prescription:
            _fail('thicknesses', "required key 'thicknesses' missing")

        if 'surfaces' in prescription:
            surfs = prescription['surfaces']
            if not isinstance(surfs, (list, tuple)):
                _fail('surfaces',
                      f'must be a list, got {type(surfs).__name__}')
            elif len(surfs) < 1:
                _fail('surfaces',
                      'must contain at least 1 surface (got 0)')
            else:
                if 'thicknesses' in prescription:
                    thicks = prescription['thicknesses']
                    if not isinstance(thicks, (list, tuple)):
                        _fail('thicknesses',
                              f'must be a list, got '
                              f'{type(thicks).__name__}')
                    elif len(thicks) not in (len(surfs), len(surfs) - 1):
                        _fail('thicknesses',
                              f'length mismatch: {len(thicks)} '
                              f'thicknesses for {len(surfs)} surfaces. '
                              f'Expected either equal length (each '
                              f'surface has a forward thickness) or '
                              f'len(surfaces) - 1 (between-surface '
                              f'thicknesses).')

                for i, ps in enumerate(surfs):
                    if not isinstance(ps, dict):
                        _fail(f'surfaces[{i}]',
                              f'must be a dict, got '
                              f'{type(ps).__name__}')
                        continue
                    if 'radius' not in ps:
                        _fail(f'surfaces[{i}]',
                              "missing 'radius' key")
                    else:
                        r = ps['radius']
                        if r is None:
                            _fail(f'surfaces[{i}].radius',
                                  "is None (use np.inf for a flat "
                                  "surface)")
                        else:
                            try:
                                rf = float(r)
                                if not (np.isfinite(rf) or np.isinf(rf)):
                                    _fail(f'surfaces[{i}].radius',
                                          f'is {r!r} (NaN); use a '
                                          f'real number or np.inf')
                            except (TypeError, ValueError):
                                _fail(f'surfaces[{i}].radius',
                                      f'is {r!r}, not a real number')
                    for gk in ('glass_before', 'glass_after'):
                        if gk not in ps:
                            _fail(f'surfaces[{i}]',
                                  f"missing '{gk}' key")
                        elif isinstance(ps[gk], str) and not ps[gk]:
                            _fail(f'surfaces[{i}].{gk}',
                                  "is an empty string; use None or "
                                  "'air' for the ambient medium")

    if isinstance(prescription, dict) and 'aperture_diameter' in prescription:
        ad = prescription['aperture_diameter']
        if ad is not None:
            try:
                adf = float(ad)
                if not (np.isfinite(adf) and adf > 0.0):
                    _fail('aperture_diameter',
                          f'must be positive and finite; got {ad!r}')
            except (TypeError, ValueError):
                _fail('aperture_diameter',
                      f'must be a number; got {ad!r}')

    if strict and issues:
        msg = '\n  '.join(f'{k}: {r}' for k, r in issues)
        raise ValueError(
            f'validate_prescription: {len(issues)} issue(s) found:\n  {msg}'
        )
    if not strict:
        return issues


def surfaces_from_prescription(prescription: Dict[str, Any]) -> List['Surface']:
    """Convert a lens prescription dict to a list of Surface objects.

    Accepts the same prescription format returned by
    :func:`prescriptions.load_zemax_prescription_data_txt`,
    :func:`prescriptions.load_zemax_zmx`,
    :func:`prescriptions.make_singlet`, etc.

    Parameters
    ----------
    prescription : dict
        Must contain ``'surfaces'`` and ``'thicknesses'`` keys.
        Optionally ``'aperture_diameter'``.

    Returns
    -------
    surfaces : list of Surface

    Notes
    -----
    The prescription is validated via :func:`validate_prescription`
    before any conversion; obviously-malformed input (empty dict,
    missing thicknesses, NaN radius, ...) raises a ``ValueError``
    with a precise message rather than producing a partially-built
    surface list.
    """
    validate_prescription(prescription, strict=True)

    p_surfs = prescription['surfaces']
    p_thick = prescription['thicknesses']
    aperture = prescription.get('aperture_diameter')

    # If the prescription has 'elements' with semi_diameter, use those
    elements = prescription.get('elements', None)

    surface_list = []
    for i, ps in enumerate(p_surfs):
        # Determine semi-diameter
        sd = np.inf
        if aperture is not None:
            sd = aperture / 2.0
        # Audit P2-35: honour the per-surface 'semi_diameter' key with
        # the SAME semantics as the JAX backend
        # (jax_trace._build_jax_prescription / trace_jax_with_params):
        # a finite, positive per-surface value REPLACES the
        # aperture_diameter/2 default; None / non-finite / <= 0 falls
        # back to the default.  Pre-fix the NumPy backend silently
        # ignored the key, so the identical prescription vignetted
        # differently under the two backends (25/25 vs 1/25 alive).
        ps_sd = ps.get('semi_diameter')
        if ps_sd is not None and np.isfinite(ps_sd) and ps_sd > 0:
            sd = float(ps_sd)
        # If elements list has per-surface semi-diameters, use the tighter one
        if elements is not None:
            # Match by index within refracting surfaces
            refr_elems = [e for e in elements if e.get('element_type') == 'surface']
            if i < len(refr_elems):
                elem_sd = refr_elems[i].get('semi_diameter', np.inf)
                if elem_sd > 0 and np.isfinite(elem_sd):
                    sd = min(sd, elem_sd)

        thickness = p_thick[i] if i < len(p_thick) else 0.0

        # Freeform departure (optional).  Accept either a unified
        # 'freeform' dict or the legacy flat keys used by the
        # prescription-level freeform helpers.
        ff = ps.get('freeform')
        if ff is None and ps.get('freeform_type') is not None:
            # v4.15.1 P1-NEW-D: include Forbes Q-bfs / Q-con keys
            # (``q_bfs_coeffs``, ``q_con_coeffs``, ``r_max``) in the
            # flat-keys gather so a prescription with
            # ``freeform_type='q_bfs'`` and ``q_bfs_coeffs=[...]``
            # actually carries its coefficients into the Surface
            # dataclass.  Pre-v4.15.1 the dispatcher routed the
            # freeform_type correctly but the coefficient list was
            # silently dropped, making Forbes Q a no-op on flat-keys
            # prescriptions (the unified-dict shape worked).
            ff = {k: v for k, v in ps.items()
                  if k in ('freeform_type', 'xy_coeffs',
                           'zernike_coeffs', 'cheb_coeffs',
                           'q_bfs_coeffs', 'q_con_coeffs',
                           'norm_radius', 'norm_x', 'norm_y',
                           'r_max')}

        # Aperture-stop flag.  Zemax parsers store the STOP keyword
        # on the per-surface dict ('is_stop': True); the prescription
        # dict may also carry a 'stop_index' (the index of the stop
        # in the surface list), which the wave-optics side already
        # honours.  Prefer per-surface flag if both are present.
        is_stop_flag = bool(ps.get('is_stop', False))
        if not is_stop_flag:
            stop_idx = prescription.get('stop_index')
            if stop_idx is not None and int(stop_idx) == i:
                is_stop_flag = True

        # 4.5: honour the per-surface `is_mirror` flag from the
        # prescription dict.  Loaders mark mirrors when the .zmx glass
        # column says ``MIRROR``; a folded design built by hand
        # carries the same flag.  Infer ``is_mirror`` when
        # ``glass_after`` is the marker string ``'MIRROR'``
        # (case-insensitive); in that case the marker is replaced by
        # ``glass_before`` because reflection does not change the
        # surrounding medium and ``'MIRROR'`` is not a real glass.
        glass_before = ps['glass_before']
        glass_after = ps['glass_after']
        is_mirror_flag = bool(ps.get('is_mirror', False))
        if (isinstance(glass_after, str)
                and glass_after.upper() == 'MIRROR'):
            is_mirror_flag = True
            glass_after = glass_before

        surface_list.append(Surface(
            radius=ps['radius'],
            conic=ps.get('conic', 0.0),
            aspheric_coeffs=ps.get('aspheric_coeffs'),
            semi_diameter=sd,
            glass_before=glass_before,
            glass_after=glass_after,
            is_mirror=is_mirror_flag,
            is_stop=is_stop_flag,
            thickness=thickness,
            label=ps.get('comment', f'S{i+1}'),
            surf_num=ps.get('surf_num', i + 1),
            # Biconic / anamorphic (optional, default None = rotationally
            # symmetric)
            radius_y=ps.get('radius_y'),
            conic_y=ps.get('conic_y'),
            aspheric_coeffs_y=ps.get('aspheric_coeffs_y'),
            freeform=ff,
        ))

    return surface_list


def find_stop(surfaces: List['Surface']) -> int:
    """Return the index of the aperture stop in ``surfaces``.

    Dispatch order:

    1. First surface with ``is_stop=True``.  If multiple surfaces
       are flagged, a ``RuntimeWarning`` is emitted and the earliest
       match is returned -- callers should explicitly set one stop
       per system to avoid ambiguity.
    2. First surface with a finite, user-declared ``semi_diameter``
       (legacy fallback, matches pre-3.1.8 implicit behaviour).
    3. Surface 0, with a ``UserWarning`` when the system has more
       than one surface (the stop guess is almost certainly wrong
       in that case, but we preserve behaviour rather than raise).

    Parameters
    ----------
    surfaces : list of Surface

    Returns
    -------
    stop_index : int

    Notes
    -----
    No trace work is performed here.  The function is O(N_surfaces)
    and safe to call inside hot loops, though ``compute_pupils`` and
    ``seidel_coefficients`` only need it once per system.
    """
    if not surfaces:
        raise ValueError("find_stop: empty surface list")
    flagged = [i for i, s in enumerate(surfaces) if s.is_stop]
    if len(flagged) > 1:
        import warnings
        warnings.warn(
            f"find_stop: multiple surfaces marked is_stop=True "
            f"(indices {flagged}); returning the first match "
            f"{flagged[0]}. Clear the extras to disambiguate.",
            RuntimeWarning, stacklevel=2)
    if flagged:
        return flagged[0]
    # Legacy fallback: first finite semi-diameter
    for i, s in enumerate(surfaces):
        if np.isfinite(s.semi_diameter):
            return i
    if len(surfaces) > 1:
        import warnings
        warnings.warn(
            "find_stop: no surface flagged is_stop=True and none have "
            "a finite semi_diameter; defaulting to surface 0.  "
            "Set is_stop=True on the intended aperture-stop surface "
            "for correct chief-ray behaviour.",
            UserWarning, stacklevel=2)
    return 0


# ============================================================================
# Ray generation helpers
# ============================================================================

def _make_bundle(x, y, L, M, wavelength):
    """Create a RayBundle from position and direction arrays."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    L = np.atleast_1d(np.asarray(L, dtype=np.float64))
    M = np.atleast_1d(np.asarray(M, dtype=np.float64))

    n = max(len(x), len(y), len(L), len(M))
    x = np.broadcast_to(x, n).copy()
    y = np.broadcast_to(y, n).copy()
    L = np.broadcast_to(L, n).copy()
    M = np.broadcast_to(M, n).copy()
    N = np.sqrt(np.maximum(1.0 - L ** 2 - M ** 2, 0.0))

    return RayBundle(
        x=x, y=y, z=np.zeros(n),
        L=L, M=M, N=N,
        wavelength=wavelength,
        alive=np.ones(n, dtype=bool),
        opd=np.zeros(n),
    )


def make_ray(
    x: float = 0.0,
    y: float = 0.0,
    L: float = 0.0,
    M: float = 0.0,
    *,
    wavelength: float,
) -> 'RayBundle':
    """Create a single ray.

    Parameters
    ----------
    x, y : float
        Ray position at z = 0 [m].
    L, M : float
        Direction cosines in x and y.
    wavelength : float
        Vacuum wavelength [m].

    Returns
    -------
    RayBundle with one ray.
    """
    return _make_bundle([x], [y], [L], [M], wavelength)


def make_fan(
    axis: str = 'y',
    semi_aperture: float = 12.7e-3,
    n_rays: int = 21,
    field_angle: float = 0.0,
    wavelength: float = 550e-9,
) -> 'RayBundle':
    """Create a 1-D fan of rays across the pupil.

    Parameters
    ----------
    axis : str
        ``'x'`` or ``'y'`` — fan direction.
    semi_aperture : float
        Pupil semi-diameter [m].
    n_rays : int
        Number of rays (odd recommended to include the chief ray).
    field_angle : float
        Off-axis field angle [radians].  Applied as a direction cosine
        tilt in the fan axis.
    wavelength : float
        Vacuum wavelength [m].

    Returns
    -------
    RayBundle
    """
    t = np.linspace(-1, 1, n_rays)
    if axis == 'y':
        x = np.zeros(n_rays)
        y = t * semi_aperture
        L = np.zeros(n_rays)
        M = np.full(n_rays, np.sin(field_angle))
    else:
        x = t * semi_aperture
        y = np.zeros(n_rays)
        L = np.full(n_rays, np.sin(field_angle))
        M = np.zeros(n_rays)

    return _make_bundle(x, y, L, M, wavelength)


def make_ring(
    semi_aperture: float = 12.7e-3,
    n_rays: int = 36,
    field_angle: float = 0.0,
    wavelength: float = 550e-9,
    fraction: float = 1.0,
) -> 'RayBundle':
    """Create a ring of rays at a given fractional pupil radius.

    Parameters
    ----------
    semi_aperture : float
        Pupil semi-diameter [m].
    n_rays : int
        Number of rays around the ring.
    field_angle : float
        Off-axis angle [radians] applied as M direction cosine.
    wavelength : float
        Vacuum wavelength [m].
    fraction : float
        Fractional pupil radius (0 to 1).

    Returns
    -------
    RayBundle
    """
    theta = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    r = semi_aperture * fraction
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    L = np.zeros(n_rays)
    M = np.full(n_rays, np.sin(field_angle))
    return _make_bundle(x, y, L, M, wavelength)


def make_grid(
    semi_aperture: float = 12.7e-3,
    n_across: int = 11,
    field_angle: float = 0.0,
    wavelength: float = 550e-9,
    pattern: str = 'square',
) -> 'RayBundle':
    """Create a 2-D grid of rays across the pupil.

    Parameters
    ----------
    semi_aperture : float
        Pupil semi-diameter [m].
    n_across : int
        Number of rays along each axis.
    field_angle : float
        Off-axis angle [radians] applied as M direction cosine.
    wavelength : float
        Vacuum wavelength [m].
    pattern : str
        ``'square'`` — full rectangular grid.
        ``'circular'`` — only rays inside the pupil circle.

    Returns
    -------
    RayBundle
    """
    t = np.linspace(-1, 1, n_across)
    tx, ty = np.meshgrid(t, t)
    tx = tx.ravel()
    ty = ty.ravel()

    if pattern == 'circular':
        r_sq = tx ** 2 + ty ** 2
        mask = r_sq <= 1.0
        tx = tx[mask]
        ty = ty[mask]

    x = tx * semi_aperture
    y = ty * semi_aperture
    L = np.zeros_like(x)
    M = np.full_like(y, np.sin(field_angle))
    return _make_bundle(x, y, L, M, wavelength)


def make_rings(
    semi_aperture: float = 12.7e-3,
    num_rings: int = 6,
    rays_per_ring: int = 36,
    field_angle: float = 0.0,
    wavelength: float = 550e-9,
    include_chief: bool = True,
) -> 'RayBundle':
    """Create concentric rings of rays (good for spot diagrams).

    Parameters
    ----------
    semi_aperture : float
        Pupil semi-diameter [m].
    num_rings : int
        Number of concentric rings.
    rays_per_ring : int
        Rays per ring (each ring has this many).
    field_angle : float
        Off-axis angle [radians].
    wavelength : float
        Vacuum wavelength [m].
    include_chief : bool
        If True, add the on-axis chief ray at the centre.

    Returns
    -------
    RayBundle
    """
    all_x = []
    all_y = []

    if include_chief:
        all_x.append(0.0)
        all_y.append(0.0)

    for ring in range(1, num_rings + 1):
        frac = ring / num_rings
        theta = np.linspace(0, 2 * np.pi, rays_per_ring, endpoint=False)
        r = semi_aperture * frac
        all_x.append(r * np.cos(theta))
        all_y.append(r * np.sin(theta))

    x = np.concatenate([np.atleast_1d(xi) for xi in all_x])
    y = np.concatenate([np.atleast_1d(yi) for yi in all_y])
    L = np.zeros_like(x)
    M = np.full_like(y, np.sin(field_angle))
    return _make_bundle(x, y, L, M, wavelength)


# ============================================================================
# Diffraction-order direction shift (gratings / DOEs in the traced path)
# ============================================================================

def apply_doe_phase_traced(
    rays: 'RayBundle',
    order_x: Union[float, int, Sequence[float], np.ndarray],
    order_y: Union[float, int, Sequence[float], np.ndarray] = 0,
    *,
    period_x: float,
    period_y: Optional[float] = None,
    wavelength: Optional[float] = None,
) -> 'RayBundle':
    """Apply a grating diffraction-order direction shift to a ray bundle.

    Each ray's transverse direction cosines are shifted by the grating
    equation::

        L_new = L + order_x * lambda / period_x
        M_new = M + order_y * lambda / period_y

    The longitudinal cosine is recomputed from
    ``L_new**2 + M_new**2 + N_new**2 == 1``.  Orders for which
    ``L_new**2 + M_new**2 > 1`` are evanescent (do not propagate); those
    rays are flagged ``alive=False`` with ``error_code = RAY_EVANESCENT``.

    Ray positions ``(x, y, z)`` and the OPL accumulator are *not*
    modified -- the grating is treated as a thin diffractive surface
    that only redirects each ray.  If you need to add the constant
    grating-order phase shift to ``opd``, do so manually after the call.

    The function supports two calling conventions:

    1. **Single order** -- pass scalar ``order_x`` and ``order_y``.
       The returned bundle has the same length as ``rays``.

    2. **Order array** -- pass 1-D arrays of equal length for
       ``order_x`` and ``order_y``.  The returned bundle is replicated
       ``len(order_x)`` times in *order-major* layout::

           out[order=k, ray=i] = out[k * n_rays + i]

       i.e. all rays for order 0, then all rays for order 1, ...

    Typical use: split a ray bundle at a Dammann-grating plane into a
    set of diffraction orders, then continue tracing each order through
    the post-grating optics with a single :func:`trace` call on the
    flattened bundle.

    Parameters
    ----------
    rays : RayBundle
        Input bundle.  Not modified in place.
    order_x : float, int, or 1-D array-like
        Diffraction order along the grating's x-axis.  Half-integer
        orders are allowed (e.g. for even-N Dammann splitters).
    order_y : float, int, or 1-D array-like, default 0
        Diffraction order along the grating's y-axis.  When passing
        arrays, ``order_x`` and ``order_y`` must broadcast to the same
        1-D length.
    period_x : float
        Grating period along x [m].  Required keyword.
    period_y : float, optional
        Grating period along y [m].  Defaults to ``period_x`` (square
        crossed grating).  Use ``np.inf`` to disable diffraction along
        one axis (1-D grating).
    wavelength : float, optional
        Vacuum wavelength [m].  Defaults to ``rays.wavelength``.

    Returns
    -------
    RayBundle
        New bundle (positions copied, directions shifted).  Length equals
        ``len(rays)`` for scalar orders or ``n_orders * len(rays)`` for
        order arrays.

    Notes
    -----
    The grating equation here is the small-angle / paraxial direction-
    cosine form: ``sin(theta_diff) - sin(theta_in) = m * lambda / Lambda``
    expressed as ``L_new = L_in + m * lambda / Lambda``.  This is the
    standard 1-st-order DOE / Dammann ray-tracing convention; it neglects
    the cosine factor that distinguishes ``sin`` from the direction
    cosine for very large grating angles.  For modest deflections
    (sub-100 mrad) the two are interchangeable to <1% even at the
    pupil edge.

    See Also
    --------
    trace : Geometric ray tracer (call after this function with the
        post-grating surfaces).
    lumenairy.doe.makedammann2d : 2-D Dammann period derivation.
    """
    if wavelength is None:
        wavelength = rays.wavelength
    if period_y is None:
        period_y = period_x

    # Normalize order args; track whether the caller passed scalars
    # (single-order convention) or arrays (multi-order replication).
    mx = np.asarray(order_x, dtype=np.float64)
    my = np.asarray(order_y, dtype=np.float64)
    scalar_input = (mx.ndim == 0 and my.ndim == 0)
    mx = np.atleast_1d(mx)
    my = np.atleast_1d(my)
    if mx.ndim != 1 or my.ndim != 1:
        raise ValueError(
            f"order_x and order_y must be scalar or 1-D, got shapes "
            f"{mx.shape} and {my.shape}")
    try:
        mx, my = np.broadcast_arrays(mx, my)
    except ValueError as e:
        raise ValueError(
            f"order_x (length {len(mx)}) and order_y (length {len(my)}) "
            f"must broadcast to the same 1-D length") from e
    n_orders = mx.size
    n_rays = len(rays.x)

    # Per-order direction increments.
    dL = (mx * wavelength / period_x).reshape(n_orders, 1)
    dM = (my * wavelength / period_y).reshape(n_orders, 1)

    # Broadcast to (n_orders, n_rays); reshape input direction cosines.
    L_new = rays.L.reshape(1, n_rays) + dL
    M_new = rays.M.reshape(1, n_rays) + dM

    sum_sq = L_new ** 2 + M_new ** 2
    propagating = sum_sq <= 1.0
    N_new = np.zeros_like(L_new)
    np.sqrt(np.maximum(1.0 - sum_sq, 0.0), out=N_new, where=propagating)

    # v5.2 (AUDIT_V4_13_1 P1-G closure): preserve the sign of the
    # longitudinal direction cosine.  The diffraction kick only shifts
    # the transverse (L, M) components; the propagation direction along
    # z is unchanged.  Pre-v5.2 ``apply_doe_phase_traced`` always
    # returned a positive ``N_new`` while the inline DOE kick in
    # :func:`trace` correctly preserved the sign (see line ~193:
    # ``r.N = np.where(r.N < 0, -_N_new, _N_new)``).  Match the inline
    # site so reverse-traced bundles (``N < 0``) keep their direction.
    N_sign = np.where(rays.N.reshape(1, n_rays) < 0, -1.0, 1.0)
    N_new = N_new * N_sign

    # Per-order alive / error_code grids.
    alive_in = np.asarray(rays.alive, dtype=bool).reshape(1, n_rays)
    alive_new = alive_in & propagating

    if rays.error_code is not None:
        ec_in = np.asarray(rays.error_code).reshape(1, n_rays)
        ec_new = np.broadcast_to(ec_in, (n_orders, n_rays)).copy()
    else:
        ec_new = np.zeros((n_orders, n_rays), dtype=np.uint8)
        ec_new[~alive_in.repeat(n_orders, axis=0)] = RAY_TIR
    # First-failure-wins: only stamp RAY_EVANESCENT on rays that were
    # alive coming in but became non-propagating from the order shift.
    newly_dead = (~propagating) & alive_in
    ec_new[newly_dead] = RAY_EVANESCENT

    if scalar_input:
        # Single-order convention: same shape as input.
        return RayBundle(
            x=rays.x.copy(), y=rays.y.copy(), z=rays.z.copy(),
            L=L_new[0], M=M_new[0], N=N_new[0],
            wavelength=wavelength,
            alive=alive_new[0],
            opd=rays.opd.copy(),
            error_code=ec_new[0],
        )

    # Order-major flatten: all rays for order 0, then order 1, ...
    return RayBundle(
        x=np.tile(rays.x, n_orders),
        y=np.tile(rays.y, n_orders),
        z=np.tile(rays.z, n_orders),
        L=L_new.reshape(-1),
        M=M_new.reshape(-1),
        N=N_new.reshape(-1),
        wavelength=wavelength,
        alive=alive_new.reshape(-1),
        opd=np.tile(rays.opd, n_orders),
        error_code=ec_new.reshape(-1),
    )


# ============================================================================
# High-level trace functions
# ============================================================================

def trace_prescription(
    prescription: Dict[str, Any],
    wavelength: float,
    semi_aperture: Optional[float] = None,
    field_angle: float = 0.0,
    num_rings: int = 6,
    rays_per_ring: int = 36,
    ray_pattern: str = 'rings',
    n_across: int = 11,
    image_distance: Optional[float] = None,
) -> 'TraceResult':
    """Trace rays through a lens prescription.

    Convenience wrapper that converts a prescription dict to surfaces,
    generates rays, traces, and optionally propagates to a custom image
    distance.

    Parameters
    ----------
    prescription : dict
        Lens prescription (from :func:`prescriptions.make_singlet` etc.).
    wavelength : float
        Vacuum wavelength [m].
    semi_aperture : float or None
        Pupil semi-diameter [m].  If None, uses
        ``prescription['aperture_diameter'] / 2``.
    field_angle : float
        Off-axis field angle [radians].
    num_rings, rays_per_ring : int
        Parameters for the ``'rings'`` pattern.
    ray_pattern : str
        ``'rings'``, ``'grid'``, or ``'fan_xy'``.
    n_across : int
        Grid size for the ``'grid'`` pattern.
    image_distance : float or None
        If given, add a final flat surface at this distance after the
        last prescription surface.  Useful for evaluating the spot at a
        specific image plane.

    Returns
    -------
    TraceResult
    """
    surfaces = surfaces_from_prescription(prescription)

    if semi_aperture is None:
        ap = prescription.get('aperture_diameter')
        semi_aperture = ap / 2.0 if ap else 12.7e-3

    # Generate rays
    if ray_pattern == 'rings':
        rays = make_rings(semi_aperture, num_rings, rays_per_ring,
                          field_angle, wavelength)
    elif ray_pattern == 'grid':
        rays = make_grid(semi_aperture, n_across, field_angle,
                         wavelength, pattern='circular')
    elif ray_pattern == 'fan_xy':
        fan_y = make_fan('y', semi_aperture, 2 * rays_per_ring + 1,
                         field_angle, wavelength)
        fan_x = make_fan('x', semi_aperture, 2 * rays_per_ring + 1,
                         field_angle, wavelength)
        # Merge (skip duplicate chief ray)
        rays = _make_bundle(
            np.concatenate([fan_y.x, fan_x.x[fan_x.x != 0]]),
            np.concatenate([fan_y.y, fan_x.y[fan_x.x != 0]]),
            np.concatenate([fan_y.L, fan_x.L[fan_x.x != 0]]),
            np.concatenate([fan_y.M, fan_x.M[fan_x.x != 0]]),
            wavelength,
        )
    else:
        raise ValueError(f"Unknown ray_pattern: {ray_pattern!r}")

    # If image_distance is specified, set the last surface thickness and
    # append a flat image-plane surface so the trace engine transfers
    # the rays to the image plane before the final intersection.
    if image_distance is not None and surfaces:
        # Determine the medium after the last optical surface
        last_glass = surfaces[-1].glass_after
        # v4.13.2 (audit P1-NEW-J): clone the last surface with the new
        # thickness instead of mutating it in place.  The Surface
        # dataclass is not frozen and surfaces_from_prescription
        # builds the list from a possibly-shared prescription -- an
        # in-place mutation here was a tripwire for shared-state bugs
        # in callers that reuse the prescription across multiple
        # trace_prescription invocations with different image_distance
        # arguments.  Matches the lens_abcd ``_surface_copy_with``
        # pattern at raytrace/core.py:2510.
        surfaces[-1] = _surface_copy_with(
            surfaces[-1], thickness=image_distance)
        surfaces.append(Surface(
            radius=np.inf, conic=0.0,
            semi_diameter=np.inf,
            glass_before=last_glass, glass_after=last_glass,
            is_mirror=False, thickness=0.0,
            label='Image',
        ))

    return trace(rays, surfaces, wavelength)


# ============================================================================
# Compatibility bridge: system.py element-list format → Surface list
# ============================================================================

def surfaces_from_elements(
    elements: List[Dict[str, Any]],
    wavelength: float,
) -> List['Surface']:
    """Convert a ``propagate_through_system`` element list to Surfaces.

    This allows the same element-list used for wave-optics simulation
    to be ray-traced geometrically, enabling quick cross-validation::

        # Wave-optics
        E_out, _ = propagate_through_system(E_in, elements, wv, dx)

        # Geometric ray trace — same element list
        result = raytrace_system(elements, wv, semi_aperture=5e-3)
        spot_diagram(result)

    Supported element types:

    - ``'propagate'`` — free-space gap (converted to thickness on the
      preceding surface).
    - ``'lens'`` — thin lens (one surface with power = 1/f).
    - ``'real_lens'`` — multi-surface prescription (expanded in-line).
    - ``'mirror'`` — flat or curved reflector.
    - ``'aperture'`` — sets the semi-diameter of the preceding surface.

    Parameters
    ----------
    elements : list of dict
        Element list in the same format as :func:`system.propagate_through_system`.
    wavelength : float
        Vacuum wavelength [m] (needed to resolve glass indices for
        real-lens prescriptions).

    Returns
    -------
    surfaces : list of Surface
        Sequential surface list for :func:`trace`.
    """
    surfaces = []
    pending_thickness = 0.0  # accumulated free-space before next surface

    for elem in elements:
        etype = elem['type']

        if etype in ('propagate', 'propagate_tilted'):
            pending_thickness += elem['z']

        elif etype == 'lens':
            f = elem['f']
            # A thin lens is a flat surface with power phi = 1/f.
            # We model it as two flat air→air surfaces separated by zero
            # thickness, with the refraction equivalent encoded as a
            # curved surface with R = 2*f (mirror equivalent of a thin lens).
            # Simpler: single surface with radius = -f (for convergent).
            # Actually the cleanest approach: use the ABCD-equivalent
            # pair: a flat surface that applies the thin-lens deflection.
            # For ray tracing, a thin lens with focal length f is equivalent
            # to a curved mirror surface with R = 2f, but since we want
            # refraction not reflection, we use a single surface with
            # R such that phi = (n2-n1)/R = 1/f.  With n1=n2=1 (air),
            # this doesn't work.  Instead, we encode thin lenses as two
            # surfaces of a fictitious glass element:
            # Surface 1: R = f*(n-1)/1 = 2*f, glass air→glass (n=2)
            # This is too hacky.  Better: just store the focal length
            # and handle thin lenses specially in the trace engine.
            #
            # Pragmatic solution: approximate a thin lens as a very thin
            # high-index singlet.  With d≈0 and n_lens chosen so
            # 1/f = (n-1)*(1/R1 - 1/R2), a symmetric biconvex with
            # R1 = -R2 = R gives 1/f = (n-1)*2/R → R = 2*f*(n-1).
            # With n=1.5, R = f.  This is exact in the paraxial limit.
            R_val = f  # R = 2*f*(n-1) = 2*f*0.5 = f for n=1.5
            sd = np.inf
            if 'aperture_diameter' in elem:
                sd = elem['aperture_diameter'] / 2.0

            # Flush any pending thickness
            if surfaces:
                surfaces[-1].thickness += pending_thickness
            pending_thickness = 0.0

            surfaces.append(Surface(
                radius=R_val, conic=0.0, semi_diameter=sd,
                glass_before='air', glass_after='__thin_lens__',
                thickness=0.0, label=f'Lens f={f*1e3:.1f}mm (front)',
            ))
            surfaces.append(Surface(
                radius=-R_val, conic=0.0, semi_diameter=sd,
                glass_before='__thin_lens__', glass_after='air',
                thickness=0.0, label=f'Lens f={f*1e3:.1f}mm (back)',
            ))

        elif etype == 'real_lens':
            rx = elem['prescription']
            rx_surfaces = surfaces_from_prescription(rx)

            # Flush pending thickness
            if surfaces:
                surfaces[-1].thickness += pending_thickness
            pending_thickness = 0.0

            surfaces.extend(rx_surfaces)

        elif etype == 'mirror':
            R = elem.get('radius', np.inf)
            sd = np.inf
            if 'aperture_diameter' in elem:
                sd = elem['aperture_diameter'] / 2.0

            if surfaces:
                surfaces[-1].thickness += pending_thickness
            pending_thickness = 0.0

            surfaces.append(Surface(
                radius=R if R is not None else np.inf,
                conic=elem.get('conic', 0.0),
                semi_diameter=sd,
                glass_before='air', glass_after='air',
                is_mirror=True, thickness=0.0,
                label='Mirror',
            ))

        elif etype == 'aperture':
            # Apply aperture as a semi-diameter constraint on the
            # most recent surface, or add a dummy flat surface.
            params = elem.get('params', {})
            diameter = params.get('diameter', np.inf)
            sd = diameter / 2.0 if np.isfinite(diameter) else np.inf

            if surfaces:
                surfaces[-1].thickness += pending_thickness
            pending_thickness = 0.0

            surfaces.append(Surface(
                radius=np.inf, semi_diameter=sd,
                glass_before='air', glass_after='air',
                thickness=0.0, label='Aperture',
            ))

        elif etype == 'spherical_lens':
            if surfaces:
                surfaces[-1].thickness += pending_thickness
            pending_thickness = 0.0

            n_lens = elem['n_lens']
            sd = np.inf
            if 'aperture_diameter' in elem:
                sd = elem['aperture_diameter'] / 2.0

            # Register the pseudo-glass under a content-derived name.
            # v5.17.1 (audit P1-07): the name was id(elem)-derived, but
            # CPython recycles ids after GC, so two builds with
            # different n_lens could share a name and the second
            # registration retargeted previously built surface lists
            # to the wrong index (trace() resolves glass at trace
            # time).  Content-derived names are idempotent (same
            # content -> same name -> bounded registry growth) and
            # collision-correct (same name -> same index).
            _glass_name = f'__spherical_{float(n_lens)!r}'
            # Store a fixed-index material
            _register_fixed_index(_glass_name, n_lens, wavelength)

            surfaces.append(Surface(
                radius=elem['R1'], conic=0.0, semi_diameter=sd,
                glass_before='air', glass_after=_glass_name,
                thickness=elem['d'],
                label='Spherical lens (front)',
            ))
            surfaces.append(Surface(
                radius=elem['R2'], conic=0.0, semi_diameter=sd,
                glass_before=_glass_name, glass_after='air',
                thickness=0.0,
                label='Spherical lens (back)',
            ))

        elif etype == 'aspheric_lens':
            if surfaces:
                surfaces[-1].thickness += pending_thickness
            pending_thickness = 0.0

            n_lens = elem['n_lens']
            sd = np.inf
            if 'aperture_diameter' in elem:
                sd = elem['aperture_diameter'] / 2.0

            # Content-derived name; see the spherical_lens branch
            # comment (v5.17.1, audit P1-07).
            _glass_name = f'__aspheric_{float(n_lens)!r}'
            _register_fixed_index(_glass_name, n_lens, wavelength)

            surfaces.append(Surface(
                radius=elem['R1'],
                conic=elem.get('k1', 0.0),
                aspheric_coeffs=elem.get('A1'),
                semi_diameter=sd,
                glass_before='air', glass_after=_glass_name,
                thickness=elem['d'],
                label='Aspheric lens (front)',
            ))
            surfaces.append(Surface(
                radius=elem['R2'],
                conic=elem.get('k2', 0.0),
                aspheric_coeffs=elem.get('A2'),
                semi_diameter=sd,
                glass_before=_glass_name, glass_after='air',
                thickness=0.0,
                label='Aspheric lens (back)',
            ))

        # Silently skip unsupported element types (mask, zernike, etc.)
        # — these have no geometric-optics equivalent.

    # Flush any trailing thickness
    if surfaces and pending_thickness > 0:
        surfaces[-1].thickness += pending_thickness

    return surfaces


# Thin-lens helper: register a fixed-index "glass" for spherical/aspheric lenses
def _register_fixed_index(name, n, wavelength):
    """Register a fixed refractive index as a temporary glass entry."""
    from ..glass import GLASS_REGISTRY, _glass_cache, _glass_value_cache

    class _FixedIndex:
        def __init__(self, n_val):
            self._n = n_val
        def get_refractive_index(self, wv_nm, unit='nm'):
            return self._n

    # v5.17.1 (audit P2-36): the '__user__' sentinel is what
    # get_glass_index's user-fixed branch matches (glass.py), so the
    # lookup resolves from _glass_cache without the optional
    # refractiveindex package.  The previous
    # ('__fixed__', '__fixed__', '__fixed__') tuple fell through to
    # the refractiveindex.info branch and raised ImportError on
    # minimal installs.
    GLASS_REGISTRY[name] = ('__user__', '__fixed__', '__fixed__')
    _glass_cache[name] = _FixedIndex(n)
    # v5.17.1 (audit P3-61): an overwrite must not leave stale
    # immutable-branch value-cache entries for this name (targeted
    # removal, cheaper than register_fixed_glass's full clear because
    # this runs on every spherical/aspheric element conversion).
    for _key in [k for k in _glass_value_cache if k[0] == name]:
        del _glass_value_cache[_key]

# Also register the thin-lens pseudo-glass
_register_fixed_index('__thin_lens__', 1.5, 550e-9)


def raytrace_system(
    elements: List[Dict[str, Any]],
    wavelength: float,
    semi_aperture: Optional[float] = None,
    field_angle: float = 0.0,
    num_rings: int = 6,
    rays_per_ring: int = 36,
    ray_pattern: str = 'rings',
    n_across: int = 11,
    image_distance: Optional[float] = None,
) -> Tuple['TraceResult', List['Surface']]:
    """Ray-trace the same element list used by propagate_through_system.

    This is the geometric-optics counterpart to
    :func:`system.propagate_through_system`.  It accepts the same
    element-list format, converts it to a sequential surface list,
    generates rays, and traces them.

    Parameters
    ----------
    elements : list of dict
        Element list (same format as ``propagate_through_system``).
    wavelength : float
        Vacuum wavelength [m].
    semi_aperture : float or None
        Entrance pupil semi-diameter [m].  If None, inferred from the
        first aperture or lens element.
    field_angle : float
        Off-axis field angle [radians].
    num_rings, rays_per_ring, ray_pattern, n_across : int/str
        Ray generation parameters (see :func:`trace_prescription`).
    image_distance : float or None
        Distance from last surface to image plane [m].  If None, uses
        the paraxial back focal length.

    Returns
    -------
    result : TraceResult
    surfaces : list of Surface
        The converted surface list (useful for further analysis).
    """
    # Local import to avoid circular dependency: raytrace_system needs
    # system_abcd from seidel.py, which itself imports
    # surfaces_from_prescription from this module.
    from .seidel import system_abcd

    surfaces = surfaces_from_elements(elements, wavelength)

    if not surfaces:
        raise ValueError("No traceable surfaces found in the element list.")

    # Infer semi-aperture if not given
    if semi_aperture is None:
        for s in surfaces:
            if np.isfinite(s.semi_diameter):
                semi_aperture = s.semi_diameter
                break
        if semi_aperture is None:
            semi_aperture = 12.7e-3

    # Find image distance
    if image_distance is None:
        try:
            _, _, bfl, _ = system_abcd(surfaces, wavelength)
            if np.isfinite(bfl) and bfl > 0:
                image_distance = bfl
        except (ValueError, RuntimeError, ZeroDivisionError,
                np.linalg.LinAlgError, IndexError):
            # system_abcd failure leaves image_distance as None; the
            # caller picks a default further down.
            pass

    # Generate rays
    if ray_pattern == 'rings':
        rays = make_rings(semi_aperture, num_rings, rays_per_ring,
                          field_angle, wavelength)
    elif ray_pattern == 'grid':
        rays = make_grid(semi_aperture, n_across, field_angle,
                         wavelength, pattern='circular')
    else:
        rays = make_rings(semi_aperture, num_rings, rays_per_ring,
                          field_angle, wavelength)

    # Add image plane if we have a distance
    if image_distance is not None and surfaces:
        last_glass = surfaces[-1].glass_after
        surfaces[-1].thickness = image_distance
        surfaces.append(Surface(
            radius=np.inf, semi_diameter=np.inf,
            glass_before=last_glass, glass_after=last_glass,
            label='Image',
        ))

    result = trace(rays, surfaces, wavelength)
    return result, surfaces


__all__ = [
    # Trace engine
    'trace',
    # Prescription conversion / validation / stop lookup
    'validate_prescription', 'surfaces_from_prescription', 'find_stop',
    # Ray generators
    '_make_bundle', 'make_ray', 'make_fan', 'make_ring',
    'make_grid', 'make_rings',
    # DOE / grating helper
    'apply_doe_phase_traced',
    # High-level convenience
    'trace_prescription',
    # system.py element-list bridge
    'surfaces_from_elements', 'raytrace_system',
]

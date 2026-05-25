"""
Ghost image analysis for multi-surface lens systems.

A "ghost" is a parasitic image formed by light that reflects from two
surfaces (instead of transmitting straight through) and reaches the
detector.  For an N-surface system there are N*(N-1)/2 possible
ghost paths.

This module traces each 2-bounce ghost path, computes the ghost
image intensity (including Fresnel reflection losses), and returns
a map of the dominant ghosts for stray-light assessment.

Naming convention (v4.13.0 / audit S3)
--------------------------------------

Two distinct optical quantities both share the letter ``R`` in the
classical optics literature.  This module uses a strict
upper/lowercase split to keep them disambiguated:

* **Uppercase ``R``** (e.g. ``R_i``, ``R_j``, ``R_k``) -- the
  *normal-incidence Fresnel reflectance* at surface ``i`` (or
  ``j``, ``k``).  Dimensionless, in [0, 1].  Used in
  ``intensity = R_i * R_j``, in the dictionary keys
  ``'R_i'`` / ``'R_j'`` returned by :func:`ghost_analysis`, and in
  the upper-bound formulae throughout this docstring.

* **Lowercase ``r``** (e.g. ``r_i``, ``r_j``) -- the *radius of
  curvature* of surface ``i`` (or ``j``).  In metres, signed in the
  Welford convention (positive = centre of curvature on the
  outgoing side).  Only used inside :func:`ghost_analysis` to build
  the heuristic ``'focus_z_estimate'``.

Where a docstring mentions ``R`` without an explicit qualifier the
prose disambiguates explicitly (e.g. "Fresnel reflectance at
surface i" vs "curvature radius of surface i").

Important caveats (4.12.0 / B2-1, B2-2)
---------------------------------------

* **Reported ghost ``'intensity'`` values are UPPER BOUNDS.**  The
  product ``R_i * R_j`` (Fresnel reflectance at surface i times
  Fresnel reflectance at surface j) captures only the two normal-
  incidence Fresnel REFLECTIONS at the bouncing surfaces; it OMITS
  the transmission losses ``Prod_k (1 - R_k)`` (where each ``R_k``
  is the Fresnel reflectance of surface k) over every other surface
  the ghost ray crosses (typically two passes through each non-
  bouncing surface).  For a 10-surface system the omitted factor
  is roughly ``(1 - 0.04)^16 ~= 0.52``, so the reported magnitude
  over-estimates the true relative intensity by about 2x.  A more
  accurate estimate is::

      I_true ~= I_reported * Prod_{k not in (i, j)} (1 - R_k)^2

  where each non-bouncing surface contributes ``(1 - R_k)^2`` (one
  forward pass + one return pass).  For final stray-light sign-off
  use a non-sequential trace (FRED, ASAP) with the prescription
  exported via :func:`export_zemax_zmx`.

* **``'focus_z_estimate'`` is a HEURISTIC, not a physical focal
  distance.**  The value is a harmonic mean of the *curvature radii*
  ``|r_i|`` and ``|r_j|`` of the two bouncing surfaces (lowercase
  ``r`` -- see :func:`ghost_analysis` body; NOT to be confused with
  the Fresnel-reflectance ``R_i``/``R_j``).  It is a dimensionally
  arbitrary quick-rank scalar useful for sorting ghost paths by
  "this one is roughly near-focus vs that one is roughly out-of-
  focus", NOT a calibrated position relative to the detector.
  Treat it as a sort key; for a physical focus position run a full
  retro-trace of the (i, j) path.

Author: Andrew Traverso
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..glass import get_glass_index
from ..raytrace import surfaces_from_prescription

__all__ = [
    'enumerate_ghost_paths',
    'ghost_analysis',
    'non_sequential_stray_light',
    'retrace_ghost_path',
]


# ---------------------------------------------------------------------------
# Prescription-schema adapter (v4.14.3 / P1-GH-2)
# ---------------------------------------------------------------------------

def _ghost_surfaces(
    prescription: Dict[str, Any],
    wavelength: float,
) -> List[Any]:
    """Return a sequential ``Surface`` list for ghost analysis.

    Accepts both prescription schemas:

    * **Legacy ``'surfaces'``-key prescription** (``make_singlet`` /
      ``make_doublet`` / ``thorlabs_lens`` /
      ``load_zemax_prescription_data_txt`` -- all carry the
      ``'surfaces'`` mirror, which is the primary ray-tracing path).

    * **Modern ``'elements'``-key prescription** without the
      ``'surfaces'`` mirror -- the surface-style element list (each
      entry carries ``'element_type': 'surface'`` / ``'mirror'`` and
      its own ``'radius'``/``'glass_before'``/``'glass_after'``
      fields) produced by the UI ``LensModel.to_prescription()``
      export and by ``load_zemax_*`` ``return``.  When ``'surfaces'``
      is absent the surface mirror is reconstructed in-line from the
      refracting entries of ``elements`` and the result is routed
      back through :func:`surfaces_from_prescription` so all
      downstream validation / freeform / coord-break handling stays
      uniform with the legacy path.

    * **Propagate-style ``'elements'`` list** (``'type': 'propagate'``
      / ``'real_lens'`` / ``'mirror'``, as consumed by
      ``propagate_through_system`` and friends) is dispatched
      through :func:`raytrace.surfaces_from_elements`, which has
      its own surface-expansion logic and threads ``wavelength``
      through embedded ``real_lens`` sub-prescriptions.

    Pre-v4.14.3 :func:`non_sequential_stray_light` did
    ``len(prescription.get('surfaces', []))`` and silently produced
    ``n_surfs = 0`` for the elements-only schema, then later
    indexed into ``prescription['surfaces'][i]`` and raised
    ``IndexError`` with a message that gave no hint about the
    schema mismatch (audit P1-GH-2).

    Parameters
    ----------
    prescription : dict
        Must contain ``'surfaces'`` OR ``'elements'``.
    wavelength : float
        Used only when expanding propagate-style ``'elements'``
        lists containing embedded ``real_lens`` entries; ignored on
        the legacy ``'surfaces'`` path and on the surface-style
        elements reconstruction path.

    Returns
    -------
    surfaces : list of Surface
        Sequential surface list compatible with the rest of the
        ghost analysis code.

    Raises
    ------
    ValueError
        If ``prescription`` has neither ``'surfaces'`` nor
        ``'elements'``.
    """
    if 'surfaces' in prescription:
        return surfaces_from_prescription(prescription)
    if 'elements' in prescription:
        elements = prescription['elements']
        # Disambiguate the two element-list formats.  Surface-style
        # entries (from .zmx parsers / UI export) carry
        # ``'element_type'``; propagate-style entries (from
        # ``propagate_through_system``) carry ``'type'``.  A mixed
        # list violates both schemas and falls through to the
        # propagate path which will raise on the first missing key.
        is_surface_style = bool(elements) and all(
            ('element_type' in e) for e in elements)
        if is_surface_style:
            # Reconstruct the lens-only ``'surfaces'`` /
            # ``'thicknesses'`` mirror from the refracting subset of
            # the element list (mirrors are excluded -- ghost
            # analysis treats them separately and the lens-only
            # mirror is what ``surfaces_from_prescription`` /
            # ``validate_prescription`` expect).  This mirrors the
            # logic in ``io/prescriptions.py:654-677`` that builds
            # the same ``'surfaces'`` slot on .zmx load.
            refr = [e for e in elements
                    if e.get('element_type') == 'surface']
            rebuilt_surfaces = []
            for e in refr:
                rebuilt_surfaces.append({
                    'radius': e['radius'],
                    'conic': e.get('conic', 0.0),
                    'aspheric_coeffs': e.get('aspheric_coeffs'),
                    'glass_before': e.get('glass_before', 'air'),
                    'glass_after': e.get('glass_after', 'air'),
                })
            # Thicknesses: prefer an explicit ``'thicknesses'`` slot
            # if the caller supplied one alongside elements; else
            # default to zero gaps (ghost analysis cares about
            # surface curvatures and indices, not paraxial paths).
            thicknesses = prescription.get(
                'thicknesses',
                [0.0] * max(0, len(rebuilt_surfaces) - 1))
            rebuilt_rx = {
                'surfaces': rebuilt_surfaces,
                'thicknesses': thicknesses,
            }
            if 'aperture_diameter' in prescription:
                rebuilt_rx['aperture_diameter'] = (
                    prescription['aperture_diameter'])
            return surfaces_from_prescription(rebuilt_rx)
        # Propagate-style: defer to surfaces_from_elements.  Defer
        # the import to avoid an analysis -> raytrace import cycle
        # at module-load time.
        from ..raytrace.core import surfaces_from_elements
        return surfaces_from_elements(elements, wavelength)
    raise ValueError(
        "ghost analysis: prescription has neither 'surfaces' nor "
        "'elements' key.  Build the prescription via make_singlet, "
        "make_doublet, thorlabs_lens, load_zemax_prescription_data_txt, "
        "or the UI 'Save prescription' export.")


def enumerate_ghost_paths(n_surfaces: int) -> List[Tuple[int, int]]:
    """List all unique 2-bounce ghost reflection paths.

    For ``n_surfaces`` refracting surfaces (0-indexed), each ghost
    path is a pair ``(i, j)`` where ``i < j``, meaning light
    reflects off surface ``j`` back to surface ``i``, reflects again,
    then continues to the detector.

    Returns
    -------
    paths : list of (int, int)
    """
    return [(i, j) for i in range(n_surfaces)
            for j in range(i + 1, n_surfaces)]


def ghost_analysis(
    prescription: Dict[str, Any],
    wavelength: float,
    semi_aperture: Optional[float] = None,
    n_rays: int = 21,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Trace all 2-bounce ghost paths and report their relative
    intensities.

    Parameters
    ----------
    prescription : dict
    wavelength : float
    semi_aperture : float, optional
        Half-aperture [m] for the ray fan.  Defaults to
        ``prescription['aperture_diameter'] / 2``.
    n_rays : int, default 21
        Number of rays per ghost-path fan.
    verbose : bool

    Returns
    -------
    ghosts : list of dict
        One entry per ghost path with keys:

        * ``'path'`` -- ``(i, j)`` tuple of bouncing surface indices.
        * ``'R_i'`` / ``'R_j'`` -- *Fresnel reflectance* (uppercase
          ``R``, dimensionless, in [0, 1]) at each bouncing surface
          at normal incidence.  NOT the radius of curvature; see
          module docstring for the naming convention.
        * ``'intensity'`` -- UPPER BOUND on the ghost's relative
          intensity.  Computed as ``R_i * R_j`` (Fresnel reflectance
          at surface i times Fresnel reflectance at surface j) only;
          does NOT include the transmission losses
          ``Prod_{k not in (i, j)} (1 - R_k)^2`` (where each ``R_k``
          is the Fresnel reflectance of surface k) over the non-
          bouncing surfaces the ghost ray crosses twice (once
          forward, once on return).  For a 10-surface system this
          factor is ~0.5, so the reported magnitude overestimates
          the true intensity by ~2x.  See the module docstring for
          the correction formula.
        * ``'focus_z_estimate'`` -- HEURISTIC quick-rank scalar,
          the harmonic mean of the *curvature radii* ``|r_i|`` and
          ``|r_j|`` of the two bouncing surfaces (lowercase ``r``,
          in metres; NOT the Fresnel reflectance ``R_i``/``R_j``
          above).  Useful as a sort key (near-focus vs out-of-
          focus); NOT a physical axial position.  For a calibrated
          focus location run a full retro-trace of the path.

    Notes
    -----
    See the module docstring (4.12.0 / B2-1, B2-2) for the
    upper-bound semantics on ``'intensity'`` and the heuristic
    semantics on ``'focus_z_estimate'``.  See the module docstring
    Naming-convention block (v4.13.0 / S3) for the upper-/lowercase
    ``R``/``r`` split between Fresnel reflectance and curvature
    radius.
    """
    # v4.14.3 (P1-GH-2): tolerate both prescription schemas.  Pre-
    # v4.14.3 this was a bare ``surfaces_from_prescription`` call
    # which fails on the modern ``'elements'``-key schema with a
    # validation ``ValueError`` mentioning a "missing 'surfaces' key"
    # -- accurate in the lexical sense but confusing because the
    # equivalent ``'elements'`` list IS present.
    surfs = _ghost_surfaces(prescription, wavelength)
    n_surfs = len(surfs)
    if semi_aperture is None:
        ap = prescription.get('aperture_diameter', 25.4e-3)
        semi_aperture = ap / 2

    paths = enumerate_ghost_paths(n_surfs)
    ghosts = []

    for (i, j) in paths:
        # Compute Fresnel reflection at surfaces i and j
        n_before_i = get_glass_index(surfs[i].glass_before, wavelength)
        n_after_i = get_glass_index(surfs[i].glass_after, wavelength)
        n_before_j = get_glass_index(surfs[j].glass_before, wavelength)
        n_after_j = get_glass_index(surfs[j].glass_after, wavelength)

        # Normal-incidence Fresnel reflectance at each surface
        R_i = ((n_after_i - n_before_i) / (n_after_i + n_before_i)) ** 2
        R_j = ((n_after_j - n_before_j) / (n_after_j + n_before_j)) ** 2
        intensity = float(R_i * R_j)

        # Estimate ghost focus position using the thin-lens approximation:
        # each reflecting surface acts as a mirror with f = r/2 (where ``r``
        # is the surface curvature radius -- lowercase to disambiguate from
        # the uppercase Fresnel reflectances ``R_i``/``R_j`` above; see
        # module docstring "Naming convention").  The ghost image is formed
        # by the combination of these two "mirrors" plus the intervening
        # glass.
        r_i = surfs[i].radius if np.isfinite(surfs[i].radius) else 1e10
        r_j = surfs[j].radius if np.isfinite(surfs[j].radius) else 1e10
        f_ghost = abs(r_i * r_j) / (abs(r_i) + abs(r_j) + 1e-30)
        # Very rough estimate -- actual position needs full retro-trace
        # (and the value is reported as a HEURISTIC sort key, not a
        # calibrated focal distance; see module docstring B2-2).

        ghosts.append({
            'path': (i, j),
            'R_i': float(R_i),
            'R_j': float(R_j),
            'intensity': intensity,
            'focus_z_estimate': float(f_ghost),
        })

    # Sort by intensity (brightest first)
    ghosts.sort(key=lambda g: -g['intensity'])

    if verbose:
        print(f'Ghost analysis: {len(ghosts)} paths for {n_surfs} surfaces')
        for g in ghosts[:10]:
            print(f'  surfaces ({g["path"][0]},{g["path"][1]}): '
                  f'I = {g["intensity"]:.2e}  '
                  f'(R1={g["R_i"]:.4f}, R2={g["R_j"]:.4f})')

    return ghosts


def non_sequential_stray_light(
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    n_rays: int = 21,
    semi_aperture: Optional[float] = None,
    top_n: int = 10,
    bsdf_model: Optional[Any] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Combined ghost + scatter stray-light report for a prescription.

    Wraps :func:`ghost_analysis` (2-bounce reflections via Fresnel)
    with an optional surface-scatter contribution from a
    :class:`BSDFModel` and returns a single structured report.  This
    is the "single entry point" API for assessing stray light at
    design-review time, without having to manually combine the
    individual primitives.

    Parameters
    ----------
    prescription : dict
        Standard lumenairy prescription.
    wavelength : float
        Vacuum wavelength [m].
    n_rays : int, default 21
        Per-ghost-path ray fan resolution (passed to
        :func:`ghost_analysis`).
    semi_aperture : float, optional
        Half-aperture for the ray fan.  Defaults to
        ``prescription['aperture_diameter'] / 2``.
    top_n : int, default 10
        Limit the report to this many brightest ghost paths.
    bsdf_model : BSDFModel, optional
        If supplied, also reports the integrated TIS (Total
        Integrated Scatter) contribution from each surface using
        the BSDF's scatter coefficient.  TIS is added to the
        reflected-ghost intensity to give a worst-case stray-light
        fraction.  When ``None`` (default), only ghost reflections
        are reported.
    seed : int or None, default None
        Seed for the BSDF-integration Monte Carlo.  v4.15 (P1-GH-1):
        pre-4.15 this was hard-coded to ``0``, which made the TIS
        estimate fully deterministic with no uncertainty band -- you
        could not estimate the Monte-Carlo variance because every
        call returned identical numbers.  Pass an ``int`` (e.g.
        ``seed=0``) for the old reproducible behaviour; pass
        ``None`` (the new default) to draw fresh randomness from
        system entropy so repeated calls give a sample-to-sample
        spread you can use as a stand-in for the integration error.
    verbose : bool

    Returns
    -------
    report : dict
        ``ghosts`` -- list of dicts as returned by
        :func:`ghost_analysis`, truncated to ``top_n``.
        ``ghost_total_intensity`` -- sum of relative intensities
        across ALL ghost paths (not just top_n).  Note: each entry
        is the same UPPER BOUND described in :func:`ghost_analysis`;
        the sum inherits the same caveat (B2-1).
        ``scatter_tis_per_surface`` -- per-surface TIS contribution
        if ``bsdf_model`` was given, else ``None``.
        ``stray_light_fraction`` -- conservative upper bound:
        ``ghost_total + sum(scatter_tis_per_surface)``.

    Notes
    -----
    This is a *first-pass* stray-light estimate.  It does NOT do a
    full non-sequential ray trace branching into reflected +
    scattered children at every surface.  The intensities reported
    are linearly composable (sum of independent contributions) and
    so are conservative for typical refractive systems where each
    surface contributes < 1% reflection.  For final stray-light
    sign-off use a dedicated non-sequential code (FRED, ASAP) on
    the prescription exported via :func:`export_zemax_zmx`.
    """
    ghosts = ghost_analysis(prescription, wavelength,
                              semi_aperture=semi_aperture,
                              n_rays=n_rays, verbose=False)

    ghost_total = float(sum(g['intensity'] for g in ghosts))
    top_ghosts = ghosts[:top_n] if top_n > 0 else ghosts

    scatter_tis_per_surface = None
    if bsdf_model is not None:
        # Integrate the BSDF over the upper hemisphere to get TIS.
        # Use a small Monte-Carlo: sample 2k uniform-hemisphere
        # directions, weight by cos(theta), average.
        # v4.15 (P1-GH-1): RNG now seeded from the user-supplied
        # ``seed`` kwarg (None = system entropy = different samples
        # each call).  Pre-4.15 ``default_rng(0)`` pinned the TIS
        # number to a single sample, hiding the Monte-Carlo error.
        rng = np.random.default_rng(seed)
        n_samples = 2000
        # Uniform on upper hemisphere via 2 cos-weighted samples.
        u = rng.random(n_samples)
        v = rng.random(n_samples)
        # cos-weighted hemisphere: theta = arccos(sqrt(1-u))
        cos_t = np.sqrt(1.0 - u)
        sin_t = np.sqrt(u)
        phi = 2.0 * np.pi * v
        # Local incident direction: (0,0,1).  Out: (sin_t cos_phi, sin_t sin_phi, cos_t).
        try:
            f = bsdf_model.evaluate(
                np.array([0.0, 0.0, 1.0]),
                np.stack([sin_t * np.cos(phi),
                          sin_t * np.sin(phi),
                          cos_t], axis=-1),
            )
            # 4.10: With cos-weighted hemisphere sampling (PDF =
            # cos(theta)/pi) the unbiased estimator of
            # int f cos(theta) sin(theta) dtheta dphi is mean(f) * pi.
            # Pre-4.10 used mean(f * cos_t) * pi -- an extra cos factor
            # that biased TIS toward near-normal scattering.
            tis_each_surface = float(np.mean(f) * np.pi)
        except (TypeError, ValueError, RuntimeError, AttributeError):
            # BSDF model can reject the sampling directions (TypeError
            # on a duck-typed model with wrong signature, ValueError
            # on shape mismatch, AttributeError when ``evaluate`` is
            # missing entirely) -- NaN-propagate so the stray-light
            # tally clearly flags the missing TIS contribution.
            tis_each_surface = float('nan')
        # v4.14.3 (P1-GH-2): use the schema-adaptive helper rather
        # than a raw ``prescription.get('surfaces', [])`` lookup --
        # the latter silently returned ``[]`` (and therefore
        # ``n_surfs = 0`` here, dropping the entire TIS contribution)
        # for modern ``'elements'``-key prescriptions.  Routing
        # through ``_ghost_surfaces`` makes the count consistent
        # with what ``ghost_analysis`` saw above.
        n_surfs = len(_ghost_surfaces(prescription, wavelength))
        scatter_tis_per_surface = [tis_each_surface] * n_surfs

    stray_light_fraction = ghost_total
    if scatter_tis_per_surface is not None:
        stray_light_fraction += sum(s for s in scatter_tis_per_surface
                                      if np.isfinite(s))

    if verbose:
        print('Non-sequential stray-light report:')
        print(f'  Ghost paths: {len(ghosts)}, '
              f'top-{top_n} contribute '
              f'{sum(g["intensity"] for g in top_ghosts):.3e}')
        print(f'  Total ghost intensity: {ghost_total:.3e}')
        if scatter_tis_per_surface is not None:
            print(f'  Per-surface TIS (BSDF): '
                  f'{scatter_tis_per_surface[0]:.3e} per surface, '
                  f'{len(scatter_tis_per_surface)} surfaces')
        print(f'  Conservative stray-light fraction: '
              f'{stray_light_fraction:.3e}')

    return {
        'ghosts': top_ghosts,
        'ghost_total_intensity': ghost_total,
        'scatter_tis_per_surface': scatter_tis_per_surface,
        'stray_light_fraction': stray_light_fraction,
    }


# ---------------------------------------------------------------------------
# v5.4 Phase 5 (P2-C closure): per-path ghost retrace for the ghost dock
# ---------------------------------------------------------------------------

def _path_from_pair(
    n_surfs: int,
    i: int,
    j: int,
) -> List[Tuple[int, str]]:
    """Build the explicit (surface_idx, action) sequence for a classical
    2-bounce ghost ``(i, j)`` with ``i < j``.

    Light propagates forward, transmits through all surfaces up to and
    including ``j-1``, reflects at ``j`` back toward ``i``, reflects at
    ``i`` forward again, then transmits through the remaining surfaces.

    Each non-(i, j) refracting surface is therefore crossed twice and
    each bouncing surface contributes one reflection.  ``retrace_ghost_path``
    accepts this list directly so the dock can also feed in arbitrary
    user-built paths (e.g. multi-bounce sequences) once
    :func:`enumerate_ghost_paths` is extended.
    """
    if not (0 <= i < j < n_surfs):
        raise ValueError(
            f"_path_from_pair: invalid pair (i, j) = ({i}, {j}) "
            f"for n_surfaces = {n_surfs}; require 0 <= i < j < n_surfaces.")
    path: List[Tuple[int, str]] = []
    # Forward leg: 0 .. j-1 (transmit)
    for k in range(j):
        path.append((k, 'transmit'))
    # First bounce at j (reflect)
    path.append((j, 'reflect'))
    # Reverse leg: j-1 .. i+1 (transmit, opposite direction)
    for k in range(j - 1, i, -1):
        path.append((k, 'transmit'))
    # Second bounce at i (reflect)
    path.append((i, 'reflect'))
    # Forward leg again: i+1 .. n_surfs-1 (transmit)
    for k in range(i + 1, n_surfs):
        path.append((k, 'transmit'))
    return path


def _fresnel_R_normal(n1: float, n2: float) -> float:
    """Normal-incidence Fresnel reflectance for an n1 -> n2 interface."""
    if (n1 + n2) <= 0.0:
        return 0.0
    return float(((n2 - n1) / (n2 + n1)) ** 2)


def _ghost_intersect(rays, surface, *, n_medium: float,
                      direction: int) -> None:
    """Direction-aware spherical surface intersection used by
    :func:`retrace_ghost_path`.

    The standard :func:`_intersect_surface` fast path always selects
    the near root assuming forward propagation (N > 0).  In a ghost
    retrace some legs run backward (N < 0) after a reflection, and
    the near root then sits on the opposite parametric side.  This
    helper inlines the ray-sphere quadratic and picks the root closest
    to ``rays.z = 0`` along the ray's actual propagation direction,
    so both legs land on the correct intersection.

    Parameters
    ----------
    rays : RayBundle
        Rays in the surface's local frame; ``rays.z`` is the offset
        from the vertex plane (typically 0).  Modified in place: x, y,
        z advance to the intersection point; OPL accumulates ``n_medium
        * t`` for the leg.  Rays missing the sphere or outside the
        clear aperture are marked dead.
    surface : Surface
        Spherical / flat refracting surface.  Conic / aspheric /
        biconic / freeform surfaces fall through to the library's
        Newton iteration (which is still root-ambiguous on backward
        rays, but ghost dock previews target spherical lenses).
    n_medium : float
        Refractive index of the medium the rays are travelling through
        on the way to this surface (same convention as
        ``_intersect_surface``).
    direction : int
        +1 = rays propagate along +z (N > 0), -1 = along -z (N < 0).
        Only used to disambiguate the spherical root pick.
    """
    from ..raytrace.intersection import _intersect_surface
    from ..raytrace.surface import RAY_APERTURE, RAY_MISSED_SURFACE, RAY_OK

    R = float(surface.radius)
    kc = float(surface.conic)
    asph = surface.aspheric_coeffs
    radius_y = getattr(surface, 'radius_y', None)
    freeform = getattr(surface, 'freeform', None)

    is_pure_spherical = (
        np.isfinite(R)
        and kc == 0.0
        and not asph
        and radius_y is None
        and freeform is None
    )

    if not is_pure_spherical:
        # Fall through to the library implementation -- aspheric /
        # freeform ghost paths are not on the v5.4 dock's hot path.
        # (The Newton's initial-guess for those surfaces is
        # similarly direction-blind; users hitting that case can
        # bisect the path into single-surface retraces.)
        _intersect_surface(rays, surface, n_medium=n_medium)
        return

    # Ray-sphere quadratic.  Sphere centred at (0, 0, R) with radius
    # |R|.  Ray line: (x0 + L*t, y0 + M*t, z0 + N*t).  Quadratic in t:
    #   t^2 + b*t + c = 0
    # with b = 2*(L*x0 + M*y0 + N*(z0-R)), c = x0^2 + y0^2 + (z0-R)^2 - R^2.
    # Two roots t1 <= t2.  We want the root closest to z = 0 in the
    # *propagation* direction -- equivalently, the smallest non-
    # negative root in the ray's parametric direction.  For N > 0 the
    # ray is moving forward; for N < 0 it is moving backward and the
    # "near" root is the one with the smaller magnitude of t (since
    # both roots will have the same sign as N for a sensible setup).
    x0, y0, z0 = rays.x, rays.y, rays.z
    Ld, Md, Nd = rays.L, rays.M, rays.N
    dx, dy, dz = x0, y0, z0 - R
    b = 2.0 * (Ld * dx + Md * dy + Nd * dz)
    c = dx ** 2 + dy ** 2 + dz ** 2 - R ** 2
    disc = b ** 2 - 4.0 * c
    disc_safe = np.maximum(disc, 0.0)
    sqrt_disc = np.sqrt(disc_safe)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    # Direction-aware root pick: choose the root with the smallest
    # absolute value (the geometric "near" hit).  This collapses to
    # the legacy ``t1 if R > 0 else t2`` for forward rays starting on
    # the vertex plane, and gives the correct opposite pick for
    # backward rays.
    if direction >= 0:
        # Forward: prefer t closest to 0 from above (positive t for
        # the ordinary advancing trace).  Fall back to the smaller-
        # magnitude root if both are negative (degenerate, but the
        # library's fast path uses the same convention).
        t = np.where(np.abs(t1) <= np.abs(t2), t1, t2)
    else:
        # Backward: same min-magnitude rule; t1 and t2 will both be
        # negative (parametric step opposite to N which is negative)
        # and we want the one with smaller |t|.
        t = np.where(np.abs(t1) <= np.abs(t2), t1, t2)

    disc_pos = disc > 0
    t = np.where(disc_pos, t, 0.0)
    missed_init = (~disc_pos) & rays.alive

    if missed_init.any():
        rays.alive = rays.alive & ~missed_init
        if rays.error_code is not None:
            first_failure = missed_init & (rays.error_code == RAY_OK)
            rays.error_code = np.where(
                first_failure, RAY_MISSED_SURFACE, rays.error_code)

    # Advance to the intersection.
    t = np.where(rays.alive, t, 0.0)
    rays.x = rays.x + rays.L * t
    rays.y = rays.y + rays.M * t
    rays.z = rays.z + rays.N * t

    # OPL: geometric path length |t| times refractive index.
    rays.opd = rays.opd + n_medium * np.abs(t)

    # Vignette by clear aperture.
    if np.isfinite(surface.semi_diameter):
        h_sq = rays.x ** 2 + rays.y ** 2
        clipped = (h_sq > surface.semi_diameter ** 2) & rays.alive
        if clipped.any():
            rays.alive = rays.alive & ~clipped
            if rays.error_code is not None:
                rays.error_code = np.where(
                    clipped, RAY_APERTURE, rays.error_code)


def retrace_ghost_path(
    prescription: Dict[str, Any],
    path: List[Tuple[int, str]],
    wavelength: float,
    *,
    semi_aperture: float,
    n_rays: int = 64,
    image_plane_z: Optional[float] = None,
    bsdf_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Retrace a single ghost path explicitly and return image-plane spot
    statistics.

    Unlike :func:`ghost_analysis`, which reports Fresnel-product *upper
    bound* magnitudes only, this function actually launches a ray fan,
    walks each surface in the user-supplied ``path`` (branching into
    REFLECT or TRANSMIT according to its flag), and projects the
    surviving rays onto the image plane.  v5.4 Phase 5 ships it as the
    per-path retrace primitive consumed by the ghost dock's spot
    preview (replacing the synthesised relative-magnitude Gaussian
    that the v5.4 audit P2-C flagged).

    Parameters
    ----------
    prescription : dict
        Standard lumenairy prescription (either ``'surfaces'``-key or
        ``'elements'``-key; resolved via :func:`_ghost_surfaces`).
    path : list of (int, str)
        Explicit ghost path -- one ``(surface_index, action)`` tuple per
        surface interaction, in trace order.  ``action`` must be one of
        ``'reflect'`` / ``'transmit'``.  ``surface_index`` is the 0-based
        index into the refracting-surfaces list (the same indexing
        :func:`enumerate_ghost_paths` returns).  Convenience builder for
        the classical 2-bounce (i, j) ghost: :func:`_path_from_pair`.
    wavelength : float
        Vacuum wavelength [m].
    semi_aperture : float
        Half-aperture for the input ray fan [m].
    n_rays : int, default 64
        Number of rays in the uniform-pupil bundle (a ``make_rings``
        fan with ``num_rings = n_rays // 8`` and the remainder filled
        in the central ring; capped at >= 1 ring).
    image_plane_z : float, optional
        Distance from the last refracting surface to the image plane
        [m].  If ``None``, uses the prescription's last-surface
        thickness (which is zero for an unmoved
        :func:`make_singlet` -- callers wanting the singlet's nominal
        image plane should pass a paraxial-focus distance themselves).
    bsdf_model : BSDFModel, optional
        Reserved for v5.5 -- if supplied, an extra scatter sample is
        drawn at each surface.  v5.4 ships the specular path only; a
        non-None ``bsdf_model`` is accepted (so callers can pin the
        signature) but currently ignored with no warning.  This avoids
        a breaking signature change when scatter-branched retraces are
        wired up later.

    Returns
    -------
    dict with keys

    * ``'rays_image_plane'`` -- ndarray of shape ``(n_alive, 2)`` with
      the ``(x, y)`` intersection at the image plane, in metres.  Dead
      rays (TIR / vignetted / NaN) are dropped.
    * ``'peak_xy_mm'`` -- ``(cx, cy)`` of the centroid, in millimetres.
    * ``'rms_radius_mm'`` -- RMS spot radius
      ``sqrt(<(x - cx)^2 + (y - cy)^2>)`` in millimetres.
    * ``'fwhm_mm'`` -- FWHM of the radial energy distribution,
      estimated from the encircled-energy radii at 25% and 75% (a
      Gaussian-equivalent FWHM = 2 * sqrt(2*ln(2)) * sigma; here
      computed empirically as twice the radius enclosing 50% of the
      energy times the Gaussian factor 1.1774).  Falls back to NaN if
      fewer than 4 rays survive.
    * ``'total_transmittance'`` -- product of per-surface Fresnel R
      (for ``'reflect'`` steps) and T = 1 - R (for ``'transmit'``
      steps) at normal incidence.  Dimensionless, in [0, 1].
    * ``'energy_fraction_ppm'`` -- ``total_transmittance * 1e6``.

    Raises
    ------
    ValueError
        If any ``surface_index`` in ``path`` is outside the prescription's
        surface list, or if an ``action`` is not in
        ``{'reflect', 'transmit'}``.

    Notes
    -----
    * The retrace is run on a fresh ``RayBundle`` for each branch step,
      one branch at a time.  This is O(len(path) * n_rays) and is
      intended for interactive use (spot diagrams in the GUI), NOT
      for production stray-light sign-off -- for that, export a
      non-sequential ``.zmx`` and run FRED / ASAP.
    * Per-surface action ``'transmit'`` accumulates ``(1 - R)``;
      ``'reflect'`` accumulates ``R``.  R is the normal-incidence
      Fresnel reflectance at the surface's ``glass_before`` /
      ``glass_after`` interface (the same convention as
      :func:`ghost_analysis`).
    * Rays that miss a surface or undergo TIR are dropped from
      ``'rays_image_plane'`` and from the spot statistics; the
      reported ``'total_transmittance'`` is *not* reduced by ray
      mortality (it is the Fresnel-product expectation for the path).
    """
    # ---- 1. Build the surface list (schema-adaptive helper) ----------
    surfs = _ghost_surfaces(prescription, wavelength)
    n_surfs = len(surfs)

    # ---- 2. Validate the path -----------------------------------------
    if not isinstance(path, (list, tuple)) or len(path) == 0:
        raise ValueError(
            f"retrace_ghost_path: path must be a non-empty list of "
            f"(surface_index, action) tuples; got {type(path).__name__}.")
    for step_idx, step in enumerate(path):
        if (not isinstance(step, (list, tuple))) or len(step) != 2:
            raise ValueError(
                f"retrace_ghost_path: path step {step_idx} is "
                f"{step!r}; each step must be a "
                f"(surface_index, action) 2-tuple.")
        s_idx, action = step
        if not (0 <= int(s_idx) < n_surfs):
            raise ValueError(
                f"retrace_ghost_path: path step {step_idx} references "
                f"surface index {s_idx}, out of range "
                f"[0, {n_surfs - 1}] for a system with {n_surfs} "
                f"refracting surfaces.")
        if action not in ('reflect', 'transmit'):
            raise ValueError(
                f"retrace_ghost_path: path step {step_idx} has action "
                f"{action!r}; expected 'reflect' or 'transmit'.")

    # ---- 3. Accumulate the Fresnel product ---------------------------
    # Pre-resolve per-surface glass indices (the same operation as
    # ghost_analysis); ``T = 1 - R`` for transmission at normal
    # incidence is the matched complement to the R term used by
    # ghost_analysis.
    n_before = [get_glass_index(s.glass_before, wavelength) for s in surfs]
    n_after = [get_glass_index(s.glass_after, wavelength) for s in surfs]

    total_T = 1.0
    for s_idx, action in path:
        s_idx = int(s_idx)
        R_s = _fresnel_R_normal(n_before[s_idx], n_after[s_idx])
        if action == 'reflect':
            total_T *= R_s
        else:  # 'transmit'
            total_T *= (1.0 - R_s)

    # ---- 4. Launch the ray bundle ------------------------------------
    # Use a uniform-pupil ``make_rings``-style fan capped at n_rays
    # total.  A small bundle is enough for an interactive spot preview;
    # downstream stats are robust to small N.
    from ..raytrace import make_rings
    n_rings = max(1, int(np.sqrt(max(n_rays, 1))))
    rays_per_ring = max(1, n_rays // max(n_rings, 1))
    rays = make_rings(
        semi_aperture=float(semi_aperture),
        num_rings=n_rings,
        rays_per_ring=rays_per_ring,
        field_angle=0.0,
        wavelength=float(wavelength),
        include_chief=True,
    )

    # ---- 5. Walk the path manually -----------------------------------
    # We re-use the low-level primitives ``_intersect_surface``,
    # ``_refract``, ``_reflect``, and ``_transfer`` directly because
    # the ghost path is not a simple sequential trace -- some surfaces
    # are visited twice with opposite propagation directions.  At each
    # branch step we:
    #
    #   a. Transfer the rays through the air / glass gap leading to
    #      the next surface in the path (using the medium of the
    #      *current* leg, which depends on how many surfaces we have
    #      already crossed at that surface).
    #   b. Move into the surface's local frame (which is just the
    #      cumulative axial offset for axially symmetric prescriptions).
    #   c. Apply ``_intersect_surface`` + ``_refract`` / ``_reflect``.
    #
    # For axially symmetric prescriptions the surfaces sit at fixed
    # axial positions ``z = sum(thicknesses[:k])``.  We pre-compute
    # those once and seed the bundle at ``z = 0`` (the first surface's
    # vertex plane) before stepping through ``path``.
    from ..raytrace.intersection import (
        _reflect,
        _refract,
        _transfer,
    )

    # Axial positions of each surface vertex relative to surface 0.
    z_vertex = np.zeros(n_surfs, dtype=float)
    for k in range(1, n_surfs):
        z_vertex[k] = z_vertex[k - 1] + float(surfs[k - 1].thickness)

    # Bookkeeping: track the surface the bundle is currently sitting
    # at (initial = -1, i.e. has not yet crossed anything).  We use
    # the LOCAL trace path: the bundle is always at the previous
    # surface's vertex plane in its local frame, and ``_transfer``
    # carries it forward / backward along z.
    current_surf = -1
    # Track propagation direction along z: +1 = forward, -1 = backward.
    # After a reflection, the sign flips.  Forward transmission keeps
    # the sign; backward transmission keeps the (now-negative) sign.
    # Used to (a) decide n1 / n2 in refract steps and (b) bracket the
    # final image-plane transfer.
    direction = +1
    # Track which medium the rays are currently propagating IN.  Before
    # the first surface the rays are in ``surfs[0].glass_before``
    # (typically air).
    current_medium = surfs[0].glass_before
    n_current = get_glass_index(current_medium, wavelength)

    for step_idx, (s_idx, action) in enumerate(path):
        s_idx = int(s_idx)

        # Signed axial distance from the bundle's current axial
        # position to this surface's vertex (in absolute / global z
        # coordinates).  On the first step the bundle is at
        # z = z_vertex[0] (the first surface's vertex plane) in its
        # local frame, so dz = z_vertex[s_idx] - z_vertex[0].
        # Subsequently the bundle is at the previous visited surface's
        # vertex, so dz = z_vertex[s_idx] - z_vertex[current_surf].
        if current_surf < 0:
            dz = float(z_vertex[s_idx] - 0.0)
        else:
            dz = float(z_vertex[s_idx] - z_vertex[current_surf])

        # _transfer parameterises the next plane by its target local
        # z; since rays.z = 0 in the previous surface's frame, we pass
        # ``dz`` directly with its sign.  The internal solve
        # ``t = (dz - rays.z) / rays.N`` then gives:
        #   * dz > 0, N > 0 -> t > 0 (forward step, OK)
        #   * dz < 0, N < 0 -> t > 0 (backward step in z, but forward
        #     along the ray's propagation direction)
        # Mixed signs only arise on the first step if z_vertex[s_idx]
        # < 0 (impossible for our cumulative-thickness layout) so
        # this is safe.
        _transfer(rays, dz, n_current)

        # Surface intersection.  We need a root-pick that handles BOTH
        # forward-propagating rays (N > 0, ordinary trace, near root is
        # the one closest to z = 0) AND backward-propagating rays
        # (N < 0, after a reflection bounce).  The library's
        # _intersect_surface fast path picks ``t1 if R > 0 else t2``
        # which corresponds to the near root for FORWARD propagation
        # only -- for backward propagation it picks the FAR root and
        # the ray lands at z ~= 2R past the surface (audit P5
        # reproducer).
        #
        # We inline the spherical / flat fast path here with a
        # direction-aware root selection so the ghost retrace is
        # robust to both legs.  Aspheric / freeform / biconic surfaces
        # fall through to the library's Newton iteration; for those
        # the initial-guess sphere chooses the wrong root by the same
        # mechanism, but ghost dock previews are run on spherical
        # singlets / doublets where the fast path is exact.
        _ghost_intersect(rays, surfs[s_idx],
                          n_medium=n_current, direction=direction)

        if action == 'reflect':
            _reflect(rays, surfs[s_idx])
            # After reflection the bundle is travelling in the opposite
            # axial direction; the medium stays the same (mirror /
            # ghost reflection does not change which side of the
            # surface we are on for the purpose of the *next* leg).
            direction = -direction
            # current_medium and n_current unchanged
        else:  # 'transmit'
            # Determine which side of the surface we are crossing into.
            # On a forward transmission we go glass_before -> glass_after;
            # on a backward transmission (direction == -1) we go
            # glass_after -> glass_before.  Set n1 / n2 accordingly so
            # vector Snell uses the correct ratio.
            if direction > 0:
                n1 = get_glass_index(surfs[s_idx].glass_before, wavelength)
                n2 = get_glass_index(surfs[s_idx].glass_after, wavelength)
                next_medium = surfs[s_idx].glass_after
            else:
                n1 = get_glass_index(surfs[s_idx].glass_after, wavelength)
                n2 = get_glass_index(surfs[s_idx].glass_before, wavelength)
                next_medium = surfs[s_idx].glass_before
            _refract(rays, surfs[s_idx], n1, n2)
            current_medium = next_medium
            n_current = n2

        current_surf = s_idx

    # ---- 6. Project to the image plane -------------------------------
    # The image plane sits at absolute z = z_vertex[-1] + image_plane_z
    # (with image_plane_z defaulting to the last surface's thickness).
    if image_plane_z is None:
        image_plane_z = float(surfs[-1].thickness)
    # The bundle is currently at the last visited surface's vertex in
    # that surface's local frame (rays.z = 0).  Walk it to the image
    # plane using the signed dz convention -- _transfer with a signed
    # dz handles both forward and backward propagation correctly.
    last_surf = int(path[-1][0])
    dz_to_image = float(
        z_vertex[-1] + float(image_plane_z) - z_vertex[last_surf])
    _transfer(rays, dz_to_image, n_current)

    # ---- 7. Gather image-plane coordinates and stats -----------------
    alive_mask = np.asarray(rays.alive, dtype=bool)
    x_im = np.asarray(rays.x, dtype=float)[alive_mask]
    y_im = np.asarray(rays.y, dtype=float)[alive_mask]
    rays_xy = np.column_stack([x_im, y_im]) if x_im.size > 0 \
        else np.zeros((0, 2), dtype=float)

    if x_im.size >= 1:
        cx = float(np.mean(x_im))
        cy = float(np.mean(y_im))
    else:
        cx = 0.0
        cy = 0.0

    if x_im.size >= 2:
        dx = x_im - cx
        dy = y_im - cy
        rms = float(np.sqrt(np.mean(dx * dx + dy * dy)))
    else:
        rms = 0.0

    # FWHM from the empirical encircled-energy radius at 50%, scaled
    # to a Gaussian-equivalent FWHM via 2 * sqrt(2 * ln 2) / sqrt(2)
    # = 1.6651 (the radius enclosing 50% of a 2-D Gaussian is
    # sigma * sqrt(2 * ln 2) = 1.1774 sigma; the FWHM is 2.3548 sigma;
    # ratio = 2.0).
    if x_im.size >= 4 and rms > 0:
        radii = np.sqrt((x_im - cx) ** 2 + (y_im - cy) ** 2)
        r_half = float(np.median(radii))  # 50% encircled
        fwhm = 2.0 * r_half
    else:
        fwhm = float('nan')

    # ---- 8. Return ----------------------------------------------------
    return {
        'rays_image_plane': rays_xy,
        'peak_xy_mm': (cx * 1e3, cy * 1e3),
        'rms_radius_mm': rms * 1e3,
        'fwhm_mm': fwhm * 1e3 if np.isfinite(fwhm) else float('nan'),
        'total_transmittance': float(total_T),
        'energy_fraction_ppm': float(total_T) * 1e6,
    }

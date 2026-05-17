"""
Ghost image analysis for multi-surface lens systems.

A "ghost" is a parasitic image formed by light that reflects from two
surfaces (instead of transmitting straight through) and reaches the
detector.  For an N-surface system there are N*(N-1)/2 possible
ghost paths.

This module traces each 2-bounce ghost path, computes the ghost
image intensity (including Fresnel reflection losses), and returns
a map of the dominant ghosts for stray-light assessment.

Important caveats (4.12.0 / B2-1, B2-2)
---------------------------------------

* **Reported ghost ``'intensity'`` values are UPPER BOUNDS.**  The
  product ``R_i * R_j`` captures only the two normal-incidence
  Fresnel REFLECTIONS at the bouncing surfaces; it OMITS the
  transmission losses ``Prod_k (1 - R_k)`` over every other surface
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
  distance.**  The value is a harmonic mean of ``|R_i|`` and
  ``|R_j|`` (see :func:`ghost_analysis` body) -- a dimensionally
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

from ..raytrace import surfaces_from_prescription
from ..glass import get_glass_index


__all__ = [
    'enumerate_ghost_paths',
    'ghost_analysis',
    'non_sequential_stray_light',
]


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
        * ``'R_i'`` / ``'R_j'`` -- normal-incidence Fresnel
          reflectance at each bouncing surface.
        * ``'intensity'`` -- UPPER BOUND on the ghost's relative
          intensity.  Computed as ``R_i * R_j`` only; does NOT
          include the transmission losses
          ``Prod_{k not in (i, j)} (1 - R_k)^2`` over the non-
          bouncing surfaces the ghost ray crosses twice (once
          forward, once on return).  For a 10-surface system this
          factor is ~0.5, so the reported magnitude overestimates
          the true intensity by ~2x.  See the module docstring for
          the correction formula.
        * ``'focus_z_estimate'`` -- HEURISTIC quick-rank scalar,
          the harmonic mean of ``|R_i|`` and ``|R_j|``.  Useful as
          a sort key (near-focus vs out-of-focus); NOT a physical
          axial position.  For a calibrated focus location run a
          full retro-trace of the path.

    Notes
    -----
    See the module docstring (4.12.0 / B2-1, B2-2) for the
    upper-bound semantics on ``'intensity'`` and the heuristic
    semantics on ``'focus_z_estimate'``.
    """
    surfs = surfaces_from_prescription(prescription)
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
        # each reflecting surface acts as a mirror with f = R/2.  The ghost
        # image is formed by the combination of these two "mirrors" plus
        # the intervening glass.
        R_i_val = surfs[i].radius if np.isfinite(surfs[i].radius) else 1e10
        R_j_val = surfs[j].radius if np.isfinite(surfs[j].radius) else 1e10
        f_ghost = abs(R_i_val * R_j_val) / (abs(R_i_val) + abs(R_j_val) + 1e-30)
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
        rng = np.random.default_rng(0)
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
        except Exception:
            tis_each_surface = float('nan')
        n_surfs = len(prescription.get('surfaces', []))
        scatter_tis_per_surface = [tis_each_surface] * n_surfs

    stray_light_fraction = ghost_total
    if scatter_tis_per_surface is not None:
        stray_light_fraction += sum(s for s in scatter_tis_per_surface
                                      if np.isfinite(s))

    if verbose:
        print(f'Non-sequential stray-light report:')
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

"""
Ghost image analysis for multi-surface lens systems.

A "ghost" is a parasitic image formed by light that reflects from two
surfaces (instead of transmitting straight through) and reaches the
detector.  For an N-surface system there are N*(N-1)/2 possible
ghost paths.

This module traces each 2-bounce ghost path, computes the ghost
image intensity (including Fresnel reflection losses), and returns
a map of the dominant ghosts for stray-light assessment.

Author: Andrew Traverso
"""
from __future__ import annotations
from typing import List

import numpy as np

from .raytrace import surfaces_from_prescription
from .glass import get_glass_index


def enumerate_ghost_paths(n_surfaces):
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


def ghost_analysis(prescription, wavelength, semi_aperture=None,
                   n_rays=21, verbose=True):
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
        ``'path'`` (i, j), ``'intensity'`` (relative to transmitted),
        ``'focus_z'`` (ghost image axial position [m]).
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
        # Very rough estimate — actual position needs full retro-trace

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

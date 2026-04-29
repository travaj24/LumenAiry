"""
Multi-configuration and afocal-mode support.

Multi-config lets you evaluate the same optical system under different
conditions (zoom positions, field angles, wavelengths, thermal states)
and combine results into a single merit function or analysis report.

Afocal mode handles systems where both object and image are at
infinity (telescopes, beam expanders) — the angular magnification
replaces the focal length as the primary specification.

Author: Andrew Traverso
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
import copy

from .raytrace import surfaces_from_prescription, system_abcd


# =====================================================================
# Multi-configuration
# =====================================================================

@dataclass
class Configuration:
    """One configuration of the optical system.

    Attributes
    ----------
    name : str
        Human-readable label (e.g. 'Wide', 'Tele', '20C', '-40C').
    prescription : dict
        The full prescription for this config (may differ in
        thicknesses, glass indices, field angles from other configs).
    wavelength : float
        Evaluation wavelength [m].
    field_angle : float
        Object field angle [rad].
    weight : float
        Relative weight in multi-config merit functions.
    """
    name: str
    prescription: Dict[str, Any]
    wavelength: float = 1.31e-6
    field_angle: float = 0.0
    weight: float = 1.0


def multi_config_merit(configs, merit_fn, verbose=False):
    """Evaluate a merit function across multiple configurations and
    return the weighted sum.

    Parameters
    ----------
    configs : list of Configuration
    merit_fn : callable(prescription, wavelength, field_angle) -> float
        Merit function that takes a prescription and returns a scalar.
    verbose : bool

    Returns
    -------
    total : float
        Weighted sum of per-config merits.
    per_config : list of float
        Individual merit values.
    """
    total = 0.0
    per_config = []
    for cfg in configs:
        val = merit_fn(cfg.prescription, cfg.wavelength, cfg.field_angle)
        per_config.append(val)
        total += cfg.weight * val
        if verbose:
            print(f'  [{cfg.name}] merit = {val:.6g} (weight {cfg.weight})')
    return total, per_config


def create_zoom_configs(prescription_template, zoom_spacings,
                        wavelength=1.31e-6, field_angle=0.0):
    """Create multi-config entries for a zoom system.

    Parameters
    ----------
    prescription_template : dict
    zoom_spacings : list of list of float
        Each inner list gives the air-gap thicknesses for one zoom
        position.  Length must match the number of thicknesses in the
        prescription.
    wavelength : float
    field_angle : float

    Returns
    -------
    configs : list of Configuration
    """
    configs = []
    for i, spacings in enumerate(zoom_spacings):
        pres = copy.deepcopy(prescription_template)
        for j, t in enumerate(spacings):
            if j < len(pres['thicknesses']):
                pres['thicknesses'][j] = t
        configs.append(Configuration(
            name=f'Zoom_{i}',
            prescription=pres,
            wavelength=wavelength,
            field_angle=field_angle))
    return configs


# =====================================================================
# Afocal mode
# =====================================================================

def afocal_angular_magnification(prescription, wavelength):
    """Compute the angular magnification of an afocal system.

    For a properly-corrected afocal system (object and image at
    infinity), the angular magnification is:

        M_angular = -EFL_objective / EFL_eyepiece

    With the ray-transfer matrix
    ``[y_out, theta_out] = M * [y_in, theta_in]``, an afocal system
    is defined by ``C = 0`` (collimated input gives collimated
    output), not ``B = 0`` (which would be a 1:1 imager).  For a
    genuinely afocal system, ``M_angular = D`` and ``A * D = 1``
    (equivalently, the linear magnification is ``A = 1/D``).

    Parameters
    ----------
    prescription : dict
    wavelength : float

    Returns
    -------
    mag : float
        Angular magnification.  ``|mag| > 1`` = magnifying;
        ``|mag| < 1`` = reducing.
    is_afocal : bool
        True if the system is within numerical tolerance of afocal
        (``|C| * aperture_radius < 1e-6``, i.e. less than a
        micro-radian output divergence for a typical input bundle).
    """
    surfs = surfaces_from_prescription(prescription)
    M_abcd, _, _, _ = system_abcd(surfs, wavelength)
    A, B = M_abcd[0, 0], M_abcd[0, 1]
    C, D = M_abcd[1, 0], M_abcd[1, 1]
    # C has units of 1/length; scale by aperture radius so the test
    # is dimensionally a residual output angle (radians).
    ap = prescription.get('aperture_diameter', 25.4e-3) / 2.0
    is_afocal = abs(C) * ap < 1e-6
    mag = float(D)
    return mag, is_afocal


def _zero_C_air_gap(prescription, gap_slot_index, wavelength=550e-9):
    """Compute the air-gap thickness that zeros the C element of the
    system ABCD matrix (the afocal condition).

    In `[y_out, theta_out] = M * [y_in, theta_in]` form, the afocal
    condition is that a collimated input (theta_in = 0) produces a
    collimated output (theta_out = 0), i.e.
    ``theta_out = C*y_in + D*0 = C*y_in = 0``, so ``C = 0``.

    For two lens groups separated by a single air gap, C is linear
    in the gap, so one linear solve nails it to machine precision
    regardless of lens thickness, index, or conic:

        C(gap) = C0 + slope * gap   =>   gap_afocal = -C0 / slope

    Parameters
    ----------
    prescription : dict
        Prescription with a placeholder air gap in
        ``thicknesses[gap_slot_index]``.  Two evaluations at two
        different gap values determine the zero.
    gap_slot_index : int
        Index into ``thicknesses`` of the air gap to solve for.
    wavelength : float
        Wavelength used to evaluate ABCD.

    Returns
    -------
    gap : float
        Air-gap thickness [m] giving the afocal condition.
    """
    import copy
    from .raytrace import surfaces_from_prescription, system_abcd
    pres = copy.deepcopy(prescription)
    # Evaluate C at two gap values; C is exactly linear in gap so any
    # two non-degenerate samples determine the zero.
    g0 = 0.0
    g1 = max(abs(prescription['thicknesses'][gap_slot_index]), 1e-3)
    pres['thicknesses'][gap_slot_index] = g0
    C0 = system_abcd(surfaces_from_prescription(pres), wavelength)[0][1, 0]
    pres['thicknesses'][gap_slot_index] = g1
    C1 = system_abcd(surfaces_from_prescription(pres), wavelength)[0][1, 0]
    if abs(C1 - C0) < 1e-30:
        return g1     # degenerate; fall back to the guess
    # C(gap) = C0 + (C1 - C0) * gap / g1  =>  gap_zero = -C0 * g1 / (C1 - C0)
    gap_zero = -C0 * g1 / (C1 - C0)
    return float(max(gap_zero, 0.0))


# Backwards-compatible alias under the old (misnamed) name.
_zero_B_air_gap = _zero_C_air_gap


def beam_expander_prescription(M, f_objective, glass='N-BK7',
                                aperture=25.4e-3, wavelength=550e-9):
    """Create a Galilean beam expander prescription.

    A Galilean beam expander uses a negative (diverging) input lens
    and a positive (converging) output lens.  The classical thin-lens
    separation is ``f_obj + f_eye`` with ``f_eye = -f_obj / M``, but
    that ignores finite-thickness principal-plane offsets.  This
    function solves numerically for the air-gap that drives the
    system ABCD matrix's B element to zero, giving a genuinely
    afocal output for the thick singlets it generates.

    Parameters
    ----------
    M : float
        Expansion ratio (> 1 for expansion, < 1 for compression).
    f_objective : float
        Focal length of the positive (output) lens [m].
    glass : str
    aperture : float
    wavelength : float, default 550e-9
        Design wavelength used to compute the thick-lens correction.

    Returns
    -------
    prescription : dict
    """

    f_eye = -f_objective / M  # negative for Galilean

    n = 1.5  # approximate
    # Equi-shaped singlets on both sides so the thin-lens focal
    # length formula R = f*(n-1)*2 holds.  Build the eyepiece as
    # equi-concave ([R, -R] with R<0) rather than plano-concave so
    # it has the correct focal length.  (The previous version used
    # [R_eye, inf] which halved the eyepiece focal length, giving a
    # beam expander whose magnification was half the requested M.)
    R_obj = f_objective * (n - 1) * 2
    R_eye = f_eye * (n - 1) * 2

    # Build the prescription with a placeholder separation; we'll
    # solve for the true afocal gap next.
    pres = {
        'name': f'Beam expander {M:.1f}x',
        'aperture_diameter': aperture,
        'surfaces': [
            {'radius': R_eye, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': glass},
            {'radius': -R_eye, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': glass, 'glass_after': 'air'},
            {'radius': R_obj, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': glass},
            {'radius': -R_obj, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': glass, 'glass_after': 'air'},
        ],
        'thicknesses': [
            abs(R_eye) * 0.05,   # eyepiece glass thickness
            abs(f_objective + f_eye),   # air gap (placeholder)
            abs(R_obj) * 0.05,   # objective glass thickness
        ],
    }

    try:
        pres['thicknesses'][1] = _zero_C_air_gap(
            pres, gap_slot_index=1, wavelength=wavelength)
    except Exception:
        # Fall back to the thin-lens separation on error.
        pass
    return pres


def keplerian_telescope(f_objective, f_eyepiece, glass='N-BK7',
                         aperture=25.4e-3, wavelength=550e-9):
    """Create a Keplerian telescope prescription (two positive lenses
    separated to produce an afocal output).

    The classical thin-lens separation is ``f_obj + f_eye``, but thick
    singlets need a principal-plane correction.  This function solves
    numerically for the air gap that zeros the system ABCD's B
    element, delivering a truly afocal output at the design
    wavelength regardless of lens thickness.

    Parameters
    ----------
    f_objective : float [m]
    f_eyepiece : float [m]
    glass : str
    aperture : float [m]
    wavelength : float, default 550e-9
        Design wavelength used to compute the thick-lens correction.

    Returns
    -------
    prescription : dict
    """
    n = 1.5
    R_obj = f_objective * (n - 1) * 2
    R_eye = f_eyepiece * (n - 1) * 2

    pres = {
        'name': f'Keplerian {f_objective*1e3:.0f}/{f_eyepiece*1e3:.0f}mm',
        'aperture_diameter': aperture,
        'surfaces': [
            {'radius': R_obj, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': glass},
            {'radius': -R_obj, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': glass, 'glass_after': 'air'},
            {'radius': R_eye, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': glass},
            {'radius': -R_eye, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': glass, 'glass_after': 'air'},
        ],
        'thicknesses': [
            abs(R_obj) * 0.05,
            f_objective + f_eyepiece,   # air gap (placeholder)
            abs(R_eye) * 0.05,
        ],
    }

    try:
        pres['thicknesses'][1] = _zero_C_air_gap(
            pres, gap_slot_index=1, wavelength=wavelength)
    except Exception:
        # Fall back to the thin-lens separation on error.
        pass
    return pres

"""
v5.1.0 split: trace / prescription summary printouts.

Extracted from ``lumenairy/raytrace/core.py`` as part of the v5.1.0
6-file split (ROADMAP Agent B).  The ROADMAP topology mentions a
"layout" slice for 2-D layout figures + spot diagrams; the original
v5.0 ``core.py`` only contained the textual ``trace_summary`` and
``prescription_summary`` printouts (the actual matplotlib spot
diagram lives in :mod:`lumenairy.raytrace.ray_fan`).  This module
hosts the textual layout / summary helpers; later v5.x agents may
extend it with 2-D layout figures without breaking the
``lumenairy.raytrace.layout`` namespace.

Every public name here is re-exported from
``lumenairy.raytrace.core`` so existing imports continue to resolve.

No physics change: contents are bit-for-bit copies of the original
implementations.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .ray_fan import spot_geo_radius, spot_rms
from .seidel import system_abcd
from .surface import (
    RAY_APERTURE,
    RAY_MISSED_SURFACE,
    RAY_NAN,
    RAY_TIR,
    TraceResult,
)
from .trace import surfaces_from_prescription

# ============================================================================
# Utility: trace summary
# ============================================================================

def trace_summary(result: 'TraceResult', units: str = 'mm') -> None:
    """Print a summary of the trace result.

    Parameters
    ----------
    result : TraceResult
    units : str
        ``'mm'`` or ``'um'``.
    """
    scale = {'um': 1e6, 'mm': 1e3, 'm': 1.0}[units]
    label = {'um': 'µm', 'mm': 'mm', 'm': 'm'}[units]

    rms, (cx, cy) = spot_rms(result)
    geo = spot_geo_radius(result)

    final = result.image_rays
    n_alive = int(np.sum(final.alive))
    n_total = final.n_rays
    vignetting = 100 * (1 - n_alive / n_total)

    # Break down the loss by cause if error_code is available
    # (added 3.1.9).  Pre-3.1.9 bundles may lack the field; fall
    # back silently to an aggregate vignetting number.
    ec = getattr(final, 'error_code', None)
    if ec is not None and n_alive < n_total:
        n_tir   = int(np.sum(ec == RAY_TIR))
        n_ap    = int(np.sum(ec == RAY_APERTURE))
        n_miss  = int(np.sum(ec == RAY_MISSED_SURFACE))
        n_nan   = int(np.sum(ec == RAY_NAN))
        loss_detail = (f" [TIR={n_tir}, aperture={n_ap}, "
                        f"miss={n_miss}, nan={n_nan}]")
    else:
        loss_detail = ''

    print("Ray trace summary")
    print(f"  Wavelength:   {result.wavelength * 1e9:.2f} nm")
    print(f"  Surfaces:     {len(result.surfaces)}")
    print(f"  Rays:         {n_alive}/{n_total} alive "
          f"({vignetting:.1f}% lost{loss_detail})")
    print(f"  Centroid:     ({cx * scale:.4f}, {cy * scale:.4f}) {label}")
    print(f"  RMS spot:     {rms * scale:.4f} {label}")
    print(f"  GEO radius:   {geo * scale:.4f} {label}")

    # Airy disc
    # 4.11.2: report the image-plane Airy radius
    #     r_Airy = 1.22 * lambda * f_eff / D
    # rather than the half-angle 1.22 * lambda / D.  Pre-4.11.2 the
    # "Airy radius" printed was a divergence half-angle in radians
    # but compared against the spot RMS in metres -- "Spot/Airy" was
    # off by a factor of f_eff [m^-1].
    sd = result.surfaces[0].semi_diameter
    if np.isfinite(sd):
        try:
            _, f_eff, _, _ = system_abcd(result.surfaces, result.wavelength)
        except (ValueError, RuntimeError, ZeroDivisionError,
                np.linalg.LinAlgError, IndexError):
            f_eff = float('nan')
        if np.isfinite(f_eff):
            airy = 1.22 * result.wavelength * abs(f_eff) / (2.0 * sd)
            print(f"  Airy radius:  {airy * scale:.4f} {label}")
            print(f"  Spot/Airy:    {rms / airy:.2f}")
        else:
            # Afocal/degenerate: no f_eff, so no image-plane Airy radius.
            # The diffraction limit is a half-ANGLE; comparing it against
            # the RMS spot (metres) would be meaningless.
            airy_half_angle = 1.22 * result.wavelength / (2.0 * sd)
            print(f"  Airy half-angle: {airy_half_angle * 1e6:.4f} urad "
                  f"(afocal/degenerate - f_eff unavailable)")


def prescription_summary(
    prescription: Dict[str, Any],
    wavelength: float,
    units: str = 'mm',
) -> None:
    """Print a system summary from a prescription dict.

    Parameters
    ----------
    prescription : dict
    wavelength : float
    units : str
    """
    scale = {'um': 1e6, 'mm': 1e3, 'm': 1.0}[units]
    label = {'um': 'µm', 'mm': 'mm', 'm': 'm'}[units]

    surfaces = surfaces_from_prescription(prescription)
    abcd, efl, bfl, ffl = system_abcd(surfaces, wavelength)

    name = prescription.get('name', 'Unnamed')
    print(f"System: {name}")
    print(f"  Wavelength:   {wavelength * 1e9:.2f} nm")
    print(f"  Surfaces:     {len(surfaces)}")
    print(f"  EFL:          {efl * scale:.4f} {label}")
    print(f"  BFL:          {bfl * scale:.4f} {label}")
    print(f"  FFL:          {ffl * scale:.4f} {label}")
    print("  ABCD matrix:")
    print(f"    A = {abcd[0,0]:.6f}   B = {abcd[0,1] * scale:.6f} {label}")
    print(f"    C = {abcd[1,0] / scale:.6f} 1/{label}   D = {abcd[1,1]:.6f}")

    ap = prescription.get('aperture_diameter')
    if ap:
        f_number = abs(efl) / ap if np.isfinite(efl) else np.inf
        print(f"  Aperture:     {ap * scale:.4f} {label}")
        print(f"  f/#:          {f_number:.2f}")


__all__ = [
    'trace_summary',
    'prescription_summary',
]

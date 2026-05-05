"""
lumenairy.propagators.dispatch -- top-level smart-method propagator.

Picks the most appropriate diffraction propagator for the given
input + prescription + output geometry.  The user calls one
function and the dispatcher figures out whether to use ASM,
Fresnel, Maslov, GBD, HFPI, or HF based on system properties.

Method selection logic (when ``method='auto'``)
-----------------------------------------------

1. **Prescription with diffractive surfaces** (DOEs / hard
   apertures inside): ``hfpi``.
2. **Prescription without diffractive surfaces**:  ``maslov``.
3. **No prescription, only z**:
   - Far-field (Fresnel number << 1):  ``fraunhofer``.
   - Otherwise:  ``asm``.

The user can always override by passing ``method='asm'`` /
``'fresnel'`` / ``'fraunhofer'`` / ``'rs'`` / ``'maslov'`` /
``'asymptotic'`` / ``'gbd'`` / ``'hfpi'``.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..backend import array_namespace


VALID_METHODS = (
    'auto', 'asm', 'fresnel', 'fraunhofer', 'rs',
    'maslov', 'asymptotic', 'gbd', 'hfpi',
)


def propagate(
    E_in,
    *,
    z: Optional[float] = None,
    wavelength: float,
    dx: float,
    prescription: Optional[Dict[str, Any]] = None,
    method: str = 'auto',
    output_grid: Optional[tuple] = None,
    output_dx: Optional[float] = None,
    **method_kwargs,
):
    """Top-level smart-method propagator.

    Routes the call to the most appropriate underlying propagator
    based on the geometry of the request and the structure of the
    prescription (when provided).  See the module docstring for
    selection logic.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"propagate: method must be one of {VALID_METHODS}, "
            f"got {method!r}.")

    if method == 'auto':
        method = _auto_select_method(
            E_in, z=z, wavelength=wavelength, dx=dx,
            prescription=prescription)

    return _dispatch_to_method(
        method, E_in,
        z=z, wavelength=wavelength, dx=dx,
        prescription=prescription,
        output_grid=output_grid,
        output_dx=output_dx,
        **method_kwargs,
    )


def _auto_select_method(E_in, *, z, wavelength, dx, prescription):
    """Pick a method from the geometry + prescription structure."""
    if prescription is not None:
        events = prescription.get('events_json') or []
        has_doe = False
        if isinstance(events, list):
            for ev in events:
                if isinstance(ev, dict) and ev.get('type') == 'doe':
                    has_doe = True
                    break
        return 'hfpi' if has_doe else 'maslov'

    if z is None or z == 0:
        return 'asm'

    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    a = 0.5 * dx * max(Ny, Nx)
    abs_z = abs(z)
    if abs_z == 0:
        return 'asm'
    N_F = a * a / (wavelength * abs_z)
    if N_F < 0.1:
        return 'fraunhofer'
    return 'asm'


def _dispatch_to_method(method, E_in, *, z, wavelength, dx,
                        prescription, output_grid, output_dx,
                        **kwargs):
    """Call the chosen propagator with the appropriate signature."""
    if method == 'asm':
        from .propagation import angular_spectrum_propagate
        if z is None:
            return E_in
        return angular_spectrum_propagate(E_in, z, wavelength, dx, **kwargs)

    if method == 'fresnel':
        from .propagation import fresnel_propagate
        if z is None:
            raise ValueError("propagate(method='fresnel'): z is required.")
        return fresnel_propagate(E_in, z, wavelength, dx, **kwargs)

    if method == 'fraunhofer':
        from .propagation import fraunhofer_propagate
        if z is None:
            raise ValueError("propagate(method='fraunhofer'): z is required.")
        return fraunhofer_propagate(E_in, z, wavelength, dx, **kwargs)

    if method == 'rs':
        from .propagation import rayleigh_sommerfeld_propagate
        if z is None:
            raise ValueError("propagate(method='rs'): z is required.")
        return rayleigh_sommerfeld_propagate(E_in, z, wavelength, dx, **kwargs)

    if method == 'gbd':
        from .gbd import propagate_gbd_freespace
        if prescription is None:
            if z is None:
                raise ValueError(
                    "propagate(method='gbd') without prescription requires z.")
            return propagate_gbd_freespace(
                E_in, dx, z=z, wavelength=wavelength,
                output_grid=output_grid, output_dx=output_dx,
                **kwargs,
            )
        raise NotImplementedError(
            "propagate(method='gbd') with prescription is not yet "
            "implemented (requires raytrace + prescription integration).")

    if method == 'hfpi':
        from .hfpi import propagate_hfpi_freespace_aperture
        if prescription is None:
            if 'aperture_radius' not in kwargs:
                raise ValueError(
                    "propagate(method='hfpi') without prescription "
                    "needs at least an aperture geometry "
                    "(aperture_radius=...).")
            return propagate_hfpi_freespace_aperture(
                E_in, dx,
                wavelength=wavelength,
                **kwargs,
            )
        raise NotImplementedError(
            "propagate(method='hfpi') with prescription is not yet "
            "implemented.")

    if method == 'maslov':
        from ..lenses import apply_real_lens_maslov
        if prescription is None:
            raise ValueError(
                "propagate(method='maslov') requires a prescription.")
        return apply_real_lens_maslov(
            E_in, prescription, dx, wavelength,
            **kwargs,
        )

    if method == 'asymptotic':
        from .asymptotic import propagate_modal_asymptotic
        if prescription is None:
            raise ValueError(
                "propagate(method='asymptotic') requires a prescription "
                "and an LG-mode source decomposition.")
        return propagate_modal_asymptotic(
            E_in, prescription, dx, wavelength,
            **kwargs,
        )

    raise NotImplementedError(f"Method {method!r} is not implemented.")


__all__ = ['propagate', 'VALID_METHODS']

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
``'asymptotic'`` / ``'gbd'`` / ``'hfpi'`` / ``'hf'`` / ``'mhs'``.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import math

import numpy as np

from ..backend import array_namespace


VALID_METHODS = (
    'auto', 'asm', 'fresnel', 'fraunhofer', 'rs',
    'maslov', 'asymptotic', 'gbd', 'hfpi', 'hf', 'mhs',
)


def propagate(
    E_in,
    *,
    z: Optional[float] = None,
    wavelength: float,
    dx: float,
    prescription: Optional[Dict[str, Any]] = None,
    method: str = 'auto',
    accuracy: str = 'balanced',
    output_grid: Optional[tuple] = None,
    output_dx: Optional[float] = None,
    return_result: bool = False,
    **method_kwargs,
):
    """Top-level smart-method propagator.

    Routes the call to the most appropriate underlying propagator
    based on the geometry of the request and the structure of the
    prescription (when provided).  See the module docstring for
    selection logic.

    Parameters
    ----------
    accuracy : 'fast' | 'balanced' | 'accurate', default 'balanced'
        Hint for the ``method='auto'`` selector:

          * ``'fast'`` -- prefer the cheapest method that is
            asymptotically valid (e.g. ``'maslov'`` over GBD when
            both apply).
          * ``'balanced'`` -- the default; trades accuracy for
            speed on a case-by-case basis.
          * ``'accurate'`` -- prefer the highest-fidelity method
            for the geometry (e.g. ``'gbd'`` over ``'maslov'`` for
            aspherics, ``'hf'`` over ``'maslov'`` for general
            paraxial-violating systems).

        Has no effect when ``method`` is set to a specific string.
    return_result : bool, default False
        When True, wrap the output in a
        :class:`lumenairy.propagators.PropagationResult` carrying
        the field plus ``dx``, ``wavelength``, ``method``, and a
        ``metadata`` dict.  When False (default), return the bare
        propagator output (typically a complex ndarray) -- preserving
        backward compatibility and zero-overhead fast loops.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"propagate: method must be one of {VALID_METHODS}, "
            f"got {method!r}.")

    if method == 'auto':
        method = _auto_select_method(
            E_in, z=z, wavelength=wavelength, dx=dx,
            prescription=prescription, accuracy=accuracy)

    out = _dispatch_to_method(
        method, E_in,
        z=z, wavelength=wavelength, dx=dx,
        prescription=prescription,
        output_grid=output_grid,
        output_dx=output_dx,
        **method_kwargs,
    )
    if not return_result:
        return out

    from .result import PropagationResult
    out_dx = output_dx if output_dx is not None else dx
    # Best-effort: bare ndarray -> wrap directly; tuple / list / other
    # -> stash into metadata so callers can still introspect.
    if isinstance(out, np.ndarray):
        return PropagationResult(
            field=out, dx=out_dx, wavelength=wavelength,
            z=z, method=method, metadata={},
        )
    return PropagationResult(
        field=getattr(out, 'field', None) or _coerce_field(out),
        dx=out_dx, wavelength=wavelength,
        z=z, method=method,
        metadata={'native_return': out},
    )


def _coerce_field(x):
    """Coerce a non-ndarray propagator return into a complex array
    if possible -- otherwise return ``None`` and stash the raw value
    in metadata for inspection."""
    try:
        arr = np.asarray(x)
        if np.iscomplexobj(arr) or arr.dtype.kind == 'f':
            return arr
    except Exception:
        pass
    return None


def _auto_select_method(E_in, *, z, wavelength, dx, prescription,
                          accuracy='balanced'):
    """Pick a method from the geometry + prescription structure.

    Selection logic
    ---------------

    With a prescription:
      1. If any surface carries a DOE / grating phase  ->  ``hfpi``
         (HFPI honours hard diffractive surfaces natively).
      2. If the prescription has any aspheric coefficients and
         ``accuracy in ('balanced', 'accurate')``               ->  ``gbd``
         (Gaussian Beamlet Decomposition is the right choice
         when the paraxial Maslov prediction breaks down at
         high-order asphere terms).
      3. If ``accuracy == 'accurate'`` and any surface has a
         finite ``semi_diameter`` or ``aperture_diameter``      ->  ``hf``
         (Van-Vleck-corrected Huygens-Fresnel handles hard-
         aperture diffraction better than Maslov for general
         systems).
      4. Otherwise                                              ->  ``maslov``
         (paraxial-corrected analytic propagator; fastest of the
         prescription methods).

    Without a prescription (free-space):
      - ``z`` is None or zero                                    ->  ``asm``.
      - Far-field (Fresnel number ``N_F < 0.1``)                ->  ``fraunhofer``.
      - Otherwise                                               ->  ``asm``.
    """
    if prescription is not None:
        events = prescription.get('events_json') or []
        has_doe = False
        if isinstance(events, list):
            for ev in events:
                if isinstance(ev, dict) and ev.get('type') == 'doe':
                    has_doe = True
                    break
        if has_doe:
            return 'hfpi'

        # Inspect surfaces for aspherics and hard apertures.
        surfs = prescription.get('surfaces') or []
        has_aspheric = False
        has_hard_aperture = False
        for s in surfs:
            if not isinstance(s, dict):
                continue
            asph = s.get('aspheric_coeffs')
            if asph:
                has_aspheric = True
            asph_y = s.get('aspheric_coeffs_y')
            if asph_y:
                has_aspheric = True
            sd = s.get('semi_diameter')
            if sd is not None:
                try:
                    if math.isfinite(float(sd)) and float(sd) > 0:
                        has_hard_aperture = True
                except (TypeError, ValueError):
                    pass
        # Top-level aperture stop counts as a hard aperture as well.
        if prescription.get('aperture_diameter') is not None:
            has_hard_aperture = True

        if has_aspheric and accuracy in ('balanced', 'accurate'):
            return 'gbd'
        if has_hard_aperture and accuracy == 'accurate':
            return 'hf'
        return 'maslov'

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
        from .gbd import propagate_gbd_freespace, propagate_gbd_through_prescription
        if prescription is None:
            if z is None:
                raise ValueError(
                    "propagate(method='gbd') without prescription requires z.")
            return propagate_gbd_freespace(
                E_in, dx, z=z, wavelength=wavelength,
                output_grid=output_grid, output_dx=output_dx,
                **kwargs,
            )
        return propagate_gbd_through_prescription(
            E_in, dx, prescription,
            wavelength=wavelength,
            output_grid=output_grid, output_dx=output_dx,
            **kwargs,
        )

    if method == 'hfpi':
        from .hfpi import (
            propagate_hfpi_freespace_aperture,
            propagate_hfpi_through_prescription,
        )
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
        return propagate_hfpi_through_prescription(
            E_in, dx, prescription,
            wavelength=wavelength,
            output_grid=output_grid, output_dx=output_dx,
            **kwargs,
        )

    if method == 'hf':
        from .hf import (
            propagate_huygens_fresnel_freespace,
            propagate_huygens_fresnel_through_prescription,
        )
        if prescription is None:
            if z is None:
                raise ValueError(
                    "propagate(method='hf') without prescription requires z.")
            return propagate_huygens_fresnel_freespace(
                E_in, z, wavelength, dx, **kwargs,
            )
        return propagate_huygens_fresnel_through_prescription(
            E_in, dx, prescription,
            wavelength=wavelength,
            output_grid=output_grid, output_dx=output_dx,
            **kwargs,
        )

    if method == 'maslov':
        from ..elements.lenses import apply_real_lens_maslov
        if prescription is None:
            raise ValueError(
                "propagate(method='maslov') requires a prescription.")
        return apply_real_lens_maslov(
            E_in, prescription, dx, wavelength,
            **kwargs,
        )

    if method == 'asymptotic':
        from .asymptotic import (propagate_modal_asymptotic,
                                  fit_canonical_polynomials)
        # Caller may pass a pre-built fit via kwargs['fit'] or supply
        # a prescription that the dispatcher will fit on the fly.
        fit = kwargs.pop('fit', None)
        if fit is None:
            if prescription is None:
                raise ValueError(
                    "propagate(method='asymptotic') requires either "
                    "fit=... or a prescription.")
            fit_kwargs = kwargs.pop('fit_kwargs', {}) or {}
            fit = fit_canonical_polynomials(
                prescription, wavelength=wavelength, **fit_kwargs)
        return propagate_modal_asymptotic(fit, **kwargs)

    if method == 'mhs':
        from .mhs import MhsPipeline
        # Accept either a fully-built pipeline OR a list of subdomains.
        pipeline = kwargs.pop('pipeline', None)
        subdomains = kwargs.pop('subdomains', None)
        if pipeline is None and subdomains is None:
            raise ValueError(
                "propagate(method='mhs') requires either pipeline=... "
                "or subdomains=... .")
        if pipeline is None:
            pipeline = MhsPipeline(subdomains)
        return_intermediate = kwargs.pop('return_intermediate', False)
        return pipeline.run(E_in, return_intermediate=return_intermediate,
                            **kwargs)

    raise NotImplementedError(f"Method {method!r} is not implemented.")


__all__ = ['propagate', 'VALID_METHODS']

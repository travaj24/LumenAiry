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
            E_in, prescription=prescription, wavelength=wavelength, dx=dx,
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


# ============================================================================
# ASM-family auto-selector (asm_propagate) and advisor (which_propagator)
# ============================================================================

ASM_FAMILY = ('asm', 'asm_tilted', 'asm_mft', 'sas', 'fresnel', 'fraunhofer')


def _select_asm_variant(
    E_in,
    z: float,
    wavelength: float,
    dx: float,
    *,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    output_dx: Optional[float] = None,
    aperture_radius: Optional[float] = None,
) -> str:
    """Choose the best ASM-family propagator for the given geometry.

    Decision order:

    1. Output grid pitch requested AND different from input -> ``asm_mft``
       (Bluestein output sampling).
    2. Significant tilt (> 1e-6 rad) -> ``asm_tilted``.
    3. ``z >> L^2 / (N * lambda)`` (small Fresnel number) -> ``sas``
       for a scalable output pitch, else ``fraunhofer`` if extreme.
    4. ``z`` and aperture given, intermediate Fresnel number:
       still use plain ``asm`` (band-limited transfer function handles
       both near- and intermediate-field).
    5. Otherwise -> ``asm``.
    """
    has_tilt = (abs(float(tilt_x)) > 1e-6) or (abs(float(tilt_y)) > 1e-6)
    if output_dx is not None and abs(float(output_dx) - float(dx)) > 0:
        return 'asm_mft'
    if has_tilt:
        return 'asm_tilted'
    # Compare propagation distance to L^2 / (N * lambda) -- the
    # SAS-regime threshold.  We need an aperture radius or the grid
    # extent L to make this judgement; if neither is supplied, fall
    # back to plain ASM.
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    N = max(Ny, Nx)
    L = N * dx
    threshold = (L * L) / (N * wavelength)
    if abs(float(z)) > 20.0 * threshold:
        # Far-field-ish; Fraunhofer is closed-form and cheaper.
        return 'fraunhofer'
    if abs(float(z)) > 2.0 * threshold:
        # The beam has spread far enough that the SAS rescaling
        # pays off.
        return 'sas'
    return 'asm'


def which_propagator(
    E_in,
    z: float,
    wavelength: float,
    dx: float,
    *,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    output_dx: Optional[float] = None,
    aperture_radius: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Advise which ASM-family propagator to use without running one.

    Returns a dict with the chosen method name and a brief reason.
    Useful for documenting a design choice or surfacing the decision
    in a notebook / GUI.

    Parameters
    ----------
    E_in : ndarray
        Input field (only the shape and dtype are consulted).
    z, wavelength, dx : float
        Propagation geometry [m].
    tilt_x, tilt_y : float, optional
        Mean-direction tilt [rad].  Non-zero values steer the
        choice toward ``asm_tilted``.
    output_dx : float, optional
        Requested output pitch [m].  When different from ``dx``,
        steers toward ``asm_mft``.
    aperture_radius : float, optional
        Source aperture [m] used in the Fresnel-number heuristic.
    verbose : bool
        Print the decision to stdout (useful in interactive use).

    Returns
    -------
    advice : dict
        ``{'method': str, 'reason': str, 'fresnel_number': float}``.
    """
    method = _select_asm_variant(
        E_in, z, wavelength, dx,
        tilt_x=tilt_x, tilt_y=tilt_y,
        output_dx=output_dx, aperture_radius=aperture_radius)

    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    N = max(Ny, Nx)
    L = N * dx
    threshold = (L * L) / (N * wavelength)
    a = aperture_radius if aperture_radius is not None else (L / 2.0)
    if abs(float(z)) > 0:
        fn = (a * a) / (wavelength * abs(float(z)))
    else:
        fn = float('inf')

    reasons = {
        'asm':       'near/intermediate field; band-limited ASM is exact.',
        'asm_tilted':'mean propagation direction is tilted; use carrier-shifted ASM.',
        'asm_mft':   'output grid pitch != input; Bluestein output sampling.',
        'sas':       (f'z = {z!r} >> L^2/(N*lambda) = {threshold:.3g}; '
                       'scalable ASM rescales the output grid.'),
        'fraunhofer':'extreme far field; closed-form Fraunhofer is cheapest.',
    }
    advice = {
        'method': method,
        'reason': reasons.get(method, ''),
        'fresnel_number': float(fn),
    }
    if verbose:
        print(f"which_propagator -> {method}: {advice['reason']}")
    return advice


def asm_propagate(
    E_in,
    z: float,
    wavelength: float,
    dx: float,
    *,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    output_dx: Optional[float] = None,
    output_N: Optional[int] = None,
    aperture_radius: Optional[float] = None,
    bandlimit: bool = True,
    verbose: bool = False,
    **method_kwargs,
):
    """Auto-select and run the best ASM-family propagator.

    Calls :func:`which_propagator` to pick between ``asm`` /
    ``asm_tilted`` / ``asm_mft`` / ``sas`` / ``fraunhofer`` based on
    the geometry, then dispatches to the chosen function.

    Parameters
    ----------
    E_in, z, wavelength, dx, tilt_x, tilt_y, output_dx, aperture_radius :
        Forwarded to :func:`which_propagator`.
    output_N : int, optional
        Output grid size when ``output_dx`` is given (required for the
        MFT-style sampler).  Defaults to the input grid size.
    bandlimit : bool
        Passed through to ASM-family propagators that accept it.
    verbose : bool
        Print the chosen method.
    **method_kwargs : dict
        Forwarded to the underlying propagator.

    Returns
    -------
    The chosen propagator's native return value (most return a bare
    ``ndarray``; the MFT variants return a 3-tuple).
    """
    advice = which_propagator(
        E_in, z, wavelength, dx,
        tilt_x=tilt_x, tilt_y=tilt_y,
        output_dx=output_dx, aperture_radius=aperture_radius,
        verbose=verbose)
    method = advice['method']

    from .propagation import (
        angular_spectrum_propagate,
        angular_spectrum_propagate_tilted,
        angular_spectrum_propagate_mft,
        scalable_angular_spectrum_propagate,
        fraunhofer_propagate,
    )

    if method == 'asm':
        return angular_spectrum_propagate(
            E_in, z, wavelength, dx, bandlimit=bandlimit,
            **method_kwargs)
    if method == 'asm_tilted':
        return angular_spectrum_propagate_tilted(
            E_in, z, wavelength, dx, tilt_x=tilt_x, tilt_y=tilt_y,
            bandlimit=bandlimit, **method_kwargs)
    if method == 'asm_mft':
        Ny, Nx = E_in.shape[-2], E_in.shape[-1]
        N_out = int(output_N) if output_N is not None else max(Ny, Nx)
        return angular_spectrum_propagate_mft(
            E_in, z, wavelength, dx, output_dx, N_out,
            bandlimit=bandlimit, **method_kwargs)
    if method == 'sas':
        return scalable_angular_spectrum_propagate(
            E_in, z, wavelength, dx, **method_kwargs)
    if method == 'fraunhofer':
        return fraunhofer_propagate(
            E_in, z, wavelength, dx, **method_kwargs)
    raise NotImplementedError(
        f"asm_propagate: internal error -- method {method!r} not "
        f"dispatched.")


__all__ = [
    'propagate', 'VALID_METHODS',
    'asm_propagate', 'which_propagator', 'ASM_FAMILY',
]

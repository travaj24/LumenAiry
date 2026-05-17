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

from typing import Any, Dict, Optional, Tuple, Union

import math

import numpy as np

from ..backend import array_namespace


VALID_METHODS = (
    'auto', 'asm', 'sas', 'fresnel', 'fraunhofer', 'rs',
    'maslov', 'asymptotic', 'gbd', 'hfpi', 'hf', 'mhs',
)


def propagate(
    E_in: np.ndarray,
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
    **method_kwargs: Any,
) -> Any:
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
    output_grid, output_dx : tuple / float, optional
        Request an output grid that differs from the input pitch.

        - GBD / HFPI / HF forward these to their underlying
          (prescription-driven) propagators directly.
        - ASM / Fresnel / Fraunhofer auto-promote to their MFT
          variants (:func:`angular_spectrum_propagate_mft`,
          :func:`fresnel_propagate_mft`,
          :func:`fraunhofer_propagate_mft`) when ``output_grid`` or
          ``output_dx`` is given.
        - SAS / RS do not support arbitrary output-grid sampling and
          raise ``ValueError`` (pointing at the ASM-MFT entry point)
          if ``output_grid`` / ``output_dx`` is passed.

        ``output_grid`` may be a ``(N_out, dx_out)`` tuple or a
        ``{'N': ..., 'dx': ...}`` dict.  ``output_dx`` is a shortcut
        when only the pitch needs to change (``N_out`` defaults to
        the input ``N``).  Pre-4.12 the ASM family silently dropped
        these kwargs and returned a bare-grid output at the input
        pitch -- a quiet wrong-physics path that audit round-4 B1-8
        flagged.
    return_result : bool, default False
        When True, wrap the output in a
        :class:`lumenairy.propagators.PropagationResult` carrying
        the field plus ``dx``, ``wavelength``, ``method``, and a
        ``metadata`` dict.  When False (default), return the bare
        propagator output (typically a complex ndarray) -- preserving
        backward compatibility and zero-overhead fast loops.

        4.12: for tuple-returning kernels (Fresnel / Fraunhofer / SAS
        return ``(E, dx_out, dy_out)``) the wrapped result now reports
        the kernel's **output** dx, not the input dx.  Pre-4.12 audit
        round-4 B1-7: tuple unpacking silently failed, ``field`` was
        ``None``, and ``dx`` was the input pitch.
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
    default_out_dx = output_dx if output_dx is not None else dx
    # Best-effort: bare ndarray -> wrap directly; tuple / list / other
    # -> unpack the field and record the propagator-reported output
    # pitch when present.  4.12 fix (audit round-4 B1-7): kernels like
    # fresnel_propagate / fraunhofer_propagate / scalable_angular_spectrum_propagate
    # return ``(E, dx_out, dy_out)``; pre-4.12 the tuple path went
    # through _coerce_field which silently dropped to None and reported
    # the INPUT dx instead of the kernel's output dx.
    if isinstance(out, np.ndarray):
        return PropagationResult(
            field=out, dx=default_out_dx, wavelength=wavelength,
            z=z, method=method, metadata={},
        )
    # PropagationResult passthrough (some propagators may already wrap).
    if isinstance(out, PropagationResult):
        return out
    field_arr, dx_from_kernel, dy_from_kernel = _coerce_field(out)
    out_dx = dx_from_kernel if dx_from_kernel is not None else default_out_dx
    # v4.13.0 (audit L3): thread the kernel-reported ``dy_out`` onto
    # the wrapped result.  For square-grid kernels that only return
    # ``dx_out`` (or a bare ndarray) ``dy`` falls back to ``out_dx``,
    # preserving back-compat.  Pre-fix the y-pitch was silently
    # discarded for anamorphic Fresnel / Fraunhofer / SAS calls.
    out_dy = dy_from_kernel if dy_from_kernel is not None else out_dx
    return PropagationResult(
        field=field_arr,
        dx=out_dx, dy=out_dy, wavelength=wavelength,
        z=z, method=method,
        metadata={'native_return': out},
    )


def _coerce_field(x):
    """Coerce a non-ndarray propagator return into a (field, dx_out,
    dy_out) triple when possible.

    Returns ``(ndarray | None, dx_out | None, dy_out | None)``.

    * ``dx_out`` / ``dy_out`` are the propagator-reported output grid
      pitches if the kernel returns a ``(E, dx_out, ...)`` /
      ``(E, dx_out, dy_out)`` tuple, else ``None``.
    * v4.13.0 (audit L3): the triple-return is the closure for the
      anamorphic Fresnel info-loss bug -- pre-fix ``_coerce_field``
      ignored the third tuple element, silently discarding the y-axis
      pitch for any anamorphic Fresnel / Fraunhofer / SAS propagation.
    * 4.12 fix (audit round-4 B1-7): pre-4.12 the tuple-returning
      propagators (fresnel/fraunhofer/SAS) silently yielded
      ``field=None`` and ``dx=<input pitch>`` instead of the kernel's
      real output.

    The dispatcher records ``dx_out`` on :attr:`PropagationResult.dx`
    and ``dy_out`` on :attr:`PropagationResult.dy`; when the kernel
    returns only ``dx_out`` (or only the bare ndarray) the dispatcher
    falls back to ``dy = dx`` for back-compat.
    """
    # Tuple / list returned by fresnel_propagate, fraunhofer_propagate,
    # scalable_angular_spectrum_propagate -- shape ``(E, dx_out, dy_out)``
    # for the all-FFT methods; ``(E, dx_out)`` for the resample helper.
    if isinstance(x, (tuple, list)) and len(x) >= 1:
        first = x[0]
        if isinstance(first, np.ndarray):
            dx_out = None
            dy_out = None
            if len(x) >= 2:
                try:
                    dx_out = float(x[1])
                except (TypeError, ValueError):
                    dx_out = None
            if len(x) >= 3:
                try:
                    dy_out = float(x[2])
                except (TypeError, ValueError):
                    dy_out = None
            return first, dx_out, dy_out
        return None, None, None
    try:
        arr = np.asarray(x)
        if np.iscomplexobj(arr) or arr.dtype.kind == 'f':
            return arr, None, None
    except (TypeError, ValueError):
        # np.asarray rejects non-array-like inputs with TypeError;
        # ragged / inhomogeneous sequences raise ValueError.  Either
        # way the kernel return doesn't look like a field and we fall
        # through to the (None, None, None) sentinel below.
        pass
    return None, None, None


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
      - Grid Fresnel ratio ``Q = z*lambda/(N*dx^2) > 1``         ->  ``sas``
        (scalable ASM rescales the output pitch so the spread
        beam fits without aliasing the ASM transfer function).
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
    N_max = max(Ny, Nx)
    a = 0.5 * dx * N_max
    abs_z = abs(z)
    if abs_z == 0:
        return 'asm'
    # 4.12 fix (audit round-4 B1-6): for z < 0 (back-propagation) the
    # forward-only kernels (Fresnel / Fraunhofer / SAS / RS) all raise
    # ValueError.  Restrict the regime check to the back-propagating
    # methods (ASM is the only auto-selectable option) so the dispatcher
    # never silently routes the user into a hard-raise from a kernel
    # they didn't pick by name.  Users who need MFT-style back-propagation
    # at custom output pitch should call angular_spectrum_propagate_mft
    # directly (the auto-selector here only routes between bare-z kernels).
    if z < 0:
        return 'asm'
    N_F = a * a / (wavelength * abs_z)
    if N_F < 0.1:
        return 'fraunhofer'
    # Grid Fresnel ratio Q = z*lambda/(N*dx**2).  When Q > 1 the plain
    # ASM transfer function aliases on the grid; scalable ASM rescales
    # the output pitch so the beam fits without aliasing.
    Q = wavelength * abs_z / (N_max * dx * dx)
    if Q > 1.0:
        return 'sas'
    return 'asm'


_FORWARD_ONLY_METHODS = ('sas', 'fresnel', 'fraunhofer', 'rs')


def _dispatch_bare_grid_with_output(method, E_in, *, z, wavelength, dx,
                                     output_grid, output_dx, **kwargs):
    """Route a bare-grid method (asm/fresnel/fraunhofer/sas/rs) to the
    correct MFT variant when the caller asks for an output-pitch /
    output-grid that differs from the natural FFT output.

    4.12 fix (audit round-4 B1-8).  Behaviour:
      - ``asm`` -> :func:`angular_spectrum_propagate_mft` (forward or
        back-prop -- ASM-MFT supports any sign of z).
      - ``fresnel`` -> :func:`fresnel_propagate_mft` (forward-only).
      - ``fraunhofer`` -> :func:`fraunhofer_propagate_mft` (forward-only).
      - ``sas`` / ``rs`` -> ValueError; no MFT analogue in 4.12.

    ``output_grid`` can be ``(N_out, dx_out)`` or a dict
    ``{'N': ..., 'dx': ...}``.  ``output_dx`` short-circuits and uses
    the input N for the MFT N_out.
    """
    if z is None:
        raise ValueError(
            f"propagate(method={method!r}, output_grid/output_dx=...): "
            f"z is required for an MFT-style output-grid call.")

    # Resolve N_out, dx_out from output_grid or output_dx.
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    N_in = max(Ny, Nx)
    dx_out = None
    N_out = None
    if output_grid is not None:
        if isinstance(output_grid, dict):
            N_out = output_grid.get('N')
            dx_out = output_grid.get('dx')
        elif isinstance(output_grid, (tuple, list)) and len(output_grid) >= 2:
            N_out, dx_out = output_grid[0], output_grid[1]
        else:
            raise ValueError(
                f"propagate(method={method!r}, output_grid=...): "
                f"output_grid must be a (N_out, dx_out) tuple or "
                f"{{'N': ..., 'dx': ...}} dict, got {type(output_grid).__name__}.")
    if output_dx is not None:
        dx_out = output_dx
    if dx_out is None:
        raise ValueError(
            f"propagate(method={method!r}, output_grid=...): could not "
            f"resolve an output dx from output_grid={output_grid!r} "
            f"or output_dx={output_dx!r}.")
    if N_out is None:
        N_out = N_in
    N_out = int(N_out)
    dx_out = float(dx_out)

    if method == 'asm':
        from .propagation import angular_spectrum_propagate_mft
        return angular_spectrum_propagate_mft(
            E_in, z, wavelength, dx, dx_out, N_out, **kwargs)
    if method == 'fresnel':
        from .propagation import fresnel_propagate_mft
        return fresnel_propagate_mft(
            E_in, z, wavelength, dx, dx_out, N_out, **kwargs)
    if method == 'fraunhofer':
        from .propagation import fraunhofer_propagate_mft
        return fraunhofer_propagate_mft(
            E_in, z, wavelength, dx, dx_out, N_out, **kwargs)
    if method == 'sas':
        raise ValueError(
            f"propagate(method='sas', output_grid/output_dx=...): "
            f"SAS does not support arbitrary output-grid sampling.  Its "
            f"output pitch is fixed by `dx_out = lambda*z/(pad*N*dx)`.  "
            f"Use method='asm' (auto-promotes to angular_spectrum_propagate_mft) "
            f"for explicit output-pitch sampling, or method='fresnel' for "
            f"the paraxial-MFT path.")
    if method == 'rs':
        raise ValueError(
            f"propagate(method='rs', output_grid/output_dx=...): "
            f"Rayleigh-Sommerfeld does not support arbitrary output-grid "
            f"sampling in 4.12 (no MFT variant).  Use method='asm' "
            f"(auto-promotes to angular_spectrum_propagate_mft) for "
            f"output-pitch sampling.")
    raise NotImplementedError(
        f"_dispatch_bare_grid_with_output: method {method!r} not "
        f"covered.")


def _dispatch_to_method(method, E_in, *, z, wavelength, dx,
                        prescription, output_grid, output_dx,
                        **kwargs):
    """Call the chosen propagator with the appropriate signature.

    4.12 fix (audit round-4 B1-6): when the user explicitly picks a
    forward-only method (Fresnel / Fraunhofer / SAS / RS) with z < 0,
    raise a dispatcher-level ValueError naming :func:`propagate` rather
    than letting the kernel raise a confusing error that mentions the
    underlying function the user didn't call by name.  ASM is the only
    auto-supported back-propagation kernel here; users who need
    MFT-style back-prop at a custom output pitch should call
    :func:`angular_spectrum_propagate_mft` directly.

    4.12 fix (audit round-4 B1-8): when the caller passes
    ``output_grid`` / ``output_dx`` and the chosen method is not an MFT
    variant, raise a clear ValueError pointing at the right MFT entry
    point rather than silently dropping the user's request.  The ASM /
    Fresnel / Fraunhofer / SAS / RS kernels in this dispatcher accept
    only the natural FFT output grid; explicit output-pitch sampling
    needs the MFT family (angular_spectrum_propagate_mft,
    fresnel_propagate_mft, fraunhofer_propagate_mft).  GBD / HFPI / HF
    forward ``output_grid`` / ``output_dx`` directly to their
    underlying propagators.
    """
    if method in _FORWARD_ONLY_METHODS and z is not None and float(z) < 0:
        raise ValueError(
            f"propagate(method={method!r}): z must be > 0 (got z={z}).  "
            f"This method is a forward-only propagator.  Use "
            f"method='asm' (or call angular_spectrum_propagate_mft "
            f"directly for custom output-pitch sampling) for "
            f"back-propagation.")

    # Bare-grid methods (no prescription) do not honour output_grid /
    # output_dx -- they always produce the natural FFT output grid.
    # When the caller supplies an output-grid request, route them to
    # the MFT variant or raise a clear ValueError.  Free-space GBD /
    # HFPI / HF *do* take output_grid / output_dx and forward them
    # through their own dispatch below.
    _BARE_GRID_METHODS = ('asm', 'sas', 'fresnel', 'fraunhofer', 'rs')
    if method in _BARE_GRID_METHODS and (output_grid is not None
                                          or output_dx is not None):
        return _dispatch_bare_grid_with_output(
            method, E_in, z=z, wavelength=wavelength, dx=dx,
            output_grid=output_grid, output_dx=output_dx, **kwargs,
        )

    if method == 'asm':
        from .propagation import angular_spectrum_propagate
        if z is None:
            return E_in
        return angular_spectrum_propagate(E_in, z, wavelength, dx, **kwargs)

    if method == 'sas':
        from .propagation import scalable_angular_spectrum_propagate
        if z is None:
            raise ValueError("propagate(method='sas'): z is required.")
        return scalable_angular_spectrum_propagate(
            E_in, z, wavelength, dx, **kwargs)

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
    E_in: np.ndarray,
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
    E_in: np.ndarray,
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
    **method_kwargs: Any,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
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

"""
Sequential optical system propagation.

Provides the :func:`propagate_through_system` function which propagates an
electric field through an ordered sequence of optical elements, dispatching
each element to the appropriate physics routine from the library submodules.

Author: Andrew Traverso
"""

from __future__ import annotations

import threading
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    # v5.0.1 (audit P1-NEW-V4-1): TYPE_CHECKING guard for forward references
    # to ``Source`` and ``PropagationResult`` in ``evaluate(...)``.  Both
    # are imported lazily inside the function body to avoid top-level
    # circular dependencies; ruff F821 still needs a typing-time binding
    # so the string-quoted annotations resolve.
    from ..sources import Source
    from .result import PropagationResult

from .propagation import (
    angular_spectrum_propagate,
    angular_spectrum_propagate_tilted,
    fresnel_propagate,
    resample_field,
    _resolve_jax_complex_dtype,
)
from ..elements.lenses import (
    apply_thin_lens,
    apply_spherical_lens,
    apply_aspheric_lens,
    apply_real_lens,
    apply_real_lens_traced,
    apply_cylindrical_lens,
    apply_grin_lens,
    apply_axicon,
)
from ..elements import (
    apply_mirror,
    apply_aperture,
    apply_gaussian_aperture,
    apply_mask,
    apply_zernike_aberration,
    generate_turbulence_screen,
)


def propagate_through_system(E_in: np.ndarray,
                             elements: Sequence[Dict[str, Any]],
                             wavelength: float,
                             dx: float,
                             dy: Optional[float] = None,
                             method: str = 'asm',
                             use_gpu: bool = False,
                             verbose: bool = False,
                             progress: Optional[Callable] = None,
                             checkpoint: Optional[str] = None,
                             store: Optional[str] = None,
                             label_prefix: str = 'system',
                             return_result: bool = False
                             ) -> Union[Tuple[np.ndarray, List[Any]], Any]:
    """
    Propagate a field through a sequence of optical elements.

    Parameters
    ----------
    E_in : ndarray (complex)
        Input electric field.

    elements : list of dict
        List of optical elements.  Each element is a dict whose ``'type'``
        key selects the physics model, and whose remaining keys supply the
        parameters for that model.

    wavelength : float
        Optical wavelength in meters.

    dx : float
        Grid spacing in x-direction in meters.

    dy : float, optional
        Grid spacing in y-direction.  If None, assumes dy = dx.

    method : str, default 'asm'
        Propagation method for free-space ``'propagate'`` steps.
        Supported values:

        - ``'asm'`` : Angular Spectrum Method (exact, fixed grid).
        - ``'fresnel'`` : Single-FFT Fresnel (paraxial, grid spacing
          changes at each step; auto-resampled back to ``dx`` before
          the next element so lens/aperture phases stay correct).
        - ``'sas'`` : Scalable Angular Spectrum Method
          (Heintzmann-Loetgering-Wechsler 2023).  Correct choice when
          the propagation distance is long enough that ASM needs an
          impractically large grid.  Output pitch is
          ``lambda*z/(pad*N*dx)`` by construction; the pipeline
          auto-resamples back to ``dx`` between elements so the rest
          of the system (lenses, apertures) keeps its physical
          coordinates.  Extra keys on the element dict:
          ``pad`` (int, default 2), ``skip_final_phase``
          (bool, default False).

        Element-level physics (lenses, mirrors, apertures) are always
        applied using their native models regardless of this setting.

        To compare methods, run the same ``elements`` list twice with
        different ``method`` values::

            E_asm, _ = propagate_through_system(E, elems, lam, dx, method='asm')
            E_fre, _ = propagate_through_system(E, elems, lam, dx, method='fresnel')

    use_gpu : bool, default=False
        Use GPU acceleration if available.

    verbose : bool, default=False
        Print progress information and record intermediate fields.
    checkpoint : callable, optional
        ``checkpoint(idx, elem, E)`` is called after every element
        finishes (idx=0 is the input plane, before any element runs).
        Use to stream intermediate fields without paying the
        per-element copy that ``verbose=True`` triggers.  Returning
        from the callback is fine; raising aborts the run.
    store : str, pathlib.Path, or storage handle, optional
        If given, each intermediate field is appended to a unified
        Zarr / HDF5 store via
        :func:`lumenairy.io.storage.append_plane`.  Plane labels are
        ``f"{label_prefix}_{idx:03d}_{elem_type}"``.
    label_prefix : str, default 'system'
        Prefix for plane labels when ``store`` is provided.
    return_result : bool, default False
        If True, return a
        :class:`lumenairy.propagators.PropagationResult` carrying the
        final field plus a ``history`` of per-element planes (labels
        derived from each element's ``type`` key).  ``intermediates``
        is auto-populated on the result.  When False (default), the
        legacy ``(E_out, intermediates)`` tuple is returned.

    Returns
    -------
    E_out : ndarray (complex)
        Output field after all elements.

    intermediates : list of ndarray
        Field at each stage (only populated when *verbose=True*).

    Supported element types
    -----------------------
    ``'propagate'``
        Free-space propagation.  Uses the ``method`` parameter to select
        ASM (default) or Fresnel.  Optional tilt parameters auto-select
        tilted ASM when present.
        Keys: ``z`` (float, propagation distance [m]),
        ``bandlimit`` (bool, optional, default True),
        ``tilt_x`` (float, optional, default 0, carrier tilt [rad]),
        ``tilt_y`` (float, optional, default 0, carrier tilt [rad]),
        ``method`` (str, optional, per-element override of the system
        ``method`` parameter).

    ``'propagate_tilted'``
        Legacy alias for ``'propagate'`` with tilt parameters.
        Prefer using ``{'type': 'propagate', 'tilt_x': ..., 'tilt_y': ...}``
        in new code.

    ``'lens'``
        Thin-lens phase screen (paraxial or higher-order model).
        Keys: ``f`` (float, focal length [m]),
        ``xc`` / ``yc`` (float, optional, lens center, default 0),
        ``lens_model`` (str, optional, default ``'paraxial'``).

    ``'spherical_lens'``
        Thick spherical lens (two curved surfaces + propagation).
        Keys: ``R1``, ``R2`` (float, surface radii of curvature),
        ``d`` (float, center thickness), ``n_lens`` (float, refractive index),
        ``aperture_diameter`` (float, optional),
        ``xc`` / ``yc`` (float, optional, default 0).

    ``'aspheric_lens'``
        Thick aspheric lens with conic and polynomial terms.
        Keys: ``R1``, ``R2``, ``d``, ``n_lens`` (as for spherical_lens),
        ``k1`` / ``k2`` (float, optional, conic constants, default 0),
        ``A1`` / ``A2`` (array-like, optional, polynomial coefficients),
        ``aperture_diameter`` (float, optional),
        ``xc`` / ``yc`` (float, optional, default 0).

    ``'real_lens'``
        Real lens from a prescription table (analytic thin-element model).
        Keys: ``prescription`` (dict/object with full lens data),
        ``bandlimit`` (bool, optional, default True),
        ``slant_correction`` / ``fresnel`` / ``absorption`` (optional,
        passed through to :func:`apply_real_lens`).

    ``'real_lens_traced'``
        Real lens via the hybrid wave/ray model with sub-nm OPD agreement
        with the geometric ray trace.  Slower but high-accuracy; the
        recommended choice for cemented doublets and other multi-surface
        curved-interface systems.
        Keys: ``prescription`` (dict),
        ``bandlimit`` (bool, optional, default True),
        ``ray_subsample`` (int, optional, default 1; 4 is the recommended
        production value — ~15x faster with sub-nm fidelity).

    ``'cylindrical_lens'``
        Thin cylindrical lens (power in one axis only).
        Keys: ``f`` (float, focal length),
        ``axis`` (str, optional, ``'x'`` or ``'y'``, default ``'x'``),
        ``xc`` / ``yc`` (float, optional, default 0).

    ``'axicon'``
        Conical lens (axicon).
        Keys: ``alpha`` (float, cone half-angle [rad]),
        ``n_axicon`` (float, refractive index),
        ``xc`` / ``yc`` (float, optional, default 0).

    ``'grin_lens'``
        Gradient-index lens segment.
        Keys: ``n0`` (float, on-axis index), ``g`` (float, gradient constant),
        ``d`` (float, length),
        ``xc`` / ``yc`` (float, optional, default 0).

    ``'mirror'``
        Reflective surface (flat or curved).
        Keys: ``radius`` (float, optional, radius of curvature),
        ``conic`` (float, optional, default 0),
        ``aperture_diameter`` (float, optional),
        ``xc`` / ``yc`` (float, optional, default 0).

    ``'aperture'``
        Hard aperture (circular, rectangular, etc.).
        Keys: ``shape`` (str, optional, default ``'circular'``),
        ``params`` (dict, optional, shape-specific parameters),
        ``xc`` / ``yc`` (float, optional, default 0).

    ``'gaussian_aperture'``
        Soft Gaussian aperture.
        Keys: ``sigma`` (float, 1/e^2 radius [m]),
        ``xc`` / ``yc`` (float, optional, default 0).

    ``'mask'``
        Arbitrary complex transmission mask.
        Keys: ``mask`` (ndarray, same shape as field).

    ``'zernike'``
        Zernike polynomial aberration.
        Keys: ``coefficients`` (dict or array, Zernike coefficients),
        ``aperture_radius`` (float [m]).

    ``'turbulence'``
        Atmospheric / random turbulence phase screen.
        Keys: ``r0`` (float, Fried parameter [m]),
        ``L0`` (float, optional, outer scale, default inf),
        ``l0`` (float, optional, inner scale, default 0),
        ``seed`` (int, optional).

    Examples
    --------
    >>> # 4f imaging system
    >>> f1, f2 = 1e-3, 4.27e-3  # focal lengths
    >>> elements = [
    ...     {'type': 'propagate', 'z': f1},           # to first lens
    ...     {'type': 'lens', 'f': f1},                # first lens
    ...     {'type': 'propagate', 'z': f1 + f2},      # to second lens
    ...     {'type': 'lens', 'f': f2},                # second lens
    ...     {'type': 'propagate', 'z': f2},           # to image plane
    ... ]
    >>> E_out, _ = propagate_through_system(E_in, elements, wavelength, dx)
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard for PartialCoherenceMCF
    # and non-2-D inputs via the shared ``_check_2d_scalar_field``
    # helper.  v4.15.2 inlined the guard at each entry point; v4.15.3
    # consolidates 10 sites (plus 9 new siblings) onto the helper so
    # future entry points can't be added unguarded.  Runs FIRST
    # (before any input copy / dx bookkeeping) so the user gets a
    # clear, actionable error rather than a downstream AttributeError
    # or a silent wrong-axis FFT.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'propagate_through_system')

    from ..progress import call_progress, ProgressScaler
    if store is not None:
        from ..io.storage import append_plane

    history_entries = []  # list of (label, field, dx) for return_result

    def _persist(idx, elem_type, field, dx_now):
        if checkpoint is not None:
            checkpoint(idx, elem_type, field)
        if store is not None:
            label = f"{label_prefix}_{idx:03d}_{elem_type}"
            append_plane(store, field, dx_now, label=label)
        if return_result:
            history_entries.append(
                (f"{idx:03d}_{elem_type}",
                 field.copy() if hasattr(field, 'copy') else np.array(field),
                 float(dx_now)))

    E = E_in.copy() if hasattr(E_in, 'copy') else np.array(E_in)
    intermediates = []
    current_dx = dx
    current_dy = dy if dy is not None else dx
    _persist(0, 'input', E, current_dx)

    n_elem = max(1, len(elements))
    for i, elem in enumerate(elements):
        if verbose:
            print(f"  Stage {i+1}/{len(elements)}: {elem['type']}")
        call_progress(progress, 'system', i / n_elem,
                      f"{i + 1}/{n_elem}: {elem.get('type', '?')}")
        # Pass a sub-scaler into lens models so each per-surface step
        # also surfaces (ha) through the main callback.
        sub_cb = ProgressScaler(progress, 'system',
                                i / n_elem, (i + 1) / n_elem)

        if elem['type'] == 'propagate':
            z = elem['z']
            bandlimit = elem.get('bandlimit', True)
            # Per-element method override: elem.get('method') > system method
            prop_method = elem.get('method', method)

            # Check for tilt parameters — if present, use tilted ASM
            tilt_x = elem.get('tilt_x', 0.0)
            tilt_y = elem.get('tilt_y', 0.0)
            has_tilt = (tilt_x != 0.0) or (tilt_y != 0.0)

            if prop_method == 'fresnel' and not has_tilt:
                E, dx_new, _dy_new = fresnel_propagate(
                    E, z, wavelength, current_dx, current_dy)
                # Resample back to the original grid spacing so
                # downstream element phases (lenses, apertures) are
                # computed on the correct coordinate system.  Both
                # axes converge to current_dx since `resample_field`
                # produces a square grid; current_dy stays in sync.
                if abs(dx_new - current_dx) > current_dx * 1e-6:
                    if verbose:
                        print(f"    Fresnel dx changed: "
                              f"{current_dx*1e6:.3f} -> {dx_new*1e6:.3f} um, "
                              f"resampling back to {current_dx*1e6:.3f} um")
                    E, _ = resample_field(E, dx_new, current_dx,
                                          N_out=E_in.shape[-1])
            elif prop_method == 'sas' and not has_tilt:
                from .propagation import scalable_angular_spectrum_propagate
                pad = elem.get('pad', 2)
                skip_final_phase = elem.get('skip_final_phase', False)
                E, dx_new, _dy_new = scalable_angular_spectrum_propagate(
                    E, z, wavelength, current_dx,
                    pad=pad, skip_final_phase=skip_final_phase,
                    use_gpu=use_gpu, verbose=verbose)
                # Resample back to the original grid pitch so downstream
                # element phases stay on the right coordinate system.
                if abs(dx_new - current_dx) > current_dx * 1e-6:
                    if verbose:
                        print(f"    SAS dx changed: "
                              f"{current_dx*1e6:.3f} -> {dx_new*1e6:.3f} um, "
                              f"resampling back to {current_dx*1e6:.3f} um")
                    E, _ = resample_field(E, dx_new, current_dx,
                                          N_out=E_in.shape[-1])
            elif has_tilt:
                # Tilted ASM (always ASM — no tilted Fresnel variant)
                if verbose and prop_method == 'fresnel':
                    print(f"    tilt specified — using tilted ASM "
                          f"instead of Fresnel")
                E = angular_spectrum_propagate_tilted(
                    E, z, wavelength, current_dx, current_dy,
                    tilt_x=tilt_x, tilt_y=tilt_y,
                    bandlimit=bandlimit)
            else:
                # Default: ASM
                E = angular_spectrum_propagate(
                    E, z, wavelength, current_dx, current_dy,
                    bandlimit=bandlimit, use_gpu=use_gpu,
                    verbose=verbose)

        elif elem['type'] == 'lens':
            f = elem['f']
            xc = elem.get('xc', 0)
            yc = elem.get('yc', 0)
            model = elem.get('lens_model', 'paraxial')
            # v4.13.2 (C-P1-3): thread current_dx/current_dy instead of
            # the original dx/dy so anamorphic grids (dx != dy) and any
            # mid-pipeline pitch changes propagate through every
            # element-handler call, not just the propagate_* steps.
            E = apply_thin_lens(E, f=f, wavelength=wavelength,
                                dx=current_dx, dy=current_dy,
                                xc=xc, yc=yc,
                                use_gpu=use_gpu, lens_model=model)

        elif elem['type'] == 'spherical_lens':
            E = apply_spherical_lens(E, R1=elem['R1'], R2=elem['R2'], d=elem['d'],
                                n_lens=elem['n_lens'], wavelength=wavelength,
                                dx=current_dx, dy=current_dy,
                                aperture_diameter=elem.get('aperture_diameter'),
                                xc=elem.get('xc', 0), yc=elem.get('yc', 0),
                                use_gpu=use_gpu)

        elif elem['type'] == 'aspheric_lens':
            E = apply_aspheric_lens(E, R1=elem['R1'], R2=elem['R2'], d=elem['d'],
                                n_lens=elem['n_lens'], wavelength=wavelength,
                                dx=current_dx, dy=current_dy,
                                k1=elem.get('k1', 0), k2=elem.get('k2', 0),
                                A1=elem.get('A1'), A2=elem.get('A2'),
                                aperture_diameter=elem.get('aperture_diameter'),
                                xc=elem.get('xc', 0), yc=elem.get('yc', 0),
                                use_gpu=use_gpu)

        elif elem['type'] == 'real_lens':
            E = apply_real_lens(
                E, prescription=elem['prescription'], wavelength=wavelength,
                dx=current_dx, dy=current_dy,
                bandlimit=elem.get('bandlimit', True),
                slant_correction=elem.get('slant_correction', False),
                fresnel=elem.get('fresnel', False),
                absorption=elem.get('absorption', False),
                progress=(lambda stage, frac, msg='': sub_cb(frac, msg))
                        if progress is not None else None,
            )

        elif elem['type'] == 'real_lens_traced':
            E = apply_real_lens_traced(
                E, prescription=elem['prescription'], wavelength=wavelength,
                dx=current_dx, dy=current_dy,
                bandlimit=elem.get('bandlimit', True),
                ray_subsample=elem.get('ray_subsample', 1),
                progress=(lambda stage, frac, msg='': sub_cb(frac, msg))
                        if progress is not None else None,
            )

        elif elem['type'] == 'mirror':
            E = apply_mirror(E, wavelength, current_dx,
                             radius=elem.get('radius'),
                             conic=elem.get('conic', 0.0),
                             aperture_diameter=elem.get('aperture_diameter'),
                             xc=elem.get('xc', 0), yc=elem.get('yc', 0),
                             dy=current_dy)

        elif elem['type'] == 'aperture':
            E = apply_aperture(E, current_dx,
                               shape=elem.get('shape', 'circular'),
                               params=elem.get('params', {}),
                               xc=elem.get('xc', 0), yc=elem.get('yc', 0),
                               dy=current_dy)

        elif elem['type'] == 'cylindrical_lens':
            E = apply_cylindrical_lens(E, f=elem['f'], wavelength=wavelength,
                                       dx=current_dx, dy=current_dy,
                                       axis=elem.get('axis', 'x'),
                                       xc=elem.get('xc', 0),
                                       yc=elem.get('yc', 0))

        elif elem['type'] == 'axicon':
            E = apply_axicon(E, elem['alpha'], elem['n_axicon'],
                             wavelength, current_dx, current_dy,
                             xc=elem.get('xc', 0), yc=elem.get('yc', 0))

        elif elem['type'] == 'grin_lens':
            E = apply_grin_lens(E, n0=elem['n0'], g=elem['g'], d=elem['d'],
                                wavelength=wavelength,
                                dx=current_dx, dy=current_dy,
                                xc=elem.get('xc', 0), yc=elem.get('yc', 0))

        elif elem['type'] == 'mask':
            E = apply_mask(E, elem['mask'])

        elif elem['type'] == 'propagate_tilted':
            # Legacy alias — redirect to the unified 'propagate' handler
            # with tilt parameters.  New code should use:
            #   {'type': 'propagate', 'z': ..., 'tilt_x': ..., 'tilt_y': ...}
            E = angular_spectrum_propagate_tilted(
                E, elem['z'], wavelength, current_dx, current_dy,
                tilt_x=elem.get('tilt_x', 0),
                tilt_y=elem.get('tilt_y', 0),
                bandlimit=elem.get('bandlimit', True))

        elif elem['type'] == 'turbulence':
            screen = generate_turbulence_screen(
                E.shape[0], current_dx,
                r0=elem['r0'],
                L0=elem.get('L0', np.inf),
                l0=elem.get('l0', 0.0),
                seed=elem.get('seed'))
            E = E * np.exp(1j * screen)

        elif elem['type'] == 'zernike':
            E = apply_zernike_aberration(E, current_dx,
                                         coefficients=elem['coefficients'],
                                         aperture_radius=elem['aperture_radius'],
                                         dy=current_dy)

        elif elem['type'] == 'gaussian_aperture':
            E = apply_gaussian_aperture(E, current_dx,
                                        sigma=elem['sigma'],
                                        xc=elem.get('xc', 0),
                                        yc=elem.get('yc', 0),
                                        dy=current_dy)

        else:
            raise ValueError(f"Unknown element type: {elem['type']}")

        if verbose:
            intermediates.append(E.copy() if hasattr(E, 'copy') else np.array(E))
        _persist(i + 1, elem.get('type', '?'), E, current_dx)

    call_progress(progress, 'system', 1.0, 'done')
    if return_result:
        from .result import PropagationResult
        return PropagationResult(
            field=E, dx=current_dx, dy=current_dy,
            wavelength=float(wavelength),
            method='system', history=history_entries,
        )
    return E, intermediates


# ----------------------------------------------------------------------
# v4.15 (ROADMAP v4.15 #3): ergonomic prescription -> result entry
# ----------------------------------------------------------------------

def evaluate(
    prescription: Dict[str, Any],
    source: 'Source',
    *,
    output_grid: Optional[Union[int, Tuple[int, int]]] = None,
    output_dx: Optional[float] = None,
    method: str = 'asm',
    use_gpu: bool = False,
    verbose: bool = False,
    progress: Optional[Callable] = None,
) -> 'PropagationResult':
    """Ergonomic one-call entry: prescription + Source -> PropagationResult.

    v4.15 fills the gap where users loading a ``.zmx`` file currently
    have to (a) call :func:`load_zemax_zmx` to get a prescription,
    (b) hand-decompose the prescription into a
    :func:`propagate_through_system` element list, and (c) build the
    input field separately.  :func:`evaluate` collapses (a)-(c) into a
    single call.

    Parameters
    ----------
    prescription : dict
        A lens prescription dict (the same shape produced by
        :func:`load_zemax_zmx`, :func:`load_zemax_prescription_data_txt`,
        :func:`make_singlet`, :func:`make_doublet`, etc.).  Must
        contain ``'elements'`` + ``'all_thicknesses'`` keys (the Zemax
        loaders' canonical shape) OR ``'surfaces'`` + ``'thicknesses'``
        keys (the :func:`make_*` factory shape -- normalised to the
        Zemax shape internally).
    source : Source
        Input :class:`Source`.  The field, grid spacing, and
        wavelength are read from the Source instance.  No call to
        ``to_source()`` is needed -- the Source dataclass already
        bundles ``E``, ``dx``, ``dy``, and ``wavelength``.
    output_grid : int or (Ny, Nx), optional
        Reserved for future use.  Currently accepts the kwarg without
        resampling so callers can future-proof their code; if you pass
        anything other than ``None`` or ``source.E.shape``, the
        argument is ignored with a ``RuntimeWarning``.
    output_dx : float, optional
        Reserved for future use.  Currently accepts the kwarg without
        resampling.
    method : str, default 'asm'
        Free-space propagation method, forwarded to
        :func:`propagate_through_system`.
    use_gpu, verbose, progress :
        Forwarded to :func:`propagate_through_system`.

    Returns
    -------
    result : PropagationResult
        The exit-plane field plus per-element history.

    Raises
    ------
    ValueError
        If ``source`` is ``None`` or not a :class:`Source` instance;
        if ``prescription`` is missing the keys needed by either
        prescription shape.

    Examples
    --------
    >>> import lumenairy as la
    >>> rx = la.load_zemax_zmx('AC254-200-C.zmx')
    >>> src = la.Source.gaussian(
    ...     N=512, dx=10e-6, wavelength=632.8e-9, w0=5e-3)
    >>> result = la.evaluate(rx, src)
    >>> result.field.shape
    (512, 512)
    """
    # Lazy import to avoid a top-level circular dependency on
    # ``lumenairy.sources``.
    from ..sources.core import Source as _SourceCls
    from .result import PropagationResult

    if source is None:
        raise ValueError(
            "lumenairy.propagators.system.evaluate: 'source' is required; got None.  "
            "Pass a Source instance (e.g. ``la.Source.gaussian(N=256, "
            "dx=10e-6, wavelength=633e-9, w0=2e-3)``).")
    if not isinstance(source, _SourceCls):
        raise ValueError(
            "lumenairy.propagators.system.evaluate: 'source' must be a "
            f"lumenairy.Source instance; got {type(source).__name__}.")
    if not isinstance(prescription, dict):
        raise ValueError(
            "lumenairy.propagators.system.evaluate: 'prescription' must be a dict; "
            f"got {type(prescription).__name__}.")
    if output_grid is not None:
        # Future-proofed; currently a no-op.  Emit a soft warning so
        # the caller knows the kwarg is accepted but inert.
        out_shape = tuple(source.E.shape[-2:])
        target_shape = (
            (output_grid, output_grid)
            if isinstance(output_grid, (int, np.integer))
                and not isinstance(output_grid, bool)
            else tuple(output_grid))
        if target_shape != out_shape:
            warnings.warn(
                "lumenairy.propagators.system.evaluate: output_grid resampling is "
                "reserved for a future release; the kwarg is accepted "
                "but currently does not resample the exit-plane field.  "
                f"Got output_grid={target_shape!r}, source.E.shape="
                f"{out_shape!r}.",
                RuntimeWarning, stacklevel=2,
            )
    if output_dx is not None and float(output_dx) != float(source.dx):
        warnings.warn(
            "lumenairy.propagators.system.evaluate: output_dx resampling is "
            "reserved for a future release; the kwarg is accepted but "
            "currently does not resample the exit-plane field.",
            RuntimeWarning, stacklevel=2,
        )

    # Build the element list from the prescription.  Accept both
    # shapes seen across the loader / factory family:
    #
    #   1. Zemax-loader shape: ``elements`` + ``all_thicknesses``
    #      (what :func:`load_zemax_zmx` returns).  This is the shape
    #      that :func:`io.codegen._decompose_prescription` was
    #      designed for, so we route it through that helper.
    #
    #   2. Factory shape: ``surfaces`` + ``thicknesses``
    #      (what :func:`make_singlet`, :func:`make_doublet` return).
    #      We wrap it as a one-group ``real_lens`` element so the
    #      rest of the pipeline is uniform.
    elements_list = _prescription_to_elements(prescription)

    # Route through propagate_through_system with return_result=True so
    # the caller gets a structured PropagationResult back.
    result = propagate_through_system(
        E_in=source.E,
        elements=elements_list,
        wavelength=float(source.wavelength),
        dx=float(source.dx),
        dy=float(source.dy) if source.dy is not None else None,
        method=method,
        use_gpu=use_gpu,
        verbose=verbose,
        progress=progress,
        return_result=True,
    )
    return result


def _prescription_to_elements(
    prescription: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert a prescription dict into a ``propagate_through_system``
    element list.

    Accepts two prescription shapes:

    1. Zemax-loader shape -- ``elements`` + ``all_thicknesses`` keys --
       routed through :func:`io.codegen._decompose_prescription`.
    2. Factory shape -- ``surfaces`` + ``thicknesses`` keys -- wrapped
       as a single ``{'type': 'real_lens', 'prescription': ...}``
       element with no free-space gap (the caller can prepend / append
       ``{'type': 'propagate', 'z': ...}`` elements via
       :func:`propagate_through_system` directly if needed).
    """
    has_elements = 'elements' in prescription and \
        'all_thicknesses' in prescription
    has_surfaces = 'surfaces' in prescription and \
        'thicknesses' in prescription

    if not (has_elements or has_surfaces):
        raise ValueError(
            "lumenairy.propagators.system.evaluate: prescription must contain "
            "either ``'elements'`` + ``'all_thicknesses'`` (Zemax-loader "
            "shape) or ``'surfaces'`` + ``'thicknesses'`` (make_singlet "
            "/ make_doublet shape); got keys "
            f"{sorted(prescription.keys())}.")

    # v4.15.1 (P1-F1-6 / Agent E): the two prescription shapes are
    # mutually exclusive.  Pre-v4.15.1 a dict carrying BOTH
    # ``surfaces``+``thicknesses`` AND ``elements``+``all_thicknesses``
    # silently routed through the Zemax branch with no warning -- a
    # genuine user error (e.g. starting from make_singlet output then
    # hand-grafting an ``elements`` list on top) had no diagnostic and
    # produced wrong-system propagation.  Make the precedence explicit:
    # if both shapes are present, raise ValueError.  Callers wanting
    # the Zemax shape should strip the ``surfaces`` / ``thicknesses``
    # keys; callers wanting the factory shape should strip the
    # ``elements`` / ``all_thicknesses`` keys.
    if has_elements and has_surfaces:
        raise ValueError(
            "lumenairy.propagators.system.evaluate: prescription contains BOTH "
            "``'elements'`` + ``'all_thicknesses'`` (Zemax-loader shape) "
            "AND ``'surfaces'`` + ``'thicknesses'`` (factory shape).  "
            "These two shapes are mutually exclusive -- mixing them is "
            "ambiguous (which shape should drive the propagation?).  "
            "Pick one: for the Zemax shape, drop ``'surfaces'`` / "
            "``'thicknesses'`` from the dict; for the factory shape, "
            "drop ``'elements'`` / ``'all_thicknesses'``.  "
            f"Got keys {sorted(prescription.keys())}.")

    elements_list: List[Dict[str, Any]] = []

    if has_elements:
        # Zemax loader shape -- decompose into propagate / real_lens /
        # mirror / aperture steps.
        from ..io.codegen import _decompose_prescription
        steps = _decompose_prescription(prescription)
        for step in steps:
            stype = step['type']
            if stype == 'propagate':
                elements_list.append({'type': 'propagate',
                                       'z': step['z']})
            elif stype == 'real_lens':
                elements_list.append({
                    'type': 'real_lens',
                    'prescription': step['prescription'],
                })
            elif stype == 'mirror':
                elements_list.append({
                    'type': 'mirror',
                    'radius': step.get('radius'),
                    'conic': step.get('conic', 0.0),
                    'aperture_diameter': step.get('aperture_diameter'),
                })
            elif stype == 'aperture':
                D = step.get('diameter')
                elements_list.append({
                    'type': 'aperture',
                    'shape': 'circular',
                    'params': {'diameter': D},
                })
            elif stype == 'doe_placeholder':
                # Air-to-air aspheric / DOE surfaces -- no element-handler
                # currently exists for raw DOE phase, so skip silently.
                # Future versions can plug a DOE handler here.
                continue
            else:
                raise ValueError(
                    "lumenairy.propagators.system.evaluate: unknown decomposition "
                    f"step type {stype!r}.")
    else:
        # Factory shape -- single real_lens element with the prescription
        # as-is.  No leading / trailing free-space gaps; the caller
        # threads those in by chaining :func:`propagate_through_system`
        # calls or by augmenting the prescription dict.
        elements_list.append({
            'type': 'real_lens',
            'prescription': prescription,
        })

    return elements_list


# ----------------------------------------------------------------------
# v4.12 perf: jit cache for propagate_through_system_jax
# ----------------------------------------------------------------------
#
# The hot path is the sequence of pure-JAX element handlers
# (``propagate`` / ``lens`` / ``aperture`` / ``mask``).  When every
# element in ``elements`` is one of these AND there are no callable /
# array-bearing dict values that would defeat hashing, we cache a
# jit-compiled kernel keyed on the static signature (etype, scalar
# params, mask shapes).  Mask DATA is threaded in as a positional JAX
# arg so different mask arrays don't trigger re-tracing -- only mask
# shape changes do.  Other element types (``spherical_lens``,
# ``aspheric_lens``, ``real_lens``, ``mirror``, etc.) still fall back
# to the per-call NumPy boundary.

#
# v4.12.2: converted to an LRU-bounded ``OrderedDict`` (was an unbounded
# plain dict) so long-running design sweeps over element lists do not
# leak compiled XLA executables.  Accessed keys are moved to the end;
# when ``len > _PROPAGATE_SYSTEM_JAX_CACHE_MAXSIZE`` the oldest entry
# is evicted.
_PROPAGATE_SYSTEM_JAX_CACHE: 'OrderedDict[Any, Any]' = OrderedDict()
_PROPAGATE_SYSTEM_JAX_CACHE_MAXSIZE = 32
# v4.14.2 (P1-NEW-2 / Agent C): thread-safety lock for
# ``_PROPAGATE_SYSTEM_JAX_CACHE``.  Without this two threads racing
# through :func:`propagate_through_system_jax` could see a torn
# OrderedDict (``get`` -> ``__setitem__`` -> ``popitem`` is a
# read-modify-write sequence).  Follows the ``_ASM_CACHE_LOCK``
# precedent in :mod:`propagators.propagation`.  Lock-scope discipline:
# the jit-compile step (``_make_system_jax_kernel``) is expensive
# (10-100ms+ XLA trace) and runs OUTSIDE the lock so a concurrent
# cache hit on a different key isn't blocked.
_PROPAGATE_SYSTEM_JAX_CACHE_LOCK = threading.Lock()


def clear_propagate_system_jax_cache() -> None:
    """Drop every cached jit'd ``propagate_through_system_jax`` kernel (v4.12.2).

    Forces the next :func:`propagate_through_system_jax` fast-path call
    to rebuild and re-cache its jit-compiled kernel from scratch.  Use
    in unit tests that pin cache mechanics or in long-running pipelines
    that want to release compiled XLA executables.
    """
    with _PROPAGATE_SYSTEM_JAX_CACHE_LOCK:
        _PROPAGATE_SYSTEM_JAX_CACHE.clear()


# v4.16.0 (ROADMAP #15): register the propagate-system JAX clearer
# with the central registry at module-import time.
# ``clear_asm_caches`` now walks the registry rather than enumerating
# clear calls by hand.  Late-binding closure preserves
# ``mock.patch.object`` test semantic.
try:
    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    import sys as _sys
    _this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        'propagate_system_jax',
        lambda: getattr(_this_mod, 'clear_propagate_system_jax_cache')(),
    )
except ImportError:
    pass


# ----------------------------------------------------------------------
# Traceable element types for propagate_through_system_jax (B1-2 fix)
# ----------------------------------------------------------------------
#
# These are the only element types that ``propagate_through_system_jax``
# can route fully through JAX (i.e. through jit-compiled / grad-able
# code paths).  Any element not in this set forces a NumPy boundary
# (``np.asarray(E)`` on the field), which raises
# ``TracerArrayConversionError`` under ``jax.jit`` / ``jax.grad`` and
# breaks end-to-end traceability.
#
# Programmatic users can read this set to decide whether an element
# list is JAX-traceable before calling.
_TRACEABLE_ELEMENT_TYPES = frozenset({
    'propagate',     # angular_spectrum_propagate (JAX backend)
    'lens',          # thin-lens phase screen
    'aperture',      # boolean mask multiplication
    'mask',          # arbitrary mask multiplication
})


# v5.0 (audit_v4_13_1 Part 5 / honest break): the legacy pre-v4.12 JAX
# aperture schema (``radius`` / ``half_width_x`` / ``inner_radius``)
# emitted a ``DeprecationWarning`` from v4.12 through v4.16.3.  v5.0
# removes the legacy branches; the canonical NumPy schema (``diameter``
# / ``width_x`` / ``inner_diameter``) is now the ONLY accepted form.
# Migration: ``radius=r`` -> ``diameter=2*r``; ``inner_radius=ri`` ->
# ``inner_diameter=2*ri``; ``half_width_x=hx`` -> ``width_x=2*hx``.
# See Migration-Guide.md section 5.0.0 for the recipe.


def _resolve_aperture_params(elem: Dict[str, Any]
                              ) -> Optional[Tuple[str, Tuple[float, ...]]]:
    """Resolve an aperture element to canonical JAX-kernel params.

    Accepts the canonical NumPy schema (matches ``apply_aperture``):
    ``params={'diameter': D}`` for circular,
    ``params={'inner_diameter': Di, 'outer_diameter': Do}`` for
    annular, ``params={'width_x': Wx, 'width_y': Wy}`` for
    rectangular.

    The pre-v4.12 JAX-only schema (``radius`` / ``half_width_x`` /
    ``inner_radius``) was deprecated in v4.12 and is **removed in v5.0**.
    Pass the canonical NumPy schema; the legacy keys raise
    ``ValueError`` with the migration recipe.

    Returns
    -------
    (shape, halves) : tuple
        ``shape`` is ``'circular'`` / ``'rectangular'`` / ``'annular'``;
        ``halves`` is a tuple of half-extents (radius / half-widths)
        ready to feed the JAX kernel.  Returns ``None`` if the aperture
        shape is unsupported or required params are missing.
    """
    shape = elem.get('shape', 'circular')
    params = elem.get('params') or {}

    def _reject_legacy(legacy_key: str, canonical_msg: str) -> None:
        raise ValueError(
            f"propagate_through_system_jax aperture element used the "
            f"legacy pre-v4.12 JAX-only {legacy_key!r} schema.  This "
            f"schema was deprecated in v4.12 and removed in v5.0.  "
            f"Migrate to the canonical NumPy schema: {canonical_msg}.  "
            f"See Migration-Guide.md section 5.0.0 for details."
        )

    if shape == 'circular':
        D = params.get('diameter')
        if D is not None:
            return ('circular', (float(D) / 2.0,))
        # Legacy schema -> hard error.
        if 'radius' in params or 'radius' in elem:
            _reject_legacy('radius', "params={'diameter': 2*radius}")
        return None

    if shape == 'rectangular':
        Wx = params.get('width_x')
        Wy = params.get('width_y')
        if Wx is not None or Wy is not None:
            hx = float(Wx) / 2.0 if Wx is not None else float('inf')
            hy = float(Wy) / 2.0 if Wy is not None else float('inf')
            return ('rectangular', (hx, hy))
        if ('half_width_x' in params or 'half_width_y' in params
                or 'width' in params or 'height' in params):
            _reject_legacy(
                'half_width_x/half_width_y/width/height',
                "params={'width_x': 2*half_width_x, "
                "'width_y': 2*half_width_y}")
        return None

    if shape == 'annular':
        Di = params.get('inner_diameter')
        Do = params.get('outer_diameter')
        if Di is not None or Do is not None:
            r_i = float(Di) / 2.0 if Di is not None else 0.0
            r_o = float(Do) / 2.0 if Do is not None else float('inf')
            return ('annular', (r_o, r_i))
        if 'inner_radius' in params or 'outer_radius' in params:
            _reject_legacy(
                'inner_radius/outer_radius',
                "params={'inner_diameter': 2*inner_radius, "
                "'outer_diameter': 2*outer_radius}")
        # v5.0.1 (audit P3-NEW-F1-3): unreachable post-_reject_legacy
        # tuple-return block removed; ``_reject_legacy`` always raises.
        return None

    return None


def _system_element_signature(elem: Dict[str, Any]) -> Optional[Tuple]:
    """Return a hashable static signature for a JAX-compatible element,
    or None if the element should bypass the jit cache (falls back to
    the per-call Python branch / NumPy boundary).
    """
    etype = elem.get('type', '')
    if etype == 'propagate':
        return ('propagate', float(elem['z']),
                bool(elem.get('bandlimit', True)))
    if etype == 'lens':
        return ('lens', float(elem['f']),
                float(elem.get('xc', 0.0)),
                float(elem.get('yc', 0.0)))
    if etype == 'aperture':
        xc = float(elem.get('xc', 0.0))
        yc = float(elem.get('yc', 0.0))
        resolved = _resolve_aperture_params(elem)
        if resolved is None:
            return None
        shape, halves = resolved
        if shape == 'circular':
            return ('aperture_circular', halves[0], xc, yc)
        if shape == 'rectangular':
            return ('aperture_rect', halves[0], halves[1], xc, yc)
        if shape == 'annular':
            return ('aperture_annular', halves[0], halves[1], xc, yc)
        return None
    if etype == 'mask':
        m = elem.get('mask')
        if m is None:
            return None
        shape = tuple(np.asarray(m).shape)
        dtype = str(np.asarray(m).dtype)
        return ('mask', shape, dtype)
    # spherical_lens / aspheric_lens / real_lens / mirror / etc.
    return None


def _make_system_jax_kernel(elem_sigs, wavelength, dx, dy):
    """Build a jit'd kernel for one static element sequence.

    ``elem_sigs`` is a tuple of element signatures (from
    :func:`_system_element_signature`).  Mask elements are passed as
    additional positional JAX arrays to the returned closure.
    """
    import jax
    import jax.numpy as jnp

    mask_indices = [i for i, sig in enumerate(elem_sigs) if sig[0] == 'mask']

    def _kernel(E, *mask_arrays):
        Ny, Nx = E.shape[-2:]
        x = (jnp.arange(Nx) - Nx / 2) * dx
        y = (jnp.arange(Ny) - Ny / 2) * dy
        X, Y = jnp.meshgrid(x, y, indexing='xy')
        k0 = 2.0 * jnp.pi / wavelength
        mask_iter = iter(mask_arrays)
        for sig in elem_sigs:
            tag = sig[0]
            if tag == 'propagate':
                _, z, bandlimit = sig
                from .propagation import angular_spectrum_propagate
                E = angular_spectrum_propagate(
                    E, z, wavelength, dx, bandlimit=bandlimit)
            elif tag == 'lens':
                _, f, xc, yc = sig
                r2 = (X - xc) ** 2 + (Y - yc) ** 2
                E = E * jnp.exp(-1j * k0 * r2 / (2.0 * f))
            elif tag == 'aperture_circular':
                _, r, xc, yc = sig
                mask = ((X - xc) ** 2 + (Y - yc) ** 2) <= r * r
                E = E * mask.astype(E.dtype)
            elif tag == 'aperture_rect':
                _, hx, hy, xc, yc = sig
                mask = (jnp.abs(X - xc) <= hx) & (jnp.abs(Y - yc) <= hy)
                E = E * mask.astype(E.dtype)
            elif tag == 'aperture_annular':
                _, r_o, r_i, xc, yc = sig
                rho2 = (X - xc) ** 2 + (Y - yc) ** 2
                mask = (rho2 <= r_o * r_o) & (rho2 >= r_i * r_i)
                E = E * mask.astype(E.dtype)
            elif tag == 'mask':
                m = next(mask_iter)
                E = E * m
        return E

    return jax.jit(_kernel)


def propagate_through_system_jax(E_in: np.ndarray,
                                 elements: Sequence[Dict[str, Any]],
                                 wavelength: float,
                                 dx: float,
                                 dy: Optional[float] = None,
                                 method: str = 'asm',
                                 verbose: bool = False) -> Any:
    """JAX-traceable variant of :func:`propagate_through_system`.

    Element-by-element walk where each element type is dispatched to a
    JAX-compatible handler.

    Supported (traceable) element types
    -----------------------------------
    The set of element types with a fully JAX-traceable code path is
    exposed as the module-level constant
    :data:`lumenairy.propagators.system._TRACEABLE_ELEMENT_TYPES`:

      * ``'propagate'``  -> ``angular_spectrum_propagate`` (JAX backend)
      * ``'lens'``       -> paraxial thin-lens phase screen
      * ``'aperture'``   -> hard boolean mask multiplication (circular /
        rectangular / annular).  Uses the same canonical NumPy schema
        as :func:`apply_aperture` (``params={'diameter': ...}`` etc.);
        the pre-v4.12 JAX-only schema (``params={'radius': ...}``) is
        still accepted with a one-shot ``DeprecationWarning``.  See
        :func:`_resolve_aperture_params`.
      * ``'mask'``       -> phase / amplitude mask multiplication

    Unsupported element types
    -------------------------
    Element types NOT in :data:`_TRACEABLE_ELEMENT_TYPES`
    (``spherical_lens``, ``aspheric_lens``, ``real_lens``,
    ``real_lens_traced``, ``mirror``, ``cylindrical_lens``, ``axicon``,
    ``grin_lens``, ``propagate_tilted``, ``turbulence``, ``zernike``,
    ``gaussian_aperture``, ...) DO NOT have a JAX handler.  Pre-v4.12
    this function silently fell back to NumPy at the element boundary
    (``np.asarray(E)`` then back to ``jnp.asarray``), which works for
    eager calls but raises ``TracerArrayConversionError`` under
    ``jax.jit`` / ``jax.grad`` -- i.e. the function was NOT actually
    JAX-end-to-end-traceable as advertised.

    v4.12 fail-fast: if any element in ``elements`` is not in
    :data:`_TRACEABLE_ELEMENT_TYPES`, this function now raises
    :class:`NotImplementedError` at call time with the list of
    offending element types.  Use :func:`propagate_through_system`
    (NumPy) for those element types, or implement a JAX handler.

    Performance
    -----------
    When every element is traceable, the full chain is cached as one
    compiled XLA graph keyed on the sequence of
    ``(etype, scalar params, mask shapes/dtypes)``.  Repeated calls
    with the same element layout reuse the cached executable.

    Returns
    -------
    E_out : JAX array, complex
        Field after the full element chain.

    Raises
    ------
    NotImplementedError
        If any element in ``elements`` has a ``'type'`` not in
        :data:`_TRACEABLE_ELEMENT_TYPES`.
    ImportError
        If JAX is not installed.
    """
    # v4.15.4 (P1-NEW-3WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  v4.15.3 scoped the walker
    # to ``propagators/`` + ``elements/`` only; the JAX sibling of
    # the (already-guarded) ``propagate_through_system`` in the SAME
    # file at :47 was missed.  Runs FIRST so PartialCoherenceMCF /
    # 3-D ensemble inputs get a clear v4.16-roadmap message rather
    # than a confusing ``TracerArrayConversionError`` or silent
    # wrong-axis broadcast at the ``jnp.asarray(E_in, ...)`` cast
    # below.  ``E_in`` is contractually a numpy array at the entry
    # point (it is wrapped via ``jnp.asarray`` further down), so
    # the helper's ``getattr(E, 'ndim', None)`` path is well-defined.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'propagate_through_system_jax')

    from ..backend import JAX_AVAILABLE
    if not JAX_AVAILABLE:
        raise ImportError(
            'JAX is not installed; install with `pip install jax` or '
            'use propagate_through_system() (NumPy).')
    import jax.numpy as jnp
    from .propagation import angular_spectrum_propagate

    if dy is None:
        dy = dx

    # ------------------------------------------------------------------
    # B1-2 fix: fail-fast on non-traceable elements.
    # ------------------------------------------------------------------
    # Pre-v4.12 the slow path tried to ``np.asarray(E)`` on a (possibly
    # traced) JAX array, which raises ``TracerArrayConversionError``
    # under ``jax.jit`` / ``jax.grad``.  Rather than leak that confusing
    # JAX-internal traceback, scan the element types up front and raise
    # an explicit, actionable error listing exactly which element types
    # have no JAX path.
    bad_types: List[str] = []
    seen: set = set()
    for elem in elements:
        etype = elem.get('type', '')
        if etype not in _TRACEABLE_ELEMENT_TYPES and etype not in seen:
            seen.add(etype)
            bad_types.append(etype)
    if bad_types:
        raise NotImplementedError(
            "propagate_through_system_jax: element type(s) "
            f"{bad_types!r} have no JAX-traceable handler.  Supported "
            f"types are {sorted(_TRACEABLE_ELEMENT_TYPES)}.  Use "
            "lumenairy.propagators.system.propagate_through_system() (NumPy) for "
            "an element list containing these types, or check "
            "lumenairy.propagators.system._TRACEABLE_ELEMENT_TYPES to filter your "
            "element list programmatically before calling."
        )

    # v4.13.0 (audit L2): pre-fix this hard-cast to ``jnp.complex64``
    # silently overrode ``set_default_complex_dtype(np.complex128)``.
    # Now goes through ``_resolve_jax_complex_dtype`` which reads the
    # library-wide default.  Complex inputs honour their own dtype; only
    # real inputs fall back to the configured default.
    #
    # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 7 / P3 / DEEP-4 MEDIUM-2):
    # the pre-v4.16.1 dtype probe used ``np.iscomplexobj(np.asarray(E_in))``
    # which raises ``jax.errors.TracerArrayConversionError`` when this
    # function is wrapped in ``jax.jit`` or ``jax.grad`` -- but this
    # function is explicitly named ``propagate_through_system_jax`` and
    # the docstring touts end-to-end jit'd caching.  The fix mirrors
    # the v4.15.3 ``_check_2d_scalar_field`` JAX-safe ``getattr(E,
    # 'attr', None)`` idiom: read the dtype via duck-typing (works for
    # NumPy arrays, JAX tracers, CuPy arrays, and the typed scalars
    # that the JAX cast accepts).  When the dtype attribute is missing
    # or non-complex, fall back to the library-default complex dtype
    # for the cast -- the real-input branch is exercised by tests that
    # build a real Gaussian envelope and rely on the autopromote.
    probe_dtype = getattr(E_in, 'dtype', None)
    if probe_dtype is not None and 'complex' in str(probe_dtype):
        cdtype = _resolve_jax_complex_dtype(probe_dtype)
    else:
        cdtype = _resolve_jax_complex_dtype()
    E = jnp.asarray(E_in, dtype=cdtype)

    # ------------------------------------------------------------------
    # Fast path: all elements have a static signature -> single jit'd
    # kernel for the whole chain.
    # ------------------------------------------------------------------
    elem_sigs = [_system_element_signature(elem) for elem in elements]
    if all(sig is not None for sig in elem_sigs) and not verbose:
        sigs_tuple = tuple(elem_sigs)
        # v4.13.0 (audit L2): include dtype in the cache key so calls
        # at complex64 and complex128 don't share the same compiled XLA
        # kernel.  Pre-fix the cache key omitted dtype entirely, so the
        # first call to win the race fixed the kernel precision for
        # every subsequent call regardless of the active default.
        cache_key = (sigs_tuple, float(wavelength), float(dx), float(dy),
                     str(np.dtype(cdtype)))
        with _PROPAGATE_SYSTEM_JAX_CACHE_LOCK:
            kernel = _PROPAGATE_SYSTEM_JAX_CACHE.get(cache_key)
            if kernel is not None:
                # LRU touch: keep recently-used kernels resident.
                _PROPAGATE_SYSTEM_JAX_CACHE.move_to_end(cache_key)
        if kernel is None:
            # Build OUTSIDE the lock -- jit tracing is the expensive
            # step (XLA compile) and holding the lock here would
            # serialise every concurrent caller even on cache hits at
            # different keys.  Two threads may double-build for the
            # same cold key -- benign waste; the second insert just
            # overwrites the first.
            kernel = _make_system_jax_kernel(
                sigs_tuple, float(wavelength), float(dx), float(dy))
            with _PROPAGATE_SYSTEM_JAX_CACHE_LOCK:
                _PROPAGATE_SYSTEM_JAX_CACHE[cache_key] = kernel
                while (len(_PROPAGATE_SYSTEM_JAX_CACHE)
                       > _PROPAGATE_SYSTEM_JAX_CACHE_MAXSIZE):
                    _PROPAGATE_SYSTEM_JAX_CACHE.popitem(last=False)
        mask_arrays = tuple(
            jnp.asarray(elem['mask'])
            for elem, sig in zip(elements, elem_sigs)
            if sig[0] == 'mask'
        )
        return kernel(E, *mask_arrays)

    # ------------------------------------------------------------------
    # Slow path: at least one element falls back to NumPy.  Keep the
    # legacy per-call Python branch loop so unsupported elements still
    # work.
    # ------------------------------------------------------------------
    Ny, Nx = E.shape[-2:]
    x = (jnp.arange(Nx) - Nx / 2) * dx
    y = (jnp.arange(Ny) - Ny / 2) * dy
    X, Y = jnp.meshgrid(x, y, indexing='xy')
    k0 = 2.0 * jnp.pi / wavelength

    for i, elem in enumerate(elements):
        etype = elem.get('type', '')
        if verbose:
            print(f'  Stage {i+1}/{len(elements)}: {etype}', flush=True)

        if etype == 'propagate':
            z = elem['z']
            bandlimit = elem.get('bandlimit', True)
            E = angular_spectrum_propagate(
                E, float(z), wavelength, dx, bandlimit=bandlimit)

        elif etype == 'lens':
            f = elem['f']
            xc = elem.get('xc', 0.0)
            yc = elem.get('yc', 0.0)
            r2 = (X - xc) ** 2 + (Y - yc) ** 2
            E = E * jnp.exp(-1j * k0 * r2 / (2.0 * f))

        elif etype == 'aperture':
            # Resolve to canonical half-extents from the NumPy schema
            # (diameter / width_x / inner_diameter).  v5.0: the legacy
            # JAX-only schema (radius / half_width_x / inner_radius)
            # was removed; legacy keys now raise ValueError inside
            # ``_resolve_aperture_params`` with the migration recipe
            # inline.  See Migration-Guide.md §5.0.0.
            xc = elem.get('xc', 0.0)
            yc = elem.get('yc', 0.0)
            dx_loc = X - xc
            dy_loc = Y - yc
            resolved = _resolve_aperture_params(elem)
            if resolved is not None:
                shape, halves = resolved
                if shape == 'circular':
                    r = halves[0]
                    mask = (dx_loc ** 2 + dy_loc ** 2) <= float(r) ** 2
                    E = E * mask.astype(E.dtype)
                elif shape == 'rectangular':
                    hx, hy = halves
                    mask = ((jnp.abs(dx_loc) <= float(hx)) &
                            (jnp.abs(dy_loc) <= float(hy)))
                    E = E * mask.astype(E.dtype)
                elif shape == 'annular':
                    r_o, r_i = halves
                    rho2 = dx_loc ** 2 + dy_loc ** 2
                    mask = ((rho2 <= float(r_o) ** 2) &
                            (rho2 >= float(r_i) ** 2))
                    E = E * mask.astype(E.dtype)
            # No-op silently if aperture params are missing (matches
            # NumPy ``apply_aperture`` default of all-infinity).

        elif etype == 'mask':
            mask = jnp.asarray(elem['mask'])
            E = E * mask

        else:
            # Unreachable: the up-front _TRACEABLE_ELEMENT_TYPES gate
            # raises NotImplementedError for any other type before we
            # get here.  Keep a defensive fallback for forward
            # compatibility (new traceable types added later).
            raise NotImplementedError(
                f"propagate_through_system_jax: element type {etype!r} "
                "reached the per-element dispatch with no handler "
                "(internal error: _TRACEABLE_ELEMENT_TYPES gate "
                "should have caught this)."
            )

    return E

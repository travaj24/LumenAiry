"""
Sequential optical system propagation.

Provides the :func:`propagate_through_system` function which propagates an
electric field through an ordered sequence of optical elements, dispatching
each element to the appropriate physics routine from the library submodules.

Author: Andrew Traverso
"""

import numpy as np

from .propagation import (
    angular_spectrum_propagate,
    angular_spectrum_propagate_tilted,
    fresnel_propagate,
    resample_field,
)
from .lenses import (
    apply_thin_lens,
    apply_spherical_lens,
    apply_aspheric_lens,
    apply_real_lens,
    apply_real_lens_traced,
    apply_cylindrical_lens,
    apply_grin_lens,
    apply_axicon,
)
from .elements import (
    apply_mirror,
    apply_aperture,
    apply_gaussian_aperture,
    apply_mask,
    apply_zernike_aberration,
    generate_turbulence_screen,
)


def propagate_through_system(E_in, elements, wavelength, dx, dy=None,
                             method='asm', use_gpu=False, verbose=False,
                             progress=None):
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
    from .progress import call_progress, ProgressScaler
    E = E_in.copy() if hasattr(E_in, 'copy') else np.array(E_in)
    intermediates = []
    current_dx = dx
    current_dy = dy if dy is not None else dx

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
                E, dx_new, dy_new = fresnel_propagate(
                    E, z, wavelength, current_dx, current_dy)
                # Resample back to the original grid spacing so
                # downstream element phases (lenses, apertures) are
                # computed on the correct coordinate system.
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
                E, dx_new, dy_new = scalable_angular_spectrum_propagate(
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
            E = apply_thin_lens(E, f, wavelength, dx, dy, xc, yc,
                                use_gpu=use_gpu, lens_model=model)

        elif elem['type'] == 'spherical_lens':
            E = apply_spherical_lens(E, elem['R1'], elem['R2'], elem['d'],
                                elem['n_lens'], wavelength, dx, dy,
                                aperture_diameter=elem.get('aperture_diameter'),
                                xc=elem.get('xc', 0), yc=elem.get('yc', 0),
                                use_gpu=use_gpu)

        elif elem['type'] == 'aspheric_lens':
            E = apply_aspheric_lens(E, elem['R1'], elem['R2'], elem['d'],
                                elem['n_lens'], wavelength, dx, dy,
                                k1=elem.get('k1', 0), k2=elem.get('k2', 0),
                                A1=elem.get('A1'), A2=elem.get('A2'),
                                aperture_diameter=elem.get('aperture_diameter'),
                                xc=elem.get('xc', 0), yc=elem.get('yc', 0),
                                use_gpu=use_gpu)

        elif elem['type'] == 'real_lens':
            E = apply_real_lens(
                E, elem['prescription'], wavelength, dx,
                bandlimit=elem.get('bandlimit', True),
                slant_correction=elem.get('slant_correction', False),
                fresnel=elem.get('fresnel', False),
                absorption=elem.get('absorption', False),
                progress=(lambda stage, frac, msg='': sub_cb(frac, msg))
                        if progress is not None else None,
            )

        elif elem['type'] == 'real_lens_traced':
            E = apply_real_lens_traced(
                E, elem['prescription'], wavelength, dx,
                bandlimit=elem.get('bandlimit', True),
                ray_subsample=elem.get('ray_subsample', 1),
                progress=(lambda stage, frac, msg='': sub_cb(frac, msg))
                        if progress is not None else None,
            )

        elif elem['type'] == 'mirror':
            E = apply_mirror(E, wavelength, dx,
                             radius=elem.get('radius'),
                             conic=elem.get('conic', 0.0),
                             aperture_diameter=elem.get('aperture_diameter'),
                             xc=elem.get('xc', 0), yc=elem.get('yc', 0))

        elif elem['type'] == 'aperture':
            E = apply_aperture(E, dx,
                               shape=elem.get('shape', 'circular'),
                               params=elem.get('params', {}),
                               xc=elem.get('xc', 0), yc=elem.get('yc', 0))

        elif elem['type'] == 'cylindrical_lens':
            E = apply_cylindrical_lens(E, elem['f'], wavelength, dx, dy,
                                       axis=elem.get('axis', 'x'),
                                       xc=elem.get('xc', 0),
                                       yc=elem.get('yc', 0))

        elif elem['type'] == 'axicon':
            E = apply_axicon(E, elem['alpha'], elem['n_axicon'],
                             wavelength, dx, dy,
                             xc=elem.get('xc', 0), yc=elem.get('yc', 0))

        elif elem['type'] == 'grin_lens':
            E = apply_grin_lens(E, elem['n0'], elem['g'], elem['d'],
                                wavelength, dx, dy,
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
                E.shape[0], dx,
                r0=elem['r0'],
                L0=elem.get('L0', np.inf),
                l0=elem.get('l0', 0.0),
                seed=elem.get('seed'))
            E = E * np.exp(1j * screen)

        elif elem['type'] == 'zernike':
            E = apply_zernike_aberration(E, dx,
                                         coefficients=elem['coefficients'],
                                         aperture_radius=elem['aperture_radius'])

        elif elem['type'] == 'gaussian_aperture':
            E = apply_gaussian_aperture(E, dx,
                                        sigma=elem['sigma'],
                                        xc=elem.get('xc', 0),
                                        yc=elem.get('yc', 0))

        else:
            raise ValueError(f"Unknown element type: {elem['type']}")

        if verbose:
            intermediates.append(E.copy() if hasattr(E, 'copy') else np.array(E))

    call_progress(progress, 'system', 1.0, 'done')
    return E, intermediates

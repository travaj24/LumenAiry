"""
Sequential optical system propagation.

Provides the :func:`propagate_through_system` function which propagates an
electric field through an ordered sequence of optical elements, dispatching
each element to the appropriate physics routine from the library submodules.

Author: Andrew Traverso
"""

# Version history for this module: ``docs/history/lumenairy.propagators.system.md``.

from __future__ import annotations

import threading
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..elements._lens_kernels import caller_stacklevel as _caller_stacklevel

if TYPE_CHECKING:
    # v5.0.1 (audit P1-NEW-V4-1): TYPE_CHECKING guard for forward references
    # to ``Source`` and ``PropagationResult`` in ``evaluate(...)``.  Both
    # are imported lazily inside the function body to avoid top-level
    # circular dependencies; ruff F821 still needs a typing-time binding
    # so the string-quoted annotations resolve.
    from ..sources import Source
    from .result import PropagationResult

from ..elements import (
    apply_aperture,
    apply_gaussian_aperture,
    apply_mask,
    apply_mirror,
    apply_zernike_aberration,
    generate_turbulence_screen,
)
from ..elements.elements import _validate_edge_kwargs
from ..elements.lenses import (
    apply_aspheric_lens,
    apply_axicon,
    apply_cylindrical_lens,
    apply_grin_lens,
    apply_real_lens,
    apply_real_lens_traced,
    apply_spherical_lens,
    apply_thin_lens,
)
from .propagation import (
    _resolve_jax_complex_dtype,
    angular_spectrum_propagate,
    angular_spectrum_propagate_tilted,
    fresnel_propagate_mft,
    resample_field,
)


def _apply_rcwa_element(E, elem):
    """Apply the rigorous zeroth-order (specular) SCALAR amplitude of a
    periodic RCWA element to a scalar field ``E``.

    The element dict is one of::

        {'type': 'rcwa', 'amplitude': complex}                  # explicit
        {'type': 'rcwa', 'result': RCWAResult,                  # from a solve
         'port': 'transmission'|'reflection', 'incident': (Ex, Ey)}

    For the ``result`` form the co-polarised zeroth-order Jones amplitude
    (the incident vector propagated through the 2x2 Jones and projected back
    onto itself) is used as a uniform complex multiplier ``E -> amplitude*E``.

    This is the SCALAR-pipeline bridge: it carries only the specular order for
    one polarization.  Polarization-resolved and non-zero-order composition
    uses the JonesField chain instead --
    :meth:`~lumenairy.elements.rcwa.RCWAResult.to_jones_field` /
    :meth:`~lumenairy.elements.rcwa.RCWAResult.to_multiorder_field` ->
    ``JonesField.propagate``.
    """
    amp = elem.get('amplitude')
    if amp is None:
        result = elem.get('result')
        if result is None:
            raise ValueError(
                "propagate_through_system: an 'rcwa' element needs either "
                "'amplitude' (complex) or 'result' (an RCWAResult).")
        port = elem.get('port', 'transmission')
        if port not in ('transmission', 'reflection'):
            raise ValueError(
                f"propagate_through_system: 'rcwa' element port must be "
                f"'transmission' or 'reflection', got {port!r}.")
        incident = elem.get('incident', (1.0, 0.0))
        from ..backend import to_numpy
        J = to_numpy(result.jones_transmission() if port == 'transmission'
                     else result.jones_reflection())
        inc = np.asarray(incident, dtype=complex).reshape(2)
        out = J @ inc                       # specular [Ex, Ey] response
        norm = np.vdot(inc, inc)
        amp = complex(np.vdot(inc, out) / (norm if norm != 0 else 1.0))
    return E * complex(amp)


def _apply_coating_element(E, elem, system_wavelength=None):
    """Apply a multilayer thin-film coating as a uniform complex amplitude on a
    scalar field ``E``.

    The element dict is::

        {'type': 'coating', 'layers': [(n, thickness_m), ...],
         'wavelength': float,                  # defaults to the system wavelength
         'angle': float (rad, default 0),
         'n_substrate': float (default 1.52),
         'n_ambient': float (default 1.0),
         'polarization': 's'|'p'|'avg' (default 'avg'),
         'port': 'reflection'|'transmission' (default 'transmission')}

    The coating's specular response at the single (wavelength, angle) point is
    used as the uniform multiplier ``E -> amp*E``.  For the ``'reflection'``
    port the complex coefficient ``sqrt(R)*exp(i*phase_r)`` carries the
    interference phase; for ``'transmission'`` only the amplitude
    ``sqrt(T)`` is applied (a lossless coating's transmitted phase is not
    returned by :func:`~lumenairy.elements.coatings.coating_reflectance`).

    This mirrors the scalar ``'rcwa'`` bridge: it carries one polarization at
    one angle.  Spectral or angular sweeps call
    :func:`~lumenairy.elements.coatings.coating_reflectance` directly.
    """
    from ..elements.coatings import coating_reflectance
    layers = elem.get('layers')
    if layers is None:
        raise ValueError(
            "propagate_through_system: a 'coating' element needs 'layers' "
            "(a list of (n, thickness_m) tuples).")
    wl = elem.get('wavelength', system_wavelength)
    if wl is None:
        raise ValueError(
            "propagate_through_system: a 'coating' element needs 'wavelength'.")
    port = elem.get('port', 'transmission')
    if port not in ('transmission', 'reflection'):
        raise ValueError(
            f"propagate_through_system: 'coating' element port must be "
            f"'transmission' or 'reflection', got {port!r}.")
    R, T, phase_r = coating_reflectance(
        layers, wl,
        angle=elem.get('angle', 0.0),
        n_substrate=elem.get('n_substrate', 1.52),
        n_ambient=elem.get('n_ambient', 1.0),
        polarization=elem.get('polarization', 'avg'))
    R = float(np.atleast_1d(R)[0])
    T = float(np.atleast_1d(T)[0])
    phase_r = float(np.atleast_1d(phase_r)[0])
    if port == 'reflection':
        amp = np.sqrt(R) * np.exp(1j * phase_r)
    else:
        amp = np.sqrt(T)
    return E * complex(amp)


def _require_square_pitch(current_dx: float, current_dy: float,
                          branch: str) -> None:
    """SY-2 guard for the pitch-CHANGING chain branches.

    The ``'fresnel'`` leg lands the integral on a SQUARE output grid
    (one ``N_out``, ``dy_out = dx_out``), so an anamorphic input pitch
    would be read as if ``dy == dx`` on the way out;
    ``scalable_angular_spectrum_propagate`` and
    ``generate_turbulence_screen`` take a single pitch and assume a
    square grid.  For an anamorphic working pitch (``current_dy !=
    current_dx``) these branches would silently produce wrong physics on
    the y-axis, so we refuse rather than mislead.  Square-pitch chains
    (the common case) pass through untouched.  ASM / GBD / HF / mask /
    lens / aperture elements all thread ``current_dy`` correctly and are
    unaffected.
    """
    if abs(float(current_dy) - float(current_dx)) > abs(float(current_dx)) * 1e-9:
        raise ValueError(
            f"propagate_through_system: the {branch!r} step assumes a square "
            f"grid pitch, but the working pitch is anamorphic "
            f"(dx={current_dx:.6g} m, dy={current_dy:.6g} m).  Use a "
            f"'propagate' (ASM) / 'gbd' / 'hf' step -- which thread the "
            f"y-pitch correctly -- or resample to an isotropic grid before "
            f"this element.")


# v5.30: the free-space kernels the ``'propagate'`` element step actually
# implements.  ``'asm'`` is the default (and the only one that honours the
# ``tilt_x`` / ``tilt_y`` keys, via ``angular_spectrum_propagate_tilted``).
_SYSTEM_METHODS = ('asm', 'sas', 'fresnel')
# Recognised elsewhere in the library but deliberately not offered here:
# RS has no resample-back path in this chain (and the default-resolver
# branch above rejects it with a dedicated message).
_SYSTEM_REJECTED_METHODS = ('rs', 'rayleigh_sommerfeld')


def _validate_system_method(method: Any, where: str) -> None:
    """Reject a free-space ``method`` this chain does not implement.

    v5.30 -- the NumPy twin of audit P6 (recorded as a new measured
    finding in ``AUDIT_ADVERSARIAL_CODEBASE_2026_07_25``).  Pre-v5.30 the
    ``'propagate'`` element step ran ``if prop_method == 'fresnel' ...
    elif prop_method == 'sas' ... else: # Default: ASM``, so ANY other
    string -- ``'gbd'``, ``'fraunhofer'``, ``'ASM'`` (wrong case), or
    outright junk -- silently produced the ASM field (measured 0.0
    relative difference vs ``method='asm'``).  Silent fall-through on a
    junk input is the forbidden class, and the JAX twin
    :func:`propagate_through_system_jax` was already fixed to name what it
    implements, so both entry points now raise with the same wording.

    ``where`` names the offending site -- ``'method'`` for the
    entry-point kwarg, ``"elements[i]['method']"`` for a per-element
    override -- so the message points at the key the caller wrote.
    """
    if method in _SYSTEM_METHODS:
        return
    if method in _SYSTEM_REJECTED_METHODS:
        raise ValueError(
            f"propagate_through_system: {where}={method!r} is not "
            f"supported in this entry point.  Supported here: "
            f"{list(_SYSTEM_METHODS)}.  Rayleigh-Sommerfeld has no "
            f"resample-back-to-input-pitch path in the element chain; "
            f"call rayleigh_sommerfeld_propagate() directly for a "
            f"single free-space leg.")
    raise ValueError(
        f"propagate_through_system: {where}={method!r} is not a "
        f"recognised free-space method.  Supported here: "
        f"{list(_SYSTEM_METHODS)}; additionally recognised by the library "
        f"(and rejected here): {list(_SYSTEM_REJECTED_METHODS)}.  "
        f"Pre-v5.30 an unrecognised value was silently ignored and you "
        f"got the ASM field.  Tilted propagation is requested with the "
        f"element's ``tilt_x`` / ``tilt_y`` keys, not via ``method``.")


# ---------------------------------------------------------------------------
# The ``'real_lens_traced'`` element's kwarg surface (audit W9-11)
# ---------------------------------------------------------------------------
# Every key on the element dict is FORWARDED, and anything the element does
# not accept RAISES.  Hard-coding a fixed argument list here drops the rest in
# silence, which is what made the v5.29 + S12 VALIDATED traced configuration
# (``amplitude_model='ray_density'`` + ``preserve_input_phase='remap'`` +
# ``remap_sampling='full'`` + ``fit_radius_beam_factor=2.0``, the shipping
# defaults of ``propagate_traced_carrier_chain``) unreachable through this
# chain API -- with no diagnostic for a caller who wrote those keys.  Silent
# fall-through on an unrecognised key is the class the P6 twin fix closed for
# ``method``.
_TRACED_ELEMENT_STRUCTURAL_KEYS = frozenset({
    'type', 'prescription', 'traced_kwargs',
})
# Managed by the chain loop itself -- an element may not set them, because the
# loop owns the working grid, the field and the progress plumbing.
_TRACED_ELEMENT_MANAGED_KWARGS = frozenset({
    'E_in', 'wavelength', 'dx', 'dy', 'progress',
})


def _traced_element_accepted_kwargs():
    """The ``apply_real_lens_traced`` keyword parameters an element dict may
    set, resolved from the function signature so the two cannot drift."""
    import inspect
    params = inspect.signature(apply_real_lens_traced).parameters
    return frozenset(
        name for name, p in params.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
        and name not in _TRACED_ELEMENT_MANAGED_KWARGS
    )


def _resolve_traced_element_kwargs(elem, index):
    """Collect + validate the ``apply_real_lens_traced`` kwargs for one
    ``'real_lens_traced'`` element (v5.31, audit W9-11).

    Accepted spellings, merged in this order (later wins):

    1. the element's own top-level keys (``{'type': 'real_lens_traced',
       'prescription': rx, 'amplitude_model': 'ray_density'}``), matching how
       ``'real_lens'`` already forwards ``slant_correction`` / ``fresnel`` /
       ``absorption``;
    2. an explicit ``'traced_kwargs'`` dict, for callers who prefer to keep the
       propagator options in one bag (and the spelling
       ``propagate_traced_carrier_chain`` uses).

    Any other key raises :class:`ValueError` naming it and listing what is
    accepted.  ``ray_subsample`` keeps an element-level default -- see
    ``_TRACED_ELEMENT_RAY_SUBSAMPLE_DEFAULT``.
    """
    accepted = _traced_element_accepted_kwargs()
    out = {}
    unknown = []
    for key, value in elem.items():
        if key in _TRACED_ELEMENT_STRUCTURAL_KEYS:
            continue
        if key in accepted:
            out[key] = value
        else:
            unknown.append(key)
    bag = elem.get('traced_kwargs') or {}
    if not isinstance(bag, dict):
        raise ValueError(
            f"propagate_through_system: elements[{index}]['traced_kwargs'] "
            f"must be a dict of apply_real_lens_traced keyword arguments, got "
            f"{type(bag).__name__}.")
    for key, value in bag.items():
        if key in accepted:
            out[key] = value
        else:
            unknown.append(f"traced_kwargs[{key!r}]")
    if unknown:
        raise ValueError(
            f"propagate_through_system: elements[{index}] "
            f"(type='real_lens_traced') has unrecognised key(s) "
            f"{sorted(unknown)!r}.  Pre-v5.31 every key this handler did not "
            f"name was dropped in SILENCE -- measured bit-identical output to "
            f"omitting it -- so a typo, or an attempt to select the validated "
            f"traced configuration, produced the legacy result with no "
            f"diagnostic.  Accepted here: 'prescription' (required), "
            f"'traced_kwargs' (a dict of the same options), and any "
            f"keyword-only parameter of apply_real_lens_traced: "
            f"{sorted(accepted)}.  Managed by the chain and therefore "
            f"refused: {sorted(_TRACED_ELEMENT_MANAGED_KWARGS)}.")
    out.setdefault('bandlimit', True)
    out.setdefault('ray_subsample', _TRACED_ELEMENT_RAY_SUBSAMPLE_DEFAULT)
    return out


# One number, three values, for the same physics (audit W9-12) --
# ``apply_real_lens_traced`` defaults ``ray_subsample=8``,
# ``propagate_traced_carrier_chain`` defaults 4 (its VALIDATED value), and
# this chain element uses 1.  The chain element's value stays 1, against the
# first instinct to align it on the chain's 4, because the measurement argues
# the other way:
#
#   * No fidelity to gain.  On the E4 corrected relay (N=1536, dx=7 um, the
#     harness the E4 pins use) the exit-wavefront Strehl at ray_subsample
#     1 / 4 / 8 measures 0.9994 / 0.9993 / 0.9974 at a beam-matched 6 mm
#     aperture and 0.9996 / 0.9995 / 0.9976 at an oversized 10 mm one: 1 and 4
#     agree to 1e-4 (8 is the one that costs, ~2e-3), while 4 runs 1.8-3.4x
#     faster.  Against the ray_subsample=1 full-density reference the single
#     element's output has an overlap deficit of 1.3e-4 at 4 and 1.3e-3 at 8,
#     for a 3.9x / 4.0x speedup.  So 4 is a pure speed choice here -- which is
#     exactly why the chain takes it -- and 1 loses nothing but time.
#   * Real breakage to pay.  ``apply_real_lens_traced`` enforces
#     ``min_coarse_samples_per_aperture=32`` with ``on_undersample='error'``, so
#     the coarse density is ``aperture/(dx*ray_subsample)`` and raising the
#     divisor to 4 QUADRUPLES the grid an existing chain needs.  MEASURED on a
#     2 mm-aperture singlet: ``ray_subsample=1`` runs at every grid tried, while
#     ``ray_subsample=4`` raises ``ValueError`` ("only 12.5 coarse samples ...
#     threshold 32") whenever the aperture spans fewer than ~128 samples --
#     broken at 50 and 100 samples, fine at 200 and 400.  Flipping the default
#     would turn working coarse-grid chains into hard failures for no measured
#     accuracy return.
#
# 4 is now REACHABLE (``{'type': 'real_lens_traced', ..., 'ray_subsample': 4}``
# forwards, as does every other traced option -- audit W9-11) and the docstring
# says so.  The ELEMENT's own default likewise stays 8: that is the direct
# ``apply_real_lens_traced`` path, outside this chain, and moving it would shift
# pinned results for callers who never touched the chain.
_TRACED_ELEMENT_RAY_SUBSAMPLE_DEFAULT = 1



def _warn_system_resample_crop(E, dx_new, dx_target, N_out, kernel_name):
    """Warn when resampling a kernel's natural output grid back to the
    chain pitch CROPS the field (audit K6).

    The ``'sas'`` leg is the one caller: after
    ``scalable_angular_spectrum_propagate`` the field lives at ``dx_new``
    over an extent ``N*dx_new``, and the chain resamples it onto
    ``N*dx_target``.  When ``dx_new > dx_target`` -- the common
    diverging-beam case -- everything outside the central ``N*dx_target``
    is discarded (by ``map_coordinates(mode='constant', cval=0.0)`` on
    the spline leg, by the chirp-Z leg's window on the other), silently.
    The ``'fresnel'`` leg evaluates onto the chain grid directly and so
    has no resample to crop; the same physical question -- does the chain
    window still hold the beam? -- is asked there by
    ``_warn_system_fresnel_window``, which measures the power the window
    keeps.  (``fresnel_propagate_mft``'s faithful-zone warning is a
    different condition: on the chain grid it reduces to
    ``z < N*dx^2/lambda``, the K1 under-sampled band.)

    Measured (grid-filling top-hat of radius 0.42*N*dx, N = 512,
    dx = 2 um, lambda = 633 nm, z = 5 mm, dx_new/dx = 1.5454 for the
    single-FFT Fresnel grid and 0.7727 for SAS's):  ``P_out/P_in`` =
    0.998990 for ``method='asm'`` (band limit, expected) against
    0.950689 for ``'sas'``; a Fresnel step onto that natural grid
    conserves power to 1.000000 and the retained window holds 0.996980
    of it, i.e. essentially ALL of the loss is the crop.

    Values are unchanged -- diagnostic only.
    """
    if not (float(dx_new) > float(dx_target)):
        return
    import warnings
    p_full = float(np.sum(np.abs(np.asarray(E)) ** 2))
    if p_full <= 0.0:
        return
    n_src = int(np.asarray(E).shape[-1])
    half = 0.5 * int(N_out) * float(dx_target)
    x = (np.arange(n_src, dtype=np.float64) - n_src / 2.0) * float(dx_new)
    keep = np.abs(x) <= half
    p_win = float(np.sum(np.abs(
        np.asarray(E)[np.ix_(keep, keep)]) ** 2)) if keep.any() else 0.0
    if p_win >= p_full * (1.0 - 1e-6):
        return
    warnings.warn(
        f"propagate_through_system: the {kernel_name} leg returned its "
        f"natural output grid at dx={float(dx_new):.4e} m (extent "
        f"{n_src * float(dx_new):.4e} m) and the chain resamples it back "
        f"to dx={float(dx_target):.4e} m (extent "
        f"{int(N_out) * float(dx_target):.4e} m), which CROPS it: only "
        f"{100.0 * p_win / p_full:.2f}% of the power falls inside the "
        f"retained window.  The beam has spread past the chain's grid.  "
        f"Use a larger N, a coarser chain pitch, or method='asm' (which "
        f"keeps the pitch and so cannot crop).",
        RuntimeWarning, stacklevel=_caller_stacklevel())


def _warn_system_fresnel_window(E_before, E_after, dx_target, wavelength, z):
    """Warn when the chain's own window holds only part of the field the
    ``'fresnel'`` leg propagated (audit K6).

    That leg evaluates the Fresnel integral directly on the chain grid, so
    there is no resample and nothing to crop -- but the physical loss the
    crop used to report is still there.  A beam that has spread past
    ``N*dx`` is simply not represented, and the step conserves power only
    while the window holds it.  ``fresnel_propagate_mft``'s own
    faithful-zone warning does NOT cover this case: its condition is
    ``N_out*dx_out > lambda*|z|/dx_in``, which at this call site
    (``dx_out = dx_in = dx``, ``N_out = N``) is ``z < N*dx^2/lambda`` --
    exactly the K1 under-sampled-chirp band -- while the window loss grows
    in the OTHER direction, at ``z`` above that bound.  The two conditions
    are disjoint here, so this diagnostic is the only one that fires when
    the beam outgrows the grid.

    The ``z > N*dx^2/lambda`` test below keeps that partition exact, and it
    is the same band ``_warn_system_resample_crop`` covered on this leg
    (``dx_new = lambda*z/(N*dx) > dx`` is the same inequality).  Below the
    bound a short window is not a crop at all: the natural grid is FINER
    than the chain's, so the reconstruction replicates rather than
    truncates, which is what the faithful-zone warning and the K1
    under-sampled-chirp warning are already saying.

    Measured (2026-09-13, lambda = 633 nm, dx = 2 um).  Contained
    Gaussians (``w0 = 0.06*N*dx``) at ``z`` = 1x, 2x and 3x the K1 bound
    read ``P_out/P_in`` = 1.000000000 for N = 64, 65, 128 and 256 -- worst
    departure 3.1e-8, four decades below the bar.  Against that: a
    top-hat of radius 3 px at N = 64 and ``z`` = 30x the bound keeps
    0.031674 of its power, a Gaussian of ``w0`` = 2.6 px at N = 128 and
    10x keeps 0.334870 (the same field evaluated on an 8x-wider window
    conserves 1.000000, so ALL of it is the chain window), and a
    grid-filling top-hat at N = 512 and 2x keeps 0.995833.  The 1e-6 bar
    is the one ``_warn_system_resample_crop`` already uses for the same
    question.

    ``dy`` equals ``dx`` on this leg (``_require_square_pitch``), so the
    two sums carry the same ``dx*dy`` factor and their ratio is the power
    ratio.  Both reductions are written with the array's own operators --
    ``float((abs(E)**2).sum())`` -- rather than through ``np.asarray``, so
    a CuPy or JAX field reaching this leg reduces on its own backend
    instead of refusing the implicit conversion.

    Values are unchanged -- diagnostic only.
    """
    z_crit = int(E_before.shape[-1]) * float(dx_target) ** 2 \
        / float(wavelength)
    # ``z_crit > 0`` as well as the band test: the propagator's own
    # validation has already refused a non-positive pitch or wavelength,
    # so this only guards the message's ``z/z_crit`` ratio.
    if not (z_crit > 0.0 and abs(float(z)) > z_crit):
        return
    p_in = float((abs(E_before) ** 2).sum())
    if not (p_in > 0.0):
        return
    p_out = float((abs(E_after) ** 2).sum())
    if p_out >= p_in * (1.0 - 1e-6):
        return
    import warnings
    n_out = int(E_after.shape[-1])
    warnings.warn(
        f"propagate_through_system: the fresnel leg evaluated the integral "
        f"on the chain window (N={n_out} at dx={float(dx_target):.4e} m, "
        f"extent {n_out * float(dx_target):.4e} m) and only "
        f"{100.0 * p_out / p_in:.2f}% of the input power lands inside it.  "
        f"The beam has spread past the chain's grid: z = {float(z):.6g} m "
        f"is {abs(float(z)) / z_crit:.3g}x N*dx^2/wavelength = "
        f"{z_crit:.6g} m, and the Fresnel integral's own scale at this z "
        f"is lambda*|z|/dx = "
        f"{float(wavelength) * abs(float(z)) / float(dx_target):.4e} m.  "
        f"Values outside the window are not cropped from anything -- they "
        f"were never evaluated.  Use a larger N, a coarser chain pitch, or "
        f"method='asm' (which keeps the pitch and band-limits instead of "
        f"windowing).",
        RuntimeWarning, stacklevel=_caller_stacklevel())


def propagate_through_system(E_in: np.ndarray,
                             elements: Sequence[Dict[str, Any]],
                             wavelength: float,
                             dx: float,
                             dy: Optional[float] = None,
                             method: Optional[str] = None,
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
        - ``'fresnel'`` : Fresnel diffraction (paraxial), evaluated by
          :func:`~lumenairy.propagators.fresnel_propagate_mft` straight
          onto the chain grid, so the pitch never changes and
          lens/aperture phases stay on the right coordinates without an
          interpolation step (audit K6).  This is the same integral the
          single-FFT :func:`~lumenairy.propagators.fresnel_propagate`
          computes, sampled where the chain wants it instead of on that
          kernel's natural ``lambda*z/(N*dx)`` grid.
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

        Anything outside ``{'asm', 'sas', 'fresnel'}`` raises
        ``ValueError`` -- including ``'rs'``, ``'gbd'``, ``'fraunhofer'``
        and wrong-case spellings like ``'ASM'``.  Same for a per-element
        ``elem['method']`` override.  Tilted propagation is requested via
        the element's ``tilt_x`` / ``tilt_y`` keys, not through ``method``.

        .. versionchanged:: 5.30
            ``method`` (and the per-element ``elem['method']`` override)
            are validated.  Pre-v5.30 an unrecognised value fell through
            to the ``else: # Default: ASM`` branch and silently returned
            the ASM field -- measured 0.0 relative difference vs
            ``method='asm'`` for ``'not_a_method'``, ``'gbd'``,
            ``'fraunhofer'`` and ``'ASM'``.  This is the NumPy twin of
            the JAX-side audit P6 fix.

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
        ``method`` parameter -- validated against the same
        ``{'asm', 'sas', 'fresnel'}`` set since v5.30).

    ``'propagate_tilted'``
        Legacy alias for ``'propagate'`` with tilt parameters.
        Prefer using ``{'type': 'propagate', 'tilt_x': ..., 'tilt_y': ...}``
        in new code.
        Keys: ``z``, ``tilt_x`` / ``tilt_y`` (optional, default 0),
        ``bandlimit`` (bool, optional, default True).

        **This element is ASM-only and ignores ``method``.**  It calls
        :func:`angular_spectrum_propagate_tilted` directly, so a ``method``
        key is neither honoured nor validated (there is no tilted Fresnel /
        SAS kernel).  Since v5.30 supplying one emits a ``UserWarning``
        naming the limitation; pre-v5.30 it was dropped silently (measured
        bit-identical output for ``'fresnel'``, ``'sas'`` and outright junk).
        Use the unified ``'propagate'`` element if you need ``method``
        validation.

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

        Keys: ``prescription`` (dict, required), plus **any keyword-only
        parameter of** :func:`~lumenairy.apply_real_lens_traced`, either as a
        top-level element key or inside a ``traced_kwargs`` dict.  Unrecognised
        keys raise ``ValueError`` listing what is accepted.

        .. versionchanged:: 5.31
            **The traced options are reachable, and typos are loud** (audit
            W9-11).  Pre-v5.31 the handler forwarded only ``prescription`` /
            ``bandlimit`` / ``ray_subsample`` and dropped every other key in
            SILENCE -- measured bit-identical output to omitting it for all of
            ``amplitude_model``, ``preserve_input_phase``, ``remap_sampling``,
            ``fit_radius_beam_factor``, ``carrier``, ``on_undersample``,
            ``n_workers``, ``traced_kwargs`` and an outright typo key.  The
            v5.29 + S12 validated traced configuration was therefore
            unreachable through this API::

                {'type': 'real_lens_traced', 'prescription': rx,
                 'traced_kwargs': {'amplitude_model': 'ray_density',
                                   'preserve_input_phase': 'remap',
                                   'remap_sampling': 'full',
                                   'fit_radius_beam_factor': 2.0}}

            (that dict is what :func:`~lumenairy.propagate_traced_carrier_chain`
            applies by default; use the chain itself when you also want its
            carrier-referenced hand-offs, which a flat element list cannot
            express).

        ``ray_subsample`` defaults to **1** here (full ray density).  The three
        entry points to the same physics disagree on purpose:
        :func:`~lumenairy.apply_real_lens_traced` defaults 8,
        :func:`~lumenairy.propagate_traced_carrier_chain` defaults 4 (its
        validated value), this element 1.

        .. versionchanged:: 5.31
            The *docstring* here is fixed, not the value (audit W9-12).  It used
            to read "default 1; 4 is the recommended production value", i.e. it
            recommended against the code beneath it.  4 IS the better production
            setting and is now reachable (``'ray_subsample': 4`` forwards, as
            does every other traced option), but it is not the default, because
            MEASURED it buys no fidelity and can break existing chains: on the
            E4 corrected relay the exit-wavefront Strehl at 1 / 4 / 8 is
            0.9994 / 0.9993 / 0.9974 (6 mm) and 0.9996 / 0.9995 / 0.9976
            (10 mm), while ``min_coarse_samples_per_aperture=32`` with
            ``on_undersample='error'`` means a divisor of 4 quadruples the grid
            a chain needs -- a 2 mm aperture spanning 50 or 100 samples runs at
            ``ray_subsample=1`` and raises ``ValueError`` at 4.  Set it
            explicitly when your grid can afford it: ~3.9x faster per element,
            1.3e-4 overlap deficit against full density.

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
        Sharp-edged (unapodized) aperture -- circular, rectangular,
        annular.  Keys: ``shape`` (str, optional, default
        ``'circular'``), ``params`` (dict, optional, shape-specific
        parameters), ``xc`` / ``yc`` (float, optional, default 0),
        ``edge`` and ``edge_samples`` (optional, forwarded to
        :func:`~lumenairy.elements.elements.apply_aperture`).  Omitting
        ``edge`` takes that function's own default, which renders the rim by
        pixel AREA; add ``'edge': 'hard'`` for the binary pixel-centre mask.
        The JAX twin reads the same keys and takes the same default.

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
        ``seed`` (int, optional),
        ``subharmonics`` (int, optional, default 0 -- the number of Lane
        subharmonic levels added below the FFT lattice's fundamental
        frequency; ``0`` is the plain FFT screen, which cannot represent
        eddies larger than the grid, and ``3`` is the usual choice.  See
        :func:`lumenairy.generate_turbulence_screen`).

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
    _check_2d_scalar_field(E_in, 'propagate_through_system',
                           input_kind='field')

    from ..progress import ProgressScaler, call_progress
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

    # v5.1.0 (default-knob resolver rollout): resolve ``method`` /
    # ``dy`` from the library-wide defaults when callers leave them
    # at the ``None`` sentinel.  Explicit values bypass the resolver.
    if method is None:
        from .propagation import get_default_wave_propagator
        method = get_default_wave_propagator()
    # ``propagate_through_system`` only supports the wave-propagator
    # subset {'asm', 'sas', 'fresnel'} for its free-space ``propagate``
    # element step; the consumer set rejects 'rs' / 'rayleigh_sommerfeld'.
    if method in ('rs', 'rayleigh_sommerfeld'):
        raise ValueError(
            f"propagate_through_system: resolved method={method!r} from "
            f"set_default_wave_propagator is not supported here.  "
            f"Supported choices in this entry point: 'asm', 'sas', "
            f"'fresnel'.  Pass ``method=`` explicitly or reset the "
            f"default via ``set_default_wave_propagator('asm')``.")
    # Validate ``method`` at ENTRY against the honoured set (the NumPy twin
    # of audit P6).  Without it an unrecognised value -- 'gbd',
    # 'fraunhofer', 'ASM', outright junk -- falls through the ``if/elif``
    # chain below to the ``else: # Default: ASM`` branch and silently
    # returns the ASM field (measured 0.0 relative difference against
    # method='asm').  The JAX twin does the same, with the same wording.
    _validate_system_method(method, 'method')
    if dy is None:
        from .propagation import get_default_dy
        dy = get_default_dy()

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
            # The per-element override takes the same silent fall-through
            # to ASM as the entry-point kwarg (measured 0.0 relative
            # difference against method='asm' for
            # ``{'method': 'not_a_method'}``), so it needs the identical
            # validation: an entry-only check leaves the hole open (NumPy
            # twin of audit P6).
            if 'method' in elem:
                _validate_system_method(
                    prop_method, f"elements[{i}]['method']")

            # Check for tilt parameters — if present, use tilted ASM
            tilt_x = elem.get('tilt_x', 0.0)
            tilt_y = elem.get('tilt_y', 0.0)
            has_tilt = (tilt_x != 0.0) or (tilt_y != 0.0)

            if prop_method == 'fresnel' and not has_tilt:
                _require_square_pitch(current_dx, current_dy, 'fresnel')
                # K6: the Fresnel integral is evaluated STRAIGHT ONTO the
                # chain grid, so downstream element phases (lenses,
                # apertures) land on the right coordinate system without
                # an interpolation step.  Taking the single-FFT kernel's
                # natural grid ``lambda z/(N dx)`` and interpolating back
                # costs two errors the MFT form does not have: everything
                # outside ``N*current_dx`` is discarded (the crop), and
                # the interpolator's MTF attenuates the output-plane
                # chirp, which sits at exactly Nyquist at the rim by
                # construction.  Measured against this direct evaluation,
                # the resample-back sat 2.8e-2 (N=512) to 4.9e-2 (N=256)
                # relative L2 away on a grid-filling top-hat at
                # lambda = 633 nm, dx = 2 um, z = 5 mm, and held
                # 0.996685 of the input power against the direct
                # evaluation's 0.996992.
                #
                # ``fresnel_propagate_mft`` carries the same K1
                # chirp-sampling guard as ``fresnel_propagate`` and adds
                # its own faithful-zone warning (period
                # ``lambda*|z|/dx_in``), so this leg needs no
                # ``_warn_system_resample_crop``: there is no resample to
                # crop.  It does still need a WINDOW diagnostic -- the
                # faithful-zone condition reduces here to
                # ``z < N*dx^2/lambda``, the K1 band, while a beam
                # outgrows the chain window at z ABOVE that bound, so the
                # two never fire on the same geometry;
                # ``_warn_system_fresnel_window`` covers the second.
                # ``N_out``/``dy_out`` keep the chain's square sample
                # count and pitch.
                if verbose:
                    _dx_natural = wavelength * z / (
                        int(E.shape[-1]) * current_dx)
                    print(f"    Fresnel evaluated directly at "
                          f"dx={current_dx*1e6:.3f} um "
                          f"(the single-FFT natural grid would be "
                          f"{_dx_natural*1e6:.3f} um)")
                _E_pre_fresnel = E
                E = fresnel_propagate_mft(
                    E, z, wavelength, current_dx, current_dx,
                    int(E_in.shape[-1]), dy_in=current_dy,
                    dy_out=current_dx)
                _warn_system_fresnel_window(
                    _E_pre_fresnel, E, current_dx, wavelength, z)
                del _E_pre_fresnel
            elif prop_method == 'sas' and not has_tilt:
                _require_square_pitch(current_dx, current_dy, 'sas')
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
                    # K6: the resample-back CROPS when dx_new > current_dx.
                    _warn_system_resample_crop(
                        E, dx_new, current_dx, E_in.shape[-1], 'SAS')
                    # K6: the band-limited (chirp-Z) interpolant has unit
                    # MTF at every frequency the grid represents, but its
                    # reconstruction is PERIODIC with period
                    # ``N_in*dx_new`` per axis, so it is faithful only
                    # while the chain window fits inside one period.
                    # Past that the extra samples are replicas of the
                    # field rather than the zeros the spline pads with:
                    # measured P_out/P_in 1.378837 at dx_new/dx = 0.7727
                    # and 9.000535 (a 3x3 tiling) at 0.3091, against the
                    # spline's 0.950689 and 0.999999.  The window/period
                    # test below is the general form of "the pitch
                    # coarsened" -- the two readings coincide only
                    # because the chain keeps ``N_out == N_in`` -- and its
                    # 1e-9 slack is ``_warn_mft_output_window``'s own
                    # tolerance, so the chirp-Z leg is chosen on exactly
                    # the windows it would not warn about.  ``min`` takes
                    # the binding axis: ``resample_field`` reads one input
                    # pitch for both, so the shorter input extent sets
                    # the period.
                    window_out = int(E_in.shape[-1]) * current_dx
                    period_in = min(E.shape[-2], E.shape[-1]) * dx_new
                    E, _ = resample_field(
                        E, dx_new, current_dx, N_out=E_in.shape[-1],
                        method=('chirpz'
                                if window_out <= period_in * (1.0 + 1e-9)
                                else 'spline'))
            elif has_tilt:
                # Tilted ASM (always ASM — no tilted Fresnel variant)
                if verbose and prop_method == 'fresnel':
                    print("    tilt specified — using tilted ASM "
                          "instead of Fresnel")
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
            # v5.35.0 (BUILD_R1_WIRING S4): ``carrier`` (+ its two policy
            # keys) join the forwarded set.  A chain element that states the
            # beam's congruence could previously do so only on the
            # ``'real_lens_traced'`` sibling; on this one the key was accepted
            # by the dict and DROPPED IN SILENCE -- the same class v5.31's
            # W9-11 closed for the traced element -- which made the analytic
            # angular correction unreachable through the chain API.
            _rl_kw = {}
            if elem.get('carrier') is not None:
                _rl_kw['carrier'] = elem['carrier']
            for _k in ('screen_obliquity', 'on_screen_obliquity'):
                if _k in elem:
                    _rl_kw[_k] = elem[_k]
            E = apply_real_lens(
                E, prescription=elem['prescription'], wavelength=wavelength,
                dx=current_dx, dy=current_dy,
                bandlimit=elem.get('bandlimit', True),
                slant_correction=elem.get('slant_correction', False),
                fresnel=elem.get('fresnel', False),
                absorption=elem.get('absorption', False),
                progress=(lambda stage, frac, msg='': sub_cb(frac, msg))
                        if progress is not None else None,
                **_rl_kw,
            )

        elif elem['type'] == 'real_lens_traced':
            _tkw = _resolve_traced_element_kwargs(elem, i)
            E = apply_real_lens_traced(
                E, prescription=elem['prescription'], wavelength=wavelength,
                dx=current_dx, dy=current_dy,
                progress=(lambda stage, frac, msg='': sub_cb(frac, msg))
                        if progress is not None else None,
                **_tkw,
            )

        elif elem['type'] == 'mirror':
            E = apply_mirror(E, wavelength, current_dx,
                             radius=elem.get('radius'),
                             conic=elem.get('conic', 0.0),
                             aperture_diameter=elem.get('aperture_diameter'),
                             xc=elem.get('xc', 0), yc=elem.get('yc', 0),
                             dy=current_dy)

        elif elem['type'] == 'aperture':
            # WP-C1: the element forwards ``edge`` / ``edge_samples``
            # so a chain can ask for the binary pixel-centre rim exactly as
            # a direct ``apply_aperture`` call can.  Omitted, both take the
            # function's own defaults (``'gray'``, 4).
            E = apply_aperture(E, current_dx,
                               shape=elem.get('shape', 'circular'),
                               params=elem.get('params', {}),
                               xc=elem.get('xc', 0), yc=elem.get('yc', 0),
                               dy=current_dy,
                               **_aperture_edge_kwargs(elem))

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

        elif elem['type'] == 'rcwa':
            E = _apply_rcwa_element(E, elem)

        elif elem['type'] == 'coating':
            E = _apply_coating_element(E, elem, system_wavelength=wavelength)

        elif elem['type'] == 'propagate_tilted':
            # Legacy alias — redirect to the unified 'propagate' handler
            # with tilt parameters.  New code should use:
            #   {'type': 'propagate', 'z': ..., 'tilt_x': ..., 'tilt_y': ...}
            #
            # This handler is ASM-ONLY: it goes straight to
            # ``angular_spectrum_propagate_tilted`` and never reads
            # ``elem['method']``.  A ``method`` key here is WARNED about
            # rather than raised on -- raising would be a new breakage
            # class for a legacy alias that has silently accepted the key
            # for many releases.  MEASURED before the warning existed: 0.0
            # relative difference and bit-identical output for
            # ``method='fresnel'``, ``'sas'`` AND ``'not_a_method'``
            # versus omitting the key.
            if 'method' in elem:
                warnings.warn(
                    f"propagate_through_system: elements[{i}] has "
                    f"type='propagate_tilted' with method="
                    f"{elem['method']!r}, which is IGNORED.  The legacy "
                    f"'propagate_tilted' element is ASM-only -- it always "
                    f"calls angular_spectrum_propagate_tilted and neither "
                    f"honours nor validates a 'method' key (there is no "
                    f"tilted Fresnel/SAS kernel).  Use the unified element "
                    f"{{'type': 'propagate', 'z': ..., 'tilt_x': ..., "
                    f"'tilt_y': ..., 'method': ...}} instead: it validates "
                    f"'method' and, for an untilted leg, honours it.  Drop "
                    f"the 'method' key to silence this warning.",
                    UserWarning, stacklevel=_caller_stacklevel())
            E = angular_spectrum_propagate_tilted(
                E, elem['z'], wavelength, current_dx, current_dy,
                tilt_x=elem.get('tilt_x', 0),
                tilt_y=elem.get('tilt_y', 0),
                bandlimit=elem.get('bandlimit', True))

        elif elem['type'] == 'turbulence':
            _require_square_pitch(current_dx, current_dy, 'turbulence')
            screen = generate_turbulence_screen(
                E.shape[0], current_dx,
                r0=elem['r0'],
                L0=elem.get('L0', np.inf),
                l0=elem.get('l0', 0.0),
                seed=elem.get('seed'),
                # The chain was the one caller that could not reach the Lane
                # subharmonic levels; the default 0 is the generator's own, so
                # an element dict without the key produces the identical screen.
                subharmonics=elem.get('subharmonics', 0))
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
                RuntimeWarning, stacklevel=_caller_stacklevel(),
            )
    if output_dx is not None and float(output_dx) != float(source.dx):
        warnings.warn(
            "lumenairy.propagators.system.evaluate: output_dx resampling is "
            "reserved for a future release; the kwarg is accepted but "
            "currently does not resample the exit-plane field.",
            RuntimeWarning, stacklevel=_caller_stacklevel(),
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
                # currently exists for raw DOE phase.  SY-1: WARN instead of
                # dropping the phase silently (a physically wrong result with
                # no diagnostic); a future DOE handler plugs in here.  For an
                # exact treatment use propagate_through_system with a
                # hand-built 'mask' element carrying the DOE phase.
                _doe_lbl = step.get('label') or step.get('surface') or '?'
                warnings.warn(
                    f"lumenairy.propagators.system.evaluate: DOE / air-to-air "
                    f"aspheric surface ({_doe_lbl}) has no element handler and "
                    f"its phase is DROPPED -- the result omits this element.  "
                    f"Use propagate_through_system with a 'mask' element "
                    f"carrying the DOE phase for an exact result.",
                    stacklevel=_caller_stacklevel())
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
    import sys as _sys

    from .._cache_registry import register_cache_clearer as _register_cache_clearer
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
    'aperture',      # amplitude mask multiplication (see _aperture_edge_kwargs)
    'mask',          # arbitrary mask multiplication
})


def _aperture_edge_kwargs(elem: Dict[str, Any]) -> Dict[str, Any]:
    """The ``edge`` / ``edge_samples`` an ``'aperture'`` element asks for.

    WP-C1.  Both backends resolve the rim rendering HERE, from one reading
    of the element, so the NumPy chain and its JAX twin cannot drift apart
    on it: an element that names neither key takes
    :func:`~lumenairy.elements.elements.apply_aperture`'s own defaults, which
    render the rim by pixel AREA.  An element that pins ``{'edge': 'hard'}``
    gets the binary pixel-centre rim, bit for bit, on either backend.

    Returns only the keys the element actually names, so an element that
    names neither cannot pin today's defaults into tomorrow's answer.

    VERIFY-C1 D1.  This is the ONE place both backends read the element, so
    it is also where the element is REFUSED.  The jit'd JAX kernel takes its
    static signature from :func:`_system_element_signature`, which has to be
    hashable and therefore coerces with ``int()`` / ``str()``; measured
    2026-09-20 on both builds, that coercion made the jit'd route accept
    ``{'edge_samples': 2.5}`` (silently using 2) and ``{'edge_samples': '4'}``
    where ``apply_aperture``, the NumPy chain and the eager JAX route all
    raised ``ValueError``.  Validating here -- through
    :func:`~lumenairy.elements.elements._validate_edge_kwargs`, which is the
    same function ``apply_aperture``'s own body calls and not a second copy
    of the guards -- gives all three routes one reading and one verdict, and
    lets the signature keep coercing, because by the time it runs the value
    is known to be one of the two legal strings and an exact positive
    integer.
    """
    kw: Dict[str, Any] = {}
    if 'edge' in elem:
        kw['edge'] = elem['edge']
    if 'edge_samples' in elem:
        kw['edge_samples'] = elem['edge_samples']
    if kw:
        _validate_edge_kwargs(**kw)
    return kw


def _jax_apply_aperture_element(E, dx, dy, shape, params, xc, yc,
                                edge, edge_samples):
    """Apply one ``'aperture'`` element on the JAX backend.

    WP-C1.  Calls the ONE aperture implementation
    (:func:`~lumenairy.elements.elements.apply_aperture`, which dispatches
    through ``backend.array_namespace`` and traces under ``jax.jit`` /
    ``jax.grad``) rather than carrying a second copy of the mask here.
    ``edge`` / ``edge_samples`` are ``None`` when the element named
    neither, in which case the function's own defaults apply -- the same
    defaults the NumPy chain takes, which is what makes the two backends
    answer the same element dict the same way.
    """
    kw: Dict[str, Any] = {}
    if edge is not None:
        kw['edge'] = edge
    if edge_samples is not None:
        kw['edge_samples'] = edge_samples
    return apply_aperture(E, dx, shape=shape, params=params,
                          xc=xc, yc=yc, dy=dy, **kw)


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
        # WP-C1: the rim rendering is part of the STATIC
        # signature -- two elements that differ only in ``edge`` are two
        # different kernels, not one kernel silently serving both.
        # VERIFY-C1 D1: the ``str()`` / ``int()`` below are here only to
        # keep the signature hashable, and they are SAFE because
        # ``_aperture_edge_kwargs`` has already refused anything that is not
        # one of the two legal strings or an exact positive integer.  Before
        # that guard existed they were the whole difference between this
        # route and the other two.
        edge_kw = _aperture_edge_kwargs(elem)
        edge = (str(edge_kw['edge']) if 'edge' in edge_kw else None)
        n_sub = (int(edge_kw['edge_samples'])
                 if 'edge_samples' in edge_kw else None)
        if shape == 'circular':
            return ('aperture_circular', halves[0], xc, yc, edge, n_sub)
        if shape == 'rectangular':
            return ('aperture_rect', halves[0], halves[1], xc, yc,
                    edge, n_sub)
        if shape == 'annular':
            return ('aperture_annular', halves[0], halves[1], xc, yc,
                    edge, n_sub)
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
                # S2-13 (audit AUDIT_V5_24_2): thread ``dy`` so an
                # anamorphic grid (dy != dx) propagates the y axis at its
                # own pitch on the JAX path, matching the NumPy
                # ``propagate_through_system`` which passes ``current_dy``.
                # Pre-fix the JAX kernel dropped ``dy`` and ASM defaulted
                # ``dy = dx`` -- a silent wrong-pitch y propagation.
                E = angular_spectrum_propagate(
                    E, z, wavelength, dx, dy=dy, bandlimit=bandlimit)
            elif tag == 'lens':
                _, f, xc, yc = sig
                r2 = (X - xc) ** 2 + (Y - yc) ** 2
                E = E * jnp.exp(-1j * k0 * r2 / (2.0 * f))
            # WP-C1: the three aperture tags call the ONE
            # aperture implementation (``elements.apply_aperture``, which
            # is ``array_namespace``-dispatched and traces under
            # ``jax.jit`` / ``jax.grad``) instead of re-deriving the mask
            # here.  A second copy would let the two backends answer
            # differently for the SAME element dict -- and the
            # cross-backend bar that guards them (5 % of pixels) is far
            # too loose to see a rim.  One implementation, one default,
            # one way back.
            elif tag == 'aperture_circular':
                _, r, xc, yc, edge, n_sub = sig
                E = _jax_apply_aperture_element(
                    E, dx, dy, 'circular', {'diameter': 2.0 * r},
                    xc, yc, edge, n_sub)
            elif tag == 'aperture_rect':
                _, hx, hy, xc, yc, edge, n_sub = sig
                E = _jax_apply_aperture_element(
                    E, dx, dy, 'rectangular',
                    {'width_x': 2.0 * hx, 'width_y': 2.0 * hy},
                    xc, yc, edge, n_sub)
            elif tag == 'aperture_annular':
                _, r_o, r_i, xc, yc, edge, n_sub = sig
                E = _jax_apply_aperture_element(
                    E, dx, dy, 'annular',
                    {'inner_diameter': 2.0 * r_i,
                     'outer_diameter': 2.0 * r_o},
                    xc, yc, edge, n_sub)
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

    Free-space method
    -----------------
    ``method`` selects the free-space kernel used for ``'propagate'``
    elements.  This entry point implements **ASM only**: the whole point
    of the JAX path is a single jit'd XLA graph, and the NumPy twin's
    ``'sas'`` branch resamples back onto the input pitch
    (``resample_field`` -> ``scipy.ndimage.map_coordinates`` on the spline
    leg) while its ``'fresnel'`` branch goes through
    ``fresnel_propagate_mft``'s Bluestein pair -- neither has a
    JAX-traceable equivalent wired up here.  Any other value raises rather
    than silently returning the ASM answer:

      * ``'asm'`` (default) -- angular-spectrum, jit-traceable.
      * ``'fresnel'`` / ``'sas'`` / ``'rs'`` -- recognised by
        :func:`propagate_through_system` but NOT implemented on the JAX
        path -> :class:`NotImplementedError` (use the NumPy twin).
      * anything else -> :class:`ValueError`.

    Note that, unlike the NumPy twin, this function does **not** consult
    ``set_default_wave_propagator()``: the JAX path is ASM-only by
    construction, so a process-wide default of ``'fresnel'`` would make
    every JAX call raise.  Pass ``method`` explicitly.

    .. versionchanged:: 5.30
        ``method`` is validated (audit P6,
        ``AUDIT_ADVERSARIAL_CODEBASE_2026_07_25``).  Pre-v5.30 the
        parameter was accepted and never read, so
        ``method='fresnel'`` silently returned the ASM field -- measured
        5.0e-2 relative L2 away from the NumPy twin's Fresnel result on
        a 128^2 / z=2 mm Gaussian probe, and bit-identical to the
        ``method='asm'`` call.

    Supported (traceable) element types
    -----------------------------------
    The set of element types with a fully JAX-traceable code path is
    exposed as the module-level constant
    :data:`lumenairy.propagators.system._TRACEABLE_ELEMENT_TYPES`:

      * ``'propagate'``  -> ``angular_spectrum_propagate`` (JAX backend)
      * ``'lens'``       -> paraxial thin-lens phase screen
      * ``'aperture'``   -> :func:`apply_aperture` (circular /
        rectangular / annular), the SAME implementation the NumPy chain
        calls -- WP-C1 replaced this kernel's own copy of the
        pixel-centre indicator with it, so both backends take the same
        ``edge`` default and read the same ``'edge'`` / ``'edge_samples'``
        element keys.  Uses the same
        canonical NumPy schema
        as :func:`apply_aperture` (``params={'diameter': ...}`` etc.);
        the pre-v4.12 JAX-only schema (``params={'radius': ...}``) was
        deprecated in v4.12 and **removed in v5.0** -- it now raises
        ``ValueError`` with the migration recipe.  See
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
        :data:`_TRACEABLE_ELEMENT_TYPES`, or if ``method`` names a
        free-space kernel the NumPy twin supports but this JAX path does
        not (``'fresnel'`` / ``'sas'`` / ``'rs'``).
    ValueError
        If ``method`` is not a recognised free-space method name.
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
    _check_2d_scalar_field(E_in, 'propagate_through_system_jax',
                           input_kind='field')

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
    # v5.30 (audit P6): validate ``method``.
    # ------------------------------------------------------------------
    # ``method`` is validated because this path implements ASM and nothing
    # else, while the NumPy twin ``propagate_through_system`` honours
    # 'fresnel' / 'sas' (and rejects 'rs').  Accepting a value it cannot
    # honour would return the ASM field bit-identically under any name --
    # the forbidden silent-fall-through class.  ``method=None`` is accepted
    # as an explicit "use this entry point's default" (it does NOT resolve
    # ``set_default_wave_propagator()`` -- see the docstring).
    _JAX_METHODS = ('asm',)
    _NUMPY_TWIN_ONLY_METHODS = ('fresnel', 'sas', 'rs', 'rayleigh_sommerfeld')
    if method is None:
        method = 'asm'
    if method not in _JAX_METHODS:
        if method in _NUMPY_TWIN_ONLY_METHODS:
            raise NotImplementedError(
                f"propagate_through_system_jax: method={method!r} has no "
                f"JAX-traceable free-space kernel here; this entry point "
                f"implements {list(_JAX_METHODS)} only (the NumPy twin's "
                f"'sas' branch resamples back onto the input pitch via "
                f"scipy map_coordinates and its 'fresnel' branch runs "
                f"fresnel_propagate_mft's Bluestein pair; neither is "
                f"wired up as a traceable step here).  Pre-v5.30 this "
                f"argument was silently "
                f"ignored and you got the ASM field.  Use "
                f"lumenairy.propagators.system.propagate_through_system() "
                f"(NumPy) for method={method!r}, or pass method='asm' if "
                f"the ASM field is what you want.")
        raise ValueError(
            f"propagate_through_system_jax: method={method!r} is not a "
            f"recognised free-space method.  Supported here: "
            f"{list(_JAX_METHODS)}; additionally recognised by the NumPy "
            f"twin (and rejected here with NotImplementedError): "
            f"{list(_NUMPY_TWIN_ONLY_METHODS)}.")

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

    # Dtype resolution goes through ``_resolve_jax_complex_dtype``, which
    # reads the library-wide default (audit L2): complex inputs honour
    # their own dtype, only real inputs fall back to the configured
    # default.  The probe reads the dtype by DUCK-TYPING (``getattr(E,
    # 'dtype', None)``, the ``_check_2d_scalar_field`` idiom) rather than
    # via ``np.iscomplexobj(np.asarray(E_in))``, which raises
    # ``jax.errors.TracerArrayConversionError`` under ``jax.jit`` /
    # ``jax.grad`` -- the end-to-end jit'd path this function exists for.
    # Duck-typing works for NumPy arrays, JAX tracers, CuPy arrays and the
    # typed scalars the JAX cast accepts; a missing or non-complex dtype
    # attribute falls back to the library-default complex dtype.
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
        # dtype is part of the cache key so calls at complex64 and
        # complex128 do not share one compiled XLA kernel; without it the
        # first call to win the race fixes the kernel precision for every
        # subsequent call, whatever the active default (audit L2).
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
            # S2-13 (audit AUDIT_V5_24_2): thread ``dy`` (anamorphic-grid
            # parity with the NumPy path; see the fast-path kernel above).
            E = angular_spectrum_propagate(
                E, float(z), wavelength, dx, dy=dy, bandlimit=bandlimit)

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
            # WP-C1: the slow path calls the same ONE aperture
            # implementation the jit'd kernel above calls, so the two JAX
            # routes and the NumPy chain all answer one element dict the
            # same way, rim included.
            xc = elem.get('xc', 0.0)
            yc = elem.get('yc', 0.0)
            resolved = _resolve_aperture_params(elem)
            if resolved is not None:
                shape, halves = resolved
                edge_kw = _aperture_edge_kwargs(elem)
                if shape == 'circular':
                    ap_params = {'diameter': 2.0 * float(halves[0])}
                elif shape == 'rectangular':
                    ap_params = {'width_x': 2.0 * float(halves[0]),
                                 'width_y': 2.0 * float(halves[1])}
                else:  # annular
                    ap_params = {'outer_diameter': 2.0 * float(halves[0]),
                                 'inner_diameter': 2.0 * float(halves[1])}
                E = apply_aperture(E, dx, shape=shape, params=ap_params,
                                   xc=xc, yc=yc, dy=dy, **edge_kw)
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

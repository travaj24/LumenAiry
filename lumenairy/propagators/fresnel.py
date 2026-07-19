"""
Single-FFT Fresnel and Fraunhofer propagators
=============================================

v5.1.0 Agent C split: extracted from ``propagation.py``.  Contains the
paraxial Fresnel kernel and the far-field Fraunhofer limit.  Both use
the natural FFT output grid (``dx_out = wavelength * z / (N * dx)``);
their MFT counterparts (arbitrary output grid via Bluestein chirp-Z)
live in ``mft.py``.

Public symbols are re-exported by ``propagation.py``.

Author:  Andrew Traverso
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import fft_infra as _state
from .fft_infra import (
    _fft2,
    _ifft2,
    _validate_propagator_inputs,
)

__all__ = [
    'fresnel_propagate',
    'fresnel_tf_propagate',
    'fraunhofer_propagate',
]


def fresnel_tf_propagate(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
) -> np.ndarray:
    """Same-grid Fresnel TRANSFER-FUNCTION step (matched-paraxial ASM).

    Applies the paraxial (Fresnel) transfer function
    ``H = exp(+i k z) * exp(-i pi lambda z (fx**2 + fy**2))`` -- i.e. the
    exact-ASM kernel ``exp(i z sqrt(k**2 - kr**2))`` truncated at the SAME
    quadratic order at which the ``'paraxial'`` thin-lens phase truncates
    the spherical wavefront.  The pair (paraxial lens x this propagator)
    is therefore self-consistent -- aberration-free BY CONSTRUCTION for
    ideal-lens chains, exactly like Zemax POP's pilot-beam re-referencing
    (thin-lens audit 2026-07-18, change 3).  Use it as the matched "ideal
    reference" mode for paraxial-thin-lens relay studies; use exact ASM +
    ``lens_model='stigmatic'`` when you want the exact propagator instead.

    Unlike :func:`fresnel_propagate` (single-FFT, grid-CHANGING, forward
    only), this is a two-FFT SAME-GRID step that composes into chains and
    accepts ``z < 0`` (back-propagation) -- the two properties chains
    need.  Unlike ASM it applies no band-limit and keeps every frequency
    bin (the paraxial kernel has no evanescent cone).

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input field.
    z : float
        Propagation distance [m]; may be negative.  ``z == 0`` returns
        the input unchanged (exact identity).
    wavelength : float
        Wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Output field on the SAME grid.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'fresnel_tf_propagate')
    _validate_propagator_inputs(E_in, z, wavelength, dx, dy,
                                fn_name='fresnel_tf_propagate')
    if dy is None:
        dy = dx
    if z == 0:
        # Exact identity (mirrors ASM's S2-11 z == 0 contract).
        return E_in.copy()

    Ny, Nx = E_in.shape
    k = 2.0 * np.pi / wavelength
    # Cached centred (2*pi*f)^2 vectors; ifftshift the 1-D vectors so H
    # is built directly in natural FFT layout (the S5-8g ASM pattern).
    kx_sq, ky_sq = _state._get_or_make_freq_grids(Ny, Nx, dy, dx, True)
    kx_sq = np.fft.ifftshift(kx_sq)
    ky_sq = np.fft.ifftshift(ky_sq)
    # Paraxial kernel: exp(i k z) * exp(-i z kr^2 / 2k); note
    # pi*lambda*z*f^2 == z * (2 pi f)^2 / (2 k).
    phase = (k * z) - (z / (2.0 * k)) * (ky_sq[:, None] + kx_sq[None, :])
    H = np.exp(1j * phase)
    out = _ifft2(_fft2(np.ascontiguousarray(E_in, dtype=np.complex128)) * H)
    if np.iscomplexobj(E_in) and E_in.dtype != np.complex128:
        out = out.astype(E_in.dtype)
    return out


def fresnel_propagate(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Propagate a field using the single-FFT Fresnel method.

    This is the Fresnel (paraxial) approximation to diffraction.  It uses a
    single FFT and is faster than ASM for long propagation distances, but
    **changes the grid spacing** in the output plane.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input field.

    z : float
        Propagation distance [m].

    wavelength : float
        Wavelength [m].

    dx : float
        Input grid spacing in x [m].

    dy : float, optional
        Input grid spacing in y [m].  Defaults to dx.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Output field.

    dx_out : float
        Output grid spacing in x [m].

    dy_out : float
        Output grid spacing in y [m].

    Notes
    -----
    The output grid spacing is::

        dx_out = wavelength * |z| / (Nx * dx)

    This method is valid when the Fresnel number is moderate::

        N_F = a^2 / (lambda * z) ~ 1

    where *a* is the beam / aperture radius.

    For very short distances (large Fresnel number), use ASM instead.
    For very long distances (small Fresnel number), this becomes equivalent
    to the Fraunhofer approximation.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper (replaces the v4.15.2 inline
    # guard).  Runs FIRST so the user gets a clear, actionable error
    # rather than a downstream AttributeError or silent wrong-axis FFT.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'fresnel_propagate')

    _validate_propagator_inputs(E_in, z, wavelength, dx, dy,
                                fn_name='fresnel_propagate')
    from ..backend import array_namespace, is_jax_array
    xp = array_namespace(E_in)
    is_jax = is_jax_array(E_in)

    if dy is None:
        dy = dx

    # 4.9 fix (audit #3.3): Fresnel is a forward-propagating paraxial
    # kernel; z <= 0 is unphysical here (the FFT direction isn't
    # flipped for back-prop, and the abs(z) in dx_out below mixes
    # signs with the raw-z phase prefactor giving mathematically
    # nonsense output rather than a back-propagated field).  Refuse
    # with a clear error pointing at ASM or RS, both of which do
    # handle back-propagation correctly for the propagating spectrum.
    if z <= 0:
        raise ValueError(
            f"fresnel_propagate: z must be > 0 (got z={z}).  Fresnel "
            f"is a forward-propagating paraxial kernel and does not "
            f"support back-propagation.  Use angular_spectrum_propagate "
            f"or rayleigh_sommerfeld_propagate (both handle negative z) "
            f"if you need to back-propagate a field.")

    Ny, Nx = E_in.shape
    k = 2 * np.pi / wavelength

    # 4.10: honour caller dtype so a complex64 E_in stays complex64
    # through the Fresnel pipeline (pre-4.10 it was silently promoted
    # to complex128 via the python-float `2 * np.pi` and `1j` constants).
    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(_state.DEFAULT_COMPLEX_DTYPE)

    # -- input / output coordinates -----------------------------------------
    # v5.17.x (P2-29): coordinate grids are ALWAYS float64 so the
    # quadratic-phase carrier argument ``k/(2z) * r^2`` (up to ~1e3-1e5
    # rad on large grids) is accumulated at f64 and only the finished
    # carrier is cast to the target dtype (the mft.py f64-carrier-then-
    # cast pattern; same contract as the ASM kernel's mod-2pi fold --
    # see ``set_default_complex_dtype``).  Pre-fix the complex64 path
    # built the grids at float32, accumulating the carrier argument
    # wholly in f32: measured max carrier error 7.7e-4 rad-equivalent
    # vs 4.1e-8 with the f64 carrier (N=2048, dx=2 um, z=5 mm).
    # complex128 inputs are byte-identical to pre-fix.
    #
    # v5.24.4 (audit S2-3): on the JAX backend the ``dtype=float64``
    # request is silently truncated to float32 whenever ``jax_enable_x64``
    # is off (the JAX default), so the carrier argument was again built
    # wholly in f32.  Build the FIELD-INDEPENDENT grids + phase screens
    # on the HOST (``np``) in float64 and only then move the finished
    # (bounded) carriers onto the JAX device -- trace-safe (the screens
    # do not depend on the field, so the field gradient survives).
    _bld = np if is_jax else xp
    x1 = (_bld.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y1 = (_bld.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    X1, Y1 = _bld.meshgrid(x1, y1, indexing='xy')
    dx_out = wavelength * z / (Nx * dx)
    dy_out = wavelength * z / (Ny * dy)
    x2 = (_bld.arange(Nx, dtype=np.float64) - Nx / 2) * dx_out
    y2 = (_bld.arange(Ny, dtype=np.float64) - Ny / 2) * dy_out
    X2, Y2 = _bld.meshgrid(x2, y2, indexing='xy')

    # -- quadratic phase in input plane --------------------------------------
    # v5.17.x (audit P3-56): astype(copy=False) throughout -- the phase
    # screen is freshly allocated by xp.exp (never aliased) and E_in is
    # only *read* by the multiply below, so neither needs the defensive
    # copy.  Pre-fix the bare astype paid two avoidable full-grid copy
    # passes (~4 GB transient each at 16384^2 complex128), partially
    # undoing the v5.17.0 lifetime hygiene in this function.
    # Byte-identical output.
    phase_in = _bld.exp(1j * k / (2 * z) * (X1**2 + Y1**2)).astype(
        target_cdtype, copy=False)
    if is_jax:
        phase_in = xp.asarray(phase_in)
    E_mod = E_in.astype(target_cdtype, copy=False) * phase_in
    # v5.17.0 lifetime hygiene (byte-identical): the input-plane grids and
    # phase screen are consumed -- free before the FFT so they don't ride
    # through it.  (JAX arrays are immutable/functional; del is a no-op
    # name-drop there, which is fine.)
    del X1, Y1, phase_in

    # -- FFT -- use _fft for NumPy/CuPy fast path; jnp.fft for JAX -----------
    if is_jax:
        E_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(E_mod)))
    else:
        E_fft = np.fft.fftshift(_fft2(np.fft.ifftshift(E_mod)))
    del E_mod

    # -- quadratic phase in output plane + prefactor -------------------------
    prefactor = (_bld.exp(1j * k * z) / (1j * wavelength * z)
                 * _bld.exp(1j * k / (2 * z) * (X2**2 + Y2**2))
                 * dx * dy)
    # v5.17.x (audit P3-56): copy=False -- prefactor is freshly built above.
    prefactor = prefactor.astype(target_cdtype, copy=False)
    if is_jax:
        prefactor = xp.asarray(prefactor)
    del X2, Y2

    E_out = prefactor * E_fft

    return E_out, dx_out, dy_out


def fraunhofer_propagate(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Propagate a field to the Fraunhofer (far-field) diffraction pattern.

    This is the far-field limit of the Fresnel propagator, valid when the
    Fresnel number is small:

        N_F = a^2 / (lambda * z) << 1

    where ``a`` is the characteristic aperture/beam radius. In practice this
    means ``z`` must be large compared to ``a^2 / lambda``. For smaller
    distances, use :func:`fresnel_propagate` or :func:`angular_spectrum_propagate`.

    Mathematically, the Fraunhofer integral reduces to a single Fourier
    transform of the input field with a quadratic phase and scaling prefactor::

        E(x2, y2) = [exp(i*k*z) / (i*lambda*z)]
                    * exp(i*k/(2z) * (x2^2 + y2^2))
                    * FFT{E(x1, y1)} * dx*dy

    Parameters
    ----------
    E_in : ndarray (complex, NxN)
        Input field.
    z : float
        Propagation distance [m].
    wavelength : float
        Free-space wavelength [m].
    dx : float
        Input grid spacing in x [m].
    dy : float, optional
        Input grid spacing in y [m]. Defaults to dx.

    Returns
    -------
    E_out : ndarray (complex, NxN)
        Field in the far-field plane.
    dx_out : float
        Output grid spacing in x [m] = wavelength * |z| / (N * dx).
    dy_out : float
        Output grid spacing in y [m] = wavelength * |z| / (N * dy).

    Notes
    -----
    The output grid spacing is the same as :func:`fresnel_propagate`:

        dx_out = wavelength * |z| / (N * dx)

    The difference from Fresnel is that Fraunhofer drops the input-plane
    quadratic phase (assumed to be negligible at large z), so there is only
    one FFT and one scalar multiplication. It is slightly faster and more
    numerically stable than Fresnel at large distances.

    For focal-plane computation of a converging beam (e.g. after a lens),
    Fraunhofer is the standard approach: place the input field at the lens,
    set z = focal length, and the output is the focal-plane field.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper (replaces the v4.15.2 inline
    # guard).
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'fraunhofer_propagate')

    _validate_propagator_inputs(E_in, z, wavelength, dx, dy,
                                fn_name='fraunhofer_propagate')
    from ..backend import array_namespace, is_jax_array
    xp = array_namespace(E_in)
    is_jax = is_jax_array(E_in)

    if dy is None:
        dy = dx

    # 4.9 fix (audit #3.3): Fraunhofer is the far-field limit of
    # Fresnel and inherits the same forward-only restriction.  See
    # the matching check in ``fresnel_propagate``.
    if z <= 0:
        raise ValueError(
            f"fraunhofer_propagate: z must be > 0 (got z={z}).  "
            f"Fraunhofer is the far-field limit of Fresnel and is "
            f"forward-only.  Use angular_spectrum_propagate or "
            f"rayleigh_sommerfeld_propagate for back-propagation.")

    Ny, Nx = E_in.shape
    k = 2 * np.pi / wavelength

    # 4.10: honour caller dtype (see fresnel_propagate for rationale).
    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(_state.DEFAULT_COMPLEX_DTYPE)

    # Output grid spacing
    dx_out = wavelength * z / (Nx * dx)
    dy_out = wavelength * z / (Ny * dy)

    # Output coordinates
    # v5.17.x (P2-29): always float64 -- the quadratic-phase carrier
    # argument is accumulated at f64 and cast to the target dtype only
    # after ``exp`` (see the matching comment in ``fresnel_propagate``).
    # v5.24.4 (audit S2-3): on JAX with ``jax_enable_x64`` off the
    # ``dtype=float64`` request truncates to float32, so build the
    # FIELD-INDEPENDENT output grid + carrier on the HOST in f64 and
    # then move the finished (bounded) prefactor onto the device.
    _bld = np if is_jax else xp
    x2 = (_bld.arange(Nx, dtype=np.float64) - Nx / 2) * dx_out
    y2 = (_bld.arange(Ny, dtype=np.float64) - Ny / 2) * dy_out
    X2, Y2 = _bld.meshgrid(x2, y2, indexing='xy')

    # Single FFT of the input field
    E_cast = E_in.astype(target_cdtype) if E_in.dtype != target_cdtype else E_in
    if is_jax:
        E_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(E_cast)))
    else:
        E_fft = np.fft.fftshift(_fft2(np.fft.ifftshift(E_cast)))

    # Output quadratic phase + prefactor
    prefactor = (_bld.exp(1j * k * z) / (1j * wavelength * z)
                 * _bld.exp(1j * k / (2 * z) * (X2**2 + Y2**2))
                 * dx * dy)
    # v5.17.x (audit P3-56): copy=False -- prefactor is freshly built above.
    prefactor = prefactor.astype(target_cdtype, copy=False)
    if is_jax:
        prefactor = xp.asarray(prefactor)

    E_out = prefactor * E_fft

    return E_out, dx_out, dy_out

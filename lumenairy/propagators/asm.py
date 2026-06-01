"""
Angular Spectrum Method (ASM) propagators
=========================================

v5.1.0 Agent C split: extracted from ``propagation.py``.  Contains the
exact-band-limited ASM kernel, the tilted / off-axis ASM, the batched
ASM variant, the shared ``_build_asm_H_square`` helper, and the
``apply_fresnel_curvature`` curvature-convention conversion utility.

All public symbols are re-exported by ``propagation.py`` for back-
compatibility with the pre-v5.1.0 import paths.

Author:  Andrew Traverso
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from . import fft_infra as _state
from .fft_infra import (
    CUPY_AVAILABLE,
    _fft2,
    _fft2_nd,
    _get_or_make_bandlimit,
    _get_or_make_freq_grids,
    _h_cache_lookup,
    _h_cache_store,
    _ifft2,
    _ifft2_nd,
    _is_cupy_array,
    _validate_propagator_inputs,
)

__all__ = [
    'angular_spectrum_propagate',
    'angular_spectrum_propagate_tilted',
    # v5.1.0 (Wave-4 integration): ``angular_spectrum_propagate_batch``
    # is a power-user 3-D-input variant reached via
    # ``lumenairy.propagators.propagation.angular_spectrum_propagate_batch``;
    # never been in the top-level ``lumenairy.__all__`` so leaving it
    # in this submodule's ``__all__`` would create a V9-walker
    # symmetry violation.  Module-attribute-accessible.
    'apply_fresnel_curvature',
    '_build_asm_H_square',
]


def _build_asm_H_square(
    N,
    dx,
    z,
    wavelength,
    dtype=None,
    bandlimit=True,
):
    """Build a square (N x N) band-limited Angular-Spectrum transfer
    function on the canonical centered frequency grid.

    This is the single source of truth for the centered ASM ``H``
    construction used by:

    * :func:`angular_spectrum_propagate` (square-grid path / JAX path
      / one-shot fallback when chunking is not needed).
    * :func:`lumenairy.analysis.detector.shack_hartmann` (per-lenslet
      sub-aperture propagation).

    Conventions
    -----------
    * Output is **centered** (not ``fftshift``-ed), matching the
      ``E_fft = fftshift(fft2(ifftshift(E))) ; E_out = fftshift(
      ifft2(ifftshift(E_fft * H)))`` propagation idiom both call sites
      use.
    * Frequency grid is ``(arange(N) - N/2) / (N * dx)``, i.e. the
      same centered convention as :func:`_get_or_make_freq_grids`
      with square ``dy == dx``.
    * Evanescent modes (``kz_sq <= 0``) are zeroed.
    * When ``bandlimit`` is True and ``z != 0`` the
      Matsushima 1-D mask (``|f| < L / (2*lambda*|z|)``) is applied
      as the outer product of the per-axis masks.
    * ``z == 0`` short-circuits to ``H = 1`` (with bandlimit ignored,
      matching the canonical propagator).

    Parameters
    ----------
    N : int
        Square grid size (Ny == Nx == N).
    dx : float
        Pixel pitch (m).  ``dy == dx`` is assumed.
    z : float
        Propagation distance (m).  May be negative for back-propagation.
    wavelength : float
        Vacuum wavelength (m).
    dtype : numpy dtype, optional
        Target complex dtype.  Defaults to ``np.complex128``.  Real
        dtypes are promoted to ``np.complex128``.
    bandlimit : bool, default True
        Apply the Matsushima 1-D bandlimit mask.

    Returns
    -------
    H : ndarray, shape (N, N), dtype as requested
        The centered ASM transfer function.

    Notes
    -----
    Numerical equivalence to the inline path is bit-exact for
    matching ``N``, ``dx``, ``z``, ``wavelength``, and ``bandlimit``
    arguments (same arithmetic; no caching / chunking detour).
    """
    if dtype is None or not np.issubdtype(dtype, np.complexfloating):
        dtype = np.complex128
    N = int(N)
    k = 2.0 * np.pi / wavelength
    fx = (np.arange(N, dtype=np.float64) - N / 2) / (N * dx)
    fy = fx  # square sub-aperture (dy == dx)
    kx_sq = (2 * np.pi * fx) ** 2
    ky_sq = (2 * np.pi * fy) ** 2
    kz_sq = k ** 2 - kx_sq[None, :] - ky_sq[:, None]
    prop = kz_sq > 0
    kz = np.where(prop, np.sqrt(np.where(prop, kz_sq, 0.0)), 0.0)
    H = np.where(prop, np.exp(1j * kz * z), 0.0).astype(dtype)
    if bandlimit and z != 0:
        L = N * dx
        f_max = L / (2 * wavelength * abs(z))
        bl_x = np.abs(fx) < f_max
        bl_y = np.abs(fy) < f_max
        mask = bl_x[None, :] & bl_y[:, None]
        H = H * mask.astype(dtype)
    return H


def angular_spectrum_propagate(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    bandlimit: bool = True,
    return_transfer_function: bool = False,
    use_gpu: bool = False,
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Propagate an optical field using the Angular Spectrum Method (ASM).

    This function propagates a 2-D complex electric field through free space
    using the exact transfer function (no paraxial approximation).

    Parameters
    ----------
    E_in : ndarray (complex)
        Input electric field, shape (Ny, Nx).  Can be a NumPy or CuPy array.

    z : float
        Propagation distance in meters.
        Positive z = forward propagation (away from source).
        Negative z = backward propagation (toward source).

    wavelength : float
        Optical wavelength in meters (e.g. 1.31e-6 for 1310 nm).

    dx : float
        Grid spacing in x-direction in meters (e.g. 1e-6 for 1 um).

    dy : float, optional
        Grid spacing in y-direction in meters.  If None, assumes dy = dx.

    bandlimit : bool, default True
        If True, applies band-limiting to suppress Fresnel aliasing.
        The band-limit cutoff per axis is:  f_max = L / (2 * lambda * |z|).
        Recommended for large propagation distances.

    return_transfer_function : bool, default False
        If True, also returns the transfer function H.

    use_gpu : bool, default False
        If True and CuPy is available, performs computation on GPU.
        If *E_in* is already a CuPy array, GPU is used automatically.

    verbose : bool, default False
        If True, prints diagnostic information.

    Returns
    -------
    E_out : ndarray (complex)
        Propagated electric field, same shape and array type as *E_in*.

    H : ndarray (complex), optional
        Transfer function (only returned when *return_transfer_function=True*).

    Notes
    -----
    Sampling requirements for accurate results:

    1. ``dx < lambda / 2`` -- Nyquist for propagating waves.
    2. ``L > 2 * lambda * z / d_min`` -- avoids Fresnel aliasing, where
       L = N * dx is the grid extent and d_min is the smallest feature size
       to be resolved.

    Memory: approximately 3x the size of the input array (E_in, E_fft, H).

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.propagation import angular_spectrum_propagate
    >>>
    >>> N = 512
    >>> dx = 1e-6                    # 1 um grid spacing
    >>> wavelength = 1.31e-6         # 1310 nm
    >>> x = (np.arange(N) - N/2) * dx
    >>> X, Y = np.meshgrid(x, x)
    >>> sigma = 10e-6                # 10 um beam waist
    >>> E_in = np.exp(-(X**2 + Y**2) / (2 * sigma**2)).astype(complex)
    >>>
    >>> E_out = angular_spectrum_propagate(E_in, z=1e-3,
    ...                                    wavelength=wavelength, dx=dx)
    >>> print(f"Input power:  {np.sum(np.abs(E_in)**2):.4f}")
    >>> print(f"Output power: {np.sum(np.abs(E_out)**2):.4f}")

    References
    ----------
    [1] Goodman, J.W. "Introduction to Fourier Optics" (3rd ed.), Ch. 3-4.
    [2] Matsushima, K. and Shimobaba, T. (2009). "Band-limited angular
        spectrum method for numerical simulation of free-space propagation
        in far and near fields." Opt. Express 17(22): 19662-19673.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  v4.15.2 inlined the guard
    # here; v4.15.3 routes through the helper so future entry points
    # can't be added unguarded.  Runs FIRST (before any input
    # validation or backend dispatch) so the user gets a clear,
    # actionable error rather than a downstream AttributeError or a
    # silent wrong-axis FFT.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'angular_spectrum_propagate')

    _validate_propagator_inputs(E_in, z, wavelength, dx, dy,
                                fn_name='angular_spectrum_propagate')

    # -- array library selection (NumPy / CuPy / JAX) ----------------------
    # JAX arrays bypass the chunked H construction and the host cache;
    # they take a one-shot all-NxN H and stay in the input backend.
    from ..backend import is_jax_array
    is_jax = is_jax_array(E_in)
    if is_jax:
        import jax.numpy as _jnp
        xp = _jnp
    elif CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = _state.cp
        if not _is_cupy_array(E_in):
            E_in = _state.cp.asarray(E_in)
    else:
        xp = np
        if _is_cupy_array(E_in):
            E_in = E_in.get()  # CuPy -> NumPy when GPU not requested

    Ny, Nx = E_in.shape

    if dy is None:
        dy = dx

    # -- wave parameters -----------------------------------------------------
    k = 2 * np.pi / wavelength

    # Target complex dtype for the transfer function and the output.
    # Inferred from E_in so the caller controls precision by the dtype of
    # the field they pass in.  Non-complex input (e.g. float arrays used
    # in examples) falls back to DEFAULT_COMPLEX_DTYPE.
    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(_state.DEFAULT_COMPLEX_DTYPE)
    target_fdtype = np.float32 if target_cdtype == np.complex64 else np.float64

    # 3.2.14 H cache
    # Geometry signature.  Hits return the previously-built H without
    # re-running the chunked kernel construction (~30-50% of total
    # ASM time on 2k+ grids).  CuPy device arrays and JAX traced
    # arrays are kept out of the cache (host-side dict can't safely
    # retain device pointers / traced objects).
    h_key = None
    H = None
    if xp is np:
        # 4.10: add 'ASM' tag string to the cache key so plain-ASM
        # entries are guaranteed disjoint from ASM_TILTED / ASM_MFT /
        # RS / SAS even if those keys ever evolve to the same tuple
        # length.  Defensive future-proofing.
        h_key = (int(Ny), int(Nx), float(dy), float(dx),
                 float(wavelength), float(z), bool(bandlimit),
                 np.dtype(target_cdtype).str, 'ASM')
        H = _h_cache_lookup(h_key)

    if H is None and is_jax:
        # JAX path: build H in one shot (no chunking, no in-place
        # writes; jax.numpy is functional / immutable).
        fx = (xp.arange(Nx, dtype=xp.float64) - Nx / 2) / (Nx * dx)
        fy = (xp.arange(Ny, dtype=xp.float64) - Ny / 2) / (Ny * dy)
        kx_sq = (2 * float(np.pi) * fx) ** 2
        ky_sq = (2 * float(np.pi) * fy) ** 2
        kz_sq = k ** 2 - kx_sq[None, :] - ky_sq[:, None]
        prop = kz_sq > 0
        kz = xp.where(prop, xp.sqrt(xp.where(prop, kz_sq, 0.0)), 0.0)
        H = xp.where(prop, xp.exp(1j * kz * z), 0.0).astype(target_cdtype)
        if bandlimit and z != 0:
            Lx = Nx * dx
            Ly = Ny * dy
            fx_max = Lx / (2 * wavelength * abs(z))
            fy_max = Ly / (2 * wavelength * abs(z))
            bl_x = xp.abs(fx) < fx_max
            bl_y = xp.abs(fy) < fy_max
            mask = bl_x[None, :] & bl_y[:, None]
            H = H * mask.astype(target_cdtype)
        # v5.5.3: store H in NATURAL (un-shifted) FFT layout so the per-call
        # propagation folds away the two spectrum-domain shifts (4 -> 2 shifts).
        H = xp.fft.ifftshift(H)

    if H is None:
        # Spatial-frequency squared vectors (cached on numpy path).
        kx_sq, ky_sq = _get_or_make_freq_grids(Ny, Nx, dy, dx, xp is np)
        if bandlimit and z != 0:
            bl_x, bl_y = _get_or_make_bandlimit(
                Ny, Nx, dy, dx, wavelength, abs(z), xp is np)
        else:
            bl_x = bl_y = None

        # Chunked H construction, sized to fit a small slice of RAM.
        from ..memory import get_ram_budget
        ram = get_ram_budget()
        row_cost = 3 * Nx * 16   # bytes per row of workspace (complex128)
        if row_cost > 0:
            max_chunk = max(1, int(ram * 0.1 / row_cost))
        else:
            max_chunk = Ny
        chunk = min(Ny, max_chunk)

        H = xp.empty((Ny, Nx), dtype=target_cdtype)
        kept_count = 0
        for j0 in range(0, Ny, chunk):
            j1 = min(Ny, j0 + chunk)
            # kz_sq is float64 regardless of target dtype to keep the
            # huge kernel argument (kz * z up to ~1e6 rad) accurate.
            kz_sq_c = k**2 - kx_sq[None, :] - ky_sq[j0:j1, None]
            prop = kz_sq_c > 0
            kz_c = xp.where(prop, xp.sqrt(xp.maximum(kz_sq_c, 0)), 0)
            if target_cdtype == np.complex128:
                H_c = xp.where(prop, xp.exp(1j * kz_c * z), 0)
            else:
                # complex64 path: fold phase mod 2*pi in float64
                # BEFORE casting to float32 so the float32 precision
                # floor doesn't inject speckle-like noise.
                phase = xp.mod(kz_c * z, 2.0 * np.pi)
                c = xp.cos(phase).astype(target_fdtype)
                s = xp.sin(phase).astype(target_fdtype)
                H_c = xp.empty((j1 - j0, Nx), dtype=target_cdtype)
                H_c.real[:] = xp.where(prop, c, target_fdtype(0))
                H_c.imag[:] = xp.where(prop, s, target_fdtype(0))
            if bl_x is not None:
                bl_mask = bl_x[None, :] & bl_y[j0:j1, None]
                H_c *= bl_mask
                if verbose:
                    kept_count += int(xp.sum(bl_mask))
            H[j0:j1, :] = H_c

        if verbose and bl_x is not None:
            kept_frac = kept_count / (Nx * Ny)
            print(f"  Band-limiting: keeping {kept_frac*100:.1f}% of spectrum")
        if verbose:
            print(f"  ASM propagation: z = {z*1e3:.3f} mm  "
                  f"(H cache miss, built in {chunk}-row chunks)")
            print(f"  Grid: {Ny}x{Nx}, dx={dx*1e6:.3f} um, dy={dy*1e6:.3f} um")
            print(f"  Wavelength: {wavelength*1e9:.1f} nm")
        # v5.5.3: cache H in NATURAL (un-shifted) FFT layout (see below).
        H = xp.fft.ifftshift(H)
        # Store under the numpy key only.  The cached H is read-only
        # in normal use; we don't deep-copy on lookup, so callers must
        # not mutate it in place.
        if h_key is not None:
            _h_cache_store(h_key, H)
    elif verbose:
        print(f"  ASM propagation: z = {z*1e3:.3f} mm  (H cache HIT)")

    # -- propagate: E_out = IFFT{ FFT{E_in} * H } ---------------------------
    # H is stored NATURAL-layout, so the two spectrum-domain shifts fold away:
    #   fftshift(ifft2(ifftshift( fftshift(fft2(ifftshift(E)))*H_centred )))
    #   == fftshift(ifft2(           fft2(ifftshift(E))      *H_natural   ))
    # (ifftshift distributes over the elementwise product; ifftshift.fftshift =
    # id).  Algebraically EXACT for any N, even or odd -- 4 shifts -> 2.
    if xp is np:
        E_out = np.fft.fftshift(_ifft2(_fft2(np.fft.ifftshift(E_in)) * H))
    else:
        E_out = xp.fft.fftshift(
            xp.fft.ifft2(xp.fft.fft2(xp.fft.ifftshift(E_in)) * H))

    if return_transfer_function:
        # 4.10: return a copy so a caller that does ``E_out, H = ...(
        # return_transfer_function=True)`` then ``H *= mask`` cannot mutate the
        # cached entry.  Re-centre H (fftshift) so the returned transfer
        # function keeps the historical CENTERED-spectrum contract callers use.
        H_returned = xp.fft.fftshift(H)
        H_returned = (H_returned.copy() if hasattr(H_returned, 'copy')
                      else xp.asarray(H_returned))
        return E_out, H_returned
    else:
        return E_out


def apply_fresnel_curvature(
    E: np.ndarray,
    dx: float,
    wavelength: float,
    R: float,
    sign: int = +1,
    dy: Optional[float] = None,
) -> np.ndarray:
    """Apply (or remove) a Fresnel quadratic phase ``exp(i*sign*k*r^2/(2R))``.

    Used to convert between phase conventions when comparing fields
    produced by different libraries.

    Background
    ----------
    Lumenairy's propagators (and the standard Fresnel/ASM family --
    LightPipes, prysm, diffractio, POPPy, Zemax POP) keep the **full
    physical phase** at the output plane.  Some ray-trace-rooted
    aberration-analysis tools (notably OPDPy and Zemax wavefront
    operands like ``OPDX``) instead store the **chief-relative OPD**,
    which implicitly subtracts the natural Gaussian-beam wavefront
    curvature at the image plane.

    The two conventions differ by exactly a Fresnel quadratic phase
    ``exp(i*k*r^2/(2*R))`` with ``R = v - f`` for a thin-lens
    imager (image distance minus focal length).

    Use this function to round-trip between conventions:

    .. code-block:: python

        # Convert OPDPy / Zemax-OPD output to Lumenairy / LightPipes:
        E_absolute = apply_fresnel_curvature(
            E_chief_relative, dx, wavelength, R=v - f, sign=+1)

        # Convert Lumenairy / LightPipes output to chief-relative:
        E_chief_relative = apply_fresnel_curvature(
            E_absolute, dx, wavelength, R=v - f, sign=-1)

    For multi-element systems, ``R`` is the wavefront radius of
    curvature at the image plane predicted by Gaussian-beam ABCD
    propagation -- see Saleh & Teich, *Fundamentals of Photonics*,
    Section 3.1.

    Parameters
    ----------
    E : ndarray, complex 2D
        Input field.  Grid is assumed to be centred on the chief image
        point (the centre pixel is at coordinate ``(0, 0)``, with the
        same half-pixel offset convention as
        :func:`angular_spectrum_propagate`).
    dx : float
        Pixel pitch in the x-direction (metres).
    wavelength : float
        Wavelength (metres).
    R : float
        Wavefront radius of curvature (metres).  For a thin-lens
        imager, ``R = image_distance - focal_length``.
    sign : int, default ``+1``
        ``+1`` adds the curvature (chief-relative -> absolute).
        ``-1`` removes the curvature (absolute -> chief-relative).
    dy : float, optional
        Pixel pitch in y.  Defaults to ``dx``.

    Returns
    -------
    E_out : ndarray, complex
        Same shape and dtype as ``E``, with the Fresnel curvature
        multiplied (or divided) in.

    See also
    --------
    Wiki: "Phase conventions and inter-library comparison"
    """
    if dy is None:
        dy = dx
    # R = 0 / inf / NaN is treated as a no-op so multi-element
    # prescriptions where v-f is ill-defined (e.g. an afocal section)
    # can pass through without curvature.  This is documented behaviour
    # locked in by the test suite.
    if R == 0 or not np.isfinite(R):
        return E.copy()
    if sign not in (+1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    Ny, Nx = E.shape
    # 4.10: drop the spurious +0.5 half-pixel offset.  Every other
    # propagator in this file builds coordinates as (arange(N) - N/2)*dx
    # (no +0.5).  The mismatch produced a half-pixel walk-off in the
    # curvature centre relative to the propagated field grid, visible
    # as a small coma-like residual in OPDPy cross-checks.
    ax_x = (np.arange(Nx) - Nx / 2) * dx
    ax_y = (np.arange(Ny) - Ny / 2) * dy
    Y, X = np.meshgrid(ax_y, ax_x, indexing='ij')
    r2 = X * X + Y * Y
    k = 2.0 * np.pi / wavelength
    return E * np.exp(sign * 1j * k * r2 / (2.0 * R))


def angular_spectrum_propagate_batch(
    E_stack: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    bandlimit: bool = True,
    use_gpu: bool = False,
) -> np.ndarray:
    """ASM propagation of a stack of fields ``(B, Ny, Nx)`` in one
    fused FFT pair (3.2.14).

    All ``B`` fields share the same grid + wavelength + propagation
    distance, so the transfer function ``H`` is built once (reusing
    the H cache) and broadcast across the batch.  Two batched FFTs
    (forward + inverse, axes ``(-2, -1)``) replace ``2*B`` separate
    2-D FFTs, which on JonesField (Ex, Ey) is ~30-60% wall-clock
    faster than calling :func:`angular_spectrum_propagate` per
    component.

    Parameters
    ----------
    E_stack : ndarray, complex, shape (B, Ny, Nx)
        Input field stack.  ``B`` must be at least 1.
    z, wavelength, dx, dy, bandlimit, use_gpu
        Same semantics as :func:`angular_spectrum_propagate`.

    Returns
    -------
    E_out : ndarray, complex, shape (B, Ny, Nx)
        Propagated stack, same dtype + array library as input.
    """
    if E_stack.ndim != 3:
        raise ValueError(
            f"angular_spectrum_propagate_batch: input must be 3-D "
            f"(B, Ny, Nx), got shape {E_stack.shape}.")
    # Validate using a representative 2-D slice; the batched call has
    # the same (z, wavelength, dx, dy) constraints as the scalar
    # propagator, so reuse the helper.
    _validate_propagator_inputs(E_stack[0], z, wavelength, dx, dy,
                                fn_name='angular_spectrum_propagate_batch')

    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_stack)):
        xp = _state.cp
        if not _is_cupy_array(E_stack):
            E_stack = _state.cp.asarray(E_stack)
    else:
        xp = np
        if _is_cupy_array(E_stack):
            E_stack = E_stack.get()

    B, Ny, Nx = E_stack.shape
    if dy is None:
        dy = dx

    if xp.iscomplexobj(E_stack):
        target_cdtype = E_stack.dtype
    else:
        target_cdtype = np.dtype(_state.DEFAULT_COMPLEX_DTYPE)
        E_stack = E_stack.astype(target_cdtype)

    # Reuse the H cache from the scalar propagator: build H by
    # delegating to the scalar function on a tiny ``Ny x Nx`` field
    # of the right dtype with ``return_transfer_function=True``.  H
    # is read-only after construction so it is safe to reuse across
    # the batch.
    _proxy = xp.empty((Ny, Nx), dtype=target_cdtype)
    _, H = angular_spectrum_propagate(
        _proxy, z, wavelength, dx, dy=dy, bandlimit=bandlimit,
        return_transfer_function=True, use_gpu=(xp is not np),
    )

    # Single batched FFT pair across the last two axes.  pyFFTW's
    # multi-slot plan cache (also new in 3.2.14) keys on the full
    # shape including the batch dimension, so a 3-D plan is built on
    # the first call and reused thereafter.  The numpy / scipy
    # fallback paths handle 3-D input natively via ``fft2`` over the
    # last two axes.
    if xp is np:
        # Use scipy.fft for ND batched (workers parameter), pyFFTW
        # plan cache picks up the (B, Ny, Nx) shape automatically via
        # ``_fft2`` if the array is large enough.
        E_fft = xp.fft.fftshift(
            _fft2_nd(xp.fft.ifftshift(E_stack, axes=(-2, -1))),
            axes=(-2, -1))
        E_out = xp.fft.fftshift(
            _ifft2_nd(xp.fft.ifftshift(E_fft * H[None, :, :],
                                        axes=(-2, -1))),
            axes=(-2, -1))
    else:
        E_fft = xp.fft.fftshift(
            xp.fft.fft2(xp.fft.ifftshift(E_stack, axes=(-2, -1)),
                        axes=(-2, -1)),
            axes=(-2, -1))
        E_out = xp.fft.fftshift(
            xp.fft.ifft2(xp.fft.ifftshift(E_fft * H[None, :, :],
                                            axes=(-2, -1)),
                          axes=(-2, -1)),
            axes=(-2, -1))
    return E_out


def angular_spectrum_propagate_tilted(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    bandlimit: bool = True,
    *,
    tilt_x_deg: Optional[float] = None,
    tilt_y_deg: Optional[float] = None,
) -> np.ndarray:
    """
    ASM propagation with a carrier tilt (off-axis propagation).

    Propagates the field while accounting for a mean propagation direction
    that is tilted relative to the optical axis.  This is useful for:

    - Beams arriving at an angle
    - Propagation after a prism or wedge
    - Off-axis portions of a wide-field system

    The tilt is handled by shifting the frequency-domain transfer function,
    which is equivalent to propagating the field in a tilted reference frame.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input electric field.

    z : float
        Propagation distance [m] along the tilted axis.

    wavelength : float
        Optical wavelength [m].

    dx : float
        Grid spacing in x [m].

    dy : float, optional
        Grid spacing in y [m].  Defaults to dx.

    tilt_x, tilt_y : float, default 0.0
        Tilt angles [radians] of the propagation direction relative to the
        z-axis.  The beam propagates at angle (tilt_x, tilt_y) from the
        optical axis.

    bandlimit : bool, default True
        Apply band-limiting to avoid aliasing.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Propagated electric field.

    Notes
    -----
    The method removes the carrier frequency (tilt) before propagation,
    then restores it afterwards.  This keeps the field well-centred on
    the grid even for large tilt angles, avoiding grid walk-off.

    The carrier spatial frequencies are::

        fx0 = sin(tilt_x) / wavelength
        fy0 = sin(tilt_y) / wavelength

    The field is demodulated as::

        E_demod = E_in * exp(-i * 2*pi * (fx0*X + fy0*Y))

    propagated with a shifted transfer function, then remodulated::

        E_out = E_prop * exp(+i * 2*pi * (fx0*X + fy0*Y))

    For ``tilt_x = tilt_y = 0`` this reduces to standard ASM propagation.

    4.7+: convenience kwargs ``tilt_x_deg`` / ``tilt_y_deg`` accept the
    angle in degrees and take precedence over the radian forms when
    supplied.  These are part of the broader push toward ``_deg`` as
    the canonical user-facing angle unit (see the polish-pass note in
    :ref:`Release Notes`).
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper -- siblings missed by the
    # v4.15.2 closure now share the same first-line guard.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'angular_spectrum_propagate_tilted')

    _validate_propagator_inputs(E_in, z, wavelength, dx, dy,
                                fn_name='angular_spectrum_propagate_tilted')
    if tilt_x_deg is not None:
        tilt_x = float(np.radians(tilt_x_deg))
    if tilt_y_deg is not None:
        tilt_y = float(np.radians(tilt_y_deg))
    if dy is None:
        dy = dx

    Ny, Nx = E_in.shape

    # -- carrier spatial frequencies from tilt angles ------------------------
    fx0 = np.sin(tilt_x) / wavelength
    fy0 = np.sin(tilt_y) / wavelength

    # Shortcut: no tilt -> fall back to standard ASM
    if abs(fx0) < 1e-15 and abs(fy0) < 1e-15:
        return angular_spectrum_propagate(E_in, z, wavelength, dx, dy,
                                          bandlimit=bandlimit)

    # Target complex dtype (matches angular_spectrum_propagate / RS).
    if np.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(_state.DEFAULT_COMPLEX_DTYPE)

    # -- spatial coordinate grids (carrier; per-call) ------------------------
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)

    # -- demodulate: remove carrier tilt -------------------------------------
    carrier = np.exp(-1j * 2 * np.pi * (fx0 * X + fy0 * Y))
    E_demod = E_in * carrier

    # H cache (NumPy backend)
    # The shifted transfer function depends on (Ny, Nx, dy, dx,
    # wavelength, z, fx0, fy0, bandlimit, dtype).  fx0/fy0 are the
    # tilt-derived carrier frequencies and they fully encode the
    # propagation-direction shift, so the cache key handles arbitrary
    # tilt angles without needing tilt_x / tilt_y in the key directly.
    # 'ASM_TILTED' tag keeps these entries disjoint from plain-ASM ones.
    h_key = (int(Ny), int(Nx), float(dy), float(dx),
             float(wavelength), float(z),
             float(fx0), float(fy0),
             bool(bandlimit),
             np.dtype(target_cdtype).str, 'ASM_TILTED')
    H = _h_cache_lookup(h_key)

    if H is None:
        # -- shifted transfer function -------------------------------------
        # kz is evaluated at (fx + fx0, fy + fy0) so the baseband field
        # propagates with the correct kz for each plane-wave component.
        k = 2 * np.pi / wavelength
        dfx = 1.0 / (Nx * dx)
        dfy = 1.0 / (Ny * dy)
        fx = (np.arange(Nx) - Nx / 2) * dfx
        fy = (np.arange(Ny) - Ny / 2) * dfy
        FX, FY = np.meshgrid(fx, fy)

        FX_shifted = FX + fx0
        FY_shifted = FY + fy0
        kx = 2 * np.pi * FX_shifted
        ky = 2 * np.pi * FY_shifted

        kz_sq = k**2 - kx**2 - ky**2
        kz = np.where(kz_sq > 0, np.sqrt(np.maximum(kz_sq, 0)), 0)
        H = np.exp(1j * kz * z)
        H = np.where(kz_sq > 0, H, 0)

        # -- band-limiting on the ORIGINAL-FRAME spectrum ----------------
        # Matsushima bounds the FREQUENCY OF THE CHIRP in the angular-
        # spectrum kernel, which depends on the original (non-shifted)
        # frequency (FX + fx0, FY + fy0): that's where the chirp's
        # phase-derivative is taken.  The H built above is also
        # evaluated at the shifted arguments, so the mask must use
        # FX_shifted = FX + fx0 (and FY + fy0) -- otherwise it clips the
        # *baseband* (around FX=0) and lets through the actual aliasing-
        # prone high-(FX+fx0) bands.  Pre-4.10 used `|FX| < fx_max`,
        # which for any non-trivial tilt killed the baseband DC and
        # zeroed the propagated field.
        if bandlimit and z != 0:
            Lx = Nx * dx
            Ly = Ny * dy
            fx_max = Lx / (2 * wavelength * abs(z))
            fy_max = Ly / (2 * wavelength * abs(z))
            H = np.where((np.abs(FX_shifted) < fx_max) &
                          (np.abs(FY_shifted) < fy_max), H, 0)

        if H.dtype != target_cdtype:
            H = H.astype(target_cdtype)

        _h_cache_store(h_key, H)

    # -- propagate baseband with shifted transfer function -------------------
    E_fft = np.fft.fftshift(_fft2(np.fft.ifftshift(E_demod)))
    E_prop = np.fft.fftshift(_ifft2(np.fft.ifftshift(E_fft * H)))

    # -- remodulate: restore carrier tilt ------------------------------------
    E_out = E_prop * np.conj(carrier)

    return E_out

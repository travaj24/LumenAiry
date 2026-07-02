"""
Matrix-Fourier-transform (MFT / Bluestein) propagators
======================================================

v5.1.0 Agent C split: extracted from ``propagation.py``.  Contains the
three propagators that evaluate the diffraction integral on an
arbitrary user-specified output grid via the Bluestein chirp-Z
transform (instead of the natural FFT grid):

* :func:`fresnel_propagate_mft`    -- single-FFT paraxial Fresnel.
* :func:`fraunhofer_propagate_mft` -- far-field Fraunhofer limit.
* :func:`angular_spectrum_propagate_mft` -- exact band-limited ASM.

Also re-houses :func:`resample_field`, the bridge utility for switching
between propagation methods that use different grid spacings.

Public symbols are re-exported by ``propagation.py``.

Author:  Andrew Traverso
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import fft_infra as _state
from .fft_infra import (
    CUPY_AVAILABLE,
    _ensure_cupy_loaded,
    _fft2,
    _h_cache_lookup,
    _h_cache_store,
    _ifft2,
    _is_cupy_array,
    _validate_propagator_inputs,
)

__all__ = [
    'angular_spectrum_propagate_mft',
    'fresnel_propagate_mft',
    'fraunhofer_propagate_mft',
    'resample_field',
]


def angular_spectrum_propagate_mft(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx_in: float,
    dx_out: float,
    N_out: int,
    *,
    dy_in: Optional[float] = None,
    dy_out: Optional[float] = None,
    centre_out: Tuple[float, float] = (0.0, 0.0),
    bandlimit: bool = True,
    use_gpu: bool = False,
) -> np.ndarray:
    """Exact Angular Spectrum Method propagation onto an arbitrary
    user-specified output grid.

    Identical math to :func:`angular_spectrum_propagate` for the
    forward (FFT + transfer-function) step, but the inverse Fourier
    step samples the output field on a user-specified grid via a
    Bluestein chirp-Z transform rather than reusing the input's FFT
    grid.  This enables **focal-plane zoom for high-NA / non-paraxial
    systems** without zero-padding the input.

    Use cases where this beats :func:`fresnel_propagate_mft`:
        - High NA (NA > ~0.1) where the paraxial Fresnel assumption breaks.
        - Fields with significant evanescent content (sub-wavelength
          structures, near-field).
        - Anywhere :func:`angular_spectrum_propagate` is preferred over
          :func:`fresnel_propagate`, but you also want non-natural output
          grid sampling.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
    z : float
        Propagation distance [m].
    wavelength : float
    dx_in : float
        Input grid spacing in x [m].
    dx_out : float
        Output grid spacing in x [m] (independent of ``dx_in`` and ``z``).
    N_out : int
        Output grid size (square).
    dy_in, dy_out : float, optional
    centre_out : (float, float), optional
        Physical ``(x, y)`` centre of the output grid [m].  Defaults to
        ``(0, 0)`` (on-axis).
    bandlimit : bool, default True
        Apply Matsushima-Shimobaba band-limiting to the ASM transfer
        function on the input frequency grid.  Same default and effect
        as :func:`angular_spectrum_propagate`.
    use_gpu : bool, default False

    Returns
    -------
    E_out : ndarray (complex, N_out x N_out)
        Output field on the user's centred grid.  Carries the absolute
        physical phase including the natural Fresnel curvature -- this
        is the same convention as :func:`angular_spectrum_propagate`,
        :func:`fresnel_propagate`, and every other Lumenairy propagator.

    Notes
    -----
    Algorithm: regular FFT of the input, transfer function multiplication,
    then Bluestein chirp-Z inverse FT onto the chosen output grid.
    Cost: 2 * O(N^2 log N)  + O((N + M) log (N + M))  per axis.

    For the same-grid case (``dx_out == dx_in`` and ``N_out == Nx_in``,
    ``centre_out == (0, 0)``) the result agrees with
    :func:`angular_spectrum_propagate` to roughly float64 round-off
    (~1e-12 relative error).

    See also
    --------
    angular_spectrum_propagate : exact ASM with the natural FFT grid.
    fresnel_propagate_mft : paraxial Fresnel with arbitrary output grid.
    fraunhofer_propagate_mft : far-field with arbitrary output grid.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper -- siblings missed by the
    # v4.15.2 closure now share the same first-line guard.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'angular_spectrum_propagate_mft')

    _validate_propagator_inputs(E_in, z, wavelength, dx_in, dy_in,
                                fn_name='angular_spectrum_propagate_mft')
    from ..backend import is_jax_array
    from ._bluestein import _bluestein_centred_2d

    is_jax = is_jax_array(E_in)
    if is_jax:
        import jax.numpy as _jnp
        xp = _jnp
        fft2 = _jnp.fft.fft2
        ifft2 = _jnp.fft.ifft2
    elif CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if not _ensure_cupy_loaded():
            raise RuntimeError("CuPy requested but failed to load.")
        xp = _state.cp
        fft2 = _state.cp.fft.fft2
        ifft2 = _state.cp.fft.ifft2
        if not _is_cupy_array(E_in):
            E_in = _state.cp.asarray(E_in)
    else:
        xp = np
        fft2 = _fft2
        ifft2 = _ifft2
        if _is_cupy_array(E_in):
            E_in = E_in.get()

    if dy_in is None:
        dy_in = dx_in
    if dy_out is None:
        dy_out = dx_out

    Ny_in, Nx_in = E_in.shape
    Ny_out = Nx_out = int(N_out)

    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(_state.DEFAULT_COMPLEX_DTYPE)

    k = 2.0 * np.pi / wavelength
    xc, yc = float(centre_out[0]), float(centre_out[1])

    # ----- 1) Build ASM transfer function H(fx, fy) on the input freq grid --
    # Same construction as angular_spectrum_propagate (centred convention,
    # exact `kz = sqrt(k^2 - kx^2 - ky^2)`, optional Matsushima band-limit).
    #
    # H depends only on the input geometry (Ny_in, Nx_in, dy_in, dx_in,
    # wavelength, z, bandlimit, dtype) -- the user-specified output grid
    # (dx_out, N_out, centre_out) only enters in the Bluestein step
    # below.  Cache H on the input-geometry signature so repeat calls
    # at the same input plane onto different output grids share one
    # H build.  ``'ASM_MFT'`` tag keeps these entries disjoint from
    # plain ASM.  4.12.0: both backends now use ``fx < fx_max`` (open
    # interval, matching the Matsushima-Shimobaba paper and plain ASM);
    # pre-4.12 the NumPy branch used `<=` (one-bin off from JAX).
    if is_jax:
        # JAX path: build under the tracer (no host-side cache).
        fx = (xp.arange(Nx_in, dtype=xp.float64) - Nx_in / 2.0) / (Nx_in * dx_in)
        fy = (xp.arange(Ny_in, dtype=xp.float64) - Ny_in / 2.0) / (Ny_in * dy_in)
        kx_sq = (2.0 * float(np.pi) * fx) ** 2
        ky_sq = (2.0 * float(np.pi) * fy) ** 2
        kz_sq = k * k - kx_sq[None, :] - ky_sq[:, None]
        prop_mask = kz_sq > 0
        kz = xp.where(prop_mask, xp.sqrt(xp.where(prop_mask, kz_sq, 0.0)),
                      0.0)
        H = xp.where(prop_mask, xp.exp(1j * kz * z), 0.0).astype(target_cdtype)
        if bandlimit and z != 0:
            Lx_phys = Nx_in * dx_in
            Ly_phys = Ny_in * dy_in
            fx_max = Lx_phys / (2.0 * wavelength * abs(z))
            fy_max = Ly_phys / (2.0 * wavelength * abs(z))
            # 4.10: use strict less-than (matches plain ASM at line
            # 1200 and the Matsushima-Shimobaba paper, which uses an
            # open-interval cutoff).  Pre-4.10 ASM-MFT used <= here,
            # one-bin off from the ASM reference.
            bl_mask = ((xp.abs(fx)[None, :] < fx_max)
                       & (xp.abs(fy)[:, None] < fy_max))
            H = xp.where(bl_mask, H, 0.0).astype(target_cdtype)
    else:
        # NumPy / CuPy paths share a NumPy-host H cache; CuPy uploads
        # via xp.asarray on demand.
        h_key = (int(Ny_in), int(Nx_in), float(dy_in), float(dx_in),
                 float(wavelength), float(z),
                 bool(bandlimit),
                 np.dtype(target_cdtype).str, 'ASM_MFT')
        H_np = _h_cache_lookup(h_key)
        if H_np is None:
            fx = (np.arange(Nx_in, dtype=np.float64) - Nx_in / 2.0) / (Nx_in * dx_in)
            fy = (np.arange(Ny_in, dtype=np.float64) - Ny_in / 2.0) / (Ny_in * dy_in)
            kx_sq = (2.0 * np.pi * fx) ** 2
            ky_sq = (2.0 * np.pi * fy) ** 2
            kz_sq = k * k - kx_sq[None, :] - ky_sq[:, None]
            prop_mask = kz_sq > 0
            kz = np.where(prop_mask,
                          np.sqrt(np.where(prop_mask, kz_sq, 0.0)), 0.0)
            H_np = np.where(prop_mask, np.exp(1j * kz * z), 0.0).astype(
                target_cdtype)
            if bandlimit and z != 0:
                Lx_phys = Nx_in * dx_in
                Ly_phys = Ny_in * dy_in
                fx_max = Lx_phys / (2.0 * wavelength * abs(z))
                fy_max = Ly_phys / (2.0 * wavelength * abs(z))
                # 4.12.0 (audit round-4 B1-4): use strict `<` to match
                # the JAX branch above (and plain ASM at line ~1200).
                # Pre-4.12 NumPy used `<=` -- one-bin disagreement
                # between backends at the band-limit boundary.
                bl_mask = ((np.abs(fx)[None, :] < fx_max)
                           & (np.abs(fy)[:, None] < fy_max))
                H_np = np.where(bl_mask, H_np, 0.0).astype(target_cdtype)
            _h_cache_store(h_key, H_np)
        H = H_np if xp is np else xp.asarray(H_np)

    # ----- 2) FFT input to angular spectrum (centred convention) ------------
    # Match the existing angular_spectrum_propagate's fftshift/ifftshift
    # idiom so the centred-frequency grid lines up.
    if is_jax:
        E_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(E_in)))
    else:
        E_fft = xp.fft.fftshift(fft2(xp.fft.ifftshift(E_in)))
    A_propagated = E_fft * H

    # ----- 3) Inverse FT onto user-specified output grid via Bluestein ------
    # The inverse FT of the propagated angular spectrum is
    #   E_out(x_out, y_out) = integral A(fx, fy) * exp(+2*pi*j*(fx*x_out + fy*y_out)) dfx dfy
    # Discretised on the centred input frequency grid:
    #   fx[nx] = (nx - Nx_in/2) / (Nx_in * dx_in)   (with dfx = 1/(Nx_in * dx_in))
    #   x_out[kx] = (kx - Nx_out/2) * dx_out + xc
    # The product fx[nx] * x_out[kx] expands to a centred Bluestein form
    # with alpha = dfx * dx_out = dx_out / (Nx_in * dx_in) and sign = +1.
    # The 1/(Nx_in*Ny_in) prefactor matches numpy/scipy's IFFT normalisation
    # so that round-tripping through ASM-MFT recovers the input on the
    # natural grid.
    alpha_x = dx_out / (Nx_in * dx_in)
    alpha_y = dy_out / (Ny_in * dy_in)
    kc_x = Nx_out / 2.0 - xc / dx_out
    kc_y = Ny_out / 2.0 - yc / dy_out

    F = _bluestein_centred_2d(
        A_propagated, alpha_x, alpha_y, Ny_out, Nx_out,
        n_centre_in_x=Nx_in / 2.0,
        n_centre_in_y=Ny_in / 2.0,
        k_centre_out_x=kc_x,
        k_centre_out_y=kc_y,
        sign=+1, xp=xp, fft2=fft2, ifft2=ifft2,
        target_cdtype=target_cdtype,
    )

    norm = target_cdtype.type(1.0 / (Nx_in * Ny_in))
    E_out = F * norm
    if E_out.dtype != target_cdtype:
        E_out = E_out.astype(target_cdtype)
    return E_out


def resample_field(
    E_in: np.ndarray,
    dx_in: float,
    dx_out: float,
    N_out: Optional[int] = None,
    order: int = 3,
) -> Tuple[np.ndarray, float]:
    """
    Resample a complex optical field from one grid spacing to another.

    This is the bridge function for switching between propagation methods
    that use different grid spacings (e.g. Fresnel output -> ASM input,
    or vice versa).  Both amplitude and phase are interpolated using
    scipy's map_coordinates.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input field on a grid with spacing ``dx_in``.
    dx_in : float
        Input grid spacing [m].
    dx_out : float
        Desired output grid spacing [m].
    N_out : int or None
        Output grid size.  If ``None``, chosen so the output covers the
        same physical extent as the input: ``N_out = round(N_in * dx_in / dx_out)``.
    order : int, default 3
        Interpolation order (1=linear, 3=cubic, 5=quintic).

    Returns
    -------
    E_out : ndarray (complex, N_out x N_out)
        Resampled field on the new grid.
    dx_out : float
        The output grid spacing (same as the input parameter, returned
        for convenience so callers can chain: ``E, dx = resample_field(...)``).

    Notes
    -----
    - Interpolation introduces a small error proportional to (dx_out/feature_size)^order.
      For order=3 (cubic), this is < 0.1% when features are sampled at >= 4 pixels.
    - For downsampling (dx_out > dx_in), consider anti-alias filtering first.
    - The field is assumed to be on a centered grid: x = (arange(N) - N/2) * dx.
    """
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Pre-v4.15.5 an MCF / 3-D
    # ensemble input failed at ``E_in.shape`` unpacking
    # (``ValueError: too many values to unpack`` for 3-D) or
    # attribute access (MCF) -- routes both to the canonical v4.16
    # message via the V6 walker.  ``resample_field`` was missed by
    # the v4.15.4 walker (name doesn't start with ``apply_`` or
    # contain ``_propagate``); the V6 first-positional-name filter
    # catches it via ``E_in``.  Input kind: 'field'.
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'resample_field')
    from scipy.ndimage import map_coordinates

    Ny_in, Nx_in = E_in.shape
    if N_out is None:
        Nx_out = int(round(Nx_in * dx_in / dx_out))
        Ny_out = int(round(Ny_in * dx_in / dx_out))
    else:
        Nx_out = Ny_out = int(N_out)

    # Output coordinates in input-pixel units.
    # Input grid:  x_in[i]  = (i - Nx_in/2)  * dx_in
    # Output grid: x_out[j] = (j - Nx_out/2) * dx_out
    # Map: i = x_out / dx_in + Nx_in/2 = (j - Nx_out/2) * dx_out/dx_in + Nx_in/2
    scale = dx_out / dx_in
    jx = np.arange(Nx_out)
    jy = np.arange(Ny_out)
    ix = (jx - Nx_out / 2) * scale + Nx_in / 2
    iy = (jy - Ny_out / 2) * scale + Ny_in / 2
    IX, IY = np.meshgrid(ix, iy)
    coords = np.array([IY.ravel(), IX.ravel()])
    # v5.17.0 lifetime hygiene: the meshgrids are folded into coords --
    # free them before interpolating, free coords before the complex
    # combine, and free each part once consumed.  Byte-identical.
    del IX, IY

    # Interpolate real and imaginary parts separately
    real_out = map_coordinates(E_in.real, coords, order=order, mode='constant', cval=0.0)
    imag_out = map_coordinates(E_in.imag, coords, order=order, mode='constant', cval=0.0)
    del coords
    E_out = (real_out + 1j * imag_out).reshape(Ny_out, Nx_out)
    del real_out, imag_out

    return E_out, dx_out


def fresnel_propagate_mft(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx_in: float,
    dx_out: float,
    N_out: int,
    *,
    dy_in: Optional[float] = None,
    dy_out: Optional[float] = None,
    centre_out: Tuple[float, float] = (0.0, 0.0),
    use_gpu: bool = False,
) -> np.ndarray:
    """Fresnel propagation onto an arbitrary user-specified output grid.

    Unlike :func:`fresnel_propagate`, which forces ``dx_out = lambda*z/(N*dx_in)``
    and ``N_out = N_in``, this routine evaluates the Fresnel diffraction
    integral at exactly the output grid you ask for.  It is the standard
    tool for **focal-plane zoom** -- sampling a tightly-focused region of
    the output plane at sub-FFT-pitch resolution without padding the input
    grid by the corresponding factor.

    The math is the same as :func:`fresnel_propagate` (single-FFT paraxial
    Fresnel: input quadratic phase + Fourier transform + output quadratic
    phase + ``exp(i*k*z) / (i*lambda*z)`` carrier prefactor).  Only the
    Fourier-transform step is replaced with a Bluestein chirp-Z transform
    that samples directly onto the chosen output grid.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input field on a centred grid (pixel ``[Ny//2, Nx//2]`` at
        coordinate ``(0, 0)``, matching the convention of
        :func:`fresnel_propagate`).
    z : float
        Propagation distance [m].
    wavelength : float
        Wavelength [m].
    dx_in : float
        Input grid spacing in x [m].
    dx_out : float
        Desired output grid spacing in x [m].  This is independent of
        ``dx_in`` and ``z`` -- pick whatever sampling you want at the
        output plane.
    N_out : int
        Output grid size (square output: ``N_out`` by ``N_out``).
    dy_in, dy_out : float, optional
        Input / output grid spacings in y.  Default to ``dx_in`` / ``dx_out``.
    centre_out : (float, float), optional
        Physical ``(x, y)`` centre of the output grid [m].  Defaults to
        ``(0, 0)`` (on-axis).  Use a non-zero value to zoom into an
        off-axis region of the output plane (e.g. the chief image of a
        field point off the optical axis).
    use_gpu : bool, default False
        Route through CuPy if available.  Auto-detected from ``E_in``
        (CuPy / JAX arrays use their native backend regardless of this
        flag).

    Returns
    -------
    E_out : ndarray (complex, N_out x N_out)
        Output field on the user's centred grid.

    Notes
    -----
    Algorithm: O((N + M) log (N + M)) per axis via Bluestein's chirp-Z
    transform with two zero-padded 2-D FFTs.  Substantially faster than
    a direct matrix-Fourier transform (O(N^2 M^2)) for typical
    focal-zoom workflows.

    Sampling: the same Fresnel-number heuristic as :func:`fresnel_propagate`
    applies to ``dx_in`` and ``z`` (no new validity restriction comes
    from the user-specified output grid).  In particular the input must
    Nyquist-resolve the input quadratic phase
    ``exp(i*k/(2z) * (X_in^2 + Y_in^2))`` -- if ``dx_in`` is too coarse
    for the chosen ``z``, the output is aliased regardless of ``dx_out``.

    See also
    --------
    fresnel_propagate : single-FFT Fresnel with the natural FFT output grid.
    fraunhofer_propagate_mft : far-field counterpart.
    angular_spectrum_propagate_mft : exact ASM with arbitrary output grid.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper -- siblings missed by the
    # v4.15.2 closure now share the same first-line guard.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'fresnel_propagate_mft')

    _validate_propagator_inputs(E_in, z, wavelength, dx_in, dy_in,
                                fn_name='fresnel_propagate_mft')
    # 4.9 fix (audit #3.3): forward-only -- see ``fresnel_propagate``.
    if z <= 0:
        raise ValueError(
            f"fresnel_propagate_mft: z must be > 0 (got z={z}).  "
            f"Fresnel-MFT is the focal-plane-zoomed variant of "
            f"fresnel_propagate and inherits its forward-only "
            f"restriction.  Use angular_spectrum_propagate_mft for "
            f"back-propagation with arbitrary output grid.")
    # ----- backend dispatch -------------------------------------------------
    from ..backend import is_jax_array
    from ._bluestein import _bluestein_centred_2d

    is_jax = is_jax_array(E_in)
    if is_jax:
        import jax.numpy as _jnp
        xp = _jnp
        fft2 = _jnp.fft.fft2
        ifft2 = _jnp.fft.ifft2
    elif CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if not _ensure_cupy_loaded():
            raise RuntimeError("CuPy requested but failed to load.")
        xp = _state.cp
        fft2 = _state.cp.fft.fft2
        ifft2 = _state.cp.fft.ifft2
        if not _is_cupy_array(E_in):
            E_in = _state.cp.asarray(E_in)
    else:
        xp = np
        fft2 = _fft2
        ifft2 = _ifft2
        if _is_cupy_array(E_in):
            E_in = E_in.get()

    if dy_in is None:
        dy_in = dx_in
    if dy_out is None:
        dy_out = dx_out

    Ny_in, Nx_in = E_in.shape
    Ny_out = Nx_out = int(N_out)

    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(_state.DEFAULT_COMPLEX_DTYPE)

    k = 2.0 * np.pi / wavelength
    xc, yc = float(centre_out[0]), float(centre_out[1])

    # ----- coordinate grids (numpy for chirp construction) ------------------
    n_x = np.arange(Nx_in, dtype=np.float64)
    n_y = np.arange(Ny_in, dtype=np.float64)
    k_x = np.arange(Nx_out, dtype=np.float64)
    k_y = np.arange(Ny_out, dtype=np.float64)
    x_in = (n_x - Nx_in / 2.0) * dx_in
    y_in = (n_y - Ny_in / 2.0) * dy_in
    x_out = (k_x - Nx_out / 2.0) * dx_out + xc
    y_out = (k_y - Ny_out / 2.0) * dy_out + yc
    X_in_np, Y_in_np = np.meshgrid(x_in, y_in, indexing='xy')
    X_out_np, Y_out_np = np.meshgrid(x_out, y_out, indexing='xy')

    def _to_xp(arr_np_complex):
        a = arr_np_complex.astype(target_cdtype, copy=False)
        if xp is np:
            return a
        return xp.asarray(a)

    # ----- 1) input-plane quadratic phase -----------------------------------
    quad_in = _to_xp(np.exp(1j * k / (2.0 * z) * (X_in_np**2 + Y_in_np**2)))
    E_mod = E_in * quad_in

    # ----- 2) Fresnel Fourier integral via Bluestein on centred grid --------
    # Sum_{ny, nx} E_mod[ny, nx] * exp(-i*k/z * (x_in[nx]*x_out[kx] + y_in[ny]*y_out[ky]))
    # = Sum_{ny, nx} E_mod[ny, nx]
    #              * exp(-2*pi*j*alpha_x*(nx - Nx_in/2)*(kx - kc_x))
    #              * exp(-2*pi*j*alpha_y*(ny - Ny_in/2)*(ky - kc_y))
    # with
    #     alpha = dx_in*dx_out/(lambda*z)   ("natural" Bluestein coefficient)
    #     kc_x  = Nx_out/2 - xc/dx_out      (output centre shifted by centre_out)
    alpha_x = dx_in * dx_out / (wavelength * z)
    alpha_y = dy_in * dy_out / (wavelength * z)
    kc_x = Nx_out / 2.0 - xc / dx_out
    kc_y = Ny_out / 2.0 - yc / dy_out

    F = _bluestein_centred_2d(
        E_mod, alpha_x, alpha_y, Ny_out, Nx_out,
        n_centre_in_x=Nx_in / 2.0,
        n_centre_in_y=Ny_in / 2.0,
        k_centre_out_x=kc_x,
        k_centre_out_y=kc_y,
        sign=-1, xp=xp, fft2=fft2, ifft2=ifft2,
        target_cdtype=target_cdtype,
    )

    # ----- 3) output-plane quadratic phase + carrier prefactor + area -------
    quad_out = _to_xp(np.exp(1j * k / (2.0 * z) * (X_out_np**2 + Y_out_np**2)))
    prefactor = (np.exp(1j * k * z) / (1j * wavelength * z)) * dx_in * dy_in
    prefactor_c = target_cdtype.type(prefactor)

    E_out = prefactor_c * quad_out * F
    if E_out.dtype != target_cdtype:
        E_out = E_out.astype(target_cdtype)
    return E_out


def fraunhofer_propagate_mft(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx_in: float,
    dx_out: float,
    N_out: int,
    *,
    dy_in: Optional[float] = None,
    dy_out: Optional[float] = None,
    centre_out: Tuple[float, float] = (0.0, 0.0),
    use_gpu: bool = False,
) -> np.ndarray:
    """Fraunhofer (far-field) propagation onto an arbitrary user-specified
    output grid.

    Identical math to :func:`fraunhofer_propagate` (single Fourier transform
    with output quadratic phase + carrier prefactor) except the FT step
    uses a Bluestein chirp-Z transform that samples directly onto the
    chosen output grid.  This is the standard tool for **coronagraph and
    high-contrast imaging codes** -- you can sample the far-field at
    sub-lambda/D resolution around an off-axis stellar PSF without zero-
    padding the input pupil to enormous sizes.

    Differs from :func:`fresnel_propagate_mft` by skipping the input-plane
    quadratic phase ``exp(i*k/(2z)*(X_in^2 + Y_in^2))`` -- it is assumed
    negligible at large z (small Fresnel number).

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
    z : float
        Propagation distance [m].
    wavelength : float
    dx_in : float
        Input grid spacing [m].
    dx_out : float
        Output grid spacing [m] (independent of ``dx_in`` and ``z``).
    N_out : int
        Output grid size (square).
    dy_in, dy_out : float, optional
    centre_out : (float, float), optional
        Physical ``(x, y)`` centre of the output grid [m].  Defaults to
        ``(0, 0)``.  Use a non-zero value to evaluate the far-field at
        an off-axis point (e.g. an exoplanet location relative to a
        stellar chief image).
    use_gpu : bool, default False

    Returns
    -------
    E_out : ndarray (complex, N_out x N_out)

    See also
    --------
    fraunhofer_propagate : single-FFT Fraunhofer with the natural FFT grid.
    fresnel_propagate_mft : near-field counterpart, includes input-plane
        quadratic phase.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper -- siblings missed by the
    # v4.15.2 closure now share the same first-line guard.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'fraunhofer_propagate_mft')

    _validate_propagator_inputs(E_in, z, wavelength, dx_in, dy_in,
                                fn_name='fraunhofer_propagate_mft')
    # 4.9 fix (audit #3.3): forward-only -- see ``fraunhofer_propagate``.
    if z <= 0:
        raise ValueError(
            f"fraunhofer_propagate_mft: z must be > 0 (got z={z}).  "
            f"Fraunhofer-MFT is forward-only (it's the far-field "
            f"limit of fresnel_propagate_mft).  Use "
            f"angular_spectrum_propagate_mft for back-propagation.")
    from ..backend import is_jax_array
    from ._bluestein import _bluestein_centred_2d

    is_jax = is_jax_array(E_in)
    if is_jax:
        import jax.numpy as _jnp
        xp = _jnp
        fft2 = _jnp.fft.fft2
        ifft2 = _jnp.fft.ifft2
    elif CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if not _ensure_cupy_loaded():
            raise RuntimeError("CuPy requested but failed to load.")
        xp = _state.cp
        fft2 = _state.cp.fft.fft2
        ifft2 = _state.cp.fft.ifft2
        if not _is_cupy_array(E_in):
            E_in = _state.cp.asarray(E_in)
    else:
        xp = np
        fft2 = _fft2
        ifft2 = _ifft2
        if _is_cupy_array(E_in):
            E_in = E_in.get()

    if dy_in is None:
        dy_in = dx_in
    if dy_out is None:
        dy_out = dx_out

    Ny_in, Nx_in = E_in.shape
    Ny_out = Nx_out = int(N_out)

    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(_state.DEFAULT_COMPLEX_DTYPE)

    k = 2.0 * np.pi / wavelength
    xc, yc = float(centre_out[0]), float(centre_out[1])

    np.arange(Nx_in, dtype=np.float64)
    np.arange(Ny_in, dtype=np.float64)
    k_x = np.arange(Nx_out, dtype=np.float64)
    k_y = np.arange(Ny_out, dtype=np.float64)
    x_out = (k_x - Nx_out / 2.0) * dx_out + xc
    y_out = (k_y - Ny_out / 2.0) * dy_out + yc
    X_out_np, Y_out_np = np.meshgrid(x_out, y_out, indexing='xy')

    def _to_xp(arr_np_complex):
        a = arr_np_complex.astype(target_cdtype, copy=False)
        if xp is np:
            return a
        return xp.asarray(a)

    # No input quadratic phase (Fraunhofer assumption).
    # Bluestein FT directly on E_in.
    alpha_x = dx_in * dx_out / (wavelength * z)
    alpha_y = dy_in * dy_out / (wavelength * z)
    kc_x = Nx_out / 2.0 - xc / dx_out
    kc_y = Ny_out / 2.0 - yc / dy_out

    F = _bluestein_centred_2d(
        E_in, alpha_x, alpha_y, Ny_out, Nx_out,
        n_centre_in_x=Nx_in / 2.0,
        n_centre_in_y=Ny_in / 2.0,
        k_centre_out_x=kc_x,
        k_centre_out_y=kc_y,
        sign=-1, xp=xp, fft2=fft2, ifft2=ifft2,
        target_cdtype=target_cdtype,
    )

    quad_out = _to_xp(np.exp(1j * k / (2.0 * z) * (X_out_np**2 + Y_out_np**2)))
    prefactor = (np.exp(1j * k * z) / (1j * wavelength * z)) * dx_in * dy_in
    prefactor_c = target_cdtype.type(prefactor)

    E_out = prefactor_c * quad_out * F
    if E_out.dtype != target_cdtype:
        E_out = E_out.astype(target_cdtype)
    return E_out

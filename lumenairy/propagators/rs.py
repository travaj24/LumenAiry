"""
Rayleigh-Sommerfeld propagator
==============================

v5.1.0 Agent C split: extracted from ``propagation.py``.  Contains the
RS Green's-function convolution implementation (first RS solution,
optional Matsushima bandlimit on the padded FFT'd kernel).

Public symbol re-exported by ``propagation.py``.

Author:  Andrew Traverso
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import fft_infra as _state
from .fft_infra import (
    CUPY_AVAILABLE,
    _fft2,
    _h_cache_lookup,
    _h_cache_store,
    _ifft2,
    _is_cupy_array,
    _validate_propagator_inputs,
)

__all__ = [
    'rayleigh_sommerfeld_propagate',
]


def rayleigh_sommerfeld_propagate(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    bandlimit: bool = False,
    use_gpu: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Propagate an optical field using the Rayleigh-Sommerfeld convolution.

    This method computes the first Rayleigh-Sommerfeld solution by
    convolving the input field with the free-space impulse response
    (Green's function).  Unlike the ASM transfer-function approach,
    the RS convolution constructs the propagation kernel in the
    *spatial* domain and performs the convolution via FFT, which
    naturally captures near-field diffraction effects without the
    band-limiting approximation used in ASM.

    The impulse response is (Goodman *Introduction to Fourier
    Optics*, 3rd ed., eq. 3-43):

        h(x, y, z) = (1 / 2pi) * (z / r^2) * (1/r - ik) * exp(ikr)

    where ``r = sqrt(x^2 + y^2 + z^2)`` and ``k = 2*pi / lambda``.
    Pre-4.10 the kernel used the negated ``(ik - 1/r)`` form, so
    superposing RS with ASM / Fresnel results was 180-degrees out of
    phase.  The docstring formula was updated in 4.11.1 to match the
    corrected code.

    The convolution is computed as::

        E_out = IFFT{ FFT{E_in} * FFT{h} }

    using zero-padded arrays (2N x 2N) to avoid circular convolution
    artifacts.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input electric field.
    z : float
        Propagation distance [m].  Positive = forward.
    wavelength : float
        Free-space wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to dx.
    bandlimit : bool, default False
        Apply a Matsushima-style frequency cutoff
        ``|f| < L2 / (2*lambda*|z|)`` to the FFT'd kernel
        ``H = FFT(h)``.  v5.30 (audit P12): that expression is the
        **z -> infinity asymptote** of Matsushima & Shimobaba's exact
        local-frequency limit ``1/(lambda*sqrt((2z/L2)^2 + 1))``, not the
        exact limit -- it is strictly the larger of the two, so it never
        over-filters (see
        :func:`~lumenairy.propagators.fft_infra._get_or_make_bandlimit`
        for the derivation and the measured over-width table).
        Default ``False`` preserves the historical
        "exact Green's function" character of RS that justifies its
        use over ASM in the near field.  Set ``True`` to suppress
        aliasing artifacts on coarse grids at long propagation
        distances (where the kernel chirp under-samples on the
        discrete grid, the same regime where ASM's ``bandlimit=True``
        default is needed).  Cutoff is computed on the padded
        (2N x 2N) grid so the resulting bandwidth budget matches the
        FFT length actually used by the convolution.
    use_gpu : bool, default False
        Use CuPy GPU acceleration if available.
    verbose : bool, default False
        Print diagnostic info.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Propagated field (same shape as input).

    Notes
    -----
    **When to use RS instead of ASM:**

    - Near-field propagation (z ~ a few wavelengths) where ASM's
      band-limiting can suppress valid high-frequency content.
    - Validation / cross-check against ASM results.
    - Situations where the exact Green's function is preferred over
      the plane-wave decomposition.

    **Computational cost:** ~4x ASM due to zero-padding (2N FFTs
    instead of N FFTs) and the spatial-domain kernel construction.

    **Memory:** ~6x input array size (padded E, padded h, FFTs).

    At large distances (z >> a^2 / lambda), RS and ASM give identical
    results.  For intermediate distances they agree to machine precision
    when ASM uses no band-limiting (``bandlimit=False``).

    **H caching:** the FFT'd kernel ``H`` is cached on the NumPy backend
    keyed on ``(2*Ny, 2*Nx, dy, dx, wavelength, z, bandlimit, dtype)``.
    Repeat calls at the same geometry skip the kernel build and FFT
    (~30-40% of total RS time on 2k+ grids).  The cache is shared
    with :func:`angular_spectrum_propagate` and obeys the same byte
    budgets configured via :func:`set_asm_cache_size`.  CuPy and JAX
    arrays are kept out of the cache (host-side dict can't safely
    retain device pointers / traced objects); rebuild every call.

    References
    ----------
    [1] Goodman, J.W. "Introduction to Fourier Optics" (3rd ed.),
        Section 3.5: Rayleigh-Sommerfeld Diffraction Theory.
    [2] Shen, F. and Wang, A. (2006). "Fast-Fourier-transform based
        numerical integration method for the Rayleigh-Sommerfeld
        diffraction formula." Appl. Opt. 45(6): 1102-1110.
    [3] Matsushima, K. and Shimobaba, T. (2009). "Band-limited angular
        spectrum method for numerical simulation of free-space
        propagation in far and near fields." Opt. Express 17(22):
        19662-19673.  NOTE (v5.30, audit P12): ``bandlimit=True`` applies
        the ``z -> infinity`` asymptote of this paper's local-frequency
        limit, not the exact expression (never over-filters).

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.propagation import rayleigh_sommerfeld_propagate
    >>>
    >>> N = 512; dx = 1e-6; wv = 0.633e-6
    >>> x = (np.arange(N) - N/2) * dx
    >>> X, Y = np.meshgrid(x, x)
    >>> E_in = (np.sqrt(X**2 + Y**2) < 50e-6).astype(complex)  # circular aperture
    >>>
    >>> E_out = rayleigh_sommerfeld_propagate(E_in, z=1e-3, wavelength=wv, dx=dx)
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper (replaces the v4.15.2 inline
    # guard).
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'rayleigh_sommerfeld_propagate',
                           input_kind='field')

    # 4.12.0 (audit round-4 B1-3): RS is forward-only.  Pre-4.12 the
    # function accepted z <= 0 silently and computed a 180-degrees-
    # wrong-phase kernel for the back-propagation case.  Match the
    # existing Fresnel / Fraunhofer / SAS guards: hard error with
    # guidance to use ASM / ASM-MFT for back-propagation.
    if z <= 0:
        raise ValueError(
            f"rayleigh_sommerfeld_propagate: z must be > 0 (got "
            f"{z!r}).  RS is forward-only; use "
            f"angular_spectrum_propagate or "
            f"angular_spectrum_propagate_mft for back-propagation "
            f"(those handle the z < 0 case correctly).")
    _validate_propagator_inputs(E_in, z, wavelength, dx, dy,
                                fn_name='rayleigh_sommerfeld_propagate')

    # -- array library selection -----------------------------------------------
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
            E_in = E_in.get()

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx

    # Target complex dtype for h, H, and the padded buffer.  Inferred
    # from E_in so the caller controls precision via input dtype.
    # Non-complex input falls back to DEFAULT_COMPLEX_DTYPE to match
    # the standardisation used by angular_spectrum_propagate and the
    # MFT propagator family.
    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(_state.DEFAULT_COMPLEX_DTYPE)

    k = 2 * np.pi / wavelength

    # -- zero-pad to avoid circular convolution --------------------------------
    Ny2 = 2 * Ny
    Nx2 = 2 * Nx

    # H cache (NumPy backend only)
    # Geometry signature.  Hits return the previously-built H without
    # re-running the kernel construction or its FFT (~30-40% of total
    # RS time on 2k+ grids).  The 'RS' tag keeps RS keys disjoint from
    # ASM keys even when the unpadded grid sizes happen to coincide.
    h_key = None
    H = None
    if xp is np:
        h_key = (int(Ny2), int(Nx2), float(dy), float(dx),
                 float(wavelength), float(z), bool(bandlimit),
                 np.dtype(target_cdtype).str, 'RS')
        H = _h_cache_lookup(h_key)


    if H is None:
        # -- build the RS impulse response h(x, y, z) on the padded grid -------
        # h = -(1/2π) ∂/∂z[exp(ikr)/r]
        #   = (z / (2π r²)) · (1/r − ik) · exp(ikr)         (Goodman 3-43)
        # Pre-4.10 implementation flipped this to (ik − 1/r), producing
        # −h_correct.  Output amplitudes look fine for |E|² consumers but
        # any coherent sum of RS with ASM/Fresnel was 180° out of phase.
        x = (xp.arange(Nx2) - Nx2 / 2) * dx
        y = (xp.arange(Ny2) - Ny2 / 2) * dy
        X, Y = xp.meshgrid(x, y, indexing='xy')
        r = xp.sqrt(X ** 2 + Y ** 2 + z ** 2)
        h = (z / (2 * np.pi * r ** 2)) * xp.exp(1j * k * r) * (1.0 / r - 1j * k)
        h = h * (dx * dy)
        if h.dtype != target_cdtype:
            h = h.astype(target_cdtype)

        if verbose:
            print(f"  RS propagation: z = {z*1e3:.3f} mm  (H cache miss)")
            print(f"  Grid: {Ny}x{Nx} -> padded {Ny2}x{Nx2}")
            print(f"  Wavelength: {wavelength*1e9:.1f} nm")
            try:
                print(f"  Kernel max |h|: {float(xp.max(xp.abs(h))):.4e}")
            except (TypeError, ValueError, RuntimeError) as _exc:
                # xp.max + float() can fail under JAX tracing where h is
                # an abstract array (TypeError on the float()
                # conversion).  Cosmetic print failure -- demote to a
                # brief diagnostic so debug runs surface the cause.
                print(f"  Kernel max |h|: <unavailable: "
                      f"{type(_exc).__name__}>")

        # -- FFT the kernel ----------------------------------------------------
        # The result is cached via _h_cache_store below and reused
        # across many subsequent _fft2/_ifft2 calls; under the 4.12
        # double-buffer contract on _fft2, we must take an explicit
        # copy here so the cached H survives the third subsequent
        # call at this shape (which would recycle the slot).  The
        # bandlimit branch below already produces a fresh array via
        # H * mask_c, but the non-bandlimit path would otherwise
        # alias the plan workspace; copy unconditionally for clarity.
        if is_jax:
            H = xp.fft.fft2(xp.fft.ifftshift(h))
        elif xp is np:
            H = _fft2(np.fft.ifftshift(h)).copy()
        else:
            H = xp.fft.fft2(xp.fft.ifftshift(h))

        # -- Matsushima-style bandlimit on the padded H ------------------------
        # Cutoff matches the ASM derivation but uses the padded extent
        # (Lx2 = 2*Nx*dx) since the FFT length is what determines the
        # discrete frequency support.  The mask is built in centred
        # order then ifftshifted to align with H's DC-at-corner layout.
        #
        # v5.30 (audit P12): the cutoff ``L / (2*lambda*|z|)`` is the
        # z -> infinity ASYMPTOTE of Matsushima & Shimobaba's exact
        # local-frequency limit, not that limit itself -- see
        # :func:`lumenairy.propagators.fft_infra._get_or_make_bandlimit`
        # for the derivation and the measured over-width table.  It is an
        # upper bound, so it never over-filters.
        #
        # v5.30 (audit P13): dropped the dead ``and z != 0`` conjunct.  RS
        # is forward-only and the guard above hard-raises for ``z <= 0``
        # (measured: ``z=0`` and ``z=-1e-3`` both ValueError), so ``z``
        # here is always > 0 and the test could never be False.  ASM /
        # ASM-MFT keep THEIR ``z != 0`` conjuncts -- those DO accept
        # ``z == 0`` (the exact identity, audit S2-11) and rely on it.
        if bandlimit:
            fx = (np.arange(Nx2) - Nx2 / 2) / (Nx2 * dx)
            fy = (np.arange(Ny2) - Ny2 / 2) / (Ny2 * dy)
            Lx2 = Nx2 * dx
            Ly2 = Ny2 * dy
            fx_max = Lx2 / (2 * wavelength * abs(z))
            fy_max = Ly2 / (2 * wavelength * abs(z))
            bl_x = np.abs(fx) < fx_max
            bl_y = np.abs(fy) < fy_max
            mask = (bl_y[:, None] & bl_x[None, :])
            mask = np.fft.ifftshift(mask)
            mask_c = mask.astype(target_cdtype)
            if xp is np:
                H = H * mask_c
            else:
                H = H * xp.asarray(mask_c)
            if verbose:
                kept_frac = float(np.mean(mask))
                print(f"  Bandlimit: keeping {kept_frac*100:.1f}% of "
                      f"padded spectrum")

        # Store under the NumPy key only.  See _h_cache_store for the
        # byte-budget eviction policy.  Cached H is used read-only.
        if h_key is not None:
            _h_cache_store(h_key, H)
    elif verbose:
        print(f"  RS propagation: z = {z*1e3:.3f} mm  (H cache HIT)")

    # -- build the padded input field -----------------------------------------
    if is_jax:
        # JAX is functional / immutable -- can't write into a pre-allocated
        # array.  Build the padded array via jnp.zeros + at[].set.
        E_padded = xp.zeros((Ny2, Nx2), dtype=target_cdtype)
        y0 = Ny // 2
        x0 = Nx // 2
        E_padded = E_padded.at[y0:y0 + Ny, x0:x0 + Nx].set(E_in)
    else:
        E_padded = xp.zeros((Ny2, Nx2), dtype=target_cdtype)
        y0 = Ny // 2
        x0 = Nx // 2
        E_padded[y0:y0 + Ny, x0:x0 + Nx] = E_in

    # -- convolve via FFT ------------------------------------------------------
    if is_jax:
        E_fft = xp.fft.fft2(E_padded)
        E_conv = xp.fft.ifft2(E_fft * H)
    elif xp is np:
        E_fft = _fft2(E_padded)
        E_conv = _ifft2(E_fft * H)
    else:
        E_fft = xp.fft.fft2(E_padded)
        E_conv = xp.fft.ifft2(E_fft * H)

    # -- extract the valid region (same location as input was placed) ----------
    # v5.4.6 (audit F-3): ``.copy()`` is REQUIRED.  For the NumPy/CuPy path
    # ``_ifft2`` returns a view into the cache-owned pyFFTW inverse
    # ping-pong buffer, which the double-buffer contract guarantees only
    # until the NEXT same-key ``_ifft2`` call.  Returning a bare slice
    # (a view) of that buffer means a subsequent RS propagation at the
    # same grid silently overwrites a previously-returned field -- a
    # data-corruption hazard on multi-distance RS sweeps.  Copy detaches
    # the output from the reused buffer.
    E_out = E_conv[y0:y0 + Ny, x0:x0 + Nx].copy()

    return E_out

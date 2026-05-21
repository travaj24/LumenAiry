"""
Scalable Angular Spectrum (SAS) propagator
==========================================

v5.1.0 Agent C split: extracted from ``propagation.py``.  Implements the
Heintzmann / Loetgering / Wechsler 2023 SAS three-FFT kernel
(:doi:`10.1364/OPTICA.497809`), which propagates with a tunable
output-pixel pitch suitable for the medium-far-field regime where
plain ASM would need impractically many samples.

Public symbol re-exported by ``propagation.py``.

Author:  Andrew Traverso
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .fft_infra import (
    CUPY_AVAILABLE,
    _is_cupy_array,
    _h_cache_lookup, _h_cache_store,
    _validate_propagator_inputs,
)
from . import fft_infra as _state


__all__ = [
    'scalable_angular_spectrum_propagate',
]


def scalable_angular_spectrum_propagate(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    pad: int = 2,
    skip_final_phase: bool = False,
    use_gpu: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """
    Scalable-angular-spectrum propagator with variable output pitch.

    Implements the Heintzmann-Loetgering-Wechsler 2023 three-FFT kernel:
    apply an ASM-minus-Fresnel precompensation phase in the spatial-frequency
    domain, then a Fresnel chirp + single FFT (+ optional final quadratic
    phase).  The output grid has pixel pitch
    ``dx_out = lambda * z / (pad * N * dx)`` which can be much larger than
    the input pitch, letting one propagate over distances where a standard
    angular-spectrum call would need impractically many samples to span the
    geometric cone of the beam.

    The kernel is exact up to the ASM-vs-Fresnel band-limit cutoff ``W``
    baked into the precompensation; beyond that cutoff the method gracefully
    reduces to a zeroed transfer function (high-NA components are dropped).
    A closed-form ``z_limit`` from the paper bounds the propagation distance
    for which the method remains valid at the input sampling; we warn (not
    raise) when ``z > z_limit`` so the caller can still experiment.

    Parameters
    ----------
    E_in : ndarray (complex, N x N)
        Input field.  Must be square (Ny == Nx = N).  NumPy or CuPy.

    z : float
        Propagation distance [m].  Positive = forward.

    wavelength : float
        Wavelength [m].

    dx : float
        Input grid pitch [m].  Input extent is ``L = N * dx``.

    pad : int, default 2
        Zero-padding factor applied before the SAS kernel.  The reference
        implementation uses 2; larger values reduce aliasing further at the
        cost of more compute.  Output is cropped back to ``N x N`` after
        the kernel runs.

    skip_final_phase : bool, default False
        If True, skip the final post-FFT quadratic phase.  The resulting
        complex field has the correct *intensity* but not the correct phase
        at the output plane.  Equivalent to the paper's
        ``skip_final_phase=True`` mode; cheaper by one N^2 multiply.

    use_gpu : bool, default False
        Run on CuPy when available.  Like the other propagators, if
        ``E_in`` is already a CuPy array this is honoured automatically.

    verbose : bool, default False
        Print grid, pitch, and band-limit diagnostics.

    Returns
    -------
    E_out : ndarray (complex, N x N)
        Propagated field on a grid of pitch ``dx_out``.

    dx_out : float
        Output grid pitch = ``wavelength * z / (pad * N * dx)``.

    dy_out : float
        Output grid pitch in y (equal to ``dx_out`` for square input).

    Notes
    -----
    Choice of propagator by regime (for free-space diffraction of an N x N
    field, extent L, wavelength lam, distance z):

    * ``z << L^2 / (N * lam)``  — use :func:`angular_spectrum_propagate`
      (Fresnel number large, output pitch = input pitch).
    * ``z ~ L^2 / (N * lam)``   — either ASM or SAS work; SAS gives a
      better-scaled output grid if the beam has diverged past the input
      window.
    * ``z >> L^2 / (N * lam)``  — use SAS.  Plain ASM needs a much larger
      N to avoid aliasing; pure Fresnel loses the phase accuracy that SAS
      recovers through its precompensation term.
    * ``z -> infinity``         — :func:`fraunhofer_propagate`.

    Sampling assumption: the input field is centred in its array and the
    returned array is centred (fftshift applied to the SAS output).  This
    differs from the reference notebook which returns FFT-natural order.

    References
    ----------
    [1] Heintzmann, R.; Loetgering, L.; Wechsler, F. (2023).  "Scalable
        angular spectrum propagation".  *Optica* 10(11): 1407-1416.
        doi:10.1364/OPTICA.497809
    [2] Reference PyTorch implementation:
        https://github.com/bionanoimaging/Scalable-Angular-Spectrum-Method-SAS
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper (replaces the v4.15.2 inline
    # guard).  (SAS already had an ``E_in.ndim != 2`` check downstream;
    # this surfaces the same error with the canonical
    # iterate-over-ensemble message and runs *before* any other input
    # validation.)
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'scalable_angular_spectrum_propagate')

    _validate_propagator_inputs(E_in, z, wavelength, dx, None,
                                fn_name='scalable_angular_spectrum_propagate')
    # 4.9 fix (audit #3.3): SAS's exp(j·k·z·(h_AS - h_Fr)) precompensation
    # factor can give exp(+real) blow-up for z < 0 with evanescent
    # components, so back-propagation is not supported.
    if z <= 0:
        raise ValueError(
            f"scalable_angular_spectrum_propagate: z must be > 0 "
            f"(got z={z}).  SAS is a forward-only propagator (the "
            f"precompensation phase isn't sign-symmetric).  Use "
            f"angular_spectrum_propagate or rayleigh_sommerfeld_propagate "
            f"for back-propagation.")
    # -- array library selection (NumPy vs. CuPy) ---------------------------
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = _state.cp
        if not _is_cupy_array(E_in):
            E_in = _state.cp.asarray(E_in)
    else:
        xp = np
        if _is_cupy_array(E_in):
            E_in = E_in.get()

    # -- validate input ------------------------------------------------------
    if E_in.ndim != 2:
        raise ValueError(
            f"scalable_angular_spectrum_propagate: expected 2-D field, "
            f"got shape {E_in.shape}.")
    Ny, Nx = E_in.shape
    if Ny != Nx:
        raise ValueError(
            f"scalable_angular_spectrum_propagate: input must be square "
            f"(got {Ny}x{Nx}).  The SAS kernel is derived for a single N.")
    N = Nx
    L = N * dx
    pad = int(pad)
    if pad < 1:
        raise ValueError(f"pad must be >= 1, got {pad}")

    # -- closed-form z-limit from Heintzmann et al. (2023) -------------------
    # Beyond z_limit the band-limit filter W kills the ASM-like components
    # that the precompensation phase is meant to correct, and SAS reduces to
    # plain Fresnel with the usual far-field error.
    lam = wavelength
    s = L ** 2 / (8 * L ** 2 + N ** 2 * lam ** 2)
    denom = lam * (-1.0 + 2.0 * np.sqrt(2.0) * np.sqrt(s))
    if abs(denom) < 1e-30:
        z_limit = float("inf")
    else:
        z_limit = float(
            -4.0 * L * np.sqrt(8.0 * L ** 2 / N ** 2 + lam ** 2)
            * np.sqrt(s) / denom)
    if z > z_limit > 0 and verbose:
        print(f"  SAS: z = {z*1e3:.2f} mm exceeds z_limit = "
              f"{z_limit*1e3:.2f} mm; accuracy may degrade.")

    # -- padded grid ---------------------------------------------------------
    L_new = pad * L
    N_new = pad * N

    # -- choose precision from input dtype (matches angular_spectrum_propagate)
    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(_state.DEFAULT_COMPLEX_DTYPE)
    target_fdtype = (np.float32
                     if target_cdtype == np.complex64 else np.float64)

    # -- zero-pad the input, centred ----------------------------------------
    # 4.12.0 (audit round-4 B1-5): `as1 = (N + 1) // 2` was only
    # correct for pad=2 (then `N_new = 2*N` and `(N_new - N)//2 = N/2`
    # ≈ `(N+1)//2`).  For pad=4 with N=512, `(N+1)//2 = 256` but
    # `N_new/2 = 1024` -- input ends up off-centre by ~N/4 pixels.
    # The correct centring is `(N_new - N) // 2`.
    psi_p = xp.zeros((N_new, N_new), dtype=target_cdtype)
    as1 = (N_new - N) // 2
    psi_p[as1:as1 + N, as1:as1 + N] = E_in

    # Kernel cache (NumPy backend)
    # SAS builds three padded-grid kernels per call: the ASM-minus-Fresnel
    # precompensation ``delta_H``, the Fresnel input chirp ``H1``, and
    # (when skip_final_phase is False) the output-plane quadratic phase
    # ``H2``.  All three depend only on (N_new, dx, lam, z,
    # skip_final_phase, dtype), so we bundle them under a single cache
    # entry keyed on that signature.  Cached as a tuple
    # ``(delta_H, H1, H2_or_None)``; _entry_bytes accounts for tuple
    # bundles in the H-cache byte budget.
    h_key = None
    cached = None
    if xp is np:
        h_key = (int(N_new), float(dx), float(lam), float(z),
                 bool(skip_final_phase),
                 np.dtype(target_cdtype).str, 'SAS')
        cached = _h_cache_lookup(h_key)

    if cached is None:
        # -- spatial-frequency axes (natural FFT order) -------------------
        #   fftfreq(N_new, d=L_new/N_new) = fftfreq(N_new, d=dx)
        f_x = xp.fft.fftfreq(N_new, d=dx).astype(target_fdtype)
        f_y = f_x  # square grid

        # -- band-limit W: ASM-vs-Fresnel validity region -----------------
        # Paper eq. (12): the precompensation is valid wherever both
        # inequalities hold, else drop the mode.
        two_z = 2.0 * z if z != 0 else 1e-30
        cx = lam * f_x[None, :]
        cy = lam * f_y[:, None]
        tx = L_new / two_z + xp.abs(cx)
        ty = L_new / two_z + xp.abs(cy)
        W = ((cx ** 2 * (1.0 + tx ** 2) / tx ** 2 + cy ** 2 <= 1.0)
             & (cy ** 2 * (1.0 + ty ** 2) / ty ** 2 + cx ** 2 <= 1.0))

        # -- ASM-minus-Fresnel precompensation phase ----------------------
        #   H_AS  = sqrt(1 - (lam*fx)^2 - (lam*fy)^2)
        #   H_Fr  = 1 - ((lam*fx)^2 + (lam*fy)^2) / 2
        #   delta_H = W * exp( i * k * z * (H_AS - H_Fr) )
        k = 2 * np.pi / lam
        h_AS = xp.sqrt((1.0 + 0j) - cx ** 2 - cy ** 2)
        h_Fr = 1.0 - 0.5 * (cx ** 2 + cy ** 2)
        delta_H = W * xp.exp(1j * k * z * (h_AS - h_Fr))
        delta_H = delta_H.astype(target_cdtype, copy=False)

        # -- Fresnel chirp on natural-order grid --------------------------
        coord_centred = xp.linspace(
            -L_new / 2, L_new / 2, N_new, endpoint=False,
            dtype=target_fdtype)
        coord_nat = xp.fft.ifftshift(coord_centred)
        x = coord_nat[None, :]
        y = coord_nat[:, None]
        H1 = xp.exp(1j * k / (2.0 * z) * (x ** 2 + y ** 2))
        if H1.dtype != target_cdtype:
            H1 = H1.astype(target_cdtype, copy=False)

        # -- output-plane quadratic phase (optional) ----------------------
        if skip_final_phase:
            H2 = None
        else:
            dq = lam * z / L_new  # output pitch on padded grid
            Q = dq * N_new        # full extent of padded output grid
            q_centred = xp.linspace(
                -Q / 2, Q / 2, N_new, endpoint=False, dtype=target_fdtype)
            q_nat = xp.fft.ifftshift(q_centred)
            qx = q_nat[None, :]
            qy = q_nat[:, None]
            H2 = xp.exp(1j * k * z) * xp.exp(
                1j * k / (2.0 * z) * (qx ** 2 + qy ** 2))
            if H2.dtype != target_cdtype:
                H2 = H2.astype(target_cdtype, copy=False)

        if h_key is not None:
            _h_cache_store(h_key, (delta_H, H1, H2))
    else:
        delta_H, H1, H2 = cached

    # -- apply precompensation in frequency space ---------------------------
    # The reference uses ifftshift(psi_p) then fft2, i.e. treat the centred
    # array as if its zero-pixel is at the centre.  Match exactly.
    psi_precomp = xp.fft.ifft2(
        xp.fft.fft2(xp.fft.ifftshift(psi_p)) * delta_H)

    # -- Fresnel chirp + single FFT -----------------------------------------
    # Fresnel-style amplitude prefactor so the output is the physical
    # diffracted field (not a raw DFT sample).  This matches the
    # normalization used by fresnel_propagate in this library and is
    # absent from the reference PyTorch notebook.
    amp_pref = (dx * dx) / (1j * lam * z)
    if skip_final_phase:
        psi_p_final = amp_pref * xp.fft.fftshift(
            xp.fft.fft2(H1 * psi_precomp))
    else:
        psi_p_final = amp_pref * xp.fft.fftshift(
            H2 * xp.fft.fft2(H1 * psi_precomp))

    # -- crop back to original N x N ----------------------------------------
    E_out = psi_p_final[as1:as1 + N, as1:as1 + N]

    # -- output grid pitch ---------------------------------------------------
    dx_out = lam * z / (pad * N * dx)

    if verbose:
        print(f"  SAS propagation: z = {z*1e3:.3f} mm")
        print(f"  Input  grid: {N}x{N}  pitch {dx*1e6:.3f} um  "
              f"extent {L*1e3:.3f} mm")
        print(f"  Output grid: {N}x{N}  pitch {dx_out*1e6:.3f} um  "
              f"extent {N*dx_out*1e3:.3f} mm  "
              f"(zoom {dx_out/dx:.2f}x)")
        # delta_H = W * exp(...) is zero exactly where W is, so the
        # nonzero count of delta_H reproduces W's count without needing
        # W itself (which is now scoped inside the cache-miss branch).
        kept = float(xp.sum(delta_H != 0)) / (N_new * N_new)
        print(f"  Band-limit kept: {kept*100:.1f}% of SAS spectrum")
        if z_limit > 0:
            print(f"  z_limit from paper: {z_limit*1e3:.2f} mm")

    return E_out, dx_out, dx_out

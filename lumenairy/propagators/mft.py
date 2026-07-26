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

import warnings
from typing import Optional, Tuple

import numpy as np

from . import fft_infra as _state
from .asm import _asm_H_from_kz
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


def _validate_mft_output_grid(dx_out, dy_out, N_out, *, fn_name):
    """Validate the caller-specified MFT output grid (audit P11).

    Pre-v5.30 none of the three MFT propagators checked ``dx_out`` /
    ``dy_out`` / ``N_out`` at all, so the Bluestein sampler happily
    consumed garbage:

    * ``dx_out = -1e-6`` was ACCEPTED and returned a finite field on a
      silently x-mirrored grid (measured);
    * ``dx_out = nan`` was ACCEPTED and returned an all-NaN field
      (measured);
    * ``dx_out = 0`` died with a bare ``ZeroDivisionError`` from inside
      the chirp construction;
    * ``N_out <= 0`` died with ``ValueError: N_out must be positive, got
      (0, 0)`` from ``_bluestein_centred_2d`` -- correct, but naming an
      internal tuple instead of the propagator the user called.

    Raising here, up front, names the offending parameter and the
    function the caller actually invoked.
    """
    for name, val in (('dx_out', dx_out), ('dy_out', dy_out)):
        if val is None:
            continue
        v = float(val)
        if not np.isfinite(v):
            raise ValueError(
                f'{fn_name}: {name} must be a finite positive output pitch '
                f'in metres; got {val!r}.')
        if v <= 0.0:
            raise ValueError(
                f'{fn_name}: {name} must be > 0 (got {val!r}).  The output '
                f'grid is built as ``(arange(N_out) - N_out/2) * {name} + '
                f'centre``, so a zero pitch collapses it to a point and a '
                f'negative pitch silently mirrors the axis.  Pass '
                f'centre_out=... to move the window instead.')
    n = int(N_out)
    if n < 1:
        raise ValueError(
            f'{fn_name}: N_out must be >= 1 (got {N_out!r}).')


def _warn_mft_output_window(period_x, period_y, dx_out, dy_out, N_out, *,
                            fn_name, period_expr):
    """Warn when the requested output window exceeds one spatial period of
    the discrete transform, i.e. when the extra samples are periodic
    REPLICAS rather than new information (audit P11).

    A discrete transform of ``N`` samples reconstructs a field that is
    periodic in the output coordinate; sampling beyond one period wraps.
    The period differs per kernel and this helper takes it as an argument:

    * ASM-MFT Bluestein-inverts the SPECTRUM (``N_in`` bins at
      ``df = 1/(N_in*dx_in)``), so the period is ``1/df = N_in*dx_in``;
    * Fresnel- / Fraunhofer-MFT transform the FIELD (``N_in`` samples at
      ``dx_in``), so the period is ``lambda*|z|/dx_in``.

    Both were verified by measurement (N_in=64, dx_in=2 um,
    lambda=633 nm, z=0.5 mm, a narrow Gaussian): at a 4x-period window
    the centre-row profile shows three equal lobes separated by exactly
    the predicted period, at 1x and 2x it shows one.  The natural /
    same-grid calls land exactly ON one period, so the strict ``>``
    comparison leaves them silent.
    """
    n = int(N_out)
    win_x = n * float(dx_out)
    win_y = n * float(dy_out)
    tol = 1.0 + 1e-9
    bad = []
    if win_x > float(period_x) * tol:
        bad.append(('x', win_x, float(period_x), float(dx_out)))
    if win_y > float(period_y) * tol:
        bad.append(('y', win_y, float(period_y), float(dy_out)))
    if not bad:
        return
    detail = '; '.join(
        f'{ax}: window N_out*d{ax}_out = {n} * {d:.6e} = {w:.6e} m '
        f'vs period {p:.6e} m ({w / p:.4g}x)'
        for ax, w, p, d in bad)
    axes = ' and '.join(ax for ax, _w, _p, _d in bad)
    raw = ', '.join(f'{p:.6e}' for _ax, _w, p, _d in bad)
    warnings.warn(
        f"{fn_name}: the requested output window exceeds one spatial period "
        f"of the discrete transform on {axes} -- {detail}.  The period is "
        f"{period_expr} (= {raw} m here); samples beyond +/-period/2 of "
        f"centre_out are PERIODIC REPLICAS of the field, not new "
        f"information, so a broad or structured field will alias into the "
        f"outer part of the window.  Reduce N_out*d_out below the period, "
        f"or use a propagator whose natural grid already spans the region "
        f"you need.",
        UserWarning, stacklevel=3)


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
        Apply Matsushima-Shimobaba band-limiting
        (``|f| < L / (2*lambda*|z|)``) to the ASM transfer function on the
        input frequency grid.  Same default and effect as
        :func:`angular_spectrum_propagate`.  v5.30 (audit P12): that
        cutoff is the **z -> infinity asymptote** of the paper's exact
        local-frequency limit ``1/(lambda*sqrt((2z/L)^2 + 1))``, not the
        exact limit; being the larger of the two it never over-filters.
        See :func:`~lumenairy.propagators.fft_infra._get_or_make_bandlimit`.
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

    Raises
    ------
    ValueError
        If ``dx_out`` / ``dy_out`` is non-finite or ``<= 0``, or
        ``N_out < 1`` (v5.30, audit P11 -- previously ``dx_out < 0`` was
        accepted and returned a finite field on a silently mirrored grid,
        and ``dx_out = nan`` returned an all-NaN field).

    Warns
    -----
    UserWarning
        When ``N_out * dx_out`` exceeds ``Nx_in * dx_in`` (or the y
        equivalent).  The Bluestein step inverse-transforms the input
        SPECTRUM, whose reconstruction is periodic with period
        ``N_in * d_in``; beyond that the extra samples are periodic
        REPLICAS, not new information (v5.30, audit P11).  The same-grid
        call sits exactly on one period and is silent.

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
    _check_2d_scalar_field(E_in, 'angular_spectrum_propagate_mft',
                           input_kind='field')

    _validate_propagator_inputs(E_in, z, wavelength, dx_in, dy_in,
                                fn_name='angular_spectrum_propagate_mft')
    # v5.30 (audit P11): validate the caller-specified OUTPUT grid too --
    # ``_validate_propagator_inputs`` only covers the input side.
    _validate_mft_output_grid(dx_out, dy_out, N_out,
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

    # v5.30 (audit P11): the Bluestein step below inverse-transforms the
    # INPUT SPECTRUM (Nx_in bins at df = 1/(Nx_in*dx_in)), so the field it
    # reconstructs is periodic in x with period 1/df = Nx_in*dx_in (and
    # likewise in y).  Asking for a window wider than that returns
    # periodic replicas, silently -- warn with the numbers.
    _warn_mft_output_window(
        Nx_in * dx_in, Ny_in * dy_in, dx_out, dy_out, Ny_out,
        fn_name='angular_spectrum_propagate_mft',
        period_expr='N_in*d_in (the input cell, since the Bluestein step '
                    'inverts the input spectrum)')

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
    # v5.24.4 (audit S2-3): H is FIELD-INDEPENDENT (input geometry,
    # wavelength and z only), so build it on the HOST in float64 and
    # cache it, then move it onto the active backend.  Building it under
    # the JAX tracer instead silently evaluated the kernel argument
    # ``kz * z`` (up to ~1e6 rad) in float32 whenever ``jax_enable_x64``
    # is off (the JAX default) -- ``jnp.arange(dtype=float64)`` truncates
    # to float32 there -- losing ~26 dB of phase accuracy vs the NumPy
    # contract.  Host-building keeps the field gradient intact (H does
    # not depend on the field); only concrete-float geometry gradients
    # are foregone.  NumPy / CuPy paths are byte-identical to before,
    # and JAX now shares the same cached, f64-built H.  4.12.0: strict
    # `<` band-limit (Matsushima-Shimobaba open interval; matches plain
    # ASM).  v5.30 (audit P12): the cutoff below is the z -> infinity
    # ASYMPTOTE ``L / (2*lambda*|z|)`` of the paper's exact
    # local-frequency limit, not that limit -- strictly larger, so it
    # never over-filters.  See ``fft_infra._get_or_make_bandlimit``.
    h_key = (int(Ny_in), int(Nx_in), float(dy_in), float(dx_in),
             float(wavelength), float(z),
             bool(bandlimit),
             np.dtype(target_cdtype).str, 'ASM_MFT')
    H_np = _h_cache_lookup(h_key)
    if H_np is None:
        # audit P1 (2026-07-25): INTEGER DC anchor ``N // 2``.  H multiplies
        # ``fftshift(fft2(ifftshift(E_in)))`` below, and fftshift anchors DC
        # at the integer centred index for every N, so the centred bin
        # labels must use ``N // 2``.  Bit-identical for even N; for ODD N
        # the float ``N / 2`` mislabelled every bin by -df/2 and the kernel
        # was evaluated at ``f_true - df/2`` -- a lateral walk of
        # ``-lambda*z/(2*N*dx)`` (measured -3.89 px, rel err 1.5e-1 vs the
        # Gaussian-ABCD oracle at N=257).  ``n_centre_in_*`` in the
        # Bluestein call below carries the SAME anchor (it is the
        # frequency-bin centre, not a spatial one).
        fx = (np.arange(Nx_in, dtype=np.float64) - Nx_in // 2) / (Nx_in * dx_in)
        fy = (np.arange(Ny_in, dtype=np.float64) - Ny_in // 2) / (Ny_in * dy_in)
        kx_sq = (2.0 * np.pi * fx) ** 2
        ky_sq = (2.0 * np.pi * fy) ** 2
        kz_sq = k * k - kx_sq[None, :] - ky_sq[:, None]
        prop_mask = kz_sq > 0
        kz = np.where(prop_mask,
                      np.sqrt(np.where(prop_mask, kz_sq, 0.0)), 0.0)
        # S2-10: shared kernel.  complex128 is byte-identical to the former
        # ``np.where(prop, np.exp(1j*kz*z), 0).astype(...)``; complex64 now
        # folds the phase mod 2*pi in float64 before the float32 cast
        # (S2-3 mitigation), matching every other ASM H builder.
        H_np = _asm_H_from_kz(kz, prop_mask, z, target_cdtype, np)
        if bandlimit and z != 0:
            Lx_phys = Nx_in * dx_in
            Ly_phys = Ny_in * dy_in
            fx_max = Lx_phys / (2.0 * wavelength * abs(z))
            fy_max = Ly_phys / (2.0 * wavelength * abs(z))
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
    #   fx[nx] = (nx - Nx_in//2) / (Nx_in * dx_in)  (with dfx = 1/(Nx_in * dx_in))
    #   x_out[kx] = (kx - Nx_out/2) * dx_out + xc
    # The product fx[nx] * x_out[kx] expands to a centred Bluestein form
    # with alpha = dfx * dx_out = dx_out / (Nx_in * dx_in) and sign = +1.
    # The 1/(Nx_in*Ny_in) prefactor matches numpy/scipy's IFFT normalisation
    # so that round-tripping through ASM-MFT recovers the input on the
    # natural grid.
    #
    # audit P1 (2026-07-25), odd N: the ``ifftshift`` in step 2 makes the
    # spectrum's implicit spatial origin the INTEGER pixel ``N_in // 2``,
    # while this family's documented input/output coordinate convention is
    # ``x = (n - N/2)*dx`` (the convention ``fresnel_propagate_mft`` and
    # ``fraunhofer_propagate_mft`` evaluate directly, with no shifts).  For
    # odd N_in the two origins differ by half an input pixel, so the
    # reconstruction coordinate is ``x_out + off_in`` with
    # ``off_in = (N_in/2 - N_in//2)*d_in``.  Folding that offset into the
    # output centre is exact and keeps the declared grid: post-fix
    # ASM-MFT reproduces angular_spectrum_propagate on the same grid to
    # 4.4e-14 at N=257 (pre-fix: rel err 1.5e-1, centroid -3.39 px).
    # ``off_in`` is exactly 0.0 for even N_in -> bit-identical.
    alpha_x = dx_out / (Nx_in * dx_in)
    alpha_y = dy_out / (Ny_in * dy_in)
    off_in_x = (Nx_in / 2.0 - Nx_in // 2) * dx_in
    off_in_y = (Ny_in / 2.0 - Ny_in // 2) * dy_in
    kc_x = Nx_out / 2.0 - (xc + off_in_x) / dx_out
    kc_y = Ny_out / 2.0 - (yc + off_in_y) / dy_out

    F = _bluestein_centred_2d(
        A_propagated, alpha_x, alpha_y, Ny_out, Nx_out,
        # Frequency-bin centre: must match the ``fx`` / ``fy`` anchor above.
        n_centre_in_x=float(Nx_in // 2),
        n_centre_in_y=float(Ny_in // 2),
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
        S11-6a: must resolve to ``>= 1`` -- a ``dx_out`` so coarse that
        the default rounds to 0 raises ``ValueError`` instead of silently
        returning a ``(0, 0)`` array.
    order : int, default 3
        Interpolation order (0=nearest, 1=linear, 3=cubic, 5=quintic).
        S11-6a: validated to an integer in ``[0, 5]`` -- out-of-range
        values used to surface as a scipy-internal
        ``RuntimeError: spline order not supported`` naming neither this
        function nor the argument.

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
    _check_2d_scalar_field(E_in, 'resample_field', input_kind='field')
    from scipy.ndimage import map_coordinates

    # S11-6a (AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1, "harness knobs
    # must ERROR on unrecognised values"): ``order`` is passed straight
    # to ``map_coordinates``, which accepts 0..5 -- but the docstring
    # advertises 1 / 3 / 5, and an out-of-range value used to surface as
    # a scipy-internal ``RuntimeError: spline order not supported``
    # naming neither this function nor the argument.
    if int(order) != order or not (0 <= int(order) <= 5):
        raise ValueError(
            f"resample_field: order must be an integer in [0, 5] "
            f"(0=nearest, 1=linear, 3=cubic, 5=quintic); got {order!r}.")
    if not np.isfinite(dx_in) or dx_in <= 0:
        raise ValueError(
            f"resample_field: dx_in must be positive and finite; "
            f"got {dx_in!r}.")
    if not np.isfinite(dx_out) or dx_out <= 0:
        raise ValueError(
            f"resample_field: dx_out must be positive and finite; "
            f"got {dx_out!r}.")

    Ny_in, Nx_in = E_in.shape
    if N_out is None:
        Nx_out = int(round(Nx_in * dx_in / dx_out))
        Ny_out = int(round(Ny_in * dx_in / dx_out))
        # S11-6a: the extent-preserving default silently rounds to 0 once
        # ``dx_out`` exceeds the whole input extent (``dx_out >
        # 2 * N_in * dx_in``), and the function then returned a (0, 0)
        # array -- a shape-valid, physics-free result that only failed
        # much later downstream.  Measured: ``N_in = 64``,
        # ``dx_in = 1e-6``, ``dx_out = 1e-3`` -> ``round(0.064) = 0`` ->
        # ``E_out.shape == (0, 0)``.
        if Nx_out < 1 or Ny_out < 1:
            raise ValueError(
                f"resample_field: the extent-preserving default "
                f"N_out = round(N_in * dx_in / dx_out) rounded to "
                f"({Ny_out}, {Nx_out}) -- dx_out={dx_out!r} is too coarse "
                f"to place even one sample across the input extent "
                f"({Ny_in}x{Nx_in} @ dx_in={dx_in!r}, i.e. "
                f"{Nx_in * dx_in!r} m).  Pass an explicit N_out >= 1 if "
                f"you really want a coarser-than-the-field grid.")
    else:
        Nx_out = Ny_out = int(N_out)
        if Nx_out < 1:
            raise ValueError(
                f"resample_field: N_out must be >= 1; got {N_out!r}.")

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

    Raises
    ------
    ValueError
        If ``dx_out`` / ``dy_out`` is non-finite or ``<= 0``, or
        ``N_out < 1`` (v5.30, audit P11).

    Warns
    -----
    UserWarning
        When ``N_out * dx_out`` exceeds ``lambda*|z|/dx_in`` (or the y
        equivalent) -- the spatial period of a Bluestein transform of the
        input FIELD.  Beyond one period the extra samples are periodic
        REPLICAS (v5.30, audit P11).  The natural Fresnel grid
        (``dx_out = lambda*z/(N*dx_in)``, ``N_out = N_in``) sits exactly on
        one period and is silent.

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
    _check_2d_scalar_field(E_in, 'fresnel_propagate_mft', input_kind='field')

    _validate_propagator_inputs(E_in, z, wavelength, dx_in, dy_in,
                                fn_name='fresnel_propagate_mft')
    # v5.30 (audit P11): validate the caller-specified OUTPUT grid too.
    _validate_mft_output_grid(dx_out, dy_out, N_out,
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

    # v5.30 (audit P11): the Bluestein step transforms the INPUT FIELD
    # (Nx_in samples at dx_in) with kernel ``exp(-i k u x / z)``, so the
    # output is periodic in x with period ``lambda*|z|/dx_in`` (NOT the
    # input cell -- that is the ASM-MFT case).  A wider window returns
    # periodic replicas; warn with the numbers.  ``dx_out = lambda*z /
    # (N*dx_in)`` with ``N_out = N_in`` (the natural Fresnel grid) sits
    # exactly on one period and stays silent.
    _warn_mft_output_window(
        wavelength * abs(z) / dx_in, wavelength * abs(z) / dy_in,
        dx_out, dy_out, Ny_out,
        fn_name='fresnel_propagate_mft',
        period_expr='lambda*|z|/d_in (the transform is of the input '
                    'field, not its spectrum)')

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

    Raises
    ------
    ValueError
        If ``dx_out`` / ``dy_out`` is non-finite or ``<= 0``, or
        ``N_out < 1`` (v5.30, audit P11).

    Warns
    -----
    UserWarning
        When ``N_out * dx_out`` exceeds ``lambda*z/dx_in`` (or the y
        equivalent) -- the spatial period of a Bluestein transform of the
        input FIELD.  Beyond one period the extra samples are periodic
        REPLICAS (v5.30, audit P11).  The natural Fraunhofer grid sits
        exactly on one period and is silent.

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
    _check_2d_scalar_field(E_in, 'fraunhofer_propagate_mft',
                           input_kind='field')

    _validate_propagator_inputs(E_in, z, wavelength, dx_in, dy_in,
                                fn_name='fraunhofer_propagate_mft')
    # v5.30 (audit P11): validate the caller-specified OUTPUT grid too.
    _validate_mft_output_grid(dx_out, dy_out, N_out,
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

    # v5.30 (audit P11): same period as Fresnel-MFT -- the transform is of
    # the input FIELD, so the output is periodic with period
    # ``lambda*|z|/d_in``.  Warn when the requested window exceeds it.
    _warn_mft_output_window(
        wavelength * abs(z) / dx_in, wavelength * abs(z) / dy_in,
        dx_out, dy_out, Ny_out,
        fn_name='fraunhofer_propagate_mft',
        period_expr='lambda*|z|/d_in (the transform is of the input '
                    'field, not its spectrum)')

    # PK-3: the input-plane index grids were removed with the Fraunhofer
    # (no input quadratic phase) simplification; only the OUTPUT index grids
    # below are used.  The two bare ``np.arange(N*_in)`` statements that
    # remained were computed-and-discarded refactor residue.
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

"""
Bluestein chirp-Z transform (internal helper).

Provides ``_bluestein_2d``, the building block used by Lumenairy's
matrix-Fourier-transform-based propagators

    fresnel_propagate_mft
    fraunhofer_propagate_mft
    angular_spectrum_propagate_mft

to evaluate a 2-D discrete Fourier integral on an arbitrary user-supplied
output grid (different size and pitch from the input grid) without
zero-padding the input.

Algorithm
---------
Given a 2-D input ``E[ny, nx]`` of shape ``(Ny_in, Nx_in)``, evaluate

.. math::

    F[k_y, k_x] = \\sum_{n_y=0}^{N_{y,\\rm in}-1}\\sum_{n_x=0}^{N_{x,\\rm in}-1}
        E[n_y, n_x]\\,
        \\exp\\!\\bigl(\\sigma\\,2\\pi\\,j\\,(\\alpha_x n_x k_x + \\alpha_y n_y k_y)\\bigr)

for ``kx`` in ``[0, N_out_x)`` and ``ky`` in ``[0, N_out_y)``, where
``sigma = +/- 1`` selects the inverse / forward transform direction
respectively.

Bluestein's identity ``n*k = (n^2 + k^2 - (n-k)^2) / 2`` lets us write

.. math::

    F[k] = e^{\\sigma\\pi j\\alpha k^2}
        \\bigl(g \\ast h\\bigr)[k],
        \\quad g[n] = E[n]\\,e^{\\sigma\\pi j\\alpha n^2},
        \\quad h[m] = e^{-\\sigma\\pi j\\alpha m^2}.

The convolution is computed with two zero-padded 2-D FFTs, giving total
cost ``O((N + M) \\log (N + M))`` per axis, where ``N = N_in`` and
``M = N_out``.  This is dramatically faster than a direct matrix-Fourier
transform (``O(N^2 M^2)``) for typical focal-zoom workflows.

Backends
--------
The 2-D FFTs are dispatched through caller-supplied ``fft2`` / ``ifft2``
callables, so the same helper drives the NumPy / pyFFTW / SciPy / CuPy /
JAX paths used by the rest of :mod:`lumenairy.propagators.propagation`.

The helper is internal (underscore prefix) -- end users should call the
public ``*_propagate_mft`` propagators in
:mod:`lumenairy.propagators.propagation` instead.

Author:  Andrew Traverso
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np
from scipy.fft import next_fast_len

# Chirp-kernel FFT cache: H_FFT = fft2(h_2d) depends only on
# (alpha_x, alpha_y, Nx_in, Ny_in, N_out_x, N_out_y, sign, dtype) -- not
# on the input field -- and costs one of the three FFTs per call.
# NumPy-default-backend entries only (see guard in :func:`_bluestein_2d`).
# Entries are np.copy'd on store AND on hit: the pyFFTW double-buffer
# hands out its internal output buffer, which the very next fft2/ifft2
# call overwrites -- there is zero buffer-ownership slack between the
# G_FFT / H_FFT pair.
_H_FFT_CACHE: 'OrderedDict[tuple, np.ndarray]' = OrderedDict()
_H_FFT_CACHE_MAXSIZE = 16
_H_FFT_CACHE_LOCK = threading.Lock()
_H_FFT_CACHE_HITS = 0


def _clear_h_fft_cache() -> None:
    """Drop every cached Bluestein chirp-kernel FFT."""
    global _H_FFT_CACHE_HITS
    with _H_FFT_CACHE_LOCK:
        _H_FFT_CACHE.clear()
        _H_FFT_CACHE_HITS = 0


try:
    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _register_cache_clearer('bluestein_h_fft', _clear_h_fft_cache)
except ImportError:
    pass


def _bluestein_2d(
    E,
    alpha_x: float,
    alpha_y: float,
    N_out_y: int,
    N_out_x: int,
    *,
    sign: int,
    xp,
    fft2,
    ifft2,
    target_cdtype=None,
):
    """2-D Bluestein chirp-Z transform.

    Computes::

        F[ky, kx] = sum_{ny, nx} E[ny, nx]
                    * exp(sign * 2*pi*j * (alpha_x * nx * kx + alpha_y * ny * ky))

    over indices ``nx, ny, kx, ky`` all starting at zero.  Use
    propagator wrappers in :mod:`propagation` to handle centred
    conventions (``(n - N/2)`` shifts).

    Parameters
    ----------
    E : ndarray, complex 2-D
        Input array of shape ``(Ny_in, Nx_in)``.  The array library
        (NumPy / CuPy / JAX) is determined by ``xp``.
    alpha_x, alpha_y : float
        Sampling-rate parameters in the Bluestein sum (cycles per
        index pair on each axis).  For a Fresnel propagator,
        ``alpha = dx_in * dx_out / (lambda * z)``.
    N_out_y, N_out_x : int
        Output grid size.
    sign : int
        ``+1`` for the inverse-FT direction (e.g., the IFT step in
        :func:`angular_spectrum_propagate_mft`).
        ``-1`` for the forward-FT direction (e.g., the FT step in
        :func:`fresnel_propagate_mft`).
    xp : module
        Array namespace -- ``numpy``, ``cupy``, or ``jax.numpy``.  The
        same module that produced ``E``.
    fft2, ifft2 : callable
        2-D FFT and inverse FFT functions appropriate for ``xp``.  For
        NumPy this is typically ``lumenairy.propagators.propagation._fft2``
        (which dispatches to pyFFTW when applicable); for CuPy /
        JAX, pass ``cp.fft.fft2`` / ``jnp.fft.fft2`` etc.
    target_cdtype : numpy dtype, optional
        Complex dtype of the output array.  Inferred from ``E`` when
        ``None``.  Chirp signals are computed in float64 for accuracy
        and cast to ``target_cdtype`` before multiplication; this
        avoids float32 chirp-phase precision loss at large indices.

    Returns
    -------
    F : ndarray, complex
        Output array of shape ``(N_out_y, N_out_x)`` in the same
        backend as ``E``.

    Notes
    -----
    Numerical sensitivity: the chirp value ``exp(sign*pi*j*alpha*n^2)``
    can wrap around float64 phase precision if ``alpha * N^2 >> 1e16``
    (i.e., the phase argument exceeds ~10^16 radians, beyond float64
    precision).  In typical Fresnel / Fraunhofer propagation parameters
    this never happens.  A guard warns the caller if ``alpha * N_max^2``
    exceeds 1e15.
    """
    if sign not in (+1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")

    Ny_in, Nx_in = E.shape
    N_out_y = int(N_out_y)
    N_out_x = int(N_out_x)
    if N_out_y < 1 or N_out_x < 1:
        raise ValueError(
            f"N_out must be positive, got ({N_out_y}, {N_out_x})")

    if target_cdtype is None:
        target_cdtype = np.dtype(E.dtype) if xp.iscomplexobj(E) \
            else np.dtype(np.complex128)
    target_cdtype = np.dtype(target_cdtype)

    # Numerical-precision guard.  The chirp signal exp(sign*pi*j*alpha*n^2)
    # has phase up to pi * |alpha| * N_max^2.  float64 gives ~16 decimal
    # digits, so phases beyond ~10^15 lose meaningful precision.  Beyond
    # ~10^16 the chirp wraps incoherently and the output is garbage.
    N_max = max(Nx_in, Ny_in, N_out_x, N_out_y)
    alpha_max = max(abs(alpha_x), abs(alpha_y))
    phase_budget = float(alpha_max) * float(N_max) ** 2
    if phase_budget > 1e15:
        import warnings
        warnings.warn(
            f"Bluestein chirp phase argument ~{phase_budget:.1e} approaches "
            f"float64 precision limit (1e15-1e16).  Reduce N or alpha, "
            f"or fall back to a regular FFT propagator.",
            RuntimeWarning, stacklevel=2)

    # ----- 1) Bluestein chirp signals (computed in float64 for accuracy) -----
    # Pre / post / kernel use the identity
    #   exp(sigma*2*pi*j*alpha*n*k)
    #     = exp(sigma*pi*j*alpha*n^2)
    #     * exp(sigma*pi*j*alpha*k^2)
    #     * exp(-sigma*pi*j*alpha*(n-k)^2)
    # so the convolution kernel has the OPPOSITE sign of the pre/post chirps.

    n_x = np.arange(Nx_in, dtype=np.float64)
    n_y = np.arange(Ny_in, dtype=np.float64)
    k_x = np.arange(N_out_x, dtype=np.float64)
    k_y = np.arange(N_out_y, dtype=np.float64)

    pre_x_np  = np.exp(1j * sign * np.pi * float(alpha_x) * n_x * n_x)
    pre_y_np  = np.exp(1j * sign * np.pi * float(alpha_y) * n_y * n_y)
    post_x_np = np.exp(1j * sign * np.pi * float(alpha_x) * k_x * k_x)
    post_y_np = np.exp(1j * sign * np.pi * float(alpha_y) * k_y * k_y)

    # ----- 2) Convolution kernel and FFT lengths -----
    # Linear convolution of length-N input with kernel covering m in
    # [-(N-1), M-1] requires output length >= N + M - 1; round up to a
    # cache-friendly FFT-fast length via scipy.fft.next_fast_len (numpy
    # types only -- this is integer arithmetic, no dispatch needed).
    Lx = int(next_fast_len(int(Nx_in + N_out_x - 1)))
    Ly = int(next_fast_len(int(Ny_in + N_out_y - 1)))

    # Fold the kernel into circular indexing on the padded length.
    # Indices [0 .. N_out-1] hold h[0..N_out-1]; indices
    # [Lx-(Nx_in-1) .. Lx-1] hold h[-(Nx_in-1)..-1].  The middle
    # "unused" region is set to whatever exp gives at those indices --
    # since g_pad is zero there, those values don't enter the answer.
    m_x_idx = np.arange(Lx, dtype=np.int64)
    m_x_signed = np.where(m_x_idx < N_out_x, m_x_idx, m_x_idx - Lx).astype(np.float64)
    h_x_np = np.exp(-1j * sign * np.pi * float(alpha_x) * m_x_signed * m_x_signed)

    m_y_idx = np.arange(Ly, dtype=np.int64)
    m_y_signed = np.where(m_y_idx < N_out_y, m_y_idx, m_y_idx - Ly).astype(np.float64)
    h_y_np = np.exp(-1j * sign * np.pi * float(alpha_y) * m_y_signed * m_y_signed)

    # ----- 3) Move to xp + cast to target dtype -----
    def _to_xp(arr_np_complex):
        a = arr_np_complex.astype(target_cdtype, copy=False)
        if xp is np:
            return a
        return xp.asarray(a)

    pre_x  = _to_xp(pre_x_np)
    pre_y  = _to_xp(pre_y_np)
    post_x = _to_xp(post_x_np)
    post_y = _to_xp(post_y_np)
    h_x    = _to_xp(h_x_np)
    h_y    = _to_xp(h_y_np)

    # ----- 4) Modulate input by the pre-chirp + zero-pad to (Ly, Lx) -----
    # Use xp.pad so the JAX path stays functional (no in-place .at[].set).
    g = E * (pre_y[:, None] * pre_x[None, :])
    g_pad = xp.pad(
        g,
        pad_width=((0, Ly - Ny_in), (0, Lx - Nx_in)),
        mode='constant',
        constant_values=0,
    )
    if g_pad.dtype != target_cdtype:
        g_pad = g_pad.astype(target_cdtype)

    # ----- 5) FFT-based circular convolution with the chirp kernel -----
    # The 2-D kernel is separable: h[my, mx] = h_y[my] * h_x[mx].
    G_FFT = fft2(g_pad)
    # Serve H_FFT from the module cache when possible.  NumPy default
    # path ONLY: the fft2 callable is caller-supplied (CuPy / JAX /
    # custom), and keying on anything else would pin device arrays in a
    # module-global or return results from a different transform.
    global _H_FFT_CACHE_HITS
    H_FFT = None
    cache_key = None
    if xp is np:
        from .fft_infra import _fft2 as _default_np_fft2
        if fft2 is _default_np_fft2:
            cache_key = (float(alpha_x), float(alpha_y), Nx_in, Ny_in,
                         N_out_x, N_out_y, sign, str(target_cdtype))
            with _H_FFT_CACHE_LOCK:
                cached = _H_FFT_CACHE.get(cache_key)
                if cached is not None:
                    _H_FFT_CACHE.move_to_end(cache_key)
                    _H_FFT_CACHE_HITS += 1
                    H_FFT = np.copy(cached)
    if H_FFT is None:
        h_2d = h_y[:, None] * h_x[None, :]
        H_FFT = fft2(h_2d)
        if cache_key is not None:
            with _H_FFT_CACHE_LOCK:
                _H_FFT_CACHE[cache_key] = np.copy(H_FFT)
                while len(_H_FFT_CACHE) > _H_FFT_CACHE_MAXSIZE:
                    _H_FFT_CACHE.popitem(last=False)
    CONV  = ifft2(G_FFT * H_FFT)

    # ----- 6) Extract the first (N_out_y, N_out_x) block and apply post-chirp -----
    block = CONV[:N_out_y, :N_out_x]
    F = block * (post_y[:, None] * post_x[None, :])

    # Ensure the output dtype matches the target.  Some backends promote
    # complex64 to complex128 through FFT pairs; force back here so the
    # caller's precision contract is honoured.
    if F.dtype != target_cdtype:
        F = F.astype(target_cdtype)
    return F


def _bluestein_centred_2d(
    E,
    alpha_x: float,
    alpha_y: float,
    N_out_y: int,
    N_out_x: int,
    *,
    n_centre_in_x: float | None = None,
    n_centre_in_y: float | None = None,
    k_centre_out_x: float | None = None,
    k_centre_out_y: float | None = None,
    sign: int,
    xp,
    fft2,
    ifft2,
    target_cdtype=None,
):
    """2-D Bluestein with centred input AND output index conventions.

    Computes::

        F[ky, kx] = sum_{ny, nx} E[ny, nx]
                    * exp(sign*2*pi*j * alpha_x * (nx - cIx) * (kx - cOx))
                    * exp(sign*2*pi*j * alpha_y * (ny - cIy) * (ky - cOy))

    where ``cIx, cIy`` are input centres (default ``Nx_in/2, Ny_in/2``)
    and ``cOx, cOy`` are output centres (default ``Nx_out/2, Ny_out/2``).

    The non-centred primitive :func:`_bluestein_2d` computes
    ``sum E[n] * exp(sign*2*pi*j*alpha*n*k)``.  Centring expands as

    .. math::

        (n - cI)(k - cO) = nk - n\\cdot cO - k\\cdot cI + cI\\cdot cO,

    and the three correction factors fold cleanly into a pre-chirp
    (depends on ``n``), a post-chirp (depends on ``k``), and a
    multiplicative constant.

    Parameters
    ----------
    E : ndarray, complex 2-D
    alpha_x, alpha_y : float
    N_out_y, N_out_x : int
    n_centre_in_x, n_centre_in_y : float, optional
        Defaults to ``Nx_in / 2`` and ``Ny_in / 2`` respectively.  Pass
        a non-default value (e.g. ``0``) to use a left-anchored input.
    k_centre_out_x, k_centre_out_y : float, optional
        Defaults to ``N_out_x / 2`` and ``N_out_y / 2``.  Non-integer
        values are valid -- they correspond to a sub-pixel shift of the
        output centre.
    sign : int
        ``+1`` (inverse FT) or ``-1`` (forward FT).
    xp, fft2, ifft2, target_cdtype
        Same as :func:`_bluestein_2d`.

    Returns
    -------
    F : ndarray, complex 2-D, shape ``(N_out_y, N_out_x)``.
    """
    Ny_in, Nx_in = E.shape
    if n_centre_in_x is None:
        n_centre_in_x = Nx_in / 2.0
    if n_centre_in_y is None:
        n_centre_in_y = Ny_in / 2.0
    if k_centre_out_x is None:
        k_centre_out_x = N_out_x / 2.0
    if k_centre_out_y is None:
        k_centre_out_y = N_out_y / 2.0

    if target_cdtype is None:
        target_cdtype = np.dtype(E.dtype) if xp.iscomplexobj(E) \
            else np.dtype(np.complex128)
    target_cdtype = np.dtype(target_cdtype)

    n_x = np.arange(Nx_in, dtype=np.float64)
    n_y = np.arange(Ny_in, dtype=np.float64)
    k_x = np.arange(N_out_x, dtype=np.float64)
    k_y = np.arange(N_out_y, dtype=np.float64)

    # Decompose (n - cI)(k - cO) = nk - n*cO - k*cI + cI*cO.
    # Then exp(sign*2*pi*j*alpha*(n-cI)(k-cO))
    #   = [exp(sign*2*pi*j*alpha*n*k)]                       (Bluestein core)
    #   * [exp(-sign*2*pi*j*alpha*n*cO)]                      (pre-chirp on n)
    #   * [exp(-sign*2*pi*j*alpha*k*cI)]                      (post-chirp on k)
    #   * [exp(sign*2*pi*j*alpha*cI*cO)]                      (constant)

    pre_x_np  = np.exp(-1j * sign * 2 * np.pi * float(alpha_x)
                        * n_x * float(k_centre_out_x))
    pre_y_np  = np.exp(-1j * sign * 2 * np.pi * float(alpha_y)
                        * n_y * float(k_centre_out_y))
    post_x_np = np.exp(-1j * sign * 2 * np.pi * float(alpha_x)
                        * k_x * float(n_centre_in_x))
    post_y_np = np.exp(-1j * sign * 2 * np.pi * float(alpha_y)
                        * k_y * float(n_centre_in_y))
    const = (np.exp(1j * sign * 2 * np.pi * float(alpha_x)
                     * float(n_centre_in_x) * float(k_centre_out_x))
             * np.exp(1j * sign * 2 * np.pi * float(alpha_y)
                       * float(n_centre_in_y) * float(k_centre_out_y)))

    def _to_xp(arr_np_complex):
        a = arr_np_complex.astype(target_cdtype, copy=False)
        if xp is np:
            return a
        return xp.asarray(a)

    pre_x  = _to_xp(pre_x_np)
    pre_y  = _to_xp(pre_y_np)
    post_x = _to_xp(post_x_np)
    post_y = _to_xp(post_y_np)
    const_c = target_cdtype.type(const)

    # Modulate input by pre-chirp and call the Bluestein primitive.
    E_mod = E * (pre_y[:, None] * pre_x[None, :])
    F_core = _bluestein_2d(
        E_mod, alpha_x, alpha_y, N_out_y, N_out_x,
        sign=sign, xp=xp, fft2=fft2, ifft2=ifft2,
        target_cdtype=target_cdtype,
    )
    F = F_core * (post_y[:, None] * post_x[None, :]) * const_c
    return F


__all__ = ['_bluestein_2d', '_bluestein_centred_2d']

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

# v5.33.2 BYTE CAPS (audit AUDIT_TRACED_MEMORY_2026_08_09 row 7).  The count
# cap above was the ONLY bound, and one entry is ``L^2`` complex128 with
# ``L = next_fast_len(N_in + N_out - 1)`` -- i.e. it scales with the CALLER's
# grid, not with anything this module controls.  MEASURED at design 121's
# shipped readout shapes: ``N_fine`` 8192 / ``N_out`` 1024 -> ``L`` = 9216 ->
# **1.359 GB per entry, 21.7 GB across the 16**; at ``window_factor`` 7 the
# same geometry gives ``L`` = 17424 -> 4.858 GB per entry and **77.7 GB**.
#
# And on a fan that memory buys nothing.  The key carries ``alpha =
# dx_out/(N_in*dx_in)``, and ``dx_in = dx_fine`` differs per congruence (C-2
# measured per-order readout periods 4734.6..4738.3 um, a 0.08 % spread), so
# every order writes a NEW entry and hits NONE: ``hits = 0`` measured after a
# full production order, and reproduced directly with two orders whose alpha
# differs in the 5th digit (2 entries, 0 hits).
#
# ``fft_infra._H_CACHE`` -- the ASM transfer-function cache two files away --
# has carried exactly these two caps since 3.2.14.1, with a comment recording
# the identical lesson ("At N=32768 each H is 16 GB complex128; without this
# cap, an 8-entry cache can hold up to 128 GB").  The numbers here are that
# cache's, deliberately: this is bringing a sibling cache up to a standard the
# library already sets, not inventing a new policy.  A cache is a cache -- no
# accuracy consequence either way, the entry is recomputed on the next miss.
_H_FFT_CACHE_MAX_BYTES_PER_ENTRY = 2 * 1024 * 1024 * 1024   # 2 GB
_H_FFT_CACHE_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024       # 8 GB


def _clear_h_fft_cache() -> None:
    """Drop every cached Bluestein chirp-kernel FFT."""
    global _H_FFT_CACHE_HITS
    with _H_FFT_CACHE_LOCK:
        _H_FFT_CACHE.clear()
        _H_FFT_CACHE_HITS = 0


def _h_fft_cache_bytes() -> int:
    """Total bytes currently retained by the chirp-kernel FFT cache."""
    with _H_FFT_CACHE_LOCK:
        return int(sum(int(v.nbytes) for v in _H_FFT_CACHE.values()))


def _h_fft_cache_store(cache_key, H_FFT) -> None:
    """Store one chirp-kernel FFT under the count AND byte bounds.

    Mirrors ``fft_infra._h_cache_store``: an entry larger than
    ``_H_FFT_CACHE_MAX_BYTES_PER_ENTRY`` is NOT stored at all (the transform
    still returns it -- only the retention is skipped), and after any store the
    oldest entries are evicted until both the count and the total-bytes bounds
    hold.  The globals are read at call time so a caller may retune them.

    One deliberate difference from the sibling: eviction stops at ONE entry.
    ``_h_cache_store`` will empty itself if a caller retunes the total cap
    below a single entry's size, which turns the cache into pure overhead (it
    stores, then immediately drops what it stored).  With the shipped caps
    (2 GiB/entry inside 8 GiB total) neither can reach that state.

    v5.33.3 (VERIFY_PERF_BRANCH_2026_08_10 D3): the size test reads
    ``H_FFT.nbytes`` and runs BEFORE the ``np.copy``, exactly as the sibling's
    ``_entry_bytes(H)`` does.  The first cut copied first and rejected second,
    which converted the retention the cap exists to avoid into an equally
    large TRANSIENT for an entry that is thrown away one line later -- 4.86 GB
    at the ``window_factor = 7`` geometry (``L = 17424``) the cap's own comment
    works through, allocated on the run whose peak is the thing being
    defended.  MEASURED with ``tracemalloc`` at a 1 B per-entry cap: retained
    +0.000 MB either way, traced peak +67.109 MB before / +0.000 MB after.
    """
    if int(getattr(H_FFT, 'nbytes', 0)) > int(_H_FFT_CACHE_MAX_BYTES_PER_ENTRY):
        return
    H = np.copy(H_FFT)
    with _H_FFT_CACHE_LOCK:
        _H_FFT_CACHE[cache_key] = H
        total = sum(int(v.nbytes) for v in _H_FFT_CACHE.values())
        while (len(_H_FFT_CACHE) > int(_H_FFT_CACHE_MAXSIZE)
               or total > int(_H_FFT_CACHE_MAX_TOTAL_BYTES)):
            if len(_H_FFT_CACHE) <= 1:
                break
            _, dropped = _H_FFT_CACHE.popitem(last=False)
            total -= int(dropped.nbytes)


try:
    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _register_cache_clearer('bluestein_h_fft', _clear_h_fft_cache)
except ImportError:
    pass


def _fft_1d(a, axis=-1, inverse=False):
    """1-D FFT along ``axis`` through the library's own NumPy-side dispatch.

    Mirrors ``fft_infra._scipy_or_numpy_fft2``'s choice (SciPy's threaded
    pocketfft when ``USE_SCIPY_FFT``, NumPy otherwise) so the separable
    Bluestein path honours the same backend switches as every other
    transform.  pyFFTW is not consulted: its cached plans are 2-D and a
    1-D plan family would double the resident workspace this module's
    separable path exists to remove.
    """
    from . import fft_infra as _fi
    if _fi.USE_SCIPY_FFT and _fi.SCIPY_FFT_AVAILABLE:
        fn = _fi._scipy_fft.ifft if inverse else _fi._scipy_fft.fft
        return fn(a, axis=axis, workers=_fi.SCIPY_FFT_WORKERS)
    fn = np.fft.ifft if inverse else np.fft.fft
    return fn(a, axis=axis)


def _bluestein_axis_1d(A, alpha, N_out, sign, target_cdtype, axis):
    """One chirp-Z pass along ``axis``: ``N_in -> N_out`` at rate ``alpha``.

    ``sum_n A[..., n] * exp(sign*2*pi*j*alpha*n*k)`` for ``k`` in
    ``[0, N_out)``, by the same Bluestein identity :func:`_bluestein_2d`
    uses -- pre-chirp, circular convolution with the folded kernel on
    ``L = next_fast_len(N_in + N_out - 1)``, post-chirp.  Chirp signals are
    built in float64 and cast to ``target_cdtype`` before multiplication,
    exactly as the 2-D primitive does (this is what keeps the float32 path's
    chirp phase from losing precision at large indices).

    The largest array here is ``(rest x L)`` rather than ``(L x L)``, which
    is the whole point: see :func:`_bluestein_2d`'s ``separable`` parameter.
    """
    A = np.moveaxis(A, axis, -1)
    n_in = int(A.shape[-1])
    N_out = int(N_out)
    alpha = float(alpha)

    n = np.arange(n_in, dtype=np.float64)
    k = np.arange(N_out, dtype=np.float64)
    pre = np.exp(1j * sign * np.pi * alpha * n * n).astype(target_cdtype,
                                                           copy=False)
    post = np.exp(1j * sign * np.pi * alpha * k * k).astype(target_cdtype,
                                                            copy=False)
    L = int(next_fast_len(int(n_in + N_out - 1)))
    m_idx = np.arange(L, dtype=np.int64)
    m_signed = np.where(m_idx < N_out, m_idx, m_idx - L).astype(np.float64)
    h = np.exp(-1j * sign * np.pi * alpha * m_signed
               * m_signed).astype(target_cdtype, copy=False)

    g = np.zeros(A.shape[:-1] + (L,), dtype=target_cdtype)
    g[..., :n_in] = A * pre
    G = _fft_1d(g, axis=-1)
    del g
    # in-place: the kernel product is where a second (rest x L) array would
    # otherwise appear, and (rest x L) is the whole point of this route.
    G *= _fft_1d(h)
    out = _fft_1d(G, axis=-1, inverse=True)[..., :N_out]
    del G
    out = out * post
    if out.dtype != target_cdtype:
        out = out.astype(target_cdtype)
    return np.moveaxis(out, -1, axis)


def _bluestein_2d_separable(E, alpha_x, alpha_y, N_out_y, N_out_x, *,
                            sign, target_cdtype):
    """:func:`_bluestein_2d` as two 1-D chirp-Z passes.  NumPy only."""
    F = _bluestein_axis_1d(E, alpha_x, N_out_x, sign, target_cdtype, -1)
    F = _bluestein_axis_1d(F, alpha_y, N_out_y, sign, target_cdtype, -2)
    return F


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
    separable: bool = False,
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
    separable : bool, default False
        Evaluate the SAME sum as two 1-D chirp-Z passes (v5.33.2, audit
        ``AUDIT_TRACED_MEMORY_2026_08_09`` row 6) instead of one 2-D
        convolution.  NumPy backends only -- with any other ``xp`` this is
        ignored and the 2-D path runs, because the 1-D transforms would have
        to be dispatched through a caller-supplied ``fft2`` that only does
        two axes at once.

        WHY IT EXISTS.  The 2-D path pads BOTH axes to
        ``L = next_fast_len(N_in + N_out - 1)``, so every working array is
        ``L^2`` -- and the transform is EXACTLY separable, which the code
        already knows (it builds the kernel as ``h_y[:, None] * h_x[None, :]``).
        Two 1-D passes give the same sum with a largest array of
        ``(N_in x L)``, and they also drop the ``L^2`` chirp-kernel cache
        entry: two length-``L`` vectors (0.15 MB) replace a 1.359 GB array at
        design 121's shipped readout shape.  MEASURED on a tapered beam:

        ======================  ==========  ==========  =========  =========
        N_in / N_out (L)        2-D peak    sep. peak   rel L2     time
        ======================  ==========  ==========  =========  =========
        2048 / 256   (L=2304)   0.854 GB    0.255 GB    8.6e-16    0.15x
        4096 / 1024  (L=5120)   3.412 GB    1.343 GB    9.1e-16    0.42x
        ======================  ==========  ==========  =========  =========

        ACCURACY: **not byte-identical** -- it is a different association
        order for the same sum, so the difference is round-off:
        ``rel L2 <= 9.1e-16``, ``max|delta|/max|F| 1.2e-15``, power ratio
        1.000000000000.  Callers that pin bits must leave this False; that is
        why it is opt-in here and why the one shipped consumer
        (:func:`~lumenairy.propagators.carrier.carrier_referenced_exact_focus_readout`)
        carries its own default-ON switch with the 2-D path one flag away.

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

    # ----- 0) separable route (v5.33.2) --------------------------------------
    # Same sum, two 1-D passes, ``(N_in x L)`` instead of ``L^2``.  Taken
    # AFTER the precision guard above so both routes warn identically, and
    # only on NumPy (see the ``separable`` docstring entry).
    if separable and xp is np:
        return _bluestein_2d_separable(
            E, alpha_x, alpha_y, N_out_y, N_out_x,
            sign=sign, target_cdtype=target_cdtype)

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
            _h_fft_cache_store(cache_key, H_FFT)
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
    separable: bool = False,
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
    xp, fft2, ifft2, target_cdtype, separable
        Same as :func:`_bluestein_2d`.  The centring corrections are
        themselves separable (a pre-chirp in ``n``, a post-chirp in ``k`` and
        a constant), so ``separable`` changes only the core primitive.

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
        target_cdtype=target_cdtype, separable=separable,
    )
    F = F_core * (post_y[:, None] * post_x[None, :]) * const_c
    return F


__all__ = ['_bluestein_2d', '_bluestein_centred_2d']

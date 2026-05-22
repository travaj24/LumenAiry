"""
lumenairy.optimize._merit_jit -- Numba JIT helpers for wrapper-merit
per-field tilt-phase construction.

Private module.  Hosts the fused parallel kernel used by
:class:`lumenairy.optimize.wrapper_merits.MultiFieldMerit` to build
the masked tilted plane wave on each per-field leg.  v5.3 (ROADMAP
v5.3 horizon -- MultiFieldMerit JIT): the pre-v5.3 NumPy path
materialises three N x N temporaries per field

    tilt_phase    = sin(tx) * k_X + sin(ty) * k_Y   # 1
    phasor        = exp(1j * tilt_phase)            # 2
    E_tilted      = where(mask, phasor, 0)          # 3

which dominates the per-field cost above N >= 128.  This kernel
fuses all three into a single parallel pass with zero temporaries:

    for each pixel (i, j) in parallel:
        if mask[i, j]:
            ph = sin_tx * k_X[i, j] + sin_ty * k_Y[i, j]
            E_tilted[i, j] = cos(ph) + 1j * sin(ph)
        else:
            E_tilted[i, j] = 0 + 0j

The kernel signature mirrors the cached-payload shape produced by
:func:`lumenairy.optimize.wrapper_merits._get_wrapper_merit_cache`:
``k_X`` and ``k_Y`` are the wavelength-scaled tilt factors, ``mask``
is the boolean aperture mask (or ``None`` -- handled at the
caller's branch by ``aperture_mask is None`` BEFORE this kernel
fires).  Output dtypes (complex64 or complex128) are dispatched via
two specialised kernels because Numba ``@njit`` does not erase
dtypes.

The NumPy fallback (when Numba is unavailable) is exactly the
pre-v5.3 path -- the call site picks which to invoke based on the
module-level ``_NUMBA_AVAILABLE`` flag.

Author: Andrew Traverso -- v5.3 (ROADMAP v5.3 horizon -- MultiFieldMerit JIT).
"""

from __future__ import annotations

import numpy as np

# v5.3 (ROADMAP v5.3 horizon -- MultiFieldMerit JIT): canonical Numba
# probe.  Mirrors the guarded imports in elements/lenses.py and
# elements/_lens_traced.py.  Two specialised @njit kernels are built
# below (one per output dtype) because Numba does not erase the
# complex-precision type of the output array.
try:
    import numba as _numba  # noqa: F401
    from numba import njit as _njit
    from numba import prange as _prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _numba = None
    _NUMBA_AVAILABLE = False


# Minimum grid pixel count above which the JIT kernel is faster than
# the NumPy path.  Below this threshold the kernel-dispatch overhead
# (~20 us) dominates the savings.  At N=128 (16384 pixels) the JIT
# is ~1.5x; at N=512 (262144 pixels) the JIT is >3x.  Tuned on the
# v5.3 audit machine; users can override at runtime via
# ``lumenairy.optimize._merit_jit._MULTI_FIELD_JIT_MIN_PIXELS = N**2``
# if the empirical crossover sits elsewhere.
_MULTI_FIELD_JIT_MIN_PIXELS = 1 << 14  # 16384 pixels (N=128 square)


if _NUMBA_AVAILABLE:
    @_njit(cache=True, parallel=True, fastmath=True)
    def _multi_field_tilt_phasor_masked_c128(
        sin_tx, sin_ty, k_X, k_Y, mask, out,
    ):
        """In-place fused tilt-phase + masked exponential -> complex128.

        Builds ``out[i,j] = where(mask[i,j], exp(1j * (sin_tx*k_X[i,j] +
        sin_ty*k_Y[i,j])), 0+0j)`` in a single parallel pass with no
        temporary arrays.  ``out`` must be complex128 contiguous and
        the same shape as ``mask``, ``k_X``, ``k_Y``.

        Parameters
        ----------
        sin_tx, sin_ty : float64
            Pre-computed ``np.sin(theta_x)``, ``np.sin(theta_y)``.
        k_X, k_Y : (Ny, Nx) float64
            Wavelength-scaled tilt factors (cache ``X_factor``/wavelength
            and ``Y_factor``/wavelength).
        mask : (Ny, Nx) bool
            Aperture mask.  ``True`` pixels carry the phasor; ``False``
            pixels are zeroed.
        out : (Ny, Nx) complex128
            Output array (written in place).
        """
        Ny = mask.shape[0]
        Nx = mask.shape[1]
        for i in _prange(Ny):
            for j in range(Nx):
                if mask[i, j]:
                    ph = sin_tx * k_X[i, j] + sin_ty * k_Y[i, j]
                    out[i, j] = complex(np.cos(ph), np.sin(ph))
                else:
                    out[i, j] = 0.0 + 0.0j

    @_njit(cache=True, parallel=True, fastmath=True)
    def _multi_field_tilt_phasor_masked_c64(
        sin_tx, sin_ty, k_X, k_Y, mask, out,
    ):
        """In-place fused tilt-phase + masked exponential -> complex64.

        Companion to :func:`_multi_field_tilt_phasor_masked_c128`; same
        contract but writes ``np.complex64`` (used when the runtime
        precision knob is set to ``'single'`` via
        :func:`lumenairy.propagators.propagation.get_default_complex_dtype`).
        ``out`` must be complex64 contiguous and the same shape as the
        other arrays.
        """
        Ny = mask.shape[0]
        Nx = mask.shape[1]
        for i in _prange(Ny):
            for j in range(Nx):
                if mask[i, j]:
                    ph = sin_tx * k_X[i, j] + sin_ty * k_Y[i, j]
                    # Cast to complex64 via real+imag separately so the
                    # write to ``out`` does not silently upcast.
                    out[i, j] = np.complex64(np.cos(ph) + 1j * np.sin(ph))
                else:
                    out[i, j] = np.complex64(0.0 + 0.0j)
else:
    _multi_field_tilt_phasor_masked_c128 = None  # noqa
    _multi_field_tilt_phasor_masked_c64 = None  # noqa


def _multi_field_tilt_phasor_masked(
    sin_tx: float, sin_ty: float,
    k_X: np.ndarray, k_Y: np.ndarray,
    mask: np.ndarray, dtype: np.dtype,
) -> np.ndarray:
    """Fused tilted-plane-wave + aperture-mask constructor.

    v5.3 (ROADMAP v5.3 horizon -- MultiFieldMerit JIT): single-call
    helper that returns ``np.where(mask, np.exp(1j * (sin_tx*k_X +
    sin_ty*k_Y)), 0).astype(dtype)`` using the Numba JIT fast path
    when ``_NUMBA_AVAILABLE`` is ``True`` AND the grid is large
    enough to amortise the JIT-call overhead, falling back to a
    pure-NumPy implementation otherwise.

    Numerical contract: the JIT and NumPy paths produce identical
    output to within 1e-12 on every entry.  The only floating-point
    difference is the ``sin_tx*k_X + sin_ty*k_Y`` term, which Numba's
    ``fastmath=True`` may evaluate via a fused multiply-add; the
    resulting ULP-level difference vanishes after the periodic
    ``cos/sin`` -- the two paths agree bit-for-bit on a typical
    f/2 grid.  See ``tests/unit/test_v5_3_multi_field_merit_jit.py``.

    Parameters
    ----------
    sin_tx, sin_ty : float
        Pre-computed ``np.sin(theta_x)``, ``np.sin(theta_y)``.
    k_X, k_Y : ndarray
        Wavelength-scaled tilt factors.  Must be C-contiguous
        float64 and the same shape as ``mask``.
    mask : ndarray of bool
        Aperture mask.  Must be C-contiguous bool and the same shape
        as ``k_X``/``k_Y``.
    dtype : numpy dtype
        Output complex dtype (``np.complex64`` or ``np.complex128``).

    Returns
    -------
    E_tilted : ndarray of dtype
        Masked tilted plane wave.

    Notes
    -----
    Caller must pre-filter the ``aperture_mask is None`` and
    ``aperture_mask is _ZERO_APERTURE_MASK`` cases -- this helper
    only handles the boolean-mask branch.  The two sentinel branches
    are unconditionally NumPy-cheap (single allocation) and don't
    benefit from JIT.
    """
    # Threshold + dtype dispatch.  Below the threshold the
    # JIT-call overhead dominates the savings, so fall through to
    # NumPy.  When Numba is unavailable, the threshold check is
    # vacuously satisfied (kernel is ``None``) and we go straight
    # to the NumPy path.
    n_pixels = mask.size
    if (_NUMBA_AVAILABLE
            and n_pixels >= _MULTI_FIELD_JIT_MIN_PIXELS
            and _multi_field_tilt_phasor_masked_c128 is not None):
        # Specialise on dtype: complex64 vs complex128.  Numba's
        # @njit signatures fix the output dtype at compile time, so
        # we maintain two kernels rather than relying on a generic
        # template.  Default is complex128.
        np_dtype = np.dtype(dtype)
        if np_dtype == np.complex64:
            out = np.empty(mask.shape, dtype=np.complex64)
            _multi_field_tilt_phasor_masked_c64(
                float(sin_tx), float(sin_ty),
                np.ascontiguousarray(k_X, dtype=np.float64),
                np.ascontiguousarray(k_Y, dtype=np.float64),
                np.ascontiguousarray(mask, dtype=np.bool_),
                out,
            )
            return out
        # complex128 path (default; covers np.complex128 and any
        # promoted alias).
        out = np.empty(mask.shape, dtype=np.complex128)
        _multi_field_tilt_phasor_masked_c128(
            float(sin_tx), float(sin_ty),
            np.ascontiguousarray(k_X, dtype=np.float64),
            np.ascontiguousarray(k_Y, dtype=np.float64),
            np.ascontiguousarray(mask, dtype=np.bool_),
            out,
        )
        # If the caller actually wanted a non-complex128 dtype that
        # we did NOT specialise (e.g. some future complex256), cast
        # at the return.  The common path (complex128) is already
        # the kernel's native output and is a no-op cast.
        if np_dtype != np.complex128:
            return out.astype(np_dtype)
        return out

    # Pure-NumPy fallback.  Exactly the pre-v5.3 path that this
    # module supersedes (kept for parity + correctness pinning).
    tilt_phase = sin_tx * k_X + sin_ty * k_Y
    return np.where(mask, np.exp(1j * tilt_phase), 0.0).astype(dtype)

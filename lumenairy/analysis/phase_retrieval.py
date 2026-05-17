"""
Phase Retrieval Algorithms
===========================

Iterative methods for recovering the phase of a complex field from
intensity-only measurements.

Algorithms implemented:
    - Gerchberg-Saxton (GS)
    - Hybrid Input-Output (HIO)
    - Error Reduction (ER)

These methods alternate between real-space and Fourier-space constraints:
- Fourier constraint: enforce the measured far-field amplitude
- Real-space constraint: enforce a known support (or input amplitude)

The output is an estimated phase that, when combined with the constraints,
produces intensities matching both measurements.

References
----------
[1] Gerchberg, R.W. and Saxton, W.O. (1972). "A practical algorithm for the
    determination of phase from image and diffraction plane pictures."
    Optik 35(2): 237-246.
[2] Fienup, J.R. (1982). "Phase retrieval algorithms: a comparison."
    Applied Optics 21(15): 2758-2769.

Author: Andrew Traverso
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ..propagators.propagation import _fft2, _ifft2


__all__ = [
    # NumPy reference implementations
    'gerchberg_saxton',
    'error_reduction',
    'hybrid_input_output',
    # JAX-accelerated variants
    'gerchberg_saxton_jax',
    'error_reduction_jax',
    'hybrid_input_output_jax',
    # Cache utilities (v4.12.2)
    'clear_phase_retrieval_caches',
]


# =============================================================================
# GERCHBERG-SAXTON (GS)
# =============================================================================

def gerchberg_saxton(
    source_amplitude: np.ndarray,
    target_amplitude: np.ndarray,
    n_iter: int = 200,
    initial_phase: Optional[np.ndarray] = None,
    return_history: bool = False,
    seed: Optional[int] = None,
    dtype: Optional[Any] = None,
    *,
    backend: str = 'numpy',
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, List[float]]]:
    """
    Gerchberg-Saxton phase retrieval between source and target amplitudes.

    Finds a phase distribution that transforms the source amplitude into
    the target amplitude under Fourier transformation. Commonly used for
    designing phase-only computer-generated holograms (CGHs) and
    diffractive optical elements.

    Parameters
    ----------
    source_amplitude : ndarray (real, N×N)
        Known amplitude in the source (input/near-field) plane.
        Must be non-negative. Typically the illumination beam shape.
    target_amplitude : ndarray (real, N×N)
        Desired amplitude in the target (output/far-field) plane.
        Must be non-negative and the same shape as source_amplitude.
    n_iter : int, default 200
        Number of GS iterations. Typically 100-1000.
    initial_phase : ndarray (real, N×N) or None
        Initial guess for the source-plane phase. If None, uses random
        phase in [-pi, pi].
    return_history : bool, default False
        If True, also returns per-iteration error metrics.

    Returns
    -------
    source_phase : ndarray (real, N×N)
        Recovered phase at the source plane. To use: place the field
        ``source_amplitude * exp(1j * source_phase)`` at the input and
        Fourier-transform to get approximately ``target_amplitude``.
    error : float
        Final mean-squared error between the target and the achieved
        far-field amplitude (smaller is better).
    history : list of float, optional
        Per-iteration error values (if return_history=True).

    Notes
    -----
    The GS algorithm is guaranteed not to increase the error at each
    iteration, but it can stall in local minima. For better results:
    - Use a good initial phase (e.g. a geometric/paraxial estimate)
    - Run many iterations (500-1000)
    - For stubborn cases, use :func:`hybrid_input_output` which can
      escape local minima.

    Examples
    --------
    >>> # Design a phase-only hologram for an LG_01 beam from a flat input
    >>> import numpy as np
    >>> N = 256
    >>> x = np.linspace(-1, 1, N)
    >>> X, Y = np.meshgrid(x, x)
    >>> source = np.exp(-(X**2 + Y**2) / 0.5**2)  # Gaussian input
    >>> target = np.sqrt(X**2 + Y**2) * np.exp(-(X**2 + Y**2) / 0.3**2)  # donut
    >>> phase, err = gerchberg_saxton(source, target, n_iter=300)

    Notes
    -----
    Pass ``backend='jax'`` to dispatch to the JAX-traced
    implementation (:func:`gerchberg_saxton_jax`).
    """
    if backend == 'jax':
        # 4.12.0 (audit round-4 B2-6): forward all reproducibility /
        # precision kwargs to the JAX path.  Pre-4.12 the dispatcher
        # only passed ``n_iter``, silently dropping ``seed``,
        # ``initial_phase``, and ``dtype``.  Function-level kwargs on
        # gerchberg_saxton_jax were wired correctly internally; the
        # unified front door just didn't forward them.
        return gerchberg_saxton_jax(
            source_amplitude, target_amplitude, n_iter=n_iter,
            seed=seed, initial_phase=initial_phase, dtype=dtype)
    if backend != 'numpy':
        raise ValueError(
            f"gerchberg_saxton: backend must be 'numpy' or 'jax'; "
            f"got {backend!r}.")
    if source_amplitude.shape != target_amplitude.shape:
        raise ValueError("Source and target must have the same shape")

    N = source_amplitude.shape[0]

    # Initial guess (4.10: seeded for reproducibility; pass seed=None
    # for the prior randomised-per-call behaviour).
    if initial_phase is None:
        rng = (np.random.default_rng() if seed is None
               else np.random.default_rng(int(seed)))
        phase = rng.uniform(-np.pi, np.pi, size=source_amplitude.shape)
    else:
        phase = initial_phase.copy()

    # Normalize target so both fields have the same total power
    source_power = np.sum(source_amplitude**2)
    target_power = np.sum(target_amplitude**2)
    if target_power > 0:
        target_scaled = target_amplitude * np.sqrt(source_power / target_power)
    else:
        target_scaled = target_amplitude

    history = []
    field = source_amplitude * np.exp(1j * phase)

    for _ in range(n_iter):
        # Forward FFT: source -> target plane
        far_field = np.fft.fftshift(_fft2(np.fft.ifftshift(field)))

        # Error metric (before applying target constraint)
        if return_history:
            achieved = np.abs(far_field)
            err = np.mean((achieved - target_scaled)**2)
            history.append(err)

        # Replace far-field amplitude with target, keep phase
        far_phase = np.angle(far_field)
        far_field = target_scaled * np.exp(1j * far_phase)

        # Inverse FFT: target -> source plane
        field = np.fft.fftshift(_ifft2(np.fft.ifftshift(far_field)))

        # Replace source amplitude, keep phase
        source_phase_new = np.angle(field)
        field = source_amplitude * np.exp(1j * source_phase_new)

    source_phase = np.angle(field)

    # Final error
    far_field = np.fft.fftshift(_fft2(np.fft.ifftshift(field)))
    final_err = float(np.mean((np.abs(far_field) - target_scaled)**2))

    if return_history:
        return source_phase, final_err, history
    return source_phase, final_err


# =============================================================================
# ERROR REDUCTION (ER)
# =============================================================================

def error_reduction(
    measured_amplitude: np.ndarray,
    support: np.ndarray,
    n_iter: int = 200,
    initial_guess: Optional[np.ndarray] = None,
    return_history: bool = False,
    seed: Optional[int] = None,
    dtype: Optional[Any] = None,
    *,
    backend: str = 'numpy',
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, List[float]]]:
    """
    Error-reduction phase retrieval from a single far-field intensity.

    Used for coherent diffractive imaging: given only the far-field
    intensity |FT(E)|^2 and a support constraint (the region where
    the object is known to be non-zero), recover the complex object.

    Parameters
    ----------
    measured_amplitude : ndarray (real, N×N)
        Measured far-field amplitude (sqrt of the intensity).
    support : ndarray (bool, N×N)
        Support mask — True where the object can be non-zero,
        False outside.
    n_iter : int, default 200
        Number of iterations.
    initial_guess : ndarray (complex, N×N) or None
        Initial object estimate. If None, uses random phase with
        unit amplitude inside the support.
    return_history : bool, default False
        If True, also returns per-iteration error metrics.

    Returns
    -------
    object_field : ndarray (complex, N×N)
        Recovered complex object field.
    error : float
        Final Fourier-magnitude error.
    history : list of float, optional
        Per-iteration errors.

    Notes
    -----
    Error Reduction is the simplest Fienup-type algorithm. It converges
    monotonically but can stagnate. Use :func:`hybrid_input_output` for
    better escape from local minima.

    Pass ``backend='jax'`` to dispatch to the JAX-traced implementation
    (:func:`error_reduction_jax`).

    Parameters (4.11.2 additions)
    -----------------------------
    seed : int, optional
        Seeds the random initial-phase draw when ``initial_guess`` is
        None.  Provides API parity with :func:`error_reduction_jax`.
        Default ``None`` -> non-reproducible random initial phase.
    dtype : complex dtype, optional
        Complex dtype for the working state.  Default
        ``np.complex128``.  Pass ``np.complex64`` for single-precision
        parity with the JAX path.
    """
    if backend == 'jax':
        return error_reduction_jax(
            measured_amplitude, support, n_iter=n_iter,
            seed=seed, dtype=dtype)
    if backend != 'numpy':
        raise ValueError(
            f"error_reduction: backend must be 'numpy' or 'jax'; "
            f"got {backend!r}.")
    N = measured_amplitude.shape[0]

    # 4.11.2: honour `seed` for reproducibility (pre-4.11.2 the kwarg
    # didn't exist on this path; users wanting deterministic runs had
    # to construct `initial_guess` manually).
    cdtype = dtype if dtype is not None else np.complex128
    if initial_guess is None:
        rng = (np.random.default_rng() if seed is None
               else np.random.default_rng(int(seed)))
        phase = rng.uniform(-np.pi, np.pi, size=measured_amplitude.shape)
        obj = np.where(support, np.exp(1j * phase), 0.0 + 0.0j).astype(cdtype)
    else:
        obj = np.asarray(initial_guess, dtype=cdtype).copy()

    history = []

    for _ in range(n_iter):
        # Forward: object -> Fourier
        F = np.fft.fftshift(_fft2(np.fft.ifftshift(obj)))

        if return_history:
            err = float(np.mean((np.abs(F) - measured_amplitude)**2))
            history.append(err)

        # Fourier-magnitude constraint
        F = measured_amplitude * np.exp(1j * np.angle(F))

        # Inverse: Fourier -> object
        obj_new = np.fft.fftshift(_ifft2(np.fft.ifftshift(F)))

        # Real-space constraint: zero outside support
        obj = np.where(support, obj_new, 0.0 + 0.0j)

    F = np.fft.fftshift(_fft2(np.fft.ifftshift(obj)))
    final_err = float(np.mean((np.abs(F) - measured_amplitude)**2))

    # 4.11.2: honour the requested dtype on the returned array.
    obj = np.asarray(obj, dtype=cdtype)

    if return_history:
        return obj, final_err, history
    return obj, final_err


# =============================================================================
# HYBRID INPUT-OUTPUT (HIO)
# =============================================================================

def hybrid_input_output(
    measured_amplitude: np.ndarray,
    support: np.ndarray,
    n_iter: int = 200,
    beta: float = 0.9,
    initial_guess: Optional[np.ndarray] = None,
    return_history: bool = False,
    seed: Optional[int] = None,
    dtype: Optional[Any] = None,
    *,
    backend: str = 'numpy',
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, List[float]]]:
    """
    Fienup's Hybrid Input-Output (HIO) algorithm for phase retrieval.

    HIO escapes local minima of Error Reduction by allowing temporary
    violations of the real-space constraint, controlled by the feedback
    parameter beta.

    Parameters
    ----------
    measured_amplitude : ndarray (real, N×N)
        Measured far-field amplitude.
    support : ndarray (bool, N×N)
        Support mask.
    n_iter : int, default 200
        Number of iterations.
    beta : float, default 0.9
        Feedback parameter, typically 0.5-1.0. Larger = more aggressive
        escape from local minima but less stable.
    initial_guess : ndarray (complex, N×N) or None
        Initial object estimate.
    return_history : bool, default False
        If True, also returns per-iteration errors.

    Returns
    -------
    object_field : ndarray (complex, N×N)
        Recovered object field.
    error : float
        Final Fourier-magnitude error.
    history : list of float, optional

    Notes
    -----
    The HIO update rule is::

        new[i] = g[i]                    if inside support
               = old[i] - beta * g[i]    if outside support

    where g is the result after applying the Fourier constraint.

    For best results, run HIO for most of the iterations then switch to
    Error Reduction for the final cleanup. This is the standard "HIO+ER"
    hybrid strategy used in coherent diffraction imaging.

    References
    ----------
    Fienup, J.R. (1982). "Phase retrieval algorithms: a comparison."
    Applied Optics 21(15): 2758-2769.

    Pass ``backend='jax'`` to dispatch to the JAX-traced implementation
    (:func:`hybrid_input_output_jax`).

    Parameters (4.11.2 additions)
    -----------------------------
    seed : int, optional
        Seeds the random initial-phase draw when ``initial_guess`` is
        None.  Provides API parity with :func:`hybrid_input_output_jax`.
    dtype : complex dtype, optional
        Complex dtype for the working state.  Default
        ``np.complex128``.
    """
    if backend == 'jax':
        return hybrid_input_output_jax(
            measured_amplitude, support, n_iter=n_iter, beta=beta,
            seed=seed, dtype=dtype)
    if backend != 'numpy':
        raise ValueError(
            f"hybrid_input_output: backend must be 'numpy' or 'jax'; "
            f"got {backend!r}.")
    # 4.11.2: honour `seed` / `dtype` for reproducibility + precision
    # control (pre-4.11.2 the kwargs didn't exist on this path).
    cdtype = dtype if dtype is not None else np.complex128
    if initial_guess is None:
        rng = (np.random.default_rng() if seed is None
               else np.random.default_rng(int(seed)))
        phase = rng.uniform(-np.pi, np.pi, size=measured_amplitude.shape)
        obj = np.where(support, np.exp(1j * phase), 0.0 + 0.0j).astype(cdtype)
    else:
        obj = np.asarray(initial_guess, dtype=cdtype).copy()

    history = []

    for _ in range(n_iter):
        # Forward: object -> Fourier
        F = np.fft.fftshift(_fft2(np.fft.ifftshift(obj)))

        if return_history:
            err = float(np.mean((np.abs(F) - measured_amplitude)**2))
            history.append(err)

        # Fourier-magnitude constraint
        F = measured_amplitude * np.exp(1j * np.angle(F))

        # Inverse: Fourier -> object
        g = np.fft.fftshift(_ifft2(np.fft.ifftshift(F)))

        # HIO update: keep inside support, feedback correction outside
        obj = np.where(support, g, obj - beta * g)

    F = np.fft.fftshift(_fft2(np.fft.ifftshift(obj)))
    final_err = float(np.mean((np.abs(F) - measured_amplitude)**2))

    # 4.11.2: honour the requested dtype on the returned array even if
    # the FFT backend silently promoted to complex128 inside the loop.
    obj = np.asarray(obj, dtype=cdtype)

    if return_history:
        return obj, final_err, history
    return obj, final_err


# ----------------------------------------------------------------------
# JAX-jit'd phase-retrieval variants
# ----------------------------------------------------------------------
#
# v4.12 perf: each outer driver below builds its iteration kernel at
# module scope (parameterised via a small cache keyed on n_iter and any
# scalar Python knobs).  Pre-4.12 the iteration body lived inside a
# closure that was re-created on every call -- the ``lax.fori_loop``
# inside was jit-traced by JAX, but the outer wrapper paid a fresh
# dispatch each invocation.  With the module-scope cache, repeated
# calls with the same n_iter reuse the same compiled XLA executable.

#
# v4.12.2: converted to LRU-bounded ``OrderedDict``s (were unbounded
# plain dicts) so repeated calls with many distinct ``n_iter`` values
# don't leak compiled XLA executables.  Accessed keys are moved to the
# end; when ``len > _PR_KERNEL_CACHE_MAXSIZE`` the oldest entry is
# evicted.
_GS_KERNEL_CACHE: 'OrderedDict[int, Any]' = OrderedDict()
_ER_KERNEL_CACHE: 'OrderedDict[int, Any]' = OrderedDict()
_HIO_KERNEL_CACHE: 'OrderedDict[int, Any]' = OrderedDict()
_PR_KERNEL_CACHE_MAXSIZE = 32


def clear_phase_retrieval_caches() -> None:
    """Drop every cached jit'd phase-retrieval kernel (v4.12.2).

    Clears the GS / ER / HIO ``n_iter``-keyed kernel caches.  Forces
    the next ``gerchberg_saxton_jax`` / ``error_reduction_jax`` /
    ``hybrid_input_output_jax`` call to rebuild and re-cache its
    jit-compiled kernel from scratch.  Useful in unit tests that pin
    cache mechanics and in long-running pipelines that want to release
    the underlying XLA executables.
    """
    _GS_KERNEL_CACHE.clear()
    _ER_KERNEL_CACHE.clear()
    _HIO_KERNEL_CACHE.clear()


def _make_gs_kernel(n_iter_int: int):
    """Build (and cache) the jit'd GS iteration kernel for one n_iter."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def _run(E0_, src_, tgt_):
        def body(i, state):
            E_in, _ = state
            F = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(E_in)))
            E_target = tgt_ * jnp.exp(1j * jnp.angle(F))
            E_back = jnp.fft.fftshift(
                jnp.fft.ifft2(jnp.fft.ifftshift(E_target)))
            E_next = src_ * jnp.exp(1j * jnp.angle(E_back))
            return (E_next, F)

        E_final_, F_final_ = jax.lax.fori_loop(
            0, n_iter_int, body, (E0_, E0_))
        phase_ = jnp.angle(E_final_)
        err_ = jnp.mean((jnp.abs(F_final_) - tgt_) ** 2)
        return phase_, err_

    return _run


def _make_er_kernel(n_iter_int: int):
    """Build (and cache) the jit'd ER iteration kernel for one n_iter."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def _run(obj0_, meas_, sup_):
        def body(i, obj):
            F = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(obj)))
            F = meas_ * jnp.exp(1j * jnp.angle(F))
            obj_new = jnp.fft.fftshift(
                jnp.fft.ifft2(jnp.fft.ifftshift(F)))
            return jnp.where(sup_, obj_new, 0.0)

        obj_final_ = jax.lax.fori_loop(
            0, n_iter_int, body, obj0_)
        obj_final_ = jnp.where(sup_, obj_final_, 0.0)
        F = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(obj_final_)))
        err_ = jnp.mean((jnp.abs(F) - meas_) ** 2)
        return obj_final_, err_

    return _run


def _make_hio_kernel(n_iter_int: int):
    """Build (and cache) the jit'd HIO iteration kernel for one n_iter."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def _run(obj0_, meas_, sup_, beta_):
        def body(i, obj):
            F = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(obj)))
            F = meas_ * jnp.exp(1j * jnp.angle(F))
            g = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(F)))
            return jnp.where(sup_, g, obj - beta_ * g)

        obj_final_ = jax.lax.fori_loop(
            0, n_iter_int, body, obj0_)
        F = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(obj_final_)))
        err_ = jnp.mean((jnp.abs(F) - meas_) ** 2)
        return obj_final_, err_

    return _run

def gerchberg_saxton_jax(
    source_amplitude: np.ndarray,
    target_amplitude: np.ndarray,
    n_iter: int = 200,
    seed: Optional[int] = None,
    dtype: Optional[Any] = None,
    initial_phase: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """JAX-jit'd Gerchberg-Saxton iteration.

    Same physics as :func:`gerchberg_saxton`; the iteration loop runs
    inside ``jax.lax.fori_loop`` so the entire run is one fused JIT
    kernel.  Best speedup on GPU; CPU benefit is modest because the
    NumPy version already uses pyFFTW.

    Parameters
    ----------
    seed : int, optional
        4.11.2: now actually seeds the random initial-phase draw.
        Pre-4.11.2 the kwarg was accepted but ignored (``_ = seed``
        with no consumer), so two calls with different seeds produced
        the same trajectory.  Pass ``None`` (default) for a uniformly-
        zero initial phase (matches the historical deterministic
        behaviour); pass an int to draw an i.i.d. uniform initial
        phase that randomises the iteration start.
    dtype : numpy/jax float dtype, optional
        4.10: caller can request float64.  Default is float32 to match
        the historical JAX path.  Pass ``np.float64`` to bring the
        backend to NumPy parity (~1e-6 -> ~1e-14 residual error).
    initial_phase : ndarray, optional
        4.11.2: explicit initial-phase array.  Overrides ``seed``.
        Mirrors the NumPy variant's API.

    Returns
    -------
    phase : ndarray (real, same shape as inputs)
        Recovered source-plane phase.
    err : float
        Final intensity-domain RMS error.
    """
    from ..backend import JAX_AVAILABLE
    if not JAX_AVAILABLE:
        raise ImportError(
            'JAX is not installed; install with `pip install jax` or '
            'use gerchberg_saxton() (NumPy).')
    import jax
    import jax.numpy as jnp
    if dtype is None:
        dtype = jnp.float32

    src = jnp.asarray(source_amplitude, dtype=dtype)
    tgt = jnp.asarray(target_amplitude, dtype=dtype)

    # 4.11.2: actually consume `seed`.  Build the initial complex field
    # from src amplitude plus a (possibly random) initial phase.  The
    # JAX loop is deterministic given E0, so different seeds genuinely
    # produce different trajectories (verified by the pinning test in
    # tests/unit/test_audit_fixes_v4_11_2_analysis.py).
    if initial_phase is not None:
        phase0 = jnp.asarray(initial_phase, dtype=dtype)
    elif seed is None:
        # Historical behaviour: zero initial phase.
        phase0 = jnp.zeros_like(src)
    else:
        key = jax.random.PRNGKey(int(seed))
        phase0 = jax.random.uniform(
            key, shape=src.shape, minval=-jnp.pi, maxval=jnp.pi,
            dtype=dtype)

    # Combine source amplitude with the (possibly seeded) initial phase.
    E0 = (src * jnp.exp(1j * phase0)).astype(jnp.complex64)

    n_iter_int = int(n_iter)
    kernel = _GS_KERNEL_CACHE.get(n_iter_int)
    if kernel is None:
        kernel = _make_gs_kernel(n_iter_int)
        _GS_KERNEL_CACHE[n_iter_int] = kernel
        while len(_GS_KERNEL_CACHE) > _PR_KERNEL_CACHE_MAXSIZE:
            _GS_KERNEL_CACHE.popitem(last=False)
    else:
        _GS_KERNEL_CACHE.move_to_end(n_iter_int)
    phase, err = kernel(E0, src, tgt)
    return np.asarray(phase), float(err)


def _seed_to_rng(seed):
    """Resolve an int seed to a numpy default_rng (4.10 helper)."""
    if seed is None:
        return np.random.default_rng(0)
    return np.random.default_rng(int(seed))


def error_reduction_jax(
    measured_amplitude: np.ndarray,
    support: np.ndarray,
    n_iter: int = 200,
    init_phase: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, float]:
    """JAX-jit'd Error-Reduction phase retrieval (Fienup 1982).

    Mirror of :func:`error_reduction` -- alternating-projection on the
    Fourier-amplitude constraint and a real-space support constraint.

    4.10: ``seed`` controls the random initial phase (was hard-coded
    to seed=0 pre-4.10).  ``dtype`` selects float32 (default) or
    float64 (matches NumPy precision).
    """
    from ..backend import JAX_AVAILABLE
    if not JAX_AVAILABLE:
        raise ImportError('JAX is not installed.')
    import jax
    import jax.numpy as jnp

    if dtype is None:
        dtype = jnp.float32
    meas = jnp.asarray(measured_amplitude, dtype=dtype)
    sup = jnp.asarray(support, dtype=bool)

    if init_phase is None:
        rng = _seed_to_rng(seed)
        init_phase_np = rng.uniform(-np.pi, np.pi, meas.shape).astype(
            np.dtype(dtype))
        init_phase = jnp.asarray(init_phase_np)

    F0 = meas * jnp.exp(1j * init_phase)
    obj0 = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(F0)))

    n_iter_int = int(n_iter)
    kernel = _ER_KERNEL_CACHE.get(n_iter_int)
    if kernel is None:
        kernel = _make_er_kernel(n_iter_int)
        _ER_KERNEL_CACHE[n_iter_int] = kernel
        while len(_ER_KERNEL_CACHE) > _PR_KERNEL_CACHE_MAXSIZE:
            _ER_KERNEL_CACHE.popitem(last=False)
    else:
        _ER_KERNEL_CACHE.move_to_end(n_iter_int)
    obj_final, err = kernel(obj0, meas, sup)
    return np.asarray(obj_final), float(err)


def hybrid_input_output_jax(
    measured_amplitude: np.ndarray,
    support: np.ndarray,
    n_iter: int = 200,
    beta: float = 0.9,
    init_phase: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, float]:
    """JAX-jit'd Hybrid Input-Output phase retrieval.

    HIO swaps the support projection for an outside-support feedback
    update with parameter ``beta`` (typically 0.7-0.9).  Generally
    converges faster + less prone to local minima than ER.

    4.10: ``seed`` controls the random initial phase; ``dtype``
    selects float32 (default) or float64 for full NumPy parity.
    """
    from ..backend import JAX_AVAILABLE
    if not JAX_AVAILABLE:
        raise ImportError('JAX is not installed.')
    import jax
    import jax.numpy as jnp

    if dtype is None:
        dtype = jnp.float32
    meas = jnp.asarray(measured_amplitude, dtype=dtype)
    sup = jnp.asarray(support, dtype=bool)
    beta_j = jnp.asarray(float(beta), dtype=dtype)

    if init_phase is None:
        rng = _seed_to_rng(seed)
        init_phase_np = rng.uniform(-np.pi, np.pi, meas.shape).astype(
            np.dtype(dtype))
        init_phase = jnp.asarray(init_phase_np)

    F0 = meas * jnp.exp(1j * init_phase)
    obj0 = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(F0)))

    n_iter_int = int(n_iter)
    kernel = _HIO_KERNEL_CACHE.get(n_iter_int)
    if kernel is None:
        kernel = _make_hio_kernel(n_iter_int)
        _HIO_KERNEL_CACHE[n_iter_int] = kernel
        while len(_HIO_KERNEL_CACHE) > _PR_KERNEL_CACHE_MAXSIZE:
            _HIO_KERNEL_CACHE.popitem(last=False)
    else:
        _HIO_KERNEL_CACHE.move_to_end(n_iter_int)
    obj_final, err = kernel(obj0, meas, sup, beta_j)
    return np.asarray(obj_final), float(err)

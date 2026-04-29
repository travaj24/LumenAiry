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

import numpy as np
from .propagation import _fft2, _ifft2


# =============================================================================
# GERCHBERG-SAXTON (GS)
# =============================================================================

def gerchberg_saxton(source_amplitude, target_amplitude, n_iter=200,
                     initial_phase=None, return_history=False):
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
    """
    if source_amplitude.shape != target_amplitude.shape:
        raise ValueError("Source and target must have the same shape")

    N = source_amplitude.shape[0]

    # Initial guess
    if initial_phase is None:
        rng = np.random.default_rng()
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

def error_reduction(measured_amplitude, support, n_iter=200,
                    initial_guess=None, return_history=False):
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
    """
    N = measured_amplitude.shape[0]

    if initial_guess is None:
        rng = np.random.default_rng()
        phase = rng.uniform(-np.pi, np.pi, size=measured_amplitude.shape)
        obj = np.where(support, np.exp(1j * phase), 0.0 + 0.0j)
    else:
        obj = initial_guess.copy()

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

    if return_history:
        return obj, final_err, history
    return obj, final_err


# =============================================================================
# HYBRID INPUT-OUTPUT (HIO)
# =============================================================================

def hybrid_input_output(measured_amplitude, support, n_iter=200, beta=0.9,
                        initial_guess=None, return_history=False):
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
    """
    if initial_guess is None:
        rng = np.random.default_rng()
        phase = rng.uniform(-np.pi, np.pi, size=measured_amplitude.shape)
        obj = np.where(support, np.exp(1j * phase), 0.0 + 0.0j)
    else:
        obj = initial_guess.copy()

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

    if return_history:
        return obj, final_err, history
    return obj, final_err

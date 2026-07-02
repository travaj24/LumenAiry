"""
Strehl-ratio and coupling-efficiency metrics (scalar + vector).

This submodule was carved out of ``lumenairy.analysis.core`` in v5.1.0
as part of the mechanical 6-file split (see ``ROADMAP.md`` v5.1
"Architecture / housekeeping").  All functions, signatures, and numerics
are unchanged -- the historical public API is preserved by a thin
re-export shell in ``lumenairy.analysis.core``.

See Also
--------
lumenairy.analysis.beam_stats : beam centroid / D4sigma / M^2 / diameter.
lumenairy.analysis.psf_mtf_otf : PSF / MTF / OTF + spec-sheet metrics.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

__all__ = [
    'strehl_ratio',
    'strehl_marechal',
    'strehl_phase_integral',
    'coupling_efficiency',
    'strehl_vector',
    'coupling_efficiency_vector',
]


# v5.2 (ROADMAP "Duplicate `_xp_of`" cleanup): see beam_stats.py.
from ..backend import array_namespace as _xp_of  # noqa: E402


def strehl_ratio(
    E: np.ndarray,
    E_ref: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
) -> float:
    """
    Compute the Strehl ratio of a field relative to a reference field.

    Both fields are normalised to the same total power before comparison
    so that the ratio reflects wavefront quality rather than throughput.

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Aberrated field (e.g. at the focal plane).
    E_ref : ndarray, complex, shape (Ny, Nx)
        Reference (diffraction-limited) field at the same plane.
    dx : float
        Grid spacing along x [m].
    dy : float, optional
        Grid spacing along y [m].  Defaults to ``dx`` (square grid).
        v4.13.2 added this kwarg so anamorphic / non-square grids no
        longer mis-scale the per-pixel area in the total-power
        normalisation.  Backward compatible: callers that omit ``dy``
        get identical behaviour to v4.13.1.

    Returns
    -------
    strehl : float
        Strehl ratio (0 to 1).  A value of 1.0 indicates a
        diffraction-limited beam.

    Notes
    -----
    ``Strehl = max(|E|^2) / max(|E_ref|^2)`` after both fields have been
    normalised to equal total power.  The Strehl ratio is dimensionless
    and the ``dx * dy`` factor cancels in the ratio, but using the
    correct pixel area keeps any external comparison consistent.
    """
    # v4.15.5 (P1-NEW-2WAY-1): defensive guards via the shared
    # ``_check_2d_scalar_field`` helper on BOTH input fields.
    # Previously an MCF / 3-D ensemble input (either ``E`` or
    # ``E_ref``) failed downstream at ``xp.abs(...)`` with an
    # unhelpful Python TypeError.  Input kind: 'field' (both args
    # are 2-D scalar complex amplitudes).
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E, 'strehl_ratio')
    _check_2d_scalar_field(E_ref, 'strehl_ratio')
    xp = _xp_of(E, E_ref)
    I = xp.abs(E) ** 2
    I_ref = xp.abs(E_ref) ** 2

    # 4.13.2 (C-P1-1): use ``dx * dy`` for the pixel area when ``dy``
    # is explicitly provided.  Pre-4.13.2 the v4.13.0 L3 sweep missed
    # this site and any anamorphic / non-square grid produced a wrong
    # total-power normalisation.  When ``dy`` is omitted we keep the
    # historical ``dx ** 2`` form bit-for-bit so callers that did not
    # pass a ``dy`` see exactly identical numerics to v4.13.1 (the
    # Strehl ratio is dimensionless and ``dx ** 2 == dx * dx`` is
    # numerically equal but uses a different IEEE rounding pathway
    # than ``dx * dy``; preserving the form keeps a small floating-
    # point identity that downstream tests rely on).
    if dy is None:
        pixel_area = dx ** 2
    else:
        pixel_area = dx * dy
    P = float(xp.sum(I) * pixel_area)
    P_ref = float(xp.sum(I_ref) * pixel_area)

    if P_ref == 0 or P == 0:
        return 0.0

    # Normalize to same total power
    return float(xp.max(I)) / P * P_ref / float(xp.max(I_ref))


def strehl_marechal(rms_waves: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Marechal-approximation Strehl ratio from wavefront RMS.

    .. math::
        S \\approx \\exp\\bigl(-(2\\pi\\sigma)^2\\bigr)

    where :math:`\\sigma` is the wavefront RMS error in waves.  Valid for
    :math:`\\sigma \\ll 1` (typically :math:`\\sigma < 0.1` waves).  For
    larger aberrations use :func:`strehl_phase_integral` or
    :func:`strehl_ratio`.

    Useful when you have an RMS-WFE estimate but no full PSF, or when
    comparing predictions against analytic small-aberration theory.

    Parameters
    ----------
    rms_waves : float or array
        RMS wavefront error in waves (NOT radians).

    Returns
    -------
    strehl : float or ndarray
        Marechal-approximation Strehl in [0, 1].

    See Also
    --------
    strehl_ratio : peak-ratio Strehl from full PSF.
    strehl_phase_integral : exact small-aberration Strehl from a pupil.

    Examples
    --------
    >>> import lumenairy as la
    >>> # Diffraction-limited rule of thumb: sigma ~ 1/14 wave -> S ~ 0.82
    >>> float(la.strehl_marechal(1.0 / 14.0))
    0.8175...
    """
    sigma = 2.0 * np.pi * np.asarray(rms_waves, dtype=float)
    return np.exp(-(sigma ** 2))


def strehl_phase_integral(pupil: np.ndarray) -> float:
    """Strehl ratio from the pupil-phase integral (Born & Wolf 9.1.10).

    .. math::
        S = \\left| \\frac{\\int A(x, y) \\, e^{i\\phi(x, y)} \\, dA}{\\int A(x, y) \\, dA} \\right|^2

    where :math:`A = |\\mathrm{pupil}|` is the pupil amplitude and
    :math:`\\phi = \\arg(\\mathrm{pupil})` is the pupil phase.  This is
    the exact small-aberration Strehl formula and avoids the
    peak-finding bias of :func:`strehl_ratio` on asymmetric PSFs where
    the diffraction-limited peak does not sit on the geometric chief
    ray.

    Parameters
    ----------
    pupil : ndarray (complex, 2-D)
        Complex pupil function.  Amplitude defines the aperture; phase
        carries the wavefront aberration.  Outside the aperture the
        amplitude should be zero so it does not contribute to the
        integral.

    Returns
    -------
    strehl : float
        Strehl ratio in [0, 1].  Returns 0.0 if the pupil has zero
        net amplitude (degenerate aperture).

    See Also
    --------
    strehl_ratio : peak-ratio Strehl from a full diffraction PSF.
    strehl_marechal : closed-form ``exp(-(2 pi sigma)^2)`` approximation
        from an RMS estimate.

    Examples
    --------
    >>> import numpy as np, lumenairy as la
    >>> N = 128
    >>> x = (np.arange(N) - N/2) / (N/2)
    >>> X, Y = np.meshgrid(x, x)
    >>> aperture = (X**2 + Y**2) <= 1.0
    >>> # Flat-phase pupil -> S = 1
    >>> P = aperture.astype(complex)
    >>> float(la.strehl_phase_integral(P))
    1.0
    """
    A = np.abs(pupil)
    A_sum = float(A.sum())
    if A_sum == 0:
        return 0.0
    num = float(np.abs(np.sum(pupil)) ** 2)
    den = A_sum ** 2
    return num / den


def coupling_efficiency(
    E: np.ndarray,
    mode: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
) -> float:
    r"""Compute the mode-overlap coupling efficiency between a field
    and a target mode.

    Returns ``\eta = |<E | mode>|^2 / (<E|E> * <mode|mode>)``, the
    standard receiver / fiber-coupling efficiency expression.  Both
    fields must be sampled on the SAME grid (same shape, dx, dy);
    centroids may differ if the mode is intentionally offset.

    Parameters
    ----------
    E : ndarray, complex
        Incoming field at the coupling plane (e.g. focal plane after
        the receive lens).
    mode : ndarray, complex
        Target mode (e.g. a fiber LP01 mode generated by
        :func:`create_fiber_mode`, or any other reference complex
        amplitude pattern).
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    eta : float
        Coupling efficiency in [0, 1].  1.0 means ``E`` is a unit
        complex multiple of ``mode``; 0.0 means orthogonal.

    Notes
    -----
    For amplitude-only matching with E and mode both real-positive,
    this reduces to the classical overlap integral.  For complex
    fields the phase structure must also match for full coupling --
    a perfectly-shaped beam with the wrong phase ramp couples to
    zero efficiency.

    The function is :class:`numpy.float`-conservative: if the mode
    or field is identically zero, returns 0.0.
    """
    # v4.15.5 (P1-NEW-2WAY-1): defensive guards via the shared
    # ``_check_2d_scalar_field`` helper on both fields.  Previously
    # an MCF / 3-D ensemble input failed at the ``.shape`` attribute
    # access (for MCF) or produced a wrong (3-D) overlap (for an
    # ensemble).  The V6 walker discovers this entry via the first-
    # positional-name ``E``; the inline guard routes both failure
    # modes to the canonical v4.16 message.  Input kind: 'field'
    # (both args are 2-D scalar complex amplitudes).
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E, 'coupling_efficiency')
    _check_2d_scalar_field(mode, 'coupling_efficiency')
    if E.shape != mode.shape:
        raise ValueError(
            f"coupling_efficiency: shape mismatch -- E is {E.shape}, "
            f"mode is {mode.shape}.  Resample to a common grid first.")
    if dy is None:
        dy = dx
    da = float(dx) * float(dy)
    overlap = np.sum(np.conj(mode) * E) * da
    p_E = np.sum(np.abs(E) ** 2) * da
    p_mode = np.sum(np.abs(mode) ** 2) * da
    denom = p_E * p_mode
    if denom == 0:
        return 0.0
    return float(np.abs(overlap) ** 2 / denom)


# ============================================================================
# v4.15.0 (C.2) -- Polarisation-aware Strehl + coupling
# ============================================================================

def _validate_vector_field_shapes(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: Optional[np.ndarray],
    *,
    name: str,
) -> None:
    """Internal: validate that vector-field components are 2-D and
    share a common shape.  Raises ``ValueError`` on any mismatch."""
    if Ex.ndim != 2:
        raise ValueError(
            f"{name}: Ex must be 2-D; got shape {Ex.shape!r} "
            f"(ndim={Ex.ndim}).")
    if Ey.ndim != 2:
        raise ValueError(
            f"{name}: Ey must be 2-D; got shape {Ey.shape!r} "
            f"(ndim={Ey.ndim}).")
    if Ex.shape != Ey.shape:
        raise ValueError(
            f"{name}: Ex and Ey shape mismatch -- Ex is {Ex.shape!r}, "
            f"Ey is {Ey.shape!r}.")
    if Ez is not None:
        if Ez.ndim != 2:
            raise ValueError(
                f"{name}: Ez must be 2-D when provided; got shape "
                f"{Ez.shape!r} (ndim={Ez.ndim}).")
        if Ez.shape != Ex.shape:
            raise ValueError(
                f"{name}: Ez shape mismatch -- Ez is {Ez.shape!r}, "
                f"expected {Ex.shape!r} to match (Ex, Ey).")


def strehl_vector(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: Optional[np.ndarray] = None,
    *,
    reference: Sequence[np.ndarray],
) -> float:
    r"""Scalar Strehl-like metric over the vector-field components.

    Generalises :func:`strehl_ratio` to a Jones field with an
    optional z-component.  The metric is the peak-intensity ratio of
    the **total** vector intensity ``|Ex|^2 + |Ey|^2 + |Ez|^2`` over
    the input field vs an explicit reference field, after
    normalising both to equal total power:

    .. math::
        S = \frac{\max(|E_x|^2 + |E_y|^2 + |E_z|^2)}{P}
            \cdot \frac{P_\mathrm{ref}}{\max(|E_x^\mathrm{ref}|^2
            + |E_y^\mathrm{ref}|^2 + |E_z^\mathrm{ref}|^2)}

    This matches the scalar :func:`strehl_ratio` definition mode-for-
    mode when ``Ey = Ez = 0`` (so a single linear polarisation
    coincides exactly with the scalar Strehl).

    Parameters
    ----------
    Ex, Ey : ndarray (complex, Ny x Nx)
        Transverse vector-field components.  Must be 2-D arrays of
        matching shape.
    Ez : ndarray (complex, Ny x Nx), optional
        Longitudinal component.  When ``None``, treated as
        identically zero -- equivalent to passing a zero-array of the
        same shape.
    reference : tuple of arrays, required (keyword-only)
        ``(Ex_ref, Ey_ref)`` or ``(Ex_ref, Ey_ref, Ez_ref)`` for the
        diffraction-limited reference field, sampled on the same grid
        as the input.  The reference is mandatory: without an
        explicit aperture-truncated reference, the
        equal-total-power-normalised ratio is unbounded above 1.0
        for any field more peaked than uniform (audit V4.15.0
        P1-F1-3).  Pass the unaberrated propagation result for the
        same aperture / system.

    Returns
    -------
    strehl : float
        Vector Strehl-like metric.  ``1.0`` when ``(Ex, Ey, Ez) ==
        (Ex_ref, Ey_ref, Ez_ref)``; ``< 1.0`` for an aberrated input
        relative to its unaberrated reference.  Returns ``0.0`` if
        either the input or reference field has zero total power.

    Raises
    ------
    ValueError
        If any input array is not 2-D, if shapes do not match, or if
        ``reference`` is missing / not a 2-tuple or 3-tuple.

    See Also
    --------
    strehl_ratio : scalar peak-ratio Strehl.
    coupling_efficiency_vector : vector mode-overlap coupling.

    Notes
    -----
    Breaking change at v4.15.1: the previous default
    ``reference=None`` branch (uniform plane wave of matching total
    power) was removed because it produced ``S > 1`` for any focused
    PSF -- a focused PSF concentrates more power into its peak than a
    uniform field of equal total power.  Callers must now supply the
    diffraction-limited reference explicitly.  The simplest
    replacement is to propagate the same aperture / system without
    aberrations and pass the resulting ``(Ex_ref, Ey_ref[, Ez_ref])``.
    """
    if reference is None:
        raise ValueError(
            "strehl_vector: 'reference' is required since v4.15.1.  "
            "Pass the diffraction-limited reference field tuple "
            "(Ex_ref, Ey_ref) or (Ex_ref, Ey_ref, Ez_ref); the prior "
            "default plane-wave reference returned Strehl > 1 for "
            "any focused PSF (audit V4.15.0 P1-F1-3).")
    _validate_vector_field_shapes(
        Ex, Ey, Ez, name='strehl_vector')
    xp = _xp_of(Ex, Ey) if Ez is None else _xp_of(Ex, Ey, Ez)

    # Total vector intensity (sum over polarisation components).
    I_total = xp.abs(Ex) ** 2 + xp.abs(Ey) ** 2
    if Ez is not None:
        I_total = I_total + xp.abs(Ez) ** 2
    P = float(xp.sum(I_total))
    if P <= 0.0:
        return 0.0
    I_max = float(xp.max(I_total))

    # User-supplied reference.  Accept (Ex_ref, Ey_ref) or
    # (Ex_ref, Ey_ref, Ez_ref).
    try:
        n_ref = len(reference)
    except TypeError:
        raise ValueError(
            f"strehl_vector: reference must be a 2- or 3-tuple of "
            f"vector components; got non-sequence {type(reference).__name__!r}.")
    if n_ref == 2:
        Ex_ref, Ey_ref = reference
        Ez_ref = None
    elif n_ref == 3:
        Ex_ref, Ey_ref, Ez_ref = reference
    else:
        raise ValueError(
            f"strehl_vector: reference must be a 2- or 3-tuple of "
            f"vector components; got length {n_ref}.")
    _validate_vector_field_shapes(
        xp.asarray(Ex_ref), xp.asarray(Ey_ref),
        None if Ez_ref is None else xp.asarray(Ez_ref),
        name='strehl_vector (reference)')
    if Ex_ref.shape != Ex.shape:
        raise ValueError(
            f"strehl_vector: reference shape {Ex_ref.shape!r} "
            f"does not match input shape {Ex.shape!r}.")
    I_ref = xp.abs(Ex_ref) ** 2 + xp.abs(Ey_ref) ** 2
    if Ez_ref is not None:
        I_ref = I_ref + xp.abs(Ez_ref) ** 2
    P_ref = float(xp.sum(I_ref))
    I_ref_max = float(xp.max(I_ref))
    if P_ref <= 0.0 or I_ref_max <= 0.0:
        return 0.0
    # Equal-total-power normalisation, identical to scalar
    # strehl_ratio convention.
    return float(I_max / P) * float(P_ref / I_ref_max)


def coupling_efficiency_vector(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: Optional[np.ndarray] = None,
    *,
    mode_Ex: np.ndarray,
    mode_Ey: np.ndarray,
    mode_Ez: Optional[np.ndarray] = None,
    dx: float,
    dy: Optional[float] = None,
) -> float:
    r"""Vector mode-overlap coupling efficiency between a field and a
    target vector mode.

    Generalises :func:`coupling_efficiency` to a Jones field with
    optional longitudinal component.  Returns

    .. math::
        \eta = \frac{|\langle \mathbf{E} | \mathbf{m} \rangle|^2}
                    {\langle \mathbf{E} | \mathbf{E} \rangle
                     \, \langle \mathbf{m} | \mathbf{m} \rangle}

    where
    :math:`\langle \mathbf{E} | \mathbf{m} \rangle = \int (E_x^* m_x +
    E_y^* m_y + E_z^* m_z) \, dA`.

    Parameters
    ----------
    Ex, Ey : ndarray (complex, Ny x Nx)
        Incoming vector-field transverse components on a regular grid.
    Ez : ndarray (complex, Ny x Nx), optional
        Longitudinal component; treated as zero when ``None``.
    mode_Ex, mode_Ey : ndarray (complex, Ny x Nx)
        Target vector-mode transverse components (same grid).
    mode_Ez : ndarray (complex, Ny x Nx), optional
        Target longitudinal component; treated as zero when ``None``.
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    eta : float
        Coupling efficiency in ``[0, 1]``.

    Raises
    ------
    ValueError
        If any field is not 2-D, or if shapes do not match across the
        Ex / Ey / Ez / mode_Ex / mode_Ey / mode_Ez set.

    See Also
    --------
    coupling_efficiency : scalar single-component overlap.
    strehl_vector : peak-ratio vector Strehl.
    """
    _validate_vector_field_shapes(
        Ex, Ey, Ez, name='coupling_efficiency_vector')
    _validate_vector_field_shapes(
        mode_Ex, mode_Ey, mode_Ez,
        name='coupling_efficiency_vector (mode)')
    if mode_Ex.shape != Ex.shape:
        raise ValueError(
            f"coupling_efficiency_vector: mode shape "
            f"{mode_Ex.shape!r} does not match field shape "
            f"{Ex.shape!r}.")
    if dy is None:
        dy = dx
    da = float(dx) * float(dy)

    xp = _xp_of(Ex, Ey, mode_Ex, mode_Ey)

    # Inner product <E | mode> with optional Ez branches.
    overlap = xp.sum(xp.conj(Ex) * mode_Ex + xp.conj(Ey) * mode_Ey) * da
    p_field = xp.sum(xp.abs(Ex) ** 2 + xp.abs(Ey) ** 2) * da
    p_mode = xp.sum(xp.abs(mode_Ex) ** 2 + xp.abs(mode_Ey) ** 2) * da
    if Ez is not None or mode_Ez is not None:
        # Pad missing components with zeros so the overlap formula is
        # symmetric and matches the scalar reduction when both Ez are
        # absent (zero contribution).
        if Ez is not None and mode_Ez is not None:
            overlap = overlap + xp.sum(xp.conj(Ez) * mode_Ez) * da
        if Ez is not None:
            p_field = p_field + xp.sum(xp.abs(Ez) ** 2) * da
        if mode_Ez is not None:
            p_mode = p_mode + xp.sum(xp.abs(mode_Ez) ** 2) * da

    p_field_f = float(p_field)
    p_mode_f = float(p_mode)
    denom = p_field_f * p_mode_f
    if denom <= 0.0:
        return 0.0
    return float(xp.abs(overlap) ** 2 / denom)

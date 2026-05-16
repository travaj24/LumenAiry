"""
lumenairy.raytrace.seidel_analysis -- field-dependent Seidel analysis.

The existing :func:`seidel_coefficients` returns Seidel sums for a
single field height (the ``field_angle`` kwarg).  This module adds:

* :func:`seidel_field_sweep` -- evaluate Seidel sums at a *grid* of
  field heights in a single call.  Returns per-surface arrays of
  shape ``(N_surfaces, N_fields)`` and total sums of shape
  ``(N_fields,)``, so users can plot S1-S5 vs field height directly.
* :func:`seidel_wfe` -- reconstruct the third-order wavefront error
  ``W(rho, theta)`` from a Seidel result dict using the standard
  Hopkins / Welford expansion.

Both functions are wrappers around the canonical
:func:`seidel_coefficients` (which remains unchanged), so any future
fix to the underlying Hopkins computation flows through.

References
----------
* Welford, *Aberrations of Optical Systems*, Adam Hilger 1986.
* Born & Wolf, *Principles of Optics*, ch. 4.4.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .core import seidel_coefficients


# ----------------------------------------------------------------------
# Field-height sweep
# ----------------------------------------------------------------------

def seidel_field_sweep(
    surfaces: List[Any],
    wavelength: float,
    field_heights: Union[float, Sequence[float], np.ndarray],
    *,
    object_distance: float = float('inf'),
    stop_index: Optional[int] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Compute Seidel coefficients across a grid of field heights.

    Wraps :func:`lumenairy.seidel_coefficients` once per requested
    field height and stacks the per-surface arrays into 2-D output of
    shape ``(N_surfaces, N_fields)``.  Useful for plotting S1-S5 vs
    field angle, locating zero-coma stops, or quickly seeing how each
    aberration scales off-axis.

    Parameters
    ----------
    surfaces : list of Surface
        Same as :func:`seidel_coefficients`.
    wavelength : float
        Vacuum wavelength [m].
    field_heights : float or 1-D array-like
        Chief-ray field angles to evaluate at [rad].  Scalars are
        promoted to a 1-element array.  Magnitudes are commonly in
        ``[1e-4, 1e-1]`` for radian field angles.
    object_distance : float, default ``np.inf``
        Object distance from the first surface [m].
    stop_index : int, optional
        Explicit stop surface; defaults to :func:`find_stop`.

    Returns
    -------
    result : dict with keys
        * ``'field_heights'`` -- ``(N_fields,)`` array of evaluated
          field angles [rad].
        * ``'S1', 'S2', 'S3', 'S4', 'S5'`` -- per-surface contributions
          of shape ``(N_surfaces, N_fields)``.
        * ``'total'`` -- dict of ``(N_fields,)`` arrays summing each
          aberration across surfaces.
        * ``'labels'`` -- human-readable per-aberration names.
        * ``'stop_index'`` -- the stop used.
        * ``'y_marginal'`` -- ``(N_surfaces,)`` marginal-ray heights
          (field-independent; identical across fields).
        * ``'y_chief'`` -- ``(N_surfaces, N_fields)`` chief-ray
          heights (linear in field_angle).
    abcd : ndarray (2 x 2)
        System ABCD matrix; identical across fields.

    See Also
    --------
    seidel_coefficients : single-field-angle workhorse.
    seidel_wfe : reconstruct W(rho, theta) from totals.

    Examples
    --------
    >>> import numpy as np
    >>> import lumenairy as la
    >>> presc = la.make_singlet(R1=50e-3, R2=np.inf, d=3e-3,
    ...                          glass='N-BK7', aperture=10e-3)
    >>> surfaces = la.surfaces_from_prescription(presc)
    >>> heights = np.linspace(0, 0.1, 11)
    >>> result, _ = la.seidel_field_sweep(surfaces, 1.31e-6, heights)
    >>> result['total']['S3'].shape   # astigmatism vs field
    (11,)
    """
    heights = np.atleast_1d(np.asarray(field_heights, dtype=float))
    if heights.ndim != 1:
        raise ValueError(
            f"field_heights must be a scalar or 1-D array; got shape "
            f"{heights.shape}")

    per_field = []
    abcd = None
    for h in heights:
        s, m = seidel_coefficients(
            surfaces, wavelength,
            object_distance=object_distance,
            stop_index=stop_index,
            field_angle=float(h),
        )
        per_field.append(s)
        if abcd is None:
            abcd = m

    keys = ('S1', 'S2', 'S3', 'S4', 'S5')
    result = {
        'field_heights': heights,
        'labels': per_field[0]['labels'],
        'stop_index': per_field[0]['stop_index'],
        # Field-independent: marginal-ray heights are the same in
        # every per-field call; report the first as representative.
        'y_marginal': per_field[0]['y_marginal'],
        # Field-dependent: stack chief-ray heights as
        # (N_surfaces, N_fields).
        'y_chief': np.stack([s['y_chief'] for s in per_field], axis=-1),
    }
    for k in keys:
        result[k] = np.stack([s[k] for s in per_field], axis=-1)
    result['total'] = {
        k: np.array([s['total'][k] for s in per_field]) for k in keys
    }
    return result, abcd


# ----------------------------------------------------------------------
# WFE reconstruction from Seidel sums
# ----------------------------------------------------------------------

def seidel_wfe(
    seidel_or_totals: Dict[str, Any],
    rho: np.ndarray,
    theta: np.ndarray,
    *,
    field_index: Optional[int] = None,
    field_angle: Optional[float] = None,
) -> np.ndarray:
    """Reconstruct the third-order wavefront error from Seidel totals.

    Uses the Hopkins / Welford expansion:

    .. math::
        W(\\rho, \\theta) = \\tfrac{1}{8} S_1\\rho^4
                          + \\tfrac{1}{2} S_2\\rho^3 \\cos\\theta
                          + \\tfrac{1}{2} S_3\\rho^2 \\cos^2\\theta
                          + \\tfrac{1}{4} S_4 \\sigma^2 \\rho^2
                          + \\tfrac{1}{2} S_5\\rho \\cos\\theta

    where :math:`\\sigma` is the chief-ray field angle used to compute
    the Seidel sums.  S1, S2, S3, S5 already encode their appropriate
    field-height powers (S2 ~ sigma, S3 ~ sigma^2, S5 ~ sigma or
    sigma^3 depending on Hopkins convention); S4 is the
    field-independent Petzval Hopkins sum and is multiplied by
    sigma^2 inside this function.

    Parameters
    ----------
    seidel_or_totals : dict
        Either:

        * A :func:`seidel_coefficients` result dict (includes
          ``'field_angle'`` since 4.3.0), or
        * A bare totals dict ``{'S1', 'S2', 'S3', 'S4', 'S5'}``
          (then ``field_angle`` must be supplied explicitly), or
        * A :func:`seidel_field_sweep` result dict (pass
          ``field_index`` to pick one).
    rho : ndarray
        Normalized pupil radius in ``[0, 1]``.
    theta : ndarray
        Pupil azimuth angle [rad].  Must broadcast with ``rho``.
    field_index : int, optional
        When ``seidel_or_totals`` came from
        :func:`seidel_field_sweep`, the index of the field height
        to reconstruct at.  Ignored for single-field inputs.
    field_angle : float, optional
        Chief-ray field angle [rad] used to scale the S4 (Petzval)
        term.  Defaults to ``seidel_or_totals['field_angle']`` for
        single-field inputs or
        ``seidel_or_totals['field_heights'][field_index]`` for
        sweep inputs.  Required only when passing a bare totals
        dict (since it has no embedded field-angle metadata).

    Returns
    -------
    W : ndarray
        Wavefront error in the same units as the Seidel sums.

    See Also
    --------
    seidel_coefficients : compute the input sums.
    seidel_field_sweep : multi-field version.
    """
    if 'total' in seidel_or_totals:
        T = seidel_or_totals['total']
    else:
        T = seidel_or_totals

    # Resolve field_angle for the S4·H² term.
    sigma = field_angle
    if sigma is None:
        if 'field_heights' in seidel_or_totals and field_index is not None:
            sigma = float(seidel_or_totals['field_heights'][field_index])
        elif 'field_angle' in seidel_or_totals:
            sigma = float(seidel_or_totals['field_angle'])
        else:
            raise ValueError(
                "seidel_wfe: cannot determine the chief-ray field angle.  "
                "Either supply field_angle=... explicitly, or pass a dict "
                "from seidel_coefficients / seidel_field_sweep (4.3.0+) "
                "which carries the field-angle metadata.")

    rho = np.asarray(rho)
    theta = np.asarray(theta)

    def _pick(v):
        if np.ndim(v) > 0 and field_index is not None:
            return float(v[field_index])
        return float(v)

    S1 = _pick(T['S1'])
    S2 = _pick(T['S2'])
    S3 = _pick(T['S3'])
    S4 = _pick(T['S4'])
    S5 = _pick(T['S5'])

    # 4.9 fix: Petzval term needs |H|² (Lagrange invariant squared),
    # NOT bare sigma².  S4 = -c·(n2-n1)/(n1·n2) is the H-less
    # surface-property form; the WFE expansion's S4 contribution
    # is (1/4)·S4·H²·ρ².  Pre-4.9 used sigma² alone, off by
    # (y_pupil)² ≈ (D/2)² ≈ 1.6e-4 m² for a 25 mm singlet at
    # f/4 -- producing ~4 orders of magnitude of phantom Petzval.
    # Use the explicit lagrange_invariant carried by 4.9+
    # seidel_coefficients results when available; fall back to
    # f·sigma (image-height proxy, dimensionally correct for
    # object-at-infinity) using the embedded ABCD if present;
    # last resort, fall back to the legacy sigma² behaviour with
    # a one-time warning so users notice the magnitude shift.
    H_sq = None
    if 'lagrange_invariant' in seidel_or_totals:
        H_sq = float(seidel_or_totals['lagrange_invariant']) ** 2
    elif 'abcd' in seidel_or_totals:
        # Recover effective focal length from the embedded ABCD.
        # H ≈ f · sigma for object at infinity, n_obj = 1.
        abcd = np.asarray(seidel_or_totals['abcd'])
        if abcd.shape == (2, 2) and abs(float(abcd[1, 0])) > 1e-30:
            f_eff = -1.0 / float(abcd[1, 0])
            H_sq = (f_eff * sigma) ** 2

    if H_sq is None:
        # 4.10: refuse to silently apply the wrong sigma² scaling here.
        # Users hitting this branch are passing a bare totals dict (no
        # 'lagrange_invariant', no 'abcd') and would otherwise get
        # Petzval magnitudes off by (D/2)² -- often several orders of
        # magnitude.  Mark the Petzval contribution as NaN so any
        # downstream consumer sees a clear failure rather than wrong
        # numbers.  To get a valid Petzval term, pass the full result
        # dict from seidel_coefficients() or supply lagrange_invariant
        # explicitly via the helper-dict route.
        H_sq = np.nan

    cos_t = np.cos(theta)
    rho2 = rho ** 2
    rho3 = rho2 * rho
    rho4 = rho3 * rho
    return ((1.0 / 8.0) * S1 * rho4
            + (1.0 / 2.0) * S2 * rho3 * cos_t
            + (1.0 / 2.0) * S3 * rho2 * cos_t ** 2
            + (1.0 / 4.0) * S4 * H_sq * rho2
            + (1.0 / 2.0) * S5 * rho * cos_t)


__all__ = [
    'seidel_field_sweep',
    'seidel_wfe',
]

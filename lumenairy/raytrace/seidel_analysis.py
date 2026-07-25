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

    # 4.13.0 Phase 3 perf: hoist the entire field-independent workload
    # (glass-index lookups, pre-stop ABCD, marginal-ray trace, system
    # ABCD, per-surface S1/S4) out of the field loop.  The paraxial
    # Seidel formalism is exactly linear in the chief-ray initial
    # conditions, and those scale linearly in ``field_angle``:
    #
    #     y_c, nu_c, A_c, H_lagrange   propto field_angle
    #     S1, S4                       field-independent
    #     S2 = -(A_m A_c) h delta_un   propto field_angle
    #     S3 = -(A_c^2) h delta_un     propto field_angle^2
    #     S5 = -(A_c/A_m)(S3 + H^2 S4) propto field_angle^3
    #
    # A single call to ``seidel_coefficients(..., field_angle=1.0)``
    # captures all per-surface marginal/chief data at unit field; the
    # analytical scaling below reproduces every per-field result to
    # machine precision.  This replaces a length-N_fields Python loop
    # (each iteration re-runs the marginal trace + system_abcd) with
    # one call plus vectorised scaling, giving an N_fields-fold
    # speedup on the seidel-sweep step.
    ref, abcd = seidel_coefficients(
        surfaces, wavelength,
        object_distance=object_distance,
        stop_index=stop_index,
        field_angle=1.0,
    )

    sigma = heights                       # (N_fields,)
    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma

    # Per-surface scaling: row = surface index, col = field index.
    # S1 / S4 are field-independent; broadcast onto (N_surf, N_fields).
    n_surf = ref['S1'].shape[0]
    n_fields = sigma.size
    S1_arr = np.broadcast_to(
        ref['S1'][:, None], (n_surf, n_fields)).copy()
    S2_arr = ref['S2'][:, None] * sigma[None, :]
    S3_arr = ref['S3'][:, None] * sigma2[None, :]
    S4_arr = np.broadcast_to(
        ref['S4'][:, None], (n_surf, n_fields)).copy()
    S5_arr = ref['S5'][:, None] * sigma3[None, :]
    y_chief_arr = ref['y_chief'][:, None] * sigma[None, :]

    result = {
        'field_heights': heights,
        'labels': ref['labels'],
        'stop_index': ref['stop_index'],
        # Field-independent: marginal-ray heights are the same in
        # every per-field call.
        'y_marginal': ref['y_marginal'],
        # Field-dependent: chief-ray heights scale linearly in field.
        'y_chief': y_chief_arr,
        'S1': S1_arr,
        'S2': S2_arr,
        'S3': S3_arr,
        'S4': S4_arr,
        'S5': S5_arr,
        # RT-9 (AUDIT_RAYTRACE_CORE): carry the per-field Lagrange invariant
        # (H propto field_angle, so H_field = H_unit * sigma) AND the system
        # ABCD so ``seidel_wfe(sweep_result, field_index=k)`` -- the pairing
        # the docstrings advertise -- reaches the corrected H^2 Petzval path
        # instead of always landing in the bare-sigma^2 fallback.
        'lagrange_invariant': float(ref['lagrange_invariant']) * sigma,
        'abcd': abcd,
    }
    # Totals follow the same field-angle scaling as the per-surface
    # arrays since np.sum commutes with the broadcast multiplication.
    ref_total = ref['total']
    result['total'] = {
        'S1': np.full(n_fields, float(ref_total['S1'])),
        'S2': float(ref_total['S2']) * sigma,
        'S3': float(ref_total['S3']) * sigma2,
        'S4': np.full(n_fields, float(ref_total['S4'])),
        'S5': float(ref_total['S5']) * sigma3,
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

    Uses the Hopkins / Welford expansion (in Lagrange-invariant
    :math:`H^2` form, with the field-curvature DC term written
    explicitly):

    .. math::
        W(\\rho, \\theta) = -\\Big[
                            \\tfrac{1}{8} S_1\\rho^4
                          + \\tfrac{1}{2} S_2\\rho^3 \\cos\\theta
                          + \\tfrac{1}{2} S_3\\rho^2 \\cos^2\\theta
                          + \\tfrac{1}{4} S_3 \\rho^2
                          + \\tfrac{1}{4} S_4 H^2 \\rho^2
                          + \\tfrac{1}{2} S_5\\rho \\cos\\theta
                          \\Big]

    where :math:`H = n_0 \\, y_c \\, u_m - n_0 \\, y_m \\, u_c` is the
    Lagrange invariant (computed inside :func:`seidel_coefficients`).
    S1, S2, S3, S5 already encode their appropriate field-height
    powers; S4 is the field-independent Petzval Hopkins sum and is
    multiplied by :math:`H^2` inside this function.  Pre-4.11 the
    docstring showed :math:`\\sigma^2` here -- the code has always
    used :math:`H^2`, which equals :math:`\\sigma^2 \\, f_{\\rm eff}^2`
    in the small-angle limit but is the right invariant for
    finite-conjugate and stop-shifted systems.

    The :math:`(1/4) S_3 \\rho^2` term is the field-curvature DC
    companion to the astigmatism term (Welford eq. 7.11; in
    Welford's mixed notation the FC DC reads :math:`(1/4)(S_{III}
    + S_{IV})`).  Pre-4.11.2 this DC term was missing from both the
    docstring and the implementation, so any synthetic / measured
    S3 contributed to astigmatism (cos^2 theta) but not to the
    rotationally symmetric field-curvature defocus.

    **Sign convention.**  The leading minus on the bracket converts this
    library's Seidel sums, which carry ``code = -S_Welford`` (the S3-1
    note in :func:`lumenairy.seidel_coefficients`'s refracting branch),
    into Welford's, for which the bracketed expansion is written.  With
    it, ``W`` is an OPTICAL PATH DIFFERENCE referenced to the paraxial
    image point -- ``W = OPL(pupil point) - OPL(reference ray)``, the
    reference ray being the one through the pupil centre (``rho = 0``),
    both paths measured from a common incoming wavefront -- so

    * ``W < 0`` means that pupil point's path is SHORTER than the
      reference (wavefront ADVANCED, leads the reference sphere);
    * ``W > 0`` means LONGER (wavefront RETARDED, lags).

    A positive singlet with undercorrected spherical aberration
    therefore has ``W(rho = 1) < 0``.  R-2
    (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25): pre-fix the expansion was
    composed directly out of the ``-S_Welford`` sums and so returned
    ``-W``; measured against an exact-trace wavefront oracle the ratio
    was ``-0.9975 ... -0.9998`` on four singlets over ``rho in
    [0.3, 1]``, and ``-1.000`` term by term for the ``rho^2``, ``rho^3``
    and ``rho^4`` terms.  Magnitude-only consumers (RMS / PV WFE, Strehl
    proxies) are unaffected by the fix; anything that ADDS ``W`` to a
    pupil phase, fits Zernikes to it, or reads the sign of coma /
    distortion asymmetry flips.  Callers passing a BARE totals dict are
    on the same convention -- the flip is applied to the ingested
    ``S1..S5`` whatever their source, so hand-written sums are
    interpreted as library-convention sums.

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
        Wavefront error in the same units as the Seidel sums (metres for
        a metre-unit prescription).  Signed -- see "Sign convention"
        above; ``W < 0`` = path shorter than the reference = wavefront
        advanced.

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

    # R-2 (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25): ingest on WELFORD's
    # sign convention.  ``seidel_coefficients`` returns ``code =
    # -S_Welford`` (see the S3-1 note in ``seidel.py``'s refracting
    # branch); the expansion composed below is Welford's.  Without this
    # flip the function returned -W -- exact-trace ratio -0.9975..-0.9998
    # across four singlets, uniformly -1 term by term.  Flipping here
    # (rather than negating the return) keeps the composition literally
    # textbook and puts the convention conversion in one visible place.
    S1 = -_pick(T['S1'])
    S2 = -_pick(T['S2'])
    S3 = -_pick(T['S3'])
    S4 = -_pick(T['S4'])
    S5 = -_pick(T['S5'])

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
        # RT-9: index by field_index too -- ``seidel_field_sweep`` stores a
        # per-field ``(N_fields,)`` invariant array, so a bare ``float(...)``
        # would raise; ``_pick`` selects the scalar for the chosen field.
        H_sq = _pick(seidel_or_totals['lagrange_invariant']) ** 2
    elif 'abcd' in seidel_or_totals:
        # Recover effective focal length from the embedded ABCD.
        # H ≈ f · sigma for object at infinity, n_obj = 1.
        abcd = np.asarray(seidel_or_totals['abcd'])
        if abcd.shape == (2, 2) and abs(float(abcd[1, 0])) > 1e-30:
            f_eff = -1.0 / float(abcd[1, 0])
            H_sq = (f_eff * sigma) ** 2

    if H_sq is None:
        # 4.10: keep the bare-sigma² fallback for callers who pass a
        # totals dict + explicit field_angle (this is the documented
        # back-compat path used by some unit tests and bench scripts),
        # but emit a one-time warning so users notice the magnitude
        # difference relative to the full lagrange-invariant path.
        import warnings
        warnings.warn(
            "seidel_wfe: input dict carries neither 'lagrange_invariant' "
            "nor 'abcd'; falling back to bare-sigma² scaling for the "
            "S4 Petzval term.  Pass a 4.9+ seidel_coefficients() result "
            "to use the corrected H² scaling (which differs by "
            "(y_pupil)² ≈ (D/2)² for typical singlets).",
            RuntimeWarning, stacklevel=2,
        )
        H_sq = sigma * sigma

    cos_t = np.cos(theta)
    rho2 = rho ** 2
    rho3 = rho2 * rho
    rho4 = rho3 * rho
    # 4.11.2: include the field-curvature DC companion (1/4)*S3*rho^2.
    # Welford eq. 7.11 expands the third-order WFE as
    #   (1/2)*S3*rho^2*cos^2 theta  +  (1/4)*(S3 + S4)*rho^2
    # (in Welford's H-folded notation; here S4 also needs H^2 to be
    # dimensionally consistent with S3).  Pre-4.11.2 the FC DC was
    # silently dropped.
    return ((1.0 / 8.0) * S1 * rho4
            + (1.0 / 2.0) * S2 * rho3 * cos_t
            + (1.0 / 2.0) * S3 * rho2 * cos_t ** 2
            + (1.0 / 4.0) * S3 * rho2
            + (1.0 / 4.0) * S4 * H_sq * rho2
            + (1.0 / 2.0) * S5 * rho * cos_t)


__all__ = [
    'seidel_field_sweep',
    'seidel_wfe',
]

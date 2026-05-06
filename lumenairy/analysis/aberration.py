"""
lumenairy.analysis.aberration -- unified aberration analysis.

A single entry point for both Seidel-coefficient (geometric, ray-trace
based) and LG aberration-tensor (asymptotic, wave-based) analyses of a
prescription.  Pulls from :mod:`lumenairy.raytrace` for Seidel
coefficients and from :mod:`lumenairy.propagators.asymptotic` for the
LG tensor; presents both behind a single
:func:`aberration_summary` call.

Use this module when you want a one-shot characterisation of a
prescription's aberrations without remembering which subpackage owns
which formalism.

Author: Andrew Traverso
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AberrationSummary:
    """Combined aberration report for a prescription.

    Attributes
    ----------
    seidel_total : ndarray, shape (5,)
        Summed Seidel coefficients (S1 spherical, S2 coma, S3
        astigmatism, S4 field curvature, S5 distortion) in metres.
    seidel_per_surface : list of dicts
        Per-surface Seidel breakdown.  ``None`` if the trace failed
        (e.g. degenerate prescription).
    efl, bfl : float
        Effective and back focal lengths [m].  ``None`` if the ABCD
        extraction failed.
    lg_tensor : object or None
        :class:`AberrationTensorResult` (NumPy) for the LG modal
        tensor at the chief-ray image of ``source_point``.  ``None``
        when ``include_lg_tensor=False`` or the asymptotic fit
        failed.
    fit : object or None
        Raw :class:`CanonicalPolyFit` used for the LG tensor.  Useful
        when callers want to drive their own propagation downstream.
    wavelength : float
        Wavelength used for the analysis [m].
    notes : list of str
        Diagnostic messages (e.g. why a step was skipped).
    """
    seidel_total: np.ndarray
    seidel_per_surface: Optional[List[Dict[str, Any]]] = None
    efl: Optional[float] = None
    bfl: Optional[float] = None
    lg_tensor: Any = None
    fit: Any = None
    wavelength: float = 0.0
    notes: List[str] = field(default_factory=list)


def aberration_summary(
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    source_point: Tuple[float, float] = (0.0, 0.0),
    include_lg_tensor: bool = True,
    output_modes: Optional[List[Tuple[int, int]]] = None,
    w_s: float = 50e-6,
    w_p: float = 0.05,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    differentiable: bool = False,
) -> AberrationSummary:
    """One-shot aberration analysis combining Seidel + LG-tensor views.

    Runs the geometric ray trace to extract Seidel coefficients,
    EFL / BFL, and (optionally) builds the canonical polynomial fit
    used by the asymptotic propagator and computes the LG aberration
    tensor at the chief-ray landing of ``source_point``.

    Parameters
    ----------
    prescription : dict
        Lumenairy prescription dict.
    wavelength : float
        Vacuum wavelength [m].
    source_point : (float, float), default (0, 0)
        Object-plane field point used to anchor the LG tensor's chief
        ray.  Ignored when ``include_lg_tensor=False``.
    include_lg_tensor : bool, default True
        If True, build a canonical polynomial fit of the prescription
        and evaluate the LG aberration tensor at the chief image
        point.  Skip this for very fast geometric-only summaries.
    output_modes : list of (p, ell), optional
        Output LG modes for the tensor.  Default: piston / defocus /
        spherical / tilts / coma / astigmatism / trefoil.
    w_s, w_p : float
        LG basis waists for source / pupil.
    fit_kwargs : dict, optional
        Keyword arguments forwarded to
        :func:`fit_canonical_polynomials`.
    differentiable : bool, default False
        If True, route the LG-tensor branch through
        :func:`aberration_tensor_lg00_jax` (JAX, differentiable via
        ``jax.grad``) instead of the NumPy
        :func:`aberration_tensor`.  The result's ``L`` field will be
        a JAX (1, 1) array (only the (0,0)-channel is computed).
        Newton solve for the envelope-stationary point still runs in
        NumPy.

    Returns
    -------
    AberrationSummary
    """
    from ..raytrace import (
        surfaces_from_prescription, system_abcd, seidel_coefficients,
    )

    notes: List[str] = []

    # --- Geometric leg: Seidel + ABCD --------------------------------
    surfs = surfaces_from_prescription(prescription)
    try:
        _, efl, bfl, _ = system_abcd(surfs, wavelength)
        efl_v = float(efl) if np.isfinite(efl) else None
        bfl_v = float(bfl) if np.isfinite(bfl) else None
    except Exception as exc:
        notes.append(f"system_abcd failed: {type(exc).__name__}: {exc}")
        efl_v = bfl_v = None

    seidel_total = np.zeros(5, dtype=np.float64)
    per_surf: Optional[List[Dict[str, Any]]] = None
    try:
        raw = seidel_coefficients(surfs, wavelength)
        per_surf_dict = raw[0] if isinstance(raw, tuple) else raw
        if isinstance(per_surf_dict, dict):
            # Total is provided as a 'total' sub-dict by the raytrace
            # module; fall back to summing the per-surface arrays if
            # absent.
            total = per_surf_dict.get('total')
            for idx, k_name in enumerate(('S1', 'S2', 'S3', 'S4', 'S5')):
                if total is not None and k_name in total:
                    seidel_total[idx] = float(total[k_name])
                else:
                    vals = per_surf_dict.get(k_name)
                    if vals is not None:
                        seidel_total[idx] = float(np.sum(np.asarray(vals)))
            # Per-surface breakdown.
            ps_arrays = {
                k: np.asarray(v) for k, v in per_surf_dict.items()
                if k in ('S1', 'S2', 'S3', 'S4', 'S5')
                and hasattr(v, '__len__')
            }
            n_surf = max((arr.size for arr in ps_arrays.values()),
                         default=0)
            if n_surf > 0:
                per_surf = []
                for i in range(n_surf):
                    per_surf.append({
                        k: float(arr[i])
                        for k, arr in ps_arrays.items() if i < arr.size
                    })
        else:
            seidel_total = np.asarray(raw, dtype=np.float64).ravel()
            if seidel_total.size != 5:
                seidel_total = np.zeros(5)
    except Exception as exc:
        notes.append(f"seidel_coefficients failed: {type(exc).__name__}: {exc}")

    # --- Wave leg: LG aberration tensor -------------------------------
    lg_result = None
    fit = None
    if include_lg_tensor:
        try:
            from ..propagators.asymptotic import (
                fit_canonical_polynomials,
                solve_envelope_stationary,
                aberration_tensor,
            )
            fk = dict(fit_kwargs or {})
            fk.setdefault('source_box_half', max(2.5 * w_s, 1e-5))
            fk.setdefault('pupil_box_half', max(2.5 * w_p, 1e-3))
            fk.setdefault('n_field', 6)
            fk.setdefault('n_pupil', 6)
            fk.setdefault('poly_order', 4)
            fit = fit_canonical_polynomials(
                prescription, wavelength=wavelength, **fk)
            # Image at the chief-ray landing of source_point
            v_star, _, _ = solve_envelope_stationary(
                fit, (fit.s2x_centre, fit.s2y_centre), source_point,
                w_s=w_s, w_p=w_p,
                v2_centre=(fit.v2x_centre, fit.v2y_centre),
            )
            if differentiable:
                from ..backend import JAX_AVAILABLE
                if not JAX_AVAILABLE:
                    raise ImportError(
                        "differentiable=True requires JAX; install "
                        "with `pip install jax`.")
                from ..propagators.asymptotic import (
                    aberration_tensor_lg00_jax,
                )
                lg_result = aberration_tensor_lg00_jax(
                    fit, (fit.s2x_centre, fit.s2y_centre), v_star,
                    source_point=source_point,
                    w_s=w_s, w_p=w_p,
                    v2_centre=(fit.v2x_centre, fit.v2y_centre),
                    return_result=True,
                )
            else:
                lg_result = aberration_tensor(
                    fit, s2_image=(fit.s2x_centre, fit.s2y_centre),
                    source_point=source_point,
                    output_modes=output_modes,
                    w_s=w_s, w_p=w_p,
                    v2_centre=(fit.v2x_centre, fit.v2y_centre),
                )
        except Exception as exc:
            notes.append(
                f"LG tensor unavailable: {type(exc).__name__}: {exc}")
            lg_result = None
            fit = None

    return AberrationSummary(
        seidel_total=seidel_total,
        seidel_per_surface=per_surf,
        efl=efl_v,
        bfl=bfl_v,
        lg_tensor=lg_result,
        fit=fit,
        wavelength=float(wavelength),
        notes=notes,
    )


def format_aberration_summary(summary: AberrationSummary,
                                units: str = 'mm') -> str:
    """Pretty-print an :class:`AberrationSummary` to a multi-line str.

    Useful for quick console output during design iterations.
    """
    scale = {'m': 1.0, 'mm': 1e3, 'um': 1e6}.get(units, 1e3)
    lines = []
    lines.append(f"AberrationSummary @ wavelength={summary.wavelength*1e9:.1f} nm")
    if summary.efl is not None:
        lines.append(f"  EFL = {summary.efl * scale:.4f} {units}")
    if summary.bfl is not None:
        lines.append(f"  BFL = {summary.bfl * scale:.4f} {units}")
    s = summary.seidel_total
    lines.append(
        f"  Seidel (m): "
        f"S1={s[0]:+.3e}  S2={s[1]:+.3e}  S3={s[2]:+.3e}  "
        f"S4={s[3]:+.3e}  S5={s[4]:+.3e}"
    )
    if summary.lg_tensor is not None:
        L = summary.lg_tensor.L
        lines.append(
            f"  LG L_(0,0),(0,0) = "
            f"{abs(L[0, 0]):.3e}  arg={np.angle(L[0, 0]):+.3f} rad"
        )
        if L.shape[0] > 1:
            for io, mode in enumerate(summary.lg_tensor.output_modes[1:],
                                       start=1):
                lines.append(
                    f"     L_{mode},(0,0)  = "
                    f"{abs(L[io, 0]):.3e}  "
                    f"arg={np.angle(L[io, 0]):+.3f} rad"
                )
    if summary.notes:
        lines.append('  Notes:')
        for n in summary.notes:
            lines.append(f'    - {n}')
    return '\n'.join(lines)


__all__ = [
    'AberrationSummary',
    'aberration_summary',
    'format_aberration_summary',
]

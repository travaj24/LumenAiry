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

    # Helper: glass-catalog failures previously got buried in `notes`
    # while Seidel returned zeros -- making a system with an unknown
    # glass look "diffraction-limited".  Bubble them up as warnings.
    def _maybe_warn_glass(exc: Exception) -> None:
        msg = str(exc)
        if 'Glass' in msg and 'not in registry' in msg:
            import warnings
            warnings.warn(
                f"aberration_summary: glass not in registry; analysis "
                f"results will be zero-filled and unreliable.  Details: "
                f"{msg}",
                stacklevel=3,
            )

    # --- Geometric leg: Seidel + ABCD --------------------------------
    surfs = surfaces_from_prescription(prescription)
    try:
        _, efl, bfl, _ = system_abcd(surfs, wavelength)
        efl_v = float(efl) if np.isfinite(efl) else None
        bfl_v = float(bfl) if np.isfinite(bfl) else None
    except Exception as exc:
        notes.append(f"system_abcd failed: {type(exc).__name__}: {exc}")
        _maybe_warn_glass(exc)
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
        _maybe_warn_glass(exc)

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


@dataclass
class CausticDiagnostic:
    """Output of :func:`caustic_diagnostic`.

    Captures where in a prescription the geometric Jacobian
    ``det(d(x_out, y_out) / d(x_in, y_in))`` crosses zero -- those
    are the caustic / focal-region neighbourhoods where the
    plane-wave / paraxial-OPL approximations of the cheaper
    propagators (``apply_real_lens``, ``apply_real_lens_traced``)
    start to alias.  Use this to decide whether the Maslov-method
    propagator is needed for a given design (``apply_real_lens_maslov``)
    or whether to bracket caustic regions in a through-focus scan.

    Attributes
    ----------
    z_samples : (n_z,) array
        Absolute z coordinates at which the Jacobian was evaluated,
        measured from the first refractive surface.
    det_J : (n_z,) array
        Sign of det(d(s_out)/d(s_in)) at each z, evaluated at the
        chief ray via finite differences across a small marginal-ray
        fan.
    caustic_z : list of float
        z coordinates where ``det_J`` flips sign (linearly
        interpolated between samples).  These are the on-axis
        caustic crossings.
    maslov_index : int
        Total number of sign flips from the entrance pupil to the
        last sample plane.  This is the Maslov index that
        ``apply_real_lens_maslov`` uses to add ``-pi/2`` per crossing
        to the geometric phase.
    surface_z : list of float
        z coordinates of the prescription's refractive surfaces,
        for context when plotting.
    chief_ray_height : (n_z,) array
        Chief-ray transverse height at each z sample.  Useful for
        identifying which caustic crossings are on-axis (height -> 0)
        versus off-axis astigmatism.
    """
    z_samples: np.ndarray
    det_J: np.ndarray
    caustic_z: List[float]
    maslov_index: int
    surface_z: List[float]
    chief_ray_height: np.ndarray


def caustic_diagnostic(prescription: Dict[str, Any],
                       wavelength: float,
                       *,
                       chief_ray: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
                       fan_radius: float = 1e-4,
                       n_z_per_gap: int = 32,
                       z_after_last_surface: Optional[float] = None,
                       ) -> CausticDiagnostic:
    """Identify caustic crossings along the chief-ray of a
    prescription.

    Traces a small fan of rays parallel to the chief ray through the
    prescription, sampling at ``n_z_per_gap`` planes between each
    pair of surfaces and (optionally) over a stretch of free-space
    after the last surface.  At each sample plane the local
    Jacobian determinant ``det(d(s_out, t_out)/d(s_in, t_in))`` is
    estimated by finite differences over the fan; sign flips
    indicate caustic crossings.

    Parameters
    ----------
    prescription : dict
        Standard lumenairy prescription (radii, conics, asphere
        coeffs, glass strings, thicknesses).
    wavelength : float
        Vacuum wavelength [m].
    chief_ray : (x, y, L, M)
        Chief-ray launch state at the entrance plane; default is
        on-axis collimated ``(0, 0, 0, 0)``.
    fan_radius : float, default 1e-4 (100 um)
        Half-spacing of the four marginal rays around the chief
        ray.  Should be smaller than the smallest aperture but
        large enough for the FD Jacobian to be numerically stable.
    n_z_per_gap : int, default 32
        Number of sample planes between successive refractive
        surfaces.  Increase for finer caustic localisation.
    z_after_last_surface : float, optional
        Length of the free-space stretch after the last refractive
        surface to sample (defaults to the prescription's final
        thickness, or 100 mm if no thickness is set).

    Returns
    -------
    diagnostic : CausticDiagnostic
        Sample positions, Jacobian determinants, caustic z values,
        Maslov index, and chief-ray heights.

    Notes
    -----
    For an axially-symmetric refractive system with a converging
    chief ray, the canonical caustic is the focal point: ``det_J``
    crosses zero at the geometric focus.  For a system with
    astigmatism the on-axis Jacobian determinant has multiple
    crossings (sagittal and tangential foci); both are reported.

    The diagnostic is geometric only -- it tells you where the
    geometric-OPL approximation breaks down, not how badly.  For a
    quantitative breakdown-severity, propagate through the system
    with both ``apply_real_lens_traced`` and ``apply_real_lens_maslov``
    and compare the on-MS field intensities.
    """
    from ..raytrace import (
        surfaces_from_prescription, _make_bundle, trace,
    )

    surfaces = surfaces_from_prescription(prescription)
    if not surfaces:
        raise ValueError("caustic_diagnostic: prescription has no surfaces")

    # Fan: chief ray + 4 marginals offset by +/- fan_radius in x and y.
    cx, cy, cL, cM = [float(v) for v in chief_ray]
    xs = np.array([cx, cx + fan_radius, cx - fan_radius, cx, cx])
    ys = np.array([cy, cy, cy, cy + fan_radius, cy - fan_radius])
    Ls = np.full(5, cL)
    Ms = np.full(5, cM)

    bundle = _make_bundle(x=xs, y=ys, L=Ls, M=Ms, wavelength=wavelength)

    # Build z-sample list: between each pair of consecutive surfaces,
    # plus optionally after the last surface.
    thicknesses = list(prescription.get('thicknesses', []))
    cumulative_z = [0.0]
    for t in thicknesses:
        cumulative_z.append(cumulative_z[-1] + float(t))
    if z_after_last_surface is None:
        z_after_last_surface = float(thicknesses[-1]) if thicknesses else 0.1
    cumulative_z.append(cumulative_z[-1] + float(z_after_last_surface))

    z_samples_list: List[float] = []
    for k in range(len(cumulative_z) - 1):
        z0 = cumulative_z[k]
        z1 = cumulative_z[k + 1]
        for j in range(n_z_per_gap):
            frac = (j + 1) / n_z_per_gap
            z_samples_list.append(z0 + frac * (z1 - z0))
    z_samples = np.asarray(z_samples_list, dtype=float)

    # Trace the bundle, capturing per-surface ray history so we can
    # interpolate between surfaces by linear ray propagation in air.
    res = trace(bundle, surfaces, wavelength=wavelength,
                output_filter='all')
    history = res.ray_history  # list of RayBundle, one per surface

    # Each history[k] is the bundle just after refraction at surface k.
    # Between surfaces k and k+1 the ray propagates linearly in the
    # medium; the z step from surface-k frame to surface-(k+1) frame
    # is thicknesses[k].

    det_J = np.empty(z_samples.size, dtype=float)
    eig_J_min = np.empty(z_samples.size, dtype=float)
    eig_J_max = np.empty(z_samples.size, dtype=float)
    chief_h = np.empty(z_samples.size, dtype=float)

    n_surfaces = len(surfaces)
    for i, z in enumerate(z_samples):
        # Find which between-surface gap this z falls in.
        gap = None
        for k in range(len(cumulative_z) - 1):
            if cumulative_z[k] <= z <= cumulative_z[k + 1] + 1e-15:
                gap = k
                break
        if gap is None:
            gap = len(cumulative_z) - 2

        # If we're at or past the last surface, propagate from history[-1].
        # Otherwise propagate from history[gap] in the medium between
        # surface gap and gap+1.
        bundle_at_surf = history[min(gap, n_surfaces - 1)]
        z_in_gap = z - cumulative_z[min(gap, n_surfaces - 1)]
        # Linear propagation: x' = x + L * t,  z' = z + N * t.
        # Bundle's z is in the surface-frame, which is z_in_gap when
        # the ray crosses z = z_in_gap relative to the surface vertex.
        L = np.asarray(bundle_at_surf.L)
        M = np.asarray(bundle_at_surf.M)
        N = np.asarray(bundle_at_surf.N)
        N_safe = np.where(np.abs(N) > 1e-30, N, 1e-30)
        # Propagate to the absolute z plane: how much t to advance to
        # reach surface-frame z = z_in_gap?
        # bundle is right after surface, its z=0 in surface-frame.  We
        # want z_in_gap (positive forward).
        t_advance = z_in_gap / N_safe
        x_at_z = np.asarray(bundle_at_surf.x) + L * t_advance
        y_at_z = np.asarray(bundle_at_surf.y) + M * t_advance

        # Chief ray (index 0)
        chief_h[i] = float(np.sqrt(x_at_z[0] ** 2 + y_at_z[0] ** 2))

        # Jacobian via central finite differences from the four
        # marginal rays:
        # (x[1]-x[2])/(2 fan_radius)  is dx_out/dx_in,  etc.
        dxdx = (x_at_z[1] - x_at_z[2]) / (2.0 * fan_radius)
        dxdy = (x_at_z[3] - x_at_z[4]) / (2.0 * fan_radius)
        dydx = (y_at_z[1] - y_at_z[2]) / (2.0 * fan_radius)
        dydy = (y_at_z[3] - y_at_z[4]) / (2.0 * fan_radius)
        det_J[i] = float(dxdx * dydy - dxdy * dydx)
        # Eigenvalues of J = [[dxdx, dxdy], [dydx, dydy]].  Caustic
        # crossings are where an eigenvalue passes through zero.  For
        # axisymmetric systems both eigenvalues cross at the focal
        # point (point caustic, Maslov contribution = 2); for
        # astigmatic systems they cross at different z (sagittal +
        # tangential foci, two separate Maslov increments).
        tr = dxdx + dydy
        det = det_J[i]
        disc = max(0.25 * tr * tr - det, 0.0)
        sqrt_disc = float(np.sqrt(disc))
        eig_J_min[i] = 0.5 * tr - sqrt_disc
        eig_J_max[i] = 0.5 * tr + sqrt_disc

    # Detect zero-crossings of EACH eigenvalue separately.  An axis-
    # symmetric focal point shows up as both eigenvalues crossing at
    # the same z (two Maslov increments, one z); an astigmatic system
    # has the two eigenvalues crossing at separate z (two crossings,
    # two Maslov increments).
    caustic_z: List[float] = []
    maslov_index = 0
    for eigvals in (eig_J_min, eig_J_max):
        for i in range(1, eigvals.size):
            a = eigvals[i - 1]
            b = eigvals[i]
            if a * b < 0:
                t = abs(a) / (abs(a) + abs(b))
                z_cross = z_samples[i - 1] + t * (z_samples[i] - z_samples[i - 1])
                caustic_z.append(float(z_cross))
                maslov_index += 1
    # Deduplicate near-coincident crossings (point caustics) within
    # one sample step so the user sees one z per focal point even
    # though the Maslov index counts both eigenvalues.
    caustic_z.sort()
    if len(caustic_z) > 1:
        sample_step = float(z_samples[1] - z_samples[0])
        merged = [caustic_z[0]]
        for zc in caustic_z[1:]:
            if zc - merged[-1] > 0.5 * sample_step:
                merged.append(zc)
        caustic_z = merged

    return CausticDiagnostic(
        z_samples=z_samples,
        det_J=det_J,
        caustic_z=caustic_z,
        maslov_index=maslov_index,
        surface_z=cumulative_z[:n_surfaces],
        chief_ray_height=chief_h,
    )


def plot_caustic_diagnostic(diag: CausticDiagnostic, *, ax=None,
                              title: Optional[str] = None):
    """Plot caustic-diagnostic output: ``det(J)`` and chief-ray
    height vs z, with caustic crossings + surface positions
    annotated.  Requires matplotlib.

    Parameters
    ----------
    diag : CausticDiagnostic
        Output of :func:`caustic_diagnostic`.
    ax : matplotlib axes, optional
        Existing axes to plot into.  If None, a new figure is
        created.
    title : str, optional
        Plot title.

    Returns
    -------
    fig, axes : matplotlib Figure and a 2-tuple of Axes
        The det(J) plot is on axes[0]; the chief-ray-height plot
        on axes[1] (linked x-axes).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_caustic_diagnostic.")

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    z_mm = diag.z_samples * 1e3

    axes[0].plot(z_mm, diag.det_J, 'b-', linewidth=1.5)
    axes[0].axhline(0, color='gray', linewidth=0.5)
    for zc in diag.caustic_z:
        axes[0].axvline(zc * 1e3, color='red', linewidth=1.0,
                         alpha=0.7)
        axes[0].annotate(f'{zc*1e3:.2f} mm', xy=(zc * 1e3, 0),
                          xytext=(zc * 1e3, 0), color='red',
                          fontsize=8, rotation=90, ha='right',
                          va='bottom')
    for sz in diag.surface_z:
        axes[0].axvline(sz * 1e3, color='black', linewidth=0.5,
                         alpha=0.4, linestyle=':')
    axes[0].set_ylabel('det J')
    axes[0].set_title(title or
        f'Caustic diagnostic: {len(diag.caustic_z)} crossings, '
        f'Maslov index = {diag.maslov_index}')

    axes[1].plot(z_mm, diag.chief_ray_height * 1e6, 'g-', linewidth=1.5)
    for zc in diag.caustic_z:
        axes[1].axvline(zc * 1e3, color='red', linewidth=1.0,
                         alpha=0.7)
    for sz in diag.surface_z:
        axes[1].axvline(sz * 1e3, color='black', linewidth=0.5,
                         alpha=0.4, linestyle=':')
    axes[1].set_xlabel('z [mm]')
    axes[1].set_ylabel('chief-ray height [um]')

    fig.tight_layout()
    return fig, axes


__all__ = [
    'AberrationSummary',
    'aberration_summary',
    'format_aberration_summary',
    'CausticDiagnostic',
    'caustic_diagnostic',
    'plot_caustic_diagnostic',
]

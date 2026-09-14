"""Lens-family LEAF kernels: the grid-versus-aperture bookkeeping every entry
point in the family runs before it touches a field.

WHY THIS MODULE EXISTS.  ``elements/lenses.py`` is two things at once: a body of
shared leaf helpers, and the FACADE that re-exports the whole lens family
(``_lens_real``, ``_lens_traced``, ``lenses_maslov``, ...).  Because the facade
half imports those modules at module scope, any family module that reaches back
into ``lenses`` for a leaf helper closes a module-level import 2-cycle -- and a
2-cycle means the family's import order is load-bearing, so a new module-level
statement in the wrong half can turn a working import into a partially
initialised one.  Audit ``AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11`` finding H6 /
WP-A16 addendum 10 counted four such cycles and enumerated what each back-edge
carries.

This module is the leaf those back-edges can point at instead.  It imports
``numpy`` and ``warnings`` and nothing from ``lumenairy``, so it can never
participate in a cycle.  ``lenses`` re-exports every name here, so
``lumenairy.elements.lenses.check_grid_vs_apertures`` and every existing
``from .lenses import _warn_if_aperture_exceeds_grid`` keep resolving.

WHAT IS HERE.  The prescription-to-semi-diameter census, the public grid check,
the grid recommendation and the warning the entry points emit.  Nothing in it
is physics: it reads a prescription dict and compares lengths.

WHAT IS NOT HERE YET.  ``surface_sag_general`` / ``surface_sag_biconic`` and
their optional-backend plumbing (the numba kernel cache, the lazy CuPy alias)
are the other half of the ``_lens_real <-> lenses`` back-edge and have not
moved; ``docs/lens_configuration.md`` section "Module layout" carries the
remaining plan and the exact edits.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def _collect_semi_diameters(prescription):
    """Return [(label, semi_diameter_m)] for every surface in the
    prescription that has a ``semi_diameter`` set.

    Looks at both ``prescription['elements']`` (Zemax-loaded path,
    where ``semi_diameter`` is the per-surface CLAP) and
    ``prescription['surfaces']`` (builder-style).  The top-level
    ``aperture_diameter`` (system-wide clear aperture) is also
    included if present.  Surfaces without a finite ``semi_diameter``
    are skipped.
    """
    out = []
    seen_labels = set()
    elements = prescription.get('elements')
    if isinstance(elements, list):
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            sd = elem.get('semi_diameter')
            if sd is None or not np.isfinite(float(sd)):
                continue
            surf_num = elem.get('surf_num', '?')
            comment = (elem.get('comment') or elem.get('name') or '').strip()
            label = f"surf {surf_num}"
            if comment:
                label = f"{label} '{comment}'"
            if label not in seen_labels:
                out.append((label, float(sd)))
                seen_labels.add(label)
    surfaces = prescription.get('surfaces')
    if isinstance(surfaces, list):
        for i, surf in enumerate(surfaces):
            if not isinstance(surf, dict):
                continue
            sd = surf.get('semi_diameter')
            if sd is None or not np.isfinite(float(sd)):
                continue
            label = f"surfaces[{i}]"
            if label not in seen_labels:
                out.append((label, float(sd)))
                seen_labels.add(label)
    ap = prescription.get('aperture_diameter')
    if ap is not None and np.isfinite(float(ap)):
        out.append(('system aperture_diameter', 0.5 * float(ap)))
    return out


def check_grid_vs_apertures(
    prescription: Dict[str, Any],
    N: int,
    dx: float,
    *,
    safety_factor: float = 1.0,
) -> List[Tuple[str, float, float, float]]:
    """Identify every prescription surface whose semi-aperture exceeds
    the simulation grid's half-extent.

    Use this as a pre-flight check before a full ASM-through-lens run.
    A surface whose ``semi_diameter`` is larger than ``safety_factor *
    N * dx / 2`` will silently truncate the field's outer rim during
    propagation, dropping any energy the real hardware would have
    transmitted past the grid edge.  This is a sim-infrastructure
    artifact, not a physical clipping by the lens itself, and will
    show up in centroid measurements as a uniform inward bias.

    Parameters
    ----------
    prescription : dict
        lumenairy prescription dict (loaded or builder-built).
    N : int
        Simulation grid size (assumed square).
    dx : float
        Grid pitch [m].
    safety_factor : float, optional
        Margin between the grid semi and the largest surface
        semi-aperture.  Pass 0.95 to flag surfaces whose semi exceeds
        95% of the grid half-extent (recommended for clean Gaussian
        wing containment); pass 1.0 (default) to flag only surfaces
        that exceed the grid outright.

    Returns
    -------
    issues : list of (label, semi_aperture_m, grid_semi_m, gap_m)
        One tuple per offending surface.  Empty if every aperture fits.

    See Also
    --------
    apply_real_lens, apply_real_lens_traced, apply_real_lens_maslov
        These functions automatically run this check on entry and emit
        a UserWarning if any aperture exceeds the grid.
    """
    if N <= 0 or dx <= 0:
        raise ValueError(f"N and dx must be positive, got N={N}, dx={dx}")
    grid_semi = 0.5 * float(N) * float(dx)
    threshold = float(safety_factor) * grid_semi
    issues = []
    for label, sd in _collect_semi_diameters(prescription):
        if sd > threshold:
            issues.append((label, sd, grid_semi, sd - grid_semi))
    return issues


def recommend_grid_for_prescription(
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    source_waist: Optional[float] = None,
    doe_orders_max: Optional[Tuple[float, float]] = None,
    doe_period: Optional[Union[float, Tuple[float, float]]] = None,
    doe_to_destination_distance: Optional[float] = None,
    margin_ratio: float = 1.05,
    samples_per_wavelength: float = 4.0,
    samples_per_source_waist: float = 6.0,
    round_n_to_power_of_two: bool = True,
) -> Dict[str, Any]:
    """Recommend simulation grid parameters ``(N, dx)`` for an ASM run
    through a sequential prescription.

    The recommendation answers two coupled questions before a long
    simulation run:

    * **How wide does the grid need to be?**  Determined by the largest
      ``semi_diameter`` in the prescription, optionally extended for
      DOE diffraction-order spread (corner-order chief rays at the
      destination plane sit at ``m * lambda * z_propagation / period``
      off-axis and must fit inside ``N * dx / 2``).

    * **How fine does the pixel pitch need to be?**  Bounded above by
      the wavelength's effective Nyquist (``lambda /
      samples_per_wavelength``) and, if a source size is supplied, by
      the source-waist resolution (``source_waist /
      samples_per_source_waist``).

    The two requirements together pin ``N`` and ``dx`` (with ``N``
    rounded up to the nearest power of two for FFT efficiency by
    default) such that ``N * dx / 2 >= margin_ratio * required_extent``
    and ``dx <= min(dx_constraints)``.

    Parameters
    ----------
    prescription : dict
        lumenairy prescription dict.  Must include either ``elements``
        with ``semi_diameter`` fields or a top-level ``aperture_diameter``;
        without one of those the function raises ``ValueError``.
    wavelength : float
        Vacuum wavelength [m].  Sets the absolute Nyquist bound on
        ``dx``.
    source_waist : float, optional
        Source 1/e Gaussian waist [m].  When supplied, ``dx`` is also
        bounded above by ``source_waist / samples_per_source_waist`` so
        the source itself is well-resolved on the grid.
    doe_orders_max : (int, int) or (float, float), optional
        Maximum diffraction-order index along (x, y) the simulation
        will need to retain (e.g. ``(5.5, 5.5)`` for a 12 x 12 Dammann
        splitter's diagonal corner).  Must be supplied together with
        ``doe_period`` and ``doe_to_destination_distance`` for the
        DOE-corner-order spread to be added to the required grid
        extent.
    doe_period : float or (float, float), optional
        Grating period [m].  Scalar = isotropic; tuple = (x, y).
    doe_to_destination_distance : float, optional
        Free-space propagation distance from the DOE plane to the
        destination plane that the recommendation is sizing for [m].
    margin_ratio : float, optional
        Multiplicative safety margin on the required grid semi
        (default 1.05 = 5%).
    samples_per_wavelength : float, optional
        Minimum samples per wavelength along each grid axis (default
        4.0).  Nyquist is 2.0; 4.0 leaves comfortable headroom for the
        ASM kernel's exp(i*k*z*sqrt(...)) phase ramp.
    samples_per_source_waist : float, optional
        Minimum samples across one source waist (default 6.0).  Below
        ~4 the Gaussian source is poorly resolved.
    round_n_to_power_of_two : bool, optional
        If True (default), round ``N`` up to the nearest power of two.
        FFT-based propagators run several times faster on power-of-two
        grids than on arbitrary sizes.

    Returns
    -------
    dict
        Recommendation with at least these keys:

          ``N`` -- recommended grid size (int)
          ``dx`` -- recommended pixel pitch [m]
          ``grid_semi_m`` -- ``N * dx / 2``
          ``required_extent_m`` -- before margin
          ``limiting_aperture_m`` -- the surface that set the grid extent
          ``limiting_aperture_label`` -- human-readable label of that surface
          ``doe_extra_extent_m`` -- additional extent demanded by the DOE
          ``dx_constraints`` -- dict of every dx upper bound considered
          ``dx_limiting_constraint`` -- name of the binding constraint
          ``samples_per_wavelength`` -- effective post-rounding value

    Raises
    ------
    ValueError
        If the prescription has no resolvable aperture, or if any of
        ``doe_*`` are partially supplied (must be all-or-none).

    See Also
    --------
    check_grid_vs_apertures
        The post-flight companion that flags whether a chosen ``(N,
        dx)`` actually contains every prescription aperture.
    """
    if wavelength <= 0:
        raise ValueError(f"wavelength must be > 0, got {wavelength}")
    if margin_ratio < 1.0:
        raise ValueError(
            f"margin_ratio must be >= 1.0 (got {margin_ratio}); use 1.0 "
            "for the bare-minimum grid that just contains every aperture")

    # DOE arguments are all-or-none
    doe_args = (doe_orders_max, doe_period, doe_to_destination_distance)
    if any(a is not None for a in doe_args) and not all(
            a is not None for a in doe_args):
        raise ValueError(
            "doe_orders_max, doe_period, and doe_to_destination_distance "
            "must all be supplied together (or all be None)")

    # 1) Find the largest aperture in the prescription
    max_semi = 0.0
    aperture_label = ''
    elements = prescription.get('elements')
    if isinstance(elements, list):
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            sd = elem.get('semi_diameter')
            if sd is None or not np.isfinite(float(sd)):
                continue
            sd = float(sd)
            if sd > max_semi:
                max_semi = sd
                lbl_parts = []
                if elem.get('comment'):
                    lbl_parts.append(str(elem['comment']))
                if elem.get('surf_num') is not None:
                    lbl_parts.append(f"surf {elem['surf_num']}")
                aperture_label = (
                    ' / '.join(lbl_parts) if lbl_parts
                    else 'unknown element')
    surfaces = prescription.get('surfaces')
    if isinstance(surfaces, list):
        for i, surf in enumerate(surfaces):
            if not isinstance(surf, dict):
                continue
            sd = surf.get('semi_diameter')
            if sd is None or not np.isfinite(float(sd)):
                continue
            sd = float(sd)
            if sd > max_semi:
                max_semi = sd
                aperture_label = f"surfaces[{i}]"
    sys_ap = prescription.get('aperture_diameter')
    if sys_ap is not None and np.isfinite(float(sys_ap)):
        if 0.5 * float(sys_ap) > max_semi:
            max_semi = 0.5 * float(sys_ap)
            aperture_label = 'system aperture_diameter'

    if max_semi <= 0.0:
        raise ValueError(
            "prescription has no resolvable aperture: pass a prescription "
            "whose elements / surfaces carry 'semi_diameter' or whose "
            "top-level 'aperture_diameter' is finite")

    # 2) DOE-order extent extension (if supplied)
    doe_extra = 0.0
    if doe_orders_max is not None:
        m_x, m_y = doe_orders_max
        if isinstance(doe_period, (tuple, list)):
            period_x, period_y = (float(doe_period[0]),
                                   float(doe_period[1]))
        else:
            period_x = period_y = float(doe_period)
        if period_x <= 0 or period_y <= 0:
            raise ValueError(f"doe_period must be > 0, got {doe_period}")
        z_doe = float(doe_to_destination_distance)
        # Diagonal corner order's chief-ray displacement at the
        # destination plane (paraxial direction-cosine approximation).
        theta_x = float(m_x) * wavelength / period_x
        theta_y = float(m_y) * wavelength / period_y
        doe_extra = abs(z_doe) * float(np.hypot(theta_x, theta_y))

    required_extent = max_semi + doe_extra
    required_grid_semi = margin_ratio * required_extent

    # 3) dx upper bounds
    dx_constraints = {
        'wavelength_nyquist': float(wavelength) / float(samples_per_wavelength),
    }
    if source_waist is not None and float(source_waist) > 0:
        dx_constraints['source_waist'] = (
            float(source_waist) / float(samples_per_source_waist))
    dx_max = min(dx_constraints.values())
    dx_limiting = min(dx_constraints, key=dx_constraints.get)

    # 4) Pick (N, dx)
    n_min = int(np.ceil(2.0 * required_grid_semi / dx_max))
    if round_n_to_power_of_two:
        n = int(2 ** int(np.ceil(np.log2(max(n_min, 1)))))
    else:
        n = int(n_min)
    # Recompute dx to land grid_semi exactly at required_grid_semi.
    dx = 2.0 * required_grid_semi / n

    return {
        'N': n,
        'dx': float(dx),
        'grid_semi_m': float(n * dx / 2.0),
        'required_extent_m': float(required_extent),
        'limiting_aperture_m': float(max_semi),
        'limiting_aperture_label': aperture_label,
        'doe_extra_extent_m': float(doe_extra),
        'dx_constraints': dx_constraints,
        'dx_limiting_constraint': dx_limiting,
        'samples_per_wavelength': float(wavelength) / float(dx),
        'margin_ratio': float(margin_ratio),
    }


def _warn_if_aperture_exceeds_grid(prescription, N, dx, *,
                                    source='apply_real_lens',
                                    safety_factor=1.0,
                                    stacklevel=3,
                                    N_y=None, dy=None):
    """Emit a UserWarning if any prescription aperture exceeds the
    simulation grid.  Called at the top of ``apply_real_lens``,
    ``apply_real_lens_traced``, and ``apply_real_lens_maslov``.

    ``N``/``dx`` describe the x axis.  ``N_y``/``dy`` describe the y axis on an
    ANAMORPHIC grid (non-square ``N``, or ``dy != dx``); both default to the x
    values, so a square grid behaves exactly as before.  The check is made
    against the SMALLER of the two semi-extents, because that is the axis that
    truncates first -- passing ``shape[0]`` (i.e. Ny) together with ``dx``, as
    the analytic model used to, describes a semi-extent that exists on neither
    axis.

    Python's default warning filter dedups by ``(module, lineno)`` so
    repeated calls from the same site only warn once.
    """
    _ny = N if N_y is None else N_y
    _dy = dx if dy is None else dy
    semi_x = 0.5 * N * dx
    semi_y = 0.5 * _ny * _dy
    if semi_y < semi_x:
        N, dx = _ny, _dy
    try:
        issues = check_grid_vs_apertures(
            prescription, N, dx, safety_factor=safety_factor)
    except (KeyError, ValueError, TypeError, AttributeError):
        # Aperture check is best-effort; a malformed prescription
        # shouldn't block the warning path entirely (the
        # propagator's own validators will raise downstream).
        return
    if not issues:
        return
    grid_semi = 0.5 * N * dx
    issues_sorted = sorted(issues, key=lambda r: -r[1])
    biggest_label, biggest_sd, _, biggest_gap = issues_sorted[0]
    body = ", ".join(
        f"{lab}={sd*1e3:.2f}mm"
        for lab, sd, _, _ in issues_sorted[:5]
    )
    if len(issues_sorted) > 5:
        body = body + f", ... (+{len(issues_sorted)-5} more)"
    msg = (
        f"{source}: {len(issues)} prescription aperture(s) exceed the "
        f"simulation grid (N={N}, dx={dx*1e6:.3f} um, "
        f"semi={grid_semi*1e3:.3f} mm). "
        f"Largest is {biggest_label} with semi_diameter="
        f"{biggest_sd*1e3:.3f} mm "
        f"({biggest_gap*1e3:+.3f} mm beyond the grid); the field will "
        f"be truncated at the grid edge during propagation, "
        f"silently dropping energy the real lens would have "
        f"transmitted. Consider increasing N or dx so "
        f"N*dx/2 >= max(semi_diameter). "
        f"Affected surfaces: {body}."
    )
    warnings.warn(msg, UserWarning, stacklevel=stacklevel)

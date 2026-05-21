"""
Prescription builders + Thorlabs catalog.

Factory helpers that construct LumenAiry lens prescription dicts
from scalar geometry (radii, thicknesses, glass names).  All
prescriptions use glass name strings rather than numeric refractive
indices so they remain wavelength-independent; indices are resolved
at runtime by the propagation engine.

v5.1.0 split (Agent F): extracted from ``lumenairy/io/prescriptions.py``
without any logic change.  Public API is unchanged: every name in this
module is re-exported through ``lumenairy.io.prescriptions`` and
``lumenairy``.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# ============================================================================
# Prescription builders
# ============================================================================

def make_singlet(R1: float, R2: float, d: float, glass: str,
                 aperture: float = 25.4e-3,
                 name: Optional[str] = None) -> Dict[str, Any]:
    """Build a lens prescription dict for a singlet lens.

    Parameters
    ----------
    R1 : float
        Front surface radius of curvature [m].  Use ``float('inf')`` for flat.
    R2 : float
        Back surface radius of curvature [m].  Use ``float('inf')`` for flat.
    d : float
        Center thickness [m].
    glass : str
        Glass name from ``GLASS_REGISTRY`` (e.g. ``'N-BK7'``).
    aperture : float, optional
        Clear aperture diameter [m].  Default 25.4 mm.
    name : str, optional
        Human-readable label.

    Returns
    -------
    prescription : dict
    """
    return {
        'name': name or f'Singlet ({glass})',
        'aperture_diameter': aperture,
        'surfaces': [
            {'radius': R1, 'conic': 0.0, 'aspheric_coeffs': None,
             'radius_y': None, 'conic_y': None,
             'aspheric_coeffs_y': None,
             'glass_before': 'air', 'glass_after': glass},
            {'radius': R2, 'conic': 0.0, 'aspheric_coeffs': None,
             'radius_y': None, 'conic_y': None,
             'aspheric_coeffs_y': None,
             'glass_before': glass, 'glass_after': 'air'},
        ],
        'thicknesses': [d],
    }


def make_cylindrical(R_focus: float, d: float, glass: str,
                     axis: str = 'x', aperture: float = 25.4e-3,
                     name: Optional[str] = None) -> Dict[str, Any]:
    """Build a cylindrical-lens prescription (flat on one face, curved
    cylindrical on the other).

    A cylindrical lens focuses in one axis only.  Use ``axis='x'`` for
    a lens that focuses along the x-direction (cross-section in x has
    radius ``R_focus``, cross-section in y is flat).  For focusing in
    y use ``axis='y'``.

    Parameters
    ----------
    R_focus : float
        Radius of curvature of the focusing axis [m]; positive for
        converging.
    d : float
        Center thickness [m].
    glass : str
        Glass name.
    axis : ``'x'`` or ``'y'``
        Which axis has curvature.  The other axis is flat.
    aperture : float
        Clear aperture diameter [m].
    name : str, optional
    """
    if axis == 'x':
        R_x, R_y = R_focus, float('inf')
    elif axis == 'y':
        R_x, R_y = float('inf'), R_focus
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    return {
        'name': name or f'CylindricalLens-{axis}({glass})',
        'aperture_diameter': aperture,
        'surfaces': [
            {'radius': R_x, 'radius_y': R_y,
             'conic': 0.0, 'conic_y': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': glass},
            {'radius': float('inf'), 'radius_y': float('inf'),
             'conic': 0.0, 'conic_y': 0.0, 'aspheric_coeffs': None,
             'glass_before': glass, 'glass_after': 'air'},
        ],
        'thicknesses': [d],
    }


def make_biconic(R1_x: float, R1_y: float, R2_x: float, R2_y: float,
                 d: float, glass: str,
                 conic1_x: float = 0.0, conic1_y: float = 0.0,
                 conic2_x: float = 0.0, conic2_y: float = 0.0,
                 aperture: float = 25.4e-3,
                 name: Optional[str] = None) -> Dict[str, Any]:
    """Build a biconic (anamorphic) singlet prescription.

    Each surface has independent x- and y-axis radii and conics -- the
    sag is
    ``z(x,y) = C_x x² / (1 + sqrt(1 - (1+K_x) C_x² x²))
             + C_y y² / (1 + sqrt(1 - (1+K_y) C_y² y²))``
    with ``C = 1/R``.  Covers cylindrical, toroidal, and freeform
    biconic elements used in anamorphic imaging and beam shaping.

    Pass ``inf`` for any radius to make that axis flat.  If ``R1_y``
    (or ``R2_y``) equals the corresponding ``R1_x`` (``R2_x``) and the
    conics match, the surface reduces to rotationally-symmetric --
    in that case use :func:`make_singlet` instead for efficiency.

    Parameters
    ----------
    R1_x, R1_y : float
        Front-surface x- and y-axis radii [m].
    R2_x, R2_y : float
        Back-surface x- and y-axis radii [m].
    d : float
        Center thickness [m].
    glass : str
    conic1_x, conic1_y, conic2_x, conic2_y : float
        Per-axis conic constants (0 = spherical in that axis).
    aperture : float
    name : str, optional
    """
    return {
        'name': name or f'Biconic({glass})',
        'aperture_diameter': aperture,
        'surfaces': [
            {'radius': R1_x, 'radius_y': R1_y,
             'conic': conic1_x, 'conic_y': conic1_y,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': glass},
            {'radius': R2_x, 'radius_y': R2_y,
             'conic': conic2_x, 'conic_y': conic2_y,
             'aspheric_coeffs': None,
             'glass_before': glass, 'glass_after': 'air'},
        ],
        'thicknesses': [d],
    }


def make_doublet(R1: float, R2: float, R3: float,
                 d1: float, d2: float,
                 glass1: str, glass2: str,
                 aperture: float = 25.4e-3,
                 name: Optional[str] = None) -> Dict[str, Any]:
    """Build a lens prescription dict for a cemented achromatic doublet.

    Parameters
    ----------
    R1, R2, R3 : float
        Radii of curvature [m] for the three surfaces.
    d1, d2 : float
        Center thicknesses [m] of the two elements.
    glass1, glass2 : str
        Glass names from ``GLASS_REGISTRY`` for elements 1 and 2.
    aperture : float, optional
        Clear aperture diameter [m].  Default 25.4 mm.
    name : str, optional
        Human-readable label.

    Returns
    -------
    prescription : dict
    """
    return {
        'name': name or f'Doublet ({glass1}/{glass2})',
        'aperture_diameter': aperture,
        'surfaces': [
            {'radius': R1, 'conic': 0.0, 'aspheric_coeffs': None,
             'radius_y': None, 'conic_y': None,
             'aspheric_coeffs_y': None,
             'glass_before': 'air', 'glass_after': glass1},
            {'radius': R2, 'conic': 0.0, 'aspheric_coeffs': None,
             'radius_y': None, 'conic_y': None,
             'aspheric_coeffs_y': None,
             'glass_before': glass1, 'glass_after': glass2},
            {'radius': R3, 'conic': 0.0, 'aspheric_coeffs': None,
             'radius_y': None, 'conic_y': None,
             'aspheric_coeffs_y': None,
             'glass_before': glass2, 'glass_after': 'air'},
        ],
        'thicknesses': [d1, d2],
    }


def make_off_axis_parabola(
        focal_length: float,
        off_axis_angle: float,
        clear_aperture: float,
        *,
        glass: str = '__MIRROR__',
        vertex_radius: Optional[float] = None,
        name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a prescription dict for an off-axis parabola (OAP).

    **Intended consumer: the traced raytrace path**
    (:func:`apply_real_lens_traced`,
    :func:`apply_real_lens_traced_jax`, and the bare
    :func:`lumenairy.raytrace.trace` pipeline with a Surface whose
    ``tilt_x_deg`` / ``decenter_x_m`` fields have been populated from
    the prescription's ``tilt`` / ``decenter`` keys).  The 3-tuple
    ``tilt`` written by this factory is the **full surface-normal
    angle** about the local x-axis; it is NOT a small-angle linear
    surface-ramp.  See the *Warnings* section below for the
    consequences of routing this prescription through
    :func:`apply_real_lens` (the paraxial-only entry).

    Parameters
    ----------
    focal_length : float
        Parent-parabola effective focal length [m].  Must be positive
        and finite.
    off_axis_angle : float
        Surface-normal off-axis angle ``alpha`` [rad] -- the angle
        between the parent optical axis and the surface normal at the
        OAP centre.  The chief-ray fold angle (in -> out) is
        ``2 * alpha``, so the popular "90-deg OAP" geometry
        corresponds to ``off_axis_angle = pi / 4``.  Must lie in the
        open interval ``(0, pi/2)``.

        (Pre-v4.15.1 the docstring described this argument as the
        chief-ray fold angle, but the surrounding formulas already
        assumed the surface-normal convention; v4.15.1 reconciles
        the docstring with the audited geometric derivation below
        and with the chief-ray geometric test ``decenter =
        2*f*tan(alpha)`` at ``alpha = pi/4`` giving ``2 f`` rather
        than the divergent ``f tan(pi/2)``.)
    clear_aperture : float
        Clear aperture diameter at the OAP face [m].  Must be positive
        and finite.
    glass : str, default ``'__MIRROR__'``
        Glass / material specifier.  The default ``'__MIRROR__'``
        sentinel marks the surface as reflective (consistent with the
        ``is_mirror`` convention elsewhere in this module).  Set to a
        glass name (e.g. ``'N-BK7'``) for a refractive parabolic
        element instead.
    vertex_radius : float, optional
        Override the geometrically-derived vertex radius
        ``2 * focal_length``.  Defaults to ``None`` (use the geometric
        relation).  Provided for advanced use cases where the parent
        parabola is specified by some other prescription convention.
        Must be ``None`` or a finite positive float; zero / negative
        / non-finite values raise ``ValueError`` (P3-F1-3 closure --
        the pre-v4.15.1 factory accepted ``vertex_radius=0`` and
        ``vertex_radius=-1`` silently, producing a flat or
        oppositely-curved surface).
    name : str, optional
        Human-readable label.  Defaults to ``"OAP (f={focal_length},
        theta={off_axis_angle})"``.

    Returns
    -------
    prescription : dict
        Prescription dict with a single paraboloidal surface
        (``conic = -1``), vertex radius ``R = 2 * focal_length``, the
        chief-ray decenter, the surface-normal tilt, and the clear-
        aperture diameter.  Intended for the traced raytrace path
        (see *Warnings*).

    Notes
    -----
    **Geometry derivation (v4.15.1).**  An OAP is a sub-aperture of a
    parent paraboloid whose sag is ``z(r) = r^2 / (4 f)`` with vertex
    at the origin and focus at ``(0, 0, f)``.  The surface normal at
    a point of radial offset ``r`` is tilted by angle ``alpha``
    relative to the optical axis, where

        ``tan(alpha) = dz/dr = r / (2 f)``.

    Solving for ``r`` in terms of the surface-normal angle
    ``alpha = off_axis_angle`` gives the radial offset of the OAP
    centre on the parent paraboloid:

        ``h_decenter = 2 * f * tan(alpha)``.

    By the law of reflection, a collimated chief ray entering
    parallel to ``z`` at offset ``h`` reflects through the parent
    focus ``(0, 0, f)`` at chief-ray fold angle ``theta = 2 * alpha``
    (incoming -> outgoing).  The 3-tuple ``tilt`` field on the
    returned surface dict carries ``alpha`` so the traced raytrace
    leg can rotate the local surface frame correctly.

    Vertex radius of the parent paraboloid (sphere best-fit at the
    apex of ``z = r^2 / (4 f)``) is ``R = 2 f`` and parent conic
    constant is ``k = -1``; these complete the surface description.

    **Convention check vs catalogue OAPs.**  Catalogue OAPs
    (Thorlabs MPD-series, Edmund Optics) typically advertise the
    chief-ray fold angle (``2 * alpha``) as the "off-axis angle".  A
    "90-deg OAP" is the popular folded-telescope geometry with
    chief-ray fold of 90 deg; pass ``off_axis_angle = pi / 4`` to
    this factory (the surface-normal angle is half the chief-ray
    fold).  Decenter for that case is ``2 f tan(pi/4) = 2 f`` --
    finite and physical, in contrast to the pre-v4.15.1
    ``f tan(theta) = f tan(pi/2) = inf``.

    Warnings
    --------
    The 3-tuple ``tilt`` field on the returned surface dict (a
    full-angle surface-normal rotation about the local x-axis, *not*
    a small-angle linear ramp) is consumed correctly by the traced
    raytrace path (which applies a proper coordinate-frame
    rotation).  It is, however, misinterpreted by the paraxial
    :func:`apply_real_lens` entry point, which reads
    ``surf.get('tilt') or (0.0, 0.0)`` and adds
    ``tilt[0] * x + tilt[1] * y`` as a linear surface-sag ramp -- a
    paraxial small-angle approximation that diverges badly at OAP-
    scale fold angles (error ~2 % at 15 deg, ~27 % at 45 deg,
    divergent at 90 deg).  Routing an OAP prescription through
    :func:`apply_real_lens` therefore gives a silently-wrong OPD;
    use :func:`apply_real_lens_traced` (or
    :func:`apply_real_lens_traced_jax`) for an OAP, or fold the
    design at the mirror via :func:`split_prescription_at_mirrors`
    and propagate the post-OAP leg with :func:`apply_mirror` plus
    the appropriate coordinate-break surface.

    Examples
    --------
    >>> import numpy as np
    >>> # 90-deg-fold OAP: surface-normal angle is pi/4
    >>> oap = make_off_axis_parabola(focal_length=152.4e-3,
    ...                              off_axis_angle=np.pi / 4,
    ...                              clear_aperture=50.8e-3)
    >>> oap['surfaces'][0]['conic']
    -1.0
    >>> abs(oap['surfaces'][0]['decenter'][0] - 2.0 * 152.4e-3) < 1e-12
    True
    """
    import math
    # Input validation (per task spec: positive f, 0 < theta < pi/2,
    # positive clear aperture).
    if not (np.isfinite(focal_length) and focal_length > 0):
        raise ValueError(
            f"make_off_axis_parabola: focal_length must be a positive "
            f"finite value [m]; got focal_length={focal_length!r}.")
    if not (np.isfinite(off_axis_angle)
            and 0.0 < off_axis_angle < math.pi / 2):
        raise ValueError(
            f"make_off_axis_parabola: off_axis_angle must lie in the "
            f"open interval (0, pi/2) [rad]; got off_axis_angle="
            f"{off_axis_angle!r}.")
    if not (np.isfinite(clear_aperture) and clear_aperture > 0):
        raise ValueError(
            f"make_off_axis_parabola: clear_aperture must be a "
            f"positive finite diameter [m]; got clear_aperture="
            f"{clear_aperture!r}.")

    # Parent-parabola vertex radius: geometric relation R = 2 f
    # unless the caller overrides via ``vertex_radius``.  v4.15.1
    # (P3-F1-3): reject zero / negative / non-finite overrides so a
    # mistyped value can't silently produce a flat or oppositely-
    # curved surface.
    if vertex_radius is None:
        R = 2.0 * focal_length
    else:
        if not (np.isfinite(vertex_radius) and float(vertex_radius) > 0):
            raise ValueError(
                f"make_off_axis_parabola: vertex_radius must be None "
                f"or a positive finite value [m]; got "
                f"vertex_radius={vertex_radius!r}.")
        R = float(vertex_radius)

    # Off-axis decenter: parent-axis offset to the OAP centre.
    #
    # Derivation (v4.15.1 fix for P0-NEW-1 Bug 2 / Part 3 F1):
    # The parent paraboloid is ``z = r^2 / (4 f)``, so the surface-
    # normal direction at radial offset ``r`` makes angle ``alpha``
    # with the optical axis, where ``tan(alpha) = r / (2 f)``.
    # Solving for ``r`` in terms of the surface-normal angle gives
    #
    #     h = 2 f tan(alpha)
    #
    # which is the chief-ray launch radius for a collimated bundle
    # that reflects through the parent focus ``(0, 0, f)`` at fold
    # angle ``2 * alpha``.  Pre-v4.15.1 the code used
    # ``h = f tan(alpha)`` -- off by a factor of two and divergent
    # at alpha approaching pi/2.  Worst case: a 90-deg-fold OAP
    # (alpha = pi/4) wants h = 2 f, not the pre-fix h = f tan(pi/4)
    # = f.
    h_decenter = 2.0 * focal_length * math.tan(off_axis_angle)

    is_mirror = (glass == '__MIRROR__')

    surface: Dict[str, Any] = {
        'radius': R,
        'conic': -1.0,
        'aspheric_coeffs': None,
        'radius_y': None,
        'conic_y': None,
        'aspheric_coeffs_y': None,
        'glass_before': 'air',
        'glass_after': 'air' if is_mirror else glass,
        'decenter': (h_decenter, 0.0),
        # The 3-tuple convention is the chief-ray fold about the
        # local x-axis, consumed correctly by the traced raytrace
        # path.  See *Warnings* in the docstring for the paraxial
        # apply_real_lens failure mode (a 3-tuple of full angles is
        # interpreted there as a 2-tuple of small-angle linear
        # surface-sag ramps).
        'tilt': (off_axis_angle, 0.0, 0.0),
        'clear_aperture': clear_aperture,
        'is_mirror': is_mirror,
    }

    return {
        'name': name or (
            f"OAP (f={focal_length:.4g} m, "
            f"theta={off_axis_angle:.4g} rad)"
        ),
        'aperture_diameter': clear_aperture,
        'surfaces': [surface],
        'thicknesses': [],
    }


# ============================================================================
# Thorlabs catalog lens presets
# ============================================================================
# Surface data from Thorlabs Zemax files.  All dimensions in meters.
# Sign convention: positive R = center of curvature to the right.

THORLABS_CATALOG = {
    # --- Plano-convex singlets (C-coated = 1050-1700 nm) ---
    'LA1050-C': {  # f=100mm, N-BK7, 1" dia
        'type': 'singlet',
        'R1': 51.5e-3, 'R2': float('inf'),
        'd': 4.1e-3, 'glass': 'N-BK7', 'aperture': 25.4e-3,
    },
    'LA1509-C': {  # f=200mm, N-BK7, 1" dia (curved side first for collimation)
        'type': 'singlet',
        'R1': 103.29e-3, 'R2': float('inf'),
        'd': 3.6e-3, 'glass': 'N-BK7', 'aperture': 25.4e-3,
    },
    'LA1301-C': {  # f=250mm, N-BK7, 1" dia
        'type': 'singlet',
        'R1': 129.2e-3, 'R2': float('inf'),
        'd': 3.4e-3, 'glass': 'N-BK7', 'aperture': 25.4e-3,
    },
    # --- Achromatic doublets (C-coated) ---
    'AC254-050-C': {  # f=50mm, 1" dia
        'type': 'doublet',
        'R1': 33.3e-3, 'R2': -24.1e-3, 'R3': -95.3e-3,
        'd1': 9.0e-3, 'd2': 3.0e-3,
        'glass1': 'N-BAF10', 'glass2': 'N-SF6HT', 'aperture': 25.4e-3,
    },
    'AC254-200-C': {  # f=200mm, 1" dia
        'type': 'doublet',
        'R1': 110.1e-3, 'R2': -80.6e-3, 'R3': -277.5e-3,
        'd1': 4.0e-3, 'd2': 2.0e-3,
        'glass1': 'N-BAF10', 'glass2': 'N-SF6HT', 'aperture': 25.4e-3,
    },
    'AC254-100-C': {  # f=100mm, 1" dia
        'type': 'doublet',
        'R1': 62.8e-3, 'R2': -46.5e-3, 'R3': -184.5e-3,
        'd1': 6.0e-3, 'd2': 2.5e-3,
        'glass1': 'N-BAF10', 'glass2': 'N-SF6HT', 'aperture': 25.4e-3,
    },
}


def thorlabs_lens(part_number: str) -> Dict[str, Any]:
    """Return a lens prescription dict for a Thorlabs catalog lens.

    The prescription uses glass name strings and is wavelength-independent.
    Refractive indices are resolved at runtime by ``apply_real_lens``.

    Parameters
    ----------
    part_number : str
        Thorlabs part number (e.g. ``'AC254-200-C'``, ``'LA1050-C'``).

    Returns
    -------
    prescription : dict
        Ready to pass to :func:`apply_real_lens`.
    """
    if part_number not in THORLABS_CATALOG:
        raise ValueError(f"Unknown part '{part_number}'. "
                         f"Available: {list(THORLABS_CATALOG.keys())}")

    entry = THORLABS_CATALOG[part_number]

    if entry['type'] == 'singlet':
        return make_singlet(
            R1=entry['R1'], R2=entry['R2'], d=entry['d'],
            glass=entry['glass'],
            aperture=entry['aperture'], name=part_number)

    elif entry['type'] == 'doublet':
        return make_doublet(
            R1=entry['R1'], R2=entry['R2'], R3=entry['R3'],
            d1=entry['d1'], d2=entry['d2'],
            glass1=entry['glass1'], glass2=entry['glass2'],
            aperture=entry['aperture'], name=part_number)

    else:
        raise ValueError(f"Unknown lens type: {entry['type']}")

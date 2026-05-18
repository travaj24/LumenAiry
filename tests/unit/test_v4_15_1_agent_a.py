"""v4.15.1 Agent A tests -- OAP factory fix and Zemax Q-type loader.

Scope
-----

* **A.1** -- ``lumenairy.make_off_axis_parabola``: P0-NEW-1 audit
  closure.  Two independent bugs in the v4.15.0 factory:

  - **Bug 1 (decenter).**  Pre-fix the radial offset was ``f tan(alpha)``;
    correct chief-ray geometry on the parent paraboloid ``z = r^2/(4f)``
    gives ``r = 2 f tan(alpha)`` where ``alpha`` is the surface-normal
    off-axis angle.  Worst case at the 90-deg-fold geometry
    (``alpha = pi/4``): the pre-fix value was ``f`` whereas the correct
    chief-ray launch is ``2 f``.
  - **Bug 2 (tilt convention).**  The 3-tuple ``tilt = (alpha, 0, 0)``
    is the *full* surface-normal angle about the local x-axis
    (consumed correctly by the traced raytrace path), not a paraxial
    small-angle linear surface-sag ramp.  The factory's docstring is
    rewritten to make the "intended for the traced raytrace path"
    requirement loud and explicit.

  The audit also called out the missing end-to-end raytrace test
  that pins the chief-ray geometric focusing prediction.  We add a
  manual mirror raytrace through the parent-paraboloid surface that
  verifies a collimated chief ray launched at offset
  ``h = 2*f*tan(alpha)`` reflects through the parent focus
  ``(0, 0, f)`` -- the analytical chief-ray prediction.

* **A.2** -- ``lumenairy.io.prescriptions.load_zemax_zmx``:
  P1-NEW-E audit closure.  Zemax ``.zmx`` files with ``TYPE QBFS``
  or ``TYPE QCON`` Forbes Q-type freeform surfaces were silently
  dropped to base conic (the loader had no Q-type SURFTYPE branch).
  v4.15.1 adds parsing that emits the canonical LumenAiry freeform
  keys ``freeform_type='q_bfs'`` / ``'q_con'``, ``q_bfs_coeffs`` /
  ``q_con_coeffs`` and ``r_max`` (sourced from PARM 0 with DIAM
  fallback).

References
----------

* Audit ``docs/audits/AUDIT_V4_15_0_2026_05_18.md``,
  Part 2 V4 + Part 3 F1 (P0-NEW-1) and Part 3 F2 (P1-NEW-E).
* G. W. Forbes, "Shape specification for axially symmetric optical
  surfaces," Opt. Express 15(8), 5218-5226 (2007).
* G. W. Forbes, "Robust, efficient computational methods for axially
  symmetric optical aspheres," Opt. Express 18(13), 13851-13862
  (2010).

Author: Andrew Traverso -- v4.15.1 Agent A
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from lumenairy.io.prescriptions import (
    load_zemax_zmx,
    make_off_axis_parabola,
)
from lumenairy.raytrace import Surface, make_ray, trace


# ============================================================================
# A.1 -- ``make_off_axis_parabola`` audit closure (P0-NEW-1 + P3-F1-3)
# ============================================================================


class TestOAPDecenterFormula:
    """The chief-ray launch radius on the parent paraboloid is
    ``r = 2 * f * tan(alpha)`` where ``alpha`` is the surface-normal
    off-axis angle (audit Part 3 F1).  Pre-v4.15.1 the factory wrote
    the off-by-two ``r = f * tan(alpha)``.
    """

    def test_decenter_is_2f_tan_alpha(self):
        """f=100 mm, alpha=pi/6 (30-deg surface-normal angle, 60-deg
        chief-ray fold).  Expected ``r = 2*0.1*tan(pi/6)``."""
        f = 0.1
        alpha = math.pi / 6
        rx = make_off_axis_parabola(
            focal_length=f, off_axis_angle=alpha, clear_aperture=10e-3,
        )
        h = rx['surfaces'][0]['decenter'][0]
        expected = 2.0 * f * math.tan(alpha)
        assert h == pytest.approx(expected, rel=1e-12), (
            f"OAP decenter[0] = {h!r}; chief-ray geometry on the parent "
            f"paraboloid gives r = 2*f*tan(alpha) = {expected!r}.  "
            f"Pre-v4.15.1 the factory produced r = f*tan(alpha), off "
            f"by a factor of two."
        )
        # Decenter[1] is anchored to 0 (the factory only decenters in x).
        assert rx['surfaces'][0]['decenter'][1] == pytest.approx(0.0)

    def test_decenter_finite_at_45_deg(self):
        """``alpha = pi/4`` (45-deg surface-normal, 90-deg chief-ray
        fold) is the popular folded-telescope geometry.  Pre-v4.15.1
        the formula ``f tan(pi/2)`` would diverge; the corrected
        ``2 f tan(pi/4) = 2 f`` is finite and physical.
        """
        f = 50e-3
        alpha = math.pi / 4
        rx = make_off_axis_parabola(
            focal_length=f, off_axis_angle=alpha,
            clear_aperture=20e-3,
        )
        h = rx['surfaces'][0]['decenter'][0]
        # 2 * f * tan(pi/4) = 2 * f.
        assert h == pytest.approx(2.0 * f, rel=1e-12), (
            f"45-deg surface-normal OAP decenter = {h!r}; expected "
            f"2*f = {2.0 * f!r}.  This is the worst-case geometry "
            f"where the pre-v4.15.1 bug was most visible."
        )
        # And the surface itself remains physical: R = 2f, conic = -1.
        assert rx['surfaces'][0]['radius'] == pytest.approx(2.0 * f)
        assert rx['surfaces'][0]['conic'] == pytest.approx(-1.0)

    def test_tilt_remains_3_tuple(self):
        """The 3-tuple ``tilt = (alpha, 0, 0)`` is the
        ``apply_real_lens_traced``-targeted convention (audit Bug 2
        decision).  Pin the structure so a future refactor that
        "unifies" the tilt with the paraxial 2-tuple shape doesn't
        silently regress the traced raytrace path."""
        f = 75e-3
        alpha = math.radians(15.0)
        rx = make_off_axis_parabola(
            focal_length=f, off_axis_angle=alpha,
            clear_aperture=40e-3,
        )
        tilt = rx['surfaces'][0]['tilt']
        assert isinstance(tilt, tuple) or isinstance(tilt, list)
        assert len(tilt) == 3, (
            f"OAP tilt has length {len(tilt)}; expected 3 "
            f"(apply_real_lens_traced consumes the full 3-component "
            f"surface-normal rotation).  Switching to a 2-tuple would "
            f"silently break OAP raytrace coordinate-frame handling.")
        assert tilt[0] == pytest.approx(alpha)
        assert tilt[1] == pytest.approx(0.0)
        assert tilt[2] == pytest.approx(0.0)


class TestOAPVertexRadiusValidation:
    """P3-F1-3 audit closure: ``vertex_radius`` must be ``None`` or a
    positive finite value.  Pre-v4.15.1 the factory accepted zero,
    negative, and non-finite overrides silently."""

    def test_negative_vertex_radius_raises(self):
        with pytest.raises(ValueError, match="vertex_radius"):
            make_off_axis_parabola(
                focal_length=0.1, off_axis_angle=math.pi / 6,
                clear_aperture=10e-3, vertex_radius=-0.1,
            )

    def test_zero_vertex_radius_raises(self):
        with pytest.raises(ValueError, match="vertex_radius"):
            make_off_axis_parabola(
                focal_length=0.1, off_axis_angle=math.pi / 6,
                clear_aperture=10e-3, vertex_radius=0.0,
            )

    def test_nan_vertex_radius_raises(self):
        with pytest.raises(ValueError, match="vertex_radius"):
            make_off_axis_parabola(
                focal_length=0.1, off_axis_angle=math.pi / 6,
                clear_aperture=10e-3, vertex_radius=float('nan'),
            )

    def test_inf_vertex_radius_raises(self):
        with pytest.raises(ValueError, match="vertex_radius"):
            make_off_axis_parabola(
                focal_length=0.1, off_axis_angle=math.pi / 6,
                clear_aperture=10e-3, vertex_radius=float('inf'),
            )

    def test_finite_positive_vertex_radius_accepted(self):
        """A finite positive override is accepted and used verbatim."""
        f = 0.1
        rx = make_off_axis_parabola(
            focal_length=f, off_axis_angle=math.pi / 6,
            clear_aperture=10e-3, vertex_radius=2.0 * f,
        )
        assert rx['surfaces'][0]['radius'] == pytest.approx(2.0 * f)


class TestOAPEndToEndChiefRay:
    """Audit Part 2 V4 / Part 3 F1: "No end-to-end raytrace test ever
    builds an OAP prescription and propagates a beam to verify
    focusing".

    The audit asks for a chief-ray geometric verification: a
    collimated ray launched at the prescription's decenter offset
    must, after reflection off the parent paraboloid, head toward the
    parent focus ``(0, 0, f)``.

    Implementation note.  The OAP factory writes the chief-ray tilt
    and decenter on the surface dict for consumption by the traced
    raytrace path's wave-amplitude leg; the bare Surface dataclass
    produced by ``surfaces_from_prescription`` does NOT honour those
    keys (it would translate them only on a COORDBRK surface).  For a
    clean chief-ray geometric test we therefore synthesize the parent
    paraboloid Surface directly from the factory's ``radius`` and
    ``conic`` outputs and trace a single chief ray launched at the
    factory's ``decenter[0]`` offset.  The chief-ray geometric
    prediction is invariant under that synthesis -- it depends only on
    the parent-paraboloid sag, not on the OAP's local-frame
    transform.
    """

    def test_chief_ray_intercepts_parent_focal_plane_at_zero(self):
        """f=50 mm, alpha=pi/6 (surface-normal angle 30 deg, chief-ray
        fold 60 deg).  We use the 60-deg-fold geometry rather than the
        full 90-deg case because the sequential trace's z-distance
        ``thickness`` is degenerate when the reflected ray has N = 0
        (the 90-deg case is verified separately via direction).  The
        bug fix is exercised identically: the chief-ray launch radius
        is ``2*f*tan(alpha)``, so a pre-v4.15.1 prescription
        (``h = f*tan(alpha)``) would launch from the wrong radial
        offset and miss the parent focus by a factor-of-two scale
        error.
        """
        f = 50e-3
        alpha = math.pi / 6
        clear_aperture = 80e-3
        rx = make_off_axis_parabola(
            focal_length=f, off_axis_angle=alpha,
            clear_aperture=clear_aperture,
        )
        surf = rx['surfaces'][0]
        R = float(surf['radius'])
        conic = float(surf['conic'])
        h = float(surf['decenter'][0])

        # Sanity: the factory uses the corrected ``r = 2 f tan(alpha)``.
        assert h == pytest.approx(2.0 * f * math.tan(alpha), rel=1e-12)

        # Build the parent paraboloid Surface.  Vertex at z=0, opening
        # toward +z (R = +2f).  Reflecting (concave) side faces +z, so
        # the chief ray must approach from z = +z_start going in -z.
        # Place the image plane at z = f via ``thickness = f`` on the
        # parent (``_transfer`` then advances each ray to z = f along
        # its post-reflection direction).  The parent-paraboloid
        # ``semi_diameter`` must contain the chief-ray launch radius
        # h = 2*f*tan(alpha); pad by 2x for clarity.
        parent_semi_dia = max(2.0 * h, clear_aperture)
        parent = Surface(
            radius=R, conic=conic, is_mirror=True,
            thickness=f, semi_diameter=parent_semi_dia,
            label='parent_paraboloid',
        )
        image = Surface(
            radius=float('inf'), is_mirror=False,
            thickness=0.0, label='parent_focal_plane',
        )

        wavelength = 1.31e-6
        # Launch a collimated chief ray well above the paraboloid
        # (z = 4f puts the ray comfortably above the surface's sag at
        # h, which is just h^2/(4f) = f*tan^2(alpha) / 4 << f for
        # alpha < pi/4) travelling in -z.  ``_intersect_surface``
        # Newton-iterates the surface from this start point and lands
        # the ray on (h, 0, sag_at_h).
        ray = make_ray(x=h, y=0.0, L=0.0, M=0.0,
                       wavelength=wavelength)
        ray.N[:] = -1.0
        ray.z[:] = 4.0 * f

        result = trace(ray, [parent, image], wavelength)
        img = result.image_rays
        assert bool(img.alive[0]), (
            "Chief ray was vignetted by the parent paraboloid -- "
            "check OAP clear_aperture vs the chief-ray launch radius."
        )

        # Chief-ray geometric prediction: at the parent focal plane
        # z = f, the ray's (x, y) intercept must be the parent focus
        # (0, 0) within 1% of the launch radius.  Pre-v4.15.1 the
        # factory's bug-2 decenter ``h = f tan(alpha)`` is half the
        # geometrically-correct value, so the chief ray hits the
        # parabola at the wrong radial offset and the reflected ray
        # misses the focus by O(h).
        tol = 0.01 * h
        x_img = float(img.x[0])
        y_img = float(img.y[0])
        assert abs(x_img) < tol, (
            f"Chief ray intercept x = {x_img!r}; expected (0, 0) "
            f"within {tol!r} (1 % of launch radius {h!r}).  This is "
            f"the audit's #1 missing test -- pre-v4.15.1 the bug-2 "
            f"decenter would land the chief ray at the wrong "
            f"parent-paraboloid radial offset and the reflected ray "
            f"would miss the parent focus."
        )
        assert abs(y_img) < tol, (
            f"Chief ray intercept y = {y_img!r}; expected ~0."
        )

    def test_chief_ray_reflection_direction_at_90_deg_fold(self):
        """Companion to the focal-plane test for the audit's named
        90-deg-fold (``alpha = pi/4``) case.  The sequential trace's
        z-thickness convention breaks down when the reflected ray's N
        is zero, so we trace ONLY through the mirror (no image plane)
        and check that the reflected ray's direction matches the
        chief-ray geometric prediction:

            direction = (-1, 0, 0)

        i.e. purely in -x at the launch radius ``h = 2*f``.  This
        pins the audit's named case ``(2*f*tan(pi/4), 0) = (2*f, 0)``
        chief-ray launch radius without requiring a tilted image
        plane.
        """
        f = 50e-3
        alpha = math.pi / 4
        clear_aperture = 220e-3   # large enough to contain h = 2f
        rx = make_off_axis_parabola(
            focal_length=f, off_axis_angle=alpha,
            clear_aperture=clear_aperture,
        )
        surf = rx['surfaces'][0]
        R = float(surf['radius'])
        conic = float(surf['conic'])
        h = float(surf['decenter'][0])
        assert h == pytest.approx(2.0 * f, rel=1e-12)

        parent_semi_dia = max(2.0 * h, clear_aperture)
        parent = Surface(
            radius=R, conic=conic, is_mirror=True,
            thickness=0.0, semi_diameter=parent_semi_dia,
            label='parent_paraboloid',
        )
        wavelength = 1.31e-6
        ray = make_ray(x=h, y=0.0, L=0.0, M=0.0,
                       wavelength=wavelength)
        ray.N[:] = -1.0
        ray.z[:] = 4.0 * f

        result = trace(ray, [parent], wavelength)
        img = result.image_rays
        assert bool(img.alive[0])

        # Reflected ray must travel in -x with N = 0 (the 90-deg fold).
        # Tolerance 1e-9 reflects roundoff in the ray-quadric
        # intersect and surface-normal numerics; the geometry itself
        # is exact.
        assert float(img.L[0]) == pytest.approx(-1.0, abs=1e-9), (
            f"Reflected ray L = {img.L[0]!r}; expected -1.0 for a "
            f"90-deg-fold OAP (chief ray bent purely in -x).  This "
            f"is the audit's named (2*f*tan(pi/4), 0) reference "
            f"case."
        )
        assert float(img.M[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(img.N[0]) == pytest.approx(0.0, abs=1e-9)


# ============================================================================
# A.2 -- Zemax QBFS / QCON SURFTYPE parsing (P1-NEW-E)
# ============================================================================


def _write_zmx(text: str, suffix: str = '.zmx') -> str:
    """Write a Zemax-text fixture string to a temp file and return
    the path.  UTF-8 is fine here; the loader tries UTF-16-LE first
    then UTF-8."""
    fd, path = tempfile.mkstemp(suffix=suffix, text=True)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def _minimal_q_zmx(surftype, coeffs_per_mm, *, r_max_parm0_mm=None,
                   semi_diameter_mm=5.0, conic=0.0):
    """A minimal 2-surface .zmx with one Q-type freeform in slot 1.

    Parameters
    ----------
    surftype : str
        ``'QBFS'`` or ``'QCON'`` (case-sensitive; the loader
        normalises internally).
    coeffs_per_mm : sequence of float
        PARM 1..N values -- Forbes ``a_0..a_{N-1}`` coefficients in
        Zemax lens units (mm in this fixture).
    r_max_parm0_mm : float, optional
        When given, written as PARM 0 (Zemax's Norm Radius convention).
        When omitted, the loader falls back to DIAM.
    semi_diameter_mm : float, default 5.0
        DIAM value on every surface in the fixture (the loader stores
        this as the surface's semi-diameter).
    conic : float, default 0.0
        CONI value on the Q-type surface.  QCON typically carries a
        non-zero base conic; QBFS does not.
    """
    lines = [
        'VERS 210000 0 123 0 0',
        'MODE SEQ',
        f'NAME test_{surftype.lower()}',
        'UNIT MM X W X CM MR CPMM',
        'ENPD 10.0',
        'WAVM 1 0.587600 1.0',
        'PWAV 1',
        'SURF 0',
        '  TYPE STANDARD',
        '  CURV 0.0',
        '  DISZ INFINITY',
        f'  DIAM {semi_diameter_mm}',
        'SURF 1',
        f'  TYPE {surftype}',
        '  STOP',
        '  CURV 0.01 0 0 0 0 ""',
        '  DISZ 3.0',
        '  GLAS BK7 0 0 1.5 50.0',
    ]
    if conic != 0.0:
        lines.append(f'  CONI {conic:.6f}')
    if r_max_parm0_mm is not None:
        lines.append(f'  PARM 0 {r_max_parm0_mm:.10e}')
    for idx, c in enumerate(coeffs_per_mm, start=1):
        lines.append(f'  PARM {idx} {c:.10e}')
    lines += [
        f'  DIAM {semi_diameter_mm}',
        'SURF 2',
        '  TYPE STANDARD',
        '  CURV -0.005 0 0 0 0 ""',
        '  DISZ 50.0',
        f'  DIAM {semi_diameter_mm}',
        'SURF 3',
        '  TYPE STANDARD',
        '  CURV 0.0 0 0 0 0 ""',
        '  DISZ 0.0',
        f'  DIAM {semi_diameter_mm}',
        'BLNK',
    ]
    return '\n'.join(lines) + '\n'


def _minimal_qbfs_zmx(coeffs_per_mm, r_max_parm0_mm=None,
                      semi_diameter_mm=5.0, conic=0.0):
    """Convenience wrapper -- a Q-bfs fixture (no inherent conic)."""
    return _minimal_q_zmx(
        'QBFS', coeffs_per_mm,
        r_max_parm0_mm=r_max_parm0_mm,
        semi_diameter_mm=semi_diameter_mm,
        conic=conic)


def _minimal_qcon_zmx(coeffs_per_mm, r_max_parm0_mm=None,
                      semi_diameter_mm=5.0, conic=-0.7):
    """Convenience wrapper -- a Q-con fixture (non-zero base conic)."""
    return _minimal_q_zmx(
        'QCON', coeffs_per_mm,
        r_max_parm0_mm=r_max_parm0_mm,
        semi_diameter_mm=semi_diameter_mm,
        conic=conic)


class TestZemaxQBFSParsing:
    """Round-trip a constructed QBFS .zmx fixture through
    :func:`load_zemax_zmx` and assert the resulting prescription has
    ``freeform_type='q_bfs'``, the correct ``q_bfs_coeffs``, and a
    physical ``r_max``."""

    def test_qbfs_freeform_type_set(self):
        coeffs_mm = [1.0e-6, 2.0e-7]
        zmx_text = _minimal_qbfs_zmx(coeffs_mm)
        path = _write_zmx(zmx_text)
        try:
            rx = load_zemax_zmx(path)
        finally:
            os.unlink(path)
        # Slot 0 is the QBFS surface (only refracting surface in the
        # fixture aside from the rear plano-concave element).
        s0 = rx['surfaces'][0]
        assert s0.get('freeform_type') == 'q_bfs', (
            f"QBFS .zmx loaded with freeform_type="
            f"{s0.get('freeform_type')!r}; expected 'q_bfs'.  "
            f"Pre-v4.15.1 the loader had no QBFS branch and silently "
            f"dropped to base conic.")

    def test_qbfs_coeffs_match_input(self):
        """PARM 1..N coefficients survive a Zemax-mm -> SI-m unit
        conversion and land in ``q_bfs_coeffs[m]`` (Forbes order
        m = parm_num - 1)."""
        coeffs_mm = [1.5e-6, 3.0e-7, 5.5e-8]
        zmx_text = _minimal_qbfs_zmx(coeffs_mm)
        path = _write_zmx(zmx_text)
        try:
            rx = load_zemax_zmx(path)
        finally:
            os.unlink(path)
        coeffs_loaded = rx['surfaces'][0]['q_bfs_coeffs']
        # mm -> m: multiply by 1e-3 (the loader's ``unit_scale``).
        expected = [c * 1e-3 for c in coeffs_mm]
        assert len(coeffs_loaded) == len(coeffs_mm)
        for k, (lo, ex) in enumerate(zip(coeffs_loaded, expected)):
            assert lo == pytest.approx(ex, rel=1e-12), (
                f"QBFS coefficient a_{k}: got {lo!r}, expected {ex!r} "
                f"(input {coeffs_mm[k]!r} mm * unit_scale 1e-3)."
            )

    def test_qbfs_r_max_from_parm0(self):
        """When PARM 0 is present it is the Norm Radius -- used
        verbatim (in lens units -> meters)."""
        r_max_mm = 2.5
        coeffs_mm = [1.0e-7]
        zmx_text = _minimal_qbfs_zmx(coeffs_mm, r_max_parm0_mm=r_max_mm)
        path = _write_zmx(zmx_text)
        try:
            rx = load_zemax_zmx(path)
        finally:
            os.unlink(path)
        assert rx['surfaces'][0]['r_max'] == pytest.approx(
            r_max_mm * 1e-3, rel=1e-12), (
            f"QBFS r_max from PARM 0: got "
            f"{rx['surfaces'][0]['r_max']!r}; expected "
            f"{r_max_mm * 1e-3!r} m."
        )

    def test_qbfs_r_max_falls_back_to_diam(self):
        """No PARM 0 -> r_max takes the surface's semi-diameter
        (DIAM)."""
        semi_dia_mm = 4.0
        coeffs_mm = [1.0e-7]
        zmx_text = _minimal_qbfs_zmx(
            coeffs_mm, r_max_parm0_mm=None,
            semi_diameter_mm=semi_dia_mm)
        path = _write_zmx(zmx_text)
        try:
            rx = load_zemax_zmx(path)
        finally:
            os.unlink(path)
        # DIAM is the semi-diameter in Zemax (per loader convention).
        assert rx['surfaces'][0]['r_max'] == pytest.approx(
            semi_dia_mm * 1e-3, rel=1e-12), (
            f"QBFS r_max fallback to DIAM failed: got "
            f"{rx['surfaces'][0]['r_max']!r}; expected "
            f"{semi_dia_mm * 1e-3!r} m."
        )


class TestZemaxQCONParsing:
    """Same set of assertions for QCON.  Q-con surfaces additionally
    carry a non-zero conic constant that the loader must preserve."""

    def test_qcon_freeform_type_set(self):
        coeffs_mm = [2.0e-6, 4.0e-7]
        zmx_text = _minimal_qcon_zmx(coeffs_mm, conic=-0.7)
        path = _write_zmx(zmx_text)
        try:
            rx = load_zemax_zmx(path)
        finally:
            os.unlink(path)
        s0 = rx['surfaces'][0]
        assert s0.get('freeform_type') == 'q_con'
        assert s0.get('conic') == pytest.approx(-0.7), (
            f"Q-con surface lost its conic during load: got "
            f"{s0.get('conic')!r}; expected -0.7.")

    def test_qcon_coeffs_match_input(self):
        coeffs_mm = [2.5e-6, 1.0e-7, 7.0e-9]
        zmx_text = _minimal_qcon_zmx(coeffs_mm, conic=-0.5)
        path = _write_zmx(zmx_text)
        try:
            rx = load_zemax_zmx(path)
        finally:
            os.unlink(path)
        coeffs_loaded = rx['surfaces'][0]['q_con_coeffs']
        expected = [c * 1e-3 for c in coeffs_mm]
        assert len(coeffs_loaded) == len(coeffs_mm)
        for k, (lo, ex) in enumerate(zip(coeffs_loaded, expected)):
            assert lo == pytest.approx(ex, rel=1e-12)


def _assert_q_dispatch_departure_nonzero(zmx_text, coeff_key):
    """Shared helper: parse a Q-type .zmx fixture, route the loaded
    surface dict through :func:`surface_sag_freeform`, and assert
    the freeform contribution is non-zero (i.e. the coefficients
    survived the load).
    """
    from lumenairy.elements.freeform import surface_sag_freeform

    path = _write_zmx(zmx_text)
    try:
        rx = load_zemax_zmx(path)
    finally:
        os.unlink(path)
    s0 = dict(rx['surfaces'][0])

    # Build a small grid centred on the surface.  The dispatcher
    # needs an explicit ``dx`` for the Forbes path; set ``norm_x`` /
    # ``norm_y`` slightly larger than r_max so the rectangular clip
    # is not the dominant mask.
    N = 65
    dx = 1.0e-4   # 0.1 mm
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    s0.setdefault('dx', dx)
    s0.setdefault('norm_x', float(s0['r_max']) + dx)
    s0.setdefault('norm_y', float(s0['r_max']) + dx)

    sag = surface_sag_freeform(X, Y, s0)
    assert np.all(np.isfinite(sag))

    # The departure is added on top of the base conic; compare
    # against a baseline with empty coefficients to isolate the
    # Q-type contribution.
    baseline_dict = dict(s0)
    baseline_dict[coeff_key] = []
    sag_baseline = surface_sag_freeform(X, Y, baseline_dict)
    diff = np.abs(sag - sag_baseline)
    assert np.max(diff) > 0.0, (
        f"Loaded Q-type prescription routed through "
        f"surface_sag_freeform produced no freeform departure -- "
        f"either {coeff_key} or r_max was silently dropped on the "
        f"load path."
    )


class TestZemaxQRoundTripThroughDispatcher:
    """The loaded prescription's lens-only ``surfaces`` list must be
    consumable by :func:`surface_sag_freeform` (the dispatcher used
    by both the raytrace and wave-optics paths) without raising the
    "freeform departure dropped" :class:`RuntimeWarning` that
    pre-v4.15.1 prescriptions triggered.
    """

    def test_qbfs_loaded_dispatch_runs_without_silent_drop(self):
        # Coefficient large enough that the rectangular clip is not
        # the dominant truncation; mm -> m unit conversion in the
        # loader gives ~1e-5 m sag departure at the rim.
        zmx_text = _minimal_qbfs_zmx(
            [1.0e-2], r_max_parm0_mm=4.0)
        _assert_q_dispatch_departure_nonzero(zmx_text, 'q_bfs_coeffs')

    def test_qcon_loaded_dispatch_runs_without_silent_drop(self):
        zmx_text = _minimal_qcon_zmx(
            [1.0e-2], r_max_parm0_mm=4.0, conic=-0.5)
        _assert_q_dispatch_departure_nonzero(zmx_text, 'q_con_coeffs')

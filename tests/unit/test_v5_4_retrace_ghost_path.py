"""v5.4 Phase 5 (audit P2-C closure): unit pins for
:func:`lumenairy.analysis.ghost.retrace_ghost_path`.

v5.4 ghost dock used to draw a synthesised Gaussian whose width
scaled with ``focus_z_estimate`` for the per-path spot preview --
useful as a relative-magnitude cue but not a real spot diagram.  The
audit (P2-C) flagged the lack of a per-path retrace primitive in the
library.  Phase 5 adds :func:`retrace_ghost_path`, which actually
launches a ray fan, branches REFLECT / TRANSMIT at each surface in
the user-supplied path, projects to the image plane, and reports the
spot statistics + Fresnel-product throughput.

This file pins the four behavioural contracts the ghost dock relies
on, plus one defensive raise:

1. ``test_single_double_bounce_ghost_focuses_at_modified_z`` --
   for an N-BK7 singlet, the (S2 reflect, S1 reflect) path returns
   an on-axis spot with a non-zero RMS radius (because the ghost
   focuses at a different z than the nominal image plane).

2. ``test_throughput_matches_fresnel_product`` -- the returned
   ``total_transmittance`` matches the analytical
   ``R_S2 * R_S1 * T_other^2`` Fresnel product to within numerical
   precision.

3. ``test_specular_path_to_image_plane`` -- the (transmit-all) path
   reproduces the nominal image-plane spot (the prescription's own
   forward ray bundle).

4. ``test_n_rays_scaling`` -- doubling n_rays from 64 to 256 does
   not change the RMS estimate by more than 5%.

5. ``test_invalid_path_raises`` -- a path with a surface index
   outside the prescription raises ``ValueError``.

Author: Andrew Traverso -- v5.4 Phase 5 (ghost-dock real-retrace closure).
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.analysis.ghost import (
    _path_from_pair,
    enumerate_ghost_paths,
    retrace_ghost_path,
)

# ----------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope='module')
def singlet():
    """A symmetric biconvex N-BK7 singlet (back focal length ~ 50 mm
    at 550 nm).  Used as the canonical ghost-retrace target across
    every test in this file -- two refracting surfaces means the
    only non-trivial 2-bounce ghost is (0, 1)."""
    return la.make_singlet(
        R1=0.050, R2=-0.050, d=3e-3,
        glass='N-BK7', aperture=20e-3,
    )


WAVELENGTH = 550e-9
SEMI_APERTURE = 8e-3
IMAGE_PLANE_Z = 0.050  # ~50 mm back focal distance for the canonical singlet


# ----------------------------------------------------------------------
# Test 1: ghost path produces an on-axis, finite-size spot
# ----------------------------------------------------------------------

def test_single_double_bounce_ghost_focuses_at_modified_z(singlet):
    """For an N-BK7 symmetric singlet, the (S1 reflect, S0 reflect)
    ghost focuses at a different z than the nominal image plane.

    We trace the ghost at the *nominal* image plane (50 mm past the
    last vertex).  Because the ghost forms its image elsewhere, the
    ray cloud at the nominal image plane has non-zero RMS radius;
    because the system is axially symmetric, the centroid is on-axis.
    """
    n_surfs = len(singlet['surfaces'])
    # Build the explicit (S1 reflect, S0 reflect) ghost path from the
    # (i, j) tuple convention used by ``enumerate_ghost_paths``.  The
    # helper expands to: transmit S0 forward -> reflect S1 -> transmit
    # back-leg (no intermediate surfaces in a 2-surface system) ->
    # reflect S0 -> transmit S1 forward.
    path = _path_from_pair(n_surfs, 0, 1)

    result = retrace_ghost_path(
        singlet, path,
        wavelength=WAVELENGTH,
        semi_aperture=SEMI_APERTURE,
        n_rays=64,
        image_plane_z=IMAGE_PLANE_Z,
    )

    # Required keys present
    for k in ('rays_image_plane', 'peak_xy_mm', 'rms_radius_mm',
              'fwhm_mm', 'total_transmittance', 'energy_fraction_ppm'):
        assert k in result, f"retrace_ghost_path output missing key {k!r}"

    rays_xy = result['rays_image_plane']
    assert rays_xy.ndim == 2 and rays_xy.shape[1] == 2, (
        f"rays_image_plane must be (n_alive, 2); got shape {rays_xy.shape}")
    assert rays_xy.shape[0] >= 4, (
        f"too few alive ghost rays at the image plane: {rays_xy.shape[0]}")

    cx_mm, cy_mm = result['peak_xy_mm']
    # On-axis ghost in an axially symmetric singlet: centroid within
    # a few microns of (0, 0).  Use 1 mm as the loose ceiling so
    # numerical jitter / RMS-binning artefacts don't trip the test.
    assert abs(cx_mm) < 1.0, (
        f"ghost centroid x off-axis: {cx_mm:.3f} mm")
    assert abs(cy_mm) < 1.0, (
        f"ghost centroid y off-axis: {cy_mm:.3f} mm")

    # Non-zero RMS because the ghost focuses at a different axial z.
    assert result['rms_radius_mm'] > 0.0, (
        "ghost spot collapsed to zero RMS -- ghost focal length "
        "unexpectedly coincides with the nominal image plane.")


# ----------------------------------------------------------------------
# Test 2: throughput matches Fresnel product
# ----------------------------------------------------------------------

def test_throughput_matches_fresnel_product(singlet):
    """``total_transmittance`` equals ``R_S0 * R_S1 * T_others^2`` (where
    the ``T_others`` are crossed twice per ghost) to within
    numerical precision.

    For the (S0, S1) double-bounce ghost on a 2-surface singlet:
    - S0 is transmitted forward at step 0, reflected at step 3,
      transmitted forward again at step 4: factors (1-R0), R0, (1-R0).
    - S1 is reflected at step 1, no intermediate transmit (2-surface
      system has no intermediate refracting surface): factor R1.

    So the predicted product is::

        T_pred = (1 - R0) * R1 * (1 - R0) * R0 * (1 - R0)
               = (1 - R0)^3 * R0 * R1

    Wait -- re-derive against ``_path_from_pair(2, 0, 1)``::

        (0, 'transmit')   -- enter, factor (1 - R0)
        (1, 'reflect')    -- bounce, factor R1
        (0, 'reflect')    -- bounce, factor R0
        (1, 'transmit')   -- exit, factor (1 - R1)

    so::

        T_pred = (1 - R0) * R1 * R0 * (1 - R1)
    """
    from lumenairy.glass import get_glass_index

    n_air = get_glass_index('air', WAVELENGTH)
    n_glass = get_glass_index('N-BK7', WAVELENGTH)
    R0 = ((n_glass - n_air) / (n_glass + n_air)) ** 2
    R1 = ((n_air - n_glass) / (n_air + n_glass)) ** 2
    T_pred = (1.0 - R0) * R1 * R0 * (1.0 - R1)

    n_surfs = len(singlet['surfaces'])
    path = _path_from_pair(n_surfs, 0, 1)

    result = retrace_ghost_path(
        singlet, path,
        wavelength=WAVELENGTH,
        semi_aperture=SEMI_APERTURE,
        n_rays=64,
        image_plane_z=IMAGE_PLANE_Z,
    )

    # Compare against the analytical product to within a strict
    # tolerance -- the Fresnel arithmetic is in float64 and uses the
    # same get_glass_index lookups internally.
    assert result['total_transmittance'] == pytest.approx(
        T_pred, rel=1e-9, abs=1e-15), (
        f"total_transmittance = {result['total_transmittance']!r}, "
        f"expected {T_pred!r} (rel diff "
        f"{(result['total_transmittance'] - T_pred) / T_pred:.3e})")

    # Energy ppm scaled directly.
    assert result['energy_fraction_ppm'] == pytest.approx(
        T_pred * 1e6, rel=1e-9, abs=1e-12)


# ----------------------------------------------------------------------
# Test 3: specular path agrees with the nominal trace
# ----------------------------------------------------------------------

def test_specular_path_to_image_plane(singlet):
    """The (transmit-all) path through the singlet returns a tight
    on-axis spot near the paraxial focus -- this IS the nominal ray
    bundle, so its peak position and RMS should match a vanilla
    :func:`trace_prescription` to within ray-fan-quantisation
    precision.

    The check confirms ``retrace_ghost_path`` reduces to the standard
    sequential trace when every step is 'transmit', which is the
    sanity sentinel for the path-stepping machinery.
    """
    n_surfs = len(singlet['surfaces'])
    path_specular = [(i, 'transmit') for i in range(n_surfs)]

    result = retrace_ghost_path(
        singlet, path_specular,
        wavelength=WAVELENGTH,
        semi_aperture=SEMI_APERTURE,
        n_rays=64,
        image_plane_z=IMAGE_PLANE_Z,
    )

    # Reference: ordinary sequential trace with the same image distance.
    ref = la.trace_prescription(
        singlet, wavelength=WAVELENGTH,
        semi_aperture=SEMI_APERTURE,
        num_rings=int(np.sqrt(64)),
        rays_per_ring=max(1, 64 // int(np.sqrt(64))),
        image_distance=IMAGE_PLANE_Z,
    )
    ref_rays = ref.image_rays
    alive = np.asarray(ref_rays.alive, dtype=bool)
    ref_xy = np.column_stack([
        np.asarray(ref_rays.x)[alive],
        np.asarray(ref_rays.y)[alive],
    ])
    ref_cx = float(np.mean(ref_xy[:, 0])) * 1e3
    ref_cy = float(np.mean(ref_xy[:, 1])) * 1e3
    ref_rms = float(np.sqrt(np.mean(
        (ref_xy[:, 0] - np.mean(ref_xy[:, 0])) ** 2
        + (ref_xy[:, 1] - np.mean(ref_xy[:, 1])) ** 2))) * 1e3

    # Centroids agree on-axis to within 1 micron.  Both fans are
    # axially symmetric so cx ~= cy ~= 0.
    cx_mm, cy_mm = result['peak_xy_mm']
    assert abs(cx_mm - ref_cx) < 1e-3, (
        f"specular centroid x mismatch: {cx_mm} vs {ref_cx}")
    assert abs(cy_mm - ref_cy) < 1e-3, (
        f"specular centroid y mismatch: {cy_mm} vs {ref_cy}")

    # RMS radii agree to within 20% -- both come from the same ray
    # generator and the same trace path, so quantisation noise from
    # the rings layout is the main remaining variance.
    assert result['rms_radius_mm'] == pytest.approx(ref_rms, rel=0.20), (
        f"specular RMS mismatch: {result['rms_radius_mm']} vs {ref_rms}")

    # Specular throughput = (1 - R)^2 (one forward pass through each
    # surface).  Sanity-check it is reasonable (~ 0.91 for two air /
    # N-BK7 interfaces) and definitely > 0.5.
    assert result['total_transmittance'] > 0.5, (
        f"specular transmittance suspiciously low: "
        f"{result['total_transmittance']}")
    assert result['total_transmittance'] < 1.0


# ----------------------------------------------------------------------
# Test 4: n_rays scaling
# ----------------------------------------------------------------------

def test_n_rays_scaling(singlet):
    """64 vs 256 rays give the same RMS to within 5%.

    This pins that the spot-statistic computation is dominated by
    the underlying optical geometry (the ghost focal length and the
    pupil radius), not by the ring quantisation of ``make_rings``.
    A failure here means either a numerical instability in the
    centroid / RMS pass or a bug in how the bundle is generated for
    large n_rays.

    Uses a tight (1 mm) sub-aperture so the ghost spot is in the
    paraxial regime where the RMS is genuinely sampling-independent.
    The 8-mm aperture used elsewhere produces a strongly-aberrated
    ghost where the RMS is dominated by the few outermost rays and
    is therefore inherently sampling-dependent -- not a useful test
    of the retrace machinery itself.
    """
    n_surfs = len(singlet['surfaces'])
    path = _path_from_pair(n_surfs, 0, 1)

    sub_ap = 1e-3  # 1 mm sub-aperture, well within paraxial for f=50 mm

    r64 = retrace_ghost_path(
        singlet, path,
        wavelength=WAVELENGTH,
        semi_aperture=sub_ap,
        n_rays=64,
        image_plane_z=IMAGE_PLANE_Z,
    )
    r256 = retrace_ghost_path(
        singlet, path,
        wavelength=WAVELENGTH,
        semi_aperture=sub_ap,
        n_rays=256,
        image_plane_z=IMAGE_PLANE_Z,
    )

    rms64 = r64['rms_radius_mm']
    rms256 = r256['rms_radius_mm']
    # Both should be finite, non-zero.
    assert rms64 > 0 and rms256 > 0, (
        f"non-positive RMS values: rms64={rms64}, rms256={rms256}")
    rel_diff = abs(rms256 - rms64) / max(rms64, 1e-12)
    assert rel_diff < 0.05, (
        f"n_rays scaling drift too large: rms64={rms64:.6f}, "
        f"rms256={rms256:.6f}, rel diff {rel_diff:.3%} > 5%")


# ----------------------------------------------------------------------
# Test 5: invalid surface index raises
# ----------------------------------------------------------------------

def test_invalid_path_raises(singlet):
    """A path containing a surface index beyond the prescription's
    surface count raises ``ValueError`` with a helpful message,
    rather than silently producing a degenerate spot or indexing
    into a non-existent surface."""
    # A 2-surface singlet has surfaces 0 and 1 only; index 2 is
    # out of range.
    bad_path = [(0, 'transmit'), (2, 'reflect'), (1, 'transmit')]
    with pytest.raises(ValueError, match=r'out of range|surface index'):
        retrace_ghost_path(
            singlet, bad_path,
            wavelength=WAVELENGTH,
            semi_aperture=SEMI_APERTURE,
            n_rays=64,
            image_plane_z=IMAGE_PLANE_Z,
        )

    # And a malformed action also raises (defensive coverage).
    bad_action = [(0, 'transmit'), (1, 'scatter'), (0, 'transmit')]
    with pytest.raises(ValueError, match=r"reflect|transmit"):
        retrace_ghost_path(
            singlet, bad_action,
            wavelength=WAVELENGTH,
            semi_aperture=SEMI_APERTURE,
            n_rays=64,
            image_plane_z=IMAGE_PLANE_Z,
        )

    # Empty path raises.
    with pytest.raises(ValueError, match=r'non-empty'):
        retrace_ghost_path(
            singlet, [],
            wavelength=WAVELENGTH,
            semi_aperture=SEMI_APERTURE,
            n_rays=64,
            image_plane_z=IMAGE_PLANE_Z,
        )

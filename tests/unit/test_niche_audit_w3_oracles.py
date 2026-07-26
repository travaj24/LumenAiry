"""W3 ORACLE WAVE (2026-07-25) -- the oracle-needing physics questions from
``docs/audits/AUDIT_ADVERSARIAL_CODEBASE_2026_07_25.md``.

Four independent oracles, each built BEFORE any verdict or fix:

* **W3-1** ``test_w3_t1_*`` -- coordinate-break frame convention
  (``raytrace/world.py`` vs ``intersection.py`` / ``differential.py`` /
  ``ui/model.py``) + the ``world_trace.py`` DOE zero-period guard.
* **W3-2** ``test_w3_t2_*`` -- mirror-parity signing of the stop-adjacent
  pupil legs in ``raytrace/seidel.py``.
* **W3-3** ``test_w3_t3_*`` -- ``aberration_tensor`` output_mode
  degeneracy in the asymptotic propagator family.
* **W3-4** ``test_w3_t4_*`` -- ``elements/polarization.py`` non-finite
  input guards.

Each section keeps its own oracle docstring below.
"""
from __future__ import annotations

# ==========================================================================
# W3-1 SECTION (was test_niche_audit_w3_oracles_t1.py)
# ==========================================================================
_W3_1_ORACLE = """W3-1 oracles: coordinate-break frame convention (AUDIT_ADVERSARIAL_
CODEBASE_2026_07_25, Territory R "Flagged, not claimed") + the world-trace
DOE zero-period guard (R-13 sibling).

THE PHYSICS
-----------
A coordinate break is a PASSIVE frame change.  Writing ``Q`` for the new
frame's LOCAL-TO-WORLD rotation (``r_world = Q @ r_local``, i.e. ``world_R``)
and ``T`` for the matrix applied to RAY coordinates to re-express them in the
new frame (``r_local = T @ r_world``), the identity

        T == Q.T

is linear algebra, not a convention: a pure frame change cannot move a
physical ray.  Zemax fixes the remaining freedom -- the SIGN -- by defining
``Tilt About X/Y/Z`` through the local-to-world matrix, using the right-hand
forms

    Rx(a) = [[1,0,0],[0,cos a,-sin a],[0,sin a,cos a]]      (and cyclic)

composed in intrinsic X->Y->Z order for ``PARM 6 == 0``, satisfying
``r_global = R @ r_local + offset`` (OpticStudio KB KA-01638, "Rotation
Matrix and Tilt About X/Y/Z"; its inverse formula ``Tilt About X =
ATAN2(N, -M)`` on ``(L, M, N) = R[:, 2]`` pins the sign).  So a
``tilt_x = +90 deg`` break puts the new local +z at world **-y**.

MEASURED (pre-fix, single ``tilt_x = +12 deg`` break in front of a flat
air->N-BK7 interface, axial ray):

  * ``raytrace.world._apply_coord_break``      Q = Rx_math(+12)  (CORRECT)
  * ``raytrace.intersection._apply_coord_break`` T = Rx_math(+12) => implied
    Q = Rx_math(-12)                            (INVERTED)
  * ``raytrace.differential._adrt_coordbreak``  identical to intersection
  * ``ui.model.recompute_element_frames``       Q = Rx_math(-12)  (INVERTED)

  local refracted direction   trace()       (0, -0.137072578, 0.990561007)
                              trace_world() (0, +0.137072578, 0.990561007)
  max|delta| = 2.741452e-01; the two engines deviated the ray by
  +4.121516 deg toward world +y and -4.121516 deg toward world -y
  respectively (8.243032 deg apart) for the SAME prescription.
  Post-fix both are (0, +0.137072578, 0.990561007), matching exact vector
  Snell in the world frame to <= 1.2e-16.

A mirror fold cannot see this: with two balanced tilts around a mirror the
frame flips together with the beam, so the final LOCAL state is identical
under both conventions (that degeneracy is why RT-4 was mis-reverted as a
"phantom" in 408b8c3).  These oracles therefore use a PURE TILT.
"""


import numpy as np
import pytest

import lumenairy as la
from lumenairy.glass import get_glass_index
from lumenairy.raytrace.differential import _adrt_coordbreak, ray_transfer_jacobian_analytic
from lumenairy.raytrace.intersection import _apply_coord_break as _local_cb
from lumenairy.raytrace.surface import RayBundle, Surface
from lumenairy.raytrace.trace import surfaces_from_prescription, trace
from lumenairy.raytrace.world import _apply_coord_break as _world_cb
from lumenairy.raytrace.world import world_surfaces_from_prescription
from lumenairy.raytrace.world_trace import trace_world

WL = 587.5618e-9
N_BK7 = get_glass_index('N-BK7', WL)
EZ = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------- helpers

def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def _T_of_intersection(**cb):
    """Recover the 3x3 matrix ``intersection._apply_coord_break`` applies to
    ray DIRECTIONS by pushing the three basis directions through it."""
    cols = []
    for e in np.eye(3):
        r = RayBundle(x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]),
                      L=np.array([e[0]]), M=np.array([e[1]]),
                      N=np.array([e[2]]), opd=np.array([0.0]),
                      alive=np.array([True]), wavelength=WL)
        _local_cb(r, Surface(radius=np.inf, is_coordbrk=True, **cb))
        cols.append([float(r.L[0]), float(r.M[0]), float(r.N[0])])
    return np.array(cols).T


def _presc(tilt_x_deg=0.0, tilt_y_deg=0.0, tilt_z_deg=0.0,
           dx=0.0, dy=0.0, order=0):
    """One coord break (surf_num 5) then one flat air->N-BK7 interface (10)."""
    return {
        'surfaces': [
            {'radius': float('inf'), 'glass_before': 'air',
             'glass_after': 'N-BK7', 'surf_num': 10},
        ],
        'thicknesses': [0.0],
        'aperture_diameter': 0.20,
        'coord_breaks': [
            {'surf_num': 5, 'decenter_x_m': dx, 'decenter_y_m': dy,
             'tilt_x_deg': tilt_x_deg, 'tilt_y_deg': tilt_y_deg,
             'tilt_z_deg': tilt_z_deg, 'order': order, 'thickness_m': 0.0},
        ],
    }


def _axial():
    return la.make_ray(x=0.0, y=0.0, L=0.0, M=0.0, wavelength=WL)


def _run_local(presc):
    surfs = surfaces_from_prescription(presc, include_coord_breaks=True)
    rb = trace(_axial(), surfs, WL).image_rays
    return (np.array([float(rb.x[0]), float(rb.y[0]), float(rb.z[0])]),
            np.array([float(rb.L[0]), float(rb.M[0]), float(rb.N[0])]))


def _run_world(presc):
    wsurfs = world_surfaces_from_prescription(presc)
    rb = trace_world(_axial(), wsurfs, WL).image_rays
    return (np.array([float(rb.x[0]), float(rb.y[0]), float(rb.z[0])]),
            np.array([float(rb.L[0]), float(rb.M[0]), float(rb.N[0])]),
            np.asarray(wsurfs[-1].world_R, float))


def _run_diff(presc):
    surfs = surfaces_from_prescription(presc, include_coord_breaks=True)
    out = ray_transfer_jacobian_analytic(
        np.array([0.0]), np.array([0.0]),
        np.array([0.0]), np.array([0.0]), surfs, WL)
    ux, uy = float(out.ux[0]), float(out.uy[0])
    n = 1.0 / np.sqrt(1.0 + ux * ux + uy * uy)
    return np.array([ux * n, uy * n, n])


def _snell_world(Q, n1=1.0, n2=N_BK7, d=EZ):
    """Exact vector Snell in the WORLD basis for a flat surface whose normal
    is ``Q``'s local +z column.  Written out from first principles:
        d' = mu*d + (cos_t - mu*cos_i) * n_hat,  mu = n1/n2."""
    n_hat = Q @ EZ
    cos_i = float(d @ n_hat)
    mu = n1 / n2
    cos_t = np.sqrt(1.0 - mu * mu * (1.0 - cos_i * cos_i))
    dp = mu * d + (cos_t - mu * cos_i) * n_hat
    return dp / np.linalg.norm(dp)


# ------------------------------------------------- 1. the frame convention

def test_w3_t1_world_frame_is_the_zemax_local_to_world_rotation():
    """``world._apply_coord_break`` composes the Zemax local-to-world tilt:
    right-hand ``R_math(+theta)`` in intrinsic X->Y->Z order, so a +90 deg
    tilt_x puts the new local +z at world -y (KB KA-01638; the
    ``test_world_surfaces`` right-hand-rule oracle pins the same)."""
    tx, ty, tz = np.radians([12.0, 24.0, -7.0])
    _, Q = _world_cb(np.zeros(3), np.eye(3),
                     {'tilt_x_deg': 12.0, 'tilt_y_deg': 24.0,
                      'tilt_z_deg': -7.0})
    np.testing.assert_allclose(Q, _Rx(tx) @ _Ry(ty) @ _Rz(tz), atol=1e-15)
    # KB KA-01638's inverse formulas on the local +z axis (L, M, N) =
    # R[:, 2] = (sin ty, -sin tx cos ty, cos tx cos ty).  (The article
    # writes them as ``Tilt About X = ATAN2(N, -M)`` with its own stated
    # "first argument is x, second is y" ordering, i.e. numpy's
    # ``arctan2(-M, N)``.)
    L, M, N = Q[:, 2]
    assert abs(np.degrees(np.arctan2(-M, N)) - 12.0) < 1e-12
    assert abs(np.degrees(np.arcsin(L)) - 24.0) < 1e-12
    # +90 deg tilt_x -> local +z at world -y.
    _, Q90 = _world_cb(np.zeros(3), np.eye(3), {'tilt_x_deg': 90.0})
    np.testing.assert_allclose(Q90[:, 2], [0.0, -1.0, 0.0], atol=1e-15)


def test_w3_t1_ray_transform_is_the_transpose_of_the_world_frame():
    """PRE-FIX FAILURE (the finding): ``intersection._apply_coord_break``
    applied ``R_math(+theta)`` to the rays, i.e. it used the frame matrix
    itself instead of its transpose -- implying a frame tilted by
    ``-theta``.  A pure frame change cannot move a physical ray, so
    ``T == Q.T`` exactly, for every axis and every combination."""
    for cb in ({'tilt_x_deg': 12.0},
               {'tilt_y_deg': -30.0},
               {'tilt_z_deg': 40.0},
               {'tilt_x_deg': 12.0, 'tilt_y_deg': 24.0},
               {'tilt_x_deg': 5.0, 'tilt_y_deg': -11.0, 'tilt_z_deg': 33.0}):
        T = _T_of_intersection(**cb)
        _, Q = _world_cb(np.zeros(3), np.eye(3), cb)
        np.testing.assert_allclose(T, Q.T, atol=1e-15, err_msg=str(cb))
        # ... and therefore the ray's WORLD direction is invariant.
        np.testing.assert_allclose(Q @ (T @ EZ), EZ, atol=1e-15)


def test_w3_t1_differential_coordbreak_replicates_intersection():
    """``differential._adrt_coordbreak`` documents itself as an op-for-op
    replica of ``intersection._apply_coord_break``; the W3-1 sign fix must
    keep it bit-identical (the analytic-Jacobian parity pins depend on it)."""
    surf = Surface(radius=np.inf, is_coordbrk=True, thickness=0.0,
                   tilt_x_deg=12.0, tilt_y_deg=24.0, tilt_z_deg=-7.0)
    ux, uy = 0.031, -0.017
    px, py, uxo, uyo, _opd, _dead = _adrt_coordbreak(
        np.array([1e-3]), np.array([-2e-3]), np.array([ux]), np.array([uy]),
        surf, WL, False, np.sqrt, lambda a: a, False)
    n = 1.0 / np.sqrt(1.0 + float(uxo[0]) ** 2 + float(uyo[0]) ** 2)
    d_diff = np.array([float(uxo[0]) * n, float(uyo[0]) * n, n])
    # Same break through the intersection primitive.
    d_in = np.array([ux, uy, 1.0]) / np.sqrt(1.0 + ux * ux + uy * uy)
    r = RayBundle(x=np.array([1e-3]), y=np.array([-2e-3]),
                  z=np.array([0.0]), L=np.array([d_in[0]]),
                  M=np.array([d_in[1]]), N=np.array([d_in[2]]),
                  opd=np.array([0.0]), alive=np.array([True]), wavelength=WL)
    _local_cb(r, surf)
    d_int = np.array([float(r.L[0]), float(r.M[0]), float(r.N[0])])
    np.testing.assert_allclose(d_diff, d_int, atol=1e-15)
    np.testing.assert_allclose([float(px[0]), float(py[0])],
                               [float(r.x[0]), float(r.y[0])], atol=1e-15)


# ------------------------------------- 2. the pure-tilt refraction oracle

@pytest.mark.parametrize('tilt_deg', [1.0, 3.0, 5.0, 12.0, 30.0, -18.0])
def test_w3_t1_pure_tilt_refraction_matches_vector_snell(tilt_deg):
    """PRE-FIX FAILURE for ``trace()`` and the analytic Jacobian.

    Oracle: one coord break carrying ONLY ``tilt_x``, then a flat
    air->N-BK7 interface, axial ray.  Ground truth = exact vector Snell in
    the WORLD basis against the surface normal ``Q @ ez`` with
    ``Q = Rx_math(+tilt)`` (the Zemax local-to-world tilt), pulled back into
    the surface's local frame by ``Q.T``.  No library paraxial/ABCD helper
    is involved.  All three engines must reproduce it.

    Reference values, tilt_x = +12 deg (n_BK7 = 1.516800 @ 587.6 nm):
      incidence 12.000000 deg, refraction 7.878484 deg,
      local refracted dir (0, +0.137072578, +0.990561007),
      world refracted dir (0, -0.071872001, +0.997413864).
    """
    p = _presc(tilt_x_deg=tilt_deg)
    Q = _Rx(np.radians(tilt_deg))
    d_local_truth = Q.T @ _snell_world(Q)

    _, d_trace = _run_local(p)
    _, d_world, Q_lib = _run_world(p)
    d_diff = _run_diff(p)

    np.testing.assert_allclose(Q_lib, Q, atol=1e-15)
    np.testing.assert_allclose(d_trace, d_local_truth, atol=1e-14)
    np.testing.assert_allclose(d_world, d_local_truth, atol=1e-14)
    np.testing.assert_allclose(d_diff, d_local_truth, atol=1e-13)
    # And the physical (world) ray is the same out of both engines.
    np.testing.assert_allclose(Q_lib @ d_world, Q @ d_local_truth, atol=1e-14)


def test_w3_t1_trace_and_trace_world_agree_on_a_pure_tilt():
    """PRE-FIX FAILURE: same prescription, two engines, both reporting the
    final state in the LAST SURFACE'S OWN LOCAL FRAME -> must be identical.
    Pre-fix max|delta| was 2.741452e-01 at +12 deg (opposite-sign M) and
    6.593e-01 at +30 deg."""
    for tilt in (1.0, 12.0, 30.0):
        p = _presc(tilt_x_deg=tilt)
        pos_l, d_l = _run_local(p)
        pos_w, d_w, _ = _run_world(p)
        np.testing.assert_allclose(d_l, d_w, atol=1e-14,
                                   err_msg=f'tilt_x={tilt}')
        np.testing.assert_allclose(pos_l, pos_w, atol=1e-15)


def test_w3_t1_tilt_composition_order_is_intrinsic_x_then_y():
    """Combined tilt_x + tilt_y: the two camps differed by a pure SIGN flip
    in the SAME intrinsic X->Y order (not an order bug), so a single-axis
    transpose test cannot separate the two failure modes -- pin the order
    explicitly.  Q must equal Rx(+tx) @ Ry(+ty) (Zemax PARM 6 = 0)."""
    tx_deg, ty_deg = 12.0, 24.0
    p = _presc(tilt_x_deg=tx_deg, tilt_y_deg=ty_deg)
    Q = _Rx(np.radians(tx_deg)) @ _Ry(np.radians(ty_deg))
    d_local_truth = Q.T @ _snell_world(Q)
    _, d_l = _run_local(p)
    _, d_w, Q_lib = _run_world(p)
    np.testing.assert_allclose(Q_lib, Q, atol=1e-15)
    np.testing.assert_allclose(d_l, d_local_truth, atol=1e-14)
    np.testing.assert_allclose(d_w, d_local_truth, atol=1e-14)
    np.testing.assert_allclose(_run_diff(p), d_local_truth, atol=1e-13)
    # The transposed-order alternative must NOT match (discriminator).
    Q_bad = _Ry(np.radians(ty_deg)) @ _Rx(np.radians(tx_deg))
    assert not np.allclose(Q_lib, Q_bad, atol=1e-6)


@pytest.mark.parametrize('order', [0, 1])
def test_w3_t1_decenter_plus_tilt_agrees_in_both_parm6_orders(order):
    """Decenter + tilt in one break.  The two camps ALREADY agreed on the
    decenter half (measured max|dpos| = 0.0 pre-fix) and disagreed only on
    the tilt half -- so the combined case is the sharpest probe: position
    and direction must BOTH match first principles, in both PARM 6 orders.

    First principles: the new frame's origin is ``t`` (order 0, expressed in
    the OLD frame) or ``Q @ t`` (order 1, expressed in the NEW frame); the
    ray from the world origin along +z meets that tilted plane, refracts,
    and the state is pulled back by ``Q.T``.
    """
    dy = 2e-3
    p = _presc(tilt_x_deg=12.0, dy=dy, order=order)
    Q = _Rx(np.radians(12.0))
    t = np.array([0.0, dy, 0.0])
    org = t if order == 0 else Q @ t
    n_hat = Q @ EZ
    tt = float((org @ n_hat) / (EZ @ n_hat))
    hit = tt * EZ
    pos_truth = Q.T @ (hit - org)
    dir_truth = Q.T @ _snell_world(Q)

    pos_l, d_l = _run_local(p)
    pos_w, d_w, _ = _run_world(p)
    np.testing.assert_allclose(pos_l, pos_truth, atol=1e-15)
    np.testing.assert_allclose(pos_w, pos_truth, atol=1e-15)
    np.testing.assert_allclose(d_l, dir_truth, atol=1e-14)
    np.testing.assert_allclose(d_w, dir_truth, atol=1e-14)


def test_w3_t1_zero_tilt_control_is_bit_identical():
    """Harness sanity + containment proof: with every coord-break tilt zero
    the W3-1 fix is a no-op and the two engines are bit-identical."""
    p = _presc()
    pos_l, d_l = _run_local(p)
    pos_w, d_w, Q = _run_world(p)
    np.testing.assert_array_equal(d_l, d_w)
    np.testing.assert_array_equal(pos_l, pos_w)
    np.testing.assert_array_equal(Q, np.eye(3))
    np.testing.assert_array_equal(d_l, EZ)


def test_w3_t1_mirror_fold_is_sign_degenerate_in_the_local_frame():
    """Why the 408b8c3 revert was wrong: a balanced two-tilt mirror fold
    (the ``periscope``) gives the SAME final local state under either
    convention, because the frame flips with the beam.  Pin the degeneracy
    so nobody re-litigates the sign on a fold oracle."""
    a = np.radians(45.0)
    #   T = Rx(-a) is the FIXED (Zemax) ray transform for tilt_x = +45;
    #   T = Rx(+a) is the pre-fix one.  Both must give the same answer.
    for T in (_Rx(-a), _Rx(+a)):
        d = T @ EZ                                  # after CB1
        d = d - 2.0 * float(d @ EZ) * EZ            # flat mirror, normal +z
        d = T @ d                                   # after CB2
        np.testing.assert_allclose(d, [0.0, 0.0, -1.0], atol=1e-15)


# ------------------------------- 3. ui.model frame builder (source pin)

def test_w3_t1_ui_model_element_frames_use_the_zemax_tilt_sign():
    """``ui.model.recompute_element_frames`` builds ``elem.R`` (local-to-
    world), consumed by ``element_frames_2d_mm`` / ``surface_frames_3d_mm``
    / ``world_trace_surfaces``, so it must use the SAME Zemax
    ``R_math(+theta)`` forms as ``world.py``.  Pre-fix it used
    ``R_math(-theta)``, mirroring every folded layout about y = 0 and
    (after the intersection.py fix) disagreeing with the model's own local
    trace.  Checked on the source text because ``lumenairy.ui.model``
    cannot be imported without PySide6, which this environment lacks."""
    import pathlib

    import lumenairy
    src = (pathlib.Path(lumenairy.__file__).parent / 'ui' / 'model.py'
           ).read_text(encoding='utf-8')
    body = src.split('def recompute_element_frames')[1].split('\n    def ')[0]
    # Rx_math(+tx) and Ry_math(+ty), twice each (cb_post and cb_pre arms).
    assert body.count('[0, c, -s],') == 2, 'Rx tilt block sign changed'
    assert body.count('[0, s,  c]]') == 2, 'Rx tilt block sign changed'
    assert body.count('[c, 0,  s],') == 2, 'Ry tilt block sign changed'
    assert body.count('[-s, 0, c]]') == 2, 'Ry tilt block sign changed'
    # The inverted forms must be gone.
    assert '[0, c,  s],' not in body
    assert '[c, 0, -s],' not in body


# --------------------------- 4. world_trace DOE zero/non-finite period

def _doe_world_surface():
    s = Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
                glass_after='air', thickness=0.0, label='DOE')
    s.world_origin = np.zeros(3)
    s.world_R = np.eye(3)
    return [s]


def _doe_run(px, py, mx=1.0, my=0.0, wl=1.31e-6):
    rays = la.make_ray(x=1e-3, y=0.0, L=0.0, M=0.0, wavelength=wl)
    rb = trace_world(rays, _doe_world_surface(), wl,
                     surface_diffraction={0: (mx, my, px, py)}).image_rays
    return (float(rb.L[0]), float(rb.M[0]), float(rb.N[0]),
            float(rb.opd[0]), bool(rb.alive[0]))


@pytest.mark.parametrize('period', [0.0, -0.0, float('nan')])
def test_w3_t1_world_trace_doe_zero_period_is_a_zero_kick(period):
    """R-13 sibling at ``world_trace.py`` (the audit's MEDIUM listed
    ``trace.py:179`` / ``world_trace.py:169`` together).  Pre-fix MEASURED
    on this exact probe: ``period=0.0`` raised ``ZeroDivisionError``
    mid-trace and ``period=nan`` returned ``(L, M, N, opd) =
    (nan, 0.0, nan, nan)`` with ``alive=True`` -- a silently NaN-poisoned
    LIVE ray.  The JAX twin's contract
    (``jax_trace._apply_doe_kick_jax._kick``) is "returns 0.0 when period
    is non-finite or zero"."""
    assert _doe_run(period, np.inf) == (0.0, 0.0, 1.0, 0.0, True)
    assert _doe_run(np.inf, period, mx=0.0, my=1.0) == (0.0, 0.0, 1.0,
                                                        0.0, True)


def test_w3_t1_world_trace_doe_real_grating_unchanged():
    """Containment: a real period and the idiomatic ``inf`` are untouched
    by the guard (``inf`` already gave 0.0 by IEEE division)."""
    L, M, N, opd, alive = _doe_run(10e-6, np.inf)
    assert alive
    assert abs(L - 1.31e-6 / 10e-6) < 1e-15
    assert abs(opd - 1.31e-6 / 10e-6 * 1e-3) < 1e-18
    assert abs(N - np.sqrt(1.0 - L * L)) < 1e-15
    assert _doe_run(np.inf, np.inf) == (0.0, 0.0, 1.0, 0.0, True)


# ==========================================================================
# W3-2 SECTION (was test_niche_audit_w3_oracles_t2.py)
# ==========================================================================
_W3_2_ORACLE = """W3-T2: mirror-parity signing of the stop-adjacent pupil legs.

AUDIT ITEM
----------
Commit 1fc8b1f (audit R-1/R-1b/R-1c) fixed ``compute_pupils``'s dropped
pre-stop transfer, its ``ep_z`` distance-vs-coordinate slip and its
air-walked post-stop leg, and recorded a REMAINING LIMITATION in
``_pre_stop_abcd``: both stop-adjacent legs used the UNSIGNED refractive
index, i.e. they did not follow ``system_abcd``'s Welford ``n' = -n``
mirror-parity bookkeeping (the house precedent for which is 403ea1f /
S11-1: the flip encodes the REFLECTION, not the power, so it is
R-independent and a FLAT fold flips too).  This file is the dedicated
fold-pupil oracle that commit called for.

VERDICT: REAL DEFECT, in two distinct places, both now fixed in
``lumenairy/raytrace/seidel.py``:

  (1) ``_pre_stop_abcd``'s final leg (surface ``stop_index - 1`` vertex
      -> stop vertex) used ``+t/|n|`` where ``system_abcd``'s own parity
      walk requires ``t / n_signed``.  For an ODD number of mirrors in
      ``surfaces[:stop_index]`` the leg came out with the wrong SIGN, so
      ``M_pre`` was a two-frame chimera.  Consumers: ``compute_pupils``
      (``ep_z``, ``ep_radius``, hence ``f/#`` and every EP-centred chief
      / fan aim in ``analysis/field.py`` + ``raytrace/ray_fan.py``) and
      ``seidel_coefficients`` (marginal / chief initial conditions).

  (2) ``compute_pupils``'s ``xp_z = -B_post / D_post`` silently assumed
      an output index of ``+1``.  The reduced image-side leg is
      ``T(z) = [[1, z/n_out], [0, 1]]``, so the imaging condition gives
      ``z_xp = -B * n_out / D``.  ``_post_stop_abcd`` works in the
      sub-system's OWN Welford frame (parity 0 on the stop's image
      side), whose output index sign is
      ``(-1)**(mirrors strictly after the stop)``.  Every mirror after
      the stop therefore flipped ``xp_z``'s sign.

ORACLE DESIGN (two independent ground truths, no library paraxial code)
-----------------------------------------------------------------------
GT1 -- EXACT REAL RAY TRACE.  ``lumenairy.raytrace.trace`` is a 3-D
vector intersect / reflect / refract engine that shares no code with
``seidel.py``.  Probe rays are launched at the surface-0 vertex plane
with exact heights and slopes and the linear response is extracted by
central differences (1e-7 m / 1e-7 rad):

  * EP -- trace ``surfaces[:k+1]``; the recorded bundle sits on the
    stop's own vertex plane, so ``y_stop = a*y0 + b*u0``.  A ray passes
    iff ``|y_stop| <= r_stop``, i.e. iff ``|y(z = b/a)| <= r_stop/|a|``
    in object space.  That IS the entrance pupil, by definition:
    ``ep_z = b/a``, ``ep_radius = r_stop/|a|``.  Object space is ahead
    of surface 0, so it always has mirror parity 0 -- there is no sign
    convention to argue about, and the consumers
    (``_chief_y_offset``/``_ep_offset`` both compute
    ``-ep_z*tan(field)``) use exactly this coordinate.
  * XP -- trace ``surfaces[k:]`` from the stop plane; the stop images
    where the ``u_stop`` dependence of the output height vanishes:
    ``z_xp = -q/s``, magnification ``m = p + r*z_xp``,
    ``xp_radius = |m| * r_stop``.

GT2 -- a hand-written 2x2 Welford paraxial product (``_walk`` below),
written from scratch in this file with the mirror convention spelled
out, converted to real coordinates through the signed output index.
GT1 and GT2 agree to <= 7.8e-14 m on all 11 fold configurations; the
library is then compared against them.

MEASURED (mm; 15 mm fold->stop gap, 6 mm stop radius)
------------------------------------------------------
  system                     quantity   pre-fix        exact-oracle    err
  flat fold before stop      ep_z       +68.389182     +16.812946   +307%
                             ep_radius    9.610969       6.439672  +49.2%
                             f/#          2.0085547      2.9976927  -33.0%
  powered fold (R=-300) "    ep_z       +72.785894     +18.407674   +295%
                             ep_radius   10.979236       5.943389  +84.7%
                             f/#          1.3441894      2.4831242  -45.9%
  flat fold after stop       xp_z        +3.547510      -3.547510   -200%
  powered fold after stop    xp_z        +2.550903      -2.550903   -200%
Post-fix residual vs GT1: <= 7.8e-14 m on every quantity, all 11
configurations.  Mirrorless control: exact before and after.

MECHANISM SIGNATURE
-------------------
* Pre-fix, the FLAT-fold-before-stop system returned BIT-IDENTICAL
  pupils to the mirrorless control -- the fold's only paraxial effect on
  that leg is its sign, so dropping the sign made the fold invisible.
* The EP error vanishes IFF the fold->stop gap is zero: with
  ``M_slice = [[a,b],[c,d]]`` and reduced leg ``tau``,
  ``A = a +- tau*c`` and ``B = b +- tau*d``, and
  ``B_t/A_t - B_n/A_n`` is proportional to ``tau * det(M_slice)``.
  Measured: 0.000% at gap 0, +4.50% at 0.5 mm, +307% at 15 mm,
  +5930% at 30 mm.
* It is exactly zero for EVEN mirror parity (2 folds before the stop is
  bit-identical pre- and post-fix).
* ``xp_z_prefix / xp_z_exact`` was measured to be exactly
  ``(-1)**(mirrors strictly after the stop)`` on all 11 configurations,
  which is why a fold BEFORE the stop leaves ``xp_z`` untouched: the
  frame conjugation and the missing ``n_out`` sign cancel there.
* ``xp_radius`` is provably parity-INVARIANT --
  ``m = A + (C/n_out) z_xp = (AD - BC)/D = det/D`` is independent of
  ``n_out``, and an upstream mirror only conjugates the post-stop matrix
  by ``diag(1,-1)``, which preserves ``A``, ``D`` and ``det``.  Measured
  exact (<= 1.1e-14 m) in all 11 configurations, pre- AND post-fix.

FLAGGED, NOT FIXED (measured here, distinct finding, no mirrors
involved): the index MAGNITUDE is still missing from both pupil
positions.  With a BK7 IMAGE space ``xp_z`` reads -24.161034 mm where
the exact oracle gives -36.647419 mm (ratio 1/1.516798 exactly); with a
BK7 OBJECT space ``ep_z`` reads +51.299438 mm vs +77.810907 mm exact
(same 1/n).  Both need their own immersed-conjugate oracle.
"""


import numpy as np
import pytest

from lumenairy.glass import get_glass_index
from lumenairy.raytrace import (
    RayBundle,
    Surface,
    compute_pupils,
    first_order_data,
    seidel_coefficients,
    surfaces_from_prescription,
    system_abcd,
    trace,
)
from lumenairy.raytrace.seidel import _post_stop_abcd, _pre_stop_abcd

WL_t2 = 587.6e-9
DY = 1e-7          # central-difference height step [m]
DU = 1e-7          # central-difference slope step [rad]
STOP_K = 4         # stop surface index in the skeleton below
R_STOP = 6.0e-3    # stop semi-diameter [m]

TOL_M = 2.0e-13    # post-fix agreement tolerance [m] (measured <= 7.8e-14)


# ===========================================================================
# GT1: exact real-ray oracle (raytrace.trace only -- no paraxial helpers)
# ===========================================================================
def _launch(y_list, u_list, wl=WL_t2):
    """Rays on the z=0 vertex plane at height ``y`` with slope ``dy/dz``."""
    y = np.asarray(y_list, float)
    u = np.asarray(u_list, float)
    nn = np.hypot(1.0, u)
    return RayBundle(
        x=np.zeros_like(y), y=y.copy(), z=np.zeros_like(y),
        L=np.zeros_like(y), M=u / nn, N=1.0 / nn, wavelength=wl,
        alive=np.ones_like(y, dtype=bool), opd=np.zeros_like(y),
        error_code=np.zeros(y.shape, dtype=np.uint8),
    )


def _exact_map(surfs, wl=WL_t2):
    """Real-coordinate (p, q, r, s) of the EXACT trace.

    ``y_out = p*y_in + q*u_in``, ``u_out = r*y_in + s*u_in``, from the
    first surface's vertex plane (pre-refraction) to the last surface's
    vertex plane (post-refraction).  ``u`` is always ``dy/dz`` in the
    trace's own (never-flipping) z frame, i.e. ``M/N``.
    """
    out = trace(_launch([+DY, -DY, 0.0, 0.0], [0.0, 0.0, +DU, -DU], wl),
                surfs, wl).ray_history[-1]
    assert out.alive.all(), f'oracle rays died: {out.error_code}'
    yo, uo = out.y, out.M / out.N
    return ((yo[0] - yo[1]) / (2 * DY), (yo[2] - yo[3]) / (2 * DU),
            (uo[0] - uo[1]) / (2 * DY), (uo[2] - uo[3]) / (2 * DU))


def _oracle_ep(surfs, k=STOP_K, wl=WL_t2):
    """(ep_z, ep_radius) from the exact trace -- see the module docstring."""
    a, b, _c, _d = _exact_map(surfs[:k + 1], wl)
    return b / a, abs(float(surfs[k].semi_diameter) / a)


def _oracle_xp(surfs, k=STOP_K, wl=WL_t2):
    """(xp_z, xp_radius) from the exact trace -- see the module docstring."""
    p, q, r, s = _exact_map(surfs[k:], wl)
    z = -q / s
    return z, abs(p + r * z) * float(surfs[k].semi_diameter)


# ===========================================================================
# GT2: hand-written Welford paraxial product (independent of seidel.py)
# ===========================================================================
def _walk(surfs, wl=WL_t2):
    """Reduced-coordinate ``(y, w = n_signed * u)`` walk, from scratch.

    WELFORD MIRROR CONVENTION, stated explicitly:
      * a mirror is a refracting surface with ``n' = -n``, hence power
        ``phi = (n' - n)/R = -2n/R``;
      * the ``n -> -n`` flip encodes the REFLECTION, not the power, so
        it is R-INDEPENDENT -- a flat fold flips too (403ea1f / S11-1);
      * therefore every leg downstream of an odd number of mirrors uses
        a NEGATIVE index and its reduced length is ``t / n_signed``.
        With positive post-fold thicknesses that reduced length is
        negative, which is exactly what reverses ``u = w / n_signed``
        at the fold.

    Returns ``(Mv, Mr, par_post)``: ``Mv[i]`` reaches surface ``i``'s
    vertex PRE-refraction, ``Mr[i]`` POST-refraction, ``par_post[i]`` is
    the mirror parity after surface ``i``.
    """
    M = np.eye(2)
    par = 0
    Mv, Mr, par_post = [], [], []
    for i, s in enumerate(surfs):
        Mv.append(M.copy())
        sg = 1.0 if par == 0 else -1.0
        n1 = sg * get_glass_index(s.glass_before, wl)
        n2 = -n1 if s.is_mirror else sg * get_glass_index(s.glass_after, wl)
        if np.isfinite(s.radius):
            phi = (n2 - n1) / s.radius
            M = np.array([[1.0, 0.0], [-phi, 1.0]]) @ M
        if s.is_mirror:
            par ^= 1
        Mr.append(M.copy())
        par_post.append(par)
        if i < len(surfs) - 1:
            M = np.array([[1.0, float(s.thickness) / n2], [0.0, 1.0]]) @ M
    return Mv, Mr, par_post


def _hand_pupils(surfs, k=STOP_K, wl=WL_t2):
    """(ep_z, ep_radius, xp_z, xp_radius) from GT2."""
    Mv, Mr, par_post = _walk(surfs, wl)
    r_stop = float(surfs[k].semi_diameter)
    n_obj = get_glass_index(surfs[0].glass_before, wl)   # parity 0 at surf 0

    A, B = Mv[k][0, 0], Mv[k][0, 1]
    ep_z = B * n_obj / A            # real-coordinate b = B * n_obj
    ep_r = abs(r_stop / A)

    Mq = Mr[-1] @ np.linalg.inv(Mr[k])          # global frame, det == 1
    A2, B2, C2, D2 = Mq[0, 0], Mq[0, 1], Mq[1, 0], Mq[1, 1]
    n_out = ((1.0 if par_post[-1] == 0 else -1.0)
             * get_glass_index(surfs[-1].glass_after, wl))
    xp_z = -B2 * n_out / D2         # real q/s: the input index cancels
    xp_r = abs(A2 + (C2 / n_out) * xp_z) * r_stop
    return ep_z, ep_r, xp_z, xp_r


# ===========================================================================
# Systems.  Mirrors only ever land on air->air plates, so swapping a
# plate for a fold never breaks the glass chain.
#   0 plate-0 | 1 R=+100 air/BK7 | 2 R=-100 BK7/air | 3 plate-A
#   4 STOP (6 mm) | 5 plate-B | 6 R=-80 air/BK7 | 7 R=+80 BK7/air
# ===========================================================================
def _build(mirrors=(), gap=15e-3, R_A=np.inf, R_B=np.inf, t_sign=+1.0,
           last_glass='air'):
    surfs = [
        {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air'},
        {'radius': 100e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': -100e-3, 'glass_before': 'N-BK7', 'glass_after': 'air'},
        {'radius': R_A, 'glass_before': 'air', 'glass_after': 'air'},
        {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
         'is_stop': True, 'semi_diameter': R_STOP},
        {'radius': R_B, 'glass_before': 'air', 'glass_after': 'air'},
        {'radius': -80e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': 80e-3, 'glass_before': 'N-BK7', 'glass_after': last_glass},
    ]
    ts = [8e-3, 5e-3, 20e-3, gap, 12e-3, 13e-3, 4e-3, 0.0]
    for i in mirrors:
        surfs[i]['glass_after'] = 'MIRROR'
        surfs[i]['is_mirror'] = True
    if mirrors:
        for j in range(min(mirrors), len(ts)):
            ts[j] = t_sign * abs(ts[j])
    return surfaces_from_prescription(
        {'name': 'w3t2', 'aperture_diameter': 25.4e-3,
         'surfaces': surfs, 'thicknesses': ts})


# label -> _build kwargs.  Covers both sides of the stop, 0-3 mirrors,
# flat and powered folds, a mirror AT the stop, and both thickness-sign
# conventions the library supports for post-fold legs.
CASES = {
    'control_no_mirror': dict(mirrors=()),
    'flat_fold_before_stop': dict(mirrors=(3,)),
    'powered_fold_before_stop': dict(mirrors=(3,), R_A=-300e-3),
    'flat_fold_before_stop_negative_t': dict(mirrors=(3,), t_sign=-1.0),
    'two_folds_before_stop': dict(mirrors=(0, 3)),
    'flat_fold_after_stop': dict(mirrors=(5,)),
    'powered_fold_after_stop': dict(mirrors=(5,), R_B=-300e-3),
    'folds_both_sides': dict(mirrors=(3, 5)),
    'mirror_at_the_stop': dict(mirrors=(4,)),
    'mirror_at_stop_plus_fold_after': dict(mirrors=(4, 5)),
    'three_folds': dict(mirrors=(0, 3, 5)),
}

# Exact-real-ray oracle values [mm], cross-checked by GT2 to <= 7.8e-14 m.
#   label -> (ep_z, ep_radius, xp_z, xp_radius)
ORACLE_MM = {
    'control_no_mirror':
        (+68.389182020, 9.610968895, -20.899105674, 4.468321389),
    'flat_fold_before_stop':
        (+16.812945758, 6.439671607, -20.899105674, 4.468321389),
    'powered_fold_before_stop':
        (+18.407673731, 5.943388682, -20.899105674, 4.468321389),
    'flat_fold_before_stop_negative_t':
        (+68.389182020, 9.610968895, +41.060152985, 9.053646153),
    'two_folds_before_stop':
        (-16.812945758, 6.439671607, -20.899105674, 4.468321389),
    'flat_fold_after_stop':
        (+68.389182020, 9.610968895, -3.547510456, 5.824870362),
    'powered_fold_after_stop':
        (+68.389182020, 9.610968895, -2.550902960, 6.416070878),
    'folds_both_sides':
        (+16.812945758, 6.439671607, -3.547510456, 5.824870362),
    'mirror_at_the_stop':
        (+68.389182020, 9.610968895, -20.899105674, 4.468321389),
    'mirror_at_stop_plus_fold_after':
        (+68.389182020, 9.610968895, -3.547510456, 5.824870362),
    'three_folds':
        (-16.812945758, 6.439671607, -3.547510456, 5.824870362),
}

# Values the PRE-FIX unsigned-index code returned; asserted rejected.
PREFIX_MM = {
    'flat_fold_before_stop':
        dict(ep_z=+68.389182020, ep_radius=9.610968895, fnum=2.008554699),
    'powered_fold_before_stop':
        dict(ep_z=+72.785894175, ep_radius=10.979235588, fnum=1.344189424),
    'flat_fold_before_stop_negative_t':
        dict(ep_z=+16.812945758, ep_radius=6.439671607, fnum=5.250266930),
    'flat_fold_after_stop': dict(xp_z=+3.547510456),
    'powered_fold_after_stop': dict(xp_z=+2.550902960),
    'folds_both_sides':
        dict(ep_z=+68.389182020, ep_radius=9.610968895, xp_z=+3.547510456),
    'three_folds': dict(xp_z=+3.547510456),
}

ODD_PRE = ('flat_fold_before_stop', 'powered_fold_before_stop',
           'flat_fold_before_stop_negative_t', 'folds_both_sides')
ODD_POST = ('flat_fold_after_stop', 'powered_fold_after_stop',
            'folds_both_sides', 'mirror_at_stop_plus_fold_after',
            'three_folds')


# ===========================================================================
# Harness self-validation
# ===========================================================================
@pytest.mark.parametrize('label', sorted(CASES))
def test_w3_t2_two_ground_truths_agree(label):
    """GT1 (exact real ray) and GT2 (hand Welford ABCD) must agree.

    Neither uses ``seidel.py``.  If they disagree the oracle is not
    trustworthy and nothing below means anything.  Measured worst
    disagreement over the 11 configurations: 7.773e-14 m.
    """
    surfs = _build(**CASES[label])
    ez, er = _oracle_ep(surfs)
    xz, xr = _oracle_xp(surfs)
    hz, hr, hxz, hxr = _hand_pupils(surfs)
    assert abs(ez - hz) < 1e-12, f'{label}: ep_z {ez} vs {hz}'
    assert abs(er - hr) < 1e-12, f'{label}: ep_radius {er} vs {hr}'
    assert abs(xz - hxz) < 1e-12, f'{label}: xp_z {xz} vs {hxz}'
    assert abs(xr - hxr) < 1e-12, f'{label}: xp_radius {xr} vs {hxr}'


def test_w3_t2_control_no_mirror_is_exact():
    """The mirrorless control proves the harness, not the fix.

    It has no mirror parity to get wrong, so ``compute_pupils`` must
    match the exact real-ray oracle both before and after the fix.
    Exact values [mm]: ep_z +68.389182020, ep_radius 9.610968895,
    xp_z -20.899105674, xp_radius 4.468321389.
    """
    surfs = _build()
    ez, er = _oracle_ep(surfs)
    xz, xr = _oracle_xp(surfs)
    pu = compute_pupils(surfs, WL_t2)
    assert abs(pu.ep_z - ez) < TOL_M
    assert abs(pu.ep_radius - er) < TOL_M
    assert abs(pu.xp_z - xz) < TOL_M
    assert abs(pu.xp_radius - xr) < TOL_M
    assert abs(pu.ep_z * 1e3 - 68.389182020) < 1e-8
    assert abs(pu.xp_z * 1e3 - (-20.899105674)) < 1e-8


# ===========================================================================
# The verdict: library vs the exact real-ray oracle
# ===========================================================================
@pytest.mark.parametrize('label', sorted(CASES))
def test_w3_t2_compute_pupils_matches_exact_real_ray_oracle(label):
    """``compute_pupils`` must equal the exact real-ray pupils.

    Post-fix residual measured <= 7.8e-14 m on every quantity of every
    configuration; pre-fix the odd-parity rows were off by up to
    +307% (ep_z), +84.7% (ep_radius) and a hard sign flip (xp_z).
    """
    surfs = _build(**CASES[label])
    ez, er = _oracle_ep(surfs)
    xz, xr = _oracle_xp(surfs)
    pu = compute_pupils(surfs, WL_t2)
    assert abs(pu.ep_z - ez) < TOL_M, f'{label}: ep_z {pu.ep_z} vs {ez}'
    assert abs(pu.ep_radius - er) < TOL_M, f'{label}: ep_radius'
    assert abs(pu.xp_z - xz) < TOL_M, f'{label}: xp_z {pu.xp_z} vs {xz}'
    assert abs(pu.xp_radius - xr) < TOL_M, f'{label}: xp_radius'


@pytest.mark.parametrize('label', sorted(ORACLE_MM))
def test_w3_t2_hardcoded_oracle_values(label):
    """Same assertion against hard-coded 12-digit oracle values.

    Freezes the answer independently of the oracle helpers above, so a
    future regression in BOTH the library and the embedded oracle still
    trips this pin.
    """
    surfs = _build(**CASES[label])
    pu = compute_pupils(surfs, WL_t2)
    ep_z, ep_r, xp_z, xp_r = ORACLE_MM[label]
    assert pu.ep_z * 1e3 == pytest.approx(ep_z, abs=1e-8), label
    assert pu.ep_radius * 1e3 == pytest.approx(ep_r, abs=1e-8), label
    assert pu.xp_z * 1e3 == pytest.approx(xp_z, abs=1e-8), label
    assert pu.xp_radius * 1e3 == pytest.approx(xp_r, abs=1e-8), label


@pytest.mark.parametrize('label', sorted(PREFIX_MM))
def test_w3_t2_prefix_unsigned_leg_values_rejected(label):
    """The pre-fix (unsigned-index) answers must NOT come back.

    Pre-fix, ``flat_fold_before_stop`` returned pupils BIT-IDENTICAL to
    the mirrorless control (ep_z +68.389182 mm, ep_radius 9.610969 mm)
    because a flat fold's only paraxial effect on that leg IS the sign.
    """
    surfs = _build(**CASES[label])
    pu = compute_pupils(surfs, WL_t2)
    fod = first_order_data(surfs, WL_t2)
    got = {'ep_z': pu.ep_z * 1e3, 'ep_radius': pu.ep_radius * 1e3,
           'xp_z': pu.xp_z * 1e3, 'fnum': fod.fnum}
    for key, bad in PREFIX_MM[label].items():
        assert abs(got[key] - bad) > 1e-4, (
            f'{label}: {key} still reports the pre-fix value {bad}')


def test_w3_t2_fold_before_stop_measured_numbers():
    """The headline mirror-before-stop measurement, spelled out.

    flat fold, 15 mm fold->stop gap, 6 mm stop radius:
        ep_z      +68.389182 mm pre-fix -> +16.812946 mm exact  (+307%)
        ep_radius   9.610969 mm pre-fix ->   6.439672 mm exact (+49.2%)
        f/#         2.0085547  pre-fix ->   2.9976927  exact  (-33.0%)
    """
    surfs = _build(mirrors=(3,))
    ez, er = _oracle_ep(surfs)
    fod = first_order_data(surfs, WL_t2)
    assert ez * 1e3 == pytest.approx(16.812945758, abs=1e-8)
    assert er * 1e3 == pytest.approx(6.439671607, abs=1e-8)
    assert fod.ep_z * 1e3 == pytest.approx(16.812945758, abs=1e-8)
    assert fod.ep_radius * 1e3 == pytest.approx(6.439671607, abs=1e-8)
    # f/# = |EFL| / (2 * ep_radius); EFL = +38.608313474 mm.
    assert fod.fnum == pytest.approx(2.997692726, abs=1e-8)
    assert abs(fod.fnum - 2.008554699) > 1e-4        # pre-fix value


def test_w3_t2_powered_mirror_before_stop_measured_numbers():
    """Powered (R = -300 mm) fold before the stop.

        ep_z      +72.785894 mm pre-fix -> +18.407674 mm exact  (+295%)
        ep_radius  10.979236 mm pre-fix ->   5.943389 mm exact (+84.7%)
        f/#         1.3441894  pre-fix ->   2.4831242  exact  (-45.9%)
    """
    surfs = _build(mirrors=(3,), R_A=-300e-3)
    ez, er = _oracle_ep(surfs)
    fod = first_order_data(surfs, WL_t2)
    assert ez * 1e3 == pytest.approx(18.407673731, abs=1e-8)
    assert er * 1e3 == pytest.approx(5.943388682, abs=1e-8)
    assert fod.ep_z * 1e3 == pytest.approx(18.407673731, abs=1e-8)
    assert fod.fnum == pytest.approx(2.483124215, abs=1e-8)
    assert abs(fod.fnum - 1.344189424) > 1e-4        # pre-fix value


def test_w3_t2_fold_after_stop_xp_z_sign():
    """Mirror AFTER the stop: ``xp_z`` was a pure sign flip.

    flat fold: +3.547510 mm pre-fix vs -3.547510 mm exact (-200%);
    powered R = -300 mm fold: +2.550903 vs -2.550903 mm.
    ``xp_radius`` is unaffected in both (see the invariance pin below).
    """
    for kw, exact_mm, radius_mm in (
            (dict(mirrors=(5,)), -3.547510456, 5.824870362),
            (dict(mirrors=(5,), R_B=-300e-3), -2.550902960, 6.416070878)):
        surfs = _build(**kw)
        xz, xr = _oracle_xp(surfs)
        pu = compute_pupils(surfs, WL_t2)
        assert xz * 1e3 == pytest.approx(exact_mm, abs=1e-8)
        assert pu.xp_z * 1e3 == pytest.approx(exact_mm, abs=1e-8)
        assert pu.xp_z < 0.0                     # pre-fix it was > 0
        assert pu.xp_radius * 1e3 == pytest.approx(radius_mm, abs=1e-8)


def test_w3_t2_both_thickness_sign_conventions():
    """The fix holds for BOTH post-fold thickness conventions.

    The repo's own folded prescriptions use POSITIVE post-fold
    thicknesses; ``intersection._transfer`` also documents the NEGATIVE
    (true forward fold) convention.  ``system_abcd``'s reduced leg
    ``t / n_signed`` reproduces the exact trace either way, so the
    stop-adjacent leg must be signed either way too.  Pre-fix the two
    conventions returned each other's answers: with negative
    thicknesses ep_z read +16.812946 mm where the exact value is
    +68.389182 mm.
    """
    pos = _build(mirrors=(3,))
    neg = _build(mirrors=(3,), t_sign=-1.0)
    for surfs, ep_mm, er_mm in ((pos, 16.812945758, 6.439671607),
                                (neg, 68.389182020, 9.610968895)):
        ez, er = _oracle_ep(surfs)
        pu = compute_pupils(surfs, WL_t2)
        assert ez * 1e3 == pytest.approx(ep_mm, abs=1e-8)
        assert pu.ep_z * 1e3 == pytest.approx(ep_mm, abs=1e-8)
        assert pu.ep_radius * 1e3 == pytest.approx(er_mm, abs=1e-8)
    assert abs(compute_pupils(pos, WL_t2).ep_z
               - compute_pupils(neg, WL_t2).ep_z) > 1e-3


# ===========================================================================
# Mechanism signature
# ===========================================================================
def test_w3_t2_even_mirror_parity_is_untouched():
    """EVEN parity is a no-op -- the fix is regression-safe there.

    ``two_folds_before_stop`` and ``three_folds`` (2 mirrors before the
    stop) were already exact pre-fix on the EP side, and the mirrorless
    control is exact on both sides.  Mirrorless designs are bit-
    identical: the parity sum is 0, so ``n_last`` is unchanged and
    ``sign_out`` is +1.
    """
    for label in ('control_no_mirror', 'two_folds_before_stop',
                  'three_folds', 'mirror_at_the_stop'):
        surfs = _build(**CASES[label])
        ez, er = _oracle_ep(surfs)
        pu = compute_pupils(surfs, WL_t2)
        assert abs(pu.ep_z - ez) < TOL_M, label
        assert abs(pu.ep_radius - er) < TOL_M, label


def test_w3_t2_error_vanishes_at_zero_gap_and_grows_with_it():
    """Zero fold->stop gap is the mechanism discriminator.

    The EP error is proportional to ``t_last * det(M_slice)``, so it is
    EXACTLY zero when the fold sits on the stop and grows monotonically
    with the gap.  Measured pre-fix ep_z relative error: 0.000000% at
    0 mm, +4.503629% at 0.5 mm, +9.211819% at 1 mm, +55.763631% at 5 mm,
    +306.765019% at 15 mm, +5929.600154% at 30 mm.  Post-fix every one
    of these is exact, which is what this pin asserts.
    """
    prev = None
    for gap_mm in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0):
        surfs = _build(mirrors=(3,), gap=gap_mm * 1e-3)
        ez, er = _oracle_ep(surfs)
        pu = compute_pupils(surfs, WL_t2)
        assert abs(pu.ep_z - ez) < TOL_M, f'gap={gap_mm} mm'
        assert abs(pu.ep_radius - er) < TOL_M, f'gap={gap_mm} mm'
        if prev is not None:
            # the exact EP walks steadily back toward the lens
            assert ez < prev, f'gap={gap_mm} mm: ep_z not monotone'
        prev = ez
    # gap = 0 is where the signed and unsigned legs coincide exactly.
    surfs0 = _build(mirrors=(3,), gap=0.0)
    ez0, _er0 = _oracle_ep(surfs0)
    assert ez0 * 1e3 == pytest.approx(37.505828651, abs=1e-8)


def test_w3_t2_flat_fold_before_stop_no_longer_mimics_the_control():
    """Pre-fix, the flat fold was invisible to the EP.  It must not be.

    A flat fold has zero power, so the ONLY thing it does to the
    pre-stop sub-system is flip the sign of the leg that follows it.
    Dropping that sign made ``compute_pupils`` return bit-identical
    pupils for the folded and the mirrorless system.
    """
    folded = compute_pupils(_build(mirrors=(3,)), WL_t2)
    control = compute_pupils(_build(mirrors=()), WL_t2)
    assert abs(folded.ep_z - control.ep_z) > 1e-3
    assert abs(folded.ep_radius - control.ep_radius) > 1e-4


@pytest.mark.parametrize('label', sorted(CASES))
def test_w3_t2_xp_radius_is_parity_invariant(label):
    """``xp_radius`` is provably immune to the parity bug -- prove it.

    At imaging ``m = A + (C/n_out) * z_xp`` with ``z_xp = -B*n_out/D``,
    so ``m = (AD - BC)/D = det(M)/D``: ``n_out`` cancels identically.
    An upstream mirror only conjugates the post-stop matrix by
    ``diag(1,-1)``, which preserves ``A``, ``D`` and ``det``.  So
    ``xp_radius`` was already exact pre-fix in all 11 configurations
    (measured <= 1.1e-14 m) and must stay exact.
    """
    surfs = _build(**CASES[label])
    _xz, xr = _oracle_xp(surfs)
    M = _post_stop_abcd(surfs, WL_t2, STOP_K)
    det = float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
    pu = compute_pupils(surfs, WL_t2)
    assert abs(pu.xp_radius - xr) < TOL_M, label
    assert abs(pu.xp_radius - abs(det * R_STOP / float(M[1, 1]))) < 1e-15
    assert det == pytest.approx(1.0, abs=1e-12)   # reduced coords


# ===========================================================================
# Internal consistency (no ray trace needed)
# ===========================================================================
@pytest.mark.parametrize('label', sorted(CASES))
def test_w3_t2_pre_stop_leg_follows_system_abcd_parity(label):
    """``_pre_stop_abcd`` must equal what ``system_abcd`` itself walks.

    Independent construction of the same sub-system: append a
    ZERO-THICKNESS dummy flat surface at the stop vertex and let
    ``system_abcd`` walk the final leg with its own parity bookkeeping.
    Post-fix the two agree BIT-FOR-BIT (0.0) on all 11 configurations.
    Pre-fix they differed by 3.074378e-01 (flat fold before the stop)
    and 4.630390e-01 (powered fold), while every even-parity row was
    already 0.0.
    """
    surfs = _build(**CASES[label])
    M_pre = _pre_stop_abcd(surfs, WL_t2, STOP_K)
    padded = list(surfs[:STOP_K]) + [Surface(
        radius=np.inf, conic=0.0, semi_diameter=np.inf,
        glass_before=surfs[STOP_K - 1].glass_after,
        glass_after=surfs[STOP_K].glass_after,
        is_mirror=False, is_stop=False, thickness=0.0, label='(at stop)')]
    M_ref, _e, _b, _f = system_abcd(padded, WL_t2)
    assert np.abs(M_pre - M_ref).max() == 0.0, (
        f'{label}: _pre_stop_abcd disagrees with system_abcd by '
        f'{np.abs(M_pre - M_ref).max():.6e}')


@pytest.mark.parametrize('label', sorted(CASES))
def test_w3_t2_seidel_marginal_ray_starts_at_the_entrance_pupil(label):
    """``seidel_coefficients`` shares ``_pre_stop_abcd`` -- pin the knock-on.

    Its marginal ray is launched at ``y_0 = r_stop / A_pre``, which is
    the entrance-pupil radius by construction, so it must equal the
    exact real-ray ``ep_radius``.  Pre-fix this was 9.610969 mm instead
    of 6.439672 mm on the flat-fold system (+49.2%), i.e. the Seidel
    sums were evaluated on a marginal ray that overfilled the pupil.
    The chief ray must still cross the axis exactly at the stop.
    """
    surfs = _build(**CASES[label])
    _ez, er = _oracle_ep(surfs)
    data, _abcd = seidel_coefficients(surfs, WL_t2, field_angle=np.radians(2.0))
    y_m = np.asarray(data['y_marginal'], float)
    y_c = np.asarray(data['y_chief'], float)
    assert data['stop_index'] == STOP_K
    assert abs(abs(y_m[0]) - er) < TOL_M, (
        f'{label}: marginal launch {abs(y_m[0])} vs ep_radius {er}')
    assert abs(y_m[STOP_K] - R_STOP) < 1e-15      # marginal fills the stop
    assert abs(y_c[STOP_K]) < 1e-15               # chief through its centre


def test_w3_t2_post_stop_leg_stays_unsigned_by_design():
    """``_post_stop_abcd`` is deliberately NOT parity-signed -- pin that.

    It is the post-stop sub-system in its OWN Welford frame (parity 0 on
    the stop's image side): ``T_first`` uses the unsigned index AND
    ``system_abcd(surfaces[stop_index+1:])`` starts its own parity walk
    at 0, so the two halves agree.  Upstream mirrors merely conjugate
    the result by ``diag(1,-1)``, which cancels out of ``xp_z`` (once
    ``compute_pupils`` supplies the frame's own output-index sign) and
    out of ``xp_radius`` entirely.  Concretely: the post-stop matrix is
    IDENTICAL whether or not there is a fold before the stop.
    """
    a = _post_stop_abcd(_build(mirrors=()), WL_t2, STOP_K)
    b = _post_stop_abcd(_build(mirrors=(3,)), WL_t2, STOP_K)
    assert np.abs(a - b).max() == 0.0
    # and the same on the powered-fold-before-stop system
    c = _post_stop_abcd(_build(mirrors=(3,), R_A=-300e-3), WL_t2, STOP_K)
    assert np.abs(a - c).max() == 0.0


def test_w3_t2_xp_z_sign_follows_post_stop_mirror_count():
    """The correction factor is exactly ``(-1) ** m_post``.

    ``m_post`` counts mirrors STRICTLY AFTER the stop.  Measured
    pre-fix ``xp_z_reported / xp_z_exact`` on all 11 configurations was
    exactly ``(-1) ** m_post`` -- +1 for a fold BEFORE the stop (the
    frame conjugation and the dropped ``n_out`` sign cancel there) and
    -1 for a fold AFTER it.  Post-fix the ratio is +1 everywhere.
    """
    for label, kw in CASES.items():
        surfs = _build(**kw)
        m_post = sum(1 for s in surfs[STOP_K + 1:] if s.is_mirror)
        M = _post_stop_abcd(surfs, WL_t2, STOP_K)
        naive = -float(M[0, 1]) / float(M[1, 1])     # the pre-fix formula
        xz, _xr = _oracle_xp(surfs)
        assert naive / xz == pytest.approx((-1.0) ** m_post, abs=1e-9), label
        assert compute_pupils(surfs, WL_t2).xp_z == pytest.approx(xz, abs=TOL_M)
        assert (label in ODD_POST) == bool(m_post % 2)


def test_w3_t2_odd_pre_stop_parity_is_what_moved_the_entrance_pupil():
    """Only ODD pre-stop mirror parity moves the EP -- pin the partition."""
    for label, kw in CASES.items():
        surfs = _build(**kw)
        m_pre = sum(1 for s in surfs[:STOP_K] if s.is_mirror)
        M_slice, _e, _b, _f = system_abcd(surfs[:STOP_K], WL_t2)
        t = float(surfs[STOP_K - 1].thickness)
        n = get_glass_index(surfs[STOP_K - 1].glass_after, WL_t2)
        naive = np.array([[1.0, t / n], [0.0, 1.0]]) @ M_slice
        ez, _er = _oracle_ep(surfs)
        moved = abs(naive[0, 1] / naive[0, 0] - ez) > 1e-9
        assert moved == bool(m_pre % 2), (
            f'{label}: m_pre={m_pre} but moved={moved}')
        assert (label in ODD_PRE) == bool(m_pre % 2), label


# ==========================================================================
# W3-3 SECTION (was test_niche_audit_w3_oracles_t3.py)
# ==========================================================================
_W3_3_ORACLE = """W3-T3 -- ``aberration_tensor`` output-mode degeneracy, pinned against a
from-scratch Laguerre-Gauss overlap oracle.

THE FINDING
-----------
``lumenairy.propagators.asymptotic.aberration_tensor`` returned the
BIT-IDENTICAL complex ``L`` for every requested ``output_mode``.  Measured
on a 500 mm N-BK7 singlet (``w_s = 20 um``, ``w_p = 0.05``, on-axis,
default ``w_o``), pre-fix::

    out=(0, 0): L = +3.33084391626186777e-02+4.57778305512435008e-02j
    out=(1, 0): L = +3.33084391626186777e-02+4.57778305512435008e-02j
    out=(2, 0): L = +3.33084391626186777e-02+4.57778305512435008e-02j
    out=(3, 0): L = +3.33084391626186638e-02+4.57778305512434869e-02j
    out=(5, 0): L = +3.33084391626186777e-02+4.57778305512435008e-02j
    max |L_k - L_(0,0)| = 1.963e-17          <-- pure rounding

...both within one call AND across separate single-mode calls, with NO
warning emitted.

MECHANISM
---------
The closed-form branch is a point-SAMPLING functional, not a projection:
its entire output-mode dependence is ``out_const``, the output LG
polynomial evaluated at one point.  Measured identity (residual 6.9e-17)::

    L_closed_form = U(chief_ray) * conj(LG_k(sigma))|_{one point}
                  = U(chief_ray) * N_{p,0}(w_o),
    N_{p,0} = sqrt(2 / (pi w_o^2))     -- INDEPENDENT of p

so every ``(p, 0)`` mode collapses onto the piston value and every
``ell != 0`` mode would give exactly 0.  The documented escape hatch
("go off-axis / ask for an ell != 0 mode") did not work either:

  * off-axis only perturbs the shared constant by O((s2_img/w_o)^2)
    -- measured 1.9e-5 relative at s2 = 30 um -- which is not an overlap;
  * the sigma-grid branch used ``extent = 4 * w_o`` with a default
    ``w_o`` derived from the pupil-space beam matrix M (units
    1/direction-cosine^2).  Measured ``w_o = 9.83e-3`` against a fit
    s2 half-box of ``1.54e-4 m``: the grid sat 255x outside the box,
    ``propagate_modal_asymptotic`` is identically zero there, and the
    whole tensor came back EXACTLY 0.0 + 0.0j.

THE FIX (lumenairy/propagators/asymptotic_aberration_tensor.py)
---------------------------------------------------------------
1. Route everything except a pure ``[(0, 0)]`` request to the
   sigma-integration (the only implementation of the documented
   projection).  ``(0, 0)`` alone keeps the closed form bit-for-bit --
   that is the cross-backend contract of ``aberration_tensor_lg00_jax``.
2. Clamp the DEFAULT sigma-grid half-extent to the fit's s2 validity box
   measured from ``s2_image``.  Nothing is lost: the integrand vanishes
   outside the box.

THE ORACLE
----------
Analytic LG modes written out from scratch below (``_lg_oracle``), plain
Riemann quadrature (``_overlap``), verified by
(a) orthonormality to 5.3e-15, (b) unaberrated overlap == identity to
4.4e-15, (c) a mode-MIXING coma + astigmatism + trefoil phase screen
producing off-diagonals up to 0.3738 -- i.e. a generic field simply
cannot give equal overlaps for all output modes.  ``decompose_lg`` is
cross-checked against it to 8.0e-14.  Post-fix the library tensor
matches the oracle element-by-element to <= 7.3e-15 relative.
"""


import functools
import math
import warnings

import numpy as np
import pytest

WL_t3 = 1.30e-6
W_S = 20e-6
W_P = 0.05

# Non-degenerate probe: source displaced in BOTH x and y so neither
# rotational nor mirror symmetry can zero an ell != 0 channel, and an
# output waist commensurate with the actual spot so the (p, 0) family is
# genuinely resolved.
SRC_OFFAXIS = (60e-6, 25e-6)
SRC_ONAXIS = (0.0, 0.0)
W_O_PROBE = 30e-6
N_PROBE = 40  # sigma-grid size; every number below is stable in n

# A deliberately cheap canonical fit (0.3 s instead of 5 s).  Checked
# against the library-default fit: |L| agrees to 4 significant figures on
# the probe below, so nothing here depends on the coarser fit.
FIT_KW = dict(n_field=6, n_pupil=6, poly_order=4)

MODES_MIXED = [(0, 0), (1, 0), (2, 0), (0, 1), (0, -1), (1, 1), (0, 2)]
MODES_ELL0 = [(0, 0), (1, 0), (2, 0), (3, 0)]


# ===========================================================================
# The oracle -- written out from scratch, no lumenairy LG helper involved
# ===========================================================================
#   LG_{p,l}(x, y; w) = N_{p,l} (sqrt(2) r / w)^|l| Lag_p^{|l|}(2 r^2/w^2)
#                       exp(i l phi) exp(-r^2 / w^2)
#   N_{p,l} = sqrt(2 p! / (pi (p + |l|)!)) / w
#   Lag_p^a(x) = sum_{k=0}^p (-1)^k C(p + a, p - k) x^k / k!
#   <f, g> = integral conj(f) g dx dy      (plain L2, no envelope weight)

def _lag_gen(p: int, a: int, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for k in range(p + 1):
        out = out + ((-1.0) ** k * math.comb(p + a, p - k)
                     / math.factorial(k)) * x ** k
    return out


def _lg_oracle(p: int, ell: int, w: float,
               X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    a = abs(int(ell))
    r2 = X * X + Y * Y
    N = math.sqrt(2.0 * math.factorial(p)
                  / (math.pi * math.factorial(p + a))) / w
    radial = ((np.sqrt(2.0) * np.sqrt(r2) / w) ** a
              * _lag_gen(p, a, 2.0 * r2 / (w * w)))
    return (N * radial * np.exp(1j * ell * np.arctan2(Y, X))
            * np.exp(-r2 / (w * w))).astype(np.complex128)


def _overlap(mode: np.ndarray, field: np.ndarray, d: float) -> complex:
    """Direct 2-D quadrature of ``integral conj(mode) field dx dy``."""
    return complex(np.sum(np.conj(mode) * field) * d * d)


def _grid(cx: float, cy: float, extent: float, n: int):
    """Same sampling the library's sigma branch builds."""
    ax = np.linspace(cx - extent, cx + extent, n)
    ay = np.linspace(cy - extent, cy + extent, n)
    X, Y = np.meshgrid(ax, ay, indexing='xy')
    return X, Y, float(ax[1] - ax[0])


def _screen(X: np.ndarray, Y: np.ndarray, w: float) -> np.ndarray:
    """A deliberately mode-MIXING screen: coma + astigmatism + trefoil.

    Not pure defocus-on-axis -- it breaks the radial AND the azimuthal
    symmetry, so the exact overlap matrix has non-trivial off-diagonals
    in p and in ell.
    """
    u, v = X / w, Y / w
    rho2 = u * u + v * v
    return np.exp(1j * (0.9 * u * (rho2 - 0.5)          # coma, x
                        + 0.7 * (u * u - v * v)          # 0-deg astigmatism
                        + 0.4 * (u ** 3 - 3.0 * u * v * v)))  # trefoil


# ===========================================================================
# Cached library objects.  RUNTIME BUDGET: the W3-T3 probes below must stay
# cheap (they are, ~13 s for the whole file at v5.28); the W3-T3b and W4
# sections cost more because the evaluator's own σ-grid default is now
# ADAPTIVE and lands on n = 256 / 192 instead of 64 on those singlets, i.e.
# ~16x the grid per call (audit W4-T1).  Measured 170 s for the file here,
# with the four dominant tests already cost-trimmed: the accuracy check runs
# ONE design against a 512 (not 768) reference, ``_w4_extent`` reads ``w_o``
# off a cheap n=64 call (the measured waist is grid-independent), the two
# curvature tests share one ``_tensor_w4`` cache key, and the cap test's
# negative case uses a small ``sigma_grid_extent``.  Keep new σ-branch tests
# on an EXPLICIT ``sigma_grid_n`` unless the default is what is under test.
# ===========================================================================

@functools.lru_cache(maxsize=1)
def _fit():
    import lumenairy
    from lumenairy.propagators.asymptotic import fit_canonical_polynomials
    pres = lumenairy.make_singlet(R1=500e-3, R2=float('inf'), d=3e-3,
                                  glass='N-BK7', aperture=4e-3)
    pres['object_distance'] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fit_canonical_polynomials(pres, wavelength=WL_t3, **FIT_KW)


@functools.lru_cache(maxsize=8)
def _tensor(src, s2_img, modes, w_o=None, n=None):
    """``modes=None`` exercises the library's own default output_modes."""
    from lumenairy.propagators.asymptotic import aberration_tensor
    kw = {}
    if n is not None:
        kw['sigma_grid_n'] = n
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return aberration_tensor(
            _fit(), s2_image=s2_img, source_point=src,
            source_modes=[(0, 0)], pupil_modes=[(0, 0)],
            output_modes=None if modes is None else list(modes),
            w_s=W_S, w_p=W_P, w_o=w_o, **kw)


@functools.lru_cache(maxsize=8)
def _field(src, s2_img, extent, n):
    """The library field on exactly the grid the sigma branch uses."""
    from lumenairy.propagators.asymptotic import propagate_modal_asymptotic
    X, Y, d = _grid(s2_img[0], s2_img[1], extent, n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        U = propagate_modal_asymptotic(
            _fit(), source_point=src, source_amplitudes={(0, 0): 1.0 + 0.0j},
            pupil_amplitudes={(0, 0): 1.0 + 0.0j}, w_s=W_S, w_p=W_P,
            s2_grid_x=X, s2_grid_y=Y)
    return X - s2_img[0], Y - s2_img[1], d, U


def _oracle_L(src, s2_img, modes, w_o, n, extent=None):
    """Independent overlap vector ``<LG_k | U>`` on the same probe grid."""
    XL, YL, d, U = _field(src, s2_img,
                          4.0 * w_o if extent is None else extent, n)
    return [_overlap(_lg_oracle(p, l, w_o, XL, YL), U, d) for (p, l) in modes]


# ===========================================================================
# 1 -- oracle trust
# ===========================================================================

def test_w3_t3_oracle_lg_basis_is_orthonormal():
    """The oracle basis must be orthonormal before it can judge anything.
    Measured max |<LG_i|LG_j> - delta_ij| = 5.329e-15 on a 193^2 grid over
    +-6w (4.441e-15 at 1025^2, so the quadrature is converged)."""
    w = W_O_PROBE
    X, Y, d = _grid(0.0, 0.0, 6.0 * w, 193)
    stack = [_lg_oracle(p, l, w, X, Y) for (p, l) in MODES_MIXED]
    G = np.array([[_overlap(mi, mj, d) for mj in stack] for mi in stack])
    err = float(np.max(np.abs(G - np.eye(len(MODES_MIXED)))))
    assert err < 1e-10, (
        f'oracle LG basis is not orthonormal: max |G - I| = {err:.3e} '
        f'(measured 5.329e-15).  Every verdict below depends on this.')


def test_w3_t3_oracle_identity_and_screen_mixes_modes():
    """(a) the UNABERRATED overlap matrix is the identity;  (b) a real
    mode-mixing aberration (coma + astigmatism + trefoil) makes the
    overlap matrix strongly non-diagonal AND spreads the (p, 0) diagonal
    -- so ``L`` being equal for every output mode is not something a
    physical field can produce."""
    w = W_O_PROBE
    X, Y, d = _grid(0.0, 0.0, 6.0 * w, 193)
    stack = [_lg_oracle(p, l, w, X, Y) for (p, l) in MODES_MIXED]
    scr = _screen(X, Y, w)
    A = np.array([[_overlap(mi, scr * mj, d) for mj in stack]
                  for mi in stack])
    I = np.array([[_overlap(mi, mj, d) for mj in stack] for mi in stack])
    assert np.max(np.abs(I - np.eye(len(MODES_MIXED)))) < 1e-10

    off = A - np.diag(np.diag(A))
    assert float(np.max(np.abs(off))) == pytest.approx(0.3738173997,
                                                       rel=1e-6), (
        'the probe screen must actually mix modes (measured max '
        'off-diagonal 0.3738173997)')
    # (p, 0) family: 0.9122117988 / 0.5735883661 / 0.3354343695 -- these
    # differ by factors of ~2.7, NOT by 1e-17.
    diag_p0 = [abs(A[i, i]) for i in range(3)]
    assert diag_p0 == pytest.approx(
        [0.9122117988, 0.5735883661, 0.3354343695], rel=1e-6)
    assert min(abs(diag_p0[i] / diag_p0[0] - 1.0) for i in (1, 2)) > 0.3


def test_w3_t3_decompose_lg_matches_oracle_quadrature():
    """Cross-check the library's own projection utility against the
    oracle so a later disagreement can be localised.  Measured worst
    relative difference 8.021e-14."""
    from lumenairy.propagators.asymptotic_modes import decompose_lg
    w = W_O_PROBE
    X, Y, d = _grid(0.0, 0.0, 6.0 * w, 193)
    fld = _screen(X, Y, w) * _lg_oracle(0, 0, w, X, Y)
    lib = decompose_lg(fld, X, Y, w=w, p_max=2, ell_max=2)
    worst = 0.0
    for (p, l) in MODES_MIXED:
        mine = _overlap(_lg_oracle(p, l, w, X, Y), fld, d)
        worst = max(worst, abs(complex(lib[(p, l)]) - mine) / abs(mine))
    assert worst < 1e-10, f'decompose_lg vs oracle worst rel = {worst:.3e}'


# ===========================================================================
# 2 -- the defect
# ===========================================================================

def test_w3_t3_ell0_output_modes_are_not_degenerate():
    """PRE-FIX RED.  On-axis, default ``w_o``, ``output_modes =
    [(0,0), (1,0), (2,0), (3,0)]`` came back BIT-IDENTICAL (max spread
    1.963e-17 == rounding).  Post-fix these are true overlaps and the
    relative spread is 1.972879e-05 / 3.945737e-05 / 5.918574e-05 --
    small only because the DEFAULT ``w_o`` (9.829e-3 'm', derived from
    the pupil-space beam matrix, so 259x the +-half-box of the fit) is
    far wider than the image field, and a basis that barely varies over
    the field genuinely IS nearly degenerate.  With a commensurate
    ``w_o`` the same channels differ by factors of 2-3 (second half)."""
    res = _tensor(SRC_ONAXIS, (0.0, 0.0), tuple(MODES_ELL0))
    L = np.asarray(res.L).ravel()
    spread = [abs(L[i] - L[0]) / abs(L[0]) for i in range(1, len(L))]
    assert all(L[i] != L[0] for i in range(1, len(L))), (
        f'aberration_tensor returned the IDENTICAL L for every output '
        f'mode {MODES_ELL0}: {L}.  The closed-form branch samples '
        f'conj(LG_k) at ONE point, which is N_(p,0) = sqrt(2/(pi w_o^2)) '
        f'for every p.')
    assert min(spread) > 1e-6, (
        f'relative spread across the (p, 0) ladder = {spread} (measured '
        f'post-fix 1.972879e-05 / 3.945737e-05 / 5.918574e-05; pre-fix '
        f'0.0 exactly)')

    # ...and with a commensurate output waist the channels are wildly
    # different, which is what the merit layer actually needs.
    r2 = _tensor(SRC_OFFAXIS, (0.0, 0.0), tuple(MODES_ELL0),
                 W_O_PROBE, N_PROBE)
    m = np.abs(np.asarray(r2.L).ravel())
    assert max(m) / min(m) > 2.0, (
        f'|L| across the (p, 0) ladder at w_o = 30 um: {m} -- expected a '
        f'spread of several x (measured 3.2070e-09 / 8.0557e-09 / '
        f'7.1603e-09 / 5.5736e-09, max/min = 2.5119), got '
        f'max/min = {max(m) / min(m):.4f}')


def test_w3_t3_single_ell0_mode_matches_overlap_oracle():
    """PRE-FIX RED.  A SEPARATE single-mode request is where the
    degeneracy is most dangerous: ``output_modes=[(2, 0)]`` silently
    returned the piston value with no warning at all, so
    ``LGAberrationMerit(targets={(2, 0): 1.0})`` was minimising the
    Strehl amplitude.  Pre-fix this exact call returned the closed-form
    sample ``0.22466104589888336-2.019522344395369j`` against the true
    overlap ``1.3324520570537e-09+7.035197375155269e-09j`` -- relative
    error 2.838e+08."""
    got = complex(_tensor(SRC_OFFAXIS, (0.0, 0.0), ((2, 0),),
                          W_O_PROBE, N_PROBE).L[0, 0])
    ref = _oracle_L(SRC_OFFAXIS, (0.0, 0.0), [(2, 0)], W_O_PROBE,
                    N_PROBE)[0]
    rel = abs(got - ref) / abs(ref)
    assert rel < 1e-9, (
        f'single-mode L[(2,0)] = {got!r} but the independent overlap '
        f'oracle gives {ref!r} (rel = {rel:.3e}).')


def test_w3_t3_default_sigma_grid_is_not_identically_zero():
    """PRE-FIX RED.  With DEFAULT kwargs the sigma branch built its grid
    at ``+-4 w_o = 3.93e-02 m`` while the fit's s2 validity half-box is
    ``1.52e-04 m`` -- 259x outside.  ``propagate_modal_asymptotic`` is
    identically zero there, so the ENTIRE tensor (including the named
    coma / astigmatism / tilt / trefoil channels the sigma path exists
    to provide) came back exactly ``0.0 + 0.0j``."""
    res = _tensor(SRC_OFFAXIS, (0.0, 0.0), None)
    L = np.asarray(res.L)
    assert not np.all(L == 0.0), (
        'aberration_tensor returned an all-zero L with default kwargs: '
        'the default sigma grid lies entirely outside the fit box.')
    mag = np.abs(L.ravel())
    assert np.count_nonzero(mag > 1e-30) >= 3, (
        f'expected several genuinely populated channels, got |L| = {mag}')
    # the extent actually used must sit inside the fit's s2 box
    fit = _fit()
    assert 4.0 * res.w_o > fit.s2x_halfrange, (
        'premise check: this probe is only interesting because the raw '
        '4*w_o default overshoots the fit box')


# ===========================================================================
# 3 -- element-by-element agreement with the independent oracle
# ===========================================================================

def test_w3_t3_tensor_matches_overlap_oracle_offaxis_mixed():
    """The full non-degenerate probe (source off-axis in x AND y, so no
    symmetry can zero a channel).  Every element of ``L`` must be the
    complex overlap ``<LG_k | U>``.  Measured worst relative difference
    1.096e-15 here (4.883e-15 at ``sigma_grid_n = 193``).

    This one is GREEN pre-fix as well (an ell != 0 mode was already
    enough to reach the sigma branch, and the explicit ``w_o = 30 um``
    keeps the grid inside the fit box) -- it is the anchor that says the
    sigma branch itself was always right, so routing to it is the fix."""
    res = _tensor(SRC_OFFAXIS, (0.0, 0.0), tuple(MODES_MIXED),
                  W_O_PROBE, N_PROBE)
    ref = _oracle_L(SRC_OFFAXIS, (0.0, 0.0), MODES_MIXED, W_O_PROBE,
                    N_PROBE)
    for io, k in enumerate(MODES_MIXED):
        got = complex(res.L[io, 0])
        rel = abs(got - ref[io]) / abs(ref[io])
        assert rel < 1e-9, (
            f'L[{k}] = {got!r} vs independent overlap oracle '
            f'{ref[io]!r} (rel = {rel:.3e})')

    mag = np.abs(np.asarray(res.L).ravel())
    assert mag.max() / mag.min() < 5.0, (
        f'probe sanity: on this non-degenerate probe every channel should '
        f'be populated at a comparable level, got |L| = {mag}')
    # +1 and -1 tilt must differ -- proof the probe broke mirror symmetry
    i_p, i_m = MODES_MIXED.index((0, 1)), MODES_MIXED.index((0, -1))
    assert abs(complex(res.L[i_p, 0]) - complex(res.L[i_m, 0])) > 1e-10 * mag.max()


def test_w3_t3_symmetric_probe_ell_nonzero_zero_is_legitimate():
    """PHANTOM GUARD.  On the ROTATIONALLY SYMMETRIC probe (on-axis
    source through a rotationally symmetric singlet) every ell != 0
    channel is legitimately zero -- the independent oracle says so too.
    So 'the ell != 0 entries are zero' is NOT by itself evidence of a
    bug on that probe; the (p, 0) degeneracy is.  Measured
    |L_(ell!=0)| / |L_(0,0)| <= 1.32e-13 (numerical noise) while the
    (p, 0) channels sit at 1.000000 / 0.581022 / 0.337604."""
    res = _tensor(SRC_ONAXIS, (0.0, 0.0), tuple(MODES_MIXED),
                  W_O_PROBE, N_PROBE)
    ref = _oracle_L(SRC_ONAXIS, (0.0, 0.0), MODES_MIXED, W_O_PROBE, N_PROBE)
    L = np.asarray(res.L).ravel()
    base = abs(L[0])
    for io, k in enumerate(MODES_MIXED):
        if k[1] != 0:
            assert abs(L[io]) / base < 1e-10, (
                f'{k} must vanish by rotational symmetry, got '
                f'{abs(L[io]) / base:.3e}')
            assert abs(ref[io]) / abs(ref[0]) < 1e-10, (
                f'oracle disagrees that {k} vanishes -- the probe is not '
                f'symmetric after all')
    ratios = [abs(L[i]) / base for i in range(3)]
    assert ratios == pytest.approx([1.0, 0.5810, 0.3376], rel=2e-3), (
        f'the (p, 0) family must NOT be degenerate even on the symmetric '
        f'probe, got {ratios}')


# ===========================================================================
# 4 -- scope guards (must stay GREEN pre- and post-fix)
# ===========================================================================

def test_w3_t3_pure_piston_request_keeps_closed_form():
    """SCOPE GUARD.  A pure ``[(0, 0)]`` request must keep the closed-form
    value ``U(chief) * N_(0,0)(w_o)`` bit-for-bit: that is the documented
    cross-backend contract with ``aberration_tensor_lg00_jax`` (pinned by
    ``test_audit_raytrace.py`` at rel < 1e-3 and by
    ``test_niche_audit_r_guards_and_merits.py::
    test_r5_numpy_lg_merit_matches_jax_twin`` at rel 1e-6).  Measured
    residual |L - U*N_o| = 5.97e-17 on |L| = 5.66e-02."""
    from lumenairy.propagators.asymptotic import propagate_modal_asymptotic
    res = _tensor(SRC_ONAXIS, (0.0, 0.0), ((0, 0),))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        U = propagate_modal_asymptotic(
            _fit(), source_point=SRC_ONAXIS, w_s=W_S, w_p=W_P,
            s2_grid_x=np.array([[0.0]]), s2_grid_y=np.array([[0.0]]))
    N_o = math.sqrt(2.0 / (math.pi * res.w_o * res.w_o))
    got = complex(res.L[0, 0])
    assert got == pytest.approx(complex(U[0, 0]) * N_o, rel=1e-13), (
        f'the pure-(0,0) closed form moved: L = {got!r}, '
        f'U(chief)*N_o = {complex(U[0, 0]) * N_o!r}')
    # ...and it is emphatically NOT on the overlap scale -- documented,
    # not desired (unifying the two needs the JAX twin to change too).
    # Integrate over the fit box, where the whole field lives.
    ref = _oracle_L(SRC_ONAXIS, (0.0, 0.0), [(0, 0)], res.w_o, N_PROBE,
                    extent=_fit().s2x_halfrange)[0]
    assert abs(got) / abs(ref) > 1e6, (
        f'sampling-vs-overlap scale split is documented in '
        f'aberration_tensor; got |closed form| / |overlap| = '
        f'{abs(got) / abs(ref):.3e} (measured ~3.4e+08)')


def test_w3_t3_result_container_shape_and_finiteness():
    """SCOPE GUARD.  Routing more requests to the sigma branch must not
    change the container contract or introduce non-finite entries."""
    res = _tensor(SRC_OFFAXIS, (0.0, 0.0), tuple(MODES_MIXED),
                  W_O_PROBE, N_PROBE)
    assert res.L.shape == (len(res.output_modes), len(res.source_modes))
    assert res.output_modes == list(MODES_MIXED)
    assert np.all(np.isfinite(res.L.real) & np.isfinite(res.L.imag))


# ==========================================================================
# W3-4 SECTION (was test_niche_audit_w3_oracles_t4.py)
# ==========================================================================
_W3_4_ORACLE = """W3-T4 oracles: non-finite angle / retardance / orientation inputs to the
public :mod:`lumenairy.elements.polarization` entry points must RAISE, not
silently return a field of ``nan+nanj``.

Follow-up to commit 5f9d82b (audit E-L16), which guarded
``create_elliptical_polarized(ellipticity=...)`` against non-finite input and
explicitly flagged the two remaining holes as "New candidates flagged, not
fixed: orientation=NaN and apply_waveplate(retardance=NaN) unguarded".

Measured pre-fix (v5.29.0, all on a 2x2 unit scalar field, ``dx=1e-6``):

    create_elliptical_polarized(SF, dx, 0.1, nan)  -> Ex = Ey = nan+nanj  (8/8)
    create_elliptical_polarized(SF, dx, 0.1, inf)  -> Ex = Ey = nan+nanj  (8/8)
    apply_waveplate(f, nan, 0.3)                   -> Ex = Ey = nan+nanj  (8/8)
    apply_waveplate(f, inf, 0.3)                   -> Ex = Ey = nan+nanj  (8/8)
    apply_waveplate(f, pi/2, nan)                  -> Ex = Ey = nan+nanj  (8/8)
    create_linear_polarized(SF, dx, nan)           -> Ex = Ey = nan+nanj  (8/8)
    apply_polarizer / apply_rotator / apply_half_wave_plate /
    apply_quarter_wave_plate / apply_polarizing_beam_splitter, each with
    ``angle=nan`` or ``angle_deg=nan``                -> every pixel nan+nanj

30 silent NaN leaks in total, none of them raising anything.  The NaN is
invisible downstream too: ``degree_of_polarization`` reports NaN (documented
NaN-in/NaN-out) and an intensity plot is simply blank.

The angle guard lives in the ONE shared helper ``_resolve_angle``, so all six
angle-taking elements inherit it (house rule: when a guard is added, grep for
the twins).  ``retardance``, ``orientation`` and ``create_linear_polarized``'s
``angle`` are guarded in place, matching the E-L16 chi guard's style, message
shape and exception type.

Deliberately NOT guarded here (see the module report): ``apply_jones_matrix``
matrix CONTENT, and the analysis helpers fed a NaN FIELD -- those are
caller-supplied array data whose NaN-in/NaN-out behaviour is documented and
already pinned by test_niche_audit_e_polarization_inputs.py.
"""

import numpy as np
import pytest

from lumenairy.elements.polarization import (
    JonesField,
    apply_half_wave_plate,
    apply_polarizer,
    apply_polarizing_beam_splitter,
    apply_quarter_wave_plate,
    apply_rotator,
    apply_waveplate,
    create_elliptical_polarized,
    create_linear_polarized,
    degree_of_polarization,
    stokes_parameters,
)

DX = 1e-6
NONFINITE = [float('nan'), float('inf'), float('-inf')]


def _sf(n=2):
    return np.ones((n, n), dtype=complex)


def _field_t4(n=2):
    """A valid x-polarized JonesField (built with a finite angle)."""
    return create_linear_polarized(_sf(n), DX, 0.0)


# ---------------------------------------------------------------------------
# The two flagged entry points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", NONFINITE)
def test_w3_t4_elliptical_orientation_nonfinite_raises(bad):
    # Pre-fix: returned a JonesField with Ex = Ey = nan+nanj, no error.
    with pytest.raises(ValueError,
                       match=r"create_elliptical_polarized: orientation "
                             r"\(psi\) must be a finite angle in radians"):
        create_elliptical_polarized(_sf(), DX, 0.1, bad)


@pytest.mark.parametrize("bad", NONFINITE)
def test_w3_t4_waveplate_retardance_nonfinite_raises(bad):
    # Pre-fix: exp(+1j*nan) == nan -> every pixel nan+nanj, no error.
    with pytest.raises(ValueError,
                       match=r"apply_waveplate: retardance must be a finite "
                             r"phase in radians"):
        apply_waveplate(_field_t4(), bad, 0.3)


@pytest.mark.parametrize("bad", NONFINITE)
def test_w3_t4_waveplate_angle_nonfinite_raises(bad):
    with pytest.raises(ValueError,
                       match=r"apply_waveplate: angle must be a finite angle "
                             r"in radians"):
        apply_waveplate(_field_t4(), np.pi / 2, bad)


@pytest.mark.parametrize("bad", NONFINITE)
def test_w3_t4_waveplate_angle_deg_nonfinite_raises(bad):
    with pytest.raises(ValueError,
                       match=r"apply_waveplate: angle_deg must be a finite "
                             r"angle in degrees"):
        apply_waveplate(_field_t4(), np.pi / 2, angle_deg=bad)


# ---------------------------------------------------------------------------
# Siblings that inherit the shared _resolve_angle guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", NONFINITE)
def test_w3_t4_polarizer_angle_nonfinite_raises(bad):
    with pytest.raises(ValueError,
                       match=r"apply_polarizer: angle must be a finite angle "
                             r"in radians"):
        apply_polarizer(_field_t4(), bad)


@pytest.mark.parametrize("bad", NONFINITE)
def test_w3_t4_polarizer_angle_deg_nonfinite_raises(bad):
    with pytest.raises(ValueError,
                       match=r"apply_polarizer: angle_deg must be a finite "
                             r"angle in degrees"):
        apply_polarizer(_field_t4(), angle_deg=bad)


@pytest.mark.parametrize("bad", NONFINITE)
def test_w3_t4_rotator_angle_nonfinite_raises(bad):
    with pytest.raises(ValueError,
                       match=r"apply_rotator: angle must be a finite angle "
                             r"in radians"):
        apply_rotator(_field_t4(), bad)


@pytest.mark.parametrize("bad", NONFINITE)
def test_w3_t4_rotator_angle_deg_nonfinite_raises(bad):
    with pytest.raises(ValueError,
                       match=r"apply_rotator: angle_deg must be a finite "
                             r"angle in degrees"):
        apply_rotator(_field_t4(), angle_deg=bad)


def test_w3_t4_half_wave_plate_angle_nonfinite_raises():
    with pytest.raises(ValueError,
                       match=r"apply_half_wave_plate: angle must be a finite "
                             r"angle in radians"):
        apply_half_wave_plate(_field_t4(), np.nan)


def test_w3_t4_half_wave_plate_angle_deg_nonfinite_raises():
    with pytest.raises(ValueError,
                       match=r"apply_half_wave_plate: angle_deg must be a "
                             r"finite angle in degrees"):
        apply_half_wave_plate(_field_t4(), angle_deg=np.nan)


def test_w3_t4_quarter_wave_plate_angle_nonfinite_raises():
    with pytest.raises(ValueError,
                       match=r"apply_quarter_wave_plate: angle must be a "
                             r"finite angle in radians"):
        apply_quarter_wave_plate(_field_t4(), np.nan)


def test_w3_t4_quarter_wave_plate_angle_deg_nonfinite_raises():
    with pytest.raises(ValueError,
                       match=r"apply_quarter_wave_plate: angle_deg must be a "
                             r"finite angle in degrees"):
        apply_quarter_wave_plate(_field_t4(), angle_deg=np.nan)


def test_w3_t4_pbs_angle_nonfinite_raises():
    with pytest.raises(ValueError,
                       match=r"apply_polarizing_beam_splitter: angle must be "
                             r"a finite angle in radians"):
        apply_polarizing_beam_splitter(_field_t4(), np.nan)


def test_w3_t4_pbs_angle_deg_nonfinite_raises():
    with pytest.raises(ValueError,
                       match=r"apply_polarizing_beam_splitter: angle_deg must "
                             r"be a finite angle in degrees"):
        apply_polarizing_beam_splitter(_field_t4(), angle_deg=np.nan)


@pytest.mark.parametrize("bad", NONFINITE)
def test_w3_t4_linear_polarized_angle_nonfinite_raises(bad):
    # Sibling of create_elliptical_polarized's orientation -- same
    # major-axis angle, same cos/sin poisoning.
    with pytest.raises(ValueError,
                       match=r"create_linear_polarized: angle must be a "
                             r"finite angle in radians"):
        create_linear_polarized(_sf(), DX, bad)


@pytest.mark.parametrize("bad", NONFINITE)
def test_w3_t4_elliptical_ellipticity_nonfinite_still_raises(bad):
    # Guarded by 5f9d82b (E-L16); pinned again so the new sibling guard
    # cannot displace it.
    with pytest.raises(ValueError,
                       match=r"create_elliptical_polarized: ellipticity "
                             r"\(chi\) must be a finite angle"):
        create_elliptical_polarized(_sf(), DX, bad, 0.1)


def test_w3_t4_nonfinite_check_precedes_angle_conflict_check():
    # NaN vs NaN is not "close", so without the finiteness check first the
    # caller would get the misleading "conflicting angle specification"
    # message for a single non-finite value.
    with pytest.raises(ValueError,
                       match=r"apply_polarizer: angle must be a finite angle "
                             r"in radians"):
        apply_polarizer(_field_t4(), np.nan, angle_deg=np.nan)
    # A genuine conflict between two FINITE angles still reports as before.
    with pytest.raises(ValueError,
                       match=r"conflicting angle specification"):
        apply_polarizer(_field_t4(), 0.0, angle_deg=90.0)


# ---------------------------------------------------------------------------
# Physics must be untouched by the guards
# ---------------------------------------------------------------------------

def test_w3_t4_valid_input_physics_unchanged():
    """Jones / Stokes values for three known-good cases.

    Frozen values measured on v5.29.0 (Windows) before the guards were
    added; the guards only read ``float(param)`` into a throwaway local.
    Held at rel 1e-12, not bitwise: the values are trig-derived and libm
    sin/cos differ by ~1 ULP across platforms (measured: CI Linux vs
    this box on e1fd64a).  WITHIN-process bit-identity to the unguarded
    formula is still asserted exactly where both sides share one libm.
    """
    one = np.ones((1, 1), dtype=complex)

    # (a) linear at pi/3 -- create_linear_polarized's own guarded path.
    lin = create_linear_polarized(one.copy(), DX, np.pi / 3)
    assert lin.Ex[0, 0] == pytest.approx(0.5000000000000001 + 0j,
                                         rel=1e-12)
    assert lin.Ey[0, 0] == pytest.approx(0.8660254037844386 + 0j,
                                         rel=1e-12)
    # Bit-identical to the un-guarded formula (same-process libm).
    assert lin.Ex[0, 0] == one[0, 0] * np.cos(np.pi / 3)
    assert lin.Ey[0, 0] == one[0, 0] * np.sin(np.pi / 3)

    # (b) QWP with fast axis at -45 deg on x-pol -> right-circular
    #     (S3 = +1), the recipe named in create_circular_polarized's
    #     docstring.  Pins apply_waveplate's retardance path.
    qwp = apply_waveplate(create_linear_polarized(one.copy(), DX, 0.0),
                          np.pi / 2, -np.pi / 4)
    assert qwp.Ex[0, 0] == pytest.approx(
        0.5000000000000001 + 0.5000000000000001j, rel=1e-12)
    assert qwp.Ey[0, 0] == pytest.approx(
        -0.5 + 0.5000000000000001j, rel=1e-12)
    S = stokes_parameters(qwp)
    assert S['S0'][0, 0] == pytest.approx(1.0, rel=1e-12)
    assert S['S3'][0, 0] == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_allclose([S['S1'][0, 0], S['S2'][0, 0]], [0.0, 0.0],
                               rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(degree_of_polarization(qwp), [[1.0]],
                               rtol=0.0, atol=0.0)

    # (c) elliptical at chi = pi/8, psi = pi/6 -- pins BOTH guarded params
    #     of create_elliptical_polarized on their valid domain.
    ell = create_elliptical_polarized(one.copy(), DX, np.pi / 8, np.pi / 6)
    assert ell.Ex[0, 0] == pytest.approx(
        0.8001031451912656 - 0.19134171618254486j, rel=1e-12)
    assert ell.Ey[0, 0] == pytest.approx(
        0.4619397662556433 + 0.3314135740355918j, rel=1e-12)
    Se = stokes_parameters(ell)
    assert Se['S0'][0, 0] == pytest.approx(1.0000000000000002, rel=1e-12)
    assert Se['S1'][0, 0] == pytest.approx(0.353553390593274, rel=1e-12)
    assert Se['S2'][0, 0] == pytest.approx(0.6123724356957945, rel=1e-12)
    assert Se['S3'][0, 0] == pytest.approx(0.7071067811865476, rel=1e-12)
    # chi = pi/8 -> S3/S0 = sin(2 chi) = sin(pi/4).
    np.testing.assert_allclose(Se['S3'][0, 0] / Se['S0'][0, 0],
                               np.sin(np.pi / 4), rtol=0.0, atol=2e-16)


def test_w3_t4_boundary_and_degenerate_valid_angles_still_accepted():
    """Zero, negative, exact +-pi/4 chi and huge-but-finite angles pass."""
    one = np.ones((1, 1), dtype=complex)
    # Explicit zeros must not be mistaken for "unset" or for NaN.
    assert create_linear_polarized(one.copy(), DX, 0.0).Ex[0, 0] == 1 + 0j
    assert apply_polarizer(_field_t4(1), 0.0).Ex[0, 0] == 1 + 0j
    assert apply_waveplate(_field_t4(1), 0.0, 0.0).Ex[0, 0] == 1 + 0j
    # chi = +-pi/4 is the documented circular boundary (still allowed).
    for chi in (+np.pi / 4, -np.pi / 4):
        f = create_elliptical_polarized(one.copy(), DX, chi, 0.0)
        S = stokes_parameters(f)
        np.testing.assert_allclose(S['S3'][0, 0] / S['S0'][0, 0],
                                   np.sign(chi), rtol=0.0, atol=1e-15)
    # A huge but finite angle is finite: accepted, no NaN.
    big = apply_rotator(_field_t4(1), 1e12)
    assert np.isfinite(big.Ex).all() and np.isfinite(big.Ey).all()
    big_deg = create_linear_polarized(one.copy(), DX, -1e6)
    assert np.isfinite(big_deg.Ex).all()


def test_w3_t4_nan_field_data_is_still_not_rejected():
    """The guards are on PARAMETERS only -- NaN array data still flows.

    ``degree_of_polarization``'s documented NaN-in/NaN-out contract (audit
    E-L13, pinned in test_niche_audit_e_polarization_inputs.py) must survive:
    a guard that also scanned ``scalar_field`` / ``Ex`` would break it.
    """
    nan_field = JonesField(np.full((2, 2), np.nan, dtype=complex),
                           np.zeros((2, 2), dtype=complex), DX)
    assert np.isnan(degree_of_polarization(nan_field)).all()
    # A NaN scalar_field with a FINITE angle is accepted (data, not param).
    f = create_linear_polarized(np.full((2, 2), np.nan, dtype=complex),
                                DX, np.pi / 4)
    assert np.isnan(f.Ex).all()


# ===========================================================================
# W3-3b SECTION -- the DEFAULT output waist ``w_o`` of the sigma path
# ===========================================================================
_W3_3B_ORACLE = """W3-T3b -- the DEFAULT ``w_o`` of the sigma-integration path.

THE FINDING (flagged as a follow-up by W3-T3, then hit CI)
-----------------------------------------------------------
``aberration_tensor``'s default ``w_o`` was ``1/sqrt(lambda_max(Re M))``.
``M = J^T J / w_s^2 + I / w_p^2 - i pi H_phi`` with ``J = ds1/dv2``
[m per direction-cosine], so ``M`` is in 1/direction-cosine^2 and that
expression is an ANGLE -- the effective pupil acceptance -- used as an
image-plane LENGTH.  Being dimensionally wrong its error had no fixed
sign: measured 1.011644e-04 "m" against a true field waist of 1.552040e-03
m (15.3x too NARROW, so the ``4*w_o`` sigma grid sampled only the flat
central ~10 % of the field) on the validation singlet at w_p = 0.02, and
255x too WIDE (grid entirely outside the fit's validity box, every entry
of L exactly 0.0) at w_p = 0.05.

Consequence, on the validation harness's own discriminator
(``LGAberrationMerit``, targets={(2,0): 1.0}, 17 % curvature change
R1 = 51.5 -> 60 mm):

    committed tree   merit 2.506416e-13 vs 2.516524e-13 -> 4.0e-3 relative
    W3-T3b           merit 9.0968975e-14 vs 7.1975598e-14 -> 2.088e-1

WHY IT MUST BE MEASURED, NOT MODELLED
-------------------------------------
The image-plane width is dominated by the DEFOCUS / aberration blur, which
lives in the sigma<->v coupling, not in the pupil-space Hessian.  Across
the same two designs the TRUE waist moves 1.558851e-03 -> 2.676815e-03 m
(+71.7 %) while every ``M``-only construction is flat:

    1/sqrt(lambda_max(Re M))            1.0116e-04 -> 1.0086e-04  (-0.3 %)
    lambda*sqrt(lambda_max(Re M))/pi    4.1219e-03 -> 4.1342e-03  (+0.3 %)
    lambda/(pi w_p)                     2.0849e-05 -> 2.0849e-05  ( 0.0 %)

A basis pinned to a design-independent scale makes every merit channel
design-independent too -- exactly the CI symptom.  So the default is the
field's own intensity second moment (D4sigma/2) on a coarse probe over the
fit's s2 validity box.  Measured convergence against a 201x201 reference:
n=16 4.9e-2, n=24 4.6e-2, n=32 4.4e-3, n=48 1.0e-3 relative -- for ~25 %
of one 64x64 projection grid.

WHAT IS DELIBERATELY UNCHANGED
------------------------------
The pure-[(0, 0)] closed form keeps ``1/sqrt(lambda_max(Re M))``
BIT-FOR-BIT: there ``w_o`` is not a length at all, only the
``sqrt(2/(pi w_o^2))`` normalisation of a point sample, and it is the
cross-backend contract of ``aberration_tensor_lg00_jax``.  Verified
bit-identical against BOTH 7ea2eb9 and e1fd64a, as is every explicit
``w_o=`` call on either branch (12-case matrix).
"""

_WL_T3B = 1.31e-6
_WS_T3B, _WP_T3B = 20e-6, 0.02
_FITKW_T3B = dict(source_box_half=20e-6, pupil_box_half=0.02,
                  n_field=6, n_pupil=6, poly_order=4)


def _singlet_t3b(R1):
    import lumenairy
    p = lumenairy.make_singlet(R1, float('inf'), 4.1e-3, 'N-BK7',
                               aperture=12.0e-3)
    p['object_distance'] = 200e-3
    return p


@functools.lru_cache(maxsize=4)
def _fit_t3b(R1):
    from lumenairy.propagators.asymptotic import fit_canonical_polynomials
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fit_canonical_polynomials(_singlet_t3b(R1),
                                         wavelength=_WL_T3B, **_FITKW_T3B)


@functools.lru_cache(maxsize=8)
def _tensor_t3b(R1, modes, w_o=None):
    from lumenairy.propagators.asymptotic import aberration_tensor
    fit = _fit_t3b(R1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return aberration_tensor(
            fit, s2_image=(fit.s2x_centre, fit.s2y_centre),
            source_point=(0.0, 0.0), source_modes=[(0, 0)],
            pupil_modes=[(0, 0)], output_modes=list(modes),
            w_s=_WS_T3B, w_p=_WP_T3B, w_o=w_o)


def _legacy_w_o(R1):
    """The pre-W3-T3b pupil-scale default, recomputed from scratch."""
    from lumenairy.propagators.asymptotic import solve_envelope_stationary
    from lumenairy.propagators.asymptotic_aberration_tensor import _compute_M_b
    fit = _fit_t3b(R1)
    s2 = (fit.s2x_centre, fit.s2y_centre)
    v_star, _, _ = solve_envelope_stationary(
        fit, s2, (0.0, 0.0), w_s=_WS_T3B, w_p=_WP_T3B, v2_centre=(0.0, 0.0))
    M = _compute_M_b(fit, s2[0], s2[1], v_star[0], v_star[1], 0.0, 0.0,
                     _WS_T3B, _WP_T3B, 0.0, 0.0)[0]
    return 1.0 / math.sqrt(float(np.linalg.eigvalsh(np.real(M)).max()))


def test_w3_t3b_lg_merit_responds_to_a_curvature_change():
    """PRE-FIX FAILURE -- the validation harness's own discriminator
    (``validation/propagators/test_asymptotic.py``) as a unit test.

    A 17 % curvature change must move the (2, 0) merit channel.  Measured:
    committed tree 2.506416e-13 vs 2.516524e-13 (4.0e-3 relative -- and the
    validation's absolute ``> 1e-12`` floor, calibrated when (2, 0) was
    still bit-identical to the ~1e+02 piston point sample, cannot be met by
    ANY correct overlap-scale value); post-fix 9.0968975e-14 vs
    7.1975598e-14 = 2.0879e-01 relative.
    """
    import lumenairy

    class _Ctx:
        wavelength = _WL_T3B
        N = 64
        dx = 20e-6

    merit = lumenairy.LGAberrationMerit(
        targets={(2, 0): 1.0}, field_points=[(0.0, 0.0)],
        w_s=_WS_T3B, w_p=_WP_T3B, fit_kwargs=_FITKW_T3B)
    ctx_a, ctx_b = _Ctx(), _Ctx()
    ctx_a.prescription = _singlet_t3b(51.5e-3)
    ctx_b.prescription = _singlet_t3b(60.0e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        val_a = merit.evaluate(ctx_a)
        val_b = merit.evaluate(ctx_b)
    assert math.isfinite(val_a) and math.isfinite(val_b)
    assert val_a > 0.0 and val_b > 0.0
    rel = abs(val_a - val_b) / max(val_a, val_b)
    assert rel > 1e-2, (
        f'LG merit is curvature-insensitive: {val_a:.6e} vs {val_b:.6e}, '
        f'relative {rel:.3e} (pre-W3-T3b 4.0e-3, W3-T3b 2.088e-1, '
        f'W4-T1 4.123e-1)')
    # Pin the measured values so the channel cannot silently rescale.
    #
    # v5.29 (audit W4-T1) RE-MEASURED at the ADAPTIVE sigma-grid default.
    # The frozen numbers moved because the default n did: 64 -> 256 (R1 =
    # 51.5) and 64 -> 192 (R1 = 60), which is the aliasing fix landing.
    #
    #   quantity              W3-T3b (n=64)   W4-T1 (adaptive)
    #   val_a (R1 = 51.5)     9.0968975e-14   8.8334897780e-14   n = 256
    #   val_b (R1 = 60.0)     7.1975598e-14   5.1912550770e-14   n = 192
    #   design response       2.088e-1        4.1232e-1
    #
    # Tolerance 2e-2 RELATIVE, not bit-level.  Measured inputs: the value is
    # exactly reproducible IN-process (0.0 spread over 3 repeats), the
    # W3-T3b-era cross-platform drift of this chirp integral was 3.1e-3
    # (CI Linux 9.068754e-14 vs Windows 9.0968975e-14 on 74cf31b), and the
    # sensitivity to a one-rung shift of the σ ladder -- the amplification
    # channel a platform's fit-coefficient ulps would act through -- is
    # 6.6e-3 (a, 256 vs 384) and 1.4e-2 (b, 192 vs 256).  2e-2 covers all
    # three with margin and still sits 20x below the design response, so it
    # separates the two designs and catches any rescale of the channel.
    assert abs(val_a - 8.8334897780e-14) < 2e-2 * 8.8334897780e-14
    assert abs(val_b - 5.1912550770e-14) < 2e-2 * 5.1912550770e-14
    # The RESPONSE is the physics and is far more stable in n than either
    # value: measured 2.892e-1 at n=128, 4.021e-1 at 192, 4.040e-1 at 256,
    # 3.920e-1 at 384, 3.795e-1 at 512.  Band it generously.
    assert 0.25 < rel < 0.55, (
        f'design response {rel:.6e} left the measured n>=128 band '
        f'[2.89e-1, 4.04e-1]')


def test_w3_t3b_curvature_response_survives_a_finer_sigma_grid():
    """The curvature response must be PHYSICS, not the default grid's
    aliasing.  Re-measure the same two designs at n = 128 (4x the default
    grid's samples): the response strengthens rather than collapsing
    (measured relative response 2.09e-1 at n=64, 2.90e-1 at n=128,
    4.04e-1 at n=256), so no part of the discriminator rests on the
    under-resolution documented for the default."""
    vals = []
    for R1 in (51.5e-3, 60.0e-3):
        from lumenairy.propagators.asymptotic import aberration_tensor
        fit = _fit_t3b(R1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = aberration_tensor(
                fit, s2_image=(fit.s2x_centre, fit.s2y_centre),
                source_point=(0.0, 0.0), source_modes=[(0, 0)],
                pupil_modes=[(0, 0)], output_modes=[(0, 0), (2, 0)],
                w_s=_WS_T3B, w_p=_WP_T3B, sigma_grid_n=128)
        vals.append(abs(complex(res.L[1, 0])) ** 2)
    rel = abs(vals[0] - vals[1]) / max(vals)
    assert rel > 1e-2, (
        f'response vanishes on a finer grid ({vals[0]:.4e} vs {vals[1]:.4e}, '
        f'relative {rel:.3e}) -- the n=64 response would then be aliasing')


@pytest.mark.parametrize('R1,expect_w_o,expect_room', [
    (51.5e-3, 1.5520401e-03, 4.0343032e-03),
    (60.0e-3, 2.6763308e-03, 4.0475961e-03),
])
def test_w3_t3b_default_w_o_is_the_measured_image_plane_waist(
        R1, expect_w_o, expect_room):
    """The sigma-path default is the field's own waist -- an image-plane
    LENGTH well inside the validity box -- not the pupil acceptance."""
    from lumenairy.propagators.asymptotic_aberration_tensor import (
        _s2_validity_room,
    )
    fit = _fit_t3b(R1)
    s2 = (fit.s2x_centre, fit.s2y_centre)
    room = _s2_validity_room(fit, s2[0], s2[1])
    assert abs(room - expect_room) < 1e-9
    res = _tensor_t3b(R1, ((0, 0), (2, 0)))
    # rel 1e-2, not abs 1e-9: the D4sigma probe integrates a chirped
    # field, drifting 3.4e-4 relative across platforms (measured: CI
    # Linux 2.6754185e-3 vs the Windows-frozen 2.6763308e-3 on 74cf31b);
    # the broken pupil-scale default is 26x off, far outside 1e-2.
    assert abs(res.w_o - expect_w_o) < 1e-2 * expect_w_o
    legacy = _legacy_w_o(R1)
    assert 0.05 * room < res.w_o < room
    assert res.w_o > 10.0 * legacy, (
        f'default w_o {res.w_o:.6e} collapsed back onto the pupil-scale '
        f'value {legacy:.6e}')


def test_w3_t3b_default_w_o_tracks_defocus_where_M_alone_cannot():
    """The discriminator for "measured, not modelled": between the two
    designs the default must move like the true field waist (+71.7 %), not
    like any function of the pupil-space beam matrix (< 0.3 %)."""
    w_a = _tensor_t3b(51.5e-3, ((0, 0), (2, 0))).w_o
    w_b = _tensor_t3b(60.0e-3, ((0, 0), (2, 0))).w_o
    moved = abs(w_b - w_a) / w_a
    assert moved > 0.5, f'default w_o barely moved ({moved:.3e})'
    leg_a, leg_b = _legacy_w_o(51.5e-3), _legacy_w_o(60.0e-3)
    assert abs(leg_b - leg_a) / leg_a < 0.01       # the M-only scale is flat
    dif_a = _WL_T3B / (math.pi * leg_a)            # ... and its diffraction
    dif_b = _WL_T3B / (math.pi * leg_b)            #     image too
    assert abs(dif_b - dif_a) / dif_a < 0.01


def test_w3_t3b_default_w_o_matches_an_independent_second_moment():
    """The probe is a D4sigma/2 estimator: for an amplitude exp(-r^2/w^2)
    the intensity has per-axis variance w^2/4.  Re-measure independently on
    a 4x finer grid (128 vs the probe's 32); require 5 % -- a basis scale
    needs no better, and the probe converges (4.4e-3 at n=32 against a
    201x201 reference)."""
    from lumenairy.propagators.asymptotic import propagate_modal_asymptotic
    from lumenairy.propagators.asymptotic_aberration_tensor import (
        _s2_validity_room,
    )
    for R1 in (51.5e-3, 60.0e-3):
        fit = _fit_t3b(R1)
        s2 = (fit.s2x_centre, fit.s2y_centre)
        ext = 0.98 * _s2_validity_room(fit, s2[0], s2[1])
        ax = np.linspace(s2[0] - ext, s2[0] + ext, 128)
        ay = np.linspace(s2[1] - ext, s2[1] + ext, 128)
        X, Y = np.meshgrid(ax, ay, indexing='xy')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            U = propagate_modal_asymptotic(
                fit, source_point=(0.0, 0.0),
                source_amplitudes={(0, 0): 1.0 + 0.0j},
                pupil_amplitudes={(0, 0): 1.0 + 0.0j},
                w_s=_WS_T3B, w_p=_WP_T3B, s2_grid_x=X, s2_grid_y=Y)
        inten = np.abs(U) ** 2
        tot = inten.sum()
        lx, ly = X - s2[0], Y - s2[1]
        cx = (inten * lx).sum() / tot
        cy = (inten * ly).sum() / tot
        var = ((inten * ((lx - cx) ** 2 + (ly - cy) ** 2)).sum() / tot) / 2.0
        w_ref = 2.0 * math.sqrt(var)
        w_lib = _tensor_t3b(R1, ((0, 0), (2, 0))).w_o
        assert abs(w_lib - w_ref) / w_ref < 0.05, (
            f'R1={R1}: default {w_lib:.6e} vs independent {w_ref:.6e}')


def test_w3_t3b_sigma_overlaps_still_match_the_oracle_at_the_default():
    """W3-T3's independent from-scratch LG quadrature must still reproduce
    the library tensor with the DEFAULT (measured) waist in force -- the
    fix moved the basis scale, not the projection."""
    from lumenairy.propagators.asymptotic import propagate_modal_asymptotic
    from lumenairy.propagators.asymptotic_aberration_tensor import (
        _s2_validity_room,
    )
    modes = ((0, 0), (1, 0), (2, 0))
    for R1 in (51.5e-3, 60.0e-3):
        res = _tensor_t3b(R1, modes)
        fit = _fit_t3b(R1)
        s2 = (fit.s2x_centre, fit.s2y_centre)
        # Exactly the grid the library used: 4*w_o clamped to the room, at
        # the σ-grid size the library actually chose.  v5.29 (audit W4-T1):
        # read it off the result instead of hard-coding 64 -- the default is
        # now ADAPTIVE (256 / 192 on these two designs), and a hard-coded 64
        # made this oracle compare two different quadratures (it read
        # rel 3.4e-1, entirely grid mismatch).
        extent = min(4.0 * res.w_o, _s2_validity_room(fit, s2[0], s2[1]))
        assert res.sigma_grid_n is not None
        X, Y, d = _grid(s2[0], s2[1], extent, int(res.sigma_grid_n))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            U = propagate_modal_asymptotic(
                fit, source_point=(0.0, 0.0),
                source_amplitudes={(0, 0): 1.0 + 0.0j},
                pupil_amplitudes={(0, 0): 1.0 + 0.0j},
                w_s=_WS_T3B, w_p=_WP_T3B, s2_grid_x=X, s2_grid_y=Y)
        XL, YL = X - s2[0], Y - s2[1]
        for i, (p, ell) in enumerate(modes):
            want = _overlap(_lg_oracle(p, ell, res.w_o, XL, YL), U, d)
            got = complex(res.L[i, 0])
            rel = abs(got - want) / max(abs(want), 1e-300)
            assert rel < 1e-12, (
                f'R1={R1} mode {(p, ell)}: library {got:.6e} vs oracle '
                f'{want:.6e} (rel {rel:.3e})')


def test_w3_t3b_pure_lg00_default_is_bit_for_bit_unchanged():
    """MUST NOT MOVE.  On the closed-form branch ``w_o`` is only the
    ``sqrt(2/(pi w_o^2))`` normalisation of a point sample, and it is the
    cross-backend contract of ``aberration_tensor_lg00_jax``.  WITHIN-
    process bit-identity vs 7ea2eb9 and e1fd64a was proven by hex-diff at
    fix time; the frozen Windows values themselves drift 8.9e-11 relative
    on CI Linux (eigensolve ulps through the fit, measured on 74cf31b),
    so the cross-platform pin is rel 1e-8 -- any actual change to the
    closed-form path moves these by orders more."""
    for R1, w_want, want in (
            (51.5e-3, 1.0116441690e-04,
             1.544807582649e+01 + 3.188059447022e+00j),
            (60.0e-3, 1.0086220541e-04,
             -6.570638987023e+00 - 1.439184599267e+01j)):
        res = _tensor_t3b(R1, ((0, 0),))
        # the default IS the legacy pupil-scale formula, exactly
        assert res.w_o == _legacy_w_o(R1)
        assert abs(res.w_o - w_want) < 1e-6 * w_want
        got = complex(res.L[0, 0])
        assert abs(got - want) / abs(want) < 1e-8, f'{got!r} != {want!r}'


def test_w3_t3b_explicit_w_o_is_honoured_verbatim_on_both_branches():
    """Explicit ``w_o=`` callers are untouched by the default change
    (verified bit-identical against e1fd64a on a 12-case matrix)."""
    for modes in (((0, 0),), ((0, 0), (2, 0)), ((0, 0), (1, 1), (0, 2))):
        res = _tensor_t3b(51.5e-3, modes, 7.5e-4)
        assert res.w_o == 7.5e-4


def test_w3_t3b_probe_helper_is_robust_to_a_dead_field():
    """The measured default must degrade gracefully, never to 0 / NaN: a
    propagator that yields no usable field returns ``None`` from the probe,
    and ``aberration_tensor`` then falls back to a quarter of the validity
    room (so ``4*w_o`` spans the box exactly)."""
    from lumenairy.propagators.asymptotic_aberration_tensor import (
        _measure_image_plane_waist,
    )
    fit = _fit_t3b(51.5e-3)
    s2 = (fit.s2x_centre, fit.s2y_centre)

    def _dead(*a, **kw):
        return np.zeros_like(kw['s2_grid_x'], dtype=complex)

    def _nan(*a, **kw):
        return np.full(kw['s2_grid_x'].shape, np.nan + 1j * np.nan)

    def _boom(*a, **kw):
        raise ValueError('propagator exploded')

    for fn in (_dead, _nan, _boom):
        assert _measure_image_plane_waist(
            fit, s2[0], s2[1], (0.0, 0.0), {(0, 0): 1.0 + 0.0j},
            _WS_T3B, _WP_T3B, (0.0, 0.0), fn) is None


def test_w3_t3b_jax_twin_default_w_o_tracks_numpy():
    """The (0, 0) default is a SHARED convention: the JAX twin hardcodes
    the same expression, so with no explicit ``w_o`` the two backends must
    agree (measured 1.011644168976e-04 both sides, 4.0e-16 relative)."""
    pytest.importorskip('jax', reason='JAX not installed')
    import jax
    jax.config.update('jax_enable_x64', True)
    from lumenairy.propagators.asymptotic import (
        aberration_tensor_lg00_jax,
        solve_envelope_stationary,
    )
    fit = _fit_t3b(51.5e-3)
    s2 = (fit.s2x_centre, fit.s2y_centre)
    v_star, _, _ = solve_envelope_stationary(
        fit, s2, (0.0, 0.0), w_s=_WS_T3B, w_p=_WP_T3B, v2_centre=(0.0, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res_j = aberration_tensor_lg00_jax(
            fit, s2, v_star, source_point=(0.0, 0.0),
            w_s=_WS_T3B, w_p=_WP_T3B, w_o=None, v2_centre=(0.0, 0.0),
            return_result=True)
    w_np = _tensor_t3b(51.5e-3, ((0, 0),)).w_o
    rel = abs(float(res_j.w_o) - w_np) / w_np
    assert rel < 1e-13, (
        f'jax default w_o {float(res_j.w_o):.12e} drifted from numpy '
        f'{w_np:.12e} (rel {rel:.3e}) -- the two defaults are one contract')


# ==========================================================================
# W4 SECTION -- closing the three items W3-T3b flagged but did not fix
# ==========================================================================
_W4_ORACLE = """W4 oracles for the LG aberration-tensor stack.  Three items were MEASURED
and FLAGGED in 74cf31b but left open; this section pins their closure.

W4-T1 -- sigma-grid aliasing, ADAPTIVE default
----------------------------------------------
The image-plane field is a chirp: dPhi/ds2 = v*/lambda (Phi in waves), so its
local fringe rate at offset sigma is |v*(sigma)|/lambda cycles/m and Nyquist
on n samples across 2*extent needs n >= 4*extent*v_max/lambda.  Three
INDEPENDENT measurements agree that this is the right model:

  * the amplitude-weighted 99.9th percentile of the field's own local fringe
    rate, from the unwrapped phase of a 1-D cut, CONVERGED in the cut's own
    sampling (4097 / 16385 / 65537 points -> 4.4938e+3 / 4.4932e+3 /
    4.4941e+3 cycles/m at R1 = 51.5 mm);
  * max |v*|/lambda over the pixels the propagator declares ALIVE:
    4.4290e+3 cycles/m -- 1.5 % from the above (0.5 % at R1 = 60 mm);
  * the 1-D power spectrum's 99th percentile, 5.8e+3 cycles/m.

(The raw MAXIMUM phase gradient is NOT bandwidth -- it diverges with the
cut's sampling, 1.19e5 -> 3.79e5 -> 6.37e5, being the pi-jump at the
amplitude zeros.)

v_max is taken from the fit's v2 VALIDITY BOX because
``propagate_modal_asymptotic`` zeroes every pixel whose Newton solve leaves
it, making the box a STRICT bound on the alive region -- and it costs no
extra propagate.  It is loose (the alive region's actual max |v*| is 0.00580
/ 0.00891 vs box 0.0199 / 0.01415) and conservative in the correct direction
for a default that used to UNDER-resolve.

Required n: 246 / 175 on the two validation singlets -> ladder 256 / 192,
against the old flat 64.  Measured worst-channel error against n = 768:
1.58e-1 -> 1.37e-2 (11.6x better) and 1.51e-1 -> 2.94e-2 (5.1x).

W4-T2 -- the (2,0) channel's basis-design oscillation
-----------------------------------------------------
FIXED, behind an opt-in flag.  The field is a strongly CURVED wavefront
(-30.1 waves of quadratic phase at the sigma-grid edge at R1 = 51.5 mm,
equivalent radius -0.206 m, sweeping to -21.5 waves at R1 = 60 mm), so a
REAL-waist LG basis needs far more than three (p, 0) modes to represent it
and the truncated (2, 0) coefficient is an interference residue whose phase
rotates with the design.  Matching the basis curvature (a complex-q basis:
the same radial profile times exp(+i*pi*sigma^T C sigma/lambda) with
C = dv*/dsigma) removes the rotation.  MEASURED, eight designs
R1 = 51.5 -> 60 mm at sigma_grid_n = 256:

    basis                       sign flips   ptp/mean
    flat (the default)            5 of 6       0.307
    curvature-matched             0 of 6       2.197

i.e. STRICTLY MONOTONE, and a 7x stronger discriminator.  The rejected
alternative -- "exclude defocus from the channel" by removing the (1, 0)
projection -- is a no-op in an orthonormal basis and moves |L(2,0)| by only
0.4 % - 20 % on the discrete grid (Gram off-diagonal 4.4e-4 .. 0.219), still
leaving 2 of 6 sign flips.  The rotating PHASE, not the defocus AMPLITUDE,
is what makes the channel non-smooth.

W4-T3 -- the local jax-fit-match red
------------------------------------
Diagnosed as the JAX twin's least-squares ESTIMATOR, not JAX: reproducing
``_differentiable_lstsq``'s algorithm in pure NumPy on the captured 1024x70
design matrix gives the same 4.49e-02.  The Tikhonov floor was 4.25x LARGER
than sigma_min^2, and even at floor 0 the normal equations sit ~1e-3 from
``lstsq`` because they square a 6.5e+06 condition number.  QR fixes both.
"""

_W4_R1S = (51.5e-3, 53.9e-3, 56.3e-3, 58.7e-3, 60.0e-3)


def _w4_extent(R1):
    """The σ half-extent the library's DEFAULT would build.

    Deliberately obtains ``w_o`` from a ``sigma_grid_n=64`` call: the
    measured waist comes from its own coarse probe (``_W_O_PROBE_N``) and is
    therefore independent of the σ grid, so this is the same number for a
    fraction of the cost of the adaptive 256/192 grid.
    """
    from lumenairy.propagators.asymptotic_aberration_tensor import (
        _s2_validity_room,
    )
    fit = _fit_t3b(R1)
    res = _tensor_w4(R1, ((0, 0), (2, 0)), 64)
    return min(4.0 * res.w_o,
               _s2_validity_room(fit, fit.s2x_centre, fit.s2y_centre))


@functools.lru_cache(maxsize=64)
def _tensor_w4(R1, modes, n=None, n_max=None, curv=False):
    from lumenairy.propagators.asymptotic import aberration_tensor
    fit = _fit_t3b(R1)
    kw = {}
    if n is not None:
        kw['sigma_grid_n'] = n
    if n_max is not None:
        kw['sigma_grid_n_max'] = n_max
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return aberration_tensor(
            fit, s2_image=(fit.s2x_centre, fit.s2y_centre),
            source_point=(0.0, 0.0), source_modes=[(0, 0)],
            pupil_modes=[(0, 0)], output_modes=list(modes),
            w_s=_WS_T3B, w_p=_WP_T3B,
            curvature_matched_basis=curv, **kw)


# ---------------------------------------------------------------------------
# W4-T1 -- adaptive sigma-grid default
# ---------------------------------------------------------------------------

def test_w4_t1_chirp_formula_is_the_measured_bandwidth():
    """The Nyquist requirement must come out of the FIT, at call time, and
    must reproduce the independently measured bandwidth.

    ``v_max`` is the fit's v2 box extremum, which is a strict bound because
    ``propagate_modal_asymptotic``'s ``in_box_v`` mask zeroes every pixel
    whose ``v2*`` leaves it -- so a NON-ZERO pixel cannot chirp faster.
    Measured n_req: 246 (R1 = 51.5 mm) and 175 (R1 = 60 mm).
    """
    from lumenairy.propagators.asymptotic_aberration_tensor import (
        _required_sigma_grid_n,
        _sigma_chirp_v_max,
    )
    for R1, n_want, v_want in ((51.5e-3, 246, 1.989729e-02),
                               (60.0e-3, 175, 1.415340e-02)):
        fit = _fit_t3b(R1)
        v_max = _sigma_chirp_v_max(fit)
        assert abs(v_max - v_want) < 1e-4 * v_want, (
            f'R1={R1}: v_max {v_max:.6e} != measured {v_want:.6e}')
        # strict-bound property: the box extremum, not something smaller
        assert v_max >= fit.v2x_halfrange - 1e-15
        assert v_max >= fit.v2y_halfrange - 1e-15
        n_req = _required_sigma_grid_n(fit, _w4_extent(R1))
        assert n_req == n_want, (
            f'R1={R1}: chirp formula asks for n={n_req}, measured {n_want} '
            f'(4*extent*v_max/lambda)')
        # ...and it IS the formula, not a constant
        assert n_req == max(64, math.ceil(
            4.0 * _w4_extent(R1) * v_max / fit.wavelength))


def test_w4_t1_default_sigma_grid_n_is_adaptive_and_on_the_ladder():
    """PRE-FIX RED.  The default was a flat 64 for every design; it is now
    the chirp requirement rounded up onto the documented cost ladder --
    256 (R1 = 51.5) and 192 (R1 = 60), i.e. it MOVES with the design."""
    from lumenairy.propagators.asymptotic_aberration_tensor import (
        _SIGMA_GRID_LADDER,
        _sigma_grid_n_on_ladder,
    )
    got = []
    for R1, n_want in ((51.5e-3, 256), (60.0e-3, 192)):
        res = _tensor_w4(R1, ((0, 0), (1, 0), (2, 0)))
        assert res.sigma_grid_n == n_want, (
            f'R1={R1}: adaptive default resolved to {res.sigma_grid_n}, '
            f'measured {n_want} (pre-fix: 64 for every design)')
        assert res.sigma_grid_n in _SIGMA_GRID_LADDER
        got.append(res.sigma_grid_n)
    assert got[0] != got[1], (
        'the default must DEPEND on the design -- a flat default is the '
        'defect W4-T1 closes')
    # the ladder rounds UP, never down
    for n in (1, 64, 65, 96, 97, 246, 256, 257, 1025, 3000):
        assert _sigma_grid_n_on_ladder(n) >= n


def test_w4_t1_adaptive_default_beats_the_old_flat_64():
    """The point of the change.  Worst error over the ``(p, 0)`` ladder
    ``[(0,0), (1,0), (2,0), (3,0)]`` against a finer reference:

        ref    R1        n_adaptive   err adaptive   err n=64    ratio
        768    51.5 mm      256        2.1932e-02   1.5794e-01   7.2x
        768    60.0 mm      192        2.9391e-02   1.7560e-01   6.0x
        512    51.5 mm      256        2.3124e-02   1.6251e-01   7.0x
        512    60.0 mm      192        2.8145e-02   1.7492e-01   6.2x

    Run on R1 = 60 mm with ref = 512 -- the same verdict at a third of the
    wall time, and the more informative of the two designs because its
    adaptive ``n`` is 192, i.e. chosen by the FORMULA rather than clipped by
    the cap (R1 = 51.5's 256 is pinned by
    ``test_w4_t1_default_sigma_grid_n_is_adaptive_and_on_the_ladder``).

    NO tight value pins here on purpose: these are DIFFERENCES of chirp
    quadratures, the most platform-sensitive quantity in this stack (the
    W3-T3b-era pins broke CI on exactly that, at 3.4e-4 on a much tamer
    number).  The load-bearing claims are the ORDERING and the
    order-of-magnitude bands, both with >2x headroom above.
    """
    modes = tuple(MODES_ELL0)
    R1 = 60.0e-3
    ref = np.abs(np.asarray(_tensor_w4(R1, modes, 512).L).ravel())
    ad = np.abs(np.asarray(_tensor_w4(R1, modes).L).ravel())
    lo = np.abs(np.asarray(_tensor_w4(R1, modes, 64).L).ravel())
    e_ad = float(np.max(np.abs(ad - ref) / ref))
    e_64 = float(np.max(np.abs(lo - ref) / ref))
    assert e_ad < 0.5 * e_64, (
        f'adaptive-default error {e_ad:.4e} is not clearly better than '
        f'n=64 ({e_64:.4e}) -- measured 2.81e-2 vs 1.75e-1')
    assert e_64 > 5e-2, (
        f'premise: the old flat n=64 must actually be ~1e-1 off on this '
        f'ladder, got {e_64:.4e} (measured 1.75e-1)')
    assert e_ad < 6e-2, (
        f'the adaptive default must land in the low 1e-2 band, got '
        f'{e_ad:.4e} (measured 2.81e-2)')


def test_w4_t1_explicit_sigma_grid_n_64_is_the_pre_fix_default_bit_for_bit():
    """SCOPE GUARD.  ``sigma_grid_n=64`` must reproduce the pre-W4-T1
    default EXACTLY -- that is the escape hatch for the optimiser loop the
    old default was tuned for.  Checked against the merit values frozen on
    74cf31b at the old default (9.0968975e-14 / 7.1975598e-14), which those
    constants carry to 8 significant figures: measured agreement 1.9e-9 /
    4.6e-9, i.e. the full precision of the frozen decimals."""
    for R1, want in ((51.5e-3, 9.0968975e-14), (60.0e-3, 7.1975598e-14)):
        res = _tensor_w4(R1, ((0, 0), (2, 0)), 64)
        assert res.sigma_grid_n == 64
        got = abs(complex(res.L[1, 0])) ** 2
        assert abs(got - want) < 1e-8 * want, (
            f'R1={R1}: sigma_grid_n=64 gives {got:.10e}, the pre-W4-T1 '
            f'default was {want:.7e} -- the explicit path must not move')


def test_w4_t1_cap_truncation_warns_with_both_numbers():
    """When the cap bites, the caller must be TOLD -- and told both the
    required ``n`` and the cap, so the message is actionable."""
    from lumenairy.propagators.asymptotic import aberration_tensor
    fit = _fit_t3b(51.5e-3)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = aberration_tensor(
            fit, s2_image=(fit.s2x_centre, fit.s2y_centre),
            source_point=(0.0, 0.0), source_modes=[(0, 0)],
            pupil_modes=[(0, 0)], output_modes=[(0, 0), (2, 0)],
            w_s=_WS_T3B, w_p=_WP_T3B, sigma_grid_n_max=64)
    assert res.sigma_grid_n == 64, (
        f'the cap must bind: got n={res.sigma_grid_n}')
    msgs = [str(w.message) for w in rec
            if w.category is UserWarning
            and 'sigma_grid_n_max' in str(w.message)]
    assert len(msgs) == 1, (
        f'expected exactly one cap-truncation UserWarning, got {len(msgs)}')
    m = msgs[0]
    assert '246' in m, f'required n missing from the warning: {m!r}'
    assert '64' in m, f'the cap missing from the warning: {m!r}'
    # user-visible text must survive a cp1252 console (a 'sigma' glyph here
    # raised UnicodeEncodeError on this box's default stdout)
    m.encode('ascii')

    # ...and NO warning when the cap does not bite.  Shrink the σ extent so
    # the chirp requirement itself drops below the floor (n_req = 4*5e-4*
    # 0.0199/1.31e-6 = 30 -> clamped to 64, well under the 256 cap) -- which
    # also keeps this negative case on a cheap 64-point grid.
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter('always')
        res2 = aberration_tensor(
            fit, s2_image=(fit.s2x_centre, fit.s2y_centre),
            source_point=(0.0, 0.0), source_modes=[(0, 0)],
            pupil_modes=[(0, 0)], output_modes=[(0, 0), (2, 0)],
            w_s=_WS_T3B, w_p=_WP_T3B, sigma_grid_extent=5e-4)
    assert res2.sigma_grid_n == 64
    assert not [w for w in rec2 if w.category is UserWarning
                and 'sigma_grid_n_max' in str(w.message)]


def test_w4_t1_pure_lg00_has_no_sigma_grid_and_is_unchanged():
    """SCOPE GUARD.  The closed-form branch builds no sigma grid, so the
    adaptive default cannot touch it -- and its value stays the W3-T3b
    frozen one (cross-backend contract of ``aberration_tensor_lg00_jax``)."""
    for R1, want in (
            (51.5e-3, 1.544807582649e+01 + 3.188059447022e+00j),
            (60.0e-3, -6.570638987023e+00 - 1.439184599267e+01j)):
        res = _tensor_w4(R1, ((0, 0),))
        assert res.sigma_grid_n is None
        assert res.sigma_curvature is None
        got = complex(res.L[0, 0])
        assert abs(got - want) / abs(want) < 1e-8, f'{got!r} != {want!r}'


# ---------------------------------------------------------------------------
# W4-T2 -- curvature-matched (complex-q) output basis
# ---------------------------------------------------------------------------

def _flips(vals):
    d = np.diff(np.asarray(vals, dtype=float))
    return int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))


def test_w4_t2_curvature_matched_basis_makes_L20_monotone_in_R1():
    """THE W4-T2 CLAIM.  Five designs R1 = 51.5 -> 60 mm at a fixed
    ``sigma_grid_n = 96`` (so the sigma quadrature is held constant and only
    the BASIS changes).  ``|L(2, 0)|`` must go from oscillating to monotone.

    Measured:
        flat basis  2.575157e-07 2.633322e-07 2.293199e-07 2.076991e-07
                    2.279086e-07      -> 2 of 3 sign flips, ptp/mean 0.235
        curv basis  4.186632e-08 1.447793e-07 2.947767e-07 5.393557e-07
                    7.295257e-07      -> 0 of 3 sign flips, ptp/mean 1.964
    Same verdict at n = 128 (2 flips vs 0) and, on eight designs at n = 256,
    5 of 6 vs 0 of 6.
    """
    modes = ((0, 0), (1, 0), (2, 0))
    raw, cur = [], []
    for R1 in _W4_R1S:
        raw.append(abs(complex(_tensor_w4(R1, modes, 96).L[2, 0])))
        cur.append(abs(complex(
            _tensor_w4(R1, modes, 96, None, True).L[2, 0])))
    assert _flips(raw) >= 1, (
        f'premise: the flat-basis channel must actually oscillate on this '
        f'design ladder, got {raw} (measured 2 of 3 sign flips)')
    assert _flips(cur) == 0, (
        f'curvature-matched |L(2,0)| is still not monotone: {cur} '
        f'({_flips(cur)} sign flips; measured 0)')
    d = np.diff(cur)
    assert np.all(d > 0.0), (
        f'curvature-matched |L(2,0)| must be strictly increasing in R1 on '
        f'this ladder, got steps {d}')
    # ...and it is a STRONGER discriminator, not merely a smoother one
    spread_raw = float(np.ptp(raw) / np.mean(raw))
    spread_cur = float(np.ptp(cur) / np.mean(cur))
    assert spread_cur > 4.0 * spread_raw, (
        f'ptp/mean: flat {spread_raw:.4f} vs curvature-matched '
        f'{spread_cur:.4f} (measured 0.235 vs 1.964)')


def test_w4_t2_measured_curvature_is_the_fields_own_quadratic_phase():
    """The basis curvature must be the field's, i.e. ``C = dv*/dsigma``, and
    it must be BIG -- the whole reason a real-waist basis fails.  Measured
    ``C_xx``: -4.841694 (R1 = 51.5) to -3.424957 (R1 = 60), monotone, i.e.
    -30.1 -> -21.5 WAVES of quadratic phase at the sigma-grid edge.

    Uses the SAME ``_tensor_w4`` cache key as the monotonicity test above so
    the five-design sweep is paid for once."""
    prev = None
    for R1, c_want in ((51.5e-3, -4.841694), (53.9e-3, -4.395232),
                       (56.3e-3, -3.987775), (58.7e-3, -3.614427),
                       (60.0e-3, -3.424957)):
        res = _tensor_w4(R1, ((0, 0), (1, 0), (2, 0)), 96, None, True)
        C = res.sigma_curvature
        assert C is not None and C.shape == (2, 2)
        assert C[0, 1] == C[1, 0], 'C is a Hessian; it must be symmetric'
        # rotationally symmetric singlet, on axis -> isotropic curvature
        assert abs(C[0, 0] - C[1, 1]) < 1e-9 * abs(C[0, 0])
        assert abs(float(C[0, 0]) - c_want) < 1e-3 * abs(c_want), (
            f'R1={R1}: C_xx {float(C[0, 0]):.6e} != measured {c_want:.6e}')
        # ...and the quadratic phase it encodes is many waves, not a nudge
        ext = _w4_extent(R1)
        waves = abs(float(C[0, 0])) * ext * ext / (2.0 * _WL_T3B)
        assert waves > 10.0, (
            f'R1={R1}: only {waves:.2f} waves of quadratic phase at the '
            f'grid edge -- the premise of W4-T2 is that this is large '
            f'(measured 30.1 down to 21.5)')
        if prev is not None:
            assert float(C[0, 0]) > prev, (
                'measured curvature must move monotonically with R1')
        prev = float(C[0, 0])


def test_w4_t2_default_off_is_bit_for_bit_the_flat_basis():
    """SCOPE GUARD.  ``curvature_matched_basis`` is OPT-IN: the default and
    an explicit ``False`` must be BIT-IDENTICAL, and must report no
    curvature -- every pinned channel value and the oracle comparisons are
    defined on the flat basis."""
    a = _tensor_w4(51.5e-3, tuple(MODES_MIXED), 96)
    b = _tensor_w4(51.5e-3, tuple(MODES_MIXED), 96, None, False)
    assert a.sigma_curvature is None and b.sigma_curvature is None
    assert np.array_equal(np.asarray(a.L).view(np.float64),
                          np.asarray(b.L).view(np.float64)), (
        'default and explicit curvature_matched_basis=False diverged')
    # ...and turning it ON must actually change the answer (otherwise the
    # flag would be silently inert)
    c = _tensor_w4(51.5e-3, tuple(MODES_MIXED), 96, None, True)
    assert c.sigma_curvature is not None
    i20 = MODES_MIXED.index((2, 0))
    assert abs(complex(c.L[i20, 0]) - complex(a.L[i20, 0])) > 0.1 * abs(
        complex(a.L[i20, 0]))


def test_w4_t2_flag_is_inert_on_the_pure_lg00_closed_form():
    """The closed form point-samples at sigma = 0, where any sigma-quadratic
    basis phase is exactly 1 -- so the flag must be a no-op there, not a
    silent rescale of the cross-backend contract."""
    on = _tensor_w4(51.5e-3, ((0, 0),), None, None, True)
    off = _tensor_w4(51.5e-3, ((0, 0),))
    assert complex(on.L[0, 0]) == complex(off.L[0, 0])
    assert on.sigma_curvature is None and on.sigma_grid_n is None


def test_w4_t2_excluding_defocus_algebraically_is_the_rejected_option():
    """MEASURED REJECTION of candidate (b).  'Exclude defocus from the
    channel' by removing the ``(1, 0)`` projection is a no-op in an
    orthonormal basis; on the discrete sigma grid the coupling is the Gram
    off-diagonal, measured 4.39e-04 at R1 = 51.5 mm rising to 0.219 at
    R1 = 60 mm (where the ``extent`` clamp truncates the basis to
    +-1.51 w_o).  So it moves ``|L(2,0)|`` by at most ~20 % -- and 2 of 6
    sign flips survive.  Pinned so the docstring's rejection stays true."""
    fit = _fit_t3b(51.5e-3)
    s2 = (fit.s2x_centre, fit.s2y_centre)
    res = _tensor_w4(51.5e-3, ((0, 0), (1, 0), (2, 0)), 96)
    w_o = res.w_o
    ext = _w4_extent(51.5e-3)
    X, Y, d = _grid(s2[0], s2[1], ext, 96)
    XL, YL = X - s2[0], Y - s2[1]
    g = [_lg_oracle(p, 0, w_o, XL, YL) for p in (0, 1, 2)]
    G = np.array([[_overlap(a, b, d) for b in g] for a in g])
    off = float(max(abs(G[0, 1]), abs(G[0, 2]), abs(G[1, 2])))
    assert off < 1e-2, (
        f'Gram off-diagonal {off:.3e} (measured 4.39e-04) -- the basis is '
        f'near-orthonormal here, which is exactly why removing the (1, 0) '
        f'projection cannot change (2, 0)')
    L10, L20 = complex(res.L[1, 0]), complex(res.L[2, 0])
    moved = abs(complex(G[2, 1]) * L10) / abs(L20)
    assert moved < 0.05, (
        f'removing the (1, 0) projection moves |L(2,0)| by {moved:.3e} '
        f'relative -- far too little to fix a channel that reverses '
        f'direction on 5 of 6 design steps (that is what W4-T2 does)')


# ---------------------------------------------------------------------------
# W4-T3 -- the jax canonical-fit estimator
# ---------------------------------------------------------------------------

def test_w4_t3_normal_equations_are_the_wrong_estimator_here():
    """PRE-FIX RED, WITH NO JAX INVOLVED.  The validation harness's
    ``fit_canonical_polynomials_jax matches NumPy fit`` red (rel coef_phi
    4.480e-02, local-only) was the JAX twin's least-squares ESTIMATOR, and
    this pure-NumPy oracle reproduces it on the same design matrix:

        estimator                                  max |dcoef| / scale
        normal eq, floor 1e-12*(tr/n + 1)              4.4898e-02
        normal eq, floor 0                             1.2213e-03
        QR (what ``_differentiable_lstsq`` now is)     1.2374e-10

    Two compounding causes, both pinned below: the old floor was LARGER
    than sigma_min^2 (so it damped real singular directions rather than
    regularising a degenerate one), and the normal equations square a
    6.54e+06 condition number so even floor 0 cannot reach 1e-5.
    """
    import lumenairy as la
    import lumenairy.propagators.asymptotic_canonical_fit as CF
    from lumenairy.propagators.asymptotic import fit_canonical_polynomials

    cap = []
    real = np.linalg.lstsq

    def _spy(A, b, rcond=None):
        cap.append((np.array(A), np.array(b)))
        return real(A, b, rcond=rcond)

    presc = la.make_singlet(R1=20e-3, R2=float('inf'), d=2e-3,
                            glass='N-BK7', aperture=4e-3)
    CF.np.linalg.lstsq = _spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit_canonical_polynomials(
                presc, 633e-9, source_box_half=20e-6, pupil_box_half=0.05,
                n_field=4, n_pupil=8, poly_order=4)
    finally:
        CF.np.linalg.lstsq = real

    A, b = cap[1]                    # cap[0] is the 5-column linear prefit
    assert A.shape == (1024, 70), f'design matrix shape moved: {A.shape}'
    sv = np.linalg.svd(A, compute_uv=False)
    cond = float(sv[0] / sv[-1])
    assert abs(cond - 6.540087e+06) < 1e-2 * 6.540087e+06, (
        f'cond(A) = {cond:.6e}, measured 6.540087e+06')
    # full rank -- so this is NOT a rank-deficiency story
    rcut = sv[0] * max(A.shape) * np.finfo(float).eps
    assert int((sv > rcut).sum()) == 70
    # ...but two singular values sit 2+ decades below the rest
    assert sv[-1] / sv[-3] < 1e-2

    n = A.shape[1]
    AtA = A.T @ A
    x_svd = real(A, b, rcond=None)[0]
    scale = float(np.max(np.abs(x_svd)))

    floor_old = 1e-12 * (float(np.trace(AtA)) / n + 1.0)
    assert floor_old > float(sv[-1]) ** 2, (
        f'the premise of the defect: the old Tikhonov floor '
        f'{floor_old:.4e} was LARGER than sigma_min^2 = '
        f'{float(sv[-1]) ** 2:.4e} (measured ratio 4.25)')

    def _rel(x):
        return float(np.max(np.abs(x - x_svd)) / scale)

    e_old = _rel(np.linalg.solve(AtA + floor_old * np.eye(n), A.T @ b))
    e_ne = _rel(real(AtA, A.T @ b, rcond=None)[0])
    Q, R = np.linalg.qr(A)
    e_qr = _rel(np.linalg.solve(R, Q.T @ b))

    assert e_old > 1e-2, (
        f'the old estimator must reproduce the red: {e_old:.4e} '
        f'(measured 4.4898e-02)')
    assert e_ne > 1e-4, (
        f'floor 0 must still miss by ~1e-3 (the cond(A)^2 penalty): '
        f'{e_ne:.4e} (measured 1.2213e-03) -- shrinking the floor was '
        f'never going to be the fix')
    assert e_qr < 1e-7, (
        f'QR must reproduce SVD lstsq: {e_qr:.4e} (measured 1.2374e-10)')
    assert e_qr < 1e-5 * e_old


def test_w4_t3_jax_fit_matches_numpy_in_coefficients_and_evaluation():
    """The validation red, as a unit pin.  Post-QR MEASURED: coef_phi
    1.223e-07, coef_s1x 2.501e-11, coef_s1y 6.479e-11, evaluated phi
    8.896e-08 relative to ptp(phi) over 4096 uniform in-box points
    (pre-fix 4.480e-02 and 3.253e-02).  Tolerances ~100x above measured:
    QR drift across LAPACK builds scales as cond(A)*eps ~ 1.5e-09."""
    pytest.importorskip('jax', reason='JAX not installed')
    import jax
    jax.config.update('jax_enable_x64', True)
    import lumenairy as la
    from lumenairy.propagators.asymptotic import (
        fit_canonical_polynomials,
        fit_canonical_polynomials_jax,
    )
    presc = la.make_singlet(R1=20e-3, R2=float('inf'), d=2e-3,
                            glass='N-BK7', aperture=4e-3)
    kw = dict(source_box_half=20e-6, pupil_box_half=0.05,
              n_field=4, n_pupil=8, poly_order=4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f_np = fit_canonical_polynomials(presc, 633e-9, **kw)
        f_jx = fit_canonical_polynomials_jax(presc, 633e-9, **kw)

    def rel(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return float(np.max(np.abs(b - a))
                     / max(float(np.max(np.abs(a))), 1e-30))

    assert rel(f_np.coef_phi, f_jx.coef_phi) < 1e-5
    assert rel(f_np.coef_s1x, f_jx.coef_s1x) < 1e-8
    assert rel(f_np.coef_s1y, f_jx.coef_s1y) < 1e-8

    rng = np.random.default_rng(20260726)
    u = rng.uniform(-1.0, 1.0, size=(4, 4096))
    pts = (f_np.s2x_centre + u[0] * f_np.s2x_halfrange,
           f_np.s2y_centre + u[1] * f_np.s2y_halfrange,
           f_np.v2x_centre + u[2] * f_np.v2x_halfrange,
           f_np.v2y_centre + u[3] * f_np.v2y_halfrange)
    p_np = np.asarray(f_np.eval_phi(*pts), dtype=np.float64)
    p_jx = np.asarray(f_jx.eval_phi(*pts), dtype=np.float64)
    ev = float(np.max(np.abs(p_jx - p_np)) / np.ptp(p_np))
    assert ev < 1e-5, (
        f'EVALUATED phi differs by {ev:.4e} of its range (pre-fix '
        f'3.253e-02 = 1.867e-01 waves; post-fix 8.896e-08)')


def test_w4_t3_jax_fit_gradient_stays_finite_through_qr():
    """The reason ``_differentiable_lstsq`` exists at all: ``jnp.linalg.
    lstsq``'s SVD gradient NaNs on near-degenerate singular values.  QR's
    VJP has no singular-value differences, so the gradient must stay finite
    -- and land on the v4.11.2 baseline that the ``test_audit_raytrace``
    pin bands at [1e2, 1e6].  Measured 1.027390e+04 (baseline ~10273.9)."""
    pytest.importorskip('jax', reason='JAX not installed')
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    import lumenairy as la
    from lumenairy.propagators.asymptotic import (
        fit_canonical_polynomials_jax,
    )
    presc = la.make_singlet(R1=20e-3, R2=float('inf'), d=2e-3,
                            glass='N-BK7', aperture=4e-3)

    def loss(sbh):
        f = fit_canonical_polynomials_jax(
            presc, 633e-9, source_box_half=sbh, pupil_box_half=0.05,
            n_field=4, n_pupil=8, poly_order=4)
        return jnp.sum(f.coef_phi ** 2)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = float(jax.grad(loss)(20e-6))
    assert np.isfinite(g), f'QR gradient is not finite: {g}'
    assert 1e2 < abs(g) < 1e6, (
        f'gradient {g:.6e} left the v4.11.2 band [1e2, 1e6] that '
        f'test_audit_raytrace pins (measured 1.027390e+04)')

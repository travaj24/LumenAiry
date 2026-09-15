"""VERIFY-WP-B12: the exit-vertex projection on the surface classes and the
exit media WP-B12's own suite does not reach.

Added 2026-09-15 by the independent re-verification of WP-B12
(``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B12.md``).

WP-B12 gave ``ray_transfer_jacobian`` / ``..._analytic`` an explicit output
reference plane and had the four ``fga.py`` sites ask for
``reference='exit_vertex'``; the projection reads the last surface's sag from
the package's SHARED kernels (``raytrace.surface._surface_sag_xy`` /
``_surface_sag_derivatives_xy``), which is what makes it right on surface
classes a ``conic_sag(radius, conic)`` copy would get wrong.  That is the
repair's central design claim -- but ``test_audit2609_b12_fga_reference_plane``
exercises only rotationally-symmetric CONIC and EVEN-ASPHERIC last surfaces in
air.  Three classes it reaches by construction and pins nowhere were confirmed
un-pinned by in-memory mutation (2026-09-15,
``validation/probe_verify_b12/vb12_mutate.py``): stripping the BICONIC
y-branch, stripping the FIELD-FRAME decenter, or hard-coding the exit index to
1.0 inside the projection each left all fourteen of that file's tests green.
This file closes those three, each as a two-sided DECISION:

* the projection agrees with the package's one definition of the vertex plane
  (``TraceResult.at_exit_vertex()``) on that class, AND
* the projection a narrower sag model would have produced is measurably
  DIFFERENT -- so none of these can pass on a build that dropped the branch.

Every bar is either bit-identity, a fixed constant stated with its measured
reading four or more decades below it, or derived at run time from a quantity
the running build measures; every separation is printed in its assertion
message.  No wall-clock assertion.  Runtime: 7.7 s warm, 13.5 s from cold
(the ``--store-durations`` run that produced the ``.test_durations`` entries;
slowest id 11.9 s) -- inside the 60 s budget either way.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import raytrace as rt
from lumenairy.raytrace import surfaces_from_prescription
from lumenairy.raytrace.differential import (
    _last_surface_sag_vanishes,
    ray_transfer_jacobian,
    ray_transfer_jacobian_analytic,
)

_LAM = 1.03e-6
_SEMI = 0.15e-3
_T = 0.55e-3
_GLASS = 'N-SSK8'          # not the glass of any other FGA test file
_N_RAYS = 121


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _presc(last, glass_after='air', glass=_GLASS, t=_T, semi=_SEMI):
    """A plano-first singlet whose LAST surface is whatever ``last`` says.

    Plano first, so every departure measured below belongs to the LAST
    surface -- the only one the projection reads.
    """
    s1 = {'radius': np.inf, 'conic': 0.0, 'thickness': t,
          'glass_before': 'air', 'glass_after': glass, 'semi_diameter': semi}
    s2 = {'conic': 0.0, 'thickness': 0.0, 'glass_before': glass,
          'glass_after': glass_after, 'semi_diameter': semi}
    s2.update(last)
    return {'name': 'vb12', 'aperture_diameter': 2 * semi,
            'surfaces': [s1, s2], 'thicknesses': [t], 'stop_index': 0}


def _surfs(presc):
    """The surface list FGA builds: a copy with the last transfer zeroed."""
    s = [copy.copy(x) for x in surfaces_from_prescription(presc)]
    s[-1].thickness = 0.0
    return s


def _fan(presc, n=_N_RAYS, semi=_SEMI, lam=_LAM, y_frac=0.4):
    """A collimated fan launched OFF the meridian (``y = 0.4 x``), so a
    surface whose x and y branches differ is actually probed in both."""
    surfs = _surfs(presc)
    h = np.linspace(semi / (2 * n), semi * 0.98, n)
    y = h * y_frac
    z = np.zeros(n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bundle = rt.RayBundle(x=h.copy(), y=y.copy(), z=z.copy(), L=z.copy(),
                              M=z.copy(), N=np.ones(n), wavelength=lam,
                              alive=np.ones(n, bool), opd=z.copy())
        res = rt.trace(bundle, surfs, lam)
        return surfs, h, y, z, res.image_rays, res.at_exit_vertex()


def _transfers(surfs, h, y, z, lam=_LAM, analytic=False):
    fn = ray_transfer_jacobian_analytic if analytic else ray_transfer_jacobian
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = fn(h.copy(), y.copy(), z.copy(), z.copy(), surfs, lam)
        v = fn(h.copy(), y.copy(), z.copy(), z.copy(), surfs, lam,
               reference='exit_vertex')
    return s, v


def _rot_sym_conic_opd(s, radius, n_exit=1.0):
    """The optical path a ROTATIONALLY-SYMMETRIC CONIC-BASE-ONLY projection
    would have produced -- ``gbd.py``'s in-line copy, written out here.

    This is the "narrower sag model" every test below is two-sided against.
    """
    r2 = np.asarray(s.x) ** 2 + np.asarray(s.y) ** 2
    c = 1.0 / radius
    sag = c * r2 / (1.0 + np.sqrt(np.maximum(1.0 - c * c * r2, 0.0)))
    sec = np.sqrt(1.0 + np.asarray(s.ux) ** 2 + np.asarray(s.uy) ** 2)
    return np.asarray(s.opd) - n_exit * sag * sec


# ===========================================================================
# 1. The biconic y-branch
# ===========================================================================
def test_a_biconic_last_surface_projects_with_both_axis_branches():
    """DECISION: on a BICONIC last surface the projection uses the surface's
    own two-branch sag, not its x radius read as a rotationally-symmetric one.

    The shared kernel ``_surface_sag_xy`` dispatches to
    ``surface_sag_biconic`` (the separable ``z_x(x) + z_y(y)`` form this
    package documents) whenever ``radius_y`` is set; a projection that read
    ``radius`` alone would evaluate a different surface everywhere off the
    x axis.  Asserted against ``TraceResult.at_exit_vertex()``, the package's
    ONE definition of the vertex plane, on a fan launched at ``y = 0.4 x`` so
    both branches carry load.

    Bars.  The agreement is asserted at 1e-14 m; MEASURED 2026-09-15 it is
    0.0 exactly on both fields (py3.14 / numpy 2.4.4 and py3.12 / numpy
    2.4.6), because the projection and ``at_exit_vertex`` evaluate the same
    kernel on the same landing point.  The two-sided half -- that a
    rotationally-symmetric read of ``radius`` is a DIFFERENT answer -- is
    asserted to exceed 100x that bar, and MEASURED 5.33e-07 m (0.517 waves),
    seven decades above it.  Both halves are premise-gated on the fan actually
    reaching the surface and on the two axis radii actually differing.

    The analytic backend refuses a biconic surface by design
    (``NotImplementedError``); that refusal is asserted rather than skipped,
    so the finite-difference backend is known to be the only one under test.
    """
    r_x, r_y = -0.86e-3, -1.20e-3
    presc = _presc({'radius': r_x, 'radius_y': r_y, 'conic_y': 0.0})
    surfs, h, y, z, img, ex = _fan(presc)
    ok = np.asarray(ex.alive, bool)
    # premises
    assert ok.sum() > _N_RAYS // 2, f'only {ok.sum()} rays survived'
    assert surfs[-1].radius_y is not None and surfs[-1].radius_y != surfs[-1].radius
    assert _last_surface_sag_vanishes(surfs[-1]) is False
    sag_w = float(np.abs(np.asarray(img.z)[ok]).max() / _LAM)
    assert sag_w > 1.0, f'premise: the last surface is flat here ({sag_w:.3f} waves)'

    s, v = _transfers(surfs, h, y, z)
    for field, got, want in (('x', v.x, ex.x), ('y', v.y, ex.y),
                             ('opd', v.opd, ex.opd)):
        e = float(np.abs(np.asarray(got)[ok] - np.asarray(want)[ok]).max())
        assert e < 1e-14, f'biconic {field}: {e:.3e} m'
    # two-sided: the x-radius-only model is a different surface
    gap = float(np.abs(_rot_sym_conic_opd(s, r_x)[ok]
                       - np.asarray(v.opd)[ok]).max())
    assert gap > 1e-12, (
        f'the biconic y-branch is not resolvable on this fixture: {gap:.3e} m '
        f'({gap / _LAM:.4f} waves) -- the two-sided claim is vacuous')
    # the analytic backend's documented refusal
    with pytest.raises(NotImplementedError):
        ray_transfer_jacobian_analytic(h.copy(), y.copy(), z.copy(), z.copy(),
                                       surfs, _LAM, reference='exit_vertex')


# ===========================================================================
# 2. The field-frame decentred last surface
# ===========================================================================
def test_a_field_frame_decentred_last_surface_projects_in_its_own_frame():
    """DECISION: a FIELD-FRAME decentred last surface (the ``apply_real_lens``
    displaced-pointwise convention, ``sag(x - dx, y - dy)``) is projected on
    its own frame, because the projection takes its sag from the shared kernel
    that honours the decenter -- not from a centred conic formula.

    This is the class ``_last_surface_sag_vanishes`` has to answer "not flat"
    for even when the base conic is flat, and the class the WP-B12 report
    lists under "correct by construction, not measured" (open item 6).

    Bars.  Agreement against ``at_exit_vertex()`` at 1e-14 m; MEASURED
    2026-09-15 it is 0.0 exactly on both builds.  Two-sided: the CENTRED
    projection is asserted to differ by more than 100x that bar and MEASURED
    5.44e-06 m (5.28 waves) -- a 35 um decenter on a R = -0.90 mm surface at a
    0.15 mm semi-aperture.  Premise-gated on the decenter surviving the
    prescription loader (the key is ``'decenter'``) and on the fan reaching
    the surface.
    """
    dx_dec, dy_dec = 35e-6, 12e-6
    radius = -0.90e-3
    presc = _presc({'radius': radius, 'decenter': (dx_dec, dy_dec)})
    surfs, h, y, z, img, ex = _fan(presc)
    ok = np.asarray(ex.alive, bool)
    # premises
    assert tuple(surfs[-1].field_decenter) == (dx_dec, dy_dec), (
        'premise: the prescription loader did not carry the field decenter')
    assert ok.sum() > _N_RAYS // 2, f'only {ok.sum()} rays survived'
    assert _last_surface_sag_vanishes(surfs[-1]) is False

    s, v = _transfers(surfs, h, y, z)
    for field, got, want in (('x', v.x, ex.x), ('y', v.y, ex.y),
                             ('opd', v.opd, ex.opd)):
        e = float(np.abs(np.asarray(got)[ok] - np.asarray(want)[ok]).max())
        assert e < 1e-14, f'field-frame {field}: {e:.3e} m'
    gap = float(np.abs(_rot_sym_conic_opd(s, radius)[ok]
                       - np.asarray(v.opd)[ok]).max())
    assert gap > 1e-12, (
        f'the field-frame decenter is not resolvable on this fixture: '
        f'{gap:.3e} m ({gap / _LAM:.4f} waves) -- the claim is vacuous')
    # a flat base with a decenter is still flat: the decenter shifts nothing
    flat_dec = _surfs(_presc({'radius': np.inf,
                              'decenter': (dx_dec, dy_dec)}))[-1]
    assert _last_surface_sag_vanishes(flat_dec) is False, (
        'a field-frame surface must never take the flat short-circuit: its '
        'sag callable / tilt ramp can be non-zero on a flat base')


# ===========================================================================
# 3. The exit-medium index
# ===========================================================================
def test_the_exit_medium_index_is_resolved_from_the_prescription():
    """DECISION: the optical path of the projected segment carries the index
    of the medium AFTER the last surface, resolved from the prescription
    (``exit_vertex.resolve_exit_index``) -- not the 1.0 the brief's suggested
    in-line edit would have hard-coded.

    Checked on an IMMERSED last surface (``glass_after='N-SF10'``), against
    this file's own formula ``opd_surface - n_exit * sag * sec`` with
    ``n_exit`` read from ``la.get_glass_index`` -- so the claim is not checked
    against ``at_exit_vertex()``, which resolves the index by the same call
    and would make the comparison circular.

    Bars.  Agreement 1e-14 m; MEASURED 2026-09-15, 1.08e-19 m on both builds
    and both backends.  Two-sided: the same formula with ``n = 1`` is asserted
    to differ by more than 100x the bar, and MEASURED 6.54e-06 m (6.35 waves)
    -- it is ``(n_exit - 1) * sag * sec``, so the separation is premise-gated
    on the exit index being well away from 1.
    """
    radius = -1.20e-3
    presc = _presc({'radius': radius}, glass_after='N-SF10')
    surfs, h, y, z, img, ex = _fan(presc)
    ok = np.asarray(ex.alive, bool)
    n_exit = float(la.get_glass_index('N-SF10', _LAM))
    # premises
    assert ok.sum() > _N_RAYS // 2, f'only {ok.sum()} rays survived'
    assert n_exit > 1.5, f'premise: exit index {n_exit:.4f} is too near 1'

    for analytic in (False, True):
        s, v = _transfers(surfs, h, y, z, analytic=analytic)
        mine = _rot_sym_conic_opd(s, radius, n_exit=n_exit)
        e = float(np.abs(np.asarray(v.opd)[ok] - mine[ok]).max())
        assert e < 1e-14, (
            f'{"analytic" if analytic else "fd"} exit-medium OPL: {e:.3e} m')
        vac = _rot_sym_conic_opd(s, radius, n_exit=1.0)
        gap = float(np.abs(np.asarray(v.opd)[ok] - vac[ok]).max())
        assert gap > 1e-12, (
            f'n_exit = 1 is not distinguishable here: {gap:.3e} m '
            f'({gap / _LAM:.4f} waves)')
    # an explicit n_exit overrides the prescription, as the keyword documents
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        forced = ray_transfer_jacobian(h.copy(), y.copy(), z.copy(), z.copy(),
                                       surfs, _LAM, reference='exit_vertex',
                                       n_exit=1.0)
        s_only = ray_transfer_jacobian(h.copy(), y.copy(), z.copy(), z.copy(),
                                       surfs, _LAM)
    want1 = _rot_sym_conic_opd(s_only, radius, n_exit=1.0)
    e1 = float(np.abs(np.asarray(forced.opd)[ok] - want1[ok]).max())
    assert e1 < 1e-14, f'n_exit override ignored: {e1:.3e} m'


# ===========================================================================
# 4. The two-backend agreement, on the mask both backends define
# ===========================================================================
def test_the_two_backends_agree_on_the_exit_vertex_plane_where_both_are_alive():
    """DECISION: the projection is backend-neutral -- asking the analytic and
    the finite-difference primitives for the exit-vertex plane does not open a
    gap between them that the last-surface plane did not already have.

    RESTATES the same property as
    ``test_audit2609_b12_fga_reference_plane::
    test_the_two_backends_agree_on_both_reference_planes_to_one_floor`` on the
    mask BOTH backends define.  That test masks on ``TraceResult.at_exit_vertex
    ().alive``, which is the BASE ray's aliveness; the finite-difference
    backend additionally kills a ray whose 9-ray FD COMPANION bundle vignettes
    (``_companion_alive``), and the outermost ray of that fixture is exactly
    such a ray, so the reading there is dominated by one ray's FD noise --
    97.05 on this build, where the docstring records 4.696e-07 (the value the
    same expression gives once the dead-companion ray is excluded; VERIFY-B12
    defect D-3).  Masking on ``fd.alive & analytic.alive`` restores it
    (measured 4.68293e-07, bit-identical on both builds).

    Bars.  The two backends' agreement is the FD backend's own truncation, so
    it is DERIVED here by halving the FD steps and taking ten times the change
    (MEASURED 2026-09-15: agreement 4.68e-07 on both planes, step-halving
    change 1.2e-07, bar 1.2e-06).  Two-sided, and this is the half the
    restated test needs: the projection must have MOVED the Jacobian, asserted
    as a relative change above 1e-3 -- without it the whole comparison passes
    trivially on a build where ``reference='exit_vertex'`` did nothing (it
    does: the identity mutation leaves that test green).  That half is scored
    PER ROW and MEASURED 2.35e-02 against a 100x-the-bar threshold of
    1.2e-04, i.e. 190x of margin.
    """
    presc = _presc({'radius': -0.90e-3})
    surfs, h, y, z, _img, _ex = _fan(presc)
    fd_s, fd_v = _transfers(surfs, h, y, z, analytic=False)
    an_s, an_v = _transfers(surfs, h, y, z, analytic=True)
    ok = np.asarray(fd_s.alive, bool) & np.asarray(an_s.alive, bool)
    assert ok.sum() > _N_RAYS // 2, f'only {ok.sum()} rays survived'

    def _rel(a, b):
        a = np.asarray(a.jacobian)[ok]
        b = np.asarray(b.jacobian)[ok]
        return float(np.abs(a - b).max()) / float(np.abs(b).max())

    # the FD backend's own truncation, measured on this build
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        coarse = ray_transfer_jacobian(h.copy(), y.copy(), z.copy(), z.copy(),
                                       surfs, _LAM, reference='exit_vertex')
        fine = ray_transfer_jacobian(h.copy(), y.copy(), z.copy(), z.copy(),
                                     surfs, _LAM, reference='exit_vertex',
                                     h_pos=5e-7, h_slope=2.5e-5)
    step = (float(np.abs(np.asarray(coarse.jacobian)[ok]
                         - np.asarray(fine.jacobian)[ok]).max())
            / float(np.abs(np.asarray(coarse.jacobian)[ok]).max()))
    bar = max(10.0 * step, 1e-8)
    r_s, r_v = _rel(fd_s, an_s), _rel(fd_v, an_v)
    assert r_s < bar, f'surface plane {r_s:.3e} vs derived bar {bar:.3e}'
    assert r_v < bar, f'exit-vertex plane {r_v:.3e} vs derived bar {bar:.3e}'
    # Two-sided: the projection actually changed the Jacobian.  Scored
    # PER ROW -- the four rows carry different units (the C / D blocks are
    # 1/f ~ 7e+02 per metre where A / B are O(1)), so a single global
    # normalisation hides the position rows, which are the only ones the
    # projection touches, behind the slope rows.
    ja = np.asarray(an_v.jacobian)[ok]
    jb = np.asarray(an_s.jacobian)[ok]
    row = np.abs(jb).max(axis=(0, 2))[None, :, None]
    moved = float((np.abs(ja - jb) / row).max())
    assert moved > 100.0 * bar, (
        f'the exit-vertex Jacobian is indistinguishable from the '
        f'last-surface one (per-row {moved:.3e} against 100x the backend '
        f'bar {100.0 * bar:.3e}): this comparison is vacuous')

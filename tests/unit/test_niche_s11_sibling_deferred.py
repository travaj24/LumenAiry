"""Regression pins for the DEFERRED report-only findings of the
sibling-pattern sweep (``docs/audits/AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25.md``
§1).  Each ``S11-*`` block below closes one row of that table.

S11-1 -- ``raytrace/seidel.py`` FLAT-FOLD MIRROR PARITY (the physics one).
    ``system_abcd`` / ``seidel_coefficients`` gated the Welford
    ``n' = -n`` index flip and the ``mirror_parity`` bookkeeping on
    ``np.isfinite(R)``, so a FLAT fold mirror skipped it.  The flip encodes
    the REFLECTION (the propagation direction reverses), not the POWER, so
    it is R-independent: with ``c = 0``, ``nu' = nu`` but
    ``u' = nu'/n' = -u``.  Three independent oracles, all in this file:

    (a) CONTINUITY IN R.  The flat answer must equal the ``R -> +-inf``
        limit of the curved answer.  Pre-fix, on ``[fold, N-BK7 singlet]``:
        ``R = -1e12`` gave ``EFL = -0.1985770345`` while ``R = inf`` gave
        ``+0.1985770345`` (sign flip), and ``S1_total`` /``S4_total`` flipped
        with it (-7.616380e-06 -> +7.616380e-06, -3.349219 -> +3.349219).
    (b) THE EXACT 3-D TRACE (``raytrace.trace``, no shared code with
        ``seidel.py``): its paraxial back focus, mapped into the unfolded
        frame by ``(-1)**n_mirrors``, must equal ``system_abcd``'s ``bfl``.
        Pre-fix ``[flat fold(0.2), concave R=-1]`` gave ``bfl = +0.5`` vs
        the trace's ``-0.5``; ``[mirror R=-1, flat fold, mirror R=-0.5]``
        was not even a sign flip -- ``EFL = -0.19230769`` vs the trace's
        ``+0.16666667``, because the marginal ray height DIVERGES past the
        fold (y = [12.700, 17.780, 22.860] mm instead of
        [12.700, 17.780, 12.700] mm).
    (c) THE REPO'S OWN PINNED FOLDED PRESCRIPTIONS (the ones
        test_v4_15_1_agent_g_matches_system_abcd.py parametrises).  Three of
        four moved, each from a value the exact trace rejects to one it
        confirms to <= 3e-13:
          folded_telephoto   bfl +0.136245702 -> +0.024092097 (trace +0.024092097)
          4fold_periscope    bfl +0.017575518 -> +0.047575518 (trace +0.047575518)
          folded_cass_2c2f   bfl -0.183636364 -> -0.143636364 (trace -0.143636364)
    |S1| is UNCHANGED by the fix (a flat fold adds no aberration): the
    repo's traced-transverse-aberration oracle gives |ratio| = 1.0000 for
    the singlet with and without a flat fold, before and after.
    ``lumenairy/algebra/from_prescription.py`` carried a deliberate copy of
    the bug (v4.15.2 audit P1-NEW-B made the algebra layer match
    ``system_abcd``); it is fixed in lockstep, so the pinned
    agreement tests keep passing on the corrected value.

S11-2 -- ``raytrace/intersection.py`` NaN-RAY APERTURE GATE.
    ``clipped = (h_sq > sd**2) & alive`` is False for a NaN ``h_sq``, so a
    ray whose position went non-finite upstream survived ALIVE with
    ``error_code == RAY_OK``.  Measured on the flat fast path,
    ``x = [0, 20mm, nan, 0]`` / ``y = [0, 0, 0, nan]``, ``sd = 10 mm``:
    ``alive = [T F T T]``, ``error_code = [0 2 0 0]``.  Now the safe
    polarity of the JAX twin (``inside = h_sq <= sd*sd``) plus the
    ``== RAY_OK`` first-failure-wins guard the old comment promised:
    ``alive = [T F F F]``, ``error_code = [0 2 4 4]`` (RAY_NAN for the
    numerical faults, so ``raytrace.layout``'s histogram does not
    mis-attribute them to vignetting).

S11-3 -- ``raytrace/from_field.py`` EDGE-ANCHORED RAY PLACEMENT.
    ``_place_uniform`` used ``np.linspace(0, N-1, n)``, putting the first
    and last sub-grid point ON the array edges -- contradicting its own
    "central pixel of each sub-cell" contract.  On a centred
    ``w0 = N*dx/4`` Gaussian (N = 64, threshold 1e-4) the edge points are
    thresholded away, so survivors went 1/2/3/4 -> **0** (``rays_from_field``
    RAISED "no pixels survived") and 5/8/9/16/25 -> 2/5/5/12/13.  The
    cell-centred class lattice ``(arange(n)+0.5)*N/n - 0.5`` gives
    1/2/3/4/5/8/9/16 -> exactly n and 25 -> 21.  NOT bit-compatible at any
    n_rays -- deliberate.

S11-4 -- ``analysis/detector.py`` n_pixels-vs-pixel_pitch CONTRACT.
    An explicit ``n_pixels`` silently redefined the pixel AREA: the integer
    fast path block-summed ``Ny/n_pixels`` field samples while the returned
    axis was spaced by ``pixel_pitch``.  Photon-scale end-to-end (uniform
    ``I0 = 1e18 /m^2/s``, 64x64 @ 1 um, QE 1, 1 s, no read noise; expected
    per-pixel electrons ``I0 * pixel_pitch**2``) measured/expected:
    (16, 4um) 1.0000 | (16, 2um) 4.0000 | (16, 8um) 0.2500 |
    (8, 2um) 16.0000 | (16, 2.5um) 2.5600  ->  all 1.0000 post-fix.

S11-5 -- ``analysis/ao.py`` INERT ``noise_sigma_pixels``.
    The noise was added AFTER the ``/ slope_scale`` calibration rescale, so
    a sigma quoted in RAW SH slope units perturbed slopes already in
    rad/m of phase.  On a 64x64 defocus residual, subaperture_grid 8,
    lenslet_focal 5e-3: ``noise_sigma_pixels = 1`` moved the reconstruction
    by 4.83e-5 relative at ``dx_pupil = 1e-4`` m and 4.81e-7 at 1e-5 m.
    The injection now happens on the RAW measurement (upstream of the
    calibration), which is algebraically ``+ n / slope_scale`` and keeps the
    RNG draw count/order -- so a given ``rng_seed`` yields the identical
    noise SEQUENCE, correctly applied.

S11-6 -- SMALL GUARDS BATCH (mft ``resample_field``, ao ``n_lenslets``,
    ray_fan telecentric ``inf * tan(0)``, ``through_focus_rms`` /
    ``trace_summary`` degenerate inputs, ``through_focus`` ``nanmax``).

S11-7 -- ``raytrace/_conic_core.py`` NaN-INPUT CLAMP.
    The documented JAX zero-clamp for out-of-domain REAL heights is KEPT;
    only a non-finite INPUT now propagates (``conic_sag(nan)`` was ``+0.0``
    -- a finite phantom-vertex sag a jax Newton could converge onto -- and
    the sibling ``conic_sag_derivs(nan)`` already returned NaN).

CI-safe: analytic surfaces / analytic fields, no external assets; the JAX
grad checks skip cleanly when jax is absent.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from lumenairy.raytrace import (
    Surface,
    first_order_data,
    make_fan,
    surfaces_from_prescription,
    trace,
)
from lumenairy.raytrace._conic_core import conic_sag, conic_sag_derivs
from lumenairy.raytrace.from_field import _place_uniform, rays_from_field
from lumenairy.raytrace.intersection import _intersect_surface
from lumenairy.raytrace.seidel import seidel_coefficients, system_abcd
from lumenairy.raytrace.surface import (
    RAY_APERTURE,
    RAY_NAN,
    RAY_OK,
    RAY_TIR,
    RayBundle,
)

_WL = 1.31e-6


# ===========================================================================
# S11-1: flat-fold mirror parity
# ===========================================================================

def _mir(radius, thickness, sd=np.inf):
    return Surface(radius=radius, conic=0.0, semi_diameter=sd,
                   glass_before='air', glass_after='air',
                   is_mirror=True, is_stop=False, thickness=thickness)


def _ref(radius, thickness, gb, ga, sd=np.inf):
    return Surface(radius=radius, conic=0.0, semi_diameter=sd,
                   glass_before=gb, glass_after=ga,
                   is_mirror=False, is_stop=False, thickness=thickness)


def _singlet():
    return [_ref(0.1, 5e-3, 'air', 'N-BK7'),
            _ref(np.inf, 0.0, 'N-BK7', 'air')]


def _exact_bfl_unfolded(surfaces, y0=1e-6):
    """Paraxial back focus from the EXACT 3-D trace, expressed in the
    unfolded frame ``system_abcd`` reports in.

    Shares no code with ``seidel.py``: ``trace`` reflects the true
    direction cosines off the true surface normal.  ``u = M / N`` is the
    geometric slope dy/dz, and ``_transfer`` reaches the next vertex plane
    ``z = thickness`` via ``t = (thickness - z) / N``, i.e. ``dy = u * t``
    exactly -- so the traced ``-y / u`` IS the paraxial back focus.  After
    an odd number of mirrors the unfolded z runs opposite to the global z,
    hence the ``(-1)**n_mirrors``.
    """
    surfs = [copy.copy(s) for s in surfaces]
    for s in surfs:
        s.semi_diameter = np.inf          # never vignette the probe ray
    r = RayBundle(x=np.zeros(1), y=np.full(1, y0), z=np.zeros(1),
                  L=np.zeros(1), M=np.zeros(1), N=np.ones(1),
                  opd=np.zeros(1), wavelength=_WL, alive=np.ones(1, bool))
    f = trace(r, surfs, _WL, output_filter='last').ray_history[-1]
    y, u = float(f.y[0]), float(f.M[0]) / float(f.N[0])
    if abs(u) < 1e-30:
        return np.inf
    n_mir = sum(1 for s in surfaces if s.is_mirror)
    return (-1.0) ** n_mir * (-y / u)


@pytest.mark.parametrize('R_fold', [-1e8, -1e9, -1e12, 1e12, 1e9, 1e8])
def test_s11_1_flat_fold_is_the_large_R_limit_of_a_curved_fold(R_fold):
    """ORACLE (a): continuity in R.  A flat fold must give the same ABCD
    and the same Seidel sums as a very-large-|R| curved fold.  Pre-fix the
    two differed by a SIGN (EFL -0.1985770345 at R=-1e12 vs +0.1985770345
    at R=inf)."""
    flat = [_mir(np.inf, 0.1)] + _singlet()
    curved = [_mir(R_fold, 0.1)] + _singlet()

    Mf, eflf, bflf, fflf = system_abcd(flat, _WL)
    Mc, eflc, bflc, fflc = system_abcd(curved, _WL)
    assert np.allclose(Mf, Mc, rtol=2e-6, atol=0), (
        f"flat fold ABCD {Mf.ravel()} is not the R={R_fold:g} limit "
        f"{Mc.ravel()} -- the isfinite(R) gating is back.")
    for name, a, b in (('efl', eflf, eflc), ('bfl', bflf, bflc),
                       ('ffl', fflf, fflc)):
        assert np.sign(a) == np.sign(b) and abs(a - b) <= 2e-6 * abs(b), (
            f"{name}: flat={a:+.10f} vs R={R_fold:g} {b:+.10f}")

    sf, _ = seidel_coefficients(flat, _WL, stop_index=1, field_angle=0.01)
    sc, _ = seidel_coefficients(curved, _WL, stop_index=1, field_angle=0.01)
    for key in ('S1', 'S2', 'S3', 'S4', 'S5'):
        a = float(sf['total'][key])
        b = float(sc['total'][key])
        assert abs(a - b) <= 2e-6 * max(abs(b), 1e-12), (
            f"total {key}: flat={a:+.8e} vs R={R_fold:g} {b:+.8e}")
    assert np.allclose(sf['y_marginal'], sc['y_marginal'], rtol=2e-6)


@pytest.mark.parametrize('name,builder', [
    ('singlet (control, no fold)', _singlet),
    ('FLAT fold -> singlet',
     lambda: [_mir(np.inf, 0.1)] + _singlet()),
    ('singlet -> FLAT fold',
     lambda: [_ref(0.1, 5e-3, 'air', 'N-BK7'),
              _ref(np.inf, 20e-3, 'N-BK7', 'air'), _mir(np.inf, 0.0)]),
    ('2 FLAT folds -> singlet',
     lambda: [_mir(np.inf, 0.1), _mir(np.inf, 0.1)] + _singlet()),
    ('FLAT fold -> mirror R=-1',
     lambda: [_mir(np.inf, 0.2), _mir(-1.0, 0.0)]),
    ('mirror R=-1 -> FLAT fold',
     lambda: [_mir(-1.0, 0.2), _mir(np.inf, 0.0)]),
    ('mirror -> FLAT fold -> mirror',
     lambda: [_mir(-1.0, 0.2), _mir(np.inf, 0.2), _mir(-0.5, 0.0)]),
    ('Cassegrain (2 curved, control)',
     lambda: [_mir(-1.0, 0.4), _mir(-0.3, 0.0)]),
])
def test_s11_1_system_abcd_bfl_matches_the_exact_3d_trace(name, builder):
    """ORACLE (b): the paraxial ABCD must agree with the exact 3-D trace.

    Pre-fix the three flat-fold cases with a downstream leg disagreed:
    ``FLAT fold -> singlet`` +0.19525164 vs -0.19525164,
    ``FLAT fold -> mirror`` +0.5 vs -0.5, and
    ``mirror -> FLAT fold -> mirror`` -0.34615385 vs +0.16666667.
    """
    surfs = builder()
    _M, _efl, bfl, _ffl = system_abcd(surfs, _WL)
    exact = _exact_bfl_unfolded(surfs)
    assert abs(bfl - exact) <= 1e-6 * max(1e-3, abs(exact)), (
        f"{name}: system_abcd bfl={bfl:+.9f} disagrees with the exact "
        f"3-D trace {exact:+.9f}.")


@pytest.mark.parametrize('name,expect_bfl', [
    # ORACLE (c): the repo's own pinned folded prescriptions.  These are
    # DELIBERATE pin moves -- the second column is the post-fix value AND
    # the exact-trace value; the pre-fix values are in this file's
    # docstring.  See the ``_build_*`` docstrings in
    # test_v4_15_1_agent_g_matches_system_abcd.py.
    ('_build_folded_singlet', +0.017696624),          # unchanged (fold last)
    ('_build_folded_telephoto', +0.024092097),        # was +0.136245702
    ('_build_folded_4fold_periscope', +0.047575518),  # was +0.017575518
    ('_build_folded_cassegrain_2curved_2flat', -0.143636364),  # was -0.183636364
])
def test_s11_1_pinned_folded_prescriptions_match_the_exact_trace(
        name, expect_bfl):
    import importlib
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    mod = importlib.import_module('test_v4_15_1_agent_g_matches_system_abcd')
    surfs = surfaces_from_prescription(getattr(mod, name)())
    _M, _efl, bfl, _ffl = system_abcd(surfs, 633e-9)
    assert abs(bfl - expect_bfl) <= 1e-8, (
        f"{name}: system_abcd bfl={bfl:+.9f}, pinned {expect_bfl:+.9f}")
    # ... and the value is the one the independent trace confirms.
    surfs633 = [copy.copy(s) for s in surfs]
    r = RayBundle(x=np.zeros(1), y=np.full(1, 1e-7), z=np.zeros(1),
                  L=np.zeros(1), M=np.zeros(1), N=np.ones(1),
                  opd=np.zeros(1), wavelength=633e-9, alive=np.ones(1, bool))
    for s in surfs633:
        s.semi_diameter = np.inf
    f = trace(r, surfs633, 633e-9, output_filter='last').ray_history[-1]
    n_mir = sum(1 for s in surfs if s.is_mirror)
    exact = (-1.0) ** n_mir * (-float(f.y[0])
                               / (float(f.M[0]) / float(f.N[0])))
    assert abs(bfl - exact) <= 1e-8, (
        f"{name}: pinned bfl={bfl:+.9f} vs exact trace {exact:+.9f}")


def test_s11_1_mirror_free_systems_are_bit_identical():
    """The fix must not touch a single mirror-free system: the branch
    reorder is pure control flow for refractors."""
    cases = [
        _singlet(),
        [_ref(0.2, 6e-3, 'air', 'N-BK7'), _ref(-0.2, 0.0, 'N-BK7', 'air')],
        [_ref(np.inf, 3e-3, 'air', 'N-BK7'),
         _ref(np.inf, 20e-3, 'N-BK7', 'air')] + _singlet(),
    ]
    # Hard-pinned values recorded on the pre-fix tree (they are unchanged
    # post-fix, which is the assertion).
    expected_bfl = [0.19525164424313793, 0.19757438028537042,
                    0.19525164424313793]
    for surfs, want in zip(cases, expected_bfl):
        _M, _efl, bfl, _ffl = system_abcd(surfs, _WL)
        assert bfl == want, f"bfl {bfl!r} != pre-fix {want!r}"


def test_s11_1_flat_fold_leaves_the_traced_spherical_magnitude_alone():
    """A flat fold adds no aberration, so |S1| must be identical with and
    without it -- cross-checked against the repo's own traced
    transverse-aberration oracle (a full geometric trace)."""
    r_stop = 12.7e-3

    def traced_S1(surfaces):
        fod = first_order_data(surfaces, _WL)
        n_mir = sum(1 for s in surfaces if s.is_mirror)
        surf = [copy.copy(s) for s in surfaces]
        surf[-1].thickness = (-1.0) ** n_mir * fod.bfl
        surf.append(Surface(radius=np.inf, thickness=0.0,
                            glass_before='air', glass_after='air'))
        for s in surf:
            s.semi_diameter = np.inf
        fan = make_fan('y', r_stop, 201, field_angle=0.0, wavelength=_WL)
        img = trace(fan, surf, _WL).image_rays
        rho = np.linspace(-1.0, 1.0, 201)
        dy = np.asarray(img.y)
        good = np.asarray(img.alive) & np.isfinite(dy)
        basis = np.vstack([rho[good] ** 3, rho[good] ** 5,
                           rho[good] ** 7]).T
        c3 = np.linalg.lstsq(basis, dy[good], rcond=None)[0][0]
        return -2.0 * (-r_stop / ((-1.0) ** n_mir * fod.efl)) * c3

    plain = _singlet()
    folded = [_mir(np.inf, 0.1)] + _singlet()
    s_plain, _ = seidel_coefficients(plain, _WL, stop_index=0,
                                    field_angle=0.0)
    s_fold, _ = seidel_coefficients(folded, _WL, stop_index=1,
                                    field_angle=0.0)
    a = abs(float(s_plain['total']['S1']))
    b = abs(float(s_fold['total']['S1']))
    assert abs(a - b) <= 1e-12 * a, (
        f"|S1| changed by inserting a flat fold: {a:.8e} -> {b:.8e}")
    for surfs, analytic in ((plain, a), (folded, b)):
        t = abs(traced_S1(surfs))
        assert 0.85 < analytic / t < 1.15, (
            f"analytic |S1|={analytic:.6e} vs traced oracle {t:.6e}")


def test_s11_1_algebra_twin_still_matches_system_abcd_on_flat_folds():
    """``algebra.from_prescription`` deliberately copied the bug in
    v4.15.2 (audit P1-NEW-B) so the two layers would agree; both are fixed
    now, so they must agree on the CORRECTED value."""
    import lumenairy as la
    rx = {
        'name': 'S11-1 fold probe',
        'aperture_diameter': 25.4e-3,
        'surfaces': [
            {'radius': 100e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -100e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'},
            {'radius': float('inf'), 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'MIRROR'},
            {'radius': -50e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-SF11'},
            {'radius': 50e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-SF11', 'glass_after': 'air'},
        ],
        'thicknesses': [3e-3, 40e-3, 30e-3, 3e-3, 0.0],
    }
    surfs = surfaces_from_prescription(rx)
    M_ref, _efl, bfl, _ffl = system_abcd(surfs, 633e-9)
    op = la.Operator.from_prescription(rx, 633e-9)
    assert np.allclose(op.abcd, M_ref, atol=1e-12)
    # The corrected value, confirmed by the exact trace.
    assert abs(bfl - _exact_bfl_unfolded_at(surfs, 633e-9)) <= 1e-9


def _exact_bfl_unfolded_at(surfaces, wavelength, y0=1e-7):
    surfs = [copy.copy(s) for s in surfaces]
    for s in surfs:
        s.semi_diameter = np.inf
    r = RayBundle(x=np.zeros(1), y=np.full(1, y0), z=np.zeros(1),
                  L=np.zeros(1), M=np.zeros(1), N=np.ones(1),
                  opd=np.zeros(1), wavelength=wavelength,
                  alive=np.ones(1, bool))
    f = trace(r, surfs, wavelength, output_filter='last').ray_history[-1]
    n_mir = sum(1 for s in surfaces if s.is_mirror)
    return (-1.0) ** n_mir * (-float(f.y[0])
                              / (float(f.M[0]) / float(f.N[0])))


# ===========================================================================
# S11-2: NaN-ray aperture gate
# ===========================================================================

def _bundle(xs, ys, codes=None):
    n = len(xs)
    return RayBundle(
        x=np.array(xs, float), y=np.array(ys, float), z=np.zeros(n),
        L=np.zeros(n), M=np.zeros(n), N=np.ones(n), opd=np.zeros(n),
        wavelength=_WL, alive=np.ones(n, bool),
        error_code=(np.zeros(n, np.uint8) if codes is None
                    else np.array(codes, np.uint8)),
    )


_SD = 10e-3
_FLAT = Surface(radius=np.inf, conic=0.0, semi_diameter=_SD,
                glass_before='air', glass_after='air', is_mirror=False,
                is_stop=False, thickness=0.0)
_SPH = Surface(radius=0.5, conic=0.0, semi_diameter=_SD,
               glass_before='air', glass_after='air', is_mirror=False,
               is_stop=False, thickness=0.0)


def test_s11_2_nan_position_ray_is_killed_not_reported_ok():
    """Pre-fix: ``alive = [T F T T]``, ``error_code = [0 2 0 0]``."""
    r = _bundle([0.0, 20e-3, np.nan, 0.0], [0.0, 0.0, 0.0, np.nan])
    _intersect_surface(r, _FLAT, n_medium=1.0)
    assert list(r.alive) == [True, False, False, False], (
        f"NaN-position ray survived the aperture gate: alive={r.alive}")
    assert list(r.error_code) == [RAY_OK, RAY_APERTURE, RAY_NAN, RAY_NAN], (
        f"error_code={r.error_code}: a non-finite intersection height is a "
        f"numerical fault (RAY_NAN), not a clear-aperture clip.")
    assert r.error_code.dtype == np.uint8


def test_s11_2_finite_vignetting_is_unchanged():
    """The polarity flip is bit-identical for every finite position:
    ``~(a <= b) == (a > b)`` exactly when neither operand is NaN."""
    h = np.array([0.0, 5e-3, _SD - 1e-15, _SD, _SD + 1e-15, 20e-3])
    for surf in (_FLAT, _SPH):
        r = _bundle(h, np.zeros_like(h))
        _intersect_surface(r, surf, n_medium=1.0)
        # Exactly-on-the-edge stays alive (<=), just as (h_sq > sd**2) did.
        assert list(r.alive) == [True, True, True, True, False, False], (
            f"{surf.radius}: alive={r.alive}")
        assert list(r.error_code) == [RAY_OK] * 4 + [RAY_APERTURE] * 2


def test_s11_2_first_failure_wins_guard_is_present():
    """The pre-fix ``np.where(clipped, RAY_APERTURE, ...)`` relabelled a
    ray that already carried an earlier diagnosis; the comment claimed
    otherwise."""
    r = _bundle([20e-3, 20e-3], [0.0, 0.0], codes=[RAY_OK, RAY_TIR])
    _intersect_surface(r, _FLAT, n_medium=1.0)
    assert list(r.error_code) == [RAY_APERTURE, RAY_TIR], (
        f"error_code={r.error_code}: the pre-existing RAY_TIR diagnosis "
        f"must not be overwritten by RAY_APERTURE.")


def test_s11_2_matches_the_jax_twin_polarity():
    """``jax_trace._apply_aperture_jax`` is the reference polarity."""
    import inspect

    from lumenairy.raytrace.jax_trace import _apply_aperture_jax
    src = inspect.getsource(_apply_aperture_jax)
    assert 'inside' in src and '<=' in src
    src_np = inspect.getsource(_intersect_surface)
    assert 'inside = h_sq <= surface.semi_diameter ** 2' in src_np, (
        "the NumPy aperture gate must use the JAX twin's safe polarity")


# ===========================================================================
# S11-3: cell-centred ray placement
# ===========================================================================

def _gauss(N=64, dx=10e-6, w_over_N=4.0):
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / (N * dx / w_over_N) ** 2).astype(
        complex), dx


@pytest.mark.parametrize('n_rays,expect', [
    (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (8, 8), (9, 9), (16, 16),
    (25, 21), (36, 32),
])
def test_s11_3_uniform_placement_survivor_counts(n_rays, expect):
    """DELIBERATE placement change.  Pre-fix (edge-anchored) counts on the
    same probe were 0/0/0/0/2/5/5/12/13/24 -- n_rays 1..4 produced NO rays
    at all and ``rays_from_field`` RAISED."""
    E, dx = _gauss()
    xs, _ys, _ix, _iy = _place_uniform(E, dx, dx, n_rays, 1e-4)
    assert xs.size == expect, f"n_rays={n_rays}: {xs.size} survivors"


@pytest.mark.parametrize('n_rays', [1, 2, 4, 9, 16])
def test_s11_3_small_n_rays_no_longer_raises(n_rays):
    """``n_rays = 1`` and ``4`` used to raise "no pixels survived
    intensity thresholding" on a perfectly valid centred Gaussian."""
    E, dx = _gauss()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rb = rays_from_field(E=E, dx=dx, wavelength=_WL, n_rays=n_rays,
                             placement='uniform')
    assert rb.n_rays == n_rays
    assert np.all(np.isfinite(rb.x)) and np.all(np.isfinite(rb.y))


def test_s11_3_single_ray_lands_on_the_centre_pixel():
    """The whole point of cell-centring: ``n_rays = 1`` samples the CENTRE
    of the field, not the corner pixel (0, 0)."""
    E, dx = _gauss()
    _xs, _ys, ix, iy = _place_uniform(E, dx, dx, 1, 1e-4)
    assert (int(ix[0]), int(iy[0])) == (32, 32), (
        f"n_rays=1 landed on pixel {(int(ix[0]), int(iy[0]))}; the "
        f"cell-centred lattice for N=64 is 31.5 -> 32.")


def test_s11_3_lattice_is_the_documented_cell_centred_class():
    """Pin the formula itself so the edge-anchored ``linspace(0, N-1, n)``
    cannot come back."""
    N = 64
    for n in (1, 2, 3, 4, 5, 8, 16):
        want = np.clip(np.round((np.arange(n) + 0.5) * N / n - 0.5),
                       0, N - 1).astype(np.intp)
        E = np.ones((N, N), dtype=complex)     # uniform: nothing thresholded
        _xs, _ys, ix, iy = _place_uniform(E, 1e-6, 1e-6, n * n, 0.0)
        if n * n <= N * N:
            assert set(np.unique(ix)).issubset(set(range(N)))
        # 1-D projection check via the square case n x n.
        got_x = np.unique(_place_uniform(E, 1e-6, 1e-6, n * n, 0.0)[2])
        assert got_x.size == n and np.array_equal(got_x, np.unique(want)), (
            f"n={n}: x lattice {got_x} != cell-centred {np.unique(want)}")


def test_s11_3_transpose_pitch_symmetry_still_holds():
    """The S9-RT3 ``aspect`` fix must survive the lattice change."""
    def pitches(Nx, Ny, n_rays=64):
        E = np.ones((Ny, Nx), dtype=complex)
        _, _, ix, iy = _place_uniform(E, 2e-6, 2e-6, n_rays, 0.0)
        ux, uy = np.unique(ix), np.unique(iy)
        return (float(np.diff(ux).mean()) if ux.size > 1 else np.nan,
                float(np.diff(uy).mean()) if uy.size > 1 else np.nan)
    for Nx, Ny in [(64, 256), (32, 512), (128, 128)]:
        px, py = pitches(Nx, Ny)
        qx, qy = pitches(Ny, Nx)
        assert abs(px - qy) <= 0.15 * max(px, qy) + 1.0
        assert abs(py - qx) <= 0.15 * max(py, qx) + 1.0


# ===========================================================================
# S11-4: detector n_pixels vs pixel_pitch
# ===========================================================================

_I0 = 1e18
_DXF = 1e-6
_NF = 64


def _uniform_field():
    return np.full((_NF, _NF), np.sqrt(_I0), dtype=complex)


@pytest.mark.parametrize('n_pixels,pitch', [
    (16, 4e-6),      # matched: detector spans the field  (was 1.0000)
    (16, 2e-6),      # was 4.0000x too much per pixel
    (16, 8e-6),      # was 0.2500x
    (8, 2e-6),       # was 16.0000x
    (16, 2.5e-6),    # was 2.5600x
    (None, 4e-6),    # default n_pixels (was already correct)
    (None, 2e-6),
    (15, 4e-6),      # non-integer Ny/n_pixels: was already correct
    (13, 2e-6),
])
def test_s11_4_per_pixel_signal_is_I0_times_pixel_pitch_squared(
        n_pixels, pitch):
    """PHOTON-SCALE end-to-end: with QE = 1, 1 s exposure and no read
    noise, a pixel of pitch ``p`` on a uniform irradiance ``I0`` must
    collect ``I0 * p**2`` electrons -- ``pixel_pitch`` IS the collecting
    area, whatever ``n_pixels`` is."""
    from lumenairy.analysis.detector import apply_detector
    img, _xd, _yd = apply_detector(
        _uniform_field(), _DXF, pixel_pitch=pitch, n_pixels=n_pixels,
        quantum_efficiency=1.0, exposure_time=1.0, read_noise_e=0.0,
        dark_current_e_per_s=0.0, full_well=np.inf, seed=7)
    lit = img[img > 0]
    assert lit.size > 0
    expected = _I0 * pitch ** 2
    # Poisson shot noise on ~1e6..1e8 electrons averaged over the lit
    # pixels: 1e-3 relative is many sigma tighter than the 4x / 16x
    # errors this pins against.
    assert abs(lit.mean() / expected - 1.0) < 1e-3, (
        f"n_pixels={n_pixels}, pitch={pitch:g}: mean {lit.mean():.6e} "
        f"e-/pixel vs the required {expected:.6e}")


def test_s11_4_default_n_pixels_path_is_bit_identical():
    """The integer block-sum fast path is retained (and taken) exactly
    when the detector tiles the field, so the common call is unchanged."""
    from lumenairy.analysis.detector import apply_detector
    E = _uniform_field()
    a, xa, ya = apply_detector(E, _DXF, pixel_pitch=4e-6, seed=3,
                               quantum_efficiency=1.0, exposure_time=1.0,
                               read_noise_e=0.0, full_well=np.inf)
    b, xb, yb = apply_detector(E, _DXF, pixel_pitch=4e-6, n_pixels=16,
                               seed=3, quantum_efficiency=1.0,
                               exposure_time=1.0, read_noise_e=0.0,
                               full_well=np.inf)
    assert np.array_equal(a, b)
    assert np.array_equal(xa, xb) and np.array_equal(ya, yb)
    # And the returned axis really is spaced by pixel_pitch.
    assert np.allclose(np.diff(xa), 4e-6, rtol=1e-12, atol=0)


def test_s11_4_detector_flux_is_the_declared_detector_area():
    """A detector SMALLER than the field collects less light; a LARGER one
    collects the whole field.  Pre-fix the block-sum always swept up the
    entire field regardless of the declared detector extent."""
    from lumenairy.analysis.detector import apply_detector
    E = _uniform_field()
    field_flux = _I0 * (_NF * _DXF) ** 2
    for n_pixels, pitch in [(16, 2e-6), (16, 4e-6), (16, 8e-6)]:
        img, _x, _y = apply_detector(
            E, _DXF, pixel_pitch=pitch, n_pixels=n_pixels,
            quantum_efficiency=1.0, exposure_time=1.0, read_noise_e=0.0,
            dark_current_e_per_s=0.0, full_well=np.inf, seed=11)
        want = _I0 * min(n_pixels * pitch, _NF * _DXF) ** 2
        assert abs(img.sum() / want - 1.0) < 1e-3, (
            f"({n_pixels}, {pitch:g}): collected {img.sum():.6e} vs "
            f"{want:.6e} (field flux {field_flux:.6e})")


@pytest.mark.parametrize('n_pixels,pitch,match', [
    (0, 4e-6, 'n_pixels must be'),
    (-2, 4e-6, 'n_pixels must be'),
    (16, 0.0, 'pixel_pitch must be'),
    (16, -1e-6, 'pixel_pitch must be'),
    (16, np.inf, 'pixel_pitch must be'),
])
def test_s11_4_apply_detector_validates_its_geometry(n_pixels, pitch, match):
    """Pre-fix these reached the binning block and died inside numpy
    ("zero-size array to reduction operation maximum" / "can only specify
    one unknown dimension"), naming neither the function nor the arg."""
    from lumenairy.analysis.detector import apply_detector
    with pytest.raises(ValueError, match=match):
        apply_detector(_uniform_field(), _DXF, pixel_pitch=pitch,
                       n_pixels=n_pixels)


@pytest.mark.parametrize('bad', [0, -1, 2.5])
def test_s11_4_shack_hartmann_validates_the_reserved_knob(bad):
    """``detector_pixels_per_lenslet`` is RESERVED and inert, but a
    nonsense value must not pass silently."""
    from lumenairy.analysis.detector import shack_hartmann
    E = np.ones((64, 64), dtype=complex)
    with pytest.raises(ValueError, match='detector_pixels_per_lenslet'):
        shack_hartmann(E, 5e-6, 633e-9, 16 * 5e-6, 4e-3,
                       n_lenslets=4, detector_pixels_per_lenslet=bad)


# ===========================================================================
# S11-5: ao noise_sigma_pixels
# ===========================================================================

def _ao_probe(N=64, dx=1e-4):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X ** 2 + Y ** 2) / (N * dx / 2)
    return 0.5 * (2 * rho ** 2 - 1)


def _wfs(sigma, seed=None, dx=1e-4):
    from lumenairy.analysis.ao import make_shack_hartmann_wfs
    return make_shack_hartmann_wfs(
        subaperture_grid=8, n_modes=10, dx_pupil=dx, wavelength=633e-9,
        lenslet_focal=5e-3, noise_sigma_pixels=sigma, rng_seed=seed)


def test_s11_5_no_noise_path_is_deterministic_and_unchanged():
    r = _ao_probe()
    a = _wfs(0.0)(r)
    b = _wfs(0.0)(r)
    assert np.array_equal(a, b)
    assert np.all(np.isfinite(a))
    assert np.sqrt(np.mean(a[a != 0] ** 2)) > 1e-3


def test_s11_5_noise_sigma_pixels_is_effective_and_exactly_linear():
    """Pre-fix ``sigma = 1`` perturbed the reconstruction by 4.83e-5
    relative on this probe (i.e. the knob was inert).  It must now scale
    the perturbation EXACTLY linearly in sigma (same RNG sequence)."""
    r = _ao_probe()
    clean = _wfs(0.0)(r)
    mask = clean != 0
    rms_clean = float(np.sqrt(np.mean(clean[mask] ** 2)))
    deltas = {}
    for sigma in (0.001, 0.01, 0.1, 1.0):
        d = _wfs(sigma, seed=99)(r) - clean
        deltas[sigma] = float(np.sqrt(np.mean(d[mask] ** 2)))
    # Effective: sigma = 1 is now O(1)-or-larger relative to the signal,
    # not 5e-5.
    assert deltas[1.0] / rms_clean > 1.0, (
        f"noise_sigma_pixels is still inert: sigma=1 gives "
        f"{deltas[1.0] / rms_clean:.3e} relative")
    # Exactly linear (identical RNG draws, only the sigma multiplier).
    for a, b in ((0.001, 0.01), (0.01, 0.1), (0.1, 1.0)):
        assert abs(deltas[b] / deltas[a] - 10.0) < 1e-9, (
            f"sigma {a} -> {b}: ratio {deltas[b] / deltas[a]!r}")


def test_s11_5_seeded_noise_sequence_is_reproducible():
    r = _ao_probe()
    w1, w2 = _wfs(0.1, seed=7), _wfs(0.1, seed=7)
    assert np.array_equal(w1(r), w2(r))
    # ... and successive calls draw FRESH noise (v5.17 P3-01 semantics).
    w3 = _wfs(0.1, seed=7)
    first = w3(r)
    assert not np.array_equal(first, w3(r))


# ===========================================================================
# S11-6: small guards batch
# ===========================================================================

def test_s11_6a_resample_field_rejects_a_degenerate_default_N_out():
    from lumenairy.propagators.mft import resample_field
    E = np.ones((64, 64), dtype=complex)
    # dx_out so coarse that round(N*dx_in/dx_out) -> 0.  Pre-fix this
    # silently returned a (0, 0) array.
    with pytest.raises(ValueError, match='rounded to'):
        resample_field(E, 1e-6, 1e-3)
    with pytest.raises(ValueError, match='N_out must be'):
        resample_field(E, 1e-6, 1e-6, N_out=0)


@pytest.mark.parametrize('order', [-1, 6, 7, 2.5])
def test_s11_6a_resample_field_validates_order(order):
    from lumenairy.propagators.mft import resample_field
    E = np.ones((16, 16), dtype=complex)
    with pytest.raises(ValueError, match='order must be'):
        resample_field(E, 1e-6, 1e-6, order=order)


def test_s11_6a_resample_field_in_contract_calls_unchanged():
    from lumenairy.propagators.mft import resample_field
    E = np.zeros((32, 32), dtype=complex)
    E[16, 16] = 1.0
    out, dxo = resample_field(E, 1e-6, 2e-6)
    assert out.shape == (16, 16) and dxo == 2e-6
    out3, _ = resample_field(E, 1e-6, 1e-6)
    assert out3.shape == (32, 32)
    assert np.allclose(out3, E, atol=1e-12)


def test_s11_6b_zernike_modal_basis_supports_one_lenslet():
    """``n_lenslets = 1`` was ``0 / 0`` -> ``p = [nan]`` -> an EMPTY basis
    (reconstructor shape (n_modes, 0)) plus a bare RuntimeWarning."""
    from lumenairy.analysis.ao import zernike_modal_basis
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        b = zernike_modal_basis(n_modes=4, n_lenslets=1, semi_aperture=1e-3)
    assert b['reconstructor'].shape == (4, 2)
    assert np.all(np.isfinite(b['reconstructor']))
    xl, yl = b['lenslet_xy']
    assert xl.size == 1 and float(xl[0]) == 0.0 and float(yl[0]) == 0.0


def test_s11_6b_zernike_modal_basis_refuses_an_empty_disk():
    """``n_lenslets = 2`` puts all four samples at rho = sqrt(2) > 1, so
    the influence matrix is empty and every coefficient identically
    zero."""
    from lumenairy.analysis.ao import zernike_modal_basis
    with pytest.raises(ValueError, match='inside the unit pupil disk'):
        zernike_modal_basis(n_modes=4, n_lenslets=2, semi_aperture=1e-3)


def test_s11_6b_zernike_modal_basis_unchanged_for_n_ge_3():
    from lumenairy.analysis.ao import zernike_modal_basis
    b = zernike_modal_basis(n_modes=4, n_lenslets=11, semi_aperture=5e-3)
    xl, _yl = b['lenslet_xy']
    assert xl.size == 81           # the pre-fix count on this geometry
    assert np.all(np.isfinite(b['influence_matrix']))


def test_s11_6c_telecentric_entrance_pupil_does_not_nan_the_fans():
    """An object-space telecentric system has ``ep_z = inf``, so
    ``-ep_z * tan(field_angle)`` was ``inf * 0 = NaN`` on axis and
    poisoned every launched ray height."""
    from lumenairy.raytrace.ray_fan import _ep_offset
    assert _ep_offset(np.inf, 0.0) == 0.0
    assert _ep_offset(-np.inf, 0.0) == 0.0
    assert _ep_offset(np.inf, 0.01) == 0.0
    assert _ep_offset(np.nan, 0.01) == 0.0
    # Bit-identical for finite ep_z -- the arithmetic is untouched.
    for ep_z in (-0.1, 0.0, 0.25):
        for fa in (0.0, 0.01, -0.03):
            assert _ep_offset(ep_z, fa) == -ep_z * np.tan(fa)


def test_s11_6d_through_focus_rms_rejects_degenerate_inputs():
    from lumenairy.raytrace.ray_fan import through_focus_rms
    surfs = _singlet()
    with pytest.raises(ValueError, match='non-empty 1-D'):
        through_focus_rms(surfs, _WL, 10e-3, focus_shifts=[])
    with pytest.raises(ValueError, match='num_rings and rays_per_ring'):
        through_focus_rms(surfs, _WL, 10e-3, focus_shifts=[0.0],
                          num_rings=0)
    with pytest.raises(ValueError, match='num_rings and rays_per_ring'):
        through_focus_rms(surfs, _WL, 10e-3, focus_shifts=[0.0],
                          rays_per_ring=0)


def test_s11_6d_through_focus_rms_healthy_call_unchanged():
    from lumenairy.raytrace.ray_fan import through_focus_rms
    surfs = _singlet()
    shifts = np.linspace(0.19, 0.20, 11)
    zs, rms, best = through_focus_rms(surfs, _WL, 10e-3,
                                      focus_shifts=shifts,
                                      num_rings=3, rays_per_ring=8)
    assert np.array_equal(zs, shifts)
    assert np.all(np.isfinite(rms))
    assert 0.19 <= best <= 0.20


def test_s11_6d_trace_summary_rejects_empty_bundle_and_bad_units():
    from lumenairy.raytrace.layout import trace_summary
    surfs = _singlet()
    rb = RayBundle(x=np.zeros(0), y=np.zeros(0), z=np.zeros(0),
                   L=np.zeros(0), M=np.zeros(0), N=np.zeros(0),
                   opd=np.zeros(0), wavelength=_WL,
                   alive=np.zeros(0, bool))
    res = trace(rb, surfs, _WL, output_filter='last')
    with pytest.raises(ValueError, match='EMPTY'):
        trace_summary(res)
    fan = make_fan('y', 5e-3, 9, field_angle=0.0, wavelength=_WL)
    ok = trace(fan, surfs, _WL, output_filter='last')
    with pytest.raises(ValueError, match='units must be one of'):
        trace_summary(ok, units='cm')


def test_s11_6e_through_focus_nanmax_guard_mirrors_its_nanmin_sibling():
    """An all-NaN Strehl array must give a clean NaN, not an "All-NaN
    slice" RuntimeWarning (which RAISES under warnings-as-errors) -- the
    ``np.nanmin(rms_r)`` sibling two lines away was already guarded."""
    import inspect

    import lumenairy.analysis.through_focus as tf
    src = inspect.getsource(tf)
    assert src.count('np.any(np.isfinite(strehl))') == 2, (
        "both through_focus_scan and its JAX twin must guard nanmax")
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        s = np.full(5, np.nan)
        got = (float(np.nanmax(s))
               if (1.0 and np.any(np.isfinite(s))) else float('nan'))
        assert np.isnan(got)


# ===========================================================================
# S11-7: conic_sag NaN propagation
# ===========================================================================

_R7, _K7 = 50e-3, 0.5


@pytest.mark.parametrize('h,want', [
    (0.0, 0.0),
    (10e-3, 0.001015467617224473),        # in-domain, bit-pinned
    (40e-3, 0.026666666666666658),        # in-domain, bit-pinned
    (60e-3, 0.0),                         # out-of-domain REAL: still 0.0
])
def test_s11_7_finite_conic_sag_is_bit_identical(h, want):
    """The documented JAX zero-clamp for out-of-domain REAL heights is
    PRESERVED; every finite value is untouched to the last bit."""
    got = float(np.asarray(conic_sag(np.array([h]), np.zeros(1),
                                     _R7, _K7, (), xp=np)).ravel()[0])
    assert got == want, f"h={h}: {got!r} != {want!r}"


def test_s11_7_non_finite_input_propagates():
    """``conic_sag(nan)`` was ``+0.0`` -- a finite sag at a phantom vertex
    that a jax-traced Newton could converge onto -- while the sibling
    ``conic_sag_derivs(nan)`` already returned NaN."""
    for R in (_R7, np.inf):               # curved AND flat branches
        s_nan = np.asarray(conic_sag(np.array([np.nan]), np.zeros(1),
                                     R, _K7, (), xp=np)).ravel()[0]
        s_inf = np.asarray(conic_sag(np.array([np.inf]), np.zeros(1),
                                     R, _K7, (), xp=np)).ravel()[0]
        assert np.isnan(s_nan), f"R={R}: conic_sag(nan) = {s_nan!r}"
        assert np.isinf(s_inf), f"R={R}: conic_sag(inf) = {s_inf!r}"
        zx, zy = conic_sag_derivs(np.array([np.nan]),
                                  np.array([np.nan]), R, _K7, (), xp=np)
        assert np.isnan(zx[0]) and np.isnan(zy[0])


def test_s11_7_flat_surface_finite_sag_still_zero():
    s = conic_sag(np.array([0.0, 10e-3]), np.zeros(2), np.inf, 0.0, (),
                  xp=np)
    assert np.array_equal(np.asarray(s), np.zeros(2))
    zx, zy = conic_sag_derivs(np.array([0.0, 10e-3]), np.zeros(2),
                              np.inf, 0.0, (), xp=np)
    assert np.array_equal(np.asarray(zx), np.zeros(2))
    assert np.array_equal(np.asarray(zy), np.zeros(2))


def test_s11_7_jax_grad_and_jit_are_unpoisoned():
    """The NaN propagation must NOT newly NaN-poison ``jax.grad`` at
    healthy or out-of-domain REAL inputs -- that is the whole reason the
    zero-clamp exists."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)

    def f(h):
        return conic_sag(h, 0.0, _R7, _K7, (), xp=jnp)

    g = jax.grad(f)
    assert abs(float(f(10e-3)) - 0.001015467617224473) < 1e-15
    assert abs(float(g(10e-3)) - 0.20628424925175864) < 1e-12
    assert abs(float(g(40e-3)) - 4.0) < 1e-12
    # Out-of-domain REAL: value 0.0 AND gradient 0.0, exactly as before.
    assert float(f(60e-3)) == 0.0
    assert float(g(60e-3)) == 0.0
    # jit still traces.
    assert abs(float(jax.jit(f)(10e-3)) - 0.001015467617224473) < 1e-15


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

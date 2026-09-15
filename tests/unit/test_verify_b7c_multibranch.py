"""VERIFY-WP-B7c: the decisions WP-B7c's own pins do not make.

WP-B7c refuses a multibranch field whose bracketed grid power leaves a derived
band (``_MB_POWER_RATIO_MAX = _ENERGY_BLOWUP_FACTOR = 2.0``) and pins the
band's derivation as two literals -- a largest ACCEPTED reading of 1.246 and a
smallest BROKEN one of 5.848, "with nothing in between".  Re-derived on four
optics WP-B7c never used (a cemented doublet, a positive meniscus, a
convex-first plano-convex at 532 nm and a fast N-LASF9 biconvex), against the
same direct Rayleigh-Sommerfeld oracle
(``validation/oracles/caustic_fold_truth.py``; probes and JSON in
``validation/probe_verify_b7c/``), three of those claims do not hold, and the
tests here pin the decisions that replace them:

1. the ACCEPTED population is not bounded by 1.246.  The completion RETURNED
   fields reading up to 1.83 with ``power_ratio_decision == 'ok'``, and at
   those planes the field carried 1.8x the launched power while the
   ray-to-wave hand-off at the same plane carried 1.00x.  CLOSED by WP-B7c
   round 2: those planes are now refused on the pixel-continuity arm, and the
   test pins that decision instead;
2. the refusal was a property of the OUTPUT GRID, not of the field.  The same
   optic, plane and launch lattice rasterised onto a pixel half the size reads
   ~1/4 of the ratio, so a plane refused at one grid was accepted at another --
   the signature of the point-sampled quadrature WP-B7c correctly diagnosed,
   carried into the guard built on top of it.  The READING still does this
   (it is the mechanism, and the test still asserts it); the DECISION no
   longer does;
3. the bar's DENOMINATOR has more headroom than the bar.  The reading is
   ``min(power_ratio, power_ratio_triangles)`` and the two denominators
   separate by up to 7.8x on the very geometry that bracket was introduced
   for, so the smallest gain the refusal can see is the bar times that spread;
4. ``_ZETA_EXTRAPOLATION_MAX = 8.0`` does not order the energy error on every
   optic: below-bar rungs carry a LARGER one-sided gain than above-bar rungs
   on two of the four fixtures;
5. ``zeta_linear_range`` is reported and cannot reach the field (this one
   CONFIRMS WP-B7c);
6. the refusal fires where no ring coalesces (``n_branch_max == 1``).  Its
   message's mechanism USED to be unconditional; CLOSED by WP-B7c round 2,
   which conditions it on the plane's own branch count, and the test pins
   both arms of that branch.

Every pathology claim is premise-gated on the running build's own readings and
carries an unconditional invariant on the other arm; no test asserts a
per-build number.

Three of the six defects above were closed by WP-B7c round 2 (the
pixel-halving arbiter, ``test_audit2609_b7c2_pixel_halving_arbiter.py``) and
their tests are restated here as the DECISIONS that replaced them, not
removed: D1 (a wrong field returned inside the bar), D2 (the decision
following the grid) and D3 (the message's mechanism sentence).  D4 is
documented in ``_MB_POWER_RATIO_MAX``'s own comment and still pinned below;
D5 is recorded in ``validation/oracles/caustic_fold_truth.py``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements._lens_traced_multibranch import (
    apply_real_lens_traced_multibranch)
from lumenairy.elements._lens_traced_uniform import (
    _MB_PIXEL_CONTINUITY_MAX,
    _MB_POWER_RATIO_MAX,
    _ZETA_EXTRAPOLATION_MAX,
    apply_real_lens_traced_uniform,
)
from lumenairy.elements import _lens_traced_uniform as _U

# WP-B7c's own pinned envelope, restated here so the decisions below can be
# compared against it without importing its test module.
_WPB7C_LARGEST_ACCEPTED = 1.246
_WPB7C_SMALLEST_BROKEN = 5.848


# ===========================================================================
# fixtures -- four optics WP-B7c did not use
# ===========================================================================
def _gauss(N, dx, w0, dtype=np.complex128):
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(dtype)


def _doublet():
    """Cemented N-BK7 / N-SF6 doublet at 1.31 um -- OVERCORRECTED spherical
    aberration, so its marginal focus lies BEYOND the paraxial one and the
    through-focus planes below are single-valued (``n_branch_max == 1``)."""
    return {'wavelength': 1.31e-6, 'aperture_diameter': 1.40e-3, 'surfaces': [
        {'radius': 3.0e-3, 'thickness': 0.90e-3, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 0.70e-3},
        {'radius': -2.0e-3, 'thickness': 0.60e-3, 'glass_before': 'N-BK7',
         'glass_after': 'N-SF6', 'semi_diameter': 0.70e-3},
        {'radius': -6.0e-3, 'thickness': 0.0, 'glass_before': 'N-SF6',
         'glass_after': 'air', 'semi_diameter': 0.70e-3}],
        'thicknesses': [0.90e-3, 0.60e-3], 'stop_index': 0}


_Q = dict(presc=_doublet, wl=1.31e-6, N=512, dx=2.60e-6, w0=480e-6)


def _planoconvex():
    """N-SK16 plano-convex, CONVEX side first, at 532 nm."""
    return {'wavelength': 532e-9, 'aperture_diameter': 1.00e-3, 'surfaces': [
        {'radius': 2.4e-3, 'thickness': 0.85e-3, 'glass_before': 'air',
         'glass_after': 'N-SK16', 'semi_diameter': 0.50e-3},
        {'radius': float('inf'), 'thickness': 0.0,
         'glass_before': 'N-SK16', 'glass_after': 'air',
         'semi_diameter': 0.50e-3}],
        'thicknesses': [0.85e-3], 'stop_index': 0}


_S = dict(presc=_planoconvex, wl=532e-9, N=512, dx=1.50e-6, w0=330e-6)


def _fast_singlet():
    """N-LASF9 biconvex stopped to 0.60 mm (f/2.0) at 633 nm."""
    return {'wavelength': 633e-9, 'aperture_diameter': 0.60e-3, 'surfaces': [
        {'radius': 2.2e-3, 'thickness': 0.80e-3, 'glass_before': 'air',
         'glass_after': 'N-LASF9', 'semi_diameter': 0.50e-3},
        {'radius': -2.2e-3, 'thickness': 0.0, 'glass_before': 'N-LASF9',
         'glass_after': 'air', 'semi_diameter': 0.50e-3}],
        'thicknesses': [0.80e-3], 'stop_index': 0}


_F = dict(presc=_fast_singlet, wl=633e-9, N=640, dx=1.40e-6, w0=220e-6)


def _d3_singlet():
    """The delta-audit's D3 air-focus singlet -- a 6 mm aperture on a grid
    that holds a small fraction of its area, the geometry the multibranch's
    own gain bracket was introduced for."""
    return {'aperture_diameter': 6e-3, 'surfaces': [
        {'radius': 25e-3, 'conic': 0., 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 3e-3},
        {'radius': -25e-3, 'conic': 0., 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 3e-3}],
        'thicknesses': [3e-3, 40e-3]}


def _mb(fx, z, E=None, dx=None, **kw):
    """The branch sum's own diagnostics and the bracket the bar reads."""
    dx = fx['dx'] if dx is None else dx
    E = _gauss(fx['N'], dx, fx['w0']) if E is None else E
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _, d = apply_real_lens_traced_multibranch(
            E, prescription=fx['presc'](), wavelength=fx['wl'], dx=dx,
            output_plane_distance=z, return_diagnostics=True, **kw)
    vals = [float(v) for v in (d.get('power_ratio'),
                               d.get('power_ratio_triangles'))
            if v is not None and np.isfinite(v)]
    return (min(vals) if vals else None), d, rec


def _uni(fx, z, E=None, dx=None, **kw):
    dx = fx['dx'] if dx is None else dx
    E = _gauss(fx['N'], dx, fx['w0']) if E is None else E
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return apply_real_lens_traced_uniform(
            E, prescription=fx['presc'](), wavelength=fx['wl'], dx=dx,
            output_plane_distance=z, return_diagnostics=True, **kw)


# ===========================================================================
# 1. the accepted population is not bounded by the pinned ceiling
# ===========================================================================
def test_vb7c_an_accepted_field_reads_above_the_pinned_accepted_ceiling():
    """VERIFY-WP-B7c's D1, CLOSED by WP-B7c round 2's pixel-halving arbiter.

    As originally measured (2026-09-14) the cemented doublet at
    z = 5.400 .. 5.420 mm returned fields whose bracketed launched-power
    reading ran 1.106 / 1.226 / 1.352 / 1.532 / 1.826 -- every one above
    WP-B7c's pinned largest-accepted 1.246 by z = 5.412 mm, every one with
    ``power_ratio_decision == 'ok'``, and the completed field's fidelity
    against the Rayleigh-Sommerfeld oracle falling 0.837 -> 0.666 across them
    while its power ran 1.11x -> 1.83x the oracle's.  That was the R-5 defect
    class surviving the round-1 fix: a wrong field returned under healthy
    diagnostics, inside the bar.

    The round-2 arbiter reads the CONTINUITY of the field this call returns --
    its deposited power over that of the same mapped triangles rasterised at
    half the pitch on the same window -- and refuses these planes on it
    (1.100 / 1.265 / 1.636 at 5.400 / 5.410 / 5.420 mm, against a 1.06 bar).

    What is pinned here is the decision, two-sided: a plane whose
    launched-power reading is INSIDE the 2.0 bar and whose continuity is
    OUTSIDE 1.06 must be refused, and the refusal must name both readings.
    Premise-gated on this build still producing such a plane; the invariant on
    the other arm is that a RETURNED field is inside BOTH bands.
    """
    inside_and_refused = []
    for z in (5400e-6, 5412e-6, 5420e-6):
        br, d, _ = _mb(_Q, z)
        assert br is not None
        try:
            E_out, ud = _uni(_Q, z)
        except RuntimeError as exc:
            msg = str(exc)
            if 'NOT CONVERGED' not in msg:
                continue          # refused on the launched-power arm instead
            assert f'{br:.4g}' in msg, msg[:300]
            assert 'continuity ratio of' in msg, msg[:300]
            if br <= _MB_POWER_RATIO_MAX:
                inside_and_refused.append((z, br, msg))
            continue
        # the invariant, on every arm: nothing outside either band is RETURNED
        got = ud.get('multibranch_power_ratio_bracketed')
        assert got is not None and got <= _MB_POWER_RATIO_MAX, (
            f'z={z * 1e6:.0f} um returned a field reading {got:.4g}, '
            f'outside the {_MB_POWER_RATIO_MAX:g} bar')
        cont = ud.get('pixel_continuity')
        assert cont is None or cont <= _MB_PIXEL_CONTINUITY_MAX, (
            f'z={z * 1e6:.0f} um returned a field whose continuity reads '
            f'{cont:.4g}, outside the {_MB_PIXEL_CONTINUITY_MAX:g} bar')
        p_out = float(np.sum(np.abs(np.asarray(E_out)) ** 2)) * _Q['dx'] ** 2
        gain = p_out / float(d['launched_power'])
        assert gain <= 1.25, (
            f'z={z * 1e6:.0f} um returned a field gaining {gain:.4g}x the '
            f'launched power with both readings inside their bands')
    if not inside_and_refused:
        pytest.skip('no rung of this ladder is inside the launched-power bar '
                    'and outside the continuity bar on this build; the '
                    'invariant arm above ran instead')
    z, br, msg = inside_and_refused[0]
    assert br <= _MB_POWER_RATIO_MAX
    assert br > 1.0, (
        f'z={z * 1e6:.0f} um reads {br:.4g}: this rung no longer gains at all')


# ===========================================================================
# 2. the refusal is a property of the OUTPUT GRID
# ===========================================================================
def test_vb7c_the_refusal_follows_the_pixel_not_the_field():
    """VERIFY-WP-B7c's D2: the READING follows the output pixel -- CONFIRMED,
    and it is the mechanism -- while the DECISION no longer does, which is
    WP-B7c round 2's repair.

    ``dx -> dx/2`` with ``ray_subsample -> 2 * ray_subsample`` leaves the
    launch pitch ``ray_subsample * dx`` and the physical window unchanged, so
    the SAME mapped triangles are rasterised onto four times as many pixels.
    Each triangle still deposits one pixel-area's worth of ``|E|^2 / ratio``
    wherever it catches a pixel centre, so the spurious power falls with the
    pixel area -- measured 2.98 -> 1.42 -> 1.04 on the fast singlet at
    z = 1080 um, 107 -> 27.3 -> 7.40 on the plano-convex at z = 3274.02 um,
    504 -> 127 -> 32.2 on the meniscus, 11238 -> 2810 -> 703 on the doublet,
    i.e. a clean 1/4 per halving, while a HEALTHY plane is invariant
    (0.937 -> 0.937 -> 0.938).

    Round 1 decided on that reading, so the plane refused at the shipped grid
    was ACCEPTED one refinement later and a caller who refined their grid to
    resolve the Airy layer crossed the bar in the direction that removed the
    guard.  Round 2 decides on the RATIO of two renders one halving apart,
    which is taken at the caller's own pitch: measured 2026-09-15, the refined
    grid still reads 1.346 and is still refused.

    Both halves are asserted: the reading must still fall by ~4 (the
    mechanism), and the decision must survive (the repair).
    """
    z = 1080e-6
    br0, _d0, _ = _mb(_F, z)
    if br0 is None or br0 <= _MB_POWER_RATIO_MAX:
        pytest.skip('this build does not refuse the coarse-grid plane')
    with pytest.raises(RuntimeError):
        _uni(_F, z)
    # same launch lattice (ray_subsample * dx), half the pixel
    E2 = _gauss(2 * _F['N'], _F['dx'] / 2.0, _F['w0'])
    br1, _d1, _ = _mb(_F, z, E=E2, dx=_F['dx'] / 2.0, ray_subsample=4)
    assert br1 is not None
    # the MECHANISM, confirmed: the excess is proportional to the pixel area
    assert br1 < br0 / 1.5, (
        f'refining the pixel did not reduce the reading: {br0:.4g} -> '
        f'{br1:.4g}; the excess is then not proportional to the pixel area')
    # ...and the same physical plane is now inside the LAUNCHED-POWER band,
    # which is exactly why a bar on that quantity followed the grid
    assert br1 <= _MB_POWER_RATIO_MAX, (
        f'the refined grid still reads {br1:.4g} on the launched-power arm; '
        f'this plane no longer exhibits D2 on this build')
    # the REPAIR: the decision does not follow it
    with pytest.raises(RuntimeError) as exc:
        _uni(_F, z, E=E2, dx=_F['dx'] / 2.0, ray_subsample=4)
    assert 'NOT CONVERGED' in str(exc.value), (
        'the refined grid is accepted again: the decision follows the pixel '
        'once more.\n' + str(exc.value)[:300])


# ===========================================================================
# 3. the bar's denominator has more headroom than the bar
# ===========================================================================
def test_vb7c_the_gain_bracket_headroom_exceeds_the_refusal_bar():
    """DECISION: the refusal reads ``min(power_ratio, power_ratio_triangles)``
    and the two denominators separate by MORE than the bar, so the smallest
    gain the refusal can see on such a geometry is the bar times that spread.

    The bracket exists because ``p_in`` counts only launch nodes whose own
    mapped point lands on the grid while ``p_out`` counts every pixel a
    triangle covers, and on an aperture far wider than the grid that mismatch
    alone reads several x with the energy conserved.  Measured 2026-09-14 on
    the delta-audit's own D3 geometry (6 mm aperture, 1.2 mm grid,
    ``ray_subsample=8``) at z = 96 .. 130 mm: ``power_ratio`` reads
    2.02 .. 3.76 while ``power_ratio_triangles`` reads 0.45 .. 0.90, a spread
    up to 7.8x, and the completion returns with ``power_ratio_decision`` of
    'ok' / 'energy_loss'.  That is the bracket working as designed; what the
    test pins is the CONSEQUENCE -- the guard's detection floor is not 2.0.

    Premise-gated on finding such a plane; the invariant is the bracket's
    own ordering, which must hold at every plane.
    """
    presc = _d3_singlet()
    N, dx, wl = 48, 25e-6, 1.0e-6
    xs = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2).astype(np.complex128)
    worst = None
    for z in (0.100, 0.110, 0.120):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            _, d = apply_real_lens_traced_multibranch(
                E, prescription=presc, wavelength=wl, dx=dx,
                output_plane_distance=z, ray_subsample=8,
                return_diagnostics=True)
        pr = d.get('power_ratio')
        prt = d.get('power_ratio_triangles')
        assert pr is not None and prt is not None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, ud = apply_real_lens_traced_uniform(
                E, prescription=presc, wavelength=wl, dx=dx,
                output_plane_distance=z, ray_subsample=8,
                return_diagnostics=True)
        # the invariant: the DECISION is taken on the smaller denominator,
        # whatever the two read
        assert ud['multibranch_power_ratio_bracketed'] == pytest.approx(
            min(pr, prt), rel=1e-12)
        spread = pr / prt
        if worst is None or spread > worst[1]:
            worst = (z, spread, pr, prt, ud['power_ratio_decision'])
    z, spread, pr, prt, decision = worst
    if not (pr > _MB_POWER_RATIO_MAX >= prt):
        pytest.skip('this build does not separate the two denominators '
                    'across the bar on the D3 geometry')
    # the decision: the completion RETURNS a field whose own node-count
    # ratio is above the bar, because the bracket reads the other denominator
    assert decision != 'refused_energy_gain', decision
    assert spread > _MB_POWER_RATIO_MAX, (
        f'z={z * 1e3:.0f} mm: power_ratio {pr:.4g} against '
        f'power_ratio_triangles {prt:.4g} is a spread of {spread:.4g}, '
        f'not above the {_MB_POWER_RATIO_MAX:g} bar')


# ===========================================================================
# 4. the zeta bar does not order the energy error on every optic
# ===========================================================================
def test_vb7c_the_zeta_bar_does_not_order_the_energy_error():
    """DECISION: ``_ZETA_EXTRAPOLATION_MAX`` separates a SIGNED, centred error
    below it from a one-sided gain above it on WP-B7c's three optics; on the
    plano-convex and the meniscus here it orders nothing.

    Measured 2026-09-14, completed-field power against the branch sum's own
    ``launched_power``.  Plano-convex: BELOW the bar -1.5 % / +10.4 % / +7.2 %
    at ``zeta_extrapolation`` 3.63 / 5.07 / 7.49, against ABOVE the bar
    +8.5 % / +7.1 % / +8.3 % / +4.4 % at 12.1 / 22.2 / 53.1 / 246.  Meniscus:
    BELOW -1.7 % / +4.0 % / +16.6 % / +11.5 % at 2.29 / 3.20 / 4.73 / 7.61,
    against ABOVE +11.6 % / +11.5 % / +10.4 % at 14.0 / 33.4 / 156.  On both
    optics the two populations overlap completely and the LARGEST excursion
    in the ladder is below the bar, against WP-B7c's -2.2 .. +4.1 % below and
    +3.6 .. +30.1 % above.  (The same ladders scored against the oracle read
    +1.8 / +14.1 / +10.8 below and +12.2 / +10.7 / +11.9 / +7.9 above -- the
    overlap is the same either way.)

    Read here through the branch sum's own ``launched_power``, so the
    assertion needs no oracle: a lossless element cannot deliver more power
    to a plane than it launched, whichever side of the bar the plane is on.
    Premise-gated on the two planes' sides of the bar.
    """
    pairs = []
    for z in (3221.30e-6, 3188.70e-6):
        _br, d, _ = _mb(_S, z)
        E_out, ud = _uni(_S, z)
        if ud.get('zeta_extrapolation') is None:
            pytest.skip(f'z={z * 1e6:.2f} um no longer fits a fold ring')
        p_out = float(np.sum(np.abs(np.asarray(E_out)) ** 2)) * _S['dx'] ** 2
        pairs.append((float(ud['zeta_extrapolation']),
                      p_out / float(d['launched_power'])))
    below = [p for p in pairs if p[0] <= _ZETA_EXTRAPOLATION_MAX]
    above = [p for p in pairs if p[0] > _ZETA_EXTRAPOLATION_MAX]
    if not (below and above):
        pytest.skip('this build no longer straddles the bar at these planes')
    # the invariant: both sides gain, so "signed and centred below" is not a
    # property of this optic
    assert below[0][1] > 1.0, (
        f'below-bar rung zeta_x={below[0][0]:.4g} reads {below[0][1]:.4g}')
    # the decision: the bar does not order the two
    assert below[0][1] >= above[0][1], (
        f'below the bar (zeta_x={below[0][0]:.4g}) the gain is '
        f'{below[0][1]:.4g} against {above[0][1]:.4g} above it '
        f'(zeta_x={above[0][0]:.4g}) -- the bar orders them after all on '
        f'this build')


# ===========================================================================
# 5. zeta_linear_range is reported and cannot reach the field -- CONFIRMS
# ===========================================================================
def test_vb7c_zeta_linear_range_is_reported_and_never_applied():
    """CONFIRMS WP-B7c section 6.2: the curvature bound ``u* = 0.1 kappa/|q|``
    is carried in the diagnostics and the dark fill is NOT clipped by it.

    Proved rather than read: ``_trace_meridional_fold`` is monkey-patched to
    return an ABSURD ``zeta_linear_range`` -- 1e-12 m, far inside one Airy
    length, and 1e+6 m -- with every other key untouched.  If the fill were
    clipped by it the returned field would move; the digest is asserted
    byte-identical on both arms.  Unconditional.
    """
    z = 3214.78e-6
    E_ref, d = _uni(_S, z)
    assert d.get('reason') == 'fold_ring', d.get('reason')
    for key in ('zeta_linear_range', 'zeta_curvature', 'zeta_linear_resid',
                'dark_fill_depth', 'l_airy'):
        assert d.get(key) is not None, f'{key} missing from the diagnostics'
    assert np.isfinite(d['zeta_linear_range']) and d['zeta_linear_range'] > 0
    base = np.asarray(E_ref).tobytes()
    real = _U._trace_meridional_fold
    try:
        for val in (1e-12, 1e6):
            def patched(*a, _v=val, **kw):
                f = real(*a, **kw)
                if isinstance(f, dict) and f.get('ok'):
                    f = dict(f)
                    f['zeta_linear_range'] = _v
                    f['zeta_curvature'] = _v
                return f
            _U._trace_meridional_fold = patched
            E_p, dp = _uni(_S, z)
            assert np.asarray(E_p).tobytes() == base, (
                f'the field moved when zeta_linear_range was forced to '
                f'{val:g}; it is applied to the fill after all')
            assert dp['zeta_linear_range'] == val, (
                'the diagnostic does not carry what the fold returned')
    finally:
        _U._trace_meridional_fold = real


# ===========================================================================
# 6. the refusal fires where no ring coalesces
# ===========================================================================
def test_vb7c_the_refusal_fires_with_a_single_valued_ray_map():
    """DECISION: the refusal is not gated on branch coalescence.

    Its message states one mechanism unconditionally -- "The output plane is
    at or near the AXIAL point focus, where a whole RING of branches
    coalesces" -- while printing the reading that contradicts it ("with up to
    1 branches on one pixel").  Measured 2026-09-14 on the cemented doublet at
    z = 5.422 / 5.460 mm: ``n_branch_max == 1`` at both, the map is
    single-valued, there is no ring, and the completion refuses on a bracketed
    reading of 2.04 / 3.22.

    The refusal itself is defensible there (the underlying field IS wrong:
    fidelity 0.637 / 0.292 against the oracle).  What the test pins is that
    the reading and the narrative are independent, so a caller cannot use the
    message to diagnose their own plane.  Premise-gated.
    """
    hits = []
    for z in (5422e-6, 5460e-6):
        br, d, _ = _mb(_Q, z)
        if br is None or br <= _MB_POWER_RATIO_MAX:
            continue
        with pytest.raises(RuntimeError) as exc:
            _uni(_Q, z)
        msg = str(exc.value)
        # the invariant: the message names the reading it refused on
        assert f'{br:.4g}' in msg, msg[:200]
        hits.append((z, d.get('n_branch_max'), msg))
    if not hits:
        pytest.skip('this build does not refuse either doublet plane')
    single = [h for h in hits if h[1] == 1]
    if not single:
        pytest.skip('no refused plane on this build has a single-valued map')
    _z, nbr, msg = single[0]
    assert nbr == 1
    # CLOSED (WP-B7c round 2): the mechanism sentence is now conditioned on
    # this plane's own branch count, so a single-valued map is told what
    # actually failed -- the same point-sampled quadrature, without any
    # coalescence -- instead of being told a ring coalesced.
    assert 'RING of branches coalesces' not in msg, (
        f'z={_z * 1e6:.0f} um refuses with n_branch_max={nbr} -- a '
        f'single-valued map -- and the message states the ring mechanism '
        f'anyway:\n{msg[:400]}')
    assert 'NOT multi-valued' in msg, msg[:400]
    assert f'up to {nbr} branches' in msg, msg[:400]
    # the other side of the branch, so the narrative was not simply deleted:
    # a genuinely multi-valued refusal still names the ring
    multi = [h for h in hits if h[1] is not None and h[1] > 2]
    if multi:
        assert 'RING of branches coalesces' in multi[0][2], multi[0][2][:400]

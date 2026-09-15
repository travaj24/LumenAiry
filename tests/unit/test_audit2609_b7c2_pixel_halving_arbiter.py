"""WP-B7c round 2: the PIXEL-HALVING ARBITER.

VERIFY-WP-B7c confirmed WP-B7c's diagnosis of the multibranch blow-up -- the
rasteriser's point-sampled area quadrature stops being an unbiased estimator
of the launched energy once a ring collapses onto a handful of pixels -- and
strengthened it with a control WP-B7c had not run: hold the optic, the plane,
the LAUNCH LATTICE and the physical window fixed, halve the OUTPUT PIXEL, and
the excess divides by exactly 4, while a healthy plane is invariant.

It also REFUTED the guard built on that diagnosis.  ``_MB_POWER_RATIO_MAX``
reads the deposited power against the LAUNCHED power, so its accepted and
broken populations overlap (largest accepted 0.9804 against smallest broken
1.106 / 1.151, a gap of 1.13x under a 2.0 bar), and a plane reading 1.151 with
``power_ratio_decision == 'ok'`` returns a field of oracle fidelity 0.858 --
the R-5 class the round-1 fix set out to close, surviving 1.74x inside the
bar.  Its decision also followed the output GRID rather than the field.

The arbiter makes that control the DECISION.  It re-rasterises the same mapped
triangles at half the pitch on the same window and reads

    pixel_continuity = p_out(dx) / p_out(dx/2),

which is ~1 for a converged render and ~4 per halving for a quadrature that
has stopped being unbiased.  Every term that made the power ratio a mixture
cancels: both renders drop the same dark-side tail, straddle the same grid
boundary and lose the same off-screen light.

What is asserted here:

1. the reading and its decision reach the diagnostics on every return path,
   and the uniform's copy is the branch sum's own number to the bit
   (unconditional);
2. asking for the arbiter does not move the field (unconditional, byte-exact)
   -- the round-1 bit-identity guarantee, as a test rather than a probe;
3. the two-sided mechanism, measured on the running build: a converged fold
   plane reads inside the band and a blown-up one reads near 4
   (premise-gated on the ladder still blowing up; the invariant arm asserts
   the healthy plane unconditionally);
4. the R-5 class is closed: a plane INSIDE the power-ratio bar is refused on
   the continuity reading (premise-gated; the invariant is that nothing
   outside either band is ever RETURNED);
5. the decision survives halving the output pixel, which is what round 1's
   did not (premise-gated);
6. the loss arm REPORTS and does not refuse (premise-gated on finding a
   loss-arm plane; the constant relationship is unconditional);
7. the refusal's mechanism sentence is conditioned on its own branch count
   (VERIFY-WP-B7c D3), so a single-valued map is never told a ring coalesced;
8. the arbiter costs one rasterisation and not a second trace (structural:
   the trace entry point is called exactly once).

Every pathology claim is premise-gated on the running build's own reading;
every invariant is unconditional; no wall-clock assertion anywhere.
"""
from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pytest

from lumenairy.elements import _lens_traced_multibranch as _MB
from lumenairy.elements._lens_traced_multibranch import (
    _ARBITER_MAX_FINE_ENTRIES,
    _ENERGY_BLOWUP_FACTOR,
    _PIXEL_CONTINUITY_MAX,
    _PIXEL_CONTINUITY_MIN,
    _multibranch_render,
    apply_real_lens_traced_multibranch,
)
from lumenairy.elements._lens_traced_uniform import (
    _MB_PIXEL_CONTINUITY_MAX,
    _MB_PIXEL_CONTINUITY_MIN,
    _MB_POWER_RATIO_MAX,
    apply_real_lens_traced_uniform,
)

# ---------------------------------------------------------------------------
# VERIFY-B7b's own fold fixture -- the one WP-B7c reproduced the blow-up on and
# the one the round-1 decision tests already run, so nothing here needs a new
# optic to make its point.  Its fold ladder runs to z ~ 1760 um, its
# TRANSITION plane (inside the power-ratio bar, refused on continuity) is at
# z ~ 1761 um and its blow-up window starts at z ~ 1762 um.
# ---------------------------------------------------------------------------
_WL = 1.064e-6
_DX = 2.20e-6
_N = 512
_W0 = 330e-6

_HEALTHY_Z = 1758e-6          # fold ring, returned-field continuity 0.9930
_BLOWUP_Z = 1768e-6           # bracket 6097, branch-sum continuity 3.9984

# The R-5 plane -- INSIDE the launched-power bar and refused on continuity --
# is on the fast N-LASF9 singlet stopped to f/2.0 (VERIFY-WP-B7c's ``F_alt``),
# whose z = 1076 um is the counterexample that study built D1 on: bracketed
# launched-power ratio 1.151, ``power_ratio_decision == 'ok'``,
# ``fell_back == False``, ``zeta_extrapolation == 1.93`` -- and oracle
# fidelity 0.858 with 1.284x the oracle's power.
_F_WL, _F_DX, _F_N, _F_W0 = 633e-9, 1.40e-6, 640, 220e-6
_F_HEALTHY_Z = 1063e-6        # returned-field continuity 1.0038, fid 0.9889
_F_R5_Z = 1076e-6             # D1's plane: bracket 1.151, continuity 1.1856


def _f_presc():
    return {'wavelength': _F_WL, 'aperture_diameter': 0.60e-3, 'surfaces': [
        {'radius': 2.2e-3, 'thickness': 0.80e-3, 'glass_before': 'air',
         'glass_after': 'N-LASF9', 'semi_diameter': 0.50e-3},
        {'radius': -2.2e-3, 'thickness': 0.0, 'glass_before': 'N-LASF9',
         'glass_after': 'air', 'semi_diameter': 0.50e-3}],
        'thicknesses': [0.80e-3], 'stop_index': 0}


def _presc():
    return {'wavelength': _WL, 'aperture_diameter': 0.90e-3, 'surfaces': [
        {'radius': 2.6e-3, 'thickness': 0.70e-3, 'glass_before': 'air',
         'glass_after': 'N-BAF10', 'semi_diameter': 0.45e-3},
        {'radius': -2.6e-3, 'thickness': 0.0, 'glass_before': 'N-BAF10',
         'glass_after': 'air', 'semi_diameter': 0.45e-3}],
        'thicknesses': [0.70e-3], 'stop_index': 0}


def _gauss(N, dx, w0):
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _mb(z, *, arbiter=True, N=_N, dx=_DX, **kw):
    E = _gauss(N, dx, _W0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        E_out, d = _multibranch_render(
            E, prescription=_presc(), wavelength=_WL, dx=dx,
            output_plane_distance=z, return_diagnostics=True,
            pixel_halving_arbiter=arbiter, **kw)
    return np.asarray(E_out), d, rec


def _bracket(d):
    vals = [float(v) for v in (d.get('power_ratio'),
                               d.get('power_ratio_triangles'))
            if v is not None and np.isfinite(v)]
    return min(vals) if vals else None


def _uni(z, *, N=_N, dx=_DX, **kw):
    E = _gauss(N, dx, _W0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return apply_real_lens_traced_uniform(
            E, prescription=_presc(), wavelength=_WL, dx=dx,
            output_plane_distance=z, return_diagnostics=True, **kw)


def _f_mb(z, *, arbiter=True, N=_F_N, dx=_F_DX, **kw):
    E = _gauss(N, dx, _F_W0)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        E_out, d = _multibranch_render(
            E, prescription=_f_presc(), wavelength=_F_WL, dx=dx,
            output_plane_distance=z, return_diagnostics=True,
            pixel_halving_arbiter=arbiter, **kw)
    return np.asarray(E_out), d


def _f_uni(z, *, N=_F_N, dx=_F_DX, **kw):
    E = _gauss(N, dx, _F_W0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return apply_real_lens_traced_uniform(
            E, prescription=_f_presc(), wavelength=_F_WL, dx=dx,
            output_plane_distance=z, return_diagnostics=True, **kw)


# ===========================================================================
# 1. the reading reaches the diagnostics, and it is the branch sum's own
# ===========================================================================
def test_b7c2_the_continuity_reading_and_its_decision_reach_the_diagnostics():
    """Unconditional.  The uniform completion carries the arbiter's reading,
    its band and its decision on the return path, and the number it carries is
    the branch sum's own to the bit -- not a re-derivation that could drift.
    """
    _E, ud = _uni(_HEALTHY_Z)
    for key in ('pixel_continuity', 'pixel_continuity_of',
                'pixel_continuity_band', 'pixel_continuity_decision',
                'multibranch_pixel_continuity'):
        assert key in ud, f'{key} missing from the uniform diagnostics'
    assert ud['pixel_continuity_band'] == (_MB_PIXEL_CONTINUITY_MIN,
                                           _MB_PIXEL_CONTINUITY_MAX)
    # the reading names WHAT it was taken on, and the two possible answers are
    # the two fields this function can return
    assert ud['pixel_continuity_of'] in (
        'the completed fold field', 'the plain multibranch field',
        'the Pearcey cusp field'), ud['pixel_continuity_of']
    if ud['reason'] == 'fold_ring':
        assert ud['pixel_continuity_of'] == 'the completed fold field'
    else:
        assert ud['pixel_continuity_of'] != 'the completed fold field'
    _Emb, md, _rec = _mb(_HEALTHY_Z)
    assert md['pixel_continuity'] is not None, (
        'the branch sum did not measure the reading when it was asked for')
    # the BRANCH SUM's reading travels bit-for-bit; the uniform must not
    # recompute it
    assert (ud['multibranch_pixel_continuity']
            == float(md['pixel_continuity']))
    # and it is exactly the two deposited powers it claims to be
    assert md['grid_power'] > 0.0 and md['pixel_halved_power'] > 0.0
    assert md['pixel_continuity'] == pytest.approx(
        md['grid_power'] / md['pixel_halved_power'], rel=1e-15)
    # the DECIDING reading is the completed field's, and on a fold ring the
    # two are different numbers -- that difference is the whole point of
    # reading the returned field (see ``_arbitrate``)
    assert ud['pixel_continuity'] is not None
    assert (_MB_PIXEL_CONTINUITY_MIN <= ud['pixel_continuity']
            <= _MB_PIXEL_CONTINUITY_MAX)
    # the half-pitch RENDER is an internal hand-off and must not ride out on
    # the diagnostics: it is four times the size of the returned field, and a
    # caller who asked for diagnostics did not ask to be handed that
    assert 'pixel_halved_field' not in ud, (
        'the (2N, 2N) half-pitch render is leaking to the caller through the '
        'completion diagnostics')
    assert all(not isinstance(v, np.ndarray) or v.shape[0] <= _N
               for k, v in ud.items() if k.startswith('pixel_')), (
        'a pixel_* diagnostic carries an array larger than the output grid')


def test_b7c2_the_public_branch_sum_does_not_pay_for_the_arbiter():
    """Unconditional.  ``apply_real_lens_traced_multibranch`` is the public
    entry point and never asks for the second render, so the reading is absent
    there and the decision says which -- 'not_requested', never a silent
    ``None`` that a consumer could read as 'converged'."""
    E = _gauss(_N, _DX, _W0)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        _E_out, d = apply_real_lens_traced_multibranch(
            E, prescription=_presc(), wavelength=_WL, dx=_DX,
            output_plane_distance=_HEALTHY_Z, return_diagnostics=True)
    assert d['pixel_continuity'] is None
    assert d['pixel_continuity_decision'] == 'not_requested'
    assert d['pixel_halved_power'] is None


def test_b7c2_past_the_entry_cap_the_reading_says_so_and_the_field_returns():
    """Unconditional.  The arbiter allocates a ``(2N, 2N)`` complex image and
    the consumer that completes it allocates one more, so it is capped.  Above
    the cap the reading must be REPORTED ABSENT -- never silently ``None``,
    which a consumer could read as "converged" -- and the call must still
    return a field.

    Forced through the constant rather than through a large grid, so the test
    costs one small call: the cap is engineered, not hoped for (the grid a
    real caller would need to cross it is N > 1322).
    """
    real = _MB._ARBITER_MAX_FINE_ENTRIES
    try:
        _MB._ARBITER_MAX_FINE_ENTRIES = 1
        _E, md, _rec = _mb(_HEALTHY_Z)
        assert md['pixel_continuity'] is None
        assert md['pixel_continuity_decision'] == 'not_measured'
        assert md['pixel_halved_field'] is None
        E_out, ud = _uni(_HEALTHY_Z)
        assert ud['pixel_continuity'] is None
        assert ud['pixel_continuity_decision'] == 'not_measured'
        assert np.all(np.isfinite(np.asarray(E_out)))
    finally:
        _MB._ARBITER_MAX_FINE_ENTRIES = real
    # and with the cap back, the same plane is measured again
    _E2, md2, _rec2 = _mb(_HEALTHY_Z)
    assert md2['pixel_continuity'] is not None


def test_b7c2_the_branch_sum_is_reached_through_exactly_one_seam():
    """Unconditional.  The completion reaches the branch sum through ONE
    module-level name, and patching it takes effect.

    Round 2 moved that name from the public
    ``apply_real_lens_traced_multibranch`` to the module-private
    ``_multibranch_render``, because the completion asks for a reading the
    public signature does not carry.  Five existing tests inject a synthetic
    multibranch field through this seam (``test_niche_r2_pearcey_cusp.py`` and
    ``test_niche_r5_gbd_vector_catastrophe.py``) and were re-pointed with it.

    What is pinned is that there is no SECOND name: if the public one were
    re-imported here as well, a test patching it would be silently ignored and
    would pass while testing nothing.  With only one name, patching the wrong
    one raises ``AttributeError`` -- loud, not silent.
    """
    import lumenairy.elements._lens_traced_uniform as U
    assert hasattr(U, '_multibranch_render')
    assert not hasattr(U, 'apply_real_lens_traced_multibranch'), (
        'a second branch-sum name is reachable from the completion module; a '
        'test patching it would not take effect')

    class _Reached(Exception):
        pass

    seen = []

    def _recorder(E_in, **kw):
        seen.append(kw.get('pixel_halving_arbiter'))
        raise _Reached

    real = U._multibranch_render
    try:
        U._multibranch_render = _recorder
        with pytest.raises(_Reached):
            _uni(_HEALTHY_Z)
    finally:
        U._multibranch_render = real
    assert seen == [True], (
        f'the completion did not reach the seam, or did not ask for the '
        f'reading: {seen}')


# ===========================================================================
# 2. asking for the arbiter does not move the field
# ===========================================================================
@pytest.mark.parametrize('z', [_HEALTHY_Z, _BLOWUP_Z])
def test_b7c2_the_arbiter_does_not_move_the_field(z):
    """Unconditional, byte-exact, on a healthy plane AND a blown-up one.

    The second render is a MEASUREMENT: it reads the same mapped triangles
    onto a finer grid and throws that grid away.  If it could touch the
    returned field -- through a shared buffer, a reordered accumulation or a
    mutated closure -- every field this module has ever returned would move,
    which is the one thing round 2 promised it would not do.
    """
    a, da, _ = _mb(z, arbiter=True)
    b, db, _ = _mb(z, arbiter=False)
    assert hashlib.sha256(a.tobytes()).hexdigest() == \
        hashlib.sha256(b.tobytes()).hexdigest(), (
            f'z={z * 1e6:.0f} um: the field moved when the arbiter was asked '
            f'for')
    for key in ('power_ratio', 'power_ratio_triangles', 'launched_power',
                'launched_power_triangles', 'n_branch_max',
                'n_triangles_degenerate', 'grid_power'):
        assert da[key] == db[key], f'{key} moved with the arbiter on'


# ===========================================================================
# 3. the mechanism, two-sided, measured on the running build
# ===========================================================================
def test_b7c2_a_converged_render_reads_one_and_a_blown_up_one_reads_four():
    """The quadrature identity, as a DECISION rather than a reading.

    A point-sampled deposit of ``dx^2 |E|^2 / ratio`` on every pixel centre a
    mapped triangle covers estimates the launched energy without bias while
    those triangles are spread over many pixels, so the deposited power does
    not depend on the pitch: a converged render reads ~1 at any ``dx``.  Where
    a ring collapses onto a handful of pixels the written power follows the
    PIXEL AREA instead, so halving the pitch divides it by 4.

    Measured 2026-09-15 on this fixture: 0.995 at the healthy plane against
    3.998 at z = 1768 um, and 3.90 / 4.00 / 4.00 across the whole blow-up
    window -- the same 1/4 per halving VERIFY-WP-B7c measured by hand on four
    other optics (S 107 -> 27.3 -> 7.40, M 504 -> 127 -> 32.2, Q 11238 ->
    2810 -> 703, F 2.98 -> 1.42 -> 1.04) against an invariant healthy plane
    (0.937 -> 0.937 -> 0.938).

    The healthy arm is UNCONDITIONAL; the blow-up arm is premise-gated on the
    ladder still blowing up on this build.
    """
    _E, d_ok, _ = _mb(_HEALTHY_Z)
    c_ok = d_ok['pixel_continuity']
    assert c_ok is not None
    # a BASIN, not the shipped bar: what is asserted is which of the two
    # regimes the reading is in (~1 against ~4), so the boundary sits far from
    # both.  The decision at this plane is asserted separately, below.
    assert 0.80 < c_ok < 1.25, (
        f'the healthy fold plane reads {c_ok:.4f}; a converged render reads '
        f'~1 at any pitch, so either the fold moved or the quadrature did')
    _Eu, ud_ok = _uni(_HEALTHY_Z)
    assert ud_ok['pixel_continuity_decision'] == 'ok', (
        f"the healthy fold plane is not accepted: "
        f"{ud_ok['pixel_continuity_decision']} at "
        f"{ud_ok['pixel_continuity']!r}")
    _E2, d_bad, _ = _mb(_BLOWUP_Z)
    if _bracket(d_bad) is None or _bracket(d_bad) <= 10.0:
        pytest.skip('this build no longer blows up at the pinned plane; the '
                    'healthy invariant above ran instead')
    c_bad = d_bad['pixel_continuity']
    assert c_bad is not None
    # the DECISION: the blow-up is a pixel-area scaling, i.e. it reads near 4
    # and not near 1.  The basin is generous (2.5 .. 5.0) because what is
    # asserted is which of the two regimes the reading is in, not its value.
    assert 2.5 < c_bad < 5.0, (
        f'the blown-up plane reads {c_bad:.4f}; the quadrature identity says '
        f'~4 per halving of the pitch')
    assert c_bad / c_ok > 2.5


# ===========================================================================
# 4. the R-5 class is closed
# ===========================================================================
def test_b7c2_a_plane_inside_the_power_ratio_bar_is_refused_on_continuity():
    """DECISION: the arm that decides the cases the round-1 tripwire cannot --
    VERIFY-WP-B7c's D1 counterexample, closed.

    At z = 1076 um on the fast N-LASF9 singlet the round-1 guard READ its best
    values on a broken field: bracketed launched-power ratio 1.151 with
    ``power_ratio_decision == 'ok'`` (1.74x inside the 2.0 bar),
    ``fell_back == False``, ``zeta_extrapolation == 1.93`` -- and the returned
    field at oracle fidelity 0.858 carrying 1.284x the oracle's power.  That
    is the R-5 defect class ("its own diagnostics read their best values on a
    broken field") surviving the round-1 fix.  The continuity of the field it
    would return reads 1.186 there, and it is refused.

    Premise-gated on this build still producing such a plane; the INVARIANT on
    the other arm is that a RETURNED field is inside BOTH bands.
    """
    _E, d = _f_mb(_F_R5_Z)
    br = _bracket(d)
    assert br is not None
    if br > _MB_POWER_RATIO_MAX:
        pytest.skip('this build refuses the pinned plane on the launched-power '
                    'arm, so it is not inside that bar here')
    try:
        _E2, ud = _f_uni(_F_R5_Z)
    except RuntimeError as exc:
        msg = str(exc)
        assert 'NOT CONVERGED' in msg, msg[:300]
        # it must name the reading it refused on, the OTHER arm's reading, and
        # a member that works
        assert f'{br:.4g}' in msg, msg[:300]
        assert 'continuity ratio of' in msg
        assert 'caustic=' in msg and 'wave' in msg, (
            'the refusal must point at a member that works')
        assert 'REFINING THE GRID IS NOT A WORKAROUND' in msg
        return
    # the invariant arm: whatever this build reads, a RETURNED field is inside
    # both bands
    assert ud['multibranch_power_ratio_bracketed'] <= _MB_POWER_RATIO_MAX
    assert ud['pixel_continuity'] is not None
    assert ud['pixel_continuity'] <= _MB_PIXEL_CONTINUITY_MAX
    pytest.skip('this build returns the pinned plane; its continuity reads '
                f"{ud['pixel_continuity']:.4f}, inside the band")


# ===========================================================================
# 5. the decision survives halving the output pixel (VERIFY-WP-B7c D2)
# ===========================================================================
def test_b7c2_the_decision_survives_halving_the_output_pixel():
    """DECISION: refine the OUTPUT PIXEL while holding the optic, the plane
    and the LAUNCH LATTICE fixed, and the refusal does NOT go away.

    This is VERIFY-WP-B7c's D2 against round 1, reproduced on its own
    reproducer.  The fast singlet at z = 1080 um reads a bracketed
    launched-power ratio of 2.982 at dx = 1.40 um and 1.420 at dx = 0.70 um,
    so round 1 refused the coarse grid and ACCEPTED the fine one -- one
    halving of the pixel removed the guard, and a caller who refined their
    grid to resolve the Airy layer (which this module's own ``l_airy`` gate
    tells them to do) crossed the bar in the direction that removed it.

    The continuity reading is a RATIO of two renders one halving apart, so it
    is taken at the caller's own pitch and does not fall with it: measured
    2026-09-15, the refined grid still reads 1.346 and is still refused.

    Premise-gated on the coarse grid being refused at all.
    """
    z = 1080e-6
    _E, d0 = _f_mb(z)
    br0 = _bracket(d0)
    assert br0 is not None
    coarse_refused = True
    try:
        _E0, _ud0 = _f_uni(z)
        coarse_refused = False
    except RuntimeError:
        pass
    if not coarse_refused:
        pytest.skip('this build does not refuse the coarse-grid plane')
    # same launch lattice (ray_subsample * dx), half the pixel, same window
    fine_entries = 4 * (2 * _F_N) ** 2
    if fine_entries > _ARBITER_MAX_FINE_ENTRIES:
        pytest.skip('the refined grid is past the arbiter entry cap on this '
                    'build, so the reading is not measured there')
    _E1, d1 = _f_mb(z, N=2 * _F_N, dx=_F_DX / 2.0, ray_subsample=4)
    br1 = _bracket(d1)
    assert br1 is not None
    # round 1's reading does exactly what VERIFY-WP-B7c said it does
    assert br1 < br0 / 1.5, (
        f'the launched-power ratio did not fall with the pixel area: '
        f'{br0:.4g} -> {br1:.4g}')
    if br1 > _MB_POWER_RATIO_MAX:
        pytest.skip('the refined grid is still outside the launched-power '
                    'bar on this build, so this plane does not exhibit D2')
    # ...and the refined grid is STILL refused, now on the continuity arm
    with pytest.raises(RuntimeError) as exc:
        _f_uni(z, N=2 * _F_N, dx=_F_DX / 2.0, ray_subsample=4)
    assert 'NOT CONVERGED' in str(exc.value), str(exc.value)[:300]


# ===========================================================================
# 6. the loss arm reports, and the band's shape
# ===========================================================================
def test_b7c2_the_band_is_two_sided_and_the_loss_arm_only_reports():
    """The band is symmetric in the LOG of the reading, and only its gain arm
    refuses.

    Why the asymmetry is physical rather than timid: the branch sum's
    BRIGHT side is kept verbatim by the completion, so a bright-side excess
    reaches the caller's field; its DARK side is exactly what the completion
    replaces, so a dark-side deficit does not.  Measured (2026-09-15): at an
    air-spaced doublet plane reading 0.9419 the completed field's power is
    0.997x the oracle's and its fidelity 0.991 -- the branch sum is 6 % short
    and the completion has already made it good.  Refusing on that arm would
    remove fields that are right.

    The constant relationship is unconditional; the loss-arm demonstration is
    premise-gated on this build producing such a plane on the ladder.
    """
    assert _MB_PIXEL_CONTINUITY_MAX == _PIXEL_CONTINUITY_MAX
    assert _MB_PIXEL_CONTINUITY_MIN == _PIXEL_CONTINUITY_MIN
    assert _MB_PIXEL_CONTINUITY_MIN == pytest.approx(
        1.0 / _MB_PIXEL_CONTINUITY_MAX, rel=1e-12), (
        'the band is no longer symmetric in the log of the reading; both arms '
        'must move together or the asymmetry needs its own derivation')
    assert 1.0 < _MB_PIXEL_CONTINUITY_MAX < _ENERGY_BLOWUP_FACTOR
    # the loss arm is read on the N-SK16 plano-convex, whose under-resolved
    # Airy layer past z = 3.24 mm makes the coarse render miss deposits the
    # finer one catches (measured 0.854 at z = 3245 um, with the returned
    # field still at oracle fidelity 0.904).
    presc = {'wavelength': 532e-9, 'aperture_diameter': 1.00e-3, 'surfaces': [
        {'radius': 2.4e-3, 'thickness': 0.85e-3, 'glass_before': 'air',
         'glass_after': 'N-SK16', 'semi_diameter': 0.50e-3},
        {'radius': float('inf'), 'thickness': 0.0,
         'glass_before': 'N-SK16', 'glass_after': 'air',
         'semi_diameter': 0.50e-3}],
        'thicknesses': [0.85e-3], 'stop_index': 0}
    N, dx, w0, wl = 512, 1.50e-6, 330e-6, 532e-9
    E = _gauss(N, dx, w0)
    loss = None
    for z_um in (3245, 3255, 3250):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            _E_mb, d = _multibranch_render(
                E, prescription=presc, wavelength=wl, dx=dx,
                output_plane_distance=z_um * 1e-6, return_diagnostics=True,
                pixel_halving_arbiter=True)
        c = d['pixel_continuity']
        if c is not None and c < _MB_PIXEL_CONTINUITY_MIN:
            loss = (z_um, c, d)
            break
    if loss is None:
        pytest.skip('no plane of this ladder reads below the loss arm on this '
                    'build')
    z_um, c, d = loss
    assert d['pixel_continuity_decision'] == 'not_converged_loss'
    # the decision: it REPORTS -- the completion still returns a FINITE field
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E2, ud = apply_real_lens_traced_uniform(
            E, prescription=presc, wavelength=wl, dx=dx,
            output_plane_distance=z_um * 1e-6, return_diagnostics=True)
    assert ud['pixel_continuity_decision'] == 'not_converged_loss'
    assert np.all(np.isfinite(np.asarray(E2)))


# ===========================================================================
# 7. the refusal's mechanism sentence (VERIFY-WP-B7c D3)
# ===========================================================================
def test_b7c2_the_refusal_does_not_claim_a_ring_on_a_single_valued_map():
    """DECISION: the mechanism sentence is conditioned on the plane's own
    branch count.

    VERIFY-WP-B7c D3: at a cemented doublet's z = 5.422 / 5.460 mm the
    round-1 refusal printed "with up to 1 branches on one pixel" and then
    "where a whole RING of branches coalesces" in the next clause.  The
    refusal was right (oracle fidelity 0.637 / 0.292) but its narrative
    contradicted its own reading, so a caller could not use it to diagnose
    their plane.

    Premise-gated on this build refusing a single-valued plane; the invariant
    on the other arm is that a MULTI-valued refusal still names the ring.
    """
    presc = {'wavelength': 1.31e-6, 'aperture_diameter': 1.40e-3,
             'surfaces': [
                 {'radius': 3.0e-3, 'thickness': 0.90e-3,
                  'glass_before': 'air', 'glass_after': 'N-BK7',
                  'semi_diameter': 0.70e-3},
                 {'radius': -2.0e-3, 'thickness': 0.60e-3,
                  'glass_before': 'N-BK7', 'glass_after': 'N-SF6',
                  'semi_diameter': 0.70e-3},
                 {'radius': -6.0e-3, 'thickness': 0.0,
                  'glass_before': 'N-SF6', 'glass_after': 'air',
                  'semi_diameter': 0.70e-3}],
             'thicknesses': [0.90e-3, 0.60e-3], 'stop_index': 0}
    N, dx, w0 = 512, 2.60e-6, 480e-6
    E = _gauss(N, dx, w0)
    single = None
    for z_um in (5460, 5430):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            _E_mb, d = apply_real_lens_traced_multibranch(
                E, prescription=presc, wavelength=1.31e-6, dx=dx,
                output_plane_distance=z_um * 1e-6, return_diagnostics=True)
        if _bracket(d) is None or _bracket(d) <= _MB_POWER_RATIO_MAX:
            continue
        if int(d['n_branch_max']) > 2:
            continue
        with pytest.raises(RuntimeError) as exc:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                apply_real_lens_traced_uniform(
                    E, prescription=presc, wavelength=1.31e-6, dx=dx,
                    output_plane_distance=z_um * 1e-6,
                    return_diagnostics=True)
        single = (z_um, int(d['n_branch_max']), str(exc.value))
        break
    if single is None:
        pytest.skip('this build has no single-valued refused plane on the '
                    'doublet ladder')
    z_um, nbr, msg = single
    assert f'up to {nbr} branches' in msg, msg[:300]
    assert 'RING of branches coalesces' not in msg, (
        f'z={z_um} um refuses with n_branch_max={nbr} -- a single-valued map '
        f'-- and the message still states the ring mechanism:\n{msg[:400]}')
    assert 'NOT multi-valued' in msg, msg[:300]
    # the invariant on the other side: a genuinely multi-valued refusal DOES
    # name the ring, so the branch was not simply deleted
    with pytest.raises(RuntimeError) as exc2:
        _uni(_BLOWUP_Z)
    assert 'RING of branches coalesces' in str(exc2.value)


# ===========================================================================
# 8. cost: one rasterisation, not a second trace
# ===========================================================================
def test_b7c2_the_arbiter_costs_one_rasterisation_and_not_a_second_trace():
    """STRUCTURAL, not wall-clock.  The expensive half of a branch-sum call is
    the ray trace through the prescription and the KMAH / Jacobian pass over
    the launch lattice; the arbiter is affordable only because it re-uses all
    of it and changes nothing but the output sampling.

    Asserted by counting calls to the trace entry point: exactly one, with the
    arbiter on and with it off.  A future refactor that obtained the second
    render by re-entering the branch sum would double the trace and this test
    would say so.
    """
    calls = []
    real = _MB._trace_launch_grid

    def counted(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    _MB._trace_launch_grid = counted
    try:
        calls.clear()
        _mb(_HEALTHY_Z, arbiter=False)
        n_off = len(calls)
        calls.clear()
        _mb(_HEALTHY_Z, arbiter=True)
        n_on = len(calls)
    finally:
        _MB._trace_launch_grid = real
    assert n_off == 1, f'the plain branch sum traced {n_off} times'
    assert n_on == 1, (
        f'the arbiter traced the prescription {n_on} times; it must re-use '
        f'the trace and re-rasterise only')

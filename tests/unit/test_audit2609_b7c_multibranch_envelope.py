"""WP-B7c: the multibranch 1/sqrt|J| blow-up the uniform completion used to
pass through, and the fold envelope's two constants.

Handoff items 4.2 and 4.3 of ``HANDOFF_2026_09_14.md`` (VERIFY-B7b request R-5
and its section 4.2).  Below ``zeta_extrapolation`` ~0.35 on VERIFY-B7b's fold
fixture ``apply_real_lens_traced_multibranch``'s reconstruction returns 94x to
6095x the correct power, and ``apply_real_lens_traced_uniform`` used to pass
that field to the caller with ``fell_back=False``, a healthy ``fit_residual``
and the best ``zeta_extrapolation`` its own docstring records -- its diagnostics
reporting the best case on a field wrong by three decades.

What is asserted here:

1. the multibranch energy reading and the DECISION taken on it reach the uniform
   diagnostics on every return path (unconditional);
2. a multibranch field outside the derived band is REFUSED, not passed through
   and not fallen back (premise-gated: the arm runs only where the running
   build actually reproduces the blow-up; where it does not, the INVARIANT --
   nothing outside the band is ever returned -- is asserted instead);
3. the two bars are coupled to the constants they were derived against
   (build-free: constants only);
4. the completion is NOT the best member on an aperture-truncated fixture, read
   through ENERGY CONSERVATION, which needs no model: a lossless element cannot
   deliver more power to a plane than it launched.

Every pathology claim is premise-gated; every invariant is unconditional.  No
wall-clock assertion anywhere.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements._lens_traced_multibranch import (
    _ENERGY_BLOWUP_FACTOR,
    _ENERGY_COLLAPSE_FACTOR,
    apply_real_lens_traced_multibranch,
)
from lumenairy.elements._lens_traced_uniform import (
    _AIRY_TAIL_CELLS,
    _MB_POWER_RATIO_MAX,
    _MB_POWER_RATIO_MIN,
    _ZETA_EXTRAPOLATION_MAX,
    apply_real_lens_traced_uniform,
)

_WL = 1.064e-6
_DX = 2.20e-6
_N = 512
_W0 = 330e-6

# VERIFY-B7b section 4.1's own fold fixture: N-BAF10 biconvex R = +/-2.6 mm,
# t = 0.70 mm, 0.90 mm aperture.  Its fold ladder runs z = 1565 .. 1760 um and
# its blow-up window starts at z ~ 1762 um.
_HEALTHY_Z = 1750e-6
_BLOWUP_LADDER = (1762e-6, 1764e-6, 1768e-6, 1780e-6)


def _presc():
    return {'wavelength': _WL, 'aperture_diameter': 0.90e-3, 'surfaces': [
        {'radius': 2.6e-3, 'thickness': 0.70e-3, 'glass_before': 'air',
         'glass_after': 'N-BAF10', 'semi_diameter': 0.45e-3},
        {'radius': -2.6e-3, 'thickness': 0.0, 'glass_before': 'N-BAF10',
         'glass_after': 'air', 'semi_diameter': 0.45e-3}],
        'thicknesses': [0.70e-3], 'stop_index': 0}


def _gauss(N=_N, dx=_DX, w0=_W0):
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _uniform(E, z, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = apply_real_lens_traced_uniform(
            E, prescription=_presc(), wavelength=_WL, dx=_DX,
            output_plane_distance=z, return_diagnostics=True, **kw)
    return out[0], out[1], rec


def _mb_bracket(E, z, **kw):
    """The multibranch's own bracketed gain reading at ``z``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _, d = apply_real_lens_traced_multibranch(
            E, prescription=_presc(), wavelength=_WL, dx=_DX,
            output_plane_distance=z, return_diagnostics=True, **kw)
    vals = [float(v) for v in (d.get('power_ratio'),
                               d.get('power_ratio_triangles'))
            if v is not None and np.isfinite(v)]
    return (min(vals) if vals else None), d, rec


# ===========================================================================
# 1. the reading and the decision reach the diagnostics -- unconditional
# ===========================================================================
def test_b7c_the_energy_reading_and_its_decision_reach_the_uniform_diagnostics():
    """The uniform completion's diagnostics carry the multibranch power ratio
    it was built on, the bracket the decision is taken on, the band and the
    decision -- and the ratio is the multibranch's OWN reading, not a
    re-derivation.

    This is the defect's minimum repair: before it the only place the blow-up
    appeared was a RuntimeWarning from another module, while this function's
    own ``fell_back`` / ``fit_residual`` / ``zeta_extrapolation`` all read
    their best values.
    """
    E = _gauss()
    _, d, _ = _uniform(E, _HEALTHY_Z)
    assert d['fell_back'] is False and d['reason'] == 'fold_ring', d['reason']
    for key in ('multibranch_power_ratio', 'multibranch_power_ratio_bracketed',
                'multibranch_power_ratio_band', 'power_ratio_decision'):
        assert key in d, f'{key} missing from the uniform diagnostics'
    assert d['power_ratio_decision'] == 'ok', d['power_ratio_decision']
    assert d['multibranch_power_ratio_band'] == (_MB_POWER_RATIO_MIN,
                                                 _MB_POWER_RATIO_MAX)
    # the same number the multibranch itself reports, to the bit
    br, mbd, _ = _mb_bracket(E, _HEALTHY_Z)
    assert d['multibranch_power_ratio'] == mbd['power_ratio']
    assert d['multibranch_power_ratio_bracketed'] == br
    # ...and the decision is the band membership, not an independent guess
    assert _MB_POWER_RATIO_MIN <= br <= _MB_POWER_RATIO_MAX, br


def test_b7c_a_fallback_also_carries_the_energy_decision():
    """A fallback returns the multibranch field itself, so the reading has to
    travel with it.  The blow-up window contains planes that fall back
    (``reason='zeta_nonlinear'``) and used to return a field carrying thousands
    of times the launched power with ``fell_back=True`` and nothing else said.
    """
    # a carrier tilt is the cheapest fallback that does not need a caustic
    E = _gauss()
    x = (np.arange(_N) - _N / 2.0) * _DX
    X, _Y = np.meshgrid(x, x)
    E_tilt = E * np.exp(1j * 2.0 * np.pi / _WL * 0.02 * X)
    _, d, _ = _uniform(E_tilt, _HEALTHY_Z, input_carrier='auto')
    assert d['fell_back'] is True, d
    assert 'power_ratio_decision' in d and 'multibranch_power_ratio' in d
    assert d['power_ratio_decision'] in ('ok', 'energy_loss',
                                         'no_launched_power'), d


# ===========================================================================
# 2. the refusal -- premise-gated pathology, unconditional invariant
# ===========================================================================
def test_b7c_a_blown_up_multibranch_is_refused_not_passed_through():
    """Two-sided DECISION test.

    Premise: on this build, at least one plane of the ladder drives the
    multibranch's bracketed gain above ``_MB_POWER_RATIO_MAX``.  Where the
    premise holds the uniform completion must RAISE (the field is wrong by
    decades and the fallback target is that same field).  Where it does not
    hold -- a build whose ladder never reaches the bar -- the INVARIANT is
    asserted instead: no plane may RETURN a completed field whose reading is
    outside the band.  The fixed-path claim is unconditional on both arms.

    Measured 2026-09-14 (py3.14 / numpy 2.4.4 Windows and py3.12 / numpy 2.4.6
    WSL): z = 1762 / 1764 / 1768 / 1780 um read 93.5 / 3545 / 6097 / 8554,
    against 0.78-1.02 at every plane of the fold ladder below them.
    """
    E = _gauss()
    fired = []
    for z in _BLOWUP_LADDER:
        br, _d, _rec = _mb_bracket(E, z)
        if br is None or br <= _MB_POWER_RATIO_MAX:
            # premise not met at this rung -- the invariant still applies
            try:
                _, d, _ = _uniform(E, z)
            except RuntimeError:
                continue
            got = d.get('multibranch_power_ratio_bracketed')
            assert got is None or got <= _MB_POWER_RATIO_MAX, (
                f'z={z * 1e6:.0f} um returned a completed field whose '
                f'multibranch reading is {got:.4g}, outside the '
                f'{_MB_POWER_RATIO_MAX:g} bar')
            continue
        with pytest.raises(RuntimeError) as exc:
            _uniform(E, z)
        msg = str(exc.value)
        assert msg.startswith('apply_real_lens_traced_uniform'), msg[:80]
        assert f'{br:.4g}' in msg, (
            f'the refusal does not name the ratio it refused on ({br:.4g}): '
            f'{msg[:200]}')
        assert 'wave' in msg, 'the refusal does not name a member that works'
        fired.append((z, br))
    if not fired:
        pytest.skip('no rung of the ladder reproduces the blow-up on this '
                    'build; the invariant arm above ran instead')
    # the healthy plane below the window is untouched -- the refusal is
    # two-sided, not a blanket refusal of the fixture
    _, d, _ = _uniform(E, _HEALTHY_Z)
    assert d['fell_back'] is False and d['power_ratio_decision'] == 'ok'


def test_b7c_the_refusal_is_never_the_first_diagnostic():
    """Every refused field has already tripped the multibranch's own energy
    warning, because the refusal bar IS that module's warn bar.  So the
    refusal can never be the first the caller hears of the problem, and a
    caller who suppressed the warning still cannot get the field silently.

    Premise-gated on the same ladder; the CONSTANT relationship below is
    unconditional.
    """
    assert _MB_POWER_RATIO_MAX >= _ENERGY_BLOWUP_FACTOR
    E = _gauss()
    seen = False
    for z in _BLOWUP_LADDER:
        br, _d, rec = _mb_bracket(E, z)
        if br is None or br <= _MB_POWER_RATIO_MAX:
            continue
        seen = True
        assert [w for w in rec if 'reconstructed grid power'
                in str(w.message)], (
            f'z={z * 1e6:.0f} um: the multibranch did not warn on a field '
            f'the completion refuses ({br:.4g}x)')
    if not seen:
        pytest.skip('no rung of the ladder reproduces the blow-up on this '
                    'build')


def test_b7c_the_public_caustic_uniform_entry_point_refuses_too():
    """``apply_real_lens_traced(caustic='uniform')`` is the other route into
    this function and must not be a way past the refusal.  Premise-gated.
    """
    E = _gauss()
    for z in _BLOWUP_LADDER:
        br, _d, _rec = _mb_bracket(E, z)
        if br is None or br <= _MB_POWER_RATIO_MAX:
            continue
        with pytest.raises(RuntimeError) as exc:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                la.apply_real_lens_traced(
                    E, prescription=_presc(), wavelength=_WL, dx=_DX,
                    output_plane_distance=z, amplitude_model='ray_density',
                    caustic='uniform', n_workers=1, on_undersample='silent')
        assert 'apply_real_lens_traced_uniform' in str(exc.value)
        return
    pytest.skip('no rung of the ladder reproduces the blow-up on this build')


# ===========================================================================
# 3. the bars -- build-free, constants only
# ===========================================================================
def test_b7c_the_refusal_band_is_coupled_to_the_constants_it_was_derived_on():
    """The refusal bar IS ``_ENERGY_BLOWUP_FACTOR`` and the reporting floor
    IS the multibranch's own collapse factor, so the two modules classify the
    same field the same way and the bars cannot drift apart.

    The measured envelope behind the coupling (WP-B7c, 2026-09-14, five optics
    x 51 fold planes against a direct Rayleigh-Sommerfeld oracle): all 42
    planes the oracle accepts read a bracketed ratio of 0.816-1.246 (only two
    of them above 1.0 at all); the smallest broken one reads 5.848, on
    WP-B7b's own fast singlet one micron past its marginal focus.  The bar
    sits 1.60x above the largest accepted and 2.92x below the smallest broken
    reading, in a gap 4.69x wide whose geometric centre is 2.70.
    """
    assert _MB_POWER_RATIO_MAX == _ENERGY_BLOWUP_FACTOR
    assert _MB_POWER_RATIO_MIN == _ENERGY_COLLAPSE_FACTOR
    # the measured envelope, with its margins -- decision, not reading
    largest_accepted = 1.246          # fixture D, z = 1725 um, fidelity 0.883
    smallest_broken = 5.848           # fixture W1, z = 2060 um, fidelity 0.350
    assert largest_accepted < _MB_POWER_RATIO_MAX < smallest_broken, (
        f'{_MB_POWER_RATIO_MAX} has left the measured gap '
        f'[{largest_accepted}, {smallest_broken}]')
    assert _MB_POWER_RATIO_MAX / largest_accepted > 1.5
    assert smallest_broken / _MB_POWER_RATIO_MAX > 2.5


def test_b7c_the_zeta_bar_sits_in_the_measured_transition():
    """``_ZETA_EXTRAPOLATION_MAX`` re-derived two-sided on THREE optics.

    WP-B7c ladders (N-BAF10 biconvex / 1.064 um, N-BK7 plano-convex flat-first
    / 780 nm, N-SF11 biconvex / 1.55 um; 30 fold planes, completed-field power
    against the direct Rayleigh-Sommerfeld oracle): BELOW the bar the energy
    error is SIGNED and centred (-2.2 % .. +4.1 %, 12 of 22 rungs negative,
    mean +0.57 %); ABOVE it, it is a one-sided GAIN at every rung (+3.6 % ..
    +30.1 %, 8 of 8 positive, mean +7.88 %).  The last signed rung is 5.65 and
    the first systematic-gain rung 11.05, and 8.0 is 7.90 = sqrt(5.65 * 11.05)
    to two figures -- the bar is a calibrated boundary after all, but of
    +/-4 % against +4 %, not the 5 % / 10 % the pre-WP-B7c docstring claimed.
    """
    last_signed = 5.653               # fixture V, z = 1614.60 um, +1.98 %
    first_gain = 11.05                # fixture C, z = 3360.00 um, +4.40 %
    assert last_signed < _ZETA_EXTRAPOLATION_MAX < first_gain, (
        f'_ZETA_EXTRAPOLATION_MAX = {_ZETA_EXTRAPOLATION_MAX} has left the '
        f'measured transition [{last_signed}, {first_gain}]')
    assert _ZETA_EXTRAPOLATION_MAX / last_signed > 1.4
    assert first_gain / _ZETA_EXTRAPOLATION_MAX > 1.3


def test_b7c_the_dark_fill_depth_is_not_the_energy_lever():
    """WP-B7b section 7 proposed clipping ``_AIRY_TAIL_CELLS`` where the
    extrapolated ``zeta`` no longer describes the tail.  MEASURED (WP-B7c, the
    same oracle, seven planes of fixture V), the premise does not hold: the
    cumulative dark-side energy the completion writes, relative to the oracle's
    in the same annulus, is essentially FLAT in depth -- 2.17 at 3 Airy lengths
    against 2.05 at 20 at ``zeta_extrapolation`` = 14 -- so more than 94 % of
    the excess is written INSIDE 3 Airy lengths, where the tail is physical and
    must not be clipped.  The depth is not the lever; the extrapolated ``zeta``
    is (``kappa_eff / kappa`` from the oracle's own decay falls 1.07 -> 0.49 as
    ``zeta_extrapolation`` runs 0.42 -> 531).

    What is asserted here is the part of that reasoning the build owns: the
    fill still covers the whole representable tail.  Solved on the running
    build, not quoted.
    """
    from scipy.special import airy
    xs = np.linspace(0.0, float(_AIRY_TAIL_CELLS), 8001)
    ratio = np.abs(airy(xs)[0]) / float(airy(0.0)[0])
    below = np.nonzero(ratio < 1e-12)[0]
    assert below.size, 'Ai never falls below 1e-12 inside the fill'
    assert float(xs[below[0]]) < float(_AIRY_TAIL_CELLS)
    # ...and clipping at the 3 Airy lengths that carry the excess would cut
    # the tail while it is still 1e-3 of the ring -- the trade the measurement
    # refuses.
    at3 = float(np.abs(airy(3.0)[0]) / airy(0.0)[0])
    assert at3 > 1e-4, at3


# ===========================================================================
# 4. the member question, read through energy conservation
# ===========================================================================
@pytest.mark.parametrize('z_um,zeta_is_above_bar', [(1594.9, True),
                                                    (1683.4, False)])
def test_b7c_the_completion_gains_energy_where_the_hand_off_does_not(
        z_um, zeta_is_above_bar):
    """The member recommendation, on a truth that needs no model: a lossless
    element cannot deliver more power to a plane than it launched.

    On this aperture-truncated fixture the ray-to-wave hand-off
    (``caustic='wave'``) conserves the launched power to better than 1 % at
    every plane measured, while the completion runs +5.9 % at
    ``zeta_extrapolation`` = 14.0 and is inside +/-2 % at 1.03.  That is the
    direction the restated docstring records -- and the reverse of the
    pre-WP-B7c sentence, which generalised WP-B7b's own fixture (where the
    aperture is 2.0 w0 and the completion IS the closer member).

    The premise -- that this plane reads above / below the bar on this build --
    is asserted, so a build whose fold geometry moved cannot pass silently.
    """
    E = _gauss()
    z = z_um * 1e-6
    _, d, _ = _uniform(E, z)
    assert d['fell_back'] is False, d['reason']
    zx = float(d['zeta_extrapolation'])
    assert (zx > _ZETA_EXTRAPOLATION_MAX) is zeta_is_above_bar, (
        f'z={z_um} um now reads zeta_extrapolation={zx:.4g}, on the other '
        f'side of the {_ZETA_EXTRAPOLATION_MAX:g} bar than this row assumes')
    # the launched power inside the traced launch radius
    x = (np.arange(_N) - _N / 2.0) * _DX
    X, Y = np.meshgrid(x, x)
    inside = np.sqrt(X ** 2 + Y ** 2) <= 0.5 * 0.90e-3 * 0.98
    p_in = float(np.sum(np.abs(E[inside]) ** 2)) * _DX * _DX
    p_uni = float(np.sum(np.abs(np.asarray(_uniform(E, z)[0])) ** 2)) * _DX * _DX
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_wv = np.asarray(la.apply_real_lens_traced(
            E, prescription=_presc(), wavelength=_WL, dx=_DX,
            output_plane_distance=z, amplitude_model='ray_density',
            caustic='wave', n_workers=1, on_undersample='silent'))
    p_wv = float(np.sum(np.abs(E_wv) ** 2)) * _DX * _DX
    assert abs(p_wv / p_in - 1.0) < 0.02, (
        f'the hand-off left the energy band: {p_wv / p_in:.4f}')
    if zeta_is_above_bar:
        assert p_uni / p_in > p_wv / p_in + 0.02, (
            f'above the bar the completion should gain energy the hand-off '
            f'does not: completion {p_uni / p_in:.4f}, hand-off '
            f'{p_wv / p_in:.4f}')
    else:
        assert abs(p_uni / p_in - 1.0) < 0.05, (
            f'below the bar the completion should stay near the launched '
            f'power: {p_uni / p_in:.4f}')

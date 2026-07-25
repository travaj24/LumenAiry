"""S12 (sibling-pattern sweep report-only row, ``carrier.py`` ``rs_fine``
clamp): the exact-final-leg fine retrace must ANNOUNCE it when the F-C
pitch-preserving ``ray_subsample`` rescale is clamped, instead of silently
breaking its own contract.

Background (AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22 F-C,
AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1): ``_fine_trace_group_exit``
rescales the chain-level ``ray_subsample`` so the retrace on the fine grid
keeps the CHAIN's physical ray pitch ``ray_subsample * cur_dx``::

    rs_fine = max(1, round(ray_subsample * cur_dx / dx_fine))

When the memory/``n_fine_cap``-capped ``dx_fine`` is itself COARSER than that
physical pitch, the round goes to 0, the ``max(1, ...)`` clamp binds, and the
retrace's ray pitch becomes ``dx_fine`` -- coarser than the chain's (measured
5.25x on the production design-121 chain at N=28672 / ``n_fine_cap=16384``:
chain pitch 0.286 um vs ``dx_fine`` 1.5 um).  Nothing finer is representable
on that grid, so the clamp is FORCED rather than wrong -- but the F-C
contract does not hold there, the final leg's traced-OPL fit silently runs on
a coarser ray lattice than the rest of the chain, and the only hint was a
docstring paragraph plus the F-D warning (which names ``dx_fine`` vs the exit
sphere's Nyquist pitch -- a different quantity, and it does not fire in every
clamped case).

These pins are self-contained (no Zemax, no design-121 assets): the same
synthetic strong singlet the F-C/F-D tests use, with a deliberately tiny
``n_fine_cap`` to force ``dx_fine`` above the chain's ray pitch.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.propagators.carrier import (
    _fine_trace_group_exit,
    _paraxial_group_r_out,
)

_WL = 1.31e-6


def _singlet(R1, R2, d, glass, ap, name='s'):
    gb, ga = ['air', glass], [glass, 'air']
    surfaces = [
        {'radius': R1, 'glass_before': gb[0], 'glass_after': ga[0],
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': R2, 'glass_before': gb[1], 'glass_after': ga[1],
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': name, 'aperture_diameter': ap,
            'surfaces': surfaces, 'thicknesses': [d]}


def _gaussian_env(N, dx, w):
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


@pytest.fixture
def _setup():
    """Strong singlet + coarse-ish chain grid: the F-C/F-D fixture, chosen so
    ``n_fine_cap`` alone decides whether the clamp binds.

    Geometry: win = min(7*w, N*cur_dx) = 2.56 mm (grid-limited), so
    ``dx_fine = 2.56 mm / n_fine`` and the chain's physical ray pitch is
    ``4 * 10 um = 40 um``.  ``n_fine_cap=32`` -> dx_fine = 80 um (clamp
    binds, 2x too coarse); ``n_fine_cap=16384`` -> dx_fine = 0.156 um (no
    clamp)."""
    presc = _singlet(3.1e-3, -3.1e-3, 1.0e-3, 'N-BK7', 2.4e-3, 'strong')
    R_out = _paraxial_group_r_out(presc, np.inf, _WL)
    N, cur_dx, w = 256, 10e-6, 0.6e-3
    env = _gaussian_env(N, cur_dx, w)
    na_exit = w / abs(R_out)
    call_kw = dict(parallel_amp=False, on_undersample='silent',
                   on_noncollimated='silent')
    return presc, R_out, env, cur_dx, na_exit, call_kw


def _s12_msgs(rec):
    """The S12 warnings among a recorded warning list (the F-D Nyquist
    warning fires in the same regime, so match on the S12 wording)."""
    return [str(w.message) for w in rec
            if issubclass(w.category, RuntimeWarning)
            and 'physical ray pitch' in str(w.message)]


def test_rs_fine_clamp_binding_warns(_setup):
    """The headline pin: when ``round(ray_subsample*cur_dx/dx_fine) < 1`` the
    retrace must emit a RuntimeWarning saying the chain's physical ray pitch
    cannot be preserved."""
    presc, R_out, env, cur_dx, na_exit, call_kw = _setup
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _, dx_fine = _fine_trace_group_exit(
            env, np.inf, cur_dx, presc, _WL, 4, 1, call_kw, R_out, na_exit,
            window_factor=7.0, n_fine_cap=32)

    # The trigger condition really is met by this setup (guard the fixture).
    assert round(4 * cur_dx / dx_fine) < 1, (
        f'fixture no longer triggers the clamp: dx_fine={dx_fine * 1e6:.4f} '
        f'um vs chain pitch {4 * cur_dx * 1e6:.4f} um')
    msgs = _s12_msgs(rec)
    assert msgs, [str(w.message) for w in rec]
    assert 'CANNOT be preserved' in msgs[0]


def test_rs_fine_clamp_warning_names_both_pitches_and_the_remedy(_setup):
    """The message must be actionable: BOTH pitches (chain vs retrace) with
    their numeric values, the clamped ``ray_subsample=1``, and the remedy
    (raise ``n_fine_cap`` / shrink ``window_factor``)."""
    presc, R_out, env, cur_dx, na_exit, call_kw = _setup
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _, dx_fine = _fine_trace_group_exit(
            env, np.inf, cur_dx, presc, _WL, 4, 1, call_kw, R_out, na_exit,
            window_factor=7.0, n_fine_cap=32)
    msg = _s12_msgs(rec)[0]

    chain_pitch_um = 4 * cur_dx * 1e6
    dx_fine_um = dx_fine * 1e6
    assert f'{chain_pitch_um:.4f} um' in msg, (chain_pitch_um, msg)
    assert f'{dx_fine_um:.4f} um' in msg, (dx_fine_um, msg)
    assert 'ray_subsample=1' in msg
    assert 'n_fine_cap' in msg and 'window_factor' in msg
    # The ratio is the quantity the reader needs (5.25x on the production
    # chain); it must be reported, not left to be recomputed.
    assert f'{dx_fine / (4 * cur_dx):.2f}x' in msg, msg


def test_rs_fine_clamp_warning_silent_when_pitch_is_preservable(_setup):
    """The negative pin: with a generous ``n_fine_cap`` the rescale is
    honoured (rs_fine >= 1 without clamping) and NO S12 warning fires --
    i.e. this is not a blanket warning on the exact final leg."""
    presc, R_out, env, cur_dx, na_exit, call_kw = _setup
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _, dx_fine = _fine_trace_group_exit(
            env, np.inf, cur_dx, presc, _WL, 4, 1, call_kw, R_out, na_exit,
            window_factor=7.0, n_fine_cap=16384)
    assert round(4 * cur_dx / dx_fine) >= 1
    assert not _s12_msgs(rec), _s12_msgs(rec)


def test_rs_fine_clamp_actually_passes_ray_subsample_one(_setup, monkeypatch):
    """The warned-about behaviour is real: the element receives
    ``ray_subsample=1`` (the clamp), i.e. a retrace ray pitch of ``dx_fine``
    -- coarser than the chain's.  Pins the warning to the actual physics
    rather than to a message string."""
    presc, R_out, env, cur_dx, na_exit, call_kw = _setup
    captured = {}

    def _fake_traced(E_full, *, prescription, wavelength, dx, carrier,
                     ray_subsample, n_workers, **kw):
        captured['ray_subsample'] = ray_subsample
        captured['dx'] = dx
        return np.asarray(E_full)

    monkeypatch.setattr('lumenairy.elements.apply_real_lens_traced',
                        _fake_traced)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _, dx_fine = _fine_trace_group_exit(
            env, np.inf, cur_dx, presc, _WL, 4, 1, call_kw, R_out, na_exit,
            window_factor=7.0, n_fine_cap=32)

    assert captured['ray_subsample'] == 1
    assert captured['dx'] == pytest.approx(dx_fine)
    # coarser than the chain's pitch -- the contract gap the warning names
    assert dx_fine > 4 * cur_dx
    assert _s12_msgs(rec)


@pytest.mark.parametrize('ray_subsample', [1, 2, 4, 8])
def test_rs_fine_clamp_warning_threshold_matches_the_clamp(
        _setup, ray_subsample):
    """The warning must fire EXACTLY when the clamp binds, for every
    ``ray_subsample``: at fixed ``dx_fine = 80 um`` and ``cur_dx = 10 um``
    the rescale is ``ray_subsample/8``, which rounds below 1 for
    ``ray_subsample <= 4`` (0.125/0.25/0.5 -> 0) and to 1 for
    ``ray_subsample = 8`` -- so the low three warn and the last does not.
    (Python's round-half-to-even makes 0.5 -> 0, hence 4 warns.)"""
    presc, R_out, env, cur_dx, na_exit, call_kw = _setup
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _, dx_fine = _fine_trace_group_exit(
            env, np.inf, cur_dx, presc, _WL, ray_subsample, 1, call_kw,
            R_out, na_exit, window_factor=7.0, n_fine_cap=32)
    expect_warn = round(ray_subsample * cur_dx / dx_fine) < 1
    assert bool(_s12_msgs(rec)) == expect_warn, (
        ray_subsample, dx_fine, _s12_msgs(rec))


def test_rs_fine_clamp_warning_does_not_change_the_result(_setup):
    """Diagnostic-only: suppressing the warning must not change the returned
    field or pitch (the S12 change is a warning, not a behaviour change)."""
    presc, R_out, env, cur_dx, na_exit, call_kw = _setup
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        E_a, dx_a = _fine_trace_group_exit(
            env, np.inf, cur_dx, presc, _WL, 4, 1, call_kw, R_out, na_exit,
            window_factor=7.0, n_fine_cap=32)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        E_b, dx_b = _fine_trace_group_exit(
            env, np.inf, cur_dx, presc, _WL, 4, 1, call_kw, R_out, na_exit,
            window_factor=7.0, n_fine_cap=32)
    assert dx_a == dx_b
    assert np.array_equal(np.asarray(E_a), np.asarray(E_b))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

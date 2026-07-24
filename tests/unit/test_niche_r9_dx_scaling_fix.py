"""AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22 (F-A / F-C / F-D): regression pins
for the traced-carrier-chain exact-final-leg pre-readout re-trace.

Addendum to the R9 final-leg tests (``test_niche_r9_highna_final_leg.py``).
That audit found, on the production design-121 chain at chain grid N >
16384:

* F-A (P0) -- ``_fourier_upsample_crop`` silently returned the wrong-sized,
  wrong-pitch array whenever asked to DOWNSAMPLE (``n_fine <= n_crop``),
  because a short-circuit ``if n_fine <= n_crop: return ec`` returned the raw
  ``n_crop``-sized crop while every downstream consumer (dx_fine, the exact
  readout's power normalisation, ...) assumed the array had ``n_fine``
  pixels.  Triggered exactly when ``n_crop`` (the window in ``cur_dx``
  pixels) exceeds ``_fine_trace_group_exit``'s ``n_fine_cap`` (16384) --
  observed 130.8% of launch power inside the readout window (should be <=
  ~0.96 after stop losses; > 1.0 is physically impossible).
* F-C (P2) -- the fine retrace reused the chain-level ``ray_subsample``
  literally in the FINE grid's (much smaller) pixel units, silently
  densifying the Newton/Chebyshev ray-fit grid by the ``cur_dx / dx_fine``
  factor; observed a (28, 20151, 20151) float64 = 84.7 GiB design matrix at
  ``ray_subsample=1`` on the production chain.
* F-D (P2) -- ``n_fine_cap`` can force ``dx_fine`` coarser than the exit
  sphere's Nyquist pitch even in cases that "work" (no crash, no energy
  bug), silently discarding outer-NA content with no diagnostic.

These tests are self-contained (no Zemax, no external design-121 assets) and
reproduce each failure mode at a small, fast scale by forcing the same
trigger conditions the production chain hits at N=28672: F-A/F-D via an
artificially small ``n_fine_cap``, F-C via a fine grid genuinely finer than
the chain's co-moving pitch (the normal condition whenever the exact final
leg engages at all).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.propagators.carrier import (
    _fine_trace_group_exit,
    _fourier_upsample_crop,
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


# ===========================================================================
# F-A: _fourier_upsample_crop must DOWNSAMPLE (not silently mis-crop) when
# n_fine < n_crop, with the same value-preserving pitch/energy contract as
# the (pre-existing, correct) upsample direction.
# ===========================================================================
@pytest.mark.parametrize('n_crop,n_fine', [
    (256, 64), (256, 128), (256, 256), (256, 512), (128, 37 * 2)])
def test_fourier_upsample_crop_shape_invariant_both_directions(n_crop, n_fine):
    """``out.shape[-1] == n_fine`` must hold whether upsampling, downsampling,
    or the identity case -- the exact invariant the audit's fix requires."""
    N = 512
    env = _gaussian_env(N, 1.0e-6, 40e-6)
    out = _fourier_upsample_crop(env, n_crop, n_fine)
    assert out.shape == (n_fine, n_fine)


def test_fourier_upsample_crop_downsample_matches_direct_construction():
    """A band-limited (smooth Gaussian) envelope, downsampled via spectral
    truncation, must closely match a DIRECT construction of the same
    continuous function on the target grid -- proving the truncation
    reconstructs the correct (not merely correctly-SHAPED) values."""
    N, dx, w = 512, 1.0e-6, 40e-6
    env = _gaussian_env(N, dx, w)
    n_crop, n_fine = 256, 64
    win = n_crop * dx
    dx_fine = win / n_fine

    out = _fourier_upsample_crop(env, n_crop, n_fine)
    direct = _gaussian_env(n_fine, dx_fine, w)
    rel_err = float(np.linalg.norm(out - direct) / np.linalg.norm(direct))
    assert rel_err < 1e-3, rel_err


def test_fourier_upsample_crop_downsample_conserves_energy():
    """Parseval: total power (sum|E|^2 * dx^2) must be preserved across the
    resample for a band-limited signal -- NOT inflated by (n_crop/n_fine)^2,
    which is exactly the bug the pre-fix short-circuit produced (a caller
    that trusts the returned shape for its pitch sees that inflation)."""
    N, dx, w = 512, 1.0e-6, 40e-6
    env = _gaussian_env(N, dx, w)
    n_crop, n_fine = 256, 64
    win = n_crop * dx
    dx_fine = win / n_fine

    c0 = N // 2 - n_crop // 2
    ec = env[c0:c0 + n_crop, c0:c0 + n_crop]
    p_crop = float(np.sum(np.abs(ec) ** 2)) * dx * dx

    out = _fourier_upsample_crop(env, n_crop, n_fine)
    p_fine = float(np.sum(np.abs(out) ** 2)) * dx_fine * dx_fine

    assert abs(p_fine / p_crop - 1.0) < 1e-3, (p_crop, p_fine)


def test_fourier_upsample_crop_downsample_round_trips():
    """crop -> downsample -> upsample back to n_crop must recover the
    original crop (lossy only in the discarded high frequencies, which are
    ~0 for a smooth envelope with the carrier divided out)."""
    N, dx, w = 512, 1.0e-6, 40e-6
    env = _gaussian_env(N, dx, w)
    n_crop, n_fine = 256, 64

    down = _fourier_upsample_crop(env, n_crop, n_fine)
    back = _fourier_upsample_crop(down, n_fine, n_crop)

    c0 = N // 2 - n_crop // 2
    ec = env[c0:c0 + n_crop, c0:c0 + n_crop]
    rel_err = float(np.linalg.norm(back - ec) / np.linalg.norm(ec))
    assert rel_err < 1e-3, rel_err


# ===========================================================================
# End-to-end through _fine_trace_group_exit: force n_crop > n_fine_cap at a
# SMALL, fast scale (the real chain hits this only at N > 16384) and confirm
# energy is NOT inflated, plus the F-C/F-D side effects.
# ===========================================================================
@pytest.fixture
def _strong_singlet_setup():
    presc = _singlet(3.1e-3, -3.1e-3, 1.0e-3, 'N-BK7', 2.4e-3, 'strong')
    R_out = _paraxial_group_r_out(presc, np.inf, _WL)
    return presc, R_out


def test_fine_trace_group_exit_n_fine_cap_trigger_does_not_inflate_energy(
        _strong_singlet_setup):
    """Forcing n_fine_cap small enough that n_crop > n_fine_cap must NOT
    inflate the exit power relative to a generous (uncapped) n_fine_cap run
    of the identical setup.  Pre-fix, the capped run's env_f silently kept
    n_crop pixels while dx_fine was computed for n_fine pixels -- a pitch
    mismatch that (depending on the downstream consumer) manifests as an
    energy blow-up on the order of (n_crop/n_fine)^2."""
    presc, R_out = _strong_singlet_setup
    N, cur_dx, w = 256, 10e-6, 0.6e-3
    env = _gaussian_env(N, cur_dx, w)
    na_exit = w / abs(R_out)
    assert na_exit > 0.15, f'setup must be high-NA (NA={na_exit:.3f})'
    call_kw = dict(parallel_amp=False, on_undersample='silent',
                   on_noncollimated='silent')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        # Generous cap: n_fine is NOT clamped (natural, uncapped resolution).
        E_ref, dx_ref = _fine_trace_group_exit(
            env, np.inf, cur_dx, presc, _WL, 4, 1, call_kw, R_out, na_exit,
            window_factor=7.0, n_fine_cap=16384)
        # Forced-tiny cap: must trigger n_crop > n_fine_cap (F-A's condition).
        E_cap, dx_cap = _fine_trace_group_exit(
            env, np.inf, cur_dx, presc, _WL, 4, 1, call_kw, R_out, na_exit,
            window_factor=7.0, n_fine_cap=32)

    assert np.isfinite(np.asarray(E_ref)).all()
    assert np.isfinite(np.asarray(E_cap)).all()

    p_ref = float(np.sum(np.abs(np.asarray(E_ref)) ** 2)) * dx_ref ** 2
    p_cap = float(np.sum(np.abs(np.asarray(E_cap)) ** 2)) * dx_cap ** 2
    # Same physical process at two resample resolutions: power should agree
    # to within a modest tolerance (real resample + ray-fit differences),
    # NOT diverge by the (n_crop/n_fine)^2 factor (>> 10x here) the bug gave.
    ratio = p_cap / p_ref
    assert 0.5 < ratio < 1.5, (
        f'capped/uncapped power ratio {ratio:.3f} suggests a pitch-mismatch '
        f'energy bug (p_ref={p_ref:.4e}, p_cap={p_cap:.4e})')


def test_fine_trace_group_exit_warns_on_nyquist_undersampling(
        _strong_singlet_setup):
    """F-D: an n_fine_cap small enough to force dx_fine above the exit
    sphere's Nyquist pitch lambda/(2*NA) must emit a RuntimeWarning."""
    presc, R_out = _strong_singlet_setup
    N, cur_dx, w = 256, 10e-6, 0.6e-3
    env = _gaussian_env(N, cur_dx, w)
    na_exit = w / abs(R_out)
    call_kw = dict(parallel_amp=False, on_undersample='silent',
                   on_noncollimated='silent')

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _fine_trace_group_exit(
            env, np.inf, cur_dx, presc, _WL, 4, 1, call_kw, R_out, na_exit,
            window_factor=7.0, n_fine_cap=32)
    msgs = [str(w.message) for w in rec
            if issubclass(w.category, RuntimeWarning)]
    assert any('Nyquist' in m or 'n_fine_cap' in m for m in msgs), msgs


# ===========================================================================
# F-C interaction check (static review, 2026-07-22): the pitch-preserving
# rescale reduces the fine leg's ray-fit density, which -- for a chain
# marginally sampled to begin with -- can trip apply_real_lens_traced's own
# min_coarse_samples_per_aperture aliasing floor (default 32,
# on_undersample='error' by default).  This is NOT a demonstrated regression
# in any realistic (beam-filling-aperture) chain -- see the docstring note
# on _fine_trace_group_exit for the margin analysis -- but the interaction
# must fail LOUDLY (a clear, actionable ValueError) rather than silently
# succeed with an aliased fit or crash somewhere unrelated.  This closes the
# coverage gap noted in review: no other test drives the REAL
# apply_real_lens_traced with the DEFAULT on_undersample.
# ===========================================================================
def test_fine_trace_group_exit_marginal_chain_trips_aliasing_guard_loudly():
    """A deliberately marginally-sampled chain (small aperture, beam radius
    sampled by only ~10 px) must raise apply_real_lens_traced's own
    min_coarse_samples_per_aperture ValueError -- confirming the F-C
    rescale's only realistic failure mode is a loud, actionable error, not a
    silent wrong answer (which is what the PRE-fix code -- effectively an
    over-dense ray fit reusing chain ray_subsample in fine-pixel units --
    would have masked)."""
    presc = _singlet(3.1e-3, -3.1e-3, 1.0e-3, 'N-BK7', 150e-6, 'strong')
    R_out = _paraxial_group_r_out(presc, np.inf, _WL)
    N, cur_dx, w = 256, 10e-6, 100e-6
    env = _gaussian_env(N, cur_dx, w)
    na_exit = w / abs(R_out)

    # call_kw={} -> apply_real_lens_traced's DEFAULT on_undersample='error'
    # (the repro script this audit is based on passes no traced_kwargs at
    # all, so this is the real-world default, not a contrived override).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with pytest.raises(ValueError, match='coarse samples'):
            _fine_trace_group_exit(
                env, np.inf, cur_dx, presc, _WL, 4, 1, {}, R_out, na_exit,
                window_factor=7.0, n_fine_cap=16384)


# ===========================================================================
# F-C: the fine retrace must preserve the CHAIN's physical ray pitch, not
# reinterpret ray_subsample in the (finer) dx_fine grid's pixel units.
# ===========================================================================
def test_fine_trace_group_exit_rescales_ray_subsample_to_chain_pitch(
        monkeypatch, _strong_singlet_setup):
    """Patch apply_real_lens_traced to capture the ray_subsample it actually
    receives; it must equal round(chain_ray_subsample * cur_dx / dx_fine),
    NOT the raw chain-level value re-used verbatim on the finer grid."""
    presc, R_out = _strong_singlet_setup
    N, cur_dx, w = 256, 10e-6, 0.6e-3
    env = _gaussian_env(N, cur_dx, w)
    na_exit = w / abs(R_out)
    chain_ray_subsample = 4
    call_kw = dict(parallel_amp=False, on_undersample='silent',
                   on_noncollimated='silent')

    captured = {}

    def _fake_traced(E_full, *, prescription, wavelength, dx, carrier,
                      ray_subsample, n_workers, **kw):
        captured['ray_subsample'] = ray_subsample
        captured['dx'] = dx
        return np.asarray(E_full)

    monkeypatch.setattr('lumenairy.elements.apply_real_lens_traced',
                         _fake_traced)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        _, dx_fine = _fine_trace_group_exit(
            env, np.inf, cur_dx, presc, _WL, chain_ray_subsample, 1, call_kw,
            R_out, na_exit, window_factor=7.0, n_fine_cap=16384)

    assert 'ray_subsample' in captured, 'patched apply_real_lens_traced was not called'
    expected_rs = max(1, round(chain_ray_subsample * cur_dx / dx_fine))
    assert captured['ray_subsample'] == expected_rs, (
        captured['ray_subsample'], expected_rs)
    # The bug this closes: reusing the chain-level value verbatim would give
    # ray_subsample == chain_ray_subsample (4) here, which must NOT be what
    # was captured once dx_fine is meaningfully finer than cur_dx.
    assert dx_fine < cur_dx, 'test setup must produce a finer retrace grid'
    assert captured['ray_subsample'] != chain_ray_subsample or expected_rs == chain_ray_subsample


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

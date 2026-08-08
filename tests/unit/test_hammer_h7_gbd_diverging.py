"""Hammer-audit H7 (2026-07-19): GBD diverging-input support.

``apply_real_lens_gbd``'s position-only (axial) beamlet decomposition
carried a strongly-diverging input's wavefront curvature in the beamlet
AMPLITUDE only, not the LAUNCH DIRECTION, so the axial base rays focused
at the COLLIMATED plane ``f`` and the beamlet frame shed energy: power
conservation collapsed to ~0.2-0.7 on the dual-oracle singlet and the
true finite-conjugate image at ``z_img`` smeared (the documented
paraxial-frame limitation -- power ~1e-4 + NaN warnings at the production
121's NA~0.23 diverging beam).

Fix (H7): ``direction_sampling='auto'`` (new default) launches each
beamlet along the input's local wavevector (Husimi / carrier normals)
when the input carries transverse angular content -- the GBD analogue of
the H6 traced carrier fix (decompose against the carrier, launch beamlets
along its normals).  A flat-wavefront (collimated) input measures ~0
angular spread and takes the byte-identical axial path.

Oracle: ABCD Gaussian q-trace through the dual-oracle singlet
(R = +/-51.68 mm, t = 5 mm, n = 1.5168, lambda = 1.31 um) -- fully
independent of the GBD implementation.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.lenses_gbd import (
    _GBD_AUTO_HUSIMI_THRESH,
    _input_angular_spread,
)

_WL = 1.31e-6
_N_GLASS = 1.5168
_R1, _R2, _TC = 51.68e-3, -51.68e-3, 5e-3
_F_PLANE = 49.163e-3            # collimated focus (BFL), dual-oracle number

# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_H7_FIX_GLASS': lambda wl: _N_GLASS}

# GBD test config: N/dx resolve the diverging fringe pitch (lambda*R/r);
# bpa keeps the beamlet frame overlapping while bounding the count/runtime.
_N, _DX, _W_L, _BPA = 768, 8e-6, 1.0e-3, 128


def _singlet():
    return {
        'wavelength': _WL, 'aperture_diameter': 20e-3,
        'surfaces': [
            {'radius': _R1, 'thickness': _TC, 'glass_before': 'air',
             'glass_after': '_H7_FIX_GLASS', 'semi_diameter': 10e-3},
            {'radius': _R2, 'thickness': 0.0, 'glass_before': '_H7_FIX_GLASS',
             'glass_after': 'air', 'semi_diameter': 10e-3},
        ],
        'thicknesses': [_TC], 'stop_index': 0,
    }


def _lens_abcd():
    """ABCD of the thick singlet, entry vertex -> exit vertex."""
    def refr(R, n1, n2):
        return np.array([[1.0, 0.0], [-(n2 - n1) / (R * n2), n1 / n2]])

    def trans(d):
        return np.array([[1.0, d], [0.0, 1.0]])

    return refr(_R2, _N_GLASS, 1.0) @ trans(_TC) @ refr(_R1, 1.0, _N_GLASS)


def _q_trace_image(R_in, w_L):
    """Gaussian q at the entry vertex -> image distance past the exit vertex
    (waist location) + waist size.  Independent oracle."""
    if np.isinf(R_in):
        q_inv = -1j * _WL / (np.pi * w_L ** 2)
    else:
        q_inv = 1.0 / R_in - 1j * _WL / (np.pi * w_L ** 2)
    q0 = 1.0 / q_inv
    M = _lens_abcd()
    q1 = (M[0, 0] * q0 + M[0, 1]) / (M[1, 0] * q0 + M[1, 1])
    z_img = -q1.real
    q_w = q1 + z_img
    w_img = np.sqrt(-_WL / (np.pi * (1.0 / q_w).imag))
    return float(z_img), float(w_img)


def _diverging_input(N, dx, w_L, R_in):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X ** 2 + Y ** 2
    env = np.exp(-r_sq / w_L ** 2)
    if np.isinf(R_in):
        ph = np.ones_like(env)
    else:
        k = 2 * np.pi / _WL
        ph = np.exp(1j * k * r_sq / (2.0 * R_in))
    return (env * ph).astype(np.complex128)


def _ee_at(E_exit, dx, z, N, r_ee=150e-6):
    E = la.angular_spectrum_propagate(E_exit, z, _WL, dx)
    I = np.abs(E) ** 2
    if not np.all(np.isfinite(I)):
        return float('nan')
    x = (np.arange(N) - N / 2) * dx
    j, i = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(x - x[i], x - x[j])
    r = np.sqrt(X ** 2 + Y ** 2)
    return float(I[r <= r_ee].sum() / I.sum())


def _best_focus(E_exit, dx, z_img, N, npts=15):
    zs = np.linspace(0.85 * z_img, 1.15 * z_img, npts)
    ees = [_ee_at(E_exit, dx, float(z), N) for z in zs]
    k = int(np.nanargmax(ees))
    return float(zs[k]), float(ees[k])


# ---------------------------------------------------------------------------
# The regression: the DEFAULT ('auto') path must conserve power and focus at
# the true finite-conjugate image for a strongly-diverging input.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('R_in', [300e-3, 150e-3, 100e-3])
def test_h7_diverging_default_conserves_power_and_focuses(R_in):
    """A diverging Gaussian through the singlet, via the DEFAULT decomposition
    (direction_sampling='auto'), must (a) conserve power > 0.99, (b) stay
    finite (no NaN), and (c) focus at the ABCD q-trace image plane to within
    5% -- and there, not at the collimated focal plane f.

    Pre-H7 (axial default): power collapsed to ~0.2-0.7 and the focus pinned
    at f (the paraxial-frame energy-collapse limitation)."""
    z_img, w_img = _q_trace_image(R_in, _W_L)
    E0 = _diverging_input(_N, _DX, _W_L, R_in)
    p_in = float(np.sum(np.abs(E0) ** 2))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        E_exit = np.asarray(la.apply_real_lens_gbd(
            E0, prescription=_singlet(), wavelength=_WL, dx=_DX,
            beamlets_per_aperture=_BPA))
        nan_warn = any(('nan' in str(w.message).lower()
                        or 'finite' in str(w.message).lower())
                       for w in caught)

    # (a) power conservation
    p_out = float(np.sum(np.abs(E_exit) ** 2))
    pc = p_out / p_in
    assert pc > 0.99, (
        f"R_in={R_in*1e3:.0f}mm: exit power ratio {pc:.4f} <= 0.99 -- the "
        f"axial (pre-H7) frame shed the diverging beam's energy")

    # (b) finiteness -- the NaN warnings the production 121 saw must be gone
    assert np.all(np.isfinite(E_exit)), "non-finite entries in the exit field"
    assert not nan_warn, "a NaN / non-finite warning fired on the auto path"

    # (c) focus at the ABCD image, within 5%, beating the collimated plane f
    z_bf, ee_bf = _best_focus(E_exit, _DX, z_img, _N)
    ferr = abs(z_bf - z_img) / z_img
    assert ferr < 0.05, (
        f"R_in={R_in*1e3:.0f}mm: best focus {z_bf*1e3:.2f}mm vs ABCD image "
        f"{z_img*1e3:.2f}mm is {ferr*100:.1f}% off (> 5%)")
    ee_img = _ee_at(E_exit, _DX, z_img, _N)
    ee_f = _ee_at(E_exit, _DX, _F_PLANE, _N)
    assert ee_img > ee_f, (
        f"focus must sit at z_img={z_img*1e3:.2f}mm, not the collimated plane "
        f"f={_F_PLANE*1e3:.2f}mm (EE_img={ee_img:.3f} vs EE_f={ee_f:.3f})")


def test_h7_axial_direction_sampling_collapses_power():
    """Fail-before signature: forcing the pre-H7 axial launch
    (direction_sampling=False) on the same diverging input sheds most of the
    beam's energy -- power ratio << 0.99 -- which the 'auto' default (the test
    above) avoids.  Locks the regression so a revert cannot pass silently."""
    R_in = 150e-3
    E0 = _diverging_input(_N, _DX, _W_L, R_in)
    p_in = float(np.sum(np.abs(E0) ** 2))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_exit = np.asarray(la.apply_real_lens_gbd(
            E0, prescription=_singlet(), wavelength=_WL, dx=_DX,
            beamlets_per_aperture=_BPA, direction_sampling=False))
    pc = float(np.sum(np.abs(E_exit) ** 2)) / p_in
    assert pc < 0.9, (
        f"axial launch should collapse the diverging beam's power (got "
        f"{pc:.4f}); if this no longer collapses, the H7 regression guard is "
        f"moot -- revisit the default")


def test_h7_collimated_default_is_byte_identical_to_axial():
    """The collimated case must be untouched: 'auto' measures ~0 angular
    spread for a flat-wavefront (real) input and takes the axial path, so the
    DEFAULT output is byte-for-byte identical to direction_sampling=False --
    the validated collimated baseline is preserved exactly."""
    xs = (np.arange(256) - 128) * _DX
    X, Y = np.meshgrid(xs, xs)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (0.5e-3) ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_auto = np.asarray(la.apply_real_lens_gbd(
            E0.copy(), prescription=_singlet(), wavelength=_WL, dx=_DX,
            sample_step=4))
        E_axial = np.asarray(la.apply_real_lens_gbd(
            E0.copy(), prescription=_singlet(), wavelength=_WL, dx=_DX,
            sample_step=4, direction_sampling=False))
    assert np.array_equal(E_auto, E_axial), (
        "auto default diverged from the axial path on a collimated input "
        f"(max |diff| = {np.max(np.abs(E_auto - E_axial)):.2e})")


def test_h7_angular_spread_detector_separates_collimated_from_diverging():
    """Unit check of the 'auto' trigger: a flat-wavefront (real / globally-
    phased) field measures exactly 0 angular spread (< threshold -> axial),
    while a diverging Gaussian measures ~its NA (> threshold -> Husimi)."""
    xs = (np.arange(_N) - _N / 2) * _DX
    X, Y = np.meshgrid(xs, xs)
    real_g = np.exp(-(X ** 2 + Y ** 2) / _W_L ** 2).astype(np.complex128)
    phased = (real_g * np.exp(1j * 0.9))            # global phase -> still flat
    assert _input_angular_spread(real_g, _DX, _DX, _WL) < _GBD_AUTO_HUSIMI_THRESH
    assert _input_angular_spread(phased, _DX, _DX, _WL) < _GBD_AUTO_HUSIMI_THRESH
    for R_in in (300e-3, 150e-3, 100e-3):
        div = _diverging_input(_N, _DX, _W_L, R_in)
        s = _input_angular_spread(div, _DX, _DX, _WL)
        assert s > _GBD_AUTO_HUSIMI_THRESH, (
            f"R_in={R_in*1e3:.0f}mm angular spread {s:.2e} below threshold "
            f"{_GBD_AUTO_HUSIMI_THRESH} -- auto would miss the divergence")


# ===========================================================================
# G1 (2026-07-19): generality of the H7 fix on the NO-121 design matrix.
# Task-2 cases: M1 cemented doublet (diverging input R_in=+150mm) and M5
# negative biconcave (converging input).  Independent oracle: general
# multi-surface ABCD Gaussian q-trace.
# ===========================================================================
MODULE_GLASSES['_H7_FIX_162'] = lambda wl: 1.62   # declared above


def _lens_abcd_multi(surfaces, thicknesses, n_of):
    """Entry-vertex -> exit-vertex ABCD for an N-surface lens (unreduced
    convention, physical thicknesses) -- generalises _lens_abcd."""
    def refr(R, n1, n2):
        if not np.isfinite(R) or R == 0:
            return np.array([[1.0, 0.0], [0.0, n1 / n2]])
        return np.array([[1.0, 0.0], [-(n2 - n1) / (R * n2), n1 / n2]])

    def trans(d):
        return np.array([[1.0, d], [0.0, 1.0]])

    M = np.eye(2)
    for i, s in enumerate(surfaces):
        n1 = 1.0 if s['glass_before'] == 'air' else n_of(s['glass_before'])
        n2 = 1.0 if s['glass_after'] == 'air' else n_of(s['glass_after'])
        M = refr(s['radius'], n1, n2) @ M
        if i < len(surfaces) - 1:
            M = trans(thicknesses[i]) @ M
    return M


def _q_image_multi(surfaces, thicknesses, n_of, R_in, w_L):
    q_inv = (-1j * _WL / (np.pi * w_L ** 2) if (R_in is None or np.isinf(R_in))
             else 1.0 / R_in - 1j * _WL / (np.pi * w_L ** 2))
    q0 = 1.0 / q_inv
    M = _lens_abcd_multi(surfaces, thicknesses, n_of)
    q1 = (M[0, 0] * q0 + M[0, 1]) / (M[1, 0] * q0 + M[1, 1])
    return -q1.real                     # image distance past the exit vertex


def _m1_doublet():
    return {'wavelength': _WL, 'aperture_diameter': 40e-3,
            'surfaces': [
                {'radius': 61.5e-3, 'thickness': 6e-3, 'glass_before': 'air',
                 'glass_after': '_H7_FIX_GLASS', 'semi_diameter': 20e-3},
                {'radius': -44.5e-3, 'thickness': 2.5e-3,
                 'glass_before': '_H7_FIX_GLASS', 'glass_after': '_H7_FIX_162',
                 'semi_diameter': 20e-3},
                {'radius': -129e-3, 'thickness': 0.0,
                 'glass_before': '_H7_FIX_162', 'glass_after': 'air',
                 'semi_diameter': 20e-3}],
            'thicknesses': [6e-3, 2.5e-3], 'stop_index': 0}


def _m5_biconcave():
    return {'wavelength': _WL, 'aperture_diameter': 24e-3,
            'surfaces': [
                {'radius': -51.68e-3, 'thickness': 3e-3, 'glass_before': 'air',
                 'glass_after': '_H7_FIX_GLASS', 'semi_diameter': 12e-3},
                {'radius': 51.68e-3, 'thickness': 0.0,
                 'glass_before': '_H7_FIX_GLASS', 'glass_after': 'air',
                 'semi_diameter': 12e-3}],
            'thicknesses': [3e-3], 'stop_index': 0}


_N_OF = {'_H7_FIX_GLASS': _N_GLASS, '_H7_FIX_162': 1.62}


def test_h7_doublet_diverging_input_conserves_and_focuses():
    """G1 task-2: the H7 default ('auto') on a cemented DOUBLET with a
    diverging input (R_in=+150mm) conserves power > 0.99 and focuses at the
    ABCD image -- the multi-element generality of the H7 launch fix."""
    p = _m1_doublet()
    z_img = _q_image_multi(p['surfaces'], p['thicknesses'], _N_OF.get, 150e-3,
                           _W_L)
    E0 = _diverging_input(_N, _DX, _W_L, 150e-3)
    p_in = float(np.sum(np.abs(E0) ** 2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        E_exit = np.asarray(la.apply_real_lens_gbd(
            E0, prescription=p, wavelength=_WL, dx=_DX,
            beamlets_per_aperture=_BPA))
        nan_warn = any('nan' in str(w.message).lower()
                       or 'finite' in str(w.message).lower() for w in caught)
    pc = float(np.sum(np.abs(E_exit) ** 2)) / p_in
    assert pc > 0.99, f"doublet diverging power {pc:.4f} <= 0.99"
    assert np.all(np.isfinite(E_exit)) and not nan_warn
    z_bf, _ = _best_focus(E_exit, _DX, z_img, _N)
    assert abs(z_bf - z_img) / z_img < 0.05, (z_bf * 1e3, z_img * 1e3)


def test_h7_negative_lens_converging_input_conserves():
    """G1 task-2: a negative biconcave (M5) with the task's CONVERGING input
    (R_in=-60mm) via the H7 default conserves power > 0.99 -- converging-input
    support extends to a diverging element when the output vergence is gentle.
    (See the strong-reconvergence test below for the frame-completeness caveat
    when the negative lens must reconverge the beam to a NEAR real focus.)

    A negative element needs a FINER beamlet frame than a positive one to tile
    its diverging exit (bpa=256 here vs the 128 the positive-lens cases use)."""
    p = _m5_biconcave()
    E0 = _diverging_input(_N, _DX, _W_L, -60e-3)     # converging (R<0)
    p_in = float(np.sum(np.abs(E0) ** 2))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_exit = np.asarray(la.apply_real_lens_gbd(
            E0, prescription=p, wavelength=_WL, dx=_DX,
            beamlets_per_aperture=256))
    pc = float(np.sum(np.abs(E_exit) ** 2)) / p_in
    assert np.all(np.isfinite(E_exit))
    assert pc > 0.99, f"negative-lens converging power {pc:.4f} <= 0.99"


def test_h7_converging_strong_reconvergence_frame_limit():
    """G1 finding: a converging input reconverged by a NEGATIVE lens to a NEAR
    real focus (M5, R_in=-35mm -> real image ~108mm) is a documented H7
    envelope EDGE: the beamlet frame is incomplete (raw power ~0.94, NOT
    density-recoverable -- it saturates at the max frame), yet the LAUNCH is
    correct (the beam focuses at the ABCD image) and normalize_output='power'
    restores absolute power exactly.  Regression guard for the frame-
    completeness caveat that G2 owns."""
    p = _m5_biconcave()
    z_img = _q_image_multi(p['surfaces'], p['thicknesses'], _N_OF.get, -35e-3,
                           _W_L)
    assert z_img > 0                                  # real downstream focus
    E0 = _diverging_input(_N, _DX, _W_L, -35e-3)
    p_in = float(np.sum(np.abs(E0) ** 2))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_raw = np.asarray(la.apply_real_lens_gbd(
            E0, prescription=p, wavelength=_WL, dx=_DX,
            beamlets_per_aperture=256))
        E_norm = np.asarray(la.apply_real_lens_gbd(
            E0, prescription=p, wavelength=_WL, dx=_DX,
            beamlets_per_aperture=256, normalize_output='power'))
    pc = float(np.sum(np.abs(E_raw) ** 2)) / p_in
    # frame-completeness loss (documented edge): lossy but bounded, NOT collapse.
    # Saturates at the max frame density (~0.94 at bpa=256; ~0.88 at bpa=128).
    assert 0.90 < pc < 0.97, f"power {pc:.4f} outside documented lossy band"
    # normalize_output restores absolute power exactly
    pcn = float(np.sum(np.abs(E_norm) ** 2)) / p_in
    assert abs(pcn - 1.0) < 1e-3, pcn
    # LAUNCH is correct: the beam still focuses at the real ABCD image
    z_bf, ee_bf = _best_focus(E_raw, _DX, z_img, _N)
    assert abs(z_bf - z_img) / z_img < 0.06, (z_bf * 1e3, z_img * 1e3)
    assert ee_bf > 0.9, ee_bf


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

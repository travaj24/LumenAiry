"""Hammer-audit H6 (2026-07-19): traced carrier entrance eikonal.

``apply_real_lens_traced`` with a carrier (``'auto'``, scalar conjugate,
or ndarray) omitted the carrier's ENTRANCE-PLANE eikonal ``k0*W(x_in)``
from the per-ray OPL, while the ``preserve_input_phase`` reference leg
(``exp(i*k0*W)`` through ``apply_real_lens``) included it.  The
mismatched ``-k0*W`` imprinted on the field cancelled the input
divergence the wave model correctly carried: EVERY diverging-input trace
collapsed to the COLLIMATED focal plane ``f`` and the true image at
``z_img`` smeared by ``NA_exit*(z_img - f)`` (production exp22: energy
over +/-1.8 mm, EE(100um) = 0.9% -- reproduced to the digit by the
root-cause reproducer; fixed: EE(100um) = 0.999).  The cruel twist: the
``on_noncollimated`` guard measures the POST-carrier residual (~0), so
the broken path was silent while ``carrier=None`` warned users INTO it.

Oracle: ABCD Gaussian q-trace through the dual-oracle singlet
(R = +/-51.68 mm, t = 5 mm, n = 1.5168, lambda = 1.31 um) -- fully
independent of the traced implementation.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la

_WL = 1.31e-6
_N_GLASS = 1.5168
_R1, _R2, _TC = 51.68e-3, -51.68e-3, 5e-3

# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_H6_FIX_GLASS': lambda wl: _N_GLASS}


def _singlet():
    return {
        'wavelength': _WL,
        'aperture_diameter': 20e-3,
        'surfaces': [
            {'radius': _R1, 'thickness': _TC,
             'glass_before': 'air', 'glass_after': '_H6_FIX_GLASS',
             'semi_diameter': 10e-3},
            {'radius': _R2, 'thickness': 0.0,
             'glass_before': '_H6_FIX_GLASS', 'glass_after': 'air',
             'semi_diameter': 10e-3},
        ],
        'thicknesses': [_TC],
        'stop_index': 0,
    }


def _lens_abcd():
    """ABCD of the thick singlet, entry vertex -> exit vertex."""
    def refr(R, n1, n2):
        return np.array([[1.0, 0.0], [-(n2 - n1) / (R * n2), n1 / n2]])

    def trans(d):
        return np.array([[1.0, d], [0.0, 1.0]])

    return refr(_R2, _N_GLASS, 1.0) @ trans(_TC) @ refr(_R1, 1.0, _N_GLASS)


def _q_trace_image(R_in, w_L):
    """Gaussian q at the entry vertex -> image distance past the exit
    vertex (waist location) + waist size.  Independent oracle."""
    q_inv = 1.0 / R_in - 1j * _WL / (np.pi * w_L ** 2)
    q0 = 1.0 / q_inv
    M = _lens_abcd()
    q1 = (M[0, 0] * q0 + M[0, 1]) / (M[1, 0] * q0 + M[1, 1])
    z_img = -q1.real                     # distance to the waist
    q_w = q1 + z_img
    w_img = np.sqrt(-_WL / (np.pi * (1.0 / q_w).imag))
    return float(z_img), float(w_img)


def _diverging_input(N, dx, w_L, R_in):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X ** 2 + Y ** 2
    k = 2 * np.pi / _WL
    return (np.exp(-r_sq / w_L ** 2)
            * np.exp(1j * k * r_sq / (2.0 * R_in))).astype(np.complex128)


def _ee100_at(E_exit, dx, z):
    E = la.angular_spectrum_propagate(E_exit, z, _WL, dx)
    I = np.abs(E) ** 2
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    j, i = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(x - x[i], x - x[j])
    r = np.sqrt(X ** 2 + Y ** 2)
    return float(I[r <= 100e-6].sum() / I.sum())


@pytest.mark.parametrize('R_in', [300e-3, 150e-3])
@pytest.mark.parametrize('carrier_mode', ['auto', 'explicit'])
def test_h6_diverging_input_focuses_at_the_abcd_image(R_in, carrier_mode):
    """A diverging Gaussian through the singlet must focus AT the
    q-trace image plane with EE(100um) > 0.9 under carrier mode.
    Pre-fix: EE(100um) ~ 0.01-0.05 (focus pinned at the collimated f,
    smear ~ NA_exit*(z_img - f))."""
    N, dx, w_L = 2048, 5e-6, 3e-3
    z_img, _ = _q_trace_image(R_in, w_L)
    E0 = _diverging_input(N, dx, w_L, R_in)
    carrier = 'auto' if carrier_mode == 'auto' else R_in
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')      # H3 guard may advise finer dx
        E_exit = la.apply_real_lens_traced(
            E0, prescription=_singlet(), wavelength=_WL, dx=dx,
            carrier=carrier)
    ee_img = _ee100_at(E_exit, dx, z_img)
    assert ee_img > 0.9, (
        f"R_in={R_in*1e3:.0f}mm carrier={carrier_mode}: EE(100um) at the "
        f"ABCD image plane z={z_img*1e3:.2f}mm is {ee_img:.3f} -- the H6 "
        f"entrance-eikonal omission gave ~0.01 (focus pinned at f)")
    # displacement guard: the image plane must beat the collimated-focus
    # plane f -- pre-fix the energy concentrated at f instead.
    f_plane = 49.163e-3                    # BFL (collimated focus)
    ee_f = _ee100_at(E_exit, dx, f_plane)
    assert ee_img > ee_f, (
        f"focus must sit at z_img={z_img*1e3:.2f}mm, not at the collimated "
        f"plane {f_plane*1e3:.2f}mm (EE_img={ee_img:.3f} vs EE_f={ee_f:.3f})")


def test_h6_collimated_carrier_auto_unchanged():
    """Collimated input (the hammer-validated case): carrier='auto' must
    agree with carrier=None -- W ~ 0, so the eikonal term is a no-op."""
    N, dx, w_L = 1024, 8e-6, 3e-3
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / w_L ** 2).astype(np.complex128)
    outs = {}
    for mode, kw in (('none', {}), ('auto', {'carrier': 'auto'})):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E_exit = la.apply_real_lens_traced(
                E0, prescription=_singlet(), wavelength=_WL, dx=dx, **kw)
        outs[mode] = _ee100_at(E_exit, dx, 49.163e-3)
    assert outs['auto'] > 0.9
    assert abs(outs['auto'] - outs['none']) < 0.05, outs


def test_h6_per_group_chain_hand_off():
    """Two-group relay applied group-by-group: every intermediate field
    carries finite curvature, so pre-fix carrier='auto' collapsed at
    EVERY hand-off (production exp22's 6-group smear).  With the eikonal
    term the chain must focus at the relayed image."""
    N, dx, w_L = 2048, 5e-6, 3e-3
    R_in = 150e-3
    z_img, _ = _q_trace_image(R_in, w_L)
    E0 = _diverging_input(N, dx, w_L, R_in)
    presc = _singlet()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # group 1 (traced, carrier auto) -> relay HALF-way -> re-apply as
        # a fresh 'group' (identity glass block is unphysical; instead
        # split the free-space leg and re-enter traced through a weak
        # second singlet placed mid-path with its own carrier fit)
        E1 = la.apply_real_lens_traced(
            E0, prescription=presc, wavelength=_WL, dx=dx, carrier='auto')
        z_half = 0.5 * z_img
        E_mid = la.angular_spectrum_propagate(E1, z_half, _WL, dx)
        # second group: a weak long-focus singlet (R = +/-500 mm) --
        # ABCD-compose for the final image location
        weak = {
            'wavelength': _WL, 'aperture_diameter': 20e-3,
            'surfaces': [
                {'radius': 500e-3, 'thickness': _TC,
                 'glass_before': 'air', 'glass_after': '_H6_FIX_GLASS',
                 'semi_diameter': 10e-3},
                {'radius': -500e-3, 'thickness': 0.0,
                 'glass_before': '_H6_FIX_GLASS', 'glass_after': 'air',
                 'semi_diameter': 10e-3},
            ],
            'thicknesses': [_TC], 'stop_index': 0,
        }
        E2 = la.apply_real_lens_traced(
            E_mid, prescription=weak, wavelength=_WL, dx=dx, carrier='auto')
    # find the best focus near the geometrically-expected remaining leg
    best = 0.0
    for dz in np.linspace(0.6, 1.1, 11) * (z_img - z_half):
        best = max(best, _ee100_at(E2, dx, float(dz)))
    assert best > 0.85, (
        f"per-group hand-off with carrier='auto' must focus (best "
        f"EE(100um) = {best:.3f}; pre-fix ~0.03)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

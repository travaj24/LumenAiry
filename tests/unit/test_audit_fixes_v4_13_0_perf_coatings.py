"""Correctness pin for the v4.13.0 coating-stack characteristic-matrix
vectorisation (Tier-2 perf, audit group alpha task 4).

The pre-v4.13.0 ``coating_reflectance`` walked the layer stack with
``M = M @ Mj`` in a Python loop.  v4.13.0 builds all per-layer 2x2
matrices as a (N, 2, 2) ndarray and reduces them with
``functools.reduce(np.matmul, ...)``.  This test pins the vectorised
reflectance / transmittance against an inline scalar-loop reference at
a 50-layer AR stack to within 1e-12.
"""
from __future__ import annotations

import numpy as np

from lumenairy.elements.coatings import coating_reflectance


def _scalar_reference(layers, wavelengths, angle=0.0,
                      n_substrate=1.52, n_ambient=1.0,
                      polarization='avg'):
    """Pre-v4.13.0 scalar-loop characteristic-matrix walker (uses
    ``M = M @ Mj``), inlined here as the pin reference."""
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=np.float64))
    n_wv = wavelengths.size
    R = np.empty(n_wv)
    T = np.empty(n_wv)
    phase_r = np.empty(n_wv)
    pols = ['s', 'p'] if polarization == 'avg' else [polarization]
    for iw, lam in enumerate(wavelengths):
        rs, ts = [], []
        eta_sub_by_pol, eta_amb_by_pol = {}, {}
        for pol in pols:
            M = np.eye(2, dtype=np.complex128)
            theta_prev = angle
            n_prev = complex(n_ambient)
            for n_layer, d in layers:
                n_layer = complex(n_layer)
                sin_t = n_prev.real * np.sin(theta_prev) / n_layer.real
                sin_t = min(sin_t, 0.9999)
                cos_t = np.sqrt(1 - sin_t ** 2)
                delta = 2 * np.pi * n_layer * d * cos_t / lam
                if pol == 's':
                    eta = n_layer * cos_t
                else:
                    eta = n_layer / cos_t
                Mj = np.array([
                    [np.cos(delta), -1j * np.sin(delta) / eta],
                    [-1j * eta * np.sin(delta), np.cos(delta)],
                ], dtype=np.complex128)
                M = M @ Mj
                theta_prev = np.arcsin(sin_t)
                n_prev = n_layer
            sin_sub = (n_prev.real * np.sin(theta_prev)
                       / complex(n_substrate).real)
            sin_sub = min(sin_sub, 0.9999)
            cos_sub = np.sqrt(1 - sin_sub ** 2)
            if pol == 's':
                eta_sub = complex(n_substrate) * cos_sub
                eta_amb = complex(n_ambient) * np.cos(angle)
            else:
                eta_sub = complex(n_substrate) / cos_sub
                eta_amb = complex(n_ambient) / np.cos(angle)
            eta_sub_by_pol[pol] = eta_sub
            eta_amb_by_pol[pol] = eta_amb
            B = M[0, 0] + M[0, 1] * eta_sub
            C = M[1, 0] + M[1, 1] * eta_sub
            r = (eta_amb * B - C) / (eta_amb * B + C)
            t_amp = 2.0 * eta_amb / (eta_amb * B + C)
            rs.append(r); ts.append(t_amp)
        if polarization == 'avg':
            R_val = 0.5 * (abs(rs[0]) ** 2 + abs(rs[1]) ** 2)
            phase_val = 0.5 * (np.angle(rs[0]) + np.angle(rs[1]))
            _ess = eta_sub_by_pol['s']; _eas = eta_amb_by_pol['s']
            _esp = eta_sub_by_pol['p']; _eap = eta_amb_by_pol['p']
            T_s = float((_ess.real / max(_eas.real, 1e-30))
                        * abs(ts[0]) ** 2)
            T_p = float((_esp.real / max(_eap.real, 1e-30))
                        * abs(ts[1]) ** 2)
            T_val = 0.5 * (T_s + T_p)
        else:
            R_val = abs(rs[0]) ** 2
            phase_val = np.angle(rs[0])
            T_val = float((eta_sub.real / max(eta_amb.real, 1e-30))
                          * abs(ts[0]) ** 2)
        R[iw] = R_val
        T[iw] = max(0.0, T_val)
        phase_r[iw] = phase_val
    return R, T, phase_r


def _build_50_layer_ar_stack(lam0: float = 550e-9):
    """50-layer alternating high/low quarter-wave stack (canonical
    test-case for TMM perf benchmarks)."""
    n_H = 2.3
    n_L = 1.38
    d_H = lam0 / (4 * n_H)
    d_L = lam0 / (4 * n_L)
    layers = []
    for i in range(25):
        layers.append((n_H, d_H))
        layers.append((n_L, d_L))
    return layers


def test_vectorised_matches_scalar_50_layer_normal():
    """50-layer AR stack at normal incidence, single wavelength."""
    layers = _build_50_layer_ar_stack()
    lam = 550e-9
    R_v, T_v, phi_v = coating_reflectance(
        layers, lam, angle=0.0, n_substrate=1.52, n_ambient=1.0,
        polarization='avg')
    R_s, T_s, phi_s = _scalar_reference(
        layers, lam, angle=0.0, n_substrate=1.52, n_ambient=1.0,
        polarization='avg')
    assert np.allclose(R_v, R_s, rtol=0, atol=1e-12), (
        f'R diff = {np.max(np.abs(R_v - R_s)):.3e}')
    assert np.allclose(T_v, T_s, rtol=0, atol=1e-12)
    assert np.allclose(phi_v, phi_s, rtol=0, atol=1e-12)


def test_vectorised_matches_scalar_50_layer_oblique_s_pol():
    """50-layer stack at 30 deg AOI, s-polarisation."""
    layers = _build_50_layer_ar_stack()
    lam = 633e-9
    R_v, T_v, phi_v = coating_reflectance(
        layers, lam, angle=np.deg2rad(30.0), n_substrate=1.52,
        n_ambient=1.0, polarization='s')
    R_s, T_s, phi_s = _scalar_reference(
        layers, lam, angle=np.deg2rad(30.0), n_substrate=1.52,
        n_ambient=1.0, polarization='s')
    assert np.allclose(R_v, R_s, rtol=0, atol=1e-12)
    assert np.allclose(T_v, T_s, rtol=0, atol=1e-12)


def test_vectorised_matches_scalar_wavelength_sweep():
    """Multi-wavelength sweep, 50 layers, avg-pol."""
    layers = _build_50_layer_ar_stack()
    wavelengths = np.linspace(400e-9, 800e-9, 11)
    R_v, T_v, _ = coating_reflectance(
        layers, wavelengths, polarization='avg')
    R_s, T_s, _ = _scalar_reference(
        layers, wavelengths, polarization='avg')
    assert np.allclose(R_v, R_s, rtol=0, atol=1e-12)
    assert np.allclose(T_v, T_s, rtol=0, atol=1e-12)

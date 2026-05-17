"""Correctness pin for the v4.13.0 thin-grating per-order loop
vectorisation (Tier-2 perf, audit group alpha task 2).

The pre-v4.13.0 thin_grating_efficiency_1d looped over diffraction
orders in Python; v4.13.0 broadcasts across the orders axis in a
single numpy expression.  This test asserts the vectorised path
matches the original analytical formula, evaluated inline, to within
machine precision, and that the output shape / order ordering is
preserved.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.thin_grating import thin_grating_efficiency_1d


def _scalar_reference(period, n_ridge, n_groove, n_substrate, n_superstrate,
                      depth, duty_cycle, wavelength, angle, n_orders):
    """Pre-v4.13.0 Python-loop implementation, inlined here as the
    pin reference."""
    k0 = 2 * np.pi / wavelength
    K = 2 * np.pi / period
    N = 2 * n_orders + 1
    orders = np.arange(-n_orders, n_orders + 1)
    f = duty_cycle
    phi_r = k0 * (complex(n_ridge) - n_substrate) * depth
    phi_g = k0 * (complex(n_groove) - n_substrate) * depth
    tm = np.zeros(N, dtype=np.complex128)
    for idx, m in enumerate(orders):
        if m == 0:
            tm[idx] = f * np.exp(1j * phi_r) + (1 - f) * np.exp(1j * phi_g)
        else:
            tm[idx] = ((np.exp(1j * phi_r) - np.exp(1j * phi_g))
                       * (np.exp(-1j * 2 * np.pi * m * f) - 1)
                       / (-1j * 2 * np.pi * m))
    kx0 = k0 * n_superstrate * np.sin(angle)
    kx = kx0 + orders * K
    k_sub = k0 * n_substrate
    propagating = np.abs(kx) < k_sub
    T_eff = np.where(propagating, np.abs(tm) ** 2, 0.0)
    R_eff = np.zeros(N)
    return orders, R_eff, T_eff


def test_vectorised_matches_scalar_loop_n_orders_20():
    """N=20 orders, generic binary grating: vectorised path must match
    inline scalar loop to within 1e-12 * max(|x|)."""
    args = dict(
        period=2e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.52,
        n_superstrate=1.0, depth=0.6e-6, duty_cycle=0.42,
        wavelength=633e-9, angle=0.0, n_orders=20,
    )
    o_v, R_v, T_v = thin_grating_efficiency_1d(**args)
    o_s, R_s, T_s = _scalar_reference(**args)
    assert np.array_equal(o_v, o_s)
    tol_T = 1e-12 * max(float(np.max(np.abs(T_s))), 1.0)
    tol_R = 1e-12 * max(float(np.max(np.abs(R_s))), 1.0)
    assert np.max(np.abs(T_v - T_s)) <= tol_T, (
        f'T mismatch: max diff {np.max(np.abs(T_v - T_s)):.3e}')
    assert np.max(np.abs(R_v - R_s)) <= tol_R


def test_vectorised_matches_scalar_loop_pi_depth():
    """Deep grating (pi-depth) splits power into +-1 orders; check
    vectorised path matches scalar loop there too."""
    lam = 1e-6
    d_pi = lam / (2 * (1.5 - 1.0))
    args = dict(
        period=5e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.52,
        n_superstrate=1.0, depth=d_pi, duty_cycle=0.5,
        wavelength=lam, angle=0.0, n_orders=25,
    )
    o_v, R_v, T_v = thin_grating_efficiency_1d(**args)
    o_s, R_s, T_s = _scalar_reference(**args)
    assert np.array_equal(o_v, o_s)
    assert np.allclose(T_v, T_s, rtol=0, atol=1e-12)
    assert np.allclose(R_v, R_s, rtol=0, atol=1e-12)


def test_vectorised_shape_and_order_axis():
    """Output shape must be (2*n_orders+1,) with orders centered on 0
    and monotonically increasing."""
    n_orders = 11
    orders, R, T = thin_grating_efficiency_1d(
        period=2e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.52,
        n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5,
        wavelength=633e-9, angle=0.0, n_orders=n_orders,
    )
    N = 2 * n_orders + 1
    assert orders.shape == (N,)
    assert R.shape == (N,)
    assert T.shape == (N,)
    # Order axis: centered on 0, strictly increasing by 1
    assert orders[0] == -n_orders
    assert orders[-1] == n_orders
    assert orders[n_orders] == 0
    assert np.array_equal(np.diff(orders), np.ones(N - 1, dtype=orders.dtype))

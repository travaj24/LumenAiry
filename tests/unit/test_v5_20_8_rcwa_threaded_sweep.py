"""Threaded RCWAStack.solve_vs_wavelength (parallel, byte-identical).

Per-wavelength solves are independent and NumPy releases the GIL inside LAPACK,
so RCWAStack.solve_vs_wavelength runs them on a bounded thread pool (each on a
private clone of the stack, with a per-worker thread-local BLAS pin).  Results
are stored by index, so the output is BYTE-IDENTICAL to a serial sweep
regardless of worker count.  These tests pin that identity (threaded ==
max_workers=1), the single-wavelength edge case, and the NaN/unstable handling.
"""
from __future__ import annotations

import warnings

import numpy as np

from lumenairy.elements.rcwa import RCWAStack


def _pillar(S, hw=0.2, eps=8.0, bg=2.1):
    x = (np.arange(S) + 0.5) / S - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    c = np.full((S, S), bg, complex)
    c[(np.abs(X) < hw) & (np.abs(Y) < hw)] = eps
    return c


def _stack():
    st = RCWAStack(period=0.5e-6, period_y=0.5e-6, n_superstrate=1.0,
                   n_substrate=1.5, n_orders=5, n_orders_y=5)
    st.add_layer(0.08e-6, eps=2.25)
    st.add_layer(0.12e-6, eps_cell=_pillar(40))
    return st.set_source(0.55e-6, theta=0.1, phi=0.2)


def test_threaded_sweep_is_byte_identical_to_serial():
    wls = np.linspace(0.50e-6, 0.62e-6, 12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o1, R1, T1, J1 = _stack().solve_vs_wavelength(wls, max_workers=1)
        o2, R2, T2, J2 = _stack().solve_vs_wavelength(wls, max_workers=4)
    assert np.array_equal(np.asarray(o1), np.asarray(o2))
    # NaN-safe byte comparison (unstable points, if any, are NaN in both)
    assert np.array_equal(np.nan_to_num(R1), np.nan_to_num(R2))
    assert np.array_equal(np.nan_to_num(T1), np.nan_to_num(T2))
    assert np.array_equal(np.nan_to_num(J1), np.nan_to_num(J2))
    assert np.array_equal(np.isnan(R1), np.isnan(R2))


def test_threaded_sweep_single_wavelength():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = _stack().solve_vs_wavelength([0.55e-6])   # auto workers
    assert R.shape[0] == 1 and T.shape[0] == 1 and J.shape[0] == 1
    # a single point -> serial path (max_workers collapses to 1); energy sane
    assert np.isfinite(R).any()


def test_threaded_sweep_matches_per_wavelength_solve():
    """A threaded sweep point matches a direct single-wavelength solve()."""
    wls = np.array([0.53e-6, 0.57e-6, 0.61e-6])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = _stack().solve_vs_wavelength(wls, max_workers=3)
        direct = _stack().set_source(0.57e-6, theta=0.1, phi=0.2).solve()
    od, Rd, Td = direct.efficiencies()
    assert np.max(np.abs(np.nan_to_num(R[1]) - np.asarray(Rd))) < 1e-12
    assert np.max(np.abs(np.nan_to_num(T[1]) - np.asarray(Td))) < 1e-12

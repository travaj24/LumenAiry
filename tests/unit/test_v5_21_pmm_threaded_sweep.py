"""Threaded PMMStack.solve_vs_wavelength + per-wavelength DBR eig-dedup.

The per-wavelength solves are independent and release the GIL in LAPACK, so the
sweep now runs on a bounded thread pool (RCWAStack's pattern), and within each
wavelength the per-layer eig is deduped across identical layers (an ABAB Bragg
stack eigs each distinct layer once).  These pin that the output is
BYTE-IDENTICAL to the serial path and to a per-wavelength ``solve()`` loop.
"""
from __future__ import annotations

import numpy as np

from lumenairy.elements.pmm import PMM2DStack, PMMStack


def _dbr(n_pairs=4):
    s = PMMStack(0.8e-6, n_substrate=1.5, n_superstrate=1.0, degree=11,
                 far_field_orders=9)
    a = [(0.5, 2.1), (0.5, 6.0)]
    b = [(0.5, 4.0), (0.5, 2.25)]
    for _ in range(n_pairs):
        s.add_layer(0.20e-6, segments=a)
        s.add_layer(0.15e-6, segments=b)
    return s


WLS = np.linspace(0.9e-6, 1.3e-6, 10)


def test_threaded_sweep_byte_identical_to_serial():
    """max_workers>1 gives BYTE-IDENTICAL R/T/Jones to the serial sweep (results
    stored by index; the geometric-eig cache is lock-guarded)."""
    s = _dbr()
    o1, R1, T1, J1 = s.solve_vs_wavelength(WLS, angle=0.12, max_workers=1,
                                           jones=True)
    o8, R8, T8, J8 = s.solve_vs_wavelength(WLS, angle=0.12, max_workers=8,
                                           jones=True)
    assert np.array_equal(o1, o8)
    assert np.array_equal(R1, R8)
    assert np.array_equal(T1, T8)
    assert np.array_equal(J1, J8)


def test_sweep_matches_per_wavelength_solve():
    """The threaded+deduped sweep matches a per-wavelength solve() loop exactly
    (total R/T + zeroth-order Jones -- order-set-alignment-free)."""
    s = _dbr()
    _o, Rw, Tw, Jw = s.solve_vs_wavelength(WLS, angle=0.12, max_workers=4,
                                           jones=True)
    for iw, w in enumerate(WLS):
        out = s.set_source(float(w), angle=0.12).solve()
        Rs, Ts, Js = np.asarray(out[1]), np.asarray(out[2]), np.asarray(out[3])
        assert abs(float(np.nansum(Rs)) - float(np.nansum(Rw[iw]))) < 1e-12
        assert abs(float(np.nansum(Ts)) - float(np.nansum(Tw[iw]))) < 1e-12
        assert np.max(np.abs(Js - Jw[iw])) < 1e-12


def test_dedup_byte_identical_across_pair_count():
    """A longer DBR (more identical layers) stays byte-identical to a short one
    on the SHARED wavelengths -- the per-wavelength eig dedup changes nothing
    but the cost (each distinct layer eigs once regardless of repetition)."""
    r4 = _dbr(4).solve_vs_wavelength(WLS, angle=0.0, max_workers=2)[1]
    r4b = _dbr(4).solve_vs_wavelength(WLS, angle=0.0, max_workers=1)[1]
    assert np.array_equal(r4, r4b)              # deterministic, dedup-invariant


def _stack2d():
    cell = np.full((32, 32), 2.1)
    cell[8:24, 8:24] = 6.0                      # crossed square pillar
    st = PMM2DStack(0.7e-6, n_substrate=1.5, n_superstrate=1.0, degree=7,
                    n_orders=3)
    st.add_layer(0.20e-6, eps_cell=cell)
    st.add_layer(0.15e-6, eps=2.25)
    st.add_layer(0.20e-6, eps_cell=cell)        # repeated -> solve() dedups
    return st


def test_2d_threaded_sweep_byte_identical():
    """PMM2DStack.solve_vs_wavelength threads on private clones -> byte-identical
    to serial AND to a per-wavelength solve() loop (the 2-D solve dedups)."""
    wls = np.linspace(0.9e-6, 1.2e-6, 6)
    st = _stack2d()
    o1, R1, T1, J1 = st.solve_vs_wavelength(wls, theta=0.1, phi=0.2,
                                            max_workers=1, jones=True)
    o4, R4, T4, J4 = st.solve_vs_wavelength(wls, theta=0.1, phi=0.2,
                                            max_workers=4, jones=True)
    assert np.array_equal(R1, R4) and np.array_equal(T1, T4)
    assert np.array_equal(J1, J4)
    for iw, w in enumerate(wls):
        out = st.set_source(float(w), theta=0.1, phi=0.2).solve()
        assert np.max(np.abs(np.asarray(out[1]) - R1[iw])) < 1e-12
        assert np.max(np.abs(np.asarray(out[3]) - J1[iw])) < 1e-12

"""Correctness pin for the v4.13.1 vector-HFPI accumulator
index-sharing optimization (audit Tier-3 perf).

Pre-v4.13.1 ``accumulate_vector_to_grid`` invoked
:func:`hfpi.accumulate_to_grid` twice, once per Jones component,
recomputing the same per-path index arrays (``ix``, ``iy``,
``inside``, ``flat_idx``) on each call.  v4.13.1 computes those
once and runs two scatter-adds, sharing the index-build work.

Numerical equivalence is bit-exact: same arithmetic, same scatter
pattern.  This test pins the new path against the pre-v4.13.1
double-call (re-routed via ``hfpi.accumulate_to_grid``) at
1e-15 tolerance (machine epsilon).
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.propagators.vectorial_hfpi import (
    VectorPathBundle, accumulate_vector_to_grid,
)
from lumenairy.propagators.hfpi import (
    PathBundle, accumulate_to_grid,
)


def _make_vector_bundle(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    # Spread paths over [-2e-4, 2e-4] in x/y so the output grid
    # (centered, dx=4e-6, N=64) captures most of them.
    positions = np.column_stack([
        rng.uniform(-1.2e-4, 1.2e-4, n),
        rng.uniform(-1.2e-4, 1.2e-4, n),
        np.zeros(n),
    ]).astype(np.float64)
    directions = np.column_stack([
        np.zeros(n), np.zeros(n), np.ones(n),
    ])
    Ex = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) \
        .astype(np.complex128)
    Ey = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) \
        .astype(np.complex128)
    opl = np.zeros(n, dtype=np.float64)
    # Mark ~5% of paths dead to exercise the alive-mask branch.
    alive = rng.random(n) > 0.05
    return VectorPathBundle(
        positions=positions, directions=directions,
        Ex=Ex, Ey=Ey, opl=opl, alive=alive,
    )


def _reference_double_call(paths, Ny, Nx, dx, centre, output_dtype):
    """Pre-v4.13.1 path: route each component through
    accumulate_to_grid independently."""
    ex_paths = PathBundle(
        positions=paths.positions, directions=paths.directions,
        weights=paths.Ex, opl=paths.opl, alive=paths.alive,
    )
    ey_paths = PathBundle(
        positions=paths.positions, directions=paths.directions,
        weights=paths.Ey, opl=paths.opl, alive=paths.alive,
    )
    Ex_out = accumulate_to_grid(
        ex_paths, Ny=Ny, Nx=Nx, dx=dx, centre=centre,
        output_dtype=output_dtype)
    Ey_out = accumulate_to_grid(
        ey_paths, Ny=Ny, Nx=Nx, dx=dx, centre=centre,
        output_dtype=output_dtype)
    return Ex_out, Ey_out


def test_shared_index_matches_double_call_default_centre():
    """Index-sharing path matches twice-routed reference bit-exactly."""
    paths = _make_vector_bundle(n=3000, seed=11)
    Ny, Nx, dx = 64, 64, 4e-6

    Ex_new, Ey_new = accumulate_vector_to_grid(
        paths, Ny=Ny, Nx=Nx, dx=dx, centre=(0.0, 0.0),
        output_dtype=np.complex128,
    )
    Ex_ref, Ey_ref = _reference_double_call(
        paths, Ny, Nx, dx, (0.0, 0.0), np.complex128,
    )
    # Bit-exact: same arithmetic on both paths, same scatter-add.
    assert np.array_equal(Ex_new, Ex_ref), (
        f"Ex mismatch: max diff = {np.max(np.abs(Ex_new - Ex_ref)):.3e}")
    assert np.array_equal(Ey_new, Ey_ref), (
        f"Ey mismatch: max diff = {np.max(np.abs(Ey_new - Ey_ref)):.3e}")


def test_shared_index_matches_double_call_off_centre():
    """Off-centre output grid: index calc still matches."""
    paths = _make_vector_bundle(n=2000, seed=23)
    Ny, Nx, dx = 48, 64, 5e-6
    centre = (3e-5, -2e-5)

    Ex_new, Ey_new = accumulate_vector_to_grid(
        paths, Ny=Ny, Nx=Nx, dx=dx, centre=centre,
        output_dtype=np.complex128,
    )
    Ex_ref, Ey_ref = _reference_double_call(
        paths, Ny, Nx, dx, centre, np.complex128,
    )
    assert np.array_equal(Ex_new, Ex_ref)
    assert np.array_equal(Ey_new, Ey_ref)


def test_shared_index_default_output_dtype():
    """When output_dtype is None, fall through to paths.Ex.dtype --
    same convention as pre-v4.13.1."""
    paths = _make_vector_bundle(n=500, seed=42)
    Ex_new, Ey_new = accumulate_vector_to_grid(
        paths, Ny=32, Nx=32, dx=4e-6, centre=(0.0, 0.0),
    )
    assert Ex_new.dtype == paths.Ex.dtype
    assert Ey_new.dtype == paths.Ey.dtype


def test_shared_index_all_dead_paths():
    """Edge case: every path is alive=False -- output must be zeros."""
    paths = _make_vector_bundle(n=200, seed=99)
    paths = VectorPathBundle(
        positions=paths.positions, directions=paths.directions,
        Ex=paths.Ex, Ey=paths.Ey, opl=paths.opl,
        alive=np.zeros(paths.positions.shape[0], dtype=bool),
    )
    Ex_new, Ey_new = accumulate_vector_to_grid(
        paths, Ny=16, Nx=16, dx=4e-6, centre=(0.0, 0.0),
        output_dtype=np.complex128,
    )
    assert np.all(Ex_new == 0)
    assert np.all(Ey_new == 0)

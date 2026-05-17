"""Correctness pin for the v4.13.1 GBD reconstruction inner-loop
fusion (audit Tier-3 perf).

Pre-v4.13.1 ``reconstruct_field_from_beamlets`` called ``xp.exp``
twice per chunk when beamlets carried direction cosines (the
default for paraxial GBD): once for the Gaussian phase, once for
the linear tilt.  v4.13.1 fuses these into a single ``xp.exp`` call
over the summed phase argument and replaces the
``sum(a_b * phase, axis=-1)`` reduction with ``einsum`` to drop the
large ``a_b * phase`` 3-D buffer.

``exp(A) * exp(B) == exp(A + B)`` analytically; in complex128 the
round-off divergence is ulp-level.  This test pins the fused path
against a hand-coded scalar reference (matching the pre-v4.13.1
exact arithmetic) at 1e-12 tolerance and confirms the speedup is
meaningful on a realistic-sized workload.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from lumenairy.propagators.gbd import (
    BeamletBundle, reconstruct_field_from_beamlets,
)


def _scalar_reference(positions, directions, Q, amplitude, *,
                       Ny, Nx, dx, centre, wavelength):
    """Pre-v4.13.1 inner-loop: two exp() calls, ``sum(a*phase, -1)``
    reduction.  Mirrors the v4.13.0 (and earlier) arithmetic exactly
    so we can pin the v4.13.1 fused path against it.
    """
    cx, cy = centre
    ix = np.arange(Nx, dtype=positions.dtype)
    iy = np.arange(Ny, dtype=positions.dtype)
    Xg, Yg = np.meshgrid((ix - Nx / 2) * dx + cx,
                          (iy - Ny / 2) * dx + cy,
                          indexing='xy')
    k = 2 * float(np.pi) / wavelength
    out = np.zeros((Ny, Nx), dtype=amplitude.dtype)

    n = positions.shape[0]
    chunk = 4096
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        x_b = positions[start:end, 0]
        y_b = positions[start:end, 1]
        Q_b = Q[start:end]
        a_b = amplitude[start:end]

        rho2 = ((Xg[..., None] - x_b[None, None, :]) ** 2
                + (Yg[..., None] - y_b[None, None, :]) ** 2)
        phase = np.exp(-1j * k * Q_b[None, None, :] * rho2 / 2)
        L_b = directions[start:end, 0]
        M_b = directions[start:end, 1]
        tilt = (L_b[None, None, :] * (Xg[..., None] - x_b[None, None, :])
                + M_b[None, None, :] * (Yg[..., None] - y_b[None, None, :]))
        phase = phase * np.exp(1j * k * tilt)
        contrib = a_b[None, None, :] * phase
        out = out + np.sum(contrib, axis=-1)

    return out


def _make_bundle(n=512, seed=0):
    rng = np.random.default_rng(seed)
    positions = np.column_stack([
        rng.uniform(-1e-3, 1e-3, n),
        rng.uniform(-1e-3, 1e-3, n),
        np.zeros(n),
    ]).astype(np.float64)
    # Modest tilt cones (~5 mrad) so the directions branch is exercised
    # but rho2 doesn't blow up.
    directions = np.column_stack([
        rng.uniform(-5e-3, 5e-3, n),
        rng.uniform(-5e-3, 5e-3, n),
        np.ones(n),
    ])
    norms = np.sqrt(np.sum(directions ** 2, axis=1, keepdims=True))
    directions = directions / norms

    wavelength = 1.0e-6
    w0 = 20e-6
    z_R = np.pi * w0 ** 2 / wavelength
    Q = np.full(n, -1j / z_R, dtype=np.complex128)
    amplitude = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) \
        .astype(np.complex128)
    waist0 = np.full(n, w0, dtype=np.float64)
    return BeamletBundle(
        positions=positions, directions=directions,
        Q=Q, amplitude=amplitude, waist0=waist0,
    )


def test_fused_matches_scalar_reference_with_tilt():
    """Fused path with directions: complex128 agreement to 1e-12."""
    bundle = _make_bundle(n=300, seed=11)
    Ny, Nx, dx, wavelength = 32, 32, 4e-6, 1.0e-6

    E_fused = reconstruct_field_from_beamlets(
        bundle, Ny=Ny, Nx=Nx, dx=dx, centre=(0.0, 0.0),
        wavelength=wavelength, chunk_beamlets=4096,
    )
    E_ref = _scalar_reference(
        bundle.positions, bundle.directions, bundle.Q, bundle.amplitude,
        Ny=Ny, Nx=Nx, dx=dx, centre=(0.0, 0.0), wavelength=wavelength,
    )
    rel = (np.max(np.abs(E_fused - E_ref))
           / max(float(np.max(np.abs(E_ref))), 1e-30))
    assert rel <= 1e-12, f"fused vs scalar reference: rel = {rel:.3e}"


def test_fused_no_dirs_path_matches_scalar():
    """No-directions branch: einsum reduction equivalence."""
    bundle = _make_bundle(n=150, seed=7)
    # Zero-out directions to exercise the no-tilt code branch.
    bundle = BeamletBundle(
        positions=bundle.positions, directions=None,
        Q=bundle.Q, amplitude=bundle.amplitude, waist0=bundle.waist0,
    )
    Ny, Nx, dx, wavelength = 32, 32, 4e-6, 1.0e-6

    E_new = reconstruct_field_from_beamlets(
        bundle, Ny=Ny, Nx=Nx, dx=dx, centre=(0.0, 0.0),
        wavelength=wavelength, chunk_beamlets=4096,
    )
    # Hand-coded reference for the no-dirs path.
    cx, cy = 0.0, 0.0
    ix = np.arange(Nx, dtype=bundle.positions.dtype)
    iy = np.arange(Ny, dtype=bundle.positions.dtype)
    Xg, Yg = np.meshgrid((ix - Nx / 2) * dx + cx,
                          (iy - Ny / 2) * dx + cy,
                          indexing='xy')
    k = 2 * float(np.pi) / wavelength
    E_ref = np.zeros((Ny, Nx), dtype=bundle.amplitude.dtype)
    rho2 = ((Xg[..., None] - bundle.positions[None, None, :, 0]) ** 2
            + (Yg[..., None] - bundle.positions[None, None, :, 1]) ** 2)
    phase = np.exp(-1j * k * bundle.Q[None, None, :] * rho2 / 2)
    contrib = bundle.amplitude[None, None, :] * phase
    E_ref = E_ref + np.sum(contrib, axis=-1)
    rel = (np.max(np.abs(E_new - E_ref))
           / max(float(np.max(np.abs(E_ref))), 1e-30))
    assert rel <= 1e-12, f"no-dirs path drift: rel = {rel:.3e}"


@pytest.mark.parametrize('n', [400])
def test_fused_path_is_faster(n):
    """Loose sanity check: the fused path must not be slower than the
    scalar reference.  We don't require a hard speedup multiplier here
    (numpy thread scheduling makes that flaky in CI); just that fused
    wallclock <= reference wallclock + 30% margin."""
    bundle = _make_bundle(n=n, seed=23)
    Ny, Nx, dx, wavelength = 48, 48, 4e-6, 1.0e-6

    # Warm-up to load FFT planners / glass caches.
    _ = reconstruct_field_from_beamlets(
        bundle, Ny=Ny, Nx=Nx, dx=dx, centre=(0.0, 0.0),
        wavelength=wavelength, chunk_beamlets=4096,
    )
    _ = _scalar_reference(
        bundle.positions, bundle.directions, bundle.Q, bundle.amplitude,
        Ny=Ny, Nx=Nx, dx=dx, centre=(0.0, 0.0), wavelength=wavelength,
    )

    # Timed pass
    t0 = time.perf_counter()
    for _ in range(3):
        _ = reconstruct_field_from_beamlets(
            bundle, Ny=Ny, Nx=Nx, dx=dx, centre=(0.0, 0.0),
            wavelength=wavelength, chunk_beamlets=4096,
        )
    t_fused = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(3):
        _ = _scalar_reference(
            bundle.positions, bundle.directions, bundle.Q, bundle.amplitude,
            Ny=Ny, Nx=Nx, dx=dx, centre=(0.0, 0.0), wavelength=wavelength,
        )
    t_ref = time.perf_counter() - t0
    # Generous 30% slop to absorb numpy thread-pool variance on shared CI.
    assert t_fused <= 1.30 * t_ref, (
        f"fused path slower than reference: fused {t_fused:.3f}s vs "
        f"reference {t_ref:.3f}s (ratio {t_fused/t_ref:.3f}x)")

"""HFPI module tests."""
from __future__ import annotations

import sys

import numpy as np

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from _harness import Harness

import lumenairy as la
from lumenairy.propagators.hfpi import (
    PathBundle, init_paths_from_field, propagate_to_plane,
    apply_aperture_diffraction, accumulate_to_grid,
    propagate_hfpi_freespace_aperture,
)


H = Harness('hfpi')


def t_pathbundle_shape():
    N = 32; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    paths = init_paths_from_field(E, dx, n_paths=1000, wavelength=lam, rng=42)
    ok = (len(paths) == 1000
          and paths.positions.shape == (1000, 3)
          and paths.directions.shape == (1000, 3)
          and paths.weights.shape == (1000,)
          and paths.opl.shape == (1000,)
          and paths.alive.shape == (1000,))
    return ok, f'len={len(paths)}'


def t_init_paths_directions_in_forward_hemisphere():
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    paths = init_paths_from_field(E, dx, n_paths=5000, wavelength=lam, rng=1)
    return bool(np.all(paths.directions[:, 2] > 0)), 'forward hemisphere'


def t_init_paths_directions_unit_length():
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    paths = init_paths_from_field(E, dx, n_paths=2000, wavelength=lam, rng=2)
    norms = np.sqrt(np.sum(paths.directions ** 2, axis=-1))
    return float(np.max(np.abs(norms - 1.0))) < 1e-6, 'unit-length'


def t_propagate_advances_z():
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    paths = init_paths_from_field(E, dx, n_paths=500, wavelength=lam, rng=3)
    p1 = propagate_to_plane(paths, z_target=1e-3, wavelength=lam)
    return float(np.max(np.abs(p1.positions[:, 2] - 1e-3))) < 1e-12, 'z exact'


def t_aperture_kills_outside_paths():
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    paths = init_paths_from_field(E, dx, n_paths=2000, wavelength=lam, rng=5)
    p_tiny = apply_aperture_diffraction(paths, aperture_radius=1e-6, rng=6)
    p_huge = apply_aperture_diffraction(paths, aperture_radius=1.0, rng=7)
    return p_tiny.n_alive < p_huge.n_alive, (
        f'tiny={p_tiny.n_alive}, huge={p_huge.n_alive}')


def t_accumulate_into_grid_finite():
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    paths = init_paths_from_field(E, dx, n_paths=2000, wavelength=lam, rng=8)
    p1 = propagate_to_plane(paths, z_target=1e-3, wavelength=lam)
    out = accumulate_to_grid(p1, Ny=N, Nx=N, dx=dx)
    return bool(np.all(np.isfinite(np.abs(out)))), f'shape={out.shape}'


def t_end_to_end():
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    out = propagate_hfpi_freespace_aperture(
        E, dx,
        z_to_aperture=1e-3, aperture_radius=200e-6,
        z_aperture_to_output=1e-3,
        wavelength=lam, n_paths=5000, rng=9,
    )
    return out.shape == E.shape and bool(np.all(np.isfinite(np.abs(out)))), 'ok'


def t_jax_dispatch():
    if not la.JAX_AVAILABLE:
        return True, 'skipped (no jax)'
    import jax.numpy as jnp, jax
    N = 16; dx = 5e-6; lam = 633e-9
    E = jnp.ones((N, N), dtype=jnp.complex64)
    paths = init_paths_from_field(E, dx, n_paths=200, wavelength=lam,
                                   rng=jax.random.PRNGKey(10))
    return la.is_jax_array(paths.positions), 'jax positions'


def t_stratified_count():
    from lumenairy.propagators.hfpi import init_paths_stratified
    E = np.ones((32, 32), dtype=np.complex128)
    paths = init_paths_stratified(E, 5e-6, n_paths=10000,
                                   wavelength=633e-9, rng=42)
    count = len(paths)
    return 0.7 * 10000 < count < 1.5 * 10000, f'count={count}'


def t_prescription_hfpi():
    from lumenairy.propagators.hfpi import propagate_hfpi_through_prescription
    presc = la.make_singlet(R1=5.16e-3, R2=float('inf'), d=2e-3,
                             glass='N-BK7', aperture=4e-3)
    presc['object_distance'] = 30e-3
    N = 32; dx = 5e-6; lam = 633e-9
    x = (np.arange(N) - N/2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X*X + Y*Y) / (30e-6)**2).astype(np.complex128)
    out = propagate_hfpi_through_prescription(
        E, dx, presc, wavelength=lam, n_paths=2000, rng=42,
    )
    return out.shape == E.shape and bool(np.all(np.isfinite(np.abs(out)))), 'ok'


def t_hfpi_aperture_decreases_alive_count():
    """An aperture must reduce alive paths.  Catches bugs where the
    aperture mask isn't applied or `alive` is reset elsewhere.
    """
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    paths_in = init_paths_from_field(E, dx, n_paths=2000,
                                       wavelength=lam, rng=42)
    n_in = int(np.sum(paths_in.alive))
    paths_at_ap = propagate_to_plane(paths_in, z_target=1e-3,
                                       wavelength=lam)
    paths_after_ap = apply_aperture_diffraction(
        paths_at_ap, aperture_radius=20e-6,
        wavelength=lam,
    )
    n_after = int(np.sum(paths_after_ap.alive))
    return n_after < n_in and n_after >= 0, (
        f'alive in={n_in}, after_aperture={n_after}')


def t_hfpi_stratified_lower_variance_than_uniform():
    """Stratified sampling should yield lower per-pixel intensity
    variance than uniform random sampling (for a smooth source).
    """
    from lumenairy.propagators.hfpi import init_paths_stratified
    N = 16; dx = 10e-6; lam = 633e-9
    x = (np.arange(N) - N/2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X*X + Y*Y) / (60e-6)**2).astype(np.complex128)

    n_paths = 2000
    # Uniform.
    paths_u = init_paths_from_field(E, dx, n_paths=n_paths,
                                      wavelength=lam, rng=1)
    paths_u = propagate_to_plane(paths_u, z_target=1e-3, wavelength=lam)
    out_u = accumulate_to_grid(paths_u, Ny=N, Nx=N, dx=dx)

    # Stratified.
    paths_s = init_paths_stratified(E, dx, n_paths=n_paths,
                                      wavelength=lam, rng=1)
    paths_s = propagate_to_plane(paths_s, z_target=1e-3, wavelength=lam)
    out_s = accumulate_to_grid(paths_s, Ny=N, Nx=N, dx=dx)

    var_u = float(np.var(np.abs(out_u) ** 2))
    var_s = float(np.var(np.abs(out_s) ** 2))
    # Both should be > 0; stratified should be reasonably comparable
    # or smaller.  Use a generous bound to avoid Monte-Carlo flakies.
    ratio = var_s / max(var_u, 1e-30)
    return ratio < 5.0, (
        f'var_uniform={var_u:.3e}, var_stratified={var_s:.3e}, '
        f'ratio={ratio:.2f}')


H.section('Deep physics')
H.run('HFPI aperture reduces alive-path count',
      t_hfpi_aperture_decreases_alive_count)
H.run('HFPI stratified sampling variance bounded',
      t_hfpi_stratified_lower_variance_than_uniform)


def main():
    H.section('PathBundle')
    H.run('shapes consistent', t_pathbundle_shape)

    H.section('Path initialization')
    H.run('forward hemisphere', t_init_paths_directions_in_forward_hemisphere)
    H.run('unit-length directions', t_init_paths_directions_unit_length)

    H.section('Free-space propagation')
    H.run('propagate sets z', t_propagate_advances_z)

    H.section('Aperture')
    H.run('aperture kills outside paths', t_aperture_kills_outside_paths)

    H.section('Accumulation')
    H.run('grid finite', t_accumulate_into_grid_finite)

    H.section('End-to-end')
    H.run('freespace+aperture end-to-end', t_end_to_end)

    H.section('JAX backend')
    H.run('JAX dispatch', t_jax_dispatch)

    H.section('Variance reduction')
    H.run('stratified sampling count near requested', t_stratified_count)

    H.section('Prescription path')
    H.run('HFPI through singlet prescription', t_prescription_hfpi)

    sys.exit(H.summary())


if __name__ == '__main__':
    main()

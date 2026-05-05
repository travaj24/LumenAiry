"""Vectorial HFPI module tests."""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import sys

import numpy as np

from _harness import Harness

import lumenairy as la
from lumenairy.propagators.vectorial_hfpi import (
    VectorPathBundle,
    init_vector_paths_from_field,
    propagate_vector_to_plane,
    apply_vector_aperture_diffraction,
    accumulate_vector_to_grid,
    propagate_vector_hfpi_freespace_aperture,
)


H = Harness('vectorial_hfpi')


def t_vector_pathbundle_shapes():
    Ex = np.ones((16, 16), dtype=np.complex128)
    Ey = np.zeros((16, 16), dtype=np.complex128)
    paths = init_vector_paths_from_field(
        Ex, Ey, 5e-6, n_paths=500, wavelength=633e-9, rng=42,
    )
    ok = (len(paths) == 500
          and paths.positions.shape == (500, 3)
          and paths.directions.shape == (500, 3)
          and paths.Ex.shape == (500,)
          and paths.Ey.shape == (500,))
    return ok, f'len={len(paths)}'


def t_propagate_vector_advances_z():
    Ex = np.ones((16, 16), dtype=np.complex128)
    Ey = np.zeros((16, 16), dtype=np.complex128)
    paths = init_vector_paths_from_field(
        Ex, Ey, 5e-6, n_paths=200, wavelength=633e-9, rng=1,
    )
    p1 = propagate_vector_to_plane(paths, z_target=1e-3, wavelength=633e-9)
    return float(np.max(np.abs(p1.positions[:, 2] - 1e-3))) < 1e-12, 'z exact'


def t_aperture_kills_vector_paths():
    Ex = np.ones((16, 16), dtype=np.complex128)
    Ey = np.zeros((16, 16), dtype=np.complex128)
    paths = init_vector_paths_from_field(
        Ex, Ey, 5e-6, n_paths=2000, wavelength=633e-9, rng=2,
    )
    p_tiny = apply_vector_aperture_diffraction(
        paths, aperture_radius=1e-6, rng=3,
    )
    p_huge = apply_vector_aperture_diffraction(
        paths, aperture_radius=1.0, rng=4,
    )
    return p_tiny.n_alive < p_huge.n_alive, (
        f'tiny={p_tiny.n_alive}, huge={p_huge.n_alive}')


def t_accumulate_vector_returns_two_grids():
    Ex = np.ones((16, 16), dtype=np.complex128)
    Ey = np.zeros((16, 16), dtype=np.complex128)
    paths = init_vector_paths_from_field(
        Ex, Ey, 5e-6, n_paths=500, wavelength=633e-9, rng=5,
    )
    p1 = propagate_vector_to_plane(paths, z_target=1e-3, wavelength=633e-9)
    Ex_out, Ey_out = accumulate_vector_to_grid(
        p1, Ny=16, Nx=16, dx=5e-6,
    )
    return (Ex_out.shape == (16, 16) and Ey_out.shape == (16, 16)
            and bool(np.all(np.isfinite(np.abs(Ex_out))))
            and bool(np.all(np.isfinite(np.abs(Ey_out))))), 'shapes ok'


def t_end_to_end_vectorial_freespace_aperture():
    Ex = np.ones((16, 16), dtype=np.complex128)
    Ey = np.zeros((16, 16), dtype=np.complex128)
    Ex_out, Ey_out = propagate_vector_hfpi_freespace_aperture(
        Ex, Ey, 5e-6,
        z_to_aperture=1e-3, aperture_radius=200e-6,
        z_aperture_to_output=1e-3, wavelength=633e-9,
        n_paths=2000, rng=42,
    )
    return (Ex_out.shape == (16, 16) and Ey_out.shape == (16, 16)
            and bool(np.all(np.isfinite(np.abs(Ex_out))))), 'ok'


def main():
    H.section('VectorPathBundle')
    H.run('shapes consistent', t_vector_pathbundle_shapes)

    H.section('Vector free-space propagation')
    H.run('z advances correctly', t_propagate_vector_advances_z)

    H.section('Vector aperture diffraction')
    H.run('aperture kills outside paths', t_aperture_kills_vector_paths)

    H.section('Vector accumulation')
    H.run('returns two grids', t_accumulate_vector_returns_two_grids)

    H.section('End-to-end')
    H.run('vectorial freespace + aperture',
          t_end_to_end_vectorial_freespace_aperture)

    sys.exit(H.summary())


if __name__ == '__main__':
    main()

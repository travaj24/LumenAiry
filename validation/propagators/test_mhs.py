"""MHS (Multiple Huygens Surface) pipeline tests."""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import sys

import numpy as np

from _harness import Harness

import lumenairy as la
from lumenairy.propagators.mhs import (
    HuygensSurface, MhsSubdomain, MhsPipeline,
    asm_subdomain, aperture_subdomain,
    gbd_freespace_subdomain, prescription_subdomain,
)


H = Harness('mhs')


def t_huygens_surface_grid_shape():
    s = HuygensSurface(z=0.0, Ny=32, Nx=32, dx=5e-6)
    X, Y = s.grid()
    return X.shape == (32, 32) and Y.shape == (32, 32), 'grid shape ok'


def t_pipeline_validates_surface_chain():
    """Mismatched surfaces between subdomains should raise ValueError."""
    s0 = HuygensSurface(z=0.0, Ny=32, Nx=32, dx=5e-6, label='a')
    s1 = HuygensSurface(z=1e-3, Ny=32, Nx=32, dx=5e-6, label='b')
    s_bad = HuygensSurface(z=2e-3, Ny=64, Nx=64, dx=5e-6, label='wrong-grid')
    sub_a = asm_subdomain(s0, s1, wavelength=633e-9)
    sub_b = asm_subdomain(s_bad, s_bad, wavelength=633e-9)
    try:
        MhsPipeline([sub_a, sub_b])
        return False, 'expected ValueError'
    except ValueError:
        return True, 'mismatch detected'


def t_pipeline_runs_3_subdomains():
    """3-subdomain pipeline (asm -> aperture -> asm)."""
    N = 32; dx = 5e-6; lam = 633e-9
    s0 = HuygensSurface(z=0.0, Ny=N, Nx=N, dx=dx, label='source')
    s1 = HuygensSurface(z=1e-3, Ny=N, Nx=N, dx=dx, label='aperture')
    s2 = HuygensSurface(z=2e-3, Ny=N, Nx=N, dx=dx, label='output')

    sub_a = asm_subdomain(s0, s1, wavelength=lam)
    sub_ap = aperture_subdomain(s1, aperture_radius=50e-6)
    sub_c = asm_subdomain(s1, s2, wavelength=lam)
    pipe = MhsPipeline([sub_a, sub_ap, sub_c])

    x = (np.arange(N) - N/2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E_in = np.exp(-(X*X + Y*Y) / (30e-6)**2).astype(np.complex128)

    history = pipe.run(E_in)
    return (len(history) == 4
            and bool(np.all(np.isfinite(np.abs(history[-1][1]))))), (
        f'history len={len(history)}')


def t_pipeline_final_field_only():
    """run(return_intermediate=False) returns just the final field."""
    N = 16; dx = 5e-6; lam = 633e-9
    s0 = HuygensSurface(z=0.0, Ny=N, Nx=N, dx=dx)
    s1 = HuygensSurface(z=1e-3, Ny=N, Nx=N, dx=dx)
    sub_a = asm_subdomain(s0, s1, wavelength=lam)
    pipe = MhsPipeline([sub_a])
    E_in = np.ones((N, N), dtype=np.complex128)
    out = pipe.run(E_in, return_intermediate=False)
    return out.shape == (N, N), 'final field shape'


def t_aperture_subdomain_clips():
    """Aperture subdomain zeros field outside the aperture."""
    N = 32; dx = 5e-6
    s = HuygensSurface(z=0.0, Ny=N, Nx=N, dx=dx)
    sub = aperture_subdomain(s, aperture_radius=10e-6, shape='circular')
    E_in = np.ones((N, N), dtype=np.complex128)
    E_out = sub.propagator(E_in, sub.in_surface, sub.out_surface, **sub.kwargs)
    # Centre pixel should be inside (E ~ 1), corner pixel outside (E ~ 0).
    centre = E_out[N//2, N//2]
    corner = E_out[0, 0]
    return abs(centre) > 0.5 and abs(corner) < 1e-12, (
        f'centre={float(abs(centre)):.4f}, corner={float(abs(corner)):.4e}')


def main():
    H.section('HuygensSurface')
    H.run('grid() returns correct shape', t_huygens_surface_grid_shape)

    H.section('Pipeline validation')
    H.run('mismatched surfaces raise ValueError',
          t_pipeline_validates_surface_chain)

    H.section('Pipeline execution')
    H.run('3-subdomain ASM->aperture->ASM pipeline',
          t_pipeline_runs_3_subdomains)
    H.run('return_intermediate=False returns final field only',
          t_pipeline_final_field_only)

    H.section('Subdomain builders')
    H.run('aperture_subdomain clips correctly',
          t_aperture_subdomain_clips)

    sys.exit(H.summary())


if __name__ == '__main__':
    main()

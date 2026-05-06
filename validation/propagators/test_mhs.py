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


def t_pipeline_checkpoint_callback():
    """checkpoint(idx, surface, field) is called after every subdomain."""
    N = 16; dx = 5e-6; lam = 633e-9
    s0 = HuygensSurface(z=0.0, Ny=N, Nx=N, dx=dx, label='in')
    s1 = HuygensSurface(z=1e-3, Ny=N, Nx=N, dx=dx, label='mid')
    s2 = HuygensSurface(z=2e-3, Ny=N, Nx=N, dx=dx, label='out')
    sub_a = asm_subdomain(s0, s1, wavelength=lam)
    sub_b = asm_subdomain(s1, s2, wavelength=lam)
    pipe = MhsPipeline([sub_a, sub_b])
    seen = []
    def cb(idx, surface, field):
        seen.append((idx, surface.label, field.shape))
    E_in = np.ones((N, N), dtype=np.complex128)
    final = pipe.run(E_in, return_intermediate=False, checkpoint=cb)
    expected = [(0, 'in', (N, N)), (1, 'mid', (N, N)), (2, 'out', (N, N))]
    return (seen == expected and final.shape == (N, N)), (
        f'seen={seen}')


def t_mhs_single_asm_matches_direct_asm():
    """MhsPipeline([asm_subdomain]) gives the SAME field as a direct
    angular_spectrum_propagate call to bit-equality.

    Catches any numerical drift in the MHS subdomain wrapper around
    the underlying ASM kernel.
    """
    N = 32; dx = 5e-6; lam = 633e-9
    z = 1e-3
    s0 = HuygensSurface(z=0.0, Ny=N, Nx=N, dx=dx, label='in')
    s1 = HuygensSurface(z=z, Ny=N, Nx=N, dx=dx, label='out')
    pipe = MhsPipeline([asm_subdomain(s0, s1, wavelength=lam)])
    x = (np.arange(N) - N/2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E_in = np.exp(-(X*X + Y*Y) / (50e-6)**2).astype(np.complex128)
    E_mhs = pipe.run(E_in, return_intermediate=False)
    E_direct = la.angular_spectrum_propagate(E_in, z, lam, dx)
    rel = float(np.max(np.abs(E_mhs - E_direct))) / max(
        float(np.max(np.abs(E_direct))), 1e-30)
    return rel < 1e-12, f'rel diff = {rel:.2e}'


def t_mhs_chain_power_monotone_through_aperture():
    """Power should be conserved through ASM legs (no aperture) and
    only decrease through the aperture subdomain.  Catches bugs where
    a propagator silently amplifies / loses energy.
    """
    N = 64; dx = 5e-6; lam = 633e-9
    aperture_radius = 80e-6
    s0 = HuygensSurface(z=0.0, Ny=N, Nx=N, dx=dx, label='in')
    s1 = HuygensSurface(z=1e-3, Ny=N, Nx=N, dx=dx, label='ap')
    s2 = HuygensSurface(z=2e-3, Ny=N, Nx=N, dx=dx, label='out')
    sub_a = asm_subdomain(s0, s1, wavelength=lam)
    sub_ap = aperture_subdomain(s1, aperture_radius=aperture_radius)
    sub_b = asm_subdomain(s1, s2, wavelength=lam)
    pipe = MhsPipeline([sub_a, sub_ap, sub_b])
    x = (np.arange(N) - N/2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    R = np.sqrt(X*X + Y*Y)
    E_in = (R < 200e-6).astype(np.complex128)

    history = pipe.run(E_in)
    powers = [float(np.sum(np.abs(h[1]) ** 2)) for h in history]
    p_in, p_pre_ap, p_post_ap, p_out = powers
    # ASM is unitary: input == pre-aperture (within FFT roundoff).
    asm_rel = abs(p_pre_ap - p_in) / p_in
    # Aperture must reduce power (or leave it equal).
    aperture_drop = p_post_ap <= p_pre_ap + 1e-9
    # Final ASM is again unitary: post-aperture == output.
    final_rel = abs(p_out - p_post_ap) / max(p_post_ap, 1e-30)
    ok = asm_rel < 1e-6 and aperture_drop and final_rel < 1e-6
    return ok, (
        f'P trace: in={p_in:.3e} pre_ap={p_pre_ap:.3e} '
        f'post_ap={p_post_ap:.3e} out={p_out:.3e} '
        f'(asm_rel={asm_rel:.2e}, drop={aperture_drop}, '
        f'final_rel={final_rel:.2e})')


def t_pipeline_from_prescription_builds_3_subdomains():
    """MhsPipeline.from_prescription with non-zero pre/post creates ASM->lens->ASM."""
    presc = la.make_singlet(R1=5.16e-3, R2=float('inf'), d=2e-3,
                             glass='N-BK7', aperture=4e-3)
    pipe = MhsPipeline.from_prescription(
        presc, wavelength=633e-9, dx=5e-6, Ny=16, Nx=16,
        pre_distance=1e-3, post_distance=1e-3, method='gbd',
        sample_step=4, chunk_beamlets=128,
    )
    n_ok = pipe.n_subdomains == 3
    labels = [s.label for s in pipe.subdomains]
    asm_count = sum(1 for L in labels if L == 'asm')
    pres_count = sum(1 for L in labels if L.startswith('prescription'))
    return (n_ok and asm_count == 2 and pres_count == 1), (
        f'n={pipe.n_subdomains}, labels={labels}')


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
    H.run('checkpoint callback fires per subdomain',
          t_pipeline_checkpoint_callback)
    H.run('from_prescription builds ASM->prescription->ASM chain',
          t_pipeline_from_prescription_builds_3_subdomains)

    H.section('MHS physics equivalence')
    H.run('MhsPipeline single ASM matches direct ASM to 1e-12',
          t_mhs_single_asm_matches_direct_asm)
    H.run('ASM->aperture->ASM chain power monotone through aperture',
          t_mhs_chain_power_monotone_through_aperture)

    H.section('Subdomain builders')
    H.run('aperture_subdomain clips correctly',
          t_aperture_subdomain_clips)

    sys.exit(H.summary())


if __name__ == '__main__':
    main()

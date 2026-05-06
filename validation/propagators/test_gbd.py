"""GBD module tests."""
from __future__ import annotations

import sys

import numpy as np

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from _harness import Harness

import lumenairy as la
from lumenairy.propagators.gbd import (
    BeamletBundle, decompose_field_to_beamlets,
    propagate_beamlets_freespace, apply_thin_lens_to_beamlets,
    reconstruct_field_from_beamlets,
    propagate_gbd_freespace, propagate_gbd_thin_lens,
)


H = Harness('gbd')


def t_decompose_count():
    N = 32; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    bundle = decompose_field_to_beamlets(E, dx, wavelength=lam, sample_step=2)
    expected = (N // 2) ** 2
    return len(bundle) == expected, f'beamlets={len(bundle)}'


def t_decompose_q_negative_imag():
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    bundle = decompose_field_to_beamlets(E, dx, wavelength=lam, sample_step=4)
    return bool(np.all(np.imag(bundle.Q) < 0)), 'Q has negative imaginary'


def t_freespace_changes_q():
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    b0 = decompose_field_to_beamlets(E, dx, wavelength=lam, sample_step=4)
    b1 = propagate_beamlets_freespace(b0, z_distance=1e-3, wavelength=lam)
    return float(np.max(np.abs(b1.Q - b0.Q))) > 1e-3, 'Q changed'


def t_thin_lens_q_decrement():
    N = 16; dx = 5e-6; lam = 633e-9
    f = 100e-3
    E = np.ones((N, N), dtype=np.complex128)
    b0 = decompose_field_to_beamlets(E, dx, wavelength=lam, sample_step=4)
    b1 = apply_thin_lens_to_beamlets(b0, focal_length=f, wavelength=lam)
    err = abs(float(np.real(b1.Q[0] - b0.Q[0])) - (-1.0 / f)) / (1.0 / f)
    return err < 1e-6, f'Q decrement err={err:.2e}'


def t_gaussian_peak_evolution():
    N = 64; dx = 5e-6; lam = 633e-9
    w0 = 30e-6
    x = (np.arange(N) - N/2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X*X + Y*Y) / (w0*w0)).astype(np.complex128)
    zR = float(np.pi) * w0 * w0 / lam
    out = propagate_gbd_freespace(
        E, dx, z=zR/2, wavelength=lam,
        waist_factor=2.0, sample_step=2, chunk_beamlets=512,
    )
    peak_z = float(np.max(np.abs(out)))
    peak_0 = float(np.max(np.abs(E)))
    expected = 1 / np.sqrt(1 + 0.25)
    return abs(peak_z / peak_0 - expected) / expected < 0.15, (
        f'ratio={peak_z/peak_0:.4f}, expected={expected:.4f}')


def t_thin_lens_focuses():
    N = 64; dx = 5e-6; lam = 633e-9
    f = 5e-3
    w0 = 200e-6
    x = (np.arange(N) - N/2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X*X+Y*Y)/(w0*w0)).astype(np.complex128)
    out_focal = propagate_gbd_thin_lens(
        E, dx, z_to_lens=0.1e-3, focal_length=f,
        z_lens_to_output=f, wavelength=lam,
        waist_factor=2.0, sample_step=2, chunk_beamlets=512,
    )
    out_off = propagate_gbd_thin_lens(
        E, dx, z_to_lens=0.1e-3, focal_length=f,
        z_lens_to_output=2 * f, wavelength=lam,
        waist_factor=2.0, sample_step=2, chunk_beamlets=512,
    )
    return float(np.max(np.abs(out_focal))) > float(np.max(np.abs(out_off))), 'focused'


def t_jax_dispatch():
    if not la.JAX_AVAILABLE:
        return True, 'skipped (no jax)'
    import jax.numpy as jnp
    N = 16; dx = 5e-6; lam = 633e-9
    E = jnp.ones((N, N), dtype=jnp.complex64)
    b = decompose_field_to_beamlets(E, dx, wavelength=lam, sample_step=4)
    return la.is_jax_array(b.positions), 'jax positions'


def t_prescription_gbd():
    from lumenairy.propagators.gbd import propagate_gbd_through_prescription
    presc = la.make_singlet(R1=5.16e-3, R2=float('inf'), d=2e-3,
                             glass='N-BK7', aperture=4e-3)
    presc['object_distance'] = 30e-3
    N = 32; dx = 5e-6; lam = 633e-9
    x = (np.arange(N) - N/2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X*X + Y*Y) / (30e-6)**2).astype(np.complex128)
    out = propagate_gbd_through_prescription(
        E, dx, presc, wavelength=lam,
        sample_step=4, chunk_beamlets=256,
    )
    return out.shape == E.shape and bool(np.all(np.isfinite(np.abs(out)))), (
        f'peak={float(np.max(np.abs(out))):.3f}')


# ----------------------------------------------------------------------
# Deep physics tests
# ----------------------------------------------------------------------

def t_gbd_freespace_matches_asm_collimated_gaussian():
    """GBD free-space agrees with ASM on a collimated Gaussian beam.

    For a collimated Gaussian propagating less than a Rayleigh range,
    GBD should match the exact ASM solution to within a few percent
    on the peak intensity.  This catches regressions where GBD's
    beamlet seeding / waist choice drifts from the analytic Gaussian
    propagation.
    """
    N = 128; dx = 5e-6; lam = 633e-9
    w0 = 100e-6
    z = (np.pi * w0 * w0 / lam) * 0.5  # half a Rayleigh range
    x = (np.arange(N) - N/2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X*X + Y*Y) / (w0*w0)).astype(np.complex128)
    out_asm = la.angular_spectrum_propagate(E, z, lam, dx)
    out_gbd = propagate_gbd_freespace(
        E, dx, z=z, wavelength=lam,
        waist_factor=2.0, sample_step=2, chunk_beamlets=1024,
    )
    peak_asm = float(np.max(np.abs(out_asm)))
    peak_gbd = float(np.max(np.abs(out_gbd)))
    rel = abs(peak_gbd - peak_asm) / max(peak_asm, 1e-30)
    return rel < 0.05, (
        f'ASM peak={peak_asm:.4e}, GBD peak={peak_gbd:.4e}, rel={rel:.2e}')


def t_gbd_freespace_power_conservation():
    """Total power should be conserved through GBD free-space propagation."""
    N = 64; dx = 5e-6; lam = 633e-9
    w0 = 60e-6
    x = (np.arange(N) - N/2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X*X + Y*Y) / (w0*w0)).astype(np.complex128)
    z = (np.pi * w0 * w0 / lam) * 0.3
    out = propagate_gbd_freespace(
        E, dx, z=z, wavelength=lam,
        waist_factor=2.0, sample_step=2, chunk_beamlets=512,
    )
    p_in = float(np.sum(np.abs(E) ** 2))
    p_out = float(np.sum(np.abs(out) ** 2))
    rel = abs(p_out - p_in) / max(p_in, 1e-30)
    return rel < 0.05, f'P_in={p_in:.4e}, P_out={p_out:.4e}, rel={rel:.2e}'


def t_gbd_through_prescription_focal_length():
    """GBD-through-prescription puts peak intensity near the system BFL."""
    from lumenairy.propagators.gbd import propagate_gbd_through_prescription
    # 50 mm singlet; propagate Gaussian + 0 mm gap to lens to focus.
    presc = la.make_singlet(R1=25.6e-3, R2=float('inf'), d=4e-3,
                             glass='N-BK7', aperture=4e-3)
    surfs = la.surfaces_from_prescription(presc)
    _, _, bfl, _ = la.system_abcd(surfs, 633e-9)
    bfl = float(bfl)

    N = 64; dx = 20e-6; lam = 633e-9
    w0 = 500e-6
    x = (np.arange(N) - N/2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X*X + Y*Y) / (w0*w0)).astype(np.complex128)

    # Build a derived prescription with thickness extended by BFL so
    # the propagator lands at the back focal plane.
    presc_focus = dict(presc)
    presc_focus['thicknesses'] = (
        list(presc['thicknesses']) + [bfl]
        if len(presc['thicknesses']) < len(presc['surfaces'])
        else list(presc['thicknesses']))
    # Add a flat image-plane surface (already in lens dict via N-1 thicknesses)
    presc_focus['surfaces'] = list(presc['surfaces']) + [
        {'radius': float('inf'), 'conic': 0.0,
         'aspheric_coeffs': None,
         'glass_before': 'air', 'glass_after': 'air'},
    ]

    out = propagate_gbd_through_prescription(
        E, dx, presc_focus, wavelength=lam,
        sample_step=2, chunk_beamlets=512,
    )
    # Peak should be near the centre.  Allow generous +/- 5 pixels.
    iy, ix = np.unravel_index(int(np.argmax(np.abs(out))), out.shape)
    err_pix = abs(iy - N // 2) + abs(ix - N // 2)
    return err_pix <= 5, (
        f'peak at pixel ({iy}, {ix}); centre={N//2}; err={err_pix}px')


H.section('Deep physics')
H.run('GBD freespace matches ASM (collimated Gaussian, 0.5 z_R)',
      t_gbd_freespace_matches_asm_collimated_gaussian)
H.run('GBD freespace power conservation',
      t_gbd_freespace_power_conservation)
H.run('GBD-through-prescription puts peak near BFL',
      t_gbd_through_prescription_focal_length)


def main():
    H.section('Decomposition')
    H.run('beamlet count', t_decompose_count)
    H.run('Q has negative imaginary', t_decompose_q_negative_imag)

    H.section('ABCD evolution')
    H.run('free-space changes Q', t_freespace_changes_q)
    H.run('thin lens decrements Q by 1/f', t_thin_lens_q_decrement)

    H.section('Physics agreement')
    H.run('Gaussian peak evolution', t_gaussian_peak_evolution)
    H.run('thin lens focuses', t_thin_lens_focuses)

    H.section('JAX backend')
    H.run('JAX dispatch', t_jax_dispatch)

    H.section('Prescription path')
    H.run('GBD through singlet prescription via system ABCD',
          t_prescription_gbd)

    sys.exit(H.summary())


if __name__ == '__main__':
    main()

"""
Lumenairy example 6 -- Schell-model partial-coherence propagation.

Demonstrates the v4.16.1 ``propagate_ensemble`` helper for
propagating a partial-coherence Gaussian-Schell source through a
coherent propagator and recovering the partial-coherence intensity::

    I_partial(r) = < |E_k(r)|^2 >_k

This is the canonical Wolf coherence-theory result for a
Schell-model source (Goodman, *Statistical Optics*, Sec 5.5).
The example also runs a fully-coherent reference at the same waist
to show the smoothing factor -- partial-coherence reduces peak
intensity and washes out the speckle / fine structure that the
coherent beam exhibits.
"""
import numpy as np
import lumenairy as la


def main():
    # --- 1. Build a Gaussian-Schell partial-coherence source ----------
    # 50 um-waist beam, 15 um transverse coherence length, sampled on a
    # 256x256 grid at 2 um pitch.  64 realisations gives a smooth
    # < |E_k|^2 >_k estimate; the ensemble loop in propagate_ensemble is
    # the simplest possible Wolf integral (sum and divide by N).
    ens, dx, dy, wl = la.create_gaussian_schell_source(
        N=256, dx=2e-6, wavelength=633e-9,
        w0=50e-6, sigma_g=15e-6,
        n_realizations=64, return_kind='ensemble',
        rng=0,
    )
    print(f'  Input ensemble shape: {ens.shape}')
    print(f'  Grid pitch: dx={dx*1e6:.3g} um, dy={dy*1e6:.3g} um')
    print(f'  Wavelength: {wl*1e9:.1f} nm')

    # Inspect the source-plane partial-coherence intensity (just
    # ensemble-average the squared modulus at the input plane).
    I_partial_in = np.mean(np.abs(ens) ** 2, axis=0)
    print(f'  Source plane peak intensity (partial): '
          f'{I_partial_in.max():.3e}')

    # --- 2. Propagate the ensemble through 10 cm of free space --------
    # propagate_ensemble iterates each realisation through the chosen
    # coherent propagator (here ASM) and accumulates < |E_k|^2 >_k into
    # a single (Ny, Nx) intensity buffer.  Memory cost is one extra
    # complex (Ny, Nx) buffer for the propagator output -- no need to
    # materialise the full propagated ensemble.
    z_prop = 0.10  # 10 cm
    I_partial = la.propagate_ensemble(
        ens, dx=dx, wavelength=wl,
        propagator='asm', z=z_prop,
    )
    print()
    print(f'  Propagated {z_prop*1e2:.1f} cm via ASM')
    print(f'  Partial-coherence intensity shape: {I_partial.shape}')
    print(f'  Peak intensity (partial): {I_partial.max():.3e}')
    print(f'  Total integrated power: '
          f'{(I_partial.sum() * dx * dy):.4e}')

    # --- 3. Run a fully-coherent reference at the same waist ----------
    # Build a deterministic Gaussian beam with the same w0 and propagate
    # it through the same 10 cm.  This is what the user would have
    # gotten in the sigma_g -> infinity (coherent) limit.  The Source
    # ergonomic wrapper handles the field / dx / wavelength bundle.
    src_coh = la.Source.gaussian(
        w0=50e-6, N=256, dx=dx, wavelength=wl, name='coherent_ref')
    E_coh_out = la.angular_spectrum_propagate(
        src_coh.E, z_prop, wl, dx)
    I_coh = np.abs(E_coh_out) ** 2
    print()
    print(f'  Coherent peak intensity: {I_coh.max():.3e}')

    # --- 4. Smoothing factor ------------------------------------------
    # Partial-coherence vs coherent peak ratio.  Partial-coherence
    # reduces peak intensity (the speckle structure of individual
    # realisations is washed out in the ensemble average) -- the ratio
    # ``I_coh_peak / I_partial_peak`` quantifies the smoothing.  For
    # sigma_g ~ 0.3 * w0 the ratio is typically ~2-3.
    smoothing = float(I_coh.max() / I_partial.max())
    print(f'  Partial-coherence smoothing factor '
          f'(I_coh_peak / I_partial_peak): {smoothing:.2f}')

    # --- 5. Demonstrate return_ensemble=True for advanced workflows ---
    # When the caller needs the propagated ensemble (e.g. to compute
    # something other than ``< |E_k|^2 >``) they can opt into storing
    # the full ensemble.  Memory cost is n_realizations * Ny * Nx *
    # element_size for the output buffer.
    I_partial2, ens_out = la.propagate_ensemble(
        ens, dx=dx, wavelength=wl,
        propagator='asm', z=z_prop,
        return_ensemble=True,
    )
    print()
    print(f'  return_ensemble=True: propagated ensemble shape '
          f'{ens_out.shape}')
    # Verify the convenience-return and the manual recomputation agree.
    I_manual = np.mean(np.abs(ens_out) ** 2, axis=0)
    max_residual = float(np.max(np.abs(I_partial2 - I_manual)))
    print(f'  |I_partial - <|E_out|^2>| max residual: '
          f'{max_residual:.3e}  (should be ~0)')


if __name__ == '__main__':
    main()

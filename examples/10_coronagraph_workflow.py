"""
Lumenairy example 10 -- coronagraph workflow (4-stop chain).

Demonstrates the canonical classical Lyot coronagraph pipeline:

    entrance pupil  ->  focal-plane occulter  ->  Lyot stop  ->  image

For the on-axis stellar PSF the focal-plane mask blocks the core,
the Lyot stop in the downstream pupil truncates the diffraction
ring outside the geometric pupil image, and the final image has
several orders of magnitude lower on-axis intensity than the
unmasked reference.

This example uses ``lumenairy.compute_psf`` for the Fraunhofer
forward/backward transforms and the ``apply_lyot_focal_plane_mask``
+ ``apply_lyot_stop`` element helpers; the final contrast curve is
reduced via ``lumenairy.coronagraph_contrast_curve``.

Run::

    python examples/10_coronagraph_workflow.py

A contrast-vs-angular-separation curve is written to
``examples/output/10_coronagraph_workflow.png``.

Author: Andrew Traverso -- v5.2 / examples roadmap.
"""
from __future__ import annotations

import os as _os

import numpy as np

import lumenairy as la


def main():
    # --- 1. Build a clear circular pupil ------------------------------
    # 256x256 pupil on a 50 um pitch -- the focal-plane oversampling is
    # automatic (compute_psf returns dx_focal = lambda * f / (N * dx)).
    N = 256
    wavelength = 1.55e-6
    f_eff = 200e-3              # 200-mm focal length collimator
    D = 8e-3                    # 8-mm clear aperture
    dx_pupil = 50e-6
    x = (np.arange(N) - N / 2) * dx_pupil
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X * X + Y * Y)
    pupil_amp = np.where(R <= D / 2, 1.0, 0.0).astype(np.complex128)

    # --- 2. Forward Fraunhofer:  pupil -> focal plane -----------------
    # compute_psf returns the focal-plane INTENSITY, but for the
    # coronagraph chain we need the complex focal-plane FIELD so we
    # can mask it then back-propagate.  Build that field directly via
    # an FFT (same convention as compute_psf).
    pupil_padded = np.pad(pupil_amp, ((0, 0), (0, 0)), mode='constant')
    E_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_padded)))
    dx_focal = wavelength * f_eff / (N * dx_pupil)
    print(f'  Focal-plane sampling: {dx_focal*1e6:.3f} um  '
          f'(lambda*f/D = {wavelength*f_eff/D*1e6:.2f} um)')

    # Reference (un-coronagraphed) PSF intensity for the contrast
    # denominator.  Same field, just |.|^2 with no mask.
    psf_ref = np.abs(E_focal) ** 2

    # --- 3. Focal-plane occulter --------------------------------------
    # 4 lambda*f/D-diameter hard-edge Lyot mask centred on chief.
    # Smooth-edge masks (gaussian / sin2) reduce downstream Lyot-stop
    # ringing -- use 'gaussian' for the cleaner contrast curve here.
    mask_diameter = 4.0 * wavelength * f_eff / D
    E_focal_masked = la.apply_lyot_focal_plane_mask(
        E_focal, dx=dx_focal, mask_diameter=mask_diameter,
        profile='gaussian')

    # --- 4. Back to pupil plane via inverse Fraunhofer ----------------
    # The Fourier inverse of the masked focal-plane field is the
    # second-pupil-plane field (the relay pupil where the Lyot stop
    # lives).
    E_pupil2 = np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(E_focal_masked)))

    # --- 5. Lyot stop -------------------------------------------------
    # Classic 0.85 * D-diameter clear hard-edge stop in the relay pupil.
    lyot_diam = 0.85 * D
    E_pupil2_stopped = la.apply_lyot_stop(
        E_pupil2, dx=dx_pupil, outer_diameter=lyot_diam)

    # --- 6. Forward Fraunhofer to final image plane -------------------
    # Same FFT convention as step 2.  The result is the coronagraphed
    # complex focal-plane field; |.|^2 is the coronagraphed PSF.
    E_final = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
        E_pupil2_stopped)))
    psf_coro = np.abs(E_final) ** 2

    # --- 7. Contrast curve reduction ----------------------------------
    # coronagraph_contrast_curve does the azimuthal-mean radial
    # reduction and returns the contrast in units of the reference
    # peak.  Pass pupil_diameter_m so it can set the absolute
    # lambda*f/D scale correctly.
    curve = la.coronagraph_contrast_curve(
        psf_coro, psf_ref, dx_focal=dx_focal,
        wavelength=wavelength, f_eff=f_eff,
        pupil_diameter_m=D, n_radii=48, max_lam_over_D=12.0,
    )
    r_lod = curve['r_lam_over_D']
    contrast = curve['contrast']

    # Contrast at exactly 5 lambda/D (linear interpolation for the
    # readable summary line).
    contrast_at_5 = float(np.interp(5.0, r_lod, contrast))

    # --- 8. Plot the contrast curve -----------------------------------
    out_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            'output')
    _os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available; skipping contrast figure.')
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(r_lod, contrast, '-o', markersize=3)
        ax.axvline(5.0, color='red', linestyle='--',
                   label=f'5 lambda/D ({contrast_at_5:.2e})')
        ax.set_xlabel('Angular separation [lambda * f / D]')
        ax.set_ylabel('Contrast (azimuthal mean)')
        ax.set_title('10_coronagraph_workflow.py -- Lyot contrast curve')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()
        plt.tight_layout()
        out_path = _os.path.join(out_dir, '10_coronagraph_workflow.png')
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f'  Figure saved to {out_path}')

    # --- 9. Summary --------------------------------------------------
    print()
    print(f'  Lyot coronagraph (mask diameter = 4 lambda*f/D, '
          f'Lyot stop = 0.85*D):')
    print(f'    Reference peak intensity:  {psf_ref.max():.3e}')
    print(f'    Coronagraphed peak:        {psf_coro.max():.3e}')
    print(f'    Contrast at 5 lambda/D:    {contrast_at_5:.3e}')
    print(f'    Min contrast in [3, 10]:   '
          f'{contrast[(r_lod > 3) & (r_lod < 10)].min():.3e}')


if __name__ == '__main__':
    main()

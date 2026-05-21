"""
Lumenairy example 12 -- ghost / stray-light analysis of a doublet.

Demonstrates the v5.2 ghost-path enumeration + analysis API:

  * ``lumenairy.enumerate_ghost_paths`` -- list all 2-bounce paths
  * ``lumenairy.ghost_analysis``        -- per-path Fresnel R + heuristic
                                            focus position
  * Combined with a nominal ``compute_psf`` for the dominant ghost's
    relative-intensity comparison.

The example builds a cemented N-BK7 / SF2 achromatic doublet (an
AC254-100 form factor), enumerates all 2-bounce ghost-reflection
paths between surface pairs, and reports the brightest ghost's
intensity relative to the nominal on-axis PSF.

Run::

    python examples/12_ghost_stray_light.py

A 2-panel figure (nominal PSF on the left, top-3 ghost intensities
on the right) is written to
``examples/output/12_ghost_stray_light.png``.

Author: Andrew Traverso -- v5.2 / examples roadmap.
"""
from __future__ import annotations

import os as _os

import numpy as np

import lumenairy as la


def main():
    # --- 1. Build a cemented achromatic doublet -----------------------
    # Approximate AC254-100-A: N-BK7 / SF2 cemented doublet, ~100 mm
    # EFL, 25.4 mm clear aperture.
    wavelength = 587.56e-9   # d-line
    aperture = 25.4e-3
    prescription = la.make_doublet(
        R1=62.0e-3, R2=-43.0e-3, R3=-127.7e-3,
        d1=4.0e-3, d2=2.5e-3,
        glass1='N-BK7', glass2='N-SF2',
        aperture=aperture,
        name='AC254-100-like doublet',
    )
    n_surfaces = len(prescription['surfaces'])
    print(f'  Prescription: {prescription["name"]} '
          f'({n_surfaces} refracting surfaces)')

    # --- 2. Enumerate the 2-bounce ghost paths ------------------------
    # For a 3-surface doublet, there are C(3, 2) = 3 unique 2-bounce
    # paths.  enumerate_ghost_paths gives them in (i, j) order with
    # i < j.  ghost_analysis traces each and reports per-path Fresnel
    # reflectances + intensity upper-bound.
    paths = la.enumerate_ghost_paths(n_surfaces)
    print(f'  2-bounce ghost-path families: {len(paths)}')
    for p in paths:
        print(f'    surfaces {p}')

    # --- 3. Run ghost_analysis ---------------------------------------
    # Returns a list of dicts, one per path, sorted by intensity
    # (brightest first).  intensity = R_i * R_j (Fresnel only) -- an
    # upper bound; see the function docstring for the
    # transmission-loss correction.
    ghosts = la.ghost_analysis(
        prescription, wavelength=wavelength,
        semi_aperture=aperture / 2, n_rays=21,
        verbose=False,
    )
    print()
    print(f'  Ghost-path report (intensity upper bound, Fresnel only):')
    print(f'    {"path":>8}   {"R_i":>8}  {"R_j":>8}  {"intensity":>12}  '
          f'{"f_ghost [mm]":>14}')
    for g in ghosts:
        print(f'    {str(g["path"]):>8}   {g["R_i"]:8.4f}  {g["R_j"]:8.4f}  '
              f'{g["intensity"]:12.3e}  {g["focus_z_estimate"]*1e3:14.2f}')

    # --- 4. Compute the nominal on-axis PSF for context ---------------
    # Build a circular clear pupil on a 128 grid; pass it through
    # compute_psf for the nominal Airy PSF.  The dominant-ghost
    # intensity is reported as a ratio against this PSF peak below.
    N = 128
    dx_pupil = aperture / 96
    x = (np.arange(N) - N / 2) * dx_pupil
    X, Y = np.meshgrid(x, x)
    pupil = np.where(np.sqrt(X * X + Y * Y) <= aperture / 2,
                     1.0, 0.0).astype(np.complex128)
    f_eff = 100e-3
    psf_nominal, dx_psf = la.compute_psf(
        pupil, wavelength=wavelength, f=f_eff,
        dx_pupil=dx_pupil, normalize='power',
    )
    nominal_peak = float(psf_nominal.max())

    # Dominant ghost.
    dom = ghosts[0]
    ghost_to_signal = dom['intensity']  # R_i * R_j vs unit input flux
    # As a fraction of the PSF peak: under power normalisation the PSF
    # peak is the system's coherent answer, so the ratio
    # ``R_i * R_j`` directly gives the ghost-to-signal flux ratio at
    # the detector (modulo the transmission-loss factor through the
    # non-bouncing surfaces, which for a 3-surface doublet is ~0.96).
    print()
    print(f'  Dominant ghost surfaces: {dom["path"]}')
    print(f'    R_i = {dom["R_i"]:.4f}, R_j = {dom["R_j"]:.4f}')
    print(f'    Ghost-to-signal flux ratio (Fresnel UB): '
          f'{ghost_to_signal:.3e}')

    # --- 5. Plot nominal PSF (log) + ghost bar chart ------------------
    out_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            'output')
    _os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available; skipping ghost figure.')
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        # PSF on log scale (centred 48x48 crop for visibility).
        crop = 48
        c0 = N // 2 - crop // 2
        c1 = c0 + crop
        spot = psf_nominal[c0:c1, c0:c1]
        im = ax1.imshow(np.log10(spot + spot.max() * 1e-10),
                        cmap='inferno', origin='lower')
        ax1.set_title(f'Nominal PSF (log10, peak={nominal_peak:.2e})')
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(im, ax=ax1, fraction=0.046)
        # Bar chart of all ghost intensities.
        labels = [f'{g["path"]}' for g in ghosts]
        intensities = [g['intensity'] for g in ghosts]
        ax2.bar(range(len(ghosts)), intensities, color='C1')
        ax2.set_xticks(range(len(ghosts)))
        ax2.set_xticklabels(labels)
        ax2.set_yscale('log')
        ax2.set_xlabel('Ghost path (surface_i, surface_j)')
        ax2.set_ylabel('Intensity (Fresnel R_i * R_j)')
        ax2.set_title('Ghost intensities (Fresnel UB)')
        ax2.grid(True, axis='y', which='both', alpha=0.3)
        fig.suptitle(
            '12_ghost_stray_light.py -- doublet ghost analysis')
        plt.tight_layout()
        out_path = _os.path.join(out_dir, '12_ghost_stray_light.png')
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f'  Figure saved to {out_path}')

    # --- 6. Summary --------------------------------------------------
    print()
    print(f'  Stray-light summary:')
    print(f'    n_ghost_paths:        {len(ghosts)}')
    print(f'    dominant path:        {dom["path"]}')
    print(f'    dominant intensity:   {ghost_to_signal:.3e}')
    print(f'    total ghost flux (UB): '
          f'{sum(g["intensity"] for g in ghosts):.3e}')


if __name__ == '__main__':
    main()

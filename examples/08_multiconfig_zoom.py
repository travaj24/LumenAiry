"""
Lumenairy example 8 -- multi-configuration / zoom system.

Demonstrates the v5.2 ``lumenairy.optimize.multiconfig`` API by
building a 3-config zoom lens (wide / mid / tele) and evaluating the
focal length at each zoom state via the system ABCD matrix.

The system is a 2-element zoom: a fixed front lens plus a movable
rear element whose air-gap (slot 1 in ``thicknesses``) is varied to
change the system EFL.  At each zoom position the system ABCD is
recomputed and an on-axis PSF is rendered through the canonical
``compute_psf`` pupil-to-image FFT.

Run::

    python examples/08_multiconfig_zoom.py

The three PSF spots are plotted side-by-side and written to
``examples/output/08_multiconfig_zoom.png``.

Author: Andrew Traverso -- v5.2 / examples roadmap.
"""
from __future__ import annotations

import os as _os

import numpy as np

import lumenairy as la
from lumenairy.optimize.multiconfig import (
    Configuration, create_zoom_configs, multi_config_merit,
)
from lumenairy.raytrace import system_abcd_prescription


def _on_axis_psf(prescription, wavelength, N, dx_pupil, semi_ap):
    """Render an on-axis PSF for a prescription using compute_psf.

    Returns a centred-crop intensity PSF + the EFL recovered from the
    system ABCD matrix.  Uses the canonical pupil-amplitude + Fraunhofer
    FFT pipeline -- no propagation through the lens, just the paraxial
    EFL feeding compute_psf.  Adequate for a side-by-side spot diagram.
    """
    M_abcd, efl, _, _ = system_abcd_prescription(prescription, wavelength)
    # Build a clear circular pupil on the requested grid.
    x = (np.arange(N) - N / 2) * dx_pupil
    X, Y = np.meshgrid(x, x)
    pupil = np.where(np.sqrt(X * X + Y * Y) <= semi_ap, 1.0, 0.0).astype(
        np.complex128)
    psf, dx_psf = la.compute_psf(
        pupil, wavelength=wavelength, f=abs(efl),
        dx_pupil=dx_pupil, normalize='power',
    )
    return psf, dx_psf, float(efl)


def main():
    # --- 1. Build a 2-element zoom template ---------------------------
    # Front element: fixed positive doublet-equivalent singlet.  Rear
    # element: positive singlet whose distance to the front element is
    # the zoom variable.  Total length is held roughly constant by
    # adjusting the front-to-rear gap so the back focal distance
    # changes while the image plane stays fixed.
    wavelength = 587.56e-9  # d-line
    aperture = 12e-3

    front = la.make_singlet(R1=60e-3, R2=-60e-3, d=3e-3,
                            glass='N-BK7', aperture=aperture)
    rear = la.make_singlet(R1=80e-3, R2=-80e-3, d=3e-3,
                           glass='N-BK7', aperture=aperture)

    # Manually stitch the two singlets into one prescription with a
    # variable air gap (the zoom degree of freedom).  Each singlet
    # contributes 2 surfaces + 1 thickness; the air gap between them
    # is the second entry in ``thicknesses``.
    template = {
        'name': 'zoom_2elem',
        'aperture_diameter': aperture,
        'object_distance': float('inf'),
        'surfaces': front['surfaces'] + rear['surfaces'],
        'thicknesses': [
            front['thicknesses'][0],   # front element thickness
            20e-3,                     # placeholder zoom air gap
            rear['thicknesses'][0],    # rear element thickness
        ],
    }

    # --- 2. Build the 3 zoom configurations ---------------------------
    # Three discrete zoom positions: wide (short gap), mid, tele
    # (long gap).  create_zoom_configs deep-copies the template and
    # overwrites the thicknesses for each config.
    zoom_spacings = [
        [3e-3, 10e-3, 3e-3],   # wide
        [3e-3, 25e-3, 3e-3],   # mid
        [3e-3, 50e-3, 3e-3],   # tele
    ]
    configs = create_zoom_configs(
        template, zoom_spacings, wavelength=wavelength, field_angle=0.0)
    # Rename for readability in the printout.
    for cfg, label in zip(configs, ('wide', 'mid', 'tele')):
        cfg.name = label

    # --- 3. Verify multi_config_merit wiring with a trivial merit ----
    # Sanity-check the multi-config evaluator: a constant merit per
    # config should give total = sum(weights).
    def _unit_merit(pres, wl, fa):
        return 1.0
    total, per = multi_config_merit(configs, _unit_merit, verbose=False)
    assert abs(total - len(configs)) < 1e-9, (
        f'multi_config_merit wiring sanity check failed: '
        f'total={total} per={per}')

    # --- 4. Render an on-axis PSF for each zoom config ----------------
    N = 128
    dx_pupil = aperture / 96  # ~75% fill across the grid
    psfs = []
    efls = []
    for cfg in configs:
        psf, dx_psf, efl = _on_axis_psf(
            cfg.prescription, cfg.wavelength,
            N=N, dx_pupil=dx_pupil, semi_ap=aperture / 2)
        psfs.append(psf)
        efls.append(efl)
        print(f'  [{cfg.name:>4}]  gap={cfg.prescription["thicknesses"][1]*1e3:5.1f} mm '
              f' EFL={efl*1e3:7.2f} mm   peak_psf={psf.max():.3e}')

    # --- 5. Plot all 3 PSF spots side-by-side -------------------------
    out_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            'output')
    _os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available; skipping PSF figure.')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        # Centred 32x32 crop around the PSF peak for a compact spot view.
        crop = 32
        c0 = N // 2 - crop // 2
        c1 = c0 + crop
        for ax, cfg, psf, efl in zip(axes, configs, psfs, efls):
            spot = psf[c0:c1, c0:c1]
            ax.imshow(spot ** 0.25, cmap='inferno', origin='lower')
            ax.set_title(f'{cfg.name}\nEFL={efl*1e3:.1f} mm')
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle('08_multiconfig_zoom.py -- on-axis PSF vs zoom state')
        plt.tight_layout()
        out_path = _os.path.join(out_dir, '08_multiconfig_zoom.png')
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f'  Figure saved to {out_path}')

    # --- 6. Summary --------------------------------------------------
    print()
    print(f'  Zoom states evaluated: {len(configs)}')
    print(f'  EFL range:             {min(efls)*1e3:.2f} mm '
          f'-> {max(efls)*1e3:.2f} mm '
          f'(zoom ratio {max(efls)/min(efls):.2f}x)')
    print(f'  Peak PSF amplitudes:   '
          f'{[f"{p.max():.2e}" for p in psfs]}')


if __name__ == '__main__':
    main()

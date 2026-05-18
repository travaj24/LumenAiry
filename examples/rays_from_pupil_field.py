"""Bridge a coherent pupil field into a ray bundle and trace it.

Demonstrates :func:`lumenairy.rays_from_field` -- the v4.15.1 / Cluster
B Item 6 wave-to-ray bridge.  Useful when you have a measured /
simulated exit-pupil field and want to overlay a geometric ray fan on
a defocus stack, seed an HFPI / GBD bundle from the wave-optical
state, or hand the field into the geometric ray tracer for hybrid
analysis.

The example builds a Gaussian source, applies a thin-lens phase to
make the field converging, samples 80 rays from the resulting pupil
field, and prints the median ray tilt and the RMS spot size at the
geometric focus.  A two-panel matplotlib figure is plotted if
matplotlib is available; otherwise the example exits cleanly with the
print outputs only.

Run::

    python examples/rays_from_pupil_field.py

Author: Andrew Traverso -- v4.15.1 / Agent F (Cluster B Item 6).
"""
import numpy as np

import lumenairy as la


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Build a representative converging field.
    # ------------------------------------------------------------------
    N, dx, lam, f = 512, 2e-6, 633e-9, 0.05
    src = la.Source.gaussian(N=N, dx=dx, wavelength=lam, w0=200e-6)
    E_pupil = la.apply_thin_lens(src.E, f=f, wavelength=lam, dx=dx)

    # ------------------------------------------------------------------
    # 2. Sample 80 rays from the pupil intensity / phase.
    # ------------------------------------------------------------------
    rays = la.rays_from_field(
        E_pupil, dx=dx, wavelength=lam,
        n_rays=80, placement='cdf',
        random_state=2026,
    )

    print(f"Sampled {rays.n_rays} rays, {int(rays.alive.sum())} alive "
          f"({int((~rays.alive).sum())} evanescent).")
    print(f"Median tilt: L = {float(np.median(rays.L)):.4f}, "
          f"M = {float(np.median(rays.M)):.4f}")
    print(f"OPD range  : [{float(rays.opd.min()) * 1e6:.3f}, "
          f"{float(rays.opd.max()) * 1e6:.3f}] um")

    # ------------------------------------------------------------------
    # 3. Project rays to the geometric focus and report the spot size.
    # ------------------------------------------------------------------
    alive = rays.alive
    xf = rays.x[alive] + (rays.L[alive] / rays.N[alive]) * f
    yf = rays.y[alive] + (rays.M[alive] / rays.N[alive]) * f
    rms_spot = float(np.sqrt(np.mean(xf * xf + yf * yf)))
    print(f"RMS spot at focus: {rms_spot * 1e6:.3f} um")

    # ------------------------------------------------------------------
    # 4. Visualise (optional -- matplotlib may not be installed).
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: pupil intensity with ray origins overlaid.
    extent = np.array([-N // 2, N // 2, -N // 2, N // 2]) * dx * 1e6
    axes[0].imshow(
        np.abs(E_pupil) ** 2, cmap='magma',
        extent=extent, origin='lower',
    )
    axes[0].scatter(
        rays.x[alive] * 1e6, rays.y[alive] * 1e6,
        s=12, edgecolor='white', facecolor='none', linewidth=0.8,
        label=f"{int(alive.sum())} rays",
    )
    axes[0].set_title('Pupil intensity + ray origins')
    axes[0].set_xlabel('x [um]')
    axes[0].set_ylabel('y [um]')
    axes[0].legend(loc='upper right', fontsize=9)

    # Panel 2: ray landings at z = f, scaled to micrometres.
    axes[1].scatter(xf * 1e6, yf * 1e6, s=18, alpha=0.7)
    axes[1].set_title(f'Ray landings at z = {f * 1e3:.0f} mm '
                       f'(RMS = {rms_spot * 1e6:.2f} um)')
    axes[1].set_xlabel('x [um]')
    axes[1].set_ylabel('y [um]')
    axes[1].set_aspect('equal')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

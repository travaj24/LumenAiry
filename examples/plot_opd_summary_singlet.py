"""Build an N-BK7 singlet and render a 4-panel OPD summary.

Demonstrates :func:`lumenairy.plot_opd_summary` -- the v4.15.4 4-panel
OPD diagnostic: 2-D heatmap, radial-RMS profile, tangential fan,
sagittal fan.  The example uses the ray-traced
:func:`lumenairy.opd_fan_data` for the 1-D fans (chief-ray reference
built in) and :func:`lumenairy.eval_image_plane_wfe` for the 2-D
heatmap, so the figure ties the wave-side and the ray-side OPD
diagnostics together on one canvas.

Run::

    python examples/plot_opd_summary_singlet.py

Produces ``singlet_opd_summary.png`` in the current working directory.

Author: Andrew Traverso -- v4.15.4 / Agent D.
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lumenairy as la


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Build a representative N-BK7 singlet.
    # ------------------------------------------------------------------
    wavelength = 633e-9
    rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=4e-3, glass='N-BK7',
                          aperture=10e-3)
    # eval_image_plane_wfe requires a finite object distance.
    rx['object_distance'] = 1.0  # 1 m object distance
    surfaces = la.surfaces_from_prescription(rx)

    # ------------------------------------------------------------------
    # 2. Compute the 2-D OPD map via image-plane WFE (chief-ray
    #    reference sphere, paraxial focus).  Scatter the per-ray
    #    result back onto a square n_pupil x n_pupil grid -- the WFE
    #    internally clips the square Cartesian grid to the unit disk,
    #    so we rebuild the n_pupil x n_pupil image with NaN at corner
    #    pixels (which then become the aperture mask).
    # ------------------------------------------------------------------
    n_pupil = 33
    wfe = la.eval_image_plane_wfe(
        rx, wavelength,
        n_pupil=n_pupil,
        image_plane='paraxial')
    # Re-derive the square Cartesian grid the WFE internally generates.
    p1 = np.linspace(-1.0, 1.0, n_pupil)
    PX, PY = np.meshgrid(p1, p1)
    aperture = (PX ** 2 + PY ** 2) <= 1.0
    opd_w_2d = np.full((n_pupil, n_pupil), np.nan, dtype=float)
    # Reconstruct the index-of-each-WFE-sample into the square grid.
    idx_flat = np.where(aperture.ravel())[0]
    opd_w_2d.flat[idx_flat] = wfe.opd_w
    opd_m_2d = opd_w_2d * wavelength

    # ------------------------------------------------------------------
    # 3. Compute the 1-D fans via ray trace (chief-ray reference).
    #    opd_fan_data returns WAVES; convert to metres.
    # ------------------------------------------------------------------
    semi_aperture = rx['aperture_diameter'] / 2.0
    py_fan, opd_y_w, px_fan, opd_x_w = la.opd_fan_data(
        surfaces, wavelength, semi_aperture, field_angle=0.0,
        n_rays=101)
    opd_y_m = opd_y_w * wavelength
    opd_x_m = opd_x_w * wavelength

    # Grid spacing on the pupil grid (m).  WFE uses normalised
    # coordinates clipped to the unit disk; physical dx is
    # aperture / (n_pupil - 1).
    dx = (2.0 * semi_aperture) / max(n_pupil - 1, 1)

    # ------------------------------------------------------------------
    # 4. Print PV / RMS, then render and save the 4-panel summary.
    # ------------------------------------------------------------------
    finite = opd_w_2d[np.isfinite(opd_w_2d)]
    pv_w = float(finite.max() - finite.min())
    rms_w = float(np.sqrt(np.mean(finite ** 2)))
    print(f"Singlet N-BK7 R1=+50/R2=-50 mm, d=4 mm, A=10 mm, "
          f"lambda = {wavelength*1e9:.0f} nm")
    print(f"  2-D OPD: PV = {pv_w:.4f} waves, RMS = {rms_w:.4f} waves")

    fig, axes = la.plot_opd_summary(
        opd_m_2d, dx=dx, aperture=aperture,
        py=py_fan, opd_y=opd_y_m,
        px=px_fan, opd_x=opd_x_m,
        units='waves', wavelength=wavelength,
        title='N-BK7 singlet -- 4-panel OPD summary')

    out_path = 'singlet_opd_summary.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


if __name__ == '__main__':
    main()

"""Anamorphic beam-shaping with cylindrical lenses.

Demonstrates the anamorphic / per-axis path through the v4.15.1
operator-algebra layer: a system that focuses in x but stays
collimated in y produces a line focus rather than a point focus.
This is the canonical use case for the :class:`CylindricalLens`
primitive (an :class:`Operator` whose ``abcd_x != abcd_y``).

We build a one-cylindrical-lens system, confirm the system reports
as anamorphic (``sys.is_anamorphic == True``), inspect its per-axis
ABCDs separately, and visualize the line-focus intensity in the
back focal plane.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np

import lumenairy as la


def main() -> None:
    # -----------------------------------------------------------------
    # 1. Build a single-cylindrical-lens system: focus in x at f_x,
    #    no focusing power in y.
    # -----------------------------------------------------------------
    f_x = 100e-3
    cyl = la.CylindricalLens(f_x=f_x, f_y=np.inf)
    sys = la.FreeSpace(f_x, method='asm') * cyl

    print("Anamorphic cylindrical-lens system:")
    print(f"  is_anamorphic = {sys.is_anamorphic}")
    print(f"  abcd_x =\n{sys.abcd_x}")
    print(f"  abcd_y =\n{sys.abcd_y}")
    print(f"  (abcd_y is identity-on-FreeSpace -- no power in y)")
    print()

    # -----------------------------------------------------------------
    # 2. Apply to a circular Gaussian source.  The output should be
    #    a line focus along y (focused in x, un-focused in y).
    # -----------------------------------------------------------------
    src = la.Source.gaussian(
        N=512, dx=4e-6, wavelength=633e-9, w0=600e-6,
    )
    print(f"Input source: {src}")
    out = sys(src)
    print(f"Output: {out}")
    print()

    # -----------------------------------------------------------------
    # 3. Compute marginal sigmas to confirm the anamorphic profile.
    # -----------------------------------------------------------------
    I = np.abs(out.E) ** 2
    ix_marg = I.sum(axis=0)
    iy_marg = I.sum(axis=1)
    x_axis = (np.arange(out.E.shape[1]) - out.E.shape[1] / 2) * out.dx
    y_axis = (np.arange(out.E.shape[0]) - out.E.shape[0] / 2) * out.dx

    cx_x = np.sum(ix_marg * x_axis) / np.sum(ix_marg)
    cx_y = np.sum(iy_marg * y_axis) / np.sum(iy_marg)
    sigma_x = np.sqrt(
        np.sum(ix_marg * (x_axis - cx_x) ** 2) / np.sum(ix_marg)
    )
    sigma_y = np.sqrt(
        np.sum(iy_marg * (y_axis - cx_y) ** 2) / np.sum(iy_marg)
    )

    print("Intensity marginal sigmas at the back focal plane:")
    print(f"  sigma_x (focused axis)   = {sigma_x * 1e6:8.2f} um")
    print(f"  sigma_y (unfocused axis) = {sigma_y * 1e6:8.2f} um")
    print(f"  asymmetry ratio sigma_x/sigma_y = "
          f"{sigma_x / sigma_y:.3f}  (< 1 -> focused in x)")
    print()

    # -----------------------------------------------------------------
    # 4. Visualization (optional).
    # -----------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(np.abs(src.E) ** 2, cmap='hot', origin='lower',
                    extent=(
                        -src.E.shape[1] / 2 * src.dx * 1e3,
                        src.E.shape[1] / 2 * src.dx * 1e3,
                        -src.E.shape[0] / 2 * src.dx * 1e3,
                        src.E.shape[0] / 2 * src.dx * 1e3,
                    ))
    axes[0].set_title('Input Gaussian')
    axes[0].set_xlabel('x [mm]')
    axes[0].set_ylabel('y [mm]')
    axes[1].imshow(np.abs(out.E) ** 2, cmap='hot', origin='lower',
                    extent=(
                        -out.E.shape[1] / 2 * out.dx * 1e3,
                        out.E.shape[1] / 2 * out.dx * 1e3,
                        -out.E.shape[0] / 2 * out.dx * 1e3,
                        out.E.shape[0] / 2 * out.dx * 1e3,
                    ))
    axes[1].set_title('Back focal plane (line focus)')
    axes[1].set_xlabel('x [mm]')
    axes[1].set_ylabel('y [mm]')
    fig.suptitle('algebra_anamorphic.py -- cylindrical line focus')
    plt.tight_layout()
    # v4.16.1 (audit item 21): write to examples/output/ alongside the
    # other example PNGs rather than the current working directory.
    import os as _os
    out_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            'output')
    _os.makedirs(out_dir, exist_ok=True)
    out_path = _os.path.join(out_dir, 'algebra_anamorphic.png')
    fig.savefig(out_path, dpi=120)
    print(f"Figure saved to {out_path}")


if __name__ == '__main__':
    main()

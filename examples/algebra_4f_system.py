"""4f imaging system constructed via operator algebra.

Demonstrates the v4.15.1 operator-algebra layer
(``lumenairy.algebra``) by building a 4f imaging system
symbolically, inspecting its ABCD matrix without applying it to
any field, and then propagating a coherent Gaussian source through
the same composite operator.

The 4f system [FreeSpace(f) -> ThinLens(f) -> FreeSpace(2f) ->
ThinLens(f) -> FreeSpace(f)] has the closed-form ABCD
``[[-1, 0], [0, -1]]`` -- an inverter with unit magnification.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np

import lumenairy as la


def main() -> None:
    # -----------------------------------------------------------------
    # 1. Build the 4f system algebraically.
    # -----------------------------------------------------------------
    f = 100e-3  # 100 mm focal length lenses
    sys = (
        la.FreeSpace(f) * la.ThinLens(f) * la.FreeSpace(2 * f)
        * la.ThinLens(f) * la.FreeSpace(f)
    )

    print("4f imaging system constructed via operator algebra:")
    print(f"  sys.abcd =\n{sys.abcd}")
    print(f"  sys.efl  = {sys.efl} m  (inf == afocal, as expected)")
    print(f"  sys.A    = {sys.A}  (== -1 means unit magnification "
          f"with inversion)")
    print(f"  stages   = {len(sys.stages())} operators")
    print()

    # -----------------------------------------------------------------
    # 2. Apply to a coherent Gaussian.
    # -----------------------------------------------------------------
    src = la.Source.gaussian(
        N=512, dx=2e-6, wavelength=633e-9, w0=80e-6,
    )
    print(f"Source: {src}")
    out = sys(src)
    print(f"Output: {out}")
    print()

    # -----------------------------------------------------------------
    # 3. Measure the centroid shift -- a Gaussian centered at +x0
    #    should image to -x0 through the 4f inverter.
    # -----------------------------------------------------------------
    x0 = 200e-6
    src_off = la.Source.gaussian(
        N=512, dx=4e-6, wavelength=633e-9, w0=200e-6, x0=x0,
    )
    sys_off = (
        la.FreeSpace(50e-3, method='asm') * la.ThinLens(50e-3)
        * la.FreeSpace(100e-3, method='asm') * la.ThinLens(50e-3)
        * la.FreeSpace(50e-3, method='asm')
    )
    out_off = sys_off(src_off)
    I = np.abs(out_off.E) ** 2
    ix = np.arange(out_off.E.shape[1])
    marginal_x = I.sum(axis=0)
    centroid_idx = np.sum(ix * marginal_x) / np.sum(marginal_x)
    centroid_x = (centroid_idx - out_off.E.shape[1] / 2) * out_off.dx
    print(f"4f inverter check:")
    print(f"  input centroid x:  {x0 * 1e6:+.2f} um")
    print(f"  output centroid x: {centroid_x * 1e6:+.2f} um")
    print(f"  (4f maps +x0 -> -x0, so they should be opposite signs)")
    print()

    # -----------------------------------------------------------------
    # 4. Visualization (optional -- skipped if matplotlib is missing).
    # -----------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(np.abs(src.E) ** 2, cmap='hot', origin='lower')
    axes[0].set_title('Input Gaussian')
    axes[1].imshow(np.abs(out.E) ** 2, cmap='hot', origin='lower')
    axes[1].set_title('After 4f system (inverted)')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('algebra_4f_system.py -- operator-algebra 4f imaging')
    plt.tight_layout()
    # v4.16.1 (audit item 21): write to examples/output/ alongside the
    # other example PNGs rather than the current working directory.
    import os as _os
    out_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            'output')
    _os.makedirs(out_dir, exist_ok=True)
    out_path = _os.path.join(out_dir, 'algebra_4f_system.png')
    fig.savefig(out_path, dpi=120)
    print(f"Figure saved to {out_path}")


if __name__ == '__main__':
    main()

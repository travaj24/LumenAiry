"""
Lumenairy example 1 -- basic propagation.

Build a Gaussian source, propagate through free space, hit a thin
lens, propagate to focus.  Demonstrates the core
"build a source, propagate, analyze" workflow with the simplest
possible system.
"""
import numpy as np
import lumenairy as la


def main():
    # --- 1. Build a source --------------------------------------------
    # 200 um-waist Gaussian on a 256x256 grid sampled at 4 um (1.024 mm
    # half-extent -- comfortably wider than the beam at the lens).
    src = la.Source.gaussian(
        w0=200e-6, N=256, dx=4e-6, wavelength=1.31e-6,
        name='input',
    )
    print(f'  Input source: {src}')
    print(f'  Initial peak |E|^2 = '
          f'{float(np.max(np.abs(src.E)**2)):.4e}')

    # --- 2. Propagate to a thin lens (10 mm gap) ----------------------
    after_gap = src.propagate(method='asm', z=10e-3)
    print(f'  After 10 mm free space: {after_gap}')

    # --- 3. Apply a thin lens (50 mm focal length) --------------------
    f_lens = 50e-3
    E_after_lens = la.apply_thin_lens(
        after_gap.E, f=f_lens, wavelength=after_gap.wavelength, dx=after_gap.dx)
    print(f'  After 50-mm thin lens applied')

    # --- 4. Propagate to focus ----------------------------------------
    E_focus = la.propagate(
        E_after_lens, z=f_lens, wavelength=src.wavelength,
        dx=src.dx, method='asm',
    )
    peak_focus = float(np.max(np.abs(E_focus) ** 2))
    print(f'  Peak |E|^2 at focus: {peak_focus:.4e}')

    # --- 5. Same call with return_result=True -------------------------
    # Demonstrates the opt-in PropagationResult container.
    result = la.propagate(
        E_after_lens, z=f_lens, wavelength=src.wavelength,
        dx=src.dx, method='asm', return_result=True,
    )
    print()
    print(f'  PropagationResult: {result}')
    print(f'    field shape: {result.shape}')
    print(f'    method: {result.method}')
    print(f'    z: {result.z*1e3:.3f} mm')


if __name__ == '__main__':
    main()

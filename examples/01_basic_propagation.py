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
    # v5.30 (audit P5 / roadmap F1): the DEFAULT return is a
    # PropagationResult for every method -- .field / .dx / .dy mean the
    # same thing whichever kernel ran, so nothing here depends on z
    # having selected ASM rather than SAS or Fraunhofer.
    result = la.propagate(
        E_after_lens, z=f_lens, wavelength=src.wavelength,
        dx=src.dx, method='asm',
    )
    peak_focus = float(np.max(np.abs(result.field) ** 2))
    print(f'  Peak |E|^2 at focus: {peak_focus:.4e}')
    print()
    print(f'  PropagationResult: {result}')
    print(f'    field shape: {result.shape}')
    print(f'    method: {result.method}')
    print(f'    z: {result.z*1e3:.3f} mm')
    print(f'    output pitch: {result.dx*1e6:.4f} um')

    # --- 5. The legacy contract, when you want the bare array ---------
    # ``return_result=False`` is a permanent, supported escape hatch: it
    # returns each kernel's native shape (a bare ndarray for ASM; an
    # ``(E, dx_out, dy_out)`` triple for SAS / Fresnel / Fraunhofer).
    # Useful for fast loops that want no wrapper allocation -- and it is
    # the migration path for pre-v5.30 code, bit-for-bit.
    E_focus = la.propagate(
        E_after_lens, z=f_lens, wavelength=src.wavelength,
        dx=src.dx, method='asm', return_result=False,
    )
    print()
    print(f'  return_result=False -> {type(E_focus).__name__}, '
          f'same peak: {float(np.max(np.abs(E_focus) ** 2)):.4e}')


if __name__ == '__main__':
    main()

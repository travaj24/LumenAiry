"""
Lumenairy example 2 -- design optimization.

Optimize the front radius of an N-BK7 singlet against a focal-length
target, with bounds.  Demonstrates the simplest design_optimize call.
"""
import numpy as np
import lumenairy as la


def main():
    # --- 1. Start prescription with one free variable -----------------
    template = la.make_singlet(
        R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
        aperture=12e-3,
    )
    print(f'  Initial template (R1=60 mm, target = 100 mm focal length)')

    # --- 2. Parameterize: free R1 -------------------------------------
    param = la.DesignParameterization(
        template=template,
        free_vars=[('surfaces', 0, 'radius')],
        bounds=[(30e-3, 100e-3)],
    )

    # --- 3. Merits: target focal length + lower-bound on Strehl -------
    merits = [
        la.FocalLengthMerit(target=100e-3, weight=1.0),
        la.StrehlMerit(min_strehl=0.95, weight=10.0),
    ]

    # --- 4. Optimize --------------------------------------------------
    res = la.design_optimize(
        parameterization=param, merit_terms=merits,
        wavelength=1.31e-6,
        N=128, dx=20e-6,
        method='L-BFGS-B', max_iter=40,
        verbose=False,
    )
    print(f'  Optimized R1 = {res.x[0]*1e3:.4f} mm')
    print(f'  Achieved EFL = {res.context_final.efl*1e3:.4f} mm')
    print(f'  Achieved Strehl = {res.context_final.strehl_best:.4f}')
    print(f'  Final merit = {res.merit:.4e}')
    print(f'  Iterations: {res.iterations}, time: {res.time_sec:.1f} s')


if __name__ == '__main__':
    main()

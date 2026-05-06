"""
Lumenairy example 4 -- JAX-differentiable optimization.

Build a JaxMeritTerm with build_args(x), wire it through
design_optimize(jac='auto'), and watch SciPy's L-BFGS-B converge using
analytic JAX gradients (no finite differences).  Falls back gracefully
when JAX isn't installed.
"""
import numpy as np
import lumenairy as la


def main():
    if not la.JAX_AVAILABLE:
        print('  JAX is not installed; skipping example 4.')
        return

    import jax.numpy as jnp

    # --- 1. Toy 1-D problem: minimize |R - 30 mm|, x[0] = R -----------
    # The point isn't the lens optics -- it's that the JAX merit
    # produces analytic gradients that drive SciPy.

    template = la.make_singlet(50e-3, float('inf'), 4e-3, 'N-BK7',
                                aperture=10e-3)
    param = la.DesignParameterization(
        template=template,
        free_vars=[('surfaces', 0, 'radius')],
        bounds=[(20e-3, 70e-3)],
    )

    target = 30e-3

    def fn(R):
        # Real-valued JAX scalar -- straight-line distance to target.
        return jnp.abs((R - target) * 1e3)

    def build_args(x):
        return (x[0],)

    term = la.JaxMeritTerm(
        fn, weight=1.0, build_args=build_args, real_part=True,
        name='|R - 30mm|',
    )

    # --- 2. Run with jac='auto' (the default) -- gradients are analytic
    print(f'  Initial R = {template["surfaces"][0]["radius"]*1e3:.2f} mm')
    print(f'  Target  R = {target*1e3:.2f} mm')
    print()

    res = la.design_optimize(
        parameterization=param, merit_terms=[term],
        wavelength=1.31e-6, N=8, dx=64e-6,
        method='L-BFGS-B', max_iter=50,
        verbose=False, jac='auto',
    )
    print(f'  jac=auto (analytic JAX): R_opt={res.x[0]*1e3:.6f} mm  '
          f'({res.iterations} evals, {res.time_sec:.2f} s)')

    # --- 3. Compare to plain FD ---------------------------------------
    res_fd = la.design_optimize(
        parameterization=param, merit_terms=[term],
        wavelength=1.31e-6, N=8, dx=64e-6,
        method='L-BFGS-B', max_iter=50,
        verbose=False, jac='fd',
    )
    print(f'  jac=fd   (finite diff):  R_opt={res_fd.x[0]*1e3:.6f} mm  '
          f'({res_fd.iterations} evals, {res_fd.time_sec:.2f} s)')

    # The JAX path typically converges in fewer evals because each
    # step gets the exact gradient instead of paying for a 2N-eval
    # FD perturbation.


if __name__ == '__main__':
    main()

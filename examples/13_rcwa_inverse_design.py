"""
Lumenairy example 13 -- RCWA adjoint / gradient-based inverse design.

Use the differentiable RCWA solver to inverse-design a 1-D transmission
grating that steers light into its ``+1`` diffraction order -- a textbook
beam-deflector / metasurface design task.  The gradient of the diffraction
efficiency w.r.t. the continuous design parameters (grating depth, ridge
index) is obtained by reverse-mode automatic differentiation straight through
the rigorous vector Maxwell solve -- including the non-Hermitian
eigendecomposition of each layer (a custom, Lorentzian-broadened eigenvector
VJP) -- so no finite differences and no adjoint bookkeeping are needed:
``jax.grad`` does it all.

The differentiable path is the unified :func:`lumenairy.rcwa_efficiency_1d`
itself: pass ``jax.numpy`` arrays for the design parameters and it
auto-dispatches to the JAX backend (the v5.5.1 fold; the old
``rcwa_efficiency_1d_jax`` is a deprecated alias).

To drive this with the library's optimizer instead of the hand-rolled loop
below, wrap the loss in :class:`lumenairy.optimize.JaxMeritTerm` --
``JaxMeritTerm(fn=loss, build_args=lambda x: (...), real_part=True)`` -- whose
analytic ``jax.grad`` is auto-detected by ``design_optimize``.

Falls back gracefully when JAX isn't installed.
"""
import numpy as np

import lumenairy as la


def main():
    if not la.JAX_AVAILABLE:
        print('  JAX is not installed; skipping example 13.')
        return

    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    wavelength = 0.633e-6
    period = 1.4e-6          # > lambda so the +/-1 orders propagate
    n_substrate = 1.5
    n_superstrate = 1.0
    duty_cycle = 0.5
    angle = 0.0
    order_target = +1        # steer power into +1 transmitted order

    # Design vector x = [depth (um), n_ridge].  We MAXIMISE the +1
    # transmitted diffraction efficiency, i.e. minimise its negative.
    m1 = 11 + order_target   # index of the +1 order in the (-11..11) list

    def neg_efficiency(x):
        depth = x[0] * 1e-6
        n_ridge = x[1]
        # Unified solver, JAX-dispatched because depth / n_ridge are jnp
        # arrays; the geometry (period, wavelength, angle, region indices) is
        # static, so the solve is jax.grad- and jax.jit-able.
        _, R, T = la.rcwa_efficiency_1d(
            period, n_ridge, n_superstrate, n_substrate, n_superstrate,
            depth, duty_cycle, wavelength, angle=angle, polarization='te',
            n_orders=11)
        return -T[m1]

    value_and_grad = jax.jit(jax.value_and_grad(neg_efficiency))

    # --- gradient ascent (Adam-lite: plain step with clipping) ----------
    x = jnp.array([0.30, 1.8])          # initial guess: 0.30 um deep, n=1.8
    lr = jnp.array([0.04, 0.05])
    bounds_lo = jnp.array([0.05, 1.1])
    bounds_hi = jnp.array([1.50, 3.0])

    f0, _ = value_and_grad(x)
    print(f'  start : depth={float(x[0]):.3f} um  n_ridge={float(x[1]):.3f}  '
          f'+1 efficiency = {float(-f0):.4f}')

    for it in range(60):
        f, g = value_and_grad(x)
        x = jnp.clip(x - lr * g, bounds_lo, bounds_hi)
    f_end, _ = value_and_grad(x)
    print(f'  final : depth={float(x[0]):.3f} um  n_ridge={float(x[1]):.3f}  '
          f'+1 efficiency = {float(-f_end):.4f}')

    # --- cross-check the optimum with the (non-differentiable) NumPy core
    orders, R, T = la.rcwa_efficiency_1d(
        period, float(x[1]), n_superstrate, n_substrate, n_superstrate,
        float(x[0]) * 1e-6, duty_cycle, wavelength, angle=angle,
        polarization='te', n_orders=11)
    t_plus1 = T[np.searchsorted(orders, order_target)]
    print(f'  NumPy-core check at the optimum: +1 efficiency = {t_plus1:.4f} '
          f'(energy sum = {R.sum() + T.sum():.6f})')
    print('  Inverse design complete: reverse-mode AD through the rigorous '
          'RCWA solve drove the deflector efficiency up.')


if __name__ == '__main__':
    main()

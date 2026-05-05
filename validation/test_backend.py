"""Multi-backend infrastructure tests."""
from __future__ import annotations

import sys

import numpy as np

from _harness import Harness

import lumenairy as la
from lumenairy.backend import (
    array_namespace, backend_name, is_numpy_array, is_cupy_array, is_jax_array,
    to_numpy, to_backend, CUPY_AVAILABLE, JAX_AVAILABLE,
    fft2, ifft2, fft, ifft, fftshift, ifftshift, fftfreq, fft_backend_for,
    RandomState,
)


H = Harness('backend')


def t_array_namespace_numpy():
    x = np.ones((4, 4))
    xp = array_namespace(x)
    return xp is np, f'name={backend_name(xp)}'


def t_array_namespace_python_scalars():
    xp = array_namespace(1.0, [1, 2, 3], None)
    return xp is np, f'name={backend_name(xp)}'


def t_array_namespace_mixed_raises():
    if not JAX_AVAILABLE:
        return True, 'skipped (no jax)'
    import jax.numpy as jnp
    x_np = np.ones(3); x_jax = jnp.ones(3)
    try:
        array_namespace(x_np, x_jax)
        return False, 'expected TypeError on mixed np+jax'
    except TypeError:
        return True, 'mixed np+jax raises'


def t_predicates_numpy():
    x = np.ones(3)
    return (is_numpy_array(x) and not is_cupy_array(x) and not is_jax_array(x),
            'np.ndarray classified correctly')


def t_fft2_numpy_roundtrip_small():
    x = np.random.default_rng(0).standard_normal((16, 16)) + 0j
    y = fft2(x); xr = ifft2(y)
    err = float(np.max(np.abs(xr - x)))
    return err < 1e-10, f'max err = {err:.2e}'


def t_fft2_numpy_roundtrip_pyfftw_size():
    x = np.random.default_rng(0).standard_normal((512, 512)) + 0j
    backend = fft_backend_for(x)
    y = fft2(x); xr = ifft2(y)
    err = float(np.max(np.abs(xr - x)))
    return err < 1e-9, f'backend={backend}, err={err:.2e}'


def t_fft1d_numpy_roundtrip():
    x = np.arange(64, dtype=np.complex128)
    y = fft(x); xr = ifft(y)
    err = float(np.max(np.abs(xr - x)))
    return err < 1e-10, f'1D max err = {err:.2e}'


def t_fftshift_consistency():
    a = np.arange(8)
    s = fftshift(a)
    expected = np.array([4, 5, 6, 7, 0, 1, 2, 3])
    return bool(np.array_equal(s, expected)), 'matches NumPy'


def t_fft2_jax_roundtrip():
    if not JAX_AVAILABLE:
        return True, 'skipped (no jax)'
    import jax.numpy as jnp
    x = jnp.ones((16, 16), dtype=jnp.complex64)
    y = fft2(x); xr = ifft2(y)
    err = float(jnp.max(jnp.abs(xr - x)))
    return err < 1e-5, f'JAX err = {err:.2e}'


def t_fft2_jax_routes_to_jax():
    if not JAX_AVAILABLE:
        return True, 'skipped (no jax)'
    import jax.numpy as jnp
    x = jnp.ones((4, 4), dtype=jnp.complex64)
    return fft_backend_for(x) == 'jax', 'jax backend selected'


def t_fft2_jax_autodiff_smoke():
    if not JAX_AVAILABLE:
        return True, 'skipped (no jax)'
    import jax
    import jax.numpy as jnp
    def loss(z):
        z_complex = z[0] + 1j * z[1]
        Z = fft2(z_complex.reshape(8, 8))
        return jnp.sum(jnp.abs(Z) ** 2)
    g = jax.grad(loss)(jnp.ones((2, 64)))
    return bool(jnp.all(jnp.isfinite(g))) and g.shape == (2, 64), 'grad ok'


def t_random_state_numpy():
    rs = RandomState(rng=42)
    u = rs.uniform((100,))
    return rs.backend == 'numpy' and u.shape == (100,), 'np ok'


def t_random_state_jax():
    if not JAX_AVAILABLE:
        return True, 'skipped (no jax)'
    import jax
    rs = RandomState(rng=jax.random.PRNGKey(42))
    u = rs.uniform((100,))
    return rs.backend == 'jax' and u.shape == (100,), 'jax ok'


def t_random_state_jax_chained():
    if not JAX_AVAILABLE:
        return True, 'skipped (no jax)'
    import jax
    import jax.numpy as jnp
    rs = RandomState(rng=jax.random.PRNGKey(42))
    a = rs.uniform((10,)); b = rs.uniform((10,))
    return not bool(jnp.allclose(a, b)), 'JAX key advances'


def t_to_numpy_passthrough():
    x = np.arange(5)
    y = to_numpy(x)
    return is_numpy_array(y) and np.array_equal(y, x), 'np->np'


def t_to_numpy_from_jax():
    if not JAX_AVAILABLE:
        return True, 'skipped (no jax)'
    import jax.numpy as jnp
    x = jnp.arange(5)
    y = to_numpy(x)
    return is_numpy_array(y) and np.array_equal(y, np.arange(5)), 'jax->np'


def t_to_backend_jax():
    if not JAX_AVAILABLE:
        return True, 'skipped (no jax)'
    import jax.numpy as jnp
    y = to_backend(np.arange(5), jnp)
    return is_jax_array(y), 'np->jax'


def main():
    H.section('array_namespace dispatch')
    H.run('numpy array detected', t_array_namespace_numpy)
    H.run('python scalars fall back to numpy', t_array_namespace_python_scalars)
    H.run('mixed-backend inputs raise TypeError', t_array_namespace_mixed_raises)
    H.run('predicates classify correctly', t_predicates_numpy)

    H.section('FFT dispatch and round-trips')
    H.run('numpy fft2 small round-trip', t_fft2_numpy_roundtrip_small)
    H.run('numpy fft2 pyFFTW-sized round-trip', t_fft2_numpy_roundtrip_pyfftw_size)
    H.run('numpy 1D fft round-trip', t_fft1d_numpy_roundtrip)
    H.run('fftshift matches NumPy', t_fftshift_consistency)
    H.run('jax fft2 round-trip', t_fft2_jax_roundtrip)
    H.run('jax fft2 routes to jax', t_fft2_jax_routes_to_jax)
    H.run('jax fft2 differentiable via jax.grad', t_fft2_jax_autodiff_smoke)

    H.section('RandomState multi-backend')
    H.run('numpy int seed', t_random_state_numpy)
    H.run('jax PRNGKey detected', t_random_state_jax)
    H.run('jax chained draws differ', t_random_state_jax_chained)

    H.section('Conversion helpers')
    H.run('to_numpy(numpy) identity', t_to_numpy_passthrough)
    H.run('to_numpy(jax) materialises host array', t_to_numpy_from_jax)
    H.run('to_backend(numpy, jax)', t_to_backend_jax)

    sys.exit(H.summary())


if __name__ == '__main__':
    main()

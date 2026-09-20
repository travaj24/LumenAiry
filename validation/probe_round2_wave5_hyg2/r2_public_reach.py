"""Round 2 (V-D3) -- what the PUBLIC ``transport='collins'`` leg does with each
array module, measured on one tree.

The defect this answers is invisible to a value test by construction: on base,
``_collins_carrier_leg`` opened with ``env_a = np.asarray(env)``, so an eager
JAX array was demoted to host NumPy and the answer was BITWISE equal to the
NumPy arm.  What moves is the returned TYPE (and, for CuPy and for a trace, the
exception).  So this probe records types and exception classes, not numbers.

    python r2_public_reach.py <tree> <out.json>
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'probe_wave5_hyg2'))
import hlib                                                  # noqa: E402

import numpy as np                                           # noqa: E402

WL = 633e-9
N = 64
DX = 8e-6
R_IN = -0.05
Z = 5e-3


def _gauss():
    ax = (np.arange(N) - N // 2) * DX
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-(X ** 2 + Y ** 2) / (60e-6) ** 2).astype(np.complex128)


def _describe(fn):
    try:
        v = fn()
    except BaseException as exc:                 # noqa: BLE001 -- recorded
        return {'outcome': 'raised', 'type': type(exc).__name__,
                'msg': str(exc)[:220]}
    return {'outcome': 'ok', 'type': type(v).__name__,
            'module': type(v).__module__.split('.')[0]}


def main(tree, out):
    lum = hlib.anchor(tree)
    import lumenairy.propagators.carrier as CA
    env = _gauss()
    kw = dict(wavelength=WL, dx=DX, transport='collins',
              gap_kernel='fresnel', on_collins_sampling='ignore')
    res = {'version': lum.__version__, 'tree': tree}

    res['numpy'] = _describe(
        lambda: CA.propagate_carrier_referenced(env, R_IN, Z, **kw).env)

    try:
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
    except ImportError:                                      # pragma: no cover
        res['jax'] = {'outcome': 'absent'}
    else:
        res['jax_version'] = jax.__version__
        res['eager_jax'] = _describe(
            lambda: CA.propagate_carrier_referenced(
                jnp.asarray(env, dtype=jnp.complex128), R_IN, Z, **kw).env)

        def _traced():
            def merit(a):
                o = CA.propagate_carrier_referenced(
                    a.astype(jnp.complex128), R_IN, Z, **kw)
                return jnp.sum(jnp.abs(o.env) ** 2)
            return jax.grad(merit)(jnp.asarray(np.real(env)))
        res['traced'] = _describe(_traced)

        # the agreement itself, on whatever the two arms return
        try:
            a = np.asarray(CA.propagate_carrier_referenced(
                env, R_IN, Z, **kw).env)
            b = np.asarray(CA.propagate_carrier_referenced(
                jnp.asarray(env, dtype=jnp.complex128), R_IN, Z, **kw).env)
            res['jax_vs_numpy_rel'] = float(
                np.linalg.norm(b - a) / np.linalg.norm(a))
            res['jax_vs_numpy_bitwise_equal'] = bool(np.array_equal(
                np.ascontiguousarray(a).view(np.float64),
                np.ascontiguousarray(b).view(np.float64)))
        except BaseException as exc:                # noqa: BLE001 -- recorded
            res['jax_vs_numpy_rel'] = f'{type(exc).__name__}: {exc}'[:220]

    try:
        import cupy
    except ImportError:                                      # pragma: no cover
        res['cupy'] = {'outcome': 'absent'}
    else:
        res['cupy_version'] = cupy.__version__
        res['cupy'] = _describe(
            lambda: CA.propagate_carrier_referenced(
                cupy.asarray(env), R_IN, Z, **kw).env)

    hlib.write_json(res, out)
    for k in ('numpy', 'eager_jax', 'traced', 'cupy'):
        if k in res:
            print(f'  {k:10s} {res[k]}', file=sys.stderr)
    print(f"  jax_vs_numpy_rel = {res.get('jax_vs_numpy_rel')!r}",
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2]))

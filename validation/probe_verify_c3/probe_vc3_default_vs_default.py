"""VERIFY-WP-C3 -- the CHANGELOG's strongest compatibility claim:

    "No public call that worked on 5.48.1 raises on 5.49.0, with ONE
     exception, and it is a JAX one."

That is a statement about the DEFAULT, not about the way back, so it cannot be
tested by naming ``transport='sziklas'``.  This probe runs the SAME call
spelled with NO ``transport=`` on both trees -- base 49ddf4bd (where the
default is 'sziklas') and this branch (where it is 'collins') -- and records
per key whether it returned or raised, and with what.

Run with cwd = the tree root, PYTHONPATH = the tree root, VC3_TREE = the tree
root, VC3_OUT = where the JSON goes.  ``lumenairy.__file__`` is asserted under
the tree and printed.
"""
import hashlib
import json
import os
import sys
import traceback
import warnings

import numpy as np

TREE = os.path.abspath(os.environ['VC3_TREE'])
sys.path.insert(0, TREE)
import lumenairy  # noqa: E402
from lumenairy.propagators import carrier as C  # noqa: E402

assert os.path.abspath(lumenairy.__file__).startswith(TREE), (
    lumenairy.__file__, TREE)

LAM = 1.064e-6


def gauss(n, dx, w, dtype=np.complex128):
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing='ij')
    return np.exp(-(xx ** 2 + yy ** 2) / w ** 2).astype(dtype)


def rec(fn):
    """Run ``fn`` and fold what came back -- or what was raised."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        try:
            val = fn()
        except BaseException as exc:                    # noqa: BLE001
            return dict(outcome='raised', exc=type(exc).__name__,
                        msg=str(exc)[:220],
                        where=traceback.extract_tb(
                            exc.__traceback__)[-1].name,
                        warnings=sorted({type(w.message).__name__
                                         for w in wl}))
    arrs = []
    if hasattr(val, 'env'):
        arrs.append(np.asarray(val.env))
    if hasattr(val, 'field') and val.field is not None:
        arrs.append(np.asarray(val.field))
    if isinstance(val, np.ndarray):
        arrs.append(val)
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
        h.update(str(a.dtype).encode())
    finite = all(bool(np.all(np.isfinite(a))) for a in arrs) if arrs else None
    return dict(outcome='returned', sha=h.hexdigest(), n_arrays=len(arrs),
                all_finite=finite,
                dx=repr(getattr(val, 'dx', None))[:60],
                R=repr(getattr(val, 'R', None))[:60],
                warnings=sorted({type(w.message).__name__ for w in wl}))


def cases():
    n, dx, w = 256, 4e-6, 60e-6
    env = gauss(n, dx, w)
    env32 = gauss(n, dx, w, np.complex64)
    big = gauss(512, 4e-6, 0.30e-3)
    P = C.propagate_carrier_referenced
    yield 'leg-diverging', lambda: P(env, -40e-3, -5e-3, LAM, dx)
    yield 'leg-converging', lambda: P(big, -40e-3, 20e-3, LAM, 4e-6)
    yield 'leg-on-focus', lambda: P(big, -40e-3, 40e-3, LAM, 4e-6)
    yield 'leg-just-past', lambda: P(big, -40e-3, 41e-3, LAM, 4e-6)
    yield 'leg-well-past', lambda: P(big, -40e-3, 60e-3, LAM, 4e-6)
    yield 'leg-collimated-inf', lambda: P(env, np.inf, 5e-3, LAM, dx)
    yield 'leg-collimated-neginf', lambda: P(env, -np.inf, 5e-3, LAM, dx)
    yield 'leg-astigmatic', lambda: P(big, (-40e-3, -55e-3), 5e-3, LAM, 4e-6)
    yield 'leg-astig-past', lambda: P(big, (-40e-3, -55e-3), 60e-3, LAM, 4e-6)
    yield 'leg-back-propagating', lambda: P(env, 40e-3, -5e-3, LAM, dx)
    yield 'leg-zero-length', lambda: P(env, -40e-3, 0.0, LAM, dx)
    yield 'leg-complex64', lambda: P(env32, -40e-3, 5e-3, LAM, dx)
    yield 'leg-tilted', lambda: P(env, -40e-3, 5e-3, LAM, dx,
                                  tilt=(0.02, -0.01))
    yield 'leg-kernel-fresnel', lambda: P(big, -40e-3, 20e-3, LAM, 4e-6,
                                          gap_kernel='fresnel')
    yield 'leg-kernel-exact', lambda: P(big, -40e-3, 20e-3, LAM, 4e-6,
                                        gap_kernel='exact')
    yield 'leg-asym-dxdy', lambda: P(env, -40e-3, 5e-3, LAM, dx, 2 * dx)
    yield 'leg-zero-carrier', lambda: P(env, 0.0, 5e-3, LAM, dx)
    yield 'leg-exact-focus-fresnel', lambda: P(big, -40e-3, 40e-3, LAM, 4e-6,
                                               gap_kernel='fresnel')

    def _jax_grad():
        import jax
        import jax.numpy as jnp
        e = jnp.asarray(gauss(128, 4e-6, 60e-6))

        def f(a):
            return jnp.sum(jnp.abs(P(a, -40e-3, 5e-3, LAM, 4e-6).env) ** 2)
        return np.asarray(jax.grad(f)(e))
    yield 'jax-grad-through-leg', _jax_grad

    def _jax_eager():
        import jax.numpy as jnp
        return np.asarray(
            P(jnp.asarray(gauss(128, 4e-6, 60e-6)), -40e-3, 5e-3,
              LAM, 4e-6).env)
    yield 'jax-eager-leg', _jax_eager

    yield 'ro-focus', lambda: C.carrier_referenced_focus_readout(
        big, -40e-3, 40e-3, LAM, 4e-6, dx_out=0.2e-6, N_out=64)
    yield 'ro-focus-standoff', lambda: C.carrier_referenced_focus_readout(
        big, -40e-3, 40e-3, LAM, 4e-6, dx_out=0.2e-6, N_out=64,
        standoff=1e-3)
    yield 'ro-exact-focus', lambda: C.carrier_referenced_exact_focus_readout(
        big, -40e-3, 40e-3, LAM, 4e-6, dx_out=0.2e-6, N_out=64)

    from tests.unit.test_audit2609_b4_collins_transport import (
        _CHAIN_TKW, _chain_fixture)
    cenv, cdx, r_in, groups = _chain_fixture()
    base = dict(r_in=r_in, ray_subsample=16, n_workers=1,
                traced_kwargs=_CHAIN_TKW, final_leg='paraxial')
    K = C.propagate_traced_carrier_chain
    fr = dict(dx_out=0.5e-6, N_out=64)
    yield 'chain-bare-final', lambda: K(cenv, groups, 1.31e-6, cdx,
                                        final_distance=8e-3, **base)
    yield 'chain-readout', lambda: K(cenv, groups, 1.31e-6, cdx,
                                     final_distance=8e-3,
                                     focus_readout=dict(fr), **base)
    yield 'chain-readout-standoff', lambda: K(
        cenv, groups, 1.31e-6, cdx, final_distance=8e-3,
        focus_readout=dict(fr, standoff=2e-3), **base)
    yield 'chain-readout-containment', lambda: K(
        cenv, groups, 1.31e-6, cdx, final_distance=8e-3,
        focus_readout=dict(fr, on_focus_containment='ignore'), **base)
    yield 'chain-readout-bandlimit', lambda: K(
        cenv, groups, 1.31e-6, cdx, final_distance=8e-3,
        focus_readout=dict(fr, bandlimit=False), **base)
    yield 'chain-readout-zero-final', lambda: K(
        cenv, groups, 1.31e-6, cdx, final_distance=0.0,
        focus_readout=dict(fr), **base)
    yield 'chain-final-leg-exact', lambda: K(
        cenv, groups, 1.31e-6, cdx, final_distance=8e-3,
        final_leg='exact', focus_readout=dict(fr),
        **{k: v for k, v in base.items() if k != 'final_leg'})
    yield 'chain-fresnel', lambda: K(cenv, groups, 1.31e-6, cdx,
                                     final_distance=8e-3,
                                     gap_kernel='fresnel', **base)
    M = C.propagate_traced_carrier_chain_multi
    yield 'multi-K1', lambda: M(
        cenv, [dict(groups=groups, weight=1.0)], 1.31e-6, cdx,
        output_grid=dict(dx_out=0.5e-6, N_out=64), final_distance=8e-3,
        **base)
    yield 'multi-K1-standoff', lambda: M(
        cenv, [dict(groups=groups, weight=1.0)], 1.31e-6, cdx,
        output_grid=dict(dx_out=0.5e-6, N_out=64, standoff=2e-3),
        final_distance=8e-3, **base)


def main():
    import inspect
    out = {'tree': TREE, 'lumenairy': lumenairy.__file__,
           'version': lumenairy.__version__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'default_transport': inspect.signature(
               C.propagate_carrier_referenced).parameters['transport'].default,
           'keys': {}}
    for name, fn in cases():
        out['keys'][name] = rec(fn)
    tag = os.environ.get('VC3_TAG', 'x')
    p = os.path.join(os.environ['VC3_OUT'], f'defaults_{tag}.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('DEFAULT =', out['default_transport'], '| keys', len(out['keys']),
          '| raised',
          sum(1 for v in out['keys'].values() if v['outcome'] == 'raised'))
    print('lumenairy', lumenairy.__file__)
    print('WROTE', p)


if __name__ == '__main__':
    main()

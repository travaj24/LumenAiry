"""VERIFY-C1 probe 3 -- the JAX chain: signed zeros, finiteness, jit-vs-eager,
grad, and the validation of the new ``'edge'`` / ``'edge_samples'`` element keys.

Run against BOTH trees (PYTHONPATH selects one) so every "before" number is
measured on the parent commit rather than quoted.

Usage:  python -P validation/probe_verify_c1/probe_jax_v.py <tag>
writes  validation/probe_verify_c1/jax_<tag>.json
"""
import json
import os
import platform
import sys

import numpy as np

import lumenairy

LAM = 1064e-9
DX = 1.25e-6


def _f(N=128, seed=2027):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))


def _neg_zeros(a):
    a = np.asarray(a)
    r, i = np.real(a), np.imag(a)
    return (int(np.count_nonzero((r == 0.0) & np.signbit(r))),
            int(np.count_nonzero((i == 0.0) & np.signbit(i))))


def _np_chain(fn, E, elem):
    out = fn(E, elem, LAM, dx=DX)
    return np.asarray(out[0] if isinstance(out, tuple) else out)


def _sha(a):
    import hashlib
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'run'
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from lumenairy.propagators.system import (
        propagate_through_system, propagate_through_system_jax)

    out = {'tag': tag, 'lumenairy_file': lumenairy.__file__,
           'lumenairy_version': getattr(lumenairy, '__version__', '?'),
           'python': sys.version, 'numpy': np.__version__,
           'jax': jax.__version__, 'platform': platform.platform(),
           'threads': {k: os.environ.get(k) for k in
                       ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'MKL_NUM_THREADS')}}

    elem = [{'type': 'aperture', 'shape': 'circular',
             'params': {'diameter': 9.1e-5}}]
    E = _f()
    Ej = jnp.asarray(E)

    # --- 1. signed zeros on each route, default arm -----------------------
    eager = np.asarray(propagate_through_system_jax(
        Ej, elem, LAM, DX, verbose=True))
    jit_ = np.asarray(propagate_through_system_jax(Ej, elem, LAM, DX))
    npy = _np_chain(propagate_through_system, E, elem)
    out['signed_zeros_default'] = {
        'jax_eager': _neg_zeros(eager), 'jax_jit': _neg_zeros(jit_),
        'numpy': _neg_zeros(npy),
        'n_zero_pixels_eager': int(np.count_nonzero(eager == 0)),
    }
    out['digests_default'] = {'jax_eager': _sha(eager), 'jax_jit': _sha(jit_),
                              'numpy': _sha(npy)}
    out['three_routes_bit_identical_default'] = bool(
        _sha(eager) == _sha(jit_) == _sha(npy))

    # --- 2. the same on the 'hard' arm ------------------------------------
    try:
        elem_h = [dict(elem[0], edge='hard')]
        eager_h = np.asarray(propagate_through_system_jax(
            Ej, elem_h, LAM, DX, verbose=True))
        jit_h = np.asarray(propagate_through_system_jax(Ej, elem_h, LAM, DX))
        npy_h = _np_chain(propagate_through_system, E, elem_h)
        out['signed_zeros_hard'] = {
            'jax_eager': _neg_zeros(eager_h), 'jax_jit': _neg_zeros(jit_h),
            'numpy': _neg_zeros(npy_h)}
        out['digests_hard'] = {'jax_eager': _sha(eager_h),
                               'jax_jit': _sha(jit_h), 'numpy': _sha(npy_h)}
        out['three_routes_bit_identical_hard'] = bool(
            _sha(eager_h) == _sha(jit_h) == _sha(npy_h))
        out['hard_key_honoured'] = bool(_sha(npy_h) != _sha(npy))
    except Exception as e:                            # noqa: BLE001
        out['hard_arm_error'] = '%s: %s' % (type(e).__name__, e)

    # --- 3. a non-finite field OUTSIDE the stop ---------------------------
    En = _f(N=64, seed=29)
    En[0, 0] = np.nan
    En[0, 1] = np.inf
    En[1, 0] = -np.inf
    elem_n = [{'type': 'aperture', 'shape': 'circular',
               'params': {'diameter': 4.1e-5}}]
    res = {}
    with np.errstate(all='ignore'):
        for name, fn in (
            ('jax_eager', lambda: propagate_through_system_jax(
                jnp.asarray(En), elem_n, LAM, DX, verbose=True)),
            ('jax_jit', lambda: propagate_through_system_jax(
                jnp.asarray(En), elem_n, LAM, DX)),
            ('numpy', lambda: _np_chain(propagate_through_system,
                                        En.copy(), elem_n)),
        ):
            try:
                a = np.asarray(fn())
                res[name] = {
                    'corner_00': [float(np.real(a[0, 0])),
                                  float(np.imag(a[0, 0]))],
                    'corner_01': [float(np.real(a[0, 1])),
                                  float(np.imag(a[0, 1]))],
                    'corner_10': [float(np.real(a[1, 0])),
                                  float(np.imag(a[1, 0]))],
                    'n_nonfinite': int(np.count_nonzero(~np.isfinite(a))),
                    'all_finite': bool(np.all(np.isfinite(a))),
                    'neg_zeros': _neg_zeros(a),
                }
            except Exception as e:                    # noqa: BLE001
                res[name] = {'error': '%s: %s' % (type(e).__name__, e)}
    out['nonfinite_outside_stop'] = res

    # --- 4. jax.grad through the mask -------------------------------------
    from lumenairy.elements.elements import apply_aperture
    grad_out = {}
    for arm, kw in (('default', {}), ('gray', {'edge': 'gray'}),
                    ('hard', {'edge': 'hard'})):
        try:
            def loss(amp, kw=kw):
                Eg = jnp.asarray(_f(N=48, seed=5)) * amp
                return jnp.sum(jnp.abs(apply_aperture(
                    Eg, DX, 'circular', {'diameter': 3.1e-5}, **kw)) ** 2)
            g = float(jax.grad(loss)(1.0))
            grad_out[arm] = {'grad': g, 'finite': bool(np.isfinite(g))}
        except Exception as e:                        # noqa: BLE001
            grad_out[arm] = {'error': '%s: %s' % (type(e).__name__, e)}
    try:
        def loss_jit(amp):
            Eg = jnp.asarray(_f(N=48, seed=5)) * amp
            return jnp.sum(jnp.abs(apply_aperture(
                Eg, DX, 'circular', {'diameter': 3.1e-5})) ** 2)
        gj = float(jax.jit(jax.grad(loss_jit))(1.0))
        grad_out['jit_grad'] = {'grad': gj, 'finite': bool(np.isfinite(gj))}
    except Exception as e:                            # noqa: BLE001
        grad_out['jit_grad'] = {'error': '%s: %s' % (type(e).__name__, e)}
    out['jax_grad'] = grad_out

    # --- 5. element-key validation ---------------------------------------
    def _try(fn):
        try:
            a = fn()
            a = a[0] if isinstance(a, tuple) else a
            return {'raised': False, 'digest': _sha(np.asarray(a))}
        except Exception as e:                        # noqa: BLE001
            return {'raised': True,
                    'exc': '%s: %s' % (type(e).__name__, str(e)[:160])}

    keycases = {}
    for label, bad in (
        ('edge_bogus', {'edge': 'soft'}),
        ('edge_none', {'edge': None}),
        ('edge_samples_zero', {'edge_samples': 0}),
        ('edge_samples_negative', {'edge_samples': -2}),
        ('edge_samples_float_2p5', {'edge_samples': 2.5}),
        ('edge_samples_float_4p0', {'edge_samples': 4.0}),
        ('edge_samples_str', {'edge_samples': '4'}),
        ('unknown_key_edgesample_typo', {'edge_sample': 1}),
        ('unknown_key_gray', {'gray': True}),
    ):
        el = [dict(elem[0], **bad)]
        keycases[label] = {
            'numpy': _try(lambda el=el: propagate_through_system(
                E.copy(), el, LAM, dx=DX)),
            'jax_jit': _try(lambda el=el: propagate_through_system_jax(
                Ej, el, LAM, DX)),
            'jax_eager': _try(lambda el=el: propagate_through_system_jax(
                Ej, el, LAM, DX, verbose=True)),
            'apply_aperture_direct': _try(
                lambda bad=bad: apply_aperture(
                    E.copy(), DX, 'circular', {'diameter': 9.1e-5},
                    **{k: v for k, v in bad.items()
                       if k in ('edge', 'edge_samples')})
                if any(k in ('edge', 'edge_samples') for k in bad)
                else None),
        }
        a = keycases[label]
        a['agree_numpy_vs_jit'] = (
            a['numpy'].get('raised') == a['jax_jit'].get('raised')
            and a['numpy'].get('digest') == a['jax_jit'].get('digest'))
        a['agree_numpy_vs_eager'] = (
            a['numpy'].get('raised') == a['jax_eager'].get('raised')
            and a['numpy'].get('digest') == a['jax_eager'].get('digest'))
    out['element_key_validation'] = keycases

    # --- 6. does the element signature separate two rims? -----------------
    try:
        from lumenairy.propagators.system import _system_element_signature
        out['element_signature'] = {
            'default': repr(_system_element_signature(elem[0])),
            'hard': repr(_system_element_signature(dict(elem[0],
                                                        edge='hard'))),
            'gray_ns8': repr(_system_element_signature(
                dict(elem[0], edge='gray', edge_samples=8))),
            'ns_2p5': repr(_system_element_signature(
                dict(elem[0], edge_samples=2.5))),
        }
    except Exception as e:                            # noqa: BLE001
        out['element_signature'] = {'error': str(e)}

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'jax_%s.json' % tag)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print('lumenairy.__file__ =', lumenairy.__file__)
    print('signed zeros default:', out['signed_zeros_default'])
    print('three routes identical (default):',
          out['three_routes_bit_identical_default'])
    print('three routes identical (hard):',
          out.get('three_routes_bit_identical_hard'),
          out.get('hard_arm_error', ''))
    print('nonfinite:', json.dumps(out['nonfinite_outside_stop']))
    print('grad:', json.dumps(out['jax_grad']))
    for k, v in out['element_key_validation'].items():
        print('KEY %-28s numpy=%s jit=%s eager=%s agree(jit)=%s '
              'agree(eager)=%s' % (
                  k, v['numpy'].get('raised'), v['jax_jit'].get('raised'),
                  v['jax_eager'].get('raised'),
                  v['agree_numpy_vs_jit'], v['agree_numpy_vs_eager']))
    print('wrote', path)


if __name__ == '__main__':
    main()

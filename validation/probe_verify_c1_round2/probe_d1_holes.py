"""VERIFY-WP-C1 ROUND 2 -- D1 holes: element shapes the census did not cover.

Round 2's D1 claim is that ``_aperture_edge_kwargs`` is "the ONE place both
backends read the element", so one element dict gets one verdict on all three
chain routes.  Both JAX routes reach that reader only THROUGH
``_resolve_aperture_params``:

* the jit'd route, in ``_system_element_signature``, which returns ``None`` and
  bypasses the kernel cache when the params do not resolve -- BEFORE it calls
  ``_aperture_edge_kwargs``;
* the eager route, whose ``if resolved is not None:`` guard wraps the
  ``_aperture_edge_kwargs`` call.

The NumPy chain has no such guard: it calls ``_aperture_edge_kwargs(elem)``
unconditionally.  This probe measures what the three routes do with an
``'aperture'`` element that carries an ILLEGAL rim keyword and params that do
not resolve, and with a few other unusual-but-legal element shapes.

It also checks the jit kernel CACHE: two elements whose ``edge_samples`` differ
only by a coercion the signature applies (``4`` / ``4.0`` / ``numpy.int64(4)``
/ ``True``) must not be served one kernel for the other's semantics.

Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH=<tree> python probe_d1_holes.py <out.json>
"""
import hashlib
import json
import sys

import numpy as np

import lumenairy  # noqa: F401

WAVELENGTH = 1064e-9
N, DX = 64, 1.25e-6


def _verdict(fn):
    try:
        out = fn()
    except BaseException as e:            # noqa: BLE001
        return [True, "{0}: {1}".format(type(e).__name__, e), None]
    arr = np.asarray(out[0] if isinstance(out, tuple) else out)
    return [False, '', hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()]


def main(out_path):
    import jax

    from lumenairy.propagators.system import propagate_through_system, propagate_through_system_jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    rng = np.random.default_rng(4242)
    E = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))

    def routes(elem):
        got = {
            'numpy_chain': _verdict(
                lambda: propagate_through_system(E, [elem], WAVELENGTH, dx=DX)),
            'jax_eager': _verdict(
                lambda: propagate_through_system_jax(
                    jnp.asarray(E), [elem], WAVELENGTH, DX, verbose=True)),
            'jax_jit': _verdict(
                lambda: propagate_through_system_jax(
                    jnp.asarray(E), [elem], WAVELENGTH, DX)),
        }
        return got

    holes = {}
    cases = [
        # params that do NOT resolve (no 'params' key at all)
        ('no_params_edge_soft',
         {'type': 'aperture', 'shape': 'circular', 'edge': 'soft'}),
        ('no_params_edge_samples_zero',
         {'type': 'aperture', 'shape': 'circular', 'edge_samples': 0}),
        ('no_params_edge_samples_two_point_five',
         {'type': 'aperture', 'shape': 'circular', 'edge_samples': 2.5}),
        ('no_params_edge_none',
         {'type': 'aperture', 'shape': 'circular', 'edge': None}),
        ('empty_params_edge_soft',
         {'type': 'aperture', 'shape': 'circular', 'params': {},
          'edge': 'soft'}),
        ('rect_no_widths_edge_soft',
         {'type': 'aperture', 'shape': 'rectangular', 'params': {},
          'edge': 'soft'}),
        ('diameter_none_edge_soft',
         {'type': 'aperture', 'shape': 'circular',
          'params': {'diameter': None}, 'edge': 'soft'}),
        # control: params DO resolve
        ('resolvable_edge_soft',
         {'type': 'aperture', 'shape': 'circular',
          'params': {'diameter': 5.3e-5}, 'edge': 'soft'}),
        # a legal rim on unresolvable params -- do the routes still agree?
        ('no_params_edge_hard',
         {'type': 'aperture', 'shape': 'circular', 'edge': 'hard'}),
    ]
    for label, elem in cases:
        got = routes(elem)
        holes[label] = {
            'elem': repr(elem),
            'routes': got,
            'routes_identical': len(set(json.dumps(v)
                                        for v in got.values())) == 1,
            'verdicts': dict((k, v[0]) for k, v in got.items()),
        }

    # ---- jit kernel cache: coercion-equivalent spellings ------------------
    # Run them in ONE process, in this order, so a stale kernel would show.
    cache = {}
    spellings = [('int_4', 4), ('float_4p0', 4.0), ('np_int64_4', np.int64(4)),
                 ('int_1', 1), ('bool_True', True), ('np_bool_True', np.True_),
                 ('float_1p0', 1.0)]
    for label, v in spellings:
        elem = {'type': 'aperture', 'shape': 'circular',
                'params': {'diameter': 9.1e-5}, 'edge_samples': v}
        r = {}
        for route, fn in (
                ('numpy_chain',
                 lambda e=elem: propagate_through_system(
                     E, [e], WAVELENGTH, dx=DX)),
                ('jax_jit',
                 lambda e=elem: propagate_through_system_jax(
                     jnp.asarray(E), [e], WAVELENGTH, DX)),
                ('jax_eager',
                 lambda e=elem: propagate_through_system_jax(
                     jnp.asarray(E), [e], WAVELENGTH, DX, verbose=True))):
            r[route] = _verdict(fn)
        from lumenairy.propagators.system import _system_element_signature
        try:
            sig = repr(_system_element_signature(elem))
        except BaseException as e:        # noqa: BLE001
            sig = "{0}: {1}".format(type(e).__name__, e)
        cache[label] = {'value': repr(v), 'signature': sig,
                        'routes': r,
                        'digest': r['numpy_chain'][2],
                        'routes_identical': len(set(
                            json.dumps(x) for x in r.values())) == 1}

    # 4 / 4.0 / int64(4) must be ONE answer; 1 / True / 1.0 another; and the
    # two groups must DIFFER, else the identity is vacuous.
    g4 = set(cache[k]['digest'] for k in ('int_4', 'float_4p0', 'np_int64_4'))
    g1 = set(cache[k]['digest'] for k in ('int_1', 'bool_True', 'np_bool_True',
                                          'float_1p0'))
    cache_verdict = {
        'group_four_is_one_answer': len(g4) == 1,
        'group_one_is_one_answer': len(g1) == 1,
        'groups_differ': bool(g4 and g1 and g4 != g1),
    }

    out = {
        'lumenairy_file': lumenairy.__file__,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'holes': holes,
        'cache': cache,
        'cache_verdict': cache_verdict,
    }
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=repr)
    print('lumenairy:', lumenairy.__file__)
    print('--- unresolvable-params holes ---')
    for k, v in holes.items():
        print("  {0:38s} identical={1!s:5s} verdicts={2}".format(
            k, v['routes_identical'], v['verdicts']))
    print('--- jit cache / coercion groups ---')
    for k, v in cache.items():
        print("  {0:14s} {1:16s} digest={2} identical={3}".format(
            k, v['value'], v['digest'], v['routes_identical']))
    print('  verdict:', cache_verdict)


if __name__ == '__main__':
    main(sys.argv[1])

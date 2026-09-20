"""VERIFY-WP-C1 ROUND 2 -- D1: ONE guard, every route, one verdict AND one message.

Re-measures round 2's claim that ``elements._validate_edge_kwargs`` is the
single refusal for the rim keywords and that all three chain routes (the NumPy
chain, the eager JAX route and the jit'd JAX kernel) now agree.  The census is
the round-2 addendum's own eight illegal dicts PLUS this verification's own
spellings, which round 2 did not measure: ``edge_samples`` of ``-1``, ``True``,
``numpy.int64(4)``, ``numpy.float64(4.0)``, a complex, a list; ``edge`` of
``'Gray'``, ``b'gray'``, ``numpy.str_('gray')``; an element naming
``edge_samples`` but no ``edge``; and an explicit ``edge=None``.

Every row records ``(raised, "Type: message")`` for FOUR entry points -- the
direct ``apply_aperture`` call as well as the three chain routes -- because the
claim is that the chain refuses exactly what the function refuses.

Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH=<tree> python probe_d1_guard_census.py <out.json>
"""
import json
import sys

import numpy as np

import lumenairy  # noqa: F401  (pin the tree)

WAVELENGTH = 1064e-9
N, DX = 64, 1.25e-6
DIAM = 5.3e-5


def _verdict(fn):
    try:
        fn()
    except BaseException as e:            # noqa: BLE001 -- census, not control
        return [True, "{0}: {1}".format(type(e).__name__, e)]
    return [False, '']


def main(out_path):
    from lumenairy.elements.elements import apply_aperture
    from lumenairy.propagators.system import propagate_through_system, propagate_through_system_jax
    try:
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        have_jax = True
    except Exception:                     # noqa: BLE001
        have_jax = False

    rng = np.random.default_rng(313)
    E = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))

    # (label, kwargs) -- kwargs go into the element dict AND into the direct call
    census = [
        # round 2's own eight
        ('edge_unknown_string', {'edge': 'soft'}),
        ('edge_none', {'edge': None}),
        ('edge_empty_string', {'edge': ''}),
        ('edge_samples_zero', {'edge_samples': 0}),
        ('edge_samples_negative_two', {'edge_samples': -2}),
        ('edge_samples_non_integer_float', {'edge_samples': 2.5}),
        ('edge_samples_string', {'edge_samples': '4'}),
        ('edge_samples_bool_false', {'edge_samples': False}),
        # this verification's own
        ('edge_samples_negative_one', {'edge_samples': -1}),
        ('edge_samples_bool_true', {'edge_samples': True}),
        ('edge_samples_numpy_int64', {'edge_samples': np.int64(4)}),
        ('edge_samples_numpy_float64', {'edge_samples': np.float64(4.0)}),
        ('edge_samples_numpy_bool_true', {'edge_samples': np.True_}),
        ('edge_samples_complex', {'edge_samples': 4 + 0j}),
        ('edge_samples_list', {'edge_samples': [4]}),
        ('edge_samples_none', {'edge_samples': None}),
        ('edge_capital_gray', {'edge': 'Gray'}),
        ('edge_bytes_gray', {'edge': b'gray'}),
        ('edge_numpy_str_gray', {'edge': np.str_('gray')}),
        ('edge_samples_without_edge', {'edge_samples': 2}),
        ('edge_hard_no_samples', {'edge': 'hard'}),
        ('edge_gray_no_samples', {'edge': 'gray'}),
        ('edge_samples_integral_float', {'edge_samples': 4.0}),
        ('edge_hard_and_samples', {'edge': 'hard', 'edge_samples': 3}),
    ]

    rows = {}
    for label, kw in census:
        elem = [dict({'type': 'aperture', 'shape': 'circular',
                      'params': {'diameter': DIAM}}, **kw)]
        got = {
            'apply_aperture': _verdict(
                lambda kw=kw: apply_aperture(
                    E, DX, shape='circular', params={'diameter': DIAM}, **kw)),
            'numpy_chain': _verdict(
                lambda elem=elem: propagate_through_system(
                    E, elem, WAVELENGTH, dx=DX)),
        }
        if have_jax:
            got['jax_eager'] = _verdict(
                lambda elem=elem: propagate_through_system_jax(
                    jnp.asarray(E), elem, WAVELENGTH, DX, verbose=True))
            got['jax_jit'] = _verdict(
                lambda elem=elem: propagate_through_system_jax(
                    jnp.asarray(E), elem, WAVELENGTH, DX))
        chain = dict((k, v) for k, v in got.items() if k != 'apply_aperture')
        rows[label] = {
            'kwargs': repr(kw),
            'routes': got,
            'chain_routes_identical': len(set(
                json.dumps(v) for v in chain.values())) == 1,
            'chain_matches_apply_aperture': len(set(
                json.dumps(v) for v in got.values())) == 1,
            'raised': got['numpy_chain'][0],
        }

    # The _EDGE_UNSET sentinel: omitting the key is NOT naming it None.
    from lumenairy.elements import elements as el_mod
    if not hasattr(el_mod, '_validate_edge_kwargs'):
        # PRE tree (7ea01ede / 49ddf4bd): the guard does not exist yet.
        sentinel = {'guard_absent': True,
                    'apply_aperture_defaults': [
                        el_mod.apply_aperture.__defaults__[-2],
                        el_mod.apply_aperture.__defaults__[-1]]}
        el_mod._validate_edge_kwargs = None
    else:
        sentinel = _sentinel_block(el_mod, _verdict)

    import pathlib
    root = pathlib.Path(el_mod.__file__).parent.parent
    call_sites = []
    for p in sorted(root.rglob('*.py')):
        txt = p.read_text(encoding='utf-8', errors='replace')
        for i, line in enumerate(txt.splitlines(), 1):
            if ('_validate_edge_kwargs' in line
                    and not line.lstrip().startswith(('#', '*', ':'))
                    and 'def _validate_edge_kwargs' not in line):
                call_sites.append("{0}:{1}: {2}".format(
                    p.relative_to(root.parent).as_posix(), i, line.strip()))

    out = {
        'lumenairy_file': lumenairy.__file__,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'have_jax': have_jax,
        'rows': rows,
        'sentinel': sentinel,
        'guard_call_sites': call_sites,
    }
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=repr)
    n_split = sum(1 for r in rows.values() if not r['chain_routes_identical'])
    n_vs_fn = sum(1 for r in rows.values()
                  if not r['chain_matches_apply_aperture'])
    print('lumenairy:', lumenairy.__file__)
    print("rows={0}  chain-routes-split={1}  chain-vs-apply_aperture-split={2}"
          .format(len(rows), n_split, n_vs_fn))
    for label, r in rows.items():
        print("  {0:34s} raised={1:5s} routes_identical={2:5s} matches_fn={3:5s}"
              .format(label, str(r['raised']), str(r['chain_routes_identical']),
                      str(r['chain_matches_apply_aperture'])))
    print('guard call sites:', len(call_sites))
    for c in call_sites:
        print('   ', c)
    print('sentinel:', json.dumps(sentinel, default=repr))


def _sentinel_block(el_mod, _verdict):
    return {
        'omitted_both': _verdict(lambda: el_mod._validate_edge_kwargs()),
        'edge_named_none': _verdict(
            lambda: el_mod._validate_edge_kwargs(edge=None)),
        'edge_samples_named_none': _verdict(
            lambda: el_mod._validate_edge_kwargs(edge_samples=None)),
        'returns_none_when_unset': el_mod._validate_edge_kwargs() is None,
        'returns_int_when_set': el_mod._validate_edge_kwargs(edge_samples=4),
        'sentinel_is_module_level': hasattr(el_mod, '_EDGE_UNSET'),
        'apply_aperture_defaults': [
            el_mod.apply_aperture.__defaults__[-2],
            el_mod.apply_aperture.__defaults__[-1]],
    }


if __name__ == '__main__':
    main(sys.argv[1])

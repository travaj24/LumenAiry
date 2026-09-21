"""VERIFY-WP-C3 CLAIM 8d -- the three censuses, re-implemented independently.

    python v_census.py <tree> <out.json>

Census 1 (xp-parametrisation) and census 3 (the FFT pair) are re-derived and
compared to the shipped classification.  Census 2 is re-implemented TWICE: once
as the shipped test spells it (``np.<normaliser>(<one of seven names>)`` inside
FIVE hand-listed functions) and once WIDENED -- every alias of numpy, the full
module name, any local, and rooted at the PUBLIC entry points rather than at a
hand-written list.  The difference between the two readings is the gap.
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import vlib  # noqa: E402
import numpy as np  # noqa: E402

import lumenairy.propagators.carrier as CA  # noqa: E402

vlib.anchor(_TREE)

SHIPPED_XP = {
    '_collins_transport': ('env',), '_collins_carrier_leg': ('env',),
    '_collins_focus_readout': ('env',), '_collins_input_box': ('env',),
    '_collins_exact_kernel_correction': ('xp', 'is_jax', 'bld'),
    '_collins_axis_chirp': ('bld',),
    '_tf_phase_to_H': ('xp', 'is_jax', 'bld'),
    '_exact_dispersion_phase': ('bld',), '_fft2_pair': ('xp', 'is_jax'),
    '_as_c_order': ('xp',), '_to_dev': ('xp', 'is_jax'),
}
SHIPPED_HOST = (
    '_collins_power_marginals', '_collins_containment_radius',
    '_collins_space_support', '_collins_angle_support',
    '_collins_envelope_half_angle', '_collins_sampling_stats',
    '_collins_kernel_wrap_ratio', '_collins_exact_kernel_departure',
    '_collins_envelope_abcd', '_collins_leg_output_axis',
    '_collins_readout_k1', '_check_collins_sampling', '_check_transport',
    '_publish_readout_route')
SHIPPED_DEMOTION_SITES = (
    '_collins_transport', '_collins_carrier_leg', '_collins_focus_readout',
    '_collins_input_box', '_collins_exact_kernel_correction')
FIELD_NAMES = ('env', 'env_a', 'E_env', 'spectrum', 'g', 'G', 'E_out')
NORMALISERS = ('asarray', 'ascontiguousarray')


def module_functions():
    src = pathlib.Path(CA.__file__).read_text(encoding='cp1252')
    tree = ast.parse(src)
    return src, {n.name: n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)}


def call_graph(funcs, roots):
    seen, stack = set(), list(roots)
    while stack:
        nm = stack.pop()
        if nm in seen or nm not in funcs:
            continue
        seen.add(nm)
        for node in ast.walk(funcs[nm]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in funcs:
                    stack.append(node.func.id)
    return seen


def numpy_aliases(src):
    """Every module-level name bound to numpy in this file."""
    tree = ast.parse(src)
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == 'numpy':
                    names.add(a.asname or 'numpy')
    return names


def demotions_shipped(node):
    hits = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Call) or not n.args:
            continue
        f = n.func
        if not (isinstance(f, ast.Attribute) and f.attr in NORMALISERS
                and isinstance(f.value, ast.Name) and f.value.id == 'np'):
            continue
        a0 = n.args[0]
        if isinstance(a0, ast.Name) and a0.id in FIELD_NAMES:
            hits.append('line %d: np.%s(%s)' % (n.lineno, f.attr, a0.id))
    return hits


def demotions_wide(node, aliases):
    """ANY numpy alias, ANY of the normalisers, ANY Name argument (not just
    the seven blessed spellings), plus the local-alias assignment pattern."""
    local = set(aliases)
    for n in ast.walk(node):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) \
                and n.value.id in local:
            for t in n.targets:
                if isinstance(t, ast.Name):
                    local.add(t.id)
    hits = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Call) or not n.args:
            continue
        f = n.func
        base = None
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            base = f.value.id
            attr = f.attr
        elif isinstance(f, ast.Attribute) and isinstance(f.value,
                                                         ast.Attribute):
            base, attr = None, f.attr
        else:
            continue
        if base not in local or attr not in NORMALISERS:
            continue
        a0 = n.args[0]
        nm = a0.id if isinstance(a0, ast.Name) else ast.dump(a0)[:40]
        hits.append('line %d: %s.%s(%s)' % (n.lineno, base, attr, nm))
    return hits


def main():
    out = sys.argv[2]
    src, funcs = module_functions()
    aliases = numpy_aliases(src)
    rec = {'build': vlib.build_tag(), 'tree': _TREE,
           'carrier_file': CA.__file__, 'numpy_aliases': sorted(aliases)}

    # ---- census 1 -------------------------------------------------------
    reached = call_graph(funcs, ('_collins_transport', '_collins_carrier_leg',
                                 '_collins_focus_readout'))
    known = set(SHIPPED_XP) | set(SHIPPED_HOST)
    rec['census1'] = {
        'n_reached': len(reached),
        'reached_sorted': sorted(reached),
        'reached_collins_prefixed': sorted(n for n in reached
                                           if n.startswith('_collins')),
        'unclassified_collins_prefixed': sorted(
            n for n in reached if n.startswith('_collins')
            and n not in known),
        # what the shipped gate does NOT look at: reached helpers whose name
        # is not _collins*
        'reached_not_collins_prefixed': sorted(
            n for n in reached if not n.startswith('_collins')),
        'unclassified_any_prefix': sorted(n for n in reached
                                          if n not in known),
        'host_side_entries': len(SHIPPED_HOST),
        'xp_parametrised_entries': len(SHIPPED_XP),
    }
    missing = []
    for nm, owed in SHIPPED_XP.items():
        if nm not in funcs:
            missing.append('%s: not defined' % nm)
            continue
        have = {a.arg for a in funcs[nm].args.args}
        have |= {a.arg for a in funcs[nm].args.kwonlyargs}
        for p in owed:
            if p not in have:
                missing.append('%s: lost %r' % (nm, p))
    rec['census1']['parameter_check_failures'] = missing
    rec['census1']['host_side_entries_actually_reached'] = sorted(
        n for n in SHIPPED_HOST if n in reached)
    rec['census1']['host_side_entries_NOT_reached'] = sorted(
        n for n in SHIPPED_HOST if n not in reached)

    # ---- census 2 -------------------------------------------------------
    shipped_hits = {nm: demotions_shipped(funcs[nm])
                    for nm in SHIPPED_DEMOTION_SITES}
    # the WIDE reading, over every module-level function in the file
    wide_all = {}
    for nm, node in funcs.items():
        h = demotions_wide(node, aliases)
        if h:
            wide_all[nm] = h
    # ... and over the public chain/leg entry points' OWN call graphs
    public_roots = ('propagate_traced_carrier_chain',
                    'propagate_carrier_referenced',
                    'propagate_traced_carrier_chain_multi')
    public_reached = call_graph(funcs, [r for r in public_roots
                                        if r in funcs])
    wide_public = {nm: wide_all[nm] for nm in sorted(wide_all)
                   if nm in public_reached}
    rec['census2'] = {
        'shipped_sites': list(SHIPPED_DEMOTION_SITES),
        'shipped_hits': {k: v for k, v in shipped_hits.items() if v},
        'shipped_reading': 'clean' if not any(shipped_hits.values())
                           else 'DIRTY',
        'wide_hits_anywhere_in_module': wide_all,
        'public_entry_call_graph_size': len(public_reached),
        'wide_hits_on_public_call_graph': wide_public,
        'collins_call_graph_functions_NOT_in_shipped_sites': sorted(
            n for n in reached if n not in SHIPPED_DEMOTION_SITES),
    }

    # ---- census 3 -------------------------------------------------------
    from lumenairy.backend import fft2 as backend_fft2
    from lumenairy.propagators import fft_infra
    f, i = CA._fft2_pair(np, False)
    n, dx = 64, 8e-6
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    E = np.exp(-(X ** 2 + Y ** 2) / (60e-6 ** 2)).astype(np.complex128)
    a = np.ascontiguousarray(f(np.ascontiguousarray(E, dtype=np.complex128)))
    b = np.ascontiguousarray(backend_fft2(np.ascontiguousarray(
        E, dtype=np.complex128)))
    disp = pathlib.Path(fft_infra.__file__).read_text(encoding='cp1252')
    heads = {}
    for fn in ('def _fft2(x):', 'def _ifft2(x):'):
        head = disp.split(fn, 1)[1][:2000]
        heads[fn] = {'has_is_cupy_array': '_is_cupy_array(x)' in head,
                     'first_120': head[:120].replace('\n', ' | ')}
    rec['census3'] = {
        'fft2_is_fft_infra_fft2': f is fft_infra._fft2,
        'ifft2_is_fft_infra_ifft2': i is fft_infra._ifft2,
        'bitwise_equal_to_backend_fft2': bool(np.array_equal(
            a.view(np.float64), b.view(np.float64))),
        'dispatcher_heads': heads,
        'jax_pair_is_jnp': None,
    }
    try:
        import jax.numpy as jnp
        fj, ij = CA._fft2_pair(jnp, True)
        rec['census3']['jax_pair_is_jnp'] = (fj is jnp.fft.fft2
                                             and ij is jnp.fft.ifft2)
    except Exception as exc:                          # noqa: BLE001
        rec['census3']['jax_pair_is_jnp'] = 'jax unavailable: %s' % exc

    vlib.write_json(rec, out)
    c1, c2 = rec['census1'], rec['census2']
    print('[census1] reached=%d  unclassified(_collins*)=%s  '
          'unclassified(any)=%s  param-failures=%s'
          % (c1['n_reached'], c1['unclassified_collins_prefixed'],
             c1['unclassified_any_prefix'],
             c1['parameter_check_failures']))
    print('[census1] reached but NOT _collins-prefixed (never checked): %s'
          % c1['reached_not_collins_prefixed'])
    print('[census1] host-side entries allow-listed but NOT on the walked '
          'graph: %s' % c1['host_side_entries_NOT_reached'])
    print('[census2] shipped reading = %s' % c2['shipped_reading'])
    print('[census2] WIDE hits anywhere in carrier.py: %s'
          % {k: v for k, v in c2['wide_hits_anywhere_in_module'].items()})
    print('[census2] WIDE hits on the PUBLIC entry call graph: %s'
          % c2['wide_hits_on_public_call_graph'])
    print('[census3] %s' % {k: v for k, v in rec['census3'].items()
                            if k != 'dispatcher_heads'})


if __name__ == '__main__':
    main()

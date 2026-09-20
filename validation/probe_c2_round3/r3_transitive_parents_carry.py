"""WP-C2 ROUND 3, VR2-D6 -- how many of the 46 transitive callers resolve
to a parent that CARRIES both way-back keywords?

The WP-C2 report's D4-transitive section says "35 of 46 resolve to a parent
that now carries both keywords".  35 is the number that RESOLVE to a named
parent at all; the parent that four of them resolve to is ``trace_jax``,
which has neither switch by design.

This probe re-reads the committed
``validation/probe_c2_round2/r2_transitive_parents_*.json``, asks
``inspect.signature`` of every named parent on the RUNNING build, and
splits the 35 into those that reach a way back and those that do not.

Usage:  python r3_transitive_parents_carry.py <parents.json> <out.json>
"""
import importlib
import inspect
import json
import pathlib
import sys

WAY_BACK = ('renormalize', 'sphere_normal')

MODULES = ('lumenairy', 'lumenairy.raytrace', 'lumenairy.analysis',
           'lumenairy.io', 'lumenairy.optimize', 'lumenairy.elements',
           'lumenairy.propagators')


def resolve(name):
    for mod in MODULES:
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        fn = getattr(m, name, None)
        if fn is not None and (inspect.isfunction(fn)
                               or inspect.isbuiltin(fn)):
            return fn
    # private parents live inside their own module; find them by walking
    # the loaded package
    for mn, m in list(sys.modules.items()):
        if not mn.startswith('lumenairy') or m is None:
            continue
        fn = getattr(m, name, None)
        if fn is not None and inspect.isfunction(fn):
            return fn
    return None


def main(parents_path, out_path):
    import lumenairy as la
    for mod in MODULES:
        try:
            importlib.import_module(mod)
        except Exception:
            pass

    d = json.loads(pathlib.Path(parents_path).read_text(encoding='utf-8'))
    tp = d['transitive_parents']
    carry, jax_only, unknown = {}, {}, {}
    for caller, entry in sorted(tp.items()):
        parent = entry.get('parent')
        if not parent:
            continue
        fn = resolve(parent)
        if fn is None:
            unknown[caller] = parent
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            unknown[caller] = parent
            continue
        if all(k in params for k in WAY_BACK):
            carry[caller] = parent
        else:
            jax_only[caller] = parent

    out = {
        'lumenairy_file': la.__file__,
        'python': sys.version.split()[0],
        'source_json': parents_path,
        'n_population': d['n_population'],
        'n_with_a_named_parent': d['n_with_a_parent'],
        'n_unresolved_by_the_walk': d['n_unresolved'],
        'n_parent_carries_both': len(carry),
        'n_parent_carries_neither': len(jax_only),
        'n_parent_not_importable': len(unknown),
        'parent_carries_both': carry,
        'parent_carries_neither': jax_only,
        'parent_not_importable': unknown,
    }
    pathlib.Path(out_path).write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items()
                      if not isinstance(v, dict)}, indent=2))
    print('carries neither:', json.dumps(jax_only, indent=2, sort_keys=True))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

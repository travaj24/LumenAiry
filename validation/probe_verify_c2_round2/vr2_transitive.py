"""VERIFY-WP-C2 ROUND 2 -- the TRANSITIVE callers, re-derived independently
on a module-scoped call graph.

The round-2 addendum claims: of the 46 transitive callers VERIFY-WP-C2
listed, **35 resolve to a parent that now carries both keywords** in one to
four hops, and of the 11 that do not, **8 are PMM grating functions that
name no tracer at all** (they entered the round-1 list through a bare-NAME
collision on ``solve``).

This probe builds its own graph rather than reading
``r2_transitive_parents_*.json``:

* every module's functions are indexed as ``module:function``;
* a call resolves to a definition in the SAME MODULE first, then to a
  globally unique definition, and an ambiguous bare name is REFUSED (which
  is the property that makes a transitive walk a census rather than a
  guess);
* imports are followed, so ``from .x import f`` inside module ``m`` makes
  ``f`` in ``m`` resolve to ``x:f`` even when the name is ambiguous
  globally;
* the seeds are the functions whose own body names a tracer (the DIRECT
  census), and the walk is backwards from them.

It then reports, for every EXPORTED function that reaches a tracer
transitively: the chain, the first parent on it that carries both way-back
keywords, and -- when there is none -- why.

Usage:  LUMENAIRY_ROOT=<root> python vr2_transitive.py <out.json>
"""
import ast
import importlib
import inspect
import json
import os
import pathlib
import pkgutil
import sys

import numpy as np  # noqa: F401  (import parity with the other probes)

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

TRACERS = {'trace', 'trace_world', 'trace_prescription', 'raytrace_system',
           'trace_jax', 'trace_jax_world'}
WAY_BACK = ('sphere_normal', 'renormalize')
MAX_HOPS = 6


def read(path):
    raw = pathlib.Path(path).read_bytes()
    for enc in ('cp1252', 'utf-8'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', 'replace')


def build():
    root = pathlib.Path(la.__file__).parent
    defs = {}           # "mod:fn" -> {'calls': [...], 'names_tracer': bool}
    by_module = {}      # mod -> {fn: "mod:fn"}
    by_name = {}        # fn -> ["mod:fn", ...]
    imports = {}        # mod -> {local name: source module}

    for f in sorted(root.rglob('*.py')):
        rel = f.relative_to(root).as_posix()
        mod = 'lumenairy.' + rel[:-3].replace('/', '.')
        if mod.endswith('.__init__'):
            mod = mod[:-len('.__init__')]
        try:
            tree = ast.parse(read(f))
        except SyntaxError:
            continue
        imp = imports.setdefault(mod, {})
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                base = node.module
                if node.level:
                    parts = mod.split('.')
                    parts = parts[:len(parts) - node.level + 1]
                    base = '.'.join(parts + ([node.module] if node.module
                                             else []))
                for a in node.names:
                    imp[a.asname or a.name] = (base, a.name)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            key = '%s:%s' % (mod, node.name)
            calls, names_tracer = set(), False
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name):
                    calls.add(sub.id)
                    if sub.id in TRACERS:
                        names_tracer = True
                elif isinstance(sub, ast.Attribute):
                    calls.add(sub.attr)
                    if sub.attr in TRACERS:
                        names_tracer = True
            defs[key] = {'module': mod, 'fn': node.name,
                         'calls': sorted(calls),
                         'names_tracer': names_tracer}
            by_module.setdefault(mod, {})[node.name] = key
            by_name.setdefault(node.name, []).append(key)
    return defs, by_module, by_name, imports


def resolve(name, mod, by_module, by_name, imports):
    """Module-scoped resolution: same module, then the module the name was
    imported FROM, then a globally unique definition; otherwise refuse."""
    local = by_module.get(mod, {}).get(name)
    if local:
        return local, 'same module'
    src = imports.get(mod, {}).get(name)
    if src:
        cand = by_module.get(src[0], {}).get(src[1])
        if cand:
            return cand, 'imported from %s' % src[0]
    cands = by_name.get(name, [])
    if len(cands) == 1:
        return cands[0], 'globally unique'
    if len(cands) > 1:
        return None, 'ambiguous (%d definitions)' % len(cands)
    return None, 'no definition in the package'


def main(out_path):
    defs, by_module, by_name, imports = build()

    # everything that is EXPORTED, by module-qualified key
    exported = {}
    modnames = ['lumenairy']
    root = pathlib.Path(la.__file__).parent
    for m in pkgutil.walk_packages([str(root)], prefix='lumenairy.'):
        modnames.append(m.name)
    for mn in sorted(set(modnames)):
        try:
            mod = importlib.import_module(mn)
        except Exception:                        # noqa: BLE001
            continue
        for attr in dir(mod):
            if attr.startswith('_'):
                continue
            try:
                obj = getattr(mod, attr)
            except Exception:                    # noqa: BLE001
                continue
            if inspect.isfunction(obj):
                key = '%s:%s' % (obj.__module__, obj.__name__)
                exported.setdefault(key, set()).add('%s.%s' % (mn, attr))

    def way_back(key):
        info = defs.get(key)
        if info is None:
            return None
        try:
            mod = importlib.import_module(info['module'])
            fn = getattr(mod, info['fn'])
            params = inspect.signature(fn).parameters
        except Exception:                        # noqa: BLE001
            return None
        return [k for k in WAY_BACK if k in params]

    direct = {k for k, v in defs.items() if v['names_tracer']}

    rows = {}
    for key in sorted(exported):
        info = defs.get(key)
        if info is None:
            continue
        if key in direct:
            continue                # the DIRECT census covers these
        # breadth-first, module-scoped
        seen = {key}
        frontier = [(key, [key])]
        chain_to_tracer = None
        parent_with_way_back = None
        refusals = []
        hops = 0
        while frontier and hops < MAX_HOPS and chain_to_tracer is None:
            nxt = []
            for cur, path in frontier:
                for name in defs.get(cur, {}).get('calls', []):
                    tgt, how = resolve(name, defs[cur]['module'], by_module,
                                       by_name, imports)
                    if tgt is None:
                        if name not in TRACERS and how.startswith(
                                'ambiguous'):
                            refusals.append('%s (%s)' % (name, how))
                        continue
                    if tgt in seen:
                        continue
                    seen.add(tgt)
                    p = path + [tgt]
                    if defs[tgt]['names_tracer']:
                        chain_to_tracer = p
                        for node in p[1:]:
                            wb = way_back(node)
                            if wb and len(wb) == 2:
                                parent_with_way_back = node
                                break
                        break
                    nxt.append((tgt, p))
                if chain_to_tracer is not None:
                    break
            frontier = nxt
            hops += 1
        if chain_to_tracer is None:
            continue
        rows[key] = {
            'exported_as': sorted(exported[key]),
            'chain': chain_to_tracer,
            'hops': len(chain_to_tracer) - 1,
            'own_way_back': way_back(key),
            'first_parent_with_a_way_back': parent_with_way_back,
            'ambiguous_names_refused': sorted(set(refusals))[:8],
        }

    resolved = {k: v for k, v in rows.items()
                if v['first_parent_with_a_way_back']}
    unresolved = {k: v for k, v in rows.items()
                  if not v['first_parent_with_a_way_back']}

    # compare with the round-2 probe's own population, if it is present
    r2 = pathlib.Path(_ROOT) / 'validation' / 'probe_c2_round2'
    r2_pop, r2_unres = [], []
    p = r2 / 'r2_transitive_parents_win.json'
    if p.exists():
        d = json.loads(p.read_text(encoding='utf-8'))
        r2_pop = sorted(d['population'])
        r2_unres = sorted(d['unresolved'])

    mine_bare = {k.split(':')[1] for k in rows}
    out = {
        'lumenairy_file': la.__file__,
        'python': sys.version.split()[0],
        'n_direct_tracing_defs': len(direct),
        'n_exported_transitive': len(rows),
        'n_with_a_parent_carrying_both': len(resolved),
        'n_without': len(unresolved),
        'without_a_parent': sorted(unresolved),
        'round2_population': r2_pop,
        'round2_unresolved': r2_unres,
        'round2_unresolved_that_are_pmm': sorted(
            n for n in r2_unres if n.startswith('pmm')),
        'round2_unresolved_that_are_not_pmm': sorted(
            n for n in r2_unres if not n.startswith('pmm')),
        'round2_population_this_walk_also_reaches': sorted(
            n for n in r2_pop if n in mine_bare),
        'round2_population_this_walk_does_not_reach': sorted(
            n for n in r2_pop if n not in mine_bare),
        'rows': rows,
    }
    pathlib.Path(out_path).write_text(json.dumps(out, indent=1),
                                      encoding='utf-8')
    for k, v in out.items():
        if k == 'rows':
            continue
        print('%-46s %s' % (k, v if not isinstance(v, list) else len(v)))
    print()
    print('round-2 unresolved, PMM       :',
          out['round2_unresolved_that_are_pmm'])
    print('round-2 unresolved, the OTHERS:',
          out['round2_unresolved_that_are_not_pmm'])
    print('this walk reaches a tracer from',
          len(out['round2_population_this_walk_also_reaches']),
          'of the round-2 population of', len(r2_pop))
    print('wrote', out_path)


if __name__ == '__main__':
    main(sys.argv[1])

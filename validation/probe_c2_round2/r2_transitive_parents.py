"""WP-C2 round 2, defect D4 -- which parent each TRANSITIVE caller reaches.

VERIFY-WP-C2's census also found 46 exported functions that reach a tracer
INDIRECTLY.  The campaign rule does not ask for a keyword on every one of
them: it asks that the way back be REACHABLE, and a transitive caller reaches
it through the parent it calls.  This probe names that parent and the chain,
so the 46 is an answer rather than a count.

WHY THE CALL GRAPH IS MODULE-SCOPED.  A graph keyed by bare NAME is sound for
a DIRECT census -- the function itself names the tracer, and `trace` bound in
its own module IS the tracer -- but not for a transitive walk: `solve`, `fn`,
`_run`, `_apply`, `_trace` and `propagate` are each defined many times across
the package, and following them by name alone manufactures chains that do not
exist (`pmm_jones_2d` "reaching" `fit_canonical_polynomials` through somebody
else's `solve`).  Here a call is resolved to a definition in the SAME MODULE
first, then to a globally unique definition, and an ambiguous name is refused
rather than guessed.  Whatever cannot be resolved inside six such hops is
reported as unresolved, with what it actually calls, so a reader can tell
"no path" from "the walk gave up".

Usage:  OMP_NUM_THREADS=1 ... python r2_transitive_parents.py --root <tree>
            --out r2_transitive_parents_win.json
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import os
import pathlib
import sys

TRACERS = {'trace', 'trace_world', 'trace_prescription', 'raytrace_system',
           'trace_jax', 'trace_jax_world'}
KEYWORDS = ('sphere_normal', 'renormalize')
MAX_HOPS = 6


def _public_names():
    out = {}
    for mod in ('lumenairy', 'lumenairy.raytrace', 'lumenairy.analysis',
                'lumenairy.io', 'lumenairy.optimize', 'lumenairy.elements',
                'lumenairy.propagators'):
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        for n in dir(m):
            if not n.startswith('_'):
                out.setdefault(n, m)
    return out


def _scan(pkg):
    """(module, name) -> called NAMES, plus a name -> defining modules index."""
    calls = {}
    where = {}
    for f in sorted(pathlib.Path(pkg).rglob('*.py')):
        rel = f.relative_to(pkg).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding='utf-8', errors='replace'))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            named = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    if isinstance(fn, ast.Name):
                        named.add(fn.id)
                    elif isinstance(fn, ast.Attribute):
                        named.add(fn.attr)
                elif isinstance(sub, ast.Name) and sub.id in TRACERS:
                    named.add(sub.id)      # a tracer PASSED, not called
            calls[(rel, node.name)] = named
            where.setdefault(node.name, set()).add(rel)
    return calls, where


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    import lumenairy
    assert os.path.abspath(lumenairy.__file__).startswith(root)
    print('lumenairy.__file__ =', lumenairy.__file__)

    pkg = pathlib.Path(lumenairy.__file__).parent
    calls, where = _scan(pkg)
    public = _public_names()

    def _keywords(name):
        m = public.get(name)
        if m is None:
            return None
        try:
            params = inspect.signature(getattr(m, name)).parameters
        except (TypeError, ValueError):
            return None
        return [k for k in KEYWORDS if k in params]

    parents = {n for n in public if len(_keywords(n) or []) == 2}
    parents |= {'trace', 'trace_world'}

    def _resolve(name, from_module):
        """(module, name) for a call: same module first, then unique global."""
        mods = where.get(name)
        if not mods:
            return None
        if from_module in mods:
            return (from_module, name)
        if len(mods) == 1:
            return (next(iter(mods)), name)
        return None                        # ambiguous: refuse to guess

    def _path_to_a_parent(start_key):
        seen, frontier = {start_key}, [(start_key, [])]
        for _ in range(MAX_HOPS):
            nxt = []
            for key, path in frontier:
                mod = key[0]
                for callee in sorted(calls.get(key, ())):
                    if callee in parents or callee in TRACERS:
                        return path + [callee]
                    nk = _resolve(callee, mod)
                    if nk is None or nk in seen:
                        continue
                    seen.add(nk)
                    nxt.append((nk, path + [callee]))
            frontier = nxt
            if not frontier:
                break
        return None

    census = json.load(open(os.path.join(
        root, 'validation', 'probe_verify_c2', 'vc2_entrypoints_win.json'),
        encoding='utf-8'))
    population = sorted(census['exported_transitive_without_a_way_back'])

    rows, unresolved = {}, {}
    for name in population:
        if len(_keywords(name) or []) == 2:
            continue                       # has its own way back
        best = None
        for key in (k for k in calls if k[1] == name):
            path = _path_to_a_parent(key)
            if path is not None and (best is None or len(path) < len(best)):
                best = path
        if best is None:
            unresolved[name] = sorted(
                c for key in calls if key[1] == name for c in calls[key])[:12]
            continue
        rows[name] = {'parent': best[-1], 'path': best}

    payload = {
        'lumenairy_file': lumenairy.__file__,
        'python': sys.version.split()[0],
        'max_hops': MAX_HOPS,
        'population': population,
        'n_population': len(population),
        'n_with_a_parent': len(rows),
        'transitive_parents': rows,
        'unresolved': unresolved,
        'n_unresolved': len(unresolved),
    }
    for k in sorted(rows):
        print(f'  {k:46s} -> {" -> ".join(rows[k]["path"])}')
    print('transitive callers with a named parent:', len(rows), 'of',
          len(population))
    print('unresolved:', sorted(unresolved))
    with open(args.out, 'w', encoding='ascii') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()

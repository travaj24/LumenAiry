"""WP-C2 ROUND 3, VR2-D1 -- the census, with and without the two
resolutions, over an arbitrary package tree.

Three readings of the same tree:

* ``name_only``   -- the round-1/2 census: a body traces if it NAMES one of
  the tracers (or calls one as an attribute).
* ``alias``       -- plus import aliases (``trace as _rt_trace``), the edit
  VERIFY-WP-C2 round 2 requested.
* ``alias_hop``   -- plus ONE hop into a private helper of the same module,
  which is what finally reaches ``apply_real_lens`` through the private
  split-out ``_apply_real_lens_impl``.

The tree is read as SOURCE only -- nothing is imported -- so it can be
pointed at a ``git archive`` of the PRE commit without that tree having to
be importable, and the "is it exported" question is answered from the
package's ``__init__`` re-export text rather than from ``dir()``.

Usage:  python r3_census_alias_hop.py <package_root> <out.json> [--tag t]
"""
import ast
import json
import pathlib
import sys

TRACERS = {'trace', 'trace_world', 'trace_prescription', 'raytrace_system',
           'trace_jax', 'trace_jax_world'}
WAY_BACK = ('sphere_normal', 'renormalize')


def read(path):
    for enc in ('cp1252', 'utf-8'):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return path.read_text(encoding='utf-8', errors='replace')


def sig_keywords(node):
    a = node.args
    names = ([p.arg for p in a.args] + [p.arg for p in a.kwonlyargs]
             + [p.arg for p in a.posonlyargs])
    return [k for k in WAY_BACK if k in names]


def census(root, use_alias, use_hop):
    root = pathlib.Path(root)
    found = {}
    for f in sorted(root.rglob('*.py')):
        try:
            tree = ast.parse(read(f))
        except SyntaxError:
            continue
        aliases = {a.asname: a.name
                   for n in ast.walk(tree)
                   if isinstance(n, (ast.Import, ast.ImportFrom))
                   for a in n.names
                   if a.asname and a.name in TRACERS} if use_alias else {}
        defs, used_by, traces = {}, {}, {}
        for n in ast.walk(tree):
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            defs[n.name] = n
            used, hit = set(), False
            for s in ast.walk(n):
                if isinstance(s, ast.Name):
                    used.add(s.id)
                    if s.id in TRACERS or s.id in aliases:
                        hit = True
                elif (isinstance(s, ast.Call)
                        and isinstance(s.func, ast.Attribute)
                        and s.func.attr in TRACERS):
                    hit = True
            used_by[n.name] = used
            if hit:
                traces[n.name] = True
        changed = bool(use_hop)
        while changed:
            changed = False
            for name, used in used_by.items():
                if name in traces:
                    continue
                if any(c.startswith('_') and c in defs and c in traces
                       for c in used):
                    traces[name] = True
                    changed = True
        for name in traces:
            if name.startswith('_'):
                continue
            found[name] = sig_keywords(defs[name])
    return found


def exported(root):
    """Names the package re-exports, read from every ``__init__.py``'s
    ``__all__`` and import statements -- no import of the tree itself."""
    root = pathlib.Path(root)
    out = set()
    for f in sorted(root.rglob('__init__.py')):
        try:
            tree = ast.parse(read(f))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    out.add(a.asname or a.name.split('.')[-1])
            elif (isinstance(n, ast.Assign)
                    and any(getattr(t, 'id', '') == '__all__'
                            for t in n.targets)
                    and isinstance(n.value, (ast.List, ast.Tuple))):
                for e in n.value.elts:
                    if isinstance(e, ast.Constant) and isinstance(e.value,
                                                                  str):
                        out.add(e.value)
    return out


def main(root, out_path, tag='?'):
    pub = exported(root)
    out = {'tag': tag, 'package_root': str(root), 'readings': {}}
    for label, (al, hop) in (('name_only', (False, False)),
                             ('alias', (True, False)),
                             ('alias_hop', (True, True))):
        c = {k: v for k, v in census(root, al, hop).items() if k in pub}
        out['readings'][label] = {
            'n_entry_points': len(c),
            'n_with_both': sum(1 for v in c.values() if len(v) == 2),
            'names': sorted(c),
            'without_both': sorted(k for k, v in c.items() if len(v) < 2),
            'apply_real_lens': c.get('apply_real_lens', '<not in census>'),
        }
    pathlib.Path(out_path).write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding='utf-8')
    for label, r in out['readings'].items():
        print(f"{tag} {label}: {r['n_entry_points']} entry points, "
              f"{r['n_with_both']} with both, apply_real_lens="
              f"{r['apply_real_lens']!r}")


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    t = sys.argv[sys.argv.index('--tag') + 1] if '--tag' in sys.argv else '?'
    main(a[0], a[1], t)

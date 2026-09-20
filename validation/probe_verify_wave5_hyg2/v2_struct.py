"""(2) STRUCTURAL INVENTORY by AST -- 'no _jax twin' for the Collins chain.

Reads the SOURCE with ``ast``; imports nothing from the tree except to
anchor it.  Three enumerations:

  a. every function/method in lumenairy/propagators/carrier.py whose name
     carries a backend suffix (_jax|_jnp|_cupy|_cp|_gpu|_np|_numpy);
  b. every import in that module of a ``*_jax*`` (or *_cupy*/_gpu*) sibling
     module;
  c. every function ANYWHERE under lumenairy/ whose name is a Collins helper
     name plus a backend suffix, plus a whole-package census of
     backend-suffixed function names and of multiply-defined function names
     that look like the SAME kernel in two flavours.

    python v2_struct.py <tree> <out.json>
"""
from __future__ import annotations

import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import anchor, build_tag, write_json                 # noqa: E402

SUFFIXES = ('_jax', '_jnp', '_cupy', '_cp', '_gpu', '_np', '_numpy',
            '_torch')


def read(p):
    for enc in ('utf-8', 'cp1252', 'latin-1'):
        try:
            return open(p, encoding=enc).read()
        except UnicodeDecodeError:
            continue
    return open(p, encoding='utf-8', errors='replace').read()


def funcs(tree_src, path):
    out = []
    try:
        mod = ast.parse(tree_src)
    except SyntaxError as exc:
        return [('<SYNTAXERROR>', str(exc), 0)]
    for node in ast.walk(mod):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((node.name, path, node.lineno))
    return out


def main():
    tree, out_path = sys.argv[1], sys.argv[2]
    anchor(tree)
    root = os.path.join(os.path.realpath(tree), 'lumenairy')
    carrier = os.path.join(root, 'propagators', 'carrier.py')
    src = read(carrier)
    cfuncs = funcs(src, 'lumenairy/propagators/carrier.py')

    res = {'build': build_tag(), 'tree': tree}

    # (a) backend-suffixed function names in carrier.py
    res['a_carrier_backend_suffixed'] = sorted(
        {'%s  (line %d)' % (n, ln) for n, _p, ln in cfuncs
         if any(n.lower().endswith(s) for s in SUFFIXES)})
    res['a_carrier_function_count'] = len(cfuncs)
    res['a_carrier_names_containing_xp'] = sorted(
        {n for n, _p, _l in cfuncs if n.lower().endswith('_xp')})

    # (b) imports of *_jax* / *_cupy* / *_gpu* siblings, anywhere in the module
    mod = ast.parse(src)
    imports = []
    for node in ast.walk(mod):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.append((a.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            base = ('.' * (node.level or 0)) + (node.module or '')
            for a in node.names:
                imports.append((base + '::' + a.name, node.lineno))
    res['b_all_imports_count'] = len(imports)
    res['b_backend_flavoured_imports'] = sorted(
        '%s  (line %d)' % (m, ln) for m, ln in imports
        if any(t in m.lower() for t in ('_jax', 'jax_', '_cupy', 'cupy_',
                                        '_gpu', 'gpu_', '_numpy_'))
    )
    res['b_jax_imports_any'] = sorted(
        '%s  (line %d)' % (m, ln) for m, ln in imports
        if 'jax' in m.lower() or 'cupy' in m.lower())

    # (c) package-wide census
    allf = []
    for dirpath, _d, files in os.walk(root):
        if '__pycache__' in dirpath:
            continue
        for f in files:
            if not f.endswith('.py'):
                continue
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, os.path.realpath(tree)).replace('\\', '/')
            allf.extend(funcs(read(p), rel))
    res['c_package_function_count'] = len(allf)

    collins = sorted({n for n, _p, _l in cfuncs if n.startswith('_collins')})
    res['c_collins_helpers'] = collins
    new_helpers = ['_is_traced', '_as_c_order', '_fft2_pair', '_backend_of',
                   '_to_dev', '_tf_phase_to_H']
    watch = collins + new_helpers
    sib = []
    for n, p, ln in allf:
        for base in watch:
            for s in SUFFIXES:
                if n == base + s or n == base.rstrip('_') + s:
                    sib.append('%s  %s:%d  (sibling of %s)' % (n, p, ln, base))
    res['c_backend_suffixed_siblings_of_collins_chain'] = sorted(set(sib))

    # every backend-suffixed function ANYWHERE in the package (context)
    pkg_suffixed = sorted({'%s  %s:%d' % (n, p, ln) for n, p, ln in allf
                           if any(n.lower().endswith(s) for s in SUFFIXES)})
    res['c_package_backend_suffixed_count'] = len(pkg_suffixed)
    res['c_package_backend_suffixed'] = pkg_suffixed

    # names defined more than once across the package with different flavours
    from collections import defaultdict
    by_base = defaultdict(set)
    for n, p, ln in allf:
        low = n.lower()
        for s in SUFFIXES:
            if low.endswith(s):
                by_base[n[:-len(s)]].add('%s  %s:%d' % (n, p, ln))
    pairs = {k: sorted(v) for k, v in by_base.items() if len(v) > 1}
    res['c_flavour_pairs'] = pairs

    # (d) does a census / single-definition gate exist, and does H2-2 meet it
    tdir = os.path.join(os.path.realpath(tree), 'tests', 'unit')
    gates = []
    if os.path.isdir(tdir):
        for f in sorted(os.listdir(tdir)):
            if not f.endswith('.py'):
                continue
            s = read(os.path.join(tdir, f))
            hits = []
            if 'census' in f.lower():
                hits.append('filename')
            for needle in ('single definition', 'single-definition',
                           'ONE implementation', 'one implementation per',
                           'per-flavour', 'per-flavor', 'backend suffix',
                           'per-backend copy', '_jax twin'):
                if needle in s:
                    hits.append(needle)
            if hits:
                gates.append({'file': 'tests/unit/' + f, 'hits': sorted(set(hits))})
    res['d_candidate_gate_files'] = gates

    write_json(res, out_path)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ('c_package_backend_suffixed',
                                   'b_jax_imports_any')}, indent=1))


if __name__ == '__main__':
    main()

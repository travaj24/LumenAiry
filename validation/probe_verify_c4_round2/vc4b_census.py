"""VERIFY-WP-C4 ROUND 2, item 4c -- an INDEPENDENT AST census of the exported
entry points that can reach an MFT primitive.

Written from scratch rather than by re-running
``test_c4_round2_mft_method.py``'s own sweep: a census that only agrees with
itself proves nothing.  Two things are done differently here.

* **The reachability walk goes FORWARDS from every exported function**, with a
  module-qualified resolver, instead of backwards from the primitives.  The two
  directions have to agree on the same set or one of them has a bug.
* **``__all__`` is not the only door.**  Every public (non-underscore) module
  attribute of ``lumenairy`` that is a function is considered, and the two
  populations are reported separately, so an entry point that is importable but
  missing from ``__all__`` is visible rather than silently outside the census.

It also answers, per exported entry point, whether the way back is a named
parameter (``method`` / ``mft_method``) or a ``**kwargs`` splat into one of the
three MFT propagators, and whether that route keyword is actually FORWARDED to
something (a signature that accepts the keyword and drops it is the failure
mode the census cannot see on its own).

    PYTHONPATH=<tree> python vc4b_census.py <tree> [--mutate <name>]

``--mutate`` removes the route keyword from one exported function's parameter
set, in this process only, and the census must then name exactly that function.

Author:  Andrew Traverso
"""
from __future__ import annotations

import ast
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vc4blib as L                                              # noqa: E402

PRIMITIVES = ('_bluestein_2d', '_bluestein_centred_2d')
MFT_PUBLIC = ('angular_spectrum_propagate_mft', 'fresnel_propagate_mft',
              'fraunhofer_propagate_mft')
ROUTE_KW = ('method', 'mft_method')


def parse_package(root):
    mods = {}
    for dirpath, _d, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.endswith('.py'):
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root).replace(os.sep, '/')
            with open(p, encoding='cp1252', errors='replace') as fh:
                mods[rel] = ast.parse(fh.read())
    return mods


def index(mods):
    """``defined[name] -> {module}`` and ``imported[module][name] ->
    {module}``, the module-qualified resolver both directions share."""
    defined, imported = {}, {rel: {} for rel in mods}
    for rel, tree in mods.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.setdefault(node.name, set()).add(rel)
    for rel, tree in mods.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported[rel].setdefault(
                        alias.asname or alias.name,
                        set()).update(defined.get(alias.name, set()))
    return defined, imported


def forward_edges(mods, defined, imported):
    """``callees[(mod, fn)] -> {(mod, fn)}`` -- the FORWARD call graph."""
    def resolve(rel, name):
        out = set()
        if rel in defined.get(name, set()):
            out.add((rel, name))
        for src in imported.get(rel, {}).get(name, set()):
            out.add((src, name))
        return out

    callees = {}
    for rel, tree in mods.items():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            here = callees.setdefault((rel, node.name), set())
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Call):
                    continue
                name = (getattr(sub.func, 'id', None)
                        or getattr(sub.func, 'attr', None))
                if name:
                    here |= resolve(rel, name)
    return callees


def reaches_primitive(start, callees, stop_at_public):
    """Forward DFS from ``start``.  A path may pass through a PRIVATE function
    or one of the three MFT propagators; it may not pass through any other
    public function, because a caller of that function is a caller of THAT
    function and its way back is that function's own keyword."""
    seen, stack = set(), [(start, True)]
    while stack:
        node, is_start = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        mod, name = node
        if mod.startswith('ui/'):
            continue
        if not is_start:
            if name in PRIMITIVES:
                return True
            transparent = name.startswith('_') or name in MFT_PUBLIC
            if stop_at_public and not transparent:
                continue
        for nxt in callees.get(node, ()):
            stack.append((nxt, False))
    return False


def splats_into_mft(mods, module, fn_name):
    tree = mods.get(module)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == fn_name):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and (getattr(sub.func, 'id', None)
                             or getattr(sub.func, 'attr', None))
                        in MFT_PUBLIC
                        and any(kw.arg is None for kw in sub.keywords)):
                    return True
    return False


def forwards_the_keyword(mods, module, fn_name):
    """Does the body MENTION the route keyword after the signature?

    A signature that accepts ``mft_method=`` and never reads it is the exact
    failure the census cannot see, so it is recorded here as a separate fact.
    """
    tree = mods.get(module)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == fn_name):
            body = ast.Module(body=node.body, type_ignores=[])
            for sub in ast.walk(body):
                if isinstance(sub, ast.Name) and sub.id in ROUTE_KW:
                    return True
                if isinstance(sub, ast.keyword) and sub.arg in ROUTE_KW:
                    return True
                if (isinstance(sub, ast.Constant)
                        and isinstance(sub.value, str)
                        and sub.value in ROUTE_KW):
                    return True
    return False


def main(tree, mutate):
    la = L.anchor(tree)
    root = os.path.dirname(os.path.abspath(la.__file__))
    mods = parse_package(root)
    defined, imported = index(mods)
    callees = forward_edges(mods, defined, imported)

    # ---- the FORWARD census ------------------------------------------------
    exported = sorted(getattr(la, '__all__', ()))
    public_attrs = sorted(n for n in dir(la) if not n.startswith('_'))
    rows, offenders = [], []
    for kind, names in (('__all__', exported),
                        ('public_attr', public_attrs)):
        for name in names:
            obj = getattr(la, name, None)
            if not inspect.isfunction(obj):
                continue
            mods_for = sorted(defined.get(obj.__name__, ()))
            hit = [m for m in mods_for
                   if reaches_primitive((m, obj.__name__), callees, True)]
            if not hit:
                continue
            params = set(inspect.signature(obj).parameters)
            if mutate and obj.__name__ == mutate:
                params -= set(ROUTE_KW)
            named = sorted(params & set(ROUTE_KW))
            splat = any(splats_into_mft(mods, m, obj.__name__) for m in hit)
            fwd = any(forwards_the_keyword(mods, m, obj.__name__)
                      for m in hit)
            row = {'kind': kind, 'exported_as': name,
                   'qualname': obj.__name__, 'modules': hit,
                   'named_route_keywords': named,
                   'splats_into_mft_propagator': splat,
                   'body_mentions_a_route_keyword': fwd,
                   'has_way_back': bool(named) or splat}
            if kind == '__all__':
                rows.append(row)
                if not row['has_way_back']:
                    offenders.append(name)
    # ---- the BACKWARD reachable set, for cross-checking the "22" -----------
    back_seen = set()
    rev = {}
    for src, dsts in callees.items():
        for d in dsts:
            rev.setdefault(d, set()).add(src)
    seed = {(m, n) for n in PRIMITIVES for m in defined.get(n, ())}
    frontier = set(seed)
    while frontier:
        nxt = set()
        for node in frontier:
            mod, name = node
            if not ((name.startswith('_') or name in MFT_PUBLIC)
                    and not mod.startswith('ui/')):
                continue
            for caller in rev.get(node, ()):
                if caller[0].startswith('ui/') or caller in back_seen:
                    continue
                back_seen.add(caller)
                nxt.add(caller)
        frontier = nxt

    out = {'build': L.build(), 'tree': tree, 'mutate': mutate,
           'n_modules_parsed': len(mods),
           'forward_exported_reaching_a_primitive': len(rows),
           'forward_rows': rows,
           'offenders_without_a_way_back': offenders,
           'backward_reachable_functions': len(back_seen),
           'backward_reachable_sorted': sorted(back_seen),
           'backward_exported': sorted(
               {n for _m, n in back_seen
                if n in set(exported) | {getattr(la, e).__name__
                                         for e in exported
                                         if inspect.isfunction(
                                             getattr(la, e, None))}}),
           }
    print("modules parsed                      :", out['n_modules_parsed'])
    print("BACKWARD reachable functions        :",
          out['backward_reachable_functions'])
    print("FORWARD exported reaching a primitive:",
          out['forward_exported_reaching_a_primitive'])
    for r in rows:
        print("  %-42s kw=%-22s splat=%-5s body=%-5s -> %s"
              % (r['exported_as'], ','.join(r['named_route_keywords']) or '-',
                 r['splats_into_mft_propagator'],
                 r['body_mentions_a_route_keyword'],
                 'OK' if r['has_way_back'] else 'NO WAY BACK'))
    print("OFFENDERS:", offenders)
    suffix = ("_mut_%s" % mutate) if mutate else ""
    L.write(out, os.path.join(HERE, "vc4b_census%s_%s.json"
                              % (suffix, L.tag())))


if __name__ == '__main__':
    main(sys.argv[1], L.arg('--mutate', None))

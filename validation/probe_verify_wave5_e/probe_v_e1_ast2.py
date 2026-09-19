"""VERIFY-WAVE5-E / E1(b) round 2 -- the AST walk the claim really needs.

Round 1 keyed on ``_fft2(...) * other`` with the dispatcher spelled INLINE.
That misses the exposed shape ``spec = _fft2(E); spec * np.exp(1j*P)`` -- the
dispatcher's non-owning view held under a NAME, multiplied by a fresh
temporary.  This pass does both:

  (A) inline:  any BinOp whose operand is a dispatcher CALL
  (B) via a name: per function, names bound DIRECTLY to a dispatcher call
      (``x = _fft2(...)``), then any BinOp using that name

and classifies the other operand for ELIDABILITY rather than for syntax:

  NAME/ATTR         -> live reference, never elidable
  BASIC-SUBSCRIPT   -> a VIEW (owndata False), never elidable
  ADV-SUBSCRIPT     -> a fresh OWNED array, ELIDABLE
  CALL/BINOP/UNARY  -> a fresh temporary, ELIDABLE if numpy-owned
  CONSTANT          -> scalar, not an array temp
"""
import ast
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
DISPATCH = {'_fft2', '_ifft2', '_fft2_nd', '_ifft2_nd',
            '_rfft2', '_irfft2', '_fftn', '_ifftn'}
OPS = (ast.Mult, ast.Add, ast.Sub, ast.Div)


def _callee(node):
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _basic_slice(sl):
    parts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
    for p in parts:
        if isinstance(p, ast.Slice):
            continue
        if isinstance(p, ast.Constant):          # int index or None
            continue
        return False
    return True


def _kind(node):
    if isinstance(node, (ast.Name, ast.Attribute)):
        return 'NAME/ATTR'
    if isinstance(node, ast.Constant):
        return 'CONSTANT'
    if isinstance(node, ast.Subscript):
        return 'BASIC-SUBSCRIPT' if _basic_slice(node.slice) \
            else 'ADV-SUBSCRIPT(ELIDABLE)'
    return 'ELIDABLE:' + type(node).__name__


rows = []
nfiles = 0
for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, 'lumenairy')):
    dirnames[:] = [d for d in dirnames if d != '__pycache__']
    for fn in sorted(filenames):
        if not fn.endswith('.py'):
            continue
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, ROOT).replace(chr(92), chr(47))
        nfiles += 1
        with open(p, 'rb') as fh:
            src = fh.read().decode('utf-8')
        tree = ast.parse(src, filename=p)
        # (B) collect names bound directly to a dispatcher call, per scope
        scopes = [n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.Module))]

        def _own(scope):
            # nodes belonging to THIS scope only -- nested defs are their own
            stack, out = list(ast.iter_child_nodes(scope)), []
            while stack:
                nd = stack.pop()
                out.append(nd)
                if isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef, ast.Lambda)):
                    continue
                stack.extend(ast.iter_child_nodes(nd))
            return out

        for sc in scopes:
            own = _own(sc)
            bound = {}
            for n in own:
                if isinstance(n, ast.Assign) and _callee(n.value) in DISPATCH:
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            bound[t.id] = (_callee(n.value), n.lineno)
            for n in own:
                if not (isinstance(n, ast.BinOp) and isinstance(n.op, OPS)):
                    continue
                for side, me, other in (('left', n.left, n.right),
                                        ('right', n.right, n.left)):
                    c = _callee(me)
                    via = None
                    if c in DISPATCH:
                        via = 'inline:' + c
                    elif isinstance(me, ast.Name) and me.id in bound:
                        via = 'name:%s=%s@%d' % ((me.id,) + bound[me.id])
                    if via is None:
                        continue
                    rows.append(dict(
                        file=rel, line=n.lineno, scope=getattr(
                            sc, 'name', '<module>'),
                        via=via, dispatcher_side=side,
                        op=type(n.op).__name__,
                        other_operand_kind=_kind(other),
                        src=ast.unparse(n)[:170]))

seen = set()
uniq = []
for r in rows:
    k = (r['file'], r['line'], r['dispatcher_side'], r['via'])
    if k not in seen:
        seen.add(k)
        uniq.append(r)
bad = [r for r in uniq if 'ELIDABLE' in r['other_operand_kind']]
print(json.dumps(dict(root=os.path.abspath(ROOT), py_files_walked=nfiles,
                      n_sites=len(uniq), n_elidable_other=len(bad),
                      elidable=bad, all_sites=uniq), indent=1))

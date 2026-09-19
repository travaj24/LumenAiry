"""VERIFY-WAVE5-E / E1(b): AST walk for FFT-dispatcher product sites.

Finds every ``BinOp(Mult)`` in ``lumenairy/`` one of whose operands is a call to
an FFT dispatcher (``_fft2``/``_ifft2``/``_fft2_nd``/``_ifft2_nd``/``fft2``...),
and classifies the OTHER operand as NAMED (a Name / Attribute / Subscript --
something with a live reference, never elidable) or UNNAMED (a Call / BinOp /
UnaryOp -- a fresh temporary NumPy may elide into).
"""
import ast
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
DISPATCH = {'_fft2', '_ifft2', '_fft2_nd', '_ifft2_nd',
            '_rfft2', '_irfft2', '_fftn', '_ifftn'}

NAMED = (ast.Name, ast.Attribute, ast.Subscript)


def _callee(node):
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _kind(node):
    if isinstance(node, NAMED):
        return 'NAMED'
    if isinstance(node, ast.Constant):
        return 'CONSTANT'
    return 'UNNAMED:' + type(node).__name__


rows = []
nfiles = 0
for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, 'lumenairy')):
    dirnames[:] = [d for d in dirnames if d != '__pycache__']
    for fn in sorted(filenames):
        if not fn.endswith('.py'):
            continue
        p = os.path.join(dirpath, fn)
        nfiles += 1
        with open(p, 'rb') as fh:
            src = fh.read().decode('utf-8')
        tree = ast.parse(src, filename=p)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.Mult)):
                continue
            lc, rc = _callee(node.left), _callee(node.right)
            if lc in DISPATCH or rc in DISPATCH:
                side = 'left' if lc in DISPATCH else 'right'
                other = node.right if side == 'left' else node.left
                rows.append(dict(
                    file=os.path.relpath(p, ROOT).replace(chr(92), chr(47)),
                    line=node.lineno,
                    dispatcher=(lc if side == 'left' else rc),
                    dispatcher_side=side,
                    other_operand_kind=_kind(other),
                    src=ast.unparse(node)[:160],
                ))

bad = [r for r in rows if r['other_operand_kind'].startswith('UNNAMED')]
out = dict(root=os.path.abspath(ROOT), py_files_walked=nfiles,
           n_product_sites=len(rows),
           n_named=len([r for r in rows if r['other_operand_kind'] == 'NAMED']),
           n_constant=len([r for r in rows
                           if r['other_operand_kind'] == 'CONSTANT']),
           n_unnamed=len(bad), unnamed=bad, all_sites=rows)
print(json.dumps(out, indent=1))

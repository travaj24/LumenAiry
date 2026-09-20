"""VERIFY-WAVE5-HYGIENE2 round 2 -- an INDEPENDENT single-definition census.

The shipped census (``test_wave5_h2_collins_jax.py::
test_the_exact_dispersion_is_written_once``) looks for three literal tokens
inside ``carrier.py``.  This one is written from the other end: it walks the
AST of EVERY module in ``lumenairy/propagators/``, strips docstrings and
comments by unparsing each function body from the AST, and then asks two
questions that do not depend on how the expression is spelled:

 1. STRUCTURAL.  Which functions subtract FROM ``k**2`` (or ``k * k``), as a
    BinOp or through ``np.subtract``?  That is the non-paraxial dispersion by
    shape, not by token, so a re-spelled copy is still counted.
 2. TOKEN.  The three tokens the shipped gate names, re-measured here
    package-wide rather than file-wide.

It also reports, for each of the three call sites, the exact expression its
call to ``_exact_dispersion_phase`` sits in -- which is how a SITE-LOCAL sign
flip (``-_exact_dispersion_phase(...)``) becomes visible to a reader even
though the consolidation is intact.

    python vh3_census.py <tree-root> <out.json>
"""
import ast
import json
import pathlib
import sys


def strip_docstrings(node):
    """Return an AST copy of ``node``'s body with every docstring removed."""
    new = ast.parse(ast.unparse(node))
    for n in ast.walk(new):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef, ast.Module)):
            body = getattr(n, 'body', None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                n.body = body[1:] or [ast.Pass()]
    return new


def is_k_squared(node):
    """``k ** 2`` or ``k * k`` on the name ``k``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return (isinstance(node.left, ast.Name) and node.left.id == 'k'
                and isinstance(node.right, ast.Constant)
                and node.right.value == 2)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return (isinstance(node.left, ast.Name) and node.left.id == 'k'
                and isinstance(node.right, ast.Name) and node.right.id == 'k')
    return False


def dispersion_shapes(fnode):
    """Every ``k^2 - <something>`` in this function body, unparsed.

    The dispersion's distinctive SHAPE is the subtraction from ``k^2``, not
    the ``sqrt`` around it: the shipped kernel writes the radical into a name
    (``rad = k * k - (ax * ax + ay * ay)``) and the untilted fast arm writes
    it through ``np.subtract(k * k, phase, out=phase)``, so a detector keyed
    on ``sqrt(...)`` sees neither.  Both forms are matched here, which is what
    makes this census independent of the shipped one's three literal tokens.
    """
    hits = []
    for n in ast.walk(fnode):
        if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)
                and is_k_squared(n.left)):
            hits.append(ast.unparse(n))
        elif isinstance(n, ast.Call):
            name = (n.func.attr if isinstance(n.func, ast.Attribute)
                    else n.func.id if isinstance(n.func, ast.Name) else None)
            if name in ('subtract',) and n.args and is_k_squared(n.args[0]):
                hits.append(ast.unparse(n))
    return hits


def main(root, out):
    root = pathlib.Path(root)
    pkg = root / 'lumenairy'
    report = {'root': str(root), 'structural': {}, 'tokens': {},
              'call_site_expressions': {}, 'assignment_forms': {}}

    tokens = {'root0': 'the q = 0 subtraction k N',
              'ax * ax + ay * ay': 'the shifted-frequency radical',
              's2 < 1.0': 'the evanescent-carrier guard'}
    tok_owners = {t: [] for t in tokens}
    struct_owners = []
    nfun = 0

    for path in sorted(pkg.rglob('*.py')):
        try:
            src = path.read_text(encoding='cp1252')
        except UnicodeDecodeError:
            src = path.read_text(encoding='utf-8')
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            nfun += 1
            clean = strip_docstrings(fn)
            body = ast.unparse(clean)
            hits = dispersion_shapes(clean)
            if hits:
                struct_owners.append(
                    {'module': str(path.relative_to(root)).replace('\\', '/'),
                     'function': fn.name, 'expressions': sorted(set(hits))})
            for t in tokens:
                if t in body:
                    tok_owners[t].append(
                        f"{str(path.relative_to(root)).replace(chr(92), '/')}"
                        f"::{fn.name}")

    report['n_functions_walked'] = nfun
    report['structural'] = struct_owners
    report['tokens'] = {t: sorted(set(v)) for t, v in tok_owners.items()}

    # The three call sites, with the EXPRESSION their kernel call sits in.
    carrier = pkg / 'propagators' / 'carrier.py'
    csrc = carrier.read_text(encoding='cp1252')
    ctree = ast.parse(csrc)
    sites = ('_exact_tf_2d_xp', '_exact_envelope_tf_step',
             '_collins_exact_kernel_correction')
    for fn in [n for n in ast.walk(ctree) if isinstance(n, ast.FunctionDef)]:
        if fn.name not in sites:
            continue
        clean = strip_docstrings(fn)
        exprs = []
        for n in ast.walk(clean):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == '_exact_dispersion_phase'):
                exprs.append('CALL')
        forms = []
        for n in ast.walk(clean):
            if isinstance(n, (ast.Assign, ast.Return, ast.AugAssign)):
                t = ast.unparse(n)
                if '_exact_dispersion_phase' in t:
                    forms.append(t)
        report['call_site_expressions'][fn.name] = {
            'n_calls': len(exprs), 'forms': forms}

    # Also: anyone ELSE in the package that calls the kernel.
    callers = []
    for path in sorted(pkg.rglob('*.py')):
        try:
            src = path.read_text(encoding='cp1252')
        except UnicodeDecodeError:
            src = path.read_text(encoding='utf-8')
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)]:
            clean = strip_docstrings(fn)
            if '_exact_dispersion_phase(' in ast.unparse(clean):
                callers.append(
                    f"{str(path.relative_to(root)).replace(chr(92), '/')}"
                    f"::{fn.name}")
    report['kernel_callers'] = sorted(set(callers))

    with open(out, 'w', encoding='cp1252') as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print(f"functions walked: {nfun}")
    print("STRUCTURAL k^2 - ... owners:")
    for s in struct_owners:
        print(f"  {s['module']}::{s['function']}: {s['expressions']}")
    print("TOKEN owners:")
    for t, v in report['tokens'].items():
        print(f"  {t!r}: {v}")
    print("kernel callers:", report['kernel_callers'])
    for k, v in report['call_site_expressions'].items():
        print(f"  {k}: {v['n_calls']} call(s) -> {v['forms']}")
    print(f"-> {out}")


if __name__ == '__main__':
    main(*sys.argv[1:])

"""Second half of the carrier.py stacklevel sweep: the THREADED literals.

``warnings.warn`` sites were rewritten by ``_sweep_helper.py apply``.  What is
left is the chain that CARRIES a literal down to those sites: private helpers
whose signature default is a literal ``stacklevel``, and the call sites that
pass one explicitly.  Both are retargeted to ``None``, which the warning
helpers read as "compute it with ``caller_stacklevel()``".

The value is REPLACED rather than the keyword deleted: minimal diff, no
argument-list surgery, and every seam stays where it was for a caller that
genuinely wants a fixed frame.

``list`` prints what would change; ``apply`` writes it.
"""
import ast
import io
import sys

P = 'lumenairy/propagators/carrier.py'


def find(src):
    tree = ast.parse(src)
    spots = []                       # (lineno, col, end_col, what)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = n.args
            names = list(args.args) + list(args.kwonlyargs)
            defaults = ([None] * (len(args.args) - len(args.defaults))
                        + list(args.defaults) + list(args.kw_defaults))
            for a, d in zip(names, defaults):
                if (a.arg == 'stacklevel' and d is not None
                        and isinstance(d, ast.Constant)
                        and isinstance(d.value, int)):
                    spots.append((d.lineno, d.col_offset, d.end_col_offset,
                                  'def %s (default %r)' % (n.name, d.value)))
        elif isinstance(n, ast.Call):
            fn = n.func
            if isinstance(fn, ast.Attribute) and fn.attr == 'warn':
                continue             # already swept
            for k in n.keywords:
                if (k.arg == 'stacklevel'
                        and isinstance(k.value, ast.Constant)
                        and isinstance(k.value.value, int)):
                    name = getattr(fn, 'id', getattr(fn, 'attr', '?'))
                    spots.append((k.value.lineno, k.value.col_offset,
                                  k.value.end_col_offset,
                                  'call %s(stacklevel=%r)'
                                  % (name, k.value.value)))
    spots.sort()
    return spots


def main():
    mode = sys.argv[1]
    with io.open(P, encoding='cp1252', newline='') as fh:
        raw = fh.read()
    nl = '\r\n' if '\r\n' in raw else '\n'
    src = raw.replace('\r\n', '\n')
    spots = find(src)
    for ln, c, ec, what in spots:
        print('L%d:%d  %s' % (ln, c, what))
    print('total:', len(spots))
    if mode != 'apply':
        return 0
    lines = src.splitlines(keepends=True)
    for ln, c, ec, _w in sorted(spots, reverse=True):
        i = ln - 1
        lines[i] = lines[i][:c] + 'None' + lines[i][ec:]
    out = ''.join(lines)
    ast.parse(out)                   # refuse to write a file that will not parse
    with io.open(P, 'w', encoding='cp1252', newline='') as fh:
        fh.write(out.replace('\n', nl))
    print('rewrote', len(spots), 'threaded literals in', P)
    return 0


if __name__ == '__main__':
    sys.exit(main())

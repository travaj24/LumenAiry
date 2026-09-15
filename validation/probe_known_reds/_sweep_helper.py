"""Support script for the carrier.py stacklevel sweep.

``list``  -- print every ``warnings.warn`` call in a file with its line span
            and how its ``stacklevel`` is spelled.
``apply`` -- rewrite the literal ``stacklevel=<int>`` keyword of every
            ``warnings.warn`` call in the file to ``_caller_stacklevel()``.

The rewrite is done on the token/line level with the AST supplying the exact
keyword position, so nothing else in the file moves.
"""
import ast
import io
import sys


def warn_calls(src):
    tree = ast.parse(src)
    out = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'warn'
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == 'warnings'):
            continue
        kw = next((k for k in n.keywords if k.arg == 'stacklevel'), None)
        spelling = None
        if kw is not None:
            spelling = ('literal' if isinstance(kw.value, ast.Constant)
                        else 'computed')
        out.append({
            'lineno': n.lineno, 'end_lineno': n.end_lineno,
            'stacklevel_spelling': spelling,
            'stacklevel_value': (kw.value.value
                                 if kw is not None
                                 and isinstance(kw.value, ast.Constant)
                                 else None),
            'kw_lineno': kw.value.lineno if kw is not None else None,
            'kw_col': kw.value.col_offset if kw is not None else None,
            'kw_end_col': kw.value.end_col_offset if kw is not None else None,
        })
    out.sort(key=lambda r: r['lineno'])
    return out


def main():
    mode, path = sys.argv[1], sys.argv[2]
    src = io.open(path, encoding='cp1252', newline='').read()
    calls = warn_calls(src)
    if mode == 'list':
        for c in calls:
            print(f"L{c['lineno']}-{c['end_lineno']}  "
                  f"stacklevel={c['stacklevel_spelling']}"
                  f"({c['stacklevel_value']})  kw at "
                  f"{c['kw_lineno']}:{c['kw_col']}")
        print('total warn calls:', len(calls),
              ' literal:', sum(1 for c in calls
                               if c['stacklevel_spelling'] == 'literal'))
        return 0

    # apply: rewrite from the bottom up so earlier offsets stay valid
    lines = src.splitlines(keepends=True)
    targets = [c for c in calls if c['stacklevel_spelling'] == 'literal']
    for c in sorted(targets, key=lambda r: (r['kw_lineno'], r['kw_col']),
                    reverse=True):
        i = c['kw_lineno'] - 1
        ln = lines[i]
        before, after = ln[:c['kw_col']], ln[c['kw_end_col']:]
        lines[i] = before + '_caller_stacklevel()' + after
    io.open(path, 'w', encoding='cp1252', newline='').write(''.join(lines))
    print('rewrote', len(targets), 'literal stacklevel keywords in', path)
    return 0


if __name__ == '__main__':
    sys.exit(main())

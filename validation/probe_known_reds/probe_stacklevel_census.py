"""Census of ``warnings.warn`` stacklevel spellings across the package.

Reports, per module: how many ``warn`` calls pass a LITERAL integer
``stacklevel`` (positional or keyword) and how many pass a COMPUTED one.  The
b11 ratchet (``test_no_literal_stacklevel_is_left_in_the_swept_lens_bodies``)
covers the eight lens-family bodies; this census is how the sweep's next
members are chosen by measurement rather than by guess.
"""
import ast
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

SWEPT = (
    'lumenairy/elements/_lens_real.py',
    'lumenairy/elements/_lens_traced.py',
    'lumenairy/elements/lenses_maslov.py',
    'lumenairy/elements/lenses_gbd.py',
    'lumenairy/elements/_lens_traced_multibranch.py',
    'lumenairy/elements/_lens_thin.py',
    'lumenairy/elements/_lens_imap.py',
    'lumenairy/elements/_lens_traced_uniform.py',
)


def census(path):
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    lits, comp = [], []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'warn'):
            continue
        kw = [k.value for k in n.keywords if k.arg == 'stacklevel']
        for a in list(n.args) + kw:
            if isinstance(a, ast.Constant) and isinstance(a.value, int):
                lits.append(n.lineno)
                break
        for a in kw:
            if not isinstance(a, ast.Constant):
                comp.append(n.lineno)
    return sorted(set(lits)), sorted(set(comp))


def main():
    rows = []
    for p in sorted(REPO.glob('lumenairy/**/*.py')):
        rel = p.relative_to(REPO).as_posix()
        lits, comp = census(p)
        if lits or comp:
            rows.append({'file': rel, 'n_literal': len(lits),
                         'n_computed': len(comp),
                         'literal_lines': lits,
                         'swept_by_b11_ratchet': rel in SWEPT})
    rows.sort(key=lambda r: (-r['n_literal'], r['file']))
    out = {'repo': str(REPO), 'rows': rows,
           'total_literal': sum(r['n_literal'] for r in rows),
           'total_computed': sum(r['n_computed'] for r in rows),
           'literal_still_in_swept_files': [
               r['file'] for r in rows
               if r['swept_by_b11_ratchet'] and r['n_literal']]}
    print(f"{'lits':>5} {'comp':>5}  swept  file")
    for r in rows:
        if r['n_literal'] or r['n_computed']:
            print(f"{r['n_literal']:5d} {r['n_computed']:5d}  "
                  f"{'Y' if r['swept_by_b11_ratchet'] else '.':^5}  "
                  f"{r['file']}")
    print('TOTAL literal:', out['total_literal'],
          ' TOTAL computed:', out['total_computed'])
    tag = os.environ.get('PROBE_TAG', 'default')
    dest = pathlib.Path(__file__).parent / f'stacklevel_census_{tag}.json'
    dest.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('wrote', dest)
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Re-anchor every CHANGELOG ``path.py:N`` citation that WP-B11c moved.

A citation is re-anchored by CONTENT, not by arithmetic: the line the number
named at the base commit (``B11C_BASE``, default ``96cb2096``) is located in
the tree as it is now, with a +-2-line context check to break ties.  A citation
whose line left its file entirely is re-pointed at the module it moved to, path
and all.

WHY NOT ARITHMETIC, AND WHY NOT V18 ALONE.  A file that loses one line shifts
every citation below it by one.  ``scripts/check_source_line_citations.py``
(V18) refuses a citation that lands on a TRIVIAL line -- blank, a lone brace, a
docstring delimiter -- so it catches the shifts that happen to land badly and
PASSES the ones that land on some other real line.  Those are the dangerous
ones: they read as authoritative and they pass.  Matching the CONTENT is what
finds them.

IDEMPOTENT, and it has to be.  The tool sources its "before" numbers from
``git show <BASE>:CHANGELOG.md``, never from the working copy, so running it a
second time recomputes the same mapping and finds nothing left to replace.  A
version that read the working copy's numbers would treat the numbers it had
just written as base numbers and shift everything a second time -- which is
exactly the failure this docstring exists to stop someone repeating.

    python validation/probe_wp_b11c/reanchor_citations.py [--check]

``--check`` reports what it would do and writes nothing.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = os.environ.get('B11C_BASE', '96cb2096')

#: token tail -> (the file it names, where its content may have moved to)
OWNED = {
    '_lens_real.py': ('lumenairy/elements/_lens_real.py', []),
    'lenses.py': ('lumenairy/elements/lenses.py',
                  ['lumenairy/elements/_lens_kernels.py']),
    'lenses_maslov.py': ('lumenairy/elements/lenses_maslov.py', []),
    'rcwa/_core.py': ('lumenairy/elements/rcwa/_core.py',
                      ['lumenairy/elements/rcwa/_blas.py']),
}
_cache = {}


def lines(path, rev=None):
    key = (path, rev)
    if key not in _cache:
        if rev:
            out = subprocess.run(['git', 'show', f'{rev}:{path}'], cwd=REPO,
                                 capture_output=True, text=True,
                                 encoding='utf-8').stdout
        else:
            out = (REPO / path).read_text(encoding='utf-8')
        _cache[key] = out.splitlines()
    return _cache[key]


def locate(want, ctx, hay):
    """1-based line of ``want`` in ``hay``, ties broken by +-2 lines of ctx."""
    cands = [i for i, ln in enumerate(hay) if ln == want]
    if not cands:
        return None, 'absent'
    if len(cands) == 1:
        return cands[0] + 1, 'unique'
    scored = sorted(
        ((sum(1 for d in (-2, -1, 1, 2)
              if 0 <= i + d < len(hay) and hay[i + d] == ctx[2 + d]), i)
         for i in cands), reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None, f'ambiguous ({len(cands)} matches, no context winner)'
    return scored[0][1] + 1, f'context ({scored[0][0]}/4)'


def main():
    check = '--check' in sys.argv
    cl = REPO / 'CHANGELOG.md'
    txt = cl.read_text(encoding='utf-8')
    before = subprocess.run(['git', 'show', f'{BASE}:CHANGELOG.md'], cwd=REPO,
                            capture_output=True, text=True,
                            encoding='utf-8').stdout
    tokens = sorted(set(re.findall(r'[A-Za-z0-9_/.]*\.py:\d+', before)),
                    key=len, reverse=True)
    changed, notes = [], []
    for tok in tokens:
        tail, num = tok.rsplit(':', 1)
        num = int(num)
        owner = next((k for k in OWNED
                      if tail.endswith(k)
                      and not (k == 'lenses.py'
                               and tail.endswith('lenses_maslov.py'))), None)
        if owner is None:
            continue
        path, elsewhere = OWNED[owner]
        base = lines(path, BASE)
        if num - 1 >= len(base):
            notes.append(f'{tok}: beyond the base file -- left alone')
            continue
        want = base[num - 1]
        ctx = [base[num - 1 + d] if 0 <= num - 1 + d < len(base) else None
               for d in (-2, -1, 0, 1, 2)]
        new, how = locate(want, ctx, lines(path))
        if new is not None:
            if new != num and tok in txt:
                txt = txt.replace(tok, f'{tail}:{new}')
                changed.append(f'{tok} -> {tail}:{new}  [{how}]  '
                               f'{want.strip()[:44]!r}')
            continue
        for dest in elsewhere:
            new, how = locate(want, ctx, lines(dest))
            if new is not None:
                if tok in txt:
                    txt = txt.replace(tok, f'{dest}:{new}')
                    changed.append(f'{tok} -> {dest}:{new}  [moved, {how}]  '
                                   f'{want.strip()[:36]!r}')
                break
        else:
            notes.append(f'{tok}: {how} -- NEEDS A HUMAN  '
                         f'{want.strip()[:48]!r}')
    if not check:
        cl.write_text(txt, encoding='utf-8', newline='\n')
    for c in changed:
        print('  ' + c)
    print(f'{len(changed)} re-anchored' + ('  (--check: nothing written)'
                                           if check else ''))
    for n in notes:
        print('  !! ' + n)
    return 1 if notes else 0


if __name__ == '__main__':
    raise SystemExit(main())

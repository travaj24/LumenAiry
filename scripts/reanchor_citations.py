"""Re-anchor every CHANGELOG ``path.py:N`` citation that a refactor moved.

A citation is re-anchored by CONTENT, not by arithmetic: the line the number
named at the BASE commit (``--base``, default ``96cb2096``) is located in the
tree as it is now, with a +-2-line context check to break ties.  A citation
whose line left its file entirely is re-pointed at the module it moved to, path
and all.

WHY NOT ARITHMETIC, AND WHY NOT V18 ALONE.  A file that loses one line shifts
every citation below it by one.  ``scripts/check_source_line_citations.py``
(V18) refuses a citation that lands on a TRIVIAL line -- blank, a lone brace, a
docstring delimiter -- so it catches the shifts that happen to land badly and
PASSES the ones that land on some other real line.  Those are the dangerous
ones: they read as authoritative and they pass.  Matching the CONTENT is what
finds them.

MEASURED 2026-09-19 (VERIFY-WAVE5-HYGIENE2 V-D2), which is why this tool moved
out of ``validation/probe_wp_b11c/`` and into ``scripts/``: on
``refactor/wave5-hygiene-2`` V18 read ``ok=107 drift=0 total=107``, rc=0 on
both builds, while FIFTEEN citations in the 5.47.0 block pointed at the wrong
line and one re-anchored range (``mft.py:645-654``) had been shifted ``+34`` by
hand where its content had moved ``+38``.  The tool could not see any of them:
its ``OWNED`` map named four lens / rcwa files and its base commit was frozen at
``96cb2096``, so run on that branch it printed ``0 re-anchored`` -- a green that
meant nothing.  Both of those are now inputs: the base commit is ``--base``,
and a citation no longer has to be on a hand-kept list to be checked -- any
tail that resolves to exactly one module under ``lumenairy/`` is resolved
automatically (``OWNED`` is kept for the cross-file MOVES, which nothing can
infer).  MEASURED on the 5.47.0 block: the hand-kept map saw 19 citations, the
resolver sees all 107.

WHAT A CITATION IS, here.  Three spellings are re-anchored:

* ``path.py:N`` -- the ordinary single-line citation;
* ``path.py:N-M`` -- a RANGE.  Both endpoints are anchored independently by
  content, so a range that straddles an insertion is widened rather than
  slid.  (``mft.py:645-654 -> 649-658`` above is the case that made this
  necessary: an arithmetic shift moved both endpoints by the same amount and
  left the sentence's own reading outside the range.)
* a bare sibling ``:N`` appearing on the SAME line after an owned citation
  (``(`carrier.py:1700`, `:1736`)``) -- attributed to that citation's file.
  The bare form is the repo's own house style for a second site in the same
  module, and it is invisible to every regex that requires a ``.py``.

IDEMPOTENT, and it has to be.  The tool sources its "before" numbers from
``git show <BASE>:CHANGELOG.md``, never from the working copy, so running it a
second time recomputes the same mapping and finds nothing left to replace.  A
version that read the working copy's numbers would treat the numbers it had
just written as base numbers and shift everything a second time -- which is
exactly the failure this docstring exists to stop someone repeating.

    python scripts/reanchor_citations.py [--check] [--base REV] [--block NAME]

``--check`` reports what it would do and writes nothing (exit 1 if anything is
stale or needs a human, 0 if the citations are already right).  ``--block``
restricts the CITING region to one ``## [version]`` CHANGELOG block, which is
how ``tests/unit/test_v5_3_2_walker_source_line_citation.py`` gates a single
release's block without being held hostage to the rest of the file.
"""
import argparse
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The default base commit.  ``B11C_BASE`` is honoured for the WP-B11c
#: invocation this tool was written for; ``--base`` overrides both.
DEFAULT_BASE = os.environ.get('B11C_BASE', '96cb2096')

#: token tail -> (the file it names, where its content may have moved to).
#:
#: A citation is matched by the LONGEST tail that ends the token, so
#: ``lumenairy/propagators/carrier.py:12`` and a bare ``carrier.py:12`` both
#: resolve to the same file.  The second element lists modules a split may
#: have moved the content INTO; a citation whose line is absent from its own
#: file is re-pointed at the first of those that holds it, path and all.
OWNED = {
    '_lens_real.py': ('lumenairy/elements/_lens_real.py', []),
    'lenses.py': ('lumenairy/elements/lenses.py',
                  ['lumenairy/elements/_lens_kernels.py']),
    'lenses_maslov.py': ('lumenairy/elements/lenses_maslov.py', []),
    'rcwa/_core.py': ('lumenairy/elements/rcwa/_core.py',
                      ['lumenairy/elements/rcwa/_blas.py']),
    # Added 2026-09-19 (V-D2): the three propagator modules Wave 5 hygiene 2
    # moved lines in.  ``_bluestein.py`` has no citations in the 5.47.0 block
    # today; it is listed because H2-1 edits it and the next block will.
    'mft.py': ('lumenairy/propagators/mft.py', []),
    'carrier.py': ('lumenairy/propagators/carrier.py', []),
    '_bluestein.py': ('lumenairy/propagators/_bluestein.py', []),
}

#: ``path.py:N`` or ``path.py:N-M``.
TOKEN_RE = re.compile(r'[A-Za-z0-9_/.]*\.py:\d+(?:-\d+)?')
#: a bare ``:N`` sibling, e.g. ``(`carrier.py:1700`, `:1736`)``.
BARE_RE = re.compile(r'`:(\d+)`')

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


def locate(want, ctx, hay, prefer=None):
    """1-based line of ``want`` in ``hay``, ties broken by +-2 lines of ctx.

    ``prefer`` is the citation's own number: when the base content is STILL at
    that number the citation is already right, so identity wins over any tie
    break.  Without that rule a line the refactor did not touch but which has a
    twin elsewhere in the file (``kxg = kx0 + m_orders * (wl / period_x)`` at
    ``rcwa/_core.py:1578`` and ``:1667``) reads as ambiguous and the tool asks
    for a human where there is nothing to do.
    """
    cands = [i for i, ln in enumerate(hay) if ln == want]
    if not cands:
        return None, 'absent'
    if prefer is not None and (prefer - 1) in cands:
        return prefer, 'unchanged'
    if len(cands) == 1:
        return cands[0] + 1, 'unique'
    scored = sorted(
        ((sum(1 for d in (-2, -1, 1, 2)
              if 0 <= i + d < len(hay) and hay[i + d] == ctx[2 + d]), i)
         for i in cands), reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None, f'ambiguous ({len(cands)} matches, no context winner)'
    return scored[0][1] + 1, f'context ({scored[0][0]}/4)'


#: a ``def`` / ``class`` statement, for the signature-change fallback below.
_DEF_RE = re.compile(r'^\s*(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)')


def locate_def(want, hay, prefer=None):
    """Fallback for a citation whose line is a DEFINITION whose text changed.

    A citation names a function; a refactor that adds a parameter to it changes
    the line's text and the exact-content match goes ``absent``.  The
    definition is still the same definition, so it is re-anchored on the
    ``def NAME`` / ``class NAME`` prefix instead.  MEASURED 2026-09-19: this is
    the only reason ``carrier.py:1626`` (``_collins_axis_chirp``, which gained
    ``bld=np``) and ``carrier.py:1906``
    (``_collins_exact_kernel_correction``, which gained ``(xp, is_jax, bld)``)
    could not be anchored by content -- H2-2 changed both signatures.
    """
    m = _DEF_RE.match(want)
    if not m:
        return None, 'absent'
    kind, name = m.group(1), m.group(2)
    pat = re.compile(rf'^\s*(?:async\s+)?{kind}\s+{re.escape(name)}\b')
    cands = [i for i, ln in enumerate(hay) if pat.match(ln)]
    if not cands:
        return None, 'absent'
    if prefer is not None and (prefer - 1) in cands:
        return prefer, 'unchanged'
    if len(cands) > 1:
        return None, f'ambiguous ({len(cands)} definitions of {name!r})'
    return cands[0] + 1, f'signature changed ({kind} {name})'


def owner_of(tail):
    """The ``OWNED`` key a citation tail belongs to, longest tail wins.

    ``OWNED`` is consulted first because it is the only place a cross-file
    MOVE can be declared.  Anything it does not name is resolved by finding
    the basename under ``lumenairy/``: a citation is checkable whenever its
    file can be identified unambiguously, and restricting the check to a
    hand-kept list is what left fifteen citations unwatched (V-D2).  A
    basename that matches two shipped modules is NOT resolved -- the citation
    does not say which, so neither does this tool.
    """
    best = None
    for k in OWNED:
        if not tail.endswith(k):
            continue
        # ``lenses_maslov.py`` ends with neither ``lenses.py`` nor
        # ``/lenses.py``; guard the one basename pair that would collide.
        if k == 'lenses.py' and tail.endswith('lenses_maslov.py'):
            continue
        if best is None or len(k) > len(best):
            best = k
    if best is not None:
        return best
    return _auto_owner(tail)


_AUTO = {}


def _auto_owner(tail):
    """Resolve a citation tail to a unique shipped module, and memoise it.

    Returns the ``OWNED`` key it registered (the tail itself), or None when
    the basename is absent from ``lumenairy/`` or matches more than one file.
    """
    if tail in _AUTO:
        return _AUTO[tail]
    base = tail.rsplit('/', 1)[-1]
    if not base.endswith('.py'):
        _AUTO[tail] = None
        return None
    hits = sorted(p for p in (REPO / 'lumenairy').rglob(base)
                  if p.is_file() and '__pycache__' not in p.parts)
    # A citation that spells part of its path pins the match further.
    if '/' in tail:
        want = tail.replace('\\', '/')
        hits = [p for p in hits
                if p.as_posix().endswith(want)] or hits
    if len(hits) != 1:
        _AUTO[tail] = None
        return None
    OWNED[tail] = (hits[0].relative_to(REPO).as_posix(), [])
    _AUTO[tail] = tail
    return tail


def base_line(path, base_num, base):
    """``(want, ctx)`` -- the base commit's content at ``path:base_num``."""
    src = lines(path, base)
    if not (1 <= base_num <= len(src)):
        return None, None
    ctx = [src[base_num - 1 + d] if 0 <= base_num - 1 + d < len(src) else None
           for d in (-2, -1, 0, 1, 2)]
    return src[base_num - 1], ctx


def anchor_one(path, base_num, base, target=None):
    """``(new_num, how)`` for one base line number, sought in ``target``.

    ``target`` defaults to ``path``; a split names the module the content moved
    INTO.  The "before" text always comes from ``path`` at ``base`` -- reading
    it from the destination instead is the bug that let a 5.2.0-era
    ``elements/lenses.py:322-810`` citation "re-anchor" onto whatever happened
    to sit at those same numbers in ``_lens_kernels.py``.
    """
    want, ctx = base_line(path, base_num, base)
    if want is None:
        return None, 'beyond the base file'
    hay = lines(target or path)
    prefer = base_num if target in (None, path) else None
    new, how = locate(want, ctx, hay, prefer=prefer)
    if new is None and how == 'absent':
        new, how = locate_def(want, hay, prefer=prefer)
    return new, how


def _block_span(text, block):
    """``(start, end)`` character offsets of one ``## [version]`` block."""
    heads = [m.start() for m in re.finditer(r'(?m)^## \[', text)]
    for i, s in enumerate(heads):
        head = text[s:text.find('\n', s)]
        if block in head:
            return s, (heads[i + 1] if i + 1 < len(heads) else len(text))
    raise SystemExit(f"no CHANGELOG block matching {block!r}")


#: every citation spelling this tool owns, in one alternation, so a line's
#: citations are read (and rewritten) left to right in one pass.
CITE_RE = re.compile(r'[A-Za-z0-9_/.]*\.py:\d+(?:-\d+)?|`:\d+`')


def skeleton(line):
    """The citing SENTENCE with every citation number blanked.

    Two CHANGELOG lines have the same skeleton exactly when they are the same
    prose citing the same files -- whatever numbers they carry.  That is what
    lets this tool find the line a citation lives on TODAY even when someone
    has already re-anchored it BY HAND to a different (possibly wrong) number,
    which the previous ``txt.replace(base_token, ...)`` spelling could not:
    the base token was simply absent and the replacement was a silent no-op.
    MEASURED 2026-09-19: that is precisely how ``mft.py:611-620`` became
    ``645-654`` (``+34``) where its content had moved ``+38``, and why running
    the tool afterwards reported nothing wrong.
    """
    return CITE_RE.sub(lambda m: re.sub(r'\d+', '#', m.group(0)), line)


def _new_number(tok, path, elsewhere, base):
    """``(replacement_token, how)`` for one citation, or ``(None, why)``."""
    if tok.startswith('`'):                       # bare sibling ``:N``
        tail, ends = None, [int(tok[2:-1])]
    else:
        tail, nums = tok.rsplit(':', 1)
        ends = [int(p) for p in nums.split('-')]
    got = [anchor_one(path, n, base) for n in ends]
    if all(g[0] is not None for g in got):
        nums = '-'.join(str(g[0]) for g in got)
        return (f'`:{nums}`' if tail is None else f'{tail}:{nums}'), got[0][1]
    for dest in elsewhere:
        alt = [anchor_one(path, n, base, target=dest) for n in ends]
        if all(a[0] is not None for a in alt):
            nums = '-'.join(str(a[0]) for a in alt)
            return (f'`:{nums}`' if tail is None else f'{dest}:{nums}'), \
                f'moved, {alt[0][1]}'
    bad = next(i for i, g in enumerate(got) if g[0] is None)
    return None, f'endpoint {ends[bad]} {got[bad][1]}'


def reanchor(base=DEFAULT_BASE, block=None):
    """Return ``(new_changelog_text, changed, notes)``.

    Pure: writes nothing.  ``changed`` is the list of re-anchorings the current
    file still needs; an empty ``changed`` with an empty ``notes`` IS the
    CONTENT check -- every owned citation names the line whose content the base
    commit's citation named.  That is the statement V18 does not make: V18 asks
    whether the cited line is non-trivial, this asks whether it is the RIGHT
    line.
    """
    cl = REPO / 'CHANGELOG.md'
    txt = cl.read_text(encoding='utf-8')
    before = subprocess.run(['git', 'show', f'{base}:CHANGELOG.md'], cwd=REPO,
                            capture_output=True, text=True,
                            encoding='utf-8').stdout
    if not before:
        raise SystemExit(f"git show {base}:CHANGELOG.md produced nothing")
    lo, hi = (0, len(before)) if block is None else _block_span(before, block)
    base_lines = before[lo:hi].splitlines()
    cur_lines = txt.splitlines()
    index = {}
    for i, ln in enumerate(cur_lines):
        index.setdefault(skeleton(ln), []).append(i)
    used = set()
    changed, notes = [], []
    for bline in base_lines:
        toks = CITE_RE.findall(bline)
        # a bare ``:N`` is attributed to the last named file on its own line
        owners, last = [], None
        for t in toks:
            if not t.startswith('`'):
                last = owner_of(t.rsplit(':', 1)[0])
            owners.append(last)
        if not any(o for o in owners):
            continue
        sk = skeleton(bline)
        cands = [i for i in index.get(sk, []) if i not in used]
        if len(cands) != 1:
            if len(cands) == 0 and sk not in index:
                notes.append(f'the citing sentence moved or was rewritten, so '
                             f'its citations cannot be checked -- NEEDS A '
                             f'HUMAN: {bline.strip()[:70]!r}')
            continue
        tgt = cands[0]
        used.add(tgt)
        out, pos = [], 0
        line = cur_lines[tgt]
        for cm, tok, owner in zip(CITE_RE.finditer(line), toks, owners):
            out.append(line[pos:cm.start()])
            pos = cm.end()
            if owner is None:
                out.append(cm.group(0))
                continue
            path, elsewhere = OWNED[owner]
            rep, how = _new_number(tok, path, elsewhere, base)
            if rep is None:
                notes.append(f'{tok}: {how} -- NEEDS A HUMAN')
                out.append(cm.group(0))
                continue
            if rep != cm.group(0):
                changed.append(f'{cm.group(0)} -> {rep}  [{how}'
                               + (f', was {tok} at {base[:8]}'
                                  if tok != cm.group(0) else '') + ']')
            out.append(rep)
        out.append(line[pos:])
        cur_lines[tgt] = ''.join(out)
    end = '\n' if txt.endswith('\n') else ''
    return '\n'.join(cur_lines) + end, changed, notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true',
                    help='report only; write nothing')
    ap.add_argument('--base', default=DEFAULT_BASE,
                    help=f'commit whose CHANGELOG line numbers are the '
                         f'"before" (default {DEFAULT_BASE})')
    ap.add_argument('--block', default=None,
                    help="restrict to one '## [version]' block, e.g. '[5.47.0]'")
    args = ap.parse_args(argv)
    cl = REPO / 'CHANGELOG.md'
    # Preserve the working tree's OWN line endings.  On a core.autocrlf=true
    # checkout CHANGELOG.md is CRLF, and a blanket newline='\n' write rewrote
    # every line of it -- invisible to ``git diff`` (which normalises) and
    # visible to every byte-level tool that reads the working copy.
    raw = cl.read_bytes()
    eol = '\r\n' if raw.count(b'\r\n') * 2 > raw.count(b'\n') else '\n'
    txt, changed, notes = reanchor(base=args.base, block=args.block)
    if not args.check:
        cl.write_bytes(txt.replace('\n', eol).encode('utf-8'))
    for c in changed:
        print('  ' + c)
    print(f'{len(changed)} re-anchored'
          + ('  (--check: nothing written)' if args.check else ''))
    for n in notes:
        print('  !! ' + n)
    if args.check:
        return 1 if (changed or notes) else 0
    return 1 if notes else 0


if __name__ == '__main__':
    raise SystemExit(main())

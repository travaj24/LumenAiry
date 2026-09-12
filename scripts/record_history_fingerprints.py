#!/usr/bin/env python3
"""record_history_fingerprints.py -- re-record a history document's fingerprints.

Maintenance companion for
``tests/unit/test_audit2609_a17_history_relocation.py``.

That gate pins every module with a document under ``docs/history/`` to the AST
and token fingerprints recorded when its version-history narrative was moved
out of the source.  The pin is the point: it proves a "documentation-only" edit
really was documentation-only.  But it also means the gate goes red on the next
*legitimate* code change to one of those modules, and a red gate with no
sanctioned way to clear it is a gate people learn to edit by hand -- which is
how a recorded hash silently stops meaning anything.

This script is the sanctioned way.

    # what drifted, and by which fingerprint?  Writes nothing, exits 1 on
    # drift.  With no target it checks every document under docs/history/.
    python scripts/record_history_fingerprints.py --check

    # re-record one module, in the SAME commit as the change that moved it
    python scripts/record_history_fingerprints.py lumenairy/propagators/fft_infra.py \\
        --reason "scipy.fft made lazy behind find_spec + first-use accessor"

THE RULE, which ``CONTRIBUTING.md`` also states: a commit that intentionally
changes code in a module with a history document re-records that document's
fingerprints IN THE SAME COMMIT and says why.  The ``re_recorded:`` lines this
script appends are what makes the trail explicit -- a reviewer reading the
header sees every time the baseline moved and the reason given each time,
instead of a single hash of unknown vintage.

Both fingerprints are computed by importing the checker's own
``ast_fingerprint`` / ``token_fingerprint``.  Re-implementing them here would
put the recorder and the gate on two independent definitions of "unchanged",
and the first time they disagreed the recorder would write a hash the gate
rejects.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import pathlib
import re
import sys

_THIS = pathlib.Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
_CHECKER = (_REPO_ROOT / 'tests' / 'unit'
            / 'test_audit2609_a17_history_relocation.py')

#: the header block every history document opens with.
_HEADER_RE = re.compile(r'<!--\s*lumenairy-history-doc\s*(?P<body>.*?)-->',
                        re.S)


# ---------------------------------------------------------------------------
# the checker's own helpers
# ---------------------------------------------------------------------------
def load_checker(path: pathlib.Path | None = None):
    """Import the relocation checker BY PATH and return the module.

    ``tests/`` is not a package on ``sys.path`` and making it one to satisfy an
    import would put test modules into the library's import namespace.  Loading
    by path also guarantees this script and the gate share one definition of
    each fingerprint, which is the only reason the recorded values mean
    anything.
    """
    path = path or _CHECKER
    if not path.is_file():
        raise SystemExit(
            f'{path} is missing.  This script computes its fingerprints with '
            f'the checker\'s own helpers; without the checker there is '
            f'nothing to be consistent with.')
    spec = importlib.util.spec_from_file_location(
        '_history_relocation_checker', path)
    module = importlib.util.module_from_spec(spec)
    # The checker imports pytest at module scope for its parametrisation.
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# document discovery and header parsing
# ---------------------------------------------------------------------------
def history_dir(root: pathlib.Path) -> pathlib.Path:
    return root / 'docs' / 'history'


def all_documents(root: pathlib.Path = _REPO_ROOT) -> list[pathlib.Path]:
    hdir = history_dir(root)
    if not hdir.is_dir():
        return []
    return [md for md in sorted(hdir.glob('*.md'))
            if md.name.upper() != 'README.MD']


def parse_header(md_path: pathlib.Path) -> dict[str, str]:
    """The header as a dict.  Repeated keys keep the LAST value, which is what
    the checker does too -- that is why an appended ``re_recorded:`` line is
    additive rather than a rewrite."""
    match = _HEADER_RE.search(md_path.read_text(encoding='utf-8'))
    if match is None:
        raise SystemExit(
            f'{md_path}: no `lumenairy-history-doc` header block, so there is '
            f'nothing to re-record.  Every document under docs/history/ must '
            f'declare its module and that module\'s fingerprints.')
    out: dict[str, str] = {}
    for line in match.group('body').splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        key, _, value = line.partition(':')
        out[key.strip()] = value.strip()
    return out


def repo_root_of(md_path: pathlib.Path) -> pathlib.Path:
    """The tree a document belongs to: ``<root>/docs/history/<doc>.md``.

    Derived from the document rather than assumed to be this checkout, so the
    tool can be pointed at a copy of the tree -- which is exactly how
    ``tests/unit/test_audit2609_a22_history_fingerprint_tool.py`` exercises it
    without touching the real documents.
    """
    return md_path.resolve().parents[2]


def resolve(target: str, root: pathlib.Path = _REPO_ROOT) -> pathlib.Path:
    """Accept any of the four spellings a caller is likely to have to hand.

    ``docs/history/fft_infra.md`` (the document), ``fft_infra`` (its stem),
    ``lumenairy/propagators/fft_infra.py`` (the module), or
    ``lumenairy.propagators.fft_infra`` (the dotted module).  Returns the
    document path.
    """
    candidate = pathlib.Path(target)
    if candidate.suffix == '.md' and candidate.is_file():
        return candidate.resolve()

    docs = all_documents(root)
    if not docs:
        raise SystemExit(f'{history_dir(root)} holds no history documents.')

    # by document stem
    for md in docs:
        if md.stem == target:
            return md

    # by module path / dotted module, read off each document's own header
    wanted = target.replace('\\', '/')
    dotted = wanted[:-3].replace('/', '.') if wanted.endswith('.py') else wanted
    for md in docs:
        module = parse_header(md).get('module', '').replace('\\', '/')
        if not module:
            continue
        if module == wanted or module[:-3].replace('/', '.') == dotted:
            return md

    raise SystemExit(
        f'no history document matches {target!r}.  Known documents: '
        + ', '.join(md.stem for md in docs)
        + '.  A module only has fingerprints once its history has been '
          'relocated; if this module has no document, there is nothing to '
          're-record.')


# ---------------------------------------------------------------------------
# the work
# ---------------------------------------------------------------------------
def measure(md_path: pathlib.Path, checker) -> dict[str, object]:
    """Recorded vs live fingerprints for one document."""
    header = parse_header(md_path)
    root = repo_root_of(md_path)
    module_rel = header.get('module')
    if not module_rel:
        raise SystemExit(f'{md_path}: header has no `module:` line.')
    src_path = root / module_rel
    if not src_path.is_file():
        raise SystemExit(
            f'{md_path}: `module: {module_rel}` does not resolve under '
            f'{root}.  Fix the header (or the move) before re-recording.')
    source = src_path.read_text(encoding='utf-8')
    return {
        'doc': md_path,
        'module': module_rel,
        'src': src_path,
        'recorded_ast': header.get('ast_sha256', ''),
        'recorded_token': header.get('token_sha256', ''),
        'live_ast': checker.ast_fingerprint(source),
        'live_token': checker.token_fingerprint(source),
    }


def drifted(row: dict[str, object]) -> list[str]:
    out = []
    if row['recorded_ast'] != row['live_ast']:
        out.append('ast')
    if row['recorded_token'] != row['live_token']:
        out.append('token')
    return out


def rewrite_header(md_path: pathlib.Path, row: dict[str, object],
                   reason: str, today: str) -> str:
    """Replace the two digests in place and append one ``re_recorded:`` line.

    In place, line by line, rather than by re-emitting the header: the header
    also carries ``pre_relocation_lines``, ``recorded_by`` and ``checker``,
    and a regenerating writer would have to know about every one of them --
    including the ones a later work package adds.
    """
    text = md_path.read_text(encoding='utf-8')
    match = _HEADER_RE.search(text)
    if match is None:                                    # pragma: no cover
        raise SystemExit(f'{md_path}: header vanished between read and write')

    body = match.group('body')
    newline = '\r\n' if '\r\n' in body else '\n'
    lines = body.split(newline)

    def _set(key: str, value: str) -> None:
        for i, line in enumerate(lines):
            if line.strip().startswith(f'{key}:'):
                indent = line[:len(line) - len(line.lstrip())]
                lines[i] = f'{indent}{key}: {value}'
                return
        raise SystemExit(
            f'{md_path}: header has no `{key}:` line to update.')

    _set('ast_sha256', str(row['live_ast']))
    _set('token_sha256', str(row['live_token']))

    # Append the audit-trail line just after the last non-blank header line, so
    # repeated re-records read top to bottom in the order they happened.
    entry = f're_recorded: {today} -- {reason}'
    last = max((i for i, line in enumerate(lines) if line.strip()),
               default=len(lines) - 1)
    lines.insert(last + 1, entry)

    new_body = newline.join(lines)
    return text[:match.start('body')] + new_body + text[match.end('body'):]


def run(targets: list[str], *, check: bool, reason: str | None,
        root: pathlib.Path = _REPO_ROOT, checker=None,
        today: str | None = None, stream=sys.stdout) -> int:
    checker = checker or load_checker()
    today = today or datetime.date.today().isoformat()

    docs = (all_documents(root) if not targets
            else [resolve(t, root) for t in targets])
    if not docs:
        print(f'no history documents under {history_dir(root)}', file=stream)
        return 1

    any_drift = False
    for md in docs:
        try:
            row = measure(md, checker)
        except SystemExit as exc:
            # One malformed document must not abort the sweep: over a whole
            # directory the useful output is EVERY problem, not the first one.
            # It is still a failure -- a header the recorder cannot read is a
            # header the gate cannot read either.
            any_drift = True
            print(f'BAD   {md.name:52s} {exc}', file=stream)
            continue
        moved = drifted(row)
        if not moved:
            print(f'OK    {md.name:52s} {row["module"]}', file=stream)
            continue
        any_drift = True
        print(f'DRIFT {md.name:52s} {row["module"]}  ({", ".join(moved)})',
              file=stream)
        for kind in moved:
            print(f'        recorded {kind:5s} {row["recorded_" + kind]}',
                  file=stream)
            print(f'        live     {kind:5s} {row["live_" + kind]}',
                  file=stream)
        if check:
            continue
        md.write_text(rewrite_header(md, row, reason or '', today),
                      encoding='utf-8')
        print(f'        re-recorded, reason: {reason}', file=stream)

    if check and any_drift:
        print('\nDrift found.  If the code change was deliberate, re-record '
              'in the SAME commit:\n'
              '  python scripts/record_history_fingerprints.py <module> '
              '--reason "<what changed and why>"\n'
              'If it was not, the change is a behaviour change hiding in a '
              'documentation-only edit -- that is what the gate is for.',
              file=stream)
        return 1
    if check:
        print('\nOK: every history document matches its module.', file=stream)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=('Re-record (or check) the AST and token fingerprints a '
                     'history document pins its module to.'))
    parser.add_argument(
        'target', nargs='*',
        help=('Module path (lumenairy/propagators/fft_infra.py), dotted '
              'module, document stem (fft_infra) or document path.  Omit to '
              'act on every document under docs/history/.'))
    parser.add_argument(
        '--check', action='store_true',
        help=('Report drift and exit 1 without writing anything.  This is the '
              'mode for CI and for finding out what a change touched.'))
    parser.add_argument(
        '--reason', default=None,
        help=('One line saying what changed and why -- appended to the header '
              'as a `re_recorded:` entry.  Required when writing: a '
              're-recorded fingerprint with no reason is indistinguishable '
              'from someone silencing the gate.'))
    parser.add_argument(
        '--root', default=None,
        help='Repository root (default: the tree this script lives in).')
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root).resolve() if args.root else _REPO_ROOT
    if not args.check and not args.reason:
        parser.error('--reason is required when re-recording; pass --check to '
                     'report drift without writing.')
    return run(args.target, check=args.check, reason=args.reason, root=root)


if __name__ == '__main__':
    sys.exit(main())

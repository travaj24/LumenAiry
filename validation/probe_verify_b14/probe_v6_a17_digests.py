"""VERIFY-B14 arm 6b -- the PEP 701 digest scheme, read on two interpreters.

The claim under test is that the re-recorded ``docs/history`` token digests are
INTERPRETER-INDEPENDENT.  This computes both fingerprints for every registered
module with the shipped helpers, compares each against what the document
records, and dumps the pair so the same run on a second interpreter can be
diffed hash-for-hash.

A second, independent reading is taken alongside it: an f-string's contribution
to the digest is recomputed here from ``ast`` node spans rather than from the
tokenizer's ``FSTRING_*`` run, so the two routes to "one record carrying the
source slice" can be compared without sharing the shipped implementation's
record loop.  Run with no pytest in the picture, so the digest is read on the
interpreter itself.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import pathlib
import sys
import tokenize

ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTMOD = ROOT / 'tests' / 'unit' / 'test_audit2609_a17_history_relocation.py'


def _load_helpers():
    spec = importlib.util.spec_from_file_location('_a17_helpers', TESTMOD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_a17_helpers'] = mod
    spec.loader.exec_module(mod)
    return mod


def _independent_fstring_slices(src):
    """Every f-string / t-string literal's exact SOURCE SLICE, found through
    ``ast`` rather than through the tokenizer -- the second route to the same
    quantity.  Nested constructs are reported once, outermost only."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    spans = []
    for node in ast.walk(tree):
        cls = type(node).__name__
        if cls not in ('JoinedStr', 'TemplateStr'):
            continue
        if node.lineno is None or node.col_offset is None:
            continue
        spans.append((node.lineno, node.col_offset,
                      node.end_lineno, node.end_col_offset))
    spans.sort()
    kept = []
    for s in spans:
        if kept and (s[0], s[1]) >= (kept[-1][0], kept[-1][1]) and \
                (s[2], s[3]) <= (kept[-1][2], kept[-1][3]):
            continue
        kept.append(s)
    out = []
    for (l0, c0, l1, c1) in kept:
        if l0 == l1:
            out.append(lines[l0 - 1][c0:c1])
        else:
            buf = [lines[l0 - 1][c0:]]
            buf += lines[l0:l1 - 1]
            buf.append(lines[l1 - 1][:c1])
            out.append(''.join(buf))
    return out


def _tokenizer_fstring_slices(src):
    """The same quantity taken the shipped way: the source text spanned by
    each top-level ``FSTRING_START`` .. ``FSTRING_END`` run."""
    lines = src.splitlines(keepends=True)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    opens = {'FSTRING_START', 'TSTRING_START'}
    closes = {'FSTRING_END', 'TSTRING_END'}
    out, depth, start = [], 0, None
    for t in toks:
        nm = tokenize.tok_name.get(t.type, '')
        if nm in opens:
            if depth == 0:
                start = t.start
            depth += 1
        elif nm in closes:
            depth -= 1
            if depth == 0 and start is not None:
                (l0, c0), (l1, c1) = start, t.end
                if l0 == l1:
                    out.append(lines[l0 - 1][c0:c1])
                else:
                    buf = [lines[l0 - 1][c0:]]
                    buf += lines[l0:l1 - 1]
                    buf.append(lines[l1 - 1][:c1])
                    out.append(''.join(buf))
                start = None
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'v6_a17.json'
    H = _load_helpers()
    res = {'python': sys.version.split()[0], 'root': str(ROOT),
           'n_registered': len(H._REGISTRY), 'modules': {},
           'ast_mismatch': [], 'token_mismatch': [],
           'fstring_route_mismatch': [], 'n_with_fstrings': 0,
           'n_fstring_nodes': 0}
    for name, md, header in H._REGISTRY:
        name = header['module']
        path = ROOT / name
        src = path.read_text(encoding='utf-8')
        a = H.ast_fingerprint(src)
        t = H.token_fingerprint(src)
        res['modules'][name] = {
            'ast': a, 'token': t,
            'ast_recorded': header.get('ast_sha256'),
            'token_recorded': header.get('token_sha256')}
        if a != header.get('ast_sha256'):
            res['ast_mismatch'].append(name)
        if t != header.get('token_sha256'):
            res['token_mismatch'].append(name)
        try:
            via_ast = _independent_fstring_slices(src)
            via_tok = _tokenizer_fstring_slices(src)
        except Exception as exc:                          # pragma: no cover
            res['fstring_route_mismatch'].append(
                name + ' ERR ' + type(exc).__name__ + ': ' + str(exc)[:80])
            continue
        if via_ast:
            res['n_with_fstrings'] += 1
            res['n_fstring_nodes'] += len(via_ast)
        if sys.version_info >= (3, 12) and via_ast != via_tok:
            res['fstring_route_mismatch'].append(
                '%s: ast=%d tok=%d' % (name, len(via_ast), len(via_tok)))
    digest_of_all = hashlib.sha256(
        json.dumps({k: (v['ast'], v['token'])
                    for k, v in sorted(res['modules'].items())},
                   sort_keys=True).encode()).hexdigest()
    res['digest_of_all_fingerprints'] = digest_of_all
    print('python', res['python'], 'registered', res['n_registered'])
    print('ast mismatches   ', len(res['ast_mismatch']))
    print('token mismatches ', len(res['token_mismatch']))
    print('modules with f-strings', res['n_with_fstrings'],
          'nodes', res['n_fstring_nodes'])
    print('f-string route mismatches (ast span vs FSTRING run):',
          len(res['fstring_route_mismatch']))
    for m in res['fstring_route_mismatch'][:5]:
        print('   ', m)
    print('digest of all fingerprints', digest_of_all)
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()

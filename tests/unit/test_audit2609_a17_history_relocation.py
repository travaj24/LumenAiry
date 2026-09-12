"""WP-A17 -- the version-history relocation must be provably behaviour-free.

Audit 2026-09-11, finding P2-4 (``TESTS-ARCH.md``) / consolidated report
sec. 14 V6 and sec. 15.7: six lens/carrier modules carried 12 484 lines
(32.3 %) of *git history* -- blocks shaped "v5.xx (audit X): pre-fix this did
A, which was wrong because B; now it does C".  WP-A17 moves those blocks
verbatim into ``docs/history/<module>.md`` and leaves a one-line pointer in
the code wherever the rationale is load-bearing for the CURRENT behaviour.

This file is the gate on that move.  A history relocation is allowed to change
comments and string literals that are *statements* (docstrings, and the
"string-as-comment" form); it is not allowed to change anything the interpreter
executes.  Two independent fingerprints pin that:

* **AST fingerprint** -- the module's abstract syntax tree with every
  string-expression *statement* removed (so docstrings are invisible to it) and
  with source positions ignored.  Comments never reach the AST, so this alone
  says nothing about them; it is the check that catches a moved, deleted,
  reordered or re-spelled *statement*.
* **token fingerprint** -- the ``tokenize`` stream reduced to NAME / OP /
  NUMBER / STRING (and the 3.12+ f-string token trio), with COMMENT tokens and
  the STRING tokens belonging to those same string-statements dropped.  This is
  the check that catches what the AST normalises away: the *spelling* of a
  literal (``1e3`` vs ``1000.0``, ``'a'`` vs ``"a"``, an implicit
  concatenation), and it re-checks statement identity through a completely
  different front end.

Both fingerprints were recorded from the PRE-relocation file and are stored in
the header of the module's history document, so this test compares today's
source against the code as it stood before a single history block moved.  An
edit that changes behaviour while claiming to be history-only fails here.

THE MAINTENANCE RULE, and the reason this file is not a trap.  A pin with no
expiry goes red on the first DELIBERATE code change to one of these modules --
which is correct, and would be unworkable if the only way to clear it were to
hand-edit a hash.  So: **a commit that intentionally changes code in a module
with a history document re-records that document's fingerprints in the same
commit, and says why.**  Not in a follow-up: a tree whose fingerprint and code
disagree cannot tell a deliberate change from an accidental one, which is the
entire value of the pin.  ``scripts/record_history_fingerprints.py`` does the
re-record with the helpers below (``--check`` reports drift without writing;
a write requires ``--reason``, which it appends to the header as a
``re_recorded:`` line so the trail of baseline moves is explicit).  The rule
and the commands are also in ``CONTRIBUTING.md``, and the recorder has its own
gate at ``tests/unit/test_audit2609_a22_history_fingerprint_tool.py``.

The registry is discovered from ``docs/history/*.md``, so WP-A17 part 2
(``_lens_traced.py``, ``_lens_real.py``, ``_lens_imap.py``,
``lenses_maslov.py``) extends this test by adding its documents -- no change to
this file is required.

TESTING_STANDARDS: no wall-clock or resource assertion; every bar here is an
exact equality between two hashes of the same file, so there is no tolerance to
derive.  ``test_the_fingerprints_are_actually_sensitive`` is the falsifiability
check the audit asked for (V1): it mutates a copy of each module in the four
ways a "documentation-only" edit could hide a real change and asserts that at
least one fingerprint moves.
"""

from __future__ import annotations

import ast
import hashlib
import io
import pathlib
import re
import tokenize

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HISTORY_DIR = REPO_ROOT / "docs" / "history"

#: token categories that carry meaning for the interpreter.  COMMENT, NL,
#: NEWLINE, INDENT, DEDENT, ENCODING and ENDMARKER are all excluded: the first
#: is what WP-A17 removes, the rest move whenever a block of prose is deleted.
_MEANING_TOKENS = frozenset({
    "NAME", "OP", "NUMBER", "STRING",
    "FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END",
})

_MISSING = object()


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------
def _string_statement_lines(tree: ast.AST) -> list[tuple[int, int]]:
    """Line spans of every string-literal used as a *statement*.

    That is the docstring form (``body[0]``) and the "string as a comment"
    form that appears mid-body in this codebase.  Returned as inclusive
    ``(first, last)`` source line pairs.
    """
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            spans.append((node.lineno, node.end_lineno or node.lineno))
    return spans


def _strip_string_statements(tree: ast.AST) -> ast.AST:
    """Delete every string-statement from every body, in place.

    A body left empty by the deletion gets a ``Pass`` so the tree stays legal.
    Deleting rather than blanking is deliberate: it makes the fingerprint
    indifferent to a docstring being *shortened* or *removed outright*, which
    are both legitimate outcomes of a history move.
    """
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            kept = [
                st for st in body
                if not (isinstance(st, ast.Expr)
                        and isinstance(st.value, ast.Constant)
                        and isinstance(st.value.value, str))
            ]
            if len(kept) != len(body):
                body[:] = kept or [ast.Pass()]
    return tree


def _normalise(node: object) -> str:
    """A position-free, version-tolerant rendering of an AST.

    ``ast.dump`` is not used directly because its output gains fields across
    CPython releases (``type_params`` in 3.12, for one), which would invalidate
    a recorded hash on an interpreter upgrade rather than on a real change.
    Fields that are absent, ``None`` or an empty list are skipped, so a new
    always-empty field is invisible; everything that carries a value is
    rendered.  Attributes (``lineno``, ``col_offset``) are never rendered, so
    deleting 3 900 lines of prose does not move the fingerprint.
    """
    if isinstance(node, ast.AST):
        parts = []
        for field in node._fields:
            value = getattr(node, field, _MISSING)
            if value is _MISSING or value is None:
                continue
            if isinstance(value, list) and not value:
                continue
            parts.append(f"{field}={_normalise(value)}")
        return f"{type(node).__name__}({','.join(parts)})"
    if isinstance(node, list):
        return "[" + ",".join(_normalise(x) for x in node) + "]"
    return repr(node)


def ast_fingerprint(source: str) -> str:
    """SHA-256 of the module's AST with docstrings and positions removed."""
    tree = _strip_string_statements(ast.parse(source))
    return hashlib.sha256(_normalise(tree).encode("utf-8")).hexdigest()


def token_fingerprint(source: str) -> str:
    """SHA-256 of the meaning-carrying token stream, comments and docstrings
    dropped."""
    spans = _string_statement_lines(ast.parse(source))
    parts: list[str] = []
    readline = io.StringIO(source).readline
    for tok in tokenize.generate_tokens(readline):
        name = tokenize.tok_name[tok.type]
        if name not in _MEANING_TOKENS:
            continue
        if name.startswith(("STRING", "FSTRING")):
            row = tok.start[0]
            if any(lo <= row <= hi for lo, hi in spans):
                continue
        parts.append(f"{name}\x1f{tok.string}")
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the registry, read off the history documents themselves
# ---------------------------------------------------------------------------
_HEADER_RE = re.compile(
    r"<!--\s*lumenairy-history-doc\s*(?P<body>.*?)-->", re.S)


def _parse_header(md_path: pathlib.Path) -> dict[str, str]:
    text = md_path.read_text(encoding="utf-8")
    match = _HEADER_RE.search(text)
    if match is None:
        raise AssertionError(
            f"{md_path.name}: no `lumenairy-history-doc` header block.  Every "
            f"document under docs/history/ must declare the module it came "
            f"from and the pre-relocation fingerprints of that module.")
    header: dict[str, str] = {}
    for line in match.group("body").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        header[key.strip()] = value.strip()
    return header


def _registry() -> list[tuple[str, pathlib.Path, dict[str, str]]]:
    if not HISTORY_DIR.is_dir():
        return []
    out = []
    for md in sorted(HISTORY_DIR.glob("*.md")):
        if md.name.upper() == "README.MD":
            continue
        header = _parse_header(md)
        out.append((md.stem, md, header))
    return out


_REGISTRY = _registry()
_IDS = [name for name, _, _ in _REGISTRY]


def test_the_history_directory_is_populated():
    """A guard on the guard: if ``docs/history`` disappeared or lost its
    headers, every parametrised test below would silently collect zero cases
    and this file would pass while checking nothing."""
    assert _REGISTRY, (
        "docs/history/ holds no history documents -- WP-A17 part 1 ships "
        "carrier.md, carrier_field.md and fft_infra.md")
    names = set(_IDS)
    assert {"carrier", "carrier_field", "fft_infra"} <= names, sorted(names)


@pytest.mark.parametrize("name,md,header", _REGISTRY, ids=_IDS)
def test_the_header_names_a_real_module_and_two_fingerprints(name, md, header):
    for key in ("module", "ast_sha256", "token_sha256"):
        assert key in header, f"{md.name}: header is missing `{key}`"
    src_path = REPO_ROOT / header["module"]
    assert src_path.is_file(), f"{md.name}: `module:` does not resolve"
    # A document is named either by the module's basename (the three part-1 documents) or, because basenames
    # collide across packages (``pmm/stack.py`` vs ``rcwa/stack.py``), by the dotted module path
    # (``lumenairy.elements.pmm.stack``) -- the convention for the library-wide sweep.
    dotted = ".".join(pathlib.PurePosixPath(header["module"]).with_suffix("").parts)
    assert name in (src_path.stem, dotted), (
        f"{md.name}: document name must be the module basename or its dotted path "
        f"({src_path.stem!r} or {dotted!r})")
    for key in ("ast_sha256", "token_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", header[key]), (
            f"{md.name}: `{key}` is not a sha-256 hex digest")


@pytest.mark.parametrize("name,md,header", _REGISTRY, ids=_IDS)
def test_the_module_ast_is_unchanged_since_the_history_move(name, md, header):
    """Executable identity.  Fails the moment a "history-only" edit deletes,
    adds, reorders or re-spells a statement."""
    src = (REPO_ROOT / header["module"]).read_text(encoding="utf-8")
    assert ast_fingerprint(src) == header["ast_sha256"], (
        f"{header['module']}: the docstring-free AST no longer matches the "
        f"fingerprint recorded in docs/history/{md.name}.  Either the edit was "
        f"not documentation-only, or a deliberate code change needs the "
        f"fingerprints in that header re-recorded in the same commit:\n"
        f"    python scripts/record_history_fingerprints.py "
        f"{header['module']} --reason \"<what changed and why>\"\n"
        f"Do not hand-edit the hash -- the --reason is what keeps the header "
        f"an audit trail rather than a silenced gate.")


@pytest.mark.parametrize("name,md,header", _REGISTRY, ids=_IDS)
def test_the_module_token_stream_is_unchanged_since_the_history_move(
        name, md, header):
    """Literal identity.  Catches what the AST normalises away -- the spelling
    of a number or of a non-docstring string."""
    src = (REPO_ROOT / header["module"]).read_text(encoding="utf-8")
    assert token_fingerprint(src) == header["token_sha256"], (
        f"{header['module']}: the comment-free, docstring-free token stream no "
        f"longer matches docs/history/{md.name}.  A literal changed spelling, "
        f"or a statement moved.  If that was deliberate, re-record in the same "
        f"commit:\n"
        f"    python scripts/record_history_fingerprints.py "
        f"{header['module']} --reason \"<what changed and why>\"")


@pytest.mark.parametrize("name,md,header", _REGISTRY, ids=_IDS)
def test_the_document_carries_a_table_of_contents_in_source_order(
        name, md, header):
    """The move is only reversible if each block still says where it came
    from.  The document's TOC is ``| L<line> | ... |`` rows in ascending
    original-line order, and every row must have a matching anchor below."""
    text = md.read_text(encoding="utf-8")
    toc = re.findall(r"^\|\s*\[?L(\d+)\]?", text, re.M)
    assert toc, f"{md.name}: no `| L<line> |` table-of-contents rows"
    lines = [int(x) for x in toc]
    assert lines == sorted(lines), (
        f"{md.name}: the table of contents is not in original-source order")
    pre_lines = int(header.get("pre_relocation_lines", "0") or 0)
    if pre_lines:
        assert max(lines) <= pre_lines, (
            f"{md.name}: a table-of-contents line ({max(lines)}) is past the "
            f"end of the pre-relocation file ({pre_lines})")
    anchors = set(re.findall(r"^#+\s*L(\d+)\b", text, re.M))
    missing = sorted(set(toc) - anchors, key=int)
    assert not missing, (
        f"{md.name}: table-of-contents rows with no block below: {missing}")


@pytest.mark.parametrize("name,md,header", _REGISTRY, ids=_IDS)
def test_the_source_still_points_at_the_history_document(name, md, header):
    """A relocation that leaves no forwarding address is a deletion.  The
    module must name its history document at least once, so a reader who hits
    a bare guard can find out which audit put it there."""
    src = (REPO_ROOT / header["module"]).read_text(encoding="utf-8")
    assert f"docs/history/{md.name}" in src, (
        f"{header['module']}: nothing in the module points at "
        f"docs/history/{md.name}")


def _patch_node_text(src_lines, node, new_text, expect):
    """Replace the exact source span of ``node`` with ``new_text``.

    Returns the patched source, or ``None`` when the span does not read back as
    ``expect`` (a non-ASCII line, or a node whose columns are byte offsets that
    do not line up).  Every mutation below goes through this rather than a
    regex, because a regex can land inside a comment or a docstring -- which is
    exactly the text these fingerprints are built to ignore, so the mutation
    would prove nothing.
    """
    row = node.lineno - 1
    line = src_lines[row]
    if not line.isascii() or node.end_lineno != node.lineno:
        return None
    lo, hi = node.col_offset, node.end_col_offset
    if line[lo:hi] != expect:
        return None
    out = list(src_lines)
    out[row] = line[:lo] + new_text + line[hi:]
    return "".join(out)


@pytest.mark.parametrize("name,md,header", _REGISTRY, ids=_IDS)
def test_the_fingerprints_are_actually_sensitive(name, md, header):
    """Falsifiability.  A fingerprint that never moves would pass this file
    forever while the module was rewritten underneath it.

    Three mutations, each a way a real change could be smuggled in behind a
    "documentation-only" claim, are applied to a COPY of the source in memory
    (nothing is written).  Each is anchored on a real AST node, so it lands in
    CODE and never in a comment or docstring:

    1. a statement deleted        -> the AST fingerprint must move
    2. an identifier renamed      -> both fingerprints must move
    3. a numeric literal re-spelled to the SAME value (``5`` -> ``0x5``)
       -> the token fingerprint must move while the AST one does NOT.  That
       asymmetry is the whole reason two fingerprints are recorded: the AST
       folds both spellings to one ``Constant``, so the AST check alone would
       not see a literal being rewritten.
    """
    src = (REPO_ROOT / header["module"]).read_text(encoding="utf-8")
    ast_ref, tok_ref = ast_fingerprint(src), token_fingerprint(src)
    tree = ast.parse(src)
    src_lines = src.splitlines(keepends=True)

    # 1. drop one whole single-line statement from a function body
    cut = None
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if len(fn.body) < 3:
            continue
        # Any single-line statement of the body will do, searched from the end; a module whose only
        # multi-statement function ends in a multi-line ``return {...}`` (analysis/coronagraph.py) has
        # single-line statements earlier in the body.  A docstring statement is skipped: deleting it
        # cannot move the AST fingerprint, by design.
        for doomed in reversed(fn.body):
            if doomed.lineno != doomed.end_lineno:
                continue
            if (isinstance(doomed, ast.Expr) and isinstance(doomed.value, ast.Constant)
                    and isinstance(doomed.value.value, str)):
                continue
            candidate = "".join(
                ln for i, ln in enumerate(src_lines, 1) if i != doomed.lineno)
            try:
                ast.parse(candidate)
            except SyntaxError:
                continue
            cut = candidate
            break
        if cut is not None:
            break
    assert cut is not None, "no single-line statement found to delete"
    assert ast_fingerprint(cut) != ast_ref, (
        "deleting a statement did not move the AST fingerprint")

    # 2. rename an identifier
    renamed = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.isidentifier():
            renamed = _patch_node_text(
                src_lines, node, node.id + "_x", node.id)
            if renamed is not None:
                try:
                    ast.parse(renamed)
                except SyntaxError:
                    renamed = None
                    continue
                break
    assert renamed is not None, "no identifier found to rename"
    assert ast_fingerprint(renamed) != ast_ref, (
        "renaming an identifier did not move the AST fingerprint")
    assert token_fingerprint(renamed) != tok_ref, (
        "renaming an identifier did not move the token fingerprint")

    # 3. re-spell an integer literal without changing its value
    respelled = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and type(node.value) is int
                and 0 <= node.value <= 9):
            respelled = _patch_node_text(
                src_lines, node, f"0x{node.value:X}", str(node.value))
            if respelled is not None:
                break
    assert respelled is not None, "no small integer literal found to re-spell"
    assert token_fingerprint(respelled) != tok_ref, (
        "re-spelling a literal did not move the token fingerprint -- the "
        "token check is not adding anything over the AST check")
    assert ast_fingerprint(respelled) == ast_ref, (
        "the AST fingerprint moved on a value-preserving re-spelling; the "
        "mutation is not testing what this assertion claims")

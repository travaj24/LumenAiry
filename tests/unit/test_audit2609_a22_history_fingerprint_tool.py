"""WP-A22: the history-fingerprint recorder is itself gated.

``tests/unit/test_audit2609_a17_history_relocation.py`` pins every module with
a document under ``docs/history/`` to the AST and token fingerprints recorded
when its version-history narrative moved out of the source.  That pin has no
expiry, so the FIRST deliberate code change to any of those modules turns it
red -- and a red gate with no sanctioned way to clear it is a gate that gets
cleared by hand-editing the recorded hash, at which point the hash records
nothing.

``scripts/record_history_fingerprints.py`` is the sanctioned way, and this file
is the gate on IT.  The recorder is a tool that can silently weaken another
gate, which is the one kind of tool that must not be trusted on inspection.

WHAT IS ASSERTED, and why none of it is a tolerance:

* the recorder shares the checker's fingerprint functions rather than carrying
  its own copy -- asserted on the recorder's AST, so it holds for a lazily
  imported name too.  Two definitions of "unchanged" is the failure mode that
  would make every recorded hash meaningless, and it cannot be caught by
  running the tool: two implementations agree until the day they do not;
* ``--check`` is READ-ONLY -- byte-compared before and after;
* a drift the AST cannot see (a literal re-spelled to the same value) is still
  caught, which is the whole reason two fingerprints are recorded;
* after a re-record the document's hashes equal the CHECKER's reading of the
  drifted source -- compared against ``ast_fingerprint`` / ``token_fingerprint``
  directly, not merely "the tool now says OK", which a tool that wrote the same
  wrong value into both places would also say;
* every other header field survives the rewrite, including the ones this tool
  knows nothing about;
* the ``re_recorded:`` trail ACCUMULATES -- two re-records leave two lines --
  and a write with no reason is refused.

Everything happens in a ``tmp_path`` copy of a two-file tree.  The real
``docs/history/`` is never written to: the recorder derives a document's
repository root from the document's own path, so pointing it at a copy is all
the isolation this needs.
"""
from __future__ import annotations

import ast
import importlib.util
import io
import pathlib
import re

import pytest


def open_devnull():
    """A sink for the recorder's progress output.

    ``io.StringIO`` and not ``os.devnull``: the tool's report is the thing a
    maintainer reads, so a test that swallowed it into a real null device
    could not later assert on it.  Every call here discards, but the seam is
    open.
    """
    return io.StringIO()

_REPO = pathlib.Path(__file__).resolve().parents[2]
_TOOL = _REPO / 'scripts' / 'record_history_fingerprints.py'
_CHECKER = _REPO / 'tests' / 'unit' / 'test_audit2609_a17_history_relocation.py'

#: A module small enough to read in one screen and shaped like the real ones:
#: a docstring, a comment, a small integer literal (mutation target 3 in the
#: checker's own falsifiability test), and a function body.
_MODULE_SRC = '''\
"""A tiny module standing in for one with a relocated history.

See docs/history/sample_mod.md.
"""

# A comment.  Comments never reach the AST and are dropped from the token
# stream, so neither fingerprint can see this line.
_LIMIT = 5


def clamp(x):
    """Clamp to :data:`_LIMIT`."""
    if x > _LIMIT:
        return _LIMIT
    return x
'''

_HEADER_TEMPLATE = '''\
<!-- lumenairy-history-doc
module: pkg/sample_mod.py
ast_sha256: {ast}
token_sha256: {token}
pre_relocation_lines: 40
recorded_by: WP-A22 self-test fixture
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `pkg/sample_mod.py`

## Contents

| original line | site | what the block records |
|---|---|---|
| L7 | `_LIMIT` | why the clamp exists |

### L7 -- `_LIMIT`

Prose that the fingerprints deliberately cannot see.
'''

_HEADER_RE = re.compile(r'<!--\s*lumenairy-history-doc\s*(?P<body>.*?)-->',
                        re.S)


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def tool():
    assert _TOOL.is_file(), (
        f'{_TOOL} is missing.  The A17 relocation gate pins modules to '
        f'fingerprints forever; this is the only sanctioned way to move one, '
        f'and without it the next legitimate change to a relocated module has '
        f'no path that is not hand-editing a hash.')
    return _load(_TOOL, '_a22_record_history_fingerprints')


@pytest.fixture(scope='module')
def checker():
    return _load(_CHECKER, '_a22_history_relocation_checker')


@pytest.fixture
def tree(tmp_path, checker):
    """A two-file repository copy: the module, and its history document with
    the module's CURRENT fingerprints recorded."""
    (tmp_path / 'pkg').mkdir()
    src = tmp_path / 'pkg' / 'sample_mod.py'
    src.write_text(_MODULE_SRC, encoding='utf-8')

    (tmp_path / 'docs' / 'history').mkdir(parents=True)
    doc = tmp_path / 'docs' / 'history' / 'sample_mod.md'
    doc.write_text(
        _HEADER_TEMPLATE.format(
            ast=checker.ast_fingerprint(_MODULE_SRC),
            token=checker.token_fingerprint(_MODULE_SRC)),
        encoding='utf-8')
    return tmp_path, src, doc


def _header(doc: pathlib.Path) -> dict[str, str]:
    body = _HEADER_RE.search(doc.read_text(encoding='utf-8')).group('body')
    out = {}
    for line in body.splitlines():
        if ':' in line:
            key, _, value = line.partition(':')
            out[key.strip()] = value.strip()
    return out


# ===========================================================================
# 1 -- the recorder and the gate must share ONE definition of "unchanged"
# ===========================================================================
def test_the_recorder_uses_the_checkers_fingerprint_functions(tool):
    """Structural, not behavioural.

    A second implementation of either fingerprint would agree with the
    checker's right up until the day it did not -- and on that day the
    recorder would write a hash the gate rejects, or worse, one it accepts for
    a module that did change.  So this is asserted on the recorder's AST: it
    must define neither function itself, and it must load the checker.
    """
    tree = ast.parse(_TOOL.read_text(encoding='utf-8'))
    defined = {node.name for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    clashing = defined & {'ast_fingerprint', 'token_fingerprint'}
    assert not clashing, (
        f'scripts/record_history_fingerprints.py defines {sorted(clashing)} '
        f'itself instead of importing the checker\'s.  Two definitions of '
        f'"unchanged" is exactly the drift this whole mechanism exists to '
        f'prevent; delete the copy and call the checker\'s.')
    assert 'load_checker' in defined, (
        'the recorder no longer loads the checker module; its fingerprints '
        'are then its own opinion rather than the gate\'s.')
    source = _TOOL.read_text(encoding='utf-8')
    assert 'test_audit2609_a17_history_relocation.py' in source, (
        'the recorder does not name the checker it borrows from.')


# ===========================================================================
# 2 -- --check is read-only and reports the truth
# ===========================================================================
def test_check_passes_on_a_clean_tree_and_writes_nothing(tool, checker, tree):
    root, _src, doc = tree
    before = doc.read_bytes()
    rc = tool.run([], check=True, reason=None, root=root, checker=checker,
                  stream=open_devnull())
    assert rc == 0, 'a document that matches its module must check clean'
    assert doc.read_bytes() == before, (
        '--check wrote to the document.  It is specified read-only so it can '
        'be run in CI and on a dirty tree without changing the thing it is '
        'measuring.')


def test_check_reports_drift_and_still_writes_nothing(tool, checker, tree):
    root, src, doc = tree
    before = doc.read_bytes()
    src.write_text(_MODULE_SRC.replace('return x', 'return int(x)'),
                   encoding='utf-8')
    rc = tool.run([], check=True, reason=None, root=root, checker=checker,
                  stream=open_devnull())
    assert rc == 1, (
        'a module whose AST moved must fail --check: that is the gate this '
        'tool exists to serve.')
    assert doc.read_bytes() == before, '--check wrote to the document'


def test_a_value_preserving_respelling_is_still_drift(tool, checker, tree):
    """The asymmetry that justifies recording TWO fingerprints.

    ``5`` -> ``0x5`` is the same value and the same AST -- ``ast.Constant(5)``
    both ways -- so an AST-only recorder would call this clean and re-record
    nothing.  The token stream sees it.
    """
    root, src, doc = tree
    drifted = _MODULE_SRC.replace('_LIMIT = 5', '_LIMIT = 0x5')
    src.write_text(drifted, encoding='utf-8')
    assert checker.ast_fingerprint(drifted) == _header(doc)['ast_sha256'], (
        'the fixture is wrong: this mutation is supposed to leave the AST '
        'fingerprint alone, and the test below proves nothing if it does not.')
    rc = tool.run([], check=True, reason=None, root=root, checker=checker,
                  stream=open_devnull())
    assert rc == 1, (
        'a literal re-spelled to the same value moved the token fingerprint '
        'and --check did not notice.  The token fingerprint is then adding '
        'nothing over the AST one.')


# ===========================================================================
# 3 -- the re-record itself
# ===========================================================================
def test_re_recording_writes_the_checkers_own_reading(tool, checker, tree):
    root, src, doc = tree
    drifted = _MODULE_SRC.replace('return x', 'return int(x)')
    src.write_text(drifted, encoding='utf-8')

    rc = tool.run(['pkg/sample_mod.py'], check=False,
                  reason='clamp returns an int', root=root, checker=checker,
                  today='2026-09-12', stream=open_devnull())
    assert rc == 0

    header = _header(doc)
    # The ORACLE is the checker's own reading of the drifted file, not "the
    # tool now says OK" -- a tool that wrote one wrong value into both the
    # header and its own comparison would say OK too.
    assert header['ast_sha256'] == checker.ast_fingerprint(drifted)
    assert header['token_sha256'] == checker.token_fingerprint(drifted)
    # and the gate, run the way it runs for real, now accepts the pair
    assert tool.run([], check=True, reason=None, root=root, checker=checker,
                    stream=open_devnull()) == 0


def test_re_recording_preserves_every_other_header_field(tool, checker, tree):
    """The rewrite touches two lines and leaves the rest alone.

    ``pre_relocation_lines`` is load-bearing for the checker's
    table-of-contents test, and ``recorded_by`` / ``checker`` are the trail.
    A regenerating writer would have to know about all of them -- including
    fields a later work package adds -- so the rewrite is in place.
    """
    root, src, doc = tree
    original = _header(doc)
    src.write_text(_MODULE_SRC.replace('return x', 'return int(x)'),
                   encoding='utf-8')
    tool.run([], check=False, reason='why', root=root, checker=checker,
             today='2026-09-12', stream=open_devnull())

    now = _header(doc)
    for key in ('module', 'pre_relocation_lines', 'recorded_by', 'checker'):
        assert now[key] == original[key], f'{key} changed in the rewrite'
    for key in ('ast_sha256', 'token_sha256'):
        assert now[key] != original[key], f'{key} did not move'
    # the prose below the header is untouched
    text = doc.read_text(encoding='utf-8')
    assert '### L7 -- `_LIMIT`' in text
    assert 'Prose that the fingerprints deliberately cannot see.' in text


def test_the_re_recorded_trail_accumulates(tool, checker, tree):
    """Two re-records leave TWO lines.

    The trail is the only thing that distinguishes a maintained baseline from
    a silenced gate, so a second re-record must not overwrite the first --
    a reader has to be able to see every time the baseline moved and the
    reason given each time.
    """
    root, src, doc = tree
    src.write_text(_MODULE_SRC.replace('return x', 'return int(x)'),
                   encoding='utf-8')
    tool.run([], check=False, reason='first change', root=root,
             checker=checker, today='2026-09-12', stream=open_devnull())
    src.write_text(_MODULE_SRC.replace('_LIMIT = 5', '_LIMIT = 7'),
                   encoding='utf-8')
    tool.run([], check=False, reason='second change', root=root,
             checker=checker, today='2026-09-13', stream=open_devnull())

    body = _HEADER_RE.search(doc.read_text(encoding='utf-8')).group('body')
    trail = [ln.strip() for ln in body.splitlines()
             if ln.strip().startswith('re_recorded:')]
    assert trail == ['re_recorded: 2026-09-12 -- first change',
                     're_recorded: 2026-09-13 -- second change'], trail
    # and the document still parses as a header the checker accepts
    assert tool.run([], check=True, reason=None, root=root, checker=checker,
                    stream=open_devnull()) == 0


def test_a_write_with_no_reason_is_refused(tool):
    """A re-recorded fingerprint with no reason is indistinguishable from
    someone silencing the gate, so the CLI refuses it."""
    with pytest.raises(SystemExit) as exc:
        tool.main(['pkg/sample_mod.py'])
    assert exc.value.code == 2, (
        'argparse must reject a write with no --reason (exit 2), not fall '
        'through to a silent re-record')


def test_a_malformed_document_fails_without_aborting_the_sweep(
        tool, checker, tree, tmp_path):
    """One bad header must not hide the state of every other document.

    Over a directory the useful output is EVERY problem, not the first one --
    and a header the recorder cannot read is one the gate cannot read either,
    so it is a failure and not a skip.
    """
    root, _src, doc = tree
    broken = tmp_path / 'docs' / 'history' / 'broken.md'
    broken.write_text('<!-- lumenairy-history-doc\n'
                      'recorded_by: nobody\n-->\n', encoding='utf-8')
    out = open_devnull()
    rc = tool.run([], check=True, reason=None, root=root, checker=checker,
                  stream=out)
    text = out.getvalue()
    assert rc == 1, 'a header with no `module:` line must fail --check'
    assert 'broken.md' in text, 'the failure must name the document'
    assert 'sample_mod.md' in text, (
        'the sweep stopped at the broken document, so the state of the '
        'others went unreported -- which is the whole reason to sweep')


def test_an_unknown_target_names_the_documents_it_does_know(tool):
    with pytest.raises(SystemExit) as exc:
        tool.resolve('lumenairy/not/a/module.py')
    message = str(exc.value)
    assert 'no history document matches' in message
    assert 'fft_infra' in message, (
        'the failure must list the documents that DO exist, or a typo reads '
        'as "this module has no history" instead of "you spelled it wrong"')


# ===========================================================================
# 4 -- the real tree stays green
# ===========================================================================
def test_every_committed_history_document_matches_its_module(tool, checker):
    """The same claim the A17 gate makes, made through the recorder.

    It is not redundant: it is the check that the recorder's own reading of
    the real tree agrees with the gate's.  If this passes and the A17 gate
    fails (or the reverse), the two have drifted apart and every re-record
    since is suspect.
    """
    out = open_devnull()
    rc = tool.run([], check=True, reason=None, root=_REPO, checker=checker,
                  stream=out)
    assert rc == 0, (
        'scripts/record_history_fingerprints.py --check reports drift on the '
        'committed tree.  Either a module with a relocated history changed '
        'without its fingerprints being re-recorded in the same commit -- '
        'run the recorder with --reason -- or the recorder and the A17 gate '
        'disagree, in which case every re-record since is suspect.  Report:\n'
        + out.getvalue())

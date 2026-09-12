"""WP-A21: the doc-identifier resolver is a GATE, not a one-off measurement.

The 2026-09-11 audit's headline for the four top-level documents was "78 % of
backticked identifiers do not resolve", measured on a 60-token sample with no
clean denominator.  WP-A18 built the denominator, drove the genuinely
unresolved count to 0, and asked (its section 9) for the resolver to be
committed with a test that keeps it there -- because a doc sweep that is not
gated is stale by the next release.  This file is that gate.

WHAT IS ASSERTED, and why none of it is a tolerance:

* the unresolved count is **0** -- an integer;
* the API-claiming DENOMINATOR has not collapsed.  A resolver can always reach
  zero unresolved by excluding everything, so the zero is only worth something
  next to the size of the set it was measured over.  MEASURED 592 distinct
  API-claiming tokens out of 1 055 distinct identifier-shaped tokens in 1 952
  backticked occurrences; the floor below is 450, which leaves room for the
  README split the ROADMAP plans while still failing on a collapse;
* the resolver actually FAILS on a fabricated name -- demonstrated in-process
  against a temporary document, so the gate is known to have teeth on this
  build rather than assumed to;
* every hand-triaged exclusion is still CITED somewhere in the four documents.
  An exclusion that no longer matches anything is a hole kept open for no
  reason -- the same counter-pin shape as
  ``test_audit2609_a15a_packaging.py::test_the_stdio_exemption_list_is_still_accurate``;
* the resolver reaches the NETWORK nowhere.  Asserted structurally, by walking
  the script's own AST for an import of any networking module, rather than by
  trusting the docstring.

COST.  The resolver imports the package and every importable submodule, then
AST-walks ``lumenairy/**/*.py``.  That index is built ONCE for this file (a
module-scoped fixture) and reused by all five tests: MEASURED 4.7 s for the
whole file on the calibration box, against 4.8 s for a single standalone run
of the script.  No wall clock is asserted anywhere (TESTING_STANDARDS S1); the
bound is stated here so a reviewer can see the file is cheap.
"""
import ast
import importlib.util
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / 'scripts' / 'check_doc_identifiers.py'

#: MEASURED 2026-09-12 on branch audit-fixes-2026-09: 592 distinct
#: API-claiming tokens.  The floor is 450 -- 24 % of headroom below today's
#: reading, which absorbs a doc edit but not a broken scanner (a regex that
#: stopped matching, an exclusion rule that swallowed a category).
_DENOMINATOR_FLOOR = 450

#: MEASURED 2026-09-12: 51 hand-triaged entries.  May shrink, never grow
#: without a reason written at the entry -- the list is the one place the
#: gate can be weakened silently.
_CURATED_CEILING = 51

#: Networking modules.  The gate must run offline on any CI runner.
_NETWORK_MODULES = {
    'socket', 'ssl', 'http', 'urllib', 'urllib3', 'ftplib', 'telnetlib',
    'smtplib', 'requests', 'httpx', 'aiohttp', 'xmlrpc', 'asyncio',
}


def _load_script():
    spec = importlib.util.spec_from_file_location('check_doc_identifiers',
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('check_doc_identifiers', mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def cdi():
    assert _SCRIPT.is_file(), (
        f'{_SCRIPT} is missing.  WP-A18 section 9 asked for the resolver to '
        f'be committed beside check_source_line_citations.py; this test is '
        f'the gate that keeps it honest and it has nothing to run.')
    return _load_script()


@pytest.fixture(scope='module')
def index(cdi):
    """The resolution index, built once for the whole file."""
    return cdi._Index()


@pytest.fixture(scope='module')
def result(cdi, index):
    return cdi.check(index=index)


def test_no_backticked_identifier_in_the_docs_is_unresolved(result):
    """The deliverable: zero unresolved, on the current tree."""
    assert not result.unresolved, (
        'these backticked identifiers in README.md / ROADMAP.md / '
        'Migration-Guide.md / CONVENTIONS.md do not resolve against the '
        'package:\n' + '\n'.join(
            f'  {tok}  ({len(sites)}x, first at {sites[0][0]}:{sites[0][1]})'
            f'  [{how}]'
            for tok, (how, sites) in sorted(result.unresolved.items())) +
        '\n\nEither the document names something that no longer exists -- fix '
        'the document -- or the token is not an API claim, in which case add '
        'it to CURATED in scripts/check_doc_identifiers.py with the reason '
        'you read at its citation site.  Do not widen a mechanical rule to '
        'make one token disappear.')


def test_the_denominator_has_not_collapsed(result):
    """COUNTER-PIN: zero unresolved must be measured over a real set.

    Without this, any change that made ``classify`` over-eager -- a new
    prefix, a broader suffix rule -- would drive the unresolved count to zero
    by deleting the denominator, and the test above would go green on a
    resolver that checks nothing.
    """
    assert result.denominator >= _DENOMINATOR_FLOOR, (
        f'the API-claiming denominator is {result.denominator}, below the '
        f'{_DENOMINATOR_FLOOR} floor (measured 592 on 2026-09-12).  An '
        f'exclusion rule has almost certainly swallowed a whole category; '
        f'zero unresolved out of nothing is not a result.')
    assert result.occurrences > result.distinct >= result.denominator, (
        f'the scan is internally inconsistent: {result.occurrences} '
        f'occurrences, {result.distinct} distinct, {result.denominator} '
        f'API-claiming.')


def test_the_gate_fails_on_a_fabricated_identifier(cdi, index, tmp_path):
    """FAIL-BEFORE, in process: a document that cites a name the package does
    not have must make the resolver unhappy, and ``main`` must exit non-zero.

    The fabricated names are deliberately the two shapes the audit found: a
    bare public-looking function and a dotted path under the package root.
    """
    (tmp_path / 'README.md').write_text(
        'Call `propagate_through_a_wormhole()` on the result, or use\n'
        '`lumenairy.propagators.definitely_not_here` for the batch form.\n'
        'For contrast, `angular_spectrum_propagate` is real.\n',
        encoding='utf-8')
    res = cdi.check(docroot=tmp_path, files=('README.md',), index=index)
    assert set(res.unresolved) == {'propagate_through_a_wormhole()',
                                   'lumenairy.propagators.definitely_not_here'}, (
        f'the resolver did not flag the fabricated names; it reported '
        f'{sorted(res.unresolved)} unresolved and '
        f'{sorted(res.resolved)} resolved.')
    assert 'angular_spectrum_propagate' in res.resolved, (
        'the control name -- a real top-level export cited the same way -- '
        'must resolve, or the arm above proves nothing about fabrication.')
    assert cdi.main(['--docroot', str(tmp_path)]) == 1, (
        'the CLI must exit 1 when a document cites a name that does not '
        'exist, or CI cannot use it as a gate.')


def test_every_hand_triaged_exclusion_is_still_cited(cdi, index):
    """COUNTER-PIN: no dead entries in the triaged list.

    An exclusion whose token has left the documents is a hole kept open for
    nothing -- and the next identifier that happens to collide with that name
    slips through it.  MEASURED 2026-09-12: 51 entries, 51 still cited.
    """
    rows = cdi.scan(cdi._REPO_ROOT)
    cited = {tok for _f, _ln, tok, _line in rows}
    cited |= {t[:-2] for t in cited if t.endswith('()')}
    stale = sorted(k for k in cdi.CURATED if k not in cited)
    assert not stale, (
        f'these hand-triaged exclusions no longer appear in any of the four '
        f'documents: {stale}.  Delete them -- the citation they were written '
        f'for is gone.')
    assert len(cdi.CURATED) <= _CURATED_CEILING, (
        f'the triaged exclusion list has grown to {len(cdi.CURATED)} entries '
        f'(was {_CURATED_CEILING} on 2026-09-12).  Growing it is how this '
        f'gate gets hollowed out one token at a time; if an addition is '
        f'genuinely right, raise this ceiling in the same change and say why.')


def test_the_resolver_reaches_no_network(cdi):
    """Structural, not a promise: the script imports nothing that can.

    Asserted on the AST rather than on the module's ``__dict__`` so it holds
    for a lazily-imported name too.  ``lumenairy`` itself is imported by the
    index and is not in scope here -- the claim is about the resolver.
    """
    tree = ast.parse(_SCRIPT.read_text(encoding='utf-8'))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split('.')[0])
    offending = sorted(imported & _NETWORK_MODULES)
    assert not offending, (
        f'scripts/check_doc_identifiers.py imports {offending}; the gate is '
        f'specified to run offline on any CI runner, reading only the four '
        f'documents and the installed source tree.')

"""Packaging / CI / test-hygiene gates (audit 2026-09-11, V4 and V7).

Each of these pins a fact the 2026-09-11 audit found had drifted with nothing
watching it.  They are all static or metadata checks -- no physics, no
wall-clock, nothing a build is entitled to move -- and they run in
milliseconds.
"""
from __future__ import annotations

import ast
import os
import re

import pytest

# ``tomllib`` is stdlib from Python 3.11.  On 3.10 -- the documented floor
# (``requires-python = ">=3.10"``) AND a live leg of the unit-tests matrix
# this very file pins below -- it does not exist, and the ``tomli`` backport
# supplies the same ``load(fp)`` API; the ``unit`` job installs it
# (``pip install tomli`` in the Install-dependencies step).
#
# MEASURED on CI run 34914295323 (5.47.0): the unconditional ``import
# tomllib`` here raised ModuleNotFoundError at COLLECTION on every one of the
# five 3.10 shards, and pytest answers a collection error with ``Interrupted:
# 1 error during collection`` -- so each shard reported ``18 skipped, 12 578
# deselected, 1 error`` and ran ZERO of its ~2 900 selected tests.  One
# missing stdlib module in one test file took the entire 3.10 lane, not just
# this file's eleven gates.  Hence the fallback, and hence the fact that
# nothing below imports ``tomllib`` at module scope again.
try:
    import tomllib                                  # Python 3.11+ stdlib
    _TOML_SKIP_REASON = None
except ModuleNotFoundError:            # pragma: no cover -- Python 3.10 only
    try:
        import tomli as tomllib  # type: ignore[no-redef]
        _TOML_SKIP_REASON = None
    except ModuleNotFoundError:        # pragma: no cover -- neither present
        tomllib = None  # type: ignore[assignment]
        _TOML_SKIP_REASON = (
            'no TOML parser: `tomllib` is Python 3.11+ stdlib and the `tomli` '
            '3.10 backport is not installed, so the four pyproject-reading '
            'gates in this file cannot be evaluated on this interpreter '
            '(`pip install tomli`).  The other seven gates -- .gitignore, '
            'MANIFEST.in, scripts/ hygiene and the two subprocess-stdio '
            'walkers -- do not need a parser and still ran.')

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _pyproject() -> dict:
    # Premise-gated, NOT module-gated: only the four gates that read
    # pyproject.toml depend on a parser being importable, so a missing
    # backport must not take the seven that do not.  Where `tomli` IS
    # installed -- which is the CI 3.10 leg -- this branch never fires and
    # all eleven gates run.
    if tomllib is None:                # pragma: no cover -- neither present
        pytest.skip(_TOML_SKIP_REASON)
    with open(os.path.join(REPO_ROOT, 'pyproject.toml'), 'rb') as fh:
        return tomllib.load(fh)


def _read(*parts) -> str:
    with open(os.path.join(REPO_ROOT, *parts), 'r', encoding='utf-8') as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# V4 -- the CI matrix and the classifiers describe the same interpreters
# ---------------------------------------------------------------------------

def test_every_declared_python_classifier_is_in_the_unit_test_matrix():
    """A ``Programming Language :: Python :: X.Y`` classifier is a promise that
    the wheel works on X.Y.  Until 2026-09-12 the repo made that promise for
    3.10-3.13 while DEVELOPING on 3.14 (tracked
    ``.benchmarks/Windows-CPython-3.14-64bit/*.json``, ``cpython-314`` pycs)
    with no CI leg on it, behind a pyproject comment asserting the accelerator
    wheels did not exist -- false on the dev box, measured.  This pins the
    other direction too: a classifier the matrix does not run is a promise
    nothing backs.
    """
    cfg = _pyproject()
    classified = {c.rsplit(' :: ', 1)[1]
                  for c in cfg['project']['classifiers']
                  if c.startswith('Programming Language :: Python :: ')
                  and re.fullmatch(r'\d+\.\d+', c.rsplit(' :: ', 1)[1])}
    wf = _read('.github', 'workflows', 'unit-tests.yml')
    m = re.search(r'^\s*python-version:\s*\[(.*?)\]', wf, re.M)
    assert m is not None, (
        'unit-tests.yml no longer declares a `python-version: [...]` matrix; '
        'this gate cannot see which interpreters CI runs.')
    matrix = set(re.findall(r"'([\d.]+)'", m.group(1)))
    assert classified <= matrix, (
        f'these Python versions are CLASSIFIED as supported but are in no leg '
        f'of the unit-tests matrix: {sorted(classified - matrix)}.  Either add '
        f'them to the matrix or drop the classifier -- a classifier no CI leg '
        f'backs is the P1-5 defect this gate closes.  (classified '
        f'{sorted(classified)}, matrix {sorted(matrix)})')


def test_the_lint_job_is_blocking():
    """``ruff`` is the only always-on static gate; it must block merge.

    Between v5.0.1 and 2026-09-12 the job carried ``continue-on-error: true``,
    so a lint failure -- including an F821 undefined name -- reported green.
    """
    wf = _read('.github', 'workflows', 'unit-tests.yml')
    m = re.search(r'\n  lint:\n(.*?)(?=\n  \w+:\n)', wf, re.S)
    assert m is not None, 'unit-tests.yml no longer has a `lint:` job.'
    body = m.group(1)
    # match the KEY, not the word inside the comment that explains the history
    live = [ln for ln in body.splitlines()
            if ln.strip().startswith('continue-on-error:')]
    assert live and all('false' in ln for ln in live), (
        'the ruff lint job is advisory again (`continue-on-error: true`).  '
        'It is the only static gate that runs on every PR; if the tree has '
        'findings, fix them or add a scoped per-file ignore with a reason in '
        '[tool.ruff.lint.per-file-ignores] -- do not silence the job.')


def test_ruff_lints_the_ui_package():
    """``lumenairy/ui/`` (~30 modules, incl. a 3 626-line ``main_window.py``)
    must not be excluded from linting wholesale again.

    It is scoped in with a per-directory ignore list instead, so the cosmetic
    Qt idioms are silenced while F821 / F632 / E711-E714 / E722 are enforced.
    """
    cfg = _pyproject()
    excl = cfg['tool']['ruff'].get('extend-exclude', [])
    assert not any('ui' in str(e) for e in excl), (
        f'lumenairy/ui/ is excluded from ruff again ({excl}).  Use the '
        f'per-file-ignore list in [tool.ruff.lint.per-file-ignores] so the '
        f'directory keeps the rules that catch real bugs.')
    per_file = cfg['tool']['ruff']['lint']['per-file-ignores']
    ui_keys = [k for k in per_file if k.startswith('lumenairy/ui/')]
    assert ui_keys, (
        'lumenairy/ui/ has no per-file ignore entry, so either the directory '
        'is now clean under the full rule set (delete this assertion and '
        'celebrate) or the entry was dropped and the lint job is red.')
    # The ignore list is a debt ledger: it may shrink, never grow.
    # MEASURED 2026-09-12: exactly these seven rules fire in ui/ (331 findings
    # -- F401 121, I001 116, E701 37, E702 37, F811 10, F841 7, F541 3).
    allowed = {'F401', 'I001', 'E701', 'E702', 'F811', 'F841', 'F541'}
    for key in ui_keys:
        extra = set(per_file[key]) - allowed
        assert not extra, (
            f'{key} ignores {sorted(extra)}, which is outside the measured '
            f'2026-09-12 baseline {sorted(allowed)}.  The ui ignore list is a '
            f'debt ledger: shrink it as the GUI is cleaned, never grow it -- '
            f'a new rule here means a new class of finding is being silenced '
            f'rather than fixed.')


def test_the_mypy_whitelist_only_grows():
    """``[tool.mypy] files`` is a ratchet.

    MEASURED 2026-09-12: 17 declared paths (22 source files) after the V4
    growth, from 6 paths / 11 files.  The audit found the gate covering 2.6 %
    of the library behind a config comment that was 43 minor versions stale;
    the remedy only works if the list is not quietly trimmed when a module
    stops type-checking.

    RAISED 2026-09-12 (release step): 25 -> 28 with the root package, elements
    package and lens_config joining after WP-A16's follow-up (33 -> 0 strict errors).
    RAISED 2026-09-12 (WP-A21, same day): 17 -> 25 declared paths, adding the
    eight subpackage `__init__.py` re-export surfaces that measure
    strict-clean after the lazy-loading rewrite.  The root
    `lumenairy/__init__.py` is the ninth and is not in yet -- see the
    `[tool.mypy]` comment for the 33 errors that block it.
    """
    files = _pyproject()['tool']['mypy']['files']
    assert len(files) >= 28, (
        f'[tool.mypy] files has shrunk to {len(files)} entries (was 28 on '
        f'2026-09-12).  A module that stopped passing --strict should be '
        f'FIXED, not removed from the gate; if a removal is genuinely right, '
        f'lower this number in the same change and say why.')
    for f in files:
        assert os.path.exists(os.path.join(REPO_ROOT, *f.split('/'))), (
            f'[tool.mypy] files lists {f!r}, which does not exist -- mypy '
            f'silently checks nothing for a missing path.')


def test_threadpoolctl_is_a_hard_dependency_in_both_places():
    """``set_blas_threads`` / ``rcwa_blas_threads`` / ``@_with_blas_limit`` are
    INERT without ``threadpoolctl`` -- measured 400x (a 163x163 complex
    ``inv``) and 140x (a 1-D TM RCWA solve at n_orders=81) on a 24-thread
    Windows OpenBLAS box.  It must be declared, and declared in both files."""
    deps = _pyproject()['project']['dependencies']
    assert any(d.split(';')[0].strip().startswith('threadpoolctl')
               for d in deps), (
        f'threadpoolctl is not in [project.dependencies] ({deps}).  Without '
        f'it every BLAS-cap the library offers is a no-op that only warns.')
    req = _read('requirements.txt')
    assert re.search(r'^threadpoolctl>=', req, re.M), (
        'requirements.txt does not mirror the threadpoolctl dependency; the '
        'two files are meant to describe the same install.')


# ---------------------------------------------------------------------------
# V4/P1-10 -- .gitignore must not blanket-ignore binary fixtures
# ---------------------------------------------------------------------------

def test_gitignore_has_no_blanket_binary_rules():
    """A repo-wide ``*.png`` / ``*.dat`` / ``*.log`` rule makes every
    legitimate test fixture or doc asset un-addable without ``git add -f``.
    The audit found 33 ``.png`` under ``validation/`` in exactly that state.
    Directory-scoped rules only.
    """
    lines = [ln.strip() for ln in _read('.gitignore').splitlines()]
    blanket = [ln for ln in lines
               if re.fullmatch(r'\*\.(png|jpe?g|pdf|fits|dat|log|npz|npy|csv)',
                               ln)]
    assert not blanket, (
        f'.gitignore carries blanket binary rules {blanket}.  Scope them to '
        f'the directories that generate them (output/, validation/**, '
        f'docs/audits/**) so a real fixture elsewhere stays addable.')


def test_gitignore_covers_the_tool_caches():
    """``.mypy_cache/``, ``.ruff_cache/`` and ``.benchmarks/`` all exist in a
    working tree as soon as anyone runs the tools; none was ignored."""
    text = _read('.gitignore')
    for entry in ('.mypy_cache/', '.ruff_cache/', '.benchmarks/'):
        assert entry in text, (
            f'{entry} is not in .gitignore, so running the tool that writes '
            f'it dirties every developer tree (and, for .benchmarks/, invites '
            f'one machine\'s timings into version control).')
    assert '.test_durations' not in re.sub(r'^\s*#.*$', '', text, flags=re.M), (
        '.test_durations must stay TRACKED -- pytest-split reads it to '
        'balance the CI shards.  Ignoring it is the mistake this comment '
        'exists to prevent.')


def test_manifest_excludes_the_per_round_probe_trees():
    """``validation/probe_*`` and ``validation/repro_*`` are 808 of the 876
    ``.py`` under ``validation/`` -- per-audit-round scratch, not the suite a
    user runs.  They must not ship in the sdist."""
    man = _read('MANIFEST.in')
    for pat in ('validation/probe_*', 'validation/repro_*'):
        assert re.search(r'^prune %s\s*$' % re.escape(pat), man, re.M), (
            f'MANIFEST.in does not prune {pat}; the sdist carries the whole '
            f'per-round probe payload.')
    # The stale "31 files" claim (a 4.4.0 count) is corrected; the comment now
    # carries the measured figure.  Assert the measured one is present rather
    # than that the old string is absent -- the corrected comment quotes the
    # old claim in order to explain it.
    assert '876' in man, (
        'MANIFEST.in no longer records the measured size of the validation '
        'tree (876 .py across 70 directories, 2026-09-12).  The comment used '
        'to claim "31 files with ~370 physics-fidelity tests", a 4.4.0 count '
        'that was 28x off; do not let it drift back to an unmeasured number.')


def test_scripts_holds_only_the_maintained_tools():
    """``scripts/`` is a tools directory, so every ``.py`` in it reads as a
    maintained tool.  The 11 ``_d5_* / _g8_*`` audit probes that used to sit
    there moved to ``validation/probe_scripts_legacy/`` on 2026-09-12."""
    here = os.path.join(REPO_ROOT, 'scripts')
    stray = sorted(f for f in os.listdir(here)
                   if f.endswith('.py') and f.startswith('_'))
    assert not stray, (
        f'scripts/ has picked up private/probe modules again: {stray}.  '
        f'One-off audit reproductions belong in '
        f'validation/probe_scripts_legacy/ (pruned from the sdist), not '
        f'beside check_dep_metadata.py and the release walkers.')


# ---------------------------------------------------------------------------
# V5 item 5 -- subprocess stdio discipline
# ---------------------------------------------------------------------------

_SPAWNERS = {'run', 'Popen', 'check_output', 'check_call', 'call'}


def _subprocess_sites():
    """Yield ``(relpath, lineno, kwargs)`` for every subprocess spawn in
    ``tests/``."""
    for root, dirs, files in os.walk(os.path.join(REPO_ROOT, 'tests')):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for fn in sorted(files):
            if not fn.endswith('.py'):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding='utf-8', errors='replace') as fh:
                src = fh.read()
            if 'subprocess' not in src:
                continue
            rel = os.path.relpath(path, REPO_ROOT).replace(os.sep, '/')
            try:
                tree = ast.parse(src)
            except SyntaxError:                      # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                if isinstance(f, ast.Attribute) and f.attr in _SPAWNERS and \
                        isinstance(f.value, ast.Name) and \
                        f.value.id in ('subprocess', 'sp'):
                    yield rel, node.lineno, {k.arg for k in node.keywords
                                             if k.arg}


# Sites that pre-date this gate and belong to other work packages; each is
# listed in the WP-A15a report as a one-kwarg request to its owner.  The list
# may shrink, never grow -- a new entry means a new spawn was written without
# naming its streams.
#
# EMPTY since 2026-09-12 (WP-A21): the last three -- ``test_niche_audit_w3_
# infra.py`` and ``test_niche_d14_deterministic_carrier_fit.py`` x2 -- took
# their ``stdin=subprocess.DEVNULL``, so the gate now covers ``tests/``
# without a hole.  Keep it empty: an exemption is a live WinError-6 flake.
_STDIO_EXEMPT: set = set()


def test_every_subprocess_spawn_names_all_three_stdio_streams():
    """A spawn must name stdin, stdout and stderr -- or none of them.

    MEASURED 2026-09-12 under this suite's own pytest configuration
    (``--capture=fd``, the default): ``subprocess.run([...], stdout=PIPE)``
    raises ``OSError: [WinError 6] The handle is invalid`` from
    ``subprocess.py:1431`` BEFORE the child is created, while the same call
    with all three streams named succeeds.  The mechanism is CPython's
    ``Popen._get_handles`` on Windows: any stream left as ``None`` is resolved
    through ``GetStdHandle()`` and then ``DuplicateHandle``, and pytest's
    fd-capture reassigns the process's file descriptors WITHOUT calling
    ``SetStdHandle``, so the Win32 handles it duplicates can be stale.  (The
    all-``None`` case is safe: CPython short-circuits it before touching any
    handle.)  This is the cause of the intermittent ``WinError 6`` / ``50``
    failures three work packages reported this campaign -- 28 partial sites in
    17 files at the time of measurement, 25 of them fixed in the same change.

    The fix is one kwarg (``stdin=subprocess.DEVNULL``), it is deterministic,
    and it costs nothing: none of these children reads stdin.
    """
    offenders = []
    for rel, line, kw in _subprocess_sites():
        if (rel, line) in _STDIO_EXEMPT:
            continue
        has_out = 'stdout' in kw or 'capture_output' in kw
        has_err = 'stderr' in kw or 'capture_output' in kw
        has_in = 'stdin' in kw or 'input' in kw
        n = sum((has_in, has_out, has_err))
        if n not in (0, 3):
            missing = [s for s, v in (('stdin', has_in), ('stdout', has_out),
                                      ('stderr', has_err)) if not v]
            offenders.append(f'{rel}:{line} inherits {"+".join(missing)}')
    assert not offenders, (
        'these subprocess spawns leave some -- but not all -- stdio streams '
        'inherited, which is the shape that raises OSError [WinError 6] under '
        'pytest fd-capture on Windows: '
        + '; '.join(offenders)
        + '.  Add `stdin=subprocess.DEVNULL` beside `capture_output=True` '
          '(or name all three explicitly).')


def test_the_stdio_exemption_list_is_still_accurate():
    """Counter-pin: every exempted site must still BE a partial spawn.

    An exemption that no longer matches anything is a hole left open for no
    reason; it should be deleted when its owner lands the one-kwarg fix.
    """
    live = {(rel, line) for rel, line, kw in _subprocess_sites()}
    stale = sorted(s for s in _STDIO_EXEMPT if s not in live)
    assert not stale, (
        f'these stdio exemptions no longer point at a subprocess call: '
        f'{stale}.  Delete them -- either the call moved (re-measure) or the '
        f'owner fixed it (good, close the hole).')

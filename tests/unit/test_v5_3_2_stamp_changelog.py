"""v5.3.2 (ROADMAP V17 fix companion) -- stamp_changelog.py tests.

The ``scripts/stamp_changelog.py`` CLI ships the V17 walker's fix
side: when the walker DETECTS recursive self-citation drift between
the CHANGELOG block's at-write-time numbers and the ship-time
empirical numbers, this script REWRITES the CHANGELOG block to use
the empirical values.

These tests pin:

* **T1** -- the script is valid Python (importlib parses it).
* **T2** -- ``--help`` exits cleanly (argparse is wired correctly).
* **T3** -- a dry-run subprocess against the current CHANGELOG.md
  produces a sensible exit code (0/1/2; never 3 input error).
* **T4** -- a synthetic CHANGELOG with deliberately-wrong test counts
  is correctly stamped when ``--apply`` is passed.

The synthetic test (T4) is the load-bearing one -- it verifies the
end-to-end stamping behaviour against a temp CHANGELOG so the test
doesn't depend on the real CHANGELOG state at any given commit.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / 'scripts' / 'stamp_changelog.py'
_CHANGELOG = _REPO_ROOT / 'CHANGELOG.md'


# ===========================================================================
# T1 -- script parses as Python
# ===========================================================================

def test_script_parses_as_python():
    """``scripts/stamp_changelog.py`` must be valid Python.

    Use importlib's spec/loader pair so the script is parsed without
    being executed at module top-level (which would trigger argparse).
    A SyntaxError or ImportError here surfaces a regression in the
    script source before any other test runs it via subprocess.
    """
    assert _SCRIPT.is_file(), (
        f'stamp_changelog.py not found at {_SCRIPT}; the v5.3.2 '
        f'ship-time-stamp injection script is missing.')
    spec = importlib.util.spec_from_file_location(
        'stamp_changelog', _SCRIPT)
    assert spec is not None, 'spec_from_file_location returned None'
    module = importlib.util.module_from_spec(spec)
    assert module is not None
    # Loading the module executes its top-level statements, which
    # for this script is just imports + constant defs + function
    # defs -- no side effects.  argparse only runs in ``main()``.
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    # Sanity check: the public entry points are present.
    assert hasattr(module, 'main'), 'stamp_changelog has no main()'
    assert hasattr(module, 'stamp'), 'stamp_changelog has no stamp()'
    assert hasattr(module, '_latest_version_block'), (
        'stamp_changelog is missing the V17/V12 block-extraction '
        'primitive `_latest_version_block`.')


# ===========================================================================
# T2 -- --help exits cleanly
# ===========================================================================

def test_script_help_exits_cleanly():
    """``python scripts/stamp_changelog.py --help`` must exit 0.

    argparse exits with code 0 on ``--help``; any other code indicates
    a wiring bug (e.g. a required argument was added without a default).
    """
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), '--help'],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f'stamp_changelog.py --help returned rc={result.returncode}; '
        f'stdout=\n{result.stdout}\nstderr=\n{result.stderr}')
    # Spot-check: the help text mentions the dry-run-by-default contract.
    assert '--apply' in result.stdout, (
        '--help output should mention --apply (the dry-run override).')


# ===========================================================================
# T3 -- dry-run against the real CHANGELOG
# ===========================================================================

def test_dry_run_against_current_changelog():
    """A dry-run subprocess against the current CHANGELOG must not
    fail with rc=3 (input error).

    rc=0 = no drift; rc=1 = drift detected but --apply not set; rc=2 =
    skip-clean (no patterns to verify).  Any of those is acceptable
    here -- the test only pins "the script understands the current
    CHANGELOG well enough to parse its topmost block".

    The test uses ``--quick`` so it finishes in under a minute even on
    machines where the full suite takes 5+ minutes.
    """
    if not _CHANGELOG.is_file():
        pytest.skip(f'CHANGELOG.md not found at {_CHANGELOG}.')
    # The dry-run contract: regardless of rc, CHANGELOG.md is NOT modified.
    # Snapshot the size BEFORE the (single) subprocess and compare after.
    # (We deliberately don't checksum the whole file -- another process
    # could legitimately edit it during the run; size is a cheap structural
    # guard.)
    #
    # AUDIT_CI_TEST_TIME_2026_08_03 §4/chunk 6: this used to run the same
    # ~20 s subprocess TWICE and capture ``pre_size`` in BETWEEN them, so the
    # "dry run does not modify" assertion only ever covered the SECOND
    # invocation -- if the first (cold) run had rewritten the file, the test
    # would have snapshotted the ALREADY-MODIFIED size and passed.  Taking
    # the snapshot first covers the invocation that actually matters and
    # deletes one subprocess: strictly better coverage at half the cost.
    pre_size = _CHANGELOG.stat().st_size
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), '--quick'],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode in (0, 1, 2), (
        f'stamp_changelog.py --quick (dry-run) returned rc='
        f'{result.returncode}; expected 0/1/2.  '
        f'stdout=\n{result.stdout}\nstderr=\n{result.stderr}')
    post_size = _CHANGELOG.stat().st_size
    assert pre_size == post_size, (
        f'Dry-run modified CHANGELOG.md (size {pre_size} -> '
        f'{post_size}); the dry-run-by-default contract is broken.')


# ===========================================================================
# T4 -- synthetic CHANGELOG stamps correctly when --apply is passed
# ===========================================================================

def _make_synthetic_changelog(tmp_path, fake_pass, fake_coll,
                              fake_skip, fake_xfail):
    """Build a small CHANGELOG.md with a deliberately-wrong test-count
    self-citation, in the same shape the V17 walker matches.
    """
    body = textwrap.dedent(f"""\
        # Changelog -- lumenairy (synthetic test fixture)

        ## [9.9.9] -- 2026-01-01

        Synthetic CHANGELOG block for the v5.3.2 stamp_changelog test.

        ### Tests

        {fake_pass} unit tests pass (collected = {fake_coll} = pass + {fake_skip} skip + {fake_xfail} xfail).

        ## [9.9.8] -- 2025-12-31

        Previous fake release; provides the second-block anchor.
        """)
    cl = tmp_path / 'CHANGELOG.md'
    cl.write_text(body, encoding='utf-8')
    return cl


def test_synthetic_stamps_correctly(tmp_path, monkeypatch):
    """A temp CHANGELOG with bogus counts is rewritten to current ones.

    Strategy: import the script as a module, monkey-patch its
    ``_CHANGELOG`` constant + its empirical-count helpers, then call
    ``stamp(apply_changes=True)`` and assert the resulting file has
    the expected stamped numbers.

    This isolates the rewrite logic from the host machine's actual
    pytest collection and git state, which is the only way to write
    a deterministic stamping test.
    """
    # Load the script as a module.
    spec = importlib.util.spec_from_file_location(
        'stamp_changelog_under_test', _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    # Build the synthetic CHANGELOG with deliberately-wrong numbers.
    fake_pass, fake_coll, fake_skip, fake_xfail = 3000, 3001, 1, 0
    cl = _make_synthetic_changelog(
        tmp_path, fake_pass, fake_coll, fake_skip, fake_xfail)
    monkeypatch.setattr(module, '_CHANGELOG', cl)

    # Empirical numbers the script should stamp.  Only ``real_coll``
    # is actually fed to the patched ``_pytest_collect_count``; the
    # ``real_pass`` / ``real_skip`` / ``real_xfail`` symbols are
    # retained for spec-readability but unused at runtime (the
    # script computes pass from coll-skip-xfail in --quick mode).
    real_coll = 3866

    # In --quick mode, only the collected count is empirical; the
    # script computes pass = collected - skip - xfail and HOLDS the
    # synthetic skip/xfail constant.  Patch _pytest_collect_count
    # to return our fixed empirical collected total.  Return-tuple
    # shape: ``(pass_n, coll_n, skip_n, xfail_n)`` -- in --quick
    # mode the pass / skip / xfail slots are None.
    monkeypatch.setattr(
        module, '_pytest_collect_count',
        lambda: (None, real_coll, None, None))
    # Disable git plumbing (no PREV_TAG in a tmp dir).
    monkeypatch.setattr(module, '_resolve_prev_tag', lambda v: None)
    monkeypatch.setattr(module, '_git_changed_file_count',
                        lambda *a, **k: None)

    # Run in --apply mode with quick collection.
    rc = module.stamp(
        version=None, apply_changes=True, mode='quick',
        stream=sys.stdout,
    )
    assert rc == 0, (
        f'stamp --apply returned rc={rc}; expected 0 (clean write).')

    # Read the rewritten file and confirm the test-count line has
    # the stamped numbers.  The synthetic skip/xfail are preserved
    # (--quick mode does not refresh them); pass = collected - skip - xfail.
    expected_coll = real_coll
    expected_skip = fake_skip   # held constant in --quick
    expected_xfail = fake_xfail
    expected_pass = expected_coll - expected_skip - expected_xfail

    rewritten = cl.read_text(encoding='utf-8')
    pattern = re.compile(
        r'(\d{3,5})\s+unit tests pass[^(]*'
        r'\(\s*collected\s*=?\s*(\d{3,5})\s*=\s*'
        r'pass\s*\+\s*(\d+)\s+skip\s*\+\s*'
        r'(\d+)\s+xfail',
        re.IGNORECASE)
    match = pattern.search(rewritten)
    assert match is not None, (
        f'Stamped CHANGELOG no longer contains the canonical '
        f'``X pass + Y skip + Z xfail`` form; output was:\n{rewritten}')
    got_pass = int(match.group(1))
    got_coll = int(match.group(2))
    got_skip = int(match.group(3))
    got_xfail = int(match.group(4))
    assert (got_pass, got_coll, got_skip, got_xfail) == (
            expected_pass, expected_coll, expected_skip, expected_xfail), (
        f'Stamped numbers do not match empirical: got '
        f'({got_pass}, {got_coll}, {got_skip}, {got_xfail}); expected '
        f'({expected_pass}, {expected_coll}, {expected_skip}, '
        f'{expected_xfail}).')


def test_synthetic_dry_run_does_not_write(tmp_path, monkeypatch):
    """Dry-run against a drifted synthetic CHANGELOG must not write.

    Companion to T4 -- pins the inverse contract: drift IS detected
    (rc=1) but the file is NOT modified.
    """
    spec = importlib.util.spec_from_file_location(
        'stamp_changelog_dry_run', _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    cl = _make_synthetic_changelog(tmp_path, 3000, 3001, 1, 0)
    monkeypatch.setattr(module, '_CHANGELOG', cl)
    monkeypatch.setattr(
        module, '_pytest_collect_count',
        lambda: (None, 3866, None, None))
    monkeypatch.setattr(module, '_resolve_prev_tag', lambda v: None)
    monkeypatch.setattr(module, '_git_changed_file_count',
                        lambda *a, **k: None)

    original = cl.read_text(encoding='utf-8')

    rc = module.stamp(
        version=None, apply_changes=False, mode='quick',
        stream=sys.stdout,
    )
    assert rc == 1, (
        f'Dry-run against drifted synthetic CHANGELOG returned '
        f'rc={rc}; expected 1 (drift detected, --apply not set).')
    after = cl.read_text(encoding='utf-8')
    assert after == original, (
        'Dry-run mutated the synthetic CHANGELOG; dry-run-by-default '
        'contract is broken.')


def test_synthetic_full_mode_refreshes_skip_xfail(tmp_path, monkeypatch):
    """--full mode refreshes pass / skip / xfail empirically.

    Companion to T4 (which exercises --quick).  --full passes the
    real pytest run-result tuple; the script honours all four numbers
    instead of holding skip/xfail constant.
    """
    spec = importlib.util.spec_from_file_location(
        'stamp_changelog_full', _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    # Synthetic block has skip=1, xfail=0.
    cl = _make_synthetic_changelog(tmp_path, 3000, 3001, 1, 0)
    monkeypatch.setattr(module, '_CHANGELOG', cl)

    # Empirical: pass=3848, coll=3866, skip=17, xfail=1 (deliberately
    # different from the synthetic's 1 / 0 so we can verify the
    # refresh actually fires).
    monkeypatch.setattr(
        module, '_pytest_full_count',
        lambda: (3848, 3866, 17, 1))
    # --full mode should NOT call collect_count; patch it to error
    # if invoked, so we catch a routing bug.
    monkeypatch.setattr(
        module, '_pytest_collect_count',
        lambda: (_ for _ in ()).throw(AssertionError(
            '_pytest_collect_count called in --full mode')))
    monkeypatch.setattr(module, '_resolve_prev_tag', lambda v: None)
    monkeypatch.setattr(module, '_git_changed_file_count',
                        lambda *a, **k: None)

    rc = module.stamp(
        version=None, apply_changes=True, mode='full',
        stream=sys.stdout,
    )
    assert rc == 0, (
        f'stamp --apply --full returned rc={rc}; expected 0.')

    rewritten = cl.read_text(encoding='utf-8')
    pattern = re.compile(
        r'(\d+)\s+unit tests pass[^(]*'
        r'\(\s*collected\s*=?\s*(\d+)\s*=\s*'
        r'pass\s*\+\s*(\d+)\s+skip\s*\+\s*'
        r'(\d+)\s+xfail',
        re.IGNORECASE)
    match = pattern.search(rewritten)
    assert match is not None, 'Stamped block lost the canonical form.'
    got = tuple(int(g) for g in match.groups())
    assert got == (3848, 3866, 17, 1), (
        f'--full stamp did not refresh skip/xfail; got {got}, '
        f'expected (3848, 3866, 17, 1).')

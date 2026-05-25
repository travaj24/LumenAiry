"""v5.4.1 (AUDIT_V5_4_0 P2 #3) -- stamp_changelog.py Net-LOC tests.

v5.3.2 shipped ``scripts/stamp_changelog.py`` to close the recursive
self-citation drift class for test-counts / file-counts / CHANGELOG
line-counts.  The v5.3.2 audit P3-7 forecast a self-circular failure
mode for the file-count stamp: it uses ``git diff PREV_TAG..HEAD
--name-only`` which captures file membership but NOT line deltas.

v5.4.0 hit exactly this.  Phase 6 deleted a 619-line tombstone test
file AFTER stamp_changelog had already run.  The CHANGELOG was left
claiming ``Net LOC: +13759 / -452`` while the actual ship-time diff
was ``+13813 / -1075``.  The -623 deletion drift slipped past V17.2
because V17.2 only tolerances the ``+`` axis.

This module pins the v5.4.1 fix: stamp_changelog now also stamps
``Net LOC: +X / -Y vs vA.B.C`` (the ``vs vA.B.C`` is optional) using
``git diff PREV_TAG..HEAD --shortstat``.

Tests:

1. **test_stamp_writes_net_loc_when_pattern_present** -- end-to-end
   stamp against a synthetic CHANGELOG with deliberately-wrong Net
   LOC numbers and a synthetic git-shortstat returning the empirical
   pair.  Asserts the block now reads the empirical numbers.
2. **test_stamp_no_net_loc_pattern_no_change** -- block without a
   Net LOC line is left untouched; script exits cleanly.
3. **test_stamp_preserves_surrounding_text** -- block with bullet
   text wrapping the Net LOC sentence: only the two numbers change;
   surrounding text is byte-identical pre/post.
4. **test_stamp_net_loc_dry_run_reports_diff** -- dry-run mode
   surfaces the proposed Net LOC stamp in stdout AND does not
   touch the file on disk.
5. **test_stamp_idempotent** -- run twice in a row; second invocation
   reports no Net LOC change (the value is already in sync).

Additional spec-check tests:

* **test_net_loc_regex_matches_canonical_forms** -- direct regex
  unit covering the two real-world v5.x phrasings.
* **test_net_loc_regex_preserves_trailing_period** -- pins the
  bug-trap that ``[0-9.]+`` would consume the sentence-final dot.
* **test_git_net_loc_parses_shortstat** -- the ``_git_net_loc``
  helper extracts the right pair from the canonical git output,
  including the "no insertions" / "no deletions" edge cases.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_script_module(unique_suffix=''):
    """Load stamp_changelog.py as a fresh module per test.

    Using a unique suffix keeps each test's module isolated so that
    ``monkeypatch.setattr(module, ...)`` doesn't leak between tests.
    """
    spec = importlib.util.spec_from_file_location(
        f'stamp_changelog_net_loc_{unique_suffix}', _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _make_changelog_with_net_loc(tmp_path, ins, del_, suffix=' vs v1.2.3'):
    """Build a synthetic CHANGELOG.md with a Net-LOC self-citation.

    ``suffix`` controls the trailing version label; set to '' to
    exercise the no-version form.  Returns the path.
    """
    body = textwrap.dedent(f"""\
        # Changelog -- lumenairy (synthetic test fixture)

        ## [9.9.9] -- 2026-01-01

        Synthetic CHANGELOG block for the v5.4.1 Net-LOC stamp test.

        ### Files touched

        7 files modified or created.  Net LOC: +{ins} / -{del_}{suffix}.

        ## [9.9.8] -- 2025-12-31

        Previous fake release; provides the second-block anchor.
        """)
    cl = tmp_path / 'CHANGELOG.md'
    cl.write_text(body, encoding='utf-8')
    return cl


def _make_changelog_no_net_loc(tmp_path):
    """Build a synthetic CHANGELOG without a Net-LOC line."""
    body = textwrap.dedent("""\
        # Changelog -- lumenairy (synthetic test fixture)

        ## [9.9.9] -- 2026-01-01

        Synthetic CHANGELOG block with no Net-LOC citation.

        ### Tests

        3000 unit tests pass (collected = 3001 = pass + 1 skip + 0 xfail).

        ## [9.9.8] -- 2025-12-31

        Previous fake release.
        """)
    cl = tmp_path / 'CHANGELOG.md'
    cl.write_text(body, encoding='utf-8')
    return cl


# ---------------------------------------------------------------------------
# 1. Stamp writes Net LOC when pattern present
# ---------------------------------------------------------------------------

def test_stamp_writes_net_loc_when_pattern_present(tmp_path, monkeypatch):
    """End-to-end: drifted Net LOC + synthetic git-shortstat -> rewrite."""
    module = _load_script_module('writes')

    cl = _make_changelog_with_net_loc(tmp_path, ins=1000, del_=50,
                                      suffix=' vs v1.2.3')
    monkeypatch.setattr(module, '_CHANGELOG', cl)
    # Disable other empirical sources so only the Net-LOC stamp fires.
    monkeypatch.setattr(module, '_pytest_collect_count',
                        lambda: (None, None, None, None))
    monkeypatch.setattr(module, '_resolve_prev_tag', lambda v: 'v9.9.8')
    monkeypatch.setattr(module, '_git_changed_file_count',
                        lambda *a, **k: None)
    monkeypatch.setattr(module, '_git_net_loc',
                        lambda *a, **k: (1500, 200))

    rc = module.stamp(version=None, apply_changes=True, mode='quick',
                      stream=sys.stdout)
    assert rc == 0, f'stamp --apply returned rc={rc}; expected 0.'

    rewritten = cl.read_text(encoding='utf-8')
    assert 'Net LOC: +1500 / -200' in rewritten, (
        f'Net LOC stamp did not refresh; file is:\n{rewritten}')
    # The "vs v1.2.3" suffix must survive the rewrite.
    assert 'vs v1.2.3' in rewritten, (
        f'Version suffix was lost from Net LOC; file is:\n{rewritten}')
    # The OLD numbers must be gone (no zombie leftovers).
    assert '+1000 / -50' not in rewritten, (
        f'Old Net LOC numbers still present after stamp; file is:\n'
        f'{rewritten}')


# ---------------------------------------------------------------------------
# 2. No-pattern block: no change
# ---------------------------------------------------------------------------

def test_stamp_no_net_loc_pattern_no_change(tmp_path, monkeypatch):
    """A CHANGELOG block without a Net-LOC line stamps cleanly."""
    module = _load_script_module('no_pattern')

    cl = _make_changelog_no_net_loc(tmp_path)
    monkeypatch.setattr(module, '_CHANGELOG', cl)
    # Provide empirical data; the stamp should simply not find a
    # pattern to update.
    monkeypatch.setattr(module, '_pytest_collect_count',
                        lambda: (None, None, None, None))
    monkeypatch.setattr(module, '_resolve_prev_tag', lambda v: 'v9.9.8')
    monkeypatch.setattr(module, '_git_changed_file_count',
                        lambda *a, **k: None)
    monkeypatch.setattr(module, '_git_net_loc',
                        lambda *a, **k: (1500, 200))

    original = cl.read_text(encoding='utf-8')
    rc = module.stamp(version=None, apply_changes=True, mode='quick',
                      stream=sys.stdout)
    # rc=0 (clean) or rc=2 (skip-clean) are both acceptable when no
    # stampable patterns exist in the block.
    assert rc in (0, 2), (
        f'stamp on no-pattern block returned rc={rc}; expected 0 or 2.')
    after = cl.read_text(encoding='utf-8')
    assert after == original, (
        'CHANGELOG mutated despite having no Net-LOC pattern; '
        f'before:\n{original}\nafter:\n{after}')


# ---------------------------------------------------------------------------
# 3. Surrounding text preserved
# ---------------------------------------------------------------------------

def test_stamp_preserves_surrounding_text(tmp_path, monkeypatch):
    """Only the two numbers change; bullet / sentence context survives."""
    module = _load_script_module('preserve')

    # Build a block where the Net-LOC sentence is wrapped in distinctive
    # surrounding text we can string-search for verbatim after the stamp.
    pre_text = 'PROLOGUE-MARKER bullet item before the LOC line.'
    post_text = 'EPILOGUE-MARKER bullet item after the LOC line.'
    body = textwrap.dedent(f"""\
        # Changelog -- lumenairy (synthetic test fixture)

        ## [9.9.9] -- 2026-01-01

        Header line.

        ### Files touched

        - {pre_text}
        - 7 files modified or created.  Net LOC: +1000 / -50 vs v1.2.3.
        - {post_text}

        ## [9.9.8] -- 2025-12-31

        Previous.
        """)
    cl = tmp_path / 'CHANGELOG.md'
    cl.write_text(body, encoding='utf-8')

    monkeypatch.setattr(module, '_CHANGELOG', cl)
    monkeypatch.setattr(module, '_pytest_collect_count',
                        lambda: (None, None, None, None))
    monkeypatch.setattr(module, '_resolve_prev_tag', lambda v: 'v9.9.8')
    monkeypatch.setattr(module, '_git_changed_file_count',
                        lambda *a, **k: None)
    monkeypatch.setattr(module, '_git_net_loc',
                        lambda *a, **k: (1500, 200))

    rc = module.stamp(version=None, apply_changes=True, mode='quick',
                      stream=sys.stdout)
    assert rc == 0

    rewritten = cl.read_text(encoding='utf-8')
    assert pre_text in rewritten, (
        'Prologue text was modified by the stamp; '
        f'rewritten:\n{rewritten}')
    assert post_text in rewritten, (
        'Epilogue text was modified by the stamp; '
        f'rewritten:\n{rewritten}')
    # Numbers refreshed.
    assert 'Net LOC: +1500 / -200 vs v1.2.3' in rewritten, (
        f'Net LOC stamp text incorrect; rewritten:\n{rewritten}')
    # The trailing period after the version should be preserved.
    assert '-200 vs v1.2.3.' in rewritten, (
        f'Sentence-terminating period after version was lost; '
        f'rewritten:\n{rewritten}')


# ---------------------------------------------------------------------------
# 4. Dry-run reports diff in stdout, leaves file untouched
# ---------------------------------------------------------------------------

def test_stamp_net_loc_dry_run_reports_diff(tmp_path, monkeypatch, capsys):
    """Dry-run prints the proposed stamp; file on disk is unchanged."""
    module = _load_script_module('dry_run')

    cl = _make_changelog_with_net_loc(tmp_path, ins=1000, del_=50,
                                      suffix=' vs v1.2.3')
    monkeypatch.setattr(module, '_CHANGELOG', cl)
    monkeypatch.setattr(module, '_pytest_collect_count',
                        lambda: (None, None, None, None))
    monkeypatch.setattr(module, '_resolve_prev_tag', lambda v: 'v9.9.8')
    monkeypatch.setattr(module, '_git_changed_file_count',
                        lambda *a, **k: None)
    monkeypatch.setattr(module, '_git_net_loc',
                        lambda *a, **k: (1500, 200))

    original = cl.read_text(encoding='utf-8')
    rc = module.stamp(version=None, apply_changes=False, mode='quick',
                      stream=sys.stdout)
    assert rc == 1, (
        f'Dry-run with drift returned rc={rc}; expected 1 (drift '
        f'detected, --apply not set).')

    out = capsys.readouterr().out
    # The proposed-stamp label includes both the old and new numbers.
    assert '+1000' in out and '-50' in out, (
        f'Dry-run stdout did not include old numbers; out:\n{out}')
    assert '+1500' in out and '-200' in out, (
        f'Dry-run stdout did not include new numbers; out:\n{out}')
    # The diff preview must mention Net LOC.
    assert re.search(r'Net\s+LOC', out, re.IGNORECASE), (
        f'Dry-run stdout did not mention Net LOC; out:\n{out}')

    # File on disk: unchanged.
    after = cl.read_text(encoding='utf-8')
    assert after == original, (
        'Dry-run mutated the CHANGELOG; dry-run-by-default '
        'contract is broken.')


# ---------------------------------------------------------------------------
# 5. Idempotence
# ---------------------------------------------------------------------------

def test_stamp_idempotent(tmp_path, monkeypatch):
    """Running stamp twice -- second pass reports no changes."""
    module = _load_script_module('idempotent')

    cl = _make_changelog_with_net_loc(tmp_path, ins=1000, del_=50,
                                      suffix=' vs v1.2.3')
    monkeypatch.setattr(module, '_CHANGELOG', cl)
    monkeypatch.setattr(module, '_pytest_collect_count',
                        lambda: (None, None, None, None))
    monkeypatch.setattr(module, '_resolve_prev_tag', lambda v: 'v9.9.8')
    monkeypatch.setattr(module, '_git_changed_file_count',
                        lambda *a, **k: None)
    monkeypatch.setattr(module, '_git_net_loc',
                        lambda *a, **k: (1500, 200))

    # First pass: drift exists, --apply rewrites.
    rc1 = module.stamp(version=None, apply_changes=True, mode='quick',
                       stream=sys.stdout)
    assert rc1 == 0, f'First pass rc={rc1}; expected 0.'
    after_first = cl.read_text(encoding='utf-8')
    assert 'Net LOC: +1500 / -200' in after_first

    # Second pass with same empirical data: should detect no drift.
    rc2 = module.stamp(version=None, apply_changes=True, mode='quick',
                       stream=sys.stdout)
    # rc=0 = clean (no drift) OR rc=2 = skip-clean (no patterns at
    # all).  In this scenario the pattern still exists but matches
    # the empirical pair, so rc=0 is the expected branch.
    assert rc2 in (0, 2), (
        f'Idempotent second pass rc={rc2}; expected 0 or 2.')
    after_second = cl.read_text(encoding='utf-8')
    assert after_second == after_first, (
        'Second --apply pass mutated the CHANGELOG; stamp is not '
        f'idempotent.\nfirst:\n{after_first}\nsecond:\n{after_second}')


# ---------------------------------------------------------------------------
# Spec-check tests
# ---------------------------------------------------------------------------

def test_net_loc_regex_matches_canonical_forms():
    """The Net-LOC regex must match the two real-world phrasings."""
    module = _load_script_module('regex_canonical')
    pat = module._NET_LOC_PATTERN

    cases = [
        # (input, expected ins, expected del_, expected match text)
        (
            '52 files modified or created.  Net LOC: +13759 / -452 vs v5.3.2.',
            13759, 452, 'Net LOC: +13759 / -452 vs v5.3.2',
        ),
        (
            '16 files modified or created.  Net LOC: +2766 / -11.',
            2766, 11, 'Net LOC: +2766 / -11',
        ),
        (
            'Net LOC: +0 / -0',
            0, 0, 'Net LOC: +0 / -0',
        ),
    ]
    for text, exp_ins, exp_del, exp_match in cases:
        m = pat.search(text)
        assert m is not None, f'Regex did not match: {text!r}'
        assert int(m.group('ins')) == exp_ins, (
            f'ins mismatch for {text!r}: got '
            f'{m.group("ins")}, want {exp_ins}')
        assert int(m.group('del_')) == exp_del, (
            f'del_ mismatch for {text!r}: got '
            f'{m.group("del_")}, want {exp_del}')
        assert m.group(0) == exp_match, (
            f'Match span mismatch for {text!r}: got {m.group(0)!r}, '
            f'want {exp_match!r}')


def test_net_loc_regex_preserves_trailing_period():
    """Trailing sentence-period must NOT be consumed by the regex.

    Bug trap: an earlier draft used ``[0-9.]+`` for the version
    label which greedily consumed the sentence-terminating ``.`` --
    so a substitute would have produced "vs v5.3.2" with the period
    deleted.  This test pins the fix.
    """
    module = _load_script_module('regex_period')
    pat = module._NET_LOC_PATTERN
    text = 'Net LOC: +13813 / -1075 vs v5.3.2.'
    m = pat.search(text)
    assert m is not None
    # The match must end BEFORE the sentence-terminating period.
    assert text[m.end():] == '.', (
        f'Regex consumed the sentence-terminating period; '
        f'tail was {text[m.end():]!r}, expected ".".')
    # Substituting and reassembling must restore the period.
    rebuilt = (
        text[:m.start()]
        + m.group('pre') + '999' + m.group('mid') + '888' + m.group('suf')
        + text[m.end():]
    )
    assert rebuilt == 'Net LOC: +999 / -888 vs v5.3.2.', (
        f'Rebuilt text lost the trailing period: {rebuilt!r}')


def test_git_net_loc_parses_shortstat(monkeypatch):
    """``_git_net_loc`` extracts (ins, del_) from canonical git output."""
    module = _load_script_module('shortstat')

    cases = [
        # (subprocess stdout, expected ins, expected del_)
        (' 54 files changed, 13813 insertions(+), 1075 deletions(-)\n',
         13813, 1075),
        (' 1 file changed, 42 insertions(+)\n', 42, 0),
        (' 1 file changed, 17 deletions(-)\n', 0, 17),
        (' 3 files changed, 100 insertions(+), 200 deletions(-)\n',
         100, 200),
    ]
    for fake_out, exp_ins, exp_del in cases:
        monkeypatch.setattr(module, '_run',
                            lambda *a, **k: (0, fake_out))
        got_ins, got_del = module._git_net_loc('v0.0.1')
        assert (got_ins, got_del) == (exp_ins, exp_del), (
            f'_git_net_loc({fake_out!r}) returned '
            f'({got_ins}, {got_del}); expected ({exp_ins}, {exp_del}).')

    # Subprocess failure: return (None, None).
    monkeypatch.setattr(module, '_run', lambda *a, **k: (1, ''))
    got = module._git_net_loc('v0.0.1')
    assert got == (None, None), (
        f'_git_net_loc on failed git invocation returned {got}; '
        f'expected (None, None).')


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def test_script_help_still_exits_cleanly():
    """v5.4.1 additions must not break --help.

    Regression guard for the v5.3.2 ``test_script_help_exits_cleanly``
    contract: argparse stays wired correctly.
    """
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), '--help'],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f'--help returned rc={result.returncode}; stamp_changelog '
        f'is broken.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}')

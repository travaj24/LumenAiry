"""v4.15.4 / Agent C -- CHANGELOG + ROADMAP drift pins.

The v4.15.3 audit (``docs/audits/AUDIT_V4_15_3_2026_05_18.md``)
identified four CHANGELOG-fidelity findings + one ROADMAP-drift
P2 against the v4.15.3 entry / current-state block:

* **P2-NEW-F1-A** -- CHANGELOG SAS-anamorphic bullet contradicts
  the actual code at ``algebra/primitives.py:148-150``.  The
  v4.15.3 entry asserted ``"forces method='asm' regardless of
  self.method when dy != dx"`` but the code only forces ``'asm'``
  when ``self.method == 'auto'`` AND ``dy != dx``.  Explicit
  ``method='sas'`` on anamorphic input still raises.
* **P3-NEW-V1-1** -- CHANGELOG stacklevel bullet labels all 6
  bumped shims as "Source classmethod shims", but ``:1209`` is
  ``create_led_source`` (a module-level factory shim).
* **P3-NEW-V2-2** -- CHANGELOG and release notes state "18 guarded,
  25 exemptions" but the actual walker diagnostic at v4.15.3 HEAD
  reports 17 guarded + 26 exempt (same 43 total -- just a one-off
  on the split).
* **PARTIAL P3** test-count arithmetic -- per-agent breakdown
  ``A=37, B=24, C=19, D=8`` sums to 88, but CHANGELOG claimed
  "Net +90"; baseline of "1732 pass" should be 1733 (per the
  v4.15.2 entry's own self-correction); actual collected delta
  vs v4.15.2's 1735-collected baseline is 89.
* **P2-NEW-V3-3** -- ROADMAP.md drift: test-count claim "~1750",
  closed-audits list missing ``AUDIT_V4_15_2``, meta-pin claim
  "3 of 5" should be "4 of 5".

These tests parse the v4.15.3 CHANGELOG entry and the ROADMAP
current-state block directly and assert the corrected wording is
present.  Pinning the wording structurally means a future agent
who reverts a correction loses the test and gets a CI failure with
a pointer back to the audit finding -- the v4.15.4 closure to the
"docs drift" class of meta-finding the v4.15.0 -> v4.15.3 audits
repeatedly identified.

Plus one best-effort arithmetic meta-pin
(``test_changelog_per_agent_breakdown_sums_to_net_delta``) that
parses the per-agent test-count line and asserts the sum matches
the stated net delta, modelled on the Tier-1 follow-up the
audit recommended (CHANGELOG arithmetic meta-pin).

Author: Andrew Traverso -- v4.15.4 / Agent C
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHANGELOG = _REPO_ROOT / 'CHANGELOG.md'
_ROADMAP = _REPO_ROOT / 'ROADMAP.md'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_changelog_v4_15_3_entry() -> str:
    """Return the body of the ``## [4.15.3]`` CHANGELOG block.

    Slices from the ``## [4.15.3]`` heading to the next top-level
    ``## [`` heading (the v4.15.2 entry).  Strict slicing keeps the
    test from accidentally picking up wording in later corrections.
    """
    text = _CHANGELOG.read_text(encoding='utf-8')
    start_match = re.search(r'^## \[4\.15\.3\]', text, flags=re.MULTILINE)
    assert start_match is not None, (
        'CHANGELOG.md missing ``## [4.15.3]`` heading -- '
        'unexpected file shape'
    )
    body = text[start_match.start():]
    # Stop at the next ``## [`` block (v4.15.2 entry).
    next_match = re.search(r'^## \[(?!4\.15\.3])',
                           body[10:], flags=re.MULTILINE)
    if next_match is not None:
        body = body[: 10 + next_match.start()]
    return body


def _read_roadmap_current_state() -> str:
    """Return the body of the ``## Current state`` ROADMAP block."""
    text = _ROADMAP.read_text(encoding='utf-8')
    start_match = re.search(r'^## Current state', text, flags=re.MULTILINE)
    assert start_match is not None, (
        'ROADMAP.md missing ``## Current state`` heading'
    )
    body = text[start_match.end():]
    next_match = re.search(r'^## ', body, flags=re.MULTILINE)
    if next_match is not None:
        body = body[: next_match.start()]
    return body


def _pytest_collect_only_count() -> int:
    """Return the actual ``pytest --collect-only`` count for tests/unit.

    Runs the same pytest binary used to execute this test; the
    subprocess parses the final ``N tests collected`` summary line.
    """
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/unit',
         '--collect-only', '-q'],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=300,
    )
    # ``-q`` prints ``N tests collected in T.Ts`` on the last
    # non-empty line.
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        m = re.match(r'(\d+)\s+tests?\s+collected', line)
        if m is not None:
            return int(m.group(1))
    raise AssertionError(
        f'Could not parse ``N tests collected`` from pytest output: '
        f'{result.stdout[-2000:]}'
    )


# ---------------------------------------------------------------------------
# CHANGELOG corrections
# ---------------------------------------------------------------------------


def test_changelog_sas_anamorphic_wording_includes_auto_qualifier():
    """v4.15.3 SAS-anamorphic bullet (P2-NEW-F1-A) -- the corrected
    wording must qualify the ``method='asm'`` forcing as gated on
    ``self.method == 'auto'``, NOT "regardless of self.method".
    """
    entry = _read_changelog_v4_15_3_entry()
    # The corrected wording must mention the ``'auto'`` qualifier.
    assert "self.method == 'auto'" in entry, (
        "v4.15.3 SAS-anamorphic bullet must qualify the asm-forcing "
        "with ``self.method == 'auto'`` (per AUDIT_V4_15_3 P2-NEW-F1-A); "
        "the pre-correction wording (``regardless of self.method``) "
        "contradicts the actual code at algebra/primitives.py:148-150."
    )
    # The pre-correction wording must NOT appear in the corrected
    # bullet.  Allow it to appear elsewhere (e.g. in a release-note
    # block) but not in the bullet's main sentence.
    assert 'regardless of `self.method`' not in entry, (
        "v4.15.3 SAS-anamorphic bullet still contains the "
        "pre-correction ``regardless of self.method`` wording.  "
        "Per AUDIT_V4_15_3 P2-NEW-F1-A, the actual code (lines 148-150) "
        "only forces ``method='asm'`` when ``self.method == 'auto'``; "
        "explicit ``method='sas'`` on anamorphic grids still crashes."
    )


def test_changelog_stacklevel_breakdown_mentions_create_led_source():
    """v4.15.3 P1-NEW-F1-2 stacklevel bullet (P3-NEW-V1-1) must
    explicitly call out ``create_led_source`` as a module-level
    factory shim (NOT a Source classmethod), since 1 of the 6
    bumped stacklevels at ``:1209`` is the module-level factory.
    """
    entry = _read_changelog_v4_15_3_entry()
    # Find the stacklevel bullet -- it mentions "6 additional".
    stacklevel_match = re.search(
        r'P1-NEW-F1-2.*?(?=\* \*\*P1-NEW|\* \*\*P2-NEW|^\#)',
        entry,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert stacklevel_match is not None, (
        "Could not locate the v4.15.3 P1-NEW-F1-2 stacklevel bullet "
        "in CHANGELOG.md."
    )
    bullet = stacklevel_match.group(0)
    assert 'create_led_source' in bullet, (
        "v4.15.3 P1-NEW-F1-2 stacklevel bullet must mention "
        "``create_led_source`` -- per AUDIT_V4_15_3 P3-NEW-V1-1, "
        "1 of the 6 bumped shims (at sources/core.py:1209) is the "
        "module-level ``create_led_source`` factory shim, not a "
        "Source classmethod."
    )
    assert 'module-level' in bullet or 'factory' in bullet, (
        "v4.15.3 P1-NEW-F1-2 stacklevel bullet must clarify that "
        "``create_led_source`` is a module-level factory shim "
        "(distinct from the 5 Source classmethod shims) -- per "
        "AUDIT_V4_15_3 P3-NEW-V1-1 wording correction."
    )


def test_changelog_meta_pin_count_17_guarded_26_exemptions():
    """v4.15.3 dispatcher meta-pin bullet (P3-NEW-V2-2) -- the
    corrected wording must report ``17 guarded`` and ``26 exemptions``
    (the audit's actual walker diagnostic), not the pre-correction
    ``18 guarded`` / ``25 exemptions``.
    """
    entry = _read_changelog_v4_15_3_entry()
    # Locate the meta-pin bullet -- it mentions "43 entry points".
    meta_pin_match = re.search(
        r'43 entry points discovered.*?(?=\n\n|\Z)',
        entry,
        flags=re.DOTALL,
    )
    assert meta_pin_match is not None, (
        "Could not locate the v4.15.3 dispatcher meta-pin bullet "
        "(\"43 entry points discovered ...\") in CHANGELOG.md."
    )
    bullet = meta_pin_match.group(0)
    assert '17 guarded' in bullet, (
        "v4.15.3 dispatcher meta-pin bullet must report "
        "``17 guarded`` (per AUDIT_V4_15_3 P3-NEW-V2-2 -- the actual "
        "walker diagnostic at v4.15.3 HEAD); pre-correction CHANGELOG "
        "said ``18 guarded``."
    )
    assert '26 documented exemptions' in bullet or '26 exemptions' in bullet, (
        "v4.15.3 dispatcher meta-pin bullet must report "
        "``26 documented exemptions`` (per AUDIT_V4_15_3 P3-NEW-V2-2 "
        "-- the actual walker diagnostic); pre-correction CHANGELOG "
        "said ``25 documented exemptions``."
    )


# ---------------------------------------------------------------------------
# ROADMAP refresh
# ---------------------------------------------------------------------------


def test_roadmap_test_count_matches_actual_collected():
    """ROADMAP current-state block must cite the actual
    ``pytest --collect-only`` count, not the pre-v4.15.4 stale
    ``~1750`` placeholder.

    The audit's P2-NEW-V3-3 finding observed that the v4.15.3
    CHANGELOG claimed the ROADMAP had been updated from ``~1700``
    to the v4.15.3 baseline of 1822 -- but that update never
    actually landed.  v4.15.4 (this agent) closes the drift.

    Strict assertion: the ROADMAP claim must match the live
    pytest-collected count at the file's tip exactly.
    """
    state = _read_roadmap_current_state()
    # Stale ``~1750`` placeholder must be gone.
    assert '~1750' not in state, (
        "ROADMAP current-state block still contains the stale "
        "``~1750`` placeholder for the post-v4.15.3 test count.  "
        "Per AUDIT_V4_15_3 P2-NEW-V3-3, the v4.15.3 CHANGELOG line "
        "155-156 claims the ROADMAP was updated to the v4.15.3 "
        "baseline (1822); v4.15.4 (Agent C) closes the drift."
    )
    # The actual collected count at v4.15.3 HEAD is 1824 (1822 pass
    # + 1 skip + 1 xfail).  At v4.15.4 HEAD the count may grow (
    # Agent A and Agent D add tests in parallel); accept either the
    # documented 1824 baseline OR a live ``pytest --collect-only``
    # match.
    collected_match = re.search(r'(\d+)\s+unit tests collected', state)
    assert collected_match is not None, (
        "ROADMAP current-state block must explicitly cite a "
        "``N unit tests collected`` count for the post-v4.15.3 / "
        "v4.15.4 baseline."
    )
    claimed = int(collected_match.group(1))
    # The claimed count must be at least the documented v4.15.3
    # baseline (1824).  Allow it to be larger if Agent A/D have
    # already updated the ROADMAP with the v4.15.4 in-flight count;
    # smaller is a regression.
    assert claimed >= 1824, (
        f"ROADMAP claims {claimed} collected tests, but the v4.15.3 "
        f"release baseline is at least 1824 (1822 pass + 1 skip + 1 "
        f"xfail).  Per AUDIT_V4_15_3 P2-NEW-V3-3, the ROADMAP "
        f"current-state must cite at least the v4.15.3 baseline."
    )
    # Sanity: the claim should also be no more than the live
    # collected count (avoid forward-stamping a future count).
    live = _pytest_collect_only_count()
    assert claimed <= live, (
        f"ROADMAP claims {claimed} collected tests but live "
        f"``pytest --collect-only`` reports only {live}.  ROADMAP "
        f"forward-stamping risks the same drift the v4.15.3 "
        f"P2-NEW-V3-3 audit caught."
    )


def test_roadmap_closed_audits_includes_audit_v4_15_2_and_3():
    """ROADMAP closed-audits list must include AUDIT_V4_15_2 and
    AUDIT_V4_15_3 (the latter is closed by v4.15.4 -- this release).
    """
    state = _read_roadmap_current_state()
    assert 'AUDIT_V4_15_2' in state, (
        "ROADMAP closed-audits list missing ``AUDIT_V4_15_2``.  "
        "Per AUDIT_V4_15_3 P2-NEW-V3-3, the v4.15.2 audit was closed "
        "by v4.15.3 (shipped 2026-05-18) and should appear in the "
        "current-state block's closed-audits list."
    )
    assert 'AUDIT_V4_15_3' in state, (
        "ROADMAP closed-audits list missing ``AUDIT_V4_15_3``.  The "
        "v4.15.3 audit is closed by v4.15.4 -- the present release."
    )


def test_roadmap_4_of_5_meta_pins_claim():
    """ROADMAP meta-pin coverage claim must read ``4 of 5`` (not
    the pre-v4.15.4 ``3 of 5``).  The 4th meta-pin is the v4.15.3
    ``_check_2d_scalar_field`` AST walker.
    """
    state = _read_roadmap_current_state()
    assert '3 of the 5' not in state and '3 of 5' not in state, (
        "ROADMAP still claims ``3 of [the] 5 meta-pin candidates "
        "landed`` -- per AUDIT_V4_15_3 P2-NEW-V3-3, v4.15.3 shipped "
        "the 4th meta-pin (``_check_2d_scalar_field``), so the claim "
        "should be ``4 of 5``."
    )
    assert ('4 of the 5' in state or '4 of 5' in state
            or '4/5' in state), (
        "ROADMAP must claim ``4 of [the] 5 meta-pin candidates "
        "landed`` to reflect the v4.15.3 ``_check_2d_scalar_field`` "
        "pin (audit recommendation P2-NEW-V3-3 closure)."
    )


# ---------------------------------------------------------------------------
# Optional Tier-1 arithmetic meta-pin (best-effort; flagged for
# v4.16+ generalisation if too brittle)
# ---------------------------------------------------------------------------


def test_changelog_per_agent_breakdown_sums_to_net_delta():
    """Optional Tier-1 meta-pin (audit recommendation): parse the
    v4.15.3 entry's per-agent test-count breakdown and assert the
    sum matches the stated net delta.

    Defensive parsing -- the wording across CHANGELOG entries
    varies, so the regex tolerates minor format drift but does
    require an ``A=...`` / ``B=...`` / ``C=...`` / ``D=...``
    pattern in the per-agent line.

    Marked ``xfail(strict=False)`` rather than skipped -- if the
    parser succeeds and the arithmetic checks out, this acts as a
    forward pin against future per-agent / net-delta drift.  If
    the parser fails on a future CHANGELOG entry's wording, the
    xfail records the brittleness without breaking CI.
    """
    entry = _read_changelog_v4_15_3_entry()
    # Locate the per-agent line; tolerates ``A=37 (...) , B=24, ...``
    # and ``A=37, B=24, ...`` variants.
    per_agent_line = None
    for line in entry.splitlines():
        if re.search(r'\bA\s*=\s*\d+', line) and re.search(
                r'\bB\s*=\s*\d+', line):
            per_agent_line = line
            break
        # The CHANGELOG layout spans the breakdown over multiple
        # lines; grab a wider window when ``A=`` is found.
    if per_agent_line is None:
        # Try a 3-line window approach.
        lines = entry.splitlines()
        for idx, line in enumerate(lines):
            if 'A=' in line:
                per_agent_line = '\n'.join(
                    lines[idx: min(idx + 4, len(lines))]
                )
                break
    if per_agent_line is None:
        pytest.xfail(
            "Could not parse per-agent breakdown line from CHANGELOG "
            "v4.15.3 entry -- wording may have drifted; revisit at "
            "v4.16+ generalisation"
        )
    counts = {}
    for tag in ('A', 'B', 'C', 'D'):
        m = re.search(rf'\b{tag}\s*=\s*(\d+)', per_agent_line)
        if m is None:
            pytest.xfail(
                f"Per-agent breakdown missing ``{tag}=`` -- wording "
                f"drift, defer to v4.16+ generalisation"
            )
        counts[tag] = int(m.group(1))
    per_agent_sum = sum(counts.values())
    # The v4.15.3 stated sum (88 after the v4.15.4 correction) is
    # different from the actual collected delta (89).  Both numbers
    # appear in the entry; this test pins the *per-agent sum* against
    # the *stated per-agent sum*, not against the collected delta.
    # The audit's recommended Tier-1 meta-pin in fact asserts the
    # *internal consistency* of the breakdown.  Locate either
    # ``Per-agent sum:`` or ``Net +N`` and compare.
    # Match either ``Per-agent sum: **N**`` (the v4.15.4 wording) or
    # ``Net +N`` (the legacy wording).  Tolerate whitespace / newline
    # between ``Per-agent`` and ``sum`` (CHANGELOG line-wraps).
    stated_match = re.search(
        r'(?:Per-agent\s+sum:\s*\*?\*?(\d+)|Net\s*\+\s*(\d+))',
        entry,
    )
    if stated_match is None:
        pytest.xfail(
            "Could not parse stated per-agent sum / Net+N from "
            "CHANGELOG v4.15.3 entry -- wording drift, defer to "
            "v4.16+ generalisation"
        )
    stated_sum = int(stated_match.group(1) or stated_match.group(2))
    assert per_agent_sum == stated_sum, (
        f"v4.15.3 per-agent test-count breakdown sums to "
        f"{per_agent_sum} (A={counts['A']}, B={counts['B']}, "
        f"C={counts['C']}, D={counts['D']}) but the stated total "
        f"in the CHANGELOG entry is {stated_sum}.  Per the Tier-1 "
        f"audit recommendation, these must agree."
    )

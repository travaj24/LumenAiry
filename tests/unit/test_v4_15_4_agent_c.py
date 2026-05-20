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
    """ROADMAP current-state block must NOT carry the stale
    ``~1750`` placeholder for the post-v4.15.3 test count.

    v4.16.1 (audit P1-NEW-2WAY-1 / C.6): release-agnostic rewrite.
    Pre-rewrite this test pinned both the exact placeholder strings
    AND an exact lower bound (``>= 1824``) and an upper bound
    matching the live ``pytest --collect-only`` count.  Both bounds
    drift as new releases land (the v4.16.0 baseline is 2106).

    The structural invariants this test now pins:

    * The stale ``~1750`` placeholder is gone.
    * Some test count IS cited in the section.  The shape of the
      claim can be either ``N unit tests collected`` (the v4.15.4
      wording) OR ``N unit tests passing`` (the v4.16.0 wording).
    * The cited number is >= 1824 (the v4.15.3 documented baseline
      floor; lower than that would be a regression below the audit
      closure target).
    * No upper bound (the live ``pytest --collect-only`` count is
      no longer checked, as it drifts with each release and creates
      a brittle pin that fails every cycle).

    The original AUDIT_V4_15_3 P2-NEW-V3-3 closure -- that the
    ROADMAP cites a current test count -- remains pinned in the
    invariants above.
    """
    state = _read_roadmap_current_state()
    # Stale ``~1750`` placeholder must be gone.
    assert '~1750' not in state, (
        "ROADMAP current-state block still contains the stale "
        "``~1750`` placeholder for the post-v4.15.3 test count.  "
        "Per AUDIT_V4_15_3 P2-NEW-V3-3, refresh per the closure."
    )
    # Some test count must be cited -- accept either the v4.15.4
    # ``collected`` wording or the v4.16.0 ``passing`` wording.
    collected_match = re.search(
        r'(\d+)\s+unit tests\s+(?:collected|passing|pass)',
        state,
        flags=re.IGNORECASE,
    )
    assert collected_match is not None, (
        "ROADMAP current-state block must explicitly cite a "
        "``N unit tests collected`` (or ``passing``) count for the "
        "library baseline."
    )
    claimed = int(collected_match.group(1))
    # The claimed count must be at least the documented v4.15.3
    # baseline (1824).  Lower than that would be a regression below
    # the audit closure target.  No upper bound -- the live count
    # drifts each release and forward-stamping is fine as long as
    # the floor is preserved.
    assert claimed >= 1824, (
        f"ROADMAP claims {claimed} collected tests, but the v4.15.3 "
        f"release baseline is at least 1824 (1822 pass + 1 skip + 1 "
        f"xfail).  Per AUDIT_V4_15_3 P2-NEW-V3-3, the ROADMAP "
        f"current-state must cite at least the v4.15.3 baseline."
    )


def test_roadmap_closed_audits_includes_audit_v4_15_2_and_3():
    """ROADMAP closed-audits list must include AUDIT_V4_15_2 and
    AUDIT_V4_15_3 -- pinning the audit closure list rather than the
    closure VERSION (which drifts with releases).

    v4.16.1 (audit P1-NEW-2WAY-1 / C.6): release-agnostic rewrite.
    Pre-rewrite this test asserted both audits were ``in state``
    of the current-state section; v4.16.0 rolled the closed-audits
    list into a single-sentence "AUDIT_V4_12_1 through AUDIT_V4_15_4
    all closed" wording that DOESN'T enumerate the individual audit
    names.  The new form scans BOTH the explicit-list wording AND
    the ``X through Y`` range wording: a range
    ``AUDIT_V4_X through AUDIT_V4_Y`` is accepted if the requested
    audit falls within the (X, Y) inclusive range.

    The original AUDIT_V4_15_3 P2-NEW-V3-3 closure -- that the
    ROADMAP records v4.15.2 and v4.15.3 closures -- remains pinned.
    """
    state = _read_roadmap_current_state()
    for audit_id in ('AUDIT_V4_15_2', 'AUDIT_V4_15_3'):
        if audit_id in state:
            continue
        # Range wording ``AUDIT_V4_A_B through AUDIT_V4_X_Y all closed``.
        # Extract the (A, B) -> (X, Y) range and verify the requested
        # audit's (m, p) tuple is contained within.
        range_match = re.search(
            r'AUDIT_V4_(\d+)_(\d+)\s+through\s+AUDIT_V4_(\d+)_(\d+)',
            state,
        )
        if range_match is not None:
            a, b = int(range_match.group(1)), int(range_match.group(2))
            x, y = int(range_match.group(3)), int(range_match.group(4))
            # Parse the requested audit's (m, p) e.g. AUDIT_V4_15_2 -> (15, 2)
            wanted_match = re.match(r'AUDIT_V4_(\d+)_(\d+)', audit_id)
            assert wanted_match is not None
            wm = int(wanted_match.group(1))
            wp = int(wanted_match.group(2))
            in_range = (a, b) <= (wm, wp) <= (x, y)
            assert in_range, (
                f"ROADMAP closed-audits range "
                f"AUDIT_V4_{a}_{b} through AUDIT_V4_{x}_{y} does not "
                f"include {audit_id} (per AUDIT_V4_15_3 P2-NEW-V3-3)."
            )
            continue
        raise AssertionError(
            f"ROADMAP closed-audits list missing {audit_id} (neither "
            f"as an explicit name nor in a ``X through Y`` range "
            f"wording).  Per AUDIT_V4_15_3 P2-NEW-V3-3, the v4.15.2 "
            f"and v4.15.3 closures must appear in the current-state "
            f"block."
        )


def test_roadmap_4_of_5_meta_pins_claim():
    """ROADMAP meta-pin coverage section must NOT regress to a
    pre-v4.15.4 ``3 of 5`` claim -- it must claim AT LEAST 4 of the
    candidate meta-pins have landed.

    v4.16.1 (audit P1-NEW-2WAY-1 / C.6): release-agnostic rewrite.
    Pre-rewrite this test pinned the exact ``4 of 5`` wording; the
    v4.16.0 ROADMAP refactored the meta-pin section to list each
    meta-pin individually (V1, V2, ..., V9) and no longer uses the
    "X of N" enumeration wording.

    The new form accepts either:

    * The original ``N of M`` wording with N >= 4 and M >= 5, OR
    * An itemised list (numbered or bulleted) of at least 4 meta-pins
      under a ``Meta-pin coverage`` heading / bullet.

    Either form satisfies the AUDIT_V4_15_3 P2-NEW-V3-3 closure
    (which was: the ROADMAP records the v4.15.3
    ``_check_2d_scalar_field`` meta-pin as the 4th landing).
    """
    state = _read_roadmap_current_state()
    # The stale ``3 of [the] 5`` claim must be gone.
    assert '3 of the 5' not in state and '3 of 5' not in state, (
        "ROADMAP still claims ``3 of [the] 5 meta-pin candidates "
        "landed`` -- per AUDIT_V4_15_3 P2-NEW-V3-3, the 4th meta-pin "
        "(``_check_2d_scalar_field``) landed in v4.15.3."
    )
    # First form: ``N of [the] M`` wording -- extract N and require N >= 4.
    n_of_m = re.search(
        r'(\d+)\s+of\s+(?:the\s+)?(\d+)\s+meta-pin', state, flags=re.IGNORECASE)
    if n_of_m is not None:
        n = int(n_of_m.group(1))
        m = int(n_of_m.group(2))
        assert n >= 4, (
            f"ROADMAP meta-pin claim ``{n} of {m}`` is below the "
            f"v4.15.3 baseline of 4 (AUDIT_V4_15_3 P2-NEW-V3-3)."
        )
        return
    # Second form: itemised list of >= 4 meta-pin items under a
    # ``Meta-pin coverage`` section.  Each item is a bullet like
    # ``- V1: ...`` / ``  - V4: ...`` etc.
    coverage_match = re.search(
        r'(?:Meta-pin\s+coverage|meta-pin\s+coverage|meta-pin\s+candidates)',
        state, flags=re.IGNORECASE)
    assert coverage_match is not None, (
        "ROADMAP must include a meta-pin coverage section (either as "
        "``N of M`` wording, or a ``Meta-pin coverage`` heading "
        "followed by a bullet list of at least 4 meta-pins).  "
        "AUDIT_V4_15_3 P2-NEW-V3-3 closure."
    )
    # Count meta-pin item lines.  Items follow the ``V<digit>:`` or
    # ``V<digit> :`` shorthand convention used since v4.14.0.
    meta_pin_items = re.findall(r'\bV\d+(?:\s*:|\s+\()', state)
    assert len(meta_pin_items) >= 4, (
        f"ROADMAP meta-pin coverage section names only "
        f"{len(meta_pin_items)} meta-pin items (V<digit>); expected "
        f">= 4 per AUDIT_V4_15_3 P2-NEW-V3-3 (the v4.15.3 "
        f"``_check_2d_scalar_field`` is the 4th)."
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

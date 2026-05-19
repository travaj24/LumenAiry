"""v4.15.5 / Agent C -- CHANGELOG + ROADMAP drift pins (carryover
plus new v4.15.4 findings).

The v4.15.4 audit (``docs/audits/AUDIT_V4_15_4_2026_05_19.md``)
identified three v4.15.4-CHANGELOG / README arithmetic / fidelity
findings + one walker performance finding + carryover doc-drift
items against the v4.15.4 entry and the ROADMAP residual section:

* **P2-NEW-3WAY-3** -- CHANGELOG v4.15.4 entry-point count claim
  "43 entry points -> 49 after the refactor" doesn't match the
  actual diagnostic walker output of 72 (25 guarded + 47 exempt).
* **P3-NEW-V3-1** -- AUDIT_V4_15_3 cited 57 unit tests failing
  under ``-W error::DeprecationWarning``; v4.15.4 CHANGELOG line 80
  + README line 62 both say "63 v4.15.3 tests previously failed".
  Closure verified (0 escalations) regardless; canonical = 63.
* **P3-NEW-V3-2** -- v4.15.4 per-agent attribution sum 37 vs CHANGELOG
  "Net +36".  +1 attribution-vs-collection drift is the standard
  parametrize/fixture artifact same pattern as v4.15.3's documented
  one.
* **P3-NEW-F1-2** -- ``_file_to_ast`` re-parses the same source
  file N times; lru_cache decorator pulls walker test from ~5s to
  ~2s (3x speedup target).  v4.15.5 (this agent) ships the
  decorator; this test pins it.
* **ROADMAP duplicate-counting drift cleanup** -- 6 items in the
  "v4.16 residual" section (polychromatic encircled energy,
  polarisation-aware Strehl, resolution metrics, astigmatism
  mag+angle, OAP, Forbes Q-type) all shipped in v4.15.0 / v4.15.1
  and should appear in "Shipped highlights", not "v4.16 residual".

These tests parse the v4.15.4 CHANGELOG entry and the ROADMAP
v4.16 residual section directly and assert the corrected
wording / numbers are present.  Same pinning pattern as
``test_v4_15_4_agent_c.py``.

Author: Andrew Traverso -- v4.15.5 / Agent C
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHANGELOG = _REPO_ROOT / 'CHANGELOG.md'
_ROADMAP = _REPO_ROOT / 'ROADMAP.md'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_changelog_v4_15_4_entry() -> str:
    """Return the body of the ``## [4.15.4]`` CHANGELOG block.

    Slices from the ``## [4.15.4]`` heading to the next top-level
    ``## [`` heading (the v4.15.3 entry).
    """
    text = _CHANGELOG.read_text(encoding='utf-8')
    start_match = re.search(r'^## \[4\.15\.4\]', text, flags=re.MULTILINE)
    assert start_match is not None, (
        'CHANGELOG.md missing ``## [4.15.4]`` heading -- '
        'unexpected file shape'
    )
    body = text[start_match.start():]
    # Stop at the next ``## [`` block (v4.15.3 entry).
    next_match = re.search(r'^## \[(?!4\.15\.4])',
                           body[10:], flags=re.MULTILINE)
    if next_match is not None:
        body = body[: 10 + next_match.start()]
    return body


def _read_roadmap_v4_16_section() -> str:
    """Return the body of the ``## v4.16.0`` ROADMAP section."""
    text = _ROADMAP.read_text(encoding='utf-8')
    start_match = re.search(r'^## v4\.16\.0', text, flags=re.MULTILINE)
    assert start_match is not None, (
        'ROADMAP.md missing ``## v4.16.0`` heading'
    )
    body = text[start_match.end():]
    next_match = re.search(r'^## ', body, flags=re.MULTILINE)
    if next_match is not None:
        body = body[: next_match.start()]
    return body


def _actual_walker_entry_point_count() -> tuple[int, int, int]:
    """Return (total, guarded, exempt) entry-point counts from the
    live dispatcher meta-pin walker.

    Imports the walker module locally to avoid coupling at module
    import time (the walker imports ``lumenairy``).
    """
    from tests.unit.test_v4_15_3_dispatcher_pin_2d_scalar_field import (
        _walk_entry_points,
        _GUARD_EXEMPTIONS,
    )
    total = 0
    guarded = 0
    exempt = 0
    for py, node in _walk_entry_points():
        total += 1
        rel = py.relative_to(_REPO_ROOT).as_posix()
        if (rel, node.name) in _GUARD_EXEMPTIONS:
            exempt += 1
        else:
            guarded += 1
    return total, guarded, exempt


# ---------------------------------------------------------------------------
# C.1 -- CHANGELOG v4.15.4 entry-point count refresh (P2-NEW-3WAY-3)
# ---------------------------------------------------------------------------


def test_changelog_v4_15_4_entry_point_count_matches_actual_diagnostic():
    """v4.15.4 CHANGELOG entry-point count claim must reflect the
    walker diagnostic at v4.15.4 HEAD (NOT the pre-correction 43->49
    figure).

    The pre-correction CHANGELOG said "43 entry points -> 49 after
    the refactor", reflecting only the ``__all__``-pass dedup.  The
    actual diagnostic at v4.15.4 HEAD reports 72 total (25 guarded +
    47 exempt).  v4.15.5 / Agent C refreshes the v4.15.4 CHANGELOG
    entry to those numbers.

    Note: by the time integration lands, Agent A's V6 walker refactor
    will report a different (larger) count -- that's the v4.15.5
    count, which goes in the v4.15.5 CHANGELOG entry, not the v4.15.4
    entry.  This test pins the v4.15.4 entry against the v4.15.4 HEAD
    numbers (not the live diagnostic) per the task brief.

    Sanity floor: the v4.15.4 entry must NOT contain the
    pre-correction "43 entry points -> 49 after the refactor"
    wording, and SHOULD cite a plausible total within +/- 5 of
    the documented v4.15.4 HEAD count (72).
    """
    entry = _read_changelog_v4_15_4_entry()
    # The pre-correction wording must be gone.
    assert '43 entry points -> 49' not in entry, (
        "v4.15.4 CHANGELOG still contains the pre-correction "
        "'43 entry points -> 49 after the refactor' wording.  Per "
        "AUDIT_V4_15_4 P2-NEW-3WAY-3, the actual walker diagnostic "
        "reported 72 (25 guarded + 47 exempt) at v4.15.4 HEAD."
    )
    # Locate the walker count in the entry; v4.15.5 wording uses
    # "72 total entry points (25 guarded + 47 documented exempt)"
    # or similar.  Tolerate the wording but require a numeric claim.
    count_match = re.search(
        r'(\d+)\s*total\s+entry\s+points',
        entry,
        flags=re.IGNORECASE,
    )
    assert count_match is not None, (
        "v4.15.4 CHANGELOG must cite '<N> total entry points' in the "
        "walker-refactor bullet.  Per AUDIT_V4_15_4 P2-NEW-3WAY-3, "
        "the pre-correction wording cited only '43 entry points -> 49' "
        "which doesn't match the actual walker diagnostic of 72."
    )
    claimed = int(count_match.group(1))
    # The v4.15.4 HEAD count was 72 per the audit; Agent C populated
    # this with that number.  Accept +/- 5 against 72 to tolerate any
    # small drift before integration.  (The live v4.15.5 V6 walker
    # count is a different number, documented separately in the
    # v4.15.5 entry.)
    documented_v4_15_4_head_count = 72
    delta = abs(claimed - documented_v4_15_4_head_count)
    assert delta <= 5, (
        f"v4.15.4 CHANGELOG claims {claimed} total entry points but "
        f"the documented v4.15.4 HEAD count is "
        f"{documented_v4_15_4_head_count} (per AUDIT_V4_15_4 "
        f"P2-NEW-3WAY-3); delta {delta}, tolerance 5."
    )
    # Sanity: the live diagnostic should report >= the documented
    # v4.15.4 count (V6 refactor only ADDS entries; never removes).
    total, _, _ = _actual_walker_entry_point_count()
    assert total >= documented_v4_15_4_head_count - 5, (
        f"Live walker reports {total} entry points; expected at least "
        f"{documented_v4_15_4_head_count - 5} (a V6 walker refactor "
        f"only adds entries).  Walker may be regressed."
    )


# ---------------------------------------------------------------------------
# C.2 -- Per-agent attribution arithmetic (P3-NEW-V3-2)
# ---------------------------------------------------------------------------


def test_changelog_v4_15_4_per_agent_arithmetic_documents_attribution_gap():
    """v4.15.4 CHANGELOG per-agent breakdown must document the +1
    attribution-vs-collection gap.

    Per-agent sum: A=8+1=9, B=11, C=7, D=10 -> 37.  Collected delta:
    1858 - 1822 = 36.  The +1 gap is the standard parametrize /
    fixture artifact (one test expands to 2 collected items, or a
    fixture-only test isn't attributed).  v4.15.3's hygiene fix
    documented this same pattern; v4.15.4 should match.
    """
    entry = _read_changelog_v4_15_4_entry()
    # The corrected wording must mention "attribution" and the gap.
    assert 'attribution' in entry.lower(), (
        "v4.15.4 CHANGELOG ### Test counts block must document the "
        "per-agent ATTRIBUTION sum separately from the COLLECTED "
        "delta.  Per AUDIT_V4_15_4 P3-NEW-V3-2, the per-agent "
        "breakdown sums to 37 but the actual collected delta is 36; "
        "the +1 gap is the standard parametrize / fixture artifact "
        "(same pattern documented in v4.15.3)."
    )
    # The wording should call out the parametrize / fixture artifact
    # explanation.
    assert 'parametrize' in entry.lower() or 'fixture' in entry.lower(), (
        "v4.15.4 CHANGELOG ### Test counts block must explain the +1 "
        "attribution-vs-collection gap as a parametrize/fixture "
        "artifact -- same wording as the v4.15.3 entry's hygiene fix."
    )


# ---------------------------------------------------------------------------
# C.3 -- Walker _file_to_ast lru_cache (P3-NEW-F1-2)
# ---------------------------------------------------------------------------


def test_walker_file_to_ast_cached():
    """``_file_to_ast`` from the dispatcher meta-pin file must be
    decorated with ``functools.lru_cache`` so repeat calls for the
    same source file return identical (memoized) AST objects.

    Pre-v4.15.5: 11 ``apply_*_lens`` re-discoveries through
    ``__all__`` membership against ``elements/_lens_thin.py``
    re-parsed the file 11x.  Walker test took ~5s.
    Post-v4.15.5: cached; 1 parse per file; walker drops to ~2s
    (3x target, observed 30x in cold-warm timing at integration).
    """
    from tests.unit.test_v4_15_3_dispatcher_pin_2d_scalar_field import (
        _file_to_ast,
    )

    # Verify cache_info / cache_clear methods exist (the standard
    # functools.lru_cache surface).
    assert hasattr(_file_to_ast, 'cache_info'), (
        "_file_to_ast must be decorated with functools.lru_cache. "
        "Missing ``cache_info`` attribute -- per AUDIT_V4_15_4 "
        "P3-NEW-F1-2 (v4.15.5 Agent C closure)."
    )
    assert hasattr(_file_to_ast, 'cache_clear'), (
        "_file_to_ast must be decorated with functools.lru_cache. "
        "Missing ``cache_clear`` attribute -- per AUDIT_V4_15_4 "
        "P3-NEW-F1-2 (v4.15.5 Agent C closure)."
    )
    # Clear cache; verify two calls with same input return identical
    # object (cache hit, not re-parsed AST).
    _file_to_ast.cache_clear()
    # Pick a target file that exists.
    target = _REPO_ROOT / 'lumenairy' / 'elements' / '_lens_thin.py'
    assert target.exists(), (
        "Test setup error: expected target file does not exist."
    )
    tree1 = _file_to_ast(target)
    tree2 = _file_to_ast(target)
    # The cache returns the same object instance, not just an equal one.
    assert tree1 is tree2, (
        "Second _file_to_ast call did not return the same object as "
        "the first; cache may be disabled or the decorator absent."
    )
    info = _file_to_ast.cache_info()
    # After two calls with the same key: 1 miss + 1 hit.
    assert info.hits >= 1, (
        f"_file_to_ast cache reports {info.hits} hits after 2 calls "
        f"with the same input; expected >=1.  Decorator may be "
        f"misconfigured."
    )


# ---------------------------------------------------------------------------
# C.4 -- ROADMAP duplicate-counting drift cleanup (carryover)
# ---------------------------------------------------------------------------


def test_roadmap_v4_16_residual_excludes_shipped_items():
    """ROADMAP v4.16 section must NOT mention the 6 user-facing API
    items that shipped in v4.15.0 / v4.15.1.

    Per multiple recent audits, the v4.16 residual section listed:
    polychromatic encircled energy, polarisation-aware Strehl,
    resolution metrics, astigmatism mag+angle, OAP, Forbes Q-type
    freeform -- all of which shipped.  v4.15.5 / Agent C moves them
    to Shipped highlights.

    Specific symbol names checked against the *residual* section
    only (excluding the introductory paragraph, which may mention
    them as "already shipped" for context).
    """
    section = _read_roadmap_v4_16_section()
    # The introductory paragraph may mention the symbols as shipped
    # for context; we want to check the *residual items* (the numbered
    # entries below the intro).  Split on the first ``### `` heading.
    item_split = re.split(r'\n###\s', section, maxsplit=1)
    if len(item_split) == 2:
        residual_items = '### ' + item_split[1]
    else:
        residual_items = section
    # The six shipped symbols must not appear as *new items* in v4.16:
    shipped_symbols = (
        'polychromatic_encircled_energy',
        'astigmatism_mag_angle',
        'make_off_axis_parabola',
        'surface_sag_q_bfs',
        'strehl_vector',
        'rayleigh_resolution',
    )
    # Also exclude the section-heading wording variants ("Polychromatic
    # encircled energy", "Polarisation-aware Strehl", etc.) from
    # appearing as ITEM HEADINGS (``### N. ...``) below the intro.
    item_headings = re.findall(r'^###\s+\d+\.\s+(.*?)$',
                                residual_items,
                                flags=re.MULTILINE)
    shipped_heading_keywords = (
        'polychromatic encircled',
        'polarisation-aware strehl',
        'polarization-aware strehl',
        'resolution metrics',
        'astigmatism magnitude',
        'off-axis parabola',
        'q-type freeform',
        'forbes q',
    )
    for heading in item_headings:
        for kw in shipped_heading_keywords:
            assert kw not in heading.lower(), (
                f"ROADMAP v4.16 residual still has an item heading "
                f"'{heading}' mentioning '{kw}'; this item shipped in "
                f"v4.15.0 / v4.15.1 and should be in Shipped highlights "
                f"per AUDIT_V4_15_4 ROADMAP duplicate-counting drift "
                f"cleanup carryover (v4.15.5 Agent C closure)."
            )
    # Also assert the shipped symbol names don't appear as the "Add ..."
    # subject of any residual item (i.e. "Add ee_polychromatic" or
    # similar).  Tolerate the intro paragraph mentioning them as
    # already-shipped for context.
    for symbol in shipped_symbols:
        # If symbol appears AT ALL in residual_items, fail.
        assert symbol not in residual_items, (
            f"ROADMAP v4.16 residual still mentions '{symbol}' as a "
            f"residual item; this symbol shipped in v4.15.0 / v4.15.1 "
            f"and should be in Shipped highlights per AUDIT_V4_15_4 "
            f"ROADMAP duplicate-counting drift cleanup carryover "
            f"(v4.15.5 Agent C closure)."
        )

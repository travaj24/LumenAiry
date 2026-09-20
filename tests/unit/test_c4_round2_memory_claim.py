"""WP-C4 round 2, V-C4-D3 -- the report's cross-build memory claim, checked
against the JSON it was measured from.

WHAT THE DEFECT WAS.  ``WP-C4_MFT_DIRECT_DEFAULT_REPORT.md`` section 3 said
"the two builds' readings are IDENTICAL TO THE BYTE at every shape (the
``identical WIN/WSL`` column is ``yes`` at 42 of 42)".  The column printed
immediately above that sentence reads ``NO`` at all 42 rows, and the two JSON
files the section was measured from agree byte for byte at 0 of 42 shapes --
by a factor of 9.98 at ``N = 64, M = 16``.  The claim that IS true, and the
one the rule actually rests on, is about the ORDERING: which route is
cheapest is identical on both builds at every shape, because it follows from
the padding law and not from a run.

A prose sentence and the data under it can drift apart silently, and this one
did in three places at once (the sentence, its own table, and the CHANGELOG).
This file makes the sentence a measurement:

* the ORDERING agrees across builds at every shape, in the branch's own
  42-shape ladder and in round 2's 51-shape one;
* the READINGS agree at none of them, on either ladder;
* the report and the CHANGELOG say so, and do not say the opposite.

The second claim is the one that can rot in the "wrong direction": if a future
tracemalloc made the two builds agree exactly, the corrected sentence would be
wrong too, and this file would fail rather than let it stand.

Author:  Andrew Traverso
"""
from __future__ import annotations

import json
import os

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_BRANCH_PROBE = os.path.join(_REPO, 'validation', 'probe_c4_mft_direct')
_ROUND2_PROBE = os.path.join(_REPO, 'validation', 'probe_c4_round2')
_REPORT = os.path.join(
    _REPO, 'docs', 'audits', 'AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11',
    'fixes', 'WP-C4_MFT_DIRECT_DEFAULT_REPORT.md')
_CHANGELOG = os.path.join(_REPO, 'CHANGELOG.md')


def _load(path):
    assert os.path.exists(path), f"{path} is not committed"
    with open(path, encoding='cp1252') as fh:
        return json.load(fh)


def _branch_ladder():
    """``{(N, M): {build: row}}`` from the branch's own square memory ladder."""
    out = {}
    for build in ('win', 'wsl'):
        data = _load(os.path.join(_BRANCH_PROBE,
                                  f"c4_ladder_mem_{build}.json"))
        for row in data['rows']:
            out.setdefault((row['N'], row['M']), {})[build] = row
    return out


def _round2_ladder():
    """``{name: {build: row}}`` from round 2's 51-shape census."""
    out = {}
    for build in ('win', 'wsl'):
        data = _load(os.path.join(_ROUND2_PROBE,
                                  f"r2_memcensus_all_{build}.json"))
        for row in data['rows']:
            out.setdefault(row['name'], {})[build] = row
    return out


def test_the_reports_cross_build_memory_claim_matches_the_committed_json():
    """The sentence and the data, in one assertion each way.

    MEASURED 2026-09-20 and re-measured in round 2: the branch's 42-shape
    ladder agrees on ``dense_cheapest`` at 42 of 42 and on the READINGS at 0
    of 42; round 2's 51-shape ladder agrees on the cheapest route at 51 of 51
    and on the readings at 0 of 51.
    """
    branch = _branch_ladder()
    assert len(branch) == 42, (
        f"the branch's memory ladder has {len(branch)} shapes, not the 42 the "
        f"report's section 3 is written about")
    same_reading = [k for k, v in branch.items()
                    if v['win']['peak_bytes'] == v['wsl']['peak_bytes']]
    same_order = [k for k, v in branch.items()
                  if v['win']['dense_cheapest'] == v['wsl']['dense_cheapest']]
    assert len(same_order) == len(branch), (
        f"the two builds disagree about which route is cheapest at "
        f"{sorted(set(branch) - set(same_order))}.  The ORDERING is the "
        f"build-free half of the memory argument -- it follows from the "
        f"padding law, not from a run -- so if it has stopped being "
        f"build-free the rule's memory half needs re-deriving, not this test "
        f"relaxing")
    assert same_reading == [], (
        f"the two builds' tracemalloc readings now agree byte for byte at "
        f"{sorted(same_reading)}.  The report says they do NOT (0 of 42, by "
        f"up to a factor of 10 at N=64, M=16); re-measure and re-word section "
        f"3 rather than editing this expectation")

    round2 = _round2_ladder()
    assert len(round2) >= 42, f"round 2's census has only {len(round2)} shapes"
    r2_same_reading = [k for k, v in round2.items()
                       if v['win']['peak_bytes'] == v['wsl']['peak_bytes']]
    r2_same_cheapest = [k for k, v in round2.items()
                        if v['win']['cheapest'] == v['wsl']['cheapest']]
    assert len(r2_same_cheapest) == len(round2), (
        f"round 2's census disagrees across builds about the cheapest route "
        f"at {sorted(set(round2) - set(r2_same_cheapest))}")
    assert r2_same_reading == [], (
        f"round 2's readings now agree byte for byte at "
        f"{sorted(r2_same_reading)}; the corrected sentence in section 3 "
        f"would then be wrong in the other direction")


def test_the_memory_half_no_longer_argues_against_the_captured_region():
    """V-C4-D1's memory half, after the second condition landed.

    The claim the report makes is NOT "the dense route is the smallest at
    every shape" -- it is not, at nine thin shapes on both builds, and
    ``2048x64 -> 64x2`` reads 5.264 MB against the separable route's 4.399 MB.
    The claim is that the dense route is the smallest at every shape the rule
    CAPTURES, which is what the rule's memory argument actually needs.
    """
    round2 = _round2_ladder()
    for build in ('win', 'wsl'):
        offenders = [name for name, v in round2.items()
                     if v[build]['rule_says_direct']
                     and v[build]['cheapest'] != 'dense']
        assert offenders == [], (
            f"{build}: the rule captures {offenders}, where the dense route "
            f"is NOT the cheapest of the three in tracemalloc peak.  At those "
            f"shapes 'auto' takes a route that is larger AND (V-C4-D1) slower, "
            f"and the only argument left for it is accuracy")
    # ... and the exception the verification found is still there, REFUSED --
    # a one-sided assertion would pass on a census that had stopped measuring.
    thin = 'w0400_2048x64b'                       # 2048x64 -> 64x2
    assert thin in round2, (
        f"{thin} is no longer on the memory ladder; it is the shape "
        f"VERIFY-WP-C4 D1 and D3 both rest on and it has to stay measured")
    for build in ('win', 'wsl'):
        row = round2[thin][build]
        assert not row['rule_says_direct'], (
            f"{build}: the rule captures {thin} again, where the dense route "
            f"is the LARGER one")
        assert row['cheapest'] != 'dense', (
            f"{build}: the dense route is now the cheapest at {thin} "
            f"({row['peak_MB']}); the memory exception this file is written "
            f"about has gone, so re-measure section 3 before trusting it")


@pytest.mark.parametrize('doc', ('report', 'changelog'))
def test_the_prose_does_not_carry_the_refuted_cross_build_sentence(doc):
    """The documents themselves.  A corrected measurement that leaves the old
    sentence somewhere else is not corrected."""
    path = _REPORT if doc == 'report' else _CHANGELOG
    # utf-8 with a replacing fallback: the CHANGELOG carries a few non-cp1252
    # bytes from older entries and this id reads prose, not bytes.
    with open(path, encoding='utf-8', errors='replace') as fh:
        text = fh.read()
    refuted = 'readings are identical to the byte across builds'
    assert refuted not in text.lower(), (
        f"{os.path.basename(path)} still claims the two builds' memory "
        f"readings are identical to the byte.  They agree at 0 of 42 shapes "
        f"in the very JSON the claim was measured from; what is build-free is "
        f"the ORDERING")
    assert 'ordering' in text.lower(), (
        f"{os.path.basename(path)} no longer says what IS build-free about "
        f"the memory half")

"""``.test_durations`` freshness gate (audit 2026-09-11, V5 / TESTS-ARCH P2-3).

``.test_durations`` is not a convenience file -- it is load-bearing CI
infrastructure.  ``unit-tests.yml`` and ``publish.yml`` both shard with
``pytest-split --splitting-algorithm least_duration --durations-path
.test_durations``, and pytest-split gives every id it cannot find in that file
a DEFAULT weight.  The more ids are missing, the closer the split gets to a
count split.  MEASURED by the audit on the same tree: a greedy 5-way split on
the committed durations is exact (max/min = **1.000**, 2 627.3 s per shard),
while an alphabetical split -- what you get with no durations at all -- is
**3.85x** imbalanced (70.4 / 34.6 / 68.8 / 18.3 / 26.8 min).  A shard handed
far more than its share is exactly the failure behind the two 30-minute
publish-verify timeouts documented in ``unit-tests.yml``.

So staleness has a direct, measured cost, and nothing was watching it.  This
file is the watch.  The audit's own remedy was "a CI step that fails when > 2 %
of collected ids are missing"; it is a TEST instead of a workflow step so it
fires in every developer's local run too, and so a regeneration is verified by
the same gate that demanded it.

THE 2 % BAR, derived rather than picked.  Untimed ids are not free: pytest-split
weights them at a default, so the balancer's error grows with their share.  At
the measured mean of ~1.0 s per timed id, 2 % of ~14 000 ids is ~280 ids
carrying ~280 s of unmodelled work -- about 5 % of one 5-shard fast lane, which
is inside the 1.5x headroom the shard count is chosen for.  Ten times that
(20 %) is ~2 800 s, which is a whole shard, i.e. a guaranteed cap hit.  The bar
sits ~decade below the failure and ~decade above "one new test file landed".

REGENERATION (the workflow's own procedure, ``unit-tests.yml`` v5.32.1): run
SERIALLY on an IDLE machine with ``OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=
MKL_NUM_THREADS=1`` for BOTH lanes, so every entry is on one scale --

    python -m pytest tests/unit -m "not integration and not slow" \\
        --store-durations --durations-path .test_durations
    python -m pytest tests/unit -m "slow and not integration" \\
        --store-durations --durations-path .test_durations_slow
    # then merge the two JSON objects into .test_durations

RUNTIME BUDGET: one ``--collect-only`` of ``tests/unit`` in a subprocess.
MEASURED 2026-09-12 on this workstation: 14.3-17.2 s for ~14 000 ids.  The
audit measured 137 s for the same collection on a fully saturated box, which is
why the subprocess timeout below is 600 s rather than 60.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DURATIONS = os.path.join(REPO_ROOT, '.test_durations')

# Fraction of collected ids permitted to carry no timing.
_MAX_MISSING_FRACTION = 0.02
# Collected ids are normalised to forward slashes; pytest prints them with the
# platform separator.
_COLLECT_TIMEOUT_S = 600


def _collect_ids() -> list[str]:
    """Collect ``tests/unit`` in a SUBPROCESS and return the node ids.

    A subprocess, not an in-process ``pytest.main``: re-entering pytest inside a
    running session shares plugin state, the capture stack and ``sys.modules``,
    and a collection error there would be attributed to this test rather than
    to the file that caused it.

    All three stdio streams are named explicitly (``capture_output`` covers
    stdout/stderr, ``stdin=DEVNULL`` covers the third).  On Windows,
    ``Popen._get_handles`` resolves any stream left as ``None`` through
    ``GetStdHandle()`` and then ``DuplicateHandle``; pytest's default
    fd-capture reassigns the process's fds without calling ``SetStdHandle``, so
    those Win32 handles can be stale and the duplication raises
    ``OSError: [WinError 6] The handle is invalid`` before the child exists.
    MEASURED 2026-09-12 under this suite's own configuration: with
    ``stdout=PIPE`` alone the spawn raises WinError 6; with all three named it
    does not.  That is why this helper never inherits a stream.
    """
    env = dict(os.environ)
    # The workstation's OpenBLAS is ~400x slower unpinned; collection imports
    # every test module, and several import paths touch BLAS at import time.
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    # Do not let the child write a pytest cache next to a read-only checkout,
    # and do not let it inherit an -x / -k from a parent invocation.
    env.pop('PYTEST_ADDOPTS', None)
    env.pop('PYTEST_CURRENT_TEST', None)
    proc = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/unit',
         '--collect-only', '-q', '-p', 'no:cacheprovider'],
        cwd=REPO_ROOT,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=_COLLECT_TIMEOUT_S,
        env=env,
    )
    ids = []
    for line in proc.stdout.splitlines():
        line = line.strip().replace('\\', '/')
        if line.startswith('tests/') and '::' in line:
            ids.append(line)
    if not ids:
        raise AssertionError(
            f"collection subprocess produced no node ids (rc={proc.returncode}).\n"
            f"--- stdout tail ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-2000:]}")
    return ids


@pytest.fixture(scope='module')
def collected_ids():
    return _collect_ids()


@pytest.fixture(scope='module')
def timed_ids():
    with open(DURATIONS, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    assert isinstance(data, dict) and data, (
        ".test_durations must be a non-empty JSON object mapping node id -> "
        "seconds; pytest-split reads it as one.")
    return data


def test_durations_file_exists_and_parses(timed_ids):
    """The file itself is intact: a JSON object of id -> positive float."""
    bad = [k for k, v in timed_ids.items()
           if not isinstance(v, (int, float)) or v < 0]
    assert not bad[:10] and not bad, (
        f"{len(bad)} .test_durations entries are not non-negative numbers, "
        f"e.g. {bad[:5]}.  pytest-split will not be able to weight them.")
    assert all('::' in k for k in list(timed_ids)[:50]), (
        ".test_durations keys must be pytest node ids (path::test).")


def test_durations_covers_at_least_98_percent_of_collected_ids(
        collected_ids, timed_ids):
    """> 2 % of collected ids with no timing = regenerate ``.test_durations``.

    See the module docstring for the derivation of the bar and for the exact
    regeneration procedure.  This gate is expected to FIRE after a campaign
    that adds test files -- that is the gate working, not a defect in it; the
    remedy is one regeneration run, not a wider bar.
    """
    timed = set(timed_ids)
    missing = [t for t in collected_ids if t not in timed]
    frac = len(missing) / len(collected_ids)

    by_file = Counter(t.split('::')[0] for t in missing)
    worst = ', '.join('%s (%d)' % (f.replace('tests/unit/', ''), n)
                      for f, n in by_file.most_common(8))
    assert frac <= _MAX_MISSING_FRACTION, (
        f".test_durations is STALE: {len(missing)} of {len(collected_ids)} "
        f"collected ids ({frac:.2%}) carry no timing, above the "
        f"{_MAX_MISSING_FRACTION:.0%} bar.  pytest-split weights every untimed "
        f"id at a default, so the CI shard split degrades toward a count split "
        f"(measured: exact 1.000 balance on fresh durations vs 3.85x on none) "
        f"and a shard eventually hits its step cap with zero failures.\n"
        f"Worst files: {worst}.\n"
        f"Fix: regenerate serially on an idle machine with BLAS pinned to one "
        f"thread, for BOTH lanes -- see this module's docstring for the exact "
        f"commands.")


def test_durations_holds_few_ids_that_no_longer_exist(collected_ids, timed_ids):
    """The other direction: timings for ids that have been renamed or deleted.

    Orphans are much cheaper than gaps -- pytest-split simply never looks them
    up -- so the bar is looser (10 %).  A large orphan share means the file was
    captured against a different tree entirely, which makes its WEIGHTS
    untrustworthy even for the ids it does cover.
    """
    collected = set(collected_ids)
    orphans = [t for t in timed_ids if t not in collected]
    frac = len(orphans) / max(len(timed_ids), 1)
    assert frac <= 0.10, (
        f".test_durations holds {len(orphans)} timings ({frac:.2%}) for node "
        f"ids that no longer collect, e.g. {sorted(orphans)[:5]}.  Above ~10 % "
        f"the file describes a different tree and its surviving weights should "
        f"not be trusted either; regenerate rather than hand-edit.")


def test_the_staleness_gate_is_not_vacuous(collected_ids, timed_ids):
    """Counter-pin: the comparison really does read both sides.

    A gate that silently compared an empty collection, or normalised the two
    sides to nothing, would pass forever.  This asserts both inputs are
    substantial and that they OVERLAP -- i.e. the id spellings on the two sides
    are actually comparable (a separator or rootdir change would make every id
    "missing" or every id "orphan", which is a gate failure, not a data
    finding).
    """
    assert len(collected_ids) > 1000, (
        f"only {len(collected_ids)} ids collected; the subprocess did not "
        f"collect the unit suite.")
    assert len(timed_ids) > 1000, (
        f"only {len(timed_ids)} timings; .test_durations is not the committed "
        f"file.")
    overlap = len(set(collected_ids) & set(timed_ids))
    assert overlap > 0.5 * min(len(collected_ids), len(timed_ids)), (
        f"only {overlap} ids are common to the collection and the durations "
        f"file.  That is an id-SPELLING mismatch (path separator, rootdir, or "
        f"a parametrisation id change), not staleness -- the two staleness "
        f"assertions above would be reading noise.")

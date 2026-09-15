"""Probe -- the slow lane's pytest-split budget at 5 vs 8 shards, and whether
``.test_durations`` actually covers the slow selection.

CI run 34914295323: all five slow-lane shards were killed at the 1 800 s step
cap having run 1 847-1 863 s with no summary line; shard 2 was at 65 % of its
selection and shard 5 at 98 %.

This probe measures the two things the remedy depends on:

1. COVERAGE.  pytest-split's ``least_duration`` gives any test with no
   ``.test_durations`` entry the AVERAGE of the known durations for the
   selected set (``algorithms._get_items_with_durations`` ->
   ``_get_avg_duration_per_test``).  A heavy file with no entry is therefore
   under-weighted by exactly the amount that recreates a pileup, so a split
   count is only worth changing if the entries exist.

2. BALANCE.  Re-runs pytest-split's own greedy algorithm (sort the selected
   ids by duration descending, assign each to the currently-lightest group)
   over the real slow selection at ``--splits`` 3, 5, 8 and 10, and reports
   the per-shard totals against the 1 800 s and 2 700 s step caps.

Run:
  cd /c/tmp/lum_reds && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 PYTHONPATH=/c/tmp/lum_reds \
  python validation/probe_known_reds/t3_slow_lane_split_probe.py

Writes ``t3_slow_lane_split_probe_slow.json`` beside this file.
"""
import collections
import heapq
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
_WINSEP = chr(92)


def _norm(nodeid):
    return nodeid.split('::')[0].replace(_WINSEP, '/')


def collect(marker_expr):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=REPO)
    proc = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/unit', '-m', marker_expr,
         '--collect-only', '-q', '-p', 'no:cacheprovider', '--capture=sys'],
        cwd=REPO, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ids = [ln.strip() for ln in proc.stdout.splitlines() if '::' in ln]
    return ids, proc.returncode, proc.stdout.strip().splitlines()[-3:]


def least_duration_groups(pairs, splits):
    """pytest-split's LeastDurationAlgorithm, on (id, duration) pairs."""
    ordered = sorted(pairs, key=lambda t: str(t[0]))
    ordered = sorted(ordered, key=lambda t: t[1], reverse=True)
    heap = [(0.0, i) for i in range(splits)]
    heapq.heapify(heap)
    groups = [[] for _ in range(splits)]
    for nid, dur in ordered:
        total, gi = heapq.heappop(heap)
        groups[gi].append((nid, dur))
        heapq.heappush(heap, (total + dur, gi))
    return groups


def main():
    ids, rc, tail = collect('slow and not integration')
    with open(os.path.join(REPO, '.test_durations'), encoding='utf-8') as fh:
        dur = json.load(fh)

    # .test_durations keys use forward slashes; a Windows collection reports
    # backslashes.  Compare on a normalised key so the gap count is a real
    # gap and not a path-separator artefact.
    dur_norm = {k.replace(_WINSEP, '/'): v for k, v in dur.items()}
    ids_norm = [i.replace(_WINSEP, '/') for i in ids]

    missing = [i for i in ids_norm if i not in dur_norm]
    known = {i: dur_norm[i] for i in ids_norm if i in dur_norm}
    avg = (sum(known.values()) / len(known)) if known else 1.0
    pairs = [(i, known.get(i, avg)) for i in ids_norm]
    lane_total = sum(d for _, d in pairs)

    by_file = collections.Counter(_norm(i) for i in missing)
    per_file = collections.Counter()
    for i, d in pairs:
        per_file[_norm(i)] += d

    rec = {
        'collect_rc': rc,
        'collect_tail': tail,
        'selected': len(ids_norm),
        'durations_entries_total': len(dur),
        'selected_with_entry': len(known),
        'selected_missing_entry': len(missing),
        'coverage_pct': round(100.0 * len(known) / max(1, len(ids_norm)), 3),
        'avg_known_duration_s': round(avg, 4),
        'lane_total_s': round(lane_total, 1),
        'missing_by_file': by_file.most_common(30),
        'heaviest_files_s': [(f, round(s, 1)) for f, s in per_file.most_common(15)],
        'splits': {},
    }
    for splits in (3, 5, 8, 10):
        groups = least_duration_groups(pairs, splits)
        tot = sorted(round(sum(d for _, d in g), 1) for g in groups)
        ideal = lane_total / splits
        rec['splits'][str(splits)] = {
            'per_shard_s': tot,
            'max_s': max(tot),
            'ideal_s': round(ideal, 1),
            'imbalance_pct': round(100.0 * (max(tot) - ideal) / ideal, 2),
        }

    path = os.path.join(HERE, 't3_slow_lane_split_probe_slow.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=2)

    print('slow selection: %d ids, %d with a .test_durations entry (%.2f %%), '
          '%d missing' % (len(ids_norm), len(known), rec['coverage_pct'],
                          len(missing)))
    print('lane total (known + %.1f s average fill): %.1f s' % (avg, lane_total))
    for splits in (3, 5, 8, 10):
        s = rec['splits'][str(splits)]
        print('  splits=%-2d  ideal %7.1f s  max shard %7.1f s  '
              'imbalance %+6.2f %%  per-shard %s'
              % (splits, s['ideal_s'], s['max_s'], s['imbalance_pct'],
                 s['per_shard_s']))
    print('heaviest files (recorded seconds in the slow selection):')
    for f, s in rec['heaviest_files_s']:
        print('   %8.1f s  %s' % (s, f))
    if missing:
        print('MISSING .test_durations entries, by file:')
        for f, n in by_file.most_common(30):
            print('   %5d  %s' % (n, f))
    else:
        print('MISSING entries: none -- every slow-selected id has a duration.')
    print('json:', path)


if __name__ == '__main__':
    main()

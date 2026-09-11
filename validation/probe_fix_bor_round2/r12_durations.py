"""ROUND 2 -- re-splice ``.test_durations`` for the files this round touched,
and clear the three defects the verification found in it.

THE DEFECTS (verification D11):

1. a STALE key -- ``tests/unit/test_bor_solve.py::
   test_structured_stack_energy_floor_nodal``, the renamed gate's OLD node id,
   still present.  This is the stale-``.test_durations`` trap by name: it makes
   ``pytest-split`` budget time for a test that no longer exists and gives the
   renamed one the file average.
2. one entry **7.6x too large** -- ``test_mode_count_is_build_independent_
   under_an_infinitesimal_loss`` recorded at 167.41 s, which is the PRE-
   narrowing cost the file's own comment says was designed away.  It is the
   only entry in either new file over the 60 s shard cap, and it is fiction.
3. missing node ids, and (round 2) node ids whose NAME changed: the near-cutoff
   gate is now parametrized over ``m``, the EME band gate is renamed, and two
   files are new.

WHAT THIS DOES.  Reads a ``--durations=0`` report AND a ``--collect-only``
listing for the named files, REPLACES every key belonging to those files, and
writes ``.test_durations`` back sorted and JSON-validated.

WHY BOTH INPUTS.  pytest hides every duration below ``--durations-min``
(default 5 ms) unless verbosity reaches 2, and those hidden ids are exactly the
ones the verification found MISSING.  Rather than pay for a second 11-minute
run at higher verbosity, the collect-only listing supplies the node ids and any
id the report did not print is written at the 5 ms cutoff -- an UPPER bound on
its true cost, which is the safe direction for a shard balancer.  The count of
ids filled that way is printed, so the substitution is visible.

Usage:
    python -m pytest <files> -q -p no:randomly --durations=0 > dur.txt
    python -m pytest <files> -q -p no:randomly --collect-only > ids.txt
    python validation/probe_fix_bor_round2/r12_durations.py dur.txt ids.txt \\
        <file> ...
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DUR = os.path.join(ROOT, ".test_durations")

#: "12.34s call     tests/unit/test_x.py::test_y[p]"
LINE = re.compile(r"^\s*([0-9.]+)s\s+(call|setup|teardown)\s+(\S+)\s*$")
#: pytest's own cutoff for printing a duration at default verbosity.
HIDDEN = 0.005


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    report, ids_file = sys.argv[1], sys.argv[2]
    files = [f.replace("\\", "/") for f in sys.argv[3:]]

    def mine(nid):
        return any(nid.split("::")[0] == f for f in files)

    measured = {}
    with open(report, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = LINE.match(line)
            if not m:
                continue
            nid = m.group(3).replace("\\", "/")
            measured[nid] = measured.get(nid, 0.0) + float(m.group(1))
    if not measured:
        raise SystemExit("no durations parsed from %s" % (report,))

    collected = []
    with open(ids_file, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            nid = line.strip().replace("\\", "/")
            if "::" in nid and mine(nid):
                collected.append(nid)
    if not collected:
        raise SystemExit("no node ids parsed from %s" % (ids_file,))

    with open(DUR, encoding="utf-8") as fh:
        data = json.load(fh)
    before = len(data)

    dropped = [k for k in data if mine(k)]
    for k in dropped:
        del data[k]
    filled = 0
    for nid in collected:
        if nid in measured:
            data[nid] = round(measured[nid], 6)
        else:
            data[nid] = HIDDEN
            filled += 1
    stray = [n for n in measured if mine(n) and n not in collected]
    for nid in stray:                      # measured but not collected: keep
        data[nid] = round(measured[nid], 6)

    out = {k: data[k] for k in sorted(data)}
    txt = json.dumps(out, indent=1)
    json.loads(txt)                        # validate before writing
    with open(DUR, "w", encoding="utf-8", newline="") as fh:
        fh.write(txt)

    print("files re-spliced : %d" % (len(files),))
    for f in files:
        n = sum(1 for k in out if k.split("::")[0] == f)
        t = sum(out[k] for k in out if k.split("::")[0] == f)
        mx = max([out[k] for k in out if k.split("::")[0] == f] or [0.0])
        print("   %-56s %3d ids  %8.1f s  max %6.2f" % (f, n, t, mx))
    print("keys before      : %d" % (before,))
    print("keys dropped     : %d" % (len(dropped),))
    print("node ids written : %d (%d at the %.3fs hidden-duration cutoff)"
          % (len(collected) + len(stray), filled, HIDDEN))
    print("keys after       : %d" % (len(out),))
    print("sorted           : %s" % (list(out) == sorted(out),))
    over = {k: v for k, v in out.items() if v > 60.0 and mine(k)}
    print("over the 60 s shard cap in these files: %s" % (over or "none",))


if __name__ == "__main__":
    main()

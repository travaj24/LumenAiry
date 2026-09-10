"""Splice one test file's measured durations into ``.test_durations``.

pytest-split's ``--store-durations`` rewrites the whole file from whatever ran;
this merges instead, so the 12 500 entries the repository already carries are
untouched.  Usage:

    python -m pytest <file> -p no:randomly --durations=0 -vv > run.log
    python splice_durations.py run.log
"""
from __future__ import annotations

import json
import os
import re
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
_DUR = os.path.join(_ROOT, ".test_durations")
_LINE = re.compile(r"^([0-9.]+)s\s+(call|setup|teardown)\s+(\S+)\s*$")


def main(log):
    got = {}
    with open(log, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            m = _LINE.match(raw.strip())
            if not m:
                continue
            secs, phase, nid = float(m.group(1)), m.group(2), m.group(3)
            nid = nid.replace("\\", "/")
            got[nid] = got.get(nid, 0.0) + secs
            del phase
    if not got:
        raise SystemExit("no durations parsed from %s" % log)
    with open(_DUR, encoding="utf-8") as fh:
        cur = json.load(fh)
    before = len(cur)
    added = sum(1 for k in got if k not in cur)
    cur.update(got)
    with open(_DUR, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(cur, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("spliced %d ids (%d new); %d -> %d entries"
          % (len(got), added, before, len(cur)))
    for k in sorted(got, key=got.get, reverse=True)[:5]:
        print("   %8.2fs %s" % (got[k], k))


if __name__ == "__main__":
    main(sys.argv[1])

"""Splice NEW test node ids into ``.test_durations`` -- dict-union, sorted.

``pytest --durations=0`` prints one line per phase::

    12.34s call     tests/unit/test_x.py::test_y
     0.01s setup    tests/unit/test_x.py::test_y

This sums the phases per node id, ADDS only ids the file does not already have
(existing timings are another machine's and are left alone), and rewrites the
file sorted, with the same two-space JSON indentation.

    python splice_durations.py <pytest-log> [<node-id-prefix>]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOG = sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else ""
PATH = os.path.join(ROOT, ".test_durations")

RE = re.compile(r"^\s*([0-9.]+)s\s+(call|setup|teardown)\s+(\S+::\S+)\s*$")
totals = {}
with open(LOG, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        m = RE.match(line)
        if m and m.group(3).startswith(PREFIX):
            totals[m.group(3)] = totals.get(m.group(3), 0.0) + float(m.group(1))

with open(PATH, encoding="utf-8") as fh:
    cur = json.load(fh)

new = {k: v for k, v in totals.items() if k not in cur}
cur.update(new)
out = {k: cur[k] for k in sorted(cur)}
# the shipped file is CRLF; preserve it, or the splice becomes a 12000-line
# diff instead of a fourteen-line one
with open(PATH, "w", encoding="utf-8", newline="\r\n") as fh:
    json.dump(out, fh, indent=2)
    fh.write("\n")
print(f"parsed {len(totals)} node ids from {LOG}; ADDED {len(new)}; "
       f"file now {len(out)} entries, sorted={list(out) == sorted(out)}")
for k in sorted(new):
    print(f"  + {k}  {new[k]:.4f}")

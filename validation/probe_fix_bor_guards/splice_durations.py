"""Splice measured node-id durations into ``.test_durations``, sorted, and
validate the result is still JSON.

``.test_durations`` is what the CI's ``pytest-split`` shards on: a node id
missing from it is assigned the file's average and can land a 200-second test
in a 30-second shard.  Usage::

    python -m pytest <file> -q -p no:randomly --durations=0 > dur.txt
    python validation/probe_fix_bor_guards/splice_durations.py dur.txt

Reads the ``NN.NNs call tests/...::node`` lines, keeps the WORST reading per
node id (setup + call + teardown summed), writes ``.test_durations`` back
sorted by key, and re-loads it to prove it parses.
"""
from __future__ import annotations

import json
import os
import re
import sys

_LINE = re.compile(r"^\s*([0-9.]+)s\s+(call|setup|teardown)\s+(\S+)\s*$")


def main(paths):
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    target = os.path.join(root, ".test_durations")
    with open(target, encoding="utf-8") as fh:
        dur = json.load(fh)
    before = len(dur)
    seen = {}
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                mm = _LINE.match(line)
                if not mm:
                    continue
                secs, _kind, node = float(mm.group(1)), mm.group(2), mm.group(3)
                if "::" not in node:
                    continue
                seen[node] = seen.get(node, 0.0) + secs
    for node, secs in seen.items():
        dur[node] = max(secs, dur.get(node, 0.0))
    out = {k: dur[k] for k in sorted(dur)}
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(target, encoding="utf-8") as fh:
        back = json.load(fh)
    assert back == out, "round-trip mismatch"
    print("spliced %d node ids; %d -> %d entries; JSON validated"
          % (len(seen), before, len(out)))
    over = {k: v for k, v in seen.items() if v >= 60.0}
    if over:
        print("WARNING: node ids at or over the 60 s cap:")
        for k, v in sorted(over.items(), key=lambda kv: -kv[1]):
            print("   %8.2fs  %s" % (v, k))


if __name__ == "__main__":
    main(sys.argv[1:])

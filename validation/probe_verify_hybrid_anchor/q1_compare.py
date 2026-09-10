"""Q1 compare -- diff the two builds' hash tables and CLASSIFY every move."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# the ONLY moves the fix's claim permits: the transmission accessors on a
# fixture whose stack contains a slanted PATTERNED (non-constant-tile) layer.
SLANTED_PATTERNED = {
    "slanted_patterned_quarter_normal",
    "slanted_patterned_quarter_oblique25",
    "slanted_patterned_quarter_conical25_40",
    "slanted_patterned_half_normal",
    "slanted_tensor_inplane_oblique25",
    "two_slanted_layers_conical",
    "slanted_over_film_oblique25",
    "slanted_patterned_xy_conical",
}
TRANSMISSION_KEYS = {"jones_transmission", "per_order_transmission"}


def main():
    build = sys.argv[1] if len(sys.argv) > 1 else "win"
    a = json.load(open(os.path.join(
        HERE, "results", "q1_hashes.v5440.%s.json" % build),
        encoding="cp1252"))["fixtures"]
    b = json.load(open(os.path.join(
        HERE, "results", "q1_hashes.fix.%s.json" % build),
        encoding="cp1252"))["fixtures"]
    print("BUILD:", build)
    same, moved, unexpected, missing_move = 0, [], [], []
    for name in sorted(a):
        for k in sorted(a[name]):
            if k.startswith("_"):
                continue
            va, vb = a[name][k], b[name].get(k)
            if va == vb:
                same += 1
                if name in SLANTED_PATTERNED and k in TRANSMISSION_KEYS:
                    missing_move.append((name, k, va))
            else:
                moved.append((name, k, va, vb))
                if not (name in SLANTED_PATTERNED and k in TRANSMISSION_KEYS):
                    unexpected.append((name, k, va, vb))
    total = same + len(moved)
    print("hashes %d: identical %d, moved %d, UNEXPECTED %d"
          % (total, same, len(moved), len(unexpected)))
    print("\nMOVED (all should be transmission accessors on a slanted "
          "PATTERNED fixture):")
    for n, k, va, vb in moved:
        tag = "  " if (n in SLANTED_PATTERNED and k in TRANSMISSION_KEYS) \
            else "!!"
        print("  %s %-42s %-22s %s -> %s" % (tag, n, k, va, vb))
    print("\nIDENTICAL on a slanted PATTERNED fixture's transmission keys "
          "(the exemptions):")
    for n, k, va in missing_move:
        print("     %-42s %-22s %s" % (n, k, va))
    payload = dict(total=total, identical=same, moved=len(moved),
                   unexpected=len(unexpected),
                   moved_rows=moved, exempt_rows=missing_move)
    with open(os.path.join(HERE, "results",
                           "q1_compare.%s.json" % build), "w",
              encoding="cp1252", errors="replace") as fh:
        json.dump(payload, fh, indent=1)
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())

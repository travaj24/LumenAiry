"""ROUND 2 -- diff two ``r10_bit_identity`` JSONs and classify every mover.

The contract round 2 must keep is: **only the rows the NEW screen REFUSES may
change.**  A fixture whose hash moved for any other reason is a regression, and
a fixture that now raises where it did not is a decision that has to be named.

Usage:  python validation/probe_fix_bor_round2/r11_identity_diff.py \
            <pre.json> <post.json>
"""
from __future__ import annotations

import json
import os
import sys


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def compare(pre, post, group):
    a, b = pre[group], post[group]
    names = sorted(set(a) | set(b))
    same, moved, newly_raised, no_longer_raised, missing = [], [], [], [], []
    for n in names:
        if n not in a or n not in b:
            missing.append(n)
            continue
        ra, rb = a[n].get("raised"), b[n].get("raised")
        if ra != rb:
            (newly_raised if rb else no_longer_raised).append(
                (n, ra, rb, (b[n] if rb else a[n]).get("msg", "")[:200]))
            continue
        if ra:                                   # both raised the same thing
            same.append(n)
            continue
        ha, hb = a[n].get("hash"), b[n].get("hash")
        (same if ha == hb else moved).append(n if ha == hb else (n, ha, hb))
    return same, moved, newly_raised, no_longer_raised, missing


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    pre, post = load(sys.argv[1]), load(sys.argv[2])
    print("PRE  tree %s  arm %s" % (pre["tree"], pre["arm"]))
    print("POST tree %s  arm %s" % (post["tree"], post["arm"]))
    out = {}
    for group in ("bor", "eme"):
        same, moved, newr, oldr, missing = compare(pre, post, group)
        print("\n=== %s ===" % (group.upper(),))
        print("  identical      : %d" % (len(same),))
        print("  MOVED (hash)   : %d" % (len(moved),))
        for n, ha, hb in moved:
            print("     %-32s %s -> %s" % (n, ha[:16], hb[:16]))
        print("  newly REFUSED  : %d" % (len(newr),))
        for n, ra, rb, msg in newr:
            print("     %-32s %s -> %s" % (n, ra, rb))
            print("        %s" % (msg[:170],))
        print("  no longer raises: %d" % (len(oldr),))
        for n, ra, rb, _m in oldr:
            print("     %-32s %s -> %s" % (n, ra, rb))
        if missing:
            print("  present in only one run: %s" % (missing,))
        out[group] = dict(identical=len(same), moved=[m[0] for m in moved],
                          newly_raised=[(n, ra, rb) for n, ra, rb, _ in newr],
                          no_longer_raised=[(n, ra, rb)
                                            for n, ra, rb, _ in oldr],
                          missing=missing)
    here = os.path.dirname(os.path.abspath(__file__))
    name = "r11_identity_diff_%s_%s_t%s.json" % (
        post["arm"].get("platform"), post["arm"].get("loaded_kernel"),
        post["arm"].get("blas_threads"))
    with open(os.path.join(here, name), "w", encoding="utf-8") as fh:
        json.dump(dict(pre=pre["arm"], post=post["arm"], groups=out), fh,
                  indent=1, sort_keys=True, default=str)
    print("\n[probe] wrote %s" % (os.path.join(here, name),))


if __name__ == "__main__":
    main()

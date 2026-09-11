"""The BASE vs ROUND-3 identity diff.

Round 3's contract is the same as round 2's: every fixture whose hash MOVED
must have moved from a HASH to a ``RAISED:BORNodalPassivityError``.  A moved
hash, or an exception of any other class, is a violation.
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("post")
    ap.add_argument("--tag", default="local")
    a = ap.parse_args()
    P = json.load(open(a.base))["hashes"]
    Q = json.load(open(a.post))["hashes"]
    same, moved, legal, illegal = [], [], [], []
    for k in sorted(set(P) & set(Q)):
        if P[k] == Q[k]:
            same.append(k)
            continue
        moved.append(k)
        if (not P[k].startswith("RAISED")
                and Q[k].startswith("RAISED:BORNodalPassivityError")):
            legal.append(k)
        else:
            illegal.append(k)

    def grp(kk, g):
        return [k for k in kk if k.startswith(g)]

    common = sorted(set(P) & set(Q))
    print("fixtures: %d common (BOR %d / EME %d); base-only %d, post-only %d"
          % (len(common), len(grp(common, "BOR")), len(grp(common, "EME")),
             len(set(P) - set(Q)), len(set(Q) - set(P))))
    print("IDENTICAL : %3d  (BOR %3d / EME %3d)"
          % (len(same), len(grp(same, "BOR")), len(grp(same, "EME"))))
    print("MOVED     : %3d  (BOR %3d / EME %3d)"
          % (len(moved), len(grp(moved, "BOR")), len(grp(moved, "EME"))))
    print("  of which NEWLY REFUSED: %d" % (len(legal),))
    for k in legal:
        print("     %-44s -> %s" % (k, Q[k][:80]))
    print("  ILLEGAL moves (a changed ANSWER, not a refusal): %d"
          % (len(illegal),))
    for k in illegal:
        print("     %-44s  base=%s  post=%s" % (k, P[k][:24], Q[k][:70]))
    out = dict(n_common=len(common), identical=len(same), moved=len(moved),
               identical_bor=len(grp(same, "BOR")),
               identical_eme=len(grp(same, "EME")),
               newly_refused=legal, illegal=illegal, moved_names=moved)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "g6_identity_diff_%s.json" % (a.tag,))
    with open(p, "w", encoding="cp1252", errors="replace") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("WROTE", p)


if __name__ == "__main__":
    main()

"""V8 -- the PRE vs POST identity diff.

Round 2's contract is that ONLY the rows its new screen REFUSES may change.
This scores that directly: every fixture whose hash moved must have moved from
a HASH to a ``RAISED:BORNodalPassivityError``.  Anything else -- a moved hash,
or a new exception of another class -- is a violation.
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pre")
    ap.add_argument("post")
    ap.add_argument("--tag", default="local")
    a = ap.parse_args()
    pre = json.load(open(a.pre))
    post = json.load(open(a.post))
    P, Q = pre["hashes"], post["hashes"]
    only_pre = sorted(set(P) - set(Q))
    only_post = sorted(set(Q) - set(P))
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
    grp = lambda kk, g: [k for k in kk if k.startswith(g)]       # noqa: E731
    print("fixtures: %d common (%d BOR, %d EME); pre-only %d, post-only %d"
          % (len(same) + len(moved), len(grp(set(P) & set(Q), "BOR")),
             len(grp(set(P) & set(Q), "EME")), len(only_pre), len(only_post)))
    print("IDENTICAL : %3d  (BOR %3d / EME %3d)"
          % (len(same), len(grp(same, "BOR")), len(grp(same, "EME"))))
    print("MOVED     : %3d  (BOR %3d / EME %3d)"
          % (len(moved), len(grp(moved, "BOR")), len(grp(moved, "EME"))))
    print("  of which NEWLY REFUSED by the round-2 screen: %d" % (len(legal),))
    for k in legal:
        print("     %-44s -> %s" % (k, Q[k][:70]))
    print("  ILLEGAL moves (a changed ANSWER, not a refusal): %d"
          % (len(illegal),))
    for k in illegal:
        print("     %-44s  pre=%s  post=%s" % (k, P[k][:24], Q[k][:60]))
    out = dict(pre_prov=pre["provenance"], post_prov=post["provenance"],
               n_common=len(same) + len(moved), identical=len(same),
               moved=len(moved), newly_refused=legal, illegal=illegal,
               identical_bor=len(grp(same, "BOR")),
               identical_eme=len(grp(same, "EME")),
               moved_names=moved)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "v8_identity_diff_%s.json" % (a.tag,))
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()

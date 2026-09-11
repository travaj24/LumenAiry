"""GAP-wide BIT IDENTITY -- round 3 must move NOTHING except the rows its
widened screen refuses.

THE BATTERY IS NOT MINE.  It is the round-2 VERIFICATION's own -- 46 BOR
fixtures (22 of them LEGACY NODAL, the only path the passivity screen can
touch) and 19 EME fixtures, each hashed to the SHA-256 of the exact IEEE-754
bytes of its answer, a fixture that raises recorded as its exception class.
Re-using it rather than writing a third one is deliberate: it is the battery
the round-2 ship recommendation was scored on, so "nothing moved that round 3
does not refuse" is a statement about the SAME rows.

``--tag`` names the arm; the JSON lands beside this file.  Diff two of them
with ``g6_identity_diff.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(HERE), "probe_verify_bor_round2")
sys.path.insert(0, V2)
sys.path.insert(0, HERE)

import _g3  # noqa: E402
import v7_identity as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    arm = _g3.arm()
    tag = a.tag or _g3.tag(arm)
    res = {}
    for group, fx in (("BOR", V.bor_fixtures()), ("EME", V.eme_fixtures())):
        for name, (fn, args) in fx.items():
            key = "%s:%s" % (group, name)
            try:
                res[key] = fn(*args)
            except Exception as exc:                           # noqa: BLE001
                res[key] = "RAISED:%s:%s" % (type(exc).__name__,
                                             str(exc)[:80])
            print("%-46s %s" % (key, res[key][:40]), flush=True)
    out = dict(arm=arm, n_bor=len(V.bor_fixtures()),
               n_eme=len(V.eme_fixtures()), hashes=res)
    p = os.path.join(HERE, "g6_identity_%s.json" % (tag,))
    with open(p, "w", encoding="cp1252", errors="replace") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("WROTE", p, "n =", len(res))


if __name__ == "__main__":
    main()

"""Re-measure ONLY the Berreman rows of a finished ``o2_census`` JSON.

The first census run reduced ``berreman_jones_1d``'s return with
``|o[0]|^2 + |o[1]|^2``, but that function returns ``(R, T, jones_r, jones_t)``
with ``R`` and ``T`` ALREADY flux-normalized -- so the reduction squared a power
and produced a number that is not a closure, which mis-binned those six rows on
one build.  ``o2_census.berreman`` is corrected; this re-runs just those rows on
the same tree and rewrites their entries in place, leaving every other row (and
every ``T22`` reading, which the reduction never touched) exactly as measured.

    PYTHONPATH=<tree> python o2_patch_berreman.py _prefix
"""
from __future__ import annotations

import json
import os
import sys

import _lib as L
import o2_census as C

_R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main(suffix):
    a = L.arm()
    fn = os.path.join(_R, "o2_census%s.%s.json" % (suffix, a["tag"]))
    with open(fn, encoding="cp1252", errors="replace") as fh:
        doc = json.load(fh)
    cases = C.build_cases()
    n = 0
    for name in sorted(cases):
        if not name.startswith("berreman"):
            continue
        rec = C.scored(cases[name])
        old = doc.get(name, {}).get("RT")
        doc[name] = rec
        n += 1
        print("%-26s RT %s -> %.9g   min rcond %.4g" % (
            name, old, rec.get("RT", float("nan")),
            rec["T22"].get("min_rcond_eq", float("nan"))))
    doc["_berreman_repatched"] = dict(tree=a["tree"], n=n)
    with open(fn, "w", encoding="cp1252", errors="replace") as fh:
        json.dump(doc, fh, indent=1, default=str)
    print("[patched] %s (%d rows)" % (fn, n))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")

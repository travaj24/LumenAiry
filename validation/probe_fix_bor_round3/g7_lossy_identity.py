"""GAP 2's OWN bit-identity battery -- the population the verifier's has not.

WHY THIS EXISTS.  ``g6_identity.py`` re-runs the round-2 verification's 65
fixtures, and they come back 65 of 65 IDENTICAL on both builds.  That is an
honest result and a WEAK one for round 3, because
``v7_identity._bor_nodal_fixture`` puts its ``im_rel`` on the MIDDLE layer
only: both half-spaces are ``F.uniform(2.0)``, exactly lossless, on every one
of its 22 legacy-nodal rows.  So the battery contains NO row of the population
round 3 changed, and "nothing moved" there says nothing about GAP 2.

THIS battery is that population: a loss on the INCIDENCE half-space alone and
on BOTH half-spaces, over four decades, on three profile families.  The
contract is the same -- a fixture whose hash MOVED must have moved from a HASH
to a ``RAISED:BORNodalPassivityError``, never to a different answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _g3  # noqa: E402

K0 = 2.0


def _hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def fixtures():
    out = {}
    for family in ("uniform", "ring", "seg"):
        for m in (0, 1, 2):
            for rbl in (0.5, 2.0):
                for where in ("inc", "both"):
                    for im in (1e-12, 1e-9, 1e-6, 1e-3):
                        out["nodal_%s_m%d_rbl%g_%s_im%g"
                            % (family, m, rbl, where, im)] = (
                                family, m, 200, rbl, where, im)
    return out


def run_one(family, m, N, rbl, where, im):
    import warnings

    import lumenairy.elements.bor.bor_solve as bs
    layers = _g3.stack("nodal", family, m, N, rbl, im, where, K0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = bs.solve(layers, K0)
    return _hash(np.asarray(r["R"], float), np.asarray(r["T"], float),
                 np.asarray(r["energy"], float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    arm = _g3.arm()
    tag = a.tag or _g3.tag(arm)
    res = {}
    for name, args in fixtures().items():
        try:
            res[name] = run_one(*args)
        except Exception as exc:                               # noqa: BLE001
            res[name] = "RAISED:%s:%s" % (type(exc).__name__, str(exc)[:80])
    out = dict(arm=arm, n=len(res), hashes=res)
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "g7_lossy_identity_%s.json" % (tag,))
    with open(p, "w", encoding="cp1252", errors="replace") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    n_raised = sum(v.startswith("RAISED") for v in res.values())
    print("ARM", arm["build"], arm["loaded_kernel"], "t%s" % arm["threads"],
          arm["lumenairy_file"])
    print("fixtures %d, refused %d" % (len(res), n_raised))
    print("WROTE", p)


if __name__ == "__main__":
    main()

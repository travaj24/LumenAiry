"""GAP 2 -- DOES THE INDEX CEILING HOLD ON A LOSSY HALF-SPACE?

The round-2 verification could not answer this ("Whether the index ceiling
would hold on a LOSSY half-space ... I did not construct a case that would
decide it"), and GAP 2's remedy assumes it does.  This measures it.

THE QUESTION.  ``_channel_index_excess`` refuses when a RETURNED channel's
axial index ``Re qn`` exceeds ``Re sqrt(eps_ceiling)`` of its own half-space.
On a LOSSLESS half-space that is a theorem (``eps k0^2 + D`` is real symmetric,
Rayleigh bounds ``q^2 <= max(eps) k0^2``).  With a complex ``eps`` the operator
is complex symmetric and the same statement about ``Re q`` is only approximate,
which is why round 2 gates the conjunct on ``_layer_is_lossless``.  The gate is
what makes GAP 2 possible, so the question is quantitative: HOW approximate?

SCORING.  A row is SET-RIGHT when the nodal cascade returns the same
(incidence, exit) channel counts as its div-conforming staggered twin on the
identical geometry, SET-WRONG otherwise.  The ceiling is sound on lossy media
iff set-right rows stay BELOW zero and set-wrong rows reach above the slack.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _g3  # noqa: E402

import lumenairy.elements.bor.bor_solve as bs  # noqa: E402

K0 = 2.0
FAMILIES = ("uniform", "ring")
MS = (0, 1, 2, 3)
NS = (120, 200)
RBLS = (0.5, 1.0, 2.0, 4.0)
WHERES = ("inc", "both")
IMS = (1e-12, 1e-9, 1e-6, 1e-3, 1e-1)


def _row(family, m, N, rbl, where, im):
    out = dict(family=family, m=m, N=N, rbl=rbl, where=where, im=im)
    try:
        lay_n = _g3.stack("nodal", family, m, N, rbl, im, where, K0)
        lay_s = _g3.stack("staggered", family, m, N, rbl, im, where, K0)
        rn = _g3.disarmed(lay_n, K0)
        rs = _g3.disarmed(lay_s, K0)
    except Exception as exc:                                   # pragma: no cover
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        return out
    cn = (int(np.size(rn["inc"])), int(np.size(rn["out"])))
    cs = (int(np.size(rs["inc"])), int(np.size(rs["out"])))
    out["count_nodal"], out["count_stag"] = list(cn), list(cs)
    out["set_right"] = bool(cn == cs)
    for tag_, lay, res in (("nodal", lay_n, rn), ("stag", lay_s, rs)):
        a, b = _g3.ceiling_excess(lay, res, K0)
        out["exc_inc_" + tag_] = a
        out["exc_exit_" + tag_] = b
        e = np.asarray(res["energy"], float)
        out["emax_" + tag_] = float(np.max(e)) if e.size else None
        out["emin_" + tag_] = float(np.min(e)) if e.size else None
    out["n_channels"] = int(np.size(rn["R"]))
    return out


def main():
    a = _g3.arm()
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"])
    t0 = time.time()
    rows = []
    for family in FAMILIES:
        for m in MS:
            for N in NS:
                for rbl in RBLS:
                    for where in WHERES:
                        for im in IMS:
                            rows.append(_row(family, m, N, rbl, where, im))
    ok = [r for r in rows if "error" not in r and r["n_channels"]]
    right = [r for r in ok if r["set_right"]]
    wrong = [r for r in ok if not r["set_right"]]

    def env(pop, key):
        v = [r[key] for r in pop if r.get(key) is not None]
        return (max(v) if v else None), (min(v) if v else None), len(v)

    summary = {}
    for name, pop in (("set_right", right), ("set_wrong", wrong)):
        for key in ("exc_inc_nodal", "exc_exit_nodal",
                    "exc_inc_stag", "exc_exit_stag"):
            hi, lo, n = env(pop, key)
            summary["%s.%s" % (name, key)] = dict(max=hi, min=lo, n=n)
    slack = float(bs._BOR_INDEX_CEILING_SLACK)
    # the decision the fix would take: refuse iff EITHER side's excess is above
    # the slack, evaluated with the lossless gate REMOVED
    def fires(r):
        return any((r.get(k) is not None and r[k] > slack)
                   for k in ("exc_inc_nodal", "exc_exit_nodal"))

    def fires_stag(r):
        return any((r.get(k) is not None and r[k] > slack)
                   for k in ("exc_inc_stag", "exc_exit_stag"))

    summary["slack"] = slack
    summary["n_rows"] = len(rows)
    summary["n_usable"] = len(ok)
    summary["n_set_right"] = len(right)
    summary["n_set_wrong"] = len(wrong)
    summary["false_positives_nodal_set_right"] = sum(fires(r) for r in right)
    summary["false_positives_staggered"] = sum(fires_stag(r) for r in ok)
    summary["fires_on_set_wrong"] = sum(fires(r) for r in wrong)
    summary["seconds"] = time.time() - t0
    for k in sorted(summary):
        print(" ", k, summary[k])
    _g3.dump("g2_lossy_ceiling", dict(rows=rows, summary=summary), a)


if __name__ == "__main__":
    main()

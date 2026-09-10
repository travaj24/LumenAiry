"""V9 -- the audit's headline numbers, on the audit's own failing fixture.

Everything here is re-measured, not read: the per-interface ``cond(a+b)``
(superstrate->layer vs layer->substrate), the count of singular values below
``1e-10 s_max``, the inverse participation ratio of the smallest right singular
vector, and the offending eigenvalue itself.  Both arms in one process.

Usage:  OPENBLAS_NUM_THREADS=<n> python v9_headline.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_fixtures as V  # noqa: E402


def _solve(symmetry):
    from lumenairy.elements.rcwa import rcwa_jones_2d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rcwa_jones_2d(V._P, V._P, V.uniaxial_cell(), 1.5, 1.0,
                             V._DEPTH, V._WL, n_orders_x=5, n_orders_y=5,
                             symmetry=symmetry)


def one(symmetry):
    eig, ifc = V.EigSpy(), V.InterfaceSpy()
    with eig, ifc:
        _solve(symmetry)
    rows = [r for r in ifc.rows if "cond" in r]
    lam2 = np.concatenate([np.asarray(w) for w in eig.seen])
    r = V.principal_root(lam2)
    ratio, scale = V.band_ratio(r)
    bad = (ratio <= 1e-8) & (r.imag < 0)
    offenders = [dict(idx=int(k), lam2_re=float(lam2[k].real),
                      lam2_im=float(lam2[k].imag),
                      lam_re=float(r[k].real), lam_im=float(r[k].imag),
                      ratio=float(ratio[k]))
                 for k in np.where(bad)[0]]
    offenders.sort(key=lambda d: d["lam2_re"])
    return dict(
        n_interfaces=len(rows),
        cond_first=rows[0]["cond"] if rows else None,     # superstrate->layer
        cond_last=rows[-1]["cond"] if rows else None,     # layer->substrate
        cond_all=[r_["cond"] for r_ in rows],
        n_tiny_sv=[r_["n_tiny"] for r_ in rows],
        ipr=[r_["ipr"] for r_ in rows],
        modes=int(lam2.size),
        n_on_cut=int(np.sum(ratio <= 1e-8)),
        n_incoming_principal=int(bad.sum()),
        offenders=offenders[:6],
        scale=float(scale))


def main():
    V.require_local_tree()
    out = sys.argv[1]
    res = {}
    for sym in (False, True):
        res["post_sym%s" % sym] = one(sym)
        with V.PreSqrtDecay():
            res["pre_sym%s" % sym] = one(sym)
    for k in sorted(res):
        d = res[k]
        print("%-12s interfaces=%d cond=%s tinySV=%s ipr=%s" % (
            k, d["n_interfaces"],
            " ".join("%.3e" % c for c in d["cond_all"]),
            d["n_tiny_sv"], " ".join("%.2f" % x for x in d["ipr"])))
        print("             modes=%d onCut=%d incomingPrincipal=%d scale=%.4f"
              % (d["modes"], d["n_on_cut"], d["n_incoming_principal"],
                 d["scale"]))
        for o in d["offenders"]:
            print("             lam2 = %.15f %+.4ej  ->  lam = %+.4e %+.15fj"
                  % (o["lam2_re"], o["lam2_im"], o["lam_re"], o["lam_im"]))
    res["openblas_num_threads"] = os.environ.get("OPENBLAS_NUM_THREADS",
                                                 "unpinned")
    V.dump(out, res)


if __name__ == "__main__":
    main()

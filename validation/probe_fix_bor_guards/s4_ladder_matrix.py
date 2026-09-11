"""STEP 4, the DECISION TABLE: the delta ladder's verdict per rung, under one
(kernel, thread-count, build) arm.

The claim this settles is the one the 5.45.0 CI finding made necessary -- that
the contract's verdicts do NOT move with the arithmetic.  It can only hold
because NEITHER conjunct is the energy violation: on this same ladder the
closure's spread across three OpenBLAS kernels is 70.90x, straddling the 1-D
``_SLIVER_TRIGGER_BAR`` of 1e-3, while the geometric conjunct is kernel-EXACT
and the spectral one kernel-stable.

Each row prints BOTH: the verdict (which must not move) and the closure (which
does, and is recorded so the difference is visible rather than asserted).

Usage: ``python s4_ladder_matrix.py <tag>``; re-run under each kernel and
thread count on each build, then diff the ``verdicts`` lists.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

RBIG = 24.0
LADDER = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
DEGREES = (6, 8, 12)


def one(delta_frac, degree):
    from lumenairy.elements.bor import BORStack
    from lumenairy.elements.bor import _sem_contract as SC
    d = delta_frac * RBIG
    prev = SC.BOR_SEM_MESH_GUARD
    SC.BOR_SEM_MESH_GUARD = False
    try:
        st = BORStack(RBIG, 1, basis="sem", degree=degree, N=200,
                      n_superstrate=1.0, n_substrate=1.5)
        st.set_source(k0=2.0)
        st.add_layer(0.5, segments=[(6.0, 6.0), (RBIG, 2.0)])
        st.add_layer(0.5, segments=[(6.0 + d, 2.0), (RBIG, 6.0)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = st.solve()
        recs = st._sem_mesh_report
        v = sorted({SC.verdict(r) for r in recs})
        verdict = ("refuse" if "refuse" in v
                   else "warn" if any(x.startswith("warn") for x in v)
                   else "ok")
        e = np.asarray(res["energy"])
        return dict(
            delta_frac=delta_frac, degree=degree, verdict=verdict,
            w_min_union_frac=min(r["w_min_union_frac"] for r in recs),
            q_excess=max(r["q_excess"] for r in recs
                         if np.isfinite(r["q_excess"])),
            closure=float(np.max(np.abs(e - 1.0))) if e.size else None,
            n_orders=int(np.size(res["R"])))
    except Exception as exc:                           # noqa: BLE001
        return dict(delta_frac=delta_frac, degree=degree,
                    error="%s: %s" % (type(exc).__name__, exc))
    finally:
        SC.BOR_SEM_MESH_GUARD = prev


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    print("TREE", C.pin_tree())
    print("KERNEL", C.kernel_tag())
    rows = [one(dl, deg) for deg in DEGREES for dl in LADDER]
    table = {}
    for r in rows:
        if "verdict" in r:
            table.setdefault(str(r["degree"]), []).append(r["verdict"])
    print("VERDICTS", table)
    cl = [r["closure"] for r in rows
          if r.get("closure") is not None]
    print("CLOSURE range %.4e .. %.4e (RECORDED, NOT DECIDED ON)"
          % (min(cl), max(cl)))
    geo = [r["w_min_union_frac"] for r in rows if "w_min_union_frac" in r]
    print("GEOMETRIC conjunct %.6e .. %.6e" % (min(geo), max(geo)))
    C.dump("s4_ladder_%s.json" % (tag,),
           dict(rows=rows, verdicts=table,
                closure_min=min(cl), closure_max=max(cl)))


if __name__ == "__main__":
    main()

"""GAPS 3 and 4 -- the SEM mesh contract's verdict over a WIDTH LADDER at all
three domain positions, on every arm.

GAP 3: ``verdict`` reads a NON-FINITE ``q_excess`` as not hot, so the most
damaged rung on the ladder is the one that reads ``ok`` -- and because ``inf``
is a backward-error outcome the verdict moves with the BLAS kernel.

GAP 4: ``_BOR_MIN_ELEM_FRAC`` is a STRICT comparison at the representation
limit, so the same physical width decides one way at the axis (``1.0e-06``
exactly, from walls ``0`` and ``w``) and the other at the outer wall
(``9.999999999917e-07``, from ``Rbig - w`` and ``Rbig``).

Both are read from ONE ladder so the two effects can be separated: GAP 3 is a
row where ``q_excess`` is non-finite, GAP 4 is a row where the three positions'
``w_min_own_frac`` differ by round-off across the edge.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _g3  # noqa: E402

from lumenairy.elements.bor import _sem_contract as sc  # noqa: E402
from lumenairy.elements.bor.bor_stack import BORStack  # noqa: E402

RBIG, N, DEGREE, K0 = 24.0, 120, 8, 2.0
POSITIONS = ("axis", "middle", "outer")

#: the ladder GAP 3 and GAP 4 are asked for, 1e-9 .. 1e-5 of Rbig, PLUS the
#: edge itself and the two representations of it that GAP 4 is about.
EDGE = float(sc._BOR_MIN_ELEM_FRAC)
LADDER = (1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, EDGE, 3e-6, 1e-5)


def segments(position, w):
    return {"outer": [(RBIG - w, 2.0), (RBIG, 6.0)],
            "axis": [(w, 6.0), (RBIG, 2.0)],
            "middle": [(6.0, 6.0), (6.0 + w, 2.0), (RBIG, 2.0)]}[position]


def record(position, w_frac):
    st = BORStack(Rbig=RBIG, m=1, N=N, n_superstrate=1.0, n_substrate=1.5,
                  basis="sem", degree=DEGREE)
    st.add_layer(0.5, segments=segments(position, w_frac * RBIG))
    st.set_source(k0=K0)
    prev = sc.BOR_SEM_MESH_GUARD
    sc.BOR_SEM_MESH_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.solve()
        recs = list(getattr(st, "_sem_mesh_report", []) or [])
    finally:
        sc.BOR_SEM_MESH_GUARD = prev
    assert recs, "%s liner at %g produced no mesh report" % (position, w_frac)
    r = recs[0]
    return dict(position=position, w_frac=w_frac,
                w_min_own_frac=float(r["w_min_own_frac"]),
                w_min_union_frac=float(r["w_min_union_frac"]),
                q_excess=float(r["q_excess"]),
                q_finite=bool(np.isfinite(r["q_excess"])),
                q_measurable=bool(r.get("q_measurable", True)),
                verdict=sc.verdict(r))


def main():
    a = _g3.arm()
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"])
    t0 = time.time()
    rows = [record(p, w) for w in LADDER for p in POSITIONS]
    by_w = {}
    for r in rows:
        by_w.setdefault(r["w_frac"], {})[r["position"]] = r
    disagree = sorted(w for w, d in by_w.items()
                      if len({x["verdict"] for x in d.values()}) > 1)
    nonfinite = [(r["position"], r["w_frac"], r["verdict"])
                 for r in rows if not r["q_finite"]]
    nonfinite_ok = [x for x in nonfinite if x[2] == "ok"]
    edge = {p: by_w[EDGE][p]["w_min_own_frac"] for p in POSITIONS}
    summary = dict(
        rungs=len(LADDER),
        rows=len(rows),
        positions_disagree_at=disagree,
        n_disagreeing_rungs=len(disagree),
        nonfinite_q_excess=nonfinite,
        nonfinite_reading_ok=nonfinite_ok,
        edge_w_min_own_frac=edge,
        edge_verdicts={p: by_w[EDGE][p]["verdict"] for p in POSITIONS},
        edge_rel_spread=(max(edge.values()) - min(edge.values())) / EDGE,
        edge_ulps=((max(edge.values()) - min(edge.values()))
                   / (np.spacing(EDGE) or 1.0)),
        seconds=time.time() - t0,
    )
    for k in sorted(summary):
        print(" ", k, summary[k])
    _g3.dump("g3_sem_ladder", dict(rows=rows, summary=summary), a)


if __name__ == "__main__":
    main()

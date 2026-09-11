"""D-V3 -- GAP 3's change reaches the SEM contract's ``refuse`` arm, so it
converts a RETURNED answer into a RAISE on one population.

``verdict``'s first arm is ``_below(w_min_union_frac, _BOR_MIN_ELEM_FRAC) and
hot``.  Round 3 changed ``hot``, so the arm moves wherever a MANUFACTURED union
cell -- one neither layer's own segment list asked for -- is narrower than the
bar AND the ratio came back non-finite.  With ``BOR_SEM_MESH_GUARD`` at its
default ``True`` a ``refuse`` is a ``BORSemMeshError``, so on that population a
caller who previously received a number now receives an exception.

The round's own text rates GAP 3's cost as "a missing ``UserWarning``, never a
wrong returned number".  This probe measures the other half.

Run against BOTH trees: ``LUM_PROBE_TAG=BASE PYTHONPATH=<1ac6de7e tree>`` and
untagged against ``fix/bor-guards-round3``.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vb3  # noqa: E402

from lumenairy.elements.bor import _sem_contract as sc  # noqa: E402
from lumenairy.elements.bor.bor_stack import BORStack  # noqa: E402

RBIG, M, N, DEG, K0 = 17.0, 2, 96, 6, 3.5
EPS_HI, EPS_LO = 5.0, 2.2

#: (layer-0 axis wall, layer-1 axis wall) as fractions of Rbig.  The union mesh
#: manufactures a cell of width ``b - a`` that NEITHER layer's own segment list
#: asked for.
PAIRS = ((1e-9, 2e-9), (2e-9, 3e-9), (1e-8, 1.1e-8), (3e-8, 3.3e-8),
         (1e-7, 1.1e-7), (1e-6, 1.1e-6), (1e-5, 1.1e-5))


def _stack(a_frac, b_frac):
    st = BORStack(Rbig=RBIG, m=M, N=N, n_superstrate=1.2, n_substrate=1.7,
                  basis="sem", degree=DEG)
    st.add_layer(0.6, segments=[(a_frac * RBIG, EPS_HI), (RBIG, EPS_LO)])
    st.add_layer(0.6, segments=[(b_frac * RBIG, EPS_HI), (RBIG, EPS_LO)])
    st.set_source(k0=K0)
    return st


def row(a_frac, b_frac):
    #: what a caller sees, guard at its DEFAULT
    st = _stack(a_frac, b_frac)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.solve()
        seen = "returned"
    except Exception as exc:                          # noqa: BLE001 recorded
        seen = type(exc).__name__
    #: and the record behind it, guard OFF so the measurement always happens
    st2 = _stack(a_frac, b_frac)
    prev = sc.BOR_SEM_MESH_GUARD
    sc.BOR_SEM_MESH_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st2.solve()
        recs = list(getattr(st2, "_sem_mesh_report", []) or [])
    finally:
        sc.BOR_SEM_MESH_GUARD = prev
    r = recs[0] if recs else {}
    return dict(a=a_frac, b=b_frac, caller_sees=seen,
                verdicts=[sc.verdict(x) for x in recs],
                w_min_union_frac=float(r.get("w_min_union_frac", float("nan"))),
                q_excess=float(r.get("q_excess", float("nan"))),
                q_finite=bool(np.isfinite(r.get("q_excess", float("nan")))),
                q_measurable=bool(r.get("q_measurable", True)))


def main():
    a = _vb3.arm()
    _vb3.require_tree(a)
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"], flush=True)
    rows = [row(x, y) for x, y in PAIRS]
    for r in rows:
        print("  a=%-9.3g b=%-9.3g fu=%-10.4g q=%-12.6g finite=%-5s -> %-18s "
              "%s" % (r["a"], r["b"], r["w_min_union_frac"], r["q_excess"],
                      r["q_finite"], r["caller_sees"], r["verdicts"]),
              flush=True)
    nonfinite = [r for r in rows if not r["q_finite"]]
    summary = dict(
        rows=len(rows),
        rows_with_a_non_finite_ratio=len(nonfinite),
        non_finite_rows_that_return=[r["a"] for r in nonfinite
                                     if r["caller_sees"] == "returned"],
        non_finite_rows_that_raise=[r["a"] for r in nonfinite
                                    if r["caller_sees"] != "returned"],
        raising=[r["a"] for r in rows if r["caller_sees"] != "returned"],
        returning=[r["a"] for r in rows if r["caller_sees"] == "returned"])
    for k in sorted(summary):
        print(" ", k, summary[k])
    _vb3.dump("v10_sliver", dict(rows=rows, summary=summary))


if __name__ == "__main__":
    main()

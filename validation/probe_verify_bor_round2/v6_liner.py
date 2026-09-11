"""V6 -- D4.  WHICH LAYER THE ``warn_own`` MESSAGE BLAMES, and whether a liner
against a DOMAIN END still has an owner.

Two independent questions, both measured PRE and POST:

  (1) ATTRIBUTION.  A liner in layer 0 next to an UNSTRUCTURED layer 1 (whose
      own segment list is a single full-radius entry).  The message says "the
      LAYER'S OWN segment list asked for" the cell; layer 1's did not.  Which
      layers carry a ``warn_own`` verdict?

  (2) COVERAGE.  The same liner at the OUTER wall, at the AXIS and in the
      INTERIOR, over a WIDTH LADDER -- because "the three positions behave
      identically" is a claim about the ladder, not about two widths.

Runs on the PRE tree (f2d331c5), which has no ``w_min_own_frac``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

import lumenairy  # noqa: E402
from lumenairy.elements.bor import _sem_contract as C  # noqa: E402
from lumenairy.elements.bor.bor_stack import BORStack  # noqa: E402

RBIG = 24.0
WIDTHS = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]


def _prov(name):
    import scipy
    import threadpoolctl
    return dict(tag=name, lumenairy_file=lumenairy.__file__,
                python=sys.version.split()[0], numpy=np.__version__,
                scipy=scipy.__version__,
                blas_arch=sorted({str(d.get("architecture"))
                                  for d in threadpoolctl.threadpool_info()}),
                env={k: os.environ.get(k) for k in
                     ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS",
                      "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})


def _report(st):
    prev = C.BOR_SEM_MESH_GUARD
    C.BOR_SEM_MESH_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.solve()
        return list(getattr(st, "_sem_mesh_report", []) or [])
    finally:
        C.BOR_SEM_MESH_GUARD = prev


def _emitted(st):
    """The messages a caller ACTUALLY receives with the contract armed."""
    prev = C.BOR_SEM_MESH_GUARD
    C.BOR_SEM_MESH_GUARD = True
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            st.solve()
        return [str(x.message) for x in w
                if "segment list" in str(x.message)
                or "sliver" in str(x.message).lower()
                or "narrow" in str(x.message).lower()]
    finally:
        C.BOR_SEM_MESH_GUARD = prev


def _stack(position, w, n_layers):
    """``n_layers`` = 1 (the liner alone) or 2 (the liner plus an UNSTRUCTURED
    neighbour whose own segment list is one full-radius entry)."""
    st = BORStack(Rbig=RBIG, m=1, N=120, n_superstrate=1.0, n_substrate=1.5,
                  basis="sem", degree=8)
    if position == "outer":
        segs = [(RBIG - w, 2.0), (RBIG, 6.0)]
    elif position == "axis":
        segs = [(w, 6.0), (RBIG, 2.0)]
    else:
        segs = [(6.0, 6.0), (6.0 + w, 2.0), (RBIG, 2.0)]
    st.add_layer(0.5, segments=segs)
    if n_layers == 2:
        st.add_layer(0.5, eps=1.21)        # asks for NOTHING but the domain
    st.set_source(k0=2.0)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="local")
    a = ap.parse_args()
    rows = []
    for n_layers in (1, 2):
        for position in ("outer", "axis", "middle"):
            for wf in WIDTHS:
                st = _stack(position, wf * RBIG, n_layers)
                recs = _report(st)
                per = []
                for r in recs:
                    per.append(dict(
                        layer=int(r["layer"]),
                        w_min_frac=float(r["w_min_frac"]),
                        w_min_own_frac=float(r.get("w_min_own_frac",
                                                   float("nan"))),
                        w_min_union_frac=float(r["w_min_union_frac"]),
                        q_excess=float(r["q_excess"]),
                        verdict=C.verdict(r)))
                st2 = _stack(position, wf * RBIG, n_layers)
                msgs = _emitted(st2)
                row = dict(n_layers=n_layers, position=position, w_frac=wf,
                           per_layer=per,
                           verdicts=sorted({p["verdict"] for p in per}),
                           warn_own_layers=[p["layer"] for p in per
                                            if p["verdict"] == "warn_own"],
                           q_max=max(p["q_excess"] for p in per),
                           n_messages=len(msgs),
                           messages=[m[:200] for m in msgs])
                rows.append(row)
                print("L=%d %-7s w=%-8.1g verdicts=%-14s warn_own_layers=%-8s "
                      "q=%.4g msgs=%d"
                      % (n_layers, position, wf, row["verdicts"],
                         row["warn_own_layers"], row["q_max"], len(msgs)),
                      flush=True)
    out = dict(provenance=_prov(a.tag), Rbig=RBIG,
               has_w_min_own=any(np.isfinite(p["w_min_own_frac"])
                                 for r in rows for p in r["per_layer"]),
               rows=rows)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "v6_liner_%s.json" % (a.tag,))
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote", p)


if __name__ == "__main__":
    main()

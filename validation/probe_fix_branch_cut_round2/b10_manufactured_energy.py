"""B10 -- the loudest reading: the pre-round-2 hybrid PMM MANUFACTURES energy.

`b2_pmm_interface.py` found a three-layer `PMM2DStackHybrid` whose pre-round-2
lossless closure defect is not 1e-04 but **+2.14**, i.e. `sum R + T = 4.14` on a
provably lossless stack -- more than twice the incident power -- with
`cond(a+b)` at the layer/spacer interface reading 1.96e+09.  This probe isolates
that fixture, records whether the library says anything about it, and walks the
truncation and the spacer detune around it.

The geometry differs from `b7_spacer.py`'s only in the CELL SAMPLING: a 32-pixel
grid whose walls sit at +/- 0.25 of the period (three strips per axis) instead
of a 6-pixel grid.  Same device class, same coincidence, two orders of magnitude
worse outcome -- which is itself the point: the size of the error is a property
of the mount, not of the defect.

Usage: OPENBLAS_NUM_THREADS=<n> PYTHONPATH=. python b10_manufactured_energy.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_fixtures as F  # noqa: E402

WL, PX, D = F.WL, F.PX, F.DEPTH
HOST, NS_OFF = 2.25, 1.63
WEAK = HOST * (1.0 + 1e-6)


def stack(M=3, spacer=HOST, nsub=NS_OFF, S=32, theta=0.0):
    from lumenairy.elements.pmm import PMM2DStackHybrid
    st = PMM2DStackHybrid(PX, PX, n_substrate=nsub, n_superstrate=1.0,
                          degree=7, n_orders=M, symmetry=False)
    if spacer is not None:
        st.add_layer(0.1e-6, eps=spacer)
    st.add_layer(D, eps_cell=F.pillar_cell(S=S, host=HOST, pillar=WEAK))
    if spacer is not None:
        st.add_layer(0.1e-6, eps=spacer)
    return st.set_source(WL, theta=theta).solve()


def run(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = fn()
        except Exception as exc:
            return dict(raised=type(exc).__name__, message=repr(exc)[:200],
                        warnings=sorted({type(x.message).__name__
                                         for x in w}))
    R = np.asarray(res[1], dtype=float)
    T = np.asarray(res[2], dtype=float)
    return dict(raised=None, sumRT=float(np.sum(R) + np.sum(T)),
                closure=float(np.sum(R) + np.sum(T) - 2.0),
                sumR=float(np.sum(R)), sumT=float(np.sum(T)),
                maxR=float(np.max(R)), maxT=float(np.max(T)),
                rt=F.rt_vec(res).tolist(),
                warnings=sorted({type(x.message).__name__ for x in w}))


def main():
    F.require_local_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "b10.json"
    cases = []
    for M in (2, 3, 4, 5, 6):
        cases.append(("M%d_spacer" % M, lambda m=M: stack(M=m)))
        cases.append(("M%d_nospacer" % M,
                      lambda m=M: stack(M=m, spacer=None)))
    for d in (0.0, 1e-9, 1e-6, 1e-3, 1e-1):
        cases.append(("detune_%g" % d,
                      lambda dd=d: stack(M=3, spacer=HOST * (1.0 + dd))))
    cases.append(("oblique", lambda: stack(M=3, theta=0.25)))
    rows = {}
    print("%-16s %-13s %-13s %-13s %-11s %s"
          % ("fixture", "PRE sum R+T", "POST sum R+T", "PRE closure",
             "motion", "PRE warnings / raise"))
    for name, fn in cases:
        with F.PreSqrtDecayPMM():
            pre = run(fn)
        with F.PostSqrtDecayPMM():
            post = run(fn)
        mv = (F.motion(pre.get("rt"), post.get("rt"))
              if pre.get("rt") and post.get("rt") else None)
        rows[name] = dict(pre=pre, post=post, motion=mv)
        print("%-16s %-13s %-13s %-13s %-11s %s"
              % (name,
                 "%.9f" % pre["sumRT"] if pre["raised"] is None else "RAISED",
                 "%.9f" % post["sumRT"] if post["raised"] is None
                 else "RAISED",
                 "%+.4e" % pre["closure"] if pre["raised"] is None else "-",
                 "%.3e" % mv if mv is not None else "-",
                 (pre["raised"] or "") + " "
                 + ",".join(pre.get("warnings", []) or ["SILENT"])))
    F.dump(out, dict(rows=rows,
                     openblas_num_threads=os.environ.get(
                         "OPENBLAS_NUM_THREADS", "unpinned")))


if __name__ == "__main__":
    main()

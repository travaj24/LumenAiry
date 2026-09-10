"""B7 -- THE HEADLINE: a UNIFORM SPACER makes the hybrid PMM's answer WRONG.

Found while measuring the JAX twins (b5): on a three-layer
``PMM2DStackHybrid`` -- uniform ``eps = 2.25`` spacer / weakly modulated
``eps = 2.25`` cell / uniform ``eps = 2.25`` spacer -- the NumPy arm's lossless
closure defect read ``+2.2091e-07`` where the SAME stack without the spacers
read ``+5.7732e-15`` and the same stack off the coincidence read
``+1.0214e-14``.  Seven decades, on a cell whose truncation error is at the
arithmetic floor.

That is the coincidence partner section 6 of the round-1 verification named on
the RCWA side -- a uniform LAYER, whose modes are built by the analytic
Rayleigh helper in EXACT arithmetic exactly as a half-space region's are, so a
mis-rooted mode of the neighbouring STRUCTURED layer is that uniform layer's own
BACKWARD mode -- reaching the PMM through its own unrepaired copy of
``_sqrt_decay``.

This probe isolates it: the same family swept over

  * the coincidence CLASS (region / spacer / both / none),
  * the spacer's permittivity walked OFF the layer background (the control
    that decides whether it is the LAYER-LAYER coincidence),
  * the truncation ``n_orders``,
  * the modulation strength (weak = converged, strong = truncation-limited),

with BOTH branch bodies installed in one interpreter, so the arm is the only
thing that changes between the two readings of one solve.

Usage: OPENBLAS_NUM_THREADS=<n> PYTHONPATH=. python b7_spacer.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_fixtures as F  # noqa: E402

WL, PX, D = F.WL, F.PX, F.DEPTH
NS, NS_OFF, HOST = 1.5, 1.63, 2.25
S = 6
LAYOUT = np.zeros((S, S), dtype=np.int64)
LAYOUT[2:4, 2:4] = 1


def cell(pillar, host=HOST):
    c = np.full((S, S), host + 0j)
    c[2:4, 2:4] = pillar
    return c


def stack(spacer_eps, nsub, pillar, M=3, sym=False, theta=0.0, deg=7):
    from lumenairy.elements.pmm import PMM2DStackHybrid
    st = PMM2DStackHybrid(PX, PX, n_substrate=nsub, n_superstrate=1.0,
                          degree=deg, n_orders=M, symmetry=sym)
    if spacer_eps is not None:
        st.add_layer(0.1e-6, eps=spacer_eps)
    st.add_layer(D, eps_cell=cell(pillar))
    if spacer_eps is not None:
        st.add_layer(0.1e-6, eps=spacer_eps)
    return st.set_source(WL, theta=theta).solve()


def fixtures():
    out = []
    WEAK = HOST * (1.0 + 1e-6)
    for M in (2, 3, 4, 5):
        out.append(("weak_none_M%d" % M, "none",
                    lambda m=M: stack(None, NS_OFF, WEAK, m)))
        out.append(("weak_region_M%d" % M, "region",
                    lambda m=M: stack(None, NS, WEAK, m)))
        out.append(("weak_spacer_M%d" % M, "spacer",
                    lambda m=M: stack(HOST, NS_OFF, WEAK, m)))
        out.append(("weak_both_M%d" % M, "both",
                    lambda m=M: stack(HOST, NS, WEAK, m)))
    # the DISCRIMINATOR: walk the SPACER off the layer background, leaving the
    # substrate exactly where it was.  If the defect follows the spacer, the
    # partner is the LAYER and not the region.
    for d in (0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1):
        out.append(("spacer_detune_%g" % d, "spacer-detune",
                    lambda dd=d: stack(HOST * (1.0 + dd), NS_OFF,
                                       HOST * (1.0 + 1e-6), 3)))
    # strength ladder: how weak must the modulation be for the defect to be
    # visible above the hybrid's own Fourier floor?
    for w in (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0):
        out.append(("mod_%g_both" % w, "both",
                    lambda ww=w: stack(HOST, NS, HOST * (1.0 + ww), 3)))
        out.append(("mod_%g_none" % w, "none",
                    lambda ww=w: stack(None, NS_OFF, HOST * (1.0 + ww), 3)))
    # fold on / oblique
    out.append(("weak_both_sym", "both",
                lambda: stack(HOST, NS, HOST * (1.0 + 1e-6), 3, sym=True)))
    out.append(("weak_both_oblique", "both",
                lambda: stack(HOST, NS, HOST * (1.0 + 1e-6), 3, theta=0.25)))
    out.append(("weak_both_deg11", "both",
                lambda: stack(HOST, NS, HOST * (1.0 + 1e-6), 3, deg=11)))
    # LOSSY control: the sign is physics there, so both arms must agree to the
    # last bit and the coincidence must cost nothing.
    out.append(("lossy_both", "lossy",
                lambda: stack(HOST + 1e-3j, NS,
                              HOST * (1.0 + 1e-6) + 1e-3j, 3)))
    return out


def one(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            with F.PMMEigSpy() as spy:
                res = fn()
        except Exception as exc:
            return dict(raised=type(exc).__name__, message=repr(exc)[:200])
    return dict(raised=None, closure=F.closure_jones(res),
                rt=F.rt_vec(res).tolist(), spy=spy.summary(),
                warnings=sorted({type(x.message).__name__ for x in w}))


def main():
    F.require_local_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "b7.json"
    rows = {}
    print("%-22s %-14s %-13s %-13s %-11s %s"
          % ("fixture", "class", "PRE closure", "POST closure", "motion",
             "incoming(PRE)"))
    for name, cls, fn in fixtures():
        with F.PreSqrtDecayPMM():
            pre = one(fn)
        with F.PostSqrtDecayPMM():
            post = one(fn)
        mv = (F.motion(pre.get("rt"), post.get("rt"))
              if pre.get("rt") and post.get("rt") else None)
        rows[name] = dict(cls=cls, pre=pre, post=post, motion=mv)
        print("%-22s %-14s %-13s %-13s %-11s %s"
              % (name, cls,
                 "%+.4e" % pre["closure"] if pre["raised"] is None
                 else pre["raised"],
                 "%+.4e" % post["closure"] if post["raised"] is None
                 else post["raised"],
                 "%.3e" % mv if mv is not None else "-",
                 pre.get("spy", {}).get("incoming_after_exact_pin")))
    F.dump(out, dict(rows=rows,
                     openblas_num_threads=os.environ.get(
                         "OPENBLAS_NUM_THREADS", "unpinned")))


if __name__ == "__main__":
    main()

"""B8 -- is the REPAIRED PMM answer the RIGHT one?

The lossless closure defect is an independent oracle for "wrong", but it cannot
by itself say the repaired answer is right: a solve can conserve energy and
still put the power in the wrong orders.  This probe answers that with a
DIFFERENT METHOD on the SAME geometry -- the Fourier RCWA stack, whose modal
branch round 1 pinned, at a truncation where its own answer is converged.

Geometry: the three-layer stack of b7 -- uniform ``eps = 2.25`` spacer, a
weakly modulated ``eps = 2.25`` cell, uniform spacer -- which the hybrid PMM
returns 2.4e-03 wrong per order on the PRE arm at one thread.

The per-order comparison is made on the (0,0) and first orders that BOTH
methods carry, so no order-set mapping is assumed beyond the shared window.

Usage: OPENBLAS_NUM_THREADS=<n> PYTHONPATH=. python b8_reference.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_fixtures as F  # noqa: E402

WL, PX, D = F.WL, F.PX, F.DEPTH
NS_OFF, HOST = 1.63, 2.25
WEAK = HOST * (1.0 + 1e-6)
S6 = 6


def cell6(pillar):
    c = np.full((S6, S6), HOST + 0j)
    c[2:4, 2:4] = pillar
    return c


def pmm_stack(M=4):
    from lumenairy.elements.pmm import PMM2DStackHybrid
    st = PMM2DStackHybrid(PX, PX, n_substrate=NS_OFF, n_superstrate=1.0,
                          degree=7, n_orders=M, symmetry=False)
    st.add_layer(0.1e-6, eps=HOST)
    st.add_layer(D, eps_cell=cell6(WEAK))
    st.add_layer(0.1e-6, eps=HOST)
    return st.set_source(WL, theta=0.0).solve()


def cell_up(pillar, rep=12):
    """The SAME geometry as :func:`cell6`, sampled ``rep`` x finer.  The cell
    is piecewise constant on the 6 x 6 walls, so block-replication is EXACT --
    it changes the sampling the Fourier path needs and not the device."""
    return np.kron(cell6(pillar), np.ones((rep, rep), dtype=complex))


def rcwa_stack(M):
    from lumenairy.elements.rcwa import RCWAStack
    st = RCWAStack(PX, period_y=PX, n_substrate=NS_OFF,
                   n_superstrate=1.0, n_orders=M, n_orders_y=M)
    st.add_layer(0.1e-6, eps=HOST)
    st.add_layer(D, eps_cell=cell_up(WEAK))
    st.add_layer(0.1e-6, eps=HOST)
    return st.set_source(WL, theta=0.0).solve().efficiencies()


def order_map(res):
    """``{(m, n): (R, T)}`` for a Jones-shaped return, summed over the two
    incident polarizations so the comparison does not assume a shared
    polarization gauge between the methods."""
    o = np.asarray(res[0])
    R = np.asarray(res[1], dtype=float)
    T = np.asarray(res[2], dtype=float)
    if R.ndim == 2:
        R, T = R.sum(axis=0), T.sum(axis=0)
    return {(int(a), int(b)): (float(R[i]), float(T[i]))
            for i, (a, b) in enumerate(o)}


def compare(a, b):
    keys = sorted(set(a) & set(b))
    dR = max((abs(a[k][0] - b[k][0]) for k in keys), default=float("nan"))
    dT = max((abs(a[k][1] - b[k][1]) for k in keys), default=float("nan"))
    return dict(n_shared=len(keys), dR=dR, dT=dT, d=max(dR, dT))


def main():
    F.require_local_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "b8.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refs = {}
        for M in (4, 6, 8):
            r = rcwa_stack(M)
            refs[M] = order_map(r)
            print("RCWA reference M=%d  closure %+.4e  n_orders %d"
                  % (M, F.closure_jones(r), len(refs[M])))
        conv = compare(refs[6], refs[8])
        print("RCWA self-convergence 6 vs 8: max |d| = %.4e" % conv["d"])

        arms = {}
        for M in (3, 4, 5):
            with F.PreSqrtDecayPMM():
                pre = pmm_stack(M)
            with F.PostSqrtDecayPMM():
                post = pmm_stack(M)
            a_pre, a_post = order_map(pre), order_map(post)
            arms[M] = dict(
                pre_closure=F.closure_jones(pre),
                post_closure=F.closure_jones(post),
                pre_vs_ref=compare(a_pre, refs[8]),
                post_vs_ref=compare(a_post, refs[8]),
                pre_vs_post=compare(a_pre, a_post))
            a = arms[M]
            print("PMM M=%d  PRE closure %+.4e vs ref %.4e | "
                  "POST closure %+.4e vs ref %.4e | PRE-POST %.4e"
                  % (M, a["pre_closure"], a["pre_vs_ref"]["d"],
                     a["post_closure"], a["post_vs_ref"]["d"],
                     a["pre_vs_post"]["d"]))
    F.dump(out, dict(arms=arms, rcwa_self_convergence=conv,
                     openblas_num_threads=os.environ.get(
                         "OPENBLAS_NUM_THREADS", "unpinned")))


if __name__ == "__main__":
    main()

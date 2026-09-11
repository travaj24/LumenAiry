"""V14 -- does the branch-cut fix close X-1?

X-1 is the library's own DOCUMENTED, OPEN instability class, pinned by
``tests/unit/test_m1_conditioning_guard.py::
test_x1_defect_is_reproduced_and_flagged_but_NOT_closed``:

  * ``19 TE`` returns ``R+T = 1.018`` on Windows and RAISES on WSL -- a literal
    build-dependent answer on the DEFAULT path;
  * ``21 TE`` returns ``sum(R) = 3.2e-2`` on BOTH builds against a converged
    ``2.0e-4`` -- 160x wrong, agreed to every digit, inside the energy
    warning's documented silent window;
  * ``12 TE`` and ``20 TE`` disagree across builds by 8.9e-04 and 8.3e-02.

Its geometry is ``THIN``: period 10 um, ridge 1.55, groove 1.5, substrate 1.5,
SUPERSTRATE 1.5, depth 0.5 um, duty 0.5, at ``n_orders`` 6..30.  The groove
index equals BOTH half-spaces' -- a coincidence on both sides at once -- which
is precisely the state the branch-cut fix addresses.  So the question is
whether X-1 was this defect all along.

Both arms in one process, the census armed exactly as the M1 test arms it, so
the flag counts are the library's own instrument rather than a re-derivation.

Usage:  OPENBLAS_NUM_THREADS=<n> python v14_x1.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_fixtures as V  # noqa: E402

WL = 700e-9
THIN = dict(period=10e-6, n_ridge=1.55, n_groove=1.5, n_substrate=1.5,
            n_superstrate=1.5, depth=0.5e-6, duty_cycle=0.5)
LADDER = tuple(range(6, 31))


def solve(M, pol):
    from lumenairy.elements.rcwa import _core as rc
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    prev = rc._INV_CENSUS
    rc._INV_CENSUS = []
    raised, close, sumR = None, float("nan"), float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _o, R, T = rcwa_efficiency_1d(
                    THIN["period"], THIN["n_ridge"], THIN["n_groove"],
                    THIN["n_substrate"], THIN["n_superstrate"], THIN["depth"],
                    THIN["duty_cycle"], WL, angle=0.0, polarization=pol,
                    n_orders=M, stabilize=False)
                close = abs(float(np.sum(R) + np.sum(T)) - 1.0)
                sumR = float(np.sum(R))
            except Exception as exc:
                raised = type(exc).__name__
        census = list(rc._INV_CENSUS)
    finally:
        rc._INV_CENSUS = prev
    flagged = [c for c in census
               if np.isfinite(c[2]) and c[2] < rc._INV_RCOND_SCREEN]
    return dict(M=M, pol=pol, raised=raised, close=close, sumR=sumR,
                n_census=len(census), n_flagged=len(flagged),
                rcond_min=(min(c[2] for c in census if np.isfinite(c[2]))
                           if census else None),
                n_refused=sum(1 for c in census if c[4]))


def ladder(pol):
    return [solve(M, pol) for M in LADDER]


def summarise(rows):
    ok = [r for r in rows if r["raised"] is None]
    clean = [r for r in ok if r["close"] < 1e-9]
    tail = sorted(clean, key=lambda r: r["M"])[-max(3, len(clean) // 3):] \
        if len(clean) >= 3 else clean
    ref = float(np.median([r["sumR"] for r in tail])) if tail else float("nan")
    worst = max(ok, key=lambda r: abs(r["sumR"] / ref - 1.0)) if (
        ok and np.isfinite(ref)) else None
    return dict(
        n=len(rows), n_raised=sum(1 for r in rows if r["raised"]),
        n_clean=len(clean), converged_sumR=ref,
        n_flagged_cells=sum(1 for r in rows if r["n_flagged"]),
        n_refused_cells=sum(1 for r in rows if r["n_refused"]),
        worst_close=max(r["close"] for r in ok) if ok else None,
        worst_relative_sumR_error=(abs(worst["sumR"] / ref - 1.0)
                                   if worst else None),
        worst_cell=(worst["M"] if worst else None),
        rcond_min=min((r["rcond_min"] for r in rows
                       if r["rcond_min"] is not None), default=None),
    )


def main():
    V.require_local_tree()
    out = sys.argv[1]
    V.claim_output(out)
    res = {}
    for pol in ("te", "tm"):
        post = ladder(pol)
        with V.PreSqrtDecay():
            pre = ladder(pol)
        res[pol] = dict(post=post, pre=pre,
                        post_summary=summarise(post),
                        pre_summary=summarise(pre))
        for arm in ("pre", "post"):
            s = res[pol][arm + "_summary"]
            print("%-4s %-4s raised=%-2s flaggedCells=%-3s refusedCells=%-2s "
                  "worstClose=%-10s worstRelSumR=%-10s (M=%s) rcondMin=%s"
                  % (pol.upper(), arm, s["n_raised"], s["n_flagged_cells"],
                     s["n_refused_cells"],
                     "%.3e" % s["worst_close"] if s["worst_close"] is not None
                     else "-",
                     "%.3e" % s["worst_relative_sumR_error"]
                     if s["worst_relative_sumR_error"] is not None else "-",
                     s["worst_cell"],
                     "%.3e" % s["rcond_min"] if s["rcond_min"] is not None
                     else "-"))
        print("     the four PINNED cells (M, arm): close / sumR / raised")
        for M in (12, 19, 20, 21):
            for arm in ("pre", "post"):
                r = next(x for x in res[pol][arm] if x["M"] == M)
                print("       M=%-3d %-4s close=%-10s sumR=%-12s raised=%s "
                      "flagged=%d" % (
                          M, arm, "%.4e" % r["close"], "%.6e" % r["sumR"],
                          r["raised"], r["n_flagged"]))
    V.dump(out, dict(ladders=res,
                     openblas_num_threads=os.environ.get(
                         "OPENBLAS_NUM_THREADS", "unpinned")))


if __name__ == "__main__":
    main()

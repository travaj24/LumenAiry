"""V1 -- the BLAS-thread-count dependence, re-measured on both trees.

CLAIM UNDER TEST (audit sec. 2): the even-sector disagreement
``max|R_full - R_even|`` on the coincident anisotropic fixture is DETERMINISTIC
within a (build, tree, thread-count) triple and moves over twelve decades ACROSS
thread counts on the PRE tree, with the ``1e-8`` bar of the original test sitting
inside that spread; and the dependence is GONE on the POST tree.

The thread count is taken from the environment the caller pinned on the command
line, so this script measures the setting it was actually given and records it.
Three repeats in ONE process establish within-process determinism; running the
script once per thread count establishes the across-process reading.

Usage:  OPENBLAS_NUM_THREADS=<n> python v1_threads.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_fixtures as V  # noqa: E402

REPEATS = 3


def _solve(sym, cell, n_sub=1.5, theta=0.0, phi=0.0, n_orders=5):
    from lumenairy.elements.rcwa import rcwa_jones_2d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rcwa_jones_2d(V._P, V._P, cell, n_sub, 1.0, V._DEPTH, V._WL,
                             theta=theta, phi=phi, n_orders_x=n_orders,
                             n_orders_y=n_orders, symmetry=sym)


def one_pass():
    cell = V.uniaxial_cell()
    full = _solve(False, cell)
    even = _solve(True, cell)
    dR = float(np.max(np.abs(np.asarray(full[1]) - np.asarray(even[1]))))
    dJ = float(np.max(np.abs(np.asarray(full[3]) - np.asarray(even[3]))))
    return dict(
        dR=dR, dJ=dJ,
        closure_full=V.closure_defect_jones(full),
        closure_even=V.closure_defect_jones(even),
        RT_full=float(np.sum(np.asarray(full[1])) + np.sum(np.asarray(full[2]))),
        RT_even=float(np.sum(np.asarray(even[1])) + np.sum(np.asarray(even[2]))),
        # a NON-coincident control at the same truncation: n_substrate = 1.6,
        # so eps_sub = 2.56 != 2.25 and no layer mode can meet a region mode.
        dR_offcoinc=float(np.max(np.abs(
            np.asarray(_solve(False, cell, n_sub=1.6)[1])
            - np.asarray(_solve(True, cell, n_sub=1.6)[1])))),
    )


def main():
    V.require_local_tree()
    out = sys.argv[1]
    V.claim_output(out)
    reps = [one_pass() for _ in range(REPEATS)]
    keys = sorted(reps[0])
    identical = {k: all(
        repr(r[k]) == repr(reps[0][k]) for r in reps) for k in keys}
    payload = dict(
        repeats=reps,
        deterministic_in_process=identical,
        all_deterministic=all(identical.values()),
        openblas_num_threads=os.environ.get("OPENBLAS_NUM_THREADS", "unpinned"),
    )
    V.dump(out, payload)
    r0 = reps[0]
    print("threads=%-8s arm=%-4s dR=%.6e dJ=%.6e closure_full=%+.4e "
          "closure_even=%+.4e dR_off=%.4e det=%s"
          % (payload["openblas_num_threads"],
             V.arm_stamp()["arm"], r0["dR"], r0["dJ"], r0["closure_full"],
             r0["closure_even"], r0["dR_offcoinc"],
             payload["all_deterministic"]))


if __name__ == "__main__":
    main()

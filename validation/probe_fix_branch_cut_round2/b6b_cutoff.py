"""B6b -- the LAYER-CUTOFF corner, driven properly.

The band-scale question (D3/D4) is decided at a mount where one modal
eigenvalue passes through zero: for ``lam^2 = -s + i eta`` the principal root's
real part is ``eta / (2 sqrt(s))``, so the ARRAY-MAX band ratio grows without
limit as ``s -> 0`` at fixed backward error, while the PER-MODE ratio does not.
The round-1 verification reached ``min|lam^2| = 4.495e-15`` by trisection and
measured a noise-side band ratio of 6.7172e-09 there -- 1.5x under the ``1e-8``
bar.

``b6_band_scale.py``'s own trisection is crude and only reached 1.7e-05, so this
probe drives the cutoff with a BOUNDED SCALAR MINIMISATION of
``g(p) = min |lam^2|`` over a bracket found by a coarse scan, which is the
right instrument for a smooth non-negative function with an interior zero.

Both candidate shapes are scored on every mode of every mount, so the decision
of section 5 of the report is made on the corner that actually sets it.

Usage: OPENBLAS_NUM_THREADS=<n> PYTHONPATH=. python b6b_cutoff.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b6_band_scale as B6  # noqa: E402
import b_fixtures as F  # noqa: E402

WL = F.WL


def eigs(n_ridge, pol, M, period=1.0e-6, depth=0.4e-6, duty=0.5):
    """Every layer eigenvalue array one 1-D solve produces."""
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with B6.RcwaEigSpy() as sp:
            try:
                rcwa_efficiency_1d(period, n_ridge, 1.0, 1.5, 1.0, depth,
                                   duty, WL, polarization=pol, n_orders=M)
            except Exception:
                pass
    return [w for w in sp.seen if np.asarray(w).size]


def smallest(n_ridge, pol, M):
    arrs = eigs(n_ridge, pol, M)
    return min((float(np.min(np.abs(w))) for w in arrs), default=float("inf"))


def drive(pol, M, lo=1.05, hi=3.2, n_scan=90):
    """Coarse scan for a bracket, then a bounded minimisation of
    ``min|lam^2|`` inside it.  Returns the mount and its eigenvalue arrays."""
    grid = np.linspace(lo, hi, n_scan)
    vals = [smallest(p, pol, M) for p in grid]
    i = int(np.argmin(vals))
    a = grid[max(i - 1, 0)]
    b = grid[min(i + 1, n_scan - 1)]
    res = minimize_scalar(lambda p: np.log10(max(smallest(p, pol, M),
                                                 1e-300)),
                          bounds=(a, b), method="bounded",
                          options=dict(xatol=1e-14, maxiter=200))
    p = float(res.x)
    return p, smallest(p, pol, M), eigs(p, pol, M)


def main():
    F.require_local_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "b6b.json"
    rows, mounts = [], []
    for pol in ("te", "tm"):
        for M in (5, 7, 11, 15):
            p, s, arrs = drive(pol, M)
            sub = []
            for w in arrs:
                sub.extend(B6.score(w))
            rows.extend(sub)
            mounts.append(dict(pol=pol, n_orders=M, n_ridge=p,
                               min_abs_lam2=s, n_scored=len(sub)))
            print("cutoff %s M=%-3d n_ridge=%.12f  min|lam^2|=%.4e  "
                  "modes scored %d" % (pol.upper(), M, p, s, len(sub)))
            # a LADDER of nearby mounts, so the corner is a family and not one
            # engineered point
            for d in (1e-3, 1e-5, 1e-7, 1e-9):
                for sgn in (-1.0, +1.0):
                    q = p * (1.0 + sgn * d)
                    sub2 = []
                    for w in eigs(q, pol, M):
                        sub2.extend(B6.score(w))
                    rows.extend(sub2)
                    mounts.append(dict(pol=pol, n_orders=M, n_ridge=q,
                                       min_abs_lam2=smallest(q, pol, M),
                                       n_scored=len(sub2)))
    summary = dict(
        n_mounts=len(mounts),
        min_abs_lam2_reached=min(m["min_abs_lam2"] for m in mounts),
        array_max=B6.gaps(rows, "array_max"),
        per_mode=B6.gaps(rows, "per_mode"),
        own=B6.gaps(rows, "own"),
        worst_own_ratio_caught_by_array_max=max(
            [r["own"] for r in rows if r["array_max"] <= 1e-8] or [0.0]),
        worst_own_ratio_caught_by_per_mode=max(
            [r["own"] for r in rows if r["per_mode"] <= 1e-8] or [0.0]),
    )
    for shape in ("array_max", "per_mode"):
        g = summary[shape]
        print("%-9s noise_max %-12s signal_min %-12s gap %s decades"
              % (shape,
                 "%.4e" % g["noise_max"] if g["noise_max"] else "-",
                 "%.4e" % g["signal_min"] if g["signal_min"] else "-",
                 "%.2f" % g["gap_decades"] if g["gap_decades"] else "-"))
    print("worst |Re r|/|r| conjugated by ARRAY-MAX: %.4e ; by PER-MODE: %.4e"
          % (summary["worst_own_ratio_caught_by_array_max"],
             summary["worst_own_ratio_caught_by_per_mode"]))
    F.dump(out, dict(summary=summary, mounts=mounts,
                     openblas_num_threads=os.environ.get(
                         "OPENBLAS_NUM_THREADS", "unpinned")))


if __name__ == "__main__":
    main()

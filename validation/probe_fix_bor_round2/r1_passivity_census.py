"""ROUND 2, D1 + D2 -- the two-sided nodal passivity population, and the
DETERMINISTIC conjuncts that do not depend on the energy.

WHAT IS MEASURED, with every 5.45.1 guard DISARMED so each row is the number
the solver RETURNS:

  * ``max(R + T) - 1``  -- the super-unity side the shipped screen already
    tests (D2's excess half);
  * ``min(R + T) - 1``  -- the DEFICIT side it does not test.  On a LOSSLESS
    passive stack closed by a PEC wall, ``R + T = 1`` is an EQUALITY: there is
    no absorption and no other exit, so a deficit is damage of exactly the same
    kind as an excess;
  * the CHANNEL COUNT against the closed-form Bessel-zero count (uniform
    half-space only, where that oracle exists), with its BASIN;
  * the INDEX-CEILING excess ``max(Re qn) / n_max - 1`` over the returned
    channels -- ``qn > n`` means ``gamma^2 < 0``, an axial index larger than
    the medium's own, which is unphysical by construction and therefore
    kernel-independent.

Run:  python validation/probe_fix_bor_round2/r1_passivity_census.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    banner,
    dump,
    exact_channel_count,
    nodal_stack,
    solve_disarmed,
    uniform_stack,
)


def _row(basis, family, m, N, rbl, k0=2.0, e_out=2.0, e_mid=4.0, e_hi=6.0):
    Rbig = float(rbl) * 2.0 * np.pi / k0
    if family == "uniform":
        layers = uniform_stack(Rbig, N, k0, m, e_out=e_out + 0j,
                               e_mid=e_mid + 0j, basis=basis)
        n_ceiling = float(np.sqrt(max(e_out, e_mid)))
        exact, basin = exact_channel_count(m, np.sqrt(e_out), k0, Rbig)
    else:
        layers = nodal_stack(e_hi + 0j, Rbig=Rbig, N=N, k0=k0, m=m,
                             e_out=e_out + 0j, basis=basis)
        n_ceiling = float(np.sqrt(max(e_out, e_hi)))
        exact, basin = exact_channel_count(m, np.sqrt(e_out), k0, Rbig)
    try:
        res = solve_disarmed(layers, k0)
    except Exception as exc:                       # pragma: no cover - probe
        return dict(basis=basis, family=family, m=m, N=N, rbl=rbl,
                    error="%s: %s" % (type(exc).__name__, exc))
    R, T = np.asarray(res["R"]), np.asarray(res["T"])
    E = R + T
    qn = np.asarray(res["q_inc"]) / k0
    # the incidence half-space's own index is what its channels are bounded by
    n_inc = float(np.sqrt(float(np.real(e_out))))
    ceil_excess = (float(np.max(np.real(qn))) / n_inc - 1.0
                   if qn.size else float("nan"))
    return dict(
        basis=basis, family=family, m=m, N=N, rbl=rbl, k0=k0, Rbig=Rbig,
        n_channels=int(R.size),
        exact_channels=int(exact), basin=float(basin),
        count_matches=bool(int(R.size) == int(exact)),
        excess=float(np.max(E)) - 1.0 if E.size else None,
        deficit=float(np.min(E)) - 1.0 if E.size else None,
        max_qn=float(np.max(np.real(qn))) if qn.size else None,
        n_inc=n_inc, n_ceiling=n_ceiling,
        ceiling_excess=ceil_excess,
    )


def main():
    rec = banner("r1_passivity_census")
    rows = []
    for basis in ("staggered", "nodal"):
        for family in ("uniform", "ring"):
            for m in (0, 1, 2, 3):
                for N in (120, 200):
                    for rbl in (0.5, 1.0, 2.0, 4.0, 8.0):
                        r = _row(basis, family, m, N, rbl)
                        rows.append(r)
                        if "error" in r:
                            print("  ERR %-10s %-8s m=%d N=%d rbl=%4.1f  %s"
                                  % (basis, family, m, N, rbl, r["error"]))
                        else:
                            print("  %-10s %-8s m=%d N=%3d rbl=%4.1f  "
                                  "nch=%3d/%3d  exc=%+.4e  def=%+.4e  "
                                  "ceil=%+.4e"
                                  % (basis, family, m, N, rbl,
                                     r["n_channels"], r["exact_channels"],
                                     r["excess"], r["deficit"],
                                     r["ceiling_excess"]))
    ok = [r for r in rows if "error" not in r]

    def stat(sel, key):
        v = [r[key] for r in sel if r.get(key) is not None
             and np.isfinite(r[key])]
        return (min(v), max(v), len(v)) if v else (None, None, 0)

    stag = [r for r in ok if r["basis"] == "staggered"]
    nod = [r for r in ok if r["basis"] == "nodal"]
    summary = {}
    for name, sel in (("staggered", stag), ("nodal", nod)):
        summary[name] = dict(
            n=len(sel),
            excess=stat(sel, "excess"),
            deficit=stat(sel, "deficit"),
            ceiling_excess=stat(sel, "ceiling_excess"),
            count_matches=sum(1 for r in sel if r["count_matches"]),
            count_matches_wide_basin=sum(
                1 for r in sel if r["count_matches"] and r["basin"] > 1e-2),
            n_wide_basin=sum(1 for r in sel if r["basin"] > 1e-2),
        )
    print("\n--- summary ---")
    for k, v in summary.items():
        print(" %s n=%d" % (k, v["n"]))
        print("   excess   min %.6e  max %.6e" % (v["excess"][0], v["excess"][1]))
        print("   deficit  min %.6e  max %.6e" % (v["deficit"][0], v["deficit"][1]))
        print("   ceiling  min %.6e  max %.6e" % (v["ceiling_excess"][0],
                                                  v["ceiling_excess"][1]))
        print("   count exact %d/%d (wide basin %d/%d)"
              % (v["count_matches"], v["n"], v["count_matches_wide_basin"],
                 v["n_wide_basin"]))
    dump("r1_passivity_census", dict(rows=rows, summary=summary), rec)


if __name__ == "__main__":
    main()

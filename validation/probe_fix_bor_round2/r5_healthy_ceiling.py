"""ROUND 2, D2 -- the LOWER edge of the two-sided nodal bar: how far from unity
an ACCURATE nodal answer gets.

The bar the guard refuses on must clear the population it must never refuse.
``r1_passivity_census`` establishes that population's shape; this probe
measures its CEILING on a deliberately wide sweep of the family the 5.45.1
build calls accurate -- UNIFORM layers on a small cell, where the nodal FD
basis's divergence-violating sea is smallest -- with the channel SET confirmed
against the div-conforming staggered twin on every row, so a row that is
secretly damaged cannot inflate the ceiling.

Swept: ``m`` 0..5, ``N`` 80/120/200, ``Rbig/lambda`` 0.25/0.5/1.0/2.0,
``k0`` 0.8/2.0/3.5 (a 4.4x range of scale), index contrasts (2, 4) and
(2.25, 9), and the three-layer and four-layer stack shapes -- 720 solves.
(``N = 300`` and ``Rbig/lambda = 1.5`` were dropped after a first pass ran past
an hour without adding a family: the ceiling is set by ``m`` and the cell
radius, not by the grid.)

Reported per row: ``|R + T - 1|`` (the TWO-SIDED measure), the channel set
against the twin, and ``Re qn / n - 1`` (the index-ceiling conjunct, which must
stay negative on every accurate row).

Run:  python validation/probe_fix_bor_round2/r5_healthy_ceiling.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import banner, dump, exact_channel_count, solve_disarmed, uni  # noqa: E402


def _stack(basis, m, Rbig, N, k0, e_out, e_mid, n_layers, thickness=0.4):
    import warnings

    from lumenairy.elements.bor.bor_solve import build_layer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = [build_layer(m, Rbig, N, uni(e_out), k0, basis=basis)]
        for j in range(n_layers - 2):
            e = e_mid if j % 2 == 0 else e_out
            out.append(build_layer(m, Rbig, N, uni(e), k0,
                                   thickness=thickness, basis=basis))
        out.append(build_layer(m, Rbig, N, uni(e_out), k0, basis=basis))
    return out


def row(m, N, rbl, k0, e_out, e_mid, n_layers):
    Rbig = float(rbl) * 2.0 * np.pi / k0
    rec = dict(m=m, N=N, rbl=rbl, k0=k0, e_out=e_out, e_mid=e_mid,
               n_layers=n_layers, Rbig=Rbig)
    try:
        nd = solve_disarmed(_stack("nodal", m, Rbig, N, k0, e_out, e_mid,
                                   n_layers), k0)
        st = solve_disarmed(_stack("staggered", m, Rbig, N, k0, e_out, e_mid,
                                   n_layers), k0)
    except Exception as exc:                       # pragma: no cover - probe
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        return rec
    En = np.asarray(nd["R"]) + np.asarray(nd["T"])
    Es = np.asarray(st["R"]) + np.asarray(st["T"])
    qn = np.asarray(nd["q_inc"]) / k0
    n_inc = float(np.sqrt(float(np.real(e_out))))
    exact, basin = exact_channel_count(m, np.sqrt(e_out), k0, Rbig)
    rec.update(
        n_nodal=int(En.size), n_stag=int(Es.size),
        set_ok=bool(En.size == Es.size),
        abs_nodal=float(np.max(np.abs(En - 1.0))) if En.size else None,
        abs_stag=float(np.max(np.abs(Es - 1.0))) if Es.size else None,
        excess=float(np.max(En)) - 1.0 if En.size else None,
        deficit=float(np.min(En)) - 1.0 if En.size else None,
        ceiling_excess=(float(np.max(np.real(qn))) / n_inc - 1.0
                        if qn.size else None),
        exact=int(exact), basin=float(basin))
    return rec


def main():
    rec = banner("r5_healthy_ceiling")
    rows = []
    sys.stdout.flush()
    contrasts = ((2.0, 4.0), (2.25, 9.0))
    for m in (0, 1, 2, 3, 4, 5):
        for N in (80, 120, 200):
            for rbl in (0.25, 0.5, 1.0, 2.0):
                for k0 in (0.8, 2.0, 3.5):
                    for (e_out, e_mid) in contrasts:
                        for n_layers in (3, 4):
                            rows.append(row(m, N, rbl, k0, e_out, e_mid,
                                            n_layers))
    ok = [r for r in rows if "error" not in r and r.get("abs_nodal") is not None]
    bad = [r for r in rows if "error" in r]
    clean = [r for r in ok if r["set_ok"] and r["ceiling_excess"] < 0.0]
    dirty = [r for r in ok if not (r["set_ok"] and r["ceiling_excess"] < 0.0)]
    print("\n--- summary ---")
    print("  rows %d (%d errors)" % (len(rows), len(bad)))
    print("  CLEAN (set matches the staggered twin AND ceiling silent): %d"
          % (len(clean),))
    if clean:
        v = sorted((r["abs_nodal"], r) for r in clean)
        print("    |R+T-1| ceiling  = %.6e" % (v[-1][0],))
        for a, r in v[-6:]:
            print("      %.6e   m=%d N=%3d rbl=%4.2f k0=%.1f eps=(%.2f,%.2f) "
                  "L=%d  nch=%d  ceil=%+.3e"
                  % (a, r["m"], r["N"], r["rbl"], r["k0"], r["e_out"],
                     r["e_mid"], r["n_layers"], r["n_nodal"],
                     r["ceiling_excess"]))
        print("    worst |R+T-1| on the STAGGERED twins of the clean rows "
              "= %.6e" % (max(r["abs_stag"] for r in clean),))
        print("    worst ceiling excess on the clean rows = %+.6e"
              % (max(r["ceiling_excess"] for r in clean),))
        print("    worst EXCESS  = %+.6e   worst DEFICIT = %+.6e"
              % (max(r["excess"] for r in clean),
                 min(r["deficit"] for r in clean)))
    if dirty:
        w = sorted(r["abs_nodal"] for r in dirty)
        print("  NOT-CLEAN rows: %d, |R+T-1| %.6e .. %.6e"
              % (len(dirty), w[0], w[-1]))
        nset = sum(1 for r in dirty if not r["set_ok"])
        ncl = sum(1 for r in dirty if r["ceiling_excess"] >= 0.0)
        print("    (%d have a wrong channel set; %d trip the ceiling)"
              % (nset, ncl))
    dump("r5_healthy_ceiling", dict(rows=rows), rec)


if __name__ == "__main__":
    main()

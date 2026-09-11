"""CLASS C on BOR SEM -- ATTRIBUTION of the union/enrichment sliver, with the
two controls c1 could not supply.

c1 measured, on a ``delta`` ladder from 1e-1 down to 1e-7 of ``Rbig``:
spurious ``|q|`` up to 5.01e+06 times the physical index ceiling, interface
``rcond`` falling linearly with ``delta`` to 6.7e-10, and a per-order R
difference from the ``delta = 0`` reference that PLATEAUS near 0.11 instead of
vanishing -- while the closure stays at its healthy 1e-10 baseline.  Two
things must be established before that is a wrong answer:

  CONTROL 1 -- ORDER IDENTITY.  Both meshes change with ``delta`` (the
  breakpoint set gains an entry and ``equalize_meshes`` then pads every mesh
  differently), so the LAPACK ordering of the modal basis may PERMUTE and an
  index-wise comparison would be comparing different physical channels.  Here
  every comparison is made between orders MATCHED BY THEIR AXIAL WAVENUMBER
  ``q``, and the match residual is reported.

  CONTROL 2 -- IS IT THE WINDOW?  The +-1 neighbour-enrichment window unions
  the wall sets of layers ``i-1``, ``i``, ``i+1``.  Put the two walls in
  layers that are TWO apart -- with a wall-free spacer between them -- and the
  window never unions them, so NO sliver is manufactured while the geometry
  keeps exactly the same ``delta``.  If the damage is the window's, the
  separated arm is clean and the adjacent arm is not.

  ORACLE -- an ``basis='fd'`` ladder in ``N`` (200 / 400 / 800).  The FD basis
  has one uniform radial grid and no manufactured cell; it converges slowly
  but INDEPENDENTLY, and it says which of the two SEM answers is the physical
  one.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import closure, dump, pin_tree  # noqa: E402

print("TREE", pin_tree())

RBIG = 24.0
K0 = 2.0
E_HI = 2.45 ** 2
E_LO = 1.41 ** 2
WALL = 6.0
WALL2 = 9.0
M = 1


def build(delta, *, basis, degree, separated, N=200):
    """``separated=False``: the two ring layers are ADJACENT, so the +-1
    window unions their walls and manufactures the sliver.
    ``separated=True``: a wall-free spacer sits between them, so the window
    never sees both walls at once -- same geometry, no sliver."""
    from lumenairy import BORStack
    s = BORStack(RBIG, M, n_substrate=1.41, n_superstrate=1.41, N=N,
                 basis=basis, degree=degree)
    s.add_layer(0.4, eps=E_LO)
    s.add_layer(0.5, segments=[(WALL, E_HI), (WALL2, E_LO), (RBIG, E_LO)])
    if separated:
        # TWO wall-free spacers, not one.  With ONE spacer the CONTROL FAILS:
        # the +-1 window of the spacer itself is
        # ``walls[i-1] | walls[i] | walls[i+1]``, so a WALL-FREE layer between
        # the two ring layers INHERITS BOTH of their walls and the sliver is
        # manufactured in a layer that has no walls at all (measured: the
        # one-spacer control carried min_element = delta exactly, and broke
        # harder than the adjacent arm).  Two spacers put the two wall sets
        # three layers apart, which no window can span.
        s.add_layer(0.5, eps=E_LO)
        s.add_layer(0.5, eps=E_LO)
    s.add_layer(0.5, segments=[(WALL + delta, E_HI), (WALL2, E_LO),
                               (RBIG, E_LO)])
    s.add_layer(0.4, eps=E_LO)
    s.set_source(k0=K0)
    return s


def solve_one(delta, *, basis, degree, separated, N=200):
    s = build(delta, basis=basis, degree=degree, separated=separated, N=N)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = s.solve()
    d = s._last
    qmax, wmin = 0.0, np.inf
    for _t, L in ([("s", d["sup"]), ("b", d["sub"])]
                  + [("m", LL) for _tt, LL in d["mids"]]):
        qmax = max(qmax, float(np.max(np.abs(np.asarray(L["q"])))))
        mesh = L.get("mesh")
        if mesh is not None:
            wmin = min(wmin, float(np.diff(mesh.b).min()))
    return dict(delta=float(delta), basis=basis, degree=degree, N=N,
                separated=separated,
                q=np.asarray(res["q"], float).tolist(),
                R=np.asarray(res["R"], float).tolist(),
                T=np.asarray(res["T"], float).tolist(),
                n_orders=int(np.size(res["R"])), closure=closure(res),
                superunity=float(np.max(np.asarray(res["energy"])) - 1.0)
                if np.size(res["R"]) else None,
                qmax=qmax, qmax_over_ceiling=qmax / (2.45 * K0),
                min_element=None if not np.isfinite(wmin) else wmin,
                warnings=[str(x.message)[:140] for x in w][:2])


def q_matched(ref, row, key="R"):
    """Compare per-order values MATCHED BY q, not by index.

    Returns (max |dR| over matched orders, worst q match residual, n matched).
    """
    qa = np.asarray(ref["q"], float)
    qb = np.asarray(row["q"], float)
    if qa.size == 0 or qb.size == 0:
        return None, None, 0
    va = np.asarray(ref[key], float)
    vb = np.asarray(row[key], float)
    scale = max(float(np.max(np.abs(qa))), 1e-300)
    worst, resid, n = 0.0, 0.0, 0
    for i, q in enumerate(qa):
        j = int(np.argmin(np.abs(qb - q)))
        rr = abs(qb[j] - q) / scale
        if rr > 1e-3:                    # no counterpart: report, do not match
            continue
        worst = max(worst, abs(va[i] - vb[j]))
        resid = max(resid, rr)
        n += 1
    return worst, resid, n


def ladder(*, degree, separated, fracs):
    ref = solve_one(0.0, basis="sem", degree=degree, separated=separated)
    rows = []
    for f in fracs:
        r = solve_one(f * RBIG, basis="sem", degree=degree,
                      separated=separated)
        dR, resid, nmatch = q_matched(ref, r, "R")
        dT, _r2, _n2 = q_matched(ref, r, "T")
        r.update(frac=f, dR_matched=dR, dT_matched=dT, q_resid=resid,
                 n_matched=nmatch, n_ref=ref["n_orders"])
        rows.append(r)
    return ref, rows


def main():
    tag = os.environ.get("PROBE_TAG", "win")
    thr = os.environ.get("OPENBLAS_NUM_THREADS", "?")
    fracs = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6, 1e-7]
    payload = dict(threads=thr, arms=[], fd_oracle=[])

    for degree in (8, 12):
        for separated in (False, True):
            ref, rows = ladder(degree=degree, separated=separated,
                               fracs=fracs)
            payload["arms"].append(dict(degree=degree, separated=separated,
                                        ref=ref, rows=rows))
            arm = "SEPARATED (no window union)" if separated \
                else "ADJACENT (window unions -> sliver)"
            print(f"\n== {arm}  degree={degree}  ref: n_ord="
                  f"{ref['n_orders']} closure={ref['closure']:.3e} "
                  f"R[0]={ref['R'][0]:.10f} ==")
            print("  frac     delta      min_elem   |q|/ceil    closure    "
                  "R+T-1       dR (q-matched)  q_resid   matched  warn")
            for r in rows:
                print(f"  {r['frac']:.0e} {r['delta']:.3e} "
                      f"{r['min_element']:.3e} {r['qmax_over_ceiling']:.3e} "
                      f"{r['closure']:.3e} {r['superunity']:+.3e} "
                      f"{r['dR_matched']:.6e}    {r['q_resid']:.2e}  "
                      f"{r['n_matched']:3d}/{r['n_ref']:3d}  "
                      f"{len(r['warnings'])}")

    # ---- the independent FD oracle ------------------------------------- #
    print("\n== FD ORACLE (uniform grid, no manufactured cell) ==")
    for N in (200, 400, 800):
        rr = []
        for dl in (0.0, 1e-2 * RBIG, 1e-4 * RBIG, 1e-7 * RBIG):
            r = solve_one(dl, basis="fd", degree=8, separated=False, N=N)
            rr.append(r)
            payload["fd_oracle"].append(r)
        base = rr[0]
        print(f"  N={N:4d}  n_ord={base['n_orders']:3d} closure="
              f"{base['closure']:.3e}  R[0] at delta = 0 / 1e-2 / 1e-4 / "
              f"1e-7 of Rbig:  "
              + "  ".join(f"{x['R'][0]:.8f}" for x in rr))

    dump(f"c4_sliver_attrib_{tag}_t{thr}.json", payload)

    print("\n== HEADLINE: the ATTRIBUTION ==")
    print("  degree   ADJACENT dR floor      SEPARATED dR floor     ratio")
    for degree in (8, 12):
        a = next(x for x in payload["arms"]
                 if x["degree"] == degree and not x["separated"])
        b = next(x for x in payload["arms"]
                 if x["degree"] == degree and x["separated"])
        # the FLOOR = the value the ladder plateaus at (the last 4 rungs)
        fa = max(r["dR_matched"] for r in a["rows"][-4:])
        fb = max(r["dR_matched"] for r in b["rows"][-4:])
        print(f"  {degree:>5}    {fa:.6e}           {fb:.6e}      "
              f"{(fa / fb if fb > 0 else float('inf')):9.1f}")
    print("\n  CONTINUITY: the physical response to the wall shift, read off "
          "the WIDE rungs")
    for degree in (8, 12):
        a = next(x for x in payload["arms"]
                 if x["degree"] == degree and not x["separated"])
        wide = a["rows"][0]
        slope = wide["dR_matched"] / wide["delta"]
        for r in a["rows"]:
            r["move_over_physical"] = (r["dR_matched"] / (slope * r["delta"])
                                       if slope > 0 and r["delta"] > 0
                                       else None)
        worst = max(r["move_over_physical"] for r in a["rows"]
                    if r["move_over_physical"] is not None)
        print(f"    degree {degree}: slope {slope:.4e} per unit wall shift; "
              f"worst move / physical shift = {worst:,.0f}x   "
              f"[1-D guard's _SLIVER_MOVE_FACTOR = 100]")
    su = [r["superunity"] for x in payload["arms"] for r in x["rows"]
          if r["superunity"] is not None]
    print(f"\n  worst super-unity anywhere on the ladder: {max(su):+.4e}  "
          f"[1-D bars: trigger 1e-3, refuse 1e-2] -> a ported super-unity "
          f"screen would fire on "
          f"{sum(1 for v in su if v > 1e-3)}/{len(su)} rungs")
    nw = sum(len(r["warnings"]) for x in payload["arms"] for r in x["rows"])
    print(f"  UserWarnings emitted anywhere: {nw}")


if __name__ == "__main__":
    main()

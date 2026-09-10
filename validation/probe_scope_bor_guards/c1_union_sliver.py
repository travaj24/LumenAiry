"""CLASS C on BOR SEM -- (i) the UNION / ENRICHMENT sliver.

THE MECHANISM, AS BUILT.  ``BORStack._solve_sem`` gives every layer its own
radial element mesh, but the breakpoint set of layer ``i`` is the WINDOW
UNION of the ring walls of layers ``i-1``, ``i`` and ``i+1``::

    u = set(walls[i]) | set(walls[i-1]) | set(walls[i+1])          (bor_stack)

so two ADJACENT layers whose walls differ by ``delta`` manufacture an element
of width exactly ``delta`` in BOTH of their meshes.  ``build_mesh`` merges
breakpoints only when they are closer than ``1e-12 * Rbig``; there is NO
minimum-feature contract, no degradation warning and no super-unity screen.
That is the 1-D ``PMMStack`` union-grid sliver (``FIX_PMMSTACK_SLIVER_WALLS``)
in cylindrical coordinates, with the neighbour-enrichment window playing the
part the shared union grid plays there.

WHAT IS MEASURED, PER RUNG OF A ``delta`` LADDER:
  * per-order R / T against the EXACT ``delta -> 0`` reference (the same
    stack with the two walls coincident) and against an INDEPENDENT oracle
    (the ``basis='fd'`` solve at the same delta, which has no union grid at
    all -- see (v) below);
  * closure ``max |R + T - 1|`` -- the DETECTOR the 1-D guard screens on;
  * the SPURIOUS modal wavenumber ``max |q|`` against the physical ceiling
    ``n_max k0``, and against the 1-D predictor ``|q| ~ 0.65 N(N+1) / 4 /
    (k0 J)`` with ``J = w / 2`` (so ``|q| ~ 0.325 p(p+1) / (k0 w)``);
  * the interface / mortar operator conditioning (equilibrated rcond, via the
    shared census instrument);
  * degrees 6 / 8 / 12, and the thread / build spread.

(v) THE FD BASIS.  ``basis='fd'`` puts every layer on ONE uniform radial grid
``_fd_grid(Rbig, N)`` that does not know where the walls are, so no cell is
ever manufactured by a wall coincidence.  The FD arm of every rung is run to
show that -- it is the control, and it doubles as the oracle.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import closure, dump, inv_census, pin_tree  # noqa: E402

print("TREE", pin_tree())

# --- the fixture, in two unit systems --------------------------------------
#  A: the library's native dimensionless system (Rbig = 24, k0 = 2)
#  B: nanometres (wl = 1550 nm, Rbig = 20 um) -- so the delta ladder can be
#     read as an absolute manufacturing tolerance as well as a fraction.
UNITS = {
    "native": dict(Rbig=24.0, k0=2.0, wall=6.0, wall2=9.0,
                   n_hi=2.45, n_lo=1.41, thick=(0.4, 0.5, 0.4)),
    "nm": dict(Rbig=20000.0, k0=2 * np.pi / 1550.0, wall=5000.0, wall2=7500.0,
               n_hi=2.45, n_lo=1.41, thick=(300.0, 400.0, 300.0)),
}


def build(u, delta, *, basis, degree, m=1, N=200):
    """Two ring-walled layers whose OUTER walls differ by ``delta``."""
    from lumenairy import BORStack
    R, k0 = u["Rbig"], u["k0"]
    ehi, elo = u["n_hi"] ** 2, u["n_lo"] ** 2
    t0, t1, t2 = u["thick"]
    s = BORStack(R, m, n_substrate=u["n_lo"], n_superstrate=u["n_lo"],
                 N=N, basis=basis, degree=degree)
    s.add_layer(t0, eps=elo)
    s.add_layer(t1, segments=[(u["wall"], ehi), (u["wall2"], elo), (R, elo)])
    s.add_layer(t2, segments=[(u["wall"] + delta, ehi),
                              (u["wall2"], elo), (R, elo)])
    s.set_source(k0=k0)
    return s


def modal_extremes(s):
    """max |q| over every layer basis, and the narrowest element built."""
    d = s._last
    qmax = 0.0
    wmin = np.inf
    ne = 0
    for _tag, L in ([("sup", d["sup"]), ("sub", d["sub"])]
                    + [("m", LL) for _t, LL in d["mids"]]):
        qmax = max(qmax, float(np.max(np.abs(np.asarray(L["q"])))))
        mesh = L.get("mesh")
        if mesh is not None:
            w = np.diff(mesh.b)
            wmin = min(wmin, float(w.min()))
            ne = max(ne, int(mesh.ne))
    return qmax, (None if not np.isfinite(wmin) else wmin), ne


def one(u, delta, *, basis, degree, m=1, census=False):
    recs = []
    s = build(u, delta, basis=basis, degree=degree, m=m)
    ctx = inv_census(recs) if census else _null()
    with ctx, warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = s.solve()
    qmax, wmin, ne = modal_extremes(s)
    R = np.asarray(res["R"], float)
    T = np.asarray(res["T"], float)
    rc = [r["rcond"] for r in recs
          if r["rcond"] is not None and np.isfinite(r["rcond"])]
    return dict(
        delta=float(delta), basis=basis, degree=degree, m=m,
        n_orders=int(R.size), R=R.tolist()[:12], T=T.tolist()[:12],
        sumR=float(R.sum()), sumT=float(T.sum()),
        closure=closure(res),
        superunity=float(np.max(np.asarray(res["energy"])) - 1.0)
        if R.size else None,
        qmax=qmax, qmax_over_ceiling=qmax / (u["n_hi"] * u["k0"]),
        min_element=wmin, n_elements=ne,
        rcond_min=float(min(rc)) if rc else None,
        n_inverses=len(recs),
        warnings=[str(x.message)[:140] for x in w][:2])


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def err_vs(ref, row, key="R"):
    a = np.asarray(ref[key], float)
    b = np.asarray(row[key], float)
    n = min(a.size, b.size)
    if n == 0:
        return None
    return float(np.max(np.abs(a[:n] - b[:n])))


def ladder(uname, *, degrees=(6, 8, 12), m=1):
    u = UNITS[uname]
    fracs = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6, 1e-7]
    out = []
    for degree in degrees:
        ref = one(u, 0.0, basis="sem", degree=degree, m=m)
        ref_fd = one(u, 0.0, basis="fd", degree=degree, m=m)
        rows = []
        for f in fracs:
            delta = f * u["Rbig"]
            sem = one(u, delta, basis="sem", degree=degree, m=m, census=True)
            fd = one(u, delta, basis="fd", degree=degree, m=m)
            rows.append(dict(
                frac=f, delta_abs=float(delta), sem=sem, fd=fd,
                errR_sem_vs_0=err_vs(ref, sem, "R"),
                errR_fd_vs_0=err_vs(ref_fd, fd, "R"),
                errR_sem_vs_fd=err_vs(fd, sem, "R"),
                errR_sem_vs_fd_at0=err_vs(ref_fd, ref, "R")))
        out.append(dict(units=uname, degree=degree, m=m,
                        ref_sem=ref, ref_fd=ref_fd, rows=rows))
    return out


def main():
    tag = os.environ.get("PROBE_TAG", "win")
    thr = os.environ.get("OPENBLAS_NUM_THREADS", "?")
    payload = dict(threads=thr, ladders=[])
    for uname in ("native", "nm"):
        payload["ladders"].extend(ladder(uname))
    dump(f"c1_union_sliver_{tag}_t{thr}.json", payload)

    for L in payload["ladders"]:
        u = UNITS[L["units"]]
        print(f"\n== UNION SLIVER  units={L['units']}  degree={L['degree']}  "
              f"m={L['m']}  (SEM vs FD gap at delta=0: "
              f"{L['rows'][0]['errR_sem_vs_fd_at0']:.3e}) ==")
        print("  frac      delta        min_elem     |q|max/ceil   closure"
              "      R+T-1        errR vs d=0   errR vs FD   move/delta  "
              "rcond_min  warn")
        for r in L["rows"]:
            s = r["sem"]
            me = s["min_element"]
            phys = r["errR_fd_vs_0"] or 0.0
            move = (r["errR_sem_vs_0"] / phys) if phys > 0 else float("inf")
            print(f"  {r['frac']:.0e}  {r['delta_abs']:.4e}  "
                  f"{me:.4e}  {s['qmax_over_ceiling']:.4e}  "
                  f"{s['closure']:.3e}  "
                  f"{(s['superunity'] if s['superunity'] is not None else float('nan')):+.3e}  "
                  f"{r['errR_sem_vs_0']:.4e}  {r['errR_sem_vs_fd']:.4e}  "
                  f"{move:10.2f}  "
                  f"{(s['rcond_min'] if s['rcond_min'] is not None else float('nan')):.2e}  "
                  f"{len(s['warnings'])}")
        # the FD control
        print("  FD control (no union grid): errR vs d=0 per rung:",
              " ".join(f"{r['errR_fd_vs_0']:.2e}" for r in L["rows"]))

    print("\n== HEADLINE ==")
    allrows = [(L, r) for L in payload["ladders"] for r in L["rows"]]
    su = [r["sem"]["superunity"] for _L, r in allrows
          if r["sem"]["superunity"] is not None]
    print(f"  worst SEM super-unity (R+T-1) over the ladder: {max(su):+.4e}"
          f"   [1-D guard bars: trigger 1e-3, refuse 1e-2]")
    print(f"  worst SEM closure: "
          f"{max(r['sem']['closure'] for _L, r in allrows):.4e}")
    print(f"  worst |q|max / physical ceiling: "
          f"{max(r['sem']['qmax_over_ceiling'] for _L, r in allrows):.4e}")
    print(f"  worst SEM-vs-FD per-order R disagreement: "
          f"{max(r['errR_sem_vs_fd'] for _L, r in allrows):.4e}")
    print(f"  worst interface rcond: "
          f"{min(r['sem']['rcond_min'] for _L, r in allrows if r['sem']['rcond_min']):.4e}")
    nw = sum(len(r["sem"]["warnings"]) for _L, r in allrows)
    print(f"  UserWarnings emitted by the SEM path over the whole ladder: {nw}")


if __name__ == "__main__":
    main()

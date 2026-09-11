"""CLASS C on BOR SEM -- (ii) the WITHIN-LAYER thin annular liner, and
(iv) whether the damage is ENERGY-INVISIBLE.

(ii) THE FIXTURE, AND WHY IT IS A CLEAN ONE.  A thin annulus of width ``w``
sits INSIDE one layer's own segment list, so the narrow element is built by
that layer's own walls and not by the neighbour-enrichment window -- the peer
of the 2-D staggered PMM's "narrow segment INSIDE one grid" defect
(``_STAG_MIN_SEG_FRAC``), not of the 1-D union-grid sliver.

The liner's permittivity is set EQUAL to the material it sits in, so THE
DEVICE IS INDEPENDENT OF ``w`` BY CONSTRUCTION -- exactly the device the 2-D
round-3 measurement used ("a y-uniform 3-layer stack whose MIDDLE layer is ALL
HOST, so the DEVICE cannot depend on the wall separation at all and every
deviation is numerical damage").  Every difference from the ``w``-free
reference is therefore numerical, with no physics to subtract.

(iv) ENERGY-INVISIBLE?  The 1-D union-grid sliver announces itself in the
closure (correct rows reach 4.13e-06, wrong rows start at +1.159).  The 2-D
mortar's does NOT -- it is "ENERGY-INVISIBLE and a FLOOR under the modal-count
ladder".  Both are measured here: the closure AND the per-order error against
the ``w``-free reference are reported at every rung, and a DEGREE ladder
(6 / 8 / 12 / 16) is run at each width so a floor can be seen as the
degree arm converges past it.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import closure, dump, inv_census, pin_tree  # noqa: E402

print("TREE", pin_tree())

RBIG = 24.0
K0 = 2.0
E_HI = 2.45 ** 2
E_LO = 1.41 ** 2
WALL = 6.0
M = 1


def build(width, *, degree, basis="sem", N=200):
    """Layer 1 carries a liner of width ``width`` at r = WALL whose eps EQUALS
    the host it replaces -- so the device does not depend on ``width``."""
    from lumenairy import BORStack
    s = BORStack(RBIG, M, n_substrate=1.41, n_superstrate=1.41, N=N,
                 basis=basis, degree=degree)
    s.add_layer(0.4, eps=E_LO)
    if width <= 0.0:
        segs = [(WALL, E_HI), (RBIG, E_LO)]
    else:
        # the liner replaces the OUTER edge of the high-index core with an
        # element of width `width` of the SAME material
        segs = [(WALL - width, E_HI), (WALL, E_HI), (RBIG, E_LO)]
    s.add_layer(0.5, segments=segs)
    s.add_layer(0.4, eps=E_LO)
    s.set_source(k0=K0)
    return s


def one(width, *, degree, basis="sem", census=False):
    recs = []
    s = build(width, degree=degree, basis=basis)
    cm = inv_census(recs) if census else _null()
    with cm, warnings.catch_warnings(record=True) as w:
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
    rc = [r["rcond"] for r in recs
          if r["rcond"] is not None and np.isfinite(r["rcond"])]
    R = np.asarray(res["R"], float)
    return dict(width=float(width), degree=degree, basis=basis,
                n_orders=int(R.size), R=R.tolist(),
                q=np.asarray(res["q"], float).tolist(),
                T=np.asarray(res["T"], float).tolist(),
                closure=closure(res),
                superunity=float(np.max(np.asarray(res["energy"])) - 1.0)
                if R.size else None,
                qmax=qmax, qmax_over_ceiling=qmax / (2.45 * K0),
                min_element=None if not np.isfinite(wmin) else wmin,
                rcond_min=float(min(rc)) if rc else None,
                warnings=[str(x.message)[:140] for x in w][:2])


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def errR(ref, row):
    """Per-order R difference, orders MATCHED BY q (the modal ordering
    permutes when the mesh changes -- an index-wise comparison would be
    comparing different physical channels)."""
    from c4_sliver_attrib import q_matched
    d, _resid, _n = q_matched(ref, row, "R")
    return d


def main():
    tag = os.environ.get("PROBE_TAG", "win")
    thr = os.environ.get("OPENBLAS_NUM_THREADS", "?")
    # widths as a fraction of Rbig AND of the local wavelength in the core
    lam_core = 2 * np.pi / (K0 * 2.45)
    # the liner lives INSIDE the core (0 < w < WALL), so the ladder starts
    # at 1e-1 of Rbig (w = 2.4, well inside WALL = 6.0)
    fracs = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6, 1e-7]
    payload = dict(threads=thr, lam_core=float(lam_core), ladders=[])
    for degree in (6, 8, 12, 16):
        ref = one(0.0, degree=degree)
        rows = []
        for f in fracs:
            w = f * RBIG
            r = one(w, degree=degree, census=True)
            r["frac_Rbig"] = f
            r["frac_lam_core"] = float(w / lam_core)
            r["errR"] = errR(ref, r)
            rows.append(r)
        payload["ladders"].append(dict(degree=degree, ref=ref, rows=rows))
        print(f"\n== WITHIN-LAYER LINER (device is width-INDEPENDENT)  "
              f"degree={degree}  ref closure={ref['closure']:.3e} ==")
        print("  w/Rbig    w (abs)     w/lam_core   min_elem    closure     "
              "R+T-1        errR vs w-free   |q|max/ceil  rcond_min   warn")
        for r in rows:
            print(f"  {r['frac_Rbig']:.0e}  {r['width']:.4e}  "
                  f"{r['frac_lam_core']:.4e}   {r['min_element']:.3e}  "
                  f"{r['closure']:.3e}  "
                  f"{(r['superunity'] if r['superunity'] is not None else float('nan')):+.3e}  "
                  f"{r['errR']:.6e}     {r['qmax_over_ceiling']:.3e}  "
                  f"{(r['rcond_min'] if r['rcond_min'] is not None else float('nan')):.2e}  "
                  f"{len(r['warnings'])}")
    dump(f"c2_within_layer_{tag}_t{thr}.json", payload)

    print("\n== (iv) IS THE DAMAGE ENERGY-INVISIBLE? ==")
    print("  degree   worst closure   worst |R+T-1|   worst errR   "
          "errR/closure ratio")
    for L in payload["ladders"]:
        cl = max(r["closure"] for r in L["rows"])
        su = max(abs(r["superunity"]) for r in L["rows"]
                 if r["superunity"] is not None)
        er = max(r["errR"] for r in L["rows"])
        print(f"  {L['degree']:>5}   {cl:.4e}      {su:.4e}     {er:.4e}   "
              f"{er / cl:10.2f}")
    print("\n== THE DEGREE LADDER AT EACH WIDTH (a FLOOR shows as a row that "
          "stops improving) ==")
    print("  w/Rbig " + "".join(f"   deg{L['degree']:<8}"
                                for L in payload["ladders"]))
    for i, f in enumerate(fracs):
        print(f"  {f:.0e}" + "".join(
            f"   {L['rows'][i]['errR']:.3e}" for L in payload["ladders"]))
    print("  w-free " + "".join(f"   {'(ref)':<11}"
                                for L in payload["ladders"]))
    nw = sum(len(r["warnings"]) for L in payload["ladders"]
             for r in L["rows"])
    print(f"\n  UserWarnings over the whole ladder: {nw}")


if __name__ == "__main__":
    main()

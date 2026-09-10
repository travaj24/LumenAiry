"""CLASS B on BOR -- the explicit-inverse census.

EVERY explicit inverse or linear solve the BOR cascade performs is UNGUARDED
today.  The sites (each census row names the file:line it came from):

  zcascade.interface_smatrix   solve(Wb, Wa) ; solve(Vb, Va) ; inv(a + b)
  zcascade.redheffer_star      inv(I - B11 A22) ; inv(I - A22 B11)
  zcascade.layer_modes         inv(Lm + k0^2 eps)          [NODAL basis only]
  coupled_radial_eigensolver._assemble_staggered           [Lei]
  sem_radial.sem_layer_modes   inv(Mz)                     [E_z elimination]
  sem_radial.sem_interface_smatrix
                               solve(M1b, G1) ; solve(M0b, G0) ;
                               solve(M1a, G1^T) ; solve(M0a, G0^T) ;
                               solve(Wb, Wa_b) = alpha ; solve(Va, Vb_a)
                               = gamma ; inv(I + gamma alpha)
  bor_stack.layer_absorption   solve(I - S22_above S11_below, ...)

The Cartesian guard is a CONJUNCTION -- equilibrated rcond BELOW a bar AND the
equilibrated inverse residual ABOVE ``_INV_RESID_REFUSE`` -- and it is armed
at exactly ONE site, because only there did the population separate two-sided.
So the question here is NOT "are these inverses ill-conditioned" but "is there
a HEALTHY population and a BROKEN population, and do they separate".

The battery is built to contain every candidate BROKEN population: coincident
uniform layers, a 1e-6 detune, exactly-degenerate twin layers, PML-adjacent
and very thin layers, near-cutoff (the cylindrical 'grazing'), high ``m``,
many rings, SEM degrees 6 / 8 / 12 / 16, and a lossy metal ring.  Each row
carries the solve's own closure so a broken inverse can be tied to a wrong
answer.
"""
from __future__ import annotations

import collections
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import closure, dump, inv_census, pin_tree  # noqa: E402

print("TREE", pin_tree())
RBIG = 24.0
LAM = 3.0
E141 = 1.41 ** 2


def fixtures():
    from lumenairy import BORStack
    out = []

    def mk(idd, fam, m, basis, k0, layers, *, degree=8, N=120,
           n_sup=1.41, n_sub=1.41, eps_seg=1):
        def build():
            s = BORStack(RBIG, m, n_substrate=n_sub, n_superstrate=n_sup,
                         N=N, basis=basis, degree=degree,
                         elements_per_segment=eps_seg)
            for t, kind, val in layers:
                if kind == "eps":
                    s.add_layer(t, eps=val)
                elif kind == "rings":
                    s.add_layer(t, rings=val)
                else:
                    s.add_layer(t, segments=val)
            s.set_source(k0=k0)
            return s
        out.append((idd, fam, build))

    ring = (LAM, 0.5, 2.45, 1.41)
    for basis in ("fd", "sem"):
        for m in (0, 1, 2, 5, 10):
            mk(f"healthy_m{m}_{basis}", "healthy", m, basis, 2.0,
               [(0.5, "rings", ring)])
            mk(f"coincide_m{m}_{basis}", "coincident", m, basis, 2.0,
               [(0.4, "eps", E141), (0.5, "rings", ring), (0.4, "eps", E141)])
            mk(f"detune_m{m}_{basis}", "detune", m, basis, 2.0,
               [(0.4, "eps", E141 * (1 + 1e-6)), (0.5, "rings", ring),
                (0.4, "eps", E141 * (1 + 1e-6))])
            mk(f"twin_m{m}_{basis}", "twin_uniform", m, basis, 2.0,
               [(0.4, "eps", E141), (0.4, "eps", E141)])
        for t in (1e-3, 1e-6, 1e-9, 1e-12):
            mk(f"thin{t:.0e}_{basis}", "thin_layer", 1, basis, 2.0,
               [(0.4, "eps", E141), (t, "rings", ring), (0.4, "eps", E141)])
        for k0 in (0.12169, 0.15773, 0.19825):
            mk(f"cutoff_k{k0}_{basis}", "near_cutoff", 1, basis, float(k0),
               [(0.4, "eps", E141), (0.5, "rings", ring), (0.4, "eps", E141)])
        for per in (LAM, 1.0, 0.6):
            mk(f"rings_p{per}_{basis}", "many_rings", 1, basis, 2.0,
               [(0.4, "eps", E141), (0.5, "rings", (per, 0.5, 2.45, 1.41)),
                (0.4, "eps", E141)])
    for deg in (6, 8, 12, 16):
        segs = [(2.0, 6.0), (3.0, 2.25), (RBIG, E141)]
        mk(f"deg{deg}_sem", "degree", 1, "sem", 2.0,
           [(0.4, "eps", E141), (0.5, "segments", segs), (0.4, "eps", E141)],
           degree=deg)
        mk(f"deg{deg}_seg3_sem", "degree_eps_seg", 1, "sem", 2.0,
           [(0.4, "eps", E141), (0.5, "segments", segs), (0.4, "eps", E141)],
           degree=deg, eps_seg=3)
    mk("metal_sem", "lossy_layer", 1, "sem", 2.0,
       [(0.4, "eps", E141),
        (0.15, "segments", [(2.0, complex(-20.0, 2.0)), (RBIG, E141)]),
        (0.4, "eps", E141)])
    mk("metal_fd", "lossy_layer", 1, "fd", 2.0,
       [(0.4, "eps", E141), (0.15, "rings", ring), (0.4, "eps", E141)])
    return out


def run():
    rows = []
    for idd, fam, build in fixtures():
        recs = []
        try:
            s = build()
            with inv_census(recs), warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                res = s.solve()
            cl = closure(res)
            sumRT = float(np.sum(res["R"]) + np.sum(res["T"]))
            n_ord = int(np.size(res["R"]))
            over = (float(np.max(np.asarray(res["energy"])) - 1.0)
                    if n_ord else None)
            msgs = [str(x.message)[:120] for x in w][:2]
            err = None
        except Exception as exc:                   # noqa: BLE001
            cl = sumRT = over = None
            n_ord = 0
            msgs, err = [], f"{type(exc).__name__}: {exc}"[:200]
        common = dict(fixture=idd, family=fam, closure=cl, superunity=over,
                      n_orders=n_ord, sumRT=sumRT, err=err, warn=msgs)
        for r in recs:
            rows.append(dict(**common, **r))
        if not recs:
            rows.append(dict(**common, site="(none)", kind="-", n=0,
                             rcond=None, resid=None))
        print(f"  {idd:26s} inv/solve={len(recs):4d} closure={cl} "
              f"superunity={over} {err or ''}")
    return rows


def summarize(rows):
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("rcond") is None or not np.isfinite(r["rcond"]):
            continue
        by[(r["site"], r["kind"])].append(r)
    print("\n== POPULATION PER SITE (equilibrated rcond / residual) ==")
    print(f"{'site':<58} {'k':<6} {'n':>5} {'rcond min':>11} "
          f"{'rcond max':>11} {'resid max':>11}")
    tab = []
    for k, v in sorted(by.items(), key=lambda x: min(r["rcond"] for r in x[1])):
        rc = [r["rcond"] for r in v]
        rs = [r["resid"] for r in v
              if r["resid"] is not None and np.isfinite(r["resid"])]
        row = dict(site=k[0], kind=k[1], count=len(v),
                   rcond_min=float(min(rc)), rcond_max=float(max(rc)),
                   rcond_med=float(np.median(rc)),
                   resid_max=float(max(rs)) if rs else None,
                   resid_med=float(np.median(rs)) if rs else None,
                   worst_fixture=min(v, key=lambda r: r["rcond"])["fixture"])
        tab.append(row)
        rm = row["resid_max"] if row["resid_max"] is not None else float("nan")
        print(f"{k[0][:56]:<58} {k[1]:<6} {len(v):>5} "
              f"{row['rcond_min']:>11.3e} {row['rcond_max']:>11.3e} "
              f"{rm:>11.3e}")
    return tab


def main():
    tag = os.environ.get("PROBE_TAG", "win")
    rows = run()
    tab = summarize(rows)
    dump(f"b1_invcensus_{tag}.json", dict(rows=rows, table=tab))
    print("\n== WORST rcond BY FAMILY ==")
    fam = collections.defaultdict(list)
    for r in rows:
        if r.get("rcond") is not None and np.isfinite(r["rcond"]):
            fam[r["family"]].append(r)
    for f, v in sorted(fam.items(),
                       key=lambda x: min(r["rcond"] for r in x[1])):
        w = min(v, key=lambda r: r["rcond"])
        su = [r["superunity"] for r in v if r["superunity"] is not None]
        print(f"  {f:<16} rcond_min={w['rcond']:.3e} resid={w['resid']:.3e} "
              f"@ {w['fixture']} [{w['site'][:40]}]  max(R+T-1)="
              f"{max(su) if su else None}")
    allrc = [r["rcond"] for r in rows
             if r.get("rcond") is not None and np.isfinite(r["rcond"])]
    allrs = [r["resid"] for r in rows
             if r.get("resid") is not None and np.isfinite(r["resid"])]
    print(f"\n  WHOLE CENSUS: {len(allrc)} inverses, rcond "
          f"{min(allrc):.3e} .. {max(allrc):.3e}, residual max "
          f"{max(allrs):.3e}")
    print("  Cartesian bars for reference: _INV_T22_RCOND_REFUSE = 1e-10, "
          "_MORTAR_RCOND_REFUSE = 1e-12, _MORTAR_RESID_REFUSE = 1e-6")


if __name__ == "__main__":
    main()

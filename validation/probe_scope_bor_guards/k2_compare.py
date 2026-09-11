"""Compare every ``k1_kernel_*.json`` and report which GUARD DECISIONS move.

A guard whose verdict moves with the BLAS kernel is not a guard.  This reads
every kernel matrix produced by ``k1_kernel_matrix.py`` (each tagged with the
architecture ACTUALLY OBTAINED, not the one requested) and reports, per class:

  A  the per-rung CLASS verdict, the orientation verdicts, and the resulting
     R/T channel count and closure -- for the SHIPPED rule and for the
     CANDIDATE spectrum-scaled band, so the candidate's kernel stability is
     measured and not assumed.
  B  the ``inv(a + b)`` rcond / residual populations and the nodal
     ``max(R + T)``, with the DECISION a ported conjunction guard would take.
  C  per sliver rung, three candidate DECISION quantities side by side --
     the GEOMETRY (narrowest element as a fraction of Rbig), the spurious
     ``|q|max`` against the physical ceiling, and the ENERGY VIOLATION -- so
     the CI's finding (the 1-D super-unity trigger came out kernel-dependent)
     can be checked here before a bar is proposed on any of them.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.dirname(os.path.abspath(__file__))


def load():
    out = {}
    for p in sorted(glob.glob(os.path.join(D, "k1_kernel_*.json"))):
        j = json.load(open(p, encoding="cp1252", errors="replace"))
        base = os.path.basename(p)[len("k1_kernel_"):-len(".json")]
        out[base] = j
    return out


def spread(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if len(v) < 2:
        return None
    lo, hi = min(v), max(v)
    return dict(lo=lo, hi=hi, ratio=(hi / lo if lo > 0 else float("inf")),
                absdiff=hi - lo)


def main():
    runs = load()
    print(f"{len(runs)} kernel runs loaded\n")
    print(f"{'run':<34} {'requested':<10} {'obtained':<12} build")
    for k, j in runs.items():
        print(f"{k:<34} {str(j['requested_kernel']):<10} "
              f"{str(j['architecture']):<12} "
              f"{j['_build']['platform']}/{j['_build']['python']}")

    keys = list(runs)
    print("\n" + "=" * 78)
    print("A -- ORIENTATION DECISIONS PER KERNEL")
    print("=" * 78)
    idx = {(r["m"], round(r["delta"], 15)) for k in keys
           for r in runs[k]["A"]}
    hdr = f"{'m':>2} {'delta':>10} {'qn':>10} |"
    for k in keys:
        hdr += f" {runs[k]['architecture'][:8]:>8}"
    print(hdr + "   << shipped CLASS (P=prop, E=evan)")
    nmove_class = nmove_orient = nmove_nord = 0
    nmove_class_c = 0
    rows_moved = []
    for key in sorted(idx):
        cells, cls, orient, nord, ncls, cclo, sclo = [], [], [], [], [], [], []
        qn = None
        for k in keys:
            r = next((x for x in runs[k]["A"]
                      if (x["m"], round(x["delta"], 15)) == key), None)
            if r is None:
                cells.append("  --")
                continue
            qn = r["qn"]
            cls.append(r["shipped_called_prop"])
            ncls.append(r["cand_called_prop"])
            orient.append((r["flux_fwd"], r["im_fwd"]))
            nord.append((r["shipped_n_orders"], r["cand_n_orders"]))
            sclo.append(r["shipped_closure"])
            cclo.append(r["cand_closure"])
            cells.append(
                f"{'P' if r['shipped_called_prop'] else 'E'}"
                f"{'/' + str(r['shipped_n_orders'])}"
                f"{'b' if r['backward_flux_shipped'] else ' '}".rjust(8))
        moved = len(set(cls)) > 1
        movedn = len(set(nord)) > 1
        if moved:
            nmove_class += 1
        if len(set(orient)) > 1:
            nmove_orient += 1
        if movedn:
            nmove_nord += 1
        if len(set(ncls)) > 1:
            nmove_class_c += 1
        flag = " <== MOVES" if (moved or movedn) else ""
        line = f"{key[0]:>2} {key[1]:>10.2e} {qn:>10.3e} |" + "".join(cells)
        print(line + flag)
        if moved or movedn:
            rows_moved.append((key, sclo, cclo, nord))
    n = len(idx)
    print(f"\n  rungs: {n}")
    print(f"  SHIPPED rule -- rungs whose CLASS verdict moves with the "
          f"kernel: {nmove_class}/{n}")
    print(f"  SHIPPED rule -- rungs whose CHANNEL COUNT moves: "
          f"{nmove_nord}/{n}")
    print(f"  SHIPPED rule -- rungs whose orientation verdicts move: "
          f"{nmove_orient}/{n}")
    print(f"  CANDIDATE band -- rungs whose CLASS verdict moves: "
          f"{nmove_class_c}/{n}")
    if rows_moved:
        print("\n  worst observable spread on a moving rung:")
        for key, sclo, cclo, nord in rows_moved[:6]:
            ss, cs = spread(sclo), spread(cclo)
            print(f"    m={key[0]} delta={key[1]:.2e}  shipped closure "
                  f"{ss['lo']:.3e}..{ss['hi']:.3e} ({ss['ratio']:.1f}x)  "
                  f"candidate {cs['lo']:.3e}..{cs['hi']:.3e} "
                  f"({cs['ratio']:.1f}x)  n_orders {sorted(set(nord))}")
    allS = [r["shipped_closure"] for k in keys for r in runs[k]["A"]]
    allC = [r["cand_closure"] for k in keys for r in runs[k]["A"]]
    print(f"\n  worst closure over ALL kernels: shipped {max(allS):.4e}   "
          f"candidate {max(allC):.4e}   ({max(allS) / max(allC):.0f}x)")

    print("\n" + "=" * 78)
    print("B -- inv(a+b) POPULATIONS AND THE REFUSAL DECISION PER KERNEL")
    print("=" * 78)
    print(f"{'Rbig/lam':>9} {'basis':<10} |" +
          "".join(f" {runs[k]['architecture'][:8]:>22}" for k in keys))
    bidx = sorted({(r["rbig_lambda"], r["basis"]) for k in keys
                   for r in runs[k]["B"]})
    for key in bidx:
        cells, energies, rconds, resids = [], [], [], []
        for k in keys:
            r = next((x for x in runs[k]["B"]
                      if (x["rbig_lambda"], x["basis"]) == key), None)
            if r is None or r.get("error"):
                cells.append(f"{'ERR':>22}")
                continue
            energies.append(r["max_energy"])
            rconds.append(r["apb_rcond_min"])
            resids.append(r["apb_resid_max"])
            cells.append(f"{r['max_energy']:.4g}/{r['apb_rcond_min']:.1e}"
                         .rjust(22))
        print(f"{key[0]:>9.1f} {key[1]:<10} |" + "".join(cells))
    for nm, get in (("max(R+T)", "max_energy"),
                    ("rcond(a+b)", "apb_rcond_min"),
                    ("residual", "apb_resid_max")):
        for basis in ("staggered", "nodal"):
            v = [r[get] for k in keys for r in runs[k]["B"]
                 if r["basis"] == basis and r.get(get) is not None]
            if v:
                print(f"  {basis:<10} {nm:<12} over all kernels: "
                      f"{min(v):.4e} .. {max(v):.4e}")
    # the DECISION a ported conjunction would take, per kernel
    print("\n  Would the Cartesian conjunction (rcond < 1e-10 AND resid > "
          "1e-8) refuse ANY row?")
    for k in keys:
        ref = [r for r in runs[k]["B"]
               if r.get("apb_rcond_min") is not None
               and r["apb_rcond_min"] < 1e-10
               and (r.get("apb_resid_max") or 0.0) > 1e-8]
        print(f"    {runs[k]['architecture']:<10} ({runs[k]['requested_kernel']}): "
              f"{len(ref)} of {len(runs[k]['B'])} rows")

    print("\n" + "=" * 78)
    print("C -- SLIVER DECISION QUANTITIES PER KERNEL")
    print("=" * 78)
    cidx = sorted({(r["degree"], r["separated"], r["frac"]) for k in keys
                   for r in runs[k]["C"]})
    print(f"{'deg':>3} {'sep':>3} {'frac':>8} | "
          f"{'min/Rbig spread':>22} {'|q|/ceil spread':>22} "
          f"{'closure spread':>22} {'dR spread':>22}")
    worst = dict(geom=1.0, q=1.0, clo=1.0, dR=1.0)
    for key in cidx:
        g, qq, cc, dd = [], [], [], []
        for k in keys:
            r = next((x for x in runs[k]["C"]
                      if (x["degree"], x["separated"], x["frac"]) == key),
                     None)
            if r is None:
                continue
            g.append(r["min_elem_frac_Rbig"])
            qq.append(r["qmax_over_ceiling"])
            cc.append(r["closure"])
            dd.append(r["dR_matched"])
        sg, sq, sc, sd = spread(g), spread(qq), spread(cc), spread(dd)

        def fmt(s):
            return (f"{s['lo']:.3e}({s['ratio']:.2f}x)" if s else "-").rjust(22)
        for nm, s in (("geom", sg), ("q", sq), ("clo", sc), ("dR", sd)):
            if s and s["ratio"] > worst[nm]:
                worst[nm] = s["ratio"]
        print(f"{key[0]:>3} {str(key[1])[0]:>3} {key[2]:>8.0e} | "
              f"{fmt(sg)} {fmt(sq)} {fmt(sc)} {fmt(sd)}")
    print("\n  WORST kernel spread of each candidate DECISION quantity:")
    print(f"    narrowest element / Rbig (GEOMETRY)   {worst['geom']:.4f}x")
    print(f"    |q|max / physical ceiling             {worst['q']:.4f}x")
    print(f"    q-matched per-order dR                {worst['dR']:.4f}x")
    print(f"    closure / super-unity (ENERGY)        {worst['clo']:.4f}x"
          "   <== the CI's kernel-dependent quantity")


if __name__ == "__main__":
    main()

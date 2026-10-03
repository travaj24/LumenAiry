"""V2 analysis: what does the twin differentiate, and does it converge to the
TRUE gradient?  Reads v2_frozen_<fam>_M<M>_*_<build>.json.

    python v2_analyse.py [build]

Per family and quantity (R00, T00 E_x, T00 E_y):
  truth          = FD(numpy) at the highest M available (Richardson),
  e_AD(M)        = |AD_M - truth|          (the twin's gradient error),
  e_FDnp(M)      = |FD(numpy)_M - truth|   (the NumPy solver's own gradient
                                            discretisation error),
  offset(M)      = |AD_M - FD(numpy)_M|    (the frozen-grid offset),
  e_val(M)       = |T_M - T_truth|         (the value discretisation error),
  delta_far(M)   = |twin(x0 -+ 0.025 P) - numpy(x0 -+ 0.025 P)| (the frozen
                    grid's re-parametrisation away from the reference).
All relative to |truth| of the quantity (max over the three).
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = sys.argv[1] if len(sys.argv) > 1 else "win"
fams = {}
pt_files = {}
for p in sorted(glob.glob(os.path.join(HERE, f"v2_frozen_*_M*_*_{B}.json"))):
    if any(t in os.path.basename(p) for t in ("_mut", "_fix")):
        continue                 # mutant / fix-tree runs are not the clean set
    d = json.load(open(p))
    if "_pt" in os.path.basename(p):
        # one NumPy point of a split run: merge into np_at
        pt_files.setdefault((d["fam"], d["M"]), []).append(d)
        continue
    if d["M"] <= 3 and d["fam"] != "sinex":
        continue
    fams.setdefault(d["fam"], {}).setdefault(d["M"], {}).update(
        {k: v for k, v in d.items() if k not in ("env",)})
for (fam, M), lst in pt_files.items():
    d = fams.setdefault(fam, {}).setdefault(M, {"fam": fam, "M": M,
                                                "x0": lst[0]["x0"]})
    at = dict(d.get("np_at", {}))
    for e in lst:
        at.update(e["np_at"])
    d["np_at"] = at
    steps = lst[0]["steps"]
    need = [repr(round(d["x0"] + s * k * 1.2, 12)) for s in steps
            for k in (-1, 1)] + [repr(d["x0"])]
    if all(k in at for k in need):
        rows = []
        for s in steps:
            h = s * 1.2
            rows.append((np.asarray(at[repr(round(d["x0"] + h, 12))])
                         - np.asarray(at[repr(round(d["x0"] - h, 12))]))
                        / (2 * h))
        rows = np.asarray(rows)
        r = steps[-2] / steps[-1]
        d["FD_numpy"] = ((r * r * rows[-1] - rows[-2]) / (r * r - 1)).tolist()
        ch = np.abs(np.diff(rows, axis=0))
        d["FD_numpy_ratios"] = (ch[:-1] / ch[1:]).tolist() if len(ch) > 1             else None
        d["np_value"] = at[repr(d["x0"])]
out = {}
for fam, byM in sorted(fams.items()):
    Ms = sorted(m for m in byM if "FD_numpy" in byM[m])
    if not Ms:
        continue
    Mt = Ms[-1]
    truth = np.asarray(byM[Mt]["FD_numpy"])
    vtruth = np.asarray(byM[Mt]["np_value"])
    sc = np.abs(truth)
    rows = {}
    for M in sorted(byM):
        d = byM[M]
        r = {}
        if "FD_numpy" in d:
            fdn = np.asarray(d["FD_numpy"])
            r["e_FDnp"] = (np.abs(fdn - truth) / sc).tolist()
            r["e_val"] = np.abs(np.asarray(d["np_value"]) - vtruth).tolist()
            if d.get("FD_numpy_ratios") is not None:
                r["np_ratios"] = np.asarray(d["FD_numpy_ratios"]).ravel(
                ).round(3).tolist()
        if "AD" in d:
            ad = np.asarray(d["AD"])
            r["e_AD"] = (np.abs(ad - truth) / sc).tolist()
            r["AD_vs_FDtwin"] = (np.abs(ad - np.asarray(d["FD_twin"]))
                                 / sc).tolist()
            if "FD_numpy" in d:
                r["offset"] = (np.abs(ad - np.asarray(d["FD_numpy"]))
                               / sc).tolist()
            if "np_at" in d:
                far = [k for k in d["tw_at"] if abs(float(k) - d["x0"])
                       > 0.02]
                r["delta_far"] = [float(np.max(np.abs(
                    np.asarray(d["tw_at"][k]) - np.asarray(d["np_at"][k]))))
                    for k in far]
                r["delta_at_x0"] = float(np.max(np.abs(
                    np.asarray(d["tw_at"][repr(d["x0"])])
                    - np.asarray(d["np_at"][repr(d["x0"])]))))
            r["tw_grad_compile_s"] = d.get("tw_grad_compile_s")
        rows[M] = r
    out[fam] = {"truth_M": Mt, "truth": truth.tolist(), "rows": rows}
    print(f"== {fam} (truth = FD(numpy) at M = {Mt}: {truth.round(6)})")
    print("  M   e_val(max)  e_FDnp(max)  e_AD(max)  offset(max)  "
          "AD-FDtwin  delta_far")
    for M, r in sorted(rows.items()):
        def mx(k):
            return f"{max(r[k]):.2e}" if k in r else "    --  "
        df = (f"{max(r['delta_far']):.1e}" if r.get("delta_far") else "--")
        print(f"  {M}   {mx('e_val')}    {mx('e_FDnp')}     {mx('e_AD')}"
              f"   {mx('offset')}     {mx('AD_vs_FDtwin')}  {df}")
with open(os.path.join(HERE, f"v2_analysis_{B}.json"), "w") as f:
    json.dump(out, f, indent=1)

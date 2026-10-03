"""Summarise f3_<case>_M<M>.json (and wsl/ copies) into f3_summary.json."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for d in ("", "wsl"):
    for fn in sorted(glob.glob(os.path.join(HERE, d, "f3_*_M*.json"))):
        r = json.load(open(fn))
        ch = np.asarray(r["FD_twin_rung_changes"])
        rows.append(dict(build="wsl" if d else "win", case=r["case"], M=r["M"],
                         AD=r["AD"], FD_twin=r["FD_twin_richardson"],
                         rel_twin=max(r["AD_vs_FD_twin_rel"]),
                         rel_numpy=(max(r["AD_vs_FD_numpy_rel"]) if "AD_vs_FD_numpy_rel" in r else None),
                         rung_ratios=(ch[:-1] / ch[1:]).round(1).tolist(),
                         grad_compile=r["jit_grad_first_s"], grad_repeat=r["jit_grad_repeat_s"]))
json.dump(rows, open(os.path.join(HERE, "f3_summary.json"), "w"), indent=1)
for r in rows:
    print(f"{r['build']} {r['case']:9s} M{r['M']} rel_twin {r['rel_twin']:.1e} rel_np {r['rel_numpy'] if r['rel_numpy'] is None else format(r['rel_numpy'], '.1e')} ratios {r['rung_ratios']} compile {r['grad_compile']:.0f}s rep {r['grad_repeat']:.2f}s AD {np.round(r['AD'], 7).tolist()}")

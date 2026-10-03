"""Markdown rows of the V4 ladders: worst of R/T, Jr, Jt per rung."""
import json

import _ve1common as V  # noqa: F401  (path pin)

d = json.load(open("v4_summary.json"))
order = ["dirgen_none", "dirgen_th3", "dirgen_th2b", "lc_th2b", "dirgen_sh4",
         "dirgen_c3", "lc_c3", "dirgen_c5", "nrgen_none", "nrgen_th3",
         "nrgen_th2b", "nrgen_c3", "nrgen_c5", "gyrol_gyrol_none",
         "gyrol_gyrol_th3", "gyrol_gyrol_th2b", "gyrol_gyrol_sh4",
         "gyrol_gyrol_c3", "gyrol_gyrol_c5", "dirgen_gyro_none",
         "dirgen_gyro_th2b", "dirgen_gyro_c3", "dirgen_gyro_c5",
         "dirgen_syml_th3", "dirgen_none_s+0.17-0.08",
         "dirgen_th3_s+0.17-0.08", "dirgen_sh4_s+0.17-0.08",
         "dirgen_c3_s+0.17-0.08", "gyrol_gyrol_none_s+0.17-0.08",
         "gyrol_gyrol_th3_s+0.17-0.08", "gyrol_gyrol_sh4_s+0.17-0.08",
         "gyrol_gyrol_c3_s+0.17-0.08", "dirgen_gyro_none_s+0.17-0.08",
         "dirgen_gyro_sh4_s+0.17-0.08"]
for k in order + sorted(set(d) - set(order)):
    if k not in d:
        continue
    cells = []
    for mo in ("n", "o", "c"):
        rows = d[k].get(mo, {})
        s = " / ".join("ERR" if v == "ERR" else f"{max(v[:3]):.1e}"
                       for M, v in sorted(rows.items(), key=lambda t:
                                          int(t[0])))
        Ms = sorted(int(m) for m in rows)
        cells.append(f"{s} (M {Ms[0]}..{Ms[-1]})" if Ms else "-")
    print(f"| {k} | " + " | ".join(cells) + " |")

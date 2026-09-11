"""TASK E -- the layer-mode COUNT table, per window, per arm, per build."""
from __future__ import annotations

import collections
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
rows = collections.defaultdict(dict)     # (window, imag) -> tag -> (n, hash)
arms = set()
for p in sorted(HERE.glob("ve_modes_*.json")):
    tag = p.stem[len("ve_modes_"):]
    if tag.endswith("_mine"):
        continue
    d = json.loads(p.read_text(encoding="cp1252"))
    arms.add(tag)
    for r in d["layer_modes"]:
        rows[(r["window"], r["imag"])][tag] = (r["n"], r["hash_q"])
wins = sorted(set(k[0] for k in rows))
print("arms present: %d" % len(arms))
for w in wins:
    print()
    print("WINDOW %s" % w)
    print("  %-26s %-22s %-22s" % ("arm", "PRE  n(im=0/1e-30/1e-12)",
                                   "POST n(im=0/1e-30/1e-12)"))
    ims = sorted(set(k[1] for k in rows if k[0] == w), reverse=True)
    for base in sorted(set(t.split("_", 1)[1] for t in arms)):
        cells = []
        for b in ("pre", "post"):
            t = "%s_%s" % (b, base)
            got = []
            for im in ims:
                v = rows.get((w, im), {}).get(t)
                got.append(str(v[0]) if v else "-")
            cells.append("/".join(got))
        mark = ""
        p_vals = cells[0].split("/")
        q_vals = cells[1].split("/")
        if len(set(p_vals) - {"-"}) > 1:
            mark += "  PRE-MOVES"
        if len(set(q_vals) - {"-"}) > 1:
            mark += "  POST-MOVES"
        print("  %-26s %-22s %-22s%s" % (base, cells[0], cells[1], mark))
    # cross-arm spread
    for b in ("pre", "post"):
        for im in ims:
            vals = {t: v for t, v in rows.get((w, im), {}).items()
                    if t.startswith(b + "_")}
            ns = sorted(set(v[0] for v in vals.values()))
            hs = len(set(v[1] for v in vals.values()))
            if vals:
                print("    %-4s im=%-8g counts across %2d arms: %s   (%d "
                      "distinct mode LISTS)" % (b.upper(), im, len(vals),
                                                ns, hs))

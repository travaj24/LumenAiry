"""Tabulate the verifier's E2-4 closure ladders: closure (both inputs),
successive-rung differences max|R(M)-R(M+1)|, |T(M)-T(M+1)|, n, wall."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
build = sys.argv[1] if len(sys.argv) > 1 else "win"
summ = {}
for pair in ("i_circ_sin", "ii_circ_circ", "iii_fil_sin"):
    rows, prev = [], None
    for M in range(4, 10):
        fn = os.path.join(HERE, f"v3g_e24_closure_{pair}_M{M}_{build}.json")
        if not os.path.exists(fn):
            continue
        d = json.load(open(fn))
        R = np.array(d["R"]) if not isinstance(d["R"], dict) else \
            np.array(d["R"]["re"])
        T = np.array(d["T"]) if not isinstance(d["T"], dict) else \
            np.array(d["T"]["re"])
        succ = None
        if prev is not None and prev[0] == M - 1:
            succ = float(max(np.abs(R - prev[1]).max(),
                             np.abs(T - prev[2]).max()))
        rows.append(dict(M=M, M_layers=d.get("M_layers"),
                         closure=d["closure"], succ_prev=succ, n=d["n"],
                         change=d["change"], xwall=d["xwall"],
                         wall=d["wall"], warnings=d["warnings"]))
        prev = (M, R, T)
    summ[pair] = rows
    for r in rows:
        print(pair, r["M"], r["M_layers"], ["%.1e" % c for c in r["closure"]],
              "succ %.1e" % r["succ_prev"] if r["succ_prev"] else "-",
              "n", r["n"], "wall %.0f" % r["wall"],
              "xwall %.1f" % sum(r["xwall"]), r["warnings"])
json.dump(summ, open(os.path.join(HERE, f"v3g_summary_{build}.json"), "w"),
          indent=1)

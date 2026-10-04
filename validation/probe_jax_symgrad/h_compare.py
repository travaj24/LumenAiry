"""Compare h_fwd_bytes_{pre,r2post}_{win,wsl}.json: every hash must match."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("numpy", "jax_eager", "jax_eager_api", "jax_jit")
bad = 0
for build in ("win", "wsl"):
    try:
        a, b = (json.load(open(os.path.join(
            HERE, f"h_fwd_bytes_{t}_{build}.json"))) for t in ("pre", "r2post"))
    except FileNotFoundError as e:
        print(build, "missing", e)
        continue
    n = nb = 0
    for k, rec in a.items():
        if k == "env":
            continue
        if "error" in rec or "error" in b.get(k, {}):
            print(build, k, "ERROR", rec.get("error"),
                  b.get(k, {}).get("error"))
            nb += 1
            continue
        for kk in KEYS:
            if kk not in rec:
                continue
            n += 1
            if rec[kk] != b[k][kk]:
                nb += 1
                print(build, k, kk, "DIFFERS")
    print(build, f"{n - nb} / {n} hashes equal")
    bad += nb
sys.exit(1 if bad else 0)

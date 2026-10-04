"""Compare g2_fwd_bytes_rcwa_{pre,r2post}_{win,wsl}.json: every hash must
match (per build: equal / total)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
bad_all = 0
for build in ("win", "wsl"):
    try:
        a, b = (json.load(open(os.path.join(
            HERE, f"g2_fwd_bytes_rcwa_{t}_{build}.json")))
            for t in ("pre", "r2post"))
    except FileNotFoundError as e:
        print(build, "missing", e)
        continue
    n = bad = 0
    for k, rec in a.items():
        if k == "env":
            continue
        for kk in ("numpy", "jax_eager", "jax_jit"):
            n += 1
            if rec[kk].startswith("ERROR") or rec[kk] != b[k][kk]:
                bad += 1
                print(build, k, kk, "DIFFERS", rec[kk][:40], b[k][kk][:40])
    bad_all += bad
    print(build, f"{n - bad} / {n} hashes equal")
sys.exit(1 if bad_all else 0)

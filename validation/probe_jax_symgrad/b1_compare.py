"""Compare b1_fwd_bytes_{pre,post}_{win,wsl}.json: every hash must match."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
bad = 0
for build in ("win", "wsl"):
    try:
        a, b = (json.load(open(os.path.join(HERE, f"b1_fwd_bytes_{t}_{build}.json")))
                for t in ("pre", "post"))
    except FileNotFoundError as e:
        print(build, "missing", e)
        continue
    n = 0
    for k, rec in a.items():
        if k == "env":
            continue
        for kk in ("numpy", "jax_eager", "jax_jit"):
            n += 1
            if rec[kk] != b[k][kk]:
                bad += 1
                print(build, k, kk, "DIFFERS")
    print(build, f"{n - bad} / {n} hashes equal")
sys.exit(1 if bad else 0)

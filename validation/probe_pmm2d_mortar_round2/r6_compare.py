"""Compare two ``r6_bitid_<tag>.json`` arms.  ``python r6_compare.py with pre``"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
a, b = (sys.argv[1:3] + ["with", "pre"])[:2]
A = json.load(open(os.path.join(HERE, f"r6_bitid_{a}.json")))
B = json.load(open(os.path.join(HERE, f"r6_bitid_{b}.json")))
keys = sorted(set(A) | set(B))
bad, miss, n = [], [], 0
for k in keys:
    if k not in A or k not in B:
        miss.append(k)
        continue
    for i, (x, y) in enumerate(zip(A[k], B[k])):
        n += 1
        if x != y:
            bad.append(f"{k}[{i}]  {a}={x[:16]}  {b}={y[:16]}")
print(f"{a} vs {b}: {n} hashes over {len(keys) - len(miss)} fixtures, "
      f"{len(bad)} MISMATCH, {len(miss)} missing")
for m in miss:
    print("  MISSING:", m)
for x in bad:
    print("  DIFF:", x)
sys.exit(1 if (bad or miss) else 0)

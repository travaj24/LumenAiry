"""R3 compare: ``python r3_compare.py LABEL_A LABEL_B M`` -- every hash of
r3_fwd_bytes equal between the two labels (same build)?"""
import json
import os
import sys

from _r2 import BUILD, HERE

a, b, M = sys.argv[1], sys.argv[2], sys.argv[3]
A = json.load(open(os.path.join(HERE, f"r3_fwd_bytes_{a}_M{M}_{BUILD}.json")))
B = json.load(open(os.path.join(HERE, f"r3_fwd_bytes_{b}_M{M}_{BUILD}.json")))
n = same = 0
diff = []
for k, v in A.items():
    if not isinstance(v, dict) or k == "env":
        continue
    for way, hs in v.items():
        if way == "T_jit":
            continue
        n += len(hs)
        same += sum(x == y for x, y in zip(hs, B[k][way]))
        if hs != B[k][way]:
            diff.append((k, way))
print(f"{same} / {n} equal", "differ:", diff)

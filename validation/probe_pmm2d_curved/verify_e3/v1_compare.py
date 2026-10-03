"""Compare the v1_bytes JSON files against the PRE reference of each build.

    python v1_compare.py BUILD     (win | wsl)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
B = sys.argv[1]


def load(lbl):
    p = os.path.join(HERE, f"v1_bytes_{lbl}_{B}.json")
    return json.load(open(p)) if os.path.exists(p) else None


pre = load("pre")
out = {"build": B, "pre_keys": pre["keys"]}
errs = [k for k, v in pre["hashes"].items() if not v[0].isalnum() or
        v.startswith(("err", "refused"))]
out["pre_error_keys"] = errs
classes = sorted({k.split(".")[0] + "." + k.split(".")[1].split("_")[0]
                  for k in pre["hashes"]})
out["classes"] = classes
for lbl in ("post", "post_blocktwin", "post_blockjax", "pre_blockjax"):
    d = load(lbl)
    if d is None:
        continue
    ref = pre if lbl != "post_blockjax" else (load("pre_blockjax") or pre)
    diff = [k for k in ref["hashes"] if d["hashes"].get(k) != ref["hashes"][k]]
    missing = [k for k in ref["hashes"] if k not in d["hashes"]]
    out[lbl] = {"keys": d["keys"], "equal": d["keys"] - len(diff),
                "differ": diff, "missing": missing,
                "jax_imported": d["jax_imported"],
                "twin_imported": d["twin_imported"],
                "blocked_hits": d["blocked_hits"]}
    print(lbl, f"{d['keys'] - len(diff)}/{ref['keys']} equal", diff[:5],
          "twin_imported", d["twin_imported"])
# cross-check: the blocked-jax PRE equals the plain PRE (jax presence does not
# move a NumPy byte in either tree)
pj = load("pre_blockjax")
if pj is not None:
    out["pre_vs_pre_blockjax_differ"] = [
        k for k in pre["hashes"] if pj["hashes"].get(k) != pre["hashes"][k]]
    print("pre vs pre_blockjax differ:", out["pre_vs_pre_blockjax_differ"])
with open(os.path.join(HERE, f"v1_compare_{B}.json"), "w") as f:
    json.dump(out, f, indent=1)

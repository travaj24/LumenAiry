"""B1 comparison (Phase B copy of build_a/a1_compare.py): the PRE (git archive 539ce4a3) and POST hash sets of
b2_bytes.py, key by key.  Writes b2_compare.json.

  python validation/probe_pmm2d_curved/build_b/b2_compare.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
pre = json.load(open(os.path.join(HERE, "b2_bytes_pre.json")))
post = json.load(open(os.path.join(HERE, "b2_bytes_post.json")))
ps, qs = pre["sha"], post["sha"]
common = sorted(set(ps) & set(qs))
same = [k for k in common if ps[k] == qs[k]]
diff = [k for k in common if ps[k] != qs[k]]
only_post = sorted(set(qs) - set(ps))
idmap = [k for k in only_post if "@idmap" in k]
# fail-before: the identity map through the quadrature path vs the no-map
# hash of the SAME block (it must differ for at least one block per fixture)
fb = {}
for k in idmap:
    base = k.replace("@idmap", "")
    if base in qs:
        fb[k] = qs[k] != qs[base]
out = {"n_common": len(common), "n_identical": len(same),
       "differing": diff, "only_post": only_post,
       "idmap_blocks_differing_from_nomap": sum(fb.values()),
       "idmap_blocks_total": len(fb),
       "idmap_per_block": fb,
       "idmap_max_abs_diff": post.get("idmap_diff", {}),
       "env_pre": pre["env"], "env_post": post["env"]}
with open(os.path.join(HERE, "b2_compare.json"), "w") as f:
    json.dump(out, f, indent=1)
print(f"common {len(common)}  identical {len(same)}  differing {diff}")
print(f"idmap blocks differing from no-map: {sum(fb.values())} / {len(fb)}")
print(json.dumps(post.get("idmap_diff", {}), indent=1))

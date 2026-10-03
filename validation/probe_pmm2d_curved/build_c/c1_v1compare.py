"""C1 on the Phase A VERIFIER's own fixture set (verify_a/v1_bytes.py, run
unchanged with tags phaseCpre / phaseCpost on the PRE tree 91d00288 and on
this tree; outputs moved here as c1_v1bytes_{pre,post}.json): key by key.

  python validation/probe_pmm2d_curved/build_c/c1_v1compare.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
pre = json.load(open(os.path.join(HERE, "c1_v1bytes_pre.json")))["hashes"]
post = json.load(open(os.path.join(HERE, "c1_v1bytes_post.json")))["hashes"]
common = sorted(set(pre) & set(post))
diff = [k for k in common if pre[k] != post[k]]
res = {"n_pre": len(pre), "n_post": len(post), "n_common": len(common),
       "n_identical": len(common) - len(diff), "differing": diff,
       "only_pre": sorted(set(pre) - set(post)),
       "only_post": sorted(set(post) - set(pre))}
json.dump(res, open(os.path.join(HERE, "c1_v1compare.json"), "w"), indent=1)
print(res)

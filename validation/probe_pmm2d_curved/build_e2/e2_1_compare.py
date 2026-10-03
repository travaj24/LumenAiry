"""E2-1 compare: every hash of the PRE tree (eae470d9) against this tree;
the '@idmap' keys (post only) are the fail-before arm (the identity map
through the quadrature path must CHANGE the operator hashes)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
pre = json.load(open(os.path.join(HERE, "e2_1_bytes_pre.json")))["sha"]
post = json.load(open(os.path.join(HERE, "e2_1_bytes_post.json")))["sha"]
common = sorted(set(pre) & set(post))
diff = [k for k in common if pre[k] != post[k]]
mortar = [k for k in common if k.startswith("mortar_")]
idm = [k for k in post if "@idmap" in k]
idm_diff = [k for k in idm if post[k] != post.get(k.replace("@idmap", ""))]
idm_none = [k for k in idm if post[k] == post.get(k.replace("@idmap", ""))]
out = dict(n_pre=len(pre), n_post=len(post), n_common=len(common),
           n_equal=len(common) - len(diff), differing=diff,
           n_mortar=len(mortar),
           n_mortar_equal=sum(pre[k] == post[k] for k in mortar),
           idmap_total=len(idm), idmap_changed=len(idm_diff),
           idmap_unchanged=idm_none)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "e2_1_compare.json"), "w"), indent=1)

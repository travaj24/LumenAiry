"""V1 compare: PRE vs tip on each build, and the cross-build split."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
res = {}
for b in ("win", "wsl"):
    pre = json.load(open(os.path.join(HERE, f"v1_bytes_pre_{b}.json")))
    post = json.load(open(os.path.join(HERE, f"v1_bytes_post_{b}.json")))
    hp, hq = pre["hashes"], post["hashes"]
    assert set(hp) == set(hq), set(hp) ^ set(hq)
    diff = sorted(k for k in hp if hp[k] != hq[k])
    res[b] = dict(n_keys=len(hp), identical=len(hp) - len(diff), differ=diff)
    print(b, len(hp), "keys,", len(hp) - len(diff), "identical; differ:", diff)
w = json.load(open(os.path.join(HERE, "v1_bytes_post_win.json")))["hashes"]
l = json.load(open(os.path.join(HERE, "v1_bytes_post_wsl.json")))["hashes"]
cross = sum(w[k] == l[k] for k in w)
res["cross_build_equal_keys"] = cross
print("win vs wsl (tip) equal keys:", cross, "/", len(w))
json.dump(res, open(os.path.join(HERE, "v1_compare.json"), "w"), indent=1)

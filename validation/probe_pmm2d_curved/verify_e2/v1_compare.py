"""E2 verifier item 1 -- compare PRE (eae470d9) vs POST byte hashes, per build.

Reads v1_bytes_{pre,post}_{win,wsl}.json (own fixture set) and, when present,
v1_capture_{pre,post}_{win,wsl}.json (every solve of the shipped per-layer
mortar suites, hashed in-process by v1_capture_plugin.py).  Writes
v1_compare.json.  No lumenairy import.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def load(fn):
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


for build in ("win", "wsl"):
    pre, post = load(f"v1_bytes_pre_{build}.json"), load(
        f"v1_bytes_post_{build}.json")
    if pre and post:
        a, b = pre["sha"], post["sha"]
        common = sorted(set(a) & set(b))
        diff = [k for k in common if a[k] != b[k]]
        exc = [k for k in common if a[k].startswith("EXC")]
        mortar = [k for k in common if k.startswith(("pl_", "shipped."))]
        OUT[f"bytes_{build}"] = dict(
            n_pre=len(a), n_post=len(b), n_common=len(common),
            n_equal=len(common) - len(diff), differing=diff,
            only_pre=sorted(set(a) - set(b)), only_post=sorted(set(b) - set(a)),
            exceptions=exc, n_mortar=len(mortar),
            n_mortar_equal=sum(a[k] == b[k] for k in mortar),
            declared_changed={k: pre["declared"][k] != post["declared"][k]
                              for k in pre["declared"]},
            behaviour_pre={k: v[:160] for k, v in pre["behaviour"].items()},
            behaviour_post={k: v[:160] for k, v in post["behaviour"].items()},
            beh_num_post=post.get("beh_num"),
            fb_post=post["fb"], fb_pre=pre["fb"])
    cpre, cpost = load(f"v1_capture_pre_{build}.json"), load(
        f"v1_capture_post_{build}.json")
    if cpre and cpost:
        a, b = cpre["sha"], cpost["sha"]
        common = sorted(set(a) & set(b))
        diff = [k for k in common if a[k] != b[k]]
        bad_out = {k: (cpre["outcomes"].get(k), v) for k, v in
                   cpost["outcomes"].items() if v != "passed"}
        kinds = {}
        for k in common:
            what = k.split("::")[-1].split("#")[0]
            kinds[what] = kinds.get(what, 0) + 1
        OUT[f"capture_{build}"] = dict(
            n_pre=len(a), n_post=len(b), n_common=len(common),
            n_equal=len(common) - len(diff), differing=diff[:50],
            only_pre=sorted(set(a) - set(b))[:50],
            only_post=sorted(set(b) - set(a))[:50], by_kind=kinds,
            n_tests=len(cpost["outcomes"]), non_passed_post=bad_out,
            exitstatus=(cpre["exitstatus"], cpost["exitstatus"]))
# cross-build (informational: different BLAS / CPython)
w, l = load("v1_bytes_post_win.json"), load("v1_bytes_post_wsl.json")
if w and l:
    ks = sorted(set(w["sha"]) & set(l["sha"]))
    OUT["post_win_vs_wsl_equal"] = sum(w["sha"][k] == l["sha"][k] for k in ks)
    OUT["post_win_vs_wsl_total"] = len(ks)
with open(os.path.join(HERE, "v1_compare.json"), "w") as f:
    json.dump(OUT, f, indent=1)
for k, v in OUT.items():
    if isinstance(v, dict):
        print(k, {kk: vv for kk, vv in v.items()
                  if kk in ("n_pre", "n_post", "n_common", "n_equal",
                            "n_mortar", "n_mortar_equal", "differing",
                            "only_pre", "only_post", "exceptions",
                            "declared_changed", "n_tests", "non_passed_post",
                            "exitstatus")})
    else:
        print(k, v)

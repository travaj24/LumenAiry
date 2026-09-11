"""TASK F -- .test_durations validity + splice check for the two new gate files."""
import json
import pathlib

TREE = pathlib.Path(r"C:/tmp/lum_vbor")
d = json.load(open(TREE / ".test_durations"))
keys = list(d)
print("entries:", len(keys), "sorted:", keys == sorted(keys), "json: OK")
ids = [l.strip() for l in open(TREE / "validation/probe_verify_bor_guards/runs/vf_nodeids.txt")
       if "::" in l]
print("collected node ids:", len(ids))
miss = [i for i in ids if i not in d]
print("MISSING from .test_durations:", len(miss))
for m in miss:
    print("    MISS", m)
for f in ("test_fix_bor_multilayer_guards", "test_fix_eme_branch_cut", "test_bor_solve"):
    n = [k for k in keys if f + ".py" in k]
    c = [i for i in ids if f + ".py" in i]
    stale = [k for k in n if k not in ids]
    print("%-36s durations=%3d collected=%3d stale=%d %s"
          % (f, len(n), len(c), len(stale), stale[:6]))
    print("   duration sum: %.1f s   max: %.1f s (%s)"
          % (sum(d[k] for k in n), max(d[k] for k in n) if n else 0,
             max(n, key=lambda k: d[k]) if n else "-"))

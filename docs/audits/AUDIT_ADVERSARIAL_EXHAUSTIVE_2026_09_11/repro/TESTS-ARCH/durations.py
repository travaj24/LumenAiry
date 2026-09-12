"""Parse .test_durations: totals, slowest, fast/slow split proposal."""
import json, os, collections, statistics

ROOT = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"
d = json.load(open(os.path.join(ROOT, ".test_durations")))
print("entries:", len(d))
tot = sum(d.values())
print(f"total wall seconds: {tot:.1f}  ({tot/60:.1f} min, {tot/3600:.2f} h)")
vals = sorted(d.values(), reverse=True)
print(f"median: {statistics.median(vals):.4f}s  mean: {tot/len(vals):.3f}s  max: {vals[0]:.1f}s")
for thr in (60, 30, 10, 5, 1, 0.1):
    n = sum(1 for v in vals if v > thr)
    s = sum(v for v in vals if v > thr)
    print(f"  > {thr:6.1f}s : {n:6d} tests ({100*n/len(vals):5.2f}%)  "
          f"{s:9.1f}s ({100*s/tot:5.1f}% of total)")

print("\n=== TOP 30 slowest tests ===")
for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:30]:
    print(f"{v:8.1f}s  {k}")

print("\n=== TOP 25 slowest FILES ===")
byfile = collections.Counter()
cnt = collections.Counter()
for k, v in d.items():
    f = k.split("::")[0]
    byfile[f] += v
    cnt[f] += 1
for f, v in byfile.most_common(25):
    print(f"{v:8.1f}s  ({cnt[f]:4d} tests)  {f}")

print("\n=== fast/slow split proposal ===")
for bar in (1.0, 2.0, 5.0):
    slow_files = {f for f in byfile if byfile[f] > bar * 60}
    fast_s = sum(v for f, v in byfile.items() if f not in slow_files)
    print(f"  files with >{bar:.0f} min total: {len(slow_files)}  "
          f"-> 'fast' lane = {fast_s:.0f}s ({fast_s/60:.1f} min) over "
          f"{sum(cnt[f] for f in byfile if f not in slow_files)} tests; "
          f"'slow' lane = {tot-fast_s:.0f}s over {sum(cnt[f] for f in slow_files)} tests")

# what a 5-shard split looks like if balanced by file
print("\n=== 5-shard balance (greedy by file, using .test_durations) ===")
shards = [[] for _ in range(5)]
loads = [0.0] * 5
for f, v in byfile.most_common():
    i = loads.index(min(loads))
    shards[i].append(f)
    loads[i] += v
for i, l in enumerate(loads):
    print(f"  shard {i}: {l:8.1f}s ({l/60:5.1f} min) over {len(shards[i])} files")
print(f"  imbalance max/min = {max(loads)/min(loads):.3f}")
# pytest-split style (alphabetical chunking, which is what --splits does without durations)
files = sorted(byfile)
print("\n=== 5-shard balance if split ALPHABETICALLY by file (no durations) ===")
chunk = (len(files) + 4) // 5
for i in range(5):
    grp = files[i * chunk:(i + 1) * chunk]
    l = sum(byfile[f] for f in grp)
    print(f"  shard {i}: {l:8.1f}s ({l/60:5.1f} min) over {len(grp)} files")

# staleness: are the recorded ids still collectable?
col = os.path.join(r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TESTS-ARCH",
                   "collect_all.txt")
ids = set()
for line in open(col, encoding="utf-8", errors="replace"):
    line = line.strip()
    if "::" in line and line.startswith("tests"):
        ids.add(line.replace("\\", "/"))
dur_ids = {k.replace("\\", "/") for k in d}
print(f"\n=== .test_durations staleness ===")
print(f"  ids in .test_durations : {len(dur_ids)}")
print(f"  ids collected today    : {len(ids)}")
print(f"  in durations but GONE  : {len(dur_ids - ids)}")
print(f"  collected but NO timing: {len(ids - dur_ids)}")
miss = sorted(ids - dur_ids)
for x in miss[:10]:
    print("     new/untimed:", x)

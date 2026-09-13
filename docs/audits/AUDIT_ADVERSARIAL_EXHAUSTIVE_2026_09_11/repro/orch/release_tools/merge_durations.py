"""merge_durations.py FAST_JSON SLOW_JSON COLLECTED_TXT OUT_JSON
Merge the two lane duration files pytest-split wrote (--store-durations), keep only ids that the
current tree collects (COLLECTED_TXT = `pytest --collect-only -q` output), write OUT_JSON sorted.
Prints the coverage the A15a staleness gate will see."""
import json, sys, pathlib
fast, slow, collected, out = map(pathlib.Path, sys.argv[1:5])
d = {}
for p in (fast, slow):
    if p.exists():
        d.update(json.loads(p.read_text(encoding="utf-8")))
ids = {l.strip() for l in collected.read_text(encoding="utf-8", errors="replace").splitlines() if "::" in l}
kept = {k: v for k, v in d.items() if k in ids}
dropped = len(d) - len(kept)
missing = sorted(ids - set(kept))
out.write_text(json.dumps(kept, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"merged {len(d)} timed ids; dropped {dropped} no longer collected; kept {len(kept)}; "
      f"collected {len(ids)}; coverage {100*len(kept)/max(1,len(ids)):.2f} %; missing {len(missing)}")
for m in missing[:20]:
    print("  missing:", m)

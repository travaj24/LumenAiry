"""V1 comparison -- tip vs base (the "without" arm), and tip-bare vs every zero
spelling.  Reads the two JSONs v1_without_identity.py wrote and prints the
per-fixture verdict."""
import json
import pathlib
import sys

R = pathlib.Path(__file__).resolve().parent / "results"
tag = sys.argv[1] if len(sys.argv) > 1 else "win"
tip = json.loads((R / f"v1_without_identity_tip_{tag}.json").read_text("cp1252"))
base = json.loads(
    (R / f"v1_without_identity_base_{tag}.json").read_text("cp1252"))

names = sorted(tip["bare"])
cross_bad, spell_bad = [], []
for n in names:
    if base["bare"].get(n) != tip["bare"][n]:
        cross_bad.append(n)
for z, row in tip["spellings"].items():
    for n, h in row.items():
        if h != tip["bare"][n]:
            spell_bad.append((z, n))

print(f"tip  : {tip['_env']['lumenairy']}  ({tip['_env']['version']})")
print(f"base : {base['_env']['lumenairy']} ({base['_env']['version']})")
print(f"fixtures            : {len(names)}")
print(f"zero spellings      : {len(tip['spellings'])} "
      f"{sorted(tip['spellings'])}")
print(f"tip bare vs base    : {len(cross_bad)} differing  {cross_bad}")
print(f"tip bare vs spelled : {len(spell_bad)} differing  {spell_bad}")
for n in names:
    ok = "SAME" if base["bare"].get(n) == tip["bare"][n] else "DIFF"
    print(f"  {ok}  {n:34s} {tip['bare'][n][:16]}")

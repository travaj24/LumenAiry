"""Compare two v1_bitid runs hash by hash and warning set by warning set."""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
a, b = (sys.argv[1:3] + ["with", "pre"])[:2]
A = json.load(open(os.path.join(here, f"v1_bitid_{a}.json")))
B = json.load(open(os.path.join(here, f"v1_bitid_{b}.json")))
print(f"{a}: {A['lumenairy']}  py{A['python']} np{A['numpy']}")
print(f"{b}: {B['lumenairy']}  py{B['python']} np{B['numpy']}")

nid = nmis = nwarn = 0
mism = []
for name in sorted(set(A["fixtures"]) | set(B["fixtures"])):
    fa, fb = A["fixtures"].get(name, {}), B["fixtures"].get(name, {})
    if "error" in fa or "error" in fb:
        mism.append(f"{name}: ERROR {fa.get('error')} / {fb.get('error')}")
        continue
    ha, hb = fa.get("hashes", {}), fb.get("hashes", {})
    for k in sorted(set(ha) | set(hb)):
        if ha.get(k) == hb.get(k) and ha.get(k) is not None:
            nid += 1
        else:
            nmis += 1
            mism.append(f"{name}.{k}: {ha.get(k)} != {hb.get(k)}")
    if fa.get("warnings") != fb.get("warnings"):
        nwarn += 1
        mism.append(f"{name}: WARNING SET differs\n   {a}: "
                    f"{fa.get('warnings')}\n   {b}: {fb.get('warnings')}")
print(f"\nIDENTICAL {nid} / {nid + nmis} hashes; {nmis} mismatches; "
      f"{nwarn} warning-set differences")
for m in mism:
    print("  MISMATCH " + m)

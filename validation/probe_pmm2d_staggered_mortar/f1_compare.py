"""F1 -- diff the builds' ``f1_crossbuild_<tag>.json`` and report, per quantity,
every arm's reading and the CROSS-BUILD SPREAD.

Spread convention: ``(max - min) / max|.|`` over the arms -- relative, because
every quantity here is an error / condition MAGNITUDE whose test bar would be a
magnitude.  These spreads are what a future test bar must clear by decades
(``docs/TESTING_STANDARDS.md`` rule 5).

Usage:  python f1_compare.py [tag ...]        (default: win wsl win_nehalem)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAGS = sys.argv[1:] or ["win", "wsl", "win_nehalem"]
D = {}
for t in list(TAGS):
    p = os.path.join(HERE, f"f1_crossbuild_{t}.json")
    if not os.path.exists(p):
        print(f"(skipping absent arm {t})")
        TAGS.remove(t)
        continue
    D[t] = json.load(open(p))
for t in TAGS:
    print(f"arm {t:12s}", D[t]["build"])

out = []


def cmp(label, vals):
    lo, hi = min(vals), max(vals)
    m = max(abs(lo), abs(hi))
    s = 0.0 if m == 0 else (hi - lo) / m
    out.append(dict(quantity=label, vals=dict(zip(TAGS, vals)), spread=s))
    print(f"{label:50s} " + " ".join(f"{v:11.4e}" for v in vals)
          + f"   spread {s:8.2e}")


def rows(path):
    """Yield tuples of the same row across every arm."""
    cur = [D[t] for t in TAGS]
    for key in path:
        cur = [c[key] for c in cur]
    return list(zip(*cur))


print("\n--- [A] M1 conforming identity ---")
for tup in rows(["A_m1", "rows"]):
    for k in ("dR", "dT", "dJ"):
        cmp(f"M1 {tup[0]['case']:28s} {k}", [r[k] for r in tup])
cmp("M1 WORST over the 5 stacks", [D[t]["A_m1"]["worst"] for t in TAGS])

print("\n--- [B] isolated mortar vs analytic Fresnel ---")
for tup in rows(["B_fresnel"]):
    tag = (f"th={tup[0]['theta']:.2f} M={tup[0]['M']} "
           f"({tup[0]['Na']},{tup[0]['Nb']})")
    for k in ("dR", "closure"):
        cmp(f"Fresnel {tag:22s} {k}", [r[k] for r in tup])

print("\n--- [C] M4 vs the exact 1-D oracle ---")
cmp("oracle deg12-vs-deg14 self-gap",
    [D[t]["C_oracle_selfgap"] for t in TAGS])
for tup in rows(["C_m4"]):
    tag = f"{tup[0]['arm']} M={tup[0]['M']}"
    for k in ("err", "mirror", "closure"):
        cmp(f"M4 {tag:16s} {k}", [r[k] for r in tup])

print("\n--- [D] M4c equal-DOF ---")
for tup in rows(["D_equal_dof"]):
    for k in ("err_union", "err_mortar", "clo_union", "clo_mortar", "ratio"):
        cmp(f"equal-DOF q={tup[0]['q']} {k}", [r[k] for r in tup])

print("\n--- [E] OOP twin vs berreman_jones_1d ---")
for tup in rows(["E_oop"]):
    tag = (f"th={tup[0]['theta_deg']:.0f} M={tup[0]['M']} "
           f"({tup[0]['Na']},{tup[0]['Nb']})")
    for k in ("dJ", "closure"):
        cmp(f"OOP {tag:22s} {k}", [r[k] for r in tup])

print("\n--- [F] conditioning extremes ---")
for tup in rows(["F_cond", "rows"]):
    cmp(f"rcond {tup[0]['config']} M={tup[0]['M']}",
        [r["worst_rcond"] for r in tup])
for site in D[TAGS[0]]["F_cond"]["per_site"]:
    for i, w in enumerate(("min", "max")):
        cmp(f"rcond site {site[:32]:32s} {w}",
            [D[t]["F_cond"]["per_site"][site][i] for t in TAGS])

out.sort(key=lambda r: -r["spread"])
print("\n=== the 15 LARGEST cross-build spreads ===")
for r in out[:15]:
    v = " / ".join(f"{r['vals'][t]:.3e}" for t in TAGS)
    print(f"  {r['spread']:8.2e}  {r['quantity']}   ({v})")


def band(lo, hi):
    return [r for r in out if lo <= r["spread"] < hi]


print(f"\nquantities compared: {len(out)}")
for lo, hi, name in ((0.5, 1e9, ">= 50%"), (0.1, 0.5, "10-50%"),
                     (0.01, 0.1, "1-10%"), (1e-6, 0.01, "1e-6 .. 1%"),
                     (0.0, 1e-6, "< 1e-6")):
    print(f"  spread {name:12s}: {len(band(lo, hi)):3d}")
print(f"  bit-identical (spread == 0): "
      f"{len([r for r in out if r['spread'] == 0.0])}")

# the useful summary: the spread envelope per FAMILY of quantity
print("\n=== spread ENVELOPE per family (what a bar must clear) ===")
fam = {"M1 identity": "M1 ", "isolated mortar (Fresnel)": "Fresnel ",
       "M4 vs 1-D oracle": "M4 ", "equal-DOF": "equal-DOF ",
       "OOP vs Berreman": "OOP ", "conditioning rcond": "rcond "}
for name, pre in fam.items():
    sel = [r for r in out if r["quantity"].startswith(pre)]
    if sel:
        print(f"  {name:28s} n={len(sel):3d}  worst spread "
              f"{max(r['spread'] for r in sel):8.2e}   median "
              f"{sorted(r['spread'] for r in sel)[len(sel)//2]:8.2e}")

json.dump(out, open(os.path.join(HERE, "f1_compare.json"), "w"), indent=1)
print("\nwrote f1_compare.json")

"""F1 -- diff the two builds' ``f1_crossbuild_<tag>.json`` and report, per
quantity, the two readings and the CROSS-BUILD SPREAD.

Spread convention: ``|a - b| / max(|a|, |b|)`` (relative) for every quantity,
because every one of them is an error/condition magnitude whose test bar would
be a magnitude.  These spreads are what a future test bar must clear by
decades (``docs/TESTING_STANDARDS.md`` rule 5)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "f1_crossbuild_win.json")))
B = json.load(open(os.path.join(HERE, "f1_crossbuild_wsl.json")))
print("build A:", A["build"])
print("build B:", B["build"])


def spread(a, b):
    m = max(abs(a), abs(b))
    return 0.0 if m == 0 else abs(a - b) / m


out = []


def cmp(label, a, b):
    s = spread(a, b)
    out.append(dict(quantity=label, win=a, wsl=b, spread=s))
    print(f"{label:52s} {a:11.4e} {b:11.4e}   spread {s:8.2e}")


print("\n--- [A] M1 conforming identity (relative to max(R,T) / max|J|) ---")
for ra, rb in zip(A["A_m1"]["rows"], B["A_m1"]["rows"]):
    for k in ("dR", "dT", "dJ"):
        cmp(f"M1 {ra['case']:28s} {k}", ra[k], rb[k])
cmp("M1 WORST over the 5 stacks", A["A_m1"]["worst"], B["A_m1"]["worst"])

print("\n--- [B] isolated mortar vs analytic Fresnel ---")
for ra, rb in zip(A["B_fresnel"], B["B_fresnel"]):
    tag = f"th={ra['theta']:.2f} M={ra['M']} ({ra['Na']},{ra['Nb']})"
    cmp(f"Fresnel {tag:24s} |dR|", ra["dR"], rb["dR"])
    cmp(f"Fresnel {tag:24s} closure", ra["closure"], rb["closure"])

print("\n--- [C] M4 vs the exact 1-D oracle ---")
cmp("oracle deg12-vs-deg14 self-gap", A["C_oracle_selfgap"],
    B["C_oracle_selfgap"])
for ra, rb in zip(A["C_m4"], B["C_m4"]):
    tag = f"{ra['arm']} M={ra['M']}"
    cmp(f"M4 {tag:16s} err", ra["err"], rb["err"])
    cmp(f"M4 {tag:16s} mirror", ra["mirror"], rb["mirror"])
    cmp(f"M4 {tag:16s} closure", ra["closure"], rb["closure"])

print("\n--- [D] M4c equal-DOF ---")
for ra, rb in zip(A["D_equal_dof"], B["D_equal_dof"]):
    for k in ("err_union", "err_mortar", "clo_union", "clo_mortar", "ratio"):
        cmp(f"equal-DOF q={ra['q']} {k}", ra[k], rb[k])

print("\n--- [E] OOP twin vs berreman_jones_1d ---")
for ra, rb in zip(A["E_oop"], B["E_oop"]):
    tag = f"th={ra['theta_deg']:.0f} M={ra['M']} ({ra['Na']},{ra['Nb']})"
    cmp(f"OOP {tag:22s} dJones", ra["dJ"], rb["dJ"])
    cmp(f"OOP {tag:22s} closure", ra["closure"], rb["closure"])

print("\n--- [F] conditioning extremes ---")
for ra, rb in zip(A["F_cond"]["rows"], B["F_cond"]["rows"]):
    cmp(f"rcond {ra['config']} M={ra['M']}", ra["worst_rcond"],
        rb["worst_rcond"])
for site in A["F_cond"]["per_site"]:
    for i, w in enumerate(("min", "max")):
        cmp(f"rcond site {site[:34]:34s} {w}",
            A["F_cond"]["per_site"][site][i], B["F_cond"]["per_site"][site][i])

out.sort(key=lambda r: -r["spread"])
print("\n=== the 12 LARGEST cross-build spreads ===")
for r in out[:12]:
    print(f"  {r['spread']:8.2e}  {r['quantity']}   "
          f"({r['win']:.4e} vs {r['wsl']:.4e})")
big = [r for r in out if r["spread"] > 0.1]
print(f"\nquantities compared: {len(out)}; spread > 10%: {len(big)}; "
      f"spread > 1%: {len([r for r in out if r['spread'] > 0.01])}; "
      f"spread == 0 (bit-identical): "
      f"{len([r for r in out if r['spread'] == 0.0])}")
json.dump(out, open(os.path.join(HERE, "f1_compare.json"), "w"), indent=1)
print("wrote f1_compare.json")

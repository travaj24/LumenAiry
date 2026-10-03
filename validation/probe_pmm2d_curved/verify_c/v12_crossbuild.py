"""V12 -- the two builds side by side: every verdict / outcome of the merge,
primitive, viewer, departure and structure probes must agree between
Windows and WSL, and the incident numbers must agree to the discretisation
level (they are different BLAS / numpy builds, so not bitwise).

  python v12_crossbuild.py  -> v12_crossbuild.json
"""
import glob
import json
import os

from _vc import HERE


def load(p):
    with open(os.path.join(HERE, p)) as f:
        return json.load(f)


out = {}


def outcomes(d):
    r = {}
    for k, v in d.items():
        if isinstance(v, dict) and "outcome" in v:
            r[k] = v["outcome"] + ":" + str(v.get("verdict", ""))
    return r


for base in ("v2_merge", "v5_primitives_geom"):
    a, b = outcomes(load(f"{base}_win.json")), outcomes(load(f"{base}_wsl.json"))
    out[base] = {"n": len(a), "agree": a == b,
                 "differ": {k: (a.get(k), b.get(k)) for k in set(a) | set(b)
                            if a.get(k) != b.get(k)}}
for base in ("v11_departures", "v5b_ellipse_layout"):
    a, b = load(f"{base}_win.json"), load(f"{base}_wsl.json")
    a.pop("env"), b.pop("env")
    out[base] = {"agree": a == b}
a, b = load("v3_struct_win.json"), load("v3_struct_wsl.json")
out["v3_struct"] = {"agree_fired": all(
    a[k].get("fired") == b[k].get("fired") for k in a
    if k != "env" and isinstance(a[k], dict) and "fired" in a[k])}
a, b = load("v8_viewer_win.json"), load("v8_viewer_wsl.json")
out["v8_viewer_max_drawn_off"] = max(
    v["drawn_off_outline_max"] for k, v in b.items()
    if isinstance(v, dict) and "drawn_off_outline_max" in v)
inc = {}
for fw in sorted(glob.glob(os.path.join(HERE, "v4_*_wsl.json"))):
    fn = os.path.basename(fw)
    fwin = fn.replace("_wsl.json", "_win.json")
    if not os.path.exists(os.path.join(HERE, fwin)):
        continue
    a, b = load(fwin), load(fn)
    row = {}
    for arm in ("l2norm", "l2", "lstsq", "modepick"):
        if arm in a and arm in b:
            for q in ("airy_err", "spacer_0.3", "(-1,0)"):
                if q in a[arm]:
                    row[f"{arm}.{q}"] = (a[arm][q], b[arm][q])
    inc[fn] = row
out["v4_incident_win_vs_wsl"] = inc
for k, v in out.items():
    print(k, v if k != "v4_incident_win_vs_wsl" else "")
for fn, row in inc.items():
    print(fn, {k: f"{x:.2e}/{y:.2e}" for k, (x, y) in row.items()})
with open(os.path.join(HERE, "v12_crossbuild.json"), "w") as f:
    json.dump(out, f, indent=1)

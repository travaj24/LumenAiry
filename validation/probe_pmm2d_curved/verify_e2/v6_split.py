"""V6b -- what the two-layer 'workaround' computes vs the ONE-layer
supercell (macro-cell composite of verify_c/v10_macro.py, re-run here on the
E2 tree).  Fixture = v10_macro's: lambda 1, air above, n 1.45 below, eps 4
pillars, TOTAL thickness 0.5, n_orders 4 (every propagating order of the
doubled period).  Devices (all per-layer stacks unless noted):
  macro   one layer t 0.5, both pillars (cmap= composite, eps_cell=)
  AB      A (t 0.25) over B (t 0.25)     -- builder-style split
  BA      B over A
  ABAB    A, B, A, B at t 0.125 each
  A_only  one layer t 0.5, pillar A only (shared stack, merged map)
  B_only  likewise pillar B
  A_tinyB A (t 0.5) over B (t 1e-6)      -- the thin-layer limit
  A_tinyA A (t 0.5) over A (t 1e-6)      -- control: same map (plain)
  A_tinyVac A (t 0.5) over vacuum (t 1e-6) -- control: homogeneous (rides)
  (optional arg 3: comma list of devices; JSON then gets suffix _sub)
  python v6_split.py <M> <layout>  -> v6_split_<layout>_M<M>_win.json
"""
import sys
import time
import warnings

import numpy as np
from _ve import dump
from v6_layouts import LAYOUTS, macro

from lumenairy.elements.pmm import PMM2DStackPure

warnings.simplefilter("ignore")
M = int(sys.argv[1])
name = sys.argv[2]
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
px, py, A, B = LAYOUTS[name]
T = 0.5


def run(dev):
    kw = dict(n_superstrate=1.0, n_substrate=1.45, n_modes=M, n_orders=4)
    if dev == "macro":
        cm, eps = macro(name)
        st = PMM2DStackPure(px, py, cmap=cm, **kw)
        st.add_layer(T, eps_cell=eps)
        grids = [list(cm.shape)]
    else:
        plan = {"AB": [(A, T / 2), (B, T / 2)], "BA": [(B, T / 2), (A, T / 2)],
                "ABAB": [(A, T / 4), (B, T / 4)] * 2,
                "A_only": [(A, T)], "B_only": [(B, T)],
                "A_tinyB": [(A, T), (B, 1e-6)],
                "A_tinyA": [(A, T), (A, 1e-6)],
                "A_tinyVac": [(A, T), (None, 1e-6)]}[dev]
        lg = "per-layer" if len(plan) > 1 else "shared"
        st = PMM2DStackPure(px, py, layer_grids=lg, **kw)
        for sh, t in plan:
            if sh is None:
                st.add_layer(t, eps=1.0)
            else:
                st.add_layer(t, shapes=[sh], background_eps=1.0)
        if lg == "per-layer":
            grids = [list(L["own"]["cell"].shape[:2]) if L.get("own")
                     else None for L in st._layers]
        else:
            grids = [list(st.cmap.shape)]
    st.set_source(1.0)
    t0 = time.perf_counter()
    o, R, Tt, J = st.solve()
    wall = time.perf_counter() - t0
    R, Tt = np.asarray(R), np.asarray(Tt)
    return dict(R=R, T=Tt, orders=np.asarray(o),
                closure=float(np.max(np.abs(R.sum(1) + Tt.sum(1) - 1))),
                grids=grids,
                pencil_dof_per_component=[g[0] * (M - 1) * g[1] * (M - 1)
                                          if g else None for g in grids],
                wall=wall)


devs = only or ["macro", "AB", "BA", "ABAB", "A_only", "B_only", "A_tinyB"]
out = {"M": M, "layout": name, "t_total": T}
for d in devs:
    try:
        out[d] = run(d)
    except Exception as ex:   # noqa: BLE001
        out[d] = {"error": f"{type(ex).__name__}: {ex}"}
    print(d, {k: v for k, v in out[d].items()
              if k in ("closure", "grids", "wall", "error")}, flush=True)
ref = out.get("macro")
if ref and "R" in ref:
    i0 = int(np.argmin(np.abs(ref["orders"]).sum(1)))
    for d in devs:
        r = out[d]
        if "R" not in r:
            continue
        r["diff_vs_macro"] = float(max(np.abs(r["R"] - ref["R"]).max(),
                                       np.abs(r["T"] - ref["T"]).max()))
        r["T00"] = r["T"][:, i0].tolist()
        r["R00"] = r["R"][:, i0].tolist()
        print(d, "diff_vs_macro", r["diff_vs_macro"], "T00", r["T00"])
for tk in ("A_tinyB", "A_tinyA", "A_tinyVac"):
    if "A_only" in out and tk in out and "R" in out[tk]:
        a, b = out["A_only"], out[tk]
        out[f"{tk[2:]}_vs_A_only"] = float(max(
            np.abs(a["R"] - b["R"]).max(), np.abs(a["T"] - b["T"]).max()))
        print(tk, "vs A_only", out[f"{tk[2:]}_vs_A_only"])
dump(f"v6_split_{name}_M{M}" + ("_sub" if only else ""), out)

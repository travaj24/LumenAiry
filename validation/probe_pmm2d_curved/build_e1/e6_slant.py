"""E1-6 / E1-7 / E1-8 -- a SLANTED circular pillar: the composite map
x = Phi(u, v) + t w of Phase E1 against the two staircase limits.

Fixture: Phase D's D4 / planning P3 circle (period 1.2, radius 0.36, depth
0.5, lambda 1, air above, n = 1.45 below), the disk slanted by the PUBLIC
tangent ``slant = (0.2, 0)`` (cross-section at the top face, translating by
0.2 * depth toward +x down to the bottom face).  Materials: ``eps4`` (the
scalar eps-4 disk; E1-6) and ``oop30`` / ``nonrec30`` (the out-of-plane
director of E1-5 and its non-reciprocal twin; E1-7 -- slant x OOP x curved).

Families:
* curved -- c3 / c5 circle maps with ``slant=`` on the layer (Phase E1);
* stair  -- the SHIPPED slant solver (no map) on the planner's 4k-step
  in-plane staircases (walls at c +- r i / k), slanted by the same tangent:
  the in-plane staircase limit;
* zstair -- the shipped 2-D RCWA multilayer (``RCWAStack``) with N_z
  VERTICAL slices, each an EXACT disk (analytic form factor,
  ``shapes=``) translated to c + slant * z_mid: the z-staircase limit.  (A
  z-staircase of the CURVED solve itself needs a different circle position
  per layer, i.e. per-layer maps -- Phase E2, not this build; the exact-disk
  RCWA z-staircase stands in for it, with its own 1 / N order floor.)

usage: python e6_slant.py curved <c3|c5> <M> [theta phi [mat [tx ty]]]
       python e6_slant.py stair <k> <M>
       python e6_slant.py zstair <Nz> <n_orders>
       python e6_slant.py summary
"""
import json
import os
import sys
import time

import _common as C
import _e1common as E
import e5_pillar as E5
import numpy as np

P, R0, DEP, WL, NSUB, NSUP = 1.2, 0.36, 0.5, 1.0, 1.45, 1.0
CTR = P / 2
SLANT = (0.2, 0.0)
MATS = {"eps4": 4.0 * np.eye(3, dtype=complex), "oop30": E5.MATS["oop30"],
        "nonrec30": E5.MATS["nonrec30"],
        "nonrec30T": E5.MATS["nonrec30"].T.copy()}


def tag(th, ph):
    return f"t{th:.4f}_p{ph:.4f}"


def curved(kind, M, th=0.0, ph=0.0, mat="eps4", slant=SLANT):
    cm, eps = E5.disk_cells(kind, MATS[mat])
    t0 = time.perf_counter()
    st = E.PMM2DStackPure(P, P, n_superstrate=NSUP, n_substrate=NSUB,
                          n_modes=M, n_orders=3, cmap=cm)
    st.add_layer(DEP, eps_cell=eps, slant=slant)
    st.set_source(WL, theta=np.deg2rad(th), phi=np.deg2rad(ph))
    o, R, T, J = st.solve(jones=True)
    res = E5.record(st, o, R, T, J, {
        "family": "curved", "map": kind, "M": M, "theta": th, "phi": ph,
        "mat": mat, "slant": list(slant),
        "dof": 4 * (cm.shape[0] * (M - 1)) ** 2})
    res["t"] = time.perf_counter() - t0
    sl = "" if tuple(slant) == SLANT else f"_s{slant[0]:.3f}_{slant[1]:.3f}"
    E.dump(f"e6_curved_{mat}_{kind}_{tag(th, ph)}{sl}_M{M}.json", res)
    print(kind, mat, M, th, ph, slant, f"clo={res['closure']:.2e} "
          f"t={res['t']:.0f}", flush=True)


def stair(k, M, mat="eps4"):
    c = CTR
    inner = sorted([c - R0 * i / k for i in range(1, k + 1)]
                   + [c + R0 * i / k for i in range(1, k + 1)])
    w = np.array([0.0] + inner + [P])
    n = len(w) - 1
    mid = 0.5 * (w[:-1] + w[1:])
    eps = np.ones((n, n), complex)
    for i in range(n):
        for j in range(n):
            if (mid[i] - c) ** 2 + (mid[j] - c) ** 2 < R0 ** 2:
                eps[i, j] = 4.0
    t0 = time.perf_counter()
    st = E.PMM2DStackPure(P, P, n_superstrate=NSUP, n_substrate=NSUB,
                          n_modes=M, n_orders=3, layer_grids="per-layer")
    st.add_layer(DEP, eps_cell=eps, x_walls=w, y_walls=w, slant=SLANT)
    st.set_source(WL)
    o, R, T, J = st.solve(jones=True)
    res = E5.record(st, o, R, T, J, {"family": "stair", "k": k, "M": M,
                                     "walls": w.tolist(), "mat": mat})
    res["t"] = time.perf_counter() - t0
    E.dump(f"e6_stair_{mat}_k{k}_M{M}.json", res)
    print("stair", k, M, f"t={res['t']:.0f}", flush=True)


def zstair(Nz, n):
    from lumenairy.elements.rcwa import RCWAStack
    t0 = time.perf_counter()
    st = RCWAStack(P, period_y=P, n_superstrate=NSUP, n_substrate=NSUB,
                   n_orders=n, n_orders_y=n)
    dz = DEP / Nz
    for j in range(Nz):
        z = (j + 0.5) * dz
        cx = (CTR + SLANT[0] * z) % P
        cy = (CTR + SLANT[1] * z) % P
        st.add_layer(dz, shapes=[{"shape": "disk", "eps": 4.0,
                                  "radius": R0, "center": (cx, cy)}],
                     eps_background=1.0)
    res_ = st.set_source(WL, theta=0.0, phi=0.0).solve()
    o, R, T = res_.efficiencies()
    o = np.asarray(o)
    if o.ndim == 1 or o.shape[-1] != 2:
        o = np.asarray(res_._require_modal()["orders2d"])
    res = E5.record(None, o, R, T, None, {"family": "zstair", "Nz": Nz,
                                          "n": n, "mat": "eps4"})
    res["t"] = time.perf_counter() - t0
    E.dump(f"e6_zstair_eps4_Nz{Nz}_n{n}.json", res)
    print("zstair", Nz, n, f"clo={res['closure']:.2e} t={res['t']:.0f}",
          flush=True)


def _load(prefix):
    here = os.path.dirname(os.path.abspath(__file__))
    return [json.load(open(os.path.join(here, fn)))
            for fn in sorted(os.listdir(here))
            if fn.startswith(prefix) and fn.endswith(".json")]


def summary():
    out = {}
    nt = tag(0.0, 0.0)
    c3 = sorted(_load(f"e6_curved_eps4_c3_{nt}_M"), key=lambda r: r["M"])
    c5 = sorted(_load(f"e6_curved_eps4_c5_{nt}_M"), key=lambda r: r["M"])
    ref = np.array(c3[-1]["vec"]) if c3 else None
    for key, rows in (("c3", c3), ("c5", c5)):
        out[key] = []
        for a, b in zip(rows, rows[1:] + [None]):
            row = {"M": a["M"], "closure": a["closure"], "t": a["t"],
                   "to_c3_top": float(np.max(np.abs(np.array(a["vec"])
                                                    - ref)))}
            if b is not None:
                row["d_next"] = float(np.max(np.abs(np.array(a["vec"])
                                                    - np.array(b["vec"]))))
            out[key].append(row)
    if c3 and c5:
        out["c3_top_vs_c5_top"] = float(np.max(np.abs(
            np.array(c3[-1]["vec"]) - np.array(c5[-1]["vec"]))))
    st = {}
    for r in _load("e6_stair_eps4_"):
        st.setdefault(r["k"], []).append(
            {"M": r["M"], "t": r["t"], "closure": r["closure"],
             "to_c3_top": float(np.max(np.abs(np.array(r["vec"]) - ref)))})
    out["stair"] = {str(k): sorted(v, key=lambda r: r["M"])
                    for k, v in sorted(st.items())}
    zs = {}
    for r in _load("e6_zstair_eps4_"):
        zs.setdefault(r["Nz"], []).append(
            {"n": r["n"], "N": 2 * r["n"] + 1, "t": r["t"],
             "closure": r["closure"],
             "to_c3_top": float(np.max(np.abs(np.array(r["vec"]) - ref))),
             "vec": r["vec"]})
    out["zstair"] = {}
    for Nz, rows in sorted(zs.items()):
        rows = sorted(rows, key=lambda r: r["n"])
        rich = []
        for a, b in zip(rows, rows[1:]):
            ext = ((b["N"] * np.array(b["vec"]) - a["N"]
                    * np.array(a["vec"])) / (b["N"] - a["N"]))
            rich.append({"pair": [a["N"], b["N"]],
                         "to_c3_top": float(np.max(np.abs(ext - ref))),
                         "ext": ext.tolist()})
        out["zstair"][str(Nz)] = {
            "rows": [{k: v for k, v in r.items() if k != "vec"}
                     for r in rows],
            "richardson_1_over_N": [{k: v for k, v in r.items()
                                     if k != "ext"} for r in rich]}
        if rich:
            out["zstair"][str(Nz)]["_ext_top"] = rich[-1]["ext"]
    # the z-limit: Richardson in 1/Nz^2 (midpoint slices) on the per-Nz
    # order-extrapolated vectors
    nzs = sorted(int(k) for k in out["zstair"]
                 if "_ext_top" in out["zstair"][k])
    zl = []
    for a, b in zip(nzs, nzs[1:]):
        va = np.array(out["zstair"][str(a)]["_ext_top"])
        vb = np.array(out["zstair"][str(b)]["_ext_top"])
        ext = (b * b * vb - a * a * va) / (b * b - a * a)
        zl.append({"pair": [a, b],
                   "to_c3_top": float(np.max(np.abs(ext - ref))),
                   "step": float(np.max(np.abs(vb - va)))})
    out["zstair_limit_1_over_Nz2"] = zl
    for k in out["zstair"]:
        out["zstair"][k].pop("_ext_top", None)
    # E1-7 / E1-8: closure, reciprocity, generalized reciprocity
    groups = {}
    for r in _load("e6_curved_"):
        if r.get("slant") != list(SLANT):
            continue
        groups.setdefault((r["mat"], r["map"], tag(r["theta"], r["phi"])),
                          {})[r["M"]] = r
    out["closure"] = {f"{m}_{k}_{t}": {str(M): r["closure"]
                                       for M, r in sorted(rs.items())}
                      for (m, k, t), rs in groups.items()}
    out["reciprocity"] = {}
    pairs = [((25.0, 0.0), (-1, 0)), ((25.0, 40.0), (-1, 0)),
             ((25.0, 40.0), (0, -1))]
    for mat, matT in (("eps4", "eps4"), ("oop30", "oop30"),
                      ("nonrec30", "nonrec30T")):
        for kind in ("c3", "c5"):
            for (th, ph), (m, n) in pairs:
                tr, pr = E5.reverse_angles(th, ph, m, n)
                fwd = groups.get((mat, kind, tag(th, ph)), {})
                rev = groups.get((mat, kind, tag(tr, pr)), {})
                revT = groups.get((matT, kind, tag(tr, pr)), {})
                k = E5.ORDS.index((m, n))
                kw = E5.ORDS.index((0, 0))
                rows = []
                for M in sorted(set(fwd) & set(rev)):
                    sf = np.linalg.svd(E5.jones_block(fwd[M], k),
                                       compute_uv=False)
                    sr = np.linalg.svd(E5.jones_block(rev[M], k),
                                       compute_uv=False)
                    sw = np.linalg.svd(E5.jones_block(rev[M], kw),
                                       compute_uv=False)
                    row = {"M": M, "recip": float(np.max(np.abs(sf - sr))),
                           "wrong_pair": float(np.max(np.abs(sf - sw)))}
                    if M in revT and matT != mat:
                        sT = np.linalg.svd(E5.jones_block(revT[M], k),
                                           compute_uv=False)
                        row["recip_vs_transposed_tensor"] = float(
                            np.max(np.abs(sf - sT)))
                    rows.append(row)
                if rows:
                    out["reciprocity"][f"{mat}_{kind}_{tag(th, ph)}_"
                                       f"({m},{n})"] = {
                        "reverse": [tr, pr], "rows": rows}
    E.dump("e6_summary.json", out)
    print(json.dumps(out, indent=1)[:8000])


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "curved":
        extra = ([float(sys.argv[4]), float(sys.argv[5])]
                 if len(sys.argv) > 5 else [0.0, 0.0])
        mat = sys.argv[6] if len(sys.argv) > 6 else "eps4"
        sl = ((float(sys.argv[7]), float(sys.argv[8]))
              if len(sys.argv) > 8 else SLANT)
        curved(sys.argv[2], int(sys.argv[3]), *extra, mat=mat, slant=sl)
    elif a == "stair":
        stair(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "zstair":
        zstair(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "summary":
        summary()
    else:
        raise SystemExit(__doc__)

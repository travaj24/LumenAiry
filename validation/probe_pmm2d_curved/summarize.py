"""Reduce the P1-P5 JSONs to the numbers the plan quotes -> summary.json.

Run:  cd /c/tmp/lum_curved && python validation/probe_pmm2d_curved/summarize.py
Every number in docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md section 3
is read from summary.json (or directly from the per-probe JSON it names).
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORD9 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]


def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def both(row):
    return np.concatenate([np.asarray(row["vec_te"]), np.asarray(row["vec_tm"])])


def ladder(runs, key="M"):
    """successive differences and differences to the top rung."""
    runs = sorted(runs, key=lambda r: r[key])
    top = both(runs[-1])
    out = []
    for i, r in enumerate(runs):
        d_top = float(np.max(np.abs(both(r) - top)))
        d_next = (float(np.max(np.abs(both(runs[i + 1]) - both(r))))
                  if i + 1 < len(runs) else None)
        out.append({key: r[key], "dof": r.get("dof"), "d_to_top": d_top,
                    "d_next": d_next,
                    "closure": max(r.get("closure_te", np.nan),
                                   r.get("closure_tm", np.nan)),
                    "R00_te": r["te"]["0,0"][0], "T00_te": r["te"]["0,0"][1],
                    "T10_te": r["te"]["1,0"][1]})
    return out


def p2():
    res = {}
    for fx in ("stripe", "pillar"):
        d = load(f"p2_stretch_{fx}.json")
        if d is None:
            continue
        runs = d["runs"]
        a_vals = sorted({r["a_over_px"] for r in runs})
        block = {}
        if fx == "stripe":
            o = d["oracle_1d"]
            ref = []
            for pol in ("te", "tm"):
                R = [o[pol]["40"].get(str(m), [0, 0])[0] if n == 0 else 0.0 for m, n in ORD9]
                T = [o[pol]["40"].get(str(m), [0, 0])[1] if n == 0 else 0.0 for m, n in ORD9]
                ref.append(np.array(R + T))
            ref = np.concatenate(ref)
            oracle_self = max(abs(o[p]["30"][k][i] - o[p]["40"][k][i])
                              for p in ("te", "tm") for k in o[p]["40"] for i in (0, 1))
            block["oracle"] = "pmm_efficiency_1d degree 40 (degree 30 vs 40 self-gap %.1e)" % oracle_self
        else:
            base = [r for r in runs if r["a_over_px"] == 0.0]
            top = max(base, key=lambda r: r["M"])
            ref = both(top)
            block["oracle"] = f"unmapped a=0 at M={top['M']}"
        for a in a_vals:
            rr = sorted([r for r in runs if r["a_over_px"] == a], key=lambda r: r["M"])
            rows = []
            for r in rr:
                same_M0 = [b for b in runs if b["a_over_px"] == 0.0 and b["M"] == r["M"]]
                rows.append({"M": r["M"], "dof": r["dof"],
                             "err_vs_oracle": float(np.max(np.abs(both(r) - ref))),
                             "diff_vs_unmapped_same_M": (float(np.max(np.abs(both(r) - both(same_M0[0]))))
                                                        if same_M0 else None),
                             "closure": max(r["closure_te"], r["closure_tm"]),
                             "geom_split_rel": r["diag"].get("geom_split_rel"),
                             "t_total": r["t_total"]})
            block[f"a={a}"] = rows
        res[fx] = block
    return res


def p3():
    res = {}
    for tag in ("curved3", "curved5"):
        d = load(f"p3_circle_{tag}.json")
        if d is None:
            continue
        res[tag] = {"disk_area_mapped": d["disk_area_mapped"],
                    "disk_area_exact": d["disk_area_exact"],
                    "detJ_min_max": d["detJ_min_max_incl_corners"],
                    "ladder": ladder(d["runs"])}
        top = max(d["runs"], key=lambda r: r["M"])
        res[tag]["top"] = {"M": top["M"], "te": top["te"], "tm": top["tm"]}
        # C4 symmetry of the map + disk: R00/T00 polarization-independent,
        # and te(m, n) == tm(n, m)
        c4 = []
        for r in d["runs"]:
            dd = 0.0
            for (m, n) in ORD9:
                a = r["te"][f"{m},{n}"]
                b = r["tm"][f"{n},{m}"]
                dd = max(dd, abs(a[0] - b[0]), abs(a[1] - b[1]))
            c4.append({"M": r["M"], "max_te_mn_minus_tm_nm": dd})
        res[tag]["c4_symmetry"] = c4
    q = load("p3_circle_quad_M8.json")
    if q is not None:
        v = [both(r) for r in q["runs"]]
        res["quadrature_M8"] = [{"nq": r["nq"], "d_to_finest": float(np.max(np.abs(x - v[-1])))}
                                for r, x in zip(q["runs"], v)]
    rc = load("p3_circle_rcwa.json")
    if rc is not None:
        res["rcwa_shapes"] = [{"n_orders": r["n_orders"], "R00_te": r["te"]["0,0"][0],
                               "T00_te": r["te"]["0,0"][1], "T10_te": r["te"]["1,0"][1],
                               "vec": both(r).tolist(), "t": r["t"]} for r in rc["shapes"]]
        res["rcwa_pixel_li"] = [{"S": r["S"], "n_orders": r["n_orders"],
                                 "R00_te": r["te"]["0,0"][0], "T00_te": r["te"]["0,0"][1],
                                 "T10_te": r["te"]["1,0"][1],
                                 "vec": both(r).tolist(), "t": r["t"]} for r in rc["pixel"]]
    for k in (1, 2, 4):
        s = load(f"p3_circle_stair_k{k}.json")
        if s is None:
            continue
        res[f"stair_k{k}"] = {"steps": s["steps"], "area": s["stair_area"],
                              "disk_area": s["disk_area"], "ladder": ladder(s["runs"])}
    # distances of every oracle family to the curved top rung
    if "curved3" in res:
        d3 = load("p3_circle_curved3.json")
        top = both(max(d3["runs"], key=lambda r: r["M"]))
        dist = {}
        if "curved5" in res:
            d5 = load("p3_circle_curved5.json")
            dist["curved5_top"] = float(np.max(np.abs(both(max(d5["runs"], key=lambda r: r["M"])) - top)))
        for key in ("rcwa_shapes", "rcwa_pixel_li"):
            if key in res:
                dist[key] = [(r.get("n_orders"), r.get("S"),
                              float(np.max(np.abs(np.asarray(r["vec"]) - top))))
                             for r in res[key]]
        for k in (1, 2, 4):
            s = load(f"p3_circle_stair_k{k}.json")
            if s is not None:
                st = both(max(s["runs"], key=lambda r: r["M"]))
                dist[f"stair_k{k}_top"] = float(np.max(np.abs(st - top)))
        res["distance_to_curved3_top"] = dist
    # the 3-D FEM oracle (fem/, NGSolve, curved elements, 3 independent meshes)
    fem = load(os.path.join("fem", "summary.json"))
    if fem is not None:
        keys = [("R", "0,0"), ("R", "1,0"), ("R", "0,1"),
                ("T", "0,0"), ("T", "1,0"), ("T", "0,1"), ("T", "1,1")]
        fv = np.array([fem[s][k]["value"] for s, k in keys])
        ferr = max(fem[s][k]["maxdev"] for s, k in keys)

        def te7(row):
            return np.array([row["te"][k][0 if s == "R" else 1] for s, k in keys])
        vs = {}
        for tag in ("curved3", "curved5"):
            d = load(f"p3_circle_{tag}.json")
            if d is not None:
                vs[tag] = [(r["M"], r["dof"], float(np.max(np.abs(te7(r) - fv))))
                           for r in sorted(d["runs"], key=lambda r: r["M"])]
        rc = load("p3_circle_rcwa.json")
        if rc is not None:
            vs["rcwa_shapes"] = [(r["n_orders"], None, float(np.max(np.abs(te7(r) - fv))))
                                 for r in rc["shapes"]]
            vs["rcwa_pixel_li"] = [(r["n_orders"], r["S"], float(np.max(np.abs(te7(r) - fv))))
                                   for r in rc["pixel"]]
        for k in (1, 2, 4):
            s_ = load(f"p3_circle_stair_k{k}.json")
            if s_ is not None:
                vs[f"stair_k{k}"] = [(r["M"], r["dof"], float(np.max(np.abs(te7(r) - fv))))
                                     for r in sorted(s_["runs"], key=lambda r: r["M"])]
        res["fem_oracle"] = {"values_te_R00_R10_R01_T00_T10_T01_T11": fv.tolist(),
                             "fem_maxdev_across_3_meshes": ferr,
                             "fem_RplusT": fem["RplusT"]["value"],
                             "max_abs_diff_vs_fem (M_or_n, dof_or_S, diff)": vs}
    return res


def p4():
    res = {}
    for f in sorted(glob.glob(os.path.join(HERE, "p4_fillet_r*.json"))):
        d = json.load(open(f))
        ratio = d["fixture"]["fillet_over_side"]
        top = max(d["runs"], key=lambda r: r["M"])
        res[str(ratio)] = {"pillar_area_mapped": d["pillar_area_mapped"],
                           "pillar_area_exact": d["pillar_area_exact"],
                           "detJ_min_max": d["detJ_min_max"],
                           "ladder": ladder(d["runs"]),
                           "top": {"M": top["M"], "dof": top["dof"],
                                   "R00_te": top["te"]["0,0"][0],
                                   "T00_te": top["te"]["0,0"][1],
                                   "R00_tm": top["tm"]["0,0"][0],
                                   "T00_tm": top["tm"]["0,0"][1]}}
    return res


def modes():
    res = {}
    for f in sorted(glob.glob(os.path.join(HERE, "p34_modes_*.json"))):
        d = json.load(open(f))
        runs = sorted(d["runs"], key=lambda r: r["M"])
        g = [np.sort(np.array([z[0] for z in r["g2_top"]]))[::-1][:4] for r in runs]
        rows = []
        for i, r in enumerate(runs):
            rows.append({"M": r["M"], "dof": r["dof"],
                         "g2_fund": float(g[i][0]),
                         "d_next": (float(np.max(np.abs(g[i + 1] - g[i])))
                                    if i + 1 < len(runs) else None),
                         "d_to_top": float(np.max(np.abs(g[i] - g[-1])))})
        res[d["case"]] = rows
    return res


def p2b():
    return load("p2b_failbefore.json")


def film():
    d = load("p3_circle_film.json")
    if d is None:
        return None
    return [{"M": r["M"], "dof": r["dof"], "err_vs_fresnel": r["err"]} for r in d["runs"]]


def rcwa_extrap():
    rc = load("p3_circle_rcwa.json")
    d3 = load("p3_circle_curved3.json")
    if rc is None or d3 is None:
        return None
    top = both(max(d3["runs"], key=lambda r: r["M"]))
    sh = sorted(rc["shapes"], key=lambda r: r["n_orders"])
    out = []
    for a, b in zip(sh[:-1], sh[1:]):
        n1, n2 = a["n_orders"], b["n_orders"]
        v1, v2 = both(a), both(b)
        ext = (n2 * v2 - n1 * v1) / (n2 - n1)          # Richardson, error ~ 1/n
        out.append({"pair": [n1, n2],
                    "raw_dist_n2": float(np.max(np.abs(v2 - top))),
                    "richardson_p1_dist": float(np.max(np.abs(ext - top)))})
    return out


def main():
    out = {"P1": None, "P2": p2(), "P2b_failbefore": p2b(), "P3": p3(),
           "P3_film_under_circle_map": film(), "P3_rcwa_richardson": rcwa_extrap(),
           "P34_modes": modes(), "P4": p4(), "P5": load("p5_cost.json")}
    p1 = load("p1_identity.json")
    if p1 is not None:
        out["P1"] = {
            "operators_max_rel": max(v["max_rel"] for c in p1["cases"]
                                     for v in c["ops"].values() if isinstance(v, dict)),
            "operators_bit_identical_any": any(v["bit_identical"] for c in p1["cases"]
                                               for v in c["ops"].values() if isinstance(v, dict)),
            "eig_matched_max_rel": max(c["eig_matched_max_rel"] for c in p1["cases"]),
            "far_projector_max_abs": max(max(c["far"]["P1_vs_xu"]["max_abs"],
                                             c["far"]["P2_vs_yv"]["max_abs"]) for c in p1["cases"]),
            "full_solve_max_abs_RT": max(max(r[k] for k in r if k.endswith(("_dR", "_dT")))
                                         for r in p1["full_solve"]),
        }
    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1)[:20000])


if __name__ == "__main__":
    main()

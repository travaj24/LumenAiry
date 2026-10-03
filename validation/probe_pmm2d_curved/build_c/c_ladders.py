"""C2 / C3 / C10 -- the shape primitives against Phase B's gate maps.

The primitives must build EXACTLY the maps Phase B measured (same walls, same
curves, same vertex images -> same fingerprint -> same bytes), and the
results must land where Phase B's did against the saved 3-D FEM oracle (the
circle), the sharp-square limit (the fillets) and reciprocity (oblique and
conical incidence).  Every rung solves the SHAPES route
(``pmm_jones_2d_staggered(..., shapes=...)`` / ``PMM2DStackPure.add_layer(
shapes=...)``); with ``both`` it also solves the EXPLICIT-MAP route (Phase B's
private builder through ``cmap=``) and records whether the two are
byte-identical.  The difference to Phase B's saved rung JSON is recorded too:
it is the incident-field fix of this phase (gate C9), not the map.

  c_ladders.py circle c3|c5 <M> [both]
  c_ladders.py fillet <r/side> <M> [both]
  c_ladders.py oblique c3|c5 <M> <theta_deg> <phi_deg> [both]
  c_ladders.py summary

Output: c_<kind>_*.json, c_ladders_summary.json
"""
import json
import os
import sys
import time

import _common as C
import numpy as np

from lumenairy.elements.pmm import (  # noqa: E402
    Circle,
    FilletRect,
    PMM2DStackPure,
)

BUILD_B = os.path.join(C.HERE, "..", "build_b")
ORD_FEM = {"0,0": [(0, 0)], "1,0": [(1, 0), (-1, 0)],
           "0,1": [(0, 1), (0, -1)],
           "1,1": [(1, 1), (-1, 1), (1, -1), (-1, -1)]}
# Phase B's b10 order list (n slowest); every summary below indexes a rung
# by the 'orders' list stored IN its own file, so files written with another
# order compare correctly
ORDS_B10 = [(m, n) for n in (-1, 0, 1) for m in (-1, 0, 1)]


def loadb(name):
    p = os.path.join(BUILD_B, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def stack_solve(M, theta=0.0, phi=0.0, shapes=None, cmap=None, eps=None):
    st = PMM2DStackPure(C.P, C.P, n_superstrate=C.N_SUP,
                        n_substrate=C.N_SUB, n_modes=M, n_orders=3,
                        cmap=cmap)
    if shapes is not None:
        st.add_layer(C.DEPTH, shapes=shapes, background_eps=1.0)
    else:
        st.add_layer(C.DEPTH, eps_cell=eps)
    st.set_source(C.WL, theta=np.deg2rad(theta), phi=np.deg2rad(phi))
    t0 = time.perf_counter()
    o, R, T, J = st.solve()
    return st, np.asarray(o), np.asarray(R), np.asarray(T), J, \
        time.perf_counter() - t0


def circle_shape(layout):
    return [Circle(C.P / 2, C.P / 2, C.R_CIRC, C.EPS_P,
                   core=None if layout == "c3" else 0.5)]


def circle_explicit(layout):
    if layout == "c3":
        cm, _w = C.CM._circle_map_3x3(C.P, C.R_CIRC)
        eps = np.ones((3, 3), complex)
        eps[1, 1] = C.EPS_P
    else:
        cm, _w = C.CM._circle_map_5x5(C.P, C.R_CIRC)
        eps = np.ones((5, 5), complex)
        eps[1:4, 1:4] = C.EPS_P
    return cm, eps


def compare(st, a, b):
    """byte identity of two (st, o, R, T, J, t) solves, and the map."""
    return {"fingerprint_equal": None,
            "bytes_equal": bool(all(np.array_equal(x, y) for x, y in
                                    zip(a[1:5], b[1:5]))),
            "max_abs_diff": float(max(np.max(np.abs(x - y)) for x, y in
                                      zip(a[2:4], b[2:4])))}


def circle(layout, M, both=False):
    a = stack_solve(M, shapes=circle_shape(layout))
    st, o, R, T, J, t = a
    res = {"env": C.env_record(), "layout": layout, "M": M, "t": t,
           "map": type(st.cmap).__name__, "vec": C.vec(o, R, T).tolist(),
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1)))}
    cm, eps = circle_explicit(layout)
    res["fingerprint_equal"] = st.cmap.fingerprint == cm.fingerprint
    res["eps_cell_equal"] = bool(np.array_equal(
        st._layers[0]["eps_cell"], eps))
    if both:
        b = stack_solve(M, cmap=cm, eps=eps)
        res["explicit"] = compare(st, a, b)
    pb = loadb(f"b4_circle_{layout}_M{M}.json")
    if pb is not None:
        res["phaseB_vec_maxdiff"] = float(np.max(np.abs(
            np.asarray(pb["vec"]) - np.asarray(res["vec"]))))
    print(res, flush=True)
    C.dump(f"c_circle_{layout}_M{M}.json", res)


def fillet(ratio, M, both=False):
    side = 0.6
    shp = [FilletRect(C.P / 2, C.P / 2, side, side, ratio * side, C.EPS_P)]
    a = stack_solve(M, shapes=shp)
    st, o, R, T, J, t = a
    i0 = C.idx(o, [(0, 0)])[0]
    res = {"env": C.env_record(), "ratio": ratio, "M": M, "t": t,
           "R00": R[:, i0].tolist(), "T00": T[:, i0].tolist(),
           "vec": C.vec(o, R, T).tolist(),
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1)))}
    cm, _w = C.CM._fillet_map_5x5(C.P, side / 2, ratio * side)
    eps = np.ones((5, 5), complex)
    eps[1:4, 1:4] = C.EPS_P
    res["fingerprint_equal"] = st.cmap.fingerprint == cm.fingerprint
    res["eps_cell_equal"] = bool(np.array_equal(
        st._layers[0]["eps_cell"], eps))
    if both:
        b = stack_solve(M, cmap=cm, eps=eps)
        res["explicit"] = compare(st, a, b)
    pb = loadb(f"b6_fillet_r{ratio}_M{M}.json")
    if pb is not None:
        res["phaseB_vec_maxdiff"] = float(np.max(np.abs(
            np.asarray(pb["vec"]) - np.asarray(res["vec"]))))
        res["phaseB_R00_T00_diff"] = [
            float(np.max(np.abs(np.asarray(pb["R00"]) - R[:, i0]))),
            float(np.max(np.abs(np.asarray(pb["T00"]) - T[:, i0])))]
    print(res, flush=True)
    C.dump(f"c_fillet_r{ratio}_M{M}.json", res)


def tag(th, ph):
    return f"t{th:.4f}_p{ph:.4f}"


def oblique(layout, M, th, ph, both=False):
    a = stack_solve(M, th, ph, shapes=circle_shape(layout))
    st, o, R, T, J, t = a
    md = st._modal
    idx = C.idx(o, ORDS_B10)
    res = {"env": C.env_record(), "map": layout, "M": M, "theta": th,
           "phi": ph, "nocof": False, "t": t,
           "orders": [list(v) for v in ORDS_B10],
           "R": R[:, idx].tolist(), "T": T[:, idx].tolist(),
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
           "kx": [float(md["kx"][i]) for i in idx],
           "ky": [float(md["ky"][i]) for i in idx],
           "kz_ref": [[float(np.real(md["kz_ref"][i])),
                       float(np.imag(md["kz_ref"][i]))] for i in idx],
           "kz_inc": float(md["kz_inc"]), "kx0": md["kx0"],
           "ky0": md["ky0"]}
    for k in ("rx", "ry"):
        arr = np.asarray(md[k])[:, idx]
        res[k] = [[[float(z.real), float(z.imag)] for z in row]
                  for row in arr]
    cm, eps = circle_explicit(layout)
    res["fingerprint_equal"] = st.cmap.fingerprint == cm.fingerprint
    if both:
        b = stack_solve(M, th, ph, cmap=cm, eps=eps)
        res["explicit"] = compare(st, a, b)
    pb = loadb(f"b10_{layout}_{tag(th, ph)}_M{M}.json")
    if pb is not None:
        res["phaseB_RT_maxdiff"] = float(max(
            np.max(np.abs(np.asarray(pb["R"]) - R[:, idx])),
            np.max(np.abs(np.asarray(pb["T"]) - T[:, idx]))))
    print({k: v for k, v in res.items() if k not in ("rx", "ry", "kx", "ky",
                                                      "kz_ref")}, flush=True)
    C.dump(f"c_oblique_{layout}_{tag(th, ph)}_M{M}.json", res)


def fem_ref():
    with open(os.path.join(C.HERE, "..", "fem", "summary.json")) as f:
        d = json.load(f)
    assert d["best_from"] == ["h1.0_e20 p4", "h0.8_e30 p4", "h1.0 p6"]
    assert abs(d["RplusT"]["value"] - 1.0) < 1e-6
    ref, dev = {}, {}
    for side in ("R", "T"):
        for key, rec in d[side].items():
            for mn in ORD_FEM[key]:
                ref[(side, mn)] = rec["value"]
                dev[(side, mn)] = rec["maxdev"]
    return ref, max(dev.values())


def dist_fem(vec, ref):
    """``vec`` is C.vec: R[:, ORD9] (2 x 9) then T[:, ORD9]; FEM = row 1."""
    v = np.asarray(vec)
    Rte = v[:18].reshape(2, 9)[1]
    Tte = v[18:].reshape(2, 9)[1]
    out = 0.0
    for (side, mn), val in ref.items():
        k = C.ORD9.index(mn)
        out = max(out, abs((Rte if side == "R" else Tte)[k] - val))
    return out


def kidx(r, mn):
    return [list(v) for v in r["orders"]].index(list(mn))


def rt_by_order(r):
    return {tuple(mn): (np.asarray(r["R"])[:, i], np.asarray(r["T"])[:, i])
            for i, mn in enumerate(r["orders"])}


def jones_block(r, k):
    kx0, ky0, kzi = r["kx0"], r["ky0"], r["kz_inc"]
    A = np.array([[complex(*r["rx"][c][k]) for c in (0, 1)],
                  [complex(*r["ry"][c][k]) for c in (0, 1)]])
    kxo, kyo = r["kx"][k], r["ky"][k]
    kzo = complex(*r["kz_ref"][k])
    if abs(kzo.imag) > 1e-12 or kzo.real <= 0:
        return None
    kzo = kzo.real
    Gin = np.eye(2) + np.outer([kx0, ky0], [kx0, ky0]) / kzi ** 2
    Wout = (kzo / kzi) * (np.eye(2) + np.outer([kxo, kyo], [kxo, kyo])
                          / kzo ** 2)

    def msqrt(S, inv=False):
        w, V = np.linalg.eigh(S)
        d = w ** (-0.5 if inv else 0.5)
        return (V * d) @ V.conj().T
    return msqrt(Wout) @ A @ msqrt(Gin, inv=True)


def reverse_angles(th_deg, ph_deg, m, n):
    st = np.sin(np.deg2rad(th_deg))
    kx = st * np.cos(np.deg2rad(ph_deg)) + m * C.WL / C.P
    ky = st * np.sin(np.deg2rad(ph_deg)) + n * C.WL / C.P
    kx, ky = -kx, -ky
    return (float(np.rad2deg(np.arcsin(np.hypot(kx, ky)))) + 0.0,
            float(np.rad2deg(np.arctan2(ky, kx))) + 0.0)


def summary():
    ref, spread = fem_ref()
    out = {"env": C.env_record(), "fem_spread": spread, "circle": {},
           "fillet": {}, "reciprocity": {}, "oblique": {}}
    files = sorted(os.listdir(C.HERE))
    for fn in files:
        if fn.startswith("c_circle_"):
            r = json.load(open(os.path.join(C.HERE, fn)))
            pb = loadb(f"b4_circle_{r['layout']}_M{r['M']}.json")
            out["circle"].setdefault(r["layout"], []).append({
                "M": r["M"], "dist_fem": dist_fem(r["vec"], ref),
                "phaseB_dist_fem": (dist_fem(pb["vec"], ref) if pb else None),
                "phaseB_vec_maxdiff": r.get("phaseB_vec_maxdiff"),
                "fingerprint_equal": r["fingerprint_equal"],
                "bytes_equal": r.get("explicit", {}).get("bytes_equal"),
                "closure": r["closure"], "t": r["t"]})
        elif fn.startswith("c_fillet_"):
            r = json.load(open(os.path.join(C.HERE, fn)))
            out["fillet"].setdefault(str(r["ratio"]), []).append({
                "M": r["M"], "R00_te": r["R00"][1], "T00_te": r["T00"][1],
                "phaseB_R00_T00_diff": r.get("phaseB_R00_T00_diff"),
                "fingerprint_equal": r["fingerprint_equal"],
                "bytes_equal": r.get("explicit", {}).get("bytes_equal"),
                "closure": r["closure"], "t": r["t"]})
    for v in out["circle"].values():
        v.sort(key=lambda z: z["M"])
    for v in out["fillet"].values():
        v.sort(key=lambda z: z["M"])
        for a, b in zip(v[:-1], v[1:]):
            a["dT00_next"] = abs(a["T00_te"] - b["T00_te"])
            a["dR00_next"] = abs(a["R00_te"] - b["R00_te"])
    sq = loadb("b6_square_M12.json")
    if sq is not None:
        out["square_M12_R00_T00"] = [sq["R00"][1], sq["T00"][1]]
    # oblique: reciprocity on this build's rungs, Phase B's beside them
    obl = {}
    for fn in files:
        if fn.startswith("c_oblique_"):
            r = json.load(open(os.path.join(C.HERE, fn)))
            obl.setdefault((r["map"], tag(r["theta"], r["phi"])), {})[
                r["M"]] = r
            pbf = loadb(f"b10_{r['map']}_{tag(r['theta'], r['phi'])}"
                        f"_M{r['M']}.json")
            dB = None
            if pbf is not None:
                a, b = rt_by_order(r), rt_by_order(pbf)
                dB = float(max(max(np.max(np.abs(a[k][0] - b[k][0])),
                                   np.max(np.abs(a[k][1] - b[k][1])))
                               for k in a))
            out["oblique"].setdefault(f"{r['map']}_{tag(r['theta'], r['phi'])}",
                                      []).append({
                "M": r["M"], "closure": r["closure"],
                "phaseB_RT_maxdiff": dB,
                "fingerprint_equal": r["fingerprint_equal"],
                "bytes_equal": r.get("explicit", {}).get("bytes_equal")})
    pairs = [((25.0, 0.0), (-1, 0)), ((25.0, 40.0), (-1, 0)),
             ((25.0, 40.0), (0, -1))]
    for kind in ("c3", "c5"):
        for (th, ph), (m, n) in pairs:
            tr, pr = reverse_angles(th, ph, m, n)
            fwd = obl.get((kind, tag(th, ph)), {})
            rev = obl.get((kind, tag(tr, pr)), {})
            rows = []
            for M in sorted(fwd):
                if M not in rev:
                    continue
                Nf = jones_block(fwd[M], kidx(fwd[M], (m, n)))
                Nr = jones_block(rev[M], kidx(rev[M], (m, n)))
                if Nf is None or Nr is None:
                    continue
                sf = np.linalg.svd(Nf, compute_uv=False)
                sr = np.linalg.svd(Nr, compute_uv=False)
                Nw = jones_block(rev[M], kidx(rev[M], (0, 0)))
                sw = (np.linalg.svd(Nw, compute_uv=False) if Nw is not None
                      else np.full(2, np.nan))
                pf = loadb(f"b10_{kind}_{tag(th, ph)}_M{M}.json")
                pr_ = loadb(f"b10_{kind}_{tag(tr, pr)}_M{M}.json")
                pbr = None
                if pf is not None and pr_ is not None:
                    pbr = float(np.max(np.abs(
                        np.linalg.svd(jones_block(pf, kidx(pf, (m, n))),
                                      compute_uv=False)
                        - np.linalg.svd(jones_block(pr_, kidx(pr_, (m, n))),
                                        compute_uv=False))))
                rows.append({"M": M, "recip": float(np.max(np.abs(sf - sr))),
                             "wrong_pair": float(np.max(np.abs(sf - sw))),
                             "phaseB_recip": pbr})
            if rows:
                out["reciprocity"][f"{kind}_{tag(th, ph)}_({m},{n})"] = rows
    print(json.dumps(out, indent=1, default=float))
    C.dump("c_ladders_summary.json", out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    both = sys.argv[-1] == "both"
    if cmd == "circle":
        circle(sys.argv[2], int(sys.argv[3]), both)
    elif cmd == "fillet":
        fillet(float(sys.argv[2]), int(sys.argv[3]), both)
    elif cmd == "oblique":
        oblique(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]),
                float(sys.argv[5]), both)
    else:
        summary()

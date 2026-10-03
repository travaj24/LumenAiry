"""V3 -- the two gauge constants under maps that are NOT 180-degree
symmetric, with a "chiral" (general-direction) out-of-plane tensor.

usage: python v3_gauge.py film <map> <M>
       python v3_gauge.py same <M>
       python v3_gauge.py pillar <M> <k>

Arms (each through the shipped code, restored afterwards):
  ok        the shipped constants;
  rot+      _OOP_ROT_SIGN = +1 everywhere (mapped and unmapped);
  h+i       _OOP_H_GAUGE = +1j everywhere;
  h+1       _OOP_H_GAUGE = +1 everywhere;
  maprot    the rotation sign NOT applied to the MAPPED out-of-plane weights
            only (_stag_scale_weight returns W) -- a map-DEPENDENT rotation
            gauge, the hypothesis the builder's argument excludes;
  maph      the H gauge conjugated on MAPPED out-of-plane regions only.

film:   a uniform DIRGEN slab under an asymmetric map vs the own (eps, mu)
        oracle at normal / oblique / conical (R/T, Jr, Jt).
same:   the SAME DEVICE two ways -- a C2-BROKEN staircase-triangle of DIRGEN
        on a 4 x 4 grid, unmapped, and under a 4 x 4 transfinite map whose
        moved vertices all sit INSIDE one material (so every material wall
        is unchanged): normal incidence, zeroth-order Jones (both) and the
        R / T vector, for every arm in both solvers.  A map-independent gauge
        <=> mapped and unmapped converge to the same device under EVERY arm
        that is applied to both; the map-only arms must break that.
pillar: an OFF-CENTRE DIRGEN disk under its own 3 x 3 circle map vs the
        shipped OOP solver's k-staircase of the same disk, normal incidence,
        per arm.
"""
import sys
import warnings

import _ve1common as V
import numpy as np

TS, CM = V.TS, V.CM


class arm:
    def __init__(self, name):
        self.name = name
        self.undo = []

    def __enter__(self):
        n = self.name
        if n == "rot+":
            self._set(TS, "_OOP_ROT_SIGN", 1.0)
        elif n == "h+i":
            self._set(TS, "_OOP_H_GAUGE", 1j)
        elif n == "h+1":
            self._set(TS, "_OOP_H_GAUGE", 1.0)
        elif n == "maprot":
            self._set(TS, "_stag_scale_weight", lambda W, s: W)
        elif n == "maph":
            orig = TS._region_modes_oop

            def f(solver, **kw):
                m = orig(solver, **kw)
                if solver.cmap is None:
                    return m
                c = np.conj(TS._OOP_H_GAUGE) / TS._OOP_H_GAUGE
                return (m[0], c * m[1], m[2], m[3], c * m[4], m[5])
            self._set(TS, "_region_modes_oop", f)
            self._set(V.SP, "_region_modes_oop", f)
        return self

    def _set(self, obj, nm, val):
        self.undo.append((obj, nm, getattr(obj, nm)))
        setattr(obj, nm, val)

    def __exit__(self, *a):
        for obj, nm, val in reversed(self.undo):
            setattr(obj, nm, val)


ARMS = ["ok", "rot+", "h+i", "h+1", "maprot", "maph"]
mode = sys.argv[1]

if mode == "film":
    mapname, M = sys.argv[2], int(sys.argv[3])
    out = {}
    for mo in ("n", "o", "c"):
        for a in ARMS:
            if mapname == "none" and a in ("maprot", "maph"):
                continue
            with arm(a):
                try:
                    r = V.slab_run(V.DIRGEN, V.make_map(mapname,
                                                        V.SLAB["P"]), M, mo)
                    out[f"{mo}_{a}"] = {k: r[k] for k in ("dRT", "dJr",
                                                          "dJt", "clo")}
                except Exception as exc:
                    out[f"{mo}_{a}"] = {"error": repr(exc)[:300]}
            print(mapname, M, mo, a, out[f"{mo}_{a}"], flush=True)
    V.dump(f"v3_film_{mapname}_M{M}.json", out)

P3 = 1.0
TRI = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)]


def tri_cell():
    e = np.broadcast_to(V.EYE, (4, 4, 3, 3)).copy()
    for c in TRI:
        e[c] = V.DIRGEN
    return e


def tri_map():
    w = np.linspace(0.0, P3, 5)
    Vv = np.stack(np.meshgrid(w, w, indexing="ij"), axis=-1)
    for (i, j), (dx, dy) in {(1, 1): (0.06, -0.04), (3, 3): (-0.05, 0.03),
                             (2, 3): (0.04, 0.05)}.items():
        Vv[i, j] += (dx * P3, dy * P3)
    return CM.TransfiniteMap(w, w, Vv, None)


def solve_same(cm, M, th=0.0, ph=0.0):
    st = V.PMM2DStackPure(P3, P3, n_superstrate=1.0, n_substrate=1.5,
                          n_modes=M, n_orders=2, cmap=cm)
    st.add_layer(0.4, eps_cell=tri_cell())
    st.set_source(1.0, theta=th, phi=ph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve(jones=True)
    return dict(o=np.asarray(o), R=np.asarray(R), T=np.asarray(T),
                Jr=np.asarray(J), Jt=V.jt_of(st))


def dist(a, b):
    return dict(RT=float(max(np.abs(a["R"] - b["R"]).max(),
                             np.abs(a["T"] - b["T"]).max())),
                Jr=float(np.abs(a["Jr"] - b["Jr"]).max()),
                Jt=float(np.abs(a["Jt"] - b["Jt"]).max()))


if mode == "same":
    M = int(sys.argv[2])
    cm = tri_map()
    res = {}
    for a in ARMS:
        with arm(a):
            res[a] = {}
            for nm, c in (("none", None), ("map", cm)):
                try:
                    res[a][nm] = solve_same(c, M)
                except Exception as exc:
                    res[a][nm] = {"error": repr(exc)[:200]}
    out = {}
    for a in ARMS:
        if "error" in res[a]["none"] or "error" in res[a]["map"]:
            out[a] = {"refused": {k: v.get("error") for k, v in
                                  res[a].items() if "error" in v}}
            print(M, a, out[a], flush=True)
            continue
        out[a] = {"map_vs_unmapped": dist(res[a]["map"], res[a]["none"]),
                  "unmapped_vs_ok": dist(res[a]["none"], res["ok"]["none"]),
                  "map_vs_ok": dist(res[a]["map"], res["ok"]["map"]),
                  "clo_map": float(np.abs(res[a]["map"]["R"].sum(1)
                                          + res[a]["map"]["T"].sum(1)
                                          - 1).max())}
        print(M, a, out[a], flush=True)
    V.dump(f"v3_same_M{M}.json", out)

if mode == "pillar":
    M, k = int(sys.argv[2]), int(sys.argv[3])
    f = dict(V.PIL)
    cen = (0.47 * f["P"], 0.56 * f["P"])
    r0 = 0.3 * f["P"]

    def curved():
        cm = CM._circle_map_3x3(f["P"], r0, center=cen)[0]
        e = np.broadcast_to(V.EYE, (3, 3, 3, 3)).copy()
        e[1, 1] = V.DIRGEN
        st = V.PMM2DStackPure(f["P"], f["P"], n_superstrate=1.0,
                              n_substrate=1.5, n_modes=M, n_orders=3,
                              cmap=cm)
        st.add_layer(f["DEP"], eps_cell=e)
        st.set_source(1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.solve(jones=True)
        return dict(v=V.vec36(o, R, T), Jr=np.asarray(J), Jt=V.jt_of(st))

    def stair():
        cx, cy = cen
        wx = np.array([0.0] + sorted([cx - r0 * i / k for i in range(1, k + 1)]
                                     + [cx + r0 * i / k
                                        for i in range(1, k + 1)])
                      + [f["P"]])
        wy = np.array([0.0] + sorted([cy - r0 * i / k for i in range(1, k + 1)]
                                     + [cy + r0 * i / k
                                        for i in range(1, k + 1)])
                      + [f["P"]])
        mx, my = 0.5 * (wx[:-1] + wx[1:]), 0.5 * (wy[:-1] + wy[1:])
        e = np.broadcast_to(V.EYE, (len(mx), len(my), 3, 3)).copy()
        for i in range(len(mx)):
            for j in range(len(my)):
                if (mx[i] - cx) ** 2 + (my[j] - cy) ** 2 < r0 ** 2:
                    e[i, j] = V.DIRGEN
        st = V.PMM2DStackPure(f["P"], f["P"], n_superstrate=1.0,
                              n_substrate=1.5, n_modes=M, n_orders=3,
                              layer_grids="per-layer")
        st.add_layer(f["DEP"], eps_cell=e, x_walls=wx, y_walls=wy)
        st.set_source(1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.solve(jones=True)
        return dict(v=V.vec36(o, R, T), Jr=np.asarray(J), Jt=V.jt_of(st))

    res = {}
    for a in ("ok", "rot+", "h+i", "maprot", "maph"):
        with arm(a):
            try:
                res[a] = {"curved": curved(), "stair": stair()}
            except Exception as exc:
                res[a] = {"error": repr(exc)[:300]}
                print(a, "error", exc)

    def d(x, y):
        return dict(v=float(np.abs(x["v"] - y["v"]).max()),
                    Jr=float(np.abs(x["Jr"] - y["Jr"]).max()),
                    Jt=float(np.abs(x["Jt"] - y["Jt"]).max()))
    out = {}
    for a, r in res.items():
        if "error" in r:
            out[a] = r
            continue
        out[a] = {"curved_vs_stair": d(r["curved"], r["stair"]),
                  "curved_vs_ok": d(r["curved"], res["ok"]["curved"]),
                  "stair_vs_ok": d(r["stair"], res["ok"]["stair"])}
        print(M, k, a, out[a], flush=True)
    V.dump(f"v3_pillar_M{M}_k{k}.json", out)

"""B10 -- OBLIQUE (theta = 25 deg, phi = 0) and CONICAL (theta = 25 deg,
phi = 40 deg) incidence on the circular pillar under the curved maps (not
probed by the planner).  Fixture of plan 3.3 otherwise.

  rung K M th ph [nocof]  -- one solve (K = c3 | c5; angles in degrees):
                     per-order R / T for both lab inputs, the per-order
                     reflected / transmitted field amplitudes, closure
                     -> b10_<K>_t<th>_p<ph>_M<M>[_nocof].json
  rcwa th ph n          -- the shipped 2-D RCWA with the EXACT disk form
                     factor (Laurent), n orders per side (2n + 1 per axis),
                     te and tm -> b10_rcwa_t<th>_p<ph>_n<n>.json
  summary               -> b10_oblique.json

Derived quantities (summary):
  * convergence: rung-to-rung change of every order of both inputs;
  * closure |sum R + sum T - 1| (lossless);
  * phi = 0: the y-MIRROR identity per order, R(m, n) = R(m, -n) for both lab
    inputs (the structure and the incidence are symmetric under y -> -y);
  * the POLARIZATION-SUMMED efficiency of each order, sum over an
    ORTHONORMAL pair of inputs, i.e. ||N||_F^2 of the power-normalized 2 x 2
    Jones block N = W_out^1/2 A G_in^-1/2 (A: lab input E_t -> order E_t,
    G_in / W_out: the power metrics of the input / output plane waves);
  * RECIPROCITY: the singular values of N for the channel pair (incident
    k_in -> order k_mn) equal those of the reversed pair (incident -k_mn ->
    the order that leaves along -k_in, which is the same (m, n)); the
    reverse angles are computed here from the order's momentum.
"""
import os
import sys
import time

import _common as C
import numpy as np

ORDS = [(m, n) for n in (-1, 0, 1) for m in (-1, 0, 1)]


def tag(th, ph):
    return f"t{th:.4f}_p{ph:.4f}"


def reverse_angles(th_deg, ph_deg, m, n):
    """(theta', phi') in degrees of the incidence along -k_mn (superstrate
    n = 1): the reversed channel of (incident -> reflected order (m, n))."""
    st = np.sin(np.deg2rad(th_deg))
    kx = st * np.cos(np.deg2rad(ph_deg)) + m * C.WL / C.P
    ky = st * np.sin(np.deg2rad(ph_deg)) + n * C.WL / C.P
    kx, ky = -kx, -ky
    return (float(np.rad2deg(np.arcsin(np.hypot(kx, ky)))) + 0.0,
            float(np.rad2deg(np.arctan2(ky, kx))) + 0.0)


def rung(kind, M, th, ph, nocof=False):
    cm, eps = C.circle3() if kind == "c3" else C.circle5()
    t0 = time.perf_counter()
    st = C.PMM2DStackPure(C.P, C.P, n_superstrate=C.N_SUP,
                          n_substrate=C.N_SUB, n_modes=M, n_orders=3,
                          cmap=cm)
    st.add_layer(C.DEPTH, eps_cell=eps)
    st.set_source(C.WL, theta=np.deg2rad(th), phi=np.deg2rad(ph))
    if nocof:
        with C.no_cofactor():
            o, R, T, _J = st.solve()
    else:
        o, R, T, _J = st.solve()
    o = np.asarray(o)
    md = st._modal
    idx = C.idx_of(o, ORDS)
    res = {"env": C.env_record(), "map": kind, "M": M, "theta": th,
           "phi": ph, "nocof": nocof, "t": time.perf_counter() - t0,
           "orders": [list(v) for v in ORDS],
           "R": R[:, idx].tolist(), "T": T[:, idx].tolist(),
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
           "kx": [float(md["kx"][i]) for i in idx],
           "ky": [float(md["ky"][i]) for i in idx],
           "kz_ref": [[float(np.real(md["kz_ref"][i])),
                       float(np.imag(md["kz_ref"][i]))] for i in idx],
           "kz_inc": float(md["kz_inc"]), "kx0": md["kx0"], "ky0": md["ky0"]}
    for k in ("rx", "ry"):
        a = np.asarray(md[k])[:, idx]
        res[k] = [[[float(z.real), float(z.imag)] for z in row] for row in a]
    sfx = "_nocof" if nocof else ""
    C.dump(f"b10_{kind}_{tag(th, ph)}_M{M}{sfx}.json", res)
    print(kind, M, th, ph, nocof, f"clo={res['closure']:.1e} "
          f"t={res['t']:.0f}s", flush=True)


def rcwa(th, ph, n):
    from lumenairy.elements.rcwa.twod import rcwa_efficiency_2d_shapes
    shp = [{"shape": "disk", "eps": C.EPS_P, "radius": C.R_CIRC,
            "center": (C.P / 2, C.P / 2)}]
    res = {"env": C.env_record(), "theta": th, "phi": ph, "n": n}
    t0 = time.perf_counter()
    for pol in ("te", "tm"):
        o, R, T = rcwa_efficiency_2d_shapes(
            C.P, C.P, 1.0, shp, C.N_SUB, C.N_SUP, C.DEPTH, C.WL,
            theta=np.deg2rad(th), phi=np.deg2rad(ph), polarization=pol,
            n_orders_x=n, n_orders_y=n)
        o = np.asarray(o)
        idx = C.idx_of(o, ORDS)
        res[pol] = {"R": np.asarray(R)[idx].tolist(),
                    "T": np.asarray(T)[idx].tolist()}
    res["t"] = time.perf_counter() - t0
    C.dump(f"b10_rcwa_{tag(th, ph)}_n{n}.json", res)
    print("rcwa", th, ph, n, f"t={res['t']:.0f}", flush=True)


def jones_block(r, k):
    """Power-normalized 2 x 2 reflection Jones block N of order k of rung
    file r (orthonormal input and output polarization bases)."""
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


def summary():
    res = {"env": C.env_record(), "ladders": {}, "reciprocity": {},
           "rcwa": {}}
    files = sorted(f for f in os.listdir(C.HERE) if f.startswith("b10_c"))
    groups = {}
    for fn in files:
        r = C.load(fn)
        key = f"{r['map']}_{tag(r['theta'], r['phi'])}" + (
            "_nocof" if r["nocof"] else "")
        groups.setdefault(key, []).append(r)
    for key, rs in groups.items():
        rs.sort(key=lambda r: r["M"])
        rows = []
        for r in rs:
            Rm, Tm = np.array(r["R"]), np.array(r["T"])
            row = {"M": r["M"], "closure": r["closure"], "t": r["t"]}
            if abs(r["phi"]) < 1e-12:
                # y-mirror: order (m, n) <-> (m, -n)
                mir = 0.0
                for k, (m, n) in enumerate(ORDS):
                    j = ORDS.index((m, -n))
                    mir = max(mir, float(np.max(np.abs(Rm[:, k] - Rm[:, j]))),
                              float(np.max(np.abs(Tm[:, k] - Tm[:, j]))))
                row["mirror_y"] = mir
            row["vec"] = np.concatenate([Rm.ravel(), Tm.ravel()]).tolist()
            fro = {}
            for k, mn in enumerate(ORDS):
                N = jones_block(r, k)
                if N is not None:
                    fro[f"{mn[0]},{mn[1]}"] = float(np.sum(np.abs(N) ** 2))
            row["polsum_R"] = fro
            rows.append(row)
        for a, b in zip(rows[:-1], rows[1:]):
            a["d_next"] = float(np.max(np.abs(np.array(a["vec"])
                                              - np.array(b["vec"]))))
        res["ladders"][key] = rows
    # reciprocity: pair forward (th, ph) order (m, n) with the reverse run
    pairs = [((25.0, 0.0), (-1, 0)), ((25.0, 40.0), (-1, 0)),
             ((25.0, 40.0), (0, -1))]
    for kind in ("c3", "c5"):
        for (th, ph), (m, n) in pairs:
            tr, pr = reverse_angles(th, ph, m, n)
            fwd = groups.get(f"{kind}_{tag(th, ph)}", [])
            rev = {r["M"]: r for r in groups.get(f"{kind}_{tag(tr, pr)}", [])}
            k = ORDS.index((m, n))
            rows = []
            for r in fwd:
                q = rev.get(r["M"])
                if q is None:
                    continue
                Nf = jones_block(r, k)
                Nr = jones_block(q, k)
                sf = np.linalg.svd(Nf, compute_uv=False)
                sr = np.linalg.svd(Nr, compute_uv=False)
                # WRONG pairing (fail-before): the reverse run's SPECULAR
                # channel (0, 0) in place of the reciprocal one -- shows the
                # resolution of the singular-value comparison
                kw = ORDS.index((0, 0))
                Nw = jones_block(q, kw)
                sw = (np.linalg.svd(Nw, compute_uv=False) if Nw is not None
                      else None)
                rows.append({"M": r["M"], "sv_fwd": sf.tolist(),
                             "sv_rev": sr.tolist(),
                             "recip": float(np.max(np.abs(sf - sr))),
                             "wrong_pair": (float(np.max(np.abs(sf - sw)))
                                            if sw is not None else None)})
            res["reciprocity"][f"{kind}_{tag(th, ph)}_({m},{n})"] = {
                "reverse_deg": [tr, pr], "rows": rows}
    for fn in sorted(f for f in os.listdir(C.HERE)
                     if f.startswith("b10_rcwa_")):
        r = C.load(fn)
        res["rcwa"].setdefault(tag(r["theta"], r["phi"]), []).append(
            {"n": r["n"], "te": r["te"], "tm": r["tm"]})
    # the engineered defect (no cofactor) through the same reciprocity gate
    for th, ph in ((25.0, 0.0),):
        tr, pr = reverse_angles(th, ph, -1, 0)
        f = groups.get(f"c3_{tag(th, ph)}_nocof", [])
        r = {q["M"]: q for q in groups.get(f"c3_{tag(tr, pr)}_nocof", [])}
        k = ORDS.index((-1, 0))
        rows = []
        for a in f:
            b = r.get(a["M"])
            if b is not None:
                sf = np.linalg.svd(jones_block(a, k), compute_uv=False)
                sr = np.linalg.svd(jones_block(b, k), compute_uv=False)
                rows.append({"M": a["M"], "recip": float(np.max(np.abs(
                    sf - sr)))})
        res["reciprocity"][f"c3_{tag(th, ph)}_(-1,0)_NOCOFACTOR"] = {
            "reverse_deg": [tr, pr], "rows": rows}
    # two independent topologies at their top rungs, and the RCWA reference
    # (exact disk form factor), compared on the polarization-summed
    # reflectance of each propagating order (basis-free)
    for th, ph in ((25.0, 0.0), (25.0, 40.0)):
        key = tag(th, ph)
        c3 = groups.get(f"c3_{key}", [])
        c5 = groups.get(f"c5_{key}", [])
        if not (c3 and c5):
            continue
        a, b = c3[-1], c5[-1]
        pa = {kk: v for kk, v in zip(ORDS, [jones_block(a, i) for i in
                                            range(len(ORDS))])}
        pb = {kk: v for kk, v in zip(ORDS, [jones_block(b, i) for i in
                                            range(len(ORDS))])}
        fro = {kk: (float(np.sum(np.abs(pa[kk]) ** 2)),
                    float(np.sum(np.abs(pb[kk]) ** 2)))
               for kk in ORDS if pa[kk] is not None}
        top = {"c3_M": a["M"], "c5_M": b["M"],
               "topologies": max(abs(x - y) for x, y in fro.values())}
        rc = []
        for q in res["rcwa"].get(key, []):
            dd = 0.0
            for i, kk in enumerate(ORDS):
                if kk in fro:
                    dd = max(dd, abs(q["te"]["R"][i] + q["tm"]["R"][i]
                                     - fro[kk][0]))
            rc.append({"n": q["n"], "orders_per_axis": 2 * q["n"] + 1,
                       "polsumR_dist_c3_top": dd})
        rc.sort(key=lambda z: z["n"])
        top["rcwa"] = rc
        res.setdefault("tops", {})[key] = top
    for v in res["ladders"].values():
        for r in v:
            r.pop("vec")
    C.dump("b10_oblique.json", res)
    print(res["reciprocity"])


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "rung":
        rung(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]),
             float(sys.argv[5]), len(sys.argv) > 6 and sys.argv[6] == "nocof")
    elif mode == "rcwa":
        rcwa(float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
    elif mode == "reverse":
        print(reverse_angles(float(sys.argv[2]), float(sys.argv[3]),
                             int(sys.argv[4]), int(sys.argv[5])))
    elif mode == "summary":
        summary()
    else:
        raise SystemExit(mode)

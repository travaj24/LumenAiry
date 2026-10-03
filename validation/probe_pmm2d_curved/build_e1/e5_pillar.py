"""E1-5 / E1-8 -- an OUT-OF-PLANE tensor circular pillar: a uniaxial disk
(n_o 1.5, n_e 1.8, director tilted 30 deg OUT OF the plane -- 60 deg from z
-- at azimuth 30 deg; ``_e1common.OOP30``) of radius 0.36 in air, period 1.2,
depth 0.5, lambda 1, air above, n = 1.45 below (Phase D's D4 fixture with the
in-plane LC30 replaced by the out-of-plane director).  Families:

* curved -- the 3 x 3 (c3) and 5 x 5 (c5) circle maps (Phase E1: the
  first-order generator with permeability blocks);
* rcwa   -- the shipped 2-D tensor RCWA (``rcwa_jones_2d``, Laurent) with the
  EXACT disk form factor, by patching the ONE convolution builder
  ``_eps_convolution_2d`` for the duration of the call (Phase D's d4 device,
  verbatim: every tensor component's Toeplitz matrix is bg_ij delta + (d_ij
  - bg_ij) F_disk).  ``fff_nv`` is NOT used: its validated-scope gate refuses
  a disk (non-separable geometry), and it builds its factorization from
  pixels, which would put the O(1/S) staircase back in;
* stair  -- the SHIPPED out-of-plane staggered solver (no map) on the
  planner's 4k-step staircases (walls at c +- r i / k), the tensor painted
  per cell.

Record: the 36-entry R / T vector of the nine orders |m|, |n| <= 1 for both
inputs, the closure, the reflected amplitudes of five orders (for the
reciprocity of E1-8) and both order-0 Jones matrices.

usage: python e5_pillar.py curved <c3|c5> <M> [theta phi [mat]]
       python e5_pillar.py rcwa <n_orders> [theta phi]
       python e5_pillar.py stair <k> <M>
       python e5_pillar.py summary
       mat in {oop30 (default), nonrec30 (the non-reciprocal control)}
"""
import json
import os
import sys
import time
from contextlib import contextmanager

import _common as C
import _e1common as E
import numpy as np

P, R0, DEP, WL, NSUB, NSUP = 1.2, 0.36, 0.5, 1.0, 1.45, 1.0
CTR = P / 2
ORDS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
NONREC30 = np.array(E.OOP30, dtype=complex)
NONREC30[0, 2] = E.OOP30[0, 2] + 0.25j
NONREC30[2, 0] = np.conj(NONREC30[0, 2])
MATS = {"oop30": E.OOP30, "nonrec30": NONREC30}


def tag(th, ph):
    return f"t{th:.4f}_p{ph:.4f}"


def disk_cells(kind, t33):
    cm = E.make_map(kind, P) if kind not in ("c3", "c5") else (
        E.CM._circle_map_3x3(P, R0)[0] if kind == "c3"
        else E.CM._circle_map_5x5(P, R0)[0])
    N = cm.shape[0]
    eps = np.broadcast_to(np.eye(3, dtype=complex), (N, N, 3, 3)).copy()
    for c in ([(1, 1)] if N == 3 else [(i, j) for i in (1, 2, 3)
                                       for j in (1, 2, 3)]):
        eps[c] = t33
    return cm, eps


def record(st, o, R, T, J, extra):
    o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)
    md = st._modal if st is not None else None
    res = dict(extra)
    res.update({"vec": C.vec(o, R, T).tolist(),
                "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
                "Jr": np.asarray(J).tolist() if J is not None else None})
    if md is not None:
        idx = C.idx(o, ORDS)
        res["Jt"] = E.jones_t(st).tolist()
        res.update({"kx": [float(md["kx"][i]) for i in idx],
                    "ky": [float(md["ky"][i]) for i in idx],
                    "kz_ref": [[float(np.real(md["kz_ref"][i])),
                                float(np.imag(md["kz_ref"][i]))]
                               for i in idx],
                    "kz_inc": float(md["kz_inc"]), "kx0": md["kx0"],
                    "ky0": md["ky0"]})
        for k in ("rx", "ry"):
            a = np.asarray(md[k])[:, idx]
            res[k] = [[[float(z.real), float(z.imag)] for z in row]
                      for row in a]
    return res


def curved(kind, M, th=0.0, ph=0.0, mat="oop30"):
    cm, eps = disk_cells(kind, MATS[mat])
    t0 = time.perf_counter()
    st = E.PMM2DStackPure(P, P, n_superstrate=NSUP, n_substrate=NSUB,
                          n_modes=M, n_orders=3, cmap=cm)
    st.add_layer(DEP, eps_cell=eps)
    st.set_source(WL, theta=np.deg2rad(th), phi=np.deg2rad(ph))
    o, R, T, J = st.solve(jones=True)
    res = record(st, o, R, T, J, {
        "family": "curved", "map": kind, "M": M, "theta": th, "phi": ph,
        "mat": mat, "dof": 4 * (cm.shape[0] * (M - 1)) ** 2})
    res["t"] = time.perf_counter() - t0
    E.dump(f"e5_curved_{mat}_{kind}_{tag(th, ph)}_M{M}.json", res)
    print(kind, mat, M, th, ph, f"clo={res['closure']:.2e} "
          f"t={res['t']:.0f}", flush=True)


@contextmanager
def exact_disk(n):
    """Phase D's d4 patch: every component c of the pixel cell is bg + (d -
    bg) * disk-mask; return bg delta + (d - bg) F_exact."""
    from lumenairy.elements.rcwa import twod as RT
    orig = RT._eps_convolution_2d

    def conv(eps_cell, orders, nx, ny):
        a = np.asarray(eps_cell)
        S = a.shape[0]
        bg, d = complex(a[0, 0]), complex(a[S // 2, S // 2])
        Mx, My = int(nx), int(ny)
        ks = np.arange(-2 * Mx, 2 * Mx + 1)
        ls = np.arange(-2 * My, 2 * My + 1)
        KK, LL = np.meshgrid(ks, ls, indexing="ij")
        F = RT._shape_form_factor(
            {"shape": "disk", "radius": R0, "center": (CTR, CTR)},
            KK * 2 * np.pi / P, LL * 2 * np.pi / P, P, P)
        c = (d - bg) * F
        c[2 * Mx, 2 * My] += bg
        dm = orders[:, 0][:, None] - orders[:, 0][None, :]
        dn = orders[:, 1][:, None] - orders[:, 1][None, :]
        return c[dm + 2 * Mx, dn + 2 * My]
    RT._eps_convolution_2d = conv
    try:
        yield
    finally:
        RT._eps_convolution_2d = orig


def pixel_cell(n, t33):
    S = 4 * n + 8
    S += S % 2
    x = np.arange(S) * P / S
    X, Y = np.meshgrid(x, x, indexing="ij")
    m = (X - CTR) ** 2 + (Y - CTR) ** 2 < R0 ** 2
    cell = np.broadcast_to(np.eye(3, dtype=complex), (S, S, 3, 3)).copy()
    cell[m] = t33
    return cell


def rcwa(n, th=0.0, ph=0.0, mat="oop30"):
    from lumenairy.elements.rcwa import rcwa_jones_2d
    t0 = time.perf_counter()
    with exact_disk(n):
        o, R, T, J = rcwa_jones_2d(P, P, pixel_cell(n, MATS[mat]), NSUB, NSUP,
                                   DEP, WL, n_orders_x=n, n_orders_y=n,
                                   symmetry=False, formulation="laurent",
                                   theta=np.deg2rad(th), phi=np.deg2rad(ph))
    res = record(None, o, R, T, J, {"family": "rcwa", "n": n, "theta": th,
                                    "phi": ph, "mat": mat})
    res["t"] = time.perf_counter() - t0
    E.dump(f"e5_rcwa_{mat}_{tag(th, ph)}_n{n}.json", res)
    print("rcwa", n, f"clo={res['closure']:.2e} t={res['t']:.0f}",
          flush=True)


def stair(k, M, mat="oop30"):
    c = CTR
    inner = sorted([c - R0 * i / k for i in range(1, k + 1)]
                   + [c + R0 * i / k for i in range(1, k + 1)])
    w = np.array([0.0] + inner + [P])
    n = len(w) - 1
    mid = 0.5 * (w[:-1] + w[1:])
    eps = np.broadcast_to(np.eye(3, dtype=complex), (n, n, 3, 3)).copy()
    for i in range(n):
        for j in range(n):
            if (mid[i] - c) ** 2 + (mid[j] - c) ** 2 < R0 ** 2:
                eps[i, j] = MATS[mat]
    t0 = time.perf_counter()
    st = E.PMM2DStackPure(P, P, n_superstrate=NSUP, n_substrate=NSUB,
                          n_modes=M, n_orders=3, layer_grids="per-layer")
    st.add_layer(DEP, eps_cell=eps, x_walls=w, y_walls=w)
    st.set_source(WL)
    o, R, T, J = st.solve(jones=True)
    res = record(None, o, R, T, J, {"family": "stair", "k": k, "M": M,
                                    "walls": w.tolist(), "mat": mat})
    res["t"] = time.perf_counter() - t0
    E.dump(f"e5_stair_{mat}_k{k}_M{M}.json", res)
    print("stair", k, M, f"t={res['t']:.0f}", flush=True)


def reverse_angles(th_deg, ph_deg, m, n):
    st = np.sin(np.deg2rad(th_deg))
    kx = st * np.cos(np.deg2rad(ph_deg)) + m * WL / P
    ky = st * np.sin(np.deg2rad(ph_deg)) + n * WL / P
    kx, ky = -kx, -ky
    return (float(np.rad2deg(np.arcsin(np.hypot(kx, ky)))) + 0.0,
            float(np.rad2deg(np.arctan2(ky, kx))) + 0.0)


def jones_block(r, k):
    """Power-normalized 2 x 2 reflection Jones block of channel
    (incidence -> order ORDS[k]) (Phase D's d8 formula)."""
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


def _load(prefix):
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for fn in sorted(os.listdir(here)):
        if fn.startswith(prefix) and fn.endswith(".json"):
            out.append(json.load(open(os.path.join(here, fn))))
    return out


def summary():
    out = {}
    nt = tag(0.0, 0.0)
    c3 = sorted([r for r in _load("e5_curved_oop30_c3_" + nt)],
                key=lambda r: r["M"])
    c5 = sorted([r for r in _load("e5_curved_oop30_c5_" + nt)],
                key=lambda r: r["M"])
    ref = np.array(c3[-1]["vec"]) if c3 else None
    for key, rows in (("c3", c3), ("c5", c5)):
        out[key] = []
        for a, b in zip(rows, rows[1:] + [None]):
            row = {"M": a["M"], "dof": a["dof"], "closure": a["closure"],
                   "t": a["t"],
                   "to_c3_top": float(np.max(np.abs(np.array(a["vec"])
                                                    - ref)))}
            if b is not None:
                row["d_next"] = float(np.max(np.abs(np.array(a["vec"])
                                                    - np.array(b["vec"]))))
            out[key].append(row)
    if c3 and c5:
        out["c3_top_vs_c5_top"] = float(np.max(np.abs(
            np.array(c3[-1]["vec"]) - np.array(c5[-1]["vec"]))))
    rc = sorted(_load("e5_rcwa_oop30_" + nt), key=lambda r: r["n"])
    out["rcwa"] = [{"n": r["n"], "N": 2 * r["n"] + 1, "closure": r["closure"],
                    "t": r["t"],
                    "to_c3_top": float(np.max(np.abs(np.array(r["vec"])
                                                     - ref)))}
                   for r in rc]
    rich = []
    for a, b in zip(rc, rc[1:]):
        Na, Nb = 2 * a["n"] + 1, 2 * b["n"] + 1
        va, vb = np.array(a["vec"]), np.array(b["vec"])
        ext = (Nb * vb - Na * va) / (Nb - Na)
        rich.append({"pair": [Na, Nb],
                     "to_c3_top": float(np.max(np.abs(ext - ref)))})
    out["rcwa_richardson_1_over_N"] = rich
    out["rcwa_distance_times_N"] = [
        {"N": 2 * r["n"] + 1, "dist_x_N": (2 * r["n"] + 1) * float(
            np.max(np.abs(np.array(r["vec"]) - ref)))} for r in rc]
    st = {}
    for r in _load("e5_stair_oop30_"):
        st.setdefault(r["k"], []).append(
            {"M": r["M"], "t": r["t"], "closure": r["closure"],
             "to_c3_top": float(np.max(np.abs(np.array(r["vec"]) - ref)))})
    out["stair"] = {str(k): sorted(v, key=lambda r: r["M"])
                    for k, v in sorted(st.items())}
    # E1-8: oblique / conical closure, the two topologies, reciprocity
    groups = {}
    for r in _load("e5_curved_"):
        groups.setdefault((r["mat"], r["map"], tag(r["theta"], r["phi"])),
                          {})[r["M"]] = r
    out["closure"] = {f"{m}_{k}_{t}": {str(M): r["closure"]
                                       for M, r in sorted(rs.items())}
                      for (m, k, t), rs in groups.items()}
    out["reciprocity"] = {}
    pairs = [((25.0, 0.0), (-1, 0)), ((25.0, 40.0), (-1, 0)),
             ((25.0, 40.0), (0, -1))]
    for mat in MATS:
        for kind in ("c3", "c5"):
            for (th, ph), (m, n) in pairs:
                tr, pr = reverse_angles(th, ph, m, n)
                fwd = groups.get((mat, kind, tag(th, ph)), {})
                rev = groups.get((mat, kind, tag(tr, pr)), {})
                k = ORDS.index((m, n))
                kw = ORDS.index((0, 0))
                rows = []
                for M in sorted(set(fwd) & set(rev)):
                    sf = np.linalg.svd(jones_block(fwd[M], k),
                                       compute_uv=False)
                    sr = np.linalg.svd(jones_block(rev[M], k),
                                       compute_uv=False)
                    sw = np.linalg.svd(jones_block(rev[M], kw),
                                       compute_uv=False)
                    rows.append({"M": M,
                                 "recip": float(np.max(np.abs(sf - sr))),
                                 "wrong_pair": float(np.max(np.abs(sf
                                                                   - sw)))})
                if rows:
                    out["reciprocity"][f"{mat}_{kind}_{tag(th, ph)}_"
                                       f"({m},{n})"] = {
                        "reverse": [tr, pr], "rows": rows}
    # the two topologies at oblique / conical (top rungs)
    topo = {}
    for (th, ph) in ((25.0, 0.0), (25.0, 40.0)):
        a = groups.get(("oop30", "c3", tag(th, ph)), {})
        b = groups.get(("oop30", "c5", tag(th, ph)), {})
        if a and b:
            Ma, Mb = max(a), max(b)
            topo[tag(th, ph)] = {"c3_M": Ma, "c5_M": Mb, "d": float(np.max(
                np.abs(np.array(a[Ma]["vec"]) - np.array(b[Mb]["vec"]))))}
    out["topologies_oblique"] = topo
    # RCWA at oblique / conical vs the curved top rung
    for (th, ph) in ((25.0, 0.0), (25.0, 40.0)):
        rr = sorted(_load(f"e5_rcwa_oop30_{tag(th, ph)}"),
                    key=lambda r: r["n"])
        a = groups.get(("oop30", "c3", tag(th, ph)), {})
        if rr and a:
            refo = np.array(a[max(a)]["vec"])
            out[f"rcwa_{tag(th, ph)}"] = [
                {"N": 2 * r["n"] + 1, "to_c3_top": float(np.max(np.abs(
                    np.array(r["vec"]) - refo)))} for r in rr]
    E.dump("e5_summary.json", out)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "curved":
        extra = ([float(sys.argv[4]), float(sys.argv[5])]
                 if len(sys.argv) > 5 else [0.0, 0.0])
        mat = sys.argv[6] if len(sys.argv) > 6 else "oop30"
        curved(sys.argv[2], int(sys.argv[3]), *extra, mat=mat)
    elif a == "rcwa":
        extra = ([float(sys.argv[3]), float(sys.argv[4])]
                 if len(sys.argv) > 4 else [0.0, 0.0])
        rcwa(int(sys.argv[2]), *extra)
    elif a == "stair":
        stair(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "summary":
        summary()
    else:
        raise SystemExit(__doc__)

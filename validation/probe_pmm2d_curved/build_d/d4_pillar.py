"""D4 -- a TENSOR circular pillar: an in-plane uniaxial disk (n_o 1.5, n_e
1.8, director at 30 deg to x) of radius 0.36 in air, period 1.2, depth 0.5,
lambda 1, air above, n = 1.45 below (the planning P3 fixture with the disk's
eps-4 replaced by the LC tensor).  Three families:

* curved -- the 3 x 3 (c3) and 5 x 5 (c5) circle maps (Phase D);
* rcwa   -- the shipped 2-D tensor RCWA (``rcwa_jones_2d``, Laurent / direct
  rule) with the EXACT disk form factor: its per-component Laurent Toeplitz
  matrices are built from ``_shape_form_factor`` (bg_ij delta + (d_ij -
  bg_ij) F) by patching the ONE convolution builder ``_eps_convolution_2d``
  for the duration of the call (the pixel cell only carries which component
  is which).  Self-check 'rcwa_scalar': the same patch on a SCALAR disk
  equals the shipped ``rcwa_efficiency_2d_shapes`` (Laurent) to round-off;
* stair  -- the shipped staggered solver (no map) on the planner's 4k-step
  staircases (walls at c +- r i / k), the tensor painted per cell.

Record: the 36-entry R / T vector of the nine orders |m|, |n| <= 1 for both
inputs (row 0 = E along x, 1 = along y) and the closure.

usage: python d4_pillar.py curved <c3|c5> <M>
       python d4_pillar.py rcwa <n_orders>
       python d4_pillar.py rcwa_scalar <n_orders>
       python d4_pillar.py stair <k> <M>
       python d4_pillar.py summary
"""
import json
import os
import sys
import time
from contextlib import contextmanager

import _common as C
import _dcommon as D
import numpy as np

P, R0, DEP, WL, NSUB, NSUP = 1.2, 0.36, 0.5, 1.0, 1.45, 1.0
CTR = P / 2


def disk_cells(kind):
    cm = D.make_map(kind, P)
    N = cm.shape[0]
    eps = np.broadcast_to(np.eye(3, dtype=complex), (N, N, 3, 3)).copy()
    for c in ([(1, 1)] if N == 3 else [(i, j) for i in (1, 2, 3)
                                       for j in (1, 2, 3)]):
        eps[c] = D.LC30
    return cm, eps


def curved(kind, M):
    cm, eps = disk_cells(kind)
    t0 = time.perf_counter()
    st = D.PMM2DStackPure(P, P, n_superstrate=NSUP, n_substrate=NSUB,
                          n_modes=M, n_orders=3, cmap=cm)
    st.add_layer(DEP, eps_cell=eps)
    st.set_source(WL)
    o, R, T, J = st.solve(jones=True)
    o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)
    res = {"family": "curved", "map": kind, "M": M,
           "dof": 2 * (cm.shape[0] * (M - 1)) ** 2,
           "vec": C.vec(o, R, T).tolist(),
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
           "t": time.perf_counter() - t0}
    D.dump(f"d4_curved_{kind}_M{M}.json", res)
    print(kind, M, f"clo={res['closure']:.2e} t={res['t']:.0f}", flush=True)


@contextmanager
def exact_disk(n):
    """Patch rcwa.twod._eps_convolution_2d: every component c of the pixel
    cell is bg + (d - bg) * disk-mask; return bg delta + (d - bg) F_exact."""
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


def rcwa(n, t33=None, tag="rcwa"):
    from lumenairy.elements.rcwa import rcwa_jones_2d
    t33 = D.LC30 if t33 is None else t33
    t0 = time.perf_counter()
    with exact_disk(n):
        o, R, T, J = rcwa_jones_2d(P, P, pixel_cell(n, t33), NSUB, NSUP, DEP,
                                   WL, n_orders_x=n, n_orders_y=n,
                                   symmetry=False, formulation="laurent")
    o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)
    res = {"family": tag, "n": n, "vec": C.vec(o, R, T).tolist(),
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
           "t": time.perf_counter() - t0}
    if tag == "rcwa_scalar":
        from lumenairy.elements.rcwa.twod import rcwa_efficiency_2d_shapes
        shp = [{"shape": "disk", "eps": 4.0, "radius": R0,
                "center": (CTR, CTR)}]
        rows = []
        for pol in ("tm", "te"):          # tm = E along x, te = E along y
            oo, RR, TT = rcwa_efficiency_2d_shapes(
                P, P, 1.0, shp, NSUB, NSUP, DEP, WL, polarization=pol,
                n_orders_x=n, n_orders_y=n)
            oo = np.asarray(oo)
            rows.append((oo, np.asarray(RR), np.asarray(TT)))
        Rs = np.stack([r[1] for r in rows])
        Ts = np.stack([r[2] for r in rows])
        res["shipped_shapes_vec"] = C.vec(rows[0][0], Rs, Ts).tolist()
        res["d_vs_shipped_shapes"] = float(np.max(np.abs(
            np.array(res["vec"]) - np.array(res["shipped_shapes_vec"]))))
        print("scalar self-check", res["d_vs_shipped_shapes"], flush=True)
    D.dump(f"d4_{tag}_n{n}.json", res)
    print(tag, n, f"clo={res['closure']:.2e} t={res['t']:.0f}", flush=True)


def stair(k, M):
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
                eps[i, j] = D.LC30
    t0 = time.perf_counter()
    st = D.PMM2DStackPure(P, P, n_superstrate=NSUP, n_substrate=NSUB,
                          n_modes=M, n_orders=3, layer_grids="per-layer")
    st.add_layer(DEP, eps_cell=eps, x_walls=w, y_walls=w)
    st.set_source(WL)
    o, R, T, J = st.solve(jones=True)
    o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)
    res = {"family": "stair", "k": k, "M": M, "walls": w.tolist(),
           "vec": C.vec(o, R, T).tolist(),
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
           "t": time.perf_counter() - t0}
    D.dump(f"d4_stair_k{k}_M{M}.json", res)
    print("stair", k, M, f"t={res['t']:.0f}", flush=True)


def summary():
    here = os.path.dirname(os.path.abspath(__file__))
    fam = {}
    for fn in sorted(os.listdir(here)):
        if fn.startswith("d4_") and fn.endswith(".json") and \
                "summary" not in fn:
            r = json.load(open(os.path.join(here, fn)))
            fam.setdefault(fn.rsplit("_", 1)[0], []).append(r)
    out = {}
    c3 = sorted(fam.get("d4_curved_c3", []), key=lambda r: r["M"])
    c5 = sorted(fam.get("d4_curved_c5", []), key=lambda r: r["M"])
    ref = np.array(c3[-1]["vec"]) if c3 else None
    for key, rows, knob in (("c3", c3, "M"), ("c5", c5, "M")):
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
    rc = sorted(fam.get("d4_rcwa", []), key=lambda r: r["n"])
    out["rcwa"] = [{"n": r["n"], "N": 2 * r["n"] + 1, "closure": r["closure"],
                    "t": r["t"],
                    "to_c3_top": float(np.max(np.abs(np.array(r["vec"])
                                                     - ref)))}
                   for r in rc]
    # Richardson in 1/N on the vector, and the local rate check
    rich = []
    for a, b in zip(rc, rc[1:]):
        Na, Nb = 2 * a["n"] + 1, 2 * b["n"] + 1
        va, vb = np.array(a["vec"]), np.array(b["vec"])
        ext = (Nb * vb - Na * va) / (Nb - Na)
        rich.append({"pair": [Na, Nb],
                     "to_c3_top": float(np.max(np.abs(ext - ref)))})
    out["rcwa_richardson_1_over_N"] = rich
    rates = []
    for a, b, c in zip(rc, rc[1:], rc[2:]):
        d1 = np.max(np.abs(np.array(a["vec"]) - np.array(b["vec"])))
        d2 = np.max(np.abs(np.array(b["vec"]) - np.array(c["vec"])))
        Na, Nb, Nc = (2 * r["n"] + 1 for r in (a, b, c))
        rates.append({"triple": [Na, Nb, Nc], "d1": float(d1),
                      "d2": float(d2),
                      "local_rate": float(np.log(d1 / d2)
                                          / np.log((Nc - Nb) / (Nb - Na)
                                                   * Nc / Na))})
    out["rcwa_rate"] = rates
    out["rcwa_distance_times_N"] = [
        {"N": 2 * r["n"] + 1, "dist_x_N": (2 * r["n"] + 1) * float(
            np.max(np.abs(np.array(r["vec"]) - ref)))} for r in rc]
    st = {}
    for key, rows in fam.items():
        if key.startswith("d4_stair_k"):
            for r in rows:
                st.setdefault(r["k"], []).append(
                    {"M": r["M"], "t": r["t"], "closure": r["closure"],
                     "to_c3_top": float(np.max(np.abs(np.array(r["vec"])
                                                      - ref)))})
    out["stair"] = {str(k): sorted(v, key=lambda r: r["M"])
                    for k, v in sorted(st.items())}
    if "d4_rcwa_scalar" in fam:
        out["rcwa_scalar_selfcheck"] = [
            {"n": r["n"], "d": r["d_vs_shipped_shapes"]}
            for r in fam["d4_rcwa_scalar"]]
    D.dump("d4_summary.json", out)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "curved":
        curved(sys.argv[2], int(sys.argv[3]))
    elif a == "rcwa":
        rcwa(int(sys.argv[2]))
    elif a == "rcwa_scalar":
        rcwa(int(sys.argv[2]), t33=4.0 * np.eye(3, dtype=complex),
             tag="rcwa_scalar")
    elif a == "stair":
        stair(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "summary":
        summary()

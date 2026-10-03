"""V4 -- the LC30 tensor pillar (in-plane uniaxial disk n_o 1.5 / n_e 1.8,
director 30 deg from x, r = 0.36, period 1.2, depth 0.5, lambda 1, air over
n = 1.45; the builder's D4 fixture) re-measured independently:

* curved KIND M     -- the c3 / c5 circle maps (to M = 12 on c3);
* rcwa N            -- the shipped 2-D tensor RCWA (Laurent) with an EXACT
  disk form factor computed HERE (own Bessel closed form, not the shipped
  ``_shape_form_factor``) patched into the ONE Toeplitz builder
  ``rcwa.twod._eps_convolution_2d``; N = orders per side, 2N + 1 per axis;
* rcwa_scalar N     -- the same patch on a SCALAR disk against the shipped
  ``rcwa_efficiency_2d_shapes`` (the patch's self-check);
* summary           -- distances to the top curved rung, distance x (2N+1),
  local rates, Richardson in 1/(2N+1).

Record: the 36-entry vector (orders |m|, |n| <= 1, R and T, both inputs)."""
import glob
import json
import os
import sys
import time
import warnings
from contextlib import contextmanager

import numpy as np
from _vdcommon import HERE, PMM2DStackPure, circle, dump, lc

P, R0, DEP, WL, NSUB, NSUP = 1.2, 0.36, 0.5, 1.0, 1.45, 1.0
CTR = P / 2
LC30 = lc(30)
ORD9 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1),
        (-1, -1)]


def vec(o, R, T):
    o = np.asarray(o)
    i = [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
         for m, n in ORD9]
    return np.concatenate([np.asarray(R)[:, i].ravel(),
                           np.asarray(T)[:, i].ravel()])


def curved(kind, M):
    cm = circle(P, kind, R0 / P)
    N = cm.shape[0]
    eps = np.broadcast_to(np.eye(3, dtype=complex), (N, N, 3, 3)).copy()
    for c in ([(1, 1)] if N == 3 else [(i, j) for i in (1, 2, 3)
                                       for j in (1, 2, 3)]):
        eps[c] = LC30
    t0 = time.perf_counter()
    st = PMM2DStackPure(P, P, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=3, cmap=cm)
    st.add_layer(DEP, eps_cell=eps)
    st.set_source(WL)
    o, R, T, _J = st.solve(jones=True)
    R, T = np.asarray(R), np.asarray(T)
    res = {"family": "curved", "map": kind, "M": M,
           "dof": 2 * (N * (M - 1)) ** 2, "vec": vec(o, R, T),
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
           "wall_s": time.perf_counter() - t0}
    dump(f"v4_curved_{kind}_M{M}.json", res)
    print(f"v4_curved_{kind}_M{M} clo {res['closure']:.2e} "
          f"wall {res['wall_s']:.0f}s")


def disk_ff(gx, gy):
    """(1/A) int_disk exp(-i G.r) d^2r, own closed form."""
    from scipy.special import j1
    q = np.hypot(gx, gy) * R0
    b = np.where(q < 1e-14, 1.0, 2 * j1(np.where(q < 1e-14, 1.0, q))
                 / np.where(q < 1e-14, 1.0, q))
    return np.pi * R0 ** 2 / P ** 2 * b * np.exp(-1j * (gx + gy) * CTR)


@contextmanager
def exact_disk():
    from lumenairy.elements.rcwa import twod as RT
    orig = RT._eps_convolution_2d

    def conv(eps_cell, orders, nx, ny):
        a = np.asarray(eps_cell)
        S = a.shape[0]
        bg, d = complex(a[0, 0]), complex(a[S // 2, S // 2])
        o = np.asarray(orders)
        dm = (o[:, 0][:, None] - o[:, 0][None, :]) * 2 * np.pi / P
        dn = (o[:, 1][:, None] - o[:, 1][None, :]) * 2 * np.pi / P
        out = (d - bg) * disk_ff(dm, dn)
        out = out + bg * np.eye(o.shape[0])
        return out
    RT._eps_convolution_2d = conv
    try:
        yield
    finally:
        RT._eps_convolution_2d = orig


def pixel_cell(t33, n):
    S = max(40, 4 * n + 8)
    x = np.arange(S) * P / S
    X, Y = np.meshgrid(x, x, indexing="ij")
    m = (X - CTR) ** 2 + (Y - CTR) ** 2 < R0 ** 2
    cell = np.broadcast_to(np.eye(3, dtype=complex), (S, S, 3, 3)).copy()
    cell[m] = t33
    return cell


def rcwa(n, t33=None, tag="rcwa"):
    from lumenairy.elements.rcwa import rcwa_jones_2d
    t33 = LC30 if t33 is None else t33
    t0 = time.perf_counter()
    with exact_disk():
        o, R, T, _J = rcwa_jones_2d(P, P, pixel_cell(t33, n), NSUB, NSUP, DEP,
                                    WL, n_orders_x=n, n_orders_y=n,
                                    symmetry=False, formulation="laurent")
    R, T = np.asarray(R), np.asarray(T)
    res = {"family": tag, "n": n, "orders_per_axis": 2 * n + 1,
           "vec": vec(o, R, T),
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
           "wall_s": time.perf_counter() - t0}
    if tag == "rcwa_scalar":
        from lumenairy.elements.rcwa.twod import rcwa_efficiency_2d_shapes
        shp = [{"shape": "disk", "eps": 4.0, "radius": R0,
                "center": (CTR, CTR)}]
        Rs, Ts = [], []
        for pol in ("tm", "te"):
            oo, RR, TT = rcwa_efficiency_2d_shapes(
                P, P, 1.0, shp, NSUB, NSUP, DEP, WL, polarization=pol,
                n_orders_x=n, n_orders_y=n)
            Rs.append(np.asarray(RR))
            Ts.append(np.asarray(TT))
        v2 = vec(oo, np.stack(Rs), np.stack(Ts))
        res["d_vs_shipped_shapes"] = float(np.max(np.abs(res["vec"] - v2)))
    dump(f"v4_{tag}_n{n}.json", res)
    print(f"v4_{tag}_n{n} clo {res['closure']:.1e} wall {res['wall_s']:.0f}s "
          f"{res.get('d_vs_shipped_shapes', '')}")


def load(pat):
    out = {}
    for f in glob.glob(os.path.join(HERE, pat)):
        d = json.load(open(f))
        out[f] = d
    return out


def summary():
    cur = {}
    for f, d in load("v4_curved_*.json").items():
        cur[(d["map"], d["M"])] = np.array(d["vec"])
    top = max(m for k, m in cur if k == "c3")
    ref = cur[("c3", top)]
    res = {"ref": f"c3 M={top}", "curved": {}, "rcwa": {}}
    for (k, m), v in sorted(cur.items()):
        res["curved"][f"{k}_M{m}"] = float(np.max(np.abs(v - ref)))
    # cross-topology table
    c5 = sorted(m for k, m in cur if k == "c5")
    c3 = sorted(m for k, m in cur if k == "c3")
    res["c3_vs_c5"] = {f"c3M{a}_c5M{b}": float(np.max(np.abs(
        cur[("c3", a)] - cur[("c5", b)]))) for a in c3 if a >= 8
        for b in c5 if b >= 6}
    res["c3_rung_change"] = {f"M{a}->{b}": float(np.max(np.abs(
        cur[("c3", a)] - cur[("c3", b)]))) for a, b in zip(c3, c3[1:])}
    rc = {}
    for f, d in load("v4_rcwa_n*.json").items():
        rc[d["n"]] = np.array(d["vec"])
    ns = sorted(rc)
    for n in ns:
        N = 2 * n + 1
        dist = float(np.max(np.abs(rc[n] - ref)))
        res["rcwa"][f"N{N}"] = {"dist": dist, "dist_x_N": dist * N}
    rich = {}
    for a, b in zip(ns, ns[1:]):
        Na, Nb = 2 * a + 1, 2 * b + 1
        ext = (Nb * rc[b] - Na * rc[a]) / (Nb - Na)
        rich[f"({Na},{Nb})"] = float(np.max(np.abs(ext - ref)))
    res["richardson"] = rich
    # same, against the c5 top rung (the other topology)
    if c5:
        ref5 = cur[("c5", c5[-1])]
        res["richardson_vs_c5top"] = {
            k: None for k in rich}
        for a, b in zip(ns, ns[1:]):
            Na, Nb = 2 * a + 1, 2 * b + 1
            ext = (Nb * rc[b] - Na * rc[a]) / (Nb - Na)
            res["richardson_vs_c5top"][f"({Na},{Nb})"] = float(
                np.max(np.abs(ext - ref5)))
    dump("v4_summary.json", res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    a = sys.argv[1]
    if a == "curved":
        curved(sys.argv[2], int(sys.argv[3]))
    elif a in ("rcwa", "rcwa_scalar"):
        n = int(sys.argv[2])
        if a == "rcwa":
            rcwa(n)
        else:
            rcwa(n, t33=4.0 * np.eye(3, dtype=complex), tag="rcwa_scalar")
    else:
        summary()

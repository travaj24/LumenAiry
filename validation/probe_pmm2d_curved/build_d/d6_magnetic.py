"""D6 -- a MATERIAL permeability under a map.

* film   -- a UNIFORM magnetic film (eps 2.0, mu = diag(1.8, 1.8, 1.3))
  under the circle maps (c3, c5) and the a = 0.15 p stretch, against the
  exact (eps, mu) Airy slab at normal incidence (mu_33 does not enter there;
  it is set != mu_t so a transposed / dropped component would show in D9).
* dual   -- a magnetic circular pillar (eps 1, mu = diag(2, 2, 1), r 0.36)
  and its eps <-> mu DUAL (eps = diag(2, 2, 1), mu 1), both under the same
  circle map with VACUUM half-spaces: by electromagnetic duality the E_x
  input of the magnetic pillar is the E_y input of the dielectric one, order
  by order.  The two sides run DIFFERENT weights (the chi blocks vs the eps
  blocks), so the identity is a cross-check of the magnetic composition.
  Control: the same comparison WITHOUT the polarization swap.
* stair  -- the shipped magnetic solver (no map) on the 4k-step staircase
  of the magnetic pillar (vacuum half-spaces), against the curved top rung.

usage: python d6_magnetic.py film <c3|c5|s15> <M>
       python d6_magnetic.py dual <c3|c5> <M>
       python d6_magnetic.py stair <k> <M>
       python d6_magnetic.py summary
"""
import json
import os
import sys
import time

import _common as C
import _dcommon as D
import numpy as np

P, R0, DEP, WL = 1.2, 0.36, 0.5, 1.0
MU_F = np.diag([1.8, 1.8, 1.3]).astype(complex)
EPS_F = 2.0
MU_P = np.diag([2.0, 2.0, 1.0]).astype(complex)


def airy(eps, mu, d, n1=1.0, n3=1.45):
    """Normal-incidence (R, T) of an isotropic (eps, mu) slab."""
    k0 = 2 * np.pi / WL
    n2 = np.sqrt(eps * mu + 0j)
    a1, a2, a3 = n1, n2 / mu, n3
    r12 = (a1 - a2) / (a1 + a2)
    r23 = (a2 - a3) / (a2 + a3)
    t12 = 2 * a1 / (a1 + a2)
    t23 = 2 * a2 / (a2 + a3)
    ph = np.exp(1j * n2 * k0 * d)
    den = 1 + r12 * r23 * ph ** 2
    r = (r12 + r23 * ph ** 2) / den
    t = t12 * t23 * ph / den
    return float(abs(r) ** 2), float(abs(t) ** 2 * a3 / a1)


def film(kind, M):
    cm = D.make_map(kind, P)
    t0 = time.perf_counter()
    st = D.PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                          n_modes=M, n_orders=2, cmap=cm)
    st.add_layer(DEP, eps=EPS_F, mu=MU_F)
    st.set_source(WL)
    o, R, T, J = st.solve(jones=True)
    o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)
    Rx, Tx = airy(EPS_F, MU_F[0, 0], DEP)
    i0 = C.idx(o, [(0, 0)])[0]
    err = 0.0
    for row in (0, 1):
        Rr, Tr = R[row].copy(), T[row].copy()
        Rr[i0] -= Rx
        Tr[i0] -= Tx
        err = max(err, float(np.abs(Rr).max()), float(np.abs(Tr).max()))
    J = np.asarray(J)
    res = {"map": kind, "M": M, "err": err, "R00": float(R[0, i0]),
           "R00_exact": Rx, "jones_offdiag": float(max(abs(J[0, 1]),
                                                       abs(J[1, 0]))),
           "closure": float(np.abs(R.sum(1) + T.sum(1) - 1).max()),
           "t": time.perf_counter() - t0}
    D.dump(f"d6_film_{kind}_M{M}.json", res)
    print("film", kind, M, f"err={err:.2e}", flush=True)


def pillar_cells(kind):
    cm = D.make_map(kind, P)
    N = cm.shape[0]
    disk = [(1, 1)] if N == 3 else [(i, j) for i in (1, 2, 3)
                                    for j in (1, 2, 3)]
    eye = np.broadcast_to(np.eye(3, dtype=complex), (N, N, 3, 3))
    mu_c, eps_c = eye.copy(), eye.copy()
    for c in disk:
        mu_c[c] = MU_P
        eps_c[c] = MU_P
    return cm, mu_c, eps_c


def solve_vac(cm, eps, mu, M):
    st = D.PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.0,
                          n_modes=M, n_orders=3, cmap=cm)
    if mu is None:
        st.add_layer(DEP, eps_cell=eps)
    else:
        st.add_layer(DEP, eps_cell=eps, mu_cell=mu)
    st.set_source(WL)
    o, R, T, _J = st.solve(jones=True)
    return np.asarray(o), np.asarray(R), np.asarray(T)


def dual(kind, M):
    cm, mu_c, eps_c = pillar_cells(kind)
    t0 = time.perf_counter()
    one = np.broadcast_to(np.eye(3, dtype=complex), mu_c.shape).copy()
    om, Rm, Tm = solve_vac(cm, one, mu_c, M)          # magnetic pillar
    oe, Re, Te = solve_vac(cm, eps_c, None, M)        # its eps dual
    im, ie = C.idx(om), C.idx(oe)
    d = max(np.abs(Rm[0, im] - Re[1, ie]).max(),
            np.abs(Tm[0, im] - Te[1, ie]).max(),
            np.abs(Rm[1, im] - Re[0, ie]).max(),
            np.abs(Tm[1, im] - Te[0, ie]).max())
    ctrl = max(np.abs(Rm[0, im] - Re[0, ie]).max(),
               np.abs(Tm[0, im] - Te[0, ie]).max())
    res = {"map": kind, "M": M, "duality": float(d),
           "no_swap_control": float(ctrl),
           "vec_mag": C.vec(om, Rm, Tm).tolist(),
           "closure_mag": float(np.abs(Rm.sum(1) + Tm.sum(1) - 1).max()),
           "closure_eps": float(np.abs(Re.sum(1) + Te.sum(1) - 1).max()),
           "t": time.perf_counter() - t0}
    D.dump(f"d6_dual_{kind}_M{M}.json", res)
    print("dual", kind, M, f"d={d:.2e} ctrl={ctrl:.2e}", flush=True)


def stair(k, M):
    c = P / 2
    inner = sorted([c - R0 * i / k for i in range(1, k + 1)]
                   + [c + R0 * i / k for i in range(1, k + 1)])
    w = np.array([0.0] + inner + [P])
    n = len(w) - 1
    mid = 0.5 * (w[:-1] + w[1:])
    mu = np.broadcast_to(np.eye(3, dtype=complex), (n, n, 3, 3)).copy()
    for i in range(n):
        for j in range(n):
            if (mid[i] - c) ** 2 + (mid[j] - c) ** 2 < R0 ** 2:
                mu[i, j] = MU_P
    t0 = time.perf_counter()
    st = D.PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.0,
                          n_modes=M, n_orders=3, layer_grids="per-layer")
    st.add_layer(DEP, eps_cell=np.ones((n, n), complex), mu_cell=mu,
                 x_walls=w, y_walls=w)
    st.set_source(WL)
    o, R, T, _J = st.solve(jones=True)
    o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)
    res = {"k": k, "M": M, "vec_mag": C.vec(o, R, T).tolist(),
           "t": time.perf_counter() - t0}
    D.dump(f"d6_stair_k{k}_M{M}.json", res)
    print("stair", k, M, flush=True)


def summary():
    here = os.path.dirname(os.path.abspath(__file__))
    rs = [json.load(open(os.path.join(here, f))) for f in os.listdir(here)
          if f.startswith("d6_") and f.endswith(".json")
          and "summary" not in f]
    out = {"film": {}, "dual": {}, "stair": {}}
    for r in rs:
        if "err" in r:
            out["film"].setdefault(r["map"], []).append(
                {"M": r["M"], "err": r["err"],
                 "jones_offdiag": r["jones_offdiag"]})
    duals = {}
    for r in rs:
        if "duality" in r:
            duals.setdefault(r["map"], []).append(r)
    ref = None
    if duals.get("c3"):
        ref = np.array(max(duals["c3"], key=lambda r: r["M"])["vec_mag"])
    for kind, rows in duals.items():
        rows.sort(key=lambda r: r["M"])
        lst = []
        for a, b in zip(rows, rows[1:] + [None]):
            row = {"M": a["M"], "duality": a["duality"],
                   "no_swap_control": a["no_swap_control"],
                   "closure_mag": a["closure_mag"],
                   "closure_eps": a["closure_eps"], "t": a["t"]}
            if ref is not None:
                row["to_c3_top"] = float(np.abs(np.array(a["vec_mag"])
                                                - ref).max())
            if b is not None:
                row["d_next"] = float(np.abs(np.array(a["vec_mag"])
                                             - np.array(b["vec_mag"])).max())
            lst.append(row)
        out["dual"][kind] = lst
    for r in rs:
        if "k" in r and ref is not None:
            out["stair"].setdefault(str(r["k"]), []).append(
                {"M": r["M"], "t": r["t"],
                 "to_c3_top": float(np.abs(np.array(r["vec_mag"])
                                           - ref).max())})
    for v in list(out["film"].values()) + list(out["stair"].values()):
        v.sort(key=lambda r: r["M"])
    D.dump("d6_summary.json", out)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "film":
        film(sys.argv[2], int(sys.argv[3]))
    elif a == "dual":
        dual(sys.argv[2], int(sys.argv[3]))
    elif a == "stair":
        stair(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "summary":
        summary()

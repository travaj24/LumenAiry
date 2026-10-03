"""V6 -- material permeability under a map (the verifier's own fixtures).

usage:
  python v6_mag.py dual KIND M MU     duality of a magnetic disk (eps 1, mu
        = MU in the disk) and its dielectric twin (eps = MU, mu 1), VACUUM
        half-spaces, the same map; KIND c3 | c5, MU sym | gyro | diag.
        Duality (E' = h, h' = -E) maps the magnetic problem's E_x input onto
        the dielectric problem's E_y input, order by order, for ANY tensors.
  python v6_mag.py muafter MAP M ARM  a strongly anisotropic ROTATED mu
        (diag(4, 0.5, 1.2) at 35 deg) in an eps-2 film under a STRONG map,
        against the verifier's own eps+mu Berreman; ARM ok | after (the
        inverse of the CELL-AVERAGED mu' instead of the pointwise inverse)
        | after_t (the same, chi_33 kept pointwise); MAP h2s | sh4s | c3
  python v6_mag.py stack M ARM        a merged-map stack with the verifier's
        own shapes: an ELLIPSE of a gyrotropic eps, a coaxial CIRCLE of eps
        2.2 + a symmetric off-diagonal mu, a uniform magnetic film; ARM
        lossless | lossyeps | lossymu
Output v6_*.json."""
import sys
import time
import warnings

import numpy as np
from _vdcommon import (
    CM,
    G3,
    GYRO_V,
    MU_GYRO,
    MU_LOSSY,
    MU_SYM,
    TS,
    PMM2DStackPure,
    TwoHarmonicStretch,
    berreman_eps_mu,
    circle,
    dump,
    film_stack,
    shear4,
)

P, R0, DEP, WL = 1.1, 0.33, 0.45, 1.0
MUS = {"sym": MU_SYM, "gyro": MU_GYRO,
       "diag": np.diag([1.9, 1.5, 1.3]).astype(complex)}


def disk(kind, M, eps_in, mu_in):
    cm = circle(P, kind, R0 / P)
    N = cm.shape[0]
    cells = [(1, 1)] if N == 3 else [(i, j) for i in (1, 2, 3)
                                     for j in (1, 2, 3)]
    eps = np.broadcast_to(np.eye(3, dtype=complex), (N, N, 3, 3)).copy()
    mu = np.broadcast_to(np.eye(3, dtype=complex), (N, N, 3, 3)).copy()
    for c in cells:
        eps[c] = eps_in
        mu[c] = mu_in
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.0,
                        n_modes=M, n_orders=3, cmap=cm)
    if np.allclose(mu_in, np.eye(3)):
        st.add_layer(DEP, eps_cell=eps)
    else:
        st.add_layer(DEP, eps_cell=eps, mu_cell=mu)
    st.set_source(WL)
    o, R, T, _J = st.solve(jones=True)
    return np.asarray(o), np.asarray(R), np.asarray(T)


def dual(kind, M, mun):
    mu = MUS[mun]
    t0 = time.perf_counter()
    o1, Rm, Tm = disk(kind, M, np.eye(3, dtype=complex), mu)
    o2, Rd, Td = disk(kind, M, mu, np.eye(3, dtype=complex))
    assert np.array_equal(o1, o2)
    swap = max(np.abs(Rm[0] - Rd[1]).max(), np.abs(Rm[1] - Rd[0]).max(),
               np.abs(Tm[0] - Td[1]).max(), np.abs(Tm[1] - Td[0]).max())
    noswap = max(np.abs(Rm - Rd).max(), np.abs(Tm - Td).max())
    clo = float(max(np.abs(Rm.sum(1) + Tm.sum(1) - 1).max(),
                    np.abs(Rd.sum(1) + Td.sum(1) - 1).max()))
    dump(f"v6_dual_{kind}_{mun}_M{M}.json", {
        "kind": kind, "mu": mun, "M": M, "duality": float(swap),
        "noswap_control": float(noswap), "closure": clo,
        "Rm": Rm, "Tm": Tm, "Rd": Rd, "Td": Td, "orders": o1,
        "wall_s": time.perf_counter() - t0})
    print(f"v6_dual_{kind}_{mun}_M{M} dual {swap:.2e} noswap {noswap:.2e} "
          f"clo {clo:.1e}")


# ---- the after-quadrature inverse (the verifier's own implementation) ------
def _inv2(a, b, c, d):
    det = a * d - b * c
    return d / det, -b / det, -c / det, a / det


def chi_after(W, rule, keep33=False):
    keys = ("c11", "c12", "c21", "c22")
    quad = rule if isinstance(rule, TS._StagMapQuad) else None
    wg = (quad.tensor[1] if quad is not None else rule[1])
    w2 = np.outer(wg, wg)

    def arr(k):
        return W[k].t if isinstance(W[k], TS._StagNodeWeight) else W[k]
    c = [arr(k) for k in keys]
    m = _inv2(*c)                                   # mu'_t at each node
    m33 = 1.0 / arr("c33")
    avg = [np.einsum("ijab,ab->ij", x, w2) / w2.sum() for x in m]
    a33 = np.einsum("ijab,ab->ij", m33, w2) / w2.sum()
    ci = _inv2(*avg)
    out = dict(W)
    for k, v in zip(keys + (("c33",) if not keep33 else ()),
                    list(ci) + ([1.0 / a33] if not keep33 else [])):
        full = np.broadcast_to(v[:, :, None, None], arr(k).shape).copy()
        if isinstance(W[k], TS._StagNodeWeight):
            pts = {}
            for cell, pv in W[k].p.items():
                # corner-rule cells: average with their own weights
                wq = quad.points[cell][2]
                pk = [W[kk].p[cell] for kk in keys]
                pm = _inv2(*pk)
                pav = [np.sum(wq * x) / np.sum(wq) for x in pm]
                pci = _inv2(*pav)
                if k == "c33":
                    pa = np.sum(wq / W["c33"].p[cell]) / np.sum(wq)
                    pts[cell] = np.full_like(pv, 1.0 / pa)
                else:
                    pts[cell] = np.full_like(pv, pci[keys.index(k)])
            out[k] = TS._StagNodeWeight(full, pts)
        else:
            out[k] = full
    return out


def muafter(mapn, M, arm):
    c, s = np.cos(np.deg2rad(35)), np.sin(np.deg2rad(35))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    mu = (Rz @ np.diag([4.0, 0.5, 1.2]) @ Rz.T).astype(complex)
    Pp = G3["P"]
    if mapn == "h2s":
        w = np.linspace(0, Pp, 4)
        cm = CM.SeparableStretch(
            w, w, fx=TwoHarmonicStretch(0.12 * Pp, 0.035 * Pp, 0.7),
            fy=TwoHarmonicStretch(-0.1 * Pp, 0.03 * Pp, -1.1))
    elif mapn == "sh4s":
        cm = shear4(Pp, amp=2.2)
    else:
        cm = circle(Pp, "c3")
    orig = TS._stag_map_weights

    def patched(bx, by, cmap, eps_cell, rule, mu_cell=None):
        W = orig(bx, by, cmap, eps_cell, rule, mu_cell=mu_cell)
        if mu_cell is None:
            return W
        return chi_after(W, rule, keep33=(arm == "after_t"))
    t0 = time.perf_counter()
    if arm != "ok":
        TS._stag_map_weights = patched
    try:
        st, o, R, T, J = film_stack(2.0, cm, M, 0.0, 0.0, mu=mu)
    finally:
        TS._stag_map_weights = orig
    Rb, Tb, rb = berreman_eps_mu([(2.0, mu, G3["DEP"])], G3["NSUB"],
                                 G3["NSUP"], G3["WL"])
    dRT = max(np.abs(R.sum(1) - Rb).max(), np.abs(T.sum(1) - Tb).max())
    dJ = np.abs(J - rb).max()
    # the spread of mu'_t inside a cell: max/min of the eigenvalues over the
    # nodes of cell (0, 0)
    dump(f"v6_muafter_{mapn}_M{M}_{arm}.json", {
        "map": mapn, "M": M, "arm": arm, "dRT": float(dRT), "dJ": float(dJ),
        "wall_s": time.perf_counter() - t0})
    print(f"v6_muafter_{mapn}_M{M}_{arm} dRT {dRT:.2e} dJ {dJ:.2e}")


def stack(M, arm):
    from lumenairy.elements.pmm import Circle, Ellipse
    gy = GYRO_V + (0.25j * np.eye(3) if arm == "lossyeps" else 0)
    mu2 = MU_LOSSY if arm == "lossymu" else MU_SYM
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=3)
    st.add_layer(0.25, shapes=[Ellipse(0.55, 0.55, 0.3, 0.24, gy)],
                 background_eps=1.0)
    st.add_layer(0.3, shapes=[Circle(0.55, 0.55, 0.18, 2.2, mu=mu2)],
                 background_eps=1.3)
    st.add_layer(0.1, eps=1.8, mu=MU_GYRO)
    st.set_source(WL, theta=0.2, phi=0.5)
    t0 = time.perf_counter()
    o, R, T, _J = st.solve(jones=True, retain_internal=True)
    A = np.asarray(st.layer_absorption())
    R, T = np.asarray(R), np.asarray(T)
    bal = 1 - R.sum(1) - T.sum(1)
    lossy_layer = {"lossless": None, "lossyeps": 0, "lossymu": 1}[arm]
    others = [i for i in range(3) if i != lossy_layer]
    res = {"M": M, "arm": arm, "closure": float(np.abs(bal).max()),
           "A_layers": A, "balance": bal,
           "A_lossless_layers": float(np.abs(A[others]).max()),
           "sumA_vs_balance": float(np.abs(A.sum(0) - bal).max()),
           "map_shape": list(st._layers[0]["cmap"].shape)
           if hasattr(st, "_layers") and isinstance(st._layers[0], dict)
           and st._layers[0].get("cmap") is not None else None,
           "wall_s": time.perf_counter() - t0}
    dump(f"v6_stack_{arm}_M{M}.json", res)
    print(f"v6_stack_{arm}_M{M} clo {res['closure']:.2e} lossless-layer A "
          f"{res['A_lossless_layers']:.1e} sumA-bal "
          f"{res['sumA_vs_balance']:.1e}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    a = sys.argv[1]
    if a == "dual":
        dual(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif a == "muafter":
        muafter(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif a == "stack":
        stack(int(sys.argv[2]), sys.argv[3])

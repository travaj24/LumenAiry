"""The readings behind every bar of tests/unit/test_pmm2d_staggered_curved_d.py,
at the UNIT-TEST sizes (each test's docstring cites the value here).

usage: python d_unit_readings.py <group>
groups: film | quad | li | mag | stack | oblique | mut | shapes
writes d_unit_<group>.json
"""
import sys
import warnings

import _common as C
import _dcommon as D
import numpy as np

from lumenairy.elements.pmm import Circle, Rect

P = 1.2
P3 = D.G3["P"]
MU2 = np.diag([1.5, 1.5, 1.2]).astype(complex)


def g_film():
    out = {}
    for mp in ("s05", "shear"):
        cm = D.make_map(mp, P3)
        for tn, t in (("lc", D.LC), ("gyro", D.GYRO)):
            for M in (4, 6):
                dRT, dJ, J = D.film_vs_berreman(t, cm, M)
                out[f"{mp}_{tn}_M{M}"] = {"RT": dRT, "J": dJ}
    cm = D.make_map("s05", P3)
    for M in (4, 5, 6):
        with D.mutate("transpose"):
            dRT, dJ, _ = D.film_vs_berreman(D.GYRO, cm, M)
        out[f"s05_gyro_transpose_M{M}"] = {"RT": dRT, "J": dJ}
        dRT, dJ, J = D.film_vs_berreman(D.GYRO, cm, M)
        out[f"s05_gyro_M{M}"] = {"RT": dRT, "J": dJ,
                                 "J01_J10": [float((J[0, 1] / J[1, 0]).real),
                                             float((J[0, 1] / J[1, 0]).imag)]}
    cm = D.make_map("shear", P3)
    with D.mutate("side"):
        dRT, dJ, _ = D.film_vs_berreman(D.LC, cm, 4)
    out["shear_lc_side_M4"] = {"RT": dRT, "J": dJ}
    cm = D.make_map("c3", P3)
    for tn, t in (("lc", D.LC), ("gyro", D.GYRO)):
        for M in (4, 6):
            dRT, dJ, _ = D.film_vs_berreman(t, cm, M)
            out[f"c3_{tn}_M{M}"] = {"RT": dRT, "J": dJ}
    return out


def g_quad():
    import d3_quad as Q
    out = {}
    cm, eps, mu = Q.cells("c3")
    with Q.nodes(16):
        a = Q.ops(cm, eps, mu, 6)
    with Q.nodes(48):
        top = Q.ops(cm, eps, mu, 6)
    with Q.nodes(24), Q.plain():
        pl = Q.ops(cm, eps, mu, 6)
    out["corner_n16_vs_n48"] = Q.rel(a, top)
    out["plain_n24_vs_corner_n48"] = Q.rel(pl, top)
    return out


def g_li():
    import d5_li2003 as L
    out = {}
    for arm, M in (("unmapped", 6), ("identity", 6), ("stretch", 8),
                   ("swap", 8), ("unmapped", 8)):
        cm = L.cmap_for(arm)
        eps = L.cell(L.LI_A, L.LI_B) if arm == "swap" else \
            L.cell(L.LI_B, L.LI_A)
        st = D.PMM2DStackPure(L.PX, L.PY, n_superstrate=1.0,
                              n_substrate=L.NSUB, n_modes=M, n_orders=4,
                              cmap=cm)
        st.add_layer(L.LAM, eps_cell=eps)
        st.set_source(L.LAM)
        o, R, T, J = st.solve(jones=True)
        o = np.asarray(o)
        R0 = np.asarray(R)[0, C.idx(o, L.ORD)]
        out[f"{arm}_M{M}"] = {
            "row1": float(max(abs(R0[k] - v) for k, v in
                              enumerate(L.TABLE1.values()))),
            "row2": float(max(abs(R0[k] - v) for k, v in
                              enumerate(L.TABLE2.values()))),
            "R": np.asarray(R).tolist(), "T": np.asarray(T).tolist(),
            "J": [[float(z.real), float(z.imag)]
                  for z in np.asarray(J).ravel()]}
    u, i = out["unmapped_M6"], out["identity_M6"]
    out["identity_vs_unmapped_M6"] = float(max(
        np.abs(np.array(u["R"]) - np.array(i["R"])).max(),
        np.abs(np.array(u["T"]) - np.array(i["T"])).max(),
        np.abs(np.array(u["J"]) - np.array(i["J"])).max()))
    for k in list(out):
        if isinstance(out[k], dict):
            for f in ("R", "T", "J"):
                out[k].pop(f, None)
    return out


def g_mag():
    import d6_magnetic as G
    out = {}
    cm = D.make_map("c3", P)

    def film(M, kind=None):
        st = D.PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                              n_modes=M, n_orders=2, cmap=cm)
        st.add_layer(G.DEP, eps=G.EPS_F, mu=G.MU_F)
        st.set_source(G.WL)
        o, R, T, _J = st.solve(jones=True)
        o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)
        Rx, Tx = G.airy(G.EPS_F, G.MU_F[0, 0], G.DEP)
        i0 = C.idx(o, [(0, 0)])[0]
        e = 0.0
        for row in (0, 1):
            Rr, Tr = R[row].copy(), T[row].copy()
            Rr[i0] -= Rx
            Tr[i0] -= Tx
            e = max(e, float(np.abs(Rr).max()), float(np.abs(Tr).max()))
        return e
    for M in (4, 5):
        out[f"film_M{M}"] = film(M)
        for mut in ("mu_after", "hgram_R"):
            with D.mutate(mut):
                out[f"film_{mut}_M{M}"] = film(M)
    # the tensor (non-magnetic) film under the circle map with hgram_R:
    # consistent-wrong in every region -> invisible (Phase A's finding)
    with D.mutate("hgram_R"):
        dRT, dJ, _ = D.film_vs_berreman(D.LC, D.make_map("c3", P3), 5)
    out["lc_film_hgram_R_M5"] = {"RT": dRT, "J": dJ}
    dRT, dJ, _ = D.film_vs_berreman(D.LC, D.make_map("c3", P3), 5)
    out["lc_film_M5"] = {"RT": dRT, "J": dJ}
    return out


def g_stack():
    """A 5 x 5 merged map: an LC30 circle (layer 1) inside a magnetic Rect
    (layer 2) -- the unit-size D7."""
    out = {}

    def mk(M, e1, mu2, layer2="mag", retain=False):
        st = D.PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                              n_modes=M, n_orders=3)
        st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.3, e1)],
                     background_eps=1.0)
        if layer2 == "mag":
            st.add_layer(0.2, shapes=[Rect(0.6, 0.6, 0.9, 0.9, 2.25,
                                           mu=mu2)], background_eps=1.0)
        elif layer2 == "vac_mu":
            st.add_layer(0.2, shapes=[Rect(0.6, 0.6, 0.9, 0.9, 1.0,
                                           mu=np.eye(3))],
                         background_eps=1.0)
        elif layer2 == "uniform":
            st.add_layer(0.2, eps=1.0)
        elif layer2 == "paint11":
            st.add_layer(0.2, shapes=[Rect(0.6, 0.6, 0.9, 0.9, 1.0,
                                           mu=1.1)], background_eps=1.0)
        st.add_layer(0.1, shapes=[Rect(0.6, 0.6, 0.9, 0.9, 1.5)],
                     background_eps=1.0)
        st.set_source(1.0)
        o, R, T, J = st.solve(retain_internal=retain, jones=True)
        return st, np.asarray(R), np.asarray(T), np.asarray(J)
    for M in (3, 4):
        st, R, T, J = mk(M, D.LC30, MU2)
        out[f"grid_M{M}"] = list(st.cmap.shape)
        out[f"closure_M{M}"] = float(np.abs(R.sum(1) + T.sum(1) - 1).max())
    M = 3
    st, R, T, _J = mk(M, D.LC30 + 0.3j * np.eye(3), MU2, retain=True)
    A = np.asarray(st.layer_absorption())
    bal = 1 - R.sum(1) - T.sum(1)
    G = st._internal["G"]
    st._internal["G"] = -D.TS.Granet2DTransverseE(
        P, P, st.cmap.u_walls, st.cmap.v_walls, M,
        np.ones(st.cmap.shape, complex), cmap=st.cmap).Rmat
    Ab = np.asarray(st.layer_absorption())
    st._internal["G"] = G
    out["abs_lossless_layer"] = float(np.abs(A[1:]).max())
    out["abs_lossless_layer_minusR"] = float(np.abs(Ab[1:]).max())
    out["abs_sum_vs_balance"] = float(np.abs(A.sum(0) - bal).max())
    runs = {k: mk(M, D.LC30, MU2, layer2=k) for k in
            ("vac_mu", "uniform", "paint11")}

    def d(a, b):
        return float(max(np.abs(runs[a][1] - runs[b][1]).max(),
                         np.abs(runs[a][2] - runs[b][2]).max(),
                         np.abs(runs[a][3] - runs[b][3]).max()))
    out["vac_mu_vs_uniform"] = d("vac_mu", "uniform")
    out["paint11_vs_uniform"] = d("paint11", "uniform")
    return out


def g_oblique():
    import d8_oblique as O
    out = {}
    th, ph = 25.0, 40.0
    tr, pr = O.reverse_angles(th, ph, -1, 0)
    for mat in ("lc", "gyro"):
        rs = {}
        for (a, b) in ((th, ph), (tr, pr)):
            cm = D.make_map("c3", P)
            eps = np.broadcast_to(np.eye(3, dtype=complex),
                                  (3, 3, 3, 3)).copy()
            eps[1, 1] = D.LC30 if mat == "lc" else D.GYRO
            st = D.PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                                  n_modes=5, n_orders=3, cmap=cm)
            st.add_layer(0.5, eps_cell=eps)
            st.set_source(1.0, theta=np.deg2rad(a), phi=np.deg2rad(b))
            o, R, T, _J = st.solve()
            o = np.asarray(o)
            md = st._modal
            idx = C.idx(o, O.ORDS)
            r = {"kx0": md["kx0"], "ky0": md["ky0"],
                 "kz_inc": float(md["kz_inc"]),
                 "kx": [float(md["kx"][i]) for i in idx],
                 "ky": [float(md["ky"][i]) for i in idx],
                 "kz_ref": [[float(np.real(md["kz_ref"][i])),
                             float(np.imag(md["kz_ref"][i]))] for i in idx],
                 "closure": float(np.abs(np.asarray(R).sum(1)
                                         + np.asarray(T).sum(1) - 1).max())}
            for k in ("rx", "ry"):
                arr = np.asarray(md[k])[:, idx]
                r[k] = [[[float(z.real), float(z.imag)] for z in row]
                        for row in arr]
            rs[(a, b)] = r
        f, q = rs[(th, ph)], rs[(tr, pr)]
        k = O.ORDS.index((-1, 0))
        sf = np.linalg.svd(O.jones_block(f, k), compute_uv=False)
        sr = np.linalg.svd(O.jones_block(q, k), compute_uv=False)
        sw = np.linalg.svd(O.jones_block(q, 0), compute_uv=False)
        out[mat] = {"recip": float(np.abs(sf - sr).max()),
                    "wrong_pair": float(np.abs(sf - sw).max()),
                    "closure": max(f["closure"], q["closure"])}
    return out


def g_mut():
    """The D9 mutations at unit-test sizes (the ones not already in the
    film / mag groups)."""
    import d4_pillar as Q
    out = {}
    cmc = D.make_map("c3", P3)

    def lc_con(M):
        return D.film_vs_berreman(D.LC, cmc, M, theta=np.deg2rad(25.0),
                                  phi=np.deg2rad(40.0))[:2]

    def pillar(M):
        cm, eps = Q.disk_cells("c3")
        st = D.PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                              n_modes=M, n_orders=3, cmap=cm)
        st.add_layer(0.5, eps_cell=eps)
        st.set_source(1.0)
        o, R, T, _J = st.solve(jones=True)
        return C.vec(np.asarray(o), np.asarray(R), np.asarray(T))
    p0 = pillar(5)
    out["lc_con_M5"] = lc_con(5)
    for mut in ("no_sg_e33", "mixed_sign", "side"):
        with D.mutate(mut):
            out[f"lc_con_{mut}_M5"] = lc_con(5)
            out[f"pillar_{mut}_M5"] = float(np.abs(pillar(5) - p0).max())
    cms = D.make_map("s05", P3)
    out["lc_s05_M5"] = D.film_vs_berreman(D.LC, cms, 5)[:2]
    with D.mutate("mixed_sign"):
        out["lc_s05_mixed_sign_M5"] = D.film_vs_berreman(D.LC, cms, 5)[:2]
    # pillar ladder change M5 -> M6 (the discretisation level)
    out["pillar_d_M5_M6"] = float(np.abs(pillar(6) - p0).max())
    return out


def g_shapes():
    """The shapes route with mu = the explicit compile_shapes(with_mu) route,
    byte for byte; and scalar mapped bytes untouched by Phase D (the shape
    circle at M = 4 vs the PRE tree's d1 hash is in d1_compare.json)."""
    from lumenairy.elements.pmm import compile_shapes, pmm_jones_2d_staggered
    sh = [Circle(0.6, 0.6, 0.36, D.LC30, mu=np.diag([1.5, 1.5, 1.0]))]
    a = pmm_jones_2d_staggered(P, P, None, 1.45, 1.0, 0.5, 1.0, n_modes=4,
                               n_orders=3, shapes=sh, background_eps=1.0,
                               background_mu=1.2)
    eps, xw, yw, cm, mu = compile_shapes(P, P, sh, 1.0, background_mu=1.2,
                                         with_mu=True)
    b = pmm_jones_2d_staggered(P, P, eps, 1.45, 1.0, 0.5, 1.0, n_modes=4,
                               n_orders=3, cmap=cm, mu_cell=mu)
    return {"bytes_equal": all(np.array_equal(np.asarray(x), np.asarray(y))
                               for x, y in zip(a, b)),
            "mu_cell": np.asarray(mu).real.tolist()}


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    g = sys.argv[1]
    res = {"film": g_film, "quad": g_quad, "li": g_li, "mag": g_mag,
           "stack": g_stack, "oblique": g_oblique, "mut": g_mut,
           "shapes": g_shapes}[g]()
    D.dump(f"d_unit_{g}.json", res)
    print(res)

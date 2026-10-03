"""A2 -- the IDENTITY map through the quadrature path reproduces the shipped
(unmapped, Kronecker) operators to round-off; plus the A2 upper gap (a map of
amplitude 1e-6 p must move the operators far above the bar) and the
quadrature-adequacy check of the mapped assembly (nq = 2M + 8 against 4x).

  python validation/probe_pmm2d_curved/build_a/a2_identity.py

Output: a2_identity.json.  Sections:
  kernel  : each of the 18 (x-flavour, y-flavour) block shapes of
            _assemble, quadrature (_stag_quad_weighted, weight = eps_cell
            broadcast to the nodes) vs the shipped kron helper
            (_eps_weighted / _eps_dir, weight = eps_cell per cell);
  ops     : the ASSEMBLED operators of Granet2DTransverseE(cmap=IdentityMap)
            vs Granet2DTransverseE(cmap=None): Rmat, Lmat, Stt, Schur,
            [eps_t] blocks, plain Gram vs -Rmat; matched pencil eigenvalues;
  far     : _far_projector_2d(cmap=IdentityMap) vs the shipped projector;
  solve   : full single-layer solve, R / T / Jones, identity map vs none;
  gap     : SineStretch(1e-6 p) operator movement;
  quad    : stretch a = 0.15 p, nq = 2M+8 vs 4(2M+8): operators and R/T.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

TS = C.TS
K0 = 2 * np.pi / C.WL
B, T = "B", "Btilde"
# the 18 weighted blocks of _assemble: (name, xspec, yspec)
BLOCKS = [
    ("R11_c22", (B, "m", B), (T, "m", T)),
    ("R12_c21", (B, "m", T), (T, "m", B)),
    ("R21_c12", (T, "m", B), (B, "m", T)),
    ("R22_c11", (T, "m", T), (B, "m", B)),
    ("Et11", (B, "m", B), (T, "m", T)),
    ("Et22", (T, "m", T), (B, "m", B)),
    ("Et12", (B, "m", T), (T, "m", B)),
    ("Et21", (T, "m", B), (B, "m", T)),
    ("Gw_chi33", (B, "m", B), (B, "m", B)),
    ("Meps33", (T, "m", T), (T, "m", T)),
    ("Ktz_1a", (B, "d", T), (T, "m", T)),
    ("Ktz_1b", (B, "m", T), (T, "d", T)),
    ("Ktz_2a", (T, "m", T), (B, "d", T)),
    ("Ktz_2b", (T, "d", T), (B, "m", T)),
    ("Kzt_1a", (T, "dL", B), (T, "m", T)),
    ("Kzt_1b", (T, "m", B), (T, "dL", T)),
    ("Kzt_2a", (T, "m", T), (T, "dL", B)),
    ("Kzt_2b", (T, "dL", T), (T, "m", B)),
]


def rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    s = float(np.max(np.abs(b)))
    d = float(np.max(np.abs(a - b)))
    return {"max_abs": d, "max_rel": d / s if s else d,
            "bit_identical": bool(np.array_equal(a, b))}


def kernel(sol_none, wx, wy, M):
    bx, by = sol_none.bx, sol_none.by
    eps = sol_none.eps_cell
    rule = TS._stag_map_quad_rule(M)
    nq = rule[0].size
    Wn = np.broadcast_to(eps[:, :, None, None], eps.shape + (nq, nq))
    cache = {}
    out = {}
    for name, xs, ys in BLOCKS:
        Q = TS._stag_quad_weighted(bx, by, xs, ys, Wn, rule, cache)
        if xs[1] == "m" and ys[1] == "m":
            K = sol_none._eps_weighted(
                (bx, bx.m_ref, getattr(bx, xs[0]), getattr(bx, xs[2])),
                (by, by.m_ref, getattr(by, ys[0]), getattr(by, ys[2])), eps)
        else:
            K = sol_none._eps_dir(bx, *xs, by, *ys, wmap=eps)
        out[name] = rel(Q, K)
    return out


def ops(s0, s1):
    """s1 = identity map, s0 = no map."""
    E11, E22 = s1.Et_blocks
    E12, E21 = s1.Et_offdiag
    G1, G2 = s1.Ggram_blocks
    qq = s1.q * s1.q
    Gfull = np.zeros_like(s1.Rmat)
    Gfull[:qq, :qq] = G1
    Gfull[qq:, qq:] = G2
    g_a = np.sort_complex(sla.eigvals(s0.Lmat, -s0.Rmat))
    g_b = np.sort_complex(sla.eigvals(s1.Lmat, -s1.Rmat))
    dm = np.abs(g_a[:, None] - g_b[None, :]).min(axis=1)
    return {"Rmat": rel(s1.Rmat, s0.Rmat), "Lmat": rel(s1.Lmat, s0.Lmat),
            "Stt": rel(s1.Stt, s0.Stt), "Schur": rel(s1.Schur, s0.Schur),
            "Et11": rel(E11, s0.Et_blocks[0]),
            "Et22": rel(E22, s0.Et_blocks[1]),
            "Et12_max": float(np.max(np.abs(E12))),
            "Et21_max": float(np.max(np.abs(E21))),
            "plainGram_vs_minusR": rel(Gfull, -s0.Rmat),
            "eig_matched_max_rel": float(np.max(dm / np.maximum(1.0,
                                                                np.abs(g_a))))}


def worst(d):
    vals = []
    for v in d.values():
        if isinstance(v, dict) and "max_rel" in v:
            vals.append(v["max_rel"])
    return max(vals)


def main():
    res = {"env": C.env_record(), "cases": []}
    eps = C.cell("pillar")
    for M in (5, 7):
        for label, wx, wy in (("uniform_int", 3, 3),
                              ("nonuniform", C.XW, C.YW["pillar"])):
            s0 = TS.Granet2DTransverseE(C.P, C.P, wx, wy, M, eps, k0=K0)
            cm = C.IdentityMap(wx, wy, C.P, C.P)
            s1 = TS.Granet2DTransverseE(C.P, C.P, wx, wy, M, eps, k0=K0,
                                        cmap=cm)
            case = {"M": M, "grid": label, "kernel": kernel(s0, wx, wy, M),
                    "ops": ops(s0, s1)}
            ox = np.arange(-3, 4)
            for a0 in ((0.0, 0.0), (0.9, 0.4)):
                bx = TS.Basis1D(C.P, wx, M, np.exp(-1j * a0[0] * C.P))
                by = TS.Basis1D(C.P, wy, M, np.exp(-1j * a0[1] * C.P))
                Pa = TS._far_projector_2d(bx, by, ox, ox, a0[0], a0[1])
                Pb = TS._far_projector_2d(bx, by, ox, ox, a0[0], a0[1],
                                          cmap=cm)
                case[f"far_a0={a0}"] = {"P1": rel(Pb[0], Pa[0]),
                                        "P2": rel(Pb[1], Pa[1]),
                                        "offdiag_absent": (Pb[2] is None
                                                           and Pb[3] is None)}
            # A2 UPPER GAP: a 1e-6 p stretch must move the operators
            cmg = C.SeparableStretch(wx, wy, fx=C.SineStretch(1e-6 * C.P),
                                     period_x=C.P, period_y=C.P)
            s2 = TS.Granet2DTransverseE(C.P, C.P, wx, wy, M, eps, k0=K0,
                                        cmap=cmg)
            case["gap_1e-6p"] = {"Rmat": rel(s2.Rmat, s1.Rmat),
                                 "Lmat": rel(s2.Lmat, s1.Lmat)}
            case["worst_kernel_rel"] = worst(case["kernel"])
            case["worst_ops_rel"] = worst(case["ops"])
            res["cases"].append(case)
            print(M, label, "kernel", f"{case['worst_kernel_rel']:.2e}",
                  "ops", f"{case['worst_ops_rel']:.2e}",
                  "eig", f"{case['ops']['eig_matched_max_rel']:.2e}",
                  "gap", f"{case['gap_1e-6p']['Lmat']['max_rel']:.2e}",
                  flush=True)
    # full solve: identity map vs none (integer 3x3 grid; the shipped shared
    # path) -- R, T, Jones
    full = []
    for M in (5, 7):
        st0 = C.PMM2DStackPure(C.P, C.P, n_superstrate=C.N_SUP,
                               n_substrate=C.N_SUB, n_modes=M, n_orders=3)
        st0.add_layer(C.DEPTH, eps_cell=eps)
        st0.set_source(C.WL)
        o0, R0, T0, J0 = st0.solve()
        st1 = C.PMM2DStackPure(C.P, C.P, n_superstrate=C.N_SUP,
                               n_substrate=C.N_SUB, n_modes=M, n_orders=3,
                               cmap=C.IdentityMap(3, 3, C.P, C.P))
        st1.add_layer(C.DEPTH, eps_cell=eps)
        st1.set_source(C.WL)
        o1, R1, T1, J1 = st1.solve()
        row = {"M": M, "dR": float(np.max(np.abs(R1 - R0))),
               "dT": float(np.max(np.abs(T1 - T0))),
               "dJones": float(np.max(np.abs(J1 - J0))),
               "bit_identical": bool(np.array_equal(R1, R0)
                                     and np.array_equal(T1, T0))}
        full.append(row)
        print("solve", json.dumps(row), flush=True)
    res["solve"] = full
    # quadrature adequacy under the stretch (a = 0.15 p, the 33:1 map)
    quad = []
    orig = TS._stag_map_quad_rule
    for M in (5, 7):
        cm = C.stretch_map(0.15, "stripe")
        rows = {}
        for fac in (1, 4):
            def rule(MM, nq=None, _f=fac):
                return orig(MM, (2 * int(MM) + 8) * _f)
            TS._stag_map_quad_rule = rule
            try:
                s = TS.Granet2DTransverseE(C.P, C.P, cm.u_walls, cm.v_walls, M,
                                           C.cell("stripe"), k0=K0, cmap=cm)
                o, R, T, _J, _st = C.solve("stripe", 0.15, M)
            finally:
                TS._stag_map_quad_rule = orig
            rows[fac] = (s, C.vec(o, R, T))
        q = {"M": M, "Lmat": rel(rows[1][0].Lmat, rows[4][0].Lmat),
             "Rmat": rel(rows[1][0].Rmat, rows[4][0].Rmat),
             "RT_max_abs": float(np.max(np.abs(rows[1][1] - rows[4][1])))}
        quad.append(q)
        print("quad", M, f"{q['Lmat']['max_rel']:.2e}", f"{q['RT_max_abs']:.2e}",
              flush=True)
    res["quad"] = quad
    with open(os.path.join(C.HERE, "a2_identity.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

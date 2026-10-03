"""V2 -- the derivation, checked independently.

usage: python v2_derivation.py a|b|c|d [args]

a  (mu'^{3t}).  A z-independent map Lambda_m = blockdiag(J, 1) cannot create
   an out-of-plane block from a block-form mu: M(mu) = sg Lambda_m^-1 mu
   Lambda_m^-T keeps (M(mu))_{t3} = sg J^-1 mu_{t3} = 0.  Measured on random J
   and random complex block-form mu (and its inverse, the chi side).  The
   COMPOSITE (slant) frame does create one: mu'^{3t} = -sg mu33 tau^T, which
   the code carries through tau -- checked: -mu'^{3t} / mu'^{33} = tau^T and
   chi'_{t3} = chi'_tt tau for the full 3 x 3 inverse.  Plus: every public
   entry point REFUSES an out-of-plane material mu, with and without a map,
   and the message is recorded (loud? names the limit?).
b  (B HPD).  The E block of B is -R = C[chi_t]C: its Hermiticity, smallest
   eigenvalue, its smallest generalized eigenvalue against the PLAIN Gram
   (how far the Duffy-weighted chi pushes it toward 0), cond, and whether
   the Cholesky ever fails -- on the 3 x 3 circle (r 0.30 / 0.45 p, centred
   and off-centre) and the 5 x 5 circle, M = 6 .. 8.
c  (tau).  The per-node variation of tau = J^-1 t across a curved cell, and
   the "tau = t" fail-before on a SLANTED PILLAR under a strongly curved
   map against the same pillar correct (36-vector change), next to the slab
   null-test reading.
d  (the reduction).  Unmapped: the general generator with mu = 1 (forced
   through _assemble_oop_general by mu_cell = ones) against the shipped
   generator, vertical and slanted, operators and modes.
"""
import sys
import time

import _ve1common as V
import numpy as np

TS, CM = V.TS, V.CM
part = sys.argv[1]


def rand_block_mu(rng):
    m = np.zeros((3, 3), complex)
    m[:2, :2] = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    m[:2, :2] += 3 * np.eye(2)
    m[2, 2] = 1.5 + 0.3j * rng.normal()
    return m


def _shape_solve(MU):
    st = V.PMM2DStackPure(1.0, 1.0, n_modes=4, n_orders=2)
    st.add_layer(0.3, shapes=[V.Circle(0.5, 0.5, 0.3, 2.0, mu=MU)],
                 background_eps=1.0)
    st.set_source(1.0)
    return st.solve()


if part == "a":
    rng = np.random.default_rng(11)
    worst_t3, worst_tau, worst_kap = 0.0, 0.0, 0.0
    for _ in range(2000):
        J = rng.normal(size=(2, 2)) + 2 * np.eye(2)
        if np.linalg.det(J) <= 0.1:
            continue
        sg = np.linalg.det(J)
        L = np.eye(3)
        L[:2, :2] = J
        mu = rand_block_mu(rng)
        Mm = sg * np.linalg.inv(L) @ mu @ np.linalg.inv(L).T
        chi = np.linalg.inv(Mm)
        worst_t3 = max(worst_t3, float(np.abs(Mm[:2, 2]).max()
                                       + np.abs(Mm[2, :2]).max()
                                       + np.abs(chi[:2, 2]).max()
                                       + np.abs(chi[2, :2]).max())
                       / float(np.abs(Mm).max()))
        # the composite frame Lambda = [[J, t], [0, 1]]
        t = rng.normal(size=2) * 0.4
        Lc = L.copy()
        Lc[:2, 2] = t
        Mc = sg * np.linalg.inv(Lc) @ mu @ np.linalg.inv(Lc).T
        chic = np.linalg.inv(Mc)
        tau = np.linalg.solve(J, t)
        worst_tau = max(worst_tau, float(np.abs(
            -Mc[2, :2] / Mc[2, 2] - tau).max()))
        worst_kap = max(worst_kap, float(np.abs(
            chic[:2, 2] - chic[:2, :2] @ tau).max()) / float(
                np.abs(chic).max()))
        # and chi'_tt unchanged by the shear, chi'33 Schur = 1 / mu'33
        worst_kap = max(worst_kap, float(np.abs(
            chic[:2, :2] - J.T @ np.linalg.inv(mu[:2, :2]) @ J / sg).max())
            / float(np.abs(chic).max()))
        sch = chic[2, 2] - chic[2, :2] @ np.linalg.solve(chic[:2, :2],
                                                         chic[:2, 2])
        worst_kap = max(worst_kap, abs(sch - 1.0 / (sg * mu[2, 2]))
                        / abs(sch))
    res = {"map_creates_t3_rel": worst_t3,
           "composite_tau_identity": worst_tau,
           "composite_kappa_chitt_schur_rel": worst_kap}
    # refusals of an out-of-plane mu at every entry point
    P = 1.0
    MU = V.MU_FULL_OOP
    eps = np.broadcast_to(V.EYE, (3, 3, 3, 3)).copy()
    eps[1, 1] = V.DIRGEN
    mucell = np.broadcast_to(V.EYE, (3, 3, 3, 3)).copy()
    mucell[1, 1] = MU
    cm = CM._circle_map_3x3(P, 0.3)[0]
    calls = {
        "solver_nomap": lambda: TS.Granet2DTransverseE(
            P, P, 3, 3, 4, eps, mu_cell=mucell),
        "solver_map": lambda: TS.Granet2DTransverseE(
            P, P, cm.u_walls, cm.v_walls, 4, eps, mu_cell=mucell, cmap=cm),
        "solver_slant": lambda: TS.Granet2DTransverseE(
            P, P, 3, 3, 4, eps, mu_cell=mucell, slant=(0.1, 0.0)),
        "jones_nomap": lambda: TS.pmm_jones_2d_staggered(
            P, P, eps, 1.5, 1.0, 0.3, 1.0, degree=4, n_orders=2,
            mu_cell=mucell),
        "jones_map": lambda: TS.pmm_jones_2d_staggered(
            P, P, eps, 1.5, 1.0, 0.3, 1.0, degree=4, n_orders=2,
            mu_cell=mucell, cmap=cm),
        "stack_uniform_mu_nomap": lambda: V.PMM2DStackPure(
            P, P, n_modes=4).add_layer(0.3, eps=V.DIRGEN, mu=MU),
        "stack_uniform_mu_map": lambda: V.PMM2DStackPure(
            P, P, n_modes=4, cmap=cm).add_layer(0.3, eps=V.DIRGEN, mu=MU),
        "stack_mucell_map": lambda: V.PMM2DStackPure(
            P, P, n_modes=4, cmap=cm).add_layer(0.3, eps_cell=eps,
                                                mu_cell=mucell),
        "stack_slant_mu": lambda: V.PMM2DStackPure(
            P, P, n_modes=4).add_layer(0.3, eps=2.0, mu=MU,
                                       slant=(0.1, 0.0)),
        "shape_circle_mu": lambda: _shape_solve(MU),
    }
    ref = {}
    for k, fn in calls.items():
        try:
            fn()
            ref[k] = "NOT REFUSED"
        except Exception as exc:
            ref[k] = f"{type(exc).__name__}: {exc}"[:400]
    res["refusals"] = ref
    V.dump("v2a_mu3t.json", res)
    print({k: v for k, v in res.items() if k != "refusals"})
    for k, v in ref.items():
        print(k, "->", v[:150])


if part == "b":
    P = 1.0
    out = {}
    maps = {"c3_r30": CM._circle_map_3x3(P, 0.30)[0],
            "c3_r45": CM._circle_map_3x3(P, 0.45)[0],
            "c3off": CM._circle_map_3x3(P, 0.27, center=(0.41, 0.56))[0],
            "c5_r30": CM._circle_map_5x5(P, 0.30)[0],
            "th2b": V.map_two_harm(P, 2, big=True)}
    Ms = [int(a) for a in sys.argv[2:]] or [6, 7, 8]
    for mn, cm in maps.items():
        N = cm.shape[0]
        eps = np.broadcast_to(V.EYE, (N, N, 3, 3)).copy()
        eps[N // 2, N // 2] = V.DIRGEN
        for M in Ms:
            if N == 5 and M > 7:
                continue
            for mu_name, mu in (("vac", None), ("gyro", V.MU_GYRO)):
                mc = None
                if mu is not None:
                    mc = np.broadcast_to(V.EYE, (N, N, 3, 3)).copy()
                    mc[N // 2, N // 2] = mu
                t0 = time.perf_counter()
                s = TS.Granet2DTransverseE(P, P, cm.u_walls, cm.v_walls, M,
                                           eps, mu_cell=mc, cmap=cm)
                qq = s.q * s.q
                B = s.Bgen
                mR = B[:2 * qq, :2 * qq]
                herm = float(np.abs(mR - mR.conj().T).max()
                             / np.abs(mR).max())
                Hm = 0.5 * (mR + mR.conj().T)
                ev = np.linalg.eigvalsh(Hm)
                Gp = np.zeros_like(mR)
                Gp[:qq, :qq] = B[3 * qq:, 3 * qq:]
                Gp[qq:, qq:] = B[2 * qq:3 * qq, 2 * qq:3 * qq]
                import scipy.linalg as sla
                gev = sla.eigh(Hm, Gp, eigvals_only=True)
                try:
                    np.linalg.cholesky(B)
                    chol = "ok"
                except np.linalg.LinAlgError as exc:
                    chol = f"FAILED {exc}"
                # pointwise: chi_t eigenvalues at the nodes
                w = s._mapw
                c11, c12, c22 = (w["c11"], w["c12"], w["c22"])

                def flat(x):
                    if isinstance(x, TS._StagNodeWeight):
                        return np.concatenate(
                            [np.ravel(x.t)] + [np.ravel(v) for v in
                                               x.p.values()])
                    return np.ravel(x)
                a, b_, d = flat(c11), flat(c12), flat(c22)
                tr, det = a + d, a * d - b_ * np.conj(b_)
                lmin = np.real(tr / 2 - np.sqrt((tr / 2) ** 2 - det))
                out[f"{mn}_{mu_name}_M{M}"] = dict(
                    q=s.q, herm_rel=herm, eig_min=float(ev[0]),
                    eig_max=float(ev[-1]), cond=float(ev[-1] / ev[0]),
                    gen_eig_min=float(gev[0]), gen_eig_max=float(gev[-1]),
                    chol=chol, bgen_hermitian=bool(s._bgen_hermitian),
                    node_chi_eig_min=float(np.min(lmin)),
                    node_chi_eig_max=float(np.max(np.real(
                        tr / 2 + np.sqrt((tr / 2) ** 2 - det)))),
                    wall=time.perf_counter() - t0)
                print(mn, mu_name, M, {k: (f"{v:.3e}" if isinstance(v, float)
                                          else v)
                                       for k, v in out[f"{mn}_{mu_name}_M{M}"]
                                       .items()}, flush=True)
    V.dump(f"v2b_hpd_{'_'.join(map(str, Ms))}.json", out)


if part == "c":
    # c1: the per-node variation of tau = J^-1 t on the curved maps
    f = V.PIL
    t = np.array([-0.15, 0.0])          # internal shear = -public
    out = {"tau_variation": {}, "pillar": {}}
    for mn, cm in (("c3", CM._circle_map_3x3(f["P"], f["R0"])[0]),
                   ("c3_r45", CM._circle_map_3x3(f["P"], 0.45 * f["P"])[0]),
                   ("c5", CM._circle_map_5x5(f["P"], f["R0"])[0])):
        uu = np.linspace(0.02, 0.98, 25)
        worst = 0.0
        for sx in range(cm.shape[0]):
            for sy in range(cm.shape[1]):
                ub, vb = cm.u_bounds, cm.v_bounds
                U = ub[sx] + uu * (ub[sx + 1] - ub[sx])
                Vv = vb[sy] + uu * (vb[sy + 1] - vb[sy])
                _X, _Y, xu, xv, yu, yv = cm.geom(sx, sy, U, Vv)
                sg = xu * yv - xv * yu
                t1 = (yv * t[0] - xv * t[1]) / sg
                t2 = (-yu * t[0] + xu * t[1]) / sg
                dev = np.hypot(t1 - t[0], t2 - t[1]) / np.hypot(*t)
                worst = max(worst, float(np.max(dev)))
        out["tau_variation"][mn] = worst
    # c2: tau = t on the slanted pillar (correct vs mutant), c3 and c5
    def slant_patch(kind):
        orig = TS._stag_map_slant_weights

        def fpatch(o, A, sg, tvec):
            orig(o, A, sg, tvec)
            if kind == "tau_t":
                o["t1"] = o["t1"] * 0 + tvec[0]
                o["t2"] = o["t2"] * 0 + tvec[1]
                o["k1"] = o["c11"] * tvec[0] + o["c12"] * tvec[1]
                o["k2"] = o["c21"] * tvec[0] + o["c22"] * tvec[1]
            elif kind == "tau_J":
                # tau = J t instead of J^-1 t; J from adj(J) = A
                xu, xv = A[1][1], -A[0][1]
                yu, yv = -A[1][0], A[0][0]
                o["t1"] = xu * tvec[0] + xv * tvec[1]
                o["t2"] = yu * tvec[0] + yv * tvec[1]
                o["k1"] = o["c11"] * o["t1"] + o["c12"] * o["t2"]
                o["k2"] = o["c21"] * o["t1"] + o["c22"] * o["t2"]
        return V.patched(TS, "_stag_map_slant_weights", fpatch)
    kinds = sys.argv[2].split(",") if len(sys.argv) > 2 else ["c3"]
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    mat = sys.argv[4] if len(sys.argv) > 4 else "eps35"
    t33 = 3.5 * V.EYE if mat == "eps35" else V.PIL_OOP
    for kind in kinds:
        for th, ph, tag in ((0.0, 0.0, "n"), (np.deg2rad(20), np.deg2rad(35),
                                              "c")):
            res = {}
            for arm in ("ok", "tau_t", "tau_J"):
                try:
                    if arm == "ok":
                        st, o, R, T, wall = V.pillar_run(
                            kind, t33, M, th, ph, slant=(0.15, 0.0))
                    else:
                        with slant_patch(arm):
                            st, o, R, T, wall = V.pillar_run(
                                kind, t33, M, th, ph, slant=(0.15, 0.0))
                    res[arm] = V.vec36(o, R, T)
                except Exception as exc:
                    res[arm] = None
                    print(arm, "raised", exc)
            out["pillar"][f"{mat}_{kind}_{tag}_M{M}"] = {
                arm: (None if res[arm] is None else
                      float(np.abs(res[arm] - res["ok"]).max()))
                for arm in ("tau_t", "tau_J")}
            print(kind, tag, M, out["pillar"][f"{mat}_{kind}_{tag}_M{M}"],
                  flush=True)
    print(out["tau_variation"])
    V.dump(f"v2c_tau_{mat}_{'_'.join(kinds)}_M{M}.json", out)


if part == "d":
    P = 1.0
    out = {}
    for name, eps, slant, a0 in (
            ("oop", None, None, (0.0, 0.0)),
            ("oop_con", None, None, (0.6, -0.4)),
            ("slant_oop_con", None, (0.2, -0.1), (0.5, 0.3)),
            ("slant_sc", "sc", (0.15, 0.1), (0.0, 0.0))):
        if eps is None:
            e = np.broadcast_to(V.EYE, (3, 3, 3, 3)).copy()
            e[0, 0] = V.NRGEN
            e[1, 2] = V.DIRGEN
        else:
            e = np.array([[1, 1, 1], [1, 3.4, 1], [2.0, 1, 1]], complex)
        kw = dict(alpha0x=a0[0], alpha0y=a0[1], slant=slant)
        s0 = TS.Granet2DTransverseE(P, P, 3, 3, 5, e, **kw)
        s1 = TS.Granet2DTransverseE(P, P, 3, 3, 5, e,
                                    mu_cell=np.ones((3, 3), complex), **kw)
        assert s1.magnetic and s1.offplane
        m0 = TS._region_modes_oop(s0)
        m1 = TS._region_modes_oop(s1)
        d = np.abs(m0[2][:, None] - m1[2][None, :]).min(axis=1).max()
        out[name] = dict(
            A=float(np.abs(s1.Agen - s0.Agen).max() / np.abs(s0.Agen).max()),
            B=float(np.abs(s1.Bgen - s0.Bgen).max() / np.abs(s0.Bgen).max()),
            eig=float(d / np.abs(m0[2]).max()))
        print(name, out[name])
    V.dump("v2d_reduction.json", out)

"""V5 -- A5 / A6 / A7 on the verifier's stripe and pillar, and a MAGNETIC
mapped region (a physical mu on top of the map's own permeability).

split   -R == [eps'_t]/eps for a uniform region under: sine 0.12, the
        asymmetric two-harmonic stretch, and the SHEARED map (g12 != 0);
        eps = 1, 6.25, 3 + 0.4i; M = 4, 6.
hgram   stripe and pillar under harm_asym at M = 6, 8: correct closure, the
        MIXED arm (half-spaces through -R, layer through the plain Gram), and
        the CONSISTENT-WRONG arm (every region through -R) -- the plan says
        the latter leaves R / T unchanged; measured here.
absorb  lossy stripe and pillar (eps 6.25 + 0.8i) under harm_asym and under
        the shear, M = 4..8: sum layer_absorption vs 1 - sum R - sum T;
        fail-before = -R as the flux Gram.
mag     PHASE-D PREVIEW (the API refuses mu + map, so the composition is
        injected through ``_stag_map_weights``: chi_t -> g / (sqrt g mu),
        chi33 -> 1 / (sqrt g mu) for a scalar cell mu, i.e. mu' = sqrt g
        g^-1 mu):  (i) a uniform MAGNETIC film (eps 2.0, mu 1.8) under the
        asymmetric stretch and under the shear vs the exact (eps, mu) slab;
        (ii) eps <-> mu DUALITY on the stripe with VACUUM half-spaces:
        R/T_yy(eps = 6.25, mu = 1) == R/T_xx(eps = 1, mu = 6.25) order by
        order, mapped, M = 5..8.
"""
import sys

import _vcommon as C
import numpy as np

from lumenairy.elements.pmm import stack2d_pure as SP, twod_staggered as TS

ARM = sys.argv[1]


def split_rel(s, eps):
    qq = s.q * s.q
    E = np.zeros_like(s.Rmat)
    E[:qq, :qq], E[qq:, qq:] = s.Et_blocks
    if s.Et_offdiag is not None:
        E[:qq, qq:], E[qq:, :qq] = s.Et_offdiag
    return float(np.max(np.abs(E / eps + s.Rmat)) / np.max(np.abs(s.Rmat)))


def maps():
    return {"sine0.12": C.stretch_map(C.sine(0.12)),
            "harm_asym": C.stretch_map(C.HarmonicStretch(0.10, 0.04, 0.9)),
            "shear": C.make_shear_map(0.06, 0.05)}


def run_split():
    rows = []
    for name, cm in maps().items():
        n = cm.shape[0]
        for eps in (1.0 + 0j, 6.25 + 0j, 3.0 + 0.4j):
            for M in (4, 6):
                s = TS.Granet2DTransverseE(C.P, C.P, cm.u_walls, cm.v_walls,
                                           M, np.full((n, n), eps), k0=C.K0,
                                           cmap=cm)
                off = (0.0 if s.Et_offdiag is None else float(max(
                    np.abs(s.Et_offdiag[0]).max(),
                    np.abs(s.Et_offdiag[1]).max())))
                r = dict(map=name, eps=eps, M=M, split=split_rel(s, eps),
                         max_offdiag=off)
                rows.append(r)
                print(r, flush=True)
    C.dump("v5_split", {"rows": rows})


def run_hgram():
    cm = maps()["harm_asym"]
    orig = SP._homog_geom_cache
    orig_rm = SP._region_modes
    res = []
    for kind in ("stripe", "pillar"):
        for M in (6, 8):
            o, R, T, J, _ = C.stack_solve(cm, [C.cell(kind)], M)

            def mixed(solver):
                W0, g2, GW0, SttW0, _Gi, qq = orig(solver)
                return W0, g2, GW0, SttW0, np.linalg.inv(-solver.Rmat), qq
            SP._homog_geom_cache = mixed
            try:
                om, Rm, Tm, Jm, _ = C.stack_solve(cm, [C.cell(kind)], M)
            finally:
                SP._homog_geom_cache = orig

            # CONSISTENT-WRONG: the patterned layer ALSO recovers H through
            # -R (its Ggram_blocks dropped), with the half-spaces as above
            def rm_wrong(sol):
                sol.Ggram_blocks = None
                return orig_rm(sol)
            SP._homog_geom_cache = mixed
            SP._region_modes = rm_wrong
            try:
                oc, Rc, Tc, Jc, _ = C.stack_solve(cm, [C.cell(kind)], M)
            finally:
                SP._homog_geom_cache = orig
                SP._region_modes = orig_rm
            r = dict(kind=kind, M=M, closure=C.closure(R, T),
                     mixed_closure=C.closure(Rm, Tm),
                     mixed_dRT=float(max(np.abs(Rm - R).max(),
                                         np.abs(Tm - T).max())),
                     consistent_closure=C.closure(Rc, Tc),
                     consistent_dRT=float(max(np.abs(Rc - R).max(),
                                              np.abs(Tc - T).max())))
            res.append(r)
            print(r, flush=True)
    C.dump("v5_hgram", {"rows": res})


def run_absorb():
    res = []
    for mname in ("harm_asym", "shear"):
        cm = maps()[mname]
        n = cm.shape[0]
        for kind in ("stripe", "pillar"):
            if mname == "shear":
                c = np.ones((2, 2), complex)
                c[0, :] = 6.25 + 0.8j
                if kind == "pillar":
                    c[0, 1] = 1.0
            else:
                c = C.cell(kind, 6.25 + 0.8j)
            for M in range(4, 9):
                o, R, T, J, st = C.stack_solve(cm, [c], M, retain=True)
                target = 1 - R.sum(axis=1) - T.sum(axis=1)
                A = st.layer_absorption().sum(axis=0)
                s = TS.Granet2DTransverseE(C.P, C.P, cm.u_walls, cm.v_walls,
                                           M, np.full((n, n), 1.0 + 0j),
                                           k0=C.K0, cmap=cm)
                G_ok = st._internal["G"]
                st._internal["G"] = (-s.Rmat).copy()
                Abad = st.layer_absorption().sum(axis=0)
                st._internal["G"] = G_ok
                r = dict(map=mname, kind=kind, M=M,
                         absorbed=[float(x) for x in np.real(target)],
                         closure=float(np.max(np.abs(A - target))),
                         failbefore_minusR=float(np.max(np.abs(Abad - target))))
                res.append(r)
                print(r, flush=True)
    C.dump("v5_absorb", {"rows": res})


# --------------------------------------------------------------- magnetic
_SENT = 1e-300          # imaginary sentinel marking the MAGNETIC layer's cell
_MU = {}


def _inject_mu():
    orig = TS._stag_map_weights

    def weights(bx, by, cmap, eps_cell, rule):
        e = np.asarray(eps_cell)
        mark = np.imag(e) == _SENT
        if not np.any(mark):
            return orig(bx, by, cmap, eps_cell, rule)
        w = orig(bx, by, cmap, np.real(e).astype(complex), rule)
        mu = _MU["mu"][:, :, None, None]
        for k in ("c11", "c12", "c21", "c22", "c33"):
            w[k] = w[k] / mu          # chi = [sqrt g g^-1 mu]^-1 = g/(sg mu)
        return w
    TS._stag_map_weights = weights
    return orig


def airy_mu(eps, mu, d, n1, n3):
    k0 = C.K0
    out = {}
    kz2 = np.sqrt(eps * mu + 0j)
    for pol in ("s", "p"):
        a1 = n1 / 1.0
        a3 = n3 / 1.0
        a2 = kz2 / (mu if pol == "s" else eps)
        if pol == "p":
            a1, a3 = n1 / n1 ** 2, n3 / n3 ** 2
        r12 = (a1 - a2) / (a1 + a2)
        r23 = (a2 - a3) / (a2 + a3)
        t12 = 2 * a1 / (a1 + a2)
        t23 = 2 * a2 / (a2 + a3)
        ph = np.exp(1j * kz2 * k0 * d)
        den = 1 + r12 * r23 * ph ** 2
        r = (r12 + r23 * ph ** 2) / den
        t = t12 * t23 * ph / den
        out[pol] = (float(abs(r) ** 2), float(abs(t) ** 2 * a3.real / a1.real))
    return out


def run_mag():
    import warnings

    from lumenairy.elements.pmm import PMM2DStackPure
    orig = _inject_mu()
    res = {"film": [], "duality": []}
    try:
        for mname in ("harm_asym", "shear"):
            cm = maps()[mname]
            n = cm.shape[0]
            _MU["mu"] = np.full((n, n), 1.8 + 0j)
            cellm = np.full((n, n), 2.0 + 1j * _SENT)
            ex = airy_mu(2.0, 1.8, C.DEPTH, C.N_SUP, C.N_SUB)
            for M in range(4, 9):
                o, R, T, J, _ = C.stack_solve(cm, [cellm], M)
                i0 = C.i00(o)
                errs = []
                for row in (0, 1):
                    Rr, Tr = R[row].copy(), T[row].copy()
                    Rr[i0] -= ex["s"][0]
                    Tr[i0] -= ex["s"][1]
                    errs.append(float(max(abs(Rr).max(), abs(Tr).max())))
                r = dict(map=mname, M=M, err=errs, closure=C.closure(R, T),
                         R00=float(R[0, i0]), R00_exact=ex["s"][0])
                res["film"].append(r)
                print("film", r, flush=True)
        # duality on the stripe, vacuum half-spaces
        for mname in ("harm_asym", "shear"):
            cm = maps()[mname]
            n = cm.shape[0]
            if mname == "shear":
                ce = np.ones((2, 2), complex)
                ce[0, :] = C.EPS_F
            else:
                ce = C.cell("stripe")
            _MU["mu"] = ce.copy()
            cm_e = np.real(ce).astype(complex)
            cm_h = np.ones((n, n), complex) + 1j * _SENT
            for M in range(5, 9):
                out = {}
                for tag, c in (("E", cm_e), ("H", cm_h)):
                    st = PMM2DStackPure(C.P, C.P, n_superstrate=1.0,
                                        n_substrate=1.0, n_modes=M,
                                        n_orders=3, cmap=cm)
                    st.add_layer(C.DEPTH, eps_cell=c)
                    st.set_source(C.WL)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        out[tag] = st.solve()
                oE, RE, TE, JE = out["E"]
                oH, RH, TH, JH = out["H"]
                d = float(max(np.abs(RE[1] - RH[0]).max(),
                              np.abs(TE[1] - TH[0]).max(),
                              np.abs(RE[0] - RH[1]).max(),
                              np.abs(TE[0] - TH[1]).max()))
                # control: the same compare WITHOUT the polarisation swap must
                # be O(1) (the stripe is strongly birefringent)
                ctrl = float(max(np.abs(RE[1] - RH[1]).max(),
                                 np.abs(TE[1] - TH[1]).max()))
                r = dict(map=mname, M=M, duality=d, no_swap_control=ctrl,
                         closure_E=C.closure(RE, TE),
                         closure_H=C.closure(RH, TH))
                res["duality"].append(r)
                print("dual", r, flush=True)
    finally:
        TS._stag_map_weights = orig
    C.dump("v5_mag", res)


{"split": run_split, "hgram": run_hgram, "absorb": run_absorb,
 "mag": run_mag}[ARM]()

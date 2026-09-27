"""P2b -- FAIL-BEFORE arms for the three traps the plan names, measured on the
P2 fixtures under the pure stretch (a = 0.05 px, M = 8) against the SAME
solve done right.

  trap H  : Eq.-25 H partner recovered with the PENCIL's -R (= C[chi_t]C)
            instead of the plain block Gram -- what the shipped
            ``_homog_region_modes`` (Ginv = inv(-Rmat)) and the nonmagnetic
            branch of ``_region_modes`` would do if handed a mapped region.
            Measured INVISIBLE in R/T when EVERY region makes it (-R is the
            same geometry-only operator in every nonmagnetic region under one
            map, so all H partners are multiplied by one matrix and the match
            is unchanged) -- hence the second arm:
  trap Hmix: the patterned layer recovers H with the plain Gram (the shipped
            MAGNETIC branch) while the half-spaces use -R (the shipped
            ``_homog_region_modes``) -- the realistic failure of reusing the
            shipped helpers unchanged.
  trap F  : far-field projector WITHOUT the cofactor cof(J) = det J J^-T,
            i.e. the shipped separable ``_far_projector_2d`` applied to the
            covariant coefficients as if they were Cartesian.
  trap J  : the field pulled back right (E = J^-T E') but the AREA element
            dx dy = det J du dv dropped -- the sketch's "J^T factor on the
            tangential fields" taken alone.
Each arm reports max|dR, dT| over 9 orders x 2 polarizations against the
correct arm, and the lossless closure |sum R + sum T - 1|.

Run:  cd /c/tmp/lum_curved && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved \
        python validation/probe_pmm2d_curved/p2b_failbefore.py
Output: p2b_failbefore.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _curved_scratch as cs  # noqa: E402
import numpy as np  # noqa: E402
from _curved_scratch import (  # noqa: E402
    _C,
    _forward_branch_flip,
    _guarded_lstsq,
    _interface_smatrix,
    _inv_lam,
    _pmm2d_order_kz,
    _project_efficiency,
    _propagation_smatrix,
    _redheffer_star,
)

PX = PY = 1.2
XW = np.array([0.0, 0.25, 0.85, PX])


def modes(sol, h_mode):
    g2, W = cs.eig_pencil(sol.Lmat, -sol.Rmat)
    q = _forward_branch_flip(np.sqrt(np.asarray(g2, dtype=_C)))
    qq = sol.qq
    E11, E12, E21, E22 = sol.Et
    LW = (np.block([[E11, E12], [E21, E22]]) + sol.Stt) @ W
    if h_mode == "gram":
        G1, G2 = sol.Gram
        Dual = np.concatenate([np.linalg.solve(G1, LW[:qq]),
                               np.linalg.solve(G2, LW[qq:])], axis=0)
    else:                                   # trap H: -R in place of the Gram
        Dual = np.linalg.solve(-sol.Rmat, LW)
    rot = np.concatenate([-Dual[qq:], Dual[:qq]], axis=0)
    return W, rot * _inv_lam(q)[None, :], -1j * q


def far(sol, mode, ox):
    if mode == "cof":
        return cs.curved_far_projector(sol, ox, ox)
    g_orig = sol.cmap.geom

    def geom_patch(sx, sy, U, V):
        g = dict(g_orig(sx, sy, U, V))
        if mode == "none":           # treat E' as Cartesian E (det J E = E')
            g = dict(g, xu=np.ones_like(g["xu"]), yv=np.ones_like(g["yv"]),
                     xv=np.zeros_like(g["xv"]), yu=np.zeros_like(g["yu"]))
        elif mode == "JT":           # E = J^-T E' but the AREA element dropped
            xu, xv, yu, yv = g["xu"], g["xv"], g["yu"], g["yv"]
            dj = xu * yv - xv * yu    # cof(J) / det J = J^-T
            g = dict(g, xu=xu / dj, xv=xv / dj, yu=yu / dj, yv=yv / dj)
        return g
    sol.cmap.geom = geom_patch
    try:
        return cs.curved_far_projector(sol, ox, ox)
    finally:
        sol.cmap.geom = g_orig


def solve(fixture, a, M, h_mode="gram", far_mode="cof", h_half=None):
    yw = np.array([0.0, 0.4, 0.8, PY]) if fixture == "stripe" else np.array([0.0, 0.3, 0.9, PY])
    eps = np.ones((3, 3), complex)
    if fixture == "stripe":
        eps[1, :] = 4.0
    else:
        eps[1, 1] = 4.0
    cmap = cs.SineStretchX(a * PX, PX)
    uw = cmap.u_of_x(XW)
    uw[0], uw[-1] = 0.0, PX
    k0 = 2 * np.pi
    sol = cs.CurvedGranet(PX, PY, uw, yw, M, eps, cmap, k0=k0)
    solh = cs.CurvedGranet(PX, PY, uw, yw, M, np.ones((3, 3), complex), cmap, k0=k0)
    Wl, Vl, ll = modes(sol, h_mode)
    h_half = h_mode if h_half is None else h_half
    Wsup, Vsup, _ = modes(solh, h_half)
    solb = cs.CurvedGranet(PX, PY, uw, yw, M, np.full((3, 3), 1.45 ** 2 + 0j), cmap, k0=k0)
    Wsub, Vsub, _ = modes(solb, h_half)
    S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(ll, k0 * 0.5))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
    S11, _, S21, _ = S
    ox = np.arange(-3, 4)
    Pf = far(sol, far_mode, ox)
    Hsup, Hsub = Pf @ Wsup, Pf @ Wsub
    order_x = np.tile(ox, 7)
    order_y = np.repeat(ox, 7)
    kxv = order_x / PX
    kyv = order_y / PY
    kz_ref, kz_trn, kz_inc, sr, st = _pmm2d_order_kz(1.0 + 0j, 1.45 ** 2 + 0j, kxv, kyv, 0.0, 0.0)
    delta = ((order_x == 0) & (order_y == 0)).astype(_C)
    out = {"orders": np.stack([order_x, order_y], 1), "R": {}, "T": {}}
    for pol in ("te", "tm"):
        ex0, ey0 = (0.0, 1.0) if pol == "te" else (1.0, 0.0)
        c = _guarded_lstsq(Hsup, np.concatenate([ex0 * delta, ey0 * delta]), "p2b")
        r = Hsup @ (S11 @ c)
        t = Hsub @ (S21 @ c)
        n = order_x.size
        rz = -(kxv * r[:n] + kyv * r[n:]) / sr
        tz = -(kxv * t[:n] + kyv * t[n:]) / st
        R, T = _project_efficiency(np, kz_ref, kz_trn, kz_inc, r[:n], r[n:], rz,
                                   t[:n], t[n:], tz, 1.0)
        out["R"][pol] = np.real(np.asarray(R))
        out["T"][pol] = np.real(np.asarray(T))
    v = np.concatenate([cs.vec(out, "te"), cs.vec(out, "tm")])
    clo = max(abs(out["R"][p].sum() + out["T"][p].sum() - 1) for p in ("te", "tm"))
    return v, float(clo)


def main():
    res = {"env": cs.env_record(), "a_over_px": 0.05, "M": 8, "rows": []}
    for fx in ("stripe", "pillar"):
        ref, clo0 = solve(fx, 0.05, 8)
        res["rows"].append({"fixture": fx, "arm": "correct", "closure": clo0, "max_dRT": 0.0})
        for arm, kw in (("trap_H_minusR_gram", {"h_mode": "minusR"}),
                        ("trap_Hmix_halfspaces_minusR", {"h_half": "minusR"}),
                        ("trap_F_no_cofactor", {"far_mode": "none"}),
                        ("trap_J_no_area_element", {"far_mode": "JT"})):
            v, clo = solve(fx, 0.05, 8, **kw)
            row = {"fixture": fx, "arm": arm, "closure": clo,
                   "max_dRT": float(np.max(np.abs(v - ref)))}
            res["rows"].append(row)
            print(json.dumps(row), flush=True)
        with open(os.path.join(cs.HERE, "p2b_failbefore.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

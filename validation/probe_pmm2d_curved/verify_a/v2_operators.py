"""V2 -- the operators, re-extracted by the verifier.

ident  IdentityMap through the quadrature path vs cmap=None (the kron path)
       on the verifier's fixture: every retained operator, at M = 4..7
       (odd and even), normal and oblique Bloch phases, integer and
       non-uniform walls; and the end-to-end R / T / Jones of a stack.
sep    an INDEPENDENT oracle of the mapped operators under a SEPARABLE map
       x = f(u) (the verifier's asymmetric two-harmonic stretch, y = v) on
       the y-uniform STRIPE: for a separable map and a y-uniform cell every
       weighted 2-D block factors as kron(Y, X) of 1-D weighted masses /
       derivatives, which this probe builds from scratch with a 2000-node
       Gauss rule per segment (self-gap vs 3000 nodes recorded), then
       re-assembles R, [eps_t], S_tt, Schur and L by the module's formulas.
       The library's operators are compared at its ADAPTIVE rule and at
       forced node counts nq = 2M+8 .. 512 (the quadrature floor, F1).

usage: v2_operators.py ident | sep
"""
import sys

import _vcommon as C
import numpy as np
from numpy.polynomial.legendre import leggauss

from lumenairy.elements.pmm import twod_staggered as TS
from lumenairy.elements.pmm._curvemap import IdentityMap

ARM = sys.argv[1] if len(sys.argv) > 1 else None


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def ops(s):
    qq = s.q * s.q
    G = np.zeros_like(s.Rmat)
    if s.Ggram_blocks is not None:
        G[:qq, :qq], G[qq:, qq:] = s.Ggram_blocks
    else:
        G = -s.Rmat
    return dict(Rmat=s.Rmat, Lmat=s.Lmat, Stt=s.Stt, Schur=s.Schur,
                Et11=s.Et_blocks[0], Et22=s.Et_blocks[1], Gram=G)


def run_ident():
    res = []
    for wx, wy, tag in ((3, 3, "int3"), (C.XW, C.YW, "nonu")):
        for M in (4, 5, 6, 7):
            for a0 in (0.0, 0.8):
                eps = C.cell("pillar")
                kw = dict(alpha0x=a0 * C.K0, alpha0y=-0.4 * a0 * C.K0,
                          k0=C.K0)
                s0 = TS.Granet2DTransverseE(C.P, C.P, wx, wy, M, eps, **kw)
                cm = IdentityMap(wx, wy, C.P, C.P)
                s1 = TS.Granet2DTransverseE(C.P, C.P, wx, wy, M, eps,
                                            cmap=cm, **kw)
                o0, o1 = ops(s0), ops(s1)
                row = dict(walls=tag, M=M, a0=a0,
                           nq=int(s1._qrule[0].size),
                           **{k: rel(o1[k], o0[k]) for k in o0})
                row["offdiag_zero"] = bool(
                    not np.any(s1.Et_offdiag[0])
                    and not np.any(s1.Et_offdiag[1]))
                res.append(row)
                print(row, flush=True)
    e2e = []
    for M in (5, 6):
        for th, ph in ((0.0, 0.0), (np.radians(20), np.radians(35))):
            # integer 3 x 3 walls: the unmapped SHARED-grid stack takes a
            # uniform lattice only, so this is the like-for-like reference
            cm = IdentityMap(3, 3, C.P, C.P)
            o1, R1, T1, J1, _ = C.stack_solve(cm, [C.cell("pillar")], M,
                                              theta=th, phi=ph)
            o0, R0, T0, J0, _ = C.stack_solve(None, [C.cell("pillar")], M,
                                              theta=th, phi=ph)
            row = dict(M=M, theta=th, phi=ph,
                       dR=float(np.abs(R1 - R0).max()),
                       dT=float(np.abs(T1 - T0).max()),
                       dJ=float(np.abs(J1 - J0).max()))
            e2e.append(row)
            print(row, flush=True)
    C.dump("v2_ident", {"operators": res, "end_to_end": e2e})


# ----------------------------------------------------------------- sep oracle
def w1d(basis, lset, op, rset, wfun, eps_seg, nq):
    """INT conj(L_i)^(a) w(u) R_j^(b) du over the period, by an nq-node Gauss
    rule per segment; op 'm' (no derivative), 'd' (on R), 'dL' (on L)."""
    xg, wg = leggauss(nq)
    V, Vp = TS._modleg_value_deriv(basis.M, xg)
    L = np.asarray(getattr(basis, lset))
    R = np.asarray(getattr(basis, rset))
    out = np.zeros((L.shape[0], R.shape[0]), complex)
    for s in range(basis.N):
        u = 0.5 * (basis.xb[s] + basis.xb[s + 1]) + basis.Jn[s] * xg
        w = wfun(u) * eps_seg[s]
        fa, gc, sc = ((V, V, basis.Jn[s]) if op == "m" else
                      (V, Vp, 1.0) if op == "d" else (Vp, V, 1.0))
        Lv = np.conj(L[:, s, :]) @ fa          # (nL, nq)
        Rv = R[:, s, :] @ gc
        out += (Lv * (w * wg * sc)) @ Rv.T
    return out


def oracle_ops(bx, by, f, eps_x, k0, nq):
    """The mapped operators of a y-uniform cell (eps_x per u-segment) under
    the separable map x = f(u), y = v, built from 1-D factors.  With
    f' = df/du: sqrt g = f', chi11 = f', chi22 = 1/f', chi33 = 1/f',
    e11 = eps/f', e22 = eps f', e33 = eps f' (no shear)."""
    per = bx.d

    def fp(u):
        return f(u, per)[1]

    def inv_fp(u):
        return 1.0 / f(u, per)[1]

    def one(u):
        return np.ones_like(u)
    ones_x = np.ones(bx.N)
    ones_y = np.ones(by.N)
    X = lambda l, op, r, w, e: w1d(bx, l, op, r, w, e, nq)   # noqa: E731
    Y = lambda l, op, r: w1d(by, l, op, r, one, ones_y, nq)  # noqa: E731
    K = np.kron
    B, T = "B", "Btilde"
    R11 = -K(Y(T, "m", T), X(B, "m", B, inv_fp, ones_x))
    R22 = -K(Y(B, "m", B), X(T, "m", T, fp, ones_x))
    Et11 = K(Y(T, "m", T), X(B, "m", B, inv_fp, eps_x))
    Et22 = K(Y(B, "m", B), X(T, "m", T, fp, eps_x))
    Mbb_x, Mbb_y = X(B, "m", B, one, ones_x), Y(B, "m", B)
    dbt_x, dbt_y = X(B, "d", T, one, ones_x) / k0, Y(B, "d", T) / k0
    Curl = np.concatenate([K(dbt_y, Mbb_x), -K(Mbb_y, dbt_x)], axis=1)
    Gw = K(Mbb_y, Mbb_x)
    Gw_chi = K(Y(B, "m", B), X(B, "m", B, inv_fp, ones_x))
    Gwi = np.linalg.inv(Gw)
    Stt = -Curl.conj().T @ (Gwi @ Gw_chi @ Gwi) @ Curl
    Ktz = np.concatenate([
        -K(Y(T, "m", T), X(B, "d", T, inv_fp, ones_x)) / k0,
        -K(Y(B, "d", T), X(T, "m", T, fp, ones_x)) / k0], axis=0)
    Meps33 = K(Y(T, "m", T), X(T, "m", T, fp, eps_x))
    Kzt = np.concatenate([
        -K(Y(T, "m", T), X(T, "dL", B, inv_fp, eps_x)) / k0,
        -K(Y(T, "dL", B), X(T, "m", T, fp, eps_x)) / k0], axis=1)
    Schur = Ktz @ np.linalg.solve(Meps33, Kzt)
    qq = R11.shape[0]
    Rm = np.zeros((2 * qq, 2 * qq), complex)
    Rm[:qq, :qq], Rm[qq:, qq:] = R11, R22
    Lm = np.zeros_like(Rm)
    Lm[:qq, :qq], Lm[qq:, qq:] = Et11, Et22
    Lm += Stt - Schur
    G = np.zeros_like(Rm)
    G[:qq, :qq] = K(Y(T, "m", T), Mbb_x)
    G[qq:, qq:] = K(Mbb_y, X(T, "m", T, one, ones_x))
    return dict(Rmat=Rm, Lmat=Lm, Stt=Stt, Schur=Schur, Et11=Et11,
                Et22=Et22, Gram=G)


def run_sep():
    f = C.HarmonicStretch(0.10, 0.04, 0.9)
    res = {"stretch": f.key(), "min_slope": f.min_slope(), "rows": []}
    cm = C.stretch_map(f)
    eps = C.cell("stripe")
    eps_x = eps[:, 0]
    orig_nodes = TS._stag_map_nodes
    for M in (4, 6, 8):
        a0 = 0.0
        bx = TS.Basis1D(C.P, cm.u_walls, M)
        by = TS.Basis1D(C.P, cm.v_walls, M)
        ref = oracle_ops(bx, by, f, eps_x, C.K0, 2000)
        ref2 = oracle_ops(bx, by, f, eps_x, C.K0, 3000)
        gap = max(rel(ref[k], ref2[k]) for k in ref)
        adaptive = orig_nodes(bx, by, cm, M)
        for nq in ("adaptive", 2 * M + 8, 2 * (2 * M + 8), 4 * (2 * M + 8),
                   8 * (2 * M + 8), 256, 512):
            if nq == "adaptive":
                TS._stag_map_nodes = orig_nodes
            else:
                TS._stag_map_nodes = (lambda *a, _n=nq, **k: _n)
            try:
                s = TS.Granet2DTransverseE(C.P, C.P, cm.u_walls, cm.v_walls,
                                           M, eps, k0=C.K0, cmap=cm,
                                           alpha0x=a0)
            finally:
                TS._stag_map_nodes = orig_nodes
            o = ops(s)
            row = dict(M=M, nq=nq, nq_used=int(s._qrule[0].size),
                       oracle_selfgap=gap,
                       **{k: rel(o[k], ref[k]) for k in ref})
            res["rows"].append(row)
            print(row, flush=True)
        res.setdefault("adaptive_nq", {})[M] = int(adaptive)
    C.dump("v2_sep_oracle", res)


if __name__ == "__main__":
    {"ident": run_ident, "sep": run_sep}[ARM]()

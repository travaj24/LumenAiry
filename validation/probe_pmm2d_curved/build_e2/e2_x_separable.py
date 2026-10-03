"""E2-X: an INDEPENDENT check of the curved cross-mass on two SEPARABLE
stretches, where the 2-D integral factors exactly: with x = f_a(u_a) and
x = f_b(u_b) the transition is u_a = psi(u_b) = f_a^-1(f_b(u_b)), and the
covariant pullback E'_{b,u} = psi'(u_b) E'_{a,u}(psi(u_b)) puts the factor
psi' on the x-integral of the u-component only.  So

  X11 = kron( Cy[Btilde],  Cx'[B] ),   X22 = kron( Cy'[B], Cx[Btilde] ),
  X12 = X21 = 0,

with C[i, j] = INT conj(phi^b_i(s)) phi^a_j(psi(s)) ds and C' the same with
psi'(s) inside, each a 1-D integral evaluated here by brute force: Gauss on
the union of b's segments and the PREIMAGES of a's walls, 200 nodes each.
Nothing of the 2-D kernel is shared but the basis stencils."""
import sys

import numpy as np
from _common import CM, TS, P, dump
from numpy.polynomial.legendre import leggauss

from lumenairy.elements.pmm import _curvemortar as CMM

M = int(sys.argv[1]) if len(sys.argv) > 1 else 5
tau = np.exp(-0.3j)
sa = CM.SeparableStretch.from_physical_walls(
    np.array([0, 0.3, 0.9, P]), np.array([0, 0.3, 0.9, P]),
    fx=CM.SineStretch(0.06 * P))
sb = CM.SeparableStretch.from_physical_walls(
    np.array([0, 0.5, 0.8, P]), np.array([0, 0.2, 0.7, P]),
    fy=CM.SineStretch(0.08 * P), fx=CM.SineStretch(-0.03 * P))
ga = TS.StagGridOps(P, P, sa.u_walls, sa.v_walls, M, tau, tau, cmap=sa)
gb = TS.StagGridOps(P, P, sb.u_walls, sb.v_walls, M + 1, tau, tau, cmap=sb)


def f_of(st, axis):
    return st.fx if axis == "x" else st.fy


def psi(s, axis):
    """u_a = f_a^-1(f_b(s)) and its derivative."""
    fb = f_of(sb, axis)
    fa = f_of(sa, axis)
    x, dx = (s, np.ones_like(s)) if fb is None else fb(s, P)
    if fa is None:
        return x, dx
    u = fa.inverse(x, P)
    _f, dfa = fa(u, P)
    return u, dx / dfa


def cross1d(basis_a, basis_b, which, axis, deriv):
    Sa = np.asarray(getattr(basis_a, which))
    Sb = np.asarray(getattr(basis_b, which))
    cuts = np.concatenate([basis_b.xb, [0.0, P]])
    # preimages in b of a's walls
    fb = f_of(sb, axis)
    fa = f_of(sa, axis)
    xa = basis_a.xb if fa is None else fa(basis_a.xb, P)[0]
    pre = xa if fb is None else fb.inverse(xa, P)
    cuts = np.unique(np.concatenate([cuts, pre]))
    xg, wg = leggauss(200)
    C = np.zeros((Sb.shape[0], Sa.shape[0]), complex)
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        if s1 - s0 < 1e-14:
            continue
        s = 0.5 * (s0 + s1) + 0.5 * (s1 - s0) * xg
        w = 0.5 * (s1 - s0) * wg
        ua, dpsi = psi(s, axis)
        eb = np.clip(np.searchsorted(basis_b.xb, 0.5 * (s0 + s1)) - 1, 0,
                     basis_b.N - 1)
        ea = np.clip(np.searchsorted(basis_a.xb, np.mean(ua)) - 1, 0,
                     basis_a.N - 1)
        Vb = CMM._local_vals(basis_b, eb, s)
        Va = CMM._local_vals(basis_a, ea, ua)
        ww = w * (dpsi if deriv else 1.0)
        C += np.conj(Sb[:, eb, :]) @ (Vb * ww) @ Va.T @ Sa[:, ea, :].T
    return C


X = CMM.curved_cross_mass_adaptive(ga, gb)[0]
X11 = np.kron(cross1d(ga.by, gb.by, "Btilde", "y", False),
              cross1d(ga.bx, gb.bx, "B", "x", True))
X22 = np.kron(cross1d(ga.by, gb.by, "B", "y", True),
              cross1d(ga.bx, gb.bx, "Btilde", "x", False))
qa, qb = ga.qq, gb.qq
sc = np.abs(X).max()
out = dict(M=M,
           X11=float(np.abs(X[:qb, :qa] - X11).max() / sc),
           X22=float(np.abs(X[qb:, qa:] - X22).max() / sc),
           X12=float(np.abs(X[:qb, qa:]).max() / sc),
           X21=float(np.abs(X[qb:, :qa]).max() / sc))
# the WRONG weight (no psi') as a fail-before of this check
X11w = np.kron(cross1d(ga.by, gb.by, "Btilde", "y", False),
               cross1d(ga.bx, gb.bx, "B", "x", False))
out["X11_without_psi_prime"] = float(np.abs(X[:qb, :qa] - X11w).max() / sc)
print(out)
dump(f"e2_x_separable_M{M}.json", out)

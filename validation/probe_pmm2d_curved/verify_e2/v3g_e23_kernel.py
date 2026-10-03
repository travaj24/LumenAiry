"""E2-3 kernel level on the verifier's own pair of separable stretches.

Derivation (the verifier's): with x = f_a(u_a), y = h_a(v_a) and
x = f_b(u_b), y = h_b(v_b), the transition Psi = Phi_a^-1 o Phi_b is
(psi_x(u_b), psi_y(v_b)) with psi_x = f_a^-1 o f_b, psi_y = h_a^-1 o h_b, and
T = J_a^-1 J_b = diag(psi_x'(u_b), psi_y'(v_b)).  The kernel's
X_{beta alpha} = INT conj(f^b_beta) T_{alpha beta} f^a_alpha(Psi) du_b dv_b
then has T_12 = T_21 = 0 (X12 = X21 = 0) and
  X11 (component u, V1 = B(u) x Btilde(v)):  T_11 = psi_x'(u_b) multiplies
        the x (u) integral of the B functions -> kron(Cy[Btilde], Cx'[B])
  X22 (component v, V2 = Btilde(u) x B(v)):  T_22 = psi_y'(v_b) multiplies
        the y (v) integral of the B functions -> kron(Cy'[B], Cx[Btilde])
(kron(Y, X): x index fastest, the kernel's  sv * qx + su  ordering), with
C[i, j] = INT conj(phi^b_i(s)) phi^a_j(psi(s)) ds and C' the same with
psi'(s) in the integrand.

The 1-D integrals are brute-forced here with the verifier's OWN
modified-Legendre evaluator (scipy eval_legendre), Gauss on the union of
b's walls and the preimages of a's walls, n_g nodes per piece (two values,
to show the oracle converged).  The evaluator is first validated against
the shipped 1-D Grams (Mtt = Btilde mass, Mbb = B mass).
Usage: v3g_e23_kernel.py Ma Mb"""
import sys

import numpy as np
from _ve import dump
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre

from lumenairy.elements.pmm import _curvemap as CM, _curvemortar as CMM, twod_staggered as TS

Ma = int(sys.argv[1]) if len(sys.argv) > 1 else 5
Mb = int(sys.argv[2]) if len(sys.argv) > 2 else 6
P = 1.1
tx, ty = np.exp(-0.41j), np.exp(0.87j)          # Bloch phases (x != y)
sa = CM.SeparableStretch.from_physical_walls(
    np.array([0, 0.2, 0.7, P]), np.array([0, 0.35, 0.95, P]),
    fx=CM.SineStretch(0.05 * P), fy=CM.SineStretch(-0.07 * P))
sb = CM.SeparableStretch.from_physical_walls(
    np.array([0, 0.45, 1.0, P]), np.array([0, 0.1, 0.6, P]),
    fx=CM.SineStretch(-0.04 * P), fy=CM.SineStretch(0.11 * P))
ga = TS.StagGridOps(P, P, sa.u_walls, sa.v_walls, Ma, tx, ty, cmap=sa)
gb = TS.StagGridOps(P, P, sb.u_walls, sb.v_walls, Mb, tx, ty, cmap=sb)


def loc(M, ref):
    """verifier's modified-Legendre local functions (M, n)."""
    out = [0.5 * (1 - ref), 0.5 * (1 + ref)]
    for a in range(2, M):
        out.append(eval_legendre(a, ref) - eval_legendre(a - 2, ref))
    return np.array(out[:M])


def gvals(basis, which, seg, s):
    S = np.asarray(getattr(basis, which))[:, seg, :]       # (dim, M)
    xl, xr = basis.xb[seg], basis.xb[seg + 1]
    return S @ loc(basis.M, (2 * s - xl - xr) / (xr - xl))   # (dim, n)


def stretch(st, axis):
    return st.fx if axis == "x" else st.fy


def psi(s, axis):
    fb, fa = stretch(sb, axis), stretch(sa, axis)
    x, dx = fb(s, P)
    u = fa.inverse(x, P)
    return u, dx / fa(u, P)[1]


def C1d(ba, bb, which, axis, deriv, ng, identity=False):
    if identity:
        cuts = np.asarray(bb.xb)
    else:
        fa, fb = stretch(sa, axis), stretch(sb, axis)
        pre = fb.inverse(fa(np.asarray(ba.xb), P)[0], P)
        cuts = np.unique(np.concatenate([bb.xb, pre]))
    xg, wg = leggauss(ng)
    C = np.zeros((bb.dim, ba.dim), complex)
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        if s1 - s0 < 1e-13:
            continue
        s = 0.5 * (s0 + s1) + 0.5 * (s1 - s0) * xg
        w = 0.5 * (s1 - s0) * wg
        if identity:
            ua, dp = s, np.ones_like(s)
        else:
            ua, dp = psi(s, axis)
        eb = int(np.clip(np.searchsorted(bb.xb, 0.5 * (s0 + s1)) - 1, 0,
                         bb.N - 1))
        ea = int(np.clip(np.searchsorted(ba.xb, np.mean(ua)) - 1, 0,
                         ba.N - 1))
        assert ba.xb[ea] - 1e-12 <= ua.min() and ua.max() <= ba.xb[ea + 1] + 1e-12
        Vb = gvals(bb, which, eb, s)
        Va = gvals(ba, which, ea, ua)
        C += np.conj(Vb) @ ((w * (dp if deriv else 1.0))[:, None] * Va.T)
    return C


out = {"Ma": Ma, "Mb": Mb}
# 0. validate the verifier's evaluator against the shipped Grams
out["eval_vs_Mtt_x"] = float(np.abs(C1d(ga.bx, ga.bx, "Btilde", "x", False,
                                         40, identity=True) - ga.Mtt_x).max()
                             / np.abs(ga.Mtt_x).max())
out["eval_vs_Mbb_y"] = float(np.abs(C1d(ga.by, ga.by, "B", "y", False, 40,
                                         identity=True) - ga.Mbb_y).max()
                             / np.abs(ga.Mbb_y).max())
res = {}
for ng in (40, 80):
    res[ng] = dict(
        X11=np.kron(C1d(ga.by, gb.by, "Btilde", "y", False, ng),
                    C1d(ga.bx, gb.bx, "B", "x", True, ng)),
        X22=np.kron(C1d(ga.by, gb.by, "B", "y", True, ng),
                    C1d(ga.bx, gb.bx, "Btilde", "x", False, ng)))
out["oracle_40_vs_80"] = float(max(np.abs(res[40][k] - res[80][k]).max()
                                   for k in ("X11", "X22")))
X11o, X22o = res[80]["X11"], res[80]["X22"]
X, n, chg = CMM.curved_cross_mass_adaptive(ga, gb)
qa, qb = ga.qq, gb.qq
sc = np.abs(X).max()
out.update(n_adaptive=n, change_adaptive=chg, scale=float(sc),
           X11=float(np.abs(X[:qb, :qa] - X11o).max() / sc),
           X22=float(np.abs(X[qb:, qa:] - X22o).max() / sc),
           X12=float(np.abs(X[:qb, qa:]).max() / sc),
           X21=float(np.abs(X[qb:, :qa]).max() / sc))
# the swapped kernel direction (b above a) must give the analogous blocks
# fail-befores: no psi', and psi' on the WRONG axis
X11_nopsi = np.kron(C1d(ga.by, gb.by, "Btilde", "y", False, 80),
                    C1d(ga.bx, gb.bx, "B", "x", False, 80))
X11_wrongaxis = np.kron(C1d(ga.by, gb.by, "Btilde", "y", True, 80),
                        C1d(ga.bx, gb.bx, "B", "x", False, 80))
out["fb_X11_without_psi_prime"] = float(np.abs(X[:qb, :qa] - X11_nopsi).max()
                                        / sc)
out["fb_X11_psi_prime_on_y"] = float(np.abs(X[:qb, :qa] - X11_wrongaxis).max()
                                     / sc)
# fixed-n convergence of the kernel against the oracle
out["fixed_n"] = {}
for nn in (6, 10, 14, 20):
    Xn = CMM.curved_cross_mass(ga, gb, nn)
    out["fixed_n"][nn] = float(max(np.abs(Xn[:qb, :qa] - X11o).max(),
                                   np.abs(Xn[qb:, qa:] - X22o).max()) / sc)
# H-row operator: cross_h_from_x must be [[X22^H, -X12^H], [-X21^H, X11^H]]
H = CMM.cross_h_from_x(X, qa, qb)
Hexp = np.block([[X22o.conj().T, np.zeros((qa, qb))],
                 [np.zeros((qa, qb)), X11o.conj().T]])
out["H_vs_oracle"] = float(np.abs(H - Hexp).max() / sc)
print(out)
dump(f"v3g_e23_kernel_Ma{Ma}_Mb{Mb}", out)

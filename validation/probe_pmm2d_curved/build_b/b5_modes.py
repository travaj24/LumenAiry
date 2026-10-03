"""B5 -- the IN-PLANE convergence rate, isolated: the layer's own Bloch-mode
propagation constants (the leading eigenvalues n_eff^2 = gamma^2 / k0^2 of
the region pencil (L, -R) at normal incidence) against M -- the quantity
Weiss et al. (Opt. Express 17, 8051 (2009), Fig. 2b) rate their
matched-coordinate FMM by.  They see ONLY the in-plane cross-section (no
rim), so a smooth curved interface must converge spectrally and a square
pillar (four 90-degree corners) must not.

The pencil comes from the LIBRARY (``Granet2DTransverseE``, the corner rule
at the singular vertices); its right-hand matrix -R is Hermitian positive
definite, so the eigenvalues are taken by Cholesky whitening + a standard
eig (the planner's P5 route; the eigenvalues do not depend on the
eigensolver).  Cases (period 1.2, eps 4 in air):

  circle3 : r 0.36, the 3 x 3 transfinite map
  circle5 : the same circle, 5 x 5 map
  rect3   : square pillar side 0.6 (3 x 3, no map) -- corner-capped
  stripe3 : y-uniform ridge x in [0.3, 0.9] (3 x 3, no map) -- corner-free
  fillet<r> : side 0.6, fillet r / side = r (5 x 5 map)

  python validation/probe_pmm2d_curved/build_b/b5_modes.py <case> <Mlo> <Mhi>
Output: b5_modes_<case>.json (top K = 8 eigenvalues per rung)
"""
import sys
import time

import _common as C
import numpy as np
import scipy.linalg as sla

K = 8


def build(case):
    if case in ("rect3", "stripe3"):
        w = np.array([0.0, 0.3, 0.9, C.P])
        eps = np.ones((3, 3), complex)
        if case == "rect3":
            eps[1, 1] = C.EPS_P
        else:
            eps[1, :] = C.EPS_P
        return None, w, eps
    if case == "circle3":
        cm, eps = C.circle3()
    elif case == "circle5":
        cm, eps = C.circle5()
    elif case.startswith("fillet"):
        cm, eps = C.fillet5(float(case[6:]))
    else:
        raise SystemExit(case)
    return cm, cm.u_bounds, eps


def top_eigs(sol):
    B = -sol.Rmat
    B = 0.5 * (B + B.conj().T)
    Lc = sla.cholesky(B, lower=True)
    A = sla.solve_triangular(Lc, sol.Lmat, lower=True)
    A = sla.solve_triangular(Lc, A.conj().T, lower=True).conj().T
    g2 = sla.eigvals(A, overwrite_a=True, check_finite=False)
    return g2[np.argsort(-g2.real)][:K]


def main(case, m_lo, m_hi):
    cm, w, eps = build(case)
    res = {"env": C.env_record(), "case": case, "walls": list(map(float, w)),
           "K": K, "runs": []}
    for M in range(m_lo, m_hi + 1):
        t0 = time.perf_counter()
        sol = C.TS.Granet2DTransverseE(C.P, C.P, w, w, M, eps,
                                       k0=2 * np.pi, cmap=cm)
        top = top_eigs(sol)
        res["runs"].append({"M": M, "dof": 2 * sol.q * sol.q,
                            "g2_top": [[float(z.real), float(z.imag)]
                                       for z in top],
                            "t": time.perf_counter() - t0})
        print(case, M, np.round(top.real[:4], 10), flush=True)
        C.dump(f"b5_modes_{case}.json", res)
        del sol
    runs = res["runs"]
    for a, b in zip(runs[:-1], runs[1:]):
        ga = np.array([complex(*z) for z in a["g2_top"][:4]])
        gb = np.array([complex(*z) for z in b["g2_top"][:4]])
        a["d_next4"] = float(np.max(np.abs(ga - gb)))
    C.dump(f"b5_modes_{case}.json", res)
    print([(r["M"], r.get("d_next4")) for r in runs])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

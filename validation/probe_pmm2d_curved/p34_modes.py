"""P3/P4 companion -- the IN-PLANE convergence rate, isolated.

A finite-height pillar has TWO kinds of edge singularity: the VERTICAL edges
(the in-plane corners of the cross-section -- what a fillet removes) and the
RIM edges where the flat top/bottom faces meet the side wall (present for
every pillar, circular or not, and for a 1-D ridge).  The diffraction
efficiencies see both.  The layer's own Bloch-mode propagation constants
(the eigenvalues n_eff^2 of the region pencil at normal incidence) see ONLY
the in-plane cross-section -- Weiss et al. 2009 use exactly this quantity
(Fig. 2b) to rate their matched-coordinate FMM on a dielectric cylinder.

For each cross-section, the TOP-K eigenvalues (largest Re n_eff^2 -- the
propagating and least-evanescent Bloch modes) are recorded against M, and the
ladder is read against its own top rung.

  rect3     : square pillar side 0.6 (3x3 identity)       -- corner-capped
  circle3   : circle r 0.36, 3x3 transfinite map          -- smooth interface,
              map det J = 0 at 4 vertices
  circle5   : the same circle on the 5x5 topology
  fillet<r> : side 0.6 with fillet r/side = 0.05, 0.1, 0.2 (5x5 map)
  stripe3   : y-uniform ridge x in [0.3, 0.9] (3x3)      -- corner-free control

Run:  cd /c/tmp/lum_curved && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved \
        python validation/probe_pmm2d_curved/p34_modes.py circle3 4 12
Output: p34_modes_<case>.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _curved_scratch as cs  # noqa: E402
import numpy as np  # noqa: E402

P = 1.2
K = 8


def build(case):
    if case in ("rect3", "stripe3"):
        w = np.array([0.0, 0.3, 0.9, P])
        eps = np.ones((3, 3), complex)
        if case == "rect3":
            eps[1, 1] = 4.0
        else:
            eps[1, :] = 4.0
        return None, w, eps
    if case == "circle3":
        cmap, w = cs.circle_map_3x3(P, 0.36)
        eps = np.ones((3, 3), complex)
        eps[1, 1] = 4.0
        return cmap, w, eps
    if case == "circle5":
        cmap, w = cs.circle_map_5x5(P, 0.36)
    elif case.startswith("fillet"):
        ratio = float(case[6:])
        cmap, w = cs.fillet_map_5x5(P, 0.3, ratio * 0.6)
    else:
        raise SystemExit(case)
    eps = np.ones((5, 5), complex)
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            eps[i, j] = 4.0
    return cmap, w, eps


def main():
    case = sys.argv[1]
    m_lo, m_hi = int(sys.argv[2]), int(sys.argv[3])
    cmap, w, eps = build(case)
    res = {"env": cs.env_record(), "case": case, "walls": w.tolist(),
           "K": K, "runs": []}
    for M in range(m_lo, m_hi + 1):
        t0 = time.perf_counter()
        sol = cs.CurvedGranet(P, P, w, w, M, eps, cmap, k0=2 * np.pi)
        g2, _ = cs.eig_pencil(sol.Lmat, -sol.Rmat)
        order = np.argsort(-g2.real)
        top = g2[order[:K]]
        res["runs"].append({"M": M, "dof": 2 * sol.qq,
                            "g2_top": [[float(z.real), float(z.imag)] for z in top],
                            "t": time.perf_counter() - t0})
        print(f"{case} M={M} dof={2 * sol.qq} g2={np.round(top.real[:4], 10)}",
              flush=True)
        with open(os.path.join(cs.HERE, f"p34_modes_{case}.json"), "w") as f:
            json.dump(res, f, indent=1)
        del sol


if __name__ == "__main__":
    main()

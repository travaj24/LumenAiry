"""P4 -- the FILLET question: a square dielectric pillar whose four corners are
rounded with radius r, r/side in {0, 0.05, 0.1, 0.2}.

  * how much does the zeroth-order efficiency move with r?
  * does the modal convergence become SPECTRAL once r > 0 (the corner-cap
    gone), or does the map's own det J = 0 at the four 45-degree fillet points
    keep it algebraic?

Fixture (lambda = 1): cell 1.2 x 1.2, pillar side 0.6 centred (the P2 pillar's
size), eps 4 in air, height 0.5, air over n = 1.45, normal incidence, both
polarizations.
  r = 0     : the 3x3 rectangular cell (walls 0.3 / 0.9), identity map (P1 shows
              this equals the shipped assembly to 1e-14)
  r > 0     : fillet_map_5x5 -- a 5x5 wall grid whose walls pass through the
              45-degree fillet points and the arc/line tangency points, so every
              cell edge is a pure arc or a pure line and the map is analytic
              inside every cell.
Run:  cd /c/tmp/lum_curved && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved \
        python validation/probe_pmm2d_curved/p4_fillet.py 0.1 4 8
Output: p4_fillet_r<ratio>.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _curved_scratch as cs  # noqa: E402
import numpy as np  # noqa: E402

P = 1.2
SIDE = 0.6
EPS_P = 4.0
WL = 1.0
DEPTH = 0.5
N_SUP, N_SUB = 1.0, 1.45


def eqarea(m_list=(9, 10, 11)):
    """Control: the SQUARE of the same area as each filleted pillar (3x3,
    identity map) -- separates the fillet's SHAPE effect from its AREA effect."""
    res = {"env": cs.env_record(), "rows": []}
    for ratio in (0.05, 0.1, 0.2):
        rf = ratio * SIDE
        area = SIDE ** 2 - (4 - np.pi) * rf ** 2
        s = np.sqrt(area)
        w = np.array([0.0, (P - s) / 2, (P + s) / 2, P])
        eps = np.ones((3, 3), complex)
        eps[1, 1] = EPS_P
        for M in m_list:
            out = cs.solve_curved(P, P, w, w, M, eps, N_SUP, N_SUB, DEPTH, WL,
                                  cmap=None, n_orders=3)
            row = {"fillet_over_side": ratio, "equal_area_side": s, "M": M,
                   "te": cs.table(out, "te"), "tm": cs.table(out, "tm"),
                   "vec_te": cs.vec(out, "te").tolist(),
                   "vec_tm": cs.vec(out, "tm").tolist()}
            res["rows"].append(row)
            print(f"eqarea r/side={ratio} side={s:.6f} M={M} "
                  f"R00={row['te']['0,0'][0]:.10f} T00={row['te']['0,0'][1]:.10f}",
                  flush=True)
            with open(os.path.join(cs.HERE, "p4_fillet_equal_area_squares.json"), "w") as f:
                json.dump(res, f, indent=1)


def main():
    if sys.argv[1] == "eqarea":
        eqarea()
        return
    ratio = float(sys.argv[1])
    m_lo, m_hi = int(sys.argv[2]), int(sys.argv[3])
    rf = ratio * SIDE
    if rf == 0:
        cmap = None
        w = np.array([0.0, (P - SIDE) / 2, (P + SIDE) / 2, P])
        eps = np.ones((3, 3), complex)
        eps[1, 1] = EPS_P
        pillar = [(1, 1)]
    else:
        cmap, w = cs.fillet_map_5x5(P, SIDE / 2, rf)
        eps = np.ones((5, 5), complex)
        pillar = [(i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
        for (i, j) in pillar:
            eps[i, j] = EPS_P
    area = (cs.mapped_area(cmap, w, w, pillar, nq=60) if cmap is not None
            else SIDE ** 2)
    res = {"env": cs.env_record(), "fixture": {
        "period": P, "side": SIDE, "fillet_over_side": ratio, "fillet_r": rf,
        "eps_pillar": EPS_P, "wl": WL, "depth": DEPTH, "n_sup": N_SUP,
        "n_sub": N_SUB}, "walls": w.tolist(),
        "pillar_area_mapped": area,
        "pillar_area_exact": SIDE ** 2 - (4 - np.pi) * rf ** 2,
        "detJ_min_max": (cs.detJ_range(cmap, w, w) if cmap is not None
                         else [1.0, 1.0]),
        "runs": []}
    for M in range(m_lo, m_hi + 1):
        t0 = time.perf_counter()
        out = cs.solve_curved(P, P, w, w, M, eps, N_SUP, N_SUB, DEPTH, WL,
                              cmap=cmap, n_orders=3)
        row = {"M": M, "dof": out["dof"], "t_total": time.perf_counter() - t0,
               "t_assemble": out["t_assemble"], "t_eig": out["t_eig"],
               "te": cs.table(out, "te"), "tm": cs.table(out, "tm"),
               "vec_te": cs.vec(out, "te").tolist(),
               "vec_tm": cs.vec(out, "tm").tolist(),
               "peak_rss_mb": cs.peak_rss_mb(), "diag": out["diag"]}
        for pol in ("te", "tm"):
            row[f"closure_{pol}"] = abs(row[pol]["sumR"] + row[pol]["sumT"] - 1)
        res["runs"].append(row)
        print(f"r/side={ratio} M={M} dof={out['dof']} t={row['t_total']:.1f}s "
              f"R00te={row['te']['0,0'][0]:.10f} T00te={row['te']['0,0'][1]:.10f} "
              f"clo={row['closure_te']:.1e}", flush=True)
        with open(os.path.join(cs.HERE, f"p4_fillet_r{ratio:g}.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

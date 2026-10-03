"""B2 / B8 / B10a -- a UNIFORM film under the curved maps is exact.

The film (eps 4, n = 2, depth 0.5) is passed as a constant PATTERNED cell
under the circle map (and the 5 x 5 circle, the ellipse, the fillet), so its
own mapped region eig, the half-spaces' mapped geometric eig, the corner
(Duffy) rule at the four singular vertices and the cofactor far field all
run, while the exact answer is the Airy slab (no material boundary: the only
thing the map can do wrong is the map).

  normal  : M = 4 .. 8 (c3), 4 .. 7 (c5, e3, f5), every order against Airy
  nocof   : the no-cofactor far projector (Phase A's engineered defect) at
            M = 6 -- the fail-before
  plain   : the Phase-A tensor rule in every cell (no corner rule) at
            M = 4 .. 8 on c3 -- does the film see the singular-vertex
            quadrature?  (B8)
  oblique : theta = 25 deg, phi = 0 and conical phi = 40 deg (B10a),
            M = 4 .. 8 on c3, 4 .. 7 on c5

  python validation/probe_pmm2d_curved/build_b/b3_film.py <part> [<part>]
Output: b3_film_<parts>.json
"""
import json
import sys
import time

import _common as C
import numpy as np

MAPS = {
    "c3": lambda: C.CM._circle_map_3x3(C.P, C.R_CIRC)[0],
    "c5": lambda: C.CM._circle_map_5x5(C.P, C.R_CIRC)[0],
    "e3": lambda: C.CM._ellipse_map_3x3(C.P, (0.36, 0.30))[0],
    "f5": lambda: C.CM._fillet_map_5x5(C.P, 0.3, 0.12)[0],
}


def film(cm, M, theta=0.0, phi=0.0):
    Nx, Ny = cm.shape
    eps = np.full((Nx, Ny), C.EPS_P, complex)
    o, R, T, _J = C.solve(cm, eps, M, theta=theta, phi=phi)
    return C.film_err(o, R, T, theta, phi)


def main(parts):
    out = "b3_film_" + "_".join(parts) + ".json"
    res = {"env": C.env_record()}
    res["airy_normal"] = list(C.airy_sp(0.0))
    rows = res.setdefault("rows", [])

    def rec(**row):
        rows.append(row)
        print(json.dumps(row), flush=True)
        C.dump(out, res)

    if "normal" in parts:
        for name, mlist in (("c3", range(4, 9)), ("c5", range(4, 8)),
                            ("e3", range(4, 8)), ("f5", range(4, 8))):
            cm = MAPS[name]()
            for M in mlist:
                t0 = time.perf_counter()
                rec(arm="normal", map=name, M=M, err=film(cm, M),
                    t=time.perf_counter() - t0)
    if "nocof" in parts:
        for name in ("c3", "c5"):
            cm = MAPS[name]()
            with C.no_cofactor():
                rec(arm="nocof", map=name, M=6, err=film(cm, 6))
    if "plain" in parts:
        cm = MAPS["c3"]()
        for M in range(4, 9):
            with C.plain_rule():
                rec(arm="plain", map="c3", M=M, err=film(cm, M))
    if "oblique" in parts:
        th = np.deg2rad(25.0)
        for phi_deg in (0.0, 40.0):
            for name, mlist in (("c3", range(4, 9)), ("c5", range(4, 8))):
                cm = MAPS[name]()
                for M in mlist:
                    rec(arm="oblique", map=name, M=M, theta_deg=25.0,
                        phi_deg=phi_deg,
                        err=film(cm, M, th, np.deg2rad(phi_deg)))


if __name__ == "__main__":
    main(sys.argv[1:])

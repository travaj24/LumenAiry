"""D2 -- a UNIFORM block-form tensor film under a map is the film: R, T AND
the complex Jones matrix against the exact Berreman 4x4 oracle
(``berreman_jones_1d``), at every rung M = 4 .. 8.

Tensors: the rotated in-plane uniaxial LC (n_o 1.5, n_e 1.8, director at
0.55 rad) and the gyrotropic tensor (e12 = -e21 = 0.5i).  Maps: a separable
sine stretch of both axes at a = 0.05 p and 0.15 p (33:1 local stretch), and
a SHEARED bilinear map (moved interior vertices, non-diagonal J).

Also the two engineered arms the gate is two-sided against (M = 7):
* transpose (eps -> eps^T inside the congruence) on the GYROTROPIC film:
  Jones error AND R / T error (the R / T must NOT see it -- dispersion is
  transpose-blind for this film);
* side (J^-T eps J^-1) on the LC film under the SHEAR map (a stretch has a
  diagonal J and cannot see it).
And the scalar-route identity: eps = s I through the tensor route against the
scalar route, under the shear and circle maps (operators and R / T / J).

usage: python d2_film.py ladder <map> <lc|gyro> <normal|oblique|conical> [Mmax]
       python d2_film.py arms
       python d2_film.py scalar
writes d2_film_<map>_<tensor>_<angle>.json / d2_arms.json / d2_scalar.json
"""
import sys
import time

import _dcommon as D
import numpy as np

ANG = {"normal": (0.0, 0.0), "oblique": (25.0, 0.0), "conical": (25.0, 40.0)}
TEN = {"lc": D.LC, "gyro": D.GYRO}


def ladder(mapname, tname, ang, Ms=(4, 5, 6, 7, 8)):
    P = D.G3["P"]
    cm = D.make_map(mapname, P)
    th, ph = ANG[ang]
    rows = []
    for M in Ms:
        t0 = time.perf_counter()
        dRT, dJ, _J = D.film_vs_berreman(TEN[tname], cm, M,
                                         theta=np.deg2rad(th),
                                         phi=np.deg2rad(ph))
        rows.append({"M": M, "dRT": dRT, "dJ": dJ,
                     "t": time.perf_counter() - t0})
        print(mapname, tname, ang, M, f"dRT={dRT:.2e} dJ={dJ:.2e}",
              flush=True)
    D.dump(f"d2_film_{mapname}_{tname}_{ang}.json",
           {"map": mapname, "tensor": tname, "theta": th, "phi": ph,
            "rows": rows})


def arms(M=7):
    P = D.G3["P"]
    out = {"M": M}
    for mapname in ("s05", "shear", "c3"):
        cm = D.make_map(mapname, P)
        dRT, dJ, J = D.film_vs_berreman(D.GYRO, cm, M)
        with D.mutate("transpose"):
            tRT, tJ, Jt = D.film_vs_berreman(D.GYRO, cm, M)
        out[f"gyro_{mapname}"] = {
            "correct": {"dRT": dRT, "dJ": dJ},
            "transpose": {"dRT": tRT, "dJ": tJ},
            "J01_over_J10": [complex(J[0, 1] / J[1, 0]).real,
                             complex(J[0, 1] / J[1, 0]).imag]}
        dRT, dJ, _ = D.film_vs_berreman(D.LC, cm, M)
        with D.mutate("side"):
            sRT, sJ, _ = D.film_vs_berreman(D.LC, cm, M)
        with D.mutate("transpose"):
            lRT, lJ, _ = D.film_vs_berreman(D.LC, cm, M)
        out[f"lc_{mapname}"] = {"correct": {"dRT": dRT, "dJ": dJ},
                                "side": {"dRT": sRT, "dJ": sJ},
                                "transpose": {"dRT": lRT, "dJ": lJ}}
        print(mapname, out[f"gyro_{mapname}"], out[f"lc_{mapname}"],
              flush=True)
    D.dump("d2_arms.json", out)


def scalar(M=5):
    """eps = s I via the tensor route vs the scalar route (same map)."""
    P = 1.2
    k0 = 2 * np.pi
    out = {}
    for mapname in ("shear", "c3"):
        cm = D.make_map(mapname, P)
        e = np.ones((3, 3), complex)
        e[1, 1] = 4.0
        e33 = np.zeros((3, 3, 3, 3), complex)
        for k in range(3):
            e33[..., k, k] = e
        s1 = D.TS.Granet2DTransverseE(P, P, cm.u_walls, cm.v_walls, M, e,
                                      k0=k0, cmap=cm)
        s2 = D.TS.Granet2DTransverseE(P, P, cm.u_walls, cm.v_walls, M, e33,
                                      k0=k0, cmap=cm)
        sc = max(np.abs(s1.Lmat).max(), np.abs(s1.Rmat).max())
        dop = max(np.abs(s1.Lmat - s2.Lmat).max(),
                  np.abs(s1.Rmat - s2.Rmat).max()) / sc
        res = []
        for cell in (e, e33):
            st = D.PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                                  n_modes=M, n_orders=3, cmap=cm)
            st.add_layer(0.5, eps_cell=cell)
            st.set_source(1.0)
            o, R, T, J = st.solve(jones=True)
            res.append((np.asarray(R), np.asarray(T), np.asarray(J)))
        drt = max(np.abs(res[0][0] - res[1][0]).max(),
                  np.abs(res[0][1] - res[1][1]).max(),
                  np.abs(res[0][2] - res[1][2]).max())
        out[mapname] = {"ops_rel": float(dop), "RTJ": float(drt)}
        print(mapname, out[mapname], flush=True)
    D.dump("d2_scalar.json", out)


if __name__ == "__main__":
    if sys.argv[1] == "ladder":
        Mx = int(sys.argv[5]) if len(sys.argv) > 5 else 8
        ladder(sys.argv[2], sys.argv[3], sys.argv[4],
               Ms=tuple(range(4, Mx + 1)))
    elif sys.argv[1] == "arms":
        arms()
    elif sys.argv[1] == "scalar":
        scalar()

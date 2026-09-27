"""P1 -- IDENTITY: the quadrature-assembled (variable-coefficient) masses with
the identity map reproduce the shipped piecewise-constant kron masses.

Run (BLAS pinned on the command line):
  cd /c/tmp/lum_curved && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved \
    python validation/probe_pmm2d_curved/p1_identity.py

Arms (same build, same process):
  shipped  = lumenairy Granet2DTransverseE (scalar, nonmagnetic) + the shipped
             pmm_efficiency_2d_staggered full solve + _far_projector_2d;
  scratch  = _curved_scratch.CurvedGranet with cmap=None (identity map through
             the MAGNETIC + TENSOR route: chi_t = I, chi33 = 1, eps'_t = eps I,
             every block by 2-D Gauss quadrature) + its own far projector.
Output: p1_identity.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _curved_scratch as cs  # noqa: E402
import numpy as np  # noqa: E402

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _far_projector_2d,
    pmm_efficiency_2d_staggered,
)

PX = PY = 1.2
WL = 1.0
K0 = 2 * np.pi / WL


def cmp(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    d = float(np.max(np.abs(a - b)))
    s = float(np.max(np.abs(b)))
    return {"max_abs": d, "max_rel": d / s if s else d,
            "bit_identical": bool(np.array_equal(a, b))}


def operators(shp, scr):
    E11, E12, E21, E22 = scr.Et
    return {
        "Et11": cmp(E11, shp.Et_blocks[0]),
        "Et22": cmp(E22, shp.Et_blocks[1]),
        "Et12_is_zero": float(np.max(np.abs(E12))),
        "Et21_is_zero": float(np.max(np.abs(E21))),
        "Rmat": cmp(scr.Rmat, shp.Rmat),
        "Gram_vs_minusR": cmp(np.block([[scr.Gram[0], 0 * scr.Gram[0]],
                                        [0 * scr.Gram[1], scr.Gram[1]]]),
                              -shp.Rmat),
        "Stt": cmp(scr.Stt, shp.Stt),
        "Schur": cmp(scr.Schur, shp.Schur),
        "Lmat": cmp(scr.Lmat, shp.Lmat),
    }


def main():
    res = {"env": cs.env_record(), "fixture": {
        "period": [PX, PY], "wl": WL, "eps_pillar": 4.0, "eps_host": 1.0,
        "grid": "3x3 centred pillar (integer N=3) and a NON-UNIFORM 3x3 wall "
                "array [0, 0.25, 0.85, 1.2] x [0, 0.3, 0.9, 1.2]"},
        "cases": []}
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    for M in (5, 7):
        for label, wx, wy in (("uniform_int", 3, 3),
                              ("nonuniform_walls",
                               np.array([0, 0.25, 0.85, 1.2]),
                               np.array([0, 0.3, 0.9, 1.2]))):
            t0 = time.perf_counter()
            shp = Granet2DTransverseE(PX, PY, wx, wy, M, eps, k0=K0)
            t_shp = time.perf_counter() - t0
            scr = cs.CurvedGranet(PX, PY, wx, wy, M, eps, None, k0=K0)
            case = {"M": M, "grid": label, "t_assemble_shipped": t_shp,
                    "t_assemble_scratch": scr.t_assemble,
                    "ops": operators(shp, scr)}
            # eigenvalues of the two pencils
            import scipy.linalg as sla
            g_a = np.sort_complex(sla.eigvals(shp.Lmat, -shp.Rmat))
            g_b = np.sort_complex(sla.eigvals(scr.Lmat, -scr.Rmat))
            # nearest-neighbour MATCH (a sort mis-pairs the spurious tail):
            # max over a of min_b |a - b| / max(1, |a|)
            dm = np.abs(g_a[:, None] - g_b[None, :]).min(axis=1)
            case["eig_matched_max_rel"] = float(np.max(dm / np.maximum(1.0, np.abs(g_a))))
            # far projector
            ox = np.arange(-3, 4)
            P1, P2 = _far_projector_2d(shp.bx, shp.by, ox, ox)
            Pf = cs.curved_far_projector(scr, ox, ox)
            qq = scr.qq
            nf = len(ox) ** 2
            case["far"] = {"P1_vs_xu": cmp(Pf[:nf, :qq], P1),
                           "P2_vs_yv": cmp(Pf[nf:, qq:], P2),
                           "offdiag_max": float(max(np.max(np.abs(Pf[:nf, qq:])),
                                                    np.max(np.abs(Pf[nf:, :qq]))))}
            res["cases"].append(case)
            print(M, label, json.dumps(case["ops"]["Lmat"]),
                  case["eig_matched_max_rel"], flush=True)
    # FULL SOLVE: shipped entry vs scratch identity (integer grid only -- the
    # shipped single-layer entry takes a uniform lattice)
    full = []
    for M in (5, 7):
        sc = cs.solve_curved(PX, PY, 3, 3, M, eps, 1.0, 1.45, 0.5, WL,
                             cmap=None, n_orders=3)
        row = {"M": M, "t_scratch": sc["t_total"]}
        for pol in ("te", "tm"):
            t0 = time.perf_counter()
            o, R, T = pmm_efficiency_2d_staggered(PX, PY, eps, 1.45, 1.0, 0.5,
                                                  WL, degree=M, n_orders=3,
                                                  polarization=pol)
            row[f"t_shipped_{pol}"] = time.perf_counter() - t0
            # same order ordering (tile / repeat) in both
            assert np.array_equal(o, sc["orders"])
            row[f"{pol}_max_abs_dR"] = float(np.max(np.abs(np.asarray(R) - sc["R"][pol])))
            row[f"{pol}_max_abs_dT"] = float(np.max(np.abs(np.asarray(T) - sc["T"][pol])))
            row[f"{pol}_closure_scratch"] = float(abs(sc["R"][pol].sum()
                                                      + sc["T"][pol].sum() - 1))
        row["geom_split_rel"] = sc["diag"]["geom_split_rel"]
        full.append(row)
        print("full", json.dumps(row), flush=True)
    res["full_solve"] = full
    with open(os.path.join(cs.HERE, "p1_identity.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

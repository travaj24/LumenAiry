"""P2 -- MECHANISM: the union grid, the sliver element, the operators.

Re-assembles exactly what ``PMMStack.solve``'s shared (union-grid) branch
assembles -- ``_pmm_union_grid`` -> ``_build_sem_tensor_segments`` ->
``_sem_modes_tensor`` -- and reports, per ``delta``:

  * the union grid's cell count and its NARROWEST cell (fraction, metres,
    element Jacobian ``J = w_phys/2``, and the dimensionless ``k0 J``);
  * conditioning of the assembled global operators (the diagonal GLL mass
    ``S0``, the unit stiffness, and the nodal ``Kx^2 = iS0 @ stiff / k0^2``);
  * the modal spectrum ``|q| = |gamma|/k0`` (max, and the count above 1e3);
  * the interface S-matrix solve's conditioning.
"""
import json
import os
import warnings

import numpy as np
from p1_repro import A0, B0, EPS_H, EPS_P, PX, THETA, WL, dz  # noqa: F401

import lumenairy
from lumenairy.elements.pmm._core import (
    _build_sem_tensor_segments,
    _interface_smatrix,
    _pmm_union_grid,
    _sem_modes_tensor,
    _sem_modes_uniform,
    _uniform_geo_eig,
)

print("lumenairy:", lumenairy.__file__, flush=True)
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))


def _t3(e):
    return dict(exx=complex(e), exy=0.0, eyx=0.0, eyy=complex(e),
                ezz=complex(e))


def diag(d, degree=14, min_feature=None):
    segs = [[(A0, EPS_H), (B0 - A0, EPS_P), (1.0 - B0, EPS_H)],
            [(A0 - d, EPS_H), (B0 + d - (A0 - d), EPS_P),
             (1.0 - (B0 + d), EPS_H)]]
    mf = (PX * 1e-5 if min_feature is None else min_feature) / PX
    uw, leps = _pmm_union_grid(segs, mf)
    k0 = 2.0 * np.pi / WL
    kx0 = np.sin(THETA) * k0
    mats = [_build_sem_tensor_segments(PX, uw, [_t3(e) for e in row],
                                       degree, 1, True) for row in leps]
    msup = _build_sem_tensor_segments(PX, uw, [_t3(1.0)] * len(uw), degree,
                                      1, True)
    w_min = float(np.min(uw))
    J = 0.5 * w_min * PX
    S0 = mats[0]["S0"]
    s0d = np.abs(np.diag(S0))
    out = dict(delta=d, degree=degree, n_cells=int(len(uw)),
               n_glob=int(mats[0]["n_glob"]),
               w_min_frac=w_min, w_min_m=w_min * PX, J=J, k0J=k0 * J,
               cond_S0=float(s0d.max() / s0d.min()),
               cond_stiff=float(np.linalg.cond(mats[0]["stiff"]["one"])))
    iS0 = np.diag(1.0 / np.diag(S0))
    Kx2 = (1.0 / (k0 * k0)) * (iS0 @ mats[0]["stiff"]["one"])
    out["norm_Kx2"] = float(np.abs(Kx2).max())
    out["cond_Kx2"] = float(np.linalg.cond(Kx2))
    geo = _uniform_geo_eig(msup, k0, kx0)
    Wsup, Vsup, _l, _g = _sem_modes_uniform(msup, k0, kx0, 1.0 + 0j, geo)
    ms = [_sem_modes_tensor(m, k0, kx0, True) for m in mats]
    for i, (W, V, lam, q) in enumerate(ms):
        qa = np.abs(np.asarray(q))
        out[f"q_max_L{i}"] = float(qa.max())
        out[f"q_gt1e3_L{i}"] = int((qa > 1e3).sum())
        out[f"cond_W_L{i}"] = float(np.linalg.cond(W))
    # interface conditioning: sup -> layer0, and layer0 -> layer1
    for lbl, (Wa, Va, Wb, Vb) in (
            ("sup_L0", (Wsup, Vsup, ms[0][0], ms[0][1])),
            ("L0_L1", (ms[0][0], ms[0][1], ms[1][0], ms[1][1]))):
        A = np.block([[Wa, -Wb], [Va, Vb]]) if Wa.shape == Wb.shape else None
        out[f"cond_ifc_{lbl}"] = (float(np.linalg.cond(A)) if A is not None
                                  else float("nan"))
        S = _interface_smatrix(Wa, Va, Wb, Vb)
        out[f"ifc_{lbl}_max"] = float(max(np.abs(np.asarray(b)).max()
                                          for b in S))
    return out


if __name__ == "__main__":
    rows = []
    hdr = (f"{'delta':>9} {'ncell':>5} {'w_min':>9} {'k0*J':>9} "
           f"{'cond S0':>9} {'|Kx2|':>9} {'q_max L0':>9} {'q_max L1':>9} "
           f"{'cond W1':>9} {'cnd ifc01':>10} {'|S ifc01|':>10}")
    print(hdr, flush=True)
    for d in (1e-2, 2.6e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0):
        r = diag(d)
        rows.append(r)
        print(f"{d:9.2e} {r['n_cells']:5d} {r['w_min_frac']:9.2e} "
              f"{r['k0J']:9.2e} {r['cond_S0']:9.2e} {r['norm_Kx2']:9.2e} "
              f"{r['q_max_L0']:9.2e} {r['q_max_L1']:9.2e} "
              f"{r['cond_W_L1']:9.2e} {r['cond_ifc_L0_L1']:10.2e} "
              f"{r['ifc_L0_L1_max']:10.2e}", flush=True)
        json.dump(rows, open(os.path.join(HERE, "p2_mech.json"), "w"), indent=1)
    print("\nwrote p2_mech.json", flush=True)

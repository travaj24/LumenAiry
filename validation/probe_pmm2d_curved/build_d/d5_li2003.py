"""D5 -- the Li 2003 (J. Opt. A 5:345, Example 1) GYROTROPIC crossed grating
under a map.  The shipped fixture of tests/unit/test_pmm2d_staggered_
anisotropic.py (periods 2.4 x 1.4 lambda, a half-filled 2 x 2 cell of the
gyrotropic tensors eps_b (pillar) / eps_a = conj(eps_b) (host), depth lambda,
substrate index 1 + 5i, incident E_x; Li tabulates six REFLECTED orders).

Arms:
* unmapped     -- the shipped solver (the 8.74e-05 reading at M = 8);
* identity     -- an IDENTITY TransfiniteMap on the same walls: the tensor
  route under a map, against the unmapped path (round-off; bit-for-bit is
  not available -- quadrature reassociates the sums, plan P1);
* stretch      -- a mild sine stretch of both axes (a = 0.05 p_x, -0.03 p_y)
  with the walls at the PREIMAGES of the physical walls (the same device):
  converges to its own limit, which must be the unmapped one;
* swap         -- the published SECOND row (eps_a <-> eps_b) under the
  stretch: the gyrotropic sign discriminator survives the map.

usage: python d5_li2003.py <unmapped|identity|stretch|swap> <M>
       python d5_li2003.py summary
"""
import json
import os
import sys
import time

import _common as C
import _dcommon as D
import numpy as np

LAM = 1.0e-6
PX, PY = 2.4 * LAM, 1.4 * LAM
LI_B = np.array([[2.25, -0.5j, 0.0], [0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                dtype=complex)
LI_A = np.conj(LI_B)
NSUB = 1.0 + 5.0j
TABLE1 = {(0, 0): 0.2980, (1, 0): 0.1195, (2, 0): 0.0222,
          (0, -1): 0.0619, (1, -1): 0.0269, (1, 1): 0.0137}
TABLE2 = {**TABLE1, (1, -1): 0.0137, (1, 1): 0.0269}
ORD = list(TABLE1)


def cell(pillar, host):
    c = np.empty((2, 2, 3, 3), dtype=complex)
    c[:] = host
    c[0, 0] = pillar
    return c


def cmap_for(arm):
    xw = np.array([0.0, 0.5 * PX, PX])
    yw = np.array([0.0, 0.5 * PY, PY])
    if arm == "identity":
        return D.CM.TransfiniteMap(xw, yw, None, None)
    if arm in ("stretch", "swap"):
        return D.CM.SeparableStretch.from_physical_walls(
            xw, yw, fx=D.CM.SineStretch(0.05 * PX),
            fy=D.CM.SineStretch(-0.03 * PY))
    return None


def run(arm, M):
    cm = cmap_for(arm)
    eps = cell(LI_A, LI_B) if arm == "swap" else cell(LI_B, LI_A)
    t0 = time.perf_counter()
    st = D.PMM2DStackPure(PX, PY, n_superstrate=1.0, n_substrate=NSUB,
                          n_modes=M, n_orders=4, cmap=cm)
    st.add_layer(LAM, eps_cell=eps)
    st.set_source(LAM)
    o, R, T, J = st.solve(jones=True)
    o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)
    i = C.idx(o, ORD)
    R0 = R[0, i]
    tab = TABLE2 if arm == "swap" else TABLE1
    res = {"arm": arm, "M": M, "orders": ORD, "R_Ex": R0.tolist(),
           "R_all": R.tolist(), "T_all": T.tolist(),
           "J": [[complex(z).real, complex(z).imag] for z in
                 np.asarray(J).ravel()],
           "maxdev_table": float(max(abs(R0[k] - v) for k, v in
                                     enumerate(tab.values()))),
           "maxdev_other_row": float(max(abs(R0[k] - v) for k, v in enumerate(
               (TABLE1 if arm == "swap" else TABLE2).values()))),
           "t": time.perf_counter() - t0}
    D.dump(f"d5_li_{arm}_M{M}.json", res)
    print(arm, M, f"table {res['maxdev_table']:.2e} other "
          f"{res['maxdev_other_row']:.2e} t={res['t']:.0f}", flush=True)


def summary():
    here = os.path.dirname(os.path.abspath(__file__))
    rs = {}
    for fn in os.listdir(here):
        if fn.startswith("d5_li_") and "summary" not in fn:
            r = json.load(open(os.path.join(here, fn)))
            rs[(r["arm"], r["M"])] = r
    out = {"rows": []}
    for (arm, M), r in sorted(rs.items()):
        row = {"arm": arm, "M": M, "maxdev_table": r["maxdev_table"],
               "maxdev_other_row": r["maxdev_other_row"]}
        u = rs.get(("unmapped", M))
        if u is not None and arm != "swap":
            row["vs_unmapped_RT"] = float(max(
                np.abs(np.array(r["R_all"]) - np.array(u["R_all"])).max(),
                np.abs(np.array(r["T_all"]) - np.array(u["T_all"])).max()))
            row["vs_unmapped_J"] = float(np.abs(
                np.array(r["J"]) - np.array(u["J"])).max())
        out["rows"].append(row)
    ums = sorted(M for a, M in rs if a == "unmapped")
    if ums:
        top = rs[("unmapped", ums[-1])]
        for row in out["rows"]:
            if row["arm"] in ("unmapped", "stretch", "identity"):
                r = rs[(row["arm"], row["M"])]
                row["vs_unmapped_top_RT"] = float(max(
                    np.abs(np.array(r["R_all"])
                           - np.array(top["R_all"])).max(),
                    np.abs(np.array(r["T_all"])
                           - np.array(top["T_all"])).max()))
        out["unmapped_top_M"] = ums[-1]
    D.dump("d5_li_summary.json", out)
    for row in out["rows"]:
        print(row)


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run(sys.argv[1], int(sys.argv[2]))

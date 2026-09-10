"""B7 (spurious census) + B8 (cascade vs depth) + B9 (layer split) +
B10 (the NO-FLOOR property) -- the three contracts the shear could break.

B7 -- the forward/backward split of a SLANTED region must stay EXACTLY
``2 q^2 / 2 q^2`` BEFORE the selector's defensive rebalance, ``min Re(lam_f)``
must not go negative on a lossless cell and must be strictly positive on a
lossy one, and ``max |q|`` must stay ``sec``-bounded (not the ~210 a
from-scratch convection form produces in 1-D).

B8 -- forward growth ``max exp(-Re(lam_f) k0 L)`` must be exactly 1 on every
lossless row, and closure must not GROW with depth.

B9 -- one slanted layer of depth ``d`` == two of ``d/2`` at the same slant.

B10 -- the answer must not move with ``n_orders`` (no Fourier floor).
"""
import time

import numpy as np
from _lib import tile, uniaxial, write

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    Granet2DTransverseE,
    _region_modes,
    _region_modes_oop,
)
from lumenairy.elements.rcwa._core import _select_forward_flux

PX = PY = 1.20
WL = 1.0
NSUP, NSUB = 1.0, 1.5

TIL = uniaxial(1.5, 1.7, np.deg2rad(35.0), np.deg2rad(25.0))
AIR = np.eye(3, dtype=complex)
SCA = np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex)
HIGH = np.array([[12.0, 1.0], [1.0, 1.0]], dtype=complex)
LOSSY = np.array([[4.0 + 0.6j, 1.0], [1.0, 1.0]], dtype=complex)
OOPC = tile(AIR, 2)
OOPC[0, 0] = TIL

T = {d: float(np.tan(np.deg2rad(d))) for d in (20, 36.9, 45, 60)}
res = {}
t00 = time.time()

# ------------------------------------------------------------------ B7
print("B7  SPURIOUS CENSUS (M=6, (2,2) grid, conical 20/35 -- dim 4q^2 = 400)")
M = 6
k0 = 2.0 * np.pi / WL
nre = float(np.real(np.sqrt(complex(NSUP) ** 2)))
th, ph = np.deg2rad(20.0), np.deg2rad(35.0)
a0x = nre * np.sin(th) * np.cos(ph) * k0
a0y = nre * np.sin(th) * np.sin(ph) * k0
cen = []
for cn, cell in (("scalar", SCA), ("oop", OOPC), ("high", HIGH),
                 ("lossy", LOSSY)):
    for sn, sv in (("vertical", None), ("x20", (T[20], 0.0)),
                   ("x45", (T[45], 0.0)),
                   ("diag45", (T[45] / np.sqrt(2), T[45] / np.sqrt(2))),
                   ("x60", (T[60], 0.0))):
        sol = Granet2DTransverseE(PX, PY, 2, 2, M, cell, alpha0x=a0x,
                                  alpha0y=a0y, k0=k0, slant=sv)
        if not sol.offplane:
            # a VERTICAL SCALAR / in-plane cell has no 4 q^2 pencil at all --
            # it runs the 2 q^2 second-order path, which is the bit-identity
            # statement, not a census row.  The vertical CONTROL for this
            # census is the out-of-plane tensor cell.
            continue
        Wf, Vf, lam_f, Wb, Vb, lam_b = _region_modes_oop(sol)
        # re-derive the RAW split (before the selector's rebalance) exactly the
        # way _region_modes_oop feeds it
        import scipy.linalg as sla
        Lc = np.linalg.cholesky(sol.Bgen)
        Ah = sla.solve_triangular(Lc, sol.Agen, lower=True)
        Ah = sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
        qv, Y = np.linalg.eig(Ah)
        X = sla.solve_triangular(Lc.conj().T, Y, lower=False)
        qq = sol.q * sol.q
        W = X[:2 * qq, :]
        Gst = X[2 * qq:, :]
        L1 = np.linalg.cholesky(sol.Bgen[:qq, :qq]).conj().T
        L2 = np.linalg.cholesky(sol.Bgen[qq:2 * qq, qq:2 * qq]).conj().T
        Vfull = np.concatenate([L1 @ W[:qq], L2 @ W[qq:],
                                L2 @ Gst[:qq], L1 @ Gst[qq:]], axis=0)
        nrm = np.linalg.norm(Vfull, axis=0)
        Vfull = Vfull / np.where(nrm == 0.0, 1.0, nrm)[None, :]
        fidx = np.asarray(_select_forward_flux(-1j * qv, Vfull, qq))
        emax = float(np.max(np.real(np.asarray(cell if cell.ndim == 2
                                               else cell[..., 0, 0]))))
        band = 3.0 * np.sqrt(max(emax, 1.0))
        row = {"cell": cn, "slant": sn, "dim": 4 * qq,
               "fwd": int(fidx.size), "bwd": int(4 * qq - fidx.size),
               "want": 2 * qq,
               "min_re_lamf": float(np.min(np.real(lam_f))),
               "max_abs_q": float(np.max(np.abs(qv))),
               "above_band": int(np.sum(np.abs(qv) > band))}
        cen.append(row)
        print("  %-7s %-9s  split %d/%d (want %d/%d)  minRe(lam_f) %+.1e  "
              "max|q| %6.2f  above-band %d"
              % (cn, sn, row["fwd"], row["bwd"], row["want"], row["want"],
                 row["min_re_lamf"], row["max_abs_q"], row["above_band"]))
res["B7"] = cen
res["B7_all_exact"] = all(r["fwd"] == r["want"] for r in cen)
res["B7_min_re_lossless"] = min(r["min_re_lamf"] for r in cen
                                if r["cell"] != "lossy")
res["B7_min_re_lossy"] = min(r["min_re_lamf"] for r in cen
                             if r["cell"] == "lossy")
res["B7_maxq_vertical"] = max(r["max_abs_q"] for r in cen
                              if r["slant"] == "vertical")
res["B7_maxq_60"] = max(r["max_abs_q"] for r in cen if r["slant"] == "x60")

# ------------------------------------------------------------------ B8/B9/B10
print("")
print("B8  CASCADE vs DEPTH (M=5)")


def solve(cell, slant, depth, theta, phi, n_orders=3, M=5, split=1):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=n_orders)
    for _ in range(split):
        st.add_layer(depth / split, eps_cell=cell, slant=slant)
    st.set_source(WL, theta=theta, phi=phi)
    return st.solve(jones=True)


def growth(cell, slant, depth, theta, phi, M=5):
    k0l = 2.0 * np.pi / WL
    nre = float(np.real(np.sqrt(complex(NSUP) ** 2)))
    ax = nre * np.sin(theta) * np.cos(phi) * k0l
    ay = nre * np.sin(theta) * np.sin(phi) * k0l
    sol = Granet2DTransverseE(PX, PY, 2, 2, M, cell, alpha0x=ax, alpha0y=ay,
                              k0=k0l, slant=slant)
    if sol.offplane:
        _Wf, _Vf, lam_f, _Wb, _Vb, _lb = _region_modes_oop(sol)
    else:                       # the VERTICAL control on the 2 q^2 path
        _W, _V, lam_f, _g2 = _region_modes(sol)
    return float(np.max(np.exp(-np.real(lam_f) * k0l * depth)))


import warnings  # noqa: E402

depth_rows = []
CFG = [("scalar", SCA, "vertical", None, "normal", 0.0, 0.0),
       ("scalar", SCA, "vertical", None, "conical", np.deg2rad(20.0),
        np.deg2rad(35.0)),
       ("scalar", SCA, "x36.9", (T[36.9], 0.0), "conical", np.deg2rad(20.0),
        np.deg2rad(35.0)),
       ("scalar", SCA, "diag45", (T[45] / np.sqrt(2), T[45] / np.sqrt(2)),
        "normal", 0.0, 0.0),
       ("oop", OOPC, "x36.9", (T[36.9], 0.0), "normal", 0.0, 0.0),
       ("oop", OOPC, "diag45", (T[45] / np.sqrt(2), T[45] / np.sqrt(2)),
        "conical", np.deg2rad(20.0), np.deg2rad(35.0)),
       ("lossy", LOSSY, "x36.9", (T[36.9], 0.0), "normal", 0.0, 0.0),
       ("lossy", LOSSY, "diag45", (T[45] / np.sqrt(2), T[45] / np.sqrt(2)),
        "conical", np.deg2rad(20.0), np.deg2rad(35.0))]
for cn, cell, sn, sv, mn, th, ph in CFG:
    row = {"cell": cn, "slant": sn, "mount": mn}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d in (0.25, 1.0, 3.0):
            o, R, Tt, J = solve(cell, sv, d, th, ph)
            row["clo_%g" % d] = float(np.max(np.abs(R.sum(1) + Tt.sum(1)
                                                    - 1.0)))
    row["growth_3lam"] = growth(cell, sv, 3.0, th, ph)
    depth_rows.append(row)
    print("  %-7s %-9s %-8s  0.25 %.2e  1 %.2e  3 %.2e   max fwd growth %.4e"
          % (cn, sn, mn, row["clo_0.25"], row["clo_1"], row["clo_3"],
             row["growth_3lam"]))
res["B8"] = depth_rows
res["B8_growth_lossless_max"] = max(r["growth_3lam"] for r in depth_rows
                                    if r["cell"] != "lossy")
res["B8_growth_lossy_max"] = max(r["growth_3lam"] for r in depth_rows
                                 if r["cell"] == "lossy")

print("")
print("B9  LAYER SPLIT (one layer of d == two of d/2, same slant)")
split_rows = []
for cn, cell, sn, sv, mn, th, ph in CFG:
    if sv is None:
        continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o1, R1, T1, J1 = solve(cell, sv, 0.8, th, ph, split=1)
        o2, R2, T2, J2 = solve(cell, sv, 0.8, th, ph, split=2)
    row = {"cell": cn, "slant": sn, "mount": mn,
           "dR": float(np.max(np.abs(R1 - R2))),
           "dT": float(np.max(np.abs(T1 - T2))),
           "dJ": float(np.max(np.abs(J1 - J2)))}
    split_rows.append(row)
    print("  %-7s %-9s %-8s  dR %.2e  dT %.2e  dJ %.2e"
          % (cn, sn, mn, row["dR"], row["dT"], row["dJ"]))
res["B9"] = split_rows
res["B9_worst"] = max(max(r["dR"], r["dT"], r["dJ"]) for r in split_rows)

print("")
print("B10  NO-FLOOR: movement of the answer with n_orders (3 -> 5 -> 8)")
nf_rows = []
for cn, cell, sn, sv, mn, th, ph in CFG:
    if sv is None or cn == "lossy":
        continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = None
        mv = {}
        for no in (3, 5, 8):
            o, R, Tt, J = solve(cell, sv, 0.8, th, ph, n_orders=no)
            idx = {(int(a), int(b)): i for i, (a, b) in enumerate(o)}
            keys = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
            vec = np.concatenate(
                [np.array([R[:, idx[k]] for k in keys]).ravel(),
                 np.array([Tt[:, idx[k]] for k in keys]).ravel(),
                 J.ravel().view(float)])
            if base is None:
                base = vec
            else:
                mv[no] = float(np.max(np.abs(vec - base)))
    nf_rows.append({"cell": cn, "slant": sn, "mount": mn,
                    "move_3_5": mv[5], "move_3_8": mv[8]})
    print("  %-7s %-9s %-8s  3->5 %.2e   3->8 %.2e"
          % (cn, sn, mn, mv[5], mv[8]))
res["B10"] = nf_rows
res["B10_worst_normal"] = max(r["move_3_8"] for r in nf_rows
                              if r["mount"] == "normal")
res["B10_worst"] = max(r["move_3_8"] for r in nf_rows)

res["wall_s"] = time.time() - t00
write("g4_census", res)

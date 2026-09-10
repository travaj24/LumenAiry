"""V2b -- ARBITRARY-WALL cells against INDEPENDENT engines, and the far-field
projection on non-uniform segments.

Sections (``python v2_oracles.py [oracle1d hybrid farfield]``):

* ``oracle1d`` a y-uniform stripe device at ARBITRARY x-walls, converged in
  ``M`` against the EXACT 1-D ``PMMStack`` (a different engine on a different
  basis) plus that oracle's own degree self-gap.
* ``hybrid``   the same class of device as a 2-D pillar at ARBITRARY walls,
  PER ORDER against ``PMM2DStackHybrid`` (exact walls through
  ``add_tapered_pillar(n_slices=1)``) and against ``pmm_efficiency_2d_cell``
  on a pixel-expressible wall set.
* ``farfield`` the Rayleigh projection on NON-UNIFORM segments: the
  y-momentum leak of a y-uniform device carried on a non-uniform y grid, the
  incident least-squares residual, and the projector's own order
  orthogonality against an exact Fourier oracle.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss

import lumenairy
from lumenairy.elements.pmm import twod_staggered as ts
from lumenairy.elements.pmm.stack import PMMStack
from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod import pmm_efficiency_2d_cell

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {_ROOT}")
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

RES = {}
WL = 1.0e-6
PX = 0.9e-6
PY = 0.9e-6
_C = complex


def _pure_nu(walls_x, walls_y, tile, M, n_orders, theta, phi, t=0.30e-6,
             n_sub=1.45):
    st = PMM2DStackPure(PX, PY, n_superstrate=1.0, n_substrate=n_sub,
                        n_modes=M, n_orders=n_orders,
                        layer_grids="per-layer")
    st.add_layer(t, eps_cell=tile,
                 x_walls=[w * PX for w in walls_x],
                 y_walls=[w * PY for w in walls_y])
    st.set_source(WL, theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve(jones=False)


def sec_oracle1d():
    """y-uniform stripes at ARBITRARY x-walls vs the EXACT 1-D PMMStack."""
    walls = [0.2371, 0.6183]
    eps = [6.0, 2.25, 3.1]
    segs = [(walls[0], eps[0]), (walls[1] - walls[0], eps[1]),
            (1.0 - walls[1], eps[2])]
    tile = np.zeros((3, 3), dtype=_C)
    for i, v in enumerate(eps):
        tile[i, :] = v
    theta, t = 0.21, 0.30e-6
    out = {"walls": walls, "eps": eps, "theta": theta}
    # --- the ORACLE and its OWN self-gap (degree 12 vs 14) ------------------
    ref = {}
    for deg in (10, 12, 14):
        st = PMMStack(PX, n_superstrate=1.0, n_substrate=1.45, degree=deg,
                      n_orders=7)
        st.add_layer(t, segments=segs)
        st.set_source(WL, theta=theta)
        o, R, T, _J = st.solve()
        ref[deg] = (o, R, T)
    def _gap(a, b):
        return float(max(np.max(np.abs(a[1] - b[1])),
                         np.max(np.abs(a[2] - b[2]))))
    out["oracle_selfgap_10_12"] = _gap(ref[10], ref[12])
    out["oracle_selfgap_12_14"] = _gap(ref[12], ref[14])
    o1, R1, T1 = ref[14]
    print(f"[oracle1d] exact 1-D PMMStack self-gap deg10-12 "
          f"{out['oracle_selfgap_10_12']:.3e}, deg12-14 "
          f"{out['oracle_selfgap_12_14']:.3e}", flush=True)
    # 1-D order m -> 2-D order (m, 0)
    ladder = {}
    for M in (4, 5, 6, 7, 8):
        t0 = time.time()
        o2, R2, T2 = _pure_nu(walls, walls, tile, M, 3, theta, 0.0, t=t)
        idx1 = {int(m): i for i, m in enumerate(np.asarray(o1).ravel())}
        err = 0.0
        for k, (mx, my) in enumerate(o2):
            if int(my) != 0 or int(mx) not in idx1:
                continue
            j = idx1[int(mx)]
            err = max(err, float(np.max(np.abs(R2[:, k] - R1[:, j]))),
                      float(np.max(np.abs(T2[:, k] - T1[:, j]))))
        clo = float(np.max(np.abs(R2.sum(axis=1) + T2.sum(axis=1) - 1.0)))
        ladder[M] = {"err_vs_exact_1d": err, "closure": clo,
                     "q": 3 * (M - 1), "wall_s": time.time() - t0}
        print(f"[oracle1d] M={M} q={3*(M-1):2d}  err vs EXACT 1-D = "
              f"{err:.4e}   closure {clo:.3e}   {time.time()-t0:.1f}s",
              flush=True)
    out["ladder"] = ladder
    RES["oracle1d"] = out


def sec_hybrid():
    """ARBITRARY-wall 2-D pillar, PER ORDER, against two hybrid engines."""
    out = {}
    theta, phi = 0.18, 0.35
    t = 0.30e-6
    # ---- arm A: genuinely arbitrary walls, hybrid via add_tapered_pillar ----
    xw = [0.2371, 0.6183]
    yw = [0.3117, 0.7402]
    e_p, e_h = 6.0, 2.25
    tile = np.full((3, 3), _C(e_h))
    tile[1, 1] = _C(e_p)
    hyb = {}
    for deg in (7, 9, 11):
        st = PMM2DStackHybrid(PX, PY, n_superstrate=1.0, n_substrate=1.45,
                              degree=deg, n_orders=5)
        st.add_tapered_pillar(t, eps_pillar=e_p, eps_host=e_h,
                              x_bounds_bottom=[xw[0] * PX, xw[1] * PX],
                              y_bounds_bottom=[yw[0] * PY, yw[1] * PY],
                              n_slices=1)
        st.set_source(WL, theta=theta, phi=phi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve()
        hyb[deg] = (np.asarray(o), R, T)

    def _gap(a, b):
        return float(max(np.max(np.abs(a[1] - b[1])),
                         np.max(np.abs(a[2] - b[2]))))
    out["hybrid_selfgap_7_9"] = _gap(hyb[7], hyb[9])
    out["hybrid_selfgap_9_11"] = _gap(hyb[9], hyb[11])
    oh, Rh, Th = hyb[11]
    print(f"[hybrid] oracle self-gap deg7-9 {out['hybrid_selfgap_7_9']:.3e},"
          f" deg9-11 {out['hybrid_selfgap_9_11']:.3e}", flush=True)
    idxh = {(int(a), int(b)): i for i, (a, b) in enumerate(oh)}
    lad = {}
    for M in (4, 5, 6, 7):
        t0 = time.time()
        o2, R2, T2 = _pure_nu(xw, yw, tile, M, 2, theta, phi, t=t)
        err = 0.0
        for k, (a, b) in enumerate(o2):
            j = idxh.get((int(a), int(b)))
            if j is None:
                continue
            err = max(err, float(np.max(np.abs(R2[:, k] - Rh[:, j]))),
                      float(np.max(np.abs(T2[:, k] - Th[:, j]))))
        clo = float(np.max(np.abs(R2.sum(axis=1) + T2.sum(axis=1) - 1.0)))
        lad[M] = {"err_vs_hybrid": err, "closure": clo,
                  "wall_s": time.time() - t0}
        print(f"[hybrid] arbitrary walls M={M}: per-order max |pure - hybrid|"
              f" = {err:.4e}  closure {clo:.3e}  {time.time()-t0:.1f}s",
              flush=True)
    out["arbitrary_walls_ladder"] = lad
    # ---- arm B: pixel-expressible walls, vs pmm_efficiency_2d_cell ---------
    # walls at 3/8 and 5/8 -> an 8 x 8 pixel grid names them EXACTLY.
    npix = 8
    cell = np.full((npix, npix), _C(e_h))
    cell[3:5, 3:5] = _C(e_p)
    hy2 = {}
    for deg in (7, 11):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eff = pmm_efficiency_2d_cell(
                PX, PY, cell, 1.45, 1.0, t, WL, degree=deg,
                polarization="te", theta=theta, phi=phi, n_orders=5)
        _o, _R, _T = eff              # Efficiency2D unpacks as (orders,R,T)
        hy2[deg] = (np.asarray(_o), np.asarray(_R), np.asarray(_T))
    out["cell_selfgap_7_11"] = float(max(
        np.max(np.abs(hy2[7][1] - hy2[11][1])),
        np.max(np.abs(hy2[7][2] - hy2[11][2]))))
    oc, Rc, Tc = hy2[11]
    idxc = {(int(a), int(b)): i for i, (a, b) in enumerate(oc)}
    tile2 = np.full((3, 3), _C(e_h))
    tile2[1, 1] = _C(e_p)
    lad2 = {}
    for M in (4, 5, 6, 7):
        o2, R2, T2 = _pure_nu([3 / 8, 5 / 8], [3 / 8, 5 / 8], tile2, M, 2,
                              theta, phi, t=t)
        err = 0.0
        for k, (a, b) in enumerate(o2):
            j = idxc.get((int(a), int(b)))
            if j is None:
                continue
            # pmm_efficiency_2d_cell is SINGLE-polarization (te row)
            err = max(err, float(abs(R2[0, k] - Rc[j])),
                      float(abs(T2[0, k] - Tc[j])))
        lad2[M] = err
        print(f"[hybrid] pixel walls (3/8,5/8) M={M}: per-order max "
              f"|pure - pmm_efficiency_2d_cell(te)| = {err:.4e}", flush=True)
    out["cell_ladder"] = lad2
    print(f"[hybrid] pmm_efficiency_2d_cell self-gap deg7-11 "
          f"{out['cell_selfgap_7_11']:.3e}", flush=True)
    RES["hybrid"] = out


def sec_farfield():
    """The Rayleigh projection on NON-UNIFORM segments."""
    out = {}
    # ---- y-MOMENTUM LEAK: a y-UNIFORM device carried on a NON-UNIFORM y grid
    walls_x = [0.2371, 0.6183]
    for walls_y, label in (([1 / 3, 2 / 3], "y_uniform_grid"),
                           ([0.1041, 0.8317], "y_nonuniform_grid"),
                           ([0.4903, 0.5102], "y_near_coincident_grid")):
        tile = np.zeros((3, 3), dtype=_C)
        for i, v in enumerate((6.0, 2.25, 3.1)):
            tile[i, :] = v                       # constant along y
        row = {}
        for M in (4, 5, 6):
            o, R, T = _pure_nu(walls_x, walls_y, tile, M, 3, 0.21, 0.0)
            ny = np.asarray(o)[:, 1]
            leak = float(np.max(R[:, ny != 0].sum(axis=1)
                                + T[:, ny != 0].sum(axis=1)))
            keep = float(np.min(R[:, ny == 0].sum(axis=1)
                                + T[:, ny == 0].sum(axis=1)))
            row[M] = {"leak_into_n_nonzero": leak, "kept_n0": keep}
            print(f"[farfield] {label:26s} M={M}: y-momentum leak = "
                  f"{leak:.3e}   (n=0 energy {keep:.6f})", flush=True)
        out[label] = row
    # ---- the projector's own ORDER ORTHOGONALITY against a Fourier oracle --
    # T[m, j] = (1/d) INT phi_j(x) e^{+i(mG + a0)x} dx.  Compare the shipped
    # quadrature (nq = 2M + 8 per segment) to a 6x-refined one.
    ortho = {}
    for label, xb in (("uniform_N3", np.array([0.0, 1 / 3, 2 / 3, 1.0])),
                      ("arb_N3", np.array([0.0, 0.2371, 0.6183, 1.0])),
                      ("skew_N3", np.array([0.0, 0.02, 0.93, 1.0]))):
        for M in (4, 6, 8):
            b = ts.Basis1D(1.0, xb, M, np.exp(-1j * 0.31))
            orders = np.arange(-6, 7)
            lib = ts._stag_fourier_projection(b, orders, 0.31)(b.Btilde)
            ora = _fourier_oracle(b, orders, 0.31, b.Btilde, mult=6)
            sc = max(float(np.max(np.abs(ora))), 1e-300)
            ortho[f"{label}_M{M}"] = float(np.max(np.abs(lib - ora)) / sc)
        print(f"[farfield] projector quadrature vs 6x-refined, {label}: "
              + ", ".join(f"M={M} {ortho[f'{label}_M{M}']:.2e}"
                          for M in (4, 6, 8)), flush=True)
    out["projector_quadrature_rel"] = ortho
    RES["farfield"] = out


def _fourier_oracle(basis, orders, alpha0, global_set, mult=6):
    d, N, M = basis.d, basis.N, basis.M
    G = 2.0 * np.pi / d
    xb = basis.xb
    xg, wg = leggauss(mult * (2 * M + 8))
    Vref, _ = ts._modleg_value_deriv(M, xg)
    orders = np.asarray(orders)
    T_local = np.zeros((len(orders), N, M), dtype=_C)
    for seg in range(N):
        J = 0.5 * (xb[seg + 1] - xb[seg])
        xphys = 0.5 * (xb[seg] + xb[seg + 1]) + J * xg
        phase = np.exp(1j * np.outer(orders * G + alpha0, xphys))
        T_local[:, seg, :] = (J / d) * (phase * wg) @ Vref.T
    S = np.array(global_set)
    return np.einsum("msa,jsa->mj", T_local, S)


SECTIONS = {"oracle1d": sec_oracle1d, "hybrid": sec_hybrid,
            "farfield": sec_farfield}


def main():
    which = sys.argv[1:] or list(SECTIONS)
    for w in which:
        t0 = time.time()
        SECTIONS[w]()
        print(f"--- {w} done in {time.time()-t0:.1f}s ---", flush=True)
    path = os.path.join(HERE, "v2_oracles.json")
    old = {}
    if os.path.exists(path):
        try:
            old = json.load(open(path))
        except Exception:          # a partial write from a crashed run
            old = {}
    old.update(RES)
    with open(path, "w") as f:
        json.dump(old, f, indent=1, sort_keys=True, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()

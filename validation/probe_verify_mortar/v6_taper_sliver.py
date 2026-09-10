"""V6 -- TAPERS on the pure engine, and THE SLIVER QUESTION on the mortar
route.

``python v6_taper_sliver.py [taper intra cross]``

* ``taper`` ``add_tapered_pillar`` on the pure per-layer stack against (a) the
  hybrid's own staircase at the IDENTICAL slices and (b) a FINE SHARED-GRID
  staircase on walls a common lattice can express; closure and cost.
* ``intra`` THE NEW SLIVER QUESTION: two walls ``delta`` of the period apart
  INSIDE ONE layer's own non-uniform grid (a taper slice whose pillar has
  nearly closed).  The 1-D mechanism is ``J = w P / 2`` -> stiffness ``1/w^2``
  -> spurious ``|q| ~ 0.65 N(N+1)/4 / (k0 J)``; the staggered basis has the
  same ``1/J_n`` stiffness, so the question is whether the same thing happens
  and whether anything REFUSES.
* ``cross`` the route ``FIX_PMMSTACK_SLIVER_WALLS`` calls SAFE: two ADJACENT
  per-layer layers whose walls differ by ``delta``, each on its OWN grid, so
  the sliver appears only in the cross-mass INTEGRATION mesh.  Re-measured
  here, with the interface conditioning and the closure.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
import warnings

import numpy as np

import lumenairy
from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    Granet2DTransverseE,
    StagCrossOps,
    StagGridOps,
    _region_modes,
)
from lumenairy.elements.rcwa import _core as _rc

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT)
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

_C = complex
RES = {}
P = 1.2
WL = 0.85
TH, PH = 0.15, 0.35
EPS_P, EPS_H = 9.0, 2.25


def solve(st, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = st.solve(**kw)
    return r, [str(x.message)[:90] for x in w]


# ==========================================================================
def sec_taper():
    out = {}
    xb0 = (0.1873 * P, 0.7241 * P)
    xb1 = (0.2917 * P, 0.6109 * P)
    kw = dict(eps_pillar=EPS_P, eps_host=EPS_H, x_bounds_bottom=xb0,
              y_bounds_bottom=xb0, x_bounds_top=xb1, y_bounds_top=xb1,
              n_slices=4)
    hyb = {}
    for deg in (7, 9, 11):
        t0 = time.time()
        h = PMM2DStackHybrid(P, n_orders=7, degree=deg)
        h.add_tapered_pillar(0.30, **kw)
        h.set_source(WL, theta=TH, phi=PH)
        (o, R, T, _J), _w = solve(h)
        o = np.asarray(o)
        p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        R2, T2 = np.atleast_2d(R), np.atleast_2d(T)
        hyb[deg] = {"R00": float(R2[0, p0]),
                    "closure": float(np.max(np.abs(R2.sum(1) + T2.sum(1)
                                                   - 1.0))),
                    "wall_s": time.time() - t0}
        print(f"[taper] hybrid degree {deg}: R(0,0)={hyb[deg]['R00']:.6f}  "
              f"closure {hyb[deg]['closure']:.3e}  "
              f"{hyb[deg]['wall_s']:.1f}s", flush=True)
    out["hybrid"] = hyb
    out["hybrid_selfgap_7_9"] = abs(hyb[7]["R00"] - hyb[9]["R00"])
    out["hybrid_selfgap_9_11"] = abs(hyb[9]["R00"] - hyb[11]["R00"])
    print(f"[taper] hybrid self-gap deg7-9 {out['hybrid_selfgap_7_9']:.3e}, "
          f"deg9-11 {out['hybrid_selfgap_9_11']:.3e}", flush=True)
    pure = {}
    for M in (4, 5, 6, 7):
        t0 = time.time()
        st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        st.add_tapered_pillar(0.30, **kw)
        st.set_source(WL, theta=TH, phi=PH)
        (o, R, T), _w = solve(st, jones=False)
        p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        pure[M] = {"R00": float(R[0, p0]),
                   "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))),
                   "wall_s": time.time() - t0, "q": 3 * (M - 1)}
        print(f"[taper] pure NU mortar M={M} (q={3*(M-1)}): "
              f"R(0,0)={pure[M]['R00']:.6f}  closure "
              f"{pure[M]['closure']:.3e}  {pure[M]['wall_s']:.1f}s",
              flush=True)
    out["pure"] = pure
    out["pure_vs_hybrid_deg11"] = {M: abs(pure[M]["R00"] - hyb[11]["R00"])
                                   for M in pure}
    print("[taper] |pure - hybrid(deg11)|: "
          + "  ".join(f"M{M} {v:.3e}"
                      for M, v in out['pure_vs_hybrid_deg11'].items()),
          flush=True)

    # ---- (b) a FINE SHARED-GRID staircase, on walls a lattice CAN express --
    # Slice walls snapped to an N = 40 lattice (the finest the shared path can
    # afford here); the SAME geometry, both engines.
    Nl = 40
    xs = []
    for _s in range(4):
        z = 1.0 - (_s + 0.5) / 4
        lo = xb0[0] + (xb1[0] - xb0[0]) * z
        hi = xb0[1] + (xb1[1] - xb0[1]) * z
        xs.append((round(lo / P * Nl) / Nl, round(hi / P * Nl) / Nl))
    print(f"[taper] lattice-snapped slice bounds on N={Nl}: {xs}", flush=True)
    shared = {}
    for M in (3, 4):
        t0 = time.time()
        st = PMM2DStackPure(P, n_modes=M, n_orders=1)
        for lo, hi in xs:
            cell = np.full((Nl, Nl), _C(EPS_H))
            a, b = int(round(lo * Nl)), int(round(hi * Nl))
            cell[a:b, a:b] = EPS_P
            st.add_layer(0.30 / 4, eps_cell=cell)
        st.set_source(WL, theta=TH, phi=PH)
        (o, R, T), _w = solve(st, jones=False)
        p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        shared[M] = {"R00": float(R[0, p0]), "q": Nl * (M - 1),
                     "closure": float(np.max(np.abs(R.sum(1) + T.sum(1)
                                                    - 1.0))),
                     "wall_s": time.time() - t0}
        print(f"[taper] SHARED N={Nl} M={M} (q={Nl*(M-1)}, eig dim "
              f"{2*(Nl*(M-1))**2}): R(0,0)={shared[M]['R00']:.6f}  closure "
              f"{shared[M]['closure']:.3e}  {shared[M]['wall_s']:.1f}s",
              flush=True)
    out["shared_lattice_staircase"] = shared
    # the SAME snapped geometry on the per-layer non-uniform route
    snap = {}
    for M in (5, 6):
        t0 = time.time()
        st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        for lo, hi in xs:
            tile = np.full((3, 3), _C(EPS_H))
            tile[1, 1] = EPS_P
            st.add_layer(0.30 / 4, eps_cell=tile,
                         x_walls=[lo * P, hi * P],
                         y_walls=[lo * P, hi * P], n_modes=M)
        st.set_source(WL, theta=TH, phi=PH)
        (o, R, T), _w = solve(st, jones=False)
        p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        snap[M] = {"R00": float(R[0, p0]),
                   "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))),
                   "wall_s": time.time() - t0}
        print(f"[taper] per-layer on the SNAPPED walls M={M}: "
              f"R(0,0)={snap[M]['R00']:.6f}  closure "
              f"{snap[M]['closure']:.3e}  {snap[M]['wall_s']:.1f}s",
              flush=True)
    out["perlayer_snapped"] = snap
    RES["taper"] = out


# ==========================================================================
def _spectrum(walls, M, eps_mid):
    """The region spectrum of ONE layer on its own non-uniform grid."""
    xb = np.array([0.0] + [w * P for w in walls] + [P])
    n = len(walls) + 1
    cell = np.full((n, n), _C(EPS_H))
    cell[n // 2, n // 2] = eps_mid
    sol = Granet2DTransverseE(P, P, xb, xb, M, cell, alpha0x=0.0, alpha0y=0.0,
                              k0=2 * np.pi / WL)
    W, V, lam, g2 = _region_modes(sol)
    Jmin = float(np.min(sol.bx.Jn))
    return {"lam_max": float(np.max(np.abs(lam))),
            "k0_Jmin": float(2 * np.pi / WL * Jmin),
            "Jmin": Jmin, "M": M, "N": n,
            "cond_Rmat": float(np.linalg.cond(-sol.Rmat)),
            "cond_Lmat": float(np.linalg.cond(sol.Lmat))}


def sec_intra():
    """Two walls ``delta`` apart INSIDE ONE layer's own grid."""
    out = {}
    # ---- kernel: the spurious-wavenumber predictor -------------------------
    spec = {}
    for delta in (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5):
        walls = [0.5 - delta / 2, 0.5 + delta / 2]
        for M in (4, 6, 8):
            s = _spectrum(walls, M, EPS_P)
            pred = 0.65 * (M * (M + 1) / 4) / s["k0_Jmin"]
            s["predictor_0.65_M(M+1)/4_over_k0J"] = pred
            s["ratio_measured_over_predicted"] = s["lam_max"] / pred
            spec[f"d{delta:g}_M{M}"] = s
        s4 = spec[f"d{delta:g}_M4"]
        print(f"[intra] delta={delta:.0e}: k0*Jmin={s4['k0_Jmin']:.3e}  "
              f"|lam|max M4/M6/M8 = "
              + " / ".join(f"{spec[f'd{delta:g}_M{M}']['lam_max']:.3e}"
                           for M in (4, 6, 8))
              + f"   (physical ceiling {np.sqrt(EPS_P):.2f})", flush=True)
    out["spectrum"] = spec
    # ---- device: does the SOLVE go wrong, and does anything say so? --------
    dev = {}
    for delta in (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 0.0):
        rec = {}
        for M in (4, 6):
            try:
                st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                                    layer_grids="per-layer")
                if delta > 0:
                    tile = np.full((3, 3), _C(EPS_H))
                    tile[1, 1] = EPS_P
                    st.add_layer(0.08, eps_cell=tile,
                                 x_walls=[(0.5 - delta / 2) * P,
                                          (0.5 + delta / 2) * P],
                                 y_walls=[(0.5 - delta / 2) * P,
                                          (0.5 + delta / 2) * P],
                                 n_modes=M)
                else:
                    st.add_layer(0.08, eps=EPS_H, grid=1, n_modes=M)
                st.set_source(WL, theta=TH, phi=PH)
                (o, R, T), warns = solve(st, jones=False)
                p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
                rec[f"M{M}"] = {
                    "R00": float(R[0, p0]),
                    "closure": float(np.max(np.abs(R.sum(1) + T.sum(1)
                                                   - 1.0))),
                    "warnings": warns}
            except Exception as exc:            # noqa: BLE001
                rec[f"M{M}"] = {"REFUSED": f"{type(exc).__name__}: {str(exc)[:110]}"}
        dev[f"{delta:g}"] = rec
        msg = f"[intra] DEVICE delta={delta:.0e}: "
        for M in (4, 6):
            r = rec[f"M{M}"]
            if "REFUSED" in r:
                msg += f"M{M} REFUSED  "
            else:
                msg += (f"M{M} R00={r['R00']:.6f} clo={r['closure']:.2e}"
                        f"{' WARN' if r['warnings'] else ''}  ")
        print(msg, flush=True)
    out["device"] = dev
    RES["intra"] = out


# ==========================================================================
def sec_cross():
    """Two ADJACENT layers whose walls differ by ``delta``, each on its OWN
    grid -- the route the sliver FIX doc calls safe."""
    out = {}
    w0 = (0.27865, 0.62505)
    rows = {}
    for delta in (1e-2, 2.6e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6,
                  0.0):
        rec = {}
        for M in (5, 6):
            tile = np.full((3, 3), _C(EPS_H))
            tile[1, 1] = EPS_P
            st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                                layer_grids="per-layer")
            st.add_layer(0.08, eps_cell=tile,
                         x_walls=[w0[0] * P, w0[1] * P],
                         y_walls=[w0[0] * P, w0[1] * P], n_modes=M)
            st.add_layer(0.08, eps_cell=tile,
                         x_walls=[(w0[0] - delta) * P, (w0[1] + delta) * P],
                         y_walls=[(w0[0] - delta) * P, (w0[1] + delta) * P],
                         n_modes=M)
            st.set_source(WL, theta=TH, phi=PH)
            prev, _rc._INV_CENSUS = _rc._INV_CENSUS, []
            try:
                (o, R, T), warns = solve(st, jones=False)
                worst_rc = 1.0
                for site, _n, rc, _res, _flag in _rc._INV_CENSUS:
                    if "mortar" in site and rc == rc:
                        worst_rc = min(worst_rc, float(rc))
            finally:
                _rc._INV_CENSUS = prev
            p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
            rec[f"M{M}"] = {"R00": float(R[0, p0]),
                      "closure": float(np.max(np.abs(R.sum(1) + T.sum(1)
                                                     - 1.0))),
                      "worst_mortar_rcond": worst_rc,
                      "warnings": warns}
        # cross-mass conditioning between the two grids at THIS delta
        xb0 = np.array([0.0, w0[0] * P, w0[1] * P, P])
        xb1 = np.array([0.0, (w0[0] - delta) * P, (w0[1] + delta) * P, P])
        ga = StagGridOps(P, P, xb0, xb0, 6, 1.0 + 0j, 1.0 + 0j)
        gb = StagGridOps(P, P, xb1, xb1, 6, 1.0 + 0j, 1.0 + 0j)
        cr = StagCrossOps(ga, gb)
        rec["cross_cond_Cbb_x"] = float(np.linalg.cond(cr.C2[1]))
        rec["cross_cond_Ctt_y"] = float(np.linalg.cond(cr.C1[0]))
        rows[f"{delta:g}"] = rec
        print(f"[cross] delta={delta:.1e}: M5 R00={rec['M5']['R00']:.9f} "
              f"clo={rec['M5']['closure']:.2e} "
              f"rcond={rec['M5']['worst_mortar_rcond']:.2e}"
              f" | M6 R00={rec['M6']['R00']:.9f} clo={rec['M6']['closure']:.2e}"
              f" | cond(C) {rec['cross_cond_Cbb_x']:.2e} /"
              f" {rec['cross_cond_Ctt_y']:.2e}", flush=True)
    ref5 = rows["0"]["M5"]["R00"]
    ref6 = rows["0"]["M6"]["R00"]
    for k, rec in rows.items():
        rec["dev_from_delta0_M5"] = abs(rec["M5"]["R00"] - ref5)
        rec["dev_from_delta0_M6"] = abs(rec["M6"]["R00"] - ref6)
    print("[cross] |R00(delta) - R00(0)|:  "
          + "  ".join(f"{k}:{rows[k]['dev_from_delta0_M6']:.2e}"
                      for k in rows), flush=True)
    out["rows"] = rows
    RES["cross"] = out


# ==========================================================================
def sec_intra2():
    """The HARD intra-layer case: a SLIVER inside ONE layer whose NEIGHBOURS
    sit on DIFFERENT grids, so the sliver's spurious modes meet a MORTAR
    rather than a same-grid modal match -- and the realistic API path, a
    ``add_tapered_pillar`` whose pillar closes to a point."""
    out = {}
    tile = np.full((3, 3), _C(EPS_H))
    tile[1, 1] = EPS_P
    # ---- (a) sliver layer sandwiched between two OTHER-grid layers ---------
    rows = {}
    for delta in (3e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        rec = {}
        for M in (4, 6):
            try:
                st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                                    layer_grids="per-layer")
                st.add_layer(0.06, eps_cell=tile,
                             x_walls=[0.21 * P, 0.68 * P],
                             y_walls=[0.21 * P, 0.68 * P], n_modes=M)
                st.add_layer(0.06, eps_cell=tile,
                             x_walls=[(0.5 - delta / 2) * P,
                                      (0.5 + delta / 2) * P],
                             y_walls=[(0.5 - delta / 2) * P,
                                      (0.5 + delta / 2) * P], n_modes=M)
                st.add_layer(0.06, eps_cell=tile,
                             x_walls=[0.33 * P, 0.79 * P],
                             y_walls=[0.33 * P, 0.79 * P], n_modes=M)
                st.set_source(WL, theta=TH, phi=PH)
                prev, _rc._INV_CENSUS = _rc._INV_CENSUS, []
                try:
                    (o, R, T), warns = solve(st, jones=False)
                    worst = 1.0
                    for site, _n, rc, _res, _flag in _rc._INV_CENSUS:
                        if rc == rc:
                            worst = min(worst, float(rc))
                finally:
                    _rc._INV_CENSUS = prev
                p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
                rec[f"M{M}"] = {
                    "R00": float(R[0, p0]),
                    "closure": float(np.max(np.abs(R.sum(1) + T.sum(1)
                                                   - 1.0))),
                    "worst_rcond": worst, "warnings": warns}
            except Exception as exc:                # noqa: BLE001
                rec[f"M{M}"] = {"REFUSED":
                                f"{type(exc).__name__}: {str(exc)[:110]}"}
        rows[f"{delta:g}"] = rec
        msg = f"[intra2] sandwiched sliver delta={delta:.0e}: "
        for M in (4, 6):
            r = rec[f"M{M}"]
            msg += ("M%d REFUSED  " % M if "REFUSED" in r else
                    f"M{M} R00={r['R00']:.6f} clo={r['closure']:.2e} "
                    f"rcond={r['worst_rcond']:.1e}"
                    f"{' WARN' if r['warnings'] else ''}  ")
        print(msg, flush=True)
    out["sandwiched"] = rows
    # ---- (b) the REALISTIC API path: a taper that CLOSES to a point --------
    tap = {}
    for wtop in (0.20, 0.02, 2e-3, 2e-4, 2e-5):
        rec = {}
        for M in (4, 6):
            try:
                st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                                    layer_grids="per-layer")
                st.add_tapered_pillar(
                    0.24, eps_pillar=EPS_P, eps_host=EPS_H,
                    x_bounds_bottom=[0.25 * P, 0.75 * P],
                    y_bounds_bottom=[0.25 * P, 0.75 * P],
                    x_bounds_top=[(0.5 - wtop / 2) * P,
                                  (0.5 + wtop / 2) * P],
                    y_bounds_top=[(0.5 - wtop / 2) * P,
                                  (0.5 + wtop / 2) * P],
                    n_slices=4)
                st.set_source(WL, theta=TH, phi=PH)
                (o, R, T), warns = solve(st, jones=False)
                p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
                narrow = min(np.min(np.diff(np.asarray(L["wx"])))
                             for L in st._layers) / P
                rec[f"M{M}"] = {
                    "R00": float(R[0, p0]),
                    "narrowest_segment_frac": float(narrow),
                    "closure": float(np.max(np.abs(R.sum(1) + T.sum(1)
                                                   - 1.0))),
                    "warnings": warns}
            except Exception as exc:                # noqa: BLE001
                rec[f"M{M}"] = {"REFUSED":
                                f"{type(exc).__name__}: {str(exc)[:110]}"}
        tap[f"{wtop:g}"] = rec
        msg = f"[intra2] closing taper w_top={wtop:.0e} of the period: "
        for M in (4, 6):
            r = rec[f"M{M}"]
            msg += ("M%d REFUSED  " % M if "REFUSED" in r else
                    f"M{M} R00={r['R00']:.6f} clo={r['closure']:.2e} "
                    f"(narrowest seg {r['narrowest_segment_frac']:.1e})"
                    f"{' WARN' if r['warnings'] else ''}  ")
        print(msg, flush=True)
    out["closing_taper"] = tap
    RES["intra2"] = out


SECTIONS = {"taper": sec_taper, "intra": sec_intra,
            "intra2": sec_intra2, "cross": sec_cross}


def main():
    for w in (sys.argv[1:] or list(SECTIONS)):
        t0 = time.time()
        SECTIONS[w]()
        print(f"--- {w} done in {time.time()-t0:.1f}s ---", flush=True)
    path = os.path.join(HERE, "v6_taper_sliver.json")
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

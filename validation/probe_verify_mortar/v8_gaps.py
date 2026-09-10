"""V8 -- the remaining build-doc gates, re-measured (all cheap).

``python v8_gaps.py [g1 berreman gyro n2 n3 census]``

* ``g1``       a VERTICAL conforming per-layer stack is BIT-EXACT vs
  ``layer_grids='shared'`` (sha256).
* ``berreman`` the GENERALIZED mortar twin across a NON-conforming split,
  against ``berreman_jones_1d`` at conical incidence -- the arm that exercises
  ``_interface_smatrix_general_mortar_2d`` on an out-of-plane layer.
* ``gyro``     two-sided lossless closure with a HERMITIAN gyrotropic tensor
  (absorbs nothing, so ``R + T = 1`` is exact) on non-conforming grids.
* ``n2``       the ULP case: an explicit ``linspace`` wall array vs the
  integer path.
* ``n3``       two EXACT-wall representations of ONE device against the
  triangle-inequality bar.
* ``census``   the conditioning census over the mortar's guarded site.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import sys
import time
import warnings

import numpy as np

import lumenairy
from lumenairy.elements.berreman import berreman_jones_1d
from lumenairy.elements.pmm.stack import PMMStack
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import Basis1D
from lumenairy.elements.rcwa import _core as _rc

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT)
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

_C = complex
RES = {}
P = 1.05
WL = 0.72
EPS_P, EPS_H = 5.0, 2.10


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def solve(st, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve(**kw)


def pillar(N, lo, hi, e_p=EPS_P, e_h=EPS_H):
    c = np.full((N, N), _C(e_h))
    c[lo:hi, lo:hi] = e_p
    return c


def stripe(N, lo, hi, e_p=EPS_P, e_h=EPS_H):
    c = np.full((N, N), _C(e_h))
    c[lo:hi, :] = e_p
    return c


def sec_g1():
    out = {}
    for name, build in (
        ("pillar|uniform|stripe N2 M5", lambda lg: _b1(lg, 5, 2)),
        ("pillar|uniform|stripe N3 M6", lambda lg: _b1(lg, 6, 3)),
    ):
        a = solve(build("shared"))
        b = solve(build("per-layer"))
        eq = all(sha(x) == sha(y) for x, y in zip(a[1:], b[1:]))
        mv = float(max(np.max(np.abs(a[i] - b[i])) for i in (1, 2, 3)))
        out[name] = {"sha_equal": eq, "max_move": mv}
        print(f"[g1] {name}: sha_equal={eq}  max move {mv:.2e}", flush=True)
    RES["g1"] = out


def _b1(lg, M, N):
    st = PMM2DStackPure(P, n_modes=M, n_orders=2, layer_grids=lg)
    extra = {"grid": N, "n_modes": M} if lg == "per-layer" else {}
    st.add_layer(0.30, eps_cell=pillar(N, 0, 1))
    st.add_layer(0.15, eps=2.0, **extra)
    st.add_layer(0.25, eps_cell=stripe(N, N - 1, N))
    st.set_source(WL, theta=0.20, phi=0.30)
    return st


def sec_berreman():
    """The GENERALIZED mortar across a split, vs berreman_jones_1d."""
    out = {}

    def uni(no, ne, tilt, azim):
        c, s = np.cos(tilt), np.sin(tilt)
        d = np.array([s * np.cos(azim), s * np.sin(azim), c])
        return no ** 2 * np.eye(3) + (ne ** 2 - no ** 2) * np.outer(d, d)

    eps33 = uni(1.48, 1.72, np.deg2rad(41.0), np.deg2rad(17.0))
    for th_deg, ph_deg in ((0.0, 0.0), (25.0, 40.0), (35.0, 70.0)):
        th, ph = np.deg2rad(th_deg), np.deg2rad(ph_deg)
        Rb, Tb, Jb, _Jt = berreman_jones_1d([(eps33, 0.33)], 1.5, 1.0, WL,
                                            angle=th, phi=ph)
        for M in (5, 7):
            for ga, gb in ((2, 2), (2, 3), (3, 4)):
                st = PMM2DStackPure(P, n_superstrate=1.0, n_substrate=1.5,
                                    n_modes=M, n_orders=1,
                                    layer_grids="per-layer")
                st.add_layer(0.165, eps=eps33, grid=ga, n_modes=M)
                st.add_layer(0.165, eps=eps33, grid=gb, n_modes=M)
                st.set_source(WL, theta=th, phi=ph)
                o, R, T, J = solve(st)
                dJ = float(np.max(np.abs(J - np.asarray(Jb))))
                dR = float(np.max(np.abs(R.sum(1) - np.asarray(Rb))))
                clo = float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))
                out[f"th{th_deg}_ph{ph_deg}_M{M}_g{ga}{gb}"] = {
                    "dJones": dJ, "dR": dR, "closure": clo}
            print(f"[berreman] theta={th_deg} phi={ph_deg} M={M}: dJones "
                  + " / ".join(
                      f"({a},{b}) "
                      f"{out[f'th{th_deg}_ph{ph_deg}_M{M}_g{a}{b}']['dJones']:.2e}"
                      for a, b in ((2, 2), (2, 3), (3, 4))), flush=True)
    RES["berreman"] = out


def sec_gyro():
    """Two-sided lossless closure with a HERMITIAN gyrotropic tensor."""
    out = {}
    t = np.zeros((2, 2, 3, 3), dtype=_C)
    t[:] = np.diag([EPS_H, EPS_H, EPS_H])
    t[0, 0] = np.array([[6.0, 0.9j, 0.0], [-0.9j, 6.0, 0.0], [0.0, 0.0, 5.0]])
    cB = pillar(3, 1, 2)
    for kind, cA in (("scalar", pillar(2, 0, 1)), ("gyrotropic", t)):
        clos = []
        for M in (4, 5, 6):
            st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                                layer_grids="per-layer")
            st.add_layer(0.30, eps_cell=cA, n_modes=M)
            st.add_layer(0.25, eps_cell=cB, n_modes=M)
            st.set_source(WL, theta=0.18, phi=0.35)
            o, R, T = solve(st, jones=False)
            clos.append(float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))))
        out[kind] = {"closures_M4_M5_M6": clos,
                     "ratios": [clos[0] / clos[1], clos[1] / clos[2]]}
        print(f"[gyro] {kind:11s} closure {clos[0]:.3e} -> {clos[1]:.3e} -> "
              f"{clos[2]:.3e}  ({clos[0]/clos[1]:.1f}x then "
              f"{clos[1]/clos[2]:.1f}x)", flush=True)
    RES["gyro"] = out


def sec_n2():
    out = {}
    worst = 0.0
    for d, N, M in ((1.2, 2, 5), (1.2, 4, 5), (0.9, 4, 5), (0.9, 3, 6),
                    (0.7, 5, 7), (1.55, 3, 8)):
        tau = np.exp(-1j * 0.31 * d)
        bi = Basis1D(d, N, M, tau)
        ba = Basis1D(d, np.linspace(0.0, d, N + 1), M, tau)
        Mi = bi.mass(bi.Btilde, bi.Btilde)
        Ma = ba.mass(ba.Btilde, ba.Btilde)
        rel = float(np.max(np.abs(Mi - Ma))) / float(np.max(np.abs(Mi)))
        same = sha(Mi) == sha(Ma)
        out[f"d{d}_N{N}_M{M}"] = {"rel": rel, "bit_identical": same,
                                  "linspace_step_eq": bool(
                                      np.all(np.diff(ba.xb) == d / N))}
        worst = max(worst, rel)
        print(f"[n2] d={d} N={N} M={M}: rel {rel:.3e}  bit-identical={same}",
              flush=True)
    out["worst"] = worst
    RES["n2"] = out


def sec_n3():
    """Two EXACT-wall representations of ONE duty-1/3 stripe."""
    out = {}
    per, wl, th = 0.9, 0.6, 0.20

    def orc(deg):
        st = PMMStack(per, degree=deg, far_field_orders=5)
        st.add_layer(0.30, segments=[(1 / 3, EPS_P), (2 / 3, EPS_H)])
        st.set_source(wl, theta=th)
        o, R, T = st.solve()[:3]
        return np.asarray(o).ravel(), np.atleast_2d(R), np.atleast_2d(T)

    o1, R1, T1 = orc(14)
    o0, R0, T0 = orc(12)
    keep = np.abs(o1) <= 1
    self_gap = float(max(np.max(np.abs(R0[:, keep] - R1[:, keep])),
                         np.max(np.abs(T0[:, keep] - T1[:, keep]))))
    out["oracle_selfgap_12_14"] = self_gap
    print(f"[n3] oracle self-gap deg12-14 = {self_gap:.4e}", flush=True)

    def score(o2, R, T):
        best = 0.0
        for m in (-1, 0, 1):
            k = int(np.where((o2[:, 0] == m) & (o2[:, 1] == 0))[0][0])
            j = int(np.where(o1 == m)[0][0])
            best = max(best, abs(float(R[1, k]) - float(R1[1, j])),
                       abs(float(T[1, k]) - float(T1[1, j])))
        return best
    cnu = np.array([[EPS_P, EPS_P], [EPS_H, EPS_H]], dtype=_C)
    cu = np.full((3, 3), EPS_H + 0j)
    cu[0, :] = EPS_P
    rows = {}
    for M in (4, 5, 6, 7, 9):
        stn = PMM2DStackPure(per, n_modes=M, n_orders=1,
                             layer_grids="per-layer")
        stn.add_layer(0.30, eps_cell=cnu, x_walls=[per / 3.0],
                      y_walls=[per / 3.0])
        stn.set_source(wl, theta=th)
        on, Rn, Tn = solve(stn, jones=False)
        stu = PMM2DStackPure(per, n_modes=M, n_orders=1)
        stu.add_layer(0.30, eps_cell=cu)
        stu.set_source(wl, theta=th)
        ou, Ru, Tu = solve(stu, jones=False)
        en, eu = score(on, Rn, Tn), score(ou, Ru, Tu)
        gap = 0.0
        for m in (-1, 0, 1):
            i1 = int(np.where((on[:, 0] == m) & (on[:, 1] == 0))[0][0])
            i2 = int(np.where((ou[:, 0] == m) & (ou[:, 1] == 0))[0][0])
            gap = max(gap, abs(float(Rn[1, i1]) - float(Ru[1, i2])),
                      abs(float(Tn[1, i1]) - float(Tu[1, i2])))
        bar = en + eu + 4.0 * self_gap
        rows[M] = {"q_NU": 2 * (M - 1), "q_uniform": 3 * (M - 1),
                   "err_NU": en, "err_uniform": eu, "gap": gap,
                   "bar": bar, "inside": bool(gap <= bar)}
        print(f"[n3] M={M} q {2*(M-1)}/{3*(M-1)}: err NU {en:.4e} uniform "
              f"{eu:.4e}  gap {gap:.4e}  bar {bar:.4e}  inside="
              f"{gap <= bar}", flush=True)
    out["rows"] = rows
    RES["n3"] = out


def sec_census():
    out = {}
    prev, _rc._INV_CENSUS = _rc._INV_CENSUS, []
    try:
        rows = {}
        for label, (cA, cB) in (
                ("pillar (2,3)", (pillar(2, 0, 1), pillar(3, 1, 2))),
                ("pillar (2,6)", (pillar(2, 0, 1), pillar(6, 2, 4))),
                ("stripe (2,3)", (stripe(2, 0, 1), stripe(3, 0, 1)))):
            for M in (4, 5, 6):
                _rc._INV_CENSUS.clear()
                st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                                    layer_grids="per-layer")
                st.add_layer(0.30, eps_cell=cA, n_modes=M)
                st.add_layer(0.25, eps_cell=cB, n_modes=M)
                st.set_source(WL, theta=0.18, phi=0.35)
                solve(st, jones=False)
                worst, sites = 1.0, set()
                for site, _n, rc, _res, _flag in _rc._INV_CENSUS:
                    sites.add(site)
                    if rc == rc:
                        worst = min(worst, float(rc))
                rows[f"{label}_M{M}"] = {"worst_rcond": worst,
                                         "sites": sorted(sites)}
                print(f"[census] {label} M={M}: worst equilibrated rcond "
                      f"{worst:.3e}  ({len(sites)} guarded sites)",
                      flush=True)
        out["rows"] = rows
    finally:
        _rc._INV_CENSUS = prev
    RES["census"] = out


SECTIONS = {"g1": sec_g1, "berreman": sec_berreman, "gyro": sec_gyro,
            "n2": sec_n2, "n3": sec_n3, "census": sec_census}


def main():
    for w in (sys.argv[1:] or list(SECTIONS)):
        t0 = time.time()
        SECTIONS[w]()
        print(f"--- {w} done in {time.time()-t0:.1f}s ---", flush=True)
    path = os.path.join(HERE, "v8_gaps.json")
    old = {}
    if os.path.exists(path):
        try:
            old = json.load(open(path))
        except Exception:
            old = {}
    old.update(RES)
    with open(path, "w") as f:
        json.dump(old, f, indent=1, sort_keys=True, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()

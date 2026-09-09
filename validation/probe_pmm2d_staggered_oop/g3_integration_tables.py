"""GATE 3 -- the measurement tables the integrated out-of-plane path's test
bars are derived from.

Six blocks, all two-arm, all on this build:

  T5  y-uniform out-of-plane STRIPE per order against ``pmm_jones_1d`` and
      ``rcwa_jones_1d_segments`` (both polarizations) + y-momentum leakage;
  T6  the (3,3) L cell with a RE-ENTRANT corner, PER ORDER against
      ``rcwa_jones_2d`` and ``pmm_jones_2d`` -- GATE 0's gate, now on the
      library gauge -- plus the NO-FLOOR property two-sided;
  T7  cascade closure at 0.25 / 1 / 3 wavelengths, Hermitian and lossy, with
      the forward growth factor;
  T8  stacks: an out-of-plane layer over an in-plane one and over a uniform
      one, and an all-uniform out-of-plane multilayer vs ``berreman_jones_1d``;
  T9  fail-before controls: the rotation sign and an ``e13 <-> e31`` transpose
      on a NON-RECIPROCAL cell, off-azimuth;
  T10 cost: dimension, wall time and peak RSS, out-of-plane vs in-plane.

Run:
  cd /c/tmp/lum_aniso_oopint && PYTHONPATH=/c/tmp/lum_aniso_oopint \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/g3_integration_tables.py
"""
import json
import os
import sys
import time
import tracemalloc
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import probe_common as pc  # noqa: E402

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm import (  # noqa: E402
    PMM2DStackPure,
    pmm_jones_1d,
    pmm_jones_2d,
)
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402
from lumenairy.elements.rcwa.oned import (  # noqa: E402
    rcwa_jones_1d_segments,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
RES = {}

WL = 1.0
NSUB, NSUP = 1.5, 1.0
_OOP = pc.uniaxial(1.5, 1.7, 35.0, azim_deg=25.0)
_OOP_LOSSY = pc.uniaxial(1.5, 1.7, 35.0, azim_deg=25.0, loss=0.08)
_NONREC = np.array(_OOP, dtype=complex)
_NONREC[0, 2] = _OOP[0, 2] + 0.22j
_NONREC[2, 0] = np.conj(_NONREC[0, 2])
_AIR = np.eye(3, dtype=complex)


def stag(px, py, ec, dep, M, th=0.0, ph=0.0, no=3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TS.pmm_jones_2d_staggered(px, py, ec, NSUB, NSUP, dep, WL,
                                         degree=M, n_orders=no,
                                         theta=th, phi=ph)


def per_order(o_a, R_a, T_a, o_b, R_b, T_b, keep=None):
    idx = {tuple(int(v) for v in r): j for j, r in enumerate(np.asarray(o_b))}
    dR = dT = 0.0
    for i, r in enumerate(np.asarray(o_a)):
        k = tuple(int(v) for v in r)
        if k not in idx or (keep is not None and k not in keep):
            continue
        dR = max(dR, float(np.max(np.abs(np.asarray(R_a)[:, i]
                                         - np.asarray(R_b)[:, idx[k]]))))
        dT = max(dT, float(np.max(np.abs(np.asarray(T_a)[:, i]
                                         - np.asarray(T_b)[:, idx[k]]))))
    return dR, dT


def upsample(ec, n):
    return np.repeat(np.repeat(np.asarray(ec), n, axis=0), n, axis=1)


# ========================== T5: 1-D stripe per order ======================= #
def t5():
    print("\n### T5  y-uniform out-of-plane stripe vs the 1-D engines "
          "(per order, both pols)")
    px = py = 1.2
    dep = 0.4
    ridge, groove = _OOP, _AIR
    ec = np.zeros((2, 2, 3, 3), dtype=complex)
    ec[0, :] = ridge
    ec[1, :] = groove
    for mount, th in (("normal", 0.0), ("oblique 25", np.deg2rad(25.0))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o_r, R_r, T_r, J_r = rcwa_jones_1d_segments(
                px, [(0.5, ridge), (0.5, groove)], NSUB, NSUP, dep, WL,
                n_orders=41, theta=th)
            o_p, R_p, T_p, J_p = pmm_jones_1d(
                px, ridge, groove, NSUB, NSUP, dep, 0.5, WL, theta=th,
                degree=18, far_field_orders=31, stabilize=False)
        o_r = np.asarray(o_r)
        o_p = np.asarray(o_p)
        ir = {int(m): j for j, m in enumerate(o_r)}
        ip = {int(m): j for j, m in enumerate(o_p)}
        common = sorted(set(ir) & set(ip) & set(range(-3, 4)))
        spread = max(float(np.max(np.abs(R_r[:, ir[m]] - R_p[:, ip[m]])))
                     for m in common)
        spreadT = max(float(np.max(np.abs(T_r[:, ir[m]] - T_p[:, ip[m]])))
                      for m in common)
        spreadJ = float(np.max(np.abs(np.asarray(J_r) - np.asarray(J_p))))
        print(f"  {mount}: the two 1-D engines' own spread  dR = {spread:.3e}"
              f"  dT = {spreadT:.3e}  dJones = {spreadJ:.3e}   <- the bar")
        RES[f"T5|{mount}|oracle_spread"] = dict(dR=spread, dT=spreadT,
                                                dJ=spreadJ)
        for M in (5, 6, 7, 8):
            o_s, R_s, T_s, J_s = stag(px, py, ec, dep, M, th=th, ph=0.0, no=3)
            o_s = np.asarray(o_s)
            sel = o_s[:, 1] == 0
            dR = max(float(np.max(np.abs(R_s[:, np.where(
                (o_s[:, 0] == m) & (o_s[:, 1] == 0))[0][0]] - R_r[:, ir[m]])))
                for m in common)
            dT = max(float(np.max(np.abs(T_s[:, np.where(
                (o_s[:, 0] == m) & (o_s[:, 1] == 0))[0][0]] - T_r[:, ir[m]])))
                for m in common)
            dJ = float(np.max(np.abs(np.asarray(J_s) - np.asarray(J_r))))
            yleakR = float(np.max(np.abs(R_s[:, ~sel])))
            yleakT = float(np.max(np.abs(T_s[:, ~sel])))
            clo = float(np.max(np.abs(R_s.sum(axis=1) + T_s.sum(axis=1) - 1)))
            print(f"    M={M} dim={4*(2*(M-1))**2:5d}  vs rcwa1d: dR = "
                  f"{dR:.3e}  dT = {dT:.3e}  dJones = {dJ:.3e}   y-leak "
                  f"R/T = {yleakR:.1e}/{yleakT:.1e}   |R+T-1| = {clo:.2e}")
            RES[f"T5|{mount}|M{M}"] = dict(dR=dR, dT=dT, dJ=dJ,
                                           yleakR=yleakR, yleakT=yleakT,
                                           closure=clo)


# ==================== T6: the (3,3) re-entrant-corner cell ================= #
def t6():
    print("\n### T6  (3,3) L cell with a RE-ENTRANT corner, PER ORDER")
    px = py = 1.2
    dep = 0.4
    ec = np.zeros((3, 3, 3, 3), dtype=complex)
    ec[:, :] = _AIR
    for i, j in ((0, 0), (1, 0), (0, 1)):
        ec[i, j] = _OOP
    for mount, th, ph in (("normal", 0.0, 0.0),
                          ("conical 20/35", np.deg2rad(20.0),
                           np.deg2rad(35.0))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arms = {}
            for no in (7, 9):
                arms[("rcwa", no)] = rcwa_jones_2d(
                    px, py, upsample(ec, int(np.ceil((4 * no + 1) / 3))),
                    NSUB, NSUP, dep, WL, theta=th, phi=ph,
                    n_orders_x=no, n_orders_y=no)
            for form in ("laurent", "li"):
                arms[("hyb", form)] = pmm_jones_2d(
                    px, py, ec, NSUB, NSUP, dep, WL, theta=th, phi=ph,
                    degree=9, n_orders=13, formulation=form, stabilize=True)
        ref = arms[("rcwa", 9)]
        d_rc, dT_rc = per_order(*arms[("rcwa", 7)][:3], *ref[:3])
        print(f"  {mount}: oracle DRIFT rcwa 7->9   dR = {d_rc:.3e}  "
              f"dT = {dT_rc:.3e}")
        for form in ("laurent", "li"):
            a = arms[("hyb", form)]
            d, dT = per_order(*a[:3], *ref[:3])
            dJ = float(np.max(np.abs(a[3] - ref[3])))
            print(f"  {mount}: oracle SPREAD hyb-{form}(13) vs rcwa(9)  "
                  f"dR = {d:.3e}  dT = {dT:.3e}  dJones = {dJ:.3e}")
            RES[f"T6|{mount}|spread_{form}"] = dict(dR=d, dT=dT, dJ=dJ)
        RES[f"T6|{mount}|rcwa_drift"] = dict(dR=d_rc, dT=dT_rc)
        prev = None
        for M in (5, 6, 7):
            s = stag(px, py, ec, dep, M, th=th, ph=ph, no=3)
            d, dT = per_order(*s[:3], *ref[:3])
            dJ = float(np.max(np.abs(s[3] - ref[3])))
            clo = float(np.max(np.abs(s[1].sum(axis=1) + s[2].sum(axis=1) - 1)))
            mv = "" if prev is None else (
                f"   self-move {max(per_order(*s[:3], *prev[:3])):.2e}")
            print(f"    staggered M={M} dim={4*(3*(M-1))**2:5d}  dR = "
                  f"{d:.3e}  dT = {dT:.3e}  dJones = {dJ:.3e}  "
                  f"|R+T-1| = {clo:.2e}{mv}")
            RES[f"T6|{mount}|M{M}"] = dict(dR=d, dT=dT, dJ=dJ, closure=clo)
            prev = s
        # NO-FLOOR, two-sided
        a = stag(px, py, ec, dep, 6, th=th, ph=ph, no=3)
        b = stag(px, py, ec, dep, 6, th=th, ph=ph, no=8)
        mv = max(per_order(*a[:3], *b[:3]))
        hy = max(per_order(*arms[("hyb", "laurent")][:3],
                           *arms[("hyb", "li")][:3]))
        print(f"    NO-FLOOR: staggered n_orders 3 -> 8 moves {mv:.3e}; "
              f"the hybrid's two E_z rules differ by {hy:.3e}")
        RES[f"T6|{mount}|nofloor"] = dict(staggered=mv, hybrid=hy)


# ============================ T7: cascade closure ========================= #
def t7():
    print("\n### T7  cascade closure vs DEPTH (a growing mode shows as "
          "exp(+|Re lam| k0 L))")
    px = py = 1.2
    ec_u = np.broadcast_to(_OOP, (2, 2, 3, 3)).copy()
    ec_p = np.zeros((2, 2, 3, 3), dtype=complex)
    ec_p[:, :] = _AIR
    ec_p[0, 0] = _OOP
    ec_l = ec_p.copy()
    ec_l[0, 0] = _OOP_LOSSY
    for name, ec, herm in (("uniform Hermitian", ec_u, True),
                           ("pillar Hermitian", ec_p, True),
                           ("pillar LOSSY", ec_l, False)):
        for mount, th, ph in (("normal", 0.0, 0.0),
                              ("conical 25/40", np.deg2rad(25.0),
                               np.deg2rad(40.0))):
            row = []
            for dep in (0.25, 1.0, 3.0):
                o, R, T, J = stag(px, py, ec, dep, 7, th=th, ph=ph, no=3)
                tot = R.sum(axis=1) + T.sum(axis=1)
                row.append(float(np.max(np.abs(tot - 1.0))) if herm
                           else float(np.max(1.0 - tot)))
            k0 = 2 * np.pi / WL
            sol = TS.Granet2DTransverseE(px, py, 2, 2, 7, ec,
                                         alpha0x=NSUP * np.sin(th) *
                                         np.cos(ph) * k0,
                                         alpha0y=NSUP * np.sin(th) *
                                         np.sin(ph) * k0, k0=k0)
            Wf, Vf, lf, Wb, Vb, lb = TS._region_modes_oop(sol)
            grow = float(np.max(np.exp(-np.real(lf) * k0 * 3.0)))
            lab = "|R+T-1|" if herm else "absorbed"
            print(f"  {name:18s} {mount:14s} {lab} at 0.25/1/3 lam = "
                  f"{row[0]:.2e} / {row[1]:.2e} / {row[2]:.2e}   "
                  f"fwd/bwd = {lf.size}/{lb.size}   max fwd growth "
                  f"@3lam = {grow:.4e}")
            RES[f"T7|{name}|{mount}"] = dict(d025=row[0], d1=row[1],
                                             d3=row[2], nfwd=int(lf.size),
                                             growth=grow)


# ================================ T8: stacks ============================== #
def t8():
    print("\n### T8  stacks")
    px = py = 1.2
    ec_o = np.zeros((2, 2, 3, 3), dtype=complex)
    ec_o[:, :] = _AIR
    ec_o[0, 0] = _OOP
    lc = pc.uniaxial(1.5, 1.8, 90.0, azim_deg=31.5)      # IN-PLANE
    ec_i = np.zeros((2, 2, 3, 3), dtype=complex)
    ec_i[:, :] = _AIR
    ec_i[0, 0] = lc
    th, ph = np.deg2rad(20.0), np.deg2rad(35.0)
    # (a) split-layer consistency: one OOP layer of depth d == two of d/2
    for mount, t_, p_ in (("normal", 0.0, 0.0), ("conical 20/35", th, ph)):
        st1 = PMM2DStackPure(px, py, n_superstrate=NSUP, n_substrate=NSUB,
                             n_modes=6, n_orders=3)
        st1.add_layer(0.4, eps_cell=ec_o).set_source(WL, theta=t_, phi=p_)
        st2 = PMM2DStackPure(px, py, n_superstrate=NSUP, n_substrate=NSUB,
                             n_modes=6, n_orders=3)
        st2.add_layer(0.2, eps_cell=ec_o).add_layer(0.2, eps_cell=ec_o)
        st2.set_source(WL, theta=t_, phi=p_)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = st1.solve(jones=True)
            b = st2.solve(jones=True)
        d = max(per_order(*a[:3], *b[:3]))
        dJ = float(np.max(np.abs(a[3] - b[3])))
        print(f"  (a) one 0.4 layer vs two 0.2 layers, {mount}: "
              f"per-order {d:.3e}  Jones {dJ:.3e}")
        RES[f"T8a|{mount}"] = dict(d=d, dJ=dJ)
    # (b) all-UNIFORM out-of-plane multilayer vs berreman multilayer, oblique
    layers = [(_OOP, 0.30), (_NONREC, 0.22), (_OOP_LOSSY, 0.17)]
    for mount, t_, p_ in (("oblique 25", np.deg2rad(25.0), 0.0),
                          ("conical 25/40", np.deg2rad(25.0),
                           np.deg2rad(40.0))):
        Rb, Tb, Jrb, _ = berreman_jones_1d(layers, NSUB, NSUP, WL,
                                           angle=t_, phi=p_)
        for M in (6, 8):
            st = PMM2DStackPure(0.9, 0.9, n_superstrate=NSUP,
                                n_substrate=NSUB, n_modes=M, n_orders=3)
            for t33, d in layers:
                st.add_layer(d, eps=t33)
            st.set_source(WL, theta=t_, phi=p_)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                o, R, T, J = st.solve(jones=True)
            dR = float(np.max(np.abs(R.sum(axis=1) - Rb)))
            dT = float(np.max(np.abs(T.sum(axis=1) - Tb)))
            dJ = float(np.max(np.abs(J - Jrb)))
            print(f"  (b) uniform OOP multilayer vs berreman, {mount}, M={M}: "
                  f"dR = {dR:.3e}  dT = {dT:.3e}  dJones = {dJ:.3e}")
            RES[f"T8b|{mount}|M{M}"] = dict(dR=dR, dT=dT, dJ=dJ)
    # (c) MIXED stack: out-of-plane layer over an in-plane one over a uniform
    st = PMM2DStackPure(px, py, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=6, n_orders=3)
    st.add_layer(0.25, eps_cell=ec_o).add_layer(0.20, eps_cell=ec_i)
    st.add_layer(0.15, eps=2.25).set_source(WL, theta=th, phi=ph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve(jones=True, retain_internal=True)
        A = st.layer_absorption()
    clo = float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1)))
    budget = float(np.max(np.abs(A.sum(axis=0)
                                 - (1 - R.sum(axis=1) - T.sum(axis=1)))))
    print(f"  (c) mixed OOP|in-plane|uniform stack, conical: |R+T-1| = "
          f"{clo:.2e}   layer_absorption budget |sum A - (1-R-T)| = "
          f"{budget:.2e}")
    RES["T8c"] = dict(closure=clo, budget=budget)
    # (d) lossy mixed stack: the absorption budget must close on a real deficit
    ec_lo = ec_o.copy()
    ec_lo[0, 0] = _OOP_LOSSY
    st = PMM2DStackPure(px, py, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=6, n_orders=3)
    st.add_layer(0.30, eps_cell=ec_lo).add_layer(0.20, eps_cell=ec_i)
    st.set_source(WL, theta=th, phi=ph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve(jones=True, retain_internal=True)
        A = st.layer_absorption()
    deficit = 1 - R.sum(axis=1) - T.sum(axis=1)
    budget = float(np.max(np.abs(A.sum(axis=0) - deficit)))
    print(f"  (d) LOSSY mixed stack: absorbed {deficit}, per-layer "
          f"{A.sum(axis=0)}, budget residual {budget:.2e}")
    RES["T8d"] = dict(budget=budget, deficit=deficit.tolist(),
                      per_layer=A.tolist())


# ========================= T9: fail-before controls ======================= #
def t9():
    print("\n### T9  fail-before controls (uniform slab vs berreman, "
          "conical 25/40, off-azimuth director)")
    px = py = 0.9
    dep = 0.35
    th, ph = np.deg2rad(25.0), np.deg2rad(40.0)
    for name, t33 in (("lossless azim25", _OOP),
                      ("NON-RECIPROCAL azim25", _NONREC)):
        Rb, Tb, Jrb, _ = berreman_jones_1d([(t33, dep)], NSUB, NSUP, WL,
                                           angle=th, phi=ph)
        variants = {"reference": t33}
        v = np.array(t33, dtype=complex)
        v[0, 2] = v[1, 2] = v[2, 0] = v[2, 1] = 0.0
        variants["drop OOP"] = v
        v = np.array(t33, dtype=complex)
        v[0, 2] *= -1
        v[1, 2] *= -1
        v[2, 0] *= -1
        v[2, 1] *= -1
        variants["negate OOP"] = v
        v = np.array(t33, dtype=complex)
        v[0, 2], v[2, 0] = t33[2, 0], t33[0, 2]
        v[1, 2], v[2, 1] = t33[2, 1], t33[1, 2]
        variants["transpose OOP"] = v
        for vn, t in variants.items():
            cell = np.broadcast_to(t, (2, 2, 3, 3)).copy()
            o, R, T, J = stag(px, py, cell, dep, 8, th=th, ph=ph, no=3)
            dR = float(np.max(np.abs(R.sum(axis=1) - Rb)))
            dJ = float(np.max(np.abs(J - Jrb)))
            print(f"  {name:22s} {vn:14s} dR = {dR:.3e}  dJones = {dJ:.3e}")
            RES[f"T9|{name}|{vn}"] = dict(dR=dR, dJ=dJ)


# ================================ T10: cost =============================== #
def t10():
    print("\n### T10  cost: dimension, wall time, peak RSS")
    px = py = 1.2
    k0 = 2 * np.pi / WL
    ec_o = np.broadcast_to(_OOP, (3, 3, 3, 3)).copy()
    ec_i = np.array(ec_o)
    ec_i[..., 0, 2] = ec_i[..., 1, 2] = 0.0
    ec_i[..., 2, 0] = ec_i[..., 2, 1] = 0.0
    for M in (5, 6, 7, 8):
        row = {}
        for tag, ec, fn in (("in-plane", ec_i, TS._region_modes),
                            ("out-of-plane", ec_o, TS._region_modes_oop)):
            tracemalloc.start()
            t0 = time.perf_counter()
            sol = TS.Granet2DTransverseE(px, py, 3, 3, M, ec, k0=k0)
            out = fn(sol)
            dt = time.perf_counter() - t0
            _cur, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            dim = sol.dimtot
            row[tag] = (dim, dt, peak / 2 ** 20)
            del sol, out
        print(f"  M={M}: in-plane dim {row['in-plane'][0]:5d} "
              f"{row['in-plane'][1]:6.2f} s {row['in-plane'][2]:7.1f} MB  |  "
              f"out-of-plane dim {row['out-of-plane'][0]:5d} "
              f"{row['out-of-plane'][1]:6.2f} s {row['out-of-plane'][2]:7.1f}"
              f" MB   ratio t = "
              f"{row['out-of-plane'][1] / row['in-plane'][1]:.2f}x  RSS = "
              f"{row['out-of-plane'][2] / row['in-plane'][2]:.2f}x")
        RES[f"T10|M{M}"] = dict(inplane=row["in-plane"],
                                oop=row["out-of-plane"])


def main():
    pc.banner("GATE 3 -- integration measurement tables")
    for f in (t5, t6, t7, t8, t9, t10):
        f()
    with open(os.path.join(OUT, "g3_integration_tables.json"), "w") as fh:
        json.dump(RES, fh, indent=1, default=str)
    print("\nwrote results/g3_integration_tables.json")


if __name__ == "__main__":
    main()

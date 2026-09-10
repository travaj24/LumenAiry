"""Q2 -- THE ANCHOR, derived here and measured against three independent
references.

The formula, from the geometry (nothing taken from the fix):

    a slanted PATTERNED region is solved in ``u = x - t_x z``, ``v = y - t_y z``,
    ``w = z``, a frame anchored at the layer's TOP face.  The state the cascade
    carries is therefore the FRAME Fourier coefficient
    ``F_m exp(i k0 alpha_m . (u, v))``.  At ``w = 0`` frame and lab coincide, so
    the superstrate side (R, T, the reflection Jones) needs nothing.  At the
    layer's BOTTOM the SAME plane is ``z = d`` with ``(u, v) = (x - t d)``, so
    against the substrate's lab basis ``exp(i k0 alpha_m . x)``

        A_lab(m) = exp(+i k0 alpha_m . W) A_frame(m),
        W = sum over the layers that enter a frame of slant_j * d_j

    with the PUBLIC sign fixed by measurement below (the conjugate arm is worse
    than applying nothing).  ``alpha_m`` is real for every order, so the factor
    is unimodular: no efficiency and no energy check can see it.

ARMS: shipped / none (x exp(-i k0 a.W)) / conjugate (x exp(-2i k0 a.W)) / the
best possible SINGLE GLOBAL phase.
REFERENCES: (a) the hybrid's own K = 5/15/25 z-staircase of the SAME solid;
(b) the PURE engine; (c) the 1-D engine (q2c_oned.py).
"""
from __future__ import annotations

import os
import time
import warnings

import _lib as L
import numpy as np

TX = 0.5                 # tx * d = 0.5 * 0.45 um = 0.225 um = PX / 4 (QUARTER)
FRAC = 4
RUNGS = (5, 15, 25)
U = L.staircase_upsample_factor(6, FRAC, RUNGS)      # = 100 -> 600-pixel cells

BASE6 = np.array([       # SQUARE cell -- the shape the PURE engine also takes
    [2.10, 2.10, 1.30, 1.30, 1.30, 1.30],
    [3.05, 2.60, 1.30, 1.72, 1.30, 1.30],
    [1.30, 1.30, 1.30, 1.30, 2.20, 1.30],
    [1.30, 2.44, 2.44, 1.30, 1.30, 1.30],
    [1.30, 1.30, 1.95, 1.95, 1.30, 1.72],
    [1.30, 1.30, 1.30, 1.30, 1.30, 1.30],
], dtype=float)


def arms(a, ref, W, k0):
    """(shipped, none, conjugate, best-global) residuals of ``a`` vs ``ref``.

    ``best_global`` is the BEST SINGLE GLOBAL PHASE the UN-ANCHORED amplitudes
    could possibly wear -- the arm that decides whether the correction is
    genuinely per-order.  Taking it on the SHIPPED amplitudes instead would be
    vacuous: the optimum over ``phi`` includes ``phi = 0``, so it can only ever
    read at or below the shipped residual."""
    ship, n = L.amp_residual(a, ref)
    unanchored = L.rephase(a, (-W[0], -W[1]), k0)
    none_, _ = L.amp_residual(unanchored, ref)
    conj_, _ = L.amp_residual(L.rephase(a, (-2 * W[0], -2 * W[1]), k0), ref)
    glob, phi = L.best_global_phase(unanchored, ref)
    return dict(shipped=ship, none=none_, conj=conj_, best_global=glob,
                best_global_phase=phi, n_orders_compared=n)


def part_a():
    """(a) against the engine's OWN fine staircase, three mounts, K ladder."""
    k0 = L.k0_of()
    W = (TX * L.DTHICK, 0.0)
    out = {"walk_m": W, "walk_over_period": W[0] / L.PX, "U": U}
    for mount in L.MOUNTS:
        st, _ = L.solve_slanted(mount, tx=TX, n_orders=5)
        a = st.per_order_amplitudes("transmission")
        Jt = st.jones_transmission()
        row = {}
        oracles = {}
        for K in RUNGS:
            t0 = time.time()
            so, _o = L.solve_staircase(mount, U=U, frac_denom=FRAC, K=K,
                                       n_orders=5)
            b = so.per_order_amplitudes("transmission")
            oracles[K] = b
            r = arms(a, b, W, k0)
            r["seconds"] = round(time.time() - t0, 2)
            # the zeroth-order Jones on the same oracle
            Jb = so.jones_transmission()
            p0 = int(np.where((np.asarray(a["orders"])[:, 0] == 0)
                              & (np.asarray(a["orders"])[:, 1] == 0))[0][0])
            P0 = complex(np.exp(1j * k0 * (a["kx"][p0] * W[0]
                                           + a["ky"][p0] * W[1])))
            r["jones_shipped"] = L.jones_residual(Jt, Jb)
            r["jones_none"] = L.jones_residual(Jt / P0, Jb)
            r["jones_conj"] = L.jones_residual(Jt * np.conj(P0) / P0, Jb)
            r["P0"] = [P0.real, P0.imag]
            r["abs_P0"] = abs(P0)
            r["arg_P0"] = float(np.angle(P0))
            row["K%d" % K] = r
        # the ORACLE's own step, and the hybrid's own truncation step
        row["oracle_step_K15_K25"] = L.amp_residual(oracles[15],
                                                    oracles[25])[0]
        row["oracle_step_K5_K15"] = L.amp_residual(oracles[5], oracles[15])[0]
        st7, _ = L.solve_slanted(mount, tx=TX, n_orders=7)
        a7 = st7.per_order_amplitudes("transmission")
        row["hybrid_own_n_orders_step_5_7"] = L.amp_residual(a, a7)[0]
        row["shipped_first_last_ratio"] = (row["K5"]["shipped"]
                                           / row["K25"]["shipped"])
        row["none_first_last_ratio"] = (row["K5"]["none"]
                                        / row["K25"]["none"])
        out[mount] = row
    return out


def part_unimodular():
    """The factor is unimodular on EVERY retained order, propagating and
    evanescent alike -- so no efficiency can move."""
    k0 = L.k0_of()
    W = (TX * L.DTHICK, 0.0)
    st, _ = L.solve_slanted("conical25_40", tx=TX, ty=0.0, n_orders=7)
    a = st.per_order_amplitudes("transmission")
    P = np.exp(1j * k0 * (a["kx"] * W[0] + a["ky"] * W[1]))
    kz = np.asarray(a["kz"])
    return dict(n_orders_total=int(P.size),
                n_evanescent=int(np.sum(np.real(kz) <= 1e-12)),
                max_abs_dev=float(np.max(np.abs(np.abs(P) - 1.0))),
                max_arg=float(np.max(np.abs(np.angle(P)))))


def part_b():
    """(b) the PURE engine, DEPRECATED HERE -- ``n_modes = 7`` on a 6 x 6 cell
    is a ~7000-dof dense complex eig and runs for tens of minutes.  The
    cross-engine arm lives in ``q2b_pure.py`` at ``n_modes`` 4 and 5, where it
    is already converged (the reading moves < 1% between the two rungs)."""
    from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
    k0 = L.k0_of()
    W = (TX * L.DTHICK, 0.0)
    res = {}
    for mount in ("oblique25", "conical25_40"):
        th, ph = L.MOUNTS[mount]
        hy = {}
        for M in (5, 7, 9):
            st = L.hybrid(n_orders=M)
            st.add_layer(L.DTHICK, eps_cell=BASE6, slant=(TX, 0.0))
            st.set_source(L.WL, theta=th, phi=ph)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st.solve()
            hy[M] = (st.per_order_amplitudes("transmission"),
                     st.jones_transmission())
        pu = PMM2DStackPure(L.PX, L.PY, n_superstrate=L.NSUP,
                            n_substrate=L.NSUB, n_modes=7, n_orders=3)
        pu.add_layer(L.DTHICK, eps_cell=BASE6, slant=(TX, 0.0))
        pu.set_source(L.WL, theta=th, phi=ph)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pu.solve()
        pa = pu.per_order_amplitudes("transmission")
        pj = pu.jones_transmission()
        a5, j5 = hy[5]
        r = arms(a5, pa, W, k0)
        r["jones_shipped"] = L.jones_residual(j5, pj)
        p0 = int(np.where((np.asarray(a5["orders"])[:, 0] == 0)
                          & (np.asarray(a5["orders"])[:, 1] == 0))[0][0])
        P0 = complex(np.exp(1j * k0 * (a5["kx"][p0] * W[0]
                                       + a5["ky"][p0] * W[1])))
        r["jones_none"] = L.jones_residual(j5 / P0, pj)
        r["jones_conj"] = L.jones_residual(j5 * np.conj(P0) / P0, pj)
        r["hybrid_own_step_5_7"] = L.amp_residual(a5, hy[7][0])[0]
        r["hybrid_own_step_7_9"] = L.amp_residual(hy[7][0], hy[9][0])[0]
        r["hybrid_own_jones_step_5_7"] = L.jones_residual(j5, hy[7][1])
        res[mount] = r
    return res


def main():
    t0 = time.time()
    payload = dict(fixture=dict(period_x=L.PX, period_y=L.PY, wl=L.WL,
                                thickness=L.DTHICK, tx=TX, n_sup=L.NSUP,
                                n_sub=L.NSUB, base_cell=L.BASE.tolist(),
                                base6_cell=BASE6.tolist(),
                                walk_fraction_of_period=1.0 / FRAC,
                                staircase_rungs=RUNGS,
                                staircase_cell_cols=6 * U))
    payload["unimodular"] = part_unimodular()
    payload["a_staircase"] = part_a()
    if os.environ.get("Q2_PURE"):          # see part_b's docstring
        payload["b_pure"] = part_b()
    payload["seconds"] = round(time.time() - t0, 1)
    for k, v in payload["a_staircase"].items():
        if not isinstance(v, dict) or "K5" not in v:
            continue
        print("--", k)
        for K in RUNGS:
            r = v["K%d" % K]
            print("   K=%-2d  ship %.4e  none %.4e  conj %.4e  glob %.4e "
                  "| Jt ship %.4e none %.4e conj %.4e"
                  % (K, r["shipped"], r["none"], r["conj"], r["best_global"],
                     r["jones_shipped"], r["jones_none"], r["jones_conj"]))
        print("   oracle step K15->K25 %.4e | hybrid n_orders 5->7 %.4e "
              "| ship first/last %.3fx  none first/last %.4fx"
              % (v["oracle_step_K15_K25"], v["hybrid_own_n_orders_step_5_7"],
                 v["shipped_first_last_ratio"], v["none_first_last_ratio"]))
    print("-- pure:", payload.get("b_pure", "skipped -- see q2b_pure.py"))
    print("-- unimodular:", payload["unimodular"])
    L.dump("q2_anchor", payload)


if __name__ == "__main__":
    main()

"""Q3 -- COMPOSITION: the sum rule, the two null shapes, the layer-split
identity's BLINDNESS to the sum, the reflection round-trip question, and the
shear-CONTINUED scope (audit O1).

Oracles are z-staircases built here with EXACT integer pixel rolls (a
``Fraction`` check refuses any fixture whose rolls are not exact), so every
oracle is lab-referenced by construction.
"""
from __future__ import annotations

import math
import time
import warnings
from fractions import Fraction

import _lib as L
import numpy as np

K_ORACLE = 15
K_COARSE = 5
NORD = 5


def _rolls(base_cols, walk_periods, offset_periods, K):
    """Exact per-slice MIDPOINT rolls, as Fractions of a cell of ``base_cols``
    columns, plus the smallest upsample ``U`` making them all integers."""
    w = Fraction(walk_periods).limit_denominator(10 ** 6)
    o = Fraction(offset_periods).limit_denominator(10 ** 6)
    fr = [o + Fraction(2 * k + 1, 2 * K) * w for k in range(K)]
    dens = [(Fraction(base_cols) * f).denominator for f in fr]
    U = 1
    for d in dens:
        U = U * d // math.gcd(U, d)
    px = [int(Fraction(base_cols * U) * f) for f in fr]
    return U, px


def sheared_layers(base, thickness, walk_periods, offset_periods, K):
    """``K`` vertical slices of a sheared region whose TOP face sits at
    ``offset_periods`` and which walks ``walk_periods`` over its thickness."""
    U, px = _rolls(base.shape[0], walk_periods, offset_periods, K)
    cell = L.upsample(base, U)
    return [dict(thickness=thickness / K,
                 eps_cell=np.roll(cell, p, axis=0)) for p in px]


def build(layers, mount, n_orders=NORD, wl=L.WL):
    st = L.hybrid(n_orders=n_orders)
    for lay in layers:
        kw = dict(lay)
        t = kw.pop("thickness")
        st.add_layer(t, **kw)
    th, ph = L.MOUNTS[mount]
    st.set_source(wl, theta=th, phi=ph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = st.solve()
    return st, out


def arms(a, ref, W, k0):
    ship, n = L.amp_residual(a, ref)
    return dict(shipped=ship, n=n,
                none=L.amp_residual(L.rephase(a, (-W[0], -W[1]), k0), ref)[0],
                conj=L.amp_residual(L.rephase(a, (-2 * W[0], -2 * W[1]), k0),
                                    ref)[0])


def main():
    t0 = time.time()
    k0 = L.k0_of()
    out = {}
    D = L.DTHICK
    TX = 0.5                                  # walk = P/4
    W1 = TX * D
    FILM = dict(thickness=0.12e-6, eps=2.30)
    REFL = dict(thickness=0.12e-6, eps=3.60)

    for mount in ("oblique25", "conical25_40"):
        row = {}

        # --- A: slanted PATTERNED over a UNIFORM film ---------------------
        st, _ = build([dict(thickness=D, eps_cell=L.BASE, slant=(TX, 0.0)),
                       FILM], mount)
        a = st.per_order_amplitudes("transmission")
        so, _ = build(sheared_layers(L.BASE, D, 0.25, 0.0, K_ORACLE) + [FILM],
                      mount)
        b = so.per_order_amplitudes("transmission")
        row["A_slanted_over_film"] = arms(a, b, (W1, 0.0), k0)
        so5, _ = build(sheared_layers(L.BASE, D, 0.25, 0.0, K_COARSE) + [FILM],
                       mount)
        row["A_oracle_step_K5_K15"] = L.amp_residual(
            so5.per_order_amplitudes("transmission"), b)[0]

        # --- B: UNIFORM film ABOVE the slanted layer ----------------------
        st, _ = build([FILM,
                       dict(thickness=D, eps_cell=L.BASE, slant=(TX, 0.0))],
                      mount)
        a = st.per_order_amplitudes("transmission")
        so, _ = build([FILM] + sheared_layers(L.BASE, D, 0.25, 0.0, K_ORACLE),
                      mount)
        b = so.per_order_amplitudes("transmission")
        row["B_film_above"] = arms(a, b, (W1, 0.0), k0)

        # --- C: the LAYER-SPLIT identity lives in q3b_split.py -------------
        # (a slanted layer's frame CONTINUES into the next slanted layer, so
        # the lower half's cell is passed AS WRITTEN; the first attempt here
        # rolled it by half the walk and so built a different solid).

        # --- D: TWO slanted layers, DIFFERENT slants and thicknesses ------
        d1, tx1 = 0.30e-6, 0.75            # walk 1 = 0.225 um = P/4
        d2, tx2 = 0.15e-6, 0.60            # walk 2 = 0.09  um = P/10
        Wa, Wb = tx1 * d1, tx2 * d2
        st, _ = build([dict(thickness=d1, eps_cell=L.BASE, slant=(tx1, 0.0)),
                       dict(thickness=d2, eps_cell=L.LOWER, slant=(tx2, 0.0))],
                      mount)
        a = st.per_order_amplitudes("transmission")
        # the oracle: layer 2 rides layer 1's accumulated walk (offset 1/4)
        so, _ = build(sheared_layers(L.BASE, d1, 0.25, 0.0, K_ORACLE)
                      + sheared_layers(L.LOWER, d2, 0.10, 0.25, K_ORACLE),
                      mount)
        b = so.per_order_amplitudes("transmission")
        Wt = Wa + Wb
        row["D_two_slants"] = dict(
            walk_um=[Wa * 1e6, Wb * 1e6, Wt * 1e6],
            sum=arms(a, b, (Wt, 0.0), k0),
            only_layer1=L.amp_residual(L.rephase(a, (-Wb, 0.0), k0), b)[0],
            only_layer2=L.amp_residual(L.rephase(a, (-Wa, 0.0), k0), b)[0])
        so5, _ = build(sheared_layers(L.BASE, d1, 0.25, 0.0, K_COARSE)
                       + sheared_layers(L.LOWER, d2, 0.10, 0.25, K_COARSE),
                       mount)
        row["D_oracle_step_K5_K15"] = L.amp_residual(
            so5.per_order_amplitudes("transmission"), b)[0]

        # --- E/F: the two NULL shapes BELOW a slanted patterned layer -----
        for tag, below in (("E_slanted_uniform_below",
                            dict(thickness=0.12e-6, eps=2.30,
                                 slant=(0.9, 0.0))),
                           ("F_consttile_slanted_below",
                            dict(thickness=0.12e-6,
                                 eps_cell=np.full((6, 4), 2.30),
                                 slant=(0.9, 0.0)))):
            st, _ = build([dict(thickness=D, eps_cell=L.BASE,
                                slant=(TX, 0.0)), below], mount)
            stv, _ = build([dict(thickness=D, eps_cell=L.BASE,
                                 slant=(TX, 0.0)), FILM], mount)
            av = stv.per_order_amplitudes("transmission")
            aa = st.per_order_amplitudes("transmission")
            Wextra = below.get("slant", (0.0, 0.0))[0] * below["thickness"]
            row[tag] = dict(
                vs_vertical_film=L.amp_residual(aa, av)[0],
                jones_sha_equal=(L.sha(st.jones_transmission())
                                 == L.sha(stv.jones_transmission())),
                cost_of_summing_it=L.amp_residual(
                    L.rephase(aa, (Wextra, 0.0), k0), av)[0])

        # --- G: the REFLECTION round-trip question ------------------------
        st, o_h = build([dict(thickness=D, eps_cell=L.BASE, slant=(TX, 0.0)),
                         REFL], mount)
        so, o_o = build(sheared_layers(L.BASE, D, 0.25, 0.0, K_ORACLE)
                        + [REFL], mount)
        so5, o_5 = build(sheared_layers(L.BASE, D, 0.25, 0.0, K_COARSE)
                         + [REFL], mount)
        ar = st.per_order_amplitudes("reflection")
        br = so.per_order_amplitudes("reflection")
        b5 = so5.per_order_amplitudes("reflection")
        p0 = int(np.where((np.asarray(ar["orders"])[:, 0] == 0)
                          & (np.asarray(ar["orders"])[:, 1] == 0))[0][0])
        P0 = complex(np.exp(1j * k0 * ar["kx"][p0] * W1))
        Jh, Jo = np.asarray(o_h[3]), np.asarray(o_o[3])
        row["G_reflection_round_trip"] = dict(
            dR=float(np.max(np.abs(np.asarray(o_h[1])
                                   - np.asarray(o_o[1])))),
            dR_oracle_step=float(np.max(np.abs(np.asarray(o_5[1])
                                               - np.asarray(o_o[1])))),
            dJones_as_returned=L.jones_residual(Jh, Jo),
            dJones_x_P0=L.jones_residual(Jh * P0, Jo),
            dJones_x_P0sq=L.jones_residual(Jh * P0 * P0, Jo),
            dJones_oracle_step_K5_K15=L.jones_residual(np.asarray(o_5[3]), Jo),
            per_order_refl_as_returned=L.amp_residual(ar, br)[0],
            per_order_refl_x_P=L.amp_residual(
                L.rephase(ar, (W1, 0.0), k0), br)[0],
            per_order_refl_x_P2=L.amp_residual(
                L.rephase(ar, (2 * W1, 0.0), k0), br)[0],
            per_order_refl_oracle_step=L.amp_residual(b5, br)[0])

        # --- H: SCOPE -- a PATTERNED layer BELOW a slanted one (O1) -------
        lower_pixels = L.LOWER.shape[0] // 4          # P/4 of a 12-col cell
        scope = {}
        st, oh = build([dict(thickness=D, eps_cell=L.BASE, slant=(TX, 0.0)),
                        dict(thickness=0.15e-6, eps_cell=L.LOWER)], mount)
        a = st.per_order_amplitudes("transmission")
        ar = st.per_order_amplitudes("reflection")
        for tag, shift in (("as_written", 0), ("plus_walk", +lower_pixels),
                           ("minus_walk", -lower_pixels)):
            lay = sheared_layers(L.BASE, D, 0.25, 0.0, K_ORACLE) + [
                dict(thickness=0.15e-6,
                     eps_cell=np.roll(L.LOWER, shift, axis=0))]
            so, oo = build(lay, mount)
            b = so.per_order_amplitudes("transmission")
            br = so.per_order_amplitudes("reflection")
            scope[tag] = dict(
                per_order_T=L.amp_residual(a, b)[0],
                dR=float(np.max(np.abs(np.asarray(oh[1])
                                       - np.asarray(oo[1])))),
                dJones_reflection=L.jones_residual(np.asarray(oh[3]),
                                                   np.asarray(oo[3])),
                per_order_R=L.amp_residual(ar, br)[0])
        row["H_scope_patterned_below"] = scope
        out[mount] = row

    out["seconds"] = round(time.time() - t0, 1)
    for m in ("oblique25", "conical25_40"):
        print("==", m)
        for k, v in out[m].items():
            print("  ", k, v)
    L.dump("q3_composition", out)


if __name__ == "__main__":
    main()

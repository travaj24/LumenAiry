"""P2 -- the COMPOSITION rule for the hybrid's frame anchor, and its SCOPE.

DERIVED (see the fix doc S2).  A slanted layer is solved in a frame
``u = x - t z`` anchored at ITS OWN top face, so the state the cascade carries
is the FRAME Fourier coefficient ``F_m``; the lab coefficient at a plane a
depth ``d`` below that anchor is ``F_m exp(-i alpha_m . t d)``.  Hence, in the
hybrid's PUBLIC (conjugated) gauge,

    A_lab(m) = exp(+i k0 (alpha_m . sum_j slant_j d_j)) A_frame(m)

summed over the layers that ACTUALLY ENTER A FRAME -- which in this engine is
NOT "every layer carrying a slant keyword":

  * a UNIFORM layer never stores a slant at all (``add_layer``'s uniform
    branch drops it) and is solved vertically;
  * a PATTERNED layer whose tile is CONSTANT-VALUED short-circuits to
    ``_homogeneous_modes`` in ``_build_layer_modes`` BEFORE the slant is read,
    so it too never enters a frame.

Both must contribute ZERO, and this probe measures what including them would
cost.

SCOPE, measured in case D: the far-field anchor is exact only while everything
BELOW a slanted layer is homogeneous (a lateral offset is a gauge there) or
continues the same shear.  With a PATTERNED layer below, the naive interface
matching is the shear-CONTINUED solid -- the lower layer riding along with the
walk -- and case D measures which geometry the cascade actually solves.
"""
import numpy as np

from _lib import align, arm, dump, mx  # noqa: I001

from lumenairy.elements.pmm import PMM2DStackHybrid

WL = 0.68e-6
K0 = 2.0 * np.pi / WL
# GEOMETRY NOTE.  px = 1.0 um at wl = 0.68 um keeps every order clear of a
# half-space cut-off (the +1 order sits at |alpha| = 1.10 against the
# superstrate's 1.0, evanescent by a wide margin), and no cell value equals a
# half-space eps.  The first fixture tried here (px = 1.2 um, a cell containing
# eps = 1.0 = the superstrate) drove the slanted-layer-over-a-film cascade to
# sum R + T = 2.6e+27 at n_orders 5 and 7 on some mounts -- the documented
# exactly-degenerate layer<->region mode match, which a slanted layer's
# generalized cascade meets more readily than a vertical one.
PX = PY = 1.00e-6
D1 = 0.50e-6                  # the slanted patterned layer
DF = 0.25e-6                  # the uniform film
DV = 0.30e-6                  # the vertical patterned layer (case D)
NSUP, NSUB = 1.0, 1.5
NXC = 6
FINE = 60
TSL = 1.0                     # t d / px = 0.5 -- a HALF-period walk
BG = 1.44
XPROF = np.array([3.24, 3.24, 2.10, 1.15, 1.15, 1.15])
NORD = 7
MOUNTS = {"oblique25": (np.deg2rad(25.0), 0.0),
          "conical25_40": (np.deg2rad(25.0), np.deg2rad(40.0))}
# case D runs a QUARTER walk on a 12-pixel lower cell so that +/- the walk are
# DIFFERENT translations (a half walk on a 6-cell is its own mirror).
TSLD = 0.5                    # t d / px = 0.25 -- a QUARTER-period walk
FINED = 120
NXD = 12
YPROF = np.array([1.44, 2.89, 2.89, 1.44, 1.44, 2.10, 2.10, 1.44, 1.44, 1.44,
                  1.44, 1.44])


def cell(n=NXC, prof=XPROF, base=NXC):
    c = np.full((n, n), BG, dtype=complex)
    p = np.repeat(prof, n // base)
    c[:, 0:n // 2] = p[:, None]
    return c


def _st(nord=NORD):
    return PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                            n_orders=nord)


def stair_layers(st, K, tsl=TSL, fine=FINE, d=D1, prof=XPROF, base=NXC):
    c = cell(fine, prof, base)
    for k in range(K):
        sh = fine * tsl * d / PX * (k + 0.5) / K
        assert abs(sh - round(sh)) < 1e-9, (fine, tsl, K, k, sh)
        st.add_layer(d / K, eps_cell=np.roll(c, int(round(sh)), axis=0))
    return st


def amps(st):
    a = st.per_order_amplitudes("transmission")
    return a


def arms(sa, sb, walk_x, d=1.0):
    """raw / x P / x conj(P), per order, both polarizations, stack ``sa``
    against the lab-referenced oracle ``sb``.  ``walk_x`` is the accumulated
    ``sum_j t_j d_j`` in metres."""
    A, B = amps(sa), amps(sb)
    j1, j2, _ = align(A["orders"], B["orders"])
    P = np.exp(1j * K0 * A["kx"] * walk_x)

    def dd(f):
        return max(mx(f(A["Ex"])[:, j1], B["Ex"][:, j2]),
                   mx(f(A["Ey"])[:, j1], B["Ey"][:, j2]))
    return dict(raw=dd(lambda z: z), times_P=dd(lambda z: z * P[None, :]),
                times_conj_P=dd(lambda z: z * np.conj(P)[None, :]),
                dJonesT_raw=mx(sa.jones_transmission(), sb.jones_transmission()),
                dJonesT_times_P=mx(sa.jones_transmission()
                                   * P[int(A_p0(sa))],
                                   sb.jones_transmission()))


def A_p0(st):
    return int(st._modal["p0"])


def main():
    out = {}
    for mname, (th, ph) in MOUNTS.items():
        row = {}

        # ---- A: slanted PATTERNED over a UNIFORM film ---------------------
        a1 = _st()
        a1.add_layer(D1, eps_cell=cell(), slant=(TSL, 0.0))
        a1.add_layer(DF, eps=3.6)
        a1.set_source(WL, theta=th, phi=ph)
        oA, RA, TA, JA = a1.solve()
        a2 = stair_layers(_st(), 15)
        a2.add_layer(DF, eps=3.6)
        a2.set_source(WL, theta=th, phi=ph)
        oA2, RA2, TA2, JA2 = a2.solve()
        a3 = stair_layers(_st(), 5)
        a3.add_layer(DF, eps=3.6)
        a3.set_source(WL, theta=th, phi=ph)
        oA3, RA3, TA3, JA3 = a3.solve()
        i1, i2, _ = align(oA, oA2)
        _i3a, i3b, _ = align(oA, oA3)
        row["A_slanted_over_uniform_film"] = dict(
            anchor_walk_m=TSL * D1,
            transmission=arms(a1, a2, TSL * D1),
            dR=mx(RA[:, i1], RA2[:, i2]),
            dJones_reflection=mx(JA, JA2),
            staircase_own_step_dR=mx(RA2[:, i2], RA3[:, i3b]),
            staircase_own_step_dJones_reflection=mx(JA2, JA3))

        # ---- B: UNIFORM film ABOVE the slanted PATTERNED layer -------------
        b1 = _st()
        b1.add_layer(DF, eps=3.6)
        b1.add_layer(D1, eps_cell=cell(), slant=(TSL, 0.0))
        b1.set_source(WL, theta=th, phi=ph)
        b1.solve()
        b2 = _st()
        b2.add_layer(DF, eps=3.6)
        stair_layers(b2, 15)
        b2.set_source(WL, theta=th, phi=ph)
        b2.solve()
        row["B_uniform_film_above"] = dict(
            anchor_walk_m=TSL * D1, transmission=arms(b1, b2, TSL * D1))

        # ---- C: the SUM rule -- two slanted halves vs the staircase --------
        c1 = _st()
        c1.add_layer(D1 / 2, eps_cell=cell(), slant=(TSL, 0.0))
        c1.add_layer(D1 / 2, eps_cell=cell(), slant=(TSL, 0.0))
        c1.set_source(WL, theta=th, phi=ph)
        c1.solve()
        c0 = _st()
        c0.add_layer(D1, eps_cell=cell(), slant=(TSL, 0.0))
        c0.set_source(WL, theta=th, phi=ph)
        c0.solve()
        cs = stair_layers(_st(), 15)
        cs.set_source(WL, theta=th, phi=ph)
        cs.solve()
        row["C_sum_rule"] = dict(
            split_identity_dJonesT=mx(c1.jones_transmission(),
                                      c0.jones_transmission()),
            two_halves_vs_staircase=arms(c1, cs, TSL * D1),
            only_one_half_in_the_sum=arms(c1, cs, TSL * D1 / 2),
            single_layer_vs_staircase=arms(c0, cs, TSL * D1))

        # ---- E / F: the two layers that must contribute NOTHING ------------
        e1 = _st()
        e1.add_layer(D1, eps_cell=cell(), slant=(TSL, 0.0))
        e1.add_layer(DF, eps=3.6, slant=(0.7, 0.0))      # slanted UNIFORM
        e1.set_source(WL, theta=th, phi=ph)
        e1.solve()
        row["E_slanted_uniform_below_contributes_nothing"] = dict(
            dJonesT_vs_vertical_film=mx(e1.jones_transmission(),
                                        a1.jones_transmission()),
            if_it_were_in_the_sum=mx(
                e1.jones_transmission()
                * np.exp(1j * K0 * amps(e1)["kx"][A_p0(e1)] * 0.7 * DF),
                a1.jones_transmission()))
        f1 = _st()
        f1.add_layer(D1, eps_cell=cell(), slant=(TSL, 0.0))
        f1.add_layer(DF, eps_cell=np.full((NXC, NXC), 3.6 + 0j),
                     slant=(0.7, 0.0))                    # CONSTANT tile
        f1.set_source(WL, theta=th, phi=ph)
        f1.solve()
        row["F_constant_tile_below_contributes_nothing"] = dict(
            dJonesT_vs_uniform_film=mx(f1.jones_transmission(),
                                       a1.jones_transmission()),
            if_it_were_in_the_sum=mx(
                f1.jones_transmission()
                * np.exp(1j * K0 * amps(f1)["kx"][A_p0(f1)] * 0.7 * DF),
                a1.jones_transmission()))

        # ---- D: a PATTERNED layer BELOW a slanted one ----------------------
        walk_px = TSLD * D1 / PX                    # 0.25 period
        roll = int(round(NXD * walk_px))            # 3 pixels of 12
        d1s = _st()
        d1s.add_layer(D1, eps_cell=cell(), slant=(TSLD, 0.0))
        d1s.add_layer(DV, eps_cell=cell(NXD, YPROF, NXD))
        d1s.set_source(WL, theta=th, phi=ph)
        oD, RD, TD, JD = d1s.solve()
        ref = {}
        for tag, sh in (("as_written", 0), ("walk_plus", +roll),
                        ("walk_minus", -roll)):
            s = stair_layers(_st(), 15, tsl=TSLD, fine=FINED)
            s.add_layer(DV, eps_cell=np.roll(cell(NXD, YPROF, NXD), sh,
                                            axis=0))
            s.set_source(WL, theta=th, phi=ph)
            o2, R2, T2, J2 = s.solve()
            k1, k2, _ = align(oD, o2)
            ref[tag] = dict(transmission=arms(d1s, s, TSLD * D1),
                            dR=mx(RD[:, k1], R2[:, k2]),
                            dJones_reflection=mx(JD, J2))
        row["D_patterned_layer_below"] = dict(
            anchor_walk_m=TSLD * D1, roll_px=roll, cell_px=NXD, **ref)

        out[mname] = row
        A = row["A_slanted_over_uniform_film"]
        print(f"[{mname}] A film-below: T raw {A['transmission']['raw']:.3e} "
              f"x P {A['transmission']['times_P']:.3e} x conj "
              f"{A['transmission']['times_conj_P']:.3e} | dR {A['dR']:.3e} "
              f"dJ(refl) {A['dJones_reflection']:.3e} (staircase step "
              f"{A['staircase_own_step_dJones_reflection']:.2e})")
        B = row["B_uniform_film_above"]["transmission"]
        print(f"[{mname}] B film-above: T raw {B['raw']:.3e} x P "
              f"{B['times_P']:.3e} x conj {B['times_conj_P']:.3e}")
        C = row["C_sum_rule"]
        print(f"[{mname}] C sum rule: split identity dJonesT "
              f"{C['split_identity_dJonesT']:.3e} | two halves vs staircase: "
              f"full sum {C['two_halves_vs_staircase']['times_P']:.3e}, HALF "
              f"the sum {C['only_one_half_in_the_sum']['times_P']:.3e}, none "
              f"{C['two_halves_vs_staircase']['raw']:.3e}")
        E = row["E_slanted_uniform_below_contributes_nothing"]
        F = row["F_constant_tile_below_contributes_nothing"]
        print(f"[{mname}] E slanted uniform below: {E['dJonesT_vs_vertical_film']:.3e}"
              f" (in the sum it would cost {E['if_it_were_in_the_sum']:.3e}); "
              f"F constant tile: {F['dJonesT_vs_uniform_film']:.3e} (in the sum "
              f"{F['if_it_were_in_the_sum']:.3e})")
        for tag in ("as_written", "walk_plus", "walk_minus"):
            d = row["D_patterned_layer_below"][tag]
            print(f"[{mname}] D lower pattern {tag}: T raw "
                  f"{d['transmission']['raw']:.3e} x P "
                  f"{d['transmission']['times_P']:.3e} | dR {d['dR']:.3e} "
                  f"dJ(refl) {d['dJones_reflection']:.3e}")
    dump("p2_composition", out)
    print("arm", arm())


if __name__ == "__main__":
    main()

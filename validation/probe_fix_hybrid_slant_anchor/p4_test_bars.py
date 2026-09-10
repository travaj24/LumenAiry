"""P4 -- every quantity ``tests/unit/test_fix_hybrid_slant_transmission_anchor.py``
asserts, measured at the TEST fixture's own size, on BOTH builds.

TESTING_STANDARDS rule 5: no bar in that file may be pinned from one build, and
every bar's derivation lives in the assertion's comment next to the two
readings this probe produces.  The fixture here IS the test's fixture
(``n_orders = 5``, a 6-pixel x-asymmetric cell, a HALF-period walk), so the
numbers transfer without rescaling.

This probe assumes the FIX is present (it is written against the shipped,
lab-referenced amplitudes).  Every alternative arm is reconstructed from those
by re-referencing the anchor -- ``A_shipped * exp(+i k0 alpha_m . (w_alt -
w_shipped))`` -- so "no anchor at all" (``w_alt = 0``, the pre-fix library),
"the conjugate" (``w_alt = -w``) and "only half the sum" (``w_alt = w/2``) are
all available without a second checkout and without a library edit.  The test
file builds its pre-fix arm the other way, by monkeypatching
``stack2d._slant_frame_walk`` to ``(0.0, 0.0)``, which is the shipped decision
point; the two constructions agree by measurement (p1's census: dividing the
shipped anchor out returns the pre-fix reading to every printed digit).
"""
import time

import numpy as np

from _lib import align, arm, dump, mx, sha  # noqa: I001

from lumenairy.elements.pmm import PMM2DStackHybrid, PMM2DStackPure

WL = 0.68e-6
K0 = 2.0 * np.pi / WL
# The TEST fixture.  px = 1.0 um at wl = 0.68 um keeps every order clear of a
# half-space cut-off and no cell value equals a half-space eps -- the px = 1.2
# um / eps = 1.0 cell first tried here drove the slanted-layer-over-a-film
# cascade to sum R + T = 2.6e+27 on some (n_orders, mount) pairs.
PX = PY = 1.00e-6
DEP = 0.50e-6
DF = 0.25e-6
FEPS = 3.6
NSUP, NSUB = 1.0, 1.5
NXC, FINE = 6, 60
TSL = 1.0                     # t d / px = 0.5 -- a HALF-period walk
BG = 1.44
XPROF = np.array([3.24, 3.24, 2.10, 1.15, 1.15, 1.15])
NORD = 5
WALK = TSL * DEP              # the accumulated frame offset, metres
MOUNTS = {"oblique25": (np.deg2rad(25.0), 0.0),
          "conical25_40": (np.deg2rad(25.0), np.deg2rad(40.0))}


def cell(n=NXC):
    c = np.full((n, n), BG, dtype=complex)
    c[:, 0:n // 2] = np.repeat(XPROF, n // NXC)[:, None]
    return c


def _st(nord=NORD):
    return PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                            n_orders=nord)


def slanted(th, ph, nord=NORD, film=False):
    st = _st(nord)
    st.add_layer(DEP, eps_cell=cell(), slant=(TSL, 0.0))
    if film:
        st.add_layer(DF, eps=FEPS)
    st.set_source(WL, theta=th, phi=ph)
    st._RTJ = st.solve()
    return st


def stair(th, ph, K, nord=NORD, film=False):
    st = _st(nord)
    c = cell(FINE)
    for k in range(K):
        sh = FINE * TSL * DEP / PX * (k + 0.5) / K
        assert abs(sh - round(sh)) < 1e-9, (K, k, sh)
        st.add_layer(DEP / K, eps_cell=np.roll(c, int(round(sh)), axis=0))
    if film:
        st.add_layer(DF, eps=FEPS)
    st.set_source(WL, theta=th, phi=ph)
    st._RTJ = st.solve()
    return st


def rephase(st, w_alt, w_shipped=WALK):
    """The transmitted amplitudes this stack WOULD have returned had its frame
    anchor been ``w_alt`` instead of the shipped ``w_shipped``."""
    a = st.per_order_amplitudes("transmission")
    P = np.exp(1j * K0 * a["kx"] * (w_alt - w_shipped))
    return a["Ex"] * P[None, :], a["Ey"] * P[None, :], a["orders"], P


def dT(st, ref, w_alt=WALK, w_shipped=WALK):
    """Per-order, both-polarization distance of ``st``'s transmitted
    amplitudes (re-anchored at ``w_alt``) from the lab-referenced ``ref``."""
    Ex, Ey, o1, _ = rephase(st, w_alt, w_shipped)
    B = ref.per_order_amplitudes("transmission")
    j1, j2, _ = align(o1, B["orders"])
    return max(mx(Ex[:, j1], B["Ex"][:, j2]), mx(Ey[:, j1], B["Ey"][:, j2]))


def best_global(st, ref, w_alt=0.0):
    Ex, Ey, o1, _ = rephase(st, w_alt)
    B = ref.per_order_amplitudes("transmission")
    j1, j2, _ = align(o1, B["orders"])
    return float(min(max(mx(Ex[:, j1] * np.exp(1j * a), B["Ex"][:, j2]),
                         mx(Ey[:, j1] * np.exp(1j * a), B["Ey"][:, j2]))
                     for a in np.linspace(-np.pi, np.pi, 721)))


def main():
    out, t0 = {}, time.time()
    for mn, (th, ph) in MOUNTS.items():
        row = {}
        s = slanted(th, ph)
        k15, k5, k3 = (stair(th, ph, K) for K in (15, 5, 3))

        # 1. the three arms, per order, against the K = 15 staircase
        row["three_arms_vs_K15"] = dict(
            shipped=dT(s, k15), none=dT(s, k15, 0.0),
            conjugate=dT(s, k15, -WALK),
            best_single_global_phase_on_the_pre_fix_arm=best_global(s, k15),
            scale=float(np.max(np.abs(
                k15.per_order_amplitudes("transmission")["Ex"]))))

        # 2. the ORACLE's own convergence ladder (its error bound)
        Bs = {K: st.per_order_amplitudes("transmission")
              for K, st in ((15, k15), (5, k5), (3, k3))}
        i15, i5, _ = align(Bs[15]["orders"], Bs[5]["orders"])
        _a, i3, _ = align(Bs[15]["orders"], Bs[3]["orders"])
        row["staircase_own_ladder"] = dict(
            K3_vs_K15=max(mx(Bs[15]["Ex"][:, i15], Bs[3]["Ex"][:, i3]),
                          mx(Bs[15]["Ey"][:, i15], Bs[3]["Ey"][:, i3])),
            K5_vs_K15=max(mx(Bs[15]["Ex"][:, i15], Bs[5]["Ex"][:, i5]),
                          mx(Bs[15]["Ey"][:, i15], Bs[5]["Ey"][:, i5])))

        # 3. the shipped arm walks TOWARD the refining staircase
        row["shipped_vs_ladder"] = {f"K{K}": dT(s, st)
                                    for K, st in ((3, k3), (5, k5), (15, k15))}
        row["none_vs_ladder"] = {f"K{K}": dT(s, st, 0.0)
                                 for K, st in ((3, k3), (5, k5), (15, k15))}

        # 4. the zeroth-order Jones alone
        p0 = int(s._modal["p0"])
        P0 = complex(np.exp(1j * K0 * s._modal["kx"][p0] * WALK))
        row["jones_transmission"] = dict(
            shipped=mx(s.jones_transmission(), k15.jones_transmission()),
            none=mx(s.jones_transmission() * np.conj(P0),
                    k15.jones_transmission()),
            argP0=float(np.angle(P0)))

        # 5. R / T / the REFLECTION Jones against the same staircase
        oA, RA, TA, JA = s._RTJ
        oB, RB, TB, JB = k15._RTJ
        oC, RC, TC, JC = k5._RTJ
        i1, i2, _ = align(oA, oB)
        _x, i4, _ = align(oA, oC)
        row["reflection_and_efficiency"] = dict(
            dR=mx(RA[:, i1], RB[:, i2]), dT=mx(TA[:, i1], TB[:, i2]),
            dJones_reflection=mx(JA, JB),
            staircase_step_dR=mx(RB[:, i2], RC[:, i4]),
            staircase_step_dJones_reflection=mx(JB, JC),
            closure=float(np.max(RA.sum(axis=1) + TA.sum(axis=1))),
            staircase_closure=float(np.max(RB.sum(axis=1) + TB.sum(axis=1))))

        # 6. cross-engine: the PURE staggered engine on the same solid
        sp = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                            n_modes=4, n_orders=3)
        sp.add_layer(DEP, eps_cell=cell(), slant=(TSL, 0.0))
        sp.set_source(WL, theta=th, phi=ph)
        sp.solve(jones=True)
        Ap = sp.per_order_amplitudes("transmission")
        Ah = s.per_order_amplitudes("transmission")
        m1, m2, _ = align(Ap["orders"], Ah["orders"])
        Ex0, Ey0, _o, _P = rephase(s, 0.0)
        s7 = slanted(th, ph, nord=7)
        A7 = s7.per_order_amplitudes("transmission")
        n1, n2, _ = align(Ah["orders"], A7["orders"])
        row["pure_vs_hybrid"] = dict(
            shipped=max(mx(Ap["Ex"][:, m1], Ah["Ex"][:, m2]),
                        mx(Ap["Ey"][:, m1], Ah["Ey"][:, m2])),
            hybrid_pre_fix=max(mx(Ap["Ex"][:, m1], Ex0[:, m2]),
                               mx(Ap["Ey"][:, m1], Ey0[:, m2])),
            hybrid_own_n_orders_step=max(mx(Ah["Ex"][:, n1], A7["Ex"][:, n2]),
                                         mx(Ah["Ey"][:, n1], A7["Ey"][:, n2])),
            pure_jones_vs_hybrid_jones=mx(sp.jones_transmission(),
                                          s.jones_transmission()),
            pure_jones_vs_hybrid_pre_fix=mx(sp.jones_transmission(),
                                            s.jones_transmission()
                                            * np.conj(P0)))

        # 7. composition: two slanted halves
        c2 = _st()
        c2.add_layer(DEP / 2, eps_cell=cell(), slant=(TSL, 0.0))
        c2.add_layer(DEP / 2, eps_cell=cell(), slant=(TSL, 0.0))
        c2.set_source(WL, theta=th, phi=ph)
        c2._RTJ = c2.solve()
        row["composition"] = dict(
            split_identity_dJonesT=mx(c2.jones_transmission(),
                                      s.jones_transmission()),
            full_sum=dT(c2, k15), half_sum=dT(c2, k15, WALK / 2),
            no_sum=dT(c2, k15, 0.0), single_layer=dT(s, k15))

        # 8. a UNIFORM film BELOW: the reflection needs no round-trip phase
        a1 = slanted(th, ph, film=True)
        b15 = stair(th, ph, 15, film=True)
        b5 = stair(th, ph, 5, film=True)
        oA, RA, TA, JA = a1._RTJ
        oB, RB, TB, JB = b15._RTJ
        oC, RC, TC, JC = b5._RTJ
        i1, i2, _ = align(oA, oB)
        _x, i4, _ = align(oA, oC)
        row["film_below"] = dict(
            dR=mx(RA[:, i1], RB[:, i2]), dJones_reflection=mx(JA, JB),
            staircase_step_dR=mx(RB[:, i2], RC[:, i4]),
            staircase_step_dJones_reflection=mx(JB, JC),
            reflection_times_P0=mx(JA * P0, JB),
            reflection_times_P0_squared=mx(JA * P0 * P0, JB),
            T_shipped=dT(a1, b15), T_none=dT(a1, b15, 0.0),
            T_conjugate=dT(a1, b15, -WALK),
            closure=float(np.max(RA.sum(axis=1) + TA.sum(axis=1))))
        out[mn] = row

    # -- SCOPE: a PATTERNED layer BELOW a slanted one ------------------------
    # A QUARTER-period walk, so +walk and -walk are DIFFERENT translations of
    # the 12-pixel lower cell (at a half walk the cell is its own image).
    th, ph = MOUNTS["oblique25"]
    TSD, FD, NXD, DV = 0.5, 120, 12, 0.30e-6
    YPROF = np.array([1.44, 2.89, 2.89, 1.44, 1.44, 2.10, 2.10, 1.44, 1.44,
                      1.44, 1.44, 1.44])

    def lower(roll=0):
        c = np.full((NXD, NXD), BG, dtype=complex)
        c[:, 0:NXD // 2] = YPROF[:, None]
        return np.roll(c, roll, axis=0)
    g0 = _st()
    g0.add_layer(DEP, eps_cell=cell(), slant=(TSD, 0.0))
    g0.add_layer(DV, eps_cell=lower())
    g0.set_source(WL, theta=th, phi=ph)
    oA, RA, TA, JA = g0.solve()
    a = g0.per_order_amplitudes("transmission")
    roll = int(round(NXD * TSD * DEP / PX))
    gres = {}
    for tag, shft in (("as_written", 0), ("plus", +roll), ("minus", -roll)):
        s2 = _st()
        c = np.full((FD, FD), BG, dtype=complex)
        c[:, 0:FD // 2] = np.repeat(XPROF, FD // NXC)[:, None]
        for k in range(15):
            off = FD * TSD * DEP / PX * (k + 0.5) / 15
            assert abs(off - round(off)) < 1e-9, (k, off)
            s2.add_layer(DEP / 15, eps_cell=np.roll(c, int(round(off)),
                                                    axis=0))
        s2.add_layer(DV, eps_cell=lower(shft))
        s2.set_source(WL, theta=th, phi=ph)
        oB, RB, TB, JB = s2.solve()
        b = s2.per_order_amplitudes("transmission")
        i1, i2, _ = align(oA, oB)
        j1, j2, _ = align(a["orders"], b["orders"])
        gres[tag] = dict(
            T=max(mx(a["Ex"][:, j1], b["Ex"][:, j2]),
                  mx(a["Ey"][:, j1], b["Ey"][:, j2])),
            dR=mx(RA[:, i1], RB[:, i2]), dJ=mx(JA, JB))
    gres["roll_px"] = roll
    gres["closure"] = float(np.max(RA.sum(axis=1) + TA.sum(axis=1)))
    out["patterned_below_oblique25"] = gres

    # -- the two NULL rows -------------------------------------------------
    th, ph = MOUNTS["oblique25"]
    u0, u1, u2 = _st(), _st(), _st()
    u0.add_layer(DEP, eps=2.25)
    u1.add_layer(DEP, eps=2.25, slant=(TSL, 0.0))
    u2.add_layer(DEP, eps_cell=np.full((NXC, NXC), 2.25 + 0j), slant=(TSL, 0.0))
    for st in (u0, u1, u2):
        st.set_source(WL, theta=th, phi=ph)
        st.solve()
    P0u = complex(np.exp(1j * K0 * u1._modal["kx"][int(u1._modal["p0"])]
                         * WALK))
    out["nulls"] = dict(
        uniform_sha_equal=(sha(u0.jones_transmission())
                           == sha(u1.jones_transmission())),
        constant_tile_sha_equal=(sha(u0.jones_transmission())
                                 == sha(u2.jones_transmission())),
        if_anchored=mx(u1.jones_transmission() * P0u,
                       u0.jones_transmission()),
        argP0=float(np.angle(P0u)))
    out["_wall_s"] = time.time() - t0

    for mn in MOUNTS:
        r = out[mn]
        t = r["three_arms_vs_K15"]
        print(f"[{mn}] per-order T vs K15: shipped {t['shipped']:.3e}  none "
              f"{t['none']:.3e}  conj {t['conjugate']:.3e}  best-global(pre) "
              f"{t['best_single_global_phase_on_the_pre_fix_arm']:.3e}  "
              f"(scale {t['scale']:.3e})")
        print(f"[{mn}] staircase own ladder: K3 "
              f"{r['staircase_own_ladder']['K3_vs_K15']:.3e}  K5 "
              f"{r['staircase_own_ladder']['K5_vs_K15']:.3e}")
        print(f"[{mn}] shipped vs ladder: " + "  ".join(
            f"{k} {v:.3e}" for k, v in r["shipped_vs_ladder"].items())
            + "   | none vs ladder: " + "  ".join(
            f"{k} {v:.3e}" for k, v in r["none_vs_ladder"].items()))
        print(f"[{mn}] jonesT: shipped {r['jones_transmission']['shipped']:.3e}"
              f"  none {r['jones_transmission']['none']:.3e}  argP0 "
              f"{r['jones_transmission']['argP0']:.3f}")
        e = r["reflection_and_efficiency"]
        print(f"[{mn}] R/T/refl: dR {e['dR']:.3e} dT {e['dT']:.3e} dJ(refl) "
              f"{e['dJones_reflection']:.3e}  (staircase step dR "
              f"{e['staircase_step_dR']:.2e} dJ "
              f"{e['staircase_step_dJones_reflection']:.2e}); closure "
              f"{e['closure']:.6f} / staircase {e['staircase_closure']:.6f}")
        p = r["pure_vs_hybrid"]
        print(f"[{mn}] pure vs hybrid: shipped {p['shipped']:.3e}  pre-fix "
              f"{p['hybrid_pre_fix']:.3e}  hybrid n_orders step "
              f"{p['hybrid_own_n_orders_step']:.3e}  | jonesT shipped "
              f"{p['pure_jones_vs_hybrid_jones']:.3e} pre-fix "
              f"{p['pure_jones_vs_hybrid_pre_fix']:.3e}")
        c = r["composition"]
        print(f"[{mn}] composition: split identity "
              f"{c['split_identity_dJonesT']:.3e}  full sum {c['full_sum']:.3e}"
              f"  half sum {c['half_sum']:.3e}  no sum {c['no_sum']:.3e}  "
              f"single {c['single_layer']:.3e}")
        f = r["film_below"]
        print(f"[{mn}] film below: dR {f['dR']:.3e} dJ(refl) "
              f"{f['dJones_reflection']:.3e} (staircase step "
              f"{f['staircase_step_dJones_reflection']:.2e}); refl x P0 "
              f"{f['reflection_times_P0']:.3e} x P0^2 "
              f"{f['reflection_times_P0_squared']:.3e}; T shipped "
              f"{f['T_shipped']:.3e} none {f['T_none']:.3e} conj "
              f"{f['T_conjugate']:.3e}; closure {f['closure']:.6f}")
    g = out["patterned_below_oblique25"]
    print(f"[patterned below, oblique25, roll {g['roll_px']} px, closure "
          f"{g['closure']:.5f}] " + "  ".join(
              f"{t}: T {g[t]['T']:.3e} dR {g[t]['dR']:.3e} dJ {g[t]['dJ']:.3e}"
              for t in ("as_written", "plus", "minus")))
    print(f"[nulls] {out['nulls']}")
    print(f"[wall] {out['_wall_s']:.1f} s")
    dump("p4_test_bars", out)
    print("arm", arm())


if __name__ == "__main__":
    main()

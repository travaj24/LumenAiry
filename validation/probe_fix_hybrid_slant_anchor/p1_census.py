"""P1 -- reproduce D1 and CENSUS every public hybrid output on a slanted
PATTERNED layer.

D1 (VERIFY_PMM2D_STAGGERED_SLANT_2026_09_10 S3.4): ``PMM2DStackHybrid``'s
transmitted amplitudes on a slanted PATTERNED layer are FRAME-referenced -- the
per-order frame-anchor phase ``P_m = exp(+i k0 (alpha_m . slant) d)`` that maps
the sheared frame's exit plane back to the lab is never applied.

The oracle is the hybrid's OWN fine z-staircase of the same solid: a staircase
has no frame at all, so its transmitted amplitudes are lab-referenced BY
CONSTRUCTION, and running it through the same engine removes every convention
question.  THE WALK IS HALF A PERIOD (``t d = px / 2``) so ``P_m`` is not a
global phase (at a whole-period walk ``exp(2 pi i m) = 1`` for every order and
the anchor degenerates -- the trap ``v8_hybrid_anchor.py`` records).

This script is the SAME on both sides of the fix: it reports the shipped
amplitudes RAW, times ``P_m``, times ``conj(P_m)`` and against the best single
GLOBAL phase, so pre-fix the ``x P_m`` column wins and post-fix the ``raw``
column does.

Census axes: normal / oblique 25 / conical 25-40; every public output
(``solve`` R / T / reflection Jones, ``jones_transmission``,
``per_order_amplitudes('transmission')``, ``per_order_amplitudes('reflection')``,
``internal_field`` / ``layer_absorption`` reachability, ``solve_vs_wavelength``,
and the single-layer ``pmm_jones_2d`` entry); plus the two NULL rows a fix must
not disturb -- a slanted UNIFORM layer (which never enters a frame) and a
CONSTANT-tile slanted PATTERNED layer (which does, and whose truth is the plain
vertical slab).
"""
import numpy as np

from _lib import align, arm, dump, mx, sha  # noqa: I001

from lumenairy.elements.pmm import (PMM2DStackHybrid, PMM2DStackPure,
                                    pmm_jones_2d)

WL = 0.68e-6
K0 = 2.0 * np.pi / WL
PX = PY = 1.20e-6
DEP = 1.20e-6
NSUP, NSUB = 1.0, 1.5
NXC = 6
FINE = 60
TSL = 0.5                     # HALF a period over the layer
XPROF = np.array([4.0, 4.0, 2.0, 1.0, 1.0, 1.0])
NORD = 9
MOUNTS = {"normal": (0.0, 0.0),
          "oblique25": (np.deg2rad(25.0), 0.0),
          "conical25_40": (np.deg2rad(25.0), np.deg2rad(40.0))}


def cell(n=NXC):
    c = np.ones((n, n), dtype=complex)
    prof = np.repeat(XPROF, n // NXC)
    c[:, 0:n // 2] = prof[:, None]
    return c


def _st(nord=NORD):
    return PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                            n_orders=nord)


def hyb_slanted(theta, phi, tsl=TSL):
    st = _st()
    st.add_layer(DEP, eps_cell=cell(NXC), slant=(tsl, 0.0))
    st.set_source(WL, theta=theta, phi=phi)
    return st, st.solve()


def hyb_stair(theta, phi, K, tsl=TSL, fine=FINE):
    st = _st()
    c = cell(fine)
    d = DEP / K
    for k in range(K):
        sh = fine * tsl * (k + 0.5) / K       # integer for K in {3, 5, 15}
        assert abs(sh - round(sh)) < 1e-9, (fine, tsl, K, k, sh)
        st.add_layer(d, eps_cell=np.roll(c, int(round(sh)), axis=0))
    st.set_source(WL, theta=theta, phi=phi)
    return st, st.solve()


def pure_slanted(theta, phi, M=4):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=3)
    st.add_layer(DEP, eps_cell=cell(NXC), slant=(TSL, 0.0))
    st.set_source(WL, theta=theta, phi=phi)
    return st, st.solve(jones=True)


def anchor(st, sx=TSL, sy=0.0, d=DEP):
    """``P_m = exp(+i k0 (alpha_m . slant) d)`` on the PUBLIC amplitudes."""
    a = st.per_order_amplitudes("transmission")
    return np.exp(1j * K0 * (a["kx"] * sx + a["ky"] * sy) * d)


def four_arms(sh, ss, tsl=TSL):
    """raw / x P_m / x conj(P_m) / best single GLOBAL phase, per order, both
    polarizations, hybrid slanted against hybrid staircase."""
    A = sh.per_order_amplitudes("transmission")
    B = ss.per_order_amplitudes("transmission")
    j1, j2, _c = align(A["orders"], B["orders"])
    Pm = anchor(sh, sx=tsl)

    def d(f):
        return max(mx(f(A["Ex"])[:, j1], B["Ex"][:, j2]),
                   mx(f(A["Ey"])[:, j1], B["Ey"][:, j2]))
    best = min(
        max(mx(A["Ex"][:, j1] * np.exp(1j * a), B["Ex"][:, j2]),
            mx(A["Ey"][:, j1] * np.exp(1j * a), B["Ey"][:, j2]))
        for a in np.linspace(-np.pi, np.pi, 721))
    return dict(raw=d(lambda z: z), times_Pm=d(lambda z: z * Pm[None, :]),
                times_conj_Pm=d(lambda z: z * np.conj(Pm)[None, :]),
                best_single_global_phase=float(best),
                amplitude_scale=float(np.max(np.abs(B["Ex"]))))


def jt_arms(sh, ss, tsl=TSL):
    """The same four arms on the ZEROTH-order transmission Jones alone."""
    Jh, Js = sh.jones_transmission(), ss.jones_transmission()
    p0 = int(sh._modal["p0"])
    P0 = complex(anchor(sh, sx=tsl)[p0])
    best = min(float(np.max(np.abs(Jh * np.exp(1j * a) - Js)))
               for a in np.linspace(-np.pi, np.pi, 721))
    return dict(raw=mx(Jh, Js), times_P0=mx(Jh * P0, Js),
                times_conj_P0=mx(Jh * np.conj(P0), Js),
                best_single_global_phase=best, argP0=float(np.angle(P0)))


def main():
    out = {"fixture": dict(wl=WL, px=PX, depth=DEP, slant=[TSL, 0.0],
                           n_orders=NORD, cell=NXC, fine=FINE,
                           walk_periods=TSL * DEP / PX)}
    for mname, (th, ph) in MOUNTS.items():
        row = {}
        sh, (oh, Rh, Th, Jh) = hyb_slanted(th, ph)
        s15, (o15, R15, T15, J15) = hyb_stair(th, ph, 15)
        s5, (o5, R5, T5, J5) = hyb_stair(th, ph, 5)
        i1, i2, _c = align(oh, o15)
        _j5a, j5b, _c5 = align(oh, o5)
        row["solve_outputs"] = dict(
            dR=mx(Rh[:, i1], R15[:, i2]), dT=mx(Th[:, i1], T15[:, i2]),
            dJones_reflection=mx(Jh, J15),
            staircase_own_step_dR=mx(R15[:, i2], R5[:, j5b]),
            staircase_own_step_dJones_reflection=mx(J15, J5))
        row["per_order_transmission"] = four_arms(sh, s15)
        row["jones_transmission"] = jt_arms(sh, s15)
        # the REFLECTION port must need nothing at all
        A = sh.per_order_amplitudes("reflection")
        B = s15.per_order_amplitudes("reflection")
        k1, k2, _ = align(A["orders"], B["orders"])
        Pm = anchor(sh)
        row["per_order_reflection"] = dict(
            raw=max(mx(A["Ex"][:, k1], B["Ex"][:, k2]),
                    mx(A["Ey"][:, k1], B["Ey"][:, k2])),
            times_Pm=max(mx((A["Ex"] * Pm[None, :])[:, k1], B["Ex"][:, k2]),
                         mx((A["Ey"] * Pm[None, :])[:, k1], B["Ey"][:, k2])))
        # the PURE engine on the same fixture (its amplitudes ARE lab-ref'd)
        sp, _ = pure_slanted(th, ph)
        Ap = sp.per_order_amplitudes("transmission")
        Bs = s15.per_order_amplitudes("transmission")
        m1, m2, _ = align(Ap["orders"], Bs["orders"])
        Pp = np.exp(1j * K0 * Ap["kx"] * TSL * DEP)
        row["pure_per_order_transmission"] = dict(
            shipped=max(mx(Ap["Ex"][:, m1], Bs["Ex"][:, m2]),
                        mx(Ap["Ey"][:, m1], Bs["Ey"][:, m2])),
            divide_out_Pm=max(
                mx((Ap["Ex"] * np.conj(Pp)[None, :])[:, m1], Bs["Ex"][:, m2]),
                mx((Ap["Ey"] * np.conj(Pp)[None, :])[:, m1], Bs["Ey"][:, m2])))
        out[mname] = row
        r = row
        print(f"[{mname}] solve: dR {r['solve_outputs']['dR']:.3e} dT "
              f"{r['solve_outputs']['dT']:.3e} dJ(refl) "
              f"{r['solve_outputs']['dJones_reflection']:.3e}  (staircase step "
              f"dR {r['solve_outputs']['staircase_own_step_dR']:.2e} dJ "
              f"{r['solve_outputs']['staircase_own_step_dJones_reflection']:.2e})")
        p = r["per_order_transmission"]
        print(f"[{mname}] per-order T: raw {p['raw']:.3e}  x P_m "
              f"{p['times_Pm']:.3e}  x conj {p['times_conj_Pm']:.3e}  best "
              f"global {p['best_single_global_phase']:.3e}")
        q = r["jones_transmission"]
        print(f"[{mname}] jones_transmission: raw {q['raw']:.3e}  x P0 "
              f"{q['times_P0']:.3e}  x conj(P0) {q['times_conj_P0']:.3e}  best "
              f"global {q['best_single_global_phase']:.3e}")
        print(f"[{mname}] per-order R: raw "
              f"{r['per_order_reflection']['raw']:.3e}  (x P_m "
              f"{r['per_order_reflection']['times_Pm']:.3e})")
        print(f"[{mname}] PURE per-order T: shipped "
              f"{r['pure_per_order_transmission']['shipped']:.3e}  P_m divided "
              f"out {r['pure_per_order_transmission']['divide_out_Pm']:.3e}")

    # ---- a QUARTER-period walk at NORMAL incidence -------------------------
    # At the half-period walk P_m = exp(i pi m) is REAL, so "x P_m" and
    # "x conj(P_m)" coincide at normal incidence (kx0 = 0) and the two-sided
    # arm degenerates -- the same trap as the WHOLE-period walk, one order
    # down.  At a quarter period P_m = i^m and the two separate.
    sh, _ = hyb_slanted(0.0, 0.0, tsl=0.25)
    s15q, _ = hyb_stair(0.0, 0.0, 15, tsl=0.25, fine=120)
    out["normal_quarter_walk"] = dict(
        per_order_transmission=four_arms(sh, s15q, tsl=0.25),
        jones_transmission=jt_arms(sh, s15q, tsl=0.25))
    p = out["normal_quarter_walk"]["per_order_transmission"]
    print(f"[normal quarter-walk] per-order T: raw {p['raw']:.3e}  x P_m "
          f"{p['times_Pm']:.3e}  x conj {p['times_conj_Pm']:.3e}  best global "
          f"{p['best_single_global_phase']:.3e}")
    q = out["normal_quarter_walk"]["jones_transmission"]
    print(f"[normal quarter-walk] jones_transmission: raw {q['raw']:.3e}  x P0 "
          f"{q['times_P0']:.3e}  x conj(P0) {q['times_conj_P0']:.3e}")

    # ---- NULL 1: a slanted UNIFORM layer never enters a frame -------------
    def uni(slant, th, ph):
        st = _st(nord=5)
        st.add_layer(DEP, eps=2.25, slant=slant)
        st.set_source(WL, theta=th, phi=ph)
        st.solve()
        return st
    th, ph = MOUNTS["oblique25"]
    u0, u1 = uni(None, th, ph), uni((TSL, 0.0), th, ph)
    P0u = complex(anchor(u1)[int(u1._modal["p0"])])
    out["null_uniform_slanted"] = dict(
        sha_vertical=sha(u0.jones_transmission()),
        sha_slanted=sha(u1.jones_transmission()),
        dJones_transmission=mx(u0.jones_transmission(),
                               u1.jones_transmission()),
        if_anchor_were_applied=mx(u0.jones_transmission(),
                                  u1.jones_transmission() * P0u),
        stores_slant_key=[("slant" in L) for L in u1._layers])
    print("[null uniform] slanted vs vertical dJonesT "
          f"{out['null_uniform_slanted']['dJones_transmission']:.3e}  "
          f"(anchoring it would cost "
          f"{out['null_uniform_slanted']['if_anchor_were_applied']:.3e}); "
          f"layer stores a slant key: "
          f"{out['null_uniform_slanted']['stores_slant_key']}")

    # ---- NULL 2: a CONSTANT-tile PATTERNED slanted layer DOES enter one ----
    cst = np.full((NXC, NXC), 2.25 + 0j)
    sc = _st(nord=5)
    sc.add_layer(DEP, eps_cell=cst, slant=(TSL, 0.0))
    sc.set_source(WL, theta=th, phi=ph)
    sc.solve()
    P0c = complex(anchor(sc)[int(sc._modal["p0"])])
    out["null_constant_tile_patterned"] = dict(
        raw=mx(sc.jones_transmission(), u0.jones_transmission()),
        times_P0=mx(sc.jones_transmission() * P0c, u0.jones_transmission()),
        times_conj_P0=mx(sc.jones_transmission() * np.conj(P0c),
                         u0.jones_transmission()),
        modal_key=str(sc._mode_key(sc._layers[0], 1.0, 0.0, 0.0, WL))[:90])
    print("[null constant tile] vs the vertical uniform slab: raw "
          f"{out['null_constant_tile_patterned']['raw']:.3e}  x P0 "
          f"{out['null_constant_tile_patterned']['times_P0']:.3e}  x conj(P0) "
          f"{out['null_constant_tile_patterned']['times_conj_P0']:.3e}")

    # ---- reachability of the remaining public surfaces ---------------------
    reach = {}
    st = _st(nord=5)
    st.add_layer(DEP, eps_cell=cell(NXC), slant=(TSL, 0.0))
    st.set_source(WL, theta=th, phi=ph)
    try:
        st.solve(retain_internal=True)
        reach["solve(retain_internal=True)"] = "RETURNED"
    except Exception as e:
        reach["solve(retain_internal=True)"] = (
            f"{type(e).__name__}: {str(e)[:70]}")
    st.solve()
    for name, fn in (("internal_field", lambda: st.internal_field(0.5 * DEP)),
                     ("layer_absorption", st.layer_absorption)):
        try:
            fn()
            reach[name] = "RETURNED"
        except Exception as e:
            reach[name] = f"{type(e).__name__}: {str(e)[:70]}"
    st2 = _st(nord=5)
    st2.add_layer(DEP, eps_cell=cell(NXC), slant=(TSL, 0.0))
    st2.set_source(WL, theta=th, phi=ph)
    st2.solve_vs_wavelength([WL], theta=th, phi=ph)
    try:
        st2.jones_transmission()
        reach["solve_vs_wavelength -> jones_transmission"] = "RETURNED"
    except Exception as e:
        reach["solve_vs_wavelength -> jones_transmission"] = (
            f"{type(e).__name__}: {str(e)[:70]}")
    tcell = np.zeros((NXC, NXC, 3, 3), dtype=complex)
    tcell[..., 0, 0] = tcell[..., 1, 1] = tcell[..., 2, 2] = cell(NXC)
    try:
        r2 = pmm_jones_2d(PX, PY, tcell, NSUB, NSUP, 0.55e-6, WL,
                          theta=th, phi=ph, n_orders=5, slant=(TSL, 0.0))
        reach["pmm_jones_2d returns"] = (
            f"tuple of {len(r2)}: " + ", ".join(str(np.shape(v)) for v in r2)
            if isinstance(r2, tuple) else
            f"{type(r2).__name__} shape {np.shape(r2)}")
        reach["pmm_jones_2d exposes a transmission surface"] = any(
            hasattr(v, "jones_transmission") for v in
            (r2 if isinstance(r2, tuple) else (r2,)))
    except Exception as e:                      # noqa: BLE001
        reach["pmm_jones_2d returns"] = f"{type(e).__name__}: {str(e)[:70]}"
    out["reachability"] = reach
    for k, v in reach.items():
        print(f"[reach] {k}: {v}")

    dump("p1_census", out)
    print("arm", arm())


if __name__ == "__main__":
    main()

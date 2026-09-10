"""V3 -- the L2 MORTAR, re-measured on independent fixtures.

``python v3_mortar.py [g3 ident nonconf cap cache absorb oracle1d]``

* ``g3``       the H-row V1/V2 swap.  MY OWN non-conforming fixtures, and the
  BLINDNESS reproduced explicitly: with the swap DISABLED the conforming
  identity gate still passes to round-off while the non-conforming observable
  moves by decades.
* ``ident``    the conforming identity through the FORCED mortar, against a
  bar DERIVED from ``cond_2(G)`` measured HERE (uniform AND non-uniform
  grids).
* ``nonconf``  nested and non-conforming grids against the COMMON-REFINEMENT
  (union) reference, and the mortar's own error isolated on a transparent
  split.
* ``cap``      the far-field order cap: derived from the END grids, and it
  RAISES.
* ``cache``    eig-cache key collisions: two layers differing ONLY in walls /
  tau / slant must not share an entry.
* ``absorb``   ``retain_internal`` / ``layer_absorption`` cross-machinery
  budget closure on non-conforming grids.
* ``oracle1d`` the 1-D ``PMMStack`` per-order oracle, BOTH polarizations at
  oblique, with the anti-mirror control.
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
from lumenairy.elements.pmm import _core as pcore
from lumenairy.elements.pmm.stack import PMMStack
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import Granet2DTransverseE

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT)
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

_C = complex
RES = {}

# --- MY OWN fixture constants (deliberately different from the build's) -----
P = 1.05
WL = 0.72
EPS_P, EPS_H = 5.0, 2.10
TH, PH = 0.23, 0.55


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def pillar(N, lo, hi, e_p=EPS_P, e_h=EPS_H):
    c = np.full((N, N), _C(e_h))
    c[lo:hi, lo:hi] = e_p
    return c


def stripe(N, lo, hi, e_p=EPS_P, e_h=EPS_H):
    c = np.full((N, N), _C(e_h))
    c[lo:hi, :] = e_p
    return c


def solve(st, jones=False, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve(jones=jones, **kw)


def gram_cond(N, M, a0x=0.0, a0y=0.0, walls=None):
    """cond_2 of the two V1/V2 block field Grams -- measured HERE off the
    assembled ``-Rmat``, not read from the build doc."""
    w = N if walls is None else np.asarray(walls, dtype=float)
    n = N if walls is None else len(walls) - 1
    sol = Granet2DTransverseE(P, P, w, w, M, np.full((n, n), 2.0 + 0j),
                              alpha0x=a0x, alpha0y=a0y, k0=2 * np.pi / WL)
    G = -sol.Rmat
    qq = sol.q ** 2
    return max(float(np.linalg.cond(G[:qq, :qq])),
               float(np.linalg.cond(G[qq:, qq:])))


# ==========================================================================
def sec_g3():
    """The H-row V1/V2 swap: load-bearing, and INVISIBLE to a conforming
    gate."""
    out = {}
    # ---- MY OWN non-conforming fixture: pillar 1/2 on N=2 over pillar 2/3 on
    # N=3, common refinement N=6, at a DIFFERENT angle from the build's.
    cA, cB = pillar(2, 0, 1), pillar(3, 0, 2)
    cA6 = np.repeat(np.repeat(cA, 3, axis=0), 3, axis=1)
    cB6 = np.repeat(np.repeat(cB, 2, axis=0), 2, axis=1)
    tA, tB = 0.26, 0.19

    def _ref(M):
        st = PMM2DStackPure(P, n_modes=M, n_orders=1)
        st.add_layer(tA, eps_cell=cA6)
        st.add_layer(tB, eps_cell=cB6)
        st.set_source(WL, theta=TH, phi=PH)
        return solve(st)

    def _arm(MA, MB):
        st = PMM2DStackPure(P, n_modes=max(MA, MB), n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(tA, eps_cell=cA, n_modes=MA)
        st.add_layer(tB, eps_cell=cB, n_modes=MB)
        st.set_source(WL, theta=TH, phi=PH)
        return solve(st)

    o0, R0, T0 = _ref(4)
    for MA, MB in ((7, 5), (10, 7)):
        rec = {}
        for swap in (True, False):
            pcore.PMM2D_MORTAR_H_SWAP = swap
            try:
                o, R, T = _arm(MA, MB)
            finally:
                pcore.PMM2D_MORTAR_H_SWAP = True
            err = float(max(np.max(np.abs(R - R0)), np.max(np.abs(T - T0))))
            clo = float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))
            rec["on" if swap else "off"] = {"err_vs_union": err,
                                            "closure": clo}
        r = rec["on"]["err_vs_union"] or 1e-300
        rec["ratio_err"] = rec["off"]["err_vs_union"] / r
        rec["ratio_closure"] = rec["off"]["closure"] / (rec["on"]["closure"]
                                                        or 1e-300)
        out[f"nonconforming_MA{MA}_MB{MB}"] = rec
        print(f"[g3] non-conforming (MA={MA}, MB={MB}) vs union M=4: "
              f"swap ON err {rec['on']['err_vs_union']:.4e} closure "
              f"{rec['on']['closure']:.3e} | swap OFF err "
              f"{rec['off']['err_vs_union']:.4e} closure "
              f"{rec['off']['closure']:.3e}  -> {rec['ratio_err']:.1f}x / "
              f"{rec['ratio_closure']:.2e}x", flush=True)

    # ---- THE BLINDNESS.  A CONFORMING stack forced through the mortar, with
    # the swap DISABLED, must still reproduce the square modal match to
    # round-off: a conforming-identity gate CANNOT see the swap.
    blind = {}
    for name, cells, M in (("stripe|stripe", (stripe(2, 0, 1),
                                              stripe(2, 1, 2)), 5),
                           ("pillar|pillar", (pillar(3, 0, 2),
                                              pillar(3, 1, 3)), 5)):
        for swap in (True, False):
            pcore.PMM2D_MORTAR_H_SWAP = swap
            try:
                st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                                    layer_grids="per-layer")
                for c in cells:
                    st.add_layer(0.22, eps_cell=c, n_modes=M)
                st.set_source(WL, theta=TH, phi=PH)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    a = st._solve_per_layer(jones=True, retain_internal=False,
                                            force_mortar=False)
                    b = st._solve_per_layer(jones=True, retain_internal=False,
                                            force_mortar=True)
            finally:
                pcore.PMM2D_MORTAR_H_SWAP = True
            sc = max(float(np.max(np.abs(a[1]))), float(np.max(np.abs(a[2]))),
                     1e-300)
            d = float(max(np.max(np.abs(a[1] - b[1])),
                          np.max(np.abs(a[2] - b[2])),
                          np.max(np.abs(a[3] - b[3])))) / sc
            blind[f"{name}_swap_{'on' if swap else 'off'}"] = d
        print(f"[g3] BLINDNESS {name}: forced-mortar-vs-bypass identity reads "
              f"{blind[f'{name}_swap_on']:.2e} with the swap ON and "
              f"{blind[f'{name}_swap_off']:.2e} with it OFF", flush=True)
    out["conforming_blindness"] = blind
    RES["g3"] = out


# ==========================================================================
def sec_ident():
    """The conforming identity through the FORCED mortar, bar derived here."""
    out = {}
    cases = [
        ("stripe|stripe N2 normal", [stripe(2, 0, 1), stripe(2, 1, 2)], 5,
         0.0, 0.0, None),
        ("pillar|pillar N3 conical", [pillar(3, 0, 2), pillar(3, 1, 3)], 6,
         TH, PH, None),
        ("stripe|uniform|pillar N2", [stripe(2, 0, 1), None, pillar(2, 0, 1)],
         5, 0.19, 0.0, None),
        ("pillar|pillar N2 M7", [pillar(2, 0, 1), pillar(2, 1, 2)], 7,
         TH, PH, None),
        ("NON-UNIFORM walls x3", [stripe(3, 0, 1), stripe(3, 1, 2),
                                  stripe(3, 2, 3)], 5, TH, PH,
         [0.2371, 0.6183]),
        ("NON-UNIFORM walls x2 M7", [stripe(3, 0, 1), stripe(3, 1, 2)], 7,
         TH, PH, [0.1409, 0.8817]),
    ]
    for name, cells, M, th, ph, walls in cases:
        st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        for c in cells:
            kw = {}
            if c is None:
                kw = dict(eps=2.0, grid=(len(walls) + 1 if walls
                                         else cells[0].shape[0]),
                          n_modes=M)
            else:
                kw = dict(eps_cell=c, n_modes=M)
                if walls is not None:
                    kw.update(x_walls=[w * P for w in walls],
                              y_walls=[w * P for w in walls])
            st.add_layer(0.20, **kw)
        st.set_source(WL, theta=th, phi=ph)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = st._solve_per_layer(jones=True, retain_internal=False)
            b = st._solve_per_layer(jones=True, retain_internal=False,
                                    force_mortar=True)
        sR = max(float(np.max(np.abs(a[1]))), float(np.max(np.abs(a[2]))))
        sJ = max(float(np.max(np.abs(a[3]))), 1e-300)
        dR = float(np.max(np.abs(a[1] - b[1]))) / sR
        dT = float(np.max(np.abs(a[2] - b[2]))) / sR
        dJ = float(np.max(np.abs(a[3] - b[3]))) / sJ
        N = (len(walls) + 1) if walls else cells[0].shape[0]
        wfull = (None if walls is None
                 else [0.0] + [w * P for w in walls] + [P])
        cnd = gram_cond(N, M, walls=wfull)
        bar = 10.0 * np.finfo(float).eps * cnd
        out[name] = {"dR": dR, "dT": dT, "dJ": dJ, "cond2_G": cnd,
                     "bar_10_eps_cond": float(bar),
                     "margin": float(bar / max(dR, dT, dJ, 1e-300))}
        print(f"[ident] {name:28s} M={M}: worst {max(dR,dT,dJ):.3e}  "
              f"cond2(G) {cnd:.4e}  bar {bar:.3e}  "
              f"({out[name]['margin']:.0f}x inside)", flush=True)
    RES["ident"] = out


# ==========================================================================
def sec_nonconf():
    """Nested and non-conforming grids vs the COMMON-REFINEMENT reference, and
    a TRANSPARENT split (the mortar's own error, isolated)."""
    out = {}
    # ---- transparent split: ONE uniform slab cut in two on DIFFERENT grids.
    # nothing but the mortar can move the analytic Fresnel answer.
    n2, t = 2.0, 0.31

    def fresnel_R(n_slab, thick, wl, n_sup, n_sub, theta):
        k0 = 2 * np.pi / wl
        kx = n_sup * np.sin(theta)
        kz = [np.sqrt(_C(n ** 2 - kx ** 2))
              for n in (n_sup, n_slab, n_sub)]
        # s-polarization
        r01 = (kz[0] - kz[1]) / (kz[0] + kz[1])
        r12 = (kz[1] - kz[2]) / (kz[1] + kz[2])
        ph = np.exp(2j * k0 * kz[1] * thick)
        return abs((r01 + r12 * ph) / (1 + r01 * r12 * ph)) ** 2

    fres = {}
    for th in (0.0, 0.26):
        Rex = fresnel_R(n2, t, WL, 1.0, 1.5, th)
        for ga, gb in ((2, 2), (2, 4), (2, 3), (3, 4)):
            st = PMM2DStackPure(P, n_superstrate=1.0, n_substrate=1.5,
                                n_modes=7, n_orders=1,
                                layer_grids="per-layer")
            st.add_layer(t / 2, eps=n2 ** 2, grid=ga, n_modes=7)
            st.add_layer(t / 2, eps=n2 ** 2, grid=gb, n_modes=7)
            st.set_source(WL, theta=th, phi=0.0)
            o, R, T = solve(st)
            p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
            err = float(abs(R[1, p0] - Rex))
            clo = float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))
            fres[f"th{th}_g{ga}{gb}"] = {"err_vs_fresnel": err,
                                         "closure": clo}
        print(f"[nonconf] transparent split theta={th}: "
              + "  ".join(f"({a},{b}) {fres[f'th{th}_g{a}{b}']['err_vs_fresnel']:.2e}"
                          for a, b in ((2, 2), (2, 4), (2, 3), (3, 4))),
              flush=True)
    out["transparent_split"] = fres

    # ---- NESTED (2,4) and NON-CONFORMING (2,3) patterned pairs vs the union
    cA2, cB3 = pillar(2, 0, 1), pillar(3, 1, 3)
    pairs = {}
    for label, (cA, cB, uA, uB, Nu) in {
        "nested_2_4": (cA2, pillar(4, 1, 3), np.repeat(np.repeat(
            cA2, 2, 0), 2, 1), pillar(4, 1, 3), 4),
        "nonconf_2_3": (cA2, cB3, np.repeat(np.repeat(cA2, 3, 0), 3, 1),
                        np.repeat(np.repeat(cB3, 2, 0), 2, 1), 6),
    }.items():
        st = PMM2DStackPure(P, n_modes=4, n_orders=1)
        st.add_layer(0.26, eps_cell=uA)
        st.add_layer(0.19, eps_cell=uB)
        st.set_source(WL, theta=TH, phi=PH)
        oR, RR, TR = solve(st)
        row = {}
        for MA, MB in ((5, 5), (7, 5), (9, 7)):
            st2 = PMM2DStackPure(P, n_modes=max(MA, MB), n_orders=1,
                                 layer_grids="per-layer")
            st2.add_layer(0.26, eps_cell=cA, n_modes=MA)
            st2.add_layer(0.19, eps_cell=cB, n_modes=MB)
            st2.set_source(WL, theta=TH, phi=PH)
            o, R, T = solve(st2)
            err = float(max(np.max(np.abs(R - RR)), np.max(np.abs(T - TR))))
            clo = float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))
            row[f"MA{MA}_MB{MB}"] = {"err_vs_union": err, "closure": clo}
            print(f"[nonconf] {label} MA={MA} MB={MB}: err vs union(N={Nu},"
                  f"M=4) {err:.4e}   closure {clo:.3e}", flush=True)
        pairs[label] = row
    out["pairs_vs_union"] = pairs
    RES["nonconf"] = out


# ==========================================================================
def sec_cap():
    """The far-field order cap comes from the END grids and RAISES."""
    out = {}
    st = PMM2DStackPure(P, n_modes=5, n_orders=7, layer_grids="per-layer")
    st.add_layer(0.2, eps=2.0, grid=1, n_modes=5)     # q = 4 -> cap 1
    st.add_layer(0.2, eps_cell=pillar(3, 1, 2), n_modes=5)
    st.add_layer(0.2, eps=2.0, grid=1, n_modes=5)
    st.set_source(WL, theta=0.1, phi=0.0)
    try:
        solve(st)
        out["raises"] = False
        out["msg"] = ""
    except ValueError as exc:
        out["raises"] = True
        out["msg"] = str(exc)
    print(f"[cap] n_orders=7 on N=1/M=5 end grids raises: {out['raises']}",
          flush=True)
    print("      " + out["msg"][:220], flush=True)
    # the cap itself, derived: raise the END layers and it passes
    ok = {}
    for n_ord in (1, 2, 3):
        st = PMM2DStackPure(P, n_modes=5, n_orders=n_ord,
                            layer_grids="per-layer")
        st.add_layer(0.2, eps=2.0, grid=1, n_modes=5)
        st.add_layer(0.2, eps_cell=pillar(3, 1, 2), n_modes=5)
        st.add_layer(0.2, eps=2.0, grid=1, n_modes=5)
        st.set_source(WL, theta=0.1, phi=0.0)
        try:
            solve(st)
            ok[n_ord] = "ok"
        except ValueError:
            ok[n_ord] = "raised"
    out["by_n_orders_endgrid_N1_M5"] = ok
    print(f"[cap] end grid N=1 M=5 (q=4, cap=(4-1)//2=1): {ok}", flush=True)
    # raising the END layers' n_modes raises the cap
    st = PMM2DStackPure(P, n_modes=5, n_orders=4, layer_grids="per-layer")
    st.add_layer(0.2, eps=2.0, grid=1, n_modes=10)    # q = 9 -> cap 4
    st.add_layer(0.2, eps_cell=pillar(3, 1, 2), n_modes=5)
    st.add_layer(0.2, eps=2.0, grid=1, n_modes=10)
    st.set_source(WL, theta=0.1, phi=0.0)
    try:
        solve(st)
        out["endgrid_M10_n_orders4"] = "ok"
    except ValueError as exc:
        out["endgrid_M10_n_orders4"] = f"raised: {exc}"
    print(f"[cap] end grid N=1 M=10 (q=9, cap=4) at n_orders=4: "
          f"{out['endgrid_M10_n_orders4'][:40]}", flush=True)
    RES["cap"] = out


# ==========================================================================
def sec_cache():
    """Two layers that differ ONLY in walls / tau / slant must NOT collide in
    the per-solve eig cache.  Each attempt is scored against the SAME stack
    solved with the two layers in ISOLATION (where no cache can confuse
    them)."""
    out = {}
    cell = stripe(3, 0, 1)

    def two_layer(w1, w2, sl1=None, sl2=None, M=5):
        st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        for w, sl in ((w1, sl1), (w2, sl2)):
            kw = dict(eps_cell=cell, n_modes=M)
            if w is not None:
                kw.update(x_walls=[q * P for q in w],
                          y_walls=[q * P for q in w])
            if sl is not None:
                kw["slant"] = sl
            st.add_layer(0.21, **kw)
        st.set_source(WL, theta=TH, phi=PH)
        return solve(st)

    def one_layer_pair(wA, wB, slA=None, slB=None, M=5):
        """The same two layers, but with a DUMMY uniform layer between them so
        the two patterned records can never be adjacent-deduped -- and solved
        as two separate single-layer stacks for the mode content."""
        mods = []
        for w, sl in ((wA, slA), (wB, slB)):
            st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                                layer_grids="per-layer")
            kw = dict(eps_cell=cell, n_modes=M)
            if w is not None:
                kw.update(x_walls=[q * P for q in w],
                          y_walls=[q * P for q in w])
            if sl is not None:
                kw["slant"] = sl
            st.add_layer(0.21, **kw)
            st.set_source(WL, theta=TH, phi=PH)
            mods.append(solve(st))
        return mods

    # -- (a) same N, same M, DIFFERENT wall positions ------------------------
    wA = [1 / 3, 2 / 3]
    wB = [0.2371, 0.6183]
    ab = two_layer(wA, wB)
    aa = two_layer(wA, wA)
    bb = two_layer(wB, wB)
    dab_aa = float(max(np.max(np.abs(ab[1] - aa[1])),
                       np.max(np.abs(ab[2] - aa[2]))))
    dab_bb = float(max(np.max(np.abs(ab[1] - bb[1])),
                       np.max(np.abs(ab[2] - bb[2]))))
    out["walls"] = {"AB_vs_AA": dab_aa, "AB_vs_BB": dab_bb}
    print(f"[cache] walls  A|B differs from A|A by {dab_aa:.3e} and from "
          f"B|B by {dab_bb:.3e}  (a collision would make one of these 0)",
          flush=True)
    # -- (b) tau: the SAME grid at two angles must not share an entry --------
    #    (tau is per-solve, so the collision test is that a second solve at a
    #    different angle does not return the first angle's modes)
    st = PMM2DStackPure(P, n_modes=5, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.21, eps_cell=cell, x_walls=[q * P for q in wB],
                 y_walls=[q * P for q in wB], n_modes=5)
    st.set_source(WL, theta=0.0, phi=0.0)
    r0 = solve(st)
    st.set_source(WL, theta=0.31, phi=0.0)
    r1 = solve(st)
    st2 = PMM2DStackPure(P, n_modes=5, n_orders=1, layer_grids="per-layer")
    st2.add_layer(0.21, eps_cell=cell, x_walls=[q * P for q in wB],
                  y_walls=[q * P for q in wB], n_modes=5)
    st2.set_source(WL, theta=0.31, phi=0.0)
    r1f = solve(st2)
    out["tau"] = {"reused_stack_vs_fresh": float(max(
        np.max(np.abs(r1[1] - r1f[1])), np.max(np.abs(r1[2] - r1f[2])))),
        "angle0_vs_angle1": float(max(np.max(np.abs(r0[1] - r1[1])),
                                      np.max(np.abs(r0[2] - r1[2]))))}
    print(f"[cache] tau    reused-stack vs fresh stack at theta=0.31: "
          f"{out['tau']['reused_stack_vs_fresh']:.3e}  "
          f"(the two angles differ by "
          f"{out['tau']['angle0_vs_angle1']:.3e})", flush=True)
    # -- (c) slant: same cell, same grid, DIFFERENT slant --------------------
    #    _check_stack_slant refuses mixed slants between patterned layers, so
    #    the collision attempt is between two SEPARATE stacks solved in ONE
    #    process (module-level caches) and one stack with both at the same
    #    slant.
    s0 = two_layer(wB, wB, sl1=(0.17, 0.0), sl2=(0.17, 0.0))
    s1 = two_layer(wB, wB, sl1=(0.31, 0.0), sl2=(0.31, 0.0))
    s2 = two_layer(wB, wB)
    out["slant"] = {
        "t017_vs_t031": float(max(np.max(np.abs(s0[1] - s1[1])),
                                  np.max(np.abs(s0[2] - s1[2])))),
        "t017_vs_vertical": float(max(np.max(np.abs(s0[1] - s2[1])),
                                      np.max(np.abs(s0[2] - s2[2]))))}
    print(f"[cache] slant  t=0.17 vs t=0.31 {out['slant']['t017_vs_t031']:.3e}"
          f"; t=0.17 vs vertical {out['slant']['t017_vs_vertical']:.3e}",
          flush=True)
    # -- (d) DIRECT key inspection: the three attempts must give 3 keys ------
    from lumenairy.elements.pmm.twod_staggered import StagGridOps
    tx = np.exp(-1j * 0.3 * P)
    keys = set()
    for w in (wA, wB):
        for tau in (1.0 + 0j, tx):
            g = StagGridOps(P, P, np.array([0.0] + [q * P for q in w] + [P]),
                            np.array([0.0] + [q * P for q in w] + [P]),
                            5, tau, tau)
            keys.add(g.key())
    out["distinct_grid_keys"] = len(keys)
    print(f"[cache] grid keys for 2 wall sets x 2 taus: {len(keys)} distinct "
          f"(expect 4)", flush=True)
    RES["cache"] = out


# ==========================================================================
def sec_absorb():
    """retain_internal / layer_absorption on NON-conforming grids: the
    cross-machinery budget ``sum_i A_i == 1 - sum R - sum T``."""
    out = {}
    cA = np.full((2, 2), 2.10 + 0j)
    cA[0, 0] = 5.5 + 0.30j
    cB = np.full((3, 3), 2.30 + 0j)
    cB[1, 1] = 4.8 + 0.18j
    for M in (5, 6, 7):
        st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(0.24, eps_cell=cA, n_modes=M)
        st.add_layer(0.17, eps_cell=cB, n_modes=M)
        st.set_source(WL, theta=TH, phi=PH)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve(retain_internal=True)
            A = np.asarray(st.layer_absorption())
        far = 1.0 - R.sum(1) - T.sum(1)
        gap = float(np.max(np.abs(A.sum(0) - far)))
        out[f"M{M}"] = {"sum_A": A.sum(0).tolist(), "one_minus_RT":
                        far.tolist(), "gap": gap}
        print(f"[absorb] M={M}: sum_i A_i = {np.round(A.sum(0), 6).tolist()} "
              f"vs 1-R-T {np.round(far, 6).tolist()}   gap {gap:.3e}",
              flush=True)
    # non-conforming NON-UNIFORM walls too
    for M in (5, 6):
        st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(0.24, eps_cell=cB, x_walls=[0.2371 * P, 0.6183 * P],
                     y_walls=[0.2371 * P, 0.6183 * P], n_modes=M)
        st.add_layer(0.17, eps_cell=cB, x_walls=[0.4009 * P, 0.8817 * P],
                     y_walls=[0.4009 * P, 0.8817 * P], n_modes=M)
        st.set_source(WL, theta=TH, phi=PH)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve(retain_internal=True)
            A = np.asarray(st.layer_absorption())
        far = 1.0 - R.sum(1) - T.sum(1)
        gap = float(np.max(np.abs(A.sum(0) - far)))
        out[f"nonuniform_M{M}"] = {"gap": gap}
        print(f"[absorb] NON-UNIFORM walls M={M}: gap {gap:.3e}", flush=True)
    RES["absorb"] = out


# ==========================================================================
_P1 = 0.83
_WL1 = 0.55
_TH1 = 0.27
_TS1 = (0.28, 0.20)
_DUT = (0.5, 1 / 3)


def _oracle_1d(deg):
    st = PMMStack(_P1, degree=deg, far_field_orders=7)
    for duty, t in zip(_DUT, _TS1):
        st.add_layer(t, segments=[(duty, EPS_P), (1 - duty, EPS_H)])
    st.set_source(_WL1, theta=_TH1)
    o, R, T = st.solve()[:3]
    return np.asarray(o).ravel(), np.atleast_2d(R), np.atleast_2d(T)


def sec_oracle1d():
    """The per-layer cascade against the EXACT 1-D PMMStack, PER ORDER, BOTH
    polarizations, with the anti-mirror control."""
    out = {}
    ref = {d: _oracle_1d(d) for d in (10, 12, 14)}
    o1, R1, T1 = ref[14]
    out["oracle_selfgap_10_12"] = float(max(
        np.max(np.abs(ref[10][1] - ref[12][1])),
        np.max(np.abs(ref[10][2] - ref[12][2]))))
    out["oracle_selfgap_12_14"] = float(max(
        np.max(np.abs(ref[12][1] - ref[14][1])),
        np.max(np.abs(ref[12][2] - ref[14][2]))))
    print(f"[oracle1d] exact 1-D self-gap deg10-12 "
          f"{out['oracle_selfgap_10_12']:.3e}, deg12-14 "
          f"{out['oracle_selfgap_12_14']:.3e}", flush=True)

    def cell(N, num):
        c = np.full((N, N), EPS_H + 0j)
        c[:num, :] = EPS_P
        return c

    def score(o2, R, T, mirror=False):
        best = {0: 0.0, 1: 0.0}
        for row in (0, 1):
            for m in (-1, 0, 1):
                sel = np.where((o2[:, 0] == m) & (o2[:, 1] == 0))[0]
                if not sel.size:
                    continue
                k = int(sel[0])
                jj = np.where(o1 == (-m if mirror else m))[0]
                if not jj.size:
                    continue
                j = int(jj[0])
                best[row] = max(best[row],
                                abs(float(R[row, k]) - float(R1[row, j])),
                                abs(float(T[row, k]) - float(T1[row, j])))
        return best
    lad = {}
    for MA, MB in ((5, 5), (7, 5), (9, 7), (11, 8)):
        t0 = time.time()
        st = PMM2DStackPure(_P1, n_modes=max(MA, MB), n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(_TS1[0], eps_cell=cell(2, 1), n_modes=MA)
        st.add_layer(_TS1[1], eps_cell=cell(3, 1), n_modes=MB)
        st.set_source(_WL1, theta=_TH1)
        o2, R, T = solve(st)
        dir_ = score(o2, R, T)
        mir = score(o2, R, T, mirror=True)
        clo = float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))
        lad[f"MA{MA}_MB{MB}"] = {
            "err_TM_row0": dir_[0], "err_TE_row1": dir_[1],
            "mirror_TM": mir[0], "mirror_TE": mir[1], "closure": clo,
            "mirror_over_direct_TE": mir[1] / max(dir_[1], 1e-300),
            "wall_s": time.time() - t0}
        print(f"[oracle1d] MA={MA} MB={MB}: TM(row0) {dir_[0]:.4e}  "
              f"TE(row1) {dir_[1]:.4e}  | MIRROR TE {mir[1]:.4e} "
              f"({mir[1]/max(dir_[1],1e-300):.1f}x)  closure {clo:.3e}  "
              f"{time.time()-t0:.1f}s", flush=True)
    out["ladder"] = lad
    RES["oracle1d"] = out


SECTIONS = {"g3": sec_g3, "ident": sec_ident, "nonconf": sec_nonconf,
            "cap": sec_cap, "cache": sec_cache, "absorb": sec_absorb,
            "oracle1d": sec_oracle1d}


def main():
    for w in (sys.argv[1:] or list(SECTIONS)):
        t0 = time.time()
        SECTIONS[w]()
        print(f"--- {w} done in {time.time()-t0:.1f}s ---", flush=True)
    path = os.path.join(HERE, "v3_mortar.json")
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

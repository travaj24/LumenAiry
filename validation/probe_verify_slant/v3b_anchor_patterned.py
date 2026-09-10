"""V3b -- the frame anchor on a PATTERNED layer, with a LAB-REFERENCED oracle
that can actually be refined, and the cross-engine consequence.

V3's staircase oracle inside the PURE engine cannot be refined: a nodal SEM
union grid admits only slice counts that divide the walk in grid cells, so the
finest ladder rung on a 6-cell grid is 6 slices and its own step (2.0e-01) is
the same size as the effect being measured.  The FOURIER hybrid has no
union-grid constraint -- every layer may carry its own pixel grid -- so a
staircase of 60-pixel cells at 10 / 20 / 30 slices is available there, and a
STAIRCASE HAS NO FRAME: its transmitted amplitudes are LAB-referenced by
construction.  That is the oracle.

Three questions, all measured on the zeroth-order TRANSMISSION Jones:

  1. does the HYBRID's own slanted layer agree with the HYBRID's own fine
     staircase?  (one engine, one basis -- nothing else can differ)
  2. does the PURE slanted layer agree with that staircase, i.e. is the shipped
     anchor the correction that makes a patterned slanted layer lab-referenced?
  3. the same question on a UNIFORM slab, where the oracle is exact: slanted vs
     vertical, in BOTH engines.

``P0`` is this verification's own derived factor (V3's docstring), applied from
outside the library.
"""
import numpy as np

from _lib import arm, dump, mx  # noqa: I001

from lumenairy.elements.pmm import PMM2DStackHybrid, PMM2DStackPure

WL = 0.68e-6
K0 = 2.0 * np.pi / WL
PX = PY = 1.20e-6
DEP = 1.20e-6
NSUP, NSUB = 1.0, 1.5
NXC = 6
FINE = 60
TSL = 1.0                      # a whole period over the layer
XPROF = np.array([4.0, 4.0, 2.0, 1.0, 1.0, 1.0])
MOUNTS = {"oblique25": (np.deg2rad(25.0), 0.0),
          "conical25_40": (np.deg2rad(25.0), np.deg2rad(40.0))}


def p0_factor(shx_public, shy_public, theta, phi):
    """``P0 = exp(-i k0 (kx0 Shx + ky0 Shy))`` with ``Sh = -sum slant*d``."""
    shx, shy = -shx_public, -shy_public
    kx0 = float(np.real(NSUP)) * np.sin(theta) * np.cos(phi)
    ky0 = float(np.real(NSUP)) * np.sin(theta) * np.sin(phi)
    return complex(np.exp(-1j * K0 * (kx0 * shx + ky0 * shy)))


def cell(n=NXC, two_d=True):
    c = np.ones((n, n), dtype=complex)
    rep = int(n // NXC)
    prof = np.repeat(XPROF, rep)
    if two_d:
        c[:, 0:n // 2] = prof[:, None]
    else:
        c[:, :] = prof[:, None]
    return c


def hyb_slant(sl, theta, phi, nord, two_d=True):
    st = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                          n_orders=nord)
    st.add_layer(DEP, eps_cell=cell(NXC, two_d), slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    st.solve()
    return st.jones_transmission()


def hyb_stair(K, theta, phi, nord, two_d=True, sign=+1):
    """A K-slice staircase on a FINE (60-pixel) grid -- no slant keyword."""
    c = cell(FINE, two_d)
    st = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                          n_orders=nord)
    d = DEP / K
    for k in range(K):
        sh = int(round(sign * FINE * (k + 0.5) / K))
        st.add_layer(d, eps_cell=np.roll(c, sh, axis=0))
    st.set_source(WL, theta=theta, phi=phi)
    st.solve()
    return st.jones_transmission()


def pure_slant(sl, theta, phi, M=3, nord=3, two_d=True):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=nord)
    st.add_layer(DEP, eps_cell=cell(NXC, two_d), slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    st.solve(jones=True)
    return st.jones_transmission()


def uniform_JT(engine, sl, theta, phi):
    if engine == "pure":
        st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                            n_modes=5, n_orders=3)
        st.add_layer(0.34e-6, eps=2.25 + 0j, slant=sl)
        st.set_source(WL, theta=theta, phi=phi)
        st.solve(jones=True)
    else:
        st = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                              n_orders=5)
        st.add_layer(0.34e-6, eps=2.25 + 0j, slant=sl)
        st.set_source(WL, theta=theta, phi=phi)
        st.solve()
    return st.jones_transmission()


def main():
    out = {"Q1_hybrid_self": {}, "Q2_pure_vs_stair": {}, "Q3_uniform": {}}
    NORD = 9
    for mname, (th, ph) in MOUNTS.items():
        P0 = p0_factor(TSL * DEP, 0.0, th, ph)
        stairs = {K: hyb_stair(K, th, ph, NORD) for K in (5, 10, 20, 30)}
        ladder = {K: mx(stairs[K], stairs[30]) for K in (5, 10, 20)}
        H = hyb_slant((TSL, 0.0), th, ph, NORD)
        P = pure_slant((TSL, 0.0), th, ph)
        ref = stairs[30]
        out["Q1_hybrid_self"][mname] = dict(
            raw=mx(H, ref), times_P0=mx(H * P0, ref),
            times_conjP0=mx(H * np.conj(P0), ref),
            stair_ladder={str(k): v for k, v in ladder.items()},
            abs_arg_P0=float(abs(np.angle(P0))))
        out["Q2_pure_vs_stair"][mname] = dict(
            shipped=mx(P, ref), none=mx(P * np.conj(P0), ref),
            conj=mx(P * np.conj(P0) ** 2, ref),
            pure_vs_hybrid_corrected=mx(P, H * P0))
        r1 = out["Q1_hybrid_self"][mname]
        r2 = out["Q2_pure_vs_stair"][mname]
        print(f"[Q1] {mname}: hybrid raw {r1['raw']:.3e}  xP0 "
              f"{r1['times_P0']:.3e}  xconjP0 {r1['times_conjP0']:.3e}  "
              f"(stair ladder K5/10/20 vs K30 "
              f"{ladder[5]:.2e}/{ladder[10]:.2e}/{ladder[20]:.2e}, "
              f"|arg P0| {r1['abs_arg_P0']:.3f})")
        print(f"[Q2] {mname}: pure shipped {r2['shipped']:.3e}  none "
              f"{r2['none']:.3e}  conj {r2['conj']:.3e}  | pure vs "
              f"hybrid*P0 {r2['pure_vs_hybrid_corrected']:.3e}")

    for mname, (th, ph) in MOUNTS.items():
        for eng in ("pure", "hybrid"):
            v = uniform_JT(eng, None, th, ph)
            for sname, sv in (("x35", (float(np.tan(np.deg2rad(35.0))), 0.0)),
                              ("x10", (float(np.tan(np.deg2rad(10.0))), 0.0))):
                s = uniform_JT(eng, sv, th, ph)
                P0 = p0_factor(sv[0] * 0.34e-6, sv[1] * 0.34e-6, th, ph)
                out["Q3_uniform"][f"{eng}/{sname}/{mname}"] = dict(
                    raw=mx(s, v), times_P0=mx(s * P0, v),
                    times_conjP0=mx(s * np.conj(P0), v),
                    abs_arg_P0=float(abs(np.angle(P0))))
                r = out["Q3_uniform"][f"{eng}/{sname}/{mname}"]
                print(f"[Q3] {eng}/{sname}/{mname}: as-returned {r['raw']:.3e}"
                      f"  xP0 {r['times_P0']:.3e}  xconjP0 "
                      f"{r['times_conjP0']:.3e}")

    dump("v3b_anchor_patterned", out)
    print("arm", arm())


if __name__ == "__main__":
    main()

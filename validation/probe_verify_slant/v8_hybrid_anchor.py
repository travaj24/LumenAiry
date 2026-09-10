"""V8 -- a DEFECT this verification found on the OTHER engine: the hybrid's
transmitted amplitudes on a SLANTED PATTERNED layer are FRAME-referenced.

V3b showed the hybrid's zeroth-order transmission Jones on a slanted patterned
layer sits 5.9e-01 from its OWN fine staircase of the same solid, and that
multiplying by the frame-anchor factor ``P0`` puts it at 1.4e-02 -- the
staircase's own convergence level.  This probe pins that PER ORDER, which is
the form that identifies it as the anchor and not as an accuracy gap: the
factor is ``P_m = exp(+i k0 (alpha_m . slant) d)``, a DIFFERENT number for every
order, so if applying it order by order collapses the disagreement, there is
nothing else it could be.

THE WALK IS HALF A PERIOD, and that is not cosmetic.  With a WHOLE-period walk
(``t d = px``) the factor is ``exp(2 pi i m t d / px) = 1`` for every order --
the per-order structure vanishes identically and the anchor degenerates into a
single global phase, which cannot be told apart from any other global phase.
The first run of this probe used a whole-period walk and duly reported "best
single global phase" equal to "x P_m" to every digit.  At half a period the
increment is ``pi`` per order and the two are distinguishable.

Scope of the defect, measured here:
  * ``R``, ``T`` and the REFLECTION Jones are UNAFFECTED (the factor is
    unimodular and the superstrate side is where the frame is anchored);
  * ``jones_transmission()`` and ``per_order_amplitudes('transmission')`` are
    affected on any hybrid stack containing a slanted PATTERNED layer;
  * a slanted UNIFORM hybrid layer is unaffected -- it short-circuits to the
    vertical homogeneous path and returns bit-identical amplitudes.

The PURE engine is measured on the same fixture, per order, as the control.
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
TSL = 0.5          # HALF a period over the layer -- see the note below
XPROF = np.array([4.0, 4.0, 2.0, 1.0, 1.0, 1.0])
MOUNTS = {"oblique25": (np.deg2rad(25.0), 0.0),
          "conical25_40": (np.deg2rad(25.0), np.deg2rad(40.0))}
NORD = 9


def cell(n=NXC):
    c = np.ones((n, n), dtype=complex)
    prof = np.repeat(XPROF, n // NXC)
    c[:, 0:n // 2] = prof[:, None]
    return c


def hyb(slanted, theta, phi, K=None):
    st = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                          n_orders=NORD)
    if slanted:
        st.add_layer(DEP, eps_cell=cell(NXC), slant=(TSL, 0.0))
    else:
        c = cell(FINE)
        d = DEP / K
        for k in range(K):
            sh = FINE * TSL * (k + 0.5) / K      # integer for K in {3, 5, 15}
            assert abs(sh - round(sh)) < 1e-9
            st.add_layer(d, eps_cell=np.roll(c, int(round(sh)), axis=0))
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T, J = st.solve()
    return st, np.asarray(o), R, T, J


def pure(theta, phi, M=4):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=3)
    st.add_layer(DEP, eps_cell=cell(NXC), slant=(TSL, 0.0))
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T, J = st.solve(jones=True)
    return st, np.asarray(o), R, T, J


def per_order_T(st):
    a = st.per_order_amplitudes("transmission")
    return a["orders"], a["Ex"], a["Ey"], a["kx"], a["ky"]


def align(o1, o2):
    k1 = {tuple(int(v) for v in o1[i]): i for i in range(len(o1))}
    k2 = {tuple(int(v) for v in o2[i]): i for i in range(len(o2))}
    common = sorted(set(k1) & set(k2))
    return [k1[c] for c in common], [k2[c] for c in common], common


def main():
    out = {}
    for mname, (th, ph) in MOUNTS.items():
        row = {}
        sh, oh, Rh, Th, Jh = hyb(True, th, ph)
        ss, os_, Rs, Ts, Js = hyb(False, th, ph, K=15)
        ss2, _, Rs2, Ts2, Js2 = hyb(False, th, ph, K=5)
        i1, i2, common = align(oh, os_)
        row["efficiencies_and_reflection_jones"] = dict(
            dR=mx(Rh[:, i1], Rs[:, i2]), dT=mx(Th[:, i1], Ts[:, i2]),
            dJones_reflection=mx(Jh, Js),
            staircase_own_step_dR=mx(Rs[:, i2], Rs2[:, i2]),
            staircase_own_step_dJ=mx(Js, Js2))
        oo, Exh, Eyh, kxh, kyh = per_order_T(sh)
        oo2, Exs, Eys, _kx, _ky = per_order_T(ss)
        j1, j2, _c = align(oo, oo2)
        Pm = np.exp(1j * K0 * (kxh * TSL * DEP) * 1.0)   # alpha_m . slant * d
        raw = max(mx(Exh[:, j1], Exs[:, j2]), mx(Eyh[:, j1], Eys[:, j2]))
        fixed = max(mx((Exh * Pm[None, :])[:, j1], Exs[:, j2]),
                    mx((Eyh * Pm[None, :])[:, j1], Eys[:, j2]))
        conj = max(mx((Exh * np.conj(Pm)[None, :])[:, j1], Exs[:, j2]),
                   mx((Eyh * np.conj(Pm)[None, :])[:, j1], Eys[:, j2]))
        # a GLOBAL (order-independent) phase, the alternative explanation
        best_global = min(
            max(mx(Exh[:, j1] * np.exp(1j * a), Exs[:, j2]),
                mx(Eyh[:, j1] * np.exp(1j * a), Eys[:, j2]))
            for a in np.linspace(-np.pi, np.pi, 721))
        row["hybrid_per_order_transmission"] = dict(
            raw=raw, times_Pm=fixed, times_conj_Pm=conj,
            best_single_global_phase=best_global,
            amplitude_scale=float(np.max(np.abs(Exs))))
        sp, op, Rp, Tp, Jp = pure(th, ph)
        oop_, Exp_, Eyp_, kxp, kyp = per_order_T(sp)
        k1, k2, _c2 = align(oop_, oo2)
        Pmp = np.exp(1j * K0 * (kxp * TSL * DEP))
        row["pure_per_order_transmission"] = dict(
            shipped=max(mx(Exp_[:, k1], Exs[:, k2]),
                        mx(Eyp_[:, k1], Eys[:, k2])),
            divide_out_Pm=max(mx((Exp_ * np.conj(Pmp)[None, :])[:, k1],
                                 Exs[:, k2]),
                              mx((Eyp_ * np.conj(Pmp)[None, :])[:, k1],
                                 Eys[:, k2])),
        )
        out[mname] = row
        r = row
        print(f"[{mname}] efficiencies/reflection: dR "
              f"{r['efficiencies_and_reflection_jones']['dR']:.3e} dT "
              f"{r['efficiencies_and_reflection_jones']['dT']:.3e} dJ(refl) "
              f"{r['efficiencies_and_reflection_jones']['dJones_reflection']:.3e}"
              f"  (staircase own step dR "
              f"{r['efficiencies_and_reflection_jones']['staircase_own_step_dR']:.2e}"
              f" dJ {r['efficiencies_and_reflection_jones']['staircase_own_step_dJ']:.2e})")
        print(f"[{mname}] HYBRID per-order T: raw {raw:.3e}  x P_m "
              f"{fixed:.3e}  x conj(P_m) {conj:.3e}  best single global "
              f"phase {best_global:.3e}  (|T| scale "
              f"{r['hybrid_per_order_transmission']['amplitude_scale']:.3e})")
        print(f"[{mname}] PURE per-order T: shipped "
              f"{r['pure_per_order_transmission']['shipped']:.3e}  with P_m "
              f"divided out "
              f"{r['pure_per_order_transmission']['divide_out_Pm']:.3e}")
    dump("v8_hybrid_anchor", out)
    print("arm", arm())


if __name__ == "__main__":
    main()

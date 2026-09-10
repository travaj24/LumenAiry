"""Q6 -- DURABILITY audit of ``tests/unit/test_fix_hybrid_slant_transmission_anchor.py``.

Re-measures every numeric bar in the file THROUGH THE FILE'S OWN FIXTURES (it
imports the test module, so the readings are the ones the assertions see), and
reports for each: the bar, the measured value, and the MARGIN -- how far the
reading is from failing.  Run on BOTH builds; the cross-build spread is the
other half of the durability question.

Nothing here asserts.  It prints and dumps.
"""
from __future__ import annotations

import importlib.util
import os
import time

import _lib as L
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEST = os.path.normpath(os.path.join(
    _HERE, "..", "..", "tests", "unit",
    "test_fix_hybrid_slant_transmission_anchor.py"))
_spec = importlib.util.spec_from_file_location("_anchor_tests", _TEST)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


def row(out, test, quantity, bar_text, measured, margin, note=""):
    out.append(dict(test=test, quantity=quantity, bar=bar_text,
                    measured=measured, margin=margin, note=note))
    print("%-6s %-38s %-22s meas=%-12s margin=%-9s %s"
          % (test, quantity, bar_text,
             ("%.4e" % measured) if isinstance(measured, float) else measured,
             ("%.3gx" % margin) if isinstance(margin, float) else margin,
             note))


def main():
    t0 = time.time()
    out = []
    for mount in T.MOUNTS:
        s = T._slanted(mount)
        k15, k5, k3 = T._stair(mount, 15), T._stair(mount, 5), T._stair(mount, 3)
        shipped = T._dT(s, k15)
        none = T._dT(s, k15, 0.0)
        conj = T._dT(s, k15, -T.WALK)
        ostep = T._stair_step(mount, coarse=3, fine=15)
        row(out, "a1", "shipped vs oracle K3->K15 step [%s]" % mount,
            "shipped < step", shipped, ostep / shipped,
            "step=%.4e" % ostep)
        row(out, "a1", "none/shipped [%s]" % mount, "> 20x",
            none / shipped, (none / shipped) / 20.0)
        row(out, "a1", "conj/shipped [%s]" % mount, "> 20x",
            conj / shipped, (conj / shipped) / 20.0)
        row(out, "a1", "conj/none [%s]" % mount, "conj > none (1x)",
            conj / none, conj / none,
            "SUB-DECADE ORDERING")

        ship = [T._dT(s, T._stair(mount, K)) for K in (3, 5, 15)]
        nn = [T._dT(s, T._stair(mount, K), 0.0) for K in (3, 5, 15)]
        row(out, "a2", "shipped first/last [%s]" % mount, "> 2x",
            ship[0] / ship[-1], (ship[0] / ship[-1]) / 2.0,
            "ladder %s" % ["%.4e" % v for v in ship])
        row(out, "a2", "none first/last [%s]" % mount, "in (0.9, 1.1)",
            nn[0] / nn[-1],
            min(nn[0] / nn[-1] - 0.9, 1.1 - nn[0] / nn[-1]) / 0.1,
            "ladder %s" % ["%.4e" % v for v in nn])

        Ex, Ey, o1 = T._rephase(s, 0.0)
        B = k15.per_order_amplitudes("transmission")
        j1, j2 = T._align(o1, B["orders"])
        best = min(max(float(np.max(np.abs(Ex[:, j1] * np.exp(1j * a)
                                           - B["Ex"][:, j2]))),
                       float(np.max(np.abs(Ey[:, j1] * np.exp(1j * a)
                                           - B["Ey"][:, j2]))))
                   for a in np.linspace(-np.pi, np.pi, 721))
        row(out, "a3", "best global / shipped [%s]" % mount, "> 10x",
            best / shipped, (best / shipped) / 10.0)

        P0 = T._P0(s)
        j_ship = float(np.max(np.abs(s.jones_transmission()
                                     - k15.jones_transmission())))
        j_none = float(np.max(np.abs(s.jones_transmission() * np.conj(P0)
                                     - k15.jones_transmission())))
        row(out, "a4", "||P0|-1| [%s]" % mount, "< 1e-14",
            abs(abs(P0) - 1.0), 1e-14 / max(abs(abs(P0) - 1.0), 1e-300))
        row(out, "a4", "|arg P0| [%s]" % mount, "> 1.0 rad",
            abs(np.angle(P0)), abs(np.angle(P0)) / 1.0, "SUB-DECADE")
        row(out, "a4", "jones none/shipped [%s]" % mount, "> 20x",
            j_none / j_ship, (j_none / j_ship) / 20.0)

        oA, RA, TA, JA = s._RTJ
        oB, RB, TB, JB = k15._RTJ
        oC, RC, TC, JC = k5._RTJ
        i1, i2 = T._align(oA, oB)
        _x, i3 = T._align(oA, oC)
        step_R = float(np.max(np.abs(RB[:, i2] - RC[:, i3])))
        step_J = float(np.max(np.abs(JB - JC)))
        dR = float(np.max(np.abs(RA[:, i1] - RB[:, i2])))
        dJ = float(np.max(np.abs(JA - JB)))
        row(out, "c2", "dR vs 2*step_R [%s]" % mount, "< 2 step_R",
            dR, 2.0 * step_R / dR, "step_R=%.4e" % step_R)
        row(out, "c2", "dJones(refl) vs step_J [%s]" % mount, "< step_J",
            dJ, step_J / dJ, "step_J=%.4e" % step_J)

        # d2 -- the round-trip row, on the FILM stack
        sf = T._slanted(mount, film=True)
        kf15, kf5 = T._stair(mount, 15, film=True), T._stair(mount, 5, film=True)
        JAf = sf._RTJ[3]
        JBf = kf15._RTJ[3]
        JCf = kf5._RTJ[3]
        stepJf = float(np.max(np.abs(JBf - JCf)))
        P0f = T._P0(sf)
        asret = float(np.max(np.abs(JAf - JBf)))
        xp0 = float(np.max(np.abs(JAf * P0f - JBf)))
        xp02 = float(np.max(np.abs(JAf * P0f * P0f - JBf)))
        row(out, "d2", "refl as returned vs step_J [%s]" % mount, "< step_J",
            asret, stepJf / asret, "step_J=%.4e" % stepJf)
        row(out, "d2", "refl x P0 vs 10 step_J [%s]" % mount, "> 10 step_J",
            xp0, (xp0 / stepJf) / 10.0,
            "xP0/step = %.2fx, xP0/asret = %.2fx"
            % (xp0 / stepJf, xp0 / asret))
        row(out, "d2", "refl x P0^2 vs 10 step_J [%s]" % mount, "> 10 step_J",
            xp02, (xp02 / stepJf) / 10.0,
            "xP0^2/step = %.2fx, xP0^2/asret = %.2fx"
            % (xp02 / stepJf, xp02 / asret))
        row(out, "d2", "T none/shipped, film [%s]" % mount, "> 5x",
            T._dT(sf, kf15, 0.0) / T._dT(sf, kf15),
            (T._dT(sf, kf15, 0.0) / T._dT(sf, kf15)) / 5.0)

        # d1 -- the two-half stack
        th, ph = T.MOUNTS[mount]
        c2 = T._st()
        c2.add_layer(T.DEP / 2, eps_cell=T._cell(), slant=(T.TSL, 0.0))
        c2.add_layer(T.DEP / 2, eps_cell=T._cell(), slant=(T.TSL, 0.0))
        c2.set_source(T.WL, theta=th, phi=ph)
        c2.solve()
        full = T._dT(c2, k15)
        row(out, "d1", "half-sum/full [%s]" % mount, "> 20x",
            T._dT(c2, k15, T.WALK / 2) / full,
            (T._dT(c2, k15, T.WALK / 2) / full) / 20.0)
        row(out, "d1", "no-sum/full [%s]" % mount, "> 20x",
            T._dT(c2, k15, 0.0) / full,
            (T._dT(c2, k15, 0.0) / full) / 20.0)

        # a5 closure
        for tag, st in (("slanted", s), ("stair15", k15), ("film", sf)):
            tot = float(np.max(st._RTJ[1].sum(axis=1) + st._RTJ[2].sum(axis=1)))
            row(out, "a5", "closure %s [%s]" % (tag, mount),
                "1.0 <= x < 1.05", tot,
                min(tot - 1.0, 1.05 - tot) / 0.05,
                "LOWER margin %.2e" % (tot - 1.0))
        del k3

    # a6
    s = T._slanted("conical25_40")
    a = s.per_order_amplitudes("transmission")
    P = np.exp(1j * T.K0 * (a["kx"] * T.WALK))
    row(out, "a6", "max||P|-1|", "< 1e-13",
        float(np.max(np.abs(np.abs(P) - 1.0))),
        1e-13 / max(float(np.max(np.abs(np.abs(P) - 1.0))), 1e-300))
    row(out, "a6", "ptp(arg P)", "> 3.0 rad",
        float(np.ptp(np.angle(P))), float(np.ptp(np.angle(P))) / 3.0,
        "SUB-DECADE")

    # c3 -- the cost of anchoring a slanted UNIFORM film
    th, ph = T.MOUNTS["oblique25"]
    u0, u1 = T._st(), T._st()
    u0.add_layer(T.DEP, eps=2.25)
    u1.add_layer(T.DEP, eps=2.25, slant=(T.TSL, 0.0))
    for st in (u0, u1):
        st.set_source(T.WL, theta=th, phi=ph)
        st.solve()
    ifa = float(np.max(np.abs(u1.jones_transmission() * T._P0(u1)
                              - u0.jones_transmission())))
    row(out, "c3", "cost of anchoring a uniform film", "> 1.0", ifa, ifa,
        "SUB-DECADE; |J| scale %.4e"
        % float(np.max(np.abs(u0.jones_transmission()))))

    # b1 -- the cross-engine row
    for mount in T.MOUNTS:
        s = T._slanted(mount)
        th, ph = T.MOUNTS[mount]
        sp = T.PMM2DStackPure(T.PX, T.PY, n_superstrate=T.NSUP,
                              n_substrate=T.NSUB, n_modes=4, n_orders=3)
        sp.add_layer(T.DEP, eps_cell=T._cell(), slant=(T.TSL, 0.0))
        sp.set_source(T.WL, theta=th, phi=ph)
        sp.solve(jones=True)
        Ap = sp.per_order_amplitudes("transmission")
        Ah = s.per_order_amplitudes("transmission")
        m1, m2 = T._align(Ap["orders"], Ah["orders"])
        Ex0, Ey0, _o = T._rephase(s, 0.0)
        sh = max(float(np.max(np.abs(Ap["Ex"][:, m1] - Ah["Ex"][:, m2]))),
                 float(np.max(np.abs(Ap["Ey"][:, m1] - Ah["Ey"][:, m2]))))
        pf = max(float(np.max(np.abs(Ap["Ex"][:, m1] - Ex0[:, m2]))),
                 float(np.max(np.abs(Ap["Ey"][:, m1] - Ey0[:, m2]))))
        A7 = T._slanted(mount, nord=7).per_order_amplitudes("transmission")
        n1, n2 = T._align(Ah["orders"], A7["orders"])
        own = max(float(np.max(np.abs(Ah["Ex"][:, n1] - A7["Ex"][:, n2]))),
                  float(np.max(np.abs(Ah["Ey"][:, n1] - A7["Ey"][:, n2]))))
        row(out, "b1", "cross-engine / own n_orders step [%s]" % mount,
            "< 10x", sh / own, 10.0 / (sh / own), "own_step=%.4e" % own)
        row(out, "b1", "pre-fix/shipped [%s]" % mount, "> 20x",
            pf / sh, (pf / sh) / 20.0)

    payload = dict(rows=out, seconds=round(time.time() - t0, 1))
    L.dump("q6_durability", payload)


if __name__ == "__main__":
    main()

"""The unit-test READINGS of ``tests/unit/test_pmm2d_staggered_curved_e1.py``:
every quantity a test asserts on, measured through the test module's OWN
helpers at the test's own sizes (so the bars cite numbers, not memory).

usage: python e_unit_readings.py            -> e_unit_readings.json
"""
import os
import sys

import _common as C  # noqa: F401 -- pins lumenairy to this tree
import _e1common as E
import numpy as np

sys.path.insert(0, os.path.join(C.ROOT, "tests", "unit"))
import test_pmm2d_staggered_curved_e1 as T  # noqa: E402


class _MP:
    """A minimal monkeypatch for the test's patch helpers."""

    def __init__(self):
        self.undo_list = []

    def setattr(self, obj, name, val):
        self.undo_list.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def undo(self):
        for obj, name, val in reversed(self.undo_list):
            setattr(obj, name, val)
        self.undo_list = []


def main():
    out = {}
    TS = T.TS
    # E1-2
    for sl in (None, (0.25, -0.1)):
        s0, s1 = T._identity_arms(sl, theta=0.3, phi=0.4)
        m0, m1 = TS._region_modes_oop(s0), TS._region_modes_oop(s1)
        d = np.abs(m0[2][:, None] - m1[2][None, :]).min(axis=1).max()
        _s0, sm = T._identity_arms(sl, move=1e-6)
        out[f"e1_2_{sl}"] = {"A": T._rel(s1.Agen, s0.Agen),
                             "B": T._rel(s1.Bgen, s0.Bgen),
                             "lam": float(d / np.abs(m0[2]).max()),
                             "moved_1e-6": T._rel(sm.Agen, _s0.Agen),
                             "full_RTJrJt": T._identity_full(sl)}
        print(out[f"e1_2_{sl}"], flush=True)
    sh = T._shear(T._SLAB["P"])
    out["e1_3_shear"] = {"normal_M4": T._slab(T._NONREC, sh, 4),
                         "conical_M5": T._slab(T._NONREC, sh, 5, *T._CON)}
    g = {}
    with T._patched(TS, "_OOP_ROT_SIGN", 1.0):
        g["rot+1_conical_M5"] = T._slab(T._NONREC, sh, 5, *T._CON)
        g["rot+1_normal_M4"] = T._slab(T._NONREC, sh, 4)
    with T._patched(TS, "_OOP_H_GAUGE", 1j):
        g["h+1j_normal_M4"] = T._slab(T._NONREC, sh, 4)
    with T._patched(TS, "_OOP_H_GAUGE", 1.0):
        g["h+1_normal_M4"] = T._slab(T._NONREC, sh, 4)
    out["e1_3_gauge"] = g
    mp = _MP()
    T._no_mu_blocks(mp)
    out["e1_3_mu"] = {"shear_M4": T._slab(T._NONREC, sh, 4),
                      "identity_M4": T._slab(T._NONREC,
                                             T._ident(T._SLAB["P"]), 4)}
    mp.undo()
    orig = TS._stag_map_eff_tensor

    def f(eps, mu, xu, xv, yu, yv, **kw):
        o = orig(eps, mu, xu, xv, yu, yv, **kw)
        if kw.get("oop"):
            o["c33"] = o["c33"] * (xu * yv - xv * yu)
        return o
    mp.setattr(TS, "_stag_map_eff_tensor", f)
    out["e1_3_chi33"] = {"oblique_M5": T._slab(T._NONREC, sh, 5, *T._OBL),
                         "normal_M5": T._slab(T._NONREC, sh, 5)}
    mp.undo()
    cm = T._stretch(T._SLAB["P"], 0.15)
    out["e1_3_stretch"] = {"M4": T._slab(T._NONREC, cm, 4),
                           "M5": T._slab(T._NONREC, cm, 5)}
    m3 = {}
    for mu, mm in (("gyro", T._MU_GYRO), ("lossy", T._MU_LOSSY)):
        ref = T._mu_berreman(T._NONREC, mm, *T._CON)
        m3[f"{mu}_none_M6"] = T._slab(T._NONREC, None, 6, *T._CON, mu=mm,
                                      oracle=ref)
    ref = T._mu_berreman(T._NONREC, T._MU_GYRO, *T._CON)
    m3["gyro_shear_M5"] = T._slab(T._NONREC, sh, 5, *T._CON, mu=T._MU_GYRO,
                                  oracle=ref)
    out["e1_3m"] = m3
    cc = T._circle(T._SLAB["P"])
    out["e1_4"] = {"M4": T._slab(T._NONREC, cc, 4),
                   "M6": T._slab(T._NONREC, cc, 6)}
    out["e1_6_slab"] = {
        "normal_M5": T._slab(T._NONREC, sh, 5, slant=(0.2, -0.1)),
        "oblique_M5": T._slab(T._NONREC, sh, 5, *T._OBL, slant=(0.2, -0.1))}
    T._slant_weights_patch(mp, "tau_unmapped")
    a = T._slab(T._NONREC, sh, 5, slant=(0.2, -0.1))
    mp.undo()
    T._slant_weights_patch(mp, "no_slant_blocks")
    b = T._slab(T._NONREC, sh, 5, *T._OBL, slant=(0.2, -0.1))
    c = T._slab(T._NONREC, sh, 5, slant=(0.2, -0.1))
    mp.undo()
    out["e1_6_arms"] = {"tau_unmapped_normal_M5": a,
                        "no_slant_blocks_oblique_M5": b,
                        "no_slant_blocks_normal_M5": c}
    # E1-5 / E1-6 pillars
    ref5 = T._saved_vec("e5_curved_oop30_c3_t0.0000_p0.0000_M10.json")
    _st, o, R, Tt = T._pillar("c5", T._OOP30, 4)
    ref6 = T._saved_vec("e6_curved_eps4_c3_t0.0000_p0.0000_M10.json")
    _st, o6, R6, T6 = T._pillar("c5", 4.0 * T._EYE3, 4, slant=(0.2, 0.0))
    out["e1_5_6_pillars"] = {
        "oop30_c5_M4_to_c3_M10": float(np.abs(T._vec(o, R, Tt)
                                              - ref5).max()),
        "slant_c5_M4_to_c3_M10": float(np.abs(T._vec(o6, R6, T6)
                                              - ref6).max())}
    th, ph = T._CON
    tr, pr = T._reverse(th, ph, -1, 0)
    f_ = T._sv(T._OOP30, 5, th, ph, (-1, 0))
    r_ = T._sv(T._OOP30, 5, tr, pr, (-1, 0))
    w_ = T._sv(T._OOP30, 5, tr, pr, (0, 0))
    fn = T._sv(T._NONREC30, 5, th, ph, (-1, 0))
    rn = T._sv(T._NONREC30, 5, tr, pr, (-1, 0))
    rT = T._sv(T._NONREC30.T.copy(), 5, tr, pr, (-1, 0))
    out["e1_7"] = {"reciprocal": float(np.abs(f_ - r_).max()),
                   "wrong_pairing": float(np.abs(f_ - w_).max()),
                   "nonreciprocal": float(np.abs(fn - rn).max()),
                   "transposed_partner": float(np.abs(fn - rT).max())}
    print(out, flush=True)
    E.dump("e_unit_readings.json", out)


if __name__ == "__main__":
    main()

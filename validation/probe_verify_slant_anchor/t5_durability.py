"""TASK 5 -- DURABILITY: re-measure, on both builds, every quantity a bar in
``tests/unit/test_fix_slant_anchor_v1_v2_o2.py`` compares against.

This imports the gate module itself and calls ITS helpers, so the numbers
below are the ones the assertions actually read -- not a re-derivation that
might drift from them.  Each row reports the reading and the MARGIN (the
factor by which the reading clears the bar); a margin near 1 is a bar whose
pass/fail boundary sits inside the cross-build spread of what it reads.
"""
from __future__ import annotations

import math
import os
import sys
import time
import warnings

if os.environ.get("LUM_ARM_TREE"):
    sys.path.insert(0, os.environ["LUM_ARM_TREE"])
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

import _lib  # noqa: E402
import numpy as np  # noqa: E402


def gate():
    tree = os.environ.get("LUM_ARM_TREE") or os.getcwd()
    sys.path.insert(0, os.path.join(tree))
    import importlib.util
    p = os.path.join(tree, "tests", "unit",
                     "test_fix_slant_anchor_v1_v2_o2.py")
    spec = importlib.util.spec_from_file_location("_gate", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():                                                  # noqa: C901
    g = gate()
    out, t0 = {}, time.time()
    R = {}

    def rec(name, reading, bar, sense, note=""):
        """``sense`` is '<' when the assertion is ``reading < bar``."""
        if reading is None or bar in (None, 0):
            m = None
        elif sense == "<":
            m = bar / reading if reading else float("inf")
        else:
            m = reading / bar if bar else float("inf")
        R[name] = dict(reading=reading, bar=bar, sense=sense,
                       margin_x=m, note=note)
        print(f"  {name:52s} {reading!r:>26} {sense} {bar!r:<10} "
              f"margin {m if m is None else round(m, 4)}  {note}")

    # ---- V1 -----------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vert = g._v1_call(slant=None)
        slan = g._v1_call()
        gap = max(g._dmax(vert[i], slan[i]) for i in (1, 2, 3))
    rec("t1_v1_defect_gap", gap, 1e-3, ">", "max over dR/dT/dJones")

    jax, jnp = g._jnp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = g.pmm_jones_2d(g.V1_PX, g.V1_PY, g._V1_CELL, g.V1_NSUB,
                             g.V1_NSUP, jnp.asarray(g.V1_DEPTH), g.V1_WL,
                             theta=g.V1_TH, phi=0.0, n_orders=g.V1_NORD,
                             degree=g.V1_DEG, slant=None,
                             region_layout=g._V1_LAYOUT)
    for i, nm in ((1, "R"), (2, "T"), (3, "J")):
        rec(f"t3_v1_vertical_control_d{nm}", g._dmax(got[i], vert[i]),
            1e-11, "<")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_sl = g._v1_call(cell=g._V1_CONST)
        c_vt = g._v1_call(cell=g._V1_CONST, slant=None)
    for i, nm in ((1, "R"), (2, "T"), (3, "J")):
        rec(f"t4_v1_const_noop_d{nm}", g._dmax(c_sl[i], c_vt[i]), 1e-11, "<")

    def f(d):
        return jnp.sum(g.pmm_jones_2d(
            g.V1_PX, g.V1_PY, g._V1_CELL, g.V1_NSUB, g.V1_NSUP, d, g.V1_WL,
            theta=g.V1_TH, phi=0.0, n_orders=g.V1_NORD, degree=g.V1_DEG,
            slant=None, region_layout=g._V1_LAYOUT)[2])
    h = 1e-11
    d0 = jnp.asarray(g.V1_DEPTH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gr = float(jax.grad(f)(d0))
        f0 = float(f(d0))
        fd = float((f(d0 + h) - f(d0 - h)) / (2.0 * h))
    rec("t5_v1_ad_vs_fd_rel", abs(gr - fd) / abs(fd), 1e-5, "<",
        f"|g|={abs(gr):.4g} fd_floor~{np.finfo(float).eps*abs(f0)/h:.3g}")
    rec("t5_v1_grad_magnitude", abs(gr), 1.0, ">")

    # ---- V2 -----------------------------------------------------------
    phi = float(math.atan(g.V2_W / g.V2_D))
    one = g._slant_frame_walk_1d([(g.V2_D, [], phi)])
    two = g._slant_frame_walk_1d([(g.V2_D, [], phi), (0.5 * g.V2_D, [], phi)])
    rec("t6_v2_walk_one_abs_err", abs(one - g.V2_W), 1e-18, "<",
        f"W={g.V2_W!r} one={one!r}")
    rec("t6_v2_walk_two_abs_err", abs(two - 1.5 * g.V2_W), 1e-18, "<")
    canc = g._slant_frame_walk_1d([(g.V2_D, [], phi), (g.V2_D, [], -phi)])
    R["t6_v2_opposite_cancels_exactly"] = dict(
        reading=canc, bar=0.0, sense="==",
        margin_x=None, note=f"exact float equality; tan(-phi)+tan(phi)="
                            f"{np.tan(-phi) + np.tan(phi)!r}")
    print(f"  t6_v2_opposite_cancels_exactly  {canc!r}  "
          f"is_exact_zero={canc == 0.0}")

    sh, osh = g._v2_sheared()
    _o, _A, kx = g._amps(sh)
    ph = g._P(kx, g.V2_W)
    rec("t7_v2_unimodular_dev", float(np.max(np.abs(np.abs(ph) - 1.0))),
        1e-14, "<")

    lad = g._v2_ladder()
    rec("t9_v2_converge_J_ratio", lad[4]["J"] / lad[12]["J"], 3.0, ">")
    rec("t9_v2_converge_amps_ratio", lad[4]["amps"] / lad[12]["amps"],
        3.0, ">")
    rec("t9_v2_J12_vs_2xstep", lad[12]["J"] / (2.0 * lad[12]["step"]),
        1.0, "<", f"J12={lad[12]['J']:.4e} step={lad[12]['step']:.4e}")
    rec("t9_v2_amps12_vs_2xstep",
        lad[12]["amps"] / (2.0 * lad[12]["step"]), 1.0, "<")
    fl_J = lad[4]["J_none"] / lad[12]["J_none"]
    fl_A = lad[4]["amps_none"] / lad[12]["amps_none"]
    R["t10_v2_flatness_J"] = dict(reading=fl_J, bar=[0.8, 1.25],
                                  sense="in", margin_x=None)
    R["t10_v2_flatness_amps"] = dict(reading=fl_A, bar=[0.8, 1.25],
                                     sense="in", margin_x=None)
    print(f"  t10_v2_flatness_J {fl_J:.6f}   amps {fl_A:.6f}  (0.8, 1.25)")
    rec("t10_v2_none_over_shipped_J", lad[12]["J_none"] / lad[12]["J"],
        20.0, ">")
    rec("t10_v2_none_over_shipped_amps",
        lad[12]["amps_none"] / lad[12]["amps"], 20.0, ">")
    rec("t10_v2_conj_over_shipped_J", lad[12]["J_conj"] / lad[12]["J"],
        20.0, ">")
    rec("t10_v2_conj_over_none_J", lad[12]["J_conj"] / lad[12]["J_none"],
        1.3, ">")
    rec("t11_v2_refl_ratio", lad[4]["refl"] / lad[12]["refl"], 2.0, ">")
    rec("t11_v2_refl12_vs_2xstep",
        lad[12]["refl"] / (2.0 * lad[12]["step"]), 1.0, "<")

    # the uniform-film analytic oracle
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stu = g._v2_stack()
        stu.add_layer(g.V2_D, eps=2.60,
                      slant_angle=float(math.atan(g.V2_W / g.V2_D)))
        stu, ou = g._v2_solve(stu)
        stv = g.PMMStack(g.V2_P, n_superstrate=1.0, n_substrate=1.6,
                         degree=g.V2_DEG, n_orders=g.V2_NORD)
        stv.add_layer(g.V2_D, eps=2.60)
        stv, ov = g._v2_solve(stv)
    Ju, Jv = np.asarray(stu.jones_transmission()), \
        np.asarray(stv.jones_transmission())
    o_u, A_u, kxu = g._amps(stu)
    o_v, A_v, _ = g._amps(stv)
    shipped = g._resid(Ju, Jv)
    unanch = g._resid(Ju / g._P(kxu, g.V2_W)[
        int(np.where(o_u == 0)[0][0])], Jv)
    rec("t8_v2_uniform_film_shipped", shipped, 1e-11, "<")
    rec("t8_v2_uniform_film_amps", g._align(o_u, A_u, o_v, A_v), 1e-11, "<")
    rec("t8_v2_uniform_film_ratio", unanch / max(shipped, 1e-300), 1e6, ">")

    # ---- O2 -----------------------------------------------------------
    _rc = g._rc
    rows = []
    _rc._INV_CENSUS = rows
    try:
        try:
            g._o2_hybrid(M=5, slant=0.5).solve()
            brk_note = "DID NOT RAISE"
        except Exception as exc:                            # noqa: BLE001
            brk_note = type(exc).__name__
        broken = [r for r in rows if r[4]]
        rec("t20_o2_broken_rcond_max",
            (max(r[2] for r in broken) if broken else None), 1e-15, "<",
            brk_note)
        rec("t20_o2_broken_resid_min",
            (min(r[3] for r in broken) if broken else None), 1e-3, ">")
        rows2 = []
        _rc._INV_CENSUS = rows2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g._o2_hybrid(M=5, degree=7).solve()
            g._o2_hybrid(M=5, slant=0.25).solve()
            g._o2_hybrid(M=5, theta=0.0).solve()
        healthy = [r for r in rows2
                   if r[0] == "rcwa generalized interface (T22)"]
        rec("t20_o2_healthy_rcond_min",
            (min(r[2] for r in healthy) if healthy else None), 1e-5, ">",
            f"n_healthy_rows={len(healthy)} (bar >= 4)")
        rec("t20_o2_n_healthy_rows", float(len(healthy)), 4.0, ">")
    finally:
        _rc._INV_CENSUS = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cures = {k: g._rt(v.solve()) for k, v in (
            ("degree7", g._o2_hybrid(M=5, degree=7)),
            ("slant0.25", g._o2_hybrid(M=5, slant=0.25)),
            ("M9", g._o2_hybrid(M=9)))}
    for k, v in cures.items():
        rec(f"t18_o2_cure_{k}_sumRT", v, 1.10, "<")
    ben = {}
    for theta in (0.0, 40.0):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ben[f"theta{theta:g}"] = g._rt(
                g._o2_hybrid(M=5, theta=theta).solve())
            ben[f"theta{theta:g}_warned"] = any(
                "energy not conserved" in str(x.message) for x in w)
    R["t19_o2_benign_rows"] = dict(reading=ben, bar=[1.0, 1.10],
                                   sense="in", margin_x=None)
    print(f"  t19_o2_benign_rows {ben}")

    out["rows"] = R
    out["total_secs"] = round(time.time() - t0, 1)
    _lib.save("t5_durability", out)


if __name__ == "__main__":
    main()

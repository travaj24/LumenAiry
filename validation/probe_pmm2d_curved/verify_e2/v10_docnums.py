"""E2 verifier item 10 -- every number in the BUILD_E2 doc's tables against
the build JSON in validation/probe_pmm2d_curved/build_e2/.

Each row: (doc section, what, the number AS PRINTED in the doc, json file,
extractor).  The JSON value is rounded to the doc's printed significant
digits and compared; MATCH / MISMATCH is printed and written to
v10_docnums.json.  No lumenairy import (pure bookkeeping).
"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, "..", "build_e2")
_cache = {}


def J(fn):
    if fn not in _cache:
        with open(os.path.join(B, fn)) as f:
            _cache[fn] = json.load(f)
    return _cache[fn]


def g(fn, *path):
    v = J(fn)
    for p in path:
        v = v[p]
    return v


def mx(fn, *path):
    return float(np.max(np.abs(np.asarray(g(fn, *path), dtype=float))))


def rt(fn, a, b):
    d = J(fn)
    return float(max(np.abs(np.asarray(d[a]["R"]) - np.asarray(d[b]["R"])).max(),
                     np.abs(np.asarray(d[a]["T"]) - np.asarray(d[b]["T"])).max()))


def lad(fn, case, n):
    for r in g(fn, case, "ladder"):
        if r["n"] == n:
            return r["rel"]
    raise KeyError(n)


def lst(fn, key, n, field):
    for r in g(fn, key):
        if r.get("n") == n and field in r:
            return r[field]
    raise KeyError((key, n, field))


def same(doc, val):
    """doc string vs value at the doc's printed significant digits."""
    s = doc.strip()
    if s in ("0", "0.0"):
        return float(val) == 0.0
    if s.lower() in ("true", "false"):
        return str(bool(val)).lower() == s.lower()
    m = re.fullmatch(r"([-+]?\d+(?:\.(\d+))?)(?:e([-+]?\d+))?", s)
    if not m:
        raise ValueError(s)
    mant = m.group(1)
    digits = len(mant.replace("-", "").replace(".", "").lstrip("0")) or 1
    if m.group(3) is None:                 # plain decimal: compare decimals
        dec = len(m.group(2) or "")
        return round(float(val), dec) == float(s)
    return float(f"{float(val):.{digits - 1}e}") == float(s)


ROWS = []


def row(sec, what, doc, fn, fx):
    ROWS.append((sec, what, doc, fn, fx))


# ---- 2.2 the design measurement (e2_d_design_M4.json) ---------------------
D = "e2_d_design_M4.json"
row("2.2", "Newton per node circle (us)", "31.5", D,
    lambda: g(D, "inversion", "circle_us_per_node"))
row("2.2", "Newton per node sinusoid (us)", "21.7", D,
    lambda: g(D, "inversion", "sinusoid_us_per_node"))
row("2.2", "max inversion error circle", "9.8e-15", D,
    lambda: g(D, "inversion", "circle_maxerr"))
row("2.2", "max inversion error sinusoid", "1.1e-15", D,
    lambda: g(D, "inversion", "sinusoid_maxerr"))
row("2.2", "Gordon-Hall us/point", "0.36", D,
    lambda: g(D, "inversion", "circle_geom_us_per_point"))
row("2.2", "cond(X) curved", "769", D,
    lambda: g(D, "conditioning", "curved_cond"))
row("2.2", "cond separable", "829", D,
    lambda: g(D, "conditioning", "separable_cond"))
row("2.2", "cond plain Gram", "878", D,
    lambda: g(D, "conditioning", "gram_b_cond"))
for n, doc in ((8, "4.2e-2"), (16, "1.4e-2"), (32, "8.1e-3"), (64, "8.6e-4"),
               (128, "3.2e-4"), (256, "2.0e-4")):
    row("2.2", f"composite rel n={n}", doc, D, lambda n=n: next(
        r["rel"] for r in g(D, "composite_vs_cut", "composite") if r["n"] == n))
row("2.2", "composite wall n=256 (s)", "31", D, lambda: next(
    r["t"] for r in g(D, "composite_vs_cut", "composite") if r["n"] == 256))
for n, doc in ((6, "1.4e-6"), (9, "3.2e-10"), (12, "5.5e-14"),
               (18, "6.8e-15"), (27, "1.0e-15")):
    row("2.2", f"cut rel n={n}", doc, D, lambda n=n: next(
        r["rel"] for r in g(D, "composite_vs_cut", "cut") if r["n"] == n))
row("2.2", "composite in stack (nocut d_RT, M5)", "6.8e-5",
    "e2_8_mutations_M5.json",
    lambda: g("e2_8_mutations_M5.json", "nocut", "d_RT"))
row("2.2", "cross-mass at M=8 (s)", "1.1", "e2_9_cost_M8.json",
    lambda: g("e2_9_cost_M8.json", "cross_mass_wall"))
row("2.2", "per-layer solve at M=8 (s)", "48", "e2_9_cost_M8.json",
    lambda: g("e2_9_cost_M8.json", "perlayer_wall"))

# ---- 2.3 kernel checks ------------------------------------------------------
E0 = "e0_smoke_M4.json"
row("2.3", "same map vs plain Gram", "2.5e-15", E0,
    lambda: g(E0, "same_circle_tau1.0", "rel"))
row("2.3", "identity maps vs StagCrossOps", "7.3e-16", E0,
    lambda: g(E0, "unmapped_walls_tau1.0", "rel"))
E0B = "e0b_consistency_M4.json"
row("2.3", "one integral two coords n=24", "7.2e-16", E0B,
    lambda: lst(E0B, "sinx_siny", 24, "a_vs_b"))
X = "e2_x_separable_M5.json"
row("2.3", "separable X11", "1.3e-14", X, lambda: g(X, "X11"))
row("2.3", "separable X22", "5.4e-15", X, lambda: g(X, "X22"))
row("2.3", "X12 = X21 = 0", "0", X, lambda: max(g(X, "X12"), g(X, "X21")))
row("2.3", "without psi'", "0.42", X, lambda: g(X, "X11_without_psi_prime"))
row("2.3", "circle/sin in sinusoid coords n=16", "6.9e-4", E0B,
    lambda: lst(E0B, "circle_sinx", 16, "sinus_coords_vs_ref"))
row("2.3", "circle/sin in sinusoid coords n=32", "1.8e-4", E0B,
    lambda: lst(E0B, "circle_sinx", 32, "sinus_coords_vs_ref"))

# ---- 2.4 q-matching ---------------------------------------------------------
for ms, doc in (("", "1.0e-6"), ("_Ms6-8", "1.0e-6"), ("_Ms8-6", "3.9e-9"),
                ("_Ms8-8", "2.2e-10")):
    fn = f"e2_h_host_sin_circ_vac_M6{ms}_th0.0_ph0.0.json"
    row("2.4", f"q-match {ms or '(6,6)'}", doc, fn,
        lambda fn=fn: g(fn, "err_vs_airy"))
row("2.4", "single map circle M6 (1e-11 class)", "2.3e-11",
    "e2_h_host_shared_circ_M6_th0.0_ph0.0.json",
    lambda: g("e2_h_host_shared_circ_M6_th0.0_ph0.0.json", "err_vs_airy"))
row("2.4", "single map sinx M6 (claimed 1e-11..1e-13 by M=6/7)", "5.3e-9",
    "e2_h_host_shared_sinx_M6_th0.0_ph0.0.json",
    lambda: g("e2_h_host_shared_sinx_M6_th0.0_ph0.0.json", "err_vs_airy"))

# ---- 2.5 riding -------------------------------------------------------------
for M, doc in ((4, "2.1e-3"), (5, "3.4e-3"), (6, "2.7e-3")):
    fn = f"e2_4_vacuum_M{M}.json"
    row("2.5", f"noride vs alone M{M}", doc, fn,
        lambda fn=fn: g(fn, "noride_vs_alone"))

# ---- 2.6 over-refusal -------------------------------------------------------
for case, docs in (("two_circles_r_0.30_0.40", ("3.7e-2", "1.1e-2", "1.6e-2")),
                   ("equal_circles_dy_0.05", ("7.5e-2", "1.9e-2", "9.6e-3")),
                   ("rect_touching_circle_30deg", (None, "1.5e-2", "1.6e-3"))):
    for M, doc in zip((3, 4, 5), docs):
        fn = f"e2_m_overrefusal_M{M}.json"
        if doc is None:
            row("2.6", f"{case} M{M} (n_orders cap)", "true", fn,
                lambda fn=fn, case=case: "n_orders" in g(fn, case,
                                                         "perlayer_error"))
            continue
        row("2.6", f"{case} closure M{M}", doc, fn,
            lambda fn=fn, case=case: mx(fn, case, "perlayer_closure"))

# ---- 3.1 E2-1 ---------------------------------------------------------------
C = "e2_1_compare.json"
row("3.1", "fixtures equal", "134", C, lambda: g(C, "n_equal"))
row("3.1", "fixtures common", "134", C, lambda: g(C, "n_common"))
row("3.1", "mortar fixtures", "12", C, lambda: g(C, "n_mortar_equal"))
row("3.1", "idmap changed", "28", C, lambda: g(C, "idmap_changed"))
row("3.1", "idmap total", "36", C, lambda: g(C, "idmap_total"))

# ---- 3.2 E2-2 ---------------------------------------------------------------
S2 = "e2_2_same_map_M4.json"
for inc, d_f, d_h in (("th0.0_ph0.0", "1.4e-15", "1.10"),
                      ("th0.3_ph0.0", "1.3e-15", "0.92"),
                      ("th0.3_ph0.7", "3.1e-15", "0.80")):
    row("3.2", f"{inc} per-layer bit-identical", "true", S2,
        lambda inc=inc: g(S2, inc, "perlayer_bytes_equal"))
    row("3.2", f"{inc} forced mortar", d_f, S2,
        lambda inc=inc: g(S2, inc, "forced_mortar_vs_shared"))
    row("3.2", f"{inc} H-swap off", d_h, S2,
        lambda inc=inc: g(S2, inc, "hswap_off_vs_shared"))

# ---- 3.3 E2-3 ---------------------------------------------------------------
for M, dmu, dms, dus, dcl in ((4, "2.1e-2", "3.6e-2", "3.6e-2", "1.2e-2"),
                              (5, "2.0e-3", "4.3e-3", "2.8e-3", "1.0e-3"),
                              (6, "6.0e-3", "5.8e-3", "6.9e-4", "8.0e-5"),
                              (7, "5.2e-4", "4.8e-4", "3.3e-4", "1.9e-5")):
    fn = f"e2_3_stretches_M{M}_th0.0_ph0.0.json"
    row("3.3", f"M{M} mapped vs unmapped", dmu, fn,
        lambda fn=fn: rt(fn, "mapped", "unmapped"))
    row("3.3", f"M{M} mapped vs union", dms, fn,
        lambda fn=fn: g(fn, "mapped_vs_shared"))
    row("3.3", f"M{M} unmapped vs union", dus, fn,
        lambda fn=fn: g(fn, "unmapped_vs_shared"))
    row("3.3", f"M{M} mapped closure", dcl, fn,
        lambda fn=fn: mx(fn, "mapped", "closure"))

# ---- 3.4 E2-4 ---------------------------------------------------------------
for M, doc in ((4, "6.5e-4"), (5, "1.8e-4"), (6, "2.4e-5"), (7, "7.6e-7"),
               (8, "9.7e-7")):
    fn = f"e2_4_closure_M{M}.json"
    row("3.4", f"closure M{M}", doc, fn, lambda fn=fn: mx(fn, "closure"))
row("3.4", "H-swap off closure in0 (M5)", "0.10", "e2_8_mutations_M5.json",
    lambda: g("e2_8_mutations_M5.json", "hswap_off", "closure")[0])
row("3.4", "H-swap off closure in1 (M5)", "0.12", "e2_8_mutations_M5.json",
    lambda: g("e2_8_mutations_M5.json", "hswap_off", "closure")[1])
A = "e2_4_absorb_M4.json"
row("3.4", "lossless layer absorbs in0", "1.5e-15", A,
    lambda: abs(g(A, "lossless_layer2")[0]))
row("3.4", "lossless layer absorbs in1", "4.6e-16", A,
    lambda: abs(g(A, "lossless_layer2")[1]))
row("3.4", "-R flux form in0", "5.0e-3", A,
    lambda: g(A, "failbefore_minusR_layer2")[0])
row("3.4", "-R flux form in1", "4.7e-3", A,
    lambda: g(A, "failbefore_minusR_layer2")[1])
row("3.4", "budget mismatch in0", "7.5e-3", A, lambda: g(A, "mismatch")[0])
row("3.4", "budget mismatch in1", "5.4e-3", A, lambda: g(A, "mismatch")[1])
for M, d_sa, d_nr in ((4, "3.6e-4", "2.1e-3"), (5, "7.4e-5", "3.4e-3"),
                      (6, "5.0e-6", "2.7e-3")):
    fn = f"e2_4_vacuum_M{M}.json"
    row("3.4", f"vacuum M{M} per-layer vs shared spacer", "0", fn,
        lambda fn=fn: g(fn, "perlayer_vs_shared_spacer"))
    row("3.4", f"vacuum M{M} bytes equal", "true", fn,
        lambda fn=fn: g(fn, "perlayer_bytes_equal_shared"))
    row("3.4", f"vacuum M{M} spacer vs alone", d_sa, fn,
        lambda fn=fn: g(fn, "shared_spacer_vs_alone"))
    row("3.4", f"vacuum M{M} noride vs alone", d_nr, fn,
        lambda fn=fn: g(fn, "noride_vs_alone"))
for M, d_pl, d_cm, d_cp in ((4, "9.9e-2", "1.5e-3", "1.5e-3"),
                            (5, "1.1e-2", "1.4e-3", "2.8e-4"),
                            (6, "9.7e-3", "1.8e-4", "2.8e-5"),
                            (7, "4.5e-4", "8.5e-6", "5.1e-6")):
    fn = f"e2_4_merged_M{M}.json"
    row("3.4", f"merged M{M} fast path bit-identical", "true", fn,
        lambda fn=fn: g(fn, "fastpath_bytes_equal_shared"))
    row("3.4", f"merged M{M} per-layer vs merged", d_pl, fn,
        lambda fn=fn: g(fn, "perlayer_maps_vs_merged"))
    row("3.4", f"merged M{M} closure merged", d_cm, fn,
        lambda fn=fn: mx(fn, "closure_merged"))
    row("3.4", f"merged M{M} closure per-layer", d_cp, fn,
        lambda fn=fn: mx(fn, "closure_perlayer"))

# ---- 3.5 E2-5 ---------------------------------------------------------------
Q4, Q6 = "e2_5_quadrature_M4.json", "e2_5_quadrature_M6.json"
tab = {4: ("1.5e-2", "4.3e-3", "1.4e-1", "1.8e-1"),
       8: ("3.7e-8", "1.9e-6", "1.6e-5", "8.4e-5"),
       12: ("9.3e-13", "3.6e-10", "6.9e-10", "1.2e-7"),
       16: ("4.6e-16", "3.5e-14", "1.7e-14", "3.5e-11"),
       20: ("3.8e-15", "8.1e-16", "1.3e-14", "5.3e-15")}
for n, docs in tab.items():
    for (fn, case), doc in zip(((Q4, "circle_sinx"), (Q4, "sinx_siny"),
                                (Q6, "circle_sinx"), (Q6, "sinx_siny")), docs):
        row("3.5", f"{fn[:-5]} {case} n={n}", doc, fn,
            lambda fn=fn, case=case, n=n: lad(fn, case, n))
for (fn, case), (dn, dc) in zip(((Q4, "circle_sinx"), (Q4, "sinx_siny"),
                                 (Q6, "circle_sinx"), (Q6, "sinx_siny")),
                                (("23", "1.3e-15"), ("27", "8.2e-15"),
                                 ("30", "9.7e-15"), ("35", "2.0e-15"))):
    row("3.5", f"{fn[:-5]} {case} adaptive n", dn, fn,
        lambda fn=fn, case=case: g(fn, case, "adaptive_n"))
    row("3.5", f"{fn[:-5]} {case} adaptive last change", dc, fn,
        lambda fn=fn, case=case: g(fn, case, "adaptive_change"))
row("3.5", "composite n=64 fail-before", "8.6e-4", D, lambda: next(
    r["rel"] for r in g(D, "composite_vs_cut", "composite") if r["n"] == 64))
NS = "e2_n_near_singular.json"
for r2, doc in (("0.3", "41"), ("0.34", "41"), ("0.355", "62"),
                ("0.3599", "96")):
    row("3.5", f"near-singular r2={r2} n", doc, NS,
        lambda r2=r2: g(NS, f"r2={r2}", "n"))
row("3.5", "near-singular max change", "2.2e-13", NS, lambda: max(
    g(NS, f"r2={r}", "change") for r in ("0.3", "0.34", "0.355", "0.3599")))
row("3.5", "1.2e-6 apart raises RuntimeError", "true", NS,
    lambda: g(NS, "r2=0.36000119999999997", "raised").startswith(
        "RuntimeError"))

# ---- 3.6 E2-6 ---------------------------------------------------------------
for M, (r0, r40, cl, w0, w40) in {
        4: ("1.4e-4", "7.5e-4", "5.8e-3", "0.039", "0.067"),
        5: ("1.1e-5", "1.1e-4", "1.8e-3", "0.041", "0.050"),
        6: ("7.7e-6", "7.2e-6", "2.3e-4", "0.045", "0.073"),
        7: ("1.0e-6", "7.3e-6", "5.6e-5", "0.038", "0.077")}.items():
    fn = f"e2_6_angles_M{M}.json"
    row("3.6", f"M{M} recip (25,0)", r0, fn,
        lambda fn=fn: g(fn, "th25_ph0", "recip"))
    row("3.6", f"M{M} recip (25,40)", r40, fn,
        lambda fn=fn: g(fn, "th25_ph40", "recip"))
    row("3.6", f"M{M} closure max", cl, fn, lambda fn=fn: max(
        mx(fn, k, c) for k in ("th25_ph0", "th25_ph40")
        for c in ("closure", "closure_reverse")))
    row("3.6", f"M{M} wrong pairing (25,0)", w0, fn,
        lambda fn=fn: g(fn, "th25_ph0", "wrong_pair"))
    row("3.6", f"M{M} wrong pairing (25,40)", w40, fn,
        lambda fn=fn: g(fn, "th25_ph40", "wrong_pair"))

# ---- 3.7 E2-7 ---------------------------------------------------------------
for M, (cl, ab, mm) in {4: ("1.5e-3", "2.5e-14", "1.2e-3"),
                        5: ("2.1e-4", "5.3e-14", "8.1e-5"),
                        6: ("7.4e-5", "2.6e-13", "6.6e-5"),
                        7: ("2.9e-6", "2.7e-13", "1.4e-6")}.items():
    fn = f"e2_7_three_M{M}.json"
    row("3.7", f"M{M} closure", cl, fn, lambda fn=fn: mx(fn, "closure"))
    row("3.7", f"M{M} lossless absorption", ab, fn,
        lambda fn=fn: g(fn, "lossless_layers"))
    row("3.7", f"M{M} sum(A) vs 1-R-T", mm, fn, lambda fn=fn: mx(fn, "mismatch"))

# ---- 3.8 E2-8 ---------------------------------------------------------------
MU = "e2_8_mutations_M5.json"
row("3.8", "shipped closure in0", "1.8e-4", MU,
    lambda: g(MU, "closure_shipped")[0])
row("3.8", "shipped closure in1", "9.0e-5", MU,
    lambda: g(MU, "closure_shipped")[1])
row("3.8", "Newton tol 1e3x d_RT", "2.6e-15", MU,
    lambda: g(MU, "inv_tol_1e3", "d_RT"))
row("3.8", "maps ignored d_RT", "6.7e-2", MU,
    lambda: g(MU, "maps_ignored", "d_RT"))
row("3.8", "maps ignored closure", "2.0e-4", MU,
    lambda: mx(MU, "maps_ignored", "closure"))
row("3.8", "both ways maps ignored", "3.9e-2", MU,
    lambda: g(MU, "nonoverlap_perlayer_vs_merged", "maps_ignored"))
row("3.8", "both ways shipped", "1.1e-2", MU,
    lambda: g(MU, "nonoverlap_perlayer_vs_merged", "shipped"))
row("3.8", "nocut d_RT", "6.8e-5", MU, lambda: g(MU, "nocut", "d_RT"))
row("3.8", "H-swap d_RT", "0.51", MU, lambda: g(MU, "hswap_off", "d_RT"))
row("3.8", "fast path forced d_RT", "0", MU, lambda: g(MU, "fast_forced",
                                                         "d_RT"))
row("3.8", "fast path forced refusal names claim", "true", MU,
    lambda: "DIFFERENT physical" in g(MU, "fast_forced", "merge_refusal"))

# ---- 3.9 E2-9 ---------------------------------------------------------------
for M, (mp, mw, ms, p0, p1, pw, cw, cn, ra) in {
        6: ("800", "34.7", "6,9", "450", "512", "10.0", "1.18", "30", "0.29"),
        8: ("1568", "252", "8,12", "882", "968", "47.8", "1.11", "36",
            "0.19")}.items():
    fn = f"e2_9_cost_M{M}.json"
    row("3.9", f"M{M} merged grid 4x4", "true", fn,
        lambda fn=fn: g(fn, "merged_grid") == [4, 4])
    row("3.9", f"M{M} merged pencil", mp, fn, lambda fn=fn: g(fn,
                                                              "merged_pencil"))
    row("3.9", f"M{M} merged wall", mw, fn, lambda fn=fn: g(fn, "merged_wall"))
    row("3.9", f"M{M} modal counts {ms}", "true", fn,
        lambda fn=fn, ms=ms: g(fn, "perlayer_modal_counts") == [
            int(x) for x in ms.split(",")])
    row("3.9", f"M{M} pencil 0", p0, fn,
        lambda fn=fn: g(fn, "perlayer_pencils")[0])
    row("3.9", f"M{M} pencil 1", p1, fn,
        lambda fn=fn: g(fn, "perlayer_pencils")[1])
    row("3.9", f"M{M} per-layer wall", pw, fn,
        lambda fn=fn: g(fn, "perlayer_wall"))
    row("3.9", f"M{M} cross-mass wall", cw, fn,
        lambda fn=fn: g(fn, "cross_mass_wall"))
    row("3.9", f"M{M} cross-mass n", cn, fn,
        lambda fn=fn: g(fn, "cross_mass", 0, "n"))
    row("3.9", f"M{M} ratio", ra, fn,
        lambda fn=fn: g(fn, "ratio_perlayer_over_merged"))
row("3.9", "cross-mass share upper end ('2-11 %')", "11",
    "e2_9_cost_M6.json", lambda: 100 * g("e2_9_cost_M6.json",
                                         "cross_mass_wall")
    / g("e2_9_cost_M6.json", "perlayer_wall"))
row("3.9", "cross-mass share lower end ('2-11 %')", "2",
    "e2_9_cost_M8.json", lambda: 100 * g("e2_9_cost_M8.json",
                                         "cross_mass_wall")
    / g("e2_9_cost_M8.json", "perlayer_wall"))

# ---- 4.3 F-E2-3 -------------------------------------------------------------
for M, (g1, o3, cf) in {4: ("4.1e-3", "3.6e-3", "4.7e-6"),
                        5: ("3.0e-3", "2.1e-3", "2.3e-7"),
                        6: ("1.2e-3", "1.7e-3", "7.7e-9")}.items():
    fn = f"e2_v_spacer_shipped_M{M}.json"
    row("4.3", f"M{M} spacer one cell", g1, fn,
        lambda fn=fn: g(fn, "grid1", "vs_alone"))
    row("4.3", f"M{M} spacer 3x3", o3, fn,
        lambda fn=fn: g(fn, "offset3", "vs_alone"))
    row("4.3", f"M{M} spacer conforming", cf, fn,
        lambda fn=fn: g(fn, "conforming", "vs_alone"))
for M, (pu, pq) in {5: ("2.8e-2", "5.1e-3"), 6: ("9.7e-3", "8.6e-4"),
                    7: ("5.6e-3", "1.4e-3")}.items():
    fn = f"e2_w_shipped_baseline_M{M}.json"
    row("4.3", f"M{M} stripe vs union", pu, fn,
        lambda fn=fn: g(fn, "perlayer", "vs_union"))
    row("4.3", f"M{M} stripe q-matched", pq, fn,
        lambda fn=fn: g(fn, "perlayer_qmatch", "vs_union"))

# ---- 4.4 / 9 ---------------------------------------------------------------
row("4.4", "q-matched 3.9e-9 (the (8,6) arm)", "3.9e-9",
    "e2_h_host_sin_circ_vac_M6_Ms8-6_th0.0_ph0.0.json",
    lambda: g("e2_h_host_sin_circ_vac_M6_Ms8-6_th0.0_ph0.0.json",
              "err_vs_airy"))
W5 = "e2_x_separable_M5_wsl.json"
row("9", "WSL separable X11", "1.27e-14", W5, lambda: g(W5, "X11"))
row("9", "WSL separable X22", "5.37e-15", W5, lambda: g(W5, "X22"))
row("9", "Win separable X11", "1.28e-14", X, lambda: g(X, "X11"))
row("9", "Win separable X22", "5.43e-15", X, lambda: g(X, "X22"))
row("9", "WSL adaptive n circle", "23", "e2_5_quadrature_M4_wsl.json",
    lambda: g("e2_5_quadrature_M4_wsl.json", "circle_sinx", "adaptive_n"))
row("9", "WSL adaptive n sinx/siny", "27", "e2_5_quadrature_M4_wsl.json",
    lambda: g("e2_5_quadrature_M4_wsl.json", "sinx_siny", "adaptive_n"))

out, nbad = [], 0
for sec, what, doc, fn, fx in ROWS:
    try:
        v = fx()
        ok = same(doc, v)
    except Exception as e:  # noqa: BLE001
        v, ok = f"ERR {type(e).__name__}: {e}", False
    nbad += not ok
    out.append(dict(section=sec, what=what, doc=doc, json=fn,
                    value=v if not isinstance(v, (np.generic,)) else float(v),
                    verdict="MATCH" if ok else "MISMATCH"))
    print(f"{'MATCH   ' if ok else 'MISMATCH'} {sec:4s} {what:55s} doc {doc:>9s}"
          f"  json {v if isinstance(v, str) else f'{float(v):.4g}'}  [{fn}]")
print(f"{len(ROWS)} numbers, {len(ROWS) - nbad} MATCH, {nbad} MISMATCH")
with open(os.path.join(HERE, "v10_docnums.json"), "w") as f:
    json.dump({"rows": out, "n": len(ROWS), "mismatch": nbad}, f, indent=1)

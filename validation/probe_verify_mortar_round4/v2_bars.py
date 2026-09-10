"""INDEPENDENT re-derivation of ROUND 4's two RESTATED test bars.

BAR 1 (round 4 S5.1).  ``max(ctrl.on) < 0.95`` became a SPREAD comparison,
``s = min(on_a, on_b) / max(on_a, on_b)`` over the two block columns of the
generalized mortar operand's near-null right singular vector::

    s_mixed < 1e-3        s_ctrl > 1e3 * s_mixed        s_ctrl > 0.1

Re-measured here on a population built from THIS verification's own fixtures
(``_vfix4.BARS_FAMILY`` at ``M`` = 4, 5, 6), with the promotion detected
STRUCTURALLY -- ``_modes_as_general`` writes ``(W, V, lam, W, -V, -lam)``, so
the test is exact, not a tolerance.

BAR 2 (round 4 S5.2).  ``e_band > 1.15 * e_ord`` at ``M`` = 6 became the same
ratio at ``M`` = 5 AND 6 plus a CONVERGENCE PRECONDITION
``e_ord(6) < 0.5 * e_ord(5)``.  Re-measured here on a DIFFERENT fixture --
period 1.19e-6, wavelength 0.83e-6, a y-INVARIANT three-layer grating whose
middle layer is uniform host, so the exact 1-D ``PMMStack`` (a different
assembly with no mortar and no element grid) is an independent oracle and the
truth does not move with the swept width.

Usage::

    VMORTAR4_TREE=C:/tmp/lum_vmortar4 python .../v2_bars.py --tag win \
        --parts pl
"""
# ``_path`` MUST be imported before anything that touches lumenairy, so this
# block is deliberately not isort-ordered.
from __future__ import annotations  # noqa: I001

import _path  # noqa: F401  (MUST be first: pins the tree under measurement)

import argparse                                             # noqa: E402
import contextlib                                           # noqa: E402
import json                                                 # noqa: E402
import pathlib                                              # noqa: E402
import time                                                 # noqa: E402
import warnings                                             # noqa: E402

import numpy as np                                          # noqa: E402

import _vfix4 as F                                          # noqa: E402
from lumenairy.elements.pmm import _core as _pc              # noqa: E402
from lumenairy.elements.pmm import stack2d_pure as _sp       # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
GEN_SITE = "pmm2d staggered GENERALIZED mortar interface"


# ---- the capture instrument, written here rather than reused -------------
def _promoted(six):
    """``_modes_as_general`` writes ``(W, V, lam, W, -V, -lam)``.  The test is
    an EXACT structural one: the second triple's ``W`` is the first's and the
    other two are its bitwise negations."""
    W, V, lam, W2, V2, lam2 = six[:6]
    if V2 is None or lam2 is None:
        return False
    same_W = (W2 is W) or bool(np.array_equal(np.asarray(W2), np.asarray(W)))
    return bool(same_W and np.array_equal(np.asarray(V2), -np.asarray(V))
                and np.array_equal(np.asarray(lam2), -np.asarray(lam)))


@contextlib.contextmanager
def capture():
    """Yield a list of one record per GENERALIZED mortar solve, with the guard
    replaced by a plain ``np.linalg.solve`` so a refused stack still runs."""
    out, cur = [], {}
    o_guard = _pc._guarded_mortar_solve
    o_gen = _pc._interface_smatrix_general_mortar_2d

    def gen(six_a, six_b, ga, gb, cr, kron_apply):
        cur.clear()
        cur.update(prom_a=_promoted(six_a), prom_b=_promoted(six_b),
                   ma=int(np.asarray(six_a[0]).shape[1]))
        try:
            return o_gen(six_a, six_b, ga, gb, cr, kron_apply)
        finally:
            cur.clear()

    def guard(A, B, site, ga=None, gb=None, hint=None, screen="rcond"):
        if site == GEN_SITE:
            out.append({"site": site, "screen": screen, "A": np.array(A),
                        "B": np.array(B), **cur})
        return np.linalg.solve(A, B)

    _pc._guarded_mortar_solve = guard
    _pc._interface_smatrix_general_mortar_2d = gen
    _sp._interface_smatrix_general_mortar_2d = gen
    try:
        yield out
    finally:
        _pc._guarded_mortar_solve = o_guard
        _pc._interface_smatrix_general_mortar_2d = o_gen
        _sp._interface_smatrix_general_mortar_2d = o_gen


def _facts(A, B, ma):
    n = A.shape[0]
    U, s, Vh = np.linalg.svd(A)
    v = Vh[-1].conj()
    half = int(n // 2 if ma is None else ma)
    on_a = float(np.linalg.norm(v[:half]))
    on_b = float(np.linalg.norm(v[half:]))
    X = np.linalg.solve(A, B)
    nB = float(np.linalg.norm(B))
    lo, hi = min(on_a, on_b), max(on_a, on_b)
    return {
        "n": int(n), "s_min": float(s[-1]), "s_max": float(s[0]),
        "s_ratio": float(s[-1] / s[0]),
        "on_a": on_a, "on_b": on_b, "max_on": hi,
        "spread": (lo / hi) if hi > 0 else 1.0,
        "residual": (float(np.linalg.norm(A @ X - B)) / nB) if nB else 0.0,
    }


# ==========================================================================
# p -- BAR 1, the SPREAD population
# ==========================================================================
def part_p(Ms=(4, 5, 6)):
    rows = []
    for name in F.BARS_FAMILY:
        for M in Ms:
            t0 = time.perf_counter()
            with capture() as ops, warnings.catch_warnings():
                warnings.simplefilter("ignore")
                F.build(name, M).solve(jones=False)
            for i, rec in enumerate(ops):
                f = _facts(rec["A"], rec["B"], rec.get("ma"))
                pa, pb = rec.get("prom_a"), rec.get("prom_b")
                cls = ("one_promoted" if pa != pb else
                       "both_promoted" if pa and pb else "neither_promoted")
                rows.append({"fixture": name, "M": M, "interface": i,
                             "class": cls, "prom_a": pa, "prom_b": pb,
                             "seconds": round(time.perf_counter() - t0, 2),
                             **f})
            print(f"  {name:24s} M={M} -> " + "; ".join(
                f"{r['class']}: on {r['on_a']:.4f}/{r['on_b']:.4f} "
                f"spread {r['spread']:.4e} s_ratio {r['s_ratio']:.3e}"
                for r in rows if r["fixture"] == name and r["M"] == M),
                flush=True)
    by = {}
    for r in rows:
        by.setdefault(r["class"], []).append(r)
    summary = {c: {"n_operands": len(rs),
                   "spread_min": min(r["spread"] for r in rs),
                   "spread_max": max(r["spread"] for r in rs),
                   "max_on_min": min(r["max_on"] for r in rs),
                   "max_on_max": max(r["max_on"] for r in rs),
                   "s_ratio_min": min(r["s_ratio"] for r in rs),
                   "s_ratio_max": max(r["s_ratio"] for r in rs),
                   "residual_max": max(r["residual"] for r in rs)}
               for c, rs in by.items()}
    one = by.get("one_promoted", [])
    ctrl = by.get("neither_promoted", []) + by.get("both_promoted", [])
    bars = {}
    if one and ctrl:
        s_mixed_worst = max(r["spread"] for r in one)
        s_ctrl_worst = min(r["spread"] for r in ctrl)
        bars = {
            "s_mixed_worst": s_mixed_worst,
            "s_mixed_lt_1e_3_margin_decades":
                float(np.log10(1e-3 / s_mixed_worst)),
            "s_ctrl_worst": s_ctrl_worst,
            "s_ctrl_over_1e3_s_mixed":
                s_ctrl_worst / (1e3 * s_mixed_worst),
            "s_ctrl_gt_0p1_margin": s_ctrl_worst / 0.1,
            "old_bar_max_on_worst_ctrl": max(
                (r["max_on"] for r in ctrl), default=None),
            "old_bar_slack_vs_0p95": 0.95 - max(
                (r["max_on"] for r in ctrl), default=0.0),
        }
    return {"rows": rows, "summary": summary, "bars": bars}


# ==========================================================================
# l -- BAR 2, the M ladder against an EXACT 1-D oracle
# ==========================================================================
def _err(frac, M, o_ref, R_ref, T_ref):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        o, R, T = F.ladder_2d(frac, M).solve(jones=False)
        dt = time.perf_counter() - t0
    o, R, T = np.asarray(o), np.atleast_2d(R), np.atleast_2d(T)
    worst = 0.0
    for m in (-1, 0, 1):
        sel = int(np.where((o[:, 0] == m) & (o[:, 1] == 0))[0][0])
        j = int(np.where(np.asarray(o_ref) == m)[0][0])
        worst = max(worst, abs(float(R[1, sel]) - float(R_ref[1, j])),
                    abs(float(T[1, sel]) - float(T_ref[1, j])))
    n_warn = len([w for w in ws if "degradation band" in str(w.message)])
    return worst, round(dt, 2), n_warn


def part_l(Ms=(5, 6, 7)):
    o14, R14, T14 = F.ladder_oracle(14)
    o12, R12, T12 = F.ladder_oracle(12)
    keep = np.abs(np.asarray(o14)) <= 1
    self_gap = float(max(np.max(np.abs(R12[:, keep] - R14[:, keep])),
                         np.max(np.abs(T12[:, keep] - T14[:, keep]))))
    print(f"  oracle self-gap (deg 12 -> 14) = {self_gap:.6e}", flush=True)
    rows = []
    for M in Ms:
        e_ord, t_o, w_o = _err(3.0e-1, M, o14, R14, T14)
        e_band, t_b, w_b = _err(3.0e-3, M, o14, R14, T14)
        rows.append({"M": M, "e_ord": e_ord, "e_band": e_band,
                     "ratio": e_band / e_ord, "warn_ord": w_o,
                     "warn_band": w_b, "seconds": t_o + t_b})
        print(f"  M={M} e_ord={e_ord:.6e} e_band={e_band:.6e} "
              f"ratio={e_band / e_ord:.4f} warn={w_o}/{w_b} "
              f"({t_o + t_b:.1f} s)", flush=True)
    for i in range(1, len(rows)):
        rows[i]["ord_fall"] = rows[i - 1]["e_ord"] / rows[i]["e_ord"]
        rows[i]["band_fall"] = rows[i - 1]["e_band"] / rows[i]["e_band"]
    gate = {}
    got = {r["M"]: r for r in rows}
    if 5 in got and 6 in got:
        gate = {
            "e_ord_6_over_e_ord_5": got[6]["e_ord"] / got[5]["e_ord"],
            "precondition_bar": 0.5,
            "precondition_margin": 0.5 / (got[6]["e_ord"]
                                          / got[5]["e_ord"]),
            "ratio_M5": got[5]["ratio"], "ratio_M6": got[6]["ratio"],
            "worst_ratio": min(got[5]["ratio"], got[6]["ratio"]),
            "ratio_bar": 1.15,
            "ratio_margin": min(got[5]["ratio"], got[6]["ratio"]) / 1.15,
            "oracle_precondition": {
                "self_gap": self_gap,
                "gap_between_arms_M6": abs(got[6]["e_band"]
                                           - got[6]["e_ord"]),
                "ratio": (abs(got[6]["e_band"] - got[6]["e_ord"])
                          / self_gap) if self_gap else float("inf"),
                "bar": 20.0,
            },
        }
    return {"self_gap": self_gap, "rows": rows, "gate": gate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--parts", default="pl")
    ap.add_argument("--ladder-M", default="5,6,7")
    a = ap.parse_args()
    out = {"tag": a.tag, "env": F.env(), "tree": _path.TREE,
           "lumenairy_file": _path.LUMENAIRY_FILE}
    if "p" in a.parts:
        t0 = time.perf_counter()
        out["p"] = part_p()
        out["p_seconds"] = round(time.perf_counter() - t0, 1)
    if "l" in a.parts:
        t0 = time.perf_counter()
        out["l"] = part_l(tuple(int(x) for x in a.ladder_M.split(",")))
        out["l_seconds"] = round(time.perf_counter() - t0, 1)
    dest = HERE / f"v2_bars_{a.tag}.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf8")
    print("wrote", dest)
    if "p" in out:
        print(json.dumps(out["p"]["summary"], indent=1, sort_keys=True))
        print(json.dumps(out["p"]["bars"], indent=1, sort_keys=True))
    if "l" in out:
        print(json.dumps(out["l"]["gate"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

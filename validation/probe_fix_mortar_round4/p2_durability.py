"""ROUND 4 / S14 -- the two SAMPLE-scoped bars in the round-3 gate file,
re-derived from the populations the verification measured.

    S14 bar                          scope it was stated at   this probe
    -------------------------------  -----------------------  -------------
    ``max(ctrl.on) < 0.95``          SAMPLE (0.015 of slack   part ``p``
                                     against a legitimate
                                     neighbouring fixture at
                                     0.935)
    ``e_band > 1.15 * e_ord`` @ M=6  SAMPLE (the one rung      part ``l``
                                     whose ratio is fixture-
                                     sensitive by construction)

PART ``p`` -- the PARTICIPATION population.  The near-null right singular
vector's split between the two block columns, over EVERY generalized-mortar
class the two round-3 fixture families build: mixed (exactly one promoted
side), both-promoted, both-out-of-plane, both-slanted, and the ADVERSARIAL
control whose tensor is in-plane to 1e-9 -- the 0.935 reading that leaves the
0.95 bar 0.015 of slack.  The quantity proposed in its place is the SPREAD
``min(on_a, on_b) / max(on_a, on_b)``: scale-free, re-measured on both sides
of the comparison every run, and a DECISION ("localised" vs "not") rather than
a reading.

PART ``l`` -- the DEGRADATION LADDER on the round-3 gate's OWN fixture, at
``M`` = 5, 6 and 7, so the rung where the ORDINARY arm has converged can be
identified from the ladder itself rather than assumed.  The oracle is the
exact 1-D ``PMMStack`` at degree 12 and 14, whose self-gap is measured beside
the errors it is used to compare.

Usage::

    python validation/probe_fix_mortar_round4/p2_durability.py --tag win
"""
# ``_path`` MUST be imported before anything that touches lumenairy, so
# this block is deliberately not isort-ordered.
from __future__ import annotations  # noqa: I001

import _path  # noqa: F401  (pins THIS worktree's lumenairy)  (MUST be first: pins this worktree's lumenairy)

import argparse                                             # noqa: E402
import json                                                 # noqa: E402
import pathlib                                              # noqa: E402
import sys                                                  # noqa: E402
import time                                                 # noqa: E402
import warnings                                             # noqa: E402

import numpy as np                                          # noqa: E402

import lumenairy                                            # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "probe_verify_mortar_round3"))

import _capture as CAP                                       # noqa: E402,I001
import _vfix as F                                            # noqa: E402
from lumenairy.elements.pmm import PMM2DStackPure, PMMStack  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import _C         # noqa: E402


def _spread(on_a, on_b):
    """``min / max`` of the two block-column norms of the near-null right
    singular vector.  1.0 = perfectly shared, 0.0 = entirely on one side."""
    lo, hi = min(on_a, on_b), max(on_a, on_b)
    return lo / hi if hi > 0 else 1.0


# ==========================================================================
# p -- the participation population
# ==========================================================================
def part_p(Ms=(4, 5, 6)):
    rows = []
    kinds = (("mix_spacer", "one_promoted"),
             ("mix_pattern", "one_promoted"),
             ("mix_magnetic", "one_promoted"),
             ("mix_slant_inplane", "one_promoted"),
             ("adv_strong_oop_next_to_inplane", "one_promoted"),
             ("adv_tiny_slant_inplane", "one_promoted"),
             ("both_promoted", "mixed_stack"),
             ("ctrl_oop_both", "neither_promoted"),
             ("ctrl_slant_both", "neither_promoted"),
             ("adv_near_inplane_oop", "neither_promoted"))
    for kind, klass in kinds:
        for M in Ms:
            t0 = time.perf_counter()
            with CAP.capture() as ops:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    F.build(kind, M).solve(jones=False)
            for i, rec in enumerate(ops):
                f = CAP.svd_facts(rec["A"], rec["B"], rec.get("ma"))
                pa, pb = rec.get("prom_a"), rec.get("prom_b")
                cls = ("one_promoted" if pa != pb else
                       "both_promoted" if pa and pb else "neither_promoted")
                rows.append({
                    "kind": kind, "family": klass, "M": M, "interface": i,
                    "class": cls, "prom_a": pa, "prom_b": pb,
                    "n": f["n"], "s_ratio": f["s_ratio"],
                    "on_a": f["on_a"], "on_b": f["on_b"],
                    "spread": _spread(f["on_a"], f["on_b"]),
                    "max_on": max(f["on_a"], f["on_b"]),
                    "residual": f["residual"],
                    "seconds": round(time.perf_counter() - t0, 2),
                })
            print(f"  {kind:32s} M={M} -> "
                  + "; ".join(f"{r['class']}: on {r['on_a']:.3f}/"
                              f"{r['on_b']:.3f} spread {r['spread']:.3e} "
                              f"s_ratio {r['s_ratio']:.3e}"
                              for r in rows if r['kind'] == kind
                              and r['M'] == M), flush=True)
    by = {}
    for r in rows:
        by.setdefault(r["class"], []).append(r)
    summary = {}
    for cls, rs in by.items():
        summary[cls] = {
            "n_operands": len(rs),
            "spread_min": min(r["spread"] for r in rs),
            "spread_max": max(r["spread"] for r in rs),
            "max_on_min": min(r["max_on"] for r in rs),
            "max_on_max": max(r["max_on"] for r in rs),
            "s_ratio_min": min(r["s_ratio"] for r in rs),
            "s_ratio_max": max(r["s_ratio"] for r in rs),
        }
    return {"rows": rows, "summary": summary}


# ==========================================================================
# l -- the degradation ladder, on the ROUND-3 GATE's own fixture
# ==========================================================================
_LP, _LWL, _LTH = 0.93, 0.66, 0.19
_LEPSP, _LEPSH, _LTT = 6.25, 2.1, 0.14
_LW0, _LW2, _LYW = (0.155, 0.585), (0.315, 0.795), (0.22, 0.68)


def _oracle(deg):
    st = PMMStack(_LP, degree=deg, far_field_orders=5)
    st.add_layer(_LTT, segments=[(_LW0[0], _LEPSH),
                                 (_LW0[1] - _LW0[0], _LEPSP),
                                 (1.0 - _LW0[1], _LEPSH)])
    st.add_layer(_LTT, segments=[(1.0, _LEPSH)])
    st.add_layer(_LTT, segments=[(_LW2[0], _LEPSH),
                                 (_LW2[1] - _LW2[0], _LEPSP),
                                 (1.0 - _LW2[1], _LEPSH)])
    st.set_source(_LWL, theta=_LTH)
    o, R, T = st.solve(stabilize=None)[:3]
    return np.asarray(o), np.atleast_2d(R), np.atleast_2d(T)


def _tile():
    c = np.full((3, 3), _C(_LEPSH))
    c[1, :] = _C(_LEPSP)
    return c


def _err(frac, M, o14, R14, T14):
    sw = [(0.41 - frac / 2) * _LP, (0.41 + frac / 2) * _LP]
    st = PMM2DStackPure(_LP, n_modes=M, n_orders=1, layer_grids="per-layer")
    yws = [_LYW[0] * _LP, _LYW[1] * _LP]
    st.add_layer(_LTT, eps_cell=_tile(),
                 x_walls=[_LW0[0] * _LP, _LW0[1] * _LP], y_walls=yws)
    st.add_layer(_LTT, eps_cell=np.full((3, 3), _C(_LEPSH)), x_walls=sw,
                 y_walls=yws)
    st.add_layer(_LTT, eps_cell=_tile(),
                 x_walls=[_LW2[0] * _LP, _LW2[1] * _LP], y_walls=yws)
    st.set_source(_LWL, theta=_LTH, phi=0.0)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        o, R, T = st.solve(jones=False)
    dt = time.perf_counter() - t0
    o, R, T = np.asarray(o), np.atleast_2d(R), np.atleast_2d(T)
    worst = 0.0
    for m in (-1, 0, 1):
        sel = int(np.where((o[:, 0] == m) & (o[:, 1] == 0))[0][0])
        j = int(np.where(o14 == m)[0][0])
        worst = max(worst, abs(float(R[1, sel]) - float(R14[1, j])),
                    abs(float(T[1, sel]) - float(T14[1, j])))
    n_warn = len([w for w in ws if "degradation band" in str(w.message)])
    return worst, round(dt, 2), n_warn


def part_l(Ms=(5, 6, 7)):
    o14, R14, T14 = _oracle(14)
    o12, R12, T12 = _oracle(12)
    keep = np.abs(o14) <= 1
    self_gap = float(max(np.max(np.abs(R12[:, keep] - R14[:, keep])),
                         np.max(np.abs(T12[:, keep] - T14[:, keep]))))
    print(f"  oracle self-gap (12 -> 14) = {self_gap:.6e}", flush=True)
    rows = []
    for M in Ms:
        e_ord, t_ord, w_ord = _err(3.0e-1, M, o14, R14, T14)
        e_band, t_band, w_band = _err(3.0e-3, M, o14, R14, T14)
        rows.append({"M": M, "e_ord": e_ord, "e_band": e_band,
                     "ratio": e_band / e_ord,
                     "warn_ord": w_ord, "warn_band": w_band,
                     "seconds_ord": t_ord, "seconds_band": t_band})
        print(f"  M={M} e_ord={e_ord:.6e} e_band={e_band:.6e} "
              f"ratio={e_band / e_ord:.4f} ({t_ord + t_band:.1f} s)",
              flush=True)
    for i in range(1, len(rows)):
        rows[i]["ord_step"] = abs(rows[i]["e_ord"] - rows[i - 1]["e_ord"])
        rows[i]["ord_fall"] = rows[i - 1]["e_ord"] / rows[i]["e_ord"]
        rows[i]["band_fall"] = rows[i - 1]["e_band"] / rows[i]["e_band"]
    return {"self_gap": self_gap, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--parts", default="pl")
    ap.add_argument("--ladder-M", default="5,6,7")
    a = ap.parse_args()
    out = {"tag": a.tag,
           "lumenairy": str(pathlib.Path(lumenairy.__file__).resolve()),
           "version": lumenairy.__version__,
           "python": sys.version.split()[0], "numpy": np.__version__}
    if "p" in a.parts:
        t0 = time.perf_counter()
        out["p"] = part_p()
        out["p_seconds"] = round(time.perf_counter() - t0, 1)
    if "l" in a.parts:
        t0 = time.perf_counter()
        out["l"] = part_l(tuple(int(x) for x in a.ladder_M.split(",")))
        out["l_seconds"] = round(time.perf_counter() - t0, 1)
    f = HERE / f"p2_durability_{a.tag}.json"
    f.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="cp1252")
    print("wrote", f)
    if "p" in out:
        print(json.dumps(out["p"]["summary"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

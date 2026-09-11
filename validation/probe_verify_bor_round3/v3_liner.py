"""GAPS 3 and 4, INDEPENDENT -- the SEM mesh contract's verdict over a width
ladder at the axis, in the interior and at the outer wall, on every arm.

GEOMETRY IS THIS VERIFICATION'S OWN, not the fix round's: ``m`` = 2, ``N`` = 96,
degree 6, ``k0`` = 3.5, superstrate 1.2 / substrate 1.7, one 0.6-thick layer
whose segments carry ``eps`` 5.0 against a 2.2 background.  The fix round
measured ``Rbig`` = 24, ``m`` = 1, ``N`` = 120, degree 8, ``k0`` = 2.0.

TWO ``Rbig`` VALUES, and the second one is the point.  ``_BOR_FRAC_DEADBAND``
closes the comparison's tie only where the mesh reproduces the requested width
AT OR BELOW the bar.  Whether it does is a property of the WALL COORDINATES,
not of the library: ``fl(r_wall + w) - r_wall`` can land either side of ``w``.
Geometry A (``Rbig`` = 17, interior wall 7.0) reproduces it BELOW -- the fix
round's direction.  Geometry B (``Rbig`` = 12.5, interior wall 5.0) reproduces
it ABOVE, by ~1.6e5 ULP, which is four decades outside a 16-ULP deadband.

Parts:

``ladder``   9 widths (1e-9 .. 1e-5 of ``Rbig``, including the
             ``_BOR_MIN_ELEM_FRAC`` edge) x 3 positions x 2 geometries.
``repr``     the pure-arithmetic scan behind geometry B: over a grid of
             ``Rbig`` and interior-wall coordinates, how far the reproduced
             width lands from the bar, in ULP and on which side.
``formed``   the ``q_measurable`` distinction, constructed directly through
             ``measure_layer`` rather than hoped for.
``deadband`` the behaviour of ``_below`` around the bar, in ULP.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vb3  # noqa: E402

from lumenairy.elements.bor import _sem_contract as sc  # noqa: E402
from lumenairy.elements.bor.bor_stack import BORStack  # noqa: E402

M, N, DEGREE, K0 = 2, 96, 6, 3.5
NSUP, NSUB = 1.2, 1.7
THICK = 0.6
EPS_HI, EPS_LO = 5.0, 2.2
POSITIONS = ("axis", "middle", "outer")

#: (label, Rbig, interior wall) -- A reproduces the edge width BELOW the bar,
#: B reproduces it ABOVE.
GEOMS = (("A", 17.0, 7.0), ("B", 12.5, 5.0))

EDGE = float(sc._BOR_MIN_ELEM_FRAC)
LADDER = (1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, EDGE, 3e-6, 1e-5)


def segments(position, w, Rbig, mid):
    if position == "axis":
        return [(w, EPS_HI), (Rbig, EPS_LO)]
    if position == "middle":
        return [(mid, EPS_HI), (mid + w, EPS_LO), (Rbig, EPS_LO)]
    if position == "outer":
        return [(Rbig - w, EPS_LO), (Rbig, EPS_HI)]
    raise ValueError(position)


def record(position, w_frac, Rbig, mid):
    st = BORStack(Rbig=Rbig, m=M, N=N, n_superstrate=NSUP, n_substrate=NSUB,
                  basis="sem", degree=DEGREE)
    st.add_layer(THICK, segments=segments(position, w_frac * Rbig, Rbig, mid))
    st.set_source(k0=K0)
    prev = sc.BOR_SEM_MESH_GUARD
    sc.BOR_SEM_MESH_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.solve()
        recs = list(getattr(st, "_sem_mesh_report", []) or [])
    finally:
        sc.BOR_SEM_MESH_GUARD = prev
    assert recs, "%s liner at %g produced no mesh report" % (position, w_frac)
    r = recs[0]
    fa = float(r["w_min_own_frac"])
    return dict(position=position, w_frac=w_frac, Rbig=Rbig, mid=mid,
                w_min_own_frac=fa,
                own_frac_ulp_from_edge=float((fa - EDGE) / np.spacing(EDGE)),
                w_min_union_frac=float(r["w_min_union_frac"]),
                q_excess=float(r["q_excess"]),
                q_finite=bool(np.isfinite(r["q_excess"])),
                q_measurable=bool(r.get("q_measurable", True)),
                verdict=sc.verdict(r))


def part_ladder():
    t0 = time.time()
    rows, summary = [], {}
    for (label, Rbig, mid) in GEOMS:
        sub = []
        for w in LADDER:
            for p in POSITIONS:
                r = record(p, w, Rbig, mid)
                r["geom"] = label
                rows.append(r)
                sub.append(r)
                print("  %s w=%-12.6g %-7s own=%-22.17g (bar%+9.0f ulp) "
                      "q=%-12.6g fin=%-5s meas=%-5s -> %s"
                      % (label, w, p, r["w_min_own_frac"],
                         r["own_frac_ulp_from_edge"], r["q_excess"],
                         r["q_finite"], r["q_measurable"], r["verdict"]),
                      flush=True)
        by_w = {}
        for r in sub:
            by_w.setdefault(r["w_frac"], {})[r["position"]] = r
        disagree = sorted(w for w, d in by_w.items()
                          if len({x["verdict"] for x in d.values()}) > 1)
        nonfinite = [(r["position"], r["w_frac"], r["verdict"],
                      r["q_measurable"]) for r in sub if not r["q_finite"]]
        edge = {p: by_w[EDGE][p]["w_min_own_frac"] for p in POSITIONS}
        spread = max(edge.values()) - min(edge.values())
        summary[label] = dict(
            Rbig=Rbig, mid=mid, rungs=len(LADDER), rows=len(sub),
            n_disagreeing_rungs=len(disagree),
            positions_disagree_at=disagree,
            nonfinite_rows=nonfinite,
            nonfinite_read_ok=[x for x in nonfinite if x[2] == "ok"],
            edge_w_min_own_frac=edge,
            edge_ulp_from_bar={p: float((edge[p] - EDGE) / np.spacing(EDGE))
                               for p in POSITIONS},
            edge_verdicts={p: by_w[EDGE][p]["verdict"] for p in POSITIONS},
            edge_rel_spread=spread / EDGE,
            edge_ulps=spread / float(np.spacing(EDGE)),
            verdict_table={("%.6g" % w): {p: by_w[w][p]["verdict"]
                                          for p in POSITIONS} for w in LADDER})
    for label in summary:
        print(" ", label, {k: summary[label][k]
                           for k in ("n_disagreeing_rungs",
                                     "positions_disagree_at",
                                     "edge_verdicts", "edge_ulp_from_bar")})
    summary["seconds"] = time.time() - t0
    _vb3.dump("v3_ladder", dict(rows=rows, summary=summary))


def part_repr():
    """How the MESH reproduces a requested width of exactly
    ``_BOR_MIN_ELEM_FRAC * Rbig``, and on which side of the bar it lands.

    This is pure IEEE-754 arithmetic on the wall coordinates -- no solve -- and
    it is what decides whether the 16-ULP deadband reaches.  ``middle`` is
    ``fl(r + w) - r`` (the mesh's ``np.diff`` over the two walls the caller
    asked for); ``outer`` is ``Rbig - fl(Rbig - w)``; ``axis`` is ``w - 0``,
    which is exact by construction.
    """
    rows = []
    for Rbig in (3.0, 8.0, 10.0, 12.5, 17.0, 24.0, 31.7, 40.0, 50.0, 64.0,
                 100.0, 6.283185307179586, 2.0 * np.pi * 4.0):
        w = EDGE * Rbig
        for frac in (0.1, 0.2, 0.25, 1.0 / 3.0, 0.4, 0.5, 0.6, 2.0 / 3.0,
                     0.75, 0.8, 0.9):
            r = Rbig * frac
            fa = ((r + w) - r) / Rbig
            rows.append(dict(Rbig=Rbig, wall=r, position="middle",
                             frac=fa,
                             ulp=float((fa - EDGE) / np.spacing(EDGE)),
                             below=bool(sc._below(fa, EDGE))))
        fo = (Rbig - (Rbig - w)) / Rbig
        rows.append(dict(Rbig=Rbig, wall=Rbig, position="outer", frac=fo,
                         ulp=float((fo - EDGE) / np.spacing(EDGE)),
                         below=bool(sc._below(fo, EDGE))))
        rows.append(dict(Rbig=Rbig, wall=0.0, position="axis",
                         frac=(w - 0.0) / Rbig,
                         ulp=float(((w - 0.0) / Rbig - EDGE)
                                   / np.spacing(EDGE)),
                         below=bool(sc._below((w - 0.0) / Rbig, EDGE))))
    above = [r for r in rows if r["ulp"] > 0]
    above_out = [r for r in above if not r["below"]]
    by_rbig = {}
    for r in rows:
        by_rbig.setdefault(r["Rbig"], set()).add(r["below"])
    split = sorted(k for k, v in by_rbig.items() if len(v) > 1)
    summary = dict(
        rows=len(rows),
        rows_above_the_bar=len(above),
        rows_above_and_outside_the_deadband=len(above_out),
        worst_ulp_above=max([r["ulp"] for r in rows] or [0.0]),
        worst_ulp_below=min([r["ulp"] for r in rows] or [0.0]),
        deadband_reach_in_ulp=float(sc._BOR_FRAC_DEADBAND * EDGE
                                    / float(np.spacing(EDGE))),
        rbig_values_where_positions_split=split,
        examples=above_out[:8])
    for k in sorted(summary):
        if k != "examples":
            print(" ", k, summary[k])
    _vb3.dump("v3_repr", dict(rows=rows, summary=summary))


def _measure(bnd, walls, q, n_max, k0, Rbig):
    return sc.measure_layer(bnd, 0, walls, Rbig, q, n_max, k0)


def part_formed():
    """The ``q_measurable`` distinction, CONSTRUCTED rather than hoped for."""
    Rbig = 10.0
    w = 1e-7 * Rbig                      # a liner 1e-7 of Rbig at the axis
    bnd = np.array([0.0, w, Rbig])       # the mesh breakpoints
    walls = [[w]]                        # layer 0's own interior wall
    cases = {}
    for name, q, n_max, k0 in (
            ("no_modes", np.zeros(0, dtype=complex), 1.5, 2.0),
            ("zero_ceiling", np.array([1.0 + 0j, 2.0 + 0j]), 0.0, 2.0),
            ("zero_k0", np.array([1.0 + 0j, 2.0 + 0j]), 1.5, 0.0),
            ("formed_inf", np.array([np.inf + 0j, 1.0 + 0j]), 1.5, 2.0),
            ("formed_nan", np.array([np.nan + 0j, 1.0 + 0j]), 1.5, 2.0),
            ("formed_huge", np.array([1e12 + 0j]), 1.5, 2.0),
            ("formed_small", np.array([1.0 + 0j]), 1.5, 2.0)):
        rec = _measure(bnd, walls, q, n_max, k0, Rbig)
        cases[name] = dict(q_excess=float(rec["q_excess"]),
                           q_measurable=bool(rec["q_measurable"]),
                           w_min_own_frac=float(rec["w_min_own_frac"]),
                           verdict=sc.verdict(rec),
                           hot_expected=(name.startswith("formed_")
                                         and name != "formed_small"))
        print("  %-14s q_excess=%-12.6g meas=%-5s -> %s"
              % (name, cases[name]["q_excess"], cases[name]["q_measurable"],
                 cases[name]["verdict"]), flush=True)
    #: a record built WITHOUT the new key -- what an older or external payload
    #: looks like to ``verdict``, which is documented as a pure function of the
    #: record
    rec = _measure(bnd, walls, np.array([np.inf + 0j]), 1.5, 2.0, Rbig)
    legacy = dict(rec)
    legacy.pop("q_measurable", None)
    cases["legacy_record_no_key"] = dict(
        q_excess=float(legacy["q_excess"]), q_measurable=None,
        verdict=sc.verdict(legacy), hot_expected=False)
    print("  %-14s -> %s (no q_measurable key)"
          % ("legacy_record", cases["legacy_record_no_key"]["verdict"]))
    derived = {
        "never_formed_stays_cold": all(
            cases[k]["verdict"] == "ok"
            for k in ("no_modes", "zero_ceiling", "zero_k0")),
        "formed_nonfinite_is_hot": all(
            cases[k]["verdict"] != "ok"
            for k in ("formed_inf", "formed_nan")),
        "formed_finite_below_bar_stays_cold":
            cases["formed_small"]["verdict"] == "ok",
        "legacy_record_reverts_to_the_pre_round_reading":
            cases["legacy_record_no_key"]["verdict"] == "ok",
    }
    print(" ", derived)
    _vb3.dump("v3_formed", dict(summary=dict(cases=cases, derived=derived)))


def part_deadband():
    """``_below`` around the bar, in ULP -- and whether it is symmetric."""
    bar = EDGE
    out = []
    for k in (-64, -33, -32, -17, -16, -8, -1, 0, 1, 8, 16, 17, 32, 33, 64):
        x = bar
        for _ in range(abs(k)):
            x = np.nextafter(x, -np.inf if k < 0 else np.inf)
        out.append(dict(ulp=k, x=float(x),
                        below=bool(sc._below(x, bar)),
                        strict=bool(x < bar)))
    flips = [r["ulp"] for r in out if r["below"] != r["strict"]]
    widest_above = max([r["ulp"] for r in out if r["ulp"] > 0 and r["below"]]
                       or [0])
    summary = dict(
        deadband=float(sc._BOR_FRAC_DEADBAND),
        deadband_in_ulp_of_bar=float(sc._BOR_FRAC_DEADBAND * bar
                                     / float(np.spacing(bar))),
        rows=out, ulps_where_below_differs_from_strict=flips,
        widest_ulp_above_bar_still_below=widest_above,
        every_x_within_16_ulp_decides_below=all(
            r["below"] for r in out if abs(r["ulp"]) <= 16),
        nonfinite_is_not_below=[bool(sc._below(np.inf, bar)),
                                bool(sc._below(np.nan, bar))])
    for k in sorted(summary):
        if k != "rows":
            print(" ", k, summary[k])
    _vb3.dump("v3_deadband", dict(summary=summary))


PARTS = {"ladder": part_ladder, "repr": part_repr, "formed": part_formed,
         "deadband": part_deadband}

if __name__ == "__main__":
    a = _vb3.arm()
    _vb3.require_tree(a)
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"], flush=True)
    for name in (sys.argv[1:] or list(PARTS)):
        print("== PART", name, flush=True)
        PARTS[name]()

"""V4 -- the round-3 DEGRADATION-BAND warning, re-measured independently.

Five things, all on fixtures built here rather than read from the fix:

1. **the ladder** -- accuracy as a function of the narrowest segment, on a
   y-uniform 3-layer stack whose MIDDLE layer is ALL HOST (so the DEVICE
   cannot depend on the wall separation and every deviation is numerical
   damage), scored against the EXACT 1-D ``PMMStack`` at two degrees so the
   oracle's own self-gap is measured beside the errors it is used to compare;
2. **the census** -- the narrowest segment ORDINARY per-layer geometries ask
   for, over the builders the library exposes, and whether any of them warns;
3. **the message** -- does it name the width, the degradation class and the
   remedies;
4. **the switch** and **once per SOLVE, not per interface**;
5. **the tapers** -- ``add_tapered_pillar`` / ``add_tapered_pillars`` at
   typical slice counts, and which land in the band.

UNITS.  This fixture is DIMENSIONLESS (period 1.07, wavelength 0.79) -- the
1-D ``PMMStack`` oracle and the 2-D stack are built in the SAME units, which
is the only thing that matters.  Do not drive it with a metre wavelength.

Run: ``python v4_band.py <tag> [ladder|rest|all]``
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _path                                     # noqa: E402,F401,I001

import json                                                 # noqa: E402
import os                                                   # noqa: E402
import time                                                 # noqa: E402
import warnings                                             # noqa: E402

import numpy as np                                          # noqa: E402

import _vfix as F                                           # noqa: E402
from lumenairy.elements.pmm import PMM2DStackPure, PMMStack  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts    # noqa: E402
from lumenairy.elements.pmm.twod_staggered import _C        # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------- fixture
#: DIMENSIONLESS.  Independent of the fix's ladder fixture (period 0.93,
#: wavelength 0.66, theta 0.19, eps 6.25/2.1, walls .155/.585 and .315/.795,
#: sliver centre 0.41) in every one of those knobs.
PER, WL, TH = 1.07, 0.79, 0.31
EPSP, EPSH, TT = 9.0, 1.69, 0.17
W0 = (0.205, 0.495)
W2 = (0.365, 0.735)
YW = (0.185, 0.615)
CENTRE = 0.585                       # where the sliver pair sits (all host)


def _oracle(deg):
    st = PMMStack(PER, degree=deg, far_field_orders=5)
    # PMMStack segments are FRACTIONS of the period and must sum to 1 -- the
    # 2-D stack's x_walls are ABSOLUTE metres.  Mixing the two conventions is
    # the units trap this file's docstring warns about.
    st.add_layer(TT, segments=[(W0[0], EPSH), (W0[1] - W0[0], EPSP),
                               (1.0 - W0[1], EPSH)])
    st.add_layer(TT, segments=[(1.0, EPSH)])
    st.add_layer(TT, segments=[(W2[0], EPSH), (W2[1] - W2[0], EPSP),
                               (1.0 - W2[1], EPSH)])
    st.set_source(WL, theta=TH)
    o, R, T = st.solve(stabilize=None)[:3]
    return np.asarray(o), np.atleast_2d(R), np.atleast_2d(T)


def _tile():
    c = np.full((3, 3), _C(EPSH))
    c[1, :] = _C(EPSP)
    return c


def _band_stack(frac, M):
    """The 3-layer y-uniform device whose MIDDLE layer is all host, its two
    walls a relative distance ``frac`` apart about ``CENTRE``."""
    sw = [(CENTRE - frac / 2) * PER, (CENTRE + frac / 2) * PER]
    st = PMM2DStackPure(PER, n_modes=M, n_orders=1, layer_grids="per-layer")
    yws = [YW[0] * PER, YW[1] * PER]
    st.add_layer(TT, eps_cell=_tile(), x_walls=[W0[0] * PER, W0[1] * PER],
                 y_walls=yws)
    st.add_layer(TT, eps_cell=np.full((3, 3), _C(EPSH)), x_walls=sw,
                 y_walls=yws)
    st.add_layer(TT, eps_cell=_tile(), x_walls=[W2[0] * PER, W2[1] * PER],
                 y_walls=yws)
    st.set_source(WL, theta=TH, phi=0.0)
    return st


def ladder(rows, Ms=None,
           fracs=(3e-1, 2e-1, 1.5e-1, 1e-1, 7e-2, 5e-2, 3e-2, 2e-2, 1e-2,
                  3e-3, 1e-3)):
    if Ms is None:
        # V4_LADDER_MS trims the ladder for a second build: an M = 8 rung is
        # ~3 minutes a step on a 3-segment grid and eleven rungs of it is an
        # hour, so the second build confirms the M = 7 row (the one whose
        # ordinary arm is 11x the oracle's own self-gap) rather than all three.
        Ms = tuple(int(x) for x in
                   os.environ.get("V4_LADDER_MS", "6,7,8").split(","))
    o14, R14, T14 = _oracle(14)
    o12, R12, T12 = _oracle(12)
    keep = np.abs(o14) <= 1
    self_gap = float(max(np.max(np.abs(R12[:, keep] - R14[:, keep])),
                         np.max(np.abs(T12[:, keep] - T14[:, keep]))))
    print(f"  oracle self-gap deg12->14: {self_gap:.10e}", flush=True)
    rows.append({"what": "oracle_self_gap", "value": self_gap,
                 "R14_row1": [float(x) for x in R14[1, keep]]})
    for M in Ms:
        base = None
        for fr in fracs:
            t0 = time.time()
            st = _band_stack(fr, M)
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                o, R, T = st.solve(jones=False)
            o, R, T = np.asarray(o), np.atleast_2d(R), np.atleast_2d(T)
            worst = 0.0
            for m in (-1, 0, 1):
                sel = int(np.where((o[:, 0] == m) & (o[:, 1] == 0))[0][0])
                j = int(np.where(o14 == m)[0][0])
                worst = max(worst, abs(float(R[1, sel]) - float(R14[1, j])),
                            abs(float(T[1, sel]) - float(T14[1, j])))
            if base is None:
                base = worst
            wtxt = [str(w.message) for w in ws]
            rows.append({"what": "ladder", "M": M, "frac": fr, "err": worst,
                         "ratio": worst / base,
                         "warned": any("degradation band" in t for t in wtxt),
                         "n_warn": len(ws),
                         "seconds": round(time.time() - t0, 2)})
            print(f"  M={M} frac={fr:.1e} err={worst:.6e} "
                  f"ratio={worst / base:.3f} warn={len(ws)} "
                  f"{time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------- census
def _narrowest(st):
    """The narrowest segment fraction a stack's per-layer grids ask for,
    computed HERE from the stored wall SPECS rather than from the library's
    own band helper -- the helper is the thing under test.

    A spec is either an INTEGER segment count (a uniform lattice, every
    segment 1/N) or a full 0..period boundary array.
    """
    fr = 1.0
    for L in st._layers:
        for spec, d in ((L.get("wx"), st.period_x),
                        (L.get("wy"), st.period_y)):
            if spec is None:
                continue
            if np.ndim(spec) == 0:
                fr = min(fr, 1.0 / int(spec))
                continue
            b = np.asarray(spec, dtype=float)
            fr = min(fr, float(np.min(np.diff(b))) / float(d))
    return fr


def _solve_warnings(st):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        st.solve(jones=False)
    return [str(w.message) for w in ws]


def census(rows):
    """ORDINARY per-layer geometries, and the narrowest segment each asks
    for.  'Ordinary' = a geometry a user builds to describe a device, not one
    built to exercise a sliver."""
    P = 1.0
    out = []

    def _st(M=4, no=1):
        return PMM2DStackPure(P, n_modes=M, n_orders=no,
                              layer_grids="per-layer")

    # 1. uniform lattices (no walls at all)
    for N in (1, 2, 3, 4, 6, 8, 12, 16):
        st = _st()
        st.add_layer(0.1, eps_cell=np.full((N, N), _C(6.0)))
        st.add_layer(0.1, eps=2.0)
        st.set_source(0.8, theta=0.2, phi=0.3)
        out.append((f"uniform_lattice_N{N}", st))
    # 2. a duty-cycle pillar over the useful range
    for duty in (0.1, 0.2, 1 / 3, 0.5, 0.7, 0.9):
        a = 0.5 - duty / 2
        st = _st()
        st.add_layer(0.1, eps_cell=F.scalar_cell((a, a + duty), a, a + duty),
                     x_walls=[a * P, (a + duty) * P],
                     y_walls=[a * P, (a + duty) * P])
        st.add_layer(0.1, eps=2.0)
        st.set_source(0.8, theta=0.2, phi=0.3)
        out.append((f"duty_{duty:.3f}", st))
    # 3. a NESTED refinement -- a pillar inside a pillar
    w = (0.25, 0.375, 0.5, 0.75)
    st = _st()
    st.add_layer(0.1, eps_cell=F.scalar_cell(w, 0.25, 0.75),
                 x_walls=[x * P for x in w], y_walls=[x * P for x in w])
    st.add_layer(0.1, eps=2.0)
    st.set_source(0.8, theta=0.2, phi=0.3)
    out.append(("nested_refinement", st))
    # 4. two pillars side by side
    w = (0.1, 0.35, 0.6, 0.85)
    st = _st()
    st.add_layer(0.1, eps_cell=F.scalar_cell(w, 0.1, 0.35),
                 x_walls=[x * P for x in w], y_walls=[x * P for x in w])
    st.add_layer(0.1, eps=2.0)
    st.set_source(0.8, theta=0.2, phi=0.3)
    out.append(("two_pillars", st))
    # 5. TAPERS -- straight and CLOSING, at the slice counts a user picks
    for ns in (4, 8, 16, 32, 64, 128):
        for close, nm in ((False, "straight"), (True, "closing")):
            st = _st()
            st.add_tapered_pillar(
                0.3, eps_pillar=6.0, eps_host=2.0,
                x_bounds_bottom=(0.25 * P, 0.75 * P),
                y_bounds_bottom=(0.25 * P, 0.75 * P),
                x_bounds_top=((0.5 * P, 0.5 * P) if close
                              else (0.30 * P, 0.70 * P)),
                y_bounds_top=((0.5 * P, 0.5 * P) if close
                              else (0.30 * P, 0.70 * P)),
                n_slices=ns)
            st.set_source(0.8, theta=0.2, phi=0.3)
            out.append((f"taper_{nm}_{ns}", st))
    # 6. add_tapered_pillars, two features
    for ns in (4, 8, 16, 32):
        st = _st()
        st.add_tapered_pillars(
            0.3, eps_host=2.0, n_slices=ns,
            pillars=[((0.28 * P, 0.28 * P), (0.14 * P, 0.14 * P),
                      (0.20 * P, 0.20 * P), 6.0),
                     ((0.72 * P, 0.72 * P), (0.14 * P, 0.14 * P),
                      (0.20 * P, 0.20 * P), 6.0)])
        st.set_source(0.8, theta=0.2, phi=0.3)
        out.append((f"tapered_pillars_{ns}", st))

    edge = _ts._STAG_SLIVER_BAND_FRAC
    for name, st in out:
        fr = _narrowest(st)
        rows.append({"what": "census", "name": name, "narrowest": fr,
                     "in_band": bool(_ts._STAG_MIN_SEG_FRAC <= fr < edge),
                     "n_layers": len(st._layers)})
        print(f"  census {name:26s} narrowest={fr:.4e} "
              f"{'IN BAND' if _ts._STAG_MIN_SEG_FRAC <= fr < edge else ''}",
              flush=True)


# ------------------------------------------------- message / switch / count
def message_and_switch(rows):
    st = _band_stack(2.0e-2, 4)
    w = _solve_warnings(st)
    band = [t for t in w if "degradation band" in t]
    txt = band[0] if band else ""
    toks = {
        "names_the_width": "2.000e-02" in txt or "of the period" in txt,
        "names_the_layer": "layer 1" in txt,
        "names_the_axis": " x axis" in txt or "the x axis" in txt,
        "names_the_wall_array": "walls [" in txt,
        "names_the_contract": "1e-03" in txt,
        "names_the_band_edges": "3e-02" in txt,
        "names_degradation_class": ("about 4-5x" in txt or "about 5-6x" in txt
                                    or "about 6x" in txt),
        "says_floor_not_removed_by_n_modes": "FLOOR" in txt,
        "says_energy_blind": "no energy tripwire can see this" in txt,
        "remedy_merge": "(1) MERGE" in txt,
        "remedy_shared": "layer_grids='shared'" in txt,
        "remedy_conforming": "(3) put the NEIGHBOURS" in txt,
        "remedy_hybrid": "PMM2DStackHybrid" in txt,
        "remedy_slices": "lower n_slices" in txt,
        "names_the_switch": "PMM2D_STAG_SLIVER_BAND_WARN = False" in txt,
    }
    rows.append({"what": "message", "n_band_warnings": len(band),
                 "tokens": toks, "text": txt})
    print(f"  message tokens: {sum(toks.values())}/{len(toks)}; "
          f"missing {[k for k, v in toks.items() if not v]}", flush=True)

    # two-sided: silent OUTSIDE, warns INSIDE, refused BELOW
    for fr, exp in ((6.0e-2, "silent"), (2.9e-2, "warn"), (1.2e-3, "warn"),
                    (9.0e-4, "refused")):
        try:
            w = _solve_warnings(_band_stack(fr, 4))
            got = ("warn" if any("degradation band" in t for t in w)
                   else "silent")
            n = len([t for t in w if "degradation band" in t])
        except ValueError as e:
            got, n = "refused", 0
            w = [str(e)[:120]]
        rows.append({"what": "two_sided", "frac": fr, "expected": exp,
                     "got": got, "n_band_warnings": n})
        print(f"  two-sided frac={fr:.1e} expected={exp} got={got} n={n}",
              flush=True)

    # the SWITCH
    prev = _ts.PMM2D_STAG_SLIVER_BAND_WARN
    try:
        _ts.PMM2D_STAG_SLIVER_BAND_WARN = False
        off = [t for t in _solve_warnings(_band_stack(2.0e-2, 4))
               if "degradation band" in t]
    finally:
        _ts.PMM2D_STAG_SLIVER_BAND_WARN = prev
    on = [t for t in _solve_warnings(_band_stack(2.0e-2, 4))
          if "degradation band" in t]
    rows.append({"what": "switch", "with_switch_off": len(off),
                 "with_switch_on": len(on)})
    print(f"  switch: off={len(off)} on={len(on)}", flush=True)

    # ONCE PER SOLVE, not per interface: FOUR layers, THREE of them in the
    # band, on FOUR distinct grids -> six mortared interfaces
    P = 1.07
    st = PMM2DStackPure(P, n_modes=4, n_orders=1, layer_grids="per-layer")
    for c, fr in ((0.20, 2.0e-2), (0.45, 1.5e-2), (0.70, 1.0e-2),
                  (0.85, 5.0e-3)):
        sw = [(c - fr / 2) * P, (c + fr / 2) * P]
        st.add_layer(0.08, eps_cell=np.full((3, 3), _C(2.5)), x_walls=sw,
                     y_walls=[0.3 * P, 0.6 * P])
    st.set_source(0.79, theta=0.31, phi=0.0)
    w = _solve_warnings(st)
    band = [t for t in w if "degradation band" in t]
    rows.append({"what": "once_per_solve", "n_layers": 4,
                 "n_band_warnings": len(band),
                 "which_layer": ([t.split("layer ")[1].split("'")[0]
                                  for t in band] if band else [])})
    print(f"  once_per_solve: {len(band)} band warning(s) for 4 banded "
          f"layers", flush=True)

    # a CONFORMING per-layer stack in the band must be SILENT (no V2 shape)
    P = 1.07
    sw = [(0.585 - 5e-3 / 2) * P, (0.585 + 5e-3 / 2) * P]
    for name, conf in (("conforming", True), ("non_conforming", False)):
        st = PMM2DStackPure(P, n_modes=4, n_orders=1, layer_grids="per-layer")
        st.add_layer(0.08, eps_cell=np.full((3, 3), _C(2.5)), x_walls=sw,
                     y_walls=[0.3 * P, 0.6 * P])
        st.add_layer(0.08, eps_cell=np.full((3, 3), _C(3.5)),
                     x_walls=(sw if conf else [0.31 * P, 0.66 * P]),
                     y_walls=[0.3 * P, 0.6 * P])
        st.set_source(0.79, theta=0.31, phi=0.0)
        w = [t for t in _solve_warnings(st) if "degradation band" in t]
        rows.append({"what": "conforming", "name": name,
                     "n_band_warnings": len(w)})
        print(f"  conforming[{name}]: {len(w)}", flush=True)

    # a Y-AXIS sliver SHARED by every layer, non-conformity on X only: the
    # y mortar is then the identity, so is the warning a FALSE POSITIVE?
    yws = [(0.585 - 5e-3 / 2) * P, (0.585 + 5e-3 / 2) * P]
    st = PMM2DStackPure(P, n_modes=4, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.08, eps_cell=np.full((3, 3), _C(2.5)),
                 x_walls=[0.25 * P, 0.60 * P], y_walls=yws)
    st.add_layer(0.08, eps_cell=np.full((3, 3), _C(3.5)),
                 x_walls=[0.31 * P, 0.66 * P], y_walls=yws)
    st.set_source(0.79, theta=0.31, phi=0.0)
    w = [t for t in _solve_warnings(st) if "degradation band" in t]
    rows.append({"what": "y_axis_shared_sliver", "n_band_warnings": len(w),
                 "axis_named": ("y axis" in w[0] if w else None)})
    print(f"  y_axis_shared_sliver: {len(w)} warning(s)", flush=True)


def main(tag, what="all"):
    rows = []
    if what in ("all", "rest"):
        print("CENSUS", flush=True)
        census(rows)
        print("MESSAGE / SWITCH / COUNT", flush=True)
        message_and_switch(rows)
    if what in ("all", "ladder"):
        print("LADDER", flush=True)
        ladder(rows)
    out = {"env": F.env(), "tag": tag, "what": what,
           "band_edge": _ts._STAG_SLIVER_BAND_FRAC,
           "contract": _ts._STAG_MIN_SEG_FRAC, "rows": rows}
    (HERE / f"v4_band_{tag}_{what}.json").write_text(
        json.dumps(out, indent=1), encoding="cp1252")
    print(f"wrote v4_band_{tag}_{what}.json ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "win",
         sys.argv[2] if len(sys.argv) > 2 else "all")

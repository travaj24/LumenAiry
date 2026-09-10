"""ROUND 4 / DEFECT 2 -- the band warning, per AXIS.

Round 3 conditions the degradation-band warning on the STACK building a
cross-grid interface *somewhere*, then scans BOTH axes of every grid for the
narrowest segment.  A stack whose layers differ on x and share the y wall
array EXACTLY therefore warns about a narrow y segment, on which the mortar is
the identity and nothing is projected across grids.

This probe measures the fix TWO-SIDEDLY, on the SAME script run against the
tree before and after the change (``--tag win_pre`` / ``--tag win_post``):

  (a) NO-MORTAR AXIS -- must stop warning, and its answer must not move a bit;
  (b) REAL warnings -- the closing taper at 9..64 slices and the round-2 delta
      sweep must warn IDENTICALLY (same count, same axis, same width);
  (c) MIXED -- x carries a mortar with a banded segment while y is conforming
      with a NARROWER banded segment: pre names y, post must name x, once;
  (d) CENSUS -- the ordinary geometries, 0 warnings before and after;
  (e) BIT-IDENTITY -- sha256 of (orders, R, T) on every fixture that solves.

Usage::

    python validation/probe_fix_mortar_round4/p1_axis_band.py --tag win_pre
"""
# ``_path`` MUST be imported before anything that touches lumenairy, so
# this block is deliberately not isort-ordered.
from __future__ import annotations  # noqa: I001

import _path  # noqa: F401  (pins THIS worktree's lumenairy)  (MUST be first: pins this worktree's lumenairy)

import argparse                                             # noqa: E402
import hashlib                                              # noqa: E402
import json                                                 # noqa: E402
import pathlib                                              # noqa: E402
import sys                                                  # noqa: E402
import time                                                 # noqa: E402
import warnings                                             # noqa: E402

import numpy as np                                          # noqa: E402

import lumenairy                                            # noqa: E402
from lumenairy.elements.pmm import PMM2DStackPure           # noqa: E402
from lumenairy.elements.pmm import stack2d_pure as _sp      # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts    # noqa: E402
from lumenairy.elements.pmm.twod_staggered import _C        # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


# ---- the ROUND-3 arm, reproduced IN PROCESS -------------------------------
# Round 3 called ``_warn_stag_sliver_band(gof, force_mortar or
# len({g.key() for g in gof}) > 1)`` and then scanned BOTH axes.  The full key
# is the fingerprint PAIR, so ``len({key}) > 1`` is exactly "at least one axis
# differs" -- i.e. ``any(_stag_mortared_axes(...))``.  Replacing the per-axis
# helper by the collapsed form therefore reproduces the round-3 behaviour
# EXACTLY, and it does so in the SAME process as the round-4 arm, which is
# what makes the bit-identity comparison hold everything else fixed.
class round3_emulation:
    def __init__(self, on=True):
        self.on = on

    def __enter__(self):
        if not self.on:
            return self
        self._orig = _sp._stag_mortared_axes

        def collapsed(grids, force=False):
            m = bool(force) or len({g.key() for g in grids}) > 1
            return (m, m)

        _sp._stag_mortared_axes = collapsed
        _ts._stag_mortared_axes = collapsed
        return self

    def __exit__(self, *a):
        if self.on:
            _sp._stag_mortared_axes = self._orig
            _ts._stag_mortared_axes = self._orig
        return False

# ---- the verifier's own band fixture, reproduced knob for knob -------------
# (VERIFY_PMM2D_MORTAR_ROUND3 S8 DEFECT 2 / v5_falsepos.py fp2)
_VP, _VWL, _VTH = 1.07, 0.79, 0.31
_YC = 0.585


def _sha(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:32]


def _band_warnings(ws):
    return [w for w in ws
            if issubclass(w.category, UserWarning)
            and "degradation band" in str(w.message)]


def _axis_of(msg):
    for ax in ("x", "y"):
        if f" on the {ax} axis" in msg:
            return ax
    return None


def _width_of(msg):
    # "... has a segment 9.000e-03 of the period wide on the x axis"
    tok = msg.split("has a segment ", 1)[1].split(" of the period", 1)[0]
    return float(tok)


def _solved(st, *, guard=True):
    """Solve, recording the band warnings and hashing the answer."""
    prev = _ts.PMM2D_STAG_MIN_SEG_GUARD
    if not guard:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            o, R, T = st.solve(jones=False)
            dt = time.perf_counter() - t0
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
    o, R, T = np.asarray(o), np.atleast_2d(R), np.atleast_2d(T)
    bw = _band_warnings(ws)
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    return {
        "hash": _sha(o, R, T),
        "R00": float(R[1, p0]),
        "closure": float(abs(np.sum(R) + np.sum(T) - R.shape[0])),
        "n_warn": len(bw),
        "axes": [_axis_of(str(w.message)) for w in bw],
        "widths": [_width_of(str(w.message)) for w in bw],
        "seconds": round(dt, 3),
    }


# ==========================================================================
# (a) the NO-MORTAR axis
# ==========================================================================
def _no_mortar_axis_stack(fy, M=5, *, xconf=False):
    """The VERIFICATION's DEFECT-2 fixture, knob for knob
    (``v5_falsepos.py fp2``): uniform permittivity in both layers, layers
    differing on x and sharing the y wall array EXACTLY.  The device cannot
    depend on the y wall separation, and the y mortar is the identity.
    ``xconf`` makes x conforming too, so the stack builds no mortar at all."""
    yws = [(_YC - fy / 2) * _VP, (_YC + fy / 2) * _VP]
    st = PMM2DStackPure(_VP, n_modes=M, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.17, eps_cell=np.full((3, 3), _C(2.5)),
                 x_walls=[0.25 * _VP, 0.60 * _VP], y_walls=yws)
    st.add_layer(0.17, eps_cell=np.full((3, 3), _C(3.5)),
                 x_walls=([0.25 * _VP, 0.60 * _VP] if xconf
                          else [0.31 * _VP, 0.66 * _VP]),
                 y_walls=yws)
    st.set_source(_VWL, theta=_VTH, phi=0.0)
    return st


def part_a():
    out = {}
    for M in (4, 5):
        for xconf in (False, True):
            rows = []
            for fy in (3.0e-1, 1.0e-1, 5.0e-2, 3.0e-2, 1.0e-2, 3.0e-3,
                       1.2e-3):
                r = _solved(_no_mortar_axis_stack(fy, M, xconf=xconf))
                r["fy"] = fy
                rows.append(r)
            wide = rows[0]
            for r in rows:
                r["rel_vs_widest"] = (abs(r["R00"] - wide["R00"])
                                      / abs(wide["R00"]))
                r["ratio_vs_widest"] = r["R00"] / wide["R00"]
            out[f"M{M}_xconf{int(xconf)}"] = rows
    return out


# ==========================================================================
# (b) the REAL warnings -- they must survive untouched
# ==========================================================================
def _closing_taper_stack(n_slices, M=4):
    """A taper whose tip CLOSES, built through the same rule the shipped
    builder uses: the midpoint of slice ``i`` of ``n`` carries a pillar of
    width ``w_bottom (1 - (i + 0.5) / n)``, so its narrowest sampled width is
    ``w_bottom / (2 n)``.  Each slice is its OWN grid, so x carries a mortar;
    the y wall array is SHARED and ordinary."""
    st = PMM2DStackPure(_VP, n_modes=M, n_orders=1, layer_grids="per-layer")
    yws = [0.27 * _VP, 0.61 * _VP]
    w_bottom = 0.5
    for i in range(n_slices):
        w = w_bottom * (1.0 - (i + 0.5) / n_slices)
        xws = [(0.5 - w / 2) * _VP, (0.5 + w / 2) * _VP]
        tile = np.full((3, 3), _C(2.1))
        tile[1, 1] = _C(6.0)
        st.add_layer(0.04, eps_cell=tile, x_walls=xws, y_walls=yws)
    st.set_source(_VWL, theta=_VTH, phi=0.0)
    return st


def _delta_sweep_stack(delta, M=5):
    """The round-2 delta sweep (``test_the_mortars_own_algebra_is_exact_at_
    every_wall_separation``), in its own units: three ALL-HOST layers, x walls
    DIFFERING (the middle one carrying the sliver), y walls SHARED."""
    P, WL, TH = 1.2, 0.85, 0.23
    st = PMM2DStackPure(P, n_modes=M, n_orders=1, layer_grids="per-layer")
    yw = [0.27 * P, 0.61 * P]
    for xw in ([0.21 * P, 0.68 * P],
               [(0.5 - delta / 2) * P, (0.5 + delta / 2) * P],
               [0.33 * P, 0.79 * P]):
        st.add_layer(0.06, eps=2.25, x_walls=xw, y_walls=yw)
    st.set_source(WL, theta=TH, phi=0.0)
    return st


def part_b():
    out = {"taper": [], "delta": []}
    # the closing taper: 8 slices is 1.04x OUTSIDE the edge, 9 is inside
    for n in (8, 9, 16, 32):
        f = 0.5 / (2 * n)
        r = _solved(_closing_taper_stack(n))
        r["n_slices"] = n
        r["narrowest_expected"] = f
        out["taper"].append(r)
    # the round-2 sweep: 0.30 is ordinary, 1e-2 is in the band, both on x
    for d in (0.30, 1.0e-2):
        r = _solved(_delta_sweep_stack(d), guard=False)
        r["delta"] = d
        out["delta"].append(r)
    return out


# ==========================================================================
# (c) MIXED -- both axes narrow, only x mortared
# ==========================================================================
def _mixed_stack(fx, fy, M=4, *, share_y=True):
    """x walls DIFFER between the two layers and carry a segment ``fx`` wide;
    y walls are IDENTICAL in both layers and carry a NARROWER segment ``fy``.
    Pre-fix the warning names y (the global narrowest); the axis that actually
    carries a cross-grid projection is x."""
    yws = [(_YC - fy / 2) * _VP, (_YC + fy / 2) * _VP]
    yws2 = yws if share_y else [(0.40 - fy / 2) * _VP, (0.40 + fy / 2) * _VP]
    x0 = [(0.44 - fx / 2) * _VP, (0.44 + fx / 2) * _VP]
    tile = np.full((3, 3), _C(2.1))
    tile[1, 1] = _C(6.0)
    st = PMM2DStackPure(_VP, n_modes=M, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.10, eps_cell=tile, x_walls=x0, y_walls=yws)
    st.add_layer(0.10, eps_cell=tile,
                 x_walls=[0.31 * _VP, 0.66 * _VP], y_walls=yws2)
    st.set_source(_VWL, theta=_VTH, phi=0.0)
    return st


def part_c():
    out = {}
    # x banded at 9e-3 (mortared), y banded NARROWER at 3e-3 (conforming)
    r = _solved(_mixed_stack(9.0e-3, 3.0e-3))
    r["fx"], r["fy"] = 9.0e-3, 3.0e-3
    out["x_mortared_y_conforming"] = r
    # the CONTROL: the same widths with y NOT shared, so both axes are
    # mortared -- the narrowest (y) is the right answer before AND after
    r2 = _solved(_mixed_stack(9.0e-3, 3.0e-3, share_y=False))
    r2["fx"], r2["fy"] = 9.0e-3, 3.0e-3
    out["both_axes_mortared"] = r2
    # x ORDINARY, y conforming and banded: nothing should warn after the fix
    r3 = _solved(_mixed_stack(2.0e-1, 3.0e-3))
    r3["fx"], r3["fy"] = 2.0e-1, 3.0e-3
    out["x_ordinary_y_conforming"] = r3
    return out


# ==========================================================================
# (d) the ordinary CENSUS -- geometry only, no solve
# ==========================================================================
def _narrowest_per_axis(st, M=4):
    """Read the narrowest segment of a built stack per AXIS, whether that axis
    carries a cross-grid interface, and the worst REGION eigenproblem the
    stack would need -- all WITHOUT solving.

    The per-axis liveness is computed HERE, from the fingerprint PAIR
    :meth:`StagGridOps.key` returns, rather than through the library's own
    helper, which is the thing under test."""
    from lumenairy.elements.pmm.twod_staggered import StagGridOps  # noqa: PLC0415
    gof = [StagGridOps(st.period_x, st.period_y, L["wx"], L["wy"], M,
                       1.0 + 0j, 1.0 + 0j) for L in st._layers]
    keys = [g.key() for g in gof]
    live = (len({k[0] for k in keys}) > 1, len({k[1] for k in keys}) > 1)
    worst, worst_live, eig = 1.0, 1.0, 0
    for g in gof:
        eig = max(eig, 2 * g.qq)
        for j, b in enumerate((g.bx, g.by)):
            f = (1.0 / float(b.N) if b.uniform
                 else float(np.min(np.diff(np.asarray(b.xb)))) / float(b.d))
            worst = min(worst, f)
            if live[j]:
                worst_live = min(worst_live, f)
    return worst, worst_live, live, int(eig)


def _census_geometries():
    """The two censuses that already exist in this tree, built through the
    PUBLIC builders and reused rather than re-invented:

      * the ROUND-2 battery (15 geometries) from
        ``tests/unit/test_fix_pmm2d_mortar_round2.py``;
      * the ROUND-3 VERIFICATION's 32-geometry census from
        ``validation/probe_verify_mortar_round3/v4_band.py`` (uniform lattices
        N = 1..16, duty 0.1..0.9 pillars, a nested refinement, two pillars,
        straight and CLOSING tapers at 4..128 slices, ``add_tapered_pillars``
        at 4..32).
    """
    out = []
    sys.path.insert(0, str(HERE.parents[1] / "tests" / "unit"))
    import test_fix_pmm2d_mortar_round2 as r2  # noqa: PLC0415
    for name, st in sorted(r2._shipped_geometry_battery().items()):
        out.append(("r2:" + name, st, r2._WL, r2._TH, r2._PH))
    sys.path.insert(0, str(HERE.parents[0] / "probe_verify_mortar_round3"))
    import v4_band as v4  # noqa: PLC0415
    # ``census`` builds its stacks locally and appends only geometry rows, so
    # SPY on the one function it calls per stack rather than editing it.
    seen, orig = [], v4._narrowest
    v4._narrowest = lambda st: (seen.append(st), orig(st))[1]
    try:
        rows = []
        v4.census(rows)
    finally:
        v4._narrowest = orig
    assert len(seen) == len(rows), (len(seen), len(rows))
    for st, r in zip(seen, rows):
        out.append(("v3:" + r["name"], st, None, None, None))
    return out


def part_d(solve_budget=8, eig_cap=1200):
    edge, lo = _ts._STAG_SLIVER_BAND_FRAC, _ts._STAG_MIN_SEG_FRAC
    geoms = _census_geometries()
    rows, keep = [], {}
    for name, st, wl, th, ph in geoms:
        w, wl_live, live, eig = _narrowest_per_axis(st)
        rows.append({
            "name": name, "narrowest": w,
            "narrowest_on_a_mortared_axis": wl_live,
            "x_mortared": bool(live[0]), "y_mortared": bool(live[1]),
            "in_band_any_axis": bool(lo <= w < edge),
            "in_band_mortared_axis": bool(lo <= wl_live < edge),
            "n_layers": len(st._layers), "worst_region_eig": eig,
        })
        keep[name] = (st, wl, th, ph)
        print(f"  {name:28s} narrowest={w:.4e} mortared={wl_live:.4e} "
              f"live={live} eig={eig}", flush=True)
    ordinary = [r for r in rows if "closing" not in r["name"]]
    # SOLVE the ordinary classes that sit CLOSEST to the edge -- a geometry
    # reading is not a warning count.  ``eig_cap`` keeps the dense REGION
    # eigenproblem inside a shared box: a 16-cell uniform lattice at M = 4 is
    # 2 q^2 = 4608 and is minutes a solve, which is exactly the reason the
    # verification BOUNDS that class by arithmetic instead of running it.
    solved, skipped = {}, []
    close = sorted(ordinary, key=lambda r: r["narrowest"])
    for r in close:
        if len(solved) >= solve_budget:
            break
        if r["worst_region_eig"] > eig_cap or r["n_layers"] > 8:
            skipped.append({"name": r["name"], "eig": r["worst_region_eig"],
                            "n_layers": r["n_layers"],
                            "narrowest": r["narrowest"]})
            continue
        st, wl, th, ph = keep[r["name"]]
        if wl is not None:              # the round-2 battery carries no source
            st.set_source(wl, theta=th, phi=ph)
        solved[r["name"]] = _solved(st)
        print(f"  SOLVED {r['name']:26s} warn={solved[r['name']]['n_warn']} "
              f"{solved[r['name']]['seconds']} s", flush=True)
    return {
        "rows": rows,
        "n_geometries": len(rows),
        "n_ordinary": len(ordinary),
        "ordinary_in_band_any_axis": sum(r["in_band_any_axis"]
                                         for r in ordinary),
        "ordinary_in_band_mortared_axis":
            sum(r["in_band_mortared_axis"] for r in ordinary),
        "narrowest_ordinary": min(r["narrowest"] for r in ordinary),
        "solved": solved,
        "solved_total_warnings": sum(v["n_warn"] for v in solved.values()),
        "skipped_too_expensive": skipped,
        "edge": edge,
    }


def _walk(o, path=""):
    """Flatten a result tree to ``{dotted path: leaf}``."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, o


def _compare(pre, post):
    """Every leaf, PRE (round 3) against POST (round 4).  Reported in three
    buckets: warning-set differences (the INTENDED change), answer-hash
    differences (there must be NONE) and everything else that moved."""
    a, b = dict(_walk(pre)), dict(_walk(post))
    keys = sorted(set(a) | set(b))
    warn, hashes, other = [], [], []
    for k in keys:
        if a.get(k, "<missing>") == b.get(k, "<missing>"):
            continue
        row = {"path": k, "round3": a.get(k, "<missing>"),
               "round4": b.get(k, "<missing>")}
        if k.endswith(".hash"):
            hashes.append(row)
        elif (".n_warn" in k or ".axes" in k or ".widths" in k
              or "warning" in k or "in_band" in k):
            warn.append(row)
        elif k.endswith(".seconds") or "_seconds" in k:
            continue
        else:
            other.append(row)
    return {"n_leaves": len(keys), "answer_hash_differences": hashes,
            "warning_differences": warn, "other_differences": other}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--parts", default="abcd")
    a = ap.parse_args()
    out = {
        "tag": a.tag,
        "lumenairy": str(pathlib.Path(lumenairy.__file__).resolve()),
        "version": lumenairy.__version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "band": [_ts._STAG_MIN_SEG_FRAC, _ts._STAG_SLIVER_BAND_FRAC],
    }
    fns = {"a": part_a, "b": part_b, "c": part_c, "d": part_d}
    for arm, emulate in (("round3", True), ("round4", False)):
        out[arm] = {}
        for p in a.parts:
            t0 = time.perf_counter()
            print(f"=== {arm} part {p} ===", flush=True)
            with round3_emulation(emulate):
                out[arm][p] = fns[p]()
            out[arm][p + "_seconds"] = round(time.perf_counter() - t0, 2)
            print(f"{arm} part {p}: {out[arm][p + '_seconds']} s", flush=True)
    out["compare"] = _compare(out["round3"], out["round4"])
    f = HERE / f"p1_axis_band_{a.tag}.json"
    f.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="cp1252")
    print("wrote", f)
    c = out["compare"]
    print(f"leaves compared           : {c['n_leaves']}")
    print(f"ANSWER HASH differences   : {len(c['answer_hash_differences'])}")
    print(f"warning differences       : {len(c['warning_differences'])}")
    print(f"other differences         : {len(c['other_differences'])}")
    for row in c["warning_differences"]:
        print(f"  WARN {row['path']}: {row['round3']} -> {row['round4']}")
    for row in c["other_differences"][:40]:
        print(f"  OTHER {row['path']}: {row['round3']} -> {row['round4']}")


if __name__ == "__main__":
    main()

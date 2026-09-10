"""INDEPENDENT verification of ROUND 4 -- the per-AXIS band warning.

Round 4 conditions the degradation-band warning on whether THE AXIS carrying
the narrowest segment actually carries a cross-grid projection, where round 3
asked the question of the STACK and then scanned both axes.  Round 4's own
probe measures this by monkey-patching the helper inside ONE interpreter.
This one does it the other way: the SAME file is run against TWO TREES --
``15af675`` (pre) and ``b7239bf`` (post) -- selected by ``VMORTAR4_TREE``, in
separate interpreters, on both builds.  Any answer difference between the two
trees is a defect; the warning differences are enumerated and classified.

PARTS

  ``f``  FAMILY -- 32 per-layer fixtures (conforming; non-conforming on x
         only / y only / both; mixed in-plane-out-of-plane; magnetic; slant;
         closing tapers at 8/9/16 slices; the four ADVERSARIAL stacks of task
         B), plus 3 solved again with ``force_mortar=True``.  Every answer is
         hashed; every band warning is recorded with its axis and width.
  ``g``  GEOMETRY -- for every fixture, the narrowest segment per axis, which
         axes carry a mortar (computed HERE from ``StagGridOps.key`` rather
         than through the library helper under test), and the DECISION the
         round-3 and round-4 rules each take.  No solve.
  (the LADDERS that ask whether a narrow segment on a given axis is actually
  damaging live in ``v3_ladder.py``, which needs only ONE tree.)
  ``w``  ``window_halfwidth`` -- the interaction task B(iv) asks about.

Usage::

    VMORTAR4_TREE=C:/tmp/lum_vmortar4 python .../v1_axis_identity.py \
        --tag win_post --parts fgw
"""
# ``_path`` MUST be imported before anything that touches lumenairy, so this
# block is deliberately not isort-ordered.
from __future__ import annotations  # noqa: I001

import _path  # noqa: F401  (MUST be first: pins the tree under measurement)

import argparse                                             # noqa: E402
import hashlib                                              # noqa: E402
import json                                                 # noqa: E402
import pathlib                                              # noqa: E402
import time                                                 # noqa: E402
import warnings                                             # noqa: E402

import numpy as np                                          # noqa: E402

import _vfix4 as F                                          # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts     # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

#: the band edges, read from the tree under measurement rather than pinned
BAND_LO = float(_ts._STAG_MIN_SEG_FRAC)
BAND_HI = float(_ts._STAG_SLIVER_BAND_FRAC)


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
    tok = msg.split("has a segment ", 1)[1].split(" of the period", 1)[0]
    return float(tok)


def _solve(st, *, force=False):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        if force:
            out = st._solve_per_layer(jones=False, retain_internal=False,
                                      force_mortar=True)
        else:
            out = st.solve(jones=False)
        dt = time.perf_counter() - t0
    o, R, T = (np.asarray(out[0]), np.atleast_2d(out[1]),
               np.atleast_2d(out[2]))
    bw = _band_warnings(ws)
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    return {
        "hash": _sha(o, R, T),
        "R00": float(R[1, p0]),
        "R00_row0": float(R[0, p0]),
        "T00": float(T[1, p0]),
        "closure": float(abs(np.sum(R) + np.sum(T) - R.shape[0])),
        "n_warn": len(bw),
        "axes": [_axis_of(str(w.message)) for w in bw],
        "widths": [_width_of(str(w.message)) for w in bw],
        "seconds": round(dt, 3),
    }


# ==========================================================================
# g -- the GEOMETRY reading, and the two RULES, computed HERE
# ==========================================================================
def _grids(st, M):
    from lumenairy.elements.pmm.twod_staggered import (  # noqa: PLC0415
        StagGridOps,
    )
    return [StagGridOps(st.period_x, st.period_y, L["wx"], L["wy"], M,
                        1.0 + 0j, 1.0 + 0j) for L in st._layers]


def _frac(b):
    if b.uniform:
        return 1.0 / float(b.N)
    return float(np.min(np.diff(np.asarray(b.xb)))) / float(b.d)


def _rules(st, M, *, force=False):
    """The narrowest segment per axis, which axes are mortared (from the
    fingerprint PAIR, computed HERE), and what each RULE decides."""
    gof = _grids(st, M)
    keys = [g.key() for g in gof]
    live = (force or len({k[0] for k in keys}) > 1,
            force or len({k[1] for k in keys}) > 1)
    per_axis = {"x": 1.0, "y": 1.0}
    owner = {"x": None, "y": None}
    for gi, g in enumerate(gof):
        for ax, b in (("x", g.bx), ("y", g.by)):
            f = _frac(b)
            if f < per_axis[ax]:
                per_axis[ax], owner[ax] = f, gi
    # ROUND 3: any axis differs anywhere -> scan BOTH axes
    r3_on = bool(live[0] or live[1])
    r3_f, r3_ax = min(((per_axis["x"], "x"), (per_axis["y"], "y")))
    # ROUND 4: scan only the LIVE axes
    cands = [(per_axis[a], a) for a, on in (("x", live[0]), ("y", live[1]))
             if on]
    r4_f, r4_ax = min(cands) if cands else (1.0, None)

    def _decide(on, f, ax):
        if not on or ax is None:
            return {"warn": False, "axis": None, "width": None}
        inband = BAND_LO * (1 - 1e-9) <= f < BAND_HI * (1 - 1e-9)
        return {"warn": bool(inband),
                "axis": ax if inband else None,
                "width": f if inband else None}

    return {
        "n_layers": len(gof),
        "narrowest_x": per_axis["x"], "narrowest_y": per_axis["y"],
        "owner_x": owner["x"], "owner_y": owner["y"],
        "narrowest_all": min(per_axis["x"], per_axis["y"]),
        "mortared_axes": [bool(live[0]), bool(live[1])],
        "round3": _decide(r3_on, r3_f, r3_ax),
        "round4": _decide(bool(cands), r4_f, r4_ax),
        "per_axis_ge_all_axes": bool(r4_f >= min(per_axis["x"],
                                                 per_axis["y"]) - 0.0),
        "library_axes": list(map(bool, _ts._stag_mortared_axes(gof, force)))
        if hasattr(_ts, "_stag_mortared_axes") else None,
    }


def part_g(M=4):
    out = {}
    for name in F.FAMILY:
        out[name] = _rules(F.build(name, M), M)
    for name in F.FORCED:
        out[name + "|forced"] = _rules(F.build(name, M), M, force=True)
    return out


def part_f(M=4):
    out = {}
    for name in F.FAMILY:
        r = _solve(F.build(name, M))
        out[name] = r
        print(f"  {name:28s} warn={r['n_warn']} {r['axes']} {r['widths']} "
              f"R00={r['R00']:.12f} {r['seconds']:.1f}s", flush=True)
    for name in F.FORCED:
        r = _solve(F.build(name, M), force=True)
        out[name + "|forced"] = r
        print(f"  {name + '|forced':28s} warn={r['n_warn']} {r['axes']} "
              f"{r['widths']} R00={r['R00']:.12f} {r['seconds']:.1f}s",
              flush=True)
    return out


# ==========================================================================
# w -- window_halfwidth (task B(iv))
# ==========================================================================
def part_w():
    from lumenairy.elements.pmm import PMM2DStackPure  # noqa: PLC0415
    out = {}
    for hw in (1, 2, None):
        try:
            PMM2DStackPure(F.P, n_modes=4, n_orders=1,
                           layer_grids="per-layer", window_halfwidth=hw)
            out[str(hw)] = {"accepted": True, "message": None}
        except Exception as exc:                            # noqa: BLE001
            out[str(hw)] = {"accepted": False,
                            "message": str(exc)[:400],
                            "type": type(exc).__name__}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--parts", default="fg")
    ap.add_argument("--M", type=int, default=4)
    a = ap.parse_args()
    res = {"tag": a.tag, "M": a.M, "env": F.env(),
           "tree": _path.TREE, "lumenairy_file": _path.LUMENAIRY_FILE,
           "band_lo": BAND_LO, "band_hi": BAND_HI}
    for p, fn in (("g", lambda: part_g(a.M)), ("f", lambda: part_f(a.M)),
                  ("w", part_w)):
        if p in a.parts:
            t0 = time.perf_counter()
            print(f"part {p} ...", flush=True)
            res[p] = fn()
            print(f"part {p}: {time.perf_counter() - t0:.2f} s", flush=True)
    dest = HERE / f"v1_axis_identity_{a.tag}.json"
    dest.write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf8")
    print("wrote", dest)


if __name__ == "__main__":
    main()

"""Is a band-width segment on a NON-mortared axis actually harmless?

This is the DECISION round 4 makes.  The fix measures it on a fixture whose
two layers are UNIFORM slabs (permittivity 2.5 and 3.5) with fictitious walls;
on such a device the modal content is nearly plane-wave and NOTHING moves --
including on the MORTARED axis -- so that measurement cannot separate "this
axis is harmless" from "this device is harmless".  ``_vfix4.ladder`` reproduces
that shape and this probe scores it, but the DECIDING population is
``_vfix4.ladder_p``: the swept wall is still a PURE PARTITION choice (the
permittivity does not vary across it, so the exact answer cannot depend on it)
while the stack is genuinely patterned, so the mortar carries real modal
content and the POSITIVE control moves.

Five shapes, each swept over 2.4 decades of the segment width:

  ``Xmort``    the swept axis IS mortared                  POSITIVE control
  ``Xconf_y``  the swept axis CONFORMS, the other is mortared -- the shape
               round 3 warned about and round 4 silences
  ``Xnomort``  no mortar anywhere                          NEGATIVE control
  ``Ymort``    the transpose of ``Xmort``                  POSITIVE control
  ``Yconf_x``  the transpose of ``Xconf_y`` -- the MIRROR the fix never
               measured (round 3 warns x, round 4 is silent)

Usage::

    VMORTAR4_TREE=C:/tmp/lum_vmortar4 python .../v3_ladder.py --tag win
"""
# ``_path`` MUST be imported before anything that touches lumenairy, so this
# block is deliberately not isort-ordered.
from __future__ import annotations  # noqa: I001

import _path  # noqa: F401  (MUST be first: pins the tree under measurement)

import argparse                                             # noqa: E402
import json                                                 # noqa: E402
import pathlib                                              # noqa: E402
import time                                                 # noqa: E402
import warnings                                             # noqa: E402

import numpy as np                                          # noqa: E402

import _vfix4 as F                                          # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def _run(st):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        o, R, T = st.solve(jones=False)
        dt = time.perf_counter() - t0
    o, R, T = np.asarray(o), np.atleast_2d(R), np.atleast_2d(T)
    bw = [w for w in ws if "degradation band" in str(w.message)]
    ax = []
    for w in bw:
        m = str(w.message)
        ax.append("x" if " on the x axis" in m else
                  "y" if " on the y axis" in m else None)
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    return {"R": R, "T": T, "R00": float(R[1, p0]),
            "closure": float(abs(np.sum(R) + np.sum(T) - R.shape[0])),
            "n_warn": len(bw), "axes": ax, "seconds": round(dt, 2)}


def _ladder(builder, kind, Ms, gs):
    out = {}
    for M in Ms:
        rows = []
        for g in gs:
            r = _run(builder(kind, g, M))
            r["g"] = g
            rows.append(r)
            print(f"  {kind:9s} M={M} g={g:.1e} warn={r['n_warn']}{r['axes']}"
                  f" R00={r['R00']:.12f} ({r['seconds']:.1f}s)", flush=True)
        wide = rows[0]
        for r in rows:
            r["abs_move"] = float(max(np.max(np.abs(r["R"] - wide["R"])),
                                      np.max(np.abs(r["T"] - wide["T"]))))
            r["rel_R00"] = abs(r["R00"] - wide["R00"]) / abs(wide["R00"])
        scale = float(max(np.max(np.abs(wide["R"])), np.max(np.abs(wide["T"]))))
        rows_j = [{k: v for k, v in r.items() if k not in ("R", "T")}
                  for r in rows]
        out[f"M{M}"] = {
            "rows": rows_j,
            "scale": scale,
            "max_abs_move": max(r["abs_move"] for r in rows_j),
            "abs_move_at_3e-2": next(r["abs_move"] for r in rows_j
                                     if abs(r["g"] - 3.0e-2) < 1e-12),
            "abs_move_at_1.2e-3": next(r["abs_move"] for r in rows_j
                                       if abs(r["g"] - 1.2e-3) < 1e-12),
            "n_warn_at_1.2e-3": next(r["n_warn"] for r in rows_j
                                     if abs(r["g"] - 1.2e-3) < 1e-12),
            "axes_at_1.2e-3": next(r["axes"] for r in rows_j
                                   if abs(r["g"] - 1.2e-3) < 1e-12),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--Ms", default="4,6,8")
    ap.add_argument("--kinds", default=",".join(F.LADDERS_P))
    ap.add_argument("--trivial", action="store_true",
                    help="also run the uniform-slab ladders of _vfix4.ladder")
    ap.add_argument("--fixdev", action="store_true",
                    help="also run the FIX's own DEFECT-2 device, swept on "
                         "the mortared axis as well as on its own")
    a = ap.parse_args()
    Ms = tuple(int(x) for x in a.Ms.split(","))
    res = {"tag": a.tag, "env": F.env(), "tree": _path.TREE,
           "lumenairy_file": _path.LUMENAIRY_FILE, "Ms": list(Ms),
           "gs": list(F.LADDER_G), "patterned": {}, "trivial": {},
           "fixdev": {}}
    for kind in a.kinds.split(","):
        if kind:
            res["patterned"][kind] = _ladder(F.ladder_p, kind, Ms,
                                             F.LADDER_G)
    if a.trivial:
        for kind in F.LADDERS:
            res["trivial"][kind] = _ladder(F.ladder, kind, Ms, F.LADDER_G)
    if a.fixdev:
        for kind in F.FIXDEV:
            res["fixdev"][kind] = _ladder(F.fixdev, kind, (4, 5),
                                          F.LADDER_G)
    dest = HERE / f"v3_ladder_{a.tag}.json"
    dest.write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf8")
    print("wrote", dest)
    for fam in ("patterned", "trivial", "fixdev"):
        for kind, d in res[fam].items():
            for mk, v in d.items():
                print(f"{fam:9s} {kind:9s} {mk}: max_abs_move="
                      f"{v['max_abs_move']:.3e} at3e-2="
                      f"{v['abs_move_at_3e-2']:.3e} at1.2e-3="
                      f"{v['abs_move_at_1.2e-3']:.3e} "
                      f"warn={v['n_warn_at_1.2e-3']}{v['axes_at_1.2e-3']}")


if __name__ == "__main__":
    main()

"""E1-10 -- cost: the mapped OUT-OF-PLANE region against Phase D's mapped
IN-PLANE region on the same 3 x 3 circle map, and against the shipped
unmapped out-of-plane region on the same (3 x 3) walls.

Arms (one region each; M from the command line; best of ``reps``):
* inplane_mapped  -- the LC30 disk (block-form) under the c3 map: Phase D's
  second-order pencil, assembly + QZ region eig (_region_modes);
* oop_mapped      -- the OOP30 disk under the c3 map: the Phase E1
  first-order generator, assembly + whitened eig + flux split
  (_region_modes_oop);
* oop_slant_mapped-- the same, slanted (0.2, 0) (the composite map);
* oop_unmapped    -- the OOP30 pillar on the shipped 3 x 3 walls (no map).
Recorded: assembly time, eig time, total, and the peak traced allocation
(tracemalloc) of the whole region solve.

usage: python e10_cost.py <M> [reps]        writes e10_cost_M<M>.json
"""
import sys
import time
import tracemalloc

import _e1common as E
import e5_pillar as E5
import numpy as np

from lumenairy.elements.rcwa._core import uniaxial_tensor

TS = E.TS
P = 1.2
LC30 = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=np.pi / 6)


def arm(name, M):
    if name == "inplane_mapped":
        cm, eps = E5.disk_cells("c3", LC30)
        kw = dict(cmap=cm)
        walls = (cm.u_walls, cm.v_walls)
    elif name in ("oop_mapped", "oop_slant_mapped"):
        cm, eps = E5.disk_cells("c3", E.OOP30)
        kw = dict(cmap=cm)
        if name == "oop_slant_mapped":
            kw["slant"] = (0.2, 0.0)
        walls = (cm.u_walls, cm.v_walls)
    else:
        cm, eps = E5.disk_cells("c3", E.OOP30)
        walls = (cm.u_walls, cm.v_walls)
        kw = {}
    tracemalloc.start()
    t0 = time.perf_counter()
    s = TS.Granet2DTransverseE(P, P, walls[0], walls[1], M, eps,
                               k0=2 * np.pi, **kw)
    t1 = time.perf_counter()
    if s.offplane:
        TS._region_modes_oop(s)
    else:
        TS._region_modes(s)
    t2 = time.perf_counter()
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"assembly": t1 - t0, "eig": t2 - t1, "total": t2 - t0,
            "peak_MB": peak / 2 ** 20, "dim": int(s.dimtot)}


def main(M, reps):
    out = {"M": M, "reps": reps}
    for name in ("inplane_mapped", "oop_mapped", "oop_slant_mapped",
                 "oop_unmapped"):
        runs = [arm(name, M) for _ in range(reps)]
        best = min(runs, key=lambda r: r["total"])
        best["assembly_best"] = min(r["assembly"] for r in runs)
        best["eig_best"] = min(r["eig"] for r in runs)
        out[name] = best
        print(name, {k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in best.items()}, flush=True)
    for k in ("oop_mapped", "oop_slant_mapped", "oop_unmapped"):
        out[f"ratio_{k}_vs_inplane_mapped"] = (
            out[k]["total"] / out["inplane_mapped"]["total"])
    out["ratio_oop_mapped_vs_oop_unmapped"] = (
        out["oop_mapped"]["total"] / out["oop_unmapped"]["total"])
    out["peak_ratio_oop_mapped_vs_inplane_mapped"] = (
        out["oop_mapped"]["peak_MB"] / out["inplane_mapped"]["peak_MB"])
    E.dump(f"e10_cost_M{M}.json", out)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 2)

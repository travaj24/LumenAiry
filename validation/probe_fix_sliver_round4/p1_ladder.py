"""ROUND 4, P1 -- the LADDER, per arm.

For every (fixture, degree, delta) row of the sliver ladder, measured with the
guard DISARMED so nothing the library decides feeds back:

  * ``err``      the distance from the exact ``delta -> 0`` reference, and the
                 continuity class ``right``/``grey``/``wrong`` that follows;
  * ``worst``    max ``R+T`` -- the reading rounds 1-3 triggered and arbitrated
                 on, and the quantity CI showed to be arm-dependent;
  * ``d0``       |A_sliver - A_snap1| on the PRESCRIBED grid (mf = 2 w_wide P)
                 -- the existing ``_sliver_answer_move`` metric;
  * ``d12``      |A_snap1 - A_snap2| between the prescribed grid and a SECOND
                 sliver-free snap (mf = 4 w_wide P) -- the device's own
                 sensitivity to where the wall is put, at the same scale;
  * ``su1``      the snapped super-unity (round 2/3's closure input).

``d0`` and ``d12`` are the round-4 arbiter's two inputs.  Everything else is
recorded so the round-2 and round-3 criteria can be re-scored from the same
rows.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_fixtures as G  # noqa: E402
import numpy as np  # noqa: E402

G.assert_tree()

DEGREES = (12, 14, 20)
DELTAS = [float(x) for x in np.geomspace(3e-3, 1e-6, 22)]


def row(name, cfg, deg, d, ref):
    st = G.wbuild(d, deg, **cfg)
    base = G.unguarded(st)
    err = G.shared_move(base, ref, pol=1)
    pre = G.prescribed(st)
    rec = dict(fixture=name, degree=deg, delta=d, err=err,
               err_over_delta=(err / d if d else None),
               kind=G.classify(err, d), worst=base["worst"])
    if pre is None:
        rec["screened"] = False
        return rec
    rec["screened"] = True
    rec.update(w_wide=pre["w_wide"], w_narrow=pre["w_narrow"],
               own=pre["own"], n_hit=pre["n_hit"], mf1=pre["mf"])
    P = float(st.period)
    s1 = G.snapped(st, pre["mf"])
    s2 = G.snapped(st, 4.0 * pre["w_wide"] * P)
    rec.update(su1=max(s1["worst"] - 1.0, 0.0),
               su2=max(s2["worst"] - 1.0, 0.0),
               d0=G.shared_move(base, s1), d12=G.shared_move(s1, s2),
               d0_pol1=G.shared_move(base, s1, pol=1),
               err_snap1=G.shared_move(s1, ref, pol=1),
               err_snap2=G.shared_move(s2, ref, pol=1))
    rec["d0_over_w"] = rec["d0"] / pre["w_wide"]
    rec["d12_over_w"] = rec["d12"] / pre["w_wide"]
    rec["d0_over_d12"] = (rec["d0"] / rec["d12"]) if rec["d12"] > 0 else float("inf")
    rec["v2"] = G.verdict_round2(base["worst"], rec["su1"], rec["d0"], pre["w_wide"])
    rec["v3"] = G.verdict_round3(base["worst"], rec["su1"], rec["d0"],
                                 pre["w_wide"], 1e-2)
    return rec


def main():
    out = []
    for name, cfg in G.FIXTURES.items():
        for deg in DEGREES:
            ref = G.unguarded(G.wbuild(0.0, deg, **cfg))
            for d in DELTAS:
                out.append(row(name, cfg, deg, d, ref))
        print(name, "done", flush=True)
    G.dump(dict(rows=out, degrees=list(DEGREES), deltas=DELTAS), "p1_ladder")


if __name__ == "__main__":
    main()

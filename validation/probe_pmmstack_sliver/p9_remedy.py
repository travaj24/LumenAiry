"""P9 -- the PRESCRIBED remedy, scored against the EXACT reference.

The refusal prescribes ``min_feature = 2 * w_widest_sliver * period``.  This
runs it and scores the snapped answer against the ``delta -> 0`` limit -- the
two layers with IDENTICAL walls, which is an exact reference because the
physical structure is continuous in ``delta``.  The BAR is derived from the
structure's own continuity: the measured shift is ``err = 1.15 delta`` over
four decades of ``delta``, so a snapped answer that lands within ``2 delta``
of the limit has paid only the geometric perturbation the snap describes.
"""
import json
import os
import warnings

import numpy as np
from p1_repro import PX, frames, orc

import lumenairy
from lumenairy.elements.pmm import stack as ps

print("lumenairy:", lumenairy.__file__, flush=True)
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))


def prescribed_mf(d, deg):
    """The min_feature the refusal names, read out of the guard itself."""
    st = ps.PMMStack(PX, degree=deg)
    for (a, b) in frames(d):
        st.add_layer(0.08e-6,
                     segments=[(a, 2.25), (b - a, 9.0), (1.0 - b, 2.25)])
    hit = ps._cross_layer_sliver([L[1] for L in st._layers],
                                 st.min_feature / PX)
    return None if hit is None else 2.0 * hit[3] * PX


if __name__ == "__main__":
    rows = []
    print(f"{'deg':>4} {'delta':>9} {'min_feature':>12} {'err vs 0':>10} "
          f"{'bar 2*delta':>12} {'err/delta':>10} {'R+T':>9} {'pass':>5}",
          flush=True)
    for deg in (12, 14, 16, 20):
        ref = orc(frames(0.0), deg)
        for d in (1e-4, 5e-5, 3e-5, 1e-5, 3e-6):
            mf = prescribed_mf(d, deg)
            # mf is None when the LIBRARY DEFAULT min_feature already snapped
            # the pair -- there is no sliver left to prescribe against.
            M, R, T = orc(frames(d), deg, min_feature=mf)
            err = float(max(np.abs(R - ref[1]).max(),
                            np.abs(T - ref[2]).max()))
            tot = float(R.sum() + T.sum())
            ok = bool(err <= 2.0 * d and abs(tot - 1.0) < 1e-6)
            rows.append(dict(degree=deg, delta=d, min_feature=mf, err=err,
                             bar=2.0 * d, ratio=err / d, total=tot, ok=ok))
            mft = "default" if mf is None else f"{mf:.4g}"
            print(f"{deg:4d} {d:9.2e} {mft:>12} {err:10.3e} {2.0 * d:12.3e} "
                  f"{err / d:10.3f} {tot:9.6f} {str(ok):>5}", flush=True)
        json.dump(rows, open(os.path.join(HERE, "p9_remedy.json"), "w"),
                  indent=1)
    print("\nall pass:", all(r["ok"] for r in rows), flush=True)
    print("wrote p9_remedy.json", flush=True)

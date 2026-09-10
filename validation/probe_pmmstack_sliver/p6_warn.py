"""P6 -- is O-11 SILENT?  Which shipped warnings fire on the broken rows.

O-11 was logged as the silent-wrongness class.  That reading came from the
2-D mortar arm's closure (1.6e-08).  This records what the 1-D ``PMMStack``
itself emits and what its OWN ``R+T`` reads, at the library default
``min_feature`` and with the snap disabled.
"""
import json
import os
import warnings

import numpy as np

import lumenairy
from p1_repro import PX, frames, orc

print("lumenairy:", lumenairy.__file__, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
MF_OFF = PX * 1e-10


def run(d, deg, mf):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        M, R, T = orc(frames(d), deg, min_feature=mf)
    msgs = sorted({str(w.message).split(" -- ")[0][:60] for w in rec})
    return M, R, T, msgs


if __name__ == "__main__":
    out = []
    for tag, mf in (("default", None), ("snap-off", MF_OFF)):
        for deg in (12, 14):
            ref = orc(frames(0.0), deg, min_feature=(mf or PX * 1e-5))
            print(f"--- {tag}, degree {deg} " + "-" * 40, flush=True)
            for d in (1e-3, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5, 5.544e-06):
                M, R, T, msgs = run(d, deg, mf)
                err = float(max(np.abs(R - ref[1]).max(),
                                np.abs(T - ref[2]).max()))
                tot = float(R.sum() + T.sum())
                out.append(dict(tag=tag, degree=deg, delta=d, err=err,
                                total=tot, warnings=msgs))
                print(f"  d={d:9.2e} err={err:9.2e} R+T={tot:9.4f}  "
                      f"warnings={msgs}", flush=True)
            json.dump(out, open(os.path.join(HERE, "p6_warn.json"), "w"),
                      indent=1)
    print("\nwrote p6_warn.json", flush=True)

"""TASK E -- diffraction_eme at the resolution where the branch decision is
LIVE (Nx = 96, above the onset the scoping put at Nx > 64)."""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import _vh; import ve_fix  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--build", required=True)
ap.add_argument("--tag", required=True); a = ap.parse_args()
import lumenairy

print("lumenairy.__file__ =", lumenairy.__file__)
_vh.require_tree(a.build)
from lumenairy.elements.eme import eme_diffraction as ed

out = dict(task="E", claim="3-e2e96", build=a.build, tag=a.tag, arm=_vh.arm())
rows = []
for name, Nx, imag in (("eme96_real", 96, 0.0), ("eme96_1e30", 96, 1e-30),
                       ("eme96_1e20", 96, 1e-20), ("eme96_1e12", 96, 1e-12)):
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ed.diffraction_eme(ve_fix.strips_two(Nx, 1.0, imag), 1.0, Nx, 1.0,
                                 20 * np.pi, 1.0, 1.0, 0.35, 1, 1, kx0=0.0,
                                 ky0=0.37, qz2_window=(12000.0, 16000.0),
                                 Ny=24, n_scan=80)
    rows.append(dict(name=name, secs=round(time.time() - t0, 1),
                     hash_r=_vh.hash_arrays(res["r"]),
                     hash_t=_vh.hash_arrays(res["t"]),
                     hash_qz2=_vh.hash_arrays(np.asarray(res["qz2"])),
                     energy=float(res["energy"]),
                     qz2=[float(v) for v in res["qz2"]]))
    print("  %-12s energy=%.9f  qz2[0]=%.6f  hr=%s  %.1fs"
          % (name, rows[-1]["energy"], rows[-1]["qz2"][0],
             rows[-1]["hash_r"][:12], rows[-1]["secs"]))
out["e2e96"] = rows
_vh.dump(HERE / ("ve_e2e96_%s.json" % a.tag), out)

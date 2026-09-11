"""TASK E -- END-TO-END answers that ride on the branch decision:
``eme_diffraction.diffraction_eme`` (layer_modes + mode_field + mode_match, so
the selector is exercised three times over) and a vector layer solve."""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _vh
import ve_fix  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True); ap.add_argument("--tag", required=True)
a = ap.parse_args()
import lumenairy

print("lumenairy.__file__ =", lumenairy.__file__)
_vh.require_tree(a.build)
from lumenairy.elements.eme import eme_2d_vector as ev
from lumenairy.elements.eme import eme_diffraction as ed

out = dict(task="E", claim="3-e2e", build=a.build, tag=a.tag, arm=_vh.arm())
rows = []
for name, Nx, imag, win in (("eme_Nx48_real", 48, 0.0, (9000.0, 16000.0)),
                            ("eme_Nx48_1e30", 48, 1e-30, (9000.0, 16000.0)),
                            ("eme_Nx48_1e12", 48, 1e-12, (9000.0, 16000.0)),
                            ("eme_Nx64_real", 64, 0.0, (9000.0, 16000.0)),
                            ("eme_Nx64_1e30", 64, 1e-30, (9000.0, 16000.0))):
    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ed.diffraction_eme(ve_fix.strips_two(Nx, 1.0, imag), 1.0, Nx,
                                     1.0, 20 * np.pi, 1.0, 1.0, 0.35, 1, 1,
                                     kx0=0.0, ky0=0.37, qz2_window=win,
                                     Ny=24, n_scan=90)
        rows.append(dict(name=name, secs=round(time.time() - t0, 1),
                         n_modes=int(len(res["qz2"])),
                         hash_r=_vh.hash_arrays(res["r"]),
                         hash_t=_vh.hash_arrays(res["t"]),
                         hash_qz2=_vh.hash_arrays(np.asarray(res["qz2"])),
                         energy=float(res["energy"]),
                         qz2=[float(v) for v in res["qz2"]],
                         R=[float(v) for v in res["R"]],
                         T=[float(v) for v in res["T"]], err=None))
        print("  %-16s n=%2d energy=%.9f  %.1fs" % (name, rows[-1]["n_modes"],
                                                    rows[-1]["energy"],
                                                    rows[-1]["secs"]))
    except Exception as exc:  # noqa: BLE001
        rows.append(dict(name=name, err="%s: %s" % (type(exc).__name__, exc),
                         secs=round(time.time() - t0, 1)))
        print("  %-16s RAISES %s" % (name, rows[-1]["err"][:90]))
out["diffraction_eme"] = rows

vrows = []
for name, Nx, imag in (("vlayer_real", 24, 0.0), ("vlayer_1e30", 24, 1e-30),
                       ("vlayer_1e12", 24, 1e-12), ("vlayer_gain", 24, -1e-6)):
    t0 = time.time()
    strips = [(ve_fix.eps_split(Nx, lo=2.25, hi=6.25, imag=imag), 0.5),
              (ve_fix.eps_centre(Nx, lo=2.25, hi=6.25, imag=imag), 0.5)]
    try:
        q = np.asarray(ev.layer_vector_modes(strips, 1.0, Nx, 1.0, 8.0,
                                             (20.0, 140.0), kx0=0.0, ky0=0.3,
                                             n_scan=60))
        vrows.append(dict(name=name, n=int(q.size), secs=round(time.time()-t0,1),
                          hash_q=_vh.hash_arrays(q),
                          q=[float(v) for v in q], err=None))
        print("  %-14s n=%2d  %.1fs" % (name, q.size, vrows[-1]["secs"]))
    except Exception as exc:  # noqa: BLE001
        vrows.append(dict(name=name, err="%s: %s" % (type(exc).__name__, exc)))
        print("  %-14s RAISES %s" % (name, vrows[-1]["err"][:90]))
out["layer_vector_modes"] = vrows
_vh.dump(HERE / ("ve_e2e_%s.json" % a.tag), out)

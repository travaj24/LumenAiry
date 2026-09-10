"""Per-kernel readings for the STRIPE mode-match degeneracy fixture.

The subject is
``tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_stripe_fixture_is_free_of_the_mode_match_degeneracy``,
which failed on the 5.45.0 slow-gate shard 1 with ``DID NOT WARN``.  Its
NEGATIVE arm asks the library to warn ("lossless energy closure violated")
while it walks an index-coincident cell up a truncation ladder.  A warning is
a DECISION, and this probe measures whether that decision -- and the closure
defect underneath it -- moves with the BLAS micro-kernel.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_CORETYPE=NEHALEM python probe_stripe_degeneracy.py

Prints one line per arm: the CLEAN arm's worst closure defect, the DEGENERATE
arm's worst, how many of the 16 truncations each arm keeps sound, the ratio
the test asserts, and whether the tripwire warned.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from lumenairy.elements.rcwa import rcwa_jones_1d_segments  # noqa: E402

# the gate file's own fixture constants, verbatim
PX = 0.7e-6
WL = 1.0e-6
DEPTH = 0.5e-6
_STRIPE_EPS_GROOVE = 2.10
_DEGENERATE_EPS_GROOVE = 2.25          # == n_sub^2 == the director's n_o^2
_ONED_SOUND_CLOSURE = 1e-9
_LADDER = range(11, 42, 2)


def _rot(psi, no, ne):
    c, s = np.cos(psi), np.sin(psi)
    d = np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex)
    Rm = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rm @ d @ Rm.T


def _arm(eps_groove, er):
    worst, sound, warned = 0.0, 0, False
    per_order = {}
    for n in _LADDER:
        eg = np.diag([eps_groove] * 3).astype(complex)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            _o, R1, T1, _J = rcwa_jones_1d_segments(
                PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0,
                n_orders=n)
        d = abs(float(np.sum(R1) + np.sum(T1) - 2.0))
        per_order[n] = d
        worst = max(worst, d)
        if d < _ONED_SOUND_CLOSURE:
            sound += 1
        if any("lossless energy closure violated" in str(w.message)
               for w in ws):
            warned = True
    return worst, sound, warned, per_order


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    er = _rot(np.deg2rad(35.0), 1.5, 2.3)
    c_worst, c_sound, c_warn, c_per = _arm(_STRIPE_EPS_GROOVE, er)
    d_worst, d_sound, d_warn, d_per = _arm(_DEGENERATE_EPS_GROOVE, er)
    ratio = d_worst / max(c_worst, 1e-13)

    arch = "unknown"
    try:
        import threadpoolctl  # noqa: I001, PLC0415
        for d in threadpoolctl.threadpool_info():
            if d.get("internal_api") in ("openblas", "mkl"):
                arch = str(d.get("architecture") or "unknown")
                break
    except Exception:                                  # noqa: BLE001
        pass
    build = "WSL" if sys.platform.startswith("linux") else "WIN"
    doc = {
        "arm": "%s-%s" % (build, arch),
        "coretype_requested": os.environ.get("OPENBLAS_CORETYPE", ""),
        "clean_worst": c_worst, "clean_sound_of_16": c_sound,
        "clean_warned": c_warn,
        "degenerate_worst": d_worst, "degenerate_sound_of_16": d_sound,
        "degenerate_warned": d_warn,
        "ratio": ratio,
        "test_would": ("PASS" if (c_worst < _ONED_SOUND_CLOSURE and d_warn
                                  and ratio > 1e5) else "FAIL"),
        "clean_per_order": c_per, "degenerate_per_order": d_per,
    }
    print("%-18s clean %.3e (%d/16 sound)  degenerate %.3e (%d/16 sound)  "
          "ratio %.2e  warned %s  -> %s"
          % (doc["arm"], c_worst, c_sound, d_worst, d_sound, ratio,
             d_warn, doc["test_would"]))
    if args.out:
        with open(args.out, "w", encoding="cp1252", errors="replace") as fh:
            fh.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The PRE-5.45.0 branch body on the stripe ladder, per (build, kernel) arm.

This probe exists because the FIX for one kernel-dependent test introduced
another.  The restated
``test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_stripe_fixture_is_free_of_the_mode_match_degeneracy``
reconstructs the mode-match degeneracy by reinstating the pre-5.45.0
``_sqrt_decay`` body, and its first version asserted that ZERO of the 16
truncation rungs stay sound under that body -- which is what the Haswell arm
reads.  It is not what the others read.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_CORETYPE=NEHALEM python probe_prefix_branch_ladder.py

MEASURED 2026-09-11, worst ``|sum R + sum T - 2|`` over ``n_orders`` 11..41
and how many of those 16 rungs stay under 1e-09:

    arm              clean          coincident      rungs sound   warns
    WIN-Haswell      2.083e-13      2.761e-02          0/16         16
    WIN-Sandybridge  3.020e-13      4.690e-02          1/16         14
    WIN-Nehalem      2.958e-13      8.158e-03          2/16         14
    WIN-Katmai       4.008e-13      7.289e-04          5/16         11
    WSL-Haswell      1.861e-13      5.001e-02          0/16         16
    WSL-Sandybridge  3.069e-13      4.802e-02          1/16         14
    WSL-Nehalem      2.918e-13      2.955e-03          2/16         12
    WSL-Katmai       4.035e-13      1.584e-03          5/16         11

WHICH rungs land on the wrong side of a rounding-level degeneracy is decided
by the kernel, so the COUNT is a reading and not a decision.  What survives
every arm is the coincident WORST -- never below 7.289e-04, seven decades
above the clean population's 4.6e-13 ceiling -- and the fact that the tripwire
fires at all.  Those are what the test asserts now.

Note also the ORDERING: the coincident worst falls monotonically as the kernel
gets older (Haswell 2.8-5.0e-02 -> Katmai 0.7-1.6e-03), i.e. the narrower the
SIMD width the less the degeneracy amplifies.  That is consistent with the
defect being a rounding-level branch choice amplified by an ill-conditioned
mode match, and it is the reason a bar placed on the Haswell reading has no
margin on an older kernel.
"""
import importlib
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from lumenairy.backend.array import array_namespace  # noqa: E402
from lumenairy.elements.rcwa import rcwa_jones_1d_segments  # noqa: E402

PX, WL, DEPTH = 0.7e-6, 1.0e-6, 0.5e-6
_SOUND = 1e-9
_LADDER = range(11, 42, 2)
_BOUND = ("lumenairy.elements.rcwa._core", "lumenairy.elements.rcwa.oned",
          "lumenairy.elements.rcwa.stack", "lumenairy.elements.pmm.twod",
          "lumenairy.elements.berreman")


def _pre(x, xp=None, band=1e-8):
    """The pre-5.45.0 body: the EXACT ``Re(r) == 0`` pin and the ``-r`` flip."""
    if xp is None:
        xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    return xp.where((r.real == 0) & (r.imag < 0), -r, r)


def _rot(psi, no, ne):
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


def _worst(eps_groove, er):
    worst, sound, nwarn = 0.0, 0, 0
    for n in _LADDER:
        eg = np.diag([eps_groove] * 3).astype(complex)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            _o, R, T, _J = rcwa_jones_1d_segments(
                PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0,
                n_orders=n)
        d = abs(float(np.sum(R) + np.sum(T) - 2.0))
        worst = max(worst, d)
        sound += d < _SOUND
        nwarn += any("lossless energy closure violated" in str(w.message)
                     for w in ws)
    return worst, sound, nwarn


def main():
    er = _rot(np.deg2rad(35.0), 1.5, 2.3)
    arch = "unknown"
    try:
        import threadpoolctl  # noqa: I001, PLC0415
        for d in threadpoolctl.threadpool_info():
            if d.get("internal_api") in ("openblas", "mkl"):
                arch = str(d.get("architecture"))
                break
    except Exception:                                  # noqa: BLE001
        pass
    build = "WSL" if sys.platform.startswith("linux") else "WIN"

    saved = []
    for name in _BOUND:
        mod = importlib.import_module(name)
        if hasattr(mod, "_sqrt_decay"):
            saved.append((mod, mod._sqrt_decay))
            mod._sqrt_decay = _pre
    try:
        pc = _worst(2.10, er)
        pd = _worst(2.25, er)
    finally:
        for mod, fn in saved:
            mod._sqrt_decay = fn

    print("%-18s PRE-FIX clean %.3e (%d/16) warn=%d | coincident %.3e (%d/16) "
          "warn=%d | ratio %.2e"
          % ("%s-%s" % (build, arch), pc[0], pc[1], pc[2], pd[0], pd[1], pd[2],
             pd[0] / max(pc[0], 1e-13)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

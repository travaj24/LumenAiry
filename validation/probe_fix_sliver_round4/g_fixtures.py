"""ROUND 4 -- shared helpers for the build-independence measurements.

The geometry and the scoring conventions are ROUND 3's: this module imports
``validation/probe_fix_sliver_round3/f_fixtures.py`` rather than restating it,
so a round-4 number and a round-3 number are measurements of the same devices
under the same continuity rule and can be compared directly.

What round 4 ADDS is the ARM: every record carries the (build, BLAS kernel)
it was measured on, because the defect this round repairs is that the guard's
decision was a property of that pair.
"""
import os
import platform
import sys

#: THE THREAD LADDER (2026-09-11, coordinator correction).  CI's fast unit
#: lane leaves BLAS UNPINNED on 4-core runners, and this family's own history
#: says the thread count is a first-class hazard -- ``test_m1_conditioning_
#: guard.py`` records a closure moving from 6.65e-06 to 2.14e+01 between ONE
#: and TWO OpenBLAS threads on the SAME cell.  So the probes must be runnable
#: unpinned as well as pinned, and the pin cannot be unconditional.
#:
#: ``PMM_PROBE_UNPINNED=1`` REMOVES the three variables, and then this module
#: imports numpy IMMEDIATELY -- before ``f_fixtures`` -- because the load of
#: ``libopenblas`` is the one and only moment the variables are read.  Round
#: 3's ``f_fixtures`` re-applies its own unconditional ``setdefault`` at ITS
#: top, ahead of ITS numpy import, so importing it first silently re-pins the
#: run to one thread; that is exactly what the first unpinned arm measured
#: (``threads_req UNPINNED`` but ``threads 1``) and why the order below is
#: load-bearing rather than cosmetic.  :data:`THREADS_REQUESTED` is likewise
#: snapshotted HERE, because ``f_fixtures`` will put ``"1"`` back into
#: ``os.environ`` a few lines down and an arm must not mis-report itself.
if os.environ.get("PMM_PROBE_UNPINNED") == "1":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.pop(_v, None)
else:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

#: what the ARM asked for, read before anything can overwrite it
THREADS_REQUESTED = os.environ.get("OPENBLAS_NUM_THREADS", "UNPINNED")

#: ``PMM_PROBE_AFFINITY=<n>`` restricts this process to the first ``n`` CPUs
#: BEFORE numpy is imported.  That is what makes the UNPINNED arm mean what
#: CI means by it: an unpinned OpenBLAS sizes its thread pool from the CPUs it
#: can SEE, so on CI's four-core runner "unpinned" is four threads, while on
#: this 24-thread box it is twenty-four -- and twenty-four threads on
#: spectral-element eigenproblems a few hundred wide spend their time in
#: thread launch (measured: the first full unpinned arm had not finished its
#: first case group after 17 minutes, against 2 minutes for the whole pinned
#: table).  Narrowing the affinity reproduces CI's configuration instead of
#: an extreme neither CI nor any user runs.
def _set_affinity(n):
    try:
        if hasattr(os, "sched_setaffinity"):
            cpus = sorted(os.sched_getaffinity(0))[:n]
            os.sched_setaffinity(0, set(cpus))
        else:                                          # Windows
            import psutil
            proc = psutil.Process()
            proc.cpu_affinity(sorted(proc.cpu_affinity())[:n])
    except Exception as exc:                           # pragma: no cover
        raise SystemExit("PMM_PROBE_AFFINITY=%r could not be applied: %r"
                         % (n, exc))


AFFINITY = os.environ.get("PMM_PROBE_AFFINITY")
if AFFINITY:
    _set_affinity(int(AFFINITY))

import numpy as np  # noqa: E402,F401  (loads libopenblas under the env above)

_HERE = os.path.dirname(os.path.abspath(__file__))
#: The tree this probe belongs to, pinned from ``__file__`` and put FIRST on
#: ``sys.path`` (round-3 verification defect V-3): this box carries an
#: EDITABLE install of ``lumenairy`` pointing at a different checkout, and
#: ``python <probe>.py`` puts the PROBE directory on ``sys.path[0]`` and not
#: the working directory, so without this line a probe silently measures the
#: OTHER checkout.  :func:`assert_tree` refuses to run against one.
ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if sys.path[:1] != [ROOT]:
    sys.path.insert(0, ROOT)
_R3 = os.path.join(os.path.dirname(_HERE), "probe_fix_sliver_round3")
if _R3 not in sys.path:
    sys.path.insert(1, _R3)
#: the round-3 VERIFICATION's fixtures -- the GMR / Fabry-Perot / near-Wood /
#: grazing mounts and the rotated directors round 4 has to decide correctly on
_V3 = os.path.join(os.path.dirname(_HERE), "probe_verify_sliver_round3")
if _V3 not in sys.path:
    sys.path.insert(1, _V3)

import f_fixtures as F  # noqa: E402

from lumenairy.elements.pmm import stack as _ps  # noqa: E402


def assert_tree():
    """Fail loudly if ``lumenairy`` did not resolve inside :data:`ROOT`."""
    got = os.path.abspath(_ps.__file__)
    if not got.lower().startswith(ROOT.lower() + os.sep):
        raise RuntimeError(
            f"probe tree mismatch: lumenairy resolved to {got}, expected a "
            f"module under {ROOT}.  Refusing to measure the wrong checkout.")
    return got

# re-export the round-3 surface unchanged
unguarded = F.unguarded
guarded = F.guarded
shared_move = F.shared_move
classify = F.classify
prescribed = F.prescribed
snapped = F.snapped
drop_factor = F.drop_factor
verdict_round2 = F.verdict_round2
verdict_round3 = F.verdict_round3
wbuild = F.wbuild
cbuild = F.cbuild
gmr = F.gmr
FIXTURES = F.FIXTURES
TRIG = F.TRIG
NO_SNAP_FRAC = F.NO_SNAP_FRAC


def arm():
    """Which measurement arm this process is: ``(build, kernel, threads)``.

    ``kernel`` is what OpenBLAS ACTUALLY dispatched to (its own ``corename``),
    never merely what ``OPENBLAS_CORETYPE`` requested -- the two differ, and
    round 4's first finding is that they differ for ``ZEN``.  ``threads`` is
    what OpenBLAS actually RUNS with, likewise read back rather than assumed,
    because an unpinned run's thread count is a property of the box."""
    build = "wsl" if (platform.system() == "Linux") else "win"
    core = None
    nthreads = None
    try:
        import threadpoolctl
        info = threadpoolctl.threadpool_info()
        for d in info:
            if d.get("internal_api") == "openblas":
                core = d.get("architecture")
                nthreads = d.get("num_threads")
                break
    except Exception:                                  # pragma: no cover
        pass
    return dict(build=build,
                requested=os.environ.get("OPENBLAS_CORETYPE", "DEFAULT"),
                kernel=core or "unknown",
                threads_requested=THREADS_REQUESTED,
                cpus=(int(AFFINITY) if AFFINITY else None),
                threads=nthreads,
                python=platform.python_version(),
                numpy=np.__version__)


def tag():
    a = arm()
    return f"{a['build']}_{a['requested']}_t{a['threads_requested']}"


def dump(obj, name):
    """Write ``<name>_<build>_<requested-coretype>.json`` beside this file."""
    import json
    path = os.path.join(_HERE, f"{name}_{tag()}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(arm=arm(), **obj), fh, indent=1, sort_keys=True,
                  default=float)
    print("wrote", path)
    return path

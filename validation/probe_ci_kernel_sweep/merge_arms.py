"""Merge the per-arm probe JSONs into the committed census ``decisions.json``.

Run ``probe_decisions.py --out arms/<build>_<CORETYPE>.json`` once per arm
(see that module's docstring for the ladder and its two measured caveats),
then::

    python validation/probe_ci_kernel_sweep/merge_arms.py

The merged file is what ``tests/unit/test_ci_kernel_consistency.py`` and the
mortar round-2 rationale test READ.  Neither of them writes it: a gate that
regenerates its own reference proves nothing.

The arm KEY is ``BUILD-Kernel-tN`` as the probe MEASURED it, not as it was
requested -- ``OPENBLAS_CORETYPE=ZEN`` and ``=BOGUSCORE`` both land on
``Haswell`` in these wheels, and recording the request would let one arm
appear twice and look like independent evidence.  A duplicate measured key is
an ERROR here for exactly that reason.

``tN`` is the thread width, ``tauto`` meaning the caps were left unset -- the
configuration CI's fast lane runs.  Both axes belong in the table; see
``tests/unit/test_ci_kernel_consistency.py`` for why.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    paths = sorted(glob.glob(os.path.join(HERE, "arms", "*.json")))
    if not paths:
        print("no arm JSONs under %s/arms" % HERE, file=sys.stderr)
        return 1
    out = {
        "_what": "CI kernel sweep census -- one arm per (build, OpenBLAS "
                 "kernel), thread caps pinned to 1.  See "
                 "docs/audits/CI_KERNEL_SWEEP_2026_09_11.md.",
        "_decisions": "library guard OUTCOMES; every arm must agree.",
        "_hypothetical": "verdicts of bars the library does NOT ship, at "
                         "sites it deliberately leaves unguarded; recorded to "
                         "be shown NON-unanimous.",
        "_readings": "the underlying floats; asserted nowhere.",
        "generated": _dt.date.today().isoformat(),
        "arms": {},
        "decisions": {},
        "hypothetical": {},
        "readings": {},
    }
    for p in paths:
        with open(p, encoding="cp1252") as fh:
            d = json.load(fh)
        arm = d["arm"]
        if arm in out["arms"]:
            print("duplicate measured arm %r (%s and %s): two requested "
                  "coretypes resolved to the same kernel, so they are ONE "
                  "arm, not two" % (arm, out["arms"][arm]["source"],
                                    os.path.basename(p)), file=sys.stderr)
            return 2
        out["arms"][arm] = {
            "source": os.path.basename(p),
            "build": d["build"],
            "kernel": d["kernel"],
            "thread_arm": d.get("thread_arm", "t1"),
            "blas_threads": d.get("blas_threads"),
            "coretype_requested": d.get("coretype_requested", ""),
            "platform": d["platform"],
            "python": d["python"],
            "numpy": d["numpy"],
            "scipy": d["scipy"],
            "threads": d["threads"],
        }
        out["decisions"][arm] = d["decisions"]
        out["hypothetical"][arm] = d.get("hypothetical", {})
        out["readings"][arm] = d["readings"]

    dest = os.path.join(HERE, "decisions.json")
    text = json.dumps(out, indent=1, sort_keys=True)
    json.loads(text)                       # self-validate before writing
    with open(dest, "w", encoding="cp1252", errors="replace") as fh:
        fh.write(text + "\n")
    print("merged %d arms -> %s" % (len(out["arms"]), dest))
    for arm, meta in sorted(out["arms"].items()):
        print("  %-18s %s py%s numpy%s  (requested %r)"
              % (arm, meta["platform"].split("-")[0], meta["python"],
                 meta["numpy"], meta["coretype_requested"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

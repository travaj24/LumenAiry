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

SYNTHETIC ARMS (2026-09-11).  An arm JSON may carry ``"synthetic": true`` and
a ``"provenance"`` string.  That is an arm nobody can run here -- today, the
CI runner, transcribed from the 5.45.0 matrix logs -- and it is carried
because it is the ONLY evidence of the one machine whose arithmetic solves
these fixtures correctly.  It is marked so that no reader mistakes a
transcription for a measurement, and the consistency gate exempts it from the
"every arm answered every row" check, because a transcription covers only the
rows the logs printed.

HISTORICAL ARMS (2026-09-12, WP-A23).  An arm JSON may instead carry
``"historical": true`` with ``"recorded"`` (the date) and ``"tree"`` (what the
library was when it was taken).  That is an arm that WAS measured, on a tree
that is no longer HEAD -- today, the ten 2026-09-11 arms taken before WP-A12's
``min_feature`` default change (``56a76f22``) moved this fixture's answer.
They are kept rather than deleted for two reasons: they are the only evidence
of the second BUILD and of the kernels the current host cannot reach, and they
are the before-side of the change that made the current arms read differently.
Like a transcription, a historical arm covers only the rows that existed when
it was taken, so the consistency gate exempts it from the "every arm answered
every row" check too -- and, unlike a transcription, it is still compared for
rule conformance, because it was a real run of real code.

THREE PROVENANCE KINDS, then: ``live`` (measured on the current tree),
``historical`` (measured, on an older tree) and ``synthetic`` (transcribed).
The gate requires the census to carry LIVE arms -- that is what stops it going
archival -- and lets the other two carry the coverage the current host cannot
reproduce.
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
        "_decisions": "library guard OUTCOMES.  Arms must agree where the "
                      "ANSWER CLASS agrees; a row whose class differs is the "
                      "guard following the answer and is reported, not "
                      "failed.  See probe_decisions.py, THE CONTRACT.",
        "_classes": "the answer class each guard row is deciding about -- "
                    "correct / grey / wrong by the campaign's own closure "
                    "rule -- measured WITH THE GUARD DISARMED.",
        "_hypothetical": "verdicts of bars the library does NOT ship, at "
                         "sites it deliberately leaves unguarded; recorded to "
                         "be shown NON-unanimous.",
        "_readings": "the underlying floats; asserted nowhere.",
        "_provenance": "every arm is one of three kinds.  LIVE: measured by "
                       "probe_decisions.py on the tree named in its 'tree' "
                       "field.  HISTORICAL ('historical': true): measured, on "
                       "an OLDER tree, kept for the coverage the current host "
                       "cannot reproduce and as the before-side of the change "
                       "that moved these rows; carries 'recorded' and 'tree'.  "
                       "SYNTHETIC ('synthetic': true): transcribed from logs, "
                       "never run here.  Only LIVE arms are required to answer "
                       "every row.",
        "_mechanism": (
            "WHY THIS TABLE WAS RE-RECORDED (2026-09-12, WP-A23).  The "
            "2026-09-11 census recorded sliver/pmm1d@1e-05 as 'wrong' on "
            "every measured arm; on 2026-09-12 every arm read it 'correct', "
            "and the cause was MEASURED rather than inferred.  Audit finding "
            "G2 (WP-A12, commit 56a76f22) raised "
            "pmm/stack.py::_MIN_FEATURE_DEFAULT_FRAC from period*1e-5 to "
            "period*1e-3.  The 1-D section's fixture is a CROSS-LAYER wall "
            "pair 1e-04 / 1e-05 of a period apart and did not pin "
            "min_feature, so at the new default both pairs were snapped to "
            "coincidence: the two layers became geometrically identical, the "
            "interface's reciprocal condition moved from 9.7297e-13 to "
            "5.2104e-04 (nine decades) and max(R+T) from 3.6124215325 to "
            "1.0000000000000628.  Flipping ONLY that constant back in "
            "process reproduced the committed readings to four significant "
            "figures and flipping it forward restored the new ones, twice.  "
            "The fixture now PINS min_feature = period*1e-5 (probe_decisions."
            "_1D_MF_PINNED), which is what makes it an ill-conditioned "
            "interface at all, and the shipped default is censused beside it "
            "under the '@mf-default' tag."),
        "generated": _dt.date.today().isoformat(),
        "arms": {},
        "decisions": {},
        "classes": {},
        "hypothetical": {},
        "readings": {},
    }
    for p in paths:
        with open(p, encoding="cp1252") as fh:
            d = json.load(fh)
        arm = d["arm"]
        # A HISTORICAL arm is keyed ``BUILD-Kernel-tN@<recorded>``.  The
        # canonical triple is not enough to identify it: the same host can be
        # re-censused on a later tree, and then two arms share the triple
        # while measuring different code.  That is not a duplicate to be
        # rejected -- it is the before/after pair this census exists to show
        # -- so the date disambiguates it, and the LIVE arm keeps the plain
        # triple so the "arm running right now" comparison still finds it.
        if d.get("historical"):
            arm = "%s@%s" % (arm, d.get("recorded", "undated"))
        if arm in out["arms"]:
            print("duplicate measured arm %r (%s and %s): two requested "
                  "coretypes resolved to the same kernel, so they are ONE "
                  "arm, not two" % (arm, out["arms"][arm]["source"],
                                    os.path.basename(p)), file=sys.stderr)
            return 2
        out["arms"][arm] = {
            "source": os.path.basename(p),
            "canonical_arm": d["arm"],
            "synthetic": bool(d.get("synthetic", False)),
            "historical": bool(d.get("historical", False)),
            "recorded": d.get("recorded", ""),
            "tree": d.get("tree", ""),
            "provenance": d.get("provenance", "measured by probe_decisions.py "
                                "on this machine"),
            "provenance_detail": d.get("provenance_detail", {}),
            "build": d["build"],
            "kernel": d["kernel"],
            "kernel_source": d.get("kernel_source", ""),
            "blas_libraries": d.get("blas_libraries", {}),
            "lumenairy": d.get("lumenairy", ""),
            "min_feature_default_frac": d.get("min_feature_default_frac"),
            "thread_arm": d.get("thread_arm", "t1"),
            "blas_threads": d.get("blas_threads"),
            "coretype_requested": d.get("coretype_requested", ""),
            "platform": d["platform"],
            "python": d["python"],
            "numpy": d["numpy"],
            "scipy": d["scipy"],
            "threads": d["threads"],
        }
        if out["arms"][arm]["synthetic"] and out["arms"][arm]["historical"]:
            print("arm %r is marked BOTH synthetic and historical (%s): a "
                  "transcription is not a run, so it cannot also be an older "
                  "run -- pick one" % (arm, os.path.basename(p)),
                  file=sys.stderr)
            return 3
        if out["arms"][arm]["synthetic"]:
            print("  (synthetic arm %r: %s)" % (arm, d.get("provenance", "")))
        elif out["arms"][arm]["historical"]:
            if not out["arms"][arm]["recorded"]:
                print("historical arm %r (%s) carries no \"recorded\" date.  "
                      "A historical reading without its date is not evidence, "
                      "it is folklore." % (arm, os.path.basename(p)),
                      file=sys.stderr)
                return 4
            print("  (historical arm %r: recorded %s on %s)"
                  % (arm, out["arms"][arm]["recorded"],
                     out["arms"][arm]["tree"] or "<tree not recorded>"))
        out["decisions"][arm] = d["decisions"]
        out["classes"][arm] = d.get("classes", {})
        out["hypothetical"][arm] = d.get("hypothetical", {})
        out["readings"][arm] = d["readings"]

    dest = os.path.join(HERE, "decisions.json")
    text = json.dumps(out, indent=1, sort_keys=True)
    json.loads(text)                       # self-validate before writing
    with open(dest, "w", encoding="cp1252", errors="replace") as fh:
        fh.write(text + "\n")
    kinds = {"live": 0, "historical": 0, "synthetic": 0}
    for meta in out["arms"].values():
        kinds["synthetic" if meta["synthetic"] else
              "historical" if meta["historical"] else "live"] += 1
    print("merged %d arms (%d live, %d historical, %d synthetic) -> %s"
          % (len(out["arms"]), kinds["live"], kinds["historical"],
             kinds["synthetic"], dest))
    for arm, meta in sorted(out["arms"].items()):
        kind = ("synthetic" if meta["synthetic"] else
                "historical" if meta["historical"] else "live")
        print("  %-22s %-10s %-8s py%s numpy%s  (requested %r)"
              % (arm, kind, meta["platform"].split("-")[0], meta["python"],
                 meta["numpy"], meta["coretype_requested"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

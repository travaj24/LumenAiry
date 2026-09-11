"""GAP 1, INDEPENDENT -- is the REPLACEMENT gate live, and was the RETIRED one
really vacuous?

The round-3 report proves its replacement live by patching
``_stack_is_provably_passive = _stack_media_are_passive`` at runtime -- the
predicate with its incidence-lossless conjunct removed -- and re-running the
gate.  This probe re-runs that mutation from scratch, and adds the half the fix
round could not run any more: the RETIRED gate's own geometry, to show
independently that at ``Im/Re`` = 1e-3 the channel set is empty and the
predicate is never reached, which is why that gate could not fail.

Nothing is edited on disk: the patch is applied to the live module object and
reverted in a ``finally``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vb3  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GATE = ("tests/unit/test_fix_bor_guards_round2.py::"
        "test_the_energy_screen_disarms_on_an_absorbing_incidence_medium"
        "_but_the_ceiling_does_not")

_PATCH = """
import lumenairy.elements.bor.bor_solve as _bs
_bs._stack_is_provably_passive = _bs._stack_media_are_passive
"""


def _run_gate(mutated):
    """Run the replacement gate, optionally with the incidence-lossless
    conjunct removed from the live predicate."""
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONPATH=ROOT)
    args = [sys.executable, "-m", "pytest", GATE, "-q", "-p", "no:randomly"]
    if mutated:
        conf = os.path.join(ROOT, "validation", "probe_verify_bor_round3",
                            "_mutate_conftest.py")
        with open(conf, "w", encoding="cp1252") as fh:
            fh.write(_PATCH)
        args = args[:3] + ["-p", "_mutate_conftest"] + args[3:]
        env["PYTHONPATH"] = (os.path.join(ROOT, "validation",
                                          "probe_verify_bor_round3")
                             + os.pathsep + ROOT)
    try:
        out = subprocess.run(args, cwd=ROOT, env=env, capture_output=True,
                             text=True)
    finally:
        if mutated and os.path.isfile(conf):   # never persists on disk
            os.remove(conf)
    tail = (out.stdout + out.stderr).strip().splitlines()
    summary = next((ln for ln in reversed(tail)
                    if ("passed" in ln or "failed" in ln or "error" in ln
                        or "no tests ran" in ln)), "<no summary line>")
    return out.returncode, summary


def part_retired_gate_premise():
    """The retired gate's own fixture, at the loss it used: how many channels
    survive, and is the predicate reached at all?"""
    import lumenairy.elements.bor.bor_solve as bs
    rows = []
    for im in (2e-3 / 2.0, 2e-5 / 2.0, 1e-7, 1e-9, 1e-11, 0.0):
        lay = _vb3.stack("nodal", "grate", 1, 120, 2.0, im, "inc",
                         e_half=2.0, k0=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            keep = np.where(bs._physical_propagating(lay[0], 2.0))[0]
        rows.append(dict(
            im_rel=im, n_channels=int(keep.size),
            predicate_reachable=bool(keep.size),
            media_passive=bool(bs._stack_media_are_passive(lay)),
            provably_passive=bool(bs._stack_is_provably_passive(lay))))
        print("  Im/Re=%-10.3g channels=%-4d reachable=%-5s media_passive=%-5s "
              "provably_passive=%s"
              % (im, keep.size, rows[-1]["predicate_reachable"],
                 rows[-1]["media_passive"], rows[-1]["provably_passive"]),
              flush=True)
    return rows


def main():
    a = _vb3.arm()
    _vb3.require_tree(a)
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"], flush=True)
    print("== the retired gate's premise", flush=True)
    premise = part_retired_gate_premise()
    print("== the replacement gate, intact and mutated", flush=True)
    rc_i, s_i = _run_gate(False)
    print("  intact   rc=%d :: %s" % (rc_i, s_i), flush=True)
    rc_m, s_m = _run_gate(True)
    print("  mutated  rc=%d :: %s" % (rc_m, s_m), flush=True)
    summary = dict(
        intact_rc=rc_i, intact_summary=s_i,
        mutated_rc=rc_m, mutated_summary=s_m,
        gate_is_live=bool(rc_i == 0 and rc_m != 0),
        retired_gate_premise=premise,
        retired_gate_channels_at_its_own_loss=premise[0]["n_channels"],
        retired_gate_was_vacuous=bool(premise[0]["n_channels"] == 0))
    for k in sorted(summary):
        if k != "retired_gate_premise":
            print(" ", k, summary[k])
    _vb3.dump("v8_mutation", dict(summary=summary))


if __name__ == "__main__":
    main()

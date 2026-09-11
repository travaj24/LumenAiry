"""Build the SYNTHETIC CI arm of the kernel census from the 5.45.0 matrix logs.

Nothing here is measured on this machine.  Every value is either transcribed
from a failure message in ``C:/tmp/ci_5450b/`` (the downloaded artifacts of
run 34566427386) or inferred from the ABSENCE of a mismatch in the
consistency test's own failure message, which named exactly three disagreeing
rows out of every cheap row it re-took.  Which of the two applies is recorded
per key in ``provenance_detail``.
"""
import io
import json
import os

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arms")
REF = os.path.join(HERE, "win_HASWELL_t1.json")

ref = json.load(io.open(REF, encoding="cp1252"))

# ---- transcribed from tests/unit/test_fix_pmm2d_mortar_round2.py's failure
#      on py3.11 shard 4: {0.0001: (rcond, R+T, nwarn), 1e-05: (...)}, taken
#      with the sliver guard DISARMED -- the same fixture and the same
#      disarming the probe's section A uses.
CI_RCOND = {"1e-04": 9.693991669399938e-11, "1e-05": 9.7296703051673e-13}
CI_RT = {"1e-04": 1.0000003658397656, "1e-05": 1.0000010471871335}
CI_NWARN = {"1e-04": 0, "1e-05": 0}

dec, cls, hyp, rea, detail = {}, {}, {}, {}, {}

for tag in ("1e-04", "1e-05"):
    rt, rc = CI_RT[tag], CI_RCOND[tag]
    dec["pmm1d_interface/answer@%s" % tag] = (
        "closes" if abs(rt - 1.0) < 1e-5 else "open")
    dec["pmm1d_interface/warned@%s" % tag] = (
        "warn" if CI_NWARN[tag] else "silent")
    dec["pmm1d_interface/returns@%s" % tag] = "return"
    dec["sliver/pmm1d@%s" % tag] = "return"
    klass = "correct" if abs(rt - 1.0) < 1e-5 else (
        "wrong" if abs(rt - 1.0) > 1e-2 else "grey")
    for k in ("pmm1d_interface/answer@%s" % tag,
              "pmm1d_interface/warned@%s" % tag,
              "pmm1d_interface/returns@%s" % tag,
              "sliver/pmm1d@%s" % tag):
        cls[k] = klass
        detail[k] = ("transcribed: the R+T reading is in the "
                     "test_the_plain_1d_interface_solve_is_left_unguarded "
                     "failure message (py3.11 shard 4); the decision is what "
                     "the test_ci_kernel_consistency failure message reports "
                     "the arm measured, or the census value it did NOT name "
                     "as a mismatch")
    rea["pmm1d_interface/rcond@%s" % tag] = rc
    rea["pmm1d_interface/R+T@%s" % tag] = rt
    hyp["pmm1d_interface/rcond_decade@%s" % tag] = "1e-11" if tag == "1e-04" \
        else "1e-13"
    hyp["pmm1d_interface/bar_1e-12_would@%s" % tag] = (
        "refuse" if rc < 1e-12 else "accept")

# ---- INFERRED, and the inference is stated: the consistency test on py3.11
#      shard 4 re-took the four cheap sections on this arm and named EXACTLY
#      three disagreeing rows (pmm1d_interface/answer@1e-05,
#      pmm1d_interface/warned@1e-05, sliver/pmm1d@1e-05).  Every other cheap
#      row therefore equalled the committed consensus, which is what these
#      are.  They are the KERNEL-INDEPENDENT sections -- a width band read off
#      the geometry and a branch-cut census over fixed spectra -- so the
#      agreement is expected rather than surprising.
for k, v in ref["decisions"].items():
    if k.startswith(("band/", "branch_cut/")):
        dec[k] = v
        detail[k] = ("inferred: the consistency test named exactly three "
                     "mismatching rows on this arm and this was not one of "
                     "them, so it equalled the committed consensus")

doc = {
    "arm": "CI-unknown-t1",
    "build": "CI",
    "kernel": "unknown",
    "thread_arm": "t1",
    "blas_threads": None,
    "coretype_requested": "",
    "synthetic": True,
    "provenance":
        "TRANSCRIBED from the 5.45.0 release matrix (GitHub Actions run "
        "34566427386, artifacts under C:/tmp/ci_5450b/), NOT measured.  "
        "ubuntu-latest runner, AMD EPYC 7763 (Zen 3), BLAS unpinned in the "
        "environment; the consistency test's own module sets "
        "OMP/OPENBLAS/MKL_NUM_THREADS=1 at import, so t1 is the REQUESTED "
        "cap -- whether OpenBLAS had already been loaded by an earlier test "
        "module is not recorded anywhere in the logs, and threadpoolctl is "
        "not installed on the runner, so the kernel reads as 'unknown' and "
        "the effective width is unknown too.  This arm exists because it is "
        "the ONLY evidence of the one machine whose arithmetic solves these "
        "ill-conditioned fixtures CORRECTLY; see "
        "docs/audits/CI_PREMISE_GATES_2026_09_11.md.",
    "provenance_detail": detail,
    "platform": "Linux-ubuntu-latest-x86_64 (GitHub Actions, AMD EPYC 7763)",
    "python": "3.11.16",
    "numpy": "unknown (the matrix installs unpinned and the pytest logs do "
             "not print it; the release round records numpy 2.4.6)",
    "scipy": "unknown (as above; the release round records scipy 1.17.1)",
    "threads": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1"},
    "decisions": dec,
    "classes": cls,
    "hypothetical": hyp,
    "readings": rea,
}
text = json.dumps(doc, indent=1, sort_keys=True)
json.loads(text)
dest = os.path.join(HERE, "ci_RUNNER_t1.json")
io.open(dest, "w", encoding="cp1252", errors="replace").write(text + "\n")
print("wrote %s (%d decisions, %d classed)" % (dest, len(dec), len(cls)))

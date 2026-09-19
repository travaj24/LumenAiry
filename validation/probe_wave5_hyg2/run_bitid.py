"""Wave-5 hygiene-2 bit-identity driver: run one probe in TWO child processes,
one bound to each extracted ``git archive`` tree, and compare the digest maps
key by key.

Neither arm is a live worktree: both are archives extracted read-only, so an
edit in either checkout cannot move a number under the measurement.  The probe
is passed by absolute path and puts its own directory plus the tree on
``sys.path`` -- pytest is never involved, because pytest puts the repository
root ahead of ``PYTHONPATH`` and both arms would then import the same tree.

argv: --probe P --base T --branch T --out-dir D --tag NAME [--python EXE]
      [--expect-differ PREFIX]...

``--expect-differ`` names a key PREFIX whose keys are EXPECTED to differ (a
deliberate behaviour change).  It does not weaken the verdict: the summary
reports the three sets separately -- keys that differ and were expected to,
keys that differ and were NOT (the failure), and expected-differ keys that did
NOT move (also a failure, because the change did not land).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def _run(python, probe, tree, out, env_extra=None):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "PYTHONPATH": tree,
                "PYTHONHASHSEED": "0",
                "LUMENAIRY_MEM_BUDGET_MB": "8192"})
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    proc = subprocess.run([python, probe, tree, out], cwd=tree, env=env,
                          capture_output=True, text=True)
    dt = time.time() - t0
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"probe failed on {tree}: rc={proc.returncode}\n"
                         f"{proc.stdout}\n{proc.stderr}")
    with open(out, encoding="utf-8") as fh:
        return json.load(fh), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--expect-differ", action="append", default=[])
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    base_json = os.path.join(a.out_dir, f"{a.tag}_base.json")
    branch_json = os.path.join(a.out_dir, f"{a.tag}_branch.json")

    base, t_base = _run(a.python, a.probe, os.path.abspath(a.base), base_json)
    branch, t_branch = _run(a.python, a.probe, os.path.abspath(a.branch),
                            branch_json)

    def expected(k):
        return any(k.startswith(pre) for pre in a.expect_differ)

    only_base = sorted(set(base) - set(branch))
    only_branch = sorted(set(branch) - set(base))
    shared = sorted(set(base) & set(branch))
    differ = [k for k in shared if base[k] != branch[k]]
    differ_expected = [k for k in differ if expected(k)]
    differ_unexpected = [k for k in differ if not expected(k)]
    expected_but_same = [k for k in shared
                         if expected(k) and base[k] == branch[k]]

    summary = {
        "tag": a.tag,
        "python": a.python,
        "base_tree": os.path.abspath(a.base),
        "branch_tree": os.path.abspath(a.branch),
        "expect_differ_prefixes": a.expect_differ,
        "n_keys_base": len(base),
        "n_keys_branch": len(branch),
        "n_shared": len(shared),
        "n_identical": len(shared) - len(differ),
        "n_differ_expected": len(differ_expected),
        "differ_unexpected": differ_unexpected,
        "expected_to_differ_but_identical": expected_but_same,
        "only_base": only_base,
        "only_branch": only_branch,
        "seconds": {"base": round(t_base, 1), "branch": round(t_branch, 1)},
        "verdict": ("IDENTICAL"
                    if not differ_unexpected and not expected_but_same
                    and not only_base and not only_branch
                    else "DIFFERS"),
    }
    path = os.path.join(a.out_dir, f"{a.tag}_compare.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    if summary["verdict"] != "IDENTICAL":
        sys.exit(3)


if __name__ == "__main__":
    main()

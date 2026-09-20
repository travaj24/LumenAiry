"""WP-C3 bit-identity driver: one probe, two child processes, two SPELLINGS.

``validation/probe_wave5_hyg2/run_bitid.py`` runs the same probe with the same
environment on two trees; this campaign needs one more axis, because the two
arms are deliberately spelled differently -- the base tree passes NOTHING
(``transport`` is ``'sziklas'`` there by default) and the branch tree passes
``transport='sziklas'`` explicitly.  That is the claim: the way back is one
keyword and it costs no bits.  So this driver is the hygiene-2 one plus a
per-arm environment, and it keeps that driver's summary shape exactly so the
two campaigns' JSON is comparable by eye.

argv: --probe P --base T --branch T --out-dir D --tag NAME [--python EXE]
      [--base-spelling S] [--branch-spelling S] [--expect-differ PREFIX]...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def _run(python, probe, tree, out, spelling):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "PYTHONPATH": tree,
                "PYTHONHASHSEED": "0",
                "LUMENAIRY_MEM_BUDGET_MB": "8192",
                "C3_SPELLING": spelling})
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
    ap.add_argument("--base-spelling", default="default")
    ap.add_argument("--branch-spelling", default="sziklas")
    ap.add_argument("--expect-differ", action="append", default=[])
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    base_json = os.path.join(a.out_dir, f"{a.tag}_base.json")
    branch_json = os.path.join(a.out_dir, f"{a.tag}_branch.json")

    base, t_base = _run(a.python, a.probe, os.path.abspath(a.base),
                        base_json, a.base_spelling)
    branch, t_branch = _run(a.python, a.probe, os.path.abspath(a.branch),
                            branch_json, a.branch_spelling)

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
        "base_spelling": a.base_spelling,
        "branch_spelling": a.branch_spelling,
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

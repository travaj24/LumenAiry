"""VERIFY-B11c bit-identity driver: run one probe in TWO child processes, one
bound to each extracted ``git archive`` tree, and compare the digest maps key by
key.

Neither arm is a live worktree: both are archives extracted read-only into the
scratch directory, so a concurrent edit in either checkout cannot move a number
under the measurement.  The probe file itself is passed by absolute path and
inserts its own directory plus the tree on ``sys.path`` -- pytest is never
involved, because pytest puts the repository root ahead of ``PYTHONPATH`` and
both arms would then import the same tree.

argv: --probe P --base T --branch T --out-dir D --tag NAME [--python EXE]
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
                "PYTHONHASHSEED": "0"})
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
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    base_json = os.path.join(a.out_dir, f"{a.tag}_base.json")
    branch_json = os.path.join(a.out_dir, f"{a.tag}_branch.json")

    base, t_base = _run(a.python, a.probe, os.path.abspath(a.base), base_json)
    branch, t_branch = _run(a.python, a.probe, os.path.abspath(a.branch),
                            branch_json)

    only_base = sorted(set(base) - set(branch))
    only_branch = sorted(set(branch) - set(base))
    shared = sorted(set(base) & set(branch))
    differ = [k for k in shared if base[k] != branch[k]]

    summary = {
        "tag": a.tag,
        "python": a.python,
        "base_tree": os.path.abspath(a.base),
        "branch_tree": os.path.abspath(a.branch),
        "n_keys_base": len(base),
        "n_keys_branch": len(branch),
        "n_shared": len(shared),
        "n_identical": len(shared) - len(differ),
        "differing": differ,
        "only_base": only_base,
        "only_branch": only_branch,
        "seconds": {"base": round(t_base, 1), "branch": round(t_branch, 1)},
        "verdict": ("IDENTICAL" if not differ and not only_base
                    and not only_branch else "DIFFERS"),
    }
    path = os.path.join(a.out_dir, f"{a.tag}_compare.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    if summary["verdict"] != "IDENTICAL":
        sys.exit(3)


if __name__ == "__main__":
    main()

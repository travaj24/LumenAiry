"""VERIFY-WP-C3 claim 4a driver -- N arms, one child process each, then every
requested pairwise key-by-key comparison.

An ARM is (tag, tree, spelling).  Each child gets ``cwd`` and ``PYTHONPATH``
set to its OWN tree and nothing else; ``probe_wayback.py`` asserts and prints
``lumenairy.__file__`` under that tree before it measures anything.

    python run_wayback.py --out-dir D --tag NAME \
        --arm base=/c/tmp/vc3_base:default \
        --arm mine=/c/tmp/vc3_v_mine:sziklas \
        --compare base:mine [--compare ...]  [--python EXE]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def run_arm(python, probe, tree, spelling, out):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
                "PYTHONPATH": tree, "VC3_SPELLING": spelling,
                "LUMENAIRY_MEM_BUDGET_MB": "8192"})
    t0 = time.time()
    pr = subprocess.run([python, probe, tree, out], cwd=tree, env=env,
                        capture_output=True, text=True)
    dt = time.time() - t0
    sys.stderr.write(pr.stderr[-4000:])
    if pr.returncode != 0:
        raise SystemExit("arm failed on %s rc=%d\n%s\n%s"
                         % (tree, pr.returncode, pr.stdout[-4000:],
                            pr.stderr[-4000:]))
    with open(out, encoding="utf-8") as fh:
        blob = json.load(fh)
    blob["_stdout"] = pr.stdout
    blob["_seconds"] = round(dt, 1)
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--probe", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "probe_wayback.py"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--arm", action="append", required=True,
                    help="name=/abs/tree:spelling")
    ap.add_argument("--compare", action="append", default=[],
                    help="left:right")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    arms = {}
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        tree, spelling = rest.rsplit(":", 1)
        out = os.path.join(a.out_dir, "%s_%s.json" % (a.tag, name))
        arms[name] = run_arm(a.python, a.probe, os.path.abspath(tree),
                             spelling, out)
        arms[name]["_spec"] = {"tree": os.path.abspath(tree),
                               "spelling": spelling, "json": out}

    summary = {"tag": a.tag, "python": a.python,
               "arms": {n: {"tree": v["_spec"]["tree"],
                            "spelling": v["_spec"]["spelling"],
                            "build": v["meta"]["build"],
                            "n_keys": len(v["digests"]),
                            "seconds": v["_seconds"],
                            "bind_stdout": v["_stdout"].strip().splitlines()}
                        for n, v in arms.items()},
               "comparisons": {}}

    for pair in a.compare:
        left, right = pair.split(":", 1)
        L, R = arms[left]["digests"], arms[right]["digests"]
        shared = sorted(set(L) & set(R))
        differ = [k for k in shared if L[k] != R[k]]
        summary["comparisons"][pair] = {
            "n_keys_left": len(L), "n_keys_right": len(R),
            "n_shared": len(shared),
            "n_identical": len(shared) - len(differ),
            "n_differ": len(differ),
            "differ": differ,
            "only_left": sorted(set(L) - set(R)),
            "only_right": sorted(set(R) - set(L)),
            "verdict": ("IDENTICAL" if not differ and len(L) == len(R)
                        == len(shared) else "DIFFERS"),
            "differ_detail": {
                k: {"left": arms[left]["notes"][k],
                    "right": arms[right]["notes"][k]} for k in differ},
        }

    path = os.path.join(a.out_dir, "%s_compare.json" % a.tag)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    slim = {"tag": summary["tag"],
            "arms": {n: {k: v[k] for k in ("tree", "spelling", "build",
                                           "n_keys", "seconds")}
                     for n, v in summary["arms"].items()},
            "comparisons": {p: {k: c[k] for k in
                                ("n_shared", "n_identical", "n_differ",
                                 "differ", "only_left", "only_right",
                                 "verdict")}
                            for p, c in summary["comparisons"].items()}}
    print(json.dumps(slim, indent=1))


if __name__ == "__main__":
    main()

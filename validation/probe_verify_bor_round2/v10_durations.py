"""V10 -- splice this verification's own file into ``.test_durations``.

Measures every test of ``tests/unit/test_verify_bor_guards_round2.py`` with
``pytest --durations=0 -vv`` (so pytest prints the sub-5 ms entries too),
replaces every key belonging to that file, and re-writes ``.test_durations``
SORTED and JSON-validated.  Prints the collected-vs-written count so a missing
id cannot pass unnoticed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

FILE = "tests/unit/test_verify_bor_guards_round2.py"
DUR = ".test_durations"
LINE = re.compile(r"^\s*([0-9.]+)s\s+call\s+(\S+)\s*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONPATH=a.root)
    coll = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:randomly", "-q",
         "--collect-only", FILE],
        cwd=a.root, env=env, capture_output=True, text=True)
    ids = [ln.strip() for ln in coll.stdout.splitlines()
           if ln.strip().startswith(FILE + "::")]
    print("collected %d test ids" % (len(ids),))

    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:randomly", "-vv",
         "--durations=0", FILE],
        cwd=a.root, env=env, capture_output=True, text=True)
    tail = run.stdout.splitlines()
    print([ln for ln in tail if " passed" in ln or " failed" in ln][-1:])
    got = {}
    for ln in tail:
        m = LINE.match(ln)
        if m and m.group(2).startswith(FILE + "::"):
            got[m.group(2)] = float(m.group(1))
    missing = [i for i in ids if i not in got]
    for i in missing:                       # pytest hides durations < 5 ms
        got[i] = 0.005
    print("measured %d, defaulted %d to pytest's 5 ms hidden-duration cutoff "
          "(an UPPER bound, the safe direction for a shard balancer)"
          % (len(got) - len(missing), len(missing)))
    over = {k: v for k, v in got.items() if v > 60.0}
    print("entries over the 60 s shard cap:", over or "none")
    print("slowest: %s" % (max(got.items(), key=lambda kv: kv[1]),))

    path = os.path.join(a.root, DUR)
    d = json.load(open(path))
    before = len(d)
    d = {k: v for k, v in d.items() if not k.startswith(FILE + "::")}
    d.update(got)
    d = dict(sorted(d.items()))
    assert list(d) == sorted(d), "the splice did not sort"
    blob = json.dumps(d, indent=1)
    json.loads(blob)                        # validate
    print("%d entries -> %d (this file contributes %d)"
          % (before, len(d), len(got)))
    if a.dry_run:
        print("dry run; not written")
        return
    with open(path, "w") as fh:
        fh.write(blob)
    print("wrote", path)


if __name__ == "__main__":
    main()

"""Re-measure ``.test_durations`` for the files round 3 touched, and splice.

WHY MEASURE RATHER THAN EDIT.  The shard balancer reads this file; a key that
is missing defaults, and a key that is stale mis-balances.  Round 3 renames one
test, moves another out of its file, and turns a 3-way parametrization into a
12-way one, so three files' key sets change.

``--durations=0 -vv`` is used deliberately: without it pytest hides entries
under 5 ms and they would be spliced in as absent.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

FILES = ("tests/unit/test_fix_bor_guards_round2.py",
         "tests/unit/test_fix_bor_multilayer_guards.py",
         "tests/unit/test_verify_bor_guards_round2.py")

#: pytest prints ``0.03s call     tests/unit/x.py::test_y``
_DUR = re.compile(r"^\s*([0-9.]+)s\s+(call|setup|teardown)\s+(\S+)\s*$")


def measure(python):
    # ``-vv`` WITHOUT ``-q``: pytest hides sub-5 ms durations unless verbosity
    # is > 1, and ``-q`` decrements it -- 12 fast tests were dropped from the
    # splice the first time this was run with both.
    cmd = [python, "-m", "pytest", *FILES, "-p", "no:randomly",
           "--durations=0", "-vv"]
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONPATH=ROOT)
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                         text=True).stdout
    tail = [ln for ln in out.splitlines()
            if re.search(r"passed|failed|error|no tests ran", ln)]
    tot = {}
    for ln in out.splitlines():
        mm = _DUR.match(ln)
        if not mm:
            continue
        tot[mm.group(3)] = tot.get(mm.group(3), 0.0) + float(mm.group(1))
    return tot, tail


def main():
    python = sys.argv[1] if len(sys.argv) > 1 else sys.executable
    measured, tail = measure(python)
    print("pytest tail:", *tail, sep="\n  ")
    assert measured, "no duration lines parsed -- pytest output shape changed"
    path = os.path.join(ROOT, ".test_durations")
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    before = len(d)
    # drop EVERY key belonging to the three files, then re-add the measured set
    d = {k: v for k, v in d.items()
         if k.split("::")[0].replace("\\", "/") not in FILES}
    dropped = before - len(d)
    for k, v in measured.items():
        d[k.replace("\\", "/")] = float(v)
    d = {k: d[k] for k in sorted(d)}
    assert list(d) == sorted(d), "the splice left the file un-sorted"
    # the shipped file is ``json.dumps(..., indent=1)``; keep it byte-shaped
    js = json.dumps(d, indent=1)
    json.loads(js)                      # the file must stay JSON-valid
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(js)
    over = {k: v for k, v in d.items() if v > 60.0}
    print("entries %d -> %d (dropped %d, added %d)"
          % (before, len(d), dropped, len(measured)))
    print("keys per file:", {f: sum(1 for k in d if k.startswith(f))
                             for f in FILES})
    print("slowest new entry:",
          max(measured.items(), key=lambda kv: kv[1]))
    mine_over = {k: v for k, v in measured.items() if v > 60.0}
    n_tests = sum(1 for ln in tail for _ in [1])
    print("entries over the 60 s shard cap, WHOLE FILE:", len(over))
    print("entries over the cap among the keys THIS round wrote:",
          len(mine_over), mine_over)
    print("tail lines parsed:", n_tests)


if __name__ == "__main__":
    main()

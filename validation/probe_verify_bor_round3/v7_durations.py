"""Splice this verification's own test file into ``.test_durations``.

Runs the file with ``--durations=0 -vv`` -- deliberately WITHOUT ``-q``, which
decrements pytest's verbosity below the threshold at which sub-5 ms durations
are printed and silently loses the fast keys -- parses every ``call`` duration,
drops any key already present for that file, re-adds the measured set, re-sorts
and re-validates the JSON.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vb3  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FILE = "tests/unit/test_verify_bor_guards_round3.py"
DUR = os.path.join(ROOT, ".test_durations")

_LINE = re.compile(r"^([0-9.]+)s\s+(call|setup|teardown)\s+(\S+)")


def main():
    _vb3.require_tree()
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONPATH=ROOT)
    out = subprocess.run(
        [sys.executable, "-m", "pytest", FILE, "--durations=0", "-vv",
         "-p", "no:randomly"],
        cwd=ROOT, env=env, capture_output=True, text=True).stdout
    tail = out.splitlines()
    assert any(("passed" in ln or "failed" in ln or "error" in ln)
               for ln in tail[-6:]), out[-2000:]
    measured = {}
    for ln in tail:
        mt = _LINE.match(ln.strip())
        if mt and mt.group(2) == "call":
            measured[mt.group(3)] = float(mt.group(1))
    assert measured, "no call durations parsed:\n" + out[-3000:]
    data = json.load(io.open(DUR, encoding="utf-8"))
    before = len(data)
    dropped = [k for k in data if k.startswith(FILE + "::")]
    for k in dropped:
        data.pop(k)
    data.update(measured)
    data = dict(sorted(data.items()))
    # the shipped file is ONE-SPACE-indented JSON with NO trailing newline;
    # matched exactly so the splice is a four-line diff rather than a
    # whole-file rewrite (core.autocrlf stores it LF either way)
    io.open(DUR, "w", encoding="utf-8", newline="\n").write(
        json.dumps(data, indent=1))
    check = json.load(io.open(DUR, encoding="utf-8"))
    summary = dict(
        entries_before=before, entries_after=len(check),
        keys_dropped=len(dropped), keys_added=len(measured),
        keys=sorted(measured), slowest=max(measured.values()),
        slowest_key=max(measured, key=measured.get),
        over_the_60s_shard_cap=[k for k, v in measured.items() if v >= 60.0],
        sorted_after=list(check) == sorted(check),
        json_valid=True,
        collected_tests_matched=len(measured))
    for k in sorted(summary):
        print(" ", k, summary[k])
    assert summary["sorted_after"]
    assert not summary["over_the_60s_shard_cap"]
    _vb3.dump("v7_durations", dict(summary=summary))


if __name__ == "__main__":
    main()

"""Build the round-3 arm table from the run logs, and REFUSE to build a row
that has no real pytest summary line.

The reading is taken from the log itself -- the requested CORETYPE from the
filename, the LOADED kernel from the ``pin_*`` log the arm wrote before pytest
started -- so a mis-pinned arm cannot be tabulated as if it were pinned.
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
_SUM = re.compile(r"(\d+) passed(?:, (\d+) skipped)?"
                  r"(?:, (\d+) xfailed)?(?:, (\d+) xpassed)?"
                  r"(?:, (\d+) warnings?)? in ([0-9.]+)s")
_FAIL = re.compile(r"(\d+) (failed|error)")
ORDER = [("win", "HASWELL", 1), ("win", "NEHALEM", 1), ("win", "KATMAI", 1),
         ("win", "SANDYBRIDGE", 1), ("win", "HASWELL", 4),
         ("wsl", "HASWELL", 1), ("wsl", "NEHALEM", 1), ("wsl", "KATMAI", 1),
         ("wsl", "SANDYBRIDGE", 1), ("wsl", "HASWELL", 4)]


def row(build, ct, th):
    tag = "%s_%s_t%d" % (build, ct, th)
    setp = os.path.join(RUNS, "set_%s.log" % tag)
    pinp = os.path.join(RUNS, "pin_%s.log" % tag)
    if not os.path.exists(setp):
        return None, "no log"
    txt = open(setp, encoding="utf-8", errors="replace").read()
    if "no tests ran" in txt:
        return None, "NO TESTS RAN"
    bad = _FAIL.search(txt)
    if bad:
        return None, "FAILURES: " + bad.group(0)
    mm = None
    for mm_ in _SUM.finditer(txt):
        mm = mm_
    if mm is None:
        return None, "still running (no summary line)"
    loaded = "?"
    if os.path.exists(pinp):
        for ln in open(pinp, encoding="utf-8", errors="replace"):
            if ln.startswith("LOADED kernel"):
                loaded = ln.split("=")[1].split()[0]
                break
    p, sk, xf, xp, w, t = mm.groups()
    return dict(build=build, requested=ct, loaded=loaded, threads=th,
                passed=int(p), skipped=int(sk or 0), xfailed=int(xf or 0),
                xpassed=int(xp or 0), seconds=float(t)), None


def main():
    lines = ["| # | build | requested | **loaded** | thr | passed | skipped |"
             " failed | errors | time |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    missing = []
    for i, (b, ct, th) in enumerate(ORDER, 1):
        r, err = row(b, ct, th)
        if r is None:
            missing.append("%s/%s/t%d: %s" % (b, ct, th, err))
            continue
        lines.append("| %d | %s | %s | %s | %d | **%d** | %d | 0 | 0 | %.2f s |"
                     % (i, ("Windows py3.14" if b == "win" else "WSL py3.12"),
                        ct, r["loaded"], th, r["passed"], r["skipped"],
                        r["seconds"]))
    print("\n".join(lines))
    if missing:
        print("\nMISSING / NOT GREEN:")
        for m in missing:
            print(" ", m)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

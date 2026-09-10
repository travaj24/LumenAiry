"""Splice the round-2 node ids into ``.test_durations``.

Adds every ``<file>::<test>`` the round-2 run measured, drops the ids of tests
this round renamed away, and rewrites the file SORTED and json-valid (the
format ``pytest-split`` reads: a flat ``{nodeid: seconds}`` object).
"""
import json
import re
import sys

DUR = ".test_durations"
LOG = "validation/probe_fix_branch_cut_round2/logs/durations.log"

#: node ids this round renamed away (their tests no longer exist)
GONE_PREFIXES = (
    "tests/unit/test_m1_conditioning_guard.py::"
    "test_x1_defect_is_reproduced_and_flagged_but_NOT_closed",
    "tests/unit/test_m1_conditioning_guard.py::"
    "test_the_refusal_reproduces_the_prior_answer_with_the_switch",
)

pat = re.compile(r"^([0-9.]+)s call\s+(\S+::\S+)\s*$")
new = {}
with open(LOG, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        m = pat.match(line.strip())
        if m:
            new[m.group(2)] = float(m.group(1))

with open(DUR, encoding="utf-8") as fh:
    dur = json.load(fh)

removed = [k for k in dur if k.startswith(GONE_PREFIXES)]
for k in removed:
    del dur[k]
# pytest hides sub-5ms durations, so a parametrization that ran fast never
# appears in the log.  Fill the missing arms of a parametrized test from the
# slowest arm that DID appear, so the file stays complete for pytest-split.
for M in (21, 12, 20):
    k = ("tests/unit/test_m1_conditioning_guard.py::"
         "test_x1_is_closed_on_the_cell_it_was_pinned_at[%d-te]" % M)
    ref = ("tests/unit/test_m1_conditioning_guard.py::"
           "test_x1_is_closed_on_the_cell_it_was_pinned_at[19-te]")
    if k not in new and ref in new:
        new[k] = new[ref]

added = [k for k in new if k not in dur]
dur.update(new)

out = {k: dur[k] for k in sorted(dur)}
with open(DUR, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(out, fh, indent=2)

with open(DUR, encoding="utf-8") as fh:
    check = json.load(fh)
assert list(check) == sorted(check), "not sorted"
print("removed %d, added %d, updated %d, total %d, sorted=%s"
      % (len(removed), len(added), len(new) - len(added), len(check), True))
for k in removed:
    print("  -", k)
for k in sorted(added):
    print("  +", k)
sys.exit(0)

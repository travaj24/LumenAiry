"""Splice the round-4 node ids into ``.test_durations``.

Adds every ``<file>::<test>`` the round-4 run measured, drops the ids of tests
this round renamed away, and rewrites the file SORTED and json-valid (the
format ``pytest-split`` reads: a flat ``{nodeid: seconds}`` object).

Same shape as ``validation/probe_fix_branch_cut_round2/_splice_durations.py``.

Usage, from the worktree root, after

    python -m pytest <the eight sliver files> -q --durations=0 -p no:randomly \\
        > validation/probe_fix_sliver_round4/logs/durations.log

    python validation/probe_fix_sliver_round4/_splice_durations.py
"""
import json
import re
import sys

DUR = ".test_durations"
LOG = "validation/probe_fix_sliver_round4/logs/durations.log"

#: node ids this round renamed away (their tests no longer exist)
GONE = (
    "tests/unit/test_fix_pmmstack_sliver_walls.py::"
    "test_fail_before_the_pre_fix_path_returns_a_wrong_energy_violating_answer",
    "tests/unit/test_fix_pmmstack_sliver_walls.py::"
    "test_the_guard_refuses_every_wrong_row_and_no_right_one",
    "tests/unit/test_fix_pmmstack_sliver_walls_round2.py::"
    "test_the_arbiter_costs_one_solve_and_only_on_a_triggered_stack",
    "tests/unit/test_fix_pmmstack_sliver_walls_round2.py::"
    "test_the_round1_misses_are_refused_or_are_below_the_trigger",
    "tests/unit/test_fix_pmmstack_sliver_walls_round2.py::"
    "test_the_trigger_sits_above_the_correct_populations_envelope",
    "tests/unit/test_fix_pmmstack_sliver_round3.py::"
    "test_the_closure_fraction_separates_the_two_drop_populations",
    "tests/unit/test_fix_pmmstack_sliver_round3.py::"
    "test_the_truncation_note_states_the_measured_drop_and_promises_nothing",
    "tests/unit/test_verify_pmmstack_sliver_round3.py::"
    "test_the_relative_closure_leaves_a_restorable_wrong_answer_returned",
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

removed = [k for k in dur if k in GONE]
for k in removed:
    del dur[k]

added = [k for k in new if k not in dur]
dur.update(new)

out = {k: dur[k] for k in sorted(dur)}
with open(DUR, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(out, fh, indent=2)

with open(DUR, encoding="utf-8") as fh:
    check = json.load(fh)
assert list(check) == sorted(check), "not sorted"
missing = [k for k in GONE if k in check]
assert not missing, missing
print("removed %d, added %d, updated %d, total %d, sorted=True"
      % (len(removed), len(added), len(new) - len(added), len(check)))
for k in removed:
    print("  -", k)
for k in sorted(added):
    print("  +", k)
sys.exit(0)

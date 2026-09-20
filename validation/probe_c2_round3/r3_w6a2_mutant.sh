#!/usr/bin/env bash
# WP-C2 ROUND 3, VR2-D5 -- the two-sided demonstration of the 1e-5 bar.
#
# The ``w6_a2`` DECISION-3 message names three unconverged modes -- "a wrong
# Hessian, a missing prior term, a sign error" -- and claimed all three give
# "a ratio of order 1, four decades above" a 1e-4 bar.  Measured, the third
# does not: the prior term is ``I / w_p**2`` against ``J^T J / w_s**2`` with
# ``w_s = 20e-6`` and ``w_p = 0.02``, so a solver that computes its step with
# the prior term dropped and RETURNS THAT STEP lands 2.53e-05 (Windows) /
# 2.57e-05 (WSL) of the first step away from the root -- four times UNDER a
# 1e-4 bar.
#
# Two mutants, because the distinction matters:
#
#   M13a  the prior term dropped from the Gauss-Newton Hessian ONLY, with
#         the solver still iterating.  Newton with an approximate Jacobian
#         converges to the same root, just more slowly, so the returned
#         point is still converged and BOTH bars are right to pass it.
#         This arm exists so "the bar catches a wrong Hessian" is not
#         claimed where it should not be.
#   M13b  the same Hessian AND one sweep instead of twelve -- the solver
#         that returns its first step.  That is the mode the message names
#         and the one the retired bar missed:
#
#             under the retired 1e-4 bar   -> PASSES (the defect)
#             under the shipped 1e-5 bar   -> FAILS  (the closure)
#
# Both readings of M13b come from the SAME mutant tree, so "the bar is what
# changed" is measured rather than argued.
#
# Usage:  bash r3_w6a2_mutant.sh <win|wsl>
set -u
BUILD="${1:-win}"
SRC=/c/tmp/lum_c2c
MUT=/c/tmp/lum_c2c_mut_w6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ "$BUILD" = "wsl" ]; then
  RUN() { wsl -e bash -lc "cd /mnt/c/tmp/lum_c2c_mut_w6 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_c2c_mut_w6 ~/lumvenv/bin/python -m pytest $* -q -p no:randomly --capture=sys -rf 2>&1 | tail -12"; }
else
  RUN() { (cd "$MUT" && PYTHONPATH="$MUT" python -m pytest $* -q -p no:randomly --capture=sys -rf 2>&1 | tail -12); }
fi

fresh() {
  rm -rf "$MUT"; mkdir -p "$MUT"
  git -C "$SRC" archive feat/c2-analytic-normal-round3 | tar -x -C "$MUT"
}

mutate() {   # mutate() <one_sweep|iterating>
  python - "$MUT" "$1" <<'PY'
import pathlib, sys
p = (pathlib.Path(sys.argv[1])
     / 'lumenairy/propagators/asymptotic_maslov.py')
t = p.read_text(encoding='cp1252')
pairs = [("        H = inv_ws2 * np.matmul(np.swapaxes(J, -1, -2), J) "
          "+ inv_wp2 * eye2",
          "        H = inv_ws2 * np.matmul(np.swapaxes(J, -1, -2), J) "
          "+ 0.0 * inv_wp2 * eye2")]
if sys.argv[2] == 'one_sweep':
    pairs.append(("    for it in range(max_iter):", "    for it in range(1):"))
for old, new in pairs:
    assert t.count(old) == 1, ('anchor missing: %r' % old[:60])
    t = t.replace(old, new, 1)
p.write_text(t, encoding='cp1252')
print('prior term dropped from the Hessian; sweeps:', sys.argv[2])
PY
}

put_back_the_old_bar() {
  python - "$MUT" <<'PY'
import pathlib, sys
p = (pathlib.Path(sys.argv[1])
     / 'tests/unit/test_niche_audit_w6_asymptotic.py')
t = p.read_text(encoding='cp1252')
assert t.count('    assert step_ratio < 1e-5, (') == 1, (
    'the shipped bar is not 1e-5 in the archived tree -- commit the '
    'VR2-D5 edit before running this script')
p.write_text(t.replace('    assert step_ratio < 1e-5, (',
                       '    assert step_ratio < 1e-4, (', 1),
             encoding='cp1252')
print('the bar put back to the retired 1e-4')
PY
}

T=tests/unit/test_niche_audit_w6_asymptotic.py
A=test_w6_a2_v2_star_is_untouched_by_the_verdict_fix

echo "=================== M0  control, shipped 1e-5 bar ==================="
fresh
RUN "$T -k $A"

echo
echo "=========== M13a  wrong Hessian, still iterating, 1e-5 bar ==========="
fresh
mutate iterating
RUN "$T -k $A"

echo
echo "=========== M13b  wrong Hessian + one sweep, SHIPPED 1e-5 bar ==========="
fresh
mutate one_sweep
RUN "$T -k $A"

echo
echo "=========== M13b  the same tree, RETIRED 1e-4 bar ==========="
put_back_the_old_bar
RUN "$T -k $A"

echo
echo "=================== done ==================="

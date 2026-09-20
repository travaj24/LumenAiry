#!/usr/bin/env bash
# VERIFY-WAVE5-E: the mutation matrix.  $1 = root of the mutant tree,
# $2 = pristine tree, $3 = python command, $4 = tag.
set -u
ROOT="$1"; PRIS="$2"; PY="$3"; TAG="$4"
MUTPY=${MUTPY:-python}   # interpreter that runs mutate.py (WSL has no bare python)
M="$PRIS/validation/probe_verify_wave5_e/mutate.py"
E5=tests/unit/test_wave5_e_exit_vertex_dead_rays.py
E2=tests/unit/test_wave5_e_c8_default_order.py
E1=tests/unit/test_wave5_e_fft_elision.py
B12="tests/unit/test_audit2609_b12_fga_reference_plane.py tests/unit/test_verify_b12_fga_reference_plane.py"
run() {  # name  files...
  local name="$1"; shift
  $MUTPY "$M" --root "$ROOT" --pristine "$PRIS" "$name" >/dev/null 2>&1 \
    || { echo "== $name  MUTATION-REFUSED"; return; }
  cd "$ROOT"
  local out
  out=$(OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        PYTHONPATH="$ROOT" $PY -m pytest "$@" --capture=sys -p no:randomly -q 2>&1 | tail -3)
  echo "== $name [$*]"
  echo "$out" | grep -E "passed|failed|error|skipped|no tests ran" | tail -2
  cd "$PRIS"
}
echo "### mutation matrix  ($TAG)  root=$ROOT"
$MUTPY "$M" --root "$ROOT" --pristine "$PRIS" --revert >/dev/null; cd "$ROOT"; echo "== identity"; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT" $PY -m pytest $E5 $E2 $E1 --capture=sys -p no:randomly -q 2>&1 | grep -E "passed|failed|error|skipped|no tests ran" | tail -1; cd "$PRIS"
run e5_unfreeze                   $E5
run e5_unfreeze                   $B12
run e5_freeze_everything          $E5
run e5_round1_alive               $E5
run e5_round1_alive               $B12
run e5_tol_drop_z                 $E5
run e5_guard_off_at_coarse        $E5
run e1_unnamed_right_operand      $E1
run e1_privatise_dispatchers      $E1
run e1_drop_scope_sentence        $E1
run e2_bound_disabled             $E2
run e2_annuli_reversed            $E2
run e2_stimulus_dead              $E2
run e2_stimulus_dead,e2_control_weak $E2
$MUTPY "$M" --root "$ROOT" --pristine "$PRIS" --revert

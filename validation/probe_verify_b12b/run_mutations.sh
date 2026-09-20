#!/usr/bin/env bash
# VERIFY-WP-B12b -- the mutation matrix.
#
#   bash validation/probe_verify_b12b/run_mutations.sh <python> <tag> [files...]
#
# Runs the named test files once per mutation with the in-memory mutation
# plugin ``vb12b_mutate`` active, and prints one line per (mutation, file)
# with the pytest summary tail plus the ids that went RED.  A durable pin must
# go RED under the mutation that breaks the property it protects.
set -u
PY="${1:?python}"
TAG="${2:?tag}"
shift 2
FILES=("$@")
OUT="validation/probe_verify_b12b/mutations_${TAG}.txt"
: > "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LUMENAIRY_MEM_BUDGET_MB=4096
export PYTHONPATH="$PWD/validation/probe_verify_b12b${PYTHONPATH:+:$PYTHONPATH}"
for MUT in none fold_restored fold_restored_renamed local_surface world_exit_vertex state_only sign_plus identity; do
  if [ "$MUT" = "none" ]; then unset VB12B_MUTATION; else export VB12B_MUTATION="$MUT"; fi
  for F in "${FILES[@]}"; do
    LOG=$(mktemp)
    "$PY" -m pytest "$F" -p vb12b_mutate -q --capture=sys -p no:randomly \
        --no-header -rf > "$LOG" 2>&1
    TAIL=$(grep -E "passed|failed|error|no tests ran" "$LOG" | tail -1)
    REDS=$(grep -E "^FAILED|^ERROR" "$LOG" | sed 's/ - .*//' | sed 's#.*::#::#' | tr '\n' ' ')
    echo "[$MUT] $F :: $TAIL" >> "$OUT"
    if [ -n "$REDS" ]; then echo "    RED: $REDS" >> "$OUT"; fi
    echo "[$MUT] $F :: $TAIL"
    if [ -n "$REDS" ]; then echo "    RED: $REDS"; fi
    rm -f "$LOG"
  done
done
echo "WROTE $OUT"

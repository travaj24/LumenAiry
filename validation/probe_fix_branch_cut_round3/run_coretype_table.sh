#!/bin/sh
# ROUND 3: every branch-cut DECISION test file x every RUNNABLE OpenBLAS
# kernel, one thread, on one build.  Usage: run_coretype_table.sh <python>
PY=${1:-python}
FILES="test_fix_rcwa_even_sector_wsl.py test_verify_rcwa_even_sector.py \
test_fix_branch_cut_round2.py test_verify_branch_cut_round2.py \
test_m1_conditioning_guard.py test_v5_14_2_backlog_batch.py \
test_audit_s1_2_rcwa_lossless_tripwire.py"
for CT in HASWELL ZEN SKYLAKEX PRESCOTT SANDYBRIDGE; do
  ARCH=$(OPENBLAS_CORETYPE=$CT OPENBLAS_NUM_THREADS=1 $PY -c \
    "import threadpoolctl,numpy;print(threadpoolctl.threadpool_info()[0]['architecture'])" 2>/dev/null)
  if [ -z "$ARCH" ]; then
    echo "CORETYPE=$CT  arch=<UNRUNNABLE: numpy aborts (SIGILL)>"
    continue
  fi
  # a real BLAS kernel call, not just the banner
  OK=$(OPENBLAS_CORETYPE=$CT OPENBLAS_NUM_THREADS=1 $PY -c \
    "import numpy as np;a=np.random.default_rng(0).standard_normal((200,200));print('%.10f'%float(np.max(np.linalg.eigvals(a@a.T).real)))" 2>/dev/null)
  if [ -z "$OK" ]; then
    echo "CORETYPE=$CT  arch=$ARCH  <UNRUNNABLE: eig aborts (SIGILL)>"
    continue
  fi
  echo "CORETYPE=$CT  arch=$ARCH  eig=$OK"
  for F in $FILES; do
    R=$(cd "$(dirname "$0")/../.." && OPENBLAS_CORETYPE=$CT OMP_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        $PY -m pytest "tests/unit/$F" -q -p no:randomly 2>&1 \
        | grep -E "passed|failed|error|no tests ran" | tail -1)
    printf "    %-46s %s\n" "$F" "$R"
  done
done

#!/bin/sh
# Wave 1: reconstruct (and exceed) the S14.6 failure condition.
#   6 x stress_pairs (the full P4 call, diagnosed/undiagnosed pairs)
# + 2 x stress_fit   (the one BLAS-adjacent step, at high count)
# all at OMP_NUM_THREADS=8 on a 24-logical box = 64 BLAS threads, against the
# 5 pytest x 8 the failure was observed under.  No retries, no supervision.
set -e
cd "$(dirname "$0")"
export LUMENAIRY_ROOT="C:/tmp/lum_p4"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
N_PAIR=${1:-60}
N_FIT=${2:-20000}
for a in 1 2 3 4 5 6; do
  python stress_pairs.py "$N_PAIR" "pair$a" "wave1_pair$a.jsonl" &
done
for a in 1 2; do
  python stress_fit.py "$N_FIT" "fit$a" "wave1_fit$a.jsonl" fit_AB.npz &
done
wait
echo "WAVE1 DONE"

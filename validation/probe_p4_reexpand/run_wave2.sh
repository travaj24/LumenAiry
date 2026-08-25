#!/bin/sh
# Wave 2: the SAME pairs under a MIXED-WIDTH load (so the arms contend for the
# same cores at different BLAS widths -- the condition a batch of independent
# pytest processes actually produces) plus an IN-PROCESS THREADED arm, which is
# the only thing that can reach shared module state concurrently.
set -e
cd "$(dirname "$0")"
export LUMENAIRY_ROOT="C:/tmp/lum_p4"
N=${1:-60}
NT=${2:-20}
for w in 1 2 4 8 16; do
  OMP_NUM_THREADS=$w OPENBLAS_NUM_THREADS=$w MKL_NUM_THREADS=$w \
    python stress_pairs.py "$N" "w$w" "wave2_pair_w$w.jsonl" &
done
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  python stress_threads.py 4 "$NT" wave2_threads.jsonl &
wait
echo "WAVE2 DONE"

#!/bin/sh
# INTERLEAVED A/B/A/B: A = one arm alone, B = the same arm inside a 6-way
# concurrent load.  Interleaved because a sequential "alone, then loaded" run
# measures the box's mood as much as the load.
cd "$(dirname "$0")"
export LUMENAIRY_ROOT="C:/tmp/lum_p4"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
N=${1:-3}
for rep in 1 2; do
  python stress_pairs.py "$N" "alone$rep" "il_alone$rep.jsonl" > /dev/null
  for a in 2 3 4 5 6; do
    python stress_pairs.py "$N" "bg$a" "il_bg${rep}_$a.jsonl" > /dev/null &
  done
  python stress_pairs.py "$N" "loaded$rep" "il_loaded$rep.jsonl" > /dev/null
  wait
done
echo INTERLEAVED_DONE

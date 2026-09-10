#!/bin/bash
set -e
P=/c/tmp/lum_lensfix/validation/probe_fix_lens_5440
R=$P/results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /c/tmp/lum_lensfix
echo "=========== p6 stages AFTER the D6 redundancy fix, N=4096 ==========="
PYTHONPATH=/c/tmp/lum_lensfix python $P/p6_d6_stages.py $R/p6_stages_5440fix_N4096.json 4096 32 1.5 2
echo "=========== p3 wall time AFTER, N=4096 ==========="
PYTHONPATH=/c/tmp/lum_lensfix python $P/p3_d6_time.py $R/p3_d6_5440fix_N4096.json 4096 32 1.5 3
echo "=========== p3 wall time AFTER, N=2048 ==========="
PYTHONPATH=/c/tmp/lum_lensfix python $P/p3_d6_time.py $R/p3_d6_5440fix_N2048.json 2048 16 3.0 3
echo "ALL DONE"

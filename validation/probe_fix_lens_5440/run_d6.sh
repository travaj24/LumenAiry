#!/bin/bash
set -e
P=/c/tmp/lum_lensfix/validation/probe_fix_lens_5440
R=$P/results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "=========== ARM 5.44.0 (fix tree) N=4096 ==========="
cd /c/tmp/lum_lensfix && PYTHONPATH=/c/tmp/lum_lensfix python $P/p3_d6_time.py $R/p3_d6_5440_N4096.json 4096 32 1.5 3
echo "=========== ARM v5.43.0 N=4096 ==========="
cd /c/tmp/lum_v5430b && PYTHONPATH=/c/tmp/lum_v5430b python $P/p3_d6_time.py $R/p3_d6_5430_N4096.json 4096 32 1.5 3
echo "=========== ARM 5.44.0 N=2048 ==========="
cd /c/tmp/lum_lensfix && PYTHONPATH=/c/tmp/lum_lensfix python $P/p3_d6_time.py $R/p3_d6_5440_N2048.json 2048 16 3.0 3
echo "=========== ARM v5.43.0 N=2048 ==========="
cd /c/tmp/lum_v5430b && PYTHONPATH=/c/tmp/lum_v5430b python $P/p3_d6_time.py $R/p3_d6_5430_N2048.json 2048 16 3.0 3
echo "ALL DONE"

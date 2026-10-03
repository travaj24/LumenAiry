#!/bin/sh
# the verifier's mutation matrix against the E1 unit tests + the verifier's
# decision tests; one log per kind.  chitrap runs the UNMAPPED suites.
cd /c/tmp/lum_vcurved_e1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="C:/tmp/lum_vcurved_e1;C:/tmp/lum_vcurved_e1/validation/probe_pmm2d_curved/verify_e1"
OUT=validation/probe_pmm2d_curved/verify_e1/mut_logs
mkdir -p $OUT
for k in $*; do
  if [ "$k" = chitrap ]; then
    FILES="tests/unit/test_pmm2d_staggered_oop.py tests/unit/test_pmm2d_staggered_slant.py tests/unit/test_pmm2d_staggered_magnetic.py tests/unit/test_pmm2d_staggered_anisotropic.py tests/unit/test_pmm2d_staggered_oop_block_eig.py tests/unit/test_pmm2d_oop_block_eig.py tests/unit/test_v5_12_0_pmm2d_staggered.py tests/unit/test_staggered.py tests/unit/test_verify_pmm2d_perlayer_slant.py tests/unit/test_pmm2d_staggered_nonuniform.py"
  else
    FILES="tests/unit/test_pmm2d_staggered_curved_e1.py tests/unit/test_verify_pmm2d_curved_e1.py"
  fi
  VE1_MUT=$k timeout 1750 python -m pytest -p ve1_mutplugin --capture=sys -p no:randomly -q -n 4 $FILES -p no:cacheprovider > $OUT/$k.txt 2>&1
  echo "$k: $(tail -1 $OUT/$k.txt)" >> $OUT/summary.txt
done

#!/bin/bash
# WP-C3 round 2 -- the D5/D6 censuses on WSL-py3.12, both trees.
set -e
PY=~/lumvenv/bin/python
OUT=/mnt/c/tmp/lum_c3b/validation/probe_c3_round2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "=== base blastwidth ==="
cd /mnt/c/tmp/c3b_base && VC3_TREE=/mnt/c/tmp/c3b_base VC3_OUT=$OUT VC3_TAG=r2_base_wsl \
  PYTHONPATH=/mnt/c/tmp/c3b_base $PY validation/probe_verify_c3/probe_vc3_blastwidth.py | tail -2
echo "=== branch blastwidth ==="
cd /mnt/c/tmp/lum_c3b && VC3_TREE=/mnt/c/tmp/lum_c3b VC3_OUT=$OUT VC3_TAG=r2_fix_wsl \
  PYTHONPATH=/mnt/c/tmp/lum_c3b $PY validation/probe_verify_c3/probe_vc3_blastwidth.py | tail -2
echo "=== compare ==="
cd /mnt/c/tmp/lum_c3b && $PY validation/probe_c3_round2/r2_blast_compare.py \
  $OUT/blastwidth_r2_base_wsl.json $OUT/blastwidth_r2_fix_wsl.json | head -10
echo "=== 12-cell reproducer, branch ==="
cd /mnt/c/tmp/lum_c3b && VC3_TREE=/mnt/c/tmp/lum_c3b VC3_OUT=$OUT VC3_TAG=r2_fix_wsl \
  PYTHONPATH=/mnt/c/tmp/lum_c3b $PY validation/probe_verify_c3/probe_vc3_newraise_indep.py | tail -13
echo "=== grid ladder, branch ==="
cd /mnt/c/tmp/lum_c3b && $PY validation/probe_verify_c3/probe_gridladder.py \
  /mnt/c/tmp/lum_c3b $OUT/gridladder_r2fix_wsl.json | tail -2
echo "=== 103-key wayback: base(default) vs branch(sziklas) and branch(default) ==="
cd /mnt/c/tmp/lum_c3b && $PY validation/probe_verify_c3/run_wayback.py \
  --out-dir $OUT --tag r2_wsl --python $PY \
  --arm base=/mnt/c/tmp/c3b_base:default \
  --arm fix=/mnt/c/tmp/lum_c3b:sziklas \
  --arm dflt=/mnt/c/tmp/lum_c3b:default \
  --compare base:fix --compare base:dflt 2>&1 | tail -40
echo "=== DONE ==="

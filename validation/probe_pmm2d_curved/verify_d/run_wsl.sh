#!/bin/bash
# the verifier's second build: WSL Ubuntu, ~/lumvenv, BLAS pinned
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/lumvenv/bin/python
V=/mnt/c/tmp/lum_vcurved_d/validation/probe_pmm2d_curved/verify_d
OUT=$V/wsl
mkdir -p $OUT
cd /mnt/c/tmp/vcd_pre_607d && PYTHONPATH=/mnt/c/tmp/vcd_pre_607d $PY $V/v1_bytes.py /mnt/c/tmp/vcd_pre_607d wslpre > $OUT/v1pre.txt 2>&1
cd /mnt/c/tmp/lum_vcurved_d && PYTHONPATH=/mnt/c/tmp/lum_vcurved_d $PY $V/v1_bytes.py /mnt/c/tmp/lum_vcurved_d wslpost > $OUT/v1post.txt 2>&1
cd $V && export PYTHONPATH=/mnt/c/tmp/lum_vcurved_d
for a in "sh4 gyro none n 4" "sh4 gyro none c 5" "sh4 rasym none c 5" "sh4 2.0 gyro n 4" "sh4 2.0 gyro c 5" "c3 lossy none n 7" "h2 lc30 none n 8" "sh4 gyro none n 4 eps_T" "sh4 2.0 gyro n 4 chi_T"; do
  $PY - $a <<'PY' >> $OUT/v2.txt 2>&1
import sys, runpy
sys.argv = ["v2_film.py"] + sys.argv[1:]
import _vdcommon
_vdcommon.dump = lambda *a, **k: None          # do not overwrite the Windows JSON
runpy.run_path("v2_film.py", run_name="__main__")
PY
done
for a in "identity 8" "unmapped 8" "sh4 6" "identity_swap 8"; do
  $PY - $a <<'PY' >> $OUT/v5.txt 2>&1
import sys, runpy
sys.argv = ["v5_li.py"] + sys.argv[1:]
import _vdcommon
_vdcommon.dump = lambda *a, **k: None
runpy.run_path("v5_li.py", run_name="__main__")
PY
done
cd /mnt/c/tmp/lum_vcurved_d && PYTHONPATH=/mnt/c/tmp/lum_vcurved_d $PY -m pytest tests/unit/test_pmm2d_staggered_curved_a.py tests/unit/test_pmm2d_staggered_curved_b.py tests/unit/test_pmm2d_staggered_curved_c.py tests/unit/test_pmm2d_staggered_curved_d.py tests/unit/test_verify_pmm2d_curved_a.py tests/unit/test_verify_pmm2d_curved_b.py tests/unit/test_verify_pmm2d_curved_d.py tests/unit/test_v4_16_0_walker_all_symmetry.py --capture=sys -p no:randomly -q -p no:cacheprovider -n 4 > $OUT/pytest.txt 2>&1
echo WSLDONE >> $OUT/pytest.txt

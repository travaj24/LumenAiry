#!/bin/bash
# Kernel ladder for the Wave-5 item-D files owned by THIS engineer:
#   b5 (D3 mirror), a14 (H4 pencil), a11 (Z3 peaks + lens memory), b8 (B8 peak).
# Output dir is unique to this engineer -- do not share it.
cd /c/tmp/lum_reds
FILES="tests/unit/test_audit2609_b5_rcwa_eme_bor.py tests/unit/test_audit2609_a14_rcwa_eme_bor.py tests/unit/test_audit2609_a11_polar_sources_infra.py tests/unit/test_audit2609_b8_analysis_sources.py"
OUT=validation/probe_known_reds/ladder_itemD_a11_a14_b5_b8
mkdir -p "$OUT"
for CT in HASWELL NEHALEM KATMAI SANDYBRIDGE; do
  for NT in 1 4; do
    tag="${CT}_t${NT}"
    OPENBLAS_CORETYPE=$CT OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT \
      PYTHONPATH=/c/tmp/lum_reds python -m pytest $FILES --capture=sys -p no:randomly -q -ra \
      > "$OUT/$tag.log" 2>&1
    echo "$tag :: $(tail -3 "$OUT/$tag.log" | grep -E 'passed|failed|no tests ran|error' | tail -1)"
  done
done

#!/bin/bash
# VERIFY-WP-B12: run the new test file under each in-memory mutation of the
# repair and record which ids go red.  A pin that stays green under every
# mutation of the property it claims is not pinning that property.
cd /c/tmp/lum_vb12 || exit 1
OUT=validation/probe_verify_b12/mutations_${TAG:-win}.txt
: > "$OUT"
for M in "" identity state_only sign_plus conic_only biconic_drop field_frame_drop always_flat opd_sign no_sec n_exit_one; do
  echo "=== MUTATION: ${M:-<none: baseline>} ===" >> "$OUT"
  VB12_MUTATION="$M" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH="C:/tmp/lum_vb12;C:/tmp/lum_vb12/validation/probe_verify_b12" \
    python -m pytest tests/unit/test_audit2609_b12_fga_reference_plane.py \
      -p vb12_mutate -p no:randomly --capture=sys -q --no-header -rf \
      2>&1 | grep -E "^FAILED|ImportError|passed|failed|error|no tests ran" >> "$OUT"
done
echo DONE >> "$OUT"

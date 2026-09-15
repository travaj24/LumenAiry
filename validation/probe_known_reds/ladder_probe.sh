#!/bin/bash
# Kernel ladder for a PROBE (not a pytest node).  Usage:
#   ladder_probe.sh <probe.py> [outdir]
# Runs 4 OPENBLAS_CORETYPE x {1,4} threads, writing <probe>_<CT>_t<NT>.json
# into outdir (default: the probe's own directory).
cd /c/tmp/lum_reds
P="$1"; OUT="${2:-/c/tmp/lum_reds/validation/probe_known_reds}"; mkdir -p "$OUT"
for CT in HASWELL NEHALEM KATMAI SANDYBRIDGE; do
  for NT in 1 4; do
    tag="${CT}_t${NT}"
    OPENBLAS_CORETYPE=$CT OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT \
      MKL_NUM_THREADS=$NT PYTHONPATH=/c/tmp/lum_reds \
      timeout 1800 python "$P" "$tag" "$OUT" > "$OUT/.ladder_$tag.log" 2>&1
    echo "--- $tag"
    grep -E '^R1=|^merit |^n=|^use_pyfftw|^arm=|Traceback|Error' \
      "$OUT/.ladder_$tag.log" | head -12
  done
done

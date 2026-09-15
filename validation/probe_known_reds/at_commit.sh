#!/bin/bash
# Run probe_c8_guard_reach.py against the `lumenairy` tree of one commit, in an
# isolated archive tree (never the shared working tree).  Usage:
#   at_commit.sh <commit> [probe.py]
set -e
C="$1"; PROBE="${2:-/c/tmp/lum_reds/validation/probe_known_reds/probe_c8_guard_reach.py}"
D="/c/tmp/lum_reds_arch/$C"
if [ ! -d "$D/lumenairy" ]; then
  mkdir -p "$D"
  git -C /c/tmp/lum_reds archive "$C" lumenairy | tar -x -C "$D"
fi
cp "$PROBE" "$D/probe_run.py"
cd "$D"
OMP_NUM_THREADS=${NT:-1} OPENBLAS_NUM_THREADS=${NT:-1} MKL_NUM_THREADS=${NT:-1} \
  PYTHONPATH="$D" PROBE_TAG="$C" timeout 1800 python "$D/probe_run.py" 2>&1 \
  | grep -E 'lumenairy_file|"version"|halo_3w|guard_max_abs|bound_max_abs'

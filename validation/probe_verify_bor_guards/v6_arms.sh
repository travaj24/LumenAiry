#!/usr/bin/env bash
# The kernel x thread x build ladder for the band decision (task B).  Every arm
# reads the LOADED OpenBLAS kernel back from threadpoolctl inside the probe and
# records it in the JSON, so a silent alias (ZEN -> Haswell on this host)
# cannot be mistaken for coverage.  One output DIRECTORY per requested arm, so
# two requests that load the same kernel do not overwrite each other.
set -u
BASE=/c/tmp/lum_vbor/validation/probe_verify_bor_guards
LOG=$BASE/runs; mkdir -p "$LOG"
for ct in HASWELL NEHALEM PRESCOTT SANDYBRIDGE UNSET; do
  for thr in 1 4; do
    for build in post pre; do
      tree=/c/tmp/lum_vbor; [ "$build" = pre ] && tree=/c/tmp/lum_vbor_pre
      tag="win_${ct}_t${thr}_${build}"
      out="$BASE/arms/$tag"; mkdir -p "$out"
      if [ "$ct" = UNSET ]; then
        OMP_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr MKL_NUM_THREADS=$thr \
          PYTHONPATH=$tree python "$BASE/v2_band.py" "$build" "$out" --fast \
          > "$LOG/v6_${tag}.log" 2>&1
      else
        OMP_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr MKL_NUM_THREADS=$thr \
          OPENBLAS_CORETYPE=$ct PYTHONPATH=$tree \
          python "$BASE/v2_band.py" "$build" "$out" --fast \
          > "$LOG/v6_${tag}.log" 2>&1
      fi
      echo "$tag exit=$? $(ls "$out" 2>/dev/null | tr '\n' ' ')"
    done
  done
done

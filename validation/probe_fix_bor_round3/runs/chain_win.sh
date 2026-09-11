#!/usr/bin/env bash
# The remaining WINDOWS arms, sequentially.
set -u
R=/c/tmp/lum_bor3/validation/probe_fix_bor_round3/runs
for a in "NEHALEM 1" "KATMAI 1" "SANDYBRIDGE 1" "HASWELL 4"; do
  bash "$R/arm_win.sh" $a
done
echo CHAIN_WIN_DONE

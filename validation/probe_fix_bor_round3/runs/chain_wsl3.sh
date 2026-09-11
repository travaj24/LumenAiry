#!/usr/bin/env bash
# The WSL arms still owed after HASWELL/1 and NEHALEM/1.
# The two builds' arms run happily CONCURRENTLY: measured, the Windows and WSL
# Haswell/1 arms took 1431 s and 1321 s against the round-2 verification's
# un-contended 1841 s and 1708 s on the same 21 files.  (An apparent stall
# during this round was ``tee`` block-buffering pytest's dots, not contention:
# the process read 1950 s of CPU in 1953 s of wall clock throughout.)
set -u
R=/mnt/c/tmp/lum_bor3/validation/probe_fix_bor_round3/runs
for a in "KATMAI 1" "SANDYBRIDGE 1" "HASWELL 4"; do
  bash "$R/arm_wsl.sh" $a
done
echo CHAIN_WSL3_DONE

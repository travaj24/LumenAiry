#!/bin/bash
# WSL build: the round-2 test list (suite_wsl.txt), -n 6, pinned to the tree
cd /mnt/c/tmp/lum_curved_e3b
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_curved_e3b
V=validation/probe_pmm2d_curved/build_e3r2
FILES=$(tr -d '\r' < $V/suite_wsl.txt)
[ -n "$FILES" ] || { echo "empty test list"; exit 2; }
~/lumvenv/bin/python -c "import lumenairy, sys; f = lumenairy.__file__; print(f); sys.exit(0 if f.startswith('/mnt/c/tmp/lum_curved_e3b') else 3)" || exit 3
timeout 1780 ~/lumvenv/bin/python -m pytest $FILES --capture=sys -p no:randomly -q -n 4 -p no:cacheprovider > $V/logs/suite_wsl.log 2>&1
echo "rc=$?"
tail -1 $V/logs/suite_wsl.log
grep "^FAILED\|^ERROR" $V/logs/suite_wsl.log | head

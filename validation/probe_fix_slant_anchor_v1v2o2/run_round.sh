#!/bin/sh
# Run every probe of this directory with one suffix, on one build.
#
#   PY=python              ./run_round.sh _postfix      (WIN, Git Bash)
#   PY=~/lumvenv/bin/python ./run_round.sh _postfix     (WSL)
#
# The build is decided by the interpreter; ``_lib.arm()`` stamps it -- and
# refuses any tree but C:/tmp/lum_slfix -- into every JSON.
SUF="${1:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
PY="${PY:-python}"
BUILD="$("$PY" -c 'import sys;print("win" if sys.platform.startswith("win") else "wsl")')"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for p in v1_routes v2_derive v2_cross v2_compose v2_census o2_census o2_identity; do
  LOG="results/$p$SUF.$BUILD.log"
  echo "=== $p$SUF ($BUILD) ==="
  "$PY" -u "$p.py" "$SUF" > "$LOG" 2>&1 || echo "   NONZERO EXIT (see $LOG)"
  tail -2 "$LOG"
done
echo "ROUND $SUF DONE ON $BUILD"

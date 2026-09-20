#!/bin/sh
# VERIFY-WP-B7c round 3 -- oracle-score the NEAR-BAR planes the L2/L3
# refinements found.  Those are the planes claim 2 lives on (the smallest
# refused reading and the largest returned one), and the ladder passes take
# them with --oracle none so that the refinement is cheap.
set -e
cd "$(dirname "$0")"
tag="${1:-win}"; shift
for o in "$@"; do
  zs=$(PYTHONPATH=/c/tmp/lum_vmb3 python - "$o" <<'PY'
import json,glob,sys
nm=sys.argv[1]; zs=set()
for p in sorted(glob.glob('ladder_*_%s.json'%'win')):
    d=json.load(open(p,encoding='cp1252'))
    for key in ('L2','L3'):
        for r in d.get(key) or []:
            c=r.get('pixel_continuity')
            if r.get('fixture')==nm and r.get('route')=='fold_ring' and c is not None and 1.0<=c<=2.5:
                zs.add(round(r['z_um'],6))
print(','.join('%.6f'%z for z in sorted(zs)))
PY
)
  if [ -z "$zs" ]; then echo "no near-bar planes for $o"; continue; fi
  echo "=== $o ($(echo $zs | tr ',' '\n' | wc -l) planes)"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=/c/tmp/lum_vmb3 python v3scan.py "$o" "$zs" "nearbar_${o}_${tag}.json" \
    > "log_nearbar_${o}_${tag}.txt" 2>&1
  echo "done $o"
done

#!/bin/bash
# VERIFY-WP-B12: the remaining Windows runs, in sequence (one heavy process at
# a time on a shared box).
set -u
P=/c/tmp/lum_vb12/validation/probe_verify_b12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# VERIFY-B12 defect D-4: apply_real_lens_fga's returned BYTES depend on the
# momentum / lattice chunking, and the chunking on a memory budget that
# defaults to a fraction of the AVAILABLE RAM at call time.  A tree-to-tree
# byte-identity comparison is only meaningful with the budget pinned.
export LUMENAIRY_MEM_BUDGET_MB=2000

echo "=== V5 archive: PRE tree 96cb2096 ==="
cd /c/tmp/lum_vb12_pre && \
PYTHONPATH="C:/tmp/lum_vb12_pre;C:/tmp/lum_vb12/validation/probe_verify_b12" \
  python -u "$P/probe_v5_archive.py" --tree C:/tmp/lum_vb12_pre --tag pre_96cb2096_pinned \
    --only asph,bicon,menisc,flat_planoconvex,flat_pair \
    --out "$P/probe_v5_archive_pre_win32_314.json" 2>&1

echo "=== V5 archive: HEAD tree 1218b24f ==="
cd /c/tmp/lum_vb12 && \
PYTHONPATH="C:/tmp/lum_vb12;C:/tmp/lum_vb12/validation/probe_verify_b12" \
  python -u "$P/probe_v5_archive.py" --tree C:/tmp/lum_vb12 --tag head_1218b24f_pinned \
    --only asph,bicon,menisc,flat_planoconvex,flat_pair \
    --out "$P/probe_v5_archive_head_win32_314.json" 2>&1

echo "=== V7 route ladder ==="
cd /c/tmp/lum_vb12 && PYTHONPATH="C:/tmp/lum_vb12" \
  python -u "$P/probe_v7_route.py" 2>&1

echo "ALLDONE"

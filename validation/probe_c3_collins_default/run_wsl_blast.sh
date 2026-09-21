#!/bin/bash
# WP-C3 -- the carrier-touching blast set on the WSL build.
#
# ``test_audit2609_b4_collins_transport.py`` is run ONE CLASS AT A TIME here
# and excluded from the bulk run.  The whole file STALLS on WSL at
# ``TestGateCTwoGroupChain`` -- pre-existing, proved so at 112c3049 by the
# Wave-5 hygiene-2 round-2 verification, which is why that report's WSL lane
# excludes the file.  Per class it completes.
set -u
cd /mnt/c/tmp/lum_c3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/c/tmp/lum_c3
PY=~/lumvenv/bin/python
B4=tests/unit/test_audit2609_b4_collins_transport.py
FILES=$(grep -v "$B4" validation/probe_c3_collins_default/blast_files.txt | tr '\n' ' ')
echo "=== bulk (b4 excluded) ==="
$PY -m pytest $FILES tests/unit/test_c3_collins_default.py \
    --capture=sys -q -p no:randomly -rf
echo "=== b4, one class at a time ==="
for C in TestVocabulary TestDefaultIsByteIdentical TestSameTheorem \
         TestGateAOracleMatrix TestGateBMismatchMatrix TestGateCTwoGroupChain \
         TestGateDMulti TestKellyGuard TestKernelRefinement \
         TestQuadratureComplementarity TestNoNearFocusApparatus \
         TestReadoutPeriodDecoupling TestLegPeriodCondition \
         TestAbsolutePhaseThroughTheFocus TestKernelRefinementNearTheFocus \
         TestTiltedAndDecentredReadout; do
  echo "--- $C ---"
  $PY -m pytest "$B4::$C" --capture=sys -q -p no:randomly -rf 2>&1 | tail -3
done

#!/bin/bash
# WP-C3 ROUND 2 -- the carrier-touching blast set on the WSL build.
#
# ``test_audit2609_b4_collins_transport.py`` is run ONE CLASS AT A TIME and
# excluded from the bulk run: the whole file STALLS on WSL at
# ``TestGateCTwoGroupChain`` (pre-existing, reproduced by VERIFY-WP-C3), so
# every class carries an explicit ``timeout 1200``.
#
# ``blast_files.txt`` is CRLF in this worktree, so the carriage return has to
# be stripped -- otherwise pytest is handed a filename with a stray CR and
# reports "file or directory not found" before running anything.
set -u
cd /mnt/c/tmp/lum_c3b
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/c/tmp/lum_c3b
PY=~/lumvenv/bin/python
B4=tests/unit/test_audit2609_b4_collins_transport.py
LIST=validation/probe_c3_collins_default/blast_files.txt
FILES=$(sed -e 's/\r$//' "$LIST" | grep -v "$B4" | tr '\n' ' ')
echo "=== bulk (b4 excluded) ==="
$PY -m pytest $FILES tests/unit/test_c3_collins_default.py \
    tests/unit/test_verify_c3_collins_default.py \
    --capture=sys -q -p no:randomly -p no:cacheprovider -rf 2>&1 | tail -25
echo "=== b4, one class at a time ==="
for C in TestVocabulary TestDefaultIsByteIdentical TestSameTheorem \
         TestGateAOracleMatrix TestGateBMismatchMatrix TestGateCTwoGroupChain \
         TestGateDMulti TestKellyGuard TestKernelRefinement \
         TestQuadratureComplementarity TestNoNearFocusApparatus \
         TestReadoutPeriodDecoupling TestLegPeriodCondition \
         TestAbsolutePhaseThroughTheFocus TestKernelRefinementNearTheFocus \
         TestTiltedAndDecentredReadout; do
  echo "--- $C ---"
  timeout 1200 $PY -m pytest "$B4::$C" --capture=sys -q -p no:randomly \
      -p no:cacheprovider -rf 2>&1 | tail -3
  echo "   rc=$?"
done
echo "=== WSL BLAST DONE ==="

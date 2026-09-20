#!/bin/bash
# WP-C3 round 2 -- rebuild the 3-way merge from the FINAL branch tip and run
# every C3 / C5-listed id on it.
set -u
cd /c/tmp/lum_c3b
git worktree remove --force C:/tmp/c3b_merge 2>/dev/null || true
rm -rf /c/tmp/c3b_merge
git worktree add --detach C:/tmp/c3b_merge HEAD 2>&1 | tail -1
cd /c/tmp/c3b_merge
git merge --no-commit --no-ff feat/c5-three-defaults-round2 2>&1 | grep -i conflict
# the documents conflict by design; take OURS for them, they do not affect a
# test run.  carrier.py auto-merges.
for f in .test_durations CHANGELOG.md Migration-Guide.md docs/history/carrier.md; do
  git checkout --ours -- "$f" 2>/dev/null || true
done
python - <<'PY'
import io, sys
P = 'C:/tmp/c3b_merge/tests/unit/test_fix_v1_v8_readout_guard_and_standoff.py'
s = io.open(P, encoding='cp1252', newline='').read()
if '<<<<<<< HEAD' in s:
    a = s.index('<<<<<<< HEAD'); b = s.index('=======', a)
    c = s.index('>>>>>>> feat/c5-three-defaults-round2', b)
    ours = s[a + len('<<<<<<< HEAD\r\n'):b]
    theirs = s[b + len('=======\r\n'):c]
    doc = [ln for ln in ours.split('\r\n')[1:] if ln.strip() != '']
    merged = theirs.rstrip('\r\n') + '\r\n' + '\r\n'.join(doc) + '\r\n'
    s = s[:a] + merged + s[c + len('>>>>>>> feat/c5-three-defaults-round2\r\n'):]
    io.open(P, 'w', encoding='cp1252', newline='').write(s)
    print('resolved the one test conflict')
PY
grep -rl "^<<<<<<< HEAD" --include=*.py . | head
echo "--- merged constants ---"
grep -n "_GAP_KERNEL_ACCURACY_TAU = \|transport: str = " lumenairy/propagators/carrier.py
echo "=== MERGED TREE TEST RUN ==="
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest \
  tests/unit/test_wave5_h2_near_focus_table.py \
  tests/unit/test_c3_collins_default.py \
  tests/unit/test_verify_c3_collins_default.py \
  tests/unit/test_niche_gap_frame_observable.py \
  tests/unit/test_niche_d3_guards.py \
  tests/unit/test_niche_exact_gap_kernel.py \
  tests/unit/test_niche_c3_gap_paraxial_guard.py \
  tests/unit/test_niche_d2_chain_multi.py \
  tests/unit/test_niche_d5_dx_flatness_gate.py \
  tests/unit/test_verify_hyg2_round2.py \
  tests/unit/test_fix_v1_v8_readout_guard_and_standoff.py \
  tests/unit/test_niche_tight_focus_readout.py \
  -q --capture=sys -p no:randomly -p no:cacheprovider -rf 2>&1 | tail -25
echo "=== b4 on the merged tree ==="
python -m pytest tests/unit/test_audit2609_b4_collins_transport.py \
  -q --capture=sys -p no:randomly -p no:cacheprovider -rf 2>&1 | tail -8
echo "=== MERGED DONE ==="

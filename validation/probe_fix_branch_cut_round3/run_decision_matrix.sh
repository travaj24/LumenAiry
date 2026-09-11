#!/bin/sh
# ROUND 3: every branch-cut DECISION over kernel x thread-count, on one build.
#
#   kernels  HASWELL (= the default, and what an AMD Zen/EPYC selects because
#            this OpenBLAS carries NO Zen target), NEHALEM, KATMAI
#            (== PRESCOTT, verified), SANDYBRIDGE
#   threads  1, 2, 4, unpinned  -- the fast CI lane leaves BLAS UNPINNED on
#            4-core runners, so the thread axis is the one that differs there
#
# Usage: run_decision_matrix.sh <python>
PY=${1:-python}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

T="\
tests/unit/test_niche_audit_w9_eig_vjp.py::test_pmm2d_near_normal_angle_gradient_improved \
tests/unit/test_v5_14_0_pmm2d_autodiff.py::test_gate_angle_grad_at_normal_offcenter_is_genuine \
tests/unit/test_v5_14_0_pmm2d_autodiff.py::test_gate_degen_angle_grad_centered_square_is_clean_zero \
tests/unit/test_fix_branch_cut_round2.py::test_the_spacer_coincidence_is_what_breaks_the_pre_round_one_branch \
tests/unit/test_fix_branch_cut_round2.py::test_the_shared_body_keeps_the_round_one_SELECTOR_and_changes_only_the_value \
tests/unit/test_m1_conditioning_guard.py::test_x1_is_closed_across_the_whole_thin_ladder_and_reopens_pre_fix \
tests/unit/test_audit_s1_2_rcwa_lossless_tripwire.py::test_the_exact_index_coincidence_now_closes_and_the_warning_is_silent \
tests/unit/test_audit_s1_2_rcwa_lossless_tripwire.py::test_the_pre_round_one_branch_reopens_it_and_the_message_names_the_cause \
tests/unit/test_verify_rcwa_even_sector.py::test_a_growing_root_can_only_come_from_the_band_and_only_by_the_band \
tests/unit/test_verify_rcwa_even_sector.py::test_the_flip_is_the_exact_minus_r_involution \
tests/unit/test_fix_rcwa_even_sector_wsl.py::test_sqrt_decay_pins_the_outgoing_root_through_eigensolver_noise \
tests/unit/test_verify_branch_cut_round2.py::test_a_lossy_spacer_alone_does_not_empty_the_bands_jurisdiction \
tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_stripe_fixture_is_free_of_the_mode_match_degeneracy"

cd "$ROOT" || exit 1
for CT in HASWELL NEHALEM KATMAI SANDYBRIDGE; do
  ARCH=$(OPENBLAS_CORETYPE=$CT OPENBLAS_NUM_THREADS=1 $PY -c \
    "import threadpoolctl,numpy;print(threadpoolctl.threadpool_info()[0]['architecture'])" 2>/dev/null)
  [ -z "$ARCH" ] && { echo "$CT  <UNRUNNABLE>"; continue; }
  # UNPINNED is run on HASWELL only -- that kernel IS what CI's AMD EPYC
  # selects (this OpenBLAS has no Zen target), so HASWELL x unpinned reproduces
  # the fast lane exactly.  It is also 20x slower here than any pinned count:
  # a 16-core/32-thread desktop thrashes on this batch's many SMALL solves,
  # which is a property of the host, not of the library.
  case $CT in HASWELL) THREADS="1 2 4 unpinned" ;; *) THREADS="1 2 4" ;; esac
  for TH in $THREADS; do
    if [ "$TH" = unpinned ]; then
      R=$(OPENBLAS_CORETYPE=$CT $PY -m pytest $T -q -p no:randomly 2>&1 \
          | grep -E "passed|failed|error|no tests ran" | tail -1)
    else
      R=$(OPENBLAS_CORETYPE=$CT OMP_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH \
          MKL_NUM_THREADS=$TH $PY -m pytest $T -q -p no:randomly 2>&1 \
          | grep -E "passed|failed|error|no tests ran" | tail -1)
    fi
    printf "%-12s arch=%-12s threads=%-8s %s\n" "$CT" "$ARCH" "$TH" "$R"
  done
done

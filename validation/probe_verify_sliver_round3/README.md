# probe_verify_sliver_round3 -- the INDEPENDENT verification of round 3

Written from scratch for
`docs/audits/VERIFY_PMMSTACK_SLIVER_ROUND3_2026_09_11.md`.  Nothing here
imports from `probe_fix_sliver_round3/` or `probe_verify_sliver_round2/`, and
no number in the report is read out of either one's JSON: every statistic is
re-measured, on DIFFERENT devices.

| file | what it measures |
|---|---|
| `v_fixtures.py` | the fixtures and helpers.  Seven device families (staircase, guided-mode resonance, Fabry-Perot cavity, near-Wood mount, rotated uniaxial in-plane and out-of-plane directors, owned liner, high-contrast Fano grating) and the 576-mount configuration box, none of them the fix's geometry |
| `v1_bitid.py` | 66 fixtures, run on the round-3 tree AND on the branch point `f2371e0`: the SHA-256 of every returned buffer, the whole warning set, the arbiter's evidence dict, and two independent continuity references per row |
| `v1_compare.py` | scores two `v1_bitid` JSONs: bit-identity, warning sets after excising the reworded `truncation` note, refusal supersets, and the classified flip census |
| `v2_closure.py` | the 2,304-row box: the closure's sizing rule re-derived, the false-refusal search, and the library's own decision on every row |
| `v3_d5.py` | the D-5 band: 24 candidate mounts screened, nine kept, 30 wall steps each; the recovery rate, the R3-A edge, and the truncation note's truth |
| `v4_move.py` | the MOVE bar: a resonance hunt for the steepest `dR/dx` this campaign can build, then the deliberate false-refusal attack on resonant and many-slice devices |
| `v5_testbars.py` | every bar in `tests/unit/test_fix_pmmstack_sliver_round3.py`, re-measured by IMPORTING that test module |
| `v6_open_items.py` | R3-B (an owned liner disarms the screen -- at pico AND at physical widths) and R3-C (a keyed `prepare()` stack is outside the guard), plus the probe-side-effect contracts |
| `v7_crossbuild.py` | the Windows / WSL comparison: decisions first, then the largest relative spread, then how far each decision sits from flipping |
| `v8_attack.py` | the DIRECTED false-refusal attack: the six mounts the ladder box came closest on, swept finely in wall step at two degrees.  This is the probe that found defect V-4 |
| `v9_falserefusal.py` | V-4 taken apart: does the library refuse; would round 2 have; is the answer correct by continuity; is it accurate against a degree-16 solve; and does the remedy the refusal names help.  Run on the round-3 tree AND on `f2371e0` |

Every probe records the RESOLVED `lumenairy` path in its JSON and calls
`v_fixtures.assert_tree()`, which refuses to run against a module outside its
own worktree.  That is defect V-3 of the report: this box carries an editable
install pointing at a different checkout, and `python <probe>.py` does not put
the working directory on `sys.path`.

Run with the threads pinned on the command line:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python v2_closure.py v2_closure_win.json

`--fast` on `v2` / `v3` / `v4` takes a small subset, for a smoke run.

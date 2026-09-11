# ROUND 4 -- the sliver guard's decision, taken off the energy reading

Probes for `docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND4_2026_09_11.md`.

The 5.45.0 release CI matrix failed twenty tests of the sliver family because
rounds 1-3 both TRIGGERED and ATTRIBUTED on `max R+T`, which is amplified
rounding through a `1/w^2`-conditioned interface and is therefore a property of
the BLAS kernel. These probes (a) measure that, (b) choose the replacement
criterion from the numbers rather than asserting it, (c) derive its two bars
two-sided, and (d) write one DECISION TABLE per `(build, kernel, threads)`
arm for the cross-arm pin in
`tests/unit/test_fix_pmmstack_sliver_round4.py`.

## Running

Every probe resolves `lumenairy` from its own tree
(`g_fixtures.assert_tree()` -- this box carries an editable install pointing at
a different checkout; round-3 verification defect V-3).

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_CORETYPE=<kernel> python -u <probe>.py [parts...]
```

The thread count is the SECOND arm axis. Pinned arms put the count on the
command line as above; the UNPINNED arm is requested with
`PMM_PROBE_UNPINNED=1`, which makes `g_fixtures` REMOVE all three variables
and then import numpy immediately -- before round 3's `f_fixtures`, whose own
unconditional `setdefault` would otherwise put the pin back before
`libopenblas` is loaded and produce an arm that mis-reports itself. Every JSON
records the count that was REQUESTED and the count `threadpoolctl` says the
run actually used; on this box unpinned resolves to 24, against CI's 4.

`OPENBLAS_CORETYPE` selects the kernel and every JSON records BOTH what was
requested and what OpenBLAS actually dispatched to
(`threadpoolctl`'s `architecture`). They differ: on this CPU `ZEN` and
`SAPPHIRERAPIDS` alias onto `Haswell` bit for bit, and `SKYLAKEX` /
`COOPERLAKE` load but die with `Illegal instruction` on the first real GEMM
(no AVX-512). The four distinct kernel families reachable per build are
Haswell, Katmai (`PRESCOTT`), Sandybridge and Nehalem.

`_run_arms_win.sh` / `_run_arms_wsl.sh` run a list of coretypes in sequence for
one build at one thread. `_run_ladder_win.sh` / `_run_ladder_wsl.sh` run the
FULL cross of a coretype list with threads {1, 2, 4, unpinned}.
`_run_tests_arms.sh` and `_run_tests_threads.sh` do the same for the eight
sliver test files.

**Why the thread ladder exists.** CI's runners are AMD EPYC 7763 (Zen 3, no
AVX-512), so they dispatch to the same Haswell-class kernel this box defaults
to: the kernel ladder alone does not reproduce CI. What CI does differently is
leave BLAS unpinned on a four-core runner. That axis has already bitten this
module once -- `tests/unit/test_m1_conditioning_guard.py` records a closure
moving from `6.65e-06` to `2.14e+01` on the same cell between one and two
OpenBLAS threads -- so it is measured, not assumed away.

## The probes

| file | what it measures |
|---|---|
| `g_fixtures.py` | the shared surface: re-exports round 3's `f_fixtures`, adds `arm()` / `tag()` / `dump()` / `assert_tree()`. |
| `p1_ladder.py` | THE FIRST REFUTATION. Records `d12` for the literal "second `min_feature` snap" proposal and finds it is **exactly 0.0** on all 309 screened rows: snapping at `2 w_wide P` and at `4 w_wide P` merges the same walls to the same midpoints. |
| `p2_candidates.py`, `p2_run.py` | THE SECOND AND THIRD REFUTATIONS, plus the candidate census. Records, per row, the shipped move `d0`, the left/right CLOSURE difference `dLR`, the bracket EXCURSION, and DEGREE stability for the sliver and the snapped solve. `dLR` is round-off (the two closures are a pure TRANSLATION), the excursion is within 0.05 % of `d0` for the same reason, and degree stability overlaps (a CORRECT row at 32.44 against a WRONG one at 2.01). |
| `p3_bars.py` | THE BAR DERIVATION. `d0`, `d0L`, and `d12 = |A_closed - A_closed(+w_wide)|` -- the device's own answer change for a wall displacement of one sliver width -- plus the continuity classification on BOTH polarizations and on the campaign's pol-1 convention, over the ladder / census / D-5 / V-4 / steep / tensor populations. |
| `p4_decisions.py` | THE DECISION TABLE. A fixed 123-row set through `PMMStack.solve` end to end: what the library DID, what the exact `delta -> 0` reference says about what it RETURNED, the arbiter's own quantities, and the energy reading (recorded, not used). One JSON per arm. |

## The JSON

`<probe>_<parts>_<build>_<requested-coretype>_t<threads>.json`, each carrying
an `arm` block: `build`, `requested`, `kernel`, `threads_requested`,
`threads`, `python`, `numpy`. The refutation and bar probes (`p1`, `p2`, `p3`)
were run before the thread axis was added and keep their two-part names; they
are single-arm evidence and nothing is cross-armed from them.

## What the numbers said

* `d0 / d12` is the discriminant; `d0 / w_wide` is a floor. Over 471 screened
  rows: CORRECT reaches 13.5531 and 35.1773, ATTRIBUTED starts at 128.57 and
  110.649, and both bars are 100 -- the campaign's own `err > 100 delta` rule,
  once with the wall step as denominator and once with a MEASURED answer
  change.
* 0 false positives on 251 CORRECT and 67 GREY rows; 4 false negatives in 153
  WRONG rows, all returned under a warning.
* Windows and WSL agree to five to six significant figures on every envelope,
  and 0 of 357 rows common to both differs in class or verdict.

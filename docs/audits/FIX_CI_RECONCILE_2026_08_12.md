# Reconciling the tip-correlated CI failures on three PR branches -- 2026-08-12

`fix/runner-pins` (1657b99), `fix/jax-nan-pins` (3032f4d) and `fix/verify-arch`
(0f46efb) were built in parallel off `origin/main` = **21802f9** (the PR #32
merge).  Each is green on both local mounts on its own scope; all three went
red on the GitHub runners at their tips.

Mounts: **M** = Windows py3.14, numpy 2.4.4, libscipy_openblas64 0.3.31,
24 threads; **W** = WSL py3.12, numpy 2.4.6, OpenBLAS, jax 0.10.2.
Runner = ubuntu-24.04, py3.10 / 3.11 / 3.12 / 3.13.

`fix/runner-pins` has since been **committed at fa8f719** and pushed; this
document is written against that state.  The other two branches' work is
uncommitted in their worktrees (section 9).

---

## 0.  HEADLINE

**Nineteen distinct test ids are red across the three logs.  Nine belong to the
branch they failed on.**  The other ten are two groups that have nothing to do
with the branches:

* **two are a pre-existing MAIN-side defect** -- `origin/main` itself is red on
  them, and has been since PR #30 -- surfaced on all three branches at once
  because these are the first three tips to run the full suite on top of it;
* **eight are the five runner-axis pins `fix/runner-pins` exists to treat**,
  appearing on the branches that do not yet carry its treatment.

Three findings are worth more than the greens:

1. **Three D7 fail-before arms were asserting the amplitude of a NULL-SPACE
   DRAW** (section 6, E2).  A `1.2e-17 m` change to the carrier eikonal --
   below one ULP of the quantity -- flipped them from 0.52 to 1.75e-4.  They
   never measured a library property; a 1-ULP nudge of the fixture's own
   decentre spreads them **5576x**.  Re-armed on the singular solve itself.
2. **The zarr container path is untested on py3.10 and the npz path was
   untested everywhere** (section 6, E4).  `pyproject` carries
   `zarr>=3.0; python_version >= "3.11"`, so the 3.10 CI leg installs no zarr
   and silently takes the npz branch -- which is why two brand-new pipeline
   tests failed on 3.10 jobs only.
3. **The EME 2-D vector mode census is LAPACK-build-dependent by construction**
   (section 5).  `layer_vector_modes._refine_accept` minimises a function that
   is not unimodal at its own detection-bracket scale, so near-threshold
   candidates carry **1.09x-3.2x** acceptance margins that flip with the build.
   The CI workflow already half-knows this ("single-threading is a mitigation,
   not a proof").

### 0.1  Where the failures live

| id | pins | nan | varch | owner | section |
|---|:--:|:--:|:--:|---|---|
| `test_v4_16_0_walker_all_symmetry::test_all_submodule_entries_reexported_or_exempt` | X | X | X | MAIN (PR #30) | 2 |
| `test_v4_15_3_dispatcher_pin_2d_scalar_field::test_all_entry_points_call_helper_first` | X | X | X | MAIN (PR #30) | 2 |
| `test_niche_audit_w3_oracles::test_w4_t1_explicit_sigma_grid_n_64_is_the_pre_fix_default_bit_for_bit` | X | X | X | pins pin 1 | 3, 4 |
| `test_niche_audit_w6_berreman::test_o7_split_fwd_bwd_matches_the_jax_twin_on_physical_tensors` | . | X | X | pins pin 2 | 3 |
| `test_niche_audit_w9_eig_vjp::test_pmm1d_angle_gradient_at_exactly_zero_stays_bounded[te-0.01]` | . | . | X | pins pin 3 | 3, 7 |
| `test_pmm_m2_window_contract::test_threshold_rule_holds_on_a_SINGLE_REGION_uncoated_taper` | . | X | . | pins pin 4 | 3 |
| `test_v5_14_1_rcwa_deferred::test_tapered_grating_shear` | . | X | X | pins pin 5 | 3, 7 |
| `test_niche_audit_w6_berreman::test_o7_partition_claim_survives_a_reorder_and_still_catches_a_side_flip` | X | . | . | pins (own new arm) | 4 |
| `test_pmm_m2_window_contract::test_the_threshold_rules_NOT_CURED_half_conditions_on_the_census_too` | X | . | . | pins (own new arm) | 4 |
| `test_niche_audit_w6_eme::test_w6_6_scaled_cell_keeps_full_recall` | . | X | . | MAIN (EME census) | 5 |
| `test_eme_2d_vector::test_vector_banded_solver` | . | X | . | MAIN (EME census) | 5 |
| `test_audit_except_budget::test_non_ui_except_exception_within_budget` | . | . | X | varch | 6 |
| `test_audit_except_budget::test_non_ui_count_substantially_below_pre_sweep` | . | . | X | varch | 6 |
| `test_niche_d7_decentred_fit::test_the_hard_mask_arm_ghosts_on_every_build` | . | . | X | varch | 6 |
| `test_niche_d7_decentred_fit::test_c13_cures_the_hard_mask_fold_at_the_d7_order` | . | . | X | varch | 6 |
| `test_niche_d7_decentred_fit::test_the_fold_regularisation_is_still_load_bearing_at_the_d7_order` | . | . | X | varch | 6 |
| `test_niche_c6_stationary_phase_launch::test_pure_congruence_input_is_inert_to_rounding` | . | . | X | varch | 6 |
| `test_pipeline::test_an_interrupted_field_write_leaves_the_previous_artifact_intact` | . | . | X | varch | 6 |
| `test_pipeline::test_a_partial_field_checkpoint_is_rejected_not_resumed` | . | . | X | varch | 6 |

Counts as exported: pins 5 distinct ids / 16 (job x id) failures, nan 8 / 17,
varch 14 / 43.

### 0.2  The logs are PARTIAL exports -- absence is not evidence

The exports do not contain every job.  `ci_pins.log` carries 10 jobs and is
missing four unit shards and all three `Slow tests` shards; `ci_verify-arch.log`
carries 13 unit shards + JAX + mypy and no `Slow tests` shard.  Only
`ci_jax-nan-pins.log` contains a `Slow tests shard 3/3 (eig-heavy)` job.

Consequence: **a failure appearing in only one log is not thereby
branch-specific.**  Every cluster below states its baseline explicitly, measured
at 21802f9 in the `C:/tmp/lum_verify` worktree, rather than inferred from which
log it appeared in.  Doing that is what re-attributed both EME clusters
(section 5) from "nan's" to "main's".

---

## 1.  METHOD

Unchanged from `FIX_RUNNER_PINS_2026_08_12` S2:

1. **Cluster by MECHANISM, not by name.**  Nineteen ids are twelve mechanisms.
2. **Adjudicate the runner reading against an oracle independent of the bar,
   BEFORE touching the bar.**
3. **Reproduce by EMULATING the runner axis**, not by tuning until green:
   out-of-process `OPENBLAS_NUM_THREADS` 1 / 2 / 4 / default on W, in-process
   `threadpool_limits` 1 / 2 / 4 / 8 on M, the slow job's full five-variable
   single-thread env, blocking `zarr` to reproduce the 3.10 leg, and the
   in-tree injectors.
4. **Fix at the layer where the wrongness is**, and keep or gain a fail-before.

No `xfail`, no `skip`, no deleted test, no weakened guard, no raised budget
anywhere in this document.  One id is deliberately RENAMED, with reasons, and it
is called out in section 7.

---

## 2.  THE MAIN-SIDE DEFECT -- `origin/main` IS RED, AND HAS BEEN SINCE PR #30

Two ids fail on all three branches and reproduce on the shared base with
nothing applied:

    C:/tmp/lum_verify @ 21802f9, Windows py3.14:
      tests/unit/test_v4_16_0_walker_all_symmetry.py
      tests/unit/test_v4_15_3_dispatcher_pin_2d_scalar_field.py
      -> 2 failed, 18 passed

Introduced by **71b35d6** (2026-08-11, `feat(carrier): CarrierField,
re_reference, aggregate -- the field-summation primitives`), which reached main
as **fae471c**, the PR #30 merge.  Both detectors are OLD (v4.15.3, v4.16.0) and
both are CORRECT: a genuine public-API gap shipped.  No branch caught it locally
because each ran its own scope; all three caught it at once because these are
the first tips to run the whole suite on top of PR #30.

### 2.1  A1 -- fourteen names public in a submodule, invisible at top level

`lumenairy/propagators/__init__.py` re-exports fourteen `carrier_field` names
and lists them in `lumenairy.propagators.__all__`; `lumenairy/__init__.py` never
imported them.  The walker reports the gap twice per name (once for
`lumenairy.propagators`, once for `lumenairy.propagators.carrier_field`) --
28 discovered pairs, 14 names:

    CARRIER_FIELD_SCHEMA  AggregateLedger  AggregateResult  CarrierField
    CarrierSpec  FieldGrid  FieldLedgerRow  NyquistReport  ReReferenceReport
    aggregate  carrier_difference_nyquist  load_carrier_field_zarr
    re_reference  save_carrier_field_zarr

This is the v4.14.0 audit P1-NEW-4 sibling-gap pattern the walker generalises.
Of the three remedies its message offers, **option 1 (re-export) is correct**:
these are the public surface of a shipped feature -- two verbs, nine value
types, a ledger row, two zarr I/O functions, one schema constant -- and their
siblings from the same subsystem (`CarrierReferencedField`,
`TracedCarrierChainResult`, `propagate_traced_carrier_chain`) have been
top-level since v5.32.  Option 2 would retract a shipped API; option 3 would
document the gap rather than close it.

**Fix.**  One import block plus fourteen `__all__` entries, in the same order as
`lumenairy.propagators.__all__`.  Placed **before** `from
.propagators.propagation import (` -- the whole 280-line region is a single
import block to ruff's isort and `carrier_field` sorts before `propagation`
(placing it after the propagation block raises `I001`).  No collision:
`lumenairy.__all__` goes **693 -> 707**, no duplicates.

### 2.2  A2 -- `re_reference` reaches the walker but is not a 2-D scalar entry point

The v4.15.3 meta-pin discovers entry points by first-positional-parameter name
(`_FIELD_PARAM_NAMES`), which is what makes it rename-proof.
`re_reference(field: CarrierField, to_carrier, target_grid, ...)` has a first
parameter named `field`, so it is in scope -- but it is typed `CarrierField`, a
dataclass of (envelope, grid, carrier, wavelength, provenance), not a 2-D scalar
ndarray.  The guard would reject the only valid input type.

**Fix.**  A `_GUARD_EXEMPTIONS` entry with cited rationale, symmetric to the
existing `propagate_ensemble` and `propagate_traced_carrier_chain_multi`
entries.  The 2-D scalar contract is not lost but enforced EARLIER and on every
instance: `CarrierField.__post_init__` raises on `envelope.ndim != 2`, casts a
real envelope to complex, and checks the envelope shape against the grid's,
before any verb can run.

### 2.3  Verification, and how it lands in the merge train

Both fixes touch files **none of the three branches otherwise modifies**, so the
identical patch went into all three worktrees and is **byte-identical** in each
(`lumenairy/__init__.py` sha256 `9ee4d7a0...`; the test file `01dffe73...`).
Git resolves identical changes on both sides without conflict.  It is now
**committed on `fix/runner-pins` as part of fa8f719**, and remains uncommitted
on the other two.

| module set | base 21802f9 | with A1+A2 | delta |
|---|---|---|---|
| walker + dispatcher + `test_public_api` | 717 collected, **2 failed** | 731 collected, **731 passed** | +14 collected (one `test_public_api` parametrization per name), 2 -> 0 failed |

| worktree | M | W | ruff M / W |
|---|---|---|---|
| `lum_pins` | 731 / 731 | 731 / 731 | clean / clean |
| `lum_nan` | 731 / 731 | 731 / 731 | clean / clean |
| `lum_varch` | 731 / 731 | 731 / 731 | clean / clean |

Blast radius on M -- every other module that reads `lumenairy.__all__` or walks
the package: **944 passed, 0 failed** (25 modules, batches of 538 and 406).

---

## 3.  THE FIVE RUNNER-AXIS PINS, SEEN FROM THE OTHER TWO BRANCHES

Eight of the nineteen ids are the five pins `FIX_RUNNER_PINS_2026_08_12` treats,
appearing on `fix/jax-nan-pins` and `fix/verify-arch` only because those
branches do not carry the treatment.  Nothing was fixed for them here; they
clear when fa8f719 merges.

| pin | id | nan | varch | state on `fix/runner-pins` @ fa8f719 |
|---|---|:--:|:--:|---|
| 1 | `w3_oracles::test_w4_t1_explicit_sigma_grid_n_64...` | X | X | GREEN (restructured twice -- section 4, C1) |
| 2 | `w6_berreman::test_o7_split_fwd_bwd..._on_physical_tensors` | X | X | GREEN |
| 3 | `w9_eig_vjp::test_pmm1d_angle_gradient_at_exactly_zero_stays_bounded[te-0.01]` | . | X | GREEN, **id renamed** -- section 7 R3 |
| 4 | `pmm_m2_window_contract::test_threshold_rule_holds_on_a_SINGLE_REGION_uncoated_taper` | X | . | GREEN |
| 5 | `v5_14_1_rcwa_deferred::test_tapered_grating_shear` | X | X | GREEN + reconciled -- section 7 R1 |

Pin 5 is additionally green on `fix/jax-nan-pins` via the R1 reconciliation,
because that branch edits the same function.  `fix/verify-arch` takes all five
from the merge train: duplicating test-side pin work onto a library branch would
create three copies of the same restructuring to reconcile later, which is the
failure mode this document exists to clean up.

---

## 4.  `fix/runner-pins` -- its own new arms, meeting real runners

Three of its five restructurings shipped NEW arms (fail-befores and
preconditions) that had never met a real runner.  All three fixes are test-side;
nothing under `lumenairy/` changed.  **Committed at fa8f719.**

| # | mechanism | root cause | fix |
|---|---|---|---|
| C1 | `sep > 1e-2` between the n=64 escape hatch and the adaptive default; runner read 5.06e-3 | `sep` is a DIFFERENCE OF TWO GRID-ALIASING ERRORS (n=64 vs n=256), both entitled to move ~2% per build; at R1=51.5 the whole ladder lives in one 5% window and `sep` is NON-MONOTONE in n | split into what the bar really guarded: the default is adaptive -- asserted on the **resolved integer** `res.sigma_grid_n` (256 at R1=51.5, 192 at R1=60); and the grid is load-bearing -- `sep > 1e-6`, a round-off bar, since a flat default makes the two arms the same call and reads exactly `0.0` |
| C2 | fail-before precondition `corr == total`; runner read 78/80 | the branch's own disease one layer up -- raw eigen ORDER asserted as fact.  Not degeneracy: min pairwise separation **8.556e-03** | SELECT the correspondence class; measure the injector's REACH (a jax-vs-jax gauge experiment) and assert the exact identity `disturbed == \|class INTERSECT reach\|`; restrict the verbatim re-run to the class; assert `ra["corr"] < base["corr"]`; add a `corr > 0` floor to the SHIPPED test |
| C3 | near-cut injector frozen at x3 fails to disarm the census | a CUT POSITION frozen as a number; where the cut sits is per-build AND per reduction width | `_first_disarming_scale` walks `_DISARM_SCALES = (1.0, 3.0) + (W, W^2, W^3, W^4)` with `W = _MODE_CUT_MARGIN_WARN` -- the scale is read off the instrument's OWN definition (a mode counts as growing only within `W` of the cut); ceiling `W^4 = 1e4 = 1/min(_INJECTOR_SCALES)` asserted |

C1's non-monotone ladder on M, `sep` vs n=64: 2.713e-1 (96), 1.517e-1 (128),
4.794e-2 (192), **2.889e-2 (256)**, 2.165e-2 (384), 3.744e-2 (512) -- and
bit-identical at `threadpool_limits` 1/2/4/8/default, so the axis is the build
and the runner reading is not locally reproducible.  The fix does not depend on
reproducing it.

C2's injector table (80 draws): shipped 80/80 corr, invariant 2.850e-15,
elementwise 1.158e-14; **partial reorder of 2 draws** 78/80 -- the runner's
condition, reproduced -- invariant and elementwise unmoved; full reorder 0/80,
elementwise 1.4925 with 75 disturbed = the measured reach; side flip 0/80 with
invariant 1.9988.  Driving the 2-draw injector through the BRANCH-TIP helper
raises the runner's sentence verbatim while the tip's shipped claim stays green:
the precondition was the only broken thing.

C3 **reproduced locally** -- WSL py3.12 at `OPENBLAS_NUM_THREADS` 2 and 4, x3
leaves `[0,0,0,1,0]` and the tip test fails with the identical assertion.
Ceiling oracle: the disarmed pre-repair ladder is bit-identical (0.0) to the
shipped repair-ON ladder at x10 / x1e2 / x1e3 / x1e4 / x1e6 and reads 7.6961e-09
at x1e9 where `n_prop -> 0` on every rung -- while the SPREAD is unchanged to six
figures, i.e. the spread cannot see an over-escalation and the bit-identity
oracle can.

### 4.1  Green, and the cost -- which was itself a finding

| module | tip M | here M | tip W | here W |
|---|---|---|---|---|
| `test_niche_audit_w3_oracles.py` | 180 | **181** | 180 | **181** |
| `test_niche_audit_w6_berreman.py` | 429 | 429 | 429 | 429 |
| `test_pmm_m2_window_contract.py` | 20 | 20 | 20 | 20 |
| **total** | **629** | **630** | **629** | **630** |

630 passed on both mounts (132 s each at one BLAS thread); the m2 module also
green on W at `OPENBLAS_NUM_THREADS` 1 / 2 / 4 / default; ruff clean on all
three files on both mounts.

**The first C3 attempt walked a six-rung ladder of full PMM solves.**  Measured
from outside, its two verification runs had each accumulated **63,438 s of CPU
at 828% over 2 h 08 m** on a module the committed `.test_durations` records at
**39.0 s total over 16 ids**, with pin 4's own neighbour id at **4.22 s**.  None
of the 19 ids in that file is `@slow`, so all of them are charged to a
wall-clock-capped unit shard balanced by `pytest-split` -- and a new id has no
durations entry, so it would have been placed blind.  That was stopped and
re-derived.  The landed version LOCATES the cut from the instrument's own margin
instead of walking to it, and is **cheaper than the tip**:

| id | tip M | here M | tip W | here W |
|---|---|---|---|---|
| m2 module | 48.11 s | **47.71 s** | 45.12 s | **44.65 s** |
| C3 id | 1.92 s | **0.96 s** | 1.75 s | **0.88 s** |
| C1 id | 5.06 s | 5.09 s | -- | -- |
| C2 fail-before | 2.64 s | 4.11 s | -- | -- |

C3 is *half* the tip because the cache fix (below) turns the nested probe into a
hit.

### 4.2  Two latent test-side defects found on the way

* **`near_cut_injector` composed its thresholds but not its cache key.**  C3's
  upward injector wraps `_uncured_below_threshold`'s downward probe; the
  thresholds compose multiplicatively, but `_CUT_SCALE` -- part of
  `_LADDER_CACHE`'s key -- was SET to the inner scale instead of composed.  On a
  build where the sibling test also reaches that probe branch un-injected (i.e.
  the runners), two different effective cuts would have shared one cache entry.
  Fixed.
* **A vacuity hole in the shipped O7 physical-tensor test**: it had no floor on
  its correspondence class, so at `corr == 0` its element-wise half would have
  passed by measuring nothing.  Fixed with a `corr > 0` floor.

---

## 5.  `fix/jax-nan-pins` -- and the EME mode census, which is main's

Both of this branch's own red ids turned out to be **pre-existing and
main-side**.  The branch's only `lumenairy/` change is `pmm/_core.py`;
`git diff --quiet 21802f9 3032f4d -- lumenairy/elements/eme/eme_2d_vector.py`
reports the file **identical**.  Baselines at 21802f9 under the slow gate's
five-variable single-thread env: `test_eme_2d_vector.py` **19 passed** on W
(599.5 s) and on M (728.4 s); `test_niche_audit_w6_eme.py` 75 passed on both.
So both clusters are hardware/build-conditional, not branch-caused -- and only
visible here because this is the only run whose slow shard was exported (0.2).

| # | mechanism | root cause | fix | layer |
|---|---|---|---|---|
| N1 `w6_eme::test_w6_6_scaled_cell_keeps_full_recall` | the pin compared the two arms' ACCEPTED MODE CENSUS element-wise | `layer_vector_modes` accepted a 4th candidate (`qz^2 = 235.8686333`) in the base arm only.  That candidate is a DOUBLE ZERO of the mode condition (`sigma_min ~ C sqrt\|q - q*\|`); its rank-drop reading is taken wherever bounded-Brent stops, giving `gaps.min` 2.077e-3 (base) / 3.246e-3 (scaled) against `ratio_tol = 1e-3` -- a **2.1x / 3.2x coin flip** the ubuntu LAPACK landed on the other side of | claim re-stated on the CONVERGED ZEROS (bidirectional, both arms), plus grid-size equality and a bounded census-difference allowance; new fail-before | test |
| N2 `test_eme_2d_vector::test_vector_banded_solver` | `np.allclose(md[:3], mb[:3])` -- a POSITIONAL slice of a build-dependent census at unjustified default tolerances | BOTH arms gained a 4th mode `205.9749734` at the front on the runner, shifting the slice; the compared 3rd entries then differed by 1.94e-3 (1.33e-5 rel) against allclose's 1e-5 rtol.  `205.975` is a GENUINE mode (converged `sigma_min = 2.69e-17`) that both our mounts MISS, because bounded-Brent traps on a ~1e-3 wiggle 3.7e-3 from the root and reads `gaps.min = 1.0918e-3` -- a **1.09x** margin | three layers: pointwise banded-vs-dense, converged-zero coincidence, value-matched census; new fail-before with an injector ladder | test |

### 5.1  The adjudication

**N1 -- a true double zero, not a tolerance graze.**  Shipped acceptance
readings at every detected dip, identical M and W:

    qz^2         gaps.min base   gaps.min scaled   verdict   vs bar 1e-3
    156.281376   4.846e-08       5.430e-07         ACCEPT    20000x under
    203.716176   1.863e-08       4.342e-07         ACCEPT
    208.250260   2.076e-07       7.468e-07         ACCEPT
    180.770337   1.613e-03       2.987e-03         reject    1.6x / 3.0x
    235.868633   2.077e-03       3.246e-03         reject    2.1x / 3.2x  <- the CI flip
    (11 others)  2.5e-01 .. 6.1e-01                reject    2-3 decades over

Oracle (nested-bracket dense polish, 7-10 refinement levels, **byte-identical on
M and W**): 235.868633368217 gives `sigma_min = 3.999e-08`, `gaps.min =
3.788e-07`.  Depth profile 7.4e-3 / 2.34e-3 / 7.1e-4 / 8.5e-5 at `dq` 1e-3 /
1e-4 / 1e-5 / 1e-6 -- **sqrt scaling**, so no minimiser can read below
~`C sqrt(sqrt(eps)|q|)` ~ 4e-4.  Irreducible.

Hypotheses REFUTED: threads (identical at 1/2/4/8 on both mounts); JAX-in-process
(run in one process with its five jax-job shard neighbours: 866 passed on W,
300 on M -- `layer_vector_modes` never touches jax); this branch (baseline
above).

**N2 -- 205.975 is a real mode and the two solvers share roots exactly.**
`sigma_min` is non-unimodal at ~1e-3 amplitude on the shipped bracket with a deep
V at 205.9749758.  Brent ends at 205.9786352762 (`f = 3.45e-5`) on M and at
205.970491503 -- the FIRST golden probe (`f = 5.17e-5`) -- on W; both read
`gaps.min = 1.0918e-3`.  Converged: `sigma_min = 2.69e-17`, indistinguishable
from the four accepted modes (1.2e-15..1.6e-15).  Not degeneracy: converged
dense-vs-banded root offset **0.0e0** at every mode (max 5.14e-9 over all
probes); pointwise `max |banded/dense - 1| = 1.46e-3` off the roots.  The banded
solver is not less accurate.

**A library fix was measured and rejected as not contained**: localising on a
33-point sub-grid before Brent takes N2's census 4 -> 5 with dense and banded
agreeing to 9 digits (fixes N2 outright), leaves N1 unchanged at 3/3 (the sqrt
zero is unreachable), and costs 2885 -> 5400 `sigma_min` evaluations (**1.87x**
on the eig-heavy slow gate) while changing the returned census on every cell.
Reported, not shipped -- see section 8.

New bars, all two-sided and measured: `_W6_SCALE_ROOT_REL = 1e-6` (true zeros
<= 9.13e-9, non-modes >= 9.75e-5); `_W6_ZERO_DEPTH = 2e-3` (<= 2.28e-4 vs
>= 2.20e-2); `_BANDED_ROOT_REL = 1e-6` (<= 5.14e-9); `_BANDED_DEPTH = 1e-6`
(<= 2.06e-8 vs >= 2.5e-2); `_BANDED_POINTWISE_REL = 1e-2` (1.46e-3 shipped,
breaks at `iters <= 10` -> 2.44e-2); `_BANDED_PARTNER_REL = 1e-4` /
`_BANDED_DISTINCT_REL = 5e-3` (worst partner 1.33e-5 on the runner; closest
distinct pair 1.98e-2).

### 5.2  Green

| module | mount | before (= tip = base) | after |
|---|---|---|---|
| `test_niche_audit_w6_eme.py` | M | 75 passed | **76 passed** (193.3 s) |
| `test_niche_audit_w6_eme.py` | W | 75 passed | **76 passed** (166.9 s) |
| `test_eme_2d_vector.py` | M | 19 passed | **20 passed** (1168.5 s) |
| `test_eme_2d_vector.py` | W | 19 passed | **20 passed** (767.2 s) |

The +1 per module is one fail-before each; both target ids keep their names.
Thread sweep of the restated N1 pin: 4 passed at 1/2/4/8 on both mounts.  Ruff
clean on both files on both mounts.

---

## 6.  `fix/verify-arch` -- its own clusters

| # | mechanism | root cause | fix | layer |
|---|---|---|---|---|
| E1 | non-`ui/` `except Exception:` count 49 vs budget 48 | the branch added exactly ONE new site: `_lens_imap.py:784`, `_incumbent_fingerprint`'s probe of `parity_invert`.  (The `validation/pipeline/artifacts.py` site does not count -- the scanner walks `lumenairy/` only) | NARROWED to `_INCUMBENT_PROBE_ERRORS = (ValueError, TypeError, ArithmeticError)`.  Budget NOT raised | library |
| E2 | three D7 arms all read `0.00017548596271227133` against `> 0.1`, every python | `_lens_traced.py:3848` (`_tilted_carrier_parts`, untilted arm) moved the carrier eikonal by **1.214e-17 m**; the arms were asserting the AMPLITUDE OF A NULL-SPACE DRAW, decided at exactly that scale | re-armed all three at the SOLVE (deterministic); magnitude recorded, not asserted | test -- library correct |
| E3 | `grad_a_rms` 1.3378644187e-13 vs `< 1e-14` | the test's `_carrier_WLM` fixture still used the SUBTRACTION sphere while the branch rationalized `_compute_carrier`; the "pure congruence" input was no longer the library's congruence | fixture follows the library form (its own docstring already claimed it did); bars re-stated as a float64 ceiling and a same-process ratio | test -- real branch finding |
| E4 | two new pipeline tests, **python 3.10 jobs only** | `pyproject` keys the extra `zarr>=3.0; python_version >= "3.11"`, so the 3.10 CI leg installs NO zarr -> `save_field` takes the npz branch: no chunk files to hole, and the patched zarr writer never called | tests made container-agnostic; interrupt injected at BOTH payload writers with a `reached` assertion | test -- library correct |

### 6.1  E2 -- the headline adjudication: NEITHER (a) nor (b)

The arms were never measuring a library property.  Bisect, driving `r_old`
directly (hard mask + pre-C13 + degree 4):

| tree state | `r_old` |
|---|---|
| base `21802f9` | **5.2129148080e-01** |
| branch `0f46efb` | **1.7549395434e-04** |
| branch, revert `_lens_traced.py` only | 5.2129148080e-01 |
| branch, revert the `_compute_carrier` hunk only | 1.7549395434e-04 (**no change**) |
| branch, revert the `_tilted_carrier_parts` hunks only | 5.2129148080e-01 (**restored**) |

One line.  `max|W_rat - W_sub| = 1.214e-17 m` (`eps*|R| = 1.332e-17 m`) against
`max|W| = 2.895e-03 m`.

That it is a coin flip, three independent ways, all in-process:

* **1-ULP nudges of the fixture's own decentre** (8.674e-19 m):
  `1.755e-04 / 0.942 / 0.978 / 0.674 / 0.946` -- spread **5576x**.  With C13 ON
  the same five read `1.7548444178e-04 .. 1.7548444187e-04`, stable to **nine**
  significant figures.
* **IID noise of 1e-18 m** on the shipped eikonal: seed 0 -> 1.755e-04,
  seed 1 -> 0.194, seed 2 -> 0.362.  The seed decides.
* **Blend** `W_rat + t(W_sub - W_rat)`: a cliff between t=0.5 and t=0.75, and
  non-monotone (0.700 at t=0.75, 0.521 at t=1.0).

Not (a) -- the branch did NOT fix the fold.  Not (b) -- the emulation path is
still reached; the census proves the singular solve is taken and answered by a
draw:

| config | min rcond | max `\|\|b-Ax\|\| / \|\|b-Ax_qr\|\|` |
|---|---|---|
| hard mask, pre-C13 | 0.0e+00 | **36972.83** |
| hard mask, pre-C13, +1 ULP | 0.0e+00 | **233325.49** |
| hard mask, C13 ON | 0.0e+00 | 1.000000 |
| weighted disc, pre-C13 | 1.336e-11 | 1.000003 |
| weighted disc, C13 ON (shipped) | 1.336e-11 | 1.000000 |

The arms' own docstrings already recorded three build answers spanning four
orders and said *"no bar on this magnitude can be both meaningful and true"* --
then kept one anyway.  The branch produced the fourth.  Re-armed on the census
(`_DRAW_RESID_RATIO = 10.0`, 324x under the smallest draw ever seen;
`_CURED_RESID_RATIO = 1.001`, three decades over the library's own tie margin).
**Fail-before proven**: making the C13 step-down unconditional in
`_solve_lstsq_thread_safe` fails all three (`assert 1.0000000000000175 > 10.0`),
then restored.

This is **not** a runner axis -- the branch is red on M AND W for these ids.  The
campaign's S11 records that it ran targeted files only, so d7 and c6 were never
run at all.

### 6.2  E1 / E3 / E4 numbers

* **E1**: base count 48, branch 49, after narrowing **48**.  The shipped G8 arm
  calls `parity_invert` unguarded ~420 lines below, so swallowing a defect only
  moves the traceback -- hence narrow, not keep.  Both budget ids read `<= 48`.
* **E3**: same process -- stale fixture `grad_a_rms` 1.338e-13; fixture
  following the library **3.459e-17**; non-congruent control **6.261e-04**.
  `max|W_sub - W_rat| = 6.586e-18 m` with grid gradient **4.011e-13**, which is
  exactly the 1.338e-13 reading.  New bars: a float64 ceiling
  `eps*max|W|/dx = 3.400e-15` (measured 98x under; the stale value was 39x over)
  and a ratio-vs-control `<= 1e-11` (measured 5.5e-14 / 4.8e-14; stale
  2.14e-10 / 2.76e-10).
* **E4**: zarr 3.1.6 (M) / 3.2.1 (W) / 3.1.6-3.3.0 (CI 3.11-3.13) / **none on
  3.10**.  Reproduced exactly on W with zarr blocked: 2 failed / 46 passed, the
  same two ids.  **No library defect** -- `save_field` stages both containers
  under `.tmp-<pid>` and `os.replace`s them, `field_power_on_disk` /
  `field_is_complete` read both, `driver.py` gates on content for both.  Proven
  by de-staging `save_field` on the npz path: both arms fail, then restored.
  Coverage GAINED: the npz branch's interrupt-atomicity and holed-container
  rejection were previously untested anywhere.

### 6.3  Green

| module | Windows before / after | WSL before / after |
|---|---|---|
| `test_audit_except_budget.py` | 0/2 -> **2/2** | 0/2 -> **2/2** |
| `test_niche_d7_decentred_fit.py` | 34/37 -> **37/37** | 34/37 -> **37/37** |
| `test_niche_c6_stationary_phase_launch.py` | 20/21 -> **21/21** | 20/21 -> **21/21** |
| `test_pipeline.py` | 48/48 -> **48/48** (and **48/48** zarr-blocked) | 46/48 zarr-blocked -> **48/48** both ways |
| `test_niche_c15_inverse_map.py` | 33/33 -> **33/33** | 33/33 -> **33/33** |

WSL combined: **141 passed** (2575 s).  `ruff check lumenairy/ tests/unit/` --
the exact CI command -- clean on both mounts.  Final non-`ui/`
`except Exception` count: **48** (budget 48).

---

## 7.  THE CROSS-BRANCH CONFLICT, RECONCILED

`fix/runner-pins` and `fix/jax-nan-pins` touch THREE files in common.  A trial
three-way merge (`git merge-tree 21802f9 fix/runner-pins fix/jax-nan-pins`)
resolves the exact state:

| file | conflict hunks | why |
|---|---|---|
| `tests/unit/test_v5_14_1_rcwa_deferred.py` | **3** | both rewrote the SAME function |
| `tests/unit/test_niche_audit_w6_berreman.py` | 0 (auto-merges) | adjacent blocks -- see R2 |
| `tests/unit/test_niche_audit_w9_eig_vjp.py` | 0 (auto-merges) | disjoint hunks -- see R3, which is the trap |

The brief named the shear pin and the o7 region; the **w9 eig-VJP file is a
third overlap that was not on the list**, and it carries the one real trap.

### R1 -- `test_tapered_grating_shear`: the only textual conflict

Each branch re-stated exactly one of the function's two claims and left the
other absolute:

| claim | `fix/runner-pins` | `fix/jax-nan-pins` |
|---|---|---|
| ENERGY closure `R + T = 2` | left ABSOLUTE `1e-7` (its own S9 watch item) | RELATIVE `_SHEAR_ENERGY_REL = 1e-6` |
| SYMMETRY `+/-1` and mirror-shear | RELATIVE `_SHEAR_SYM_REL = 1e-9` | left ABSOLUTE `1e-12` |
| builder | `_shear_build(shear)` | `_shear_stack(shear, eps_ridge=4.0)` |
| fail-before | one raster pixel of shear | a lossy ridge, linear ladder |

**And each branch went red on the runner on precisely the bar it had not
treated.**  `fix/jax-nan-pins` failed the absolute symmetry line
(`|0.33449837513783104 - 0.3344983751392616| = 1.4306e-12` vs `1e-12`).

Reconciled, taking the jax-nan form where it supersedes: one builder --
`_shear_stack(shear, eps_ridge=4.0)`, a strict generalisation carrying the knob
the energy fail-before injects through; both envelope constants with both
derivation tables; both relative bars; both fail-befores.  `_shear_build` is
gone.

The result is **byte-identical** in the committed `fix/runner-pins` tree
(fa8f719) and the `fix/jax-nan-pins` worktree -- verified after the commit,
sha256 `67e9f8af...` on both -- so the merge is a **no-op** on this file rather
than a resolved conflict.  Both runner readings now pass with room: symmetry
`1.4306e-12 / 0.3345 = 4.28e-12` vs `1e-9` (234x), energy `2.25e-8` vs `1e-6`
(44x).

| module | base | pins tip | nan tip | reconciled |
|---|---|---|---|---|
| `test_v5_14_1_rcwa_deferred.py` | 20 | 21 | 21 | **22** |

| worktree | M | W | ruff M / W |
|---|---|---|---|
| `lum_pins` (now committed) | 22 / 22 (64.9 s) | 22 / 22 (205.8 s) | clean / clean |
| `lum_nan` | 22 / 22 (82.5 s) | 22 / 22 (194.5 s) | clean / clean |

### R2 -- the o7 region: below-disjoint, and reconciled against fa8f719

`fix/runner-pins` owns the PHYSICAL family and everything above
`test_o7_split_fwd_bwd_matches_the_jax_twin_in_the_degenerate_fallback`; its
committed edits span lines **641-982** (`_o7_reorder_eig`, the dict-returning
`_o7_score` with `sel=` / `reorder=`, the physical-tensor test with its new
`corr > 0` floor, `_o7_reorder_reach`, and the rewritten side-flip fail-before).
`fix/jax-nan-pins` owns the DEGENERATE fallback below it (`_o7_general_draws`,
`_o7_legacy_decay_modes`, `_o7_degenerate_score`, the rewritten degenerate test
and its rule-fork fail-before).

Nothing competes: the two branches independently converged on the SAME
treatment -- per-draw power-sum precondition, correspondence class, claim
restricted to the class -- on two different tensor families, which is why the
merge is clean.  (The jax-nan form is the older of the two and was used as the
prior art for the runner-pins C2 fix.)

Reconciled by adopting the **committed fa8f719 head verbatim** and splicing the
jax-nan degenerate block below it, into the `fix/jax-nan-pins` worktree only.
All twelve o7 symbols present exactly once -- no duplication, no loss:

    _o7_draws  _o7_reorder_eig  _o7_score  _o7_reorder_reach
    _o7_general_draws  _o7_legacy_decay_modes  _o7_degenerate_score
    test_o7_split_fwd_bwd_matches_the_jax_twin_on_physical_tensors
    test_o7_partition_claim_survives_a_reorder_and_still_catches_a_side_flip
    test_o7_split_fwd_bwd_matches_the_jax_twin_in_the_degenerate_fallback
    test_o7_degenerate_claim_survives_a_reorder_and_still_catches_a_rule_fork
    test_o7b_gain_layer_matches_the_scalar_tmm_exactly

| module | base | pins @ fa8f719 | nan tip | reconciled (nan worktree) |
|---|---|---|---|---|
| `test_niche_audit_w6_berreman.py` | 428 | 429 | 429 | **430** |

Each branch added exactly one arm to this module and they are different arms
(runner-pins rewrote the physical-family fail-before in place; jax-nan added the
degenerate rule-fork fail-before), so the merged count is 428 + 1 + 1.

| mount | result | ruff |
|---|---|---|
| M | **430 / 430 passed** (19.0 s, 1 BLAS thread) | clean |
| W | **430 / 430 passed** (18.0 s, 1 BLAS thread) | clean |

### R3 -- `test_niche_audit_w9_eig_vjp.py`: merges clean, but the test and the
library change MUST travel together

The hunks are disjoint -- jax-nan extends
`test_pmm1d_off_normal_angle_gradient_no_regression`'s docstring and adds
`test_the_lstsq_projection_route_is_refuted_on_a_degenerate_projection` at base
line 311; runner-pins edits the module docstring (31-48) and restructures pin 3
(336-357).  Git auto-merges with zero conflict markers.

**The trap:** the jax-nan fail-before monkeypatches
`lumenairy.elements.pmm._core.PMM_JAX_MINNORM_PROJECTION` and calls
`_jpmm_min_norm_projection`.  Neither symbol exists on `fix/runner-pins`
(`grep -c` on `lum_pins/lumenairy/elements/pmm/_core.py` = 0; on `lum_nan` = 5)
-- they ARE the jax-nan S3 library fix.  Cherry-picking the test hunk without
`lumenairy/elements/pmm/_core.py` gives a green collection and a run-time
`AttributeError`.  That is why R3 was **not** cross-applied the way R1 was.

Verified directly instead: the post-merge file was constructed and run in
`lum_nan`, the worktree that carries the library change.

| module | base | pins tip | nan tip | post-merge |
|---|---|---|---|---|
| `test_niche_audit_w9_eig_vjp.py` | 29 | 30 | 30 | **31** |

| mount | post-merge | ruff |
|---|---|---|
| M | **31 / 31 passed** (201.0 s) | clean |
| W | **31 / 31 passed** (518.7 s) | clean |

`lum_nan`'s own copy was then restored; the verification artefact is kept at
`scratchpad/_w9_merged.py` for whoever performs the merge.

**One naming note the merge reviewer must know.**  Pin 3 does not merely
re-parametrize -- it RENAMES the function
`test_pmm1d_angle_gradient_at_exactly_zero_stays_bounded` ->
`test_pmm1d_angle_gradient_at_exactly_zero_is_an_OPEN_defect`, and `[te-0.01]` /
`[tm-0.5]` -> `[te]` / `[tm]`.  Its own S8 records only the parametrization half.
The rename is defensible and is not a rename-to-green: the old name asserted
"stays bounded", which the pin never established (it fences a value it agrees is
WRONG), and the new form keeps BOTH halves as in-process ratios plus a
two-directional fail-before.  But it is the only id in this campaign whose NAME
changes, so anyone grepping the CI id `...stays_bounded[te-0.01]` after the
merge will not find it.

---

## 8.  MERGE TRAIN

**Order: `fix/runner-pins` -> `fix/jax-nan-pins` -> `fix/verify-arch`.**

1. **`fix/runner-pins` @ fa8f719** -- already committed and pushed.  Clears
   eight of the nineteen ids in one merge (its own five pins on all three
   branches plus its three new arms) and carries the main-side A1+A2 fix.
2. **`fix/jax-nan-pins`** -- the three shared files are pre-reconciled against
   fa8f719: `test_v5_14_1_rcwa_deferred.py` is byte-identical (no-op merge),
   `test_niche_audit_w6_berreman.py` already contains the fa8f719 head verbatim,
   and `test_niche_audit_w9_eig_vjp.py` auto-merges -- but **merge its
   `lumenairy/elements/pmm/_core.py` hunk in the same commit as its w9 test
   hunk**, never separately (R3).  Its A1+A2 hunks are byte-identical to
   fa8f719's and merge as no-ops.
3. **`fix/verify-arch`** -- disjoint from both others (`git merge-tree` reports
   no file changed in both, in either pairing).  Its A1+A2 hunks are again
   byte-identical no-ops.  It takes all five runner-axis pins from step 1.

### 8.1  Open items for the train -- not fixed here

* **`lumenairy/elements/eme/eme_2d_vector.py::layer_vector_modes._refine_accept`
  returns a LAPACK-build-dependent mode census.**  Two independent causes:
  scipy's bounded Brent terminates on `tol1 = sqrt(eps)*|x| + xatol/3`, so
  `xatol=1e-7` buys nothing below ~3.5e-6 at `|q| ~ 236`, and it is a LOCAL
  minimiser on a function that is not unimodal at the detection-bracket scale
  (measured: it stops at 205.9786 against the root 205.9749758); and at a double
  zero the reading is floor-limited at ~4e-4 regardless of minimiser.
  Consequences, both observed on the runner: near-threshold candidates have
  **1.09x-3.2x** acceptance margins that flip with the build (recall 4 vs 5 on
  the `test_eme_2d_vector` grating, 3 vs 4 on the w6 cell), and the returned
  `qz^2` for such a candidate carries up to ~4e-3 absolute error (runner's
  201.88626458 against a converged 201.8868828 = 3.1e-6 relative).  Both
  restated pins are now immune, with the derivation in-tree.  The measured
  candidate fix costs **1.87x** on the eig-heavy gate and changes the census on
  every cell -- it needs its own round with a re-validation of the whole slow
  EME gate.
* **`pyproject`'s `zarr>=3.0; python_version >= "3.11"` marker** leaves the
  entire zarr container path untested on the 3.10 leg while the npz path was
  untested everywhere else.  The pipeline arms now cover both on every python;
  the same lens should be applied to
  `tests/unit/test_audit_s4_9_io_silent_fallback.py`, which still SKIPs its zarr
  parity test on 3.10.
* **`.test_durations` has no entry for any of the new ids.**  Module totals are
  unchanged (section 4.1), so the shard risk is bounded, but regenerating
  durations remains the durable fix -- this repo has already lost a release-tag
  verify shard to a stale entry.
* **BLAS oversubscription is a three-decade wall-clock axis on this box**: the
  m2 module takes over two hours with four uncapped 24-thread pytest processes
  and 48 s at one thread, identically at the tip.  A measurement hazard, not a
  defect -- but it is why every timing in this document names its thread count.

---

## 9.  UNCOMMITTED STATE PER WORKTREE

**`C:/tmp/lum_pins` (`fix/runner-pins` @ fa8f719) -- CLEAN, nothing
uncommitted.**  fa8f719 contains: `docs/audits/FIX_RUNNER_PINS_2026_08_12.md`
(+329, new S3.1 / S4.1 / S5.1 / S8.0 / S8.0.1), `lumenairy/__init__.py` (+42,
A1), `tests/unit/test_niche_audit_w3_oracles.py` (C1),
`tests/unit/test_niche_audit_w6_berreman.py` (C2),
`tests/unit/test_pmm_m2_window_contract.py` (C3),
`tests/unit/test_v4_15_3_dispatcher_pin_2d_scalar_field.py` (+17, A2),
`tests/unit/test_v5_14_1_rcwa_deferred.py` (+106, R1).

**`C:/tmp/lum_nan` (`fix/jax-nan-pins` @ 3032f4d) -- 6 files modified,
uncommitted:**

    M lumenairy/__init__.py                                        (+42)  A1
    M tests/unit/test_eme_2d_vector.py                             (+196) N2
    M tests/unit/test_niche_audit_w6_berreman.py                   (+390) R2
    M tests/unit/test_niche_audit_w6_eme.py                        (+171) N1
    M tests/unit/test_v4_15_3_dispatcher_pin_2d_scalar_field.py    (+17)  A2
    M tests/unit/test_v5_14_1_rcwa_deferred.py                     (+85)  R1

**`C:/tmp/lum_varch` (`fix/verify-arch` @ 0f46efb) -- 7 files modified + this
document, uncommitted:**

    M lumenairy/__init__.py                                        (+42)  A1
    M lumenairy/elements/_lens_imap.py                             (+31)  E1
    M tests/unit/test_niche_c15_inverse_map.py                     (+18)  E1 teeth
    M tests/unit/test_niche_c6_stationary_phase_launch.py          (+73)  E3
    M tests/unit/test_niche_d7_decentred_fit.py                    (+269) E2
    M tests/unit/test_pipeline.py                                  (+68)  E4
    M tests/unit/test_v4_15_3_dispatcher_pin_2d_scalar_field.py    (+17)  A2
    ?? docs/audits/FIX_CI_RECONCILE_2026_08_12.md                         this file

`C:/tmp/lum_verify` (the 21802f9 baseline worktree) is unmodified.
`C:/tmp/lum_base` is a plain file copy used as a tip control by the pins work.

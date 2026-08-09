# ADDENDUM to HANDOFF_TRACED_EXACT_2026_08_05.md
**2026-08-05, written after the handoff went out.** Corrects two claims in the
original handoff and in commit `6dfc79d`'s message. Read this before acting on
either.

> **Bottom line:** the `newton_fit` default flip to spline in `6dfc79d` is
> **WRONG and has been reverted** in the working tree. And the test-status
> claim in that commit ("only a pre-existing `filelock` failure") was made on an
> INCOMPLETE regression run — the completed run has 15 genuine failures caused
> by that flip. Everything else in the original handoff stands.

---

## 1. CORRECTION — `newton_fit` must stay `polynomial`

The original handoff §1 lists `newton_fit` changing to `'auto'` -> spline on CPU.
**Do not ship that.** Reverted: `'auto'` now resolves to `polynomial` everywhere.

### Why (this is the substantive part)

Selecting the spline backend **silently disables the ray-fit-radius
restriction**. Two guards in `_lens_traced.py` are written
`... and newton_fit != 'spline'`:

```
6746:                and newton_fit != 'spline'):
7297:    if _fit_r_max is not None and newton_fit != 'spline':
```

That restriction is `fit_radius_beam_factor`, and it is **not optional for real
designs**. Design 121's post-DOE groups carry 20–32 mm apertures against a
sub-millimetre beam (~75×), far past the 1.5× aperture:beam ratio at which the
traced OPL fit is corrupted by marginal rays the beam never occupies. Measured
this session: without it the post-DOE exit field comes back **non-finite**.

So the flip would have quietly turned off an accuracy guard on exactly the chain
this campaign exists to serve. A fit-backend choice must not disable an accuracy
guard as a side effect. The justification for the flip was a modest parallel
speed-up (1.29–1.31× vs 1.10–1.13× at 8 workers) on a metric where the two fits
are otherwise **tied** — differences sit in the 4th–5th significant figure and
swap direction with `ray_subsample`. That does not buy a silent guard removal.

Spline remains fully supported as an **explicit** `newton_fit='spline'`.

### What the 15 failures actually were

`test_niche_c6_fit_guard` (2), `test_niche_c11_decentred_fit_arbiter` (7),
`test_niche_c12_physics_fit_selection` (6). Causation confirmed by forcing
polynomial: **54 passed**; with spline, 15 failed.

They are not stale calibration. They exercise machinery that only runs on the
polynomial path — the stationary-phase fit guard, the decentred-fit arbiter,
physics fit selection — so under spline the toggles they flip become no-ops and
the tests correctly detect that the feature is gone. Note the failure *reads*
like a trivial one (`assert not np.array_equal(a, b)` failing with what look
like all-zero arrays); the arrays are not zero, that is just zero corners in the
repr. The real content is `a == b`: the guard toggle had no effect.

`test_niche_c3_gap_paraxial_guard`, `c9`, `c10`, `c13` all PASSED — so
**`gap_kernel='exact'` is clean** and is not implicated in any of this.

### KEEP the polynomial-parallelism fix
Independent of the default, and still valid: the Newton pool previously only
knew how to rebuild a spline, so `n_workers` was a silent no-op on the
polynomial default (measured 0.98–0.99×). The worker now rebuilds either fit,
bit-identically to serial, giving 1.10–1.13× on the path that is actually the
default. The two-tier pool threshold (§3 of the original) feeds this and stands.

---

## 2. CORRECTION — test status in `6dfc79d` was overstated

The commit message says the full-suite regression was incomplete and that its
one observed failure was a pre-existing missing `filelock` dependency. The
`filelock` part is true and still true. **The rest was reported from a run that
had only reached ~34%** and was read as if complete. The completed run shows a
cluster of failures at 44–46% of collection order — the `test_niche_c*` block
above.

Corrected status, with the revert applied:

| suite | result |
|---|---|
| `c6_fit_guard` + `c11_decentred_fit_arbiter` + `c12_physics_fit_selection` + `newton_pool_both_fits` | **60 passed** |
| `d2_chain_multi` | 38 |
| `d6_exact_tilted_leg` | 38 |
| `exact_gap_kernel` | 23 (at the time of writing; another agent has since added ~243 lines) |
| `tight_focus_readout` | 10 |
| nine focus-readout-dependent suites | 469 |

**A complete, uninterrupted full-suite run has still never been observed.** It
is the single most important outstanding task, and the reason this addendum
exists: every premature status claim in this campaign came from reading a
partial run.

---

## 3. Repository state as of writing

* `6dfc79d` — the campaign commit. Contains the **bad spline default** and the
  overstated commit message. Local, and at the time of writing not pushed.
* `91ed9b0` — `docs(review): adversarial review of the traced-exact campaign`,
  authored by a different agent (Claude Fable 5), 597 lines.
* **Uncommitted in the working tree**, and NOT all from the same author:
  * `lumenairy/elements/_lens_traced.py` — the `newton_fit` revert described
    above (mine).
  * `lumenairy/propagators/carrier.py` (+384/-87),
    `tests/unit/test_niche_exact_gap_kernel.py` (+243),
    `validation/repro_traced_carrier_121/_d121_common.py` — in-flight work from
    the review, NOT mine. Do not attribute these to the original handoff.
  * `lumenairy/elements/pmm/stack.py`, `tests/unit/test_v5_13_0_pmm_tapered.py`
    — unrelated PMM work, deliberately excluded from `6dfc79d`.

Because more than one agent is editing this tree, **check authorship before
amending `6dfc79d`.** The revert in §1 is a one-line resolution change plus its
comment block; it can land as a separate commit without touching anyone else's
work.

---

## 4. What in the original handoff is UNCHANGED and still correct

* `gap_kernel='auto'` -> **exact on every backend** (JAX validated at 3–5e-16).
* `_FOCUS_STANDOFF_ZR = 0.8` and the fixture repairs (`d2`, `d6` green).
* The two-tier Newton pool threshold and its cold/warm measurements.
* `final_leg` — library default `'auto'` was already correct; the harness
  override and the `_chainA_*.npz` cache-key fix stand.
* All of §2 (the Sziklas–Siegman frame remains the one structural paraxial
  element), §4 (memory: ~8·N², the exact leg dominates, quadratic in
  `window_factor`), and §5 (open items).
* **`n_fine_cap` still must NOT be lowered to 8192 as a default** — D6's
  paraxial pre-check refuses it. Opt-in only.

---

## 5. Lesson worth carrying

Three times in this campaign a *plausible* result turned out to be an absent
one: a "converged" FWHM identical to five decimals across two grids that was
measuring an empty window; a warm-pool sweep that would have justified lowering
a threshold and making one-shot runs 1.6× slower; and a partial regression read
as a pass. In each case the number looked right and the measurement was not
being made. Prefer an explicit liveness check (`in_window`, a cold/warm split, a
run that prints its own summary line) over a value that merely looks reasonable.

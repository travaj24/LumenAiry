# The traced-lens layer map

**Living document.**  Started 2026-08-03 (niche C14), from
`ARCH_TRACED_ENCAPSULATION_2026_08_03.md` S8 step 1.  Subjects
`lumenairy/elements/_lens_traced.py` and `lumenairy/propagators/carrier.py`.

**What it is for.**  Fifteen labelled corrections landed across six weeks, each
with its own module-level switch, its own fail-before contract and its own
era-pinned tests.  Nothing was wrong with any of them individually; what was
missing was a single place that says *which switch belongs to which layer, what
turning it off restores, which sites must change together, and which of the
claims in the prose are still true*.  Without that, the only complete map is a
96-edge cross-reference graph maintained by hand — and it had already frayed in
both directions (a `:data:` reference to a symbol that does not exist, and a
roadmap entry calling a closed item open).

**It is machine-checked.**  Every identifier in S2's table is verified to exist,
and to agree with the runtime registry, by
`tests/unit/test_niche_c14_encapsulation.py`.  A renamed or deleted constant
fails CI here.  The registry itself is
[`lumenairy/elements/_traced_flags.py`](../../lumenairy/elements/_traced_flags.py).

---

## S1. How to use this document

* **Changing a flag's default?**  Update S2's table AND `_TRACED_ERA_FLAGS`'s
  newest era, or `test_the_newest_era_reproduces_the_live_shipped_values`
  fails.  That test exists because the campaign lost a nine-minute measurement
  to exactly this (S7 trap 1).
* **Changing the entrance eikonal?**  Read S4 UNIT A first: there are **three**
  sites and they must move together.
* **Changing anything about the exit support?**  Read S4 UNIT C: there is now
  **one** object, and the three views on it are deliberately not the same rule.
* **Writing a new probe or runner?**  Read S6 (the byte-identity devices) and
  S7 (the three harness traps).  All three traps are measured, not theoretical.
* **Reading a "still open" claim anywhere in `docs/audits/`?**  Check S8 first;
  several have been closed without the original document being updated.

---

## S2. The switch table

Shipped values and fail-before values are as of 2026-08-03 on
`feat/d121-final-closure`.  **Line numbers are deliberately absent**: they move
every campaign and a stale number is worse than none.  Grep the identifier.

`FAIL-BEFORE` is the value that restores the previous library behaviour.  `--`
means the fail-before is *not a value of this constant* (it is a kwarg, or the
absence of a carrier); those are called out in S2.2.

| # | layer | identifier | module | shipped | fail-before | era it shipped |
|---|---|---|---|---|---|---|
| 1 | F3 | `_F3_GUARD_TILTAWARE_EXPLICIT_CARRIER` | `_lens_traced` | `True` | `False` | 5.28.0 |
| 2 | R7/F2 | `_CARRIER_FIT_RADIUS_FRAC` | `_lens_traced` | `0.5` | -- | 5.28.0 |
| 3 | R7/F2 | `_CARRIER_FIT_MIN_SAMPLES` | `_lens_traced` | `64` | -- | 5.28.0 |
| 4 | P2 | `_FIT_RADIUS_BEAM_FACTOR_DEFAULT` | `_lens_traced` | `2.0` | -- | 5.29.0 |
| 5 | P2 | `_APERTURE_BEAM_WARN_RATIO` | `_lens_traced` | `1.5` | -- | 5.29.0 |
| 6 | D1 | `_FIT_DISC_OUTSIDE_WEIGHT_REL` | `_lens_traced` | `1e-8` | `0.0` | 5.32.0 |
| 7 | D7 | `_DECENTRED_FIT_POLY_ORDER` | `_lens_traced` | `10` | -- | 5.32.0 |
| 8 | C1 | `_DECENTRE_GATE_PIXELS` | `_lens_traced` | `0.5` | `0.0` | 5.32.0 |
| 9 | C1 | `_DECENTRE_GATE_W_FRAC` | `_lens_traced` | `0.05` | `0.0` | 5.32.0 |
| 10 | C5 | `TILTED_CARRIER_EXACT_EIKONAL` | `_lens_traced` | `True` | `False` | 5.32.0 |
| 11 | C6 | `REMAP_STATIONARY_PHASE_LAUNCH` | `_lens_traced` | `True` | `False` | 5.32.0 |
| 12 | C6-guard | `REMAP_STATIONARY_PHASE_FIT_GUARD` | `_lens_traced` | `False` | (opt-in) | 5.32.0 |
| 13 | C6 | `_REMAP_RESID_FREEZE_MARGIN` | `_lens_traced` | `1.25` | -- | 5.32.0 |
| 14 | C7 | `RAY_DENSITY_HALO_CHECK` | `_lens_traced` | `'warn'` | `'silent'` | 5.32.0 |
| 15 | C7 | `_RD_HALO_AMP_CONTOUR` | `_lens_traced` | `9.0` | -- | 5.32.0 |
| 16 | C7 | `_RD_HALO_RADIUS_FACTOR` | `_lens_traced` | `1.25` | -- | 5.32.0 |
| 17 | C7 | `_RD_HALO_AMAX_TOL` | `_lens_traced` | `1.0e-03` | -- | 5.32.0 |
| 18 | C8 | `REMAP_INVERSE_SUPPORT_BOUND` | `_lens_traced` | `True` | `False` | 5.32.0 |
| 19 | C8 | `_SUPPORT_BOUND_FEATHER_CELLS` | `_lens_traced` | `1.0` | -- | 5.32.0 |
| 20 | C9 | `SPHERE_PARAB_CONVERSION_EXACT` | `carrier` | `True` | `False` | 5.32.1 |
| 21 | C10 | `_REMAP_RESID_EIKONAL_DEGREE` | `_lens_traced` | `6` | `4` | 5.32.1 |
| 22 | C10 | `_REMAP_RESID_DEGREE_CAP` | `_lens_traced` | `6` | -- | 5.32.0 |
| 23 | C11 | `DECENTRED_FIT_ARBITER` | `_lens_traced` | `True` | `False` | 5.32.1 |
| 24 | C12 | `DECENTRED_FIT_PREDICTOR` | `_lens_traced` | `False` | (opt-in) | 5.32.1 |
| 25 | C12 | `_DECENTRED_FIT_SCORE_FLOOR` | `_lens_traced` | `0.0` | (inert) | 5.32.1 |
| 26 | C12 | `_DECENTRED_FIT_SPECTRUM_ORDER` | `_lens_traced` | `14` | `0` | 5.32.1 |
| 27 | C13 | `LSTSQ_CONDITIONING_STEPDOWN` | `_lens_traced` | `True` | `False` | 5.32.1 |
| 28 | C13 | `_LSTSQ_GRAM_RCOND_MIN` | `_lens_traced` | `1e-8` | -- | 5.32.1 |
| 29 | C13 | `_LSTSQ_RESID_MARGIN` | `_lens_traced` | `1e-6` | -- | 5.32.1 |
| 30 | C14 | `SUPPORT_BAND_CHECK` | `_lens_traced` | `'warn'` | `'silent'` | 5.32.1 |
| 31 | C14 | `_SUPPORT_BAND_PEAK_RATIO_TOL` | `_lens_traced` | `1.0` | -- | 5.32.1 |

**`5.32.1` is a SOURCE-ONLY era.**  `pyproject.toml` and `lumenairy/__init__.py`
both still read `5.32.0`; `CHANGELOG.md` has no `5.32.1` header (C9-C12 sit under
`[Unreleased]`) and C13/C14 are not in the CHANGELOG at all.  The name is used
because the source docstrings already use it.

### S2.1 The era presets

`_TRACED_ERA_FLAGS` expresses three points on the sequence — `v5.31` (before the
D1-D7 / C1-C8 campaign), `v5.32` (v5.32.0 as released), `v5.32.1` (this tree).
**They are presets, not a replacement for the flags**, and the reason is in
`ARCH_TRACED_ENCAPSULATION` S5.2: the flags are a *lattice*, not a timeline.
The most-cited comparison in the campaign — `REMAP_STATIONARY_PHASE_LAUNCH=True`
with `REMAP_INVERSE_SUPPORT_BOUND=False`, on which niche C8's whole case rests —
exists at no point in history.  So:

```python
from lumenairy.elements._traced_flags import traced_era, traced_flags

with traced_era('v5.32.1', REMAP_INVERSE_SUPPORT_BOUND=False):
    ...                 # C6 on, C8 off — a corner no era name can reach
```

Note that rows 1-5 shipped at 5.28.0/5.29.0, i.e. *before* the oldest era in
the table; at every era listed they are already at their shipped values.

### S2.2 The fail-befores that are not values

| layer | what restores the previous behaviour |
|---|---|
| R7/F2 (`_CARRIER_FIT_RADIUS_FRAC`) | the gate is `carrier is not None`; with no carrier the restriction is byte-identically absent |
| P2 (`_FIT_RADIUS_BEAM_FACTOR_DEFAULT`) | the kwarg `fit_radius_beam_factor=None` |
| D7 (`_DECENTRED_FIT_POLY_ORDER`) | the kwarg `decentred_fit_poly_order=<newton_poly_order>` |
| C8 (`_SUPPORT_BOUND_FEATHER_CELLS`) | `0.0` is a HARD CUT, not the pre-C8 library; row 18 is the switch |
| C11+C12 | **both** `DECENTRED_FIT_ARBITER` and `DECENTRED_FIT_PREDICTOR` `False` is the v5.32 gate, bit for bit |

### S2.3 Reporting-only switches

These change what the element **says** and never what it **returns**.  Rows 14-17
(C7) and rows 30-31 (C14).  Both are pinned field-neutral by test, in both
states — C14's by `test_the_fail_before_switch_restores_pre_c14_reporting`, on
a fixture where the check actually fires.

`carrier.py` also carries a disjoint per-call guard family from the C2/C3 era —
`on_gap_paraxial`, `on_decentred_fit`, `on_na_proximity`, `on_tilt_exact_grid`,
`on_chain_entry_congruence` — routed through one validator
(`_check_guard_action`) and one dispatcher (`_guard_dispose`).  **That is
already the consolidated form**, it is bitwise-neutral, and it is deliberately
NOT in the registry: these are per-call kwargs, not process-global switches.

---

## S3. The dependency graph

```
                       carrier resolution (S5.1 / N5 / R7 / F3)
                                    |
                        +-----------+-----------+
                        |                       |
              C5  exact tilted eikonal          |
                        |                       |
                        v                       v
              C6  stationary-phase launch <--- gate: preserve_input_phase=='remap'
                        |                            AND _r7_carrier_path
                 C10 residual degree 4->6
                        |
                        |  needs freeze radius to clear the fit disc
                        v
   R7/F2 --> P2 --> D1 --> D7 --> C1 --> C11 --> C12 --> ray-fit domain --> C6-guard
                        |                                       |
                        |                                       v
                        |                              C13 lstsq conditioning
                        v                                       |
              forward-map fit  ------> Newton inverse ------> C8 support bound
                                                                |
                                                                v
                                              C7 halo check + C14 band check
```

Eight edges a refactor must preserve, each stated by the audits:

1. **C5 -> C6 (magnitude, not existence).**  C5 changed the reference wavefront
   and so grew the input residual's own slope: `grad a` rms 1.46 -> 2.30 mrad,
   exit WFE 0.036 -> 0.089 waves, ratio 2.48 against the predicted
   `(2.30/1.46)^2 = 2.48`.  C6 exists to cancel exactly that quadratic term.
2. **C6 -> D1 (precondition destruction).**  D1's concentric hard mask is only
   safe because "the unconstrained directions of the fit inherit the map's
   RADIAL SYMMETRY".  C6 augments every launch by `grad(a_fit)` of a general
   non-radial polynomial and therefore destroys that precondition on the one
   branch D1 left alone.  **The most important cross-layer edge in the
   campaign.**
3. **C6 -> C6-guard -> (superseded by) C8.**  The guard was the first response
   to (2); C8 is the structural one.  C8 matches the guard's conservation
   result to five decimals while costing no EE, regresses none of the six
   synthetic fixtures the guard regressed two of, and fixes `(-2,0)`/`(-3,0)`
   which the guard *structurally cannot*.
4. **C6 -> C10 (unblocked by C8).**  With C8 off at degree 6 the chain
   manufactures 5.2 % of the input power; with C8 on, degree 6's conservation
   and halo are within noise of degree 4's.
5. **C8 is NOT downstream of C6.**  At `rs=2` on `(-4,-2)` it is **C6-OFF**
   that violates the halo criterion (28x/78x over), and C8 repairs that row
   too.  "The mechanism was never C6's; C6 only made it large enough to see."
6. **C8 -> C7 (monotone).**  The halo check can only go quieter under the
   bound.  No new firing anywhere.  **C14 exists because that monotonicity has
   a cost** — see S4 UNIT C.
7. **C6 freeze -> ray-fit disc (an ordering constraint across ~480 lines).**
   `_REMAP_RESID_FREEZE_MARGIN = 1.25` exists because the residual model's
   radial freeze circle must sit strictly *outside* the ray-fit disc; with the
   two coincident the polynomial and spline backends stop describing the same
   map (skirt error 5.608 um vs 0.006 um — a 130x step across the ray-fit disc
   radius and nowhere else).
8. **C9 is disjoint.**  It lives in `carrier.py`, changes a conversion the
   element consumes, and reaches the element only through the input field.

---

## S4. The three units — which sites must agree

### UNIT A — the entrance-eikonal jet (C5 + C6 + C10)

> Wherever the element uses the entrance congruence it must use **the same**
> total eikonal `Phi = W + a_fit` — the launch direction is `grad Phi`, the H6
> term added to the traced OPL is `Phi(x_in)`, and the residual de-chirp
> removes `Phi` from the input phasor so that what is transported pointwise is
> `exp(i k0 (a - a_fit))`.

**Three sites, and they are still three.**  Grep `_resid_eik`:

| site | what it needs | current form |
|---|---|---|
| ray launch | `grad Phi` | `_carrier_grad(...)` then `+= _resid_eik.grad(...)` under `if _resid_eik is not None` |
| H6 entrance eikonal | `Phi(x_in)` | `+= _carrier_W_fn(...)` then `+= _resid_eik.value(...)` under a second `if` |
| residual de-chirp | `Phi` on the wave grid | `* exp(-1j k _pip_remap_W)` then `*= exp(-1j k _resid_eik.value(...))` under a third `if` |
| (consumer) fit-guard predicate | "does a residual term exist" | `_resid_eik is not None and REMAP_STATIONARY_PHASE_FIT_GUARD` |
| (consumer) launch diagnostic | `.diag`, mutated post-construction | `_remap_launch_out.update(_resid_eik.diag)` |

The source already calls these "the three halves of one substitution
`W -> W + a_fit`".  **This unit is NOT extracted.**  It is the next one to do
(ARCH S8 step 3 order: C, then A, then B), and the hazard is live: the same
class of defect fired one module over in niche C3, where the chief-ray closure
existed in two places and the shipped mitigation was a runtime cross-check
that *raises on mismatch* — an admission in code that duplication had made
correctness un-guaranteeable by inspection.

### UNIT B — the ray-fit domain policy (R7/F2 + P2 + D1 + D7 + C1 + C11 + C12 + C6-guard)

Resolved in one block (`on_aperture_beam` validation, `beam_centre`, the C1
two-stage gate, the beam radius, `_beam_fit_radius`, the P2 warn, `_fit_r_geom`,
`_fit_r_max`, `_fit_r_about_beam`) and applied **~480 lines later**.  The split
is *forced*, by edge 7 above, and the source documents it at both ends.

Ten locals must survive that gap: `_frbf`, `_dec_order`, `_bcx`, `_bcy`,
`_beam_decentred`, `_w_in_beam`, `_beam_fit_radius`, `_fit_r_geom`,
`_fit_r_max`, `_fit_r_about_beam`.  All are pure functions of the inputs and
none is touched in between.  **Not extracted.**

### UNIT C — the traced exit support (C7 + C8 + the direct-fit hull) — **EXTRACTED (C14)**

Three notions of "the region the traced rays reached", computed from the same
arrays at nearly the same point by three rules:

| # | notion | rule | consumer |
|---|---|---|---|
| 1 | halo radius | amplitude-weighted centroid + max radius over samples above the `e^-9` contour, x1.25 at report time | C7 |
| 2 | support hull | convex hull of alive **stop-passing** landings, + `sqrt(2) sub dx` plateau, + 1 exit-lattice-cell feather | C8, C14 |
| 3 | direct-fit hull | the `inversion_method='fit'` path's own hull over the POST-restriction samples | direct fit |

They now hang off one `_TracedExitSupport`, sharing one alive mask, one hull
builder (`half_planes`) and one signed-distance rule (`signed_distance`).

**The rules are deliberately NOT merged.**  C7's radius is amplitude-weighted on
purpose (a *reporting* radius calibrated over 180 element calls, with a measured
123x separation between clean and defective populations at factor 1.25); C8's is
a convex hull of stop-passing rays on purpose (convexity "can only make the
bound LOOSER, never tighter, so it cannot manufacture a cut"); the direct-fit
hull is over a different point set on purpose.  Merging them would re-open a
calibration that cost 177 readings.  **One object, three named views, one set of
conventions** — that is the whole of it.

**What having one object bought.**  The blind spot named in
`RECON_PINS_POST_C8_2026_08_01` S7 item 1 became statable.  C8 retains a band
outside the traced hull *deliberately* (the plateau, which makes the upsample's
bleed identically zero); C7 reports only beyond `1.25 x r_hull`, which under the
bound is territory C8 has already zeroed.  So on the E-M6 fixture 0.19998 of
`P_ap` of manufactured light — carrying the field's **global maximum** — sat
where neither check looks, and the energy check read 1.01931, inside its band.
`SUPPORT_BAND_CHECK` now asks the one question that needs no new calibration:
*does this field peak somewhere no traced ray of this call reached?*  Measured
on that fixture: **2.14x**.  Silent on clean calls, silent on the C8 ghost
fixture, silent under its fail-before, and byte-neutral in every state.

### What does NOT unify

* **C9** is the *removal* of an approximation, not a feature; its natural home
  is the existing `carrier_reference` convention.
* **The `on_*` guard family** is already the consolidated form (S2.3).
* **The five beam-radius implementations** (`_input_beam_amp_radius`,
  `carrier._envelope_amp_radius`, `_axis_amp_radius`, `_gap_amp_radius`,
  `_chain_envelope_stats`) each have a stated, load-bearing reason — the
  separable one exists because the full meshgrid would be 6.6 GB at N=28672 and
  it runs on *every* inter-group leg.  **But they carry three different
  centring policies**, and that is true by circumstance rather than by
  construction.  Worth a docstring line each; not worth a merge.

---

## S5. The seam table

`apply_real_lens_traced`'s phases, for anyone deciding where to cut.  **There is
no clean cut in the middle third**: the live-variable count starts at 43 (the
parameters), climbs to 68, and never drops below 50 until two-thirds of the way
through.  Splitting into sequential sub-functions requires either a ~50-field
context object or sub-functions with 30-50 parameters each.  **The decomposition
that works is object extraction, not phase splitting.**

| phase | contents | unit |
|---|---|---|
| 1 | kwarg/enum validation, opt-in dispatch decisions | |
| 2 | multibranch (K1) early dispatch | |
| 3 | stop handling, row-banding, spline order | |
| 4 | carrier resolution | A |
| 5-6 | residual-phasor closures, reference input | A |
| 7 | Step 1: amplitude leg (double `apply_real_lens`) | |
| 8 | launch geometry + fit-domain RESOLVE + C6 residual fit | **A + B** |
| 9-10 | subsampling guardrail, launch grid, ray launch + trace | A (site 1) |
| 11 | exit-vertex correction + H6 + C6 eikonal | A (site 2) |
| 12 | exit-NA Nyquist guard, reshape, axis reference | |
| 13 | **`_TracedExitSupport.from_landings`** | **C** |
| 14 | fit-domain restriction APPLY | B |
| 15 | direct-fit inverse path (third hull) | C |
| 16-18 | forward-map fits, Newton machinery | |
| 19 | `_support_taper` -> `_exit_support.taper` | **C** |
| 20-22 | `_ray_density_amp_grid`, coarse Newton, upsample | |
| 23-24 | combine, mask, energy check, **C7 halo + C14 band check**, return | **C** |

---

## S6. The byte-identity device catalogue

Three devices, and **which one you need is decided by how the changed code is
reached**, not by preference.

| device | when | example |
|---|---|---|
| **shadow module** — load a second copy of `_lens_traced.py` under a different name inside the live package | the element is called through ONE name | `probe_c6_byte_identity.py` (29 configs), `probe_c8_byte_identity.py` (26), `probe_c14_byte_identity.py` (36) |
| **`git archive` in a separate process** | the changed module is resolved under one name from several call sites | `fc_c9_byte_identity.py` (52 configs) — "the chain entry point, the element hand-off and half a dozen helpers all resolve it as `lumenairy.propagators.carrier`, so a shadow copy would be reached by some call sites and not others" |
| **in-process patch** | the change is one value read at one site | C10's `rc_failbefore_121.py` |

**THE BOTH-SIDES RULE, and it is not optional.**  A probe that pins a flag on
the LIVE side only goes stale the moment the default moves.
`probe_c6_byte_identity.py` now prints `array_equal=False` on **17 of its 29
arms** for exactly that reason; `probe_c6_tilted_failbefore.py` did not go stale
because it sets the flag on the **shadow module as well as the live one**.
Write every new probe the second way.

**When there is no commit to point at.**  `probe_c14_byte_identity.py` compares
against the working tree as it stood immediately before the edit, captured
verbatim as `_c14_pre_baseline_lens_traced.py` and shipped beside it.  Use this
when the branch carries verified but uncommitted work, where HEAD is *not* the
thing the change must reproduce.

**Warm-up is load-bearing.**  Both members of every byte-identity pair must sit
on the same side of the traced pipeline's first-call ulp boundary (the W9
determinism calibration).  Every probe here runs both implementations twice
before comparing.

---

## S7. Standing rules for any new runner

Three measured harness failures, all the same shape: **an intervention
expressed relative to a default, evaluated after the default moved.**

1. **Pin every arm's own value; never inherit the default.**  `TAPER='on'`
   stopped meaning "the taper" the moment the library default flipped:
   `fc_sampling_121.py` returned byte-identical taper-on/off rows, and
   `fc_production_taper.py` ran a nine-minute "BASELINE (taper as shipped)" row
   that was in fact the *exact* conversion — its baseline read 89.235 where
   v5.32.0 reads 87.834.  Use `traced_flags(**overrides)` or a script-side
   `Patch`; print `traced_flag_state()` in the provenance banner.
2. **A harness that silently selects a library is worse than one that
   crashes.**  `approx_common.py` defaulted `LUMEN_PIN` to a frozen v5.31
   export that still existed on the machine; it was caught only by an
   `AttributeError` on a constant that does not exist in v5.31.  Every runner
   now forces `LUMEN_PIN=0` and prints file hashes.
3. **Cache keys must include the LIBRARY, not just the configuration.**
   `wfe_probe_orders.py` cached on the configuration alone and would have
   re-scored a pre-C9 chain as the post-C9 verdict.  Hit twice in one session.
   *"Knowing about a trap is not the same as being immune to it; only moving
   the file is."*

And three of reading, not writing:

4. **`pytest -q` prints one character per test** and names failures only in the
   end-of-run summary, so a multi-hour shard is opaque until it finishes.
   `p2diag_shardmap.py` maps progress characters back onto collected node IDs.
   **Do NOT attribute a mid-run `F` by counting characters against a separate
   `--collect-only` listing** -- the two orderings are easy to desynchronise
   and it has already produced one confident, wrong attribution (niche C14
   S6.1: an `FF` blamed on `test_niche_c6_stationary_phase_launch.py`'s oracle
   tests, which were never failing; the real pair was in `test_niche_c12_*`).
   Wait for the summary, or use the shardmap.
5. **A failure seen only under box contention is a hypothesis, not a finding.**
   Re-run it on a quiet box before spending anything on it: two
   `test_niche_c12_*` spectral tests failed in a leg launched alongside three
   other pytest runs, and pass in every composition without them -- including
   the identical 15-file single process (376 passed).
6. **Write long runs to a file with `python -u` and filter on read.**  `grep -v`
   without `--line-buffered` swallowed three long runs.

---

## S8. Known open

Carried forward so that the next reader does not have to rediscover them.
Items are removed from this list only when a measurement closes them.

1. **The `_DECENTRED_FIT_POLY_ORDER` order-10 anomaly.**  Orders 6, 8 and 12 all
   close the `(-1,0)` chain residual (-0.066 / -0.051 / -0.017 points) and **the
   shipped 10 does not** (+0.934); at `(-4,-2)` order 6 helps by 0.10 and order
   8 *hurts* by 0.08.  "A quantity that is good at 6, good at 8, bad at 10 and
   good again at 12 is not an approximation error converging in a degree —
   something discrete is happening at 10 on this geometry, and this study did
   not find out what."
2. **The C1 decentre gate sits 10-14x below its own measured crossover.**
   Forcing the branch puts the concentric/off-centre crossover between 0.48 w
   and 0.72 w on design 121, while `_DECENTRE_GATE_W_FRAC` switches at 0.05 w.
   *Partially addressed*: niche C11's `DECENTRED_FIT_ARBITER` now measures the
   branch instead of gating on the constant, but the constant remains the floor.
3. **`_REMAP_RESID_DEGREE_CAP` is vestigial.**  C10's raise consumed the entire
   headroom the cap existed to provide; it is now *equal* to the default, and
   its justification text was written when the default was 4.
4. **Two `cos^2` taper onsets disagree.**  `carrier._tilt_exactness_phase`'s
   docstring says 3.2 beam radii; `D121_RESIDUAL_CLOSURE` S7 item 8 says 2.5
   (group-6 exit at `(-4,0)`).  The reconciling per-call census
   (`rc_c5taper_121.py`) is written and unrun.  This is a one-run item.
5. **C14's band check inherits C7's declination.**  On a grid whose extent is
   comparable to its own exit fan — design 121's production readout leg — both
   checks decline, for the measured reason at `_RD_HALO_AMAX_TOL` SCOPE (d).
   Closing the blind spot *there* needs a hull that fits the grid, which is a
   different problem.
6. **C7's `e^-9` gate understates the true support** (1.6161 mm against
   1.8115 mm over all alive rays), and the two D6 call labels were swapped in
   the C7 record.
7. **Era-pinning is a race the guards will lose.**  "Every 'the guard is still
   needed' test is a race between the guard's value and the rest of the
   library, and it will lose eventually."  Two instances fired in one
   afternoon: D7's fold witness (off-beam amplitude ~0.35 -> 1.8e-04 at degree
   6, so it was re-pinned at degree 4) and D6's paraxial-FWHM discriminator
   (3.19x -> 1.857x -> 1.762x -> floored at 1.25x).  The option neither took:
   **find a fixture where the guard is still load-bearing on the current tree
   and move the witness there.**

### Closed, but recorded because a document still says otherwise

* **`_chain_chief_ray_at_target` WAS converted** to the exact chief-ray trace.
  `ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27`'s "Still open (2026-07-30)"
  says it is on the lumped paraxial ABCD; the CHANGELOG says the opposite; the
  **source agrees with the CHANGELOG** (`carrier.py`, `_group_chief_transfer`,
  with a runtime cross-check that RAISES on mismatch).
* **The dangling `CHAIN_EXACT_TILTED_REFERENCE` reference is fixed** (niche
  C14) and now points at `carrier._exact_tilt_reference`.  A CI check
  (`test_no_prose_cross_reference_dangles`) prevents the next one.
* **`DECENTRED_FIT_PREDICTOR`'s docstring is self-contradictory.**  One line
  still says "SHIPPED ON since 5.32.1" while another says the 5.32.1 flip "was
  REVERTED on evidence, 2026-08-03".  The code ships `False` and the C12 test
  pins `is False`.  The stale line is the first.
* **`C11_PHYSICAL_DECENTRE_GATE_2026_08_03` contradicts itself** on whether
  `DECENTRED_FIT_ARBITER` ships `False` (headline) or `True` (S-body).  It
  ships `True`; the flip is recorded only in the C13 audit.
* **`_REMAP_RESID_EIKONAL_DEGREE`'s fail-before line attributes its era to
  "niche-C9"**; the CHANGELOG attributes the degree-6 change to C10, and C9 is
  `SPHERE_PARAB_CONVERSION_EXACT`.

---

## S9. Test and probe index

| what | where |
|---|---|
| registry / manifest / cross-reference CI checks | `tests/unit/test_niche_c14_encapsulation.py` |
| UNIT C object, band check, fail-before | same file |
| dx self-check's four exits | same file |
| C14 byte-identity (36 configs) | `validation/repro_traced_carrier_121/probe_c14_byte_identity.py` |
| C8 byte-identity (26 configs) | `probe_c8_byte_identity.py` |
| C6 byte-identity (29 configs, 17 stale by design) | `probe_c6_byte_identity.py` |
| C9 byte-identity (52 configs, `git archive`) | `fc_c9_byte_identity.py` |
| production acceptance | `focus_scan_121.py` via `c13_with_stepdown.py` |
| conservation / halo 6-of-6 | `energy_stage_audit_121.py` via `c13_with_stepdown.py` |
| production warning dedup | `p2diag_prod_dedup.py` |

# Niche C14 -- encapsulation, and the guards that were passing silently

**Date** 2026-08-03.  **Tree** `feat/d121-final-closure`, uncommitted.
**Subjects** `lumenairy/elements/_lens_traced.py`,
`lumenairy/propagators/carrier.py`, and one new module
`lumenairy/elements/_traced_flags.py`.

**Predecessors, adopted as the design and the diagnosis rather than re-argued:**
`ARCH_TRACED_ENCAPSULATION_2026_08_03.md` (S8 steps 1-3),
`P2_DIDNOTWARN_DIAGNOSIS_2026_08_03.md` (S4.1, S4.2),
`RECON_PINS_POST_C8_2026_08_01.md` (S7 item 1).

---

## 0. Headline

Five pieces of work, in two workstreams, all additive and all bit-preserving.

| # | what | evidence |
|---|---|---|
| A1 | `docs/audits/TRACED_LAYER_MAP.md` + a CI-checkable identifier manifest | 3 tests bind doc <-> registry <-> library; the manifest **immediately caught the dangling `CHAIN_EXACT_TILTED_REFERENCE`** the architecture study had found by hand, and it is fixed |
| A2 | the flag registry, context manager and era presets (`_traced_flags.py`) | 31 switches, 3 eras, zero migration; `test_the_newest_era_reproduces_the_live_shipped_values` pins the table to the library |
| A3 | **UNIT C extracted** -- three notions of the traced exit support into one `_TracedExitSupport`, and the documented JOINT BLIND SPOT closed | **36 of 36 configs byte-identical, `max\|dE\| = 0.000e+00`, on BOTH builds**; the new band check fires at **2.14x** on the fixture the blindness was measured on and is silent everywhere else measured |
| B1 | the `NOT dx-STABLE` flag is no longer dedup-silenced after the first call per caller line | `p2diag_prod_dedup.py`: **1 of 2 -> 2 of 2**, on both builds, same text and same caller-line attribution |
| B2 | the dx self-check's silent-pass holes closed -- **four, not three** | a fourth was found while closing the three: a cross-branch comparison of two quantities that share a key name and mean different things |

**Nothing about what the library RETURNS changed.**  Production acceptance,
conservation and the halo report all reproduce their recorded values exactly
(S5), and the element's fields are byte-identical to the pre-C14 tree across
the whole reachable configuration matrix.

**One thing about what the library SAYS changed, deliberately, and it is behind
a fail-before switch**: `SUPPORT_BAND_CHECK`, which watches a band that
previously nothing watched.  `'silent'` restores the pre-C14 reporting exactly.

---

## 1. What was actually wrong (and what was not)

The architecture study's central finding was that the accretion across fifteen
layers is *evidence, not mechanism* -- 61 % of `_lens_traced.py` is commentary,
the executable content added per layer is measured in tens of lines, and "the
code is unmanageable" aims at the wrong 39 %.  That finding is accepted here
and it is why **no phase split was attempted**: the live-variable envelope never
drops below ~50 names in the middle third, so cutting `apply_real_lens_traced`
into sequential stages trades one large function for one large struct plus a
new class of ordering bug.

What *was* wrong, in the three places this work touches:

1. **The layer map existed only as a 96-edge prose graph maintained by hand,
   and it had frayed in both directions** -- a `:data:` reference to a symbol
   that does not exist anywhere in the repository, and a roadmap entry calling
   a closed item open.  Neither is a behaviour defect; both are the leading
   indicator of a map that no longer describes the territory.
2. **Three notions of "the region the traced rays reached" were computed from
   the same arrays, 40 lines apart, by three rules and three copies of the same
   convex-hull algebra** -- and the inconsistency had a *measured* consequence
   that an adversarial re-check found and neither self-check did.
3. **A guard whose entire contract is "a non-converged result is never returned
   as if it were converged" had four exits, and three of them returned
   silently** -- plus a delivery contract that silenced it after the first call
   in a batch loop.

---

## 2. Workstream A1-A2 -- the map and the registry

### 2.1 The layer map is checked, not merely written

`TRACED_LAYER_MAP.md` carries the switch table (31 rows), the eight-edge
dependency graph, the three-unit map, the seam table, the byte-identity device
catalogue, the three harness traps as standing rules, and a "known open"
section.  Three tests bind it to the code so it cannot rot silently:

| test | what it catches |
|---|---|
| `test_the_layer_map_table_names_only_identifiers_that_exist` | a renamed or deleted constant still named by the map |
| `test_the_layer_map_shipped_column_matches_the_library` | a default that moved without the table moving -- the documentation form of the campaign's own worst harness trap |
| `test_the_layer_map_and_the_registry_agree_on_the_switches` | two artefacts giving a reader two different answers, which is how the C11-vs-C13 arbiter-default contradiction happened |

Plus a fourth that is not about the map but about the same rot:

**`test_no_prose_cross_reference_dangles`** walks every
``:data:`~lumenairy.<module>.<NAME>``` in both files and resolves it.  On its
first run it failed, naming
`lumenairy.propagators.carrier.CHAIN_EXACT_TILTED_REFERENCE` -- exactly the
dangling reference `ARCH_TRACED_ENCAPSULATION` S2.3(a) had found by reading.
It now points at `carrier._exact_tilt_reference`, which is the real mechanism
(a helper that imports and returns the element's flag at call time).  Scope is
deliberately narrow -- only `:data:` forms, which name a module and a
module-level constant unambiguously -- so it can be exact rather than noisy.

The manifest also earned its keep immediately in the other direction: the
registry/map agreement test failed on the first run because C14's own two
constants were in the document and not in the registry.

### 2.2 The registry is presets, not an era switch

`lumenairy/elements/_traced_flags.py` ships `_TRACED_ERA_FLAGS` (31 entries,
each carrying its layer, its documented fail-before, its era values, and a note
quoting the constant's own contract), plus:

```python
traced_flag_state()            # every switch's current value, for a provenance
                               # banner or a cache key
traced_flags(**overrides)      # atomic save/restore by bare name
traced_era(era, **overrides)   # preset PLUS per-flag override
resolve_era(era)               # {(module, name): value}
```

**The era is a preset and the overrides are not optional sugar.**  The
architecture study's Argument 1 is that the flags are a *lattice*, not a
timeline, and the single most-cited configuration in the campaign --
`REMAP_STATIONARY_PHASE_LAUNCH=True` with `REMAP_INVERSE_SUPPORT_BOUND=False`,
which is how niche C8's entire case is made -- exists at no point in history.
`test_the_lattice_corner_the_c8_case_rests_on_is_reachable` pins that it stays
expressible.

**Zero migration.**  No existing flag, test, probe or validation runner was
touched.  The ~117 existing assignment sites keep working exactly as they are;
the registry is what NEW work should use, and what a runner should print.

Design points worth recording:

* **Unknown override names RAISE.**  A typo'd override that silently did
  nothing is precisely the failure class the module exists to close.
* **Flags are read at CALL time everywhere**, never captured as a class
  default -- the trap S7.2 of the architecture study names, whose failure mode
  is "a test that still passes because it was written against a helper that
  also captured the default".
* **The registry does not replace `tests/conftest.py`'s niche-C11 leak
  guard.**  That guard *discovers* ~91 flags so it cannot go stale; this
  registry *curates* the ones with a documented fail-before, with the values
  that contract names.  Discovery answers "what could leak"; the registry
  answers "what does turning this off restore".
* **`v5.32.1` is a source-only era.**  `pyproject.toml` and
  `lumenairy/__init__.py` read `5.32.0`, `CHANGELOG.md` has no `5.32.1` header,
  and C13/C14 are not in the CHANGELOG at all.  The module says so.

---

## 3. Workstream A3 -- UNIT C, and the blind spot

### 3.1 What was extracted, and what deliberately was not

Three notions, three rules, one object:

| notion | rule | consumer |
|---|---|---|
| halo radius | amplitude-weighted centroid + max radius over samples above the `e^-9` contour | C7 |
| support hull | convex hull of alive **stop-passing** landings + `sqrt(2) sub dx` plateau + 1 feather cell | C8, C14 |
| direct-fit hull | convex hull of the **post-restriction** samples | `inversion_method='fit'` |

`_TracedExitSupport` now owns: one alive mask (the two blocks each built their
own), one hull builder `half_planes()`, one signed-distance rule
`signed_distance()`, the C7 view (`centroid`, `radius`), the C8 view (`bound`,
`taper`), and the new C14 view (`retained_band_masks`).  `_support_taper`'s
closure body is gone; the direct-fit path's own `ConvexHull` call, its
`equations -> (A, b)` unpacking and its half-plane evaluator are gone.

**The rules were NOT merged, and that is the whole design.**  C7's radius is
amplitude-weighted on purpose (a *reporting* radius calibrated over 180 element
calls, with a measured 123x separation between clean and defective populations
at factor 1.25).  C8's is a convex hull of stop-passing rays on purpose
(convexity "can only make the bound LOOSER, never tighter, so it cannot
manufacture a cut").  The direct-fit hull is over a different point set on
purpose.  Merging them would re-open a calibration that cost 177 readings.
**One object, three named views, one set of conventions.**

Two contract details that had to be preserved explicitly, because sharing a
builder is where this kind of refactor silently changes behaviour:

* `half_planes(..., strict=False)` **declines** (returns `None`) on a
  degenerate support; `strict=True` lets the original qhull exception
  propagate.  The two consumers genuinely differ: the C8 bound is an optional
  containment, so a support it cannot measure means "do not bound"; the
  direct-fit hull IS the output domain, and a fit with no domain has always
  raised.
* The two view gates stayed separate and `want_halo` was **not** widened to
  cover the band check.  Widening it would make `_rd_hull_r` non-`None` under
  `RAY_DENSITY_HALO_CHECK = 'silent'` and C7 would warn from a policy that says
  it must not -- which is what `test_policy_silent_suppresses` exists to catch.

### 3.2 The blind spot, and what closes it

`RECON_PINS_POST_C8` S7 item 1, on the E-M6 fixture:

> the post-C8 field still carries **0.19998 of `P_ap` outside the exact-ray
> hull** (the `sqrt(2) sub dx` plateau plus one exit-lattice cell of feather,
> which C8 keeps deliberately), and **its global `|E|` maximum sits in that
> band**.  ... neither the energy check (1.01931 is inside the band) nor the C7
> halo check (the taper's outer edge, 1.4996 mm, is inside the halo radius
> `1.25 x r_hull`) reports it.

The geometry is the point.  C8 zeroes everything beyond its retained band.  C7
reports only beyond `1.25 x r_hull`, which under the bound is territory C8 has
*already* zeroed -- so **under C8 the halo check cannot fire at all**, and the
band C8 retains is watched by nobody.  Edge 6 of the dependency graph ("C8 -> C7
is monotone: the halo check can only go quieter under the bound") is true, and
this is its cost.

`SUPPORT_BAND_CHECK` measures exactly that band -- `0 < s <= plateau + feather`
in the hull's own signed distance -- and asks the one question that needs no new
calibration:

> **does this field peak somewhere no traced ray of this call reached?**

A skirt decays outward and cannot; a manufactured lobe does.  The criterion is
`max|E| in band > _SUPPORT_BAND_PEAK_RATIO_TOL * max|E| inside the support`,
with the tolerance at `1.0`.  It is scale-free and unit-free, and at 1.0 no
correct field can answer yes.

**Measured, both directions:**

| fixture | verdict | reading |
|---|---|---|
| E-M6 (`test_niche_audit_w3_elements`'s own, C8 ON) | **FIRES** | band max is **2.14x** the in-support max |
| E-M6 with `SUPPORT_BAND_CHECK='silent'` | silent | the fail-before |
| C8's `_GHOST` fixture, C8 ON and OFF, guard ON | silent | the lobe is at 3 w, but the peak is still at the focus |
| C8's `_CLEAN` fixture, C8 ON and OFF | silent | |
| a clean singlet at `rs=4` | silent | |
| design 121 production focus scan (six-order energy audit + focus scan) | **silent, 0 firings** | S5 |

The instrument therefore separates populations rather than firing everywhere,
and it is not a restatement of C7 (which is silent on E-M6 under the bound) or
of the energy check (which reads 1.01931, inside its band).

**Cost is bounded exactly.**  The band mask runs on the WAVE grid, where a
naive `O(pixels x facets)` reduction would put a ~30x BLAS pass on every
ray-density call for a diagnostic.  Two *strict* radial screens cut it to a thin
annulus: everything closer to an interior point than the hull's inradius is
inside, and everything beyond `hull_rmax + w` has `s > w`.
`test_the_band_mask_annulus_shortcut_is_exact` measures the verdict against the
brute-force evaluation rather than assuming it.

**It inherits C7's declination**, deliberately.  On a grid whose extent is
comparable to its own exit fan -- design 121's production readout leg -- the
band check declines for the same measured reason C7 does.  Closing the blind
spot *there* needs a hull that fits the grid, which is a different problem, and
it is recorded as open (S7).

### 3.3 A new warning is a new way to break someone else's test

The first draft of the band message explained itself with the clause *"neither
the energy self-check nor the HALO self-check can see this band"*.
`test_niche_audit_w3_elements` collects warnings with
`[t for t in texts if 'energy self-check' in t]` and then asserts that list is
**empty** on the bounded arm -- so a purely explanatory clause turned a green
pin red on a fixture where the new check is *supposed* to fire.

Fixed on the library side (the message was reworded; the test was not touched),
and pinned by
`test_the_band_message_does_not_collide_with_another_checks_filter`, which
asserts the message contains none of the phrases the suite filters warning text
on.  **This is a general hazard for any new diagnostic in this codebase** and it
is now in the layer map's standing rules.

Recorded as a latent instance of the same class: C7's own message contains
"the energy self-check CANNOT see this".  It has never collided because C7 does
not fire on the E-M6 fixture, but the collision is one fixture away.  Not
changed here -- C7's text is pinned by
`test_message_reports_radius_amplitude_and_power` and moving it is a separate
adjudication.

---

## 4. Workstream B -- the guards that were passing silently

### 4.1 The convergence flag was deduped in PRODUCTION

`_run_chain_dx_self_check` warned through `warnings.warn(..., stacklevel=3)`.
`stacklevel=3` attributes the warning to the CALLER of
`propagate_traced_carrier_chain` -- correct for blame, fatal for delivery:
under CPython's stock filters an unmatched `RuntimeWarning` takes the
`"default"` action, which is **once per (text, category, module, lineno)**, and
a batch loop calls the chain from ONE line.

Measured on the real chain, before and after, with the P2 study's own probe:

| | Windows / MKL / py3.14.6 | Linux / OpenBLAS / py3.12.3 |
|---|---|---|
| before (`warnings.warn`) | **1 of 2** | (P2 diagnosis: 1 of 2) |
| after (`_warn_undeduped`) | **2 of 2** | **2 of 2** |

Same message text, same caller-line attribution (`p2diag_prod_dedup.py:50`).

**The fix is `warn_explicit` with a throwaway registry, and the choice of
mechanism matters.**  `warn_explicit` consults the `registry` mapping it is
handed; passing `registry=None` makes it allocate a fresh dict, so the
`'default'` action's bookkeeping is written where nothing will read it.
Everything else in the warnings protocol is untouched -- which is the point:

* an `'ignore'` filter still returns early, so a caller who has silenced this
  category KEEPS it silenced (pinned by
  `test_an_ignore_filter_still_silences_it`).  **This is why the fix is not
  `catch_warnings() + simplefilter('always')`**, which would override the
  caller's own configuration and is process-global and not thread-safe;
* an `'error'` filter still raises;
* `'once'` still dedups, because it keys off the module-global `onceregistry`;
* only the `'default'` action changes, from once-per-caller-line to
  every-qualifying-call.

The frame resolution reproduces `warnings.warn`'s own, so the emitted location
is identical to what the plain `warn` printed (pinned by
`test_it_is_attributed_to_the_callers_line_not_the_library`).

Scope is deliberately narrow: the `on_*` guard family keeps `_guard_dispose`
and its ordinary `warn`.  Those fire on a per-CONFIGURATION fault (a bad
geometry the caller fixes once), where once-per-location is the right dose.
This one fires on a per-RESULT fault, where suppressing repeats silently drops
results.

### 4.2 Four silent-pass holes, closed

The P2 diagnosis named three.  Closing them surfaced a fourth.

**(a) A degenerate PRIMARY result read as dx-stable.**
`_chain_result_metrics` returns `{}` whenever the field's total intensity is
non-finite or `<= 0`; the guard `return`ed silently, and the refined chain was
never even run.  Now warns -- and a degenerate primary is a *stronger* fault
than the grid-convergence one the check was looking for.  It still returns
before the second chain, so the refusal is cheaper than the false pass
(`test_hole_a_declines_before_paying_for_the_second_chain` proves the second
run is never started, by poisoning the upsample).

**(b) A degenerate REFINED run read as dx-stable, after paying for both
chains.**  `m2` empty -> empty key intersection -> `bad` empty -> silent return.
Now warns, and says the diagnostic thing: the finer grid degenerated where the
coarse one did not, which is the opposite of convergence.

**(b') NEW -- a cross-branch comparison of two different quantities.**  While
closing (b) it became clear that intersecting the two metric dicts is unsafe
for a second reason: the two branches of `_chain_result_metrics` share the key
`'power'` and **mean different things by it** (envelope window power from
`_chain_envelope_stats` vs `sum|E|^2 dx^2` on a readout grid).  If a refinement
changes the chain's own routing, the old code would have compared those two
numbers as though they were the same quantity.  A new one-line predicate
`_chain_metric_kind()` makes the branch explicit and the guard now refuses a
cross-branch pair instead of trusting the overlap.

The predicate is deliberately a SEPARATE function rather than folded into
`_chain_result_metrics`'s return, so that function's signature and its
`%r`-logged dict stay exactly as they are: `c11_p2dx_recon.py` calls it
directly, and `test_niche_p2_guards.py` parses the logged dicts with a numeric
regex.  Neither should have to change to close a guard hole.

**(c) The readout-less mode is now REFUSED rather than run.**  Without a focus
readout the compared quantities -- `w_env`, `power`, `R` -- are dx-INVARIANT by
construction: measured 0.0867 %, 0.0015 % and 0 % on the same beyond-Nyquist
fixture that moves 52.5 % through a readout.  The mode was very nearly a no-op
that cost a second full chain to report "stable" whatever the truth was.  It
now declines up front, names the remedy (`focus_readout=dict(...)`), and skips
the second run -- so the refusal is *faster* than the false pass it replaces.

**The INFO line moved earlier**, so it is emitted on every path that runs both
chains including the two refusals.  That is not cosmetic: the P2 study's margin
instrument (`_margin_report`) already handles a `NO SHARED METRIC KEYS` case and
could never reach it, because the guard returned before logging.  It can now.
The line's format and its position as the LAST `self_check='dx'` record are
unchanged, because that instrument parses `lines[-1]`.

**Nothing new fires on a converged chain** --
`test_a_converged_chain_is_still_silent` is the regression guard, and the
existing `test_self_check_dx_passes_on_a_dx_stable_chain` is untouched and
green.  This work adds four ways to speak and zero ways to cry wolf.

---

## 5. Verification

### 5.1 Byte-identity -- 36 configs, both builds

`validation/repro_traced_carrier_121/probe_c14_byte_identity.py`, against the
working tree captured verbatim immediately before the edit
(`_c14_pre_baseline_lens_traced.py`, md5 `c8e1a870221565832545144bb1baeb5d`,
8827 lines), loaded as a shadow module inside the live package.

**The reference is a file and not `git show HEAD` on purpose**: this branch
carries a large body of uncommitted, already-verified C11/C12/C13 work, so HEAD
is not the thing the extraction must reproduce.  Every pinned flag is written to
BOTH modules from one table, which is the both-sides rule that kept
`probe_c6_tilted_failbefore.py` alive while `probe_c6_byte_identity.py` went
stale on 17 of its 29 arms.

| part | configs | what it covers | result |
|---|---:|---|---|
| (a) | 24 | every `preserve_input_phase` x `amplitude_model` at `rs` 1 and 4, plus lattice and no-carrier, with C8 ON **and** OFF | `array_equal=True`, `max\|dE\| 0.000e+00` |
| (b) | 8 | `inversion_method='fit'` -- **the third hull**, which `probe_c8_byte_identity.py` does not exercise at all | `array_equal=True`, `max\|dE\| 0.000e+00` |
| (c) | 4 | the band check is field-neutral: `warn` == `silent` == baseline | identical in all three |

**36 of 36 on Windows/MKL and 36 of 36 on Linux/OpenBLAS.**

### 5.2 Production acceptance -- exact

```
LUMEN_PIN=0 STEPDOWN=1 SELECTOR=arbiter python -u c13_with_stepdown.py focus_scan_121.py
```
(`SELECTOR=arbiter` is the SHIPPED pair: `DECENTRED_FIT_ARBITER=True`,
`DECENTRED_FIT_PREDICTOR=False` -- the post-flip default, after C13 reverted the
predictor on evidence.)

```
AT-PLANE: FWHM=3.350um EE3=90.3% EE6=99.7% EE12=99.8% off=(+0.00,+0.00)um
BEST-FOCUS[peak] dz=+0um: FWHM=3.350um EE3=90.3% EE6=99.7% EE12=99.8% pk=5.529e+03
BEST-FOCUS dz=+5um:      FWHM=3.450um EE3=89.6% EE6=99.7% EE12=99.8%
```

**3.350 / 90.3 / 99.7 / 99.8 to every printed digit, peak 5.529e+03**, and the
`dz=+5um` row matches C13 S10.4's `3.450 / 89.6 / 99.7 / 99.8` as well.  The
best-focus plane does not move.

### 5.3 Conservation and halo -- 6 of 6, all six orders

```
LUMEN_PIN=0 STEPDOWN=1 SELECTOR=arbiter ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' \
    CONFIGS='ship' NULL=1 python -u c13_with_stepdown.py energy_stage_audit_121.py
```

| order | `P_out/P_in` | `g4` | `amax4` | `r_rms` (mm) | NULL |
|---|---|---|---|---|---|
| (0,0) | 0.994315 | 0.000e+00 | 0.000e+00 | 0.8383 | `0.000e+00` |
| (-1,0) | 0.994063 | 1.962e-11 | 1.716e-05 | 0.8384 | `0.000e+00` |
| (-2,0) | 0.994132 | 6.783e-11 | 3.213e-05 | 0.8382 | `0.000e+00` |
| (-3,0) | 0.994071 | 1.302e-09 | 5.625e-05 | 0.8380 | `0.000e+00` |
| (-4,0) | 0.994004 | 8.841e-09 | 1.075e-04 | 0.8376 | `0.000e+00` |
| (-4,-2) | 0.993826 | 9.114e-09 | 1.116e-04 | 0.8375 | `0.000e+00` |

Against the six criteria of `ENERGY_CONSERVATION_AUDIT_2026_07_31` S6:

* **C1a** per-element `P_out/P_ap` in `[0.9900, 1.00020]` -- worst group 0.995932;
* **C1b** deficit floor -- unchanged from the reference (no field bit moved);
* **C2** end-to-end in `[0.9850, 1.00050]` -- worst 0.993826;
* **C3** `g4 <= 3.0e-07` -- worst 9.114e-09, three orders of magnitude clear;
* **C4** `amax4 <= 1.0e-03` -- worst 1.116e-04;
* **C5** `r_rms` within 3 % -- 0.8375-0.8384 mm across all six;
* **C6** reporting -- **the NULL INTERVENTION is bit-exact on all six stages of
  all six orders** (`array_equal=True`, `max|dE| = 0.000e+00`), and the library
  file hashes are in the log header.

**Every `P_out/P_in` digit matches C13 S10.4's shipped-selector table exactly.**

### 5.4 The self-checks are silent on production

`grep -c` over both production logs:

| | focus scan | energy audit |
|---|---:|---:|
| `HALO self-check FAILED` (C7) | **0** | **0** |
| `SUPPORT-BAND self-check FAILED` (C14) | **0** | **0** |
| `energy self-check FAILED` | **0** | **0** |

The new check adds no noise to the production configuration.

### 5.5 Suites and lint

| leg | Windows / MKL (py3.14.6, numpy 2.4.4, scipy 1.17.1) | Linux / OpenBLAS (py3.12.3, numpy 2.4.6, scipy 1.17.1, OpenBLAS 0.3.31) |
|---|---|---|
| `test_niche_c14_encapsulation.py` (new, 32 tests) | **32 passed** | **32 passed** |
| c7 + c8 + w3 + except-budget + p2-guards + s12 + upsample + c14 | **142 passed** | **142 passed** (9:33) |
| the 15-file niche family, one process | **376 passed** (30:35) | -- |
| the `c1,c3,c5,c6` prefix under `LUMEN_TEST_FLAG_LEAK_STRICT=1` | **119 passed, no leak raised** (6:40) | -- |
| `probe_c14_byte_identity.py` | **36/36 identical** | **36/36 identical** |
| `p2diag_prod_dedup.py` | **1 of 2 -> 2 of 2** | **2 of 2** |
| `ruff check lumenairy/ tests/unit/` | **All checks passed** | **All checks passed** |

Zero failures on every leg.  Details, and a contention artefact chased to
ground, in S6.

---

## 6. The three verification legs, and a contention artefact worth recording

| leg | scope | result |
|---|---|---|
| **A** | the 15-file niche family in ONE process: `test_niche_{c1,c3,c5,c6,c9,c10,c11,c12,c13,d1,d3,d5,d6,d7,s8}_*.py` | **376 passed, 0 failed** (30:35) |
| **B** | leg A's `c1,c3,c5,c6` prefix under `LUMEN_TEST_FLAG_LEAK_STRICT=1` | **119 passed, 0 failed, no leak raised** (6:40) |
| **C** | the directly-affected selection on **Linux/OpenBLAS**: `c7, c8, w3, except-budget, p2-guards, s12, upsample, c14` | **142 passed, 0 failed** (9:33) |

Leg B is the one worth naming: with the niche-C11 leak guard promoted to
FAIL-the-leaking-test, the four files that carry the heaviest flag toggling in
the campaign complete with **no flag left dirty across any module boundary**.
The registry's save/restore participates in that run (the C14 tests use it),
so `traced_flags` is exercised under strict leak detection rather than merely
asserted in isolation.

### 6.1 The contention artefact, recorded because it will happen again

A FIRST run of leg A reported `2 failed, 373 passed` --
`test_niche_c12_physics_fit_selection.py::test_the_spectral_tail_carries_the_whole_candidate_residual`
and `::test_the_shell_spectrum_of_a_rotationally_symmetric_lens_is_even`.  That
run was launched while a WSL pytest leg, a leak-strict leg and two ad-hoc runs
were on the same box.

Chased to ground rather than retried blindly:

| experiment | result |
|---|---|
| the c12 file alone | **20 passed** |
| `c11 + c12` | **41 passed** |
| the whole c-block `c1,c3,c5,c6,c9,c10,c11,c12,c13` (202 tests, one process) | **202 passed** |
| the full 15-file leg A, **quiet box**, one process | **376 passed** |

So it is **not** order dependence and **not** this work: the same process
composition that failed under contention is green without it.  Recorded as a
CONTENTION artefact -- these two tests compare spectral quantities produced by
a threaded least-squares path, and the box was oversubscribed.

**The general lesson, and it cost an hour here:** `pytest -q` names failures
only in the end-of-run summary, so a mid-run `F` in the progress stream can
only be located by counting characters against a separate `--collect-only`
listing -- and that mapping is easy to get wrong (it was, here: the `FF` was
mis-attributed to `test_niche_c6_stationary_phase_launch.py`'s two oracle
tests, which were never failing).  **Do not attribute a mid-run `F` by
position; wait for the summary, or use `p2diag_shardmap.py`.**  This is now in
the layer map's standing rules.

### 6.2 Why C14 is excluded as a cause of any of it, by measurement

1. The element is **byte-identical across 36 configurations on both builds**
   (S5.1), covering every reachable `preserve_input_phase` x
   `amplitude_model` x `inversion_method` combination.
2. C14's only new *observable behaviour* is the `SUPPORT_BAND_CHECK`
   evaluation, proved field-neutral in the same probe (part (c)).
3. The `carrier.py` edits are confined to `_run_chain_dx_self_check`, which
   runs only under `self_check='dx'` -- a mode none of the leg-A files uses.

---

## 6. Files

```text
lumenairy/elements/_traced_flags.py                    NEW -- the registry, ctx manager, era presets
lumenairy/elements/_lens_traced.py                     _TracedExitSupport; SUPPORT_BAND_CHECK;
                                                       the two inline support blocks -> one construction;
                                                       _support_taper and the direct-fit hull rewired;
                                                       the dangling :data: reference fixed
lumenairy/propagators/carrier.py                       _warn_undeduped; _chain_metric_kind;
                                                       _run_chain_dx_self_check's four exits;
                                                       the self_check docstring
tests/unit/test_niche_c14_encapsulation.py             NEW -- 32 tests
docs/audits/TRACED_LAYER_MAP.md                        NEW -- the living map
docs/audits/C14_ENCAPSULATION_GUARDS_2026_08_03.md     this document
validation/repro_traced_carrier_121/
    probe_c14_byte_identity.py                         NEW -- 36-config extraction proof
    _c14_pre_baseline_lens_traced.py                   the verbatim pre-edit reference
    _c14_focus_ship.txt, _c14_energy_ship.txt          the two production logs
```

No `CHANGELOG.md`, no `pmm/**`, no CI workflow, no `tests/conftest.py`, and no
existing test or validation runner was edited.  Everything is uncommitted.

---

## 7. What remains open

1. **UNIT A and UNIT B are not extracted.**  The architecture study's order is
   C, then A, then B, one per release, each with its own byte-identity proof.
   UNIT C is done; UNIT A's three-site invariant is still maintained by three
   comments, and the layer map now at least *tells* a future author that there
   are three.
2. **The band check inherits C7's declination on the production readout leg.**
   Where the grid's extent is comparable to its own exit fan, both checks
   decline.  Closing the blind spot there needs a hull that fits the grid.
3. **`_SUPPORT_BAND_PEAK_RATIO_TOL` is calibrated by argument, not by a
   population.**  At 1.0 it asks a question no correct field can answer yes to,
   which is why it needed no sweep -- but it is therefore a *coarse* instrument:
   it is silent on C8's `_GHOST` fixture, where a manufactured lobe sits at
   4.6e-02 of peak beyond 3 w but the global maximum is still at the focus.  A
   tighter ratio would catch more and would need the kind of 180-call
   calibration `_RD_HALO_RADIUS_FACTOR` got.  **What is shipped catches the
   E-M6 class and nothing else, and that is the claim.**
4. **C7's message contains "the energy self-check"**, one fixture away from the
   same filter collision C14's first draft hit (S3.3).
5. **The `carrier.py` side has no byte-identity device here.**  It does not need
   one for this change -- every edit is inside `_run_chain_dx_self_check`, which
   only runs under `self_check='dx'` and never touches the returned field -- but
   that is an argument, not a measurement.  A `carrier.py` change that DID touch
   the field would need the `git archive` device, for the reason
   `fc_c9_byte_identity.py`'s header states.
6. **The three documentation contradictions the layer map records** (the
   `DECENTRED_FIT_PREDICTOR` docstring, the C11 audit's headline, the C10/C9
   attribution) are recorded, not fixed -- two of them live in other agents'
   files.

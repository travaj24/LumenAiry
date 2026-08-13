# FIX RELEASE FIFTEEN -- the release run's 15, recovered from 7 names and closed

**2026-08-13.  Branch `fix/release-fifteen` off `origin/main` (`1e6b1b5`),
worktree `C:/tmp/lum_r15` (checkout only -- nothing committed, nothing
pushed).  Closes the 15 failures the full serial release run reported on
merged main, of which only 7 names survived the output pipe.  No `xfail`, no
`skip`, no `importorskip`, no tolerance moved, no assertion deleted, ASCII
only, no CHANGELOG edit.**

---

## 0. HEADLINE

> **All 15 are one class -- the same class `FIX_G8_PROBE` opened and
> `FIX_FINAL_WAVE` closed at six members -- and the class is now 21 members
> in 12 files.**
>
> The 8 lost names were recovered by running the complement of the final
> wave's 42-file family: **113 files, 4 717 passed, 15 failed**.  The
> recovered set contains all 7 knowns, and **the count closes exactly at 15**
> with no widening needed (a further 8 files that consume the traced element
> by other names were run too -- 250 passed, 0 failed).
>
> **Nothing here is a library defect.**  Every one of the 15 asserts a
> property of the coarse-lattice inversion -- an aliasing signature, a
> band-decomposition identity, a saved `map_coordinates` pass, a degraded
> reference, a support ring -- and the inverse-characteristic evaluator
> either supplies that quantity per exit pixel or is structurally excluded
> from one side of the comparison.  Each member is scoped where its claim is
> real, with the mechanism measured on both sides, and **five of the seven
> files gained assertions they did not have** -- four of them stating the
> scoping's own mechanism, so no scoping here can outlive its reason.
>
> **Two members were not scoped at all** in the sense the class usually
> needs: `r8`'s two chain pins are fixed by making the MANUAL reference
> reproduce the orchestrator's own documented leg scoping
> (`propagators/carrier.py` :8326), after which the two sides agree at
> **exactly 0.0** instead of 8.9e-03 -- the `d1` precedent, and the control
> that proves it (hand the ORCHESTRATOR `inverse_map=True` instead and the
> two sides agree at exactly 0.0 as well) is recorded in S2.3.
>
> **One finding is recorded rather than pinned** (S4.1): with the niche-C8
> support bound OFF, `_SUPPORT_BOUND_FEATHER_CELLS` is no longer inert,
> because the evaluator's domain relaxation falls back to it.  A pin on that
> coupling would block the fix for it, so it is written down and left open.
>
> **Two library edits, both comments** (S3): the `sag_chunk_rows`
> byte-identity claim now names its one exception, because the band path
> refuses the evaluator by construction and AUTO-bands at `N >= 4096`.
>
> **GREEN, at the shipped default, on both mounts.**  The whole 113-file
> complement re-reads **4 740 passed, 0 failed** where merged main reads 15
> failed; the seven fixed files read **113 passed** on Windows/MKL/py3.14
> AND, whole rather than sampled, **113 passed** on WSL/OpenBLAS/py3.12; and
> eight of `FIX_FINAL_WAVE`'s 42 reproduce its counts test for test.

---

## 1. RECOVERING THE 8 LOST NAMES

### 1.1 The complement, built by rule and not by guess

Every `tests/unit` file matching `test_niche_*`, `test_hammer_*`,
`test_audit_lens*` or `test_carrier*`, plus every file whose text contains
`apply_real_lens_traced`, minus `FIX_FINAL_WAVE`'s swept 42:

```
  all test files under tests/unit           471
  family-name match                         124
  contains apply_real_lens_traced            88
  union                                     155
  minus the swept 42                        113   <- the complement, run whole
```

Run serially, whole files, `-p no:randomly`, `OPENBLAS/MKL/OMP/NUMEXPR = 1`,
five concurrent single-threaded pytest processes over disjoint file sets
(Windows 11 / py3.14.6 / numpy 2.4.4 / MKL).

| shard | files | result |
|---|---|---|
| 0 | 21 | 952 passed, 53 skipped |
| 1 | 23 | **1 failed**, 755 passed, 4 skipped |
| 2 | 23 | **8 failed**, 908 passed, 1 skipped, 2 deselected |
| 3 | 23 | **3 failed**, 1101 passed |
| 4 | 23 | **3 failed**, 1001 passed, 3 skipped |
| **total** | **113** | **15 failed, 4 717 passed, 61 skipped** |

**15, exactly, and the 7 known names are inside it.**  The reconciliation
needed no widening -- but the widening was run anyway, because a count that
closes on the first try is the moment to check the rule rather than trust
it.  Every remaining `tests/unit` file mentioning
`propagate_traced_carrier_chain`, `prepare_real_lens_traced`,
`carrier_referenced_*`, `inverse_map`, `_lens_imap`, `_lens_traced`,
`TracedScreen` or `traced_kwargs` -- 8 more files -- reads **250 passed, 4
skipped, 0 failed**.

### 1.2 The 15, and which were lost

| # | id | in the 7? |
|---|---|---|
| 1 | `s12_remap_sampling::test_full_is_ray_subsample_independent[2-0.02-0.2]` | yes |
| 2 | `s12_remap_sampling::test_full_is_ray_subsample_independent[4-0.02-0.4]` | yes |
| 3 | `s12_remap_sampling::test_full_is_ray_subsample_independent[8-0.05-0.6]` | yes |
| 4 | `s12_remap_sampling::test_full_actually_changes_the_result_at_coarse_lattice` | yes |
| 5 | `s12_remap_sampling::test_lattice_error_is_confined_beyond_the_alias_radius` | yes |
| 6 | `s10_sibling_patterns::test_row_band_assembly_matches_whole_grid_under_a_carrier` | yes |
| 7 | `r8_tiltaware_chain_api::test_r8_chain_orchestrator_final_focus_readout_matches_manual` | yes |
| 8 | `r8_tiltaware_chain_api::test_r8_chain_orchestrator_matches_manual` | **recovered** |
| 9 | `test_lens_chunked_sag::test_chunked_traced_byte_identical[True]` | **recovered** |
| 10 | `test_lens_chunked_sag::test_chunked_traced_byte_identical[False]` | **recovered** |
| 11 | `test_lens_chunked_sag::test_amp_freed_before_assembly_on_preserve_path` | **recovered** |
| 12 | `perf_round2_2026_08_10::test_nan_pass_guard_is_byte_identical_and_actually_fires` | **recovered** |
| 13 | `r6_auto_carrier_fit::test_auto_matches_explicit_endtoend_vs_oracle` | **recovered** |
| 14 | `c8_inverse_support_bound::test_with_the_flag_off_the_feather_constant_is_inert` | **recovered** |
| 15 | `c8_inverse_support_bound::test_it_leaves_the_beam_exactly_alone` | **recovered** |

Seven files.  The `lastfailed` cache was not consulted at any point.

---

## 2. THE ADJUDICATION, MEMBER BY MEMBER

The house rule (`FIX_FINAL_WAVE` S1.1): **(a)** the asserted property is made
unobservable or augmented by the model -> `inverse_map=False` with the
mechanism recorded; **(b)** map-compatible with the observable read at the
right layer -> fix without the kwarg, preferred; **(c)** a real defect ->
stop and report.

| # | id | what merged main measures | adjudication | why |
|---|---|---|---|---|
| 1-3 | s12 `full_is_ray_subsample_independent` | `d_lat0` = **4.105e-05 / 4.456e-05 / 5.325e-05** vs bars 0.2 / 0.4 / 0.6 | **(a)** arm (a) gains `inverse_map=False` | the pre-C6 arm is where G8 ACCEPTS, and the model then supplies the entrance pullback per pixel -- which is what `'full'` upsamples the lattice to get.  No coarse lattice in the path, no aliasing to measure |
| 4 | s12 `full_actually_changes_the_result_at_coarse_lattice` | `d` = **5.372e-17** rad vs a 0.1 bar | **(a)** + a new arm (c) asserting the mechanism | same; the two modes become indistinguishable to 5e-17 rad |
| 5 | s12 `lattice_error_is_confined_beyond_the_alias_radius` | outer/inner = **1.945** vs a 5.0 bar | **(a)** | same; the signature flattens to 4.263e-05 inner / 8.292e-05 outer |
| 6 | s10 `row_band_assembly_matches_whole_grid_under_a_carrier` | `rel` = **2.189e-02** vs a 1e-12 bar | **(a)** + a new STRUCTURAL arm on the gate | `_imap_domain_gate` carries `not _chunk_assembly`, so band vs whole is two different inversions, not two decompositions of one |
| 7 | r8 `chain_orchestrator_matches_manual` | `rel` = **8.915e-03** vs 1e-9 | **(b) NO SCOPING KWARG ON THE SUBJECT.**  The manual REFERENCE reproduces the orchestrator's own leg scoping | `carrier.py` :8326 sets `inverse_map=False` on every ordinary chain leg, structurally.  A manual pattern without it is a different configuration.  After: **exactly 0.0** |
| 8 | r8 `chain_orchestrator_final_focus_readout_matches_manual` | `rel` = **1.880e-03** vs 1e-9 | **(b)**, same one-line fix in the shared helper | same.  After: **exactly 0.0** |
| 9-10 | chunked_sag `chunked_traced_byte_identical[True/False]` | `np.array_equal` **False** | **(a)** | the band path refuses the evaluator by construction; identity is against the whole-grid path AT THE SAME INVERSION |
| 11 | chunked_sag `amp_freed_before_assembly_on_preserve_path` | `np.array_equal` **False** | **(a)**, on the comparison PAIR only | the free itself -- the claim the test is named for -- still runs at every shipped default and still passes |
| 12 | perf_round2 `nan_pass_guard_is_byte_identical_and_actually_fires` | `3 > 3` map_coordinates calls | **(a)** + an h3-style default contract | the guard optimises the coarse upsample chain the model replaces wholesale: **3 / 3** at the default, **7 / 9** with the evaluator off |
| 13 | r6 `auto_matches_explicit_endtoend_vs_oracle` | `r4_n / r4_a` = **2.031** vs a 3.0 bar | **(a)** on the r4 control ONLY, + a control and the repair asserted | the model repairs the plane-wave-referenced arm's 4th-order residual 15.1x (0.073648 -> 0.004875); the rms control (11.6x) stays at the shipped default |
| 14 | c8 `with_the_flag_off_the_feather_constant_is_inert` | `np.array_equal` **False**, max abs 7.667e-03 | **(a)**, and the coupling RECORDED not pinned (S4.1) | with the bound off there is no measured hull feather, so the evaluator's domain relaxation falls back to the constant |
| 15 | c8 `it_leaves_the_beam_exactly_alone` | 16 core pixels move; power moves 3.974e-07 vs a 1e-9 bar | **(a)** on the `r <= 3 w` PROXY, + a stronger shipped-default arm | the model deliberately keeps a band outside the landing hull alive; on this fixture it lands at 2.32-2.36 w, INSIDE the 3 w proxy, and C8 removes it.  That is C8 working |

### 2.1 s12 -- the model bypasses the lattice the file is about

`test_niche_s12_remap_sampling.py` measures what `remap_sampling='full'`
buys: the transported residual phasor must be sampled at wave-grid
resolution, not off the coarse ray lattice.  Since 2026-08-01 each pin has
two arms -- arm (a) at `REMAP_STATIONARY_PHASE_LAUNCH = False`, the library
state the numbers were calibrated in, and arm (b) at the shipped default,
where niche C6 absorbs this fixture's degree-4 residual and the two modes
collapse together.

**Arm (b) is untouched and was never red.**  At the shipped default G8
REFUSES to build on this fixture -- off-lattice OPL error 2.3699e-03 waves
against the incumbent Newton's 1.7146e-03, i.e. 1.38x on a 1.00x bar -- so
every arm (b) number is byte-for-byte what it was (2.6853e-06, 1.6340e-02 /
5.6227e-02 / 9.1257e-02, inner 3.6215e-04 / 3.6080e-04, outer 1.1284e-01).

**Arm (a) is where the model engages**, because with C6's launch off the
incumbent forward path is the worse pre-C6 one.  Measured on this file's own
fixture at `rs = 4`:

| quantity | model OFF | model ON (merged main) |
|---|---|---|
| `d('lattice', 'full')` | **9.2267e-01** | **5.3717e-17** |
| `d(rs=1 ref, 'lattice')` | **8.3863e-01** | 4.4563e-05 |
| `d(rs=1 ref, 'full')` | 2.6291e-04 | 4.4563e-05 |
| inner / outer (lattice) | 5.166e-02 / 1.896e+00 | 4.263e-05 / 8.292e-05 |

and at `rs = 2` / `8` the `'lattice'` column reads 4.105e-05 / 5.325e-05
where the pins ask for `> 0.2` / `> 0.6`.

The model-OFF column reproduces the file's own 2026-07-25 numbers to every
printed digit.  **The mechanism is not that the aliasing went away: it is
that the model supplies the entrance pullback coordinates and the OPL PER
EXIT PIXEL, which is exactly the quantity `'full'` upsamples the coarse
lattice to obtain.**  `C6 off + model on` is a combination the library has
never shipped.

So arm (a) names both flags, through one module-level `_PRE_C6` dict that
carries the whole argument in its comment, and every assertion and every
tolerance is word for word.  **`test_full_actually_changes_the_result_at_
coarse_lattice` gained an arm (c)** that asserts the mechanism directly --
hand the pre-C6 forward path to the evaluator and the two modes agree to
`< 1e-12` rad -- so if a future build stops engaging the model there, the
scoping's reason is measured to be gone rather than assumed to hold.

### 2.2 s10 and chunked_sag -- the band path refuses the model by construction

`_imap_domain_gate` (`_lens_traced.py` :8466):

```
    _imap_domain_gate = (sub > 1 and inversion_method == 'newton'
                         and not _chunk_assembly and not use_gpu)
```

and the gate's own note (:10311): *"the band path exists to never
materialise a full-grid float64; handing it one would undo the memory fix it
is"*.  Measured with `_imap_out` on the s10 fixture:

| call | `gate_open` | `engaged` |
|---|---|---|
| `sag_chunk_rows=0` (whole grid) | **True** | **True** |
| `sag_chunk_rows=32` (banded) | **False** | **False** |

so at the shipped default the two sides of every `sag_chunk_rows`
byte-identity pin are the MODEL and the INCUMBENT, and the difference is the
size of the accuracy gain, not of a bug:

| fixture / arm | banded vs whole, model ON | model OFF |
|---|---|---|
| s10, carrier arm, `cr = 32 / 128 / 7` | **2.188649e-02** (all three) | **0.000000e+00** |
| s10, `carrier=None` arm | **2.421623e-01** | **0.000000e+00** |

Both files keep their assertions word for word at `inverse_map=False`, where
the band loop and the whole-grid loop are the same algorithm and the R7
interpolation-order claim is the claim being made.  **s10 gained a
structural arm** that asserts the exclusion at the gate itself -- whole-grid
opens it, banded does not -- so the scoping is pinned to its cause and not
to a number.  `test_lens_chunked_sag.py` cites that pin rather than
duplicating it, and `test_amp_freed_before_assembly_on_preserve_path` keeps
its own claim (the eager `del`) on a call at every shipped default; only the
comparison pair is scoped.

**The consequence for large N is real and is now documented** (S3): banding
is AUTO-ON at `N >= 4096`, and `_chunk_assembly` additionally requires the
`'screen'` amplitude (`ray_density` forces it off).  So a large-N `'screen'`
call keeps the incumbent coarse-Newton inversion -- refused, never degraded,
i.e. the pre-5.35 shipped behaviour -- and `sag_chunk_rows=0` is the way to
keep the evaluator.  Design 121's production path is `ray_density`, so it is
unaffected; the shipping banner is not moved by this.

### 2.3 r8 -- the manual pattern was missing the orchestrator's own scoping

`propagate_traced_carrier_chain` sets, structurally, on every ORDINARY chain
leg (`propagators/carrier.py` :8305-:8326, niche C15):

```
    _leg_kw = dict(call_kw)
    _leg_kw.setdefault('inverse_map', False)
```

with the reason stated there: an intermediate leg's output is re-fitted by
every leg after it, so the evaluator is scoped to the leg nothing re-fits.
`test_niche_r8_*`'s `_manual_chain` re-implements that leg with the public
API and did NOT pass it, so it followed the module default `True`.  The two
sides were therefore running different configurations.  Measured:

| arm | F4.1 (2 groups, 15 mm) | F4.2 (1 group + focus readout) |
|---|---|---|
| manual at the module default (merged main) | rel **8.914547e-03** | rel **1.880114e-03** |
| **manual with `inverse_map=False`** (this fix) | **0.000000e+00** | **0.000000e+00** |
| orchestrator with `traced_kwargs=dict(inverse_map=True)` vs manual at the default | **0.000000e+00** | -- |

The last row is the control, and it is what makes this **(b)** rather than
**(a)**: the two sides agree exactly whenever they take the SAME setting, in
both directions, so the leg scoping is the whole of the difference and
nothing else moved.  The orchestrator -- the subject of both pins -- runs at
every shipped default, and the fix is one kwarg in the test's own reference
helper, with the argument in its docstring.  The `setdefault` escape hatch
that the control exercises is itself pinned in
`test_niche_c15_inverse_map.py`.

### 2.4 perf_round2 -- the guard's subject is not on the shipped path

Item 2 of `FIX_PERF_ROUND2_2026_08_10` skips the second, NaN-mask
`map_coordinates` pass when the coarse array carries no NaN.  The model
replaces that whole chain with one per-pixel evaluation
(`_lens_traced.py` :10455).  Counted on the test's own design-121-like
fixture:

| arm | guard active | guard defeated | `isnan` calls seen |
|---|---|---|---|
| shipped default | **3** | **3** | 1 |
| `inverse_map=False` | **7** | **9** | 7 |

So `cnt_off > cnt_on` is a measurement of which inversion ran, not of the
guard.  The counting pair moves to `inverse_map=False`, where the guard
executes and saves its two calls, and the byte-identity it claims is
measured there.  **A default arm was added** in the h3 style: the shipped
path must price strictly fewer `map_coordinates` calls than the incumbent
needs (3 < 7).  That is a contract -- if a future build puts the coarse
upsample chain back behind the model, it fails.

### 2.5 r6 -- the model repairs the degraded reference

`test_auto_matches_explicit_endtoend_vs_oracle` proves `carrier='auto'`
recovers the same correction as an explicit `carrier=R`, and uses
`carrier=None` (the plane-wave reference, the state `'auto'` used to fall
back to silently) as the non-vacuity control.  Measured against the inline
meridional oracle, both settings, `gate_open` True on all six calls:

| arm | rms, model ON | r4, model ON | rms, model OFF | r4, model OFF |
|---|---|---|---|---|
| `explicit` | 0.012811 | 0.002305 | 0.012811 | 0.002305 |
| `auto` | 0.016434 | 0.002400 | 0.016434 | 0.002400 |
| `none` | 0.191150 | **0.004875** | 0.219320 | **0.073648** |
| control `r4_n / r4_a` | **2.031** | | **30.685** | |
| control `rms_n / rms_a` | **11.632** | | 13.346 | |

**The two carrier arms do not move with the flag at all** -- six digits --
and that is now asserted, so the scoping is provably about the degraded
reference and nothing else.  The `rms` control keeps its bar at the shipped
default.  Only the `r4` control moves to `inverse_map=False`, where it was
calibrated, and the repair that caused the collapse is asserted at the
default (`r4_n < 0.2 * r4_n0`, measured 0.0662).

### 2.6 c8 -- the bound is doing its job, on a band the model keeps alive

The evaluator's domain mask deliberately relaxes past the landing hull by
`sqrt(2) sub dx` of plateau plus one feather (`_lens_traced.py` :10499),
because *"the element EMITS a ring outside the hull at full amplitude ...
model exactly the support the element emits"*.  Niche C8's taper is what
removes that ring.  On C8's own CLEAN fixture, with the model engaged:

| quantity | value |
|---|---|
| pixels the bound moves | **16** of 147 456 |
| their radii | **2.321 - 2.357 w** (inside the test's `r <= 3 w` proxy) |
| their amplitude, bound OFF -> ON | **6.290162e-03 of peak -> exactly 0.0** |
| power removed | **3.974e-07** of the total |
| with `inverse_map=False` | **0** pixels move in the core; whole grid 1.167e-11 |

So the `r <= 3 w` mask is a PROXY for "inside the traced support", and the
model moved the support-relative band inside it.  Arm (a) keeps the byte
identity where the proxy is exact.  **Arm (b) states what the bound actually
guarantees, at the shipped default, and it is stronger than a byte
comparison over a radius**: every pixel it moves it drives to EXACTLY zero,
all of them lie outside 2 w, and the power it removes is bounded at 1e-5.

The second c8 pin is the flag's OFF state.  With
`REMAP_INVERSE_SUPPORT_BOUND = False` there is no hull and therefore no
MEASURED feather, so the evaluator's relaxation falls back to
`_SUPPORT_BOUND_FEATHER_CELLS * sub * dx` and an absurd constant reaches the
field (7.667e-03 of peak between `feather = 1.0` and `1e6`).  See S4.1: it
is recorded, not pinned.

---

## 3. THE LIBRARY EDITS -- TWO COMMENTS

Both in `lumenairy/elements/_lens_traced.py`; neither changes a returned bit.

| site | was | now |
|---|---|---|
| `sag_chunk_rows` parameter doc (:6839) | "Byte-identical to the whole-grid path." | ...WITH ONE NAMED EXCEPTION since v5.35: the band path refuses the evaluator by construction, so on a call the evaluator would engage, banding selects the incumbent inversion; identity holds against the whole-grid path at `inverse_map=False` (measured 2.19e-02 otherwise); `sag_chunk_rows=0` keeps the evaluator |
| the row-band block comment (:7628) | "Values are byte-identical to the whole-grid path ... pinned by `test_chunked_assembly_byte_identical`" | ...that identity is against the whole-grid path AT THE SAME INVERSION, with the gate and the S10 pin named |

This is the `FIX_FINAL_WAVE` S3 class -- a doc-vs-code contradiction that the
default flip created -- and it is the one the AUTO threshold makes worth
stating: `N >= 4096` bands without being asked.

---

## 4. FINDINGS RECORDED, NOT PINNED

### 4.1 The C8 feather constant is not inert while the C8 bound is off

`_SUPPORT_BOUND_FEATHER_CELLS` is documented and pinned as inert when
`REMAP_INVERSE_SUPPORT_BOUND` is False -- "the whole support computation is
skipped, so the constant cannot reach the field".  Since the evaluator
shipped on, that switch no longer removes the constant from the path: the
model's domain relaxation reads `_exit_support.feather` and falls back to
`_SUPPORT_BOUND_FEATHER_CELLS * sub * dx` exactly when the bound is off,
because that is the state in which no hull, and therefore no measured
feather, is built.

**Not reachable from a shipped configuration through this switch** -- with
the bound ON (the default) the relaxation uses the measured hull feather.
The fallback IS reachable at shipped defaults by another route: the
`'screen'` amplitude never sets `want_bound`, so a `'screen'` call's
relaxation is one constant-sized cell by design.

It is left OPEN and unpinned deliberately: a pin on the coupling would block
the fix for it.  The fix, if it is taken, is for the relaxation to name its
own allowance instead of borrowing C8's constant.

### 4.2 The band path's refusal is silent

`_imap_domain_gate` closes before any build is attempted, so no
`report_refusal` fires and a large-N `'screen'` call gives no indication
that it kept the incumbent inversion.  Every other refusal in this subsystem
announces itself (guards G1-G8 all route through `report_refusal`).  Not
changed here -- it would move warning output across the suite -- and
recorded as the natural next step for whoever owns the AUTO threshold.

---

## 5. VERIFICATION

### 5.1 The seven fixed files, whole, at the shipped default

Windows 11 / py3.14.6 / numpy 2.4.4 / MKL, `-p no:randomly`, one process per
file, threads pinned to 1.

| file | before (merged main) | after |
|---|---|---|
| `test_niche_s12_remap_sampling.py` | 5 failed / 5 passed | **10 passed** (19.92 s) |
| `test_niche_s10_sibling_patterns.py` | 1 failed / 27 passed | **28 passed** (27.01 s) |
| `test_niche_r8_tiltaware_chain_api.py` | 2 failed / 11 passed | **13 passed** (68.13 s) |
| `test_lens_chunked_sag.py` | 3 failed / 13 passed | **16 passed** (34.39 s) |
| `test_niche_c8_inverse_support_bound.py` | 2 failed / 11 passed | **13 passed** (54.53 s) |
| `test_niche_perf_round2_2026_08_10.py` | 1 failed / 25 passed | **26 passed** (22.24 s) |
| `test_niche_r6_auto_carrier_fit.py` | 1 failed / 6 passed | **7 passed** (13.53 s) |
| **total** | **15 failed / 98 passed** | **113 passed, 0 failed** |

`test_nan_pass_guard_is_byte_identical_and_actually_fires` carries
`@pytest.mark.slow`, so it was additionally run by id on its own: **1
passed**.

### 5.2 The whole complement, re-run at the shipped default

All 113 files again, same five shards, same pins, `TRACED_INVERSE_MAP =
True`, no marker filter beyond the repo's own `addopts` (`-m "not
integration"`, which is what the release run uses; `slow` is NOT deselected,
so the `perf_round2` pin runs).

| shard | before | after |
|---|---|---|
| 0 | 952 passed, 53 skipped | **960 passed**, 45 skipped |
| 1 | **1 failed**, 755 passed, 4 skipped | **754 passed**, 6 skipped |
| 2 | **8 failed**, 908 passed, 1 skipped | **916 passed**, 1 skipped |
| 3 | **3 failed**, 1101 passed | **1104 passed** |
| 4 | **3 failed**, 1001 passed, 3 skipped | **1006 passed**, 1 skipped |
| **total** | **15 failed / 4 717 passed / 61 skipped** | **4 740 passed, 0 failed, 53 skipped** |

`4 732 + 61 = 4 740 + 53 = 4 793` collected ids either way -- the skip count
moves because eight of the RAM-guarded cases (`available_memory_bytes()`,
`_skip_if_low_ram`) ran this time; nothing was deselected or removed.
2 deselected in shard 2 both runs (the `integration` marker).

### 5.3 The 42-family spot-checks

Eight of `FIX_FINAL_WAVE`'s 42, chosen as the five files that wave itself
fixed or scoped plus the three that consume what this wave touched
(ray-density amplitude, the flag registry, the decentred fit).  **Every
count matches `FIX_FINAL_WAVE` S2 exactly.**

| file | this wave | FIX_FINAL_WAVE S2 |
|---|---|---|
| `test_niche_c15_inverse_map.py` | 40 | 40 |
| `test_niche_c6_fit_guard.py` | 13 | 13 |
| `test_niche_d5_dx_flatness_gate.py` | 13 | 13 |
| `test_hammer_h3_traced_nyquist_guard.py` | 5 | 5 |
| `test_niche_d1_tilted_carrier.py` | 33 | 33 |
| `test_niche_d7_decentred_fit.py` | 37 | 37 |
| `test_niche_p11_ray_density_amplitude.py` | 10 | 10 |
| `test_niche_c14_encapsulation.py` | 32 | 32 |
| **total** | **179 passed, 0 failed** (4 RAM-guarded skips) | 183 |

### 5.4 The second mount

WSL / Ubuntu 24.04 / py3.12.3 / numpy 2.4.6 / scipy 1.17.1 / OpenBLAS, the
same worktree over `/mnt/c`, threads pinned to 1.  **All seven fixed files,
whole -- not a representative half -- and every count is identical to the
Windows one.**

| file | WSL | Windows |
|---|---|---|
| `test_niche_s12_remap_sampling.py` | 10 | 10 |
| `test_niche_s10_sibling_patterns.py` | 28 | 28 |
| `test_lens_chunked_sag.py` | 16 | 16 |
| `test_niche_c8_inverse_support_bound.py` | 13 | 13 |
| `test_niche_r6_auto_carrier_fit.py` | 7 | 7 |
| `test_niche_perf_round2_2026_08_10.py` | 26 | 26 |
| `test_niche_r8_tiltaware_chain_api.py` | 13 | 13 |
| **total** | **113 passed, 0 failed** | **113** |

(two runs: `54 passed in 263.85s` and `59 passed in 499.41s`.)

Three of these carry the arithmetic risk worth naming, and all three hold on
OpenBLAS/py3.12 as well as on MKL/py3.14:

* **r8's exact 0.0** -- the orchestrator and the manual pattern agree to the
  last bit on the masked norm, on both BLAS libraries, for both the two-group
  chain and the focus-readout chain.
* **c8's "every moved pixel is exactly 0.0"** and its 2 w radius bound -- a
  claim about which pixels the model's support relaxation keeps alive, i.e.
  about a convex-hull test on traced landings.
* **perf_round2's map_coordinates counts** (3 at the default against 7 with
  the evaluator off) -- a dispatch-count contract, which a different SciPy
  could in principle route differently; it does not.

---

## 6. HYGIENE

| check | result |
|---|---|
| `ruff check lumenairy/ tests/unit/` (the exact CI command) | **All checks passed** |
| non-ASCII on lines ADDED by this branch | **0** |
| lines added over 79 columns | **0** |
| `xfail` / `skipif` / `pytest.skip` / `importorskip` added | **none** |
| tolerances moved | **none**.  Every bar is the original number: 0.02/0.2, 0.02/0.4, 0.05/0.6, 20x, 0.1, 5x, 1e-12, `array_equal`, 1e-9, `cnt_off > cnt_on`, 3x, 1e-9 power |
| assertions deleted | **none**.  Six of the seven files GAINED assertions: s12 the mechanism arm, s10 the gate arm, perf_round2 the no-coarse-chain contract, r6 the flag-invariance control and the repair, c8 the removal-to-exactly-zero arm |
| library behaviour changed | **none**.  Both edits under `lumenairy/` are comments |
| `.test_durations` | **not regenerated** -- the orchestrator does it in the release run |
| CHANGELOG | **not edited**, per the campaign's convention |
| committed / pushed / tagged | **nothing** |

---

## 7. UNCOMMITTED FILES

`git status --porcelain` on `C:/tmp/lum_r15` (branch `fix/release-fifteen`,
base `1e6b1b5`).  7 modified test files + 1 modified library file + 1 new
doc; `git diff --stat` = **369 insertions, 37 deletions**.

```
 M lumenairy/elements/_lens_traced.py                 sag_chunk_rows byte-identity claim, 2 comment sites (S3)
 M tests/unit/test_lens_chunked_sag.py                3 pins: the traced byte-identity pair scoped
 M tests/unit/test_niche_c8_inverse_support_bound.py  2 pins scoped + the removal-to-exactly-zero arm
 M tests/unit/test_niche_perf_round2_2026_08_10.py    the NaN-pass counting pair scoped + the no-coarse-chain contract
 M tests/unit/test_niche_r6_auto_carrier_fit.py       the r4 control scoped + the flag-invariance control + the repair
 M tests/unit/test_niche_r8_tiltaware_chain_api.py    _manual_chain reproduces the orchestrator's leg scoping
 M tests/unit/test_niche_s10_sibling_patterns.py      1 pin scoped + the structural gate arm
 M tests/unit/test_niche_s12_remap_sampling.py        5 pins: arm (a) names both flags + the mechanism arm
?? docs/audits/FIX_RELEASE_FIFTEEN_2026_08_13.md      this file
```

**Both edits under `lumenairy/` are comments.**  Nothing was committed,
nothing was pushed, no tag was touched, and no artefact under `validation/`
was regenerated.

---

## 8. NOT COVERED BY THIS WAVE

1. **The 42-family was not re-run whole**, only spot-checked (S5.3).  This
   branch changes no library behaviour -- both `lumenairy/` edits are
   comments -- so the family cannot move; `FIX_FINAL_WAVE` S2 is its
   measurement.
2. **`.test_durations`**, the version bump, the era rename and the banner
   re-baseline are all the orchestrator's steps and are untouched.
3. **S4.1 and S4.2 are open** -- the C8 feather coupling and the band path's
   silent refusal.  Both are recorded with their mechanism; neither is a
   wrong answer at any shipped configuration.
4. **No GPU / CuPy path** and no JAX-guarded file was separately verified.
5. **The whole fast gate was not run.**  The search was the complement of
   the final wave's family plus the widening of S1.1, and the count closed
   at 15 inside it.

# FIX FINAL WAVE -- the P1, the four P2s, and four more of the P1's class

**2026-08-13.  Branch `fix/final-wave` off `origin/main` (`3154fa7`), worktree
`C:/tmp/lum_fw` (checkout only -- nothing committed, nothing pushed).  Closes
the ONE P1 and the P2s of `docs/audits/VERIFY_FINAL_2026_08_12.md`, plus the
four further members of the P1's class that the mandated whole-family sweep
turned up.  No `xfail`, no `skip`, no `importorskip`, no tolerance moved, no
assertion deleted, ASCII only, no CHANGELOG edit.**

---

## 0. HEADLINE

> **The P1 is closed, and it is closed at the SHIPPED DEFAULT.**
>
> All four red ids pass with `TRACED_INVERSE_MAP = True`, and
> `test_niche_d1_tilted_carrier.py` + `test_niche_d7_decentred_fit.py` read
> **70 passed, 0 failed** where merged main reads 4 failed / 66 passed.
>
> **One of the four did not need the scoping kwarg at all.**  The d1 beam-centre
> pin's non-vacuity arm was being read off the RETURNED FIELD, which is a
> shadow of the difference it is about; it is now read at the fit-domain
> resolution, ABOVE the model, where the element names the carrier-supplied
> beam centre to four decimals in its own announcement.  That reading is
> byte-identical under both settings of the flag, so the pin runs at the
> shipped default and cannot go inert again the way the field reading did.
>
> The three d7 pins are scoped to `inverse_map=False`, the knob the five
> siblings of `FIX_G8_PROBE` use, each with the measurement that shows WHICH
> asserted property the model alters.  **The verifier's alternative -- filter
> `_solve_census`'s rows by shape -- was tested and is provably insufficient**:
> two of the three fail on `folds >= 1`, a FIELD-level assertion that reads 0
> with the model on, which no census filter can reach.  One of the three gained
> a SECOND arm that is strictly stronger than what it asserted before (exact
> byte identity, `np.array_equal`, in place of a tolerance).
>
> **THE SWEEP FOUND FOUR MORE MEMBERS OF THE SAME CLASS, in three more files
> `VERIFY_FINAL` never ran** -- two in `test_niche_d5_dx_flatness_gate.py`, one
> in `test_hammer_h3_traced_nyquist_guard.py`, one in
> `test_audit_lens_models_2026_07.py` (the suite `FIX_VERIFY_ARCH` S10 itself
> called "the ONE outstanding suite").  All four were red on merged main at the
> shipped default; none can be caused by this branch, whose every library edit
> is a comment.  **So the class is six members in five files, not four in two**
> -- and the verifier's own prediction that "more members may be there" is what
> found them (S2.1, S2.2).
>
> Three of the four gained a default-arm statement STRONGER than what they had:
> the aperture:beam ray-FIT cliff is now asserted **not to occur at all** on the
> shipped path (residual 2.5e+06x smaller than the independent GBD
> propagator's); the dx-flatness lesson is stated in the SHIPPED gate's own
> currency instead of a FWHM bar its own docstring had flagged as weak; and the
> Newton pool test now asserts that the shipped path prices **no pool worker at
> any box size**, which is a contract, not a workaround.  In that last case the
> re-derivation its failure message asks for was tried first and **measured to
> be impossible** -- the clamp prices a per-pixel Newton that the shipped path
> does not have.
>
> The four doc-vs-code contradictions are fixed (six sites -- two more of the
> same union scar were found and fixed).  The obliquity margin was re-measured
> on both mounts at three thread settings: **digit-identical in all six
> configurations, and the JSON payloads reproduce with zero numeric
> difference**.  The margin is EVIDENCED and the constant is left at 0.10, with
> the reason recorded: it is not a thin one-sided 2x, it is the geometric
> centre of a two-sided window `(0.0496, 0.2091)` in which no shipped
> disposition changes.  The comment's own "2x over the worst" was itself
> wrong -- the worst measured ratio is 0.055224, i.e. 1.81x -- and now says so.
>
> `.test_durations` was NOT regenerated: the orchestrator does it in the
> release run.

---

## 1. THE P1, PER TEST

### 1.1 The class, and the pattern it is fixed with

`FIX_G8_PROBE_2026_08_12` S6.1/S6.2 identified the class the
`TRACED_INVERSE_MAP` default flip creates -- pins whose fail-before or
non-vacuity arm goes INERT once the inverse-characteristic model engages,
because the model supplies the OPL, the entrance coordinates and `det J` per
pixel and therefore removes the returned field's dependence on the FORWARD
fit -- and re-scoped **five** members with a per-call `inverse_map=False`,
keeping every assertion word for word and recording, in each docstring, which
property the model alters.  Four more members existed in two files that branch
never ran (`FIX_CI_RECONCILE` S6.1 E2: "Campaign S11 records d7 and c6 were
never run at all on that branch").  This wave fixes those four.

**The house pattern has two forms and the campaign has used both.**  Where the
claim can only be made about the forward fit, scope to `inverse_map=False`
(the five siblings; `c6_fit_guard::test_guard_raises_the_fit_order_like_d7`).
Where the model's arm is itself a stronger true statement, keep BOTH arms
(`c6::test_the_two_newton_fit_backends_still_describe_the_same_map`, whose
default arm asserts exactly 0.0 where it used to assert 5e-04).  A third form
is used once here, and it is the best outcome available: **move the observable
upstream of the model** so the pin needs no scoping at all.

### 1.2 Adjudication table

| pin | what merged main measures | adjudication | why |
|---|---|---|---|
| d1 `test_tilted_carrier_supplies_the_beam_centre_by_default` | `max\|derived - origin\|` = **4.155e-08** vs a 1.021e-06 bar | **MAP-COMPATIBLE.  Observable read upstream**, no scoping kwarg on the primary arms | the difference is MADE at the fit-domain resolution, which is above the model and byte-identical across the flag.  The original field assertion is kept word for word on a forward-path arm |
| d7 `test_the_decentred_path_really_did_change` | `max\|a - b\|` = **exactly 0.0** vs a 1.000e-08 bar | **BOTH ARMS.**  forward arm `inverse_map=False` keeps the assertion word for word; default arm asserts `np.array_equal` | fix D5 deliberately does not hand the D7 ORDER raise to the degree-14 exit model (`_lens_traced.py` :9366-9370, "only the WEIGHTS travel"), so the field cannot depend on it.  Byte identity is STRONGER than the 1e-8 tolerance it replaces at the default |
| d7 `test_the_fold_regularisation_is_still_load_bearing_at_the_d7_order` | `ratio_good` = **1.029199** (rcond 1.797e-14) vs a 1.001 bar | **`inverse_map=False`** on both censuses | the census GROWS 6 -> 8 rows; the two new rows are the model's own `(203401, 120)` degree-14 exit fit and `_worst_draw` maxes over them.  Census filtering is NOT sufficient: `folds_bad` also reads **0** where the test needs `>= 1` |
| d7 `test_c13_cures_the_hard_mask_fold_at_the_d7_order` | `folds_pre` = **0** vs `>= 1` | **`inverse_map=False`** on both censuses | the FAILING assertion is field-level, so census filtering cannot reach it at all.  `ratio_pre` stays 68546.42 either way -- the forward draws are still MADE, they just no longer reach the returned field |

### 1.3 d1 -- the observable, measured on both sides

The claim is "a TiltedCarrier supplies the beam centre".  The beam centre
resolves a ray-fit disc, and the element names that disc's centre in
millimetres in its own announcement (`_lens_traced.py` :9190, :9392).  Measured
on the test's own fixture (n = 256, dx = 20 um, w = 300 um, x_c = 1.2 mm,
`fit_radius_beam_factor = 2.0`, `ray_subsample = 8`):

```
  arm        flag ON (shipped)                          flag OFF
  derived    "|r - (1.2000, 0.0000) mm| <= 0.6000 mm"   IDENTICAL STRING
             "...the beam is 1.2000 mm off the grid centre..."
  explicit   byte-identical message to derived           IDENTICAL
  origin     NO ray-fit-domain message at all            IDENTICAL
```

The origin arm has none because its origin-referenced second moment
`sqrt(2 x_c^2 + w^2)` -- the very failure niche D1 exists to fix, pinned by
`test_beam_radius_is_measured_about_the_beam_not_the_grid_origin` -- sizes a
disc that covers the whole launch square, so there is no off-axis restriction
to report.  That is the difference itself, named, rather than its downstream
shadow, and it is strictly more specific than "the two fields differ".

The downstream shadow, for the record:

| flag | scale | `max\|derived - explicit\|` | `max\|derived - origin\|` | 1e-6 bar |
|---|---|---|---|---|
| ON (shipped) | 1.02063 | 0.000e+00 | **4.155e-08** | 1.021e-06 |
| OFF | 0.98552 | 0.000e+00 | **2.715e-03** | 9.855e-07 |

A 65 000x collapse, landing 25x under the bar.  No returned bit is wrong; the
observable moved.

`on_aperture_beam` moves `'silent' -> 'warn'` to read the announcement.  That
is bit-neutral by construction (`_w_in_beam` is measured whenever
`fit_radius_beam_factor is not None`, :8537) and by measurement: both arms are
**byte-identical across the switch** (`np.array_equal` True, max abs diff
0.000e+00).

### 1.4 d7 -- why the census filter was rejected

`VERIFY_FINAL` S6.5 offered filtering `_solve_census`'s rows to the decentred
forward fit's own shape before `_worst_draw` maxes over them.  Measured on the
two fold pins, at their own era pins, under both flag settings:

```
  test_the_fold_regularisation_is_still_load_bearing_at_the_d7_order
    flag ON   good  rows=8  worst=(1.797e-14, 1.029199)  <- (203401,120) x2 added
              bad   rows=8  worst=(0.000e+00, 68546.42)  folds_bad=0   r_bad=0.000000
    flag OFF  good  rows=6  worst=(1.336e-11, 1.000003)
              bad   rows=6  worst=(0.000e+00, 68546.42)  folds_bad=1   r_bad=0.000175

  test_c13_cures_the_hard_mask_fold_at_the_d7_order
    flag ON   pre   rows=8  worst=(0.000e+00, 68546.42)  folds_pre=0   r_pre=0.000000
              post  rows=8  worst=(0.000e+00, 1.000000)  folds_post=0  r_post=0.000000
    flag OFF  pre   rows=6  worst=(0.000e+00, 68546.42)  folds_pre=1   r_pre=0.000175
              post  rows=6  worst=(0.000e+00, 1.000000)  folds_post=0  r_post=0.000175
```

`folds_bad` and `folds_pre` are counts of the element's own fold-caustic
warning in the RETURNED call, not census rows.  Both go 1 -> 0 with the model
on, and no filter over `_solve_census`'s rows can restore either.  The fold
lives in the fitted forward map; the model does not read the fitted forward
map.  So `inverse_map=False` is the only scoping that keeps these two pins
measuring what they name.

Two further facts are recorded in the test docstrings because they are the
reason the scoping is honest rather than convenient:

* the `(203401, 120)` and `(1137, 120)` rows are a total-degree-14 2-D fit,
  `(14+1)(14+2)/2 = 120` terms, and they are ill-conditioned only under THESE
  FIXTURES' era pin.  In the SHIPPED configuration (`LSTSQ_CONDITIONING_STEPDOWN
  = True`) the same model fit returns the least-squares answer exactly
  (`ratio = 1.000000`, `VERIFY_FINAL` S6.3c).  Nothing about the model is
  broken; the pins force the pre-C13 solver on purpose.
* the physical output does not move: the off-beam ratio the C13 cure exists to
  protect reads 0.000176 under both flag settings on the shipped
  (non-degenerate) arm.

### 1.5 Fail-before / pass-after

```
  BEFORE (merged main 3154fa7, shipped default)
    FAILED test_niche_d1_tilted_carrier.py::test_tilted_carrier_supplies_the_beam_centre_by_default
           assert 4.15530831578945e-08 > (1e-06 * 1.0206255118836145)
    FAILED test_niche_d7_decentred_fit.py::test_the_decentred_path_really_did_change
           assert 0.0 > (1e-08 * 0.9999983120845224)                          (:384)
    FAILED test_niche_d7_decentred_fit.py::test_the_fold_regularisation_is_still_load_bearing_at_the_d7_order
           the weighted restriction stopped conditioning the solve: ... 1.029199x (rcond 1.797e-14)   (:645)
    FAILED test_niche_d7_decentred_fit.py::test_c13_cures_the_hard_mask_fold_at_the_d7_order
           the pre-C13 arm stopped folding -- assert 0 >= 1                    (:819)

  AFTER (this branch, shipped default, one command)
    4 passed in 18.09s
    tests/unit/test_niche_d1_tilted_carrier.py tests/unit/test_niche_d7_decentred_fit.py
      -> 70 passed, 7 warnings in 287.45s
```

`VERIFY_FINAL` S4.2 flags `FIX_CI_RECONCILE`'s **37/37** claim for
`test_niche_d7_decentred_fit.py` as stale downward (main measures 34/37).  It
now measures **37/37**.

---

## 2. THE FULL TRACED FAMILY, AT THE SHIPPED DEFAULT

`VERIFY_FINAL` S8.10 made this the fix wave's job: "the fix wave must run that
whole family at the shipped default -- more members of the class may be
there."  **All 42 files, 788 collected ids, GREEN**, with
`TRACED_INVERSE_MAP = True` and no marker filter (so this is a superset of
CI's `-m "not integration and not slow"` shard).  Windows 11 / py3.14.6 /
numpy 2.4.4 / MKL, `-p no:randomly`, whole files.

The sweep was run from both ends concurrently (a forward job over
`d1 -> p7` and a reverse job over `audit_lens_models -> p6`) so the two met in
the middle; `p8` / `p9` were finished by a third direct run, and the three
files this wave FIXED were re-run whole afterwards.  Where two jobs covered
the same file the counts agreed exactly (`p6` 16/16, `p7` 18/18, `p8` 6/6,
`p9` 13/13).

| # | file | tests | # | file | tests |
|---|---|---|---|---|---|
| 1 | `test_niche_d1_tilted_carrier.py` | **33** | 22 | `test_niche_p5_sampling.py` | **11** |
| 2 | `test_niche_d2_chain_multi.py` | **38** | 23 | `test_niche_p6_astigmatic_aperture.py` | **16** |
| 3 | `test_niche_d5_dx_flatness_gate.py` | **13** (fixed) | 24 | `test_niche_p7_seidel_gate.py` | **18** |
| 4 | `test_niche_d6_exact_tilted_leg.py` | **38** | 25 | `test_niche_p8_capstone.py` | **6** |
| 5 | `test_niche_d7_decentred_fit.py` | **37** (fixed) | 26 | `test_niche_p9_decenter_tilt.py` | **13** |
| 6 | `test_niche_d8_congruence_workers.py` | **35** | 27 | `test_niche_p10_transverse_walk_remap.py` | **13** |
| 7 | `test_niche_d9_grid_origin.py` | **23** | 28 | `test_niche_p11_ray_density_amplitude.py` | **10** |
| 8 | `test_niche_s8_sphere_carrier_reference.py` | **13** | 29 | `test_niche_e4_corrected_relay_oracle.py` | **10** |
| 9 | `test_niche_tight_focus_readout.py` | **15** | 30 | `test_niche_c1_consolidation.py` | **38** |
| 10 | `test_hammer_h1_slant_obliquity.py` | **3** | 31 | `test_niche_c3_gap_paraxial_guard.py` | **18** |
| 11 | `test_hammer_h2_displaced_projection.py` | **7** | 32 | `test_niche_c5_exact_tilted_reference.py` | **29** |
| 12 | `test_hammer_h3_traced_nyquist_guard.py` | **5** (fixed) | 33 | `test_niche_c6_fit_guard.py` | **13** |
| 13 | `test_hammer_h6_traced_carrier_eikonal.py` | **6** | 34 | `test_niche_c6_stationary_phase_launch.py` | **21** |
| 14 | `test_hammer_h7_gbd_diverging.py` | **9** | 35 | `test_niche_c9_sphere_parab_exact_conversion.py` | **13** |
| 15 | `test_niche_p1_gbd_chain.py` | **8** | 36 | `test_niche_c11_decentred_fit_arbiter.py` | **21** |
| 16 | `test_niche_p1_traced_tiltaware.py` | **6** | 37 | `test_niche_c12_physics_fit_selection.py` | **20** |
| 17 | `test_niche_p2_design_battery.py` | **26** | 38 | `test_niche_c14_encapsulation.py` | **32** |
| 18 | `test_niche_p2_displaced_extreme.py` | **9** | 39 | `test_niche_c15_inverse_map.py` | **40** |
| 19 | `test_niche_p2_guards.py` | **12** | 40 | `test_fix_tilt_quadratic_opl.py` | **7** |
| 20 | `test_niche_p3_pointwise_obliquity.py` | **13** | 41 | `test_niche_audit_w9_traced_determinism.py` | **7** |
| 21 | `test_niche_p4_gbd_reexpand.py` | **14** | 42 | `test_audit_lens_models_2026_07.py` | **69** (fixed) |

**788 passed, 0 failed, 0 skipped, 0 xfailed.**  The three "(fixed)" rows and
`d1` / `d7` are post-fix whole-file re-runs; `test_audit_lens_models_2026_07.py`
went **1 failed / 68 passed (23:26)** before the fix and **69 passed (19:54)**
after.

### 2.0 A SECOND REGRESSION BATCH, for the modules this wave edited

Not part of the mandated family, run because this wave touched
`_lens_real.py`, `_lens_imap.py`, `_traced_flags.py`, `doe_rcwa.py` and
`test_niche_c15_inverse_map.py`.  **12 files, 412 tests, all green:**
`test_screen_obliquity.py` 47, `test_doe_rcwa.py` 41,
`test_fix_d5_fit_domain_basis.py` 45,
`test_niche_c10_residual_eikonal_degree.py` 9,
`test_niche_c13_lstsq_conditioning.py` 20, `test_niche_d3_guards.py` 41,
`test_niche_d4_dgrating.py` 59, `test_audit_lens.py` 52,
`test_audit_w5_elements_misc.py` 22,
`test_niche_audit_e_prepared_and_enums.py` 39,
`test_v4_14_0_dispatcher_pin_apply_lens.py` 35,
`test_audit_except_budget.py` 2.

### 2.-1 THE SECOND MOUNT

WSL / Ubuntu / py3.12.3 / numpy 2.4.6 / scipy-openblas 0.3.31, same worktree
over `/mnt/c`, `D121_ROOT` pointed at the design assets.  A representative
half was run there; **every file green, and the counts are identical to the
Windows ones**:

| file | WSL | Windows |
|---|---|---|
| `test_niche_d1_tilted_carrier.py` | 33 | 33 |
| `test_niche_d7_decentred_fit.py` | **37** | 37 |
| `test_niche_c15_inverse_map.py` | 40 | 40 |
| `test_niche_c6_fit_guard.py` | 13 | 13 |
| `test_niche_c6_stationary_phase_launch.py` | 21 | 21 |
| `test_fix_d5_fit_domain_basis.py` | 45 | 45 |
| `test_niche_c11_decentred_fit_arbiter.py` | 21 | 21 |
| `test_niche_c12_physics_fit_selection.py` | 20 | 20 |
| `test_niche_c14_encapsulation.py` | 32 | 32 |
| `test_niche_s8_sphere_carrier_reference.py` | 13 | 13 |
| `test_niche_audit_w9_traced_determinism.py` | 7 | 7 |
| `test_screen_obliquity.py` | 47 | 47 |
| `test_hammer_h3_traced_nyquist_guard.py` | **5** | 5 |
| `test_niche_d5_dx_flatness_gate.py` | **13** | 13 |
| `..._2026_07.py::test_tp2_fit_inversion_matches_newton` | **1** | 1 |

**All six fixed pins are confirmed on the second mount**, and two rows carry
the most arithmetic risk:

* `d7`'s new default arm asserts `np.array_equal` -- **exact byte identity
  between the D7-raised and pre-D7 arms with the model engaged** -- and it
  holds on OpenBLAS/py3.12 as well as on MKL/py3.14.  That is the claim most
  likely to have been arithmetic-fragile, and it is not.
* `h3`'s new arm asserts that the shipped path prices NO Newton pool worker.
  The pool is a resource decision that reads `psutil` and the RAM budget, so
  a different OS could in principle route it differently; it does not.

### 2.1 THE SWEEP'S OWN FINDING: TWO MORE MEMBERS OF THE SAME CLASS

`VERIFY_FINAL` S8.10 listed `d5` among the files that "were still running or
not started", and predicted "more members of the class may be there".  There
are exactly two, both in `tests/unit/test_niche_d5_dx_flatness_gate.py`, both
red on merged main at the shipped default, both invisible to `VERIFY_FINAL`:

```
  FAILED test_niche_d5_dx_flatness_gate.py::test_dx_flatness_alone_is_not_sufficient
         assert rows[-1]['fwhm'] / oracle['fwhm'] > 2.5    -- reads 2.0669
  FAILED test_niche_d5_dx_flatness_gate.py::test_the_level_gap_is_the_traced_fit_radius_cliff
         assert e_def > 10 * e_gbd                         -- reads 1.2346e-08 vs 3.0870e-02
  2 failed, 11 passed in 373.63s
```

Neither can be caused by this branch: every library edit in this wave is a
comment or a docstring.  Both were re-measured under both settings of the flag
and the flag is the whole cause.

**`test_the_level_gap_is_the_traced_fit_radius_cliff`** -- the exit-wavefront
error at `r = w` on one stigmatic conic singlet, collimated input, NA 0.20:

| arm | no fit restriction | `frbf = 1.5` |
|---|---|---|
| `inverse_map=False` | **4.428078** | 0.0873737 |
| SHIPPED default | **1.234590e-08** | 1.680842e-11 |
| `apply_real_lens_gbd` (independent oracle) | 0.0308696 | -- |

The forward arm reproduces the test's own 2026-07-29 recorded ladder
(4.4281 / 0.0874 / 0.0309) **to every printed digit**, so scoping it to
`inverse_map=False` keeps the three assertions unchanged and measures them
where the cliff is real.  The cliff is a property of the element's GLOBAL OPL
FIT -- it bites because that fit has to represent the whole exit sphere -- and
the model does not read that fit.  A FOURTH assertion was added at the shipped
default: `e_map < 0.1 * e_gbd`, measured 1.23e-08 against 3.09e-02, i.e. the
model's residual is **2.5e+06x smaller than the independent GBD propagator's**
and the cliff does not occur at all on the shipped path.  Asserted rather than
recorded, so a future build in which the model silently stops engaging here
brings the cliff back loudly.

**`test_dx_flatness_alone_is_not_sufficient`** -- and this one needed NO
scoping for its lesson, only for its bar.  The test's own docstring, written
2026-08-01, says the FWHM bar was the weak assertion and that "a further
accuracy improvement could walk through" it.  One did:

| arm | FWHM um (512 / 1024) | dx-spread | / oracle | EE2 | window |
|---|---|---|---|---|---|
| `inverse_map=False` | 7.73940 / 7.74532 | 7.65e-04 | **2.8234** | 9.184 | 75.721 |
| SHIPPED default | 5.66997 / 5.66998 | **1.61e-06** | **2.0669** | 18.279 | 77.949 |

At the shipped default the deliberately-broken `carrier_reference='parabola'`
configuration is dx-FLAT to **1.6e-06** -- 3100x inside the 5e-03 bar, FLATTER
than before -- while delivering 0.226 of the oracle's enclosed energy against
the gate's 0.70 bar and losing 22 % of the launched power out of the window.
**The lesson is untouched: a flatness-only gate still passes it silently.**  So
the flatness half and both level assertions stay at the shipped default, a new
assertion states the lesson in the SHIPPED gate's own currency
(`dx_flatness_gate(rows, oracle)` returns violations, one of them naming the
independent oracle -- no bar of this test's own), and only the FWHM ratio moves
to the `inverse_map=False` arm where the 2.5 was calibrated and where it
measures 2.8234.  Read at the ladder's finest rung, which is the rung that
assertion has always used.

```
  AFTER:  2 passed in 200.79s   (the two ids)
          13 passed in 405.25s  (the whole file)
```

### 2.2 AND TWO MORE, IN TWO MORE FILES THE VERIFIER NEVER RAN

**`test_hammer_h3_traced_nyquist_guard.py::test_the_pool_cap_notice_is_silenced_by_its_own_knob`**
-- the Newton pool memory clamp, emulated on a pinned 12 GB box.  Its own
failure message asks the next reader to "re-derive the pin from
`_newton_worker_bytes` rather than deleting the assertion".  **That was tried
first, and it cannot work**: the clamp prices a PER-PIXEL NEWTON INVERSION, and
with the model engaged there is no per-pixel Newton to dispatch, so the pool
decision is never taken at ANY box size.  Measured over a pinned-RAM ladder:

| pinned free RAM | `inverse_map=False` | SHIPPED default |
|---|---|---|
| 12 / 8 / 6 / 4 / 3 / 2 / 1 GB | clamp prices a worker and warns at **every** rung (1.871 GB per worker, 4096 Newton points/chunk, 140625-point ray-fit grid) | `_newton_worker_bytes` **never called**, 0 warnings, at every rung |

So the 12 GB pin is kept exactly as CI's runners have it, on the arm where the
pool exists, and a THIRD block was added that asserts the shipped path prices
no worker at all -- which is both the reason for the scoping and a real
contract: if a future build reintroduces a per-pixel Newton dispatch behind the
model, it fails.  `5 passed`.

**`test_audit_lens_models_2026_07.py::test_tp2_fit_inversion_matches_newton`**
-- `inversion_method='fit'` against the Newton inversion, `rel < 1e-3`.  The
model is gated on `inversion_method == 'newton'` **by design**
(`_lens_traced.py` :8466, and the gate's own comment at :10308: "the 'fit' path
is already a per-pixel exit polynomial"), so at the shipped default this
compares the MODEL against the 'fit' polynomial -- two different per-pixel exit
representations -- and not the two forward-fit inversions the test names:

| sub | `inverse_map=False` | SHIPPED default | map built? |
|---|---|---|---|
| 1 | 3.1849e-06 | 3.1849e-06 | neither (the gate needs `sub > 1`) |
| 4 | 3.1833e-06 | **6.1318e-02** | `'newton'` yes, `'fit'` no |

`sub = 1`, where the gate refuses on both arms, reads the SAME number under
both flags -- the control that says the scoping changed nothing else.  The
6.1e-02 is the accuracy the model ADDS on the `newton` arm where the coarse
Newton had to upsample a 4x-subsampled lattice.  Assertion and 1e-3 bar
unchanged.

**Six members of this class in five files, then, not four in two.**  All five
files are green at the shipped default.

---

## 3. P2-3 / P2-4 -- THE DOC-VS-CODE CONTRADICTIONS

Six sites; the verifier named four and two more of the same union scar were
found by grepping the retired probe's name across the tree.  None changes a
returned bit.

| # | site | was | now |
|---|---|---|---|
| 1 | `validation/pipeline/doe_rcwa.py` :255 | `n_orders` "default 12; the convergence ladder and its chosen headroom are in the build note" | DEFAULT 6, stated as a CEILING and as NOT CONVERGED, in the same words the default's own site (:154) already used, with the build-note reference |
| 2 | `lumenairy/elements/_lens_real.py` :2896 | the guard fires "when **40%** of it still does" | **10%**, naming `_SCREEN_OBLIQUITY_RESIDUAL_FRAC` and the R1 term that took it 0.40 -> 0.10 |
| 3 | `lumenairy/elements/_lens_imap.py` `_incumbent_fingerprint` | G8 accepts "on **held-out samples**"; quotes a refusal string `G8: held-out OPL error ...` | OFF-LATTICE probe points, with the site references (:1521, :1531, :1537) and the real refusal string; the retired probe is named as retired and the scar is recorded as a scar |
| 4 | `validation/repro_traced_carrier_121/imap_prod_121.py` :11, :156 | "held-out ray samples" (prose AND the printed report line) | "off-lattice probe points", both places |
| 5 | `lumenairy/elements/_traced_flags.py` `ERAS` block | "``v5.32.1`` is the current tree"; "``pyproject.toml`` and ``__init__.py`` both still say ``5.32.0``" | rewritten: `v5.34` is the current tree, both files read 5.34.0, and the source-only-era note now covers BOTH names |
| 6 | `tests/unit/test_niche_c15_inverse_map.py` :779 (found by this wave) | "the map is accepted only if it beats the INCUMBENT on held-out samples" -- present tense, same scar | off-lattice probe points, with the retirement named |

**P2-4, the era label.**  The verifier's adjudication is adopted and its two
checks were re-run here: `git cat-file -e v5.34.0:lumenairy/elements/_lens_imap.py`
**fails** (the module does not exist at that tag), and `ERAS` at that tag is
`('v5.31', 'v5.32', 'v5.32.1')`.  So the era was never a reproduction of
released 5.34.0 and the fix is to the LABEL's description, not to the label
(renaming `ERAS[-1]` is the version bump's step, and it would collide with it).
`_traced_flags.py` and `TRACED_LAYER_MAP.md` S2 now both say, in the same
terms:

* `v5.32.1` and `v5.34` are SOURCE-ONLY era names and neither reproduces the
  release it resembles;
* `v5.34` is the post-inverse-characteristic development tree, i.e. the tree
  that ships as 5.35.0 -- which is exactly what `TRACED_LAYER_MAP` rows 32-33's
  `since` column already said, so the two documents no longer contradict;
* `ERAS[-1]` is PINNED to reproduce the live shipped defaults
  (`test_the_newest_era_reproduces_the_live_shipped_values`), so its entries
  move when a default moves -- `TRACED_INVERSE_MAP` went `False` -> `True`
  under this name.  The newest era is a coordinate on the CURRENT SOURCE, not
  a fixed coordinate across commits; only the older eras are archaeology.  Cite
  it with the file hash `traced_flag_state` prints, never alone.

No value in the registry changed, so `test_niche_c14_encapsulation.py`'s
liveness check is untouched.

---

## 4. P2-2 -- THE OBLIQUITY RESIDUAL MARGIN

### 4.1 What was asked, and what was measured

`_SCREEN_OBLIQUITY_RESIDUAL_FRAC` moved 0.40 -> 0.10 in the LOOSENING
direction (the guard fires on `estimate * FRAC > TOL`, so a smaller FRAC warns
less) on rotationally symmetric surfaces only, with a claimed 2x of margin.
Re-measured here with `validation/repro_traced_carrier_121/screen_obliquity_derive.py`
in modes `sphere` (the symmetric single-surface ladder: N-BK7 / N-SF11,
R = +-25 / 50 mm, 10 / 30 / 54.87 / 100 mrad) and `d121` (the six design-121
groups at each group's own extreme-order carrier, 1 / 2 / 3 mm pupils).

**Six configurations, two mounts, three thread settings each:**

| mount | python | numpy | BLAS | threads | result |
|---|---|---|---|---|---|
| Windows 11 | 3.14.6 | 2.4.4 | MKL | 1, 2, 8 | every printed digit identical |
| WSL (Ubuntu) | 3.12.3 | 2.4.6 | scipy-openblas 0.3.31 | 1, 2, 8 | every printed digit identical, and identical to Windows |

Stronger than the printed digits: both runs regenerate
`_screen_obl_d121.json` and `_screen_obl_sphere.json`, and a full-precision
field-by-field comparison of the WSL-regenerated files against the committed
(Windows-generated) ones reads **max relative difference exactly 0.000e+00**.
The only textual difference is the line ending.  The number is not an
arithmetic accident.

### 4.2 The ratios, exactly

From `_screen_obl_d121.json` rather than its printed 5 decimals:

| case | r = 1 mm | r = 2 mm | r = 3 mm |
|---|---|---|---|
| group 5 (the binding case) | 0.040260 | 0.043894 | **0.047966** |
| group 4 | 0.037474 | 0.050153 | **0.055224** |
| group 2 | 0.041997 | 0.035100 | 0.029664 |
| group 3 | 0.001773 | 0.001930 | 0.002033 |
| groups 0, 1 (plates) | 0 exactly | 0 exactly | 0 exactly |

Single spherical surfaces (`sphere` mode, 20 rows): **0.0012 to 0.0064**.

So the worst ratio measured anywhere is **0.055224** -- group 4 at 3 mm, whose
absolute error is 0.00048896 waves, 102x inside the lambda/20 tolerance -- and
the worst that is materially large is **0.047966**.  **The shipped comment's
"2x margin over the worst" was therefore wrong in the unsafe direction: it is
1.81x over the worst ratio and 2.09x over the binding case.**  The comment now
says so, and carries the table.

### 4.3 The margin is EVIDENCED, and the constant is left at 0.10

The new fact this wave adds is that the constant is bounded on BOTH sides, so
it is a choice inside a window rather than a thin one-sided margin.  Each
shipped fixture's guard estimate names the FRAC at which its disposition
flips (`fires iff estimate * FRAC > 0.05`):

| fixture | estimate (waves) | requirement | boundary |
|---|---|---|---|
| `_out_of_envelope_case` (must fire) | 1.00880 | `FRAC > 0.05 / 1.00880` | **0.0496** |
| design 121 group 5 (must NOT fire) | 0.23910 | `FRAC < 0.05 / 0.23910` | **0.2091** |
| `_steep_case` (must NOT fire) | 0.12860 | `FRAC < 0.05 / 0.12860` | 0.3888 |

Every disposition the suite pins is unchanged for any FRAC in
`(0.0496, 0.2091)` -- a 4.2x-wide window -- and **0.10 sits essentially at its
geometric centre**, `sqrt(0.0496 * 0.2091) = 0.1018`.

The safer value was derived rather than merely declined: **FRAC = 0.15** would
buy 2.72x over the worst measured ratio, still fire on the out-of-envelope
case and still stay silent on design 121 group 5 -- but it spends the other
side, moving the design-of-record false alarm from 2.09x away to 1.39x away.
It was NOT taken, because no value inside the window addresses the actual open
item, which the comment already names and this wave leaves open: **the leftover
has not been measured on a decentred / tilted / biconic / freeform element.**
That is a measurement someone has to make, not a constant someone can choose.
The derivation is recorded at the constant so the next reader can act on it
without re-deriving it.

No returned bit changed: `_SCREEN_OBLIQUITY_RESIDUAL_FRAC` is still `0.10`.

---

## 5. HYGIENE

| check | result |
|---|---|
| `ruff check lumenairy/ tests/unit/` (the exact CI command), Windows ruff 0.15.13 | **All checks passed** |
| the same command under WSL, ruff 0.15.16 | **All checks passed** |
| non-ASCII on lines ADDED by this branch | **0** (three em-dashes slipped into `TRACED_LAYER_MAP.md` and were replaced with `--` before the sweep; re-checked after every later edit) |
| non-ASCII anywhere in this document | **0** |
| lines added over 79 columns | **0** |
| `xfail` / `skipif` / `pytest.skip` / `importorskip` added | **none** |
| tolerances moved | **none**.  Every bar in all six pins is the original number: 1e-6, 1e-10, 1e-8, 1.001, `folds >= 1`, 2.5, `10 * e_gbd`, `0.4 * e_def`, 1e-3, the 12 GB pin |
| assertions deleted | **none**.  Every arm that moved to `inverse_map=False` kept its assertion word for word; four tests GAINED an assertion (d7 exact byte identity, d5 the shipped gate's own verdict, d5 the no-cliff bound, h3 the no-worker-priced contract) |
| `.test_durations` | **deliberately not regenerated** -- the orchestrator does it in the release run |
| CHANGELOG | **not edited**, per the campaign's convention |
| library behaviour changed | **none**.  Every edit under `lumenairy/` is a comment or a docstring; `_SCREEN_OBLIQUITY_RESIDUAL_FRAC` is still `0.10` |

---

## 6. UNCOMMITTED FILES (this branch, nothing committed, nothing pushed)

`git status --porcelain` on `C:/tmp/lum_fw` (branch `fix/final-wave`, base
`3154fa7`).  12 modified + 1 new; `git diff --stat` = **440 insertions, 80
deletions**.

```
 M docs/audits/TRACED_LAYER_MAP.md                         era/source-only note (P2-3 #5, P2-4)
 M lumenairy/elements/_lens_imap.py                        _incumbent_fingerprint docstring (P2-3 #3)
 M lumenairy/elements/_lens_real.py                        40% -> 10% (P2-3 #2) + the FRAC evidence block (P2-2)
 M lumenairy/elements/_traced_flags.py                     ERAS prose (P2-3 #5, P2-4)
 M tests/unit/test_audit_lens_models_2026_07.py            tp2 scoped to inverse_map=False
 M tests/unit/test_hammer_h3_traced_nyquist_guard.py       pool-cap pair scoped + no-worker-priced contract
 M tests/unit/test_niche_c15_inverse_map.py                held-out -> off-lattice comment (P2-3 #6)
 M tests/unit/test_niche_d1_tilted_carrier.py              observable read upstream (P1)
 M tests/unit/test_niche_d5_dx_flatness_gate.py            two pins scoped + two new default-arm assertions
 M tests/unit/test_niche_d7_decentred_fit.py               three pins scoped + exact-identity arm (P1)
 M validation/pipeline/doe_rcwa.py                         n_orders default 12 -> 6 (P2-3 #1)
 M validation/repro_traced_carrier_121/imap_prod_121.py    held-out -> off-lattice, prose + report (P2-3 #4)
?? docs/audits/FIX_FINAL_WAVE_2026_08_13.md                this file
```

**Every edit under `lumenairy/` is a comment or a docstring.**  Nothing was
committed, nothing was pushed, no tag was touched, and the two regenerable
JSON artefacts the obliquity probes rewrite
(`_screen_obl_d121.json`, `_screen_obl_sphere.json`) were restored to their
committed line endings after the cross-mount runs, so they do not appear
above.

---

## 7. NOT COVERED BY THIS WAVE

1. **`.test_durations` (P2-1)** -- 383 collected ids without a duration entry.
   Left to the release run by instruction.
2. **P3s of `VERIFY_FINAL` S7.4** -- the pre-existing `lumenairy/__init__.py`
   cp1252 mojibake, `probe_c6_energy.py`'s stale `_REMAP_RESID_TAPER_IN/_OUT`
   attributes, and the probe scripts committed into `scripts/`.  All
   pre-existing and byte-unchanged here.
3. **Release mechanics (S7.5)** -- the 5.34.0 -> 5.35.0 version bump, the 14
   sites already stamping `v5.35.0`, the empty `[Unreleased]` CHANGELOG section
   and the era rename that the bump makes natural.  The orchestrator's step.
4. **The decentred / tilted / biconic / freeform obliquity residual** -- see
   S4.3.  It is the open item P2-2 is really about and it is still open.
5. **The 32-order design-121 fan and the RCWA-vs-scalar A/B** were not re-run.
6. **No GPU / CuPy path**, and no JAX-guarded file was separately verified.
7. **The banner was not re-run.**  Nothing in this wave can move it -- every
   library edit is a comment -- but it is stated rather than assumed.
8. **The whole fast gate was not run**, only the 42-file traced family
   (788 ids), the 12-file regression batch (412 ids) and the 15 WSL runs.  A
   member of the P1's class could still exist OUTSIDE the traced family; the
   search was as wide as `VERIFY_FINAL` S8.10 asked for and no wider.  The
   mechanism to grep for is a test that compares the traced element's returned
   field across a knob of the FORWARD fit, or that instruments a per-pixel
   Newton.
9. **The second mount is a representative half, not the whole family** -- 15
   files there against 42 on Windows, chosen as the five sibling files, the
   five files this wave touched, and the BLAS-sensitive determinism suites.

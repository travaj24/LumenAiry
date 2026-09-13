# WP-A25 -- the P2 battery's unclipped doublet cell scores a periodic REPLICA of its own focus; WP-A6/C1 only moved which replica

Subject: `tests/unit/test_niche_p2_design_battery.py::test_battery_through_focus_unclipped_doublet_matches_gaussian`
stepping from FWHM 16.50 um / EE(2 waists) 0.953 to 20.50 um / 0.495 at
`a18ab074` (WP-A6, finding C1 "focus readout sized from the beam") and staying
there through HEAD `6c83ac91`.

Everything below is a MEASUREMENT taken on this machine with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, Python 3.14.6 /
numpy 2.4.6, against the battery fixture's own analytic-Gaussian oracle
(`fwhm_th = 1.177 lambda |R| / (pi w_exit)`, which shares no readout knob with
the code under test).  The pre-C1 library was extracted READ-ONLY with
`git archive 818251fd lumenairy` into the scratchpad and the CURRENT test
module run against it, so only the library varies.  Nothing was checked out,
stashed or written to the repository; no git write command was issued.

---

## 1. Summary

| finding | status | files:lines | tests | oracle | measured before -> after |
|---|---|---|---|---|---|
| **A25-1** the 0.953 -> 0.495 step is a defect in WP-A6/C1 | **NOT A C1 DEFECT -- the attribution is corrected, C1's leg is untouched** | no change to the resolver or its guard | `test_audit2609_a25_carrier_focus_readout.py::test_c1s_beam_referenced_leg_is_still_the_leg_this_cell_runs_on` | analytic Gaussian | C1 lengthened the leg 337.468 -> 372.144 um (correct: the envelope's fitted residual curvature is +0.01762 /m against a carrier `1/R` = -15.1596 /m); the Bluestein period followed by the same 10.27 %, 112.548 -> 124.113 um, and the brightest REPLICA moved from output pixel (30, 30) to the window CORNER (0, 0) |
| **A25-2** a caller who waives the replica refusal has no way to keep the window without the replicas | **FIXED** -- new `replica_fill={'repeat','zero'}` on both public readouts, default `'repeat'` (nothing existing moves) | `lumenairy/propagators/carrier.py:3829-3926` (`_REPLICA_FILLS`, `_check_replica_fill`, `_fill_readout_replicas`), `:2773`/`:3019`/`:3108-3109` (paraxial), `:5238`/`:5511-5512`/`:5800-5806` (exact), `:9531`/`:9805` (reachable from the chain's `focus_readout` and the multi entry point's `output_grid`) | `::test_the_two_fills_differ_only_outside_one_period`, `::test_a_faithful_window_is_untouched_by_either_fill`, `::test_the_fill_is_validated_and_not_a_silent_fall_through`, `::test_the_knob_keeps_the_readouts_input_dtype`, `::test_replica_fill_reaches_the_readout_through_the_chain`, plus the seven battery-cell pins | analytic Gaussian; the stop-plane power | battery cell **20.50 um / EE2w 0.4953 -> 18.50 um / 0.9970**, returned-window power **5.7004 -> 0.99873** of the stop plane's, best plane **+0.393 -> +0.131 mm** |
| **A25-3** the replica refusal told callers that a peak or a width still reads correctly | **FIXED** (message + both readouts' `on_replica` docs) | `carrier.py:3652-3676` (docstring), `:3769-3789` (the two regimes) + `:3788-3826` (the message), `:2855-2879` / `:5383-5401` (both `on_replica` entries) | `::test_the_refusal_names_the_knob_and_no_longer_promises_a_safe_peak` (fail-before on the text), `::test_the_core_s_replica_enters_the_window_at_exactly_two_periods` (the criterion it now states) | the same battery cell | "the spot CORE is unaffected -- so a width or a peak still looks right" -> the two measured regimes, split at the derived two-period line, with the counter-example |
| **A25-4** the faithful window size was computed and thrown away | **ADDED** | `carrier.py:3917-3918` (`_period_out['faithful_samples']`), `:3513-3530` (`readout_faithful_samples` per chain stage) | `::test_the_cell_really_does_ask_for_more_window_than_one_period`, `::test_the_chain_publishes_the_faithful_window_size_per_stage` | the reported period | nothing published -> `(249, 249)` of 512 on the battery cell, asserted against the reported period rather than a stored number |
| **A25-5** the battery helper asks for a window the readout cannot deliver | **REQUESTED** (outside my ownership -- section 5) | `tests/unit/test_niche_p2_design_battery.py:339-347`, one dict key | verified in process by `::test_the_unclipped_cell_reproduces_the_analytic_gaussian_focus` and by re-running the battery file with the same key injected (26 passed) | analytic Gaussian | subject test RED (20.50 um / 0.4953) -> GREEN (18.50 um / 0.9970); the two sibling through-focus cells do not move at all |
| **A25-6** the whitelist fixture must carry every key the whitelist accepts | **REQUESTED** (outside my ownership -- section 5) | `tests/unit/test_niche_c1_consolidation.py:1070`, one dict key | `test_the_focus_readout_whitelist_is_exactly_what_the_chain_consumes` itself | -- | the only test the change breaks, and it is the gate doing its job |

**Headline.**  The step at `a18ab074` is real and reproducible, and it is not a
defect in C1.  The fixture has never measured its own focus through this
readout: it asks for a 256.000 um window at a focus whose Bluestein period is
~124 um and waives the refusal (`on_replica='ignore'`), so more than half of
every sample it scores is a periodic replica of the core rather than signal,
and its `np.argmax`-led metric locks onto one.  C1's 10.3 % longer leg -- the
right leg, the envelope really does carry residual curvature here -- moved the
period by the same 10.3 % and moved the brightest replica from a harmless place
to the window's corner, where three quarters of the encircled-energy disc falls
off the grid.

The library keeps returning the replicas by default, because three shipped
fixtures deliberately depend on that (section 4 measures the cost of changing
it: 9 tests in 4 files).  What it gains is the ability to say "keep the window,
drop the replicas" -- `replica_fill='zero'` -- plus the size of the faithful
window and a refusal message that no longer promises a safe peak.  With that
one key the battery cell reads 18.500 um / 1.0624x / EE2w 0.9970, which is what
the same readout returns at any standoff long enough to cover the window, and
what this fixture's own 2026-07-25 docstring records.

---

## 2. Per finding

### A25-1 -- what actually moved at `a18ab074`, measured on both sides

**The readout's inputs are identical across the step.**  Instrumenting
`carrier_referenced_focus_readout` on `818251fd` (= `a18ab074^`) and on HEAD
with the same fixture (`_through_focus(_d_doublet, 2.0e-3, 2.5)`):

```
                         818251fd (pre-C1)      6c83ac91 (HEAD)
R (carrier at exit)      -65.96537 mm           -65.96537 mm      identical
z (final_distance)       +65.96537 mm           +65.96537 mm      identical
grid                     1024 x 21.484375 um    1024 x 21.484375 um
beam radius w_env         1.8599 mm              1.8599 mm        identical
grid half-extent          11.000 mm (5.914 w)    11.000 mm        identical
fitted 1/R_env           +0.0176178 /m          +0.0176161 /m     (C2's centred fit)
resolved standoff        337.468 um             372.144 um        +10.27 %
Bluestein period         112.548 um             124.113 um        +10.27 %
requested window         256.000 um             256.000 um        identical
window / period          2.2746                 2.0627
returned power / P_in    8.7542                 5.7004
peak of the best plane   pixel (30, 30)         pixel (0, 0)
  ... in metres          (-113.0, -113.0) um    (-128.0, -128.0) um
  ... modulo the period  (-0.452, -0.452) um    (-3.887, -3.887) um
FWHM / EE1w / EE2w       16.50 um / .843 / .953 20.50 um / .353 / .495
dz_best                  +0.1311 mm             +0.3934 mm
```

So the step is ONE quantity: the leg, and with it the period.  Everything the
resolver measures -- the beam, the grid, the containment (3.1893 measured
against a 3.2000 modelled target on both sides) -- is unchanged.

**The leg C1 chose is the right leg.**  The envelope handed to the readout is
not flat: `_fit_carrier_inv` reads `1/R_env = +0.01762 /m` against the
carrier's `1/R = -15.1596 /m`.  That is the beam's own focus sitting PAST the
carrier's, and it is independently visible in the scan -- the best focus is at
`dz = +0.131 mm`, which needs `1/R_env = +0.0300 /m` to explain: same sign,
same order.  C1's `_beam_containment_standoff` therefore asks for a longer leg
than the carrier-referenced law (372.144 um against 337.468 um, a factor
1.1027) and the readout takes the longer of the two, exactly as documented.
Nothing in this work package shortens it, and
`::test_c1s_beam_referenced_leg_is_still_the_leg_this_cell_runs_on` re-measures
both legs from the fixture's own exit envelope and asserts the readout used the
beam-referenced one.

**The fixture was already scoring a replica before C1.**  At `818251fd` the
window is 2.2746 periods wide and the peak of the best plane sits at pixel
(30, 30) -- 113.0 um from the window centre, which reduces to 0.452 um from the
origin modulo the period, i.e. a near-exact image of the core.  The
encircled-energy disc (2 waists = 29.6 um) still fitted around it with only a
circular segment lost, which is why 0.9527 looked plausible against a truth of
0.9970.  At 124.113 um of period the brightest image lands at (0, 0), the
window's corner: three quarters of that disc is off the grid and the reading
collapses to 0.4953.  Both readings are artefacts; only one of them was loud.

**THE CRITERION, and where the artefact came in.**  A replica of the CORE is
inside the window exactly when the window is wider than TWO periods: the
nearest replica's centre sits one period from the origin and the window's edge
at half its span.  Under that line the replicas are confined to the wings and
the fixture's core-confined metrics survive; over it they do not.  Measured in
process on the same fixture, same fill, only the leg varied -- the two legs the
module shipped before the 2026-08-06 extent-following law, and the two it
resolves now:

```
standoff                 period      win/per   FWHM    ratio    EE2w    dz_best   peak
6.0 z_R  = 3147.166 um  1049.606 um   0.244   18.500  1.0624   0.9970  +0.1311  (256,256)
0.8 z_R  =  419.622 um   139.948 um   1.829   18.500  1.0624   0.9970  +0.1311  (256,256)
337.468 um (pre-C1)      112.548 um   2.275   16.500  0.9476   0.9527  +0.1311  ( 30, 30)
372.144 um (C1, HEAD)    124.113 um   2.063   20.500  1.1773   0.4953  +0.3934  (  0,  0)
```

So the 2026-07-25 record was taken on a readout with a 1.05 mm period -- a
quarter of the requested window, entirely replica-free -- and the fixture's own
docstring ("FWHM 18.5 um vs 17.4 um theory (1.062x), EE1w 86.0% / EE2w 99.7% /
EE3w 99.8%") is that replica-free reading to every digit it quotes.  The
0.8 z_R leg still reads it at 1.829 periods, with 1.2348x of the window's power
already replicas: the wings were wrong and nothing the fixture measures looked
at them.  The extent-following law took the ratio past two, and from there the
fixture has been scoring an image of its own core.

### A25-2 -- `replica_fill`: keep the window, drop the replicas

`angular_spectrum_propagate_mft` reconstructs a field that obeys
`E(u + period) == E(u)` identically in the ABSOLUTE output coordinate, so only
`|u| <= period/2` about the transform's own origin carries measurement.  The
module has refused a wider window by default since D3 (2026-08-06).  What a
caller who WAIVES that refusal got was the replicas, and a replica is not a
degraded reading: it is a full-amplitude image of the core laid down where the
real field is weak, so it wins every max / argmax / centroid /
encircled-energy reduction taken over the window -- including the one a spot
budget uses to decide where the spot IS.

`replica_fill` (`'repeat'` default, `'zero'`) is now a per-call choice on both
public readouts, implemented by `_fill_readout_replicas`
(`lumenairy/propagators/carrier.py:3841`).  Three properties make it safe:

* the region it governs is the EXACT complement of `_check_readout_replica`'s
  condition, so it is empty precisely when
  `2|centre_out| + N_out*dx_out <= period` holds on both axes.  A faithful
  window is returned by IDENTITY on either setting -- the same object, not
  merely equal values (`::test_a_faithful_window_is_untouched_by_either_fill`);
* the two settings are BIT-IDENTICAL inside one period, so nothing about the
  measurement depends on the knob
  (`::test_the_two_fills_differ_only_outside_one_period`);
* it does not move the leg.  The standoff stays the accuracy-optimal one
  `_default_focus_standoff` resolves from the beam, for the reason stated
  there: buying window with leg length costs the core
  (`L ~ 0.155 NA^3 f^1.6`, so the 12.5x leg a 204.8 um window needs on the
  tight-focus fixture would cost ~20 % at NA 0.278), and coupling the leg to
  the window breaks the `K == 1` contract between
  `propagate_traced_carrier_chain_multi` and `propagate_traced_carrier_chain`.

Measured on the battery cell, only the window content moving (same leg, same
period, same containment, to the bit):

| quantity | pre-C1 trees | HEAD / `'repeat'` | **`replica_fill='zero'`** | standoff 768 um | standoff 1536 um |
|---|---|---|---|---|---|
| best-focus FWHM | 16.500 um | 20.500 um | **18.500 um** | 18.500 um | 18.500 um |
| FWHM / analytic (17.413 um) | 0.9476 | 1.1773 | **1.0624** | 1.0624 | 1.0624 |
| EE 1 waist | 0.8431 | 0.3531 | **0.8585** | 0.8585 | 0.8585 |
| EE 2 waists | 0.9527 | 0.4953 | **0.9970** | 0.9970 | 0.9970 |
| EE 3 waists | 0.9532 | 0.5032 | **0.9980** | 0.9980 | 0.9980 |
| best plane | +0.1311 mm | +0.3934 mm | **+0.1311 mm** | +0.1311 mm | +0.1311 mm |
| returned / stop-plane power | 8.7542 | 5.7004 | **0.99873** | 0.99800 | 0.99800 |
| peak of the best plane | (30, 30) | (0, 0) | **(256, 256)** | (256, 256) | (256, 256) |

The last three columns are three INDEPENDENT readout geometries with no
replicas in them -- one produced by blanking at the shipped leg, two by
lengthening the leg until one period covers the window -- and they agree to the
digit.  That agreement is what says 18.500 um is the beam and not another
artefact.

**The two sibling battery cells do not move.**  Both also ask for windows wider
than one period (1.267 and 1.544 periods), but their brightest replica never
wins, so blanking changes nothing they score:

| cell | `'repeat'` | `replica_fill='zero'` |
|---|---|---|
| `..._truncated_aperture_broadens_predictably` (1.2x) | 23.500 um / 1.2257x / EE 0.7535 / 0.9229 / 0.9292 / dz +0.0000 mm | **identical to the digit** (window energy 0.99643 -> 0.99391) |
| `..._relay_cliff_is_a_focal_catastrophe`, guard ON | 28.500 um / 1.0962x / EE 0.8381 / 0.9797 / 0.9817 / dz +0.0000 mm | **identical** (0.99345 -> 0.98853) |
| `..._relay_cliff_is_a_focal_catastrophe`, guard OFF | FWHM nan / EE 0.0000 / 0.0001 / 0.0002 / dz -0.2300 mm | **identical** (period 10.925 mm, nothing to blank) |

### A25-3 -- the guidance that licensed the waiver

`_check_readout_replica`'s refusal message and both readouts' `on_replica`
docs said, in the module's own words, that past one period "the spot CORE is
unaffected -- so a width or a peak still looks right -- while second-moment /
r^2-weighted / large-radius encircled-energy / centroid metrics read wildly
wrong".  That is true up to ~1.5 periods and FALSE past about two: a replica of
the core lands INSIDE the window there, and the battery cell is the measured
counter-example -- at 2.063 periods the peak of the best plane IS a replica, an
argmax-led width reads 20.50 um against an analytic 17.41 um, and the encircled
energy about it reads 49.5 % against 99.70 %.  The fixture's waiver cites
exactly that premise ("every metric it takes (FWHM, EE inside a few waists) is
confined to the core").

The message now states both regimes, gives the counter-example, and names
`replica_fill='zero'` as the way to keep the window without the replicas.  The
`m over` / `sample(s) per edge` wording, the `ALIASES` token and the
`REPLICAS` token that four existing tests match on are preserved verbatim.

### A25-4 -- how much of the window is measurement

`_check_readout_replica` computes `n_safe` (the largest faithful `N_out` at
this offset) only to print it inside a refusal a waiving caller never sees.
`_fill_readout_replicas` now publishes the same fact on BOTH fills as
`_period_out['faithful_samples']`, and `_publish_readout_containment` copies it
onto the chain's stage dict as `readout_faithful_samples`, beside
`readout_period` / `readout_containment` / `readout_window_energy`.  On the
battery cell it reads `(249, 249)` of 512, and 249 is asserted to be
`2*floor(period/2/dx_out)+1` of the REPORTED period rather than a stored
number.

---

## 3. Files touched

| file | what |
|---|---|
| `lumenairy/propagators/carrier.py` | `_REPLICA_FILLS` / `_check_replica_fill` / `_fill_readout_replicas` (`:3829-3926`); `replica_fill` on both public readouts, their validation and their call sites (`:2773`, `:3019`, `:3108-3109`, `:5238`, `:5511-5512`, `:5800-5806`); reachable from the chain's `focus_readout` and the multi entry point's `output_grid` (`:9531`, `:9805`); `readout_faithful_samples` publication (`:3513-3530`, `:3917-3918`); the replica refusal's two regimes and its message (`:3652-3676`, `:3769-3826`); `on_replica` / `replica_fill` / `on_focus_containment` / `Returns` / `_period_out` docs on both readouts (`:2855-2895`, `:2926-2960`, `:5383-5412`); the window-energy tripwire's remedy list (`:3110-3135`) |
| `docs/history/carrier.md` | fingerprints re-recorded (`scripts/record_history_fingerprints.py`, `re_recorded: 2026-09-13 -- WP-A25: ...`) |
| `tests/unit/test_audit2609_a25_carrier_focus_readout.py` | new, 17 tests |
| `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A25_REPORT.md` | this file |
| `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A25_CHANGELOG.md` | release text |

No other file was modified.

---

## 4. Why `replica_fill='zero'` is not the default

The obvious fix -- have the readouts stop returning replicas at all -- was
built, run against the suite, and backed out.  It is not a neutral change:
three shipped fixtures are written against the replicas being there, and two of
them are the demonstrations that justify the refusal itself.

Measured (the same branch with the blanking unconditional, everything else
identical; `-p no:randomly`, 1333 s):

```
9 failed, 348 passed
  test_fix_v1_v8_readout_guard_and_standoff.py::TestV3ChainScope::test_a_walking_chief_ray_gives_a_full_amplitude_ghost
  test_fix_v1_v8_readout_guard_and_standoff.py::TestV3ReplicaGuardSeesCentreOut::test_the_verifiers_own_ghost_case_is_refused
  test_niche_tight_focus_readout.py::test_the_refused_window_really_would_have_been_corrupt
  test_niche_d2_chain_multi.py::test_k1_keeps_the_requested_field_of_view
  test_niche_d2_chain_multi.py::test_default_refuses_the_periodic_replica_regime
  test_niche_d2_chain_multi.py::test_tiles_avoid_the_periodic_replica_regime
  test_niche_d8_congruence_workers.py::test_the_worker_clamp_and_the_grid_clamp_share_one_model
  test_niche_d8_congruence_workers.py::test_pool_uses_spawn_never_the_platform_default_fork
  test_niche_d8_congruence_workers.py::test_futures_are_drained_by_completion_not_submission_order
```

That experimental build also dropped the `REPLICAS` token from the refusal
message (it was restored for the shipped change, and four tests match on it),
so some of the nine are message-token failures rather than data failures.  The
three below were read assertion by assertion and are about the DATA; they are
the ones that make the default un-changeable from inside this work package:

* `test_a_walking_chief_ray_gives_a_full_amplitude_ghost` asserts that a window
  one whole period off the chief ray returns a peak bit-identical to the real
  spot's.  Blanking returns zeros there, so the ghost -- the thing the V3
  finding exists to demonstrate -- disappears.
* `test_the_refused_window_really_would_have_been_corrupt` asserts that with
  the refusal waived the wing metric is wrecked (`2 sigma` more than 5x the
  faithful reading) while the core survives.  Blanked, the wing metric becomes
  correct and the demonstration has nothing to show.
* `test_k1_keeps_the_requested_field_of_view` asserts, in its own words, that
  "the whole requested window is live -- nothing was zeroed away", and its
  docstring records that an earlier cut which "silently shrank the requested
  2.87 mm window to 1.29 mm and returned ZEROS over the rest" was a defect.

That is a deliberate, twice-decided contract (D3 2026-08-06 and V3), so this
work package does not overturn it from inside a single-fixture regression.
`replica_fill` gives the honest answer to any caller who wants it, per call,
and leaves every one of those fixtures bit-identical.

---

## 5. Requested changes outside my ownership

### R1 (required for the subject test to pass) -- `tests/unit/test_niche_p2_design_battery.py`, `_through_focus`

The helper asks for a window the readout cannot deliver and waives the
refusal.  The BAR is right -- the truth on this cell is 1.0624x and 0.9970,
inside `fwhm/fwhm_th < 1.10` and `ee[2] > 0.95` with room -- so nothing about
the assertions should change; what is wrong is the readout request.  One dict
key, at `tests/unit/test_niche_p2_design_battery.py:339-347`:

```python
    res, _, env0, dx = _run_chain(
        design, w0, ratio, guard, final_distance=-R,
        # A25 (2026-09-13): the requested window exceeds one Bluestein period
        # (2.063 of them on the 2.5x doublet cell, 1.267 and 1.544 on the
        # others), and ``n_out`` also sizes the scan's own transform, so the
        # window stays.  What changes is that the part of it the transform
        # cannot measure comes back ZERO instead of as periodic replicas of
        # the core: a replica is a full-amplitude image of the spot, and the
        # scan's ``argmax`` duly locked onto one -- 181.0 um from the window
        # centre, in its corner -- reading FWHM 20.50 um / EE2w 0.495 against
        # an analytic 17.41 um / 0.997.
        focus_readout=dict(dx_out=dx_out, N_out=n_out,
                           on_replica='ignore', replica_fill='zero'))
```

Verified two ways: in process by
`test_audit2609_a25_carrier_focus_readout.py` (which injects exactly this key
at the readout call and pins the resulting metrics against the analytic
Gaussian), and by re-running the whole battery file with the same injection --
**26 passed in 83.15 s**, against **1 failed, 25 passed** as it ships (the
subject test, 2.05e-05 against a 1.10 x 1.7413e-05 bar).  The other two
through-focus cells and the nine wavefront / cliff cells are unaffected either
way.

The three docstrings that quote measured numbers need no change: the subject
test's "Measured 2026-07-25: FWHM 18.5 um vs 17.4 um theory (1.062x), EE1w
86.0% / EE2w 99.7% / EE3w 99.8%" is exactly what the corrected request returns,
and the truncated cell's "FWHM 23.5 um ... EE2w 92.3%" is unchanged (measured
23.500 um / 0.9229).

### R2 (required for `test_niche_c1_consolidation.py` to pass) -- the focus-readout whitelist fixture

`test_the_focus_readout_whitelist_is_exactly_what_the_chain_consumes` asserts
`set(sample) == _FOCUS_READOUT_KEYS` against a hand-written `sample` dict, and
then runs a chain with it -- deliberately, so that every accepted key is proved
to pass validation rather than merely be listed.  Adding `replica_fill` to
`_OUTPUT_GRID_PASSTHROUGH` (which is what makes it reachable from
`focus_readout` and `output_grid`) therefore requires the fixture to carry it.
The same edit WP-A6 made for `on_focus_containment`, at
`tests/unit/test_niche_c1_consolidation.py:1070`:

```python
              # WP-A6 / C1 (2026-09-12): the paraxial readout's beam-vs-grid
              # containment guard, likewise -- and likewise 'ignore', because
              # this fixture is the whitelist's fixture and not the guard's.
              'on_focus_containment': 'ignore',
              # A25 (2026-09-13): what the readout writes outside one Bluestein
              # period once ``on_replica`` has let the window through.  'zero'
              # here rather than the default, so the whitelist fixture proves
              # the non-default value survives validation and the forwarding.
              'replica_fill': 'zero'}
```

Measured: this is the ONLY failure that the whole change produces across the
replica-dependent suites (section 6), and the key is exercised independently by
`test_audit2609_a25_carrier_focus_readout.py::test_replica_fill_reaches_the_readout_through_the_chain`,
which drives the same chain entry point with `replica_fill` in the
`focus_readout` dict and asserts the returned field differs exactly outside one
period and nowhere inside it.

### R3 (optional) -- none

No other file carries prose about the replica contract that this change makes
stale (`CHANGELOG.md`'s entries are historical release text and describe the
releases they shipped in).

---

## 6. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one
process at a time, on this branch with the change in.

| what | result | time |
|---|---|---|
| `test_audit2609_a25_carrier_focus_readout.py` (new) | **17 passed** | 17.6 s |
| `test_audit2609_a6_carrier.py` + `test_audit2609_a6_verify_carrier.py` + `test_audit2609_a24_decentre_calibration.py` + `test_niche_p2_design_battery.py` | **197 passed, 1 failed** -- the 165 WP-A6/VERIFY-A6 pins and the 7 WP-A24 pins are green; the one failure is the subject test, still red as the fixture ships (2.05e-05 against its 1.10 x 1.7413e-05 bar) and green with the R1 edit | 144.7 s |
| `test_niche_p2_design_battery.py` with the R1 edit injected in process | **26 passed** | 83.2 s |
| `test_fix_v1_v8_readout_guard_and_standoff.py`, `test_niche_tight_focus_readout.py`, `test_niche_d1_tilted_carrier.py`, `test_niche_d2_chain_multi.py`, `test_niche_d3_guards.py`, `test_niche_d4_dgrating.py`, `test_niche_d5_dx_flatness_gate.py`, `test_niche_d6_exact_tilted_leg.py` (the 38 d6 pins), `test_niche_d8_congruence_workers.py`, `test_niche_c1_consolidation.py`, `test_niche_r9_highna_final_leg.py`, `test_pipeline.py`, `test_audit2609_a17_history_relocation.py`, `test_audit2609_a25_...py` | **1156 passed, 5 failed** -- see the note below; exactly ONE is mine (`test_the_focus_readout_whitelist_is_exactly_what_the_chain_consumes`, the R2 edit) | 1474 s |
| `test_niche_d1_tilted_carrier.py` alone, re-run | **33 passed** | 73.6 s |
| `validation/run_all.py` | **ALL 37 files passed** | 224 s |

**Note on the other four failures in the long run.**  A concurrent work
package was rewriting `lumenairy/elements/_lens_traced.py` (+112 / -49 lines)
while that session was running.  Two of the failures are its history
fingerprints (`test_audit2609_a17_history_relocation.py::...[lumenairy.elements._lens_traced]`
-- `carrier` passes, so my own re-record is correct) and two are
`test_niche_d1_tilted_carrier.py` cells that run through that file; d1 re-run
in isolation immediately afterwards is **33 passed**.  None of the four touches
anything this work package changed.

**Blast-radius measurement.**  Before settling on the opt-in design I ran the
same suites with the blanking unconditional; that is the 9-failure run quoted
in section 4 (1333 s).  It is the evidence for `'repeat'` being the default.

---

## 7. Deferred

* **The exact readout's `replica_fill` is wired and validated but not
  exercised by a fixture of its own beyond the vocabulary check.**  The A25
  measurement is entirely on the paraxial leg (WP-A24 established that
  `final_leg='exact'` never enters `_default_focus_standoff`), and the exact
  readout's period is the fine crop window rather than a function of a
  standoff, so its window/period ratios are set by `window_factor` and are
  1.707 on the one shipped fixture that waives the guard.  A cell that
  measures a focus through the exact leg on a >1-period window would be the
  right place to pin it; that needs a fixture this work package did not build.
* **Whether the default should eventually become `'zero'`.**  Section 4 is the
  measured cost today: three fixtures whose subject IS the replicas, plus six
  more in the same run that a message-token change accounted for.  If the
  library decides that a readout should never return samples it did not
  measure, those three demonstrations need restating first -- and each of them
  can be restated against an explicit `replica_fill='repeat'`, which is
  precisely why the knob is a per-call choice and not a global one.
* **`propagate_traced_carrier_chain_multi`'s `readout_tile='auto'` and
  `replica_fill` overlap.**  `'auto'` avoids the regime by SIZING the tile;
  `replica_fill='zero'` keeps the caller's grid and blanks what it cannot
  fill.  Both are now reachable from `output_grid`; no fixture combines them,
  and the interaction (a tile that fits, plus a fill that then does nothing) is
  a no-op by construction but unpinned.

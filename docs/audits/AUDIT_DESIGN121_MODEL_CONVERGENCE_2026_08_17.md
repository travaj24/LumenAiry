# AUDIT -- design 121: four models, one design, and where they converge

Campaign audit of the Tx02-MSOP16 (design 121) relay + 4x8 Dammann fan across
the four modelling routes the library now offers, run 2026-08-11 .. 2026-08-17
on the Tailscale mesh.  The question the campaign answers is not "what does
design 121 do" but "**which of the numbers we have been quoting are the
design, and which are the model**".

The short answer: the per-frame power non-uniformity this project has reported
since exp06 -- **13 to 18 % down on the outer fan orders** -- is a MODEL
ARTEFACT of the vertex-plane thin screen.  It falls monotonically to 2.4 % as
the screen model improves and to ~0.3 % on the traced route.  It should not be
carried into a design decision.

Binding law throughout: `docs/TESTING_STANDARDS.md`.  Every number below is
measured on the runs named beside it; nothing is scaled or inferred except
where explicitly labelled as such.

## 0. Builds and boxes

| tag | box | spec | role |
|---|---|---|---|
| **A** | tesla-ryzen | Win 11, py 3.14.6, 137.4 GB | analytic exp27/exp28, diagnosis |
| **B** | austinoffice-2 | Win 11, py 3.14.6, 136.6 GB | traced exp29/exp30, analytic exp31 |

Library versions span 5.34.0 -> 5.38.1 across the campaign; each run records
its own in the table of S2.  Box B dropped off the mesh once mid-campaign
(2026-08-16, ~1 h); the staged pipeline's checkpoints made that cost ~25 min of
recompute out of a 7 h stage, which is recorded in S5.4 because it is the
strongest available argument for the staged driver.

---

## 1. The headline -- model convergence

Four independent runs of the SAME design and the SAME 8x8 source, differing
only in how the lens groups are modelled:

| run | route | focusing-group model error | per-frame uniformity | **outer/inner** |
|---|---|---|---|---|
| exp27 | `thin`, angle-blind | **~170 waves rms** | 16.80 % | **0.824** |
| exp28 | `thin` + `carrier=` | ~17 waves residual | 13.32 % | **0.872** |
| exp31 | `tangent_facet` | **0.0032 waves** | 4.32 % | **0.9759** |
| exp30 | traced staged pipeline | not screen-limited | 0.23 % | **~1.00** |

Five orders of magnitude of model error, and the "outer-order deficit"
tracks it monotonically to zero.  The deficit is the model.

### 1.1 Why the earlier 13-18 % reading survived its own control

The 13-18 % figure was measured correctly.  It was integrated power in
480 um cells on the MSoP field, and it was cross-checked against the
hypothesis "the spots are aberrated and spilling outside their cell" by
WIDENING the integration cell to 1.25x and 1.5x pitch.  The outer/inner ratio
did not recover (0.824 -> 0.794 -> 0.766), which was read as "the energy never
arrived, therefore vignetting".

That inference is sound and still wrong.  It tested spillover-vs-loss WITHIN
one model's output; it could not test whether that model had propagated the
off-axis orders correctly in the first place.  No re-integration of a wrong
field reveals that the field is wrong.  It took a better model, not a better
metric.

**Rule extracted:** a metric-side control (re-integrate, re-window, re-bin)
can only falsify metric-side explanations.  A claim about the OPTICS needs a
second MODEL, not a second integral.

---

## 2. The runs

| exp | route | N | dx | lib | conservation | outer/inner | where |
|---|---|---|---|---|---|---|---|
| exp27 | analytic `thin` | 32768 | 0.903 um | 5.34.0 | 0.96489 | 0.824 | A |
| exp28 | analytic `thin`+carrier | 16384 | 1.806 um | 5.35.0 | 0.94205 | 0.872 | A |
| exp31 | analytic `tangent_facet` | 16384 | 1.806 um | 5.38.0 | 0.96200 | 0.9759 | B |
| exp29 | traced pipeline, 1 src | -- | dx_out 0.2 um | 5.35.1 | P/P_exit 1.000000000 | ~1.00 | B |
| exp30 | traced pipeline, 8x8 | -- | dx_out 0.2 um | 5.35.1 | P/P_exit 1.000000000 | 0.9969 | B |

All analytic runs: 8x8 emitters, 35 um pitch, MFD 8 um, DOE on, `complex64`
field, aperture-containing 29.58 mm extent.

Note exp31 vs exp28: **same grid, +2.0 points of power conservation** from the
screen model alone (0.94205 -> 0.96200), and it nearly matches exp27 at half
the grid.

---

## 3. The screen-model error ladder

### 3.1 Angle-blindness of the vertex-plane screen (5.35.0 guard)

`apply_real_lens` with `carrier=` reports how angle-blind the shipped
`(n2-n1)*sag` screen is on the supplied prescription.  Measured on design 121
at two resolutions:

| group | N=8192 | N=16384 | change | residual after correction |
|---|---|---|---|---|
| S3-4 | 16.958 | 22.393 | +32 % | 2.24 |
| S5-7 | below 0.05 tol | below tol | -- | -- |
| S14-15 | below tol | below tol | -- | -- |
| S16-17 | below tol | below tol | -- | -- |
| S18-20 | 3.071 | 4.673 | +52 % | 0.47 |
| S21-22 | 1.285 | 2.006 | +56 % | 0.20 |
| S23-24 | 2.910 | 3.120 | +7 % | 0.31 |
| **S25-27** | **158.599** | **168.637** | **+6 %** | **16.86** |

waves rms, piston/tilt-free.  Two readings matter:

1. **The error is concentrated in the final focusing group**, the one that
   forms every spot at the MSoP.  S25-27 alone is ten times the rest combined.
2. **The dominant term is the most converged** (+6 % between resolutions) while
   the small terms are still moving (+32..56 %), so ~170 waves is a real value
   and not a sampling artefact.

Every analytic run in this project before 5.35.0 -- exp06, exp12, exp14, exp27
-- ran that screen with NO carrier, i.e. assuming normal incidence, i.e.
carrying the full ~170 waves.

### 3.2 What the carrier correction can and cannot carry

The 5.35.0 closed-form correction removes ~90 % of it and states the remainder
explicitly:

> the closed-form correction is applied (the sag-obliquity term AND the R1
> drift term), but its own next-order residual (**the DEFLECTION channel
> proper, which is not the gradient of any scalar and so no screen can carry**)
> is budgeted at 10.0 % of that, leaving ~16.86 waves.

That "no screen can carry" is scoped to a VERTEX-PLANE screen.  It is easy to
read as a bound on analytic modelling generally; it is not, and the
tangent-facet family carries the term by moving the screen to the facet.  See
S9.2 trap 1.

### 3.3 `tangent_facet` (5.36.0, banded 5.38.0)

Library-measured on design 121 group 5: **0.0032381 waves rms** against exact
rays, where the vertex-plane screen leaves the transverse walk (up to 55 um at
one face across a 3 mm pupil) unrepresentable.  The REMAP rung
(`tangent_facet_remap`) reads **2.56e-08** but costs ~6 float64 grids more and
~9x the wall clock; at 0.0032 waves route 3 is already ~5000x below the error
that mattered, so the remap rung was not used.

Consumer requirements found by running it: `slant_correction=True` and
`screen_obliquity=True` are REFUSED as double-counts, so the runner must turn
slant off with the route.  Wired as `POC_SURFACE_MODEL` with an automatic
slant disable.

---

## 4. The traced staged pipeline

`validation/pipeline` (decompose -> chains -> aggregate -> leg -> readout) was
used for exp29/exp30.  Findings that are properties of the ARCHITECTURE rather
than of this design:

### 4.1 It is the only route that is not screen-limited

Per-frame, 32 orders, single source (exp29):

```
EE3   89.829 - 90.631 %   mean 90.503     (skew-ray/Debye oracle 90.7)
FWHM  3.400 - 3.800 um    mean 3.537      (re-baselined acceptance 3.450)
P/P_ref worst deviation   4.084e-05       (campaign bar 4e-05)
piston worst              2.257e-05       (lambda/100 admissibility)
```

### 4.2 The chains conserve energy exactly

`P/P_exit = 1.000000000` per chain, against the +-10-28 % per-group wander
measured on the exp26 adaptive-regrid fields (S8.1).

### 4.3 The common-carrier precondition holds, measured

`aggregate` requires every beam to share one exit sphere.  All 32 orders AND
all 64 emitters returned

```
R_doe = 703642.7361 mm
```

identical to the digit, across the single-source run and all three 8x8 chain-A
rebuilds.  A runtime assertion was added to the 8x8 decomposer to refuse a sum
if any emitter disagreed; it never fired.

### 4.4 The staged checkpoints paid for themselves

Box B dropped off the mesh at ~30 of 32 chains.  On return, `--from chains`
resumed all 30; the outage cost ~25 min of chain-A rebuild out of a ~7 h
stage.  A monolithic runner would have lost the lot.

---

## 5. The 8x8 multi-source question

`apply_real_lens_traced` propagates ONE congruence; the library gates
multi-congruence inputs at `_NONCOLLIMATED_RESID_THRESH = 0.02 rad`, a gate
that exists because pushing the 32-order fan through one chain once produced
"a populated, credible-looking frame lattice whose per-frame power was
scrambled ... with nothing raised and nothing warned".

Measured on design 121's 8x8 array:

| plane | emitter angular spread | vs 0.02 rad gate |
|---|---|---|
| launch (`Z1 = 2 mm`) | **0.0634 rad** | 3.2x OVER -- refused |
| DOE plane (`R_doe = 703.6 m`) | **7.4e-06 rad** | 2700x UNDER -- admissible |

The relay magnifies emitter SEPARATION ~30x (122.5 um -> 3663.9 um at the DOE)
while collapsing their ANGULAR spread, because the beam is near-collimated
there.  So the array is 64 separate congruences at launch and ONE congruence at
the DOE.

**Consequence for cost:** chain A must run per-emitter (64 runs, 21.7 s each,
~23 min) but the 32 order-chains do not multiply.  That is what makes the 8x8
a ~12 h job instead of 64 x 32 = 2048 chains.

### 5.1 What the 8x8 costs at the sampling gates

The array widens the beam, and every downstream gate notices:

| gate | 1 source | 8x8 |
|---|---|---|
| field angular half-band | 0.106 rad | **0.391 rad** (3.7x) |
| `aggregate` binding pitch | 1.1082 um | **0.4989 um** |
| `aggregate` grid | 12288 | **24576** |
| chain fine-retrace grid | 8192 | **16384** |
| peak working set | 37.25 GB | **122.89 GB** |

Both grid doublings trace to the same fact: at the last group the 8x8's
entrance beam radius is 3.8749 mm against a smaller single-source beam, and
Nyquist on an NA=0.5024 exit sphere across the resulting 15.4764 mm window
needs 16384.  Each doubling is 4x memory and ~2.7x time.

### 5.2 exp30 result

```
in-spot fraction (64 intended spots)  98.167 - 99.420 %   mean 98.837
outer columns (m = -4, +3)            98.646 %
inner columns                         98.955 %
OUTER/INNER                           0.9969
in-spot : background contrast         76:1 (outer) .. 113:1 (best inner)
peak : background                     1.6-2.5e4  ->  42-44 dB
```

2048 spots (64 emitters x 32 orders), 98.8 % of the light in the intended
spots, outer orders delivering 99.69 % of what the inner ones do.

---

## 6. Metrics that do not transfer to the array

`EE3` (encircled energy within r < 3 um of the FRAME CENTRE) reads **1.385 %**
on the 8x8 frames.  That is not a defect: with 64 emitters spread over 167.6 um
the metric captures about one emitter out of 64, and 1/64 = 1.56 %.

**Any acceptance criterion written against the single-source runs must be
restated for the array.**  The replacement used here is the in-spot fraction of
S5.2 (energy within r < 6 um of each of the 64 design positions), which is also
the crosstalk-relevant quantity for a comms link.

The `FWHM` panel in the standard figure set is quantised to 3.40/3.80 um
because `dx_out = 0.2 um` makes the estimator step in 2-pixel increments; the
trend across the fan is real, the two-level appearance is sampling.

---

## 7. Memory -- what is priced, what is not, and what is still available

### 7.1 Why N=32768 analytic does not fit, and what would make it fit

`_preflight_memory_check` refuses `tangent_facet` + carrier at N=32768:
`need ~202.5 GB, have 121.5 GB free`.  Its docstring calls the overlap between
the `screen_obliquity` and `tangent_facet` terms deliberate conservatism.

5.38.0 gates the screen-obliquity BLOCK off at the source under the
tangent-facet family ("dead work ... into an accumulator no surviving reader
touches"), which was read as making the preflight term a phantom.  The term was
dropped under a version gate; the preflight then passed at 115.6 GB against
120.5 free, and **the FIRST lens group alone consumed ~102 GB** (free
106.8 -> 8 GB) and had to be killed.

**The failure is one of MARGIN, not of pricing.**  Verified at 5.38.1:

```python
_obl_active = (carrier is not None
               and surface_model not in _TANGENT_FACET_MODELS)
```
> v5.37: the tangent-facet family bands here too.  ``_obl_active`` is False
> under those models (they supersede equations 4 and 7), so the two ``_here``
> flags are mutually exclusive by construction.

- the obliquity block IS gated off under the family, so the term IS a phantom;
- the tangent-facet family DOES band (`_tf_here` / `_tf_halo_rows`);
- design 121 has **0 of 19** surfaces outside the chunk-eligible case
  (checked for decenter / tilt / form_error / radius_y / freeform /
  clear_aperture / stop).

So the term prices nothing real, and removing it makes the estimate ACCURATE.
The run dies anyway:

| quantity | value |
|---|---|
| corrected preflight predicted peak | 100.5 GB |
| observed `run_poc` RSS | ~104 GB |
| `need` after the 1.15 safety factor | 115.6 GB |
| `free` reported at launch | 120.5 GB |
| system usage when the watchdog fired | ~129 GB of 136.6 |

**The corrected preflight was ACCURATE about the process** (100.5 predicted vs
~104 observed).  What it could not cover is that ~25 GB of the box was already
committed elsewhere, so a `need` of 115.6 GB against a `free` of 120.5 GB left
about 5 GB of real headroom and the run walked through it.

The term was not modelling an allocation; it was supplying headroom.  The
margin belongs in a free-RAM floor, not in a phantom physics term that happens
to be the right size -- but until that floor exists, **the phantom is the only
thing holding the run inside the box**, and it is left in place with a comment
at the term saying so.

This is the blocker for S11 item 1: `sag_dtype=np.float32` would halve the
geometry stack to ~50 GB, which is comfortably inside the box, but the
preflight has no `sag_dtype` term and will still refuse at 202 GB.  Fixing the
pricing and fixing the margin are the same change and must land together.

### 7.2 The watchdog

The run was launched behind a free-RAM watchdog (kill under 10 GB) precisely
because this box's failure mode under memory pressure is SILENT process death,
not `MemoryError`.  It fired and produced a diagnosis instead of a mystery --
but it killed the wrapper while the allocating process survived at 103.8 GB and
needed a targeted `taskkill /F /PID`.  **Harden before reuse: kill by PID of the
allocating process, verify, then exit.**

### 7.3 `sag_dtype=np.float32` -- accuracy-safe, and worth nothing here

`lens_sag_float32_opd_error` per group on design 121 at the production
sampling (`field_check_n=512`, `field_check_dx=0.90 um`):

| group | max OPD error | max OPD | field rel error | ok |
|---|---|---|---|---|
| S3-4 | 1.907e-04 waves | 0.250 nm | 1.338e-06 | True |
| S5-7 | 1.585e-04 | 0.208 nm | 7.626e-07 | True |
| S14-15 | 0 | 0 | 0 | True |
| S16-17 | 0 | 0 | 0 | True |
| S18-20 | 1.886e-04 | 0.247 nm | 3.762e-07 | True |
| S21-22 | 2.796e-04 | 0.366 nm | 4.188e-07 | True |
| S23-24 | 2.130e-04 | 0.279 nm | 6.064e-07 | True |
| **S25-27** | **7.738e-04** | **1.014 nm** | **1.226e-06** | True |

Worst field relative error 1.338e-06 against the 1e-3 bar; worst OPD error
7.74e-04 waves, below `tangent_facet`'s own 0.0032-wave residual.  **Accuracy
is not the obstacle.**  Two qualifications on that margin: the check runs at
`field_check_n=512`, i.e. ~2 % of a 32768 pupil, and it is per-prescription --
do not carry it to another design.

**MEASURED 2026-08-17: the credit is 0.001 grids, and the lever is dead at
this N.**  The tangent-facet + carrier route reads 14.002 grids at N=4096 and
14.001 at N=8192 in BOTH dtypes.  float32 is worth ~3.5 grids only BELOW the
`N >= 4096` auto-band threshold: once the route bands, the full-grid geometry
is never materialised, so halving its dtype halves nothing.  The knob is
wired and priced; set it only in the whole-grid regime.

### 7.4 `sag_chunk_rows` -- also flat, also dead here

The preflight prices this route off a BINARY `_banded = N >= 4096` flag
(7.7 grids with a carrier), whereas the slant term beside it scales with
`_band_rows / N`.  That asymmetry looked like an unclaimed credit.  It is not.

Warmed `tracemalloc` on S25-27, extras over the paraxial no-carrier call at
the same N, in float64 grids of `8*N*N`:

```
N = 4096  (AUTO = 256 rows)        N = 8192  (AUTO = 512 rows)
  whole-grid (0)   19.74             whole-grid (0)   19.74
  256 (AUTO)       11.23             512 (AUTO)       11.23
  512             11.23              256              11.23
  128             11.23              128              11.23
```

**Flat from 512 rows down to 128 -- a 4x band reduction changes nothing.**
Banding is binary exactly as the preflight models it: the ~8.5-grid credit is
for banding at all (19.74 -> 11.23), and there is none for banding harder.
That follows from the halo being a FIXED 3 rows of sag and 2 of the
accumulator per band rather than a proportional cost.  The extras are also
identical at N=4096 and N=8192, confirming the "flat in N at and above 4096"
claim the ANCHOR makes.

**Side finding, worth the anchor owner's attention:** the measurement reads
**11.23 grids against the preflight's 7.7** -- the term UNDER-prices by ~3.5
grids on this prescription.  That is the wrong direction for a preflight and
is further reason the free-RAM floor of S7.1 is load-bearing rather than
belt-and-braces.  (The ANCHOR was taken on a biconvex singlet; design 121's
S25-27 is a three-surface group.)

### 7.5 Conclusion -- N=32768 analytic is out of reach on this hardware

Full probe, with the per-lever measurements and method:
`PROBE_D121_ANALYTIC_32K_FOOTPRINT_2026_08_17.md`.

Every lever is now measured rather than argued:

| lever | outcome |
|---|---|
| drop the phantom obliquity term | DONE (5.39.0 remediation); still 5.5 GB short of the floor |
| `sag_dtype=np.float32` | 0.001 grids (S7.3) |
| `sag_chunk_rows` | flat across a 4x range (S7.4) |
| a bigger box | nothing on the mesh exceeds 136.6 GB |
| drop `carrier=` | 7.7 -> 4.1 grids, would fit -- but changes the model and breaks the like-for-like against exp31 |

The run needs **135.6 GB free on a 136.6 GB box** (`need` 115.6 + a 20 GB
floor) against 121.1 GB actually free.  **exp31 at N=16384 is the analytic
result of record**, and the residual 2.4 % of S1 stays unattributed between
model and grid as a HARDWARE limit, not an open task.

---

## 8. Prior-art correction -- what exp26 was actually showing

The campaign opened from an adaptive-regrid traced fan (exp26, RN=1024) whose
saved intermediate planes showed visible cross-hatch and a square support.
Measured on the stored complex envelopes:

- dominant periodicity **3.99-4.21 px = `ray_subsample` exactly**; physical ray
  pitch 4 x 51.23 = **204.9 um** against the ~4 um validated for this design
- spectral occupancy beyond 0.8 Nyquist: **1e-07 (planes 00-02) -> ~0.50**
- physical power (dx^2 weighted) wandering **+-10 to 28 % per group**,
  non-monotonic

The library's own pipeline docstring reaches the same conclusion
independently: the coarse co-moving exit vertex is **7.8x under-sampled on its
own exit sphere (design 121: dx 33.2 um against 4.26 um needed)**, which is why
the staged driver's only legal summable plane is the fine-retrace exit.

Root cause was the parent grid, reduced to RN=1024 under memory pressure --
where the parent grid cost 0.4 GB and the per-congruence exact readout cost
103 GB.  **99.6 % of the memory was in the final leg; the knob that was cut was
the one that cost nothing and controlled all the sampling.**

---

## 9. Defects, and the traps around them

### 9.1 Library defects found

| # | where | defect | status |
|---|---|---|---|
| 1 | `_lens_real._check_screen_obliquity_support` | `screen_obliquity` validated with `is` against `('auto', True, False)`; a runtime-built `'auto'` (os.environ, config, f-string) is not the interned literal and was REFUSED with a message naming it valid.  Tests only passed literals, which are interned, so it was invisible. | **FIXED**, committed `cbef685` |
| 2 | `validation/pipeline/specs/d121_32order.json` | shipped reference spec cannot complete on the shipped library: `dx_common = 1.2292 um` against the band-aware guard's 1.1082 um requirement.  Guard tightened in `0f46efb`; spec last touched at `087d151`, before it. | OPEN |
| 3 | runner `_preflight_memory_check` | (5.35-era) no term for the obliquity arrays -> silent OOM at N=32768 instead of a refusal | FIXED upstream; the run now refuses cleanly |

### 9.2 Three traps, for anyone implementing the S11 items

Each of these is a reading the source or the release notes actively invite,
and each is wrong in a way that costs a run.  They are here because an
implementer touching S11 will meet all three.

**Trap 1 -- "no screen can carry the deflection residual", so analytic is
finished on this design.**  The 5.35.0 guard says the deflection channel "is
not the gradient of any scalar and so no screen can carry" it.  That is true of
a screen imprinted on the surface VERTEX PLANE.  The tangent-facet family
(5.36.0) carries it by moving the screen to the facet the pixel's ray actually
meets, and reads 0.0032381 waves rms on design 121 group 5 where the corrected
vertex screen leaves ~16.9.  Do not read the 5.35.0 sentence as a bound on
analytic modelling.

**Trap 2 -- the multi-congruence gate is evaluated at the LAUNCH plane, not at
the object.**  Design 121's emitters sit 35 um apart at an object 47.9 mm from
the first lens, which suggests a ~5 mrad chief-ray spread and a comfortable
pass against the 0.02 rad gate.  The launch plane is `Z1 = 2 mm`, so the
spread against the reference sphere there is `122.5 um / 2 mm = 0.0634 rad`
-- 3.2x OVER.  Measure the spread on the plane the chain actually starts from.
The same array reads 7.4e-06 rad at the DOE plane (S5), so the answer changes
by four orders of magnitude depending on which plane is asked.

**Trap 3 -- the `screen_obliquity` preflight term IS a phantom under the
tangent-facet family, and removing it still breaks the run.**  `_obl_active`
excludes the family outright, so the block never executes and the term prices
nothing.  Removing it is therefore "correct" and makes the estimate accurate
(100.5 GB predicted vs ~104 GB observed).  The run still dies, because that
phantom was the only headroom: `need` 115.6 GB against `free` 120.5 GB leaves
~5 GB while the box already commits ~25 GB elsewhere.  **Do not remove the
term without adding the free-RAM floor of S11 item 2 in the same change.**

Two claims that are NOT traps, only stated here because they appear in
superseded comments elsewhere in the tree: design 121 has **0 of 19**
chunk-ineligible surfaces (the family bands normally here), and
`slant_correction` does not disable chunking (`_slant_narrow_chunk`, v5.17,
exists for exactly that case).

---

## 10. Guards that earned their keep

Three configurations were refused during this campaign that would each have
produced a populated, credible-looking, WRONG field:

1. `aggregate` at `dx_common = 1.2292 um` (1 source) and `1.0 um` (8x8) --
   band-aware Nyquist.  The refusal states that without the envelope's own
   half-band folded in, "the ramp bound alone would read 2.3756 um and this
   call would have been accepted with an aliased answer".
2. `_fine_trace_group_exit` at `n_fine_cap = 8192` for the 8x8 -- would have
   "SILENTLY DISCARD every spatial frequency above NA=0.3467".
3. the multi-congruence gate at the launch plane (S5).

Given that this campaign began by diagnosing exactly that failure mode in
exp26, the guards are working as designed and should not be relaxed to make a
run fit.

---

## 11. Open items

**Re-verified against lumenairy 5.39.1 on 2026-08-17.**  The 5.39.0
remediation (`fix/d121-audit-items`) landed the free-RAM floor and the
version-gated phantom-term drop; items closed by it, or closed by measurement
since, are recorded as CLOSED rather than deleted, because each one is a lever
someone will otherwise reach for again.

1. **`d121_32order.json`** -- `dx_common = 1.2292e-06` against the band-aware
   guard's 1.1082 um requirement.  Untouched since `087d151`, which predates
   the guard (`0f46efb`).  The shipped reference spec still cannot complete on
   the shipped library.  **OPEN, one line.**
2. **The tangent-facet preflight ANCHOR under-prices by ~3.5 grids** on this
   prescription (7.7 priced, 11.23 measured -- S7.4).  The anchor was taken on
   a biconvex singlet.  **OPEN**, and the reason the S7.1 floor is
   load-bearing rather than belt-and-braces.
3. **Restate the array acceptance criteria** (S6).  `EE3` reads 1.385 % on the
   8x8 frames and is behaving correctly; anything written against the
   single-source runs needs the in-spot fraction instead.  **OPEN.**
4. **Watchdog hardening** (S7.2) -- kill by PID of the ALLOCATING process and
   verify; the current one killed the wrapper while the allocation survived at
   103.8 GB.  **OPEN** if the watchdog is reused.
5. **The RCWA DOE table** (5.35.0, opt-in) flags the two outermost fan lines
   as under-delivering in the rigorous solve.  Not exercised here; a DIFFERENT
   mechanism from the screen artefact this audit closes, and it would stack
   with it.  **OPEN, unexercised.**

### Closed by the 5.39.0 remediation

- **A headroom policy for `_preflight_memory_check`** -- landed as
  `FREE_RAM_FLOOR_BYTES = 20.0e9`, an absolute reserve checked as
  `free - need > floor`, derived from the S7.1 failure into two measured
  terms (baseline-commitment drift +8.9 GB, operational reserve 10.0 GB).
  The phantom obliquity term is now dropped behind a `>= 5.37.0` version gate,
  so the estimate is honest AND the margin is explicit.  This is the right
  shape: `safety_factor` multiplies `need` (estimate error), the floor is
  absolute (baseline commitment, which does not scale with N).

### Closed by measurement -- do not reach for these again

- **`sag_dtype=np.float32`** -- accuracy-safe on this design (747x margin, with
  the ~2 %-of-pupil caveat) but worth **0.001 grids** at N >= 4096, because the
  banded route never materialises the full-grid geometry.  S7.3.
- **`sag_chunk_rows` on the analytic path** -- peak is **flat** from 512 rows
  down to 128; banding is binary, and the halo is a fixed 3+2 rows per band
  rather than a proportional cost.  S7.4.
- **N=32768 analytic** -- needs 135.6 GB free on a 136.6 GB box.  Out of reach
  on this hardware with every lever exhausted (S7.5).  The residual 2.4 % of
  S1 is therefore a hardware limit, not a task.
  **[SUPERSEDED 2026-08-22]** "every lever exhausted" held at the 5.39.1
  re-verify above; the 5.40.0 wave found the levers this audit did not price
  (the carrier='auto' setup fit, -9.2 grids; memmap accumulator spill;
  streamed transfer function) and brought need to **81.7 GB -- ADMITTED**.
  See `BUILD_LENS_32K_MEMORY_2026_08_22.md`.

## 12. Deliverables

All under `Reverse_Symmetric_ASM/output_tx_design/design121/`:

| dir | what |
|---|---|
| `exp27_..._v5.34_angleblind/` | analytic thin, N=32768, 8x8 (the archived pre-carrier baseline) |
| `exp28_..._N16384/` (unprefixed at time of writing) | analytic thin + carrier |
| `exp29_121_traced_pipeline_32order_v5.35.1_nfc8192_rs1_legfull/` | traced, 32 orders, 1 source + standard figure set |
| `exp30_121_traced_pipeline_32order_8x8_v5.35.1_nfc16384_rs1_legfull/` | traced, 32 orders, FULL 8x8 + standard figure set |
| `exp31_121_analytic_tangentfacet_8x8_v5.38.0_N16384_dx1.81um/` | analytic tangent-facet, 8x8 + standard figure set |

Bulk stores (traced chains/aggregate ~6.7 GB, analytic zarr 42 GB) remain on
box B under `C:\tmp\`; only figures, metrics and manifests were synced.

Author: campaign audit, 2026-08-17.

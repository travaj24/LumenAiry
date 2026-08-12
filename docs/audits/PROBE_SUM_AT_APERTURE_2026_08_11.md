# PROBE -- sum-at-aperture aggregation for the design-121 traced fan

**2026-08-11.  Branch `main` @ `755ad99` (v5.34.0), checked out and NOT
modified.  No file under `lumenairy/**` was touched; no git command was run.
Everything added by this probe is under
`validation/repro_traced_carrier_121/` plus this note.**

The architecture under test, in the user's words: *per-order chains to the
BACK APERTURE of the last group, re-reference all orders onto ONE COMMON
ANALYTIC CARRIER (the mean converging sphere to the central frame), sum the
envelopes on one common grid, run ONE exact final leg on the summed field,
read each frame with its own Bluestein zoom -- and this equals the shipped
per-order path by linearity, at ~1/32 the leg cost.*

---

## 0. VERDICT

> **The physics half of the hypothesis is CONFIRMED and the cost half is
> REFUTED.**
>
> The null control -- one order through the whole aggregation path -- returns
> the shipped answer to **2.8e-05 field relative L2**, **1e-07 on energy**
> (bar 4e-05), **0.000 EE points** (bar 0.1), FWHM identical, and **zero
> piston**.  The 3-order sum holds every bar as well: worst energy delta
> **1.6e-05**, worst EE3 delta **0.004 points**, FWHM identical on all three
> frames, per-frame piston below **2.3e-06 rad** with an inter-frame spread
> of **2.3e-06 rad**, and exact linearity (**2.3e-16**).  Re-reference,
> resample and summation are all sound, and the architecture measures a term
> the shipped path structurally cannot: inter-frame crosstalk (worst
> **9.1e-08** of a frame's own power).
>
> But the leg the architecture shares is **11.6 % of the per-order cost**,
> not 100 % of it.  Measured end to end, **arm B is 1.17x-1.21x SLOWER than
> the shipped path**, and the 32-order projection keeps that sign.  Even with
> the resample eliminated, a shared-leg entry point added, and the
> aggregation made free -- an upper bound that cannot be reached -- the
> ceiling is **1.10x faster**.
>
> **GO for the physics, NO-GO for the architecture as a performance change**,
> with three named blockers in section 9.

The one-line reason the cost claim fails: the expensive part of an order's
exact final leg is the **fine RE-TRACE of the last group** (76.4 % of the
order's wall), and that is *upstream* of any plane where two orders may be
added -- `apply_real_lens_traced`'s entrance->exit map is single-congruence by
construction, which is the very reason the per-order fan architecture exists.
What is left to share is the **readout** (11.6 %), and even that does not
become free: 19 % of it is an irreducibly per-frame Bluestein zoom, and
building the shared field costs more per order than the whole readout does.

---

## 1. BOX, BUILD, CONFIGURATION

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X, 24 logical CPUs
137.4 GB physical RAM            111 GB free at launch
python 3.14.6   numpy 2.4.4      lumenairy 5.34.0 (working tree = main, clean)
```

Both arms run the shipped fan's own configuration, at the `n_fine_cap` the
protocol names for affordability:

```
RN=1024  RS=4  NW=1 (Newton pool serial -- CAPSTONE_D121 sec 5)
lumenairy_source_sha256 = f68ac2e884ece6f9...  (see 1.2 -- it moved mid-probe)
DXO=0.2 um   TILE=1024   LEG='auto' (resolves EXACT at na_exit 0.406)
NFC=8192     WF=4.0      output_grid ram_budget=inf (no clamp may bind)
orders (0,0) / (-2,0) / (-4,-2), weights from the design's own Dammann table
```

`ram_budget=inf` is passed in BOTH arms for the reason
`ADJUDICATION_NFC_8192_2026_08_10` sec 2.1 gives: on this box
`get_ram_budget()` cannot approve the requested fine grid, so without the
override the run silently degrades.  The probe additionally ASSERTS the grid
of record -- it refuses if the retrace did not run at `n_fine = NFC` -- rather
than trusting the absence of a warning.

### 1.1 Arm A is the shipped path, and that is checked, not assumed

Arm A calls `propagate_traced_carrier_chain` once per order with exactly the
`focus_readout` that `propagate_traced_carrier_chain_multi._window` builds
(same `dx_out`, same `N_out`, same lattice-snapped `centre_out` from
`_chain_chief_ray_at_target`), and weights the returned tile.  That is the
whole of what the multi orchestrator does per congruence; the recombination it
does afterwards is tile placement on a 32768-square canvas, which for
non-overlapping frames is a copy.

Against the published 32-order adjudication rows for the same three orders at
the same `n_fine_cap` (`_adj_nfc_8192_rows.csv`, arm A, lumenairy 5.33.1):

| order | EE3 % published | EE3 % here | FWHM um pub / here | power_out published | here |
|---|---|---|---|---|---|
| (0,0)   | 90.74076 | **90.741** | 3.400 / **3.400** | 1.8727656e-09 | **1.872766e-09** |
| (-2,0)  | 90.63451 | **90.635** | 3.400 / **3.400** | 1.8735554e-09 | **1.873555e-09** |
| (-4,-2) | 90.07041 | **90.070** | 3.800 / **3.800** | 1.8794212e-09 | **1.879421e-09** |

So the reference arm reproduces the campaign's own banner to every printed
digit, on a different release (5.34.0 vs 5.33.1) and through a different
entry point.

### 1.2 A HAZARD THIS PROBE HIT, AND WHAT IT COST

**The working tree's `lumenairy/**` was edited by another process WHILE this
probe was running** (`lumenairy/elements/_lens_traced.py`, mtime 11:01), and
the effect is worth recording because it is invisible to every intensity
metric the campaign uses.

Arm A was run twice on identical inputs, 09:50 and 10:52.  Every intensity
number reproduced: FWHM identical, EE3 within 0.001 points, window power
within 1.1e-06.  **The returned field's global phase moved by up to 2.88 rad.**

Localised, and it is not non-determinism -- the chain is exactly reproducible:

* `sumap_repro_121.py`, order (-2,0), the same chain twice IN ONE PROCESS:
  `relL2 0.000e+00`, `piston +0.000000 rad`, `power ratio 1.000000000` --
  **bit-identical**; and identical again against the tile a DIFFERENT process
  had cached, so cross-process reproducibility holds too;
* the chain-A cache key did its job and caught the edit: the only key field
  that changed between the two runs is `lumenairy_source_sha256`
  (`7e7749d6ab198008` -> `f68ac2e884ece6f9`), which orphaned the cache and
  forced a cold rebuild;
* comparing the two cached chain-A envelopes directly: `R` and `dx` identical
  to 9 digits, power ratio `1.000000000`, **amplitude identical to 3.0e-14**,
  and a **pure global piston of -1.230 rad** (`relL2 1.154` = `|e^{i*1.23}-1|`
  exactly);
* downstream, that piston does NOT arrive at the image plane as a common
  constant: the three orders' tiles moved by **+0.302 / +2.88 / +0.030 rad**.
  An order-dependent phase shift, from an edit that leaves every amplitude
  bit-identical.

Consequences, both applied here:

1. **Every arm-A/arm-B pair in this note is taken within ONE library state**
   (`lumenairy_source_sha256 = f68ac2e884ece6f9`, recorded in every arm-B
   result file as `lib_sha`).  The first arm-B set, computed from the
   pre-edit aperture fields, is archived under `_sumap_r1/` and is NOT scored
   against the post-edit arm A.
2. **A phase comparison across library states is meaningless**, and no
   shipped metric would have told you: this probe only noticed because it
   compares FIELDS.  That is a finding for the campaign, not for this
   architecture.  (It is also a caution for the parallel piston study: a
   perturbation measured across an edit of the tree measures the edit.)

---

## 2. THE SEAM -- where "the back aperture" actually is, and why

`propagate_traced_carrier_chain` **can** be stopped at the last group's exit
vertex the way chain A is stopped at the DOE (`final_distance=0`, no
`focus_readout`).  MEASURED, order (0,0), 36.5 s:

```
returns dx = 33.2112 um on a 1024-square grid, R_out = -7.712425 mm
co-moving envelope radius w = 1.1853 mm -> exit NA 0.1537
   -> the exit sphere needs dx <= 4.2620 um
UNDER-SAMPLED BY 7.8x     (20.6x against the na_exit = 0.4062 the leg is
                           sized from, which uses the ENTRANCE radius)
```

That plane cannot carry the exit congruence -- which is the entire reason the
exact leg exists.  The exact leg re-traces the last group onto a grid that
Nyquist-samples its exit sphere (`_fine_trace_group_exit`: 8192-square at
1.5324 um, 21.7x finer) and only then propagates.

The earliest plane at which two orders may legitimately be added is therefore
**the last group's back aperture ON THE EXACT LEG'S FINE RETRACE GRID**.
Earlier is impossible in principle: `apply_real_lens_traced` maps entrance to
exit along ONE ray congruence, and a sum of orders is not one.

That plane is internal to the chain, so this probe reaches it with a read-only
spy on `_fine_trace_group_exit`, and **arm B consumes the byte-identical array
arm A propagated**.  Nothing upstream of the seam can differ between the arms,
which is what makes the null control in section 5 a test of the probe's four
steps (re-reference / resample / sum / leg) and of nothing else.

---

## 3. THE GEOMETRY THE GRID IS SIZED FROM

`sumap_census_121.py`, analytic chief-ray traces only (~10 s).  The result
that matters is not the one the protocol anticipated:

```
order      a(m,n)              BACK-APERTURE chief (mm)   exit cosines       image (mm)
 (0,0)   0.166150 e^i+3.0159    ( +0.00000, +0.00000)   (+0.000000,+0.000000)  ( 0.0000, 0.0000)
 (-2,0)  0.166203 e^i-0.0499    ( -0.95951, +0.00000)   (-0.000058,+0.000000)  (-0.9600, 0.0000)
 (-4,-2) 0.166504 e^i+2.0383    ( -1.91510, -0.95755)   (-0.000687,-0.000344)  (-1.9204,-0.9602)

back-aperture carrier radius R = -7.712425 mm  (order-INDEPENDENT)
```

**At this plane the orders are separated SPATIALLY, not angularly.**  Every
exit chief ray is parallel to the axis to within 0.7 mrad; what differs
between orders is *where* the same converging sphere is centred.  So the three
Nyquist bounds a common grid could be sized from are:

| bound | quantity | dx bound | with 2x margin |
|---|---|---|---|
| (a) chief-ray cosine spread (what "tilt spread" reads literally) | 7.68e-04 rad | 852.5 um | 426.3 um |
| (b) **carrier-offset ramp** -- `max|c_k - c_0| / |R|` | **0.277623 rad** | **2.3593 um** | **1.1797 um** |
| (c) each beam's own ABSOLUTE band over its own support, `r/sqrt(r^2+R^2)` at the measured `r(99.999 %) = 2.64 mm` | **0.3239** | **2.0225 um** | **1.0112 um** |

(a) is a red herring here and would have chosen a 400x too coarse grid.  (b)
is the inter-order ramp the protocol means: referenced against ONE sphere,
order k's residual wavefront carries `|c_k - c_0|/|R|` of linear phase, 300x
(a).  (c) is the beams themselves.

**(b) and (c) do NOT add.**  That deserves a sentence, because adding them is
the natural mistake and it costs a factor of 4 in grid: the ramp is a
DIFFERENCE between two spheres, not a displacement of a beam's absolute
angular band.  Order k's field is `env_k * exp(i k S(r - c_k))`, whose local
frequency is `(r - c_k)/(lambda |R|)` -- centred on the beam wherever the beam
sits.  The 0.2776 rad ramp appears only where a COMMON-carrier envelope is
formed and resampled, which on this probe's path happens once, inside the
`crop` leg's upsample.  So the binding pitch is `min` of (b) and (c), i.e.
**2.0225 um**, not `lambda/(2*(0.278 + NA))` = 0.8089 um.

**Chosen: `dx_c = 1.2292 um`, `N_c = 8192`, window 10.0696 mm**, centred on
the common carrier's chief ray (the optical axis) -- 1.65x inside (c) and
1.92x inside (b).  The whole probe was ALSO run at `dx_c = 0.6146 um` /
`N_c = 16384` (the same window at twice the sampling, 3.3x and 3.8x inside the
same two bounds, i.e. past the protocol's 2x margin on both).  **The two
agree to every printed digit on every frame and every metric** -- section 8.1
-- so the answer is pitch-converged and the cheaper grid is the configuration
of record for the cost table.

The window is set by the beams, measured on arm A's own aperture field:
99.999 % of an order's power lies inside 2.64 mm of its chief ray, and the
most displaced order sits 2.14 mm off axis, so a 10.07 mm window clears it by
3.1 mm.  **It does not grow with K**: the full 32-order fan's chief rays span
only -1.915..+1.436 mm in x and -0.958..+0.479 mm in y at this plane, so the
same 10.07 mm window holds all 32 beams with 4.5 mm to spare.  Every per-frame
leg cost in section 7 is therefore K-independent.

---

## 4. WHAT ARM B DOES

Per order, on the aperture field arm A produced:

1. **divide out the order's own congruence** -- exact sphere about its exit
   chief ray, tilt ramp, and the niche-C5 exactness term -- using the
   library's own `_exact_sphere_eikonal` / `_tilt_ramp` /
   `_tilt_exactness_phase`, i.e. the same three lines
   `carrier_referenced_exact_focus_readout` uses;
2. **resample onto the common lattice** with a zero-distance band-limited
   ASM-MFT (`angular_spectrum_propagate_mft(..., z=0, bandlimit=False,
   _bluestein_separable=True)`) -- the library's own exact interpolation, and
   the only route that can hit an arbitrary pitch AND an arbitrary origin,
   which is required because the retrace grids differ per order in both
   (1.5324 vs 1.5243 um; origins 0.000 / -1.508 / -3.016 mm);
3. **restore the same analytic congruence in absolute coordinates**, so what
   survives is the ANALYTIC difference between the order's carrier and the
   common one -- derived, never fitted;
4. **accumulate** `weight * field`.

Then ONE leg on the sum, in two variants:

* **`full`** -- the architecture as specified: reference the ONE common
  sphere, no crop (`window_factor` set so the crop is the whole grid and
  `N_fine == N_c`), then one Bluestein zoom per frame.
* **`crop`** -- the like-for-like variant: crop the SUM about *this* frame's
  chief ray to the same physical window and the same fine grid arm A used
  (4.738 mm, 4096-square, dx_fine 1.157 um), still referencing the ONE common
  sphere, only re-centred.  Legal precisely because every order shares `R` at
  this plane.  This exists because `full` turned out to be the expensive way
  to do it (section 7).

**Energy through steps 1-3 is exact**, measured per order as the power on the
common grid against the power the chain reports at its own exit stage:

```
(0,0)   1.872770e-09  vs  1.872770e-09    ratio 1.000000000
(-2,0)  1.873587e-09  vs  1.873587e-09    ratio 1.000000000
(-4,-2) 1.879556e-09  vs  1.879556e-09    ratio 1.000000000
```

A second, independent check of the same steps: the amplitude radius of the
resampled single-order field measured about its own chief ray on the COMMON
grid is `1.184442977497e-03` m against `1.184442977448e-03` m on arm A's own
retrace grid -- agreement to 4e-11 relative, across a change of pitch (1.5324
-> 0.6146 um in that run) and of grid origin.

Two further construction checks, both passing, both worth stating because
they are the kind of thing a plausible-looking wrong answer hides behind:

* the `crop` leg's window and internal fine grid, recomputed by the harness
  from arm A's own aperture field with the readout's own sizing arithmetic,
  land on **4.738343 mm / 4096 / 1.1568220807 um** for (0,0) and (-2,0) and
  **4.734613 mm / 4096 / 1.1559114902 um** for (-4,-2) -- identical to the
  published `ro_win_mm` / `ro_n_fine` / `ro_dx_fine_um` columns of
  `_adj_nfc_8192_rows.csv` to every digit, so the like-for-like leg really is
  like for like;
* the probe REFUSES if the retrace did not run at `n_fine = NFC` (the grid of
  record is asserted, not inferred from the absence of a warning).

---

## 5. NULL CONTROL FIRST -- single-order arm B vs arm A

The protocol's null control: with ONE order in the sum, arm B must return arm
A.  Any difference is owned by re-reference + resample + leg; summation is
excluded by construction.  Run for all three orders, on the `crop` leg (whose
window and fine grid are arm A's own, so the comparison isolates the
aggregation rather than a windowing difference):

| order | FWHM um A/B | EE3 % A/B (delta) | EE6 % delta | power ratio | field rel L2 | piston (rad) | core phase rms |
|---|---|---|---|---|---|---|---|
| (0,0)   | 3.400 / 3.400 | 90.741 / 90.741 (**0.000**) | 0.000 | **0.9999999** | **2.778e-05** | +0.0e+00 | 1.8e-06 |
| (-2,0)  | 3.400 / 3.400 | 90.634 / 90.634 (**0.000**) | 0.000 | **0.9999997** | **1.403e-04** | -0.0e+00 | 2.3e-05 |
| (-4,-2) | 3.800 / 3.800 | 90.071 / 90.071 (**0.000**) | 0.000 | **1.0000001** | **9.342e-05** | -0.0e+00 | 5.3e-06 |

**NULL CONTROL: PASS on every bar, by 2-3 orders of magnitude.**  Energy to
1e-07 against a 4e-05 bar; EE to the printed digits against a 0.1-point bar;
FWHM identical; zero piston.  The residual **2.8e-05 to 1.4e-04 of field rel
L2 is the aggregation's entire error budget** -- one band-limited resample
between two grids that differ in pitch (1.5324 or 1.5243 -> 1.2292 um) and in
origin (0.000 / -1.508 / -3.016 mm), plus the analytic re-reference either
side of it.

Same control on the `full` leg (no crop, whole 10.07 mm aperture propagated),
order (0,0): rel L2 **7.317e-04**, power ratio 0.9999998, EE3 delta -0.001.
The 26x larger rel L2 is not a worse answer -- it is a DIFFERENT window: the
full leg carries the halo arm A's 4.738 mm crop discards, so the two are not
propagating the same field.  Section 8.2 keeps the two apart.

---

## 6. THE 3-ORDER SUM

All three orders summed on one grid against one carrier, then read at the same
three windows arm A used.

### 6.1 `crop` leg (arm A's own window and fine grid)

| frame | FWHM um | EE3 % (delta) | EE6 % (delta) | power ratio | rel L2 | piston (rad) |
|---|---|---|---|---|---|---|
| (0,0)   | 3.400 (=A) | 90.740 (**-0.001**) | 99.897 (0.000) | **0.9999978** | 1.796e-03 | +2e-07 |
| (-2,0)  | 3.400 (=A) | 90.634 (**0.000**) | 99.861 (0.000) | **1.0000001** | 1.502e-03 | +2e-06 |
| (-4,-2) | 3.800 (=A) | 90.071 (**0.000**) | 99.851 (0.000) | **1.0000001** | 9.382e-05 | -3e-07 |

### 6.2 `full` leg (one leg on the whole summed field, as specified)

| frame | FWHM um | EE3 % (delta) | EE6 % (delta) | power ratio | rel L2 | piston (rad) |
|---|---|---|---|---|---|---|
| (0,0)   | 3.400 (=A) | 90.740 (**-0.001**) | 99.897 (0.000) | **0.9999953** | 7.912e-04 | -4e-07 |
| (-2,0)  | 3.400 (=A) | 90.630 (**-0.004**) | 99.858 (-0.003) | **1.0000068** | 3.206e-03 | +1e-06 |
| (-4,-2) | 3.800 (=A) | 90.068 (**-0.003**) | 99.850 (-0.001) | **1.0000158** | 3.835e-03 | -1e-06 |

**Every energy delta is inside the 4e-05 bar (worst 1.6e-05), every EE delta
is inside the 0.1-point bar (worst 0.004), and every FWHM is identical to the
printed digits.**  The field rel L2 rises from the null control's 2.8e-05 to
1.5e-03 the moment the other two orders are added, and that rise is not error:
it is the crosstalk of 6.4, which arm A cannot contain.

### 6.3 PISTON / FRINGE CHECK

The check that decides whether a summed-aperture architecture can be used for
anything phase-sensitive:

```text
crop leg   pistons  +0.000000  +0.000002  -0.000000  rad   spread 1.673e-06
full leg   pistons  -0.000000  +0.000001  -0.000001  rad   spread 2.255e-06
```

No frame carries a piston against the shipped path above 2.3e-06 rad, and --
the part that matters -- **the piston is the same for every frame to
2.3e-06 rad**, so the arms agree not merely on each frame's intensity but on
the RELATIVE PHASE BETWEEN FRAMES.  A chain-to-chain piston would have shown
here as a per-frame offset; an inconsistent one, which is what would break
coherent use of the fan, would have shown as a spread.  Neither is present.

For scale, the same measurement applied to the library edit of section 1.2
gives per-frame pistons of +0.302 / +2.88 / +0.030 rad -- a spread of 2.85
rad, six decades larger.  So this check is not insensitive; the aggregation
simply does not move the phase.

*(A separate agent is measuring piston STABILITY under perturbation.  This is
the arm-A/arm-B delta only, as instructed; the section-1.2 numbers are
reported because the probe hit them, not as a perturbation study.)*

### 6.4 CROSSTALK

Measured the way the protocol asks: arm B run with a SINGLE order in the sum,
reading ALL THREE frames, so the off-diagonal is the power frame i receives
from order j.  **Arm A has no such quantity**: its frame i is order i's own
tile placed on the common lattice and nothing else, so its off-diagonal is
zero by construction, not by physics.

Power in frame i (same units as the tables above) and the ratio to that
frame's own order -- `full` leg:

```text
frame i \ order j        (0,0)          (-2,0)         (-4,-2)
   (0,0)             1.872765e-09    1.708338e-16    5.691235e-19
   (-2,0)            2.865372e-17    1.873573e-09    7.265866e-18
   (-4,-2)           6.393729e-22    8.481403e-21    1.879452e-09
ratio to diagonal
   (0,0)                1.000          9.122e-08       3.039e-10
   (-2,0)               1.529e-08      1.000           3.878e-09
   (-4,-2)              3.402e-13      4.513e-12       1.000
```

`crop` leg, ratios only:

```text
   (0,0)                1.000          3.210e-06       4.221e-11
   (-2,0)               2.410e-06      1.000           4.563e-11
   (-4,-2)              3.459e-11      1.053e-11       1.000
```

**Worst crosstalk is 9.1e-08 of a frame's own power on the `full` leg and
3.2e-06 on the `crop` leg** -- 440x and 12x inside the 4e-05 energy bar -- and
both are between NEAREST-NEIGHBOUR frames, falling 3-5 decades to the
next-nearest.  On design 121 the frames are optically independent well below
the campaign's own tolerance, so the shipped architecture's structural
blindness to crosstalk costs it nothing HERE.  It is still a blindness:
nothing in the shipped path would report it if a different design put it at
1e-03.

The 35x gap between the two legs is not noise, and it is the one place the
like-for-like leg is the DIRTIER of the two: `crop` cuts a 4.738 mm window out
of the SUM, truncating the neighbouring beams mid-aperture, and a hard
truncation of a neighbour diffracts into this frame.  `full` truncates nothing
and reports the genuine tail.

### 6.5 LINEARITY, as an internal control

Arm B run once per single order and once on the 3-order sum, then compared:

```text
|| sum_j B_j - B_all || / || B_all ||
   full leg   2.321e-16   2.315e-16   2.320e-16
   crop leg   3.002e-16   2.952e-16   3.581e-16
```

Float64 round-off.  Summation and the leg are exactly linear, which is what
makes the crosstalk table a decomposition rather than an approximation.

---

## 7. COST -- the half of the hypothesis that fails

Timings are wall-clock on a shared box; a concurrent job moved the same
operation by up to 2.5x during this probe (section 8.4).  Every figure below
is therefore quoted from runs taken in the same load window, and the
conclusion is checked against the extremes of the observed range.

### 7.1 Where an order's time actually goes (arm A, measured)

| order | coarse chain | fine RETRACE | exact READOUT | wall | peak working set |
|---|---|---|---|---|---|
| (0,0)   | 14.1 s | 111.2 s | 11.9 s | 137.2 s | 22.8 GB |
| (-2,0)  | 20.0 s | 104.4 s | 18.4 s | 142.8 s | 26.8 GB |
| (-4,-2) | 15.8 s | 101.2 s | 17.8 s | 134.9 s | 26.9 GB |
| **mean** | **16.7 s (12.0 %)** | **105.6 s (76.4 %)** | **16.0 s (11.6 %)** | **138.3 s** | |

**The leg the architecture shares is 11.6 % of the order.  The 76.4 % that
dominates is the fine re-trace of the last group, which is UPSTREAM of any
plane where two orders may be added** (section 2).  So "1/32 of the leg cost"
is 1/32 of 11.6 % -- an 11.2 % ceiling on the run, before arm B's own overhead
is counted.

### 7.2 What arm B costs (measured)

Per order, on top of the same coarse chain + retrace arm A pays:

| step | measured | note |
|---|---|---|
| re-reference (both directions) | 16.9 s | exact-sphere + ramp + exactness screens, twice, on the common grid |
| resample onto the common lattice | 13.1 s | one z=0 band-limited ASM-MFT per order |
| **aggregation overhead / order** | **30.0 s** | against the **16.0 s** readout it is trying to amortize |
| leg `full`, per FRAME | 14.9 s | 8192-square fine grid |
| leg `crop`, per FRAME | 9.3 s | 4096-square, arm A's own |
| peak working set | **17.5 GB** | *below* arm A's 22.8-26.9 GB |

The one genuinely favourable number in this section is the last: because arm B
never builds a per-order retrace grid and readout grid at the same time, its
peak working set is 25-35 % BELOW the shipped path's, at the same answer.

### 7.3 The 32-order projection

Per-order cost is measured on three orders spanning the fan's whole chief-ray
excursion, so the mean is used rather than the best case.

```text
ARM A   32 x 138.3                                       = 4 426 s = 1.23 h
ARM B   32 x (122.3 + 30.0) + 32 x 14.9   (full leg)     = 5 350 s = 1.49 h   1.21x SLOWER
ARM B   32 x (122.3 + 30.0) + 32 x  9.3   (crop leg)     = 5 171 s = 1.44 h   1.17x SLOWER
```

Three hypothetical improvements, priced so the ceiling is explicit:

```text
resample eliminated (retrace emitted directly on the common lattice):
        32 x (122.3 + 16.9) + 32 x 9.3                   = 4 752 s = 1.32 h   1.07x slower
PLUS a 'propagate once, zoom K times' entry point (8.3: 81 % shareable):
        32 x (122.3 + 16.9) + 14.5 + 32 x 3.5            = 4 581 s = 1.27 h   1.03x slower
BOTH, and the aggregation free as well -- an upper bound that cannot be
reached, since re-referencing 32 orders is not free:
        32 x 122.3 + 14.5 + 32 x 3.5                     = 4 040 s = 1.12 h   1.10x FASTER
```

**The architecture's absolute ceiling on this design is a 10 % speed-up; the
measured version of it is a 17-21 % slowdown.**  The common grid does not grow
with K (section 3), so none of this improves at 32 orders; and at the shipped
`n_fine_cap = 16384` the retrace's share grows with the pixel count, so the
ceiling falls further.

---

## 8. ABLATIONS

### 8.1 Common-grid pitch -- converged

The probe was run at `dx_c` = 0.6146 um / `N_c` = 16384 and at
`dx_c` = 1.2292 um / `N_c` = 8192 -- the same 10.07 mm window, half the
sampling, 4x apart in memory.  On the `full` leg every scored quantity on
every frame agrees **to every printed digit**:

```text
full leg, 3-order sum, BOTH grids:
  (0,0)   EE3 90.7409  EE6 99.8969  FWHM 3.400  P/P_A-1 +1.082e-05  relL2 8.074e-04  piston +2.99e-07
  (-2,0)  EE3 90.6351  EE6 99.8616  FWHM 3.400  P/P_A-1 +1.261e-05  relL2 3.096e-03  piston -2.12e-06
  (-4,-2) EE3 90.0713  EE6 99.8505  FWHM 3.800  P/P_A-1 +1.870e-05  relL2 3.898e-03  piston +2.45e-07
```

(the `crop` leg is the only place the pitch is visible at all, in the 4th
decimal of EE3: deltas vs arm A go +0.0009/+0.0001/+0.0000 at the fine grid
and +0.0008/-0.0006/+0.0000 at the coarse one -- 100x inside the EE bar.  Both
runs above are on the same aperture realisation, archived under `_sumap_r1/`.)

That is the measurement behind section 3's claim that the inter-order ramp and
the beams' own band do NOT add.  At 1.2292 um the grid is 1.5x INSIDE
`lambda/(2*(ramp + NA))`; if that were the operative bound the extreme order
would have aliased and moved.  It does not move.

### 8.2 Leg variant

| | `full` | `crop` |
|---|---|---|
| what it is | one leg on the whole summed aperture, as specified | crop the sum about each frame's chief ray to arm A's own window/grid |
| leg cost per frame | 14.9 s | 9.3 s |
| worst crosstalk | 9.1e-08 | 3.2e-06 (truncates the neighbours) |
| worst energy delta vs A | 1.6e-05 | 2.2e-06 |
| null-control rel L2 | 7.3e-04 (a different window from A) | 2.8e-05 |
| K-dependence | none | none |

Both are inside every bar.  `crop` is cheaper and closer to arm A field for
field; `full` is the cleaner physics for crosstalk.

### 8.3 How much of one leg is shareable at all

`sumap_legsplit_121.py` times the readout on a synthetic field of the
production shape with the Bluestein inverse instrumented, because the shipped
`carrier_referenced_exact_focus_readout` does the whole leg per call and no
entry point separates the two:

```text
N_fine =  8192, dx_fine 1.2292 um:  18.85 s = 15.77 shareable + 3.08 per-frame  (16.3 %)
                                    17.05 s = 13.18 shareable + 3.87 per-frame  (22.7 %)
N_fine = 16384, dx_fine 0.6146 um:  88.51 s = 72.61 shareable + 15.90 per-frame (18.0 %)
```

So **~81 % of a leg is in principle shareable across frames** (sphere
reference, crop/upsample, reconstruct, forward FFT, transfer function) and
~19 % is irreducibly per-frame.  Section 7.3 prices what a library entry point
exposing that split would be worth: ~360 s on a 4 400 s run.

### 8.4 Run-to-run variance, and why the tables are paired

The same `full`-leg zoom at `N_c` = 16384 measured 157/165/128 s in one
process, 86/93/92 s in another and 67/68/66 s in a third, on identical inputs;
the coarse-grid crop zoom measured 9.1 s early and 41.7 s late.  The spread is
FFTW plan caching plus a concurrent job on the same box, not physics -- the
FIELDS are bit-reproducible (section 1.2).  Every ratio in section 7 is
therefore taken from runs in the same load window, and the sign of the verdict
survives the extremes: at the most favourable arm-B timing observed and the
least favourable arm-A one, arm B is still not faster.

---

## 9. GO / NO-GO

**GO -- the physics.**  Summing per-order fields at the last group's back
aperture against one common analytic carrier reproduces the shipped per-order
path inside every bar this campaign uses: null control 2.8e-05 field rel L2
and 1e-07 energy; 3-order sum worst 1.6e-05 energy (bar 4e-05) and 0.004 EE
points (bar 0.1); FWHM identical everywhere; piston below 2.3e-06 rad with an
inter-frame spread of 2.3e-06 rad; linearity 2.3e-16.  Nothing in the
re-reference, the resample or the summation introduces a defect; the
architecture measures inter-frame crosstalk, which the shipped path cannot
represent at all; and it runs in 25-35 % LESS memory.

**NO-GO -- as a performance change.**  The cost premise does not hold:

1. **The leg is 11.6 % of an order, not 100 %.**  76.4 % is the fine re-trace
   of the last group, which cannot be shared because
   `apply_real_lens_traced` maps entrance to exit along ONE congruence.
2. **Sharing it does not make it free.**  19 % of a leg is an irreducible
   per-frame Bluestein zoom, and the aggregation that builds the shared field
   costs 30.0 s/order against the 16.0 s/order the leg is worth.
3. Measured end to end, arm B is **1.17x-1.21x slower**; the absolute ceiling,
   with the resample eliminated AND a shared-leg entry point AND a free
   aggregation, is **1.10x faster**.

### Blocking items, named

1. **The retrace is the cost, and it is not addressable at this seam.**  Any
   real 32x has to attack `_fine_trace_group_exit` -- its Newton inversion and
   its 8192-square OPL fit -- not the readout.  A shared-pupil formulation of
   the last group's ray map (one map reused by every order, instead of one per
   congruence) is the only route that could reach the claimed factor, and it
   is a library change of a different order of magnitude from this probe.
2. **The retrace grid is per-order in BOTH pitch and origin** (1.5324 vs
   1.5243 um; origins 0.000 / -1.508 / -3.016 mm), so the aggregation pays a
   full band-limited resample per order.  If `_fine_trace_group_exit` accepted
   a CALLER-SUPPLIED lattice, 13.1 s/order disappears -- the cheapest of the
   three items, and the one worth doing first if the architecture is pursued
   for its crosstalk visibility rather than for speed.
3. **No library entry point separates the shareable part of a leg from the
   per-frame Bluestein.**  Worth ~8 % of a run here, and strictly more on a
   design whose readout fraction is larger; it is a real API addition
   (`carrier_referenced_exact_focus_readout` would have to return, or accept,
   an already-propagated spectrum).

Two further items that do not block the architecture but are findings of this
probe:

4. **The working tree was edited under a running measurement** (section 1.2)
   and it moved the chain's absolute output phase by up to 2.88 rad while
   every intensity metric held to 0.001 EE points.  The chain-A cache key
   caught it, which is exactly what it exists for; nothing else would have.
5. **Arm A's own recombination cannot see crosstalk.**  The shipped
   acceptance's per-frame powers are, by construction, per-order powers -- on
   this design a 9e-08 approximation, but an approximation whose size is only
   measurable through something like this probe.

### What is NOT claimed

* Nothing is measured at 32 orders; the projection is from 3, chosen to span
  the fan's chief-ray excursion, and the per-order cost varies by 1.06x across
  them (137-143 s) in the unloaded run of record.
* `n_fine_cap = 8192` throughout, per the protocol's affordability
  instruction.  At the shipped 16384 the retrace's share grows, so section 7's
  conclusion gets stronger -- an inference, not a measurement.
* Crosstalk is measured for 3 of 32 orders.  The two nearest-neighbour pairs
  present are the ones that matter most, but a full-fan census would have 32
  frames with up to 8 neighbours each.
* Each frame is read on the SAME window arm A uses.  A finer frame lattice, or
  a readout window wide enough to overlap its neighbour, would change the
  crosstalk numbers (and nothing else).
* The `crop` leg's carrier choice -- the ONE common sphere, re-centred on each
  frame's chief ray -- is legal because every order shares `R` at this plane
  (measured order-independent, -7.712425 mm).  A design whose exit radius
  varies across the fan would need the mean sphere plus a residual quadratic
  per order; that case is not tested here.

---

## 10. FILES

| file | what |
|---|---|
| `validation/repro_traced_carrier_121/sumap_census_121.py` | geometry census: exit congruences, the three Nyquist bounds, the window |
| `validation/repro_traced_carrier_121/sumap_probe_121.py` | `census` / `seam` / `arma` / `armb`; the spy, the re-reference, the resample, the sum, both legs |
| `validation/repro_traced_carrier_121/sumap_repro_121.py` | run-to-run reproducibility of the shipped path (section 1.2) |
| `validation/repro_traced_carrier_121/sumap_legsplit_121.py` | shareable vs per-frame split of one exact leg |
| `validation/repro_traced_carrier_121/sumap_score_121.py` | every table in sections 5-6; writes `_sumap_score*.json` |
| `validation/repro_traced_carrier_121/sumap_runall.sh`, `sumap_runall2.sh` | the arm-B run sets, serially (peak memory forbids overlap) |

Every script is import-safe (`if __name__ == '__main__'`).  Results of record
are `_sumap_A_*.npz` (arm A tiles + metadata) and `_sumap_B[fc]_*_r2.npz` (arm
B), with the pre-edit set archived under `_sumap_r1/`.

**Housekeeping note:** the back-aperture caches `_sumap_ap_*.npy` are 1.07 GB
each and the directory's `.gitignore` covers `_*.npz` only, so those three
files are NOT ignored.  Delete them (they rebuild in ~7 min via
`sumap_probe_121.py arma`) or extend the ignore rule before committing
anything from this directory.

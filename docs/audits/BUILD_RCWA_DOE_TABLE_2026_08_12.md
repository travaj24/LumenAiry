# BUILD -- the rigorous (RCWA) DOE order table for design 121

**2026-08-12.  Branch `feat/rcwa-doe-table` off `origin/main` @ `21802f9`, in
the dedicated worktree `C:\tmp\lum_doe`.  NEW FILES, and none under
`lumenairy/**`: `validation/repro_traced_carrier_121/doe_rcwa_table.py`,
`validation/pipeline/doe_rcwa.py`, two spec files, `tests/unit/test_doe_rcwa.py`,
this note.  `validation/pipeline/sources.py` gains SEVEN LINES (a comment and
one import, for the registration side effect) and nothing else.  `CHANGELOG.md`
not touched.  No `git commit`, no `git push`, no `gh`.**

Replaces the scalar Dammann FFT order table of
`validation/repro_traced_carrier_121/_d121_common.order_table` with a rigorous
angle x polarization RCWA table over the reconstructed etched cell, and plugs it
into the pipeline of `docs/audits/BUILD_PIPELINE_2026_08_11.md` as a second
decomposer so the two can be A/B'd with one variable moving.

---

## 0. VERDICT

> **BUILT, PLUGGED IN, AND IT PRODUCED THREE FINDINGS -- ONE OF WHICH IS THAT
> THE RIGOROUS ANSWER IS NOT REACHABLE.**
>
> **F1 -- THE TABLE DOES NOT CONVERGE, AND CANNOT.**  The design-121 cell is a
> 113.7566 um period (86.8 wavelengths) carrying 0.8887 um features
> (0.68 wavelengths).  A converged RCWA needs harmonics out to
> `+-period/feature = +-128` per axis -- 66049 harmonics, a 132098-square
> eigenproblem, which is out of reach by many orders of magnitude.  Long
> before that the solver's own energy guard refuses: on the default
> factorization **every truncation from `n_orders = 7` upward blows up**
> (`R+T` reaching 2.5e+40), and **nothing reaches `n_orders = 10` in any
> combination of the two factorizations, the two order sets and the two
> admissible plate mountings**.  The clean ladder is `n_orders = 4, 5, 6` and
> it is still moving: in-band efficiency **0.6557 / 0.6176 / 0.5947**, worst
> per-order amplitude movement **0.0262** and phase **0.686 rad** on the last
> rung.  Every number below carries that as its dominant uncertainty.
>
> **F2 -- WHAT THE THIN-ELEMENT ASSUMPTION HIDES IS NOT SMALL.**  At the
> highest truncation that solves, against the scalar table on the SAME cell:
> in-band power **0.5947 vs 0.8851** (-32.8 % relative; -30.4 % on the
> transmitted fraction, so it is redistribution, not Fresnel loss), and
> **uniformity min/max 0.0021 against 0.9954** -- the design's own headline
> uniformity claim does not survive at all.  Per-order efficiency goes from
> `0.027658 +- 0.000030` (scalar, by construction) to `0.0186 +- 0.0151`
> (RCWA): the weakest order carries `1.4e-04` and the strongest `0.0646`, a
> **475x spread**.  Piston-removed per-order phase deltas reach **3.09 rad**
> (rms 1.65 rad) -- the phases are not perturbed, they are unrelated.  These
> are the numbers at `n_orders = 6`; F1 is why they are directional and not
> quotable to three digits.
>
> **F3 -- THE SCALAR TABLE'S ANGLE-BLINDNESS IS IMMATERIAL, AND THAT IS
> MEASURED.**  Chain A puts the DOE in near-collimated space: `R = +703.64 m`
> over a 13.69 mm support, so the beam's cone at the DOE is **1.25e-04 rad at
> 99.9 % enclosed** (`theta_rms = 4.75e-05`).  Across that cone, on a 3 x 8
> Chebyshev x uniform angle grid solved CLEANLY at every node, the per-order
> efficiency varies by **4.55e-04 relative** and the per-order phase by
> **2.36e-04 rad** -- four decades below the truncation uncertainty of F1.
> **Polarization is immaterial too**: cross-polarized power stays below
> **7.8e-06** of every order.  So of the three things the scalar model throws
> away -- rigour, angle, polarization -- only the FIRST matters on this design,
> and it matters enormously.
>
> **THE A/B RAN END TO END, BOTH ARMS, AND THE DELTA IS 134 %.**  Two full
> 3-order pipeline runs (3 traced chains each to the 8192-square fine-retrace
> exit, one common carrier, one leg plan, 3 Bluestein zooms), identical in
> every stage except the decomposer.  Frame powers move by **x2.338 / x1.078 /
> x0.00827** -- reproducing `|w_RCWA|^2 / |w_scalar|^2` to six digits, which is
> the control that proves the pipeline is exactly linear in the weights and
> contributes nothing itself.  Shape is untouched (worst `dEE3` 0.0018 points,
> FWHM identical on 3/3), and **both arms hold every campaign bar against their
> own shipped tiles** (worst `|P/P_shipped - 1|` 8.0e-06 and 1.9e-05 against
> the 4e-05 bar).  **FLAGGED: the energy delta is 67x past a 2 % line.**  It is
> flagged rather than reported as a result, because it is the same size as the
> table's own model uncertainty (F1) -- what it establishes is that the DOE
> model is load-bearing for the answer, not what the answer is.
>
> **The design's own carrier order is the worst hit.**  `(-4,-2)` -- the
> `.zmx`'s multi-config design order, and the CORNER of the 8x4 block -- gets
> **0.83 % of its designed power**.  The loss is not spread: the `my = -2` row
> keeps 4.7 % and the `mx = -4` column 3.2 %, while the interior 21 orders keep
> their total to **1.0001x** and merely redistribute inside it (S5.2).
>
> **The 0.60 in-band figure survives a change of instrument.**  Five different
> discretizations -- two Fourier factorizations x two reciprocal-lattice order
> sets, retained harmonics `N` from 113 to 225 -- put the in-band FRACTION in
> **0.5997 .. 0.6044** and the uniformity in **0.0003 .. 0.0049** (S3.2).  A
> truncation artifact does not survive that; a physical answer does.
>
> **A SILENT-WRONG DEFECT WAS FOUND AND KILLED IN THE BUILD.**  The solver's
> cell is indexed `[x, y]`; the Dammann cell is indexed `[y, x]`.  Handing one
> to the other untransposed solves the TRANSPOSED structure -- energy-clean to
> 1e-10, convergent, and wrong -- and reports `sum |a|^2 = 0.4488` against the
> true `0.8851`, exactly the transposed order block's power.  It is pinned by
> two regression tests, one on the array and one on the physics (S4.1).
>
> **41 new tests pass** on Windows, no xfail, no skip, in 10 s on a fresh
> checkout with no Zemax file and no design-121 cache; `test_pipeline.py` +
> `test_carrier_field.py` + this file = **115 passed**; `ruff check lumenairy/
> tests/unit/` clean.

---

## 1. STRUCTURE RECONSTRUCTION -- STATED LOUDLY

**The physical DOE is not recorded anywhere in the design tree.**  This is not
a case of a spec being hard to find; it is a case of every model of the part --
Zemax's and lumenairy's alike -- being an idealized zero-thickness phase screen
in air.  Everything in this section that is not marked RECORDED is an
assumption, is a parameter of `DoeStructure`, and is in the cache key, so a
corrected value is one re-run.

| # | quantity | status | value used | evidence |
|---|---|---|---|---|
| A1 | material / index | **ASSUMED** | fused silica, `n = 1.446804` (Malitson @ 1.31 um) | `.zmx` surfaces 9 and 11 are `DGRATING` with **no `GLAS` line** and `DISZ 0` -- zero-thickness, in air.  `_NEW_GLASSES` registers N-SK2 / N-SF1 / N-PK52A / N-LAK8 / N-LAK9, none at the DOE.  The only DOE substrate named anywhere in this project lineage is the `F_SILICA` alias `tx_design_study_sim` registers for the Design-36/71-era part. |
| A2 | relief depth | **DERIVED** | step `dz = 0.366492 um`, total `2.565443 um` | `lambda*phi/(2 pi (n-1))`; no etch depth exists in the tree ("etch" has zero hits). |
| A3 | phase levels | **RECORDED** | 8 | `DAMMANN_PHASELEVELS = 8`; the cached cell takes exactly 8 values, multiples of `2 pi / 8`. |
| A3b | mask / etch steps | **NOT 4** | (not needed) | `DAMMANN_PHASESTEPS = 4` is the ANNEALING-SCHEDULE EXPONENT of `makedammann2d` (`phaselevelscur = phaselevels * 2**floor(phasesteps*(itr-it)/itr)`), not a mask count.  An 8-level binary optic is **3** masks.  No mask count is recorded.  **The task brief's "4 steps" is a misreading of that parameter and is corrected here.** |
| A4 | which face the beam meets first | **ASSUMED** | `relief_first` (air -> relief -> substrate) | Both mountings are admissible and the order tilts do NOT discriminate (S1.1). |
| A5 | cell resolution | **CHOSEN** | 128 px, pixel `0.888724 um` = 0.679 lambda | The pipeline's own (`_d121_common.order_table(n_per=128)`), so the comparison is on ONE geometry.  The design tree also carries a 174-px variant (`doe_cache/...samp0.5...`), unused here. |
| A6 | walls, rounding, bias, loss | **IDEALISED** | vertical, sharp, unbiased, lossless | Nothing is recorded.  Every real deviation makes the agreement with the scalar table WORSE, so S5's deltas are a LOWER BOUND. |
| -- | period | **RECORDED** | `113.7566259645458 um` | `WAVELENGTH * abs(B_DOE_TO_MS) / FRAME_PITCH` in the runner; the `.zmx` `PARM 1 = 0.00879` lines/um agrees to 0.008 %; the pre-PDR deck says "DOE 8.79 lines per mm". |
| -- | design efficiency | **RECORDED** | 88.51 % into 32 orders, per-frame `2.766 +- 0.003 %`, uniformity 0.9954 | `exp26_run.log:10`.  **Reproduced here to six digits by the scalar table: 0.885056, per-order `0.027658 +- 0.000030`, uniformity 0.995435.** |

### 1.1 The mounting is genuinely undetermined -- and it is tempting to think it is not

The obvious argument is that the pipeline assigns order `(m, n)` the tilt
`(m, n) lambda / period` with **no index**, so the diffracted orders travel in
AIR, so the exit half-space must be air, so the beam must arrive through the
substrate.  **That argument is wrong**, and it was believed for one build
iteration here: the plate's OTHER face is FLAT, and a flat face refracts the
in-substrate tilts back to exactly `m lambda / period` in air.  Both mountings
reproduce the pipeline's tilts.

They are different rigorous problems and they disagree measurably: at
`n_orders = 5`, the only rung where BOTH solve, in-band power reads **0.6176
(`relief_first`) against 0.6397 (`substrate_first`)** -- a 3.5 % spread that is
pure mounting ambiguity.  `substrate_first` is also worse conditioned (it loses
the energy guard at `n_orders = 4` and 6, leaving only 5).  `relief_first` is
the model of record **for that numerical reason alone**, which is stated rather
than dressed up as physics.

Either way one flat air/glass face sits OUTSIDE the cell.  It is flat, so it
scales every order alike and cannot redistribute them (`flat_face_fresnel`
reports 0.96665 for both polarizations at normal incidence; its variation
across this design's 0.046 rad order cone is ~1e-04 relative).  It is
deliberately NOT folded in: a millimetre-thick plate modelled with both faces
coherently is a millimetre etalon, which is not the physics of a real wedged or
incoherently-thick substrate.

### 1.2 There is no recorded 2 % fan tolerance

The brief asks for deltas to be adjudicated against "the fan's design tolerance
2 %".  **No such tolerance is recorded anywhere in the tree.**  The only
2-ish-percent figure in the DOE's context is the per-frame POWER SHARE,
`2.766 +- 0.003 %` (= 100 %/32 x 0.8851), which is a design output, not a
tolerance.  The bars actually used below are therefore the campaign's own, the
ones `report.py` and `pipeline_accept_121.py` already score against
(`BAR_ENERGY = 4e-05` on `|P/P_shipped - 1|`, `BAR_EE_POINTS = 0.1`), plus a 2 %
line drawn purely so the brief's question has an answer.

---

## 2. ANGULAR DOMAIN -- measured from chain A, not assumed

`_d121_common.chain_a(n=1024, rs=4)` to the DOE plane, then the angular
spectrum of the full field (envelope x carrier) on that grid:

```text
R at the DOE     +703.642736 m          <- NEAR-COLLIMATED (the fan is formed
dx                51.2334 um               downstream: B(DOE->MS) = 41.68 mm)
grid              1024^2 = 52.46 mm
w_amp              6.3168 mm
support r(99.999 %)  13.6942 mm

angular content of the FULL field, enclosed-power half-angle
   50 %    3.531e-05 rad   (0.00202 deg)
   90 %    7.491e-05 rad   (0.00429 deg)
   99 %    1.030e-04 rad   (0.00590 deg)
   99.9 %  1.249e-04 rad   (0.00715 deg)
   99.999% 3.754e-04 rad   (0.02151 deg)
   theta_rms  4.729e-05 rad
carrier's own ray angle at the 99.999 % radius:  1.946e-05 rad  (negligible)
```

The envelope and the full field give the same numbers to three digits -- the
`R = 703 m` sphere contributes 2e-05 rad over the whole aperture -- so the cone
is set by the beam, not by the carrier.

**The grid.**  `theta` on **Chebyshev-Gauss-Lobatto nodes** of
`[0, theta_max = 5.0e-04]` (4.0x the 99.999 %-enclosed half-angle, 10x
`theta_rms`); `phi` **uniform** on `[0, 2 pi)`.  The split is deliberate:

* `a_m(theta)` is analytic across this cone -- the nearest Rayleigh anomaly is
  at `theta = 0.108 deg` (order -87 crossing cutoff), 30x outside it -- and the
  quantity the decomposer needs is a WEIGHTED INTEGRAL, so Chebyshev nodes
  carry Clenshaw-Curtis quadrature (spectral) where a uniform grid carries the
  trapezoid rule (`O(h^2)`).  The same nodes give a well-conditioned
  barycentric interpolant, which is what an angle-RESOLVED decomposition would
  need later.  Pinned by a test: on `INT_0^1 cos(3t) dt` the 9-node
  Clenshaw-Curtis beats the 9-point trapezoid by >100x.
* `phi` is PERIODIC, where the trapezoid rule is already spectral, so
  Chebyshev there would be a pessimisation.

`theta = 0` is azimuth-degenerate and is solved ONCE and broadcast (not an
approximation -- the incident wavevector is identical), so a `3 x 8` grid is
**17 distinct solves**, both incident linear polarizations per solve.

---

## 3. THE SWEEP

### 3.1 The convergence ladder -- and the wall

Normal incidence, `relief_first`, `laurent`, rectangular truncation, the
design's own 32 orders.  Energy accounting is per solve.

| `n_orders` | N | wall | `sum T` | `sum R` | `|R+T-1|` | in-band `sum|a|^2` | in-band / `sum T` | uniformity | max x-pol |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 81 | 1.0 s | 0.989393 | 0.010608 | 1.7e-07 | **0.655676** | 0.662705 | 0.0039 | 4.5e-06 |
| 5 | 121 | 3.8 s | 0.986435 | 0.013553 | 1.9e-03 | **0.617597** | 0.626090 | 0.0050 | 2.0e-06 |
| 6 | 169 | 7.3 s | 0.983866 | 0.016169 | 3.5e-05 | **0.594687** | 0.604439 | 0.0021 | 7.8e-06 |
| 7 | -- | 13.8 s | -- | -- | -- | **`_EnergyError`** | | | |
| 2, 3 | | | | | | (design orders not retained) | | | |

Rung-to-rung movement of the observable the table exports -- the complex
per-order amplitude, NOT the total power (a lossless cell conserves energy at
every truncation, so energy proves nothing about the split; the lossless trap):

```text
M 4->5   d(in-band) -0.038079   worst d_eta 0.019067   worst d|a| 0.035058   worst d_arg 0.8596 rad
M 5->6   d(in-band) -0.022910   worst d_eta 0.007214   worst d|a| 0.026217   worst d_arg 0.6861 rad
```

**There is no headroom, because there is no rung above 6.**  The chosen
truncation is the ceiling, and it is reported as such.

### 3.2 The wall moves by one rung with the formulation -- and the discretizations that get past it AGREE

The wall is not identical for every factorization.  Measured on the
`relief_first` mounting, `on_unstable='raise'`, normal incidence
(OK = solved; the closure is the number that matters, not the survival):

| truncation | formulation | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|
| rectangular | laurent | OK (3.5e-05) | UNSTABLE | UNSTABLE | UNSTABLE |
| rectangular | li | UNSTABLE | OK (5.6e-02) | UNSTABLE | UNSTABLE |
| circular | laurent | OK (1.2e-05) | UNSTABLE | OK (2.4e-02) | UNSTABLE |
| circular | li | UNSTABLE | OK (4.0e-02) | UNSTABLE | UNSTABLE |

**Nothing reaches 10 in any combination, and every rung above 6 violates
lossless energy conservation by 2.4 % to 5.6 %** -- which the solver's own
guard describes as *"the PER-ORDER efficiencies are suspect"*.  So they are not
usable individually.

**But they agree with each other, and that is the strongest evidence in this
note**, because they are genuinely different discretizations -- different
Fourier factorization, different reciprocal-lattice order SET, different
retained-harmonic count `N` spanning 113 to 225:

| truncation | formulation | `n_orders` | N | `|R+T-1|` | in-band | in-band / `sum T` | uniformity |
|---|---|---|---|---|---|---|---|
| rectangular | laurent | 6 | 169 | 3.5e-05 | 0.594687 | 0.604439 | 0.0021 |
| circular | laurent | 6 | 113 | 1.2e-05 | 0.621662 | 0.630321 | 0.0049 |
| circular | li | 7 | 149 | 4.0e-02 | 0.595097 | 0.604435 | 0.0004 |
| rectangular | li | 7 | 225 | 5.6e-02 | 0.589040 | 0.599678 | 0.0003 |
| circular | laurent | 8 | 197 | 2.4e-02 | 0.592822 | 0.602917 | 0.0007 |

**The in-band FRACTION lands in 0.5997 .. 0.6044 on all five -- a 0.8 %
spread -- and the uniformity in 0.0003 .. 0.0049 on all five.**  A truncation
artifact does not survive a change of factorization AND of order set AND of
`N`; a physical answer does.  That is why F2's claim is stated as
"approximately 0.60 in-band and a uniformity collapse of two to three decades"
rather than as `n_orders = 6`'s third digit.

Also tried and ineffective: `symmetry=False` (the cell is not centro-symmetric,
so the even-parity path was never engaged anyway), and jittering the seven slice
thicknesses by 1e-03 and 1e-02 relative to break a suspected
identical-thickness degeneracy -- both still unstable.

The solver's own message names the regime exactly: *"a near-degenerate
layer<->region mode-match at a measure-zero period / n_orders coincidence,
common at very large period / low index contrast"*.  That is this structure:
`period/lambda = 86.8` means every retained order has `kz/k0` within 0.001 of 1,
so the region modes form one enormous near-degenerate cluster and the interface
mode-match matrices lose rank as more of them are retained.  It gets worse with
`n_orders`, not better, which is why no truncation above the wall exists.

**`stabilize=True` does not rescue it and is not silently used.**  Where the
guard fires, `solve_orders(on_unstable='stabilize')` retries with the library's
consensus search and RECORDS `stabilized=True`; where it retries into a lower
truncation, the result is a DIFFERENT truncation wearing the requested label,
which is exactly what must not be averaged (S3.3).  At `n_orders >= 7` even the
consensus search fails outright (`every n_orders in the window ... tripped the
energy guard`).

### 3.3 The angle x polarization sweep, and why the production table is one node

Run at every rung; `n_stabilized` counts nodes that fell back:

| `n_orders` | nodes | `n_stabilized` | worst closure | worst RELATIVE efficiency spread across the cone | worst phase spread |
|---|---|---|---|---|---|
| 4 | 17 | **0** | 1.3e-02 | **4.546e-04** | **2.364e-04 rad** |
| 5 | 17 | 2 | 1.4e-02 | 1.694e+00 | 8.598e-01 rad |
| 6 | 17 | 5 | 4.0e-02 | 2.616e+00 | 6.848e-01 rad |

The `n_orders = 5` and 6 "angular spreads" are **not angular**: they are the
truncation switching under the stability guard at 2 and 5 of the 17 nodes.  The
one truncation that solves cleanly at every node reports the cone as flat to
`4.5e-04` relative and `2.4e-04` rad -- **four decades below the truncation
uncertainty of S3.1**.

That measurement is what justifies the production table's `n_theta = n_phi = 1`:
collapsing the cone is not an approximation being tolerated, it is the REMOVAL
OF A CONFOUND.  One clean solve at the highest truncation beats an average that
mixes truncations.  The angle-resolved table remains buildable (and is the
measurement of record above) by raising `n_theta` / `n_phi` in the spec.

### 3.4 Cache key discipline

Same discipline as `_d121_common._chain_a_key`, for the same reason (defect
D6): the filename is a digest of a key dict that is ALSO stored inside the
`.npz` and CHECKED on load, so a file renamed onto a matching hash is refused.
The key covers the schema salt, `lumenairy.__version__`, **a content hash of
every `lumenairy/**/*.py`**, a content hash of the builder module, the full
structure record (including a content hash of the level map, `n_doe`,
`relief_sign`, `n_levels`, period, wavelength), the order set, `n_orders`,
`formulation`, `truncation`, `mount`, `cell_upsample`, and **every theta and phi
node at full float repr**.  Pinned by tests in both directions (a change to any
of them re-keys; a foreign key is refused).

### 3.5 What the A/B artifacts are keyed to -- recorded, not papered over

`validation/pipeline/doe_rcwa.py` was edited between the two A/B arms (the
truncation and angle-grid defaults were set from S3.1/S3.3, and the local
imports were split after the WSL parity run caught a portability defect --
S7.1).  Every `*.py` under `validation/pipeline` is content-hashed into every
artifact key, so the two arms' checkpoints carry different
`pipeline_source_sha256` values and a RESUME of either would recompute it.

That does **not** touch the comparison, and the reason is worth one sentence:
the only module that changed is the RCWA decomposer, which arm A never calls,
and every module both arms share (`sources.py`, `driver.py`, `artifacts.py`,
`spec.py`, `metrics.py`) is byte-identical between them.  So the physics of
both arms ran under one driver; it is only the resume KEY that moved.
`BUILD_PIPELINE` S6.5 records the same situation for its 32-order run, and the
same disposition applies: the mechanism is working as designed (the driver is
an input to its artifacts), and re-running is available at the cost of the
wall in S6.

---

## 4. VALIDATING THE INSTRUMENT -- three oracles and one kill

An RCWA table that disagrees with the scalar model by 33 % is a claim that has
to be defended against the possibility that the RCWA model is simply wrong.
Four checks, three of which are not RCWA:

1. **The scalar arm is byte-exact.**  `doe_rcwa_table.scalar_table` +
   `design_order_set` reproduce `_d121_common.order_table` -- the very function
   being replaced -- to **1.130e-16** on all 32 complex amplitudes, with
   identical order indices.  So the two sides of every comparison below share
   their reference.
2. **A uniform cell against a closed form.**  A cell at one phase level is a
   plain slab, and its zeroth-order transmission must be the analytic
   Airy/Fabry-Perot amplitude.  It is, to ~1e-14, at three thicknesses.  This
   pins the layer stacking, the half-space handling, the eps-vs-index
   convention and the phase reference at once.  **It also corrected a
   convention**: the solver's transmitted amplitude carries the FULL optical
   path `k0 n d`, not the excess over an air leg `k0 (n-1) d`; the first
   version of this test subtracted the leg and missed by 1.7576 rad = `k0 d`
   exactly.  For the order table it is a global piston (every order crosses the
   same stack), which is why S5 reports piston-removed phase deltas.
3. **The thin-element limit, as a TREND rather than a level.**  The two knobs
   cannot both be relaxed on a full-wave element -- the relief that realises
   `2 pi` is `lambda/(n-1)`, so a shallow relief demands a high index (strong
   walls) and a low index demands a deep relief (strong propagation).
   Weakening the PHASE relaxes both.  On a 20-wavelength-period checkerboard of
   fused silica, the RCWA-vs-scalar SPLITTING error collapses two decades as
   the phase step weakens:

   | phase step | pi | pi/2 | pi/4 | pi/8 | pi/16 |
   |---|---|---|---|---|---|
   | worst splitting error | 0.164 | 0.148 | 0.0285 | 0.0068 | **0.0017** |

   Scored on the SPLITTING RATIO `eta / sum T`, not raw efficiency: a
   unit-modulus phase screen has no Fresnel reflection at all, so it can never
   agree with a real dielectric on throughput.  Separating the two is what
   makes S5's headline a statement about redistribution rather than about loss.
4. **The kill.**  See below.

### 4.1 The silent-wrong defect this build found

`_eps_convolution_2d`'s pixel contract makes `eps_cell[j, i]` the node
`(j Px/Sx, i Py/Sy)` -- **axis 0 is x**.  The Dammann cell is the other way
round: `_d121_common` reads `A[my + cy, mx + cx]`, so **axis 0 is y**.

Handing one to the other untransposed solves the TRANSPOSED structure.  It is
energy-clean (closure 1e-10), it converges, it produces plausible per-order
numbers, and it is wrong: on this design's 8-wide-by-4-tall order block it
reports `sum |a|^2 = 0.4465 .. 0.4270` across the ladder, against a
transposed-block scalar prediction of **0.4488** and a true **0.8851**.  It was
caught by the coarse-cell thin-element check (item 3 above) reading a factor of 2 where
the two models had to agree, and confirmed in one line against the scalar table.

Pinned twice: `test_relief_heights_transposes_into_the_solvers_axis_order` on
the array (with an assertion that the fixture cell is ASYMMETRIC, so the test
cannot go vacuous), and
`test_a_y_invariant_cell_puts_no_power_in_a_y_diffracted_order` on the physics
-- a stripe grating's `(m, n != 0)` orders are exactly decoupled and must carry
nothing, and a transposed cell radiates into `(0, n)` instead.

---

## 5. THE COMPARISON -- scalar FFT vs RCWA

At normal incidence, `n_orders = 6`, `relief_first`, on the SAME 128-pixel cell.

### 5.1 The headline

| quantity | scalar (thin-element) | RCWA | delta |
|---|---|---|---|
| in-band power, `sum|a|^2` over the 32 orders | **0.885056** | **0.594687** | **-0.290369 (-32.8 % rel.)** |
| in-band FRACTION of transmitted light | 0.885056 | **0.604439** | **-31.7 % rel.** |
| total transmitted, `sum T` (all retained orders) | 1.0 (lossless by construction) | 0.983866 | -1.6 % (relief Fresnel) |
| reflected, `sum R` | 0 (the model has none) | 0.016169 | |
| per-order efficiency | `0.027658 +- 0.000030` | `0.018584 +- 0.015072` | |
| weakest / strongest order | 0.027606 / 0.027732 | **1.36e-04 / 0.064589** | |
| uniformity (min/max) | **0.995435** | **0.002110** | |
| worst piston-removed phase delta | -- | **3.0937 rad** | |
| rms piston-removed phase delta | -- | **1.6528 rad** | |
| global piston (reference offset, not physical) | -- | 2.7582 rad | |
| max cross-polarized fraction | 0 (scalar) | **7.8e-06** | |

**What the ideal-thin-element assumption hides, in one sentence:** it hides
that a third of the in-band power is not there, and that the fan is not uniform
at all -- the weakest of the 32 orders carries `1.4e-04` against a design
target of `0.0277`, a factor of **203 down**, while the strongest carries
`0.0646`, a factor of **2.3 up**.  The three weakest are `(+2,-2)`, `(-4,-2)`
and `(+3,-2)`; the three strongest are `(+2,0)`, `(-3,+1)` and `(0,0)`.

**What it does NOT hide:** angle (S3.3: `4.5e-04` relative across the cone) and
polarization (`7.8e-06` cross-pol).  Both are genuinely negligible on this
design, and the reason is structural rather than lucky -- the DOE sits in
collimated space at `R = 703 m`, and a low-contrast dielectric relief at normal
incidence has almost no form birefringence.

### 5.2 The loss is not spread -- it is TWO LINES of the fan

The 32 orders are an 8-wide (`mx = -4..3`) by 4-tall (`my = -2..1`) block.
Summed along each line at `n_orders = 6`:

| `my` row | RCWA | scalar | ratio |
|---|---|---|---|
| **-2** | **0.010479** | 0.221229 | **0.0474** |
| -1 | 0.168885 | 0.221353 | 0.7630 |
| +0 | 0.210027 | 0.221131 | 0.9498 |
| +1 | 0.205295 | 0.221343 | 0.9275 |

| `mx` column | RCWA | scalar | ratio |
|---|---|---|---|
| **-4** | **0.003509** | 0.110677 | **0.0317** |
| -3 | 0.078440 | 0.110614 | 0.7091 |
| -2 | 0.091119 | 0.110673 | 0.8233 |
| -1 | 0.080897 | 0.110611 | 0.7314 |
| +0 | 0.115903 | 0.110581 | 1.0481 |
| +1 | 0.076185 | 0.110611 | 0.6888 |
| +2 | 0.088709 | 0.110697 | 0.8014 |
| +3 | 0.059925 | 0.110592 | 0.5419 |

**The two OUTERMOST lines of the block are extinguished** -- the `my = -2` row
keeps 4.7 % of its design power and the `mx = -4` column 3.2 % -- and

```text
interior block (my >= -1, mx >= -3, 21 orders)
    RCWA 0.580928    scalar 0.580874    ratio 1.0001    uniformity 0.2366
```

**the interior's TOTAL is preserved to 1e-04.**  So the missing ~0.29 did not
move inward; it left the 32-order block entirely, and what remains inside is
redistributed (uniformity 0.2366 against 0.9954) but not depleted in aggregate.

That is a physically coherent picture and it is what makes F2 credible as more
than a truncation artifact: the design's order block is DELIBERATELY OFF-CENTRE
(`-4..3` x `-2..1` about the specular), the thin-element optimiser can place
power asymmetrically at no cost because a phase screen has no preferred
direction, and the rigorous structure -- which does -- fails hardest exactly on
the two lines furthest from specular.  An artifact of truncation would not know
where the block's edges are.

### 5.3 The honesty line under the headline

`n_orders = 6` is the ceiling, not a converged answer.  The same numbers at the
three available rungs:

| | `n_orders = 4` | 5 | 6 |
|---|---|---|---|
| in-band `sum|a|^2` | 0.655676 | 0.617597 | 0.594687 |
| uniformity | 0.0039 | 0.0050 | 0.0021 |

and the `substrate_first` mounting reads 0.639702 at `n_orders = 5` against
`relief_first`'s 0.617597.  So the DIRECTION and the ORDER OF MAGNITUDE of F2
are solid -- every rung and both mountings agree that in-band power is
0.59-0.66 and that uniformity has collapsed by two to three decades -- while
the third digit is not.  **The claim defended here is "the thin-element table
overstates in-band power by roughly a third and its uniformity claim does not
survive", not any particular value.**

Two things that would move it further, both in the same direction (S1, A6):
real sidewall angle, corner rounding and overlay error are all absent, and all
of them scatter more power out of band.

---

## 6. THE PLUG-IN AND THE A/B

### 6.1 `design121_doe_rcwa`

Registered in the pipeline's decomposer registry.  It is
`decompose_design121_doe` with ONE thing changed -- where the complex order
amplitudes come from.  The chain-A launch, the group split, the period, the
order tilts, the library's own chief-ray frame centres snapped to the output
lattice, the exact skew-trace diagnostic, and every context key the `traced`
runner consumes are computed the SAME way, deliberately, so an A/B moves the
weights and nothing else.  The beam payload gains an `rcwa` record (the table
digest, the truncation, the averaging mode, and per-order `eta_mean`,
`xpol_max`, `coherence`, `amp_incoherent`).

**The averaging choice, documented because it is a modelling decision.**  The
pipeline's beam basis carries ONE complex weight per order, so an
angle-resolved table has to collapse.  The weight used is the beam-weighted
**COHERENT** mean over the incident angular spectrum, then the unpolarized mean
over the two incident linear states:

* COHERENT because the pipeline's beams are summed COHERENTLY -- what must
  survive the collapse is the FIELD each order contributes, not its power.
  Where the phase varies across the cone the coherent mean loses power, and
  that loss is real decoherence, not a modelling artifact.
* the INCOHERENT (power-preserving) alternative `sqrt(<|a|^2>)` is computed
  alongside, and `coherence = |coherent|^2 / incoherent^2` MEASURES how much
  the choice mattered.  On the production (collapsed) table it is exactly 1 by
  construction; on the angle-resolved `n_orders = 4` table it is 1 to
  `4.5e-04`, consistent with S3.3.
* **full ANGLE-RESOLVED decomposition -- one beam per (order, angle node), so
  nothing collapses -- is a STATED FUTURE REFINEMENT.**  It is not
  implementable against today's beam basis: `Beam` carries a weight, a frame
  centre and a payload but no angle, and the `traced` runner launches ONE
  congruence per beam from `payload['tilt']`.  Adding it means adding an angle
  to the basis and K x A chains -- a pipeline change, not a decomposer change.

**Placement.**  The table BUILDER lives in
`validation/repro_traced_carrier_121/` next to the scalar `order_table` it
replaces, NOT under `validation/pipeline/`, because every `*.py` under that
package is content-hashed into every artifact key -- an instrument under active
development would orphan every checkpoint in every workdir on each edit.
`validation/pipeline/doe_rcwa.py` is the thin, stable adapter, and
`sources.py`'s delta is seven lines (a comment plus one import for the
registration side effect) so the parallel worktree's edits merge cleanly.

### 6.2 The A/B

Two full pipeline runs over the 3-order acceptance triple `(0,0)`, `(-2,0)`,
`(-4,-2)`, `chain.kind='traced'` at `n_fine_cap = 8192`, `rs = 1`, one common
8192-square carrier at 1.2292 um, `leg.mode='crop'`.  The spec files are
byte-identical apart from `decompose.kind` and `decompose.params`.

**The weights, which are the only input that differs:**

| order | scalar `w` | RCWA `w` | `|w|^2` scalar | `|w|^2` RCWA | ratio | `d arg` |
|---|---|---|---|---|---|---|
| (0,0) | `-0.164839+0.020830j` | `-0.227501+0.113109j` | 0.027606 | 0.064550 | **2.3383** | -0.3357 |
| (-2,0) | `+0.165996-0.008284j` | `-0.132401+0.110667j` | 0.027623 | 0.029777 | **1.0780** | +2.4952 |
| (-4,-2) | `-0.075038+0.148637j` | `+0.014035+0.005678j` | 0.027724 | 0.000229 | **0.0083** | -1.6539 |

Note the third order: `(-4,-2)` is the CORNER of the block -- it sits on both
of the two lines S5.2 found extinguished -- and it is also the design's own
carrier order (the `.zmx` multi-config `PRAM 9 3 -4` / `PRAM 11 3 -2`).  The
rigorous table gives it **0.83 % of its designed power**.

**The frames:**

| frame | power scalar | power RCWA | `P_r/P_s` | EE3 scalar | EE3 RCWA | `dEE3` | FWHM scalar | FWHM RCWA |
|---|---|---|---|---|---|---|---|---|
| (0,0) | 1.8845232e-09 | 4.4065374e-09 | **2.338277** | 90.6068 | 90.6066 | -0.0002 pt | 3.400 um | 3.400 um |
| (-2,0) | 1.8856730e-09 | 2.0327391e-09 | **1.077991** | 90.4969 | 90.4988 | +0.0018 pt | 3.400 um | 3.400 um |
| (-4,-2) | 1.8923614e-09 | 1.5646234e-11 | **0.008268** | 89.9411 | 89.9411 | -0.0000 pt | 3.800 um | 3.800 um |

**The frame powers reproduce `|w_r|^2 / |w_s|^2` to six digits** (2.338277 vs
2.3383; 1.077991 vs 1.0780; 0.008268 vs 0.0083).  That is the control the A/B
needed: the pipeline is EXACTLY linear in the decomposer's weights and adds
nothing of its own, so the delta below is the DOE model and not the plumbing.
Shape is untouched -- EE3 moves by at most 0.0018 points and FWHM is identical
on all three frames -- because a scaled field has the same shape.

**Banner-class metrics, each arm against ITS OWN shipped per-order tile:**

| frame | `P/P_shipped` scalar | `P/P_shipped` RCWA | `dEE3` vs shipped, scalar | RCWA |
|---|---|---|---|---|
| (0,0) | 1.000006e+00 | 1.000002e+00 | 0.0001 pt | 0.0001 pt |
| (-2,0) | 9.999920e-01 | 1.000019e+00 | 0.0012 pt | 0.0006 pt |
| (-4,-2) | 1.000000e+00 | 1.000001e+00 | 0.0001 pt | 0.0001 pt |

```text
worst |P/P_shipped - 1|   scalar 8.023e-06   RCWA 1.907e-05   (bar 4e-05)   ok / ok
worst |dEE3| vs shipped   scalar 0.0012 pt   RCWA 0.0006 pt   (bar 0.1 pt)  ok / ok
FWHM vs shipped           identical on 3/3 frames, both arms
```

**Both arms hold every campaign bar**, which says the decomposer swap does not
disturb the pipeline's own accuracy at all -- as it should not, since it
changes three complex numbers.

### 6.3 THE DELTA -- and it is flagged, not reported as a result

```text
worst |P_RCWA / P_scalar - 1| = 1.338277   =  133.83 %
worst |dEE3|                  = 0.0018 points
```

**FLAG.  The energy delta is 67x past a 2 % line and 33458x past the campaign's
own 4e-05 energy bar.**  On the three carried orders one frame gains 134 %, one
gains 8 %, and one loses 99.2 %.  If these numbers were a measurement of the
design they would be a redesign trigger.

**They are not a measurement of the design, and the reason is S3.1.**  The
table they come from is unconverged at a truncation that is a hard ceiling,
and the deltas are the same size as the model uncertainty on the table itself
(the mounting alone moves in-band power 3.5 %, the ladder moves it 10 % between
its only three rungs, and the structure's material and relief are assumptions).
What the A/B establishes is:

* **the plumbing is exact** -- frames scale as `|w|^2` to six digits, shape is
  untouched, both arms hold every bar (S6.2);
* **the DOE model is where the entire delta lives**, and it is enormous -- so
  the ideal-thin-element assumption is not a second-order convenience on this
  design, it is load-bearing for the answer;
* **an accurate answer needs an instrument this design does not have.**  RCWA
  cannot converge a 86.8-wavelength period with sub-wavelength features, and
  the honest next step is a different method (a domain-decomposition or
  FDTD/FEM solve of one cell, or a measurement of the fabricated part), not a
  bigger `n_orders`.

### 6.4 Cost

```text
             total     chains   aggregate   leg   readout   decompose   peak wset
scalar arm  2234.5 s   1894.6      119.1    11.7    52.3      150.0      28.11 GB
RCWA arm    1047.0 s    885.6       87.8     9.7    51.2        6.9      28.23 GB
```

The chain-stage difference is box load, not the decomposer (the scalar arm ran
alongside the RCWA ladder).  The `decompose` stage is 150.0 s vs 6.9 s only
because chain A was uncached on the first arm and cached on the second: **the
whole RCWA table costs 2.3 s** at the production configuration (one clean solve
at `n_orders = 6`), and is then cached and keyed.  The rigorous decomposer is
not the expensive part of anything.

---

## 7. FILES

| file | what |
|---|---|
| `validation/repro_traced_carrier_121/doe_rcwa_table.py` | the instrument: structure reconstruction, the RCWA stack, the per-order scalar, the angle grid + quadrature, the convergence ladder, the cache and its key, the scalar comparison, and a `ladder` / `sweep` / `compare` CLI (which is what `build_table`'s own refusal message points at).  Design-AGNOSTIC (takes a level map), so the tests need no `.zmx`. |
| `validation/pipeline/doe_rcwa.py` | the `design121_doe_rcwa` decomposer + `rcwa_weights` (the weight source, callable without a spec and with an injectable structure, which is what makes it testable). |
| `validation/pipeline/sources.py` | **+7 lines**: a comment and the import that registers the new decomposer. |
| `validation/pipeline/specs/d121_3order_ab_scalar.json` | A/B arm A. |
| `validation/pipeline/specs/d121_3order_ab_rcwa.json` | A/B arm B -- identical except `decompose`. |
| `tests/unit/test_doe_rcwa.py` | 41 tests: structure + conventions, the analytic-slab oracle, closure, the y-invariance axis control, the thin-element trend, the convergence ladder, the angular quadrature, the averaging, the cache key discipline, the decomposer contract. |
| `docs/audits/BUILD_RCWA_DOE_TABLE_2026_08_12.md` | this note. |

---

### 7.1 Housekeeping

The cached RCWA tables land in `validation/repro_traced_carrier_121/` as
`_doe_rcwa_v1_<digest16>.npz`, which the existing `_*.npz` ignore rule already
covers; they are tiny (6.7 KB for a single-node table, 47 KB for the 17-node
3 x 8 angle sweep).  The two A/B workdirs are **2.7 GB** together under
`validation/pipeline/_work/`, already covered by that package's `.gitignore`.
The design-121 caches in the shared tree
(`D:/.../Lumenairy/validation/repro_traced_carrier_121`) and the `.zmx` /
design-study runner in `Reverse_Symmetric_ASM` are consumed READ-ONLY by
absolute path; nothing was written there.

### 7.2 GREEN

| check | result |
|---|---|
| `tests/unit/test_doe_rcwa.py` (Windows, py3.14.6, numpy 2.4.4) | **41 passed**, no xfail, no skip, ~7 s |
| WSL parity -- same file under `/home/travaj/lumen_venv`, a different BLAS, **and no design-121 tree at all** | **41 passed** |
| `test_doe_rcwa.py` + `test_pipeline.py` (WSL) | **81 passed** |
| `test_doe_rcwa.py` + `test_pipeline.py` + `test_carrier_field.py` (Windows) | **115 passed** -- the pipeline plumbing and the carrier primitives are unaffected |
| `ruff check lumenairy/ tests/unit/` (the project's own CI scope) | **All checks passed** |
| design-121 A/B, both arms, end to end | **completed** (S6) |

**The WSL mount earned its keep.**  Its first run failed 4 tests that passed on
Windows: `rcwa_weights` reached the table builder through an import that also
pulled in `_d121_common`, which READS THE DESIGN-STUDY RUNNER AT IMPORT TIME to
register the Sellmeier table -- so it raised `FileNotFoundError` on any machine
without the LOCAL-ONLY design tree, and Windows (where that tree exists) could
never see it.  The import is now split (`_import_table` / `_import_d121`) and
the weight source needs no design files at all.  That is exactly the class of
defect the parity mount exists for.

**What the 41 tests actually pin**, grouped, because a count is not evidence:

* **the structure and its two silently-wrong conventions** -- the level map
  round-trips and an unquantised or lossy mask is REFUSED rather than rounded;
  the axis transpose is pinned on the array AND on the physics; the
  phase-to-height sign is pinned; `n_levels` steps of `dz` is exactly one wave;
  four bad structure inputs are refused by name;
* **the analytic oracle** -- a uniform cell reproduces the closed-form Airy
  amplitude at three thicknesses, which pins the stacking, the half-spaces, the
  eps-vs-index convention and the phase reference;
* **energy** -- lossless closure, non-negative efficiencies, and an assertion
  that the fixture is a CLEAN solve rather than a stabilized one;
* **the thin-element limit** -- reproduced at weak phase, and the ERROR TREND
  across five phase depths (a defect that shifted the table would break the
  trend, not just the level);
* **convergence** -- the ladder's deltas shrink and land under a bar; the
  piston-removed phase movement is reported separately and is invariant under a
  global rotation, checked directly;
* **the angular machinery** -- Clenshaw-Curtis is exact on polynomials and
  beats the equal-count trapezoid by >100x on an analytic integrand; the grid
  puts `theta = 0` first and carries the disk Jacobian; the degenerate
  one-node grid takes unit weight instead of zero;
* **the averaging** -- coherent and incoherent agree exactly when the table is
  angle-flat, and `coherence` drops to exactly 0.5 on a constructed pi/2 phase
  spread (so the metric is not vacuous);
* **the cache key** -- round trip, a file that does not carry its own key is
  refused, and every one of `n_doe` / `relief_sign` / `n_levels` / the level
  map / `n_orders` / `thetas` / `phis` / `truncation` / `formulation` /
  `cell_upsample` re-keys;
* **the decomposer** -- registered under its own name and distinct from the
  scalar one, JSON-safe provenance, the two averaging arms ordered correctly,
  and both refusals (unknown averaging mode; missing table with
  `build_if_missing` false) fire before any RCWA is spent.

---

## 8. WHAT IS NOT CLAIMED

* **No converged rigorous table exists for this DOE, and none is claimed.**
  S3.1's wall is the headline caveat on everything else.  A converged answer
  needs `n_orders ~ 128` (a 132098-square eigenproblem) and the solver refuses
  above 6; the gap is not closable by tuning.
* **The structure is reconstructed, not read.**  Material, relief depth,
  mounting, wall profile and etch bias are all assumptions (S1).  A corrected
  material is one re-run -- `n_doe` is a parameter and is in the key -- but
  every number here moves with it.
* **The comparison is against the design's OWN scalar table, not against a
  measurement.**  No fabricated-part data, no vendor rigorous simulation and no
  metrology exist in the tree, so nothing here says which model matches the
  real part.  It says what the two MODELS disagree about.
* **Off-normal solves fall back.**  At `n_orders = 5` and 6 the angle grid has
  2 and 5 stabilized nodes.  They are recorded (`n_stabilized` in the table
  meta and in the beam payload) and are the reason the production table is one
  node.
* **The A/B is 3 orders, not 32.**  The 32-order fan is ~3.3 h per arm
  (`BUILD_PIPELINE` S6.1) and was not run.  The 3 orders carried are the
  acceptance triple, and one of them -- `(-4,-2)` -- happens to be the corner
  order S5.2 finds extinguished, so the A/B's 134 % headline is not a typical
  order.  The per-order table of S5 is where the full distribution lives.
* **The angle-independence result is design-121's, not the DOE's.**  It follows
  from the DOE sitting in collimated space (`R = 703 m`).  A DOE in converging
  space would sample a real cone and the angle-resolved table would then be
  load-bearing -- which is why the sweep machinery is built and kept rather
  than short-circuited.

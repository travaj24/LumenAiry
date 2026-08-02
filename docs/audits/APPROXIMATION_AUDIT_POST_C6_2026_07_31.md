# Post-C6 re-measurement of the traced-carrier approximation audit -- design 121, 2026-07-31

**What this is.**  `APPROXIMATION_AUDIT_TRACED_2026_07_30` measured every
non-null row of its ranked table against pinned HEAD `d2e60ca`, i.e. against a
tree carrying the OPEN niche-C6 defect.  Its own headline warning was that the
ONE row it re-controlled against C6 -- `_paraxial_group_r_out` -- collapsed by
97 % and changed sign (+5.97 -> -0.16 EE3 points), because the apparent gain
was the open defect being fed a badly-conditioned carrier and seen through a
proxy.  This document re-measures the rest against the LANDED C6, resolves the
on-axis ghost, settles the taper contradiction and closes the two loose ends.

**Headline.**  **Three of the six non-null rows are retractions, and one of
them reverses into a 77-point catastrophe.**  Against the landed C6, on a
bit-exact differential floor (`relL2` 0.000e+00) with two live positive
controls:

| row | HEAD dEE3 | **C6 dEE3** | verdict |
|---|---|---|---|
| `carrier_reference` sphere -> parabola | -30.15 | **-38.49** | confirmed, 1.3x larger -- do not touch |
| `remap_sampling` full -> lattice | -17.73 | **+0.0988** | **NOT A DEFECT -- LEAVE ALONE** (collapsed 99.4 %, sign flipped) |
| `_sphere_parab_conversion` taper OFF | +0.67 | **+1.41** | real, 2.1x LARGER -- and it is a RADIUS, not a convention (S2) |
| `fit_radius_beam_factor` 2.0 -> 3.0 | +0.66 | **-77.48** | **NOT A DEFECT -- LEAVE ALONE**; 2.0 is load-bearing |
| `ray_subsample` 4 -> 2 | +0.38 | **+0.3664** | survives unchanged |
| `_paraxial_group_r_out` -> traced sphere | +5.97 | -0.16 | already retracted; carried unchanged |

The prior audit's warning was correct and understated: **the SIGN of a
HEAD-measured row is not reliable, not merely its magnitude.**  Disabling C6 on
the C6 tree reproduces pinned HEAD's baseline (EE3 72.501) and its
`exit_power_above_nyquist` (7.5134e-04) to the digit, so every delta here is
attributable to C6 alone.

**The on-axis ghost is understood, the hypothesis on record is REFUTED, and
the remedy is a lever rather than a fix.**  The ghost is not "the order-6
forward-map fit cannot carry a degree-4 launch augmentation" -- raising that
order makes it **86x worse**.  It is the D1 fold: the tensor-Chebyshev fit of
the entrance->exit ray map is Newton-inverted far outside its own data support
and `amplitude_model='ray_density'` hands the spurious roots real amplitude.
The concentric hard NaN mask is documented as safe only while "the
unconstrained directions of the fit inherit the map's RADIAL SYMMETRY", and the
C6 launch augments every ray by `grad(a_fit)` of a general NON-radial
polynomial -- so C6 destroys that precondition on the one branch D1 left alone.
Routing the C6-engaged concentric fit through D1's weighted restriction takes
the ghost from 1.03e-03 of the input power to **exactly 0.000e+00**, but the
same switch INTRODUCES a 4.5e-03 lobe on a synthetic singlet at design 121's
own scale, so it ships **opt-in and default OFF** as
`REMAP_STATIONARY_PHASE_FIT_GUARD` (S3).

**Every knob on that ray fit is a conditioning lottery, and this is the most
important oracle-readiness finding in this document.**  `fit_radius_beam_factor`,
`_FIT_DISC_OUTSIDE_WEIGHT_REL`, the fit order, the restriction method and the
residual degree all have clean and dirty settings with no monotone structure,
and the direction of every one of them reverses between the on-axis and tilted
branches of the SAME design.  Two consequences:

* `_FIT_DISC_OUTSIDE_WEIGHT_REL = 1e-8` -- which the shipped TILTED path uses
  today -- sits at the bottom of a **~1-decade well** on design 121's
  production order (1e-10 reads 2.1e-04, 1e-8 reads 2.1e-08, 1e-6 reads
  1.7e-03), not the "middle of the plateau, ~4 decades clear of the fold on one
  side and ~2 of the cliff on the other" its own note claims from a low-NA
  fixture;
* `decentred_fit_poly_order = 14`, which the prior audit priced END TO END at
  **-0.005 EE3 points** ("already converged, marginally worse"), returns a
  field carrying **P/Pin 1.82 at order (-2,0) and 2.21 at (-4,0)** -- 82 % and
  121 % of manufactured energy.  The encircled-energy metric is blind to it
  because it all lands outside the 19.2 um readout tile.  **No EE-based number
  in either audit constrains the halo.**

**The taper contradiction is RESOLVED: it is a RADIUS, not a convention.**
Sweeping `r_safe` rather than removing the taper gives a monotone, saturating
response -- x0.5 costs **-41.62** EE3 points (and raises 2 fold-caustic
warnings), x2 gains **+1.4147**, and no taper at all gains **+1.4147**,
agreeing with x2 to four decimal places in every metric.  So the taper is doing
the same thing on axis and off; its radius is simply too small once the
congruence is tilted, and the beam sits in the mixed-convention annulus (onset
at 1.63-1.64 w on the last two planes, holding 0.5-0.6 % of the envelope power
-- 25x the "~2e-4" the docstring concluded from a plane list that omits them).
The docstring's validation is now labelled **on-axis only**, and its `w_beam`
guard, which tested `r_safe` when the taper starts at `0.75 r_safe` and so
could not detect this case at all, now tests the ONSET (warning-only; no field
moves).  The default was NOT flipped to `T == 1` even though design 121
measures that as optimal -- see S2.2 (S6 item 10).

**Loose end 2 is BOUNDED, in both directions.**  The exit ray map really does
change (`exit_power_above_nyquist` 7.5134e-04 -> 7.8996e-04, measured as a
same-process C6-on/C6-off pair, not live-vs-pinned), but the **fold-caustic
warning fires on NEITHER tree** at any of the sixteen chain runs made here,
under `simplefilter('always')` so de-duplication cannot hide it.  The detector
is not dead: it fires three times on `fit_radius_beam_factor=3.0` and twice on
`r_safe x0.5`.

**The degree non-monotonicity (loose end 1) is RESOLVED and is an instrument
artefact -- but not the one on record.**  It is not the oracle's band-limited
representation of `a` (the oracle's upsample factor is converged: 4 / 8 / 16
move the degree-4 reading by 0.4 %).  It is the probe's **2 %-of-peak amplitude
threshold**: raising it to 10 % REVERSES the ordering to degree 4 < 6 < 3,
restoring the order the model's own slope residual predicts, and drops
degree 4's reading 6x (0.0140 -> 0.0024 waves).  Degree 4's apparent penalty
lives entirely in the 2-10 %-of-peak skirt (S4.1).

---

## 0. Provenance, instruments, floors

### 0.1 The pins

The C6 tree is UNCOMMITTED on `fix/pmm-union-grid-conditioning` @ `d2e60ca`.
It is therefore pinned the way the previous audit pinned HEAD -- a frozen copy
outside the repo, with the sha256 of the two files actually imported printed by
every runner:

| tree | `elements/_lens_traced.py` | `propagators/carrier.py` |
|---|---|---|
| pinned HEAD `d2e60ca` | `957f00129f8b467c` | `2d30f1ed7beb3c7e` |
| **pinned C6** (`<scratch>/pin_live_c6`) | **`f06da6ab8e15ce2a`** | `2d30f1ed7beb3c7e` |
| working tree after this study | +158 lines (3 executable) | +73 lines (1 executable) |

The C6 pin is **byte-identical to the working tree as this study started**
(both `f06da6ab8e15ce2a`, 381 322 bytes), so the prior audit's "in-flight C6
snapshot" and the LANDED C6 are the same library: its S3.2 `R_out` control was
already a post-C6 control and is carried into S1 unchanged rather than re-run.

`carrier.py` is identical between the two PINS, so every chain-row difference in
this document is `_lens_traced.py`'s.  **Every measurement below was taken
against the pins, before this study's own edits**, and those edits are in turn
verified not to move any returned array at the shipped defaults (S3.6, S2.1).

### 0.2 The instruments

| instrument | what it measures | cost |
|---|---|---|
| `approx_post_c6.py` | the end-to-end chain rows of S1/S2 | ~6 min per row |
| `probe_ghost_c6.py` | halo metrics of ONE element call, branch x order x degree x `frbf` x weight sweeps | 1-4 s per case |
| `probe_ghost_locate.py` | where the ghost is, against the EXACT traced exit hull | ~1 min |
| `probe_ghost_tradeoff.py` | halo AND element-vs-oracle wavefront, together | ~17 s per pair |
| `probe_ghost_synthetic.py` | the same question on singlets with NO design-121 asset | ~2 min |
| `probe_c6_degree_oracle.py` | loose end 1: does the degree ORDERING move with the comparison? | ~25 s per grid point |
| `probe_c6_fitguard_verify.py` | `np.array_equal` byte-identity across the two pinned trees | ~2 min per tree |

`approx_post_c6.py` reuses `approx_common.py` VERBATIM -- same `Patch` /
`run_chain` / `metrics` / `field_diff`, the same complete shipped path (chain A
cached -> `TiltedCarrier(R, L, M)` for the DOE order -> the six post-DOE groups
-> the 7.7058 mm trailing leg -> the exact Bluestein readout), the same fixed
output lattice centred on the order's exact chief ray, and **no hand-off
plane**.  It differs only in the row list, in capturing WARNINGS per row under
`simplefilter('always')` (loose end 2 needs a controlled pair and Python
de-duplicates warnings per location), and in printing per-row sampling
adequacy.  It is a SEPARATE file so the prior audit's own reproduction stays
byte-identical.

Chain configuration, identical to the prior audit: `RN=1024`
(`dx = 51.2334 um` at the DOE), `ray_subsample=4`, `n_fine_cap=12288`,
`window_factor=4.0`, `N_out=192`, `dx_out=0.1 um`, `final_leg='exact'`, order
`(-4,-2)`.

Element configuration (every halo measurement): design 121's LAST group, the
chain's own captured input / carrier / decentre, `ray_subsample=4`,
`fit_radius_beam_factor=2.0`, `remap_sampling='full'`,
`amplitude_model='ray_density'`, `preserve_input_phase='remap'`, N=1024,
`dx = 33.2112 um`.

### 0.3 Differential floors -- established, not assumed

Every instrument in this document ran a NULL intervention before any delta was
quoted, and all of them are bit-exact:

| instrument | null | reading |
|---|---|---|
| `approx_post_c6.py` chain | identity monkeypatch of `_sphere_parab_conversion` | (S1) |
| `probe_ghost_c6.py` element | two identical shipped runs | `array_equal = True`, `max|dE| = 0.000e+00`, on all four orders |
| `probe_ghost_synthetic.py` | two identical runs per fixture | `array_equal = True`, all six fixtures |
| `probe_c6_element.py` (loose end 1) | two identical flag-OFF runs | `array_equal = True` |

### 0.4 Sampling adequacy

Stated for every wave measurement as the **amplitude-weighted wrapped
nearest-neighbour-step p99.9 against pi**, never a max (the prior audit's own
artefact 5(a), and `DIAG_LAST_GROUP_DECENTRE` artefact 4, are both the
consequence of quoting a max or an area-weighted percentile over a skirt).

| where | measured | limit |
|---|---|---|
| element-vs-oracle patch, C6 ON, order (-4,-2) | **0.0046 rad** | pi = 3.1416 |
| element-vs-oracle patch, C6 OFF, order (-4,-2) | 0.2342 rad | pi |
| element-vs-oracle patch, on axis, C6 ON | 0.0007-0.0040 rad | pi |
| synthetic free-leg fixture, all degrees | 0.0000-0.0969 rad | pi |
| chain last group, exit power above the retrace-grid Nyquist | **7.8996e-04** | refusal threshold 1e-2 |

The production leg runs **12.7x inside its own `on_tilt_exact_grid` guard**, so
every end-to-end number here inherits that margin.  The COARSE co-moving grid
remains 9-20x short of `lambda/(2 NA_exit)` and is never quoted as an absolute
wavefront figure, exactly as in the prior audit.

---

## 1. THE RE-MEASURED TABLE

`ORD=-4,-2 SET=p1 LUMEN_PIN=<pin_c6> python approx_post_c6.py`, one process,
one lattice, ~5.5 min per row.  Raw log `_postc6_p1_-4-2.txt`.

### 1.0 The instrument is alive and the floor is bit-exact

| row | EE3 % | dEE3 | relL2 | verdict |
|---|---|---|---|---|
| BASELINE (C6, shipping defaults) | 87.771 | -- | 0.000e+00 | matches the prior audit's C6 snapshot **87.771** exactly |
| NULL identity patch | 87.771 | **+0.0000** | **0.000e+00** | bit-exact differential floor |
| CONTROL `TILTED_CARRIER_EXACT_EIKONAL=False` (C5) | 67.451 | **-20.32** | 7.929e-01 | live positive control (it read -8.07 at HEAD; the C6 defect was masking it) |
| CONTROL `REMAP_STATIONARY_PHASE_LAUNCH=False` (C6) | **72.501** | **-15.27** | 1.884 | reproduces pinned HEAD's baseline **72.501 to the digit** |

The last row is the strongest cross-check in this document: disabling C6 on the
C6 tree reproduces the pinned-HEAD baseline EE3 (72.501) AND its
`exit_power_above_nyquist` (7.5134e-04 against HEAD's 7.513e-04) exactly, so
the two trees differ in nothing else that this measurement can see, and every
HEAD-vs-C6 delta below is attributable to C6 alone.

### 1.1 The re-measurement

`dEE3` in points against each tree's OWN baseline (HEAD 72.501, C6 87.771).

| row | HEAD dEE3 | **C6 dEE3** | C6 relL2 | change | verdict |
|---|---|---|---|---|---|
| `carrier_reference` `'sphere'` -> `'parabola'` | -30.15 | **-38.49** | 1.115 | 1.3x LARGER | **CONFIRMED, and worse.**  Not a taper or a band limit -- the difference between the exact sphere and its paraxial parabola at design 121's carrier NA.  Do not touch. |
| `remap_sampling` `'full'` -> `'lattice'` | -17.73 | **+0.0988** | 2.217e-02 | **collapsed 99.4 %, sign flipped** | **NOT A DEFECT -- LEAVE ALONE.**  A `_paraxial_group_r_out`-class retraction: the ray-lattice residual sampling was worth -17.7 points only because the `grad(W)` launch was sampling the residual at the wrong foot.  With C6 landed the library's own signature default (`'lattice'`) is marginally BETTER than the chain's `'full'` override. |
| `_sphere_parab_conversion` taper OFF (`T == 1`) | +0.67 | **+1.41** | 6.649e-02 | **2.1x LARGER** | **REAL, and it grew.**  See S2. |
| `fit_radius_beam_factor` 2.0 -> 3.0 | +0.66 | **-77.48** | 1.145 | **catastrophic reversal** | **NOT A DEFECT -- LEAVE ALONE, and 2.0 is load-bearing.**  EE3 10.290 %, `P_tile` -22.2 points, and the only row in the batch that raises a **fold-caustic warning (x3)**.  Confirmed independently at element level: `frbf` 2.5+ puts 1.0-1.1e-02 of the input power beyond 4 mm on this order (S3.5). |
| `ray_subsample` 4 -> 2 | +0.38 | **+0.3664** | 3.986e-02 | unchanged | **SURVIVES.**  The only non-null row of the prior audit that measures the same post-C6.  A genuine, if small, discretisation cost; `dEE6` +0.51, `dPtile` +0.50. |
| `_paraxial_group_r_out` -> traced exit sphere | +5.97 | **-0.16** | 2.33e-02 | (already retracted) | **NOT A DEFECT -- LEAVE ALONE.**  Carried unchanged from the prior audit's S3.2, which was already a post-C6 control (S0.1). |

**Three of the six non-null rows are retractions.**  Two collapse to nothing
(`remap_sampling`, `_paraxial_group_r_out`) and one reverses into a 77-point
catastrophe (`fit_radius_beam_factor`).  Only `ray_subsample` reads the same,
and `carrier_reference` and the taper read LARGER.  The prior audit's
methodological warning was correct and, if anything, understated: **the sign of
a HEAD-measured row is not reliable, not merely its magnitude.**

### 1.2 What the retractions have in common

`remap_sampling`, `fit_radius_beam_factor` and `_paraxial_group_r_out` all
perturb the conditioning of the residual transport or the ray fit -- exactly
the quantities C6's stationary-phase launch couples to.  `carrier_reference`,
the taper and `ray_subsample` do not: they change the reference convention, a
phase, and the ray discretisation respectively, and all three survive.  That is
a usable rule for the next campaign: **a row that perturbs the conditioning of
a fit measured against a tree with an open conditioning defect is measuring the
defect.**

---

## 2. The taper contradiction -- RESOLVED: it is a RADIUS, not a convention

`ORD=-4,-2 SET=taper LUMEN_PIN=<pin_c6> python approx_post_c6.py`.  Raw log
`_postc6_taper_-4-2.txt`.

The contradiction to settle: removing `_sphere_parab_conversion`'s `cos^2`
taper IMPROVES the tilted order, against a docstring recording that
`r_safe*1.5` and `r_safe=inf` "reproduce the shipping design-121 result to the
digit"; while a prior probe found the taper's radius must not be REDUCED
(halving it moves 0.13 of the field energy into a ring caustic).  So the taper
looked like it was doing something right on axis and something wrong off axis.

**It is doing the same thing in both places, at a radius that is too small once
the congruence is tilted.**  Sweeping the radius rather than removing the
taper:

| intervention | EE3 % | dEE3 | dEE6 | dPtile | relL2 | `P>nyq` | fold warnings |
|---|---|---|---|---|---|---|---|
| BASELINE (C6) | 87.771 | -- | -- | -- | 0.000e+00 | 7.8996e-04 | 0 |
| NULL identity patch | 87.771 | +0.0000 | +0.0000 | +0.00000 | **0.000e+00** | 7.8996e-04 | 0 |
| `r_safe` x **0.5** | 46.151 | **-41.6193** | -25.4461 | -23.06118 | 1.823 | 7.2901e-04 | **2** |
| `r_safe` x **2** | 89.186 | **+1.4147** | +0.5670 | +0.49576 | 6.649e-02 | 8.3409e-04 | 0 |
| taper OFF (`T == 1`) | 89.186 | **+1.4147** | +0.5670 | +0.49575 | 6.649e-02 | 8.3411e-04 | 0 |
| `_tilt_exactness_phase` taper OFF | 87.772 | **+0.0008** | +0.0008 | +0.00072 | 2.830e-03 | 7.9062e-04 | 0 |

Three readings, and they settle it:

1. **`r_safe` x2 and NO TAPER agree to four decimal places** in EE3, EE6,
   `P_tile` and `exit_power_above_nyquist` (8.3409e-04 vs 8.3411e-04).  At
   twice the radius the taper no longer touches anything the result depends on.
   The response is **monotone and saturating**, so the effect is entirely
   WHERE the taper acts, not WHAT it does.  There is no "convention" component
   to find.
2. **Halving the radius costs 41.6 EE3 points** and is the only row besides
   `fit_radius_beam_factor=3.0` to raise a fold-caustic warning.  The prior
   probe's ring caustic is confirmed and is much larger than it looked.
3. The **tilt-exactness taper stays a measured NULL post-C6** (+0.0008 points,
   `relL2` 2.83e-03, against -0.0003 at HEAD), which also nulls its
   coarse/fine `r_safe` mismatch.  That row of the prior audit survives.

**The geometry says the same thing.**  The taper onset `0.75 r_safe` sits at
**1.64 w and 1.63 w on the last two planes**, with **5.0e-03 and 5.7e-03 of
the envelope power beyond it** (prior audit S6).  The beam is in the
mixed-convention annulus, and the amount of power there is **25x** the
"at most ~2e-4 of the power ever sees a mixed convention" the docstring
concluded.

### 2.1 What was wrong with the docstring, and what was corrected

| claim | status |
|---|---|
| "reproduces the shipping design-121 result to the digit" at `r_safe*1.5` and `inf` | TRUE **on axis** -- every metric quoted (3.650 um / 87.3 / 99.3 at plane; 3.550 um / EE3 89.57 / EE6 99.26 at focus) is an on-axis one.  Now labelled as such. |
| "the taper onset sits at 2.73 w, 3.60 w and 2.07 w" | INCOMPLETE -- the plane list omits the last two, where it is 1.64 w and 1.63 w. |
| "at most ~2e-4 of the power ever sees a mixed convention" | **RETRACTED for tilted congruences**: 5.0e-03 and 5.7e-03. |
| reason (i), the conversion is POINTWISE so a wider taper is smoother not rougher | **CONFIRMED** -- it is exactly why `T == 1` wins. |
| the `r_safe < 2*w` warning is "a validity flag for OTHER geometries, not a known design-121 defect" | **WRONG on both halves**, and the guard could not have detected it -- see below. |

**The guard was under-triggering by construction, and is fixed.**  The taper
rolls off from `0.75 r_safe`, but the warning tested `r_safe < 2 w`.  Design
121's last two planes sit at `r_safe` = **2.18 w** -- clear of that threshold,
so no warning -- while the ONSET is at 1.63 w, inside the beam, costing 1.41
EE3 points in silence.  The threshold now tests the onset
(`0.75 * r_safe < 2 * w_beam`), which is the condition the warning's own
message describes.  **Warning-only: the returned array is unchanged, so no
field anywhere moves.**

### 2.2 What was NOT done, and why

Design 121 measures the optimum as **no taper at all**, at both tilts (on axis
`T == 1` is a documented no-op; off axis it is +1.41 points).  The default was
NOT flipped, because the same docstring records that "the untapered swap breaks
a coarse chain" -- a configuration not re-measured here.  A one-design monotone
sweep is not grounds for deleting a guard whose failure case is documented
elsewhere, and the `r_safe x0.5` row is a standing demonstration that this
radius has a 41-point cliff on the other side.  The counter-evidence is,
however, a single sentence with no reproduction attached; re-deriving it is the
cheapest way to unlock 1.4 points on every tilted order (S6 item 10).

---

## 3. THE ON-AXIS GHOST

`probe_ghost_c6.py`, `probe_ghost_locate.py`, `probe_ghost_tradeoff.py`,
`probe_ghost_synthetic.py`.

### 3.1 Reproduced, and quantified with a halo metric

The C6 note records the ghost as "0.103 % of the input power ... exit power
0.9959 -> 0.9970" on the ON-AXIS call at residual degree 4.  Reproduced
exactly, on the last group's own element call, with halo metrics rather than
EE3 (`gN` = returned power beyond N mm from the TRACED exit chief ray, over
input power; `amax4` = largest `|E|` beyond 4 mm over the peak; `r_rms` = the
power-weighted second moment of the whole returned field; `core` = the same
restricted to r < 1 mm):

| config | P/Pin | g1 | g2 | g4 | g8 | amax4 | r_rms (mm) | core (mm) |
|---|---|---|---|---|---|---|---|---|
| C6 OFF | 0.99590 | 2.396e-01 | 4.242e-03 | 3.61e-11 | 0 | 1.40e-05 | 0.8422 | 0.6201 |
| **C6 deg4 (SHIPPED)** | **0.99701** | 2.377e-01 | 4.648e-03 | **1.034e-03** | 0 | **3.30e-01** | **0.8638** | 0.6204 |

`P/Pin` 0.99590 -> 0.99701 is the recorded 0.103 %, to the digit.  The metric
that matters for an oracle is the last three columns: **1.03e-03 of the input
power at 33 % of the peak amplitude**, moving the field's second moment by
2.6 % -- while the CORE second moment moves by 0.0003 mm, i.e. **the spot does
not move at all**.  No encircled-energy number at 3 um can see any of this, and
the chain's 19.2 um readout tile is 300x too small to contain it.

### 3.2 Where it is, and what reaches there

`probe_ghost_locate.py` traces the a_fit congruence EXACTLY (the library's own
`_fit_residual_eikonal` model, handed to `wfe_probe_remap.trace_total`, so
nothing is reimplemented) over the whole launch square, and measures the traced
exit hull:

| launched from | exit radius max | p99.9 |
|---|---|---|
| the ILLUMINATED pupil, `r <= 2 w` (= the ray-fit disc) | **2.358 mm** | 2.357 mm |
| `r <= 3 w` | 3.681 mm | 3.680 mm |
| the whole launch square (to r ~ 17 mm, `|E_in|` ~ e^-59) | 9.549 mm | 9.549 mm |

The ghost is an **annulus at 6.298-7.216 mm exit radius, peaking at 6.675 mm**.
So it sits 2.8x beyond where the entire illuminated pupil lands, in a region
reachable only by launch radii of ~16-18 mm where the input amplitude is
numerically zero -- and it is handed 33 % of the peak amplitude.  That is the
fitted forward map being Newton-inverted, and given ray-density amplitude,
where its own data says nothing lands.

### 3.3 The recorded hypothesis is REFUTED

The hypothesis on record is that "the order-6 forward-map fit [is] unable to
carry a degree-4 launch augmentation".  If that were the mechanism, more terms
would help.  Measured on the same call, degree 4:

| restriction | fit order | P/Pin | g4 | amax4 | r_rms (mm) | fold warnings |
|---|---|---|---|---|---|---|
| hard NaN mask | 6 (SHIPPED) | 0.99701 | 1.034e-03 | 3.30e-01 | 0.8638 | 0 |
| hard NaN mask | **10** | **1.08514** | **8.917e-02** | 9.30e-01 | 2.4192 | 2 |
| hard NaN mask | 14 | 0.99773 | 1.419e-03 | 4.05e-01 | 0.8927 | 1 |
| D1 weighted | 6 | 0.99605 | 7.450e-05 | 1.12e-01 | 0.8388 | 0 |
| **D1 weighted** | **10** | **0.99598** | **0.000e+00** | **0.00e+00** | **0.8371** | 0 |

Raising the order on the hard-mask branch makes it **86x worse and gains 8.5 %
of the input power**.  That is what an unconstrained extrapolation does when
given more free directions; it is not what an under-resolved fit does.  The
hypothesis is refuted and the note that carries it has been corrected in the
source.

### 3.4 The actual mechanism -- and it is already in the file

`_FIT_DISC_OUTSIDE_WEIGHT_REL`'s note (`_lens_traced.py:1190-1263`) states the
precondition under which the concentric hard sample mask is safe:

> ... that is safe only while the retained disc is CONCENTRIC with the
> tensor-Chebyshev basis's own domain (the launch square).  **Then the
> unconstrained directions of the fit inherit the map's radial symmetry, the
> extrapolation outside the disc stays MONOTONE, and the Newton inversion
> cannot find a second root.**

`REMAP_STATIONARY_PHASE_LAUNCH` augments every launch direction by
`grad(a_fit)` where `a_fit` is a general degree-4 polynomial fitted to the
MEASURED input residual -- which has no radial symmetry.  **C6 destroys the
stated precondition, on the one branch D1 left alone.**  Three independent
confirmations:

1. setting `_FIT_DISC_OUTSIDE_WEIGHT_REL = 0` on the forced-weighted branch --
   documented as the exact restoration of the hard NaN mask -- reproduces the
   shipped ghost to every digit (`g4` 1.034e-03, `P/Pin` 0.99701);
2. the ghost tracks the FIT DOMAIN: `fit_radius_beam_factor` 1.5 -> 6.45e-04,
   2.0 -> 1.03e-03, and 2.5 / 3.0 / 4.0 -> 6.5e-11 (at 2.5 and above the
   beam-tied disc is superseded by the geometric `_CARRIER_FIT_RADIUS_FRAC`
   one, so the fit is constrained further out and the fold leaves the grid);
3. it tracks the out-of-disc WEIGHT, which is the only thing that pins the
   extrapolation.

### 3.5 Why the remedy ships OPT-IN and DEFAULT OFF

`REMAP_STATIONARY_PHASE_FIT_GUARD` (new, `_lens_traced.py`) routes a
C6-engaged CONCENTRIC ray fit through D1's weighted restriction and D7's raised
order -- the branch every tilted order already takes.  On design 121 it is a
clean win: `g4` 1.034e-03 -> exactly 0.  It is **not** shipped on, because the
axis it moves along is a conditioning lottery in both directions:

| sweep | on axis (concentric) | tilted (-4,-2) (off-centre) |
|---|---|---|
| `fit_radius_beam_factor` 1.5 / 2.0 / 2.5 / 3.0 / 4.0 (`g4`) | 6.5e-04 / **1.0e-03** / 6.5e-11 / 6.5e-11 / 6.5e-11 | 1.9e-04 / **2.1e-08** / 1.1e-02 / 1.0e-02 / 6.0e-03 |
| `_FIT_DISC_OUTSIDE_WEIGHT_REL` 0 / 1e-14 / 1e-10 / **1e-8** / 1e-6 / 1e-4 (`g4`) | 1.0e-03 / 7.7e-01 / 0 / **0** / 1.4e-02 / 1.4e-02 | 5.7e-02 / 1.3e-01 / 2.1e-04 / **2.1e-08** / 1.7e-03 / 2.1e-03 |
| residual degree 2 / 3 / 4 / 5 / 6, hard mask (`g4`) | 3.3e-11 / 3.3e-11 / **1.0e-03** / 1.3e-02 / 1.0e-12 | -- |
| residual degree 2 / 3 / 4 / 5 / 6, weighted o10 (`g4`) | 3.3e-11 / 3.3e-11 / **0** / 0 / 9.8e-03 | -- |

Read the two columns against each other: `fit_radius_beam_factor = 2.0` is the
WORST setting on axis and the BEST on the tilted order; the degree axis is
non-monotone on the hard mask (4 and 5 ghost, 2, 3 and 6 do not); and the
weighted branch that cures degree 4 leaves degree 6 -- the shipped
`_REMAP_RESID_DEGREE_CAP` -- at 9.8e-03.

The decisive adversarial test is `probe_ghost_synthetic.py`: singlets with a
converging carrier and the same `alpha (r/w)^4` residual the C6 unit tests use,
no design-121 asset anywhere, INCLUDING one built at design 121's own last-group
scale (N=1024, `dx` = 33 um, w = 3.1 mm, R_c = -21 mm, 20 mm aperture):

| fixture | branch | P/Pin | P beyond 3 w | max \|E\|/peak beyond 3 w |
|---|---|---|---|---|
| weak f/70 | hard mask | 0.99761 | 1.92e-08 | 1.20e-04 |
| | weighted | 0.99761 | 1.75e-08 | 1.18e-04 |
| medium, finer grid | hard mask | 0.99823 | 4.27e-08 | 1.59e-04 |
| | **weighted** | 0.99827 | **3.51e-05** | **1.02e-01** |
| **DESIGN-121-SCALE stand-in** | hard mask | 0.99828 | **0.000e+00** | 0.00e+00 |
| | **weighted** | 1.00281 | **4.526e-03** | **6.06e-01** |

On the stand-in built at design 121's own scale the direction REVERSES
completely: the hard mask is exactly clean and the weighted branch manufactures
0.45 % of the input power at 61 % of peak.  **Turning the guard on by default
would trade design 121's on-axis ghost for someone else's.**

It also costs accuracy where the hard-mask fit was exact.  Against the same
exact-ray oracle:

| where | C6 OFF | C6 ON, guard OFF (shipped) | C6 ON, guard ON |
|---|---|---|---|
| design 121, on-axis element WFE | 0.01098 waves | 0.01574 | **0.01629** (+3.5 %) |
| design 121, (-4,-2) element WFE | 0.06586 waves | 0.01401 | 0.01401 (byte-identical) |
| synthetic free leg (unit-test fixture) | 2.065e-02 waves | **2.34e-05** | 6.93e-04 (30x worse) |

The synthetic figure is the guard's worst case and is understood: that fixture's
augmented map is a CUBIC inside the fit disc, which order 6 fits exactly, while
outside it the residual model's RADIAL FREEZE continues `a` LINEARLY IN r --
not a polynomial at all -- so a weighted fit imports a shape the Chebyshev
basis cannot represent.  A real relay's in-disc fit error dominates that term,
which is why design 121 pays 3.5 % and the free leg pays 30x.

### 3.6 What shipped, and the byte-identity contract

`REMAP_STATIONARY_PHASE_FIT_GUARD = False` in `_lens_traced.py`, plus a 15-line
gate at the ray-fit restriction (`_c6_fit_guard`), a rewritten note on
`_REMAP_RESID_EIKONAL_DEGREE`'s hypothesis, and
`tests/unit/test_niche_c6_fit_guard.py` (12 tests).  Verified with
`np.array_equal` on the element's returned complex field, across the two pinned
trees, by `probe_c6_fitguard_verify.py`:

| claim | (0,0) | (-2,0) | (-4,0) | (-4,-2) |
|---|---|---|---|---|
| patched at its DEFAULT == pre-patch tree, C6 ON | **equal** | **equal** | **equal** | **equal** |
| patched == pre-patch, C6 OFF | equal | equal | equal | equal |
| guard forced OFF == pre-patch, C6 ON (the fail-before switch) | equal | equal | equal | equal |
| guard forced ON vs pre-patch (must differ on axis ONLY) | **differs**, max\|dE\| 5.85e-02 | equal | equal | equal |
| guard forced ON == the off-centre branch reached via a forced null decentre | equal | equal | equal | equal |

Row 1 is the shipping claim: **with the flag at its default the library is
byte-identical to the tree that existed before this study, on every order.**
Row 4 is the safety argument for turning it on: the guard acts only on the
CONCENTRIC branch, so C6's recovery on every tilted order cannot move.  Row 5
pins that the guard is a re-use of D1/D7 and not a new code path -- if it ever
fails, the D1/D7 evidence no longer covers the guard.

Rows 1-3 and 5 are also pinned as unit tests
(`tests/unit/test_niche_c6_fit_guard.py`, **13 tests**, on a synthetic free leg
with no design-121 asset), together with inertness under: C6 disengaged, no
engaged carrier, non-`remap` modes, a DECENTRED beam, no
`fit_radius_beam_factor`, and `_FIT_DISC_OUTSIDE_WEIGHT_REL = 0`.
`ruff check lumenairy/ tests/unit/` is clean and the existing 19-test C6 file
passes **unchanged** (no assertion was relaxed -- the default is
byte-identical).  The library diff is 110 lines, all of it the flag's note, a
15-line gate and two corrected notes.

### 3.7 The finding that outranks the ghost

The same sweeps price two SHIPPED settings that no EE-based measurement can
see.

**An effective ray-fit order of 14** (reached here by raising
`newton_poly_order`, which on the weighted branch yields the same
`_fit_poly_order = max(newton_poly_order, _dec_order)` = 14 that
`decentred_fit_poly_order=14` does at this configuration) was priced by the
prior audit END TO END at
**-0.005 EE3 points** and described as "already converged ... marginally
worse".  Measured as a halo on the element call, at the shipped residual
degree 4:

| order | config | P/Pin | g4 | amax4 |
|---|---|---|---|---|
| (-2,0) | shipped (10) | 0.99604 | 5.66e-06 | 2.85e-02 |
| (-2,0) | **14** | **1.81955** | **8.24e-01** | 9.94e-01 |
| (-4,0) | shipped (10) | 0.99591 | 2.62e-08 | 2.70e-04 |
| (-4,0) | **14** | **2.20714** | **1.21e+00** | 1.00e+00 |

i.e. a field carrying 82 % and 121 % of manufactured energy reads as a 0.005-point
EE3 change.  **EE3/EE6/EE12 and `P_tile` cannot constrain the halo at all**, and
every "converged" verdict in both audits that rests on them inherits that blind
spot.

**D7's order raise is load-bearing, not an optimisation.**  Forcing the
off-centre branch back to order 6 gives `P/Pin` 1.10096 / `g4` 1.05e-01 at
(-2,0) and 1.09551 / 9.96e-02 at (-4,0).

**And C6 introduces a weak ghost at (-2,0) even on the weighted branch**: `g4`
7.75e-09 (C6 OFF) -> 5.66e-06 (C6 ON) at 2.85e-02 of peak.  Orders (-4,0) and
(-4,-2) are clean (2.6e-08 and 2.1e-08 against 2.8e-08 and 2.5e-08 off).

---

## 4. The two loose ends

### 4.1 The NON-MONOTONE degree response -- RESOLVED, and it is the amplitude threshold

`probe_c6_degree_oracle.py`.  The note on record:

> the ELEMENT-vs-oracle column is NOT monotone in the degree (3 beats 4 and 6,
> 0.0074 against 0.0140, while the model's own slope residual keeps improving)
> ... The likely explanation is that the oracle's own band-limited
> representation of `a` and this fit's differ at high spatial frequency, i.e.
> it is a property of the comparison rather than of the field.  Not resolved.

**The conclusion is right and the stated cause is wrong.**  Three knobs of the
COMPARISON were moved with the field held fixed, order (-4,-2), `sigmaF` in
waves (all readings amplitude-weighted-p99.9 adequate, 0.0038-0.2356 rad
against pi):

| oracle `up` | amp thresh | patch half | OFF | deg 2 | deg 3 | deg 4 | deg 6 | best |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.02 | 96 | 0.06597 | 0.03895 | **0.00745** | 0.01422 | 0.01389 | 3 |
| 8 | 0.02 | 96 | 0.06586 | 0.03883 | **0.00740** | 0.01401 | 0.01362 | 3 |
| 16 | 0.02 | 96 | 0.06585 | 0.03882 | **0.00744** | 0.01397 | 0.01360 | 3 |
| 8 | **0.10** | 96 | 0.06286 | 0.03381 | 0.00456 | **0.00239** | 0.00391 | **4** |
| 8 | 0.02 | 64 | 0.06565 | 0.03862 | **0.00702** | 0.01304 | 0.01243 | 3 |

* the ORACLE's band-limited upsample is **converged**: 4 -> 8 -> 16 moves the
  degree-4 reading by 1.8 % and never changes the ordering.  The cause on
  record is refuted;
* the patch size is irrelevant;
* raising the AMPLITUDE THRESHOLD from 2 % to 10 % of peak **reverses the
  ordering** to 4 < 6 < 3 -- the order the model's own slope residual predicts
  (`grad(a - a_fit)` rms 2.99e-04 / 1.03e-04 / 8.47e-05 at degrees 3 / 4 / 6)
  -- and drops degree 4's reading **6x**, from 0.01401 to 0.00239 waves.

So degree 4's entire apparent penalty lives in the **2-10 %-of-peak skirt**,
15118 -> 9206 kept pixels.  This is the same artefact class as the prior
audit's own S8.5(a) and `DIAG_LAST_GROUP_DECENTRE` artefact 4: a statistic
dominated by skirt samples.  The mechanism is plausible and unforced -- the
residual fit's own bright mask is `_REMAP_RESID_BRIGHT_FRAC = 0.05`, so
degree 4's extra terms are constrained by the core and then EVALUATED in a
skirt the fit never saw, where a stiffer degree-3 model extrapolates better.

**A monotone CONTROL confirms it is not in the fit.**  On the synthetic
free-leg fixture of `tests/unit/test_niche_c6_stationary_phase_launch.py`,
whose oracle is ANALYTIC (no band-limited residual at all) and whose residual
is exactly `alpha (r/w)^4`:

| degree | OFF | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| sigma (waves) | 2.065e-02 | 1.406e-02 | 1.406e-02 | **2.344e-05** | 2.344e-05 | 2.344e-05 |

Perfectly ordered, with the cliff exactly at the degree that spans the residual
(2 and 3 agree because the residual is even, so degree 3's extra odd terms fit
zero).  **The shipped choice of degree 4 is vindicated** -- it is the best model
over the bright core, in the order the slope residual predicts.

### 4.2 The fold-caustic warning -- BOUNDED: the ray map really does change, the warning really does not fire

The note on record (prior audit S7.1) is that the shipped exact leg emitted
`amplitude_model='ray_density' detected a fold caustic (det J -> 0 or a sign
change)` on the LIVE C6 tree but not on pinned HEAD, and that this could not be
run as a controlled pair (three ablation batches had exhausted memory).  Its
second, independent trace was that `exit_power_above_nyquist` read 7.979e-04
live and 7.513e-04 pinned.

Run as a controlled pair here.  Every row of every batch executes inside
`warnings.catch_warnings(record=True)` with `simplefilter('always')`, so
Python's once-per-location de-duplication -- the prior audit's own artefact 8,
and the reason it filed this as unmeasured -- cannot hide anything:

| tree / configuration | EE3 % | `exit_power_above_nyquist` | `na_exit` | `na_measured` | fold warnings |
|---|---|---|---|---|---|
| pinned HEAD `d2e60ca`, baseline | 72.501 | **7.5134e-04** | 0.4049 | 0.5393 | **0** |
| pinned C6, `REMAP_STATIONARY_PHASE_LAUNCH=False` | 72.501 | **7.5134e-04** | 0.4049 | 0.5393 | **0** |
| pinned C6, baseline (C6 ON) | 87.771 | **7.8996e-04** | 0.4052 | 0.5397 | **0** |
| pinned C6, `fit_radius_beam_factor = 3.0` | 10.290 | 8.6030e-04 | -- | -- | **3** |

**Both halves of the note resolve, in opposite directions.**

* The exit ray map genuinely does change: `exit_power_above_nyquist` moves
  7.5134e-04 -> 7.8996e-04 (+5.1 %) and the measured exit NA 0.5393 -> 0.5397,
  and this is now a SAME-PROCESS, same-lattice pair differing only in the C6
  flag -- not a live-vs-pinned comparison.  Disabling C6 on the C6 tree
  reproduces HEAD's value to five digits, which is as tight a control as this
  instrument can give.
* The fold-caustic warning **does not fire on either tree**, at any of the
  sixteen chain runs made for this document.  The prior audit's live-tree
  sighting is not reproduced under control and should be read as either warning
  de-duplication across the many chains in one ablation process, or a
  work-in-progress snapshot -- it explicitly says the live tree "was not hashed
  at run time".

The fourth row is the reason to trust the negative: the detector is not dead.
It fires three times on `fit_radius_beam_factor = 3.0`, which is exactly the
row that loses 77 EE3 points, so a real fold in this chain does raise it.

**Bound, stated as a bound.**  This is one order, `(-4,-2)`, at one
configuration.  It says the shipped design-121 configuration does not fold at
the chain's exact leg on either tree; it does not say the map is fold-free
elsewhere -- and S3 shows the ELEMENT's fitted map folds badly on the on-axis
call, where this chain-level detector never runs.

---

## 5. The residual EE3 gap (priority 5) -- a negative result that narrows it

The chain reads EE3 87.771 % at this configuration against an exact-ray oracle
ceiling of 89.78 %.  The standing account attributes most of the gap to the
element's residual exit wavefront (0.014 waves) plus C5's -0.96 in the leg.
**The element's wavefront cannot be the explanation, in either reading.**

S4.1 shows that 0.014 waves is a skirt-dominated statistic: over the bright
core (>= 10 % of peak) the same field reads **0.00239 waves**.  Taking both as
Marechal:

| reading | sigma (waves) | implied Strehl deficit |
|---|---|---|
| 2 %-of-peak threshold | 0.01401 | 7.7e-03 |
| 10 %-of-peak threshold | 0.00239 | 2.3e-04 |

i.e. between 0.02 % and 0.8 % of peak intensity, against a gap of **2.0 EE3
points**.  Even the pessimistic reading is 2.5x too small, and the honest one
is 100x too small.  The remaining gap is therefore NOT the last group's exit
wavefront; it is somewhere else in the chain (the coarse-grid transport, the
earlier groups, or the leg), and localising it needs the per-leg instrument,
not a better element.  That is left open, but the element is now excluded.

---

## 6. What remains unmeasured

Stated explicitly, because the prior audit's biggest gap was a systematic
un-measured control and this document exists to close it.

1. **The prior audit's NULL rows were not re-measured post-C6, with one
   exception.**  The Fresnel kernel (`relL2` 3.56e-07), `n_fine_cap`
   12288 -> 16384, `newton_max_iters` 12 -> 60 and `_fourier_upsample_crop`'s
   parity trap are all still HEAD-only readings.  S1.2's rule says they are
   safe (none of them perturbs the conditioning of a fit), but that is an
   argument, not a measurement.  The tilt-exactness taper IS re-measured (S2).
2. **`newton_poly_order` 6 -> 10 is bit-identical only for TILTED orders.**
   The prior audit's row 12 ("BIT-IDENTICAL at 10 ... every group on a tilted
   congruence takes the off-centre branch, so this knob is inert here") is
   correct for order (-4,-2) and WRONG as a general statement: on the ON-AXIS
   call the same change gains 8.5 % of the input power (S3.3).  The end-to-end
   consequence on an on-axis chain run is not measured.
3. **Everything end to end is still ONE congruence,** order (-4,-2).  The halo
   measurements cover four orders at ELEMENT level ((0,0), (-2,0), (-4,0),
   (-4,-2)); the chain rows do not.  `propagate_traced_carrier_chain_multi`'s
   readout tiling and recombination remain untouched.
4. **The ghost is measured at `ray_subsample=4` only.**  The element-call
   captures are cached at that value and re-capturing at 2 or 8 costs a full
   chain run per order.  Since the ghost is a fit-conditioning effect and the
   coarse launch lattice pitch is `dx * ray_subsample`, an `rs` dependence is
   likely and is not bounded here.
5. **The chain-level cost of turning `REMAP_STATIONARY_PHASE_FIT_GUARD` on is
   not measured** -- on axis it would change the result (the flag exists
   because it does), and no on-axis chain pair was run.  It is default OFF, so
   this costs nothing today.
6. **`_FIT_DISC_OUTSIDE_WEIGHT_REL`'s ~1-decade well (S3.5) is measured on
   design 121's last group at two orders only.**  It is enough to retract the
   "4 decades clear of the fold, 2 of the cliff" claim in that constant's note
   for this regime; it is NOT enough to propose a different default.
7. **The structural fix for the ghost is not attempted.**  Bounding the Newton
   inverse to the traced samples' own support would kill every ghost in this
   document deterministically and is the obvious candidate, but it changes the
   returned field on every path (the legitimate skirt beyond the fit disc
   carries ~3e-4 of the power) and cannot be validated inside this pass.  The
   caustic-faithful alternative (`apply_real_lens_gbd` / `apply_real_lens_fga`)
   is a different propagator, not a knob.
8. **`caustic='multibranch'` still RAISES on this configuration**, so the
   KMAH/Maslov sum remains unavailable as a comparison, exactly as recorded in
   the prior audit.
9. **The on-axis 2.34-point floor** remains out of scope, as in all four prior
   studies.
10. **"The untapered swap breaks a coarse chain."**  That single sentence in
    `_sphere_parab_conversion`'s docstring is the only thing standing between
    design 121 and 1.41 EE3 points on every tilted order (S2.2), and it has no
    reproduction attached -- no design, no grid, no metric.  Re-deriving it is
    the cheapest open item in this document.  Until it is, the taper stays.

---

## 7. Artefacts found and killed in MY OWN instruments

Recorded because this project has now had ~25 artefacts pass as findings.

1. **I shipped the ghost fix ON by default, and was caught twice.**  The
   weighted restriction removed design 121's on-axis ghost exactly, every
   byte-identity claim held, and the tilted recovery was untouched -- so it
   looked finished.  The FIRST catch was the C6 author's own unit test
   (`test_stationary_phase_launch_removes_the_residual_model_error`), which
   went from 2.34e-05 to 6.93e-04 waves; the honest reading of that was "the
   fix costs accuracy on a fixture where the hard mask is exact", and I was
   about to relax the assertion from `0.02*off` to `0.05*off` and ship anyway.
   The SECOND catch was the adversarial synthetic sweep, which showed the
   weighted branch MANUFACTURING 4.5e-03 of the input power on a stand-in built
   at design 121's own scale where the hard mask is exactly clean.  That
   reversal is what turned the flag default-off and restored the test
   assertion untouched.  **A fix validated on one design, however cleanly,
   is a tuning.**
2. **My first synthetic fixtures were broken before the intervention.**  The
   first batch of singlets read `P/Pin` 1.12-1.28 with the C6 launch OFF --
   near-caustic geometries where `amplitude_model='ray_density'` is documented
   as unreliable.  Reading a guard-on/guard-off delta on those would have
   "measured" the guard on a configuration that was already invalid, in either
   direction.  Fixed by requiring the C6-OFF row to be clean before any delta
   is quoted, which is why the shipped fixture list puts the exit plane far
   from focus.
3. **A numerical coincidence I nearly wrote up as a smoking gun.**  The ghost
   annulus starts at 6.2975 mm and the ray-fit disc radius is 6.2578 mm -- a
   0.6 % agreement that reads as "the ghost begins exactly where the hard mask
   stops constraining the fit".  It is meaningless: those are the EXIT and
   ENTRANCE planes and the map's magnification is ~0.38, so exit 6.30 mm
   corresponds to entrance ~16.5 mm.  Caught by the traced-hull table in the
   same script, which is in `probe_ghost_locate.py` precisely so the two
   planes are never compared by eye.
4. **`probe_c6_energy.py` is STALE and will not run.**  It sets
   `LT._REMAP_RESID_TAPER_IN` / `_REMAP_RESID_TAPER_OUT`, which no longer exist
   (the multiplicative window was replaced by the radial freeze).  Left in
   place, but nothing in this document uses it; `probe_ghost_c6.py` is its
   replacement.
5. **A silent kwarg collision.**  The first mechanism sweep passed
   `fit_radius_beam_factor` both through `probe_c6_element.OPTS` and through
   the per-case overrides, which Python raised on.  Had the harness instead
   merged them silently in the wrong order, every row of the fit-domain sweep
   would have been run at the default and read as a flat null.

---

## 8. Reproduction

All commands from `validation/repro_traced_carrier_121/`.  The C6 pin is
created once with `cp -r lumenairy <scratch>/pin_live_c6/` (then remove
`__pycache__`); the HEAD pin is the prior audit's
`git archive d2e60ca lumenairy | tar -x -C <scratch>/pin_d2e60ca`.  Every
runner prints the sha256 of the two files it actually imported.

```bash
# S1 -- the re-measured table.  ~5.5 min per row, 9 rows.  Do NOT run chain
#       batches concurrently: each holds ~45 GB.
LUMEN_PIN=<pin_c6>   ORD=-4,-2 SET=p1    python approx_post_c6.py

# S2 -- the taper.  ~5.5 min per row, 6 rows.
LUMEN_PIN=<pin_c6>   ORD=-4,-2 SET=taper python approx_post_c6.py

# S4.2 -- the fold-caustic controlled pair (the C6 half is SET=p1's
#         'CONTROL C6 launch OFF' row, same process as its baseline).
LUMEN_PIN=<pin_head> ORD=-4,-2 SET=fold  python approx_post_c6.py

# S3 -- the ghost.  Element-level, seconds per case, no chain run.
ORDERS='0,0' DEG=4          python probe_ghost_c6.py        # branch x order
ORDERS='0,0' DEG=2,3,5,6    python probe_ghost_c6.py        # degree sweep
ORDERS='0,0' DEG=4 MODE=sweep  python probe_ghost_c6.py     # frbf + weight
ORDERS='-4,-2' DEG=4 MODE=sweep python probe_ghost_c6.py    # ... and tilted
ORDERS='0,0' DEG=4          python probe_ghost_locate.py    # vs the exit hull
ORDERS='0,0 -4,-2'          python probe_ghost_tradeoff.py  # halo vs WFE
                            python probe_ghost_synthetic.py # the reversal

# S3.6 -- byte identity across the two trees
PIN=<pin_c6> TAG=pre python probe_c6_fitguard_verify.py
             TAG=post python probe_c6_fitguard_verify.py
             CMP=pre,post python probe_c6_fitguard_verify.py

# S4.1 -- loose end 1
ORDERS='-4,-2' python probe_c6_degree_oracle.py
```

### Files added by this study

```text
validation/repro_traced_carrier_121/approx_post_c6.py          S1/S2/S4.2 chain rows
validation/repro_traced_carrier_121/probe_ghost_c6.py          halo metrics + sweeps
validation/repro_traced_carrier_121/probe_ghost_locate.py      ghost vs traced exit hull
validation/repro_traced_carrier_121/probe_ghost_tradeoff.py    halo vs wavefront
validation/repro_traced_carrier_121/probe_ghost_synthetic.py   the adversarial reversal
validation/repro_traced_carrier_121/probe_c6_degree_oracle.py  loose end 1
validation/repro_traced_carrier_121/probe_c6_fitguard_verify.py byte identity
tests/unit/test_niche_c6_fit_guard.py                          13 tests
docs/audits/APPROXIMATION_AUDIT_POST_C6_2026_07_31.md          this document
```

Raw logs: `_postc6_p1_-4-2.txt`, `_postc6_taper_-4-2.txt`,
`_postc6_fold_head.txt`, `_ghost_c6_00.txt`, `_ghost_synth.txt`,
`_c6_degree_oracle.txt`.

### Library changes

**No returned array changes at the shipped defaults** -- verified
`np.array_equal` on four orders (S3.6).  Diffed against the C6 pin, the study's
own delta is +158 lines in `_lens_traced.py` and +73 in `carrier.py`, of which
the ENTIRE executable content is three lines:

```text
_lens_traced.py:  REMAP_STATIONARY_PHASE_FIT_GUARD = False
                  _c6_fit_guard = (_resid_eik is not None
                                   and REMAP_STATIONARY_PHASE_FIT_GUARD)
                  if (_beam_fit_radius is not None
             -             and _beam_decentred
             +             and (_beam_decentred or _c6_fit_guard)
                           and _FIT_DISC_OUTSIDE_WEIGHT_REL > 0.0):
carrier.py:  -   if w_beam is not None and w_beam > 0.0 and r_safe < 2.0*w_beam:
             +   if (w_beam is not None and w_beam > 0.0
             +           and 0.75 * r_safe < 2.0 * w_beam):
```

Everything else is documentation.  In detail:

`lumenairy/elements/_lens_traced.py`:

* `REMAP_STATIONARY_PHASE_FIT_GUARD = False` -- new opt-in flag with its
  measurement table and the reason it is opt-in;
* a 15-line gate at the ray-fit restriction (`_c6_fit_guard`);
* `_REMAP_RESID_EIKONAL_DEGREE`'s note: the ghost mechanism CORRECTED (the
  order-6 hypothesis is refuted) and the degree non-monotonicity RESOLVED;
* `fit_radius_beam_factor`'s docstring: the -77.5 EE3-point measurement and
  the sign instability across the C6 fix;
* `remap_sampling`'s docstring: the point gain is gone post-C6, the
  convergence argument is not.

`lumenairy/propagators/carrier.py`:

* `_sphere_parab_conversion`'s docstring: the validation envelope labelled
  ON-AXIS ONLY, the tilted sweep added, the "~2e-4 of the power" figure
  retracted for tilted congruences, and the not-taken default flip recorded;
* its `w_beam` guard now tests the taper ONSET (`0.75*r_safe < 2*w_beam`)
  rather than `r_safe`, so it can detect the case its own message describes.
  Warning-only -- the returned factor is bit-identical, pinned by
  `test_conversion_guard_is_warning_only`.

`tests/unit/test_niche_s8_sphere_carrier_reference.py` (+2 tests): the onset
semantics (`0.375 r_safe < w <= 0.5 r_safe` warns now and did not before --
design 121 sits at `w = 0.459 r_safe`) and the warning-only contract.  The
existing warning test keeps its `match='band-limit radius'` assertion; the
message was written to retain that phrase.

`ruff check lumenairy/ tests/unit/` clean.  Tests:
`-k "traced or carrier or sphere or c5 or c6 or d1 or s8"` reads **489 passed,
9 skipped** (the skips are CuPy/JAX-x64 environmental), against 330 passed for
`-k "traced or carrier"` before the taper work widened the selection.  No
existing assertion was relaxed anywhere.

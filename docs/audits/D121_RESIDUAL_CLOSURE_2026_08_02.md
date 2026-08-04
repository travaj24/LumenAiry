# The last residual on design 121: the arbiter's verdict, and why the pattern was not smooth

**Date** 2026-08-02 - **Tree** `feat/d121-final-closure` @ `ff7c703` (= niche C9
`3753739` plus a concurrent session's D6 test commit; **no library file is
touched by this study** -- `carrier.py` `5a1b0d1021969df1` and
`_lens_traced.py` `9717ad88dd959889`, the C9 hashes, in every run below) -
**Subject** the 0.049-0.935 EE3 points per order that
`D121_FINAL_CLOSURE_2026_08_02` S6 left as "real, per-order, and not
converging away" - **Questions** (1) is it the element's remap model error, as
S6.3 proposed?  (2) why is it non-monotone across neighbouring orders?

---

## 0. Headline

**The non-monotonicity was the campaign's own EE metric.  The remap candidate
was right about the TERM and wrong about the PLACE, and the fix is one
constant.**

The residual left by `D121_FINAL_CLOSURE` was
`0.049 / 0.935 / 0.775 / 0.278 / 0.588 / 0.125` EE3 points.  It decomposes,
completely, into three things:

| | what | S |
|---|---|---|
| **instrument** | the shipped EE3 mask is a hard binary pixel mask on a 0.4 um lattice, worth **+-0.45 points** of quantisation that does not cancel between the arms.  **This is the whole of the non-monotonicity.** | S3 |
| **library** | `_REMAP_RESID_EIKONAL_DEGREE = 4` was missing the residual eikonal's `r^6` term, and the only thing keeping it at 4 was a ghost that niche C8 now bounds.  **Raised to 6 (niche C10).** | S5.3, S6 |
| **floor** | ~0.05 points, present at ZERO tilt and unchanged by every knob swept | S4.2 |

**The per-order result, on `fc_table_121.py`'s convention with the mask fixed
(EE3 points below the exact-ray oracle's true ceiling):**

| order | as `D121_FINAL_CLOSURE` states it | mask fixed (S3) | **+ C10 (S6)** |
|---|---|---|---|
| (0,0)   | 0.049 | 0.048 | **-0.048** |
| (-1,0)  | 0.935 | 0.934 | **0.029** |
| (-2,0)  | 0.775 | 0.774 | **0.063** |
| (-3,0)  | 0.278 | 0.527 | **0.090** |
| (-4,0)  | 0.588 | 0.305 | **0.141** |
| (-4,-2) | 0.125 | 0.279 | **0.152** |
| **spread** | **0.886** | **0.886** | **0.200** |
| monotone in field angle? | **no** (0.278 between 0.775 and 0.588) | **yes** | **yes** |

The mask fix does not shrink the spread -- it makes the column MONOTONE, which
is what turned an unexplained pattern into a tractable one.  The C10 raise is
what shrinks it.  And the production acceptance is **unchanged** -- `BEST-FOCUS[peak]` `dz = 0`,
3.350 um / EE3 90.3 / EE6 99.7 / EE12 99.8, with the peak 0.24 % higher
(S6.3).

1. **THE REMAP CANDIDATE IS RIGHT ABOUT THE TERM AND WRONG ABOUT THE PLACE**
   (S2).  The `wfe_probe_*` pointwise
   arbiter, re-run on the post-C9 tree against freshly captured element calls,
   puts the last group's exit field at **0.00023-0.00442 waves rms** of the
   exact ray trace -- 8-50x better than the pre-C9 0.011-0.036 that
   `APPROXIMATION_AUDIT_POST_C6` S5 argued against.  Converted to energy the
   most generous way available (ALL the Marechal-scattered light landing
   outside 3 um, PLUS the whole measured amplitude error), the element can
   account for **0.001-0.141 points** against a **0.048-0.934 point**
   residual: **12x short** at the two worst orders and **48x** on axis.  And
   the trend has the wrong sign -- the element's own error **grows** with
   field angle (0.001 -> 0.141 points) while the residual **falls**
   (0.934 -> 0.279).  **But that arbiter captures `calls[-1]`.**  It
   exonerates the LAST group, which is the only place this campaign has ever
   measured; the residual is made in groups 2-4, and the term the brief named
   is exactly what closes it there (item 3).  A true statement about one
   element was about to be published as a statement about six.

2. **THE NON-MONOTONE PATTERN IS THE EE MASK** (S3), and it is instrument.
   `hybrid_localize_121.rs_spot` scores EE3 with a HARD BINARY PIXEL MASK
   (`I[hypot(Xg-cx, Yg-cy) <= 3e-6].sum() / I.sum()`) on a **0.4 um** lattice.
   At r = 3 um that mask's boundary is a **47.1-pixel** ring and each of those
   pixels carries **0.128 EE3 points** (measured).  Swept over one readout
   pixel the HARD EE3 moves by **0.99-1.46 points** where the same circle
   scored area-exact moves by only **0.47-0.56** -- so the shipped metric
   carries about **+-0.45 points** of pure quantisation, keyed to where each
   arm's centroid falls modulo 0.4 um.  The two arms' centroids differ by
   0.9-7.2 % of a pixel, so the quantisation cancels at three of the six
   orders and does not at the other three.  Rescoring the SAME intensity
   arrays with an area-exact circular mask:

   | order | centroid frac. pixel, oracle / chain | shipped hard mask | **area-exact mask** |
   |---|---|---|---|
   | (0,0)   | 0.000 / 0.000 | 0.0488 | **0.0482** |
   | (-1,0)  | 0.061 / 0.032 | 0.9347 | **0.9344** |
   | (-2,0)  | 0.057 / 0.009 | 0.7745 | **0.7740** |
   | (-3,0)  | 0.921 / 0.853 | **0.2783** | **0.5274** |
   | (-4,0)  | 0.577 / 0.505 | **0.5877** | **0.3054** |
   | (-4,-2) | 0.389 / 0.336 | **0.1246** | **0.2786** |

   **The 0.9-point swing between neighbours is gone.**  The residual reads
   `0.048 / 0.934 / 0.774 / 0.527 / 0.305 / 0.279` -- monotone in |field
   angle| from the first order on, with (-4,-2) sitting where (-4,0) does.
   Scoring about the sub-pixel PEAK instead of the centroid, or about a
   COMMON centre, returns the same column, so it is the MASK and not the
   CENTRE (S3.2).

3. **THE SMOOTH REMAINDER IS THE C6 RESIDUAL FIT'S MISSING `r^6` TERM**
   (S4-S6).  With the mask fixed the residual is monotone in field angle and
   sits ~0 at zero tilt.  Two independent instruments locate it: it accumulates
   over the CONVERGING groups 2-5 and is free in the two collimated ones
   (S4.1), and swept CONTINUOUSLY in tilt it is a STEP function whose steps
   land on the per-group crossings of `_lens_traced`'s C1 decentre gate to
   three digits (S4.2).  Forcing the branch confirms it causally (S5.1).  But
   the FIX is not the gate: `_REMAP_RESID_EIKONAL_DEGREE` -- the degree of
   `a_fit`, the model of the input residual the C6 stationary-phase launch
   aims along -- was **4**, and a carrier-referenced relay's residual is
   `r^4`-dominant with an `r^6` next term.  Sweeping it reads **4 ~ 5 << 6** at
   every order, which is what "the missing next RADIAL order" predicts and
   what "more resolution" does not.  **Raised to 6 (niche C10).**  The only
   thing that had kept it at 4 -- a self-caustic ghost at degree 5-6 -- is
   reproduced exactly here (5.2 % of the input power manufactured) and is
   removed entirely by `REMAP_INVERSE_SUPPORT_BOUND`, niche C8, which shipped
   the day AFTER that measurement was taken (S5.3).

---

## 1. Provenance, floors, and what was invalidated first

### 1.1 Which library each number was taken against

| file | sha256 (16) | state |
|---|---|---|
| `lumenairy/propagators/carrier.py` | `5a1b0d1021969df1` | **C9 as shipped, unmodified by this study**, every run |
| `lumenairy/elements/_lens_traced.py` | `9717ad88dd959889` | C9 -- **every measurement in S2-S5** |
| `lumenairy/elements/_lens_traced.py` | `34ef5a9d95279b8f` | **after** the one-constant C10 change -- S6 onward |

Every runner prints the version, path and both hashes, and forces
`LUMEN_PIN=0` -- `approx_common` still defaults that pin to a frozen v5.31
export that still exists on this box (`D121_FINAL_CLOSURE` S10 item 1).

**No measurement in this document depends on which of those two trees it ran
on.**  Every arm of every table pins its own value of the constant explicitly
through a script-side `Patch` or through `rc_with_gate.py`, never through "the
library default" -- which is `D121_FINAL_CLOSURE` S10 item 7's trap ("an
intervention expressed as 'leave the library alone' is not an intervention
once the library moves"), and this time it was designed around rather than
discovered.  The check that this actually holds is S8.1's fail-before: on the
POST-change tree, with the constant patched back to 4, all six orders return
the pre-change intensity array bit for bit.

`HEAD` moved from `3753739` to `ff7c703` MID-STUDY, by a different session
committing the D6 discriminator floor that was in the working tree when this
study started.  It touches one test file; both library hashes above are
unchanged across it, and no measurement here spans the move.

### 1.2 THE FIRST THING THIS STUDY DID WAS THROW AWAY THE ARBITER'S CACHE

`wfe_probe_orders.py` caches each order's captured element call to
`_wfe_probe_orders_<m>_<n>_<rn>_<rs>.npz`.  Five such files were on disk,
**dated 2026-07-30** -- i.e. captured before niches C7, C8 and C9.  The
capture is the chain's OWN last-group input and output, so a C9 tree
re-running with those caches would have measured the **pre-C9 chain** and
called it the post-C9 arbiter verdict.  All six were moved to
`_stale_pre_c9/` before anything was run, and every number in S2 is from a
fresh capture.

**A cache keyed on the CONFIGURATION but not on the LIBRARY is a stale-result
generator**, and this one was pointed straight at the study's headline
question.  (It is the same failure mode as S10 item 1 of the previous
document, one level down: there the pin selected the wrong library, here the
cache selected the wrong library's output.)

### 1.3 Differential floors

| instrument | null | reading |
|---|---|---|
| `rc_readout_121.py`, both arms, all six orders | the shipped-resolution EE3/EE6/FWHM against `_fc_table.txt` | **identical to every printed digit**, and the intensity sha256 reproduces the audit's own (`8db002a1c1bd58ef` on axis, `5e8550468cb6061b` at (-4,-2)) |
| `rc_tilt_121.py` | tilt fraction `f = 0` vs `f = 1e-4` (0.0012 mrad -- the tilted BRANCH at negligible tilt) | **90.5324 / 90.4842 / 0.0482 in both**, to four decimals: the tilted code path costs nothing by itself |
| area-exact mask | supersampling `ss` = 8 / 16 / 32 | 90.5205 / 90.5355 / 90.5324 -- **spread 0.015 points**, and 16 -> 32 moves 0.003; `ss = 32` is used throughout |
| `rc_ng_121.py` | full launch lattice vs `NL=257` (12x fewer rays at the early planes) | **<= 0.005 points at every `ng`** on (-1,0) -- the band-limited-envelope stride null |
| `rc_resdeg_121.py` / `rc_levers_121.py` | the same intervention, two scripts, two processes | `degree 6` at (-1,0) and (-4,-2) returns the SAME intensity sha (`b2a8b150`, `c4b850ea`) in both |
| `rc_gate_121.py` | on axis, three branch settings | **byte-identical** (`8db002a1` x 3): no decentre, no branch to choose |
| `energy_stage_audit_121.py` | two identical chain runs, all six stages, every order | **`array_equal=True`, `max\|dE\| = 0.000e+00`** |

Every delta below is against a floor of zero or a stated 0.003-point estimator
noise, both far below the 0.05-0.94-point subject.

---

## 2. The arbiter: the LAST GROUP is not it

### 2.1 What was measured

`wfe_probe_orders.py` runs design 121's real post-DOE chain for one order,
captures the REAL `apply_real_lens_traced` call the chain makes on the last
group (input, carrier, returned field -- no synthetic stand-in), and compares
the returned field POINTWISE against an exact skew ray trace of that same
input, inverted by Newton onto the element's own exit nodes.  `TOTAL` is the
unwrap-free equivalent rms wavefront error of the SHIPPED field against rays
launched along `grad(W + a)` -- i.e. the element's remap IMPLEMENTATION plus
its MODEL, which is exactly the quantity `D121_FINAL_CLOSURE` S6.3 nominated.

Post-C9, fresh captures, `HALF=96` px, amplitude threshold 0.02:

| order | dec/w | `grad a` rms | RAYMAP | RESID | **TOTAL (waves)** | Strehl | **AMP err** | `Pkeep` |
|---|---|---|---|---|---|---|---|---|
| (0,0)   | 0.000 | 0.000693 | 0.00016 | 0.01448 | **0.00023** | 1.0000 | 0.0029 | 0.9996 |
| (-1,0)  | 0.241 | 0.000812 | 0.00025 | 0.01803 | **0.00442** | 0.9992 | 0.0028 | 0.9996 |
| (-2,0)  | 0.481 | 0.001113 | 0.00023 | 0.02627 | **0.00413** | 0.9993 | 0.0028 | 0.9996 |
| (-3,0)  | 0.723 | 0.001533 | 0.00020 | 0.03785 | **0.00347** | 0.9995 | 0.0137 | 0.9992 |
| (-4,0)  | 0.965 | 0.002054 | 0.00027 | 0.05128 | **0.00323** | 0.9996 | 0.0278 | 0.9978 |
| (-4,-2) | 1.079 | 0.002335 | 0.00039 | 0.05789 | **0.00321** | 0.9996 | 0.0317 | 0.9971 |

For scale: the same probe on the pre-C9 tree read **0.011 waves on axis and
0.036 at (-4,-2)** (`wfe_probe_readout.py`'s header records those figures).
C9 improved the element's own exit field by **48x on axis** and **11x** at
(-4,-2) -- the taper was falsifying the field the element was handed, so
fixing the conversion fixed the element pass too.

### 2.2 What it can and cannot explain, in energy

Convert each row to the EE3 it could possibly explain, taking every choice in
the candidate's favour: Marechal `1 - exp(-(2 pi sigma)^2)` for the phase,
`(rel. rms)^2` for the amplitude, and ALL of the scattered light assumed to
land outside the 3 um circle (it cannot all do so; some of it lands inside).

| order | phase bound | amplitude bound | **element, upper bound** | **residual, area-exact** | shortfall |
|---|---|---|---|---|---|
| (0,0)   | 0.0002 | 0.0008 | **0.001** | **0.048** | **48x** |
| (-1,0)  | 0.0771 | 0.0008 | **0.078** | **0.934** | **12.0x** |
| (-2,0)  | 0.0673 | 0.0008 | **0.068** | **0.774** | **11.4x** |
| (-3,0)  | 0.0475 | 0.0188 | **0.066** | **0.527** | **8.0x** |
| (-4,0)  | 0.0412 | 0.0773 | **0.118** | **0.305** | **2.6x** |
| (-4,-2) | 0.0407 | 0.1004 | **0.141** | **0.279** | **2.0x** |

**The candidate fails twice.**  On MAGNITUDE, at the two orders that carry the
residual it is short by a factor of 12, and on axis by 48.  On SHAPE, the
element's bound RISES monotonically with field angle (0.001 -> 0.141) while
the residual FALLS (0.934 -> 0.279): the two orders where the bound is closest
(2.0-2.6x) are the two where the residual is SMALLEST.  A mechanism cannot
explain a quantity it is anti-correlated with, and no amount of second-order
stationary-phase term changes the sign of a trend.

That much is solid, and it is what `APPROXIMATION_AUDIT_POST_C6` S5 was
reaching for when it excluded "the element" by comparing 0.02-0.8 % of peak
against a 2.0-point gap: post-C9 the element's deficit is 0.00-0.08 % of peak
and the gap is 0.05-0.93 points, so the ratio moved the same way twice and its
arithmetic survived the target moving.

**But read the scope before reading the verdict, because this is where every
previous attempt went wrong.**  `wfe_probe_orders.py`
captures `calls[-1]` -- it validates the LAST group, given its input, and
nothing else.  S4.1 then puts most of the residual in groups **2-4**, which no
pointwise arbiter in this campaign has ever looked at, and S5.2 closes it by
changing a fit that runs on ALL SIX groups.

So the honest verdict on the brief's candidate is **not** "refuted".  It is:

* the term the brief named -- `1/2 grad(a - a_fit)^T H^-1 grad(a - a_fit)`,
  the second-order stationary-phase remainder quadratic in what the residual
  fit MISSES -- **is the mechanism** (S5.2 (c), S6);
* but it is **not made at the last group**, where the arbiter measures it at
  0.0002-0.0044 waves and 12x too small, and where every previous attempt to
  find it looked (`APPROXIMATION_AUDIT_POST_C6` S5, `D121_FINAL_CLOSURE` S6.3);
* and the fix is not a new correction term.  It is giving `a_fit` the degree
  it needs -- the residual's `r^6` order -- which the library had capped at
  `r^4` for a reason that has since been superseded (S5.3).

**Both of the standing readings were half right.**  S5 of
`APPROXIMATION_AUDIT_POST_C6` excluded "the element" and was correct about the
last group's WAVEFRONT; `D121_FINAL_CLOSURE` S6.3 reversed that and nominated
the remap model error, and was correct about the TERM.  What neither had was a
per-group instrument.

---

## 3. The non-monotone pattern: the EE mask

`D121_FINAL_CLOSURE` S9 item 2 recorded the pattern as an unexplained finding:
`0.049 / 0.935 / 0.775 / 0.278 / 0.588 / 0.125` across six neighbouring orders
of one relay, with a 0.9-point swing between adjacent field angles.  It is the
metric.

### 3.1 What the shipped EE3 actually computes

`hybrid_localize_121.rs_spot` -- the readout BOTH arms go through, and the one
every table in this campaign quotes -- ends with

```python
r = np.hypot(Xg - cx, Yg - cy)          # cx, cy = each arm's own centroid
out['ee3'] = float(I[r <= 3e-6].sum()) / tot
```

on a lattice of `dx_out = 0.4 um`.  That is a **hard binary pixel mask**: a
pixel is counted whole or not at all according to where its CENTRE falls.
Measured on the stored intensities (`rc_score_121.py`):

| quantity | value |
|---|---|
| `dEE3/dr` at r = 3 um | **15.06-15.53 points / um** |
| mask boundary ring at 3 um, `dx_out = 0.4 um` | **47.1 pixels** |
| **energy per boundary pixel** | **0.128 EE3 points** |

So flipping four boundary pixels moves EE3 by half a point, and the boundary
moves when the centroid does.

### 3.2 The quantisation, measured two ways

**(a) Hard minus area-exact, on ONE intensity array.**  The area-exact mask is
the same circle with each pixel weighted by the fraction of its area inside it
(32 x 32 supersampled; converged, S1.3).  The difference is pure quantisation:

| order | oracle `hard - area` | chain `hard - area` | **does it cancel?** |
|---|---|---|---|
| (0,0)   | +0.2097 | +0.2092 | yes (0.0005) |
| (-1,0)  | +0.2108 | +0.2105 | yes (0.0003) |
| (-2,0)  | +0.2128 | +0.2123 | yes (0.0005) |
| (-3,0)  | +0.2106 | **+0.4597** | **NO (-0.2491)** |
| (-4,0)  | **+0.4259** | +0.1436 | **NO (+0.2823)** |
| (-4,-2) | **-0.0821** | +0.0720 | **NO (-0.1541)** |

The hard mask over-counts by ~0.21 points on a centred spot -- harmless while
both arms over-count equally.  At (-3,0), (-4,0) and (-4,-2) the two arms'
centroids sit either side of a mask step and it does not cancel.  **Those are
exactly the three orders whose residual moves when the mask is fixed.**

**(b) Sweeping the mask centre over ONE readout pixel.**  Same intensity
array, 9 x 9 sub-pixel offsets:

| order | arm | HARD span | AREA span | **excess = quantisation** |
|---|---|---|---|---|
| (0,0)   | oracle | 1.4352 | 0.5414 | **0.894** |
| (0,0)   | chain  | 1.4367 | 0.5415 | **0.895** |
| (-1,0)  | chain  | 1.2401 | 0.5505 | **0.690** |
| (-2,0)  | chain  | 1.4558 | 0.5616 | **0.894** |
| (-3,0)  | chain  | 1.4636 | 0.5497 | **0.914** |
| (-4,0)  | oracle | 1.2763 | 0.5015 | **0.775** |
| (-4,-2) | chain  | 1.1250 | 0.4849 | **0.640** |

The AREA span (0.47-0.56 points) is the GENUINE cost of putting a 3 um circle
half a pixel off centre -- it is real and it is even in the offset.  The extra
0.64-0.91 points the HARD mask adds is a sawtooth that exists only because the
mask is quantised.  **The shipped EE3 therefore carries roughly +-0.45 points
of instrument, against a "+-0.005-point band" (`D121_FINAL_CLOSURE` S3.5) that
was established by sweeping `CLIP` and `UP` -- knobs that barely move the
centroid, and so could not see this.**

### 3.3 The pattern, after the fix

| order | shipped hard | **area @ own centroid** | area @ sub-pixel PEAK | area @ COMMON centre |
|---|---|---|---|---|
| (0,0)   | 0.0488 | **0.0482** | 0.0480 | 0.0489 |
| (-1,0)  | 0.9347 | **0.9344** | 0.9421 | 0.9382 |
| (-2,0)  | 0.7745 | **0.7740** | 0.7908 | 0.7809 |
| (-3,0)  | 0.2783 | **0.5274** | 0.5287 | 0.5315 |
| (-4,0)  | 0.5877 | **0.3054** | 0.2899 | 0.3145 |
| (-4,-2) | 0.1246 | **0.2786** | 0.2054 | 0.2847 |

All three fixed conventions agree, so it is the MASK and not the CENTRE.
EE6, whose boundary ring carries far less energy, barely moves at all
(0.0177 / 0.5698 / 0.5203 / 0.3937 / 0.2533 / 0.2435 area-exact, against
0.0178 / 0.5703 / 0.5208 / 0.3942 / 0.2587 / 0.2431 hard) -- which is itself
the mechanism showing itself: the defect scales with `dEE/dr` at the mask
radius, and EE6 sits far out on the flat.

**Scope.** This is the campaign's DIAGNOSTIC readout, not the library.  The
library's own production scan (`focus_scan_121.py`) measures EE on a
**0.05 um** lattice -- 8x finer, so ~8x less quantisation -- and reports to
0.1 points, which swamps it.  The shipped acceptance line
(3.350 um / EE3 90.3 / EE6 99.7) is not affected.  `hybrid_localize_121.py`
was NOT edited: every prior table in this campaign remains exactly what it
was, and the area-exact scoring lives in this study's own module.

---

## 4. Where the smooth remainder is made

With the mask fixed the residual reads
`0.048 / 0.934 / 0.774 / 0.527 / 0.305 / 0.279` -- ~0 on axis, maximal at the
FIRST order, falling thereafter.  Two instruments locate it.

### 4.1 It is not one group; it accumulates over the CONVERGING groups

`rc_ng_121.py` is `hybrid_localize_121`'s `n_chain` semantics with the
converged (exact-eikonal) launch split and an area-exact mask: the chain
propagates the first `ng` post-DOE groups and the EXACT ray trace plus
Rayleigh-Sommerfeld integral finishes the rest, so the step from `ng` to
`ng+1` is the cost of putting that group through the chain with everything
downstream identical.  `ng = 0` IS the exact-ray oracle.  Order (-1,0), full
launch lattice:

| `ng` | the group it adds | R_in -> R_out (mm) | gap NA | EE3 area | **d(EE3)** | EE6 area | d(EE6) |
|---|---|---|---|---|---|---|---|
| 0 | -- (the oracle) | -- | -- | 90.5768 | -- | 99.9008 | -- |
| 1 | group 0 | 703650 -> 703665 | 0.0000 | 90.5818 | **+0.0050** | 99.9014 | +0.0007 |
| 2 | group 1 | 703670 -> 703672 | 0.0000 | 90.5870 | **+0.0052** | 99.9015 | +0.0001 |
| 3 | group 2 | 703677 -> **-263.19** | 0.0000 | 90.3517 | **-0.2353** | 99.7721 | -0.1294 |
| 4 | group 3 | -230.72 -> -60.15 | 0.0227 | 90.2755 | **-0.0762** | 99.7112 | -0.0609 |
| 5 | group 4 | -51.47 -> -24.46 | 0.0806 | 90.0241 | **-0.2514** | 99.5920 | -0.1192 |
| 6 | group 5 | -21.14 -> -7.71 | 0.1484 | 89.6424 | **-0.3817** | 99.3310 | -0.2610 |

**The two collimated groups are free** (they IMPROVE on the oracle by
0.005 points each, i.e. they are at the instrument's floor).  The cost appears
the moment the beam becomes convergent and accumulates over four groups.  It
is therefore not a single defect and, in particular, **not the last element** --
which S2 had already exonerated pointwise.

Repeating the sweep with the launch lattice decimated to 257 samples across
(12x fewer rays at the early planes, where the beam is 6.3 mm on a 51 um
pitch) reproduces the (-1,0) column to **<= 0.005 points at every `ng`** --
the band-limited-envelope stride null -- and extends it to three more orders:

| `ng` (group added) | (0,0) | (-1,0) | (-4,0) | (-4,-2) |
|---|---|---|---|---|
| 1 (group 0, collimated) | +0.0014 | +0.0048 | +0.0001 | +0.0047 |
| 2 (group 1, collimated) | +0.0007 | +0.0054 | +0.0097 | +0.0027 |
| 3 (group 2, focusing)   | -0.0841 | -0.2397 | -0.1459 | -0.0692 |
| 4 (group 3)             | +0.0504 | -0.0752 | -0.0579 | -0.0840 |
| 5 (group 4)             | -0.0874 | -0.2484 | *-0.7133* | *-0.9404* |
| 6 (group 5)             | +0.0708 | -0.3812 | *+0.6019* | *+0.8076* |
| **total (ng 0 -> 6)** | **-0.048** | **-0.934** | **-0.305** | **-0.279** |

**Read the last two rows of the (-4,0) / (-4,-2) columns as ONE number, not
two.**  Those two `ng` values read out at an INTERMEDIATE plane whose envelope
max per-pixel step is **2.50 rad** -- close enough to pi that the readout's own
finite-differenced launch directions are unreliable there, and the two entries
are large and opposite, which is the signature.  Their SUM (-0.111 and -0.133)
is meaningful; the split is not.  On axis every step is within +-0.09 and they
alternate in sign: the on-axis chain is free to the intermediate readout's own
noise, all the way through.

What survives from this instrument is the robust part, and it is enough:
**groups 0 and 1 are free on every order (|d| <= 0.010), and the entire
residual is made in groups 2-5.**

### 4.2 It is a STEP FUNCTION of tilt, and the steps are at a code threshold

Nothing in the chain requires the tilt to be a DOE order, so `rc_tilt_121.py`
sweeps it continuously in units of the first order's
`L1 = lambda/period = 11.5158 mrad`, both arms moving together:

| `f` | L (mrad) | oracle EE3 | chain EE3 | **residual** | residual EE6 |
|---|---|---|---|---|---|
| 0        | 0.0000  | 90.5324 | 90.4842 | **0.0482** | 0.0177 |
| 0.0001   | -0.0012 | 90.5324 | 90.4842 | **0.0482** | 0.0177 |
| 0.01     | -0.1152 | 90.5342 | 90.4850 | **0.0491** | 0.0177 |
| 0.05     | -0.5758 | 90.5343 | 90.4860 | **0.0483** | 0.0176 |
| 0.1      | -1.1516 | 90.5349 | 90.4877 | **0.0472** | 0.0177 |
| 0.2      | -2.3032 | 90.5318 | 90.4850 | **0.0468** | 0.0178 |
| **0.35** | -4.0305 | 90.5398 | 89.8999 | **0.6399** | 0.3287 |
| 0.5      | -5.7579 | 90.5457 | 89.9134 | **0.6323** | 0.3279 |
| 0.7      | -8.0611 | 90.5500 | 89.9372 | **0.6129** | 0.3241 |
| **1**    | -11.5158| 90.5768 | 89.6424 | **0.9344** | 0.5698 |
| 1.5      | -17.2737| 90.6200 | 89.7557 | **0.8643** | 0.5495 |
| 2        | -23.0316| 90.6650 | 89.8910 | **0.7740** | 0.5203 |
| 3        | -34.5474| 90.6961 | 90.1686 | **0.5274** | 0.3937 |
| 4        | -46.0633| 90.5035 | 90.1981 | **0.3054** | 0.2533 |

The `f` = 1 / 2 / 3 / 4 rows reproduce the (-1,0) / (-2,0) / (-3,0) / (-4,0)
rows of S0 to every printed digit -- the sweep IS the per-order table, with
the integer orders as four of its points.

Two things fall out at once.

**(a) The tilted BRANCH costs nothing.**  `f = 1e-4` (1.2 urad) puts the chain
on its tilted code path -- `_shift_envelope`, `_tilt_ramp`,
`_tilt_exactness_phase`, `TiltedCarrier` -- and reproduces the untilted row to
four decimals in every column.  There is no discontinuity at `L != 0`.

**(b) The residual is FLAT to 0.2 of an order and then STEPS.**  A step is not
a model error; models are smooth.  It is a threshold in the code, and there is
exactly one threshold of the right kind:
`_lens_traced._DECENTRE_GATE_W_FRAC = 0.05` (niche C1), which selects, per
element call, between the historical CONCENTRIC ray-fit path (hard NaN sample
mask, origin-referenced beam radius) and the D1/D7 OFF-CENTRE path (weighted
restriction `_FIT_DISC_OUTSIDE_WEIGHT_REL = 1e-8`, fit order
`_DECENTRED_FIT_POLY_ORDER = 10`, centre-referenced radius).

Measured from the chain's OWN captured element calls, design 121's per-group
decentre in beam radii at `f = 1` and the `f` at which each group crosses the
gate:

| group | `\|c\|` (um) | `w` (mm) | `\|c\|/w` at f=1 | **crosses the gate at f =** |
|---|---|---|---|---|
| 0 | 80.6  | 6.3181 | 0.01276 | **3.919** |
| 1 | 311.6 | 6.3189 | 0.04931 | **1.014** |
| 2 | 393.7 | 6.3197 | 0.06229 | **0.803** |
| 3 | 805.8 | 5.2661 | 0.15301 | **0.327** |
| 4 | 825.6 | 4.1518 | 0.19886 | **0.251** |
| 5 | 753.9 | 3.1337 | 0.24059 | **0.208** |

(The first six figures reproduce `C6_FIT_GUARD_DECISION_2026_07_31`'s own
per-group decentre table to four decimals, on a completely different
instrument.)

**The steps and the crossings coincide, quantitatively:**

* nothing crosses below `f = 0.208` -> the residual is flat at 0.047-0.049;
* groups **5, 4 and 3** cross at 0.208 / 0.251 / 0.327 -> between `f = 0.2`
  and `f = 0.35` the residual jumps **+0.593**;
* nothing crosses between 0.327 and 0.803 -> `f = 0.35 / 0.5 / 0.7` is a
  plateau (0.640 / 0.632 / 0.613);
* group **2** crosses at 0.803 -> between `f = 0.7` and `f = 1` the residual
  jumps **+0.322**;
* group **1** crosses at 1.014 and costs nothing (0.934 -> 0.864), which is
  the NG sweep's result from the other side: groups 0 and 1 are free.

The two independent localisations agree: the groups that cost EE3 in S4.1
(2, 3, 4, 5) are exactly the groups whose gate crossings produce the steps
here, and the two that are free in S4.1 (0, 1) are the two whose crossings
cost nothing.

**Within a plateau the residual then DECLINES with decentre** (0.640 -> 0.613
across f = 0.35-0.7; 0.934 -> 0.864 -> 0.774 -> 0.527 -> 0.305 across
f = 1-4).  That is the second half of the shape, and it is what makes the
per-order column fall with field angle: the off-centre branch's error is worst
just above its own switch-on and improves as the decentre grows.

**Read the gate as the SYMPTOM'S ADDRESS, not as the cause.**  It says WHERE
the residual is made -- in the element calls that have crossed onto the
weighted branch -- with a sharpness no smooth model error could produce, and
that is what made the rest of this tractable.  It is not what is wrong: S5.2
finds a lever that closes the residual without touching the branch at all, and
S5.1 shows that moving the gate is not available as a fix anyway.

---

## 5. Forcing the branch: the mechanism, causally

The gate correlation of S4.2 is circumstantial until the branch is forced
instead of chosen.  `rc_gate_121.py` runs every order three ways -- the gate as
it ships, `_DECENTRE_GATE_W_FRAC = inf` (**every** element call takes the
historical CONCENTRIC path), and `_DECENTRE_GATE_W_FRAC = _DECENTRE_GATE_PIXELS
= 0` (every call takes the D1/D7 OFF-CENTRE path, i.e. the pre-C1 selector) --
all script-side, all through the campaign's own readout, all area-exact.

### 5.1 The three arms

EE3 (area-exact) and the residual against the CARRY=1 ceiling:

| order | last-group `\|c\|/w` | oracle | shipped | **concentric** | offcentre | res shipped | **res concentric** | res offcentre |
|---|---|---|---|---|---|---|---|---|
| (0,0)   | 0.000 | 90.5324 | 90.4842 | 90.4842 | 90.4842 | 0.0482 | **0.0482** | 0.0482 |
| (-1,0)  | 0.241 | 90.5768 | 89.6424 | **90.5290** | 89.6453 | 0.9344 | **0.0478** | 0.9314 |
| (-2,0)  | 0.481 | 90.6650 | 89.8910 | **90.5769** | 89.8923 | 0.7740 | **0.0881** | 0.7728 |
| (-3,0)  | 0.723 | 90.6961 | 90.1686 | *80.9276* | 90.1697 | 0.5274 | ***9.7685*** | 0.5263 |
| (-4,0)  | 0.965 | 90.5035 | 90.1981 | *70.5743* | 90.1981 | 0.3054 | ***19.9293*** | 0.3054 |
| (-4,-2) | 1.079 | 90.1071 | 89.8285 | *78.8619* | 89.8285 | 0.2786 | ***11.2452*** | 0.2786 |

**Nulls.**  On axis all three arms are BYTE-IDENTICAL (`8db002a1` three times):
zero decentre, no branch to choose.  At (-4,0) and (-4,-2) `offcentre` is
byte-identical to `shipped` (`8a589d3d`, `5e855046`), because no group of
those orders is below the gate -- which is `C6_FIT_GUARD_DECISION`'s own
per-group table, reproduced as a byte-identity.

**The result, and it cuts both ways.**

* At **0.24 w and 0.48 w** the concentric branch takes the residual to
  **0.048 and 0.088** -- i.e. to the zero-tilt floor.  **The whole of
  (-1,0)'s 0.934 and 0.686 of (-2,0)'s 0.774 is the element having taken the
  D1/D7 off-centre branch where the historical one was better.**
* At **0.72 w and above** the concentric branch is a catastrophe: 9.8, 19.9
  and 11.2 points.  That is D1's own failure mode, reproduced end to end, and
  it is why the branch exists.

So the two branches CROSS OVER between **0.48 w and 0.72 w** of decentre on
design 121, and `_DECENTRE_GATE_W_FRAC` puts the switch at **0.05 w** --
**10 to 14 times too early**.  Everything between the gate and the crossover
runs on the wrong side.  Design 121 has three groups in that band at (-1,0)
and two at (-2,0), and that is exactly where the residual is.

**And the halo currency does not see any of it.**  `energy_stage_audit_121.py`
(unedited, via `rc_with_gate.py`), `RN=1024`, `rs=4`, six groups,
`final_leg='paraxial'`, NULL intervention `array_equal=True` /
`max|dE| = 0.000e+00` on all six stages of every order:

| order | `elem(5)` | end to end | `g4` | `amax4` | `r_rms` mm |
|---|---|---|---|---|---|
| (0,0)   | 0.995930 | 0.994314 | 0.000e+00 | 0.000e+00 | 0.8382 |
| (-1,0)  | 0.996005 | 0.994062 | 1.285e-12 | 7.933e-06 | 0.8383 |
| (-2,0)  | 0.996043 | 0.994130 | 7.307e-12 | 1.169e-05 | 0.8381 |
| (-3,0)  | 0.995924 | 0.994071 | 1.060e-09 | 6.270e-05 | 0.8378 |
| (-4,0)  | 0.995931 | 0.994016 | 7.700e-11 | 2.175e-05 | 0.8372 |
| (-4,-2) | 0.996052 | 0.993830 | 2.167e-10 | 3.570e-05 | 0.8372 |

Every row is 6 of 6 on `ENERGY_CONSERVATION_AUDIT`'s bounds -- **including
(-4,0), which is losing 19.9 EE3 points in the image.**  The campaign has said
since the energy audit that EE is blind to the halo; this is the same
statement from the other side, and it is worth recording as a matched pair:
**a field can pass every conservation and halo bound while its spot is
destroyed, and the only instrument that sees it is the one that looks at the
image.**

**What "concentric" is, exactly.**  Forcing that branch is a COMPOUND
intervention, and the document says so rather than pretending otherwise.  Four
things move together:

1. the ray-fit sample restriction: the D1 weighted down-weight
   (`_FIT_DISC_OUTSIDE_WEIGHT_REL = 1e-8`) becomes the historical hard NaN
   mask;
2. the fit disc's CENTRE and RADIUS: `min(2 w, launch_radius)` about the beam
   becomes `min(2 w_origin, launch_radius)` about the GRID ORIGIN, where
   `w_origin = sqrt(2 c^2 + w^2)` -- so at (-4,-2)'s 1.08 w decentre the disc
   grows from 2.0 w to 3.65 w;
3. the ray-fit polynomial order: `_DECENTRED_FIT_POLY_ORDER = 10` reverts to
   the caller's `newton_poly_order`;
4. the C6 residual-eikonal fit's own centre and radius revert with it
   (`centre=(bcx, bcy) if _beam_decentred else None`).

S5.2 separates (1) and (3); (2) and (4) are not separated here and are named
in S7 as unmeasured.

### 5.2 The branch's own levers, and the one that is NOT a branch

`rc_levers_121.py`, all script-side, area-exact, against the true ceiling.
Order (-1,0) (`|c|/w` 0.24 at the last group) and (-4,-2) (1.08):

| lever | (-1,0) EE3 | **residual** | (-4,-2) EE3 | **residual** |
|---|---|---|---|---|
| **shipped** | 89.6424 | **0.9344** | 89.8285 | **0.2786** |
| `_DECENTRED_FIT_POLY_ORDER` 6 (= the D7 fail-before) | 90.6426 | **-0.0658** | 89.9259 | **0.1812** |
| `_DECENTRED_FIT_POLY_ORDER` 8 | 90.6278 | **-0.0510** | 89.7493 | **0.3578** |
| `_DECENTRED_FIT_POLY_ORDER` 12 | 90.5939 | **-0.0171** | 89.9371 | **0.1700** |
| `_FIT_DISC_OUTSIDE_WEIGHT_REL` 1e-4 | 49.3360 | *41.241* | 17.8063 | *72.301* |
| `_FIT_DISC_OUTSIDE_WEIGHT_REL` 1e-2 | 3.6342 | *86.943* | 4.2368 | *85.870* |
| `_REMAP_RESID_EIKONAL_DEGREE` 2 | 88.3859 | *2.1909* | 82.4474 | *7.6597* |
| **`_REMAP_RESID_EIKONAL_DEGREE` 6** | **90.5477** | **0.0290** | **89.9554** | **0.1517** |

Three separate readings.

**(a) The weighted restriction is already at the right end of its axis.**
Raising `_FIT_DISC_OUTSIDE_WEIGHT_REL` from 1e-8 towards a real weight is
catastrophic -- 41 and 87 points at (-1,0), 72 and 86 at (-4,-2), with the
FWHM going to 5.3 um and then to NaN.  The down-weight has to be
indistinguishable from a mask, and 1e-8 is.  **That lever is closed**, and
this is also the measurement that says the D1/D7 restriction is still very
much load-bearing (S8.3).

**(b) The ray-fit order result is NOT monotone and is therefore not, by
itself, a mechanism.**  At (-1,0) orders 6, 8 and 12 all close the residual
(-0.066, -0.051, -0.017) and only the shipped 10 does not (+0.934); at
(-4,-2) order 6 helps by 0.10 and order 8 HURTS by 0.08.  A quantity that is
good at 6, good at 8, bad at 10 and good again at 12 is not an approximation
error converging in a degree -- something discrete is happening at 10 on this
geometry, and this study did not find out what.  It is recorded in S7, not
built on.

**(c) The residual-eikonal degree converges in its EVEN orders, and it is the
brief's own candidate.**  `_REMAP_RESID_EIKONAL_DEGREE` is the total degree of `a_fit`,
the model of the input residual that niche C6 launches along; C6's derivation
says what survives that launch is exactly

```
1/2 grad(a - a_fit)^T H^-1 grad(a - a_fit) ,   H = Hess(W + V)
```

-- **quadratic in what the fit MISSES**, which is the term the brief named.
Sweeping only that degree at (-1,0): **2 -> 2.191, 3 -> 1.815,
4 (shipped) -> 0.934, 5 -> 0.946, 6 -> 0.029.**  The EVEN orders converge and
the odd ones sit on the even below them -- which is the signature of a
near-RADIAL residual, and is what makes this a statement about the residual's
form rather than about resolution (S6.2).  It takes the residual to the
on-axis floor.

### 5.3 The counter-evidence that kept the degree at 4, and why it is spent

`_REMAP_RESID_EIKONAL_DEGREE`'s own docstring says *"Degrees 5-6 buy nothing
and start self-caustiking in the 2-3 w skirt"*, with a `ghost power` column
reading **1.255e-02** at degree 6; `REMAP_STATIONARY_PHASE_FIT_GUARD` records
the same thing independently (*"degree 6 ... still reads 9.78e-03"*).  **Both
measurements predate `REMAP_INVERSE_SUPPORT_BOUND` (niche C8, 2026-08-01),
whose entire job is to stop the library claiming amplitude outside the traced
ray support -- which is what that ghost is.**

Re-measured on the post-C9 tree through `energy_stage_audit_121.py`
(**unedited**, via `rc_with_gate.py`), design 121 order (-4,-2), `RN=1024`,
`rs=4`, six post-DOE groups:

| degree | C8 | `P_out/P_in` | `g4` | `amax4` | `r_rms` mm |
|---|---|---|---|---|---|
| 4 | ON | 0.993839 | 8.653e-09 | 1.147e-04 | 0.8373 |
| 4 | OFF | 0.993839 | 8.653e-09 | 1.147e-04 | 0.8373 |
| 6 | ON | **0.993843** | **9.694e-09** | **1.117e-04** | **0.8376** |
| 6 | **OFF** | ***1.051890*** | ***5.818e-02*** | ***9.448e-01*** | ***2.3601*** |

**The ghost is real, it reproduces, and C8 removes it entirely.**  At degree 4
the support bound is INERT (the two rows are identical to every printed
digit); at degree 6 it is decisive -- without it the chain MANUFACTURES 5.2 %
of the input power and the exit second moment triples.  With it on, degree 6's
conservation and halo sit within noise of degree 4's:

| order | `g4` deg 4 (C9 record) | **`g4` deg 6** | `amax4` deg 4 | **`amax4` deg 6** | e2e deg 4 | **e2e deg 6** |
|---|---|---|---|---|---|---|
| (0,0)   | 0.000e+00 | **0.000e+00** | 0.000e+00 | **0.000e+00** | 0.994314 | **0.994315** |
| (-1,0)  | -- | **2.663e-11** | -- | **1.309e-05** | -- | **0.994065** |
| (-2,0)  | 7.947e-12 | **7.659e-11** | 1.148e-05 | **3.326e-05** | 0.994129 | **0.994133** |
| (-4,-2) | 8.653e-09 | **9.694e-09** | 1.147e-04 | **1.117e-04** | 0.993839 | **0.993843** |

Every `g4` is 1e-3 or less of its C3 bound, `amax4` is 10x under the C4 bound
of 1.0e-03, and `P_out/P_in` moves by <= 4e-06.  **The reason for degree 4 was
correct when it was taken and is superseded by a guard that shipped the day
after.**

---

## 6. What shipped

### 6.1 `_REMAP_RESID_EIKONAL_DEGREE = 4 -> 6`, and nothing else

`lumenairy/elements/_lens_traced.py`, one constant plus its docstring.  No
signature moved, no other default flipped, no public entry point added;
`lumenairy/propagators/carrier.py` is **unmodified** (`5a1b0d1021969df1`, the
C9 hash), `CHANGELOG.md` and `lumenairy/elements/pmm/**` untouched.

**The fail-before is the constant itself.**  It is read once, at
`_fit_residual_eikonal`, and clamped by `_REMAP_RESID_DEGREE_CAP` (= 6, so the
new default sits exactly at the existing cap and adds no new headroom).
Setting it back to `4` restores the v5.32.0 / C9 model exactly -- pinned by
`test_the_default_is_read_from_the_module_constant`, which asserts
`array_equal` on the fitted coefficients against an explicit `degree=4` fit,
and by S6.3 (c), which reproduces all six of design 121's intensity arrays bit
for bit.

**It is NOT byte-identical anywhere the fit engages, and deliberately so** --
the same statement `REMAP_STATIONARY_PHASE_LAUNCH` makes about niche C6
itself.  A different degree is a different polynomial; where the residual has
no `r^6` content the two models agree to well under the fit's own noise (test
6 of S8.2) but not to the bit.  The contract offered here is the fail-before,
not inertness, and it is exact.

### 6.2 The per-order recovery, measured

`rc_resdeg_121.py`, EE3 area-exact against the CARRY=1 exact-ray ceiling,
chain readout split against the exact eikonal -- i.e. the converged instrument
of S3 on the convention of `fc_table_121.py`.  **The residual left, in
points:**

| order | true ceiling | deg 3 | **deg 4 (was shipped)** | deg 5 | **deg 6 (C10)** | **recovered** |
|---|---|---|---|---|---|---|
| (0,0)   | 90.5324 | 1.577 | **0.048** | 0.048 | **-0.048** | **+0.096** |
| (-1,0)  | 90.5768 | 1.815 | **0.934** | 0.946 | **0.029** | **+0.905** |
| (-2,0)  | 90.6650 | 1.461 | **0.774** | 0.796 | **0.063** | **+0.711** |
| (-3,0)  | 90.6961 | 1.087 | **0.527** | 0.554 | **0.090** | **+0.438** |
| (-4,0)  | 90.5035 | 0.999 | **0.305** | 0.338 | **0.141** | **+0.164** |
| (-4,-2) | 90.1071 | 0.967 | **0.279** | 0.386 | **0.152** | **+0.127** |

**The residual is closed.**  It goes from `0.048 / 0.934 / 0.774 / 0.527 /
0.305 / 0.279` to `-0.048 / 0.029 / 0.063 / 0.090 / 0.141 / 0.152` -- every
order inside **+-0.16 points** of the exact-ray oracle, with the FIELD-ANGLE
SPREAD collapsing from **0.886 to 0.200 points**.  Two orders now read
slightly ABOVE the ray oracle, which is the expected sign of agreement rather
than a defect: the "ceiling" is a geometrical-optics construction with its own
error, and the on-axis arms have bracketed it by +-0.05 all along.

**EE6 collapses with it**, which matters because EE6 is where the halo the
residual was made of actually sat.  Residual EE6 points below the ceiling,
deg 4 -> deg 6: `0.018 -> -0.011`, `0.570 -> 0.076`, `0.520 -> 0.082`,
`0.394 -> 0.093`, `0.253 -> 0.100`, `0.244 -> 0.092`.  The chain's FWHM also
comes down on every tilted order (e.g. (-1,0) 3.461 -> 3.446 um, (-4,0)
3.423 -> 3.413), so this is not EE3 being bought from somewhere else.

**The FORM signature is there at every order.**  `deg 5 - deg 4` is
+0.000 / +0.012 / +0.022 / +0.027 / +0.032 / +0.107 -- i.e. nothing, or
slightly worse, everywhere -- while `deg 6 - deg 5` is
-0.096 / -0.917 / -0.732 / -0.465 / -0.197 / -0.234.  **Odd degrees buy
nothing from a near-radial residual and the next EVEN radial order buys all of
it.**  That is the mechanism identifying itself, not a constant tuned to a
table: a fix that were merely "more resolution" would improve monotonically
through 5, and this does not.

### 6.3 Acceptance

**(a) The shipped production acceptance does not regress -- it is identical,
with a higher peak.**  `focus_scan_121.py` (**unedited**, pure library
defaults `CREF`/`AM`/`PIP` unset, N=2048, `rs=4`, NFC=8192, WF=4.0, NOUT=2048),
run twice in this session through `rc_with_gate.py` with the degree PINNED
either way so neither arm depends on what the default happens to be:

| | **deg 4 (fail-before)** | **deg 6 (shipped)** |
|---|---|---|
| `AT-PLANE` | 3.350 um / 90.3 / 99.7 / 99.8 | **3.350 um / 90.3 / 99.7 / 99.8** |
| `BEST-FOCUS[peak]` plane | dz = **+0 um** | dz = **+0 um** |
| FWHM / EE3 / EE6 / EE12 | 3.350 um / 90.3 / 99.7 / 99.8 | **3.350 um / 90.3 / 99.7 / 99.8** |
| peak | 5.516e+03 | **5.529e+03** (+0.24 %) |

The deg-4 arm reproduces `D121_FINAL_CLOSURE` S5.1's recorded line to every
printed digit, including the peak -- so the device is the campaign's own
measurement, not a new one.  **The recorded acceptance line is unchanged.**
(As `_FIT_DISC_OUTSIDE_WEIGHT_REL`'s note already says, production re-traces
the last group on a fine grid where much of this is inert; the diagnostic
paraxial route and every per-order oracle comparison in this campaign are not,
and that is where the 0.9 points lived.)

**(b) Conservation and halo, on the campaign's own instrument.**
`energy_stage_audit_121.py` (**unedited**, via `rc_with_gate.py`), `RN=1024`,
`rs=4`, six post-DOE groups, `final_leg='paraxial'`.  NULL intervention
`array_equal=True`, `max|dE| = 0.000e+00` on all six stages of every order.
See the S5.3 table: `g4` moves from 0.000e+00 / 7.947e-12 / 8.653e-09 to
0.000e+00 / 7.659e-11 / 9.694e-09 at (0,0) / (-2,0) / (-4,-2), all **1e-3 or
less of their C3 bounds**; `amax4` stays 10x under the C4 bound of 1.0e-03;
`P_out/P_in` moves by <= 4e-06 and stays inside C2's [0.9850, 1.00050];
`r_rms` moves by <= 0.0004 mm against a C5 tolerance of 0.030.  **6 of 6 on
every order, in both states.**

**(c) The fail-before is bit-exact on the DESIGN, not just on a fixture.**  In
a fresh process on the POST-change tree (`_lens_traced.py`
`34ef5a9d95279b8f`), with `_REMAP_RESID_EIKONAL_DEGREE` patched back to 4:

| order | pre-C10 intensity sha256 (16) | fail-before, post-change | |
|---|---|---|---|
| (0,0)   | `8db002a1c1bd58ef` | `8db002a1c1bd58ef` | OK |
| (-1,0)  | `eef5a64eb2f808a3` | `eef5a64eb2f808a3` | OK |
| (-2,0)  | `41a950e7767eb956` | `41a950e7767eb956` | OK |
| (-3,0)  | `7db7995c34afbdec` | `7db7995c34afbdec` | OK |
| (-4,0)  | `8a589d3d7013ade5` | `8a589d3d7013ade5` | OK |
| (-4,-2) | `5e8550468cb6061b` | `5e8550468cb6061b` | OK |

**6 of 6 bit-identical**, against shas recorded BEFORE the constant was
touched (they are `D121_FINAL_CLOSURE`'s own, for the two orders it prints).
The fail-before is therefore proven on the DESIGN, in a fresh process, not
merely on the unit fixture -- which is the contract `D121_FINAL_CLOSURE` S7.1
had to build a whole `git archive` device for, and which is trivial here only
because the change is one integer.

**(d) The C7 halo self-check stays silent.**  `grep -c "HALO self-check FAILED"`
reads **0** across every run of this study -- the degree-6 energy audit, the
degree-6 AND degree-4 production focus scans, the six-order degree sweep, and
the niche suites.

**(e) THE ONE THING THAT MOVES THE WRONG WAY, measured and stated.**  The
arbiter of S2, re-run on the C10 tree (caches invalidated AGAIN -- artefact 1,
second time), says the LAST GROUP's own element pass gets slightly WORSE:

| order | TOTAL waves, deg 4 | **TOTAL waves, deg 6** | Strehl | AMP err |
|---|---|---|---|---|
| (0,0)   | 0.00023 | **0.00113** | 0.9999 | 0.0029 |
| (-1,0)  | 0.00442 | **0.00570** | 0.9987 | 0.0028 |
| (-4,-2) | 0.00321 | **0.00344** | 0.9995 | 0.0327 |

That is a real counter-movement and it is small: the element's Marechal bound
goes from 0.078 to 0.128 points at (-1,0) while the CHAIN's residual there
goes from 0.934 to 0.029.  Two readings of it, and this study cannot separate
them: the last group's INPUT is different now (it is the exit of five
better-corrected groups, so the quantity being fitted has changed), and/or
degree 6 is marginally the worse model for that particular group's residual.
Either way the trade is 0.05 points of last-group wavefront bound against
0.9 points of measured chain EE3, in the same direction as every other
currency here -- and it is one more demonstration that the last group was
never where this lived.

**(f) Suites and lint.**

```
python -m pytest tests/unit/test_niche_{c1,c3,c5,c6,c7,c8,c9,c10,d1,d2,d3,d6,d7,s8}_*.py -q
->  365 passed, 72 warnings in 1834.35s          (30m34s)

grep -c "HALO self-check FAILED"   ->  0
grep -c "^FAILED"                  ->  0

python -m pytest tests/unit/test_niche_c10_residual_eikonal_degree.py -q
->  9 passed in 1.09s

python -m pytest tests/unit     -k "traced or carrier or sphere or c5 or c6 or c8 or c9 or c10 or d1 or s8" -q
->  527 passed, 9 skipped, 120 warnings in 861.75s        (14m21s)
    HALO self-check firings: 0

python -m ruff check lumenairy/ tests/unit/    ->  All checks passed!
```

**365 = the 355 of `D121_FINAL_CLOSURE` S8.4, plus C10's 9 and the sibling
added in S8.3.**  The 72 warnings are the pre-existing physics diagnostics the
suite is documented to emit (71 before, plus the deliberate fold-caustic
warning the new S8.3 sibling provokes and asserts).

The WIDER selection is the one that would catch a reader of this constant
outside the niche files -- `D121_FINAL_CLOSURE` S8.4 ran the same `-k`
expression and got 517 passed / 9 skipped; this run is 527 / 9, the difference
being C10's 9 tests and the S8.3 sibling.  **There is no such reader.**  The
9 skips are the documented CuPy / JAX-x64 environmental ones.  `validation/`
is `extend-exclude`d from ruff by `pyproject.toml`, as it is for every
existing runner in that directory.

---

## 7. What remains unmeasured

1. **The ray-fit order anomaly at 10.** `_DECENTRED_FIT_POLY_ORDER` 6, 8 and
   12 all close the (-1,0) residual and the shipped 10 does not (S5.2 (b));
   at (-4,-2), 6 helps and 8 hurts.  That is not an approximation error
   converging in a degree and this study did not identify it.  It is a second,
   independent thing wrong in the same neighbourhood, and it is untouched.
2. **The C1 decentre gate is 10-14x below the crossover** (S5.1): the
   concentric branch is the better one out to ~0.5 w on design 121 and the
   gate switches at 0.05 w.  Raising the gate would recover 0.89 points at
   (-1,0) and 0.69 at (-2,0) -- but it would have to be raised to somewhere in
   (0.48, 0.72) w on the evidence of ONE design, against a branch (D1/D7) that
   exists because the other one fails catastrophically above that, and the
   crossover is certainly design-dependent.  **Not attempted.**  The C10 raise
   recovers the same points without touching the gate, which is why it is the
   one that shipped; whether the gate is ALSO wrong is now a separate,
   well-posed question with a measured instrument
   (`rc_gate_121.py`, `rc_tilt_121.py`) waiting for it.
3. **The arbiter measures the LAST GROUP only.**  `wfe_probe_orders.py`
   captures `calls[-1]`.  S2's exoneration is therefore an exoneration of
   group 5, and the NG sweep (S4.1) puts most of the residual in groups 2-4,
   which no pointwise arbiter in this campaign has ever looked at.  A
   per-group arbiter is the obvious next instrument and was not built.
4. **The through-focus discrimination was started and abandoned.**
   `rc_focus_121.py` asks whether the residual is a WAVEFRONT defect or a
   FOCUS OFFSET (both arms scanned through +-10 um, area-exact).  At 21 planes
   x 2 arms x 4 orders it did not fit the budget and was killed after one arm.
   The question is still open and the script is committed.
5. **Two of the four things the concentric branch changes were not
   separated** (S5's list items 2 and 4: the fit disc's centre/radius, and the
   C6 residual fit's own centre/radius).  S5.2 separates the sample weighting
   and the polynomial order; the other two move with them.
6. **Degree 6 is measured on ONE design at one wavelength.**  What is not
   design-specific is the argument (S5.3, S6.1): the ghost that kept the
   degree at 4 is bounded by a shipped guard, and the term degree 6 adds is
   the next RADIAL order of a residual whose form is set by the carrier
   reference.  No second design was run.
7. **`_REMAP_RESID_DEGREE_CAP` is now equal to the default.**  The raise
   consumes the whole of the existing headroom; nothing here says 8 would be
   worse (it cannot be reached), and the cap's own justification was written
   when the default was 4.
8. **The other `cos^2` taper was not priced.**  `_tilt_exactness_phase`
   (niche C5) has the same entrance/exit structure niche C9 removed from
   `_sphere_parab_conversion`, with an onset that tightens with tilt (computed
   at 2.5 beam radii for the group-6 exit at (-4,0)).  `rc_c5taper_121.py` was
   written to census and ablate it and was not run.
9. **The tilt-obliquity axis was not priced either.**  `rc_zeff_121.py` sweeps
   the Sziklas-Siegman envelope leg's effective distance to bound the
   unimplemented `z/(1-L^2)^{3/2}` correction `_tilt_obliquity` documents; it
   was written and not run.  The argument against it is a trend argument only:
   the correction grows as `L^2` while the residual falls with `L`.
10. **The last group's own element pass got 0.001-0.005 waves WORSE** and
    this study cannot say whether that is the changed input or a marginally
    worse model for that one group (S6.3 (e)).  It is 0.05 points of Marechal
    bound against 0.9 points of measured chain EE3, so it does not change the
    decision, but it is not explained.
11. **The `_REMAP_RESID_EIKONAL_DEGREE` docstring's own synthetic fixtures
    were NOT re-run.**  The "degree 4 spans r^4 exactly" and "2.344e-05 at
    degrees 4, 5, 6" figures are CITED from that docstring, not re-measured
    here; what was re-measured is the ghost (S5.3) and design 121 (S6.2).
12. **The D7 witness's quieting was measured on ONE fixture** (S8.3).  That
    the C6 launch's residual model was part of what made that fold is now
    recorded; how much of D1's original failure mode is the same thing is not.
13. **`propagate_traced_carrier_chain_multi`, chain A, and every design other
    than 121 are untouched**, as in the previous document.

---

## 8. Contracts

### 8.1 The diff

| file | before | after | what |
|---|---|---|---|
| `lumenairy/elements/_lens_traced.py` | `9717ad88dd959889` | `34ef5a9d95279b8f` | **one constant** (`4 -> 6`) + its docstring |
| `lumenairy/propagators/carrier.py` | `5a1b0d1021969df1` | `5a1b0d1021969df1` | **unmodified** |
| `tests/unit/test_niche_d7_decentred_fit.py` | | | one era pin + one added test (S8.3) |
| `tests/unit/test_niche_c10_residual_eikonal_degree.py` | | | new, 9 tests |
| `CHANGELOG.md`, `lumenairy/elements/pmm/**` | | | **untouched** |

`git diff --stat` reads two files, `+121 -2`, of which the LIBRARY change is
one integer.  No signature moved, no other
default flipped, no public entry point added.  Every measurement in S2-S5 was
taken through a script-side `Patch` of a module attribute inside a
`try/finally`, never a library edit, and both the "before" and "after" arms of
every table are pinned EXPLICITLY -- so none of them changed meaning when the
default moved (`D121_FINAL_CLOSURE` S10 item 7's trap, avoided by
construction).

### 8.2 Tests added

`tests/unit/test_niche_c10_residual_eikonal_degree.py`, **9 tests, ~1.1 s**,
no proprietary asset -- the fit is exercised directly on a synthetic Gaussian
carrying a KNOWN radial residual `a(r) = c4 (r/w)^4 + c6 (r/w)^6`:

1. the shipped degree is 6 and the cap permits it;
2. **the fail-before**: `degree=None` resolves to the module constant, so
   setting it back to 4 reproduces an explicit `degree=4` fit
   (`array_equal` on the coefficients) -- and the two really are different
   models on this fixture;
3. the cap clamps a higher request (`array_equal` between `cap` and `cap+4`);
4. **the point of the change**: degree 6's slope error is under 0.2x degree
   4's on an `r^6` residual, with degree 4's own liveness asserted against the
   residual's own slope scale so a dead fixture cannot pass it;
5. **the FORM statement**: degree 5 lands on degree 4 (within 0.25x) while
   degree 6 does not -- odd degrees buy nothing from a radial residual;
6. a PURE `r^4` residual is indifferent to the raise (both arms under 1 % of
   the residual's own slope, and 6 is not more than 3x worse than 4);
7. the model stays curl-free at the raised degree (a necessary condition for
   the launched bundle to be a congruence at all);
8-9. `value` and `grad` are one polynomial at degree 4 AND 6.

Every numeric bar is a RATIO between two arms measured in the same process on
the same fixture, or an exact-arithmetic identity (`array_equal` on
coefficients); there is no absolute bar on a BLAS-dependent magnitude
anywhere.

### 8.3 Existing tests changed -- ONE fail-before witness that stopped witnessing

Nothing in `tests/` pins `_REMAP_RESID_EIKONAL_DEGREE`'s VALUE (the only
reference is `test_niche_c6_stationary_phase_launch.py`, which temporarily
sets it to 40 to exercise the sample-count step-down and restores it).  One
test broke anyway, and what it says is interesting enough to be worth the
words.

**`test_niche_d7::test_the_fold_regularisation_is_still_load_bearing_at_the_d7_order`.**
It degenerates `_FIT_DISC_OUTSIDE_WEIGHT_REL` back to D1's hard mask and
asserts the same call then folds and ghosts -- a guard-is-still-needed
witness.  At degree 6 the fixture's off-beam amplitude falls from ~0.35 to
**1.8e-04 of peak**, so the witness stops witnessing: **a better model of the
input residual removes part of what was making that fold.**

Fixed the way niche C9 fixed the same class of breakage
(`test_niche_s8::test_conversion_factor_band_limited_taper`): **the assertions
are kept word for word** and are now scored ERA-PINNED at
`_REMAP_RESID_EIKONAL_DEGREE = 4`, the library state the case was calibrated
in, with the reason in its docstring.  Nothing was relaxed and no threshold
moved.

A sibling test, `test_c10_shrinks_this_fixtures_hard_mask_ghost`, records the
NEW fact as a measurement rather than losing it: a RATIO between the two eras
on one fixture in one process, with the degree-4 arm's own liveness asserted.
It also sets `RAY_DENSITY_HALO_CHECK = 'silent'` for its own duration --
because it deliberately BUILDS the halo the v5.32 self-check exists to report,
and leaving that on would put the campaign's "zero halo self-check firings
across the niche suites" property at the mercy of a test that asserts the halo
is there.

**And the guard is still load-bearing where it matters.**  This is a witness
going quiet on ONE synthetic fixture, not the restriction becoming
unnecessary: on design 121's real chain, degenerating that same weight from
1e-8 to 1e-4 costs **41 EE3 points** at (-1,0) and **72** at (-4,-2) (S5.2).


---

## 9. Artefacts found and killed in MY OWN instruments

1. **THE ARBITER'S CACHE WAS FROM BEFORE THE FIX IT WAS BEING ASKED TO
   JUDGE.**  `wfe_probe_orders.py` caches the captured element call keyed on
   `(m, n, RN, rs)` -- the CONFIGURATION -- and five such files sat on disk
   dated 2026-07-30, i.e. pre-C7/C8/C9.  Re-running the arbiter on the C9 tree
   would have silently re-scored the PRE-C9 chain and reported it as the
   post-C9 verdict, on the study's headline question.  Caught by looking at
   the file dates before running anything, and all six were moved to
   `_stale_pre_c9/`.  **A cache keyed on the configuration but not on the
   library is a stale-result generator.**
2. **MY OWN SCOPE ERROR, and it nearly became the headline.**  S2's first
   draft read "the remap candidate is REFUTED", full stop.  The arbiter's
   numbers are correct and its reasoning is correct -- but it measures
   `calls[-1]`, and the residual is made in groups 2-4.  A true statement
   about group 5 was about to be published as a statement about the chain.
   Caught only because a lever that acts on ALL SIX groups
   (`_REMAP_RESID_EIKONAL_DEGREE`) closed the residual outright, which an
   exonerated mechanism cannot do.  **An instrument's SCOPE is part of its
   result, and "the element" is not one object when there are six of them.**
3. **`grep -v` without `--line-buffered` swallowed three long runs.**  The
   per-group sweep produced ZERO bytes for 100 minutes and read as hung; two
   others were invisible until they exited.  The filter, not the job, was
   holding the output.  Every subsequent runner writes to a file with
   `python -u` and is filtered on read.
4. **The harness's own `timeout` is capped at 600 s.**  Passing 1800000
   silently clamps and the job is moved to the background mid-run -- which is
   fine, but the first time it happened it read as a failure.
5. **HEAD moved under this study.**  A concurrent session committed the D6
   discriminator floor that was in the working tree at the start
   (`3753739 -> ff7c703`).  Caught by `git status` showing an expected
   modification gone.  Both library hashes are unchanged across it and no
   measurement spans it, but the provenance line had to be rewritten.
6. **THE MATCHED PAIR TO THE CAMPAIGN'S OWN LESSON.**  The forced-concentric
   arm scores **6 of 6** on every conservation and halo bound at (-4,0) --
   `P_out/P_in` 0.994016, `g4` 7.700e-11, `amax4` 2.175e-05, `r_rms` 0.8372 --
   while losing **19.9 EE3 points** in the image.  The campaign has said since
   the energy audit that EE is blind to the halo; this is the same statement
   from the other side, and neither currency alone would have caught it.
7. **`np.ndarray.ptp` is gone in NumPy 2**, and it took out the scoring script
   AFTER it had printed the section that mattered -- which is the worst place
   for a crash, because the useful output was already on screen and the run
   looked complete until the traceback was read.
8. **ARTEFACT 1, A SECOND TIME, WITH THE FIX IN HAND.**  Re-running the
   arbiter on the C10 tree (S6.3 (e)) would have re-read the caches the
   DEGREE-4 run had just written, and reported the pre-C10 element as the
   post-C10 one.  The same trap, in the same session, twenty minutes after
   writing it up.  The caches are now under `_stale_deg4/`.  **Knowing about a
   trap is not the same as being immune to it; only moving the file is.**
9. **A FAIL-BEFORE WITNESS ERODED, FOR THE SECOND TIME IN ONE DAY.**  The D7
   fold witness stopped failing (S8.3) because the fix made its fixture stop
   breaking -- and a CONCURRENT session spent the same afternoon flooring the
   D6 paraxial-FWHM discriminator for exactly the same reason ("every physics
   fix also better-places the deliberately inferior route's spot", `ff7c703`).
   Two independent instances in one tree is a pattern, not a coincidence:
   **every "the guard is still needed" test is a race between the guard's
   value and the rest of the library, and it will lose eventually.**  The two
   sessions reached for the same two remedies -- floor the discriminator, or
   era-pin it and record the new fact separately -- which is at least a
   consistent house style.  What neither did is the third option: find a
   fixture where the guard is still load-bearing on the CURRENT tree and move
   the witness there.

---

## 10. Reproduction

All commands from `validation/repro_traced_carrier_121/`.  Every runner prints
the library version, path and the sha256 of the two files it imported, and
forces `LUMEN_PIN=0`.

```bash
# S2 -- the pointwise arbiter.  DELETE THE CACHES FIRST (S1.2).
mv _wfe_probe_orders_*.npz _wfe_probe_g5_input.npz _stale_pre_c9/
ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python wfe_probe_orders.py

# S3 -- the readout, and the mask
ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python rc_readout_121.py   # saves the npz
python rc_score_121.py                                            # scores it every way

# S4.1 -- which group
ORDERS='-1,0' python rc_ng_121.py                     # full launch lattice
NL=257 ORDERS='-1,0 0,0 -4,0 -4,-2' python rc_ng_121.py   # decimated, 4 orders

# S4.2 -- the residual as a continuous function of tilt
FRACS='0 0.0001 0.01 0.05 0.1 0.2 0.35 0.5 0.7 1 1.5 2 3 4' python rc_tilt_121.py
python rc_stages_121.py                               # the chain's own per-stage table

# S5.1 -- forcing the C1 decentre branch
ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python rc_gate_121.py
# S5.1 -- and the same intervention in the HALO currency, runner unedited
GATE=concentric ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' CONFIGS='ship' NULL=1 \n    python rc_with_gate.py energy_stage_audit_121.py

# S5.2 -- the branch's own levers
ORDERS='-1,0 -4,-2' python rc_levers_121.py

# S5.3 -- the ghost that kept the degree at 4, and C8 removing it
for d in 4 6; do for c8 in 1 0; do \n  RESID_DEG=$d LUMEN_C8=$c8 ORDERS='-4,-2' CONFIGS='ship' NULL=0 \n      python rc_with_gate.py energy_stage_audit_121.py; done; done

# S6.2 -- THE RESULT: the degree sweep, every order
ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' DEGS='3,4,5,6' python rc_resdeg_121.py

# S6.3(a) -- the production acceptance, BOTH arms pinned, runner unedited
RESID_DEG=4 python rc_with_gate.py focus_scan_121.py     # the fail-before
RESID_DEG=6 python rc_with_gate.py focus_scan_121.py     # shipped
# S6.3(b) -- conservation and halo at the shipped degree
RESID_DEG=6 ORDERS='0,0 -1,0 -2,0 -4,-2' CONFIGS='ship' NULL=0 \n    python rc_with_gate.py energy_stage_audit_121.py

# S8.2 / S8.4
python -m pytest tests/unit/test_niche_c10_residual_eikonal_degree.py -q
python -m pytest tests/unit/test_niche_{c1,c3,c5,c6,c7,c8,c9,c10,d1,d2,d3,d6,d7,s8}_*.py -q
python -m ruff check lumenairy/ tests/unit/

# S8.1 -- the fail-before, on the DESIGN, in a fresh post-change process
python rc_failbefore_121.py

# started and NOT completed (S7 item 4): wavefront defect or focus offset?
ORDERS='0,0 -1,0 -4,0 -4,-2' DZ='-10,10,21' python rc_focus_121.py

# WRITTEN AND NOT RUN (S7 items 8, 9) -- committed so the next study starts
# from an instrument rather than from an argument
ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python rc_c5taper_121.py   # the OTHER cos^2 taper
ORDERS='-1,0 -4,0' python rc_zeff_121.py                          # the tilt-obliquity axis
ORD=-1,0 python rc_converge_121.py                                # convergence, area-exact
ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python rc_c6_121.py        # C6 / fit guard / C8 arms
```

### Files added by this study

```
validation/repro_traced_carrier_121/rc_readout_121.py    the two arms + the mask machinery
validation/repro_traced_carrier_121/rc_score_121.py      every scoring convention
validation/repro_traced_carrier_121/rc_ng_121.py         per-group localisation
validation/repro_traced_carrier_121/rc_tilt_121.py       residual vs CONTINUOUS tilt
validation/repro_traced_carrier_121/rc_stages_121.py     the chain's per-stage diagnostics
validation/repro_traced_carrier_121/rc_gate_121.py       forcing the C1 decentre branch
validation/repro_traced_carrier_121/rc_c6_121.py         C6 launch / fit guard / C8 bound
validation/repro_traced_carrier_121/rc_levers_121.py     the weighted branch's own levers
validation/repro_traced_carrier_121/rc_resdeg_121.py     THE RESULT: the degree sweep
validation/repro_traced_carrier_121/rc_with_gate.py      run any runner with the knobs forced
validation/repro_traced_carrier_121/rc_failbefore_121.py the fail-before, on the design
validation/repro_traced_carrier_121/rc_focus_121.py      through-focus, both arms (S7.4)
validation/repro_traced_carrier_121/rc_c5taper_121.py    the C5 exactness taper (S7.8)
validation/repro_traced_carrier_121/rc_zeff_121.py       the envelope-leg distance (S7.9)
validation/repro_traced_carrier_121/rc_converge_121.py   area-exact convergence
tests/unit/test_niche_c10_residual_eikonal_degree.py      9 tests
docs/audits/D121_RESIDUAL_CLOSURE_2026_08_02.md          this document
```

`hybrid_localize_121.py`, `fc_instrument_121.py`, `fc_table_121.py`,
`wfe_probe_*.py`, `energy_stage_audit_121.py` and `focus_scan_121.py` were
**reused unedited**; every arm above reaches the readout through
`fc_instrument_121`, so it is the campaign's own instrument throughout.

Two directories of invalidated caches are left in place rather than deleted,
because what they are is part of the record: `_stale_pre_c9/` (the arbiter
captures from 2026-07-30, S1.2) and `_stale_deg4/` (the captures the degree-4
arbiter run wrote, S9 item 8).

# BUILD -- route 3, the per-pixel tangent-facet screen

**2026-08-16.  Branch `feat/tangent-facet-screens`, cut from `origin/main`
@ `40f28ae` = v5.35.5, in the worktree `C:/tmp/lum_tf`.  Commit on the branch
only -- no merge, no push, no `gh`.**

Route 3 of the analytic real-lens obliquity generalisation.  Routes 1 and 2
CORRECT the vertex-plane thin screen (`BUILD_SCREEN_OBLIQUITY_2026_08_11`
equation (4), `BUILD_R1_WIRING_2026_08_12` equation (7)).  This one REPLACES
it.

---

## 0. VERDICT

> **SHIPPED AS `surface_model='tangent_facet'`, OPT-IN AND DEFAULT-OFF.  THE
> CLOSED FORM IS EXACT TO 1.7e-13 RELATIVE ON THE CONFIGURATION WHERE
> "EXACT" IS DEFINED, IT BEATS THE SHIPPED v5.35.5 CORRECTION ON EVERY
> POWERED GROUP OF DESIGN 121 BY 6x TO 108x, AND IT REPAIRS A DEFECT NO
> CARRIER CAN REACH.  THE PRIZE'S 0.000372 IS *NOT* REACHED, AND THE REASON
> IS A PROOF RATHER THAN AN EXCUSE.**
>
> **1. THE PRIZE WAS NOT A WAVE MODEL, AND THAT IS DEMONSTRABLE.**  The
> tangent-facet ARM of `BUILD_R1_WIRING` S1.2 imprints `dz * sag` and kicks
> by `-dz grad sag`.  Those are inconsistent: a screen's kick is the gradient
> of its own value, and `grad(dz * sag) = dz grad sag + sag grad dz`.  The
> arm is an excellent RAY model and not a Lagrangian one, so no screen can
> reproduce it.  Measured directly (S2.2): with the arm's momentum carried
> along the ray and its OPD kept, replacing only the kick by the gradient of
> that OPD moves design 121 group 5 from **0.000372 to 0.009305 waves rms**.
> That 25x is the arm's own inconsistency, not a shortfall of route 3.
>
> **2. THE IDENTITY IS EXACT, AND NOW MEASURED WITHOUT A RAY-VS-WAVE SCORE.**
> For a tilted PLANE facet under a plane wave both eikonals are closed-form,
> so the screen a wave model must imprint is exactly `S_in - S_out` -- no
> oracle, no common-mode control, no tolerance.  `_tangent_facet_screen`
> reproduces it to **1.739e-13 relative worst case over 27 cells** (slopes
> 0.05 / 0.12 / 0.24, `(n1,n2)` including `1.8047 -> 1.0000`, tilts 0 / 55 /
> 150 mrad).  Machine precision, with the residual growing exactly where the
> `b - a` cancellation says it should.
>
> **3. THE CLAIM THAT NEEDS NO CARRIER.**  A steep facet is angle-wrong at
> NORMAL incidence, because the FACET is tilted even when the light is not.
> Routes 1 and 2 are differences against the model's own zero-angle value, so
> they are identically zero there -- structurally, not approximately.
> Against `apply_real_lens_traced` at 0 mrad on an R = 12.6 mm N-SSK2
> biconvex: blind **0.00141** waves rms, `carrier=` **0.00141** (the same
> bits), `'tangent_facet'` **0.00008** -- **17.6x**, on the library's own
> end-to-end output.
>
> **4. AND IT IS BETTER WITH A CARRIER.**  100 mrad, R = 19.6 mm N-SSK2
> singlet, same arm: blind 0.00423 -> `carrier=` 0.00050 -> **0.00017**.  On
> design 121's own prescriptions against exact rays, 3 mm pupil (waves rms):
>
> ```
>   group          blind      v5.35.5      ROUTE 3      facet arm
>   g2 doublet   0.016799     0.000498    0.0000046     0.0000196
>   g3 singlet   0.009908     0.000020    0.0000033     0.0000159
>   g4 singlet   0.000489     0.000027    0.0000005     0.0000004
>   g5 doublet   0.258480     0.012398    0.0032381     0.0003724
> ```
>
> **Route 3 beats the ARM ITSELF on g2 (4.3x) and g3 (4.8x)** -- which is
> what item 1 predicts, since the arm carries an inconsistency route 3 does
> not.
>
> **5. THE ACCEPTANCE BAR IS MET AT 1 mm AND 2 mm AND MISSED AT 3 mm.**  The
> brief's bar is 0.001 waves rms on group 5.  The pupil ladder reads
> **0.0000564 / 0.0005963 / 0.0032381** at 1 / 2 / 3 mm.  The 3 mm rung
> misses by 3.2x and S4 names why, bounds it, and shows the term that would
> close it is the one a vertex-plane screen structurally cannot carry.
>
> **6. THE PLATE IS EXACT AND THE DEFAULT IS UNTOUCHED.**  `sag == 0` makes
> all three terms identically zero and one reduction skips the block, so a
> plane plate is **byte-identical** to the thin screen at 0 / 10 / 20 / 55 /
> 100 / 200 mrad (`np.array_equal`).  `surface_model` already defaulted to
> `'thin'`, so no pre-5.36 call site can reach the model at all.
>
> **7. THE COST IS REAL AND IS PRICED, NOT GLOSSED.**  Whole-grid only:
> **+17.8 grids without a carrier, +21.8 with one** at N >= 4096, against the
> banded `carrier=` path's +8.8.  Refused from the band loop rather than
> approximated into it, and the runner preflight is updated version-gated.

---

## 1. THE DERIVATION

### 1.1 What the thin screen actually approximates

`apply_real_lens` imprints `(n2 - n1) sag(x)` on each surface's VERTEX plane
and propagates homogeneously between them.  Two approximations hide in that
one line: the coefficient `(n2 - n1)` is the normal-incidence, zero-slope
limit of the true facet coefficient, and the facet is evaluated at the vertex
plane rather than where the ray meets the surface.

`BUILD_SCREEN_OBLIQUITY` S2.2 derived the exact replacement for the first:
for a plane facet of unit normal `nu` at height `s`, both sides referenced to
the vertex plane,

```
    OPD = dz * s ,   dz = pz2 - pz1 ,
    pz1 = sqrt(n1^2 - |p|^2),
    pz2 = pz1 + (B - A) nu_z ,
    nu  = (-grad sag, 1)/sqrt(1 + |grad sag|^2),
    A   = (-gx px - gy py + pz1) nu_z ,   B = sqrt(n2^2 - n1^2 + A^2).    (T1)
```

Routes 1 and 2 apply (T1) MINUS its carrier-free value, so that the shipped
normal-incidence calibration is preserved and only the angular part moves.
Route 3 applies (T1) itself.  That is the whole difference in one sentence,
and it is what buys S0.3: the carrier-free value of (T1) is not
`(n2 - n1) sag` on a steep facet, and the difference is exactly the error the
correction cannot see.

### 1.2 (T1) is not an expansion -- the independent check

The identity was previously checked with `mpmath` against a ray trace, to the
sag-proportional order.  A stronger check is available and is what this build
uses, because it removes the oracle entirely.

For a plane facet `z = s + g.x` between `n1` and `n2` under a plane wave
`p_in`, BOTH fields are plane waves, so both eikonals are known in closed
form at the vertex plane and the screen a wave model must imprint is exactly

```
    OPD(x) = S_in(x) - S_out(x) .
```

`validation/.../tangent_facet_derive.py facet` builds `S_out` from exact ray
algebra -- intersect, refract at the true normal, return to `z = 0`, and read
off the exit plane wave's constant -- and compares.  The worst relative error
over 27 cells is **1.739e-13**, at slope 0.24, `1.5917 -> 1.8047`, 150 mrad;
the median cell is ~1e-15.  The growth with slope is the `b - a` cancellation
and nothing else.

This also proves the two second-order terms below VANISH for a plane facet,
which is why the plate is byte-exact rather than approximately zero.

### 1.3 (T2) the facet belongs where the ray meets the surface

The pixel's ray leaves the vertex plane at `x` and rises to the surface along
`p/pz1`, meeting it at `x_h = x + w`, `w = s p / pz1`.  The tangent plane
belongs THERE.  Taking it there and extrapolating that PLANE back to the
pixel's own coordinate:

```
    OPD = dz(x_h) * [ sag(x_h) + grad sag(x_h) . (x - x_h) ] .
```

Expanding, the first-order pieces cancel exactly -- which is precisely why
(T1) is exact for a plane -- and what survives is the surface's CURVATURE
across the traverse:

```
    + s (w . grad dz)  -  (dz/2) w^T (grad grad sag) w .                  (T2)
```

**The cancellation is load-bearing and was measured by breaking it.**  An arm
that moves the sample point to `x_h` WITHOUT the back-extrapolation reads
**0.1109** waves rms on group 5 against (T1)'s 0.0073 -- 15x WORSE.  The two
moves are a pair; either alone is a defect.

### 1.4 (T3) the walk has to be referenced back to the pixel

The ray re-crosses the vertex plane at `x + W`,
`W = s (p/pz1 - p_out/pz2)`, so the exit eikonal is known at the wrong place.
The FIRST order of referencing it back is already inside (T1) (that is what
the `- p_in . (s tan a1)` step of the S2.2 derivation is).  The second is

```
    - (1/2) W . (W . grad) p_out .                                        (T3)
```

Carried to THIRD order the term moves group 5 by 0.25 % (0.0028620 ->
0.0028547 with a per-pixel Newton intersection), so the series is truncated
on a measurement, not on taste.

### 1.5 The accumulator, and why it must be transported

`(T1)`-`(T3)` all read `p`, the FIELD's own transverse optical momentum at
the pixel: the carrier's, minus the gradient of every screen imprinted so
far.  A grid-LOCAL accumulation -- the shape routes 1 and 2 use for `p0`,
where it is a sub-1 % effect on a small correction -- is **not** adequate
when it feeds the whole screen: measured on group 5 it reads **0.0335 waves
rms** against 0.0032 for the transported one, i.e. it would be WORSE than the
shipped correction.

The field arriving at pixel `x` came from pixel `x - w`, `w = t p / pz`, so
the accumulator is resampled across each gap.  One Taylor term suffices,
because `p` is very nearly linear in `x` across a lens pupil and the
remainder is second order in the gap walk against a THIRD derivative of the
sag: emulated on the group-5 fixture, the one-term transport reads 0.0028958
against an exactly-carried 0.0028621.  It is also EXACT for a linear momentum
field, which is what `test_the_transport_is_exact_for_a_linear_momentum_field`
pins as an equality rather than a tolerance.

### 1.6 What ships: the grid form, and why

Two implementations were built and measured.

| form | group 5 | works for |
|---|---|---|
| per-pixel fixed-point intersection + tangent extrapolation | **0.0028621** | conic / asphere (needs an analytic sag derivative) |
| **grid-gradient Taylor (T1)+(T2)+(T3)** | **0.0032381** | conic, asphere, biconic, Q-freeform, form-error map, `sag_callable` |

The grid form is 13 % worse on the binding case and needs no per-surface
Newton, no analytic sag derivative and no special-casing: every quantity it
reads is a gradient of a grid field, taken after the decenter shift, the tilt
ramp, the form-error map and any freeform departure are already folded into
`sag`.  **The grid form ships.**

---

## 2. THE MEASUREMENTS

The model arm is traced on a REGULAR BUNDLE rather than a scattered disc, so
that every gradient the library takes with `xp.gradient` on its grid is taken
in the arm as the same physical gradient of the same ray-carried field, via
the bundle's own Jacobian.  That is the only faithful way to score a model
whose screens are gradients of accumulator FIELDS; a scattered-ray arm has to
nest finite differences and its noise floor swamps the terms being measured.

The instrument is calibrated before it is trusted: on design 121 group 5 it
reproduces the published blind screen (**0.258480**), the shipped
eq-(4)+R1 arm (**0.012398**) and the tangent-facet arm (**0.000372**) to every
printed digit, and it is converged (n = 33 / 65 / 129 / 257 read 0.0072400 /
0.0073091 / 0.0073098 / 0.0073182 for (T1)).

### 2.1 Design 121, the four powered groups

`tangent_facet_derive.py d121`, waves rms, exit-plane common-mode control
`D(theta) - D(0)` with piston and tilt removed:

| group | tilt | pupil | blind | ROUTE 3 | gain | facet arm |
|---|---|---|---|---|---|---|
| 2 doublet PK52A/SF57 | 51.50 mrad | 1 mm | 0.001135 | **0.0000000** | 25806x | 0.0000002 |
| 2 | | 2 mm | 0.005807 | **0.0000008** | 7726x | 0.0000037 |
| 2 | | 3 mm | 0.016799 | **0.0000046** | 3616x | 0.0000196 |
| 3 singlet LAK8 | 46.69 mrad | 3 mm | 0.009908 | **0.0000033** | 2961x | 0.0000159 |
| 4 singlet LAK9 | 7.38 mrad | 3 mm | 0.000489 | **0.0000005** | 930x | 0.0000004 |
| **5 doublet SK2/SF57** | **54.87 mrad** | **1 mm** | 0.024564 | **0.0000564** | 436x | 0.0000068 |
| 5 | | 2 mm | 0.104356 | **0.0005963** | 175x | 0.0000651 |
| 5 | | 3 mm | 0.258480 | **0.0032381** | 80x | 0.0003724 |

Against the SHIPPED v5.35.5 correction (0.000498 / 0.000020 / 0.000027 /
0.012398 at 3 mm) route 3 is **108x / 6.1x / 54x / 3.8x** better.  On g2 and
g3 it is also **4.3x and 4.8x better than the tangent-facet ARM**, for the
reason S0.1 gives.

### 2.2 The channel decomposition, redone for a screen

`BUILD_R1_WIRING` S1.2 decomposed the residual into an OPD channel and a
deflection channel.  The same decomposition, run on the arm that produced the
prize, isolates what a screen can and cannot have:

| arm (momentum source / kick) | group 5 |
|---|---|
| ray-carried `p` / EXACT facet kick (**the prize arm**) | **0.000372** |
| ray-carried `p` / the gradient of the arm's own OPD | **0.009305** |
| grid-local `p` / exact kick | 0.033498 |
| grid-local `p` / gradient kick | 0.037459 |

**Row 2 is the ceiling for any screen** that imprints this OPD, and it is
25x above the prize.  Route 3's 0.0032 is BELOW that row, because (T2) and
(T3) change the OPD as well as the kick.

### 2.3 The term ladder

`tangent_facet_derive.py ladder`, 3 mm pupil, waves rms:

| group | (T1) alone | + (T2) | + (T3) | facet arm |
|---|---|---|---|---|
| 5 | 0.0073091 | 0.0067711 | **0.0032381** | 0.0003724 |
| 2 | 0.0000056 | 0.0000053 | **0.0000046** | 0.0000196 |
| 3 | 0.0000083 | 0.0000076 | **0.0000033** | 0.0000159 |
| 4 | 0.0000008 | 0.0000008 | **0.0000005** | 0.0000004 |

(T2) is a 7 % refinement on the binding case; (T3) is a 2.1x one.  Every
row improves monotonically, which is the arithmetic check that the terms are
what the derivation says rather than three things that happen to help once.

### 2.4 End to end, on the library's own output

Against `apply_real_lens_traced` (the shipped exact ray tracer) on the
Nyquist-sampled arm `BUILD_SCREEN_OBLIQUITY` S3.6 established as resolvable,
N = 1536, `dx = 4 um`, 0.7 mm probe disc.  Waves rms, common-mode controlled:

```
  R = 19.6 mm N-SSK2, 4 mm
    floor @ 0 mrad   blind 0.00058   carrier 0.00058   ROUTE 3 0.00008
     20 mrad         blind 0.00048   carrier 0.00012   ROUTE 3 0.00010
     50 mrad         blind 0.00144   carrier 0.00021   ROUTE 3 0.00011
    100 mrad         blind 0.00423   carrier 0.00050   ROUTE 3 0.00017

  R = 12.6 mm N-SSK2, 4 mm  (slope 0.24, the steep case)
    floor @ 0 mrad   blind 0.00141   carrier 0.00141   ROUTE 3 0.00008
     20 mrad         blind 0.00090   carrier 0.00022   ROUTE 3 0.00010
     50 mrad         blind 0.00256   carrier 0.00055   ROUTE 3 0.00011
    100 mrad         blind 0.00747   carrier 0.00280   ROUTE 3 0.00228
```

Two readings.  **The 0 mrad column is the headline**: the carriered arm is
byte-for-byte the blind one there (the correction is structurally zero), and
route 3 drops the floor 7.3x and 17.6x.  **And route 3 without a carrier**
does exactly what the design says it should: 0.00008 at 0 mrad (it fixes the
facet) and 0.00413 at 100 mrad against blind's 0.00423 (it does not know the
arrival angle).  The two effects are separable and both were measured.

---

## 3. WHAT WAS REFUTED ALONG THE WAY

Each of these was a candidate term that was built, measured, and KILLED.

| # | candidate | outcome |
|---|---|---|
| 1 | resample subsequent surfaces by the drift the screen's spurious kick accumulates (`+V . grad OPD`) | **REFUTED.** Made group 5 worse (0.0073 -> 0.0178); a coefficient scan peaked at a FITTED 0.25, which is disqualifying on its own. |
| 2 | resample by the accumulated transverse WALK | **REFUTED.** 0.0073 -> 0.053 at `+0.5`, 0.064 at `-0.5`; the identity already carries the walk's first order, so the term double-counts. |
| 3 | reference the exit eikonal by `+ W . p_out` on top of the identity | **REFUTED**, catastrophically (0.0073 -> 0.483).  Same double-count, and the algebra confirms it: the plane-facet check of S1.2 leaves no room for it. |
| 4 | move the sample point to the hit point WITHOUT back-extrapolating the tangent plane | **REFUTED**, 15x worse (S1.3).  The pair cancels at first order; half of it is a defect. |
| 5 | third order of the walk referencing | **REFUTED as not worth it**: 0.25 % on the binding case, for a third derivative of the momentum field. |
| 6 | per-pixel Newton intersection instead of the grid Taylor | **MEASURED AND NOT TAKEN**: 13 % better, at the price of an analytic sag derivative and a sag-source restriction (S1.6). |
| 7 | is the 0.0073 a discretisation artefact of the bundle? | **NO.** Converged across n = 33..257 to 4 significant figures. |
| 8 | is the residual an edge/high-slope effect? | **NO.** The residual is flat in radius (0.0062 rms inside r <= 0.94 mm against 0.0073 over the full 3 mm), i.e. a low-order term across the whole pupil. |

---

## 4. WHAT IS NOT CLAIMED

* **The prize's 0.000372 is not reached (0.0032381 on g5 at 3 mm), and no
  screen can reach it.**  S0.1 measures the arm's own inconsistency at
  0.009305 -- 25x above the prize -- and S1.4 of `BUILD_R1_WIRING` gives the
  curl obstruction independently.  The residual is the transverse walk, which
  on group 5's exit face (slope 0.244, `n 1.80 -> 1.00`) reaches **140 um
  across a 3 mm pupil**; a vertex-plane screen can reference that away to
  second order (which (T3) does) but cannot REPRESENT it.  Closing it is the
  remap axis (`surface_model='displaced'`), not this one.
* **The brief's 0.001-waves bar is met at 1 mm and 2 mm and missed at 3 mm**
  (0.0000564 / 0.0005963 / 0.0032381).  Recorded rather than rounded.
* **Whole-grid only.**  The model differentiates a gradient twice, so an
  exact band needs a 3-row sag halo AND a halo on the persistent accumulator.
  That was priced and refused rather than approximated -- a silently-wrong
  band is worse than an expensive right one.  The banded `carrier=` path is
  untouched and still byte-identical (198 tests green), and
  `sag_chunk_rows` is pinned INERT for this model at `cr` in
  {0, 1, 7, 64, 4096} with `np.array_equal` -- so a future change that lets
  the model into the band loop has to arrive with its own byte-identity
  argument instead of taking effect silently.
* **No GPU run, no `surface_frame` run, no non-ASM propagator run.**  All
  three RAISE rather than silently running unmeasured.
* **The guard is silent under this model.**  Its estimator measures the size
  of the correction the thin screen needs; route 3 does not make that
  correction, so there is nothing to accumulate.  Stated, not faked.
* **`prepare_real_lens` does not support the model** (a prepared screen is
  input-independent; this one reads the field's own momentum), and
  `apply_real_lens_traced`'s delegate branch does not forward it.
* **Measured only on rotationally symmetric surfaces.**  The model is
  structurally correct for decentred / tilted / freeform faces -- it reads
  `sag` after all of those are folded in, unlike `'displaced'` -- but no
  oracle run was made on one.
* **The runner preflight's tangent-facet term OVERLAPS the `screen_obliquity`
  term when both are set**, because the +carrier coefficients were
  differenced against the no-carrier baseline.  The sum over-estimates, which
  is the safe direction for a preflight; noted there and here.

---

## 5. COST

Warmed `tracemalloc` peak in float64 grids (`8*N*N` bytes) and wall clock,
biconvex singlet R = +19.6 / -27.4 mm N-SSK2, best of the second pass.  The
warm-up matters: the first `apply_real_lens` of a process also pays FFT-plan
and lazy-import allocations (~10 grids at N >= 4096), the mistake
`BUILD_OBL_BANDED_HALO` S5.5 records.

```
  N      thin   thin+carrier  tangent_facet  tf+carrier      (extras over thin)
  2048  12.13      26.25          22.13         26.13     +14.13 / +10.0 / +14.0
  4096   4.38      13.19          22.19         26.19     + 8.80 / +17.8 / +21.8
  8192   4.38      13.19          22.19         26.19     + 8.80 / +17.8 / +21.8

  wall clock (s)
  2048   0.645      3.566          3.258         4.970
  4096   2.037     13.965         11.081        19.067
  8192   7.861     50.199         34.528        61.867
```

At N >= 4096 the thin baseline row-bands and route 3 does not, which is where
the +17.8 / +21.8 comes from; at N = 2048 both are whole-grid and the extras
are +10.0 / +14.0.  At N = 32768 one grid is 8.59 GB, so the carriered model
costs **+187 GB** -- affordable only with the grid budget the preflight now
enforces, and the honest reason banding is the first follow-on.

`Reverse_Symmetric_ASM/tx_design_study_sim.py` (outside the repo, not git)
gains a `LENS_SURFACE_MODEL` selector and a version-gated
`_preflight_memory_check` term, ANCHOR 2026-08-16.  The gate is inverted
relative to the `screen_obliquity` term and the comment says so: an
unparseable version is treated as OLD, and for THIS term "old" means the
feature is UNREACHABLE, so the term is zero -- pricing an unreachable term
would refuse runs that will fit.

---

## 6. FILES

| file | change |
|---|---|
| `lumenairy/elements/_lens_real.py` | the route-3 derivation block, `_tangent_facet_screen` / `_tangent_facet_transport` / `_TANGENT_FACET_MIN_PZ_SQ`; `_VALID_SURFACE_MODELS` gains `'tangent_facet'`; the validators in `_check_screen_obliquity_support` and `_check_displaced_support`; `_tf_active` / `_tf_px` / `_tf_py`; the whole-grid screen block, the gap transport, the two band-gate exclusions, the guard gate, the accumulator teardown, and the `surface_model` docstring |
| `tests/unit/test_tangent_facet.py` | NEW.  68 tests |
| `validation/repro_traced_carrier_121/tangent_facet_derive.py` | NEW.  `facet` / `d121` / `ladder` |
| `validation/repro_traced_carrier_121/_tangent_facet_all.json` | results of record |
| `CHANGELOG.md` | `[Unreleased]` |
| `docs/audits/BUILD_TANGENT_FACET_2026_08_16.md` | this note |

---

## 7. SUITES

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X
python 3.14.6   numpy 2.4.4      lumenairy 5.35.5 (worktree C:/tmp/lum_tf,
numexpr 2.14.1                   branch feat/tangent-facet-screens off
                                 origin/main 40f28ae)
```

| gate | result |
|---|---|
| `test_tangent_facet.py` (NEW) | **68 passed** |
| FINAL-TREE re-verify -- the six byte-identity-critical files (`test_screen_obliquity` + `test_obl_banded_halo` + `test_slant_chunk_byte_identical` + `test_lens_chunked_sag` + `test_tangent_facet` + `test_niche_audit_e_prepared_and_enums`) | **307 passed** in 67.5 s |
| `tests/unit -k "lens or obliquity or facet or slant or chunk or displaced"`, on the COMMITTED tree | **939 passed, 2 skipped** in 1607.8 s |
| `ruff check lumenairy/ tests/ validation/.../tangent_facet_derive.py` | **All checks passed** |
| `xfail` / `skip` added | **ZERO** |
| pre-existing assertions relaxed or retargeted | **ZERO** |

The two skips in the wide run are pre-existing and named as such by
`BUILD_OBL_BANDED_HALO` S5.4: `test_niche_audit_w5_shim_removals`'s
host-specific SHA-256 digests, and the `_PERSISTENT_POOL_LOCK` exemption pin.
Neither is on this build's path.

**Ordering note, so the gate is not overclaimed.**  The 927-test run was
started before the last library edit landed (hoisting the NaN-sentinel zeroing
above the accumulator gradient, S6).  That edit sits entirely inside
`if _tf_active and ...`, which no test in that selection can enter --
`'tangent_facet'` did not exist before this branch -- and the 307-test
re-verify above WAS run on the final committed tree.  The wide run was
repeated on the committed tree for completeness.

Reproducing the study:

```
cd validation/repro_traced_carrier_121
python tangent_facet_derive.py facet    # S1.2, the closed form vs exact algebra
python tangent_facet_derive.py d121     # S2.1, the four powered groups
python tangent_facet_derive.py ladder   # S2.3, the term ladder
```

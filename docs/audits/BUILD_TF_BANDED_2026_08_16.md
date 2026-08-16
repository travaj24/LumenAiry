# BUILD -- the tangent-facet family, row-banded with a halo

**2026-08-16.  Branch `feat/tangent-facet-banded`, cut from `origin/main`
@ `dea5e47` (the remap merge), in the worktree `C:/tmp/lum_tb`.  Commit on the
branch only -- no merge, no push, no `gh`.**

`BUILD_TANGENT_FACET_2026_08_16` S4 and `BUILD_TF_REMAP_2026_08_16` S7 both
refused the band loop, correctly and explicitly, and both pinned
`sag_chunk_rows` INERT with `np.array_equal` so that a later change could not
let the models in silently.  This is the byte-identity argument that change had
to arrive with.

---

## 0. VERDICT

> **BOTH MODELS BAND.  THE BANDED FIELD IS BYTE-IDENTICAL TO THE SHIPPED
> WHOLE-GRID FIELD -- NOT TO THIS BUILD'S OWN, TO `origin/main`'S -- ACROSS
> 10778 ADVERSARIAL CONFIGURATIONS AND A 960-ARM TWO-TREE COMPARISON.  ROUTE 3
> DROPS FROM +21.7 GRIDS TO +7.6, AND ONE OF THOSE 14 GRIDS WAS A BLOCK THAT
> WAS ALREADY DEAD.**
>
> **1. THE HALOS ARE DERIVED, AND THE SURPRISE IS WHICH RUNG NEEDS THE WIDER
> ONE.**  Route 3's accumulator is minus the gradient of its own screen, and
> the screen is five gradients of quantities built from `grad sag`, so the band
> needs **3 rows of sag and 2 of the accumulator**.  The REMAP rung's
> accumulator is `p_out` in CLOSED FORM -- that is (R3), the Lagrangian
> identity `BUILD_TF_REMAP` S0.3 measured -- so its deepest chain is only the
> (R4)/(R5) Hessian: **2 rows of sag and NO accumulator halo at all**.  The
> rung that costs more memory needs the narrower halo, and the reason is the
> one term that separates the two models.
>
> **2. NOTHING MOVED, AGAINST THE SHIPPED TREE.**  10778 adversarial
> configurations (8 prescriptions x 4 grids including odd N and `dx != dy` x
> 2 models x 3 carriers x 7 option combinations x 2 complex dtypes x band sizes
> {1, 3, N, N+7}), compared on the returned field's BYTES, its dtype and the
> warning set: **0 differences**.  And a TWO-TREE run against `origin/main`
> @ `dea5e47`, where the models are whole-grid-only by construction: **960
> arms, 0 differences** for the tangent-facet family and **640 arms, 0
> differences** for the untouched `'thin'` / `'displaced'` paths.  So the
> banded output equals the *shipped* output, which is the claim that matters.
>
> **3. THE FOLD GUARD FIRES IDENTICALLY.**  On the engineered folding
> prescription the banded call raises the SAME `ValueError` at the SAME surface
> with the SAME message as the whole-grid call, at every band size -- which it
> must, because `min(det)` is a whole-grid reduction by construction and a band
> that had not yet seen the folding row would have run and returned a field.
> The control at an amplitude the guard passes is byte-identical.
>
> **4. THE COST IS PAID.**  Warmed `tracemalloc` peak in float64 grids
> (`8*N*N`), extras over the paraxial no-carrier call at the SAME banding,
> N = 4096:
>
> ```
>   arm                              SHIPPED 5.36   this build   this build
>                                      whole-grid   whole-grid       BANDED
>   tangent_facet                          +17.74       +17.74        +4.06
>     + collimated carrier                 +19.74       +17.74        +4.06
>     + finite-radius carrier              +21.74       +17.74        +7.62
>   tangent_facet_remap                    +23.74       +23.74       +13.62
> ```
>
> At N = 32768 one grid is 8.59 GB, so route 3 with a carrier drops from
> **+187 GB to +65 GB** and the remap rung from **+204 GB to +117 GB**.
>
> **5. ONE OF THOSE GRIDS WAS ALREADY DEAD, AND THAT IS A FINDING, NOT A
> TUNING.**  The middle column above is not a typo: under these models the
> whole SCREEN-OBLIQUITY block was computing a correction nobody added
> (`_obl_apply` is False for them) into an accumulator nobody read (`_obl_total`
> is None and the guard is gated off).  It cost **+2.0 grids with a collimated
> carrier and +4.0 with a finite-radius one**, in the shipped build, for
> nothing.  Gating `_obl_active` off for the family is a dead-code removal
> whose byte-identity is part of item 2's 960 arms.
>
> **6. THE PULL-BACK IS REFUSED FROM THE BAND, AND THE REFUSAL IS PRICED AND
> PRINTED.**  The remap rung's SCREEN half bands; its second half -- resample
> the field at `x + W` -- does not.  Its halo is the WALK, which is a LENGTH:
> measured 93.0 / 67.3 / 31.7 um on the three faces of a design-121-like
> SSK2/SF57 doublet, i.e. **15 / 27 / 50 halo rows at dx = 8 / 4 / 2 um**
> against a 256-row auto band.  Three further steps are globally coupled, and
> S3 measures each rather than asserting it.  The refusal is reported through
> `progress` with the measured `max|W|`, the halo it implies and that as a
> percentage of the band actually in use.
>
> **7. IT FOLLOWS THE AUTO CONVENTION NOW, AND S6 ADJUDICATES THAT.**  Byte-
> identical to the shipped output, 4.4x lighter, and at wall-clock PARITY
> (0.947 / 0.995 / 1.007 banded-over-whole on the three best-sampled rows) --
> the same bargain `sag_chunk_rows` AUTO already makes for every other screen.
> The box carried a peer session's ten-worker pytest run from 16:10 onward, so
> S5.5 shows the run-to-run spread of a NULL comparison alongside the effect
> and bounds the claim at "not materially slower" rather than sharpening it.

---

## 1. THE HARD PART, NAMED BEFORE IT WAS DESIGNED

The brief named it and it is worth restating, because the answer is not the one
the shape of the question suggests.

**The remap's pull-back reads the field at `x + W`.**  `W` is a physical
displacement -- the transverse walk -- so its halo in ROWS is `|W| / dy` and
grows without bound as the grid refines at a fixed aperture.  A fixed 1-3 row
halo, which is what every other banded block in `_lens_real.py` uses, is not
the right shape of object at all.

What the design had to establish was therefore not "how wide" but "where the
walk actually enters".  It enters in exactly one place:

```
  the remap SURFACE BLOCK          the remap APPLY
  -----------------------          ---------------
  sag, grad sag, grad grad sag     A = I + dW/dx, det A
  (R4) hit point, (R5) normal      min(det)              <- fold guard
  (R1) OPD, (R2) walk W            the demodulating eikonal fit
  (R3) p_out                       spline_filter(W/dx)
  the phase screen                 the pull-back fixed point
  the vignetting masks             map_coordinates(field)
       ^                                ^
   pointwise in (sag, H, p):        reads the field |W| ROWS AWAY
   2-ROW SAG HALO, no p halo        and is globally coupled besides
```

So the walk does not enter the screen half at all.  The screen half bands on a
FIXED 2-row halo; the apply half is where the dynamic halo lives, and S3
prices it and refuses it.  That split is the whole design, and it is why the
answer is "band most of it" rather than "band none of it" or "band all of it
with a dynamic halo".

---

## 2. THE HALOS, DERIVED

Backwards from what a band must produce.  ROUTE 3:

```
  the ACCUMULATOR at rows [r0, r1)  =  p - grad(opd) there
    -> grad(opd) at [r0, r1)          -> opd at [r0-1, r1+1)
    -> the screen's five gradients (of dz, gx, gy, ox, oy) there
    -> dz / ox / oy -- and hence p -- at [r0-2, r1+2)
    -> grad sag at [r0-2, r1+2)
    -> sag at [r0-3, r1+3)

  SAG HALO 3 ROWS,  ACCUMULATOR HALO 2 ROWS.
```

THE REMAP RUNG:

```
  the ACCUMULATOR at [r0, r1)  =  p_out(sag, grad sag, grad grad sag, p) there
    -- CLOSED FORM (R3), no gradient of the screen --
    -> grad grad sag at [r0, r1)      -> grad sag at [r0-1, r1+1)
    -> sag at [r0-2, r1+2);  p at [r0, r1) ONLY.

  SAG HALO 2 ROWS,  ACCUMULATOR HALO 0 ROWS.
```

THE GAP TRANSPORT (both rungs): one gradient of the accumulator, **1 row**.

`_tf_rows_grad` is the primitive all three rest on.  `np.gradient` uses a
central difference in the interior and a one-sided difference at the array's
own first / last row, and its interior stencil does not know how tall the array
is -- so a slab reproduces the whole-grid gradient EXACTLY on every row except
a slab edge that is not also a grid edge.  Those two rows are dropped rather
than trusted, and the halo is CLIPPED at the true grid edges so rows 0 and
`Ny-1` keep the one-sided stencil that is correct for them.  That is the same
argument `test_slant_chunk_byte_identical` pinned for the refraction leg and
`test_obl_banded_halo` re-pinned at `cr = 1`, where every row is simultaneously
a band interior and a band boundary.

### 2.1 The staleness hazard is absent by construction, not handled

`BUILD_OBL_BANDED_HALO` S3.2a had to hold each band's `p0` write back one band,
because the next band's halo reads row `r0-1` at its PRE-surface value and an
in-place update had already overwritten it.  With a 2-row halo that deferral
would have needed a queue.

It is not needed at all here: the whole-grid path REBINDS (`_tf_px = _tf_px -
_tk_x` allocates), so the faithful mirror is a FRESH destination grid filled
band by band, and a band can then never read a row this surface has already
rewritten.  The same choice kills S3.2b's NEP-50 hazard outright -- every band
of a surface reads the same pinned source object, so the Python-float seed
cannot promote to a float32 array of zeros mid-loop.  Pinned anyway, on the
float32-geometry arm of both the test file and the adversarial sweep.

### 2.2 The one reduction that had to be argued rather than reproduced

The whole-grid block decides whether to run the thin-screen fallback with a
WHOLE-GRID reduction:

```python
    _tf_all_ok = bool(xp.all(_tf_ok)) and bool(xp.all(xp.isfinite(_tf_opd)))
    if not _tf_all_ok:
        _tf_opd = xp.where(_tf_ok & xp.isfinite(_tf_opd), _tf_opd, opd)
```

A band takes that reduction over its own rows, so the two can disagree: the
grid may be not-all-ok while a particular band is all-ok.  On those rows the
whole grid runs the `where` and the band skips it.  They agree element for
element anyway, and for a reason that has to be stated rather than assumed:
`xp.where` on an all-True mask returns the left operand's values exactly, and
the result DTYPE is `result_type` of the two.  The tangent-facet screen's dtype
is `result_type(sag, p)`; the thin screen's is `sag`'s alone, because `n2 - n1`
is a weak Python float.  **The tangent-facet screen is therefore never the
narrower of the two**, so `result_type` is its own dtype and the `where` is the
identity in bits as well as in values.  Were it ever the narrower, the band
would return a float32 array where the whole grid returned a float64 one.
Pinned by the `sag_dtype='float32'` x `carrier=` arms, which is exactly the
configuration where a promotion would show.

---

## 3. THE REFUSAL, PRICED

### 3.1 The walk halo is a length, measured

`_walk_tb.py`, instrumenting the shipped apply on a design-121-like SSK2/SF57
doublet (R = 12.6 / -9.8 / -40.0 mm, 54.9 mrad carrier, 3 mm pupil):

```
  N=  768 dx=8.0 um   auto band = 256 rows
      surface 0: max|W| =   92.236 um =    15 halo rows   ( 5.9 % of the band)
      surface 1: max|W| =   66.714 um =    12 halo rows   ( 4.7 %)
      surface 2: max|W| =   31.641 um =     7 halo rows   ( 2.7 %)
  N= 1536 dx=4.0 um
      surface 0: max|W| =   92.674 um =    27 halo rows   (10.5 %)
      surface 1: max|W| =   67.122 um =    20 halo rows   ( 7.8 %)
      surface 2: max|W| =   31.653 um =    11 halo rows   ( 4.3 %)
  N= 3072 dx=2.0 um
      surface 0: max|W| =   93.022 um =    50 halo rows   (19.5 %)
      surface 1: max|W| =   67.327 um =    37 halo rows   (14.5 %)
      surface 2: max|W| =   31.658 um =    19 halo rows   ( 7.4 %)
```

**`max|W|` is constant in METRES to three figures across a 4x refinement** --
which is the honest reading: it is a physical displacement, not a
discretisation, so the halo in rows scales as `1/dx` while the auto band scales
as `N/16`.  For a fixed physical window those two grow together, and the
asymptotic ratio is `16 |W| / (N dx)` = about 24 % on this fixture.  A band
whose halo is a quarter of its own height is not a band; it is a whole-grid
evaluation with extra bookkeeping.

**This is the number the design is priced on, and it is reported at run time**
rather than assumed: `_tf_price_walk_halo` measures `max|W|` for THIS call --
it cannot be known before the walk is computed -- and prints the halo it
implies against the band actually in use.

### 3.2 And the pull-back is globally coupled besides

Three separate reasons, each measured rather than asserted (`_walk_tb.py`,
second half):

**(a) `spline_filter` is an IIR.**  `map_coordinates` at order > 1 prefilters,
and that filter's output at one pixel depends on every pixel of the column,
decaying like `(-0.268)^k`.  On a 32-row band:

```
  halo    1 rows -> max |slab - whole| = 1.080e+00   DIFFERS
  halo    4 rows ->                      1.674e-02   DIFFERS
  halo   16 rows ->                      2.449e-09   DIFFERS
  halo   64 rows ->                      0.000e+00   IDENTICAL
  halo  128 rows ->                      0.000e+00   IDENTICAL
```

Byte-identity is reachable -- at a halo TWICE the band, and only because the
decay underflows for this data.  "It underflows on my fixture" is not a
byte-identity argument, and a halo twice the band is the regime where banding
buys nothing.  **Both halves of that are disqualifying on their own.**

**(b) The demodulating eikonal's six moments are whole-grid `np.sum`.**
numpy's pairwise summation is not re-associable in general:

```
  band    1 rows -> sum - whole = +5.821e-11   DIFFERS
  band   32 rows ->               +0.000e+00   IDENTICAL
  band  256 rows ->               +0.000e+00   IDENTICAL
```

Again: it happens to agree at two of three band sizes.  "Happens to agree" is
not a proof, and the one that differs is `cr = 1`, which is the band size the
whole halo argument is stress-tested at.

**(c) `min(det)` is the fold guard**, and it must refuse the CALL.  A band that
has not yet reached the folding row would run and return a field.  This one is
structural: no halo width is even relevant to it.

### 3.3 What the refusal costs, and what it does not

`BUILD_TF_REMAP` S6 measured the rung's surcharge over route 3 at a constant
**+6.00 grids** -- the two walk components, the determinant, the two pull-back
coordinate grids and the demodulated copy.  This build reproduces that exactly
in the whole-grid column (23.742 - 17.742 = +6.00 at N = 4096, 23.358 - 17.358
= +6.00 at 2048), and the BANDED surcharge is **+9.56** (13.618 - 4.056; +9.12
at 2048).

That the surcharge GREW is the right answer, not a regression: banding removes
transients from both models, but the ones it removes from route 3 are its whole
screen pipeline while the ones it removes from the rung are only its screen
half.  The apply half's fixed set is unbanded on both sides of the subtraction,
so it survives into the difference at full width.  What matters is the total:
banding the screen half alone takes the rung from +23.74 to +13.62, i.e.
**43 % of the model's memory came off even though its single most expensive
block was refused.**

---

## 4. THE DEAD BLOCK

Not sought; found while reading `_obl_active`'s call sites for the halo work.

`_check_screen_obliquity_support` returns False for the tangent-facet family
(they supersede equations 4 and 7 rather than composing with them), and
`_obl_total` is left None for them, and the guard block at the end of the call
is gated on `not _tf_active`.  So under those models the obliquity block ran
with `_obl_apply` False and `_obl_total` None -- computing `_screen_obliquity_
delta` and discarding it, and accumulating `_obl_p0*`, which no surviving
reader touches.  `_obl_gap_advance` is gated on `_obl_apply` and so never ran.

Measured on the SHIPPED tree at N = 4096, extras over the banded thin baseline:

```
  tangent_facet                          +17.742
  tangent_facet + collimated carrier     +19.742     <- +2.0 for nothing
  tangent_facet + finite-radius carrier  +21.742     <- +4.0 for nothing
```

The +2.0 is the `_obl_p0*` pair; the extra +2.0 at a finite radius is the
materialised carrier momentum field.  **The published +21.8 is reproduced to
the printed digit, which is what makes this a reading of the shipped build
rather than a claim about it.**

Gating `_obl_active` off for the family removes both.  The change is
byte-null -- pinned in the 960-arm two-tree comparison, and structurally by
`test_the_screen_obliquity_block_is_dead_under_this_family`, which asserts
`_screen_obliquity_delta` is never CALLED under either model with a carrier,
with the `'thin'` control asserting it still is.  `test_the_carrier_still_
seeds_the_accumulator` is the other control: gating the block off must not gate
the CARRIER off, and the field must still depend on it.

---

## 5. THE MEASUREMENTS

### 5.1 Byte identity -- the load-bearing one

`_adv_tb.py`, run once against the finished build.  8 prescriptions (singlet,
biconvex, plate, oblate k=+4, leading-plate triplet, a triplet with a
DECENTERED face that falls through mid-prescription, a per-surface clear
aperture, a conic+aspheric asphere) x 4 grids (96 / 65 odd / 72 with
`dx != dy` / 129 odd with `dx != dy`) x 2 models x 3 carriers (none, collimated
tilt, finite radius) x 7 option combinations (plain, fresnel, fresnel +
absorption, bandlimit, absorption, float32 geometry, fresnel + float32) x 2
complex dtypes x band sizes {1, 3, N, N+7}; plus the fold ladder and the
aperture-stop arms.

```
  total 10778   fails 0
```

Compared per configuration: the returned field (`np.array_equal` on the bytes),
its dtype, and the sorted set of warnings -- or, where the call refuses, the
exception type and message.

### 5.2 Two-tree, against the SHIPPED whole-grid output

`_twotree_tb.py`, the same arm list executed under both library roots with the
path assert keyed to each.  5 prescriptions x models x carriers x
`sag_chunk_rows` in {0, 1, 7, AUTO} x 8 option combinations, compared on a
SHA-256 of the returned field plus its dtype plus the warning set:

```
  'thin' + 'displaced'                     640 arms   0 differ
  'tangent_facet' + 'tangent_facet_remap'  960 arms   0 differ
```

The second line is the one that matters: on `origin/main` those models ignore
`sag_chunk_rows` entirely, so every banded arm of this build is being compared
against the SHIPPED whole-grid answer, not against this build's own.

### 5.3 The suite

```
  tests/unit/test_tf_banded_halo.py                    68 passed   (new)
  tests/unit/test_tangent_facet.py                     68 passed   (unchanged)
  tests/unit/test_tangent_facet_remap.py               36 passed   (unchanged)
```

`test_sag_chunk_rows_is_INERT_for_this_model` and
`test_sag_chunk_rows_is_inert_for_this_model` -- the two pins the shipped
builds left specifically so this change could not land silently -- PASS
UNCHANGED, because they assert `np.array_equal` of the outputs and that is
exactly what this build preserves.  Neither was edited, retargeted or relaxed.

### 5.4 Memory

Warmed `tracemalloc` peak in float64 grids (`8*N*N` bytes), biconvex singlet
R = +19.6 / -27.4 mm N-SSK2 -- the same fixture as `BUILD_TANGENT_FACET` S5 and
`BUILD_TF_REMAP` S6.  Two protocol notes carried over from those builds and one
new one:

* every arm is WARMED at its own N before anything is measured (the first
  `apply_real_lens` of a process also pays FFT-plan and lazy-import
  allocations, ~10 grids at N >= 4096);
* memory and wall clock are measured in SEPARATE passes (`tracemalloc`'s
  per-allocation hook inflates the remap rung's wall clock ~20x);
* **the physical WINDOW is held fixed at 8.192 mm** rather than `dx`.  With
  `dx = 4 um` fixed, N = 4096 puts the grid edge at 8.2 mm on an R = 19.6 mm
  face, the walk map genuinely FOLDS there and the remap rung correctly
  REFUSES -- which would have been read as a probe failure instead of as the
  guard working.  It cost one run to find and is recorded rather than quietly
  fixed.

```
  N = 2048   band = 256                     grids     extra over thin@same band
    thin      banded                        4.768        +0.000
    thin      whole                        12.127        +7.358
    tangent_facet          banded           8.881        +4.112
    tangent_facet          whole           22.127       +17.358
      + collimated carrier banded           8.881        +4.112
      + collimated carrier whole           22.127       +17.358
      + sphere carrier     banded          12.003        +7.235
      + sphere carrier     whole           22.127       +17.358
    tangent_facet_remap    banded          18.004       +13.236
    tangent_facet_remap    whole           28.127       +23.358
      + sphere carrier     banded          18.004       +13.236
      + sphere carrier     whole           28.127       +23.358

  N = 4096   band = 256
    thin      banded                        4.384        +0.000
    thin      whole                        12.126        +7.742
    tangent_facet          banded           8.440        +4.056
    tangent_facet          whole           22.126       +17.742
      + collimated carrier banded           8.440        +4.056
      + collimated carrier whole           22.126       +17.742
      + sphere carrier     banded          12.002        +7.618
      + sphere carrier     whole           22.126       +17.742
    tangent_facet_remap    banded          18.002       +13.618
    tangent_facet_remap    whole           28.126       +23.742
```

**The instrument reproduces every shipped anchor before it is trusted on
anything new.**  Differencing the way the two shipped builds did:

```
  N = 2048, against the WHOLE-GRID thin baseline
    tangent_facet        22.127 - 12.127 = +10.00   BUILD_TANGENT_FACET: +10.0
    tangent_facet_remap  28.127 - 12.127 = +16.00   BUILD_TF_REMAP:      +16.00
  N = 4096, against the BANDED thin baseline
    tangent_facet        22.126 -  4.384 = +17.74   BUILD_TANGENT_FACET: +17.8
                                                    BUILD_TF_REMAP:      +17.74
    tangent_facet_remap  28.126 -  4.384 = +23.74   BUILD_TF_REMAP:      +23.74
  and on the SHIPPED tree, run directly:
    tangent_facet + finite-radius carrier   +21.742  BUILD_TANGENT_FACET: +21.8
```

Five anchors, five reproductions.  The `+21.742` reading is also what
identifies the shipped `+21.8` arm as a finite-radius carrier, which is what
makes the S4 comparison an apples-to-apples one.

At N = 32768 one grid is 8.59 GB:

```
  tangent_facet, finite-radius carrier   +187 GB  ->   +65 GB
  tangent_facet, no carrier              +152 GB  ->   +35 GB
  tangent_facet_remap                    +204 GB  ->  +117 GB
```

### 5.5 Wall clock

**The box was shared from 16:10** -- a peer session started ~13 concurrent
python jobs and the count reached 19 while these ran -- so an A-then-B timing
would have read the load rather than the code.  The protocol is therefore
INTERLEAVED: the banded and whole-grid arms alternate inside one process for
`reps` rounds and the MIN of each is taken.  Contention that hits a round hits
both of its arms, and the min is the least-contended observation of each.  The
RATIO is what is quoted; the absolute seconds are reported but are not
clean-box numbers.

```
  N = 2048, band 256
    thin                 3 reps   banded  1.233 s   whole  1.072 s   1.150
    tangent_facet        3 reps   banded  6.854 s   whole  6.315 s   1.085
    tangent_facet_remap  3 reps   banded 36.594 s   whole 36.790 s   0.995

  N = 4096, band 256
    thin                 3 reps   banded  4.165 s   whole  4.635 s   0.898
    thin                 5 reps   banded  4.426 s   whole  4.394 s   1.007
    tangent_facet        3 reps   banded 28.105 s   whole 23.132 s   1.215
    tangent_facet        5 reps   banded 22.659 s   whole 23.917 s   0.947
    tangent_facet_remap                  NOT OBTAINED -- see below
```

**The 3-rep and 5-rep rows at N = 4096 are both shown on purpose.**  The 3-rep
`tangent_facet` reading of 1.215 is what a first pass produced, and it would
have been the number in this note had it not been re-measured: with 5 reps the
same pair reads 0.947, and the `thin` control moves 0.898 -> 1.007 across
those same passes.  So the run-to-run spread of a NULL comparison is +-10 %,
which is the size of the effect being measured.

Read honestly, then: the banded arm is **at parity** -- 0.947 / 0.995 / 1.007
on the three best-sampled rows -- and the defensible claim is **"not materially
slower"** rather than "faster", because on a box carrying a peer session's
ten-worker pytest run no tighter claim is supportable.  What is not in doubt is
the memory, which `tracemalloc` measures by counting allocations and which
contention cannot move.

**The N = 4096 remap row is NOT REPORTED, not approximated.**  Its whole-grid
arm alone runs ~180 s under this load and the interleaved pair did not complete
before the peer job's `pytest tests/unit -n 10` had been saturating all cores
for over an hour.  A contended single-rep number printed next to two well-
sampled rows would invite exactly the comparison it cannot support -- the same
call `BUILD_TF_REMAP` S6 made about its own N = 8192 wall clock.  What the
measured rows establish stands on its own: the rung is at parity at N = 2048
(0.995), and the banded/whole ratio for a model is set by the band-loop
overhead, which does not grow with N (the `thin` and `tangent_facet` rows are
flat across the 2048 -> 4096 step).

---

## 6. THE ADJUDICATION: THE AUTO CONVENTION

`sag_chunk_rows=None` -> AUTO (banded at N >= 4096) is what every other screen
in `apply_real_lens` does, and the brief left it open whether this family
should join.  It does, and the reasoning is:

* **byte-identity is not a hope here, it is measured against the SHIPPED
  output** (S5.1, S5.2).  The usual reason to hold a new path back -- "it is
  probably the same" -- does not apply;
* the memory win is 4.4x on route 3 and 1.7x on the remap rung, and at the
  grid sizes AUTO engages (N >= 4096) that is the difference between a run
  that fits and one that does not.  `BUILD_TANGENT_FACET` S5 called banding
  "the honest reason banding is the first follow-on" for exactly this;
* the wall clock is within the measurement's own spread (S5.5);
* and holding it back would mean shipping a path that is byte-identical,
  strictly lighter, and reachable only by an explicit keyword -- which is the
  configuration users do not find.

**What this changes for an existing caller:** a `surface_model='tangent_facet'`
or `'tangent_facet_remap'` call at N >= 4096 that passed `sag_chunk_rows` not
at all now bands, and uses less memory for the same bits.  A caller who wants
the old allocation profile passes `sag_chunk_rows=0`, which is the same escape
hatch every other screen has.  Nothing about the DEFAULT `surface_model='thin'`
moves at all -- 640 arms, 0 differences.

The runner preflight (`Reverse_Symmetric_ASM/tx_design_study_sim.py`, outside
the repo, not git) needs its version-gated tangent-facet term re-pointed at the
new coefficients; see S8.

---

## 7. WHAT WAS REFUTED ALONG THE WAY

| # | attack / candidate | outcome |
|---|---|---|
| 1 | band the remap's PULL-BACK too, with a halo sized from `max\|W\|` | **REFUSED, and priced.** The halo is 15/27/50 rows at dx = 8/4/2 um and grows as `1/dx`; and the `spline_filter` IIR needs a 64-row halo on a 32-row band before it underflows to identical, the moments are whole-grid `np.sum`, and `min(det)` must refuse the CALL. Byte-identity would have been a data-dependent accident at a halo twice the band. (S3) |
| 2 | reuse `BUILD_OBL_BANDED_HALO`'s deferred in-place accumulator write | **NOT TAKEN, and it turned out not to be needed.** The whole-grid path REBINDS the accumulator, so a fresh destination is the faithful mirror AND makes both the staleness hazard and the NEP-50 promotion hazard structurally absent instead of handled. (S2.1) |
| 3 | "a 1-row halo, like the slant path" | **WRONG for both rungs, and by different amounts.** Route 3 needs 3 rows of sag, the remap 2. Controlled by narrowing each by one row: route 3's field MOVES, the remap's Hessian level runs out of rows and raises. Both are non-reproductions; `test_a_narrower_halo_breaks_it` accepts either and rejects "passes anyway". |
| 4 | assume the remap needs the WIDER halo, since it is the more expensive model | **BACKWARDS.** (R3) hands the remap its accumulator in closed form, so it needs 2 rows of sag and NO momentum halo against route 3's 3 and 2. |
| 5 | the band-local `all(ok)` reduction is a different decision from the whole-grid one | **TRUE, AND HARMLESS -- but only after the dtype argument.** `xp.where` on an all-True mask is the identity in bits only if the result dtype is the left operand's, which holds because the tangent-facet screen is never narrower than the thin one. Stated rather than assumed, and pinned on float32 geometry. (S2.2) |
| 6 | measure the memory with `dx` fixed across N, as the shipped builds' tables read | **BROKE THE PROBE.** At N = 4096 the grid edge reaches 8.2 mm on an R = 19.6 mm face and the remap correctly REFUSES a genuine fold. Fixed by holding the physical WINDOW constant; recorded because a "probe failure" here was the guard working. |
| 7 | quote a wall clock from a first A-then-B pass | **NOT TRUSTWORTHY.** The box was shared from 16:10. Re-run interleaved with min-of-reps, and even then the run-to-run spread of the `thin` control (1.150 / 0.898 / 1.007) is the size of the effect, so the claim is bounded at "not materially slower" rather than sharpened. (S5.5) |
| 8 | the adversarial sweep's aperture-stop arms passed on the first run | **THEY PASSED VACUOUSLY.** `aperture=` / `stop_index=` are PRESCRIPTION keys, not kwargs, so both arms raised `TypeError` and the comparison compared two identical refusals. Found by porting the same arms into the test file, fixed in both, re-run. A green that comes from a refusal on both sides is not a green. |
| 9 | the sweep's `slant_correction` arms likewise | **ALSO VACUOUS, and correctly so.** `slant_correction=True` is REFUSED with these models (both replace the same facet coefficient), so every such arm was RAISE-vs-RAISE. Replaced with `fresnel=True`, which is the option that actually routes the family through the SECOND band gate (`_slant_narrow_chunk`). Without that substitution the whole fresnel/TIR/aperture band path would have been untested while looking covered. |

---

## 8. FILES

| file | change |
|---|---|
| `lumenairy/elements/_lens_real.py` | the halo derivation block and `_TF_SAG_HALO_ROWS` / `_TF_MOM_HALO_ROWS` / `_TF_REMAP_SAG_HALO_ROWS` / `_TF_GAP_HALO_ROWS`; `_tf_sl` / `_tf_rows_grad` / `_tangent_facet_screen_rows` / `_tangent_facet_transport_rows` (the two whole-grid entry points become wrappers at `lo, hi = 0, Ny`); `_obl_active` gated off for the family; `_obl_any_sag` renamed `_band_any_sag` and shared; the `_tf_begin_surface` / `_tf_store` / `_tf_end_surface` / `_tf_band_screen` / `_tf_gap_transport` / `_tf_price_walk_halo` closures; `not _tf_active` removed from both band gates and the tangent-facet block wired into both band loops; the whole-grid gap transport routed through `_tf_gap_transport`; the `surface_model` and `sag_chunk_rows` docstrings |
| `tests/unit/test_tf_banded_halo.py` | NEW.  68 tests |
| `CHANGELOG.md` | `[Unreleased]` |
| `docs/audits/BUILD_TF_BANDED_2026_08_16.md` | this note |

Not in git, in the scratchpad: `_adv_tb.py` (S5.1), `_twotree_tb.py` (S5.2),
`_mem_tb.py` (S5.4), `_time_tb.py` (S5.5), `_walk_tb.py` (S3).

---

## 9. SUITES

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X
python 3.14.6   numpy 2.4.4      lumenairy 5.36.1 (worktree C:/tmp/lum_tb,
scipy 1.17.1                     branch feat/tangent-facet-banded off
numexpr 2.14.1                   origin/main dea5e47)
```

| gate | result |
|---|---|
| `test_tf_banded_halo.py` (NEW) | **68 passed** |
| `test_tangent_facet.py` + `test_tangent_facet_remap.py` (UNCHANGED, incl. both INERT pins) | **104 passed** |
| the six byte-identity-critical files `BUILD_TANGENT_FACET` S7 nominated for this area (`test_screen_obliquity` + `test_obl_banded_halo` + `test_slant_chunk_byte_identical` + `test_lens_chunked_sag` + `test_tangent_facet` + `test_niche_audit_e_prepared_and_enums`) + `test_tangent_facet_remap` + the new file + `test_audit_lens` (the docstring/signature audits, because this build rewrote two `surface_model` docstring paragraphs) | **463 passed** in 482 s |
| `test_obl_banded_halo` + `test_slant_chunk_byte_identical` re-run alone | **135 passed** |
| adversarial whole-grid-vs-banded sweep | **10778 configurations, 0 differences** |
| two-tree vs `origin/main` -- tangent-facet family | **960 arms, 0 differences** |
| two-tree vs `origin/main` -- `'thin'` + `'displaced'` | **640 arms, 0 differences** |
| `ruff check lumenairy/ tests/` | **All checks passed** |
| `xfail` / `skip` added | **ZERO** |
| pre-existing assertions relaxed or retargeted | **ZERO** |

**Path pinning.**  Every run in this note was made with `PYTHONPATH` pinned to
the worktree and `lumenairy.__file__` asserted in-process before anything else;
the two-tree comparisons ran the SAME script under both roots with that assert
keyed to each root, so a silently-imported installed wheel cannot have produced
any of these numbers.

**Box contention.**  A peer session started `pytest tests/unit -q -n 10` on
this box at 16:10 and it was still saturating all cores hours later (process
count 14-19 throughout).  Memory is unaffected -- `tracemalloc` counts
allocations, not seconds -- and every memory number here is therefore a clean
reading.  Wall clock is affected, and S5.5 both bounds its claim accordingly
and declines to report the one row that did not finish.  Two suite runs that
would normally take minutes ran for over an hour; that is the load, not this
branch, and it is recorded rather than diagnosed as something else.

**A NOTE ON WHAT WAS MEASURED AGAINST WHAT.**  Every number in S5 was produced
against the tree that is committed.  One cosmetic reorder (moving
`_obl_begin_surface()` between two blocks carrying the identical `if _obl_here`
guard) was made after the sweeps had run and was REVERTED rather than
re-measured, so the committed tree is byte-for-byte the tree the 10778-config
sweep, the two two-tree comparisons and the 463-test run all exercised.  The
reordered form was green too (68 + 135 passed) -- it is simply not what is
being claimed about.

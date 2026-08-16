# BUILD -- the REMAP rung of the tangent-facet screens

**2026-08-16.  Branch `feat/tangent-facet-remap`, cut from `origin/main`
@ `97d431f` = v5.36.0, in the worktree `C:/tmp/lum_rm`.  Commit on the branch
only -- no merge, no push, no `gh`.**

The axis `BUILD_TANGENT_FACET_2026_08_16` S4 named as the next one: route 3
leaves the TRANSVERSE WALK, and a vertex-plane screen can reference that away
but never represent it.  This represents it.

---

## 0. VERDICT

> **SHIPPED AS `surface_model='tangent_facet_remap'`, OPT-IN AND DEFAULT-OFF.
> DESIGN 121 GROUP 5 AT 3 mm GOES FROM 0.0032381 TO 2.56e-08 WAVES RMS --
> 39000x UNDER THE 0.001 BAR, AND 14500x BELOW THE "PRIZE" THAT
> BUILD_TANGENT_FACET PROVED NO SCREEN COULD REACH.  THE REASON IT IS THAT
> LARGE IS THAT THE REMAP REMOVES A TRUNCATION RATHER THAN ADDING A TERM.**
>
> **1. THE OPD COLLAPSED.**  Route 3's three terms -- the axial-translation
> identity, the facet-at-crossing back-extrapolation, the second-order walk
> referencing -- are what ONE exact line becomes after `S_out` is Taylor-
> referenced back to the pixel and truncated.  Evaluate it where the ray
> actually lands instead and the screen is just the path difference,
> `OPD = s (n2^2/pz2 - n1^2/pz1)`, with the walk `W = s (p/pz1 - p_out/pz2)`.
> The identity `OPD_remap = (T1) + (T2) - p_out . W` holds term for term.
>
> **2. THE PLANE-FACET IDENTITY IS EXACT, AND SO ARE THE WALK AND THE EXIT
> MOMENTUM.**  Measured against closed-form eikonals on the SAME 27 cells route
> 3 used (slopes 0.05 / 0.12 / 0.24, `(n1,n2)` including `1.8047 -> 1.0000`,
> tilts 0 / 55 / 150 mrad): **worst 5.95e-14 relative** over OPD, walk and
> `p_out` together.  No oracle, no common-mode control, no tolerance to choose.
>
> **3. IT IS A LAGRANGIAN MODEL, WHICH BUILD_TANGENT_FACET S0.1 PROVED NO
> SCREEN IS.**  A screen's kick is the gradient of its own value.  A remap's is
> the gradient of the COMPOSITE, and with `A = I + dW/dx` that is `A^T p_out`
> IDENTICALLY -- the screen supplies `A^T p_out` and the coordinate change
> divides `A^T` back out.  On a plane facet the two independently-computed sides
> agree to **1.5e-15 - 4.6e-14** relative.  On a curved one the remap's gap is
> its own hit-point truncation and is **365x - 591x** below route 3's
> screen-kick-vs-exact-kick gap on the identical fixture.
>
> **4. THE ACCEPTANCE TABLE, waves rms against exact rays, 3 mm pupil.**
>
> ```
>   group      blind    v5.35.5   tangent_facet       REMAP    facet arm
>   g2      0.016799   0.000498       0.0000046    1.12e-10    0.0000196
>   g3      0.009908   0.000020       0.0000033    5.03e-11    0.0000159
>   g4      0.000489   0.000027       0.0000005    4.17e-12    0.0000004
>   g5      0.258480   0.012398       0.0032381    2.56e-08    0.0003724
> ```
>
> **The 0.001-wave bar is met at 1 mm, 2 mm AND 3 mm** (2.53e-11 / 1.61e-09 /
> 2.56e-08), where route 3 met it only at 1 and 2.  No group and no pupil
> regressed.  Improvement over route 3 at the 3 mm pupil, where both models'
> numbers are large enough for a ratio to mean anything: **g2 41000x, g3
> 67000x, g4 126000x, g5 126000x**.  (At 1 and 2 mm route 3's own figures round
> to 0.0000000-0.0000008 in its published table, so ratios there would be
> reading its rounding rather than this model's gain, and are not quoted.)
>
> **5. THE RESIDUAL IS ONE NAMED TERM, AND ITS BOUND IS MEASURED.**  The
> 2.56e-08 is the (R4) hit-point fixed point's truncation: pushing it to
> convergence with a per-pixel Newton reads **5.67e-12**.  That Newton needs an
> analytic sag off the grid, which is the sag-source restriction route 3's S1.6
> already refused; at 39000x under the bar there is nothing to buy.
>
> **6. THE FOLD GUARD REFUSES, AND THE REFUSAL EARNS ITS KEEP.**  A remap is a
> ray map, so it must be single-valued -- exactly `det(I + dW/dx) > 0`.  On an
> ENGINEERED folding prescription the guard fires; with both bars removed the
> same call returns a **NON-FINITE** field, and `x + W(x) = u` genuinely has
> **4 roots** there.  Design 121's own interiors run at `det` in [0.927, 1.021].
>
> **7. THE OFF PATHS DID NOT MOVE.**  `'thin'` and `'tangent_facet'` are
> **byte-identical to the v5.36.0 tree across a 72-arm two-tree comparison**,
> and a plane plate is byte-identical to the thin screen at every tilt.
>
> **8. WHAT IT COSTS.**  See S6.  The end-to-end arm against
> `apply_real_lens_traced` is ORACLE-limited, not model-limited, and S5 proves
> that by refining the grid and watching the residual RISE.

---

## 1. THE DERIVATION

### 1.1 What route 3 truncates

Along a ray the eikonal grows as `dS = p . dx + pz dz`.  A pixel's ray leaves
the vertex plane at `x`, rises to the surface at `(x + s q, s)` with
`q = p/pz1`, refracts, and descends to `(x + W, 0)`.  Both legs are straight, so
both eikonal increments are closed form:

```
    rise:     p . (s q) + pz1 s  =  s n1^2 / pz1
    descend:  -s n2^2 / pz2
    =>        S_out(x + W) = S_in(x) + s (n1^2/pz1 - n2^2/pz2) .
```

Route 3 needs `S_out` at `x`, not at `x + W`, because its screen lives on the
vertex plane.  So it Taylor-references:

```
    S_out(x) = S_out(x+W) - p_out . W - (1/2) W .(W . grad) p_out - ...
```

`(T1)` and `(T2)` are what the first two pieces become; `(T3)` is the third; the
series is cut there.  **That cut is the entire residual** -- and on group 5's
exit face `|W|` reaches 54.9 um across a 3 mm pupil, so it is not a small cut.

### 1.2 The remap does not truncate

Keep the field at `x + W` and the referencing disappears:

```
    OPD = S_in(x) - S_out(x + W) = s (n2^2/pz2 - n1^2/pz1) ,             (R1)
    W   = s (p/pz1 - p_out/pz2) .                                        (R2)
```

Consistency with route 3 is exact and algebraic:
`OPD_R1 = (T1) + (T2) - p_out . W`, i.e. **(R1) is (T1)+(T2) with the
referencing term restored in full instead of expanded to second order.**  (T3)
is not dropped -- it is subsumed.

Reduction checks: `s -> 0` gives `OPD -> 0`; at normal incidence on a flat facet
`pz1 = n1`, `pz2 = n2`, and `OPD = s (n2 - n1)`, the thin screen exactly.

### 1.3 (R3) -- why the kick is now allowed to be exact

The composite operation on the field is: multiply by `exp(i k0 (S - OPD))`, then
move the value at `x` to `u = x + W(x)`.  The new eikonal is
`S_new(u) = S(x(u)) - OPD(x(u))`, so `grad_u S_new = A^{-T}(p - grad OPD)`.
Setting that equal to the exact refracted `p_out` is the statement

```
    grad_x [ S_in - OPD ]  =  A^T p_out ,      A = I + dW/dx .           (R3)
```

`(R3)` holds identically when `s` and the facet normal are exact.
`BUILD_TANGENT_FACET` S0.1 measured that a SCREEN cannot satisfy the analogous
statement -- its kick is `grad OPD` by construction, and forcing the exact facet
kick on top moves group 5 from 0.000372 to 0.009305.  A remap has an extra
`A^T` to spend, and spends it exactly.

### 1.4 (R4)/(R5) -- the hit point is a fixed point, not a series

`s` solves `s = sag(x + s q)`.  With `a = grad sag . q` and
`b = q^T (grad grad sag) q`:

```
    s = sag/(1-a) + (b/2) [sag/(1-a)]^2 / (1-a) ,                        (R4)
    grad sag |_hit = grad sag + (grad grad sag)(s q) .                   (R5)
```

`(R4)`'s first piece is the EXACT solution for a plane facet (`b = 0`), which is
why S0.2's plane-facet check is machine-exact rather than approximately zero --
route 3's corresponding step is a Taylor expansion and is not.

Both pieces are load-bearing, measured by breaking each (design 121, 3 mm pupil,
waves rms; `tangent_facet_derive.py remap_ladder`):

| group | route 3 | hit at vertex | (R4) linear only | (R5) dropped | **SHIPPED** | Newton |
|---|---|---|---|---|---|---|
| 5 | 3.238e-03 | 1.144e-01 | 5.666e-04 | 1.657e-04 | **2.560e-08** | 5.674e-12 |
| 2 | 4.645e-06 | 1.754e-02 | 1.490e-05 | 4.008e-06 | **1.121e-10** | 3.254e-11 |
| 3 | 3.347e-06 | 1.131e-02 | 1.709e-05 | 1.019e-06 | **5.031e-11** | 3.391e-11 |
| 4 | 5.257e-07 | 1.525e-03 | 4.231e-07 | 1.761e-08 | **4.168e-12** | 4.116e-12 |

Dropping (R5) alone costs **6500x** on group 5.  Note the middle two columns are
both WORSE than route 3 on g2/g3: a half-built hit point is a defect, exactly as
route 3's S1.3 found for its own pair.

### 1.5 The amplitude Jacobian, derived

The remap is a coordinate transform of a field, so it must move ENERGY:
`|E_out(u)|^2 d^2u = |E_in(x)|^2 d^2x` with `d^2u = |det A| d^2x`, hence

```
    |E_out(u)| = |E_in(x)| / sqrt(|det A(x)|) ,
```

the reciprocal square root of the FORWARD determinant at the SOURCE point --
the same factor and the same evaluation point `_apply_displaced_remap_2d`
already uses for the `'displaced'` 2-D walk remap.  Nothing is fitted and
nothing is renormalised; the power is measured afterwards as a CONSEQUENCE
(S3.3).  Momentum is not a density and carries no factor.

### 1.6 The resampling, and why the field is demodulated first

The pull-back `x(u)` solves `x + W(x) = u` by fixed point.  `W` contracts
exactly while the map is unfolded, so the iteration's convergence and the fold
guard are the same statement -- which is why a non-convergence is also a
refusal.  The field is then sampled there with `scipy.ndimage.map_coordinates`,
the library's standing high-order resampler (`_apply_displaced_remap_2d`,
`_lens_imap`), at `remap_order` (1 / 3 / 5, default 3).

A lens-interior field runs at a few pixels per fringe, and a spline through that
is where the model's accuracy would go.  So the field is DEMODULATED by an
analytic quadratic eikonal fitted to its own momentum (`A^T p_out`, weighted by
`|E|^2`), resampled, and remodulated -- a similarity transform, so it cannot
change the physics, only the interpolation error.  The fit shares its cross term
between the two momentum components, which is the curl-free constraint that
makes it a phase; it is therefore EXACT for a linear momentum field, the same
condition `_tangent_facet_transport`'s gap transport is exact under.  Because
`Phi` is analytic, the remodulation at the pulled-back point costs no second
interpolation and adds no second interpolation error.

**The sign was measured, not assumed.**  The library's field is
`A exp(+i k0 S)` with `p = grad S`: a plane wave `exp(+i k0 p x)` propagated
through the library's own ASM moves its centroid **+200.3 um** against a
+200.0 um geometric prediction at `p = 0.05`, `z = 4 mm`.  The first
implementation had it backwards, which DOUBLES the fringe rate instead of
flattening it; it was caught by the power it destroyed (0.944 of the input
against 0.9999) rather than by reading the code.

---

## 2. THE MEASUREMENTS

Same instrument as `BUILD_TANGENT_FACET`: a Hamiltonian ray system on a REGULAR
BUNDLE, exit-plane common-mode control `D(theta) - D(0)` with piston and tilt
removed, every gradient taken as the same physical gradient via the bundle's own
Jacobian.  Two lines change for this model -- the position advances by `W` at
each screen, and the momentum becomes `p_out`.

**The instrument is calibrated before it is trusted**: on the same fixtures it
reproduces `BUILD_TANGENT_FACET`'s published blind (0.258480), v5.35.5 and route
3 (0.0032381) numbers to every printed digit.

### 2.1 Design 121, the four powered groups x the three pupils

`tangent_facet_derive.py remap`, waves rms:

| group | pupil | blind | route 3 | **REMAP** | facet arm |
|---|---|---|---|---|---|
| 2 | 1 mm | 0.001135 | 0.0000000 | **3.12e-11** | 0.0000002 |
| 2 | 2 mm | 0.005807 | 0.0000008 | **3.28e-11** | 0.0000037 |
| 2 | 3 mm | 0.016799 | 0.0000046 | **1.12e-10** | 0.0000196 |
| 3 | 1 mm | 0.000823 | 0.0000000 | **3.27e-11** | 0.0000002 |
| 3 | 2 mm | 0.003747 | 0.0000005 | **3.32e-11** | 0.0000031 |
| 3 | 3 mm | 0.009908 | 0.0000033 | **5.03e-11** | 0.0000159 |
| 4 | 1 mm | 0.000031 | 0.0000000 | **4.23e-12** | 0.0000000 |
| 4 | 2 mm | 0.000166 | 0.0000001 | **4.33e-12** | 0.0000001 |
| 4 | 3 mm | 0.000489 | 0.0000005 | **4.17e-12** | 0.0000004 |
| **5** | **1 mm** | 0.024564 | 0.0000564 | **2.53e-11** | 0.0000068 |
| **5** | **2 mm** | 0.104356 | 0.0005963 | **1.61e-09** | 0.0000651 |
| **5** | **3 mm** | 0.258480 | 0.0032381 | **2.56e-08** | 0.0003724 |

Converged in the bundle: g5 at 3 mm reads 3.79e-08 / 2.56e-08 / 2.55e-08 /
2.58e-08 / 2.59e-08 at n = 33 / 65 / 129 / 257 / 513.

### 2.2 The walk this model represents

Per surface on group 5 at a 3 mm pupil, and the determinant of its map:

```
  surface   n            |W| max     det(I + dW/dx) range
  0         1.000->1.592   18.15 um  [0.99987, 1.02138]
  1         1.592->1.805    2.18 um  [0.99999, 1.00362]
  2         1.805->1.000   54.87 um  [0.92712, 1.00017]
```

The exit face carries the term, which is why the remap runs at the LAST surface
too rather than stopping one short.  `det` departs from 1 by up to 7.3 %, i.e.
the amplitude Jacobian is a 3.7 % effect and not a formality -- and it never
approaches zero, which is the design contract this model is for.

### 2.3 The Lagrangian claim, both sides

`A^T p_out` against `p - grad OPD`, computed independently, relative to `|p|`:

```
  PLANE facet (model exact)      n = 41/81/161, slopes 0.05 and 0.24
      1.53e-15  3.61e-15  3.89e-15  8.33e-15  1.31e-14  4.65e-14

  CURVED facet R = 12.6 mm N-SF57, 100 mrad, 3 mm semi-pupil
      h (um)      REMAP (R3) gap    route 3 screen-vs-exact-kick    ratio
      75.00           1.770e-04                   6.469e-02          365x
      37.50           1.334e-04                   7.202e-02          540x
      18.75           1.285e-04                   7.589e-02          591x
```

The remap's curved-facet gap converges to a FLOOR (its own (R4)/(R5)
truncation) rather than to zero, which is the honest shape of the claim and is
what the test asserts.

### 2.4 The grid, which is what limits the realisation

The bundle arm above feeds ANALYTIC sag derivatives, isolating the model.  The
library takes them with `xp.gradient`.  Re-running the arm with grid gradients
(group 5, 3 mm):

```
  bundle h    analytic     GRID-gradient
   187.50 um  3.794e-08       8.824e-06
    93.75 um  2.560e-08       2.217e-06
    46.88 um  2.553e-08       5.509e-07
    23.44 um  2.579e-08       1.357e-07
    11.72 um  2.592e-08       3.848e-08
```

Exactly 4x per halving -- second order, as `np.gradient` must be -- and at the
11.7 um rung the discretisation has fallen to the model's own truncation.  **So
the library realisation is grid-limited, not model-limited, and the grid is a
knob the caller already owns.**

---

## 3. THE WAVE SIDE

### 3.1 Energy

Warmed, R = 12.6 mm biconvex, N = 1536, dx = 4 um (the Nyquist-resolved arm
`BUILD_SCREEN_OBLIQUITY` S3.6 established), power as a fraction of the input:

```
  thin            0.999897294
  tangent_facet   0.999897294     (a screen: unitary by construction)
  REMAP  order 1  0.999895709
  REMAP  order 3  0.999897039     <- 2.6e-07 relative deficit
  REMAP  order 5  0.999897152
```

The common 1.03e-04 is the aperture.  The remap's own deficit is 2.6e-07.

### 3.2 The demodulation, priced

Same fixture, order 3:

```
  demodulation OFF   0.996601316      (3.4e-03 of the input lost to the spline)
  demodulation ON    0.999897039      (2.6e-07)
```

A 1.3e4 improvement, from an operation that provably cannot change the physics.

### 3.3 The Jacobian against a closed-form map

`W = c x` is the dilation `u = (1+c) x`, so `det A = (1+c)^2` exactly and the
answer is `E_out(u) = E_in(u/(1+c))/(1+c)`.  Relative to the peak, on a
`w0 = 24 dx` Gaussian:

```
  c        order 1     order 3     order 5     energy dev (order 3)
  -0.20   8.306e-04   1.771e-07   7.578e-11        5.014e-08
  +0.15   7.951e-04   1.652e-07   6.883e-11        5.034e-08
  +0.35   8.532e-04   1.842e-07   7.987e-11        5.032e-08
```

The residual depends on the ORDER; a wrong Jacobian exponent would not, and
would read `O(|c|)` ~ 0.2 at every order.  At `w0 = 12 dx` the order-3 column
reads 2.3e-06, i.e. the fourth-order rate a cubic spline must have.

---

## 4. CAUSTIC SAFETY

`det(I + dW/dx) > 0` is exactly "the output pixel has ONE source".  The guard
refuses on a non-positive or near-zero determinant AND on a non-convergent
pull-back, and it never degrades.

**The fold had to be ENGINEERED**, because a lens interior does not fold -- that
is the design contract that makes a ray map legal here at all.  With
`sag = A cos(k x)` the dominant walk term is
`-s p_out/pz2 ~ (A^2 k dz / 2 pz2) sin(2 k x)`, so `dW/dx` reaches -1 at a slope
amplitude `A k ~ sqrt(pz2/dz)` = 1.48 for N-SF57.  The departure is injected as
a `form_error` map, which is the hook this model's whole-grid sag pipeline
actually reads (`sag_callable` is the displaced-pointwise path's and is NOT read
here -- a wrong first attempt, recorded in S7).

Ladder, 40 um period, N-SF57, 200 um thick:

```
  amp (um)   slope A*k     min det    outcome
      0.40       0.063     0.99838    ran
      0.80       0.126     0.99352    ran
      1.60       0.251     0.97410    ran
      3.00       0.471     0.90912    ran
      6.00       0.942     0.63916    ran
      8.00       1.257     0.36329    REFUSED (pull-back did not converge)
      9.00       1.414     0.19774    REFUSED (pull-back did not converge)
     10.00       1.571     0.01439    REFUSED (pull-back did not converge)
     11.00       1.728    -0.18629    REFUSED (the map folds)
     12.00       1.885    -0.40377    REFUSED (the map folds)
     14.00       2.199    -0.88706    REFUSED (the map folds)
```

The measured fold onset (`A k` between 1.571 and 1.728) matches the derived
`sqrt(pz2/dz)` = 1.48 to ~15 %, so the mechanism is the one the derivation
names.  **The two bars catch different failures and both refuse**: `det <= 0` is
the physical statement, the pull-back's convergence is the numerical one, and
the second fires first because the fixed point's contraction rate approaches 1
as the determinant approaches 0.

**The un-guarded arm is wrong.**  With `_TF_REMAP_MIN_DET` and the pull-back
tolerance both removed, the `amp = 12 um` call returns a **NON-FINITE** field
(`1/sqrt(det)` at `det < 0`).  Independently, `x + W(x) = u` has **4 roots** at a
sampled `u` there, so the field at that exit pixel is a sum over four branches
and no pull-back of any interpolation order could be right.  Both are exact
structural claims.

---

## 5. WHAT WAS REFUTED ALONG THE WAY

| # | candidate | outcome |
|---|---|---|
| 1 | demodulate with `exp(+i k0 Phi)` (the sign the accumulator's `p -= grad OPD` step "looks like") | **WRONG SIGN.**  The library's field is `A exp(+i k0 S)`, measured by propagating a tilted plane wave through the library's own ASM (+200.3 um against +200.0 um geometric).  The wrong sign doubles the fringe rate: power fell to 0.944 of the input against 0.9999. |
| 2 | score the acceptance end to end against `apply_real_lens_traced` | **REFUTED AS AN INSTRUMENT.**  It is oracle-limited, not model-limited, and refining the grid PROVES it: on the biconvex at 50 mrad over a fixed 3.07 mm window the residual reads 0.0234 / 0.00013 / 0.00070 / 0.00259 at dx = 8 / 4 / 2 / 1 um -- it RISES, tracking the traced tracer's own Newton non-convergence fraction. A model-limited residual would fall. The exact-ray BUNDLE arm is the acceptance instrument, exactly as it was for route 3. |
| 3 | carry the accumulator as `p - grad OPD` (route 3's rule) and remap that | **NOT TAKEN.**  (R3) says the composite kick IS `p_out`, so taking `p_out` in closed form is both cheaper and continuum-exact; the grid-differenced route is the same quantity up to `A^T` and the grid's own O(h^2). Pinned as an equality instead of used. |
| 4 | push (R4) to a converged per-pixel Newton | **MEASURED AND NOT TAKEN.**  5.67e-12 against the shipped 2.56e-08 on group 5 -- 4500x better, for an analytic sag off the grid and the sag-source restriction route 3's S1.6 refused.  At 39000x under the bar it buys nothing. |
| 5 | inject the folding departure as `sag_callable` | **WRONG HOOK.**  The whole-grid sag pipeline reads `sag_callable` only on the `displaced`-pointwise path; for this model the departure must arrive as `form_error`. The first fold ladder ran to exhaustion with `min det` never moving off 1.0, which is what surfaced it. |
| 6 | apply the remap before the vignetting masks | **NOT TAKEN.**  The aperture is a property of the SURFACE and the walk is what happens between the vertex plane and back, so the masks stay at the pixel's own incoming coordinate exactly as for every other model.  Stated rather than measured -- the two differ only in a sub-100-um annulus at the clear-aperture edge. |
| 7 | is the 2.56e-08 a bundle discretisation artefact? | **NO.**  Converged over n = 33..513 to three significant figures, and the (R4) Newton arm lands 4500x below it, which a discretisation floor could not. |

---

## 6. COST

Warmed `tracemalloc` peak in float64 grids (`8*N*N` bytes) and wall clock,
biconvex singlet R = +19.6 / -27.4 mm N-SSK2, same fixture and same baseline as
`BUILD_TANGENT_FACET` S5 so the two are directly comparable.

**Two protocol notes, because they are what makes the numbers mean anything.**
Every arm is warmed AT ITS OWN N before anything is measured -- the first
`apply_real_lens` of a process also pays FFT-plan and lazy-import allocations
(the mistake `BUILD_OBL_BANDED_HALO` S5.5 records).  And memory and wall clock
are measured in SEPARATE passes, because `tracemalloc`'s per-allocation hook
inflates this model's wall clock by roughly 20x: the traced N = 4096 REMAP pass
took ~25 minutes against 67 s untraced.  Quoting a traced wall clock would have
made the model look ~20x worse than it is.

```
  N      arm              grids    extra    wall_s
  2048   thin             12.13    +0.00     0.318
  2048   thin+carrier     23.25   +11.13     1.319
  2048   tangent_facet    22.13   +10.00     1.860
  2048   tf+carrier       24.13   +12.00     2.504
  2048   REMAP            28.13   +16.00    15.124
  2048   REMAP+carrier    30.13   +18.00    16.732

  4096   thin              4.38    +0.00     1.256
  4096   thin+carrier      8.89    +4.51     5.460
  4096   tangent_facet    22.13   +17.74     7.158
  4096   tf+carrier       24.13   +19.74    10.117
  4096   REMAP            28.13   +23.74    67.352
  4096   REMAP+carrier    30.13   +25.74    74.678

  8192   thin              4.38    +0.00        --
  8192   tangent_facet    22.13   +17.75        --
  8192   REMAP           (23.74) (+23.74)       --      derived, see below
  8192   REMAP+carrier   (25.74) (+25.74)       --      derived, see below
```

**The memory surcharge is +6.00 grids over `'tangent_facet'`**, at the same N
and the same carrier state, at BOTH N = 2048 and N = 4096 -- a constant, which
is what a fixed set of extra full-grid temporaries (the two walk components, the
determinant, the two pull-back coordinate grids, the demodulated copy) has to
be.  The N = 2048 rows are smaller only because the thin BASELINE they are
differenced against is itself whole-grid below the N >= 4096 auto-band
threshold, exactly as `BUILD_TANGENT_FACET` S5 documents for route 3.

**The instrument reproduces the shipped anchor.**  `'tangent_facet'` reads
+17.74 at N = 4096 and +17.75 at N = 8192 against that build's published +17.8
at both -- so the protocol is validated on a number this build did not produce,
before it is trusted on numbers it did.

**WHAT IS MEASURED AND WHAT IS NOT, at N = 8192.**  The `thin` baseline (4.38)
and `'tangent_facet'` (22.13, i.e. **+17.75**) ARE measured, and that is the
load-bearing pair: it confirms the term is FLAT from 4096 to 8192, which is the
premise the derivation below rests on.  The two REMAP cells are **derived, not
measured**, and are written in parentheses everywhere they appear (here and in
the runner preflight) so they cannot be mistaken for readings.  The derivation
is two measured facts: `'tangent_facet'` is flat across 4096 -> 8192 (+17.74 ->
+17.75), and the REMAP surcharge over `'tangent_facet'` is a constant **+6.00**
at BOTH measured rungs (2048 and 4096) -- which is what a fixed set of extra
full-grid temporaries must be.  Why not simply measured: the traced pass at that
size runs ~100 minutes per arm for the tracemalloc reason above, and the box
became shared partway through this build (another session started eight
concurrent jobs on it), which turned that into an unbounded wait.

**Wall clock at N = 8192 is NOT reported at all.**  Not "approximately", not
"under load" -- the box was shared from partway through this build, and a
contended wall clock presented next to two clean rungs would invite exactly the
comparison it cannot support.  What the two CLEAN rungs establish stands on its
own: the model is **~9x `'tangent_facet'`** (15.1 vs 1.9 s at N = 2048; 67.4 vs
7.2 s at N = 4096) and scales linearly in pixel count (4.5x per 4x the pixels,
against `'tangent_facet'`'s 3.8x), so nothing is degrading with size -- the
constant is simply large.  The cause is structural and worth naming: the
pull-back is a fixed-point iteration whose every step is a
`scipy.ndimage.map_coordinates` call, and `map_coordinates` is serial, so this
model gets none of the threading the FFT legs get.  Lowering `remap_order` to 1
buys ~1.7x (9.2 s against 15.5 s at N = 1536) at the interpolation accuracy S3
prices.  A threaded or blocked pull-back is the obvious follow-on and is not
attempted here.

`Reverse_Symmetric_ASM/tx_design_study_sim.py` (outside the repo, not git) gains
the `'tangent_facet_remap'` selector and a version-gated
`_preflight_memory_check` term, ANCHOR 2026-08-16b.  The gate follows the
tangent-facet term's convention and for the same reason: an unparseable version
is treated as OLD, and for this term "old" means the feature is UNREACHABLE, so
the term is zero.

---

## 7. WHAT IS NOT CLAIMED

* **No GPU run, no `surface_frame` run, no non-ASM propagator run.**  All three
  RAISE.  This model is additionally scipy-bound (the pull-back is
  `scipy.ndimage.map_coordinates`), so the GPU refusal is structural here and
  not only unmeasured.
* **Whole-grid only**, for route 3's reason and one more: the field ITSELF is
  resampled, so an exact band would need a halo on the sag, on the accumulator
  AND on the walk.  `sag_chunk_rows` is pinned INERT with `np.array_equal` at
  `cr` in {0, 1, 7, 64, 4096}.
* **The demodulating eikonal is fitted to the MODEL's accumulated momentum.**
  An input field carrying a strong phase the model does not know about (a
  converging beam passed with no `carrier=`) is not demodulated, and the
  resampling is then interpolation-limited in the way S3.2's OFF column shows.
  The supported shape is the one the model is built for: `carrier=` seeds the
  accumulator and the lens screens build the rest.
* **Measured only on rotationally symmetric surfaces**, plus the engineered
  `form_error` corrugation used for the fold guard.  The model is structurally
  correct for decentred / tilted / freeform faces -- it reads `sag` after all of
  those are folded in -- but no oracle run was made on one.
* **The `screen_obliquity` guard is silent under this model**, as under route 3
  and for the same reason: its estimator measures the size of a correction this
  model does not make.
* **`prepare_real_lens` does not support the model** and
  `apply_real_lens_traced`'s delegate branch does not forward it -- both
  inherited from route 3, both for route 3's reason (a prepared screen is
  input-independent; this one reads the field's own momentum).
* **The end-to-end residual against `apply_real_lens_traced` is NOT the model's
  accuracy.**  See S5 row 2.  On the arms measured, the remap is at or below
  route 3 everywhere and 1.5x better on the biconvex at 50 mrad, but all of
  those numbers sit on the oracle's floor and none of them should be quoted as
  this model's error.

---

## 8. FILES

| file | change |
|---|---|
| `lumenairy/elements/_lens_real.py` | the REMAP derivation block, `_tangent_facet_remap_screen` / `_tangent_facet_remap_apply` / `_tf_remap_quadratic_eikonal` / `_tf_remap_phi` / `_TF_REMAP_MIN_DET` / `_TF_REMAP_MAX_ITERS` / `_TF_REMAP_PULLBACK_TOL_PX` / `_VALID_REMAP_ORDERS`; `_VALID_SURFACE_MODELS` + `_TANGENT_FACET_MODELS`; the two validators; the `remap_order` kwarg; `_tf_remap` and the screen / accumulator / remap blocks in the surface loop; the `surface_model` and `remap_order` docstrings |
| `tests/unit/test_tangent_facet_remap.py` | NEW |
| `validation/repro_traced_carrier_121/tangent_facet_derive.py` | `remap_screen` / `trace_remap` / `remap_error` and the `remap` / `remap_ladder` / `remap_facet` modes |
| `validation/repro_traced_carrier_121/_tangent_facet_remap*.json` | results of record |
| `CHANGELOG.md` | `[Unreleased]` |
| `docs/audits/BUILD_TF_REMAP_2026_08_16.md` | this note |

Reproducing the study:

```
cd validation/repro_traced_carrier_121
python tangent_facet_derive.py remap_facet   # S0.2, the closed form, no oracle
python tangent_facet_derive.py remap         # S2.1, the acceptance table
python tangent_facet_derive.py remap_ladder  # S1.4, the term ladder
```

---

## 9. SUITES

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X
python 3.14.6   numpy 2.4.4      lumenairy 5.36.0 (worktree C:/tmp/lum_rm,
scipy 1.17.1                     branch feat/tangent-facet-remap off
numexpr 2.14.1                   origin/main 97d431f = v5.36.0)
```

| gate | result |
|---|---|
| `test_tangent_facet_remap.py` (NEW) | **36 passed** |
| `test_tangent_facet_remap.py` + `test_tangent_facet.py` (route 3, UNCHANGED) | **104 passed** in 116.4 s |
| the six byte-identity-critical files (`test_screen_obliquity` + `test_obl_banded_halo` + `test_slant_chunk_byte_identical` + `test_lens_chunked_sag` + `test_tangent_facet` + `test_niche_audit_e_prepared_and_enums`) + the new file | **343 passed** in 159.2 s |
| `test_audit_lens.py` (the docstring/signature audit file, because this build rewrote `apply_real_lens`'s docstring) | **52 passed** in 10.1 s |
| TWO-TREE byte identity of both OFF paths vs a detached worktree at the `v5.36.0` tag -- 72 arms (3 prescriptions x 2 tilts x 12 option combinations covering `'thin'`, banded, slant, fresnel, bandlimit, carrier, `'displaced'` and `'tangent_facet'`) | **72/72 `np.array_equal`** |
| `ruff check lumenairy/ tests/unit/test_tangent_facet_remap.py validation/.../tangent_facet_derive.py` | **All checks passed** |
| `xfail` / `skip` added | **ZERO** |
| pre-existing assertions relaxed or retargeted | **ZERO** |
| `tests/unit -k "lens or obliquity or facet or slant or chunk or displaced"` (the wide sweep route 3 ran at 1607 s) | **ABANDONED, not green and not claimed** -- see below |

**THE WIDE SWEEP WAS NOT COMPLETED, AND IS NOT BEING PASSED OFF AS GREEN.**  It
was started on the committed tree and reached **7 %** in three hours before it
was stopped: another session had started eight concurrent `p3_matrix_dump.py`
jobs on the same box, and the run's CPU share collapsed to ~10 % of one core.
That is an environment fact, not a signal about this branch, but a
three-hour-old 7 % is not evidence of anything and is recorded as such rather
than dropped or rounded up.

What stands in its place is narrower but is the part that actually bears on this
build: the six byte-identity-critical files are the set `BUILD_TANGENT_FACET`
S7 itself nominated as the gate for this area, and they are green together with
the new file (343).  Two specific risks the wide sweep would have covered were
checked directly instead -- `test_audit_lens.py` (52 passed), because this build
rewrote `apply_real_lens`'s docstring and that file carries the
docstring/signature audits; and a grep establishing that NO test in the tree
parametrises over `_VALID_SURFACE_MODELS`, so adding a fourth value cannot have
silently widened any existing suite.  **Re-running the wide sweep on an idle box
is the one open verification item on this branch.**

**Path pinning.**  Every run in this note was made with `PYTHONPATH` pinned to
the worktree and `assert lumenairy.__file__.startswith('C:/tmp/lum_rm')`
checked in-process before anything else, and the two-tree byte comparison ran
the SAME script under both roots with that assert keyed to each root -- so a
silently-imported installed wheel cannot have produced any of these numbers.

**Two measurement mistakes this build made and caught, recorded because both
are the shape `docs/TESTING_STANDARDS.md` warns about.**  (1) Two constants
were written into docstrings from expectation before being measured
(`|o5 - o3|` and the linear-walk Jacobian residual); re-measuring moved them by
2.3x and 5 orders respectively, and both were corrected -- "right conclusion,
wrong numbers" reads as authoritative and passes.  (2) A first benchmark ran
two processes concurrently and a third overlapped the timing pass; the memory
column is contention-insensitive but the wall clock is not, so the timing
column was re-measured alone.

**Branch-point note.**  `origin/main` advanced to `4f75842` (v5.36.1) while this
build was in progress -- a PMM2D eig-recycle refutation note, a deprecation
horizon slip and the version bump.  None of it touches `_lens_real.py`.  This
branch stays cut from `97d431f` = v5.36.0 as briefed, and the byte-identity
comparison in S0.7 was run against BOTH trees with the same result.

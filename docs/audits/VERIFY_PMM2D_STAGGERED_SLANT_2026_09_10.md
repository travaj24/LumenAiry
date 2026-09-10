# VERIFY -- the native constant-shear SLANT on the PURE staggered 2-D PMM

**Date:** 2026-09-10 - **Subject:** `feat/pmm2d-staggered-slant`
(`96374af..6546f3a`), merged into `wave2/pmm2d` at **`8b9af801`**.
**Verification worktree:** `C:/tmp/lum_vslant`, branch `verify/slant`, HEAD
`8b9af801`.
**"Without" arm:** the main clone at `2efc7a2` (= everything on `wave2/pmm2d`
EXCEPT the slant), read-only.
**Under verification:**
`docs/audits/BUILD_PMM2D_STAGGERED_SLANT_2026_09_10.md` (the build) and
`docs/audits/EXPERIMENT_PMM2D_STAGGERED_SLANT_2026_09_10.md` (the spec).
**Probes:** `validation/probe_verify_slant/` (own README).
**Binding law:** `docs/TESTING_STANDARDS.md`.

This is a RE-MEASUREMENT, not a reading.  Every fixture, oracle and reference
construction below is this verification's own; where the build states a
cross-engine fact, the fact is re-asked in a form that does not depend on any
other engine's conventions.

---

## VERDICT UP FRONT

**The feature is CONFIRMED.  Nothing in the pure staggered slant is refuted.**
Two things came out of the verification that the build did not have:

1. **A DEFECT, on the OTHER engine.** `PMM2DStackHybrid`'s transmitted
   amplitudes on a SLANTED PATTERNED layer are **FRAME-referenced**: it never
   applies the frame-anchor phase the pure engine applies.  Measured against
   the hybrid's OWN fine staircase of the same solid, PER ORDER, its
   transmitted amplitudes are wrong by **5.914e-01 / 5.610e-01** (oblique 25 /
   conical 25-40); multiplying by the per-order anchor `P_m` puts them at
   **3.122e-02 / 1.636e-02**, and **the best possible SINGLE GLOBAL phase
   reaches only 4.631e-01 / 4.447e-01** -- so the correction is genuinely
   per-order and there is nothing else it could be.  `R`, `T` and the
   REFLECTION Jones are unaffected (they sit inside the staircase's own
   convergence step).  This is pre-existing (the 2026-08-16 hybrid slant
   metric), NOT introduced by this build, and it is undocumented -- while the
   pure engine documents its amplitudes as lab-referenced.  Reproducer and
   scope in S3.4.
2. **The parity accelerator is measured SAFE on slanted cells, more broadly
   than the build measured it** -- including the END-TO-END arm the build did
   not run (forced reduction vs dense: `dR/dT/dJones <= 3.4e-14` on 9 rows
   while the slant is a `1.3e-02 .. 8.2e-02` effect).  The verdict on whether
   to enable it, and the two-sided gate a build would need, is S5.

| # | claim | verdict | the number that decides it |
|---|---|---|---|
| 1 | `slant = 0` / absent is BIT-IDENTICAL to the pre-slant library | **CONFIRMED** | 18 fixtures x 7 zero spellings x 2 checkouts, **0 differing bytes** |
| 1b | the shipped assembly is BIT-IDENTICAL to the PROTOTYPE's | **CONFIRMED, widened** | 48 rows (4 cells x 4 slants incl. `(1.30, 0.60)` x 3 mounts), **0 differing bytes**, `max\|dA\| = 0.000e+00`; the wrong sign relation reads `1.814e-01` |
| 2 | the SIGN: `slant = +tan(slant_angle_1D) = t_hybrid`, and it draws a `+x` walk | **CONFIRMED, chain closed to geometry** | 1-D oracle per order, TE: `+tan` `1.1e-06 .. 4.3e-06` vs `-tan` `6.2e-03 .. 4.2e-01`; a 1-D explicit-centre staircase converges `5.7e-02 -> 6.0e-03` toward `+shear` and does not converge to `-shear`; the 2-D index/order directions match the 1-D at `2.2e-05` against `1.3e-02` for the mirrored profile |
| 2b | "energy carries zero information about the sign" | **CONFIRMED** | both sign arms close to `1.000000055` / `1.000000187` / `1.000000385` -- the SAME 9-digit value -- while sitting `7.3e-02 .. 4.2e-01` apart per order |
| 3 | the FRAME-ANCHOR PHASE, shipped sign right, two-sided | **CONFIRMED** | uniform null: shipped `9.9e-08 .. 2.6e-05`, none `1.4e-01 .. 7.4e-01`, conjugate `2.9e-01 .. 1.3e+00` |
| 3b | the anchor on a PATTERNED layer (no 1-D oracle) | **CONFIRMED** | vs a refinable lab-referenced staircase, per order, at a HALF-period walk (so the factor is not a global phase): pure shipped `7.8e-02 / 4.9e-02` against `6.0e-01 / 5.9e-01` with it divided out; the best possible SINGLE GLOBAL phase reaches only `4.6e-01 / 4.4e-01` |
| 4a | a uniform layer at any slant is a no-op, SPECTRALLY | **CONFIRMED, extended to 60 deg** | worst of 90 rows `1.47e-04` at M=5 **slant 60 deg**; ladder `6.9e-03 -> 7.0e-11` over M 4..8 |
| 4b | sheared-frame dispersion == exact quartic roots; 3 ablations | **CONFIRMED** | physical `1.4e-04` (over harmonics `\|m\|,\|n\| <= 2`); six-blocks-off `3.78e-01 .. 6.82e-01`, congruence-off `3.78e-01 .. 6.25e-01`, wrong-gauge `4.12e-01 .. 6.36e-01` |
| 4c | census `2q^2/2q^2`, growth 1.0, `sec`-bounded radius | **CONFIRMED, extended to a METAL** | 21 rows, split EXACTLY 200/200 **before** any rebalance; `min Re(lam_f) >= -1.50e-14` lossless, `+2.95e-03 .. +7.50e-02` lossy/metal; growth `1.0000e+00` at 1 and 3 wavelengths; radius ratio `1.000 -> 1.509` |
| 5 | the parity refusal: structure does NOT catch a shear | **CONFIRMED, and strengthened** | 28 rows: `resid_A 4.0e-16 .. 3.9e-15` vs a `1e-10` bar; forced pencil residual `4.6e-16 .. 1.7e-12` vs the shipped VERTICAL reduction's own `1.5e-13` on the same grid; end-to-end `<= 3.4e-14` |
| 6 | every refusal fires; the accepted shapes are right | **CONFIRMED** | 12/12 refusals, right type, message names the construction; layer split `<= 1.20e-15` incl. the transmission Jones; sneak attempts all land loudly |
| 7 | the WOOD decision: LAB eps, not the covariant diagonal | **CONFIRMED** | slanted layer `min\|q\|` = **1.03e-03** at the lab cut-off vs **1.2000e+00** at `eps(1+t^2)` -- and 1.2000 is the exact analytic value, so the covariant listing would nudge on a non-event |
| 8 | test-file durability | **CONFIRMED with 22 sub-decade bars flagged** | every doc'd reading reproduces to 3-4 digits on BOTH builds; cross-build spread `<= 5.4e-02` and usually `< 1e-8`; the thinnest bar has a **9%** margin |

---

## 0. Arms

| | WIN | WSL | BASE (the "without" arm) |
|---|---|---|---|
| tree | `C:/tmp/lum_vslant` @ `8b9af801` | same, via `/mnt/c` | `D:/Metacept/.../Lumenairy` @ `2efc7a2` |
| interpreter | CPython 3.14.6 | CPython 3.12.3 | CPython 3.14.6 |
| numpy / scipy | 2.4.4 / 1.17.1 | 2.4.6 / 1.17.1 | 2.4.4 / 1.17.1 |
| threads | `OMP` = `OPENBLAS` = `MKL` = 1 | `OMP` = `OPENBLAS` = 1 | as WIN |

Every probe calls `_lib.arm()`, which decides `tip` / `base` from
`lumenairy.__file__` itself -- never from a flag -- and stamps the resolved
path, interpreter, numpy, scipy and thread caps into its JSON.

**Cross-build spread, measured (S8):** of the 42 scalar quantities re-measured
on both builds, 31 agree to `< 1e-8` relative, 10 more to `< 6e-2` (all of them
machine-precision quantities where `1.5e-15` vs `1.8e-15` is a 16% "spread"),
and the only quantity that genuinely moves is a wall clock.

---

## 1. THE WITHOUT ARM -- bit-identity

`v1_without_identity.py`, `v1_compare.py`.

**18 fixtures of this verification's own:** `pmm_jones_2d_staggered` on a
scalar, an in-plane tensor and an out-of-plane tensor cell at normal / oblique
25 / conical 25-40; a CENTRO-symmetric out-of-plane cell at normal incidence
under `symmetry='auto'` AND `symmetry=False` (the parity-reduction path the
build touched); a `(3,3)` grid at conical; `pmm_efficiency_2d_staggered` TE and
TM; and four `PMM2DStackPure` stacks -- two-layer, three-layer with an
out-of-plane pattern, and a MAGNETIC (`mu_cell`) stack.

**7 spellings of "no slant":** `None`, `0.0`, `(0, 0)`, `[0, 0]`,
`np.zeros(2)`, `0` (int), `np.float64(0.0)`.

| comparison | differing sha256 |
|---|---|
| tip bare vs the BASE checkout at `2efc7a2` (18 rows) | **0** |
| tip bare vs each of the 7 zero spellings (126 rows) | **0** |

The `symmetry='auto'` fixture is not vacuous: on the vertical centro cell
`auto` and `False` hash DIFFERENTLY (`95c6dd61...` vs `0f8c2df7...`), i.e. the
reduction really does engage there.

**CONFIRMED.**  On the stacks the keyword was passed on EVERY layer, including
the magnetic one, which is the path where `_add_magnetic_layer` hard-codes
`slant=(0.0, 0.0)`.

### 1.1 The TRANSPLANT, re-measured

`v10_transplant.py`.  The build's B1b claims the shipped assembly is
BIT-IDENTICAL to the prototype's -- which is the check that the six blocks used
the distributionally exact `-(dbt)^H` and not the element-wise
`b.mixed(b.Btilde, b.B)` that drops the jump deltas.  Re-measured on this
verification's own fixtures: 4 cells (scalar, in-plane tensor, out-of-plane
tensor, NON-RECIPROCAL) x 4 slants (`(0.30, 0)`, `(0, 0.45)`, `(0.5, -0.25)`
and a LARGE `(1.30, 0.60)`) x 3 mounts (normal, oblique, conical) = **48 rows**:

| | reading |
|---|---|
| rows with a differing `A` or `B` byte | **0 / 48** |
| `max abs(A_shipped - A_prototype)` over all rows | **0.000e+00** |
| the WRONG sign relation (`slant = +s` into the prototype) | `dA = 1.814e-01` |

So the public-to-internal relation `slant_public = -t_prototype` is measured
here too, and the transplant is exact.  **CONFIRMED.**

---

## 2. THE SIGN, derived and then measured

### 2.1 What the code says (read, then checked against the measurement)

* `Granet2DTransverseE.__init__` calls
  `_slant_rot_gauge(cell33, -slant[0], -slant[1])`, and `_slant_rot_gauge`
  multiplies by `_OOP_ROT_SIGN = -1`, so the assembly's internal shear is
  `t_basis = +slant` **in the basis frame**;
* the frame-anchor phase is written `_shx = -sum(slant_x d)` and
  `exp(-i k0 (kx _shx))` = `exp(+i k0 (alpha . slant) d)`, i.e. `t = -slant` in
  the FAR-FIELD frame.

Those two are consistent with each other only if the basis frame is the
180-degree rotation of the `eps_cell` / far-field frame -- which is exactly
what `_OOP_ROT_SIGN`'s comment claims, and which no test asserts directly.
Take the internal `t = -slant` at face value in the CALLER's frame and you
conclude that a positive `slant` walks the pattern toward `-x`, the opposite of
what the docstring says.  **The source alone therefore does not settle what the
caller sees**; it settles it only once `_OOP_ROT_SIGN`'s stated meaning is
granted.  So the docstring's geometric statement has to be MEASURED, and that
is what S2.2-S2.4 do -- with the answer that the docstring is right and the
naive reading is the one that is wrong.

### 2.2 What the 1-D `slant_angle` DRAWS -- an explicit-centre staircase

`v2c_chain_sign.py`, LINK 1.  The 1-D engine is the anchor because it is the
one place where a fine staircase of the SAME solid can be built with the ridge
centre placed EXPLICITLY: `PMMStack.add_layer(segments=..., slant_angle=...)`
against a `K`-slice stack of vertical `segments=` layers whose ridge centre is
put at `0.5 + shear (zeta - 0.5)` in period fractions (`_ridge_slice_segments`'
documented convention: the ridge is `[centre - duty/2, centre + duty/2)` laid
out from `u = 0`, so a centre that GROWS with depth is a walk toward `+x`).

| mount / shear | staircase walking `+x`, K = 2 / 4 / 8 / 16, vs `shear = +s` | vs `shear = -s` |
|---|---|---|
| normal, 0.25 | 5.746e-02  2.185e-02  1.172e-02  **5.987e-03** | 3.382e-01  3.526e-01  3.558e-01  **3.565e-01** |
| normal, 0.40 | 5.965e-02  1.993e-02  1.137e-02  **6.116e-03** | 5.766e-01  5.985e-01  6.034e-01  **6.045e-01** |
| oblique 17, 0.25 | 5.867e-02  1.383e-02  7.818e-03  **4.292e-03** | 8.149e-02  8.290e-02  7.820e-02  **7.467e-02** |
| oblique 17, 0.40 | 5.179e-02  1.042e-02  3.652e-03  **2.275e-03** | 1.529e-01  1.782e-01  1.857e-01  **1.875e-01** |

A clean `1/K` convergence toward `+shear` (a factor 9.6 over 8x the slices) and
no convergence at all toward `-shear`.  And `add_sheared_grating(shear=+s)` is
the same solve as `add_layer(slant_angle=+arctan(s P / d))` to
**1.10e-06 .. 5.86e-06**, against **7.04e-02 .. 6.05e-01** for `-arctan`.

**So: `slant_angle > 0` in the 1-D engine draws a ridge that walks toward `+x`
as depth increases.**  That is a measurement of a DRAWING, not a convention.

### 2.3 The 2-D engine's cell-index and order directions match the 1-D's

`v2c_chain_sign.py`, LINK 2.  A sign error in the slant plus a mirrored cell
index (or a mirrored order list) would cancel and hide.  A THREE-LEVEL
x-ASYMMETRIC y-uniform VERTICAL grating (`[4, 4, 2, 1, 1, 1]` over six equal
segments -- its own mirror image is NOT a translate of it) at oblique
incidence, per order, 2-D against 1-D:

| mount | same profile | MIRRORED profile |
|---|---|---|
| oblique +25 | **2.168e-05** | 1.273e-02 |
| oblique -25 | **3.279e-05** | 1.273e-02 |

A factor 390-590.  **The 2-D cell index and the 2-D order index run the same
way as the 1-D engine's**, so the slanted comparison below cannot be a double
flip.

### 2.4 The 2-D public `slant` against the 1-D oracle, per order, both signs

`v2b_oracle_sign.py`.  y-uniform binary stripe (`n = 2/1`, duty 0.5,
`px = py = 0.75 lam`, `depth = 0.30 lam`, `n_sub = 1.5`), `M = 7`, orders
`-2..+2`, against `pmm_efficiency_1d_slanted` at degree 24 (with the degree-20
run as the oracle's own drift bound).

| slant | mount | TE, `+tan` | TE, `-tan` | TM, `+tan` | TM, `-tan` | oracle drift (TE) |
|---|---|---|---|---|---|---|
| 0 (control) | normal | 1.070e-07 | -- | 1.137e-03 | -- | -- |
| 0 (control) | oblique 25 | 1.787e-06 | -- | 1.249e-04 | -- | -- |
| 10 | normal | **1.791e-06** | 7.290e-02 | 9.813e-04 | 8.371e-02 | 1.53e-10 |
| 10 | oblique 25 | **1.436e-06** | 6.235e-03 | 1.973e-04 | 2.992e-02 | 4.46e-10 |
| 20 | normal | **3.370e-06** | 1.711e-01 | 7.293e-04 | 1.594e-01 | 4.66e-11 |
| 20 | oblique 25 | **1.115e-06** | 8.142e-03 | 2.376e-04 | 6.310e-02 | 1.04e-09 |
| 35 | normal | **4.304e-06** | 4.153e-01 | 1.053e-03 | 2.239e-01 | 2.16e-08 |
| 35 | oblique 25 | **1.440e-06** | 2.016e-02 | 2.228e-04 | 1.268e-01 | 2.75e-10 |

The `+tan` arm TRACKS the vertical control at every slant (TE `1.1e-06 ..
4.3e-06` against a control of `1.1e-07 / 1.8e-06`); the `-tan` arm is
**4.3e+03 .. 1.4e+05** times worse on TE.  The TM column is oracle-limited, as
the build says, and tracks its own vertical control.

**And the third engine.**  `v2c` LINK 3, a genuinely 2-D pillar, hybrid against
pure on an `n_orders` = 3/5/7/9 ladder:

| mount | hybrid(`+t`) vs pure(`+t`) | hybrid(`-t`) vs pure(`+t`) | the slant's own size |
|---|---|---|---|
| normal | 4.866e-02  3.475e-02  3.036e-02  **2.548e-02** | 7.565e-02  8.165e-02  7.653e-02  **7.736e-02** | 5.892e-01 |
| conical 20/35 | 5.853e-02  2.015e-02  1.192e-02  **1.041e-02** | 4.457e-01  3.968e-01  4.060e-01  **4.099e-01** | 4.647e-01 |

The correct sign walks toward the pure answer; the wrong one is flat.

### 2.5 The chain, closed

```
   public slant (pure 2-D)  ==  + tan(slant_angle)  [1-D]  ==  t  [hybrid]
        (S2.4, 1e-06 vs 7e-02)          (S2.4)            (S2.4 LINK 3)

   and slant_angle > 0 DRAWS a walk toward +x            (S2.2, an explicit
                                                          -centre staircase)

   and the 2-D cell index / order index run the SAME way (S2.3, 2e-05 vs 1e-02)
```

**So the docstring's geometric statement is CORRECT: `slant = (t_x, t_y)`
translates the cross-section by `+(t_x, t_y) * thickness` from the layer's TOP
face to its bottom, in the caller's own `eps_cell` index frame.  CONFIRMED --
and note that this could NOT have been established by reading the source (S2.1).**

### 2.6 The lossless trap, reproduced

`v2b_oracle_sign.py`, the closure column.  On the same stripe, at normal
incidence:

| slant | `max(sum R + sum T)`, `+tan` | `-tan` | the WRONG arm's per-order TE error |
|---|---|---|---|
| 10 deg | 1.000000055 | **1.000000055** | 7.3e-02 |
| 20 deg | 1.000000187 | **1.000000187** | 1.7e-01 |
| 35 deg | 1.000000385 | **1.000000385** | 4.2e-01 |

Identical to nine digits.  **Energy carries literally zero information about
the sign.  CONFIRMED.**  (At oblique 25 the two closures differ in the ninth
digit -- `1.000000014` vs `1.000000010`, a difference of `4e-09` -- which is
six decades below the per-order disagreement there (`6.2e-03`), so the
statement holds at oblique incidence too.)

### 2.7 What did NOT work, and why it is recorded

`v2a_geometry_sign.py` tried to settle S2.5 inside the PURE engine alone, with
a `np.roll` staircase.  It is **INCONCLUSIVE BY CONSTRUCTION** and the reason
is the same one the build's M4c records: a nodal SEM union grid admits only
slice counts that DIVIDE the walk in grid cells, so on a 6-cell grid with a
whole-period walk (`t = 1.0`) the finest rung is 6 slices and its own geometric
error (`7.3e-02`) is the same size as the `+t` / `-t` separation (`1.8e-01`).
The pillar rows lean the right way at every mount; the stripe rows do not
separate at all.  The 1-D engine (S2.2), where the ridge centre is placed
explicitly and `K = 16` is available, is where the question is answerable.

A second trap, recorded because it wasted a run: the first fixture was a single
block, which is its own mirror image UP TO A TRANSLATION, so the `+t` and `-t`
structures are mirror-related and their zeroth-order reflection Jones agree to
**5.5e-15** while their per-order `T` differs by **7.9e-02**.  Read per-order
quantities, and use an x-asymmetric profile.

---

## 3. THE FRAME-ANCHOR PHASE

### 3.1 Derived here, independently

A slanted region is solved in the frame `u = x - t_int w`, `v = y - t_int_y w`,
`w = z`, anchored at the layer TOP, so the layer occupies `0 <= w <= d`.  Every
modal field is a function of `(u, v)`; the region's Rayleigh expansion at the
BOTTOM interface `w = d` is `sum_m A_m exp(i alpha_m . (u, v))`, and in the lab
that plane is `z = d` with `(u, v) = (x - t_int d, y - t_int_y d)`.  Against the
substrate's OWN basis `exp(i alpha_m . (x, y))`:

```
    A_lab(m) = exp(-i alpha_m . t_int d) A_frame(m)
```

`alpha_m` is real for every order, propagating or evanescent, so the factor is
UNIMODULAR: no efficiency can move, and the superstrate side is where the frame
is anchored, so the reflection Jones needs nothing.  In a stack the offsets ADD.
With the shipped internal shear `t_int = -slant` (S2.1), in the PUBLIC vector:

```
    A_lab(m) = exp(+i k0 (alpha_m . slant) d) A_frame(m)
```

which is exactly what `stack2d_pure.solve` computes.  **The formula and its
sign are CONFIRMED by derivation and by S3.2-S3.4.**

### 3.2 The uniform null, three arms

`v3_frame_anchor_phase.py`, oracle A.  A uniform slab's transmission Jones is
slant-INDEPENDENT (a shear of a homogeneous medium is a coordinate change and
both arms are referenced to the same lab plane), so the truth is known exactly.
The "no correction" and "conjugate" arms are recovered from the shipped answer
by dividing `P0` out once and twice -- **no library edit**.

| tensor | mount | slant | NO correction | **shipped** | conjugate | `\|arg P0\|` |
|---|---|---|---|---|---|---|
| isotropic | oblique 25 | x10 | 1.933e-01 | **8.575e-07** | 3.839e-01 | 0.234 |
| isotropic | oblique 25 | x35 | 7.419e-01 | **1.657e-05** | 1.326e+00 | 0.930 |
| isotropic | oblique 25 | diag35 | 5.342e-01 | **7.897e-06** | 1.011e+00 | 0.657 |
| isotropic | conical 25/40 | x10 | 1.442e-01 | **9.856e-08** | 2.872e-01 | 0.179 |
| isotropic | conical 25/40 | x35 | 5.612e-01 | **1.912e-06** | 1.052e+00 | 0.712 |
| isotropic | conical 25/40 | diag35 | 7.191e-01 | **9.097e-07** | 1.287e+00 | 0.926 |
| oop uniaxial | oblique 25 | x35 | 7.381e-01 | **2.637e-05** | 1.320e+00 | 0.930 |
| oop uniaxial | conical 25/40 | diag35 | 7.151e-01 | **1.592e-06** | 1.280e+00 | 0.926 |

Twelve rows measured, eight shown.  The build's table B3 is **reproduced to
three significant digits in every row**.  The conjugate arm is `1.787x .. 1.991x`
worse than doing nothing, so the correction cannot be a fudge absorbing an
arbitrary residual.  **CONFIRMED.**

### 3.3 A PATTERNED layer, against a REFINABLE lab-referenced oracle

`v3b_anchor_patterned.py`.  A z-STAIRCASE has no frame at all, so its
transmitted amplitudes are lab-referenced by construction -- and the FOURIER
hybrid has no union-grid constraint, so a staircase of 60-pixel cells at 5 / 10
/ 20 / 30 slices is available there.  That is the oracle.  Fixture: the
x-asymmetric 2-D cell, a whole-period walk, `n_orders = 9`.

| mount | staircase ladder (K = 5 / 10 / 20 vs K = 30) | pure **shipped** | pure none | pure conj | pure vs hybrid corrected |
|---|---|---|---|---|---|
| oblique 25 | 6.72e-02 / 1.96e-02 / 8.32e-03 | **1.373e-01** | 6.371e-01 | 8.532e-01 | 1.335e-01 |
| conical 25/40 | 1.75e-01 / 8.75e-02 / 2.57e-02 | **6.342e-02** | 5.696e-01 | 2.931e-01 | 6.344e-02 |

The shipped arm wins in both mounts (4.6x and 9.0x over the next-best arm), and
the residual it leaves is IDENTICAL to the pure-vs-hybrid engine gap
(`1.335e-01` / `6.344e-02` against `1.373e-01` / `6.342e-02`), i.e. it is
discretization, not phase.  **CONFIRMED, BOUNDED by the pure solve's own
accuracy at these sizes (`M = 3` on a 6-cell grid).**

The same arm attempted with a staircase built INSIDE the pure engine
(`v3_frame_anchor_phase.py`, oracle B) is inconclusive for the S2.7 reason --
the union grid caps the ladder at 6 slices and its own step (`1.96e-01`) is the
size of the effect -- and is reported here as inconclusive rather than dropped.

### 3.4 [DEFECT] the HYBRID does not apply the anchor

`v3b_anchor_patterned.py` Q1, `v8_hybrid_anchor.py`.  Same fixture, same
staircase, but comparing the HYBRID's own slanted layer against the HYBRID's
own fine staircase -- one engine, one basis, nothing else can differ:

| mount | hybrid slanted, **as returned** | hybrid `x P0` | hybrid `x conj(P0)` | staircase's own ladder |
|---|---|---|---|---|
| oblique 25 | **5.912e-01** | **1.428e-02** | 8.254e-01 | 6.7e-02 / 2.0e-02 / 8.3e-03 |
| conical 25/40 | **6.143e-01** | **1.724e-02** | 2.607e-01 | 1.8e-01 / 8.8e-02 / 2.6e-02 |

and the scope, measured on the same rows:

| quantity | hybrid slanted vs its own staircase | the staircase's own step |
|---|---|---|
| `R` per order | 1.776e-03 | 3.60e-03 |
| `T` per order | 1.669e-02 | -- |
| REFLECTION Jones | 4.620e-03 | 4.95e-03 |
| **zeroth-order TRANSMISSION Jones** | **5.912e-01** | -- |

A slanted **UNIFORM** hybrid layer is unaffected, and the reason is in the
source: `PMM2DStackHybrid._geom_key` deliberately omits the slant from its two
UNIFORM branches, so a slanted film keys as -- and is solved as -- the vertical
film.  It never enters a frame, so it needs no anchor, and its transmission
Jones is **bit-identical** (`0.000e+00`) to the vertical one, while applying
`P0` to it would make it wrong by `1.933e-01 .. 7.419e-01`.  The pure engine on
the same uniform fixture reads `1.70e-06 .. 3.47e-05` as returned (it DOES
enter the frame there, and does pay the anchor).

**That asymmetry is also what the hybrid's fix has to respect**: the offset it
must undo accumulates over its slanted PATTERNED layers only, not over every
layer carrying a slant keyword -- unlike the pure engine, where a slanted
uniform layer is materialised on the union grid, genuinely sheared, and
therefore genuinely contributes.

**Reproducer** (`validation/probe_verify_slant/v8_hybrid_anchor.py`):

```python
st = PMM2DStackHybrid(1.2e-6, 1.2e-6, n_superstrate=1.0, n_substrate=1.5,
                      n_orders=9)
st.add_layer(1.2e-6, eps_cell=cell, slant=(1.0, 0.0))   # a PATTERNED layer
st.set_source(0.68e-6, theta=np.deg2rad(25.0), phi=0.0)
st.solve()
J = st.jones_transmission()      # FRAME-referenced; multiply by
                                 # exp(+i k0 (alpha_m . slant) d) to fix
```

**Assessment.** `R`, `T` and the reflection Jones are right, so no energy or
efficiency result is affected and no existing hybrid test can see it -- the
same silent shape the pure build's B3 gate was written to catch.  It is
pre-existing (`BUILD_PMM2D_SLANT_METRIC_2026_08_16`), it is NOT a regression of
this build, and it is a cross-engine inconsistency the moment a caller takes
the build's own advice that "a layer moves between the engines unchanged": the
pure engine's `jones_transmission` / `per_order_amplitudes('transmission')` are
LAB-referenced and the hybrid's are not.  **No library edit was made.**

*A trap this probe fell into first, and the fix.*  With a WHOLE-period walk
`exp(2 pi i m t d / px) = 1` for every order, so the anchor DEGENERATES into a
single global phase and the measurement cannot tell it from any other global
phase.  The first run duly reported "best single global phase" = `1.428e-02`,
equal to "x P_m" to every digit.  **At a HALF-period walk the increment is `pi`
per order and the two separate** (`v8_hybrid_anchor.py`, `n_orders = 9`, the
reference a 15-slice 60-pixel staircase):

| mount | hybrid raw | **hybrid `x P_m`** | hybrid `x conj(P_m)` | the BEST single global phase | staircase own step (dR / dJ) |
|---|---|---|---|---|---|
| oblique 25 | 5.914e-01 | **3.122e-02** | 4.841e-01 | **4.631e-01** | 1.35e-02 / 8.00e-02 |
| conical 25/40 | 5.610e-01 | **1.636e-02** | 6.955e-01 | **4.447e-01** | 1.34e-02 / 6.04e-02 |

**No single global phase can do what `P_m` does** -- the best one is 14.8x
(oblique) / 27.2x (conical) worse.  The correction is genuinely per-order, so
there is nothing else it could be: it IS the frame anchor.  On the same rows
the PURE engine reads **7.848e-02 / 4.915e-02** as shipped against
**5.962e-01 / 5.897e-01** with `P_m` divided out (7.6x / 12.0x), and the
efficiencies and reflection Jones of the hybrid slanted layer sit at
`dR 3.9e-03 / 1.8e-03`, `dJ(refl) 5.9e-03 / 5.3e-03` -- inside the staircase's
own step, i.e. unaffected.

### 3.5 A 2-D CHIRAL slanted cell, where no 1-D oracle exists

`v3_frame_anchor_phase.py`, oracle C.  A `(3,3)` cell with an in-plane chiral
(`exy = eyx = 0.35`) inclusion, slanted, against the hybrid's transmission
Jones on an `n_orders` = 5/7/9/11 ladder:

| mount | pure shipped | pure none | pure conj | hybrid's own step |
|---|---|---|---|---|
| oblique 25 | 8.738e-01 | **6.546e-03** | 8.789e-01 | 3.520e-03 |
| conical 25/40 | 7.959e-01 | **6.028e-03** | 8.031e-01 | 3.292e-03 |

**Read this row correctly.**  It is NOT evidence against the pure engine: it is
the S3.4 defect seen from the other side.  The hybrid is frame-referenced, so
the arm that matches it is the one with the pure engine's (correct) anchor
DIVIDED OUT.  Taken together with S3.2 (where the truth is known exactly and
the shipped arm wins by four decades) and S3.3 (where the oracle is
lab-referenced by construction and the shipped arm wins), the pure engine is
right and the hybrid is not.  **This row is the one place a wrong phase could
have hidden, and what it actually caught is a wrong phase -- on the other
engine.**

---

## 4. NULL TESTS, DISPERSION, CENSUS

### 4.1 The uniform null, extended to 60 degrees

`v4a_null.py`.  Five tensors (isotropic, IN-PLANE uniaxial, OUT-OF-PLANE
uniaxial, gyrotropic, LOSSY out-of-plane) x six slants (x10, x35, **x60**, y35,
diag35, **diag60**) x three mounts = **90 rows**, at `M = 5`, on `R`, `T`, the
reflection Jones AND the transmission Jones (which carries the anchor).

| quantity | reading |
|---|---|
| worst of the 90 rows | **1.471e-04** (out-of-plane uniaxial, **slant 60 deg**, oblique 25, on the TRANSMISSION Jones) |
| worst at NORMAL incidence | **2.729e-14** |
| order leak at NORMAL incidence | **3.713e-28** |
| worst leak (oblique) | 4.860e-06 |

**The M-ladder on that worst row** (the "is it discretization?" arm):

| M | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| dR | 3.137e-03 | 1.216e-05 | 1.452e-07 | 1.146e-09 | **6.706e-12** |
| dT | 3.524e-03 | 1.478e-05 | 1.437e-07 | 1.142e-09 | **6.724e-12** |
| dJones | 8.827e-04 | 3.158e-05 | 3.750e-07 | 2.951e-09 | **1.724e-11** |
| dJones (transmission) | 6.878e-03 | 1.471e-04 | 1.585e-06 | 1.214e-08 | **7.007e-11** |

Eight decades over four steps.  **CONFIRMED: the null residual is
discretization and it is spectral, at 60 degrees as at 10.**

**A scope note the build does not make.**  Its B2 bar (`< 1e-04` at `M = 5`) is
a statement about slants `<= 35 deg`: at 60 degrees the same measurement reads
`1.47e-04` and would cross it.  Nothing is wrong -- the ladder shows it is the
`M = 5` rung of a spectral sequence -- but "exact at any slant magnitude" costs
`M` at steep tilt, and the test file does not exercise a 60-degree slant end to
end anywhere.

### 4.2 The sheared-frame dispersion against the EXACT quartic roots

`v4b_dispersion.py`.  Own fixture (`px = py = 0.85`, `M = 8`, `(2,2)` grid,
normalized Bloch shift `(0.30, 0.21)`), own quartic: the coefficients of
`det(k k^T - (k.k) I + eps)` built by exact polynomial arithmetic on the 3x3
cofactor expansion.  The exact set is
`kz_root(eps_lab; alpha_m) + t_int . alpha_m` over the harmonics
`|m|, |n| <= 2` -- **a strictly harder comparison than the build's, which asks
only the FUNDAMENTAL's four roots** (that is why the numbers below are `1e-04`
where the build reads `1e-14`: at `|m| = 2` the `(2,2) x M = 8` basis simply
does not resolve `alpha_x = 2.65` as well).

**The four gauge arms** (`a+/a-` = harmonics `+/-alpha`; `s+/s-` = shift
`+/- t.alpha`):

| tensor | slant | mount | **a+s+** | a+s- | a-s+ | a-s- |
|---|---|---|---|---|---|---|
| uniaxial tilt35 | none | conical | **1.299e-04** | 1.299e-04 | 2.800e-01 | 2.800e-01 |
| uniaxial tilt35 | x20 | normal | **3.567e-05** | 3.219e-01 | 3.567e-05 | 3.219e-01 |
| uniaxial tilt35 | x20 | conical | **1.468e-04** | 3.772e-01 | 4.473e-01 | 3.215e-01 |
| uniaxial tilt35 | diag35 | conical | **1.741e-04** | 5.892e-01 | 5.685e-01 | 3.282e-01 |
| non-reciprocal | x20 | conical | **1.423e-04** | 3.979e-01 | 3.685e-01 | 9.604e-02 |
| non-reciprocal | diag35 | conical | **1.653e-04** | 6.281e-01 | 6.137e-01 | 1.106e-01 |
| isotropic | diag35 | conical | **1.677e-04** | 5.794e-01 | 5.794e-01 | 1.677e-04 |

Three decades between the physical arm and every wrong one.  Two structural
facts fall out and both are handled by construction rather than by tolerance:
an ISOTROPIC tensor is `alpha -> -alpha` symmetric, so its `a-s-` arm is the
same solution relabelled; and at NORMAL incidence with slant 0 all four arms
coincide.  **A finding the build's fundamental-only comparison could not make:
at NORMAL incidence with a NONZERO slant the SHIFT sign is still observable**
(`3.567e-05` vs `3.219e-01`), because the `m != 0` harmonics carry `alpha != 0`
even when the fundamental does not.

**The three ablations**, each driven through the SHIPPED `_assemble_oop` by
setting the attributes it reads and re-assembling (no function is patched):

| tensor | slant | full | SIX BLOCKS removed | CONGRUENCE removed | WRONG GAUGE |
|---|---|---|---|---|---|
| uniaxial | x20 | **1.468e-04** | 4.778e-01 | 4.641e-01 | 4.367e-01 |
| uniaxial | diag35 | **1.741e-04** | 6.823e-01 | 6.254e-01 | 4.750e-01 |
| non-reciprocal | x20 | **1.423e-04** | 4.509e-01 | 5.229e-01 | 4.121e-01 |
| non-reciprocal | diag35 | **1.653e-04** | 5.346e-01 | 5.077e-01 | 6.361e-01 |
| isotropic | x20 | **1.444e-04** | 3.778e-01 | 3.778e-01 | 5.279e-01 |
| isotropic | diag35 | **1.677e-04** | 5.343e-01 | 5.343e-01 | 5.347e-01 |

**Both halves of the formulation, and the rotation gauge's `t`-half, are
load-bearing, each by 3.4 decades.  CONFIRMED.**

**The sum-of-roots discriminator** (uniaxial, conical): the four eigenvalues
nearest the fundamental's four exact roots must sum to those roots' sum plus
`4 t . alpha`.  Generator / exact -- `none`: `-0.080946 / -0.080946`; `x20`:
`-0.517711 / -0.517711`; `diag35`: `-1.090994 / -1.090994`.  Identical to six
decimals while the slant moves the sum by 0.44 and 1.01.

**The M-ladder** (uniaxial, diag35, over the harmonics `|m|, |n| <= 2`):

| M | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| gap | 6.519e-01 | 2.048e-01 | 2.530e-02 | 2.732e-03 | **1.741e-04** |

Spectral, 3.6 decades over four steps -- the higher harmonics are what a
`(2,2) x M = 8` basis is still resolving, exactly as expected.

### 4.3 Census, growth and the spectral radius -- with a METAL

`v4c_census.py`.  The library's own split lives inside `_region_modes_oop`, and
`_select_forward_flux` rebalances to exactly `2 q^2` unconditionally, so reading
its output can never fail.  This probe therefore **classifies the modes itself**
from the whitened eigenvectors, by the physical rule (net Poynting `Sz` summed
over all harmonics, decay sign for the flux-null ones), and reports the count
BEFORE any rebalance.  `M = 6`, `(2,2)` grid, conical 20/35, `dim = 4 q^2 = 400`.

| cell | slant | split (want 200/200) | `min Re(lam_f)` | `max abs(q)` | growth @1 lam | @3 lam |
|---|---|---|---|---|---|---|
| out-of-plane uniaxial | vertical | **200/200** | -9.03e-15 | 9.558 | 1.0000e+00 | 1.0000e+00 |
| scalar `eps` 4 | x20 | **200/200** | -5.06e-15 | 9.726 | 1.0000e+00 | 1.0000e+00 |
| scalar | x60 | **200/200** | -9.38e-15 | 14.196 | 1.0000e+00 | 1.0000e+00 |
| out-of-plane | x60 | **200/200** | -2.33e-15 | 14.419 | 1.0000e+00 | 1.0000e+00 |
| high contrast `eps` 12 | diag45 | **200/200** | -1.50e-14 | 10.837 | 1.0000e+00 | 1.0000e+00 |
| LOSSY `4 + 0.6i` | x20 | **200/200** | **+7.50e-02** | 9.743 | 6.2429e-01 | 2.4331e-01 |
| LOSSY | x60 | **200/200** | **+3.11e-02** | 14.217 | 8.2225e-01 | 5.5593e-01 |
| **METAL `-20 + 1.5i`** | x20 | **200/200** | **+2.95e-03** | 10.746 | 9.8165e-01 | 9.4596e-01 |
| **METAL** | x60 | **200/200** | **+4.45e-03** | 14.304 | 9.7240e-01 | 9.1946e-01 |

21 rows, **every one exactly 200/200 before any rebalance**; `min Re(lam_f) >=
-1.50e-14` on every lossless row and strictly positive on every lossy row --
including a metal, which is the cell a flux-sign rule is most likely to
misclassify.  Radius ratio against the vertical control: `1.000` (x20) ->
`1.178` (x45) -> `1.206` (diag45) -> **`1.509`** (x60), i.e. `sec(60) = 2`
bounded, not the `~210` a from-scratch convection form produces in 1-D.
**CONFIRMED, and extended.**

---

## 5. THE PARITY QUESTION -- verdict

`v5_parity.py`.  Five centro-symmetric cell families (out-of-plane `(2,2)` and
`(3,3)`, scalar, LOSSY, HIGH CONTRAST) x six slants (vertical, x20, x35, **y35**,
diag35, **x60**) = **28 rows**, `M = 6`, normal incidence.

Forcing is done TWO ways, neither of them `_slant_is_zero` (the build records
that patching that name silently builds the VERTICAL pencil, because
`Granet2DTransverseE.__init__` reads it to decide whether to shear at all):

* `_Shim` -- a proxy over an ALREADY-BUILT slanted solver whose `slant` reads
  `None`.  The pencil it exposes is the genuine slanted one;
* for the END-TO-END arm, `stack2d_pure._region_modes_oop` is wrapped so it
  receives the shim.  The ASSEMBLY is untouched -- the wrapper runs after
  `__init__` has already sheared the cell.

### 5.1 The structure does NOT catch a shear -- reproduced and widened

| quantity | over the 28 rows |
|---|---|
| `max\|R A R + A\| / max\|A\|` | **3.998e-16 .. 3.881e-15** (bar `_STAG_BLOCK_TOL` = `1e-10`) |
| `max\|R B R - B\| / max\|B\|` | 1.739e-16 .. 1.956e-16 |
| `_stag_block_eig` runs when forced | **28 / 28** |
| the shipped gauge REFUSES the slanted rows | **23 / 23**; and ACCEPTS all 5 vertical controls |

**Five decades below the bar, at every slant to 60 degrees, on every cell
family including a lossy one.  CONFIRMED.**

The check still DISCRIMINATES: an OFF-CENTRE pillar (not its own parity image)
reads `resid_A = 5.768e-01`, nine decades above the bar, and the reduction
refuses -- slanted and vertical alike.

### 5.2 Why -- derived, because a coincidence and a symmetry warrant different
decisions

`R` is a 180-degree ROTATION about `z` and the pencil relation is `R A R = -A`
with `R B R = +B`, i.e. `(q, x)` pairs with `(-q, R x)`.  That is the algebraic
form of invariance under the FULL INVERSION `(x, y, z) -> (-x, -y, -z)`.  A
sheared solid `x = u + t z` maps under that inversion to `-x = u + t(-z)`, i.e.
`x = (-u) + t z` -- **the same shear, on the inverted cross-section**.  So a
centro-symmetric cross-section sheared by any `t` is inversion-symmetric,
exactly as the vertical one is.  The structure surviving the shear is a
SYMMETRY OF THE GEOMETRY, not a numerical accident.  (This is the build's
explanation, re-derived; it is correct.)

### 5.3 Forced, the reduction is right -- including END TO END

| quantity | forced (slanted) | the SHIPPED vertical reduction, same fixtures |
|---|---|---|
| pencil residual `max \|\|A x - q B x\|\| / (max\|A\| \|\|x\|\|)` | **4.595e-16 .. 1.667e-12** | dense: 6.576e-16 .. 1.280e-14 |
| worst row | `centro_oop_3x3/y35` at 1.667e-12 | `centro_oop_3x3/vertical` (SHIPPED) at 1.548e-13 |
| Hausdorff gap to the DENSE spectrum | **5.2e-13 .. 3.43e-12** | the shipped vertical rows read 7.42e-13 .. 1.85e-12 |

*(A trap, recorded: the first spectrum metric here sorted both spectra with
`np.sort_complex` and differenced them elementwise.  Most of a lossless
region's eigenvalues are purely imaginary, so their real parts are `+/-`
round-off and the lexicographic sort scrambles them -- two IDENTICAL spectra
reported a gap of `1.2e+01`.  The VERTICAL control, where both branches are
shipped code and must agree, is what caught it.  The metric is a Hausdorff
distance now.)*

The worst forced slanted row is within 11x of the shipped vertical reduction's
own residual on the same grid, and both are twelve decades below the answer.

**END TO END** (the arm the build did not run) -- the forced reduction driven
through a full `PMM2DStackPure.solve` against the dense path:

| cell | slant | dR | dT | dJones | the slant's own effect |
|---|---|---|---|---|---|
| centro out-of-plane | x35 | 8.465e-16 | 8.216e-15 | 4.515e-15 | 2.205e-02 |
| centro out-of-plane | diag35 | 1.846e-15 | 1.210e-14 | 6.607e-15 | 1.584e-02 |
| centro out-of-plane | **x60** | 1.056e-15 | 7.550e-15 | 5.962e-15 | **8.247e-02** |
| centro scalar | x35 | 5.249e-15 | 1.832e-14 | 2.483e-14 | -- |
| centro scalar | diag35 | 1.020e-15 | 1.060e-14 | 4.832e-15 | -- |
| centro scalar | **x60** | **7.126e-15** | **3.370e-14** | 1.471e-14 | -- |
| centro LOSSY | x35 | 3.296e-16 | 9.215e-15 | 2.568e-15 | 1.869e-02 |
| centro LOSSY | diag35 | 2.290e-16 | 1.943e-15 | 5.011e-15 | 1.270e-02 |
| centro LOSSY | **x60** | 9.680e-16 | 6.772e-15 | 3.630e-15 | **6.877e-02** |

**Machine precision, end to end, at every slant to 60 degrees, on a lossy cell
as on a lossless one.**

### 5.4 VERDICT

**BOUNDED-SAFE, and the refusal is no longer justified by anything measured --
but it should NOT be lifted on this evidence alone.**  Precisely:

* the build's stated reason ("the shear is a new geometry whose forward/backward
  split, gauge and reconstruction are validated only on the dense branch") is
  **half-refuted by the code**: `_stag_block_eig` returns ALL `4 q^2` pairs and
  `_region_modes_oop` then runs the SAME flux split and the SAME `_OOP_H_GAUGE`
  on them, so the split and the gauge are not on the reduction's side of the
  line at all.  What the reduction owns is the eig and the reconstruction
  `um = Yh up / q`, and both are measured here at machine precision on 28
  pencils and 9 end-to-end solves;
* nothing in this verification produced a wrong forced answer;
* the counterweight is real and is not a measurement: the hybrid's
  normal-incidence silent-wrong (`BUILD_PMM2D_SLANT_METRIC_2026_08_16` S8) was
  an even-parity fold that was ELIGIBLE at normal incidence, bypassed the
  convection, and returned the VERTICAL answer with energy conserved.  The
  shape of that failure is exactly "an accelerator that is structurally
  eligible on a slanted cell";
* **and the decisive gap: every fixture that passes the structural check here
  is CENTRO-SYMMETRIC, which is the condition.  What has NOT been shown is a
  slanted cell that PASSES the structural residual and is nonetheless
  reconstructed wrongly.**  Without that, the two-sided gate cannot be closed,
  and TESTING_STANDARDS rule 3 ("engineer the state; don't hope") says the
  build must construct it, not wait for it.

**The two-sided gate a build would need**, stated so it can be executed:

1. **Positive arm, end to end, not just the pencil.**  Forced reduction vs
   dense on `R`, `T` and BOTH Jones, over a family that spans: `(2,2)` and
   `(3,3)` grids, `M` 5..8, scalar / out-of-plane / lossy / high-contrast /
   non-reciprocal cells, slants to 60 degrees in x, y and diagonal.  Bar: the
   dense path's OWN pencil residual on the same fixture times documented
   decades -- never an absolute constant (the readings here are `<= 3.4e-14`
   against a shipped-vertical reference of `1.5e-13`).
2. **Negative arm 1 -- the check refuses what it must.**  An off-centre pillar,
   a parity-breaking tensor, an unmirrored wall layout, each SLANTED: the
   structural residual must sit decades ABOVE the bar (measured here:
   `5.768e-01` for the off-centre pillar) and `_stag_block_eig` must return
   `None`.
3. **Negative arm 2 -- the one that does not exist yet, and is the gate.**  A
   slanted cell for which the structural residual PASSES and the forced
   reduction is nonetheless wrong, OR a proof that no such cell exists.  The
   candidates to try: a slant whose `t` is NOT commensurate with the cell's
   centro-symmetry centre; a cell whose eps is its own parity image but whose
   WALL layout is not (the build's own note that the condition is on the
   discretisation, not on eps); a slanted cell at a Wood coincidence, where
   `min|q| -> 0` and the `um = Yh up / q` reconstruction divides by it
   (`_STAG_GAM_FLOOR` is the existing guard -- walk it two-sided at slant);
   and a slanted LOSSY cell whose `+/-q` pair is nearly degenerate.
4. **The `symmetry='auto'` end-to-end equality** must then flip from
   "identical to `symmetry=False`" to "identical to the DENSE answer within
   the bar in (1)", and the vertical control must keep proving the arms are
   not vacuous.

Until (3) exists, **keeping the refusal is the correct engineering call**, and
the library comment saying so is accurate.  Its cost is now quantified: the
accelerator is `1.5x .. 1.9x` on the out-of-plane region solve, which a slanted
cell always takes.

---

## 6. THE REFUSALS

`v6_refusals.py`, `v6b_readings.py`.  All twelve documented refusals fire, with
the right exception TYPE and a message that names the offending construction:

| construction | type | message names |
|---|---|---|
| two PATTERNED layers at different slants | `NotImplementedError` | "MIXED SLANTS", "PATTERNED", both layer indices and both slant values |
| a slanted PATTERNED layer + a VERTICAL patterned layer | `NotImplementedError` | same, and states that a vertical patterned layer counts as slant `(0.0, 0.0)` |
| vertical/slanted MIX above a pattern | `NotImplementedError` | "MIX of vertical and slanted regions", the layer index, the list of slants above it |
| `mu` + slant at `add_layer` | `NotImplementedError` | "mu" |
| `mu_cell` + slant at `add_layer` | `NotImplementedError` | "mu" |
| `Granet2DTransverseE(slant=, mu_cell=)` | `NotImplementedError` | "mu_cell" |
| `solve(retain_internal=True)` on a slanted stack | `NotImplementedError` | "retain_internal" |
| `pmm_efficiency_2d_staggered(slant=)` | `NotImplementedError` | points at `pmm_jones_2d_staggered` |
| `slant=(0.1, 0.2, 0.3)` | `ValueError` | "slant must be a (t_x, t_y) pair" |
| `slant=(nan, 0.0)` | `ValueError` | "finite" |
| hybrid: slant x OUT-OF-PLANE | `NotImplementedError` | "SLANTED", "OUT-OF-PLANE", and now points at the pure engine |
| `e33 = 0` with a slant | `ValueError` | the `_require_nonzero_ezz` gate |

### 6.1 The two ADMITTED frame-offset readings, measured

* **READING 2** (`Sh_i = t_i Z_i`, one global shear) -- the layer-split
  identity, one slanted layer of `d` against two of `d/2`:

| mount | dR | dT | dJones | dJones (transmission) |
|---|---|---|---|---|
| normal | 8.41e-17 | 1.11e-15 | 5.56e-16 | 1.20e-15 |
| conical | 1.87e-16 | 7.77e-16 | 7.74e-16 | 7.39e-16 |

  **EXACT to machine precision, on the transmission Jones too** (which is where
  the accumulated anchor lives).

* **READING 1** (`Sh_i = 0`, every layer above vertical) has no identity of its
  own, so `v6b_readings.py` builds the one comparison that pins both at once: a
  UNIFORM film above one slanted pattern, once with the film VERTICAL (reading
  1) and once at the SAME slant (reading 2).  A homogeneous film is
  translation-invariant, so the two describe the same solid up to a rigid
  lateral translation, and a rigid translation `delta` maps
  `S_mn -> exp(i(alpha_m - alpha_n) . delta) S_mn` -- leaving every efficiency
  and the zeroth-order (`m = n = 0`) Jones unchanged.  So the two stacks must
  agree, and the residual must FALL with `M`:

| M | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| oblique 25, dJones | 2.288e-03 | 1.152e-03 | 2.901e-04 | 7.408e-05 | **9.956e-06** |
| oblique 25, dJones (transmission) | 4.793e-03 | 1.011e-03 | 4.308e-04 | 5.085e-05 | **1.539e-05** |
| conical, dJones | 1.475e-03 | 3.232e-04 | 1.776e-04 | 4.339e-05 | **9.438e-06** |

  Monotone, 2.5 decades over `M` 4..8, **no floor**.  The convergence is
  ALGEBRAIC, which is the documented behaviour of this basis on a right-angle
  pillar (corner singularities), not a frame-offset error.
  **BOTH READINGS CONFIRMED EXACT, reading 1 bounded by the basis's own
  algebraic rate.**

* **UNIFORM-only layers at different slants** (the third accepted shape) is a
  physical no-op and reads `dR 1.27e-09 / dT 1.85e-09 / dJones 4.56e-09` at
  `M = 5` against the all-vertical stack.

* **A FOURTH accepted shape the build does not list**, found by reading
  `_check_stack_slant`: a slanted UNIFORM layer BELOW a VERTICAL patterned one
  is admitted (`pats = [0]`, nothing above it, so neither branch fires).  It is
  also a physical no-op, and the anchor correctly sums only that layer's
  `t d`.  Measured against the same stack with the film vertical, at conical
  incidence:

| M | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| dR | 1.716e-03 | 3.408e-04 | 1.383e-04 | 3.039e-05 | **1.147e-05** |
| dJones | 5.247e-03 | 2.466e-03 | 4.494e-04 | 1.609e-04 | **3.565e-05** |
| dJones (transmission) | 8.403e-03 | 1.605e-03 | 3.432e-04 | 1.724e-04 | **1.961e-05** |

  Monotone, two decades over `M` 4..8, no floor -- but note it is two decades
  ABOVE the single-layer uniform null at the same `M` (`4.1e-06`), because the
  slanted uniform layer is MATERIALISED as a constant cell on the union grid
  and takes its own `4 q^2` solve while the vertical one rides the shared
  eps-free geometric eig.  The two are different discretizations of the same
  homogeneous medium, and their difference is what this measures.  **BOUNDED:
  correct, and converging, but a caller who slants a uniform spacer pays two
  decades of accuracy at fixed `M` compared with leaving it vertical -- which
  nothing documents.**

### 6.2 Sneaking a slanted layer past a refusal

`_slant_is_zero` is an EXACT `== 0.0` test, so:

| attempt | what happens |
|---|---|
| `slant=(1e-17, 0)` on one of two patterned layers | **RAISES** `NotImplementedError`, message reads "layer 0 is slanted (1e-17, 0.0) while layer 1 is slanted (0.0, 0.0)" |
| `slant=(1e-17, 0)` into `pmm_efficiency_2d_staggered` | **RAISES**, message prints `slant=(1e-17, 0.0)` |
| `slant=(1e-17, 0)` on a single patterned layer | accepted; routes to the `4 q^2` OUT-OF-PLANE path (`offplane` flips `False -> True`) and the answer differs from vertical by `dR 9.4e-16 / dT 1.2e-14 / dJones 5.0e-15` |
| `slant` as a list / ndarray / bare scalar / `np.float64` | accepted, and **byte-identical** to the tuple spelling (`dR = dJones = 0.0` on all four) |
| a UNIFORM layer with a slant | accepted; no-op to `dR 2.2e-09 / dT 2.3e-09 / dJones 6.1e-09 / dJonesT 2.0e-08` at `M = 5` |
| `symmetry='auto'` on a slanted cell | **sha256-IDENTICAL** to `symmetry=False`; on the VERTICAL control they DIFFER, so the equality is not vacuous; the slant is worth `dJones 2.11e-02` on that cell |
| `_stag_parity_gauge` on a slanted solver | returns `None`; on the vertical control it returns a gauge |

### 6.3 The eig cache key, exercised rather than read

The build's `test_b11_slant_is_in_the_eig_cache_key` asserts the key's contents
directly, because the stack that WOULD collide is the one `_check_stack_slant`
refuses.  There is, however, one construction that reaches a genuine collision
through the shipped API: **two UNIFORM layers with byte-identical `eps` and
DIFFERENT slants** -- an accepted shape, and the only way two layers can differ
by slant alone.  Both are physical no-ops, so the correct answer is the
all-vertical stack; if the key dropped the slant, the second layer would be
solved with the FIRST one's modes (an x-shear's modes for a y-sheared layer)
and the error would be O(1):

| M | dR | dT | dJones | dJones (transmission) |
|---|---|---|---|---|
| 5 | 4.905e-08 | 1.809e-08 | 1.467e-07 | 2.206e-07 |
| 6 | 2.944e-10 | 2.719e-10 | 8.762e-10 | 1.289e-09 |
| 7 | **1.085e-12** | **1.077e-12** | **3.194e-12** | **4.627e-12** |

Spectral, to `1e-12`.  And the genuine cache HIT (two uniform layers at the
SAME slant and the same `eps`) also reproduces the vertical stack
(`dR 2.96e-10`, `dJones 7.79e-10` at `M = 6`), so the hit path is right too.
**CONFIRMED.**

**Nothing gets past a refusal silently.**  The one behaviour worth a note is
the third row: a float-noise slant (`tan(0)` computed as `1e-17`, say) is
treated as SLANTED everywhere -- it costs the `4 q^2` path (a `1.9x .. 3.0x`
region solve) and it makes the scalar efficiency entry raise.  That is the loud
side of the exact-zero test and is arguably correct, but a caller who
constructs `slant` arithmetically will meet it.

---

## 7. THE WOOD-LIST DECISION

`v7_wood.py`.  Own fixture: a slanted scalar layer (`eps = 4`, `t = (0.6, 0)`)
walked onto each candidate cut-off by choosing the period, at NORMAL incidence
on a SQUARE cell so that all four first orders sit at
`|alpha| = sqrt(candidate)` together.  `M = 6`, `(2,2)` grid.

**The prediction, stated before the run.**  At the LAB candidate the `(0, +/-1)`
orders have `t . alpha = 0`, so `q = +/- sqrt(eps - |alpha|^2) = 0` -- a genuine
null mode.  At the COVARIANT candidate the same orders give
`q = +/- i sqrt(5.44 - 4) = +/- 1.2 i`, i.e. `|q| = 1.2` exactly.

| candidate | value | `wl / px` | the SLANTED layer's `min abs(q)` | vertical control | nudge fires (LAB list) | (covariant list) |
|---|---|---|---|---|---|---|
| **LAB `eps`** | 4.0000 | 2.0000 | **1.0277e-03** | 1.0277e-03 | **yes** | yes |
| covariant `eps (1 + t^2)` | 5.4400 | 2.3324 | **1.2000e+00** | 1.2000e+00 | **no** | yes |
| per-order `abs(a)^2 + (t.a)^2` | 1.3600 | 1.1662 | 4.3164e-01 | 1.1314e+00 | no | yes |

`1.2000e+00` is the analytic value to five digits, so the covariant row is not
a near-miss: **it is not a cut-off at all**, and listing `eps (1 + t^2)` would
nudge the wavelength on a NON-EVENT.  The build's own readings (`7.672e-03` and
`1.200`) are reproduced in structure and, for the covariant row, to the digit;
the lab row differs because it is a discretization residue of a true null mode
and my `M` / grid / period differ from the build's.

**And a slanted layer sitting on its LAB cut-off IS nudged**: solving the same
stack at the exact coincidence and at `wl (1 + 1e-7)` gives `dJones = 0.000e+00`
-- bit-identical, because the guard has already moved the first solve to the
second's wavelength -- with closure `1.000000000000` in both.  Same for the
vertical control.

**CONFIRMED**, including the claim that the covariant reading would nudge on a
non-event.

---

## 8. DURABILITY AUDIT of `tests/unit/test_pmm2d_staggered_slant.py`

`v9_durability.py` re-measures every numeric bar in the file, on the file's OWN
fixtures (imported from it), on BOTH builds.

### 8.1 Do the comment numbers reproduce?

**Yes -- every one, to three or four significant digits, on both builds.**  The
table below gives reading, bar, margin and the comment's claim; `spread` is the
WIN/WSL relative difference.

| quantity | WIN | WSL | spread | bar | margin | comment says |
|---|---|---|---|---|---|---|
| b1 congruence round-trip | 5.0717e-16 | 5.0717e-16 | 0 | 1e-10 | 197172x | `<= 1e-15` OK |
| b2 uniform-null worst of 60 | 4.0977e-06 | 4.0977e-06 | 6e-08 | 1e-04 | 24.4x | 4.098e-06 OK |
| b2 order leak worst of 60 | 1.7664e-07 | 1.7664e-07 | 1e-08 | 1e-05 | 56.6x | 1.766e-07 OK |
| b2 worst at NORMAL (`R` only) | 5.294e-15 | -- | -- | -- | -- | 5.29e-15 OK |
| b2 leak at NORMAL | 1.9488e-28 | 1.5882e-28 | 1.9e-01 | -- | -- | 1.95e-28 / 1.59e-28 OK |
| b2 ladder M 4..8 | 1.337e-04 2.955e-06 3.930e-08 3.389e-10 2.034e-12 | ... 2.022e-12 | 6e-03 | 1e-10 | 49.2x | all five OK |
| b3 shipped worst | 2.6367e-05 | 2.6367e-05 | 2e-09 | 1e-03 | 37.9x | 2.64e-05 OK |
| b3 "none" smallest | 1.4336e-01 | 1.4336e-01 | 3e-13 | 1e-02 | 14.3x | 1.43e-01 OK |
| b3 conj / none, smallest | 1.7878e+00 | 1.7878e+00 | 7e-14 | 1.5 | **1.2x** | "~2x" -- the SMALLEST row is 1.79 |
| b4 physical worst | 2.2619e-08 | 2.1498e-08 | 5e-02 | 1e-06 | 44.2x | 2.26e-08 / 2.15e-08 OK |
| b4 wrong-arm smallest | 3.5299e-02 | 3.5299e-02 | 6e-13 | 1e-03 | 35.3x | 3.53e-02 OK |
| b4 ablation smallest | 1.5779e-03 | 1.5778e-03 | 1e-05 | 1e-04 | 15.8x | 1.58e-03 OK |
| b4 ladder M 4/5/6 | 3.550e-07 6.351e-10 7.561e-13 | ... 7.661e-13 | 1e-02 | 1e-11 | 13.2x | OK |
| b5 TE worst | 4.2833e-06 | 4.2833e-06 | 8e-09 | 1e-04 | 23.3x | 4.28e-06 OK |
| b5 TM worst | 1.2195e-03 | 1.2195e-03 | 1e-11 | 5e-03 | **4.1x** | 1.22e-03 OK |
| b5 wrong-TE smallest | 6.2348e-03 | 6.2348e-03 | 3e-09 | 1e-03 | **6.2x** | 6.23e-03 OK |
| b5 wrong/right ratio smallest | 4.3412e+03 | 4.3412e+03 | 2e-05 | 100 | 43.4x | 4.3e+03 OK |
| b5 wrong-sign closure | 3.8495e-07 | 3.8495e-07 | 2e-07 | 1e-04 | 259.8x | 3.850e-07 OK |
| b6 Ex worst | 6.5968e-05 | 6.5844e-05 | 2e-03 | 1e-03 | 15.2x | 6.60e-05 OK |
| b6 Ey worst | 2.3200e-06 | 2.3548e-06 | 1e-02 | 1e-04 | 43.1x | 2.32e-06 OK |
| b6 slanted / control | 7.8833e-01 | 7.8833e-01 | 6e-09 | 3.0 | **3.8x** | -- |
| b7 split exact, all rows | True | True | -- | exact | -- | OK |
| b7 lossless `\|min Re lam_f\|` | 1.7139e-14 | 1.8648e-14 | 8e-02 | 1e-12 | 58.3x | -1.71e-14 OK |
| b7 lossy `min Re lam_f` | 3.1147e-02 | 3.1147e-02 | 2e-14 | 1e-03 | 31.1x | +3.11e-02 OK |
| b7 `max\|q\|` ratio at 60 deg | 1.5085 | 1.5085 | 6e-15 | 3.0 | **2.0x** | 1.51 OK |
| b8 forward growth - 1 | 1.5277e-13 | 9.9476e-14 | 3e-01 | 1e-09 | 6546x | 1.5e-13 OK |
| b8 closure at 3 lam | 1.4347e-02 | 1.4347e-02 | 5e-11 | 5e-02 | **3.5x** | 1.48e-02 (1.4347e-02 here) |
| b9 layer split worst | 1.5572e-15 | 1.4722e-15 | 5e-02 | 1e-12 | 642x | 1.67e-15 OK |
| b10 no-floor normal | 1.7764e-15 | 1.4988e-15 | 2e-01 | 1e-12 | 563x | 1.55e-15 OK |
| b10 no-floor conical | 2.7580e-06 | 2.7580e-06 | 3e-07 | 1e-04 | 36.3x | 2.758e-06 OK |
| b11 forced structural residual | 2.3129e-15 | 2.3129e-15 | 0 | 1e-10 | 43235x | 2.31e-15 OK |
| b11 slant visible dJones | 2.1125e-02 | 2.1125e-02 | 1e-13 | 1e-03 | 21.1x | 2.11e-02 OK |
| m4 normal ladder `+t` | 2.688e-02 1.926e-02 1.234e-02 | identical | 1e-12 | -- | -- | OK |
| m4 normal ladder `-t` | 2.843e-01 2.856e-01 2.787e-01 | identical | 1e-15 | -- | -- | OK |
| m4 conical ladder `+t` | 3.924e-02 2.630e-02 1.787e-02 | identical | 9e-14 | -- | -- | OK |
| m4 conical ladder `-t` | 3.730e-01 3.756e-01 3.683e-01 | identical | 2e-15 | -- | -- | OK |
| m4 normal `+t` final | 1.2345e-02 | 1.2345e-02 | 9e-13 | 5e-02 | **4.1x** | -- |
| m4 normal `-t` final | 2.7873e-01 | 2.7873e-01 | 1e-15 | 1e-01 | **2.8x** | -- |
| m4 normal `-t` no-improvement | 0.98035 | 0.98035 | 2e-14 | 0.8 | **1.2x** | -- |
| m4c normal with-ratio | 0.35814 | 0.35814 | 9e-14 | 0.5 | **1.4x** | 0.358 OK |
| m4c normal against-ratio | **0.87511** | **0.87511** | 6e-14 | 0.8 | **1.09x** | 0.875 OK |
| m4c conical with-ratio | 0.45288 | 0.45288 | 2e-15 | 0.5 | **1.1x** | 0.453 OK |
| m4c conical against-ratio | 0.98581 | 0.98581 | 2e-16 | 0.8 | **1.2x** | 0.986 OK |
| cost ratios (M = 5 / 6) | 0.915 / **1.167** | 0.865 / 1.105 | 5e-02 | 2.0 (min) | **2.2x** | "0.96x .. 1.07x" -- **the M=6 rung reads 1.167 here** |

Two entries need a word:

* **b2 "worst at NORMAL incidence"** reads `2.58e-14` if you take
  `max(dR, dT, dJones)` and `5.294e-15` if you take `dR` alone.  The build's
  table labels it `(R)`, and with that definition it reproduces to the digit.
  Not a defect; noted because the difference looks like one.
* **the COST ratio moved.**  The build cites `0.96x .. 1.07x` across `M` 4..7;
  the `M = 6` rung reads `1.167` on WIN and `1.105` on WSL here.  A wall clock
  is the one quantity a runner is entitled to move (this box was under load
  from a second worktree), and the test takes the MIN over two rungs against a
  `2.0` bar, so the assertion is safe -- but the cited RANGE is not a durable
  number and the comment should say "measured under no load".

### 8.2 Sub-decade bars (test-only; nothing here is a library problem)

**22 of the file's bars have a margin under a decade.**  Read against the
cross-build spread, almost all of them are nonetheless SAFE: the quantities are
deterministic discretization, and WIN/WSL agree to `1e-13` or better on
`b3_conj_over_none`, `b5_TM`, `b5_wrong_TE`, `b6_slanted_over_control`,
`b7_maxq_ratio`, `b8_closure`, every `m4` and every `m4c` reading.  A 9% margin
on a quantity whose measured build-to-build spread is `6e-14` clears the
envelope by twelve decades, which is what rule 5 actually asks.

The genuine risks, in order:

1. **`cost_min_ratio` (margin 2.2x, and the one quantity with real spread).**
   A wall clock under load.  `min` over two rungs saves it today
   (`0.915` / `0.865`), but both rungs' M=6 reading has already drifted 9-16%
   above the comment's range.  **Restate**: keep the `2.0` bar, delete the
   "0.96x .. 1.07x" range from the docstring or qualify it as unloaded.
2. **`m4c_*_against_ratio` (0.87511 / 0.98581 against a `0.8` bar).**  These
   are the file's thinnest bars and the quantity is a STAIRCASE's geometric
   error ratio -- exactly the thing an intentional basis or grid change would
   move by tens of percent.  **Restate** as a decision that does not need the
   number: assert that the AGAINST direction does not IMPROVE more than the
   WITH direction does (`against[1]/against[0] > with[1]/with[0]`, measured
   `0.875` vs `0.358` and `0.986` vs `0.453` -- a 2.2x-2.4x separation), which
   is the claim being made and carries its own two-sidedness.
3. **`m4_*_plus_monotone` (strict 3-rung monotonicity on a CROSS-ENGINE
   ladder).**  Measured step ratios `1.40` and `1.47`; the hybrid's OWN
   `n_orders` step at these sizes is `1.26e-02 .. 1.73e-02`, the same size as
   the gaps between the rungs (`0.76e-02`, `0.70e-02`).  It passes identically
   on both builds, but a third BLAS is not covered by two.  **Restate** as
   first-to-last improvement plus a floor, not rung-by-rung strictness.
4. **`b3_conj_over_none_smallest` (1.79 vs a `1.5` bar).**  Safe on spread; the
   comment says "about TWICE" while the smallest row is `1.79` -- worth stating
   the actual range (`1.787 .. 1.991`, measured over the 12 rows of
   S3.2) so the bar's origin is visible.

Everything else in the file follows TESTING_STANDARDS: decisions rather than
readings, engineered wrong arms next to every right one, sha256 for every
bit-identity claim, and no `pytest.skip` anywhere.  **No S1-S5 shape violation
was found.**

### 8.3 Timing

| build | `test_pmm2d_staggered_slant.py` | slowest test |
|---|---|---|
| **WIN (this run, under load)** | **70 passed in 132.34 s** | 8.74 s (`test_m4_slanted_pillar_vs_independent_hybrid_metric[conical]`) |
| **WSL (this run, under load)** | **70 passed in 123.37 s** | 8.36 s (the same test) |
| the build doc's claim | 81.2 s (WIN, 85.8 s under load) / 84.7 s (WSL) | 5.68 s (the same test) |

Both arms GREEN, and both are ~1.5x the build's numbers -- this box was running
a second worktree's suites throughout (a 4.2 GB python at 100% of a core), which
is the same factor the COST ratio drifted by (S8.1).  The plan's cost rules
still hold with room: the file is 2.2 min against a 4-minute budget and the
slowest test is 8.7 s against a 40 s cap.  **The three slowest tests are the
same three on both builds**, which is the durable statement; the wall itself is
not.

### 8.4 `.test_durations`

The build claims the file was spliced by dict union, 70 new entries and nothing
else touched.  Re-measured against the base clone's copy:

| | reading |
|---|---|
| keys on the tip | 12443 |
| keys on the base (`2efc7a2`) | 12373 |
| tip-only keys | **70, all of them `test_pmm2d_staggered_slant.py::`** |
| base-only keys (i.e. dropped) | **0** |
| sum of the 70 slant durations | 81.0 s |

**CONFIRMED** -- the `--clean-durations` trap the build records was avoided.

---

## 9. SUITE RUNS

All nine files named in the verification brief, on WIN, `OMP` = `OPENBLAS` =
`MKL` = 1, with a second worktree's suites running throughout:

| files | result |
|---|---|
| `test_pmm2d_staggered_slant.py` | **70 passed in 132.34 s** |
| `test_pmm2d_staggered_oop.py`, `test_pmm2d_staggered_oop_block_eig.py`, `test_pmm2d_staggered_anisotropic.py`, `test_pmm2d_staggered_magnetic.py`, `test_pmm2d_staggered_wood_list.py`, `test_v5_12_0_pmm2d_staggered.py`, `test_v5_21_pmm2d_staggered_oblique.py`, `test_v5_14_0_pmm2d_stack.py` | **195 passed, 23 warnings in 700.73 s** |
| **total** | **265 passed, 0 failed** |

**265 is exactly the count the `wave2/pmm2d` merge (`8b9af801`) was
coherence-tested at**, so this reproduces the merge's own gate.  (The build
doc's 296 counts one more file, `test_p2c_pmm2d_stack_cascade.py`, which the
verification brief does not list; `296 - 265 = 31` is that file's test count.)
The 23 warnings are all the pre-existing `PMM2DStack is a TRANSITIONAL alias`
deprecation, raised by `test_v5_14_0_pmm2d_stack.py` on purpose.

`test_pmm2d_staggered_slant.py` was additionally run on WSL: **70 passed in
123.37 s** (S8.3).

**`ruff check lumenairy/ tests/` -- `All checks passed!`**  (`validation/` is in
`pyproject.toml`'s `extend-exclude`, so the probe directory is out of scope by
configuration; it is not lint-clean and does not need to be.)

**`.test_durations`**: nothing to splice -- this verification adds no tests.
The build's own splice was audited instead and is correct (S8.4).

---

## 10. Defects and open items

| # | severity | what | where |
|---|---|---|---|
| **D1** | **silent-wrong, pre-existing, OTHER engine** | `PMM2DStackHybrid.jones_transmission()` / `per_order_amplitudes('transmission')` return FRAME-referenced amplitudes on a stack containing a slanted PATTERNED layer -- wrong by `5.914e-01 / 5.610e-01` PER ORDER at 25-degree incidence, and no single global phase can repair it (the best reaches `4.631e-01 / 4.447e-01` where the per-order factor reaches `3.122e-02 / 1.636e-02`).  `R`, `T` and the reflection Jones are correct, so no energy or efficiency check can see it, and no existing test covers it.  Fix: the same per-order factor `exp(+i k0 (alpha_m . slant) d)` on the transmitted amplitudes that `stack2d_pure.solve` applies.  **No library edit made.** | `lumenairy/elements/pmm/stack2d.py` (`PMM2DStackHybrid.solve`); reproducer `validation/probe_verify_slant/v8_hybrid_anchor.py` |
| **D2** | documentation | The build's COST range "`0.87x .. 1.07x`" / the test's "`0.96x .. 1.07x`" does not hold under load: `M = 6` reads `1.167` (WIN) / `1.105` (WSL) here.  The assertion is unaffected. | `docs/audits/BUILD_..._SLANT_2026_09_10.md` S10, `tests/unit/test_pmm2d_staggered_slant.py::test_cost_...` |
| **D4** | scope, undocumented | A slanted UNIFORM layer is MATERIALISED as a constant cell on the union grid and takes its own `4 q^2` solve, while a vertical one rides the shared eps-free geometric eig.  The two are different discretizations of the same homogeneous medium, so slanting a uniform spacer costs about two decades of accuracy at fixed `M` (measured `3.4e-04` at `M = 5` against the single-layer uniform null's `4.1e-06`), and converges only algebraically.  Nothing documents this. | S6.1, fourth accepted shape |
| **D3** | scope, undocumented | B2's `1e-04` null bar at `M = 5` is a statement about slants `<= 35 deg`; at 60 degrees the same measurement reads `1.471e-04`.  The ladder shows it is spectral, so "exact at any slant magnitude" is true but costs `M` at steep tilt, and nothing in the test file exercises 60 degrees end to end. | S4.1 |
| **O1** | open | The parity accelerator on slanted cells: measured safe on 28 pencils and 9 end-to-end solves, but the third arm of the gate (a slanted cell that PASSES the structural check and is nonetheless wrong, or a proof there is none) does not exist.  S5.4 states the gate. | -- |
| **O2** | usability note | `_slant_is_zero` is an exact `== 0.0` test, so an arithmetically-constructed `1e-17` slant routes to the `4 q^2` path and makes the scalar efficiency entry raise.  Loud, not silent -- but a caller computing `tan(angle)` will meet it. | S6.2 |

---

## 11. What this verification could NOT establish

* **The public slant's geometry from the SOURCE.**  S2.1: the congruence and the
  frame-anchor phase are consistent with each other only through
  `_OOP_ROT_SIGN`'s claim that the basis frame is the 180-degree image of the
  cell / far-field frame.  That claim is confirmed by measurement (S2.2-S2.4)
  and by the fact that the whole chain closes, but it is not derivable from the
  code as written, and no test asserts it directly.
* **A finer PURE-engine staircase.**  The nodal union grid admits only slice
  counts dividing the walk in grid cells (S2.7), so within the pure engine the
  staircase oracle saturates at the size of the effect for slants near 45
  degrees.  Every staircase claim here is therefore either 1-D (S2.2) or
  Fourier-hybrid (S3.3).
* **The third arm of the parity gate** (S5.4 item 3) -- deliberately, because
  constructing a counterexample is a build task, not a verification one.
* **`retain_internal` / `layer_absorption` under a slant** -- refused, so there
  is nothing to measure.
* **Anything about TAPERS** -- out of scope by construction, and the refusal
  wording is correct.

---

## 12. Commits

On `verify/slant` (off `8b9af801`).  **No library file was edited, nothing was
merged, pushed, tagged or version-bumped, and no test was added or changed** --
the defects in S10 are reported with reproducers, not patched.

| commit | what |
|---|---|
| `24c6a16` | `test(pmm2d slant verify)`: `validation/probe_verify_slant/` -- the 16 probes, the README (including the two traps they hit first) and every run's JSON, on both builds |
| (this one) | `docs(pmm2d slant verify)`: this document |

Re-running everything:

```
cd validation/probe_verify_slant
PYTHONPATH=C:/tmp/lum_vslant OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   MKL_NUM_THREADS=1 python -W ignore -u v9_durability.py       # and v1..v10
```

and the WITHOUT arm against the read-only main clone:

```
PYTHONPATH="D:/Metacept/.../Lumenairy" python v1_without_identity.py
python v1_compare.py
```


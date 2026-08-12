# BUILD -- the screen-obliquity correction, and its accuracy guard

**2026-08-11.  Branch `feat/screen-obliquity`, cut from `main` @ `755ad99`
= v5.34.0, in the worktree `C:/tmp/lum_obl`.  No commit, no push, no `gh`,
no `CHANGELOG`.**

Builds route 1 of `BUILD_ANGLE_AWARE_LENS_2026_08_11` S6 -- the ONE angular
defect that refutation leaves standing.  The prize it sized is "~0.2 waves on
a fast doublet at 55 mrad, not ~40"; this measures that term against exact ray
traces, derives it in closed form, ships it, and turns the same expression
into an error estimator that warns.

---

## 0. VERDICT

> **SHIPPED, DEFAULT-ON BEHIND A NEW `carrier=` KEYWORD, WITH AN EXACT
> CLOSED FORM AND A MEASURED GUARD.**
>
> **1. THE DERIVATION IS AN IDENTITY, NOT AN EXPANSION.**  Moving a plane
> facet of unit normal `nu` a height `s` onto the vertex plane changes the
> EXIT-REFERENCED eikonal by exactly `s (n1 cos a_in - n2 cos a_out)`, with
> both angles to the **z-axis** and `a_out` from exact vector Snell at `nu`.
> No small-angle expansion, no small-sag expansion.  The shipped screen is
> that expression at `a_in = a_out = 0`; the correction is the difference.
> Checked against `mpmath` ray algebra on a tilted facet at 40 digits: the
> residual is 0 to the sag-proportional order at every slope, index ratio
> (including `n1 > n2`) and angle up to 200 mrad tried (S2.4).
>
> **2. THE PLATE IS EXACTLY ZERO, AND SO IS THE CARRIER-FREE CALL.**  Both by
> construction (the correction is PROPORTIONAL to the sag, and is a
> DIFFERENCE against the zero-angle screen), both pinned with
> `np.array_equal` on the returned complex arrays, not with a tolerance, and
> the carrier-free null holds CROSS-TREE against a pristine v5.34.0 checkout
> on 10/10 option combinations (S7).  The plate is the control that killed
> the previous design, which was wrong by **2.77 waves** there.
>
> **2a. AND THE SIGN IS CONTROLLED THE SAME WAY THE PREVIOUS ONE WAS KILLED.**
> Negating the correction on a single R = 19.6 mm facet at 54.9 mrad lands on
> **2.004x** the blind error while adding it lands on 0.9 % of it -- the
> "wrong by twice the term" signature, now pointing the other way (S2.6,
> `test_the_sign_is_load_bearing`).
>
> **3. AGAINST EXACT RAY TRACES IT REMOVES 99 % OF THE TERM ON A SINGLE
> CURVED FACET AND 65 % ON THE HARDEST ELEMENT IN THE CAMPAIGN.**  Screen-model
> exit-plane angular error, common-mode controlled, 3 mm pupil:
>
> ```
>   N-BK7  R=+50 mm, 54.9 mrad     0.01345 w rms  ->  0.000018 w   (736x)
>   N-SF11 R=+25 mm, 54.9 mrad     0.05290 w      ->  0.000341 w   (155x)
>   d121 group 2 (PK52A/SF57)      0.01680 w      ->  0.000552 w   ( 30x)
>   d121 group 3 (LAK8 singlet)    0.00991 w      ->  0.000020 w   (496x)
>   d121 group 5 (SK2/SF57), 54.9  0.25848 w      ->  0.090692 w   (2.9x)
> ```
>
> **4. AND THE SIZE THE REFUTATION PREDICTED IS CONFIRMED.**  S2.3 of the
> refutation put the screen's own angular error at "the same order" as the
> 0.2118-wave S7 residual column.  Measured directly here, against exact
> rays at the exit plane, design 121 group 5 reads **0.25848 waves rms** and
> the 1/2/3 mm ladder reads **0.02459 / 0.10449 / 0.25848** against the
> refutation's **0.0200 / 0.0849 / 0.2118** -- the same number to a constant
> factor **1.227 +- 0.005 across all three radii.**  The refutation's scale
> argument was right and its column IS this term.
>
> **5. THE d121 GROUP-5 FLOOR IS A DIFFERENT DEFECT, AND IT IS NAMED.**  A
> tangent-facet arm (exact refraction at the local tangent plane on the vertex
> plane + the translation identity) reproduces the exact rays on that group to
> **0.000372 waves** -- 695x better than the shipped screen and 244x better
> than the corrected screen.  So the identity is exact and the residual is
> entirely the OTHER angle-blindness of a screen: its momentum kick
> `-(n2-n1) grad sag` does not depend on incidence, while a real facet's
> deflection does.  That is a DEFLECTION error, not a sag OPD, it enters only
> through the subsequent propagation, and **no screen of the form
> `f(x, y) sag(x, y)` can carry it.**  Nothing here claims to.
>
> **6. THE REFERENCE-PLANE BAR IS CLEARED.**  On the refutation's own
> `hmap_consumer2_121.py field` configuration and its own branch base
> (`b67130a`), reproduced to every printed digit: blind **0.37447 w rms**,
> naive (refuted) correction **0.40688**, this correction **0.37294**.  Below
> the bar.  The arm is nearly blind to the feature exactly as S4.4 says
> (1.004x) -- so the same control was re-run on a case it CAN resolve
> (Nyquist-sampled fast singlet, model-vs-model floor 0.00159 w) where it
> reads **0.00448 -> 0.00157 w rms, 2.86x, i.e. the corrected arm sits ON the
> floor.**
>
> **7. THE GUARD FIRES ON EXACTLY ONE OF DESIGN 121'S SIX GROUPS.**  Group 5
> (estimate **0.23910 waves**, against the exact-ray truth 0.25848 -- 7.5 %
> low); silent on the two plates, the two 51.5 mrad groups, the singlet, and
> the 7.4 mrad group.  Silent for every carrier-free call by construction.
>
> **8. NO PIN MOVED.**  The correction is reachable ONLY through a `carrier=`
> keyword that did not exist in v5.34.0, so no pre-existing call site can
> reach it; the adjudication ledger (S7) is empty and that is a structural
> statement, not a survey.

---

## 1. WHAT SHIPPED

| file | change |
|---|---|
| `lumenairy/elements/_lens_real.py` | the derivation comment, `_screen_obliquity_angle_field` / `_facet_axial_momenta` / `_screen_obliquity_delta` / `_screen_obliquity_pupil_radius` / `_screen_obliquity_rms_waves` / `_check_screen_obliquity_support`, three new `apply_real_lens` kwargs, the in-loop correction and the post-loop guard |
| `tests/unit/test_screen_obliquity.py` | NEW.  28 tests: plate zero, both byte-nulls, the closed form against exact ray traces, the guard, the validators, the carrier vocabulary |
| `validation/repro_traced_carrier_121/screen_obliquity_derive.py` | NEW.  `plate` / `sphere` / `d121` / `ablate` / `field` / `fieldb` / `guard` |
| `docs/audits/BUILD_SCREEN_OBLIQUITY_2026_08_11.md` | this note |
| `validation/repro_traced_carrier_121/hmap_consumer2_121.py`, `hmap_screen_proto.py` | COPIED UNCHANGED from `feat/angle-aware-lens` @ `d8c4080` (they are committed there, not on `main`).  They are the refutation's own control instruments and `screen_obliquity_derive.py` imports `mode_field`'s helpers and `geometry()` from them, so its `field` arm IS that arm and not a re-implementation.  Not edited by a byte. |

Public surface, all three defaulting to the pre-5.35 behaviour:

```
apply_real_lens(...,
    carrier=None,               # TiltedCarrier / signed conjugate / 'auto' / ndarray
    screen_obliquity='auto',    # 'auto' (on iff carrier) / True / False
    on_screen_obliquity='warn') # 'warn' / 'error' / 'silent'
```

---

## 2. THE DERIVATION

### 2.1 What is and is not missing

`apply_real_lens` is a split-step model: a zero-thickness phase screen on each
surface's VERTEX plane, homogeneous angular-spectrum propagation between them.
`BUILD_ANGLE_AWARE_LENS_2026_08_11` S2 proves in closed form that the ASM steps
carry the GAPS' angular optical path exactly -- a plane-parallel plate is
machine-exact at every tilt -- so the entrance-referenced obliquity piston
`n t (1/cos - 1)` is NOT missing and adding it is a double-count.

What is left is the screen itself.  It imprints `(n2 - n1) sag(x, y)` however
obliquely the ray crosses the sag, which is the `~ sag * theta**2` bound
`apply_real_lens`'s docstring has always quoted.  **That is the only term this
build corrects.**

### 2.2 The axial-translation identity (exact)

Take a locally PLANE facet with unit normal `nu`, media `n1 -> n2`, sitting at
height `s` above the vertex plane, between two fixed reference planes.
Compare it to the same facet translated down onto the vertex plane.

Split the "facet at height `s`" system exactly into three pieces:

```
  [ n1 slab, thickness s ]  o  [ facet at height 0, over the remaining span ]
                            o  [ remove an n2 slab, thickness s ]
```

* The middle system is a plane interface, so its exit direction `p_out` is a
  constant of the pupil and its total eikonal obeys
  `Lam(u) - p_out . x_out(u) = const` in the entrance coordinate `u` -- that
  is just "the exit wavefront exists".
* The first slab shifts the entrance coordinate by `u -> u + s tan a1` and
  adds `n1 s / cos a1`.  Using the constant above, the shift costs
  `- p_in . (s tan a1) = - n1 s sin a1 tan a1`, and

  ```
  n1 s / cos a1 - n1 s sin^2 a1 / cos a1  =  n1 s cos a1.
  ```

* The third piece removes `n2 s / cos a2` and un-shifts the exit point by
  `- s tan a2`, contributing `+ p_out . (s tan a2)`; by the same algebra that
  is `- n2 s cos a2`.

Adding them:

```
  Lam(facet at height s)  =  Lam(facet at height 0)  +  s (n1 cos a1 - n2 cos a2)
                                                                              (2)
```

**exactly**, with `a1`, `a2` the ray's angles to the **z-AXIS** before and
after refraction (not to the facet normal), and `a2` the TRUE post-refraction
angle at the tilted facet.  In the screen convention `exp(-i k0 OPD)`
(`Lam -= OPD`) the eikonal-exact screen OPD is therefore

```
  OPD_i(x, y) = (pz2 - pz1) * sag_i(x, y),                                    (3)
  pz1 = n1 cos a1 = sqrt(n1^2 - |p|^2),
  pz2 = n2 cos a2 = pz1 + (B - A) nu_z,
  nu  = (-grad sag_i, 1) / sqrt(1 + |grad sag_i|^2),
  A   = (n1 d_in) . nu = (-gx px - gy py + pz1) * nu_z,
  B   = sqrt(n2^2 - n1^2 + A^2),   nu_z = 1 / sqrt(1 + gx^2 + gy^2).
```

Equation (3) is the module's own equation (1) for `surface_model='displaced'`
-- **now derived rather than back-projected from a local plane wave**, and
with the exact tilted-facet `a2` rather than a fan-interpolated cosine.

### 2.3 What is APPLIED: the angular part only

The zero-angle value of (3) is the analytic model's documented normal-incidence
behaviour, which everything downstream is calibrated on (and which the
`slant_correction` / `'displaced'` axis already addresses).  So the correction
is (3) MINUS its carrier-free value:

```
  dOPD_i(x, y) = [ (pz2 - pz1)|_{p0 + q}  -  (pz2 - pz1)|_{p0} ] * sag_i(x, y)
                                                                              (4)
```

* `q(x, y)` -- the carrier's transverse optical momentum, i.e. its eikonal
  gradient `(L, M)`.  Constant for a collimated tilt; a full field for a
  conjugate / `'auto'` / explicit-wavefront carrier.
* `p0(x, y) = - sum_{j<i} (n2_j - n1_j) grad sag_j(x, y)` -- the carrier-FREE
  momentum the screen model itself accumulates, evaluated at the same field
  point.  Closed form; no ray trace, no map, no cache.

Properties, all exact rather than approximate:

1. `sag == 0` (a plane plate, at any tilt) -> `dOPD == 0`.
2. `q == 0` (no carrier, or a zero-angle carrier) -> `dOPD == 0`.  **The
   byte-null.**
3. It composes with whichever screen the caller selected, because it is a
   difference against that screen's own zero-angle value.

### 2.4 The leading order, and what bounds the rest

Expanding (3) for a facet whose slope can be neglected, `pz = sqrt(n^2-|p|^2)`
gives `Phi(p) = sqrt(n2^2-p^2) - sqrt(n1^2-p^2)` and

```
  dOPD_i  ~  sag_i * (n2 - n1) / (2 n1 n2) * ( |p0 + q|^2 - |p0|^2 )
          +  O(|p|^4) * sag_i * (n2^3 - n1^3) / (8 n1^3 n2^3)
```

-- i.e. `(n-1) sag theta^2 / 2n` for a collimated air-side surface: the
docstring's `sag * theta**2` with its exact prefactor (0.206 for `n = 1.7`).
**The quadratic form alone is NOT enough**: on design 121 group 5 it makes the
error WORSE (0.25848 -> 0.29321 w) because the cross term `2 p0 . q` is 3.7x
the pure `|q|^2` there and the facet slope reaches 0.22.  Only the full
tilted-facet form (3) is safe (S3.2).

The exact residual of (4) has three named sources, in measured order:

| # | term | scales as | measured |
|---|---|---|---|
| R1 | the screen's angle-blind DEFLECTION kick (S5) | gap thickness x `(n2-n1) sigma` x angle | 0.0907 w on d121 g5, 0.0006 w on a lone R=19.6 mm facet |
| R2 | `p0` taken at the field point rather than along the model ray | transverse walk x `d(grad sag)/dx` | < 1 % of R1 (measured by swapping in the true model momentum: 0.090692 vs 0.091586) |
| R3 | facet curvature across the sag traverse | `s'' w^2 / 2`, `w ~ sag tan a2` | ~1e-9 m = 0.001 w at d121 g5 |

An intermediate form was also derived and measured -- `Phi(|p|) sag(x + w)`
with `w` the refracted ray's transverse walk across the sag height, which is
(3) linearised in the slope.  It is 3-4x worse than (3) at `sigma ~ 0.12-0.22`
and is kept in the validation script as `form='full'` so the ladder shows what
the exact Snell leg buys.

### 2.5 Verification of the identity itself

`mpmath` at 40 digits, exact tilted-facet ray algebra vs (2), differenced
against normal incidence.  `s0` from 100 to 296 um, `sigma` from 0.06 to 0.30,
`(n1, n2)` including `1.5 -> 1.0`, angle to 200 mrad: the residual is the
s-INDEPENDENT prism term only.  Proved by setting `s0 = 0` (the identity then
predicts exactly zero) and finding the whole residual there, scaling linearly
with the following gap thickness: 0 at `t = 0`, -0.0104 w/mm on the steepest
facet.  **That is R1, isolated.**

### 2.6 THE SIGN CONTROL

The refutation's verdict turned on a sign: the previous design was "wrong by
twice the term".  So the same test is run here deliberately.  Negating (4) on a
single R = 19.6 mm N-SSK2 facet at 54.9 mrad, 3 mm pupil:

| arm | exit-plane angular error |
|---|---|
| shipped screen | 0.072821 w rms |
| **+ correction (4)** | **0.000641 w rms** (114x better) |
| - correction (4) | 0.145937 w rms (**2.00x** worse) |

**2.004x, not 1.9x or 2.1x** -- the signature of a term that exactly cancels
the defect.  A correction of the right size and the wrong sign lands on
exactly double; this one lands on 0.9 % of it.

---

## 3. THE MEASUREMENTS

The oracle is the shipped exact ray tracer.  The model arm is
`apply_real_lens`'s OWN system traced as a Hamiltonian ray system --
`Lam -= OPD`, `p -= grad OPD` at each screen; `x += t p/pz`,
`Lam += t n^2/pz` through each gap (the Legendre transform of the ASM kernel,
so the trace IS the wave model's geometric limit).  Both arms are eikonals of
the same kind, so they can be differenced AT THE EXIT PLANE: the model's value
is carried to the exact ray's exit point by `Lam += p . (x_e - x_m)`.  The
score is the COMMON-MODE-controlled `D(theta) - D(0)`, piston and tilt
removed, which cancels everything angle-independent -- the model's documented
normal-incidence ceiling included.

### 3.1 (a) The plane plate -- exactly zero

`screen_obliquity_derive.py plate`:

| tilt | correction - shipped | byte-identical | shipped angular error |
|---|---|---|---|
| 0.0 mrad | 0.000e+00 m | True | 0.000e+00 w |
| 10.0 mrad | 0.000e+00 m | True | 1.413e-26 w |
| 20.0 mrad | 0.000e+00 m | True | 1.413e-26 w |
| 41.5 mrad | 0.000e+00 m | True | 1.413e-26 w |
| 100.0 mrad | 0.000e+00 m | True | 1.413e-26 w |

Two facts in one table.  The correction is identically zero (zero sag), and
**the shipped screen model has NO angular error on a plate at all** -- 1.4e-26
waves, an independent eikonal-level confirmation of the refutation's
field-level plate control.  The same property is pinned on the LIBRARY in
`test_plane_plate_correction_is_exactly_zero` with `np.array_equal`.

### 3.2 (b) A single spherical surface -- the closed form vs exact rays

`screen_obliquity_derive.py sphere`, 3 mm pupil, error in waves rms:

| glass | R (mm) | sag(3mm) | tilt | SHIPPED | (3) exact-Snell | walk form | quadratic | gain |
|---|---|---|---|---|---|---|---|---|
| N-BK7 | +50 | 90.1 um | 10.0 mrad | 0.00166 | **0.000002** | 0.000006 | 0.00163 | 891x |
| N-BK7 | +50 | 90.1 um | 30.0 mrad | 0.00573 | **0.000007** | 0.000025 | 0.00488 | 810x |
| N-BK7 | +50 | 90.1 um | 54.9 mrad | 0.01345 | **0.000018** | 0.000067 | 0.00893 | 736x |
| N-BK7 | +50 | 90.1 um | 100.0 mrad | 0.03730 | **0.000055** | 0.000201 | 0.01635 | 683x |
| N-BK7 | +25 | 180.7 um | 54.9 mrad | 0.04144 | **0.000201** | 0.000723 | 0.03602 | 206x |
| N-BK7 | +25 | 180.7 um | 100.0 mrad | 0.09492 | **0.000503** | 0.001843 | 0.06592 | 189x |
| N-SF11 | +50 | 90.1 um | 54.9 mrad | 0.01718 | **0.000028** | 0.000092 | 0.01141 | 615x |
| N-SF11 | +25 | 180.7 um | 54.9 mrad | 0.05290 | **0.000341** | 0.001129 | 0.04595 | 155x |
| N-SF11 | +25 | 180.7 um | 100.0 mrad | 0.12120 | **0.000776** | 0.002576 | 0.08404 | 156x |
| N-BK7 | -50 | -90.1 um | 54.9 mrad | 0.01346 | **0.000019** | 0.000067 | 0.00893 | 727x |
| N-BK7 | -50 | -90.1 um | 100.0 mrad | 0.03732 | **0.000055** | 0.000201 | 0.01635 | 676x |

Three readings.  **(a) The sign is right at both signs of the sag** (the
`R = -50` rows mirror the `+50` rows).  **(b) The QUADRATIC form is a poor
approximation** -- it recovers only 55-65 % of a term the exact form removes to
0.1 %.  **(c) The gain degrades with the facet SLOPE, not with the angle**:
891x -> 683x across a 10x angle range at `sigma = 0.06`, but 891x -> 225x
going from `sigma = 0.06` to 0.12 at fixed angle.  That is R1, which is
proportional to the slope.

A ladder in the facet radius x the following glass thickness pins that reading
directly (single N-SK2-class curved facet + a flat exit face, 54.9 mrad, 3 mm
pupil; the entries are the gain `shipped / corrected`):

| R (mm) | sigma | sag(3mm) | t = 2 mm | 4 mm | 8.6 mm | 13.6 mm |
|---|---|---|---|---|---|---|
| 50.0 | 0.060 | 90.1 um | 682x | 686x | 693x | 701x |
| 30.0 | 0.101 | 150.4 um | 257x | 259x | 264x | 270x |
| 19.6 | 0.155 | 231.0 um | 112x | 114x | 118x | 122x |
| 12.6 | 0.245 | 362.4 um | 46x | 47x | 50x | 54x |

The gain moves by 4-17 % across a 7x range of following glass thickness and
by 15x across a 4x range of facet slope.  **The residual is a property of the
SLOPE, not of the propagation distance** -- which is the signature of R1 (the
kick error is set at the facet; the gap only converts it into wavefront) and
rules out any leftover gap-obliquity term, which would scale with `t`.

### 3.3 (c) Design 121, all six post-DOE groups

`screen_obliquity_derive.py d121`, each group at its own extreme-order
carrier, waves rms:

| group | prescription | tilt | pupil | SHIPPED | corrected (3) | quadratic | gain |
|---|---|---|---|---|---|---|---|
| 0 | plate N-SF1 | 51.50 mrad | 3 mm | **0.00000** | 0.00000 | 0.00000 | -- |
| 1 | plate N-BK7 | 51.50 mrad | 3 mm | **0.00000** | 0.00000 | 0.00000 | -- |
| 2 | doublet PK52A/SF57 | 51.50 mrad | 3 mm | 0.01680 | **0.000552** | 0.01304 | 30x |
| 3 | singlet LAK8 | 46.69 mrad | 3 mm | 0.00991 | **0.000020** | 0.00663 | 496x |
| 4 | singlet LAK9 | 7.38 mrad | 3 mm | 0.00049 | **0.000088** | 0.00074 | 5.6x |
| **5** | **doublet SK2/SF57** | **54.87 mrad** | **3 mm** | **0.25848** | **0.090692** | 0.29321 | **2.9x** |
| 5 | doublet SK2/SF57 | 54.87 mrad | 2 mm | 0.10449 | **0.032784** | 0.11653 | 3.2x |
| 5 | doublet SK2/SF57 | 54.87 mrad | 1 mm | 0.02459 | **0.006971** | 0.02703 | 3.5x |

**The plates return exact zeros on BOTH arms** -- the refutation's central
claim, at the eikonal level, on the design's own prescriptions.

**AND THE 0.2118 IS RECOVERED.**  The refutation's S7 residual column, which
it argued is "the same order" as the screen's own angular error, versus the
screen's angular error measured here directly:

| pupil | S7 entrance-referenced residual | screen angular error (this build) | ratio |
|---|---|---|---|
| 1 mm | 0.0200 w | 0.02459 w | 1.2295 |
| 2 mm | 0.0849 w | 0.10449 w | 1.2307 |
| 3 mm | 0.2118 w | 0.25848 w | 1.2204 |

**A constant 1.227 +- 0.005 across a 3x pupil range** -- the two quantities are
the same function of the pupil to within a scale factor, which is what "the
same order, and it IS the documented `sag * theta^2` bound" had to mean if it
was right.  It was.  (The factor is not 1 and should not be: the S7 column is
the residual of a DIFFERENT decomposition -- the entrance-referenced OPL
differential, which per S2 of the refutation is the exit-referenced one with
the gap obliquity flipped -- so it inherits the same pupil dependence with a
different weight.)

### 3.4 The remaining floor is a DEFLECTION defect, not a sag defect

Group 5's 2.9x is the campaign's worst rung, so it was decomposed.  A
per-surface ablation of the correction:

| corrected at | error (waves rms) |
|---|---|
| nothing (shipped) | 0.25848 |
| surface 0 only | 0.30899 |
| surface 1 only | 0.26578 |
| surface 2 only | **0.04233** |
| surfaces 1, 2 | 0.04559 |
| surfaces 0, 1, 2 | 0.09069 |

Correcting surface 0 ALONE makes it worse, which looks like a broken closed
form -- and is not.  A **tangent-facet arm** settles it: replace each screen's
momentum kick by exact vector refraction at the local tangent plane placed on
the vertex plane, keep the identity's OPD, keep homogeneous gaps:

| arm | group 2 | group 3 | group 5 |
|---|---|---|---|
| shipped screen | 0.01680 | 0.00991 | 0.25848 |
| screen + correction (4) | 0.00055 | 0.00002 | 0.09069 |
| tangent facet, NO translation OPD | 0.01644 | 0.01001 | 0.29119 |
| **tangent facet + translation OPD (2)** | **0.000020** | **0.000016** | **0.000372** |

**695x on group 5.**  The identity is exact; what the SCREEN cannot do is
deflect correctly.  Its kick `-(n2-n1) grad sag` is right to `O(sigma^3)` at
normal incidence (that is the model's known ceiling) but is independent of
incidence, so its angular part is a genuine second defect.  It is not a
per-surface OPD at all -- it enters only by propagating a wrong direction
through the following gap -- and a screen whose value is `f(x,y) sag(x,y)`
cannot supply it, because the required momentum field is not in general the
gradient of any scalar.  **Fixing it is a different object (the
`surface_model='displaced'` / transverse-walk-remap axis), and it is what
bounds this feature at 2.9x on the fastest elements.**

### 3.5 (d) THE REFERENCE-PLANE DISCIPLINE -- the exit-plane field arm

The refutation's own `mode_field` control: two tilts, wavefront gap against
`apply_real_lens_traced` at the exit plane, `D(theta) - D(0)` with piston and
tilt removed.  Reproduced with the SAME order pick (its scoring compares
`hypot(y_chief, L)` rather than `hypot(L, M)` and lands on 50.52 mrad -- kept
verbatim so the bar is apples-to-apples) on the refutation's OWN branch base
`b67130a`:

| arm | raw vs traced @ 0 mrad | raw @ 50.52 mrad | ANGULAR rms | max |
|---|---|---|---|---|
| shipped screen | 0.2895 w | 0.2351 w | **0.37447 w** | 1.09212 w |
| the REFUTED correction | 0.2895 w | 0.2879 w | 0.40688 w | 0.99377 w |
| **this correction** | 0.2895 w | 0.2334 w | **0.37294 w** | 1.04990 w |

**0.37294 < 0.37447.**  The bar is cleared, and the blind column is
reproduced to every printed digit, so this is the same measurement the
refutation made and not a re-scaled one.

On `main` (`755ad99`), which lacks the two traced/carrier fixes on
`feat/angle-aware-lens`, the same configuration reads blind **0.38420** ->
corrected **0.37743** (1.018x), and at the corrected 54.87 mrad order
**0.37859** -> **0.37795**.  **Corrected is below blind on every base and
every order pick tried.**

But the recovery factor is 1.004-1.018x, i.e. the arm is nearly blind to the
feature -- exactly S4.4's finding, for exactly its reasons: the
analytic-vs-traced floor at normal incidence is 0.29 waves rms, an order of
magnitude larger than the whole term, and the exit wavefront is 13x
under-sampled at `dx = 12 um` (NA_exit 0.73 needs 0.9 um).

### 3.6 The same control, on a case it CAN resolve

`screen_obliquity_derive.py fieldb`.  Identical common-mode control, on a
NYQUIST-SAMPLED fast singlet (R = 19.6 mm N-SSK2, 4 mm of glass, 3 mm
aperture, N = 1536, `dx = 4 um`, 0.7 mm probe disc).  Model-vs-model floor at
normal incidence: **0.00159 waves rms**, 180x lower than d121's.

| tilt | blind | corrected | recovery |
|---|---|---|---|
| 20 mrad | 0.00156 w | 0.00148 w | 1.05x |
| 50 mrad | 0.00206 w | 0.00149 w | 1.38x |
| **100 mrad** | **0.00448 w** | **0.00157 w** | **2.86x** |
| 150 mrad | 0.23859 w | 0.23246 w | 1.03x |

At 100 mrad the corrected arm lands ON the floor (0.00157 vs 0.00159), and the
blind arm's excess over the floor, `sqrt(0.00448^2 - 0.00159^2) = 0.00419 w`,
is within 25 % of the eikonal arm's independent 0.00543 w for the same
configuration.  **Two arms with different oracles, agreeing on the size of the
term.**

The 150 mrad rung is the ARM breaking, not the correction: the eikonal arm on
the identical configuration reads 0.012032 -> 0.000007 w (1814x) there.  At
that tilt the traced oracle and the grid are both past their envelope, and
both arms degrade together.  Recorded rather than dropped.

---

## 4. THE GUARD

The same closed form, read as an estimator.  With a `carrier` supplied,
`apply_real_lens` accumulates `sum_i dOPD_i` and scores its
piston-and-tilt-free rms over the pupil disc (the declared aperture, else the
widest semi-diameter, else the grid's inscribed radius), in waves.  **That IS
the wavefront error the angle-blind screens carry at those ray angles** -- it
is not a proxy.  The 3x3 fit is done through normal equations on scaled
coordinates, so the estimator costs three grid reductions, not a dense solve.

* Tolerance **0.05 waves = lambda/20** (`_SCREEN_OBLIQUITY_TOL_WAVES`).  Below
  it, the term is inside the analytic model's own normal-incidence ceiling on
  every element measured in this campaign.
* With the correction APPLIED the guard scores **40 %** of the estimate
  (`_SCREEN_OBLIQUITY_RESIDUAL_FRAC`), the budgeted R1 leftover.  The worst
  measured ratio (residual / uncorrected) in the campaign is **0.351** on d121
  group 5; the other powered groups are 0.002 (g3) and 0.033 (g2), the single
  facets 0.001-0.007, and g4 reads 0.18 on a 0.0005-wave error so small the
  ratio is not meaningful.  Rounded UP from the worst, not fitted to the set.
* `'warn'` (default) emits a `RuntimeWarning` naming the estimate, the
  tolerance, whether the correction was applied, the leftover, and
  `apply_real_lens_traced`.  `'error'` raises `ValueError`.  `'silent'` skips
  the estimator entirely.
* **Carrier-free calls never reach it.**  There is no angle to estimate
  against, and the estimator would be identically zero.

Measured on design 121's own six groups (`screen_obliquity_derive.py guard`,
6 mm pupil, N = 512, `dx = 12 um`):

| group | tilt | estimate | fires, correction OFF | fires, correction ON |
|---|---|---|---|---|
| 0 plate N-SF1 | 51.50 mrad | <= 0.05 w | no | no |
| 1 plate N-BK7 | 51.50 mrad | <= 0.05 w | no | no |
| 2 doublet PK52A/SF57 | 51.50 mrad | <= 0.05 w | no | no |
| 3 singlet LAK8 | 46.69 mrad | <= 0.05 w | no | no |
| 4 singlet LAK9 | 7.38 mrad | <= 0.05 w | no | no |
| **5 doublet SK2/SF57** | **54.87 mrad** | **0.23910 w** | **YES** | **YES** |

**One group of six, and it is the one the campaign has been pointing at since
the proto.**  The estimate 0.23910 against the exact-ray truth 0.25848 is
7.5 % low -- the estimator is the leading-order size of the defect, not a
bound on it, and the docstring says so.

---

## 5. THE DEFAULT DECISION

The brief's rule: default ON only if the exit-plane common-mode control shows
improvement AND the carrier-free byte-null holds.

| condition | evidence | verdict |
|---|---|---|
| exit-plane common-mode improvement | S3.5: 0.37447 -> 0.37294 on the refutation's own base and configuration; 0.38420 -> 0.37743 on `main`; 0.00448 -> 0.00157 (2.86x) on the resolvable arm of S3.6 | **MET** |
| carrier-free byte-null | `np.array_equal` on the returned complex arrays for `carrier=None`, `screen_obliquity=False`, `on_screen_obliquity='error'` and a ZERO-ANGLE carrier (`test_carrier_free_call_is_byte_identical`, `test_zero_angle_carrier_is_byte_identical`), AND cross-tree against a pristine v5.34.0 checkout on 10/10 shipped option combinations (S7) | **MET, and stronger** |
| the closed form is right | S2 identity + S3.1-S3.3: exactly zero on a plate, 154-891x on single facets, 30-496x on three of design 121's four powered groups | **MET** |

**DEFAULT: ON when a `carrier` is supplied** (`screen_obliquity='auto'`), OFF
otherwise.  Since `carrier=` is a NEW keyword, "default on" cannot change any
existing call: a pre-5.35 call site has no way to reach the correction at all.
The byte-null is therefore a STRUCTURAL property of the surface, measured
anyway.

Two honesty notes attached to that decision:

* the field-arm improvement on d121 is 1.004-1.018x, which alone would not
  justify anything.  What justifies it is the exact-ray arm, where the term is
  resolvable and the correction removes 65-99.9 % of it;
* on the fastest elements the corrected screen is still 0.09 waves from truth
  because of R1.  The guard says so, in waves, on exactly those calls.

---

## 6. COST AND SCOPE

* **Cost, measured.**  Per POWERED surface: one `gradient`, one
  `_screen_obliquity_delta` (two `_facet_axial_momenta` sharing one `nu_z`),
  two momentum updates -- ~20 full-grid float ops.  Flat faces are skipped by
  a single `any(sag)` reduction, so plates and plano faces cost nothing.  No
  ray trace, no fit, no cache, no I/O -- but not free either: on a
  three-surface cemented element the carriered call is **2.19x / 2.86x /
  3.61x** the carrier-free wall clock at N = 512 / 1024 / 2048 (best of 5,
  guard on).  Quoted in the docstring rather than glossed.
* **Path.**  The correction needs a sag GRADIENT, which the row-banded sag
  path does not carry a halo for, so `_obl_active` routes the surface loop to
  the whole-grid path.  Peak memory is that path's, plus `p0x`/`p0y`/`total`
  (and `qx`/`qy` only for a non-collimated carrier -- a collimated
  `TiltedCarrier` collapses to two floats analytically, without building the
  three full-grid arrays `_compute_carrier` would).  Documented in the
  `screen_obliquity` docstring.
* **Refusals, not degradations.**  `screen_obliquity=True` without a carrier
  raises (there is no angle).  `carrier=` with `surface_model='displaced'`
  raises: that path is ALREADY angle-aware through `conjugate=`, modifying the
  same per-surface sag OPD with true ray cosines, so stacking the two would
  double-count.  Where either arm's refraction is non-propagating (evanescent
  input momentum, or TIR at the facet) the correction is DROPPED on those
  pixels rather than clamped -- a clamped cosine is a wrong OPD and the
  shipped screen is the safe neutral.
* **Not forwarded through delegate paths.**  Consistent with the refutation's
  S5 argument: `on_noncollimated='delegate'` fires precisely when the carrier
  has been measured NOT to describe the field.

---

## 7. THE COMPATIBILITY / ADJUDICATION LEDGER

| item | status | evidence |
|---|---|---|
| carrier-free byte-null | **HELD**, four ways IN-TREE | `test_carrier_free_call_is_byte_identical` (`carrier=None`, `screen_obliquity=False`, `on_screen_obliquity='error'`, defaults), `np.array_equal` on the complex arrays |
| carrier-free byte-null CROSS-TREE vs a pristine v5.34.0 checkout | **HELD on 10/10 option combinations** | a conic+asphere singlet (both faces powered, both aspheric) run in `C:/tmp/lum_base` (detached @ `755ad99`) and in the feature worktree, `np.array_equal` on `.view(np.uint8)`: `default`, `slant_correction`, `fresnel`, both, `sag_chunk_rows=32`, `seidel_correction`, `surface_model='displaced'`, `displaced`+`conjugate=0.5`, complex64 input, `sag_dtype=float32` |
| zero-angle-carrier byte-null | **HELD** | `test_zero_angle_carrier_is_byte_identical` |
| plane-plate byte-null WITH a carrier | **HELD at 5 tilts to 100 mrad** | `test_plane_plate_correction_is_exactly_zero` |
| moved test pins | **ZERO** | the feature is reachable only through a keyword that did not exist in v5.34.0; no pre-existing call site can reach it |
| relaxed / retargeted assertions | **ZERO** | no existing test file edited |
| `xfail` / `skip` added | **ZERO** | -- |
| `on_noncollimated='delegate'` (E-L22) | **UNTOUCHED** | no edit to `_lens_traced.py` |
| `prepare_real_lens` / `PreparedAnalyticLens` | **UNTOUCHED** | no new kwarg; the prepared path is input-independent by construction and a carrier is not |
| design-121 acceptance banner | **CANNOT HAVE MOVED** | every d121 consumer call is carrier-free |
| `CHANGELOG` | **NOT TOUCHED** | as instructed |

---

## 8. SUITES

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X, 24 logical CPUs
python 3.14.6   numpy 2.4.4      lumenairy 5.34.0 (worktree C:/tmp/lum_obl,
                                 branch feat/screen-obliquity off main 755ad99)
```

| gate | result |
|---|---|
| consumer set A+B (Windows) -- 33 files: every `tests/unit` file carrying a real `apply_real_lens(` call site, plus `test_niche_c13` / `test_niche_c14` / `test_audit_lens`, INCLUDING the 28 new tests | **1119 passed** in 959.8 s |
| consumer set C (Windows) -- the three heavy files (`test_audit_lens_models_2026_07`, `test_fga`, `test_lens_gbd`) | **101 passed** in 1120.1 s |
| WSL / OpenBLAS proxy (`~/lumvenv`, BLAS pinned to 1 thread), 10 files: the new file + chunked-sag, slant byte-identity, prepared/enums, c13, c14, hammer H1 / H2, G2 displaced, P3 pointwise | **209 passed** in 739.2 s |
| WSL re-verify on the FINAL library (the 5 byte-identity-critical files incl. the 28-test new file) | **160 passed** in 204.2 s |
| `ruff check lumenairy/ tests/unit/` + the new validation file, Windows AND WSL | **All checks passed** |

**1220 tests on Windows + 369 on WSL, 0 failed, 0 skipped by this work, no
`xfail` added.**
The consumer set was surveyed rather than assumed: `grep -rl 'apply_real_lens('
tests/unit/` finds 33 files, plus `test_niche_c13` / `test_niche_c14` /
`test_audit_lens` added by hand = 36; three are the heavy set C and the other
33 are set A+B.  The survey deliberately keeps `test_audit_optimize.py`'s
`capturing_apply_real_lens` stubs, which REPLACE the function -- run anyway,
because they pin the CALL-SITE contract the three new keywords extend.

Reproducing the study:

```
python screen_obliquity_derive.py plate    # S3.1, the plate zero
python screen_obliquity_derive.py sphere   # S3.2, the closed-form table
python screen_obliquity_derive.py d121     # S3.3, all six groups
python screen_obliquity_derive.py ablate   # S3.4, per-surface + tangent facet
python screen_obliquity_derive.py field    # S3.5, the refutation's own arm
python screen_obliquity_derive.py fieldb   # S3.6, the resolvable arm
python screen_obliquity_derive.py guard    # S4, the estimator on d121
```

The S3.5 bar was measured on the refutation's OWN base, which is a
throwaway worktree plus two file copies:

```
git worktree add --detach C:/tmp/lum_obl_b b67130a
cp <feature>/lumenairy/elements/_lens_real.py            C:/tmp/lum_obl_b/lumenairy/elements/
cp <feature>/validation/.../screen_obliquity_derive.py   C:/tmp/lum_obl_b/validation/repro_traced_carrier_121/
cp <d8c4080>/validation/.../hmap_consumer2_121.py hmap_screen_proto.py  (same dir)
cd C:/tmp/lum_obl_b/validation/repro_traced_carrier_121
python -c "import screen_obliquity_derive as S; S.mode_field(select='refutation')"
```

(`_lens_real.py` is byte-identical between `main` and `b67130a`, so dropping
the feature file in is the whole port; the three commits that differ are all
in `_lens_traced.py` / `carrier.py` / `_lens_imap.py`, i.e. in the ORACLE.)

Results of record are `_screen_obl_all.json` (and the per-mode
`_screen_obl_<mode>.json` when a mode is run alone).  Every mode is ray tracing plus
short analytic-element calls except `fieldb` (three N=1536 traced calls), so
the whole study is under two minutes.

---

## 9. WHAT IS NOT CLAIMED

* **The deflection defect R1 is not fixed, and cannot be by this object.**
  It is 0.09 waves on design 121's group 5 -- 35 % of that element's angular
  error -- and it is the reason the corrected screen is 2.9x rather than 700x
  there.  S3.4 measures it, names it, and shows a tangent-facet model that
  does remove it; building that is the `surface_model='displaced'` axis, not
  this one.
* **The estimator is a leading-order SIZE, not a bound.**  It reads 7.5 % low
  against the exact-ray truth on the one case where both were measured, and
  its pupil is the declared aperture, so it over-reads when the beam
  underfills.
* **The field arm on design 121 cannot resolve the feature** and is reported
  only because it is the specified bar.  S4.4 of the refutation is right; the
  decisive controls here are the exact-ray eikonal arms and the plate.
* **Nothing is measured on the GPU path.**  The code is `xp`-generic and
  `cupy` has `gradient`, but no CuPy run was made.
* **The `p0` grid-local approximation (R2) is measured on d121 group 5 only**
  (0.090692 vs 0.091586 against the true model momentum, < 1 %).  It is not
  measured on a strongly walking element.
* **Only ROTATIONALLY SYMMETRIC surfaces were measured.**  The correction
  takes `grad sag` from `xp.gradient(sag, ...)` AFTER the decenter shift, the
  tilt ramp, the form-error map and any freeform departure have been folded
  into `sag`, so it is structurally correct for those -- but no oracle run was
  made on a decentred, tilted, biconic or freeform element.  The CONIC and
  ASPHERIC axes ARE covered (R = 25 mm N-BK7, 54.9 mrad, 3 mm pupil):
  sphere 206x, parabola 210x, hyperbola (k = -2.5) 215x, oblate (k = +0.8)
  203x, conic+asphere (k = -0.4, A4 = 2e4, A6 = 5e6) 199x -- the gain tracks
  the facet SLOPE and is otherwise insensitive to the surface family, as S3.2
  predicts.
* **`JonesField.apply_real_lens` does not expose the new keywords.**  Its
  signature is explicit (no `**kwargs`), so the polarized wrapper keeps the
  pre-5.35 surface exactly.  Widening it is a one-line change that was NOT
  made, because the per-component dispatch already carries its own v5.4.6
  caveat and a carrier would need its own argument there.
* **`prepare_real_lens` does not support the correction.**  A prepared lens
  caches input-independent screens; the correction is carrier-dependent.  It
  could be prepared per-carrier; it was not.
* **The `'auto'` and explicit-wavefront carriers are exercised but not
  measured against an oracle** (`test_other_carrier_vocabularies_run_and_...`,
  `test_explicit_wavefront_carrier_matches_the_equivalent_tilt`).  Every
  quantitative number in this note uses a collimated `TiltedCarrier`.

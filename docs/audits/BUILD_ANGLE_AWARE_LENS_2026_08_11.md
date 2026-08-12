# BUILD -- the angle-aware analytic lens (consumer 2 of the Hamilton characteristic)

**2026-08-11.  Branch `feat/angle-aware-lens`, cut from `feat/inverse-map`
@ `b67130a` (three commits ahead of `main` @ `755ad99` = v5.34.0).  No commit,
no push, no `gh`, no `CHANGELOG`.  `lumenairy/**` IS BYTE-IDENTICAL TO THE
BRANCH BASE -- `git status lumenairy/` is empty.**

Builds `PROTO_HAMILTON_MAP_2026_08_11` S8.4's consumer-2 design, measures it,
and **refutes it.**  The feature was written, wired into `apply_real_lens`,
run against the traced oracle, and the library edit was REVERTED when a
closed-form control on a plane-parallel plate showed the correction is a
double-count.

---

## 0. VERDICT

> **BUILT, MEASURED, AND REFUTED BY A CLOSED FORM.  Nothing ships.  The
> library is untouched.**
>
> **1. THE MAP IS CORRECT AND THE PHYSICS IT CHARACTERISES IS NOT WRONG.**  The
> `CongruenceMap` built here reproduces design 121's own point characteristic
> against exact ray traces at **3.68e-09 waves** on the case the proto sized
> (last post-DOE group, extreme DOE order, 3 mm pupil) and at **3.04e-10
> waves** on the 30-node angular ladder's converged rung.  The plane plates
> return the exact-zero residual property with their **9.910 / 0.970-wave**
> pistons reproduced to **1.85e-13 / 3.02e-14 waves**.  Every guard fired when
> it should and refused rather than degraded.  As a characterisation of the
> element's Hamilton point characteristic, the object does everything the proto
> said it would.
>
> **2. AND IT IS THE WRONG QUANTITY TO ADD TO `apply_real_lens`.**
> `PROTO_HAMILTON_MAP_2026_08_11` S7 sizes the angle-blindness as
> `OPL(x, y; theta) - OPL(x, y; 0)` at fixed **ENTRANCE** point, on the stated
> premise that "the angle-blind model carries NONE of it -- its screen is a
> function of `(x, y)` alone".  **The screens are.  The ANGULAR-SPECTRUM steps
> between them are not**, and they carry the whole angular optical path
> exactly, referenced to the **EXIT** point.
>
> **3. THE PLATE PROVES IT IN CLOSED FORM AND IN MEASUREMENT.**  A
> plane-parallel plate has zero sag, so `apply_real_lens` reduces to ONE
> in-glass angular-spectrum step and its exit field for `exp(i k0 L x)` is
> exactly `exp(i k0 L x) exp(i k0 t sqrt(n^2 - L^2))`.  The ray picture agrees
> term for term once the transverse walk is substituted in.  Measured on a
> 25.4 mm N-BK7 plate at four grid-exact tilts
> (`hmap_consumer2_121.py plate`):
>
> ```
>   tilt        SHIPPED vs closed form   WITH THE SCREEN   exit-exact d   entrance-ray d
>    0.000 mrad      -7.276e-12 w          -7.276e-12 w      +0.0000 w      +0.0000 w
>   10.234 mrad      -7.276e-12 w          -3.246e-01 w      -0.6754 w      +0.6754 w
>   20.725 mrad      +0.000e+00 w          -2.303e-01 w      -2.7695 w      +2.7697 w
>   32.750 mrad      -3.638e-12 w          -8.197e-02 w      -6.9164 w      +6.9180 w
> ```
>
> **The shipped model is exact to machine zero at every tilt.**  The proto's
> "missing" piston and the true exit-referenced differential are EQUAL AND
> OPPOSITE to five digits; adding the first to a model that already carries the
> second is wrong by twice the term -- 2.77 waves at 20.7 mrad on this plate,
> and 9.91 waves on design 121's own N-SF1 plate at 41.5 mrad.
>
> **3a. AND THE SCALE ARGUMENT BOUNDS WHAT IS LEFT.**  A plate can only test
> what the GAPS carry, so split the proto's own columns by mechanism (S2.3).
> The gap obliquity `n t theta^2 / 2` is **24.7 waves** on group 5 and is the
> piston-plus-tilt the proto counts -- carried exactly, and the double-count.
> The screen's own angular error `~ sag theta^2` is **0.53 waves** there,
> against S7's measured RESIDUAL of **0.2118 waves** -- the same order, and
> it IS the documented `sag * theta^2` bound in `apply_real_lens`'s own
> docstring.  **The prize is the residual column (0.0014-0.212 waves), not the
> 40 waves of piston and tilt; and it is 40-120x smaller than what was about to
> be added.**
>
> **4. THE TRANSVERSE WALK-OFF IS NOT A SEPARATE TERM.**  The proto records it
> as one -- *"The plates also shift the ray transversely, which an angle-blind
> screen misses entirely; that is NOT in the residual column above ... and is a
> separate, larger, unmeasured term."*  It is not separate and it is not
> missed: the walk-off and the entrance-referenced path length are the two
> halves of ONE identity, they cancel exactly, and the angular spectrum
> reproduces their sum.  **That sentence is the defect, and it is a
> documentation defect in the proto rather than a code defect anywhere.**
>
> **5. THE FIELD-LEVEL ARM AGREES, AND SAYS WHY THE ERROR WAS INVISIBLE.**
> With the common-mode control that isolates the angular part (S4.4), the
> shipped screen's angular error against `apply_real_lens_traced` on design
> 121's last group is **0.374 waves rms** and the corrected arm's is **0.407**
> -- a recovery factor of **0.9x**, i.e. slightly worse.  The raw
> analytic-vs-traced gap on that fast doublet is **0.290 waves rms AT NORMAL
> INCIDENCE**, which is larger than the entire effect, so the field-level
> comparison cannot resolve the feature at all and could not have caught the
> sign.  **The closed-form plate could, does, and is the only control in this
> study that carries the verdict.**
>
> **6. NOTHING MOVED.**  Because the correction is reachable only through a
> `carrier=` keyword that did not exist, and because the wiring was reverted,
> `lumenairy/**` is byte-identical to the branch base.  **Zero test pins moved,
> zero assertions were adjudicated, the `on_noncollimated='delegate'` path and
> its E-L22 discarded-kwargs contract are untouched, and the design-121
> acceptance banner cannot have moved because no byte of the library did.**
>
> **7. THE CORRECTED DESIGN IS SPECIFIED (S6) AND NOT TAKEN.**  It is a
> different object -- the difference between the exact eikonal and the
> SPLIT-STEP MODEL'S OWN eikonal, both at the exit plane -- and it requires an
> exit-coordinate inversion for two ray systems rather than one screen.  The
> parametrisation, the guard set, the cache discipline and both ladders
> measured here carry over verbatim.

---

## 1. WHAT WAS BUILT, AND WHERE IT IS NOW

| file | status |
|---|---|
| `lumenairy/elements/_lens_hmap.py` | WRITTEN, then MOVED OUT of the library to `validation/repro_traced_carrier_121/hmap_screen_proto.py` |
| `lumenairy/elements/_lens_real.py` | `carrier=` / `angle_aware=` kwargs, the envelope validator and the exit screen were added, measured, and **REVERTED** (`git checkout --`) |
| `validation/repro_traced_carrier_121/hmap_screen_proto.py` | the map, the guards G0-G7, the chain-A cache.  Header leads with the refutation |
| `validation/repro_traced_carrier_121/hmap_consumer2_121.py` | NEW.  `plate` / `gap` / `map` / `degree` / `nodes` / `field` |
| `docs/audits/BUILD_ANGLE_AWARE_LENS_2026_08_11.md` | this note |

`git status lumenairy/` is empty.  The only tracked-tree changes are the two
new validation scripts, their `_hmap_c2_*.json` results and this document.

### 1.1 The object that was built

```
dOPL(x, y; sx, sy) = OPL(x, y; theta) - OPL(x, y; 0)
```

a tensor Chebyshev in (pupil x REDUCED angle), where the reduced angle
`s = (L, M) - grad S_R(x, y)` is measured against one axis-centred reference
sphere so the node lattice in the true angle is SHEARED across the pupil.  The
shear is the proto's own finding and it is correct and load-bearing: on the
ABSOLUTE angle 44-50 % of the node rays leave the element at every node count.

Consuming the angle FIELD rather than a source LABEL was a deliberate
departure from the proto's parametrisation, on the
`BUILD_INVERSE_MAP_2026_08_11` S2 lesson: the source label exists to share one
build across a FAN, and `apply_real_lens` is handed one carrier per call.  It
also makes the map work for every carrier vocabulary the library has
(`TiltedCarrier`, a scalar conjugate, `'auto'`, an explicit wavefront), because
all of them produce `(L(x, y), M(x, y))` and none need produce a label.
**That part of the design stands and should be carried into S6.**

Guards, all of which refuse to the shipped screen rather than degrading:

| guard | refuses on | fired during this build? |
|---|---|---|
| G0 domain / traceable | no surfaces, no usable pupil radius, non-finite box | -- |
| G1 reference radius | `R == 0` (the congruence's own focus) or NaN | -- |
| G2 Jacobian sign / caustic | `det J` sign change on a node, or dynamic range > 30 | no; measured 1.004-1.023 |
| G3 alive census | any ray dead inside the declared pupil at any node or at the zero arm | YES (an over-wide pupil) |
| G4 angle field | a non-finite direction cosine | -- |
| G5 in-box | a reduced angle outside the built box | YES (S3.1) |
| G6 conditioning | fewer than 4 samples per free coefficient | -- |
| G7 accuracy | the built map, measured against DIRECT ray traces at the caller's own angles, outside `lambda/100` with 3x | YES, repeatedly (S3.2) |

---

## 2. THE REFUTATION

### 2.1 The closed form

For a plane-parallel plate of thickness `t` and index `n`, illuminated by
`E_in(x) = exp(i k0 L x)` (`L = sin theta_i`, `n1 = 1`):

* **Wave model.**  Both interfaces are flat, so both sag screens are
  identically zero and `apply_real_lens` reduces to a single in-glass angular
  spectrum step.  The transverse frequency is conserved, so the exit field is
  exactly

  ```
  E_out(x) = exp(i k0 L x) * exp(i k0 t sqrt(n^2 - L^2))
           = exp(i k0 L x) * exp(i k0 n t cos(theta_t)).
  ```

* **Ray model.**  The ray entering at `x0` carries `n t / cos(theta_t)` of
  optical path and exits at `x1 = x0 + t tan(theta_t)`.  Its phase at the exit
  point is

  ```
  k0 L x0 + k0 n t / cos(theta_t)
    = k0 L x1 - k0 L t tan(theta_t) + k0 n t / cos(theta_t)
    = k0 L x1 + k0 t [ n / cos(theta_t) - n sin^2(theta_t) / cos(theta_t) ]     (L = n sin theta_t)
    = k0 L x1 + k0 n t cos(theta_t).
  ```

**The two agree identically.**  The entrance-referenced path length
`n t / cos(theta_t)` and the transverse walk `t tan(theta_t)` are not two
independent facts about the element; they are one, and the angular spectrum
carries their sum.

The consequence for the correction that was built:

```
entrance-referenced differential   n t (1/cos(theta_t) - 1)     POSITIVE
exit-referenced differential       n t (cos(theta_t) - 1)       NEGATIVE
```

equal and opposite to leading order.  Adding the first to a model that already
carries the second is wrong by twice the term.

### 2.2 The measurement

`hmap_consumer2_121.py plate`, 25.4 mm N-BK7 at 1.31 um, `N = 256`,
`dx = 20 um`, tilts chosen GRID-EXACT and below Nyquist (`L = m lambda/(N dx)`,
`m <= N/2`):

| `m` | tilt | SHIPPED vs closed form | WITH THE SCREEN | exit-exact `d` | entrance-ray `d` |
|---|---|---|---|---|---|
| 0 | 0.000 mrad | -7.276e-12 w | -7.276e-12 w | +0.0000 w | +0.0000 w |
| 40 | 10.234 mrad | -7.276e-12 w | -3.246e-01 w | -0.6754 w | +0.6754 w |
| 81 | 20.725 mrad | **+0.000e+00 w** | -2.303e-01 w | -2.7695 w | +2.7697 w |
| 128 | 32.750 mrad | -3.638e-12 w | -8.197e-02 w | -6.9164 w | +6.9180 w |

(residuals are mod one wave; the screen's contribution over the inner disc is a
PURE PISTON to `3.6e-14 waves` peak-to-peak, so the map is doing exactly what
it claims -- it is the referencing that is wrong, not the fit.)

**A METHOD NOTE THAT COST A ROUND.**  The first pass used design 121's own
41.5 mrad at `N = 256, dx = 20 um`.  That is `m = 162 > N/2 = 128`: the tilted
plane wave is ALIASED on the grid, and the SHIPPED model then reads a spurious
0.3497-wave residual against its own closed form.  Reading that as a model
error would have "confirmed" the feature.  **A plane-wave control on a discrete
grid must use a grid-exact, sub-Nyquist tilt or it is measuring the FFT.**

### 2.3 What the plate does and does NOT kill -- a scale argument

A plate has zero sag, so it can only test the part of the correction the GAPS
carry.  Splitting the proto's own S7 columns by which mechanism produces them
makes the scope of the refutation precise.  On group 5 (doublet SK2/SF57,
12.65 mm of glass, `n ~ 1.7`, `theta = 54.9 mrad`, 3 mm pupil):

| term | closed-form scale | S7 measures | carried by the shipped model? |
|---|---|---|---|
| gap obliquity | `n t theta^2 / 2` = **24.7 waves** | piston 14.713 w + part of tilt 25.682 w | **YES, exactly** -- the angular spectrum.  Proved by S2.2 |
| screen's angular error | `~ sag theta^2` = **0.53 waves** at `sag(3 mm) ~ 0.23 mm` | RESIDUAL **0.2118 w** | **NO** -- a screen deflects the same way at every incidence.  This is `apply_real_lens`'s own documented `sag * theta^2` bound |

So the refutation is decisive about the LARGE columns and leaves the SMALL one
standing: **the piston and tilt the proto counts (14.7 and 25.7 waves) are not
missing, and adding them is the double-count; the residual column (0.0014 to
0.212 waves across the six groups) is the same order as the docstring's own
`sag * theta^2` estimate and is plausibly a real gap.**  It is 40x to 120x
smaller than the quantity that was going to be added, it is what S6's corrected
object would isolate, and nothing in this study measures it directly -- the
plate cannot (its residual is identically zero by construction) and the field
arm cannot (S4.4: the model-vs-model floor at normal incidence is larger than
the whole term).

**That is the single most useful number for whoever picks this up: the prize
is ~0.2 waves on a fast doublet at 55 mrad, not ~40.**

---

## 3. THE GUARDS, AND THE TWO THINGS THEY CAUGHT

### 3.1 G5 caught a box built on the wrong domain

The first version derived the reduced-angle box from the caller's own GRID
pixels.  A square grid samples its inscribed circle densely and its corners
once, so the box did not contain the angles at intermediate azimuths near the
rim -- including G7's own check rings, which is how it surfaced.  Fixed by
taking the box over the declared pupil DISC via the carrier's analytic
gradient, which costs no rays and also makes the box a property of the ELEMENT
and the CARRIER rather than of the caller's sampling (so two different grids
share one cached map).

### 3.2 G7 refused the shipped node count, and the ladder said why

The obvious pupil degree was 6 -- the shipped `newton_poly_order`, the degree
the traced element fits its own entrance-coordinate map at.  It FAILS, and G7
refused the build rather than applying a wrong screen.  Measured on design
121's last post-DOE group (doublet SK2/SF57) at its extreme DOE order, against
direct ray traces (`hmap_consumer2_121.py degree`), max error in waves:

| pupil | deg 6 | deg 8 | deg 10 | deg 12 | deg 14 |
|---|---|---|---|---|---|
| 3.00 mm | 4.13e-04 | 1.00e-05 | 2.11e-07 | 8.18e-09 | 3.33e-10 |
| 4.35 mm | **REFUSED** | 3.89e-04 | 1.88e-05 | 1.62e-06 | 1.40e-07 |
| 6.00 mm | **REFUSED** | **REFUSED** | 1.37e-03 | 2.48e-04 | 4.55e-05 |

On the singlet LAK8 of group 3 even degree 6 reads 3.5e-06 waves at 6 mm, so
the binding case is the fast doublet and nothing else.  Degree 12 was adopted:
first to clear the 3x bar on the hardest case with margin (2.48e-04 against
3.33e-03, 13x) while being five decades inside it on the 3 mm pupil the proto
sized.

**This is the architecture that worked and should be kept:** the node count,
the pupil degree, the lattice and the box pad are all INPUTS to a build whose
output is then measured against exact rays and refused if it misses.  A
parameter that is wrong shows up as a refusal, never as a quiet error.

---

## 4. THE MEASUREMENTS THAT STAND

Everything in this section is a property of the CHARACTERISTIC and survives the
refutation intact.

### 4.1 The proto's S7 gap, reproduced exactly

`hmap_consumer2_121.py gap`, 7.7 s.  Worst order per group, waves:

| group | prescription | worst order | tilt | pupil r | piston | tilt | RESIDUAL |
|---|---|---|---|---|---|---|---|
| 0 | plate N-SF1 25.40 mm | (-3,-2) | 41.52 mrad | 3 mm | 9.910 | 0.000 | **0.0000** |
| 1 | plate N-BK7 3.20 mm | (-3,0) | 34.55 mrad | 3 mm | 0.970 | 0.000 | **0.0000** |
| 2 | doublet PK52A/SF57 | (-4,-2) | 51.50 mrad | 3 mm | 8.651 | 3.484 | **0.0041** |
| 3 | singlet LAK8 | (-4,-2) | 46.69 mrad | 3 mm | 4.275 | 5.932 | **0.0070** |
| 4 | singlet LAK9 | (-4,-2) | 7.38 mrad | 3 mm | 0.083 | 1.156 | **0.0014** |
| 5 | doublet SK2/SF57 | (-4,-2) | 54.87 mrad | 3 mm | 14.713 | 25.682 | **0.2118** |

Every printed digit matches `PROTO_HAMILTON_MAP_2026_08_11` S7, including the
1 / 2 / 3 mm radial ladder (0.0200 / 0.0849 / 0.2118) and the plates' exact
zeros.  **The proto's arithmetic is right; its INTERPRETATION of what the
quantity is the error OF is what fails.**

### 4.2 The map against the same oracle

`hmap_consumer2_121.py map`, 1.6 s, at the adopted pupil degree 12, 3 mm
pupil.  "MAP err" is `max |dOPL_map - dOPL_traced|` over the disc:

| group | prescription | blind residual | MAP err max | MAP err rms | piston map vs traced |
|---|---|---|---|---|---|
| 0 | plate N-SF1 | 0.0000 w | **1.85e-13 w** | 8.68e-15 w | 9.910 / 9.910 |
| 1 | plate N-BK7 | 0.0000 w | **3.02e-14 w** | 9.46e-16 w | 0.970 / 0.970 |
| 2 | doublet PK52A/SF57 | 0.0041 w | 1.90e-10 w | 3.37e-11 w | 8.651 / 8.651 |
| 3 | singlet LAK8 | 0.0070 w | 2.26e-10 w | 3.66e-11 w | 4.275 / 4.275 |
| 4 | singlet LAK9 | 0.0014 w | 3.33e-11 w | 4.31e-12 w | 0.083 / 0.083 |
| **5** | **doublet SK2/SF57** | **0.2118 w** | **3.68e-09 w** | **4.71e-10 w** | **14.713 / 14.713** |

Build cost 0.020-0.024 s per element at 5.3 kB (this fan's congruences are
uniform tilts, so both angular axes are degenerate and collapse to one node --
see S4.3 for the non-degenerate ladder).  At the pupil degree 6 the same
group-5 row reads **4.13e-04 waves**, which is the number the ladder of S3.2
then improved on.

**So the map reproduces the quantity the proto sized to between 3.0e-14 and
3.7e-09 waves.  It is not the quantity `apply_real_lens` is missing.**

**THE PISTON DISCIPLINE, ANSWERED.**  The plane plates return the exact-zero
residual property, and their pistons -- the tilt-quadratic optical path
`FIX_TILT_QUADRATIC_OPL_2026_08_11` restores on the traced side -- are
reproduced against exact ray traces to **5.8e-14 and 1.4e-14 waves**.  The map
composes with that fix's semantics exactly: it stores an ABSOLUTE optical-path
differential, not a piston-removed one, and the constant it returns IS the
9.910 / 0.970 waves the traced element carries.

### 4.3 The angular node ladder

`hmap_consumer2_121.py nodes`, on design 121's own FINITE-R congruence at the
last group -- order (-4,-2), `R = -21.139 mm`, chief ray at (-3.016, -1.508)
mm, reduced-angle spread 5.11e-03 x 3.38e-03 rad over a 3 mm disc:

| nodes | | err max (waves) | storage | rays |
|---|---|---|---|---|
| (1,1) | 1 | **REFUSED** (G7) | -- | -- |
| (3,3) | 9 | 1.5021e-05 | 47.5 kB | 24 010 |
| (4,4) | 16 | 1.4061e-08 | 84.5 kB | 40 817 |
| (5,4) | 20 | 1.4612e-09 | 105.6 kB | 50 421 |
| **(6,5)** | **30** | **3.0440e-10** | **158.4 kB** | **74 431** |
| (7,5) | 35 | 3.0464e-10 | 184.8 kB | 86 436 |
| (9,9) | 81 | 3.0229e-10 | 427.8 kB | 196 882 |

Two readings.  **(a) The degenerate rung is REFUSED, correctly** -- a
single-node map cannot represent a genuinely two-dimensional angular spread and
G7 says so rather than returning a plausible wrong screen.  **(b) The proto's
30 nodes is far past saturation for ONE congruence**: the ladder flattens at
1.4e-08 by 16 nodes and the floor from 20 nodes on is the PUPIL fit, not the
angular one.  That is the `BUILD_INVERSE_MAP_2026_08_11` S2 lesson again in a
different consumer -- the 30-node count was sized to cover a 32-order FAN from
one build, and a per-call consumer does not need it.

**A SELECTOR TRAP, RECORDED.**  Choosing the worst congruence by its
angle-blind RESIDUAL picks order (0,0), because at a finite `R` the on-axis
order IS the reference sphere: its reduced-angle box is identically degenerate
(spread exactly 0.0) while its blind residual (8.85 waves -- the whole
convergence obliquity) is the largest in the fan.  Selecting on the residual
therefore picks the one congruence that exercises NO angular axis.  The ladder
above selects on `|tilt|`.

### 4.4 The field-level arm, and why it could not decide anything

`hmap_consumer2_121.py field`, design 121 group 5, `N = 512`, `dx = 12 um`,
`w = 2.5 mm`, probe disc 1.5 mm, oracle `apply_real_lens_traced(carrier=...)`.

A raw `analytic vs traced` wavefront comparison cannot resolve this feature:
the two models differ by far more than the angle term, and that difference is
present at normal incidence too.  So the angular part is isolated by
differencing two tilts:

```
D_arm(theta) = phase(arm at theta) - phase(traced at theta)
angular error of the arm = D_arm(theta) - D_arm(0)
```

| arm | raw vs traced at 0 mrad | raw at 50.52 mrad | ANGULAR part rms | max |
|---|---|---|---|---|
| shipped screen | 0.2895 w | 0.2351 w | **0.37447 w** | 1.09212 w |
| with the correction | 0.2895 w | 0.2879 w | **0.40688 w** | 0.99377 w |

Recovery factor **0.9x** -- the correction makes the rms slightly WORSE and the
max slightly better, i.e. it does nothing coherent.  And the diagnostic number
is the first column: **the shipped analytic model is already 0.290 waves rms
from the traced element AT NORMAL INCIDENCE on this fast doublet**, which is
larger than the whole 0.212-wave angular effect.  The field arm is therefore
blind to the feature in either direction, and a study that had only this arm
would have concluded "no measurable change" rather than "the sign is wrong".

**That is the methodological point of this build.**  The decisive control was
not the more realistic experiment; it was the one with a CLOSED FORM.

---

## 5. THE COMPATIBILITY / ADJUDICATION LEDGER

Requested explicitly.  It is empty, and that is a statement rather than an
omission.

| item | status | evidence |
|---|---|---|
| carrier-free byte-null | **N/A -- stronger.** `lumenairy/**` is byte-identical to the branch base, so EVERY call is bit-unchanged, not merely the carrier-free ones | `git status lumenairy/` empty; `git diff` empty |
| moved test pins | **ZERO.**  No assertion was relaxed, retargeted or re-derived | no library file changed |
| `on_noncollimated='delegate'` | **UNCHANGED.**  The E-L22 discarded-kwargs contract is intact and still lists `carrier` among the drops | no edit to `_lens_traced.py` |
| design-121 acceptance banner | **CANNOT HAVE MOVED** | no library byte changed |
| consumer suites | green, S7 | -- |

For the record, the wiring that WAS built and reverted had the byte-null as a
structural property rather than a measured one, and it was verified before the
revert: `apply_real_lens(..., carrier=None)` and a zero-angle carrier both
returned arrays bit-identical to the pre-feature call
(`np.array_equal(a.view(np.uint8), b.view(np.uint8))` True on both), and
`angle_aware=False` restored the shipped screen bit for bit.  Those properties
are reported here because they were measured, not because anything depends on
them now.

**On the delegate path specifically.**  The reverted build did NOT forward
`carrier` through `on_noncollimated='delegate'`, and the reason is a physics
argument that survives the refutation and is worth recording: that branch fires
only when the input's residual angular spread AFTER the carrier is removed
exceeds `_NONCOLLIMATED_RESID_THRESH = 0.02 rad`, i.e. **precisely when the
carrier is measured NOT to describe the field.**  Handing an angle-aware
element a congruence label the element's own F1 guard has just rejected is the
wrong call whichever way the correction is referenced.  If a corrected feature
(S6) is ever built, the delegate hand-off should still not forward `carrier`
without its own separate derivation.

---

## 6. THE CORRECTED DESIGN -- specified, not taken

The correction a SCREEN model needs is not the angular differential of the
exact ray OPL.  It is the difference between the exact eikonal and the
**split-step model's own eikonal**, both at the EXIT plane, and both minus
their normal-incidence values so the null is preserved:

```
C(x_out) = [ Lam_exact(theta) - Lam_screen(theta) ]
         - [ Lam_exact(0)     - Lam_screen(0)     ]
```

* `Lam_exact` is what this build already computes: `W_in(x_in) + OPL(x_in ->
  x_out)` from the shipped ray tracer, resolved in EXIT coordinates.
* `Lam_screen` is the eikonal of the FICTITIOUS system `apply_real_lens`
  actually implements -- zero-thickness phase plates separated by homogeneous
  slabs -- and it is a short Hamiltonian ray trace, not a new model:

  ```
  at screen i:   Lam -= (n2 - n1) sag_i(x, y)
                 p   -= grad[(n2 - n1) sag_i](x, y)
  through gap i: pz   = sqrt(n_i^2 - |p|^2)
                 x   += t_i p / pz
                 Lam += t_i n_i^2 / pz
  ```

  (the last line is the Legendre transform of the ASM kernel
  `exp(i k0 t sqrt(n^2 - |p|^2))`, so the trace reproduces the wave model's own
  geometric limit by construction rather than by assumption).

Properties that make this the right object, all checkable before any field is
propagated:

1. **Exactly zero for a plane-parallel plate at every angle** -- the two traces
   coincide term for term, which is the control this build's design failed.
   That is the FIRST test to write.
2. **Exactly zero at normal incidence** by construction, so the carrier-free
   byte-null survives.
3. It leaves the model's DOCUMENTED normal-incidence behaviour alone
   (`Lam_exact(0) - Lam_screen(0)` is the analytic model's known accuracy
   ceiling, which everything downstream is calibrated on) and corrects only the
   ANGULAR part -- which is the scoping that makes a default-on change
   arguable at all.
4. It subsumes the transverse walk-off automatically instead of excluding it,
   because both eikonals are resolved at the exit point.

What carries over verbatim from this build: the reduced-angle parametrisation
and the shear argument, the angle-FIELD (not label) interface, the guard set
G0-G7 with G7's measure-then-refuse architecture, the chain-A content-hash
cache key, the pupil-degree ladder of S3.2, and the plate control of S2.

What is new work: an exit-coordinate inversion for TWO ray systems rather than
one (the `_lens_imap.py` machinery is the obvious source), and the screen-model
ray tracer above.

**Size it before building it.**  The honest open question is whether `C` is
large enough to be worth a default change on any real element.  S4.4 measures
the shipped analytic model at 0.290 waves rms from traced AT NORMAL INCIDENCE
on design 121's last group, so the ANGULAR part of its error is plausibly a
small fraction of a much larger, already-documented, already-accepted gap --
and the shipped answer for a caller who cannot afford that gap is
`apply_real_lens_traced`, or `surface_model='displaced'`, which already exists
and already carries a congruence.

### 6.1 A smaller, adjacent gap the proto also names

`PROTO_HAMILTON_MAP_2026_08_11` S8.4 observes that `apply_real_lens` "takes
`conjugate=`, a SCALAR on-axis conjugate distance that can express neither a
tilt nor a decentre".  That is a real limitation of the ALREADY-ANGLE-AWARE
`surface_model='displaced'` path, whose obliquity fan is launched along
`conjugate` through `_compute_carrier` -- **the same resolver that already
accepts a `TiltedCarrier`.**  Confirmed by running it:

```
surface_model='displaced', conjugate=None                    -> OK
surface_model='displaced', conjugate=0.5                     -> OK
surface_model='displaced', conjugate=TiltedCarrier(inf,0.02,0) -> ValueError:
    "conjugate must be None, a signed scalar distance, 'auto', or a wavefront
     ndarray, got TiltedCarrier."
surface_model='displaced', conjugate=TiltedCarrier(0.5,0.02,0) -> same
```

So the refusal is a VALIDATOR clause, not a capability gap: the resolver
downstream of it handles a `TiltedCarrier` today, and `_displaced_carrier_*_fn`
would launch the obliquity fan along the tilted/decentred congruence with no
new physics at all.  Widening that one clause is a much smaller change than S6,
targets a model whose angular referencing is already CORRECT (it modifies the
per-surface sag OPD with true ray cosines and leaves the angular spectrum to do
the gaps -- exactly the split S2 shows is the right one), and is the
recommended next step ahead of the corrected screen.  **What it would BUY is
unmeasured here** -- the validator refusal is measured, the accuracy gain is
not.

---

## 7. SUITES AND PROVENANCE

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X, 24 logical CPUs
python 3.14.6   numpy 2.4.4      lumenairy 5.34.0 (source tree, feat/angle-aware-lens)
branch base feat/inverse-map @ b67130a; lumenairy/** unmodified
```

| gate | result |
|---|---|
| consumer set A + c14 + c15 + `test_audit_misc` (Windows) | **451 passed** in 155.8 s |
| consumer set B, remaining files incl. the `slow` marks (Windows) | **568 passed** in 651.2 s |
| consumer set C, the three heavy files (`test_audit_lens_models_2026_07`, `test_fga`, `test_lens_gbd`) | **101 passed** in 965.8 s |
| WSL / OpenBLAS proxy (`~/lumvenv`, BLAS pinned to 1 thread): chunked-sag, slant byte-identity, prepared/enums, c14, c15, hammer H1 | **160 passed** in 150.4 s |
| `ruff check lumenairy/ tests/unit/` + both new validation files (WSL) | **All checks passed** |

**1 120 tests on Windows + 160 on WSL, 0 failed, 0 skipped by this work, no
`xfail` added.**

The consumer set was surveyed rather than assumed: 29 files under `tests/unit`
carry a real `apply_real_lens(` call site (not the `_traced` / `_maslov`
siblings, and not the three `capturing_apply_real_lens` stubs in
`test_audit_optimize.py`, which REPLACE the function -- run anyway, in set B,
because they pin the call-site contract).  All 29 are in sets A-C.  **Every one
of these gates is a formality in the strict sense
-- `lumenairy/**` is byte-identical to the branch base, so no test COULD have
moved -- and they are run anyway, because "no library file changed" is a claim
about a `git diff` and a green suite is a claim about the library.**

Reproducing the study:

```
python hmap_consumer2_121.py plate    # S2.2, the refutation, ~2 s
python hmap_consumer2_121.py gap      # S4.1, the proto's S7 reproduced, ~8 s
python hmap_consumer2_121.py map      # S4.2, ~2 s
python hmap_consumer2_121.py degree   # S3.2, ~1 s
python hmap_consumer2_121.py nodes    # S4.3, ~2 s
python hmap_consumer2_121.py field    # S4.4, ~3 s
```

Results of record are `_hmap_c2_*.json`.  Every mode is pure ray tracing plus
(for `plate` and `field`) short analytic-element calls, so the whole study is
seconds and is free to re-run.

---

## 8. WHAT IS NOT CLAIMED

* **The refutation is proved on a PLANE-PARALLEL PLATE in closed form and
  measured there.**  For a CURVED element the shipped model is not exact, and
  this study does not measure how much of its angular error the angular
  spectrum still carries -- only that the leading, plate-exact part of the
  proto's quantity is entirely carried and therefore must not be added.  The
  sign of the verdict does not depend on that: a correction that is provably
  wrong by 2.77 waves on the simplest element in the prescription cannot ship
  whatever it does on the hardest.
* **The field arm (S4.4) is not a null control in the campaign's usual sense.**
  It has no independent oracle: `apply_real_lens_traced` is a different model,
  not truth, and its own error at this configuration is unmeasured.  It is
  reported because it is the arm the brief asked for and because its
  0.290-waves-at-normal-incidence column is the reason it could not decide the
  question.
* **The corrected design of S6 is UNMEASURED.**  No error, no cost, no
  Chebyshev degree in exit coordinates, and no statement about whether `C` is
  large enough to matter.
* **`conjugate=` accepting a `TiltedCarrier` (S6.1) was probed, not built.**
  The refusal is measured (a validator `ValueError`); what widening it would
  buy in waves is not.
* **The proto's S7 numbers are not disputed** -- they are reproduced here to
  every printed digit.  What is disputed is one sentence of interpretation, and
  the correction to it is in S2.

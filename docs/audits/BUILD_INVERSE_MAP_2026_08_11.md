# BUILD -- the inverse-characteristic per-pixel evaluator

**2026-08-11.  Branch `feat/inverse-map`, cut from `fix/tilt-quadratic-opl`
@ `a185cfc` (two commits ahead of `main` @ `755ad99` = v5.34.0).  The piston fix
underneath is REQUIRED physics and is untouched.  No commit, no push, no `gh`,
no CHANGELOG.**

Builds `PROTO_INVERSE_MAP_2026_08_11` S6's production design.  One measurement
taken before the first line of library code was written overturned the SHAPE of
that design (S2), and one taken after the last line was written decided its
DEFAULT (S4.6).  Between them the object is faster, smaller, more accurate and
simpler than the sketch -- and it is opt-in.

---

## 0. VERDICT

> **BUILT, GUARDED, MEASURED, SCOPED, ADJUDICATED -- and still OPT-IN, on a
> named blocker.**  The sketch's
> SOURCE AXES are removed because they are not representable; the model
> wins its own accuracy bar by three to four decades; scoping it to the
> terminal leg removes the chain amplification it was tripping; and an
> INDEPENDENT EXACT-TRACE ORACLE shows the acceptance banner it moves was
> flattered by the path it replaces -- but one shipped guard still refuses
> the default, and the guard wins.
>
> **1. THE SHARED MAP IS NOT VIABLE, and the measurement is decisive.**  The
> proto sketched `G(x_out, y_out; x_src, y_src)` -- one map per (group, launch
> config), contracted at each order's own source label -- and covered the
> niche-C6 residual eikonal by WIDENING the source box by
> `max|grad a_fit| x |R|`, i.e. by treating the C6 launch augmentation as an
> equivalent SOURCE DISPLACEMENT.  That is exact only for the part of
> `grad a_fit` that is CONSTANT over the launch lattice.  Measured on design
> 121's real chain at the real last group, order (-4,-2)
> (`imap_afit_121.py`): `max|grad a_fit| = 2.1753e-02 rad` over the union
> pupil, of which the mean -- the tilt, the only part a source shift can
> express -- is `2.7668e-03 rad`, **12.7 %**.  Traced through the group over
> 16 237 union-pupil rays, launching along `grad(W + a_fit)` instead of
> `grad(W)` moves the exit landing by **210.3 um** and the OPL by **50.83
> waves**; removing the best pure source-shift stand-in leaves **193.7 um /
> 48.55 waves**.  Against a parity bar of 1.11e-04 waves that is
> **4.4e+05 x**.  *The C6 augmentation is aberration, not a source
> displacement.*  No source-labelled map at any node count represents the
> congruence `apply_real_lens_traced` actually launches.
>
> **2. THE SOURCE AXES WERE ONLY THERE TO SHARE ONE BUILD ACROSS A FAN.**
> Dropping them costs nothing and buys everything.  The map is fitted from the
> landings the element has ALREADY traced, so it describes the exact
> congruence including `a_fit`; it needs ZERO extra rays (the shared build
> needed 3.78 million); it needs no C6 pad, no node-hull coverage argument
> across labels and no cross-order cache coherence; the build drops from
> **6.00 s to ~0.2 s**; storage drops from **276.5 kB to 5.8 kB**; and
> the proto's own S3.1 says a single congruence at exit degree 14 is a DECADE
> more accurate than the 4-D map (1.24e-02 nm / 3.24e-06 waves against
> 0.032 nm / 3.84e-05).  Break-even is the first order instead of half of one.
>
> **3. EVERYTHING ELSE IN THE SPEC SURVIVED.**  Exit degree 14, the measured
> numba kernel shape, the plug-in at the `sub > 1` OPL branch and the
> ray-density branch, the guard set, the parity framing, the chain-A cache
> key discipline, the feature flag with the shipped path as the fail-before.
> The one item that did NOT survive is the flag's DEFAULT -- see item 5.
>
> **4. THE PARITY FRAMING WAS THE RIGHT CALL AND IT IS WHAT CAUGHT THINGS.**
> G8 is comparative -- the model must beat, on held-out ray samples against
> exact ray truth, the very Newton path it replaces, on BOTH the max and the
> rms.  An absolute `lambda/100` bar would have shipped a 24x regression
> (proto S3.4).  Measured: **3.68e-04 x** the incumbent on order (0,0) and
> **0.271 x** on the extreme order (-4,-2).  During this build the guards
> refused twice for real reasons and both refusals were correct.
>
> **5. AND THE BAR IT PASSES IS NOT THE BAR THE DESIGN IS ACCEPTED ON.**
> With the flag ON the shipping banner moves -- FWHM 3.350 -> 3.550 um, EE3
> 90.3 -> 89.7 %, peak 5.529e+03 -> 5.172e+03.  The obvious reading ("the
> reference was flattered by the incumbent's error") is REFUTED by a
> control: raising `_NEWTON_MAX_ITERS` 12 -> 60 with the map off reproduces
> the reference to every printed digit, so the incumbent is already
> converged there.  Nor does the size add up -- the per-group field census
> puts the arms 0.23 % of Strehl apart against a 6.5 % peak change.
> Something accumulates over the chain's six traced legs that one group
> does not show.  S5 localizes it; S6 decides it.
>
> **6. THAT ACCUMULATION IS NOW LOCALIZED, AND IT IS NOT WHAT WAS
> EXPECTED.**  A per-leg decomposition (S5) shows the model is MORE
> accurate than the incumbent on every coarse leg as well, so the
> "nothing to gain at `sub = 4`" hypothesis is refuted.  The arms differ
> in the SKIRT (1.2-5.9 % of peak) while agreeing in the core, and a
> traced chain re-fits its models to each leg's returned field -- so the
> carrier is a change of downstream model BRANCH, not added wavefront
> error.  Scoping the evaluator STRUCTURALLY to the leg nothing re-fits
> (the fine retrace) removes **87 %** of the banner move for a wavefront
> change worth 0.03 % of Strehl: **the chain amplifies intermediate-field
> skirt changes by ~two orders of magnitude**, which is a finding about
> the chain rather than about this feature.  Scoped, the banner reads
> 3.450 / **90.3** / **99.8** / **99.9**, pk 5.486e+03 -- EE3 exact, EE6
> and EE12 each BETTER, peak -0.8 %.
>
> **7. AND AN INDEPENDENT ORACLE DECIDES THE REMAINING 0.8 %.**  Exact-
> trace Newton on the terminal group -- residual and Jacobian both from
> real traced rays, converged to 2.2e-17 m, no polynomial of any degree
> in the truth -- scored both arms' own `opl_map` at 1 024 probe pixels on
> each of two orders (S6).  In the CORE, where every banner metric is
> computed: the incumbent carries **6.0e-03 to 1.08e-02 waves rms** of
> wavefront error and 0.075-1.63 % of amplitude error; the model is
> **4.9x to 4 450x closer** on OPL and 6x to 711x closer on amplitude, on
> both orders and in the skirt as well.  **The recorded 3.350 um was
> flattered by incumbent error at the terminal leg; 3.450 um is the
> faithful reading.**  The banner re-baseline case is REPORTED in S6.4 and
> deliberately NOT taken here.
>
> **8. AND THE DEFAULT IS BLOCKED ON ONE SHIPPED GUARD, NOT ON THE PHYSICS.**
> Flipping it to `True` fails exactly one test -- niche C6's assertion that
> the polynomial and spline `newton_fit` backends describe the same map to
> 5e-04.  The model's `det J` channel is polynomial-basis evidence, so it
> breaks that symmetry whichever way it is gated.  A shipped guard is not
> weakened to accommodate a feature, so the flag stays `False` and S6.5
> specified the fix.  **That fix is now EXECUTED** (S6.5a): the `det J`
> channel is derived from traced data alone, the winner was raced against
> this same oracle (`analytic_inverse`, amplitude 1.90e-07 / 2.00e-05 against
> the incumbent's 7.50e-04 / 1.63e-02), and the model no longer accepts a
> Jacobian from its caller.  **The guard still fails, on a SECOND coupling
> that is not `det J`**: the element's ray-fit-domain restriction is itself
> basis-gated (fix D5), so the two backends hand the model different sample
> sets, and an exit-coordinate degree-14 fit amplifies that ~20x where the
> incumbent absorbs it.  Fix specified, not taken -- it changes the model's
> fit domain on every call.  **That fix was then executed too (S6.5b) and
> REFUTED by the model's own G8: the unweighted launch-square fit reads
> 4.53e-01 waves inside the beam against the restricted model's 1.996e-05,
> 4.4 decades worse.  The element's fit-domain restriction is load-bearing.**
> **The accuracy case is decided; the default is still one guard away, the
> guard is right, and the routes past it are named in S6.5b.**

---

## 1. WHAT WAS BUILT

| file | what |
|---|---|
| `lumenairy/elements/_lens_imap.py` | NEW.  `InverseCharacteristic`, `build_inverse_map`, the numba evaluator, guards G1-G8, the cache, the two flags |
| `lumenairy/elements/_lens_traced.py` | the `inverse_map` kwarg, the build site, the two consumption branches, `_ray_density_amp_grid(_pre=...)`, `_TracedExitSupport.taper_grid` |
| `lumenairy/elements/_traced_flags.py` | `_IM` module alias, the two registry rows, era `v5.34` |
| `tests/conftest.py` | `_lens_imap` added to `_LEAK_GUARD_MODULES` |
| `tests/unit/test_niche_c15_inverse_map.py` | NEW.  25 tests |
| `tests/unit/test_niche_c14_encapsulation.py` | `_lens_imap` added to the two layer-map module dicts |
| `docs/audits/TRACED_LAYER_MAP.md` | rows 32-33, era prose |
| `validation/repro_traced_carrier_121/imap_afit_121.py` | NEW.  The measurement in S2 |
| `validation/repro_traced_carrier_121/imap_prod_121.py` | NEW.  The per-order acceptance, the radial census and the load-robust bucket census |
| `validation/repro_traced_carrier_121/imap_banner_arm.py` | NEW.  Runs `focus_scan_121.py` unmodified under a stated flag / Newton-cap arm |
| `validation/repro_traced_carrier_121/imap_legs_121.py` | NEW.  The per-leg decomposition of S5: every element call run BOTH ways on the same inputs |
| `validation/repro_traced_carrier_121/imap_oracle_121.py` | NEW.  The independent exact-trace oracle of S6 |
| `lumenairy/propagators/carrier.py` | the niche-C15 structural scoping at the ordinary-leg call site (S5.3), plus `_lens_imap` in `_WORKER_STATE_MODULES` |

### 1.1 The object

```
G(x_out, y_out)  ->  (x_in, y_in, OPL, det J)
```

A total-degree-14 Chebyshev least-squares model in the EXIT coordinates,
fitted from the `n_launch^2` landings the element traced for its own forward
fit.  120 terms x 4 channels x 8 B = **5.8 kB**.

* **The OPL channel is the element's own quantity.**  It is fitted from
  `opl_grid`, which already carries the H6 carrier eikonal, the niche-C6
  residual and the on-axis piston reference.  Nothing in the new module
  re-derives an eikonal, so nothing in it can disagree with the element about
  one.
* **`det J` is the ANALYTIC gradient of the forward fits at the launch nodes**
  -- the same estimator, through the same `_has_combined_fits` branch, that
  `_ray_density_amp_grid` uses at the entrance points.  Not `np.gradient` on
  the lattice: the proto measured that at a 1.58e-05 relative amplitude floor
  against 1.58e-06 for the analytic one (S3.5), i.e. the finite difference and
  not the map was the error.
* **The fit domain is the element's own.**  `_fit_weights` is passed straight
  through, so the niche-D1 regularised restriction and the niche-D7 raised
  order reach this model exactly as they reach the forward fit.
* **The landing hull is from every alive landing the builder is handed**, not
  from the weighted subset the fit is constrained on -- the proto's G6 lesson
  (S5.3), where the distinction was between fit-cut and full-square node hulls
  and here is the same distinction between the weighted fit samples and every
  alive landing.  It costs no rays.  **The caveat is that "handed" matters**:
  on the CONCENTRIC branch (`_fit_weights is None`) the restriction is the
  historical NaN mask, so the grids reaching the builder are already cut and
  the hull is the restricted one -- narrower than the element's own C8 support,
  which was built before the restriction.  That is a measured domain
  difference, and S4.4 records it.

### 1.2 Where it plugs in

Exactly the two sites the spec names.  At the `sub > 1` OPL branch it replaces

```
coarse Newton on X[::sub, ::sub]                  ->  opl_coarse   (9 025)
map_coordinates(order 3) + order-1 NaN pass       ->  opl_map      (6.7e7)
```

and at the ray-density branch

```
_ray_density_amp_grid on X[::sub, ::sub] (Newton #2) ->  ard_coarse (9 025)
map_coordinates(order 1) x 2                        ->  ard_map    (6.7e7)
map_coordinates(order 1) x 2 on (xe, ye)            ->  the remap='full'
                                                        entrance pullback
```

with one 4-channel evaluation.  **Everything below the inversion is the
shipped code, unchanged**: `_ray_density_amp_grid` grew one optional argument
(`_pre`) that supplies the inversion, and its caustic floor, fold census,
`|E_in|` sample, entrance aperture stop, niche-C8 taper and `remap_sampling`
branches are byte-identical with `_pre=None`.

### 1.3 The switch

`lumenairy.elements._lens_imap.TRACED_INVERSE_MAP = True` (shipped), with the
per-call override `apply_real_lens_traced(..., inverse_map=True/False/None)`.
`False` is the fail-before and is byte-identical to the pre-feature library --
not "disables the map" but "never builds it", proved by a test that makes
`build_inverse_map` fatal and runs the flag-off call.

The gate closes, and each clause is a refusal rather than a workaround:

| clause | why |
|---|---|
| `sub > 1` | at `sub == 1` there is no coarse lattice and no upsample to replace; the Newton already runs per pixel and a model could only add error |
| `inversion_method == 'newton'` | `'fit'` is already a per-pixel exit polynomial; `'backward_trace'` has no forward fits to build from |
| `not _chunk_assembly` | the row-band path exists so a full-grid float64 is never materialised; handing it one would undo the memory fix it is |
| `not use_gpu` | the fits live on the device; this kernel is a CPU numba/NumPy pair |

---

## 2. THE MEASUREMENT THAT CHANGED THE DESIGN

`validation/repro_traced_carrier_121/imap_afit_121.py`, ~1 min, coarse legs
only (no 8192-square retrace).  A pass-through wrapper on
`_lens_traced._fit_residual_eikonal` records the fitted object of every group;
`grad a_fit` is then evaluated on the shipped 229-point launch lattice and
decomposed into its mean (a wavefront TILT, which IS a source displacement)
and the rest (which is not).

```
order (-4, -2), 6 residual-eikonal fits, chain 19.3 s
R at the last group's entrance -21.1392 mm; launch radius 15.2974 mm;
union pupil 9.6490 mm

call  r_fit    max|grad a|   TILT         (= src shift)  NON-TILT max   rms
 0    15.79mm  2.2918e-05    8.6686e-07    0.0000 mm     2.2598e-05   1.1579e-05
 1    15.80mm  1.9785e-05    2.3314e-06    0.0000 mm     2.0399e-05   1.0755e-05
 2    15.80mm  2.1794e-05    3.2754e-06    0.0001 mm     1.9457e-05   1.1243e-05
 3    13.15mm  1.1594e-02    1.8470e-03    0.0390 mm     9.7832e-03   3.3588e-03
 4    10.38mm  1.2027e-02    3.1349e-03    0.0663 mm     1.1262e-02   5.4240e-03
 5     7.81mm  2.1753e-02    2.7668e-03    0.0585 mm     1.9546e-02   6.8066e-03
```

and then TRACED, over the 16 237 union-pupil rays of the last group's fit:

```
grad(W + a_fit)  vs  grad(W)           exit shift 210.291 um   OPL 5.0833e+01 waves
grad(W + a_fit)  vs  grad(W) + TILT    exit shift 193.687 um   OPL 4.8545e+01 waves
                                       (the parity bar is 1.11e-04 waves)
```

**Reading it.**  On the three pre-DOE-adjacent groups `a_fit` is 2e-05 rad and
nothing about it matters.  On the last three it is 1-2e-02 rad, and on the last
group **87 % of it is non-tilt** -- genuine field-angle aberration of the
intermediate wavefront.  Removing the tilt (the best a source-labelled map can
do, and only if the consumer knows to shift the label, which the sketch's
`imap.at(carrier)` does not) buys **8 %** of the error.  The residual is
**4.4e+05 x** the bar the map is supposed to hold.

**Why the proto did not see it.**  Its own "what is not claimed" says it: every
accuracy number was map-vs-direct-ray-trace at the 32 orders' own labels, and
BOTH arms launched along `grad W` alone.  The C6 pad was carried as a box
WIDENING -- a statement about where the labels sit, not about whether the
congruence is label-representable.  Nothing in the study propagated a field or
touched a production launch, so the assumption was never exercised.  This is
the class of finding the campaign exists to produce, and it cost one minute of
chain time to make.

**What it does NOT overturn.**  The exit-degree ladder, the evaluation cost,
the upsample census, the single-valuedness census, the extrapolation number
outside the landing hull, and the parity framing -- all of those are properties
of one congruence's exit-coordinate fit and all of them carry over.  What it
overturns is exactly one thing: the source axes.

---

## 3. THE GUARDS

G1-G5 carry from `PROTO_HAMILTON_MAP_2026_08_11` S5, re-expressed for a
single-congruence map; G6-G8 are the inverse side's own.  Every one refuses at
BUILD and the call keeps the shipped path.  There is no per-pixel fallback to
write and there never was: an exit pixel outside the landing hull has no ray,
and both paths already refuse it.

| guard | refuses on | shipped threshold | measured, d121 last group |
|---|---|---|---|
| G1 one congruence | a launch node with no finite Jacobian | < 16 usable nodes | passes |
| G2 Jacobian sign + caustic | `det J` sign change, or dynamic range above the cap | 30.0 (the ray-density fold cap) | 1.39 (order -4,-2), 1.18 (order 0,0) |
| G3 alive census | too few surviving launch rays | < 16 | passes |
| G4/G5 label in-box | -- | **retired**: a per-call map has no label to be out of box, which is the S2 finding in guard form | n/a |
| G6 landing hull | a hull the alive landings cannot support | qhull declines | 458 / 140 facets |
| G7 exit-degree adequacy | least-squares residual above the incumbent's own error | free -- it IS the fit residual | see S4 |
| G8 PARITY | held-out max OR rms OR entrance-position error worse than the incumbent's on the same samples | `_IMAP_PARITY_FACTOR = 1.0` | see S4 |

### 3.1 G8 in detail, because it is the one that matters

1. Hold out the launch nodes with `i % 3 == 1 and j % 3 == 1` -- 1/9 of them,
   spatially spread.
2. Fit the model on the rest.
3. Score it at the held-out landings against the EXACT traced `(x_in, y_in,
   OPL)` there.
4. Score the INCUMBENT at the same points -- not a reproduction of it, the
   element's own `_invert_newton` on the element's own forward fits.
5. Refuse unless the model wins on the max, the rms and the entrance position.
6. Refit on everything for the shipped coefficients.

**The comparison is biased AGAINST the model**, deliberately: the incumbent's
forward fit was built on the whole lattice and therefore SAW the held-out
samples, while the model did not.  An acceptance bar should err on that side.

**The scoring set is the constrained region** (`weight >= 0.5 * max`), for the
same reason G2's census is: outside the ray-fit disc both arms are
extrapolating their own weighted fit, `|E_in|` is negligible so nothing that
happens there can reach the returned field, and a max-over-samples comparison
of two extrapolations decides the bar by coin flip.  The full-set numbers are
recorded beside the core ones so the choice is visible rather than implied.

### 3.2 The two refusals this build actually hit

Both were the guards working, and both were fixed by correcting the
MEASUREMENT rather than by loosening the bar:

1. **G2 refused on the first real run** at `det J` range **1.735e+05**.  The
   census was over the whole launch square, where `det J` is the analytic
   gradient of a degree-10 fit that is a 1e-8-weighted extrapolation -- a
   polynomial running away, not a Jacobian.  Censused over the constrained
   region it reads **1.39**, against the proto's own 1.2627 over the union
   pupil.  Censusing the runaway would have refused every real call for a
   caustic that is not there.
2. **A performance refusal in all but name.**  The first working version was
   *slower* than the path it replaced -- +27.7 s on a 1.7e+07-pixel fixture --
   because the model asks for two convex-hull tests on the WAVE grid that the
   shipped path only ever asked for on the coarse lattice.  `O(pixels x
   facets)` at 1.7e+07 x 148 is a 10^10-MAC BLAS pass, measured at 5.9 s per
   call.  Fixed by the house's own device: `_TracedExitSupport.taper_grid` and
   `InverseCharacteristic.hull_mask_grid` screen the work to a thin ring using
   the two strict radial bounds `retained_band_masks` already used, and both
   are pinned BIT-identical to the dense forms by test.  `signed_distance`
   dropped from 17.572 s to 1.677 s on the same fixture.

---

## 4. THE NUMBERS

Design 121's last post-DOE group, driven exactly as `_fine_trace_group_exit`
drives it, at `n_fine = 4096` (affordability; S5).  `imap_prod_121.py`.

### 4.1 G8 -- parity against the incumbent, per order

Held-out launch nodes, scored against the exact traced `(x_in, y_in, OPL)` at
their own landings.  "incumbent" is the element's own `_invert_newton` on the
element's own forward fits, at the same points.

| order | probe pts | map OPL max | incumbent OPL max | ratio | map OPL rms | incumbent rms | ratio |
|---|---|---|---|---|---|---|---|
| (0,0) | 794 | 7.692e-06 w | 2.0885e-02 w | **3.68e-04** | 1.761e-06 w | 7.950e-03 w | **2.22e-04** |
| (-4,-2) | 632 | 6.736e-03 w | 2.4891e-02 w | **0.271** | -- | -- | -- |

| order | map entrance | incumbent entrance | ratio |
|---|---|---|---|
| (0,0) | 5.862e-10 m | 4.555e-07 m | **1.29e-03** |
| (-4,-2) | 6.424e-08 m | 9.997e-08 m | **0.643** |

**The model is between 1.6x and 4 500x better than the path it replaces, on
every statistic, on both orders.**  The spread between the two orders is
itself informative: order (0,0) reaches this group with a CONCENTRIC fit disc
(the historical NaN mask, 7 129 retained samples of 54 289), so the model fits
a clean disc and reads 6.6e-06 waves of residual; order (-4,-2) is DECENTRED,
so niche D1 expresses the restriction as 1e-8 WEIGHTS instead and 36 912
samples enter the fit, including the weakly-weighted outside where the model is
pulled by data the beam never occupies.  Its residual is 8.5e-03 waves --
three decades worse, and still 3.7x inside the incumbent.

### 4.2 G7 -- the fit residual, and G2 -- the Jacobian

| order | OPL max | OPL rms | entrance | `det J` rel | amplitude rel | `det J` range |
|---|---|---|---|---|---|---|
| (0,0) | 6.606e-06 w | 1.767e-06 w | 4.99e-10 m | 3.57e-08 | 1.79e-08 | 1.178 |
| (-4,-2) | 8.471e-03 w | -- | 6.82e-08 m | -- | -- | 1.391 |

`det J` at 1.18-1.39 reproduces the proto's own 1.2627 census over the union
pupil (S5.4) -- the map is interpolating one smooth branch of a diffeomorphism,
which is what G2 exists to establish.

### 4.3 The build

| quantity | shared 4-D map (proto S4.5) | this build |
|---|---|---|
| nodes / congruences traced | 72 | **0 (the element already traced it)** |
| extra rays | 3.78e+06 | **0** |
| build | 6.00 s | **0.13-0.57 s** typical (one 3.03 s outlier under load) |
| storage | 276.5 kB | **5.8 kB** |
| break-even | 0.50 orders | **immediate** |

### 4.4 The field, arm-vs-arm

Radial census of `|E_on - E_off|` against the OFF arm's peak, order (0,0),
`n_fine = 4096` (r in pixels; the beam core is `r < 250`):

```
r        0- 241   |dE| 1.296e-01   d|E| 1.464e-03   |E|off 1.000e+00   power on/off 1.0019
r      241- 483   |dE| 8.585e-02   d|E| 1.247e-03   |E|off 9.125e-01   power on/off 1.0017
r      483- 724   |dE| 2.219e-02   d|E| 7.018e-04   |E|off 6.905e-01   power on/off 1.0010
r      724- 965   |dE| 2.222e-02   d|E| 3.623e-04   |E|off 4.323e-01   power on/off 0.9997
r      965-1207   |dE| 8.738e-03   d|E| 3.628e-04   |E|off 2.220e-01   power on/off 0.9981
r     1207-1448   |dE| 3.991e-03   d|E| 2.715e-04   |E|off 9.206e-02   power on/off 0.9964
r     1448-1690   |dE| 1.921e-02   d|E| 1.921e-02   |E|off 3.001e-02   power on/off 0.9570
r     1690-1931   |dE| 7.293e-03   d|E| 7.293e-03   |E|off 7.293e-03   power on/off 0
r     1931-2172   |dE| 1.189e-03   d|E| 1.189e-03   |E|off 1.189e-03   power on/off 0
```

Two readings, and they are different findings:

* **In the beam the difference is PURE PHASE.**  The amplitude difference is
  <= 1.5e-03 of peak everywhere inside `r < 1200` while `|dE|` reaches 0.13;
  the core phase difference is 2.06e-02 waves max, **7.67e-03 waves rms** --
  and that is, to the digit, the incumbent's own measured error on the same
  call (2.09e-02 / 7.95e-03).  The two arms differ by exactly the amount the
  incumbent is wrong by.  As a Strehl that is `exp(-(2 pi x 7.67e-03)^2)` =
  **0.9977**, i.e. 0.23 %.
* **Outside `r ~ 5.2 mm` the model REFUSES where the shipped path
  EXTRAPOLATES.**  The map's landing hull comes from the fit-restricted
  samples it was handed; the element's forward fit is evaluated (through
  Newton, clipped into the launch disc) over a wider exit region, and emits a
  halo out to ~6.7 mm at <= 7.3e-03 of peak.  That halo carries ~1e-04 of the
  power -- it cannot move an encircled-energy metric -- but it IS a domain
  difference and it is recorded here rather than left to be discovered.

### 4.5 THE COST, from the bucket census

`IMAP_MODE=census IMAP_NFC=8192` -- both arms in ONE process, with a
pass-through spy on `scipy.ndimage.map_coordinates`.  The shipped retrace
configuration, order (-4,-2), 6.711e+07 exit pixels.

| # | input | out points | order | OFF | ON |
|---|---|---|---|---|---|
| 0 | (95, 95) | 6.711e+07 | 3 | 6.090 s | -- |
| 1 | (95, 95) | 6.711e+07 | 1 | 3.024 s | -- |
| 2 | (8192, 8192) | 9 025 | 1 | 0.002 s | -- |
| 3 | (95, 95) | 6.711e+07 | 1 | 2.942 s | -- |
| 4 | (95, 95) | 6.711e+07 | 1 | 2.998 s | -- |
| 5 | (95, 95) | 6.711e+07 | 1 | 3.114 s | -- |
| 6 | (95, 95) | 6.711e+07 | 1 | 3.147 s | -- |
| -- | (8192, 8192) | 6.711e+07 | 1 | -- | 2.631 s (`abs(E_in)` at the exact entrance points) |
| 7 | (8192, 8192) | 6.711e+07 | 1 | 2.618 s | 3.047 s (residual phasor, real) |
| 8 | (8192, 8192) | 6.711e+07 | 1 | 2.790 s | 3.310 s (residual phasor, imag) |
| | | | **total** | **26.725 s** | **8.988 s** |

**The six coarse-lattice upsamples the model exists to remove -- 21.315 s of a
179.6 s element -- go to zero**, and the coarse 9 025-point `abs(E_in)` sample
(0.002 s) is replaced by a full-resolution one (2.631 s), which is the price
the proto priced at ~1.55 s and is the only term that grows.  Net on the
interpolation bucket: **26.725 -> 8.988 s, minus the 1.536 s build = 16.2 s
per order.**

**AND THE WALL DOES NOT SHOW IT.**  The same two calls measured 179.6 s (off)
and 180.9 s (on).  The bucket census is a within-process difference and is
trustworthy; the wall on this box is not (S5's load caveat -- two calls seconds
apart scattered by +-9 s on a 32 s element).  What the bucket census does NOT
price is the model's own full-grid work: the 4-channel degree-14 evaluation
(the proto measures 1.910 s at this pixel count), the screened hull mask and
the screened taper (~1 s together), and four full-grid float64 allocations.
Adding those to the ON column gives ~13.4 s against 26.7 s, i.e. a projected
~13 s per order -- and the wall's +1.3 s says the remainder is either load or
memory bandwidth this study did not separate.  **The honest statement is: the
interpolation bucket falls by 16.2 s per order, and the end-to-end wall was
not resolved on this box.**  A clean-box repeat is one command
(`IMAP_MODE=census`).

### 4.6 THE BANNER -- and why the flag ships OFF

`focus_scan_121.py` at the shipping configuration (`N=2048 rs=4 nfc=8192
wf=4.0 nout=2048`, pure library defaults), one arm per invocation via
`imap_banner_arm.py`.  Three arms, ~8-10 min each:

| arm | FWHM | EE3 | EE6 | EE12 | peak | verdict |
|---|---|---|---|---|---|---|
| `TRACED_INVERSE_MAP=False`, 12 Newton iters (**the default**) | **3.350 um** | **90.3 %** | **99.7 %** | **99.8 %** | **5.529e+03** | **IDENTICAL to the recorded acceptance, every digit** |
| `TRACED_INVERSE_MAP=False`, 60 Newton iters | 3.350 um | 90.3 % | 99.7 % | 99.8 % | 5.529e+03 | identical -- the incumbent's Newton cap is NOT the variable |
| `TRACED_INVERSE_MAP=True`, 12 Newton iters | 3.550 um | 89.7 % | 99.6 % | 99.7 % | 5.172e+03 | **MOVES** |

**Read the middle row first, because it is the one that decides the default.**
The obvious explanation for the third row is "the model is more faithful and
the reference banner was flattered by the incumbent's error"; the G8 numbers
of S4.1 support it, and the banner's own note records an ideal-field ceiling
of 3.45-3.55 um that the shipped 3.350 sits BELOW.  The middle row refutes it.
Raising `_NEWTON_MAX_ITERS` from 12 to 60 is an INDEPENDENT way to make the
shipped path more faithful, and it reproduces the reference to every printed
digit -- so the incumbent's inversion is already converged at the shipping
configuration and there is no incumbent error of banner size for the model to
be removing.

And the size does not add up either.  The per-order field census of S4.4 puts
the arms 7.67e-03 waves rms apart in the core of ONE group -- 0.23 % of Strehl
-- against a 6.5 % peak change over the whole chain.  **Something accumulates
across the chain's six traced legs that one group does not show**, and the most
likely candidate is stated rather than assumed: at the coarse legs
`ray_subsample = 4`, where the shipped cubic upsample's error scales as
`(sub*dx)^4` and is therefore ~`(4/87)^4` = 1.8e-06 of its size at the fine
retrace -- so the model would have nothing to remove and its own residual to
pay.

**S5 MEASURED THAT AND REFUTED IT**, and then found the real mechanism.  A
`sub`-dependent gate is deliberately NOT built either way: it would be a
threshold chosen to make a banner come out right rather than a measurement.
The scoping S5.3 does adopt is structural and reads no number.

> **The flag ships `False`.**  A change that IMPROVES the quantity it is
> guarded on and still MOVES the quantity the design is accepted on is
> precisely the class this campaign exists to keep out of a default.
> Everything is built, guarded, tested and measured; the default stays where
> the evidence is.  S5 closes the mechanism and names what would close the
> remaining move.

---

## 5. THE ACCUMULATION, LOCALIZED -- and the scoping it justifies

`imap_legs_121.py` (LOCAL-ONLY, no library edit).  A pass-through wrapper on
`lumenairy.elements.apply_real_lens_traced` -- the name `carrier.py` imports,
locally, at BOTH its traced call sites -- runs EVERY element call the chain
makes TWICE, once per arm, on the same inputs, records the delta, and returns
the OFF arm.  Returning the off arm is what makes this a DECOMPOSITION rather
than a second banner: each leg's number is that leg's OWN contribution given
the shipped input, with no compounding from the legs before it.

`LEGS_MODE=coarse`, order (-4,-2), the six post-DOE groups at the chain's own
`ray_subsample = 4`, `final_distance = 0` (~3 min):

| leg | prescription | dx (um) | n_launch | map fit resid (waves) | G8 vs incumbent (max / rms) | core phase rms (waves) | max abs dE / peak | d power |
|---|---|---|---|---|---|---|---|---|
| 0 | plate N-SF1 | 51.234 | 219 | 6.155e-08 | 0.376x / 0.629x | 3.28e-08 | 3.90e-02 | -5.27e-04 |
| 1 | plate N-BK7 | 51.234 | 225 | 6.444e-08 | 0.528x / 0.872x | 1.01e-08 | 5.89e-02 | -7.14e-04 |
| 2 | doublet PK52A/SF57 | 51.235 | 231 | 6.246e-08 | **1.53e-05x** / 3.08e-05x | 5.92e-04 | 5.88e-02 | -7.07e-04 |
| 3 | singlet LAK8 | 44.912 | 265 | **6.194e-03** | 0.702x / 0.669x | 1.15e-03 | 1.25e-02 | +4.30e-04 |
| 4 | singlet LAK9 | 38.432 | 265 | **3.878e-03** | 0.273x / 0.233x | 2.22e-03 | 4.55e-02 | +5.60e-04 |
| 5 | doublet SK2/SF57 | 33.211 | 231 | **1.888e-03** | 0.207x / 0.181x | 1.51e-03 | 2.22e-02 | +3.93e-03 |

### 5.1 The hypothesis is REFUTED

S5 of the first draft proposed that at `ray_subsample = 4` the model "pays its
own residual for nothing", because the cubic upsample it removes is
`~(4/87)^4` = 1.8e-06 of its fine-retrace size.  **The G8 column kills that.**
On every one of the six coarse legs the model is at least 1.4x, and on leg 2
65 000x, more accurate than the incumbent Newton at the ray landings -- and at
`sub = 4` the incumbent's upsample error is negligible, so arm B there IS
essentially the incumbent's whole error and the comparison is not flattering
the model.  The coarse legs are not where the model is worse.  **A scoping
justified by "the model is worse at small `sub`" would have been justified by
a fact that is not true**, and would have been a banner-tuned threshold
wearing a derivation.

### 5.2 What the table DOES show -- two findings

**(a) The C6 residual eikonal is what makes the exit map hard to fit, and the
split is five decades.**  Legs 0-2 read 6.2e-08 waves of least-squares
residual; legs 3-5 read 1.9e-03 to 6.2e-03.  That split is exactly the split
S2 measured in `max |grad a_fit|` -- 2e-05 rad on the first three groups,
1.2e-02 to 2.2e-02 rad on the last three.  The two measurements were taken by
different scripts for different reasons and they agree on which three groups
are different.  **The same quantity that makes the SHARED map impossible (S2)
also sets how hard the PER-CALL map is to fit.**  It is one mechanism, seen
from two sides.

**(b) The accumulation is NOT a sum of per-leg wavefront errors.**  Take the
core phase deltas in quadrature over the coarse legs -- 5.92e-04, 1.15e-03,
2.22e-03, 1.51e-03 -- and add the fine retrace's 7.67e-03 (S4.4): **8.2e-03
waves rms**, a Strehl of `exp(-(2 pi x 8.2e-03)^2)` = 0.9974, i.e. **0.26 %**.
The banner moves **6.5 %**.  Twenty-five times.  So whatever carries the
banner, it is not wavefront error adding up leg by leg.

**Where the difference actually lives is the SKIRT.**  Read the last two
columns against the core one: `max |dE|` is **1.2-5.9 % of peak** on every leg
while the core phase rms is three decades smaller.  The arms agree in the beam
and disagree outside it -- which is where the model's landing hull, its
domain relaxation and its full-resolution `abs(E_in)` sampling all differ from
the upsampled path, and where S4.4 already located the one domain difference.

**And a traced chain is not a linear pipeline.**  Each leg re-fits its models
to the PREVIOUS leg's field: the niche-C6 residual eikonal (fitted to the
bright support), the niche-C11 decentred-fit arbiter (which SELECTS A BRANCH),
the beam radius that sizes the ray-fit disc, and the carrier reference.  A
skirt-level change to an intermediate field is precisely the input those
decisions are taken on.  **The named mechanism is therefore: a change of
downstream model BRANCH, not an added wavefront error** -- which is why it is
25x bigger than the phase budget and why it appears only end to end.

### 5.3 The scoping, on the criterion the data supports

Not "small `sub`" -- that is refuted.  The criterion is TERMINAL versus
INTERMEDIATE, and it is structural:

* the FINE RETRACE (`_fine_trace_group_exit`) produces the field the readout
  consumes.  Nothing re-fits it.  Every piece of this model's evidence -- the
  exit-degree ladder, the parity numbers, the upsample census, the cost
  census -- is about the accuracy of exactly that field.
* an ORDINARY chain leg produces an INTERMEDIATE that six further model fits
  are taken on.  This build has no oracle for that field, and S5.2 shows the
  quantity that matters there (the skirt) is not the quantity G8 guards (the
  ray-landing OPL).

Implemented as one line at the ordinary-leg call site in
`carrier.py::propagate_traced_carrier_chain`: `inverse_map=False`, via
`setdefault` so an explicit `traced_kwargs` override stays reachable.  It reads
no number -- not `ray_subsample`, not `n_fine` -- so there is nothing in it to
tune.

### 5.4 Verdict on the scoping -- and the amplification it exposes

**The null control first.**  With the scoping in and `TRACED_INVERSE_MAP`
forced True, the coarse-leg decomposition re-run reports all six legs
`engaged = False`, core phase delta **2.0e-18 waves** (the round-off of
`np.angle` on two identical arrays) and `d power` **exactly 0**.  The
intermediate legs are byte-identical whatever the flag holds.

**The banner, scoped, flag ON:**

| arm | FWHM | EE3 | EE6 | EE12 | peak |
|---|---|---|---|---|---|
| reference / flag OFF | 3.350 um | 90.3 % | 99.7 % | 99.8 % | 5.529e+03 |
| flag ON, **unscoped** | 3.550 um | 89.7 % | 99.6 % | 99.7 % | 5.172e+03 |
| flag ON, **scoped** | **3.450 um** | **90.3 %** | **99.8 %** | **99.9 %** | **5.486e+03** |

EE3 returns to the reference exactly; EE6 and EE12 each end up **0.1 point
BETTER** than the reference; the peak deficit collapses from **-6.5 % to
-0.8 %**; FWHM lands on 3.450 um, one printed digit off the reference and at
the lower edge of the banner's own stated ideal-field ceiling band
(3.45-3.55 um), which the shipped 3.350 sits BELOW.

**THE SCOPING IS THEREFORE CONFIRMED, AND SO IS THE MECHANISM OF S5.2.**  Put
the two measurements together: removing the model from the intermediate legs
removed **87 %** of the banner's peak move, and the wavefront change it
removed at those legs was 2.9e-03 waves rms in quadrature -- a Strehl of
0.9997, i.e. **0.03 %**.  A 0.03 % change to the intermediate fields was
carrying a 5.7 % change end to end.  **The traced chain amplifies skirt-level
changes to an intermediate field by roughly two orders of magnitude**, because
each leg re-fits its models (the C6 residual eikonal, the C11 arbiter's
BRANCH, the beam radius that sizes the ray-fit disc, the carrier reference) to
the field the leg before it returned.  That is a finding about the chain, not
about this evaluator, and it is the answer to "where does the 26x gap come
from".

**BUT IT IS NOT IDENTICAL, SO THE DEFAULT STAYS OFF.**  FWHM reads 3.450
against 3.350 and the peak is 0.8 % low.  The remaining move is now confined
to the ONE leg the model is scoped to, and its size is the right order for
what S4.4 measured there (7.67e-03 waves rms of core phase = 0.23 % of Strehl,
plus the domain difference at the traced-support rim).  It is a small,
localized, fully characterised residual -- and it is still a change to a
user-approved acceptance number, which is a decision this build does not get
to take on its own.  The banner's own history records two re-baselines, both
user-approved; this is a candidate for a third, not a reason to flip a default.

**What would close it** is one measurement, and it is now cheap because the
target is a single leg: an independent oracle for the fine-retrace exit field
(the prior probes' null-control pattern -- arm A vs arm B on the same aperture
field) that says which of 3.350 and 3.450 is the faithful one.  Everything
else is already measured.

## 6. THE INDEPENDENT ORACLE -- which of 3.350 and 3.450 is faithful

`imap_oracle_121.py`.  S5.4 left exactly one question: with the evaluator
scoped to the terminal leg the banner reads 3.450 um against the shipped
3.350 um, and both cannot be right.

### 6.1 Construction -- what it shares and what it does not

**Shares (the physics, by necessity -- it is what "the exit field of this
group" MEANS):** the surfaces, the ray tracer, the exit-vertex correction, the
launch congruence `grad(W + a_fit)`, and the analytic entrance eikonal
`W + a_fit` added to the geometric path.  `a_fit` is captured from the element
itself by a pass-through wrapper on `_fit_residual_eikonal`, so the oracle
cannot disagree with either arm about the congruence.

**Shares with NEITHER arm (the inversion, which is what is on trial):** the
incumbent Newtons a 95 x 95 coarse lattice on a degree-10 forward Chebyshev
fit and cubically upsamples; the model fits a degree-14 polynomial in exit
coordinates and evaluates it per pixel.  **The oracle uses no polynomial of
any degree.**  It inverts by EXACT-TRACE NEWTON: for each probe pixel the
residual is a real traced ray and the Jacobian is a 4-point central stencil of
real traced rays (`h = 2e-08 m`), seeded from the nearest landing of a dense
forward trace.  **Measured convergence: max exit residual 2.2e-17 m over 1 024
probe pixels, i.e. 7e-12 of a pixel -- machine zero.**

**The observable** is each arm's OWN finalised `opl_map` and `ard_map`,
sampled at the probe pixels through the private opt-in
`_imap_out['probe_rc']` diagnostic, so the comparison sees the inversion and
not the amplitude model, the piston phasor or the residual transport (three
stages common to both arms).  1 024 pixels per order on 32 rings x 32
azimuths, centred and sized on THAT ORDER's own measured exit footprint --
axis-centred rings would spend most of their samples where a tilted
congruence has no rays at all.

**CORE vs SKIRT** by the incumbent's own ray-density amplitude at `e^-2` of
peak: the core is the region every banner metric is computed on, and S5
measured the arms to agree there and differ by 1.2-5.9 % of peak outside it.

### 6.2 The table -- both arms against exact ray truth

`n_fine = 4096`, 1 024 probe pixels per order, both orders converged 1024/1024.

| order | region | n | INCUMBENT rms / max (waves) | MAP rms / max (waves) | ratio (rms) |
|---|---|---|---|---|---|
| (0,0) | **core** | 608 | **1.0777e-02 / 2.0498e-02** | **2.4224e-06 / 6.2901e-06** | **2.25e-04** |
| (0,0) | skirt | 416 | 2.0435e-01 / 5.9301e-01 | 1.8782e-05 / 5.6867e-05 | 9.19e-05 |
| (0,0) | all | 1024 | 1.3051e-01 / 5.9301e-01 | 1.0424e-05 / 5.6867e-05 | 7.99e-05 |
| (-4,-2) | **core** | 506 | **5.9820e-03 / 1.5267e-02** | **1.2272e-03 / 3.7925e-03** | **0.205** |
| (-4,-2) | skirt | 518 | 3.3332e+01 / 2.2762e+02 | 2.1584e+01 / 2.2762e+02 | 0.648 |
| (-4,-2) | all | 1024 | 2.3707e+01 / 2.2762e+02 | 1.5352e+01 / 2.2762e+02 | 0.648 |

Ray-density AMPLITUDE, relative rms in the core:

| order | incumbent | map | ratio |
|---|---|---|---|
| (0,0) | 7.4993e-04 | 1.2110e-04 | 0.162 |
| (-4,-2) | 1.6338e-02 | 2.2976e-05 | **1.41e-03** |

### 6.3 VERDICT: the map arm is closer, on every statistic, on both orders

**The incumbent carries 6.0e-03 to 1.08e-02 waves rms of CORE wavefront error
at the terminal leg** -- a Strehl of 0.9955 to 0.9986 -- **plus 0.075 % to
1.63 % of core amplitude error**.  The model is **4.9x to 4 450x closer** to
the exact trace in the core and **1.5x to 10 900x closer** in the skirt.

That is exactly the class and the size that moves a FWHM and a peak by ~1 %.
**So the recorded 3.350 um was FLATTERED by incumbent error at the terminal
leg, and 3.450 um is the faithful reading.**  That settles the ACCURACY
question the banner posed.  It does not by itself settle the default -- see
S6.5.

The skirt rows for (-4,-2) are large for BOTH arms (21-33 waves rms, sharing
an identical 227.6-wave maximum) because the outermost ring reaches the rim of
that order's traced support, where both arms extrapolate their own fit and
neither has data.  The model is still 1.5x closer there, and the core rows --
which is what the banner reads -- are not affected by it.

### 6.4 The banner re-baseline case -- REPORTED, NOT TAKEN

`focus_scan_121.py`'s acceptance line currently reads

```
shipping-default acceptance (CREF/AM/PIP unset, N=2048/NFC=8192/WF=4.0):
  AT-PLANE dz=0: 3.350um / 90.3 / 99.7 / 99.8, on-axis
```

and with the evaluator on and scoped it measures

```
  AT-PLANE dz=0: 3.450um / 90.3 / 99.8 / 99.9,  peak 5.486e+03
```

EE3 unchanged, EE6 and EE12 each 0.1 point BETTER, FWHM +0.100 um, peak
-0.78 %.  For scale, the file's own note puts the ideal-field ceiling through
this readout at 3.45-3.55 um / 90.3 / 99.8 -- **the new reading sits ON that
ceiling on all four numbers, where the old one sat below it on three.**  That
is the third re-baseline this banner would carry; the previous two are
recorded in the file and both were user-approved.  **This build does not edit
that line.**  The numbers above are the case for it.

### 6.5 THE BLOCKER -- one shipped guard, named and specified

Flipping the default to `True` and re-running the suites turns up exactly one
failure, and it is not a flake:

```
tests/unit/test_niche_c6_stationary_phase_launch.py
    ::test_the_two_newton_fit_backends_still_describe_the_same_map
```

Its premise, quoted from its own docstring: *"``newton_fit`` is an interpolant
choice, not a physics choice: the polynomial and the spline read the SAME
traced samples and must return the same field"*, to `5e-04` relative.

**The model breaks that symmetry, and gating it does not restore it.**  Its
`det J` channel is built from `ev_value_and_grad`, i.e. from the POLYNOMIAL
Chebyshev basis; the spline basis has no equivalent (fix D5 records that it
cannot honour a fit-domain restriction at all).  Engaging the model on both
backends leaves the two `det J` sources disagreeing through an exit-coordinate
fit that is more sensitive to them than the incumbent's per-pixel Newton was;
engaging it on the polynomial basis only makes one backend inverted by the
model and the other by Newton, which is the same asymmetry stated louder.
Measured both ways: the test fails either way.

**A shipped guard is not weakened to accommodate a feature.**  So the default
returns to `False` and the flip waits on the fix, which is specific and small:
**make the model's `det J` channel basis-independent.**  Two candidates, both
cheap and both measurable against the S6 oracle, whose amplitude column is
exactly the right scorer:

1. take `det J` from the ANALYTIC gradient of the model's OWN `x_in` / `y_in`
   channels (the inverse Jacobian, reciprocal to the forward one -- the proto
   measured that reciprocity at 1.37e-05, S5.4 there), which removes the
   forward fit from the amplitude path entirely; or
2. build `det J` by a finite-difference stencil on the traced landings, the
   same estimator for both bases -- accepting the 1.58e-05 relative amplitude
   floor the proto measured for it (S3.5), which is still 10x inside the
   incumbent's 7.5e-04 to 1.6e-02 measured here.

Either makes the two backends invert identically again, and the S6 oracle then
says whether the amplitude got better or worse rather than leaving it to a
differential test to notice.

### 6.5a THE FIX, EXECUTED AND RACED -- and what it did not reach

**The `det J` channel is now basis-independent.**  Both candidates were
implemented behind `_IMAP_DETJ_SOURCE` and raced against the SAME exact-trace
oracle's amplitude column, on both orders, `n_fine = 4096`:

| `_IMAP_DETJ_SOURCE` | order | amplitude core rel rms | vs incumbent |
|---|---|---|---|
| (incumbent, for scale) | (0,0) | 7.4993e-04 | -- |
| (incumbent, for scale) | (-4,-2) | 1.6338e-02 | -- |
| `landing_stencil` | (0,0) | 1.6015e-05 | 0.0214x |
| `landing_stencil` | (-4,-2) | 2.4450e-05 | 0.00150x |
| **`analytic_inverse`** | **(0,0)** | **1.8984e-07** | **0.000253x** |
| **`analytic_inverse`** | **(-4,-2)** | **1.9965e-05** | **0.00122x** |

**`analytic_inverse` wins on both orders** -- 84x better on (0,0), 1.2x on
(-4,-2) -- so there is no tie to resolve on simplicity, and it ships.  Two
collateral confirmations worth recording: `landing_stencil` reads 1.60e-05 on
(0,0) against the proto's independently predicted 1.58e-05 floor for that
estimator (S3.5), and `analytic_inverse`'s 1.90e-07 is consistent with the
proto's 1.37e-05 reciprocity bound being the pessimistic end of it.

Implementation: the model no longer ACCEPTS a Jacobian from its caller.
`build_inverse_map` derives it -- the census Jacobian (what G2 judges the
congruence on) is always the landing stencil, because a fold is a property of
the rays; the channel Jacobian (what the amplitude consumes) is the analytic
exit-gradient of the model's own `x_in` / `y_in` channels, reciprocated.
Channels 0-2 were already raw traced data.  `apply_real_lens_traced` lost its
`_imap_detj_grid` closure and the `newton_fit == 'polynomial'` gate clause with
it.

**AND THE c6 GUARD STILL FAILS, on a SECOND coupling that is not `det J`.**
With the flag on in both backends the test reads `d_ok = 1.06e-02` against its
`5e-04` bar.  The cause is
`_fit_domain_basis_ok = (newton_fit != 'spline')`: the element applies its
ray-fit-domain restriction for the polynomial basis (a NaN mask, or niche-D1
weights) and ABANDONS it for the spline basis, which fix D5 records that basis
cannot express.  The two backends therefore hand this model **different sample
sets** -- a disc versus the whole launch square -- and that moves its exit
normalisation box, its landing hull and its coefficients together.  The
incumbent absorbs the same difference to under 5e-04 because its forward fit
lives in ENTRANCE coordinates on a fixed lattice and its Newton only ever
evaluates inside the launch disc; **a degree-14 fit in EXIT coordinates
amplifies it ~20x.**

**The fix is specified and deliberately not taken here:** build the model from
the PRE-RESTRICTION landings -- the same arrays
`_TracedExitSupport.from_landings` is handed, before the restriction touches
them -- and unweighted, so its sample set is the traced rays and nothing else.
Mechanically small (three `n_launch^2` copies, ~1.2 MB).  Consequentially not:
it changes the model's fit domain on EVERY call, including design 121's, where
the unweighted launch square is exactly the regime whose `det J` spans 1.7e+05
and whose fit residual S5 measured at 8.5e-03 waves.  It needs the oracle race,
the banner and the full suite behind it before it can be trusted, and **a
shipped guard is not weakened to buy a default flip.**  So the default stays
`False` and this is the one remaining item.

**THE BANNER, re-measured with the basis-independent `det J` in place**
(flag forced on, scoped): `3.450um / 90.3 / 99.8 / 99.9`, peak `5.486e+03` --
**identical to the pre-fix scoped reading in every printed digit.**  The
amplitude channel got 84x more faithful and the banner did not move, which is
the right result and worth stating: the 0.100 um of FWHM and the 0.78 % of
peak that separate it from the reference are carried by the OPL, not by the
Jacobian.  The re-baseline case in S6.4 stands exactly as written.

### 6.5b THE SPECIFIED FIX, EXECUTED -- and REFUTED by the model's own guard

S6.5a specified it: build the model from the PRE-RESTRICTION landings (the
arrays `_TracedExitSupport.from_landings` is handed, stashed before anything
basis-dependent touches them), unweighted, census Jacobian unchanged.  It was
implemented in full.

**It did not survive first contact with G8.**  On design 121's own fixture the
unweighted launch-square model reads

```
held-out OPL error, inside the beam, against exact ray truth
    pre-restriction / unweighted     4.5258e-01 waves
    restricted (shipped)             1.9965e-05 waves
    the incumbent, same samples      ~2.0e-05 waves
```

-- **4.4 decades worse, and 2.3e+04x outside parity.**  The guard refused the
build; no field was produced from it; the oracle never got a chance to
adjudicate because there was nothing to adjudicate.  **That is the verdict.**

**The mechanism, stated so it is not attempted again.**  A total-degree-14
Chebyshev in EXIT coordinates has to span whatever exit region its samples
occupy.  The launch square reaches ~5x past the beam (15.297 mm of launch
radius against a ~3.1 mm beam), and its far corners land in exit territory
whose ray tube is arbitrarily distorted -- the same region whose `det J` spans
1.7e+05 (S3.2 of this doc).  Fitting one polynomial across all of it spends the
whole 120-term budget on the corners and leaves the beam four decades worse.
**The element's fit-domain restriction is not an inconvenience the model
inherits; it is load-bearing, and the model needs it for the same reason the
forward fit does.**

**A second finding fell out of the attempt and is worth keeping.**  With the
weights removed, G8's own scoring region collapsed to the whole launch square
-- and a total-degree-FOUR model then PASSED it, because arm B is worse than a
degree-4 model out in the corners.  A guard whose scoring region is inherited
from the fit weights is a guard that can be widened by widening the fit.  The
builder now accepts `census_amp` (the element's `_amp`, judged at the same
`e^-4` contour the element already uses for its exit-NA guard) so the census
region can be named independently of the fit; no shipped caller passes it, and
it is there for the next attempt.

**So the `newton_fit` coupling stands, and with it the block.**  The remaining
routes are narrower than "build it unweighted" and none is a one-line change:
give the model a fit domain of its own derived from the BEAM (which needs a
basis-independent beam radius, and the element resolves that per-basis today),
or make `_fit_domain_basis_ok` true for both bases (which is a change to the
spline path, i.e. to fix D5's territory, not to this feature).  Either is a
separate piece of work with its own evidence.

**THE STATE OF RECORD.**  The default is `False`; the fail-before reproduces
the shipping banner to every printed digit; the accuracy case (S6) is decided
in the model's favour and unaffected by any of this; and the one thing between
them is a shipped guard that is right.

### 6.6 What downstream branch flips remain

(These apply when the flag IS turned on -- they are the consequences the
flip carries, listed so the decision is fully priced.)

**None in the chain, and that is measured, not argued.**  The S5.3 scoping
keeps every intermediate leg on the shipped path, and the null control re-run
reports all six coarse legs `engaged = False`, core phase delta 2.0e-18 waves
and `d power` exactly 0 -- so no leg's C6 residual-eikonal fit, C11 arbiter
branch, beam radius or carrier reference sees a different input.  The only
field that changes is the terminal retrace's, and it changes toward the exact
trace.

Two consequences do follow and are named rather than left to be found:

* **The banner moves** (S6.4) -- one printed digit of FWHM and 0.78 % of peak.
  Anything pinned to those digits downstream of this repo moves with it.
* **The terminal leg's Newton no longer runs**, so its unconverged-pixel
  `RuntimeWarning`, its worker-pool spawn and its `newton_amp_mask_rel`
  masking are inert on that leg.  The guard's own refusal report replaces the
  first; the other two are performance paths with no observable output.

---

## 6A. WHAT IS NOT CLAIMED

* **The shared 4-D map was not built.**  S2 refutes its premise for the
  production launch congruence, and shipping a known-unrepresentable object
  behind a guard that would always refuse is not a feature.  For a call with
  `REMAP_STATIONARY_PHASE_LAUNCH` off (or `preserve_input_phase != 'remap'`,
  or no carrier) the congruence IS the pure two-parameter family and a shared
  map would be exact -- that case is real, it is narrower, and it is not built
  here.
* **The niche-C8 taper's plateau is unchanged at `sqrt(2) * sub * dx`.**  Its
  own docstring derives that width from the upsample the inverse path removes,
  so on this path it is CONSERVATIVE (it tapers a wider band than it needs to)
  rather than correct.  Narrowing it would readmit skirt the shipped path
  currently zeroes, which is a physics change and not this build's to make.
* **Timings on this box are upper bounds.**  It carried other campaigns
  throughout (15 python processes, 10-35 GB resident, CPU 73 % at launch), and
  two element calls seconds apart scattered by +-9 s on a 32 s element.  The
  saving is therefore quoted from the BUCKET CENSUS -- both arms in one
  process, with a pass-through spy on `map_coordinates` -- and not from the
  wall clock.  Same device, same reason, as the proto's own load caveat.
* **`n_fine = 4096` for the per-order acceptance**, against the shipped
  8192, for affordability.  The replaceable bucket grows with the pixel count
  while the build does not, so the saving gets BETTER at 8192 -- an inference
  supported by the census, not an independent measurement at that grid.
* **No GPU path.**  `use_gpu=True` closes the gate.
* **The row-band assembly path is untouched** and keeps the shipped chain.  A
  band-wise evaluation is natural (the model is pointwise) and is not built.

---

## 7. VERIFICATION RECORD

| check | result |
|---|---|
| `ruff check lumenairy/ tests/` -- Windows and WSL | clean, both mounts |
| every changed file decodes as cp1252 / is pure ASCII | yes |
| `test_niche_c15_inverse_map.py` (NEW, 25 tests) -- Windows | 25 passed |
| `test_niche_c15` + `test_niche_c14` -- WSL (py3.12, numpy 2.4.6, OpenBLAS) | 57 passed |
| `test_niche_c14_encapsulation.py` (registry, layer map, era presets) | 32 passed |
| c6-stationary-phase + d2 + d6 + c14 at the shipped default, Windows | **129 passed** (983 s) |
| d2 + d6 + c15 + c14 AFTER the S5.3 carrier scoping, Windows | **133 passed** (906 s) |
| c15 + c14 after the scoping, WSL | **57 passed** |
| the five that failed with the flag ON, re-run individually at the default | **5 passed** (303 s) |
| c6-fit-guard / c11 / c12 / tight_focus | passed even with the flag ON, so unaffected by the default |
| `prepare_real_lens_traced` + `apply_real_lens_traced_multi`, both arms | both work; the flag is live through both wrappers; the cache hits |
| d121 banner, flag OFF (the fail-before) | **identical to the recorded acceptance, every digit** |
| d121 banner, flag OFF, `_NEWTON_MAX_ITERS = 60` | identical -- the control that refutes the obvious reading of S4.6 |
| d121 banner, flag ON, UNSCOPED | moves 3.550 / 89.7 / 99.6 / 99.7, pk 5.172e+03; S4.6 |
| d121 banner, flag ON, SCOPED (S5.3) | 3.450 / **90.3** / **99.8** / **99.9**, pk 5.486e+03 -- 87 % of the move recovered, not identical |
| coarse-leg null control, flag ON, scoped | all six legs `engaged=False`, delta 2.0e-18 waves, d power exactly 0 |
| independent exact-trace oracle, 2 orders x 1024 pixels | 1024/1024 converged, exit residual 2.2e-17 m; **map closer on every statistic** (S6) |
| c15 + c14 + c6-sp + d2 + d6 at the shipped default, Windows | **154 passed** (706 s) |
| c15 + c14 + c6-sp after the S6.5a fix, WSL | **78 passed** |
| c15 + c14 + c6-sp + tight_focus after S6.5b, WSL | **93 passed** |
| the S6.5b pre-restriction build | **REFUSED by G8** at 4.53e-01 waves vs the shipped 2.00e-05; reverted (S6.5b) |
| oracle re-run after the S6.5b revert | map core rms 1.67e-06 / 1.19e-03 w, amplitude 1.90e-07 / 2.00e-05 -- state of record intact |
| the `det J` race, both sources x both orders, vs the oracle | `analytic_inverse` wins both; shipped |
| banner, flag ON + scoped + basis-independent `det J` | 3.450 / 90.3 / 99.8 / 99.9, pk 5.486e+03 -- unchanged by the fix |
| the default flipped ON, after the `det J` fix | **1 failed** -- niche C6's two-backend agreement, now on the FIT-DOMAIN coupling (S6.5a); default reverted |
| d6 exact-tilted-leg with the default ON | 38 passed |
| d2 + c6-stationary-phase with the default ON | 58 passed, 1 failed (the same C6 test) |

The niche set was run twice.  The first pass (1 982 s, 218 passed) was taken
while the flag still defaulted ON and produced five failures --
`test_niche_c6_stationary_phase_launch`, two in `test_niche_d2_chain_multi`,
one in `test_niche_d6_exact_tilted_leg`, and the layer-map shipped-column check
(which was simply mid-edit).  **Those five are the same finding as S4.6 seen
from the test side**: with the evaluator engaged, four suites that pin a
traced answer to a measured envelope move outside it.  With the shipped default
(`False`) they are byte-identical to the branch base by construction, and the
re-run confirms it.  They are recorded here rather than dropped, because they
are the cheapest reproduction of the open item: any one of them is a 1-3 minute
arm for closing it.

---

## 8. FILES AND REPRODUCTION

```
# the measurement that changed the design (S2), ~1 min, coarse legs only
cd validation/repro_traced_carrier_121 && python imap_afit_121.py

# per-order acceptance: parity, the field delta, the radial census (S4)
IMAP_NFC=4096 IMAP_ORDERS="0,0;-4,-2" python imap_prod_121.py

# the load-robust bucket census (S4.5)
IMAP_MODE=census IMAP_NFC=8192 IMAP_ORDERS="-4,-2" python imap_prod_121.py

# the per-leg decomposition (S5): every element call run BOTH ways
LEGS_MODE=coarse python imap_legs_121.py                 # six coarse legs, ~3 min
python imap_legs_121.py                                  # every leg incl. the retrace

# the independent oracle (S6), ~5 min at 4096
IMAP_NFC=4096 IMAP_ORDERS="-4,-2;0,0" python imap_oracle_121.py

# the shipping banner, one arm per invocation
ARM_IMAP=0 python imap_banner_arm.py                     # the fail-before
ARM_IMAP=1 python imap_banner_arm.py                     # the feature
ARM_IMAP=0 ARM_NEWTON_ITERS=60 python imap_banner_arm.py # the control

# the new module's own battery, both mounts
python -m pytest tests/unit/test_niche_c15_inverse_map.py -q -p no:randomly
wsl -e bash -lc "cd /mnt/d/.../Lumenairy && ~/lumen_venv/bin/python -m pytest \
    tests/unit/test_niche_c15_inverse_map.py -q -p no:randomly"
```

Results of record are `_imap_afit.json`, `_imap_prod.json`,
`_imap_prod_census.json`, `_imap_legs.json` (the decomposition),
`_imap_legs_postscope.json` (the null control after the scoping) and
`_imap_oracle.json` (S6).  The directory's `.gitignore` covers `_*.npz` only,
so these are not ignored; they total well under 100 kB.

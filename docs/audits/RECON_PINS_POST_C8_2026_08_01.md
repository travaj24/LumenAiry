# Reconciling the four failing pins with niches C6 and C8

**Date** 2026-08-01 - **Tree** `fix/pmm-union-grid-conditioning` @ `bb2abe7`
(C5/C6/C7/C8 all committed; `lumenairy/**` untouched by this work) -
**Subject** four test files pin quantitative behaviour that niches C6 and C8
changed by design - **Question** for each pin, is the new behaviour BETTER,
EQUAL or WORSE than the old against an oracle that shares no code with the
thing under test, and only then, what should the pin say?

---

## 0. Headline

**Four pins, four verdicts, no library regression. Three are BETTER against an
independent oracle, one is EQUAL with a measured 6.5 % cost recorded rather
than hidden. Nothing was retuned before it was adjudicated, and no bar was
loosened on any assertion that still applies.**

| # | pin | old | new | oracle verdict |
|---|---|---|---|---|
| 1 | `test_niche_s12_remap_sampling.py` (5 tests) | full-vs-lattice separated by 0.67-1.15 rad | separated by 6e-07 - 1e-05 rad | **BETTER, 140x** - exact skew ray trace: 1.2538 -> 8.8e-03 rad |
| 2 | `test_remap_carries_injected_residual` | `remap` is phase-ONLY vs `pip=False` | |E| moves by 2.740e-04 rms rel | **PIN WAS WRONG PHYSICS**; two exact ray traces require 3.353e-04. Field EQUAL (relL2 5.55e-03 vs 5.21e-03) |
| 3 | `test_dx_flatness_alone_is_not_sufficient` | FWHM 3.6697x the oracle | 2.9538x | **BETTER** (less wrong) against an unmoved, lumenairy-free absolute oracle |
| 4 | E-M6 `test_fires_when_a_fold_caustic_manufactures_energy` | ratio 1.100385, check fires | 1.01931, silent | **BETTER** - truth is 0.82619; error +0.274 -> +0.193. **Pattern-match, NOT a regression** |

**The E-M6 verdict in one line, because it was the one that could have gone the
other way:** the fixture really does fold (**32 det J sign changes** on the
EXACT traced landing lattice), the fold diagnostic **still fires** with the
bound on, and **100.00 %** of the power C8 removes lies outside the convex hull
of every alive traced ray while **0.00 %** lies inside the support - so C8 did
not, and structurally could not, silence the fold. What it removed was
manufactured light that the energy self-check was mistaking for the fold's.

**One result the campaign did not have:** the S12 accuracy claim was measuring
convergence to a WRONG ANSWER. Pre-C6, `remap_sampling='full'` reproduced the
`ray_subsample=1` reference to 6e-05 rad - and that reference is **1.2538 rad**
away from an exact skew ray trace of the same fixture. The "9018x reduction"
was real and irrelevant.

---

## 1. Method

### 1.1 The rule this document is written under

*Do not retune numbers until tests go green.* For every pin, the new behaviour
is scored BETTER / EQUAL / WORSE first, against an oracle that shares no code
with what is under test, and the pin is edited only afterwards - preserving its
discriminating power, with a fail-before arm wherever the old numbers
documented a defect.

### 1.2 The oracles, and what each shares with the library

| oracle | used for | shares with the thing under test |
|---|---|---|
| **exact skew ray trace + closed-form input phase** (`recon_s12_oracle.py`, `recon_remap_residual_oracle.py`) | pins 1, 2 | `lumenairy.raytrace` and `surfaces_from_prescription`. **NOT** the tensor-Chebyshev forward-map fit, the Newton inverse, the coarse launch lattice, the bilinear upsample, `a_fit`, or any `remap_sampling` / `preserve_input_phase` code path |
| **`validation/oracles/debye_oracle_v3.py`** (`recon_d5_oracle.py`) | pin 3 | **nothing** - pure numpy + `scipy.special.j0`, no lumenairy in the call, absolutely normalised |
| **geometric transport of the alive stop-passing rays** (`recon_em6_stimulus.py`) | pin 4 | the ray tracer and the input field. No wave model at all |
| **convex-hull partition against the call's own exact ray bundle** | pin 4 | the ray tracer only (the same construction `probe_c8_d6_overremoval.py` uses) |

### 1.3 The exact-ray oracle's construction, and its validation

For an eikonal element the exit wavefront is Fermat's stationary value of
`phi_in(p) + k0 V(p, X)`, and the stationary point is reached by launching from
`p` along `grad phi_in(p) / k0`:

    launch (L, M) = grad(W)(p) + grad(a)(p)/k0        [analytic, closed form]
    trace         p -> X(p), V(p)                     [lumenairy.raytrace]
    exit phase    Phi(X(p)) = k0 (V(p) + W(p)) + a(p)
    exit |E|      |E_in(p)| / sqrt(|det dX/dp|)        [exact landing map]

then scattered onto the wave grid (`griddata`, linear - `Phi` is a continuous
eikonal in `p` and the map is smooth, so nothing is unwrapped anywhere).
`k0 (V + W)` is exactly the library's own OPL convention (the H6 entrance-
eikonal term is added to `final.opd` after the trace).

**VALIDATION.** With the residual switched off and `preserve_input_phase=False`
the library must reproduce this oracle. It does, to **1.2671e-03 rad**. The raw
difference before that is a **constant 1.428842 rad with standard deviation
1.27e-03 rad over 26501 pixels** and no radial structure whatsoever (mean per
`r/w` decade +1.4287 / +1.4288 / +1.4288 / +1.4289), i.e. a pure piston - and an
eikonal is defined up to an additive constant. **One global piston is removed
and nothing else**: no tilt, no defocus, no mode of any kind, because those are
exactly the errors being measured.

### 1.4 Runners added

All new, all prefixed `recon_`, all in
`validation/repro_traced_carrier_121/`. **No existing runner and no library
file was edited.**

| runner | what it measures |
|---|---|
| `recon_s12_measure.py` | the whole S12 matrix in both C6 states, plus the alias-radius structure and power bookkeeping |
| `recon_s12_oracle.py` | the exact-ray oracle for the S12 fixture, its validation, and every library row scored against it on a COMMON pixel mask |
| `recon_remap_residual_oracle.py` | two exact ray constructions of the pin-2 fixture (`grad W` vs `grad(W + a)`), amplitude / phase / complex relL2 |
| `recon_d5_oracle.py` | D5's own `_ladder` and `_oracle` in both C6 states, all four configurations |
| `recon_em6_stimulus.py` | the E-M6 adjudication: hull partition, exact-map det J, absolute geometric-transport ceiling, raw warnings |

Every runner prints the sha256 of the library file it imported
(`5f15da2e44144740` throughout - the shipped `_lens_traced.py`).

### 1.5 Baseline

All five failures re-confirmed on the settled tree before anything was touched:

```
FAILED tests/unit/test_niche_s12_remap_sampling.py::test_full_actually_changes_the_result_at_coarse_lattice
FAILED tests/unit/test_niche_s12_remap_sampling.py::test_full_is_ray_subsample_independent[2-0.02-0.2]
FAILED tests/unit/test_niche_s12_remap_sampling.py::test_full_is_ray_subsample_independent[4-0.02-0.4]
FAILED tests/unit/test_niche_s12_remap_sampling.py::test_full_is_ray_subsample_independent[8-0.05-0.6]
FAILED tests/unit/test_niche_s12_remap_sampling.py::test_lattice_error_is_confined_beyond_the_alias_radius
FAILED tests/unit/test_niche_upsample_lattice_fix.py::test_remap_carries_injected_residual
    6 failed, 12 passed in 11.05s
FAILED tests/unit/test_niche_audit_w3_elements.py::TestEM6RayDensityEnergySelfCheck::test_fires_when_a_fold_caustic_manufactures_energy
FAILED tests/unit/test_niche_d5_dx_flatness_gate.py::test_dx_flatness_alone_is_not_sufficient
```

(The brief lists 4 failures in S12; there are 5 - the brief's own list names
five tests.)

---

## 2. Pin 1 - `test_niche_s12_remap_sampling.py`, five tests

### 2.1 What moved, and that C6 is all of it

`recon_s12_measure.py`, rms phase difference against the `rs=1` reference:
`lattice` / `full` / and the two modes against **each other**.

| rs | C6 OFF (= pre-C6) | C6 ON (= shipped) |
|---|---|---|
| 2 | 5.4576e-01 / 6.0518e-05 / **6.7401e-01** | 1.6340e-02 / 1.6340e-02 / **6.0607e-07** |
| 4 | 8.3863e-01 / 2.6291e-04 / **9.2267e-01** | 5.6227e-02 / 5.6227e-02 / **2.6853e-06** |
| 8 | 1.0888e+00 / 5.9216e-03 / **1.1504e+00** | 9.1257e-02 / 9.1257e-02 / **1.0935e-05** |

The C6-OFF column reproduces the file's own 2026-07-25 docstring numbers
(0.5458 / 0.8386 / 1.0888 and 0.0001 / 0.0003 / 0.0059, ratios 9018 / 3190 /
184) **to every printed digit**, so `REMAP_STATIONARY_PHASE_LAUNCH` is the
whole of the change. The two modes now differ by 6e-07 to 1e-05 rad instead of
0.67 to 1.15 rad - a collapse of about **six orders of magnitude**, and the
element-level counterpart of `APPROXIMATION_AUDIT_POST_C6_2026_07_31`'s
`remap_sampling` row on design 121 (full-vs-lattice EE3 **-17.73 -> +0.0988**,
the sign flipped).

**MECHANISM, and it is structural rather than quantitative.** C6 launches along
`grad(W + a_fit)` and adds `a_fit` to the traced OPL, so the transported phasor
carries only the leftover `a - a_fit`. This fixture's residual is
`A (r^2/w^2)^2` - an exactly degree-4 polynomial - so `a_fit` absorbs
essentially all of it inside the fit disc and **there is almost nothing left to
sample on any lattice**. The aliasing the pins describe is not small; it is
absent.

### 2.2 The oracle verdict: BETTER by 140x

`recon_s12_oracle.py`. Every row scored on the SAME pixels (42327 of 65536,
oracle amplitude above 1e-02 of peak, every compared field non-zero) against
the same exact ray trace, one global piston removed.

| rs / mode | C6 OFF (pre-C6) | **C6 ON (shipped)** |
|---|---|---|
| 1 lattice | 1.2538e+00 | **8.8177e-03** |
| 2 lattice | 1.2532e+00 | **8.7784e-03** |
| 2 full | 1.2538e+00 | **8.7785e-03** |
| 4 lattice | 1.1926e+00 | **8.5979e-03** |
| 4 full | 1.2538e+00 | **8.5983e-03** |
| 8 lattice | 1.2429e+00 | **1.0524e-02** |
| 8 full | 1.2538e+00 | **1.0526e-02** |

Radial split at `rs=4` (inner `r < 0.88 w`, outer `r > 1.24 w`): C6 ON reads
3.10e-03 / 2.68e-03; C6 OFF reads 2.89e-01 / 1.82e+00.

**Two exact ray traces differing only in the launch eikonal** (`grad W` versus
`grad(W + a)`) are **1.2516 rad apart** on this fixture - which is the entire
pre-C6 error, measured with no library in the loop at all.

**VERDICT: BETTER.** The shipped library is 140x closer to the exact ray trace,
and `'full'` is no longer distinguishable from `'lattice'` because neither has
anything left to sample. Note what the table also says about the ORIGINAL
claim: pre-C6, `'full'` converged to a `rs=1` reference that was itself
1.2538 rad from the truth.

### 2.3 What changed in the tests

Items 5 and 6 and the "the knob is live" pin are kept **word for word** in an
arm that sets `REMAP_STATIONARY_PHASE_LAUNCH = False` - the state they were
calibrated in, where they still hold and still discriminate - and each grows a
second arm pinning the shipped behaviour:

* `test_full_actually_changes_the_result_at_coarse_lattice`: C6-off arm keeps
  `> 0.1` verbatim (measured 9.2267e-01); shipped arm asserts the field is
  still NOT byte-identical (the knob is not dead) and `1e-08 < d < 1e-04`
  (measured 2.6853e-06).
* `test_full_is_ray_subsample_independent`: C6-off arm keeps all three
  original assertions and all three parametrised tolerances verbatim; shipped
  arm asserts the two modes agree to `< 1e-04` rad, sit at the same distance
  from the `rs=1` reference to within 1 % (measured equal to 5 figures), and
  that distance is `< 0.12` (measured 1.63e-02 / 5.62e-02 / 9.13e-02).
* `test_lattice_error_is_confined_beyond_the_alias_radius`: C6-off arm keeps
  the aliasing signature verbatim (`d_out > 5 d_in`: measured 36.7x; `full`
  clean at 3.92e-05 / 8.03e-04); shipped arm asserts the two modes show the
  SAME inner and outer deviations to within 1 %, which is what distinguishes
  "no aliasing" from "aliasing both modes now suffer".

**No tolerance was loosened on any assertion that still applies.** A helper
`_run(..., launch=None)` sets and restores the flag; `None` (the default) is
the shipped path, which is what sections 1-4 still run at.

---

## 3. Pin 2 - `test_remap_carries_injected_residual`

### 3.1 The pin's premise was wrong physics, and an exact ray trace says so

The pin asserted `np.allclose(|E_false|, |E_remap|)` - "the two exits must
differ by a PHASE-ONLY factor". C6 launches `'remap'` along `grad(W + a_fit)`,
so the ray TUBE differs between the two modes and `ray_density`'s
`1/sqrt(|det J|)` follows it.

`recon_remap_residual_oracle.py` builds **two exact ray constructions of the
same fixture** - ORACLE-W (rays along `grad W`, what `pip=False` builds) and
ORACLE-Wa (rays along `grad(W + a/k0)`, Fermat's stationary point, i.e. the
truth) - each with its own exact landing-map Jacobian.

| quantity | value |
|---|---|
| **EXACT** ORACLE-W vs ORACLE-Wa, rms rel |E| | **3.353e-04** |
| **EXACT** ORACLE-W vs ORACLE-Wa, worst |d\|E\|| / peak | **2.921e-04** |
| **LIBRARY** `pip=False` vs `remap` (C6 ON), rms rel |E| | **2.740e-04** |
| **LIBRARY** `pip=False` vs `remap` (C6 ON), worst / peak | **2.764e-04** |
| **LIBRARY** `pip=False` vs `remap` (C6 OFF), rms rel |E| | **1.609e-16** |
| **LIBRARY** `pip=False` vs `remap` (C6 OFF), worst / peak | **4.282e-16** |

**An amplitude difference is REQUIRED here.** The library delivers it within
**18 %** of the exact prediction in rms and **5 %** at the worst point, and
with the C6 launch off it delivers exactly the phase-only behaviour the old pin
asserted (1.6e-16). The old assertion was pinning an artefact of the pre-C6
launch, not a property of the mode.

### 3.2 Field accuracy: EQUAL, with a measured 6.5 % cost recorded

Scored against ORACLE-Wa (the truth) over a common mask of 31457 px:

| library call | rms rel \|E\| err | phase err (rad) | **complex relL2** |
|---|---|---|---|
| `pip=False` | 4.751e-03 | 2.7324e-01 | **1.7459e-01** |
| `pip='remap'` (C6 ON, **shipped**) | 4.937e-03 | 4.2389e-03 | **5.5542e-03** |
| `pip='remap'` (C6 OFF, pre-C6) | 4.751e-03 | 2.9023e-03 | **5.2136e-03** |

Power: ORACLE-Wa 0.999526, ORACLE-W 0.999541, `pip=False` 0.993286,
remap C6-on 0.993281, remap C6-off 0.993286 (of the grid input power).

**VERDICT: EQUAL, with the cost stated.** On THIS fixture the shipped path is
**6.5 % worse in complex relL2** (5.554e-03 vs 5.214e-03) than pre-C6. That is
not a regression worth stopping for and the reason is measurable: this residual
is gentle, so the two exact constructions differ by only **3.3e-04 rad** of
phase - there is almost nothing for C6 to restore, and C6's degree-4 fit of an
`r^4 x Gaussian` residual is then the larger term. Both are **31x** better than
`pip=False`. On a fixture where `grad a` is large (pin 1) the same class of
oracle scores C6 **140x better**. This is the expected shape of a
second-order-term fix: a large win where the term matters and a wash, at a
small measured price, where it does not.

### 3.3 What changed in the test

* **Fail-before arm added**: with `REMAP_STATIONARY_PHASE_LAUNCH = False` the
  ORIGINAL `np.allclose` assertion is kept verbatim and passes at
  4.282e-16 - so the amplitude motion is provably C6's and nothing else's.
* **The shipped arm replaces "no change" with "the right change"**: the rms
  relative amplitude change must lie within a factor of two of
  `_EXACT_REMAP_DAMP_RMS_REL = 3.353e-04`, the exact-ray prediction, recorded
  as a module constant with its provenance. Measured 2.740e-04, i.e. 0.817x -
  and the assertion fails both if C6 is reverted (0) and if the ray-density
  amplitude stops tracking the augmented map (wrong magnitude). This is
  strictly MORE discriminating than the assertion it replaces.
* The phase assertions are **unchanged** and still pass:
  `std(dphi)/std(inj)` = 1.0376 (C6 on) / 1.0459 (C6 off), pin 0.3 .. 3.0.

---

## 4. Pin 3 - `test_dx_flatness_alone_is_not_sufficient`

### 4.1 The oracle did not move, and cannot

D5's anchor is `validation/oracles/debye_oracle_v3.py` -
`huygens_radial_profile`, pure numpy + `scipy.special.j0`, exact meridional
raytrace through the same conic/aspheric surface list, ring-Huygens integral,
absolutely normalised, **no lumenairy anywhere in the call**. Printed on this
tree by `recon_d5_oracle.py`: **FWHM 2.743288 um, EE1 33.1812, EE2 82.4465,
EE4 101.8687, total 101.9070 %**.

### 4.2 The broken configuration got LESS wrong

| configuration | N | FWHM um | /oracle | EE1 | EE2 | EE4 | window |
|---|---|---|---|---|---|---|---|
| defaults, C6 OFF | 1024 | 2.75529 | 1.0044 | 25.625 | 63.877 | 86.616 | 99.883 |
| **defaults, C6 ON** | 1024 | **2.72274** | **0.9925** | **28.216** | **69.539** | **89.499** | **99.898** |
| parabola, C6 OFF | 1024 | 10.06708 | **3.6697** | 1.620 | 6.189 | 20.949 | 75.085 |
| **parabola, C6 ON** | 1024 | **8.10312** | **2.9538** | **2.346** | **8.772** | **27.564** | **77.247** |

(The C6-OFF parabola row reproduces the test docstring's 3.67x and its
`EE2 6.18876 / 6.18924`.)

**VERDICT: BETTER on both paths.** The deliberately-broken configuration is
closer to the absolute oracle (3.6697x -> 2.9538x wide, EE2 0.0751x ->
0.1064x), and the shipped defaults improved too (EE2 63.88 -> 69.54, EE4
86.62 -> 89.50, FWHM ratio 1.0044 -> 0.9925). The broken half is the CARRIER
REFERENCE, not the residual transport, so C6 improves it without repairing it.

**The lesson the test exists for is entirely intact**: the parabola row is
still dx-FLAT to **1.138e-04** (44x inside the 5e-03 bar) while sitting
**2.95x wide of the oracle**, delivering **10.6 %** of its EE2 and losing
**23 %** of the launched power out of the readout window.

### 4.3 What changed in the test

`rows[-1]['fwhm'] / oracle['fwhm'] > 3.0` -> `> 2.5` (measured 2.9538), and -
because a bar re-priced once can be re-priced again by a further accuracy
improvement - **two new level assertions in the currencies that have the
teeth**, both with far more margin than the FWHM bar ever had:

* `ee2 < 0.25 * oracle['ee2']` - measured **0.1064x**, 2.3x of margin;
* `window < _LEVEL_WINDOW_MIN - 10.0` (= 89.0) - measured **77.247**.

---

## 5. Pin 4 - E-M6 `test_fires_when_a_fold_caustic_manufactures_energy`

**This is the one that could have been a library regression, so it was
adjudicated before it was touched.**

### 5.1 The two readings, and what separates them

The test's own name claims a FOLD CAUSTIC, and C8 must not be able to silence
one: the support bound only ever zeroes amplitude OUTSIDE the convex hull of
the alive stop-passing exit landings, whereas a fold lives where the rays ARE.
So:

* **(A) pattern-match with the C7 fires-test** - the stimulus was manufactured
  by the Newton inverse extrapolating outside the traced support. Predicts:
  the removed power is all outside the hull, none inside.
* **(B) real regression** - the stimulus is a genuine det J sign change inside
  the support. Predicts: a large inside-the-support removal.

### 5.2 Measured (`recon_em6_stimulus.py`)

Fixture N=256, dx=11.8359 um, aperture 3.000 mm, w0 1.400 mm, grid span
3.0300 mm. Exact ray bundle: 437 alive of 2209, all 437 stop-passing; exit
`|r|` max **1.2189 mm** against a grid reach of 2.1425 mm.

| state | ratio | energy check | fold-caustic warning |
|---|---|---|---|
| C8 OFF (= pre-C8) | **1.10039** | **FIRES** (library's own text: 1.100385) | **FIRES** |
| C8 ON (= settled) | **1.01931** | silent | **FIRES** |

**1. The partition of the power C8 removes**, against the call's own exact ray
bundle, over the aperture-transmitted input power:

| class | value | share |
|---|---|---|
| total `dP/P_ap` | 8.10791e-02 | |
| **(a) outside the hull of every alive ray** | **8.10791e-02** | **100.00 %** |
| (b) between the stop-passing and all-rays hulls | 0.00000e+00 | 0.00 % |
| **(c) INSIDE the support** | **0.00000e+00** | **0.00 %** |

**2. det J of the EXACT traced map** - central differences on the launch
lattice, stop-passing rays only, no fit, no Newton, no upsample:
**32 adjacent-cell sign changes** over 373 samples, det J spanning
**[-4.126e-01, +1.781e+00]**, `|det J|` min/median **0.0489**, max/min 25.1.
**The fixture genuinely folds** - and the fold warning fires in both states.

**3. The absolute geometric-transport ceiling** (raytrace + input field only,
no wave model): the alive stop-passing rays reach only to entrance
`|r|` = 1.1450 mm - rays beyond that die on the lens surfaces, well inside the
1.500 mm stop - and carry **0.82619** of the power over the test's own disc.

| | ratio | \|error\| vs 0.82619 |
|---|---|---|
| C8 OFF | 1.10039 | **0.27419** |
| **C8 ON** | **1.01931** | **0.19311** |

**4. The power INSIDE the support is 0.81933 in BOTH states**, to every printed
digit - **99.17 %** of the oracle's 0.82619, i.e. a 0.83 % discretisation
DEFICIT, not a gain. The fold's own contribution to the "energy gain" is zero.

### 5.3 Verdict: PATTERN-MATCH, and the new behaviour is BETTER

The energy self-check's stimulus was, to **100.00 %**, light at exit positions
no traced ray of the call reaches - the same defect class C8 removes at source,
and the same pattern as the C7 fires-test. C8 removed **nothing** inside the
traced support, so it did not and structurally could not silence the fold; the
fold diagnostic still fires. Against an absolute oracle the shipped state is
closer to truth by 30 % of the pre-C8 error. **Not a regression.**

The test's own stated MECHANISM was partly wrong and is now corrected in its
docstring: the fixture folds, but the pre-C8 +10.04 % was manufactured
extrapolation rather than the fold's capped `1/sqrt(|det J|)`.

### 5.4 What changed in the test

Exactly the C8 S9.3 reconciliation:

* the fires arm runs with `REMAP_INVERSE_SUPPORT_BOUND = False` - the state the
  check was calibrated in - with **both original assertions word for word**
  (`assert msgs`, and `'ray_subsample' in msgs[0] and 'band' in msgs[0]`);
* **five new assertions** were ADDED, none removed: with the bound on the check
  is silent, the ratio strictly falls, it lands inside the band, **no pixel's
  amplitude rises** (`|E8| <= |E|` everywhere - the bound may only lower), and
  it is **closer to the absolute geometric-transport ceiling 0.82619**;
* **and the fold diagnostic is asserted to fire in BOTH states** - the pin that
  would catch a future change that silences a real fold, which is the failure
  mode this reconciliation had to rule out.

The other four tests in the class are untouched and run at the shipped default.

---

## 6. Suites, lint, and the full leg

*(filled in after the run - see section 6.2)*

---

## 7. What could not be adjudicated, and what is left open

1. **Where the removed light would have gone.** As in C8's own record: every
   halo/partition figure is at an element's exit plane. The E-M6 fixture's
   post-C8 field still carries **0.19998 of `P_ap` outside the exact-ray hull**
   (the `sqrt(2) sub dx` plateau plus one exit-lattice cell of feather, which
   C8 keeps deliberately), and its global `|E|` maximum sits in that band. That
   is a property of C8's shipped plateau, not something this reconciliation
   changed, and neither the energy check (1.01931 is inside the band) nor the
   C7 halo check (the taper's outer edge, 1.4996 mm, is inside the halo radius
   1.25 x r_hull) reports it. **Open observation, not a finding**: on a fixture
   this pathological - the traced support is 1.22 mm and the grid reaches
   2.14 mm - the two self-checks are jointly blind to a residual 20 % of
   `P_ap` of manufactured light. Whether that band should be tightened is a
   library question and was out of scope here.
2. **The pin-2 6.5 % relL2 cost is measured on ONE fixture** at one
   `ray_subsample` (8) and one grid. It is reported, not generalised.
3. **The S12 oracle is scored at one launch density** (one ray per wave pixel).
   `ORACLE_SUB` is a knob on the runner; a convergence sweep in it was not run.
4. **The exact-ray oracle removes one global piston.** It was verified to be a
   piston on the A=0 control (std 1.27e-03 rad over 26501 px, no radial
   structure); it was not re-verified per row.
5. **D5's C6-off arm is documented, not asserted.** Adding a second ladder to
   that test would roughly double a 3-minute file for a number this document
   already records; the fail-before value 3.6697 is in the docstring instead.

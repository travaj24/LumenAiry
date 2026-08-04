# Closing design 121: what was instrument, what was the library, what is left

**Date** 2026-08-02 - **Tree** `feat/d121-final-closure`, cut from released
`main` = **v5.32.0** @ `999b9e6`, plus this study's uncommitted change to
`lumenairy/propagators/carrier.py` - **Subject** the last ~1.3 EE3 points per
order between the traced-carrier chain and the exact-ray oracle on design 121 -
**Question** how much of that gap is real?

---

## 0. Headline

**Roughly a quarter of the gap was the instrument, two thirds was one library
defect, and what remains is 0.13-1.0 points that is real, per-order, and does
not converge away.**

The brief's numbers are `probe_c6_chain.py`'s: at the chain's group-5 exit,
(0,0) **89.21** vs oracle **90.08**, (-4,0) **88.94** vs **90.78**, (-4,-2)
**88.49** vs **89.78**.  Reproduced on this tree to 0.01 points.  Three things
were wrong with that comparison and one thing was wrong with the library.

1. **The oracle was scoring a different input field.**  `hybrid_localize_121`'s
   `n = 0` arm launches `|env_doe|` on a PERFECT sphere -- it throws away the
   DOE-plane field's own residual phase, which the chain carries.  That
   residual is only 0.0012-0.0084 waves rms, but it is the pre-DOE half of a
   design optimised end to end, so it PRE-COMPENSATES the relay: the true
   ceiling for the field the chain is actually handed is **0.14-0.66 points
   HIGHER** than the one every table in this campaign quotes.  The gap was
   understated, not overstated (S3.2).

2. **The readout's launch-phase split cost 0.34-0.47 points, and it is
   provably instrument.**  The chain arm launches rays along
   `grad(parabola) + grad(residual)`, with the residual finite-differenced.
   Splitting the SAME field against the exact niche-C5 eikonal instead leaves
   the launch phase `ph0` identical to **9e-12 rad** over the bright rays while
   moving launch DIRECTIONS by up to **1.97e-02 in direction cosine**.  Same
   field, same phase, different derivative -- and 0.4 EE3 points (S3.3).

3. **The chain arm's headline sampling alarm was an artefact.**  `rs_spot`
   prints an integrand step of **1.31 cycles** (p99.9) against its own 0.25
   bar.  It differences a 2-D-UNWRAPPED launch phase; the RS kernel only ever
   consumes `exp(i*ph0)`, so the unwrap's jumps cannot reach the result.  The
   honest wrapped statistic is **0.0389 cycles** -- better than the oracle
   arm's 0.0518 (S3.1).

4. **THE LIBRARY DEFECT: `_sphere_parab_conversion`'s `cos^2` band-limit taper
   is worth 0.96-1.11 EE3 points on every order, and it should not exist.**
   The taper falsifies the field pointwise in the annulus where it acts, and
   the one piece of counter-evidence on record -- *"the untapered swap breaks a
   coarse chain"* -- is a **mis-citation of a measurement that says the
   opposite** (S4.2).  Shipped as `SPHERE_PARAB_CONVERSION_EXACT = True`
   (niche C9); fail-before `= False` reproduces v5.32.0 bit for bit.

### The answer, per order

| order | shipped, as the brief states it | INSTRUMENT: ceiling | INSTRUMENT: readout | **LIBRARY: the taper (fixed)** | **REAL residual** |
|---|---|---|---|---|---|
| (0,0)   | 89.201 vs 90.081 = 0.880 | +0.661 | +0.461 | **+1.032** | **0.049** |
| (-1,0)  | 88.404 vs 90.206 = 1.802 | +0.582 | +0.338 | **+1.111** | **0.935** |
| (-2,0)  | 88.633 vs 90.499 = 1.865 | +0.379 | +0.469 | **+1.001** | **0.775** |
| (-3,0)  | 89.282 vs 90.742 = 1.460 | +0.164 | +0.379 | **+0.968** | **0.278** |
| (-4,0)  | 88.978 vs 90.791 = 1.813 | +0.138 | +0.407 | **+0.956** | **0.588** |
| (-4,-2) | 88.527 vs 89.783 = 1.255 | +0.242 | +0.376 | **+0.997** | **0.125** |

Read a row as: the gap to the TRUE ceiling is the sum of the last four columns.
"Ceiling" is the oracle correction of item 1 -- it makes the gap BIGGER;
"readout" and "the taper" are recovered; the residual is what is left.

**X was instrument, Y was the taper, Z is physical:**

* **X = 0.34-0.47 points** of measurement error in the readout, removed by
  splitting the launch against the exact eikonal -- plus a ceiling that was
  0.14-0.66 points too LOW, also instrument, which had been making the gap look
  smaller than it was.
* **Y = 0.96-1.11 points**, the conversion taper.  Fixed and shipped, scored on
  both currencies: EE3 up on every order, the halo `g4` **3-676x smaller** and
  `amax4` **2-14x smaller** on every order, `P_out/P_in` within 4.1e-05, and
  the at-plane acceptance **3.450 um -> 3.350 um / EE3 90.2 -> 90.3**.
* **Z = 0.049-0.935 points at the campaign's own configuration**
  (`RN=1024`, `ray_subsample=4`), and **0.135-1.008 under refinement** -- the
  refined figure is the honest one, because on axis the coarse reading is the
  luckier of the two (S6.2).  It is real: the instrument's band at the
  converged readout is **+-0.005 points** (S3.5), so even the smallest of the
  six exceeds it by 10x and the largest by 190x.  And it is **stable under
  both discretisation refinements available** -- halving `ray_subsample` and
  doubling the grid move it by <= 0.10 points and neither shrinks it (S6.2),
  so it is a MODEL error rather than a discretisation one.  It lives in the
  halo, not the core: the chain's FWHM matches the exact ray trace's to
  0.02-0.06 um on every order while EE6 runs 0.02-0.57 points low.  And it is
  NOT the leg -- the leg was the taper.

---

## 1. Provenance, instruments, floors, sampling

### 1.1 What was measured, against which library

| file | sha256 (16) | state |
|---|---|---|
| `lumenairy/propagators/carrier.py` | `1a90453a4ef65399` | **v5.32.0 / `999b9e6`** -- every "before" number |
| `lumenairy/propagators/carrier.py` | `5a1b0d1021969df1` | **after** -- the only library file this study changes |
| `lumenairy/elements/_lens_traced.py` | `9717ad88dd959889` | **unmodified**, both states |

`CHANGELOG.md` and `lumenairy/elements/pmm/**` were not touched.  Every runner
prints the version, path and both hashes of the library it imported.

**A LIVE TRAP, found and disarmed before any number was quoted.**
`approx_common.py` DEFAULTS `LUMEN_PIN` to a frozen **v5.31** export in the
2026-07-31 study's scratchpad, and that directory still exists on this box.
Any script importing `approx_common` (or `approx_ablate_121`) without setting
`LUMEN_PIN=0` first silently measures v5.31 -- confirmed by import: it resolves
`lumenairy 5.31.0 @ <scratch>/pin_d2e60ca/...` and
`REMAP_INVERSE_SUPPORT_BOUND` does not exist there at all.  Every script in
this study sets `LUMEN_PIN=0` before that import and prints a provenance
banner; the three baseline numbers were re-run after the fix and reproduce
bit for bit.

### 1.2 Instruments

| instrument | what it measures | new? |
|---|---|---|
| `fc_instrument_121.py` | ONE arm (oracle or chain) through `hybrid_localize_121.rs_spot` VERBATIM, with every route knob exposed and an honest WRAPPED sampling statistic | **new** |
| `fc_table_121.py` | the six-arm per-order table of S0, in one process | **new** |
| `fc_taper_locality.py` | is the taper's effect inside or outside its own onset, per group | **new** |
| `fc_taper_census.py` | every `_sphere_parab_conversion` call with its geometry, plus a one-call-at-a-time ablation | **new** |
| `fc_with_taper.py` | run any existing runner with the taper / the C9 flag forced, without editing it (the `c8_with_bound.py` device) | **new** |
| `fc_c9_byte_identity.py` | the fail-before contract against a `git archive HEAD` export, in a separate process | **new** |
| `fc_sampling_121.py` | amplitude-weighted wrapped nn-step p99.9 per arm | **new** |
| `fc_production_taper.py` | the taper through the LIBRARY's own production readout, with no ray launch of this study's | **new** |
| `hybrid_localize_121.py`, `focus_scan_121.py`, `energy_stage_audit_121.py` | reused, **unedited** | reused |

**No existing runner was edited.**

### 1.3 Differential floors -- bit-exact, established before any delta

| instrument | null intervention | reading |
|---|---|---|
| `fc_instrument_121.py`, oracle arm | two identical runs | **identical**, sha256 of the intensity `f53dbc32bfbd2372` twice |
| `fc_instrument_121.py`, chain arm | two identical runs | **identical**, sha `863ecd785bce19d8` twice |
| `energy_stage_audit_121.py` | two identical chain runs, all six stages | **`array_equal=True`, `max\|dE\| = 0.000e+00`**, all four orders, both flag states |
| `fc_c9_byte_identity.py` | flag OFF vs a `git archive HEAD` export, separate process | S7.1 |

Every delta below is against a floor of exactly zero.

**And the two routes to the same intervention agree bit for bit.**  Every
number taken before the library change used a MONKEYPATCH of
`_sphere_parab_conversion`; every number after uses the shipped flag.  Run both
ways, the chain arm returns the same intensity array to the last bit
(`863ecd785bce19d8` tapered, `8db002a1c1bd58ef` exact, on axis), so the two
halves of this document are on one scale.

### 1.4 Sampling adequacy

Stated as the **amplitude-weighted wrapped nearest-neighbour step at p99.9
against pi**, never a max -- `DIAG_LAST_GROUP_DECENTRE` artefact 4 is exactly
the max-quoting trap, and `_phase_gradient` returns a max.

Two quantities matter and they are different: the **ENVELOPE** step, of the
object whose gradient becomes the launch direction (if it is not `<< pi` the
directions are meaningless), and the **INTEGRAND** step of the RS quadrature,
WRAPPED.  `fc_sampling_121.py`, pi = 3.1416:

| order | arm | env p50 | **env p99.9** | env max | RS p99.9 (cycles) |
|---|---|---|---|---|---|
| (0,0) | oracle | 0.00000 | **0.00196** | 0.00272 | 0.0259 |
| (0,0) | chain, SHIPPED (taper, parabola split) | 0.00064 | **0.15852** | 0.16846 | 0.0389 |
| (0,0) | chain, taper off, parabola split | 0.00033 | 0.10406 | 0.10485 | 0.0389 |
| (0,0) | **chain, taper off, exact split** | 0.00010 | **0.00628** | 0.00636 | 0.0389 |
| (-4,-2) | oracle | 0.00000 | **0.00196** | 0.00272 | 0.0259 |
| (-4,-2) | chain, SHIPPED | 0.00369 | **0.18117** | 0.22272 | 0.0385 |
| (-4,-2) | **chain, taper off, exact split** | 0.00026 | **0.01187** | 0.01556 | 0.0384 |

**What this permits and forbids.**  No arm's amplitude-weighted p99.9 comes
near pi, so no result here is alias-DOMINATED, and the RS quadrature is
comfortably inside its own 0.25-cycle bar everywhere.  What the shipped arm
does have is a small population of bright pixels whose residual gradient is
large: `_phase_gradient`'s MAX reads **3.1355 rad = pi** on it against
**0.0322 rad** on the converged arm, a factor of 97, and it is those pixels
that carry the 1.97e-02 direction-cosine error of S3.3.  The two statistics
disagree because the defect is localised, not because either is wrong -- both
are quoted throughout and neither is used alone.

---

## 2. Why the taper IS the leg remnant, and where it acts

`fc_taper_locality.py` runs the chain twice -- taper on, taper off -- truncated
after each group in turn, and partitions `||E_on - E_off||^2` against the FINAL
conversion's own onset, inside which BOTH states use the same convention by
construction.  Order (0,0):

| groups | `R_out` mm | `dx` um | `w` um | `r_safe` mm | onset/w | `dP/P` | INSIDE onset | OUTSIDE | P inside onset |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 703664.79 | 51.234 | 6318.6 | 207302 | 24606 | 0.000e+00 | 0 | 0 | 1.000000 |
| 2 | 703671.92 | 51.234 | 6318.7 | 207304 | 24606 | 0.000e+00 | 0 | 0 | 1.000000 |
| 3 | -263.194 | 51.235 | 5982.4 | 77.538 | 9.721 | 0.000e+00 | 0 | 0 | 1.000000 |
| 4 | -60.148 | 44.912 | 4844.4 | 18.515 | 2.866 | 1.132e-42 | 0 | 1.1e-42 | 1.000000 |
| 5 | -24.463 | 38.432 | 3629.2 | 7.932 | 1.639 | 5.070e-11 | 4.7e-14 | 5.1e-11 | 0.994775 |
| **6** | **-7.712** | **33.211** | **1185.8** | **2.625** | **1.661** | **8.594e-03** | **2.2e-05** | **8.6e-03** | 0.995486 |

**Groups 1-5 are bit-identical.**  The whole effect appears at group 6, and
**99.74 % of it lies outside the onset** -- in the annulus beyond 1.66 w, which
carries 0.45 % of the exit power.  The taper does not perturb the core: it
scrambles the PHASE of the skirt, and the skirt then interferes with the core
in the image.  (0.45 % of the power at a scrambled phase is a 6.7 % amplitude
term, so a +-13 % intensity cross-term with the core -- comfortably enough for
1 EE3 point.)

`fc_taper_census.py` logs all 13 conversion calls of a run and ablates them one
at a time.  Nine are inert (onset at 3.0-24607 beam radii).  The two that
matter are

* call 9, `sign=-1`, `R = -24.4625 mm`, `dx = 38.432 um`, onset **1.64 w** --
  **group 5's EXIT re-envelope**, i.e. the object the 3.3233 mm leg into group
  6 then Sziklas-Siegman-transports; and
* call 10, `sign=+1`, `R = -21.1392 mm`, `dx = 33.211 um`, onset **1.42 w** --
  group 6's ENTRANCE, the field handed to `apply_real_lens_traced`.

**That is exactly where niche C5 left its -0.96-point leg remnant.**  The
brief's candidate list named "the exit re-envelope step" and "the
`_sphere_parab_conversion` taper" as two separate suspects.  They are the same
suspect, and it is the taper.

**And the calls are a matched chain -- the fix cannot be a per-call
heuristic.**  Ablated ONE AT A TIME, on axis, against the shipped 89.661:

| call ablated | dEE3 |
|---|---|
| calls 0-8 (onset at 3.0-24607 w) | **+0.0000** each, to four decimals |
| call 9, group 5 EXIT | +0.459 |
| call 10, group 6 ENTRANCE | **-0.046** |
| call 11, group 6 EXIT | **-0.242** |
| call 12, the chain's FINAL `+1` | **-0.087** |
| **all of them** | **+1.032** |

Three of the four live calls are WORSE alone than left alone, the best single
ablation recovers 0.459 of the 1.032 available, and only removing the taper
everywhere gets all of it.  A hand-off needs the entrance and exit conventions
to AGREE; that is a property of the SET, not of any call, which is why the fix
is a default and not a threshold.

---

## 3. The instrument's own error band

### 3.1 The alarm that was not real

The chain arm of `hybrid_localize_121` prints

```
envelope per-pixel phase step 3.1355 rad (ALIASED -- launch directions unreliable);
integrand step p50/p99.9/max 0.0016/1.3121/1.85 cycles (CHECK)
```

The integrand statistic is computed from `phres = unwrap(unwrap(angle(envk)))`.
The RS kernel is `exp(1j*(ph + K0*rho))` -- it consumes `ph0` only through
`exp(i*ph0)`, so a 2*pi jump is invisible to the RESULT and dominant in the
DIAGNOSTIC.  Differenced WRAPPED, the same launch reads

| arm | integrand p50 | **p99.9** | max |
|---|---|---|---|
| oracle (`n = 0`) | 0.0033 | **0.0518** | 0.0521 |
| chain (`n = 6`), as printed | 0.0016 | *1.3121* | *1.85* |
| chain (`n = 6`), **wrapped** | 0.0005 | **0.0389** | 0.0395 |

**The RS quadrature was never undersampled** -- the chain arm is better sampled
than the oracle arm.  This is `DIAG_LAST_GROUP_DECENTRE` artefact 7, on a
different instrument, and it very nearly sent this study down the wrong road
(S10 item 6).

### 3.2 The oracle was scoring the wrong field

`hybrid_localize_121`'s `n = 0` arm does

```python
amp = _bilinear(np.abs(env_doe), dx_doe, x0, y0)     # MAGNITUDE only
ph0 = K0 * (L*x0 + M*y0 + sign(R)*(den - abs(R)))    # a PERFECT sphere
```

so it discards `arg(env_doe)`, which the chain carries.  Measured, that
residual is **0.0012 waves rms** over the >50 %-of-peak core and **0.0084
waves** over the >2 % core -- small, but this is a relay whose two halves were
optimised together, so it pre-compensates.  Carrying it moves the ceiling:

| order | oracle, phase discarded | **oracle, phase carried** | delta |
|---|---|---|---|
| (0,0) | 90.081 | **90.742** | +0.661 |
| (-1,0) | 90.206 | **90.788** | +0.582 |
| (-2,0) | 90.499 | **90.878** | +0.379 |
| (-3,0) | 90.742 | **90.907** | +0.164 |
| (-4,0) | 90.791 | **90.929** | +0.138 |
| (-4,-2) | 89.783 | **90.025** | +0.242 |

Converged: launch density 161 / 241 / 321 / 481 / 741 reads
90.7413 / 90.7427 / 90.7421 / 90.7428 / 90.7429 on axis -- **spread 0.0016
points** -- with the wrapped integrand step falling exactly as 1/NL
(0.0519 -> 0.0112 cycles).  At NL = 741 the launch pitch equals `dx_doe`, so
the residual is not being subsampled at all.

**Independently corroborated on the other arm.**  Feeding the CHAIN the
phase-stripped field instead (the same field the discarding oracle launches)
costs the chain **-0.685 points** on axis (90.693 -> 90.009), against the
oracle's own **+0.661**.  Two routes, one physical statement.  And on that
MATCHED field the chain reads **90.009** against the matched oracle's
**90.081**: **0.072 points apart**, i.e. the on-axis comparison very nearly
closes outright once both arms see the same light.

**One artefact killed on the way.**  `exact_ray_oracle_121.oracle_spot`'s own
`carry_phase=True` arm bilinear-interpolates a GLOBAL 2-D unwrap of `env_doe`.
95 % of that grid is dark skirt whose phase is numerical noise, and
`np.unwrap` along a row carries it across the beam.  Measured, that arm reads
EE3 **49.6 / 9.9 / 52.0 / 28.0** at NL = 321 / 481 / 741 / 981 -- no
convergence at all, with a wrapped integrand step of 0.33-0.42 cycles.
Interpolating the unit PHASOR instead (exact here: the residual's own step is
0.0049 rad) gives the converged column above.  **That option is unsafe as it
stands; nothing in this campaign has used it, and it is left untouched.**

### 3.3 The readout's launch-phase split, and the proof that it is instrument

The chain launch splits the field's phase into an ANALYTIC part (closed-form
gradient) and a RESIDUAL (wrapped central difference).  The shipped split is
the parabola plus the tilt ramp, which leaves the whole sphere-minus-parabola
quartic -- and, with the taper on, an aliased piece of it -- in the
finite-differenced residual.

Splitting against the exact C5 eikonal instead
(`W = sign(R)(sqrt((u + R L/N)^2 + (v + R M/N)^2 + R^2) - |R|/N)`, whose
gradient is the congruence's own direction cosines) is a **pure
re-partition**.  Over the bright rays (>5 % of peak):

| order | max wrapped `\|ph0_exact - ph0_parabola\|` | max `\|p_exact - p_parabola\|` (direction cosine) | mean |
|---|---|---|---|
| (0,0) | **3.9e-13 rad** | **1.97e-02** | 7.2e-04 |
| (-4,-2) | **6.3e-13 rad** | **1.97e-02** | 6.0e-04 |

Same field, same launch phase to machine precision, launch directions different
by up to 19.7 mrad -- against a 3 um spot at 7.7 mm, which subtends 0.4 mrad.
It moves EE3 by **+0.34 to +0.47 points**.  That is measurement error, not
physics.

At `L = M = 0` the exact split reduces to the sphere split analytically, and
does numerically: both read EE3 90.6934 / FWHM 3.442 / EE6 99.8777 to every
printed digit.

### 3.4 The knobs that were already converged

| knob | arm | sweep | EE3 spread |
|---|---|---|---|
| launch density `NL` | oracle | 121, 161, 221, 301, 401 | **0.000** (90.08 throughout; integrand step falls as 1/NL) |
| crop radius `CLIP` | chain, shipped readout | 2.0, 2.5, 3.0, 3.5, 4.0, 5.0 | **0.01** (89.19-89.20) |
| Fourier upsample `UP` | chain, shipped readout | 1, 2 | **0.34** (89.20 -> 89.54) |

`UP` is the one that moved -- and it moved because the shipped split's
finite-differenced residual is where the error is, so refining the lattice
changes the directions.

### 3.5 The band at the converged readout

Re-run at the exact split with the taper off, on axis:

| knob | sweep | EE3 |
|---|---|---|
| `CLIP` | 1.5 / 2.0 / **2.5** / 3.0 / 3.5 / 4.5 | 89.476 / 90.688 / **90.693** / 90.693 / 90.693 / 90.693 |
| `UP` | 1 / 2 | 90.6934 / 90.6949 |

**Band: +-0.005 points** for `CLIP >= 2.5` (the campaign uses 3.0), against
+-0.34 before.  The envelope MAX step falls **3.1355 -> 0.0322 rad** and the
wrapped integrand step halves under `UP=2` as it should.

**So the honest instrument band is +-0.005 points, not +-0.7.**  The brief's
hypothesis that part of the residual might be instrument is therefore
**REFUTED for the residual** (the smallest of the six exceeds the band by
10x and the largest by 190x) and **CONFIRMED for 0.34-0.47 points of the
original gap plus the whole ceiling correction**.

---

## 4. The taper

### 4.1 The mechanism, stated so the fix can be judged against it

Under the shipping `carrier_reference='sphere'` the chain STORES and transports
the exact-sphere-referenced envelope `env_S = E exp(-i k S)`, which is the
aberration residual and is smooth.  Before an element call it reconstructs

```
E_full = env_S * exp(i k r^2/2R) * exp(i k (S - r^2/2R) * T(r))
```

With `T == 1` that is `env_S exp(i k S)` -- the physical field, exactly.  The
element then de-chirps against the SAME exact sphere (`carrier=R`, or
`TiltedCarrier`), recovering `env_S`.  **The conversion and the element's own
reference are an identity pair.**

With `T < 1` the element instead recovers `env_S * exp(-i k (S - r^2/2R)(1-T))`
-- a spurious quartic whose own phase slope is at or past the grid's Nyquist by
construction, since that is what `r_safe` means.  The same happens in reverse
at the exit, where the mixed object is then transported across the next gap.

So on this path the taper protects nothing and falsifies the field where it
acts.  It has a surviving argument in exactly one place -- a consumer that FFTs
or RESAMPLES the CONVERTED field, i.e. the PARAXIAL focus readout -- and that
is what the (unchanged) `w_beam` guard now says.

### 4.2 The counter-evidence on record is a mis-citation

`_sphere_parab_conversion`'s docstring said the untapered swap "breaks a coarse
chain", and `APPROXIMATION_AUDIT_POST_C6` S6 item 10 correctly flagged that as
"a single sentence with no reproduction attached ... the cheapest open item in
this document".  Re-derived: the source is
`AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24` S6.6, which says

> (The taper worked as designed -- stage traces identical to the whole-grid
> swap to 4 digits, i.e. the guard band truly carries nothing -- so the
> breakage is in-band and intrinsic, not an aliasing artifact.  The conversion
> runs' spot is walked AND genuinely blurred; the narrow-window 7.1 % was the
> walk clipping the readout corner.)

Three things follow.  (a) At that tree the tapered and untapered conversions
were measured **identical to 4 digits** -- the opposite of "the untapered swap
breaks it".  (b) What broke that chain was the CONVERSION ITSELF, in the
pre-`ray_density` era; the conversion has been the shipped default since v5.29
and `amplitude_model='ray_density'` is the fix that audit landed.  (c) The
"77.5 % -> 7.1 %" figure is a spot walking out of a narrow readout window, and
the same paragraph says so.

The taper WAS inert then and is not now, for a measurable reason: that audit's
own plan records `r_safe/w` = 10.5 / 3.6 / 5.1 / **2.6** / 5.5 across the
planes, i.e. an onset at 1.95 w at worst.  Today the last two planes sit at
**1.42 w and 1.66 w** (S2), and the field around them is far better corrected,
so the annulus's phase matters where it did not before.

### 4.3 What the change is worth, per order

`fc_table_121.py`, EE3 at the chain's group-5 exit, read out against the exact
eikonal so the instrument is not part of the delta:

| order | taper ON | **taper OFF** | delta | delta through the SHIPPED (parabola) readout |
|---|---|---|---|---|
| (0,0)   | 89.661 | **90.693** | **+1.032** | +1.081 |
| (-1,0)  | 88.742 | **89.853** | **+1.111** | +1.078 |
| (-2,0)  | 89.102 | **90.103** | **+1.001** | +1.100 |
| (-3,0)  | 89.661 | **90.628** | **+0.968** | +1.009 |
| (-4,0)  | 89.385 | **90.342** | **+0.956** | +1.055 |
| (-4,-2) | 88.904 | **89.900** | **+0.997** | +1.011 |

The last column matters: the taper's value is the same whichever readout scores
it, so the taper fix and the readout fix are independent rather than
compensating.

**FWHM falls on every order too** (on axis 3.657 -> 3.442 um; at (-4,-2)
3.596 -> 3.383), and the envelope sampling statistic falls with it (S1.4) --
which is the mechanism showing itself: what the taper added was aliased phase.

### 4.4 `T == 1` is the SATURATED END of a monotone axis, not a lucky point

The strongest structural argument for the fix is that the answer is converged
in the taper RADIUS.  Sweeping `r_safe` by a scale factor -- the shipped
`cos^2` shape and everything else untouched -- at (-4,-2), exact split:

| `r_safe` scale | EE3 | FWHM um | note |
|---|---|---|---|
| x 0.5 | **42.813** | 7.627 | the 41-point cliff `APPROXIMATION_AUDIT_POST_C6` S2 records, confirmed post-C8 |
| **x 1.0** | **88.9036** | 3.596 | **byte-identical to the shipped taper** (sha `55b92b4d9f187774`) |
| x 1.5 | 89.9005 | 3.383 | |
| x 2.0 | 89.9005 | 3.383 | |
| **x 3.0** | **89.9005** | **3.383** | **byte-identical to `T == 1`** (sha `5e8550468cb6061b`) |
| **T == 1** | **89.9005** | **3.383** | the shipped C9 state |

Two bit-exact nulls in one table: `x1.0` reproduces the shipped taper and
`x3.0` reproduces no taper at all, each to the last bit of the returned
intensity.  So the axis is **monotone and saturating**, the taper has stopped
touching the result by 1.5x its radius, and `T == 1` is where it converges --
not a point chosen because it happened to score well.  The `x0.5` cliff is
unchanged, which is the same statement from the other side: this radius has a
41-point failure mode below it and a plateau above it, and the shipped value
sat on the shoulder.

---

## 5. Acceptance

### 5.1 The shipped 121 acceptance, through the LIBRARY's own readout

`focus_scan_121.py`, pure library defaults (`CREF`/`AM`/`PIP` unset), N=2048,
`rs=4`, NFC=8192, WF=4.0, NOUT=2048.  No ray launch of this study's is involved
anywhere: this is the exact fine retrace plus the Bluestein readout.  **This is
the independent confirmation that the change improves the FIELD and not merely
this study's own instrument.**

| | recorded acceptance (taper on) | **measured, taper off** |
|---|---|---|
| `BEST-FOCUS[peak]` plane | dz = +0 um | **dz = +0 um** |
| FWHM | 3.450 um | **3.350 um** |
| EE3 | 90.2 % | **90.3 %** |
| EE6 | 99.7 % | **99.7 %** |
| EE12 | 99.8 % | **99.8 %** |
| peak | 5.471e+03 | **5.516e+03** (+0.8 %) |

**No plane of the +-80 um through-focus scan is worse**, the plane of best
focus does not move, and the EE6-selected line is unchanged
(dz = +10 um / 3.650 um / 85.3 -> 85.4).  The acceptance therefore does not
regress; it improves, and the recorded line moves to
**3.350 um / 90.3 / 99.7 / 99.8**.

The production gain (+0.1 EE3) is much smaller than the diagnostic
configuration's (+1.03), and that is understood rather than surprising: with
`final_leg='exact'` the LAST group is re-traced from its entrance on a ~1.5 um
fine grid, where `r_safe` exceeds the whole grid and the taper is inert either
way.  Production was already half-protected; the paraxial route, and every
per-order oracle comparison this campaign quotes, was not.

### 5.2 Conservation and halo -- the other currency

`energy_stage_audit_121.py` (unedited) via `fc_with_taper.py`, `RN=1024`,
`rs=4`, six post-DOE groups, `final_distance=0`, `final_leg='paraxial'`.
NULL intervention `array_equal=True`, `max|dE| = 0.000e+00` on all six stages,
every order, both flag states.

| order | metric | v5.32.0 (taper) | **C9 (exact)** | verdict |
|---|---|---|---|---|
| (0,0) | `elem(5)` | 0.995971 | 0.995930 | deficit ratio 1.01x -- **C1b PASS** |
| | end to end | 0.994355 | 0.994314 | **C2 PASS** |
| | `g4` / `amax4` | 0 / 0 | **0 / 0** | **C3, C4 PASS** |
| | `r_rms` | 0.8385 | **0.8382** | exact-ray 0.8407, dev -0.30 % -- **C5 PASS** |
| (-2,0) | `g4` | 5.370e-09 | **7.947e-12** | **676x smaller**; 0.0001x of the C3 bound |
| | `amax4` | 1.611e-04 | **1.148e-05** | **14x smaller** |
| | end to end | 0.994131 | 0.994129 | unchanged |
| (-4,0) | `g4` | 2.628e-08 | **8.479e-09** | **3.1x smaller**; 0.037x of bound |
| | `amax4` | 2.368e-04 | **9.885e-05** | **2.4x smaller** |
| | end to end | 0.993992 | 0.994007 | +1.5e-05 |
| (-4,-2) | `g4` | 2.653e-08 | **8.653e-09** | **3.1x smaller**; 0.039x of bound |
| | `amax4` | 2.335e-04 | **1.147e-04** | **2.0x smaller** |
| | end to end | 0.993816 | 0.993839 | +2.3e-05 |

Bounds as `ENERGY_CONSERVATION_AUDIT` S6 states them (C1a `P_out/P_ap` in
[0.9900, 1.00020]; C1b last-group deficit >= 0.5x the C6-off deficit; C2 end to
end in [0.9850, 1.00050]; C3 `g4 <= 3 g4_exact`; C4 `amax4 <= 1.0e-03`;
C5 `|dr_rms| <= 0.030`).  **Every order scores 6 of 6 in both states, and the
halo is strictly quieter in the shipped one.**  EE alone is proven blind; here
the halo currency points the same way.

The niche suite emitted **zero** halo self-check firings
(`grep -c "HALO self-check FAILED"` -> 0 over 355 tests).

### 5.3 The production readout on a TILTED order

`fc_production_taper.py` -- `approx_post_c6.py`'s row machinery reduced to the
rows this study needs, on the COMPLETE shipped path (chain A ->
`TiltedCarrier(order)` -> six post-DOE groups -> the 7.7058 mm trailing leg ->
`final_leg='exact'` -> the exact Bluestein readout), `RN=1024`, `rs=4`,
`n_fine_cap=12288`, `window_factor=4.0`, `N_out=192`, `dx_out=0.1 um`, fixed
output lattice on the order's exact chief ray.  **No ray launch of this
study's anywhere**, so this cannot be the readout flattering itself.  Order
(-4,-2):

| row | EE3 % | dEE3 | EE6 % | `P_tile` % | relL2 | fold warnings |
|---|---|---|---|---|---|---|
| **BASELINE = v5.32.0** (C9 off, tapered) | 87.8342 | -- | 98.3395 | 98.7384 | 0.000e+00 | 0 |
| NULL identity patch | 87.8342 | **+0.0000** | 98.3395 | 98.7384 | **0.000e+00** | 0 |
| **C9 ON** (exact conversion, shipped) | **89.2353** | **+1.4011** | **98.9006** | **99.2280** | 5.810e-02 | 0 |
| the same state via the PRE-FLAG monkeypatch | 89.2353 | +1.4011 | 98.9006 | 99.2280 | 5.810e-02 | 0 |

**+1.401 EE3, +0.561 EE6, +0.490 `P_tile`, on a bit-exact differential
floor**, and no fold-caustic warning is raised in any row.  The last row is
S7.3's cross-check carried onto the production path: the flag and the
monkeypatch agree to every printed digit of every metric, including `relL2`
and the amplitude-weighted rms phase difference.

This is also a clean cross-era check: `APPROXIMATION_AUDIT_POST_C6` S2 measured
`taper OFF` at **+1.4147** on the C6 tree, before niches C7 and C8 landed.  Two
studies, two trees, two implementations of the same intervention, **0.014
points apart** -- so the taper's cost was not an artefact of the tree it was
first seen on, and the C8 support bound did not change it.

---

## 6. What is left

### 6.1 The residual, per order

| order | true ceiling | chain, converged readout | **residual** | EE6 gap | FWHM chain - oracle |
|---|---|---|---|---|---|
| (0,0)   | 90.742 | 90.693 | **0.049** | 0.018 | +0.001 um |
| (-1,0)  | 90.788 | 89.853 | **0.935** | 0.570 | +0.010 |
| (-2,0)  | 90.878 | 90.103 | **0.775** | 0.521 | +0.010 |
| (-3,0)  | 90.907 | 90.628 | **0.278** | 0.394 | -0.021 |
| (-4,0)  | 90.929 | 90.342 | **0.588** | 0.259 | +0.064 |
| (-4,-2) | 90.025 | 89.900 | **0.125** | 0.243 | +0.019 |

**It is a halo deficit, not a core blur.**  The chain's FWHM matches the exact
ray trace's to 0.02-0.06 um on every order, so the spot is the right size; what
is missing is 0.02-0.57 points of energy inside 6 um.

### 6.2 It does not converge away

At the converged readout, on the best and worst orders:

| order | `RN` / `rs` | oracle | chain | **residual** |
|---|---|---|---|---|
| (0,0) | 1024 / 4 | 90.742 | 90.693 | **0.049** |
| (0,0) | 1024 / **2** | 90.730 | 90.586 | **0.144** |
| (0,0) | **2048** / 4 | 90.718 | 90.583 | **0.135** |
| (-1,0) | 1024 / 4 | 90.788 | 89.853 | **0.935** |
| (-1,0) | 1024 / **2** | 90.776 | 89.768 | **1.008** |
| (-1,0) | **2048** / 4 | 90.766 | 89.773 | **0.993** |

Halving the ray subsample and doubling the grid each move the residual by
<= 0.10 points and **neither shrinks it** -- on axis it grows slightly, which
says the 0.049 at the campaign's own configuration was the luckier of two
readings rather than a converged one.  **The residual is a MODEL error, not a
discretisation error**, and the honest on-axis figure is ~0.13 points, not
0.05.

### 6.3 What it is not, and what it now probably is

* **Not the instrument.**  S3.5: +-0.005 points.
* **Not the leg.**  Niche C5's -0.96-point leg remnant at (-4,-2) is the taper
  (S2), and the taper is gone.
* **Not `ray_subsample` or the grid.**  S6.2.
* **Probably the element's own `remap` model error, and this is a REVERSAL
  this study is entitled to state only because the gap shrank.**
  `APPROXIMATION_AUDIT_POST_C6` S5 excluded it by measuring its implied Strehl
  deficit at 2.3e-04 to 7.7e-03 -- 0.02 % to 0.8 % of peak intensity -- and
  noting that even the pessimistic reading was 2.5x too small for the 2.0-point
  gap of the day.  Against a **0.13-1.0 point** residual that same number is
  the right order.  The exclusion was correct arithmetic against the wrong
  target; it does not survive the target moving.  **Not verified here**: doing
  so needs the `wfe_probe_*` pointwise arbiter re-run on the post-C9 tree,
  which is S9 item 1.

---

## 7. Contracts

### 7.1 The fail-before switch, against a `git archive HEAD` export

`probe_c8_byte_identity.py`'s shadow-module device imports a second copy of
`_lens_traced.py` INSIDE the live package, which works because the element is
reached through one name.  `propagators/carrier.py` is not: the chain entry
point, the element hand-off and half a dozen helpers all resolve it as
`lumenairy.propagators.carrier`, so a shadow copy would be reached by some call
sites and not others.  The reference here is instead a WHOLE-PACKAGE
`git archive 999b9e6` export in a separate PROCESS, driven through
`approx_common`'s `LUMEN_PIN` mechanism, with `np.array_equal` on npz dumps.

Case matrix, per order (0,0) and (-4,-2): `RN` 1024 and 2048, `ray_subsample`
4 and 2, 3-, 5- and 6-group runs, both final-leg routes -- plus a 40-case
sweep of the conversion FACTOR itself over `R` in
{-8 mm, -24.46 mm, -230 mm, +0.7 m, inf} x `dx` in {33.2 um, 1.5 um} x
`sign` x `centre` in {(0,0), (1.9, -0.6) mm}.

| arm | cases | flag OFF |
|---|---|---|
| design 121 chain, (0,0) | 6: `RN` 1024/2048, `rs` 4/2, 3-/5-/6-group, paraxial + `final_leg='exact'` | **`array_equal=True`, `max\|dE\| = 0.000e+00`, 6/6** |
| design 121 chain, (-4,-2) | the same 6 | **`array_equal=True`, `max\|dE\| = 0.000e+00`, 6/6** |
| the conversion factor itself | 40: 5 `R` x 2 `dx` x 2 `sign` x 2 `centre` | **`array_equal=True`, 40/40** |

**52 of 52.  With `SPHERE_PARAB_CONVERSION_EXACT = False` the working tree IS
v5.32.0, bit for bit**, against a reference that is a separately-imported
`git archive` of the release commit in another process, not a re-run of the
same code.

With the flag ON, read as a map of the defect rather than of the fix:

| case | `array_equal` | `max\|dE\|` of peak | `dP/P` |
|---|---|---|---|
| (0,0) `RN=1024 rs=4 paraxial` | False | 1.08e-01 | +4.12e-05 |
| (0,0) `RN=1024 rs=2 paraxial` | False | 1.07e-01 | +2.08e-06 |
| (0,0) `RN=2048 rs=4 paraxial` | False | 2.49e-02 | +6.75e-07 |
| **(0,0) `RN=1024 rs=4, 3 groups`** | **True** | **0** | **0** |
| (0,0) `RN=1024 rs=4, 5 groups` | False | 1.40e-05 | -5.06e-12 |
| (0,0) `RN=1024 rs=4 final_leg=exact` | False | 5.14e-02 | -5.15e-03 |
| (-4,-2) `RN=1024 rs=4 paraxial` | False | 1.34e-01 | -2.30e-05 |
| **(-4,-2) `RN=1024 rs=4, 3 groups`** | **True** | **0** | **0** |
| (-4,-2) `RN=1024 rs=4, 5 groups` | False | 5.02e-02 | -5.33e-10 |

**The 3-group chain is byte-identical on both orders** and the 5-group chain
moves by 1.4e-05 of peak on axis: the conversion is inert until the last two
groups, which is S2's locality result reached by a completely different route.

**One row is NOT interpretable and is flagged rather than quoted.**
`(-4,-2) final_leg='exact'` is run with `on_tilt_exact_grid='ignore'` -- at a
tilted order the exact leg's window needs `n_fine=16384` against this probe's
8192 cap and the guard REFUSES by default.  Both arms therefore discard the
same outer NA, which keeps the byte-identity comparison exact (it PASSES, at
`max|dE| = 0`), but the resulting field is not a valid readout: HEAD's peak in
that window is ~7.8e-07, so its "of peak" and "dP/P" columns are noise over
noise.  The tilted exact path is scored properly in S5.3 instead, at
`n_fine_cap=12288`, where the guard is satisfied.

### 7.2 Where the change is inert by construction

`_sphere_parab_conversion` is called from six sites and **every one of them is
inside an `if _sphere_ref:` / `if sphere_reference:` branch**, so under the
legacy `carrier_reference='parabola'` the flag cannot be reached at all.  The
element never calls it: `apply_real_lens_traced` is byte-identical across the
switch, pinned by
`test_niche_c9::test_the_element_alone_is_byte_identical_across_the_flag`.

The untilted path is **not** claimed inert -- this is an on-axis defect as much
as a tilted one (+1.032 points at (0,0)), and that is the point.  What IS
claimed, and pinned, is that with the flag off the whole library reproduces
v5.32.0 bit for bit on both.

### 7.3 The two intervention routes agree

Every measurement taken before the library change used a monkeypatch of
`_sphere_parab_conversion`; every one after uses the shipped flag.  Both give
the same intensity array to the last bit on the chain arm
(sha `863ecd785bce19d8` tapered / `8db002a1c1bd58ef` exact, on axis), so the
two halves of this document are on one scale.

---

## 8. What shipped

### 8.1 Library

`lumenairy/propagators/carrier.py` **only**:

* `SPHERE_PARAB_CONVERSION_EXACT = True` -- the flag, carrying the measured
  record, the per-call census, the mis-citation and its source quotation;
* one branch in `_sphere_parab_conversion` (`return np.exp(sign*1j*k*diff)`);
* the `w_beam` guard's MESSAGE rewritten -- its trigger, its threshold and its
  warning-only contract are untouched;
* the function's docstring: "NOT TAKEN, and why" -> "TAKEN, 2026-08-02".

**No signature moved, no other default flipped, no public entry point added.**
`lumenairy/elements/_lens_traced.py` is unmodified (`9717ad88dd959889`).

### 8.2 Tests added

`tests/unit/test_niche_c9_sphere_parab_exact_conversion.py`, **13 tests, ~3 s**,
no proprietary asset:

1. the default is exact, and the switch really is a switch;
2. the shipped factor is the exact closed form on EVERY pixel (`array_equal`);
3. the fail-before restores the historical taper bit for bit, INCLUDING the
   old test's own defining property (`f[r > r_safe] == 1+0j`) word for word;
4. the two arms are byte-identical inside the old onset, so the change is
   confined to the annulus;
5. collimated / degenerate still returns `None`;
6. the `+1`/`-1` pair is a pointwise identity over the whole grid, decentred
   and not;
7. **the point of the change, as a measurement**: the store -> reconstruct ->
   convert round trip recovers the physical field, exact arm over tapered arm,
   as a RATIO with the fixture's own liveness asserted;
8. **the element hand-off pair is consistent**, scored as the
   amplitude-weighted p99.9 wrapped nn step of what the element recovers --
   again a ratio, again with a liveness assertion on the tapered arm;
9. the guard still fires, is warning-only, and its message no longer claims a
   taper;
10. `apply_real_lens_traced` is byte-identical across the flag.

Every numeric bar is a ratio between two arms measured in the same process on
the same fixture; the only absolute assertions are exact-arithmetic identities
(`array_equal` on a closed form, unit modulus of `exp(i*phi)`), which are
elementwise `exp`/`sqrt` and carry no BLAS dependence.

### 8.3 Existing tests changed -- two, and both are the ones this fix breaks

**`test_niche_s8::test_conversion_factor_band_limited_taper`.**  Its assertion
is *"beyond `r_safe` the factor rolls off to identity"* -- a direct assertion
of the taper.  The assertion is kept **word for word** and is now scored on the
FAIL-BEFORE arm, the library state it was calibrated in; three assertions were
ADDED for the shipped arm (unit modulus, NOT identity beyond `r_safe`, and
byte-identical to the tapered arm inside the onset).  Nothing was relaxed.

**`test_niche_s8::test_conversion_guard_tests_the_taper_ONSET_not_r_safe`.**
Its `match='taper ONSET'` phrase no longer appears in the message, because
there is no taper to have an onset.  The phrase moved to `'band-limit radius'`,
which the sibling test already matched on.  **No threshold, radius or
assertion changed** -- the trigger, the 0.375-0.5 `r_safe` regime it pins and
the silent-just-outside control are untouched.

`test_niche_c5::test_the_sphere_parabola_conversion_is_untouched` PASSES
unchanged: its fixture grid (64 px x 30 um, `R` = -30 mm) has a half-extent of
0.96 mm against an onset at 7.9 mm, so the taper was identically 1 on it and
the test was never sensitive to this.

### 8.4 Suites and lint

```
python -m pytest tests/unit/test_niche_{c1,c3,c5,c6,c7,c8,c9,d1,d2,d3,d6,d7,s8}_*.py -q
-> 355 passed, 71 warnings in 1468.89s        (24m28s)

python -m pytest tests/unit -k "traced or carrier or sphere or c5 or c6 or c8 or c9 or d1 or s8" -q
-> 517 passed, 9 skipped, 119 warnings in 763.81s      (12m44s)

grep -c "HALO self-check FAILED"  ->  0    (both runs)
python -m ruff check lumenairy/ tests/unit/    ->  All checks passed!
```

The wider selection is the one that would catch a reader of this flag outside
the niche files; there is none.  The 9 skips are the documented CuPy / JAX-x64
environmental ones.

The 71 warnings are the pre-existing physics diagnostics the suite is
documented to emit.  `validation/` is `extend-exclude`d from ruff by
`pyproject.toml`, as it is for every existing runner in that directory.

---

## 9. What remains unmeasured

1. **The residual's mechanism is not identified.**  S6.3 names the element's
   `remap` model error as the leading candidate and says why the previous
   exclusion does not survive; it is not verified.  The test is the
   `wfe_probe_*` pointwise exact-trace arbiter re-run on the post-C9 tree,
   which is a day's work and was not attempted here.
2. **Why the residual is non-monotone in field angle** (0.049 / 0.935 / 0.775 /
   0.278 / 0.588 / 0.125) is not explained.  A 0.9-point spread across six
   neighbouring orders of one relay is itself a finding and is unexplained.
3. **One design, one wavelength, six orders.**  Nothing here says the taper was
   costing anyone else a point; what it says is that the taper's own
   justification does not survive scrutiny anywhere.
4. **`ray_subsample`** was measured at 4 and 2 on design 121.  The library's
   shipped default is 8; design 121 does not use it.
5. **The paraxial-focus-readout aliasing risk the guard now describes is
   argued, not measured.**  No design was constructed where an untapered
   conversion feeds a coarse paraxial Bluestein readout and the aliasing bites.
   That is the one configuration in which the taper's original argument
   survives, and it is why the guard was kept rather than deleted.
6. **`propagate_traced_carrier_chain_multi`** (readout tiling and
   recombination) and chain A (source -> DOE) are untouched.
7. **The production acceptance is on-axis only** (S5.1); the tilted production
   readout is S5.3.
8. **The DOE-plane residual's +0.66-point benefit is measured, not
   explained.**  That it is a pre-compensation of the post-DOE relay is the
   natural reading of an end-to-end-optimised design, and both arms agree on
   the magnitude, but no Zernike decomposition was done.

---

## 10. Artefacts found and killed in MY OWN instruments

Recorded because this project has now had ~30 artefacts pass as findings.

1. **A STALE LIBRARY PIN NEARLY MEASURED v5.31.**  `approx_common` defaults
   `LUMEN_PIN` to a scratchpad export that still exists on this box, and it
   inserts that path and imports `lumenairy` from it BEFORE `_d121_common`
   prepends the repo.  Caught by an `AttributeError` on
   `REMAP_INVERSE_SUPPORT_BOUND` -- a constant that does not exist in v5.31 --
   when a helper happened to import `approx_ablate_121` first.  Every runner
   now forces `LUMEN_PIN=0` and prints its provenance; the affected baselines
   were re-run and reproduce bit for bit.  **A harness that silently selects a
   library is worse than one that crashes.**
2. **`nohup ... &` inside a tool call reported success on a job that was still
   running, and then a second launch raced the same output file.**  The parent
   shell returns immediately (so "exit 0" means nothing), a later foreground
   timeout SIGTERMed the process group and killed the orphan mid-run, and two
   processes had `>`-truncated one log.  The taper-off acceptance was re-run to
   a uniquely named file under the harness's own background mode.  **An exit
   code from a shell that backgrounded its child is not a result.**
3. **A FOREIGN `pytest -n 12` from another session saturated the box** (cwd
   `C:/tmp/lum_rel`, a release-verification tree, not this repo) and made one
   instrument row take 176 s where it takes 32 s.  Deprioritised to
   `IDLE_PRIORITY_CLASS` rather than killed -- it is not this study's work to
   destroy -- after which the row returned to 10 s.  No number in this document
   is a wall-clock claim, but a 25x slowdown had already invalidated one
   convergence sweep by making it look like it had stalled.
4. **MY OWN TEST HAD THE ALGEBRA BACKWARDS AND STILL "PASSED" ITS OWN FIXTURE
   CHECK.**  `test_the_conversion_recovers_the_physical_field...` re-applied
   the conversion to a field that already carried it, so both arms read
   identically 1.805 -- and its liveness assertion (`e_taper > 1e-2`) was
   satisfied by that very number.  Caught only because the EXACT arm's error
   was not ~0 as an algebraic identity demands.  **A liveness check on the
   wrong quantity is not a liveness check.**
5. **The library's own `oracle_spot(carry_phase=True)` -- the pre-existing
   instrument for exactly the question in S3.2 -- is broken**, and its EE3 of
   49.6 was nearly quoted as "the true ceiling".  Caught because it did not
   converge in the launch density (49.6 / 9.9 / 52.0 / 28.0 at four
   densities).  **The convergence sweep is what catches a broken oracle; a
   single reading never does.**
6. **THE MOST DANGEROUS ONE: the false alarm and the real defect pointed the
   same way.**  `rs_spot` reported the chain readout's integrand step at 1.31
   cycles against a 0.25 bar, which reads as "the quadrature is undersampled".
   Refining it (`UP=2`) DID move EE3 by +0.34 points -- apparent confirmation.
   The wrapped statistic shows the quadrature was fine at 0.039 cycles all
   along; what `UP` was actually refining was the finite-differenced launch
   DIRECTION, i.e. a different defect with the same signature.  Had the wrapped
   statistic not been computed, this study would have shipped "refine the
   readout lattice" and never found the split.
7. **`TAPER='on'` stopped meaning "the taper" the moment the library default
   flipped**, and two probes silently changed what they measured:
   `fc_sampling_121.py` returned byte-identical taper-on and taper-off rows,
   and `fc_production_taper.py` ran a 9-minute "BASELINE (taper as shipped)"
   row that was in fact the EXACT conversion.  Caught in the first because two
   rows that must differ did not; caught in the second because its baseline
   read 89.235 where v5.32.0 reads 87.834.  **An intervention expressed as
   "leave the library alone" is not an intervention once the library moves.**
   All arms are now pinned through the C9 flag rather than through the default,
   and the flag and monkeypatch routes are checked against each other bit for
   bit on both the diagnostic (S7.3) and the production path (S5.3).
8. **My own byte-identity probe printed `FAILED -- not bit-identical` under
   its own "WHAT MOVES" heading**, i.e. at the table where differing IS the
   contract.  The fail-before arm above it read `OK`, so the pair was correct
   and the summary line was not; at a glance it reads as a broken contract.
   Reworded to say what each arm expects, with the inverted case ("the flag ON
   is identical to HEAD everywhere") named as the real failure.  **A verdict
   line that can only say FAILED will eventually say it about a pass.**
9. **A coincidence, avoided.**  The taper's onset at group 6
   (1.661 w) and at group 5 (1.639 w) are close enough to read as one number,
   and the first draft of S2 treated them as one plane.  They are different
   planes at different `dx` with different `R`, and the census in
   `fc_taper_census.py` exists so the two are never compared by eye -- the same
   discipline `probe_ghost_locate.py` was written for.

---

## 11. Reproduction

All commands from `validation/repro_traced_carrier_121/`.  Every runner prints
the library version, path and the sha256 of the two files it imported, and
forces `LUMEN_PIN=0`.

```bash
# S0/S4.3/S6.1 -- the six-arm per-order table.  ~35 s per arm, 36 arms.
ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python fc_table_121.py

# S3 -- the instrument band, one arm at a time
ARM=oracle ORD=0,0 NL=161 python fc_instrument_121.py            # the old ceiling
ARM=oracle ORD=0,0 NL=321 CARRY=1 python fc_instrument_121.py    # the true one
for nl in 121 161 221 301 401; do ARM=oracle NL=$nl python fc_instrument_121.py; done
for cl in 2.0 2.5 3.0 3.5 4.0 5.0; do ARM=chain CLIP=$cl python fc_instrument_121.py; done
ARM=chain UP=2 NL=9999 python fc_instrument_121.py
ARM=chain TAPER=off SPLIT=exact python fc_instrument_121.py      # the converged readout
ARM=chain TAPER=off SPLIT=exact STRIP=1 python fc_instrument_121.py   # matched field

# S1.4 -- sampling adequacy, amplitude-weighted p99.9
ORDERS='0,0 -4,-2' python fc_sampling_121.py

# S2 -- where the taper acts
ORD=0,0 python fc_taper_locality.py
ORD=0,0 python fc_taper_census.py

# S5.1 -- the at-plane acceptance, the library's own readout (~25 min each)
C9=0 python fc_with_taper.py focus_scan_121.py
C9=1 python fc_with_taper.py focus_scan_121.py

# S5.2 -- conservation and halo
C9=0 ORDERS='0,0 -2,0 -4,0 -4,-2' CONFIGS='ship' NULL=1 \
    python fc_with_taper.py energy_stage_audit_121.py
C9=1 ORDERS='0,0 -2,0 -4,0 -4,-2' CONFIGS='ship' NULL=1 \
    python fc_with_taper.py energy_stage_audit_121.py

# S5.3 -- the production readout on a TILTED order
ORDERS='-4,-2' python fc_production_taper.py

# S6.2 -- the residual does not converge away
for cfg in "1024 4" "1024 2" "2048 4"; do ... python fc_instrument_121.py; done

# S7.1 -- the fail-before contract
git archive 999b9e6 lumenairy | tar -x -C <scratch>/pin_c9_head
PIN=<scratch>/pin_c9_head MODE=all python fc_c9_byte_identity.py

# S8.4
python -m pytest tests/unit/test_niche_c9_sphere_parab_exact_conversion.py -q
python -m ruff check lumenairy/ tests/unit/
```

### Files added by this study

```
validation/repro_traced_carrier_121/fc_instrument_121.py       the instrument + its band
validation/repro_traced_carrier_121/fc_table_121.py            the per-order table
validation/repro_traced_carrier_121/fc_sampling_121.py         sampling adequacy per arm
validation/repro_traced_carrier_121/fc_taper_locality.py       where the taper acts, per group
validation/repro_traced_carrier_121/fc_taper_census.py         per-call census + ablation
validation/repro_traced_carrier_121/fc_with_taper.py           run any runner with the flag forced
validation/repro_traced_carrier_121/fc_production_taper.py     the library's own production readout
validation/repro_traced_carrier_121/fc_c9_byte_identity.py     the fail-before contract
tests/unit/test_niche_c9_sphere_parab_exact_conversion.py      13 tests
docs/audits/D121_FINAL_CLOSURE_2026_08_02.md                   this document
```

Raw logs: `_fc_table.txt`, `_fc_sampling.txt`, `_fc_energy_c9.txt`,
`_fc_focus_on.txt`, `_fc_focus_taperoff.txt`, `_fc_resid_conv.txt`,
`_fc_c9_byteid.txt`, `_fc_prod_taper.txt`.

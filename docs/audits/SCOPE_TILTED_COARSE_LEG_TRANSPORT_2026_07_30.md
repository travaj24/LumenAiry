# SCOPE: where design 121's per-order image quality is lost

**2026-07-30.  Investigation + scoping only -- no library change is proposed in
code here.**  Branch `fix/pmm-union-grid-conditioning` @ `1597a8c`.

All scripts live in `validation/repro_traced_carrier_121/` and are LOCAL-ONLY
(they need the 121 `.zmx` and the design-study runner).  Every number below has
a command next to it.

---

## 0. Headline

**The loss is NUMERICAL, not physical, and it is NOT where the roadmap put it.**

1. Design 121's post-DOE relay is **diffraction-limited at every DOE order**.
   An exact skew ray trace plus an energy-conserving Rayleigh-Sommerfeld
   integral -- no propagator, no carrier, no FFT, no window factor -- reads
   **EE3 89.9 / 90.1 / 90.5 / 90.1 %** and **EE6 99.95 / 99.95 / 99.95 /
   99.93 %** for orders (0,0) / (-1,0) / (-4,0) / (-4,-2), with rms wavefront
   error **0.017 waves or better** and Marechal Strehl **>= 0.995**.  The order
   at 46.1 + 23.0 mrad is, to measurement precision, as good as the on-axis one.
   This independently re-derives the roadmap's 2026-07-29 retraction from a
   different oracle construction.

2. The chain's loss appears **entirely at the LAST GROUP** (`Lens S25-S27`).
   Running the chain over the first *n* post-DOE groups and finishing the rest
   with the exact-ray oracle, order (-4,-2) reads EE3

   | n_chain | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
   |---|---|---|---|---|---|---|---|
   | EE3 % | 90.11 | 90.36 | 90.37 | 90.26 | 89.72 | 89.80 | **65.75** |

   **Five coarse legs and five element passes cost 0.31 EE3 points in total and
   show no field-angle trend.  The sixth element pass costs 24.05.**

3. Therefore the roadmap's conclusion -- *"by elimination it lies in the
   chain's TILTED-CONGRUENCE TRANSPORT across the coarse legs"*
   (`ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27.md`, ~line 255) -- **is
   wrong.**  The coarse legs are innocent.  So is my own leading prior lead
   (paraxial envelope transport / `_tilt_obliquity`'s anisotropic effective
   distance): both live on the coarse legs, and the coarse legs are bounded
   above at 0.31 points.

4. **On "Zemax seems to handle this already": the read is correct, and now
   precise.**  Zemax's sequential ray trace is exactly what the oracle above
   uses (`lumenairy.raytrace`, Zemax-validated), and it says the design is
   perfect at 46 mrad -- we match it exactly.  The failing half IS the
   diffracting-wave side.  But the failing wave step is **not** the inter-group
   POP-analogue transport; it is the **ray -> wave reconstruction inside one
   element**, `apply_real_lens_traced` on the final group, when the congruence's
   chief ray sits more than about one beam amplitude radius off that element's
   own grid centre.  Zemax POP would face the same class of problem there (it is
   the same off-axis beam on an axis-centred grid); Zemax's *sequential* trace
   does not, because it never reconstructs a wave.

---

## 1. Is the loss physical?  (Item 4 of the brief -- tested FIRST)

### 1.1 Pure geometry and wavefront -- no diffraction integral at all

```
cd validation/repro_traced_carrier_121
NL=81,161,241 BACK=5.0 python geom_wfe_121.py
```

`geom_wfe_121.py` launches a ray grid from the DOE plane with the exact local
wavefront normal of the field that actually arrives there (chain-A envelope,
exact sphere `R_doe`, plus the order's grating tilt), traces the post-DOE
surfaces with `lumenairy.raytrace`, and reports the illumination-weighted
geometric spot and the wavefront error against a reference sphere.

| order | field angle | geo rms radius (um) | WFE rms @ fixed centroid (waves) | WFE rms @ best focus | Marechal Strehl | unvignetted |
|---|---|---|---|---|---|---|
| (0,0)   | 0                | 0.4272 | 0.01688 | 0.00357 | 0.9995 | 100.000000 % |
| (-1,0)  | 11.5 mrad        | 0.4019 | 0.01577 | 0.00387 | 0.9994 | 100.000000 % |
| (-4,0)  | 46.1 mrad        | 0.2960 | 0.01068 | 0.00697 | 0.9981 |  99.999979 % |
| (-4,-2) | 46.1 + 23.0 mrad | 0.4215 | 0.01522 | 0.01170 | 0.9946 |  99.999936 % |

The geometric radii reproduce the roadmap's independently-computed
0.427/0.402/0.296/0.422 um to 4 digits.

**Sampling adequacy.** Both quantities are POINTWISE (no quadrature), so they
converge with few rays: NL = 81 / 161 / 241 agree to 5 significant figures.
`BACK` (the exit reference plane, 2.0 / 5.0 / 6.5 mm short of the image plane)
moves the WFE in the 5th digit; it must, because the quantity is
reference-plane-invariant by construction.

### 1.2 Exact-ray + Rayleigh-Sommerfeld PSF

```
NL=161 NOUT=261 DXO=0.1 CLIP=3.0 BACK=5.0 python exact_ray_oracle_121.py
```

`exact_ray_oracle_121.py` adds the first Rayleigh-Sommerfeld integral (exact
spherical-wave kernel, no Fresnel/Fraunhofer expansion) from a flat exit
reference plane to the image plane, with an energy-conserving ray-density
amplitude `|E_doe| sqrt(N0 |J| / N) h^2`, `J = |d(x_exit,y_exit)/d(x_launch,
y_launch)|` by central differences.

| order | FWHM (um) | EE3 % | EE6 % | EE12 % |
|---|---|---|---|---|
| (0,0)   | 3.339 | 89.89 | 99.95 | 100.00 |
| (-1,0)  | 3.325 | 90.07 | 99.95 | 100.00 |
| (-4,0)  | 3.297 | 90.47 | 99.95 | 100.00 |
| (-4,-2) | 3.323 | 90.05 | 99.93 | 100.00 |

Corroboration: `focus_scan_121.py`'s own printed ideal-field ceiling is
3.45-3.55 um / 90.3 / 99.8 on axis, and the shipped single-beam acceptance is
3.450 um / 88.8 / 99.6.

**Sampling adequacy, order (-4,-2), EE3 %:**

| knob | values | EE3 |
|---|---|---|
| launch grid `NL` | 121 / 161 / 241 | 90.05 / 90.05 / 90.05 |
| launch aperture `CLIP` (amplitude radii) | 2.6 / 3.0 / 3.4 | 90.05 / 90.05 / 90.05 |
| exit reference plane `BACK` (mm) | 2.0 / 5.0 / 6.5 | 90.05 / 90.05 / 90.05 |
| image pitch `DXO` (um) | 0.05 / 0.1 / 0.2 | 90.05 / 90.05 / 90.11 |

The measured phase step of the ACTUAL integrand between adjacent launch rays,
at the far corner of the evaluated patch, is 0.064-0.13 cycles (max) and
0.038-0.075 amplitude-weighted -- i.e. 8 to 26 samples per fringe.  Vignetted
power is < 1e-6 of the beam at every order.

### 1.3 Verdict

The design does not degrade off axis.  **All 24-27 EE3 points are the
simulator.**

---

## 2. Localisation

```
ORD="-4,-2" NMIN=0 NMAX=6 NOUT=131 DXO=0.2 python hybrid_localize_121.py
```

`hybrid_localize_121.py` runs `propagate_traced_carrier_chain` over the first
`n` post-DOE groups (tilted congruence, shipped defaults), takes the field it
hands out, and finishes with the exact-ray + RS oracle.  `n = 0` is the pure
oracle; `n = 6` is the chain doing every group.  The EE is measured identically
at every `n`.

### 2.1 The step is at the last group, and it carries the whole field-angle trend

Chain does groups 0..4, oracle does group 5 (`n = 5`) vs chain does all six
(`n = 6`).  Configuration `NOUT=131 DXO=0.2 UP=2 NL=201`:

| order | n=5 (oracle does group 5) | n=6 (chain does group 5) | loss | shipped chain, exact final leg |
|---|---|---|---|---|
| (0,0)   | 89.94 | **87.60** | -2.34  | 87.62 |
| (-4,0)  | 89.96 | **70.15** | -19.81 | 68.13 |
| (-4,-2) | 89.80 | **65.75** | -24.05 | 65.26 |

The `n = 6` column reproduces the shipped per-order EE3 (roadmap D7 table) to
**0.02 points on axis**, 2.0 at (-4,0) and 0.5 at (-4,-2) -- while going through
a *different* final path (coarse element pass + my RS transport, versus the
shipped exact final leg's fine retrace + Bluestein readout).  Two independent
implementations of the last step agreeing to that tolerance is the strongest
single validation in this study.

### 2.2 Scaling across the fan (`n = 6`, `NOUT=61 DXO=0.4 UP=1 NL=121`)

| order | field angle | EE3 % | excess loss over the on-axis floor | `m^2` scaled |
|---|---|---|---|---|
| (0,0)   | 0         | 87.99 | 0.0  | 0.0  |
| (-1,0)  | 11.5 mrad | 86.48 | 1.5  | 1.2  |
| (-2,0)  | 23.0 mrad | 83.15 | 4.8  | 4.8  |
| (-3,0)  | 34.5 mrad | 77.02 | 11.0 | 10.8 |
| (-4,0)  | 46.1 mrad | 70.15 | 17.8 | 19.2 |
| (-4,-2) | 46.1+23.0 | 66.24 | 21.8 | -    |

The excess loss is **quadratic in the field angle** to within a point out to
m = 3, saturating slightly at m = 4 (as it must, since EE3 is bounded).  A
quadratic EE loss corresponds to a spurious wavefront error whose rms grows
**linearly** with the field angle: `1 - EE_loss ~ exp(-(2 pi sigma)^2)` gives
`sigma ~ 0.018 waves` at m = 1 and **`sigma ~ 0.067 waves` (88 nm) at m = 4**,
on top of a field-angle-independent on-axis floor of ~0.024 waves.

### 2.3 Is the `n = 5` hand-off honest?

This is the number the whole localisation rests on, so it was attacked:

* **Fourier upsampling of the hand-off envelope** (`UP` = 1 / 2 / 4): the
  per-pixel residual-phase step falls 0.943 -> 0.507 -> 0.260 rad, exactly
  `1/UP`, proving the envelope is band-limited on the chain grid; EE3 reads
  89.80 / 89.80 / 89.79.  The reading is converged.
* **On axis** the hand-off is trivial (no tilt, no decentre) and still shows a
  2.34-point step at group 5, matching the shipped 87.62 to 0.02.  A hand-off
  that idealised the field away would not reproduce the shipped number.
* **Chain grid** `RN` 1024 -> 2048 (dx halved everywhere): `n=5` 89.80 ->
  89.98, `n=6` 65.75 -> 66.14.  Both flat.
* At `n = 6` the hand-off's own phase step is pinned at pi (3.14 rad/px) and
  does NOT fall under upsampling -- i.e. the chain's exit envelope at the last
  group is genuinely at/over its grid Nyquist.  The EE3 nevertheless reads
  65.75 / 66.09 / 66.04 at UP 1/2/4, and 66.24 on a different readout grid, so
  the ~0.5-point spread from the readout choice is the honest error bar on that
  number.

---

## 3. Ranked candidates

`Explained` = EE3 points of the 24-point (-4,-2) gap the mechanism accounts
for, measured.

| # | candidate | explained | confidence | decisive experiment | result |
|---|---|---|---|---|---|
| 1 | **`apply_real_lens_traced` on the LAST group under a >1 w decentred, tilted congruence at high exit NA** | **21.7 of 24** (all of the field-angle-dependent part) | HIGH | hybrid `n=5` vs `n=6` sweep, 6 orders | 89.8 -> 65.8 at (-4,-2), 89.9 -> 87.6 on axis; step is the LAST group only |
| 2 | Same element's field-angle-INDEPENDENT floor at high exit NA | 2.3 of 24 | HIGH | hybrid `n=5` vs `n=6`, order (0,0) (zero tilt, zero decentre) | 89.94 -> 87.60 |
| 3 | Paraxial envelope transport on the coarse legs (the prior leading lead) | <= 0.31, no field-angle trend | HIGH (refuted) | hybrid `n=0..5` | 90.11 -> 89.80 over five legs |
| 4 | `_tilt_obliquity` anisotropic effective distance (`z/(1-L^2)^{3/2}`) | ~0 | HIGH (refuted) | bounded by #3; direct estimate: +0.45 % of a 3.85 mm reduced distance = 17 um of extra propagation on a 3.6 mm beam | negligible by 4+ orders of magnitude |
| 5 | Fold caustic / single-branch `ray_density` (coordinator's candidate) | 0 | HIGH (refuted) | warning capture at every group and every order; `caustic='multibranch'` attempt | the fold-caustic warning **never fires** on design 121; `caustic='multibranch'` is **REFUSED** by the library for an explicit `TiltedCarrier` |
| 6 | Tilt-ramp aliasing on the co-moving grid (`\|k L dx\|/pi` = 2.8-4.0 at 5 of 6 groups) | 0 | MEDIUM-HIGH (refuted as cause) | `RN` 1024 -> 2048 halves the ratio; the exact final leg applies the ramp on the fine grid (ratio 0.13) | 66.24 -> 66.14; exact leg still 65.26 |
| 7 | Exit ray-density AMPLITUDE at the last group | +0.05 | HIGH (refuted) | Gaussian-smooth the exit amplitude before transport (`AMPSM=3`) | 66.24 -> 66.29 |
| 8 | Element ray lattice (`ray_subsample`) | 0.1 | HIGH (refuted) | rs 4 / 2 / 1 | 66.24 / 66.12 / 66.24 |
| 9 | Newton iteration cap | 0.0 | HIGH (refuted) | `newton_max_iters` 12 -> 60 | 66.24 -> 66.24 (and 81.2 % still unconverged) |
| 10 | Element OPL fit order / fit disc | <= 0.1 | HIGH (refuted) | `decentred_fit_poly_order` 10 -> 14; `newton_poly_order` 6 -> 12; `fit_radius_beam_factor` -> 3.0 | 66.22 / 66.19 / 66.28 |
| 11 | `remap_sampling` | +1.6 | MEDIUM | `'full'` (default) vs `'lattice'` | 66.24 vs 67.87 -- the only non-null knob found, and it is the NON-default |
| 12 | `preserve_input_phase=True` (legacy) | -55 | HIGH | legacy config on group 5 | 10.78 -- far worse; the shipped `'remap'` is much better |
| 13 | Sphere/parabola conversion taper | 0 | MEDIUM-HIGH | field-angle independent by construction (taper measured from the chief ray); documented S12 null | not a field-angle mechanism |

### 3.1 On the coordinator's fold-caustic candidate specifically

Tested, and **refuted for design 121**:

* Warnings were captured at every group for orders (0,0) and (-4,-2) with
  `warnings.catch_warnings(record=True)`.  The fold-caustic warning
  (`"detected a fold caustic (det J -> 0 or a sign change)"`) **does not fire
  once**, at any group, at any order.  The warnings that DO fire are two
  others, listed in section 5.
* The suggested remedy is architecturally unavailable on this path anyway:
  `caustic='multibranch'` raises
  `"caustic='multibranch' supports carrier=None or carrier='auto' only (the
  launch is one tilted congruence)"`.  `apply_real_lens_gbd` / `_fga` are not
  reachable from `propagate_traced_carrier_chain` at all.
* The correlation the candidate predicted (silent on axis, firing by (-4,-2))
  is absent: silent everywhere.

The candidate was a good one -- it predicted the right shape of failure -- but
the ray map of design 121's relay simply does not fold.  That is consistent
with section 1: a folded ray map cannot produce a 0.30 um geometric spot.

---

## 4. What in the last group?  (partially determined)

### 4.1 The chain's own stage table, order (-4,-2)

```
python - <<'EOF'   # see section 7 for the full snippet
EOF
```

| group | dx (um) | w (um) | R_in (mm) | R_out (mm) | entrance `\|k L dx\|/pi` | chief-ray decentre at ENTRANCE (guard's own numbers) |
|---|---|---|---|---|---|---|
| 0 `S13-S14` | 51.234 | 6318.8 | 703649.7 | 703664.8 | 4.028 | (below guard) |
| 1 `S15-S16` | 51.234 | 6320.5 | 703669.8 | 703671.9 | 4.028 | (below guard) |
| 2 `S17-S19` | 51.235 | 5990.8 | 703676.9 | -263.194 | 4.028 | (below guard) |
| 3 `S21-S22` | 44.912 | 4852.3 | -230.716 | -60.148  | 3.201 | 3.6052 mm = **0.685 w** |
| 4 `S23-S24` | 38.432 | 3625.3 | -51.470  | -24.462  | 0.433 | 3.6948 mm = **0.890 w** |
| 5 `S25-S27` | 33.211 | 1204.9 | -21.139  | -7.712   | 2.782 | 3.3723 mm = **1.079 w** |

Measured exit NA at the element (from its own warning): group 3 **0.0875**,
group 4 **0.1995**, group 5 **0.3633** at (-4,-2) / **0.2879** on axis.

**Group 5 is the only group that is both >1 beam amplitude radius decentred AND
at high exit NA.**  Groups 3 and 4 reach 0.685 w and 0.890 w with the same tilt
and cost nothing measurable (hybrid steps -0.11 and -0.54 EE3 points).  The
on-axis order reaches the same exit NA at zero decentre and costs 2.34 points.
Neither condition alone reproduces the failure.

### 4.2 Decentre x tilt probe on the last group alone

```
DEC=0,0.5,1.0,1.778 TILTS=0,1 NOUT=61 DXO=0.4 python last_group_probe_121.py
```

Same synthetic input field (Gaussian, measured entrance radius, measured
entrance carrier) pushed through the last group two ways -- the chain's own
element pass, and the exact-ray oracle -- with the complex overlap `F` of the
two image-plane fields as the metric (`sigma` = the equivalent rms wavefront
error from `1 - F = 1 - exp(-(2 pi sigma)^2)`).  EE3 alone is useless here: the
synthetic reference is itself aberrated (EE3 5-33 %), so a 1 % element defect
is invisible in it.

| tilt | decentre (probe w) | oracle EE3 % | chain EE3 % | 1 - F | sigma (waves) |
|---|---|---|---|---|---|
| OFF | 0.000 | 31.67 | 30.92 | 0.00294 | 0.0086 |
| OFF | 0.500 | 33.40 | 33.74 | 0.00482 | 0.0111 |
| OFF | 1.000 | 22.45 | 22.89 | 0.00303 | 0.0088 |
| OFF | 1.778 | 10.09 | 10.74 | **0.02908** | **0.0273** |
| ON  | 0.000 | 33.13 | 33.09 | 0.00098 | 0.0050 |
| ON  | 0.500 | 26.36 | 26.62 | 0.00420 | 0.0103 |
| ON  | 1.000 |  7.61 |  7.47 | 0.00555 | 0.0119 |
| ON  | 1.778 |  4.91 |  4.63 | **0.16704** | **0.0680** |

Two things this DOES establish:

* **Tilt alone is harmless.**  The `tilt ON, decentre 0` cell is the BEST of all
  eight (`1 - F` = 0.00098).
* **Decentre and tilt combine super-additively.**  At 1.778 probe-w the error
  variance is 5.7x larger with the tilt on than off.  The onset is a cliff, not
  a ramp: flat at 0.003-0.006 out to 1.0 w, then 30x.

What this does **NOT** establish, and I am flagging it rather than glossing it:
the design's actual entrance decentre at group 5 is **0.931 probe-w** (3.3723 mm
against the probe's 3.6225 mm unit; = 1.079 of the guard's own `w`), where the
probe reads `sigma` = 0.010-0.012 waves -- **5x below the ~0.067 waves the real
case implies** (section 2.2).  So the probe reproduces the *shape* of the
dependence but not the *magnitude* at the design point.  Two known reasons the
probe under-reads, neither eliminated:

1. its reference is heavily aberrated, so much of the field lies outside the
   +-12 um patch on which `1 - F` is computed;
2. its input is a clean Gaussian on a sphere, whereas the real field at that
   plane carries five groups of accumulated correction.

**Conclusion for section 4: the mechanism is inside the last group's element
pass and requires a >~1 w decentre together with the group's high exit NA;
which construction inside `apply_real_lens_traced` breaks is NOT determined.**

### 4.3 What inside the element is already exonerated

Every knob below was swept on the REAL case (order (-4,-2), hybrid `n = 6`) and
moves EE3 by <= 0.1 points: `ray_subsample` (4/2/1), `newton_max_iters`
(12/60), `decentred_fit_poly_order` (10/14), `newton_poly_order` (6/12),
`fit_radius_beam_factor` (default/3.0), `newton_mask_dilate_coarse_px` (2/16),
chain grid `RN` (1024/2048), exit grid (coarse hybrid vs the shipped exact
leg's fine retrace).  Exit AMPLITUDE smoothing: +0.05.  This is why D7's
fit-order fix bought only 4.8 points and then stopped: **the fit is not the
defect.**

Two knobs are closed off by the library and could not be tested:
`inversion_method='fit'` (`"amplitude_model='ray_density' requires
inversion_method='newton'"`) and `newton_amp_mask_rel` (`"conflicts with
amplitude_model='ray_density'"`).

---

## 5. Two unexplained warnings at the last group

Both fire on EVERY order including on-axis, so neither is the field-angle
mechanism, but neither is understood either:

1. `apply_real_lens_traced: the exit beam converges at NA_exit=0.3633, so the
   exit wavefront needs dx <= 1.80 um but the grid has dx = 33.211 um` -- an
   18x undersampling of the exit wavefront on the coarse path.  The shipped
   exact final leg satisfies this (dx_fine 1.51 um at `n_fine_cap` 12288) and
   returns the SAME answer, which is why it is not ranked as causal.
2. `Newton inversion: 53228/65536 pixels (81.2 %) did not converge to
   tol=3.321e-07 m within 12 iterations` -- and **still 81.2 % at 60
   iterations**, with the EE3 unchanged to 5 significant figures.  The roadmap
   calls these out-of-domain edge pixels; that is consistent with the EE3
   insensitivity, but 81 % of the grid is a lot of pixels to be out of domain
   and the claim has not been verified here.

---

## 6. Proposed fix for the leading candidate

**Root requirement.** The last group's element must stop being handed a beam
that sits ~1.1 beam amplitude radii off its own grid centre.  The chain already
runs each congruence in a chief-ray-tracking frame everywhere EXCEPT inside the
element: `propagate_traced_carrier_chain` deliberately un-tracks
(`_shift_envelope(env, x_c, y_c, cur_dx)` at `carrier.py:5334`) so the element
sees the beam at its true transverse position, because the surface zones the
trace uses must be the physical ones.  That is correct physics and must be
kept.  What must change is the element's NUMERICAL frame.

**Option A (the real fix).  Decouple the element's numerical frame from the
optical axis.**  `apply_real_lens_traced` builds its launch grid, its Newton
inversion domain and its fit disc symmetrically about the optical axis; give it
a `beam_centre` that moves all three onto the chief ray while the ray trace
itself continues to use absolute coordinates.  A `beam_centre` argument already
exists in the signature, and D7 already established that the fit BASIS remap
alone is a no-op (total-degree polynomial spaces are affine-invariant) -- so
the work is in the Newton domain and the launch geometry, not the fit.

* Effort: **medium-large**, 3-6 days.  Touches `_lens_traced.py` in several
  places; needs a byte-identity proof on the on-axis path (the D7 pattern:
  21 configurations, max |dE| = 0.0) and a new decentre-swept regression.
* Risk: the on-axis path must stay byte-identical or the whole 121 acceptance
  re-baselines.

**Option B (cheap, honest, not a fix).  Tighten the guard.**  Make
`on_decentred_fit` an ERROR above ~1.0 w rather than a warning at 0.5 w, and
say in the message that the metric is not a lower bound but a ~24-point loss at
1.08 w and high exit NA.  Effort: **hours.**  This stops the number being
quoted, it does not recover it.

**Option C (do this FIRST, before either).  Instrument the element.**  Add a
diagnostic that reports the element's own exit wavefront error against the
exact ray trace as a function of decentre, on the real design-121 last group.
That is the measurement that would name the broken construction, which section
4.2 could not.  Effort: **1-2 days**, and it de-risks Option A entirely.

**Recommended order: C, then A, with B shipped immediately as a stopgap.**

---

## 7. What I could NOT determine

1. **Which construction inside `apply_real_lens_traced` produces the
   decentre-driven error.**  Ten knobs exonerated (section 4.3); the remaining
   untested surface is the Newton initial guess / bracket at large decentre,
   the entrance-pullback domain, and `preserve_input_phase='remap'`'s
   resampling of the input phasor at the pulled-back point.
2. **Why the last-group probe under-reads the magnitude by 5x at the design's
   own decentre** (section 4.2).  Two candidate reasons given, neither
   eliminated.  Until this is closed, "decentre is the mechanism" is
   well-supported but not proven; what IS proven is "the last group's element
   pass is the mechanism".
3. **The on-axis 2.34-point floor** at group 5 (zero tilt, zero decentre).  Not
   investigated.  It is the same ~2-point gap that separates the shipped
   single-beam acceptance (88.8) from the ideal ceiling (90.3-90.7), so it is a
   pre-existing, separately-known item.
4. **The 81 % Newton non-convergence** at the last group, unchanged at 60
   iterations (section 5).
5. **A discrepancy with the roadmap's oracle.**  My WFE rms reads 0.011-0.017
   waves where the roadmap's 2026-07-29 oracle reads 0.078-0.082.  Both
   conclude "diffraction-limited at every order" and my PSF EE3 (89.9-90.5)
   matches the roadmap's (90.4-90.8) to within 0.7 points, so the difference is
   almost certainly a reference convention (fixed centroid vs best focus vs
   weighting).  Not reconciled -- the roadmap's script was not found in the
   tree.
6. **I did not re-run the shipped `fan_multi_121.py` exact-final-leg path
   end-to-end.**  Section 2.1's "shipped" column is the roadmap's recorded D7
   table.  My independent path reproduces it to 0.02 / 2.0 / 0.5 EE3 points.

---

## 8. Measurement artefacts found and killed in MY OWN instruments

Recorded because this project has had three successive artefacts pass as
findings, and six more were caught here before they could:

1. **`post_surfaces(back_off=...)` applied the pull-back to every group**, not
   just the last, shortening the system by 30 mm.  Symptom: the best-focus fit
   ran away to +7.5 mm.  Caught by the fit's own implausibility.
2. **Omitted launch-plane OPL.**  `trace` starts every ray at `opd = 0` on the
   z = 0 plane, but the constant-phase surface at launch is the carrier sphere
   TILTED by the DOE order.  The tilt piston `k(Lx+My)` is **562 waves at the
   beam edge for m = -4**.  Omitting it made the wavefront appear to converge
   on the optical axis and read **27.8 to 124 waves rms for a 0.30-0.42 um
   geometric spot.**  Caught by cross-checking the WFE against the geometric
   spot -- two quantities that cannot disagree by that much.
3. **Nelder-Mead in metres** on the 3-dof focus fit converged to a 1.68-wave
   "solution".  Replaced by a Gauss-Newton whose steps are a weighted linear
   removal of piston plus the three direction-cosine modes (scale-free).
4. **A z-sign error** in the reference-plane distance (`P[2] - back` for
   `back + P[2]`): `rho` was unaffected (it is squared) but the Gauss-Newton
   derivative had the wrong sign.
5. **The first sampling metric measured the wrong thing** -- the exit-plane
   fringe pitch rather than the integrand's phase step.  It read "209 cycles,
   UNDERSAMPLED" on a configuration whose real step was 0.17 cycles.  The
   integrand is stationary-phase; `ph_exit` and `k*rho` very nearly cancel.
6. **The hybrid's first launch used the whole co-moving grid** (26 mm
   half-extent for a 6.3 mm beam), wasting 95 % of the rays on amplitude
   < 1e-15 whose traced OPL is meaningless, and poisoning the 2-D phase unwrap.
   Spurious "119-cycle integrand step" on a grid FINER than the oracle's.

The one artefact I could not fully eliminate is flagged inline: at `n = 6` the
hand-off's residual-phase step is pinned at pi and does not fall under Fourier
upsampling, so the chain's exit envelope there genuinely sits at its grid
Nyquist.  The EE3 is nevertheless stable to +-0.5 points across UP 1/2/4 and
two readout grids, and the independent shipped exact-leg number sits inside
that band.

---

## 9. Reproduction

All commands from `validation/repro_traced_carrier_121/`.  Chain A
(source -> DOE) is cached to `_chainA_<N>_<dx0>nm_rs<rs>.npz` on first use
(~6 s at N=1024); delete to force a re-run.

```bash
# 1. Is the loss physical?  Geometry + wavefront (no diffraction integral).
NL=81,161,241 BACK=5.0 python geom_wfe_121.py

# 2. Exact-ray + Rayleigh-Sommerfeld PSF, all four orders (~3.5 min).
NL=161 NOUT=261 DXO=0.1 CLIP=3.0 BACK=5.0 python exact_ray_oracle_121.py
#    convergence sweeps (each ~1 min):
ORD="-4,-2" NL=121 NOUT=261 DXO=0.1 python exact_ray_oracle_121.py
ORD="-4,-2" NL=241 NOUT=261 DXO=0.1 python exact_ray_oracle_121.py
ORD="-4,-2" CLIP=2.6 python exact_ray_oracle_121.py
ORD="-4,-2" BACK=2.0 python exact_ray_oracle_121.py
ORD="-4,-2" DXO=0.05 NOUT=521 python exact_ray_oracle_121.py

# 3. Localisation: n_chain = 0..6 (~5 min).
ORD="-4,-2" NMIN=0 NMAX=6 NOUT=131 DXO=0.2 python hybrid_localize_121.py
#    hand-off convergence:
ORD="-4,-2" NMIN=5 NMAX=6 UP=2 NL=201 NOUT=131 DXO=0.2 python hybrid_localize_121.py
ORD="-4,-2" NMIN=5 NMAX=6 UP=4 NL=201 NOUT=131 DXO=0.2 python hybrid_localize_121.py
#    field-angle scan at n = 6:
for o in 0,0 -1,0 -2,0 -3,0 -4,0 -4,-2; do \
  ORD="$o" NMIN=6 NMAX=6 NOUT=61 DXO=0.4 NL=121 python hybrid_localize_121.py; done

# 4. Candidate knockouts (each ~30 s), order (-4,-2), n = 6:
AMPSM=3 ORD="-4,-2" NMIN=6 NMAX=6 NOUT=61 DXO=0.4 NL=121 python hybrid_localize_121.py
RS=1    ORD="-4,-2" NMIN=6 NMAX=6 NOUT=61 DXO=0.4 NL=121 python hybrid_localize_121.py
RN=2048 ORD="-4,-2" NMIN=5 NMAX=6 NOUT=61 DXO=0.4 NL=121 python hybrid_localize_121.py
TKW=newton_max_iters=60          ORD="-4,-2" NMIN=6 NMAX=6 NOUT=61 DXO=0.4 NL=121 python hybrid_localize_121.py
TKW=decentred_fit_poly_order=14  ORD="-4,-2" NMIN=6 NMAX=6 NOUT=61 DXO=0.4 NL=121 python hybrid_localize_121.py
TKW=newton_poly_order=12         ORD="-4,-2" NMIN=6 NMAX=6 NOUT=61 DXO=0.4 NL=121 python hybrid_localize_121.py
TKW=fit_radius_beam_factor=3.0   ORD="-4,-2" NMIN=6 NMAX=6 NOUT=61 DXO=0.4 NL=121 python hybrid_localize_121.py
TKW=remap_sampling=lattice       ORD="-4,-2" NMIN=6 NMAX=6 NOUT=61 DXO=0.4 NL=121 python hybrid_localize_121.py
TKW=caustic=multibranch          ORD="-4,-2" NMIN=6 NMAX=6 NOUT=61 DXO=0.4 NL=121 python hybrid_localize_121.py   # raises

# 5. Decentre x tilt probe on the last group (~5 min).
DEC=0,0.5,1.0,1.778 TILTS=0,1 NOUT=61 DXO=0.4 python last_group_probe_121.py
```

Stage table and decentre-guard numbers (section 4.1):

```python
import warnings, numpy as np
warnings.filterwarnings('ignore')
import _d121_common as C
pre, post, gap, period = C.geometry()
env, R, dx, P = C.chain_a(n=1024)
k = 2 * np.pi / C.LAM
for m, n in [(0, 0), (-4, 0), (-4, -2)]:
    L, M = m * C.LAM / period, n * C.LAM / period
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        res = C.la.propagate_traced_carrier_chain(
            env, post, C.LAM, dx, r_in=C.la.TiltedCarrier(R, L, M),
            ray_subsample=4, n_workers=8, final_distance=0.0,
            final_leg='paraxial', on_decentred_fit='warn')
    print(f"=== order ({m},{n}) ===")
    for i, s in enumerate(res.stages):
        if s.get('target'):
            continue
        Li = L if i == 0 else res.stages[i - 1].get('L_out', 0.0)
        Mi = M if i == 0 else res.stages[i - 1].get('M_out', 0.0)
        print(f"  grp {i}: dx {s['dx']*1e6:7.3f} um  w {s['w']*1e6:7.1f} um  "
              f"R_in {s['R_in']*1e3:10.3f}  R_out {s['R_out']*1e3:10.3f}  "
              f"|k L dx|/pi {k*np.hypot(Li,Mi)*s['dx']/np.pi:6.3f}")
    for w in wl:
        if 'off the element grid centre' in str(w.message):
            print("  GUARD:", str(w.message)[:200].replace('\n', ' '))
```

### Files added by this study (none are library code)

```
validation/repro_traced_carrier_121/_d121_common.py          shared 121 setup + chain-A cache
validation/repro_traced_carrier_121/geom_wfe_121.py          geometry + wavefront oracle
validation/repro_traced_carrier_121/exact_ray_oracle_121.py  exact-ray + Rayleigh-Sommerfeld PSF
validation/repro_traced_carrier_121/hybrid_localize_121.py   chain(0..n) + oracle(n..6) bisection
validation/repro_traced_carrier_121/last_group_probe_121.py  decentre x tilt probe, fidelity metric
```

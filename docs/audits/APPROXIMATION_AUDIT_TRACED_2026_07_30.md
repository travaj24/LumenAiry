# Approximation audit of the traced-carrier path -- design 121, 2026-07-30

**Question.** The traced-carrier path is to be used as an ORACLE, so every
remaining APPROXIMATION in it has to be named, priced, and either justified or
replaced.  This document enumerates them, measures each one, and classifies it
as a *reference / conditioning* choice (something a downstream residual
absorbs exactly, so it costs nothing when the downstream is exact) or a
*physics* approximation (information genuinely lost).

**Scope.** `lumenairy/propagators/carrier.py`,
`lumenairy/elements/_lens_traced.py`, and the raytrace / glass / MFT helpers
they call.

**Headline.** Every classic suspect measures NULL on design 121 at HEAD:

* the paraxial **Fresnel / Sziklas-Siegman envelope transport** drops
  **9.4e-07 waves rms** at its worst coarse leg, and the *unimplemented*
  anisotropic tilt-stretch drops **7.1e-05 waves rms** (S2, computed in closed
  form against the envelope's own measured angular spectrum -- not estimated);
* the **tensor-Chebyshev ray-map fit** costs **1.1e-05 waves rms** at the
  shipped off-centre order 10, 3.1e-03 waves at the on-axis order 6 (S4);
* the **intersection / refraction / OPD / glass** layer is exact to
  **1.6e-10 waves**, and design 121 never enters the 10-iteration Newton
  intersection path at all -- all 16 relay surfaces are closed-form (S5);
* every **band limit** in the exact readout is INERT (the ASM mask sits at
  NA 1.20 and the evanescent cut at NA 1.0, against a grid Nyquist of 0.4343),
  and the readout transform is an exact Bluestein DFT, not an approximation
  (S5);
* every band limit, taper, interpolation, cache and parity convention in the
  chain is either INERT on this design or measures below 0.7 EE3 points (S1).

**`_paraxial_group_r_out` looks like a 6-point defect at HEAD and is worth
nothing once C6 lands.**  Replacing the paraxial-Moebius exit carrier radius
with the sphere fitted to each group's own TRACED exit wavefront, iterated to
self-consistency, measures

| library | baseline EE3 % | traced exit sphere EE3 % | delta |
|---|---|---|---|
| pinned HEAD `d2e60ca` | 72.501 | 78.472 | **+5.97 pts** |
| in-flight C6 snapshot | 87.771 | 87.609 | **-0.16 pts** |

The gain is 97 % gone once the C6 stationary-phase launch is in.  That is
exactly what the mechanism predicts: at HEAD, `remap`'s error scales with
`grad a`, and a mis-set carrier is what makes `grad a` large -- so the
apparent `R_out` sensitivity was the open C6 defect being fed a
badly-conditioned carrier, not an independent loss.  **`_paraxial_group_r_out`
is a clean REFERENCE choice and should be left alone** (S3.2).

**This is the single most important methodological warning in this
document.**  Every NON-NULL row of the ranked table was measured against HEAD,
i.e. against a library with a large open defect whose sensitivity is coupled
to exactly the carrier/residual conditioning most of those rows perturb.  The
one row that was re-measured post-C6 collapsed by 97 %.  **Treat every
non-null row as an UPPER BOUND pending re-measurement against the landed C6**
(S7.3).

**The one approximation on this path that is NOT small is the one already
being fixed.**  Two runs of the identical configuration, one against pinned
HEAD and one against the in-flight niche-C6 working tree, differ by

| | EE3 % | EE6 % | Ptile % | last group `P > nyq` |
|---|---|---|---|---|
| pinned HEAD `d2e60ca` | **72.501** | 92.278 | 95.648 | 7.513e-04 |
| in-flight C6 tree (snapshot, unhashed) | **87.943** | 98.509 | 98.732 | 7.979e-04 |

i.e. the `preserve_input_phase='remap'` stationary-phase launch term is worth
about **+15.4 EE3 points** on order (-4,-2), and recovers 3.1 points of power
into the 19.2 um tile as well.  Nothing else in this audit is within two
orders of magnitude of that.  The comparison is reported with the caveats in
S7.2 (the live tree was a work-in-progress snapshot and was not hashed at run
time), and the C6 tree also changes the exit ray map enough to alter
`exit_power_above_nyquist` and, apparently, to raise a fold-caustic warning
that HEAD does not raise (S7.1).

One convention that is decisive and must not be touched:
**`carrier_reference='sphere'`**.  Switching it to `'parabola'` costs
**-30.15 EE3 points** (`relL2` 0.537).  That is not a taper or a band limit --
it is the difference between referencing the exact sphere and its paraxial
parabola, and design 121's carrier NA makes it dominant.

---

## 0. Provenance, instruments, and floors -- read this before any number

### 0.1 What was measured, exactly

`lumenairy/elements/_lens_traced.py` **was being edited by another agent while
this audit ran** (the niche-C6 `preserve_input_phase='remap'` stationary-phase
fix; 441 inserted lines).  A moving target is not auditable -- the first batch
of runs died on a half-saved file (`NameError: _REMAP_RESID_TAPER_IN`).  Every
number below is therefore taken against a **frozen `git archive HEAD` export**
of the package:

| | |
|---|---|
| branch / HEAD | `fix/pmm-union-grid-conditioning` @ `d2e60ca` |
| pinned tree | `<scratchpad>/pin_d2e60ca/lumenairy/` |
| `elements/_lens_traced.py` | sha256 `957f00129f8b467c` (353 908 bytes) |
| `propagators/carrier.py` | sha256 `2d30f1ed7beb3c7e` (405 470 bytes) |

`validation/repro_traced_carrier_121/approx_common.py` prepends that tree to
`sys.path` **before** `_d121_common` prepends the live repo, and every runner
prints the two sha256s it actually imported.  `LUMEN_PIN=0` measures the live
working tree instead.

**Consequence:** the open niche-C6 defect (`remap` launches along `grad(W)`
alone, dropping the stationary-phase term that scales with `grad a`) is
PRESENT in this baseline and is deliberately NOT a row in the table below --
it is already owned and being fixed.  One incidental observation about the
in-flight version is recorded in S7.

### 0.2 The instrument, and why it is not the one ABLATE S6 broke

`approx_ablate_121.py` runs the **complete shipped path** once per row --
chain A (cached, source -> DOE) -> `TiltedCarrier(R, L, M)` for the DOE order
-> the six post-DOE groups -> the 7.7058 mm trailing leg -> the **exact**
Bluestein readout -- on a **fixed** output lattice centred on that order's
exact chief ray (`_chain_chief_ray_at_target`), and differs from the baseline
in exactly ONE construction.

There is **no hand-off plane**.  `ABLATE_LAST_GROUP_2026_07_30` S6 established
that a hand-off taken at a group's front vertex is not a valid measurement
(splitting an *innocent* leg reproduced a 7-point artefact, and the endpoints
bracketed at ~90 % while the middle dipped to 82.9 %).  Nothing is handed off
here; one construction is swapped inside a complete run and the same image
plane is read.

Configuration (all rows): `RN=1024` (`dx = 51.2334 um` at the DOE),
`ray_subsample=4`, `n_fine_cap=12288`, `window_factor=4.0`, `N_out=192`,
`dx_out=0.1 um`, `final_leg='exact'`, order `(-4,-2)` -- the worst order in
the fan and the one the previous three studies localise on.  `EE3/EE6/EE12`
are peak-centred encircled energies normalised to the chain's INPUT power, so
a row that *loses* energy is visible in `Ptile` as well.

### 0.3 Differential floor -- established, not assumed

A **NULL** row (an identity monkeypatch: `_sphere_parab_conversion` replaced
by itself) was run in the same session as the baseline:

```
NULL (identity patch)   relL2 0.000e+00   dphi 2.37e-17 rad   dEE3 +0.000e+00 pts
```

The instrument is **bit-exact**.  Any non-zero `relL2` below is a real change
in the delivered field; any `dEE3` is real to the digit printed.

### 0.4 Positive control -- the instrument is not dead

`TILTED_CARRIER_EXACT_EIKONAL = False` reverts the niche-C5 fix that landed at
`d2e60ca`.  Run in the same session, on the same lattice, it reads

```text
CONTROL tilt eikonal OFF (C5)   EE3 64.435 %   dEE3 -8.07 pts   relL2 1.716
```

so the instrument resolves a known real defect at 8 points while its null row
sits at exactly 0.  A table of nulls with a live positive control is evidence;
a table of nulls without one is not.

### 0.5 Sampling adequacy -- stated for every wave measurement

| where | measured |
|---|---|
| last group's exit NA, paraxial (`na_exit`, the value that SIZES the retrace grid) | 0.4049 |
| last group's exit NA, MEASURED from traced direction cosines | 0.5393 |
| retrace-grid Nyquist NA (`lambda / 2 dx_fine`, `dx_fine = 1.5081 um`) | 0.4343 |
| exit power above that Nyquist | **7.51e-04** |

The shipped refusal threshold (`on_tilt_exact_grid`,
`_TILT_EXACT_NA_POWER_TOL`) is 1e-2, so the production leg runs **13x inside**
its own guard.  Every end-to-end number in this document inherits that margin.

**The coarse co-moving grid is a different story and is flagged separately.**
At the last coarse group the pitch is 33.2 um against a required
`lambda/(2 NA_exit)` of 1.8-3.9 um -- 9x to 20x short, exactly the "18x" the
brief warns about.  Coarse-grid wave measurements are therefore never quoted
here as absolute wavefront figures; they appear only as *representation*
diagnostics (phase slope in cycles per pixel), which is the one quantity a
too-coarse grid makes meaningful rather than meaningless.

---

## 1. RANKED table

Ranked by MEASURED cost on design 121, worst first.  `R` = reference /
conditioning choice (a residual absorbs it exactly in the continuum);
`P` = physics approximation (information lost).  Every "cost" cell names its
method; `relL2` is the relative L2 change of the delivered complex field
against the bit-exact baseline (floor 0.000e+00), `dEE3` the change in
peak-centred EE3 in points.

| # | approximation | file:symbol | exact alternative | MEASURED cost (method) | R / P | effort | risk (on-axis byte-identity?) |
|---|---|---|---|---|---|---|---|
| 1 | `preserve_input_phase='remap'` launches along `grad(W)` only, dropping the stationary-phase term that scales with `grad a` | `_lens_traced.py`: the `remap` launch + `_pip_sample_residual` (3656-3690) | launch along `grad(W + a_fit)` (exactly what the in-flight C6 branch does) | **+15.44 EE3 pts** -- identical end-to-end config, pinned HEAD 72.501 % vs in-flight C6 snapshot 87.943 % (`Ptile` 95.65 -> 98.73) | **P** | ALREADY IN FLIGHT (441 lines) | **owned by another agent** -- not this audit's to touch.  Changes the ray launch, so on-axis is NOT byte-identical unless gated |
| 2 | `_paraxial_group_r_out` -- exit carrier radius from the group's **paraxial** ABCD via the wavefront Moebius law | `carrier.py:3139` `_paraxial_group_r_out` (via `_group_abcd` -> `seidel.system_abcd`), used at `carrier.py:5605` | the sphere fitted to the group's **own traced exit wavefront** about the traced chief ray, iterated to self-consistency (prototype: `approx_rout_traced_121.py`) | **at HEAD +5.97 EE3 pts** (72.501 -> 78.472); **against the C6 snapshot -0.16 pts** (87.771 -> 87.609, `relL2` 2.33e-02).  The HEAD figure is the open C6 defect being fed a badly-conditioned carrier, not an independent loss.  Independently: the residual it leaves is band-limited (0.062 cyc/px amplitude-weighted, 1.3e-04 of the beam above the limit, S3) | **R** -- a clean reference choice, confirmed by two independent methods | **DO NOT REPLACE** | n/a |
| 3 | `_sphere_parab_conversion`'s `cos^2` taper at `r_safe = (\|R\|^3 lambda/dx)^(1/3)` | `carrier.py:1970` `_sphere_parab_conversion` (taper at `:2050-2051`) | `T == 1`, the whole-grid convention swap | **+0.67 EE3 pts recoverable** (`relL2` 3.31e-02).  The taper reaches inside 1.63 w on the last two planes and 0.50-0.57 % of envelope power sits beyond its onset (S6) | **R** (band-limit guard) | LOW -- one constant, but the taper exists to stop aliased guard-band junk, so it needs its own convergence check | breaks byte-identity everywhere `R` is finite, on axis included |
| 4 | `fit_radius_beam_factor = 2.0` -- the ray-fit domain is tied to 2 beam radii | `_lens_traced.py:1182` `_FIT_RADIUS_BEAM_FACTOR_DEFAULT`, applied by the chain at `carrier.py:5406` | none "exact"; it is a conditioning knob | **+0.66 EE3 pts recoverable at 3.0** (`relL2` 2.55e-02) | **R** | TRIVIAL (a default) | changes the fit on every path |
| 5 | `ray_subsample = 4` -- rays launched on a `4 dx` lattice, OPL and ray-density amplitude upsampled back (`map_coordinates`, order 3 / order 1) | `_lens_traced.py:4209` (launch), `:5410` (OPL, cubic when a carrier is engaged), `:5474` (amplitude, **always linear**) | `ray_subsample = 1` | **+0.38 EE3 pts at rs=2** (`relL2` 3.87e-03); `Ptile` +0.52 | **P** | TRIVIAL (a default) x ~4 runtime | changes everything |
| 6 | `n_fine_cap = 12288` / `dx_fine = lambda/(3 NA)` heuristic with the `[0.02, 0.95]` NA clamp | `carrier.py:4248` and `:4279-4289` | no cap; `dx <= lambda/(2 NA_measured)` | **+0.015 EE3 pts at 16384** (`relL2` 3.13e-04).  Independently: `exit_power_above_nyquist` = **7.51e-04** against the 1e-2 refusal threshold | **P** where it binds | TRIVIAL x 1.8 memory/time | readout-only; changes on axis too |
| 7 | tensor-Chebyshev fit of the entrance->exit ray map, **off-centre order 10** | `_lens_traced.py:726` `_Cheb2DEvaluator`, order from `_DECENTRED_FIT_POLY_ORDER = 10` (`:1325`) | `newton_fit='spline'` (bicubic), or `ray_subsample=1` + direct lookup | **1.08e-05 waves rms** at order 10 (worst group; direct residual against traced points not on the fit lattice, S4).  End-to-end: raising to 14 gives **-0.005 EE3 pts** -- i.e. already converged, and 14 is marginally WORSE (normal-equation conditioning) | **P** | n/a -- already converged | raising only the decentred order is on-axis-inert (D7 recorded byte-identical on axis) |
| 8 | `_tilt_exactness_phase`'s `cos^2` taper, and its **`r_safe` mismatch between the coarse and fine calls** of the readout's +1/-1 pair | `carrier.py:1871`, taper at `:1965-1967`; coarse `-1` at `:2888` vs fine `+1` at `:3003` | `T == 1`, or a shared `r_safe` | **-0.0003 EE3 pts** with the taper removed entirely (`relL2` 2.42e-05).  Removing the taper also removes the mismatch, so this nulls both.  Statically: 1.2e-05 of envelope power lies beyond the onset (S6) | **R** | LOW | **on-axis byte-identical by construction** -- the function returns `None` for `L == M == 0` |
| 9 | paraxial **Fresnel / Sziklas-Siegman** envelope transfer function between groups | `carrier.py:513` `_carrier_step_fast` -> `fresnel.fresnel_tf_propagate` | `exp(i k z sqrt(1 - (lambda f)^2))` on the same reduced leg | **+0.0000 EE3 pts**, `relL2` **3.56e-07** (end-to-end substitution).  Independently: **9.4e-07 waves rms** dropped at the worst coarse leg (closed-form, S2) | **P** | LOW, but pointless here | changes every leg |
| 10 | the **unimplemented** anisotropic tilt stretch: the envelope's own diffraction uses `z`, not `z(1-M^2)/N^3` etc. | `carrier.py:3234-3265` `_tilt_obliquity` docstring names it; nothing implements it | the exact second-order form about `(L, M)` | **7.1e-05 waves rms** at the worst coarse leg (closed-form against the measured envelope spectrum, S2).  No end-to-end knob exists | **P** | MEDIUM (a new anisotropic kernel) | would change every tilted leg |
| 11 | Newton inversion cap `_NEWTON_MAX_ITERS = 12`, `tol = 0.01 dx`, clip bracket `0.999 launch_radius` | `_lens_traced.py:4859` `_invert_newton` | iterate to convergence | **`relL2` = 0.000e+00 -- BIT-IDENTICAL at 60 iterations**, despite the run reporting 2.9 % / 38.2 % / 44.3 % non-convergence at groups 3 / 4 / 5.  More iterations change nothing: the unconverged pixels are frozen (`\|det\| < 1e-12` or clipped at the bound), not slowly converging | **P** in principle | n/a | n/a -- but the warning text is misleading and should say so |
| 12 | `newton_poly_order = 6` (the concentric-branch fit order) | `_lens_traced.py` signature default | order 10+ | **`relL2` = 0.000e+00 -- BIT-IDENTICAL at 10.**  Every group on a tilted congruence takes the off-centre branch, so this knob is inert here.  Its cost on a concentric path would be **3.05e-03 waves rms** (S4) | **P** | TRIVIAL | on-axis paths WOULD change |
| 13 | glass value cache quantises the wavelength to 1 pm | `glass.py:1344` `_cached_glass_value` | no quantisation | measured `dn` over a 0.5 pm step: **0.0** for every cached glass (the quantisation is real), **-6.03e-09** for the uncached `N-BK7` (bounds it).  Over the relay's worst 25.4 mm glass path that is **1.9e-04 waves** | **P** | TRIVIAL | changes every index everywhere |
| 14 | DOE grating kick has no `1/n2` medium factor | `trace.py:215-218`, `carrier.py:4074-4075` | `dL = m lambda / (n2 Lambda)` | **INERT on design 121** -- both DGRATINGs are air-immersed (the chain hard-refuses a glass-immersed DOE gap, `carrier.py:4062`, and did not fire) | **P** (latent) | TRIVIAL | zero for air-immersed designs |
| 15 | 10-iteration Newton **surface intersection** with a `1e-15` step tolerance, non-convergence silently reclassified as a vignette | `intersection.py:276-313` | closed-form conic root | **NEVER ENTERED on design 121** -- all 16 relay surfaces are flat or `conic == 0` with no asphere.  Measured intersection residual on the closed-form path: **1.585e-10 waves** of OPL, 0 ray deaths in 25 441 rays | **P** (latent) | MEDIUM (a conic closed form) | zero for design 121 |
| 16 | `_fourier_upsample_crop` uses `//2` integer-centre anchors while every coordinate builder uses the float `N/2` convention | `carrier.py:2097, 2106, 2111` | fold `(n/2 - n//2) dx` into a shift, as `mft.py:432-435` already does | **LATENT** -- costs half a pixel of lateral registration for an ODD input `N` or an ODD user-supplied `N_fine`; the shipped path forces `n_crop` even and `n_fine` a power of two (measured run: `N` 1024, `n_fine` 12288) | **P** (latent) | LOW | a parity guard is inert for even sizes -- zero risk |
| 17 | `_group_chief_transfer` falls back to the paraxial ABCD on ANY exception, silently and unguarded | `carrier.py:3353-3356` | the exact trace above it | **NEVER FIRED** in any run here.  If it did, it would feed direction cosines into a `[height, slope]` ABCD -- the exact convention error niche C3 removed (0.1214 um/group) | **P** (latent) | TRIVIAL (add a guard) | adding a `warn` disposition is inert -- zero risk |
| 18 | the exact readout's ASM band-limit mask and evanescent cut | `mft.py:378-393` | none needed | **INERT** -- the mask sits at NA 1.2024 and the cut at NA 1.0, against a fine-grid Nyquist NA of 0.4343.  Not one bin is removed | n/a | n/a | n/a |

**Two settings that are already the right one and must NOT be "simplified":**

| setting | shipped value | cost of the alternative |
|---|---|---|
| `carrier_reference` | `'sphere'` | `'parabola'` costs **-30.15 EE3 pts** (`relL2` 0.537) |
| `remap_sampling` (chain override of the library default) | `'full'` | `'lattice'` -- which is still the *library* signature default -- costs **-17.73 EE3 pts** (`relL2` 0.361, `Ptile` -19.8) |

**Rows that could not be run** (the guard refuses them at this configuration,
which is itself information):

| row | why |
|---|---|
| `window_factor` 4 -> 6 | `RuntimeError` from `_fine_trace_group_exit`: the wider window needs `n_fine` 32768 against `n_fine_cap` 12288.  `window_factor` cannot be raised without raising the cap, so the readout window's truncation could not be priced upward |
| `newton_fit` polynomial -> spline | `ValueError` from `_fine_trace_group_exit` (retrace-grid requirement) |
| `amplitude_model` ray_density -> screen | `RuntimeError`, same guard.  So the ray-density geometric amplitude could not be compared against the screen amplitude on the exact leg |

### 1.1 How to read the ranking

**Row 2 keeps its position but not its verdict.**  It is ranked by its HEAD
reading (+5.97 pts), which the post-C6 control retracts (-0.16 pts, S3.2).
It is left at #2 so the retraction is visible rather than quietly deleted --
and because it is the worked example of the contamination that every other
non-null row is still exposed to.

The ranking is by measured cost **on design 121, order (-4,-2), at HEAD**.
Three of these rows would rank very differently on a different design and the
document says where:

* the **Fresnel kernel** and the **tilt anisotropy** scale as `NA_env^4` and
  `NA_env^2` of the *envelope's own* angular content, which is 0.0001-0.012
  here.  A design whose co-moving envelope carried 0.1 NA would move them by
  10^4 and 10^2 respectively;
* the **Chebyshev fit** converges geometrically in order but degrades steeply
  with the fit RADIUS -- see S4;
* the **closed-form intersection** is exact only because all 16 of design
  121's relay surfaces have `conic == 0` and no asphere.  One conic or
  aspheric surface routes every ray through the 10-iteration Newton with its
  hard-coded `1e-15` step tolerance and its silent
  non-convergence-becomes-a-vignette failure mode (S5.1).

---

## 2. The transport: paraxial Fresnel / Sziklas-Siegman -- priced exactly, and it is NULL

`approx_leg_budget_121.py`.  This is deliberately **not** a substitution
experiment, because the honest exact counterpart of the reduced-frame leg is
not obvious.  Instead the two dropped terms are written in closed form and
integrated against the envelope's OWN measured angular spectrum at each leg,
so there is no oracle and no differential floor to argue about.

The chain's leg is `env_out = FresnelTF(env, z_eff)`,
`z_eff = z R / (R + z)`, whose transfer phase is `k z_eff (1 - (p^2+q^2)/2)`
in direction sines `(p,q) = lambda (fx,fy)`.  Two things are dropped:

1. **the paraxial kernel truncation** -- exact is `k z_eff sqrt(1-p^2-q^2)`;
2. **the tilt anisotropy** -- for a leg carried at mean tilt `(L,M)` the exact
   second-order form is
   `-(k z_eff/2)[ (1-M^2)/N^3 p^2 + 2 (LM/N^3) pq + (1-L^2)/N^3 q^2 ]`,
   `N = sqrt(1-L^2-M^2)`, while the chain applies the isotropic
   `-(k z_eff/2)(p^2+q^2)`.  The zeroth (piston) and first (chief-ray advance)
   orders ARE handled exactly by `_tilt_obliquity`; only this second-order
   stretch is unimplemented (the docstring at `carrier.py:3262-3265` names it).

Order `(-4,-2)`, tilt 51.50 mrad, `RN=1024`.  Power-weighted rms and the max
over the band holding 99.9 % of the envelope's spectral power, both in WAVES:

| leg | z (mm) | z_eff (mm) | R_in (mm) | dx (um) | NA(99.9%) | Fresnel rms | Fresnel max | tilt rms | tilt max |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 7.0000 | 7.0000 | 703642.7 | 51.23 | 0.00012 | 8.78e-09 | 5.93e-13 | 5.87e-07 | 1.63e-07 |
| 1 | 5.0000 | 5.0000 | 703664.8 | 51.23 | 0.00012 | 4.16e-09 | 4.24e-13 | 3.11e-07 | 1.16e-07 |
| 2 | 5.0000 | 5.0000 | 703671.9 | 51.23 | 0.00012 | 3.83e-09 | 4.24e-13 | 2.84e-07 | 1.16e-07 |
| 3 | 32.4787 | 37.0508 | -263.19 | 51.23 | 0.00390 | 1.45e-07 | 1.02e-06 | 4.18e-05 | 7.96e-04 |
| 4 | 8.6779 | 10.1410 | -60.15 | 44.91 | 0.00349 | 3.67e-08 | 2.14e-07 | 8.58e-06 | 1.66e-04 |
| 5 | 3.3233 | 3.8457 | -24.46 | 38.43 | 0.01198 | **9.36e-07** | 5.72e-05 | **7.08e-05** | 1.20e-03 |

**Verdict.**  Over the six coarse legs the paraxial kernel costs at most
**9.4e-07 waves rms** (worst leg) and 5.8e-05 waves summed as band maxima; the
unimplemented tilt anisotropy costs at most **7.1e-05 waves rms** and 2.2e-03
waves summed as band maxima.  Both are three to five orders of magnitude below
anything that moves an EE3 point.  The reason is visible in the `NA(99.9%)`
column: the co-moving envelope's own angular content is 0.0001-0.012 NA, and
the dropped terms scale as NA^4 and NA^2 respectively.

This is the quantitative form of what niche C3 established qualitatively (the
quartic cancels exactly under `carrier_reference='sphere'`): **what is
genuinely left is nothing.**  Neither an ASM envelope kernel nor the
anisotropic tilt stretch is worth implementing for design-121-class geometry.

**Confirmed end to end, independently.**  Substituting the exact
`exp(i k z sqrt(1 - (lambda f)^2))` kernel for `fresnel_tf_propagate` on every
leg of the complete shipped run gives

```text
Fresnel kernel -> exact kz   EE3 72.501 %   dEE3 +0.0000 pts   relL2 3.56e-07
```

against a bit-exact null floor.  The closed-form budget and the substitution
agree that this term is dead.

A seventh leg appears in the trace (`z=7.7058 mm`, `R_in=-7.712 mm`,
`z_eff = 8970 mm` -- the carrier is 6 um from its focus).  It is the
`final_leg='paraxial'` route used to make the capture cheap, and its numbers
(0.51 waves Fresnel, 9.85 waves tilt) are exactly why the shipped
`final_leg='auto'` refuses it: `na_exit = 0.405 > na_exact_threshold = 0.15`,
so production takes the exact Bluestein leg and that row never happens.  It is
excluded from the verdict above, and its `z_eff` model does not even describe
what the code does there (the near-focus ASM bridge fires).

---

## 3. `_paraxial_group_r_out` -- wrong by up to 3.5 %, and worth +5.97 EE3 points

`approx_reference_fit_121.py` PART 1.  The chain's exit carrier radius comes
from the group's air-to-air **paraxial** ABCD (`seidel.system_abcd`, a
linearised `phi = (n2-n1)/R` y-nu trace) mapped by the wavefront Moebius law
`R_out = (A R_in + B)/(C R_in + D)` (`carrier.py:3139-3154`, used at
`carrier.py:5605`).  The exact counterpart is the sphere that fits the group's
**own traced exit wavefront** about the traced chief ray.

Method: for each group, the chain's real inputs are captured
(`apply_real_lens_traced` is wrapped, not modified); the group's own
congruence is launched on a 161x161 lattice out to 1.6 beam radii with
directions from `_tilted_carrier_parts` (analytic, exact); the rays are traced
through the group's own surfaces with apertures stripped and advanced to the
exit VERTEX plane exactly as the element does (`t = -z/N`, `opl += n_exit t`);
the entrance eikonal `W` is added, as the element adds it.  The residual is
then `a = W_exit - S_chain`, with `S_chain` the exact
`TiltedCarrier(R_out, L_out, M_out, x_c_out, y_c_out)` eikonal the chain
actually writes.

The residual's phase SLOPE on the co-moving grid is the quantity that decides
whether a reference is absorbed or not: the residual enters as
`exp(i k0 a)`, so `|grad a| dx / lambda` must stay under **0.5 cycles per
pixel** or the envelope is not a faithful representation at all.  It is
reported both as a 99th percentile at three radii and as an
**amplitude-weighted** rms, plus the amplitude-weighted fraction of the beam
above 0.5.

| group | dx (um) | R_out paraxial (mm) | R_out traced-fit (mm) | frac err | defocus slope | p99 @0.5w | p99 @1w | p99 @1.5w | **ampl-wtd rms** | frac >0.5 | a rms (wv) | a rms -defocus |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Lens S14-S15 | 51.234 | 703664.7904 | 703664.7711 | 2.7e-08 | 2.2e-11 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0.000 | 0.000 |
| Lens S16-S17 | 51.234 | 703671.9187 | 703671.9131 | 7.9e-09 | 6.3e-12 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0.000 | 0.000 |
| Lens S18-S20 | 51.235 | -263.1942 | -276.9551 | **5.2e-02** | 0.104 | 0.005 | 0.018 | 0.078 | **0.016** | 0 | 3.193 | 1.291 |
| Lens S21-S22 | 44.912 | -60.1480 | -59.7979 | 5.8e-03 | 0.037 | 0.004 | 0.012 | 0.028 | **0.008** | 0 | 0.993 | 0.242 |
| Lens S23-S24 | 38.432 | -24.4625 | -23.8503 | **2.5e-02** | 0.252 | 0.035 | 0.132 | 0.320 | **0.062** | 1.3e-04 | 7.847 | 2.543 |
| Lens S25-S27 | 33.211 | -7.7124 | -8.1743 | **6.0e-02** | 0.542 | 0.122 | 0.374 | 0.796 | **0.157** | 1.9e-02 | 7.976 | 2.777 |

Column definitions:

* **defocus slope** = the extra envelope phase slope the *reference error
  alone* leaves at the **beam edge of the launch disc (1.6 w)**,
  `|1/R_fit - 1/R_out| * r_edge * dx / lambda`.  This is an edge-of-skirt
  number and is retained only for comparison with the earlier revision.
* **p99 @Xw / ampl-wtd rms** = the 99th percentile and the amplitude-weighted
  (Gaussian `exp(-2 r^2/w^2)`) rms of `|grad a| dx / lambda` for the FULL
  residual, aberration included.
* `a rms` is the whole residual (physics + reference); `-defocus` is the
  genuine aberration no reference choice can take away.

**Reading.**  On the two near-collimated groups the paraxial law is exact to
1e-8 -- as it must be, since `R -> inf` is where it is derived.  On the
powered groups it is wrong by 0.6-6 % against this (unweighted, 1.3 w) fit,
and by 0.07-1.2 % against the self-consistent amplitude-weighted one of S3.1.
The residual it leaves is nonetheless comfortably band-limited: the worst
coarse group (`Lens S23-S24`) carries an amplitude-weighted rms slope of
**0.062 cyc/px**, 12 % of the limit, with **1.3e-04** of the beam above it.
`Lens S25-S27` reads worse on the coarse grid (0.157, 1.9e-02 above), but on
the shipped `final_leg='exact'` route its exit is retraced at
`dx_fine = 1.5081 um`, so those columns scale by `1.5081/33.211 = 0.0454` --
0.007 cyc/px and nothing above the limit.  **Read S3.1 before drawing any
conclusion from that: the band-limit argument is necessary but not
sufficient, and the end-to-end measurement overrules it.**

### 3.1 The representation argument says "harmless"; the end-to-end measurement says otherwise

The slope table above is a *necessary* condition, not a sufficient one, and
taking it as sufficient would have been this audit's third artefact.  The
end-to-end rows settle it:

| intervention | EE3 % | dEE3 (pts) | relL2 |
|---|---|---|---|
| baseline (paraxial Moebius `R_out`) | 72.501 | -- | 0.000e+00 |
| `R_out` x (1 + 1e-3) | 71.099 | **-1.40** | 1.377 |
| `R_out` x (1 + 1e-2) | 50.403 | **-22.10** | 1.860 |
| **traced best-fit exit sphere, self-consistent** | **78.472** | **+5.97** | 1.504 |

The last row is the decisive one: replacing the paraxial Moebius radius with
the sphere fitted to each group's own traced exit wavefront -- iterated twice
to self-consistency, because each group's `R_out` is the next group's `R_in`
-- is worth **+5.97 EE3 points, +3.36 EE6 points and +1.76 points of power
into the readout tile**.  The converged overrides are

```text
703664.741240  703671.823767  -263.697544  -60.009416  -24.168812  -7.762330   (mm)
```

against the shipped `703664.790420  703671.918670  -263.194184  -60.147979
-24.462480  -7.712425`, i.e. 1e-8 on the collimated pair and 0.07 % / 0.23 %
/ 1.20 % / 0.65 % on the powered four.

**Why a "reference" costs 6 points.**  `R_out` is not only the phase that is
divided out.  The same number is (i) the physical curvature the
Sziklas-Siegman step uses to set `z_eff = z R/(R+z)` and the magnification
`m = R_out/R` on the following leg, (ii) the carrier the NEXT group's element
launches its rays along -- `carrier=TiltedCarrier(R_use, ...)`, so a 1 % error
mis-points every launch ray by ~1 % of the convergence angle -- and (iii) the
`na_exit = w/|R_out|` that sizes the fine retrace grid and routes
`final_leg='auto'`.  Only (i)'s phase part is absorbed.

### 3.2 ...and the post-C6 control takes it away again

At HEAD the element transports the carried residual with
`preserve_input_phase='remap'`, whose error scales with `grad a` -- and a
mis-set carrier is exactly what makes `grad a` large.  So the +5.97 might be
the open C6 defect being fed a better-conditioned carrier rather than an
independent gain.  The identical experiment was therefore repeated against a
frozen snapshot of the in-flight C6 tree (`_lens_traced.py` sha256
`f06da6ab8e15ce2a`, 381 322 bytes; `carrier.py` unchanged):

| library | baseline EE3 % | traced exit sphere EE3 % | delta EE3 | delta EE6 | relL2 |
|---|---|---|---|---|---|
| pinned HEAD `d2e60ca` | 72.501 | 78.472 | **+5.97** | +3.36 | 1.504 |
| C6 snapshot | 87.771 | 87.609 | **-0.16** | -0.08 | 2.33e-02 |

The fit itself is library-independent -- the two trees converge to the same
overrides to five digits -- so the difference is entirely in how much the
chain CARES about the reference.  Post-C6 it does not: the gain collapses by
97 % and changes sign.

**Final classification: a clean REFERENCE choice.  Leave it alone.**  Two
independent methods now agree -- the representation diagnostic (the residual
is band-limited with ~8x of headroom) and the post-C6 ablation (-0.16 pts).
The HEAD ablation disagreed because it was measuring the C6 defect through a
proxy.  The `x(1+1e-3)` / `x(1+1e-2)` sensitivity rows carry the same
contamination and should be read the same way.

**Caveat, stated because it matters for the fix.**  "The exact R_out" is not
unique -- it depends on the weighting of the wavefront fit.  An
amplitude-weighted fit (`approx_rout_traced_121.py`, Gaussian weight
`exp(-2 r^2/w^2)`) gives *different* radii from the unweighted 1.3 w fit
above: `-263.698` vs `-276.955` mm at group 2, `-7.983` vs `-8.174` mm at
group 5.  The right objective is not "minimise the wavefront rms" but
"minimise the residual's peak SLOPE", which is what conditioning means.  Any
replacement must state its objective.

---

## 4. The tensor-Chebyshev fit of the entrance -> exit ray map

`approx_reference_fit_121.py` PART 2.  The element's fit stage is reproduced
verbatim -- the same `_Cheb2DEvaluator`, the same total-degree tensor basis,
the same launch lattice pitch (`dx * ray_subsample`, count bumped odd), NaN
for dead rays, piston removed -- then evaluated at 7 800+ **traced** points
that are not fit nodes, inside one beam radius.  rms errors:

| group | order 4 | order 6 | order 8 | **order 10** | order 12 | order 14 |
|---|---|---|---|---|---|---|
| S14-S15 OPL (waves) | 2.7e-08 | 2.7e-08 | 2.7e-08 | 2.7e-08 | 2.7e-08 | 2.7e-08 |
| S16-S17 OPL | 2.6e-08 | 2.6e-08 | 2.6e-08 | 2.6e-08 | 2.6e-08 | 2.6e-08 |
| S18-S20 OPL | 5.33e-02 | **3.05e-03** | 1.73e-04 | **1.08e-05** | 6.38e-07 | 4.50e-08 |
| S21-S22 OPL | 8.90e-03 | 5.46e-05 | 3.73e-07 | 2.91e-09 | 3.27e-11 | 2.42e-11 |
| S23-S24 OPL | 4.59e-02 | **1.53e-03** | 5.34e-05 | **2.09e-06** | 7.65e-08 | 2.77e-09 |
| S25-S27 OPL | 1.88e-02 | **7.69e-04** | 1.90e-05 | **2.41e-06** | 1.02e-07 | 5.29e-09 |
| S18-S20 x_out (um) | 9.97e-01 | 3.59e-02 | 1.57e-03 | 7.95e-05 | 3.82e-06 | 1.84e-07 |
| S23-S24 x_out (um) | 1.38e-01 | 3.20e-03 | 9.07e-05 | 3.07e-06 | 1.01e-07 | 3.40e-09 |

(The 2.7e-08 floor on the two collimated groups is the arithmetic floor of the
piston removal, not a fit error -- their map is affine.)

**Verdict.**  At the shipped **off-centre order 10** -- which is what every
tilted group takes, `_DECENTRED_FIT_POLY_ORDER = 10` at
`_lens_traced.py:1325` -- the fit costs **1.1e-05 waves rms** at the worst
group.  At the on-axis default `newton_poly_order = 6` it costs **3.05e-03
waves**.  Both are negligible; the brief's earlier probe value of 0.00045
waves sits between the two and is consistent with an order-6-to-10 mixture.

**Where it stops being negligible.**  The convergence is geometric and
measured: each +2 orders divides the error by 17-20 at S18-S20, ~150 at
S21-S22.  Running the *decentred* branch at order 6 instead of 10 would cost
280x more (3.05e-03 waves) -- still small.  The fit only becomes a live risk
if the fit DOMAIN grows: the launch disc here is 1.6 w, and the error grows
steeply with radius because the polynomial must then represent marginal rays.
That is exactly what `fit_radius_beam_factor` controls, and it is priced
end-to-end in S1.

**Confirmed end to end.**  On the complete shipped run,

```text
newton_poly_order 6 -> 10            dEE3 +0.0000   relL2 0.000e+00  (BIT-IDENTICAL)
decentred_fit_poly_order 10 -> 14    dEE3 -0.0051   relL2 2.54e-02
```

The first is bit-identical because every group on a tilted congruence takes
the off-centre branch, so `newton_poly_order` is never read here.  The second
shows the shipped order 10 is already converged -- order 14 is marginally
WORSE (EE6 -0.33), which is what the normal-equation conditioning
(`cond(Gram)` up to 1.9e13 per the library's own docstring) predicts.

**Caveat.**  This reproduction uses the *concentric* branch (hard NaN mask
over a disc, unweighted).  The shipped off-centre branch uses WEIGHTED least
squares over the whole launch square with weight
`sqrt(1e-8 n_in/n_out)` outside the disc, solved by normal equations
(`_lens_traced.py:855`, `cond(Gram)` up to 1.9e13 per its own docstring).  The
numbers above price the BASIS's approximation power, not that conditioning.
The end-to-end rows `newton_poly_order 6 -> 10` and
`decentred_fit_poly_order 10 -> 14` in S1 price the shipped branch.

---

## 5. The layers that are ALREADY EXACT -- do not "fix" these

`approx_raytrace_census.py`, `approx_static_checks.py`.

1. **Surface intersection.**  All **16** surfaces of design 121's post-DOE
   relay are flat or `conic == 0` with no asphere, so every intersection takes
   the **closed-form quadratic** root (`intersection.py:100-107`).  The
   10-iteration Newton path (`intersection.py:276-313`) is **never entered on
   this design**.  Measured residual `|z - sag(x,y)|` at every recorded
   intersection of a 25 441-ray bundle at the (-4,-2) tilt: max **1.585e-10
   waves** of OPL (worst surface, N-LAK8).  That is float64 round-off.
2. **Ray deaths.**  Zero.  25 441 / 25 441 survive a 3.2 mm-radius bundle
   through the whole relay -- no TIR, no aperture clip, no
   `RAY_MISSED_SURFACE`.  The hard-edge pupil the audit lists as a hazard
   (`_lens_traced.py:5417-5428`) does not bite on this design at this radius.
3. **Refraction.**  Full vector Snell (`_conic_core.py:162-179`), no
   small-angle form, TIR by radicand sign, no `arcsin` anywhere on the traced
   path.
4. **OPD accumulation.**  `n_medium * t` per segment with `|(L,M,N)| = 1`,
   index resolved per surface at the exact trace wavelength
   (`trace.py:149-150`).  No reference index is substituted.
5. **Glass / dispersion.**  Six of design 121's seven relay glasses are
   closed-form Sellmeier.  `N-BK7` alone resolves through the external
   `refractiveindex` package -- verified to land on its **formula** (Sellmeier-2)
   branch, not a tabulated interpolation.  The value cache quantises the
   wavelength to 1 pm (`glass.py:1344`); measured `dn` over a 0.5 pm step is
   **0.0** for every cached glass (i.e. the quantisation is real) and
   **-6.03e-09** for the uncached `N-BK7`, which bounds the quantisation
   error.  Over the relay's worst single glass path (25.4 mm) that is
   **1.9e-04 waves**.  Real, bounded, and irrelevant at this scale.
6. **DOE grating equation.**  Applied in direction cosines
   (`trace.py:213-237`, `carrier.py:4073-4078`), with the z-cosine from the
   unit-vector constraint and evanescent rejection.  This is exact at any
   angle, including conical -- the `trace.py:1131-1138` docstring calling it
   "small-angle / paraxial" is WRONG and should be corrected, not the code.
   Design 121's two DGRATINGs are air-immersed (the chain refuses a
   glass-immersed DOE gap, `carrier.py:4062`, and did not fire), so the one
   real gap in that layer -- the missing `1/n2` medium factor in the kick --
   is inert here.  `_doe_axes` snaps the 270.0 deg azimuth of the second
   grating to exact `(0,-1)`, removing a 1.8e-16 trig error.
7. **Chief-ray transfer.**  Exact ray trace through the group's own surfaces
   since `7f45874`; the paraxial ABCD survives only as a silent
   exception-handler fallback which did not fire in any run here.
8. **The readout's band limits are INERT.**  On the shipped fine grid
   (12288 x 1.5081 um = 18.532 mm) over the 7.7058 mm final leg, the
   Matsushima-Shimobaba band-limit mask sits at **NA 1.2024** and the
   evanescent cut at NA 1.0, both far above the grid Nyquist NA of **0.4343**.
   Neither removes a single bin.  The readout transform itself is a Bluestein
   / chirp-Z evaluation of the exact centred DFT -- not a padded FFT, not an
   approximation.
9. **No interpolation in `carrier.py` at all.**  Every regrid is spectral
   (`_fourier_upsample_crop`, `_shift_envelope`, `_crop_about_centre`).  The
   only interpolations in the chain are inside `apply_real_lens_traced`, and
   are priced by the `ray_subsample` row of S1.
10. **`_fourier_upsample_crop`'s half-pixel parity trap is LATENT, not
    active.**  The `//2` integer-centre anchors mis-register by half a pixel
    for an ODD input `N` or an ODD user-supplied `N_fine`.  The shipped chain
    forces `n_crop = 2*round(...)` (even) and `n_fine = 2**ceil(...)` capped by
    `n_fine_cap`; the measured run used `n_fine = 12288`, `N = 1024`, both
    even.  Reachable only by a user passing an odd `N_fine` or an odd input
    grid, neither of which is validated -- worth a parity guard, but it costs
    nothing today.
11. **`_tilt_ramp`, `_tilt_obliquity`, `_exact_sphere_eikonal`,
    `_tilted_carrier_parts`, the Chebyshev recurrences.**  Closed forms,
    exactly invertible.  The obliquity `1/sqrt(1-L^2-M^2)` is the exact factor,
    not a small-angle expansion.

---

## 6. The two band-limit tapers -- geometry measured

`approx_static_checks.py`.  Both `_sphere_parab_conversion` and
`_tilt_exactness_phase` roll off as `cos^2` from `0.75 r_safe` to `r_safe`.
The question is only whether the taper reaches the beam.

**`_sphere_parab_conversion`,** `r_safe = (|R|^3 lambda / dx)^(1/3)`:

| plane (R) | dx (um) | w (um) | r_safe (mm) | 0.75 r_safe / w | envelope power beyond onset |
|---|---|---|---|---|---|
| 703642.7 mm | 51.23 | 6318 | 207296 | 24607 | 0 |
| -263.2 mm | 51.23 | 5989 | 77.54 | 9.71 | 0 |
| -60.1 mm | 44.91 | 4847 | 18.52 | 2.87 | 1.6e-09 |
| -24.5 mm | 38.43 | 3622 | 7.93 | 1.64 | **5.0e-03** |
| -7.7 mm | 33.21 | 1205 | 2.63 | 1.63 | **5.7e-03** |

**`_tilt_exactness_phase`,** `r_safe` from the coma+astigmatism slope solve --
and note this one's radius is **not the same on the two calls of the exact
readout's +1/-1 pair**, because the `-1` runs at the coarse `dx` and the `+1`
restore at `dx_fine`:

| plane (R) | dx | r_safe (mm) | 0.75 r_safe / w | power beyond onset |
|---|---|---|---|---|
| -263.2 mm | coarse 51.23 um | 102.6 | 12.85 | 0 |
| -263.2 mm | fine 1.508 um | 619.6 | 77.59 | 0 |
| -60.1 mm | coarse 44.91 um | 25.12 | 3.89 | 1.2e-10 |
| -24.5 mm | coarse 38.43 um | 11.08 | 2.29 | **1.2e-05** |
| -24.5 mm | fine 1.508 um | 57.59 | 11.93 | 0 |

So the sphere/parabola taper touches **0.5-0.6 % of the envelope power** at
the last two planes, and the tilt-exactness taper touches 1.2e-05 of it; the
coarse/fine radius mismatch can only act inside that 1.2e-05.  Settled
end to end by replacing `T` with 1 (the whole-grid swap), which also removes
the coarse/fine mismatch:

```text
sphere<->parab taper OFF    dEE3 +0.6719   dEE6 +0.0554   relL2 3.31e-02
tilt-exactness taper OFF    dEE3 -0.0003   dEE6 +0.0000   relL2 2.42e-05
```

The tilt-exactness taper (and therefore its `r_safe` mismatch) is a measured
null.  The sphere/parabola taper is NOT: removing it **improves** design 121
by 0.67 EE3 points and 0.22 points of tile power.  That contradicts the
function's own docstring, which records `r_safe*1.5` and `r_safe=inf`
reproducing "the shipping design-121 result to the digit" -- true on the
ON-AXIS best-focus metrics that docstring quotes, evidently not on a tilted
congruence at 51 mrad, where the taper onset sits at 1.63 w and 0.5 % of the
power is in a mixed-convention region.  The docstring should be narrowed.

---

## 7. What I could NOT measure, and why

1. **Whether the in-flight niche-C6 change alters the fold-caustic verdict.**
   On the LIVE working tree (pre-pin), the shipped exact leg for order (-4,-2)
   emitted `amplitude_model='ray_density' detected a fold caustic (det J -> 0
   or a sign change)` from `_fine_trace_group_exit`.  On the **pinned HEAD**
   the same run emits no such warning.  If that difference is real (rather
   than Python's once-per-location warning de-duplication), the C6
   stationary-phase launch introduces a fold in the ray map where HEAD has
   none -- which would matter, because the single-branch ray-density amplitude
   is CAPPED at a fold and does not sum branches with KMAH/Maslov phase.
   The mechanism is available: the C6 diff makes `remap` launch along
   `grad(W + a_fit)` instead of `grad(W)`, which changes the entrance->exit
   map and therefore `det J`.  (The C6 author is evidently tracking this --
   their own added docstring carries a `fold` column next to `ghost power`.)
   A second, independent trace of the same difference: the last group's
   `exit_power_above_nyquist` reads **7.979e-04** on the live tree and
   **7.513e-04** pinned, on otherwise identical runs -- the exit ray map is
   genuinely different, so a `det J` sign change is not far-fetched.
   I could not run the controlled live-vs-pinned pair: at the time, the three
   ablation batches held ~40 GB and system available memory hit 0.
   It nonetheless revises the brief's premise that "the fold-caustic warning
   never fires on design 121" -- it does not fire at HEAD, but it did on the
   in-flight tree, and the note may be stale in either direction.
2. **The exact multibranch alternative to `amplitude_model='ray_density'`.**
   `caustic='multibranch'` RAISES on this configuration (already recorded in
   `SCOPE_TILTED_COARSE_LEG_TRANSPORT` S4 knockouts), so the KMAH/Maslov sum
   could not be run as a comparison.  What S1 prices instead is
   `ray_density` -> `screen`, i.e. the geometric-optics magnitude against the
   thin-element phase-screen magnitude -- a different question.  The true
   caustic-faithful reference would be `apply_real_lens_gbd` /
   `apply_real_lens_fga`, which are separate propagators, not knobs.
3. **Whether the OTHER non-null rows survive the C6 fix.**  This is the
   biggest gap in the document and it is a systematic one.  `R_out` was
   re-measured post-C6 and its 5.97-point gain collapsed to -0.16 (S3.2).
   Every other non-null row -- `carrier_reference` (-30.15),
   `remap_sampling` (-17.73), the C5 control (-8.07), the sphere/parabola
   taper (+0.67), `fit_radius_beam_factor` (+0.66), `ray_subsample` (+0.38)
   -- perturbs the same carrier/residual conditioning that C6 couples to, and
   NONE of them was re-measured post-C6.  There was time for exactly one such
   control and it was spent on the row the audit was about to promote.
   **Every non-null row is an upper bound until the landed C6 is re-measured.**
   The null rows (`relL2` <= 1e-3: the Fresnel kernel, `n_fine_cap`,
   `newton_max_iters`, `newton_poly_order`, the tilt-exactness taper) are
   safer but not proven: a construction that is inert at HEAD could in
   principle matter post-C6.
4. **The amplitude-weighted form of the residual-slope metric.**  The 0.619
   cyc/px figure is the 99th percentile over the beam DISC, i.e. area-weighted,
   not power-weighted.  A power-weighted version would be smaller; the sign of
   the conclusion (group 4 is at or past its representation limit) is robust
   to that, but the exact fraction of power that aliases is not measured.
5. **Orders other than (-4,-2), and the multi-congruence orchestrator.**
   Everything here is one congruence.  `propagate_traced_carrier_chain_multi`'s
   readout tiling, replica guard and per-frame recombination are untouched;
   the tile-placement snap was verified by code inspection to move the WINDOW
   and not the field (each tile is a full exact MFT evaluated at the true
   physical coordinates), but not measured.
6. **The on-axis 2.34-point floor.**  Out of scope here as in the two prior
   studies.

---

## 8. Measurement artefacts found and killed in MY OWN instruments

Recorded because ~20 artefacts have passed as findings in this project.

1. **The library was a moving target.**  The first batch of ablations died on
   a half-saved `_lens_traced.py` (`NameError: _REMAP_RESID_TAPER_IN`).  Had
   it merely *changed* rather than crashed, every delta in the table would have
   been a mixture of my intervention and someone else's edit.  Fixed by pinning
   to a `git archive HEAD` export and printing the sha256 of the two files
   actually imported in every run.
2. **The pin did not take on the first attempt.**  `_d121_common` prepends the
   live repo root at import time, so inserting the pinned path first was not
   enough -- `lumenairy` was already resolved from the live tree.  Caught by
   printing `LT.__file__`.  Fixed by importing the package from the pin BEFORE
   importing `_d121_common`.
3. **`z_eff` for the seventh leg is meaningless.**  The leg-budget script
   computes `z_eff = z R/(R+z)` analytically, which is only what the code does
   on the NON-crossing fast path.  The trailing leg lands 6 um from the
   carrier focus, where `_near_focus_needs_bridge` fires and an ASM bridge runs
   instead.  Its 0.51 / 9.85 wave numbers are excluded from the verdict and are
   labelled as the paraxial route the shipped `final_leg='auto'` refuses.
4. **rms > band-max in the leg table is not a bug.**  The power-weighted rms
   integrates over ALL spectral bins, including a far tail with negligible
   power but a large dropped phase; the "max" column is restricted to the band
   holding 99.9 % of the power.  Both are reported so the reader can see which
   is which; the rms is the figure that maps onto a Strehl.
5. **My own headline was wrong TWICE on the same row, in opposite
   directions, before the ablation settled it.**  (a) The first S3 table
   reported the residual slope as a bare 99th percentile of `|grad a|` over
   the whole 1.6 w launch disc.  At 1.6 w the Gaussian amplitude is
   `exp(-2*1.6^2) = 0.6 %` of peak, so a handful of skirt samples set the
   number: it read **0.619 cyc/px at group 4 -- "past Nyquist"** -- and the
   document called `_paraxial_group_r_out` a conditioning defect on that
   basis.  This is precisely artefact 4 of
   `DIAG_LAST_GROUP_DECENTRE_2026_07_30` S8 ("the nn-step sampling diagnostic
   was reported as a MAX and read pi on configurations whose core was clean to
   0.02 rad"), reproduced independently in a new instrument.  (b) The
   amplitude-weighted rms of the same field is **0.062 cyc/px** with
   **1.3e-04** of the beam above the limit, so the next revision called the
   row harmless.  That was wrong too: a band-limit argument is a NECESSARY
   condition for a reference to be absorbed, not a sufficient one, and this
   particular "reference" is also a physical transport parameter.  The
   end-to-end ablation at HEAD says **+5.97 EE3 points**, so the document
   promoted the row to #2.  (c) The post-C6 control then read **-0.16
   points** and the row went back to "leave it alone" -- the HEAD ablation
   had been measuring the OPEN C6 DEFECT through a proxy.  Three readings,
   two reversals, one row.
   **The new artefact class this exposes, which is not in any of the three
   prior catalogues:** *an ablation run against a tree with a large open
   defect can attribute that defect's sensitivity to an innocent
   construction, and it will pass every convergence check you throw at it
   because it is a real, reproducible, bit-exact difference -- of the wrong
   thing.*  The only defence is a control run against a tree WITHOUT the open
   defect.  This audit could afford exactly one such control; S7.3 says which
   rows still lack it.
6. **"The exact R_out" is weighting-dependent** and the first version of the
   fit quietly used an unweighted disc.  Reported both (S3 caveat) rather than
   picking one, because a reference's replacement has to state its objective.
7. **The first census bundle crashed `make_ray`.**  `make_ray` wraps its scalar
   arguments in lists, so passing arrays produced a nested-array broadcast
   error; the bundle builder `_make_bundle` is the array-valued entry point.
   Caught immediately by the traceback, not by a wrong number -- but the same
   mistake with a silently-broadcasting API would have traced 1 ray and called
   it 25 441.
8. **Warning de-duplication can hide a regime change.**  Each ablation process
   runs many chains; Python's default filter prints a given warning once per
   location, so "no fold-caustic warning in the log" is NOT proof the fold
   never occurred after the first run.  This is exactly why item 7.1 is filed
   as unmeasured rather than as a finding.

---

## 9. Reproduction

All commands from `validation/repro_traced_carrier_121/`.  Chain A is cached to
`_chainA_1024_2000nm_rs4.npz` on first use.  The pinned library tree is created
once with

```bash
git archive HEAD lumenairy | tar -x -C <scratchpad>/pin_d2e60ca
```

and is picked up automatically by `approx_common.py` (`LUMEN_PIN=0` to measure
the live tree instead).

```bash
# 1. Raytrace / glass / DOE exactness census (~1 min, no chain).
ORD=-4,-2 python approx_raytrace_census.py

# 2. Exact per-leg accounting of the dropped Fresnel + tilt-anisotropy terms
#    (~3 min; runs the chain once on the paraxial route to capture the legs).
ORD=-4,-2 python approx_leg_budget_121.py

# 3. R_out reference error + Chebyshev fit-order sweep (~5 min).
ORD=-4,-2 NW=4 python approx_reference_fit_121.py

# 4. Taper geometry, ASM band-limit inertness, upsample parity (~3 min).
ORD=-4,-2 NW=4 python approx_static_checks.py

# 5. Harness smoke + differential floor (2 full runs, ~15 min).
ORD=-4,-2 python approx_smoke.py

# 6. The ablation table.  ~7 min per row on an IDLE machine; do NOT run the
#    three batches concurrently -- three 12288-point fine retraces exhaust
#    128 GB and the third process is killed.
ORD=-4,-2 SET=ctl,a python approx_ablate_121.py
ORD=-4,-2 SET=b,c   python approx_ablate_121.py
ORD=-4,-2 SET=d     python approx_ablate_121.py

# 7. The self-consistent traced exit sphere, end to end (~25 min IDLE).
ORD=-4,-2 ITERS=2 python approx_rout_traced_121.py

# 8. THE CONTROL that retracts row 2: the same experiment against a tree
#    WITHOUT the open C6 defect.  Snapshot the working tree first so the
#    control is itself reproducible:
#        cp -r lumenairy <scratch>/pin_live_c6/ && rm -rf <scratch>/pin_live_c6/**/__pycache__
LUMEN_PIN='<scratch>/pin_live_c6' ORD=-4,-2 ITERS=2 python approx_rout_traced_121.py
```

Runtime warning: a single row is ~7 min on an idle 24-core / 128 GB box at
`n_fine_cap=12288`.  Three batches in parallel drove available memory to zero
and killed a fourth process; run them one at a time.

### Files added by this study (none are library code)

```
validation/repro_traced_carrier_121/approx_common.py            pinned harness + metrics
validation/repro_traced_carrier_121/approx_smoke.py             floor + positive-control smoke
validation/repro_traced_carrier_121/approx_ablate_121.py        end-to-end ablation table
validation/repro_traced_carrier_121/approx_leg_budget_121.py    per-leg dropped-term budget
validation/repro_traced_carrier_121/approx_reference_fit_121.py R_out + Chebyshev fit
validation/repro_traced_carrier_121/approx_rout_traced_121.py   traced exit sphere, end to end
validation/repro_traced_carrier_121/approx_raytrace_census.py   raytrace/glass/DOE exactness
validation/repro_traced_carrier_121/approx_static_checks.py     tapers, band limits, parity
```

Raw run logs (the evidence behind every number, kept alongside the existing
`_chainA_*.npz` cache convention):

```text
_approx_out_ctl_a.txt   _approx_out_b_c.txt   _approx_out_d.txt   ablation batches
_approx_reffit2_-4-2.txt                        R_out + Chebyshev (final)
_approx_reffit_-4-2.txt                         R_out (first revision, skirt artefact)
_approx_rout_-4-2.txt                           traced exit sphere, HEAD
_approx_rout_c6_-4-2.txt                        traced exit sphere, C6 snapshot (the control)
_approx_static.txt                              tapers / band limits / parity
```

No library file was modified.  Every "exact alternative" is installed as a
runtime monkeypatch on a module attribute inside an `approx_common.Patch`
context manager and removed in its `finally`.

# VERIFY-WP-B3 — adversarial re-verification of the propagator-kernel work package (K6, K9, K13, K22)

Verifier: VERIFY-B3 (independent; did not write WP-B3).
Subject: commit **`284daccc`** on `audit-fixes-2026-09` (parent **`81d5b586`** = release
5.46.0), the report `fixes/WP-B3_REPORT.md`, the changelog `fixes/WP-B3_CHANGELOG.md`,
and `tests/unit/test_audit2609_b3_propagator_kernels.py`.

Every claim below was **re-measured, not read**. Where WP-B3 used an oracle I built a
different one, on a fixture WP-B3 did not use. Date of every measurement: **2026-09-13**,
this host, every Python run under
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process at a time.
No git write command was run; the pre-change and post-change libraries were obtained with
`git archive` into read-only scratch trees.

> **Tree note, and it matters for §4.** Three other Wave-4 engineers edited this working
> tree throughout. During my run `lumenairy/raytrace/{trace,intersection,surface}.py`,
> `lumenairy/elements/rcwa/*`, `carrier.py`, `_lens_real.py`, `lens_config.py`,
> `lenses_maslov.py` and `analysis/psf_mtf_otf.py` all moved under me, and for one window
> `import lumenairy` was broken by an in-flight edit to `analysis/psf_mtf_otf.py`
> (`AttributeError: module … has no attribute 'encircled_energy_profile'`), exactly the
> kind of transient WP-B3 §7.1 recorded for `carrier.py`. Every byte-identity and mutation
> measurement below was therefore taken **archive-to-archive**, never against the working
> tree. That change of method is itself a finding — see §4.1.

---

## 1. Verdict

| # | WP-B3 claim | Verdict | My oracle and my numbers |
|---|---|---|---|
| **K13-a** | `z_output` closes the walk; photometric scale vs band-limited ASM **0.9906** at the native pitch | **VERIFIED** | My own hand-written angular-spectrum propagator (cross-checked against my own Hankel quadrature), on a fixture WP-B3 did not use: N=40, dx=3 µm, w0=9 µm, **λ=532 nm**, **z=1.35 mm**, **cone 0.07 rad**, 4 seeds × 2 M paths. Least-squares scale **0.9903** (seeds 0.9911 / 0.9833 / 0.9921 / 0.9946, max \|s−1\| 0.0167). WP-B3 measured 0.9906 on its own geometry. |
| **K13-b** | the scale is **flat in output pixel area** (0.9906 / 0.9637 / 0.8695 at ×1 / ×2 / ×4) | **VERIFIED — and the report understates it** | I reproduce 0.9903 / 0.9636 / 0.8689 against a point-sampled reference. Against a **correctly centred bin-averaged** reference (a ×8 fine ASM, rolled by half a coarse pixel and block-averaged over each output pixel's own footprint) the same walk reads **0.9974 / 0.9981 / 0.9962** — flat to 0.4 % over the 4× range. The ×4 roll-off is entirely the reference, exactly as §2.1 said; the Jacobian itself does not move. |
| **K13-c** | flat in **path count** (0.9841 / 0.9906 / 0.9899 at 0.5 / 2 / 8 M) | **VERIFIED** | 3 seeds at 0.25 M / 1 M / 4 M / **16 M**: **0.9887 / 0.9924 / 0.9902 / 0.9914** — 0.4 % over a **64×** range (WP-B3 covered 16×). |
| **K13-d** | fail-before: the same call without `z_output` scores **1.16e-7** | **VERIFIED** | **1.5081e-07** on my fixture, and the "NOT photometric" warning fires. Seven decades. |
| **K13-e** | an **open stop is transparent**: two-leg / one-leg = 1.0524 at 0.05 rad | **VERIFIED-WITH-NOTES** | 4 seeds × 2 M on my fixture: ratio **0.9374** at 0.05 rad (two-leg 0.9293 ± 0.0682, one-leg 0.9914 ± 0.0019), **0.8911** at 0.07 rad, **3.0449** at 0.12 rad. The property holds inside the test's 0.25 bar at the stated cone, and the variance blow-up with cone is reproduced. **Note:** §8's "converges slowly **and from above**" is fixture-specific — mine converges from *below* at 0.05 and 0.07 rad. The sign is not a property of the estimator. |
| **K13-f** | `'auto'` refuses `'physical'` through a powered element; **4879×** at a thin singlet's image plane | **VERIFIED-WITH-NOTES** | Oracle-free (power ratio through a lossless walk), 500 k paths, WP-B3's own singlet: forced-`'physical'` **P_out/P_in = 4595** at 28.70 mm and **35.3** at 14.35 mm, both warning. My **flat free-space control at the same path count reads 2.47 and 7.58** — that is the Monte-Carlo power-ratio noise floor, because \|Σ\|² carries the estimator variance (the reason WP-B3's own tests use an unbiased least-squares projection). So the image-plane row is ~600× above its noise floor and solid; the **14.35 mm "13.886×" row is only a handful of times above it** and should not be quoted as a measurement of the Jacobian error. |
| **K13-g** | the `'auto'` resolution rule | **VERIFIED-WITH-NOTES → defect V1, fixed by me** | Matrix re-derived: flat air / no `z_output` → legacy; flat air + `z_output` → physical; **index step** (air \| N-BK7) → legacy; index-**matched** immersion (N-BK7 \| N-BK7) → physical; singlet → legacy. All correct. **But a third condition was missing** and `'auto'` walked into a hard `ValueError` — §2, V1. |
| **K13-h** | the closing hop reads `glass_after`, and a mirror folds it | **VERIFIED** | Mutating the index lookup to air turns the structural pin red. Mirror: `z_output` behind → 256/256 pixels non-zero; in front → 0/256 **and the default `on_undersampled='warn'` does warn** ("only 0 landed on the 16x16 grid (0.00 %)"). It is silent only when the caller passes `on_undersampled='silent'`, which WP-B3's own test does. See F3. |
| **K13-i** | `propagate(method='hfpi', …)` already reaches the new keywords (§5.4) | **VERIFIED — more strongly than claimed** | No keyword allow-list in `propagate()`; the dispatcher result is **byte-identical** to the direct call; both `z_output` and `sampler` change the answer (so they are consumed, not swallowed); a bad `sampler` surfaces `propagate_hfpi_through_prescription`'s own message. |
| **K22-a** | both samplers are Monte-Carlo, **p ≈ 0.537 / 0.557** | **VERIFIED** | My reference, my geometry (N=40, dx=3 µm, λ=532 nm, z=1.35 mm, cone 0.07), my seeds (101…606), 7 path counts 2¹⁴…2²⁰: **p(err_total) 0.519 (jit) / 0.518 (sob)**, **p(err_noise) 0.515 / 0.528**. 16× error ratios **4.229 / 4.401 / 4.180** (jit) and **4.240 / 4.282 / 4.307** (sob) against 4.0 for `O(N^-1/2)` and 16.0 for `O(N^-1)`. Same conclusion, no rate gain. |
| **K22-b** | the constant is **1.00–1.13×** in Sobol's favour | **VERIFIED-WITH-NOTES** | On my fixture the noise ratio is **1.002 / 1.057 / 1.002 / 1.006 / 1.038 / 1.028 / 1.032** — inside WP-B3's range but only at its bottom. The 1.13× top of the advertised band is fixture-specific; "≈1.0–1.06× on a second fixture" is the honest read. |
| **K22-c** | `n_paths` honoured **exactly**; jittered rounds | **VERIFIED** | 16 384 requested → **14 641** (jittered) / **16 384** (sobol), reproduced. |
| **K22-d** | Owen scrambling is on and the bundle is a **pure function of `rng`** | **VERIFIED** | Attacked five ways: identical for the same int seed **after the global NumPy state is disturbed**; identical after an interleaved jittered draw; identical for two fresh `default_rng(7)` objects; different for a different seed; a reused `Generator` advances; `rng=None` differs run to run. Scramble is genuinely on: 60 % of position entries differ between seeds 7 and 8. Mutating `scramble=False` **or** pinning the scramble seed to a constant both turn `test_the_sobol_bundle_is_a_pure_function_of_rng` red. |
| **K22-e** | the low-discrepancy property is real even though the estimator gains nothing | **VERIFIED (new evidence)** | At 4096 points the Sobol `(cos θ, φ)` projection on a 16×16 grid is **perfectly balanced — std 0.000, every cell exactly 16** — against jittered std 3.52. The sequence is doing its job; the integrand's hard edges are what eat the gain. That is the cleanest statement of K22's finding and it is not in the report. |
| **K22-f** | a non-power-of-two `n_paths` **warns** rather than rounding | **VERIFIED — and warning-only is right** | 1000 → 1000 + warning; 4095 → 4095 + warning; 1024 and **1** → silent (1 = 2⁰). Rounding would contradict the documented "`n_paths` honoured exactly", which is the keyword's whole selling point. |
| **K9-a** | `_rs_pixel_integrated_kernel` is the exact pixel integral; 6 nodes put the floor at **4.7e-11** at the worst legal geometry | **VERIFIED** | Oracle: **`scipy.integrate.dblquad`** (adaptive Gauss–Kronrod + QUADPACK subdivision) per pixel — a different quadrature family from the module's fixed Gauss–Legendre *and* from WP-B3's super-sampled midpoint. Worst relative error over probe pixels at `z = z_alias`: **3.20e-10** (N=64, 2 µm), **1.50e-10** (128, 1 µm), **3.49e-10** (128, 2 µm), **2.69e-10** (96, 1.5 µm — a grid not in the report), **8.67e-12** at 1.95 `z_alias`. The adaptive rule's own reported error is 5e-9…3e-7 relative at the rim, so the 6-node build is **at or below the reference's floor**. The point-sampled build on the same grid is off by up to **129 %** at the rim. |
| **K9-b** | on a **staircase** aperture the integrated kernel is 1032× closer and *below the oracle's own floor* | **VERIFIED — the report's number is its oracle's floor, as it said** | My oracle: the exact continuum RS-I integral of the cell-constant field as a **direct spatial sum** over lit pixels with per-pixel integrals from a **Clenshaw–Curtis-32** rule (validated against QUADPACK to 8e-14…1.2e-12), no FFT, no padding, no library call. Fixtures WP-B3 did not use — N=48/dx=6 µm/a=78 µm/z=9.5 mm and N=64/5 µm/90 µm/12 mm: `'spatial-integrated'` **3.67e-13 / 4.64e-13**, `'spatial'` **4.96e-3 / 2.50e-3**. The integrated kernel is exact to round-off; "1032×" was the midpoint oracle's own limit. |
| **K9-c** | on a **sampled smooth** field the ranking reverses by 4–5 decades, so `'auto'` keeps `'spatial'` | **VERIFIED — by 9–10 decades on an exact oracle** | Oracle: the exact continuum RS-I integral of the **continuous Gaussian** by 2-D adaptive quadrature (non-oscillatory at these numbers, so the adaptive rule is exact to round-off). N=64/1.5 µm/w0 7 µm/z 2.2 mm and N=96/0.75 µm/5 µm/0.6 mm: `'spatial'` **1.83e-13 / 3.19e-14**, `'spatial-integrated'` **8.40e-4 / 7.51e-4**. `'auto'` keeping the point kernel is right, and `auto.tobytes() == spatial.tobytes()` re-confirmed. |
| **K9-d** | §3.2 — the convergence lever is the **aperture's edge**, not the kernel; 25× at N=1024, kernel ≈2× in the point kernel's favour | **VERIFIED** | Oracle: the exact on-axis closed form `U = e^{ikz} − (z/r_a) e^{ik r_a}`, written here, on a fixture WP-B3 did not use (a = 80 µm, window 640 µm, **z = 20 mm**, **λ = 532 nm**, F = 0.60). Orders 128→256→512→1024: staircase **1.860 / 1.095 / 0.623** (point) and **1.858 / 1.105 / 0.626** (integrated) — non-monotone and identical between kernels to three digits; grey **2.039 / 2.035 / 2.023** (point) and **2.020 / 2.017 / 2.013** (integrated). Grey/staircase at N=1024: **37.6×** (point), 19.2× (integrated); the kernel moves the grey constant **1.96×** in the point kernel's favour. Every claim in §3.2 reproduces. |
| **K9-e** | the `'RS_INT'` cache tag cannot be handed `'spatial'`'s array | **VERIFIED** | Called in **both** orders at the same geometry: the two arrays differ, and a repeat of the first token returns its own array byte-identically. Mutating the tag to share `'RS'` turns **3** tests red. |
| **K9-f** | one alias guard covers both spatial tokens | **VERIFIED** | At **0.999 `z_alias`** both `'spatial'` and `'spatial-integrated'` raise, each message naming the token actually passed; at 1.000 and 1.001 both run; `'auto'` runs everywhere. |
| **K9-g** | the build is `xp`-generic and survives complex64 / anamorphic grids (§8.3 desk-check) | **VERIFIED on NumPy, extended** | complex64: dtype preserved, relL2 **1.48e-7** against complex128. Anamorphic **48×64 at dy/dx = 2** and odd **49×65 at dy/dx = 1.5**: the kernel matches my adaptive pixel integral to **1.05e-12 / 1.01e-12** and the propagation returns the right shape and finite values. WP-B3's node-count table covers only `dy = dx`; the quadrature holds off-square too. |
| **K6-a** | the chirp-Z leg is the trigonometric interpolant | **VERIFIED** | My own Dirichlet-kernel double sum, written for **non-square** grids: agreement **7.1e-15 … 2.8e-14** on seven fixtures — odd 65×65, odd non-square 45×63 and 63×45, odd no-op 31×49, even non-square 32×48, odd×even 33×20, and `dx_out < dx_in` on odd grids (the brief's edge; no spurious warning, window/period 0.9969–1.0000). |
| **K6-b** | unit MTF (0.7177 → 1.0000 at 0.40 cyc/px) | **VERIFIED-WITH-NOTES** | My own fixture (N=96, w0 = 24 µm, a **diagonal** carrier), at ×0.5: chirpz **1.000000** at 0.00 / 0.05 / 0.15 / 0.25 / 0.35 / 0.45 cyc/px against spline 0.999987 / 0.999958 / 0.997404 / 0.972442 / 0.849642 / **0.570883**. **Note:** at ×1.7 chirpz reads 0.999956–0.999984, not 1.000000, because the extent-preserving `N_out` rounds the window 0.8 % off the reconstruction period; WP-B3's ×1.5 column lands exactly on the period. The unit MTF is a property of the window being one period, not of the method alone. |
| **K6-c** | a window wider than the period returns replicas; **power ratio exactly 4.000000** for a 2× window | **VERIFIED — and it is m×m, not just 2×2** | 2× window **4.000000**, **3× window 9.000000**, spline 1.000000 in both, chirp-Z warns in both. |
| **K6-d** | `dx_out == dx_in` returns the input to 9.5e-15 | **VERIFIED** | Reproduced through the interpolant check (the odd no-op 31×49 case agrees with the Dirichlet sum to 1.36e-14). |
| **K6-e** | the **odd-`N` half-pixel origin** is folded into the output centre | **VERIFIED in the code, NOT PINNED — defect V3, fixed by me** | Correct as shipped (1e-14 class against my oracle), but **dropping it leaves all 33 WP-B3 tests green** while costing relL2 **1.077–1.098**; so does replacing the `N_in//2` bin centre with `N_in/2` (**1.038–1.094**). Both are exactly 0 on even grids, and every K6 test used an even `N_in`. §2, V3. |
| **byte identity** | 52 comparisons, all identical | **VERIFIED — on 60 of my own, by a cleaner method** | §4. |
| **§5 call-site edits** | direct MFT on the Fresnel leg; chirp-Z **only when the pitch coarsens** | **VERIFIED, with one wording correction** | §5. |
| **test quality** | 33 tests, derived envelopes, no wall-clock assertions | **VERIFIED** | §3. |

**Totals.** 18 VERIFIED, 7 VERIFIED-WITH-NOTES, **0 NOT FIXED**, **0 REGRESSION against
the parent**. Three defects found **inside the new code paths** and fixed by me (§2); none
of them reachable from a pre-`z_output` call, which is why the byte-identity set is clean.

---

## 2. Defects found, and what I did about them

### V1 — `normalisation='auto'` could raise instead of falling back (P1, **fixed**)

`'auto'` asked two questions (is there an output plane? are the legs free space?) where
the estimator needs **three**. `_reemission_measure` scales every re-emitted path by
`r_in`, the length of the leg that reached the surface. A diffracting surface sitting on
the plane the paths were last emitted from makes that zero for every path, and the measure
**refuses**. So `'auto'` resolved to `'physical'` and walked straight into that refusal.

**Fail-before** (WP-B3 `284daccc`, both fixtures):

```
ValueError: apply_aperture_diffraction: normalisation='physical' needs the geometric
length of the leg that ended at this surface … apply it to the field before calling
init_paths_from_field, propagate to a non-zero distance first, or pass
normalisation='legacy' …
```

raised from a private helper, naming a function the caller never called and offering a
remedy (`init_paths_from_field`) that has nothing to do with a prescription walk. Two
ordinary prescriptions reach it:

| fixture | before `z_output` existed | WP-B3 `284daccc` | after this fix |
|---|---|---|---|
| a flat apertured surface with `object_distance = 0` (which is what `_plane_prescription(semi_diameter=…)` builds by default) | works, legacy + warning | **ValueError** | works, legacy + warning naming surface 0 |
| `object_distance = 1 mm`, **two coincident stops** (`thicknesses = [0.0, 0.0]`) — a stop placed at a surface | works, legacy + warning | **ValueError** | works, legacy + warning naming surface 1 |

Nothing that worked before could hit this (the raise needs `z_output`, which is new), so
it is a defect in the new feature rather than a regression — but it turns the *default*
normalisation into a hard error for a layout as common as "the stop is at the first
surface".

**Fix** (`lumenairy/propagators/hfpi.py`): a new
`_walk_zero_length_reemission(surfaces, diffracting_surfaces, object_distance)` reads the
axial gap in front of every re-emitting surface **from the prescription, before the walk
starts**, and returns the first offender or `None`. `'auto'` now needs all three answers;
the diffractor list is resolved above the estimator decision so both can use it. Forcing
`normalisation='physical'` there still raises — there is no factor to apply — but now at
the walk's own altitude, with the `CONVENTIONS.md` §2 prefix, the offending surface index
and four real remedies.

**After:** `'auto'` falls back, and the field it returns is **byte-identical** to the same
call with `normalisation='legacy'` spelled out — so the fallback is the legacy estimator
itself, not a third thing. Pinned by
`TestK13PrescriptionWalkOutputPlane::test_a_zero_length_re_emission_does_not_get_a_photometric_default`
(2 cases).

### V2 — the legacy warning stated a cause that was often false (P2, **fixed**)

Every legacy walk got the same sentence:

> This walk bins the bundle at the last surface rather than propagating it to a separate
> output plane … the last leg has zero length. … **Pass `z_output=<the plane you want the
> field on>`**

A caller who passed `z_output` through a singlet got told to pass `z_output`; a caller
behind a mirror got told the walk had not propagated to a separate plane when it had.
Three independent conditions send the walk down this branch and a caller can hit more than
one at once.

**Fail-before:** on `284daccc`, `assert 'no z_output was given' in <the singlet walk's
warning>` fails — the text contains the opposite claim.

**Fix:** the warning is assembled from the answers the estimator was actually resolved on
and names the condition(s) that failed, keeping the `NOT photometric` phrase every
existing matcher uses. Explicit `normalisation='legacy'` now says so rather than borrowing
the missing-plane story. Pinned by
`TestK13PrescriptionWalkOutputPlane::test_the_legacy_warning_names_the_condition_that_failed`.

### V3 — three chirp-Z mechanisms shipped correct and unpinned (P2, **fixed by adding tests**)

Mutation testing found three edits to `mft.py` that leave **all 33 WP-B3 tests green**:

| mutation | cost, measured against my Dirichlet oracle | tests red on `284daccc` |
|---|---|---|
| drop the odd-`N` half-pixel `off_in` shift | relL2 **1.077 / 1.093 / 1.098** on odd grids, **0** on even | **0** |
| use `N_in/2` instead of `N_in//2` for the input bin centre | relL2 **1.038 / 1.079 / 1.094** on odd grids, **0** on even | **0** |
| ignore `_warn_mft_output_window(N_out_y=…)` | a 45×63 input at dx 1→0.7 µm warns about an axis that fits | **0** |

Every K6 test used an even, mostly square `N_in`, so the odd-`N` handling the report
calls out by name ("folding the odd-`N` half-pixel origin offset into the output centre
exactly as `angular_spectrum_propagate_mft` does") and the per-axis parameter the report
says was added for `resample_field`'s non-square default were both invisible.

**Fix:** two new tests, six cases, no library change —
`TestK6ChirpZResampler::test_the_chirpz_leg_places_an_odd_grids_origin` (65×65, 45×63,
33×21, 17×17, against the Dirichlet double sum written for non-square grids, bar 1e-12,
measured 6.0e-15…1.8e-14) and
`TestK6ChirpZResampler::test_the_faithful_zone_warning_sizes_each_axis_separately`
(44×63 must warn **on y alone**, 45×63 must be **silent**, both by exact products). All
three mutations now go red.

### Not a defect, recorded: the mirror's wrong side

`z_output` on the non-reflected side of a mirror returns an all-zero field. With the
**default** `on_undersampled='warn'` the existing sampling-adequacy guard does fire
("of 20000 paths only 0 landed on the 16x16 grid (0.00 %)"), so the user is not left
silent; it is silent only under `on_undersampled='silent'`, which WP-B3's own test passes.
The guard blames sampling rather than the fold, which is a diagnostic-quality item, not a
correctness one — F3 below.

---

## 3. The pins: do they bite, and are they envelopes?

**Every numeric bar in the file is a derived envelope, a round-off bar, or an exact
identity. There is no per-build number anywhere in it, and nothing reads a clock** —
the convergence claims are stated as error ratios at fixed path/grid factors, as S5
requires. The bars are: `abs(scale−1) < 0.10 / 0.12 / 0.25` (6.7× / 2.7× / 4.8× above the
measured seed envelope), `3.2 < ratio < 8.0` (my own measurement lands at 4.18–4.40),
`1.75 < order < 2.25` (mine: 2.012–2.039), `> 100×` separations, `< 1e-9 / 1e-12 / 1e-14`
round-off bars whose measured residuals sit 2–5 decades below, and `tobytes()` identities.
The two one-sided structural bounds (`_RS_PIXEL_QUAD_NODES >= 4`, `floor < 1e-4`) are
sanity guards, not pins on 6 or on a captured value.

Mutating the fix and watching the file (all runs on the isolated `284daccc` archive):

| mutation | tests red | strongest signal |
|---|---|---|
| `_binning_jacobian` drops the `r` | 1 | scale **494.5** against a 0.1 bar (3.7 decades) |
| the closing hop is a no-op | 1 | hard `ValueError` from `accumulate_to_grid` |
| closing hop always `n_medium = 1` | 1 | the structural pin (`glass_after` lookup) |
| `'auto'` ignores the free-space condition | 1 | the powered-prescription test |
| Sobol scramble off | 1 | pure-function-of-`rng` |
| Sobol scramble seed constant | 1 | pure-function-of-`rng` |
| `'spatial-integrated'` samples at the pixel centre | **5** | cell-constant, smooth ranking, node count, fold, cache |
| `'RS_INT'` shares the `'RS'` cache tag | 3 | cache slot + both physics tests |
| the quadrant gather slips one pixel | 2 | cell-constant + folded-vs-unfolded |
| chirp-Z `alpha_x`/`alpha_y` swapped | 1 | the non-square test |
| **chirp-Z odd-`N` origin dropped** | **0 → now 4** | V3 |
| **chirp-Z bin centre `N/2`** | **0 → now 4** | V3 |
| **`_warn_mft_output_window` ignores `N_out_y`** | **0 → now 1** | V3 |

---

## 4. Byte identity, attacked

### 4.1 Method — and why WP-B3's own harness could not have been clean

WP-B3 §6 loaded the HEAD module under `lumenairy.propagators._baseline_<name>` *inside the
live package* and re-ran "on the final tree after every edit in this work package had
landed". By then that tree carried three other engineers' uncommitted work. I re-ran my
probe set against the **dirty working tree** and got 45/47 — with
`walk_singlet_strat` and `walk_singlet_uniform` differing by relL2 **6.5e-13** and
**2.6e-12**. Those two differences are **not WP-B3's**: they come from concurrent
uncommitted edits to `lumenairy/raytrace/{trace,intersection,surface}.py`, which the
prescription walk traces through. Against clean archives the same two probes are
byte-identical.

So the correct method — and the one every number below uses — is **archive-to-archive**:

```
git archive 81d5b586 lumenairy | tar -x -C <scratch>/verify_b3
git archive 284daccc lumenairy | tar -x -C <scratch>/verify_b3_head
# each probe run from inside its own tree, cwd == PYTHONPATH == that tree,
# with lumenairy.__file__ asserted in the output before any measurement
```

This is a note on WP-B3's *evidence*, not on its result: its conclusion is right, but the
harness as described cannot separate the work package from its neighbours, and on this
tree it demonstrably did not.

### 4.2 Result

**60 arrays byte-identical out of 60**, in two independently constructed probe sets
(47 + 13), against **both** `81d5b586` and `284daccc`, and again after my own fix:

| surface | cases |
|---|---|
| `rayleigh_sommerfeld_propagate` | `'auto'` at (64, 2 µm, 2 mm), (128, 1 µm, 3 mm), (128, 1 µm, 50 µm — the transfer branch), odd 65; `'spatial'`; `'transfer'`; `bandlimit=True`; `bandlimit + 'transfer'`; complex64; complex64 + `'spatial'`; anamorphic 64×48 at `dy = 2 dx`; **odd anamorphic 65×49 at `dy = 1.5 dx`** |
| `resample_field` (default leg) | nine `(N, dx_in, dx_out, N_out, order)` combinations including odd 65, the exact no-op, `order` 0/1/3/5 and a **non-square 24×18** input; returned `dx_out` too |
| `angular_spectrum_propagate_mft`, `fresnel_propagate_mft`, `fraunhofer_propagate_mft` | natural and warning-triggering windows |
| `_warn_mft_output_window` | warning text, two fixtures (square callers) |
| `init_paths_stratified` | jittered default, all six `PathBundle` fields, at 8×8/1024, 16×16/4096, the K23 cap case and non-square 12×9/5000 — **24 array comparisons** |
| `propagate_hfpi_through_prescription` | the singlet walk at `'stratified'` and `'uniform'`; the flat walk; the flat walk with a stop |
| `propagate_hfpi_freespace_aperture`, `propagate_hfpi` | untouched entry points |
| `propagate_huygens_fresnel_freespace` | bare, resample leg (both returns), anamorphic |

**The probe the brief singled out.** `propagate_hfpi_through_prescription` with the old
default spelled out (`normalisation='legacy'`) against the new default (`'auto'` → legacy
with no `z_output`): **identical bits and identical warning text, in both trees** — and
forced `'physical'` without `z_output` raises the **identical message** in both.

**Text records: 108 of 113 identical.** The five that differ are all one thing — the
walk's legacy warning, which WP-B3 rewrote and which I rewrote again (V2). WP-B3's §6
lists "the flat walk at explicit `normalisation='legacy'`" among its byte-identical
surfaces; the array is identical, the **warning text is not**, and the report's
"2 warning-text comparisons" were `_warn_mft_output_window`'s. Worth one line in the
migration notes: a caller filtering on `"since v5.46"` or
`"Pass normalisation='physical' if your surface list ends"` no longer matches.

---

## 5. §5's call-site edits, measured on my fixtures

**§5.1's table reproduces exactly** (same fixture definition, deterministic):
N=256/z=5 mm top-hat spline **4.3589e-2** / chirpz **4.9204e-2**, window power
**0.986005 / 0.986947** against the direct evaluation's **0.986945**; N=256/z=2 mm
**4.8591e-2 / 5.3308e-2**, **0.995421 / 0.996056 / 0.996072**; N=512/z=5 mm
**2.8108e-2 / 3.1474e-2**, **0.996685 / 0.996993 / 0.996992**. Every digit.

On a **contained** field my own Gaussian (w0 = 0.06 N dx) gives chirpz **2.50e-14** against
spline **4.51e-5** at `dx_new/dx = 3.09` — the direction of §5.1 reading 1 confirmed; the
constant (WP-B3 says 145×) is fixture-dependent and can be far larger.

**The gating rule holds**, with one wording correction:

* The chirp-Z leg returns replicas exactly when `N_out*dx_out > N_in*dx_in`, and it warns
  there. At `dx_new/dx = 0.6182` I measure chirpz `P/P_in = 2.684` with the warning, spline
  0.927, relL2 1.244 / 0.894 — both unusable, and the gate is right to send that case to
  the spline.
* **But `dx_new >= current_dx` is only the right gate because `N_out == N_in` at those two
  call sites.** The general condition is `N_out*dx_out <= N_in*dx_in`. I recommend the
  requested edits in §5.1(b) and §5.2 be spelled
  `method=('chirpz' if N_out * current_dx <= E.shape[-1] * dx_new else 'spline')`, or
  keep `dx_new >= current_dx` with a comment saying it is the `N_out == N_in` special case.
  As written it is correct today and silently wrong if anyone changes `N_out`.
* One honest caveat for whoever applies it: the gate is **conservative**. A *contained*
  field in the converging direction is fine under chirp-Z (my Gaussian row at
  `dx_new/dx = 0.6182` gives `P/P_in = 1.000181`, matching the direct evaluation's
  1.000181) but the period test warns and the gate routes it to the spline, costing it the
  unit MTF. That is the right default for an automatic gate; it is not free.

§5.3 (no pixel integral for `hf.py`'s OPL quadrature) and §5.4 (nothing requested of
`dispatch.py`) both hold — see K13-i.

---

## 6. Follow-up (ruled open, not fixed here)

| # | Severity | Item |
|---|---|---|
| **F1** | P2 — **outside my ownership** (`.test_durations`) | My eight new tests carry no timing. `test_audit2609_a15a_durations_staleness` is already red at **769 of 15809 ids (4.86 %)** against a 2 % bar, dominated by other work packages (`b8` 256, the dispatcher pin 233, `b4` 84, `b9` 62, `b6` 44, `b5` 38, `b1` 25); mine add 8. The exact lines to add are in §7.1. |
| **F2** | P3 | WP-B3 §2.1's **13.886× at 14.35 mm** is not separated from the Monte-Carlo power-ratio noise floor (my flat control reads 2.47–7.58 at 500 k paths). The image-plane 4879× is solid. Either re-measure the half-distance row with an unbiased least-squares scale or drop it; the argument does not need it. |
| **F3** | P3 | `z_output` on the wrong side of a folded stack returns all zeros; the existing guard warns about *sampling*. A walk-level diagnostic naming the fold would be better, but it needs the direction state the walk does not currently keep — a design change to `propagate_to_plane`'s kill mask, not a patch. |
| **F4** | P3 | §8's "the cascaded estimator converges **from above**" is fixture-specific: mine converges from below at 0.05 and 0.07 rad and from above at 0.12. The variance observation stands; the sign should be dropped. |
| **F5** | P3 | K22's advertised **1.00–1.13×** constant reads 1.00–1.06× on a second fixture. Suggest "≈1.0–1.1×, fixture-dependent" in the changelog. |
| **F6** | P3 | `resample_field`'s unit MTF is a property of the output window being exactly one reconstruction period. Whenever the extent-preserving `N_out` rounds off the period the MTF reads 0.99996 rather than 1.000000 (measured at ×1.7). The docstring's table should say at which scale factors it is exact. |
| **F7** | P4 | `'spatial-integrated'` on CuPy/JAX remains desk-checked only (WP-B3 §8.3). I extended the NumPy evidence to complex64 and to anamorphic and odd-anamorphic grids; the backend question is unchanged. |

---

## 7. Everything I ran

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_b3_propagator_kernels.py` (WP-B3 as committed, baseline) | **33 passed** | 56.76 s |
| `pytest tests_b3/…b3_propagator_kernels.py` **inside the `284daccc` archive** (isolation check) | **33 passed** | 62.93 s |
| the same, inside the archive **with my `hfpi.py`** | **33 passed** | 59.62 s |
| `pytest tests/unit/test_audit2609_b3_propagator_kernels.py` (final tree, +8 tests) | **41 passed** | 62.06 s, re-run 53.68 s |
| `pytest tests/unit -k "a5 or hfpi or hf_ or rs_ or resample or mft or bluestein"` | **819 passed, 12 skipped, 2 failed** — both other packages' (§7.2); the skips are PySide6 / CuPy / JAX-x64 / the W5 host-digest set, all pre-existing | 304.42 s |
| `pytest tests/unit/test_niche_d2_chain_multi.py tests/unit/test_audit2609_a25_carrier_focus_readout.py tests/unit/test_audit2609_a17_history_lint.py tests/unit/test_audit2609_a17_history_relocation.py` | 778 passed, **21 failed — every one another package's module**, and the count fell to **2** (both `lumenairy.elements._lens_traced`) an hour later as its owner re-recorded | 589.56 s |
| `pytest tests/unit/test_audit2609_a17_history_relocation.py tests/unit/test_audit2609_a17_history_lint.py` (re-check) | **742 passed, 2 failed** (`_lens_traced`, not mine) | 55.77 s |
| `python validation/run_all.py test_propagation test_hfpi test_hf test_dispatch test_advanced_diffraction --quiet` | **ALL 5 passed** (7.0 / 3.6 / 2.5 / 1.6 / 2.1 s) | 17.0 s |
| `python -m ruff check lumenairy/ tests/unit/test_audit2609_b3_propagator_kernels.py` | **All checks passed** | — |
| `python scripts/record_history_fingerprints.py --check` | my four documents **OK**; the one DRIFT in the tree is `lumenairy/elements/_lens_traced.py`, not mine | — |
| 13 mutation runs of the b3 file (§3) | as tabulated | ~60 s each |
| byte-identity probe sets 1 and 2, six runs (parent / HEAD / fixed × 2 sets) | **60/60 identical**, 108/113 texts | ~40 s each |
| the independent oracles (§1): adaptive QUADPACK kernel probe, Clenshaw–Curtis cell-constant sum, continuum-Gaussian quadrature, Dirichlet interpolant, on-axis closed form, hand-written ASM + Hankel, K13 sweep, K22 sweep, edge sweep | — | 2.1 / 5.6 / 1.2 / 1.2 / 29.2 / 134.5 / 21.4 / 19.6 s |

No test asserts a duration, and every duration here is indicative only — three engineers
shared this box throughout.

### 7.1 The `.test_durations` lines my tests need (F1, outside my ownership)

```json
    "tests/unit/test_audit2609_b3_propagator_kernels.py::TestK13PrescriptionWalkOutputPlane::test_a_zero_length_re_emission_does_not_get_a_photometric_default[coincident_second_stop]": 0.03,
    "tests/unit/test_audit2609_b3_propagator_kernels.py::TestK13PrescriptionWalkOutputPlane::test_a_zero_length_re_emission_does_not_get_a_photometric_default[stop_on_the_source_plane]": 0.73,
    "tests/unit/test_audit2609_b3_propagator_kernels.py::TestK13PrescriptionWalkOutputPlane::test_the_legacy_warning_names_the_condition_that_failed": 0.48,
    "tests/unit/test_audit2609_b3_propagator_kernels.py::TestK6ChirpZResampler::test_the_chirpz_leg_places_an_odd_grids_origin[17-17-0.55]": 0.01,
    "tests/unit/test_audit2609_b3_propagator_kernels.py::TestK6ChirpZResampler::test_the_chirpz_leg_places_an_odd_grids_origin[33-21-1.4]": 0.01,
    "tests/unit/test_audit2609_b3_propagator_kernels.py::TestK6ChirpZResampler::test_the_chirpz_leg_places_an_odd_grids_origin[45-63-0.7]": 0.01,
    "tests/unit/test_audit2609_b3_propagator_kernels.py::TestK6ChirpZResampler::test_the_chirpz_leg_places_an_odd_grids_origin[65-65-0.5]": 0.01,
    "tests/unit/test_audit2609_b3_propagator_kernels.py::TestK6ChirpZResampler::test_the_faithful_zone_warning_sizes_each_axis_separately": 0.01,
```

### 7.2 Every failure is another work package's, named

* `test_audit2609_a15a_durations_staleness::test_durations_covers_at_least_98_percent_of_collected_ids`
  — F1; 769 missing ids, my eight are 1 % of them and none of the eight worst files is mine.
* `test_niche_c12_physics_fit_selection::test_the_two_selectors_agree_on_the_slow_fixture`
  — `TypeError: s_spy() got an unexpected keyword argument 'basis'` in
  `lumenairy/elements/_lens_traced.py`, a module I never opened.
* `test_audit2609_a17_history_relocation` ×21 → ×2 — AST/token drift on `carrier`,
  `lumenairy.elements.{_lens_traced, lenses_maslov}` and
  `lumenairy.raytrace.{differential, intersection, jax_trace, ray_fan, surface, trace, world_trace}`.
  All owners' re-records; by my last check only `_lens_traced` remained.

---

## 8. Files I touched

| file | what |
|---|---|
| `lumenairy/propagators/hfpi.py` | `_walk_zero_length_reemission` (new); the diffractor list resolved above the estimator decision; `'auto'`'s third condition; a walk-level refusal for forced `'physical'` on a zero-length re-emission; the legacy warning rebuilt from the conditions that failed; `z_output` and `normalisation` docstrings and the `normalisation` validator updated to three conditions |
| `docs/history/lumenairy.propagators.hfpi.md` | re-recorded in the same change (`--reason` states V1 and V2) |
| `tests/unit/test_audit2609_b3_propagator_kernels.py` | **+8 tests** (33 → 41); nothing existing weakened or removed |
| `docs/audits/…/fixes/VERIFY_WP-B3.md`, `VERIFY_WP-B3_CHANGELOG.md` | this report and its release text |

`rs.py`, `mft.py` and `hf.py` needed no change. No file outside this list was modified;
`system.py`, `_lens_real.py`, `carrier.py`, `dispatch.py`, `raytrace/*` and
`elements/elements.py` were read only.
